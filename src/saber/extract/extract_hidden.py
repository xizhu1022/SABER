"""Hidden-state extraction from a frozen backbone.

For each DatasetRow in ``data/datasets/{dataset}.jsonl`` we run two forward
passes per row and read the layer-L hidden state at the last prompt token:

  - CK variant   : prompt = f"{question}\\n{ck_text}"  -> model sees PK + CK
  - no-CK variant: prompt = f"{question}"              -> model sees PK alone

Both variants are extracted in one model load.

Outputs (one set per dataset, per backbone):
  artifacts/{backbone}/hidden/{dataset}.hidden.npy        (N, d_model) -- with CK
  artifacts/{backbone}/hidden/{dataset}.hidden_nock.npy   (N, d_model) -- no CK
  artifacts/{backbone}/hidden/{dataset}.meta.jsonl        per-row metadata
  artifacts/{backbone}/hidden/{dataset}.config.json       extraction config

meta.jsonl per-row fields:
  qid, dataset, ck_relation, ck_relation_is_core, raw_idx_in_jsonl,
  token_len, token_len_nock, truncated

Defaults:
  - Layer        : mid-layer (n_layers // 2); overridden via --layer-frac
  - Position     : last prompt token (pre-response state)
  - Input format : "{question}\\n{ck_text}" (CK), "{question}" (no-CK); no chat template
  - Truncation   : only applied to the CK variant

Usage::

    PYTHONPATH=src python -m saber.extract_hidden \\
        --datasets confiqa-qa redditqa \\
        --layer 16 --max-len 4096

The script is idempotent: if the hidden files + config match, the dataset is
skipped. Pass --force to rerun.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from saber.config import DATA_ROOT, HF_CACHE_DIR
from saber.models import get_model_spec
from saber.prompts import hidden_state_probe_prompt

ART_DIR = DATA_ROOT / "artifacts"
DATASET_DIR = DATA_ROOT / "datasets"

VARIANTS = ("ck", "nock")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _truncate_to_max_len(
    tokenizer,
    prompt: str,
    question: str,
    max_len: int,
) -> tuple[str, bool]:
    """Tokenize; if > max_len, truncate ck tail while preserving question.

    Only meaningful for the CK variant; the no-CK prompt is just the question
    and will essentially never exceed max_len.

    Returns (final_prompt, was_truncated).
    """
    ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if len(ids) <= max_len:
        return prompt, False

    q_ids = tokenizer(question, add_special_tokens=True).input_ids
    marker = " [truncated]"
    marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
    ck_budget = max_len - len(q_ids) - len(marker_ids) - 4  # buffer
    if ck_budget <= 32:
        truncated_ids = ids[:max_len]
        return tokenizer.decode(truncated_ids, skip_special_tokens=True), True

    ck = prompt[len(question) + 1 :]  # skip question + "\n"
    ck_ids = tokenizer(ck, add_special_tokens=False).input_ids[:ck_budget]
    ck_trunc = tokenizer.decode(ck_ids, skip_special_tokens=True).rstrip()
    return f"{question}\n{ck_trunc}{marker}", True


@torch.no_grad()
def _extract_one(
    model,
    tokenizer,
    prompt: str,
    layer: int,
) -> tuple[np.ndarray, int]:
    """Run forward, return (hidden_state@layer@last_prompt_token, token_len)."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_len = int(enc["input_ids"].shape[1])
    out = model(**enc, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return h.astype(np.float32), token_len


def _cfg_base_matches(prev: dict, layer: int, max_len: int, backbone: str) -> bool:
    """Per-variant idempotency requires same layer/max_len/backbone."""
    return (
        prev.get("layer") == layer
        and prev.get("max_len") == max_len
        and prev.get("backbone") == backbone
    )


def extract_dataset(
    model,
    tokenizer,
    dataset_name: str,
    layer: int,
    max_len: int,
    out_dir: Path,
    force: bool,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> None:
    """Extract CK + no-CK hidden states per row, idempotent per variant.

    If the CK variant was already extracted by a pre-refactor run (npy + matching
    layer/max_len/backbone) we skip the CK forward and only run the no-CK forward.

    Optional sharding: pass start_idx/end_idx to process only rows[start:end].
    Outputs are suffixed `.range{start}_{end}.npy` and merged later.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATASET_DIR / f"{dataset_name}.jsonl"
    suffix = f".range{start_idx}_{end_idx}" if start_idx is not None or end_idx is not None else ""
    npy_ck_path = out_dir / f"{dataset_name}{suffix}.hidden.npy"
    npy_nock_path = out_dir / f"{dataset_name}{suffix}.hidden_nock.npy"
    meta_path = out_dir / f"{dataset_name}{suffix}.meta.jsonl"
    cfg_path = out_dir / f"{dataset_name}{suffix}.config.json"

    prev_cfg: dict | None = None
    if cfg_path.exists():
        try:
            prev_cfg = json.loads(cfg_path.read_text())
        except Exception:
            prev_cfg = None
    base_ok = (prev_cfg is not None
               and _cfg_base_matches(prev_cfg, layer, max_len, model.config._name_or_path))

    ck_done = npy_ck_path.exists() and base_ok
    nock_done = npy_nock_path.exists() and base_ok

    if ck_done and nock_done and not force:
        print(f"[{dataset_name}] both variants present with matching config; skipping (use --force to rerun)")
        return

    if force:
        ck_done = nock_done = False

    run_ck = not ck_done
    run_nock = not nock_done
    print(f"[{dataset_name}{suffix}] loading {jsonl_path}  "
          f"(run_ck={run_ck}, run_nock={run_nock})")
    all_rows = _load_jsonl(jsonl_path)
    if start_idx is not None or end_idx is not None:
        s = start_idx or 0
        e = end_idx if end_idx is not None else len(all_rows)
        rows = all_rows[s:e]
        print(f"[{dataset_name}{suffix}] sharded: rows[{s}:{e}] = {len(rows)} of {len(all_rows)}")
    else:
        rows = all_rows
    n = len(rows)
    d_model = model.config.hidden_size

    hs_ck = None
    hs_nock = None
    if run_ck:
        hs_ck = np.lib.format.open_memmap(
            npy_ck_path, mode="w+", dtype=np.float32, shape=(n, d_model),
        )
    if run_nock:
        hs_nock = np.lib.format.open_memmap(
            npy_nock_path, mode="w+", dtype=np.float32, shape=(n, d_model),
        )

    # Preserve prior meta (e.g. token_len from the CK run) if we're only backfilling nock.
    prev_meta: list[dict] = []
    if not run_ck and meta_path.exists():
        with meta_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    prev_meta.append(json.loads(line))
        if len(prev_meta) != n:
            prev_meta = []  # mismatched — fall back to rewriting everything

    meta_rows: list[dict] = []
    t0 = time.time()
    trunc_n = int(prev_cfg.get("n_truncated", 0)) if (not run_ck and prev_cfg) else 0
    for i, row in enumerate(tqdm(rows, desc=f"extract {dataset_name}")):
        q = str(row["question"])
        truncated = False
        tl_ck: int | None = None
        tl_nock: int | None = None

        if run_ck:
            prompt_ck = hidden_state_probe_prompt(row, include_ck=True)
            prompt_ck, truncated = _truncate_to_max_len(tokenizer, prompt_ck, q, max_len)
            if truncated:
                trunc_n += 1
            h_ck, tl_ck = _extract_one(model, tokenizer, prompt_ck, layer)
            hs_ck[i] = h_ck
        elif prev_meta:
            pm = prev_meta[i]
            tl_ck = int(pm.get("token_len", 0))
            truncated = bool(pm.get("truncated", False))

        if run_nock:
            prompt_nock = hidden_state_probe_prompt(row, include_ck=False)
            h_nock, tl_nock = _extract_one(model, tokenizer, prompt_nock, layer)
            hs_nock[i] = h_nock
        elif prev_meta:
            tl_nock = int(prev_meta[i].get("token_len_nock", 0)) or None

        meta_rows.append({
            "qid": row["qid"],
            "dataset": row["dataset"],
            "ck_relation": row["ck_relation"],
            "ck_relation_is_core": row["ck_relation_is_core"],
            "raw_idx_in_jsonl": i,
            "token_len": tl_ck,
            "token_len_nock": tl_nock,
            "truncated": truncated,
        })
    if hs_ck is not None:
        hs_ck.flush(); del hs_ck
    if hs_nock is not None:
        hs_nock.flush(); del hs_nock

    with meta_path.open("w") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    variants_done: list[str] = []
    if run_ck or npy_ck_path.exists():
        variants_done.append("ck")
    if run_nock or npy_nock_path.exists():
        variants_done.append("nock")
    cfg = {
        "dataset": dataset_name,
        "n_rows": n,
        "layer": layer,
        "position": "last_prompt",
        "max_len": max_len,
        "backbone": model.config._name_or_path,
        "d_model": d_model,
        "dtype": "float32",
        "variants": variants_done,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 2),
        "n_truncated": trunc_n,
        "truncation_rate": round(trunc_n / max(n, 1), 4),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[{dataset_name}] done in {elapsed:.0f}s  "
          f"(n={n}, ran ck={run_ck} nock={run_nock}, truncated={trunc_n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Dataset names matching data/datasets/{name}.jsonl")
    ap.add_argument("--layer", type=int, default=None,
                    help="Transformer layer index. Default: mid-layer (n_layers // 2).")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--backbone", default="llama-3.1-8b-instruct")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: data/artifacts/exp_1/{backbone}/")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--start-idx", type=int, default=None,
                    help="Sharding: process rows[start:end] only. Output suffixed.")
    ap.add_argument("--end-idx", type=int, default=None)
    args = ap.parse_args()

    print(f"[extract] loading backbone: {args.backbone}")
    spec = get_model_spec(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, cache_dir=str(HF_CACHE_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, spec.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo, cache_dir=str(HF_CACHE_DIR),
        torch_dtype=dtype, device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else n_layers // 2
    out_dir = args.out_dir if args.out_dir is not None else ART_DIR / args.backbone

    print(f"[extract] model loaded; hidden_size={model.config.hidden_size}, "
          f"n_hidden_layers={n_layers}  (extracting layer {layer} / {n_layers})")
    print(f"[extract] output dir: {out_dir}")
    print(f"[extract] variants: {list(VARIANTS)}  (CK-inclusive + query-only)")

    for ds in args.datasets:
        extract_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset_name=ds,
            layer=layer,
            max_len=args.max_len,
            out_dir=out_dir,
            force=args.force,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )


if __name__ == "__main__":
    main()
