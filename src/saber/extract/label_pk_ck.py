"""Extract ModelBehaviorRow for 4-cell labeling.

Pipeline (datasets.md §5.3 + §7):
  Phase 1: PK greedy with dedupe per (model_key, question_text)
  Phase 2: CK greedy per row (uses cached PK answer)
  Phase 3: alias_match → pk_correct_alias / ck_correct_alias
  Phase 5 (assembly): write ModelBehaviorRow JSONL

Judge (§4.2/§4.3) is run by score_with_judge.py as a separate phase.

Outputs:
  data/evaluation/{model_key}/pk_cache.jsonl              (Phase 1 cache, shared across datasets)
  data/evaluation/{model_key}/behavior_{dataset}.jsonl    (final per-dataset)

Usage::

    PYTHONPATH=src python -m saber.extract_model_behavior \\
        --backbone llama-3.1-8b-instruct \\
        --datasets confiqa-qa confiqa-mr confiqa-mc \\
        --max-new-tokens 64 --max-len 4096
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from saber.config import DATA_ROOT, HF_CACHE_DIR
from saber.data.alias_match import alias_match
from saber.models import get_model_spec
from saber.prompts import (
    CK_ANSWER_USER_TEMPLATE,
    PK_ANSWER_USER_TEMPLATE,
    ANSWER_SYSTEM,
    ck_answer_messages,
    pk_answer_messages,
)

DATASET_DIR = DATA_ROOT / "datasets"
EVAL_DIR = DATA_ROOT / "evaluation"


def _qhash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_jsonl(path: Path, n: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def _greedy(model, tokenizer, messages, max_new_tokens: int,
            max_len: int) -> tuple[str, int]:
    """Apply chat template, run greedy, return (answer_text, new_token_count).

    If the rendered prompt exceeds max_len, the user message is truncated
    from the END (preserving system + question prefix) before tokenizing.
    """
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(prompt_str, return_tensors="pt")
    if enc["input_ids"].shape[1] > max_len:
        # Truncate the rendered prompt's middle (the long ck_text) by
        # tokenizing the user content alone, truncating, then re-rendering.
        # Simpler fallback: just truncate the input_ids tail of the text body.
        ids = enc["input_ids"][0]
        # Keep first 512 tokens + last (max_len - 512) tokens
        head_n = 512
        tail_n = max_len - head_n
        kept = torch.cat([ids[:head_n], ids[-tail_n:]])
        enc = {"input_ids": kept.unsqueeze(0),
               "attention_mask": torch.ones_like(kept).unsqueeze(0)}
    enc = {k: v.to(model.device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_ids = out[0, in_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    # Take first line only
    text = text.split("\n")[0].strip()
    return text, int(new_ids.shape[0])


def _load_pk_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    cache = {}
    with cache_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                cache[d["question_hash"]] = d
    return cache


def _append_pk_cache(cache_path: Path, entry: dict) -> None:
    with cache_path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _pk_pass(model, tokenizer, questions: list[str], cache_path: Path,
               model_key: str, max_new_tokens: int, max_len: int,
               desc: str) -> dict[str, dict]:
    """For each unique question, compute pk_answer (use cache where present)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_pk_cache(cache_path)
    todo = []
    for q in questions:
        h = _qhash(q)
        if h not in cache:
            todo.append((h, q))
    if not todo:
        print(f"[{desc}] PK cache hit 100% ({len(cache)} entries)")
        return cache
    print(f"[{desc}] PK cache miss {len(todo)}/{len(questions)}; computing")
    t0 = time.time()
    for h, q in tqdm(todo, desc=f"PK {desc}"):
        ans, ntok = _greedy(model, tokenizer, pk_answer_messages({"question": q}),
                            max_new_tokens, max_len)
        entry = {
            "question_hash": h,
            "question": q,
            "pk_answer": ans,
            "pk_token_count": ntok,
            "model_key": model_key,
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        cache[h] = entry
        _append_pk_cache(cache_path, entry)
    print(f"[{desc}] PK phase done in {time.time()-t0:.0f}s")
    return cache


def extract_dataset(
    model,
    tokenizer,
    model_key: str,
    dataset_name: str,
    n: int | None,
    max_new_tokens: int,
    max_len: int,
    out_dir: Path,
    pk_cache_path: Path,
    force: bool,
    seed: int = 0,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sharded = start_idx is not None or end_idx is not None
    if sharded:
        s = start_idx or 0
        e_str = "" if end_idx is None else str(end_idx)
        suffix = f".range{s}_{e_str}"
        out_path = out_dir / f"behavior_{dataset_name}{suffix}.jsonl"
        # Per-shard pk_cache to avoid concurrent-append races across shards.
        pk_cache_path = out_dir / f"pk_cache{suffix}.jsonl"
    else:
        out_path = out_dir / f"behavior_{dataset_name}.jsonl"
    jsonl_path = DATASET_DIR / f"{dataset_name}.jsonl"
    rows = _load_jsonl(jsonl_path, n=n)
    if sharded:
        s = start_idx or 0
        e = end_idx if end_idx is not None else len(rows)
        rows = rows[s:e]
        print(f"[{dataset_name}] sharded rows[{s}:{e}] = {len(rows)} of total")
    print(f"[{dataset_name}] loaded {len(rows)} rows from {jsonl_path}")

    # Idempotency: if out file exists and not --force, read existing qids and skip
    done_qids: set[str] = set()
    if out_path.exists() and not force:
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    done_qids.add(json.loads(line)["qid"])
        print(f"[{dataset_name}] {len(done_qids)} rows already done; skipping those")

    # Phase 1: PK dedupe
    unique_q = []
    seen = set()
    for r in rows:
        if r["question"] not in seen:
            seen.add(r["question"])
            unique_q.append(r["question"])
    pk_cache = _pk_pass(model, tokenizer, unique_q, pk_cache_path,
                          model_key, max_new_tokens, max_len, dataset_name)

    # Phase 2: CK greedy per row (skip done)
    counts = {"C00": 0, "C01": 0, "C10": 0, "C11": 0}
    n_truncated = 0
    t0 = time.time()
    write_mode = "a" if (out_path.exists() and not force and done_qids) else "w"
    with out_path.open(write_mode) as fout:
        for row in tqdm(rows, desc=f"CK {dataset_name}"):
            if row["qid"] in done_qids:
                continue
            qhash = _qhash(row["question"])
            pk_entry = pk_cache[qhash]
            pk_ans = pk_entry["pk_answer"]
            pk_tok = pk_entry["pk_token_count"]

            ck_text_raw = row.get("ck_text")
            if ck_text_raw:
                # Truncate ck_text if needed (passing max_len to _greedy handles it
                # post-render; we still try to be polite at input level)
                ck_msgs = ck_answer_messages(row)
                ck_ans, ck_tok = _greedy(model, tokenizer, ck_msgs,
                                         max_new_tokens, max_len)
            else:
                ck_ans, ck_tok = "", 0

            gt_aliases = row["ground_truth"]
            pk_correct_alias = alias_match(pk_ans, gt_aliases)
            ck_correct_alias = alias_match(ck_ans, gt_aliases) if ck_text_raw else False

            # Final pk_correct = alias_only at this stage; judge runs separately
            pk_correct = pk_correct_alias
            ck_correct = ck_correct_alias
            cell = f"C{int(pk_correct)}{int(ck_correct)}"
            counts[cell] += 1

            out_row = {
                "qid": row["qid"],
                "model_key": model_key,
                "dataset": dataset_name,
                "pk_answer": pk_ans,
                "ck_answer": ck_ans,
                "pk_token_count": pk_tok,
                "ck_token_count": ck_tok,
                "pk_correct_alias": pk_correct_alias,
                "ck_correct_alias": ck_correct_alias,
                "pk_correct_judge": None,
                "ck_correct_judge": None,
                "pk_judge_response": None,
                "ck_judge_response": None,
                "judge_model": None,
                "pk_correct": pk_correct,
                "ck_correct": ck_correct,
                "pk_correct_source": "alias_only",
                "ck_correct_source": "alias_only",
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "pk_extracted_at": pk_entry["extracted_at"],
                "ck_extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "judge_extracted_at": None,
                "metadata": {
                    "ck_relation": row.get("ck_relation"),
                    "ground_truth_text": row.get("ground_truth_text"),
                    "dataset_family": row.get("dataset_family"),
                },
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            fout.flush()

    elapsed = time.time() - t0
    n_total = sum(counts.values())
    if n_total == 0:
        print(f"[{dataset_name}] no new rows processed (all already done)")
        return
    print(f"[{dataset_name}] CK phase done in {elapsed:.0f}s ({n_total} new rows)")
    print(f"[{dataset_name}] cell distribution (alias_only):")
    for k in ("C00", "C01", "C10", "C11"):
        v = counts[k]
        print(f"    {k}: {v:5d}  ({v/n_total*100:5.1f}%)")
    print(f"    pk_correct overall: {(counts['C10']+counts['C11'])/n_total:.3f}")
    print(f"    ck_correct overall: {(counts['C01']+counts['C11'])/n_total:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="llama-3.1-8b-instruct")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--start-idx", type=int, default=None,
                    help="Shard start (0-based, inclusive). Outputs to behavior_{ds}.range{s}_{e}.jsonl.")
    ap.add_argument("--end-idx", type=int, default=None,
                    help="Shard end (exclusive).")
    args = ap.parse_args()

    print(f"[behavior] loading {args.backbone}")
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
    print(f"[behavior] model loaded; hidden_size={model.config.hidden_size}")

    out_dir = args.out_dir if args.out_dir is not None else EVAL_DIR / args.backbone
    pk_cache_path = out_dir / "pk_cache.jsonl"

    for ds in args.datasets:
        extract_dataset(
            model=model,
            tokenizer=tokenizer,
            model_key=args.backbone,
            dataset_name=ds,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            max_len=args.max_len,
            out_dir=out_dir,
            pk_cache_path=pk_cache_path,
            force=args.force,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )


if __name__ == "__main__":
    main()
