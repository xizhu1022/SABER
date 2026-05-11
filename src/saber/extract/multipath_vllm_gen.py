"""Multi-path reasoning-trace generation with vLLM (chat-template version).

Per backbone, loads vLLM once and generates K reasoning traces per
``(qid, side)`` under different ``(temperature, seed)`` settings, using the
SABER prompt template:

  - chat template (system + user roles via apply_chat_template)
  - "brief" instruction in user content
  - assume-correct framing per side
  - repetition_penalty configurable
  - max_tokens fixed at 512

Output (``artifacts/{backbone}/traces/``):
  {dataset}.K{k}.traces.jsonl    # one row per (qid, side); fields:
                                 # qid, side, k, temperature, seed,
                                 # trace, n_tok
  {dataset}.K{k}.traces_done     # marker, atomic write

Skip-already-done: if ``.traces_done`` exists for a given ``(dataset, K)``,
that combination is skipped.

Usage::
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3 \\
      python -u -m saber.extract.multipath_vllm_gen \\
        --backbone llama-3.1-8b-instruct \\
        --tp 4 \\
        --K-list 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


SYSTEM_PROMPT = "You are a concise assistant. Answer the question with brief reasoning."

USER_PK_TEMPLATE = (
    "Question: {question}\n"
    "Assume your own knowledge is correct and complete for this question. "
    "Briefly reason using only your prior knowledge, then state your final answer."
)

USER_CK_TEMPLATE = (
    "Question: {question}\n"
    "Document: {ck_text}\n"
    "Assume the document is reliable and contains the correct answer. "
    "Briefly reason using only the document, then state your final answer."
)


BACKBONE_PATHS = {
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
}


# K → (temperature, seed)
K_CONFIG = {
    1: (0.0, 42),
    2: (0.7, 43),
    3: (0.9, 44),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONE_PATHS.keys()))
    ap.add_argument("--datasets", nargs="+", default=[
        "confiqa-qa", "confiqa-mr", "confiqa-mc",
        "conflictqa-popqa-llama2-7b", "conflictbank-pilot10k",
        "triviaqa-huang", "nq-huang",
    ])
    ap.add_argument("--manifest", default="data/splits/saber_split.json",
                    help="Path to the SABER benchmark split JSON.")
    ap.add_argument("--K-list", nargs="+", type=int, default=[1, 2, 3],
                    help="which K values to run (uses K_CONFIG temperature/seed)")
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out-suffix", default="",
                    help="suffix for output dir (so test runs don't clobber)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from saber.config import DATA_ROOT, ARTIFACTS_DIR
    EVAL = DATA_ROOT / "evaluation" / args.backbone
    DATASET_DIR = DATA_ROOT / "datasets"
    OUT_DIR = ARTIFACTS_DIR / args.backbone / f"traces{args.out_suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[multipath-gen] backbone={args.backbone}  out={OUT_DIR}", flush=True)

    # Load manifest
    manifest_qids: dict[str, set] = {}
    if args.manifest:
        manifest_data = json.loads(Path(args.manifest).read_text())
        manifest_qids = {ds: set(qs) for ds, qs in manifest_data["datasets"].items()}
        print(f"[multipath-gen] manifest: {sum(len(s) for s in manifest_qids.values())} qids "
              f"across {len(manifest_qids)} datasets", flush=True)

    # Load tokenizer for chat template
    print(f"[multipath-gen] loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_PATHS[args.backbone])

    # Decide which (dataset, K) to process — skip done
    work: list[tuple[str, int]] = []
    for ds in args.datasets:
        for K in args.K_list:
            done_marker = OUT_DIR / f"{ds}.K{K}.traces_done"
            if done_marker.exists() and not args.force:
                print(f"[multipath-gen] {ds} K={K} done, skip", flush=True)
                continue
            work.append((ds, K))

    if not work:
        print("[multipath-gen] all datasets×K done", flush=True)
        return

    # Load vLLM once
    print(f"[multipath-gen] loading {args.backbone} via vLLM (tp={args.tp}) ...", flush=True)
    t_load = time.time()
    llm = LLM(
        model=BACKBONE_PATHS[args.backbone],
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
    )
    print(f"[multipath-gen] loaded in {time.time()-t_load:.1f}s", flush=True)

    for ds, K in work:
        ds_path = DATASET_DIR / f"{ds}.jsonl"
        beh_path = EVAL / f"behavior_{ds}.jsonl"
        if not ds_path.exists() or not beh_path.exists():
            print(f"[multipath-gen] {ds}: missing files, skip", flush=True)
            continue

        # Load qids
        ds_by_qid = {}
        with ds_path.open() as f:
            for line in f:
                d = json.loads(line)
                ds_by_qid[d["qid"]] = d
        beh_qids = set()
        manifest_set = manifest_qids.get(ds) if args.manifest else None
        with beh_path.open() as f:
            for line in f:
                b = json.loads(line)
                if b["qid"] not in ds_by_qid:
                    continue
                if manifest_set is not None and b["qid"] not in manifest_set:
                    continue
                beh_qids.add(b["qid"])

        ordered_qids = sorted(beh_qids)
        n_qids = len(ordered_qids)
        if n_qids == 0:
            print(f"[multipath-gen] {ds} K={K}: 0 qids, mark done", flush=True)
            (OUT_DIR / f"{ds}.K{K}.traces.jsonl").touch()
            (OUT_DIR / f"{ds}.K{K}.traces_done").touch()
            continue

        # Build prompts via chat template
        prompts = []
        meta = []
        for qid in ordered_qids:
            row = ds_by_qid[qid]
            messages_pk = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PK_TEMPLATE.format(question=row["question"])},
            ]
            messages_ck = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_CK_TEMPLATE.format(
                    question=row["question"], ck_text=row["ck_text"])},
            ]
            prompts.append(tok.apply_chat_template(
                messages_pk, tokenize=False, add_generation_prompt=True))
            meta.append((qid, "PK"))
            prompts.append(tok.apply_chat_template(
                messages_ck, tokenize=False, add_generation_prompt=True))
            meta.append((qid, "CK"))

        T, seed = K_CONFIG[K]
        sp = SamplingParams(
            temperature=T, top_p=1.0,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens, seed=seed,
        )
        n_traces = len(prompts)
        print(f"[multipath-gen] {ds} K={K} (T={T}, seed={seed}): generating {n_traces} traces "
              f"({n_qids} qids × 2 sides) ...", flush=True)

        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        elapsed = time.time() - t0
        rate = n_traces / max(elapsed, 1e-9)
        n_tok_total = sum(len(r.outputs[0].token_ids) for r in outputs)
        print(f"[multipath-gen] {ds} K={K}: done in {elapsed:.1f}s "
              f"({rate:.2f} traces/s, {n_tok_total/elapsed:.0f} tok/s)", flush=True)

        # Write JSONL atomically
        out_path = OUT_DIR / f"{ds}.K{K}.traces.jsonl"
        tmp_path = out_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w") as f:
            for (qid, side), out in zip(meta, outputs):
                f.write(json.dumps({
                    "qid": qid, "side": side, "k": K,
                    "temperature": T, "seed": seed,
                    "trace": out.outputs[0].text,
                    "n_tok": len(out.outputs[0].token_ids),
                }, ensure_ascii=False) + "\n")
        tmp_path.replace(out_path)
        (OUT_DIR / f"{ds}.K{K}.traces_done").touch()
        print(f"[multipath-gen] {ds} K={K}: wrote {out_path}", flush=True)

    print("[multipath-gen] all done", flush=True)


if __name__ == "__main__":
    main()
