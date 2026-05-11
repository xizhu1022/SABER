"""Generic baseline runner.

Features:
- Idempotency / resume: skip qids already present in output files
- Multi-dataset: iterate all Pool 2 datasets in one launch
- Multi-backbone parallel: all 3 OpenRouter backbones dispatched concurrently
  (single semaphore limits global API concurrency)
- Stream-save: each result written to disk immediately after task completion,
  with per-bucket asyncio lock to avoid interleaving

Usage:
    OPENROUTER_API_KEY=sk-or-... python -m saber.baselines.run \
        --run-name sanity20 --n 20 --datasets confiqa-qa
    OPENROUTER_API_KEY=sk-or-... python -m saber.baselines.run \
        --run-name pool2_full --datasets confiqa-qa confiqa-mr ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from saber.data.alias_match import alias_match
from saber.baselines.openrouter_client import OpenRouterPool
from saber.baselines.runners import ASYNC_BASELINES, LOCAL_BASELINES


DATASET_DIR = Path("$SABER_DATA/datasets")
EVAL_DIR = Path("$SABER_DATA/evaluation")
OUT_BASE = Path("$SABER_DATA/baselines")

# OpenRouter model IDs for our 3 API backbones (Qwen-2.5-3B not on OpenRouter).
# Note: previously tried :free Llama-3.2-3B with auto-fallback, but free tier was
# rate-limited heavily — slowed throughput from ~85 task/s to ~6 task/s. Reverted
# to paid for all 3 backbones (cost premium ~$10-15 acceptable for ~3-5x speed).
OPENROUTER_MODELS = {
    "llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct": "meta-llama/llama-3.2-3b-instruct",
    "qwen2.5-7b-instruct": "qwen/qwen-2.5-7b-instruct",
}

# Local backend: HF transformers, one model copy per GPU (LocalHFPool spawns
# 8 worker processes). Map backbone alias → HF repo id OR absolute model dir.
# Qwen-7B uses an absolute path because its HF cache copy is missing tokenizer
# files; the Knowledgeable-R1 dir has a complete snapshot.
LOCAL_MODELS = {
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
}

ALL_BASELINES = [
    "closed_book", "dia", "internal_eval", "context_eval",
    "implicit_scr", "explicit_scr", "tacs_lr",
]

POOL2_DATASETS = [
    "confiqa-qa", "confiqa-mr", "confiqa-mc",
    "conflictqa-popqa-llama2-7b", "conflictbank-pilot10k",
    "triviaqa-huang", "nq-huang",
]


# ─── IO helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists(): return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def output_path(run_name: str, method: str, backbone: str, dataset: str) -> Path:
    return OUT_BASE / run_name / method / backbone / f"answers_{dataset}.jsonl"


def load_done_qids(run_name: str, method: str, backbone: str, dataset: str) -> set[str]:
    """Read existing output file and return set of qids already processed.

    A qid is "done" iff it has at least one row with no error (errored rows
    can be retried). The error field is a free-form sentinel: real Python
    None, the literal string "None", or empty string all mean "success".
    Anything else is a real error message.
    """
    path = output_path(run_name, method, backbone, dataset)
    if not path.exists(): return set()
    done = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            err = d.get("error")
            qid = d.get("qid")
            if qid and (err is None or err in ("", "None")):
                done.add(qid)
    return done


def cell_label(pk: bool, ck: bool) -> str:
    return f"C{int(bool(pk))}{int(bool(ck))}"


def sample_balanced(
    rows: list[dict], behavior_by_qid: dict, n: int | None, seed: int = 0,
) -> list[dict]:
    """Cell-balanced sampling. n=None returns all rows in order."""
    if n is None or n >= len(rows):
        return rows
    by_cell: dict[str, list[dict]] = {"C00": [], "C01": [], "C10": [], "C11": []}
    for r in rows:
        b = behavior_by_qid.get(r["qid"])
        if not b: continue
        c = cell_label(b.get("pk_correct_alias"), b.get("ck_correct_alias"))
        by_cell[c].append(r)
    rng = random.Random(seed)
    for cell in by_cell:
        rng.shuffle(by_cell[cell])
    per_cell = max(1, n // 4)
    picked: list[dict] = []
    for cell in ("C10", "C01", "C11", "C00"):
        picked.extend(by_cell[cell][:per_cell])
    rest = n - len(picked)
    if rest > 0:
        all_remaining = sum((by_cell[c][per_cell:] for c in by_cell), [])
        rng.shuffle(all_remaining)
        picked.extend(all_remaining[:rest])
    return picked[:n]


# ─── Streaming saves ──────────────────────────────────────────────────────────

class StreamingWriter:
    """Per-bucket append-mode writer with asyncio Lock to avoid interleaving."""

    def __init__(self):
        self.locks: dict[tuple, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def write(self, run_name: str, method: str, bb: str, ds: str, record: dict):
        path = output_path(run_name, method, bb, ds)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = (method, bb, ds)
        async with self.locks[key]:
            # File I/O in async context: ok for small writes, no aiofiles needed
            with path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── Main runner ──────────────────────────────────────────────────────────────

async def run_one_task(
    pool: OpenRouterPool, writer: StreamingWriter,
    run_name: str, method: str, bb: str, ds: str, model_id: str,
    row: dict, behavior: dict,
) -> tuple[str, str, str, dict]:
    """Run one (method, bb, row) task and stream-save the result.

    Returns (method, bb, ds, result) for downstream metric aggregation.
    """
    runner = ASYNC_BASELINES[method]
    try:
        result = await runner(pool, model_id, row, behavior)
    except Exception as e:
        result = {"method": method, "qid": row["qid"], "error": f"exception: {type(e).__name__}: {e}"}
    result["backbone"] = bb
    result["dataset"] = ds
    await writer.write(run_name, method, bb, ds, result)
    return (method, bb, ds, result)


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True,
                    help="Output dir tag, e.g. 'sanity20' or 'pool2_full'")
    ap.add_argument("--datasets", nargs="+", default=POOL2_DATASETS)
    ap.add_argument("--n", type=int, default=None,
                    help="Per-dataset sample count (None = use all rows)")
    ap.add_argument("--backend", choices=["openrouter", "local"], default="openrouter",
                    help="openrouter = API; local = 8 HF transformers workers (1/GPU)")
    ap.add_argument("--gpu-ids", nargs="*", type=int, default=None,
                    help="(local) which GPU ids to use; default 0..7")
    ap.add_argument("--backbones", nargs="+", default=None)
    ap.add_argument("--baselines", nargs="+", default=ALL_BASELINES)
    ap.add_argument("--reference-backbone", default="llama-3.1-8b-instruct",
                    help="Backbone whose behavior labels are used for cell-balanced sampling")
    ap.add_argument("--concurrency-per-key", type=int, default=16,
                    help="Per-key (or per-server) in-flight requests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    backbone_to_model = OPENROUTER_MODELS if args.backend == "openrouter" else LOCAL_MODELS
    if args.backbones is None:
        args.backbones = list(backbone_to_model.keys())

    print(f"[run] backend={args.backend}  run-name={args.run_name}", flush=True)
    print(f"[run] datasets={args.datasets}", flush=True)
    print(f"[run] backbones={args.backbones}", flush=True)
    print(f"[run] baselines={args.baselines}", flush=True)
    if args.backend == "openrouter":
        n_keys = len([k for k in (
            os.environ.get("OPENROUTER_API_KEYS") or os.environ.get("OPENROUTER_API_KEY", "")
        ).split(",") if k.strip()])
        print(f"[run] n={args.n} concurrency-per-key={args.concurrency_per_key} "
              f"keys={n_keys} → effective concurrent={args.concurrency_per_key * max(n_keys, 1)}",
              flush=True)
    else:
        if args.gpu_ids is None:
            args.gpu_ids = list(range(8))
        print(f"[run] local HF workers on GPUs: {args.gpu_ids}", flush=True)

    # Build per-dataset row lists + behavior cache
    rows_by_ds: dict[str, list[dict]] = {}
    rows_by_qid_by_ds: dict[str, dict] = {}
    sample_by_ds: dict[str, list[dict]] = {}
    for ds in args.datasets:
        rows = load_jsonl(DATASET_DIR / f"{ds}.jsonl")
        rows_by_ds[ds] = rows
        rows_by_qid_by_ds[ds] = {r["qid"]: r for r in rows}
        # Cell-balanced sample using reference backbone
        ref_beh = load_jsonl(EVAL_DIR / args.reference_backbone / f"behavior_{ds}.jsonl")
        ref_beh_by_qid = {b["qid"]: b for b in ref_beh}
        sample_by_ds[ds] = sample_balanced(rows, ref_beh_by_qid, args.n, seed=args.seed)
        print(f"  [{ds}] {len(rows)} rows total, sampled {len(sample_by_ds[ds])}")

    # Per (bb, ds) behavior cache
    beh_cache: dict[tuple[str, str], dict] = {}
    for bb in args.backbones:
        for ds in args.datasets:
            beh = load_jsonl(EVAL_DIR / bb / f"behavior_{ds}.jsonl")
            beh_cache[(bb, ds)] = {b["qid"]: b for b in beh}

    if args.backend == "openrouter":
        pool = OpenRouterPool(concurrency_per_key=args.concurrency_per_key)
    else:
        # Local backend: lazy-import to avoid loading torch/transformers when
        # running with --backend openrouter.
        from saber.baselines.local_hf_pool import LocalHFPool
        # Local backbones must be a single one — one model loaded per GPU,
        # not multiple models. Validate.
        if len(args.backbones) != 1:
            raise ValueError("local backend supports exactly one --backbones value "
                             "(one HF model loaded per worker)")
        bb = args.backbones[0]
        pool = LocalHFPool(LOCAL_MODELS[bb], gpu_ids=args.gpu_ids)
    writer = StreamingWriter()

    # Chunked dispatch: process one (dataset, backbone, method) chunk at a time.
    # Bounded RAM (only one chunk's tasks in flight) and live progress per chunk.
    t_start = time.time()
    grand_total_async = 0
    grand_total_local = 0
    grand_total_skipped = 0
    chunk_idx = 0
    n_chunks = len(args.datasets) * len(args.backbones) * len(args.baselines)
    try:
        for ds in args.datasets:
            for bb in args.backbones:
                model_id = backbone_to_model[bb]
                beh_by_qid = beh_cache[(bb, ds)]
                for method in args.baselines:
                    chunk_idx += 1
                    chunk_label = f"{ds}/{bb}/{method}"
                    done = load_done_qids(args.run_name, method, bb, ds)
                    pending = []
                    for row in sample_by_ds[ds]:
                        if row["qid"] in done:
                            grand_total_skipped += 1
                            continue
                        if beh_by_qid.get(row["qid"]) is None:
                            continue
                        pending.append(row)
                    if not pending:
                        print(f"[chunk {chunk_idx}/{n_chunks}] {chunk_label}: skip "
                              f"(all {len(done)} already done)", flush=True)
                        continue
                    chunk_t0 = time.time()
                    if method in LOCAL_BASELINES:
                        # Synchronous local writes, no API call.
                        path = output_path(args.run_name, method, bb, ds)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("a") as f:
                            for row in pending:
                                result = LOCAL_BASELINES[method](beh_by_qid[row["qid"]])
                                result["backbone"] = bb
                                result["dataset"] = ds
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                                grand_total_local += 1
                        print(f"[chunk {chunk_idx}/{n_chunks}] {chunk_label}: "
                              f"{len(pending)} local saved in {time.time()-chunk_t0:.1f}s",
                              flush=True)
                        continue
                    # Async chunk — dispatch in micro-batches to avoid asyncio
                    # task pile-up (creating 40k tasks at once stalls the event
                    # loop before any task completes).
                    completed = 0
                    progress_interval = max(50, len(pending) // 4)
                    micro = 200
                    for batch_start in range(0, len(pending), micro):
                        batch = pending[batch_start:batch_start + micro]
                        tasks = [
                            asyncio.create_task(run_one_task(
                                pool, writer, args.run_name, method, bb, ds, model_id,
                                row, beh_by_qid[row["qid"]],
                            ))
                            for row in batch
                        ]
                        for coro in asyncio.as_completed(tasks):
                            await coro
                            completed += 1
                            grand_total_async += 1
                            if completed % progress_interval == 0 or completed == len(pending):
                                elapsed = time.time() - chunk_t0
                                rate = completed / max(elapsed, 1e-9)
                                cost_so_far = sum(s["cost_usd"] for s in pool.summary().values())
                                eta_chunk = (len(pending) - completed) / max(rate, 1e-9)
                                grand_elapsed = time.time() - t_start
                                print(f"  [chunk {chunk_idx}/{n_chunks}] {chunk_label} "
                                      f"[{completed}/{len(pending)}] "
                                      f"{elapsed:.0f}s, {rate:.1f} task/s, "
                                      f"ETA chunk {eta_chunk:.0f}s, "
                                      f"total cost ${cost_so_far:.2f}, "
                                      f"total elapsed {grand_elapsed/60:.1f}m",
                                      flush=True)
    finally:
        await pool.close()

    elapsed = time.time() - t_start
    print(f"\n[run] all done in {elapsed:.0f}s "
          f"(local saved={grand_total_local}, async completed={grand_total_async}, "
          f"skipped={grand_total_skipped})", flush=True)
    print(f"[run] {args.backend} summary (per-model):", flush=True)
    for model, st in pool.summary().items():
        print(f"  {model}: {st}", flush=True)
    print(f"[run] {args.backend} summary (per-{'key' if args.backend=='openrouter' else 'server'}):",
          flush=True)
    for ks in pool.key_summary():
        print(f"  {ks}", flush=True)

    # Compute metrics by reading back from output files
    print(f"\n[run] computing metrics ...")
    metrics = compute_all_metrics(args)
    metrics_path = OUT_BASE / args.run_name / f"metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[run] metrics → {metrics_path}")

    # Print summary table
    print()
    print(f"{'method':<14} {'backbone':<22} {'dataset':<28} | n   | acc  | C10 | C01 | C11 | C00")
    print("-" * 130)
    for ds in args.datasets:
        for bb in args.backbones:
            for method in args.baselines:
                m = metrics["per"][method][bb].get(ds)
                if not m: continue
                pc = m["per_cell_acc"]
                print(f"  {method:<12} {bb:<22} {ds:<28} | {m['n']:>3} | "
                      f"{m['acc']:>4} | "
                      f"{pc.get('C10', '—')} | {pc.get('C01', '—')} | "
                      f"{pc.get('C11', '—')} | {pc.get('C00', '—')}")
        print()


def compute_all_metrics(args) -> dict:
    """Read back all output files and compute metrics per (method, backbone, dataset)."""
    out = {"per": {}, "openrouter_summary": None}
    for method in args.baselines:
        out["per"][method] = {}
        for bb in args.backbones:
            out["per"][method][bb] = {}
            for ds in args.datasets:
                path = output_path(args.run_name, method, bb, ds)
                if not path.exists(): continue
                results = load_jsonl(path)
                rows_by_qid = {r["qid"]: r for r in load_jsonl(DATASET_DIR / f"{ds}.jsonl")}
                beh_by_qid = {b["qid"]: b for b in load_jsonl(EVAL_DIR / bb / f"behavior_{ds}.jsonl")}
                m = evaluate_baseline(results, rows_by_qid, beh_by_qid)
                out["per"][method][bb][ds] = m
    return out


def evaluate_baseline(results: list[dict], rows_by_qid: dict, behaviors: dict) -> dict:
    n = 0; n_correct = 0; n_err = 0
    by_cell = {c: [0, 0] for c in ("C00", "C01", "C10", "C11")}
    by_relation = {"supportive": [0, 0], "conflicting": [0, 0]}
    total_cost = 0.0; total_lat = 0.0
    src_dist: Counter = Counter()
    for r in results:
        if r.get("error"):
            n_err += 1
            continue
        n += 1
        qid = r["qid"]
        ans = r.get("final_answer") or ""
        gt = rows_by_qid.get(qid, {}).get("ground_truth", [])
        correct = alias_match(ans, gt)
        if correct: n_correct += 1
        b = behaviors.get(qid, {})
        cell = cell_label(b.get("pk_correct_alias"), b.get("ck_correct_alias"))
        if cell in by_cell:
            by_cell[cell][0] += 1
            if correct: by_cell[cell][1] += 1
        rel = rows_by_qid.get(qid, {}).get("ck_relation")
        if rel in by_relation:
            by_relation[rel][0] += 1
            if correct: by_relation[rel][1] += 1
        total_cost += r.get("cost_usd") or 0.0
        total_lat += r.get("latency_sec") or 0.0
        src_dist[r.get("predicted_source") or "implicit"] += 1
    metrics = {
        "n": n, "n_err": n_err,
        "acc": round(n_correct / n, 3) if n else None,
        "per_cell_acc": {c: round(by_cell[c][1] / by_cell[c][0], 3)
                         if by_cell[c][0] else None for c in by_cell},
        "per_cell_n": {c: by_cell[c][0] for c in by_cell},
        "acc_t": (round(by_relation["supportive"][1] / by_relation["supportive"][0], 3)
                  if by_relation["supportive"][0] else None),
        "acc_f": (round(by_relation["conflicting"][1] / by_relation["conflicting"][0], 3)
                  if by_relation["conflicting"][0] else None),
        "predicted_source_dist": dict(src_dist),
        "total_cost_usd": round(total_cost, 4),
        "total_latency_sec": round(total_lat, 1),
    }
    if metrics["acc_t"] is not None and metrics["acc_f"] is not None:
        metrics["sf"] = round((metrics["acc_t"] + metrics["acc_f"]) / 2, 3)
    return metrics


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
