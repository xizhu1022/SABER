"""Generate a frozen train/val/test split for the SABER benchmark pool.

Group-aware stratified split (group_by=(dataset, question), stratify_by=dataset)
with a fixed seed. The output JSON is committed so every training run reads
from the same partition.

Run once::
    PYTHONPATH=src python -m saber.data.build_split --name saber_split

Schema::
    {"name": ..., "seed": ..., "frac_val": ..., "frac_test": ...,
     "datasets": [...], "split_strategy": "...",
     "stats_per_dataset": {ds: {n_groups, n_train, n_val, n_test, ...}},
     "partition": {ds: {train: [qid...], val: [qid...], test: [qid...]}},
     "ts": ...}
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from saber.config import DATA_ROOT


REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = REPO_ROOT / "splits"
DATASET_DIR = DATA_ROOT / "datasets"

POOL2_DATASETS = [
    "confiqa-qa", "confiqa-mr", "confiqa-mc",
    "conflictqa-popqa-llama2-7b", "conflictbank-pilot10k",
    "triviaqa-huang", "nq-huang",
]
HELD_OUT_DATASETS: list[str] = []


def _load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="saber_split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac-val", type=float, default=0.1)
    ap.add_argument("--frac-test", type=float, default=0.1)
    args = ap.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[build_split] generating {args.name} (seed={args.seed})", flush=True)
    rng = np.random.default_rng(args.seed)

    # Per-dataset: {question -> [qids]} (question is the group key)
    per_ds_groups: dict[str, dict[str, list[str]]] = {}
    for ds in POOL2_DATASETS:
        rows = _load_jsonl(DATASET_DIR / f"{ds}.jsonl")
        groups: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            groups[r["question"]].append(r["qid"])
        per_ds_groups[ds] = dict(groups)
        print(f"  [{ds}] n_rows={len(rows)} n_groups={len(groups)}", flush=True)

    # Per-dataset 80/10/10 split — groups are shuffled within each dataset, so
    # stratification by dataset is exact (each dataset gets the same fractions).
    partition: dict[str, dict[str, list[str]]] = {}
    stats: dict[str, dict] = {}
    for ds, groups in per_ds_groups.items():
        gids = list(groups.keys())
        rng.shuffle(gids)
        n = len(gids)
        n_te = int(round(n * args.frac_test))
        n_va = int(round(n * args.frac_val))
        test_g = gids[:n_te]
        val_g = gids[n_te:n_te + n_va]
        train_g = gids[n_te + n_va:]
        train_q = [q for g in train_g for q in groups[g]]
        val_q = [q for g in val_g for q in groups[g]]
        test_q = [q for g in test_g for q in groups[g]]
        partition[ds] = {"train": train_q, "val": val_q, "test": test_q}
        stats[ds] = {
            "n_rows": sum(len(v) for v in groups.values()),
            "n_groups": n, "n_groups_train": len(train_g),
            "n_groups_val": len(val_g), "n_groups_test": len(test_g),
            "n_train": len(train_q), "n_val": len(val_q), "n_test": len(test_q),
        }

    n_train = sum(s["n_train"] for s in stats.values())
    n_val = sum(s["n_val"] for s in stats.values())
    n_test = sum(s["n_test"] for s in stats.values())

    manifest = {
        "name": args.name,
        "seed": args.seed,
        "frac_val": args.frac_val,
        "frac_test": args.frac_test,
        "datasets": POOL2_DATASETS,
        "held_out": HELD_OUT_DATASETS,
        "split_strategy": "group_by_(dataset,question) + stratify_by_dataset",
        "n_train_total": n_train,
        "n_val_total": n_val,
        "n_test_total": n_test,
        "stats_per_dataset": stats,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "partition": partition,
    }

    out_path = SPLIT_DIR / f"{args.name}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[build_split] wrote {out_path}", flush=True)
    print(f"  total: train={n_train}  val={n_val}  test={n_test}", flush=True)
    print(f"  per-dataset breakdown:", flush=True)
    for ds, s in stats.items():
        print(f"    {ds:<32} train={s['n_train']:<6} val={s['n_val']:<5} "
              f"test={s['n_test']:<5} groups={s['n_groups']}", flush=True)


if __name__ == "__main__":
    main()
