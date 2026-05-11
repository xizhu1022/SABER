"""Build all SABER datasets to JSONL under ``data/datasets/{name}.jsonl``.

Usage::

    PYTHONPATH=src python -m saber.data.build                # build all
    PYTHONPATH=src python -m saber.data.build --only confiqa # subset by family

Produces one JSONL per dataset; validates each against the dataset schema
via ``validate_dataset`` (see ``schema.py``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from saber.config import DATA_ROOT
from saber.data.builders import (
    clasheval,
    confiqa,
    conflictbank,
    conflictqa_popqa,
    huang_situated,
    redditqa,
)
from saber.data.schema import row_to_dict, validate_dataset

OUT_DIR = DATA_ROOT / "datasets"

# Registered builders: (family, build_fn).  Each build_fn yields (name, rows) tuples.
BUILDERS = {
    "confiqa":          confiqa.build,                  # yields confiqa-qa, confiqa-mr, confiqa-mc
    "clasheval":        clasheval.build,                # yields clasheval
    "redditqa":         redditqa.build,                 # yields redditqa (test-only)
    "conflictqa":       conflictqa_popqa.build,         # yields conflictqa-popqa-llama2-7b
    "conflictbank":     conflictbank.build,             # pilot10k  (10k raw × 4 = 40k processed)
    "conflictbank-100k": conflictbank.build_pilot100k,  # pilot100k (100k raw × 4 = 400k processed, ~1.4 GB)
    "huang":            huang_situated.build,            # triviaqa-huang + nq-huang (Huang 2025)
}


def _write_jsonl(rows, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(row_to_dict(r), ensure_ascii=False, default=str) + "\n")
    return out_path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None,
                    help="Build only these dataset families (repeatable). "
                         "Options: confiqa / clasheval / redditqa / conflictqa")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    targets = args.only if args.only else list(BUILDERS.keys())
    unknown = [t for t in targets if t not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown family names: {unknown}. Registered: {list(BUILDERS.keys())}")

    print(f"[build] building {len(targets)} families → {args.out_dir}")
    all_summaries = []
    for family in targets:
        print(f"\n[build] === family: {family} ===")
        t0 = time.time()
        build_fn = BUILDERS[family]
        for name, rows in build_fn():
            summary = validate_dataset(rows, name)
            out_path = args.out_dir / f"{name}.jsonl"
            size = _write_jsonl(rows, out_path)
            summary["size_kb"] = round(size / 1024, 1)
            summary["path"] = str(out_path)
            all_summaries.append(summary)
            print(f"[build] {name:<32} n={summary['n_rows']:>7}  "
                  f"sup={summary['n_supportive']:>6}  conf={summary['n_conflicting']:>6}  "
                  f"size={summary['size_kb']:>8} KB  → {out_path.name}")
        print(f"[build] family={family} done in {time.time()-t0:.1f}s")

    # Overall summary
    print("\n[build] === SUMMARY ===")
    total_rows = sum(s["n_rows"] for s in all_summaries)
    total_sup = sum(s["n_supportive"] for s in all_summaries)
    total_conf = sum(s["n_conflicting"] for s in all_summaries)
    print(f"[build] total datasets: {len(all_summaries)}")
    print(f"[build] total rows    : {total_rows}")
    print(f"[build] total sup     : {total_sup}")
    print(f"[build] total conf    : {total_conf}")

    # Persist summary index (merge-preserving if partial build)
    index_path = args.out_dir / "_index.json"
    existing = []
    if index_path.exists() and args.only:
        try:
            prev = json.loads(index_path.read_text()).get("datasets", [])
            just_built = {s["dataset"] for s in all_summaries}
            existing = [s for s in prev if s.get("dataset") not in just_built]
        except json.JSONDecodeError:
            existing = []
    index_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": existing + all_summaries,
    }, indent=2))
    print(f"[build] wrote index {index_path}  (total {len(existing + all_summaries)} datasets)")


if __name__ == "__main__":
    main()
