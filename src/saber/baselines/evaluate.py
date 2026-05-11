"""Compute simplified metrics A1-A4, B1, C3 from saved baseline outputs.

Pure I/O wrapper around `saber.metrics`: loads saved baseline answers + the
oracle behavior + GT aliases, tallies into the shared cell-counts schema,
and writes per (method, backbone, dataset) + aggregate JSON / CSV.

Inputs:
    data/baselines/<run-name>/<method>/<backbone>/answers_<dataset>.jsonl
    data/evaluation/<backbone>/behavior_<dataset>.jsonl   (oracle pk_correct/ck_correct)
    data/datasets/<dataset>.jsonl                          (ground-truth aliases)

Outputs:
    data/baselines/<run-name>/metrics_simplified.json
    data/baselines/<run-name>/metrics_simplified.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from saber import metrics


DATASET_DIR = Path("$SABER_DATA/datasets")
EVAL_DIR = Path("$SABER_DATA/evaluation")
OUT_BASE = Path("$SABER_DATA/baselines")

ALL_BASELINES = [
    "closed_book", "dia", "internal_eval", "context_eval",
    "implicit_scr", "explicit_scr", "tacs_lr",
]
BACKBONES = ["llama-3.1-8b-instruct", "llama-3.2-3b-instruct", "qwen2.5-7b-instruct"]
POOL2_DATASETS = [
    "confiqa-qa", "confiqa-mr", "confiqa-mc",
    "conflictqa-popqa-llama2-7b", "conflictbank-pilot10k",
    "triviaqa-huang", "nq-huang",
]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def tally_records(records: list[dict], ds_rows_by_qid: dict, beh_by_qid: dict) -> dict:
    """Run each baseline answer record through metrics.tally."""
    cells = metrics.empty_cells()
    for r in records:
        if r.get("error"):
            continue
        qid = r.get("qid")
        if qid is None:
            continue
        ds_row = ds_rows_by_qid.get(qid)
        beh = beh_by_qid.get(qid)
        if ds_row is None or beh is None:
            continue
        oracle = metrics.cell_label(beh.get("pk_correct_alias"), beh.get("ck_correct_alias"))
        metrics.tally(
            cells,
            oracle_cell=oracle,
            answer=r.get("final_answer") or "",
            ground_truth=ds_row.get("ground_truth") or [],
            pk_answer=beh.get("pk_answer") or "",
            ck_answer=beh.get("ck_answer") or "",
        )
    return cells


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--baselines", nargs="+", default=ALL_BASELINES)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=POOL2_DATASETS)
    args = ap.parse_args()

    print(f"[evaluate] run-name={args.run_name}", flush=True)
    print(f"[evaluate] {len(args.baselines)} baselines × {len(args.backbones)} backbones × "
          f"{len(args.datasets)} datasets", flush=True)
    t0 = time.time()

    print(f"[evaluate] loading datasets ...", flush=True)
    ds_cache = {ds: {r["qid"]: r for r in load_jsonl(DATASET_DIR / f"{ds}.jsonl")}
                for ds in args.datasets}
    print(f"[evaluate] loading behaviors ...", flush=True)
    beh_cache = {(bb, ds): {b["qid"]: b for b in load_jsonl(EVAL_DIR / bb / f"behavior_{ds}.jsonl")}
                 for bb in args.backbones for ds in args.datasets}
    print(f"[evaluate] loaded in {time.time()-t0:.1f}s", flush=True)

    results: dict = {}
    for method in args.baselines:
        results[method] = {}
        for bb in args.backbones:
            results[method][bb] = {"per_dataset": {}, "aggregated": None}
            agg = metrics.empty_cells()
            for ds in args.datasets:
                path = OUT_BASE / args.run_name / method / bb / f"answers_{ds}.jsonl"
                if not path.exists():
                    continue
                cells = tally_records(load_jsonl(path), ds_cache[ds], beh_cache[(bb, ds)])
                results[method][bb]["per_dataset"][ds] = metrics.compute_metrics(cells)
                agg = metrics.add_cells(agg, cells)
            results[method][bb]["aggregated"] = metrics.compute_metrics(agg)
        print(f"[evaluate] {method} done in {time.time()-t0:.1f}s cumulative", flush=True)

    out_dir = OUT_BASE / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metrics_simplified.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[evaluate] saved {out_json}", flush=True)

    out_csv = out_dir / "metrics_simplified.csv"
    with out_csv.open("w") as f:
        f.write("method,backbone,dataset,N,A1,A2,A3,A4,B1_C00,B1_C01,B1_C10,B1_C11,C3\n")
        for method in args.baselines:
            for bb in args.backbones:
                for ds, m in results[method][bb]["per_dataset"].items():
                    b1 = m["B1_per_cell_acc"]
                    row = [method, bb, ds, m["N"],
                           m["A1_e2e_acc"], m["A2_acc_t"], m["A3_acc_f"], m["A4_sf"],
                           b1["C00"], b1["C01"], b1["C10"], b1["C11"], m["C3_mr"]]
                    f.write(",".join("" if v is None else str(v) for v in row) + "\n")
                magg = results[method][bb]["aggregated"]
                if magg and magg["N"] > 0:
                    b1 = magg["B1_per_cell_acc"]
                    row = [method, bb, "ALL_AGG", magg["N"],
                           magg["A1_e2e_acc"], magg["A2_acc_t"], magg["A3_acc_f"], magg["A4_sf"],
                           b1["C00"], b1["C01"], b1["C10"], b1["C11"], magg["C3_mr"]]
                    f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    print(f"[evaluate] saved {out_csv}", flush=True)

    print()
    print(f"=== {args.run_name} aggregate metrics across {len(args.datasets)} datasets ===")
    print(f"{'method':<14} {'backbone':<22} {'N':>7} "
          f"{'A1':>6} {'A2':>6} {'A3':>6} {'A4':>6} "
          f"{'C00':>6} {'C01':>6} {'C10':>6} {'C11':>6} {'C3':>6}", flush=True)
    print("-" * 110, flush=True)
    for method in args.baselines:
        for bb in args.backbones:
            agg = results[method][bb].get("aggregated") or {}
            b1 = agg.get("B1_per_cell_acc", {}) or {}
            print(f"  {method:<12} {bb:<22} {agg.get('N', 0):>7} "
                  f"{fmt(agg.get('A1_e2e_acc')):>6} "
                  f"{fmt(agg.get('A2_acc_t')):>6} "
                  f"{fmt(agg.get('A3_acc_f')):>6} "
                  f"{fmt(agg.get('A4_sf')):>6} "
                  f"{fmt(b1.get('C00')):>6} "
                  f"{fmt(b1.get('C01')):>6} "
                  f"{fmt(b1.get('C10')):>6} "
                  f"{fmt(b1.get('C11')):>6} "
                  f"{fmt(agg.get('C3_mr')):>6}", flush=True)
        print(flush=True)
    print(f"[evaluate] total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
