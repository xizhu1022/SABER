"""SABER probe: joint PK + CK binary heads on the query-only hidden state.

Two independent binary predictors share the same mid-layer last-prompt-token
hidden state of the frozen backbone, extracted from the query alone (no
context appended, `hidden_nock.npy`). They predict `pk_correct_alias` and
`ck_correct_alias` independently, and their joint output (sigmoid each,
threshold tau) maps to one of four reliability cells {C00, C01, C10, C11},
which then drives the 4-cell routing decision (abstain / use CK / use PK /
either).

    h_query in R^d   (no CK in the prompt)
    +-- head_PK -> P(pk_correct = 1)
    +-- head_CK -> P(ck_correct = 1)
        joint -> cell in {C00, C01, C10, C11}
        cell  -> routing in {abstain, use_CK, use_PK, either}

Generic helpers (data loading, group-aware splits, head trainers, per-head
metrics, cell metrics) live in `probe_utils.py` and `saber.metrics`.

Usage::
    PYTHONPATH=src python -m saber.methods.saber_probe \\
        --backbone llama-3.1-8b-instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.preprocessing import StandardScaler

from saber import metrics
from saber.methods import probe_utils as pu


LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
TAG = "saber"
SPLIT_NAME = "saber_split"
INPUT_VARIANT = "hidden_nock"  # query-only; routing fires before CK


def joint_cell_metrics(
    meta: list[dict], pk_proba: np.ndarray, ck_proba: np.ndarray,
    *, threshold: float = 0.5,
) -> dict:
    """4-cell metrics from two independent head outputs (uses metrics.py)."""
    pk_pred = (pk_proba >= threshold).astype(int)
    ck_pred = (ck_proba >= threshold).astype(int)
    cells = metrics.empty_cells()
    for i, m in enumerate(meta):
        predicted = f"C{int(pk_pred[i])}{int(ck_pred[i])}"
        answer = metrics.routed_answer(predicted, m["pk_answer"], m["ck_answer"])
        metrics.tally(
            cells,
            oracle_cell=m["cell"],
            answer=answer,
            ground_truth=m["ground_truth"] or [],
            pk_answer=m["pk_answer"],
            ck_answer=m["ck_answer"],
            predicted_cell=predicted,
        )
    return metrics.compute_metrics(cells)


def per_dataset_joint_metrics(meta, pk_proba, ck_proba) -> dict:
    by_ds: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by_ds.setdefault(m["dataset"], []).append(i)
    out: dict = {}
    for ds, idx in by_ds.items():
        r = joint_cell_metrics([meta[i] for i in idx], pk_proba[idx], ck_proba[idx])
        r.pop("_cell_counts", None)
        out[ds] = r
    return out


def run_backbone(backbone: str, seed: int = 42, tag: str = TAG) -> dict:
    print(f"\n{'='*60}\n[{backbone}] starting\n{'='*60}", flush=True)
    t0 = time.time()

    split = pu.load_split(SPLIT_NAME)
    print(f"\n[{backbone}] loading Pool 2 ({len(split['datasets'])} datasets) "
          f"variant={INPUT_VARIANT} ...", flush=True)
    X_pool, meta_pool = pu.load_pool(backbone, split["datasets"], variant=INPUT_VARIANT)
    cell_dist = {c: 0 for c in metrics.CELLS}
    for m in meta_pool:
        cell_dist[m["cell"]] += 1
    print(f"  total: N={len(meta_pool)}  X.shape={X_pool.shape}  cells={cell_dist}", flush=True)

    print(f"\n[{backbone}] applying frozen split '{SPLIT_NAME}' "
          f"({split['split_strategy']}) ...", flush=True)
    train_idx, val_idx, test_idx = pu.apply_split(meta_pool, split)
    print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    print(f"\n[{backbone}] loading held-out {split['held_out']} ...", flush=True)
    X_ho, meta_ho = pu.load_pool(backbone, split["held_out"], variant=INPUT_VARIANT)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_pool[train_idx])
    X_va = scaler.transform(X_pool[val_idx])
    X_te = scaler.transform(X_pool[test_idx])
    X_ho_s = scaler.transform(X_ho) if len(X_ho) > 0 else None

    def y_for(field: str, idx) -> np.ndarray:
        return np.array([meta_pool[i][field] for i in idx], dtype=int)

    targets = {
        "pk": (y_for("pk_correct", train_idx), y_for("pk_correct", val_idx),
               y_for("pk_correct", test_idx),
               np.array([m["pk_correct"] for m in meta_ho], dtype=int) if meta_ho else None),
        "ck": (y_for("ck_correct", train_idx), y_for("ck_correct", val_idx),
               y_for("ck_correct", test_idx),
               np.array([m["ck_correct"] for m in meta_ho], dtype=int) if meta_ho else None),
    }

    results: dict = {
        "backbone": backbone, "tag": tag,
        "config": {
            "split_manifest": SPLIT_NAME,
            "train_datasets": split["datasets"], "held_out": split["held_out"],
            "drop_c00": False,
            "split_strategy": split["split_strategy"],
            "input": f"{INPUT_VARIANT} (query-only) layer=L/2 last_prompt_token",
            "label_source": "alias_match",
        },
        "split_sizes": {
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "n_pool_total": len(meta_pool), "cell_dist_pool": cell_dist,
            "n_held_out": len(meta_ho),
        },
        "heads": {"pk": {}, "ck": {}}, "joint_metrics": {},
    }

    for head_name in ("pk", "ck"):
        y_tr, y_va, y_te, y_ho = targets[head_name]
        print(f"\n[{backbone}] training head='{head_name}' ...", flush=True)
        for probe_name, train_fn in pu.HEAD_TRAINERS.items():
            print(f"  --- probe={probe_name} ---", flush=True)
            r = train_fn(X_tr, y_tr, X_va, y_va, head_name=head_name, seed=seed)
            best = r["best"]
            clf = best.pop("clf")
            best.pop("auroc_val", None)
            proba_te = clf.predict_proba(X_te)[:, 1]
            test_m = pu.head_metrics(y_te, proba_te)
            per_ds = pu.per_dataset_auroc([meta_pool[i] for i in test_idx], y_te, proba_te)
            ho_m, proba_ho = None, None
            if X_ho_s is not None and y_ho is not None and len(set(y_ho.tolist())) >= 2:
                proba_ho = clf.predict_proba(X_ho_s)[:, 1]
                ho_m = pu.head_metrics(y_ho, proba_ho)
            results["heads"][head_name][probe_name] = {
                "best_hyperparams": best, "trace": r["trace"],
                "test_metrics": test_m, "per_dataset_auroc_test": per_ds,
                "redditqa_ood_metrics": ho_m,
                "_proba_test": proba_te.tolist(),
                "_proba_ho": proba_ho.tolist() if proba_ho is not None else None,
            }
            print(f"    test AUROC={test_m['auroc']:.4f} acc={test_m['accuracy']:.4f}"
                  + (f"  ood={ho_m['auroc']:.4f}" if ho_m else ""), flush=True)

    print(f"\n[{backbone}] computing joint cell metrics (A1-A4, B1, B3[C00], C3) ...", flush=True)
    test_meta = [meta_pool[i] for i in test_idx]
    for pk_probe in pu.HEAD_TRAINERS:
        for ck_probe in pu.HEAD_TRAINERS:
            pk_proba = np.array(results["heads"]["pk"][pk_probe]["_proba_test"])
            ck_proba = np.array(results["heads"]["ck"][ck_probe]["_proba_test"])
            joint = joint_cell_metrics(test_meta, pk_proba, ck_proba)
            joint.pop("_cell_counts", None)
            joint["per_dataset"] = per_dataset_joint_metrics(test_meta, pk_proba, ck_proba)
            pk_ho = results["heads"]["pk"][pk_probe]["_proba_ho"]
            ck_ho = results["heads"]["ck"][ck_probe]["_proba_ho"]
            if pk_ho is not None and ck_ho is not None:
                ho = joint_cell_metrics(meta_ho, np.array(pk_ho), np.array(ck_ho))
                ho.pop("_cell_counts", None)
                joint["redditqa_ood"] = ho
            results["joint_metrics"][f"{pk_probe}+{ck_probe}"] = joint
            print(f"  [{pk_probe}+{ck_probe}] A1={joint['A1_e2e_acc']} A4={joint['A4_sf']} "
                  f"B3[C00]={joint['B3_C00_abstain_coverage']} C3={joint['C3_mr']}", flush=True)

    for h in ("pk", "ck"):
        for probe in pu.HEAD_TRAINERS:
            results["heads"][h][probe].pop("_proba_test", None)
            results["heads"][h][probe].pop("_proba_ho", None)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    print(f"\n[{backbone}] DONE in {results['elapsed_sec']:.0f}s", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=["llama-3.1-8b-instruct", "qwen2.5-3b-instruct",
                             "llama-3.2-3b-instruct", "qwen2.5-7b-instruct"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = run_backbone(args.backbone, seed=args.seed, tag=args.tag)
    out_path = LOG_DIR / f"exp_joint_probe_{args.tag}_{args.backbone}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
