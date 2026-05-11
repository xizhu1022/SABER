"""Joint 4-cell routing metrics for SABER.

For each of three multi-trace aggregation configurations
(``K=1`` only, ``mean3``, ``max3``), trains two concat-MLP heads on
``[self-prior, conditional]`` features, then computes the 4-cell routing
metrics (Acc, CF, KF, MFS, ...) on the held-out test split.

Usage::
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \\
      python -u -m saber.metrics.joint_metrics \\
        --backbone qwen2.5-3b-instruct --device cuda:0 \\
        --split-file data/splits/saber_split.json \\
        --cond-layer 12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from saber.metrics.probe_helpers import (
    build_split_indices,
    load_behavior_meta,
    joint_cell_metrics,
    fit_mlp_torch,
    mlp_proba,
    load_aligned_K,
    agg_mean,
    agg_max,
)


def fit_concat_mlp(X_tr, y_tr, X_va, y_va, X_te, *, device,
                   hidden=(512, 128), seed=42):
    model, val_auc, _ = fit_mlp_torch(
        X_tr, y_tr, X_va, y_va, hidden=hidden, device=device, seed=seed)
    proba_te = mlp_proba(model, X_te, device)
    return proba_te, float(val_auc)


def fit_pk_ck_concat_mlp(h_query, cond_pk, cond_ck, indices, y_pk, y_ck, *,
                         device, seed=42):
    """Fit independent PK and CK concat-MLP heads, return test probabilities."""
    s_prior = StandardScaler().fit(h_query[indices["train"]])
    s_pk = StandardScaler().fit(cond_pk[indices["train"]])
    s_ck = StandardScaler().fit(cond_ck[indices["train"]])

    Xp_tr = s_prior.transform(h_query[indices["train"]])
    Xp_va = s_prior.transform(h_query[indices["val"]])
    Xp_te = s_prior.transform(h_query[indices["test"]])
    Xpk_tr = s_pk.transform(cond_pk[indices["train"]])
    Xpk_va = s_pk.transform(cond_pk[indices["val"]])
    Xpk_te = s_pk.transform(cond_pk[indices["test"]])
    Xck_tr = s_ck.transform(cond_ck[indices["train"]])
    Xck_va = s_ck.transform(cond_ck[indices["val"]])
    Xck_te = s_ck.transform(cond_ck[indices["test"]])

    cat_pk_tr = np.concatenate([Xp_tr, Xpk_tr], axis=1).astype(np.float32)
    cat_pk_va = np.concatenate([Xp_va, Xpk_va], axis=1).astype(np.float32)
    cat_pk_te = np.concatenate([Xp_te, Xpk_te], axis=1).astype(np.float32)
    cat_ck_tr = np.concatenate([Xp_tr, Xck_tr], axis=1).astype(np.float32)
    cat_ck_va = np.concatenate([Xp_va, Xck_va], axis=1).astype(np.float32)
    cat_ck_te = np.concatenate([Xp_te, Xck_te], axis=1).astype(np.float32)

    pk_te, pk_val = fit_concat_mlp(
        cat_pk_tr, y_pk[indices["train"]], cat_pk_va, y_pk[indices["val"]],
        cat_pk_te, device=device, seed=seed)
    ck_te, ck_val = fit_concat_mlp(
        cat_ck_tr, y_ck[indices["train"]], cat_ck_va, y_ck[indices["val"]],
        cat_ck_te, device=device, seed=seed)
    return pk_te, ck_te, pk_val, ck_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--datasets", nargs="+", default=[
        "confiqa-qa", "confiqa-mr", "confiqa-mc",
        "conflictqa-popqa-llama2-7b", "conflictbank-pilot10k",
        "triviaqa-huang", "nq-huang"])
    ap.add_argument("--split-file", default="data/splits/saber_split.json")
    ap.add_argument("--cond-layer", type=int, required=True,
                    help="1-indexed model layer at which to read the "
                         "conditional reasoning representation.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-base-suffix", default="",
                    help="Suffix on the cond artefact directory.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Defaults to ``logs/{backbone}_joint_metrics.json``.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[joint] backbone={args.backbone}  device={device}  "
          f"cond_layer={args.cond_layer}", flush=True)

    h_query, cond_pk_K, cond_ck_K, meta = load_aligned_K(
        args.backbone, args.datasets, args.cond_layer, K_list=(1, 2, 3),
        cond_base_suffix=args.cond_base_suffix)
    print(f"[joint] N={len(meta)}  d_prior={h_query.shape[1]}  "
          f"d_cond={cond_pk_K[1].shape[1]}", flush=True)

    split_manifest = json.loads(Path(args.split_file).read_text())
    indices = build_split_indices(meta, split_manifest)
    print(f"[joint] split: train={len(indices['train'])}  "
          f"val={len(indices['val'])}  test={len(indices['test'])}", flush=True)

    y_pk = np.array([1 if m["pk_correct_alias"] else 0 for m in meta], dtype=np.int64)
    y_ck = np.array([1 if m["ck_correct_alias"] else 0 for m in meta], dtype=np.int64)

    test_qids = [meta[i]["qid"] for i in indices["test"]]
    behavior_meta = load_behavior_meta(args.backbone, test_qids, args.datasets)
    test_meta: list[dict] = []
    keep_mask: list[bool] = []
    for i in indices["test"]:
        qid = meta[i]["qid"]
        if qid in behavior_meta:
            test_meta.append(behavior_meta[qid])
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    keep_mask = np.array(keep_mask, dtype=bool)
    print(f"[joint] kept {keep_mask.sum()}/{len(indices['test'])} test rows "
          f"with full behaviour", flush=True)

    configs = [
        ("K1", cond_pk_K[1], cond_ck_K[1]),
        ("mean3", agg_mean(cond_pk_K), agg_mean(cond_ck_K)),
        ("max3", agg_max(cond_pk_K), agg_max(cond_ck_K)),
    ]

    out = {
        "backbone": args.backbone,
        "cond_layer": args.cond_layer,
        "split_sizes": {k: int(len(v)) for k, v in indices.items()},
        "n_test_full_behavior": int(keep_mask.sum()),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": {},
    }

    for cfg_name, cond_pk_X, cond_ck_X in configs:
        print(f"\n[joint] === config: {cfg_name} ===", flush=True)
        t0 = time.time()
        pk_te, ck_te, pk_val, ck_val = fit_pk_ck_concat_mlp(
            h_query, cond_pk_X, cond_ck_X, indices, y_pk, y_ck,
            device=device, seed=args.seed)
        print(f"  fit  pk_val={pk_val:.4f}  ck_val={ck_val:.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        pk_proba = pk_te[keep_mask]
        ck_proba = ck_te[keep_mask]

        overall = joint_cell_metrics(test_meta, pk_proba, ck_proba)
        overall.pop("_cell_counts", None)

        by_ds: dict[str, list[int]] = {}
        for i, m in enumerate(test_meta):
            by_ds.setdefault(m["dataset"], []).append(i)
        per_ds = {}
        for ds, idx in by_ds.items():
            m_ds = [test_meta[i] for i in idx]
            r = joint_cell_metrics(m_ds, pk_proba[idx], ck_proba[idx])
            r.pop("_cell_counts", None)
            per_ds[ds] = r

        out["configs"][cfg_name] = {
            "pk_val_auroc": float(pk_val),
            "ck_val_auroc": float(ck_val),
            "joint_metrics": overall,
            "per_dataset": per_ds,
        }

        print(f"  Acc={overall.get('A1_e2e_acc', overall.get('Acc')):.4f}  "
              f"summary written for {cfg_name}", flush=True)

    out_path = Path(args.out) if args.out else (
        Path("logs") / f"{args.backbone}_joint_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[joint] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
