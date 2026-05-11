"""Shared helpers for the SABER probe: data loading, group-aware splits, and
LogReg / MLP head trainers.

Filesystem conventions used throughout this package:

    data/datasets/{dataset}.jsonl                              raw rows
    data/evaluation/{backbone}/behavior_{dataset}.jsonl         PK/CK labels
    artifacts/{backbone}/hidden/{dataset}.hidden.npy            h(query, CK)
    artifacts/{backbone}/hidden/{dataset}.hidden_nock.npy       h(query)
    artifacts/{backbone}/hidden/{dataset}.meta.jsonl            qid alignment
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from saber.config import DATA_ROOT


ART_BASE = DATA_ROOT / "artifacts"
EVAL_BASE = DATA_ROOT / "evaluation"
DATASET_DIR = DATA_ROOT / "datasets"


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dataset_with_hidden(
    backbone: str, dataset: str, *, variant: str = "hidden",
) -> tuple[np.ndarray, list[dict]]:
    """Join hidden.npy + meta.jsonl + behavior + dataset rows by qid.

    `variant`: "hidden" = h(query+CK); "hidden_nock" = h(query) only.

    Returns (X, rows) where each row carries qid, dataset, question,
    ground_truth, ck_relation, pk_correct/ck_correct (alias-based),
    pk_answer/ck_answer, cell, and a 2×2 joint_label int.
    """
    art_dir = ART_BASE / backbone / "hidden"
    hs = np.load(art_dir / f"{dataset}.{variant}.npy")
    hs_meta = load_jsonl(art_dir / f"{dataset}.meta.jsonl")
    behavior = {b["qid"]: b for b in load_jsonl(EVAL_BASE / backbone / f"behavior_{dataset}.jsonl")}
    ds_rows = {r["qid"]: r for r in load_jsonl(DATASET_DIR / f"{dataset}.jsonl")}
    assert len(hs) == len(hs_meta), f"{backbone}/{dataset}: hs/meta length mismatch"

    rows: list[dict] = []
    keep: list[int] = []
    for i, m in enumerate(hs_meta):
        qid = m["qid"]
        bh = behavior.get(qid)
        ds_row = ds_rows.get(qid)
        if bh is None or ds_row is None:
            continue
        pk = bool(bh["pk_correct_alias"])
        ck = bool(bh["ck_correct_alias"])
        rows.append({
            "qid": qid,
            "dataset": dataset,
            "question": ds_row["question"],
            "ground_truth": ds_row.get("ground_truth", []),
            "ck_relation": ds_row.get("ck_relation"),
            "pk_correct": pk,
            "ck_correct": ck,
            "pk_answer": bh.get("pk_answer", ""),
            "ck_answer": bh.get("ck_answer", ""),
            "cell": f"C{int(pk)}{int(ck)}",
            "joint_label": int(pk) * 2 + int(ck),
        })
        keep.append(i)
    return hs[keep].astype(np.float32), rows


def load_pool(
    backbone: str, datasets: list[str], *, variant: str = "hidden",
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    Xs: list[np.ndarray] = []
    metas: list[dict] = []
    for ds in datasets:
        X, m = load_dataset_with_hidden(backbone, ds, variant=variant)
        Xs.append(X)
        metas.extend(m)
        if verbose:
            print(f"  [{ds}] n={len(m)}", flush=True)
    return np.concatenate(Xs, axis=0), metas


# ─── Frozen split manifest (committed to git for cross-experiment comparability) ──

REPO_ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = REPO_ROOT / "splits"


def load_split(name: str) -> dict:
    """Load a frozen split manifest from `splits/{name}.json`.

    The manifest is generated once by `python -m saber.data.build_split` and
    committed to git so all experiments using Pool 2 share the same
    train/val/test partition by qid.
    """
    p = SPLIT_DIR / f"{name}.json"
    return json.loads(p.read_text())


def apply_split(meta: list[dict], manifest: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map rows in `meta` to (train_idx, val_idx, test_idx) using `manifest`.

    Each row is keyed by (dataset, qid). Rows whose dataset is not in the
    manifest, or whose qid is in none of the train/val/test lists, are
    dropped silently — caller verifies counts.
    """
    qid_part: dict[tuple[str, str], str] = {}
    for ds, parts in manifest["partition"].items():
        for p in ("train", "val", "test"):
            for qid in parts[p]:
                qid_part[(ds, qid)] = p

    train, val, test = [], [], []
    for i, m in enumerate(meta):
        p = qid_part.get((m["dataset"], m["qid"]))
        if p == "train":
            train.append(i)
        elif p == "val":
            val.append(i)
        elif p == "test":
            test.append(i)
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


# ─── Head trainers (LogReg + MLP, val-AUROC selection) ────────────────────────

def train_logreg(
    X_tr, y_tr, X_va, y_va, *, head_name: str = "",
    Cs: tuple[float, ...] = (0.01, 0.1), seed: int = 42,
) -> dict:
    """LogReg sweep over C; pick val-AUROC best."""
    trace, best = [], {"auroc_val": -1.0, "clf": None}
    for C in Cs:
        t0 = time.time()
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=seed)
        clf.fit(X_tr, y_tr)
        auroc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
        rec = {"C": C, "class_weight": None, "auroc_val": auroc,
               "elapsed_s": round(time.time() - t0, 1)}
        trace.append(rec)
        print(f"    [logreg-{head_name}] C={C} val_auroc={auroc:.4f} ({rec['elapsed_s']:.0f}s)",
              flush=True)
        if auroc > best["auroc_val"]:
            best = {**rec, "clf": clf}
    return {"trace": trace, "best": best}


def train_mlp(
    X_tr, y_tr, X_va, y_va, *, head_name: str = "",
    grids: tuple[tuple[int, ...], ...] = ((256,), (256, 64)),
    alpha: float = 0.01, seed: int = 42,
) -> dict:
    """MLP sweep over hidden-layer shape; pick val-AUROC best."""
    trace, best = [], {"auroc_val": -1.0, "clf": None}
    for hidden in grids:
        t0 = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
            early_stopping=True, validation_fraction=0.1,
            max_iter=200, random_state=seed,
        )
        clf.fit(X_tr, y_tr)
        auroc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
        rec = {"hidden_layer_sizes": list(hidden), "alpha": alpha, "auroc_val": auroc,
               "elapsed_s": round(time.time() - t0, 1)}
        trace.append(rec)
        print(f"    [mlp-{head_name}] hidden={hidden} val_auroc={auroc:.4f} "
              f"({rec['elapsed_s']:.0f}s)", flush=True)
        if auroc > best["auroc_val"]:
            best = {**rec, "clf": clf}
    return {"trace": trace, "best": best}


HEAD_TRAINERS = {"logreg": train_logreg, "mlp": train_mlp}


# ─── Head-level metrics (AUROC / F1 / etc.) ───────────────────────────────────

def head_metrics(y: np.ndarray, proba: np.ndarray, *, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def per_dataset_auroc(meta: list[dict], y: np.ndarray, proba: np.ndarray) -> dict:
    by_ds: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        by_ds.setdefault(m["dataset"], []).append(i)
    out: dict = {}
    for ds, idx in by_ds.items():
        y_ds = y[idx]
        if len(set(y_ds.tolist())) < 2:
            out[ds] = {"auroc": None, "n": len(idx)}
        else:
            out[ds] = {"auroc": float(roc_auc_score(y_ds, proba[idx])), "n": len(idx)}
    return out
