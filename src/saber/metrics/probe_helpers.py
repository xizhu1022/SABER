"""Helpers used by the SABER joint-cell metrics evaluator.

Bundles the split resolver, behaviour-meta loader, concat-MLP head trainer,
multi-trace condition loader, and aggregation primitives that the
joint-metrics script needs. Kept here so the metrics module is self-contained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from saber import metrics as cell_metrics
from saber.config import DATA_ROOT, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Split resolver
# ---------------------------------------------------------------------------

def build_split_indices(meta: list[dict], split_manifest: dict) -> dict:
    """Resolve a partition manifest into per-split row indices."""
    qid_to_i = {m["qid"]: i for i, m in enumerate(meta)}
    qid_to_split: dict[str, str] = {}
    partition = split_manifest.get("partition", {})

    for ds, parts in partition.items():
        for split_name in ("train", "val", "test"):
            for qid_or_pair in parts.get(split_name, []):
                if isinstance(qid_or_pair, list):
                    for qid in qid_or_pair:
                        qid_to_split[qid] = split_name
                else:
                    qid_to_split[qid_or_pair] = split_name

    # Datasets not in the manifest fall back to a deterministic 80/10/10 by
    # sorted qid order.
    held_out: dict[str, list[str]] = {}
    for m in meta:
        if m["qid"] not in qid_to_split and m["dataset"] not in partition:
            held_out.setdefault(m["dataset"], []).append(m["qid"])
    for ds, qids in held_out.items():
        qids = sorted(qids)
        n = len(qids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        for i, qid in enumerate(qids):
            if i < n_train:
                qid_to_split[qid] = "train"
            elif i < n_train + n_val:
                qid_to_split[qid] = "val"
            else:
                qid_to_split[qid] = "test"

    out = {"train": [], "val": [], "test": []}
    for m in meta:
        s = qid_to_split.get(m["qid"])
        if s in out:
            out[s].append(qid_to_i[m["qid"]])
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Behaviour meta loader
# ---------------------------------------------------------------------------

def load_behavior_meta(backbone: str, manifest_qids: list[str],
                       datasets: list[str]) -> dict[str, dict]:
    """Build ``{qid: {pk_answer, ck_answer, ground_truth, cell, ...}}`` from
    dataset + behaviour files, restricted to ``manifest_qids``."""
    out: dict[str, dict] = {}
    requested = set(manifest_qids)
    for ds in datasets:
        ds_path = DATA_ROOT / "datasets" / f"{ds}.jsonl"
        beh_path = DATA_ROOT / "evaluation" / backbone / f"behavior_{ds}.jsonl"
        if not ds_path.exists() or not beh_path.exists():
            continue
        ds_by_qid = {}
        with ds_path.open() as f:
            for line in f:
                r = json.loads(line)
                ds_by_qid[r["qid"]] = r
        with beh_path.open() as f:
            for line in f:
                b = json.loads(line)
                qid = b["qid"]
                if qid not in ds_by_qid:
                    continue
                if requested and qid not in requested:
                    continue
                ds_row = ds_by_qid[qid]
                pk = bool(b.get("pk_correct_alias"))
                ck = bool(b.get("ck_correct_alias"))
                gt = ds_row.get("ground_truth") or []
                if isinstance(gt, str):
                    gt = [gt]
                out[qid] = {
                    "qid": qid, "dataset": ds,
                    "pk_answer": b.get("pk_answer", ""),
                    "ck_answer": b.get("ck_answer", ""),
                    "ground_truth": gt,
                    "cell": f"C{int(pk)}{int(ck)}",
                    "pk_correct_alias": pk, "ck_correct_alias": ck,
                }
    return out


# ---------------------------------------------------------------------------
# 4-cell joint metrics on probe output
# ---------------------------------------------------------------------------

def joint_cell_metrics(meta_subset, pk_proba, ck_proba, *, threshold: float = 0.5):
    pk_pred = (pk_proba >= threshold).astype(int)
    ck_pred = (ck_proba >= threshold).astype(int)
    cells = cell_metrics.empty_cells()
    for i, m in enumerate(meta_subset):
        predicted = f"C{int(pk_pred[i])}{int(ck_pred[i])}"
        answer = cell_metrics.routed_answer(
            predicted, m["pk_answer"], m["ck_answer"])
        cell_metrics.tally(
            cells,
            oracle_cell=m["cell"],
            answer=answer,
            ground_truth=m["ground_truth"] or [],
            pk_answer=m["pk_answer"], ck_answer=m["ck_answer"],
            predicted_cell=predicted,
        )
    return cell_metrics.compute_metrics(cells)


# ---------------------------------------------------------------------------
# MLP head with early stopping on val AUROC
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float = 0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp_torch(
    X_tr, y_tr, X_va, y_va, *, hidden, device, lr=1e-3, weight_decay=1e-4,
    batch_size=256, max_epochs=80, patience=10, dropout=0.2, seed=42,
):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr).float().to(device)
    Xv = torch.from_numpy(X_va).float().to(device)

    model = _MLP(X_tr.shape[1], hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    n = len(Xt)
    best_auc, best_state, since_best = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            logit = model(Xt[idx])
            bce(logit, yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_logit = model(Xv).cpu().numpy()
        v_proba = 1.0 / (1.0 + np.exp(-v_logit))
        auc = roc_auc_score(y_va, v_proba)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return model, float(best_auc), epoch + 1


def mlp_proba(model, X, device, batch_size: int = 512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).float().to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Conditional-trace loader and multi-trace aggregators
# ---------------------------------------------------------------------------

def _load_cond_layer(cond_dir: Path, ds: str, side: str, layer_idx: int):
    """Load ``trace_mp[:, layer_idx - 1, :]`` for one ``(dataset, side)``,
    combining the canonical ``.npz`` plus any per-checkpoint shards that
    have not yet been merged. Output rows are aligned with ``meta.jsonl``.
    """
    canonical = cond_dir / f"{ds}.cond_{side}.npz"
    meta_path = cond_dir / f"{ds}.meta.jsonl"

    canon_arr = None
    if canonical.exists():
        z = np.load(canonical)
        canon_arr = z["trace_mp"][:, layer_idx - 1, :].astype(np.float32)

    shards = sorted(cond_dir.glob(f"{ds}.cond_{side}.shard*.npz"))
    if not shards:
        return canon_arr

    n_canon = canon_arr.shape[0] if canon_arr is not None else 0
    new_qids: list = []
    with meta_path.open() as f:
        for i, line in enumerate(f):
            if i < n_canon:
                continue
            new_qids.append(json.loads(line)["qid"])
    new_idx = {q: i for i, q in enumerate(new_qids)}

    if not new_qids:
        return canon_arr

    sample = np.load(shards[0])["trace_mp"]
    hidden = sample.shape[2]
    new_arr = np.zeros((len(new_qids), hidden), dtype=np.float32)
    filled = [False] * len(new_qids)
    for sp in shards:
        z = np.load(sp)
        if "qids" not in z.files:
            continue
        shard_qids = z["qids"]
        shard_trace = z["trace_mp"][:, layer_idx - 1, :]
        for i, q in enumerate(shard_qids):
            q = str(q)
            idx = new_idx.get(q)
            if idx is None or filled[idx]:
                continue
            new_arr[idx] = shard_trace[i].astype(np.float32)
            filled[idx] = True

    if canon_arr is None:
        return new_arr
    return np.concatenate([canon_arr, new_arr], axis=0)


def load_aligned_K(backbone: str, datasets: list[str], cond_layer_idx: int,
                   K_list=(1, 2, 3), cond_base_suffix: str = ""):
    """Load query-only prior + K-aware PK / CK conditional reprs, aligned by
    qid across K. Returns ``(h_query, cond_pk_K, cond_ck_K, meta_rows)``.
    """
    prior_dir = ARTIFACTS_DIR / backbone / "hidden"
    cond_dir = ARTIFACTS_DIR / backbone / f"cond{cond_base_suffix}"

    h_list = []
    p_lists = {K: [] for K in K_list}
    c_lists = {K: [] for K in K_list}
    m_list: list[dict] = []

    for ds in datasets:
        prior_meta_p = prior_dir / f"{ds}.meta.jsonl"
        if not prior_meta_p.exists():
            continue
        h_full = np.load(prior_dir / f"{ds}.hidden_nock.npy")
        with prior_meta_p.open() as f:
            prior_meta = [json.loads(l) for l in f]
        prior_idx = {m["qid"]: i for i, m in enumerate(prior_meta)}

        cond_K_pk: dict[int, np.ndarray] = {}
        cond_K_ck: dict[int, np.ndarray] = {}
        cond_meta_K: dict[int, list[dict]] = {}
        skip_ds = False
        for K in K_list:
            prefix = f"{ds}.K{K}"
            ck_meta_p = cond_dir / f"{prefix}.meta.jsonl"
            if not ck_meta_p.exists():
                skip_ds = True
                break
            cond_K_pk[K] = _load_cond_layer(cond_dir, prefix, "pk", cond_layer_idx)
            cond_K_ck[K] = _load_cond_layer(cond_dir, prefix, "ck", cond_layer_idx)
            with ck_meta_p.open() as f:
                cond_meta_K[K] = [json.loads(l) for l in f]
        if skip_ds:
            continue

        anchor_meta = cond_meta_K[K_list[0]]
        per_K_idx = {
            K: {m["qid"]: j for j, m in enumerate(meta_K)}
            for K, meta_K in cond_meta_K.items()
        }
        for m in anchor_meta:
            qid = m["qid"]
            if qid not in prior_idx:
                continue
            if not all(qid in per_K_idx[K] for K in K_list):
                continue
            i_prior = prior_idx[qid]
            h_list.append(h_full[i_prior])
            for K in K_list:
                jK = per_K_idx[K][qid]
                p_lists[K].append(cond_K_pk[K][jK])
                c_lists[K].append(cond_K_ck[K][jK])
            m_list.append(m)

    if not h_list:
        raise RuntimeError("No aligned rows found; check artifacts directory")
    h_query = np.stack(h_list)
    cond_pk = {K: np.stack(p_lists[K]) for K in K_list}
    cond_ck = {K: np.stack(c_lists[K]) for K in K_list}
    return h_query, cond_pk, cond_ck, m_list


def agg_mean(cond_K_dict: dict) -> np.ndarray:
    """Element-wise mean over the K reprs."""
    return np.mean(np.stack(list(cond_K_dict.values()), axis=0), axis=0)


def agg_max(cond_K_dict: dict) -> np.ndarray:
    """Element-wise max over the K reprs."""
    return np.max(np.stack(list(cond_K_dict.values()), axis=0), axis=0)


def agg_concat(cond_K_dict: dict) -> np.ndarray:
    """Concatenate the K reprs along the feature axis."""
    return np.concatenate(list(cond_K_dict.values()), axis=1)
