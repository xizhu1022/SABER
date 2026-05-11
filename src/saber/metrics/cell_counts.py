"""Shared cell-count → routing metrics (A1-A4, B1, B3, C3).

Pure functions, no I/O. The 4-cell framework partitions samples by oracle
(pk_correct, ck_correct):

    C00 = (0, 0)    abstain     (neither path right)
    C01 = (0, 1)    use_CK      (only CK right)
    C10 = (1, 0)    use_PK      (only PK right)
    C11 = (1, 1)    either      (both right)

Each "cell" tracks how many rows landed there + outcomes after some routing
mechanism produces a final answer. Used by `baselines/evaluate.py` (where
the routing is whatever the baseline method does) and probe experiments
(where the routing follows a 2-head joint cell prediction).

Definitions (baselines.md §11.2 user-selected core):
    A1 E2E Acc       = correct_total / N
    A2 Acc_t         = correct_(C01∪C11) / N_(C01∪C11)
    A3 Acc_f         = correct_(C00∪C10) / N_(C00∪C10)
    A4 SF            = (A2 + A3) / 2
    B1[C]            = correct_C / N_C        ∀ C ∈ {C00,C01,C10,C11}
    B3[C00]          = pred_C00 ∧ oracle=C00 / N_C00         (abstain coverage)
    C3 MR            = pk_aligned / (pk_aligned + ck_aligned)

B3 requires a routing model that emits a predicted cell; it is None for
plain baselines (no cell prediction).
"""
from __future__ import annotations

from saber.data.alias_match import alias_match


CELLS: tuple[str, ...] = ("C00", "C01", "C10", "C11")
COUNT_KEYS: tuple[str, ...] = (
    "N", "correct", "pk_only", "ck_only", "either", "abstain",
    "pred_C00", "pred_C01", "pred_C10", "pred_C11",
)


def cell_label(pk: bool, ck: bool) -> str:
    return f"C{int(bool(pk))}{int(bool(ck))}"


def empty_cells() -> dict:
    return {c: {k: 0 for k in COUNT_KEYS} for c in CELLS}


def add_cells(a: dict, b: dict) -> dict:
    """Element-wise sum two cell-count dicts."""
    out = empty_cells()
    for c in CELLS:
        for k in COUNT_KEYS:
            out[c][k] = a.get(c, {}).get(k, 0) + b.get(c, {}).get(k, 0)
    return out


def tally(
    cells: dict,
    *,
    oracle_cell: str,
    answer: str,
    ground_truth: list,
    pk_answer: str,
    ck_answer: str,
    predicted_cell: str | None = None,
) -> None:
    """Accumulate one row into `cells` (in place).

    `predicted_cell` is the routing model's joint output (e.g. "C10"); pass
    None for plain baselines that don't predict a cell. When supplied, it
    populates pred_* counters used by B3.
    """
    if oracle_cell not in cells:
        return
    a = (answer or "").strip()
    p = (pk_answer or "").strip()
    ck = (ck_answer or "").strip()
    a_eq_g = alias_match(a, ground_truth) if a else False
    a_eq_p = alias_match(a, [p]) if a and p else False
    a_eq_c = alias_match(a, [ck]) if a and ck else False
    cells[oracle_cell]["N"] += 1
    if a_eq_g:
        cells[oracle_cell]["correct"] += 1
    if a_eq_p and a_eq_c:
        cells[oracle_cell]["either"] += 1
    elif a_eq_p:
        cells[oracle_cell]["pk_only"] += 1
    elif a_eq_c:
        cells[oracle_cell]["ck_only"] += 1
    else:
        cells[oracle_cell]["abstain"] += 1
    if predicted_cell in CELLS:
        cells[oracle_cell][f"pred_{predicted_cell}"] += 1


def safe_div(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def compute_metrics(cells: dict) -> dict:
    """A1-A4, B1, B3[C00], C3 from per-cell tallies.

    B3[C00] is None when no predicted-cell tally was recorded (plain baselines).
    """
    n_total = sum(cells[c]["N"] for c in CELLS)
    cor_total = sum(cells[c]["correct"] for c in CELLS)
    n_t = cells["C01"]["N"] + cells["C11"]["N"]
    cor_t = cells["C01"]["correct"] + cells["C11"]["correct"]
    n_f = cells["C00"]["N"] + cells["C10"]["N"]
    cor_f = cells["C00"]["correct"] + cells["C10"]["correct"]
    pk_aligned = sum(cells[c]["pk_only"] + cells[c]["either"] for c in CELLS)
    ck_aligned = sum(cells[c]["ck_only"] + cells[c]["either"] for c in CELLS)
    pred_total = sum(cells[c][f"pred_{p}"] for c in CELLS for p in CELLS)

    A1 = safe_div(cor_total, n_total)
    A2 = safe_div(cor_t, n_t)
    A3 = safe_div(cor_f, n_f)
    A4 = None if A2 is None or A3 is None else round((A2 + A3) / 2, 4)
    B1 = {c: safe_div(cells[c]["correct"], cells[c]["N"]) for c in CELLS}
    B3_C00 = safe_div(cells["C00"]["pred_C00"], cells["C00"]["N"]) if pred_total > 0 else None
    C3 = safe_div(pk_aligned, pk_aligned + ck_aligned)
    # New "macro" metrics (cleaner than A2/A3 — exclude trivial C11 and hopeless C00):
    #   A2_supp   = B1[C01]            (correct when only CK is right)
    #   A3_resist = B1[C10]            (correct when only PK is right; tests robustness to misleading context)
    #   A4_macro  = (A2_supp + A3_resist) / 2
    A2_supp = B1["C01"]
    A3_resist = B1["C10"]
    A4_macro = None if A2_supp is None or A3_resist is None else round((A2_supp + A3_resist) / 2, 4)
    return {
        "N": n_total,
        "A1_e2e_acc": A1,
        "A2_acc_t": A2,
        "A3_acc_f": A3,
        "A4_sf": A4,
        "A2_supp": A2_supp,
        "A3_resist": A3_resist,
        "A4_macro": A4_macro,
        "B1_per_cell_acc": B1,
        "B3_C00_abstain_coverage": B3_C00,
        "C3_mr": C3,
        "_cell_counts": cells,
    }


def routed_answer(predicted_cell: str, pk_answer: str, ck_answer: str,
                  abstain_str: str = "") -> str:
    """4-way routing decision: C10|C11→PK, C01→CK, C00→abstain."""
    if predicted_cell in ("C10", "C11"):
        return pk_answer
    if predicted_cell == "C01":
        return ck_answer
    return abstain_str
