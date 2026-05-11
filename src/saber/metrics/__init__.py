"""Cell-count metrics (Acc / CF / KF / MFS) and helpers shared across SABER
and the baseline evaluators.

The public surface re-exports the cell-count primitives so that ``from saber
import metrics`` continues to give callers ``metrics.empty_cells``,
``metrics.tally``, ``metrics.routed_answer``, and ``metrics.compute_metrics``.
"""
from saber.metrics.cell_counts import (
    empty_cells,
    tally,
    routed_answer,
    compute_metrics,
)

__all__ = [
    "empty_cells",
    "tally",
    "routed_answer",
    "compute_metrics",
]
