"""Evaluator layer: ``SolveTrace`` -> ``ScoreRecord``.

Anytime + quality metrics (primal integral, optimality gap), right-censoring
with a clearly-labeled PAR10 auxiliary, and a BO scalarization hook.
"""

from __future__ import annotations

from .censoring import PAR_FACTOR, is_censored, par10, runtime
from .evaluator import evaluate, scalarize
from .metrics import (
    final_gap,
    gap_series,
    is_feasible,
    is_optimal,
    normalized_gap,
    objective,
    primal_dual_gap_integral,
    primal_integral,
)

__all__ = [
    "PAR_FACTOR",
    "evaluate",
    "final_gap",
    "gap_series",
    "is_censored",
    "is_feasible",
    "is_optimal",
    "normalized_gap",
    "objective",
    "par10",
    "primal_dual_gap_integral",
    "primal_integral",
    "runtime",
    "scalarize",
]
