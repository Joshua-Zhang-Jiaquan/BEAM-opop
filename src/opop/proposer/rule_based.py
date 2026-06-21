"""Deterministic rule-based proposer (the Phase-1 fallback).

Used whenever no LLM is supplied OR the LLM produces no usable selection (parse
failure, empty, all-hallucinated). It ranks the typed candidate pool with a
fixed, network-free policy so tests are fully reproducible:

* **Cuts first, gap-driven budget.** The analyzer-flagged valid inequalities
  (class-B) directly tighten the LP relaxation, so when the integrality gap is
  large they get more of the ``max_deltas`` budget. At least one cut is always
  emitted when any candidate cut exists (the proposer must "respect analysis
  candidates").
* **A few param variations.** The remaining budget is filled from the curated
  class-C knob deltas (separator frequency, branching, presolving, gap limit),
  in their stable curated order.

The ranker operates on a pre-built ``pool`` (cuts ++ params, in pool order) so
it shares one candidate set with the LLM path; :func:`propose_rule_based` is a
standalone convenience that builds the pool itself.
"""

from __future__ import annotations

from opop.analyzer.report import AnalysisReport
from opop.model.state import Delta, DeltaClass, ProblemState
from opop.proposer.params import curated_param_deltas
from opop.proposer.templates import cut_deltas_from_report

__all__ = ["GAP_PRIORITY", "propose_rule_based", "rank"]

#: Integrality-gap threshold above which cuts receive the larger budget share.
#: A loose relaxation (gap >= 5%) benefits most from added valid inequalities.
GAP_PRIORITY = 0.05


def _cut_budget(gap: float, max_deltas: int, n_cuts: int) -> int:
    """How many cut slots to allocate (>=1 when cuts exist and budget allows)."""
    if gap >= GAP_PRIORITY:
        # Loose relaxation: cuts dominate, but leave room for >=1 param variation.
        budget = max(1, max_deltas - 1)
    else:
        # Tight relaxation: split budget, still guaranteeing >=1 cut.
        budget = max(1, max_deltas // 2)
    return min(budget, n_cuts)


def rank(report: AnalysisReport, pool: list[Delta], *, max_deltas: int = 5) -> list[Delta]:
    """Return a deterministic ranked sub-list of ``pool`` (length <= ``max_deltas``).

    ``pool`` is split into class-B cuts and class-C params by ``declared_class``.
    Cuts are prioritised by the integrality gap (:data:`GAP_PRIORITY`); at least
    one cut is kept whenever cuts exist and ``max_deltas >= 1``. The remaining
    budget is filled with curated param variations in pool order.
    """
    if max_deltas <= 0:
        return []

    cuts = [d for d in pool if d.declared_class is DeltaClass.B]
    params = [d for d in pool if d.declared_class is DeltaClass.C]

    if not cuts:
        return params[:max_deltas]
    if not params:
        return cuts[:max_deltas]

    gap = report.lp_gap or 0.0
    cut_budget = _cut_budget(gap, max_deltas, len(cuts))
    chosen = cuts[:cut_budget]
    chosen.extend(params[: max_deltas - len(chosen)])
    return chosen[:max_deltas]


def propose_rule_based(
    state: ProblemState,
    report: AnalysisReport,
    *,
    max_deltas: int = 5,
) -> list[Delta]:
    """Build the candidate pool from ``report`` and rank it (standalone fallback).

    ``state`` is accepted for interface parity with :func:`opop.proposer.propose`
    (the Phase-1 ranking is driven entirely by ``report``). The pool is
    ``cut_deltas_from_report(report)`` followed by ``curated_param_deltas()``;
    the decomposition stub is intentionally excluded here (it is dormant unless
    ``report.decomposability != "NONE"``, handled in :func:`opop.proposer.propose`).
    """
    _ = state
    pool = cut_deltas_from_report(report) + curated_param_deltas()
    return rank(report, pool, max_deltas=max_deltas)
