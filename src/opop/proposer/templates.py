"""Whitelisted valid-inequality templates → class-B deltas for the proposer.

The Phase-1 proposer NEVER invents cuts. It may only emit the valid-inequality
candidates the Analyzer already flagged (cover / clique cuts; task 10), each
wrapped as a class-B :class:`Delta` via
:func:`opop.model.ir.make_add_constraint_delta`. Validity is *not* asserted here
— every emitted cut is still a CANDIDATE that the verification gate (task 11)
must certify (class-B: cuts fractional LP points, removes no feasible integer
incumbent) before evaluation.

Restricting the source to ``report.candidate_cuts`` is the whitelist: a cut the
Analyzer did not flag can never enter the proposal stream.
"""

from __future__ import annotations

from opop.analyzer.report import AnalysisReport
from opop.model.ir import make_add_constraint_delta
from opop.model.state import Delta

__all__ = ["cut_deltas_from_report"]


def cut_deltas_from_report(report: AnalysisReport) -> list[Delta]:
    """Return one class-B :class:`Delta` per analyzer-flagged candidate cut.

    Each :class:`opop.model.ir.LinearConstraint` in ``report.candidate_cuts`` is
    decomposed into ``(name, coeffs, sense, rhs)`` and wrapped by
    :func:`opop.model.ir.make_add_constraint_delta` (declared class B). The
    ``Delta.target`` carries a human-readable rationale. Order follows
    ``report.candidate_cuts`` so the cut section of the candidate pool is
    deterministic.
    """
    deltas: list[Delta] = []
    for cut in report.candidate_cuts:
        support = "+".join(sorted(cut.coeffs))
        rationale = (
            f"valid inequality '{cut.name}' ({support} {cut.sense.value} {cut.rhs:g}); "
            "analyzer-flagged class-B candidate"
        )
        deltas.append(
            make_add_constraint_delta(
                cut.name,
                cut.coeffs,
                cut.sense,
                cut.rhs,
                target=rationale,
            )
        )
    return deltas
