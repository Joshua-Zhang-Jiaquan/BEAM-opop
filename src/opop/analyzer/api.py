"""Top-level analyzer entry point: ``analyze(ir) -> AnalysisReport``.

Runs all Phase-1 deterministic checks over a :class:`opop.model.ir.MILP`
and assembles a single structured :class:`AnalysisReport`:

1. consistency    — dimension / units / index annotations (:mod:`consistency`).
2. redundancy     — duplicates, dominance, trivial infeasibility, conflicts
   (:mod:`redundancy`).
3. relaxation     — LP-relaxation objective, integrality-gap estimate, and the
   fractional-variable pattern via SCIP (:mod:`relaxation`).
4. valid inequalities — cover / clique candidate cuts (:mod:`valid_inequalities`).
5. decomposability — block-diagonal / DW / Benders structure from the bipartite
   model graph (:mod:`decompose`); the proposer consumes this to drive the GCG
   decomposition kernel (task 24).
6. decomposition readiness — Benders / Dantzig--Wolfe readiness signal layered on
   the structural verdict plus the integer/continuous staging (:mod:`benders_dw`).
7. symmetry        — variable orbits + column dominance via graph automorphism
   (:mod:`symmetry`).
8. lagrangian      — Lagrangian dual-bound estimate over the coupling constraints
   (:mod:`lagrangian`); off by default as it is the only solver-backed section.

The task-26 expansions (readiness, symmetry, lagrangian) sit behind ``with_*``
flags. Readiness and symmetry are pure and run by default; the Lagrangian bound
needs SCIP and is opt-in. The input IR is never mutated.
"""

from __future__ import annotations

from opop.analyzer.benders_dw import classify_readiness
from opop.analyzer.consistency import check_consistency
from opop.analyzer.decompose import detect_decomposition
from opop.analyzer.lagrangian import estimate_lagrangian_bound
from opop.analyzer.redundancy import detect_redundancy
from opop.analyzer.relaxation import analyze_relaxation
from opop.analyzer.report import AnalysisReport, Flag, RelaxationMetrics
from opop.analyzer.symmetry import detect_symmetry
from opop.analyzer.valid_inequalities import generate_valid_inequalities
from opop.model.ir import MILP

__all__ = ["analyze"]


def analyze(
    ir: MILP,
    *,
    ip_bound: float | None = None,
    estimate_ip_bound: bool = True,
    solve_relaxation: bool = True,
    max_cuts: int = 64,
    with_readiness: bool = True,
    with_symmetry: bool = True,
    with_lagrangian: bool = False,
) -> AnalysisReport:
    """Analyze ``ir`` and return a structured :class:`AnalysisReport`.

    Args:
        ir: The MILP to analyze.
        ip_bound: A known integer objective bound for the integrality gap. When
            ``None``, the relaxation step consults ``ir.metadata`` and may
            estimate (see ``estimate_ip_bound``).
        estimate_ip_bound: Let the relaxation step estimate the IP bound by a
            bounded integer solve when none is supplied.
        solve_relaxation: Solve the LP relaxation via SCIP. Set ``False`` to
            skip the solver entirely (``relaxation_metrics`` then reports
            ``lp_status="SKIPPED"``).
        max_cuts: Cap on candidate cuts per family.
        with_readiness: Classify Benders / Dantzig--Wolfe readiness (pure).
        with_symmetry: Detect variable orbits + column dominance (pure).
        with_lagrangian: Estimate the Lagrangian dual bound over the detected
            coupling constraints (needs SCIP; off by default).

    Returns:
        An :class:`AnalysisReport` with structural flags, relaxation metrics,
        candidate cuts, the decomposability verdict, and — behind their flags —
        the readiness, symmetry, and Lagrangian sections. Decomposability,
        readiness, and symmetry are pure and solver-free.
    """
    flags: list[Flag] = []
    flags.extend(check_consistency(ir))
    flags.extend(detect_redundancy(ir))

    if solve_relaxation:
        relaxation = analyze_relaxation(
            ir, ip_bound=ip_bound, estimate_ip_bound=estimate_ip_bound
        )
    else:
        relaxation = RelaxationMetrics(lp_status="SKIPPED", ip_bound=ip_bound)

    candidate_cuts = generate_valid_inequalities(ir, max_cuts=max_cuts)
    decomposition = detect_decomposition(ir)
    readiness = classify_readiness(ir, decomposition=decomposition) if with_readiness else None
    symmetry = detect_symmetry(ir) if with_symmetry else None
    lagrangian = (
        estimate_lagrangian_bound(ir, coupling=decomposition.linking_constraints)
        if with_lagrangian
        else None
    )

    return AnalysisReport(
        flags=tuple(flags),
        relaxation_metrics=relaxation,
        candidate_cuts=tuple(candidate_cuts),
        decomposability=decomposition.decomposability,
        decomposition=decomposition,
        decomposition_readiness=readiness,
        symmetry=symmetry,
        lagrangian=lagrangian,
    )
