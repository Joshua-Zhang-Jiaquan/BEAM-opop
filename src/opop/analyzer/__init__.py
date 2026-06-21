"""Deterministic OR-analysis layer for OPOP (Phase-1).

The analyzer consumes a :class:`opop.model.ir.MILP` and produces a structured
:class:`AnalysisReport` — the framework's key differentiator over modeling-only
agents. It ships these checks behind a single :func:`analyze` entry point:

* :func:`~opop.analyzer.consistency.check_consistency` — dimension / units /
  index annotation consistency.
* :func:`~opop.analyzer.redundancy.detect_redundancy` — duplicate / dominated
  constraints, trivial infeasibility, conflicts.
* :func:`~opop.analyzer.relaxation.analyze_relaxation` — LP-relaxation objective,
  integrality-gap estimate, fractional-variable pattern (SCIP root LP).
* :func:`~opop.analyzer.valid_inequalities.generate_valid_inequalities` — cover /
  clique candidate cuts (CANDIDATES only; certified by the verification gate).
* :func:`~opop.analyzer.decompose.detect_decomposition` — block-diagonal / DW /
  Benders decomposability from the bipartite model graph (task 24).
* :func:`~opop.analyzer.benders_dw.classify_readiness` — Benders / Dantzig--Wolfe
  readiness signal (task 26).
* :func:`~opop.analyzer.symmetry.detect_symmetry` — variable orbits + column
  dominance via graph automorphism (task 26).
* :func:`~opop.analyzer.lagrangian.estimate_lagrangian_bound` — Lagrangian dual
  bound over the coupling constraints (task 26; solver-backed).
"""

from __future__ import annotations

from opop.analyzer.api import analyze
from opop.analyzer.benders_dw import (
    READY_BENDERS,
    READY_BOTH,
    READY_DW,
    READY_NONE,
    DecompositionReadiness,
    classify_readiness,
)
from opop.analyzer.consistency import check_consistency
from opop.analyzer.decompose import (
    DECOMP_BENDERS,
    DECOMP_BLOCK,
    DECOMP_DW,
    DECOMP_NONE,
    DecompositionReport,
    decomposition_delta,
    detect_decomposition,
)
from opop.analyzer.lagrangian import (
    LAGRANGIAN_ANALYZED,
    LAGRANGIAN_NO_COUPLING,
    LAGRANGIAN_UNAVAILABLE,
    LagrangianBound,
    estimate_lagrangian_bound,
)
from opop.analyzer.redundancy import detect_redundancy
from opop.analyzer.relaxation import analyze_relaxation, relaxed_ir
from opop.analyzer.report import (
    CONFLICT,
    DIMENSION_MISMATCH,
    INDEX_ERROR,
    REDUNDANT,
    TRIVIAL_INFEASIBILITY,
    UNITS_MISMATCH,
    AnalysisReport,
    Flag,
    RelaxationMetrics,
)
from opop.analyzer.symmetry import (
    SYMMETRY_ANALYZED,
    SYMMETRY_EMPTY,
    SYMMETRY_SKIPPED,
    SymmetryInfo,
    detect_symmetry,
)
from opop.analyzer.valid_inequalities import (
    generate_clique_cuts,
    generate_cover_cuts,
    generate_valid_inequalities,
)

__all__ = [
    "CONFLICT",
    "DECOMP_BENDERS",
    "DECOMP_BLOCK",
    "DECOMP_DW",
    "DECOMP_NONE",
    "DIMENSION_MISMATCH",
    "INDEX_ERROR",
    "LAGRANGIAN_ANALYZED",
    "LAGRANGIAN_NO_COUPLING",
    "LAGRANGIAN_UNAVAILABLE",
    "READY_BENDERS",
    "READY_BOTH",
    "READY_DW",
    "READY_NONE",
    "REDUNDANT",
    "SYMMETRY_ANALYZED",
    "SYMMETRY_EMPTY",
    "SYMMETRY_SKIPPED",
    "TRIVIAL_INFEASIBILITY",
    "UNITS_MISMATCH",
    "AnalysisReport",
    "DecompositionReadiness",
    "DecompositionReport",
    "Flag",
    "LagrangianBound",
    "RelaxationMetrics",
    "SymmetryInfo",
    "analyze",
    "analyze_relaxation",
    "check_consistency",
    "classify_readiness",
    "decomposition_delta",
    "detect_decomposition",
    "detect_redundancy",
    "detect_symmetry",
    "estimate_lagrangian_bound",
    "generate_clique_cuts",
    "generate_cover_cuts",
    "generate_valid_inequalities",
    "relaxed_ir",
]
