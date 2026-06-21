"""Structured analysis output for the OPOP analyzer (Phase-1).

The analyzer is the framework's deterministic OR-analysis layer. It consumes a
:class:`opop.model.ir.MILP` and emits an :class:`AnalysisReport` capturing:

* ``flags``              — structural issues found (index/dimension/units
  inconsistencies, redundant/dominated constraints, trivial infeasibility,
  conflicts), each a ``{type, message, location}`` record (:class:`Flag`).
* ``relaxation_metrics`` — LP-relaxation statistics (:class:`RelaxationMetrics`):
  LP objective, integrality-gap estimate, and the fractional-variable pattern.
* ``candidate_cuts``     — proposed valid-inequality CANDIDATES (cover / clique).
  These are *candidates only*; validity is certified by the verification gate
  (task 11), never here.
* ``decomposability``    — the structural decomposition verdict string
  (``"NONE"`` / ``"DW"`` / ``"BENDERS"`` / ``"BLOCK"``), kept as a convenience
  shortcut for ``decomposition.decomposability``.
* ``decomposition``      — the full :class:`~opop.analyzer.decompose.DecompositionReport`
  (block count, block variables, linking elements, reasoning), or ``None`` when
  decomposability was not analyzed.
* ``decomposition_readiness`` — the Benders / Dantzig--Wolfe readiness verdict
  (:class:`~opop.analyzer.benders_dw.DecompositionReadiness`), or ``None``.
* ``symmetry``           — variable orbits + column dominance signal
  (:class:`~opop.analyzer.symmetry.SymmetryInfo`), or ``None``.
* ``lagrangian``         — Lagrangian dual-bound estimate
  (:class:`~opop.analyzer.lagrangian.LagrangianBound`), or ``None`` (off by
  default; it is the only section that needs a solver).

All records are immutable frozen dataclasses, consistent with the IR and state
layers, and expose ``to_dict()`` for JSON serialisation / evidence capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opop.analyzer.benders_dw import DecompositionReadiness
from opop.analyzer.decompose import DecompositionReport
from opop.analyzer.lagrangian import LagrangianBound
from opop.analyzer.symmetry import SymmetryInfo
from opop.model.ir import LinearConstraint

__all__ = [
    "CONFLICT",
    "DIMENSION_MISMATCH",
    "INDEX_ERROR",
    "REDUNDANT",
    "TRIVIAL_INFEASIBILITY",
    "UNITS_MISMATCH",
    "AnalysisReport",
    "DecompositionReadiness",
    "DecompositionReport",
    "Flag",
    "LagrangianBound",
    "RelaxationMetrics",
    "SymmetryInfo",
]


# ---------------------------------------------------------------------------
# Flag type constants — the closed vocabulary of structural issue kinds.
# ---------------------------------------------------------------------------
#: A constraint/variable references an index member outside its declared set.
INDEX_ERROR = "index_error"
#: A constraint's term count disagrees with its declared dimension.
DIMENSION_MISMATCH = "dimension_mismatch"
#: A constraint linearly combines variables carrying incompatible units.
UNITS_MISMATCH = "units_mismatch"
#: A constraint is implied by another (dominated or exact / scaled duplicate).
REDUNDANT = "redundant"
#: A single constraint (or variable bound) is infeasible on its own.
TRIVIAL_INFEASIBILITY = "trivial_infeasibility"
#: Two constraints directly contradict each other (no feasible point).
CONFLICT = "conflict"


# ---------------------------------------------------------------------------
# Flag — one structural issue
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Flag:
    """A single structural issue found by the analyzer.

    Attributes:
        type: One of the module-level flag constants (e.g. :data:`INDEX_ERROR`,
            :data:`REDUNDANT`, :data:`CONFLICT`).
        message: Human-readable explanation of the issue.
        location: The constraint or variable name the flag applies to
            (``""`` when the issue is model-global).
    """

    type: str
    message: str
    location: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return the ``{type, message, location}`` mapping."""
        return {"type": self.type, "message": self.message, "location": self.location}


# ---------------------------------------------------------------------------
# RelaxationMetrics — LP-relaxation statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RelaxationMetrics:
    """LP-relaxation statistics for one MILP.

    The LP relaxation drops integrality (every variable becomes continuous over
    its original bounds) and is solved to optimality by SCIP.

    Attributes:
        lp_obj: LP-relaxation optimal objective, or ``None`` when the relaxation
            was not solved to optimality (see ``lp_status``).
        gap: Integrality-gap estimate ``(ip_bound - lp_obj) / |ip_bound|``.
            ``None`` when no IP bound is known and none was estimated. For a
            minimisation problem the IP bound is an upper bound on the LP value,
            so the gap is non-negative; for maximisation the sign flips.
        n_fractional: Number of integer-constrained variables taking a
            fractional value at the LP optimum.
        fractional_vars: Names of those fractional variables (declaration order).
        lp_status: SCIP termination status of the relaxation solve, upper-cased
            (e.g. ``"OPTIMAL"``, ``"INFEASIBLE"``, ``"UNBOUNDED"``,
            ``"UNAVAILABLE"`` when SCIP could not be used).
        ip_bound: The IP objective bound used for ``gap`` (provided or estimated),
            or ``None``.
    """

    lp_obj: float | None = None
    gap: float | None = None
    n_fractional: int = 0
    fractional_vars: tuple[str, ...] = ()
    lp_status: str = "UNKNOWN"
    ip_bound: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of all metric fields."""
        return {
            "lp_obj": self.lp_obj,
            "gap": self.gap,
            "n_fractional": self.n_fractional,
            "fractional_vars": list(self.fractional_vars),
            "lp_status": self.lp_status,
            "ip_bound": self.ip_bound,
        }


# ---------------------------------------------------------------------------
# AnalysisReport — the structured analyzer output
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Structured output of :func:`opop.analyzer.api.analyze`.

    Attributes:
        flags: All structural issues found (:class:`Flag`), in detection order.
        relaxation_metrics: LP-relaxation statistics (:class:`RelaxationMetrics`).
        candidate_cuts: Proposed valid-inequality CANDIDATES
            (:class:`opop.model.ir.LinearConstraint`). Candidates only — validity
            is certified downstream by the verification gate (task 11).
        decomposability: Structural decomposition verdict string (``"NONE"`` /
            ``"DW"`` / ``"BENDERS"`` / ``"BLOCK"``); mirrors
            ``decomposition.decomposability`` when ``decomposition`` is present.
        decomposition: The full :class:`~opop.analyzer.decompose.DecompositionReport`
            (block count, block variables, linking elements, reasoning), or
            ``None`` when decomposability was not analyzed.
        decomposition_readiness: The Benders / Dantzig--Wolfe readiness verdict
            (:class:`~opop.analyzer.benders_dw.DecompositionReadiness`), or
            ``None`` when readiness was not classified.
        symmetry: Variable-orbit and column-dominance signal
            (:class:`~opop.analyzer.symmetry.SymmetryInfo`), or ``None`` when
            symmetry detection was not run.
        lagrangian: Lagrangian dual-bound estimate
            (:class:`~opop.analyzer.lagrangian.LagrangianBound`), or ``None``
            when it was not estimated (it is the only solver-backed section).
    """

    flags: tuple[Flag, ...] = ()
    relaxation_metrics: RelaxationMetrics = field(default_factory=RelaxationMetrics)
    candidate_cuts: tuple[LinearConstraint, ...] = ()
    decomposability: str = "NONE"
    decomposition: DecompositionReport | None = None
    decomposition_readiness: DecompositionReadiness | None = None
    symmetry: SymmetryInfo | None = None
    lagrangian: LagrangianBound | None = None

    # -- convenience views --------------------------------------------------
    @property
    def lp_obj(self) -> float | None:
        """Shortcut to ``relaxation_metrics.lp_obj``."""
        return self.relaxation_metrics.lp_obj

    @property
    def lp_gap(self) -> float | None:
        """Shortcut to ``relaxation_metrics.gap`` (the integrality-gap estimate)."""
        return self.relaxation_metrics.gap

    def flags_by_type(self, flag_type: str) -> tuple[Flag, ...]:
        """Return all flags whose ``type`` equals ``flag_type``."""
        return tuple(f for f in self.flags if f.type == flag_type)

    def locations_by_type(self, flag_type: str) -> list[str]:
        """Return the ``location`` of every flag of ``flag_type`` (in order)."""
        return [f.location for f in self.flags if f.type == flag_type]

    def has_flag(self, flag_type: str) -> bool:
        """Return ``True`` iff at least one flag of ``flag_type`` is present."""
        return any(f.type == flag_type for f in self.flags)

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serialisable mapping of the report."""
        readiness = self.decomposition_readiness
        return {
            "flags": [f.to_dict() for f in self.flags],
            "relaxation_metrics": self.relaxation_metrics.to_dict(),
            "candidate_cuts": [_cut_to_dict(c) for c in self.candidate_cuts],
            "decomposability": self.decomposability,
            "decomposition": self.decomposition.to_dict() if self.decomposition else None,
            "decomposition_readiness": readiness.to_dict() if readiness else None,
            "symmetry": self.symmetry.to_dict() if self.symmetry else None,
            "lagrangian": self.lagrangian.to_dict() if self.lagrangian else None,
        }


def _cut_to_dict(cut: LinearConstraint) -> dict[str, Any]:
    """Serialise a candidate-cut :class:`LinearConstraint` to a plain dict."""
    return {
        "name": cut.name,
        "coeffs": dict(cut.coeffs),
        "sense": cut.sense.value,
        "rhs": cut.rhs,
    }
