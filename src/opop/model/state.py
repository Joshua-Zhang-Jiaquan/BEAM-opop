"""Core immutable state types for the OPOP agent loop.

Every module depends on these contracts.  Pure data + validation only —
no solver imports, no behaviour, no mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# DeltaClass — verification delta categorisation (HARD gate)
# ---------------------------------------------------------------------------


class DeltaClass(Enum):
    """Verification delta classes per the Verification Strategy.

    A — Equivalent reformulation: preserves feasible integer solutions + objective.
    B — Valid inequality / relaxation strengthening: may cut fractional LP points,
        must NOT remove any feasible integer incumbent.
    C — Heuristic / search-param: search path only; no semantic change.
    D — Risky / non-certified: sandbox only; NEVER enters main evaluation.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"


# ---------------------------------------------------------------------------
# Delta — a single proposed change
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Delta:
    """A single formulation or search delta with verification class.

    Attributes:
        target: Human-readable description of what is being changed.
        before_fragment: Reference (fragment id / hash) to the state BEFORE the change.
        after_fragment: Reference (fragment id / hash) to the state AFTER the change.
        declared_class: The verification class the proposer claims for this delta.
    """

    target: str
    before_fragment: str | None = None
    after_fragment: str | None = None
    declared_class: DeltaClass = DeltaClass.D


# ---------------------------------------------------------------------------
# Phi — design vector for Bayesian optimisation
# ---------------------------------------------------------------------------

# Per-field type tags consumed by the BO encoder to select kernels / encoding.
_PhiFieldType = dict[str, str]  # field_name → {"categorical","ordinal","bool","continuous"}

# Canonical type map: field name → type tag.  Must stay in sync with Phi.__slots__.
_PHI_TYPE_MAP: _PhiFieldType = {
    "m": "categorical",  # formulation_family
    "v": "categorical",  # var_encoding
    "c": "categorical",  # constraint_templates
    "d": "categorical",  # decomposition
    "h": "ordinal",  # heuristics
    "p": "continuous",  # solver_params (dict → flat continuous vector at encode time)
    "s": "ordinal",  # fidelity
    "rho": "continuous",  # risk_thresholds
}


@dataclass(frozen=True, slots=True)
class Phi:
    """Design vector encoding a complete formulation+search configuration.

    Single-letter field names are canonical and map to a type tag consumed
    by the Bayesian optimisation encoder.

    Attributes:
        m: Formulation family (e.g. "standard", "extended", "aggregated").
        v: Variable encoding strategy (e.g. "binary", "integer", "one-hot").
        c: Constraint template set id.
        d: Decomposition strategy (e.g. "none", "benders", "dantzig-wolfe").
        h: Heuristic intensity level (0=none, 1=mild, 2=aggressive).
        p: Solver parameterisation (dict of key→numeric value).
        s: Fidelity level (1=low, 2=medium, 3=high).
        rho: Risk threshold values (dict of key→float in [0,1]).
    """

    m: str = "standard"
    v: str = "binary"
    c: str = "default"
    d: str = "none"
    h: int = 0
    p: dict[str, float] = field(default_factory=dict)
    s: int = 1
    rho: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def field_types() -> _PhiFieldType:
        """Return a copy of the canonical field→type-tag mapping.

        Tags: ``categorical``, ``ordinal``, ``bool``, ``continuous``.
        """
        return dict(_PHI_TYPE_MAP)

    def to_flat_dict(self) -> dict[str, object]:
        """Return all field values as a flat dict with stable key order."""
        result: dict[str, object] = {}
        for f in fields(self):
            result[f.name] = getattr(self, f.name)
        return result


# ---------------------------------------------------------------------------
# SolveTrace — solver trajectory for one run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveTrace:
    """Trajectory of a single solver run.

    All series are recorded in chronological order with timestamps
    in the ``*_times`` arrays (same length as corresponding series).

    Attributes:
        primal_bound_series: Sequence of primal (upper for min) bounds.
        dual_bound_series: Sequence of dual (lower for min) bounds.
        time_series: Wall-clock timestamps (seconds) for each recorded point.
        nodes: Total B&B nodes explored.
        lp_iters: Total simplex/barrier LP iterations.
        cuts: Total cutting planes generated.
        first_feasible_time: Wall-clock seconds until first feasible solution found.
        status: Solver termination status string (e.g. "OPTIMAL", "TIMEOUT").
        censored: ``True`` when run was terminated by limit before optimality proof.
        memory_peak: Peak memory usage in MiB.
        instance_id: Identifier of the instance this trace belongs to.
        solver: Name of the solver backend that produced this trace.
    """

    primal_bound_series: list[float] = field(default_factory=list)
    dual_bound_series: list[float] = field(default_factory=list)
    time_series: list[float] = field(default_factory=list)
    nodes: int = 0
    lp_iters: int = 0
    cuts: int = 0
    first_feasible_time: float = float("nan")
    status: str = "UNKNOWN"
    censored: bool = False
    memory_peak: float = 0.0
    instance_id: str = ""
    solver: str = ""


# ---------------------------------------------------------------------------
# ScoreRecord — evaluated metrics for one or more solve traces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """Multi-metric evaluation vector produced by the Evaluator.

    ``metrics`` is the primary output — a flat dict of named numeric scores.
    ``uncertainty`` provides per-metric standard-deviation estimates (if replays
    are available). ``risks`` is a list of human-readable risk flag descriptions.

    Attributes:
        metrics: Named metric → numeric value (e.g. ``primal_integral``, ``gap``).
        uncertainty: Per-metric uncertainty estimates (stddev), if available.
        risks: Human-readable risk/warning flags (e.g. "censored", "low-confidence").
        instance_id: The instance this score applies to.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] | None = None
    risks: list[str] = field(default_factory=list)
    instance_id: str = ""


# ---------------------------------------------------------------------------
# ProblemState — top-level aggregate state container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProblemState:
    """Aggregate state object tracking the full lifecycle of one instance.

    This is the single source of truth passed through the loop:
    Proposer → Analyzer → Solver → Evaluator → Bayesian Controller.

    All fields are immutable; state transitions create a *new* instance
    via ``dataclasses.replace``.

    Attributes:
        instance_id: Unique identifier for the problem instance.
        task_family: Coarse problem class (e.g. "MILP", "MIQP", "QUBO").
        symbolic_model_ref: Opaque reference to the current symbolic model IR.
        model_graph_ref: Opaque reference to the bipartite var–con graph.
        formulation_history: Ordered log of all deltas applied so far.
        solver_trace_history: All solver runs executed (may be empty).
        posterior_state_ref: Reference to the Bayesian posterior state.
        budget_state: Budget counters (trials, time, tokens, etc.).
        incumbent_solution: Best-known feasible solution mapping var→value.
        incumbent_certificate: Certificate (bound, proof) for the incumbent.
        risk_flags: Active risk/warning flags (e.g. "unstable").
    """

    instance_id: str = ""
    task_family: str = "MILP"
    symbolic_model_ref: str | None = None
    model_graph_ref: str | None = None
    formulation_history: list[Delta] = field(default_factory=list)
    solver_trace_history: list[SolveTrace] = field(default_factory=list)
    posterior_state_ref: str | None = None
    budget_state: dict[str, Any] = field(default_factory=dict)
    incumbent_solution: dict[str, Any] | None = None
    incumbent_certificate: dict[str, Any] | None = None
    risk_flags: list[str] = field(default_factory=list)
