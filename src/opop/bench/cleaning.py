"""Solver-backed re-verification + cleaning for labeled benchmark items (task 35).

Modeling-agent benchmark datasets ship a *labeled* optimum for each natural-language
problem; those labels are produced by humans/LLMs and are NOT trustworthy by
default. This module is the OptiTrust-style integrity gate: it **re-derives** every
label by actually building the model and solving it, then quarantines any item whose
solver-computed optimum disagrees with the provided label.

The harness is problem-class agnostic. Each labeled :class:`CleaningItem` carries a
plain symbolic IR (:class:`~opop.model.ir.MILP`, optionally with the task-30
quadratic extension or the task-31 nonlinear-terms metadata). Routing is via the
capability registry (:func:`opop.model.adapter.find_adapter`):

* a plain linear MILP (no adapter claims it) is solved directly on the kernel;
* an instance whose adapter declares the chosen solver as a *native* kernel
  (SCIP for MIQP / MIQCP / QUBO / structured MINLP) is solved with
  :meth:`~opop.model.adapter.ProblemClassAdapter.native_solve` — the faithful,
  true-optimum route (no relaxation), which is what re-verification requires;
* otherwise the adapter's exact linearization
  (:meth:`~opop.model.adapter.ProblemClassAdapter.to_milp`) is solved on the
  kernel (e.g. the classic-CO families on CP-SAT / HiGHS).

A solve that fails, is censored by a resource limit, or does not prove optimality
quarantines the item (fail-closed: an unverifiable label is never declared clean).
:func:`verify_and_clean` returns a :class:`CleaningReport` with the partitioned
``clean`` / ``quarantined`` results and a ``to_json`` writer for
``cleaning_report.json``.

This module imports no solver backend at import time: the kernel and the
solver-layer adapters (:mod:`opop.solver.miqp` / :mod:`opop.solver.qubo`) are
imported lazily inside :func:`verify_and_clean`, so ``import opop.bench.cleaning``
stays light (e.g. for registry generation).
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from opop.model.adapter import find_adapter
from opop.model.ir import MILP, ObjSense, VarType
from opop.model.minlp import NONLINEAR_TERMS_KEY
from opop.model.state import Phi

if TYPE_CHECKING:
    from opop.model.state import SolveTrace
    from opop.solver.kernel import SolverKernel

__all__ = [
    "CleaningItem",
    "CleaningReport",
    "CleaningResult",
    "verify_and_clean",
]

#: Per-item verdicts.
STATUS_CLEAN = "clean"
STATUS_QUARANTINED = "quarantined"

#: Deterministic budget knobs (the public API exposes only ``time_limit``).
_MEMORY_LIMIT_MB = 4096
_SEED = 0

#: Solvers this harness can instantiate by name.
_SUPPORTED_SOLVERS: tuple[str, ...] = ("SCIP", "CP-SAT", "HiGHS")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CleaningItem:
    """One labeled benchmark item awaiting solver-backed re-verification.

    Attributes:
        id: Globally unique item id (the registry / report key).
        ir: The symbolic model to (re)solve.
        labeled_optimum: The provided optimal objective value to confirm.
        sense: Declared optimisation direction; cross-checked against
            ``ir.objective.sense`` (a mismatch is itself a label defect).
        source_dataset: Originating dataset tag (informational; e.g. ``"nl4opt"``).
    """

    id: str
    ir: MILP
    labeled_optimum: float
    sense: ObjSense = ObjSense.MINIMIZE
    source_dataset: str = ""


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """The re-verification outcome for one :class:`CleaningItem`.

    Attributes:
        id: The item id.
        status: :data:`STATUS_CLEAN` or :data:`STATUS_QUARANTINED`.
        computed: The solver-computed optimum (``None`` when no finite value).
        labeled: The provided label that was checked.
        sense: Optimisation direction (``"minimize"`` / ``"maximize"``).
        solver_status: Raw solver termination status (e.g. ``"optimal"``).
        problem_type: Inferred class (``MILP`` / ``MIQP`` / ``MIQCP`` / ``QUBO`` /
            ``MINLP``).
        source_dataset: Originating dataset tag.
        reason: Human-readable explanation of the verdict (names computed vs
            labeled on a quarantine).
    """

    id: str
    status: str
    computed: float | None
    labeled: float
    sense: str
    solver_status: str
    problem_type: str
    source_dataset: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping of this result."""
        return {
            "id": self.id,
            "status": self.status,
            "computed": self.computed,
            "labeled": self.labeled,
            "sense": self.sense,
            "solver_status": self.solver_status,
            "problem_type": self.problem_type,
            "source_dataset": self.source_dataset,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """The partitioned outcome of a :func:`verify_and_clean` pass.

    Attributes:
        clean: Results whose computed optimum matched the label.
        quarantined: Results that failed re-verification (mismatch / non-optimal /
            solve error / sense defect).
        solver_name: The solver used for re-verification.
        tol: The match tolerance (combined relative + absolute).
        time_limit: Per-item wall-clock ceiling (seconds) used for each solve.
    """

    clean: tuple[CleaningResult, ...]
    quarantined: tuple[CleaningResult, ...]
    solver_name: str
    tol: float
    time_limit: float

    @property
    def n_items(self) -> int:
        """Total number of items re-verified."""
        return len(self.clean) + len(self.quarantined)

    @property
    def clean_ids(self) -> tuple[str, ...]:
        """Ids of the clean items, in result order."""
        return tuple(r.id for r in self.clean)

    @property
    def quarantined_ids(self) -> tuple[str, ...]:
        """Ids of the quarantined items, in result order."""
        return tuple(r.id for r in self.quarantined)

    def to_dict(self) -> dict[str, object]:
        """Return the full report as a JSON-friendly mapping."""
        return {
            "solver_name": self.solver_name,
            "tol": self.tol,
            "time_limit": self.time_limit,
            "n_items": self.n_items,
            "n_clean": len(self.clean),
            "n_quarantined": len(self.quarantined),
            "clean": [r.to_dict() for r in self.clean],
            "quarantined": [r.to_dict() for r in self.quarantined],
        }

    def to_json(self, path: str | Path) -> Path:
        """Write ``cleaning_report.json`` (sorted keys, trailing newline); return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out


# ---------------------------------------------------------------------------
# Solver routing + classification
# ---------------------------------------------------------------------------
def _ensure_adapters_registered() -> None:
    """Import the solver-layer adapters so they self-register (idempotent).

    The structured-MINLP adapter is already registered by importing this module
    (via :data:`NONLINEAR_TERMS_KEY`); the quadratic adapters live in the solver
    layer and are imported here lazily so module import stays solver-free.
    ``importlib`` is used (not a bare ``import``) so the side-effect-only imports
    register their adapters without tripping unused-import diagnostics.
    """
    importlib.import_module("opop.solver.miqp")  # registers MiqpAdapter
    importlib.import_module("opop.solver.qubo")  # registers QuboAdapter


def _kernel_for(solver_name: str) -> SolverKernel:
    """Instantiate the :class:`~opop.solver.kernel.SolverKernel` for ``solver_name``."""
    if solver_name == "SCIP":
        from opop.solver.scip import ScipKernel

        return ScipKernel()
    if solver_name == "CP-SAT":
        from opop.solver.cpsat import CpsatKernel

        return CpsatKernel()
    if solver_name == "HiGHS":
        from opop.solver.highs import HighsKernel

        return HighsKernel()
    raise ValueError(
        f"unknown solver {solver_name!r}; supported: {', '.join(_SUPPORTED_SOLVERS)}"
    )


def _classify(ir: MILP) -> str:
    """Infer ``ir``'s problem class for the report (no solver needed)."""
    if ir.metadata.get(NONLINEAR_TERMS_KEY):
        return "MINLP"
    ext = ir.quadratic
    if ext is not None and not ext.is_empty:
        if ext.has_constraint_terms():
            return "MIQCP"
        if ir.variables and all(v.vtype is VarType.BINARY for v in ir.variables):
            return "QUBO"
        return "MIQP"
    return "MILP"


def _solve(
    ir: MILP,
    kernel: SolverKernel,
    *,
    solver_name: str,
    time_limit: float,
) -> SolveTrace:
    """Solve ``ir`` through the capability registry and return its trace.

    A plain linear MILP (no adapter) is solved directly; an adapter that declares
    ``solver_name`` as a native kernel is solved natively (the faithful, exact
    route for MIQP / MIQCP / QUBO / MINLP); otherwise the adapter's exact
    linearization is solved on the kernel.
    """
    phi = Phi()
    adapter = find_adapter(ir)
    if adapter is None:
        return kernel.solve(
            ir, phi, time_limit=time_limit, memory_limit_mb=_MEMORY_LIMIT_MB, seed=_SEED
        )
    if solver_name in adapter.capabilities.native_kernels:
        return adapter.native_solve(
            ir, kernel, phi=phi, time_limit=time_limit, memory_limit_mb=_MEMORY_LIMIT_MB, seed=_SEED
        )
    milp = adapter.to_milp(ir)
    return kernel.solve(
        milp, phi, time_limit=time_limit, memory_limit_mb=_MEMORY_LIMIT_MB, seed=_SEED
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _matches(computed: float, labeled: float, tol: float) -> bool:
    """Return ``True`` iff ``computed`` reproduces ``labeled`` within ``tol``.

    Combined relative + absolute tolerance so both near-zero and large optima are
    handled (``math.isclose`` semantics).
    """
    return math.isclose(computed, labeled, rel_tol=tol, abs_tol=tol)


def _fmt(value: float | None) -> str:
    """Render an optional float compactly for a reason string."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:g}"


def _verify_item(
    item: CleaningItem,
    kernel: SolverKernel,
    *,
    solver_name: str,
    tol: float,
    time_limit: float,
) -> CleaningResult:
    """Re-verify one item and return its :class:`CleaningResult` (never raises)."""
    problem_type = _classify(item.ir)
    sense_str = item.sense.value
    labeled = item.labeled_optimum

    def result(status: str, *, computed: float | None, solver_status: str, reason: str) -> CleaningResult:
        return CleaningResult(
            id=item.id,
            status=status,
            computed=computed,
            labeled=labeled,
            sense=sense_str,
            solver_status=solver_status,
            problem_type=problem_type,
            source_dataset=item.source_dataset,
            reason=reason,
        )

    # Defensive integrity check: a declared sense that disagrees with the model's
    # own objective sense is itself a label defect (the optimum cannot be trusted).
    if item.sense is not item.ir.objective.sense:
        return result(
            STATUS_QUARANTINED,
            computed=None,
            solver_status="not_solved",
            reason=(
                f"declared sense {item.sense.value!r} does not match model objective "
                + f"sense {item.ir.objective.sense.value!r}"
            ),
        )

    try:
        trace = _solve(item.ir, kernel, solver_name=solver_name, time_limit=time_limit)
    except Exception as exc:  # noqa: BLE001 (fail-closed: any solve failure quarantines)
        return result(
            STATUS_QUARANTINED,
            computed=None,
            solver_status="error",
            reason=f"solve failed: {type(exc).__name__}: {exc}",
        )

    computed = trace.primal_bound_series[-1] if trace.primal_bound_series else math.nan
    solver_status = trace.status

    if solver_status.strip().lower() != "optimal" or trace.censored:
        return result(
            STATUS_QUARANTINED,
            computed=computed if math.isfinite(computed) else None,
            solver_status=solver_status,
            reason=(
                f"solver did not prove optimality (status={solver_status!r}, "
                + f"censored={trace.censored}); computed={_fmt(computed)} "
                + f"vs labeled={_fmt(labeled)}"
            ),
        )

    if not math.isfinite(computed):
        return result(
            STATUS_QUARANTINED,
            computed=None,
            solver_status=solver_status,
            reason=f"no finite objective computed; labeled={_fmt(labeled)}",
        )

    if _matches(computed, labeled, tol):
        return result(
            STATUS_CLEAN,
            computed=computed,
            solver_status=solver_status,
            reason=f"computed={_fmt(computed)} matches labeled={_fmt(labeled)} within tol={tol:g}",
        )

    gap = abs(computed - labeled)
    return result(
        STATUS_QUARANTINED,
        computed=computed,
        solver_status=solver_status,
        reason=(
            f"objective mismatch: computed={_fmt(computed)} vs labeled={_fmt(labeled)} "
            + f"(|delta|={gap:g} > tol={tol:g})"
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def verify_and_clean(
    items: Sequence[CleaningItem],
    *,
    solver_name: str = "SCIP",
    tol: float = 1e-4,
    time_limit: float = 60.0,
) -> CleaningReport:
    """Re-solve every labeled item and partition into clean / quarantined.

    Each item is routed through :func:`opop.model.adapter.find_adapter` and solved
    on ``solver_name`` (default SCIP). An item is **clean** iff the solver proves
    optimality and the computed optimum reproduces ``labeled_optimum`` within
    ``tol`` (combined relative + absolute). Anything else — a sense mismatch, a
    build/solve error, a censored or non-optimal run, or an objective mismatch —
    is **quarantined** with a reason that names the computed vs labeled values.

    Args:
        items: The labeled items to re-verify.
        solver_name: Backend to use (``SCIP`` / ``CP-SAT`` / ``HiGHS``).
        tol: Match tolerance for the computed-vs-labeled comparison.
        time_limit: Per-item wall-clock ceiling in seconds.

    Returns:
        A :class:`CleaningReport` with the partitioned results.
    """
    _ensure_adapters_registered()
    kernel = _kernel_for(solver_name)

    clean: list[CleaningResult] = []
    quarantined: list[CleaningResult] = []
    for item in items:
        outcome = _verify_item(
            item, kernel, solver_name=solver_name, tol=tol, time_limit=time_limit
        )
        if outcome.status == STATUS_CLEAN:
            clean.append(outcome)
        else:
            quarantined.append(outcome)

    return CleaningReport(
        clean=tuple(clean),
        quarantined=tuple(quarantined),
        solver_name=solver_name,
        tol=float(tol),
        time_limit=float(time_limit),
    )
