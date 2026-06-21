"""SCIP solver kernel + event-based trajectory extraction (task 12).

:class:`ScipKernel` is the Phase-1 reference :class:`~opop.solver.kernel.SolverKernel`:
it compiles a symbolic MILP IR (:func:`opop.model.ir.to_pyscipopt`) plus a
:class:`~opop.model.state.Phi` design vector into a PySCIPOpt model, runs it
single-threaded under hard time/memory limits with a fixed seed, and returns a
rich :class:`~opop.model.state.SolveTrace`.

Trajectory capture
------------------
A :class:`_TrajectoryEventhdlr` (subclassing :class:`pyscipopt.Eventhdlr`) is
registered before the solve and catches ``BESTSOLFOUND`` and
``DUALBOUNDIMPROVED`` (optionally ``GAPUPDATED``). Each caught event appends one
``(solving_time, primal_bound, dual_bound)`` triple to three parallel series.

**Callback-timing trap (verified on PySCIPOpt 6.2.1 / SCIP 10.0.2):** inside the
``BESTSOLFOUND`` callback ``model.getPrimalbound()`` still returns the *previous*
incumbent (the new bound is committed after handlers run). The new incumbent is
read correctly via ``model.getSolObjVal(model.getBestSol())`` — that is what this
kernel uses for the primal series. ``getDualbound()`` is current inside both
events. See the notepad for the probe evidence.

Statistics use ``getNTotalNodes`` (NOT ``getNNodes``, which resets at restart),
``getNLPIterations``, ``getNCutsApplied``, ``getPrimalDualIntegral``, ``getGap``,
``getSolvingTime``, and ``getMemUsed``. Errors are never silently swallowed:
import/solve failures propagate to the caller.
"""

from __future__ import annotations

import math
from typing import Any, override

from opop.model.ir import MILP, to_pyscipopt
from opop.model.state import Phi, SolveTrace

__all__ = ["ScipKernel", "WHITELISTED_SEPARATORS"]

#: SCIP separator families accepted by the Phase-1 proposer hook. Every entry
#: produces *globally valid inequalities* (class-B: may cut fractional LP points
#: but never removes a feasible integer incumbent), so injecting them is safe
#: under the Verification Strategy. A ``separating/<name>/...`` param whose
#: ``<name>`` is absent here is rejected (fail-closed) rather than applied.
WHITELISTED_SEPARATORS: frozenset[str] = frozenset(
    {
        "gomory",
        "gomorymi",
        "strongcg",
        "cmir",
        "aggregation",
        "flowcover",
        "zerohalf",
        "clique",
        "impliedbounds",
        "intobj",
        "mcf",
        "oddcycle",
        "disjunctive",
        "mixing",
        "rlt",
    }
)

# SCIP termination statuses that mean "stopped early by a resource budget
# without an optimality proof" — i.e. right-censored. Definitive statuses
# (``optimal``/``infeasible``/``unbounded``/``inforunbd``) are NOT censored.
_LIMIT_STATUSES: frozenset[str] = frozenset(
    {
        "timelimit",
        "memlimit",
        "gaplimit",
        "sollimit",
        "bestsollimit",
        "nodelimit",
        "totalnodelimit",
        "stallnodelimit",
        "restartlimit",
        "userinterrupt",
        "terminate",
    }
)

# Core determinism / budget parameters. Applied AFTER ``phi.p`` so the budget is
# authoritative and a design vector can never weaken reproducibility (e.g. by
# enabling LP threads or relaxing the time limit).
_THREADS_PARAM = "lp/threads"
_TIME_PARAM = "limits/time"
_MEMORY_PARAM = "limits/memory"
_SEED_PARAM = "randomization/randomseedshift"

_BYTES_PER_MIB = 1024.0 * 1024.0


def _finite_or_inf(model: Any, value: float) -> float:
    """Map a SCIP bound (``+-1e20`` sentinel) to a Python float / ``math.inf``."""
    if model.isInfinity(value):
        return math.inf
    if model.isInfinity(-value):
        return -math.inf
    return float(value)


def _is_censored(status: str) -> bool:
    """``True`` iff ``status`` is a resource-limit termination (right-censored).

    Per the task contract: a run terminated by a limit (time/node/memory/gap/...)
    without an optimality proof is censored; ``optimal`` and the definitive
    infeasible/unbounded statuses are not.
    """
    return status in _LIMIT_STATUSES


def _make_trajectory_eventhdlr(capture_gap_events: bool) -> Any:
    """Build a fresh trajectory :class:`pyscipopt.Eventhdlr` instance.

    Defined as a factory (not a module-level class) so importing this module
    never requires PySCIPOpt — the base class is only referenced when a solve is
    actually requested.
    """
    from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

    bestsol = SCIP_EVENTTYPE.BESTSOLFOUND
    dualimp = SCIP_EVENTTYPE.DUALBOUNDIMPROVED
    gapupd = SCIP_EVENTTYPE.GAPUPDATED
    bestsol_type = int(bestsol)

    class _TrajectoryEventhdlr(Eventhdlr):  # type: ignore[misc]
        """Capture the primal/dual trajectory as three parallel series.

        ``primal``/``dual``/``times`` are appended together on every caught event
        so they stay index-aligned. ``first_feasible_time`` records the solving
        time of the first ``BESTSOLFOUND`` (the first feasible solution).
        """

        def __init__(self) -> None:
            super().__init__()
            self.primal: list[float] = []
            self.dual: list[float] = []
            self.times: list[float] = []
            self.first_feasible_time: float = math.nan

        @override
        def eventinit(self) -> None:
            self.model.catchEvent(bestsol, self)
            self.model.catchEvent(dualimp, self)
            if capture_gap_events:
                self.model.catchEvent(gapupd, self)

        @override
        def eventexit(self) -> None:
            self.model.dropEvent(bestsol, self)
            self.model.dropEvent(dualimp, self)
            if capture_gap_events:
                self.model.dropEvent(gapupd, self)

        @override
        def eventexec(self, event: Any) -> None:
            model = self.model
            t = float(model.getSolvingTime())
            if event.getType() == bestsol_type:
                # getPrimalbound() is stale inside BESTSOLFOUND; read the new
                # incumbent directly from the solution that triggered the event.
                sol = model.getBestSol()
                primal = (
                    _finite_or_inf(model, model.getSolObjVal(sol))
                    if sol is not None
                    else _finite_or_inf(model, model.getPrimalbound())
                )
                if math.isnan(self.first_feasible_time):
                    self.first_feasible_time = t
            else:
                primal = _finite_or_inf(model, model.getPrimalbound())
            self.times.append(t)
            self.primal.append(primal)
            self.dual.append(_finite_or_inf(model, model.getDualbound()))

    return _TrajectoryEventhdlr()


class ScipKernel:
    """A SCIP-backed :class:`~opop.solver.kernel.SolverKernel`.

    Args:
        capture_gap_events: When ``True`` also catch ``GAPUPDATED`` events
            (denser trajectory). Defaults to ``False`` — the canonical
            primal/dual event series uses ``BESTSOLFOUND``/``DUALBOUNDIMPROVED``
            only, which keeps the series compact on hard instances.

    Each :meth:`solve` call builds a fresh PySCIPOpt model, so a single kernel
    instance is safe to reuse across instances and seeds.
    """

    solver_name: str = "SCIP"

    def __init__(self, *, capture_gap_events: bool = False) -> None:
        self.capture_gap_events: bool = capture_gap_events

    # -- proposer hooks (Phase-1 stub; full proposer is task 14) ------------
    def apply_proposer_hooks(self, model: Any, phi: Phi) -> None:
        """Apply ``phi.p`` parameters to ``model`` (the proposer channel).

        ``phi.p`` is the single parameter channel for Phase-1. Separator
        injections (``separating/<name>/...``) are accepted only when ``<name>``
        is in :data:`WHITELISTED_SEPARATORS` (class-B valid inequalities);
        anything else raises :class:`ValueError` (fail-closed — never silently
        applied). All other keys (including decomposition flag toggles such as
        ``decomposition/...`` or ``constraints/benders/...``) pass through.

        Translating the high-level ``phi.d``/``phi.h`` fields into concrete
        parameter sets is deferred to the task-14 proposer.
        """
        for key, value in phi.p.items():
            if key.startswith("separating/"):
                parts = key.split("/")
                sep_name = parts[1] if len(parts) >= 2 else ""
                if sep_name not in WHITELISTED_SEPARATORS:
                    allowed = sorted(WHITELISTED_SEPARATORS)
                    raise ValueError(
                        f"separator {sep_name!r} (param {key!r}) is not class-B whitelisted; "
                        + f"allowed separators: {allowed}"
                    )
            model.setParam(key, value)

    def solve(
        self,
        ir: MILP,
        phi: Phi,
        *,
        time_limit: float,
        memory_limit_mb: int,
        seed: int,
    ) -> SolveTrace:
        """Compile ``ir`` + ``phi``, solve under the budget, return a trace.

        See :class:`~opop.solver.kernel.SolverKernel` for the contract. Solver
        import/build/solve errors propagate to the caller (never swallowed).
        """
        model = to_pyscipopt(ir)
        model.hideOutput()

        # Proposer params first, then the authoritative determinism/budget knobs
        # so phi.p can never weaken reproducibility or the resource ceilings.
        self.apply_proposer_hooks(model, phi)
        model.setIntParam(_THREADS_PARAM, 1)
        model.setRealParam(_TIME_PARAM, float(time_limit))
        model.setRealParam(_MEMORY_PARAM, float(memory_limit_mb))
        model.setIntParam(_SEED_PARAM, int(seed))

        handler = _make_trajectory_eventhdlr(self.capture_gap_events)
        model.includeEventhdlr(handler, "opop_trajectory", "OPOP primal/dual trajectory capture")

        model.optimize()

        status = str(model.getStatus())
        final_primal = _finite_or_inf(model, model.getPrimalbound())
        final_dual = _finite_or_inf(model, model.getDualbound())
        solving_time = float(model.getSolvingTime())

        # Always close the series with the proven terminal bounds so it is
        # non-empty even if no event fired and always ends at the final state.
        primal_series = [*handler.primal, final_primal]
        dual_series = [*handler.dual, final_dual]
        time_series = [*handler.times, solving_time]

        return SolveTrace(
            primal_bound_series=primal_series,
            dual_bound_series=dual_series,
            time_series=time_series,
            nodes=int(model.getNTotalNodes()),
            lp_iters=int(model.getNLPIterations()),
            cuts=int(model.getNCutsApplied()),
            first_feasible_time=handler.first_feasible_time,
            status=status,
            censored=_is_censored(status),
            memory_peak=float(model.getMemUsed()) / _BYTES_PER_MIB,
            instance_id=ir.name,
            solver=self.solver_name,
        )
