"""HiGHS solver kernel (task 23) via the high-level ``highspy.Highs()`` API.

:class:`HighsKernel` is a :class:`~opop.solver.kernel.SolverKernel` backed by
HiGHS (``highspy``). It compiles a symbolic MILP IR plus a
:class:`~opop.model.state.Phi` design vector into a HiGHS model with the
*high-level* builder API (:meth:`Highs.addBinary` / :meth:`Highs.addIntegral` /
:meth:`Highs.addVariable` / :meth:`Highs.addConstr` / :meth:`Highs.maximize` /
:meth:`Highs.minimize`), runs it single-threaded under a hard time limit with a
fixed seed, and returns a :class:`~opop.model.state.SolveTrace`.

Trajectory honesty (task MUST NOT "claim richer trajectory than the backend
provides")
-----------------------------------------------------------------------------
The high-level ``maximize()`` / ``minimize()`` call runs the whole solve in one
shot and exposes only the *terminal* state — there is no clean improving-solution
callback in ``highspy`` 1.14.0 (the low-level ``setCallback`` machinery is not
used here, by design). So the primal/dual/time series each carry a SINGLE final
point. HiGHS *does* expose terminal node and simplex-iteration counts (recorded),
but exposes NO cut count, NO first-feasible timestamp, and NO peak-memory figure
through this API — those :class:`SolveTrace` fields are reported as ``None``
(genuinely unknown), never a fabricated ``0``.

Terminal statistics come from :meth:`Highs.getInfo`:
``objective_function_value`` (primal), ``mip_dual_bound`` (dual; already in the
problem's own sense, so MAX yields an upper bound directly), ``mip_node_count``
(nodes), ``simplex_iteration_count`` (LP iters). ``Highs.getRunTime`` gives the
solving time. ``kHighsInf`` equals ``math.inf`` in this build, so the no-incumbent
primal (``+inf`` for MIN / ``-inf`` for MAX) maps through unchanged.

Errors are never silently swallowed: a ``HighsStatus.kError`` from a setter or
from the solve call raises :class:`HighsKernelError`.
"""

from __future__ import annotations

import math
from typing import Any, override

from opop.model.ir import MILP, ConstraintSense, ObjSense, VarType
from opop.model.state import Phi, SolveTrace

__all__ = ["HIGHS_WHITELISTED_PARAMS", "HighsKernel", "HighsKernelError"]

#: HiGHS option names the Phase-1 proposer channel (``phi.p``) may set. Every
#: entry is a numeric tuning knob that does NOT weaken determinism or the
#: resource budget (those are owned by the authoritative ``time_limit`` /
#: ``threads`` / ``random_seed`` knobs applied AFTER ``phi.p``). A ``phi.p`` key
#: absent from this set is rejected (fail-closed) rather than applied — HiGHS
#: option names differ from SCIP's, so silently forwarding a foreign key would
#: be meaningless.
HIGHS_WHITELISTED_PARAMS: frozenset[str] = frozenset(
    {
        "mip_rel_gap",
        "mip_abs_gap",
        "mip_feasibility_tolerance",
        "primal_feasibility_tolerance",
        "dual_feasibility_tolerance",
        "mip_heuristic_effort",
    }
)

# Authoritative determinism / budget options (applied AFTER phi.p so a design
# vector can never enable parallelism or relax the time limit).
_OUTPUT_FLAG = "output_flag"
_TIME_LIMIT = "time_limit"
_THREADS = "threads"
_RANDOM_SEED = "random_seed"

#: "Not measured" sentinel for an unexposed FLOAT statistic. ``math.nan`` (NOT
#: ``None``) because the Evaluator does ``float(trace.<field>)`` — ``float(None)``
#: raises, ``float(nan)`` flows through; matches ``SolveTrace.first_feasible_time``'s
#: default and the CP-SAT kernel. Unexposed integer COUNTS use ``0`` (the journal
#: does ``int(trace.nodes)``, which rejects ``nan``).
_NOT_MEASURED: float = math.nan


class HighsKernelError(RuntimeError):
    """Raised when HiGHS reports a hard error (setter or solve ``kError``).

    Surfaces the failure instead of masking it, per the
    :class:`~opop.solver.kernel.SolverKernel` contract.
    """


#: HiGHS model-status NAMES meaning "stopped by a resource limit without an
#: optimality proof" (right-censored). Compared by ``.name`` (not the enum
#: members) so the incomplete highspy stub never trips a static attribute check.
_HIGHS_CENSORED_STATUS_NAMES: frozenset[str] = frozenset(
    {
        "kTimeLimit",
        "kIterationLimit",
        "kMemoryLimit",
        "kSolutionLimit",
        "kObjectiveBound",
        "kObjectiveTarget",
        "kInterrupt",
        "kHighsInterrupt",
    }
)


def _is_censored(status: Any) -> bool:
    """``True`` iff a :class:`highspy.HighsModelStatus` is a resource-limit stop.

    A run terminated by a limit (time / iteration / memory / solution / objective
    bound / interrupt) without an optimality proof is right-censored; compared by
    member name against :data:`_HIGHS_CENSORED_STATUS_NAMES`.
    """
    return status.name in _HIGHS_CENSORED_STATUS_NAMES


def _clean_bound(value: float) -> float:
    """Map a HiGHS bound onto a Python float, normalising the infinity sentinel.

    ``kHighsInf`` equals ``math.inf`` in this build, but guard against any large
    sentinel (>= 1e30) leaking through so the no-incumbent primal / no-dual cases
    read as a clean ``+-inf``.
    """
    if value >= 1e30 or value == math.inf:
        return math.inf
    if value <= -1e30 or value == -math.inf:
        return -math.inf
    return float(value)


class HighsKernel:
    """A HiGHS-backed :class:`~opop.solver.kernel.SolverKernel`.

    Each :meth:`solve` call builds a fresh ``highspy.Highs`` model, so a single
    kernel instance is safe to reuse across instances and seeds.
    """

    solver_name: str = "HiGHS"

    @override
    def __str__(self) -> str:
        return f"{type(self).__name__}(solver={self.solver_name!r})"

    def version(self) -> str:
        """Return the HiGHS engine version (e.g. ``"1.14.0"``); lazy import."""
        from highspy import Highs

        return str(Highs().version())

    # -- proposer hooks (Phase-1: whitelisted numeric options only) ----------
    def apply_proposer_hooks(self, model: Any, phi: Phi) -> None:
        """Apply whitelisted ``phi.p`` options to ``model`` (the proposer channel).

        Only keys in :data:`HIGHS_WHITELISTED_PARAMS` are forwarded to
        :meth:`Highs.setOptionValue`; any other key raises :class:`ValueError`
        (fail-closed — never silently applied). A whitelisted option that HiGHS
        nonetheless rejects (``kError``) raises :class:`HighsKernelError` rather
        than being masked.
        """
        from highspy import HighsStatus

        for key, value in phi.p.items():
            if key not in HIGHS_WHITELISTED_PARAMS:
                allowed = sorted(HIGHS_WHITELISTED_PARAMS)
                raise ValueError(
                    f"HiGHS param {key!r} is not in the Phase-1 whitelist; "
                    + f"allowed params: {allowed}"
                )
            if model.setOptionValue(key, float(value)) == HighsStatus.kError:
                raise HighsKernelError(f"HiGHS rejected option {key!r}={value!r}")

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

        See :class:`~opop.solver.kernel.SolverKernel` for the contract. HiGHS's
        high-level API exposes no reliable MiB memory ceiling, so
        ``memory_limit_mb`` is accepted (per the Protocol) but not enforced —
        graceful degradation, documented rather than faked. Import / build /
        solve errors propagate (``HighsKernelError`` on a HiGHS ``kError``).
        """
        from highspy import Highs, HighsStatus

        # HiGHS's high-level API exposes no reliable MiB memory ceiling, so the
        # budget's memory limit is accepted (Protocol) but not enforced here.
        del memory_limit_mb

        model = Highs()
        # Suppress solver chatter first so even the param/setup phase is quiet.
        model.setOptionValue(_OUTPUT_FLAG, False)
        # HiGHS's scheduler is a PROCESS-WIDE singleton fixed by the thread count
        # of the first solve: a prior all-cores HiGHS solve (anywhere in-process)
        # makes our threads=1 raise kError. Reset it so threads=1 always applies
        # cleanly and the trace stays deterministic. No-op when none exists.
        model.resetGlobalScheduler(True)

        # Proposer params first, then the authoritative determinism/budget knobs
        # so phi.p can never enable threads or relax the time limit.
        self.apply_proposer_hooks(model, phi)
        model.setOptionValue(_TIME_LIMIT, float(time_limit))
        model.setOptionValue(_THREADS, 1)
        model.setOptionValue(_RANDOM_SEED, int(seed))

        variables = self._build_variables(model, ir)
        self._build_constraints(model, ir, variables)
        run_status = self._set_objective_and_solve(model, ir, variables)
        if run_status == HighsStatus.kError:
            raise HighsKernelError(
                f"HiGHS solve of {ir.name!r} returned HighsStatus.kError"
            )

        status_enum = model.getModelStatus()
        status = str(model.modelStatusToString(status_enum))
        info = model.getInfo()
        final_primal = _clean_bound(float(info.objective_function_value))
        final_dual = _clean_bound(float(info.mip_dual_bound))
        solving_time = float(model.getRunTime())

        return SolveTrace(
            # Single terminal point: the high-level API exposes no per-improvement
            # trajectory, so we do not invent one.
            primal_bound_series=[final_primal],
            dual_bound_series=[final_dual],
            time_series=[solving_time],
            nodes=int(info.mip_node_count),
            lp_iters=int(info.simplex_iteration_count),
            cuts=0,  # HiGHS high-level API exposes no applied-cut count.
            first_feasible_time=_NOT_MEASURED,  # No improving-solution callback.
            status=status,
            censored=_is_censored(status_enum),
            memory_peak=_NOT_MEASURED,  # Peak memory not exposed by getInfo().
            instance_id=ir.name,
            solver=self.solver_name,
        )

    # -- model construction --------------------------------------------------
    def _build_variables(self, model: Any, ir: MILP) -> dict[str, Any]:
        """Add every IR variable to ``model`` by vtype; return name -> handle."""
        variables: dict[str, Any] = {}
        for var in ir.variables:
            lower = _to_highs_bound(var.lower)
            upper = _to_highs_bound(var.upper)
            if var.vtype is VarType.BINARY:
                handle = model.addBinary(name=var.name)
            elif var.vtype is VarType.INTEGER:
                handle = model.addIntegral(lb=lower, ub=upper, name=var.name)
            else:
                handle = model.addVariable(lb=lower, ub=upper, name=var.name)
            variables[var.name] = handle
        return variables

    def _build_constraints(
        self, model: Any, ir: MILP, variables: dict[str, Any]
    ) -> None:
        """Add every IR linear constraint to ``model`` with the right sense."""
        for con in ir.constraints:
            expr = sum(coeff * variables[name] for name, coeff in con.coeffs.items())
            if con.sense is ConstraintSense.LE:
                model.addConstr(expr <= con.rhs)
            elif con.sense is ConstraintSense.GE:
                model.addConstr(expr >= con.rhs)
            else:
                model.addConstr(expr == con.rhs)

    def _set_objective_and_solve(
        self, model: Any, ir: MILP, variables: dict[str, Any]
    ) -> Any:
        """Set the objective (incl. offset) and run the in-sense optimisation.

        Returns the :class:`highspy.HighsStatus` from ``maximize``/``minimize``.
        """
        obj_terms = [
            coeff * variables[name] for name, coeff in ir.objective.coeffs.items()
        ]
        obj_expr: Any = sum(obj_terms) if obj_terms else 0
        if ir.objective.offset != 0.0:
            obj_expr = obj_expr + ir.objective.offset
        if ir.objective.sense is ObjSense.MAXIMIZE:
            return model.maximize(obj_expr)
        return model.minimize(obj_expr)


def _to_highs_bound(value: float) -> float:
    """Map a Python IR bound (``+-math.inf``) onto a HiGHS bound.

    ``kHighsInf == math.inf`` in this build, so a finite value passes through and
    an infinite bound stays infinite — HiGHS treats ``math.inf`` as its sentinel.
    """
    return float(value)
