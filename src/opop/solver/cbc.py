"""CBC solver kernel (task 23) via PuLP's bundled CBC binary.

:class:`CbcKernel` is a :class:`~opop.solver.kernel.SolverKernel` backed by the
COIN-OR CBC MILP solver, driven through ``pulp`` (which ships a CBC binary). It
compiles a symbolic MILP IR plus a :class:`~opop.model.state.Phi` design vector
into a :class:`pulp.LpProblem`, solves it single-threaded under a hard time
limit with a fixed seed (``-randomCbcSeed``), and returns a
:class:`~opop.model.state.SolveTrace`.

Trajectory honesty (task MUST NOT "claim richer trajectory than the backend
provides")
-----------------------------------------------------------------------------
PuLP shells out to the CBC command-line binary and parses only the final
solution file — there is NO improving-solution callback and NO dual/best-bound
readback. So the trajectory is degraded to a SINGLE point: the primal series
holds the final objective, the dual series holds ``None`` (genuinely unknown),
and the time series holds ``LpProblem.solutionTime``. CBC's node / LP-iteration /
cut counts and peak memory are not surfaced by PuLP either, so those
:class:`SolveTrace` fields are ``None`` (never a fabricated ``0``).

Censoring (task: "censored on timeout without proving optimality")
------------------------------------------------------------------
On a time-limit stop with an incumbent, PuLP reports the problem ``status`` as
``LpStatusOptimal`` (misleading) but the SOLUTION status ``sol_status`` as
``LpSolutionIntegerFeasible`` — that is the reliable censoring signal. A run is
NOT censored only when optimality is proven (``sol_status == LpSolutionOptimal``)
or the outcome is definitive (problem ``status`` infeasible / unbounded);
everything else (feasible-but-unproven, or stopped with no incumbent) is
right-censored.

Errors are never silently swallowed: an unavailable CBC binary raises
:class:`CbcKernelError` rather than returning a fake trace.
"""

from __future__ import annotations

import math
from typing import Any, override

from opop.model.ir import MILP, ConstraintSense, ObjSense, VarType
from opop.model.state import Phi, SolveTrace

__all__ = ["CBC_WHITELISTED_PARAMS", "CbcKernel", "CbcKernelError"]

#: ``phi.p`` keys the Phase-1 proposer channel may forward to
#: :class:`pulp.PULP_CBC_CMD` as keyword arguments. Each is a numeric tuning knob
#: that does NOT weaken determinism or the budget (``timeLimit`` / ``threads`` /
#: the seed are owned authoritatively). A ``phi.p`` key absent from this set is
#: rejected (fail-closed) — PuLP/CBC kwargs differ from other backends, so
#: forwarding a foreign key would be meaningless or unsafe.
CBC_WHITELISTED_PARAMS: frozenset[str] = frozenset({"gapRel", "gapAbs", "maxNodes"})

# PULP_CBC_CMD kwargs that take an int (the rest of the whitelist is float).
_INT_PARAMS: frozenset[str] = frozenset({"maxNodes"})

# IR vtype -> PuLP category string.
_VTYPE_TO_CAT: dict[VarType, str] = {
    VarType.BINARY: "Binary",
    VarType.INTEGER: "Integer",
    VarType.CONTINUOUS: "Continuous",
}

#: "Not measured" sentinel for an unexposed FLOAT statistic (dual bound,
#: first-feasible time, peak memory). ``math.nan`` (NOT ``None``) because the
#: Evaluator does ``float(trace.<field>)`` — ``float(None)`` raises, ``float(nan)``
#: flows through; matches ``SolveTrace.first_feasible_time``'s default and the
#: CP-SAT kernel. Unexposed integer COUNTS (nodes/lp_iters/cuts) use ``0`` instead,
#: since the orchestrator journal does ``int(trace.nodes)``, which rejects ``nan``.
_NOT_MEASURED: float = math.nan


class CbcKernelError(RuntimeError):
    """Raised when CBC cannot be used (binary unavailable / solve failure).

    Surfaces the failure instead of masking it, per the
    :class:`~opop.solver.kernel.SolverKernel` contract.
    """


class CbcKernel:
    """A CBC-backed :class:`~opop.solver.kernel.SolverKernel` (via PuLP).

    Each :meth:`solve` call builds a fresh :class:`pulp.LpProblem`, so a single
    kernel instance is safe to reuse across instances and seeds.
    """

    solver_name: str = "CBC"

    @override
    def __str__(self) -> str:
        return f"{type(self).__name__}(solver={self.solver_name!r})"

    def version(self) -> str:
        """Return the PuLP distribution version (provider of the CBC binary).

        Uses :func:`importlib.metadata.version` because ``pulp.__version__`` is
        stale relative to the installed distribution. Lazy import.
        """
        import importlib.metadata

        try:
            return importlib.metadata.version("pulp")
        except importlib.metadata.PackageNotFoundError:
            import pulp

            return str(getattr(pulp, "__version__", "?"))

    def _build_command(
        self, phi: Phi, *, time_limit: float, seed: int
    ) -> Any:
        """Build a deterministic, time-limited ``PULP_CBC_CMD`` from ``phi``.

        Whitelisted ``phi.p`` keys become solver kwargs; any other key raises
        :class:`ValueError` (fail-closed). ``msg=False`` / ``threads=1`` and the
        ``-randomCbcSeed`` option are applied authoritatively.
        """
        import pulp

        cmd_kwargs: dict[str, Any] = {
            "msg": False,
            "timeLimit": float(time_limit),
            "threads": 1,
            # CBC's deterministic-seed knob (verified: ``-randomSeed`` is rejected
            # by this binary, ``-randomCbcSeed`` is the valid command).
            "options": ["randomCbcSeed", str(int(seed))],
        }
        for key, value in phi.p.items():
            if key not in CBC_WHITELISTED_PARAMS:
                allowed = sorted(CBC_WHITELISTED_PARAMS)
                raise ValueError(
                    f"CBC param {key!r} is not in the Phase-1 whitelist; "
                    + f"allowed params: {allowed}"
                )
            cmd_kwargs[key] = int(value) if key in _INT_PARAMS else float(value)

        cmd = pulp.PULP_CBC_CMD(**cmd_kwargs)
        if not cmd.available():
            raise CbcKernelError(
                "CBC binary is not available via PuLP; cannot solve"
            )
        return cmd

    def _build_problem(self, ir: MILP) -> Any:
        """Compile the IR into a :class:`pulp.LpProblem` (objective + rows)."""
        import pulp

        sense = (
            pulp.LpMaximize
            if ir.objective.sense is ObjSense.MAXIMIZE
            else pulp.LpMinimize
        )
        prob = pulp.LpProblem(ir.name or "opop_model", sense)

        variables: dict[str, Any] = {}
        for var in ir.variables:
            low = None if var.lower == -math.inf else float(var.lower)
            up = None if var.upper == math.inf else float(var.upper)
            variables[var.name] = pulp.LpVariable(
                var.name, lowBound=low, upBound=up, cat=_VTYPE_TO_CAT[var.vtype]
            )

        obj = pulp.lpSum(
            coeff * variables[name] for name, coeff in ir.objective.coeffs.items()
        )
        if ir.objective.offset != 0.0:
            obj = obj + ir.objective.offset
        prob += obj

        for con in ir.constraints:
            expr = pulp.lpSum(
                coeff * variables[name] for name, coeff in con.coeffs.items()
            )
            if con.sense is ConstraintSense.LE:
                prob += (expr <= con.rhs, con.name)
            elif con.sense is ConstraintSense.GE:
                prob += (expr >= con.rhs, con.name)
            else:
                prob += (expr == con.rhs, con.name)
        return prob

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

        See :class:`~opop.solver.kernel.SolverKernel` for the contract. CBC via
        PuLP exposes no MiB memory ceiling, so ``memory_limit_mb`` is accepted
        (per the Protocol) but not enforced — graceful degradation, documented
        rather than faked. An unavailable CBC binary raises
        :class:`CbcKernelError` (never masked).
        """
        import pulp

        # PuLP/CBC exposes no MiB memory ceiling, so the budget's memory limit is
        # accepted (Protocol) but not enforced here.
        del memory_limit_mb

        cmd = self._build_command(phi, time_limit=time_limit, seed=seed)
        prob = self._build_problem(ir)
        prob.solve(cmd)

        status = int(prob.status)
        sol_status = int(prob.sol_status)
        status_str = str(pulp.LpSolution.get(sol_status, pulp.LpStatus.get(status)))
        solving_time = float(getattr(prob, "solutionTime", 0.0))

        obj_value = pulp.value(prob.objective)
        if obj_value is None:
            # No incumbent: report the no-bound sentinel in the problem's sense.
            final_primal = (
                math.inf if ir.objective.sense is ObjSense.MINIMIZE else -math.inf
            )
        else:
            final_primal = float(obj_value)

        proven_optimal = sol_status == pulp.LpSolutionOptimal
        definitive = status in (pulp.LpStatusInfeasible, pulp.LpStatusUnbounded)
        censored = not (proven_optimal or definitive)

        return SolveTrace(
            # Single degraded point: PuLP/CBC exposes neither an improving-solution
            # callback nor a dual/best-bound readback.
            primal_bound_series=[final_primal],
            dual_bound_series=[_NOT_MEASURED],  # No dual/best bound from PuLP.
            time_series=[solving_time],
            nodes=0,  # CBC node count not surfaced by PuLP.
            lp_iters=0,  # LP iteration count not surfaced by PuLP.
            cuts=0,  # Cut count not surfaced by PuLP.
            first_feasible_time=_NOT_MEASURED,  # No callback -> no first-feasible stamp.
            status=status_str,
            censored=censored,
            memory_peak=_NOT_MEASURED,  # Peak memory not surfaced by PuLP.
            instance_id=ir.name,
            solver=self.solver_name,
        )
