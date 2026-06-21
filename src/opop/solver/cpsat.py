"""CP-SAT solver kernel + solution-callback trajectory extraction (task 22).

:class:`CpsatKernel` is an open, integer-programming :class:`~opop.solver.kernel.SolverKernel`
backed by OR-Tools CP-SAT (``ortools==9.14.6206``). It compiles the OPOP MILP IR
plus a :class:`~opop.model.state.Phi` design vector into a ``cp_model.CpModel``,
runs it single-threaded under hard time/memory limits with a fixed seed, captures
a primal/dual trajectory via a ``CpSolverSolutionCallback``, and returns a
:class:`~opop.model.state.SolveTrace`.

How CP-SAT differs from the SCIP reference (:class:`opop.solver.scip.ScipKernel`)
-------------------------------------------------------------------------------
CP-SAT is an **integer** solver, so this kernel is *not* a drop-in MILP backend
and its trace fields carry CP-SAT-specific semantics:

* **Integer-only model.** Every coefficient / rhs / bound must be an integer.
  Float / rational coefficients are scaled to integers by a per-row common
  denominator (:mod:`opop.solver._cpsat_utils`), which is an exact, sense-
  preserving transformation. Coefficients that cannot be represented exactly
  raise :class:`~opop.model.ir.UnsupportedModelError` — never a silently wrong
  scaling. ``CONTINUOUS`` variables are rejected (CP-SAT cannot represent them);
  ``INTEGER`` variables require finite bounds.
* **Objective space.** CP-SAT optimises the *scaled* integer objective
  ``sum (c_j * K) x_j``; reported primal/dual values are mapped back to the true
  space via ``true = cpsat_value / K + offset``. ``best_objective_bound`` is
  CP-SAT's proven dual bound (equals the objective for ``OPTIMAL``); it is *not*
  the same quantity as SCIP's LP-relaxation ``getDualbound``, though both bracket
  the optimum from the dual side.
* **Status / censoring.** ``status`` is the raw CP-SAT status name in UPPER case
  (``"OPTIMAL"``/``"FEASIBLE"``/``"INFEASIBLE"``/``"UNKNOWN"``), unlike SCIP's
  lowercase strings. A run is ``censored`` iff the status is ``FEASIBLE`` (an
  incumbent without an optimality proof) or ``UNKNOWN`` (stopped by a limit) —
  the CP-SAT analogues of "terminated by a budget without proof".
* **Statistics.** ``nodes`` is ``CpSolver.num_branches`` (CP-SAT's branch count,
  not SCIP B&B nodes), ``lp_iters`` is ``num_lp_iterations``. CP-SAT exposes no
  applied-cut count, so ``cuts`` is ``0``. It exposes a memory *limit*
  (``max_memory_in_mb``, enforced here) but **not** a peak-usage readback, so
  ``memory_peak`` is reported as ``math.nan`` — the codebase's float
  "not-measured" sentinel (cf. ``SolveTrace.first_feasible_time``) — rather than
  a misleading ``0.0``.

Errors are never silently swallowed: an import / build / ``MODEL_INVALID`` solve
failure propagates to the caller.
"""

from __future__ import annotations

import math
from typing import Any

from opop.model.ir import (
    MILP,
    ConstraintSense,
    ObjSense,
    UnsupportedModelError,
    Variable,
    VarType,
)
from opop.model.state import Phi, SolveTrace
from opop.solver._cpsat_utils import (
    DEFAULT_MAX_DENOMINATOR,
    DEFAULT_SCALE_TOL,
    MAX_INT_MAGNITUDE,
    scale_row_to_integers,
)

__all__ = ["KNOWN_CPSAT_PARAMS", "CpsatKernel"]

#: CP-SAT ``SatParameters`` knobs the Phase-1 proposer hook may set via ``phi.p``,
#: mapped to the Python type each field expects. A ``phi.p`` key absent here is
#: rejected (fail-closed) rather than applied — CP-SAT has hundreds of parameters
#: and silently accepting an unknown/typo'd one could change semantics. The
#: determinism / budget knobs (``num_workers``/``random_seed``/``max_time_in_seconds``/
#: ``max_memory_in_mb``) are *accepted* here so a design vector may reference them,
#: but :meth:`CpsatKernel.solve` always overrides them afterwards so the budget
#: stays authoritative (mirrors :class:`opop.solver.scip.ScipKernel`).
KNOWN_CPSAT_PARAMS: dict[str, type] = {
    # determinism / budget — accepted but ALWAYS overridden by solve() below.
    "num_workers": int,
    "num_search_workers": int,
    "random_seed": int,
    "max_time_in_seconds": float,
    "max_memory_in_mb": int,
    "log_search_progress": bool,
    # safe search / presolve tuning knobs a proposer may explore.
    "linearization_level": int,
    "cp_model_probing_level": int,
    "cp_model_presolve": bool,
    "search_branching": int,
    "symmetry_level": int,
    "optimize_with_core": bool,
    "use_lns_only": bool,
    "use_lns": bool,
    "boolean_encoding_level": int,
    "max_num_cuts": int,
    "repair_hint": bool,
    "interleave_search": bool,
    "relative_gap_limit": float,
    "absolute_gap_limit": float,
}

# CP-SAT termination statuses that mean "stopped without an optimality proof" —
# right-censored. ``OPTIMAL`` and ``INFEASIBLE`` are definitive (not censored);
# ``FEASIBLE`` = incumbent but no proof, ``UNKNOWN`` = stopped by a limit.
_CENSORED_STATUSES: frozenset[str] = frozenset({"FEASIBLE", "UNKNOWN"})

# Statuses for which CP-SAT exposes a meaningful incumbent objective value.
_HAS_SOLUTION_STATUSES: frozenset[str] = frozenset({"OPTIMAL", "FEASIBLE"})


def _to_true(value: float, scale: int, offset: float) -> float:
    """Map a scaled CP-SAT objective/bound back to the true (unscaled) space."""
    return value / scale + offset


def _no_incumbent(sense: ObjSense) -> float:
    """Primal sentinel when no feasible solution exists (worst possible side)."""
    return math.inf if sense is ObjSense.MINIMIZE else -math.inf


def _no_bound(sense: ObjSense) -> float:
    """Dual sentinel when no objective bound is available (optimistic side)."""
    return -math.inf if sense is ObjSense.MINIMIZE else math.inf


def _make_trajectory_callback() -> Any:
    """Build a fresh trajectory ``CpSolverSolutionCallback`` instance.

    Defined as a factory (not a module-level class) so importing this module
    never requires OR-Tools — the base class is only referenced when a solve is
    actually requested. The callback fires once per *incumbent* solution and
    records the (raw, still-scaled) ``(wall_time, objective_value,
    best_objective_bound)`` triple; the kernel unscales them after the solve.
    """
    from ortools.sat.python import cp_model

    class _TrajectoryCallback(cp_model.CpSolverSolutionCallback):  # type: ignore[misc]
        """Capture the primal/dual trajectory as three index-aligned series."""

        def __init__(self) -> None:
            super().__init__()
            self.times: list[float] = []
            self.primal_raw: list[float] = []
            self.dual_raw: list[float] = []
            self.first_feasible_time: float = math.nan

        def on_solution_callback(self) -> None:
            t = float(self.wall_time)
            self.times.append(t)
            self.primal_raw.append(float(self.objective_value))
            self.dual_raw.append(float(self.best_objective_bound))
            if math.isnan(self.first_feasible_time):
                self.first_feasible_time = t

    return _TrajectoryCallback()


class CpsatKernel:
    """An OR-Tools CP-SAT-backed :class:`~opop.solver.kernel.SolverKernel`.

    Args:
        max_denominator: Largest denominator tried when recovering a rational
            from a float coefficient (see :mod:`opop.solver._cpsat_utils`).
        scale_tol: Absolute tolerance for accepting a recovered rational as a
            coefficient's exact value before scaling it to an integer.

    Each :meth:`solve` call builds a fresh ``CpModel``/``CpSolver``, so a single
    kernel instance is safe to reuse across instances and seeds.
    """

    solver_name: str = "CP-SAT"

    def __init__(
        self,
        *,
        max_denominator: int = DEFAULT_MAX_DENOMINATOR,
        scale_tol: float = DEFAULT_SCALE_TOL,
    ) -> None:
        self.max_denominator: int = int(max_denominator)
        self.scale_tol: float = float(scale_tol)

    # -- proposer hooks (Phase-1 stub; full proposer is task 14) ------------
    def apply_proposer_hooks(self, parameters: Any, phi: Phi) -> None:
        """Apply whitelisted ``phi.p`` knobs to CP-SAT ``parameters`` (fail-closed).

        Only keys in :data:`KNOWN_CPSAT_PARAMS` are accepted; any other key
        raises :class:`ValueError` (never silently applied). Values arrive as
        floats (``Phi.p`` is ``dict[str, float]``) and are coerced to each
        parameter's declared Python type.
        """
        for key, value in phi.p.items():
            ptype = KNOWN_CPSAT_PARAMS.get(key)
            if ptype is None:
                allowed = sorted(KNOWN_CPSAT_PARAMS)
                raise ValueError(
                    f"CP-SAT parameter {key!r} is not a known/whitelisted knob (fail-closed); "
                    + f"allowed parameters: {allowed}"
                )
            setattr(parameters, key, self._coerce_param(ptype, value))

    @staticmethod
    def _coerce_param(ptype: type, value: float) -> Any:
        """Coerce a ``phi.p`` float to the CP-SAT parameter's declared type."""
        if ptype is bool:
            return bool(value)
        if ptype is int:
            return int(round(value))
        return float(value)

    # -- IR -> CpModel compilation ------------------------------------------
    def _integer_domain(self, var: Variable) -> tuple[int, int]:
        """Return the finite integer ``[lo, hi]`` domain for an INTEGER variable.

        CP-SAT needs an explicit finite integer domain. Non-finite bounds, an
        empty domain, or a magnitude beyond CP-SAT's safe integer range raise
        :class:`~opop.model.ir.UnsupportedModelError` (fail-closed).
        """
        lo, hi = var.lower, var.upper
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise UnsupportedModelError(
                f"integer variable {var.name!r} has a non-finite bound (lower={lo}, upper={hi}); "
                + "CP-SAT requires a finite integer domain — provide explicit finite bounds."
            )
        ilo = math.ceil(lo)
        ihi = math.floor(hi)
        if ilo > ihi:
            raise UnsupportedModelError(
                f"integer variable {var.name!r} has an empty integer domain "
                + f"(ceil({lo})={ilo} > floor({hi})={ihi})"
            )
        if abs(ilo) > MAX_INT_MAGNITUDE or abs(ihi) > MAX_INT_MAGNITUDE:
            raise UnsupportedModelError(
                f"integer variable {var.name!r} domain [{ilo}, {ihi}] exceeds CP-SAT's safe "
                + f"integer magnitude {MAX_INT_MAGNITUDE}"
            )
        return ilo, ihi

    def _add_variable(self, model: Any, var: Variable) -> Any:
        """Create the CP-SAT integer variable for ``var`` (rejects CONTINUOUS)."""
        if var.vtype is VarType.BINARY:
            return model.new_int_var(0, 1, var.name)
        if var.vtype is VarType.INTEGER:
            ilo, ihi = self._integer_domain(var)
            return model.new_int_var(ilo, ihi, var.name)
        raise UnsupportedModelError(
            f"variable {var.name!r} is CONTINUOUS; CP-SAT is an integer-only solver and cannot "
            + "represent continuous variables without discretisation that risks a silently wrong "
            + "optimum. Use the SCIP or HiGHS kernel for models with continuous variables."
        )

    def _add_constraint(self, model: Any, var_objs: dict[str, Any], con: Any) -> None:
        """Scale ``con`` to integers and add it to ``model`` (exact, sense-preserving)."""
        from ortools.sat.python import cp_model

        names = list(con.coeffs)
        # Scale the coefficients AND the rhs by ONE common factor so the relation
        # ``sum a_j x_j (sense) rhs`` is multiplied through identically (exact).
        _scale, ints = scale_row_to_integers(
            [con.coeffs[n] for n in names] + [con.rhs],
            max_denominator=self.max_denominator,
            tol=self.scale_tol,
            what=f"constraint {con.name!r}",
        )
        int_rhs = ints[-1]
        expr = cp_model.LinearExpr.weighted_sum([var_objs[n] for n in names], ints[:-1])
        if con.sense is ConstraintSense.LE:
            model.add(expr <= int_rhs)
        elif con.sense is ConstraintSense.GE:
            model.add(expr >= int_rhs)
        else:
            model.add(expr == int_rhs)

    def _compile(self, ir: MILP) -> tuple[Any, int, float, ObjSense]:
        """Compile ``ir`` into a ``CpModel``; return (model, obj_scale, offset, sense).

        ``obj_scale`` is the positive integer the objective coefficients were
        multiplied by (1 when there is no objective); the true objective value is
        recovered as ``cpsat_value / obj_scale + offset``.
        """
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        var_objs: dict[str, Any] = {var.name: self._add_variable(model, var) for var in ir.variables}

        for con in ir.constraints:
            self._add_constraint(model, var_objs, con)

        obj = ir.objective
        has_objective = bool(obj.coeffs)
        obj_scale = 1
        if has_objective:
            names = list(obj.coeffs)
            obj_scale, ints = scale_row_to_integers(
                [obj.coeffs[n] for n in names],
                max_denominator=self.max_denominator,
                tol=self.scale_tol,
                what="objective",
            )
            expr = cp_model.LinearExpr.weighted_sum([var_objs[n] for n in names], ints)
            if obj.sense is ObjSense.MAXIMIZE:
                model.maximize(expr)
            else:
                model.minimize(expr)
        return model, obj_scale, float(obj.offset), obj.sense

    # -- the SolverKernel contract ------------------------------------------
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

        See :class:`~opop.solver.kernel.SolverKernel` for the contract. Coefficient
        scaling / unsupported constructs raise :class:`~opop.model.ir.UnsupportedModelError`;
        a CP-SAT ``MODEL_INVALID`` result raises :class:`RuntimeError`. Errors are
        never swallowed.
        """
        from ortools.sat.python import cp_model

        model, obj_scale, obj_offset, sense = self._compile(ir)

        solver = cp_model.CpSolver()
        parameters = solver.parameters
        # Proposer knobs first, then the authoritative determinism/budget knobs so
        # phi.p can never weaken reproducibility or relax the resource ceilings.
        self.apply_proposer_hooks(parameters, phi)
        parameters.num_workers = 1
        parameters.max_time_in_seconds = float(time_limit)
        parameters.max_memory_in_mb = int(memory_limit_mb)
        parameters.random_seed = int(seed)
        parameters.log_search_progress = False

        callback = _make_trajectory_callback()
        status = solver.solve(model, callback)
        status_name = str(solver.status_name(status))
        if status_name == "MODEL_INVALID":
            # Build error surfaced, never masked (honours the kernel contract).
            raise RuntimeError(
                f"CP-SAT returned MODEL_INVALID for instance {ir.name!r}: "
                + f"{model.validate()!r}"
            )

        has_solution = status_name in _HAS_SOLUTION_STATUSES
        if has_solution:
            final_primal = _to_true(float(solver.objective_value), obj_scale, obj_offset)
            final_dual = _to_true(float(solver.best_objective_bound), obj_scale, obj_offset)
        else:
            final_primal = _no_incumbent(sense)
            final_dual = _no_bound(sense)

        # Close the trajectory with the proven terminal bounds so the series is
        # non-empty even if no solution callback fired and always ends at the
        # final state (mirrors the SCIP kernel's final-point append).
        primal_series = [_to_true(p, obj_scale, obj_offset) for p in callback.primal_raw]
        dual_series = [_to_true(d, obj_scale, obj_offset) for d in callback.dual_raw]
        time_series = list(callback.times)
        primal_series.append(final_primal)
        dual_series.append(final_dual)
        time_series.append(float(solver.wall_time))

        return SolveTrace(
            primal_bound_series=primal_series,
            dual_bound_series=dual_series,
            time_series=time_series,
            nodes=int(solver.num_branches),
            lp_iters=int(solver.response_proto.num_lp_iterations),
            cuts=0,  # CP-SAT exposes no applied-cut count via the Python API.
            first_feasible_time=callback.first_feasible_time,
            status=status_name,
            censored=status_name in _CENSORED_STATUSES,
            memory_peak=math.nan,  # CP-SAT exposes a memory LIMIT, not a peak readback.
            instance_id=ir.name,
            solver=self.solver_name,
        )
