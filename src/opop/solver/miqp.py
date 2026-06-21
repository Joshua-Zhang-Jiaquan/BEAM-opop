"""MIQP / MIQCP problem-class adapter routed to SCIP (task 30).

:class:`MiqpAdapter` is the :class:`~opop.model.adapter.ProblemClassAdapter` for
mixed-integer *quadratic* programs (quadratic objective, MIQP) and
quadratically-*constrained* programs (MIQCP). It is the general quadratic
adapter; the pure-binary unconstrained-objective special case is claimed by
:class:`opop.solver.qubo.QuboAdapter` (the two ``can_handle`` predicates are
mutually exclusive).

Two routes (per the adapter contract):

* :meth:`MiqpAdapter.to_milp` — EXACT linearization, available ONLY when every
  quadratic product is a product of two BINARY variables (Fortet linearization,
  shared with the QUBO adapter). Any product touching an INTEGER / CONTINUOUS
  variable has no exact MILP form, so it raises
  :class:`~opop.model.ir.UnsupportedModelError` — never a silently wrong
  relaxation.
* :meth:`MiqpAdapter.native_solve` — build the quadratic model directly with
  PySCIPOpt's nonlinear API and solve it on SCIP. SCIP supports MIQCP natively;
  CP-SAT / HiGHS cannot represent general quadratic structure, so a non-SCIP
  kernel is rejected (fail-closed).

SCIP quadratic-build facts (verified live on PySCIPOpt 6.2.1 / SCIP 10):

* A quadratic constraint is added with ``model.addCons(quad_expr <=|>=|== rhs)``
  (handler reported as ``"nonlinear"``).
* ``model.setObjective`` REJECTS a nonlinear objective ("SCIP does not support
  nonlinear objective functions"). A quadratic objective is therefore modelled
  with a free continuous auxiliary ``t`` and a single quadratic constraint
  pinning it to the objective value (``t >= q`` for minimisation, ``t <= q`` for
  maximisation, with the offset folded in); the linear objective is then ``t``.

The returned :class:`~opop.model.state.SolveTrace` is a single terminal point
(proven primal/dual/time at termination); ``first_feasible_time`` is the float
"not-measured" sentinel ``math.nan`` (no event handler is registered — the
trajectory-eventhdlr machinery lives in :class:`opop.solver.scip.ScipKernel`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, final

from opop.model.adapter import AdapterCapabilities, register_adapter
from opop.model.ir import MILP, ConstraintSense, ObjSense, UnsupportedModelError, VarType
from opop.model.state import Phi, SolveTrace
from opop.solver.scip import ScipKernel

if TYPE_CHECKING:
    from opop.solver.kernel import SolverKernel

__all__ = ["MiqpAdapter", "solve_scip_quadratic"]

_VTYPE_TO_SCIP: dict[VarType, str] = {
    VarType.BINARY: "B",
    VarType.INTEGER: "I",
    VarType.CONTINUOUS: "C",
}

# Free continuous auxiliary that carries a quadratic objective value (SCIP only
# accepts a LINEAR objective, so the quadratic is pinned to this variable).
_OBJ_AUX = "_opop_obj_aux"
_OBJ_AUX_DEF = "_opop_obj_aux_def"

_BYTES_PER_MIB = 1024.0 * 1024.0

# SCIP statuses that mean "stopped early by a resource budget without an
# optimality proof" (right-censored). Mirrors opop.solver.scip._LIMIT_STATUSES;
# re-declared locally to avoid importing a private name (zero-diagnostic bar).
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


def _to_scip_bound(model: Any, value: float) -> float:
    """Map a Python bound (``math.inf``) to a SCIP infinity sentinel."""
    if value == math.inf:
        return model.infinity()
    if value == -math.inf:
        return -model.infinity()
    return value


def _finite_or_inf(model: Any, value: float) -> float:
    """Map a SCIP bound (``+-1e20`` sentinel) to a Python float / ``math.inf``."""
    if model.isInfinity(value):
        return math.inf
    if model.isInfinity(-value):
        return -math.inf
    return float(value)


def _is_censored(status: str) -> bool:
    """``True`` iff ``status`` is a resource-limit termination (right-censored)."""
    return status in _LIMIT_STATUSES


def _build_scip_model(ir: MILP) -> Any:
    """Build a fresh PySCIPOpt model carrying ``ir``'s linear + quadratic terms.

    Quadratic constraint terms augment the LHS of their (already linear)
    constraint; a quadratic objective is modelled via the free auxiliary
    ``_OBJ_AUX`` because SCIP rejects nonlinear objectives. The model is NOT
    solved and output is NOT suppressed (the caller owns its lifecycle).
    """
    from pyscipopt import Model, quicksum

    ext = ir.quadratic
    if ext is not None and _OBJ_AUX in {v.name for v in ir.variables}:
        raise UnsupportedModelError(
            f"variable name {_OBJ_AUX!r} is reserved for the quadratic-objective auxiliary"
        )

    model = Model(ir.name or "opop_quadratic")
    scip_vars: dict[str, Any] = {}
    for var in ir.variables:
        scip_vars[var.name] = model.addVar(
            name=var.name,
            vtype=_VTYPE_TO_SCIP[var.vtype],
            lb=_to_scip_bound(model, var.lower),
            ub=_to_scip_bound(model, var.upper),
        )

    con_terms = ext.constraint_terms if ext is not None else {}
    for con in ir.constraints:
        expr = quicksum(coeff * scip_vars[name] for name, coeff in con.coeffs.items())
        for term in con_terms.get(con.name, ()):
            expr = expr + term.coeff * scip_vars[term.var1] * scip_vars[term.var2]
        if con.sense is ConstraintSense.LE:
            model.addCons(expr <= con.rhs, name=con.name)
        elif con.sense is ConstraintSense.GE:
            model.addCons(expr >= con.rhs, name=con.name)
        else:
            model.addCons(expr == con.rhs, name=con.name)

    obj_linear = quicksum(
        coeff * scip_vars[name] for name, coeff in ir.objective.coeffs.items()
    )
    obj_terms = ext.objective_terms if ext is not None else ()
    if obj_terms:
        quad_obj = obj_linear
        for term in obj_terms:
            quad_obj = quad_obj + term.coeff * scip_vars[term.var1] * scip_vars[term.var2]
        aux = model.addVar(
            name=_OBJ_AUX, vtype="C", lb=-model.infinity(), ub=model.infinity()
        )
        target = quad_obj + ir.objective.offset
        if ir.objective.sense is ObjSense.MINIMIZE:
            model.addCons(aux >= target, name=_OBJ_AUX_DEF)
            model.setObjective(aux, sense="minimize")
        else:
            model.addCons(aux <= target, name=_OBJ_AUX_DEF)
            model.setObjective(aux, sense="maximize")
    else:
        model.setObjective(obj_linear, sense=ir.objective.sense.value)
        if ir.objective.offset != 0.0:
            model.addObjoffset(ir.objective.offset)

    return model


def solve_scip_quadratic(
    ir: MILP,
    *,
    phi: Phi | None = None,
    time_limit: float = 60.0,
    memory_limit_mb: int = 4096,
    seed: int = 0,
) -> SolveTrace:
    """Build ``ir`` (linear + quadratic) and solve it on SCIP under the budget.

    Honours the :class:`~opop.solver.kernel.SolverKernel` determinism contract
    (single LP thread, hard time/memory ceilings, seeded). ``phi.p`` proposer
    params are applied through the public
    :meth:`opop.solver.scip.ScipKernel.apply_proposer_hooks` (separator
    whitelist, fail-closed) BEFORE the authoritative budget knobs. Import / build
    / solve errors propagate (never swallowed).
    """
    model = _build_scip_model(ir)
    model.hideOutput()

    ScipKernel().apply_proposer_hooks(model, phi if phi is not None else Phi())
    model.setIntParam("lp/threads", 1)
    model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/memory", float(memory_limit_mb))
    model.setIntParam("randomization/randomseedshift", int(seed))

    model.optimize()

    status = str(model.getStatus())
    final_primal = _finite_or_inf(model, model.getPrimalbound())
    final_dual = _finite_or_inf(model, model.getDualbound())
    solving_time = float(model.getSolvingTime())

    return SolveTrace(
        primal_bound_series=[final_primal],
        dual_bound_series=[final_dual],
        time_series=[solving_time],
        nodes=int(model.getNTotalNodes()),
        lp_iters=int(model.getNLPIterations()),
        cuts=int(model.getNCutsApplied()),
        first_feasible_time=math.nan,
        status=status,
        censored=_is_censored(status),
        memory_peak=float(model.getMemUsed()) / _BYTES_PER_MIB,
        instance_id=ir.name,
        solver="SCIP",
    )


@final
class MiqpAdapter:
    """General MIQP / MIQCP adapter (quadratic objective and/or constraints) → SCIP."""

    @property
    def name(self) -> str:
        """Registry key for this adapter."""
        return "miqp"

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declared capabilities (exact linearization only for binary products)."""
        return AdapterCapabilities(
            name="miqp",
            problem_class="MIQP/MIQCP",
            handles_quadratic_objective=True,
            handles_quadratic_constraints=True,
            exact_linearization=False,
            native_kernels=("SCIP",),
            linear_kernels=("CP-SAT", "SCIP", "HiGHS"),
        )

    def can_handle(self, ir: MILP) -> bool:
        """Handle any quadratic model EXCEPT the pure-binary QUBO (QuboAdapter's)."""
        ext = ir.quadratic
        if ext is None or ext.is_empty:
            return False
        if ext.has_constraint_terms():
            return True
        all_binary = all(v.vtype is VarType.BINARY for v in ir.variables)
        return not all_binary

    def to_milp(self, ir: MILP) -> MILP:
        """Exactly linearize ``ir`` (binary products only) or raise.

        Delegates to the shared Fortet linearizer
        (:func:`opop.model.quadratic.linearize_quadratic`); a product touching a
        non-binary variable raises :class:`~opop.model.ir.UnsupportedModelError`
        (use :meth:`native_solve` for those).
        """
        from opop.model.quadratic import linearize_quadratic

        return linearize_quadratic(ir)

    def native_solve(
        self,
        ir: MILP,
        kernel: SolverKernel,
        *,
        phi: Phi | None = None,
        time_limit: float = 60.0,
        memory_limit_mb: int = 4096,
        seed: int = 0,
    ) -> SolveTrace:
        """Solve ``ir`` natively on SCIP (no linearization). Rejects non-SCIP kernels."""
        if getattr(kernel, "solver_name", "") != "SCIP":
            got = getattr(kernel, "solver_name", type(kernel).__name__)
            raise UnsupportedModelError(
                "MIQP/MIQCP native solve requires a SCIP kernel (solver_name='SCIP'); "
                + f"got {got!r}. CP-SAT / HiGHS cannot represent general quadratic "
                + "structure natively — linearize binary products via to_milp() instead."
            )
        return solve_scip_quadratic(
            ir,
            phi=phi,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            seed=seed,
        )


register_adapter(MiqpAdapter())
