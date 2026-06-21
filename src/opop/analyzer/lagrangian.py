"""Lagrangian-relaxation dual-bound estimation for the OPOP analyzer (task 26).

A MILP with *coupling* constraints (the bordered-block-diagonal rows that
Dantzig--Wolfe would put in the master, see :mod:`opop.analyzer.decompose`) often
has a Lagrangian dual bound strictly tighter than its LP relaxation. Relaxing the
coupling rows into the objective with multipliers ``lambda`` leaves a subproblem
that keeps integrality; maximising the resulting concave dual function gives a
bound ``q*`` that, for a minimisation problem, satisfies

    z_LP  <=  q*  <=  z_IP

(and the mirror inequality for maximisation). The improvement over ``z_LP`` is
exactly the integrality the subproblem retains — the Lagrangian "integrality
property" gap. This module estimates ``q*`` by **projected subgradient ascent**.

Why each iterate is a *valid* bound. For a minimisation in the internal
"min-space" (the objective negated for a maximise model), every iterate uses the
subproblem's **dual bound** (a global lower bound on the subproblem optimum), so

    L(lambda) = subproblem_dual_bound(lambda)  +  constant(lambda)  <=  z_IP

for every dual-feasible ``lambda`` (``lambda_i >= 0`` for inequality rows, free
for equalities). We report ``L_best = max_k L(lambda_k)``, which is therefore a
*certified* dual bound, never an optimistic guess. Polyak step sizing uses a
genuine upper bound (the integer optimum of the full model when cheaply solved),
so on a healthy-gap instance the very first iterate already dominates ``z_LP``.

This is the **only** analyzer module that requires a solver. When SCIP is
unavailable it degrades cleanly to ``status="UNAVAILABLE"`` (never raises); when
the model has no coupling constraints it returns ``status="NO_COUPLING"``. The
input IR is never mutated (subproblems are built with :func:`dataclasses.replace`
on fresh copies).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from opop.analyzer.decompose import detect_decomposition
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    VarType,
    to_pyscipopt,
)

__all__ = [
    "LAGRANGIAN_ANALYZED",
    "LAGRANGIAN_NO_COUPLING",
    "LAGRANGIAN_UNAVAILABLE",
    "LagrangianBound",
    "estimate_lagrangian_bound",
]

#: Subgradient ascent ran and produced a certified dual bound.
LAGRANGIAN_ANALYZED = "ANALYZED"
#: No coupling constraints to dualize (no bordered-block structure / none supplied).
LAGRANGIAN_NO_COUPLING = "NO_COUPLING"
#: SCIP could not be used; no bound was computed.
LAGRANGIAN_UNAVAILABLE = "UNAVAILABLE"

#: Dual bound for a minimisation problem is a lower bound on the optimum.
BOUND_LOWER = "lower"
#: Dual bound for a maximisation problem is an upper bound on the optimum.
BOUND_UPPER = "upper"


# ---------------------------------------------------------------------------
# LagrangianBound
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LagrangianBound:
    """Lagrangian dual-bound estimate for one MILP.

    Attributes:
        bound: The certified dual bound in the model's ORIGINAL objective sense
            (a lower bound for MINIMIZE, an upper bound for MAXIMIZE), or ``None``
            when no bound was computed (``status != "ANALYZED"``).
        lp_obj: The LP-relaxation objective (original sense) used for the
            ``dominates_lp`` comparison, or ``None`` if it was not solved.
        bound_kind: :data:`BOUND_LOWER` or :data:`BOUND_UPPER`.
        sense: The model's objective sense value (``"minimize"`` / ``"maximize"``).
        dominates_lp: ``True`` iff the bound is at least as tight as the LP
            relaxation within tolerance (``>= lp_obj`` for MINIMIZE, ``<= lp_obj``
            for MAXIMIZE); ``None`` when either bound is missing.
        coupling_constraints: Names of the constraints that were dualized.
        n_iterations: Subgradient iterations actually performed.
        status: :data:`LAGRANGIAN_ANALYZED`, :data:`LAGRANGIAN_NO_COUPLING`, or
            :data:`LAGRANGIAN_UNAVAILABLE`.
    """

    bound: float | None = None
    lp_obj: float | None = None
    bound_kind: str = BOUND_LOWER
    sense: str = ObjSense.MINIMIZE.value
    dominates_lp: bool | None = None
    coupling_constraints: tuple[str, ...] = ()
    n_iterations: int = 0
    status: str = LAGRANGIAN_NO_COUPLING

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the dual-bound estimate."""
        return {
            "bound": self.bound,
            "lp_obj": self.lp_obj,
            "bound_kind": self.bound_kind,
            "sense": self.sense,
            "dominates_lp": self.dominates_lp,
            "coupling_constraints": list(self.coupling_constraints),
            "n_iterations": self.n_iterations,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Internal solver result
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SolveResult:
    status: str
    obj_val: float | None
    dual_bound: float | None
    solution: dict[str, float]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def estimate_lagrangian_bound(
    ir: MILP,
    *,
    coupling: tuple[str, ...] | None = None,
    max_iters: int = 200,
    step_scale: float = 1.0,
    patience: int = 25,
    subproblem_node_limit: int | None = None,
    tol: float = 1e-6,
) -> LagrangianBound:
    """Estimate the Lagrangian dual bound of ``ir`` by subgradient ascent.

    Args:
        ir: The MILP to bound (never mutated).
        coupling: Explicit coupling-constraint names to dualize. When ``None``
            the bordered-block-diagonal linking constraints from
            :func:`~opop.analyzer.decompose.detect_decomposition` are used.
        max_iters: Maximum subgradient iterations.
        step_scale: Polyak step multiplier in ``(0, 2]`` (and the diminishing
            fallback scale when no upper bound is available).
        patience: Stop early after this many iterations without improvement.
        subproblem_node_limit: Node cap for each subproblem solve (``None`` =
            solve to optimality; only the subproblem DUAL bound is used, so the
            reported bound stays valid even under a cap).
        tol: Convergence / comparison tolerance.

    Returns:
        A :class:`LagrangianBound`. ``status="NO_COUPLING"`` when there is nothing
        to dualize and ``status="UNAVAILABLE"`` when SCIP cannot be used.
    """
    coupling_names = _resolve_coupling(ir, coupling)
    sense_value = ir.objective.sense.value
    if not coupling_names:
        return LagrangianBound(sense=sense_value, status=LAGRANGIAN_NO_COUPLING)

    is_min = ir.objective.sense is ObjSense.MINIMIZE
    bound_kind = BOUND_LOWER if is_min else BOUND_UPPER

    lp_obj = _solve_lp_obj(ir)
    if lp_obj is None and not _scip_available(ir):
        return LagrangianBound(
            sense=sense_value,
            bound_kind=bound_kind,
            coupling_constraints=coupling_names,
            status=LAGRANGIAN_UNAVAILABLE,
        )

    ascent = _subgradient_ascent(
        ir,
        coupling_names,
        is_min=is_min,
        max_iters=max_iters,
        step_scale=step_scale,
        patience=patience,
        node_limit=subproblem_node_limit,
        tol=tol,
    )
    if ascent is None:
        return LagrangianBound(
            sense=sense_value,
            bound_kind=bound_kind,
            lp_obj=lp_obj,
            coupling_constraints=coupling_names,
            status=LAGRANGIAN_UNAVAILABLE,
        )

    best_min, n_iterations = ascent
    bound = best_min if is_min else -best_min
    dominates = _dominates_lp(bound, lp_obj, is_min=is_min, tol=tol)
    return LagrangianBound(
        bound=bound,
        lp_obj=lp_obj,
        bound_kind=bound_kind,
        sense=sense_value,
        dominates_lp=dominates,
        coupling_constraints=coupling_names,
        n_iterations=n_iterations,
        status=LAGRANGIAN_ANALYZED,
    )


# ---------------------------------------------------------------------------
# Coupling resolution
# ---------------------------------------------------------------------------
def _resolve_coupling(ir: MILP, coupling: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve the coupling-constraint set: explicit, else DW linking rows."""
    valid = {c.name for c in ir.constraints}
    if coupling is not None:
        return tuple(name for name in coupling if name in valid)
    return tuple(name for name in detect_decomposition(ir).linking_constraints if name in valid)


def _dominates_lp(
    bound: float, lp_obj: float | None, *, is_min: bool, tol: float
) -> bool | None:
    """Whether ``bound`` is at least as tight as the LP bound within ``tol``."""
    if lp_obj is None:
        return None
    return bound >= lp_obj - tol if is_min else bound <= lp_obj + tol


# ---------------------------------------------------------------------------
# Subgradient ascent (min-space)
# ---------------------------------------------------------------------------
def _subgradient_ascent(
    ir: MILP,
    coupling_names: tuple[str, ...],
    *,
    is_min: bool,
    max_iters: int,
    step_scale: float,
    patience: int,
    node_limit: int | None,
    tol: float,
) -> tuple[float, int] | None:
    """Run projected subgradient ascent in min-space; return ``(best, n_iters)``.

    Returns ``None`` only when SCIP is unavailable (the first subproblem solve
    fails to build a model). ``best`` is the largest certified min-space dual
    bound seen; convert to the original sense in the caller.
    """
    obj_sign = 1.0 if is_min else -1.0
    base_obj = {v.name: 0.0 for v in ir.variables}
    for name, coeff in ir.objective.coeffs.items():
        base_obj[name] = obj_sign * coeff

    coupling_set = set(coupling_names)
    coupling_rows = tuple(c for c in ir.constraints if c.name in coupling_set)
    sub_constraints = tuple(c for c in ir.constraints if c.name not in coupling_set)
    sub_template = replace(ir, constraints=sub_constraints, metadata={})

    upper_bound = _min_space_upper_bound(ir, obj_sign)
    multipliers = {c.name: 0.0 for c in coupling_rows}

    best = -math.inf
    stalled = 0
    performed = 0
    for k in range(max_iters):
        penalized, constant = _penalized_objective(base_obj, coupling_rows, multipliers)
        result = _solve_min_subproblem(sub_template, penalized, node_limit)
        if result is None:
            return None if performed == 0 else (best, performed)
        performed += 1

        bound_value = _subproblem_bound(result)
        if bound_value is not None:
            candidate = bound_value + constant
            if candidate > best:
                best, stalled = candidate, 0
            else:
                stalled += 1

        gradient, grad_norm2 = _subgradient(coupling_rows, result.solution)
        if grad_norm2 <= tol * tol or stalled >= patience:
            break

        step = _step_size(
            best=best,
            upper_bound=upper_bound,
            grad_norm2=grad_norm2,
            iteration=k,
            step_scale=step_scale,
        )
        if step <= 1e-12:
            break
        _update_multipliers(multipliers, coupling_rows, gradient, step)

    return (best if math.isfinite(best) else -math.inf), performed


def _penalized_objective(
    base_obj: dict[str, float],
    coupling_rows: tuple[LinearConstraint, ...],
    multipliers: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Min-space objective ``c + sum_i sigma_i lambda_i a_i`` and constant ``-sum sigma_i lambda_i b_i``."""
    penalized = dict(base_obj)
    constant = 0.0
    for con in coupling_rows:
        sigma = _sigma(con.sense)
        weight = sigma * multipliers[con.name]
        if weight == 0.0:
            continue
        for name, coeff in con.coeffs.items():
            penalized[name] = penalized.get(name, 0.0) + weight * coeff
        constant -= weight * con.rhs
    return penalized, constant


def _subgradient(
    coupling_rows: tuple[LinearConstraint, ...], solution: dict[str, float]
) -> tuple[dict[str, float], float]:
    """Subgradient ``v_i(x) = sigma_i (a_i . x - b_i)`` and its squared norm."""
    gradient: dict[str, float] = {}
    norm2 = 0.0
    for con in coupling_rows:
        activity = sum(coeff * solution.get(name, 0.0) for name, coeff in con.coeffs.items())
        value = _sigma(con.sense) * (activity - con.rhs)
        gradient[con.name] = value
        norm2 += value * value
    return gradient, norm2


def _update_multipliers(
    multipliers: dict[str, float],
    coupling_rows: tuple[LinearConstraint, ...],
    gradient: dict[str, float],
    step: float,
) -> None:
    """Ascend then project: ``lambda_i >= 0`` for inequalities, free for equalities."""
    for con in coupling_rows:
        updated = multipliers[con.name] + step * gradient[con.name]
        if con.sense is not ConstraintSense.EQ:
            updated = max(0.0, updated)
        multipliers[con.name] = updated


def _step_size(
    *,
    best: float,
    upper_bound: float | None,
    grad_norm2: float,
    iteration: int,
    step_scale: float,
) -> float:
    """Polyak step ``alpha (UB - L) / ||g||^2`` if a UB is known, else diminishing."""
    if upper_bound is not None and math.isfinite(best) and upper_bound > best:
        return step_scale * (upper_bound - best) / grad_norm2
    return step_scale / (math.sqrt(iteration + 1) * math.sqrt(grad_norm2))


def _sigma(sense: ConstraintSense) -> float:
    """Sign so a dual-feasible penalty is non-negative: +1 for LE/EQ, -1 for GE."""
    return -1.0 if sense is ConstraintSense.GE else 1.0


# ---------------------------------------------------------------------------
# SCIP solves (the only impure part — guarded so missing SCIP never raises)
# ---------------------------------------------------------------------------
def _min_space_upper_bound(ir: MILP, obj_sign: float) -> float | None:
    """Min-space upper bound on ``q*``: the full integer optimum (best primal)."""
    result = _solve(ir, relax_integrality=False, node_limit=10_000)
    if result is None or result.obj_val is None:
        return None
    return obj_sign * result.obj_val


def _solve_lp_obj(ir: MILP) -> float | None:
    """LP-relaxation objective in the ORIGINAL sense (clean root LP), or ``None``."""
    result = _solve(ir, relax_integrality=True, clean_lp=True)
    if result is None or result.status != "optimal":
        return None
    return result.obj_val


def _solve_min_subproblem(
    sub_template: MILP, penalized: dict[str, float], node_limit: int | None
) -> _SolveResult | None:
    """Solve the integer subproblem in min-space with the penalized objective."""
    nonzero = {name: coeff for name, coeff in penalized.items() if coeff != 0.0}
    sub_ir = replace(
        sub_template,
        objective=Objective(coeffs=nonzero, sense=ObjSense.MINIMIZE, offset=0.0),
    )
    return _solve(sub_ir, relax_integrality=False, node_limit=node_limit)


def _subproblem_bound(result: _SolveResult) -> float | None:
    """Valid min-space lower bound for the subproblem: its dual bound (finite)."""
    if result.status == "optimal" and result.obj_val is not None:
        return result.obj_val
    if result.dual_bound is not None and math.isfinite(result.dual_bound):
        return result.dual_bound
    return None


def _scip_available(ir: MILP) -> bool:
    """Whether a SCIP model can be built for ``ir`` (probe, never raises)."""
    try:
        to_pyscipopt(ir)
    except Exception:
        return False
    return True


def _solve(
    ir: MILP,
    *,
    relax_integrality: bool,
    node_limit: int | None = None,
    time_limit: float | None = None,
    clean_lp: bool = False,
) -> _SolveResult | None:
    """Build a SCIP model for ``ir`` and solve it (threads=1, fixed seed).

    Returns ``None`` when SCIP cannot be used (so callers fail closed). The
    objective value is in ``ir``'s own sense; the dual bound is SCIP's global
    bound and is used to keep Lagrangian iterates valid under node caps.
    """
    target = _continuous(ir) if relax_integrality else ir
    try:
        model = to_pyscipopt(target)
    except Exception:
        return None

    model.hideOutput()
    model.setParam("randomization/randomseedshift", 0)
    model.setParam("parallel/maxnthreads", 1)
    if clean_lp:
        model.setParam("presolving/maxrounds", 0)
        model.setParam("separating/maxrounds", 0)
        model.setParam("separating/maxroundsroot", 0)
    if node_limit is not None:
        model.setParam("limits/nodes", node_limit)
    if time_limit is not None:
        model.setParam("limits/time", time_limit)

    try:
        model.optimize()
        status = str(model.getStatus())
        has_solution = model.getNSols() > 0
        obj_val = float(model.getObjVal()) if has_solution else None
        dual_bound = _finite_or_none(float(model.getDualbound()))
        solution = (
            {var.name: float(model.getVal(var)) for var in model.getVars()}
            if has_solution
            else {}
        )
    except Exception:
        return None
    return _SolveResult(
        status=status, obj_val=obj_val, dual_bound=dual_bound, solution=solution
    )


def _continuous(ir: MILP) -> MILP:
    """Return a NEW IR with every variable CONTINUOUS (LP relaxation; pure)."""
    relaxed = tuple(
        v if v.vtype is VarType.CONTINUOUS else replace(v, vtype=VarType.CONTINUOUS)
        for v in ir.variables
    )
    return replace(ir, variables=relaxed)


def _finite_or_none(value: float) -> float | None:
    """Map SCIP's +/-1e20 infinity sentinels (and non-finite) to ``None``."""
    if not math.isfinite(value) or abs(value) >= 1e20:
        return None
    return value
