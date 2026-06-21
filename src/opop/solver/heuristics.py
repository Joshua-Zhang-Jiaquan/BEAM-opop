"""Generic MIP matheuristic cores for OPOP (task 25).

Four schedulable, *problem-agnostic* neighbourhood heuristics that operate on an
OPOP :class:`~opop.model.ir.MILP` IR plus an incumbent assignment and return an
improved (or at worst equal) incumbent together with the solver trace(s):

* :func:`local_branching` — Fischetti--Lodi local branching: add a Hamming
  ``sum |x - x_bar| <= k`` neighbourhood constraint over the *discrete* variables
  and solve the restricted MIP.
* :func:`rins` — Danna--Rothberg--Le Pape Relaxation Induced Neighbourhood
  Search: solve the LP relaxation, fix the discrete variables whose LP value
  agrees with the incumbent, then solve the restricted MIP.
* :func:`large_neighborhood_search` — generic destroy/repair LNS: free a random
  fraction of the discrete variables, fix the rest to the incumbent, repair by a
  short MIP solve, accept improving repairs, and repeat (seeded → deterministic).
* :func:`repair_solution` — solve a feasibility MIP that *minimises the distance*
  to a (possibly infeasible/partial) target assignment.

Every returned incumbent is verified feasible against the **original** IR by
direct arithmetic constraint evaluation (:func:`is_solution_feasible`); a
heuristic NEVER hands back an infeasible assignment without a failed status
(``incumbent is None`` + ``feasible=False``).

Design contract
---------------
* The input IR is never mutated — neighbourhood models are built with
  :func:`dataclasses.replace` over fresh tuples/dicts.
* The inner MIP solver reuses :class:`opop.solver.scip.ScipKernel` machinery
  (its public ``apply_proposer_hooks`` param channel) and mirrors its
  determinism budget (single-threaded ``lp/threads=1``, seeded
  ``randomization/randomseedshift``, hard ``limits/time`` / ``limits/memory``)
  and right-censoring convention. Unexposed float stats use the ``math.nan``
  sentinel and unexposed integer counts use ``0`` (the codebase convention).
* SCIP is imported lazily through :func:`opop.model.ir.to_pyscipopt`, so
  importing this module needs no solver backend; only the solves do.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
    to_pyscipopt,
)
from opop.model.state import Phi, SolveTrace
from opop.solver.scip import ScipKernel

__all__ = [
    "DEFAULT_MEMORY_LIMIT_MB",
    "FEAS_TOL",
    "HeuristicResult",
    "INT_TOL",
    "OBJ_TOL",
    "is_solution_feasible",
    "large_neighborhood_search",
    "local_branching",
    "repair_solution",
    "rins",
    "solution_violations",
]

#: Default hard memory ceiling (MiB) for the inner MIP/LP solves.
DEFAULT_MEMORY_LIMIT_MB: int = 4096
#: Absolute tolerance for linear-constraint satisfaction (direct evaluation).
FEAS_TOL: float = 1e-6
#: Absolute tolerance for integrality / RINS LP-vs-incumbent agreement.
INT_TOL: float = 1e-6
#: Absolute tolerance for declaring one objective value strictly better.
OBJ_TOL: float = 1e-9

# Terminal heuristic statuses (carried on :class:`HeuristicResult`).
STATUS_IMPROVED = "improved"
STATUS_NO_IMPROVEMENT = "no_improvement"
STATUS_INFEASIBLE = "infeasible"
STATUS_REPAIRED = "repaired"
STATUS_REPAIR_FAILED = "repair_failed"

_DISCRETE: frozenset[VarType] = frozenset({VarType.BINARY, VarType.INTEGER})

_BYTES_PER_MIB: float = 1024.0 * 1024.0

# SCIP termination statuses that mean "stopped early by a resource budget without
# an optimality proof" — i.e. right-censored. Mirrors ``opop.solver.scip`` so the
# heuristic sub-solves agree with the kernel on the ``censored`` flag.
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

# A single shared kernel reused purely for its (public) phi.p param channel; each
# heuristic builds its own fresh PySCIPOpt model so this is safe to share.
_KERNEL = ScipKernel()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HeuristicResult:
    """Outcome of one matheuristic call.

    Attributes:
        incumbent: The best **feasible** assignment found (``var name -> value``),
            or ``None`` when no feasible solution could be verified (a failed run).
        objective: Original-objective value of ``incumbent`` (``math.nan`` when
            ``incumbent`` is ``None``).
        improved: ``True`` iff the returned incumbent is strictly preferable to the
            input (strictly better objective, or feasible where the input was not).
        feasible: ``True`` iff ``incumbent`` passed the IR feasibility check.
        status: One of ``improved`` / ``no_improvement`` / ``infeasible`` /
            ``repaired`` / ``repair_failed``.
        traces: The :class:`~opop.model.state.SolveTrace` of each inner solve, in
            execution order (LP relaxation + restricted MIP for RINS; one per LNS
            iteration; a single solve for local branching / repair).
        info: Heuristic-specific diagnostics (``k`` / ``n_fixed`` / ``n_accepted``
            / ``distance`` / ...).
    """

    incumbent: dict[str, float] | None
    objective: float
    improved: bool
    feasible: bool
    status: str
    traces: tuple[SolveTrace, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict (non-finite ``objective`` → ``None``)."""
        obj: float | None = self.objective if math.isfinite(self.objective) else None
        return {
            "incumbent": self.incumbent,
            "objective": obj,
            "improved": self.improved,
            "feasible": self.feasible,
            "status": self.status,
            "n_traces": len(self.traces),
            "trace_statuses": [t.status for t in self.traces],
            "info": self.info,
        }


# ---------------------------------------------------------------------------
# Feasibility / objective evaluation (pure; no solver)
# ---------------------------------------------------------------------------
def solution_violations(
    ir: MILP,
    assignment: Mapping[str, float],
    *,
    feas_tol: float = FEAS_TOL,
    int_tol: float = INT_TOL,
) -> list[str]:
    """Return human-readable feasibility violations of ``assignment`` w.r.t. ``ir``.

    Checks (within tolerance): every variable is assigned, lies within its bounds,
    and is integral when BINARY/INTEGER; and every linear constraint is satisfied
    for its sense. An empty list means feasible.
    """
    violations: list[str] = []
    for var in ir.variables:
        if var.name not in assignment:
            violations.append(f"variable {var.name!r} is not assigned")
            continue
        val = float(assignment[var.name])
        if val < var.lower - feas_tol or val > var.upper + feas_tol:
            violations.append(
                f"variable {var.name!r} value {val} out of bounds [{var.lower}, {var.upper}]"
            )
        if var.vtype in _DISCRETE and abs(val - round(val)) > int_tol:
            violations.append(f"variable {var.name!r} value {val} is not integral")

    for con in ir.constraints:
        missing = [n for n in con.coeffs if n not in assignment]
        if missing:
            violations.append(
                f"constraint {con.name!r} references unassigned variables: {sorted(missing)}"
            )
            continue
        lhs = math.fsum(coeff * float(assignment[n]) for n, coeff in con.coeffs.items())
        if con.sense is ConstraintSense.LE and lhs > con.rhs + feas_tol:
            violations.append(f"constraint {con.name!r}: lhs {lhs} > rhs {con.rhs}")
        elif con.sense is ConstraintSense.GE and lhs < con.rhs - feas_tol:
            violations.append(f"constraint {con.name!r}: lhs {lhs} < rhs {con.rhs}")
        elif con.sense is ConstraintSense.EQ and abs(lhs - con.rhs) > feas_tol:
            violations.append(f"constraint {con.name!r}: |lhs {lhs} - rhs {con.rhs}| > tol")
    return violations


def is_solution_feasible(
    ir: MILP,
    assignment: Mapping[str, float],
    *,
    feas_tol: float = FEAS_TOL,
    int_tol: float = INT_TOL,
) -> bool:
    """Return ``True`` iff ``assignment`` satisfies every IR bound/integrality/constraint."""
    return not solution_violations(ir, assignment, feas_tol=feas_tol, int_tol=int_tol)


def _objective_value(ir: MILP, assignment: Mapping[str, float]) -> float:
    """Evaluate the IR's original objective at ``assignment`` (includes the offset)."""
    total = ir.objective.offset
    for name, coeff in ir.objective.coeffs.items():
        total += coeff * float(assignment[name])
    return total


def _is_strictly_better(new: float, old: float, sense: ObjSense, tol: float = OBJ_TOL) -> bool:
    """``True`` iff ``new`` strictly beats ``old`` for the optimisation ``sense``."""
    if sense is ObjSense.MINIMIZE:
        return new < old - tol
    return new > old + tol


def _require_complete(ir: MILP, incumbent: Mapping[str, float]) -> None:
    """Raise :class:`ValueError` unless ``incumbent`` assigns every IR variable."""
    missing = [v.name for v in ir.variables if v.name not in incumbent]
    if missing:
        raise ValueError(f"incumbent is missing values for variables: {sorted(missing)}")


# ---------------------------------------------------------------------------
# Inner MIP/LP solve (reuses ScipKernel's public param channel + conventions)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SubSolve:
    """Result of one inner SCIP solve: the solution vector + a :class:`SolveTrace`."""

    solution: dict[str, float] | None
    objective: float
    status: str
    trace: SolveTrace


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


def _clean_value(value: float, vtype: VarType) -> float:
    """Round a discrete variable's solved value to the nearest integer."""
    if vtype in _DISCRETE:
        return float(round(value))
    return float(value)


def _solve_milp(
    ir: MILP,
    *,
    extract_names: Sequence[str],
    phi: Phi,
    time_limit: float,
    memory_limit_mb: int,
    seed: int,
) -> _SubSolve:
    """Compile ``ir`` to PySCIPOpt, solve under the budget, extract a solution + trace.

    Determinism mirrors :class:`opop.solver.scip.ScipKernel`: ``phi.p`` params are
    applied first (via the kernel's whitelisted hook), then the authoritative
    single-threaded / time / memory / seed knobs. ``extract_names`` selects which
    variables (the *original* problem variables, excluding any neighbourhood
    auxiliaries) are read back into ``solution``.
    """
    model = to_pyscipopt(ir)
    model.hideOutput()
    _KERNEL.apply_proposer_hooks(model, phi)
    model.setIntParam("lp/threads", 1)
    model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/memory", float(memory_limit_mb))
    model.setIntParam("randomization/randomseedshift", int(seed))

    model.optimize()

    status = str(model.getStatus())
    solving_time = float(model.getSolvingTime())
    primal = _finite_or_inf(model, model.getPrimalbound())
    dual = _finite_or_inf(model, model.getDualbound())

    solution: dict[str, float] | None = None
    objective = math.nan
    sol = model.getBestSol()
    if sol is not None:
        scip_vars = {v.name: v for v in model.getVars()}
        vtype_of = {v.name: v.vtype for v in ir.variables}
        extracted: dict[str, float] = {}
        for name in extract_names:
            raw = float(model.getSolVal(sol, scip_vars[name]))
            extracted[name] = _clean_value(raw, vtype_of[name])
        solution = extracted
        objective = float(model.getSolObjVal(sol))

    trace = SolveTrace(
        primal_bound_series=[primal],
        dual_bound_series=[dual],
        time_series=[solving_time],
        nodes=int(model.getNTotalNodes()),
        lp_iters=int(model.getNLPIterations()),
        cuts=int(model.getNCutsApplied()),
        # Heuristic sub-solves register no trajectory event handler; the
        # first-feasible timestamp is not tracked → math.nan sentinel.
        first_feasible_time=math.nan,
        status=status,
        censored=_is_censored(status),
        memory_peak=float(model.getMemUsed()) / _BYTES_PER_MIB,
        instance_id=ir.name,
        solver=_KERNEL.solver_name,
    )
    return _SubSolve(solution=solution, objective=objective, status=status, trace=trace)


# ---------------------------------------------------------------------------
# Neighbourhood IR builders (pure; dataclasses.replace over fresh copies)
# ---------------------------------------------------------------------------
def _lp_relaxation(ir: MILP) -> MILP:
    """Return the LP relaxation: every variable continuous (binary keeps [0, 1])."""
    relaxed = tuple(
        v if v.vtype is VarType.CONTINUOUS else replace(v, vtype=VarType.CONTINUOUS)
        for v in ir.variables
    )
    return replace(ir, variables=relaxed)


def _build_local_branching_ir(ir: MILP, incumbent: Mapping[str, float], k: int) -> MILP:
    """Add a Hamming-distance ``sum |x - x_bar| <= k`` constraint over discrete vars.

    Binary variables are linearised exactly (``x`` if ``x_bar==0`` else ``1-x``);
    INTEGER variables use one auxiliary continuous ``d >= |x - x_bar|`` each.
    """
    discrete = [v for v in ir.variables if v.vtype in _DISCRETE]
    if not discrete:
        raise ValueError("local_branching requires at least one BINARY or INTEGER variable")

    lb_coeffs: dict[str, float] = {}
    rhs = float(k)
    extra_vars: list[Variable] = []
    extra_cons: list[LinearConstraint] = []
    for var in discrete:
        target = float(incumbent[var.name])
        if var.vtype is VarType.BINARY:
            if round(target) <= 0:
                lb_coeffs[var.name] = lb_coeffs.get(var.name, 0.0) + 1.0
            else:
                lb_coeffs[var.name] = lb_coeffs.get(var.name, 0.0) - 1.0
                rhs -= 1.0
        else:  # INTEGER: d >= x - target  AND  d >= target - x
            aux = f"_opop_lb_abs_{var.name}"
            extra_vars.append(Variable(name=aux, vtype=VarType.CONTINUOUS, lower=0.0, upper=math.inf))
            extra_cons.append(
                LinearConstraint(
                    name=f"_opop_lb_pos_{var.name}",
                    coeffs={aux: 1.0, var.name: -1.0},
                    sense=ConstraintSense.GE,
                    rhs=-target,
                )
            )
            extra_cons.append(
                LinearConstraint(
                    name=f"_opop_lb_neg_{var.name}",
                    coeffs={aux: 1.0, var.name: 1.0},
                    sense=ConstraintSense.GE,
                    rhs=target,
                )
            )
            lb_coeffs[aux] = 1.0

    lb_con = LinearConstraint(
        name="_opop_local_branching", coeffs=lb_coeffs, sense=ConstraintSense.LE, rhs=rhs
    )
    return replace(
        ir,
        variables=ir.variables + tuple(extra_vars),
        constraints=ir.constraints + tuple(extra_cons) + (lb_con,),
    )


def _fix_agreeing_vars(
    ir: MILP, incumbent: Mapping[str, float], lp_solution: Mapping[str, float], tol: float
) -> tuple[MILP, int]:
    """Fix each discrete var whose LP value agrees with the incumbent (RINS)."""
    new_vars: list[Variable] = []
    n_fixed = 0
    for var in ir.variables:
        if var.vtype in _DISCRETE and var.name in lp_solution and var.name in incumbent:
            lp_val = float(lp_solution[var.name])
            inc_val = float(incumbent[var.name])
            if abs(lp_val - inc_val) <= tol:
                new_vars.append(replace(var, lower=inc_val, upper=inc_val))
                n_fixed += 1
                continue
        new_vars.append(var)
    return replace(ir, variables=tuple(new_vars)), n_fixed


def _fix_except(ir: MILP, incumbent: Mapping[str, float], free_names: Sequence[str]) -> MILP:
    """Fix every discrete var to its incumbent value except those in ``free_names`` (LNS)."""
    free = set(free_names)
    new_vars: list[Variable] = []
    for var in ir.variables:
        if var.vtype in _DISCRETE and var.name not in free:
            val = float(incumbent[var.name])
            new_vars.append(replace(var, lower=val, upper=val))
        else:
            new_vars.append(var)
    return replace(ir, variables=tuple(new_vars))


def _build_repair_ir(ir: MILP, partial: Mapping[str, float]) -> MILP:
    """Replace the objective with ``minimise sum |x_j - a_j|`` over assigned vars ``j``.

    Original constraints and bounds are preserved, so any optimum is feasible for
    ``ir``; one auxiliary continuous ``d_j >= |x_j - a_j|`` is added per target.
    """
    known = {v.name for v in ir.variables}
    unknown = set(partial) - known
    if unknown:
        raise ValueError(
            f"partial_assignment references unknown variables: {sorted(unknown)}"
        )

    extra_vars: list[Variable] = []
    extra_cons: list[LinearConstraint] = []
    obj_coeffs: dict[str, float] = {}
    for name in partial:
        target = float(partial[name])
        aux = f"_opop_repair_abs_{name}"
        extra_vars.append(Variable(name=aux, vtype=VarType.CONTINUOUS, lower=0.0, upper=math.inf))
        extra_cons.append(
            LinearConstraint(
                name=f"_opop_repair_pos_{name}",
                coeffs={aux: 1.0, name: -1.0},
                sense=ConstraintSense.GE,
                rhs=-target,
            )
        )
        extra_cons.append(
            LinearConstraint(
                name=f"_opop_repair_neg_{name}",
                coeffs={aux: 1.0, name: 1.0},
                sense=ConstraintSense.GE,
                rhs=target,
            )
        )
        obj_coeffs[aux] = 1.0

    repair_obj = Objective(coeffs=obj_coeffs, sense=ObjSense.MINIMIZE, offset=0.0)
    return replace(
        ir,
        variables=ir.variables + tuple(extra_vars),
        constraints=ir.constraints + tuple(extra_cons),
        objective=repair_obj,
    )


# ---------------------------------------------------------------------------
# Shared result finalisation for the search heuristics (LB / RINS / LNS)
# ---------------------------------------------------------------------------
def _finalize_search_result(
    ir: MILP,
    incumbent: Mapping[str, float],
    candidate: dict[str, float] | None,
    *,
    traces: Sequence[SolveTrace],
    info: dict[str, Any],
) -> HeuristicResult:
    """Pick the better of input/candidate (both feasibility-checked) and package it.

    The returned ``incumbent`` is always either a verified-feasible assignment or
    ``None`` (with ``feasible=False`` + ``status=infeasible``); an unchecked or
    infeasible assignment is never returned.
    """
    sense = ir.objective.sense
    input_obj = _objective_value(ir, incumbent)
    input_feasible = is_solution_feasible(ir, incumbent)

    best: dict[str, float] = {k: float(v) for k, v in incumbent.items()}
    best_obj = input_obj
    best_feasible = input_feasible
    improved = False

    if candidate is not None and is_solution_feasible(ir, candidate):
        cand_obj = _objective_value(ir, candidate)
        if (not input_feasible) or _is_strictly_better(cand_obj, input_obj, sense):
            best = dict(candidate)
            best_obj = cand_obj
            best_feasible = True
            improved = True

    if improved:
        status = STATUS_IMPROVED
    elif best_feasible:
        status = STATUS_NO_IMPROVEMENT
    else:
        status = STATUS_INFEASIBLE

    return HeuristicResult(
        incumbent=best if best_feasible else None,
        objective=best_obj if best_feasible else math.nan,
        improved=improved,
        feasible=best_feasible,
        status=status,
        traces=tuple(traces),
        info=info,
    )


# ---------------------------------------------------------------------------
# Public heuristic cores
# ---------------------------------------------------------------------------
def local_branching(
    ir: MILP,
    incumbent: Mapping[str, float],
    k: int,
    time_limit: float,
    seed: int,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    phi: Phi | None = None,
) -> HeuristicResult:
    """Local branching: solve the ``sum |x - incumbent| <= k`` neighbourhood MIP.

    Args:
        ir: The MILP whose original objective is optimised over the neighbourhood.
        incumbent: A complete assignment (every IR variable) defining the centre.
        k: Hamming radius (``>= 0``); ``0`` admits only the incumbent itself.
        time_limit: Wall-clock ceiling (seconds) for the neighbourhood solve.
        seed: SCIP master seed (determinism).
        memory_limit_mb: Hard memory ceiling for the solve.
        phi: Optional design vector; its ``p`` params are applied to the solve.

    Returns:
        A :class:`HeuristicResult` whose incumbent is the better (feasibility-checked)
        of the input and the neighbourhood optimum.
    """
    if k < 0:
        raise ValueError(f"local_branching radius k must be >= 0, got {k}")
    _require_complete(ir, incumbent)
    phi = phi if phi is not None else Phi()

    nbr_ir = _build_local_branching_ir(ir, incumbent, k)
    sub = _solve_milp(
        nbr_ir,
        extract_names=ir.var_names(),
        phi=phi,
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        seed=seed,
    )
    info: dict[str, Any] = {"k": k, "neighborhood_status": sub.status}
    return _finalize_search_result(ir, incumbent, sub.solution, traces=(sub.trace,), info=info)


def rins(
    ir: MILP,
    incumbent: Mapping[str, float],
    time_limit: float,
    seed: int,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    agreement_tol: float = INT_TOL,
    phi: Phi | None = None,
) -> HeuristicResult:
    """RINS: fix discrete vars where the LP relaxation agrees with the incumbent.

    Solves the LP relaxation, fixes each discrete variable whose LP value matches
    the incumbent value within ``agreement_tol``, then solves the restricted MIP.

    Returns:
        A :class:`HeuristicResult`; ``info`` carries ``n_fixed`` and ``lp_objective``.
        If the LP relaxation yields no usable solution the input incumbent is
        returned unchanged (``info['reason'] == 'lp_not_solved'``).
    """
    _require_complete(ir, incumbent)
    phi = phi if phi is not None else Phi()

    lp_sub = _solve_milp(
        _lp_relaxation(ir),
        extract_names=ir.var_names(),
        phi=phi,
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        seed=seed,
    )
    if lp_sub.solution is None:
        info_lp: dict[str, Any] = {"reason": "lp_not_solved", "lp_status": lp_sub.status}
        return _finalize_search_result(ir, incumbent, None, traces=(lp_sub.trace,), info=info_lp)

    restricted_ir, n_fixed = _fix_agreeing_vars(ir, incumbent, lp_sub.solution, agreement_tol)
    sub = _solve_milp(
        restricted_ir,
        extract_names=ir.var_names(),
        phi=phi,
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        seed=seed,
    )
    info: dict[str, Any] = {
        "n_fixed": n_fixed,
        "n_discrete": sum(1 for v in ir.variables if v.vtype in _DISCRETE),
        "lp_objective": lp_sub.objective,
        "lp_status": lp_sub.status,
        "restricted_status": sub.status,
    }
    return _finalize_search_result(
        ir, incumbent, sub.solution, traces=(lp_sub.trace, sub.trace), info=info
    )


def large_neighborhood_search(
    ir: MILP,
    incumbent: Mapping[str, float],
    destroy_frac: float,
    n_iter: int,
    time_limit: float,
    seed: int,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    phi: Phi | None = None,
) -> HeuristicResult:
    """Generic destroy/repair LNS over the discrete variables.

    Each of ``n_iter`` iterations frees ``ceil(destroy_frac * n_discrete)`` randomly
    chosen discrete variables (seeded RNG → deterministic), fixes the rest to the
    current incumbent, and repairs by a ``time_limit``-second MIP solve; a feasible,
    strictly-improving repair is accepted. ``time_limit`` is the *per-iteration*
    sub-solve budget.

    Returns:
        A :class:`HeuristicResult` whose incumbent is never worse than the input;
        ``info`` carries ``n_accepted`` / ``n_destroy`` / ``n_iter``.
    """
    if not 0.0 < destroy_frac <= 1.0:
        raise ValueError(f"destroy_frac must be in (0, 1], got {destroy_frac}")
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    _require_complete(ir, incumbent)
    phi = phi if phi is not None else Phi()

    discrete = [v.name for v in ir.variables if v.vtype in _DISCRETE]
    if not discrete:
        raise ValueError("large_neighborhood_search requires at least one discrete variable")

    rng = random.Random(seed)
    n_destroy = min(len(discrete), max(1, round(destroy_frac * len(discrete))))

    current: dict[str, float] = {k: float(v) for k, v in incumbent.items()}
    current_feasible = is_solution_feasible(ir, current)
    current_obj = _objective_value(ir, current)
    sense = ir.objective.sense

    traces: list[SolveTrace] = []
    n_accepted = 0
    for i in range(n_iter):
        free_names = rng.sample(discrete, n_destroy)
        sub = _solve_milp(
            _fix_except(ir, current, free_names),
            extract_names=ir.var_names(),
            phi=phi,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            seed=seed + i + 1,
        )
        traces.append(sub.trace)
        cand = sub.solution
        if cand is not None and is_solution_feasible(ir, cand):
            cand_obj = _objective_value(ir, cand)
            if (not current_feasible) or _is_strictly_better(cand_obj, current_obj, sense):
                current = cand
                current_obj = cand_obj
                current_feasible = True
                n_accepted += 1

    info: dict[str, Any] = {
        "n_iter": n_iter,
        "n_accepted": n_accepted,
        "destroy_frac": destroy_frac,
        "n_destroy": n_destroy,
        "n_discrete": len(discrete),
    }
    candidate = current if current_feasible else None
    return _finalize_search_result(ir, incumbent, candidate, traces=traces, info=info)


def repair_solution(
    ir: MILP,
    partial_assignment: Mapping[str, float],
    time_limit: float,
    seed: int,
    *,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    phi: Phi | None = None,
) -> HeuristicResult:
    """Repair: find the feasible solution closest to ``partial_assignment``.

    Solves a feasibility MIP over the original constraints whose objective is
    ``minimise sum |x_j - a_j|`` over the assigned variables ``j``. The result is
    either a verified-feasible incumbent (``status='repaired'``) or a reported
    failure (``incumbent=None``, ``feasible=False``, ``status='repair_failed'``)
    when the model admits no feasible solution — never an unchecked infeasible one.

    Returns:
        A :class:`HeuristicResult`; ``info['distance']`` is the achieved
        ``sum |x_j - a_j|`` when a feasible solution is found.
    """
    phi = phi if phi is not None else Phi()

    repair_ir = _build_repair_ir(ir, partial_assignment)
    sub = _solve_milp(
        repair_ir,
        extract_names=ir.var_names(),
        phi=phi,
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        seed=seed,
    )
    candidate = sub.solution
    if candidate is not None and is_solution_feasible(ir, candidate):
        info: dict[str, Any] = {"distance": sub.objective, "solve_status": sub.status}
        return HeuristicResult(
            incumbent=candidate,
            objective=_objective_value(ir, candidate),
            improved=True,
            feasible=True,
            status=STATUS_REPAIRED,
            traces=(sub.trace,),
            info=info,
        )
    fail_info: dict[str, Any] = {"reason": "no_feasible_solution", "solve_status": sub.status}
    return HeuristicResult(
        incumbent=None,
        objective=math.nan,
        improved=False,
        feasible=False,
        status=STATUS_REPAIR_FAILED,
        traces=(sub.trace,),
        info=fail_info,
    )
