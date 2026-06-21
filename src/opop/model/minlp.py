"""Structured MINLP adapter for the separable / factorable subset (task 31).

:class:`StructuredMinlpAdapter` is the :class:`~opop.model.adapter.ProblemClassAdapter`
for the *decomposable* slice of mixed-integer **nonlinear** programs: objectives
and constraints that are a sum of **separable univariate** nonlinear terms
``coeff * f(x_j)`` where ``f`` is one of a small, curvature-known set
(:data:`SUPPORTED_FUNCTIONS` = ``square``/``exp`` convex, ``log``/``sqrt``
concave) on a **bounded** variable interval. This is exactly the structure that
classical *outer approximation* (Duran--Grossmann) linearises with supporting
hyperplanes, and that decomposes cleanly into blocks (the GCG / Benders link of
task 24).

Representation (additive, IR-validation-safe)
---------------------------------------------
The nonlinear layer rides on ``MILP.metadata[``:data:`NONLINEAR_TERMS_KEY```]`` as
a tuple of typed records, mirroring the *additive* pattern of the task-30
:class:`~opop.model.ir.QuadraticExtension` without touching the IR's
referential-integrity validation (metadata is free-form and not serialised to
MPS). Two records are defined here:

* :class:`NonlinearTerm` — a supported separable univariate term ``coeff * f(var)``
  attached to the objective (:data:`OBJECTIVE_TARGET`) or a named constraint.
* :class:`BilinearTerm` — a product ``coeff * var1 * var2`` of *distinct*
  variables. It is **never** in the supported subset (a product of two variables
  is not factorable into univariate convex/concave pieces) and is the canonical
  "offending term" the adapter rejects.

Two routes (the adapter contract)
---------------------------------
* :meth:`StructuredMinlpAdapter.to_milp` — build an **outer-approximation MILP**:
  each separable term gets a fresh continuous auxiliary ``u`` pinned by tangent
  (gradient) cuts ``u (>=|<=) f(a) + f'(a)(x - a)`` sampled at breakpoints, and
  the term ``coeff * f(var)`` is replaced by the linear ``coeff * u``. For a
  convex term the tangents under-estimate ``f`` (``u >= ...``); for a concave
  term they over-estimate it (``u <= ...``); in both cases the resulting MILP is
  a **valid relaxation**, and it is **exact at the sampled breakpoints** — so for
  an INTEGER variable sampled at every integer the OA-MILP optimum equals the
  MINLP optimum. The OA-MILP is purely linear and routes to any linear kernel.
* :meth:`StructuredMinlpAdapter.native_solve` — build the genuine nonlinear model
  with PySCIPOpt's expression API (``x*x`` / ``exp`` / ``log`` / ``sqrt``; a
  nonlinear objective is pinned to a free auxiliary because SCIP rejects
  nonlinear objectives) and solve it on SCIP under the determinism budget. A
  non-SCIP kernel is rejected fail-closed (CP-SAT / HiGHS cannot represent the
  structure natively).

Anything outside the subset — an unsupported function, a bilinear/multivariate
product, a non-finite variable bound, a curvature that is nonconvex in its
placement (e.g. a convex term in a ``>=`` constraint, or a concave term in a
MINIMIZE objective) — raises :class:`~opop.model.ir.UnsupportedModelError` naming
the offending term, never a silent (and optimistically wrong) relaxation.

This module is pure model-layer at import time: the only solver / analyzer
imports (PySCIPOpt, :class:`opop.solver.scip.ScipKernel`,
:func:`opop.analyzer.decompose.detect_decomposition`) are **function-local**, so
``import opop.model.minlp`` never pulls a solver backend. The adapter
self-registers on import (like the task-30 solver-layer adapters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, final

from opop.model.adapter import AdapterCapabilities, register_adapter
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    UnsupportedModelError,
    Variable,
    VarType,
)
from opop.model.state import Phi, SolveTrace

if TYPE_CHECKING:
    from opop.analyzer.decompose import DecompositionReport
    from opop.solver.kernel import SolverKernel

__all__ = [
    "CONCAVE_FUNCTIONS",
    "CONVEX_FUNCTIONS",
    "NONLINEAR_TERMS_KEY",
    "OBJECTIVE_TARGET",
    "SUPPORTED_FUNCTIONS",
    "BilinearTerm",
    "NonlinearTerm",
    "StructuredMinlpAdapter",
    "solve_scip_minlp",
]

#: Metadata key carrying the tuple of nonlinear term records.
NONLINEAR_TERMS_KEY = "nonlinear_terms"

#: Sentinel ``target`` meaning "this term augments the objective".
OBJECTIVE_TARGET = "__objective__"

#: Convex univariate functions (supported in a MINIMIZE objective or a ``<=`` row).
CONVEX_FUNCTIONS: frozenset[str] = frozenset({"square", "exp"})

#: Concave univariate functions (supported in a MAXIMIZE objective or a ``>=`` row).
CONCAVE_FUNCTIONS: frozenset[str] = frozenset({"log", "sqrt"})

#: Every supported separable univariate function.
SUPPORTED_FUNCTIONS: frozenset[str] = CONVEX_FUNCTIONS | CONCAVE_FUNCTIONS

# Outer-approximation breakpoint budgets.
_CONTINUOUS_BREAKPOINTS = 8
_MAX_INTEGER_BREAKPOINTS = 64

# Free continuous auxiliary carrying a nonlinear objective value (SCIP only
# accepts a LINEAR objective, so the nonlinear objective is pinned to this var).
_OBJ_AUX = "_opop_minlp_obj_aux"
_OBJ_AUX_DEF = "_opop_minlp_obj_aux_def"

_BYTES_PER_MIB = 1024.0 * 1024.0

_VTYPE_TO_SCIP: dict[VarType, str] = {
    VarType.BINARY: "B",
    VarType.INTEGER: "I",
    VarType.CONTINUOUS: "C",
}

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


# ---------------------------------------------------------------------------
# Nonlinear term records (additive metadata layer)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NonlinearTerm:
    """A separable univariate nonlinear term ``coeff * func(var)``.

    Attributes:
        func: One of :data:`SUPPORTED_FUNCTIONS` (``square``/``exp`` convex,
            ``log``/``sqrt`` concave).
        var: The single variable name the function is applied to.
        coeff: Positive real coefficient (a non-positive coefficient flips the
            curvature and is rejected as nonconvex).
        target: :data:`OBJECTIVE_TARGET` to augment the objective, or a declared
            constraint name to augment that constraint's LHS.
    """

    func: str
    var: str
    coeff: float = 1.0
    target: str = OBJECTIVE_TARGET


@dataclass(frozen=True, slots=True)
class BilinearTerm:
    """A product ``coeff * var1 * var2`` of two variables (``var1 != var2``).

    Always outside the separable subset: the adapter rejects it with
    :class:`~opop.model.ir.UnsupportedModelError`. It exists so callers can hand
    the adapter a genuine out-of-subset MINLP and get a typed, named rejection.

    Attributes:
        var1: First variable name.
        var2: Second variable name.
        coeff: Product coefficient.
        target: :data:`OBJECTIVE_TARGET` or a declared constraint name.
    """

    var1: str
    var2: str
    coeff: float = 1.0
    target: str = OBJECTIVE_TARGET


# ---------------------------------------------------------------------------
# Univariate function evaluation + derivatives
# ---------------------------------------------------------------------------
def _f(func: str, x: float) -> float:
    """Evaluate the supported univariate ``func`` at ``x``."""
    if func == "square":
        return x * x
    if func == "exp":
        return math.exp(x)
    if func == "log":
        return math.log(x)
    if func == "sqrt":
        return math.sqrt(x)
    raise ValueError(f"unsupported function {func!r}")


def _fprime(func: str, x: float) -> float:
    """Evaluate the derivative of the supported univariate ``func`` at ``x``."""
    if func == "square":
        return 2.0 * x
    if func == "exp":
        return math.exp(x)
    if func == "log":
        return 1.0 / x
    if func == "sqrt":
        return 0.5 / math.sqrt(x)
    raise ValueError(f"unsupported function {func!r}")


def _tangent(func: str, a: float) -> tuple[float, float]:
    """Return ``(slope, intercept)`` of the tangent of ``func`` at ``a``.

    The tangent line is ``T_a(x) = f(a) + f'(a)(x - a) = slope * x + intercept``.
    """
    slope = _fprime(func, a)
    return slope, _f(func, a) - slope * a


# ---------------------------------------------------------------------------
# Scope detection
# ---------------------------------------------------------------------------
def _collect_terms(ir: MILP) -> tuple[Any, ...]:
    """Read the nonlinear term records off ``ir.metadata`` (``()`` if none)."""
    raw = ir.metadata.get(NONLINEAR_TERMS_KEY)
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        seq: Any = raw
        return tuple(seq)
    return (raw,)


def _var_by_name(ir: MILP) -> dict[str, Variable]:
    return {v.name: v for v in ir.variables}


def _con_by_name(ir: MILP) -> dict[str, LinearConstraint]:
    return {c.name: c for c in ir.constraints}


def _term_problem(
    term: Any,
    ir: MILP,
    var_by_name: dict[str, Variable],
    con_by_name: dict[str, LinearConstraint],
) -> str | None:
    """Return ``None`` if ``term`` is in the supported subset, else why it is not.

    The returned string names the offending term so callers can surface it in an
    :class:`~opop.model.ir.UnsupportedModelError`.
    """
    if isinstance(term, BilinearTerm):
        return (
            f"bilinear product {term.var1!r}*{term.var2!r} (target={term.target!r}) is outside "
            + "the separable subset: a product of two variables is not factorable into "
            + "univariate convex/concave pieces"
        )
    if not isinstance(term, NonlinearTerm):
        return f"unsupported nonlinear record {type(term).__name__!r} (expected NonlinearTerm)"
    if term.func not in SUPPORTED_FUNCTIONS:
        allowed = ", ".join(sorted(SUPPORTED_FUNCTIONS))
        return (
            f"unsupported nonlinear function {term.func!r} on variable {term.var!r}; "
            + f"supported separable univariate functions are: {allowed}"
        )
    if term.var not in var_by_name:
        return f"nonlinear term {term.func}({term.var!r}) references an unknown variable"
    if not math.isfinite(term.coeff) or term.coeff <= 0.0:
        return (
            f"nonlinear term {term.func}({term.var!r}) needs a finite positive coefficient; "
            + f"got {term.coeff!r} (a non-positive coefficient flips the curvature)"
        )
    var = var_by_name[term.var]
    if not (math.isfinite(var.lower) and math.isfinite(var.upper)):
        return (
            f"variable {term.var!r} needs finite bounds for the outer-approximation of "
            + f"{term.func}() (got [{var.lower}, {var.upper}])"
        )
    if term.func in {"log", "sqrt"} and var.lower <= 0.0:
        return (
            f"variable {term.var!r} needs a strictly positive lower bound for {term.func}() "
            + f"(got lower={var.lower})"
        )
    convex = term.func in CONVEX_FUNCTIONS
    if term.target == OBJECTIVE_TARGET:
        if convex and ir.objective.sense is not ObjSense.MINIMIZE:
            return (
                f"convex term {term.func}({term.var!r}) in a MAXIMIZE objective is nonconvex; "
                + "convex terms are supported only in a MINIMIZE objective"
            )
        if (not convex) and ir.objective.sense is not ObjSense.MAXIMIZE:
            return (
                f"concave term {term.func}({term.var!r}) in a MINIMIZE objective is nonconvex; "
                + "concave terms are supported only in a MAXIMIZE objective"
            )
        return None
    con = con_by_name.get(term.target)
    if con is None:
        return f"nonlinear term {term.func}({term.var!r}) targets unknown constraint {term.target!r}"
    if convex and con.sense is not ConstraintSense.LE:
        return (
            f"convex term {term.func}({term.var!r}) in a {con.sense.value!r} constraint defines a "
            + "nonconvex region; convex terms are supported only in '<=' constraints"
        )
    if (not convex) and con.sense is not ConstraintSense.GE:
        return (
            f"concave term {term.func}({term.var!r}) in a {con.sense.value!r} constraint defines a "
            + "nonconvex region; concave terms are supported only in '>=' constraints"
        )
    return None


def _require_supported(ir: MILP, terms: tuple[Any, ...]) -> None:
    """Raise :class:`~opop.model.ir.UnsupportedModelError` on the first bad term."""
    if not terms:
        return
    var_by_name = _var_by_name(ir)
    con_by_name = _con_by_name(ir)
    for term in terms:
        problem = _term_problem(term, ir, var_by_name, con_by_name)
        if problem is not None:
            raise UnsupportedModelError(problem)


# ---------------------------------------------------------------------------
# Outer-approximation MILP construction (to_milp)
# ---------------------------------------------------------------------------
def _breakpoints(var: Variable) -> list[float]:
    """Sample tangent breakpoints across ``var``'s bounded domain.

    Integer / binary variables are sampled at EVERY integer in their range (so the
    outer approximation is *exact* at integer points); continuous variables get an
    even grid. Both partitions are capped to keep the cut count bounded.
    """
    lo, hi = float(var.lower), float(var.upper)
    if var.vtype is VarType.CONTINUOUS:
        if hi <= lo:
            return [lo]
        n = _CONTINUOUS_BREAKPOINTS
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    start = math.ceil(lo - 1e-9)
    end = math.floor(hi + 1e-9)
    if end < start:
        return [lo]
    count = end - start + 1
    if count <= _MAX_INTEGER_BREAKPOINTS:
        return [float(v) for v in range(start, end + 1)]
    span = end - start
    return [
        float(start + round(span * i / (_MAX_INTEGER_BREAKPOINTS - 1)))
        for i in range(_MAX_INTEGER_BREAKPOINTS)
    ]


def _fresh_name(base: str, used: set[str]) -> str:
    """Return ``base`` or a ``base_k`` variant not present in ``used``."""
    if base not in used:
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def _outer_approximate(ir: MILP) -> MILP:
    """Return a linear outer-approximation :class:`~opop.model.ir.MILP` of ``ir``.

    Each separable term ``coeff * f(var)`` becomes ``coeff * u`` for a fresh
    continuous auxiliary ``u`` constrained by tangent cuts (``u >= T_a(var)`` for a
    convex ``f``; ``u <= T_a(var)`` for a concave ``f``) at sampled breakpoints.
    Raises :class:`~opop.model.ir.UnsupportedModelError` for any out-of-subset
    term. The input ``ir`` is never mutated.
    """
    terms = _collect_terms(ir)
    base_meta = {k: v for k, v in ir.metadata.items() if k != NONLINEAR_TERMS_KEY}
    if not terms:
        return replace(ir, quadratic=None, metadata=base_meta)
    _require_supported(ir, terms)
    var_by_name = _var_by_name(ir)

    new_vars: list[Variable] = list(ir.variables)
    used_var: set[str] = set(var_by_name)
    used_con: set[str] = {c.name for c in ir.constraints}
    obj_coeffs: dict[str, float] = dict(ir.objective.coeffs)
    con_extra: dict[str, dict[str, float]] = {}
    oa_cons: list[LinearConstraint] = []

    for idx, term in enumerate(terms):
        aux = _fresh_name(f"_oa_{term.func}_{term.var}_{idx}", used_var)
        used_var.add(aux)
        new_vars.append(Variable(aux, VarType.CONTINUOUS, -math.inf, math.inf))
        convex = term.func in CONVEX_FUNCTIONS
        tan_sense = ConstraintSense.GE if convex else ConstraintSense.LE
        for bp_idx, a in enumerate(_breakpoints(var_by_name[term.var])):
            slope, intercept = _tangent(term.func, a)
            coeffs: dict[str, float] = {aux: 1.0}
            if slope != 0.0:
                coeffs[term.var] = -slope
            cname = _fresh_name(f"_oa_cut_{aux}_{bp_idx}", used_con)
            used_con.add(cname)
            oa_cons.append(LinearConstraint(cname, coeffs, tan_sense, intercept))
        if term.target == OBJECTIVE_TARGET:
            obj_coeffs[aux] = obj_coeffs.get(aux, 0.0) + term.coeff
        else:
            bucket = con_extra.setdefault(term.target, {})
            bucket[aux] = bucket.get(aux, 0.0) + term.coeff

    new_constraints: list[LinearConstraint] = []
    for con in ir.constraints:
        extra = con_extra.get(con.name)
        if extra:
            merged = dict(con.coeffs)
            for name, coeff in extra.items():
                merged[name] = merged.get(name, 0.0) + coeff
            new_constraints.append(LinearConstraint(con.name, merged, con.sense, con.rhs))
        else:
            new_constraints.append(con)
    new_constraints.extend(oa_cons)

    objective = Objective(
        coeffs=obj_coeffs, sense=ir.objective.sense, offset=ir.objective.offset
    )
    return MILP(
        name=ir.name,
        variables=tuple(new_vars),
        constraints=tuple(new_constraints),
        objective=objective,
        index_sets=ir.index_sets,
        metadata={**base_meta, "linearization": "outer_approximation"},
        quadratic=None,
    )


# ---------------------------------------------------------------------------
# Native SCIP solve (explicit nonlinear expressions)
# ---------------------------------------------------------------------------
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


def _build_scip_minlp_model(ir: MILP) -> Any:
    """Build a fresh PySCIPOpt model carrying ``ir``'s linear + nonlinear terms.

    Nonlinear constraint terms augment their constraint's LHS; a nonlinear
    objective is pinned to the free auxiliary ``_OBJ_AUX`` (SCIP rejects nonlinear
    objectives). The model is NOT solved and output is NOT suppressed.
    """
    from pyscipopt import Model, exp, log, quicksum, sqrt

    terms = _collect_terms(ir)
    _require_supported(ir, terms)
    if _OBJ_AUX in {v.name for v in ir.variables}:
        raise UnsupportedModelError(
            f"variable name {_OBJ_AUX!r} is reserved for the MINLP objective auxiliary"
        )

    model = Model(ir.name or "opop_minlp")
    scip_vars: dict[str, Any] = {}
    for var in ir.variables:
        scip_vars[var.name] = model.addVar(
            name=var.name,
            vtype=_VTYPE_TO_SCIP[var.vtype],
            lb=_to_scip_bound(model, var.lower),
            ub=_to_scip_bound(model, var.upper),
        )

    def nl_expr(term: NonlinearTerm) -> Any:
        x = scip_vars[term.var]
        if term.func == "square":
            base = x * x
        elif term.func == "exp":
            base = exp(x)
        elif term.func == "log":
            base = log(x)
        else:
            base = sqrt(x)
        return term.coeff * base

    obj_terms = [t for t in terms if t.target == OBJECTIVE_TARGET]
    con_terms: dict[str, list[NonlinearTerm]] = {}
    for t in terms:
        if t.target != OBJECTIVE_TARGET:
            con_terms.setdefault(t.target, []).append(t)

    for con in ir.constraints:
        expr = quicksum(coeff * scip_vars[name] for name, coeff in con.coeffs.items())
        for term in con_terms.get(con.name, ()):
            expr = expr + nl_expr(term)
        if con.sense is ConstraintSense.LE:
            model.addCons(expr <= con.rhs, name=con.name)
        elif con.sense is ConstraintSense.GE:
            model.addCons(expr >= con.rhs, name=con.name)
        else:
            model.addCons(expr == con.rhs, name=con.name)

    obj_linear = quicksum(
        coeff * scip_vars[name] for name, coeff in ir.objective.coeffs.items()
    )
    if obj_terms:
        full = obj_linear
        for term in obj_terms:
            full = full + nl_expr(term)
        aux = model.addVar(
            name=_OBJ_AUX, vtype="C", lb=-model.infinity(), ub=model.infinity()
        )
        target = full + ir.objective.offset
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


def solve_scip_minlp(
    ir: MILP,
    *,
    phi: Phi | None = None,
    time_limit: float = 60.0,
    memory_limit_mb: int = 4096,
    seed: int = 0,
) -> SolveTrace:
    """Build ``ir`` (linear + separable nonlinear) and solve it on SCIP.

    Honours the :class:`~opop.solver.kernel.SolverKernel` determinism contract
    (single LP thread, hard time/memory ceilings, seeded). ``phi.p`` proposer
    params are applied through the public
    :meth:`opop.solver.scip.ScipKernel.apply_proposer_hooks` (separator whitelist,
    fail-closed) BEFORE the authoritative budget knobs. Import / build / solve
    errors propagate (never swallowed). Raises
    :class:`~opop.model.ir.UnsupportedModelError` for out-of-subset terms.
    """
    from opop.solver.scip import ScipKernel

    model = _build_scip_minlp_model(ir)
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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
@final
class StructuredMinlpAdapter:
    """Adapter for the separable / factorable MINLP subset (outer approximation)."""

    @property
    def name(self) -> str:
        """Registry key for this adapter."""
        return "structured_minlp"

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declared capabilities (OA linearization is exact only at breakpoints)."""
        return AdapterCapabilities(
            name="structured_minlp",
            problem_class="MINLP (separable/factorable)",
            handles_quadratic_objective=True,
            handles_quadratic_constraints=True,
            exact_linearization=False,
            native_kernels=("SCIP",),
            linear_kernels=("CP-SAT", "SCIP", "HiGHS"),
        )

    def can_handle(self, ir: MILP) -> bool:
        """Return ``True`` iff ``ir`` declares ONLY supported separable terms.

        Cheap and side-effect free: reads ``ir.metadata`` and classifies each
        declared nonlinear term. A plain linear/quadratic MILP (no
        :data:`NONLINEAR_TERMS_KEY` metadata) is not claimed.
        """
        terms = _collect_terms(ir)
        if not terms:
            return False
        var_by_name = _var_by_name(ir)
        con_by_name = _con_by_name(ir)
        return all(
            _term_problem(term, ir, var_by_name, con_by_name) is None for term in terms
        )

    def to_milp(self, ir: MILP) -> MILP:
        """Return the outer-approximation linear MILP (or raise for bad terms)."""
        return _outer_approximate(ir)

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
        """Solve ``ir`` natively on SCIP (explicit nonlinear). Rejects non-SCIP."""
        if getattr(kernel, "solver_name", "") != "SCIP":
            got = getattr(kernel, "solver_name", type(kernel).__name__)
            raise UnsupportedModelError(
                "structured MINLP native solve requires a SCIP kernel (solver_name='SCIP'); "
                + f"got {got!r}. CP-SAT / HiGHS cannot represent nonlinear structure natively — "
                + "use to_milp() for the outer-approximation MILP instead."
            )
        return solve_scip_minlp(
            ir,
            phi=phi,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            seed=seed,
        )

    def decomposition_report(self, ir: MILP) -> DecompositionReport:
        """Decomposability verdict of the OA MILP (the task-24 GCG/Benders link).

        Separable MINLPs are block-structured: each nonlinear term contributes an
        independent ``(variable, auxiliary)`` block coupled only by the shared
        linear constraints, so :func:`opop.analyzer.decompose.detect_decomposition`
        on the outer-approximation MILP recovers that structure for GCG /
        Benders routing.
        """
        from opop.analyzer.decompose import detect_decomposition

        return detect_decomposition(self.to_milp(ir))


register_adapter(StructuredMinlpAdapter())
