"""Verification gate: classify a :class:`~opop.model.state.Delta` into one of
the A--D delta classes and run the matching, solver-backed certificate BEFORE
any evaluation. Fail-closed.

This is the scientific-integrity keystone of OPOP: no proposed change touches
the main solver/evaluator path unless it is *certified*. The four classes (per
the Verification Strategy) are:

* **A -- equivalent reformulation**: preserves feasible integer solutions and
  the objective. Certified by a *structural* alignment proof (relabel the
  *after* model back into the *before* variable namespace and require an
  identical math model) **plus** a solver confirmation that both models reach
  the same optimum (status + objective within :data:`OBJ_TOL`).
* **B -- valid inequality**: adds a linear constraint that must not remove any
  feasible integer solution of the *before* model. Certified by *solver-backed
  separation*: maximise (for ``<=``) / minimise (for ``>=``) the new
  constraint's left-hand side over the *before* feasible region. If the
  extremum violates the constraint, that optimiser IS a feasible integer point
  the cut removes -- recorded as the ``counterexample`` and the delta is
  rejected. Otherwise no feasible integer point is removed and a certificate is
  recorded.
* **C -- heuristic / search-param**: a semantic no-op. Certified by requiring an
  unchanged math model (variables, bounds, constraints, objective); any
  semantic change is rejected (a semantic change cannot be class C).
* **D -- risky / non-certified**: routed to the sandbox. NEVER returns ``pass``.

Anything unknown or unprovable is **rejected** (fail-closed). The verdict is a
:class:`~opop.verify.certificate.VerificationReport`, emitted to
``verification/report.json`` via :func:`~opop.verify.certificate.write_report`.

Tolerances (locked by the plan): feasibility ``1e-7``, objective ``1e-6``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    apply_delta,
    milp_diffs,
    to_pyscipopt,
)
from opop.model.state import Delta, DeltaClass
from opop.verify.certificate import (
    STATUS_PASS,
    STATUS_REJECT,
    STATUS_SANDBOX,
    VerificationReport,
)

__all__ = [
    "FEAS_TOL",
    "OBJ_TOL",
    "SEPARATION_TIME_LIMIT",
    "verify_delta",
]

#: Feasibility tolerance: a point violates ``a.x (<=|>=) rhs`` only beyond this.
FEAS_TOL: float = 1e-7
#: Objective tolerance for declaring two optima equal (class-A equivalence).
OBJ_TOL: float = 1e-6
#: Wall-clock limit (seconds) for each solver-backed certificate solve.
SEPARATION_TIME_LIMIT: float = 30.0

# Structural equivalence is exact (a rename produces byte-identical coeffs).
_STRUCT_TOL: float = 1e-9


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def verify_delta(
    before_ir: MILP,
    delta: Delta,
    after_ir: MILP | None = None,
    *,
    time_limit: float = SEPARATION_TIME_LIMIT,
) -> VerificationReport:
    """Classify ``delta`` and run the matching certificate; return the verdict.

    If ``after_ir`` is ``None`` it is computed via
    :func:`opop.model.ir.apply_delta`; any failure to apply the delta is a
    fail-closed *reject* (never an exception). Class-D deltas short-circuit to a
    *sandbox* verdict (the delta is never applied and never passes). The
    certificate inspects the ACTUAL ``before -> after`` transformation, with the
    declared class only selecting which contract to enforce.
    """
    declared = delta.declared_class

    # Class D: route to sandbox; never apply, never pass.
    if declared is DeltaClass.D:
        return VerificationReport(
            status=STATUS_SANDBOX,
            delta_class=DeltaClass.D.value,
            feasible_region_integer_preserved=None,
            objective_preserved=None,
            counterexample=None,
            reason=(
                "class-D risky/non-certified delta routed to sandbox; "
                "never enters the main evaluation"
            ),
        )

    class_label = _class_label(declared)

    # Materialise the post-delta model (fail-closed if it cannot be applied).
    if after_ir is None:
        try:
            after_ir = apply_delta(before_ir, delta)
        except Exception as exc:  # noqa: BLE001 -- any failure is fail-closed
            return _reject(
                class_label,
                f"delta could not be applied (fail-closed): {type(exc).__name__}: {exc}",
            )

    if declared is DeltaClass.A:
        return _certify_class_a(before_ir, after_ir, time_limit)
    if declared is DeltaClass.B:
        return _certify_class_b(before_ir, after_ir, time_limit)
    if declared is DeltaClass.C:
        return _certify_class_c(before_ir, after_ir)

    # Unknown declared class (DeltaClass is A/B/C/D, so this is defensive).
    return _reject(class_label, "unknown delta class (fail-closed)")


# ---------------------------------------------------------------------------
# Class A -- equivalent reformulation
# ---------------------------------------------------------------------------
def _certify_class_a(before: MILP, after: MILP, time_limit: float) -> VerificationReport:
    """Certify an equivalent reformulation (structural alignment + solver)."""
    mapping = _infer_var_mapping(before, after)
    if mapping is None:
        return _reject(
            DeltaClass.A.value,
            "class-A: cannot derive a 1-1 variable mapping; equivalence unprovable (fail-closed)",
        )

    # Relabel `after` back into `before`'s variable namespace, then compare.
    inverse = {new: old for old, new in mapping.items()}
    try:
        aligned = _relabel_vars(after, inverse)
    except ValueError as exc:
        return _reject(
            DeltaClass.A.value,
            f"class-A: variable relabel collision; cannot prove equivalence: {exc}",
            feas=False,
        )

    diffs = milp_diffs(before, aligned, tol=_STRUCT_TOL)
    if diffs:
        # Symbolic evidence the after-model is NOT an equivalent reformulation;
        # rejecting on symbolic evidence is fail-closed (we never PASS on it).
        return _reject(
            DeltaClass.A.value,
            "class-A is not an equivalent reformulation: " + "; ".join(diffs[:6]),
            feas=False,
            obj=False,
        )

    # Structurally equivalent -> CONFIRM with the solver (never pass on symbolic alone).
    solve = _solver_equivalence(before, after, time_limit)
    if not solve.ran:
        return _reject(
            DeltaClass.A.value,
            f"class-A solver-backed confirmation unavailable (fail-closed): {solve.detail}",
            feas=True,
            obj=None,
        )
    if not solve.status_match:
        return _reject(
            DeltaClass.A.value,
            (
                "class-A solver contradicts structural equivalence: status "
                f"{solve.status_before!r} (before) vs {solve.status_after!r} (after)"
            ),
            feas=True,
            obj=False,
        )
    if not solve.obj_match:
        return _reject(
            DeltaClass.A.value,
            (
                "class-A objective not preserved within tolerance: "
                f"{solve.obj_before} (before) vs {solve.obj_after} (after)"
            ),
            feas=True,
            obj=False,
        )

    return VerificationReport(
        status=STATUS_PASS,
        delta_class=DeltaClass.A.value,
        feasible_region_integer_preserved=True,
        objective_preserved=True,
        counterexample=None,
        reason=(
            "class-A equivalence certified: structural alignment is identical "
            "and the solver confirms the same optimum"
        ),
        certificate={
            "method": "structural-alignment+solver",
            "variable_mapping": dict(mapping),
            "status_before": solve.status_before,
            "status_after": solve.status_after,
            "objective_before": solve.obj_before,
            "objective_after": solve.obj_after,
            "objective_tolerance": OBJ_TOL,
        },
    )


# ---------------------------------------------------------------------------
# Class B -- valid inequality
# ---------------------------------------------------------------------------
def _certify_class_b(before: MILP, after: MILP, time_limit: float) -> VerificationReport:
    """Certify a valid inequality via solver-backed separation."""
    change = _diff_added_constraints(before, after)
    if change.error is not None:
        return _reject(
            DeltaClass.B.value,
            f"class-B must ONLY add constraints: {change.error}",
        )
    if not change.added:
        return _reject(
            DeltaClass.B.value,
            "class-B declared but no constraint was added",
        )

    separations: list[dict[str, Any]] = []
    for con in change.added:
        result = _separate(before, con, time_limit)
        if result.unprovable:
            return _reject(
                DeltaClass.B.value,
                f"class-B validity unprovable for {con.name!r}: {result.detail}",
                feas=None,
            )
        separations.append(result.as_dict(con))
        if not result.valid:
            return VerificationReport(
                status=STATUS_REJECT,
                delta_class=DeltaClass.B.value,
                feasible_region_integer_preserved=False,
                objective_preserved=True,
                counterexample=result.counterexample,
                reason=(
                    f"class-B inequality {con.name!r} removes a feasible integer "
                    "solution of the before-model (not a valid inequality)"
                ),
                certificate={"separations": separations},
            )

    return VerificationReport(
        status=STATUS_PASS,
        delta_class=DeltaClass.B.value,
        feasible_region_integer_preserved=True,
        objective_preserved=True,
        counterexample=None,
        reason=(
            "class-B valid inequality certified: solver-backed separation proves no "
            "feasible integer solution of the before-model is removed"
        ),
        certificate={"separations": separations},
    )


# ---------------------------------------------------------------------------
# Class C -- heuristic / search-param (semantic no-op)
# ---------------------------------------------------------------------------
def _certify_class_c(before: MILP, after: MILP) -> VerificationReport:
    """Certify a semantic no-op: vars / bounds / constraints / objective unchanged."""
    diffs = milp_diffs(before, after, tol=_STRUCT_TOL)
    if diffs:
        return VerificationReport(
            status=STATUS_REJECT,
            delta_class=DeltaClass.C.value,
            feasible_region_integer_preserved=False,
            objective_preserved=False,
            counterexample=None,
            reason=(
                "class-C must be a semantic no-op but the math model changed (a semantic "
                f"change cannot be class C): {'; '.join(diffs[:6])}"
            ),
        )
    return VerificationReport(
        status=STATUS_PASS,
        delta_class=DeltaClass.C.value,
        feasible_region_integer_preserved=True,
        objective_preserved=True,
        counterexample=None,
        reason=(
            "class-C semantic no-op certified: variables, bounds, constraints, and "
            "objective are unchanged"
        ),
    )


# ---------------------------------------------------------------------------
# Variable mapping / relabel helpers (class A)
# ---------------------------------------------------------------------------
def _infer_var_mapping(before: MILP, after: MILP) -> dict[str, str] | None:
    """Infer a 1-1 ``old -> new`` variable mapping from the before/after var sets.

    Handles the identity case (no rename -> ``{}``) and a single rename
    (``{old: new}``). Returns ``None`` for any ambiguous case (variable count
    changed, or more than one variable renamed) so class A fails closed.
    """
    b_names = [v.name for v in before.variables]
    a_names = [v.name for v in after.variables]
    if len(b_names) != len(a_names):
        return None
    removed = set(b_names) - set(a_names)
    added = set(a_names) - set(b_names)
    if not removed and not added:
        return {}
    if len(removed) == 1 and len(added) == 1:
        return {next(iter(removed)): next(iter(added))}
    return None


def _relabel_vars(ir: MILP, mapping: dict[str, str]) -> MILP:
    """Return a NEW :class:`MILP` with variable names remapped via ``mapping``.

    Names absent from ``mapping`` are left unchanged. The rebuilt :class:`MILP`
    re-validates referential integrity (raising :class:`ValueError` on a name
    collision).
    """
    if not mapping:
        return ir

    def rn(name: str) -> str:
        return mapping.get(name, name)

    new_vars = tuple(replace(v, name=rn(v.name)) for v in ir.variables)
    new_cons = tuple(
        replace(c, coeffs={rn(k): val for k, val in c.coeffs.items()}) for c in ir.constraints
    )
    new_obj = replace(
        ir.objective, coeffs={rn(k): val for k, val in ir.objective.coeffs.items()}
    )
    return replace(ir, variables=new_vars, constraints=new_cons, objective=new_obj)


# ---------------------------------------------------------------------------
# Added-constraint diff (class B)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Change:
    added: list[LinearConstraint]
    error: str | None


def _diff_added_constraints(before: MILP, after: MILP) -> _Change:
    """Return the constraints added by ``after`` relative to ``before``.

    A valid class-B delta ONLY appends constraints. Sets ``error`` (and an empty
    ``added``) if variables, bounds, the objective, or any pre-existing
    constraint changed, or if a constraint was removed.
    """
    b_names = {c.name for c in before.constraints}
    a_names = {c.name for c in after.constraints}

    removed = b_names - a_names
    if removed:
        return _Change([], f"existing constraints removed: {sorted(removed)}")

    added = [c for c in after.constraints if c.name not in b_names]

    # The after-model minus the added constraints must equal the before-model.
    kept = tuple(c for c in after.constraints if c.name in b_names)
    try:
        after_kept = replace(after, constraints=kept)
    except ValueError as exc:
        return _Change([], f"after-model is malformed: {exc}")
    diffs = milp_diffs(before, after_kept, tol=_STRUCT_TOL)
    if diffs:
        return _Change([], "more than added constraints changed: " + "; ".join(diffs[:6]))

    return _Change(added, None)


# ---------------------------------------------------------------------------
# Solver-backed separation (class B)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SepResult:
    valid: bool
    unprovable: bool
    counterexample: dict[str, Any] | None
    detail: str
    extremum: float | None

    def as_dict(self, con: LinearConstraint) -> dict[str, Any]:
        return {
            "constraint": con.name,
            "sense": con.sense.value,
            "rhs": con.rhs,
            "extremal_lhs": self.extremum,
            "feasibility_tolerance": FEAS_TOL,
            "valid": self.valid,
            "detail": self.detail,
        }


def _separate(before: MILP, con: LinearConstraint, time_limit: float) -> _SepResult:
    """Decide whether ``con`` is a valid inequality for ``before``'s integer region.

    Optimises the constraint LHS over the before feasible region: maximise for
    ``<=``, minimise for ``>=``, and both for ``=``. The optimiser is a feasible
    integer point; if it violates ``con`` it is returned as the counterexample.
    """
    coeffs = dict(con.coeffs)

    if con.sense is ConstraintSense.LE:
        out = _scip_optimize(before, coeffs, ObjSense.MAXIMIZE, time_limit)
        return _judge_one_sided(out, con, maximise=True)
    if con.sense is ConstraintSense.GE:
        out = _scip_optimize(before, coeffs, ObjSense.MINIMIZE, time_limit)
        return _judge_one_sided(out, con, maximise=False)

    # Equality: LHS must equal rhs for EVERY feasible integer point.
    out_max = _scip_optimize(before, coeffs, ObjSense.MAXIMIZE, time_limit)
    high = _judge_one_sided(out_max, con, maximise=True)
    if not high.valid or high.unprovable:
        return high
    out_min = _scip_optimize(before, coeffs, ObjSense.MINIMIZE, time_limit)
    return _judge_one_sided(out_min, con, maximise=False)


def _judge_one_sided(
    out: _SolveOutcome, con: LinearConstraint, *, maximise: bool
) -> _SepResult:
    """Turn a single max/min separation solve into a :class:`_SepResult`."""
    if not out.ran:
        return _SepResult(False, True, None, out.detail, None)
    if out.status == "infeasible":
        # No feasible integer point exists -> the cut removes nothing.
        return _SepResult(True, False, None, "before-model infeasible; vacuously valid", None)
    if out.status != "optimal" or out.objective is None:
        return _SepResult(False, True, None, f"separation status {out.status!r}", None)

    lhs = out.objective
    rhs = con.rhs
    if maximise:
        valid = lhs <= rhs + FEAS_TOL
        detail = f"max LHS={lhs} {'<=' if valid else '>'} rhs={rhs}"
    else:
        valid = lhs >= rhs - FEAS_TOL
        detail = f"min LHS={lhs} {'>=' if valid else '<'} rhs={rhs}"

    counterexample = None if valid else _counterexample(out.solution, con, lhs)
    return _SepResult(valid, False, counterexample, detail, lhs)


def _counterexample(
    solution: dict[str, float] | None, con: LinearConstraint, lhs: float
) -> dict[str, Any]:
    """Build a structured counterexample: a removed feasible integer point."""
    point = {name: _clean(val) for name, val in (solution or {}).items()}
    return {
        "point": point,
        "constraint": con.name,
        "sense": con.sense.value,
        "rhs": con.rhs,
        "lhs_value": _clean(lhs),
        "violation": _clean(lhs - con.rhs),
    }


# ---------------------------------------------------------------------------
# Solver-backed equivalence (class A)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _EquivSolve:
    ran: bool
    status_match: bool
    obj_match: bool
    obj_before: float | None
    obj_after: float | None
    status_before: str
    status_after: str
    detail: str


def _solver_equivalence(before: MILP, after: MILP, time_limit: float) -> _EquivSolve:
    """Solve both models and compare termination status + optimum."""
    b = _scip_optimize(before, None, None, time_limit)
    a = _scip_optimize(after, None, None, time_limit)
    if not (b.ran and a.ran):
        return _EquivSolve(
            ran=False,
            status_match=False,
            obj_match=False,
            obj_before=b.objective,
            obj_after=a.objective,
            status_before=b.status,
            status_after=a.status,
            detail=f"before: {b.detail or b.status}; after: {a.detail or a.status}",
        )

    conclusive = {"optimal", "infeasible", "unbounded"}
    if b.status not in conclusive or a.status not in conclusive:
        return _EquivSolve(
            ran=True,
            status_match=False,
            obj_match=False,
            obj_before=b.objective,
            obj_after=a.objective,
            status_before=b.status,
            status_after=a.status,
            detail="solver did not reach a conclusive status",
        )

    status_match = b.status == a.status
    obj_match = False
    if status_match:
        if b.status == "optimal" and b.objective is not None and a.objective is not None:
            obj_match = abs(b.objective - a.objective) <= OBJ_TOL
        else:  # both infeasible or both unbounded -> no objective to compare
            obj_match = True

    return _EquivSolve(
        ran=True,
        status_match=status_match,
        obj_match=obj_match,
        obj_before=b.objective,
        obj_after=a.objective,
        status_before=b.status,
        status_after=a.status,
        detail="",
    )


# ---------------------------------------------------------------------------
# SCIP solve primitive
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _SolveOutcome:
    ran: bool
    status: str  # "optimal" | "infeasible" | "unbounded" | "unknown"
    objective: float | None
    solution: dict[str, float] | None
    detail: str


def _scip_optimize(
    ir: MILP,
    objective_coeffs: dict[str, float] | None,
    objective_sense: ObjSense | None,
    time_limit: float,
) -> _SolveOutcome:
    """Solve ``ir`` (optionally with an overridden linear objective) via SCIP.

    When ``objective_coeffs``/``objective_sense`` are given, the model's
    objective is replaced (used by class-B separation to optimise a constraint
    LHS). Any import/build/solve failure is captured -> ``ran=False`` so the
    caller fails closed rather than crashing.
    """
    if objective_coeffs is not None and objective_sense is not None:
        ir = replace(
            ir,
            objective=Objective(coeffs=dict(objective_coeffs), sense=objective_sense, offset=0.0),
        )

    try:
        model = to_pyscipopt(ir)
    except Exception as exc:  # noqa: BLE001 -- e.g. pyscipopt absent / build error
        return _SolveOutcome(False, "unknown", None, None, f"build: {type(exc).__name__}: {exc}")

    try:
        model.hideOutput()
        model.setParam("limits/time", float(time_limit))
        try:
            model.setParam("randomization/randomseedshift", 0)
        except Exception:  # noqa: BLE001 -- param name varies; determinism is best-effort
            pass
        model.optimize()
        status = _normalize_status(model.getStatus())
        if status == "optimal":
            objective = float(model.getObjVal())
            solution = {var.name: float(model.getVal(var)) for var in model.getVars()}
            return _SolveOutcome(True, status, objective, solution, "")
        return _SolveOutcome(True, status, None, None, "")
    except Exception as exc:  # noqa: BLE001 -- never crash the gate
        return _SolveOutcome(False, "unknown", None, None, f"solve: {type(exc).__name__}: {exc}")


def _normalize_status(raw: object) -> str:
    """Map a SCIP status string to one of optimal/infeasible/unbounded/unknown."""
    r = str(raw).lower()
    if r == "optimal":
        return "optimal"
    if r == "infeasible":
        return "infeasible"
    if r in ("unbounded", "inforunbd"):
        return "unbounded"
    return "unknown"


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def _clean(value: float) -> float:
    """Round a value to the nearest integer when within :data:`FEAS_TOL`.

    Keeps counterexamples / certificate numbers readable and deterministic
    (e.g. a binary variable reported as ``0.9999999`` becomes ``1.0``).
    """
    if math.isinf(value) or math.isnan(value):
        return value
    nearest = round(value)
    if abs(value - nearest) <= FEAS_TOL:
        return float(nearest)
    return float(value)


def _class_label(declared: DeltaClass) -> str:
    return declared.value


def _reject(
    delta_class: str,
    reason: str,
    *,
    feas: bool | None = None,
    obj: bool | None = None,
    counterexample: dict[str, Any] | None = None,
) -> VerificationReport:
    """Build a fail-closed *reject* report."""
    return VerificationReport(
        status=STATUS_REJECT,
        delta_class=delta_class,
        feasible_region_integer_preserved=feas,
        objective_preserved=obj,
        counterexample=counterexample,
        reason=reason,
    )
