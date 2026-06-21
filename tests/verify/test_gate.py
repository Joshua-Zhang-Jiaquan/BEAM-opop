"""Tests for the verification gate (delta classes A--D + certificates).

Pure-IR cases (class C no-op, class D sandbox, fail-closed apply errors,
non-equivalent class-A rejection, report JSON emission) always run. The
solver-backed cases (class A rename PASS, class B separation PASS/REJECT) skip
cleanly when SCIP is absent via ``solver_skip_if_missing``.

The base fixture is a 3-variable set-packing instance::

    maximize a + b + c
    s.t.  a + b <= 1   (c_ab)
          b + c <= 1   (c_bc)
          a, b, c in {0, 1}

Feasible integer solutions: {000, 100, 010, 001, 101}.  Note a and c never
conflict, so the *clique* cut ``a + b + c <= 1`` is INVALID (it removes 101),
while ``a + b + c <= 2`` is VALID (optimum a+b+c over the region is 2).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
    apply_delta,
    make_add_constraint_delta,
    make_metadata_delta,
    make_rename_delta,
)
from opop.model.state import Delta, DeltaClass
from opop.verify import (
    STATUS_PASS,
    STATUS_REJECT,
    STATUS_SANDBOX,
    VerificationReport,
    verify_delta,
    write_report,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _packing_ir() -> MILP:
    """3-var set-packing base model (see module docstring)."""
    return MILP(
        name="packing3",
        variables=(
            Variable("a", VarType.BINARY, 0.0, 1.0),
            Variable("b", VarType.BINARY, 0.0, 1.0),
            Variable("c", VarType.BINARY, 0.0, 1.0),
        ),
        constraints=(
            LinearConstraint("c_ab", {"a": 1.0, "b": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("c_bc", {"b": 1.0, "c": 1.0}, ConstraintSense.LE, 1.0),
        ),
        objective=Objective({"a": 1.0, "b": 1.0, "c": 1.0}, ObjSense.MAXIMIZE, 0.0),
    )


def _satisfies(point: dict[str, float], con: LinearConstraint) -> bool:
    lhs = sum(coeff * point.get(name, 0.0) for name, coeff in con.coeffs.items())
    if con.sense is ConstraintSense.LE:
        return lhs <= con.rhs + 1e-7
    if con.sense is ConstraintSense.GE:
        return lhs >= con.rhs - 1e-7
    return abs(lhs - con.rhs) <= 1e-7


# ---------------------------------------------------------------------------
# Class C -- semantic no-op
# ---------------------------------------------------------------------------
def test_class_c_metadata_noop_passes() -> None:
    before = _packing_ir()
    delta = make_metadata_delta({"note": "phase1", "iter": 3})
    report = verify_delta(before, delta)

    assert report.status == STATUS_PASS
    assert report.delta_class == "C"
    assert report.feasible_region_integer_preserved is True
    assert report.objective_preserved is True
    assert report.counterexample is None


def test_class_c_with_semantic_change_is_rejected() -> None:
    """A delta declared class C that actually changes the model must reject."""
    before = _packing_ir()
    # after_ir actually adds a constraint -> a semantic change, not a no-op.
    after = apply_delta(
        before, make_add_constraint_delta("cut", {"a": 1.0, "c": 1.0}, ConstraintSense.LE, 1.0)
    )
    delta = Delta(target="mislabelled", declared_class=DeltaClass.C)
    report = verify_delta(before, delta, after_ir=after)

    assert report.status == STATUS_REJECT
    assert report.delta_class == "C"
    assert report.feasible_region_integer_preserved is False
    assert "semantic" in report.reason.lower()


# ---------------------------------------------------------------------------
# Class D -- risky -> sandbox (never main eval)
# ---------------------------------------------------------------------------
def test_class_d_is_routed_to_sandbox_never_pass() -> None:
    before = _packing_ir()
    # A class-D delta is never applied; the gate must short-circuit to sandbox.
    delta = Delta(
        target="risky reformulation",
        after_fragment='{"op": "rename_var", "old": "a", "new": "z"}',
        declared_class=DeltaClass.D,
    )
    report = verify_delta(before, delta)

    assert report.status == STATUS_SANDBOX
    assert report.status != STATUS_PASS
    assert report.passed is False
    assert report.is_sandbox is True
    assert report.delta_class == "D"
    assert report.feasible_region_integer_preserved is None
    assert report.objective_preserved is None


# ---------------------------------------------------------------------------
# Fail-closed: unknown / unprovable
# ---------------------------------------------------------------------------
def test_unapplyable_delta_fails_closed() -> None:
    """A class/op mismatch cannot be applied -> fail-closed reject (no crash)."""
    before = _packing_ir()
    # op add_constraint requires declared_class B, but it is declared A.
    bad_payload = json.dumps(
        {"op": "add_constraint", "name": "x", "coeffs": {"a": 1.0}, "sense": "<=", "rhs": 1.0}
    )
    delta = Delta(target="bad", after_fragment=bad_payload, declared_class=DeltaClass.A)
    report = verify_delta(before, delta)

    assert report.status == STATUS_REJECT
    assert report.status != STATUS_PASS
    assert "could not be applied" in report.reason


def test_bad_json_delta_fails_closed() -> None:
    before = _packing_ir()
    delta = Delta(target="bad", after_fragment="not-json", declared_class=DeltaClass.A)
    report = verify_delta(before, delta)
    assert report.status == STATUS_REJECT


def test_class_a_non_equivalent_after_is_rejected() -> None:
    """A class-A delta whose after-model is NOT equivalent fails closed."""
    before = _packing_ir()
    # Same variables, but the objective changed -> not an equivalent reformulation.
    after = MILP(
        name=before.name,
        variables=before.variables,
        constraints=before.constraints,
        objective=Objective({"a": 5.0, "b": 1.0, "c": 1.0}, ObjSense.MAXIMIZE, 0.0),
    )
    delta = Delta(target="fake equivalence", declared_class=DeltaClass.A)
    report = verify_delta(before, delta, after_ir=after)

    assert report.status == STATUS_REJECT
    assert report.delta_class == "A"
    assert report.feasible_region_integer_preserved is False
    assert "equivalent reformulation" in report.reason


def test_class_a_ambiguous_mapping_fails_closed() -> None:
    """Two renamed variables -> no 1-1 mapping inferable -> reject."""
    before = _packing_ir()
    after = MILP(
        name=before.name,
        variables=(
            Variable("aa", VarType.BINARY, 0.0, 1.0),
            Variable("bb", VarType.BINARY, 0.0, 1.0),
            Variable("c", VarType.BINARY, 0.0, 1.0),
        ),
        constraints=(
            LinearConstraint("c_ab", {"aa": 1.0, "bb": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("c_bc", {"bb": 1.0, "c": 1.0}, ConstraintSense.LE, 1.0),
        ),
        objective=Objective({"aa": 1.0, "bb": 1.0, "c": 1.0}, ObjSense.MAXIMIZE, 0.0),
    )
    delta = Delta(target="double rename", declared_class=DeltaClass.A)
    report = verify_delta(before, delta, after_ir=after)

    assert report.status == STATUS_REJECT
    assert "unprovable" in report.reason


# ---------------------------------------------------------------------------
# Class A -- equivalent reformulation (rename) [solver-backed]
# ---------------------------------------------------------------------------
def test_class_a_rename_passes(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    before = _packing_ir()
    delta = make_rename_delta("a", "alpha")
    report = verify_delta(before, delta)

    assert report.status == STATUS_PASS
    assert report.delta_class == "A"
    assert report.feasible_region_integer_preserved is True
    assert report.objective_preserved is True
    assert report.counterexample is None
    assert report.certificate is not None
    assert report.certificate["variable_mapping"] == {"a": "alpha"}


# ---------------------------------------------------------------------------
# Class B -- valid inequality [solver-backed separation]
# ---------------------------------------------------------------------------
def test_class_b_valid_inequality_passes(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    before = _packing_ir()
    # a + b + c <= 2 is valid: the optimum of a+b+c over the region is 2 (point 101).
    delta = make_add_constraint_delta(
        "clique_valid", {"a": 1.0, "b": 1.0, "c": 1.0}, ConstraintSense.LE, 2.0
    )
    report = verify_delta(before, delta)

    assert report.status == STATUS_PASS
    assert report.delta_class == "B"
    assert report.feasible_region_integer_preserved is True
    assert report.counterexample is None
    assert report.certificate is not None
    assert report.certificate["separations"][0]["valid"] is True


def test_class_b_feasibility_breaking_cut_is_rejected_with_counterexample(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    before = _packing_ir()
    # a + b + c <= 1 is INVALID: it removes the feasible integer point 101.
    delta = make_add_constraint_delta(
        "clique_bad", {"a": 1.0, "b": 1.0, "c": 1.0}, ConstraintSense.LE, 1.0
    )
    report = verify_delta(before, delta)

    assert report.status == STATUS_REJECT
    assert report.status != STATUS_PASS
    assert report.delta_class == "B"
    assert report.feasible_region_integer_preserved is False

    ce = report.counterexample
    assert ce is not None
    assert ce["constraint"] == "clique_bad"
    # The counterexample is a feasible integer point of `before` that violates the cut.
    point = ce["point"]
    assert _satisfies(point, before.constraints[0])  # c_ab
    assert _satisfies(point, before.constraints[1])  # c_bc
    assert ce["lhs_value"] > ce["rhs"]
    # Only 101 attains a+b+c == 2 in this region.
    assert point["a"] == 1.0
    assert point["b"] == 0.0
    assert point["c"] == 1.0
    assert ce["violation"] == pytest.approx(1.0)


def test_class_b_ge_cut_reject_with_counterexample(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    before = _packing_ir()
    # a + b + c >= 1 is INVALID: it removes the all-zero feasible point 000.
    delta = make_add_constraint_delta(
        "atleast_one", {"a": 1.0, "b": 1.0, "c": 1.0}, ConstraintSense.GE, 1.0
    )
    report = verify_delta(before, delta)

    assert report.status == STATUS_REJECT
    ce = report.counterexample
    assert ce is not None
    assert ce["point"] == {"a": 0.0, "b": 0.0, "c": 0.0}
    assert ce["lhs_value"] < ce["rhs"]


def test_class_b_that_changes_objective_is_rejected() -> None:
    """A 'class-B' delta that touches more than added constraints fails closed."""
    before = _packing_ir()
    after = MILP(
        name=before.name,
        variables=before.variables,
        constraints=(
            *before.constraints,
            LinearConstraint("cut", {"a": 1.0, "c": 1.0}, ConstraintSense.LE, 2.0),
        ),
        objective=Objective({"a": 9.0, "b": 1.0, "c": 1.0}, ObjSense.MAXIMIZE, 0.0),
    )
    delta = Delta(target="sneaky", declared_class=DeltaClass.B)
    report = verify_delta(before, delta, after_ir=after)

    assert report.status == STATUS_REJECT
    assert "ONLY add constraints" in report.reason


# ---------------------------------------------------------------------------
# Report emission (verification/report.json)
# ---------------------------------------------------------------------------
def test_write_report_emits_verification_report_json(tmp_run_dir: Path) -> None:
    report = VerificationReport(
        status=STATUS_REJECT,
        delta_class="B",
        feasible_region_integer_preserved=False,
        objective_preserved=True,
        counterexample={"point": {"a": 1.0, "b": 0.0, "c": 1.0}, "constraint": "clique_bad"},
        reason="removes a feasible integer solution",
    )
    path = write_report(report, tmp_run_dir)

    assert path == tmp_run_dir / "verification" / "report.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "reject"
    assert loaded["delta_class"] == "B"
    assert loaded["feasible_region_integer_preserved"] is False
    assert loaded["objective_preserved"] is True
    assert loaded["counterexample"]["constraint"] == "clique_bad"
    assert loaded["reason"] == "removes a feasible integer solution"


def test_invalid_status_is_rejected_by_report() -> None:
    with pytest.raises(ValueError, match="invalid status"):
        VerificationReport(status="approved", delta_class="A")
