"""Task-26 acceptance: Lagrangian bound, symmetry/dominance, Benders/DW readiness.

The pure signals (symmetry orbits/dominance, decomposition readiness) run without
a solver. The Lagrangian dual-bound tests solve integer subproblems with SCIP and
skip cleanly when it is unavailable.

Reference fixtures with hand-verified values:

* ``coupling_milp`` — ``min sum x_i`` over two blocks ``3(x in block) >= 4``
  (each forces ``>= 2`` integer picks) coupled by ``sum x_i >= 3``. LP relaxation
  optimum ``z_LP = 3``; integer optimum ``z_IP = 4``. Relaxing the single coupling
  row recovers the integer blocks, so the Lagrangian bound equals ``z_IP = 4`` —
  strictly above ``z_LP``, a valid dual bound.
* ``symmetric_milp`` — ``max x1+x2+x3`` with one symmetric cardinality row: the
  variables are interchangeable, so ``{x1, x2, x3}`` is one orbit (group ``S3``).
* ``asymmetric_milp`` — distinct objective coefficients and weights: the only
  automorphism is the identity, so there are no orbits.
* ``dominance_milp`` — ``min a + 3b`` with the identical column ``a + b <= 1``:
  ``a`` dominates ``b`` (same column, strictly better objective), no orbit.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from opop.analyzer import (
    LAGRANGIAN_ANALYZED,
    LAGRANGIAN_NO_COUPLING,
    READY_BENDERS,
    READY_BOTH,
    READY_DW,
    READY_NONE,
    SYMMETRY_ANALYZED,
    SYMMETRY_EMPTY,
    SYMMETRY_SKIPPED,
    AnalysisReport,
    analyze,
    classify_readiness,
    detect_decomposition,
    detect_symmetry,
    estimate_lagrangian_bound,
)
from opop.analyzer.lagrangian import BOUND_LOWER, BOUND_UPPER
from opop.analyzer.relaxation import analyze_relaxation
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)


# ---------------------------------------------------------------------------
# Fixture builders (hand-built IR; no file I/O)
# ---------------------------------------------------------------------------
def _bin(name: str) -> Variable:
    return Variable(name, VarType.BINARY, 0.0, 1.0)


def coupling_milp() -> MILP:
    """min sum x_i; two blocks (3*sum>=4) coupled by sum>=3. z_LP=3, z_IP=4."""
    variables = tuple(_bin(f"x{i}") for i in range(6))
    blk0 = LinearConstraint("blk0", {"x0": 3.0, "x1": 3.0, "x2": 3.0}, ConstraintSense.GE, 4.0)
    blk1 = LinearConstraint("blk1", {"x3": 3.0, "x4": 3.0, "x5": 3.0}, ConstraintSense.GE, 4.0)
    link = LinearConstraint("link", {f"x{i}": 1.0 for i in range(6)}, ConstraintSense.GE, 3.0)
    obj = Objective({f"x{i}": 1.0 for i in range(6)}, ObjSense.MINIMIZE)
    return MILP("coupling", variables, (blk0, blk1, link), obj)


def three_block_dw() -> MILP:
    """MAX with budget<=2 coupling over three 2-var blocks. z_LP=z_IP=9."""
    variables = tuple(_bin(n) for i in range(3) for n in (f"a{i}", f"b{i}"))
    blocks = tuple(
        LinearConstraint(f"blk{i}", {f"a{i}": 1.0, f"b{i}": 1.0}, ConstraintSense.LE, 1.0)
        for i in range(3)
    )
    budget = LinearConstraint(
        "budget",
        {f"a{i}": 1.0 for i in range(3)} | {f"b{i}": 1.0 for i in range(3)},
        ConstraintSense.LE,
        2.0,
    )
    obj = Objective(
        {"a0": 3.0, "a1": 4.0, "a2": 5.0, "b0": 1.0, "b1": 1.0, "b2": 1.0},
        ObjSense.MAXIMIZE,
    )
    return MILP("three_block_dw", variables, (*blocks, budget), obj)


def single_knapsack() -> MILP:
    """One <= row over all binaries: monolithic, no coupling constraint."""
    weights = {"i0": 5.0, "i1": 3.0, "i2": 7.0, "i3": 4.0, "i4": 6.0}
    values = {"i0": 8.0, "i1": 5.0, "i2": 11.0, "i3": 6.0, "i4": 9.0}
    return MILP(
        "knapsack",
        tuple(_bin(n) for n in weights),
        (LinearConstraint("cap", weights, ConstraintSense.LE, 12.0),),
        Objective(values, ObjSense.MAXIMIZE),
    )


def symmetric_milp() -> MILP:
    """max x1+x2+x3 with one symmetric cardinality row -> orbit {x1,x2,x3}."""
    return MILP(
        "sym",
        (_bin("x1"), _bin("x2"), _bin("x3")),
        (LinearConstraint("card", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, ConstraintSense.LE, 2.0),),
        Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjSense.MAXIMIZE),
    )


def asymmetric_milp() -> MILP:
    """Distinct objective coeffs + weights -> identity is the only automorphism."""
    weights = {"i0": 5.0, "i1": 3.0, "i2": 7.0, "i3": 4.0, "i4": 6.0}
    values = {"i0": 8.0, "i1": 5.0, "i2": 11.0, "i3": 6.0, "i4": 9.0}
    return MILP(
        "asym",
        tuple(_bin(n) for n in weights),
        (LinearConstraint("cap", weights, ConstraintSense.LE, 12.0),),
        Objective(values, ObjSense.MAXIMIZE),
    )


def dominance_milp() -> MILP:
    """min a + 3b s.t. a + b <= 1: identical column, a dominates b, no orbit."""
    return MILP(
        "dom",
        (_bin("a"), _bin("b")),
        (LinearConstraint("c", {"a": 1.0, "b": 1.0}, ConstraintSense.LE, 1.0),),
        Objective({"a": 1.0, "b": 3.0}, ObjSense.MINIMIZE),
    )


def benders_linking_var() -> MILP:
    """Two constraint blocks coupled ONLY by a complicating variable y (all binary)."""
    variables = (_bin("y"), _bin("x0"), _bin("x1"), _bin("x2"), _bin("x3"))
    constraints = (
        LinearConstraint("r0", {"x0": 1.0, "x1": 1.0, "y": 1.0}, ConstraintSense.LE, 2.0),
        LinearConstraint("r1", {"x0": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
        LinearConstraint("r2", {"x2": 1.0, "x3": 1.0, "y": 1.0}, ConstraintSense.LE, 2.0),
        LinearConstraint("r3", {"x3": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
    )
    return MILP("benders", variables, constraints, Objective({"y": 1.0}, ObjSense.MAXIMIZE))


def staging_milp() -> MILP:
    """Single block, mixed integer/continuous: fixing y leaves a continuous LP."""
    variables = (
        Variable("y", VarType.BINARY, 0.0, 1.0),
        Variable("x", VarType.CONTINUOUS, 0.0, 10.0),
    )
    constraints = (
        LinearConstraint("link", {"x": 1.0, "y": -5.0}, ConstraintSense.LE, 0.0),
    )
    return MILP("staging", variables, constraints, Objective({"y": 2.0, "x": -1.0}, ObjSense.MINIMIZE))


def both_milp() -> MILP:
    """Two int+continuous blocks coupled by a budget row -> DW + integer staging."""
    variables = (
        Variable("y0", VarType.BINARY, 0.0, 1.0),
        Variable("x0", VarType.CONTINUOUS, 0.0, 10.0),
        Variable("y1", VarType.BINARY, 0.0, 1.0),
        Variable("x1", VarType.CONTINUOUS, 0.0, 10.0),
    )
    constraints = (
        LinearConstraint("link0", {"x0": 1.0, "y0": -5.0}, ConstraintSense.LE, 0.0),
        LinearConstraint("link1", {"x1": 1.0, "y1": -5.0}, ConstraintSense.LE, 0.0),
        LinearConstraint("budget", {"x0": 1.0, "x1": 1.0}, ConstraintSense.LE, 7.0),
    )
    obj = Objective({"x0": -1.0, "x1": -1.0, "y0": 2.0, "y1": 2.0}, ObjSense.MINIMIZE)
    return MILP("both", variables, constraints, obj)


# ===========================================================================
# Lagrangian dual bound (SCIP-dependent)
# ===========================================================================
def test_lagrangian_bound_dominates_lp_on_coupling_fixture(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = coupling_milp()
    z_lp = analyze_relaxation(ir, estimate_ip_bound=False).lp_obj
    assert z_lp is not None
    assert z_lp == pytest.approx(3.0, abs=1e-6)

    lag = estimate_lagrangian_bound(ir)
    assert lag.status == LAGRANGIAN_ANALYZED
    assert lag.bound_kind == BOUND_LOWER
    assert lag.coupling_constraints == ("link",)
    assert lag.bound is not None
    # Valid dual bound: >= LP (the task assertion) and <= IP (never optimistic).
    assert lag.bound >= z_lp - 1e-6
    assert lag.bound == pytest.approx(4.0, abs=1e-6)
    assert lag.bound <= 4.0 + 1e-6
    # Strictly tighter than the LP relaxation on this fixture.
    assert lag.bound > z_lp + 0.5
    assert lag.dominates_lp is True


def test_lagrangian_explicit_coupling_matches_auto_detection(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = coupling_milp()
    auto = estimate_lagrangian_bound(ir)
    explicit = estimate_lagrangian_bound(ir, coupling=("link",))
    assert explicit.status == LAGRANGIAN_ANALYZED
    assert explicit.bound == pytest.approx(auto.bound)
    assert explicit.coupling_constraints == ("link",)


def test_lagrangian_maximization_is_valid_upper_bound(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = three_block_dw()
    z_lp = analyze_relaxation(ir, estimate_ip_bound=False).lp_obj
    lag = estimate_lagrangian_bound(ir)
    assert lag.status == LAGRANGIAN_ANALYZED
    assert lag.bound_kind == BOUND_UPPER
    assert lag.coupling_constraints == ("budget",)
    assert lag.bound is not None
    # For MAX the dual bound is an UPPER bound: <= LP, and >= the integer optimum (9).
    assert z_lp is not None
    assert lag.bound <= z_lp + 1e-6
    assert lag.bound >= 9.0 - 1e-6
    assert lag.dominates_lp is True


def test_lagrangian_no_coupling_on_monolithic_model() -> None:
    # Single knapsack row -> no DW linking constraint -> nothing to dualize.
    lag = estimate_lagrangian_bound(single_knapsack())
    assert lag.status == LAGRANGIAN_NO_COUPLING
    assert lag.bound is None
    assert lag.coupling_constraints == ()


def test_lagrangian_empty_explicit_coupling_is_no_coupling() -> None:
    lag = estimate_lagrangian_bound(coupling_milp(), coupling=())
    assert lag.status == LAGRANGIAN_NO_COUPLING
    assert lag.bound is None


def test_lagrangian_to_dict_serializable(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    payload = estimate_lagrangian_bound(coupling_milp()).to_dict()
    text = json.dumps(payload)  # must not raise
    assert payload["status"] == LAGRANGIAN_ANALYZED
    assert payload["bound_kind"] == BOUND_LOWER
    assert "link" in text


def test_lagrangian_does_not_mutate_ir(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = coupling_milp()
    before = (ir.variables, ir.constraints, ir.objective, dict(ir.metadata))
    estimate_lagrangian_bound(ir)
    assert ir.variables == before[0]
    assert ir.constraints == before[1]
    assert ir.objective == before[2]
    assert dict(ir.metadata) == before[3]


# ===========================================================================
# Symmetry / dominance (pure)
# ===========================================================================
def test_symmetry_orbits_on_symmetric_instance() -> None:
    info = detect_symmetry(symmetric_milp())
    assert info.status == SYMMETRY_ANALYZED
    assert info.has_symmetry
    assert info.orbits == (("x1", "x2", "x3"),)
    assert info.n_automorphisms >= 6  # S3 acts on the three interchangeable vars
    assert info.n_symmetric_vars == 3


def test_symmetry_none_on_asymmetric_instance() -> None:
    info = detect_symmetry(asymmetric_milp())
    assert info.status == SYMMETRY_ANALYZED
    assert info.orbits == ()
    assert not info.has_symmetry
    assert info.n_automorphisms == 1  # identity only


def test_symmetry_dominance_pair_detected() -> None:
    info = detect_symmetry(dominance_milp())
    # a and b share a column but a has the better (lower) MIN objective -> a dominates b.
    assert info.dominance_pairs == (("a", "b"),)
    # Different objective coefficients break the symmetry -> no orbit.
    assert info.orbits == ()


def test_symmetry_no_dominance_when_objectives_equal() -> None:
    # Symmetric vars share columns AND objective -> orbit, but NOT a dominance pair.
    info = detect_symmetry(symmetric_milp())
    assert info.dominance_pairs == ()


def test_symmetry_empty_model() -> None:
    info = detect_symmetry(MILP("empty"))
    assert info.status == SYMMETRY_EMPTY
    assert not info.has_symmetry


def test_symmetry_skipped_when_over_node_cap() -> None:
    info = detect_symmetry(symmetric_milp(), max_nodes=1)
    assert info.status == SYMMETRY_SKIPPED
    assert info.orbits == ()


def test_symmetry_to_dict_serializable() -> None:
    payload = detect_symmetry(symmetric_milp()).to_dict()
    text = json.dumps(payload)  # must not raise
    assert payload["has_symmetry"] is True
    assert ["x1", "x2", "x3"] in payload["orbits"]
    assert "x1" in text


def test_symmetry_does_not_mutate_ir() -> None:
    ir = symmetric_milp()
    before = (ir.variables, ir.constraints, ir.objective)
    detect_symmetry(ir)
    assert ir.variables == before[0]
    assert ir.constraints == before[1]
    assert ir.objective == before[2]


# ===========================================================================
# Benders / Dantzig-Wolfe readiness (pure)
# ===========================================================================
def test_readiness_dw_on_coupling_constraints() -> None:
    info = classify_readiness(coupling_milp())
    assert info.recommendation == READY_DW
    assert info.dw_ready is True
    assert info.benders_ready is False
    assert info.linking_constraints == ("link",)


def test_readiness_benders_on_complicating_variable() -> None:
    info = classify_readiness(benders_linking_var())
    assert info.recommendation == READY_BENDERS
    assert info.benders_ready is True
    assert info.dw_ready is False
    assert info.linking_variables == ("y",)


def test_readiness_benders_on_integer_continuous_staging() -> None:
    info = classify_readiness(staging_milp())
    assert info.recommendation == READY_BENDERS
    assert info.benders_ready is True
    assert info.n_integer == 1
    assert info.n_continuous == 1


def test_readiness_both_on_dw_plus_staging() -> None:
    info = classify_readiness(both_milp())
    assert info.recommendation == READY_BOTH
    assert info.dw_ready is True
    assert info.benders_ready is True


def test_readiness_none_on_monolithic_single_domain() -> None:
    info = classify_readiness(single_knapsack())
    assert info.recommendation == READY_NONE
    assert info.dw_ready is False
    assert info.benders_ready is False


def test_readiness_reasoning_is_nonempty() -> None:
    assert classify_readiness(coupling_milp()).reasoning
    assert classify_readiness(single_knapsack()).reasoning


def test_readiness_reuses_precomputed_decomposition() -> None:
    ir = coupling_milp()
    decomp = detect_decomposition(ir)
    info = classify_readiness(ir, decomposition=decomp)
    assert info.structure == decomp.decomposability
    assert info.n_blocks == decomp.n_blocks


def test_readiness_to_dict_serializable() -> None:
    payload = classify_readiness(both_milp()).to_dict()
    text = json.dumps(payload)  # must not raise
    assert payload["recommendation"] == READY_BOTH
    assert "budget" in text


# ===========================================================================
# analyze() integration + purity
# ===========================================================================
def test_analyze_populates_symmetry_and_readiness_by_default() -> None:
    report = analyze(symmetric_milp(), solve_relaxation=False)
    assert isinstance(report, AnalysisReport)
    assert report.symmetry is not None
    assert report.symmetry.has_symmetry
    assert report.decomposition_readiness is not None
    # Lagrangian is opt-in (solver-backed): off by default.
    assert report.lagrangian is None


def test_analyze_with_lagrangian_flag(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    report = analyze(coupling_milp(), with_lagrangian=True, solve_relaxation=False)
    assert report.lagrangian is not None
    assert report.lagrangian.status == LAGRANGIAN_ANALYZED
    assert report.lagrangian.bound == pytest.approx(4.0, abs=1e-6)


def test_analyze_flags_disable_sections() -> None:
    report = analyze(
        symmetric_milp(),
        solve_relaxation=False,
        with_symmetry=False,
        with_readiness=False,
    )
    assert report.symmetry is None
    assert report.decomposition_readiness is None


def test_analyze_report_to_dict_includes_new_sections() -> None:
    payload = analyze(coupling_milp(), solve_relaxation=False).to_dict()
    text = json.dumps(payload)  # must not raise
    assert "symmetry" in payload
    assert "decomposition_readiness" in payload
    assert "lagrangian" in payload
    assert payload["decomposition_readiness"]["recommendation"] == READY_DW
    assert payload["lagrangian"] is None  # off by default
    assert isinstance(payload["symmetry"], dict)
    assert text  # serialisation succeeded


def test_analyze_does_not_mutate_ir_with_all_sections(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = coupling_milp()
    before = (ir.variables, ir.constraints, ir.objective, dict(ir.metadata))
    analyze(ir, with_lagrangian=True)
    assert ir.variables == before[0]
    assert ir.constraints == before[1]
    assert ir.objective == before[2]
    assert dict(ir.metadata) == before[3]
