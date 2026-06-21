"""Tests for the symbolic MILP IR, MPS/LP round-trip, graph, and apply_delta.

SCIP-dependent tests (file I/O, nonlinear rejection) skip cleanly when SCIP is
absent; the pure IR / graph / delta tests always run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    UnsupportedModelError,
    Variable,
    VarType,
    apply_delta,
    from_pyscipopt,
    make_add_constraint_delta,
    make_metadata_delta,
    make_rename_delta,
    milp_diffs,
    milps_equivalent,
    model_graph,
    read_lp,
    read_mps,
    write_lp,
    write_mps,
)
from opop.model.state import Delta, DeltaClass

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# (fixture filename, n_vars, n_constraints, nnz)
FIXTURE_SHAPES = [
    ("knapsack.mps", 5, 1, 5),
    ("assignment.mps", 9, 6, 18),
    ("production.mps", 4, 3, 9),
]
FIXTURE_NAMES = [name for name, *_ in FIXTURE_SHAPES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_ir() -> MILP:
    """A hand-built 3-var MILP (no SCIP needed) exercising every vtype/sense."""
    return MILP(
        name="tiny",
        variables=(
            Variable("x", VarType.BINARY, 0.0, 1.0),
            Variable("y", VarType.INTEGER, 0.0, 10.0),
            Variable("z", VarType.CONTINUOUS, -math.inf, math.inf),
        ),
        constraints=(
            LinearConstraint("c1", {"x": 1.0, "y": 2.0, "z": -1.0}, ConstraintSense.LE, 4.0),
            LinearConstraint("c2", {"x": 3.0, "y": 1.0}, ConstraintSense.GE, 1.0),
        ),
        objective=Objective({"x": 2.0, "y": -1.0, "z": 3.0}, ObjSense.MINIMIZE, 0.0),
    )


def _scip_nnz(path: Path) -> int:
    """Sum of non-zeros across all linear constraints, computed directly by SCIP."""
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.readProblem(str(path))
    return sum(len(model.getValsLinear(c)) for c in model.getConss())


# ---------------------------------------------------------------------------
# MPS / LP round-trip (lossless on all fixtures)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_mps_roundtrip_lossless(
    fixture: str, tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    solver_skip_if_missing("scip")
    ir0 = read_mps(str(FIXTURES / fixture))
    out = tmp_path / fixture
    write_mps(ir0, str(out))
    ir1 = read_mps(str(out))

    diffs = milp_diffs(ir0, ir1, tol=1e-9)
    assert diffs == [], f"{fixture} round-trip not lossless: {diffs}"
    assert ir0.n_vars == ir1.n_vars
    assert ir0.n_constraints == ir1.n_constraints
    assert ir0.objective.sense is ir1.objective.sense
    assert ir0.objective.offset == pytest.approx(ir1.objective.offset, abs=1e-9)


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_lp_roundtrip_lossless(
    fixture: str, tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    solver_skip_if_missing("scip")
    ir0 = read_mps(str(FIXTURES / fixture))
    out = tmp_path / (Path(fixture).stem + ".lp")
    write_lp(ir0, str(out))
    ir1 = read_lp(str(out))
    assert milps_equivalent(ir0, ir1), milp_diffs(ir0, ir1)


def test_infinite_bounds_roundtrip(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    solver_skip_if_missing("scip")
    ir0 = MILP(
        name="freevar",
        variables=(
            Variable("a", VarType.CONTINUOUS, -math.inf, math.inf),
            Variable("b", VarType.CONTINUOUS, 0.0, math.inf),
        ),
        constraints=(LinearConstraint("c", {"a": 1.0, "b": 1.0}, ConstraintSense.GE, 0.0),),
        objective=Objective({"a": 1.0, "b": 1.0}, ObjSense.MINIMIZE),
    )
    out = tmp_path / "free.mps"
    write_mps(ir0, str(out))
    ir1 = read_mps(str(out))
    by_name = {v.name: v for v in ir1.variables}
    assert by_name["a"].lower == -math.inf
    assert by_name["a"].upper == math.inf
    assert by_name["b"].lower == 0.0
    assert by_name["b"].upper == math.inf
    assert milps_equivalent(ir0, ir1), milp_diffs(ir0, ir1)


# ---------------------------------------------------------------------------
# Extraction fidelity (types / senses / objective)
# ---------------------------------------------------------------------------
def test_production_types_senses_and_offset(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir0 = read_mps(str(FIXTURES / "production.mps"))
    by_name = {v.name: v for v in ir0.variables}
    assert by_name["make_a"].vtype is VarType.INTEGER
    assert by_name["make_b"].vtype is VarType.INTEGER
    assert by_name["buy"].vtype is VarType.CONTINUOUS
    assert by_name["line"].vtype is VarType.BINARY
    assert by_name["make_a"].lower == pytest.approx(0.0)
    assert by_name["make_a"].upper == pytest.approx(10.0)
    assert by_name["buy"].upper == pytest.approx(20.0)

    senses = {c.name: c.sense for c in ir0.constraints}
    assert senses["demand"] is ConstraintSense.GE
    assert senses["capacity"] is ConstraintSense.LE
    assert senses["balance"] is ConstraintSense.EQ

    assert ir0.objective.sense is ObjSense.MINIMIZE
    assert ir0.objective.offset == pytest.approx(7.0)


def test_knapsack_maximize_and_coeffs(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    ir0 = read_mps(str(FIXTURES / "knapsack.mps"))
    assert ir0.objective.sense is ObjSense.MAXIMIZE
    assert all(v.vtype is VarType.BINARY for v in ir0.variables)
    cap = ir0.constraints[0]
    assert cap.sense is ConstraintSense.LE
    assert cap.rhs == pytest.approx(12.0)
    assert ir0.objective.coeffs["item2"] == pytest.approx(11.0)
    assert cap.coeffs["item2"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Bipartite model graph (node/edge counts == constraint nnz)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("fixture", "n_vars", "n_cons", "nnz"), FIXTURE_SHAPES)
def test_model_graph_counts_match_nnz(
    fixture: str,
    n_vars: int,
    n_cons: int,
    nnz: int,
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir0 = read_mps(str(FIXTURES / fixture))
    graph = ir0.model_graph()
    assert ir0.n_vars == n_vars
    assert ir0.n_constraints == n_cons
    assert graph.n_nodes == n_vars + n_cons
    assert graph.n_edges == nnz
    assert graph.n_edges == ir0.nnz
    # Cross-check the edge count against SCIP's own constraint non-zeros.
    assert graph.n_edges == _scip_nnz(FIXTURES / fixture)


def test_model_graph_pure_ir() -> None:
    ir0 = _tiny_ir()
    graph = model_graph(ir0)
    assert graph.n_nodes == 5
    assert graph.n_edges == 5
    assert set(graph.var_nodes) == {"x", "y", "z"}
    assert set(graph.con_nodes) == {"c1", "c2"}
    assert ("x", "c1") in graph.edges
    assert ("y", "c2") in graph.edges


def test_model_graph_to_networkx_is_bipartite() -> None:
    nx = pytest.importorskip("networkx")
    graph = _tiny_ir().model_graph().to_networkx()
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 5
    assert nx.is_bipartite(graph)


# ---------------------------------------------------------------------------
# Unsupported (nonlinear/quadratic) -> UnsupportedModelError
# ---------------------------------------------------------------------------
def test_quadratic_constraint_raises(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    from pyscipopt import Model

    model = Model("q")
    a = model.addVar(vtype="C", name="a", lb=0, ub=10)
    b = model.addVar(vtype="C", name="b", lb=0, ub=10)
    model.addCons(a * b <= 5, name="quad")
    with pytest.raises(UnsupportedModelError):
        from_pyscipopt(model)


def test_squared_term_raises(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    from pyscipopt import Model

    model = Model("sq")
    p = model.addVar(vtype="C", name="p", lb=0, ub=10)
    model.addCons(p * p <= 5, name="sq")
    with pytest.raises(UnsupportedModelError):
        from_pyscipopt(model)


# ---------------------------------------------------------------------------
# apply_delta — class A (equivalent reformulation: variable rename)
# ---------------------------------------------------------------------------
def test_apply_delta_rename_is_pure_and_correct() -> None:
    ir0 = _tiny_ir()
    ir1 = apply_delta(ir0, make_rename_delta("x", "w"))

    # purity: ir0 untouched
    assert "x" in ir0.var_names()
    assert "w" not in ir0.var_names()

    # ir1 has the rename applied everywhere
    assert "w" in ir1.var_names()
    assert "x" not in ir1.var_names()
    assert ir1.n_vars == ir0.n_vars
    assert ir1.objective.coeffs["w"] == pytest.approx(ir0.objective.coeffs["x"])
    assert ir1.constraints[0].coeffs["w"] == pytest.approx(ir0.constraints[0].coeffs["x"])
    assert "x" not in ir1.constraints[0].coeffs


def test_apply_delta_rename_equivalent_mps(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    solver_skip_if_missing("scip")
    ir0 = read_mps(str(FIXTURES / "knapsack.mps"))
    ir1 = apply_delta(ir0, make_rename_delta("item0", "gold"))

    assert "gold" in ir1.var_names()
    assert "item0" not in ir1.var_names()
    assert ir1.objective.coeffs["gold"] == pytest.approx(ir0.objective.coeffs["item0"])

    # The renamed IR re-exports to an MPS that round-trips losslessly.
    out = tmp_path / "renamed.mps"
    write_mps(ir1, str(out))
    ir2 = read_mps(str(out))
    assert milps_equivalent(ir1, ir2), milp_diffs(ir1, ir2)

    # Renaming back reproduces the original math model exactly (equivalence).
    ir_back = apply_delta(ir1, make_rename_delta("gold", "item0"))
    assert milps_equivalent(ir0, ir_back), milp_diffs(ir0, ir_back)


def test_apply_delta_rename_collision_rejected() -> None:
    ir0 = _tiny_ir()
    with pytest.raises(ValueError, match="already exists"):
        apply_delta(ir0, make_rename_delta("x", "y"))


def test_apply_delta_rename_missing_source_rejected() -> None:
    ir0 = _tiny_ir()
    with pytest.raises(ValueError, match="not a variable"):
        apply_delta(ir0, make_rename_delta("nope", "w"))


# ---------------------------------------------------------------------------
# apply_delta — class B (valid inequality) and class C (metadata no-op)
# ---------------------------------------------------------------------------
def test_apply_delta_add_constraint_class_b() -> None:
    ir0 = _tiny_ir()
    delta = make_add_constraint_delta("cut0", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 5.0)
    ir1 = apply_delta(ir0, delta)

    assert ir0.n_constraints == 2  # purity
    assert ir1.n_constraints == 3
    added = ir1.constraints[-1]
    assert added.name == "cut0"
    assert added.sense is ConstraintSense.LE
    assert added.rhs == pytest.approx(5.0)
    # adding a 2-term row adds exactly 2 graph edges (nnz += 2)
    assert ir1.model_graph().n_edges == ir0.model_graph().n_edges + 2


def test_apply_delta_add_constraint_unknown_var_rejected() -> None:
    ir0 = _tiny_ir()
    delta = make_add_constraint_delta("bad", {"ghost": 1.0}, ConstraintSense.LE, 1.0)
    with pytest.raises(ValueError, match="unknown variables"):
        apply_delta(ir0, delta)


def test_apply_delta_add_constraint_duplicate_name_rejected() -> None:
    ir0 = _tiny_ir()
    delta = make_add_constraint_delta("c1", {"x": 1.0}, ConstraintSense.LE, 1.0)
    with pytest.raises(ValueError, match="already exists"):
        apply_delta(ir0, delta)


def test_apply_delta_update_metadata_class_c() -> None:
    ir0 = _tiny_ir()
    ir1 = apply_delta(ir0, make_metadata_delta({"origin": "test", "iter": 3}))

    assert ir1.metadata["origin"] == "test"
    assert ir1.metadata["iter"] == 3
    assert ir0.metadata == {}  # purity
    # class C is a semantic no-op: the math model is unchanged
    assert milps_equivalent(ir0, ir1)


# ---------------------------------------------------------------------------
# apply_delta — fail-closed: class D, mismatched class, unknown op, bad payload
# ---------------------------------------------------------------------------
def test_apply_delta_class_d_rejected() -> None:
    ir0 = _tiny_ir()
    delta = Delta(
        target="risky",
        after_fragment='{"op": "rename_var", "old": "x", "new": "w"}',
        declared_class=DeltaClass.D,
    )
    with pytest.raises(ValueError, match="class-D"):
        apply_delta(ir0, delta)


def test_apply_delta_op_class_mismatch_rejected() -> None:
    ir0 = _tiny_ir()
    delta = Delta(
        target="x",
        after_fragment='{"op": "rename_var", "old": "x", "new": "w"}',
        declared_class=DeltaClass.C,
    )
    with pytest.raises(ValueError, match="requires declared_class"):
        apply_delta(ir0, delta)


def test_apply_delta_unknown_op_rejected() -> None:
    ir0 = _tiny_ir()
    delta = Delta(target="x", after_fragment='{"op": "frobnicate"}', declared_class=DeltaClass.A)
    with pytest.raises(ValueError, match="unknown delta op"):
        apply_delta(ir0, delta)


def test_apply_delta_bad_json_rejected() -> None:
    ir0 = _tiny_ir()
    delta = Delta(target="x", after_fragment="not json", declared_class=DeltaClass.A)
    with pytest.raises(ValueError, match="valid JSON"):
        apply_delta(ir0, delta)


def test_apply_delta_missing_payload_rejected() -> None:
    ir0 = _tiny_ir()
    delta = Delta(target="x", after_fragment=None, declared_class=DeltaClass.A)
    with pytest.raises(ValueError, match="JSON payload"):
        apply_delta(ir0, delta)


# ---------------------------------------------------------------------------
# MILP referential-integrity validation
# ---------------------------------------------------------------------------
def test_milp_rejects_unknown_constraint_variable() -> None:
    with pytest.raises(ValueError, match="unknown variables"):
        MILP(
            variables=(Variable("x", VarType.BINARY, 0.0, 1.0),),
            constraints=(LinearConstraint("c", {"y": 1.0}, ConstraintSense.LE, 1.0),),
        )


def test_milp_rejects_duplicate_variable_names() -> None:
    with pytest.raises(ValueError, match="duplicate variable"):
        MILP(
            variables=(
                Variable("x", VarType.BINARY, 0.0, 1.0),
                Variable("x", VarType.INTEGER, 0.0, 5.0),
            ),
        )


def test_milp_rejects_unknown_objective_variable() -> None:
    with pytest.raises(ValueError, match="objective references unknown"):
        MILP(
            variables=(Variable("x", VarType.BINARY, 0.0, 1.0),),
            objective=Objective({"ghost": 1.0}, ObjSense.MINIMIZE),
        )


# ---------------------------------------------------------------------------
# Equivalence helper (negative path)
# ---------------------------------------------------------------------------
def test_milps_equivalent_detects_difference() -> None:
    ir0 = _tiny_ir()
    ir1 = apply_delta(
        ir0, make_add_constraint_delta("c3", {"x": 1.0}, ConstraintSense.LE, 1.0)
    )
    assert not milps_equivalent(ir0, ir1)
    assert any("constraint set" in d for d in milp_diffs(ir0, ir1))


def test_milps_equivalent_tolerates_small_perturbation() -> None:
    ir0 = _tiny_ir()
    perturbed = MILP(
        name=ir0.name,
        variables=ir0.variables,
        constraints=(
            LinearConstraint("c1", {"x": 1.0, "y": 2.0, "z": -1.0 + 1e-12}, ConstraintSense.LE, 4.0),
            ir0.constraints[1],
        ),
        objective=ir0.objective,
    )
    assert milps_equivalent(ir0, perturbed, tol=1e-9)
