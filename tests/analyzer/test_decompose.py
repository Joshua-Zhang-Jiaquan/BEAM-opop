"""Task-24 acceptance: decomposability detection from the bipartite model graph.

All tests are pure (no solver): :func:`opop.analyzer.decompose.detect_decomposition`
reads only the variable--constraint incidence, and the class-C certification of a
decomposition delta uses the verification gate's symbolic class-C contract (no SCIP).

Reference fixtures with known structure:

* ``three_block_dw`` — three independent 2-variable blocks coupled by ONE budget
  (linking) constraint. Removing the coupling row yields 3 blocks, so the verdict
  is ``DW`` (Dantzig--Wolfe amenable). Integer optimum (MAX) is ``9``.
* ``pure_block`` — the same three blocks with NO coupling row: pure block-diagonal
  → ``BLOCK`` with 3 blocks.
* ``benders`` — two constraint blocks sharing one complicating variable ``y``;
  removing ``y`` splits the constraints into 2 blocks → ``BENDERS``.
* ``dense`` — every row touches every variable: no small border splits it → ``NONE``.
"""

from __future__ import annotations

import json

from opop.analyzer import (
    AnalysisReport,
    DecompositionReport,
    analyze,
    decomposition_delta,
    detect_decomposition,
)
from opop.analyzer.decompose import (
    DECOMP_BENDERS,
    DECOMP_BLOCK,
    DECOMP_DW,
    DECOMP_NONE,
)
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)
from opop.model.state import DeltaClass
from opop.verify.gate import verify_delta


# ---------------------------------------------------------------------------
# Fixture builders (hand-built IR; no file I/O, no solver)
# ---------------------------------------------------------------------------
def _bin(name: str) -> Variable:
    return Variable(name, VarType.BINARY, 0.0, 1.0)


def three_block_dw() -> MILP:
    """3 blocks {a_i, b_i} (a_i+b_i<=1) coupled by one budget row (sum<=2).

    Per-block best pick is ``a_i`` (values 3, 4, 5); the budget allows 2 picks,
    so the integer optimum is ``a2 + a1 = 5 + 4 = 9``.
    """
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


def pure_block() -> MILP:
    """The same three 2-variable blocks with NO coupling row (pure block-diagonal)."""
    variables = tuple(_bin(n) for i in range(3) for n in (f"a{i}", f"b{i}"))
    blocks = tuple(
        LinearConstraint(f"blk{i}", {f"a{i}": 1.0, f"b{i}": 1.0}, ConstraintSense.LE, 1.0)
        for i in range(3)
    )
    obj = Objective({f"a{i}": 1.0 for i in range(3)}, ObjSense.MAXIMIZE)
    return MILP("pure_block", variables, blocks, obj)


def benders() -> MILP:
    """Two constraint blocks coupled ONLY by a complicating variable ``y``."""
    variables = (_bin("y"), _bin("x0"), _bin("x1"), _bin("x2"), _bin("x3"))
    constraints = (
        LinearConstraint("r0", {"x0": 1.0, "x1": 1.0, "y": 1.0}, ConstraintSense.LE, 2.0),
        LinearConstraint("r1", {"x0": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
        LinearConstraint("r2", {"x2": 1.0, "x3": 1.0, "y": 1.0}, ConstraintSense.LE, 2.0),
        LinearConstraint("r3", {"x3": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
    )
    return MILP("benders", variables, constraints, Objective({"y": 1.0}, ObjSense.MAXIMIZE))


def dense() -> MILP:
    """A dense matrix: every row touches every variable -> no decomposition."""
    variables = tuple(_bin(f"x{i}") for i in range(5))
    constraints = tuple(
        LinearConstraint(
            f"c{j}", {f"x{i}": float(i + j + 1) for i in range(5)}, ConstraintSense.LE, 3.0
        )
        for j in range(5)
    )
    obj = Objective({f"x{i}": 1.0 for i in range(5)}, ObjSense.MAXIMIZE)
    return MILP("dense", variables, constraints, obj)


# ===========================================================================
# Verdicts
# ===========================================================================
def test_three_block_instance_is_dw_with_three_blocks() -> None:
    report = detect_decomposition(three_block_dw())
    assert report.decomposability == DECOMP_DW
    assert report.n_blocks == 3
    assert report.linking_constraints == ("budget",)
    assert not report.linking_variables
    assert {frozenset(b) for b in report.block_vars} == {
        frozenset({"a0", "b0"}),
        frozenset({"a1", "b1"}),
        frozenset({"a2", "b2"}),
    }


def test_pure_block_diagonal_is_block() -> None:
    report = detect_decomposition(pure_block())
    assert report.decomposability == DECOMP_BLOCK
    assert report.n_blocks == 3
    assert not report.linking_constraints
    assert not report.linking_variables


def test_linking_variable_is_benders() -> None:
    report = detect_decomposition(benders())
    assert report.decomposability == DECOMP_BENDERS
    assert report.n_blocks == 2
    assert report.linking_variables == ("y",)
    assert not report.linking_constraints
    # Block variables exclude the complicating variable y.
    flat = {v for block in report.block_vars for v in block}
    assert "y" not in flat
    assert flat == {"x0", "x1", "x2", "x3"}


def test_dense_instance_is_none() -> None:
    report = detect_decomposition(dense())
    assert report.decomposability == DECOMP_NONE
    assert report.n_blocks == 0
    assert not report.block_vars


def test_single_coupling_constraint_does_not_force_decomposition() -> None:
    # One row over all variables is monolithic; there is no genuine border.
    milp = MILP(
        "single_dense_row",
        tuple(_bin(f"x{i}") for i in range(4)),
        (LinearConstraint("only", {f"x{i}": 1.0 for i in range(4)}, ConstraintSense.LE, 2.0),),
        Objective({f"x{i}": 1.0 for i in range(4)}, ObjSense.MAXIMIZE),
    )
    assert detect_decomposition(milp).decomposability == DECOMP_NONE


def test_two_block_dw_minimal() -> None:
    milp = MILP(
        "two_block",
        (_bin("a"), _bin("b"), _bin("c"), _bin("d")),
        (
            LinearConstraint("blkA", {"a": 1.0, "b": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("blkB", {"c": 1.0, "d": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("link", {"a": 1.0, "c": 1.0}, ConstraintSense.LE, 1.0),
        ),
        Objective({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, ObjSense.MAXIMIZE),
    )
    report = detect_decomposition(milp)
    assert report.decomposability == DECOMP_DW
    assert report.n_blocks == 2
    assert report.linking_constraints == ("link",)


# ===========================================================================
# Decomposition delta: class-C, certified by the verification gate
# ===========================================================================
def test_decomposition_delta_is_class_c_and_certified() -> None:
    ir = three_block_dw()
    report = detect_decomposition(ir)
    delta = decomposition_delta(report)
    assert delta is not None
    assert delta.declared_class is DeltaClass.C

    verdict = verify_delta(ir, delta)
    assert verdict.passed, verdict.reason
    assert verdict.delta_class == "C"
    # The decomposition is a solver-strategy no-op: the feasible set + objective
    # are preserved, so it is CERTIFIED (never treated as uncertified).
    assert verdict.feasible_region_integer_preserved is True
    assert verdict.objective_preserved is True


def test_decomposition_delta_none_for_dense() -> None:
    assert decomposition_delta(detect_decomposition(dense())) is None


def test_decomposition_delta_carries_report_payload() -> None:
    report = detect_decomposition(three_block_dw())
    delta = decomposition_delta(report)
    assert delta is not None and delta.after_fragment is not None
    payload = json.loads(delta.after_fragment)
    assert payload["op"] == "update_metadata"
    assert payload["updates"]["decomposition"]["decomposability"] == DECOMP_DW
    assert payload["updates"]["decomposition"]["n_blocks"] == 3


# ===========================================================================
# Report structure / purity / api integration
# ===========================================================================
def test_report_to_dict_is_json_serializable() -> None:
    payload = detect_decomposition(three_block_dw()).to_dict()
    text = json.dumps(payload)  # must not raise
    assert '"DW"' in text
    assert payload["n_blocks"] == 3
    assert isinstance(payload["block_vars"], list)
    assert payload["linking_constraints"] == ["budget"]
    assert payload["reasoning"]


def test_empty_model_is_none() -> None:
    report = detect_decomposition(MILP("empty"))
    assert report.decomposability == DECOMP_NONE
    assert isinstance(report, DecompositionReport)


def test_detect_does_not_mutate_ir() -> None:
    ir = three_block_dw()
    before = (ir.variables, ir.constraints, ir.objective, dict(ir.metadata))
    detect_decomposition(ir)
    assert ir.variables == before[0]
    assert ir.constraints == before[1]
    assert ir.objective == before[2]
    assert dict(ir.metadata) == before[3]


def test_analyze_populates_decomposition() -> None:
    report = analyze(three_block_dw(), solve_relaxation=False)
    assert isinstance(report, AnalysisReport)
    assert report.decomposability == DECOMP_DW
    assert report.decomposition is not None
    assert report.decomposition.n_blocks == 3
    # The string verdict mirrors the structured report.
    assert report.decomposability == report.decomposition.decomposability


def test_analyze_dense_reports_none_decomposition() -> None:
    report = analyze(dense(), solve_relaxation=False)
    assert report.decomposability == DECOMP_NONE
    assert report.decomposition is not None
    assert report.decomposition.n_blocks == 0


def test_analysis_report_to_dict_includes_decomposition() -> None:
    payload = analyze(three_block_dw(), solve_relaxation=False).to_dict()
    text = json.dumps(payload)  # must not raise
    assert payload["decomposability"] == DECOMP_DW
    assert payload["decomposition"]["n_blocks"] == 3
    assert "budget" in text
