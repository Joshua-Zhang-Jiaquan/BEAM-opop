"""Unit tests for OPOP core immutable state types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from opop.model.state import (
    Delta,
    DeltaClass,
    Phi,
    ProblemState,
    ScoreRecord,
    SolveTrace,
)


def test_delta_class_enum_values() -> None:
    assert DeltaClass.A.value == "A"
    assert DeltaClass.B.value == "B"
    assert DeltaClass.C.value == "C"
    assert DeltaClass.D.value == "D"


def test_delta_construct_and_defaults() -> None:
    delta = Delta(
        target="params.seed",
        before_fragment="frag-a",
        after_fragment="frag-b",
        declared_class=DeltaClass.C,
    )
    assert delta.target == "params.seed"
    assert delta.before_fragment == "frag-a"
    assert delta.after_fragment == "frag-b"
    assert delta.declared_class == DeltaClass.C

    # Default declared class is D (risky / non-certified) so proposers must opt in.
    assert Delta(target="x").declared_class == DeltaClass.D


def test_phi_field_type_tags() -> None:
    tags = Phi.field_types()
    expected = {
        "m": "categorical",
        "v": "categorical",
        "c": "categorical",
        "d": "categorical",
        "h": "ordinal",
        "p": "continuous",
        "s": "ordinal",
        "rho": "continuous",
    }
    assert tags == expected
    assert set(tags.values()) <= {"categorical", "ordinal", "bool", "continuous"}


def test_phi_to_flat_dict_stable_keys() -> None:
    phi = Phi(
        m="extended",
        v="integer",
        c="cover",
        d="benders",
        h=2,
        p={"seed": 42, "threads": 1},
        s=3,
        rho={"gap": 0.05},
    )
    flat = phi.to_flat_dict()
    assert list(flat.keys()) == ["m", "v", "c", "d", "h", "p", "s", "rho"]
    assert flat["m"] == "extended"
    assert flat["p"] == {"seed": 42, "threads": 1}
    assert flat["s"] == 3


def test_phi_immutability() -> None:
    phi = Phi()
    with pytest.raises(FrozenInstanceError):
        phi.m = "other"  # type: ignore[misc]


def test_solve_trace_construct_and_censored() -> None:
    trace = SolveTrace(
        primal_bound_series=[1.0, 0.5, 0.0],
        dual_bound_series=[0.0, 0.0, 0.0],
        time_series=[0.1, 1.0, 2.0],
        nodes=42,
        lp_iters=120,
        cuts=3,
        first_feasible_time=0.1,
        status="TIMEOUT",
        censored=True,
        memory_peak=128.0,
        instance_id="i1",
        solver="scip",
    )
    assert trace.primal_bound_series == [1.0, 0.5, 0.0]
    assert trace.dual_bound_series == [0.0, 0.0, 0.0]
    assert trace.nodes == 42
    assert trace.lp_iters == 120
    assert trace.cuts == 3
    assert trace.first_feasible_time == pytest.approx(0.1)
    assert trace.censored is True
    assert trace.status == "TIMEOUT"


def test_score_record_construct() -> None:
    rec = ScoreRecord(
        metrics={"primal_integral": 1.5, "gap": 0.0},
        uncertainty={"primal_integral": 0.1},
        risks=["censored", "low-confidence"],
        instance_id="i1",
    )
    assert rec.metrics["primal_integral"] == pytest.approx(1.5)
    assert rec.uncertainty == {"primal_integral": 0.1}
    assert rec.risks == ["censored", "low-confidence"]


def test_problem_state_construct_and_history() -> None:
    delta = Delta(target="c", declared_class=DeltaClass.B)
    trace = SolveTrace(status="OPTIMAL")
    state = ProblemState(
        instance_id="i1",
        task_family="MILP",
        symbolic_model_ref="model.json",
        model_graph_ref="graph.json",
        formulation_history=[delta],
        solver_trace_history=[trace],
        posterior_state_ref="posterior.pkl",
        budget_state={"trials": 1},
        incumbent_solution={"x": 1.0},
        incumbent_certificate={"proof": "trivial"},
        risk_flags=["unstable"],
    )
    assert state.instance_id == "i1"
    assert state.task_family == "MILP"
    assert state.formulation_history[0].declared_class == DeltaClass.B
    assert state.solver_trace_history[0].status == "OPTIMAL"
    assert state.risk_flags == ["unstable"]


def test_problem_state_is_frozen() -> None:
    state = ProblemState(instance_id="i1")
    with pytest.raises(FrozenInstanceError):
        state.instance_id = "i2"  # type: ignore[misc]


def test_problem_state_transitions_via_replace() -> None:
    state = ProblemState(instance_id="i1")
    new_state = replace(state, instance_id="i2")
    assert new_state.instance_id == "i2"
    assert state.instance_id == "i1"
