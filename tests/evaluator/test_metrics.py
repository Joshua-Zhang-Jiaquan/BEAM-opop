"""Tests for the Evaluator: anytime metrics, primal integral, right-censoring."""

from __future__ import annotations

import math

import pytest

from opop.evaluator import (
    evaluate,
    final_gap,
    is_feasible,
    is_optimal,
    par10,
    primal_dual_gap_integral,
    primal_integral,
    runtime,
    scalarize,
)
from opop.model.state import ScoreRecord, SolveTrace

# Required ScoreRecord.metrics keys per the task-13 spec.
_REQUIRED_METRICS = {
    "feasible",
    "objective",
    "gap",
    "time_to_first_feasible",
    "primal_integral",
    "primal_dual_gap_integral",
    "nodes",
    "cuts",
    "memory_peak",
    "censored",
}


def _three_step_trace() -> SolveTrace:
    """Primal pinned at 10; dual 0 -> 5 -> 10 so the gap steps 1.0, 0.5, 0.0.

    ``gap(t) = |primal - dual| / |primal|``:
        [0,1): 1.0, [1,2): 0.5, [2,3): 0.0  =>  step integral = 1.5
    """
    return SolveTrace(
        primal_bound_series=[10.0, 10.0, 10.0, 10.0],
        dual_bound_series=[0.0, 5.0, 10.0, 10.0],
        time_series=[0.0, 1.0, 2.0, 3.0],
        nodes=7,
        lp_iters=20,
        cuts=3,
        first_feasible_time=0.0,
        status="optimal",
        censored=False,
        memory_peak=64.0,
        instance_id="synthetic_3step",
        solver="scip",
    )


def _censored_trace() -> SolveTrace:
    """A timed-out run: a feasible incumbent exists but no optimality proof."""
    return SolveTrace(
        primal_bound_series=[float("inf"), 10.0, 6.0],
        dual_bound_series=[0.0, 0.0, 2.0],
        time_series=[0.0, 0.5, 2.0],
        nodes=5000,
        lp_iters=99999,
        cuts=42,
        first_feasible_time=0.5,
        status="timelimit",
        censored=True,
        memory_peak=256.0,
        instance_id="hard_market_split",
        solver="scip",
    )


@pytest.mark.smoke
def test_primal_integral_matches_hand_value() -> None:
    """The primal integral == 1.5 on the synthetic 3-step trajectory."""
    trace = _three_step_trace()
    assert primal_integral(trace) == pytest.approx(1.5, abs=1e-9)
    # Computed from the SERIES, not the endpoints: the endpoint gap is 0.0,
    # yet the trajectory integral is 1.5.
    assert final_gap(trace) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.smoke
def test_primal_integral_uses_reference_optimum() -> None:
    """A reference optimum measures the gap to it, not to the dual bound."""
    trace = _three_step_trace()
    # Primal pinned at the true optimum 10 => zero primal gap throughout.
    assert primal_integral(trace, reference_optimum=10.0) == pytest.approx(0.0, abs=1e-12)
    # primal_dual_gap_integral ignores the reference and stays at 1.5.
    assert primal_dual_gap_integral(trace) == pytest.approx(1.5, abs=1e-9)


@pytest.mark.smoke
def test_evaluate_solved_record() -> None:
    """A solved run is feasible, optimal, zero-gap, with the right integral."""
    rec = evaluate(_three_step_trace(), time_limit=30.0)
    assert isinstance(rec, ScoreRecord)
    m = rec.metrics
    assert _REQUIRED_METRICS <= set(m)
    assert m["feasible"] == 1.0
    assert m["optimal"] == 1.0
    assert m["objective"] == pytest.approx(10.0)
    assert m["gap"] == pytest.approx(0.0, abs=1e-12)
    assert m["primal_integral"] == pytest.approx(1.5, abs=1e-9)
    assert m["censored"] == 0.0
    assert m["nodes"] == 7.0
    assert m["cuts"] == 3.0
    assert m["memory_peak"] == 64.0
    # Not censored => PAR10 aux is just the actual runtime (final timestamp).
    assert m["par10_aux"] == pytest.approx(3.0)
    assert rec.uncertainty is None
    assert rec.risks == []
    assert rec.instance_id == "synthetic_3step"


@pytest.mark.smoke
def test_censored_run_not_counted_as_solved() -> None:
    """A censored run stays feasible-but-not-optimal; flag + risk preserved."""
    trace = _censored_trace()
    rec = evaluate(trace, time_limit=2.0)
    m = rec.metrics
    assert m["feasible"] == 1.0  # a feasible incumbent was found
    assert m["optimal"] == 0.0  # but NOT solved / optimal
    assert m["censored"] == 1.0  # censored flag preserved
    assert "censored" in rec.risks
    assert not is_optimal(trace)
    # Recorded runtime is the (lower-bound) final timestamp, NOT a penalty.
    assert m["solve_time"] == pytest.approx(2.0)


@pytest.mark.smoke
def test_par10_auxiliary_on_timeout() -> None:
    """PAR10 aux == 10 x limit on a timeout; == actual runtime otherwise."""
    # Censored => 10 * time_limit.
    assert par10(2.0, 2.0, censored=True) == pytest.approx(20.0)
    assert par10(2.0, 10.0, censored=True) == pytest.approx(100.0)
    # Not censored => actual runtime, unchanged.
    assert par10(2.5, 10.0, censored=False) == pytest.approx(2.5)
    # No finite limit but censored => undefined (NaN), never a silent 0.
    assert math.isnan(par10(2.0, None, censored=True))
    # Flows through evaluate() as the labeled auxiliary key on a timeout.
    rec = evaluate(_censored_trace(), time_limit=2.0)
    assert rec.metrics["par10_aux"] == pytest.approx(20.0)
    # ... and the primary record is NOT the penalty (still the censored runtime).
    assert rec.metrics["solve_time"] == pytest.approx(2.0)


@pytest.mark.smoke
def test_feasibility_and_runtime_edge_cases() -> None:
    """Infeasible / no-incumbent traces are not feasible; empty trace is safe."""
    infeasible = SolveTrace(
        primal_bound_series=[float("inf")],
        dual_bound_series=[float("inf")],
        time_series=[1.0],
        first_feasible_time=float("nan"),
        status="infeasible",
        censored=False,
        instance_id="infeasible",
        solver="scip",
    )
    assert is_feasible(infeasible) is False
    rec = evaluate(infeasible, time_limit=30.0)
    assert rec.metrics["feasible"] == 0.0
    assert "no-feasible-incumbent" in rec.risks

    empty = SolveTrace()
    assert math.isnan(runtime(empty))
    assert primal_integral(empty) == 0.0  # <2 points => 0 integral, no crash


def test_scalarize_matches_coip_reward() -> None:
    """scalarize(record) is identical to task-15 coip_reward(record.metrics)."""
    from opop.controller.phase1 import coip_reward

    rec = evaluate(_censored_trace(), time_limit=2.0)
    assert scalarize(rec) == pytest.approx(coip_reward(rec.metrics), abs=1e-12)

    # Weight overrides apply with the documented formula.
    weights = {"w_gap": 2.0, "w_time": 0.0, "w_pi": 0.5}
    m = rec.metrics
    expected = -(2.0 * m["gap"]) - (0.0 * m["solve_time"]) - (0.5 * m["primal_integral"])
    assert scalarize(rec, weights) == pytest.approx(expected, abs=1e-12)
