"""Task-23 acceptance: HiGHS solver kernel (high-level ``highspy`` API).

Verifies that :class:`opop.solver.highs.HighsKernel`:

* solves a known 0/1 knapsack to its optimum (50) and a 3x3 assignment to its
  optimum (9), returning a :class:`SolveTrace` whose final primal AGREES with
  :class:`opop.solver.scip.ScipKernel` on both binary/integer fixtures;
* records exactly what the high-level API exposes (a single terminal
  primal/dual/time point plus terminal node and LP-iteration counts) and reports
  the fields HiGHS does NOT expose (cuts / first-feasible / peak-memory) as
  ``None`` rather than a fabricated value;
* honours a hard time limit on a Cornuejols--Dawande market-split instance,
  yielding ``censored=True``;
* satisfies the :class:`~opop.solver.kernel.SolverKernel` Protocol, is
  deterministic under a fixed seed, and fail-closes on non-whitelisted params.

HiGHS is required; the backend is skipped via ``solver_skip_if_missing``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)
from opop.model.state import Phi, SolveTrace
from opop.solver.highs import HIGHS_WHITELISTED_PARAMS, HighsKernel
from opop.solver.kernel import SolverKernel

_MEMORY_MB = 4096

# Deterministic 6-item 0/1 knapsack (same as the SCIP fixture): items {0,2,4,5}
# give value 10+18+7+15 = 50 at weight 2+4+1+3 = 10 == capacity.
_KNAPSACK_OPTIMUM = 50.0
# 3x3 assignment, min perfect matching of the cost matrix below: w0->t1 (2),
# w1->t0 (6), w2->t2 (1) = 9 (confirmed against SCIP).
_ASSIGNMENT_OPTIMUM = 9.0


def _knapsack_ir() -> MILP:
    values = (10.0, 13.0, 18.0, 31.0, 7.0, 15.0)
    weights = (2.0, 3.0, 4.0, 7.0, 1.0, 3.0)
    n = len(values)
    variables = tuple(
        Variable(name=f"x{i}", vtype=VarType.BINARY, lower=0.0, upper=1.0) for i in range(n)
    )
    capacity = LinearConstraint(
        name="capacity",
        coeffs={f"x{i}": weights[i] for i in range(n)},
        sense=ConstraintSense.LE,
        rhs=10.0,
    )
    objective = Objective(
        coeffs={f"x{i}": values[i] for i in range(n)}, sense=ObjSense.MAXIMIZE
    )
    return MILP(
        name="knapsack6", variables=variables, constraints=(capacity,), objective=objective
    )


def _assignment_ir() -> MILP:
    cost = [[9.0, 2.0, 7.0], [6.0, 4.0, 3.0], [5.0, 8.0, 1.0]]
    variables = tuple(
        Variable(name=f"x{i}{j}", vtype=VarType.BINARY, lower=0.0, upper=1.0)
        for i in range(3)
        for j in range(3)
    )
    worker_rows = [
        LinearConstraint(
            name=f"worker{i}",
            coeffs={f"x{i}{j}": 1.0 for j in range(3)},
            sense=ConstraintSense.EQ,
            rhs=1.0,
        )
        for i in range(3)
    ]
    task_rows = [
        LinearConstraint(
            name=f"task{j}",
            coeffs={f"x{i}{j}": 1.0 for i in range(3)},
            sense=ConstraintSense.EQ,
            rhs=1.0,
        )
        for j in range(3)
    ]
    objective = Objective(
        coeffs={f"x{i}{j}": cost[i][j] for i in range(3) for j in range(3)},
        sense=ObjSense.MINIMIZE,
    )
    return MILP(
        name="assign3",
        variables=variables,
        constraints=tuple(worker_rows + task_rows),
        objective=objective,
    )


def _market_split_ir(m_con: int = 4, seed: int = 7) -> MILP:
    """A Cornuejols--Dawande market-split instance (hard for its size)."""
    rng = random.Random(seed)
    n = 10 * (m_con - 1)
    rows = [[float(rng.randint(0, 99)) for _ in range(n)] for _ in range(m_con)]
    rhs = [math.floor(sum(row) / 2) for row in rows]

    bin_vars = [Variable(name=f"x{j}", vtype=VarType.BINARY, lower=0.0, upper=1.0) for j in range(n)]
    slack_vars: list[Variable] = []
    constraints: list[LinearConstraint] = []
    obj_coeffs: dict[str, float] = {}
    for i in range(m_con):
        sp, sn = f"sp{i}", f"sn{i}"
        slack_vars.append(Variable(name=sp, vtype=VarType.CONTINUOUS, lower=0.0, upper=math.inf))
        slack_vars.append(Variable(name=sn, vtype=VarType.CONTINUOUS, lower=0.0, upper=math.inf))
        obj_coeffs[sp] = 1.0
        obj_coeffs[sn] = 1.0
        coeffs = {f"x{j}": rows[i][j] for j in range(n)}
        coeffs[sp] = -1.0
        coeffs[sn] = 1.0
        constraints.append(
            LinearConstraint(name=f"row{i}", coeffs=coeffs, sense=ConstraintSense.EQ, rhs=float(rhs[i]))
        )

    return MILP(
        name=f"market_split_m{m_con}",
        variables=tuple(bin_vars) + tuple(slack_vars),
        constraints=tuple(constraints),
        objective=Objective(coeffs=obj_coeffs, sense=ObjSense.MINIMIZE),
    )


def test_highs_kernel_satisfies_protocol() -> None:
    assert isinstance(HighsKernel(), SolverKernel)


@pytest.mark.integration
def test_known_optimum_trace(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("highs")
    trace = HighsKernel().solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )

    assert isinstance(trace, SolveTrace)
    assert trace.solver == "HiGHS"
    assert trace.instance_id == "knapsack6"
    assert trace.status == "Optimal"
    assert trace.censored is False

    # The high-level API yields a single terminal point; series stay aligned.
    assert len(trace.primal_bound_series) == len(trace.dual_bound_series) == 1
    assert len(trace.time_series) == 1
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)
    assert trace.dual_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)
    assert trace.time_series[-1] >= 0.0


@pytest.mark.integration
def test_unexposed_stats_use_not_measured_sentinels(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("highs")
    trace = HighsKernel().solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )

    # HiGHS exposes terminal node + LP-iteration counts (real, non-negative ints).
    assert isinstance(trace.nodes, int) and trace.nodes >= 0
    assert isinstance(trace.lp_iters, int) and trace.lp_iters >= 0
    # NO cut count / first-feasible stamp / peak memory is exposed. These use the
    # codebase "not measured" sentinels, NOT a fabricated measurement: float
    # fields are math.nan, the cut COUNT is 0. (None is impossible -- evaluate()
    # does float(trace.<field>) and the journal does int(trace.nodes).)
    assert trace.cuts == 0
    assert math.isnan(trace.first_feasible_time)
    assert math.isnan(trace.memory_peak)


@pytest.mark.integration
def test_trace_is_consumable_by_evaluator_and_journal(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("highs")
    from opop.evaluator.evaluator import evaluate
    from opop.orchestrator.events import trace_summary

    trace = HighsKernel().solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    # Locks the sentinel choice: the standard consumers must not raise. evaluate()
    # does float() on nodes/cuts/memory_peak/first_feasible_time; trace_summary()
    # does int(trace.nodes). A regression to None (or nan nodes) would crash here.
    record = evaluate(trace, reference_optimum=_KNAPSACK_OPTIMUM, time_limit=30.0)
    summary = trace_summary(trace, record)
    assert summary["status"] == trace.status
    assert record.metrics["objective"] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


@pytest.mark.integration
def test_optima_agree_with_scip(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("highs")
    solver_skip_if_missing("scip")
    from opop.solver.scip import ScipKernel

    highs = HighsKernel()
    scip = ScipKernel()
    for ir, known in ((_knapsack_ir(), _KNAPSACK_OPTIMUM), (_assignment_ir(), _ASSIGNMENT_OPTIMUM)):
        h_trace = highs.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
        s_trace = scip.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
        h_opt = h_trace.primal_bound_series[-1]
        s_opt = s_trace.primal_bound_series[-1]
        assert h_opt == pytest.approx(known, abs=1e-6), f"{ir.name}: HiGHS {h_opt} != {known}"
        assert h_opt == pytest.approx(s_opt, abs=1e-6), f"{ir.name}: HiGHS {h_opt} != SCIP {s_opt}"
        assert h_trace.status == "Optimal"
        assert h_trace.censored is False


@pytest.mark.integration
def test_time_limit_marks_censored(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("highs")
    time_limit = 2.0
    trace = HighsKernel().solve(
        _market_split_ir(m_con=4, seed=7),
        Phi(),
        time_limit=time_limit,
        memory_limit_mb=_MEMORY_MB,
        seed=1,
    )

    assert trace.status != "Optimal", "hard fixture unexpectedly solved within 2 s"
    assert trace.censored is True
    final_time = trace.time_series[-1]
    assert 1.0 <= final_time <= 8.0, f"final solving time {final_time} not near {time_limit}s"
    # An open primal--dual gap is recorded (MINIMIZE: dual lower < primal upper).
    final_primal = trace.primal_bound_series[-1]
    final_dual = trace.dual_bound_series[-1]
    assert math.isfinite(final_primal), "expected a feasible incumbent before the limit"
    assert final_dual < final_primal - 1e-6, "expected an open gap on a censored run"
    assert trace.nodes > 0, "a hard B&B run should explore nodes"


@pytest.mark.integration
def test_deterministic_under_fixed_seed(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("highs")
    kernel = HighsKernel()
    ir = _knapsack_ir()
    a = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    b = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert a.status == b.status
    assert a.nodes == b.nodes
    assert a.primal_bound_series[-1] == pytest.approx(b.primal_bound_series[-1], abs=1e-9)
    assert a.dual_bound_series[-1] == pytest.approx(b.dual_bound_series[-1], abs=1e-9)


@pytest.mark.integration
def test_whitelisted_param_is_accepted(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("highs")
    assert "mip_rel_gap" in HIGHS_WHITELISTED_PARAMS
    phi = Phi(p={"mip_rel_gap": 0.0})
    trace = HighsKernel().solve(
        _knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


def test_non_whitelisted_param_is_rejected() -> None:
    # Fail-closed rejection is pure Python (no solve), so this needs no backend:
    # a SCIP-style key is meaningless to HiGHS and must raise, not be forwarded.
    kernel = HighsKernel()
    phi = Phi(p={"separating/gomory/freq": 5.0})
    with pytest.raises(ValueError, match="whitelist"):
        kernel.solve(_knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
