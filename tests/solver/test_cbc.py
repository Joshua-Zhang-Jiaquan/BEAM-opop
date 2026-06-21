"""Task-23 acceptance: CBC solver kernel (via PuLP's bundled CBC binary).

Verifies that :class:`opop.solver.cbc.CbcKernel`:

* solves a known 0/1 knapsack to its optimum (50) and a 3x3 assignment to its
  optimum (9), with the final primal AGREEING with
  :class:`opop.solver.scip.ScipKernel` on both binary/integer fixtures;
* degrades the trajectory to a SINGLE point and reports every signal PuLP/CBC
  does not expose (dual bound, nodes, LP iters, cuts, first-feasible, peak
  memory) as ``None`` -- never a fabricated value;
* marks a time-limited run on a hard market-split instance as ``censored=True``
  (the reliable signal is ``sol_status`` feasible-but-not-proven, since PuLP
  mislabels the problem ``status`` as optimal on a timeout);
* satisfies the :class:`~opop.solver.kernel.SolverKernel` Protocol, is
  deterministic in objective/status under a fixed seed, and fail-closes on
  non-whitelisted params.

CBC is OPTIONAL: every solving test SKIPS (never fails) via
``solver_skip_if_missing("cbc")`` when the bundled binary is unavailable.
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
from opop.solver.cbc import CBC_WHITELISTED_PARAMS, CbcKernel
from opop.solver.kernel import SolverKernel

_MEMORY_MB = 4096

# Same fixtures as the HiGHS/SCIP suites: 6-item knapsack (optimum 50) and a
# 3x3 assignment whose min perfect matching costs 9 (confirmed against SCIP).
_KNAPSACK_OPTIMUM = 50.0
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


def test_cbc_kernel_satisfies_protocol() -> None:
    assert isinstance(CbcKernel(), SolverKernel)


@pytest.mark.integration
def test_known_optimum_trace(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("cbc")
    trace = CbcKernel().solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )

    assert isinstance(trace, SolveTrace)
    assert trace.solver == "CBC"
    assert trace.instance_id == "knapsack6"
    assert trace.status == "Optimal Solution Found"
    assert trace.censored is False
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


@pytest.mark.integration
def test_trajectory_degraded_to_single_point(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("cbc")
    trace = CbcKernel().solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )

    # Single point: a real primal, an aligned (not-measured) dual, a timestamp.
    assert len(trace.primal_bound_series) == 1
    assert len(trace.dual_bound_series) == 1
    assert len(trace.time_series) == 1
    assert math.isfinite(trace.primal_bound_series[-1])
    assert trace.time_series[-1] >= 0.0

    # PuLP/CBC surfaces no dual bound and no B&B counters. Float signals (dual,
    # first-feasible, peak memory) use the math.nan "not measured" sentinel; the
    # integer COUNTS use 0. None is never used -- it would crash evaluate()'s
    # float(...) and the journal's int(trace.nodes) (see the consumability test).
    assert math.isnan(trace.dual_bound_series[-1])
    assert trace.nodes == 0
    assert trace.lp_iters == 0
    assert trace.cuts == 0
    assert math.isnan(trace.first_feasible_time)
    assert math.isnan(trace.memory_peak)


@pytest.mark.integration
def test_trace_is_consumable_by_evaluator_and_journal(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("cbc")
    from opop.evaluator.evaluator import evaluate
    from opop.orchestrator.events import trace_summary

    trace = CbcKernel().solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    # Locks the sentinel choice: the standard consumers must not raise. evaluate()
    # does float() on nodes/cuts/memory_peak/first_feasible_time; trace_summary()
    # does int(trace.nodes). A regression to None (or nan counts) would crash here.
    record = evaluate(trace, reference_optimum=_KNAPSACK_OPTIMUM, time_limit=30.0)
    summary = trace_summary(trace, record)
    assert summary["status"] == trace.status
    assert record.metrics["objective"] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


@pytest.mark.integration
def test_optima_agree_with_scip(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("cbc")
    solver_skip_if_missing("scip")
    from opop.solver.scip import ScipKernel

    cbc = CbcKernel()
    scip = ScipKernel()
    for ir, known in ((_knapsack_ir(), _KNAPSACK_OPTIMUM), (_assignment_ir(), _ASSIGNMENT_OPTIMUM)):
        c_trace = cbc.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
        s_trace = scip.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
        c_opt = c_trace.primal_bound_series[-1]
        s_opt = s_trace.primal_bound_series[-1]
        assert c_opt == pytest.approx(known, abs=1e-6), f"{ir.name}: CBC {c_opt} != {known}"
        assert c_opt == pytest.approx(s_opt, abs=1e-6), f"{ir.name}: CBC {c_opt} != SCIP {s_opt}"
        assert c_trace.status == "Optimal Solution Found"
        assert c_trace.censored is False


@pytest.mark.integration
def test_time_limit_marks_censored(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("cbc")
    time_limit = 2.0
    trace = CbcKernel().solve(
        _market_split_ir(m_con=4, seed=7),
        Phi(),
        time_limit=time_limit,
        memory_limit_mb=_MEMORY_MB,
        seed=1,
    )

    # Feasible-but-not-proven: PuLP labels the problem "Optimal" but the SOLUTION
    # status is "Solution Found" -> the run is right-censored.
    assert trace.censored is True
    assert trace.status == "Solution Found"
    assert math.isfinite(trace.primal_bound_series[-1]), "expected a feasible incumbent"
    final_time = trace.time_series[-1]
    assert 1.0 <= final_time <= 10.0, f"final solving time {final_time} not near {time_limit}s"


@pytest.mark.integration
def test_deterministic_under_fixed_seed(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("cbc")
    kernel = CbcKernel()
    ir = _knapsack_ir()
    # Objective + status are deterministic under a fixed -randomCbcSeed; the
    # subprocess wall-clock time is not, so it is deliberately not compared.
    a = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    b = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert a.status == b.status
    assert a.censored == b.censored
    assert a.primal_bound_series[-1] == pytest.approx(b.primal_bound_series[-1], abs=1e-9)


@pytest.mark.integration
def test_whitelisted_param_is_accepted(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("cbc")
    assert "gapRel" in CBC_WHITELISTED_PARAMS
    phi = Phi(p={"gapRel": 0.0})
    trace = CbcKernel().solve(
        _knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


def test_non_whitelisted_param_is_rejected(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("cbc")
    # A HiGHS/SCIP-style key is meaningless to CBC and must raise (fail-closed),
    # not be silently forwarded as a PULP_CBC_CMD kwarg.
    kernel = CbcKernel()
    phi = Phi(p={"mip_rel_gap": 0.5})
    with pytest.raises(ValueError, match="whitelist"):
        kernel.solve(_knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
