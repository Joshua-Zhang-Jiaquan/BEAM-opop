"""Task-12 acceptance: SCIP solver kernel + event-based trajectory extraction.

Verifies that :class:`opop.solver.scip.ScipKernel`:

* solves a known MILP to its known optimum, returning a :class:`SolveTrace`
  with a non-empty primal series, a monotone dual series, and ``status``
  ``"optimal"`` / ``censored=False``;
* honours a 2 s time limit on a hard fixture (a Cornuejols--Dawande market-split
  instance), yielding ``censored=True``, ``status != "optimal"``, an open
  primal--dual gap, and a solving time of about 2 s;
* satisfies the :class:`~opop.solver.kernel.SolverKernel` Protocol, is
  deterministic under a fixed seed, and fail-closes on non-whitelisted
  separator injections.

SCIP is required; absent backends are skipped via ``solver_skip_if_missing``.
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
from opop.solver.kernel import SolverKernel
from opop.solver.scip import WHITELISTED_SEPARATORS, ScipKernel

_MEMORY_MB = 4096

# Deterministic 6-item 0/1 knapsack solved by hand: items {0,2,4,5} give value
# 10+18+7+15 = 50 at weight 2+4+1+3 = 10 == capacity. Optimum is 50.
_KNAPSACK_VALUES = (10.0, 13.0, 18.0, 31.0, 7.0, 15.0)
_KNAPSACK_WEIGHTS = (2.0, 3.0, 4.0, 7.0, 1.0, 3.0)
_KNAPSACK_CAPACITY = 10.0
_KNAPSACK_OPTIMUM = 50.0


def _knapsack_ir() -> MILP:
    """A MAXIMIZE 0/1 knapsack with a known optimum of 50."""
    n = len(_KNAPSACK_VALUES)
    variables = tuple(
        Variable(name=f"x{i}", vtype=VarType.BINARY, lower=0.0, upper=1.0) for i in range(n)
    )
    capacity = LinearConstraint(
        name="capacity",
        coeffs={f"x{i}": _KNAPSACK_WEIGHTS[i] for i in range(n)},
        sense=ConstraintSense.LE,
        rhs=_KNAPSACK_CAPACITY,
    )
    objective = Objective(
        coeffs={f"x{i}": _KNAPSACK_VALUES[i] for i in range(n)},
        sense=ObjSense.MAXIMIZE,
    )
    return MILP(
        name="knapsack6",
        variables=variables,
        constraints=(capacity,),
        objective=objective,
    )


def _market_split_ir(m_con: int = 4, seed: int = 7) -> MILP:
    """A Cornuejols--Dawande market-split instance (hard for its size).

    ``m_con`` equality rows over ``n = 10*(m_con-1)`` binaries with coefficients
    in ``[0, 99]`` and rhs ``floor(sum/2)``; per-row continuous slacks ``sp``/``sn``
    turn feasibility into a MINIMIZE objective ``sum(sp + sn)``. ``m_con = 4``
    (30 binaries) reliably exceeds a 2 s SCIP budget.
    """
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
        # sum_j A[i][j] x_j - sp_i + sn_i == rhs_i
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


def _finite(series: list[float]) -> list[float]:
    return [x for x in series if math.isfinite(x)]


def _is_monotone_dual(series: list[float], sense: ObjSense, tol: float = 1e-6) -> bool:
    """Dual bound is non-decreasing for MINIMIZE, non-increasing for MAXIMIZE."""
    finite = _finite(series)
    if sense is ObjSense.MINIMIZE:
        return all(finite[i] <= finite[i + 1] + tol for i in range(len(finite) - 1))
    return all(finite[i] >= finite[i + 1] - tol for i in range(len(finite) - 1))


def test_scip_kernel_satisfies_protocol() -> None:
    """ScipKernel structurally satisfies the SolverKernel Protocol."""
    assert isinstance(ScipKernel(), SolverKernel)


@pytest.mark.smoke
def test_known_optimum_trace(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("SCIP")
    kernel = ScipKernel()
    trace = kernel.solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )

    assert isinstance(trace, SolveTrace)
    assert trace.solver == "SCIP"
    assert trace.instance_id == "knapsack6"
    assert trace.status == "optimal"
    assert trace.censored is False

    # Series are parallel and non-empty; final point is the proven optimum.
    assert len(trace.primal_bound_series) == len(trace.dual_bound_series) == len(trace.time_series)
    assert trace.primal_bound_series, "expected a non-empty primal series"
    assert trace.dual_bound_series, "expected a non-empty dual series"
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)
    assert trace.dual_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)

    # A feasible incumbent was recorded (finite primal) and timed.
    assert _finite(trace.primal_bound_series), "expected at least one finite primal bound"
    assert math.isfinite(trace.first_feasible_time)
    assert trace.first_feasible_time >= 0.0

    # MAXIMIZE: dual (upper) bound is non-increasing toward the optimum.
    assert _is_monotone_dual(trace.dual_bound_series, ObjSense.MAXIMIZE)

    assert trace.nodes >= 0
    assert trace.lp_iters >= 0
    assert trace.cuts >= 0
    assert trace.memory_peak >= 0.0


@pytest.mark.smoke
def test_time_limit_marks_censored(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("SCIP")
    kernel = ScipKernel()
    time_limit = 2.0
    trace = kernel.solve(
        _market_split_ir(m_con=4, seed=7),
        Phi(),
        time_limit=time_limit,
        memory_limit_mb=_MEMORY_MB,
        seed=1,
    )

    assert trace.status != "optimal", "hard fixture unexpectedly solved within 2 s"
    assert trace.censored is True

    # Solving time is governed by the limit (SCIP's internal clock, ~2 s).
    final_time = trace.time_series[-1]
    assert 1.5 <= final_time <= 4.0, f"final solving time {final_time} not near {time_limit}s"

    # An open primal--dual gap is recorded (MINIMIZE: dual lower < primal upper).
    final_primal = trace.primal_bound_series[-1]
    final_dual = trace.dual_bound_series[-1]
    assert math.isfinite(final_primal), "expected a feasible incumbent before the limit"
    assert final_dual < final_primal - 1e-6, "expected an open gap on a censored run"

    # The dual lower bound improves monotonically over the trajectory.
    assert _is_monotone_dual(trace.dual_bound_series, ObjSense.MINIMIZE)
    assert trace.primal_bound_series, "expected a non-empty primal series"
    assert math.isfinite(trace.first_feasible_time)
    assert trace.nodes > 0, "a hard B&B run should explore nodes"


@pytest.mark.smoke
def test_deterministic_under_fixed_seed(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("SCIP")
    kernel = ScipKernel()
    ir = _knapsack_ir()
    a = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    b = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert a.status == b.status
    assert a.nodes == b.nodes
    assert a.primal_bound_series[-1] == pytest.approx(b.primal_bound_series[-1], abs=1e-9)
    assert a.dual_bound_series[-1] == pytest.approx(b.dual_bound_series[-1], abs=1e-9)


@pytest.mark.smoke
def test_whitelisted_separator_param_is_accepted(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    assert "gomory" in WHITELISTED_SEPARATORS
    kernel = ScipKernel()
    phi = Phi(p={"separating/gomory/freq": 5.0})
    trace = kernel.solve(_knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


@pytest.mark.smoke
def test_non_whitelisted_separator_is_rejected(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    kernel = ScipKernel()
    phi = Phi(p={"separating/notawhitelistedsep/freq": 5.0})
    with pytest.raises(ValueError, match="whitelist"):
        kernel.solve(_knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
