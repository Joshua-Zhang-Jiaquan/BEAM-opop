"""Task-22 acceptance: CP-SAT solver kernel + solution-callback trajectory.

Verifies that :class:`opop.solver.cpsat.CpsatKernel`:

* satisfies the :class:`~opop.solver.kernel.SolverKernel` Protocol;
* solves known integer/binary MILPs to their known optima, returning a
  :class:`SolveTrace` with parallel non-empty series, ``status == "OPTIMAL"``
  and ``censored=False``, and ``memory_peak`` reported as ``nan`` / ``cuts == 0``
  (CP-SAT exposes neither a peak-memory readback nor an applied-cut count);
* honours a 2 s time limit on a hard *pure-integer* market-split fixture
  (continuous slacks would be rejected), yielding ``censored=True`` and an open
  primal--dual gap;
* is deterministic under a fixed seed, lets the budget override ``phi.p``, and
  fail-closes on unknown ``phi.p`` knobs;
* scales rational/float coefficients to integers *exactly* (never a silently
  wrong optimum) and raises a clear :class:`UnsupportedModelError` when a
  coefficient cannot be represented or a variable is continuous;
* agrees with the SCIP reference kernel on three integer/binary instances.

CP-SAT (``ortools``) is required for the solver-backed tests; absent backends are
skipped via ``solver_skip_if_missing`` / ``pytest.importorskip``. Pure-Python
scaling and Protocol tests run without any solver.
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
    UnsupportedModelError,
    Variable,
    VarType,
)
from opop.model.state import Phi, SolveTrace
from opop.solver._cpsat_utils import scale_row_to_integers
from opop.solver.cpsat import KNOWN_CPSAT_PARAMS, CpsatKernel
from opop.solver.kernel import SolverKernel

_MEMORY_MB = 4096

# Deterministic 6-item 0/1 knapsack solved by hand: items {0,2,4,5} give value
# 10+18+7+15 = 50 at weight 2+4+1+3 = 10 == capacity. Optimum is 50.
_KNAPSACK_VALUES = (10.0, 13.0, 18.0, 31.0, 7.0, 15.0)
_KNAPSACK_WEIGHTS = (2.0, 3.0, 4.0, 7.0, 1.0, 3.0)
_KNAPSACK_CAPACITY = 10.0
_KNAPSACK_OPTIMUM = 50.0


def _knapsack_ir() -> MILP:
    """A MAXIMIZE 0/1 knapsack with a known optimum of 50 (integer coefficients)."""
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


# 3x3 assignment, MINIMIZE cost. Unique optimum 5 via (0,1)+(1,0)+(2,2)=1+2+2.
_ASSIGN_COST = ((4.0, 1.0, 3.0), (2.0, 0.0, 5.0), (3.0, 2.0, 2.0))
_ASSIGN_OPTIMUM = 5.0


def _assignment_ir() -> MILP:
    """A MINIMIZE 3x3 assignment problem (binary) with a known optimum of 5."""
    n = 3
    variables = tuple(
        Variable(name=f"x_{i}_{j}", vtype=VarType.BINARY, lower=0.0, upper=1.0)
        for i in range(n)
        for j in range(n)
    )
    constraints: list[LinearConstraint] = []
    for i in range(n):
        constraints.append(
            LinearConstraint(
                name=f"row{i}",
                coeffs={f"x_{i}_{j}": 1.0 for j in range(n)},
                sense=ConstraintSense.EQ,
                rhs=1.0,
            )
        )
    for j in range(n):
        constraints.append(
            LinearConstraint(
                name=f"col{j}",
                coeffs={f"x_{i}_{j}": 1.0 for i in range(n)},
                sense=ConstraintSense.EQ,
                rhs=1.0,
            )
        )
    objective = Objective(
        coeffs={f"x_{i}_{j}": _ASSIGN_COST[i][j] for i in range(n) for j in range(n)},
        sense=ObjSense.MINIMIZE,
    )
    return MILP(
        name="assign3",
        variables=variables,
        constraints=tuple(constraints),
        objective=objective,
    )


_BOUNDED_INT_OPTIMUM = 11.0


def _bounded_integer_ir() -> MILP:
    """MAXIMIZE 3x+2y s.t. x+y<=4, x,y in [0,3] integer. Known optimum 11 (x=3,y=1)."""
    variables = (
        Variable(name="x", vtype=VarType.INTEGER, lower=0.0, upper=3.0),
        Variable(name="y", vtype=VarType.INTEGER, lower=0.0, upper=3.0),
    )
    capacity = LinearConstraint(
        name="cap", coeffs={"x": 1.0, "y": 1.0}, sense=ConstraintSense.LE, rhs=4.0
    )
    objective = Objective(coeffs={"x": 3.0, "y": 2.0}, sense=ObjSense.MAXIMIZE)
    return MILP(
        name="bounded_int",
        variables=variables,
        constraints=(capacity,),
        objective=objective,
    )


def _integer_market_split_ir(m_con: int = 6, seed: int = 7) -> MILP:
    """A pure-INTEGER Cornuejols--Dawande market split (hard for its size).

    Mirrors the SCIP test fixture but uses bounded INTEGER slacks instead of
    continuous ones (CP-SAT rejects continuous variables). ``m_con = 6`` (62
    variables) reliably exceeds a 2 s single-threaded CP-SAT budget with a wide
    open primal--dual gap.
    """
    rng = random.Random(seed)
    n = 10 * (m_con - 1)
    rows = [[rng.randint(0, 99) for _ in range(n)] for _ in range(m_con)]
    rhs = [sum(row) // 2 for row in rows]

    bin_vars = [Variable(name=f"x{j}", vtype=VarType.BINARY, lower=0.0, upper=1.0) for j in range(n)]
    slack_vars: list[Variable] = []
    constraints: list[LinearConstraint] = []
    obj_coeffs: dict[str, float] = {}
    for i in range(m_con):
        rowsum = float(sum(rows[i]))
        sp, sn = f"sp{i}", f"sn{i}"
        slack_vars.append(Variable(name=sp, vtype=VarType.INTEGER, lower=0.0, upper=rowsum))
        slack_vars.append(Variable(name=sn, vtype=VarType.INTEGER, lower=0.0, upper=rowsum))
        obj_coeffs[sp] = 1.0
        obj_coeffs[sn] = 1.0
        coeffs = {f"x{j}": float(rows[i][j]) for j in range(n)}
        coeffs[sp] = -1.0
        coeffs[sn] = 1.0
        constraints.append(
            LinearConstraint(name=f"row{i}", coeffs=coeffs, sense=ConstraintSense.EQ, rhs=float(rhs[i]))
        )

    return MILP(
        name=f"int_market_split_m{m_con}",
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


# ---------------------------------------------------------------------------
# Protocol + pure-Python scaling unit tests (no solver required)
# ---------------------------------------------------------------------------
def test_cpsat_kernel_satisfies_protocol() -> None:
    """CpsatKernel structurally satisfies the SolverKernel Protocol."""
    assert isinstance(CpsatKernel(), SolverKernel)


def test_scale_row_recovers_halves() -> None:
    """Float halves scale exactly by 2: [2.5, 1.5, 1.0] -> (2, [5, 3, 2])."""
    scale, ints = scale_row_to_integers([2.5, 1.5, 1.0])
    assert scale == 2
    assert ints == [5, 3, 2]


def test_scale_row_recovers_thirds() -> None:
    """Float thirds are recovered exactly: [1/3, 2/3] -> (3, [1, 2])."""
    scale, ints = scale_row_to_integers([1.0 / 3.0, 2.0 / 3.0])
    assert scale == 3
    assert ints == [1, 2]


def test_scale_row_empty() -> None:
    """An empty row scales to (1, [])."""
    assert scale_row_to_integers([]) == (1, [])


def test_scale_row_rejects_imprecise_coefficient() -> None:
    """A coefficient needing a denominator past the cap raises (never approximated)."""
    with pytest.raises(UnsupportedModelError, match="not exactly representable"):
        scale_row_to_integers([1.0 / 11.0], max_denominator=10)


# ---------------------------------------------------------------------------
# Solver-backed tests (CP-SAT required)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_known_optimum_trace(solver_skip_if_missing: Callable[[str], None]) -> None:
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    kernel = CpsatKernel()
    trace = kernel.solve(
        _knapsack_ir(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )

    assert isinstance(trace, SolveTrace)
    assert trace.solver == "CP-SAT"
    assert trace.instance_id == "knapsack6"
    assert trace.status == "OPTIMAL"
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

    # CP-SAT-specific terminal stats (documented semantics).
    assert trace.nodes >= 0
    assert trace.lp_iters >= 0
    assert trace.cuts == 0, "CP-SAT exposes no applied-cut count"
    assert math.isnan(trace.memory_peak), "CP-SAT exposes no peak-memory readback (nan sentinel)"


@pytest.mark.integration
def test_time_limit_marks_censored(solver_skip_if_missing: Callable[[str], None]) -> None:
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    kernel = CpsatKernel()
    time_limit = 2.0
    trace = kernel.solve(
        _integer_market_split_ir(m_con=6, seed=7),
        Phi(),
        time_limit=time_limit,
        memory_limit_mb=_MEMORY_MB,
        seed=1,
    )

    assert trace.status != "OPTIMAL", "hard fixture unexpectedly solved within 2 s"
    assert trace.censored is True
    assert trace.status in {"FEASIBLE", "UNKNOWN"}

    # Solving time is governed by the limit (CP-SAT's wall clock, ~2 s).
    final_time = trace.time_series[-1]
    assert 1.5 <= final_time <= 5.0, f"final solving time {final_time} not near {time_limit}s"

    # The dual lower bound improves monotonically over the trajectory (MINIMIZE).
    assert _is_monotone_dual(trace.dual_bound_series, ObjSense.MINIMIZE)
    assert trace.primal_bound_series, "expected a non-empty primal series"

    # The calibrated fixture reliably yields a FEASIBLE incumbent with an open
    # gap; guard the incumbent-specific assertions so a pathologically slow host
    # returning UNKNOWN (also censored) does not flake the test.
    final_primal = trace.primal_bound_series[-1]
    final_dual = trace.dual_bound_series[-1]
    if math.isfinite(final_primal):
        assert final_dual < final_primal - 1e-6, "expected an open gap on a censored run"
        assert math.isfinite(trace.first_feasible_time)
        assert trace.nodes > 0, "a hard CP-SAT run should branch"


@pytest.mark.integration
def test_deterministic_under_fixed_seed(solver_skip_if_missing: Callable[[str], None]) -> None:
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    kernel = CpsatKernel()
    ir = _knapsack_ir()
    a = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    b = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert a.status == b.status
    assert a.nodes == b.nodes
    assert a.primal_bound_series[-1] == pytest.approx(b.primal_bound_series[-1], abs=1e-9)
    assert a.dual_bound_series[-1] == pytest.approx(b.dual_bound_series[-1], abs=1e-9)


@pytest.mark.integration
def test_unknown_phi_param_is_rejected(solver_skip_if_missing: Callable[[str], None]) -> None:
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    kernel = CpsatKernel()
    phi = Phi(p={"not_a_real_cpsat_param": 1.0})
    with pytest.raises(ValueError, match="not a known/whitelisted"):
        kernel.solve(_knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)


@pytest.mark.integration
def test_whitelisted_phi_param_is_accepted(solver_skip_if_missing: Callable[[str], None]) -> None:
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    assert "linearization_level" in KNOWN_CPSAT_PARAMS
    kernel = CpsatKernel()
    phi = Phi(p={"linearization_level": 2.0, "cp_model_presolve": 1.0})
    trace = kernel.solve(_knapsack_ir(), phi, time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert trace.status == "OPTIMAL"
    assert trace.primal_bound_series[-1] == pytest.approx(_KNAPSACK_OPTIMUM, abs=1e-6)


@pytest.mark.integration
def test_budget_overrides_phi_p(solver_skip_if_missing: Callable[[str], None]) -> None:
    """phi.p is applied BEFORE the hard budget params, so the budget wins.

    phi.p asks for a 1 ms time limit and 8 workers; the kernel must override both
    with the call's 2 s budget / single worker, so the hard fixture still runs
    for ~2 s rather than stopping after 1 ms.
    """
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    kernel = CpsatKernel()
    phi = Phi(p={"max_time_in_seconds": 0.001, "num_workers": 8.0})
    trace = kernel.solve(
        _integer_market_split_ir(m_con=6, seed=7),
        phi,
        time_limit=2.0,
        memory_limit_mb=_MEMORY_MB,
        seed=1,
    )
    assert trace.time_series[-1] >= 1.5, "budget time limit must override phi.p's 1 ms request"


@pytest.mark.integration
def test_fractional_coefficients_scaled_to_correct_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """Fractional objective AND constraint coefficients are scaled exactly.

    MAX 2.5x + 1.5y + 1.0z  s.t.  0.5x + 0.5y + 0.5z <= 1.0  (binary).
    The constraint scales to x+y+z<=2 and the objective to 5x+3y+2z; picking the
    two best items {x, y} gives 2.5+1.5 = 4.0 — the exact unscaled optimum.
    """
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    variables = tuple(
        Variable(name=name, vtype=VarType.BINARY, lower=0.0, upper=1.0) for name in ("x", "y", "z")
    )
    con = LinearConstraint(
        name="cap",
        coeffs={"x": 0.5, "y": 0.5, "z": 0.5},
        sense=ConstraintSense.LE,
        rhs=1.0,
    )
    objective = Objective(coeffs={"x": 2.5, "y": 1.5, "z": 1.0}, sense=ObjSense.MAXIMIZE)
    ir = MILP(name="frac_coef", variables=variables, constraints=(con,), objective=objective)

    trace = CpsatKernel().solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert trace.status == "OPTIMAL"
    assert trace.primal_bound_series[-1] == pytest.approx(4.0, abs=1e-6)


@pytest.mark.integration
def test_unscalable_coefficient_raises_clear_error(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """A coefficient that cannot be scaled exactly raises, never a wrong optimum."""
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    variables = (Variable(name="x", vtype=VarType.BINARY, lower=0.0, upper=1.0),)
    objective = Objective(coeffs={"x": 1.0 / 11.0}, sense=ObjSense.MAXIMIZE)
    ir = MILP(name="unscalable", variables=variables, objective=objective)
    # max_denominator=10 cannot represent 1/11 within tolerance.
    kernel = CpsatKernel(max_denominator=10)
    with pytest.raises(UnsupportedModelError, match="not exactly representable"):
        kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)


@pytest.mark.integration
def test_continuous_variable_rejected(solver_skip_if_missing: Callable[[str], None]) -> None:
    """CP-SAT is integer-only: a continuous variable raises a clear error."""
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    variables = (
        Variable(name="x", vtype=VarType.BINARY, lower=0.0, upper=1.0),
        Variable(name="c", vtype=VarType.CONTINUOUS, lower=0.0, upper=1.0),
    )
    con = LinearConstraint(
        name="link", coeffs={"x": 1.0, "c": 1.0}, sense=ConstraintSense.LE, rhs=1.0
    )
    objective = Objective(coeffs={"x": 1.0, "c": 1.0}, sense=ObjSense.MAXIMIZE)
    ir = MILP(name="has_continuous", variables=variables, constraints=(con,), objective=objective)
    with pytest.raises(UnsupportedModelError, match="CONTINUOUS"):
        CpsatKernel().solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)


@pytest.mark.integration
def test_infinite_integer_bound_rejected(solver_skip_if_missing: Callable[[str], None]) -> None:
    """An INTEGER variable with a non-finite bound raises (CP-SAT needs finite domains)."""
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    variables = (Variable(name="n", vtype=VarType.INTEGER, lower=0.0, upper=math.inf),)
    objective = Objective(coeffs={"n": 1.0}, sense=ObjSense.MINIMIZE)
    ir = MILP(name="unbounded_int", variables=variables, objective=objective)
    with pytest.raises(UnsupportedModelError, match="finite integer domain"):
        CpsatKernel().solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)


# ---------------------------------------------------------------------------
# Cross-solver agreement with the SCIP reference kernel (the headline test)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize(
    ("ir_builder", "known_optimum"),
    [
        (_knapsack_ir, _KNAPSACK_OPTIMUM),
        (_assignment_ir, _ASSIGN_OPTIMUM),
        (_bounded_integer_ir, _BOUNDED_INT_OPTIMUM),
    ],
)
def test_cpsat_agrees_with_scip(
    solver_skip_if_missing: Callable[[str], None],
    ir_builder: Callable[[], MILP],
    known_optimum: float,
) -> None:
    """CP-SAT and SCIP reach the same proven optimum on integer/binary instances."""
    pytest.importorskip("ortools")
    solver_skip_if_missing("cpsat")
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    ir = ir_builder()
    cpsat_trace = CpsatKernel().solve(
        ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    scip_trace = ScipKernel().solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)

    assert cpsat_trace.status == "OPTIMAL"
    assert scip_trace.status == "optimal"
    assert cpsat_trace.censored is False
    assert scip_trace.censored is False

    cpsat_opt = cpsat_trace.primal_bound_series[-1]
    scip_opt = scip_trace.primal_bound_series[-1]
    assert cpsat_opt == pytest.approx(known_optimum, abs=1e-6)
    assert scip_opt == pytest.approx(known_optimum, abs=1e-6)
    assert cpsat_opt == pytest.approx(scip_opt, abs=1e-6)
