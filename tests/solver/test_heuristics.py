"""Task-25 acceptance: generic MIP matheuristic cores.

Verifies :mod:`opop.solver.heuristics`:

* the pure feasibility checker flags out-of-capacity / non-integral assignments
  (no solver needed);
* ``local_branching`` reaches the optimum within a 1-flip neighbourhood and
  returns the incumbent unchanged at ``k=0``;
* ``rins`` fixes incumbent-agreeing variables and improves to the optimum;
* ``large_neighborhood_search`` improves or matches a deliberately-suboptimal
  incumbent (and reaches the optimum at full destroy);
* ``repair_solution`` returns a verified-feasible solution closest to a target,
  and reports failure (never an unchecked infeasible result) on an infeasible
  model;
* every returned incumbent is feasibility-checked and the heuristics are
  deterministic under a fixed seed.

SCIP-backed cases are marked ``integration`` and skipped if SCIP is absent.
"""

from __future__ import annotations

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
from opop.solver.heuristics import (
    HeuristicResult,
    is_solution_feasible,
    large_neighborhood_search,
    local_branching,
    repair_solution,
    rins,
    solution_violations,
)

_MEMORY_MB = 4096
_TIME_LIMIT = 30.0

# Deterministic 6-item 0/1 knapsack (== task-12 fixture): items {0,2,4,5} give
# value 10+18+7+15 = 50 at weight 2+4+1+3 = 10 == capacity. Optimum is 50.
_VALUES = (10.0, 13.0, 18.0, 31.0, 7.0, 15.0)
_WEIGHTS = (2.0, 3.0, 4.0, 7.0, 1.0, 3.0)
_CAPACITY = 10.0
_OPTIMUM = 50.0
_OPTIMAL_SET = {"x0", "x2", "x4", "x5"}


def _knapsack_ir() -> MILP:
    """A MAXIMIZE 0/1 knapsack with a known optimum of 50."""
    n = len(_VALUES)
    variables = tuple(
        Variable(name=f"x{i}", vtype=VarType.BINARY, lower=0.0, upper=1.0) for i in range(n)
    )
    capacity = LinearConstraint(
        name="capacity",
        coeffs={f"x{i}": _WEIGHTS[i] for i in range(n)},
        sense=ConstraintSense.LE,
        rhs=_CAPACITY,
    )
    objective = Objective(
        coeffs={f"x{i}": _VALUES[i] for i in range(n)}, sense=ObjSense.MAXIMIZE
    )
    return MILP(name="knap6", variables=variables, constraints=(capacity,), objective=objective)


def _assignment(selected: set[str]) -> dict[str, float]:
    """Build a complete binary assignment selecting exactly ``selected``."""
    return {f"x{i}": (1.0 if f"x{i}" in selected else 0.0) for i in range(len(_VALUES))}


# A deliberately-suboptimal feasible incumbent: {x0,x2,x4} → weight 7, value 35.
# It differs from the optimum {x0,x2,x4,x5} by exactly ONE flip (x5: 0 -> 1).
_SUBOPTIMAL = _assignment({"x0", "x2", "x4"})
_SUBOPTIMAL_VALUE = 35.0
_ALL_ZERO = _assignment(set())


def _infeasible_ir() -> MILP:
    """A trivially infeasible MILP: a binary x constrained to be both >=1 and <=0."""
    variables = (Variable(name="x", vtype=VarType.BINARY, lower=0.0, upper=1.0),)
    constraints = (
        LinearConstraint(name="lo", coeffs={"x": 1.0}, sense=ConstraintSense.GE, rhs=1.0),
        LinearConstraint(name="hi", coeffs={"x": 1.0}, sense=ConstraintSense.LE, rhs=0.0),
    )
    objective = Objective(coeffs={"x": 1.0}, sense=ObjSense.MINIMIZE)
    return MILP(name="infeasible", variables=variables, constraints=constraints, objective=objective)


# ---------------------------------------------------------------------------
# Pure feasibility checker (no solver)
# ---------------------------------------------------------------------------
def test_feasibility_checker_accepts_valid_assignment() -> None:
    ir = _knapsack_ir()
    assert is_solution_feasible(ir, _SUBOPTIMAL)
    assert is_solution_feasible(ir, _assignment(_OPTIMAL_SET))
    assert solution_violations(ir, _ALL_ZERO) == []


def test_feasibility_checker_rejects_over_capacity() -> None:
    ir = _knapsack_ir()
    over = _assignment({"x0", "x1", "x2", "x3", "x4", "x5"})  # weight 20 > capacity 10
    violations = solution_violations(ir, over)
    assert violations, "all-ones must violate the capacity constraint"
    assert not is_solution_feasible(ir, over)


def test_feasibility_checker_rejects_non_integral_and_incomplete() -> None:
    ir = _knapsack_ir()
    fractional = dict(_ALL_ZERO)
    fractional["x0"] = 0.5
    assert not is_solution_feasible(ir, fractional)

    incomplete = {"x0": 1.0}
    assert not is_solution_feasible(ir, incomplete)


# ---------------------------------------------------------------------------
# local branching
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_local_branching_reaches_optimum_within_one_flip(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()
    result = local_branching(ir, _SUBOPTIMAL, k=1, time_limit=_TIME_LIMIT, seed=0)

    assert isinstance(result, HeuristicResult)
    assert result.feasible
    assert result.incumbent is not None
    assert is_solution_feasible(ir, result.incumbent)
    assert result.improved
    assert result.objective == pytest.approx(_OPTIMUM, abs=1e-6)
    assert result.info["k"] == 1
    assert result.traces and result.traces[0].solver == "SCIP"


@pytest.mark.integration
def test_local_branching_k0_returns_incumbent(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()
    result = local_branching(ir, _SUBOPTIMAL, k=0, time_limit=_TIME_LIMIT, seed=0)

    assert result.feasible
    assert not result.improved
    assert result.incumbent is not None
    assert is_solution_feasible(ir, result.incumbent)
    assert result.objective == pytest.approx(_SUBOPTIMAL_VALUE, abs=1e-6)


# ---------------------------------------------------------------------------
# RINS
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_rins_improves_to_optimum(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()
    result = rins(ir, _SUBOPTIMAL, time_limit=_TIME_LIMIT, seed=0)

    assert result.feasible
    assert result.incumbent is not None
    assert is_solution_feasible(ir, result.incumbent)
    assert result.objective >= _SUBOPTIMAL_VALUE - 1e-6
    assert result.objective == pytest.approx(_OPTIMUM, abs=1e-6)
    assert "n_fixed" in result.info
    assert "lp_objective" in result.info
    assert len(result.traces) == 2  # LP relaxation + restricted MIP


# ---------------------------------------------------------------------------
# Large Neighborhood Search
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_lns_improves_or_matches_suboptimal(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()
    result = large_neighborhood_search(
        ir, _SUBOPTIMAL, destroy_frac=0.5, n_iter=3, time_limit=5.0, seed=7
    )

    assert result.feasible
    assert result.incumbent is not None
    assert is_solution_feasible(ir, result.incumbent)
    # Improves or matches the deliberately-suboptimal incumbent (MAXIMIZE).
    assert result.objective >= _SUBOPTIMAL_VALUE - 1e-6
    assert result.info["n_iter"] == 3
    assert result.info["n_accepted"] >= 0


@pytest.mark.integration
def test_lns_full_destroy_reaches_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()
    # destroy_frac=1.0 frees every variable → the sub-MIP is the full problem.
    result = large_neighborhood_search(
        ir, _ALL_ZERO, destroy_frac=1.0, n_iter=1, time_limit=_TIME_LIMIT, seed=1
    )

    assert result.feasible
    assert result.improved
    assert result.incumbent is not None
    assert is_solution_feasible(ir, result.incumbent)
    assert result.objective == pytest.approx(_OPTIMUM, abs=1e-6)
    assert result.info["n_accepted"] == 1


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_repair_returns_feasible_solution(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()
    # All-ones is infeasible (weight 20 > 10); repair finds the closest feasible.
    target = _assignment({"x0", "x1", "x2", "x3", "x4", "x5"})
    result = repair_solution(ir, target, time_limit=_TIME_LIMIT, seed=0)

    assert result.status == "repaired"
    assert result.feasible
    assert result.incumbent is not None
    assert is_solution_feasible(ir, result.incumbent)
    assert "distance" in result.info


@pytest.mark.integration
def test_repair_reports_failure_on_infeasible_model(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _infeasible_ir()
    result = repair_solution(ir, {"x": 1.0}, time_limit=_TIME_LIMIT, seed=0)

    # Never an unchecked infeasible result: failure is reported explicitly.
    assert result.status == "repair_failed"
    assert result.feasible is False
    assert result.incumbent is None


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_heuristics_are_deterministic(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    ir = _knapsack_ir()

    a = rins(ir, _SUBOPTIMAL, time_limit=_TIME_LIMIT, seed=3)
    b = rins(ir, _SUBOPTIMAL, time_limit=_TIME_LIMIT, seed=3)
    assert a.incumbent == b.incumbent
    assert a.objective == pytest.approx(b.objective, abs=1e-9)

    c = large_neighborhood_search(
        ir, _ALL_ZERO, destroy_frac=0.5, n_iter=2, time_limit=5.0, seed=11
    )
    d = large_neighborhood_search(
        ir, _ALL_ZERO, destroy_frac=0.5, n_iter=2, time_limit=5.0, seed=11
    )
    assert c.incumbent == d.incumbent
    assert c.objective == pytest.approx(d.objective, abs=1e-9)
