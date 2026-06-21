"""Tiny-MILP smoke tests for each open solver (task-3 agreement check).

Canonical model (binary knapsack)::

    maximize   x + y
    subject to x + y <= 1
               x, y in {0, 1}

Optimal objective = 1. Every installed backend must reach status OPTIMAL with
objective 1, proving the solver is installed, callable, and *mutually
consistent* with the others.

Each ``smoke_*`` returns a :class:`SmokeResult`. A backend that is absent or
fails to import yields ``available=False`` (to be skipped, never failed) so the
suite stays green on machines missing a solver. Import errors are captured in
``detail`` and surfaced, not masked.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

#: Known optimum of the canonical smoke model.
OPTIMUM: float = 1.0
_TOL: float = 1e-6


@dataclasses.dataclass(frozen=True)
class SmokeResult:
    """Outcome of solving the canonical tiny MILP with one backend."""

    solver: str
    available: bool
    objective: float | None
    status: str
    optimal: bool
    detail: str = ""

    def agrees(self, optimum: float = OPTIMUM, tol: float = _TOL) -> bool:
        """True iff the backend ran, hit OPTIMAL, and matched ``optimum``."""
        return (
            self.available
            and self.optimal
            and self.objective is not None
            and abs(self.objective - optimum) <= tol
        )


def _unavailable(name: str, exc: BaseException) -> SmokeResult:
    return SmokeResult(
        name, False, None, "unavailable", False,
        f"{type(exc).__name__}: {exc}",
    )


def smoke_scip() -> SmokeResult:
    name = "SCIP"
    try:
        from pyscipopt import Model
    except Exception as exc:
        return _unavailable(name, exc)
    model = Model()
    model.hideOutput()
    x = model.addVar(vtype="B", name="x")
    y = model.addVar(vtype="B", name="y")
    model.addCons(x + y <= 1)
    model.setObjective(x + y, sense="maximize")
    model.optimize()
    status = model.getStatus()  # 'optimal'
    optimal = status == "optimal"
    objective = model.getObjVal() if optimal else None
    return SmokeResult(name, True, objective, status, optimal)


def smoke_cpsat() -> SmokeResult:
    name = "CP-SAT"
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        return _unavailable(name, exc)
    model = cp_model.CpModel()
    x = model.new_bool_var("x")
    y = model.new_bool_var("y")
    model.add(x + y <= 1)
    model.maximize(x + y)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    status_name = solver.status_name(status)  # 'OPTIMAL'
    optimal = status_name == "OPTIMAL"
    objective = float(solver.objective_value) if optimal else None
    return SmokeResult(name, True, objective, status_name, optimal)


def smoke_highs() -> SmokeResult:
    name = "HiGHS"
    try:
        import highspy
    except Exception as exc:
        return _unavailable(name, exc)
    h = highspy.Highs()
    h.silent()
    x = h.addBinary()
    y = h.addBinary()
    h.addConstr(x + y <= 1)
    h.maximize(x + y)
    status = h.modelStatusToString(h.getModelStatus())  # 'Optimal'
    optimal = status.lower() == "optimal"
    objective = h.getObjectiveValue() if optimal else None
    return SmokeResult(name, True, objective, status, optimal)


def smoke_cbc() -> SmokeResult:
    name = "CBC"
    try:
        import pulp
    except Exception as exc:
        return _unavailable(name, exc)
    cbc = pulp.PULP_CBC_CMD(msg=False)
    if not cbc.available():
        return SmokeResult(name, False, None, "unavailable", False, "PuLP bundled CBC not found")
    problem = pulp.LpProblem("opop_smoke_knapsack", pulp.LpMaximize)
    x = pulp.LpVariable("x", cat="Binary")
    y = pulp.LpVariable("y", cat="Binary")
    problem += x + y  # objective
    problem += x + y <= 1  # constraint
    problem.solve(cbc)
    status = pulp.LpStatus[problem.status]  # 'Optimal'
    optimal = status == "Optimal"
    objective = float(pulp.value(problem.objective)) if optimal else None
    return SmokeResult(name, True, objective, status, optimal)


_SMOKES: dict[str, Callable[[], SmokeResult]] = {
    "SCIP": smoke_scip,
    "CP-SAT": smoke_cpsat,
    "HiGHS": smoke_highs,
    "CBC": smoke_cbc,
}


def smoke_solver(name: str) -> SmokeResult:
    """Run the smoke model for one canonical solver name (see SOLVER_NAMES)."""
    try:
        fn = _SMOKES[name]
    except KeyError:
        raise KeyError(f"unknown solver {name!r}; known: {list(_SMOKES)}") from None
    return fn()


def run_all_smoke() -> list[SmokeResult]:
    """Run the smoke model for every known solver (order = SOLVER_NAMES)."""
    return [smoke_scip(), smoke_cpsat(), smoke_highs(), smoke_cbc()]
