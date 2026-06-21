"""Generate the three static MPS fixtures used by ``tests/model/test_ir.py``.

These fixtures are produced once with PySCIPOpt (SCIP 10.x) so the round-trip
tests read *real* SCIP-written MPS files, not hand-authored text. Regenerate
with::

    python tests/model/fixtures/_generate_fixtures.py

The three fixtures deliberately span the full Phase-1 linear MILP subset:

* ``knapsack.mps``   — BINARY vars, single ``<=`` capacity row, **maximize**.
* ``assignment.mps`` — BINARY vars, six ``=`` rows (3x3 assignment), minimize.
* ``production.mps``  — INTEGER + CONTINUOUS + BINARY vars, ``>=`` / ``<=`` / ``=``
  rows, **minimize**, plus a non-zero objective offset.

Together they exercise every supported variable type, every constraint sense,
both objective senses, and the objective-offset path.
"""

from __future__ import annotations

from pathlib import Path

from pyscipopt import Model, quicksum

_HERE = Path(__file__).resolve().parent


def _write(model: Model, name: str) -> Path:
    path = _HERE / name
    model.hideOutput()
    model.writeProblem(str(path))
    return path


def build_knapsack() -> Model:
    """0/1 knapsack: maximize value subject to a single capacity row."""
    model = Model("knapsack")
    values = [8.0, 5.0, 11.0, 6.0, 9.0]
    weights = [5.0, 3.0, 7.0, 4.0, 6.0]
    capacity = 12.0
    items = [model.addVar(vtype="B", name=f"item{i}") for i in range(len(values))]
    model.addCons(
        quicksum(w * x for w, x in zip(weights, items, strict=True)) <= capacity,
        name="capacity",
    )
    model.setObjective(
        quicksum(v * x for v, x in zip(values, items, strict=True)),
        sense="maximize",
    )
    return model


def build_assignment() -> Model:
    """3x3 assignment: minimize cost with one-per-row / one-per-col equalities."""
    model = Model("assignment")
    cost = [
        [4.0, 2.0, 8.0],
        [4.0, 3.0, 7.0],
        [3.0, 1.0, 6.0],
    ]
    n = 3
    x = {
        (i, j): model.addVar(vtype="B", name=f"x_{i}_{j}")
        for i in range(n)
        for j in range(n)
    }
    for i in range(n):
        model.addCons(quicksum(x[(i, j)] for j in range(n)) == 1, name=f"row_{i}")
    for j in range(n):
        model.addCons(quicksum(x[(i, j)] for i in range(n)) == 1, name=f"col_{j}")
    model.setObjective(
        quicksum(cost[i][j] * x[(i, j)] for i in range(n) for j in range(n)),
        sense="minimize",
    )
    return model


def build_production() -> Model:
    """Tiny mixed-integer plan: INTEGER + CONTINUOUS + BINARY, all three senses.

    Includes a non-zero objective offset so the offset round-trip is covered.
    """
    model = Model("production")
    make_a = model.addVar(vtype="I", name="make_a", lb=0, ub=10)
    make_b = model.addVar(vtype="I", name="make_b", lb=0, ub=10)
    buy = model.addVar(vtype="C", name="buy", lb=0, ub=20)
    line = model.addVar(vtype="B", name="line")
    # demand satisfaction (>=)
    model.addCons(make_a + make_b + buy >= 8, name="demand")
    # capacity with a big-M switch (<=)
    model.addCons(make_a + make_b - 100 * line <= 6, name="capacity")
    # material balance (=)
    model.addCons(make_a - 2 * make_b + buy == 3, name="balance")
    model.setObjective(
        2 * make_a + 3 * make_b + 5 * buy + 10 * line,
        sense="minimize",
    )
    model.addObjoffset(7.0)
    return model


def main() -> None:
    builders = {
        "knapsack.mps": build_knapsack,
        "assignment.mps": build_assignment,
        "production.mps": build_production,
    }
    for filename, builder in builders.items():
        path = _write(builder(), filename)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
