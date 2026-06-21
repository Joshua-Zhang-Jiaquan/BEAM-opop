"""Task-3 acceptance: solver availability probe + cross-solver agreement.

Verifies that:
  * ``available_solvers()`` returns a normalized ``{name, version, available}``
    record for every known open solver, in the canonical order.
  * Each *present* solver solves the 2-var binary knapsack
    (max x+y s.t. x+y<=1, x,y in {0,1}) to the SAME optimum (=1, OPTIMAL).

Absent solvers are skipped (``solver_skip_if_missing``), never failed, so the
suite stays green on environments missing a backend.
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from opop.solver import smoke as smoke_mod
from opop.solver.availability import (
    SOLVER_NAMES,
    available_solvers,
    is_solver_available,
    solver_infos,
)

_REQUIRED_KEYS = {"name", "version", "available"}


@pytest.mark.smoke
def test_available_solvers_schema() -> None:
    records = available_solvers()
    assert isinstance(records, list) and records, "expected a non-empty list"
    names = {r["name"] for r in records}
    assert set(SOLVER_NAMES) <= names, f"missing solvers: {set(SOLVER_NAMES) - names}"
    for record in records:
        assert _REQUIRED_KEYS <= set(record), f"record missing keys: {record}"
        assert isinstance(record["available"], bool)
        if record["available"]:
            assert record["version"], f"{record['name']} available but version is empty"


@pytest.mark.smoke
def test_solver_infos_align_with_available_solvers() -> None:
    info_names = [i.name for i in solver_infos()]
    record_names = [r["name"] for r in available_solvers()]
    assert info_names == record_names == list(SOLVER_NAMES)


@pytest.mark.smoke
@pytest.mark.parametrize("solver_name", SOLVER_NAMES)
def test_smoke_reaches_known_optimum(
    solver_name: str,
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing(solver_name)
    result = smoke_mod.smoke_solver(solver_name)
    assert result.available, f"{solver_name}: {result.detail}"
    assert result.optimal, f"{solver_name} not OPTIMAL: status={result.status!r}"
    assert result.objective == pytest.approx(smoke_mod.OPTIMUM, abs=1e-6)
    assert result.agrees()


@pytest.mark.smoke
def test_present_solvers_agree_on_optimum() -> None:
    results = [
        smoke_mod.smoke_solver(name)
        for name in SOLVER_NAMES
        if is_solver_available(name)
    ]
    if len(results) < 2:
        pytest.skip("need >=2 solvers present to check cross-solver agreement")
    assert all(r.agrees() for r in results), {r.solver: r.objective for r in results}
    distinct = {round(obj, 6) for r in results if (obj := r.objective) is not None}
    assert distinct == {round(smoke_mod.OPTIMUM, 6)}, {r.solver: r.objective for r in results}
