"""Pytest harness fixtures for OPOP tests.

Adds ``<repo>/src`` to ``sys.path[0]`` so that ``import opop`` resolves
regardless of ``PYTHONPATH``.  Supplies shared fixtures: ``fake_llm``,
``tmp_run_dir``, ``tiny_milp_fixture``, ``solver_skip_if_missing``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path -- make ``import opop`` work without PYTHONPATH=src
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_llm() -> object:
    """A deterministic, offline fake LLM client (``opop.llm.FakeLLMClient``)."""
    from opop.llm import FakeLLMClient

    return FakeLLMClient(response='{"answer": 42}')


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """A temporary run directory (``runs/<uuid>``-like) cleaned up after each test."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return run_dir


@pytest.fixture
def tiny_milp_fixture() -> dict[str, object]:
    """Return a tiny MILP (2-var knapsack) as a small dict.

    Problem:  max x + y
              s.t. x + y <= 1
              x, y in {0, 1}
    Optimal solution: (1, 0) or (0, 1) with objective = 1.
    """
    return {
        "name": "tiny_knapsack",
        "num_vars": 2,
        "num_constraints": 1,
        "sense": "maximize",
        "variables": [
            {"name": "x", "lower": 0, "upper": 1, "type": "BINARY"},
            {"name": "y", "lower": 0, "upper": 1, "type": "BINARY"},
        ],
        "constraints": [
            {
                "name": "c0",
                "sense": "<=",
                "rhs": 1.0,
                "coeffs": {"x": 1.0, "y": 1.0},
            }
        ],
        "objective": {"x": 1.0, "y": 1.0},
        "known_optimum": 1.0,
    }


@pytest.fixture
def solver_skip_if_missing() -> Callable[[str], None]:
    """Return ``skip(name)`` that skips the test if a solver is unavailable.

    Availability is probed with ``opop.solver.availability.is_solver_available``.
    """
    from opop.solver.availability import is_solver_available

    def _skip(name: str) -> None:
        if not is_solver_available(name):
            pytest.skip(f"solver {name!r} not available in this environment")

    return _skip
