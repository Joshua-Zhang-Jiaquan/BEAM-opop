"""Local fixtures for the solver test package.

The repository now has a project-wide ``tests/conftest.py`` that inserts
``src/`` on ``sys.path``; this module is kept for backward compatibility with
task-3 tests and provides a ``solver_skip_if_missing`` fixture that is
behaviourally identical to the global one.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture(scope="session")
def solver_skip_if_missing() -> Callable[[str], None]:
    """Return ``skip(name)`` that skips the test if a solver is unavailable.

    Availability is probed with ``opop.solver.availability.is_solver_available``.
    """
    from opop.solver.availability import is_solver_available

    def _skip(name: str) -> None:
        if not is_solver_available(name):
            pytest.skip(f"solver {name!r} not available in this environment")

    return _skip
