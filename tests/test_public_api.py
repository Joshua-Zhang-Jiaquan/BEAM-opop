"""Public-API contract tests for the ``opop`` package (task 41).

Locks the re-export surface defined in ``src/opop/__init__.py``: every name in
``opop.__all__`` resolves, the lazy loader maps the documented names to the right
objects, the two runnable entry points are modules, and no commercial solver
(``gurobipy``) ever loads on the public path.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import opop


def test_version_matches_metadata() -> None:
    """``opop.__version__`` is the packaged version string."""
    assert opop.__version__ == "0.1.0"


def test_all_is_sorted_and_unique() -> None:
    """``__all__`` is a sorted, duplicate-free list (stable public surface)."""
    names = opop.__all__
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_every_public_name_resolves() -> None:
    """Every name in ``__all__`` is importable via attribute access and from-import."""
    for name in opop.__all__:
        assert hasattr(opop, name), f"opop.{name} failed to resolve"


def test_required_public_names_present() -> None:
    """The names the task contract enumerates are all exported."""
    required = {
        "run",
        "replay",
        "BenchmarkRegistry",
        "ScipKernel",
        "Phase1Controller",
        "run_loop",
        "evaluate",
        "compare",
        "load_config",
    }
    assert required <= set(opop.__all__)


def test_run_and_replay_are_modules_with_main() -> None:
    """``opop.run`` / ``opop.replay`` are the runnable entry-point modules."""
    assert isinstance(opop.run, ModuleType)
    assert isinstance(opop.replay, ModuleType)
    assert callable(opop.run.main)
    assert callable(opop.replay.main)
    assert callable(opop.run.run_phase1_smoke)
    assert callable(opop.replay.replay_run)


def test_reexports_have_correct_identity() -> None:
    """Re-exported symbols are the *same* objects as their source modules'."""
    from opop.bench.registry import BenchmarkRegistry
    from opop.config import load_config
    from opop.controller.phase1 import Phase1Controller
    from opop.evaluator import evaluate
    from opop.experiments.compare import compare
    from opop.orchestrator import run_loop
    from opop.solver.scip import ScipKernel
    from opop.verify import verify_delta

    assert opop.run_loop is run_loop
    assert opop.load_config is load_config
    assert opop.BenchmarkRegistry is BenchmarkRegistry
    assert opop.ScipKernel is ScipKernel
    assert opop.Phase1Controller is Phase1Controller
    assert opop.evaluate is evaluate
    assert opop.compare is compare
    assert opop.verify_delta is verify_delta


def test_lazy_attribute_is_cached() -> None:
    """Accessing a lazy symbol twice returns the identical cached object."""
    first = opop.ScipKernel
    second = opop.ScipKernel
    assert first is second


def test_unknown_attribute_raises_attribute_error() -> None:
    """An undefined attribute raises ``AttributeError`` (not a bare import error)."""
    with pytest.raises(AttributeError):
        _ = opop.definitely_not_a_public_symbol


def test_scip_kernel_satisfies_solver_kernel_protocol() -> None:
    """``ScipKernel`` structurally satisfies the ``SolverKernel`` Protocol."""
    if not opop.is_solver_available("SCIP"):
        pytest.skip("SCIP backend not available")
    assert isinstance(opop.ScipKernel(), opop.SolverKernel)


def test_available_solvers_shape() -> None:
    """``available_solvers()`` returns capability dicts with the documented keys."""
    infos = opop.available_solvers()
    assert isinstance(infos, list)
    assert infos, "expected at least one known solver record"
    for info in infos:
        assert {"name", "version", "available", "detail"} <= set(info)


def test_qubo_round_trip_via_public_api() -> None:
    """``max_cut_qubo`` -> ``qubo_to_ir`` -> ``ir_to_qubo`` preserves the QUBO."""
    qubo = opop.max_cut_qubo(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    ir = opop.qubo_to_ir(qubo)
    recovered = opop.ir_to_qubo(ir)
    assert recovered.linear == pytest.approx(qubo.linear)
    assert recovered.quadratic == pytest.approx(qubo.quadratic)
    assert recovered.offset == pytest.approx(qubo.offset)


def test_milp_ir_constructs_via_public_api() -> None:
    """The IR records are exposed and build a valid MILP."""
    milp = opop.MILP(
        name="two_var_knapsack",
        variables=(
            opop.Variable("x", opop.VarType.BINARY, 0.0, 1.0),
            opop.Variable("y", opop.VarType.BINARY, 0.0, 1.0),
        ),
        constraints=(
            opop.LinearConstraint("cap", {"x": 1.0, "y": 1.0}, opop.ConstraintSense.LE, 1.0),
        ),
        objective=opop.Objective({"x": 1.0, "y": 1.0}, opop.ObjSense.MAXIMIZE, 0.0),
    )
    assert len(milp.variables) == 2
    assert len(milp.constraints) == 1


def test_no_gurobi_on_public_path() -> None:
    """Loading the entire public API must never import a commercial solver."""
    for name in opop.__all__:
        getattr(opop, name)
    assert "gurobipy" not in sys.modules
    assert "gurobi" not in sys.modules
