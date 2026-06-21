"""Task-24 acceptance: GCG branch-price-and-cut kernel (Dantzig--Wolfe).

GCG is an OPTIONAL backend. This suite verifies both halves of the contract:

* When ``pygcgopt`` is NOT importable, :class:`~opop.solver.gcg.GcgKernel` raises
  a typed :class:`~opop.solver.gcg.SolverUnavailableError` on CONSTRUCTION (the
  clean import-time failure path). This is the path exercised in an environment
  without GCG and runs unconditionally.
* When GCG IS available, the kernel satisfies the
  :class:`~opop.solver.kernel.SolverKernel` Protocol and solves a decomposable
  fixture (three blocks coupled by one budget row; the detector reports DW with
  3 blocks) to its known optimum of ``9``. These tests skip cleanly via
  ``solver_skip_if_missing("gcg")`` + ``pytest.importorskip("pygcgopt")``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

import pytest

from opop.analyzer.decompose import DECOMP_DW, decomposition_delta, detect_decomposition
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
    apply_delta,
)
from opop.model.state import Phi, SolveTrace
from opop.solver.gcg import GcgKernel, SolverUnavailableError
from opop.solver.kernel import SolverKernel

_MEMORY_MB = 4096
_KNOWN_OPTIMUM = 9.0

#: ``True`` iff the GCG binding is importable in this environment.
_GCG_PRESENT = importlib.util.find_spec("pygcgopt") is not None


def _bin(name: str) -> Variable:
    return Variable(name, VarType.BINARY, 0.0, 1.0)


def _three_block_dw() -> MILP:
    """Three 2-variable blocks (a_i+b_i<=1) coupled by one budget row (sum<=2).

    Per-block best pick is ``a_i`` (values 3, 4, 5); the budget allows 2 picks,
    so the integer optimum is ``a2 + a1 = 5 + 4 = 9``. The detector classifies
    this as DW with 3 blocks coupled by the ``budget`` row.
    """
    variables = tuple(_bin(n) for i in range(3) for n in (f"a{i}", f"b{i}"))
    blocks = tuple(
        LinearConstraint(f"blk{i}", {f"a{i}": 1.0, f"b{i}": 1.0}, ConstraintSense.LE, 1.0)
        for i in range(3)
    )
    budget = LinearConstraint(
        "budget",
        {f"a{i}": 1.0 for i in range(3)} | {f"b{i}": 1.0 for i in range(3)},
        ConstraintSense.LE,
        2.0,
    )
    obj = Objective(
        {"a0": 3.0, "a1": 4.0, "a2": 5.0, "b0": 1.0, "b1": 1.0, "b2": 1.0},
        ObjSense.MAXIMIZE,
    )
    return MILP("three_block_dw", variables, (*blocks, budget), obj)


# ===========================================================================
# Import-time failure path (runs without GCG)
# ===========================================================================
def test_solver_unavailable_error_is_typed() -> None:
    assert issubclass(SolverUnavailableError, RuntimeError)


@pytest.mark.smoke
def test_construction_raises_when_pygcgopt_missing() -> None:
    if _GCG_PRESENT:
        pytest.skip("pygcgopt is installed; the unavailable-path assertion does not apply")
    with pytest.raises(SolverUnavailableError, match="pygcgopt"):
        GcgKernel()


def test_fixture_is_detected_as_dw_with_three_blocks() -> None:
    # The decomposition the GCG kernel exploits is verifiable WITHOUT GCG.
    report = detect_decomposition(_three_block_dw())
    assert report.decomposability == DECOMP_DW
    assert report.n_blocks == 3
    assert report.linking_constraints == ("budget",)


# ===========================================================================
# GCG-present path (skips cleanly when the binding is absent)
# ===========================================================================
@pytest.mark.smoke
def test_gcg_satisfies_protocol(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("gcg")
    pytest.importorskip("pygcgopt")
    assert isinstance(GcgKernel(), SolverKernel)


@pytest.mark.smoke
def test_gcg_solves_decomposable_fixture_to_known_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("gcg")
    pytest.importorskip("pygcgopt")
    kernel = GcgKernel()
    trace = kernel.solve(
        _three_block_dw(), Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    assert isinstance(trace, SolveTrace)
    assert trace.solver == "GCG"
    assert trace.instance_id == "three_block_dw"
    assert trace.status == "optimal"
    assert trace.censored is False
    assert trace.primal_bound_series[-1] == pytest.approx(_KNOWN_OPTIMUM, abs=1e-6)
    assert trace.dual_bound_series[-1] == pytest.approx(_KNOWN_OPTIMUM, abs=1e-6)
    assert trace.nodes >= 0


def test_gcg_solves_with_certified_decomposition_metadata(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("gcg")
    pytest.importorskip("pygcgopt")
    ir = _three_block_dw()
    # Attach the certified (class-C) decomposition annotation, then solve: the
    # math model is unchanged, so GCG must still reach the same optimum.
    delta = decomposition_delta(detect_decomposition(ir))
    assert delta is not None
    decomposed_ir = apply_delta(ir, delta)
    trace = GcgKernel(apply_decomposition=True).solve(
        decomposed_ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0
    )
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(_KNOWN_OPTIMUM, abs=1e-6)


def test_gcg_deterministic_under_fixed_seed(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("gcg")
    pytest.importorskip("pygcgopt")
    kernel = GcgKernel()
    ir = _three_block_dw()
    a = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    b = kernel.solve(ir, Phi(), time_limit=30.0, memory_limit_mb=_MEMORY_MB, seed=0)
    assert a.status == b.status
    assert a.primal_bound_series[-1] == pytest.approx(b.primal_bound_series[-1], abs=1e-9)
