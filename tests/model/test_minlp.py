"""Tests for the structured MINLP adapter (separable / factorable subset, task 31).

Pure tests (registration, capability dispatch, scope rejection, OA structure, and
the task-24 decomposition link) always run. Solver-backed tests (outer-approximation
and native SCIP solves to known optima) skip cleanly when SCIP is unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import final

import pytest

from opop.model.adapter import (
    AdapterCapabilities,
    ProblemClassAdapter,
    find_adapter,
    get_adapter,
    registered_adapters,
)
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    QuadraticExtension,
    QuadraticTerm,
    UnsupportedModelError,
    Variable,
    VarType,
)
from opop.model.minlp import (
    NONLINEAR_TERMS_KEY,
    OBJECTIVE_TARGET,
    SUPPORTED_FUNCTIONS,
    BilinearTerm,
    NonlinearTerm,
    StructuredMinlpAdapter,
    solve_scip_minlp,
)
from opop.model.state import Phi, SolveTrace


# ---------------------------------------------------------------------------
# Fixtures: separable convex/concave MINLPs with hand-verified optima
# ---------------------------------------------------------------------------
def _convex_objective_minlp() -> MILP:
    """minimise x^2 + y^2 s.t. x + y >= 5, x, y integer in [0, 5]. Optimum = 13."""
    return MILP(
        name="sep_sq_obj",
        variables=(
            Variable("x", VarType.INTEGER, 0.0, 5.0),
            Variable("y", VarType.INTEGER, 0.0, 5.0),
        ),
        constraints=(LinearConstraint("c", {"x": 1.0, "y": 1.0}, ConstraintSense.GE, 5.0),),
        objective=Objective({}, ObjSense.MINIMIZE),
        metadata={
            NONLINEAR_TERMS_KEY: (
                NonlinearTerm("square", "x", 1.0, OBJECTIVE_TARGET),
                NonlinearTerm("square", "y", 1.0, OBJECTIVE_TARGET),
            )
        },
    )


def _convex_constraint_minlp() -> MILP:
    """maximise x + y s.t. x^2 + y^2 <= 13, x, y integer in [0, 5]. Optimum = 5."""
    return MILP(
        name="sep_sq_con",
        variables=(
            Variable("x", VarType.INTEGER, 0.0, 5.0),
            Variable("y", VarType.INTEGER, 0.0, 5.0),
        ),
        constraints=(LinearConstraint("ball", {}, ConstraintSense.LE, 13.0),),
        objective=Objective({"x": 1.0, "y": 1.0}, ObjSense.MAXIMIZE),
        metadata={
            NONLINEAR_TERMS_KEY: (
                NonlinearTerm("square", "x", 1.0, "ball"),
                NonlinearTerm("square", "y", 1.0, "ball"),
            )
        },
    )


def _concave_objective_minlp() -> MILP:
    """maximise log(x), x integer in [1, 4]. Optimum = log(4) ~ 1.3862943611."""
    return MILP(
        name="concave_log",
        variables=(Variable("x", VarType.INTEGER, 1.0, 4.0),),
        objective=Objective({}, ObjSense.MAXIMIZE),
        metadata={NONLINEAR_TERMS_KEY: (NonlinearTerm("log", "x", 1.0, OBJECTIVE_TARGET),)},
    )


# ---------------------------------------------------------------------------
# Registration + capability dispatch (pure)
# ---------------------------------------------------------------------------
def test_adapter_registers_and_satisfies_protocol() -> None:
    names = {a.name for a in registered_adapters()}
    assert "structured_minlp" in names
    assert isinstance(StructuredMinlpAdapter(), ProblemClassAdapter)
    assert get_adapter("structured_minlp") is not None
    assert get_adapter("nope") is None


def test_capabilities_are_declared() -> None:
    caps = StructuredMinlpAdapter().capabilities
    assert isinstance(caps, AdapterCapabilities)
    assert caps.name == "structured_minlp"
    assert caps.native_kernels == ("SCIP",)
    assert "SCIP" in caps.linear_kernels
    # Outer approximation is exact only at breakpoints, never a general exact form.
    assert caps.exact_linearization is False


def test_supported_functions_split_convex_concave() -> None:
    assert SUPPORTED_FUNCTIONS == {"square", "exp", "log", "sqrt"}


def test_can_handle_only_claims_declared_separable_minlp() -> None:
    adapter = StructuredMinlpAdapter()
    assert adapter.can_handle(_convex_objective_minlp())
    assert adapter.can_handle(_convex_constraint_minlp())
    assert adapter.can_handle(_concave_objective_minlp())

    # A plain linear MILP carries no nonlinear-term metadata -> not claimed.
    linear = MILP(
        variables=(Variable("x", VarType.BINARY),), objective=Objective({"x": 1.0})
    )
    assert not adapter.can_handle(linear)

    # A quadratic-extension MILP (task 30) is the MIQP adapter's job, not this one.
    quad = MILP(
        variables=(Variable("x", VarType.BINARY), Variable("y", VarType.BINARY)),
        objective=Objective({"x": 1.0}, ObjSense.MAXIMIZE),
        quadratic=QuadraticExtension(objective_terms=(QuadraticTerm("x", "y", -2.0),)),
    )
    assert not adapter.can_handle(quad)


def test_find_adapter_routes_separable_minlp() -> None:
    adapter = find_adapter(_convex_objective_minlp())
    assert adapter is not None
    assert adapter.name == "structured_minlp"

    linear = MILP(
        variables=(Variable("x", VarType.BINARY),), objective=Objective({"x": 1.0})
    )
    found = find_adapter(linear)
    assert found is None or found.name != "structured_minlp"


# ---------------------------------------------------------------------------
# to_milp: outer-approximation structure (pure)
# ---------------------------------------------------------------------------
def test_to_milp_produces_pure_linear_outer_approximation() -> None:
    ir = _convex_objective_minlp()
    milp = StructuredMinlpAdapter().to_milp(ir)

    # Linear only: no quadratic layer, nonlinear metadata stripped, OA tag set.
    assert milp.quadratic is None
    assert NONLINEAR_TERMS_KEY not in milp.metadata
    assert milp.metadata["linearization"] == "outer_approximation"

    # Two fresh continuous auxiliaries (one per square term) on top of x, y.
    assert milp.n_vars == ir.n_vars + 2
    aux = [v for v in milp.variables if v.name not in {"x", "y"}]
    assert len(aux) == 2
    assert all(v.vtype is VarType.CONTINUOUS for v in aux)

    # 1 coupling row + 6 tangent cuts per variable (integers 0..5).
    assert milp.n_constraints == 1 + 2 * 6


def test_to_milp_is_noop_on_a_term_free_model() -> None:
    ir = MILP(
        name="lin",
        variables=(Variable("x", VarType.BINARY),),
        objective=Objective({"x": 1.0}, ObjSense.MAXIMIZE),
    )
    out = StructuredMinlpAdapter().to_milp(ir)
    assert out.quadratic is None
    assert out.n_vars == 1
    assert out.n_constraints == 0


# ---------------------------------------------------------------------------
# Decomposition link to task 24 (pure: detect_decomposition needs no solver)
# ---------------------------------------------------------------------------
def test_decomposition_report_recovers_separable_blocks() -> None:
    report = StructuredMinlpAdapter().decomposition_report(_convex_objective_minlp())
    # Each (variable, auxiliary) pair is an independent block coupled only by the
    # shared linear constraint -> Dantzig-Wolfe with two blocks.
    assert report.decomposability == "DW"
    assert report.n_blocks == 2
    assert "c" in report.linking_constraints


# ---------------------------------------------------------------------------
# Scope rejection: out-of-subset terms raise UnsupportedModelError (pure)
# ---------------------------------------------------------------------------
def test_rejects_bilinear_product_of_continuous_variables() -> None:
    ir = MILP(
        name="bad_bilinear",
        variables=(
            Variable("x", VarType.CONTINUOUS, 0.0, 5.0),
            Variable("y", VarType.CONTINUOUS, 0.0, 5.0),
        ),
        objective=Objective({}, ObjSense.MINIMIZE),
        metadata={NONLINEAR_TERMS_KEY: (BilinearTerm("x", "y", 1.0, OBJECTIVE_TARGET),)},
    )
    assert not StructuredMinlpAdapter().can_handle(ir)
    with pytest.raises(UnsupportedModelError, match=r"bilinear product 'x'\*'y'"):
        StructuredMinlpAdapter().to_milp(ir)


def test_rejects_unsupported_nonlinear_function() -> None:
    ir = MILP(
        name="bad_func",
        variables=(Variable("x", VarType.INTEGER, 0.0, 5.0),),
        objective=Objective({}, ObjSense.MINIMIZE),
        metadata={NONLINEAR_TERMS_KEY: (NonlinearTerm("sin", "x", 1.0, OBJECTIVE_TARGET),)},
    )
    assert not StructuredMinlpAdapter().can_handle(ir)
    with pytest.raises(UnsupportedModelError, match=r"unsupported nonlinear function 'sin'"):
        StructuredMinlpAdapter().to_milp(ir)


def test_rejects_convex_term_in_geq_constraint_as_nonconvex() -> None:
    ir = MILP(
        name="bad_curvature",
        variables=(Variable("x", VarType.INTEGER, 0.0, 5.0),),
        constraints=(LinearConstraint("c", {}, ConstraintSense.GE, 4.0),),
        objective=Objective({"x": 1.0}, ObjSense.MINIMIZE),
        metadata={NONLINEAR_TERMS_KEY: (NonlinearTerm("square", "x", 1.0, "c"),)},
    )
    assert not StructuredMinlpAdapter().can_handle(ir)
    with pytest.raises(UnsupportedModelError, match=r"nonconvex region"):
        StructuredMinlpAdapter().to_milp(ir)


def test_rejects_unbounded_variable_for_outer_approximation() -> None:
    ir = MILP(
        name="bad_bounds",
        variables=(Variable("x", VarType.INTEGER, 0.0, math.inf),),
        objective=Objective({}, ObjSense.MINIMIZE),
        metadata={NONLINEAR_TERMS_KEY: (NonlinearTerm("square", "x", 1.0, OBJECTIVE_TARGET),)},
    )
    assert not StructuredMinlpAdapter().can_handle(ir)
    with pytest.raises(UnsupportedModelError, match=r"finite bounds"):
        StructuredMinlpAdapter().to_milp(ir)


def test_rejects_log_on_nonpositive_domain() -> None:
    ir = MILP(
        name="bad_domain",
        variables=(Variable("x", VarType.INTEGER, 0.0, 4.0),),
        objective=Objective({}, ObjSense.MAXIMIZE),
        metadata={NONLINEAR_TERMS_KEY: (NonlinearTerm("log", "x", 1.0, OBJECTIVE_TARGET),)},
    )
    assert not StructuredMinlpAdapter().can_handle(ir)
    with pytest.raises(UnsupportedModelError, match=r"strictly positive lower bound"):
        StructuredMinlpAdapter().to_milp(ir)


# ---------------------------------------------------------------------------
# native_solve rejects a non-SCIP kernel fail-closed (pure)
# ---------------------------------------------------------------------------
@final
class _FakeKernel:
    solver_name: str = "FAKE"

    def solve(
        self, ir: MILP, phi: Phi, *, time_limit: float, memory_limit_mb: int, seed: int
    ) -> SolveTrace:
        del ir, phi, time_limit, memory_limit_mb, seed
        raise AssertionError("solve must not be called when the kernel is rejected")


def test_native_solve_rejects_non_scip_kernel() -> None:
    with pytest.raises(UnsupportedModelError, match="SCIP"):
        StructuredMinlpAdapter().native_solve(_convex_objective_minlp(), _FakeKernel())


# ---------------------------------------------------------------------------
# Solver-backed: outer-approximation solves to the known optimum via SCIP
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_separable_objective_outer_approximation_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    milp = StructuredMinlpAdapter().to_milp(_convex_objective_minlp())
    trace = ScipKernel().solve(milp, Phi(), time_limit=20.0, memory_limit_mb=2048, seed=0)
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(13.0, abs=1e-6)


@pytest.mark.integration
def test_separable_constraint_outer_approximation_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    milp = StructuredMinlpAdapter().to_milp(_convex_constraint_minlp())
    trace = ScipKernel().solve(milp, Phi(), time_limit=20.0, memory_limit_mb=2048, seed=0)
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(5.0, abs=1e-6)


@pytest.mark.integration
def test_concave_objective_outer_approximation_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    milp = StructuredMinlpAdapter().to_milp(_concave_objective_minlp())
    trace = ScipKernel().solve(milp, Phi(), time_limit=20.0, memory_limit_mb=2048, seed=0)
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(math.log(4.0), abs=1e-6)


# ---------------------------------------------------------------------------
# Solver-backed: native SCIP solve reaches the known optimum
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_native_solve_reaches_known_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    trace = StructuredMinlpAdapter().native_solve(
        _convex_objective_minlp(), ScipKernel(), time_limit=20.0, seed=0
    )
    assert trace.solver == "SCIP"
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(13.0, abs=1e-6)


@pytest.mark.integration
def test_native_solve_handles_concave_log(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    trace = solve_scip_minlp(_concave_objective_minlp(), time_limit=20.0, seed=0)
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(math.log(4.0), abs=1e-6)


@pytest.mark.integration
def test_outer_approximation_and_native_agree(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    adapter = StructuredMinlpAdapter()
    ir = _convex_constraint_minlp()
    oa = ScipKernel().solve(adapter.to_milp(ir), Phi(), time_limit=20.0, memory_limit_mb=2048, seed=0)
    native = adapter.native_solve(ir, ScipKernel(), time_limit=20.0, seed=0)
    assert oa.primal_bound_series[-1] == pytest.approx(native.primal_bound_series[-1], abs=1e-6)
