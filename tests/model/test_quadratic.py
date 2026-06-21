"""Tests for the quadratic IR layer, QUBO/Ising, and the MIQP/MIQCP/QUBO adapters.

Pure tests (IR validation, QUBO<->Ising, registry, agnostic-core scan) always
run. Solver-backed tests (Max-Cut linearization, MIQCP/MIQP via SCIP) skip
cleanly when the required backend is unavailable.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from pathlib import Path
from typing import final

import pytest

import opop
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
    milps_equivalent,
)
from opop.model.quadratic import (
    QUBO,
    Ising,
    bits_from_spins,
    ir_to_qubo,
    ising_energy,
    ising_to_qubo,
    linearize_quadratic,
    max_cut_qubo,
    qubo_energy,
    qubo_to_ir,
    qubo_to_ising,
    spins_from_bits,
)
from opop.model.state import Phi, SolveTrace
from opop.solver.miqp import MiqpAdapter, solve_scip_quadratic
from opop.solver.qubo import QuboAdapter, route_qubo, solve_qubo

# A deterministic 6-node Max-Cut: the triangular prism (two triangles joined by a
# perfect matching). Each triangle is an odd cycle (>= 1 uncut edge), so the
# max-cut is 2 + 2 + 3 = 7 of its 9 edges (hand-verified, non-trivial).
PRISM_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 0),  # triangle A
    (3, 4), (4, 5), (5, 3),  # triangle B
    (0, 3), (1, 4), (2, 5),  # matching
]
PRISM_MAXCUT = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _brute_force_qubo_min(qubo: QUBO) -> float:
    """Exhaustively minimise the QUBO energy over all binary assignments."""
    names = qubo.variables()
    best = float("inf")
    for bits in itertools.product([0.0, 1.0], repeat=len(names)):
        energy = qubo_energy(qubo, dict(zip(names, bits, strict=True)))
        best = min(best, energy)
    return best


def _small_qubo() -> QUBO:
    """A 3-variable QUBO with a non-zero offset and mixed-sign coefficients."""
    return QUBO(
        linear={"a": 1.0, "b": -2.0, "c": 0.5},
        quadratic={("a", "b"): 3.0, ("b", "c"): -1.5, ("a", "c"): 2.0},
        offset=0.7,
    )


# ---------------------------------------------------------------------------
# QuadraticTerm / QuadraticExtension records
# ---------------------------------------------------------------------------
def test_quadratic_term_key_and_square() -> None:
    assert QuadraticTerm("y", "x", 2.0).key() == ("x", "y")
    assert QuadraticTerm("x", "x", 1.0).is_square
    assert not QuadraticTerm("x", "y", 1.0).is_square


def test_quadratic_extension_introspection() -> None:
    ext = QuadraticExtension(
        objective_terms=(QuadraticTerm("x", "y", 1.0),),
        constraint_terms={"c": (QuadraticTerm("y", "z", 2.0),)},
    )
    assert not ext.is_empty
    assert ext.has_objective_terms()
    assert ext.has_constraint_terms()
    assert ext.referenced_variables() == frozenset({"x", "y", "z"})
    assert ext.constraint_names() == frozenset({"c"})
    assert QuadraticExtension().is_empty


# ---------------------------------------------------------------------------
# MILP integration (optional field + referential-integrity validation)
# ---------------------------------------------------------------------------
def test_milp_accepts_quadratic_extension() -> None:
    ir = MILP(
        name="q",
        variables=(Variable("x", VarType.BINARY), Variable("y", VarType.BINARY)),
        objective=Objective({"x": 1.0}, ObjSense.MAXIMIZE),
        quadratic=QuadraticExtension(objective_terms=(QuadraticTerm("x", "y", -2.0),)),
    )
    assert ir.quadratic is not None
    assert ir.quadratic.objective_terms[0].coeff == pytest.approx(-2.0)


def test_milp_rejects_quadratic_unknown_variable() -> None:
    with pytest.raises(ValueError, match="quadratic extension references unknown variables"):
        MILP(
            variables=(Variable("x", VarType.BINARY),),
            quadratic=QuadraticExtension(objective_terms=(QuadraticTerm("x", "ghost", 1.0),)),
        )


def test_milp_rejects_quadratic_unknown_constraint() -> None:
    with pytest.raises(ValueError, match="quadratic extension references unknown constraints"):
        MILP(
            variables=(Variable("x", VarType.BINARY),),
            constraints=(LinearConstraint("c", {"x": 1.0}, ConstraintSense.LE, 1.0),),
            quadratic=QuadraticExtension(constraint_terms={"ghost": (QuadraticTerm("x", "x", 1.0),)}),
        )


def test_linear_milp_defaults_to_no_quadratic() -> None:
    ir = MILP(
        name="lin",
        variables=(Variable("x", VarType.BINARY),),
        objective=Objective({"x": 1.0}, ObjSense.MAXIMIZE),
    )
    assert ir.quadratic is None
    assert milps_equivalent(ir, ir)


# ---------------------------------------------------------------------------
# milp_diffs / milps_equivalent over the quadratic extension
# ---------------------------------------------------------------------------
def test_milps_equivalent_treats_none_and_empty_extension_alike() -> None:
    base = (Variable("x", VarType.BINARY),)
    ir_none = MILP(variables=base, objective=Objective({"x": 1.0}))
    ir_empty = MILP(variables=base, objective=Objective({"x": 1.0}), quadratic=QuadraticExtension())
    assert milps_equivalent(ir_none, ir_empty)


def test_milp_diffs_detects_quadratic_difference() -> None:
    variables = (Variable("x", VarType.BINARY), Variable("y", VarType.BINARY))
    obj = Objective({"x": 1.0}, ObjSense.MAXIMIZE)
    ir_a = MILP(
        variables=variables,
        objective=obj,
        quadratic=QuadraticExtension(objective_terms=(QuadraticTerm("x", "y", 3.0),)),
    )
    ir_b = MILP(
        variables=variables,
        objective=obj,
        quadratic=QuadraticExtension(objective_terms=(QuadraticTerm("x", "y", 5.0),)),
    )
    assert not milps_equivalent(ir_a, ir_b)


# ---------------------------------------------------------------------------
# QUBO <-> Ising conversions (x_i = (1 - s_i) / 2)
# ---------------------------------------------------------------------------
def test_bits_spins_are_inverse_under_one_minus_s_over_two() -> None:
    bits = {"a": 0.0, "b": 1.0}
    spins = spins_from_bits(bits)
    assert spins == {"a": 1.0, "b": -1.0}  # x=0 -> s=+1, x=1 -> s=-1
    assert bits_from_spins(spins) == bits


def test_qubo_to_ising_preserves_energy_for_all_assignments() -> None:
    qubo = _small_qubo()
    ising = qubo_to_ising(qubo)
    names = qubo.variables()
    for bits in itertools.product([0.0, 1.0], repeat=len(names)):
        x = dict(zip(names, bits, strict=True))
        s = spins_from_bits(x)
        assert qubo_energy(qubo, x) == pytest.approx(ising_energy(ising, s), abs=1e-12)


def test_ising_to_qubo_round_trip_recovers_coefficients() -> None:
    qubo = _small_qubo()
    back = ising_to_qubo(qubo_to_ising(qubo))
    assert back.linear == pytest.approx(qubo.linear)
    assert back.quadratic == pytest.approx(qubo.quadratic)
    assert back.offset == pytest.approx(qubo.offset)


def test_ising_to_qubo_energy_preserved() -> None:
    ising = Ising(h={"a": 0.5, "b": -1.0}, J={("a", "b"): 0.25}, offset=0.3)
    qubo = ising_to_qubo(ising)
    for spins in itertools.product([-1.0, 1.0], repeat=2):
        s: dict[str, float] = dict(zip(("a", "b"), spins, strict=True))
        x = bits_from_spins(s)
        assert ising_energy(ising, s) == pytest.approx(qubo_energy(qubo, x), abs=1e-12)


# ---------------------------------------------------------------------------
# QUBO <-> IR bridges + Max-Cut builder
# ---------------------------------------------------------------------------
def test_qubo_to_ir_and_back() -> None:
    qubo = _small_qubo()
    ir = qubo_to_ir(qubo)
    assert all(v.vtype is VarType.BINARY for v in ir.variables)
    assert ir.quadratic is not None
    recovered = ir_to_qubo(ir)
    assert recovered.quadratic == pytest.approx(qubo.quadratic)
    assert recovered.offset == pytest.approx(qubo.offset)


def test_max_cut_qubo_structure_and_brute_force_optimum() -> None:
    qubo = max_cut_qubo(6, PRISM_EDGES)
    assert qubo.variables() == tuple(f"x{i}" for i in range(6))
    # Each node has degree 3 in the prism -> linear coeff -3, each edge -> +2.
    assert all(coeff == pytest.approx(-3.0) for coeff in qubo.linear.values())
    assert all(coeff == pytest.approx(2.0) for coeff in qubo.quadratic.values())
    assert -_brute_force_qubo_min(qubo) == pytest.approx(PRISM_MAXCUT)


# ---------------------------------------------------------------------------
# Adapter Protocol + capability registry (capability-driven dispatch)
# ---------------------------------------------------------------------------
def test_adapters_register_and_satisfy_protocol() -> None:
    names = {a.name for a in registered_adapters()}
    assert {"qubo", "miqp"} <= names
    assert isinstance(QuboAdapter(), ProblemClassAdapter)
    assert isinstance(MiqpAdapter(), ProblemClassAdapter)
    assert get_adapter("qubo") is not None
    assert get_adapter("nope") is None


def test_adapter_capabilities_are_declared() -> None:
    qcaps = QuboAdapter().capabilities
    assert isinstance(qcaps, AdapterCapabilities)
    assert qcaps.exact_linearization
    assert "CP-SAT" in qcaps.linear_kernels
    mcaps = MiqpAdapter().capabilities
    assert mcaps.handles_quadratic_constraints
    assert mcaps.native_kernels == ("SCIP",)


def test_find_adapter_routes_by_capability() -> None:
    qubo_ir = qubo_to_ir(max_cut_qubo(6, PRISM_EDGES))
    qubo_adapter = find_adapter(qubo_ir)
    assert qubo_adapter is not None
    assert qubo_adapter.name == "qubo"

    miqcp_ir = _miqcp_fixture()
    miqcp_adapter = find_adapter(miqcp_ir)
    assert miqcp_adapter is not None
    assert miqcp_adapter.name == "miqp"

    linear_ir = MILP(variables=(Variable("x", VarType.BINARY),), objective=Objective({"x": 1.0}))
    assert find_adapter(linear_ir) is None


def test_qubo_and_miqp_can_handle_are_mutually_exclusive() -> None:
    qubo_ir = qubo_to_ir(max_cut_qubo(6, PRISM_EDGES))
    assert QuboAdapter().can_handle(qubo_ir)
    assert not MiqpAdapter().can_handle(qubo_ir)
    miqcp_ir = _miqcp_fixture()
    assert MiqpAdapter().can_handle(miqcp_ir)
    assert not QuboAdapter().can_handle(miqcp_ir)


# ---------------------------------------------------------------------------
# Linearization (pure): exactness preconditions
# ---------------------------------------------------------------------------
def test_linearize_quadratic_is_noop_on_linear_milp() -> None:
    ir = MILP(variables=(Variable("x", VarType.BINARY),), objective=Objective({"x": 1.0}))
    out = linearize_quadratic(ir)
    assert out.quadratic is None
    assert out.n_vars == 1


def test_qubo_to_milp_introduces_one_edge_var_per_product() -> None:
    ir = qubo_to_ir(max_cut_qubo(6, PRISM_EDGES))
    milp = QuboAdapter().to_milp(ir)
    assert milp.quadratic is None
    # 6 node vars + one product/edge var per quadratic pair (9 edges).
    assert milp.n_vars == 6 + len(PRISM_EDGES)
    assert all(v.vtype is VarType.BINARY for v in milp.variables)


def test_to_milp_raises_for_non_binary_quadratic() -> None:
    with pytest.raises(UnsupportedModelError, match="BINARY"):
        MiqpAdapter().to_milp(_miqcp_fixture())


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
        MiqpAdapter().native_solve(_miqcp_fixture(), _FakeKernel())


# ---------------------------------------------------------------------------
# MIQCP / MIQP fixtures
# ---------------------------------------------------------------------------
def _miqcp_fixture() -> MILP:
    """maximise x + y s.t. x^2 + y^2 <= 10, x, y integer in [0, 5]. Optimum = 4."""
    return MILP(
        name="miqcp",
        variables=(
            Variable("x", VarType.INTEGER, 0.0, 5.0),
            Variable("y", VarType.INTEGER, 0.0, 5.0),
        ),
        constraints=(LinearConstraint("ball", {}, ConstraintSense.LE, 10.0),),
        objective=Objective({"x": 1.0, "y": 1.0}, ObjSense.MAXIMIZE),
        quadratic=QuadraticExtension(
            constraint_terms={
                "ball": (QuadraticTerm("x", "x", 1.0), QuadraticTerm("y", "y", 1.0))
            }
        ),
    )


def _miqp_objective_fixture() -> MILP:
    """minimise x^2 - 3x + y^2 - 3y, x, y integer in [0, 5]. Optimum = -4."""
    return MILP(
        name="miqp_obj",
        variables=(
            Variable("x", VarType.INTEGER, 0.0, 5.0),
            Variable("y", VarType.INTEGER, 0.0, 5.0),
        ),
        objective=Objective({"x": -3.0, "y": -3.0}, ObjSense.MINIMIZE),
        quadratic=QuadraticExtension(
            objective_terms=(QuadraticTerm("x", "x", 1.0), QuadraticTerm("y", "y", 1.0))
        ),
    )


# ---------------------------------------------------------------------------
# Solver-backed: QUBO linearization preserves optimum on a 6-node Max-Cut
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_max_cut_linearization_preserves_optimum(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("CP-SAT")
    qubo = max_cut_qubo(6, PRISM_EDGES)
    ir = qubo_to_ir(qubo)
    expected_min = _brute_force_qubo_min(qubo)

    trace = solve_qubo(ir, time_limit=15.0, seed=0)
    assert trace.status == "OPTIMAL"
    assert trace.primal_bound_series[-1] == pytest.approx(expected_min, abs=1e-6)
    # The QUBO minimum equals -(max-cut weight); the prism max-cut is 7.
    assert -trace.primal_bound_series[-1] == pytest.approx(PRISM_MAXCUT, abs=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["CP-SAT", "SCIP"])
def test_max_cut_linearization_agrees_across_kernels(
    solver: str, solver_skip_if_missing: Callable[[str], None]
) -> None:
    solver_skip_if_missing(solver)
    qubo = max_cut_qubo(6, PRISM_EDGES)
    ir = qubo_to_ir(qubo)
    milp = QuboAdapter().to_milp(ir)
    kernel = route_qubo(ir, prefer=solver)
    trace = kernel.solve(milp, Phi(), time_limit=15.0, memory_limit_mb=2048, seed=0)
    assert trace.primal_bound_series[-1] == pytest.approx(_brute_force_qubo_min(qubo), abs=1e-6)


# ---------------------------------------------------------------------------
# Solver-backed: MIQCP / MIQP solve to their known optima via SCIP
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_miqcp_solves_to_known_optimum_via_scip(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    from opop.solver.scip import ScipKernel

    trace = MiqpAdapter().native_solve(_miqcp_fixture(), ScipKernel(), time_limit=15.0, seed=0)
    assert trace.solver == "SCIP"
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(4.0, abs=1e-6)


@pytest.mark.integration
def test_miqp_quadratic_objective_solves_to_known_optimum_via_scip(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("SCIP")
    trace = solve_scip_quadratic(_miqp_objective_fixture(), time_limit=15.0, seed=0)
    assert trace.status == "optimal"
    assert trace.primal_bound_series[-1] == pytest.approx(-4.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The core loop stays problem-class agnostic (no leakage into orchestrator/
# controller/evaluator)
# ---------------------------------------------------------------------------
_CORE_PACKAGES = ("orchestrator", "controller", "evaluator")
_FORBIDDEN_NAME = re.compile(
    r"\b(QuboAdapter|MiqpAdapter|ProblemClassAdapter|QuadraticExtension|QuadraticTerm"
    + r"|find_adapter|register_adapter|linearize_quadratic|to_milp|qubo_to_ir|ir_to_qubo"
    + r"|qubo|miqp|miqcp|ising)\b",
    re.IGNORECASE,
)
_FORBIDDEN_IMPORT = re.compile(
    r"opop\.(model\.(quadratic|adapter)|solver\.(qubo|miqp))\b"
)
_FORBIDDEN_BRANCH = re.compile(r"(task_family|problem_class)\s*==")


def test_core_loop_is_problem_class_agnostic() -> None:
    root = Path(opop.__file__).resolve().parent
    offenders: list[str] = []
    for package in _CORE_PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for pattern, label in (
                (_FORBIDDEN_NAME, "problem-class name"),
                (_FORBIDDEN_IMPORT, "adapter/quadratic import"),
                (_FORBIDDEN_BRANCH, "problem-class branch"),
            ):
                match = pattern.search(text)
                if match is not None:
                    offenders.append(f"{package}/{path.name}: {label} {match.group(0)!r}")
    assert offenders == [], f"problem-class logic leaked into the core loop: {offenders}"
