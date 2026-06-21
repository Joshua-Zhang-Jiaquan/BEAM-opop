"""QUBO problem-class adapter: exact MILP linearization + kernel routing (task 30).

:class:`QuboAdapter` is the :class:`~opop.model.adapter.ProblemClassAdapter` for
binary quadratic programs — a quadratic objective over all-BINARY variables with
no quadratic constraints (pure QUBO, plus the linearly-constrained binary
quadratic generalisation). The general MIQP / MIQCP case (continuous / integer
quadratics, quadratic constraints) is claimed by
:class:`opop.solver.miqp.MiqpAdapter`; the two ``can_handle`` predicates are
mutually exclusive.

Linearization (Fortet)
----------------------
The exact transform :func:`opop.model.quadratic.linearize_quadratic` is the
shared, problem-class-agnostic linearizer (it lives in the model layer so both
this adapter and :class:`opop.solver.miqp.MiqpAdapter` reuse it without an import
cycle); it is re-exported here as :func:`linearize_quadratic` so callers treat
this module as the QUBO linearization entry point. Every product
``c * x_i * x_j`` of two binary variables is replaced by ``c * y_ij`` where
``y_ij`` is a fresh binary "edge variable" pinned to the product by the standard
Fortet constraints::

    y_ij <= x_i ,   y_ij <= x_j ,   y_ij >= x_i + x_j - 1

At any integer ``x`` these force ``y_ij == x_i AND x_j`` exactly, so the MILP
optimum equals the quadratic optimum. A square term ``c * x_i^2`` folds into the
linear coefficient of ``x_i`` (``x_i^2 == x_i`` for binary ``x``).

For a Max-Cut QUBO ``minimise sum_e w_e (2 x_u x_v - x_u - x_v)`` this yields one
edge variable ``y_uv`` per edge; ``x_u + x_v - 2 y_uv`` is exactly the cut
indicator ``z_e = x_u XOR x_v``, so the linearized objective is the textbook
edge-variable Max-Cut formulation ``maximise sum_e w_e z_e`` (the optimum matches
the direct QUBO optimum — locked by the test).

Routing
-------
The linearized model is all-binary, so it routes to ANY linear kernel.
:func:`route_qubo` prefers the integer-exact CP-SAT, then SCIP, then HiGHS;
:func:`solve_qubo` linearizes and solves in one call. :meth:`QuboAdapter.native_solve`
offers the no-linearization route via SCIP's quadratic API (a QUBO is a binary
MIQP) for callers that explicitly want it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from opop.model.adapter import AdapterCapabilities, register_adapter
from opop.model.ir import MILP, UnsupportedModelError, VarType
from opop.model.quadratic import linearize_quadratic
from opop.model.state import Phi, SolveTrace

if TYPE_CHECKING:
    from opop.solver.kernel import SolverKernel

__all__ = [
    "QuboAdapter",
    "linearize_quadratic",
    "route_qubo",
    "solve_qubo",
]

#: Kernel preference order for a linearized (all-binary) QUBO MILP. CP-SAT is an
#: exact integer solver (ideal for 0-1 models); SCIP / HiGHS are general fallbacks.
_QUBO_KERNEL_ORDER: tuple[str, ...] = ("CP-SAT", "SCIP", "HiGHS")


def _kernel_for(name: str) -> SolverKernel:
    """Instantiate the kernel for a canonical solver ``name`` (lazy import)."""
    if name == "CP-SAT":
        from opop.solver.cpsat import CpsatKernel

        return CpsatKernel()
    if name == "SCIP":
        from opop.solver.scip import ScipKernel

        return ScipKernel()
    if name == "HiGHS":
        from opop.solver.highs import HighsKernel

        return HighsKernel()
    raise ValueError(f"unknown kernel name {name!r}")


def route_qubo(ir: MILP, *, prefer: str | None = None) -> SolverKernel:
    """Return a live kernel for a (linearized) QUBO, preferring CP-SAT then SCIP.

    CP-SAT is integer-only, so it is skipped when ``ir`` declares any CONTINUOUS
    variable (a pure QUBO is all-binary, so CP-SAT applies). ``prefer`` (a solver
    name) is tried first when available. Raises
    :class:`~opop.model.ir.UnsupportedModelError` if no capable solver is
    installed.
    """
    from opop.solver.availability import is_solver_available

    has_continuous = any(v.vtype is VarType.CONTINUOUS for v in ir.variables)
    order = ([prefer] if prefer else []) + [
        name
        for name in _QUBO_KERNEL_ORDER
        if not (name == "CP-SAT" and has_continuous)
    ]
    for name in order:
        if name and is_solver_available(name):
            return _kernel_for(name)
    raise UnsupportedModelError(
        "no capable solver available for QUBO routing "
        + f"(need one of {', '.join(_QUBO_KERNEL_ORDER)})"
    )


def solve_qubo(
    ir: MILP,
    *,
    kernel: SolverKernel | None = None,
    phi: Phi | None = None,
    time_limit: float = 60.0,
    memory_limit_mb: int = 4096,
    seed: int = 0,
) -> SolveTrace:
    """Linearize ``ir`` and solve the resulting MILP on a routed (or given) kernel.

    When ``kernel`` is ``None`` one is chosen by :func:`route_qubo`. The reported
    objective is in the QUBO's own sense (the linearization preserves it).
    """
    milp = linearize_quadratic(ir)
    solver = kernel if kernel is not None else route_qubo(ir)
    return solver.solve(
        milp,
        phi if phi is not None else Phi(),
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        seed=seed,
    )


@final
class QuboAdapter:
    """Adapter for binary quadratic programs (QUBO): exact linearization → MILP."""

    @property
    def name(self) -> str:
        """Registry key for this adapter."""
        return "qubo"

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declared capabilities (exact linearization to an all-binary MILP)."""
        return AdapterCapabilities(
            name="qubo",
            problem_class="QUBO",
            handles_quadratic_objective=True,
            handles_quadratic_constraints=False,
            exact_linearization=True,
            native_kernels=("SCIP",),
            linear_kernels=("CP-SAT", "SCIP", "HiGHS"),
        )

    def can_handle(self, ir: MILP) -> bool:
        """Handle an all-binary quadratic OBJECTIVE with no quadratic constraints."""
        ext = ir.quadratic
        if ext is None or not ext.has_objective_terms():
            return False
        if ext.has_constraint_terms():
            return False
        return all(v.vtype is VarType.BINARY for v in ir.variables)

    def to_milp(self, ir: MILP) -> MILP:
        """Exactly linearize the QUBO to a plain linear MILP (Fortet)."""
        if not self.can_handle(ir):
            raise UnsupportedModelError(
                "QuboAdapter.to_milp expects a binary quadratic objective with no "
                + "quadratic constraints (all-binary variables)"
            )
        return linearize_quadratic(ir)

    def native_solve(
        self,
        ir: MILP,
        kernel: SolverKernel,
        *,
        phi: Phi | None = None,
        time_limit: float = 60.0,
        memory_limit_mb: int = 4096,
        seed: int = 0,
    ) -> SolveTrace:
        """Solve the QUBO natively via SCIP's quadratic API (no linearization).

        For CP-SAT / HiGHS use :meth:`to_milp` + ``kernel.solve`` (or
        :func:`solve_qubo`) — those are MILP-only and cannot take a quadratic
        model directly.
        """
        if getattr(kernel, "solver_name", "") != "SCIP":
            got = getattr(kernel, "solver_name", type(kernel).__name__)
            raise UnsupportedModelError(
                "QuboAdapter.native_solve requires a SCIP kernel (solver_name='SCIP'); "
                + f"got {got!r}. For a MILP-only kernel call to_milp() then "
                + "kernel.solve(), or use solve_qubo()."
            )
        from opop.solver.miqp import solve_scip_quadratic

        return solve_scip_quadratic(
            ir,
            phi=phi,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            seed=seed,
        )


register_adapter(QuboAdapter())
