"""Problem-class adapter Protocol + capability registry (task 30).

The OPOP core loop (orchestrator / controller / evaluator) is *problem-class
agnostic*: it never branches on whether an instance is a plain MILP, a QUBO, a
MIQP, or a MIQCP. Instead, each non-linear problem class is handled by a
*plugin* implementing :class:`ProblemClassAdapter` that DECLARES its capabilities
via :class:`AdapterCapabilities`. Adapters register themselves in a process-wide
registry; generic code discovers the right one for an IR through
:func:`find_adapter` (capability-driven dispatch) — never through hard-coded
``if problem_class == ...`` checks.

Every adapter offers two routes:

* :meth:`ProblemClassAdapter.to_milp` — an EXACT linearization to a plain linear
  :class:`~opop.model.ir.MILP` (when one exists), so any linear kernel
  (CP-SAT / SCIP / HiGHS) can solve it. Raises
  :class:`~opop.model.ir.UnsupportedModelError` when no exact linearization is
  available (e.g. a product of two continuous variables).
* :meth:`ProblemClassAdapter.native_solve` — solve the quadratic model directly
  on a kernel that natively supports the structure (SCIP's nonlinear API for
  MIQP / MIQCP), WITHOUT linearization.

This module is pure model-layer: the only solver reference is the
:class:`~opop.solver.kernel.SolverKernel` *type*, imported under
``TYPE_CHECKING`` so importing :mod:`opop.model.adapter` never pulls in a solver
backend. Concrete adapters live in the ``opop.solver`` layer
(:class:`opop.solver.qubo.QuboAdapter`, :class:`opop.solver.miqp.MiqpAdapter`)
and self-register on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from opop.model.ir import MILP
from opop.model.state import Phi, SolveTrace

if TYPE_CHECKING:
    from opop.solver.kernel import SolverKernel

__all__ = [
    "AdapterCapabilities",
    "ProblemClassAdapter",
    "find_adapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
    "unregister_adapter",
]


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Declared capabilities of a :class:`ProblemClassAdapter`.

    Capability-driven dispatch (the core never special-cases a problem class)
    reads these flags to decide how to route an instance.

    Attributes:
        name: Unique adapter name (matches :attr:`ProblemClassAdapter.name`).
        problem_class: Human-readable class tag (e.g. ``"QUBO"``, ``"MIQP/MIQCP"``).
        handles_quadratic_objective: Adapter accepts quadratic objective terms.
        handles_quadratic_constraints: Adapter accepts quadratic constraint terms.
        exact_linearization: :meth:`ProblemClassAdapter.to_milp` is exact for the
            instances this adapter accepts (may still raise for sub-cases it
            documents — e.g. products of continuous variables).
        native_kernels: Solver names usable for
            :meth:`ProblemClassAdapter.native_solve` (no linearization).
        linear_kernels: Solver names usable AFTER
            :meth:`ProblemClassAdapter.to_milp` (the linearized MILP).
    """

    name: str
    problem_class: str
    handles_quadratic_objective: bool
    handles_quadratic_constraints: bool
    exact_linearization: bool
    native_kernels: tuple[str, ...]
    linear_kernels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping of the declared capabilities."""
        return {
            "name": self.name,
            "problem_class": self.problem_class,
            "handles_quadratic_objective": self.handles_quadratic_objective,
            "handles_quadratic_constraints": self.handles_quadratic_constraints,
            "exact_linearization": self.exact_linearization,
            "native_kernels": list(self.native_kernels),
            "linear_kernels": list(self.linear_kernels),
        }


@runtime_checkable
class ProblemClassAdapter(Protocol):
    """Plugin contract for a non-linear problem class.

    Implementations are stateless (a single instance is safe to register once and
    reuse). ``can_handle`` must be cheap and side-effect free; ``to_milp`` returns
    a NEW linear :class:`~opop.model.ir.MILP` (never mutates the input);
    ``native_solve`` builds and solves a backend model under the usual
    determinism/budget contract of :class:`~opop.solver.kernel.SolverKernel`.
    """

    @property
    def name(self) -> str:
        """Unique adapter name (the registry key)."""
        ...

    @property
    def capabilities(self) -> AdapterCapabilities:
        """The adapter's declared :class:`AdapterCapabilities`."""
        ...

    def can_handle(self, ir: MILP) -> bool:
        """Return ``True`` iff this adapter handles ``ir``'s problem class."""
        ...

    def to_milp(self, ir: MILP) -> MILP:
        """Return an exact linear-MILP reformulation of ``ir`` (or raise)."""
        ...

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
        """Solve ``ir`` natively on ``kernel`` (no linearization) and return a trace."""
        ...


# ---------------------------------------------------------------------------
# Process-wide adapter registry (capability-driven dispatch)
# ---------------------------------------------------------------------------
_REGISTRY: list[ProblemClassAdapter] = []


def register_adapter(adapter: ProblemClassAdapter) -> None:
    """Register ``adapter`` (idempotent: replaces any same-named registration).

    Idempotency keeps module re-import (e.g. across test runs) from accumulating
    duplicate adapters in the registry.
    """
    for index, existing in enumerate(_REGISTRY):
        if existing.name == adapter.name:
            _REGISTRY[index] = adapter
            return
    _REGISTRY.append(adapter)


def unregister_adapter(name: str) -> bool:
    """Remove the adapter named ``name``; return ``True`` iff one was removed."""
    for index, existing in enumerate(_REGISTRY):
        if existing.name == name:
            del _REGISTRY[index]
            return True
    return False


def registered_adapters() -> tuple[ProblemClassAdapter, ...]:
    """Return all currently registered adapters in registration order."""
    return tuple(_REGISTRY)


def get_adapter(name: str) -> ProblemClassAdapter | None:
    """Return the adapter named ``name``, or ``None`` if not registered."""
    for adapter in _REGISTRY:
        if adapter.name == name:
            return adapter
    return None


def find_adapter(ir: MILP) -> ProblemClassAdapter | None:
    """Return the first registered adapter whose ``can_handle(ir)`` is ``True``.

    Returns ``None`` for a plain linear MILP (no adapter claims it), so generic
    code can dispatch uniformly: ``adapter = find_adapter(ir)`` then route to
    ``adapter`` when non-``None`` else solve ``ir`` directly as a linear MILP.
    """
    for adapter in _REGISTRY:
        if adapter.can_handle(ir):
            return adapter
    return None
