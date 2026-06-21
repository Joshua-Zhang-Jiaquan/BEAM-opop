"""Solver-backend kernel Protocol for OPOP.

A :class:`SolverKernel` compiles a symbolic MILP IR plus a :class:`~opop.model.state.Phi`
design vector into a concrete solver model, runs it under a fixed resource
budget (single-threaded, time/memory-limited, seeded), and returns a rich
:class:`~opop.model.state.SolveTrace` describing the primal/dual trajectory and
terminal statistics.

This Protocol is the contract reused by every backend kernel:

* :class:`opop.solver.scip.ScipKernel` (task 12, the Phase-1 reference).
* Future open backends (HiGHS / CP-SAT, tasks 22--23) implement the same
  signature so the controller/evaluator stay solver-agnostic.

Determinism contract (all kernels MUST honour):

* ``threads == 1`` — no LP/solver parallelism, so traces are reproducible.
* ``time_limit`` (seconds) and ``memory_limit_mb`` (MiB) are hard ceilings;
  hitting either terminates the run and yields ``SolveTrace.censored == True``.
* ``seed`` drives the backend's master randomisation so repeated calls with the
  same ``(ir, phi, limits, seed)`` produce identical traces.

Kernels MUST NOT silently swallow solver errors: an import/initialisation/solve
failure propagates (or is surfaced via a typed exception), never masked.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from opop.model.ir import MILP
from opop.model.state import Phi, SolveTrace

__all__ = ["SolverKernel"]


@runtime_checkable
class SolverKernel(Protocol):
    """Compile an IR + design vector, solve under a budget, return a trace.

    Implementations are expected to be stateless with respect to a single
    :meth:`solve` call (each call builds a fresh backend model) so that kernels
    are safe to reuse across instances and seeds.
    """

    def solve(
        self,
        ir: MILP,
        phi: Phi,
        *,
        time_limit: float,
        memory_limit_mb: int,
        seed: int,
    ) -> SolveTrace:
        """Solve ``ir`` configured by ``phi`` and return a :class:`SolveTrace`.

        Args:
            ir: The symbolic MILP to compile and solve.
            phi: Design vector; ``phi.p`` carries backend parameter overrides and
                the proposer-hook channel (whitelisted separators, decomposition
                toggles).
            time_limit: Wall-clock ceiling in seconds (hard limit).
            memory_limit_mb: Memory ceiling in MiB (hard limit).
            seed: Master randomisation seed for reproducibility.

        Returns:
            A :class:`SolveTrace` with the primal/dual bound series, terminal
            statistics, status string, and the ``censored`` flag (``True`` when
            the run was stopped by a resource limit without an optimality proof).
        """
        ...
