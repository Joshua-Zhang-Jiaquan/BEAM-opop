"""Expansion example: MIQP / QUBO through the problem-class adapter layer.

OPOP's core loop is problem-class *agnostic*; non-linear classes are handled by
declared :class:`opop.ProblemClassAdapter` plugins discovered via capability
dispatch (``opop.find_adapter``), never by a hard-coded ``if problem_class ==``
check. This script demonstrates both adapter routes on the open SCIP solver:

  1. QUBO (Max-Cut): the QUBO adapter exactly linearizes a binary quadratic
     program to a plain MILP (Fortet), which the SCIP kernel solves. The optimum
     is checked against brute force.
  2. MIQP (integer quadratic): the MIQP adapter solves the quadratic model
     natively on SCIP, with no linearization.

Usage:
    python examples/expansion_miqp.py
"""

from __future__ import annotations

import itertools

import opop
from opop.model import QuadraticExtension, QuadraticTerm, qubo_energy
from opop.solver.miqp import MiqpAdapter  # importing this module self-registers MiqpAdapter
from opop.solver.qubo import QuboAdapter  # importing this module self-registers QuboAdapter

TIME_LIMIT = 10.0
MEMORY_LIMIT_MB = 2048


def _brute_force_min_energy(qubo: opop.QUBO) -> float:
    """Minimum QUBO energy by exhaustive enumeration (tiny instances only)."""
    names = qubo.variables()
    best = float("inf")
    for bits in itertools.product((0.0, 1.0), repeat=len(names)):
        assignment = dict(zip(names, bits, strict=True))
        best = min(best, qubo_energy(qubo, assignment))
    return best


def run_qubo_demo() -> None:
    """Build a Max-Cut QUBO, linearize via the adapter, solve on SCIP, verify."""
    print("== QUBO (Max-Cut): QuboAdapter exact linearization -> MILP -> SCIP ==")
    # A 5-node graph: the cycle 0-1-2-3-4-0 plus one chord (0, 2).
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    qubo = opop.max_cut_qubo(5, edges)
    ir = opop.qubo_to_ir(qubo)

    adapter = opop.find_adapter(ir)  # capability-driven dispatch
    if not isinstance(adapter, QuboAdapter):
        raise RuntimeError("expected the QUBO adapter to claim a binary quadratic IR")
    print(f"  adapter={adapter.name!r}  class={adapter.capabilities.problem_class!r}")

    milp = adapter.to_milp(ir)  # exact Fortet linearization (all-binary MILP)
    print(
        f"  linearized: {len(ir.variables)} vars"
        + f" -> {len(milp.variables)} vars / {len(milp.constraints)} constraints"
    )

    trace = opop.ScipKernel().solve(
        milp, opop.Phi(), time_limit=TIME_LIMIT, memory_limit_mb=MEMORY_LIMIT_MB, seed=0
    )
    solved_min = trace.primal_bound_series[-1]
    brute_min = _brute_force_min_energy(qubo)
    print(
        f"  status={trace.status} min_energy={solved_min:.1f}"
        + f" (brute force {brute_min:.1f}) -> max-cut weight {-solved_min:.1f}"
    )
    if abs(solved_min - brute_min) > 1e-6:
        raise RuntimeError("linearization must preserve the QUBO optimum")
    print("  OK: linearized MILP optimum == QUBO optimum\n")


def run_miqp_demo() -> None:
    """Build a tiny integer MIQP and solve it natively on SCIP via the adapter."""
    print("== MIQP (integer quadratic): MiqpAdapter native SCIP solve ==")
    # minimise (x - 2)^2 = x^2 - 4x + 4 over integer x in [0, 5]; optimum 0 at x=2.
    x = opop.Variable("x", opop.VarType.INTEGER, 0.0, 5.0)
    objective = opop.Objective({"x": -4.0}, opop.ObjSense.MINIMIZE, 4.0)
    extension = QuadraticExtension(objective_terms=(QuadraticTerm("x", "x", 1.0),))
    miqp = opop.MILP(
        name="miqp_demo",
        variables=(x,),
        constraints=(),
        objective=objective,
        quadratic=extension,
    )

    adapter = opop.find_adapter(miqp)
    if not isinstance(adapter, MiqpAdapter):
        raise RuntimeError("expected the MIQP adapter to claim an integer quadratic IR")
    print(f"  adapter={adapter.name!r}  class={adapter.capabilities.problem_class!r}")

    trace = adapter.native_solve(
        miqp,
        opop.ScipKernel(),
        time_limit=TIME_LIMIT,
        memory_limit_mb=MEMORY_LIMIT_MB,
        seed=0,
    )
    solved = trace.primal_bound_series[-1]
    print(f"  status={trace.status} objective={solved:.3f} (expected 0.0 at x=2)")
    if abs(solved) > 1e-6:
        raise RuntimeError("native MIQP solve should reach the optimum")
    print("  OK: native MIQP optimum reached\n")


def main() -> int:
    """Run the QUBO + MIQP adapter demonstrations."""
    if not opop.is_solver_available("SCIP"):
        print("This example requires the SCIP backend (pip install pyscipopt).")
        return 1
    run_qubo_demo()
    run_miqp_demo()
    print("expansion example complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
