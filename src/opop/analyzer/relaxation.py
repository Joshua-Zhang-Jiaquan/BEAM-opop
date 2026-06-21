"""LP-relaxation statistics for the OPOP analyzer (SCIP root LP).

The LP relaxation of a MILP drops integrality: every BINARY / INTEGER variable
is treated as CONTINUOUS over its original bounds. Solving that LP to optimality
yields the *LP-relaxation objective* — a bound on the true integer optimum and
the cheapest structural signal the analyzer produces.

This module builds the relaxation by replacing every variable's ``vtype`` with
``CONTINUOUS`` in a NEW IR (the input IR is never mutated), compiling it with
:func:`opop.model.ir.to_pyscipopt`, and solving it as a pure LP with presolve and
separation disabled so the reported value is the clean root LP relaxation in the
original variable space. From the optimal LP solution it derives:

* ``lp_obj``           — the LP-relaxation objective.
* ``fractional_vars``  — integer-constrained variables whose LP value is not
  (within ``frac_tol``) integral; these are exactly the variables a solver
  would have to branch on.
* ``gap``              — the integrality-gap estimate
  ``(ip_bound - lp_obj) / |ip_bound|`` where ``ip_bound`` is a known integer
  objective (passed explicitly, or read from ``metadata['known_optimum']`` /
  ``metadata['ip_bound']``) or, when ``estimate_ip_bound`` is set, the primal
  bound from a node/time-limited SCIP integer solve.

All solves pin ``threads=1`` for determinism.
"""

from __future__ import annotations

import math
from dataclasses import replace

from opop.analyzer.report import RelaxationMetrics
from opop.model.ir import MILP, VarType, to_pyscipopt

__all__ = ["analyze_relaxation", "relaxed_ir"]

_INTEGER_VTYPES = frozenset({VarType.BINARY, VarType.INTEGER})


def relaxed_ir(ir: MILP) -> MILP:
    """Return a NEW IR equal to ``ir`` but with every variable CONTINUOUS.

    Pure: ``ir`` is not mutated. Bounds, constraints, and objective are
    preserved exactly, so the result is the LP relaxation of ``ir``.
    """
    relaxed_vars = tuple(
        v if v.vtype is VarType.CONTINUOUS else replace(v, vtype=VarType.CONTINUOUS)
        for v in ir.variables
    )
    return replace(ir, variables=relaxed_vars)


def analyze_relaxation(
    ir: MILP,
    *,
    ip_bound: float | None = None,
    estimate_ip_bound: bool = True,
    estimate_node_limit: int | None = 10_000,
    estimate_time_limit: float | None = None,
    frac_tol: float = 1e-6,
) -> RelaxationMetrics:
    """Solve the LP relaxation of ``ir`` and return :class:`RelaxationMetrics`.

    Args:
        ir: The MILP to analyze.
        ip_bound: A known integer objective bound for the gap. When ``None`` the
            IR metadata keys ``known_optimum`` / ``ip_bound`` are consulted, then
            (if ``estimate_ip_bound``) an estimate is computed by solving the
            integer model under the node/time limits below.
        estimate_ip_bound: Estimate the IP bound by a bounded integer solve when
            none is otherwise available.
        estimate_node_limit: Node limit for the IP-bound estimate solve
            (``None`` = unlimited). Keeps the estimate cheap on hard instances.
        estimate_time_limit: Wall-clock limit (seconds) for the estimate solve
            (``None`` = unlimited).
        frac_tol: A variable value within this tolerance of the nearest integer
            is treated as integral.

    Returns:
        A :class:`RelaxationMetrics`. If SCIP is unavailable the result carries
        ``lp_status="UNAVAILABLE"`` and ``lp_obj=None`` rather than raising.
    """
    try:
        model = to_pyscipopt(relaxed_ir(ir))
    except Exception:  # pragma: no cover - exercised only without SCIP installed
        return RelaxationMetrics(lp_status="UNAVAILABLE", ip_bound=ip_bound)

    model.hideOutput()
    model.setParam("randomization/randomseedshift", 0)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("presolving/maxrounds", 0)
    model.setParam("separating/maxrounds", 0)
    model.setParam("separating/maxroundsroot", 0)
    model.optimize()

    status = model.getStatus()
    if status != "optimal":
        return RelaxationMetrics(lp_status=status.upper(), ip_bound=ip_bound)

    lp_obj = float(model.getObjVal())
    integer_names = {v.name for v in ir.variables if v.vtype in _INTEGER_VTYPES}
    scip_by_name = {v.name: v for v in model.getVars()}
    fractional: list[str] = [
        name
        for name in ir.var_names()
        if name in integer_names
        and abs((val := float(model.getVal(scip_by_name[name]))) - round(val)) > frac_tol
    ]

    bound = _resolve_ip_bound(
        ir,
        ip_bound=ip_bound,
        estimate_ip_bound=estimate_ip_bound,
        node_limit=estimate_node_limit,
        time_limit=estimate_time_limit,
    )
    gap = _integrality_gap(bound, lp_obj)

    return RelaxationMetrics(
        lp_obj=lp_obj,
        gap=gap,
        n_fractional=len(fractional),
        fractional_vars=tuple(fractional),
        lp_status="OPTIMAL",
        ip_bound=bound,
    )


# ---------------------------------------------------------------------------
# IP bound resolution + gap
# ---------------------------------------------------------------------------
def _resolve_ip_bound(
    ir: MILP,
    *,
    ip_bound: float | None,
    estimate_ip_bound: bool,
    node_limit: int | None,
    time_limit: float | None,
) -> float | None:
    """Resolve the IP objective bound: explicit > metadata > estimate > None."""
    if ip_bound is not None:
        return float(ip_bound)
    for key in ("known_optimum", "ip_bound"):
        value = ir.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    if estimate_ip_bound:
        return _estimate_ip_bound(ir, node_limit=node_limit, time_limit=time_limit)
    return None


def _estimate_ip_bound(
    ir: MILP, *, node_limit: int | None, time_limit: float | None
) -> float | None:
    """Estimate the IP bound via a node/time-limited integer solve (threads=1).

    Returns the best primal objective found, or ``None`` if no feasible solution
    was reached within the limits (or SCIP is unavailable).
    """
    try:
        model = to_pyscipopt(ir)
    except Exception:  # pragma: no cover - SCIP missing
        return None
    model.hideOutput()
    model.setParam("randomization/randomseedshift", 0)
    model.setParam("parallel/maxnthreads", 1)
    if node_limit is not None:
        model.setParam("limits/nodes", node_limit)
    if time_limit is not None:
        model.setParam("limits/time", time_limit)
    model.optimize()
    if model.getNSols() > 0:
        return float(model.getPrimalbound())
    return None


def _integrality_gap(ip_bound: float | None, lp_obj: float) -> float | None:
    """Return ``(ip_bound - lp_obj) / |ip_bound|`` or ``None`` when undefined."""
    if ip_bound is None or ip_bound == 0.0 or math.isinf(ip_bound):
        return None
    return (ip_bound - lp_obj) / abs(ip_bound)
