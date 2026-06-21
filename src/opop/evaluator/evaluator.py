"""Turn a :class:`SolveTrace` into a :class:`ScoreRecord`.

This is the Evaluator layer of the OPOP loop
(Proposer -> Analyzer -> Solver -> **Evaluator** -> Bayesian Controller).  It
maps one solver trajectory (plus an optional reference optimum and time limit)
to a flat, multi-metric ``ScoreRecord`` carrying the anytime primal integral,
optimality gap, censoring flags, and an auxiliary PAR10 runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opop.model.state import ScoreRecord

from .censoring import is_censored, par10, runtime
from .metrics import (
    final_gap,
    is_feasible,
    is_optimal,
    objective,
    primal_dual_gap_integral,
    primal_integral,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opop.model.state import SolveTrace


def evaluate(
    trace: SolveTrace,
    *,
    reference_optimum: float | None = None,
    time_limit: float | None = None,
) -> ScoreRecord:
    """Score a single solve trajectory.

    Args:
        trace: The solver trajectory to evaluate.
        reference_optimum: Known / best objective used as the primal-integral
            and gap target.  When ``None`` the solver's own dual bound is used.
        time_limit: Run time limit (seconds); required for the PAR10 auxiliary.

    Returns:
        A :class:`ScoreRecord` whose ``metrics`` dict carries (all floats):
        ``feasible``, ``optimal`` (solver-certified optimality), ``objective``,
        ``gap`` (gap@T), ``time_to_first_feasible``, ``solve_time``,
        ``primal_integral``, ``primal_dual_gap_integral``, ``nodes``, ``cuts``,
        ``memory_peak``, ``censored``, and ``par10_aux`` (labeled auxiliary).
        ``uncertainty`` is ``None`` (no replays in Phase-1); ``risks`` lists
        human-readable warnings.
    """
    feasible = is_feasible(trace)
    optimal = is_optimal(trace)
    censored = is_censored(trace)
    solve_time = runtime(trace)

    metrics: dict[str, float] = {
        "feasible": float(feasible),
        # ``optimal`` doubles as the solver-certified-optimality flag (Phase-1
        # has no separate formulation-delta certificate in the ScoreRecord).
        "optimal": float(optimal),
        "objective": objective(trace),
        "gap": final_gap(trace, reference_optimum=reference_optimum),
        "time_to_first_feasible": float(trace.first_feasible_time),
        "solve_time": solve_time,
        "primal_integral": primal_integral(trace, reference_optimum=reference_optimum),
        "primal_dual_gap_integral": primal_dual_gap_integral(trace),
        "nodes": float(trace.nodes),
        "cuts": float(trace.cuts),
        "memory_peak": float(trace.memory_peak),
        "censored": float(censored),
        # Auxiliary only (10 x limit on timeout); NOT the primary runtime.
        "par10_aux": par10(solve_time, time_limit, censored=censored),
    }

    risks: list[str] = []
    if censored:
        risks.append("censored")
    if not feasible:
        risks.append("no-feasible-incumbent")

    # uncertainty is None in Phase-1 (no replays / repeated solves yet).
    return ScoreRecord(
        metrics=metrics,
        uncertainty=None,
        risks=risks,
        instance_id=trace.instance_id,
    )


# Default CO/IP scalarization weights — mirror task-15 ``coip_reward``.
_DEFAULT_WEIGHTS: dict[str, float] = {"w_gap": 1.0, "w_time": 1e-3, "w_pi": 1.0}


def scalarize(record: ScoreRecord, weights: Mapping[str, float] | None = None) -> float:
    """Scalarize a :class:`ScoreRecord` into a BO reward (higher is better).

    Local mirror of :func:`opop.controller.phase1.coip_reward`:
    ``reward = -w_gap*gap - w_time*solve_time - w_pi*primal_integral``.

    Implemented here (rather than imported) so the Evaluator stays independent
    of the controller layer.  The ``ScoreRecord.metrics`` keys (``gap``,
    ``solve_time``, ``primal_integral``) are chosen to be drop-in compatible
    with ``coip_reward``; a regression test locks the two to identical values.

    Args:
        record: Score record to scalarize.
        weights: Optional ``{w_gap, w_time, w_pi}`` overrides.

    Returns:
        Scalar reward (higher is better).
    """
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    m = record.metrics
    gap = float(m.get("gap", 0.0))
    solve_time = float(m.get("solve_time", m.get("time", m.get("runtime_seconds", 0.0))))
    primal = float(m.get("primal_integral", 0.0))
    return -(w["w_gap"] * gap) - (w["w_time"] * solve_time) - (w["w_pi"] * primal)
