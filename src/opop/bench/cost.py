"""Solver-only **and** end-to-end cost accounting for the Phase-1 loop.

The scientific-integrity rule this module enforces: **no compute-efficiency
claim may be made from solver-only time**. Every event/result row therefore
carries a full per-phase cost breakdown, and the headline aggregate reports the
*honest* end-to-end wall time (analyzer + proposer + controller + verification +
solver + evaluate) alongside the solver-only wall time. By construction
``total_wall_time >= solver_wall_time`` for every row and every aggregate.

Per-row cost schema (flat columns merged into each ``events.jsonl`` record):

    solver_wall_time   — time inside ``kernel.solve`` for this delta (0 if unsolved)
    analyzer_time      — structural analysis time (charged once per run)
    proposer_time      — candidate-delta proposal time (charged once per iteration)
    controller_time    — ask/tell time (charged once per iteration)
    verification_time  — verification-gate time for this delta
    evaluate_time      — metric-evaluation time for this delta (an honest extra)
    llm_tokens_in      — prompt tokens consumed producing this row
    llm_tokens_out     — completion tokens consumed producing this row
    llm_cost_usd       — USD cost of those tokens
    total_wall_time    — sum of the six time components above

Attribution (see :class:`CostAccountant`): run-level costs (``analyzer_time``)
and iteration-level costs (``proposer_time`` / ``controller_time``) are charged
in full to the **first** event of their scope and zero to the rest, so that
``sum`` over the per-event rows reconstructs the run total exactly — letting
:func:`cost_summary` aggregate by a plain sum without double-counting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "COST_FIELDS",
    "REQUIRED_COST_COLUMNS",
    "CostAccountant",
    "cost_summary",
    "empty_cost",
    "make_event_cost",
]

#: The nine cost columns the plan mandates on every result/event row.
REQUIRED_COST_COLUMNS: tuple[str, ...] = (
    "solver_wall_time",
    "analyzer_time",
    "proposer_time",
    "controller_time",
    "verification_time",
    "llm_tokens_in",
    "llm_tokens_out",
    "llm_cost_usd",
    "total_wall_time",
)

#: Every column emitted by :func:`make_event_cost` — the nine required columns
#: plus ``evaluate_time`` (kept explicit so ``total_wall_time`` reconciles with
#: the sum of its time components rather than hiding the evaluator's cost).
COST_FIELDS: tuple[str, ...] = (*REQUIRED_COST_COLUMNS, "evaluate_time")

#: Time components that sum to ``total_wall_time``.
_TIME_COMPONENTS: tuple[str, ...] = (
    "solver_wall_time",
    "analyzer_time",
    "proposer_time",
    "controller_time",
    "verification_time",
    "evaluate_time",
)


def make_event_cost(
    *,
    solver_wall_time: float = 0.0,
    analyzer_time: float = 0.0,
    proposer_time: float = 0.0,
    controller_time: float = 0.0,
    verification_time: float = 0.0,
    evaluate_time: float = 0.0,
    llm_tokens_in: int = 0,
    llm_tokens_out: int = 0,
    llm_cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Build one flat cost row; ``total_wall_time`` = sum of the time components.

    Each time component is clamped to ``>= 0`` (a monotonic clock never goes
    backwards, but clamping makes the ``total_wall_time >= solver_wall_time``
    invariant hold unconditionally). Token counts are non-negative integers.
    """
    solve = max(0.0, float(solver_wall_time))
    analyze = max(0.0, float(analyzer_time))
    propose = max(0.0, float(proposer_time))
    control = max(0.0, float(controller_time))
    verify = max(0.0, float(verification_time))
    evaluate = max(0.0, float(evaluate_time))
    total = solve + analyze + propose + control + verify + evaluate
    return {
        "solver_wall_time": solve,
        "analyzer_time": analyze,
        "proposer_time": propose,
        "controller_time": control,
        "verification_time": verify,
        "evaluate_time": evaluate,
        "llm_tokens_in": int(llm_tokens_in),
        "llm_tokens_out": int(llm_tokens_out),
        "llm_cost_usd": max(0.0, float(llm_cost_usd)),
        "total_wall_time": total,
    }


def empty_cost() -> dict[str, Any]:
    """An all-zero cost row carrying every column (the no-instrumentation default)."""
    return make_event_cost()


def _tracker_totals(tracker: Any) -> tuple[int, int, float]:
    """Read cumulative ``(tokens_in, tokens_out, cost_usd)`` from a TokenTracker.

    Tolerant of ``None`` (no LLM client) and of trackers missing an attribute,
    so cost accounting never crashes the loop over telemetry.
    """
    if tracker is None:
        return (0, 0, 0.0)
    tokens_in = int(getattr(tracker, "total_tokens_in", 0) or 0)
    tokens_out = int(getattr(tracker, "total_tokens_out", 0) or 0)
    cost_usd = float(getattr(tracker, "total_cost_usd", 0.0) or 0.0)
    return (tokens_in, tokens_out, cost_usd)


class CostAccountant:
    """Threads per-phase timing + LLM token deltas through one ``run_loop``.

    Usage mirrors the loop's control flow::

        acct = CostAccountant(tracker=llm.tracker if llm else None)
        acct.record_analyzer(dt_analyze)            # once, before the loop
        for iteration:
            acct.start_iteration(ask_t=..., proposer_t=...)
            for delta:
                cost = acct.event_cost(verify_t=..., solve_t=..., eval_t=...)
                writer.append(build_event(..., cost=cost))
            acct.record_tell(tell_t)                # after the iteration's events
        run_cost = acct.run_summary()               # -> result.json

    Run-level costs (analyzer) and iteration-level costs (proposer, controller
    ask) are charged to the *first* event of their scope; ``controller`` also
    carries the previous iteration's ``tell`` forward. The final ``tell`` (after
    the last event) lands only in :meth:`run_summary`, so per-event rows sum to
    the run total up to that single trailing tell — a deliberate, documented gap.
    """

    def __init__(self, tracker: Any = None) -> None:
        self._tracker: Any = tracker
        # Cumulative run-level phase totals.
        self.analyzer_time: float = 0.0
        self.proposer_time: float = 0.0
        self.controller_time: float = 0.0
        self.verification_time: float = 0.0
        self.solver_wall_time: float = 0.0
        self.evaluate_time: float = 0.0
        # Pending charges consumed by the next emitted event.
        self._pending_analyzer: float = 0.0
        self._pending_proposer: float = 0.0
        self._pending_controller: float = 0.0
        # Last observed cumulative token/cost totals (for per-event deltas).
        self._prev_in, self._prev_out, self._prev_cost = _tracker_totals(tracker)

    def record_analyzer(self, dt: float) -> None:
        """Record the one-shot analyzer phase (charged to the first event)."""
        dt = max(0.0, float(dt))
        self.analyzer_time += dt
        self._pending_analyzer += dt

    def start_iteration(self, *, ask_t: float, proposer_t: float) -> None:
        """Open an iteration: queue this iteration's ask + proposer charges."""
        ask_t = max(0.0, float(ask_t))
        proposer_t = max(0.0, float(proposer_t))
        self.controller_time += ask_t
        self.proposer_time += proposer_t
        self._pending_controller += ask_t
        self._pending_proposer += proposer_t

    def record_tell(self, dt: float) -> None:
        """Record a ``controller.tell``; carried into the next event's charge."""
        dt = max(0.0, float(dt))
        self.controller_time += dt
        self._pending_controller += dt

    def event_cost(self, *, verify_t: float, solve_t: float, eval_t: float) -> dict[str, Any]:
        """Emit one per-event cost row and advance all internal accounting."""
        verify_t = max(0.0, float(verify_t))
        solve_t = max(0.0, float(solve_t))
        eval_t = max(0.0, float(eval_t))
        self.verification_time += verify_t
        self.solver_wall_time += solve_t
        self.evaluate_time += eval_t

        cur_in, cur_out, cur_cost = _tracker_totals(self._tracker)
        d_in = max(0, cur_in - self._prev_in)
        d_out = max(0, cur_out - self._prev_out)
        d_cost = max(0.0, cur_cost - self._prev_cost)
        self._prev_in, self._prev_out, self._prev_cost = cur_in, cur_out, cur_cost

        analyze = self._pending_analyzer
        propose = self._pending_proposer
        control = self._pending_controller
        self._pending_analyzer = 0.0
        self._pending_proposer = 0.0
        self._pending_controller = 0.0

        return make_event_cost(
            solver_wall_time=solve_t,
            analyzer_time=analyze,
            proposer_time=propose,
            controller_time=control,
            verification_time=verify_t,
            evaluate_time=eval_t,
            llm_tokens_in=d_in,
            llm_tokens_out=d_out,
            llm_cost_usd=d_cost,
        )

    def run_summary(self) -> dict[str, Any]:
        """The authoritative run-level cost row (full totals + final tell)."""
        tokens_in, tokens_out, cost_usd = _tracker_totals(self._tracker)
        return make_event_cost(
            solver_wall_time=self.solver_wall_time,
            analyzer_time=self.analyzer_time,
            proposer_time=self.proposer_time,
            controller_time=self.controller_time,
            verification_time=self.verification_time,
            evaluate_time=self.evaluate_time,
            llm_tokens_in=tokens_in,
            llm_tokens_out=tokens_out,
            llm_cost_usd=cost_usd,
        )


def _num(value: Any) -> float:
    """Coerce a possibly-``None`` JSON numeric to a finite float (``None`` -> 0)."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def cost_summary(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate solver-only and end-to-end cost over a run's event rows.

    Returns headline ``solver_only_wall_time`` (sum of ``solver_wall_time``) and
    ``end_to_end_wall_time`` (sum of ``total_wall_time``) — by construction
    ``end_to_end_wall_time >= solver_only_wall_time`` — plus per-phase totals,
    token/cost totals, and a ``per_iteration`` breakdown keyed by the event's
    ``iter`` field (each with its own solver-only and end-to-end wall time).
    """
    time_totals: dict[str, float] = {name: 0.0 for name in _TIME_COMPONENTS}
    end_to_end_total = 0.0
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0
    n_events = 0
    per_iteration: dict[str, dict[str, float]] = {}

    for event in events:
        n_events += 1
        for name in _TIME_COMPONENTS:
            time_totals[name] += _num(event.get(name))
        end_to_end_total += _num(event.get("total_wall_time"))
        tokens_in += int(_num(event.get("llm_tokens_in")))
        tokens_out += int(_num(event.get("llm_tokens_out")))
        cost_usd += _num(event.get("llm_cost_usd"))

        key = str(event.get("iter", ""))
        bucket = per_iteration.setdefault(
            key, {"solver_only_wall_time": 0.0, "end_to_end_wall_time": 0.0}
        )
        bucket["solver_only_wall_time"] += _num(event.get("solver_wall_time"))
        bucket["end_to_end_wall_time"] += _num(event.get("total_wall_time"))

    return {
        "n_events": n_events,
        "solver_only_wall_time": time_totals["solver_wall_time"],
        "end_to_end_wall_time": end_to_end_total,
        "analyzer_time": time_totals["analyzer_time"],
        "proposer_time": time_totals["proposer_time"],
        "controller_time": time_totals["controller_time"],
        "verification_time": time_totals["verification_time"],
        "evaluate_time": time_totals["evaluate_time"],
        "llm_tokens_in": tokens_in,
        "llm_tokens_out": tokens_out,
        "llm_cost_usd": cost_usd,
        "per_iteration": per_iteration,
    }
