"""Anytime + quality metrics for a :class:`~opop.model.state.SolveTrace`.

All functions here are pure: they read a frozen ``SolveTrace`` and return
scalars.  No solver imports, no mutation.

The headline metric is the **primal integral** (Berthold, 2013): the
time-integral of the normalised primal gap along the solve trajectory.  It is
computed from the *whole* bound trajectory (NOT just the endpoints), so a
solver that finds a good incumbent early scores strictly better than one that
only matches it at the time limit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from opop.model.state import SolveTrace

# Guard floor for the gap denominator (matches the spec: max(|primal|, 1e-12)).
_GAP_DENOM_FLOOR = 1e-12


def _trapezoid(y: ArrayLike, x: ArrayLike) -> float:
    """Trapezoidal integral ``∫ y dx`` via numpy.

    Prefers :func:`numpy.trapezoid` (NumPy >= 2.0) and falls back to the legacy
    :func:`numpy.trapz` on NumPy 1.x, so the same call works on the pinned-1.26
    dev environment and on a future 2.x upgrade.
    """
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(np.asarray(y, dtype=np.float64), x=np.asarray(x, dtype=np.float64)))


def normalized_gap(primal: float, reference: float) -> float:
    """Normalised primal gap ``|primal - reference| / max(|primal|, 1e-12)``.

    ``reference`` is the dual bound (the solver's own optimality gap) or a
    known/reference optimum.  Non-finite inputs (e.g. a ``+inf`` primal before
    the first incumbent) collapse to ``1.0`` — the conventional "no useful
    incumbent, 100% gap" value — so the integral stays finite.
    """
    if not (math.isfinite(primal) and math.isfinite(reference)):
        return 1.0
    denom = max(abs(primal), _GAP_DENOM_FLOOR)
    return abs(primal - reference) / denom


def _step_integral(values: Sequence[float], times: Sequence[float]) -> float:
    """Integrate a piecewise-constant (left-held) series via the trapezoid rule.

    The primal integral is the integral of a *step* function: the gap recorded
    at ``times[i]`` is held constant until ``times[i+1]``.  We integrate it with
    the trapezoidal rule over a breakpoint-duplicated series, which equals the
    left-Riemann sum ``Σ values[i] * (times[i+1] - times[i])`` — but routed
    through numpy's trapezoid integrator per the task spec.  Fewer than two
    points means no elapsed interval, so the integral is ``0.0``.
    """
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = min(t.size, v.size)
    if n < 2:
        return 0.0
    t = t[:n]
    v = v[:n]
    # Duplicate breakpoints -> [t0, t1, t1, t2, ..., t_{n-1}] paired with
    # [v0, v0, v1, v1, ..., v_{n-2}, v_{n-2}] so the trapezoid rule reproduces
    # the left-held step function exactly.
    step_t = np.repeat(t, 2)[1:-1]
    step_v = np.repeat(v[:-1], 2)
    return _trapezoid(step_v, step_t)


def gap_series(
    trace: SolveTrace, *, reference_optimum: float | None = None
) -> tuple[list[float], list[float]]:
    """Return ``(normalised-gap-per-point, times)`` along the trajectory.

    Each point's gap is ``|primal_i - other| / max(|primal_i|, 1e-12)`` where
    ``other`` is ``reference_optimum`` if given, else the per-point dual bound.
    The series are truncated to their common length (they are index-aligned by
    the ``SolveTrace`` contract).
    """
    primal = trace.primal_bound_series
    times = trace.time_series
    if reference_optimum is not None:
        n = min(len(primal), len(times))
        gaps = [normalized_gap(primal[i], reference_optimum) for i in range(n)]
    else:
        dual = trace.dual_bound_series
        n = min(len(primal), len(dual), len(times))
        gaps = [normalized_gap(primal[i], dual[i]) for i in range(n)]
    return gaps, list(times[:n])


def primal_integral(trace: SolveTrace, *, reference_optimum: float | None = None) -> float:
    """Time-integral of the normalised primal gap (Berthold anytime metric).

    Uses ``reference_optimum`` as the convergence target when provided (gap to
    the known/best optimum), else the solver's own dual bound.  Computed from
    the full bound trajectory, not the endpoints.
    """
    gaps, times = gap_series(trace, reference_optimum=reference_optimum)
    return _step_integral(gaps, times)


def primal_dual_gap_integral(trace: SolveTrace) -> float:
    """Time-integral of the primal-dual gap (always primal vs dual bound).

    Unlike :func:`primal_integral`, this ignores any reference optimum and
    always measures the solver's own optimality-gap closure over time.
    """
    gaps, times = gap_series(trace, reference_optimum=None)
    return _step_integral(gaps, times)


def final_gap(trace: SolveTrace, *, reference_optimum: float | None = None) -> float:
    """Final optimality gap at the time limit (gap@T).

    ``|primal_T - other| / max(|primal_T|, 1e-12)`` where ``other`` is the
    reference optimum if given, else the final dual bound.  ``NaN`` if the
    trajectory carries no primal (or, without a reference, no dual) point.
    """
    primal = trace.primal_bound_series
    if not primal:
        return float("nan")
    last_primal = primal[-1]
    if reference_optimum is not None:
        other = reference_optimum
    else:
        dual = trace.dual_bound_series
        if not dual:
            return float("nan")
        other = dual[-1]
    return normalized_gap(last_primal, other)


def objective(trace: SolveTrace) -> float:
    """Final primal (incumbent) objective value, or ``NaN`` if none is finite."""
    primal = trace.primal_bound_series
    if not primal:
        return float("nan")
    last = primal[-1]
    return last if math.isfinite(last) else float("nan")


def is_feasible(trace: SolveTrace) -> bool:
    """``True`` iff a finite primal incumbent exists.

    Honours the spec heuristic (``first_feasible_time`` set, or a primal series
    present) but refined for the ``+inf`` no-incumbent sentinel: an all-``inf``
    primal series from an infeasible / no-incumbent run does NOT count.
    """
    if not math.isnan(trace.first_feasible_time):
        return True
    return any(math.isfinite(p) for p in trace.primal_bound_series)


def is_optimal(trace: SolveTrace) -> bool:
    """``True`` iff the solver certified optimality (and the run was not censored).

    Enforces the guardrail "never treat a censored run as optimal": a censored
    trace is never optimal regardless of its status string.
    """
    if trace.censored:
        return False
    return trace.status.strip().lower() == "optimal"
