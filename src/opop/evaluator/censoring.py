"""Right-censoring + PAR10 handling for solver runtimes.

A solver run that hits the time / resource limit before proving optimality is
*right-censored*: its recorded runtime is a LOWER BOUND on the true solve time,
not the true value.  The primary score keeps this censored runtime as-is and
flags ``censored=True``; it MUST NOT be replaced by a fixed penalty.

PAR10 (Penalised Average Runtime, factor 10) is provided as a CLEARLY-LABELED
AUXILIARY only: a censored run contributes ``10 * time_limit``.  This is a
standard SAT / ML-for-solver aggregate, but it is biased and must never be used
as the primary anytime metric.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opop.model.state import SolveTrace

#: PAR penalty factor: a censored run is charged ``PAR_FACTOR * time_limit``.
PAR_FACTOR = 10.0


def is_censored(trace: SolveTrace) -> bool:
    """Whether the run was terminated by a limit before an optimality proof."""
    return bool(trace.censored)


def runtime(trace: SolveTrace) -> float:
    """Recorded wall-clock solve time = final trajectory timestamp.

    For a censored run this is a LOWER BOUND on the true solve time (the run
    was cut off at the limit).  Returns ``NaN`` for an empty trajectory.
    """
    times = trace.time_series
    if not times:
        return float("nan")
    return float(times[-1])


def par10(runtime_value: float, time_limit: float | None, *, censored: bool) -> float:
    """PAR10 auxiliary runtime (penalised average runtime, factor 10).

    A censored run is penalised to ``PAR_FACTOR * time_limit``; an uncensored
    run keeps its actual runtime.  Returns ``NaN`` if a penalty is required but
    no finite ``time_limit`` is available.

    This is an AUXILIARY metric only — never the primary anytime score, and it
    never overwrites the censored runtime in the primary record.
    """
    if censored:
        if time_limit is None or not math.isfinite(time_limit):
            return float("nan")
        return PAR_FACTOR * float(time_limit)
    return float(runtime_value)
