"""Static leaderboard builder + integrity-gated submission protocol (task 42).

Public API:

* :class:`LeaderboardBuilder` — reads ``results.parquet`` + ``thesis_report.json``
  from a run directory, aggregates per-method metrics with confidence intervals,
  and emits a static ``site/index.html`` (+ ``site/leaderboard.md`` fallback).
* :class:`SubmissionValidator` — validates a run directory for the required
  integrity artifacts (``repro_manifest.json``, ``leakage_audit.json``, sealed
  registry lock, results file) and returns accepted/rejected with reason.

CLI: ``python -m opop.leaderboard build|submit`` (see :mod:`__main__`).
"""

from __future__ import annotations

from opop.leaderboard.builder import LeaderboardBuilder, LeaderboardData
from opop.leaderboard.submit import SubmissionValidator, SubmissionResult

__all__ = [
    "LeaderboardBuilder",
    "LeaderboardData",
    "SubmissionResult",
    "SubmissionValidator",
]
