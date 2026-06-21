"""Phase-1 orchestrator: the closed loop integrating every Phase-1 module.

Public API:

* :func:`run_loop` — drive Proposer -> Analyzer -> Verify gate -> Solver ->
  Evaluator -> Controller until budget / stagnation, persisting a journal,
  the running incumbent, and a result summary.
* :class:`RunResult` / :class:`Incumbent` — the run summary + best config.
* :class:`EventWriter` — the ``events.jsonl`` journal writer.
* :class:`OrchestratorError` — unrecoverable wiring error.
"""

from __future__ import annotations

from opop.orchestrator.events import EventWriter, build_event
from opop.orchestrator.loop import OrchestratorError, run_loop
from opop.orchestrator.result import Incumbent, RunResult

__all__ = [
    "EventWriter",
    "Incumbent",
    "OrchestratorError",
    "RunResult",
    "build_event",
    "run_loop",
]
