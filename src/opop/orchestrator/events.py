"""Append-only ``events.jsonl`` journal for the Phase-1 closed loop.

Every proposal the orchestrator processes — whether it was solved or rejected
by the verification gate — produces exactly one JSON record (one line). The
schema (per the plan) is::

    {
      "iter": int,
      "instance_id": str,
      "phi": {flat design vector},
      "delta_target": str,
      "delta_class": "A" | "B" | "C" | "D" | null,
      "verify_status": "pass" | "reject" | "sandbox" | "apply_error" | "solve_error",
      "trace_summary": {status, objective, gap, censored, nodes, time} | null,
      "score": {primal_integral, gap, time} | null,
      "incumbent_so_far": float | null,
      "accepted": bool,
      "reward": float | null,
      "reason": str,
      ...flat cost columns from opop.bench.cost.COST_FIELDS...
    }

The ``instance_id`` ties every row to the benchmark instance being tuned so the
leakage audit (``opop.bench.audit_leakage``) can cross-reference it against the
registry's held-out split manifest. When the loop passes a cost row, the flat
columns of :data:`opop.bench.cost.COST_FIELDS` (``solver_wall_time``,
``analyzer_time``, ``proposer_time``, ``controller_time``, ``verification_time``,
``evaluate_time``, ``llm_tokens_in``, ``llm_tokens_out``, ``llm_cost_usd``,
``total_wall_time``) are merged in so cost accounting can report solver-only
**and** end-to-end wall time.

Non-finite metric values (``inf`` / ``nan``) are sanitised to ``None`` so each
line is strictly well-formed JSON (``json.dumps(..., allow_nan=False)`` would
otherwise raise; we sanitise first and keep ``allow_nan=False`` as a fail-loud
guard).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from types import TracebackType
    from typing import TextIO

    from opop.model.state import Delta, Phi, ScoreRecord, SolveTrace

__all__ = ["EventWriter", "build_event", "score_summary", "trace_summary"]


def _json_num(value: float | None) -> float | None:
    """Map a numeric value to a JSON-safe float (``inf`` / ``nan`` -> ``None``)."""
    if value is None:
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def trace_summary(trace: SolveTrace, score: ScoreRecord) -> dict[str, Any]:
    """Summarise a solve for the journal: status / objective / gap / censored / nodes / time.

    Status / censoring / node count come straight from the :class:`SolveTrace`;
    the objective, gap, and elapsed solve time are taken from the (already
    computed) :class:`ScoreRecord` so the numbers match the evaluator exactly.
    """
    m = score.metrics
    return {
        "status": trace.status,
        "objective": _json_num(m.get("objective")),
        "gap": _json_num(m.get("gap")),
        "censored": bool(trace.censored),
        "nodes": int(trace.nodes),
        "time": _json_num(m.get("solve_time")),
    }


def score_summary(score: ScoreRecord) -> dict[str, Any]:
    """Summarise the score for the journal: primal integral / gap / time."""
    m = score.metrics
    return {
        "primal_integral": _json_num(m.get("primal_integral")),
        "gap": _json_num(m.get("gap")),
        "time": _json_num(m.get("solve_time")),
    }


def build_event(
    *,
    iteration: int,
    phi: Phi,
    delta: Delta,
    verify_status: str,
    trace: SolveTrace | None = None,
    score: ScoreRecord | None = None,
    incumbent_so_far: float | None = None,
    reward: float | None = None,
    reason: str = "",
    accepted: bool = False,
    instance_id: str = "",
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one journal record for a single processed proposal.

    ``trace_summary`` and ``score`` are populated only for a solved (passed)
    delta; gate-rejected / errored proposals carry ``null`` for both.
    ``instance_id`` tags the row for the leakage audit; ``cost`` (when given)
    merges its flat columns in for cost accounting.
    """
    record: dict[str, Any] = {
        "iter": int(iteration),
        "instance_id": str(instance_id),
        "phi": phi.to_flat_dict(),
        "delta_target": delta.target,
        "delta_class": delta.declared_class.value,
        "verify_status": verify_status,
        "trace_summary": (
            trace_summary(trace, score) if trace is not None and score is not None else None
        ),
        "score": score_summary(score) if score is not None else None,
        "incumbent_so_far": _json_num(incumbent_so_far),
        "accepted": bool(accepted),
        "reward": _json_num(reward),
        "reason": reason,
    }
    if cost is not None:
        record.update(cost)
    return record


class EventWriter:
    """Append-only writer for ``events.jsonl`` (one JSON object per line).

    Opens the file fresh (truncating any prior run) and flushes after every
    record so the journal survives an interrupt mid-loop. Usable as a context
    manager.
    """

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("w", encoding="utf-8")
        self.count: int = 0

    def append(self, record: dict[str, Any]) -> None:
        """Write one record as a single JSON line (fail-loud on ``inf`` / ``nan``)."""
        line = json.dumps(record, allow_nan=False, sort_keys=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        """Close the underlying file (idempotent)."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> EventWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False
