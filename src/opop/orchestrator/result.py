"""Result containers for the Phase-1 orchestrator closed loop.

:class:`RunResult` is the machine-readable summary :func:`opop.orchestrator.loop.run_loop`
returns (and persists to ``result.json``). It carries the final
:class:`Incumbent` (the best certified, evaluated configuration found), the
accepted/rejected counts, where the event journal lives, why the loop stopped,
and a ``repro_manifest_ref`` hook the reproducibility task (task 17) fills in.

Both records are JSON-serialisable via ``to_dict()``; non-finite metric values
(``inf`` / ``nan``) are sanitised to ``None`` so the emitted JSON is strictly
well-formed (no ``NaN`` / ``Infinity`` tokens).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opop.model.state import Phi, ScoreRecord


def _json_num(value: float | None) -> float | None:
    """Map a numeric value to a JSON-safe float (``inf`` / ``nan`` -> ``None``)."""
    if value is None:
        return None
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _sanitize_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Return a copy of ``metrics`` with non-finite values mapped to ``None``."""
    return {k: _json_num(v) for k, v in metrics.items()}


@dataclass(frozen=True, slots=True)
class Incumbent:
    """The best certified, evaluated configuration found by the loop.

    Attributes:
        phi: The effective design vector that produced this incumbent (the
            controller's proposal merged with any verified param delta).
        score: The evaluator's :class:`~opop.model.state.ScoreRecord`.
        reward: The scalarised reward (higher is better) that beat the running
            best.
        certificate: The verification report (``to_dict``) of the delta that
            produced this incumbent — the auditable proof it cleared the gate.
        delta_target: Human-readable description of the producing delta.
        delta_class: The declared verification class of the producing delta
            (``"A"`` / ``"B"`` / ``"C"``), or ``None`` for a baseline solve.
        iteration: The 0-based loop iteration that produced this incumbent.
    """

    phi: Phi
    score: ScoreRecord
    reward: float
    certificate: dict[str, Any] | None
    delta_target: str
    delta_class: str | None
    iteration: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (metrics sanitised, phi flattened)."""
        return {
            "phi": self.phi.to_flat_dict(),
            "score": _sanitize_metrics(self.score.metrics),
            "risks": list(self.score.risks),
            "reward": _json_num(self.reward),
            "certificate": self.certificate,
            "delta_target": self.delta_target,
            "delta_class": self.delta_class,
            "iteration": self.iteration,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Summary of one Phase-1 closed-loop run.

    Attributes:
        incumbent: The best configuration found, or ``None`` if nothing was
            ever solved (every proposal was gate-rejected / errored).
        n_iterations: Number of loop iterations actually executed.
        n_accepted: Deltas that passed the gate AND were solved + scored.
        n_rejected: Deltas the gate rejected / sandboxed or that failed to
            apply (NEVER solved).
        events_path: Path to the ``events.jsonl`` journal.
        out_dir: The run output directory.
        stopped_reason: Why the loop stopped — ``"budget"``, ``"time_budget"``,
            ``"stagnation"``, or ``"interrupted"``.
        repro_manifest_ref: Hook for task 17 (reproducibility manifest); left
            ``None`` by Phase-1.
    """

    incumbent: Incumbent | None
    n_iterations: int
    n_accepted: int
    n_rejected: int
    events_path: Path | None
    out_dir: Path | None
    stopped_reason: str
    repro_manifest_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (``Path`` fields stringified)."""
        return {
            "incumbent": self.incumbent.to_dict() if self.incumbent is not None else None,
            "n_iterations": self.n_iterations,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "events_path": str(self.events_path) if self.events_path is not None else None,
            "out_dir": str(self.out_dir) if self.out_dir is not None else None,
            "stopped_reason": self.stopped_reason,
            "repro_manifest_ref": self.repro_manifest_ref,
        }
