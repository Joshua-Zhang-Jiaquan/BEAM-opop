"""Verification report + certificate record and JSON emission for the gate.

The :class:`VerificationReport` is the single, machine-readable verdict the
verification gate produces for every proposed :class:`~opop.model.state.Delta`.
It is emitted to ``verification/report.json`` *before* any evaluation runs, so a
run is auditable and fail-closed: the orchestrator (task 16) only forwards a
delta to the main solver/evaluator when ``report.status == "pass"``.

Status semantics (mutually exclusive):

* ``pass``    — the matching certificate succeeded; the delta is safe for the
  main evaluation (class A equivalence proven, or class B valid inequality
  certified, or class C semantic no-op).
* ``reject``  — fail-closed: the certificate failed, was unprovable, or the
  delta could not even be applied. NEVER eligible for the main evaluation.
* ``sandbox`` — a class-D (risky / non-certified) delta: routed to sandbox
  experiments only; NEVER returns ``pass``.

The report carries the two preserved-property flags (``bool | None``; ``None``
when not applicable / not computed), a structured ``counterexample`` (a feasible
integer point removed by an invalid "cut", or ``None``), a human-readable
``reason``, and an optional ``certificate`` payload recording the solver-backed
evidence (e.g. the class-B separation bound).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "STATUS_PASS",
    "STATUS_REJECT",
    "STATUS_SANDBOX",
    "VALID_STATUSES",
    "VerificationReport",
    "write_report",
]

#: The delta is safe for the main evaluation.
STATUS_PASS = "pass"
#: Fail-closed: certificate failed / unprovable / inapplicable.
STATUS_REJECT = "reject"
#: Class-D risky delta routed to sandbox; never the main evaluation.
STATUS_SANDBOX = "sandbox"

#: The only legal :attr:`VerificationReport.status` values.
VALID_STATUSES: frozenset[str] = frozenset({STATUS_PASS, STATUS_REJECT, STATUS_SANDBOX})


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """The verdict for one verified :class:`~opop.model.state.Delta`.

    Attributes:
        status: One of :data:`STATUS_PASS`, :data:`STATUS_REJECT`,
            :data:`STATUS_SANDBOX`.
        delta_class: The declared delta class (``"A"`` / ``"B"`` / ``"C"`` /
            ``"D"``) that selected the certificate.
        feasible_region_integer_preserved: ``True`` iff every feasible integer
            solution of the *before* model is preserved; ``False`` when removed;
            ``None`` when not applicable / not computed.
        objective_preserved: ``True`` iff the objective is preserved within
            tolerance; ``False`` when changed; ``None`` when not applicable.
        counterexample: A removed feasible integer point (and the offending
            constraint) when a certificate fails, else ``None``.
        reason: Human-readable explanation of the verdict.
        certificate: Optional solver-backed evidence payload (e.g. the class-B
            separation bounds), or ``None``.
    """

    status: str
    delta_class: str
    feasible_region_integer_preserved: bool | None = None
    objective_preserved: bool | None = None
    counterexample: dict[str, Any] | None = None
    reason: str = ""
    certificate: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"invalid status {self.status!r}; expected one of {sorted(VALID_STATUSES)}"
            )

    @property
    def passed(self) -> bool:
        """``True`` iff the delta is safe for the main evaluation."""
        return self.status == STATUS_PASS

    @property
    def is_sandbox(self) -> bool:
        """``True`` iff the delta was routed to the sandbox (class D)."""
        return self.status == STATUS_SANDBOX

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable mapping (stable key set)."""
        return {
            "status": self.status,
            "delta_class": self.delta_class,
            "feasible_region_integer_preserved": self.feasible_region_integer_preserved,
            "objective_preserved": self.objective_preserved,
            "counterexample": self.counterexample,
            "reason": self.reason,
            "certificate": self.certificate,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the report to a JSON string (sorted keys, deterministic)."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def write_report(
    report: VerificationReport, run_dir: str | Path, *, filename: str = "report.json"
) -> Path:
    """Emit ``report`` to ``<run_dir>/verification/<filename>`` and return its path.

    The ``verification/`` subdirectory is created if missing. ``filename``
    defaults to ``report.json`` (one report per run); the orchestrator passes a
    per-delta name (``report_<iteration>_<delta_idx>.json``) so a multi-delta run
    keeps an auditable certificate for every accepted delta. A trailing newline
    is added for POSIX-friendly diffs.
    """
    out_dir = Path(run_dir) / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(report.to_json() + "\n", encoding="utf-8")
    return path
