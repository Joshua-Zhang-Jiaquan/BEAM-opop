"""OPOP verification gate: solver-backed delta-class A--D certificates.

Public API:

* :func:`verify_delta` -- classify a :class:`~opop.model.state.Delta` and run
  the matching certificate; returns a :class:`VerificationReport` (fail-closed).
* :class:`VerificationReport` + :func:`write_report` -- the verdict record and
  its ``verification/report.json`` emitter.
* ``STATUS_PASS`` / ``STATUS_REJECT`` / ``STATUS_SANDBOX`` -- status constants.
* ``FEAS_TOL`` (1e-7) / ``OBJ_TOL`` (1e-6) -- the locked certificate tolerances.
"""

from __future__ import annotations

from opop.verify.certificate import (
    STATUS_PASS,
    STATUS_REJECT,
    STATUS_SANDBOX,
    VALID_STATUSES,
    VerificationReport,
    write_report,
)
from opop.verify.gate import (
    FEAS_TOL,
    OBJ_TOL,
    SEPARATION_TIME_LIMIT,
    verify_delta,
)

__all__ = [
    "FEAS_TOL",
    "OBJ_TOL",
    "SEPARATION_TIME_LIMIT",
    "STATUS_PASS",
    "STATUS_REJECT",
    "STATUS_SANDBOX",
    "VALID_STATUSES",
    "VerificationReport",
    "verify_delta",
    "write_report",
]
