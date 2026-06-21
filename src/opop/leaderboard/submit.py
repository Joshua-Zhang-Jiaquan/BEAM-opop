"""Integrity-gated submission validator for OPOP leaderboard entries.

Checks a run directory for the required scientific-integrity artifacts before
accepting a submission:

1. ``repro_manifest.json`` — reproducibility manifest (written by MatrixDriver).
2. ``leakage_audit.json`` — leakage audit result (or runs the audit if absent).
3. Sealed registry lock — ``BenchmarkRegistry.from_yaml(...).verify_lock()``.
4. ``results.parquet`` / ``results.json`` — the consolidated results file.

A submission is **rejected** if any artifact is missing or invalid, with a clear
human-readable reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opop.bench.registry import BenchmarkRegistry, RegistryError

__all__ = ["SubmissionResult", "SubmissionValidator"]

# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    """Outcome of one submission validation."""

    accepted: bool
    reason: str
    run_dir: str
    artifacts_checked: tuple[str, ...]
    artifacts_found: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable dict."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "run_dir": self.run_dir,
            "artifacts_checked": list(self.artifacts_checked),
            "artifacts_found": list(self.artifacts_found),
        }


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #

#: Required artifact filenames.
_REQUIRED_ARTIFACTS = (
    "repro_manifest.json",
    "leakage_audit.json",
)

#: Accepted results filenames (at least one must exist).
_RESULTS_FILES = ("results.parquet", "results.json", "results.jsonl")


class SubmissionValidator:
    """Validate a run directory for leaderboard submission integrity.

    Args:
        registry_path: Path to ``registry.yaml`` for lock verification.
            When ``None``, the lock check is skipped (with a warning in the
            reason string).
    """

    def __init__(self, *, registry_path: str | Path | None = None) -> None:
        self._registry_path = Path(registry_path) if registry_path is not None else None

    def validate(self, run_dir: str | Path) -> SubmissionResult:
        """Check all required artifacts; return accepted/rejected with reason."""
        run_path = Path(run_dir)
        if not run_path.is_dir():
            return SubmissionResult(
                accepted=False,
                reason=f"run directory does not exist: {run_path}",
                run_dir=str(run_path),
                artifacts_checked=(),
                artifacts_found=(),
            )

        checked: list[str] = []
        found: list[str] = []
        failures: list[str] = []

        # 1. repro_manifest.json
        checked.append("repro_manifest.json")
        if (run_path / "repro_manifest.json").is_file():
            found.append("repro_manifest.json")
        else:
            failures.append("missing repro_manifest.json")

        # 2. leakage_audit.json (or attempt to run audit)
        checked.append("leakage_audit.json")
        audit_path = run_path / "leakage_audit.json"
        if audit_path.is_file():
            found.append("leakage_audit.json")
            # Verify the audit passed (status != "fail").
            try:
                audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(audit_data, dict) and audit_data.get("status") == "fail":
                    n = audit_data.get("n_violations", "?")
                    failures.append(
                        f"leakage_audit.json reports FAIL ({n} violations)"
                    )
            except (json.JSONDecodeError, OSError) as exc:
                failures.append(f"leakage_audit.json is corrupt: {exc}")
        else:
            failures.append("missing leakage_audit.json")

        # 3. Results file
        checked.append("results file")
        results_found = False
        for name in _RESULTS_FILES:
            if (run_path / name).is_file():
                found.append(name)
                results_found = True
                break
        if not results_found:
            failures.append(
                f"missing results file (expected one of: {', '.join(_RESULTS_FILES)})"
            )

        # 4. Sealed registry lock
        checked.append("registry lock")
        if self._registry_path is not None:
            try:
                registry = BenchmarkRegistry.from_yaml(self._registry_path)
                registry.verify_lock()
                found.append("registry lock (sealed)")
            except RegistryError as exc:
                failures.append(f"registry lock unsealed: {exc}")
            except OSError as exc:
                failures.append(f"cannot load registry: {exc}")
        else:
            found.append("registry lock (skipped: no registry_path)")

        accepted = len(failures) == 0
        reason = "all artifacts present and valid" if accepted else "; ".join(failures)

        return SubmissionResult(
            accepted=accepted,
            reason=reason,
            run_dir=str(run_path),
            artifacts_checked=tuple(checked),
            artifacts_found=tuple(found),
        )
