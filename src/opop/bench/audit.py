"""Leakage audit: cross-reference a run's tuned instances against held-out splits.

Scientific-integrity rule (Metis leakage policy): NO ``test`` / ``ood_test``
instance may ever enter a tuning / proposal path. Every ``events.jsonl`` row tags
the instance being tuned via its ``instance_id`` (written by
:func:`opop.orchestrator.loop.run_loop`); this module collects those ids, loads
the registry's held-out split manifest, and fails loudly if any held-out instance
id appears in the journal.

The CLI is exposed as ``python -m opop.bench.audit_leakage`` (a thin entry-point
shim re-exports :func:`main`); ``python -m opop.bench.audit`` works too::

    python -m opop.bench.audit_leakage --run <run_dir> --registry benchmarks/registry.yaml

It emits ``<run_dir>/leakage_audit.json`` (override with ``--out``)::

    {
      "status": "pass" | "fail",
      "test_instances_used_for_tuning": [ids...],
      "ood_instances_used_for_tuning": [ids...],
      "n_violations": int
    }

Exit codes: ``0`` = pass (no leakage), ``1`` = fail (leakage found), ``2`` =
file / argument error. The audit is allowed to inspect held-out splits, so it
loads them with ``one_shot_final=True``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from opop.bench.registry import BenchmarkRegistry, RegistryError

__all__ = [
    "EVENTS_FILENAME",
    "LEAKAGE_AUDIT_FILENAME",
    "AuditError",
    "audit_leakage",
    "main",
]

#: Journal filename the audit reads (relative to the run directory).
EVENTS_FILENAME: str = "events.jsonl"
#: Default report filename the audit writes (relative to the run directory).
LEAKAGE_AUDIT_FILENAME: str = "leakage_audit.json"


class AuditError(Exception):
    """Raised for unrecoverable audit IO / input errors (mapped to exit code 2)."""


def _held_out_ids(registry: BenchmarkRegistry) -> tuple[set[str], set[str]]:
    """Return ``(test_ids, ood_test_ids)`` from the registry's held-out splits.

    Held-out splits require ``one_shot_final=True``; the audit is explicitly
    permitted to inspect them (it is verifying they were NOT used for tuning).
    """
    test_ids = {inst for _bench, inst in registry.get_split("test", one_shot_final=True)}
    ood_ids = {inst for _bench, inst in registry.get_split("ood_test", one_shot_final=True)}
    return test_ids, ood_ids


def _tuned_instance_ids(events_path: Path) -> list[str]:
    """Collect every non-empty ``instance_id`` from a run's ``events.jsonl``.

    Each non-blank line is one JSON object. A malformed line is a hard
    :class:`AuditError` — a corrupt journal must never silently pass the audit.
    """
    try:
        text = events_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read events journal: {exc}") from exc

    ids: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{events_path}:{lineno}: malformed JSON: {exc}") from exc
        if not isinstance(record, dict):
            continue
        inst = record.get("instance_id")
        if isinstance(inst, str) and inst:
            ids.append(inst)
    return ids


def audit_leakage(run_dir: str | Path, registry_path: str | Path) -> dict[str, Any]:
    """Audit one run for held-out-instance leakage; return the report dict.

    Loads the registry's held-out (``test`` + ``ood_test``) split manifest and
    the run's ``events.jsonl``, then reports any instance id used for tuning that
    belongs to a held-out split.

    Raises:
        AuditError: if the journal is absent / corrupt or the registry cannot be
            loaded.
    """
    run_path = Path(run_dir)
    events_path = run_path / EVENTS_FILENAME
    if not events_path.is_file():
        raise AuditError(f"events journal not found: {events_path}")

    try:
        registry = BenchmarkRegistry.from_yaml(registry_path)
    except (RegistryError, OSError) as exc:
        raise AuditError(f"cannot load registry {registry_path!r}: {exc}") from exc

    test_ids, ood_ids = _held_out_ids(registry)
    tuned = set(_tuned_instance_ids(events_path))

    test_hits = sorted(tuned & test_ids)
    ood_hits = sorted(tuned & ood_ids)
    n_violations = len(test_hits) + len(ood_hits)

    return {
        "status": "fail" if n_violations else "pass",
        "test_instances_used_for_tuning": test_hits,
        "ood_instances_used_for_tuning": ood_hits,
        "n_violations": n_violations,
    }


def _write_report(report: dict[str, Any], out_path: Path) -> None:
    """Persist the audit report as deterministic, sorted JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns ``0`` (pass), ``1`` (leakage), or ``2`` (error)."""
    parser = argparse.ArgumentParser(
        prog="python -m opop.bench.audit_leakage",
        description="Audit a run's events.jsonl for held-out (test/ood_test) instance leakage.",
    )
    parser.add_argument("--run", required=True, metavar="DIR", help="run output directory")
    parser.add_argument(
        "--registry", required=True, metavar="REGISTRY", help="path to registry.yaml"
    )
    parser.add_argument(
        "--out",
        metavar="OUT",
        default=None,
        help="output path for leakage_audit.json (default: <run>/leakage_audit.json)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    out_path = Path(args.out) if args.out else run_dir / LEAKAGE_AUDIT_FILENAME

    try:
        report = audit_leakage(run_dir, args.registry)
    except AuditError as exc:
        print(f"leakage audit error: {exc}", file=sys.stderr)
        return 2

    _write_report(report, out_path)

    if report["status"] == "fail":
        msg = (
            f"LEAKAGE DETECTED: {report['n_violations']} held-out instance(s) used for tuning "
            + f"(test={report['test_instances_used_for_tuning']}, "
            + f"ood_test={report['ood_instances_used_for_tuning']}); wrote {out_path}"
        )
        print(msg, file=sys.stderr)
        return 1

    print(f"leakage audit pass: 0 held-out instances used for tuning; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
