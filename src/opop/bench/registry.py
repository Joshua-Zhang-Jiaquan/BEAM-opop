"""Benchmark registry with immutable splits and leakage-aware validation.

This module is the scientific-integrity backbone for OPOP experiments. It defines
the schema for ``benchmarks/registry.yaml``, enforces that every benchmark entry
carries a ``phase`` and ``thesis`` tag, and guards against data leakage by

* refusing any instance assignment that places the same instance in more than one
  split;
* refusing any ``leakage_group`` that spans free splits (``dev``/``validation``)
  and held-out splits (``test``/``ood_test``);
* sealing the instance→split assignment in ``split_manifest.lock`` and refusing
  to load the registry if the lock changes unless ``--reseal`` is given.

Test/ood_test splits are only loadable when ``one_shot_final=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FREE_SPLITS = frozenset({"dev", "validation"})
HELD_SPLITS = frozenset({"test", "ood_test"})
SPLITS = FREE_SPLITS | HELD_SPLITS

_REQUIRED_FIELDS = frozenset({
    "name",
    "problem_type",
    "source",
    "split",
    "license",
    "instance_count",
    "time_limit_sec",
    "baseline_set",
    "leakage_group",
    "checksum",
    "phase",
    "thesis",
})


class RegistryError(Exception):
    """Base class for registry failures."""


class SchemaError(RegistryError):
    """Raised when a registry entry violates the schema."""


class LeakageError(RegistryError):
    """Raised when a split assignment creates a leakage path."""


class LockMismatchError(RegistryError):
    """Raised when ``split_manifest.lock`` does not match the current registry."""


class FinalModeRequiredError(RegistryError):
    """Raised when a held-out split is requested without ``one_shot_final=True``."""


@dataclass(frozen=True)
class BenchmarkEntry:
    """One benchmark family in the registry."""

    name: str
    problem_type: str
    source: str
    split: Mapping[str, tuple[str, ...]]
    license: str
    instance_count: int
    time_limit_sec: float
    baseline_set: str
    leakage_group: str
    checksum: str
    phase: int | str
    thesis: str


class BenchmarkRegistry:
    """In-memory registry loaded from YAML with split integrity checks."""

    entries: tuple[BenchmarkEntry, ...]
    lock_path: Path | None
    source_path: Path | None

    def __init__(
        self,
        entries: Sequence[BenchmarkEntry],
        lock_path: Path | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.entries = tuple(entries)
        self.lock_path = lock_path
        self.source_path = source_path
        for entry in self.entries:
            self._validate_entry(entry)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        lock_path: str | Path | None = None,
    ) -> "BenchmarkRegistry":
        """Load a registry from YAML.

        If ``lock_path`` is not provided, it defaults to
        ``<registry_dir>/split_manifest.lock``.
        """
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SchemaError(f"cannot read registry: {exc}") from exc

        if not isinstance(data, dict):
            raise SchemaError("registry root must be a mapping")

        raw_entries = data.get("benchmarks")
        if not isinstance(raw_entries, list):
            raise SchemaError("'benchmarks' must be a list")
        raw_entries_list: list[Any] = raw_entries

        entries: list[BenchmarkEntry] = []
        for idx, raw in enumerate(raw_entries_list):
            entries.append(_build_entry(raw, idx))

        if lock_path is None:
            lock_path = path.parent / "split_manifest.lock"

        return cls(entries, lock_path=Path(lock_path), source_path=path)

    def register(self, entry: BenchmarkEntry) -> BenchmarkEntry:
        """Add a validated entry to the in-memory registry."""
        self._validate_entry(entry)
        self.entries = self.entries + (entry,)
        return entry

    @staticmethod
    def _validate_entry(entry: BenchmarkEntry) -> None:
        """Validate a single entry's internal consistency."""
        seen_splits = set()
        total_instances = 0
        seen_ids: set[str] = set()

        for split_name, ids in entry.split.items():
            if split_name not in SPLITS:
                raise SchemaError(
                    f"{entry.name}: invalid split key {split_name!r}; "
                    + f"must be one of {sorted(SPLITS)}"
                )
            if split_name in seen_splits:
                raise SchemaError(f"{entry.name}: duplicate split key {split_name!r}")
            seen_splits.add(split_name)

            total_instances += len(ids)

            for inst in ids:
                if inst in seen_ids:
                    raise SchemaError(
                        f"{entry.name}: duplicate instance id {inst!r} within the same entry"
                    )
                seen_ids.add(inst)

        if total_instances != entry.instance_count:
            raise SchemaError(
                f"{entry.name}: instance_count {entry.instance_count} does not match "
                + f"{total_instances} ids declared across splits"
            )

    def validate(self) -> None:
        """Run all schema-level checks on the loaded registry."""
        for entry in self.entries:
            self._validate_entry(entry)
        self.assert_no_overlap()

    def assert_no_overlap(self) -> None:
        """Check that no instance appears in more than one split.

        Also checks that no ``leakage_group`` spans free splits and held-out
        splits.
        """
        instance_location: dict[str, tuple[str, str]] = {}

        for entry in self.entries:
            for split_name, ids in entry.split.items():
                for inst in ids:
                    if inst in instance_location:
                        other_bench, other_split = instance_location[inst]
                        raise LeakageError(
                            f"instance {inst!r} appears in both "
                            + f"{other_bench}/{other_split} and "
                            + f"{entry.name}/{split_name}"
                        )
                    instance_location[inst] = (entry.name, split_name)

        group_splits: dict[str, set[str]] = {}
        for entry in self.entries:
            present_splits = {
                split_name for split_name, ids in entry.split.items() if ids
            }
            group_splits.setdefault(entry.leakage_group, set()).update(present_splits)

        for group, splits in group_splits.items():
            if (splits & FREE_SPLITS) and (splits & HELD_SPLITS):
                raise LeakageError(
                    f"leakage_group {group!r} spans free and held-out splits: "
                    + f"{sorted(splits)}"
                )

    def get_split(
        self,
        split: str,
        one_shot_final: bool = False,
    ) -> list[tuple[str, str]]:
        """Return ``(benchmark_name, instance_id)`` pairs for ``split``.

        ``test`` and ``ood_test`` require ``one_shot_final=True``.
        """
        if split not in SPLITS:
            raise RegistryError(
                f"unknown split {split!r}; must be one of {sorted(SPLITS)}"
            )

        if split in HELD_SPLITS and not one_shot_final:
            raise FinalModeRequiredError(
                f"split {split!r} is held-out and requires one_shot_final=True"
            )

        return [
            (entry.name, inst)
            for entry in self.entries
            for inst in entry.split.get(split, ())
        ]

    def _split_manifest(self) -> dict[str, Any]:
        """Canonical instance→split assignment used for lock hashing."""
        assignment: dict[str, str] = {}
        for entry in self.entries:
            for split_name, ids in entry.split.items():
                for inst in ids:
                    assignment[f"{entry.name}::{inst}"] = split_name
        return {"format": "instance-to-split", "assignment": assignment}

    def compute_lock_hash(self) -> str:
        """Compute the SHA-256 hash of the current split assignment."""
        canonical = json.dumps(
            self._split_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        """Write ``split_manifest.lock`` for the current split assignment."""
        if self.lock_path is None:
            raise RegistryError("no lock_path configured")
        lock_data = {"hash": self.compute_lock_hash(), "algorithm": "sha256"}
        self.lock_path.write_text(
            json.dumps(lock_data, indent=2) + "\n",
            encoding="utf-8",
        )

    def verify_lock(self, reseal: bool = False) -> None:
        """Verify that ``split_manifest.lock`` matches the current registry."""
        if self.lock_path is None:
            raise RegistryError("no lock_path configured")

        if not self.lock_path.exists():
            if reseal:
                self.seal()
                return
            raise LockMismatchError(f"split lock not found: {self.lock_path}")

        try:
            lock_data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LockMismatchError(f"cannot parse lock: {exc}") from exc

        expected = lock_data.get("hash")
        if expected is None:
            raise LockMismatchError("lock file missing 'hash' field")

        actual = self.compute_lock_hash()
        if expected != actual:
            if not reseal:
                raise LockMismatchError(
                    f"split lock mismatch: lock={expected!r} computed={actual!r}; "
                    + "use --reseal to regenerate"
                )
            self.seal()


def _build_entry(raw: Any, idx: int) -> BenchmarkEntry:
    """Construct and schema-check a ``BenchmarkEntry`` from raw YAML data."""
    if not isinstance(raw, dict):
        raise SchemaError(f"entry {idx}: must be a mapping")

    raw_dict: dict[str, Any] = raw
    name = raw_dict.get("name", f"<entry {idx}>")

    missing = _REQUIRED_FIELDS - set(raw_dict.keys())
    if missing:
        raise SchemaError(f"{name}: missing required fields {sorted(missing)}")

    raw_split = raw_dict["split"]
    if not isinstance(raw_split, dict):
        raise SchemaError(f"{name}: 'split' must be a mapping")

    split: dict[str, tuple[str, ...]] = {}
    for split_name, ids in raw_split.items():
        if split_name not in SPLITS:
            raise SchemaError(
                f"{name}: invalid split key {split_name!r}; "
                + f"must be one of {sorted(SPLITS)}"
            )
        if not isinstance(ids, list):
            raise SchemaError(f"{name}: split {split_name!r} must contain a list")
        ids_list: list[Any] = ids
        id_strs: list[str] = []
        for i in ids_list:
            id_strs.append(str(i))
        split[split_name] = tuple(id_strs)

    try:
        return BenchmarkEntry(
            name=str(name),
            problem_type=str(raw_dict["problem_type"]),
            source=str(raw_dict["source"]),
            split=split,
            license=str(raw_dict["license"]),
            instance_count=int(raw_dict["instance_count"]),
            time_limit_sec=float(raw_dict["time_limit_sec"]),
            baseline_set=str(raw_dict["baseline_set"]),
            leakage_group=str(raw_dict["leakage_group"]),
            checksum=str(raw_dict["checksum"]),
            phase=raw_dict["phase"],
            thesis=str(raw_dict["thesis"]),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name}: field type error: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for registry validation and sealing."""
    parser = argparse.ArgumentParser(
        description="Validate benchmark registry and split integrity."
    )
    parser.add_argument(
        "--validate",
        metavar="REGISTRY",
        help="validate a registry YAML file",
    )
    parser.add_argument(
        "--reseal",
        action="store_true",
        help="regenerate split_manifest.lock instead of failing on mismatch",
    )
    parser.add_argument(
        "--lock",
        metavar="LOCK",
        help="path to split_manifest.lock (default: next to REGISTRY)",
    )
    args = parser.parse_args(argv)

    if args.validate:
        try:
            registry = BenchmarkRegistry.from_yaml(
                args.validate,
                lock_path=args.lock,
            )
            registry.validate()
            registry.verify_lock(reseal=args.reseal)
            print(f"registry valid: {len(registry.entries)} benchmark(s)")
            return 0
        except RegistryError as exc:
            print(f"registry invalid: {exc}", file=sys.stderr)
            return 1

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
