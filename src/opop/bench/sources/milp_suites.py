"""Wave-6 MILP benchmark suites: combined registry assembly + sealing (task 33).

This module is the single canonical generator for the committed
``benchmarks/registry.yaml``. It concatenates the Phase-1 free-split families
(:func:`opop.bench.sources.phase1_set.build_registry_entries`) with the Wave-6
held-out suites:

* **MIPLIB 2017** held-out collection — :func:`opop.bench.sources.miplib.build_heldout_entries`.
* **Distributional MIPLIB** — :func:`opop.bench.sources.distributional.build_entries`.
* **MILPBench** — :func:`opop.bench.sources.milpbench.build_entries`.

The held-out suites populate the immutable ``test`` / ``ood_test`` splits; every
suite family is its own ``leakage_group`` (grouped by instance family / domain /
generator) and is name-disjoint from the free splits, so no group spans a free
and a held-out split. :func:`write_registry_yaml` serialises the combined entry
list deterministically (reusing the Phase-1 serializer); :func:`seal_splits`
validates the registry and seals ``split_manifest.lock``.

Regenerate the committed registry + lock from code with::

    python -m opop.bench.sources.milp_suites --write --reseal [--download]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from opop.bench.classic import build_classic_entries
from opop.bench.registry import BenchmarkEntry, BenchmarkRegistry, RegistryError
from opop.bench.sources import distributional, milpbench, modeling_agents, qplib
from opop.bench.sources.miplib import build_heldout_entries
from opop.bench.sources.phase1_set import (
    LOCK_PATH,
    REGISTRY_PATH,
    build_registry_entries,
    entry_to_dict,
)

__all__ = [
    "LOCK_PATH",
    "REGISTRY_PATH",
    "build_all_entries",
    "build_combined_registry",
    "build_suite_entries",
    "seal_splits",
    "write_registry_yaml",
]

_REGISTRY_HEADER = """\
# OPOP benchmark registry -- Phase-1 dev set + Wave-6 held-out MILP suites.
#
# GENERATED FILE: produced from the catalogs via
#   python -m opop.bench.sources.milp_suites --write --reseal
# Edit the source catalogs (phase1_set / miplib / distributional / milpbench),
# not this file. split_manifest.lock seals the instance->split assignment;
# re-sealing requires --reseal.
#
# Free splits (dev/validation) come from the Phase-1 families. Held-out splits
# (test/ood_test) come from MIPLIB 2017, Distributional MIPLIB, MILPBench, the
# classic CO families (TSP/CVRP/OR-Library/JSP/MaxSAT/MaxCut), the QPLIB MIQP/MIQCP
# fixtures, and the cleaned modeling-agent datasets (NL4Opt/OptiBench); all phase 6,
# thesis T3. Each held-out family is its own leakage_group, name-disjoint from the
# free splits, so no leakage_group spans a free and a held-out split.
"""


def build_suite_entries() -> list[BenchmarkEntry]:
    """Return the Wave-6 held-out suite entries (MIPLIB 2017 + D-MIPLIB + MILPBench)."""
    return [
        *build_heldout_entries(),
        *distributional.build_entries(),
        *milpbench.build_entries(),
    ]


def build_all_entries() -> list[BenchmarkEntry]:
    """Return the combined entry list: Phase-1 free families + Wave-6 held-out suites.

    Held-out suites are the three MILP libraries (:func:`build_suite_entries`), the
    six classic-CO families (:func:`opop.bench.classic.build_classic_entries`, plan
    task 34), and the task-35 generality families: the QPLIB MIQP/MIQCP fixtures
    (:func:`opop.bench.sources.qplib.build_entries`) and the cleaned modeling-agent
    datasets (:func:`opop.bench.sources.modeling_agents.build_entries`). All are
    ``phase=6`` / ``thesis=T3`` and live in the ``test`` split, so they extend the
    held-out collection without touching the free splits.
    """
    return [
        *build_registry_entries(),
        *build_suite_entries(),
        *build_classic_entries(),
        *qplib.build_entries(),
        *modeling_agents.build_entries(),
    ]


def build_combined_registry(
    *,
    lock_path: str | Path = LOCK_PATH,
    source_path: str | Path = REGISTRY_PATH,
) -> BenchmarkRegistry:
    """Build an in-memory combined :class:`BenchmarkRegistry` from the catalogs."""
    return BenchmarkRegistry(
        build_all_entries(),
        lock_path=Path(lock_path),
        source_path=Path(source_path),
    )


def write_registry_yaml(path: str | Path = REGISTRY_PATH) -> Path:
    """Write the combined ``benchmarks/registry.yaml`` from the catalogs; return the path."""
    data = {"benchmarks": [entry_to_dict(entry) for entry in build_all_entries()]}
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, indent=2)
    out = Path(path)
    out.write_text(_REGISTRY_HEADER + body, encoding="utf-8")
    return out


def seal_splits(
    *,
    reseal: bool = False,
    registry_path: str | Path = REGISTRY_PATH,
    lock_path: str | Path | None = None,
) -> BenchmarkRegistry:
    """Load the combined registry, validate it, and seal ``split_manifest.lock``.

    Idempotent: a re-run does not change the lock unless the instance->split
    assignment changed and ``reseal=True`` is given.

    Raises:
        RegistryError: On schema/overlap/leakage/lock failures.
    """
    registry = BenchmarkRegistry.from_yaml(registry_path, lock_path=lock_path)
    registry.validate()
    if registry.lock_path is None:
        raise RegistryError("no lock_path configured for combined splits")
    if registry.lock_path.exists():
        registry.verify_lock(reseal=reseal)
    else:
        registry.seal()
    return registry


def _download_suites() -> None:
    """Best-effort fetch of the verifiable held-out subsets into their caches."""
    dpaths = distributional.download_subset()
    print(f"dmiplib: cached {len(dpaths)}/{len(distributional.DMIPLIB_SUBSET)} distribution(s)")
    mpaths = milpbench.download_subset()
    print(f"milpbench: cached {len(mpaths)}/{len(milpbench.MILPBENCH_SUBSET)} instance(s)")


def main(argv: list[str] | None = None) -> int:
    """CLI: regenerate the combined registry, (optionally) fetch suites, and seal the lock."""
    parser = argparse.ArgumentParser(
        description="Generate + seal the combined Phase-1 + Wave-6 MILP benchmark registry."
    )
    parser.add_argument(
        "--write", action="store_true", help="(re)write benchmarks/registry.yaml from the catalogs"
    )
    parser.add_argument(
        "--reseal", action="store_true", help="reseal split_manifest.lock on assignment change"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="best-effort fetch the verifiable held-out subsets into the cache",
    )
    args = parser.parse_args(argv)

    if args.download:
        _download_suites()

    if args.write:
        write_registry_yaml()
        print(f"wrote {REGISTRY_PATH}")

    try:
        registry = seal_splits(reseal=args.reseal or args.write)
    except RegistryError as exc:
        print(f"combined registry error: {exc}", file=sys.stderr)
        return 1

    dev = registry.get_split("dev")
    validation = registry.get_split("validation")
    test = registry.get_split("test", one_shot_final=True)
    ood = registry.get_split("ood_test", one_shot_final=True)
    print(
        f"registry ok: {len(registry.entries)} families, "
        + f"dev={len(dev)} validation={len(validation)} test={len(test)} ood_test={len(ood)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
