"""Phase-1 MILP dev set: catalog, registry generation, sealed dev/validation splits.

This module is the single source of truth for the Phase-1 dev set used by the
closed-loop smoke + sanity experiment (plan task 21). It is intentionally
**small** (20-50 instances) and split into ``dev`` / ``validation`` only — no
``test`` / ``ood_test`` (those arrive in Wave 6, task 33).

Two instance sources feed the catalog:

* **synthetic** — deterministic set-cover / knapsack / facility generators
  (:mod:`opop.bench.sources.synthetic`). Fully offline; reproducible by seed.
* **miplib2017** — a checksum-verified curated subset of real MIPLIB 2017
  instances (:mod:`opop.bench.sources.miplib`). Best effort: materialised on
  demand from the network cache.

The catalog (:data:`PHASE1_CATALOG`) drives everything:

* :func:`build_registry_entries` groups recipes into one
  :class:`~opop.bench.registry.BenchmarkEntry` per family, computing a content
  checksum that locks the generated/downloaded instances.
* :func:`write_registry_yaml` serialises those entries to
  ``benchmarks/registry.yaml``.
* :func:`make_phase1_splits` loads that registry, validates it (dev/validation
  only), and seals ``split_manifest.lock`` (idempotent unless ``reseal=True``).
* :func:`get_phase1_instances` materialises the instances of a split as
  :class:`~opop.model.ir.MILP` objects.

The committed ``benchmarks/registry.yaml`` now combines these Phase-1 free splits
with the Wave-6 held-out suites (task 33); regenerate it + the lock with
``python -m opop.bench.sources.milp_suites --write --reseal`` (this module's
``--write`` delegates there). ``build_registry_entries`` itself stays Phase-1 only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources as _resources
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from opop.bench.registry import (
    FREE_SPLITS,
    HELD_SPLITS,
    BenchmarkEntry,
    BenchmarkRegistry,
    RegistryError,
)
from opop.bench.sources.miplib import (
    MIPLIB_PHASE1_SUBSET,
    instance_by_name,
    load_miplib_instance,
)
from opop.bench.sources.synthetic import (
    generate_facility,
    generate_knapsack,
    generate_set_cover,
    milp_digest,
)
from opop.model.ir import MILP

__all__ = [
    "PHASE1_CATALOG",
    "Phase1Error",
    "Recipe",
    "build_phase1_registry",
    "build_registry_entries",
    "entry_to_dict",
    "get_phase1_instances",
    "make_phase1_splits",
    "materialize",
    "write_registry_yaml",
]

SOURCE_SYNTHETIC = "synthetic"
SOURCE_MIPLIB = "miplib2017"

TIME_LIMIT_SEC = 30
BASELINE_SET = "scip_default"
PROBLEM_TYPE = "MILP"
PHASE = 1
THESIS = "T1"

# Deterministic split ordering for serialisation (frozenset iteration is not stable).
_SPLIT_ORDER = ("dev", "validation", "test", "ood_test")

_LICENSE = {
    SOURCE_SYNTHETIC: "MIT",
    SOURCE_MIPLIB: "MIPLIB2017-public",
}


def _package_data_path(name: str) -> Path:
    """Return a filesystem path to a packaged benchmark metadata file.

    When the package is installed as a wheel, the packaged file is extracted
    to a user-cache directory so downstream callers that need a ``Path`` can
    read it. The extracted copy is refreshed on every call to avoid stale
    reads after package upgrades.
    """
    ref = _resources.files("opop.bench.data") / name
    if isinstance(ref, Path):
        return ref
    cache = Path.home() / ".cache" / "opop" / "bench" / "data"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / name
    dest.write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


REGISTRY_PATH = _package_data_path("registry.yaml")
LOCK_PATH = _package_data_path("split_manifest.lock")

_DEV_FRACTION = 0.7

_REGISTRY_HEADER = """\
# OPOP benchmark registry -- Phase-1 MILP dev set (plan task 20).
#
# GENERATED FILE: produced from opop.bench.sources.phase1_set.PHASE1_CATALOG via
#   python -m opop.bench.sources.phase1_set --write --reseal
# Edit the catalog, not this file. The split_manifest.lock seals the
# instance->split assignment; re-sealing requires --reseal.
#
# Phase-1 declares dev/validation ONLY (test/ood_test arrive in Wave 6). A
# leakage_group may span dev and validation because both are free splits.
"""


class Phase1Error(Exception):
    """Raised on Phase-1 dev-set invariant violations (e.g. a held-out split)."""


@dataclass(frozen=True, slots=True)
class Recipe:
    """A single Phase-1 instance: how to (re)produce it and where it belongs.

    Attributes:
        id: Globally unique instance id (registry namespace).
        source: ``synthetic`` or ``miplib2017``.
        kind: Generator/loader kind (``set_cover``/``knapsack``/``facility``/``miplib``).
        params: Generator parameters (or ``{"name": ...}`` for MIPLIB).
        seed: RNG seed for synthetic generators (ignored for MIPLIB).
        split: ``dev`` or ``validation``.
        entry_name: Registry family entry this instance belongs to.
        leakage_group: Leakage group (one per family in Phase-1).
    """

    id: str
    source: str
    kind: str
    params: dict[str, int | float | str]
    seed: int
    split: str
    entry_name: str
    leakage_group: str


def _split_for(index: int, total: int) -> str:
    """Assign instance ``index`` of ``total`` to dev (first ~70%) or validation."""
    n_dev = round(_DEV_FRACTION * total)
    return "dev" if index < n_dev else "validation"


def _build_catalog() -> tuple[Recipe, ...]:
    """Construct the fixed Phase-1 catalog (deterministic, controllable sizes)."""
    recipes: list[Recipe] = []

    n_each = 10
    densities = (0.15, 0.17, 0.19, 0.21)
    for i in range(n_each):
        recipes.append(
            Recipe(
                id=f"synth_set_cover_{i:03d}",
                source=SOURCE_SYNTHETIC,
                kind="set_cover",
                params={
                    "n_rows": 8 + i,
                    "n_cols": 12 + i,
                    "density": densities[i % len(densities)],
                },
                seed=1000 + i,
                split=_split_for(i, n_each),
                entry_name="synthetic_set_cover_phase1",
                leakage_group="synthetic_set_cover_phase1",
            )
        )

    for i in range(n_each):
        recipes.append(
            Recipe(
                id=f"synth_knapsack_{i:03d}",
                source=SOURCE_SYNTHETIC,
                kind="knapsack",
                params={"n_items": 12 + 2 * i},
                seed=2000 + i,
                split=_split_for(i, n_each),
                entry_name="synthetic_knapsack_phase1",
                leakage_group="synthetic_knapsack_phase1",
            )
        )

    facility_sizes = (
        (6, 3), (7, 3), (8, 4), (6, 4), (7, 4),
        (8, 3), (9, 4), (7, 5), (8, 5), (6, 5),
    )
    for i, (n_customers, n_facilities) in enumerate(facility_sizes):
        recipes.append(
            Recipe(
                id=f"synth_facility_{i:03d}",
                source=SOURCE_SYNTHETIC,
                kind="facility",
                params={"n_customers": n_customers, "n_facilities": n_facilities},
                seed=3000 + i,
                split=_split_for(i, len(facility_sizes)),
                entry_name="synthetic_facility_phase1",
                leakage_group="synthetic_facility_phase1",
            )
        )

    n_miplib = len(MIPLIB_PHASE1_SUBSET)
    for i, inst in enumerate(MIPLIB_PHASE1_SUBSET):
        recipes.append(
            Recipe(
                id=inst.name,
                source=SOURCE_MIPLIB,
                kind="miplib",
                params={"name": inst.name},
                seed=0,
                split=_split_for(i, n_miplib),
                entry_name="miplib2017_phase1",
                leakage_group="miplib2017_phase1",
            )
        )

    return tuple(recipes)


PHASE1_CATALOG: tuple[Recipe, ...] = _build_catalog()
_CATALOG_BY_ID: dict[str, Recipe] = {r.id: r for r in PHASE1_CATALOG}


def materialize(
    recipe: Recipe,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> MILP:
    """Materialise a recipe into a :class:`MILP` (generate synthetic / load MIPLIB)."""
    params = recipe.params
    if recipe.kind == "set_cover":
        return generate_set_cover(
            int(params["n_rows"]), int(params["n_cols"]), float(params["density"]), recipe.seed
        )
    if recipe.kind == "knapsack":
        return generate_knapsack(int(params["n_items"]), recipe.seed)
    if recipe.kind == "facility":
        return generate_facility(
            int(params["n_customers"]), int(params["n_facilities"]), recipe.seed
        )
    if recipe.kind == "miplib":
        return load_miplib_instance(
            str(params["name"]), cache_dir=cache_dir, allow_download=allow_download
        )
    raise Phase1Error(f"unknown recipe kind {recipe.kind!r} for instance {recipe.id!r}")


def _content_digest(recipe: Recipe) -> str:
    """Per-instance content digest: generated-model hash (synthetic) / file sha256 (MIPLIB)."""
    if recipe.source == SOURCE_MIPLIB:
        return instance_by_name(str(recipe.params["name"])).sha256
    return milp_digest(materialize(recipe))


def _family_checksum(recipes: Iterable[Recipe]) -> str:
    """``sha256:`` over the sorted ``id=content_digest`` manifest of a family."""
    items = sorted((r.id, _content_digest(r)) for r in recipes)
    manifest = "\n".join(f"{rid}={digest}" for rid, digest in items)
    return "sha256:" + hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def build_registry_entries() -> list[BenchmarkEntry]:
    """Group the catalog into one :class:`BenchmarkEntry` per family (with checksums)."""
    grouped: dict[str, list[Recipe]] = {}
    order: list[str] = []
    for recipe in PHASE1_CATALOG:
        if recipe.entry_name not in grouped:
            grouped[recipe.entry_name] = []
            order.append(recipe.entry_name)
        grouped[recipe.entry_name].append(recipe)

    entries: list[BenchmarkEntry] = []
    for entry_name in order:
        recipes = grouped[entry_name]
        split: dict[str, list[str]] = {"dev": [], "validation": []}
        for recipe in recipes:
            split[recipe.split].append(recipe.id)
        head = recipes[0]
        entries.append(
            BenchmarkEntry(
                name=entry_name,
                problem_type=PROBLEM_TYPE,
                source=head.source,
                split={name: tuple(ids) for name, ids in split.items() if ids},
                license=_LICENSE[head.source],
                instance_count=len(recipes),
                time_limit_sec=float(TIME_LIMIT_SEC),
                baseline_set=BASELINE_SET,
                leakage_group=head.leakage_group,
                checksum=_family_checksum(recipes),
                phase=PHASE,
                thesis=THESIS,
            )
        )
    return entries


def build_phase1_registry(
    *,
    lock_path: str | Path = LOCK_PATH,
    source_path: str | Path = REGISTRY_PATH,
) -> BenchmarkRegistry:
    """Build an in-memory :class:`BenchmarkRegistry` straight from the catalog."""
    return BenchmarkRegistry(
        build_registry_entries(),
        lock_path=Path(lock_path),
        source_path=Path(source_path),
    )


def entry_to_dict(entry: BenchmarkEntry) -> dict[str, object]:
    """Serialise one entry to an ordered, YAML-friendly mapping."""
    time_limit: int | float = (
        int(entry.time_limit_sec)
        if float(entry.time_limit_sec).is_integer()
        else entry.time_limit_sec
    )
    return {
        "name": entry.name,
        "problem_type": entry.problem_type,
        "source": entry.source,
        "split": {
            name: list(entry.split[name]) for name in _SPLIT_ORDER if name in entry.split
        },
        "license": entry.license,
        "instance_count": entry.instance_count,
        "time_limit_sec": time_limit,
        "baseline_set": entry.baseline_set,
        "leakage_group": entry.leakage_group,
        "checksum": entry.checksum,
        "phase": entry.phase,
        "thesis": entry.thesis,
    }


def write_registry_yaml(path: str | Path = REGISTRY_PATH) -> Path:
    """Write ``benchmarks/registry.yaml`` from the catalog; return the path."""
    data = {"benchmarks": [entry_to_dict(entry) for entry in build_registry_entries()]}
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, indent=2)
    out = Path(path)
    out.write_text(_REGISTRY_HEADER + body, encoding="utf-8")
    return out


def _assert_phase1_only(registry: BenchmarkRegistry) -> None:
    """Reject any Phase-1 entry that assigns instances to held-out (test/ood) splits.

    The committed registry now combines Phase-1 free splits with the Wave-6 held-out
    suites (task 33); those carry a non-Phase-1 ``phase`` tag and are exempt. A
    ``phase == PHASE`` (Phase-1) entry must still be dev/validation only.
    """
    for entry in registry.entries:
        if entry.phase != PHASE:
            continue
        for held in sorted(HELD_SPLITS):
            if entry.split.get(held):
                raise Phase1Error(
                    f"{entry.name}: Phase-1 forbids a {held!r} assignment "
                    + "(test/ood_test arrive in Wave 6)"
                )


def make_phase1_splits(
    *,
    reseal: bool = False,
    registry_path: str | Path = REGISTRY_PATH,
    lock_path: str | Path | None = None,
) -> BenchmarkRegistry:
    """Load the Phase-1 registry, validate (dev/validation only), and seal the lock.

    Idempotent: a re-run does not change ``split_manifest.lock`` unless the
    instance->split assignment changed and ``reseal=True`` is given.

    Raises:
        Phase1Error: If any entry assigns instances to ``test``/``ood_test``.
        RegistryError: On schema/overlap/leakage/lock failures.
    """
    registry = BenchmarkRegistry.from_yaml(registry_path, lock_path=lock_path)
    registry.validate()
    _assert_phase1_only(registry)
    if registry.lock_path is None:
        raise RegistryError("no lock_path configured for Phase-1 splits")
    if registry.lock_path.exists():
        registry.verify_lock(reseal=reseal)
    else:
        registry.seal()
    return registry


def get_phase1_instances(
    split: str,
    *,
    sources: tuple[str, ...] | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    registry_path: str | Path = REGISTRY_PATH,
) -> list[MILP]:
    """Materialise the instances assigned to ``split`` as :class:`MILP` objects.

    Args:
        split: ``dev`` or ``validation`` (Phase-1 exposes no held-out splits).
        sources: Optional source filter (e.g. ``("synthetic",)`` to stay offline).
        cache_dir: MIPLIB cache directory (defaults to the package cache).
        allow_download: Allow on-demand MIPLIB downloads when not cached.
        registry_path: Registry YAML to read the split assignment from.

    Raises:
        Phase1Error: If ``split`` is not a free split or an id lacks a recipe.
    """
    if split not in FREE_SPLITS:
        raise Phase1Error(
            f"Phase-1 exposes only {sorted(FREE_SPLITS)}; got {split!r}"
        )
    registry = BenchmarkRegistry.from_yaml(registry_path)
    instances: list[MILP] = []
    for _bench, instance_id in registry.get_split(split):
        recipe = _CATALOG_BY_ID.get(instance_id)
        if recipe is None:
            raise Phase1Error(
                f"registry instance {instance_id!r} in split {split!r} has no catalog recipe"
            )
        if sources is not None and recipe.source not in sources:
            continue
        instances.append(
            materialize(recipe, cache_dir=cache_dir, allow_download=allow_download)
        )
    return instances


def main(argv: list[str] | None = None) -> int:
    """CLI: regenerate the registry, (optionally) fetch MIPLIB, and seal the lock."""
    parser = argparse.ArgumentParser(
        description="Generate + seal the Phase-1 MILP dev set registry."
    )
    parser.add_argument(
        "--write", action="store_true", help="(re)write benchmarks/registry.yaml from the catalog"
    )
    parser.add_argument(
        "--reseal", action="store_true", help="reseal split_manifest.lock on assignment change"
    )
    parser.add_argument(
        "--download-miplib",
        action="store_true",
        help="best-effort fetch the curated MIPLIB subset into the cache",
    )
    args = parser.parse_args(argv)

    if args.download_miplib:
        from opop.bench.sources.miplib import download_miplib_subset

        paths = download_miplib_subset()
        print(f"miplib: cached {len(paths)}/{len(MIPLIB_PHASE1_SUBSET)} instance(s)")

    if args.write:
        import importlib

        milp_suites = importlib.import_module("opop.bench.sources.milp_suites")
        milp_suites.write_registry_yaml()
        print(f"wrote {REGISTRY_PATH}")

    try:
        registry = make_phase1_splits(reseal=args.reseal or args.write)
    except RegistryError as exc:
        print(f"phase-1 registry error: {exc}", file=sys.stderr)
        return 1

    dev = registry.get_split("dev")
    validation = registry.get_split("validation")
    print(
        f"phase-1 ok: {len(registry.entries)} families, "
        + f"dev={len(dev)} validation={len(validation)} (no test/ood)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
