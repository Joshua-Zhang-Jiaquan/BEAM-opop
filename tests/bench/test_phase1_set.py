"""Tests for the Phase-1 MILP dev set (plan task 20).

Covers: synthetic generator determinism + structure, committed-registry
validation, ``make_phase1_splits`` sealing + idempotency, the dev/validation-only
(no test/ood) invariant, ``get_phase1_instances`` materialisation, content
checksum integrity, and catalog<->registry.yaml consistency. MIPLIB download /
load / checksum paths are ``integration`` tests gated on network + SCIP.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from opop.bench.registry import (
    SPLITS,
    BenchmarkRegistry,
    LockMismatchError,
)
from opop.bench.sources.miplib import (
    MIPLIB_PHASE1_SUBSET,
    MiplibChecksumError,
    MiplibInstance,
    default_cache_dir,
    download_miplib_instance,
    load_miplib_instance,
    network_available,
    subset_manifest_checksum,
    verify_checksum,
)
from opop.bench.sources.phase1_set import (
    PHASE1_CATALOG,
    REGISTRY_PATH,
    Phase1Error,
    build_registry_entries,
    get_phase1_instances,
    make_phase1_splits,
    materialize,
)
from opop.bench.sources.synthetic import (
    canonical_milp_repr,
    generate_facility,
    generate_knapsack,
    generate_set_cover,
    milp_digest,
)
from opop.model.ir import MILP, ConstraintSense, ObjSense, VarType


def _miplib_available() -> bool:
    """True if every curated MIPLIB file is cached+valid, or the mirror is reachable."""
    cache = default_cache_dir()
    cached = all(
        (cache / inst.filename).exists() and verify_checksum(cache / inst.filename, inst.sha256)
        for inst in MIPLIB_PHASE1_SUBSET
    )
    return cached or network_available()


# ---------------------------------------------------------------------------
# Synthetic generators: determinism
# ---------------------------------------------------------------------------
class TestSyntheticDeterminism:
    def test_set_cover_deterministic_by_seed(self) -> None:
        a = generate_set_cover(12, 18, 0.2, seed=7)
        b = generate_set_cover(12, 18, 0.2, seed=7)
        assert milp_digest(a) == milp_digest(b)
        assert canonical_milp_repr(a) == canonical_milp_repr(b)

    def test_knapsack_deterministic_by_seed(self) -> None:
        assert milp_digest(generate_knapsack(20, seed=3)) == milp_digest(
            generate_knapsack(20, seed=3)
        )

    def test_facility_deterministic_by_seed(self) -> None:
        assert milp_digest(generate_facility(8, 4, seed=5)) == milp_digest(
            generate_facility(8, 4, seed=5)
        )

    def test_different_seed_changes_instance(self) -> None:
        assert milp_digest(generate_set_cover(12, 18, 0.2, seed=1)) != milp_digest(
            generate_set_cover(12, 18, 0.2, seed=2)
        )


# ---------------------------------------------------------------------------
# Synthetic generators: structure + validation
# ---------------------------------------------------------------------------
class TestSyntheticStructure:
    def test_set_cover_structure(self) -> None:
        milp = generate_set_cover(10, 15, 0.25, seed=0)
        assert milp.n_vars == 15
        assert milp.n_constraints == 10
        assert milp.objective.sense is ObjSense.MINIMIZE
        assert all(c.sense is ConstraintSense.GE for c in milp.constraints)
        assert all(v.vtype is VarType.BINARY for v in milp.variables)

    def test_knapsack_structure(self) -> None:
        milp = generate_knapsack(16, seed=0)
        assert milp.n_vars == 16
        assert milp.n_constraints == 1
        assert milp.constraints[0].sense is ConstraintSense.LE
        assert milp.objective.sense is ObjSense.MAXIMIZE

    def test_facility_structure(self) -> None:
        milp = generate_facility(6, 3, seed=0)
        assert milp.n_vars == 3 + 6 * 3
        assert milp.n_constraints == 6 + 6 * 3
        eq_rows = [c for c in milp.constraints if c.sense is ConstraintSense.EQ]
        le_rows = [c for c in milp.constraints if c.sense is ConstraintSense.LE]
        assert len(eq_rows) == 6
        assert len(le_rows) == 6 * 3
        assert milp.objective.sense is ObjSense.MINIMIZE

    def test_set_cover_rows_are_all_covered(self) -> None:
        milp = generate_set_cover(20, 8, 0.05, seed=0)
        for con in milp.constraints:
            assert con.coeffs, f"row {con.name} has no covering column"

    @pytest.mark.parametrize(
        ("func", "args"),
        [
            (generate_set_cover, (0, 5, 0.2, 0)),
            (generate_set_cover, (5, 0, 0.2, 0)),
            (generate_set_cover, (5, 5, 0.0, 0)),
            (generate_set_cover, (5, 5, 1.5, 0)),
            (generate_knapsack, (0, 0)),
            (generate_facility, (0, 3, 0)),
            (generate_facility, (3, 0, 0)),
        ],
    )
    def test_invalid_params_raise(
        self, func: Callable[..., MILP], args: tuple[object, ...]
    ) -> None:
        with pytest.raises(ValueError):
            func(*args)


# ---------------------------------------------------------------------------
# Committed registry validates + matches the catalog
# ---------------------------------------------------------------------------
class TestRegistryValidation:
    def test_committed_registry_validates(self) -> None:
        registry = BenchmarkRegistry.from_yaml(REGISTRY_PATH)
        registry.validate()
        phase1 = [e for e in registry.entries if e.phase == 1]
        assert len(phase1) == 4

    def test_committed_registry_matches_catalog(self) -> None:
        loaded = {
            e.name: e
            for e in BenchmarkRegistry.from_yaml(REGISTRY_PATH).entries
            if e.phase == 1
        }
        built = {e.name: e for e in build_registry_entries()}
        assert loaded.keys() == built.keys()
        for name, entry in built.items():
            assert loaded[name] == entry

    def test_instance_counts_match_catalog(self) -> None:
        registry = BenchmarkRegistry.from_yaml(REGISTRY_PATH)
        total = sum(e.instance_count for e in registry.entries if e.phase == 1)
        assert total == len(PHASE1_CATALOG)
        assert 20 <= total <= 50


# ---------------------------------------------------------------------------
# make_phase1_splits: seals dev/validation, idempotent, reseal-on-change
# ---------------------------------------------------------------------------
class TestMakePhase1Splits:
    def test_seal_creates_lock_and_is_idempotent(self, tmp_path: Path) -> None:
        reg = tmp_path / "registry.yaml"
        reg.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        lock = tmp_path / "split_manifest.lock"

        assert not lock.exists()
        make_phase1_splits(registry_path=reg, lock_path=lock)
        assert lock.exists()
        first = lock.read_text(encoding="utf-8")

        make_phase1_splits(registry_path=reg, lock_path=lock)
        assert lock.read_text(encoding="utf-8") == first

    def test_committed_lock_matches_committed_registry(self) -> None:
        registry = make_phase1_splits()
        registry.verify_lock()

    def test_changed_assignment_requires_reseal(self, tmp_path: Path) -> None:
        data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
        moved = data["benchmarks"][0]["split"]["dev"].pop()
        data["benchmarks"][0]["split"]["validation"].append(moved)
        reg = tmp_path / "registry.yaml"
        reg.write_text(yaml.safe_dump(data), encoding="utf-8")
        lock = tmp_path / "split_manifest.lock"

        make_phase1_splits(registry_path=reg, lock_path=lock)

        data["benchmarks"][0]["split"]["validation"].remove(moved)
        data["benchmarks"][0]["split"]["dev"].append(moved)
        reg.write_text(yaml.safe_dump(data), encoding="utf-8")

        with pytest.raises(LockMismatchError):
            make_phase1_splits(registry_path=reg, lock_path=lock, reseal=False)
        make_phase1_splits(registry_path=reg, lock_path=lock, reseal=True)


# ---------------------------------------------------------------------------
# No test/ood assignment anywhere
# ---------------------------------------------------------------------------
class TestNoHeldOutSplits:
    def test_phase1_entries_have_only_free_splits(self) -> None:
        registry = BenchmarkRegistry.from_yaml(REGISTRY_PATH)
        phase1 = [e for e in registry.entries if e.phase == 1]
        assert len(phase1) == 4
        for entry in phase1:
            assert not entry.split.get("test")
            assert not entry.split.get("ood_test")

    def test_catalog_has_no_held_out_recipes(self) -> None:
        assert all(r.split in {"dev", "validation"} for r in PHASE1_CATALOG)

    def test_make_phase1_splits_rejects_test_assignment(self, tmp_path: Path) -> None:
        rogue = {
            "benchmarks": [
                {
                    "name": "rogue_heldout",
                    "problem_type": "MILP",
                    "source": "synthetic",
                    "split": {"test": ["z1"]},
                    "license": "MIT",
                    "instance_count": 1,
                    "time_limit_sec": 30,
                    "baseline_set": "scip_default",
                    "leakage_group": "rogue_heldout_only",
                    "checksum": "sha256:" + "0" * 64,
                    "phase": 1,
                    "thesis": "T1",
                }
            ]
        }
        reg = tmp_path / "registry.yaml"
        reg.write_text(yaml.safe_dump(rogue), encoding="utf-8")
        with pytest.raises(Phase1Error, match="test"):
            make_phase1_splits(registry_path=reg, lock_path=tmp_path / "lock")

    def test_get_phase1_instances_rejects_held_out_split(self) -> None:
        with pytest.raises(Phase1Error):
            get_phase1_instances("test")


# ---------------------------------------------------------------------------
# get_phase1_instances materialises MILPs (synthetic path is offline)
# ---------------------------------------------------------------------------
class TestGetPhase1Instances:
    def test_dev_synthetic_count_and_types(self) -> None:
        milps = get_phase1_instances("dev", sources=("synthetic",))
        assert len(milps) == 21
        assert all(isinstance(m, MILP) for m in milps)
        assert all(m.n_vars > 0 and m.n_constraints > 0 for m in milps)

    def test_validation_synthetic_count(self) -> None:
        assert len(get_phase1_instances("validation", sources=("synthetic",))) == 9

    def test_synthetic_instances_are_reproducible(self) -> None:
        first = [milp_digest(m) for m in get_phase1_instances("dev", sources=("synthetic",))]
        second = [milp_digest(m) for m in get_phase1_instances("dev", sources=("synthetic",))]
        assert first == second


# ---------------------------------------------------------------------------
# Content checksum integrity (locks generators + MIPLIB file hashes)
# ---------------------------------------------------------------------------
class TestChecksumIntegrity:
    def test_synthetic_family_checksums_match_registry(self) -> None:
        entries = {e.name: e for e in BenchmarkRegistry.from_yaml(REGISTRY_PATH).entries}
        for name in (
            "synthetic_set_cover_phase1",
            "synthetic_knapsack_phase1",
            "synthetic_facility_phase1",
        ):
            recipes = [r for r in PHASE1_CATALOG if r.entry_name == name]
            items = sorted((r.id, milp_digest(materialize(r))) for r in recipes)
            manifest = "\n".join(f"{rid}={digest}" for rid, digest in items)
            expected = "sha256:" + hashlib.sha256(manifest.encode("utf-8")).hexdigest()
            assert expected == entries[name].checksum

    def test_miplib_family_checksum_matches_registry(self) -> None:
        entries = {e.name: e for e in BenchmarkRegistry.from_yaml(REGISTRY_PATH).entries}
        assert subset_manifest_checksum() == entries["miplib2017_phase1"].checksum


# ---------------------------------------------------------------------------
# MIPLIB acquisition (network + SCIP gated)
# ---------------------------------------------------------------------------
class TestMiplibChecksumHelpers:
    def test_verify_checksum_roundtrip(self, tmp_path: Path) -> None:
        payload = b"opop phase-1 checksum probe"
        target = tmp_path / "probe.bin"
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        assert verify_checksum(target, digest)
        assert verify_checksum(target, "sha256:" + digest)
        assert not verify_checksum(target, "0" * 64)


@pytest.mark.integration
class TestMiplibAcquisition:
    def test_download_one_verify_and_load(
        self, tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("scip")
        if not network_available():
            pytest.skip("MIPLIB mirror unreachable")
        inst = MIPLIB_PHASE1_SUBSET[0]
        path = download_miplib_instance(inst, tmp_path)
        assert verify_checksum(path, inst.sha256)
        milp = load_miplib_instance(inst, cache_dir=tmp_path, allow_download=False)
        assert isinstance(milp, MILP)
        assert milp.n_vars > 0 and milp.n_constraints > 0

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        if not network_available():
            pytest.skip("MIPLIB mirror unreachable")
        tampered = MiplibInstance(MIPLIB_PHASE1_SUBSET[0].name, "0" * 64, 0)
        with pytest.raises(MiplibChecksumError):
            download_miplib_instance(tampered, tmp_path)

    def test_get_phase1_instances_full_includes_miplib(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("scip")
        if not _miplib_available():
            pytest.skip("MIPLIB not cached and mirror unreachable")
        dev = get_phase1_instances("dev")
        validation = get_phase1_instances("validation")
        assert len(dev) == 29
        assert len(validation) == 13
        assert all(isinstance(m, MILP) for m in dev + validation)


# ---------------------------------------------------------------------------
# Registry schema sanity for the curated MIPLIB subset
# ---------------------------------------------------------------------------
def test_split_keys_are_valid() -> None:
    registry = BenchmarkRegistry.from_yaml(REGISTRY_PATH)
    for entry in registry.entries:
        assert set(entry.split).issubset(SPLITS)
