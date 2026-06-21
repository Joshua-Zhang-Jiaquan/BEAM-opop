"""Tests for the Wave-6 held-out MILP suites (plan task 33).

Covers: combined registry validation (Phase-1 free splits + Wave-6 held-out
suites), leakage-safety (no leakage_group spans a free and a held-out split; no
held-out instance shares a name/family with a free instance), per-suite content
checksums matching the committed registry, sealed-lock integrity, and the
held-out split-access policy. Download / load / checksum-verify paths for MIPLIB
2017, Distributional MIPLIB, and MILPBench are ``integration`` tests gated on
network + SCIP + cache, so the suite stays green fully offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from opop.bench.registry import (
    FREE_SPLITS,
    HELD_SPLITS,
    BenchmarkRegistry,
    FinalModeRequiredError,
)
from opop.bench.sources import distributional, milpbench, miplib, modeling_agents, qplib
from opop.bench.sources.milp_suites import (
    LOCK_PATH,
    REGISTRY_PATH,
    build_all_entries,
    build_suite_entries,
)
from opop.bench.sources.miplib import MIPLIB_PHASE1_SUBSET, default_cache_dir, verify_checksum
from opop.model.ir import MILP

_SUITE_SOURCES = frozenset({"miplib2017", "dmiplib", "milpbench"})


def _committed_registry() -> BenchmarkRegistry:
    return BenchmarkRegistry.from_yaml(REGISTRY_PATH, lock_path=LOCK_PATH)


def _heldout_miplib_cached() -> bool:
    """True if every held-out MIPLIB file is cached+valid, or the mirror is reachable."""
    cache = default_cache_dir()
    cached = all(
        (cache / inst.filename).exists() and verify_checksum(cache / inst.filename, inst.sha256)
        for inst in miplib.MIPLIB_HELDOUT_SUBSET
    )
    return cached or miplib.network_available()


# ---------------------------------------------------------------------------
# Combined registry validates and matches the catalogs
# ---------------------------------------------------------------------------
class TestCombinedRegistry:
    def test_committed_registry_validates(self) -> None:
        registry = _committed_registry()
        registry.validate()
        assert len(registry.entries) == 23

    def test_committed_registry_matches_combined_catalog(self) -> None:
        loaded = {e.name: e for e in _committed_registry().entries}
        built = {e.name: e for e in build_all_entries()}
        assert loaded.keys() == built.keys()
        for name, entry in built.items():
            assert loaded[name] == entry

    def test_committed_lock_matches_registry(self) -> None:
        _committed_registry().verify_lock()

    def test_suite_entry_names_and_sources(self) -> None:
        suites = {e.name: e for e in build_suite_entries()}
        assert set(suites) == {
            "miplib2017_collection_test",
            "miplib2017_collection_ood",
            "dmiplib_MIS",
            "dmiplib_MVC",
            "dmiplib_SC",
            "dmiplib_CA",
            "dmiplib_GISP",
            "dmiplib_IP",
            "milpbench_knapsack",
        }
        for entry in suites.values():
            assert entry.source in _SUITE_SOURCES
            assert entry.phase == 6
            assert entry.thesis == "T1"

    def test_phase1_entries_unchanged(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        phase1 = [e for e in entries.values() if e.phase == 1]
        assert len(phase1) == 4
        for entry in phase1:
            assert not entry.split.get("test")
            assert not entry.split.get("ood_test")


# ---------------------------------------------------------------------------
# Immutable test / ood_test splits + access policy
# ---------------------------------------------------------------------------
class TestHeldOutSplits:
    def test_split_totals(self) -> None:
        registry = _committed_registry()
        test = registry.get_split("test", one_shot_final=True)
        ood = registry.get_split("ood_test", one_shot_final=True)
        assert len(test) == 31
        assert len(ood) == 18

    def test_held_out_requires_one_shot_final(self) -> None:
        registry = _committed_registry()
        with pytest.raises(FinalModeRequiredError):
            registry.get_split("test")
        with pytest.raises(FinalModeRequiredError):
            registry.get_split("ood_test")
        assert registry.get_split("test", one_shot_final=True)
        assert registry.get_split("ood_test", one_shot_final=True)

    def test_every_suite_instance_is_held_out(self) -> None:
        registry = _committed_registry()
        for entry in registry.entries:
            if entry.phase == 6:
                assert not entry.split.get("dev")
                assert not entry.split.get("validation")
                assert set(entry.split).issubset(HELD_SPLITS)


# ---------------------------------------------------------------------------
# Leakage safety (the scientific-integrity backbone of task 33)
# ---------------------------------------------------------------------------
class TestLeakageSafety:
    def test_no_group_spans_free_and_held(self) -> None:
        group_splits: dict[str, set[str]] = {}
        for entry in _committed_registry().entries:
            present = {name for name, ids in entry.split.items() if ids}
            group_splits.setdefault(entry.leakage_group, set()).update(present)
        for group, splits in group_splits.items():
            assert not (
                (splits & FREE_SPLITS) and (splits & HELD_SPLITS)
            ), f"leakage_group {group!r} spans free and held-out splits: {sorted(splits)}"

    def test_no_instance_in_two_splits(self) -> None:
        location: dict[str, str] = {}
        for entry in _committed_registry().entries:
            for split_name, ids in entry.split.items():
                for inst in ids:
                    assert inst not in location, f"{inst} in two splits"
                    location[inst] = split_name

    def test_held_out_miplib_disjoint_from_phase1(self) -> None:
        phase1_names = {inst.name for inst in MIPLIB_PHASE1_SUBSET}
        heldout_names = {inst.name for inst in miplib.MIPLIB_HELDOUT_SUBSET}
        assert phase1_names.isdisjoint(heldout_names)
        # mas74 must NOT be held out: it shares the `mas` family with free-split mas76.
        assert "mas74" not in heldout_names
        assert "mas76" in phase1_names

    def test_suite_leakage_groups_distinct_from_free(self) -> None:
        registry = _committed_registry()
        free_groups = {e.leakage_group for e in registry.entries if e.phase == 1}
        held_groups = {e.leakage_group for e in registry.entries if e.phase == 6}
        assert free_groups.isdisjoint(held_groups)


# ---------------------------------------------------------------------------
# Content checksums lock generators / downloads (offline; no SCIP)
# ---------------------------------------------------------------------------
class TestChecksumIntegrity:
    def test_miplib_heldout_checksums_match_registry(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        test_insts = tuple(miplib.instance_by_name(n) for n in miplib.MIPLIB_HELDOUT_TEST)
        ood_insts = tuple(miplib.instance_by_name(n) for n in miplib.MIPLIB_HELDOUT_OOD)
        assert (
            miplib.subset_manifest_checksum(test_insts)
            == entries["miplib2017_collection_test"].checksum
        )
        assert (
            miplib.subset_manifest_checksum(ood_insts)
            == entries["miplib2017_collection_ood"].checksum
        )

    def test_dmiplib_checksums_match_registry(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        by_domain: dict[str, list[distributional.DMiplibDistribution]] = {}
        for dist in distributional.DMIPLIB_SUBSET:
            by_domain.setdefault(dist.domain, []).append(dist)
        for domain, dists in by_domain.items():
            expected = distributional.subset_manifest_checksum(tuple(dists))
            assert expected == entries[f"dmiplib_{domain}"].checksum

    def test_milpbench_checksum_matches_registry(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        assert milpbench.subset_manifest_checksum() == entries["milpbench_knapsack"].checksum

    def test_checksum_manifest_is_deterministic(self) -> None:
        assert distributional.subset_manifest_checksum() == distributional.subset_manifest_checksum()
        assert milpbench.subset_manifest_checksum() == milpbench.subset_manifest_checksum()

    def test_qplib_checksums_match_registry(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        for fixture in qplib.QPLIB_FIXTURES:
            assert entries[fixture.entry_name].checksum == "sha256:" + fixture.sha256

    def test_modeling_checksums_match_registry(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        for dataset in modeling_agents.MODELING_CATALOG:
            assert entries[dataset.entry_name].checksum == "sha256:" + dataset.sha256

    def test_qplib_fixture_hashes_match_catalog(self) -> None:
        base = Path(__file__).parent / "fixtures" / "qplib"
        for fixture in qplib.QPLIB_FIXTURES:
            digest = hashlib.sha256((base / fixture.filename).read_bytes()).hexdigest()
            assert digest == fixture.sha256, f"fixture drift: {fixture.filename}"

    def test_modeling_fixture_hashes_match_catalog(self) -> None:
        base = Path(__file__).parent / "fixtures" / "modeling"
        for dataset in modeling_agents.MODELING_CATALOG:
            digest = hashlib.sha256((base / dataset.filename).read_bytes()).hexdigest()
            assert digest == dataset.sha256, f"fixture drift: {dataset.filename}"


# ---------------------------------------------------------------------------
# Task-35 generality families: QPLIB MIQP/MIQCP + cleaned modeling-agent sets
# ---------------------------------------------------------------------------
class TestGeneralityFamilies:
    def test_entries_present_and_held_out(self) -> None:
        entries = {e.name: e for e in _committed_registry().entries}
        expected = {
            "qplib_miqcp_tiny": ("qplib", "MIQCP"),
            "qplib_miqp_tiny": ("qplib", "MIQP"),
            "modeling_nl4opt_cleaned": ("modeling_agent", "MILP"),
            "modeling_optibench_cleaned": ("modeling_agent", "MINLP"),
        }
        for name, (source, problem_type) in expected.items():
            entry = entries[name]
            assert entry.source == source
            assert entry.problem_type == problem_type
            assert entry.phase == 6
            assert entry.thesis == "T3"
            assert entry.leakage_group == name
            assert set(entry.split).issubset(HELD_SPLITS)
            assert not entry.split.get("dev")
            assert not entry.split.get("validation")

    def test_in_build_all_entries(self) -> None:
        names = {e.name for e in build_all_entries()}
        assert {
            "qplib_miqcp_tiny",
            "qplib_miqp_tiny",
            "modeling_nl4opt_cleaned",
            "modeling_optibench_cleaned",
        } <= names

    def test_planted_wrong_not_in_registry(self) -> None:
        all_ids = {
            inst
            for entry in _committed_registry().entries
            for ids in entry.split.values()
            for inst in ids
        }
        assert "optibench/planted_wrong" not in all_ids


# ---------------------------------------------------------------------------
# Instance-id namespacing + source-of-truth metadata (offline)
# ---------------------------------------------------------------------------
class TestInstanceIds:
    def test_dmiplib_ids_are_namespaced_and_unique(self) -> None:
        ids = [d.id for d in distributional.DMIPLIB_SUBSET]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("dmiplib/") for i in ids)

    def test_milpbench_ids_are_namespaced_and_unique(self) -> None:
        ids = [i.id for i in milpbench.MILPBENCH_SUBSET]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("milpbench/knapsack/") for i in ids)

    def test_all_registry_instance_ids_are_globally_unique(self) -> None:
        seen: set[str] = set()
        for entry in _committed_registry().entries:
            for ids in entry.split.values():
                for inst in ids:
                    assert inst not in seen
                    seen.add(inst)

    def test_milpbench_gdrive_source_documented(self) -> None:
        assert milpbench.MILPBENCH_GDRIVE
        url = milpbench.gdrive_url("MIS_easy")
        assert url.startswith("https://drive.google.com/")


# ---------------------------------------------------------------------------
# Checksum-verification helper behaviour (offline)
# ---------------------------------------------------------------------------
class TestVerifyChecksum:
    def test_verify_checksum_roundtrip(self, tmp_path: Path) -> None:
        payload = b"opop task-33 checksum probe"
        target = tmp_path / "probe.bin"
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        assert verify_checksum(target, digest)
        assert verify_checksum(target, "sha256:" + digest)
        assert not verify_checksum(target, "0" * 64)


# ---------------------------------------------------------------------------
# MIPLIB 2017 held-out acquisition (cache + SCIP gated)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestMiplibHeldoutAcquisition:
    def test_download_verify_and_load(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("scip")
        if not _heldout_miplib_cached():
            pytest.skip("MIPLIB mirror unreachable and held-out cache empty")
        inst = miplib.MIPLIB_HELDOUT_SUBSET[0]
        milp = miplib.load_miplib_instance(inst, cache_dir=default_cache_dir())
        assert isinstance(milp, MILP)
        assert milp.n_vars > 0 and milp.n_constraints > 0

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        if not miplib.network_available():
            pytest.skip("MIPLIB mirror unreachable")
        tampered = miplib.MiplibInstance(miplib.MIPLIB_HELDOUT_SUBSET[0].name, "0" * 64, 0)
        with pytest.raises(miplib.MiplibChecksumError):
            miplib.download_miplib_instance(tampered, tmp_path)


# ---------------------------------------------------------------------------
# Distributional MIPLIB acquisition (network gated)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestDMiplibAcquisition:
    def test_download_smallest_and_verify(self, tmp_path: Path) -> None:
        if not distributional.network_available():
            pytest.skip("D-MIPLIB (Hugging Face) mirror unreachable")
        smallest = min(distributional.DMIPLIB_SUBSET, key=lambda d: d.n_bytes)
        try:
            path = distributional.download_distribution(smallest, tmp_path, timeout=180.0)
        except distributional.DMiplibDownloadError:
            pytest.skip("D-MIPLIB download failed (network)")
        assert verify_checksum(path, smallest.sha256)


# ---------------------------------------------------------------------------
# MILPBench acquisition (network + SCIP gated)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestMilpbenchAcquisition:
    def test_download_verify_and_load(
        self, tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("scip")
        if not milpbench.network_available():
            pytest.skip("MILPBench (raw GitHub) host unreachable")
        inst = milpbench.MILPBENCH_SUBSET[0]
        try:
            path = milpbench.download_instance(inst, tmp_path)
        except milpbench.MilpbenchDownloadError:
            pytest.skip("MILPBench download failed (network)")
        assert verify_checksum(path, inst.sha256)
        milp = milpbench.load_instance(inst, cache_dir=tmp_path, allow_download=False)
        assert isinstance(milp, MILP)
        assert milp.n_vars > 0 and milp.n_constraints > 0

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        if not milpbench.network_available():
            pytest.skip("MILPBench host unreachable")
        original = milpbench.MILPBENCH_SUBSET[0]
        tampered = milpbench.MilpbenchInstance(
            original.problem_class, original.name, original.repo_path, "0" * 64, 0
        )
        with pytest.raises(milpbench.MilpbenchChecksumError):
            milpbench.download_instance(tampered, tmp_path)
