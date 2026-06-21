from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from opop.bench.registry import (
    BenchmarkEntry,
    BenchmarkRegistry,
    FinalModeRequiredError,
    LeakageError,
    LockMismatchError,
    SchemaError,
    main,
)


def _split(**kwargs: list[str]) -> dict[str, tuple[str, ...]]:
    return {k: tuple(v) for k, v in kwargs.items()}


def _entry(
    name: str = "bench_a",
    split: dict[str, list[str]] | None = None,
    leakage_group: str = "group_a",
    phase: int = 1,
    thesis: str = "T1",
) -> BenchmarkEntry:
    split = split or {"dev": ["i1"], "validation": [], "test": [], "ood_test": []}
    return BenchmarkEntry(
        name=name,
        problem_type="MILP",
        source="synthetic",
        split=_split(**split),
        license="MIT",
        instance_count=sum(len(v) for v in split.values()),
        time_limit_sec=60.0,
        baseline_set="scip_default",
        leakage_group=leakage_group,
        checksum="sha256:" + "0" * 64,
        phase=phase,
        thesis=thesis,
    )


class TestSchemaValidation:
    def test_phase_and_thesis_required(self, tmp_path: Path) -> None:
        data = {
            "benchmarks": [
                {
                    "name": "missing_phase",
                    "problem_type": "MILP",
                    "source": "synthetic",
                    "split": {"dev": ["x"], "validation": [], "test": [], "ood_test": []},
                    "license": "MIT",
                    "instance_count": 1,
                    "time_limit_sec": 60,
                    "baseline_set": "scip_default",
                    "leakage_group": "g",
                    "checksum": "sha256:" + "0" * 64,
                    # missing phase and thesis
                }
            ]
        }
        path = tmp_path / "registry.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(SchemaError, match="missing required fields"):
            BenchmarkRegistry.from_yaml(path)

    def test_instance_count_mismatch(self) -> None:
        entry = _entry(split={"dev": ["i1", "i2"], "validation": [], "test": [], "ood_test": []})
        entry = BenchmarkEntry(
            name=entry.name,
            problem_type=entry.problem_type,
            source=entry.source,
            split=entry.split,
            license=entry.license,
            instance_count=99,
            time_limit_sec=entry.time_limit_sec,
            baseline_set=entry.baseline_set,
            leakage_group=entry.leakage_group,
            checksum=entry.checksum,
            phase=entry.phase,
            thesis=entry.thesis,
        )
        with pytest.raises(SchemaError, match="instance_count"):
            BenchmarkRegistry([entry])


class TestLeakage:
    def test_overlap_across_splits_is_rejected(self) -> None:
        registry = BenchmarkRegistry(
            [
                _entry(
                    name="bench_a",
                    split={
                        "dev": ["shared"],
                        "validation": [],
                        "test": [],
                        "ood_test": [],
                    },
                    leakage_group="group_a",
                ),
                _entry(
                    name="bench_b",
                    split={
                        "dev": [],
                        "validation": [],
                        "test": ["shared"],
                        "ood_test": [],
                    },
                    leakage_group="group_b",
                ),
            ]
        )
        with pytest.raises(LeakageError, match="shared"):
            registry.assert_no_overlap()

    def test_leakage_group_spanning_free_and_held_out_is_rejected(self) -> None:
        registry = BenchmarkRegistry(
            [
                _entry(
                    name="bench_free",
                    split={"dev": ["f1"], "validation": [], "test": [], "ood_test": []},
                    leakage_group="mixed",
                ),
                _entry(
                    name="bench_held",
                    split={"dev": [], "validation": [], "test": ["h1"], "ood_test": []},
                    leakage_group="mixed",
                ),
            ]
        )
        with pytest.raises(LeakageError, match="mixed"):
            registry.assert_no_overlap()

    def test_leakage_group_may_span_free_splits(self) -> None:
        registry = BenchmarkRegistry(
            [
                _entry(
                    name="bench_a",
                    split={"dev": ["a1"], "validation": [], "test": [], "ood_test": []},
                    leakage_group="free_group",
                ),
                _entry(
                    name="bench_b",
                    split={"dev": [], "validation": ["b1"], "test": [], "ood_test": []},
                    leakage_group="free_group",
                ),
            ]
        )
        registry.assert_no_overlap()


class TestSplitAccess:
    def test_dev_and_validation_loadable_freely(self) -> None:
        registry = BenchmarkRegistry([_entry()])
        assert registry.get_split("dev") == [("bench_a", "i1")]
        assert registry.get_split("validation") == []

    def test_test_requires_one_shot_final(self) -> None:
        registry = BenchmarkRegistry(
            [
                _entry(
                    split={
                        "dev": [],
                        "validation": [],
                        "test": ["t1"],
                        "ood_test": [],
                    }
                )
            ]
        )
        with pytest.raises(FinalModeRequiredError, match="one_shot_final"):
            registry.get_split("test")
        assert registry.get_split("test", one_shot_final=True) == [("bench_a", "t1")]

    def test_ood_requires_one_shot_final(self) -> None:
        registry = BenchmarkRegistry(
            [
                _entry(
                    split={
                        "dev": [],
                        "validation": [],
                        "test": [],
                        "ood_test": ["o1"],
                    }
                )
            ]
        )
        with pytest.raises(FinalModeRequiredError):
            registry.get_split("ood_test")
        assert registry.get_split("ood_test", one_shot_final=True) == [("bench_a", "o1")]


class TestLock:
    def test_seal_and_verify(self, tmp_path: Path) -> None:
        registry = BenchmarkRegistry([_entry()], lock_path=tmp_path / "lock.json")
        registry.seal()
        registry.verify_lock()

    def test_tampered_lock_refused(self, tmp_path: Path) -> None:
        registry = BenchmarkRegistry([_entry()], lock_path=tmp_path / "lock.json")
        registry.seal()

        # Tamper with the split assignment.
        registry.entries = (
            BenchmarkEntry(
                name="bench_a",
                problem_type="MILP",
                source="synthetic",
                split=_split(dev=["i1"], validation=["i2"], test=[], ood_test=[]),
                license="MIT",
                instance_count=2,
                time_limit_sec=60.0,
                baseline_set="scip_default",
                leakage_group="group_a",
                checksum="sha256:" + "0" * 64,
                phase=1,
                thesis="T1",
            ),
        )
        with pytest.raises(LockMismatchError, match="split lock mismatch"):
            registry.verify_lock()

    def test_reseal_recovers_mismatch(self, tmp_path: Path) -> None:
        registry = BenchmarkRegistry([_entry()], lock_path=tmp_path / "lock.json")
        registry.seal()
        registry.entries = (
            BenchmarkRegistry(
                [_entry(split={"dev": ["i1", "i2"], "validation": [], "test": [], "ood_test": []})]
            ).entries[0],
        )
        registry.verify_lock(reseal=True)
        assert registry.lock_path is not None
        assert registry.lock_path.exists()


class TestCLI:
    def test_cli_validate_valid_registry(self, tmp_path: Path) -> None:
        data = {
            "benchmarks": [
                {
                    "name": "valid",
                    "problem_type": "MILP",
                    "source": "synthetic",
                    "split": {
                        "dev": ["v1"],
                        "validation": [],
                        "test": [],
                        "ood_test": [],
                    },
                    "license": "MIT",
                    "instance_count": 1,
                    "time_limit_sec": 60,
                    "baseline_set": "scip_default",
                    "leakage_group": "g",
                    "checksum": "sha256:" + "0" * 64,
                    "phase": 1,
                    "thesis": "T1",
                }
            ]
        }
        path = tmp_path / "registry.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert main(["--validate", str(path), "--reseal"]) == 0
        assert main(["--validate", str(path)]) == 0

    def test_cli_validate_invalid_registry(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.yaml"
        path.write_text("not_a_mapping\n", encoding="utf-8")
        assert main(["--validate", str(path)]) == 1

    def test_cli_validate_real_registry(self) -> None:
        from opop.bench.sources.phase1_set import REGISTRY_PATH

        registry_path = REGISTRY_PATH
        result = subprocess.run(
            [sys.executable, "-m", "opop.bench.registry", "--validate", str(registry_path)],
            cwd=str(Path(__file__).resolve().parents[2]),
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "registry valid" in result.stdout
