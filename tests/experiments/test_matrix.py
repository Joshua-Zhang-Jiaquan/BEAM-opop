"""Tests for the experiment-matrix job-runner foundation (plan task 39, chunk 1).

Covers: Cartesian expansion counts + determinism, the resume layer (a cell with an
``ok`` marker is skipped), the runner Protocol + factory (local / dry-run / cluster
stubs), local execution (resume-safe + error-tolerant), and the leakage-audit gate
(sealed-lock + held-out-split refusal). All pure / offline — no solver, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from opop.bench.registry import BenchmarkRegistry
from opop.experiments.audit_gate import MatrixAuditError, assert_can_run_split
from opop.experiments.matrix import (
    CANONICAL_ABLATIONS,
    STAGED_ABLATIONS,
    AblationRow,
    ExperimentMatrix,
    MatrixCell,
    MatrixStatus,
    as_cells,
    cell_marker_path,
    expand_matrix,
    is_cell_done,
    write_cell_marker,
)
from opop.experiments.runner import (
    DryRunRunner,
    Job,
    LocalRunner,
    Runner,
    runner_for,
)


def _matrix(**overrides: Any) -> ExperimentMatrix:
    """Build a small matrix, overriding any factor list."""
    factors: dict[str, Any] = {
        "instances": ("a", "b"),
        "methods": ("opop",),
        "ablations": ("full_opop",),
        "seeds": (0,),
        "time_limits": (5.0,),
    }
    factors.update(overrides)
    return ExperimentMatrix(**factors)


def _noop_work(cell: MatrixCell) -> dict[str, object]:
    """A work_fn that returns a trivial result (never raises)."""
    return {"ran": cell.slug}


def _registry_data() -> dict[str, object]:
    """A minimal one-entry registry payload (dev-only, schema-complete)."""
    return {
        "benchmarks": [
            {
                "name": "tmp_fam",
                "problem_type": "MILP",
                "source": "synthetic",
                "split": {"dev": ["tmp_001"], "validation": [], "test": [], "ood_test": []},
                "license": "MIT",
                "instance_count": 1,
                "time_limit_sec": 30,
                "baseline_set": "scip_default",
                "leakage_group": "tmp_fam",
                "checksum": "sha256:" + "0" * 64,
                "phase": 1,
                "thesis": "T1",
            }
        ]
    }


def _write_registry(tmp_path: Path, *, sealed: bool) -> Path:
    """Write a tmp registry.yaml + lock (correctly sealed or deliberately wrong)."""
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump(_registry_data()), encoding="utf-8")
    if sealed:
        BenchmarkRegistry.from_yaml(reg).seal()
    else:
        lock = tmp_path / "split_manifest.lock"
        lock.write_text(
            json.dumps({"hash": "0" * 64, "algorithm": "sha256"}) + "\n", encoding="utf-8"
        )
    return reg


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------
class TestExpansion:
    def test_matrix_expansion_counts_cells(self) -> None:
        matrix = _matrix(
            instances=("a", "b", "c"),
            methods=("opop", "scip-default"),
            ablations=("S0", "S2", "S4"),
            seeds=(0, 1),
            time_limits=(5.0,),
        )
        cells = matrix.expand()
        assert len(cells) == 3 * 2 * 3 * 2 * 1
        assert matrix.n_cells == len(cells)
        assert len(expand_matrix(matrix)) == len(cells)
        assert len({c.slug for c in cells}) == len(cells)

        first = cells[0]
        assert (first.instance_id, first.method, first.ablation, first.seed, first.time_limit) == (
            "a",
            "opop",
            "S0",
            0,
            5.0,
        )

    def test_expand_is_deterministic_and_innermost_is_time_limit(self) -> None:
        matrix = _matrix(instances=("a",), seeds=(0, 1), time_limits=(1.0, 2.0))
        cells = matrix.expand()
        assert [(c.seed, c.time_limit) for c in cells] == [(0, 1.0), (0, 2.0), (1, 1.0), (1, 2.0)]
        assert matrix.expand() == cells

    def test_factor_lists_are_normalised_to_tuples(self) -> None:
        matrix = ExperimentMatrix(
            instances=["a", "b"],
            methods=["opop"],
            ablations=[AblationRow.FULL_OPOP],
            seeds=[0],
            time_limits=[5],
        )
        assert isinstance(matrix.instances, tuple)
        assert matrix.ablations == ("full_opop",)
        assert matrix.time_limits == (5.0,)

    def test_cell_slug_is_filesystem_safe(self) -> None:
        cell = MatrixCell(
            instance_id="qplib/ball_miqcp",
            method="scip-default",
            ablation="full_opop",
            seed=0,
            time_limit=30.0,
        )
        slug = cell.slug
        assert "/" not in slug
        assert " " not in slug
        assert slug == MatrixCell("qplib/ball_miqcp", "scip-default", "full_opop", 0, 30.0).slug
        marker = cell_marker_path(cell, "/tmp/run")
        assert marker.name == "cell_done.json"
        assert slug in str(marker)


# ---------------------------------------------------------------------------
# Ablation vocabulary
# ---------------------------------------------------------------------------
class TestAblationRows:
    def test_canonical_and_staged_rows(self) -> None:
        assert AblationRow.SCIP_DEFAULT == "scip_default"
        assert set(CANONICAL_ABLATIONS) == {
            AblationRow.SCIP_DEFAULT,
            AblationRow.PARAMS_ONLY,
            AblationRow.ANALYZER_CUTS_ONLY,
            AblationRow.PARAMS_PLUS_CUTS,
            AblationRow.FULL_OPOP,
        }
        assert set(STAGED_ABLATIONS) == {
            AblationRow.S0,
            AblationRow.S1,
            AblationRow.S2,
            AblationRow.S3,
            AblationRow.S4,
        }

    def test_ablation_rows_usable_as_factor_levels(self) -> None:
        matrix = _matrix(instances=("a",), ablations=CANONICAL_ABLATIONS)
        cells = matrix.expand()
        assert len(cells) == 5
        assert {c.ablation for c in cells} == {
            "scip_default",
            "params_only",
            "analyzer_cuts_only",
            "params_plus_cuts",
            "full_opop",
        }


# ---------------------------------------------------------------------------
# Resume layer
# ---------------------------------------------------------------------------
class TestResume:
    def test_resume_skips_completed_cells(self, tmp_path: Path) -> None:
        cells = _matrix().expand()
        write_cell_marker(cells[0], tmp_path, status="ok", result={"x": 1}, runner="test")
        assert is_cell_done(cells[0], tmp_path)
        assert not is_cell_done(cells[1], tmp_path)

        seen: list[str] = []

        def work_fn(cell: MatrixCell) -> dict[str, object]:
            seen.append(cell.slug)
            return {"ran": cell.slug}

        jobs = LocalRunner().submit_jobs(cells, work_fn=work_fn, out_dir=tmp_path)
        assert seen == [cells[1].slug]
        by_slug = {j.cell.slug: j for j in jobs}
        assert by_slug[cells[0].slug].status == "skipped"
        assert by_slug[cells[1].slug].status == "ok"
        assert is_cell_done(cells[1], tmp_path)

    def test_non_ok_marker_is_not_done(self, tmp_path: Path) -> None:
        cell = _matrix().expand()[0]
        write_cell_marker(cell, tmp_path, status="error", error="boom", runner="test")
        assert not is_cell_done(cell, tmp_path)

    def test_matrix_status_scan(self, tmp_path: Path) -> None:
        matrix = _matrix(instances=("a", "b", "c"))
        cells = matrix.expand()
        write_cell_marker(cells[0], tmp_path, status="ok", runner="t")
        write_cell_marker(cells[1], tmp_path, status="error", error="x", runner="t")
        status = MatrixStatus.scan(matrix, tmp_path)
        assert (status.total, status.n_done, status.n_pending) == (3, 1, 2)
        assert cells[0] in status.done
        assert cells[1] in status.pending
        assert cells[2] in status.pending
        assert status.to_dict()["n_done"] == 1


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
class TestRunners:
    def test_runner_for_factory(self) -> None:
        for kind in ("local", "dry-run", "slurm", "qz"):
            runner = runner_for(kind)
            assert runner.kind == kind
            assert isinstance(runner, Runner)
        with pytest.raises(ValueError, match="unknown runner kind"):
            runner_for("k8s")

    def test_local_dry_run_lists_expected_jobs(self) -> None:
        matrix = _matrix(
            instances=("a", "b"),
            methods=("opop", "scip-default"),
            ablations=("S0", "S4"),
            seeds=(0, 1),
        )
        calls: list[str] = []

        def work_fn(cell: MatrixCell) -> dict[str, object]:
            calls.append(cell.slug)
            return {}

        jobs = DryRunRunner().submit_jobs(matrix, work_fn=work_fn)
        assert [j.cell for j in jobs] == list(matrix.expand())
        assert calls == []
        assert all(isinstance(j, Job) for j in jobs)
        assert all(j.status == "dry-run" for j in jobs)
        assert all(j.runner_kind == "dry-run" for j in jobs)
        assert all(j.command for j in jobs)

    def test_local_runner_requires_work_fn(self) -> None:
        with pytest.raises(ValueError, match="work_fn"):
            LocalRunner().submit_jobs(_matrix())

    def test_local_runner_records_ok_and_catches_errors(self, tmp_path: Path) -> None:
        cells = _matrix(instances=("good", "bad")).expand()

        def work_fn(cell: MatrixCell) -> dict[str, object]:
            if cell.instance_id == "bad":
                raise RuntimeError("boom")
            return {"ok": True}

        jobs = LocalRunner().submit_jobs(cells, work_fn=work_fn, out_dir=tmp_path)
        by_id = {j.cell.instance_id: j for j in jobs}
        assert by_id["good"].status == "ok"
        assert by_id["bad"].status == "error"
        assert "boom" in by_id["bad"].detail
        # The good cell is marked done; the failed cell is NOT (so it re-runs).
        assert is_cell_done(cells[0], tmp_path)
        assert not is_cell_done(cells[1], tmp_path)

    @pytest.mark.parametrize("kind", ["slurm", "qz"])
    def test_cluster_runner_plans_but_submit_raises(self, kind: str) -> None:
        matrix = _matrix()
        runner = runner_for(kind)
        jobs = runner.plan_jobs(matrix)
        assert len(jobs) == len(matrix.expand())
        assert all(j.runner_kind == kind and j.command for j in jobs)
        with pytest.raises(NotImplementedError):
            runner.submit_jobs(matrix, work_fn=_noop_work)

    def test_plan_jobs_accepts_cells_or_matrix(self) -> None:
        matrix = _matrix(instances=("a", "b"))
        from_matrix = LocalRunner().plan_jobs(matrix)
        from_cells = LocalRunner().plan_jobs(as_cells(matrix))
        assert [j.cell for j in from_matrix] == [j.cell for j in from_cells]


# ---------------------------------------------------------------------------
# Leakage-audit gate
# ---------------------------------------------------------------------------
class TestAuditGate:
    def test_leakage_gate_blocks_test_split(self) -> None:
        with pytest.raises(MatrixAuditError, match="held-out"):
            assert_can_run_split("test", one_shot_final=False)
        with pytest.raises(MatrixAuditError, match="held-out"):
            assert_can_run_split("ood_test", one_shot_final=False)

    def test_leakage_gate_allows_free_splits_and_final_mode(self) -> None:
        assert_can_run_split("dev")
        assert_can_run_split("validation")
        assert_can_run_split("test", one_shot_final=True)
        assert_can_run_split("ood_test", one_shot_final=True)

    def test_leakage_gate_unknown_split_raises(self) -> None:
        with pytest.raises(MatrixAuditError, match="unknown split"):
            assert_can_run_split("train")

    def test_leakage_gate_unsealed_lock_raises(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, sealed=False)
        with pytest.raises(MatrixAuditError, match="lock"):
            assert_can_run_split("dev", registry_path=reg)

    def test_leakage_gate_sealed_tmp_registry(self, tmp_path: Path) -> None:
        reg = _write_registry(tmp_path, sealed=True)
        assert_can_run_split("dev", registry_path=reg)
        with pytest.raises(MatrixAuditError, match="held-out"):
            assert_can_run_split("test", registry_path=reg, one_shot_final=False)
