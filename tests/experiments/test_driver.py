"""Tests for the experiment-matrix driver (plan task 39, chunk 2).

Pure (offline, no-solver) tests cover: the audit gate (held-out split blocked,
unsealed lock refused), unknown-ablation rejection, the dry-run plan listing, the
resume layer (pre-written ``ok`` markers are skipped — the work_fn never runs), and
the consolidated ``results.parquet`` schema. A SCIP-backed ``@pytest.mark.slow`` /
``integration`` test drives a tiny real sweep (1 instance x 2 ablations x 1 seed)
and checks the per-ablation method tags + aggregated artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from opop.experiments.audit_gate import MatrixAuditError
from opop.experiments.driver import (
    MATRIX_RESULT_COLUMNS,
    MatrixDriver,
    MatrixDriverError,
    run_matrix,
)
from opop.experiments.matrix import ExperimentMatrix, MatrixCell, write_cell_marker


def _matrix(
    *,
    instances: tuple[str, ...] = ("inst0",),
    ablations: tuple[str, ...] = ("scip_default", "full_opop"),
    seeds: tuple[int, ...] = (0,),
    time_limits: tuple[float, ...] = (5.0,),
) -> ExperimentMatrix:
    return ExperimentMatrix(
        instances=instances,
        methods=("matrix",),
        ablations=ablations,
        seeds=seeds,
        time_limits=time_limits,
    )


def _fake_row(cell: MatrixCell, *, primal: float) -> dict[str, Any]:
    """A pre-baked canonical result row (sentinel ``primal`` proves the skip)."""
    return {
        "instance_id": cell.instance_id,
        "method": "fake",
        "ablation": str(cell.ablation),
        "seed": int(cell.seed),
        "time_limit": float(cell.time_limit),
        "primal_integral": primal,
        "gap": 0.0,
        "time": 0.1,
        "solved": True,
        "censored": False,
        "n_accepted": 0,
    }


def _write_unsealed_registry(tmp_path: Path) -> Path:
    """Write a schema-valid registry with a deliberately WRONG lock."""
    data = {
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
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "split_manifest.lock").write_text(
        json.dumps({"hash": "0" * 64, "algorithm": "sha256"}) + "\n", encoding="utf-8"
    )
    return reg


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read the consolidated rows from a parquet (pandas) or JSON-fallback file."""
    if path.suffix == ".parquet":
        pd = pytest.importorskip("pandas")
        frame = pd.read_parquet(path)
        records: list[dict[str, Any]] = frame.to_dict(orient="records")
        return records
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    return list(raw)


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
def test_result_columns_contract() -> None:
    required = {
        "instance_id",
        "method",
        "ablation",
        "seed",
        "time_limit",
        "primal_integral",
        "gap",
        "time",
        "solved",
        "censored",
        "n_accepted",
    }
    assert required <= set(MATRIX_RESULT_COLUMNS)


# ---------------------------------------------------------------------------
# Audit gate
# ---------------------------------------------------------------------------
def test_leakage_guard_blocks_test_split(tmp_path: Path) -> None:
    driver = MatrixDriver(
        _matrix(ablations=("full_opop",)),
        out_dir=tmp_path / "run",
        split="test",
        one_shot_final=False,
    )
    with pytest.raises(MatrixAuditError, match="held-out"):
        driver.run()
    assert not (tmp_path / "run" / "results.parquet").exists()


def test_unsealed_lock_raises(tmp_path: Path) -> None:
    reg = _write_unsealed_registry(tmp_path)
    driver = MatrixDriver(
        _matrix(ablations=("full_opop",)),
        out_dir=tmp_path / "run",
        split="dev",
        registry_path=reg,
    )
    with pytest.raises(MatrixAuditError, match="lock"):
        driver.run()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_unknown_ablation_raises_value_error(tmp_path: Path) -> None:
    driver = MatrixDriver(
        _matrix(ablations=("bogus_ablation",)),
        out_dir=tmp_path / "run",
        split="dev",
        runner_kind="dry-run",
    )
    with pytest.raises(ValueError, match="unknown ablation"):
        driver.run()


def test_missing_instance_raises(tmp_path: Path) -> None:
    driver = MatrixDriver(
        _matrix(instances=("not_a_real_instance",), ablations=("full_opop",)),
        out_dir=tmp_path / "run",
        split="dev",
        instances={"some_other_id": object()},
    )
    with pytest.raises(MatrixDriverError, match="unmaterialised"):
        driver.run()


# ---------------------------------------------------------------------------
# Dry run (no solver)
# ---------------------------------------------------------------------------
def test_dry_run_lists_plan_without_solving(tmp_path: Path) -> None:
    matrix = _matrix(
        instances=("inst0", "inst1"),
        ablations=("scip_default", "full_opop"),
        seeds=(0, 1),
    )
    driver = MatrixDriver(matrix, out_dir=tmp_path, split="dev", runner_kind="dry-run")
    path = driver.run()

    assert path.name == "matrix_plan.json"
    plan: Any = json.loads(path.read_text(encoding="utf-8"))
    assert plan["n_jobs"] == len(matrix.expand())  # 2 x 1 x 2 x 2 x 1 == 8
    assert len(plan["jobs"]) == plan["n_jobs"]
    assert all(job["runner_kind"] == "dry-run" for job in plan["jobs"])
    assert not (tmp_path / "results.parquet").exists()
    assert not (tmp_path / "results.json").exists()


def test_run_matrix_dry_run(tmp_path: Path) -> None:
    path = run_matrix(
        out_dir=tmp_path,
        split="dev",
        ablations=("scip_default", "params_only", "full_opop"),
        instance_limit=2,
        seeds=(0,),
        time_limits=(5.0,),
        runner_kind="dry-run",
    )
    assert path.name == "matrix_plan.json"
    plan: Any = json.loads(path.read_text(encoding="utf-8"))
    # 2 synthetic instances x 3 ablations x 1 seed x 1 time-limit == 6 jobs.
    assert plan["n_jobs"] == 6


# ---------------------------------------------------------------------------
# Resume + consolidated schema (no solver: pre-written markers are skipped)
# ---------------------------------------------------------------------------
def test_resume_skips_completed_cells_and_writes_schema(tmp_path: Path) -> None:
    matrix = _matrix(ablations=("scip_default", "full_opop"), time_limits=(3.0,))
    cells = matrix.expand()
    sentinel = 123.456
    for cell in cells:
        write_cell_marker(
            cell, tmp_path, status="ok", result=_fake_row(cell, primal=sentinel), runner="pretest"
        )

    # ``instances`` value is never used: every cell has an ``ok`` marker, so the
    # work_fn (which would solve) is never invoked — proving resume + offline.
    driver = MatrixDriver(matrix, out_dir=tmp_path, split="dev", instances={"inst0": object()})
    path = driver.run()

    rows = _read_rows(path)
    assert len(rows) == len(cells)
    assert all(set(MATRIX_RESULT_COLUMNS) <= set(row) for row in rows)
    # The sentinel survived -> the pre-written marker was reused (cell skipped).
    assert all(float(row["primal_integral"]) == pytest.approx(sentinel) for row in rows)

    manifest: Any = json.loads((tmp_path / "repro_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_cells"] == len(cells)
    assert manifest["status"]["n_done"] == len(cells)
    assert manifest["status"]["n_pending"] == 0


# ---------------------------------------------------------------------------
# Integration: a tiny real local sweep (SCIP-gated, slow)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
def test_local_sweep_produces_results(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    solver_skip_if_missing("scip")
    path = run_matrix(
        out_dir=tmp_path,
        split="dev",
        ablations=("scip_default", "full_opop"),
        instance_limit=1,
        seeds=(0,),
        time_limits=(3.0,),
        trials=2,
        runner_kind="local",
    )

    rows = _read_rows(path)
    assert len(rows) == 2  # 1 instance x 2 ablations x 1 seed x 1 time-limit.
    assert all(set(MATRIX_RESULT_COLUMNS) <= set(row) for row in rows)
    assert {str(row["ablation"]) for row in rows} == {"scip_default", "full_opop"}
    assert {str(row["method"]) for row in rows} == {"scip-default", "opop"}
    assert (tmp_path / "repro_manifest.json").is_file()
    # full_opop runs the closed loop -> its events are aggregated to the top level.
    assert (tmp_path / "events.jsonl").is_file()
