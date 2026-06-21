"""End-to-end smoke test for the Phase-1 run module (plan task 21).

Drives ``python -m opop.run --config configs/phase1_smoke.yaml --out <tmp>`` as a
subprocess and asserts the closed loop + SCIP-default baseline produced ALL six
experiment artifacts, that the leakage audit passed, that the comparison report
is well-formed, that at least one opop row certified-and-accepted a delta
(``n_accepted >= 1``), and that the run strict-replayed the first instance from
disk. Marked ``integration`` (real SCIP solves) and skipped when SCIP / pandas /
pyarrow are unavailable. The committed smoke config keeps it small (5 instances x
5 seeds, 2 trials, 5s limit) so it finishes in well under two minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_CONFIG = _REPO_ROOT / "configs" / "phase1_smoke.yaml"

#: The machine-readable comparison-report schema the run must emit.
_COMPARISON_REQUIRED_FIELDS = {
    "baseline",
    "method",
    "metric",
    "significant",
    "p_value",
    "relative_improvement",
    "clears_min_effect",
    "is_win",
    "n_seeds",
    "baseline_value",
    "method_value",
}


@pytest.mark.integration
def test_phase1_smoke_run_emits_all_artifacts(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """The smoke run exits 0 and writes every Phase-1 artifact with a certified delta."""
    solver_skip_if_missing("scip")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    out_dir = tmp_path / "smoke"
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    completed = subprocess.run(
        [sys.executable, "-m", "opop.run", "--config", str(_CONFIG), "--out", str(out_dir)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"run failed (rc={completed.returncode})\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )

    # All six artifacts exist.
    assert (out_dir / "results.parquet").is_file()
    assert (out_dir / "events.jsonl").is_file()
    assert (out_dir / "repro_manifest.json").is_file()
    assert (out_dir / "comparison_report.json").is_file()
    assert (out_dir / "leakage_audit.json").is_file()
    verification_reports = list((out_dir / "verification").glob("*.json"))
    assert verification_reports, "no per-delta verification certificates were written"

    # Leakage audit passed (Phase-1 declares no held-out splits).
    audit = json.loads((out_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["n_violations"] == 0

    # The comparison report carries the full machine-readable schema.
    report = json.loads((out_dir / "comparison_report.json").read_text(encoding="utf-8"))
    assert _COMPARISON_REQUIRED_FIELDS <= set(report)
    assert report["baseline"] == "scip-default"
    assert report["method"] == "opop"
    assert report["metric"] == "primal_integral"

    # The top-level reproducibility summary describes the run.
    manifest = json.loads((out_dir / "repro_manifest.json").read_text(encoding="utf-8"))
    assert manifest["plan_name"] == "phase1_smoke"
    assert manifest["n_instances"] >= 1
    assert manifest["instances"]

    # At least one opop row certified + accepted a delta.
    frame = pd.read_parquet(out_dir / "results.parquet")
    opop_rows = frame[frame["method"] == "opop"]
    baseline_rows = frame[frame["method"] == "scip-default"]
    assert not opop_rows.empty
    assert not baseline_rows.empty
    assert int(cast("int", opop_rows["n_accepted"].max())) >= 1

    # The run strict-replayed the first instance from disk.
    assert "strict replay reproduced" in completed.stdout
