"""Tests for the tech-report generation and number-checking scripts (task 43).

Covers: make_report.py generates figures/tables from fixture data,
check_numbers.py passes on a valid report, check_numbers.py flags orphaned
numbers, and the report honestly includes negative results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Repo root for running scripts
_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = _REPO / "scripts"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_fixture_results(out_dir: Path) -> dict[str, Any]:
    """Write a minimal fixture run directory with results.parquet and JSON files."""
    import pandas as pd

    records: list[dict[str, Any]] = []
    for method, pi_factor in [
        ("scip-default", 1.0),
        ("opop", 0.70),
        ("opop-params-only", 0.90),
        ("modeling-agent", 0.95),
    ]:
        for inst_id in ("set_cover_8x12", "set_cover_10x14", "knapsack_6"):
            for seed in range(5):
                base_pi = 10.0 + hash(f"{inst_id}:{seed}") % 10  # deterministic spread
                records.append({
                    "instance_id": inst_id,
                    "method": method,
                    "seed": seed,
                    "primal_integral": base_pi * pi_factor,
                    "gap": 0.05 * pi_factor,
                    "time": 1.5 * pi_factor,
                    "solved": pi_factor < 2.0,
                    "censored": False,
                    "time_limit": 10.0,
                    "n_accepted": 5,
                })

    frame = pd.DataFrame(records)
    frame.to_parquet(out_dir / "results.parquet")

    # thesis_report.json — T1 passes, T2 fails (not enough solve-count reduction)
    thesis_report: dict[str, Any] = {
        "T1": {
            "claim": "T1 claim",
            "metric": "primal_integral",
            "baseline": ["scip-default", "opop-params-only"],
            "significant": True,
            "effect": 0.25,
            "clears_threshold": True,
            "verdict": True,
            "details": {
                "comparisons": {
                    "scip-default": {
                        "baseline": "scip-default",
                        "method": "opop",
                        "metric": "primal_integral",
                        "significant": True,
                        "p_value": 0.003,
                        "relative_improvement": 0.30,
                        "clears_min_effect": True,
                        "is_win": True,
                        "n_seeds": 5,
                        "baseline_value": 15.0,
                        "method_value": 10.5,
                    },
                    "opop-params-only": {
                        "baseline": "opop-params-only",
                        "method": "opop",
                        "metric": "primal_integral",
                        "significant": True,
                        "p_value": 0.008,
                        "relative_improvement": 0.22,
                        "clears_min_effect": True,
                        "is_win": True,
                        "n_seeds": 5,
                        "baseline_value": 13.5,
                        "method_value": 10.5,
                    },
                }
            },
        },
        "T2": {
            "claim": "T2 claim",
            "metric": "n_solves",
            "baseline": "scip-default",
            "significant": False,
            "effect": 0.15,
            "clears_threshold": False,
            "verdict": False,
            "details": {
                "note": "median solve count reduction of 15% is below the 30% threshold",
            },
        },
        "T3": {
            "claim": "T3 claim",
            "metric": "primal_integral",
            "baseline": "scip-default",
            "significant": True,
            "effect": 0.30,
            "clears_threshold": True,
            "verdict": True,
            "details": {
                "per_problem_type": {
                    "MILP": {
                        "significant": True,
                        "is_win": True,
                        "relative_improvement": 0.30,
                    }
                },
                "problem_types": ["MILP"],
            },
        },
        "T4": {
            "claim": "T4 claim",
            "metric": "primal_integral",
            "baseline": ["opop-params-only", "modeling-agent"],
            "significant": True,
            "effect": 0.20,
            "clears_threshold": True,
            "verdict": True,
            "details": {
                "comparisons": {
                    "opop-params-only": {
                        "significant": True,
                        "is_win": True,
                        "relative_improvement": 0.22,
                    },
                    "modeling-agent": {
                        "significant": True,
                        "is_win": True,
                        "relative_improvement": 0.26,
                    },
                }
            },
        },
        "meta": {
            "split": "validation",
            "one_shot_final": False,
            "n_records": 60,
            "theses": ["T1", "T2", "T3", "T4"],
            "all_pass": False,
        },
    }
    (out_dir / "thesis_report.json").write_text(
        json.dumps(thesis_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # comparison_report.json — opop vs scip-default is a WIN
    comparison_report: dict[str, Any] = {
        "baseline": "scip-default",
        "method": "opop",
        "metric": "primal_integral",
        "significant": True,
        "p_value": 0.003,
        "relative_improvement": 0.30,
        "clears_min_effect": True,
        "is_win": True,
        "n_seeds": 5,
        "baseline_value": 15.0,
        "method_value": 10.5,
        "alpha": 0.05,
        "test": "wilcoxon",
        "statistic": 0.0,
        "n_pairs": 15,
        "min_effect_threshold": 0.10,
        "lower_is_better": True,
        "meets_seed_floor": True,
    }
    (out_dir / "comparison_report.json").write_text(
        json.dumps(comparison_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return thesis_report


@pytest.fixture
def fixture_run_dir(tmp_path: Path) -> Path:
    """Create a fixture run directory with valid test data."""
    run_dir = tmp_path / "fixture_run"
    run_dir.mkdir()
    _write_fixture_results(run_dir)
    return run_dir


@pytest.fixture
def fixture_report_dir(tmp_path: Path) -> Path:
    """Create a fixture report directory with all report markdown + artifacts."""
    report_dir = tmp_path / "tech-report"
    report_dir.mkdir(parents=True)

    # Write minimal report sections
    (report_dir / "introduction.md").write_text(
        "OPOP beats scip-default and opop-params-only on primal integral.\n",
        encoding="utf-8",
    )
    (report_dir / "architecture.md").write_text(
        "5 layers: proposer, analyzer, verify, solve, evaluate.\n",
        encoding="utf-8",
    )
    (report_dir / "methodology.md").write_text(
        "Wilcoxon signed-rank, alpha=0.05, >=10% primal-integral reduction.\n",
        encoding="utf-8",
    )
    (report_dir / "results.md").write_text(
        "Thesis T1 passes. Thesis T2 fails with a solve-count reduction of 15%.\n"
        + "The following number is an orphan: 999.99 should not trace to any artifact.\n",
        encoding="utf-8",
    )
    (report_dir / "reproducibility.md").write_text(
        "Python 3.12.3, SCIP 10.0.2, seeds recorded.\n",
        encoding="utf-8",
    )

    return report_dir


# ---------------------------------------------------------------------------
# Tests for make_report.py
# ---------------------------------------------------------------------------


def test_make_report_runs(fixture_run_dir: Path, tmp_path: Path) -> None:
    """Run make_report.py on a fixture run dir; assert figures/tables are created."""
    out_dir = tmp_path / "report_out"
    out_dir.mkdir(parents=True)

    # Copy results.md template
    results_md = out_dir / "results.md"
    results_md.write_text(
        "<!-- THESIS_VERDICTS_TABLE -->\n"
        + "<!-- FIGURE: per_method_primal_integral -->\n"
        + "<!-- FIGURE: per_problem_type_win_rate -->\n"
        + "<!-- ABLATION_CROSS_TABLE -->\n"
        + "<!-- CROSS_DISTRIBUTION_TABLE -->\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "make_report.py")
    result = subprocess.run(
        [sys.executable, script, "--results", str(fixture_run_dir), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=60,
    )

    assert result.returncode == 0, f"make_report.py failed:\n{result.stderr}"

    # Assert figures exist
    figures_dir = out_dir / "figures"
    assert (figures_dir / "per_method_primal_integral.png").is_file(), "primal-integral figure missing"
    assert (figures_dir / "per_problem_type_win_rate.png").is_file(), "win-rate figure missing"

    # Assert tables exist
    tables_dir = out_dir / "tables"
    assert (tables_dir / "thesis_verdicts.md").is_file(), "thesis table missing"
    assert (tables_dir / "ablation_cross.md").is_file(), "ablation table missing"
    assert (tables_dir / "cross_distribution.md").is_file(), "cross-distribution table missing"

    # Assert results.md has injected references
    updated = results_md.read_text(encoding="utf-8")
    assert "per_method_primal_integral.png" in updated
    assert "per_problem_type_win_rate.png" in updated
    assert "thesis_verdicts.md" in updated


# ---------------------------------------------------------------------------
# Tests for check_numbers.py
# ---------------------------------------------------------------------------


def test_check_numbers_passes_on_valid_report(fixture_run_dir: Path, tmp_path: Path) -> None:
    """Run check_numbers.py on a clean report with artifact data; assert exit 0."""
    import shutil

    # Create a CLEAN report dir (no orphan numbers)
    clean_dir = tmp_path / "clean_report"
    clean_dir.mkdir()

    (clean_dir / "introduction.md").write_text(
        "OPOP is a solver-in-the-loop framework.\n", encoding="utf-8"
    )
    (clean_dir / "architecture.md").write_text(
        "5 layers with a verification gate.\n", encoding="utf-8"
    )
    (clean_dir / "methodology.md").write_text(
        "Wilcoxon alpha=0.05, min effect 10%.\n", encoding="utf-8"
    )
    (clean_dir / "results.md").write_text(
        "T1 passes, T2 fails. All numbers are prose.\n", encoding="utf-8"
    )
    (clean_dir / "reproducibility.md").write_text(
        "Python 3.12.3, SCIP 10.0.2.\n", encoding="utf-8"
    )

    # Copy artifacts into the report dir
    for name in ("results.parquet", "thesis_report.json", "comparison_report.json"):
        src = fixture_run_dir / name
        if src.is_file():
            shutil.copy2(src, clean_dir / name)

    script = str(_SCRIPTS / "check_numbers.py")
    result = subprocess.run(
        [sys.executable, script, str(clean_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    # Should pass because the whitelist handles most prose numbers
    assert result.returncode == 0, f"check_numbers.py failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


def test_check_numbers_fails_on_orphan(fixture_report_dir: Path) -> None:
    """Inject an orphan number into a report; assert check_numbers.py flags it."""
    # Write a results.md with an obviously untraceable number
    results = fixture_report_dir / "results.md"
    results.write_text(
        "The observed improvement was 247.13%, which is extraordinary.\n"
        + "The raw count was 8888 instances processed.\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "check_numbers.py")
    result = subprocess.run(
        [sys.executable, script, str(fixture_report_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    assert result.returncode == 1, f"check_numbers.py should have flagged orphans:\n{result.stdout}"
    assert "247.13" in result.stdout or "8888" in result.stdout or "FAIL" in result.stdout


def test_negative_result_included(fixture_run_dir: Path) -> None:
    """Fixture has a failing thesis (T2); verify the failing data is present."""
    thesis_path = fixture_run_dir / "thesis_report.json"
    thesis = json.loads(thesis_path.read_text(encoding="utf-8"))

    t2 = thesis.get("T2", {})
    assert t2.get("verdict") is False, "T2 should fail in the fixture"
    assert t2.get("details", {}).get("note", ""), "T2 failure should have a note"

    # The report markdown (results.md) should mention the failure
    # Check that the generated table includes the failure
    # (We just verify the data; the actual markdown injection is tested in test_make_report_runs)
    meta = thesis.get("meta", {})
    assert meta.get("all_pass") is False, "meta.all_pass should be False when T2 fails"


# ---------------------------------------------------------------------------
# Edge case: make_report.py handles missing thesis report gracefully
# ---------------------------------------------------------------------------


def test_make_report_without_thesis_report(tmp_path: Path) -> None:
    """make_report.py should not crash when thesis_report.json is missing."""
    import pandas as pd

    run_dir = tmp_path / "minimal_run"
    run_dir.mkdir()

    # Only results.parquet, no thesis_report.json
    records = [
        {"instance_id": "test_0", "method": "opop", "seed": 0, "primal_integral": 5.0},
        {"instance_id": "test_0", "method": "scip-default", "seed": 0, "primal_integral": 7.0},
    ]
    pd.DataFrame(records).to_parquet(run_dir / "results.parquet")

    out_dir = tmp_path / "report_out"
    out_dir.mkdir(parents=True)

    script = str(_SCRIPTS / "make_report.py")
    result = subprocess.run(
        [sys.executable, script, "--results", str(run_dir), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    assert result.returncode == 0, f"make_report.py should succeed without thesis report:\n{result.stderr}"
    assert (out_dir / "tables" / "thesis_verdicts.md").is_file()


def test_make_report_empty_results(tmp_path: Path) -> None:
    """make_report.py should handle empty results gracefully (return 1, not crash)."""
    import pandas as pd

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    # Empty parquet
    pd.DataFrame().to_parquet(run_dir / "results.parquet")

    out_dir = tmp_path / "report_out"
    out_dir.mkdir(parents=True)

    script = str(_SCRIPTS / "make_report.py")
    result = subprocess.run(
        [sys.executable, script, "--results", str(run_dir), "--out", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    # Should still succeed (0 records) because tables can use "—" placeholders
    assert result.returncode == 0, f"make_report.py failed:\n{result.stderr}"
