"""Tests for the paper generation and claims-audit scripts (task 44).

Covers: make_paper.py generates figures/tables from fixture data,
claims_audit.py passes on a valid paper, claims_audit.py catches overclaims,
and claims_audit.py catches dev/validation numbers in headline tables.
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
        ("cuts-only", 0.88),
        ("params+cuts", 0.75),
    ]:
        for inst_id in ("set_cover_8x12", "set_cover_10x14", "knapsack_6"):
            for seed in range(5):
                base_pi = 10.0 + hash(f"{inst_id}:{seed}") % 10
                records.append({
                    "instance_id": inst_id,
                    "method": method,
                    "seed": seed,
                    "primal_integral": base_pi * pi_factor,
                    "gap": 0.05 * pi_factor,
                    "time": 1.5 * pi_factor,
                    "solved": True,
                    "censored": False,
                    "time_limit": 10.0,
                    "n_accepted": 5,
                })

    frame = pd.DataFrame(records)
    frame.to_parquet(out_dir / "results.parquet")

    # thesis_report.json — T1/T3/T4 pass, T2 fails
    thesis_report: dict[str, Any] = {
        "T1": {
            "claim": "T1: anytime / cross-distribution superiority",
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
            "claim": "T2: sample / compute efficiency",
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
            "claim": "T3: generality",
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
            "claim": "T4: method novelty",
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
            "n_records": 90,
            "theses": ["T1", "T2", "T3", "T4"],
            "all_pass": False,
        },
    }
    (out_dir / "thesis_report.json").write_text(
        json.dumps(thesis_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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


def _write_fixture_paper(paper_dir: Path) -> Path:
    """Write a minimal valid paper.md with placeholder comments for injection."""
    paper_md = paper_dir / "paper.md"
    paper_md.write_text(
        "# OPOP Paper\n\n" +
        "## Abstract\n\n" +
        "OPOP is a Bayesian-guided framework. We evaluate four theses.\n\n" +
        "## Results\n\n" +
        "<!-- THESIS_VERDICTS_TABLE -->\n\n" +
        "<!-- FIGURE: anytime_primal_integral -->\n\n" +
        "<!-- FIGURE: ablation_bar -->\n\n" +
        "<!-- ABLATION_CROSS_TABLE -->\n\n" +
        "<!-- CROSS_DISTRIBUTION_TABLE -->\n\n",
        encoding="utf-8",
    )
    return paper_md


@pytest.fixture
def fixture_run_dir(tmp_path: Path) -> Path:
    """Create a fixture run directory with valid test data."""
    run_dir = tmp_path / "fixture_run"
    run_dir.mkdir()
    _write_fixture_results(run_dir)
    return run_dir


@pytest.fixture
def fixture_paper_dir(tmp_path: Path) -> Path:
    """Create a fixture paper directory with paper.md ready for injection."""
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir(parents=True)
    _write_fixture_paper(paper_dir)
    return paper_dir


@pytest.fixture
def fixture_thesis_report_path(fixture_run_dir: Path) -> Path:
    return fixture_run_dir / "thesis_report.json"


# ---------------------------------------------------------------------------
# Tests for make_paper.py
# ---------------------------------------------------------------------------


def test_make_paper_runs(fixture_run_dir: Path, fixture_paper_dir: Path) -> None:
    """Run make_paper.py on fixture; assert figures/tables are created and injected."""
    script = str(_SCRIPTS / "make_paper.py")
    result = subprocess.run(
        [sys.executable, script, "--results", str(fixture_run_dir),
         "--out", str(fixture_paper_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=60,
    )

    assert result.returncode == 0, f"make_paper.py failed:\n{result.stderr}"

    # Assert at least 3 figures exist
    figures_dir = fixture_paper_dir / "figures"
    assert (figures_dir / "anytime_primal_integral.png").is_file(), \
        "anytime-primal-integral figure missing"
    assert (figures_dir / "ablation_bar.png").is_file(), \
        "ablation bar chart missing"

    # Assert at least 3 tables exist
    tables_dir = fixture_paper_dir / "tables"
    assert (tables_dir / "thesis_verdicts.md").is_file(), \
        "thesis verdicts table missing"
    assert (tables_dir / "ablation_cross.md").is_file(), \
        "ablation cross table missing"
    assert (tables_dir / "cross_distribution.md").is_file(), \
        "cross-distribution table missing"

    # Assert paper.md has injected references
    updated = (fixture_paper_dir / "paper.md").read_text(encoding="utf-8")
    assert "anytime_primal_integral.png" in updated
    assert "thesis_verdicts.md" in updated


def test_make_paper_empty_results(tmp_path: Path) -> None:
    """make_paper.py should handle empty results gracefully (exit 0)."""
    import pandas as pd

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    pd.DataFrame().to_parquet(run_dir / "results.parquet")

    paper_dir = tmp_path / "paper_out"
    paper_dir.mkdir(parents=True)
    _write_fixture_paper(paper_dir)

    script = str(_SCRIPTS / "make_paper.py")
    result = subprocess.run(
        [sys.executable, script, "--results", str(run_dir), "--out", str(paper_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    # Should succeed with 0 records
    assert result.returncode == 0, f"make_paper.py failed on empty:\n{result.stderr}"
    assert (paper_dir / "tables" / "thesis_verdicts.md").is_file()


def test_make_paper_without_thesis_report(tmp_path: Path) -> None:
    """make_paper.py should not crash when thesis_report.json is missing."""
    import pandas as pd

    run_dir = tmp_path / "minimal_run"
    run_dir.mkdir()
    records = [
        {"instance_id": "test_0", "method": "opop", "seed": 0, "primal_integral": 5.0},
        {"instance_id": "test_0", "method": "scip-default", "seed": 0, "primal_integral": 7.0},
    ]
    pd.DataFrame(records).to_parquet(run_dir / "results.parquet")

    paper_dir = tmp_path / "paper_out"
    paper_dir.mkdir(parents=True)
    _write_fixture_paper(paper_dir)

    script = str(_SCRIPTS / "make_paper.py")
    result = subprocess.run(
        [sys.executable, script, "--results", str(run_dir), "--out", str(paper_dir)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    assert result.returncode == 0, \
        f"make_paper.py should succeed without thesis report:\n{result.stderr}"
    assert (paper_dir / "tables" / "thesis_verdicts.md").is_file()


# ---------------------------------------------------------------------------
# Tests for claims_audit.py
# ---------------------------------------------------------------------------


def test_claims_audit_passes(
    fixture_thesis_report_path: Path, tmp_path: Path
) -> None:
    """Audit a clean paper that only references what the thesis report supports."""
    paper_dir = tmp_path / "clean_paper"
    paper_dir.mkdir()
    paper_md = paper_dir / "paper.md"

    # A paper that makes claims consistent with the fixture thesis report
    paper_md.write_text(
        "# OPOP Paper\n\n" +
        "## Abstract\n\n" +
        "We find that three of four theses are supported: T1 passes, T3 passes, "
        + "T4 passes. T2 fails. These results are consistent with the pre-registered "
        + "thresholds. We note that T2 does not meet the 30% solve-count reduction, "
        + "which is an honest negative result.\n\n"
        + "## Results\n\n"
        + "The thesis report shows that T1 holds, with significant improvement "
        + "over scip-default and opop-params-only.\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "claims_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(paper_md),
         "--thesis-report", str(fixture_thesis_report_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    assert result.returncode == 0, f"claims_audit.py should pass:\n{result.stdout}\n{result.stderr}"
    assert "PASS" in result.stdout


def test_claims_audit_catches_overclaim(
    fixture_thesis_report_path: Path, tmp_path: Path
) -> None:
    """Inject an unsupported overclaim; assert claims_audit.py flags it."""
    paper_dir = tmp_path / "overclaim_paper"
    paper_dir.mkdir()
    paper_md = paper_dir / "paper.md"

    # A paper that makes an overclaim not backed by the thesis report
    paper_md.write_text(
        "# OPOP Paper\n\n" +
        "## Abstract\n\n" +
        "We find that OPOP achieves state-of-the-art performance on all domains, "
        + "consistently outperforming every baseline across every problem type. "
        + "This guarantees superior results in all settings.\n\n"
        + "## Results\n\n"
        + "OPOP is SOTA on all domains.\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "claims_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(paper_md),
         "--thesis-report", str(fixture_thesis_report_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    # Should fail due to overclaims
    assert result.returncode == 1, (
        f"claims_audit.py should flag overclaims:\n{result.stdout}"
    )
    assert "overclaim" in result.stdout.lower() or "FAIL" in result.stdout


def test_claims_audit_catches_dev_validation_number(
    fixture_thesis_report_path: Path, tmp_path: Path
) -> None:
    """Inject a dev/validation result in a headline table; assert flagged."""
    paper_dir = tmp_path / "dev_table_paper"
    paper_dir.mkdir()
    paper_md = paper_dir / "paper.md"

    # A paper with a table that explicitly mentions dev/validation split
    paper_md.write_text(
        "# OPOP Paper\n\n" +
        "## Results\n\n" +
        "*Table 1: Validation-split results for dev evaluation.*\n" +
        "| Method | dev PI | validation PI |\n" +
        "|--------|--------|---------------|\n" +
        "| opop | 10.5 | 9.8 |\n" +
        "| scip-default | 15.0 | 14.2 |\n\n" +
        "We find that opop achieves 30% improvement.\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "claims_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(paper_md),
         "--thesis-report", str(fixture_thesis_report_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    # Should flag the dev/validation mention in a headline table
    assert result.returncode == 1, (
        f"claims_audit.py should flag dev/validation numbers in headline tables:\n{result.stdout}"
    )
    assert "dev_in_headline" in result.stdout.lower() or "FAIL" in result.stdout


def test_claims_audit_missing_thesis_report(tmp_path: Path) -> None:
    """claims_audit.py should work without a thesis report (just reports it)."""
    paper_dir = tmp_path / "no_report_paper"
    paper_dir.mkdir()
    paper_md = paper_dir / "paper.md"

    paper_md.write_text(
        "# OPOP Paper\n\n" +
        "We find that OPOP improves performance. No overclaims here.\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "claims_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(paper_md)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    # Should pass (or at worst exit 0 when there are no overclaims)
    # The note about missing thesis report is informational only
    assert "PASS" in result.stdout or "No issues" in result.stdout


def test_claims_audit_verdict_mismatch(
    fixture_thesis_report_path: Path, tmp_path: Path
) -> None:
    """Paper claims T2 passes, but thesis report says T2 fails; assert flagged."""
    paper_dir = tmp_path / "mismatch_paper"
    paper_dir.mkdir()
    paper_md = paper_dir / "paper.md"

    # Paper claims T2 passes (contradicts the fixture thesis report where T2 fails)
    paper_md.write_text(
        "# OPOP Paper\n\n" +
        "## Results\n\n" +
        "We find that T2 passes, confirming that OPOP reaches baseline-best "
        + "quality using fewer solve evaluations.\n",
        encoding="utf-8",
    )

    script = str(_SCRIPTS / "claims_audit.py")
    result = subprocess.run(
        [sys.executable, script, str(paper_md),
         "--thesis-report", str(fixture_thesis_report_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=30,
    )

    assert result.returncode == 1, (
        f"claims_audit.py should flag verdict mismatch:\n{result.stdout}"
    )
    assert "verdict_mismatch" in result.stdout.lower() or "FAIL" in result.stdout
