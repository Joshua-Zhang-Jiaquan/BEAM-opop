"""Tests for the leaderboard builder + submission validator (task 42).

Covers:
- LeaderboardBuilder builds from fixture records and produces HTML with expected
  columns, methodology section, and limitations section.
- Headline table excludes dev/validation rows.
- SubmissionValidator accepts a complete fixture run and rejects a run missing
  ``repro_manifest.json``.
- CLI ``build`` and ``submit`` subcommands.
- Bootstrap CI produces sensible bounds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opop.leaderboard.builder import LeaderboardBuilder, LeaderboardData, _bootstrap_ci
from opop.leaderboard.submit import SubmissionResult, SubmissionValidator


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _rec(
    method: str,
    instance: str,
    seed: int,
    *,
    pi: float = 1.0,
    time: float = 1.0,
    solved: bool = True,
    split: str = "test",
) -> dict[str, Any]:
    """Build one result record."""
    return {
        "instance_id": instance,
        "method": method,
        "seed": seed,
        "primal_integral": pi,
        "gap": 0.0,
        "time": time,
        "solved": solved,
        "censored": False,
        "split": split,
    }


def _write_fixture_results(run_dir: Path, *, split: str = "test") -> Path:
    """Write a small results.json to a run directory."""
    records = [
        _rec("opop", "inst0", 0, pi=5.0, time=2.0, solved=True, split=split),
        _rec("opop", "inst0", 1, pi=4.5, time=2.5, solved=True, split=split),
        _rec("opop", "inst1", 0, pi=6.0, time=3.0, solved=True, split=split),
        _rec("opop", "inst1", 1, pi=5.5, time=2.8, solved=False, split=split),
        _rec("scip-default", "inst0", 0, pi=10.0, time=4.0, solved=True, split=split),
        _rec("scip-default", "inst0", 1, pi=9.5, time=4.5, solved=True, split=split),
        _rec("scip-default", "inst1", 0, pi=11.0, time=5.0, solved=False, split=split),
        _rec("scip-default", "inst1", 1, pi=10.5, time=4.8, solved=False, split=split),
    ]
    path = run_dir / "results.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return path


def _write_thesis_report(run_dir: Path) -> Path:
    """Write a minimal thesis_report.json."""
    report = {
        "T1": {
            "claim": "opop beats scip-default on primal_integral.",
            "metric": "primal_integral",
            "baseline": "scip-default",
            "significant": True,
            "effect": 0.45,
            "clears_threshold": True,
            "verdict": True,
            "details": {},
        },
        "T2": {
            "claim": "opop uses fewer solves.",
            "metric": "n_solves",
            "baseline": "scip-default",
            "significant": True,
            "effect": 0.35,
            "clears_threshold": True,
            "verdict": True,
            "details": {},
        },
        "T3": {
            "claim": "opop wins on every problem type.",
            "metric": "primal_integral",
            "baseline": "scip-default",
            "significant": True,
            "effect": 0.40,
            "clears_threshold": True,
            "verdict": True,
            "details": {},
        },
        "T4": {
            "claim": "opop beats params-only and modeling-agent.",
            "metric": "primal_integral",
            "baseline": ["opop-params-only", "modeling-agent"],
            "significant": False,
            "effect": 0.08,
            "clears_threshold": False,
            "verdict": False,
            "details": {},
        },
        "meta": {
            "split": "test",
            "one_shot_final": True,
            "n_records": 8,
            "theses": ["T1", "T2", "T3", "T4"],
            "all_pass": False,
        },
    }
    path = run_dir / "thesis_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_complete_run(tmp_path: Path) -> Path:
    """Create a complete fixture run directory with all required artifacts."""
    run_dir = tmp_path / "complete_run"
    run_dir.mkdir()
    _write_fixture_results(run_dir)
    _write_thesis_report(run_dir)
    # repro_manifest.json
    (run_dir / "repro_manifest.json").write_text(
        json.dumps({"split": "test", "n_rows": 8}) + "\n", encoding="utf-8"
    )
    # leakage_audit.json (passing)
    (run_dir / "leakage_audit.json").write_text(
        json.dumps({
            "status": "pass",
            "test_instances_used_for_tuning": [],
            "ood_instances_used_for_tuning": [],
            "n_violations": 0,
        }) + "\n",
        encoding="utf-8",
    )
    return run_dir


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #


class TestBootstrapCI:
    def test_empty_values(self) -> None:
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_single_value(self) -> None:
        lo, hi = _bootstrap_ci([5.0])
        assert lo == 5.0
        assert hi == 5.0

    def test_multiple_values_bracket_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = _bootstrap_ci(values)
        mean = sum(values) / len(values)
        assert lo <= mean <= hi
        assert lo < hi  # non-degenerate CI


# --------------------------------------------------------------------------- #
# LeaderboardBuilder
# --------------------------------------------------------------------------- #


class TestLeaderboardBuilder:
    def test_build_produces_data(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        builder = LeaderboardBuilder(run_dir / "results.json")
        data = builder.build()
        assert isinstance(data, LeaderboardData)
        assert len(data.rows) > 0

    def test_headline_excludes_dev(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Write records with mixed splits.
        records = [
            _rec("opop", "inst0", 0, pi=5.0, split="test"),
            _rec("opop", "inst1", 0, pi=6.0, split="dev"),
            _rec("scip-default", "inst0", 0, pi=10.0, split="test"),
            _rec("scip-default", "inst1", 0, pi=11.0, split="dev"),
        ]
        path = run_dir / "results.json"
        path.write_text(json.dumps(records) + "\n", encoding="utf-8")
        builder = LeaderboardBuilder(path)
        data = builder.build()
        headline = data.headline_rows()
        for row in headline:
            assert row.split in {"test", "ood_test"}
        # Dev rows excluded from headline.
        headline_splits = {r.split for r in headline}
        assert "dev" not in headline_splits

    def test_html_contains_required_columns(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        _write_thesis_report(run_dir)
        builder = LeaderboardBuilder(
            run_dir / "results.json",
            thesis_path=run_dir / "thesis_report.json",
        )
        out_dir = tmp_path / "site"
        html_path, md_path = builder.write(out_dir)

        assert html_path.is_file()
        assert md_path.is_file()

        html_content = html_path.read_text(encoding="utf-8")
        # Required columns
        assert "Method" in html_content
        assert "Split" in html_content
        assert "Primal Int." in html_content
        assert "Solved Rate" in html_content
        assert "Time (SGM)" in html_content
        assert "95% CI" in html_content

    def test_html_contains_methodology_section(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        builder = LeaderboardBuilder(run_dir / "results.json")
        out_dir = tmp_path / "site"
        html_path, _ = builder.write(out_dir)
        html_content = html_path.read_text(encoding="utf-8")
        assert "Methodology" in html_content
        assert "Primal Integral" in html_content
        assert "shifted geometric mean" in html_content

    def test_html_contains_limitations_section(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        builder = LeaderboardBuilder(run_dir / "results.json")
        out_dir = tmp_path / "site"
        html_path, _ = builder.write(out_dir)
        html_content = html_path.read_text(encoding="utf-8")
        assert "Limitations" in html_content
        assert "Leakage Policy" in html_content
        assert "held-out" in html_content

    def test_html_contains_thesis_panel(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        _write_thesis_report(run_dir)
        builder = LeaderboardBuilder(
            run_dir / "results.json",
            thesis_path=run_dir / "thesis_report.json",
        )
        out_dir = tmp_path / "site"
        html_path, _ = builder.write(out_dir)
        html_content = html_path.read_text(encoding="utf-8")
        assert "T1" in html_content
        assert "T2" in html_content
        assert "T3" in html_content
        assert "T4" in html_content
        assert "PASS" in html_content
        assert "FAIL" in html_content

    def test_markdown_fallback(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        builder = LeaderboardBuilder(run_dir / "results.json")
        out_dir = tmp_path / "site"
        _, md_path = builder.write(out_dir)
        md_content = md_path.read_text(encoding="utf-8")
        assert "# OPOP Leaderboard" in md_content
        assert "Methodology" in md_content
        assert "Limitations" in md_content

    def test_aggregation_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir, split="test")
        builder = LeaderboardBuilder(run_dir / "results.json")
        data = builder.build()
        # Find the opop row.
        opop_rows = [r for r in data.rows if r.method == "opop"]
        assert len(opop_rows) == 1
        row = opop_rows[0]
        assert row.n_instances == 2
        assert row.n_seeds == 2
        # Mean PI: (5.0 + 4.5 + 6.0 + 5.5) / 4 = 5.25
        assert abs(row.primal_integral_mean - 5.25) < 1e-6
        # Solved rate: 3/4 = 0.75
        assert abs(row.solved_rate - 0.75) < 1e-6


# --------------------------------------------------------------------------- #
# SubmissionValidator
# --------------------------------------------------------------------------- #


class TestSubmissionValidator:
    def test_accepts_complete_run(self, tmp_path: Path) -> None:
        run_dir = _make_complete_run(tmp_path)
        validator = SubmissionValidator()
        result = validator.validate(run_dir)
        assert isinstance(result, SubmissionResult)
        assert result.accepted is True
        assert "all artifacts present" in result.reason

    def test_rejects_missing_repro_manifest(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "incomplete_run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        # leakage_audit.json present
        (run_dir / "leakage_audit.json").write_text(
            json.dumps({"status": "pass", "n_violations": 0}) + "\n",
            encoding="utf-8",
        )
        # NO repro_manifest.json
        validator = SubmissionValidator()
        result = validator.validate(run_dir)
        assert result.accepted is False
        assert "repro_manifest.json" in result.reason

    def test_rejects_missing_leakage_audit(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_audit_run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        (run_dir / "repro_manifest.json").write_text(
            json.dumps({"split": "test"}) + "\n", encoding="utf-8"
        )
        # NO leakage_audit.json
        validator = SubmissionValidator()
        result = validator.validate(run_dir)
        assert result.accepted is False
        assert "leakage_audit.json" in result.reason

    def test_rejects_missing_results(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_results_run"
        run_dir.mkdir()
        (run_dir / "repro_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "leakage_audit.json").write_text(
            json.dumps({"status": "pass", "n_violations": 0}) + "\n",
            encoding="utf-8",
        )
        # NO results file
        validator = SubmissionValidator()
        result = validator.validate(run_dir)
        assert result.accepted is False
        assert "results file" in result.reason

    def test_rejects_failing_leakage_audit(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "failed_audit_run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        (run_dir / "repro_manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "leakage_audit.json").write_text(
            json.dumps({
                "status": "fail",
                "test_instances_used_for_tuning": ["inst0"],
                "ood_instances_used_for_tuning": [],
                "n_violations": 1,
            }) + "\n",
            encoding="utf-8",
        )
        validator = SubmissionValidator()
        result = validator.validate(run_dir)
        assert result.accepted is False
        assert "FAIL" in result.reason

    def test_rejects_nonexistent_directory(self, tmp_path: Path) -> None:
        validator = SubmissionValidator()
        result = validator.validate(tmp_path / "nonexistent")
        assert result.accepted is False
        assert "does not exist" in result.reason

    def test_to_dict(self, tmp_path: Path) -> None:
        run_dir = _make_complete_run(tmp_path)
        validator = SubmissionValidator()
        result = validator.validate(run_dir)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "accepted" in d
        assert "reason" in d
        assert "artifacts_checked" in d


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCLI:
    def test_build_cli(self, tmp_path: Path) -> None:
        from opop.leaderboard.__main__ import main

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_fixture_results(run_dir)
        out_dir = tmp_path / "site"
        rc = main(["build", "--results", str(run_dir), "--out", str(out_dir)])
        assert rc == 0
        assert (out_dir / "index.html").is_file()
        assert (out_dir / "leaderboard.md").is_file()

    def test_submit_cli_accepted(self, tmp_path: Path) -> None:
        from opop.leaderboard.__main__ import main

        run_dir = _make_complete_run(tmp_path)
        rc = main(["submit", "--run", str(run_dir)])
        assert rc == 0

    def test_submit_cli_rejected(self, tmp_path: Path) -> None:
        from opop.leaderboard.__main__ import main

        run_dir = tmp_path / "bad_run"
        run_dir.mkdir()
        rc = main(["submit", "--run", str(run_dir)])
        assert rc == 1

    def test_build_cli_missing_results(self, tmp_path: Path) -> None:
        from opop.leaderboard.__main__ import main

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rc = main(["build", "--results", str(empty_dir), "--out", str(tmp_path / "site")])
        assert rc == 1
