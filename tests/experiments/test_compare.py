"""Tests for the comparison report + statistical tests (task 18).

Covers: Wilcoxon p-value matches scipy, shifted-geometric-mean hand value,
min-effect gating, below-threshold improvement is NOT a win, censored-aware
time handling, the all-equal (no-difference) guard, JSON + parquet loading,
``comparison_report.json`` validity, and the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from scipy.stats import wilcoxon

from opop.experiments.compare import (
    DEFAULT_MIN_EFFECT,
    ComparisonReport,
    build_min_effect,
    compare,
    load_results,
    main,
    shifted_geometric_mean,
    write_report,
)

# Required ComparisonReport schema (the machine-readable contract).
_REQUIRED_FIELDS = {
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

# Distinct, all-positive baseline values -> uniform-factor methods give distinct
# positive paired differences (no ties), so scipy uses the EXACT test.
_BASELINE_PI = [10.0, 11.0, 9.0, 12.0, 8.0, 14.0]


def _rec(
    method: str,
    seed: int,
    *,
    instance: str = "inst0",
    pi: float = 1.0,
    gap: float = 0.0,
    time: float = 1.0,
    solved: bool = True,
    censored: bool = False,
    time_limit: float | None = None,
) -> dict[str, Any]:
    """Build one full result record (all task-13 fields)."""
    record: dict[str, Any] = {
        "instance_id": instance,
        "method": method,
        "seed": seed,
        "primal_integral": pi,
        "gap": gap,
        "time": time,
        "solved": solved,
        "censored": censored,
    }
    if time_limit is not None:
        record["time_limit"] = time_limit
    return record


def _pi_records(factor: float) -> list[dict[str, Any]]:
    """Paired PI records: ``opop`` is ``factor`` x the ``scip-default`` PI."""
    records: list[dict[str, Any]] = []
    for seed, value in enumerate(_BASELINE_PI):
        records.append(_rec("scip-default", seed, pi=value))
        records.append(_rec("opop", seed, pi=value * factor))
    return records


# --------------------------------------------------------------------------- #
# Shifted geometric mean
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_shifted_geomean_matches_hand_value() -> None:
    """``sg([6, 15], s=10) = sqrt(16*25) - 10 = 20 - 10 = 10`` exactly."""
    assert shifted_geometric_mean([6.0, 15.0]) == pytest.approx(10.0, abs=1e-9)
    # All-zeros: (10*10*10)^(1/3) - 10 == 0.
    assert shifted_geometric_mean([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    # A constant series returns the constant (sg([c]*n) == c).
    assert shifted_geometric_mean([42.0] * 5) == pytest.approx(42.0, abs=1e-9)
    # Empty -> 0.0 (no crash).
    assert shifted_geometric_mean([]) == 0.0


@pytest.mark.smoke
def test_shifted_geomean_three_value_hand_check() -> None:
    """``sg([1, 4, 10], s=10) = (11*14*20)^(1/3) - 10``."""
    expected = (11.0 * 14.0 * 20.0) ** (1.0 / 3.0) - 10.0
    assert shifted_geometric_mean([1.0, 4.0, 10.0]) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# Wilcoxon matches scipy
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_wilcoxon_pvalue_matches_scipy() -> None:
    """The report p-value and statistic equal scipy on the same paired data."""
    factor = 0.85
    report = compare(
        _pi_records(factor),
        baseline="scip-default",
        method="opop",
        metric="primal_integral",
    )
    scipy_stat, scipy_p = cast(
        "tuple[float, float]", wilcoxon(_BASELINE_PI, [v * factor for v in _BASELINE_PI])
    )
    assert report.p_value == pytest.approx(float(scipy_p))
    assert report.statistic == pytest.approx(float(scipy_stat))


# --------------------------------------------------------------------------- #
# Min-effect gating
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_primal_integral_15pct_is_a_win() -> None:
    """15% PI reduction across 6 seeds: significant, clears 10%, is a win."""
    report = compare(
        _pi_records(0.85),
        baseline="scip-default",
        method="opop",
        metric="primal_integral",
    )
    assert report.relative_improvement == pytest.approx(0.15, abs=1e-9)
    assert report.significant is True
    assert report.clears_min_effect is True
    assert report.is_win is True
    assert report.n_seeds == 6
    assert report.n_pairs == 6
    assert report.meets_seed_floor is True
    assert report.lower_is_better is True
    assert report.min_effect_threshold == pytest.approx(0.10)
    # The point aggregates are the means of the paired distributions.
    assert report.baseline_value == pytest.approx(sum(_BASELINE_PI) / 6)
    assert report.baseline_distribution["n"] == 6
    assert report.method_distribution["mean"] == pytest.approx(report.method_value)


@pytest.mark.smoke
def test_below_threshold_improvement_is_not_a_win() -> None:
    """3% PI reduction is significant but does NOT clear the 10% min-effect."""
    report = compare(
        _pi_records(0.97),
        baseline="scip-default",
        method="opop",
        metric="primal_integral",
    )
    assert report.relative_improvement == pytest.approx(0.03, abs=1e-9)
    assert report.significant is True  # statistically detectable
    assert report.clears_min_effect is False  # but below 10%
    assert report.is_win is False  # => not a win


@pytest.mark.smoke
def test_min_effect_override_blocks_a_win() -> None:
    """Raising the threshold above the observed effect turns a win into no-win."""
    records = _pi_records(0.85)  # 15% improvement
    strict = compare(
        records,
        baseline="scip-default",
        method="opop",
        metric="primal_integral",
        min_effect=build_min_effect("primal_integral", 0.50),
    )
    assert strict.min_effect_threshold == pytest.approx(0.50)
    assert strict.clears_min_effect is False
    assert strict.is_win is False


@pytest.mark.smoke
def test_worse_method_is_never_a_win() -> None:
    """A method that is significantly WORSE has negative improvement, no win."""
    report = compare(
        _pi_records(1.20),  # 20% worse (higher PI)
        baseline="scip-default",
        method="opop",
        metric="primal_integral",
    )
    assert report.relative_improvement < 0
    assert report.clears_min_effect is False
    assert report.is_win is False


# --------------------------------------------------------------------------- #
# Time metric (shifted geomean, censored-aware)
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_time_uses_shifted_geomean_and_clears_20pct() -> None:
    """Time aggregate is the shifted geomean; a ~30% cut clears the 20% gate."""
    b_times = [100.0, 120.0, 80.0, 110.0, 90.0, 140.0]
    records: list[dict[str, Any]] = []
    for seed, value in enumerate(b_times):
        records.append(_rec("base", seed, time=value))
        records.append(_rec("cand", seed, time=value * 0.7))
    report = compare(records, baseline="base", method="cand", metric="time")
    assert report.baseline_value == pytest.approx(shifted_geometric_mean(b_times))
    assert report.method_value == pytest.approx(
        shifted_geometric_mean([v * 0.7 for v in b_times])
    )
    expected_rel = (report.baseline_value - report.method_value) / report.baseline_value
    assert report.relative_improvement == pytest.approx(expected_rel)
    assert report.relative_improvement >= 0.20
    assert report.significant is True
    assert report.clears_min_effect is True
    assert report.is_win is True


@pytest.mark.smoke
def test_censored_runtime_is_lifted_to_time_limit() -> None:
    """Censored runtimes enter the time aggregate at the limit, not the cutoff stamp.

    Candidate seed 1 is censored at a 2s cutoff with a 100s limit (-> 100), seed 2
    is censored with no recorded limit (-> its 55s lower bound is used as-is), and
    seed 0 finished cleanly (-> its 8s runtime).
    """
    records = [
        _rec("base", 0, time=10.0),
        _rec("base", 1, time=10.0),
        _rec("base", 2, time=10.0),
        _rec("cand", 0, time=8.0, solved=True, censored=False),
        _rec("cand", 1, time=2.0, solved=False, censored=True, time_limit=100.0),
        _rec("cand", 2, time=55.0, solved=False, censored=True),
    ]
    report = compare(records, baseline="base", method="cand", metric="time")
    assert sorted(report.method_distribution["values"]) == pytest.approx([8.0, 55.0, 100.0])
    assert report.method_value == pytest.approx(shifted_geometric_mean([8.0, 100.0, 55.0]))


# --------------------------------------------------------------------------- #
# Solved-rate metric
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:Sample size too small:UserWarning")
def test_solved_rate_relative_improvement_is_absolute_pp() -> None:
    """solved_rate improvement is the absolute pp gain (method - baseline).

    The tied 0/1 solved indicators make scipy fall back to the (small-sample)
    normal approximation; that warning is expected here and is filtered.
    """
    solved_b = [True, True, True, False, False, False]  # 0.5
    solved_m = [True, True, True, True, True, True]  # 1.0
    records: list[dict[str, Any]] = []
    for seed, (sb, sm) in enumerate(zip(solved_b, solved_m, strict=True)):
        records.append(_rec("base", seed, solved=sb, censored=not sb))
        records.append(_rec("cand", seed, solved=sm, censored=not sm))
    report = compare(records, baseline="base", method="cand", metric="solved_rate")
    assert report.baseline_value == pytest.approx(0.5)
    assert report.method_value == pytest.approx(1.0)
    assert report.relative_improvement == pytest.approx(0.5)  # 50 pp, absolute
    assert report.lower_is_better is False
    assert report.clears_min_effect is True  # >= 5 pp
    # Only 3 non-zero paired diffs -> not significant -> not a win.
    assert report.significant is False
    assert report.is_win is False


@pytest.mark.smoke
def test_solved_rate_below_5pp_is_not_a_win() -> None:
    """A 3 pp solved-rate gain does not clear the 5 pp min-effect."""
    records: list[dict[str, Any]] = []
    for seed in range(100):
        sb = seed < 50  # baseline solves 50/100
        sm = seed < 53  # method solves 53/100 (the extra 3 are 50,51,52)
        records.append(_rec("base", seed, solved=sb, censored=not sb))
        records.append(_rec("cand", seed, solved=sm, censored=not sm))
    report = compare(records, baseline="base", method="cand", metric="solved_rate")
    assert report.baseline_value == pytest.approx(0.50)
    assert report.method_value == pytest.approx(0.53)
    assert report.relative_improvement == pytest.approx(0.03, abs=1e-9)
    assert report.clears_min_effect is False
    assert report.is_win is False
    assert report.n_pairs == 100


# --------------------------------------------------------------------------- #
# Guards / edge cases
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_all_equal_is_not_significant() -> None:
    """Identical paired values -> p == 1.0, not significant, not a win."""
    records: list[dict[str, Any]] = []
    for seed, value in enumerate(_BASELINE_PI):
        records.append(_rec("base", seed, pi=value))
        records.append(_rec("cand", seed, pi=value))
    report = compare(records, baseline="base", method="cand", metric="primal_integral")
    assert report.p_value == pytest.approx(1.0)
    assert report.significant is False
    assert report.relative_improvement == pytest.approx(0.0)
    assert report.is_win is False


@pytest.mark.smoke
def test_compare_validation_errors() -> None:
    """Unknown metric, self-comparison, and empty pairing all raise ValueError."""
    records = _pi_records(0.85)
    with pytest.raises(ValueError, match="unknown metric"):
        compare(records, baseline="scip-default", method="opop", metric="bogus")
    with pytest.raises(ValueError, match="different"):
        compare(records, baseline="opop", method="opop", metric="primal_integral")
    with pytest.raises(ValueError, match="no paired"):
        compare(records, baseline="scip-default", method="missing", metric="primal_integral")


@pytest.mark.smoke
def test_default_min_effect_thresholds() -> None:
    """The locked Win Definition thresholds are 10% / 20% / 5pp."""
    assert DEFAULT_MIN_EFFECT == {
        "primal_integral": 0.10,
        "time": 0.20,
        "solved_rate": 0.05,
    }


# --------------------------------------------------------------------------- #
# Loading + report serialisation
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_load_results_json_and_report_is_valid(tmp_path: Path) -> None:
    """Round-trip through a JSON file and a written comparison_report.json."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_pi_records(0.85)), encoding="utf-8")

    records = load_results(results_path)
    report = compare(
        records, baseline="scip-default", method="opop", metric="primal_integral"
    )
    out = write_report(report, tmp_path / "comparison_report.json")
    assert out.exists()

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert _REQUIRED_FIELDS <= set(loaded)
    assert loaded["baseline"] == "scip-default"
    assert loaded["method"] == "opop"
    assert loaded["metric"] == "primal_integral"
    assert loaded["is_win"] is True
    assert isinstance(loaded["p_value"], float)
    assert loaded["n_seeds"] == 6


@pytest.mark.smoke
def test_load_results_json_dict_wrapper(tmp_path: Path) -> None:
    """A JSON object wrapping the list under 'records' loads correctly."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps({"records": _pi_records(0.85)}), encoding="utf-8")
    records = load_results(results_path)
    assert len(records) == 12
    report = compare(
        records, baseline="scip-default", method="opop", metric="primal_integral"
    )
    assert report.is_win is True


@pytest.mark.smoke
def test_load_results_parquet(tmp_path: Path) -> None:
    """Parquet loads to the same comparison as JSON (pandas + pyarrow)."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    parquet_path = tmp_path / "results.parquet"
    pd.DataFrame(_pi_records(0.85)).to_parquet(parquet_path)

    records = load_results(parquet_path)
    report = compare(
        records, baseline="scip-default", method="opop", metric="primal_integral"
    )
    assert report.relative_improvement == pytest.approx(0.15, abs=1e-9)
    assert report.is_win is True


@pytest.mark.smoke
def test_load_results_rejects_unknown_extension(tmp_path: Path) -> None:
    """An unsupported results extension raises a clear ValueError."""
    bad = tmp_path / "results.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported results extension"):
        load_results(bad)


def test_comparison_report_to_dict_is_complete() -> None:
    """ComparisonReport.to_dict carries every required schema field."""
    report = ComparisonReport(
        baseline="b",
        method="m",
        metric="primal_integral",
        significant=True,
        p_value=0.01,
        relative_improvement=0.2,
        clears_min_effect=True,
        is_win=True,
        n_seeds=5,
        baseline_value=10.0,
        method_value=8.0,
    )
    payload = report.to_dict()
    assert _REQUIRED_FIELDS <= set(payload)
    # Round-trips through JSON without error.
    assert json.loads(report.to_json())["is_win"] is True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_cli_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI loads results, prints a table, and writes comparison_report.json."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_pi_records(0.85)), encoding="utf-8")
    out_path = tmp_path / "comparison_report.json"

    code = main(
        [
            "--results",
            str(results_path),
            "--baseline",
            "scip-default",
            "--method",
            "opop",
            "--metric",
            "primal_integral",
            "--test",
            "wilcoxon",
            "--alpha",
            "0.05",
            "--min-effect",
            "0.10",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "VERDICT" in captured.out
    assert "WIN" in captured.out

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["is_win"] is True
    assert payload["relative_improvement"] == pytest.approx(0.15, abs=1e-9)
    assert payload["significant"] is True


@pytest.mark.smoke
def test_cli_below_threshold_reports_no_win(tmp_path: Path) -> None:
    """CLI on a 3% improvement marks clears_min_effect=false (not a win)."""
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_pi_records(0.97)), encoding="utf-8")
    out_path = tmp_path / "comparison_report.json"

    code = main(
        [
            "--results",
            str(results_path),
            "--baseline",
            "scip-default",
            "--method",
            "opop",
            "--metric",
            "primal_integral",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["clears_min_effect"] is False
    assert payload["is_win"] is False


@pytest.mark.smoke
def test_cli_missing_file_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing results file yields a non-zero exit and a stderr message."""
    code = main(
        [
            "--results",
            str(tmp_path / "nope.json"),
            "--baseline",
            "a",
            "--method",
            "b",
            "--metric",
            "primal_integral",
        ]
    )
    assert code == 1
    assert "compare failed" in capsys.readouterr().err
