"""Tests for the falsifiable T1-T4 thesis evaluator (task 40 chunk 1).

Covers: T1 win with an honest T2 efficiency failure, T3 generality across
problem types, T4 method novelty, the held-out one-shot guard, negative results
that stay visible in the report, the events.jsonl solve-count path, the
no-solve-data fallback, deterministic JSON output, the registry problem-type
map, and the CLI (direct file + run-directory resolution).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from opop.bench.registry import BenchmarkRegistry
from opop.bench.sources.phase1_set import REGISTRY_PATH
from opop.eval.theses import (
    ThesisError,
    ThesisEvaluator,
    ThesisReport,
    ThesisVerdict,
    build_problem_type_map,
    evaluate_theses,
    main,
)

# Eight distinct, all-positive primal-integral baselines: a uniform factor gives
# distinct positive paired differences (no ties) so scipy uses the EXACT
# Wilcoxon test and p < 0.05 is reachable from a single seed.
_PI_BASE: tuple[float, ...] = (10.0, 11.0, 9.0, 12.0, 8.0, 14.0, 7.0, 13.0)

# Locked method tags + factors: opop is 30% better than scip-default, params-only
# is 10% better, modeling-agent is 5% better -> opop beats every baseline by the
# >= 10% margin required for a WIN.
_FACTORS: dict[str, float] = {
    "scip-default": 1.0,
    "opop-params-only": 0.90,
    "modeling-agent": 0.95,
    "opop": 0.70,
}

_REQUIRED_VERDICT_KEYS = {
    "claim",
    "metric",
    "baseline",
    "significant",
    "effect",
    "clears_threshold",
    "verdict",
    "details",
}


def _rec(
    method: str,
    instance: str,
    pi: float,
    *,
    seed: int = 0,
    n_solves: int | None = None,
) -> dict[str, Any]:
    """Build one consolidated result record (matrix-driver row schema)."""
    record: dict[str, Any] = {
        "instance_id": instance,
        "method": method,
        "seed": seed,
        "primal_integral": pi,
        "gap": 0.0,
        "time": 1.0,
        "solved": True,
        "censored": False,
    }
    if n_solves is not None:
        record["n_solves"] = n_solves
    return record


def _matrix(
    factors: dict[str, float],
    *,
    instances: list[str] | None = None,
    base: tuple[float, ...] = _PI_BASE,
    n_solves: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Build a paired record matrix: ``pi = base * factors[method]`` per cell."""
    inst_ids = instances or [f"inst{i}" for i in range(len(base))]
    records: list[dict[str, Any]] = []
    for method, factor in factors.items():
        solves = n_solves.get(method) if n_solves else None
        for inst, value in zip(inst_ids, base, strict=False):
            records.append(_rec(method, inst, value * factor, n_solves=solves))
    return records


def _milp_map() -> dict[str, str]:
    """All ``inst*`` instances classified as MILP (hermetic T3 map)."""
    return {f"inst{i}": "MILP" for i in range(len(_PI_BASE))}


# --------------------------------------------------------------------------- #
# T1 win + T2 efficiency failure (both honest)
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_thesis_T1_win_and_T2_fail() -> None:
    """opop beats scip-default + params-only (T1) but is only 10% leaner (T2 fail)."""
    records = _matrix(_FACTORS, n_solves={"scip-default": 20, "opop": 18})
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")

    t1 = report.verdicts["T1"]
    assert t1.verdict is True
    assert t1.significant is True
    assert t1.clears_threshold is True
    assert t1.baseline == ["scip-default", "opop-params-only"]
    assert {"scip-default", "opop-params-only"} <= set(t1.details["comparisons"])

    t2 = report.verdicts["T2"]
    assert t2.verdict is False
    assert t2.clears_threshold is False
    assert t2.metric == "n_solves"
    assert t2.effect == pytest.approx(0.10, abs=1e-9)
    assert t2.details["source"] == "n_solves"


# --------------------------------------------------------------------------- #
# T3 generality across problem types
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_thesis_T3_generality() -> None:
    """opop wins on BOTH MILP and QUBO instances -> T3 holds."""
    milp = [f"milp{i}" for i in range(6)]
    qubo = [f"qubo{i}" for i in range(6)]
    base6 = _PI_BASE[:6]
    records: list[dict[str, Any]] = []
    for inst, value in zip(milp, base6, strict=True):
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.70))
    for inst, value in zip(qubo, base6, strict=True):
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.70))

    type_map = {**{i: "MILP" for i in milp}, **{i: "QUBO" for i in qubo}}
    report = ThesisEvaluator(problem_types=type_map).evaluate(records, split="validation")

    t3 = report.verdicts["T3"]
    assert t3.verdict is True
    assert sorted(t3.details["problem_types"]) == ["MILP", "QUBO"]
    assert t3.details["per_problem_type"]["MILP"]["is_win"] is True
    assert t3.details["per_problem_type"]["QUBO"]["is_win"] is True


@pytest.mark.smoke
def test_thesis_T3_fails_when_one_type_loses() -> None:
    """A single problem type where opop does not win sinks the whole T3 verdict."""
    milp = [f"milp{i}" for i in range(6)]
    qubo = [f"qubo{i}" for i in range(6)]
    base6 = _PI_BASE[:6]
    records: list[dict[str, Any]] = []
    for inst, value in zip(milp, base6, strict=True):
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.70))  # MILP: win
    for inst, value in zip(qubo, base6, strict=True):
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.98))  # QUBO: 2% only, no win

    type_map = {**{i: "MILP" for i in milp}, **{i: "QUBO" for i in qubo}}
    report = ThesisEvaluator(problem_types=type_map).evaluate(records, split="validation")

    t3 = report.verdicts["T3"]
    assert t3.verdict is False
    assert t3.details["per_problem_type"]["MILP"]["is_win"] is True
    assert t3.details["per_problem_type"]["QUBO"]["is_win"] is False


# --------------------------------------------------------------------------- #
# T4 method novelty
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_thesis_T4_novelty() -> None:
    """opop beats params-only AND modeling-agent -> analyzer deltas add value."""
    records = _matrix(_FACTORS)
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")

    t4 = report.verdicts["T4"]
    assert t4.verdict is True
    assert t4.significant is True
    assert t4.clears_threshold is True
    assert t4.baseline == ["opop-params-only", "modeling-agent"]
    assert {"opop-params-only", "modeling-agent"} <= set(t4.details["comparisons"])


@pytest.mark.smoke
def test_thesis_T4_fails_when_modeling_agent_ties() -> None:
    """opop beating params-only is not enough if modeling-agent is not beaten."""
    factors = {"opop-params-only": 0.90, "modeling-agent": 0.71, "opop": 0.70}
    records = _matrix(factors)
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")

    t4 = report.verdicts["T4"]
    assert t4.verdict is False
    assert t4.details["comparisons"]["opop-params-only"]["is_win"] is True
    assert t4.details["comparisons"]["modeling-agent"]["is_win"] is False


# --------------------------------------------------------------------------- #
# One-shot guard
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
@pytest.mark.parametrize("split", ["test", "ood_test"])
def test_one_shot_guard_blocks_test_split(split: str) -> None:
    """A held-out split without one_shot_final raises BEFORE any computation."""
    records = _matrix(_FACTORS)
    evaluator = ThesisEvaluator(problem_types=_milp_map())
    with pytest.raises(ThesisError, match="held-out"):
        evaluator.evaluate(records, split=split)


@pytest.mark.smoke
def test_one_shot_final_flag_allows_test_split() -> None:
    """With one_shot_final=True the held-out split is evaluated normally."""
    records = _matrix(_FACTORS)
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(
        records, split="test", one_shot_final=True
    )
    assert report.one_shot_final is True
    assert report.verdicts["T1"].verdict is True


# --------------------------------------------------------------------------- #
# Negative results stay visible
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_negative_result_reported() -> None:
    """opop only 3% better -> verdict false, but the comparison is still reported."""
    factors = {"scip-default": 1.0, "opop-params-only": 0.99, "opop": 0.97}
    records = _matrix(factors)
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")

    t1 = report.verdicts["T1"]
    assert t1.verdict is False
    assert t1.clears_threshold is False
    cmp_scip = t1.details["comparisons"]["scip-default"]
    assert cmp_scip["is_win"] is False
    assert cmp_scip["relative_improvement"] == pytest.approx(0.03, abs=1e-9)
    # Every thesis must appear regardless of outcome.
    assert {"T1", "T2", "T3", "T4"} <= set(report.to_dict())


# --------------------------------------------------------------------------- #
# T2 solve-count sources
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_t2_events_path_solve_counts() -> None:
    """events.jsonl solve rows drive T2: opop 6 solves vs scip 10 -> 40% fewer (win)."""
    base6 = _PI_BASE[:6]
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for i, value in enumerate(base6):
        inst = f"inst{i}"
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.70))
        events += [
            {"event_type": "solve", "method": "scip-default", "instance_id": inst, "seed": 0}
            for _ in range(10)
        ]
        events += [
            {"event_type": "solve", "method": "opop", "instance_id": inst, "seed": 0}
            for _ in range(6)
        ]

    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(
        records, events, split="validation"
    )
    t2 = report.verdicts["T2"]
    assert t2.details["source"] == "events"
    assert t2.effect == pytest.approx(0.40, abs=1e-9)
    assert t2.clears_threshold is True
    assert t2.verdict is True
    assert t2.details["n_pairs"] == 6


@pytest.mark.smoke
def test_t2_events_jsonl_file(tmp_path: Path) -> None:
    """A real events.jsonl path is read line-by-line for the solve counts."""
    base6 = _PI_BASE[:6]
    records: list[dict[str, Any]] = []
    lines: list[str] = []
    for i, value in enumerate(base6):
        inst = f"inst{i}"
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.70))
        for _ in range(12):
            lines.append(json.dumps({"event_type": "solve", "method": "scip-default", "instance_id": inst, "seed": 0}))
        for _ in range(6):
            lines.append(json.dumps({"event_type": "solve", "method": "opop", "instance_id": inst, "seed": 0}))
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(
        records, str(events_path), split="validation"
    )
    t2 = report.verdicts["T2"]
    assert t2.details["source"] == "events"
    assert t2.effect == pytest.approx(0.50, abs=1e-9)
    assert t2.verdict is True


@pytest.mark.smoke
def test_t2_no_solve_data_reported() -> None:
    """No events and no n_solves column -> T2 honestly reports a no-data non-win."""
    records = _matrix(_FACTORS)  # no n_solves column
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")
    t2 = report.verdicts["T2"]
    assert t2.verdict is False
    assert t2.significant is False
    assert t2.effect == 0.0
    assert "no solve-count data" in t2.details["note"]


@pytest.mark.smoke
def test_t2_ignores_unmethoded_events_and_falls_back() -> None:
    """Closed-loop journal rows lacking a method cannot be attributed -> n_solves wins."""
    base6 = _PI_BASE[:6]
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for i, value in enumerate(base6):
        inst = f"inst{i}"
        records.append(_rec("scip-default", inst, value, n_solves=20))
        records.append(_rec("opop", inst, value * 0.70, n_solves=10))
        # Raw journal rows (no method) — cannot attribute, must be skipped.
        events.append({"verify_status": "pass", "instance_id": inst, "score": {"primal_integral": 1.0}})

    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(
        records, events, split="validation"
    )
    t2 = report.verdicts["T2"]
    assert t2.details["source"] == "n_solves"
    assert t2.effect == pytest.approx(0.50, abs=1e-9)
    assert t2.verdict is True


# --------------------------------------------------------------------------- #
# Report serialisation + dataclasses
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_verdict_to_dict_carries_required_schema() -> None:
    """ThesisVerdict.to_dict exposes every required field; the list baseline survives."""
    verdict = ThesisVerdict(
        claim="c",
        metric="primal_integral",
        baseline=["scip-default", "opop-params-only"],
        significant=True,
        effect=0.2,
        clears_threshold=True,
        verdict=True,
        details={"comparisons": {}},
    )
    payload = verdict.to_dict()
    assert _REQUIRED_VERDICT_KEYS == set(payload)
    assert payload["baseline"] == ["scip-default", "opop-params-only"]


@pytest.mark.smoke
def test_report_json_is_deterministic_and_finite() -> None:
    """to_json is key-sorted, NaN-free, and stable across calls; write adds a newline."""
    records = _matrix(_FACTORS, n_solves={"scip-default": 20, "opop": 18})
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")

    first = report.to_json()
    second = report.to_json()
    assert first == second
    loaded = json.loads(first)  # parses, and allow_nan=False never produced NaN/Inf
    assert {"T1", "T2", "T3", "T4"} <= set(loaded)
    for name in ("T1", "T2", "T3", "T4"):
        assert _REQUIRED_VERDICT_KEYS <= set(loaded[name])
    # Sorted-keys invariant.
    assert json.dumps(loaded, sort_keys=True) == json.dumps(json.loads(first), sort_keys=True)


@pytest.mark.smoke
def test_report_write_trailing_newline(tmp_path: Path) -> None:
    """ThesisReport.write emits a trailing newline and a re-loadable JSON file."""
    records = _matrix(_FACTORS)
    report = ThesisEvaluator(problem_types=_milp_map()).evaluate(records, split="validation")
    out = report.write(tmp_path / "nested" / "thesis_report.json")
    text = out.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text)["meta"]["split"] == "validation"


# --------------------------------------------------------------------------- #
# Registry problem-type map
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_problem_type_map_from_registry() -> None:
    """The registry map covers free + held-out instances with their problem_type."""
    registry = BenchmarkRegistry.from_yaml(REGISTRY_PATH)
    type_map = build_problem_type_map(registry)
    assert type_map["flugpl"] == "MILP"
    assert type_map["qplib/box_miqp"] == "MIQP"
    assert type_map["qplib/ball_miqcp"] == "MIQCP"
    assert type_map["classic/tsp/tiny4"] == "TSP"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_cli_writes_thesis_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI loads a results file, prints a summary, and writes thesis_report.json."""
    records = _matrix(_FACTORS, n_solves={"scip-default": 20, "opop": 8})
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(records), encoding="utf-8")
    out_path = tmp_path / "thesis_report.json"

    code = main(
        [
            "--results",
            str(results_path),
            "--split",
            "validation",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "thesis evaluation" in captured.out

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert {"T1", "T2", "T3", "T4"} <= set(payload)
    assert payload["T1"]["verdict"] is True
    # n_solves 20 -> 8 is a 60% reduction: T2 clears its threshold here.
    assert payload["T2"]["verdict"] is True
    assert payload["T2"]["details"]["source"] == "n_solves"


@pytest.mark.smoke
def test_cli_resolves_run_directory_and_events(tmp_path: Path) -> None:
    """--results <dir> finds results.json and auto-discovers events.jsonl for T2."""
    run_dir = tmp_path / "final_eval"
    run_dir.mkdir()
    base6 = _PI_BASE[:6]
    records: list[dict[str, Any]] = []
    event_lines: list[str] = []
    for i, value in enumerate(base6):
        inst = f"inst{i}"
        records.append(_rec("scip-default", inst, value))
        records.append(_rec("opop", inst, value * 0.70))
        for _ in range(10):
            event_lines.append(json.dumps({"event_type": "solve", "method": "scip-default", "instance_id": inst, "seed": 0}))
        for _ in range(5):
            event_lines.append(json.dumps({"event_type": "solve", "method": "opop", "instance_id": inst, "seed": 0}))
    (run_dir / "results.json").write_text(json.dumps(records), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("\n".join(event_lines) + "\n", encoding="utf-8")
    out_path = tmp_path / "thesis_report.json"

    code = main(["--results", str(run_dir), "--out", str(out_path)])
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["T2"]["details"]["source"] == "events"
    assert payload["T2"]["effect"] == pytest.approx(0.50, abs=1e-9)
    assert payload["T2"]["verdict"] is True


@pytest.mark.smoke
def test_cli_missing_results_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing results file yields a non-zero exit and a stderr message."""
    code = main(["--results", str(tmp_path / "nope.json"), "--out", str(tmp_path / "r.json")])
    assert code == 1
    assert "theses failed" in capsys.readouterr().err


@pytest.mark.smoke
def test_evaluate_theses_in_memory_records() -> None:
    """The evaluate_theses helper accepts in-memory records + an injected map."""
    records = _matrix(_FACTORS)
    report = evaluate_theses(
        records, split="validation", problem_types=_milp_map()
    )
    assert isinstance(report, ThesisReport)
    assert report.verdicts["T1"].verdict is True
    assert report.verdicts["T4"].verdict is True
