"""Tests for baselines 5 & 6 — classic matheuristics + LLM-enhanced CO (task 38).

Pure (solver-free) tests cover the LLM heuristic selector (vocabulary, alias
normalisation, config sanitisation, fail-closed fallback) and the schema/cost
contracts. SCIP-backed tests (``integration`` + ``solver_skip_if_missing``) drive
both baselines end to end on a small knapsack and assert: the matheuristic
improves a warm-start incumbent, the LLM-enhanced baseline selects/evolves and
runs the chosen core, and both emit a schema-identical ``results.parquet`` with
LLM cost columns populated only for the LLM baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from opop.bench.cost import COST_FIELDS
from opop.experiments.baselines_56 import (
    LLM_HEURISTIC_METHOD,
    MATHEURISTIC_METHOD,
    RESULT_COLUMNS,
    BaselineOutcome,
    run_baseline_suite,
    run_llm_enhanced_baseline,
    run_matheuristic_baseline,
    write_results,
)
from opop.experiments.heuristic_selector import (
    ALLOWED_HEURISTICS,
    DEFAULT_HEURISTIC,
    HeuristicChoice,
    normalize_heuristic_name,
    sanitize_config,
    select_heuristic,
)
from opop.llm import FakeLLMClient
from opop.model.ir import MILP
from opop.bench.sources.synthetic import generate_knapsack

# An LLM reply that always picks full-neighbourhood LNS (reaches the optimum).
_LNS_REPLY = (
    '{"heuristic": "lns", "config": {"destroy_frac": 1.0, "n_iter": 1},'
    ' "rationale": "full reoptimization"}'
)


def _knapsack() -> MILP:
    """Deterministic 10-item knapsack (MAX); all-zeros is a feasible warm start."""
    return generate_knapsack(10, seed=0)


def _zeros(ir: MILP) -> dict[str, float]:
    """The trivial all-zeros (feasible) incumbent for ``ir``."""
    return {v.name: 0.0 for v in ir.variables}


# ---------------------------------------------------------------------------
# Pure: heuristic selector (no solver, no network)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_normalize_and_sanitize() -> None:
    """Name aliases canonicalise and config sanitisation drops junk + booleans."""
    assert normalize_heuristic_name("Large Neighborhood Search") == "lns"
    assert normalize_heuristic_name("local-branching") == "local_branching"
    assert normalize_heuristic_name("RINS") == "rins"
    assert normalize_heuristic_name("repair_solution") == "repair"
    assert normalize_heuristic_name("teleport") is None
    assert normalize_heuristic_name(42) is None

    clean = sanitize_config({"k": 3, "destroy_frac": 0.4, "bogus": 9, "n_iter": True})
    assert clean == {"k": 3.0, "destroy_frac": 0.4}  # bogus dropped, bool n_iter rejected


@pytest.mark.smoke
def test_select_heuristic_valid_and_aliases() -> None:
    """A valid reply is parsed; the chosen name is always in the vocabulary."""
    llm = FakeLLMClient(response=_LNS_REPLY)
    choice = select_heuristic(llm, {"n_vars": 5})
    assert isinstance(choice, HeuristicChoice)
    assert choice.heuristic == "lns"
    assert choice.heuristic in ALLOWED_HEURISTICS
    assert choice.config == {"destroy_frac": 1.0, "n_iter": 1.0}
    assert choice.fell_back is False
    # FakeLLM records token usage for the call.
    assert llm.tracker.total_tokens_in > 0


@pytest.mark.smoke
def test_select_heuristic_falls_back_on_garbage() -> None:
    """An unparseable / out-of-vocabulary reply deterministically falls back."""
    bad_json = FakeLLMClient(response="not json at all")
    fb1 = select_heuristic(bad_json, {"n_vars": 5}, default="rins")
    assert fb1.fell_back is True
    assert fb1.heuristic == "rins"
    assert fb1.heuristic in ALLOWED_HEURISTICS

    illegal = FakeLLMClient(response='{"heuristic": "delete_all_constraints"}')
    fb2 = select_heuristic(illegal, {"n_vars": 5})
    assert fb2.fell_back is True
    assert fb2.heuristic == DEFAULT_HEURISTIC


# ---------------------------------------------------------------------------
# Pure: schema contracts
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_result_columns_cover_base_and_cost() -> None:
    """The canonical schema carries the base + every cost column, with LLM cost."""
    assert set(COST_FIELDS) <= set(RESULT_COLUMNS)
    for col in ("instance_id", "method", "seed", "heuristic", "primal_integral", "time"):
        assert col in RESULT_COLUMNS
    for col in ("llm_tokens_in", "llm_tokens_out", "llm_cost_usd"):
        assert col in RESULT_COLUMNS


# ---------------------------------------------------------------------------
# SCIP-backed: Baseline 5 (matheuristic)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize("core", ["local_branching", "rins", "lns"])
def test_matheuristic_improves_incumbent(
    core: str, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """Each core strictly improves the all-zeros warm start on the knapsack."""
    solver_skip_if_missing("scip")
    ir = _knapsack()
    config: dict[str, float] = {"k": 10.0} if core == "local_branching" else {}
    outcome = run_matheuristic_baseline(
        ir,
        core,
        seed=0,
        time_limit=5.0,
        initial_incumbent=_zeros(ir),
        config=config,
    )
    assert outcome.method == MATHEURISTIC_METHOD
    assert outcome.heuristic == core
    assert outcome.feasible
    assert outcome.improved
    assert outcome.n_accepted == 1
    assert outcome.objective > 0.0  # strictly better than the all-zeros start
    assert outcome.incumbent is not None

    # Matheuristic rows carry ZERO LLM cost and never touch analyzer/verify/control.
    cost = outcome.cost
    assert cost["llm_tokens_in"] == 0
    assert cost["llm_tokens_out"] == 0
    assert cost["llm_cost_usd"] == 0.0
    assert cost["analyzer_time"] == 0.0
    assert cost["controller_time"] == 0.0
    assert cost["verification_time"] == 0.0
    assert cost["proposer_time"] == 0.0
    assert cost["total_wall_time"] >= cost["solver_wall_time"]
    assert outcome.n_llm_calls == 0


@pytest.mark.integration
def test_matheuristic_reaches_known_optimum_marks_solved(
    solver_skip_if_missing: Callable[[str], None]
) -> None:
    """With a known optimum reached, the row is honestly marked solved."""
    solver_skip_if_missing("scip")
    ir = _knapsack()
    # Optimum from a full solve (matches the heuristic's reachable best).
    full = run_matheuristic_baseline(ir, "lns", seed=0, time_limit=5.0, initial_incumbent=_zeros(ir))
    optimum = full.objective
    outcome = run_matheuristic_baseline(
        ir,
        "lns",
        seed=0,
        time_limit=5.0,
        initial_incumbent=_zeros(ir),
        reference_optimum=optimum,
    )
    row = outcome.to_row()
    assert row["solved"] is True
    assert row["censored"] is False
    assert 0.0 <= float(row["gap"]) <= 1e-6


# ---------------------------------------------------------------------------
# SCIP-backed: Baseline 6 (LLM-enhanced)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_llm_baseline_selects_and_runs(
    solver_skip_if_missing: Callable[[str], None]
) -> None:
    """The LLM selects a core, runs it, improves the incumbent, and tracks cost."""
    solver_skip_if_missing("scip")
    ir = _knapsack()
    llm = FakeLLMClient(response=_LNS_REPLY, price_input_1m=0.5, price_output_1m=1.5)
    outcome = run_llm_enhanced_baseline(
        ir, llm, seed=0, time_limit=5.0, n_rounds=2, initial_incumbent=_zeros(ir)
    )
    assert outcome.method == LLM_HEURISTIC_METHOD
    assert outcome.heuristic in ALLOWED_HEURISTICS
    assert outcome.feasible
    assert outcome.improved
    assert outcome.objective > 0.0
    assert outcome.n_llm_calls == 2
    assert len(outcome.selection_history) == 2
    assert all(rec["heuristic"] == "lns" for rec in outcome.selection_history)

    # LLM cost MUST be tracked (tokens + USD), and folded into end-to-end time.
    cost = outcome.cost
    assert cost["llm_tokens_in"] > 0
    assert cost["llm_tokens_out"] > 0
    assert cost["llm_cost_usd"] > 0.0
    assert cost["total_wall_time"] >= cost["solver_wall_time"]
    # Still standalone: no analyzer / verify / controller activity.
    assert cost["analyzer_time"] == 0.0
    assert cost["controller_time"] == 0.0
    assert cost["verification_time"] == 0.0


@pytest.mark.integration
def test_llm_baseline_evolution_switches_heuristic(
    solver_skip_if_missing: Callable[[str], None]
) -> None:
    """A round-dependent LLM evolves its choice; the history records each pick."""
    solver_skip_if_missing("scip")
    ir = _knapsack()

    def _by_round(message: str) -> str:
        # The prompt embeds the round index; pick rins first, then local_branching.
        if '"round": 0' in message:
            return '{"heuristic": "rins", "config": {}, "rationale": "warm restart"}'
        return '{"heuristic": "local_branching", "config": {"k": 10}, "rationale": "intensify"}'

    llm = FakeLLMClient(response=_by_round, price_input_1m=0.5, price_output_1m=1.5)
    outcome = run_llm_enhanced_baseline(
        ir, llm, seed=0, time_limit=5.0, n_rounds=2, initial_incumbent=_zeros(ir)
    )
    picks = [rec["heuristic"] for rec in outcome.selection_history]
    assert picks == ["rins", "local_branching"]
    assert outcome.feasible
    assert outcome.n_llm_calls == 2


# ---------------------------------------------------------------------------
# SCIP-backed: shared harness + schema-identical results.parquet
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_suite_writes_schema_identical_parquet(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """Both baselines share one results schema; LLM cost only where applicable."""
    solver_skip_if_missing("scip")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    ir = _knapsack()
    llm = FakeLLMClient(response=_LNS_REPLY, price_input_1m=0.5, price_output_1m=1.5)
    outcomes = run_baseline_suite(
        [ir], [0, 1], llm=llm, cores=("lns", "rins"), time_limit=3.0, n_rounds=2
    )
    # 2 seeds x (2 matheuristic cores + 1 llm) = 6 rows.
    assert len(outcomes) == 6

    # Every outcome emits exactly the canonical column set.
    for outcome in outcomes:
        assert set(outcome.to_row().keys()) == set(RESULT_COLUMNS)

    path = write_results(outcomes, tmp_path)
    assert path.name == "results.parquet"
    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(RESULT_COLUMNS)

    math_rows = frame[frame["method"] == MATHEURISTIC_METHOD]
    llm_rows = frame[frame["method"] == LLM_HEURISTIC_METHOD]
    assert not math_rows.empty
    assert not llm_rows.empty

    # Matheuristic rows: zero LLM cost. LLM rows: positive token usage.
    assert bool((math_rows["llm_cost_usd"] == 0.0).all())
    assert bool((math_rows["llm_tokens_in"] == 0).all())
    assert bool((llm_rows["llm_tokens_in"] > 0).all())
    assert bool((llm_rows["llm_cost_usd"] > 0.0).all())

    # Honest end-to-end time: never below solver-only time, for every row.
    assert bool((frame["time"] >= frame["solver_wall_time"] - 1e-9).all())


@pytest.mark.integration
def test_quick_incumbent_warm_start_when_no_explicit_incumbent(
    solver_skip_if_missing: Callable[[str], None]
) -> None:
    """Without an explicit incumbent, a quick SCIP solve supplies the warm start."""
    solver_skip_if_missing("scip")
    ir = _knapsack()
    outcome: BaselineOutcome = run_matheuristic_baseline(ir, "lns", seed=0, time_limit=5.0)
    assert outcome.feasible
    assert outcome.incumbent is not None
    # The warm-start solve contributes real solver wall time.
    assert outcome.cost["solver_wall_time"] > 0.0
    row = outcome.to_row()
    assert row["instance_id"] == ir.name
    assert row["method"] == MATHEURISTIC_METHOD
