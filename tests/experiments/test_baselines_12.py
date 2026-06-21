"""Tests for baselines 1--2 + the fairness harness (plan task 36).

Covers, in order:

* **Fairness** (pure, no solver): ``BudgetSpec.from_config``, ``trials`` /
  ``time_limit_sec`` / ``seeds`` equality checks, the held-out-split tuning
  guard, and the harness rejecting an unfair budget or a tuning runner on
  ``test`` / ``ood_test``.
* **Wiring** (pure): per-solver default SMAC spaces, ``ParamSpec.decode``, and
  the ``SMACTunedRunner`` param-space / method-name wiring (no ``smac`` needed —
  the dependency is imported lazily only when a tuning solve actually runs).
* **DefaultRunner** (integration, SCIP/HiGHS/CP-SAT): solves the fixed
  formulation with the solver's defaults and emits the exact opop ``results``
  schema + cost columns.
* **Harness + compare** (integration, SCIP): ``run_baselines`` writes
  ``results.parquet`` and the rows are consumed by ``compare()`` unchanged.
* **SMACTunedRunner** (integration, SCIP + ``smac``): tunes under the shared
  budget; skipped via ``pytest.importorskip("smac")`` when SMAC is absent.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from opop.bench.sources.synthetic import generate_knapsack
from opop.config import BudgetConfig, RunConfig
from opop.experiments.baselines import (
    CPSAT_PARAM_SPACE,
    HIGHS_PARAM_SPACE,
    RESULT_COLUMNS,
    SCIP_PARAM_SPACE,
    DefaultRunner,
    ParamSpec,
    SMACTunedRunner,
    default_param_space,
    run_baselines,
)
from opop.experiments.fairness import (
    BudgetSpec,
    FairnessError,
    assert_tunable_split,
    check_budget_fairness,
)
from opop.model.ir import MILP
from opop.model.state import Phi, SolveTrace
from opop.solver.scip import ScipKernel

#: The opop ``results.parquet`` columns every baseline row must carry.
_OPOP_COLUMNS = {
    "instance_id",
    "method",
    "seed",
    "primal_integral",
    "gap",
    "time",
    "solved",
    "censored",
    "time_limit",
    "n_accepted",
}

#: The cost-accounting columns the baseline harness adds on top.
_COST_COLUMNS = {"solver_time_sec", "n_solves"}


def _make_config(
    *,
    trials: int = 3,
    time_limit_sec: float = 2.0,
    seeds: tuple[int, ...] = (0, 1),
    split: str = "dev",
) -> RunConfig:
    """A minimal :class:`RunConfig` with the given budget / seeds / split."""
    return RunConfig(
        seeds=list(seeds),
        budget=BudgetConfig(trials=trials, time_limit_sec=time_limit_sec),
        split=split,
    )


def _make_kernel(name: str) -> object:
    """Build the solver kernel for a backend tag (lazy, per-backend import)."""
    if name == "scip":
        return ScipKernel()
    if name == "highs":
        from opop.solver.highs import HighsKernel

        return HighsKernel()
    if name == "cpsat":
        from opop.solver.cpsat import CpsatKernel

        return CpsatKernel()
    raise ValueError(f"unknown solver tag {name!r}")


class _UnknownKernel:
    """A kernel whose ``solver_name`` has no built-in SMAC tuning space."""

    solver_name = "Mystery"

    def solve(
        self, ir: MILP, phi: Phi, *, time_limit: float, memory_limit_mb: int, seed: int
    ) -> SolveTrace:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Fairness — budget / seed equality (pure)
# --------------------------------------------------------------------------- #


def test_budget_spec_from_config_extracts_budget() -> None:
    spec = BudgetSpec.from_config(_make_config(trials=7, time_limit_sec=12.5, seeds=(0, 3, 9)))
    assert spec.trials == 7
    assert spec.time_limit_sec == 12.5
    assert spec.seeds == (0, 3, 9)
    assert spec.to_dict() == {"trials": 7, "time_limit_sec": 12.5, "seeds": [0, 3, 9]}


def test_check_budget_fairness_passes_for_identical_budgets() -> None:
    ref = BudgetSpec(trials=4, time_limit_sec=2.0, seeds=(0, 1, 2))
    cand = BudgetSpec(trials=4, time_limit_sec=2.0, seeds=(0, 1, 2))
    # Must not raise.
    check_budget_fairness(ref, cand)


def test_check_budget_fairness_rejects_trial_mismatch() -> None:
    ref = BudgetSpec(trials=4, time_limit_sec=2.0, seeds=(0, 1))
    cand = BudgetSpec(trials=8, time_limit_sec=2.0, seeds=(0, 1))
    with pytest.raises(FairnessError, match="trials differ"):
        check_budget_fairness(ref, cand)


def test_check_budget_fairness_rejects_time_limit_mismatch() -> None:
    ref = BudgetSpec(trials=4, time_limit_sec=2.0, seeds=(0, 1))
    cand = BudgetSpec(trials=4, time_limit_sec=5.0, seeds=(0, 1))
    with pytest.raises(FairnessError, match="time_limit_sec differ"):
        check_budget_fairness(ref, cand)


def test_check_budget_fairness_rejects_seed_mismatch() -> None:
    ref = BudgetSpec(trials=4, time_limit_sec=2.0, seeds=(0, 1, 2))
    cand = BudgetSpec(trials=4, time_limit_sec=2.0, seeds=(0, 1))
    with pytest.raises(FairnessError, match="seeds differ"):
        check_budget_fairness(ref, cand)


def test_check_budget_fairness_tolerates_float_time_noise() -> None:
    ref = BudgetSpec(trials=2, time_limit_sec=2.0, seeds=(0,))
    cand = BudgetSpec(trials=2, time_limit_sec=2.0 + 1e-12, seeds=(0,))
    # Within the default abs/rel tolerance -> not a violation.
    check_budget_fairness(ref, cand)


def test_assert_tunable_split_allows_dev_and_validation() -> None:
    assert_tunable_split("dev")
    assert_tunable_split("validation")


@pytest.mark.parametrize("split", ["test", "ood_test"])
def test_assert_tunable_split_rejects_held_out(split: str) -> None:
    with pytest.raises(FairnessError, match="held-out"):
        assert_tunable_split(split)


# --------------------------------------------------------------------------- #
# Wiring — param spaces + decode (pure)
# --------------------------------------------------------------------------- #


def test_default_param_space_resolves_per_solver() -> None:
    assert default_param_space(ScipKernel()) is SCIP_PARAM_SPACE
    # HiGHS / CP-SAT spaces are looked up by solver_name without importing them.
    from opop.solver.cpsat import CpsatKernel
    from opop.solver.highs import HighsKernel

    assert default_param_space(HighsKernel()) is HIGHS_PARAM_SPACE
    assert default_param_space(CpsatKernel()) is CPSAT_PARAM_SPACE


def test_default_param_space_rejects_unknown_solver() -> None:
    with pytest.raises(ValueError, match="no default SMAC tuning space"):
        default_param_space(_UnknownKernel())


def test_param_spec_decode_endpoints_midpoint_and_clamp() -> None:
    knob = ParamSpec("separating/gomory/freq", 0.0, 10.0, is_int=True)
    assert knob.decode(0.0) == 0.0
    assert knob.decode(1.0) == 10.0
    assert knob.decode(0.5) == 5.0  # rounded to the nearest integer

    cont = ParamSpec("branching/scorefactor", 0.0, 1.0)
    assert cont.decode(0.25) == 0.25
    # Units outside [0, 1] are clamped, never extrapolated.
    assert cont.decode(-2.0) == 0.0
    assert cont.decode(3.0) == 1.0


def test_smac_tuned_runner_wiring_needs_no_smac() -> None:
    runner = SMACTunedRunner(ScipKernel())
    assert runner.is_tuning is True
    assert runner.variant == "smac"
    assert runner.method_name == "scip-smac"
    assert runner.param_space == SCIP_PARAM_SPACE
    assert all(isinstance(p, ParamSpec) for p in runner.param_space)


def test_default_runner_wiring_method_name() -> None:
    runner = DefaultRunner(ScipKernel())
    assert runner.is_tuning is False
    assert runner.variant == "default"
    assert runner.method_name == "scip-default"


# --------------------------------------------------------------------------- #
# Harness fairness gating (pure — fails before any solve)
# --------------------------------------------------------------------------- #


def test_run_baselines_rejects_unfair_budget() -> None:
    config = _make_config(trials=5, time_limit_sec=2.0, seeds=(0, 1))
    reference = BudgetSpec(trials=3, time_limit_sec=2.0, seeds=(0, 1))
    with pytest.raises(FairnessError, match="trials differ"):
        run_baselines([], [], config=config, reference_budget=reference)


def test_run_baselines_refuses_tuning_on_held_out_split() -> None:
    config = _make_config(split="test")
    reference = BudgetSpec.from_config(config)  # budget matches -> reach split guard
    runner = SMACTunedRunner(ScipKernel())  # constructed lazily, no smac needed
    with pytest.raises(FairnessError, match="held-out"):
        run_baselines(
            [generate_knapsack(6, seed=0)],
            [runner],
            config=config,
            reference_budget=reference,
        )


# --------------------------------------------------------------------------- #
# DefaultRunner — baseline 1 (integration: real solves)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.parametrize("solver", ["scip", "highs", "cpsat"])
def test_default_runner_solves_with_solver_defaults(
    solver: str, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """Baseline 1 solves the fixed formulation once with each solver's defaults."""
    solver_skip_if_missing(solver)
    kernel = _make_kernel(solver)
    runner = DefaultRunner(kernel)  # type: ignore[arg-type]
    assert runner.method_name == f"{solver}-default"

    ir = generate_knapsack(6, seed=0)
    rows = runner.run([ir], trials=1, time_limit_sec=5.0, seeds=[0, 1])

    assert len(rows) == 2  # one row per (instance, seed)
    for row in rows:
        assert set(row) == set(RESULT_COLUMNS)
        assert _OPOP_COLUMNS <= set(row)
        assert _COST_COLUMNS <= set(row)
        assert row["instance_id"] == "knapsack_6"
        assert row["method"] == f"{solver}-default"
        assert row["n_accepted"] == 0  # the default baseline never searches
        assert row["n_solves"] == 1
        assert row["solver_time_sec"] >= 0.0
        assert row["time_limit"] == 5.0
        assert row["censored"] is False
        assert row["solved"] is True  # knapsack_6 is solved to optimality
        assert float(row["gap"]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.integration
def test_default_runner_row_schema_matches_opop(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """The default-baseline row carries every opop.run column (+ cost columns)."""
    solver_skip_if_missing("scip")
    runner = DefaultRunner(ScipKernel())
    row = runner.solve_one(generate_knapsack(6, seed=0), seed=0, trials=1, time_limit=5.0)
    assert _OPOP_COLUMNS <= set(row)
    assert set(row) == set(RESULT_COLUMNS)


# --------------------------------------------------------------------------- #
# Harness + persistence + compare (integration: SCIP + pandas)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_run_baselines_writes_parquet_and_compare_consumes(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """run_baselines writes results.parquet and the rows feed compare() unchanged."""
    solver_skip_if_missing("scip")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from opop.experiments.compare import compare, load_results

    instances = [generate_knapsack(6, seed=0), generate_knapsack(8, seed=1)]
    config = _make_config(trials=2, time_limit_sec=5.0, seeds=(0, 1))
    reference = BudgetSpec.from_config(config)

    rows = run_baselines(
        instances,
        [DefaultRunner(ScipKernel())],
        config=config,
        reference_budget=reference,
        out_dir=tmp_path,
    )
    assert len(rows) == 4  # 2 instances x 2 seeds

    parquet = tmp_path / "results.parquet"
    assert parquet.is_file()
    loaded = load_results(parquet)
    assert {r["method"] for r in loaded} == {"scip-default"}
    for r in loaded:
        assert _OPOP_COLUMNS <= set(r)
        assert _COST_COLUMNS <= set(r)

    # Pair opop-tagged rows over the SAME (instance, seed) keys so compare() can
    # consume the baseline schema directly (identical metric pipeline).
    opop_rows = []
    for r in loaded:
        clone = dict(r)
        clone["method"] = "opop"
        clone["primal_integral"] = float(r["primal_integral"]) * 0.5
        opop_rows.append(clone)

    report = compare(
        [dict(r) for r in loaded] + opop_rows,
        baseline="scip-default",
        method="opop",
        metric="primal_integral",
    )
    assert report.baseline == "scip-default"
    assert report.method == "opop"
    assert report.n_pairs == len(loaded)


# --------------------------------------------------------------------------- #
# SMACTunedRunner — baseline 2 (integration: SCIP + smac; skipped w/o smac)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_smac_tuned_runner_tunes_under_budget(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """Baseline 2 runs the SMAC ask-tell loop under the shared budget/seed."""
    pytest.importorskip("smac")
    solver_skip_if_missing("scip")

    runner = SMACTunedRunner(ScipKernel())
    ir = generate_knapsack(6, seed=0)
    row = runner.solve_one(ir, seed=0, trials=4, time_limit=5.0)

    assert set(row) == set(RESULT_COLUMNS)
    assert row["method"] == "scip-smac"
    assert row["instance_id"] == "knapsack_6"
    assert row["n_solves"] == 4  # one solve per tuning trial
    assert row["n_accepted"] >= 1  # at least the initial incumbent
    assert row["solver_time_sec"] >= 0.0
    assert row["solved"] is True  # knapsack_6 solves to optimality under any params
