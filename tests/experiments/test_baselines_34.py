"""Tests for baselines 3-4 — params-only ablation + LLM modeling-agent-only (task 37).

Pure (solver-free) tests cover the spec <-> MILP round-trip, malformed-spec
handling, the modeling-agent pipeline / self-correction loop (via a stub kernel),
the no-opop-loop import guard (AST), the S0 params-only proposer filter, and the
schema contracts. SCIP-backed tests (``integration`` + ``solver_skip_if_missing``)
drive both baselines end to end and assert: baseline 3 emits ONLY class-C
parameter deltas (zero cut/formulation/decomposition), baseline 4 produces a
solved model end-to-end with a ``FakeLLMClient`` and never touches the
analyzer/verify/controller, and both emit a schema-identical ``results.parquet``
with LLM cost populated only for the modeling agent.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, final

import pytest

from opop.analyzer.api import analyze
from opop.bench.cost import COST_FIELDS
from opop.bench.sources.synthetic import generate_knapsack
from opop.config import BudgetConfig, RunConfig
from opop.experiments.baselines_34 import (
    MODELING_AGENT_METHOD,
    PARAMS_ONLY_METHOD,
    RESULT_COLUMNS,
    BaselineOutcome,
    run_baseline_suite,
    run_baselines_34,
    run_modeling_agent_baseline,
    run_params_only_baseline,
    write_results,
)
from opop.experiments.modeling_agent import (
    FORBIDDEN_LOOP_PHASES,
    MODELING_AGENT_PHASES,
    ModelSpecError,
    build_milp_from_spec,
    describe_milp,
    milp_to_spec,
    run_modeling_agent,
)
from opop.llm.client import FakeLLMClient
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
    milps_equivalent,
)
from opop.model.state import Phi, ProblemState, SolveTrace
from opop.proposer import (
    KIND_CUT,
    KIND_DECOMPOSITION,
    KIND_FORMULATION,
    KIND_PARAM,
    Stage,
    delta_kind,
    propose,
)


# ---------------------------------------------------------------------------
# Fixtures + stubs
# ---------------------------------------------------------------------------
def _knapsack_with_cover() -> MILP:
    """A 5-item knapsack (the task-10 analyzer fixture) with a known cover cut.

    ``max sum x_i  s.t.  5x0+3x1+7x2+4x3+6x4 <= 12``; the minimal cover {x2, x4}
    (7+6=13 > 12) yields the valid inequality ``x2 + x4 <= 1``, so the analyzer
    emits a candidate cut (making the S0 filter meaningful). Optimum = 3 items
    (3+4+5 = 12).
    """
    weights = [5, 3, 7, 4, 6]
    variables = tuple(
        Variable(name=f"x{i}", vtype=VarType.BINARY, lower=0.0, upper=1.0) for i in range(5)
    )
    constraints = (
        LinearConstraint(
            name="cap",
            coeffs={f"x{i}": float(weights[i]) for i in range(5)},
            sense=ConstraintSense.LE,
            rhs=12.0,
        ),
    )
    objective = Objective(
        coeffs={f"x{i}": 1.0 for i in range(5)}, sense=ObjSense.MAXIMIZE
    )
    return MILP(
        name="knap_cover", variables=variables, constraints=constraints, objective=objective
    )


@final
class _StubKernel:
    """A SCIP-free kernel returning a fixed optimal trace (offline pipeline tests)."""

    solver_name: str = "STUB"

    def solve(
        self, ir: MILP, phi: Phi, *, time_limit: float, memory_limit_mb: int, seed: int
    ) -> SolveTrace:
        del phi, time_limit, memory_limit_mb, seed
        return SolveTrace(
            primal_bound_series=[3.0, 3.0],
            dual_bound_series=[3.0, 3.0],
            time_series=[0.0, 0.01],
            nodes=1,
            lp_iters=1,
            cuts=0,
            first_feasible_time=0.0,
            status="optimal",
            censored=False,
            memory_peak=1.0,
            instance_id=ir.name,
            solver="STUB",
        )


# ---------------------------------------------------------------------------
# Pure: spec <-> MILP IR
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_milp_spec_roundtrip_binary() -> None:
    """``build_milp_from_spec(milp_to_spec(ir))`` reproduces a binary knapsack."""
    ir = _knapsack_with_cover()
    rebuilt = build_milp_from_spec(milp_to_spec(ir), name=ir.name)
    assert milps_equivalent(ir, rebuilt)


@pytest.mark.smoke
def test_milp_spec_roundtrip_mixed_bounds_and_offset() -> None:
    """The round-trip preserves integer/continuous bounds, ``-inf``, and the offset."""
    ir = MILP(
        name="mixed",
        variables=(
            Variable("a", VarType.INTEGER, 0.0, 10.0),
            Variable("b", VarType.CONTINUOUS, -math.inf, math.inf),
            Variable("c", VarType.BINARY, 0.0, 1.0),
        ),
        constraints=(
            LinearConstraint("r", {"a": 1.0, "b": 2.0, "c": 3.0}, ConstraintSense.GE, 4.0),
        ),
        objective=Objective({"a": 1.0, "b": -1.0}, ObjSense.MINIMIZE, offset=7.0),
    )
    rebuilt = build_milp_from_spec(milp_to_spec(ir), name=ir.name)
    assert milps_equivalent(ir, rebuilt)


@pytest.mark.smoke
def test_build_milp_from_explicit_json_spec() -> None:
    """A hand-written JSON spec parses to the expected MILP structure."""
    spec: dict[str, Any] = {
        "name": "kp",
        "sense": "maximize",
        "variables": [{"name": "x", "type": "binary"}, {"name": "y", "type": "binary"}],
        "objective": {"x": 10.0, "y": 13.0},
        "constraints": [
            {"name": "cap", "coeffs": {"x": 2.0, "y": 3.0}, "sense": "<=", "rhs": 4.0}
        ],
    }
    ir = build_milp_from_spec(spec, name="kp")
    assert ir.n_vars == 2
    assert ir.n_constraints == 1
    assert ir.objective.sense is ObjSense.MAXIMIZE
    assert all(v.vtype is VarType.BINARY for v in ir.variables)
    assert ir.constraints[0].sense is ConstraintSense.LE


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad",
    [
        {},  # no variables
        {"variables": []},  # empty variables
        {"variables": [{"name": "x", "type": "binary"}], "sense": "sideways"},  # bad obj sense
        {"variables": [{"name": "x", "type": "teleport"}]},  # bad vtype
        {"variables": [{"name": "x", "type": "binary"}], "objective": {"x": "abc"}},  # bad coeff
        {  # coefficient names an undeclared variable
            "variables": [{"name": "x", "type": "binary"}],
            "constraints": [{"coeffs": {"z": 1.0}, "sense": "<=", "rhs": 1.0}],
        },
        {  # bad constraint sense
            "variables": [{"name": "x", "type": "binary"}],
            "constraints": [{"coeffs": {"x": 1.0}, "sense": "<>", "rhs": 1.0}],
        },
    ],
)
def test_build_milp_from_bad_spec_raises(bad: dict[str, Any]) -> None:
    """Every malformed spec raises a typed :class:`ModelSpecError`."""
    with pytest.raises(ModelSpecError):
        build_milp_from_spec(bad, name="bad")


# ---------------------------------------------------------------------------
# Pure: the modeling agent NEVER imports an opop-loop component
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_modeling_agent_module_has_no_opop_loop_imports() -> None:
    """`modeling_agent` must not import the analyzer / verify gate / controller."""
    import opop.experiments.modeling_agent as mod

    assert mod.__file__ is not None
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    forbidden = ("opop.analyzer", "opop.verify", "opop.controller")
    offenders = sorted(
        name for name in imported for f in forbidden if name == f or name.startswith(f + ".")
    )
    assert not offenders, f"modeling agent must not import opop-loop modules: {offenders}"


# ---------------------------------------------------------------------------
# Pure: schema contracts
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_result_columns_cover_base_and_cost() -> None:
    """The canonical schema carries every cost column plus the LLM cost columns."""
    assert set(COST_FIELDS) <= set(RESULT_COLUMNS)
    for col in ("instance_id", "method", "seed", "primal_integral", "time", "n_llm_calls"):
        assert col in RESULT_COLUMNS
    for col in ("llm_tokens_in", "llm_tokens_out", "llm_cost_usd"):
        assert col in RESULT_COLUMNS


@pytest.mark.smoke
def test_outcome_rows_schema_identical_across_methods() -> None:
    """Both baselines' rows carry EXACTLY the canonical column set."""
    cost = {col: (0 if "tokens" in col else 0.0) for col in COST_FIELDS}
    metrics = {"primal_integral": 0.0, "gap": 0.0, "optimal": 1.0, "censored": 0.0}
    params_row = BaselineOutcome(
        method=PARAMS_ONLY_METHOD,
        instance_id="kp",
        seed=0,
        metrics=metrics,
        cost=cost,
        time_limit=5.0,
        n_accepted=3,
        n_llm_calls=0,
    ).to_row()
    agent_row = BaselineOutcome(
        method=MODELING_AGENT_METHOD,
        instance_id="kp",
        seed=0,
        metrics=metrics,
        cost=cost,
        time_limit=5.0,
        n_accepted=0,
        n_llm_calls=1,
    ).to_row()
    assert set(params_row) == set(agent_row) == set(RESULT_COLUMNS)
    assert params_row["method"] == PARAMS_ONLY_METHOD
    assert agent_row["method"] == MODELING_AGENT_METHOD


# ---------------------------------------------------------------------------
# Pure-ish: S0 params-only filter (no SCIP — analyze without the LP relaxation)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_params_only_proposer_emits_only_param_deltas() -> None:
    """At S0 the proposer emits only params; at S4 it CAN emit a cut (the ablation)."""
    ir = _knapsack_with_cover()
    report = analyze(ir, solve_relaxation=False)
    assert report.candidate_cuts, "fixture must yield candidate cuts so S0 filtering is meaningful"
    state = ProblemState(instance_id=ir.name, task_family="MILP", budget_state={"ir": ir})

    # Full ladder (S4) can propose a class-B cut...
    s4 = propose(state, report, max_deltas=16, stage=Stage.S4)
    assert any(delta_kind(d) == KIND_CUT for d in s4)

    # ...but S0 emits ONLY parameter deltas (zero cut/formulation/decomposition).
    s0 = propose(state, report, max_deltas=16, stage=Stage.S0)
    assert s0
    assert all(delta_kind(d) == KIND_PARAM for d in s0)
    assert not any(
        delta_kind(d) in {KIND_CUT, KIND_FORMULATION, KIND_DECOMPOSITION} for d in s0
    )


# ---------------------------------------------------------------------------
# Pure: modeling-agent pipeline + self-correction (stub kernel, no SCIP)
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_modeling_agent_pipeline_excludes_opop_loop() -> None:
    """The happy-path pipeline is exactly NL->model->solve (no opop-loop phases)."""
    ir = _knapsack_with_cover()
    llm = FakeLLMClient(response=json.dumps(milp_to_spec(ir)))
    result = run_modeling_agent(describe_milp(ir), llm=llm, kernel=_StubKernel(), time_limit=1.0)

    assert result.error is None
    assert result.pipeline == ("formulate", "build", "solve", "evaluate")
    assert set(result.pipeline) <= MODELING_AGENT_PHASES
    assert not (set(result.pipeline) & FORBIDDEN_LOOP_PHASES)
    assert result.n_llm_calls == 1


@pytest.mark.smoke
def test_modeling_agent_repairs_invalid_model() -> None:
    """A malformed first reply triggers ONE LLM self-correction that then builds."""
    ir = _knapsack_with_cover()
    good = json.dumps(milp_to_spec(ir))
    calls = {"n": 0}

    def _responder(_message: str) -> str:
        calls["n"] += 1
        return "not json at all" if calls["n"] == 1 else good

    llm = FakeLLMClient(response=_responder)
    result = run_modeling_agent(
        describe_milp(ir), llm=llm, kernel=_StubKernel(), time_limit=1.0, max_repairs=1
    )
    assert result.error is None
    assert result.ir is not None
    assert "repair" in result.pipeline
    assert result.n_repairs == 1
    assert result.n_llm_calls == 2


@pytest.mark.smoke
def test_modeling_agent_unrecoverable_model_records_error() -> None:
    """When every reply is unusable the run fails honestly (no model, no solve)."""
    llm = FakeLLMClient(response="still not valid json")
    result = run_modeling_agent(
        "Minimize nothing.", llm=llm, kernel=_StubKernel(), time_limit=1.0, max_repairs=1
    )
    assert result.ir is None
    assert result.score is None
    assert result.error is not None
    assert not result.solved
    assert result.n_llm_calls == 2  # formulate + 1 repair, both unusable
    assert "build" not in result.pipeline
    assert "solve" not in result.pipeline


# ---------------------------------------------------------------------------
# Integration: modeling-agent end-to-end (real SCIP solve)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_modeling_agent_end_to_end_with_scip(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """Baseline 4 produces a solved model end-to-end with a FakeLLMClient + SCIP."""
    from opop.solver.scip import ScipKernel

    solver_skip_if_missing("scip")
    ir = _knapsack_with_cover()
    llm = FakeLLMClient(
        response=json.dumps(milp_to_spec(ir)), price_input_1m=0.5, price_output_1m=1.5
    )
    result = run_modeling_agent(
        describe_milp(ir), llm=llm, kernel=ScipKernel(), time_limit=5.0, seed=0
    )
    assert result.error is None
    assert result.ir is not None
    assert result.score is not None
    assert result.solved
    assert result.score.metrics["objective"] == pytest.approx(3.0)
    assert result.pipeline == ("formulate", "build", "solve", "evaluate")
    # LLM cost is tracked.
    assert result.llm_summary["calls"] == 1
    assert result.llm_summary["cost_usd"] > 0.0


@pytest.mark.integration
def test_modeling_agent_baseline_single_cell(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """The baseline-4 single-cell runner emits a schema row with tracked LLM cost."""
    solver_skip_if_missing("scip")
    ir = _knapsack_with_cover()
    llm = FakeLLMClient(
        response=json.dumps(milp_to_spec(ir)), price_input_1m=0.5, price_output_1m=1.5
    )
    outcome = run_modeling_agent_baseline(ir, llm, seed=0, time_limit=5.0)

    assert outcome.method == MODELING_AGENT_METHOD
    assert outcome.n_accepted == 0
    assert outcome.n_llm_calls == 1
    assert outcome.solved

    row = outcome.to_row()
    assert set(row) == set(RESULT_COLUMNS)
    assert row["llm_cost_usd"] > 0.0
    assert row["llm_tokens_in"] > 0
    # No opop-loop phases: analyzer / controller / verification stay zero.
    assert row["analyzer_time"] == 0.0
    assert row["controller_time"] == 0.0
    assert row["verification_time"] == 0.0
    assert float(row["time"]) >= float(row["solver_wall_time"]) - 1e-9


# ---------------------------------------------------------------------------
# Integration: params-only baseline emits ONLY class-C param deltas
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_params_only_baseline_emits_only_param_deltas(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """Baseline 3 accepts >=1 param delta and emits ONLY class-C deltas end-to-end."""
    solver_skip_if_missing("scip")
    ir = generate_knapsack(6, seed=0)
    cell_dir = tmp_path / "cell"
    outcome = run_params_only_baseline(
        ir, seed=0, trials=2, time_limit=5.0, out_dir=cell_dir
    )

    assert outcome.method == PARAMS_ONLY_METHOD
    assert outcome.n_accepted >= 1
    assert outcome.n_llm_calls == 0
    # The params-only ablation uses no LLM -> zero LLM cost.
    assert outcome.cost["llm_cost_usd"] == 0.0
    assert outcome.cost["llm_tokens_in"] == 0

    row = outcome.to_row()
    assert set(row) == set(RESULT_COLUMNS)
    assert row["method"] == PARAMS_ONLY_METHOD
    assert float(row["time"]) >= float(row["solver_wall_time"]) - 1e-9

    # Every processed delta is class C (a parameter delta) -> no cuts(B)/formulations(A).
    events_path = cell_dir / "events.jsonl"
    assert events_path.is_file()
    classes = {
        json.loads(line)["delta_class"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert classes, "no events were journalled"
    assert classes <= {"C"}, f"params-only must emit only class-C deltas, saw {classes}"


# ---------------------------------------------------------------------------
# Integration: shared harness + schema-identical results.parquet
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_both_baselines_schema_identical_parquet(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """Both baselines share one results schema; LLM cost only for the modeling agent."""
    solver_skip_if_missing("scip")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    ir = generate_knapsack(6, seed=0)
    outcomes = run_baseline_suite([ir], [0, 1], out_dir=tmp_path, trials=2, time_limit=5.0)
    # 2 seeds x (params-only + modeling-agent) = 4 rows.
    assert len(outcomes) == 4
    for outcome in outcomes:
        assert set(outcome.to_row().keys()) == set(RESULT_COLUMNS)

    path = write_results(outcomes, tmp_path)
    assert path.name == "results.parquet"
    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(RESULT_COLUMNS)

    params_rows = frame[frame["method"] == PARAMS_ONLY_METHOD]
    agent_rows = frame[frame["method"] == MODELING_AGENT_METHOD]
    assert not params_rows.empty
    assert not agent_rows.empty

    # Modeling-agent rows: positive LLM token usage + cost. Params-only: zero.
    assert bool((agent_rows["llm_tokens_in"] > 0).all())
    assert bool((agent_rows["llm_cost_usd"] > 0.0).all())
    assert bool((params_rows["llm_cost_usd"] == 0.0).all())
    assert bool((params_rows["llm_tokens_in"] == 0).all())

    # Honest end-to-end time: never below solver-only time, for every row.
    assert bool((frame["time"] >= frame["solver_wall_time"] - 1e-9).all())

    # Modeling-agent solved the model end-to-end and touched NO opop-loop phase.
    assert bool(agent_rows["solved"].all())
    assert bool((agent_rows["analyzer_time"] == 0.0).all())
    assert bool((agent_rows["controller_time"] == 0.0).all())
    assert bool((agent_rows["verification_time"] == 0.0).all())


@pytest.mark.integration
def test_run_baselines_34_config_entry_writes_results(
    tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
) -> None:
    """The config-driven entry materialises a synthetic instance and writes results."""
    solver_skip_if_missing("scip")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    config = RunConfig(
        name="task37-test",
        split="dev",
        seeds=[0],
        budget=BudgetConfig(trials=1, time_limit_sec=5.0),
        instance_limit=1,
    )
    path = run_baselines_34(config, tmp_path)
    frame = pd.read_parquet(path)
    assert list(frame.columns) == list(RESULT_COLUMNS)
    assert set(frame["method"]) == {PARAMS_ONLY_METHOD, MODELING_AGENT_METHOD}
    assert len(frame) == 2  # 1 instance x 1 seed x 2 baselines
