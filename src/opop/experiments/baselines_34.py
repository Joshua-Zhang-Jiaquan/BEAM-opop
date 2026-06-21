"""Baselines 3-4 (plan task 37): params-only ablation + LLM modeling-agent-only.

Two baselines that share ONE harness + cost accounting (:mod:`opop.bench.cost`)
and emit a schema-identical ``results.parquet`` so they slot straight into the
same comparison pipeline (:mod:`opop.experiments.compare`) as opop and the other
baseline families (tasks 36 / 38):

* **Baseline 3 — ``opop-params-only``** (:func:`run_params_only_baseline`).
  The FULL opop closed loop (:func:`opop.orchestrator.loop.run_loop` — analyzer +
  verification gate + Bayesian controller) with the proposer restricted to the
  staged search space **S0** (:class:`opop.proposer.stages.Stage`). Because S0
  filters out every analyzer cut, formulation family, and decomposition lever
  BEFORE selection, only class-C SCIP parameter deltas are ever proposed/accepted.
  This is the credit-assignment ablation that isolates the analyzer/formulation
  contribution (thesis T4): everything but the design space matches full opop, so
  its cost row carries real ``analyzer_time`` / ``controller_time`` /
  ``verification_time``.

* **Baseline 4 — ``modeling-agent``** (:func:`run_modeling_agent_baseline`).
  An LLM modeling-agent-only baseline (OptiMUS / LLMOPT / ORLM / OR-R1 style) that
  runs NL -> model -> solve via :mod:`opop.experiments.modeling_agent`, WITHOUT the
  analyzer, the verification gate, or the BO loop. Its ``analyzer_time`` /
  ``controller_time`` / ``verification_time`` are therefore always ``0.0`` — itself
  the evidence of no opop-loop blending — while the LLM modeling call is booked
  under ``proposer_time`` and per-row LLM tokens/cost are tracked.

Schema (:data:`RESULT_COLUMNS`) = base columns + every
:data:`opop.bench.cost.COST_FIELDS` column, so the headline ``time`` is honest
end-to-end wall time (``time >= solver_wall_time`` always) and LLM cost is a
first-class column populated only where an LLM is actually used.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opop.analyzer.api import analyze
from opop.bench.cost import COST_FIELDS, empty_cost, make_event_cost
from opop.config import BudgetConfig, RunConfig, load_config
from opop.controller.encoder import default_phase1_space
from opop.controller.phase1 import Phase1Controller
from opop.evaluator import evaluate
from opop.experiments.modeling_agent import (
    describe_milp,
    milp_to_spec,
    run_modeling_agent,
)
from opop.llm.client import FakeLLMClient
from opop.model.state import ProblemState
from opop.orchestrator.loop import run_loop
from opop.proposer import Stage, propose
from opop.solver.scip import ScipKernel
from opop.verify.gate import verify_delta

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from opop.analyzer.report import AnalysisReport
    from opop.llm.client import LLMClient
    from opop.model.ir import MILP
    from opop.model.state import Delta

logger = logging.getLogger(__name__)

__all__ = [
    "MEMORY_LIMIT_MB",
    "MODELING_AGENT_METHOD",
    "PARAMS_ONLY_METHOD",
    "RESULT_COLUMNS",
    "BaselineError",
    "BaselineOutcome",
    "main",
    "run_baseline_suite",
    "run_baselines_34",
    "run_modeling_agent_baseline",
    "run_params_only_baseline",
    "write_results",
]

#: Method tag for baseline 3 (the S0 params-only opop ablation).
PARAMS_ONLY_METHOD: str = "opop-params-only"
#: Method tag for baseline 4 (the LLM modeling-agent-only baseline).
MODELING_AGENT_METHOD: str = "modeling-agent"

#: Per-solve memory ceiling (MiB); mirrors :data:`opop.run.MEMORY_LIMIT_MB`.
MEMORY_LIMIT_MB: int = 4096

#: Default per-1M-token prices for the offline modeling-agent fake client, so
#: ``llm_cost_usd`` is non-zero and exercised (only the relative scale matters).
_FAKE_PRICE_INPUT_1M: float = 0.5
_FAKE_PRICE_OUTPUT_1M: float = 1.5

# Base (non-cost) result columns; the full schema appends every cost column.
_BASE_COLUMNS: tuple[str, ...] = (
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
    "n_llm_calls",
)

#: The canonical (stable-order) results schema both baselines emit: the base
#: columns followed by every :data:`opop.bench.cost.COST_FIELDS` column. Both
#: baseline 3 and baseline 4 emit EXACTLY these columns so results are
#: schema-identical.
RESULT_COLUMNS: tuple[str, ...] = (*_BASE_COLUMNS, *COST_FIELDS)

#: Metrics for a cell that produced no feasible incumbent / solved model.
_NO_INCUMBENT_METRICS: dict[str, float] = {
    "primal_integral": float("nan"),
    "gap": 1.0,
    "optimal": 0.0,
    "censored": 1.0,
    "feasible": 0.0,
}


class BaselineError(RuntimeError):
    """Raised when a baseline run cannot proceed (e.g. no instances to run)."""


# ---------------------------------------------------------------------------
# Outcome record
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    """The result of one baseline run on a single ``(instance, seed)``.

    Attributes:
        method: ``opop-params-only`` or ``modeling-agent``.
        instance_id: The MILP instance name.
        seed: The RNG seed used.
        metrics: The evaluator metric dict (``primal_integral`` / ``gap`` /
            ``optimal`` / ``censored`` / ...); the no-incumbent fallback otherwise.
        cost: The flat per-row cost dict (:func:`opop.bench.cost.make_event_cost`).
        time_limit: The per-solve time budget used.
        n_accepted: Accepted deltas (params for baseline 3; 0 for baseline 4).
        n_llm_calls: Number of LLM calls (0 for the no-LLM params-only run).
        detail: Baseline-specific context (stage / pipeline / error); not a column.
    """

    method: str
    instance_id: str
    seed: int
    metrics: dict[str, float]
    cost: dict[str, Any]
    time_limit: float
    n_accepted: int
    n_llm_calls: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        """Whether the solver certified optimality."""
        return bool(self.metrics.get("optimal", 0.0))

    @property
    def feasible(self) -> bool:
        """Whether a feasible incumbent was found."""
        return bool(self.metrics.get("feasible", 0.0))

    def to_row(self) -> dict[str, Any]:
        """Return the schema-identical ``results.parquet`` row for this outcome."""
        m = self.metrics
        row: dict[str, Any] = {
            "instance_id": self.instance_id,
            "method": self.method,
            "seed": int(self.seed),
            "primal_integral": float(m.get("primal_integral", float("nan"))),
            "gap": float(m.get("gap", 1.0)),
            "time": float(self.cost["total_wall_time"]),
            "solved": bool(m.get("optimal", 0.0)),
            "censored": bool(m.get("censored", 0.0)),
            "time_limit": float(self.time_limit),
            "n_accepted": int(self.n_accepted),
            "n_llm_calls": int(self.n_llm_calls),
        }
        for col in COST_FIELDS:
            row[col] = self.cost[col]
        return row


# ---------------------------------------------------------------------------
# LLM token snapshotting (per-run delta over a possibly-shared tracker)
# ---------------------------------------------------------------------------
def _tracker_snapshot(llm: LLMClient | None) -> tuple[int, int, float]:
    """Read cumulative ``(tokens_in, tokens_out, cost_usd)`` from ``llm.tracker``."""
    if llm is None:
        return (0, 0, 0.0)
    tracker = getattr(llm, "tracker", None)
    if tracker is None:
        return (0, 0, 0.0)
    tokens_in = int(getattr(tracker, "total_tokens_in", 0) or 0)
    tokens_out = int(getattr(tracker, "total_tokens_out", 0) or 0)
    cost_usd = float(getattr(tracker, "total_cost_usd", 0.0) or 0.0)
    return (tokens_in, tokens_out, cost_usd)


def _read_run_cost(out_dir: Path) -> dict[str, Any]:
    """Read run_loop's authoritative ``cost_run_total`` (a make_event_cost dict).

    Returns an all-zero cost row when the result file or the cost block is
    absent, so the harness never crashes on a missing artifact.
    """
    merged = empty_cost()
    result_path = out_dir / "result.json"
    if not result_path.is_file():
        return merged
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return merged
    if not isinstance(raw, dict):
        return merged
    payload: dict[str, Any] = raw
    cost = payload.get("cost_run_total")
    if isinstance(cost, dict):
        cost_map: dict[str, Any] = cost
        for key in COST_FIELDS:
            if key in cost_map:
                merged[key] = cost_map[key]
    return merged


# ---------------------------------------------------------------------------
# Baseline 3 — opop restricted to S0 (params only)
# ---------------------------------------------------------------------------
def _s0_proposer(
    state: ProblemState,
    report: AnalysisReport,
    *,
    llm: LLMClient | None = None,
    max_deltas: int = 5,
) -> list[Delta]:
    """opop proposer restricted to S0 — emits only class-C parameter deltas.

    Wraps :func:`opop.proposer.propose` with ``stage=Stage.S0`` so the staged
    search-space gate drops every cut / formulation / decomposition candidate
    BEFORE selection. The signature matches the orchestrator's ``ProposerProto``
    so it drops straight into :func:`opop.orchestrator.loop.run_loop`.
    """
    return propose(state, report, llm=llm, max_deltas=max_deltas, stage=Stage.S0)


def run_params_only_baseline(
    ir: MILP,
    *,
    seed: int,
    trials: int,
    time_limit: float,
    out_dir: str | Path,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    llm: LLMClient | None = None,
    reference_optimum: float | None = None,
) -> BaselineOutcome:
    """Run baseline 3 (params-only opop ablation) on one ``(instance, seed)``.

    Drives the full opop closed loop with the proposer pinned to S0, so only
    parameter deltas are proposed/accepted. The run's artifacts (incl.
    ``events.jsonl`` and the authoritative ``cost_run_total``) are written under
    ``out_dir``; the headline cost is read straight back from ``result.json``.

    Args:
        ir: The MILP instance.
        seed: RNG / solver seed.
        trials: BO trial budget (loop iterations).
        time_limit: Per-solve time budget (seconds).
        out_dir: Directory for this cell's run_loop artifacts.
        memory_limit_mb: Per-solve memory ceiling (MiB).
        llm: Optional LLM client for S0 delta selection (the canonical ablation
            uses the rule-based ranker, so this is ``None`` and LLM cost is zero).
        reference_optimum: Known optimum forwarded to the evaluator.

    Returns:
        A :class:`BaselineOutcome` tagged :data:`PARAMS_ONLY_METHOD`.
    """
    cell_dir = Path(out_dir)
    n_trials = max(1, int(trials))
    controller = Phase1Controller.bo(
        default_phase1_space(),
        n_trials=n_trials,
        n_init=min(3, n_trials),
        n_candidates=64,
        seed=int(seed),
    )
    state = ProblemState(instance_id=ir.name, task_family="MILP", budget_state={"ir": ir})
    config = RunConfig(
        seeds=[int(seed)],
        budget=BudgetConfig(trials=n_trials, time_limit_sec=float(time_limit)),
    )
    result = run_loop(
        state,
        config,
        kernel=ScipKernel(),
        proposer=_s0_proposer,
        analyzer=analyze,
        verifier=verify_delta,
        evaluator=evaluate,
        controller=controller,
        llm=llm,
        out_dir=cell_dir,
        reference_optimum=reference_optimum,
        memory_limit_mb=int(memory_limit_mb),
        instance_id=ir.name,
    )

    if result.incumbent is not None:
        metrics = dict(result.incumbent.score.metrics)
    else:
        metrics = dict(_NO_INCUMBENT_METRICS)

    cost = _read_run_cost(cell_dir)
    n_llm_calls = int(getattr(getattr(llm, "tracker", None), "calls", 0) or 0)
    return BaselineOutcome(
        method=PARAMS_ONLY_METHOD,
        instance_id=ir.name,
        seed=int(seed),
        metrics=metrics,
        cost=cost,
        time_limit=float(time_limit),
        n_accepted=int(result.n_accepted),
        n_llm_calls=n_llm_calls,
        detail={"stage": "S0", "events_path": str(cell_dir / "events.jsonl")},
    )


# ---------------------------------------------------------------------------
# Baseline 4 — LLM modeling-agent-only (NL -> model -> solve)
# ---------------------------------------------------------------------------
def _default_modeling_llm_factory(ir: MILP) -> LLMClient:
    """Build a deterministic offline modeling LLM that models ``ir`` correctly.

    The returned :class:`FakeLLMClient` replies with the instance's exact JSON
    model spec — a competent modeling agent — so the NL -> model -> solve pipeline
    runs end-to-end offline. Non-zero token prices make ``llm_cost_usd`` realistic.
    """
    response = json.dumps(milp_to_spec(ir))
    return FakeLLMClient(
        response=response,
        price_input_1m=_FAKE_PRICE_INPUT_1M,
        price_output_1m=_FAKE_PRICE_OUTPUT_1M,
    )


def run_modeling_agent_baseline(
    ir: MILP,
    llm: LLMClient,
    *,
    seed: int,
    time_limit: float,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    reference_optimum: float | None = None,
    max_repairs: int = 1,
) -> BaselineOutcome:
    """Run baseline 4 (LLM modeling-agent-only) on one ``(instance, seed)``.

    Renders ``ir`` to a natural-language statement, lets ``llm`` model it
    (NL -> JSON -> MILP via :func:`opop.experiments.modeling_agent.run_modeling_agent`),
    solves it, and scores the trace — with NO analyzer / verification gate / BO
    loop. The LLM modeling wall time is booked under ``proposer_time`` and the
    per-run LLM tokens/cost are tracked; ``analyzer_time`` / ``controller_time`` /
    ``verification_time`` stay ``0.0`` (the evidence of no opop-loop blending).

    Args:
        ir: The MILP instance.
        llm: The LLM client (a ``FakeLLMClient`` in tests).
        seed: Solver seed.
        time_limit: Per-solve time budget (seconds).
        memory_limit_mb: Per-solve memory ceiling (MiB).
        reference_optimum: Known optimum forwarded to the evaluator.
        max_repairs: Maximum LLM self-correction calls on a malformed model.

    Returns:
        A :class:`BaselineOutcome` tagged :data:`MODELING_AGENT_METHOD`.
    """
    token_base = _tracker_snapshot(llm)
    result = run_modeling_agent(
        describe_milp(ir),
        llm=llm,
        kernel=ScipKernel(),
        time_limit=float(time_limit),
        memory_limit_mb=int(memory_limit_mb),
        seed=int(seed),
        reference_optimum=reference_optimum,
        instance_id=ir.name,
        max_repairs=int(max_repairs),
    )

    cur = _tracker_snapshot(llm)
    d_in = max(0, cur[0] - token_base[0])
    d_out = max(0, cur[1] - token_base[1])
    d_cost = max(0.0, cur[2] - token_base[2])
    cost = make_event_cost(
        solver_wall_time=result.timings.get("solve", 0.0),
        proposer_time=result.timings.get("formulate", 0.0),
        evaluate_time=result.timings.get("evaluate", 0.0),
        llm_tokens_in=d_in,
        llm_tokens_out=d_out,
        llm_cost_usd=d_cost,
    )

    metrics = (
        dict(result.score.metrics) if result.score is not None else dict(_NO_INCUMBENT_METRICS)
    )
    return BaselineOutcome(
        method=MODELING_AGENT_METHOD,
        instance_id=ir.name,
        seed=int(seed),
        metrics=metrics,
        cost=cost,
        time_limit=float(time_limit),
        n_accepted=0,
        n_llm_calls=result.n_llm_calls,
        detail={
            "pipeline": list(result.pipeline),
            "n_repairs": result.n_repairs,
            "error": result.error,
        },
    )


# ---------------------------------------------------------------------------
# Shared harness + results IO
# ---------------------------------------------------------------------------
def run_baseline_suite(
    instances: Sequence[MILP],
    seeds: Sequence[int],
    *,
    out_dir: str | Path,
    trials: int = 2,
    time_limit: float = 5.0,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    llm_factory: Callable[[MILP], LLMClient] | None = None,
    reference_optima: Mapping[str, float] | None = None,
) -> list[BaselineOutcome]:
    """Run BOTH baselines over ``instances`` x ``seeds`` (the shared harness).

    For each ``(instance, seed)`` runs baseline 3 (params-only opop) and baseline
    4 (modeling-agent). Baseline 3's run_loop artifacts are written under
    ``out_dir/instances/opop-params-only/<id>_<seed>``; baseline 4 is pure (no
    artifacts). A fresh LLM client is built per modeling-agent cell.

    Args:
        instances: MILP instances.
        seeds: RNG seeds.
        out_dir: Root directory for per-cell baseline-3 artifacts.
        trials: BO trial budget for baseline 3.
        time_limit: Per-solve time budget.
        memory_limit_mb: Inner-solve memory ceiling.
        llm_factory: ``ir -> LLMClient`` factory for baseline 4 (defaults to a
            deterministic offline fake modeling each instance correctly).
        reference_optima: Optional ``instance_id -> known optimum`` mapping.

    Returns:
        A flat list of :class:`BaselineOutcome` (two per ``(instance, seed)``).
    """
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = llm_factory if llm_factory is not None else _default_modeling_llm_factory
    outcomes: list[BaselineOutcome] = []
    for ir in instances:
        ref = reference_optima.get(ir.name) if reference_optima else None
        for seed in seeds:
            cell_dir = run_dir / "instances" / PARAMS_ONLY_METHOD / f"{ir.name}_{int(seed)}"
            outcomes.append(
                run_params_only_baseline(
                    ir,
                    seed=int(seed),
                    trials=trials,
                    time_limit=time_limit,
                    out_dir=cell_dir,
                    memory_limit_mb=memory_limit_mb,
                    reference_optimum=ref,
                )
            )
            outcomes.append(
                run_modeling_agent_baseline(
                    ir,
                    factory(ir),
                    seed=int(seed),
                    time_limit=time_limit,
                    memory_limit_mb=memory_limit_mb,
                    reference_optimum=ref,
                )
            )
    return outcomes


def write_results(outcomes: Sequence[BaselineOutcome], out_dir: str | Path) -> Path:
    """Persist outcomes to ``<out_dir>/results.parquet`` (pandas) else JSON fallback.

    Columns are reindexed to the stable :data:`RESULT_COLUMNS` order so the two
    baselines always share an identical on-disk schema.
    """
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [o.to_row() for o in outcomes]
    parquet_path = run_dir / "results.parquet"
    try:
        import pandas as pd

        frame = pd.DataFrame(rows).reindex(columns=list(RESULT_COLUMNS))
        frame.to_parquet(parquet_path)
    except ImportError:
        json_path = run_dir / "results.json"
        json_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return json_path
    return parquet_path


def _materialize_instances(config: RunConfig) -> list[MILP]:
    """Materialise the synthetic dev/validation instances for ``config`` (offline)."""
    from opop.bench.sources.phase1_set import get_phase1_instances

    instances = get_phase1_instances(config.split, sources=("synthetic",))
    return instances[: config.instance_limit]


def run_baselines_34(
    config: RunConfig,
    out_dir: str | Path,
    *,
    llm_factory: Callable[[MILP], LLMClient] | None = None,
) -> Path:
    """Materialise ``config``'s synthetic instances, run both baselines, write results.

    Returns the path to the written ``results.parquet`` (or ``results.json``).
    """
    instances = _materialize_instances(config)
    if not instances:
        raise BaselineError(
            f"no synthetic instances for split {config.split!r}; nothing to run"
        )
    outcomes = run_baseline_suite(
        instances,
        config.seeds,
        out_dir=out_dir,
        trials=int(config.budget.trials),
        time_limit=float(config.budget.time_limit_sec),
        llm_factory=llm_factory,
    )
    return write_results(outcomes, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opop.experiments.baselines_34",
        description="Baselines 3-4: params-only opop ablation + LLM modeling-agent-only.",
    )
    parser.add_argument("--config", required=True, type=Path, help="run config (.yaml/.json)")
    parser.add_argument("--out", required=True, type=Path, help="output run directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--config``/``--out`` -> run both baselines and write ``results.parquet``."""
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    path = run_baselines_34(config, args.out)
    print(f"baselines_34: wrote results to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
