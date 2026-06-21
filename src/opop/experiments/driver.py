"""Experiment-matrix driver: wire the sweep to real OPOP + baseline runners (task 39).

:class:`MatrixDriver` turns an :class:`~opop.experiments.matrix.ExperimentMatrix`
into a consolidated ``results.parquet`` by routing each cell to a real runner
keyed on its ``ablation``:

* ``scip_default``       → :class:`opop.experiments.baselines.DefaultRunner` (one
  default SCIP solve). Method tag ``scip-default``.
* ``params_only``        → :func:`opop.experiments.baselines_34.run_params_only_baseline`
  (the S0 opop loop). Method tag ``opop-params-only``.
* ``analyzer_cuts_only`` → the opop closed loop with a CUT-ONLY proposer (S1, with
  ``param`` deltas dropped). Method tag ``opop-analyzer-cuts-only``.
* ``params_plus_cuts``   → the opop loop with the S1 proposer. Method tag
  ``opop-params-plus-cuts``.
* ``full_opop``          → the opop loop with the default (S4) proposer. Method
  tag ``opop``.
* ``S0``–``S4``          → the opop loop with the matching
  :class:`opop.proposer.stages.Stage`. Method tag ``opop-<stage>`` (lowercased).

The driver calls :func:`opop.experiments.audit_gate.assert_can_run_split` BEFORE
any work (sealed-lock + held-out guard), reuses the resume-safe
:class:`opop.experiments.runner.LocalRunner` for execution (a completed cell with
an ``ok`` ``cell_done.json`` marker is skipped), writes per-cell artifacts under
``<out_dir>/cells/<slug>/``, aggregates every cell's ``events.jsonl`` +
``verification/*.json`` to the top level, and emits a deterministic
``repro_manifest.json``.

``python -m opop.experiments.driver --out runs/matrix --split dev --instances 2
--ablations full_opop params_only scip_default --seeds 0 1 --time-limits 5`` runs a
small offline sweep over the synthetic dev set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, final

from opop.bench.sources.phase1_set import REGISTRY_PATH, get_phase1_instances
from opop.config import BudgetConfig, RunConfig, load_config
from opop.experiments.audit_gate import assert_can_run_split
from opop.experiments.matrix import (
    CANONICAL_ABLATIONS,
    ExperimentMatrix,
    MatrixCell,
    MatrixStatus,
    cell_marker_path,
    cell_out_dir,
)
from opop.experiments.runner import Job, runner_for
from opop.model.state import ProblemState
from opop.proposer import KIND_PARAM, Stage, delta_kind, propose

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from opop.analyzer.report import AnalysisReport
    from opop.llm.client import LLMClient
    from opop.model.state import Delta
    from opop.orchestrator.result import RunResult

    Proposer = Callable[..., list[Delta]]

__all__ = [
    "MATRIX_RESULT_COLUMNS",
    "MatrixDriver",
    "MatrixDriverError",
    "main",
    "run_matrix",
]

#: Per-solve memory ceiling (MiB); mirrors :data:`opop.run.MEMORY_LIMIT_MB`.
MEMORY_LIMIT_MB: int = 4096

#: The consolidated ``results.parquet`` columns the driver emits for every cell.
MATRIX_RESULT_COLUMNS: tuple[str, ...] = (
    "instance_id",
    "method",
    "ablation",
    "seed",
    "time_limit",
    "primal_integral",
    "gap",
    "time",
    "solved",
    "censored",
    "n_accepted",
)

#: Placeholder ``method`` factor used when building a matrix from factor lists
#: (the comparison-method tag in each row is derived from the cell's ablation).
_MATRIX_METHOD = "matrix"

#: Method tag emitted per ablation (the ``method`` column in the result rows).
_METHOD_BY_ABLATION: dict[str, str] = {
    "scip_default": "scip-default",
    "params_only": "opop-params-only",
    "analyzer_cuts_only": "opop-analyzer-cuts-only",
    "params_plus_cuts": "opop-params-plus-cuts",
    "full_opop": "opop",
    "S0": "opop-s0",
    "S1": "opop-s1",
    "S2": "opop-s2",
    "S3": "opop-s3",
    "S4": "opop-s4",
}

#: opop-closed-loop ablations and the :class:`Stage` each runs the proposer at.
_STAGE_BY_ABLATION: dict[str, Stage] = {
    "params_plus_cuts": Stage.S1,
    "full_opop": Stage.S4,
    "S0": Stage.S0,
    "S1": Stage.S1,
    "S2": Stage.S2,
    "S3": Stage.S3,
    "S4": Stage.S4,
}

#: Ablations routed through the opop closed loop (``run_loop``).
_OPOP_LOOP_ABLATIONS: frozenset[str] = frozenset(_STAGE_BY_ABLATION) | {"analyzer_cuts_only"}

#: Every ablation the driver knows how to run.
_KNOWN_ABLATIONS: frozenset[str] = (
    frozenset({"scip_default", "params_only"}) | _OPOP_LOOP_ABLATIONS
)


class MatrixDriverError(RuntimeError):
    """Raised when the matrix cannot be run (no instances / missing instance id)."""


def _method_for(ablation: str) -> str:
    """Return the comparison ``method`` tag for an ``ablation``."""
    if ablation not in _METHOD_BY_ABLATION:
        raise ValueError(f"unknown ablation {ablation!r}; known: {sorted(_KNOWN_ABLATIONS)}")
    return _METHOD_BY_ABLATION[ablation]


# ---------------------------------------------------------------------------
# Proposer wrappers (match the orchestrator ProposerProto signature)
# ---------------------------------------------------------------------------
def _staged_proposer(stage: Stage) -> Proposer:
    """Return a proposer that restricts the candidate pool to ``stage``."""

    def _proposer(
        state: ProblemState,
        report: AnalysisReport,
        *,
        llm: LLMClient | None = None,
        max_deltas: int = 5,
    ) -> list[Delta]:
        return propose(state, report, llm=llm, max_deltas=max_deltas, stage=stage)

    return _proposer


def _cut_only_proposer(
    state: ProblemState,
    report: AnalysisReport,
    *,
    llm: LLMClient | None = None,
    max_deltas: int = 5,
) -> list[Delta]:
    """Propose at S1 then drop every ``param`` delta — analyzer cuts only."""
    deltas = propose(state, report, llm=llm, max_deltas=max_deltas, stage=Stage.S1)
    return [delta for delta in deltas if delta_kind(delta) != KIND_PARAM]


def _proposer_for(ablation: str) -> Proposer:
    """Return the proposer wrapper for an opop-loop ``ablation``."""
    if ablation == "analyzer_cuts_only":
        return _cut_only_proposer
    return _staged_proposer(_STAGE_BY_ABLATION.get(ablation, Stage.S4))


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
def _matrix_row(cell: MatrixCell, method: str, base: dict[str, Any]) -> dict[str, object]:
    """Normalise a runner's row ``base`` to the canonical matrix-row schema."""
    return {
        "instance_id": cell.instance_id,
        "method": method,
        "ablation": str(cell.ablation),
        "seed": int(cell.seed),
        "time_limit": float(cell.time_limit),
        "primal_integral": float(base.get("primal_integral", float("nan"))),
        "gap": float(base.get("gap", 1.0)),
        "time": float(base.get("time", cell.time_limit)),
        "solved": bool(base.get("solved", False)),
        "censored": bool(base.get("censored", False)),
        "n_accepted": int(base.get("n_accepted", 0)),
    }


def _opop_metrics(result: RunResult, time_limit: float) -> dict[str, Any]:
    """Extract the canonical metrics from an opop ``RunResult`` (cf. ``opop.run``)."""
    if result.incumbent is not None:
        metrics = result.incumbent.score.metrics
        return {
            "primal_integral": metrics.get("primal_integral", float("nan")),
            "gap": metrics.get("gap", 1.0),
            "time": metrics.get("solve_time", time_limit),
            "solved": bool(metrics.get("optimal", 0.0)),
            "censored": bool(metrics.get("censored", 0.0)),
            "n_accepted": result.n_accepted,
        }
    return {
        "primal_integral": float("nan"),
        "gap": 1.0,
        "time": float(time_limit),
        "solved": False,
        "censored": True,
        "n_accepted": result.n_accepted,
    }


# ---------------------------------------------------------------------------
# Artifact aggregation (mirrors opop.run._append_events / _collect_verification)
# ---------------------------------------------------------------------------
def _append_events(cell_events: Path, top_events: Path) -> None:
    """Append a cell's ``events.jsonl`` rows onto the top-level journal."""
    if not cell_events.is_file():
        return
    with top_events.open("a", encoding="utf-8") as out:
        for line in cell_events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.write(line + "\n")


def _collect_verification(cell_dir: Path, top_verification: Path, prefix: str) -> int:
    """Copy a cell's verification certificates up under a namespaced prefix."""
    source = cell_dir / "verification"
    if not source.is_dir():
        return 0
    top_verification.mkdir(parents=True, exist_ok=True)
    copied = 0
    for report in sorted(source.glob("*.json")):
        shutil.copy2(report, top_verification / f"{prefix}_{report.name}")
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@final
class MatrixDriver:
    """Run an :class:`ExperimentMatrix` through real OPOP / baseline runners.

    Args:
        matrix: The sweep to run.
        out_dir: Top-level output directory.
        split: Dataset split (gated by :func:`assert_can_run_split`).
        runner_kind: ``local`` (execute) or ``dry-run`` (list jobs only).
        trials: BO trial budget for the opop-loop / params-only ablations.
        memory_limit_mb: Per-solve memory ceiling.
        one_shot_final: Required ``True`` to run on a held-out split.
        registry_path: Registry whose lock must be sealed.
        instances: Optional pre-materialised ``{instance_id: MILP}`` (skips the
            internal materialisation).
    """

    def __init__(
        self,
        matrix: ExperimentMatrix,
        *,
        out_dir: str | Path,
        split: str = "dev",
        runner_kind: str = "local",
        trials: int = 2,
        memory_limit_mb: int = MEMORY_LIMIT_MB,
        one_shot_final: bool = False,
        registry_path: str | Path = REGISTRY_PATH,
        instances: dict[str, Any] | None = None,
    ) -> None:
        self.matrix = matrix
        self.out_dir = Path(out_dir)
        self.split = split
        self.runner_kind = runner_kind
        self.trials = int(trials)
        self.memory_limit_mb = int(memory_limit_mb)
        self.one_shot_final = one_shot_final
        self.registry_path = registry_path
        self._instances = instances

    def run(self) -> Path:
        """Run the matrix; return the path to ``results.parquet`` (or the plan)."""
        assert_can_run_split(
            self.split, registry_path=self.registry_path, one_shot_final=self.one_shot_final
        )
        cells = self.matrix.expand()
        self._validate_ablations(cells)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        runner = runner_for(self.runner_kind)
        if self.runner_kind == "dry-run":
            return self._write_plan(runner.submit_jobs(cells))

        by_id = self._instances if self._instances is not None else self._materialize()
        self._validate_instances(cells, by_id)
        runner.submit_jobs(cells, work_fn=self._make_work_fn(by_id), out_dir=self.out_dir)
        return self._consolidate(cells)

    # -- validation ---------------------------------------------------------
    @staticmethod
    def _validate_ablations(cells: Sequence[MatrixCell]) -> None:
        unknown = sorted({str(c.ablation) for c in cells} - _KNOWN_ABLATIONS)
        if unknown:
            raise ValueError(
                f"unknown ablation(s) {unknown}; known: {sorted(_KNOWN_ABLATIONS)}"
            )

    def _materialize(self) -> dict[str, Any]:
        instances = get_phase1_instances(self.split, sources=("synthetic",))
        if not instances:
            raise MatrixDriverError(f"no synthetic instances for split {self.split!r}")
        return {ir.name: ir for ir in instances}

    @staticmethod
    def _validate_instances(cells: Sequence[MatrixCell], by_id: dict[str, Any]) -> None:
        missing = sorted({c.instance_id for c in cells} - set(by_id))
        if missing:
            raise MatrixDriverError(f"matrix references unmaterialised instances: {missing}")

    # -- per-cell work ------------------------------------------------------
    def _make_work_fn(self, by_id: dict[str, Any]) -> Callable[[MatrixCell], dict[str, object]]:
        def _work(cell: MatrixCell) -> dict[str, object]:
            return self._dispatch(cell, by_id)

        return _work

    def _dispatch(self, cell: MatrixCell, by_id: dict[str, Any]) -> dict[str, object]:
        ablation = str(cell.ablation)
        ir = by_id[cell.instance_id]
        if ablation == "scip_default":
            return self._scip_default_row(ir, cell)
        if ablation == "params_only":
            return self._params_only_row(ir, cell)
        if ablation in _OPOP_LOOP_ABLATIONS:
            return self._opop_loop_row(ir, cell, ablation)
        raise ValueError(f"unknown ablation {ablation!r}")

    def _scip_default_row(self, ir: Any, cell: MatrixCell) -> dict[str, object]:
        from opop.experiments.baselines import DefaultRunner
        from opop.solver.scip import ScipKernel

        runner = DefaultRunner(ScipKernel(), method_name="scip-default")
        base = runner.solve_one(
            ir, int(cell.seed), trials=1, time_limit=float(cell.time_limit)
        )
        return _matrix_row(cell, "scip-default", base)

    def _params_only_row(self, ir: Any, cell: MatrixCell) -> dict[str, object]:
        from opop.experiments.baselines_34 import run_params_only_baseline

        outcome = run_params_only_baseline(
            ir,
            seed=int(cell.seed),
            trials=self.trials,
            time_limit=float(cell.time_limit),
            out_dir=cell_out_dir(cell, self.out_dir),
            memory_limit_mb=self.memory_limit_mb,
        )
        return _matrix_row(cell, outcome.method, outcome.to_row())

    def _opop_loop_row(self, ir: Any, cell: MatrixCell, ablation: str) -> dict[str, object]:
        from opop.analyzer.api import analyze
        from opop.controller.encoder import default_phase1_space
        from opop.controller.phase1 import Phase1Controller
        from opop.evaluator import evaluate
        from opop.orchestrator.loop import run_loop
        from opop.solver.scip import ScipKernel
        from opop.verify.gate import verify_delta

        seed = int(cell.seed)
        time_limit = float(cell.time_limit)
        n_trials = max(1, self.trials)
        controller = Phase1Controller.bo(
            default_phase1_space(),
            n_trials=n_trials,
            n_init=min(3, n_trials),
            n_candidates=64,
            seed=seed,
        )
        state = ProblemState(instance_id=ir.name, task_family="MILP", budget_state={"ir": ir})
        config = RunConfig(
            seeds=[seed], budget=BudgetConfig(trials=n_trials, time_limit_sec=time_limit)
        )
        result = run_loop(
            state,
            config,
            kernel=ScipKernel(),
            proposer=_proposer_for(ablation),
            analyzer=analyze,
            verifier=verify_delta,
            evaluator=evaluate,
            controller=controller,
            out_dir=cell_out_dir(cell, self.out_dir),
            reference_optimum=None,
            memory_limit_mb=self.memory_limit_mb,
            instance_id=ir.name,
        )
        return _matrix_row(cell, _method_for(ablation), _opop_metrics(result, time_limit))

    # -- consolidation ------------------------------------------------------
    def _write_plan(self, jobs: Sequence[Job]) -> Path:
        plan_path = self.out_dir / "matrix_plan.json"
        payload = {
            "split": self.split,
            "runner_kind": self.runner_kind,
            "n_jobs": len(jobs),
            "jobs": [job.to_dict() for job in jobs],
        }
        plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return plan_path

    def _collect_rows(self, cells: Sequence[MatrixCell]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell in cells:
            marker = cell_marker_path(cell, self.out_dir)
            if not marker.is_file():
                continue
            try:
                data: Any = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("status") == "ok":
                result = data.get("result")
                if isinstance(result, dict):
                    rows.append(cast("dict[str, Any]", result))
        return rows

    def _write_results(self, rows: Sequence[dict[str, Any]]) -> Path:
        parquet_path = self.out_dir / "results.parquet"
        records = [{col: row.get(col) for col in MATRIX_RESULT_COLUMNS} for row in rows]
        try:
            import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

            pd.DataFrame(records).reindex(columns=list(MATRIX_RESULT_COLUMNS)).to_parquet(
                parquet_path
            )
        except ImportError:
            json_path = self.out_dir / "results.json"
            json_path.write_text(
                json.dumps(records, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            return json_path
        return parquet_path

    def _aggregate_artifacts(self, cells: Sequence[MatrixCell]) -> None:
        top_events = self.out_dir / "events.jsonl"
        top_events.write_text("", encoding="utf-8")
        top_verification = self.out_dir / "verification"
        for cell in cells:
            cdir = cell_out_dir(cell, self.out_dir)
            _append_events(cdir / "events.jsonl", top_events)
            _collect_verification(cdir, top_verification, cell.slug)

    def _write_manifest(
        self, cells: Sequence[MatrixCell], n_rows: int, results_path: Path
    ) -> None:
        status = MatrixStatus.scan(cells, self.out_dir)
        manifest = {
            "split": self.split,
            "runner_kind": self.runner_kind,
            "one_shot_final": self.one_shot_final,
            "trials": self.trials,
            "matrix": {
                "instances": list(self.matrix.instances),
                "methods": list(self.matrix.methods),
                "ablations": list(self.matrix.ablations),
                "seeds": list(self.matrix.seeds),
                "time_limits": list(self.matrix.time_limits),
            },
            "n_cells": len(cells),
            "n_rows": n_rows,
            "results": results_path.name,
            "status": status.to_dict(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.out_dir / "repro_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _consolidate(self, cells: Sequence[MatrixCell]) -> Path:
        rows = self._collect_rows(cells)
        results_path = self._write_results(rows)
        self._aggregate_artifacts(cells)
        self._write_manifest(cells, len(rows), results_path)
        return results_path


# ---------------------------------------------------------------------------
# High-level helper + CLI
# ---------------------------------------------------------------------------
def run_matrix(
    *,
    out_dir: str | Path,
    split: str = "dev",
    ablations: Sequence[str] = CANONICAL_ABLATIONS,
    instance_limit: int = 2,
    seeds: Sequence[int] = (0,),
    time_limits: Sequence[float] = (5.0,),
    trials: int = 2,
    runner_kind: str = "local",
    one_shot_final: bool = False,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    registry_path: str | Path = REGISTRY_PATH,
) -> Path:
    """Build a matrix over the synthetic ``split`` instances and run it.

    Gates the split FIRST (so a held-out split is blocked before any
    materialisation), materialises the first ``instance_limit`` synthetic
    instances, builds an :class:`ExperimentMatrix` over them, and runs a
    :class:`MatrixDriver`. Returns the consolidated ``results.parquet`` path (or
    the plan path for ``runner_kind="dry-run"``).
    """
    assert_can_run_split(split, registry_path=registry_path, one_shot_final=one_shot_final)
    materialized = get_phase1_instances(split, sources=("synthetic",))[: max(0, int(instance_limit))]
    if not materialized:
        raise MatrixDriverError(f"no synthetic instances for split {split!r}")
    by_id = {ir.name: ir for ir in materialized}
    matrix = ExperimentMatrix(
        instances=tuple(by_id),
        methods=(_MATRIX_METHOD,),
        ablations=tuple(str(a) for a in ablations),
        seeds=tuple(int(s) for s in seeds),
        time_limits=tuple(float(t) for t in time_limits),
    )
    driver = MatrixDriver(
        matrix,
        out_dir=out_dir,
        split=split,
        runner_kind=runner_kind,
        trials=trials,
        memory_limit_mb=memory_limit_mb,
        one_shot_final=one_shot_final,
        registry_path=registry_path,
        instances=by_id,
    )
    return driver.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opop.experiments.driver",
        description="Run the OPOP experiment matrix (ablation sweep) on the synthetic dev set.",
    )
    parser.add_argument("--config", type=Path, default=None, help="run config (.yaml/.json)")
    parser.add_argument("--out", required=True, type=Path, help="output run directory")
    parser.add_argument("--split", default=None, help="dataset split (default: dev / config)")
    parser.add_argument("--instances", type=int, default=None, help="instance cap (first N)")
    parser.add_argument("--ablations", nargs="+", default=None, help="ablation rows to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds")
    parser.add_argument("--time-limits", nargs="+", type=float, default=None, help="per-solve budgets")
    parser.add_argument("--trials", type=int, default=None, help="BO trial budget per loop cell")
    parser.add_argument("--runner", default="local", help="runner kind (local / dry-run)")
    parser.add_argument(
        "--one-shot-final", action="store_true", help="permit a held-out split (final eval)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build the sweep from flags / config and run it."""
    args = _build_parser().parse_args(argv)
    config = load_config(args.config) if args.config is not None else None

    split = args.split or (config.split if config is not None else "dev")
    if args.instances is not None:
        instance_limit = args.instances
    elif config is not None and config.instance_limit is not None:
        instance_limit = config.instance_limit
    else:
        instance_limit = 2
    ablations = args.ablations or ["full_opop", "params_only", "scip_default"]
    seeds = args.seeds if args.seeds is not None else (list(config.seeds) if config is not None else [0])
    if args.time_limits is not None:
        time_limits = args.time_limits
    elif config is not None:
        time_limits = [config.budget.time_limit_sec]
    else:
        time_limits = [5.0]
    trials = args.trials if args.trials is not None else (config.budget.trials if config is not None else 2)

    path = run_matrix(
        out_dir=args.out,
        split=split,
        ablations=ablations,
        instance_limit=instance_limit,
        seeds=seeds,
        time_limits=time_limits,
        trials=trials,
        runner_kind=args.runner,
        one_shot_final=args.one_shot_final,
    )
    print(f"matrix driver: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
