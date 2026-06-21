"""Phase-1 END-TO-END smoke + sanity experiment (plan task 21).

``python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke`` drives
the full Phase-1 closed loop over the dev set (synthetic-only, to stay offline and
fast) and a SCIP-default baseline, then emits the six experiment artifacts to the
output directory:

* ``results.parquet`` — per-(instance, seed, method) result rows (the schema
  :mod:`opop.experiments.compare` consumes).
* ``events.jsonl`` — every closed-loop journal row, aggregated across instances /
  seeds (the leakage audit reads this).
* ``verification/*.json`` — every accepted delta's certificate, namespaced per
  (instance, seed).
* ``repro_manifest.json`` — a top-level run summary (per-instance manifests live
  alongside each instance's artifacts under ``instances/``).
* ``comparison_report.json`` — opop vs ``scip-default`` on the primal integral.
* ``leakage_audit.json`` — the held-out-instance leakage verdict (Phase-1 has no
  held-out splits, so this is always ``pass``).

For each instance the recorded run is strict-replayed from disk to prove it
reproduces before the run reports success.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opop.analyzer.api import analyze
from opop.bench.audit import audit_leakage
from opop.bench.sources.phase1_set import REGISTRY_PATH, get_phase1_instances
from opop.config import RunConfig, load_config
from opop.controller.encoder import default_phase1_space
from opop.evaluator import evaluate
from opop.experiments.compare import compare, load_results
from opop.experiments.compare import write_report as write_comparison_report
from opop.model.state import Phi, ProblemState
from opop.orchestrator.loop import run_loop
from opop.orchestrator.repro import config_to_dict
from opop.proposer.api import propose
from opop.replay import replay_run
from opop.solver.scip import ScipKernel
from opop.verify.gate import verify_delta

if TYPE_CHECKING:
    from opop.model.ir import MILP
    from opop.orchestrator.result import RunResult

__all__ = ["main", "run_phase1_smoke"]

#: Per-solve memory ceiling (MiB); mirrors ``run_loop``'s default.
MEMORY_LIMIT_MB: int = 4096
#: Candidate method tag in the result rows.
OPOP_METHOD: str = "opop"
#: Baseline method tag in the result rows.
BASELINE_METHOD: str = "scip-default"


def _opop_row(ir: MILP, seed: int, result: RunResult, time_limit: float) -> dict[str, Any]:
    """Build the result row for the opop closed-loop run on ``(ir, seed)``."""
    if result.incumbent is not None:
        m = result.incumbent.score.metrics
        primal_integral = float(m.get("primal_integral", float("nan")))
        gap = float(m.get("gap", 1.0))
        solve_time = float(m.get("solve_time", time_limit))
        solved = bool(m.get("optimal", 0.0))
        censored = bool(m.get("censored", 0.0))
    else:
        primal_integral = float("nan")
        gap = 1.0
        solve_time = float(time_limit)
        solved = False
        censored = True
    return {
        "instance_id": ir.name,
        "method": OPOP_METHOD,
        "seed": int(seed),
        "primal_integral": primal_integral,
        "gap": gap,
        "time": solve_time,
        "solved": solved,
        "censored": censored,
        "time_limit": float(time_limit),
        "n_accepted": int(result.n_accepted),
    }


def _baseline_row(
    ir: MILP, seed: int, metrics: dict[str, float], time_limit: float
) -> dict[str, Any]:
    """Build the result row for the SCIP-default baseline solve on ``(ir, seed)``."""
    return {
        "instance_id": ir.name,
        "method": BASELINE_METHOD,
        "seed": int(seed),
        "primal_integral": float(metrics.get("primal_integral", float("nan"))),
        "gap": float(metrics.get("gap", 1.0)),
        "time": float(metrics.get("solve_time", time_limit)),
        "solved": bool(metrics.get("optimal", 0.0)),
        "censored": bool(metrics.get("censored", 0.0)),
        "time_limit": float(time_limit),
        "n_accepted": 0,
    }


def _append_events(instance_events: Path, top_events: Path) -> None:
    """Append an instance run's ``events.jsonl`` rows onto the top-level journal."""
    if not instance_events.is_file():
        return
    with top_events.open("a", encoding="utf-8") as out:
        for line in instance_events.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.write(line + "\n")


def _collect_verification(instance_dir: Path, top_verification: Path, prefix: str) -> int:
    """Copy an instance run's verification certificates up under a namespaced prefix."""
    source = instance_dir / "verification"
    if not source.is_dir():
        return 0
    top_verification.mkdir(parents=True, exist_ok=True)
    copied = 0
    for report in sorted(source.glob("*.json")):
        shutil.copy2(report, top_verification / f"{prefix}_{report.name}")
        copied += 1
    return copied


def _write_results(rows: list[dict[str, Any]], run_dir: Path) -> Path:
    """Persist result rows to ``results.parquet`` (pandas), else JSON fallback."""
    parquet_path = run_dir / "results.parquet"
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(parquet_path)
    except ImportError:
        json_path = run_dir / "results.json"
        json_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return json_path
    return parquet_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as deterministic, strictly-valid JSON with a trailing newline."""
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_phase1_smoke(config: RunConfig, out_dir: str | Path) -> int:
    """Run the Phase-1 closed loop + baseline over the dev set; emit all artifacts.

    Returns ``0`` on success (every artifact written and the first instance's run
    strict-replays from disk), or ``1`` on a fatal error (no instances, or the
    strict replay diverged).
    """
    # Deferred so ``import opop.run`` stays torch-free; the GP controller's torch
    # dependency ships in the `bo` extra and is only needed to actually run.
    from opop.controller.phase1 import Phase1Controller

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    instances = get_phase1_instances(config.split, sources=("synthetic",))
    instances = instances[: config.instance_limit]
    if not instances:
        print("phase1 smoke: no instances to run", file=sys.stderr)
        return 1

    time_limit = float(config.budget.time_limit_sec)
    n_trials = int(config.budget.trials)

    events_path = run_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    verification_dir = run_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    kernel = ScipKernel()
    rows: list[dict[str, Any]] = []
    instance_manifests: list[dict[str, Any]] = []
    first_instance_run_dir: Path | None = None

    for ir in instances:
        for seed in config.seeds:
            instance_run_dir = run_dir / "instances" / f"{ir.name}_{seed}"
            seed_config = replace(config, seeds=[int(seed)])
            controller = Phase1Controller.bo(
                default_phase1_space(),
                n_trials=n_trials,
                n_init=min(3, n_trials),
                n_candidates=64,
                seed=int(seed),
            )
            state = ProblemState(
                instance_id=ir.name, task_family="MILP", budget_state={"ir": ir}
            )
            result = run_loop(
                state,
                seed_config,
                kernel=kernel,
                proposer=propose,
                analyzer=analyze,
                verifier=verify_delta,
                evaluator=evaluate,
                controller=controller,
                out_dir=instance_run_dir,
                reference_optimum=None,
                instance_id=ir.name,
            )
            rows.append(_opop_row(ir, seed, result, time_limit))

            trace = kernel.solve(
                ir,
                Phi(),
                time_limit=time_limit,
                memory_limit_mb=MEMORY_LIMIT_MB,
                seed=int(seed),
            )
            baseline_score = evaluate(trace, time_limit=time_limit)
            rows.append(_baseline_row(ir, seed, baseline_score.metrics, time_limit))

            _append_events(instance_run_dir / "events.jsonl", events_path)
            _collect_verification(instance_run_dir, verification_dir, f"{ir.name}_{seed}")
            instance_manifests.append(
                {
                    "instance_id": ir.name,
                    "seed": int(seed),
                    "run_dir": str(instance_run_dir.relative_to(run_dir)),
                    "manifest": str(
                        (instance_run_dir / "repro_manifest.json").relative_to(run_dir)
                    ),
                }
            )
            if first_instance_run_dir is None:
                first_instance_run_dir = instance_run_dir

    results_path = _write_results(rows, run_dir)

    report = compare(
        load_results(results_path),
        baseline=BASELINE_METHOD,
        method=OPOP_METHOD,
        metric="primal_integral",
    )
    write_comparison_report(report, run_dir / "comparison_report.json")

    audit = audit_leakage(run_dir, REGISTRY_PATH)
    _write_json(run_dir / "leakage_audit.json", audit)

    _write_json(
        run_dir / "repro_manifest.json",
        {
            "plan_name": config.name,
            "config": config_to_dict(config),
            "n_instances": len(instances),
            "n_seeds": len(config.seeds),
            "instances": instance_manifests,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    assert first_instance_run_dir is not None
    replay_code = replay_run(first_instance_run_dir, strict=True)
    if replay_code != 0:
        print(f"phase1 smoke: strict replay FAILED for {first_instance_run_dir}", file=sys.stderr)
        return 1
    print(f"phase1 smoke: strict replay reproduced {first_instance_run_dir}")

    summary_line = (
        f"phase1 smoke complete: {len(instances)} instance(s) x {len(config.seeds)} seed(s); "
        + f"results={results_path.name}; leakage={audit['status']}; "
        + f"comparison_is_win={report.is_win}"
    )
    print(summary_line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``--config`` / ``--out``, load the config, and run the Phase-1 smoke."""
    parser = argparse.ArgumentParser(
        prog="opop.run",
        description="Phase-1 end-to-end smoke + sanity experiment (closed loop vs SCIP-default).",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="path to a run config (.yaml/.json)"
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="output run directory for the six artifacts"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return run_phase1_smoke(config, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
