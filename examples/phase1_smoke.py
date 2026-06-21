"""Runnable Phase-1 smoke example.

Drives the OPOP closed loop (Analyzer -> Proposer -> Verify gate -> Solver ->
Evaluator -> Controller) over the synthetic Phase-1 dev set and a SCIP-default
baseline, then emits the six experiment artifacts into an output directory:

    results.parquet (or results.json)   per-(instance, seed, method) result rows
    events.jsonl                        the closed-loop journal
    verification/*.json                 every accepted delta's certificate
    repro_manifest.json                 the reproducibility fingerprint
    comparison_report.json              opop vs scip-default on the primal integral
    leakage_audit.json                  held-out-instance leakage verdict

Everything runs offline (synthetic instances, open SCIP solver) and the recorded
run is strict-replayed from disk before success is reported.

Usage:
    python examples/phase1_smoke.py [--out runs/example_phase1_smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import opop

#: A minimal Phase-1 run config: synthetic dev set, 2 instances, 2 trials, 5s each.
SMOKE_CONFIG = """\
name: phase1_smoke_example
split: dev
seeds: [0]
instance_limit: 2
solver:
  name: scip
budget:
  trials: 2
  time_limit_sec: 5
"""

#: Artifacts emitted by ``run_phase1_smoke`` (results.parquet degrades to JSON
#: when pandas is unavailable).
ARTIFACTS = (
    "results.parquet",
    "results.json",
    "events.jsonl",
    "verification",
    "repro_manifest.json",
    "comparison_report.json",
    "leakage_audit.json",
)


def main(argv: list[str] | None = None) -> int:
    """Run the Phase-1 smoke and list the artifacts it produced."""
    parser = argparse.ArgumentParser(description="Run the OPOP Phase-1 smoke example.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/example_phase1_smoke"),
        help="output directory for the six run artifacts",
    )
    args = parser.parse_args(argv)
    out_dir: Path = args.out

    if not opop.is_solver_available("SCIP"):
        print("This example requires the SCIP backend (pip install pyscipopt).")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

    config = opop.load_config(config_path)
    print(f"[phase1_smoke] running closed loop vs scip-default -> {out_dir}")
    code = opop.run.run_phase1_smoke(config, out_dir)
    if code != 0:
        print("[phase1_smoke] FAILED")
        return code

    print("[phase1_smoke] artifacts produced:")
    for name in ARTIFACTS:
        if (out_dir / name).exists():
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
