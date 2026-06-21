# OPOP Technical Report — Reproducibility

## Reproducibility Manifest

Every OPOP experimental run writes a `repro_manifest.json` at the top-level output directory. The manifest is a JSON object with the following required fields:

- `plan_name`: the experiment plan identifier (e.g., `"phase1_smoke"`)
- `config`: the full run configuration (budget, solver, controller, etc.)
- `seeds`: a mapping of seed values used:
  - `python_random`: Python `random` module seed
  - `numpy`: NumPy random seed
  - `scip`: SCIP solver seed (via `randomization/randomseedshift`)
  - `torch`: PyTorch seed (when torch is available)
- `solver_version`: the solver version string from `Model().version()` or equivalent
- `instance_digest`: SHA-256 of the canonical MILP representation for each instance
- `time_limit_sec`: the wall-clock time limit enforced on each solve
- `memory_limit_mb`: the memory limit
- `tolerances`: numeric tolerances used (feasibility, optimality, equivalence checks)
- `threads`: set to `1` for deterministic single-thread execution
- `git_commit`: the Git commit hash at run time (when available)
- `container_digest`: Docker/Podman image digest (when running in a container)
- `created_at`: ISO-8601 timestamp

## Seed Policy

All randomness is seeded and recorded. The per-kernel seed ensures deterministic solve behavior when combined with single-thread execution. The Phase-1 `ScipKernel` sets `randomization/randomseedshift` before optimization and forces `lp/threads=1`.

## Replay Instructions

A run can be replayed from its output directory:

```bash
# Non-strict replay (re-execute the loop, report completion)
python -m opop.replay --run runs/smoke

# Strict replay (re-execute and compare incumbent objective + n_accepted)
python -m opop.replay --run runs/smoke --strict
```

Strict replay rebuilds every Phase-1 object from the manifest (controller, proposer, analyzer, verifier, evaluator, kernel) with the recorded seeds, re-executes the loop, and asserts that the replay's incumbent objective and acceptance count match the original within tolerance. A mismatch produces a human-readable diff.

### What replay covers

- Controller: same GP surrogate, acquisition function, candidate pool, seed
- Proposer: same pool construction, same LLM (or rule-based fallback)
- Analyzer: same base IR analysis (LP relaxation, redundancy, valid-inequality candidates)
- Verifier: same delta certification (structural + solver-backed)
- Kernel: same solver, same seed, same budget
- Evaluator: same metric computation from the SolveTrace

### What replay does not guarantee

- Wall-clock times (these are inherently non-deterministic)
- LLM responses (if using a live API; the random seed does not control external services)

## Open-Source Solver Versions

| Solver | Python Binding | Engine Version |
|--------|---------------|----------------|
| SCIP | PySCIPOpt 6.2.1 | SCIP 10.0.2 (+ SoPlex 8.0.2) |
| CP-SAT | OR-Tools 9.14.6206 | — |
| HiGHS | highspy 1.14.0 | HiGHS 1.14.0 |
| CBC | PuLP 3.2.1 | CBC 2.10.3 |

All solvers are open-source. No commercial solver (Gurobi, CPLEX) is required.

## Python Environment

```
Python 3.12.3
numpy==1.26.4
scipy==1.14.1
torch (2.8.0a0 local build)
pyscipopt==6.2.1
ortools==9.14.6206
highspy==1.14.0
pulp==3.2.1
botorch==0.17.2
openai==2.38.0
```

Full pinned requirements are in `requirements.txt`.

## Benchmark Splits

The benchmark registry (`benchmarks/registry.yaml`) defines immutable splits with checksum verification:

- `dev` (70%): used for development, hyperparameter search, and ablation studies
- `validation` (30%): used for model selection and early-stopping decisions
- `test`: held out; evaluated only in one-shot final mode
- `ood_test`: held out; out-of-distribution test instances

The split assignment is sealed by `split_manifest.lock` (SHA-256 hash). Any change to the assignment is detected and blocks evaluation.

## Artifact Directory Structure

After a full run:

```
runs/<name>/
  results.parquet           # consolidated per-cell results
  events.jsonl              # append-only event journal
  thesis_report.json        # T1–T4 verdicts
  comparison_report.json    # baseline-vs-method comparison
  repro_manifest.json       # top-level reproducibility summary
  leakage_audit.json        # split-integrity audit result
  verification/             # per-delta verification certificates
  instances/                # per-instance run directories
    <name>_<seed>/
      events.jsonl
      incumbent.json
      result.json
      repro_manifest.json
      verification/
```
