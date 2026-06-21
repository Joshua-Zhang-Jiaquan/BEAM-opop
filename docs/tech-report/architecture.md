# OPOP Technical Report — Architecture

## Overview

OPOP is organized as five layers, coordinated by an orchestrator, with a verification gate on the critical path and a Bayesian controller driving the outer loop.

```
┌─────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                              │
│  (closed loop: budget, stagnation, interrupt, events journal)   │
└──┬───────┬──────────┬──────────┬──────────┬──────────┬─────────┘
   │       │          │          │          │          │
   ▼       ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐ ┌───────────┐
│PROPOSE│ │ANALYZE│ │VERIFY  │ │SOLVE │ │ EVALUATE │ │CONTROLLER │
│Layer 1│ │Layer 2│ │Gate    │ │Lay 4 │ │ Layer 5  │ │(BO+phi)   │
└──────┘ └──────┘ └────────┘ └──────┘ └──────────┘ └───────────┘
```

## Layer 1 — Proposer

The proposer builds a pool of typed, legal delta candidates and selects a subset for the current iteration. It has two selection paths:

- **LLM-guided**: the LLM receives a structured prompt with analysis features (LP objective, gap, fractional pattern, number of candidate cuts) and the typed pool entries. The LLM's output is a list of pool indices. The LLM therefore only ever re-ranks candidates, never injects a raw delta. On parse failure or hallucination, the system falls back to rule-based selection.
- **Rule-based**: a deterministic ranker that allocates slots between candidate cuts (from the analyzer) and curated solver parameters, prioritizing cuts when the LP gap exceeds 5%.

Proposals are drawn from staged search spaces:
- **S0**: solver parameters only (6 curated SCIP knobs)
- **S1**: S0 + safe valid-inequality cuts (Class B, whitelist only)
- **S2**: S1 + heuristic selection (Class C param deltas for heuristics)
- **S3**: S2 + formulation/decomposition deltas (Class A/B, e.g., rename vars, add Benders/DW decomposition flags)
- **S4**: S3 + multi-kernel scheduling + historical transfer priors

## Layer 2 — Analyzer

The analyzer performs deterministic, solver-backed structural analysis of the current MILP IR:

- **LP relaxation**: solves the continuous relaxation (SCIP, presolve off) and reports the LP objective, gap to an IP bound, and the fractional variable pattern.
- **Consistency checks**: validates index sets, dimension specifications, and unit annotations when metadata is present.
- **Redundancy detection**: normalized pivot-row comparison catches duplicate, dominated, and conflicting rows. Proportional rows (e.g., `2x+2y<=4` and `x+y<=2`) are recognized as identical.
- **Valid-inequality candidates**: generates cover cuts (from knapsack rows) and clique cuts (from set-packing rows) as candidate Class B deltas. All candidates are whitelisted by the analyzer; the verification gate later certifies or rejects each one.

## Layer 3 — Verification Gate (HARD gate)

Every Class A or B delta must pass through the verification gate before the solver touches it. The gate is fail-closed: if a certificate cannot be produced, the delta is rejected.

- **Class A (equivalent reformulation)**: structural alignment (check that the model is identical up to variable renaming) plus a solver confirmation (solve both the before and after models, require identical normalized optimal status and objective within tolerance).
- **Class B (valid inequality)**: for each added constraint, optimize over the *original* feasible region to check that the constraint does not cut any integer feasible point. Uses a SCIP integer separation solve.
- **Class C (heuristic/param)**: verified purely structurally (`milps_equivalent` must hold; the math model is unchanged).
- **Class D (risky)**: immediately sandboxed; never enters evaluation.

## Layer 4 — Solver Kernels

OPOP wraps open-source solvers behind a common `SolverKernel` Protocol:

- **SCIP** (PySCIPOpt 6.2.1 / SCIP 10.0.2): the high-fidelity core. Full event-handler trajectory extraction (primal/dual series at each incumbent and bound improvement), time/memory limits, determinism via single-thread + fixed seed.
- **CP-SAT** (OR-Tools 9.14): integer-only models with solution-callback trajectory. Coefficient rationalization for exact integer scaling.
- **HiGHS** (highspy 1.14): LP/MILP via the high-level API.
- **CBC** (PuLP 3.2.1 / CBC 2.10.3): bundled MILP solver.

Each kernel returns a `SolveTrace` with primal/dual bound series, scalar totals (nodes, LP iterations, cuts), a final status, and a censoring flag.

## Layer 5 — Evaluator

The evaluator converts a `SolveTrace` into a multi-metric `ScoreRecord` and scalarizes for the Bayesian controller:

- **Primal integral** (Berthold step-function integral, the primary metric): `Σ gap(t_i) · Δt_i` over the trajectory, using a left-held piecewise-constant integral.
- **Gap**: normalized `|primal - reference| / max(|primal|, 1e-12)`.
- **Right-censoring**: runs terminated by a resource limit are flagged `censored=True`. Their runtime is recorded as a lower bound (never overwritten by a penalty). PAR10 is provided as a labeled auxiliary.
- **Scalarization**: `reward = -gap - 1e-3 * time - primal_integral` (higher is better), used by the Bayesian controller.

## Controller — Bayesian Optimization

The controller maintains a Gaussian Process surrogate (Matern-5/2 kernel, Cholesky inference) over the mixed-categorical search space encoded as a continuous `[0,1]^d` cube via one-hot/ordinal/continuous encodings. Expected Improvement (EI) drives candidate selection from a finite candidate pool. The controller implements an ask-tell interface compatible with the Phase-1 `Acquisition` Protocol, enabling later swap-in of SMAC, TPE, BoTorch, or multi-fidelity surrogates.

A **controller ladder** gates the complexity: random search → SMAC/TPE/RF → BoTorch qLogNEI → structured Bayesian optimization only after fidelity-correlation evidence (Spearman ρ ≥ 0.5).

## Orchestrator

The orchestrator runs the closed loop:

1. Analyze the base IR once (the IR does not accumulate deltas across iterations; each delta is transient for one solve).
2. Per iteration: controller asks for a phi → proposer selects deltas → verify each delta → solve with verified phi → evaluate → controller tell (reward).
3. Stagnation detection, budget enforcement, interrupt handling.
4. Artifact emission: `events.jsonl` (append-only journal), `incumbent.json`, `result.json`, verification certificates.

Every run also produces a reproducibility manifest (`repro_manifest.json`) with pinned seeds, solver versions, configuration, instance hashes, and tolerance values, enabling byte-identical replay via `python -m opop.replay --strict`.

## Cost Accounting

A `CostAccountant` tracks per-phase wall-clock time: analyzer (once), proposer (per iteration), verification (per delta), solver (per delta), evaluator (per delta), and controller tell (per iteration). LLM token counts and cost estimates are recorded when an LLM backend is active. Both solver-only and end-to-end costs are reported.
