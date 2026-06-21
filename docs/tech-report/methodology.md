# OPOP Technical Report — Methodology

## Staged Search Spaces (S0–S4)

Credit assignment in OPOP is achieved through staged ablation. Each stage adds a class of delta to the proposer's pool, so the marginal contribution of each class is measurable:

| Stage | Delta Classes | Description |
|-------|---------------|-------------|
| S0 | Class C (params) | BO-driven solver parameter tuning only. 6 curated SCIP knobs. No formulation changes. |
| S1 | S0 + Class B (safe cuts) | S0 plus analyzer-generated valid-inequality candidates (cover cuts, clique cuts), each verified by the gate before solving. |
| S2 | S1 + heuristic params | S1 plus heuristic-selection param deltas (Class C). |
| S3 | S2 + formulation/decomposition | S2 plus Class A reformulation deltas (variable renaming, index restructuring) and decomposition flags (Benders, Dantzig-Wolfe). |
| S4 | S3 + multi-kernel/MF/transfer | S3 plus multi-kernel scheduling, multi-fidelity evaluations, and historical transfer priors. |

Each ablation row is the full OPOP loop at that stage: proposer + analyzer + verification gate + solver + evaluator + Bayesian controller.

## Ablation Matrix

The full experiment matrix sweeps: `{benchmark family} × {method} × {ablation row} × {seed} × {time limit}`.

**Methods** (6 baseline families + OPOP):
1. `scip-default` — SCIP with factory settings, no tuning.
2. `scip-tuned` — SCIP with SMAC/OAT-tuned parameters (static tuning, per-family).
3. `opop-params-only` — BO-driven parameter tuning (S0), no formulation deltas.
4. `cuts-only` — analyzer-generated valid inequalities only (S1 without params), no BO.
5. `params+cuts` — S0 + S1 combined (BO over both), no formulation deltas.
6. `modeling-agent` — LLM-as-modeler baseline: NL problem description → LLM generates solver code → solve. No analyzer, verification, or BO loop.

**Ablation rows** (within OPOP): scip-default, params-only, analyzer-cuts-only, params+cuts, full-opop. Each row is a complete closed-loop run under the corresponding stage restrictions.

## Win Definition (Locked)

A comparison between method M and baseline B is a **win** if and only if:

1. **Statistical significance**: paired Wilcoxon signed-rank test, two-sided, α = 0.05, on metric values paired by `(instance_id, seed)`. p < 0.05 is required.
2. **Minimum effect**: the relative improvement clears a per-metric threshold:
   - Primal integral: ≥ 10% reduction (fractional: `(b - m) / b`)
   - Shifted geometric mean time (shift s=10): ≥ 20% reduction
   - Solved rate: ≥ 5 percentage point absolute gain (`m - b`)
3. **Seed floor**: at least 5 seeds (reported as a flag; not part of `is_win` per the locked formula).

`is_win = significant AND clears_min_effect`. A result can be significant without clearing the threshold, or clear the threshold without significance. Neither alone constitutes a win.

## Thesis Evaluation Protocol

Each thesis (T1–T4) is evaluated over the consolidated `results.parquet` via the thesis evaluator (`opop.eval.theses`):

- **T1** and **T4**: logical AND of two or more `compare()` calls. Every required comparison must individually be a win. The binding effect is the minimum relative improvement across comparisons.
- **T2**: paired Wilcoxon on per-cell solve counts (from `events.jsonl` or the `n_solves` column). Median solve-count reduction must exceed 30%. All overhead solves (verification, analyzer LP) are counted.
- **T3**: per-problem-type comparison. OPOP must win on every problem type present in the results. A single non-win on any type fails T3.
- **One-shot guard**: the evaluator refuses to touch held-out splits (`test`, `ood_test`) unless `one_shot_final=True` is passed explicitly, preventing accidental leakage.

## Leakage Policy

- Immutable splits: `dev` (70%), `validation` (30%), `test`, `ood_test`. Assigned at benchmark curation time.
- No instance may appear in more than one split (global namespace).
- No `leakage_group` may span free splits (`dev`/`validation`) and held-out splits (`test`/`ood_test`).
- A `split_manifest.lock` (SHA-256 over the canonical instance→split assignment) seals the assignment.
- Every experimental run undergoes a leakage audit (`opop.bench.audit_leakage`) that cross-references the run's `events.jsonl` against the registry. An audit failure blocks publication.

## Reproducibility

Every run writes a `repro_manifest.json` containing: configuration, seed values (Python `random`, NumPy, torch, SCIP), solver version strings, instance content hashes, time/memory budget, tolerance values, and Git commit / container digest when available. Byte-identical replay is supported via `python -m opop.replay --strict`, which re-executes the same objects from the manifest and asserts the replay's incumbent objective and acceptance count match the original.

## Metrics

- **Primary**: primal integral (anytime quality, the thesis-deciding metric).
- **Secondary**: shifted geometric mean end-to-end wall-clock, solved rate (fraction of instances reaching optimality).
- **Cost**: solver-only time, end-to-end wall-clock (including analyzer, proposer, verification, controller), LLM token counts, and memory peak.
- **Right-censored runtimes**: treated as lower bounds for the shifted geometric mean (lifted to `time_limit` when available). PAR10 is reported as a labeled auxiliary and never replaces the actual censored runtime.
