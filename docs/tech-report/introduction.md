# OPOP Technical Report — Introduction

## Problem Framing

Combinatorial optimization (CO) and integer programming (IP) underpin critical decisions in logistics, scheduling, network design, and resource allocation. The dominant approach today pairs a human expert modeler with a general-purpose solver (SCIP, Gurobi, CPLEX). The expert manually translates a business problem into a mathematical formulation, selects solver parameters, and iterates until the solver delivers acceptable results within a time budget.

This workflow has three well-known failure modes:

1. **Formulation brittleness.** Two mathematically equivalent formulations can differ by orders of magnitude in solve time. An expert chooses one, sometimes suboptimally, and rarely has the budget to explore alternatives.
2. **Parameter blindness.** Modern solvers expose hundreds of parameters. Tuning them by hand or even with automated tools (SMAC, Optuna) is expensive and does not transfer across instances.
3. **No feedback loop.** The solver is a black box. When performance is poor, the expert guesses whether the formulation, the parameters, or both are at fault, then manually adjusts. There is no closed-loop automated reasoning.

Large language models (LLMs) have recently been applied to CO/IP, most prominently as *modeling agents* that translate natural-language problem descriptions into solver code (OptiMUS, LLMOPT, ORLM). These systems treat the LLM as a one-shot translator: the LLM writes a model, the solver runs it, and the result is returned to the user. There is no iterative refinement, no symbolic verification, and no structured optimization of the formulation-search process.

## OPOP: LLM as Proposer, Not Solver

OPOP inverts this relationship. The LLM never writes solver code directly. It never chooses arbitrary solver parameters. Instead, the LLM proposes **symbolically verifiable deltas** inside a closed loop:

- **Class A** (equivalent reformulation): rename a variable, re-index a set, restate a constraint in an algebraically equivalent form. Must preserve the integer feasible region and objective exactly.
- **Class B** (valid inequality / strengthening): add a cut that removes fractional LP solutions but keeps every feasible integer point. Certified by an integer separation check.
- **Class C** (heuristic / search-parameter): adjust a solver knob within a curated, whitelisted set. No semantic model change; purely a search-path choice.
- **Class D** (risky / non-certified): sandbox only. Never enters the main evaluation.

Every Class A or B delta passes through a symbolic verification gate before the solver touches it. If a delta cannot be proven safe, it is rejected. This fail-closed design is the keystone of scientific integrity in OPOP: no formulation change ever enters the evaluation without a certificate.

The loop then feeds solver outcomes (primal integral, gap, runtime, cost) into a Bayesian controller that selects the next delta. Structured Bayesian optimization over a mixed categorical/ordinal/continuous search space drives the search, with the LLM selecting from a curated pool of typed, legal proposals. The LLM therefore acts as a *proposal ranker*, not a generator of raw model code.

## Four Falsifiable Theses

OPOP's research campaign is structured around four theses, each of which is empirically falsifiable using the locked Win Definition (Wilcoxon signed-rank, α=0.05, minimum primal-integral reduction ≥10%):

### T1 — Anytime / Cross-Distribution Superiority

OPOP beats both `scip-default` (SCIP with factory settings) and `opop-params-only` (BO-driven parameter tuning without formulation changes) on primal integral, at an equal end-to-end budget, on held-out instances drawn from distributions not seen during development.

**Falsifiable by**: a statistical non-win against either baseline on the held-out test or OOD-test split.

### T2 — Sample / Compute Efficiency

OPOP reaches baseline-best quality (the best primal integral achieved by any baseline) using at least 30% fewer full-solve evaluations than `scip-default`, counting all overhead (verification solves, analyzer LP relaxations, etc.).

**Falsifiable by**: a median solve-count reduction below 30%, or a non-significant difference after accounting for overhead.

### T3 — Generality

OPOP beats `scip-default` on primal integral on **every** problem type present in the benchmark suite (MILP, QUBO, MIQP, MIQCP, structured MINLP), not just the majority.

**Falsifiable by**: a single problem type where OPOP does not achieve a statistical win.

### T4 — Method Novelty

OPOP beats both `opop-params-only` and `modeling-agent` (an LLM-as-modeler baseline without the analyzer/verification/BO loop) on primal integral, demonstrating that the analyzer-certified deltas and closed-loop optimization add value beyond parameter-only BO and single-shot modeling.

**Falsifiable by**: a non-win against either baseline.

## Scope of This Report

This report documents OPOP's architecture, methodology, experimental results, and reproducibility plan. It is the honest, complete record, including ablations, negative results, and limitations. No numbers are hand-drawn; every table cell and figure data point derives from the experiment artifacts (`results.parquet`, `thesis_report.json`, `comparison_report.json`).
