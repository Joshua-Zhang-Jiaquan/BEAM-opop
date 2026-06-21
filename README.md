# BEAM-opop

> **OPOP** — Bayesian-guided, solver-in-the-loop, symbolically-verified formulation-and-search engine for combinatorial optimization and integer programming.

[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-782%20passed%2C%2011%20skipped-blue.svg)]()
[![Open Solvers](https://img.shields.io/badge/solvers-open--only-orange.svg)]()

**Run a closed loop that proposes, certifies, and evaluates formulation/search deltas for MILP/MIQP/QUBO/MINLP — entirely with open-source solvers.**

OPOP combines an LLM-driven proposer, a deterministic symbolic analyzer, five open-source solver backends, a fail-closed verification gate, and a Bayesian controller ladder. The loop proposes typed deltas (parameter changes, valid inequalities, equivalent reformulations), certifies each delta before evaluation, and uses structured Bayesian optimization to drive the next proposal. No Gurobi. No commercial solver. No black-box magic.

This repository is part of the **BEAM** research series.

[Quickstart](#quickstart) · [Goals](#goals) · [Methods](#methods) · [Usage](#usage) · [Development](#further-development) · [Limitations](#limitations) · [Repository Structure](#repository-structure)

---

## Goals

OPOP is built around a single research question: *Can an analytical–numerical hybrid agent loop automatically search the space of solver configurations and formulation reformulations, and demonstrably beat modeling-agent-only and solver-tuning baselines on anytime + cross-distribution metrics?*

To answer this, the framework pursues four falsifiable theses:

| Thesis | Claim | Falsification criterion |
|--------|-------|------------------------|
| **T1** | Anytime / cross-distribution superiority | OPOP does **not** beat scip-default and params-only on primal integral by ≥10% on held-out instances. |
| **T2** | Sample / compute efficiency | OPOP does **not** reach baseline-best quality with ≥30% fewer full-solve evaluations. |
| **T3** | Generality | OPOP does **not** run unchanged across MILP, MIQP, QUBO, and structured MINLP instances. |
| **T4** | Method novelty | OPOP does **not** outperform params-only BO and modeling-agent-only baselines, showing analyzer-certified deltas add value. |

Every thesis is evaluated through agent-runnable experiments with pre-registered thresholds. Negative results are reported, not suppressed.

---

## Methods

### Five-layer architecture

1. **Analyzer** (Layer 1) — reads the MILP intermediate representation (IR) and produces LP-relaxation metrics, consistency flags, redundancy checks, and candidate valid inequalities (cover cuts, clique cuts).

2. **Proposer** (Layer 2) — an LLM selects deltas from a candidate pool produced by the analyzer. The pool is typed: **Class A** (equivalent reformulations), **Class B** (valid inequalities), **Class C** (parameter changes). The LLM never generates deltas from scratch; it only selects indices from a bounded, verified pool. A deterministic rule-based fallback is used when the LLM is unavailable or returns an unparseable response.

3. **Verification Gate** (Layer 3) — every delta is certified before evaluation. Class A is confirmed by structural equivalence + solver-backed optimal-value comparison. Class B is certified by optimizing the cut's left-hand side over the feasible region. Class C is confirmed by model equivalence. Class D (uncertified/risky) is rejected immediately. The gate is **fail-closed**.

4. **Solver Backend** (Layer 4) — a runtime-checkable `SolverKernel` Protocol. Five backends are supported: **SCIP** (PySCIPOpt), **OR-Tools CP-SAT**, **HiGHS** (highspy), **CBC** (PuLP), and **GCG** (PySCIPOpt decomposition API). Phase-1 evaluation uses SCIP exclusively.

5. **Controller** (Layer 5) — a Bayesian optimization ladder. Phase-1 uses an in-house Gaussian Process (Matern-5/2) with Expected Improvement over a normalized finite candidate pool. Later phases plug in structured BO (SMAC, BoTorch) via the `Surrogate` / `Acquisition` Protocols.

### Staged search spaces (S0–S4)

| Stage | Description |
|-------|-------------|
| **S0** | SCIP defaults. No search. |
| **S1** | Parameter-only Bayesian optimization. No cuts, no decomposition, no LLM. |
| **S2** | Analyzer-proposed valid inequalities only. Class B deltas certified by the gate. |
| **S3** | Combined S1 parameter search + S2 cut selection. No LLM. |
| **S4** | Full OPOP: S3 plus LLM-guided delta selection from the full pool. |

### Reproducibility

Every run records:
- `repro_manifest.json` — seeds, git SHA, solver versions, hardware metadata.
- `events.jsonl` — journal of every solve, proposal, acceptance, and rejection.
- `results.parquet` — per-(instance, seed, method) metrics.
- `thesis_report.json` — T1–T4 verdicts.
- `comparison_report.json` — statistical comparison vs baseline.
- `leakage_audit.json` — split-leakage verification.

The benchmark registry is sealed by a SHA-256 lock to prevent data leakage.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/Joshua-Zhang-Jiaquan/BEAM-opop.git
cd BEAM-opop

# Install (use --system-site-packages if your environment has a non-PyPI torch build)
python -m venv .venv
source .venv/bin/activate
pip install .

# Run the Phase-1 smoke experiment
python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke

# Replay a recorded instance strictly to verify reproducibility
python -m opop.replay --run runs/smoke/instances/<instance_name> --strict
```

---

## Usage

### 1. Run Phase-1 smoke

```bash
python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke
```

Produces:
- `runs/smoke/results.parquet`
- `runs/smoke/events.jsonl`
- `runs/smoke/repro_manifest.json`
- `runs/smoke/comparison_report.json`
- `runs/smoke/leakage_audit.json`
- `runs/smoke/verification/*.json`

### 2. Replay a recorded run

```bash
python -m opop.replay --run runs/smoke/instances/set_cover_8x12_0 --strict
```

Expected output: `REPRODUCED`.

### 3. Audit leakage

```bash
python -m opop.bench.audit_leakage --run runs/smoke --registry benchmarks/registry.yaml
```

Expected output: `leakage audit pass: 0 held-out instances used for tuning`.

### 4. Generate the leaderboard

```bash
python -m opop.leaderboard build --results runs/smoke --out runs/smoke/leaderboard-site
```

Open `runs/smoke/leaderboard-site/index.html` in a browser.

### 5. Generate the paper / tech report

```bash
python scripts/make_report.py --results runs/smoke --out docs/tech-report
python scripts/make_paper.py --results runs/smoke --out docs/paper
```

### 6. Audit paper claims

```bash
python scripts/claims_audit.py docs/paper/paper.md
```

### 7. Run the full test suite

```bash
pytest tests/ -q
```

Expected: `782 passed, 11 skipped` (skips are for absent optional deps `smac`, `botorch`, `gcg`).

---

## Further Development

### Add a solver backend

Implement the `SolverKernel` Protocol in `src/opop/solver/` and register it in `src/opop/solver/availability.py`. See `docs/howto-add-solver.md`.

### Add a problem-class adapter

Add a `ProblemClassAdapter` in `src/opop/model/` and register it in the adapter registry. See `docs/howto-add-problem-class.md`.

### Add a surrogate / acquisition

Implement the `Surrogate` and `Acquisition` Protocols in `src/opop/controller/` and wire them into the controller ladder. See `docs/howto-add-surrogate.md`.

### Run the optional OR-LLM stretch

Task 45 (fine-tuned OR-LLM proposer backend) is deferred to future compute allocation. To pick it up later:

1. Synthesize OR-Instruct-style training data from solved instances + certified deltas (dev/validation only — never test/ood).
2. SFT/RL a small open model and serve it via vLLM on the 4× RTX 4090.
3. Plug the local backend into the existing `LLMClient` adapter behind a feature flag.
4. Evaluate it as an **additional** method row, not a replacement.

---

## Limitations

- **Phase-1 MILP-only scope** in the main evaluation. MIQP, QUBO, and MINLP backends exist but the full cross-problem-class experiment matrix has not been run.
- **Curated parameter set**: the parameter space covers 6 SCIP knobs, not the full hundreds-knob space.
- **Single LLM path**: ablations across LLM providers, prompt templates, and selection strategies are deferred.
- **No multi-fidelity BO** in Phase-1: the fidelity-correlation gate is implemented, but multi-fidelity extensions require additional compute.
- **Time-limit dependence**: results are reported at fixed budgets; the relationship between budget and relative improvement may not be monotonic.

This is a research framework, not a production solver. It wraps existing solvers; it does not replace them.

---

## Repository Structure

```
BEAM-opop/
├── src/opop/                  # Main package
│   ├── analyzer/              # Deterministic OR analyzer
│   ├── bench/                 # Benchmark registry, splits, leakage audit
│   ├── controller/            # Bayesian optimization ladder
│   ├── eval/                  # Comparison + thesis evaluation
│   ├── evaluator/             # Metrics, primal integral, censoring
│   ├── experiments/           # Experiment runner, matrix driver, baselines
│   ├── leaderboard/           # Static leaderboard builder
│   ├── llm/                   # LLM client adapter
│   ├── model/                 # MILP/MIQP/MINLP/QUBO IR
│   ├── orchestrator/          # Closed loop + reproducibility
│   ├── proposer/              # LLM-guided / rule-based proposer
│   ├── solver/                # SCIP, CP-SAT, HiGHS, CBC, GCG kernels
│   └── verify/                # Fail-closed verification gate
├── tests/                     # Pytest suite (mirrors src/opop/)
├── configs/                   # Run configurations
├── benchmarks/                # Registry + sealed split manifest
├── docs/                      # Architecture, API, how-to, paper, tech report
├── scripts/                   # make_paper.py, make_report.py, claims_audit.py
├── examples/                  # Runnable examples
├── pyproject.toml             # Build metadata + tool config
├── requirements.txt           # Pinned dependencies
└── AGENTS.md                  # Agent context for the codebase
```

---

## License

[MIT License](./LICENSE). Free to use, modify, and distribute, including commercial use.

---

## Connect

Part of the **BEAM** research series. Built with agentic orchestration by Sisyphus.

Repository: https://github.com/Joshua-Zhang-Jiaquan/BEAM-opop
