# OPOP — Agent Context

## Purpose
OPOP is a Bayesian-guided, solver-in-the-loop, symbolically-verified formulation-and-search engine for combinatorial optimization (CO) and integer programming (IP). It combines LLM-driven problem understanding, symbolic structural analysis, multiple solver backends (open-source MILP/SAT/CP), Bayesian optimization for hyperparameter and reformulation search, and post-hoc correctness verification — all orchestrated by a structured controller loop. The goal is to produce verifiably optimal or near-optimal formulations without requiring expert manual modeling.

## Layout
```
src/opop/
  __init__.py          — package root + lazy public-API re-exports (PEP 562)
  cli.py               — argparse CLI entry point (run / replay / bench); console script `opop`
  config.py            — dataclass RunConfig + JSON/YAML loader (load_config)
  run.py               — Phase-1 end-to-end smoke entry point (python -m opop.run)
  replay.py            — strict replay of a recorded run (python -m opop.replay)
  llm/                 — LLM client wrappers, prompt templates, chain-of-thought tooling
  model/               — mathematical model IR + quadratic extension + problem-class adapters
  analyzer/            — structural analysis of problem instances and formulations
  proposer/            — LLM-guided / rule-based typed delta proposal
  solver/              — solver backend adapters (SCIP, OR-Tools, HiGHS, CBC, GCG) + QUBO/MIQP
  evaluator/           — feasibility/optimality evaluators and metric collection
  controller/          — Bayesian controller ladder (GP+EI/UCB, Surrogate/Acquisition protocols)
  orchestrator/        — closed loop (run_loop), events journal, reproducibility manifest
  verify/              — post-hoc verification gate (delta classes A–D, fail-closed)
  bench/               — benchmark registry, immutable splits, leakage audit
  experiments/         — comparison report + statistical tests + baseline runners
docs/                  — architecture + API + how-to guides (see Docs below)
examples/              — runnable scripts (phase1_smoke.py, expansion_miqp.py)
tests/                 — test suite (mirrors src/opop/)
pyproject.toml         — buildable package metadata + tooling config (setuptools; pytest/ruff/mypy)
requirements.txt       — pinned dependencies (pip install ".[dev]" / ".[bo]" for extras)
CONTRIBUTING.md        — dev setup, test/lint commands, open-solver-only policy
LICENSE                — MIT
```

## Public API
`import opop` exposes a small, stable, **lazily-loaded** (PEP 562) surface — all
re-exports of internal modules; no internal mutable state is exposed. Headline
names: the runnable modules `opop.run` / `opop.replay`; `opop.run_loop`,
`opop.load_config`, `opop.RunConfig`; `opop.BenchmarkRegistry`;
`opop.SolverKernel` / `opop.ScipKernel` / `opop.available_solvers`;
`opop.Phase1Controller` / `opop.Surrogate` / `opop.Acquisition` / `opop.EI` /
`opop.UCB` / `opop.RandomSearch`; `opop.analyze` / `opop.propose`;
`opop.verify_delta`; `opop.evaluate` / `opop.scalarize`; `opop.compare`; the
model IR (`opop.MILP`, `opop.Variable`, `opop.Phi`, ...); and the problem-class
adapters (`opop.ProblemClassAdapter`, `opop.find_adapter`, `opop.QUBO`,
`opop.max_cut_qubo`, ...). The console script `opop` maps to `opop.cli:main`.
Open solvers only — Gurobi is absent from every install/public path.

## Docs
- `docs/architecture.md` — the five layers, the verification gate, the controller ladder.
- `docs/api.md` — public API overview with import examples.
- `docs/howto-add-solver.md` — implement the `SolverKernel` Protocol.
- `docs/howto-add-problem-class.md` — add a `ProblemClassAdapter` + a registry entry.
- `docs/howto-add-surrogate.md` — implement the `Surrogate` / `Acquisition` Protocols.
- `docs/design/` — design decisions (e.g. `solver-stack.md`).

## Roadmap
See `.omo/plans/coip-agent-loop-framework.md` for the full architecture and implementation plan.
