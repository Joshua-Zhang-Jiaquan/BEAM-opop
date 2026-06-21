# OPOP Architecture

OPOP is a **Bayesian-guided, solver-in-the-loop, symbolically-verified
formulation-and-search engine** for combinatorial optimization (CO) and
integer programming (IP). The LLM never "solves" a problem; it *proposes*
symbolically-verified formulation/search **deltas** inside a closed loop whose
feedback drives structured Bayesian optimization over a mixed design space.

This document describes the **five layers**, the **verification gate**, and the
**controller ladder** that make up the engine.

---

## The closed loop

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                     Orchestrator                          │
                    │  (budget, interrupts, events.jsonl, incumbent, manifest)  │
                    └──────────────────────────────────────────────────────────┘
                                          │  drives, in order:
        ProblemState (immutable) ─────────┼───────────────────────────────────────────┐
                                          ▼                                             │
   ┌───────────┐   ┌───────────┐   ┌────────────────┐   ┌──────────┐   ┌────────────┐  │
   │ Analyzer  │──▶│ Proposer  │──▶│ Verification    │──▶│  Solver  │──▶│ Evaluator  │──┤
   │ (Layer 2) │   │ (Layer 1) │   │ Gate (A–D, HARD)│   │ (Layer 3)│   │ (Layer 4)  │  │
   └───────────┘   └───────────┘   └────────────────┘   └──────────┘   └────────────┘  │
        ▲                                  │ reject → record + skip          │          │
        │                                  ▼ (never solved)                  ▼          │
        │                            verification/report.json          ScoreRecord      │
        │                                                                    │          │
        │                          ┌────────────────────────┐               │          │
        └──────────────────────────│ Controller (Layer 5)   │◀──────────────┘          │
                                    │ Bayesian opt / ask-tell│  scalarized reward       │
                                    └────────────────────────┘                          │
                                          │ next Phi / next delta                       │
                                          └─────────────────────────────────────────────┘
```

The loop repeats **Analyzer → Proposer → Verify → Solver → Evaluator →
Controller.update** until a budget (trials/wall-time) is exhausted or a
stagnation criterion fires. Every iteration appends one record to
`events.jsonl`; the running incumbent and its certificate are persisted to the
run directory.

The shared substrate is the **symbolic model IR** (`opop.model`) — frozen,
pure-data dataclasses (`MILP`, `Variable`, `LinearConstraint`, `Objective`) plus
the loop's state objects (`ProblemState`, `Phi`, `SolveTrace`, `ScoreRecord`,
`Delta`, `DeltaClass`). All state transitions use `dataclasses.replace`; no layer
mutates another layer's data in place.

---

## The five layers

### Layer 1 — Proposer (`opop.proposer`)
Given a `ProblemState` and an `AnalysisReport`, the proposer emits a small set of
**typed `Delta`s** restricted to a declared search space. The LLM (via
`opop.llm`) only *selects/ranks* from a finite, typed candidate pool — its output
is provably a **subset** of the legal pool, so it can never inject a free-form
edit or raw solver code. A deterministic rule-based proposer is the offline
fallback (used by tests via `FakeLLMClient`).

Phase-1 space: curated SCIP parameter deltas (class C), whitelisted
valid-inequality templates that the analyzer flagged (class B), and an optional
decomposition flag.

### Layer 2 — Analyzer (`opop.analyzer`)
Deterministic OR analysis of the formulation: dimension/units/index consistency,
LP-relaxation statistics (LP objective, integrality-gap estimate, fractional
pattern), redundancy / trivial-infeasibility / conflict detection, and
valid-inequality **candidate** generation from a whitelist (cover, clique). The
analyzer *proposes* candidate cuts; it never certifies them (that is the gate's
job). Output is a structured `AnalysisReport`.

### Layer 3 — Solver (`opop.solver`)
Open-source solver backends behind a single `SolverKernel` Protocol. Each kernel
compiles `IR` + `Phi.p` (parameters) into the backend's model, enforces a
deterministic envelope (`threads=1`, hard time/memory ceilings, seed), solves,
and returns a `SolveTrace` (primal/dual bound series, nodes, LP iters, cuts,
first-feasible time, status, `censored`). Backends: SCIP (`ScipKernel`, the
high-fidelity core), OR-Tools CP-SAT, HiGHS, CBC, and GCG. See
[howto-add-solver.md](howto-add-solver.md).

### Layer 4 — Evaluator (`opop.evaluator`)
Turns a `SolveTrace` (+ optional reference optimum) into a `ScoreRecord`:
feasibility, objective, final gap, time-to-first-feasible, **primal integral**
(Berthold step-function integral, anytime metric), primal-dual gap integral,
nodes, cuts, memory, and `censored`. **Right-censoring is first-class**: a
timeout is recorded as a censored lower bound, never as "solved@limit", and a
censored run is never treated as optimal. A `scalarize` hook produces the scalar
reward the controller maximizes.

### Layer 5 — Controller (`opop.controller`)
An **ask-tell** Bayesian optimizer over the encoded `Phi` subspace. It encodes
the mixed design space (categorical / ordinal / bool / continuous) into a
normalized numeric vector, fits a surrogate, and uses an acquisition function to
choose the next `Phi` from the proposer's candidate pool. The surrogate is
refit after every `tell` (the posterior update). See the
[controller ladder](#the-controller-ladder) and
[howto-add-surrogate.md](howto-add-surrogate.md).

---

## The verification gate (HARD gate)

The gate (`opop.verify`) sits **between the proposer and the solver**. Every
delta that touches variables / constraints / objective / bounds / Big-M /
indexing must pass a certificate **before** it is evaluated. It is **fail-closed**:
anything unknown or unprovable is rejected; a rejected delta is recorded but
*never solved*.

| Class | Meaning | Certificate |
|-------|---------|-------------|
| **A** | Equivalent reformulation | Structural alignment (rename mapping + `milp_diffs == []`) **and** solver confirmation that both formulations share status and optimum within tolerance. |
| **B** | Valid inequality / relaxation strengthening | Solver-backed separation: optimize the cut's LHS over the original feasible region; if no feasible integer point is removed, the cut is valid; otherwise the optimizer **is** the counterexample → reject. |
| **C** | Heuristic / search-parameter | Semantic no-op: `milps_equivalent(before, after)` must hold — any math change ⇒ reject. |
| **D** | Risky / non-certified | Short-circuited to `sandbox`; **never** returns `pass`; never enters main evaluation. |

Tolerances: feasibility `1e-7`, objective `1e-6`. The gate emits
`verification/report.json` with `status`, `delta_class`, the preserved-flags, and
a `counterexample` (or `null`). Missing/broken solver ⇒ the gate fail-closes
(reject) rather than crashing.

This gate is the project's **scientific-integrity keystone**: no silent change to
the feasible region is ever possible.

---

## The controller ladder

Bayesian-optimization learnability is **not assumed**. The controller climbs a
ladder of rungs behind the `Surrogate` / `Acquisition` Protocols, starting simple
and escalating only when evidence justifies it:

1. **Random search** — the always-available baseline (`RandomSearch`), same
   Protocol as every other rung.
2. **GP + EI / UCB** — the Phase-1 default: a Matérn-5/2 Gaussian process
   (`opop.controller.gp`) with Expected-Improvement / Upper-Confidence-Bound
   acquisitions. Dependency-light (numpy + torch).
3. **SMAC / TPE / RF** — censored-aware sequential model-based optimization
   (Wave-4 ladder), dropped in unchanged via the Protocols.
4. **Structured BO** — space-shape-aware surrogates selected by a router:
   BOCS (binary), COMBO (pure discrete), CoCaBO / HyBO (mixed),
   dictionary-embedding (high-dim discrete), and BoTorch
   `qLogNoisyExpectedImprovement` (default) / `qKnowledgeGradient` (batch).

**Fidelity gate.** Multi-fidelity BO (cost-aware MFKG over a `Phi.s` fidelity
dimension) is enabled **only** after a fidelity-correlation study passes
Spearman ρ ≥ 0.5 on the dev set; otherwise the negative result is recorded and
the engine stays single-fidelity. The default route is
`random → SMAC/TPE → qLogNEI`; structured rungs activate only when space shape +
budget justify it.

---

## Orchestration, reproducibility, and integrity

- **Orchestrator** (`opop.orchestrator`) drives the loop, honors budget /
  interrupts, persists `events.jsonl` and the incumbent, and stops on stagnation.
- **Reproducibility manifest** (`repro_manifest.json`) records git commit,
  python + solver versions, hardware, `threads=1`, **all** seeds (SCIP, CP-SAT,
  numpy/torch, LLM sampling), time/memory limits, and tolerances. `opop.replay`
  re-executes a run from its manifest and asserts artifacts regenerate within
  tolerance.
- **Benchmark registry** (`opop.bench`) enforces immutable dev/validation/
  test/ood_test splits via a sealed `split_manifest.lock`, with leakage groups so
  no instance family spans free and held-out splits.
- **Comparison report** computes per-method primal integral, solved-rate, and
  shifted-geometric-mean time, runs Wilcoxon signed-rank (α = 0.05, ≥ 5 seeds),
  and gates a "win" on a minimum effect size.

## Phase discipline

Phase-1 proves a **narrow vertical slice** end-to-end (SCIP only; params +
whitelisted cuts + decomposition flag; GP+EI vs random) before any expansion.
Multi-kernel scheduling, MIQP/MIQCP/QUBO/MINLP generality, the SMAC/structured
controller ladder, multi-fidelity BO, and historical transfer are deliberately
gated on Phase-1 evidence. See `.omo/plans/coip-agent-loop-framework.md` for the
full roadmap.
