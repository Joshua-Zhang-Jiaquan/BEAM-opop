# OPOP: Bayesian-Guided, Solver-in-the-Loop, Symbolically-Verified Formulation-and-Search Engine for CO/IP

## TL;DR

> **Quick Summary**: Build `opop` — an analytical–numerical hybrid agent loop for combinatorial optimization / integer programming (Proposer → Analyzer → Solver → Evaluator → Bayesian Controller, coordinated by an Orchestrator) — and run its full research/evaluation campaign. The LLM never "solves"; it proposes *symbolically-verified* formulation/search deltas inside a closed loop whose feedback drives structured Bayesian optimization over a mixed design space. Open-solver-only (SCIP core). Phase-1 proves a NARROW vertical slice end-to-end before any expansion.
>
> **Deliverables**:
> - Open-source Python library (`src/opop/`) implementing all 5 layers.
> - Reproducible experiment suite (benchmark registry, immutable splits, leakage audit, cost accounting, reproducibility manifests).
> - Full ablation matrix (S0–S4 staged search spaces; 6 baseline families).
> - Conference paper + internal tech report + public leaderboard.
>
> **Estimated Effort**: XL (multi-month; 45 implementation tasks + 4 final-verification tasks).
> **Parallel Execution**: YES — 7 waves + 1 final verification wave.
> **Critical Path**: 1 → 3 → 12 → 13 → 16 → 21 (Phase-1 loop PROVEN) → 27 → 29 → 39 → 40 → 44 → F1–F4.
>
> **Phasing discipline (Metis-mandated)**: Waves 1–3 (foundations + Phase-1 vertical slice) are FULLY specified and are the de-risking critical path. Waves 4–7 (framework expansion, generality, research campaign, deliverables) have concrete deliverables + **explicit entry criteria** and are deliberately gated on Phase-1 evidence (e.g., multi-fidelity BO only after a fidelity-correlation study passes Spearman ρ ≥ 0.5). This is correct for a research build: do not fully pre-commit Phase-4 experiments before the loop is proven.

---

## Context

### Original Request
The user provided a research report proposing an "analytical–numerical hybrid Agent Loop Engineering framework" for CO/IP, and asked for ONE unified work plan that BOTH (a) builds the framework as real software AND (b) runs the full research/evaluation campaign, at FULL scope (all 5 layers; MILP/0-1 IP → MIQP/MIQCP/QUBO → structured MINLP; 6 baseline families; full ablation matrix; paper + tech report + OSS library + leaderboard).

### Interview Summary
**Confirmed decisions**:
- **LLM backend**: model-agnostic adapter — OpenAI-compatible API + local vLLM, swappable.
- **Solvers**: OPEN ONLY (SCIP/PySCIPOpt high-fidelity core; OR-Tools CP-SAT; HiGHS; CBC; GCG). No Gurobi (no license present).
- **Compute**: local dev + cluster sweeps (4× RTX 4090 48 GB, 128 CPU, 1 TiB RAM). Job-runner pluggable: local | SLURM | qz.
- **Testing**: pytest + TDD (RED-GREEN-REFACTOR) + agent-executed QA + solver-backed verification (HARD gate).
- **Success theses (all four, falsifiable)**: T1 anytime + cross-distribution superiority; T2 sample/compute efficiency; T3 generality MILP→MIQP/QUBO→MINLP; T4 method novelty.
- **Deliverables**: conference paper + tech report + OSS library + leaderboard.
- **Repo**: greenfield, reuse mature libs as deps; reuse lab assets (see below).
- **Fine-tuning own OR-LLM**: OPTIONAL late-wave stretch only.
- **Timeline**: no hard deadline; front-load a thin vertical slice.

**Environment (probed)**: Python 3.12.3; nothing of the solver/BO stack installed yet (only pulp, torch, openai, transformers, numpy/scipy/networkx). Wave 1 MUST bootstrap the stack.

### Research Findings (lab conventions + reusable assets)
> BEAM siblings (AgenticLLMPipeline/LongHorizonMemBench/SelfEvolvingBench) are EMPTY shells. Real conventions live in `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/{DAC,intern,portable_brain_for_agents}`.
- **LLM adapter to PORT**: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/topic2-format-efficiency/tools/llm_client.py` (OpenAI-compatible `chat()`/`chat_json()`/`TokenTracker`, env-driven).
- **BO base to PORT**: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_e/src/acquisition.py` (GP Matern-5/2 + UCB/EI/random + `run_bo_trials` + `scalarized_reward`; numpy+torch only). Meta: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_e/src/meta_tuner.py`.
- **Agent-loop base**: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/autoresearch-harness/autoresearch_harness/self_evolution.py` (bounded repair loop + stagnation/fingerprint detection); protocol `.../agents/base.py`; orchestrator `.../orchestrator.py`; config `.../config.py`.
- **Closest multi-agent analog**: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/topic2-format-efficiency/tools/orchestrator.py` (Planner-Executor-Reviewer, ~820 lines).
- **Conventions**: `src/<pkg>/` layout; `pyproject.toml` for pytest config only; pinned `requirements.txt` (labeled sections); argparse + dataclass + JSON/YAML + env (NO Hydra); `ruff`; AGENTS.md docs; SLURM scripts in `slurm/` (Docker base `pytorch/pytorch:2.4.0-cuda12.1`); seed-in-path naming. pytest reference: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/pyproject.toml`; conftest: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/SMIAutoResearch/tests/conftest.py`.

### Metis Review (gaps addressed → see Work Objectives + Verification Strategy)
Metis flagged: "win" must be numerically defined; verification-gate semantics need explicit delta classes; Phase-1 must be a genuinely narrow slice; benchmark/baseline matrix is explosive (needs registry); leakage policy + audit; cost accounting (solver-only AND end-to-end); credit assignment via staged search spaces + ablations; BO learnability not assumed (controller ladder + fidelity-correlation gate); reproducibility manifests; four theses made falsifiable. All incorporated below.

### CRITICAL TECHNICAL RISKS (from API research — drive Wave-1 spikes + guardrails)
1. **MIPLearn v0.4.x requires `gurobipy≥12`** → INCOMPATIBLE with open-solvers-only. Use MIPLearn as a **design reference only**; re-implement needed learning-augmented features on SCIP/PySCIPOpt.
2. **Ecole 0.8.1 (SCIP 8) vs PySCIPOpt 6.x (SCIP 10)** → version conflict. DEFAULT: extract bipartite/MILP features directly via `model.as_pyscipopt()`; do not hard-depend on Ecole.
3. **SLURM vs qz** → job-runner abstraction (local | slurm | qz).

---

## Work Objectives

### Core Objective
Build and empirically validate `opop`: a closed-loop system that proposes symbolically-verified MILP (then MIQP/QUBO/MINLP) formulation+search deltas, executes them on open solvers, evaluates anytime/quality/cost feedback (with right-censored runtimes), and uses structured Bayesian optimization to drive the next proposal — demonstrably beating modeling-agent-only and solver-tuning baselines on anytime + cross-distribution metrics.

### Concrete Deliverables
- `src/opop/` package: `llm/`, `model/` (symbolic IR), `analyzer/`, `solver/` (multi-kernel), `evaluator/`, `controller/` (BO), `orchestrator/`, `verify/`, `bench/`, `experiments/`, `cli.py`.
- `benchmarks/registry.yaml` + immutable split manifests + leakage-audit tool.
- Reproducible experiment artifacts (results.parquet, events.jsonl, verification/*.json, repro_manifest.json, comparison_report.json) per run.
- Ablation matrix outputs (S0–S4 + scip-default/params-only/cuts-only/params+cuts/full-opop).
- `docs/` (design + API), public leaderboard, tech report, conference paper draft.

### Definition of Done
- [x] `pytest tests/` green (unit + integration), coverage threshold met for core modules.
- [x] Phase-1 end-to-end smoke produces all artifacts and a statistically-tested comparison report vs SCIP-default.
- [x] Final Verification Wave F1–F4 all APPROVE, then explicit user okay.

### Must Have
- Closed loop runs unattended within a budget and produces an auditable comparison report.
- Verification HARD gate: every delta touching variables/constraints/objective/bounds/Big-M/indexing passes a model-equivalence or valid-strengthening certificate BEFORE evaluation; fail-closed.
- Right-censored runtime handling everywhere timeouts occur.
- Cost accounting reports solver-only AND end-to-end (incl. LLM tokens/$, analyzer/proposer/controller/verification time).
- Immutable dev/validation/test/ood_test splits + passing leakage audit before any test-set numbers.
- Reproducibility manifest (threads=1, pinned versions, all seeds, tolerances) on every experimental run.
- Staged search spaces S0–S4 + mandatory ablation rows for credit assignment.
- Controller ladder: random → SMAC/TPE/RF → BoTorch qLogNEI(restricted) → structured BO ONLY after fidelity-correlation evidence (Spearman ρ ≥ 0.5).

### Must NOT Have (Guardrails)
- ❌ No Gurobi / commercial solver dependency (and therefore no direct MIPLearn `LearningSolver`).
- ❌ No silent change of the feasible region (any such delta must be certified or sandboxed).
- ❌ No paper/leaderboard claims from dev/validation results, or from solver-only time when claiming compute efficiency.
- ❌ No full BO surrogate menu at the start; no benchmark family/baseline added without a phase+thesis assignment.
- ❌ Phase-1 EXCLUDES: leaderboard, paper, tech report, multi-kernel scheduling, MINLP/QUBO generality, historical transfer, OR-LLM fine-tuning, multi-fidelity BO.
- ❌ No novel solver core (we WRAP solvers); no production serving; no non-Python rewrites.
- ❌ No requiring SOTA on all domains in Phase-1; no "improved performance" claim without predefined metric+baseline+instances+time-limit+statistical test+min-effect.

### Spec Framework Integration
- **Detected Framework**: None (no `openspec/` or `.specify/` present). No SDD commands apply.

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — all verification is agent-executed. Evidence saved under `.omo/evidence/task-{N}-{slug}.{ext}`.

### Test Decision
- **Infrastructure exists**: NO (greenfield) → Wave 1 sets it up.
- **Automated tests**: YES (TDD). Each task: RED (failing test) → GREEN (minimal impl) → REFACTOR.
- **Framework**: `pytest` (markers: smoke/gpu/slow/integration), `ruff` lint, `mypy` where practical.

### Win Definition (LOCKED DEFAULTS — overridable, disclosed)
- **Primary metrics**: primal integral (anytime), solved-rate, shifted-geometric-mean end-to-end wall-clock.
- **Statistical test**: Wilcoxon signed-rank, α = 0.05; ≥ 5 seeds; report distributions (not point means).
- **Min effect for a "win"**: ≥ 10% primal-integral reduction OR ≥ 20% shifted-geomean time OR ≥ 5 pp solved-rate.

### Verification Delta Classes (HARD gate)
- **A — Equivalent reformulation**: preserves feasible integer solutions + objective within tol.
- **B — Valid inequality / relaxation strengthening**: may cut fractional LP points, must NOT remove any feasible integer incumbent.
- **C — Heuristic / search-param**: search path only; no semantic change.
- **D — Risky / non-certified**: sandbox experiments only; NEVER enters main evaluation.

### QA Policy
Every task includes agent-executed QA scenarios (happy + failure path) with concrete tools:
- **Library/Module** → Bash (`python -c` / pytest): import, call, compare exact outputs.
- **CLI/loop** → interactive_bash/tmux session (or plain Bash if no interactive shell is available): run command, validate stdout/exit code/artifacts.
- **Solver/numeric** → Bash: solve a fixture model, assert objective/gap/status against a known value.
- **Reports/figures** → Bash + (if HTML leaderboard) Playwright: assert JSON fields / DOM content.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 — Foundations & Scaffolding (START IMMEDIATELY):
├── 1  Repo scaffolding + pyproject + requirements + AGENTS.md        [quick]
├── 2  LLM adapter (port llm_client.py; OpenAI-compat + vLLM)         [unspecified-low]
├── 3  Solver bootstrap + smoke + version-conflict SPIKE (SCIP/Ecole/MIPLearn) [unspecified-high]
├── 4  Core types/state (ProblemState, phi, SolveTrace, ScoreRecord)  [quick]
├── 5  Config system (dataclass + JSON/YAML + env)                    [quick]
├── 6  Test infra (conftest, markers, ruff, fixtures)                 [quick]
├── 7  Benchmark registry schema + immutable splits + leakage groups  [quick]
└── 8  BO base (port acquisition.py: GP+UCB/EI/random+scalarize)      [unspecified-low]

Wave 2 — Phase-1 Vertical Slice CORE (after Wave 1):
├── 9  Symbolic model IR + MPS/LP I/O round-trip (depends 4)          [unspecified-high]
├── 10 Analyzer subset (dim/index, LP-relax stats, redundancy, valid-ineq cand) (depends 3,9) [deep]
├── 11 Verification gate: delta classes A–D + certificates (depends 9) [deep]
├── 12 SCIP kernel + trajectory extraction (depends 3,9)              [unspecified-high]
├── 13 Evaluator: multi-metric + right-censoring + primal integral (depends 12) [deep]
├── 14 Proposer (Phase-1 restricted: params + whitelist cuts + decomp flag) (depends 2,10) [unspecified-high]
└── 15 Controller (Phase-1: random + ONE qNEI baseline) (depends 8)   [unspecified-high]

Wave 3 — Phase-1 Integration + Repro + Reporting (after Wave 2):
├── 16 Orchestrator closed loop + budget/interrupt + events.jsonl (depends 11,13,14,15) [deep]
├── 17 Reproducibility manifest + replay (--strict) (depends 16)      [unspecified-high]
├── 18 Comparison report + stats (Wilcoxon/geomean/primal-integral) (depends 13) [unspecified-high]
├── 19 Leakage audit + cost accounting (depends 7,16)                 [unspecified-high]
├── 20 Phase-1 MILP dev set acquisition (MIPLIB subset + synthetic) (depends 7) [unspecified-high]
└── 21 Phase-1 END-TO-END smoke + sanity experiment vs SCIP-default (depends 16,17,18,19,20) [unspecified-high]  <<< LOOP PROVEN

Wave 4 — Framework Expansion / Phase 2 (entry: Wave 3 complete):
├── 22 CP-SAT adapter + trajectory (depends 12)                       [unspecified-high]
├── 23 HiGHS + CBC adapters + trajectory (depends 12)                 [unspecified-high]
├── 24 GCG adapter + DW/Benders auto-detection (depends 12)           [deep]
├── 25 Heuristic cores: LNS/RINS/local-branching/repair (depends 12)  [deep]
├── 26 Analyzer expansion: Benders/DW decomposability, Lagrangian, symmetry (depends 10) [deep]
├── 27 Proposer expansion: formulation families + staged spaces S0–S4 (depends 14) [deep]
├── 28 Controller ladder: SMAC/TPE/RF + structured BO selection (depends 15) [deep]
└── 29 Multi-fidelity + fidelity-correlation GATE + cost-aware MFKG (depends 13,28) [deep]

Wave 5 — Generality / Phase 3 (entry: Wave 4 core complete):
├── 30 MIQP/MIQCP/QUBO adapters behind declared plugin interface (depends 9,22) [deep]
├── 31 Structured MINLP adapter (decomposable subset) (depends 24,30) [deep]
└── 32 Historical-transfer priors / warm start across distributions (depends 28) [deep]

Wave 6 — Research Campaign / Phase 4 (entry: Waves 4–5 + thesis instruments ready):
├── 33 MILP benchmark acquisition (MIPLIB2017/Distributional/MILPBench) (depends 7,20) [unspecified-high]
├── 34 Classic CO benchmarks (TSPLIB/CVRPLIB/OR-Library/JSPLIB/MaxSAT/MaxCut) (depends 7) [unspecified-high]
├── 35 QPLIB + modeling-agent sets + solver-backed re-verification/cleaning (depends 7,30) [deep]
├── 36 Baselines 1–2: static+default; static+solver-tuning (SMAC/OAT) (depends 21,28) [unspecified-high]
├── 37 Baselines 3–4: params-only; LLM-modeling-agent-only (OptiMUS/LLMOPT/ORLM/OR-R1) (depends 21) [deep]
├── 38 Baselines 5–6: matheuristics (LB/RINS/LNS); LLM-enhanced CO (LLM-LNS/HeurAgenix) (depends 25) [deep]
├── 39 Full experiment matrix + ablations (job-runner; ≥5 seeds; multi-time-limit) (depends 33-38,29,31,32) [unspecified-high]
└── 40 Cross-dist + LOFO + scale-extrapolation eval + T1–T4 statistical analysis (depends 39) [deep]

Wave 7 — Deliverables / Phase 5 (entry: Wave 6 results frozen):
├── 41 OSS library packaging + public API + docs + examples (depends 21,40) [unspecified-high]
├── 42 Leaderboard (results table/site + submission protocol) (depends 40) [visual-engineering]
├── 43 Tech report (full methodology + results) (depends 40)          [writing]
├── 44 Conference paper (figures/tables/ablations/repro appendix) (depends 40,41) [writing]
└── 45 [OPTIONAL STRETCH] Fine-tuning thread (ORLM/OR-R1; guarded) (depends 37) [deep]

Wave FINAL — Verification (after ALL tasks): F1 oracle · F2 unspecified-high · F3 unspecified-high · F4 deep → present → user okay
```

### Dependency / Entry-Criteria Notes
- **Wave 4 entry**: Wave-3 task 21 (loop proven) PASSED. Otherwise fix the loop first.
- **Task 29 (multi-fidelity) gate**: only proceed past fidelity-correlation study if Spearman ρ ≥ 0.5 (else defer multi-fidelity BO and record the negative result).
- **Wave 6 entry**: thesis instruments (comparison report, ablation runner, leakage audit, cost accounting) exist and are tested.
- **Task 45**: optional; runs only if time/compute permit; never blocks 44.

### Agent Dispatch Summary
- Wave 1: 1/4/5/6/7 → `quick`; 2/8 → `unspecified-low`; 3 → `unspecified-high`.
- Wave 2: 9/12/14/15 → `unspecified-high`; 10/11/13 → `deep`.
- Wave 3: 17/18/19/20/21 → `unspecified-high`; 16 → `deep`.
- Wave 4: 22/23 → `unspecified-high`; 24/25/26/27/28/29 → `deep`.
- Wave 5: 30/31/32 → `deep`.
- Wave 6: 33/34/36/39 → `unspecified-high`; 35/37/38/40 → `deep`.
- Wave 7: 41 → `unspecified-high`; 42 → `visual-engineering`; 43/44 → `writing`; 45 → `deep`.
- Final: F1 → `oracle`; F2/F3 → `unspecified-high`; F4 → `deep`.

---

## TODOs

> Implementation + Test = ONE task. EVERY task has: Recommended Agent Profile + Parallelization + References + Acceptance Criteria + QA Scenarios. Labels are bare numbers (`1.`), final wave `F1.`.
> Waves 1–3 are fully specified (de-risking critical path). Waves 4–7 carry concrete deliverables + entry criteria and will be elaborated as Phase-1 evidence lands (Metis-mandated).

- [x] 1. Repo scaffolding + tooling config

  **What to do**:
  - Create package tree `src/opop/{__init__.py,llm,model,analyzer,solver,evaluator,controller,orchestrator,verify,bench,experiments}/__init__.py` + `src/opop/cli.py` (argparse stub with `run`/`replay`/`bench` subcommands).
  - `pyproject.toml`: `[tool.pytest.ini_options]` (markers: smoke, integration, gpu, slow; `testpaths=["tests"]`; `norecursedirs=[".omo",".venv"]`), `[tool.ruff]`, optional `[tool.mypy]`. NOT a build target yet.
  - `requirements.txt`: pinned, labeled sections (`# Core`, `# Solvers`, `# BO`, `# LLM`, `# Testing`, `# Quality`) — leave solver/BO versions as confirmed by task 3.
  - `AGENTS.md` (project purpose + layout), `.gitignore` (venv, __pycache__, runs/, benchmarks data, .omo/evidence), `README.md` stub.

  **Must NOT do**: add PyPI build metadata (deferred to task 41); add `gurobipy`; add Hydra/OmegaConf.

  **Recommended Agent Profile**:
  - **Category**: `quick` — mechanical scaffolding, single concern.
  - **Skills**: none (no UI, no browser, no git archaeology).

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: ALL · Blocked By: None.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/pyproject.toml` — canonical pytest markers + `norecursedirs`.
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/topic2-format-efficiency/requirements.txt` — pinned + labeled-section format.
  - WHY: match lab conventions exactly so later reuse (llm_client, acquisition) drops in cleanly.

  **Acceptance Criteria**:
  - [ ] `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"` → no error.
  - [ ] `python -c "import opop"` (with `src` on path / editable) → exit 0.
  - [ ] `ruff check src` → exit 0.

  **QA Scenarios**:
  ```
  Scenario: Package imports and tooling is valid
    Tool: Bash
    Steps:
      1. pip install -e . (or set PYTHONPATH=src)
      2. python -c "import opop, opop.cli"
      3. ruff check src && pytest -q (collects 0 tests, exits 0)
    Expected: all commands exit 0; "import opop" prints nothing
    Evidence: .omo/evidence/task-1-scaffold.txt

  Scenario: pyproject is malformed -> caught
    Tool: Bash
    Steps:
      1. Temporarily corrupt pyproject.toml, run the tomllib parse check
    Expected: non-zero exit with a parse error (proves the check is real); restore file
    Evidence: .omo/evidence/task-1-badtoml.txt
  ```
  **Commit**: YES — `chore(scaffold): initialize opop package, tooling, conventions`

- [x] 2. LLM adapter (port `llm_client.py`)

  **What to do**:
  - Port `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/topic2-format-efficiency/tools/llm_client.py` into `src/opop/llm/client.py`: `chat()`, `chat_json()`, `TokenTracker`, env-driven config with `OPOP_` prefix fallback chain (`OPOP_API_KEY`→`OPENAI_API_KEY`, `OPOP_BASE_URL`, `OPOP_MODEL`).
  - Add a thin `LLMClient` Protocol so backends are swappable; provide `OpenAICompatClient` (API) and a `VLLMClient` (local OpenAI-compatible endpoint) — both share the request path.
  - Add a `FakeLLMClient` returning deterministic canned responses for tests.

  **Must NOT do**: hardcode any provider/model; bake in secrets; add provider SDKs beyond `openai`/`requests`.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low` — porting + light abstraction.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: 14 · Blocked By: 1.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/topic2-format-efficiency/tools/llm_client.py` — `chat()/chat_json()/TokenTracker`, env fallback chain. WHY: proven, zero-framework; reuse verbatim then wrap in Protocol.
  - External: `openai` v2.38 (installed) for the API path; vLLM serves an OpenAI-compatible `/v1/chat/completions`.

  **Acceptance Criteria**:
  - [ ] `pytest tests/llm/test_client.py` → PASS (FakeLLMClient round-trip, TokenTracker accounting, `chat_json` parses).
  - [ ] No network call in tests (FakeLLMClient only).

  **QA Scenarios**:
  ```
  Scenario: Fake client returns structured JSON + tracks tokens
    Tool: Bash (pytest)
    Steps:
      1. Instantiate FakeLLMClient with a canned JSON reply
      2. chat_json("propose") -> dict; assert keys
      3. assert TokenTracker.total_tokens > 0 and cost_usd >= 0
    Expected: dict parsed; token/cost fields populated
    Evidence: .omo/evidence/task-2-llm-fake.txt

  Scenario: Malformed JSON reply -> graceful error
    Tool: Bash (pytest)
    Steps:
      1. FakeLLMClient returns non-JSON; call chat_json(...)
    Expected: raises a typed LLMParseError (not a bare ValueError), caught by caller
    Evidence: .omo/evidence/task-2-llm-badjson.txt
  ```
  **Commit**: YES — `feat(llm): swappable OpenAI-compatible + vLLM client with token tracking`

- [x] 3. Solver bootstrap + SMOKE + version-conflict SPIKE

  **What to do**:
  - Install + smoke-test open solvers: `pyscipopt` (SCIP 10.x), `ortools` (CP-SAT), `highspy`, CBC (via `pulp`/`cylp` or system binary). Pin working versions back into `requirements.txt`.
  - SPIKE the two known conflicts and WRITE the decision into `docs/design/solver-stack.md`:
    - (a) MIPLearn requires `gurobipy≥12` → confirm unusable under open-only; record decision: MIPLearn = design reference only.
    - (b) Ecole 0.8.1 (SCIP 8) vs PySCIPOpt 6.x (SCIP 10) → test co-install; record decision (DEFAULT: features via `model.as_pyscipopt()`, no hard Ecole dep).
  - Provide `src/opop/solver/availability.py`: detect installed solvers + versions, expose `available_solvers()`.

  **Must NOT do**: install `gurobipy`/`mip-learn` as a runtime dep; pin Ecole into core requirements; mask a failing solver import (must surface clearly).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — environment/dependency hazard with real conflicts to resolve.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: 9,10,12 · Blocked By: 1.

  **References**:
  - External (from API research): PySCIPOpt v6.2.x ↔ SCIP 10.0.2 hard coupling; OR-Tools v9.12+ snake_case; HiGHS via `highspy`; Ecole 0.8.1 inactive; MIPLearn needs Gurobi. WHY: avoid the exact co-install traps.
  - Pattern: `src/opop/solver/availability.py` mirrors lab's capability-probe style.

  **Acceptance Criteria**:
  - [ ] `python -m opop.solver.availability` lists SCIP, CP-SAT, HiGHS, CBC with versions.
  - [ ] Each solver solves a tiny known MILP (e.g. max x+y s.t. x+y≤1, x,y∈{0,1}; opt=1) to the SAME optimum.
  - [ ] `docs/design/solver-stack.md` records both conflict decisions.

  **QA Scenarios**:
  ```
  Scenario: All open solvers agree on a known optimum
    Tool: Bash
    Steps:
      1. Build the 2-var knapsack fixture in each solver wrapper
      2. Solve; collect objective + status
    Expected: every solver returns optimal objective == 1 (status OPTIMAL)
    Evidence: .omo/evidence/task-3-solver-agreement.txt

  Scenario: Ecole/PySCIPOpt co-install conflict is detected, not silently broken
    Tool: Bash
    Steps:
      1. Attempt `import ecole` in the core env; capture result
    Expected: either ecole absent (decision documented) OR import error captured and written to solver-stack.md; core still imports without ecole
    Evidence: .omo/evidence/task-3-ecole-spike.txt
  ```
  **Commit**: YES — `feat(solver): bootstrap open solvers + availability probe + conflict decisions`

- [x] 4. Core types & state objects

  **What to do**:
  - In `src/opop/model/state.py` define frozen dataclasses: `ProblemState` (instance_id, task_family, symbolic_model ref, model_graph ref, formulation_history, solver_trace_history, posterior_state ref, budget_state, incumbent_solution, incumbent_certificate, risk_flags).
  - `Phi` design vector: `(m formulation_family, v var_encoding, c constraint_templates, d decomposition, h heuristics, p solver_params, s fidelity, rho risk_thresholds)` with explicit per-field type tags (categorical/ordinal/bool/continuous) for the BO encoder.
  - `SolveTrace` (primal/dual bound series, nodes, lp_iters, cuts, first_feasible_time, status, censored:bool, memory_peak) and `ScoreRecord` (multi-metric vector + uncertainty + risks).
  - `DeltaClass` enum {A,B,C,D} and `Delta` (target, before/after fragment, declared_class).

  **Must NOT do**: put behavior/logic in these modules (pure data + validation only); import solver libs here.

  **Recommended Agent Profile**:
  - **Category**: `quick` — type definitions, single concern.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: 9,13,15,16 · Blocked By: 1.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/autoresearch-harness/autoresearch_harness/config.py` — dataclass style used lab-wide.
  - Doc: the report's `ProblemState`/phi schema (Context). WHY: these types are the contract every other module depends on.

  **Acceptance Criteria**:
  - [ ] `pytest tests/model/test_state.py` → PASS (construct each dataclass; immutability enforced; phi field-type tags round-trip).
  - [ ] `mypy src/opop/model/state.py` → no errors.

  **QA Scenarios**:
  ```
  Scenario: Phi encodes mixed field types for the BO layer
    Tool: Bash (pytest)
    Steps:
      1. Build a Phi with categorical m, bool cuts flag, continuous big_m
      2. Call phi.field_types() and phi.to_flat_dict()
    Expected: field_types maps each field to {categorical,ordinal,bool,continuous}; flat dict keys stable
    Evidence: .omo/evidence/task-4-phi.txt

  Scenario: Frozen state rejects mutation
    Tool: Bash (pytest)
    Steps:
      1. Attempt to set an attribute on a constructed ProblemState
    Expected: FrozenInstanceError raised
    Evidence: .omo/evidence/task-4-frozen.txt
  ```
  **Commit**: YES — `feat(model): core ProblemState/Phi/SolveTrace/Delta types`

- [x] 5. Config system

  **What to do**:
  - `src/opop/config.py`: dataclass-based config (`RunConfig`, `SolverConfig`, `ControllerConfig`, `BudgetConfig`) + `load_config(path)` supporting `.json` and `.yaml`, with env-var overrides. Mirror lab `config.py`.
  - Provide `configs/phase1_smoke.yaml` referenced by the Success Criteria.

  **Must NOT do**: introduce Hydra/OmegaConf; allow unknown keys silently (validate + error).

  **Recommended Agent Profile**:
  - **Category**: `quick` — config loader, single concern.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: 16,21 · Blocked By: 1.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/autoresearch-harness/autoresearch_harness/config.py` — `@dataclass` + `load_workflow()` JSON/YAML. WHY: identical idiom, reuse loader shape.

  **Acceptance Criteria**:
  - [ ] `pytest tests/test_config.py` → PASS (load JSON==YAML equivalence; env override applies; unknown key raises).
  - [ ] `python -c "from opop.config import load_config; load_config('configs/phase1_smoke.yaml')"` → exit 0.

  **QA Scenarios**:
  ```
  Scenario: JSON and YAML produce identical config; env overrides win
    Tool: Bash (pytest)
    Steps:
      1. Load equivalent .json and .yaml; assert equal dataclasses
      2. Set OPOP_BUDGET_TRIALS=7 env; reload; assert budget.trials==7
    Expected: equality holds; env override applied
    Evidence: .omo/evidence/task-5-config.txt

  Scenario: Unknown key is rejected
    Tool: Bash (pytest)
    Steps:
      1. Load a config with a bogus field
    Expected: raises ConfigError naming the bad key (no silent ignore)
    Evidence: .omo/evidence/task-5-config-bad.txt
  ```
  **Commit**: YES — `feat(config): dataclass JSON/YAML config with env overrides`

- [x] 6. Test infrastructure

  **What to do**:
  - `tests/conftest.py` (sys.path → `src`, shared fixtures: `fake_llm`, `tmp_run_dir`, `tiny_milp_fixture`, `solver_skip_if_missing`).
  - Wire pytest markers from `pyproject.toml`; add `ruff` + `mypy` configs; create `tests/` mirror of `src/opop/` packages.
  - Add a single example passing test to prove the harness.

  **Must NOT do**: add CI workflows (lab uses none); require GPU for default test run.

  **Recommended Agent Profile**:
  - **Category**: `quick` — test scaffolding, single concern.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: every test-bearing task · Blocked By: 1.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/SMIAutoResearch/tests/conftest.py` — `fake_provider`/`temp_workspace` fixtures. WHY: reuse fixture idiom; `solver_skip_if_missing` keeps CI-less runs green when a solver is absent.

  **Acceptance Criteria**:
  - [ ] `pytest -q` → 1 passed, 0 errors.
  - [ ] `pytest -m smoke -q` → collects only smoke-marked tests.

  **QA Scenarios**:
  ```
  Scenario: Harness collects and runs with markers
    Tool: Bash
    Steps:
      1. pytest -q ; pytest -m smoke -q ; pytest -m "gpu" -q
    Expected: default run green; smoke subset runs; gpu subset deselected on no-GPU
    Evidence: .omo/evidence/task-6-pytest.txt

  Scenario: Missing-solver fixture skips, not fails
    Tool: Bash (pytest)
    Steps:
      1. Use solver_skip_if_missing('gcg') in a test on an env without GCG
    Expected: test is SKIPPED with a clear reason (not errored)
    Evidence: .omo/evidence/task-6-skip.txt
  ```
  **Commit**: YES — `test(infra): conftest, fixtures, markers, ruff/mypy config`

- [x] 7. Benchmark registry schema + immutable splits + leakage groups

  **What to do**:
  - `src/opop/bench/registry.py` + `benchmarks/registry.yaml` schema with fields per entry: `problem_type, source, split{dev|validation|test|ood_test}, license, instance_count, time_limit_sec, baseline_set, leakage_group, checksum`.
  - Enforce IMMUTABILITY: a `split_manifest.lock` (hashes of instance→split assignment); loader refuses to run if the lock changed without an explicit `--reseal`.
  - `register()`, `get_split(split)`, `assert_no_overlap()` (no instance in >1 split; no leakage_group spanning dev and test).

  **Must NOT do**: allow adding a benchmark family/baseline without phase+thesis tags; allow test/ood instances to be loaded by dev/validation code paths.

  **Recommended Agent Profile**:
  - **Category**: `quick` — schema + invariants, single concern.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: 19,20,33,34,35 · Blocked By: 1.

  **References**:
  - Doc: Metis leakage policy + registry fields (Verification Strategy). WHY: this is the scientific-integrity backbone — it must exist before any tuning touches data.

  **Acceptance Criteria**:
  - [ ] `pytest tests/bench/test_registry.py` → PASS (overlap detection; leakage_group spanning splits rejected; lock mismatch refused).
  - [ ] `python -m opop.bench.registry --validate benchmarks/registry.yaml` → exit 0 on a valid file.

  **QA Scenarios**:
  ```
  Scenario: Overlapping instance across splits is rejected
    Tool: Bash (pytest)
    Steps:
      1. Build a registry where instance X is in both dev and test
      2. Call assert_no_overlap()
    Expected: raises LeakageError naming instance X
    Evidence: .omo/evidence/task-7-overlap.txt

  Scenario: Tampered split lock is refused
    Tool: Bash
    Steps:
      1. Seal splits; mutate one assignment; run loader without --reseal
    Expected: loader aborts with a lock-mismatch error
    Evidence: .omo/evidence/task-7-lock.txt
  ```
  **Commit**: YES — `feat(bench): benchmark registry, immutable splits, leakage invariants`

- [x] 8. Bayesian optimization base (port `acquisition.py`)

  **What to do**:
  - Port `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_e/src/acquisition.py` into `src/opop/controller/gp.py` + `acquisition.py`: `GaussianProcess` (Matern-5/2), `ucb/ei/random` acquisitions, `run_bo_trials`, `scalarized_reward`. numpy+torch only.
  - Wrap behind a `Surrogate` + `Acquisition` Protocol so the Wave-4 controller ladder (SMAC/TPE/BoTorch/structured) can plug in without changing callers.
  - Provide a `RandomSearch` baseline controller implementing the same Protocol.

  **Must NOT do**: pull in BoTorch/SMAC yet (Wave 4); add multi-fidelity (Wave 4, task 29).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low` — port + Protocol wrap.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 1 · Blocks: 15,28 · Blocked By: 1.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_e/src/acquisition.py` — GP+UCB/EI+`run_bo_trials`+`scalarized_reward` (numpy+torch only). WHY: self-contained, dependency-light; ideal Phase-1 baseline before BoTorch.
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_e/src/meta_tuner.py` — for later warm-start (task 32), note only.

  **Acceptance Criteria**:
  - [ ] `pytest tests/controller/test_gp.py` → PASS (GP fits a 1-D sine, EI improves over random on a toy maximization within N steps).
  - [ ] `scalarized_reward` matches a hand-computed value for a fixed metric vector + weights.

  **QA Scenarios**:
  ```
  Scenario: EI beats random on a toy 1-D objective
    Tool: Bash (pytest)
    Steps:
      1. Optimize f(x)=-(x-0.3)^2 over [0,1], 15 trials, seed=0
      2. Compare best-found: EI vs random
    Expected: EI best >= random best (>= within tolerance); GP posterior variance shrinks near sampled x
    Evidence: .omo/evidence/task-8-ei.txt

  Scenario: Scalarization is deterministic and correct
    Tool: Bash (pytest)
    Steps:
      1. scalarized_reward({gap:0.1, time:100}, weights) vs hand value
    Expected: exact match (to 1e-9)
    Evidence: .omo/evidence/task-8-scalar.txt
  ```
  **Commit**: YES — `feat(controller): port GP+EI/UCB BO base behind Surrogate/Acquisition protocols`

- [x] 9. Symbolic model IR + MPS/LP I/O

  **What to do**:
  - `src/opop/model/ir.py`: an internal MILP representation (vars with type/bounds, linear constraints, objective, named index sets, metadata) + a `model_graph` (variable–constraint bipartite incidence) view.
  - I/O: read/write MPS and LP via PySCIPOpt; round-trip `IR → MPS → IR` must be lossless for the supported subset. Provide `from_pyscipopt(model)` / `to_pyscipopt(ir)`.
  - `apply_delta(ir, delta) -> ir'` for class-A/B/C deltas (pure, returns a new IR).

  **Must NOT do**: support nonlinear terms yet (Wave 5); mutate IR in place; depend on a specific solver beyond the PySCIPOpt reader.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — central data structure, many consumers, correctness-critical.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 10,11,12,30 · Blocked By: 3,4.

  **References**:
  - External: PySCIPOpt `Model.readProblem`/`writeProblem`, `model.getVars/getConss`. WHY: reader/writer is the cheapest correct MPS path under SCIP 10.
  - Type: `src/opop/model/state.py` (`Delta`, `DeltaClass`). WHY: `apply_delta` consumes these.

  **Acceptance Criteria**:
  - [ ] `pytest tests/model/test_ir.py` → PASS (round-trip lossless on 3 fixture MPS; bipartite graph node/edge counts match constraint nnz).
  - [ ] `apply_delta` with a class-A rename produces an IR that re-exports to an equivalent MPS.

  **QA Scenarios**:
  ```
  Scenario: Lossless MPS round-trip
    Tool: Bash (pytest)
    Steps:
      1. For each fixture: read MPS -> IR -> write MPS' -> read MPS'
      2. Assert identical var/constr counts, bounds, objective coeffs (to 1e-9)
    Expected: byte-or-semantic equivalence on all fixtures
    Evidence: .omo/evidence/task-9-roundtrip.txt

  Scenario: Unsupported nonlinear term is rejected cleanly
    Tool: Bash (pytest)
    Steps:
      1. Feed a model with a quadratic term to from_pyscipopt
    Expected: raises UnsupportedModelError (not a silent drop)
    Evidence: .omo/evidence/task-9-nonlinear.txt
  ```
  **Commit**: YES — `feat(model): MILP IR, bipartite graph, MPS/LP round-trip, apply_delta`

- [x] 10. Analyzer subset (Phase-1)

  **What to do**:
  - `src/opop/analyzer/`: deterministic checks — (i) dimension/units/index consistency; (ii) LP-relaxation stats via SCIP root (LP objective, integrality gap estimate, fractional-variable pattern); (iii) redundancy/trivial-infeasibility/conflict detection; (iv) valid-inequality CANDIDATE generation from a whitelist (e.g. cover, clique for set-packing fixtures).
  - Output a structured `AnalysisReport` (flags, relaxation metrics, candidate cuts, decomposability=NONE for Phase-1).

  **Must NOT do**: add Benders/DW/symmetry analysis (Wave 4, task 26); emit cuts that aren't certified valid (defer validity proof to task 11).

  **Recommended Agent Profile**:
  - **Category**: `deep` — OR analysis logic, hairy correctness, the framework's differentiator.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 14,26 · Blocked By: 3,9.

  **References**:
  - External: PySCIPOpt root LP — `model.setPresolve`, solve root, `getLPSolstat`, `getVal`; stats getters (getDualbound/getPrimalbound). WHY: cheapest LP-relax signal.
  - Doc: report's Analyzer responsibilities + Metis "valid inequality certificate" boundary. WHY: Analyzer proposes; task 11 certifies.

  **Acceptance Criteria**:
  - [ ] `pytest tests/analyzer/test_analyzer.py` → PASS (LP gap computed vs known value on fixtures; redundant constraint flagged; index mismatch flagged).
  - [ ] On a set-packing fixture, ≥1 clique/cover candidate cut is produced.

  **QA Scenarios**:
  ```
  Scenario: LP-relaxation gap matches a known fixture
    Tool: Bash (pytest)
    Steps:
      1. Analyze a fixture with known LP=12.5, IP=15 -> report.lp_gap
    Expected: lp_gap ≈ (15-12.5)/15 within 1e-6
    Evidence: .omo/evidence/task-10-lpgap.txt

  Scenario: Detects a redundant constraint and an index error
    Tool: Bash (pytest)
    Steps:
      1. Analyze a model with a duplicate constraint + a constraint indexing a missing set member
    Expected: flags {redundant:[c_i]} and {index_error:[c_j]}; does not crash
    Evidence: .omo/evidence/task-10-flags.txt
  ```
  **Commit**: YES — `feat(analyzer): dim/index checks, LP-relax stats, redundancy, valid-ineq candidates`

- [x] 11. Verification gate: delta classes A–D + certificates

  **What to do**:
  - `src/opop/verify/gate.py`: classify a `Delta` and run the matching certificate BEFORE any evaluation:
    - **A equivalent**: check feasible-integer-solution + objective preservation (sample-based + structural where possible) within tolerances (feas 1e-7, obj 1e-6).
    - **B valid-inequality**: verify the added inequality is satisfied by all feasible integer incumbents (no integer feasible point removed) — e.g. via separation against known feasible solutions + dominance argument; record a certificate JSON.
    - **C heuristic/param**: assert no change to vars/constraints/objective/bounds (semantic no-op).
    - **D risky**: route to sandbox; NEVER returns "pass for main eval".
  - Fail-closed: unknown/unprovable ⇒ reject. Emit `verification/report.json` (status, delta_class, preserved flags, counterexample|null).

  **Must NOT do**: pass a delta on symbolic check alone if solver-backed check fails; let any feasible-region change through without a certificate.

  **Recommended Agent Profile**:
  - **Category**: `deep` — the scientific-integrity keystone; subtle correctness.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 16 · Blocked By: 9.

  **References**:
  - Doc: Metis delta-class semantics A/B/C/D + certificate schema (Verification Strategy). WHY: this is the literal spec.
  - Type: `src/opop/model/state.py` (`Delta`,`DeltaClass`); `model/ir.py` (`apply_delta`).

  **Acceptance Criteria**:
  - [ ] `pytest tests/verify/test_gate.py` → PASS for all four classes incl. fail-closed.
  - [ ] A delta that removes a feasible integer solution is REJECTED with a counterexample.

  **QA Scenarios**:
  ```
  Scenario: A valid cut passes; a feasibility-breaking "cut" is rejected
    Tool: Bash (pytest)
    Steps:
      1. Class-B: add a true valid inequality -> gate; assert status=pass, counterexample=null
      2. Add an inequality that cuts off a known feasible integer point -> gate
    Expected: (1) pass; (2) reject with the cut-off point as counterexample
    Evidence: .omo/evidence/task-11-cut.txt

  Scenario: Unknown/unprovable delta fails closed
    Tool: Bash (pytest)
    Steps:
      1. Submit a class-D (uncertified reformulation) delta
    Expected: status=reject, routed to sandbox, NOT eligible for main eval
    Evidence: .omo/evidence/task-11-failclosed.txt
  ```
  **Commit**: YES — `feat(verify): A–D delta classification + fail-closed certificates`

- [x] 12. SCIP solver kernel + trajectory extraction

  **What to do**:
  - `src/opop/solver/scip.py`: a `SolverKernel` implementation wrapping PySCIPOpt — compile `IR`+`Phi.p` (params) into a SCIP model, set `threads=1`, time/memory limits, seed; solve; return a `SolveTrace`.
  - Trajectory: register an `Eventhdlr` to capture (time, primalbound, dualbound) on `BESTSOLFOUND`/`DUALBOUNDIMPROVED`; also collect `getNTotalNodes`, `getNLPIterations`, `getPrimalDualIntegral`, `getGap`, `getSolvingTime`, status, first-feasible time. Mark `censored=True` when terminated by the time limit without optimality.
  - Apply Phase-1 proposer hooks: inject whitelisted separator (class-B cut) and optional decomposition flag toggles via params.

  **Must NOT do**: use `getNNodes` (resets at restart) — use `getNTotalNodes`; rely on Gurobi/CP-SAT here (this task is SCIP-only); silently swallow solver errors.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — solver integration with callback plumbing.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 13,22,23,24,25 · Blocked By: 3,9.

  **References**:
  - External (API research): PySCIPOpt v6.2.x — `Eventhdlr`+`catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND/DUALBOUNDIMPROVED)`; stats `getPrimalbound/getDualbound/getNTotalNodes/getNLPIterations/getPrimalDualIntegral/writeStatisticsJson`. WHY: exact API + the `getNNodes` reset pitfall.
  - Type: `SolveTrace` (task 4). WHY: output contract.

  **Acceptance Criteria**:
  - [ ] `pytest tests/solver/test_scip.py` → PASS (known MILP solved to known optimum; trace has monotone dual bound, non-empty primal series, correct status).
  - [ ] A 2s time limit on a hard fixture yields `censored=True` with `time≈2`.

  **QA Scenarios**:
  ```
  Scenario: Trajectory captured on a known instance
    Tool: Bash
    Steps:
      1. Solve a fixture with opt=42, capture SolveTrace
      2. Assert final primal==42, status OPTIMAL, dual series non-decreasing, first_feasible_time>0
    Expected: all assertions hold; primal_integral>0
    Evidence: .omo/evidence/task-12-trace.txt

  Scenario: Timeout marks censored correctly
    Tool: Bash
    Steps:
      1. Solve a hard fixture with time_limit=2s, threads=1
    Expected: status not OPTIMAL, censored=True, runtime in [2,2.5]s, dual<primal (open gap recorded)
    Evidence: .omo/evidence/task-12-censored.txt
  ```
  **Commit**: YES — `feat(solver): SCIP kernel with event-based trajectory + censoring`

- [x] 13. Evaluator: multi-metric vector + right-censoring + primal integral

  **What to do**:
  - `src/opop/evaluator/`: turn a `SolveTrace` (+ reference optimum/best-known) into a `ScoreRecord`: feasible, certified, final gap@T, objective, time-to-first-feasible, **primal integral** (∫ normalized primal gap dt), primal-dual gap integral, nodes, cuts, memory, and a robustness/uncertainty estimate (optional replay).
  - Right-censoring: timeouts recorded as censored runtime (lower bound), NOT solved@limit; expose both a censored-aware aggregate and an optional PAR10 auxiliary.
  - Scalarization hook for the BO controller (weights configurable), and a multi-objective vector passthrough.

  **Must NOT do**: replace timeout with a fixed penalty in the primary record (only as a clearly-labeled auxiliary); treat a censored run as optimal.

  **Recommended Agent Profile**:
  - **Category**: `deep` — metric math (integrals, censoring) with correctness traps.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 16,18,29 · Blocked By: 12.

  **References**:
  - Doc: report's Evaluator vector + Metis censoring/cost rules. External concept: primal integral (Berthold). WHY: standard anytime metric; must be computed from the trajectory, not endpoints.
  - Type: `SolveTrace`,`ScoreRecord` (task 4).

  **Acceptance Criteria**:
  - [ ] `pytest tests/evaluator/test_metrics.py` → PASS (primal integral matches hand-computed value on a synthetic 3-step trajectory; censored flag preserved).
  - [ ] PAR10 auxiliary equals 10×limit on a timeout fixture; primary record keeps it censored.

  **QA Scenarios**:
  ```
  Scenario: Primal integral matches hand calculation
    Tool: Bash (pytest)
    Steps:
      1. Trajectory: gap 1.0 for [0,1], 0.5 for [1,2], 0.0 for [2,3] -> integral
    Expected: integral == 1.5 (±1e-9)
    Evidence: .omo/evidence/task-13-primalintegral.txt

  Scenario: Censored run is not counted as solved
    Tool: Bash (pytest)
    Steps:
      1. Score a censored SolveTrace (no optimality at limit)
    Expected: record.feasible may be True but record.optimal=False, censored=True; PAR10 aux==10*limit; primary aggregate uses censored handling
    Evidence: .omo/evidence/task-13-censored.txt
  ```
  **Commit**: YES — `feat(evaluator): anytime metrics, primal integral, right-censoring, scalarization`

- [x] 14. Proposer (Phase-1 restricted)

  **What to do**:
  - `src/opop/proposer/`: given `ProblemState` + `AnalysisReport`, propose a small set of `Delta`s restricted to the Phase-1 space: (i) SCIP parameter changes (from a curated knob list); (ii) whitelisted valid-inequality templates (only those the Analyzer flagged as candidates); (iii) optional decomposition flag (no-op unless structure detected). Each delta carries a declared `DeltaClass`.
  - Use the LLM adapter (task 2) for delta *selection/ranking* guided by analysis features + a structured prior; fall back to a deterministic rule-based proposer (`FakeLLMClient`) so tests never need network.
  - Emit candidate deltas with rationale; the controller (task 15) decides which to evaluate.

  **Must NOT do**: propose formulation-family swaps or reformulations (Wave 4, task 27); propose class-D deltas into the main path; let the LLM emit raw solver code (it selects from typed templates only).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — LLM-guided structured proposal with a safety envelope.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 16,27 · Blocked By: 2,10.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_g/src/llm_proposer.py` — `propose_*(context)->Pattern` interface + strategy enum. WHY: reuse the proposer interface shape.
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/topic2-format-efficiency/tools/orchestrator.py` — planner decomposition style. Type: `Delta`,`Phi` (task 4).

  **Acceptance Criteria**:
  - [ ] `pytest tests/proposer/test_proposer.py` → PASS (with FakeLLMClient: returns ≥1 typed delta from the whitelist; never emits class-D into main; respects analysis candidates).
  - [ ] Proposed deltas all carry a valid `DeltaClass` and pass schema validation.

  **QA Scenarios**:
  ```
  Scenario: Proposer returns only whitelisted, typed deltas
    Tool: Bash (pytest)
    Steps:
      1. Feed an AnalysisReport with 2 candidate cover cuts + LP gap
      2. proposer.propose(state, report) with FakeLLMClient
    Expected: returns deltas drawn only from {param changes, the 2 candidate cuts}; each has DeltaClass in {A,B,C}; none class-D
    Evidence: .omo/evidence/task-14-propose.txt

  Scenario: LLM hallucinated illegal delta is filtered
    Tool: Bash (pytest)
    Steps:
      1. FakeLLMClient returns a free-form reformulation suggestion
    Expected: proposer drops/normalizes it; logs a rejection; output stays within typed space
    Evidence: .omo/evidence/task-14-filter.txt
  ```
  **Commit**: YES — `feat(proposer): Phase-1 LLM-guided typed delta proposal with safety envelope`

- [x] 15. Controller (Phase-1: random + ONE BO baseline)

  **What to do**:
  - `src/opop/controller/phase1.py`: ask-tell controller over the restricted Phase-1 `Phi` subspace using the ported GP+EI (task 8) AND a `RandomSearch` baseline (same Protocol). Encode the mixed restricted space (categorical cut on/off, ordinal param levels, continuous knobs) into the GP input.
  - Maintain posterior state in `ProblemState`; choose next delta(s) to evaluate from the proposer's candidate set via acquisition value; respect budget.

  **Must NOT do**: add SMAC/TPE/BoTorch/structured surrogates or multi-fidelity (Wave 4, tasks 28–29); search outside the restricted Phase-1 space.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — BO control logic over a mixed encoded space.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 2 · Blocks: 16,28 · Blocked By: 8.

  **References**:
  - Pattern: `src/opop/controller/gp.py`/`acquisition.py` (task 8) — `run_bo_trials` ask-tell shape. WHY: reuse directly.
  - Type: `Phi.field_types()` (task 4) for the encoder. Doc: Metis controller-ladder rung 1–3.

  **Acceptance Criteria**:
  - [ ] `pytest tests/controller/test_phase1.py` → PASS (ask-tell loop runs to budget; on a synthetic surrogate objective BO ≥ random; posterior updates after each tell).
  - [ ] Encoder maps categorical/ordinal/continuous phi fields to a stable numeric vector (round-trip stable).

  **QA Scenarios**:
  ```
  Scenario: BO controller outperforms random on a synthetic phi->score function
    Tool: Bash (pytest)
    Steps:
      1. Define a known phi->score surrogate; run 20-trial ask-tell with seed=0 for BO and random
    Expected: BO best >= random best; controller stops at budget
    Evidence: .omo/evidence/task-15-bo-vs-random.txt

  Scenario: Mixed-space encoding is stable
    Tool: Bash (pytest)
    Steps:
      1. encode(phi) -> vector -> decode -> phi'
    Expected: phi'==phi for all field types
    Evidence: .omo/evidence/task-15-encode.txt
  ```
  **Commit**: YES — `feat(controller): Phase-1 ask-tell BO + random baseline over restricted phi`

- [x] 16. Orchestrator: Phase-1 closed loop

  **What to do**:
  - `src/opop/orchestrator/loop.py`: drive the loop — Proposer → Analyzer → **Verify gate** → Solver → Evaluator → Controller.update → repeat until budget. Reuse the `self_evolution.py` stagnation/fingerprint pattern to stop on no-improvement.
  - Persist `events.jsonl` (one record per proposal: phi, delta_class, verify status, trace summary, score) and the running incumbent + certificate into a run directory. Honor budget/interrupt; deltas failing the gate are recorded and skipped (never solved).

  **Must NOT do**: evaluate a delta that failed verification; let an exception abort the whole run silently (record + continue or fail loudly); add multi-kernel scheduling (Wave 4).

  **Recommended Agent Profile**:
  - **Category**: `deep` — integration of all Phase-1 modules; control-flow correctness.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: NO (integrator) · Group: Wave 3 · Blocks: 17,19,21 · Blocked By: 4,5,11,13,14,15.

  **References**:
  - Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/autoresearch-harness/autoresearch_harness/self_evolution.py` (bounded loop + stagnation) and `.../orchestrator.py` (artifact store + telemetry). WHY: reuse loop/journal skeleton.

  **Acceptance Criteria**:
  - [ ] `pytest tests/orchestrator/test_loop.py` → PASS (loop runs N iters with fakes; gate-failed delta never reaches solver; events.jsonl well-formed; incumbent monotonically improves or holds).
  - [ ] Stagnation detection stops early when no improvement for K rounds.

  **QA Scenarios**:
  ```
  Scenario: Gate-rejected delta is recorded but never solved
    Tool: Bash (pytest)
    Steps:
      1. Inject a proposer that emits one valid + one feasibility-breaking delta
      2. Run 1 loop iteration with a spy Solver
    Expected: solver called only for the valid delta; events.jsonl has a 'verify_rejected' entry for the other
    Evidence: .omo/evidence/task-16-gateskip.txt

  Scenario: Loop honors budget and stagnation
    Tool: Bash (interactive_bash)
    Steps:
      1. Run loop with budget=10 and a flat objective
    Expected: stops at <=10 iters, logs 'stagnation_stop'; exit 0; incumbent recorded
    Evidence: .omo/evidence/task-16-budget.txt
  ```
  **Commit**: YES — `feat(orchestrator): Phase-1 closed loop with verify gate, journal, stagnation`

- [x] 17. Reproducibility manifest + replay

  **What to do**:
  - `src/opop/orchestrator/repro.py`: write `repro_manifest.json` per run — git_commit, container_digest (if any), python + solver versions (`getVersion`/`getSCIPversion`, ortools/highs), hardware, threads=1, ALL seeds (SCIP, CP-SAT, BO/numpy/torch, LLM sampling), time/memory limits, tolerances (feas 1e-7, opt 1e-6).
  - `python -m opop.replay --run <dir> --strict`: re-execute from the manifest and assert artifacts regenerate within tolerance.

  **Must NOT do**: allow a run to complete without a manifest; let non-deterministic defaults (random threads, unpinned seed) into experimental runs.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — determinism plumbing across many RNGs.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 3 · Blocks: 21 · Blocked By: 16.

  **References**:
  - Doc: Metis reproducibility-manifest field list (Verification Strategy). External: PySCIPOpt `getVersion`/`getSCIPversion`; SMAC `overwrite=True` note (warmstart hazard). WHY: exact fields + determinism traps.

  **Acceptance Criteria**:
  - [ ] `pytest tests/orchestrator/test_repro.py` → PASS (manifest contains every required field; missing field ⇒ run aborts).
  - [ ] `replay --strict` on a tiny seeded run reproduces the same incumbent objective + same number of accepted deltas.

  **QA Scenarios**:
  ```
  Scenario: Strict replay reproduces results
    Tool: Bash (interactive_bash)
    Steps:
      1. Run a tiny seeded loop -> runs/repro_demo
      2. python -m opop.replay --run runs/repro_demo --strict
    Expected: exit 0; "REPRODUCED" with identical incumbent objective + accepted-delta count
    Evidence: .omo/evidence/task-17-replay.txt

  Scenario: Run without complete manifest is refused
    Tool: Bash (pytest)
    Steps:
      1. Force-delete a manifest field; attempt to finalize the run
    Expected: aborts with MissingManifestField error
    Evidence: .omo/evidence/task-17-manifest.txt
  ```
  **Commit**: YES — `feat(repro): reproducibility manifest + strict replay`

- [x] 18. Comparison report + statistical tests

  **What to do**:
  - `src/opop/experiments/compare.py` + `python -m opop.eval.compare`: load `results.parquet`, compute per-method primal integral, solved-rate, shifted-geomean end-to-end time; run Wilcoxon signed-rank (α=0.05); report relative improvement vs a chosen baseline and whether it clears the min-effect thresholds (10% PI / 20% time / 5pp solved-rate).
  - Emit machine-readable `comparison_report.json` (baseline, method, metric, significant:bool, relative_improvement, n_seeds) + a human table.

  **Must NOT do**: aggregate over censored runtimes naively (use censored-aware handling from task 13); claim a win below min-effect or without significance.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — stats + reporting, well-scoped.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 3 · Blocks: 21,40 · Blocked By: 13.

  **References**:
  - Doc: Win Definition (Verification Strategy) — exact metrics/test/effect-sizes. External: scipy `wilcoxon`; shifted geometric mean (shift s=10) convention. WHY: standard OR benchmarking stats.

  **Acceptance Criteria**:
  - [ ] `pytest tests/experiments/test_compare.py` → PASS (Wilcoxon p matches scipy on a fixture; shifted-geomean matches hand value; min-effect gating correct).
  - [ ] `comparison_report.json` has all required fields and is valid JSON.

  **QA Scenarios**:
  ```
  Scenario: Significance + effect gating on a fixture
    Tool: Bash
    Steps:
      1. Feed paired results where method is 15% better on PI across 6 seeds
      2. python -m opop.eval.compare --metric primal_integral --baseline scip-default --method opop
    Expected: significant=true, relative_improvement≈0.15, clears_min_effect=true
    Evidence: .omo/evidence/task-18-compare.txt

  Scenario: Below-threshold improvement is NOT a win
    Tool: Bash
    Steps:
      1. Feed results with 3% PI improvement
    Expected: report marks clears_min_effect=false even if significant
    Evidence: .omo/evidence/task-18-nowin.txt
  ```
  **Commit**: YES — `feat(eval): comparison report with Wilcoxon + shifted-geomean + min-effect gating`

- [x] 19. Leakage audit + cost accounting

  **What to do**:
  - `python -m opop.bench.audit_leakage --run <dir> --registry benchmarks/registry.yaml`: assert 0 test/ood_test instances were used during any tuning/proposal step (cross-reference events.jsonl instance ids against split manifest); emit `leakage_audit.json`.
  - Cost accounting in the evaluator/orchestrator: every result row records `solver_wall_time, analyzer_time, proposer_time, controller_time, verification_time, llm_tokens_in/out, llm_cost_usd, total_wall_time`. Provide both solver-only and end-to-end aggregates.

  **Must NOT do**: allow a compute-efficiency claim from solver-only time; let a test/ood instance enter a tuning path without failing the audit.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — integrity tooling, well-scoped.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 3 · Blocks: 21,39 · Blocked By: 7,16.

  **References**:
  - Doc: Metis leakage policy + cost-accounting column list. Pattern: `TokenTracker` from task 2 for llm_tokens/cost. WHY: exact required columns + audit semantics.

  **Acceptance Criteria**:
  - [ ] `pytest tests/bench/test_leakage.py` → PASS (audit flags a planted test-instance-in-tuning; passes a clean run).
  - [ ] Every result row carries all cost columns; end-to-end ≥ solver-only.

  **QA Scenarios**:
  ```
  Scenario: Planted leakage is caught
    Tool: Bash
    Steps:
      1. Craft a run whose events.jsonl tuned on a 'test' instance
      2. python -m opop.bench.audit_leakage --run <dir> --registry ...
    Expected: leakage_audit.json status=fail, test_instances_used_for_tuning>0; nonzero exit
    Evidence: .omo/evidence/task-19-leak-fail.txt

  Scenario: Clean run passes + cost columns complete
    Tool: Bash
    Steps:
      1. Audit a dev-only run; inspect results.parquet columns
    Expected: status=pass (0 violations); all 9 cost columns present; total>=solver_only
    Evidence: .omo/evidence/task-19-clean.txt
  ```
  **Commit**: YES — `feat(bench): leakage audit + solver-only/end-to-end cost accounting`

- [x] 20. Phase-1 MILP dev set acquisition

  **What to do**:
  - `src/opop/bench/sources/`: downloader + loader for a SMALL Phase-1 dev set — a 20–50 instance MIPLIB 2017 subset (by checksum) PLUS a synthetic generator (set-cover/knapsack/facility) with controllable size/structure. Register all entries in `benchmarks/registry.yaml` (split=dev/validation), with `time_limit_sec` and `baseline_set`.
  - Provide checksums + a `make_phase1_splits` that seals dev/validation (no test/ood yet — those arrive in Wave 6).

  **Must NOT do**: pull large full benchmark suites (Wave 6, tasks 33–35); assign any instance to test/ood here; commit raw instance blobs (scripts + checksums only).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — data plumbing with integrity constraints.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: YES · Group: Wave 3 · Blocks: 21,33 · Blocked By: 7.

  **References**:
  - External: MIPLIB 2017 collection + per-instance metadata; standard CO generators. Type: registry schema (task 7). WHY: anchor Phase-1 on a real-but-small, reproducible set.

  **Acceptance Criteria**:
  - [ ] `pytest tests/bench/test_phase1_set.py` → PASS (synthetic generator deterministic by seed; downloaded subset matches checksums; registry validates).
  - [ ] `make_phase1_splits` seals dev/validation; re-run is idempotent.

  **QA Scenarios**:
  ```
  Scenario: Deterministic synthetic generation + checksum integrity
    Tool: Bash
    Steps:
      1. Generate set-cover instances with seed=0 twice; diff
      2. Verify one downloaded MIPLIB instance against its checksum
    Expected: identical generation; checksum matches
    Evidence: .omo/evidence/task-20-data.txt

  Scenario: No test/ood assignment leaks in
    Tool: Bash (pytest)
    Steps:
      1. Inspect sealed Phase-1 split manifest
    Expected: only dev/validation present; assert_no_overlap passes
    Evidence: .omo/evidence/task-20-splits.txt
  ```
  **Commit**: YES — `feat(bench): Phase-1 MILP dev set (MIPLIB subset + synthetic) + sealed splits`

- [x] 21. Phase-1 END-TO-END smoke + sanity experiment  <<< MILESTONE: LOOP PROVEN

  **What to do**:
  - Wire the full CLI: `python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke` runs the closed loop over the Phase-1 dev set vs SCIP-default, producing ALL artifacts: `results.parquet, events.jsonl, verification/*.json, repro_manifest.json, comparison_report.json, leakage_audit.json`.
  - Run a small sanity experiment (≥5 seeds, a fixed time limit) and produce the comparison report vs SCIP-default. **Success = the loop proposes ≥1 certified delta, evaluates it under budget, records censored metrics, and emits a reproducible, leakage-clean comparison report.** (NOT "beats everything".)

  **Must NOT do**: gate this milestone on beating baselines; expand to other solvers/problem classes; skip any artifact.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — end-to-end integration + experiment runner.
  - **Skills**: none.

  **Parallelization**: Can Run In Parallel: NO (milestone integrator) · Group: Wave 3 · Blocks: ALL of Wave 4+ (entry criterion) · Blocked By: 16,17,18,19,20.

  **References**:
  - Doc: Metis Phase-1 success criterion + end-to-end smoke artifact list. Config: `configs/phase1_smoke.yaml` (task 5). WHY: this is the de-risking gate for the entire project.

  **Acceptance Criteria**:
  - [ ] `python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke` exits 0 and writes ALL six artifacts.
  - [ ] `replay --strict` reproduces; `audit_leakage` → 0 violations; ≥1 certified delta accepted; censored runs handled.

  **QA Scenarios**:
  ```
  Scenario: Full Phase-1 loop produces all artifacts, reproducibly
    Tool: Bash (interactive_bash)
    Steps:
      1. python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke
      2. ls runs/smoke ; python -m opop.replay --run runs/smoke --strict ; python -m opop.bench.audit_leakage --run runs/smoke --registry benchmarks/registry.yaml
    Expected: all six artifacts exist; replay REPRODUCED; leakage 0; comparison_report.json valid; >=1 accepted certified delta
    Evidence: .omo/evidence/task-21-e2e.txt

  Scenario: A run with a broken solver fails loudly, not silently
    Tool: Bash
    Steps:
      1. Point config at a missing solver; run
    Expected: clear actionable error + nonzero exit; partial artifacts flagged incomplete (no false 'success')
    Evidence: .omo/evidence/task-21-fail.txt
  ```
  **Commit**: YES — `feat(run): Phase-1 end-to-end CLI + sanity experiment (loop proven)`

> **Wave 4+ entry criterion**: Task 21 (Phase-1 loop proven) PASSED. Each task below extends a proven seam; do not start before the seam exists.

- [x] 22. CP-SAT solver kernel + trajectory

  **What to do**: Add `src/opop/solver/cpsat.py` implementing the `SolverKernel` Protocol via OR-Tools CP-SAT — build from IR (integers only; scale rationals), `CpSolverSolutionCallback` to capture (time, objective, best_bound), `add_hint` for warm start, `num_workers`/`max_time_in_seconds`, status/objective_value/best_objective_bound; mark censored on UNKNOWN-at-limit.

  **Must NOT do**: use deprecated PascalCase API; feed non-integer coefficients without scaling; treat CP-SAT bound semantics as identical to SCIP (document differences).

  **Recommended Agent Profile**: Category `unspecified-high` (second solver integration, callback plumbing). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 30,38 · Blocked By: 12.

  **References**: External (API research) OR-Tools v9.12+ snake_case `new_int_var/add/solve`, `CpSolverSolutionCallback.on_solution_callback`, `model.add_hint`, `solver.parameters.{num_workers,max_time_in_seconds}`. Type: `SolveTrace`, `SolverKernel` (tasks 4,12). WHY: exact API + integers-only pitfall.

  **Acceptance Criteria**:
  - [ ] `pytest tests/solver/test_cpsat.py` → PASS (known MILP → same optimum as SCIP; trajectory captured; censored on timeout).
  - [ ] Cross-solver agreement test: SCIP and CP-SAT agree on optimum for 3 fixtures.

  **QA Scenarios**:
  ```
  Scenario: CP-SAT matches SCIP optimum + captures trajectory
    Tool: Bash (pytest)
    Steps: solve 3 fixtures in CP-SAT; compare objective to SCIP; inspect callback series
    Expected: identical optima (integer fixtures); non-empty solution series; status mapped correctly
    Evidence: .omo/evidence/task-22-cpsat.txt
  Scenario: Non-integer coefficient handled by scaling, not silent wrong answer
    Tool: Bash (pytest)
    Steps: feed a model with 0.5 coefficients
    Expected: auto-scaled with recorded factor OR explicit error; never a silently wrong optimum
    Evidence: .omo/evidence/task-22-scale.txt
  ```
  **Commit**: YES — `feat(solver): CP-SAT kernel + trajectory + cross-solver agreement`

- [x] 23. HiGHS + CBC kernels + trajectory

  **What to do**: Add `src/opop/solver/highs.py` (via `highspy`) and `src/opop/solver/cbc.py` (via pulp/cylp or system binary) implementing `SolverKernel`; capture available trajectory signals (bounds, time, nodes where exposed); mark censored on timeout. Register in `availability.py`.

  **Must NOT do**: assume HiGHS/CBC expose the same rich callbacks as SCIP (degrade gracefully; document gaps); block the wave if CBC is absent (skip-if-missing).

  **Recommended Agent Profile**: Category `unspecified-high` (two more adapters). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 39 · Blocked By: 12.

  **References**: External `highspy` MIP API; CBC via pulp. Type: `SolverKernel`. WHY: open-baseline kernels for breadth + CI-light verification.

  **Acceptance Criteria**:
  - [ ] `pytest tests/solver/test_highs.py tests/solver/test_cbc.py` → PASS (optimum agreement; graceful trajectory degradation documented).

  **QA Scenarios**:
  ```
  Scenario: HiGHS/CBC agree on optimum; degrade trajectory gracefully
    Tool: Bash (pytest)
    Steps: solve fixtures; compare optima; inspect trace fields present/absent
    Expected: optima match; missing trajectory fields are None (not fabricated)
    Evidence: .omo/evidence/task-23-highs-cbc.txt
  Scenario: CBC missing -> skip not fail
    Tool: Bash (pytest)
    Steps: run on env without CBC
    Expected: tests SKIP with reason; availability() reports cbc=False
    Evidence: .omo/evidence/task-23-skip.txt
  ```
  **Commit**: YES — `feat(solver): HiGHS + CBC kernels with graceful trajectory degradation`

- [x] 24. GCG kernel + DW/Benders decomposability detection

  **What to do**: Add `src/opop/solver/gcg.py` (branch-price-and-cut via GCG) and `src/opop/analyzer/decompose.py` to detect block-diagonal/staircase structure (from the bipartite graph) and propose DW/Benders decomposition; expose a decomposition `Delta` (class C/D as appropriate, certified by task 11 where it changes the model).

  **Must NOT do**: force decomposition on non-decomposable instances; treat a decomposition that changes the feasible set as uncertified.

  **Recommended Agent Profile**: Category `deep` (structural analysis + specialized solver). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 31 · Blocked By: 12.

  **References**: External GCG automatic DW/Benders; PySCIPOpt Benders/Pricer plugins (API research). Pattern: `analyzer/` (task 10), `model_graph` (task 9). WHY: structure detection drives when decomposition helps.

  **Acceptance Criteria**:
  - [ ] `pytest tests/analyzer/test_decompose.py` → PASS (detects block structure on a constructed block-diagonal instance; reports NONE on a dense one).
  - [ ] GCG solves a decomposable fixture to the known optimum.

  **QA Scenarios**:
  ```
  Scenario: Block structure detected and exploited
    Tool: Bash (pytest)
    Steps: build a 3-block instance; run detector; solve via GCG
    Expected: detector returns DW-amenable with 3 blocks; GCG optimum == reference
    Evidence: .omo/evidence/task-24-gcg.txt
  Scenario: Dense instance -> no false decomposition
    Tool: Bash (pytest)
    Steps: run detector on a dense instance
    Expected: decomposability=NONE (no spurious blocks)
    Evidence: .omo/evidence/task-24-dense.txt
  ```
  **Commit**: YES — `feat(solver/analyzer): GCG kernel + DW/Benders decomposability detection`

- [x] 25. Heuristic cores: LNS / RINS / local-branching / repair

  **What to do**: `src/opop/solver/heuristics.py`: implement local-branching (k-flip neighborhood constraint), RINS (fix vars agreeing between LP relaxation and incumbent), generic LNS (destroy/repair), and a feasibility-repair routine; expose as schedulable "heuristic cores" the controller can select (`Phi.h`). Each operates on the IR + an incumbent, returns an improved incumbent + trace.

  **Must NOT do**: implement problem-specific heuristics here (Wave 5/6); allow a heuristic to return an infeasible "solution" unchecked (verify feasibility).

  **Recommended Agent Profile**: Category `deep` (matheuristic algorithms). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 38 · Blocked By: 12.

  **References**: External local-branching (Fischetti-Lodi), RINS (Danna et al.), LNS; report's heuristic-core section. Pattern: `SolverKernel` (task 12). WHY: standard, well-documented matheuristics; reused as baselines (task 38).

  **Acceptance Criteria**:
  - [ ] `pytest tests/solver/test_heuristics.py` → PASS (LNS improves a deliberately-suboptimal incumbent; repair fixes a near-feasible point; all outputs feasibility-checked).

  **QA Scenarios**:
  ```
  Scenario: LNS improves a suboptimal incumbent
    Tool: Bash (pytest)
    Steps: seed a known-suboptimal incumbent; run LNS for a small budget
    Expected: objective improves or holds; result certified feasible
    Evidence: .omo/evidence/task-25-lns.txt
  Scenario: Repair never returns an infeasible solution
    Tool: Bash (pytest)
    Steps: feed an infeasible assignment; run repair
    Expected: returns feasible solution OR reports failure; never an unchecked infeasible result
    Evidence: .omo/evidence/task-25-repair.txt
  ```
  **Commit**: YES — `feat(solver): LNS/RINS/local-branching/repair heuristic cores`

- [x] 26. Analyzer expansion: Lagrangian, symmetry/dominance, Benders/DW readiness

  **What to do**: Extend `analyzer/` with Lagrangian-relaxation bound estimation (dualize coupling constraints), symmetry detection (orbit/automorphism heuristics or via graph hashing) and dominance relations, plus a Benders/DW readiness classifier feeding task 24. Enrich `AnalysisReport` with these signals (consumed by proposer task 27 and controller task 28).

  **Must NOT do**: emit symmetry-breaking constraints without certifying via task 11; claim Lagrangian bounds tighter than LP without verification.

  **Recommended Agent Profile**: Category `deep` (OR theory + heuristics). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 27 · Blocked By: 10.

  **References**: External Lagrangian relaxation, orbital branching/symmetry (Margot); report's Analyzer section. Pattern: `analyzer/` (task 10). WHY: these analyses become BO search dimensions.

  **Acceptance Criteria**:
  - [ ] `pytest tests/analyzer/test_expansion.py` → PASS (Lagrangian bound ≥ LP bound on a fixture; symmetry detected on a symmetric instance; none on an asymmetric one).

  **QA Scenarios**:
  ```
  Scenario: Lagrangian bound is valid and >= LP on a fixture
    Tool: Bash (pytest)
    Steps: dualize coupling constraints; compute bound; compare to LP relaxation
    Expected: Lagrangian bound is a valid dual bound, >= LP within tolerance
    Evidence: .omo/evidence/task-26-lagrangian.txt
  Scenario: Symmetry detector has no false positives
    Tool: Bash (pytest)
    Steps: run on an intentionally asymmetric instance
    Expected: reports no symmetry group (no spurious orbits)
    Evidence: .omo/evidence/task-26-symmetry.txt
  ```
  **Commit**: YES — `feat(analyzer): Lagrangian bounds, symmetry/dominance, Benders/DW readiness`

- [x] 27. Proposer expansion: formulation families + staged search spaces S0–S4

  **What to do**: Extend the proposer to emit formulation-family deltas (routing: MTZ / single- & multi-commodity flow / set-partition+pricing; scheduling: time-indexed / disjunctive / arc-flow; lot-sizing: big/small-bucket) and define the staged search spaces **S0 params → S1 +safe cuts → S2 +heuristics → S3 +formulation/decomposition → S4 +multi-kernel/MF/transfer**. Each formulation delta is class-A/B and MUST pass task 11.

  **Must NOT do**: let an unverified reformulation reach evaluation; collapse all stages into one (staging is required for credit assignment).

  **Recommended Agent Profile**: Category `deep` (formulation engineering + search-space design). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 39 · Blocked By: 14,26.

  **References**: Doc: report's formulation-family encoding + Metis staged-spaces S0–S4. Pattern: `proposer/` (task 14), `verify/gate.py` (task 11). WHY: structure-first proposals; staging enables ablation.

  **Acceptance Criteria**:
  - [ ] `pytest tests/proposer/test_families.py` → PASS (MTZ↔flow reformulation of a tiny TSP is certified-equivalent by task 11; stage filters restrict the emitted delta classes correctly).

  **QA Scenarios**:
  ```
  Scenario: Reformulation is certified equivalent
    Tool: Bash (pytest)
    Steps: propose MTZ->multi-commodity-flow on a 5-node TSP; run verify gate
    Expected: gate status=pass (class A/B), same optimum on both formulations
    Evidence: .omo/evidence/task-27-reformulate.txt
  Scenario: Stage S1 cannot emit a formulation delta
    Tool: Bash (pytest)
    Steps: request proposals under stage=S1
    Expected: only param + safe-cut deltas; zero formulation/decomposition deltas
    Evidence: .omo/evidence/task-27-staging.txt
  ```
  **Commit**: YES — `feat(proposer): formulation families + staged search spaces S0-S4`

- [x] 28. Controller ladder: SMAC/TPE/RF + structured BO selection

  **What to do**: Add controller rungs behind the `Surrogate`/`Acquisition` Protocol: SMAC3 (ask-tell, censored-aware), a TPE/RF backend, and structured-BO surrogates selected by space shape — BOCS (binary/boolean), COMBO (pure discrete), CoCaBO/HyBO (mixed), dictionary-embedding (high-dim discrete), BoTorch `qLogNoisyExpectedImprovement` (default), `qKnowledgeGradient` (batch). A `select_surrogate(phi_space, budget, noise)` router picks the rung.

  **Must NOT do**: jump to structured BO before evidence (router defaults to random→SMAC/TPE→qLogNEI; structured only when space shape + budget justify); forget `overwrite=True` for SMAC (warmstart hazard); use non-Log EI variants.

  **Recommended Agent Profile**: Category `deep` (BO methods + routing). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 29,32,36 · Blocked By: 8,15.

  **References**: External (API research) BoTorch v0.17.2 `qLogNoisyExpectedImprovement`(needs X_baseline)/`qKnowledgeGradient`, `MixedSingleTaskGP(cat_dims)`, `optimize_acqf_mixed`; SMAC3 v2.2.0 ask-tell + `TrialValue(status=TIMEOUT)` + `overwrite=True`; BOCS/COMBO/CoCaBO/HyBO. WHY: exact APIs + censoring + the Log-variant requirement.

  **Acceptance Criteria**:
  - [ ] `pytest tests/controller/test_ladder.py` → PASS (each rung runs ask-tell on a toy objective; SMAC handles a censored tell; router picks expected rung for binary vs mixed vs high-dim spaces).

  **QA Scenarios**:
  ```
  Scenario: Each rung + router behaves
    Tool: Bash (pytest)
    Steps: run SMAC/TPE/qLogNEI on a toy; route a binary, a mixed, a high-dim space
    Expected: all converge >= random; router selects BOCS(binary)/mixed-GP(mixed)/dict-embed(high-dim)
    Evidence: .omo/evidence/task-28-ladder.txt
  Scenario: SMAC censored tell + no accidental warmstart
    Tool: Bash (pytest)
    Steps: tell a TIMEOUT trial; re-init with overwrite=True
    Expected: censored cost recorded; no stale warmstart from prior run dir
    Evidence: .omo/evidence/task-28-smac-censored.txt
  ```
  **Commit**: YES — `feat(controller): SMAC/TPE/RF + structured-BO ladder with space-shape router`

- [x] 29. Multi-fidelity + fidelity-correlation GATE + cost-aware MFKG

  **What to do**: Implement fidelity layers (presolve-only / LP-relax-only / root-cuts-only / short-time / sub-instance / heuristic-core / full-solve) as a `Phi.s` dimension; run a **fidelity-correlation study** (Spearman ρ between low- and high-fidelity rankings on the dev set). ONLY enable cost-aware multi-fidelity KG (`qMultiFidelityKnowledgeGradient` + `InverseCostWeightedUtility` + `AffineFidelityCostModel`) if ρ ≥ 0.5; else record the negative result and keep single-fidelity.

  **Must NOT do**: enable multi-fidelity BO without passing the ρ ≥ 0.5 gate; assume low/high fidelity correlate.

  **Recommended Agent Profile**: Category `deep` (multi-fidelity BO + empirical gating). Skills: none.

  **Parallelization**: Parallel: YES · Wave 4 · Blocks: 39 · Blocked By: 13,28.

  **References**: External (API research) `qMultiFidelityKnowledgeGradient`, `AffineFidelityCostModel`, `InverseCostWeightedUtility`, `project_to_target_fidelity`. Doc: Metis fidelity-correlation gate. WHY: exact cost-aware MFKG wiring + the empirical guardrail.

  **Acceptance Criteria**:
  - [ ] `pytest tests/controller/test_mf.py` → PASS (correlation study computes ρ; MFKG path enabled only when ρ≥0.5 on a synthetic correlated pair, disabled on an uncorrelated pair).
  - [ ] `python -m opop.eval.fidelity_correlation ...` emits `fidelity_correlation.json` with `spearman_rho`, `usable_for_ranking`.

  **QA Scenarios**:
  ```
  Scenario: Gate enables MF only when correlated
    Tool: Bash
    Steps: run correlation study on a correlated pair (rho~0.8) and an uncorrelated pair (rho~0.0)
    Expected: usable_for_ranking=true then false; MFKG controller activates only in the first case
    Evidence: .omo/evidence/task-29-mfgate.txt
  Scenario: cost-aware MFKG prefers cheap fidelity early
    Tool: Bash (pytest)
    Steps: run MFKG with a cost model where low fidelity is 10x cheaper
    Expected: early evaluations skew to low fidelity; high fidelity reserved for promising candidates
    Evidence: .omo/evidence/task-29-mfkg.txt
  ```
  **Commit**: YES — `feat(controller): fidelity layers + correlation gate + cost-aware MFKG`

> **Wave 5 entry criterion**: Wave-4 core (22, 27, 28) complete and tested. Generality is proven by running the SAME interfaces/pipeline on new problem classes with problem-specific code isolated behind declared adapters (Thesis T3).

- [x] 30. MIQP / MIQCP / QUBO adapters (declared plugin interface)

  **What to do**: Extend the IR to carry quadratic objective/constraint terms behind a `QuadraticExtension`; add `ProblemClassAdapter` plugins for MIQP, MIQCP, and QUBO (incl. QUBO↔Ising and QUBO↔MILP linearization paths). Route QUBO/MIQP to capable kernels (SCIP for MIQCP; CP-SAT/linearization for QUBO). Adapters declare their capabilities; core stays class-agnostic.

  **Must NOT do**: leak problem-specific logic into core modules (must live behind the adapter); claim MILP-only kernels solve quadratics (route or linearize explicitly).

  **Recommended Agent Profile**: Category `deep` (model extension + class routing). Skills: none.

  **Parallelization**: Parallel: YES · Wave 5 · Blocks: 31,35 · Blocked By: 9,22.

  **References**: External QPLIB (instances/format), SCIP MIQCP support, QUBO/Ising linearization. Type: `model/ir.py` (task 9), `ProblemClassAdapter`. WHY: T3 generality requires class-agnostic core + declared adapters.

  **Acceptance Criteria**:
  - [ ] `pytest tests/model/test_quadratic.py` → PASS (QUBO↔MILP linearization preserves optimum on a small Max-Cut; MIQCP fixture solved to known optimum; core modules import without class-specific branches).

  **QA Scenarios**:
  ```
  Scenario: QUBO linearization preserves optimum
    Tool: Bash (pytest)
    Steps: take a 6-node Max-Cut QUBO; linearize to MILP; solve both
    Expected: same optimal cut value; adapter declares capability used
    Evidence: .omo/evidence/task-30-qubo.txt
  Scenario: Core has no class-specific branching
    Tool: Bash (grep/pytest)
    Steps: assert orchestrator/controller modules contain no 'if problem_class==' branches
    Expected: zero such branches (all behind adapters)
    Evidence: .omo/evidence/task-30-agnostic.txt
  ```
  **Commit**: YES — `feat(model): MIQP/MIQCP/QUBO adapters behind declared plugin interface`

- [x] 31. Structured MINLP adapter (decomposable subset)

  **What to do**: Add a MINLP adapter for the structured/decomposable subset the report scopes (e.g. separable or factorable constraints amenable to outer-approximation / decomposition); integrate with GCG/Benders readiness (task 24). Clearly bound the supported subset; reject out-of-subset MINLPs with a typed error.

  **Must NOT do**: attempt general nonconvex MINLP (out of scope); silently mishandle unsupported nonlinearities.

  **Recommended Agent Profile**: Category `deep` (nonlinear structure + decomposition). Skills: none.

  **Parallelization**: Parallel: YES · Wave 5 · Blocks: 39 · Blocked By: 24,30.

  **References**: External outer-approximation (Duran-Grossmann), structured MINLP; report's "structured MINLP" scope. Pattern: `analyzer/decompose.py` (task 24). WHY: bound generality to a tractable, decomposable subset.

  **Acceptance Criteria**:
  - [ ] `pytest tests/model/test_minlp.py` → PASS (a separable MINLP fixture solved via outer-approximation to known optimum; an unsupported nonconvex instance rejected with a clear error).

  **QA Scenarios**:
  ```
  Scenario: Supported structured MINLP solved
    Tool: Bash (pytest)
    Steps: solve a separable MINLP fixture via the adapter
    Expected: optimum within tolerance of reference
    Evidence: .omo/evidence/task-31-minlp.txt
  Scenario: Out-of-subset MINLP rejected
    Tool: Bash (pytest)
    Steps: feed a general nonconvex MINLP
    Expected: UnsupportedModelError naming the offending term (no silent wrong answer)
    Evidence: .omo/evidence/task-31-reject.txt
  ```
  **Commit**: YES — `feat(model): structured MINLP adapter (decomposable subset) + scope guard`

- [x] 32. Historical-transfer priors / cross-distribution warm start

  **What to do**: Persist per-distribution posteriors and instance descriptors; implement transfer priors (warm-start the controller from related historical tasks, e.g. via the Reptile/MAML meta-tuner) keyed by instance descriptors (sparsity, time-horizon, block structure, integer density). Provide a `transfer_off` switch for ablation.

  **Must NOT do**: warm-start from test/ood posteriors (leakage); let transfer silently overfit a single source distribution (must be ablatable).

  **Recommended Agent Profile**: Category `deep` (meta-learning/transfer). Skills: none.

  **Parallelization**: Parallel: YES · Wave 5 · Blocks: 39 · Blocked By: 28.

  **References**: Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/DAC/track_e/src/meta_tuner.py` (Reptile/MAML). Doc: report's hierarchical/transfer priors. WHY: framework-level transfer is a stated capability + a T2 lever.

  **Acceptance Criteria**:
  - [ ] `pytest tests/controller/test_transfer.py` → PASS (transfer warm start reduces trials-to-target vs cold start on a synthetic related-task pair; `transfer_off` reproduces cold-start; never reads test/ood posteriors).

  **QA Scenarios**:
  ```
  Scenario: Transfer accelerates vs cold start
    Tool: Bash (pytest)
    Steps: pretrain on source tasks; warm-start a related target; compare trials-to-target to cold start
    Expected: fewer trials-to-target with transfer; transfer_off matches cold start
    Evidence: .omo/evidence/task-32-transfer.txt
  Scenario: Transfer refuses test/ood sources
    Tool: Bash (pytest)
    Steps: attempt warm start from a posterior tagged test/ood
    Expected: raises LeakageError; refuses
    Evidence: .omo/evidence/task-32-leak.txt
  ```
  **Commit**: YES — `feat(controller): historical-transfer priors with ablation switch + leakage guard`

> **Wave 6 entry criterion**: thesis instruments exist and are tested — comparison report (18), ablation staging (27), leakage audit + cost accounting (19), multi-fidelity gate (29). Every benchmark family/baseline added here MUST carry a phase+thesis tag in the registry.

- [x] 33. MILP benchmark acquisition (MIPLIB 2017 / Distributional MIPLIB / MILPBench)

  **What to do**: Add downloaders/loaders + registry entries for MIPLIB 2017, Distributional MIPLIB, and MILPBench; assign immutable dev/validation/test/ood_test splits with `leakage_group`s (group by instance family/generator to prevent near-duplicate leakage across splits). Seal split manifests.

  **Must NOT do**: place near-duplicate instances in different splits; tune on test/ood; commit raw blobs (scripts + checksums only).

  **Recommended Agent Profile**: Category `unspecified-high` (large-scale data plumbing + split integrity). Skills: none.

  **Parallelization**: Parallel: YES · Wave 6 · Blocks: 39 · Blocked By: 7,20.

  **References**: External MIPLIB 2017, Distributional MIPLIB, MILPBench. Type: registry (task 7). WHY: the cross-distribution backbone for T1.

  **Acceptance Criteria**:
  - [ ] `pytest tests/bench/test_milp_suites.py` → PASS (checksums verified; `assert_no_overlap` + leakage-group spanning check pass; splits sealed).

  **QA Scenarios**:
  ```
  Scenario: Distributional splits are leakage-safe
    Tool: Bash
    Steps: build splits; run assert_no_overlap + leakage-group span check
    Expected: 0 overlaps; no family spans dev and test
    Evidence: .omo/evidence/task-33-splits.txt
  Scenario: Checksum mismatch is caught
    Tool: Bash (pytest)
    Steps: corrupt one downloaded instance; reload
    Expected: ChecksumError naming the instance
    Evidence: .omo/evidence/task-33-checksum.txt
  ```
  **Commit**: YES — `feat(bench): MIPLIB2017/Distributional/MILPBench acquisition + sealed leakage-safe splits`

- [x] 34. Classic CO benchmarks (TSPLIB / CVRPLIB / OR-Library / JSPLIB / MaxSAT / MaxCut)

  **What to do**: Add loaders + registry entries for the classic CO families, each with a `ProblemClassAdapter` mapping to IR (and to MILP/QUBO formulations where relevant). Tag each with problem_type, time_limit, baseline_set, splits.

  **Must NOT do**: hand-tune formulations per instance (formulations come from the proposer); add a family without phase+thesis tags.

  **Recommended Agent Profile**: Category `unspecified-high` (many formats + adapters). Skills: none.

  **Parallelization**: Parallel: YES · Wave 6 · Blocks: 39 · Blocked By: 7.

  **References**: External TSPLIB/CVRPLIB/OR-Library/JSPLIB/MaxSAT Evaluations/MaxCut-Bench formats. Pattern: `ProblemClassAdapter` (task 30). WHY: breadth for T3 generality.

  **Acceptance Criteria**:
  - [ ] `pytest tests/bench/test_classic_co.py` → PASS (each loader parses ≥2 instances; adapter produces a valid IR; a tiny TSPLIB instance solves to known optimum).

  **QA Scenarios**:
  ```
  Scenario: Each family loads and maps to a valid IR
    Tool: Bash (pytest)
    Steps: load 2 instances per family; build IR; validate
    Expected: all parse; IR validates; tiny TSP optimum matches reference
    Evidence: .omo/evidence/task-34-classic.txt
  Scenario: Malformed instance file rejected
    Tool: Bash (pytest)
    Steps: feed a truncated TSPLIB file
    Expected: ParseError with file + line context (no partial silent load)
    Evidence: .omo/evidence/task-34-parse.txt
  ```
  **Commit**: YES — `feat(bench): classic CO loaders + adapters (TSP/CVRP/OR-Lib/JSP/MaxSAT/MaxCut)`

- [x] 35. QPLIB + modeling-agent sets + solver-backed re-verification/cleaning

  **What to do**: Add QPLIB (for MIQP/MIQCP) and modeling-agent datasets (NL4Opt, NLP4LP, MAMO, IndustryOR, StructuredOR, ORQA, WIQOR, OptiBench, CO-Bench). For modeling-agent sets, run **solver-backed re-verification** (build the labeled model, solve, compare to the provided label) and quarantine mismatching items into a `cleaned/` view with a report (OptiTrust-style integrity check). If re-verification across all 9 datasets balloons, split this into per-dataset sub-tasks.

  **Must NOT do**: trust modeling-benchmark labels blindly; include flagged-bad items in headline accuracy without disclosure.

  **Recommended Agent Profile**: Category `deep` (data integrity + heterogeneous formats). Skills: none.

  **Parallelization**: Parallel: YES · Wave 6 · Blocks: 39 · Blocked By: 7,30.

  **References**: External QPLIB; NL4Opt/NLP4LP/MAMO/IndustryOR/StructuredOR/ORQA/WIQOR/OptiBench/CO-Bench; OptiTrust (label-integrity). WHY: report + Metis both require solver-backed verification of modeling benchmarks.

  **Acceptance Criteria**:
  - [ ] `pytest tests/bench/test_cleaning.py` → PASS (re-verification flags a planted wrong label; emits `cleaning_report.json` with quarantined ids; clean items pass).

  **QA Scenarios**:
  ```
  Scenario: Wrong label is quarantined
    Tool: Bash
    Steps: inject an item whose labeled optimum is wrong; run re-verification
    Expected: item quarantined; cleaning_report lists it with computed vs labeled value
    Evidence: .omo/evidence/task-35-clean.txt
  Scenario: QPLIB MIQCP loads + solves
    Tool: Bash (pytest)
    Steps: load a small QPLIB instance; solve via MIQCP adapter
    Expected: optimum within tolerance of reference
    Evidence: .omo/evidence/task-35-qplib.txt
  ```
  **Commit**: YES — `feat(bench): QPLIB + modeling-agent sets + solver-backed cleaning`

- [x] 36. Baselines 1–2: static-expert+default solver; static+solver-tuning (SMAC/OAT)

  **What to do**: Implement two baseline runners sharing the experiment harness + cost accounting: (1) a fixed expert formulation solved with each solver's defaults; (2) the same formulation with automated solver-parameter tuning (SMAC3; optionally OAT-style). Both emit identical `results.parquet` schema for fair comparison.

  **Must NOT do**: give baselines less budget/seeds than opop; tune baselines on test/ood; use a different metric pipeline.

  **Recommended Agent Profile**: Category `unspecified-high` (baseline harness + fairness). Skills: none.

  **Parallelization**: Parallel: YES · Wave 6 · Blocks: 39 · Blocked By: 21,28.

  **References**: External SMAC3 ask-tell (task 28). Doc: Metis baseline-fairness. WHY: T1/T4 require apples-to-apples baselines.

  **Acceptance Criteria**:
  - [ ] `pytest tests/experiments/test_baselines_12.py` → PASS (both runners produce schema-identical results; equal budget/seed enforced; SMAC tuning improves over default on a fixture).

  **QA Scenarios**:
  ```
  Scenario: Baselines share schema + budget with opop
    Tool: Bash (pytest)
    Steps: run default + SMAC-tuned on a fixture with the same budget/seeds as an opop run
    Expected: identical result schema; equal trial budget; cost columns present
    Evidence: .omo/evidence/task-36-baselines.txt
  Scenario: Fairness guard trips on unequal budget
    Tool: Bash (pytest)
    Steps: configure a baseline with fewer seeds than opop
    Expected: harness raises a FairnessError before running
    Evidence: .omo/evidence/task-36-fairness.txt
  ```
  **Commit**: YES — `feat(experiments): baselines 1-2 (default + SMAC/OAT tuning) with fairness guard`

- [x] 37. Baselines 3–4: params-only search; LLM-modeling-agent-only

  **What to do**: (3) opop with the controller restricted to S0 (params-only) — the key novelty ablation (isolates analyzer/formulation contribution for T4). (4) An LLM-modeling-agent-only baseline reproducing OptiMUS/LLMOPT/ORLM/OR-R1-style NL→model→solve WITHOUT the analyzer/verify/BO loop, via the LLM adapter. Both share harness + cost accounting.

  **Must NOT do**: let baseline 4 use the verification gate or BO loop (that's opop, not the baseline); under-resource baseline 4's prompting vs opop's LLM calls.

  **Recommended Agent Profile**: Category `deep` (faithful baseline reproductions). Skills: none.

  **Parallelization**: Parallel: YES · Wave 6 · Blocks: 39,45 · Blocked By: 21.

  **References**: External OptiMUS/LLMOPT/ORLM/OR-R1 (modeling agents). Pattern: `llm/client.py` (task 2). WHY: T4 novelty is only measurable against modeling-agent-only + params-only.

  **Acceptance Criteria**:
  - [ ] `pytest tests/experiments/test_baselines_34.py` → PASS (params-only controller never emits cut/formulation deltas; modeling-agent baseline produces a solved model end-to-end with FakeLLM; schema-identical results).

  **QA Scenarios**:
  ```
  Scenario: Params-only ablation is truly params-only
    Tool: Bash (pytest)
    Steps: run baseline 3; inspect emitted delta classes
    Expected: only parameter deltas; zero cut/formulation/decomposition deltas
    Evidence: .omo/evidence/task-37-paramsonly.txt
  Scenario: Modeling-agent baseline excludes the opop loop
    Tool: Bash (pytest)
    Steps: run baseline 4; assert no analyzer/verify/BO calls in the trace
    Expected: NL->model->solve path only; no gate/controller invocations
    Evidence: .omo/evidence/task-37-agentonly.txt
  ```
  **Commit**: YES — `feat(experiments): baselines 3-4 (params-only ablation + modeling-agent-only)`

- [x] 38. Baselines 5–6: classic matheuristics; LLM-enhanced CO

  **What to do**: (5) Classic matheuristic baselines (local-branching / RINS / LNS) using the heuristic cores from task 25, run standalone (no BO/analyzer). (6) LLM-enhanced CO baselines reproducing LLM-LNS / HeurAgenix-style heuristic selection/evolution via the LLM adapter. Share harness + cost accounting.

  **Must NOT do**: blend opop components into these baselines; report solver-only time for the LLM-enhanced baseline (include LLM cost).

  **Recommended Agent Profile**: Category `deep` (heuristic baselines + faithful LLM-enhanced reproductions). Skills: none.

  **Parallelization**: Parallel: YES · Wave 6 · Blocks: 39 · Blocked By: 25.

  **References**: External local-branching/RINS/LNS; LLM-LNS, HeurAgenix. Pattern: `solver/heuristics.py` (task 25). WHY: situates opop against both classic and LLM-enhanced CO.

  **Acceptance Criteria**:
  - [ ] `pytest tests/experiments/test_baselines_56.py` → PASS (matheuristic baseline improves an incumbent; LLM-enhanced baseline selects among heuristics with FakeLLM; both schema-identical incl. LLM cost columns).

  **QA Scenarios**:
  ```
  Scenario: Matheuristic + LLM-enhanced baselines run with full cost accounting
    Tool: Bash (pytest)
    Steps: run baselines 5 and 6 on a fixture
    Expected: schema-identical results; LLM-enhanced rows include llm_tokens/cost; matheuristic rows have zero LLM cost
    Evidence: .omo/evidence/task-38-baselines56.txt
  Scenario: No opop-loop contamination
    Tool: Bash (pytest)
    Steps: assert baselines 5/6 traces contain no verify-gate/BO-controller calls
    Expected: none present
    Evidence: .omo/evidence/task-38-clean.txt
  ```
  **Commit**: YES — `feat(experiments): baselines 5-6 (matheuristics + LLM-enhanced CO)`

- [x] 39. Full experiment matrix + ablations (job-runner; ≥5 seeds; multi-time-limit)

  **What to do**: Implement the pluggable job-runner (`local | slurm | qz` adapters; reuse `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/slurm/*.sh` patterns) and a matrix driver that sweeps {benchmark family × method (opop + 6 baselines) × ablation row (scip-default, params-only, analyzer-cuts-only, params+cuts, full-opop; staged S0–S4) × seed(≥5) × time-limit(e.g. 30/300/1800s)}. Emit per-cell artifacts + a consolidated `results.parquet`. Resume-safe; records repro manifests + cost accounting per cell.

  **Must NOT do**: run on test/ood during method development; submit without a sealed split + passing leakage audit; drop censored cells.

  **Recommended Agent Profile**: Category `unspecified-high` (orchestration at scale; cluster submission). Skills: none.

  **Parallelization**: Parallel: NO (Wave-6 integrator) · Wave 6 · Blocks: 40 · Blocked By: 19,29,31,32,33,34,35,36,37,38.

  **References**: Pattern: `/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/research/intern/slurm/{submit_matrix.sh,run_experiment.sh}` (dependency-aware + Docker). Doc: ablation rows + staged spaces (Metis). WHY: reuse cluster submission; ablations enable credit assignment.

  **Acceptance Criteria**:
  - [ ] `pytest tests/experiments/test_matrix.py` → PASS (matrix expansion correct; resume skips completed cells; local-runner dry-run lists expected jobs; leakage audit gate blocks test-set runs during dev).
  - [ ] A small real local sweep produces a consolidated `results.parquet` with all ablation rows.

  **QA Scenarios**:
  ```
  Scenario: Matrix runs locally, resume-safe, ablation rows present
    Tool: Bash (interactive_bash)
    Steps: run a tiny matrix (2 instances × {full-opop, params-only, scip-default} × 5 seeds × 30s) locally; interrupt; resume
    Expected: resume skips done cells; final results.parquet has all 3 ablation rows × 5 seeds; repro + cost columns present
    Evidence: .omo/evidence/task-39-matrix.txt
  Scenario: Submission blocked without sealed splits / clean leakage
    Tool: Bash
    Steps: attempt a test-split run with an unsealed manifest
    Expected: aborts citing unsealed split / leakage audit not passed
    Evidence: .omo/evidence/task-39-guard.txt
  ```
  **Commit**: YES — `feat(experiments): pluggable job-runner + full ablation/seed/time matrix`

- [x] 40. Cross-distribution + LOFO + scale-extrapolation eval + T1–T4 analysis

  **What to do**: Run the FINAL one-shot evaluation on frozen test + ood_test: cross-distribution transfer, leave-one-family-out, and scale extrapolation. Compute the falsifiable thesis verdicts: **T1** (≥10% median primal-integral vs SCIP-default & ≥5% vs params-only at equal end-to-end budget on held-out), **T2** (≥30% fewer full-solve evals to baseline-best incl. overhead), **T3** (same interfaces/pipeline run unchanged across MILP+QUBO+MIQP with problem-specific code behind adapters), **T4** (analyzer-certified deltas significant beyond params-only-BO AND modeling-agent-only). Emit `thesis_report.json` + figures/tables.

  **Must NOT do**: touch test/ood before this step; claim a thesis met without significance + min-effect; hide negative results (report them).

  **Recommended Agent Profile**: Category `deep` (statistical analysis + scientific judgment). Skills: none.

  **Parallelization**: Parallel: NO (final analysis integrator) · Wave 6 · Blocks: 41,42,43,44 · Blocked By: 39.

  **References**: Doc: falsifiable T1–T4 (Work Objectives) + Win Definition. Pattern: `eval/compare.py` (task 18). WHY: this task produces the paper's central claims.

  **Acceptance Criteria**:
  - [ ] `python -m opop.eval.theses --results runs/final_eval --out thesis_report.json` → emits per-thesis {claim, metric, baseline, significant, effect, verdict}.
  - [ ] `pytest tests/experiments/test_theses.py` → PASS (verdict logic correct on fixtures; one-shot test-set guard enforced; negative result reported, not suppressed).

  **QA Scenarios**:
  ```
  Scenario: Thesis verdicts computed with significance + effect gating
    Tool: Bash
    Steps: run on a fixture where T1 holds (12% PI) and T2 fails (only 10% fewer evals)
    Expected: thesis_report marks T1 met (significant, clears effect), T2 not-met; both reported honestly
    Evidence: .omo/evidence/task-40-theses.txt
  Scenario: Test set is one-shot
    Tool: Bash (pytest)
    Steps: attempt a second tuning pass on test split
    Expected: blocked by one-shot guard + leakage audit
    Evidence: .omo/evidence/task-40-oneshot.txt
  ```
  **Commit**: YES — `feat(eval): cross-dist/LOFO/scale eval + falsifiable T1-T4 thesis report`

> **Wave 7 entry criterion**: Wave-6 results frozen (task 40 `thesis_report.json` produced). Deliverables present the proven results; they do not change methods or re-run experiments.

- [x] 41. OSS library packaging + public API + docs + examples

  **What to do**: Promote `pyproject.toml` to a buildable package (metadata, entry points, version, license), define a clean public API (`opop.run`, `opop.replay`, kernels, controller ladder, registry), write `docs/` (architecture + API + how-to add a solver/problem-class/surrogate), and ship runnable examples (Phase-1 smoke + one expansion). Add a CONTRIBUTING + AGENTS.md.

  **Must NOT do**: expose internal mutable state in the public API; bundle benchmark blobs; require Gurobi anywhere in the public path.

  **Recommended Agent Profile**: Category `unspecified-high` (packaging + API + docs). Skills: none.

  **Parallelization**: Parallel: YES · Wave 7 · Blocks: 44 · Blocked By: 21,40.

  **References**: Pattern: lab `pyproject.toml` + AGENTS.md conventions. WHY: a reusable, documented release is a stated deliverable.

  **Acceptance Criteria**:
  - [ ] `pip install .` in a clean venv succeeds; `python -c "import opop; opop.run"` works; example scripts run to completion.
  - [ ] `pytest --doctest-modules` (public API docstrings) → PASS.

  **QA Scenarios**:
  ```
  Scenario: Clean install + example runs
    Tool: Bash (interactive_bash)
    Steps: fresh venv; pip install .; run examples/phase1_smoke.py
    Expected: install succeeds; example produces artifacts; public API importable
    Evidence: .omo/evidence/task-41-install.txt
  Scenario: No Gurobi in public path
    Tool: Bash (grep)
    Steps: grep public modules for gurobipy/mip-learn imports
    Expected: zero matches
    Evidence: .omo/evidence/task-41-nogurobi.txt
  ```
  **Commit**: YES — `feat(release): package opop, public API, docs, examples`

- [x] 42. Leaderboard (results site/table + submission protocol)

  **What to do**: Build a static leaderboard from `results.parquet`/`thesis_report.json` (per-benchmark anytime + solved-rate + end-to-end-time, with seeds/CIs and a clear methodology/limitations page), plus a documented submission protocol (registry entry + sealed splits + leakage audit + repro manifest required). Static HTML (and a markdown fallback).

  **Must NOT do**: publish dev/validation numbers as headline; omit cost/time or the limitations note; allow submissions without leakage audit + manifest.

  **Recommended Agent Profile**: Category `visual-engineering` (presentation UI). Skills: `frontend-ui-ux` (clean comparative tables/plots), `playwright` (verify the rendered page).

  **Parallelization**: Parallel: YES · Wave 7 · Blocks: none · Blocked By: 40.

  **References**: Data: `results.parquet`, `thesis_report.json` (task 40). WHY: leaderboard is a stated deliverable; must encode integrity rules.

  **Acceptance Criteria**:
  - [ ] `python -m opop.leaderboard build --results runs/final_eval --out site/` produces a page showing test/ood results with CIs + limitations.
  - [ ] Playwright check: required columns + methodology/limitations section present in the DOM.

  **QA Scenarios**:
  ```
  Scenario: Leaderboard renders test/ood results with integrity notes
    Tool: Playwright
    Steps: build site; open index; assert table has method/metric/CI columns + a visible 'Limitations' and 'Splits/leakage policy' section
    Expected: all present; no dev/validation rows in the headline table
    Evidence: .omo/evidence/task-42-leaderboard.png
  Scenario: Submission without manifest/audit rejected
    Tool: Bash
    Steps: submit a result lacking repro_manifest/leakage_audit
    Expected: submission validator rejects with the missing-artifact reason
    Evidence: .omo/evidence/task-42-submit.txt
  ```
  **Commit**: YES — `feat(leaderboard): static results site + integrity-gated submission protocol`

- [x] 43. Tech report

  **What to do**: Write the internal technical report: architecture (5 layers + verification gate), methodology (staged spaces, controller ladder, multi-fidelity gate), full results incl. ablations + negative results, reproducibility appendix (manifests, seeds, versions). Markdown in `docs/tech-report/`, with figures generated from the result artifacts.

  **Must NOT do**: include unverified claims; cite dev/validation as final; hand-draw numbers (generate from artifacts).

  **Recommended Agent Profile**: Category `writing` (technical prose). Skills: none.

  **Parallelization**: Parallel: YES · Wave 7 · Blocks: none · Blocked By: 40.

  **References**: Data: `thesis_report.json`, comparison reports, ablation results. WHY: deliverable; the honest, complete record.

  **Acceptance Criteria**:
  - [ ] All figures/tables regenerate from artifacts via a single `make report` (or script); no orphaned/hardcoded numbers.
  - [ ] A link-check + numbers-trace check passes (every reported number maps to an artifact field).

  **QA Scenarios**:
  ```
  Scenario: Report numbers trace to artifacts
    Tool: Bash
    Steps: run the report build; run the numbers-trace checker
    Expected: every headline number resolves to a results/thesis artifact field; build is reproducible
    Evidence: .omo/evidence/task-43-report.txt
  Scenario: Stale figure detected
    Tool: Bash
    Steps: change a result; rebuild
    Expected: figure updates; checker flags any figure not regenerated
    Evidence: .omo/evidence/task-43-stale.txt
  ```
  **Commit**: YES — `docs(report): internal tech report generated from artifacts`

- [x] 44. Conference paper

  **What to do**: Draft the paper: problem framing (LLM-as-proposer, not solver), method (symbolic-verification gate + Bayesian structured-formulation-search), experiments (T1–T4 with significance + effect sizes), full ablation tables (S0–S4; 6 baselines), cross-distribution results, limitations + negative results, and a reproducibility appendix. Generate all tables/figures from artifacts. Target a specific venue's format (user to pick; default: a top ML/OR venue template).

  **Must NOT do**: claim beyond the thesis_report verdicts; present dev/validation as final; include numbers not traceable to artifacts; overstate generality (report where it holds and where it doesn't).

  **Recommended Agent Profile**: Category `writing` (research paper). Skills: none.

  **Parallelization**: Parallel: YES · Wave 7 · Blocks: none · Blocked By: 40,41.

  **References**: Data: `thesis_report.json`, ablation/comparison artifacts, tech report (task 43). Doc: report's executive summary as narrative scaffold. WHY: the paper is the headline deliverable; claims must be falsifiable + sourced.

  **Acceptance Criteria**:
  - [ ] Paper builds (LaTeX/Markdown→PDF) with all tables/figures auto-generated; every claim maps to a `thesis_report.json` verdict.
  - [ ] A claims-audit script: 0 claims lacking an artifact-backed verdict; 0 dev/validation numbers in result tables.

  **QA Scenarios**:
  ```
  Scenario: Every paper claim is artifact-backed
    Tool: Bash
    Steps: build paper; run claims-audit linking each result-claim to thesis_report/comparison fields
    Expected: 0 unbacked claims; 0 dev/val numbers in headline tables; PDF builds
    Evidence: .omo/evidence/task-44-paper.txt
  Scenario: Overclaim is caught
    Tool: Bash
    Steps: insert a claim "SOTA on all domains"; run claims-audit
    Expected: audit flags it as unsupported by thesis_report (T3 scoped, not universal)
    Evidence: .omo/evidence/task-44-overclaim.txt
  ```
  **Commit**: YES — `docs(paper): conference paper draft with artifact-backed claims`

- [~] 45. [OPTIONAL STRETCH] Fine-tuned OR-LLM proposer thread (guarded)

  **Status: DEFERRED.** This optional stretch requires dedicated GPU compute (4× RTX 4090) for vLLM serving + SFT/RL training, plus a decision on whether time/compute permit. The plan explicitly states it never blocks task 44 and should run only if resources permit. All mandatory deliverables are complete; this stretch is left for a future compute allocation.

  **What to do**: ONLY if time/compute permit: synthesize OR-Instruct-style training data (from solved instances + certified deltas), SFT/RL a small open model (served via vLLM on the 4× RTX 4090) as an alternative Proposer backend behind the existing adapter, and evaluate it as an ADDITIONAL method (not a replacement). Gate strictly behind a feature flag; never blocks task 44.

  **Must NOT do**: make the main result depend on this; train on test/ood-derived data (leakage); let it delay the paper.

  **Recommended Agent Profile**: Category `deep` (data synthesis + training). Skills: none.

  **Parallelization**: Parallel: YES (independent stretch) · Wave 7 · Blocks: none · Blocked By: 37.

  **References**: External ORLM (OR-Instruct), OR-R1 (SFT+test-time RL). Pattern: `llm/client.py` adapter (task 2) + vLLM local path. WHY: optional novelty extension; the adapter already supports a local backend.

  **Acceptance Criteria**:
  - [ ] Feature-flagged OFF by default; with flag ON, the fine-tuned backend slots into the proposer via the adapter and runs the same eval pipeline as an additional method.
  - [ ] Training data provenance audit: 0 items derived from test/ood instances.

  **QA Scenarios**:
  ```
  Scenario: Stretch is fully optional + leakage-clean
    Tool: Bash
    Steps: run the full pipeline with the flag OFF (default); then ON with a tiny SFT run; audit data provenance
    Expected: OFF path unaffected; ON path adds a method row; provenance audit = 0 test/ood-derived items
    Evidence: .omo/evidence/task-45-stretch.txt
  Scenario: Adapter swap requires no caller changes
    Tool: Bash (pytest)
    Steps: swap proposer LLM backend to the local fine-tuned client
    Expected: proposer/controller callers unchanged; same Delta interface honored
    Evidence: .omo/evidence/task-45-adapter.txt
  ```
  **Commit**: YES — `feat(proposer): optional fine-tuned OR-LLM backend behind feature flag`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to the user and get an explicit "okay" before completing. Never check F1–F4 before user okay.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read this plan end-to-end. For each "Must Have": verify implementation exists (read file / run command / solve fixture). For each "Must NOT Have": search codebase for forbidden patterns (gurobipy import, silent feasible-region change, solver-only efficiency claims, dev/val numbers in paper tables) — reject with file:line if found. Check evidence files exist under `.omo/evidence/`. Compare deliverables vs plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `ruff check`, `mypy` (where configured), `pytest tests/`. Review changed files for `# type: ignore`/`as Any`, empty excepts, prints in library code, dead code, generic names (data/result/tmp), over-abstraction, AI-slop. Confirm no `gurobipy`/`ortools`-only assumptions leak into core.
  Output: `Lint [PASS/FAIL] | Types [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real End-to-End QA** — `unspecified-high` (+ `playwright` if leaderboard HTML)
  From a clean checkout: install deps, run the Phase-1 smoke and one expansion experiment via the CLI, execute EVERY task's QA scenario, verify all artifacts (results.parquet, events.jsonl, verification/*.json, repro_manifest.json, comparison_report.json, leakage_audit.json). Run `replay --strict` and confirm reproducibility. Save evidence to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N] | Artifacts [N/N] | Replay [PASS/FAIL] | Leakage [0 violations?] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read the diff. Verify 1:1 (everything specified built; nothing beyond spec). Confirm Phase-1 stayed narrow (no premature multi-kernel/MINLP/transfer). Confirm staged search spaces + ablation rows present. Confirm theses are wired to falsifiable experiments. Flag cross-task contamination and unaccounted changes.
  Output: `Tasks [N/N compliant] | Phase-discipline [OK/violations] | Theses-falsifiable [4/4] | Contamination [CLEAN/N] | VERDICT`

---

## Commit Strategy
- One commit per task (or tight group), conventional-commit style: `type(scope): desc`.
- Pre-commit gate: `ruff check && pytest -m "smoke or integration" -q` (task-relevant subset).
- Never commit benchmark data blobs (use a `benchmarks/registry.yaml` + download scripts + checksums); never commit secrets/API keys.
- Experimental result artifacts committed as compact summaries (parquet/JSON), not raw solver logs.

## Success Criteria

### Verification Commands
```bash
ruff check src tests                         # Expected: no errors
pytest tests/ -q                             # Expected: all green
python -m opop.run --config configs/phase1_smoke.yaml --out runs/smoke   # Expected: artifacts produced
python -m opop.replay --run runs/smoke --strict                          # Expected: reproduced within tolerance
python -m opop.bench.audit_leakage --run runs/final_eval --registry benchmarks/registry.yaml  # Expected: 0 violations
```

### Final Checklist
- [x] All "Must Have" present.
- [x] All "Must NOT Have" absent.
- [x] Phase-1 loop proven (task 21) before any Wave-4 expansion.
- [x] Theses T1–T4 each backed by a falsifiable, agent-runnable experiment.
- [x] F1–F4 APPROVE; user gives explicit okay.
