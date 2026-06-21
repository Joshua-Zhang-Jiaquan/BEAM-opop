# Solver Stack — Task 3 (Bootstrap · Smoke · Version-Conflict Spike)

- **Status**: CONFIRMED (installed + smoke-tested)
- **Environment**: Python 3.12.3, 128 CPU
- **Date**: 2026-06-20
- **Guardrail**: open-source solvers **only** — no Gurobi / commercial dependency.

## 1. Installed & smoke-tested backends

| Solver | Python binding | Solver engine | `available_solvers()` version | Notes |
|--------|----------------|---------------|-------------------------------|-------|
| SCIP   | `pyscipopt` 6.2.1 | SCIP 10.0.2 (+ SoPlex 8.0.2) | `10.0` | High-fidelity core; wheel bundles SCIP — no system install. |
| CP-SAT | `ortools` 9.14.6206 | OR-Tools CP-SAT | `9.14.6206` | snake_case API (`new_int_var`/`add`/`solve`). |
| HiGHS  | `highspy` 1.14.0 | HiGHS 1.14.0 | `1.14.0` | No module `__version__`; use `Highs().version()`. |
| CBC    | `PuLP` 3.2.1 | CBC 2.10.3 | `2.10.3` | CBC binary bundled inside PuLP — no cylp / system CBC. |

> `available_solvers()` reports SCIP as `10.0` because `pyscipopt.Model().version()`
> returns the `major.minor` float; the full engine build is **SCIP 10.0.2** (from
> `Model.printVersion()`), matching the inherited "PySCIPOpt 6.2.x ↔ SCIP 10.0.2" coupling.

**Smoke agreement.** Canonical model `maximize x+y  s.t.  x+y<=1,  x,y∈{0,1}` (optimum = 1).
All four backends return objective **1.0** at status **OPTIMAL** (`agrees=True`).
- Probe: `python -m opop.solver.availability`
- Smoke: `opop.solver.smoke.run_all_smoke()`
- Tests: `tests/solver/test_availability.py` (green; skip-if-missing for absent solvers)
- Evidence: `.omo/evidence/task-3-solver-agreement.txt`

### Environment constraints honored
- **numpy held at 1.26.4** — the local torch `2.8.0a0` build needs numpy<2. This caps
  `ortools` at the **9.14.x** line; `ortools>=9.15` requires numpy>=2.
- `ortools` requires **protobuf<6.32**, so protobuf was downgraded `6.33.6 → 6.31.1`.
  Side effect: a **pre-existing** `wandb 0.18.3` (wants protobuf<6) conflict warning persists.
  This is unrelated to OPOP and is left as-is (outside task-3 scope; wandb is not an OPOP dep).

## 2. Conflict spike (a): MIPLearn → `gurobipy>=12`  ⇒  DESIGN REFERENCE ONLY

**Finding (primary source).** MIPLearn 0.4.3 `setup.py` (`ANL-CEEESA/MIPLearn@v0.4.3`)
declares in `install_requires` a **hard, non-optional** dependency:

```
install_requires=[ ..., "gurobipy>=12,<13", "pandas>=1,<2", ... ]
```

`gurobipy` is Gurobi's commercial, license-gated Python binding → **incompatible with the
open-only guardrail**. Secondary blockers observed on this environment:
- `pandas>=1,<2` conflicts with the installed pandas 2.2.3.
- The sdist **fails to build on Python 3.12** under pip (`setup.py` uses `pkg_resources`;
  build-isolation also tries to Cython-compile a pinned pandas from source and errors out).

**Decision.** **MIPLearn = design reference only.** Do **not** install `gurobipy` or
`miplearn` as runtime deps (verified: `gurobipy` absent). Re-implement the learning-augmented
capabilities we actually want (warm-start prediction, learned incumbents, learned cut/branch
priorities) directly on **SCIP / PySCIPOpt** within `opop/analyzer` + `opop/controller`,
treating MIPLearn's papers and API surface purely as a blueprint.

## 3. Conflict spike (b): Ecole 0.8.1 (SCIP 8) vs PySCIPOpt 6.x (SCIP 10)  ⇒  NO HARD DEP

**Finding.** `import ecole` in the core env → `ModuleNotFoundError` (absent; the core imports
fine without it). On PyPI the newest Ecole is **0.8.1 (2022)**, distributed **sdist-only**
(no cp312 wheel — `pip install ecole` would build from source). Ecole links SCIP at **build
time** against the **SCIP 8.x** C API/ABI, whereas our core ships **SCIP 10.0.2** bundled in
`pyscipopt` 6.2.1. A from-source Ecole build would require a separate SCIP 8 dev install and be
**ABI-incompatible** with SCIP 10 — the two cannot coexist in one SCIP-10 environment. Ecole
upstream is effectively inactive.

**Decision (DEFAULT).** **No hard Ecole dependency; do not pin Ecole into core requirements.**
Extract MILP / bipartite (variable–constraint) features **directly from the PySCIPOpt `Model`**
— the `model.as_pyscipopt()` path named in the plan — via `getVars()`/`getConss()` and root-LP
data in `opop/analyzer` (task 10). If Ecole-style observation functions are ever needed, isolate
them behind an optional extra in a separate SCIP-8 sidecar environment — never in the core.
The core MUST continue to import without `ecole`.

## 4. Capability probe (implementation)

- `src/opop/solver/availability.py`
  - `available_solvers() -> list[{name, version, available, detail}]` (detection is
    deterministic per process; import/init failures are captured in `detail`, **never masked**).
  - `is_solver_available(name)`, `solver_infos()`, and a `__main__` CLI printing the table.
- `src/opop/solver/smoke.py` — per-backend tiny-MILP solve + `SmokeResult.agrees()` for the
  cross-solver agreement assertion.

## 5. Downstream impact (feeds tasks 9, 10, 12, 22–24)

- Tasks **9 / 10 / 12** (model IR, analyzer, SCIP kernel) build on **PySCIPOpt 6.2.1 / SCIP 10.0.2**.
- Tasks **22–24** add **CP-SAT** (`ortools` 9.14.6206), **HiGHS** (`highspy` 1.14.0),
  **CBC** (`PuLP` 3.2.1) adapters.
- **GCG** (task 24) is **not** bundled by `pyscipopt` and will need its own bootstrap spike.
