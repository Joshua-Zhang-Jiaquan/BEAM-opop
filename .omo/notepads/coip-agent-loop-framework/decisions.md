
## Task 1 — scaffolding decisions

### torch pinning
- torch 2.8.0a0 is an alpha build installed from a custom NV wheel; it cannot be pinned in `requirements.txt` as a standard version. Decision: leave `torch` unpinned with a comment `# torch pinned to installed; use torch==2.8.0a0+... for the local build`. This avoids breaking `pip install -r requirements.txt` on machines without the custom wheel while documenting the expected version.

### pyproject.toml scope
- Strictly tooling-only (`[tool.pytest.ini_options]`, `[tool.ruff]`, `[tool.mypy]`). No `[project]` or `[build-system]` — deferred to task 41 (package build) as instructed. This matches the DAC sibling project convention.

### requirements.txt version sources
- Installed versions used as baseline for already-present packages (numpy, scipy, networkx, openai, python-dotenv, requests, pyyaml, ruff, mypy).
- Target versions used for not-yet-installed packages (pytest 8.3.3, pylint 3.3.1) and solver/BO packages (task 3 will confirm).
- Reference file from intern/topic2-format-efficiency used for section labeling style and overall structure.

### CLI subcommands
- Three subcommands defined: `run`, `replay`, `bench` — matching the AGENTS.md description and the planned controller loop. All are no-op stubs (`print("...: not implemented")`).

## Task 3 — solver version-conflict decisions

### (a) MIPLearn → DESIGN REFERENCE ONLY
- Primary source: MIPLearn 0.4.3 `setup.py` `install_requires` includes a HARD dep `gurobipy>=12,<13` (commercial Gurobi) → incompatible with open-only. Also pins `pandas>=1,<2` (clashes with installed 2.2.3); sdist fails to build on py3.12. `gurobipy` confirmed ABSENT and NOT installed.
- Decision: MIPLearn = **design reference only**; re-implement learning-augmented features on SCIP/PySCIPOpt. (docs/design/solver-stack.md §2; evidence: `.omo/evidence/task-3-miplearn-spike.txt`)

### (b) Ecole 0.8.1 (SCIP 8) vs PySCIPOpt 6.2.1 (SCIP 10) → NO HARD DEP
- `import ecole` → ModuleNotFoundError (absent; core imports fine without it). PyPI newest = 0.8.1 (2022), sdist-only (no cp312 wheel) → builds from source against SCIP 8 ABI; conflicts with the bundled SCIP 10.0.2. Upstream inactive.
- Decision (DEFAULT): **no hard Ecole dep; do NOT pin Ecole into core requirements.** Extract bipartite/MILP features directly from the PySCIPOpt `Model` (the `model.as_pyscipopt()` path) in the analyzer (task 10). Core MUST import without ecole. (docs/design/solver-stack.md §3; evidence: `.omo/evidence/task-3-ecole-spike.txt`)

### requirements.txt solver pins
- `pyscipopt==6.2.1`, `ortools==9.14.6206`, `highspy==1.14.0`, `pulp==3.2.1` (CBC 2.10.3). The `# --- Bayesian Optimization ---` pins (botorch/gpytorch/ax/smac/ConfigSpace) were left UNTOUCHED — they belong to task 8 and were NOT installed/validated here.

## Task 7 — registry lock + leakage policy decisions

### Lock content and scope
- `split_manifest.lock` is a JSON file containing only `{hash, algorithm}` rather than the full assignment. The full assignment stays in `benchmarks/registry.yaml`; the lock is a tamper-evident seal.
- The hash covers `{benchmark_name}::{instance_id} -> split` so that ids are scoped to their owning benchmark entry. Instance ids are still required to be globally unique via `assert_no_overlap()`.
- `--reseal` regenerates the lock when the registry legitimately changes; without it, any mismatch aborts loading.

### Leakage-group semantics
- A `leakage_group` tags a set of instances that share a non-independent generative process (e.g., same synthetic generator seed/distribution).
- To avoid leakage, a group may live only in free splits (`dev`/`validation`) **or** only in held-out splits (`test`/`ood_test`), never both.
- Empty split declarations do not count toward the span; this lets a single registry entry declare all four split keys while remaining policy-compliant when only free or only held-out ids are present.

### Held-out split access
- Test/ood splits are gated behind `one_shot_final=True` in `get_split()`. This is a runtime guard, not a file permission; it prevents accidental use during development/validation loops while still allowing the final evaluation harness to load them.

### Phase + thesis enforcement
- Every benchmark entry must carry `phase` and `thesis`. These are schema-level required fields, not optional annotations, so benchmark families cannot be added to the registry without an explicit research justification and phase assignment.
