# Contributing to OPOP

Thanks for your interest in OPOP — a Bayesian-guided, solver-in-the-loop,
symbolically-verified formulation-and-search engine for CO/IP. This guide covers
the development setup, the test/lint/type commands, and the project conventions
that every change is expected to follow.

## Development setup

OPOP uses a `src/` layout and targets **Python 3.12**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install OPOP in editable mode with the dev toolchain
pip install -e ".[dev]"

# (optional) the Bayesian-optimization ladder backends (SMAC / BoTorch / ...)
pip install -e ".[dev,bo]"
```

`pip install -e .` puts the `opop` package on the path and installs the core
runtime dependencies (numpy, scipy, networkx, torch, pyscipopt, ortools, highspy,
pulp, openai, ...). The native solver wheels bundle their engines (SCIP 10 via
PySCIPOpt, CBC via PuLP, HiGHS via highspy), so no system solver install is
needed.

> **Note on `torch`**: the controller's Gaussian-process surrogate needs
> `torch`. If you develop against a local/custom torch build, install OPOP with
> `pip install -e . --no-build-isolation` so the existing torch is reused
> instead of pulled from PyPI.

The test suite also works without an editable install via the `src/` path shim
in `tests/conftest.py`, so `pytest` works straight from a checkout.

## Running tests

```bash
pytest -q                      # full suite
pytest -m smoke -q             # CPU-safe unit tests only
pytest tests/solver -q         # one package
pytest --doctest-modules src/opop   # doctests on public API docstrings
```

Markers (declared in `pyproject.toml`):

| Marker | Meaning |
|--------|---------|
| `smoke` | CPU-safe unit tests (the fast default gate) |
| `integration` | needs external services or pre-built data |
| `gpu` | needs GPU execution |
| `slow` | long-running; excluded from the smoke policy |

A missing optional solver should **skip**, not fail: use the
`solver_skip_if_missing` fixture (backed by `opop.solver.availability`).

## Lint and type checks

```bash
ruff check src tests docs examples     # lint (line-length 100, py312)
mypy src/opop                          # type check (ignore_missing_imports)
```

The configured type checker is **mypy**. `basedpyright` is wired for LSP import
resolution only (see `[tool.basedpyright]`); the project does not enforce its
strict-mode diagnostics. New or changed files should report **zero**
`lsp_diagnostics` — that is the bar the existing codebase holds. Prefer fixing a
diagnostic in code over adding `# type: ignore`; reserve ignores for inherent
no-stub warnings (e.g. ortools/pulp ship no type stubs) and document why.

## Solver policy: open solvers only

OPOP is **open-source-solver only**. Do **not** add `gurobipy`, MIPLearn's
`LearningSolver`, or any commercial-solver dependency to the runtime, the public
API, the docs, or the examples. MIPLearn (which hard-requires `gurobipy`) is a
**design reference only**. Ecole is not a hard dependency (its SCIP-8 ABI
conflicts with the bundled SCIP 10); extract features via
`model.as_pyscipopt()` instead. The supported backends are SCIP, OR-Tools
CP-SAT, HiGHS, CBC, and GCG.

## Code conventions

- **`src/` layout.** Library code lives under `src/opop/`; tests mirror it under
  `tests/`.
- **Pure-data state.** State objects are frozen dataclasses
  (`@dataclass(frozen=True, slots=True)`); transition with `dataclasses.replace`,
  never in-place mutation. Keep solver/network imports out of the data modules.
- **Fail-closed.** Anything unknown or unprovable in the verification path must be
  rejected, never silently accepted. Never let a delta change the feasible region
  without a certificate.
- **Typed Protocols.** New solver backends implement `opop.solver.kernel.SolverKernel`;
  new surrogates/acquisitions implement the `opop.controller.protocol` Protocols;
  new problem classes go behind a `ProblemClassAdapter`. Keep the core
  class-agnostic — no `if problem_class == ...` branches in the orchestrator or
  controller.
- **No internal mutable state in the public API.** `opop/__init__.py` only
  re-exports stable callables/classes/modules.
- **No bundled benchmark blobs.** Register datasets in `benchmarks/registry.yaml`
  with checksums + download scripts; never commit instance files or result blobs.

## Adding components

- A solver backend → [docs/howto-add-solver.md](docs/howto-add-solver.md)
- A problem class (e.g. a new CO family) → [docs/howto-add-problem-class.md](docs/howto-add-problem-class.md)
- A BO surrogate / acquisition → [docs/howto-add-surrogate.md](docs/howto-add-surrogate.md)

See [docs/architecture.md](docs/architecture.md) for the five layers, the
verification gate, and the controller ladder, and
[docs/api.md](docs/api.md) for the public API surface.

## Commits and pull requests

- Conventional-commit style: `type(scope): description`
  (e.g. `feat(solver): add Xpress kernel`, `fix(verify): tighten class-B separation`).
- Keep changes atomic; one logical change per commit.
- Before opening a PR, run the task-relevant subset of
  `ruff check && pytest -m "smoke or integration" -q` and ensure changed files
  are lint-, type-, and diagnostic-clean.
- Never commit secrets/API keys, benchmark data blobs, or raw solver logs.

## Reproducibility

Experimental runs must emit a reproducibility manifest (`threads=1`, pinned
versions, all seeds, tolerances). Use `opop.replay` to verify a run reproduces
within tolerance. Never publish dev/validation numbers as headline results, and
never tune on the sealed test/ood splits.
