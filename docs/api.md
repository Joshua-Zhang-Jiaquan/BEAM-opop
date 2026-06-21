# OPOP Public API

`import opop` exposes a small, stable public surface: the two runnable
entry-point **modules** (`run`, `replay`), the closed-loop driver, the solver
kernels, the controller ladder, the verification gate, the evaluator, the
benchmark registry, the comparison report, and the symbolic model IR (including
the problem-class adapters). Everything is **re-exported** from internal modules;
no internal mutable state is exposed.

Attributes are loaded lazily (PEP 562 `__getattr__`), so `import opop` is cheap
and a partial install (for example, without the optional `bo` extra) still
imports.

## Install

```bash
pip install .            # core (open solvers: SCIP, CP-SAT, HiGHS, CBC)
pip install ".[bo]"      # + SMAC / BoTorch controller-ladder backends
pip install ".[dev]"     # + pytest / ruff / mypy toolchain
```

## Quick start

### Command line

```bash
# Run the Phase-1 closed loop vs SCIP-default and emit all six artifacts.
opop run --config configs/phase1_smoke.yaml --out runs/smoke
# Equivalent module entry points:
python -m opop.run    --config configs/phase1_smoke.yaml --out runs/smoke
python -m opop.replay --run runs/smoke/instances/<instance>_<seed> --strict
```

### Library

```python
import opop

# Load a run config and drive the Phase-1 smoke end-to-end.
config = opop.load_config("configs/phase1_smoke.yaml")
opop.run.run_phase1_smoke(config, "runs/smoke")

# Inspect which open solvers are installed.
for info in opop.available_solvers():
    print(info["name"], info.get("version"))

# Build a tiny MILP IR and solve it with the SCIP kernel.
milp = opop.MILP(
    name="knapsack",
    variables=(
        opop.Variable("x", opop.VarType.BINARY, 0.0, 1.0),
        opop.Variable("y", opop.VarType.BINARY, 0.0, 1.0),
    ),
    constraints=(
        opop.LinearConstraint("cap", {"x": 1.0, "y": 1.0}, opop.ConstraintSense.LE, 1.0),
    ),
    objective=opop.Objective({"x": 1.0, "y": 1.0}, opop.ObjSense.MAXIMIZE, 0.0),
)
trace = opop.ScipKernel().solve(
    milp, opop.Phi(), time_limit=10.0, memory_limit_mb=2048, seed=0
)
score = opop.evaluate(trace, time_limit=10.0)
print(score.metrics["objective"])  # 1.0
```

See `examples/phase1_smoke.py` and `examples/expansion_miqp.py` for complete,
runnable scripts.

## Public surface

### Runnable entry-point modules
| Name | Kind | Purpose |
|------|------|---------|
| `opop.run` | module | Phase-1 end-to-end smoke (`run_phase1_smoke`, `main`). |
| `opop.replay` | module | Strict replay of a recorded run (`replay_run`, `main`). |

### Orchestration & config
| Name | Kind | Purpose |
|------|------|---------|
| `opop.run_loop` | function | Drive Analyzer→Proposer→Verify→Solver→Evaluator→Controller to budget. |
| `opop.RunResult`, `opop.Incumbent` | class | Run summary + best configuration. |
| `opop.load_config` | function | Load a `RunConfig` from JSON/YAML with `OPOP_*` env overrides. |
| `opop.RunConfig` | class | Top-level experiment configuration. |

### Benchmark registry
| Name | Kind | Purpose |
|------|------|---------|
| `opop.BenchmarkRegistry` | class | Immutable dev/validation/test/ood splits + leakage invariants. |

### Solver kernels (open solvers only)
| Name | Kind | Purpose |
|------|------|---------|
| `opop.SolverKernel` | Protocol | The kernel contract (`solve(ir, phi, *, time_limit, memory_limit_mb, seed)`). |
| `opop.ScipKernel` | class | SCIP backend (the Phase-1 reference kernel). |
| `opop.available_solvers`, `opop.is_solver_available` | function | Capability probe. |

### Controller ladder
| Name | Kind | Purpose |
|------|------|---------|
| `opop.Phase1Controller` | class | Ask-tell BO over the encoded `Phi` subspace (`.bo(...)`, `.random(...)`). |
| `opop.default_phase1_space` | function | The restricted Phase-1 `Phi` search space. |
| `opop.Surrogate`, `opop.Acquisition` | Protocol | Swappable surrogate / acquisition contracts. |
| `opop.GaussianProcess` | class | Matérn-5/2 GP surrogate. |
| `opop.RandomSearch`, `opop.EI`, `opop.UCB` | class | Baseline + acquisition policies. |

### Analyzer, proposer, verification, evaluation
| Name | Kind | Purpose |
|------|------|---------|
| `opop.analyze` | function | Deterministic OR analysis → `AnalysisReport`. |
| `opop.propose` | function | LLM-guided / rule-based typed `Delta` proposal. |
| `opop.verify_delta`, `opop.VerificationReport` | function/class | The fail-closed A–D gate. |
| `opop.evaluate`, `opop.scalarize` | function | `SolveTrace` → `ScoreRecord`; BO scalarization. |
| `opop.compare`, `opop.ComparisonReport` | function/class | Wilcoxon + shifted-geomean + min-effect gating. |

### Symbolic model IR & loop state
| Name | Kind | Purpose |
|------|------|---------|
| `opop.MILP`, `opop.Variable`, `opop.LinearConstraint`, `opop.Objective` | class | The MILP IR records. |
| `opop.VarType`, `opop.ObjSense`, `opop.ConstraintSense` | enum | IR enumerations. |
| `opop.Phi`, `opop.ProblemState`, `opop.SolveTrace`, `opop.ScoreRecord` | class | Loop state objects. |
| `opop.Delta`, `opop.DeltaClass` | class/enum | Typed edits + their verification classes. |

### Problem-class adapters (MIQP / MIQCP / QUBO expansion)
| Name | Kind | Purpose |
|------|------|---------|
| `opop.ProblemClassAdapter`, `opop.AdapterCapabilities` | Protocol/class | The plugin contract + declared capabilities. |
| `opop.register_adapter`, `opop.find_adapter`, `opop.get_adapter` | function | Capability-driven dispatch registry. |
| `opop.QUBO`, `opop.Ising` | class | Binary-quadratic / spin energy models. |
| `opop.max_cut_qubo`, `opop.qubo_to_ir`, `opop.ir_to_qubo` | function | Problem builders + IR bridges. |

## How-to guides
- Add a solver backend → [howto-add-solver.md](howto-add-solver.md)
- Add a problem class → [howto-add-problem-class.md](howto-add-problem-class.md)
- Add a BO surrogate / acquisition → [howto-add-surrogate.md](howto-add-surrogate.md)

For the engine's layers, verification gate, and controller ladder, see
[architecture.md](architecture.md).
