# How to add a problem class

OPOP's core loop (orchestrator / controller / evaluator) is **problem-class
agnostic**: it never branches on whether an instance is a plain MILP, a QUBO, a
MIQP, or a MIQCP. Each non-linear problem class is handled by a *plugin* that
implements the `ProblemClassAdapter` Protocol (`opop.ProblemClassAdapter`) and
**declares** its capabilities. Adapters self-register in a process-wide registry;
generic code finds the right one with `opop.find_adapter(ir)` — never with a
hard-coded `if problem_class == ...` check.

This guide covers both halves of "adding a problem class": the **adapter** (code)
and the **benchmark registry entry** (data).

## 1. Implement the `ProblemClassAdapter` Protocol

```python
from typing import Protocol, runtime_checkable
from opop.model.ir import MILP
from opop.model.state import Phi, SolveTrace
from opop.solver.kernel import SolverKernel

@runtime_checkable
class ProblemClassAdapter(Protocol):
    @property
    def name(self) -> str: ...                 # unique registry key

    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def can_handle(self, ir: MILP) -> bool: ...  # cheap, side-effect-free predicate

    def to_milp(self, ir: MILP) -> MILP: ...     # EXACT linearization (or raise)

    def native_solve(
        self, ir: MILP, kernel: SolverKernel, *,
        phi: Phi | None = None,
        time_limit: float = 60.0, memory_limit_mb: int = 4096, seed: int = 0,
    ) -> SolveTrace: ...                          # solve the structure directly
```

Every adapter offers **two routes**:

- **`to_milp`** — an *exact* linearization to a plain linear `MILP`, so any linear
  kernel (CP-SAT / SCIP / HiGHS) can solve it. Raise
  `opop.model.ir.UnsupportedModelError` when no exact linearization exists (e.g. a
  product of two continuous variables) — **never** return a silently wrong
  relaxation.
- **`native_solve`** — build and solve the structure directly on a kernel that
  natively supports it (e.g. SCIP's nonlinear API for MIQP/MIQCP). Reject kernels
  that cannot represent the structure (fail-closed).

### Declare capabilities

`AdapterCapabilities` (`opop.AdapterCapabilities`) is what capability-driven
dispatch reads:

```python
from opop.model.adapter import AdapterCapabilities

AdapterCapabilities(
    name="mybqp",
    problem_class="My Binary Quadratic Program",
    handles_quadratic_objective=True,
    handles_quadratic_constraints=False,
    exact_linearization=True,        # to_milp is exact for what can_handle accepts
    native_kernels=("SCIP",),        # kernels usable by native_solve
    linear_kernels=("CP-SAT", "SCIP", "HiGHS"),  # kernels usable after to_milp
)
```

Keep `can_handle` predicates **mutually exclusive** across adapters so dispatch is
unambiguous (e.g. `QuboAdapter` claims all-binary unconstrained quadratics;
`MiqpAdapter` claims everything else quadratic).

## 2. Self-register on import

Register a single stateless instance at the bottom of your module:

```python
from opop.model.adapter import register_adapter

@final
class MyBqpAdapter:
    ...  # implements the Protocol above

register_adapter(MyBqpAdapter())   # idempotent: replaces any same-named registration
```

Generic code then dispatches uniformly:

```python
import opop
import opop.solver.mybqp   # importing the module runs register_adapter(...)

adapter = opop.find_adapter(ir)          # None for a plain linear MILP
if adapter is not None:
    milp = adapter.to_milp(ir)           # exact linear reformulation
    trace = opop.ScipKernel().solve(milp, opop.Phi(),
                                    time_limit=30.0, memory_limit_mb=4096, seed=0)
```

`opop.get_adapter("mybqp")` fetches by name; `opop.find_adapter(ir)` returns the
first registered adapter whose `can_handle(ir)` is `True`.

### Reference adapters
- `opop.solver.qubo.QuboAdapter` — binary quadratic programs; exact Fortet
  linearization to an all-binary MILP (plus `solve_qubo` / `route_qubo` helpers).
- `opop.solver.miqp.MiqpAdapter` — general MIQP / MIQCP; native SCIP solve, with
  exact linearization available only for binary products.

The model layer provides the building blocks: `opop.QUBO`, `opop.Ising`,
`opop.max_cut_qubo`, `opop.qubo_to_ir`, `opop.ir_to_qubo`, and the shared
`opop.model.quadratic.linearize_quadratic`.

## 3. Add a benchmark registry entry

A problem class usually arrives with instances. Register them (never the raw
blobs) in `benchmarks/registry.yaml`, one entry per dataset, with the required
fields:

```yaml
benchmarks:
  - name: my-bqp-suite
    problem_type: QUBO
    source: "https://example.org/my-bqp"   # download script + checksum, not blobs
    split:
      dev: [mybqp-0001, mybqp-0002]
      validation: [mybqp-0003]
      test: []
      ood_test: []
    license: CC-BY-4.0
    instance_count: 3
    time_limit_sec: 60
    baseline_set: scip-default
    leakage_group: mybqp-generator-v1     # families never span free & held-out splits
    checksum: "sha256:..."
    phase: 3                              # mandatory phase tag
    thesis: T3                            # mandatory thesis tag (generality)
```

`BenchmarkRegistry` enforces immutability (a sealed `split_manifest.lock`),
`assert_no_overlap()` (no instance in two splits; no `leakage_group` spanning free
and held-out splits), and split-access gating (`test`/`ood_test` require one-shot
final mode). Validate with:

```bash
python -m opop.bench.registry --validate benchmarks/registry.yaml
```

## 4. Test it

Under `tests/model/` or `tests/solver/`: assert `can_handle` is correct, that
`to_milp` preserves the optimum on a tiny instance (e.g. a 6-node Max-Cut: solve
the linearized MILP and compare to the brute-force QUBO energy), that
`native_solve` matches, and that an unsupported sub-case raises
`UnsupportedModelError`. Also assert the **core stays agnostic** — no
`if problem_class ==` branches leak into the orchestrator/controller.
