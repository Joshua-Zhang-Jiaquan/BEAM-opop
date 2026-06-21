# How to add a solver backend

A solver backend is any class that satisfies the `SolverKernel` **Protocol**
(`opop.SolverKernel`). The controller, evaluator, and orchestrator are
solver-agnostic — they only ever call `kernel.solve(...)` and read the returned
`SolveTrace`. OPOP ships SCIP, OR-Tools CP-SAT, HiGHS, CBC, and GCG kernels;
adding another (open) backend means implementing one method.

> **Open solvers only.** Do not add Gurobi or any commercial-solver dependency.

## 1. The Protocol

`opop.solver.kernel.SolverKernel` is a `@runtime_checkable` Protocol with a single
method:

```python
from typing import Protocol, runtime_checkable
from opop.model.ir import MILP
from opop.model.state import Phi, SolveTrace

@runtime_checkable
class SolverKernel(Protocol):
    def solve(
        self,
        ir: MILP,
        phi: Phi,
        *,
        time_limit: float,      # wall-clock ceiling, seconds (hard limit)
        memory_limit_mb: int,   # memory ceiling, MiB (hard limit)
        seed: int,              # master randomisation seed
    ) -> SolveTrace: ...
```

Because the Protocol is structural, your class does **not** need to subclass
anything — implementing `solve` with this signature is enough, and
`isinstance(MyKernel(), opop.SolverKernel)` will hold.

### Determinism contract (every kernel MUST honour)
- **`threads == 1`** — no LP/solver parallelism, so traces are reproducible.
- **`time_limit` / `memory_limit_mb` are hard ceilings.** Hitting either
  terminates the run and yields `SolveTrace.censored == True`.
- **`seed`** drives the backend's master randomisation, so repeated calls with
  the same `(ir, phi, limits, seed)` produce identical traces.
- **Never swallow solver errors.** An import/initialisation/solve failure
  propagates (or surfaces as a typed exception); it is never masked.

## 2. Implement `solve`

```python
# src/opop/solver/mybackend.py
from __future__ import annotations
import math
from opop.model.ir import MILP
from opop.model.state import Phi, SolveTrace

class MyKernel:
    solver_name = "MyBackend"  # canonical name (used by routing + native_solve guards)

    def solve(
        self, ir: MILP, phi: Phi, *,
        time_limit: float, memory_limit_mb: int, seed: int,
    ) -> SolveTrace:
        model = self._build(ir)          # compile IR -> backend model
        self._apply_budget(model, time_limit, memory_limit_mb, seed)  # threads=1!
        self._apply_params(model, phi.p)  # phi.p carries proposer param overrides
        model.optimize()

        status = self._status_string(model)              # backend status, normalised
        return SolveTrace(
            primal_bound_series=[self._primal(model)],    # ≥1 terminal point
            dual_bound_series=[self._dual(model)],
            time_series=[self._solve_time(model)],
            nodes=self._nodes(model),
            lp_iters=self._lp_iters(model),
            cuts=self._cuts(model),
            first_feasible_time=math.nan,                 # nan = "not measured"
            status=status,
            censored=status in _LIMIT_STATUSES,           # True iff stopped by a budget
            memory_peak=self._mem_mib(model),
            instance_id=ir.name,
            solver=self.solver_name,
        )
```

### Populating `SolveTrace`
- `primal_bound_series` / `dual_bound_series` / `time_series` are **parallel,
  index-aligned** arrays — one triple per captured event (for a callback-rich
  backend) plus a final terminal point. Map a backend's `±inf`/sentinel bounds to
  Python `math.inf` / `-math.inf`.
- `censored` must be `True` **iff** the run was stopped by a resource limit
  *without* an optimality proof (timeout / memory / node limit / interrupt) — and
  `False` for definitive `optimal` / `infeasible` / `unbounded`.
- If your backend cannot expose a field (e.g. no node count), use the documented
  sentinel (`math.nan` for `first_feasible_time`) — **never fabricate** a value.

For a richer reference (event-based trajectory capture), read
`opop.solver.scip.ScipKernel`; for graceful degradation when a backend exposes
few signals, read `opop.solver.highs` / `opop.solver.cbc`.

## 3. Advertise the backend

Register the backend (name + version probe) in
`opop.solver.availability` so `opop.available_solvers()` and
`opop.is_solver_available("MyBackend")` see it. Routing helpers (for example
`opop.solver.qubo.route_qubo`) consult these to pick a capable kernel.

## 4. Test it

Mirror the source under `tests/solver/`. Guard the test so a missing backend
**skips** rather than fails:

```python
def test_mybackend_matches_scip(solver_skip_if_missing):
    solver_skip_if_missing("MyBackend")
    trace = MyKernel().solve(knapsack_ir, Phi(), time_limit=10.0, memory_limit_mb=2048, seed=0)
    assert trace.primal_bound_series[-1] == 1.0   # known optimum
    assert isinstance(MyKernel(), SolverKernel)    # structural Protocol check
```

Cross-check the optimum against an existing kernel (e.g. SCIP) on a few fixtures.
Keep `ruff`, `mypy`, and `lsp_diagnostics` clean on the new files.
