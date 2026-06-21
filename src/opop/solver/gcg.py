"""GCG solver kernel: Dantzig--Wolfe branch-price-and-cut via pygcgopt (task 24).

:class:`GcgKernel` is a :class:`~opop.solver.kernel.SolverKernel` backed by GCG
(the generic column-generation / branch-price-and-cut solver) through its
``pygcgopt`` Python binding. GCG extends SCIP with automatic Dantzig--Wolfe
reformulation: it detects a block-angular structure, moves the coupling
constraints into a master, prices the blocks as subproblems, and runs
branch-price-and-cut. ``pygcgopt.Model`` exposes the PySCIPOpt model API, so
model construction, the determinism/budget knobs, and the terminal statistics
mirror :class:`~opop.solver.scip.ScipKernel`.

GCG is an OPTIONAL backend. When ``pygcgopt`` is not importable, the kernel
raises a typed :class:`SolverUnavailableError` on **construction** — a clean
import-time failure path — so callers / tests detect-and-skip rather than crash
mid-solve. Probe availability with
``opop.solver.availability.is_solver_available("gcg")``.

Decomposition source. When the IR carries a certified decomposition annotation
(``ir.metadata["decomposition"]`` — produced by
:func:`opop.analyzer.decompose.decomposition_delta` and certified class-C by the
verification gate), the kernel hands GCG that block structure
(``addDecompositionFromConss``); otherwise it lets GCG auto-detect. Either way
the *math model* is unchanged, so the solve is over the same feasible set.

Determinism contract (shared with ScipKernel): single-threaded, hard time/memory
limits, fixed seed. The trajectory is reported as the terminal primal/dual/time
state (one point); event-based trajectory capture mirrors ScipKernel and is
deferred until a live GCG build is available to verify column-generation bounds.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

from opop.model.ir import MILP, ConstraintSense, VarType
from opop.model.state import Phi, SolveTrace
from opop.solver.scip import WHITELISTED_SEPARATORS

__all__ = ["GcgKernel", "SolverUnavailableError"]


class SolverUnavailableError(RuntimeError):
    """A solver backend could not be constructed (binding / engine absent)."""


# Determinism / budget knobs GCG inherits from SCIP. Applied AFTER phi.p so a
# design vector can never weaken reproducibility or the resource ceilings.
_THREADS_PARAM = "lp/threads"
_TIME_PARAM = "limits/time"
_MEMORY_PARAM = "limits/memory"
_SEED_PARAM = "randomization/randomseedshift"
_BYTES_PER_MIB = 1024.0 * 1024.0

_VTYPE_TO_GCG: dict[VarType, str] = {
    VarType.BINARY: "B",
    VarType.INTEGER: "I",
    VarType.CONTINUOUS: "C",
}

# Resource-limit terminations (right-censored): stopped early without an
# optimality proof. Mirrors ScipKernel (GCG reports SCIP's status strings).
_LIMIT_STATUSES: frozenset[str] = frozenset(
    {
        "timelimit",
        "memlimit",
        "gaplimit",
        "sollimit",
        "bestsollimit",
        "nodelimit",
        "totalnodelimit",
        "stallnodelimit",
        "restartlimit",
        "userinterrupt",
        "terminate",
    }
)


def _finite_or_inf(model: Any, value: float) -> float:
    """Map a SCIP/GCG bound (``+-1e20`` sentinel) to a float / ``math.inf``."""
    if model.isInfinity(value):
        return math.inf
    if model.isInfinity(-value):
        return -math.inf
    return float(value)


def _is_censored(status: str) -> bool:
    """``True`` iff ``status`` is a resource-limit termination (right-censored)."""
    return status in _LIMIT_STATUSES


class GcgKernel:
    """A GCG-backed :class:`~opop.solver.kernel.SolverKernel`.

    Args:
        apply_decomposition: When ``True`` (default) and the IR carries a
            certified ``metadata["decomposition"]`` annotation, hand that block
            structure to GCG; otherwise GCG auto-detects. Set ``False`` to always
            auto-detect.

    Raises:
        SolverUnavailableError: If ``pygcgopt`` is not importable at construction.
    """

    solver_name: str = "GCG"

    def __init__(self, *, apply_decomposition: bool = True) -> None:
        if importlib.util.find_spec("pygcgopt") is None:
            raise SolverUnavailableError(
                "GCG backend unavailable: the 'pygcgopt' binding is not importable. "
                + "Install it with `pip install PyGCGOpt` (it bundles the GCG engine)."
            )
        self.apply_decomposition: bool = apply_decomposition

    # -- proposer hooks (mirror ScipKernel: class-B separator whitelist) -----
    def _apply_params(self, model: Any, phi: Phi) -> None:
        for key, value in phi.p.items():
            if key.startswith("separating/"):
                parts = key.split("/")
                sep_name = parts[1] if len(parts) >= 2 else ""
                if sep_name not in WHITELISTED_SEPARATORS:
                    allowed = sorted(WHITELISTED_SEPARATORS)
                    raise ValueError(
                        f"separator {sep_name!r} (param {key!r}) is not class-B whitelisted; "
                        + f"allowed separators: {allowed}"
                    )
            model.setParam(key, value)

    def solve(
        self,
        ir: MILP,
        phi: Phi,
        *,
        time_limit: float,
        memory_limit_mb: int,
        seed: int,
    ) -> SolveTrace:
        """Compile ``ir`` + ``phi``, solve via GCG under the budget, return a trace.

        See :class:`~opop.solver.kernel.SolverKernel` for the contract. Solver
        import/build/solve errors propagate to the caller (never swallowed); a
        manual-decomposition hint that GCG cannot apply falls back to GCG's
        automatic detection rather than failing the solve.
        """
        pygcgopt = importlib.import_module("pygcgopt")
        model_factory = getattr(pygcgopt, "Model")
        quicksum = getattr(pygcgopt, "quicksum")

        model = model_factory(ir.name or "opop_gcg")
        cons_by_name = _build_model(model, ir, quicksum)
        model.hideOutput()

        # Proposer params first, then the authoritative budget knobs.
        self._apply_params(model, phi)
        model.setIntParam(_THREADS_PARAM, 1)
        model.setRealParam(_TIME_PARAM, float(time_limit))
        model.setRealParam(_MEMORY_PARAM, float(memory_limit_mb))
        model.setIntParam(_SEED_PARAM, int(seed))

        if self.apply_decomposition:
            _apply_decomposition(model, ir, cons_by_name)

        model.optimize()

        status = str(model.getStatus())
        final_primal = _finite_or_inf(model, model.getPrimalbound())
        final_dual = _finite_or_inf(model, model.getDualbound())
        solving_time = float(model.getSolvingTime())

        return SolveTrace(
            primal_bound_series=[final_primal],
            dual_bound_series=[final_dual],
            time_series=[solving_time],
            nodes=int(model.getNTotalNodes()),
            lp_iters=int(model.getNLPIterations()),
            cuts=int(model.getNCutsApplied()),
            first_feasible_time=math.nan,
            status=status,
            censored=_is_censored(status),
            memory_peak=float(model.getMemUsed()) / _BYTES_PER_MIB,
            instance_id=ir.name,
            solver=self.solver_name,
        )


# ---------------------------------------------------------------------------
# Model compilation (pygcgopt.Model speaks the PySCIPOpt build API)
# ---------------------------------------------------------------------------
def _bound(model: Any, value: float) -> float:
    if value == math.inf:
        return model.infinity()
    if value == -math.inf:
        return -model.infinity()
    return value


def _build_model(model: Any, ir: MILP, quicksum: Any) -> dict[str, Any]:
    """Populate ``model`` from ``ir``; return the ``{con name -> cons}`` map."""
    gcg_vars: dict[str, Any] = {}
    for var in ir.variables:
        gcg_vars[var.name] = model.addVar(
            name=var.name,
            vtype=_VTYPE_TO_GCG[var.vtype],
            lb=_bound(model, var.lower),
            ub=_bound(model, var.upper),
        )

    cons_by_name: dict[str, Any] = {}
    for con in ir.constraints:
        expr = quicksum(coeff * gcg_vars[name] for name, coeff in con.coeffs.items())
        if con.sense is ConstraintSense.LE:
            cons_by_name[con.name] = model.addCons(expr <= con.rhs, name=con.name)
        elif con.sense is ConstraintSense.GE:
            cons_by_name[con.name] = model.addCons(expr >= con.rhs, name=con.name)
        else:
            cons_by_name[con.name] = model.addCons(expr == con.rhs, name=con.name)

    obj_terms = [coeff * gcg_vars[name] for name, coeff in ir.objective.coeffs.items()]
    model.setObjective(quicksum(obj_terms) if obj_terms else 0, sense=ir.objective.sense.value)
    if ir.objective.offset != 0.0:
        model.addObjoffset(ir.objective.offset)
    return cons_by_name


# ---------------------------------------------------------------------------
# Decomposition hand-off (best-effort; falls back to GCG auto-detection)
# ---------------------------------------------------------------------------
def _apply_decomposition(model: Any, ir: MILP, cons_by_name: dict[str, Any]) -> bool:
    """Hand a certified block-angular decomposition to GCG, if the IR carries one.

    Reads ``ir.metadata["decomposition"]`` (a coupling-constraint border for
    ``DW`` / ``BLOCK``) and registers it via ``addDecompositionFromConss``.
    Returns ``True`` when applied; any absence / malformed payload / binding
    incompatibility yields ``False`` so GCG falls back to automatic detection
    (never failing the solve). Variable-linking (``BENDERS``) is left to GCG's
    own detectors.
    """
    decomp = ir.metadata.get("decomposition")
    if not isinstance(decomp, dict):
        return False
    payload: dict[str, Any] = decomp
    if payload.get("decomposability") not in ("DW", "BLOCK"):
        return False
    raw_blocks: Any = payload.get("block_vars") or []
    raw_linking: Any = payload.get("linking_constraints") or []
    linking = {str(name) for name in raw_linking}
    block_var_sets = [frozenset(str(v) for v in block) for block in raw_blocks]
    if len(block_var_sets) < 2:
        return False

    block_cons: list[list[Any]] = [[] for _ in block_var_sets]
    for con in ir.constraints:
        if con.name in linking:
            continue
        support = set(con.coeffs)
        for idx, members in enumerate(block_var_sets):
            if support <= members:
                block_cons[idx].append(cons_by_name[con.name])
                break
    if any(not block for block in block_cons):
        return False

    master_cons = [cons_by_name[name] for name in linking if name in cons_by_name]
    try:
        model.addDecompositionFromConss(master_cons, *block_cons)
    except Exception:  # noqa: BLE001 -- binding/version drift -> GCG auto-detects
        return False
    return True
