"""OPOP: Bayesian-guided, solver-in-the-loop, symbolically-verified formulation-and-search engine for CO/IP.

OPOP runs a closed loop — Analyzer -> Proposer -> Verification gate -> Solver ->
Evaluator -> Bayesian Controller — that proposes *symbolically-verified*
formulation/search deltas and uses structured Bayesian optimization to drive the
next proposal. See ``docs/architecture.md`` for the five layers, the
verification gate, and the controller ladder, and ``docs/api.md`` for this
public API.

The public names below are re-exports of stable callables, classes, and the two
runnable entry-point modules (``run``/``replay``); no internal mutable state is
exposed. Attributes are loaded lazily (PEP 562) so ``import opop`` stays cheap
and a partial install (e.g. without the optional ``bo`` extra) still imports.

Example:
    >>> import opop
    >>> opop.__version__
    '0.1.0'
    >>> "run_loop" in opop.__all__ and "ScipKernel" in opop.__all__
    True
    >>> sorted(opop.__all__) == list(opop.__all__)
    True
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name -> (module path, attribute). A ``None`` attribute re-exports the
# module object itself (the runnable ``run``/``replay`` entry points).
_PUBLIC_API: dict[str, tuple[str, str | None]] = {
    # Runnable entry-point modules.
    "run": ("opop.run", None),
    "replay": ("opop.replay", None),
    # Orchestrator: the closed loop + its run summary.
    "run_loop": ("opop.orchestrator", "run_loop"),
    "RunResult": ("opop.orchestrator", "RunResult"),
    "Incumbent": ("opop.orchestrator", "Incumbent"),
    # Configuration.
    "load_config": ("opop.config", "load_config"),
    "RunConfig": ("opop.config", "RunConfig"),
    # Benchmark registry (immutable splits + leakage invariants).
    "BenchmarkRegistry": ("opop.bench.registry", "BenchmarkRegistry"),
    # Solver kernels (open solvers only).
    "SolverKernel": ("opop.solver.kernel", "SolverKernel"),
    "ScipKernel": ("opop.solver.scip", "ScipKernel"),
    "available_solvers": ("opop.solver.availability", "available_solvers"),
    "is_solver_available": ("opop.solver.availability", "is_solver_available"),
    # Controller ladder (Surrogate/Acquisition protocols + Phase-1 BO).
    "Phase1Controller": ("opop.controller.phase1", "Phase1Controller"),
    "default_phase1_space": ("opop.controller.encoder", "default_phase1_space"),
    "Surrogate": ("opop.controller.protocol", "Surrogate"),
    "Acquisition": ("opop.controller.protocol", "Acquisition"),
    "RandomSearch": ("opop.controller.protocol", "RandomSearch"),
    "EI": ("opop.controller.protocol", "EI"),
    "UCB": ("opop.controller.protocol", "UCB"),
    "GaussianProcess": ("opop.controller.gp", "GaussianProcess"),
    # Analyzer + proposer (deterministic OR analysis + typed delta proposal).
    "analyze": ("opop.analyzer.api", "analyze"),
    "propose": ("opop.proposer.api", "propose"),
    # Verification gate (delta classes A--D, fail-closed).
    "verify_delta": ("opop.verify", "verify_delta"),
    "VerificationReport": ("opop.verify", "VerificationReport"),
    # Evaluator (anytime metrics + right-censoring + scalarization).
    "evaluate": ("opop.evaluator", "evaluate"),
    "scalarize": ("opop.evaluator", "scalarize"),
    # Comparison report (Wilcoxon + shifted geomean + min-effect gating).
    "compare": ("opop.experiments.compare", "compare"),
    "ComparisonReport": ("opop.experiments.compare", "ComparisonReport"),
    # Symbolic model IR.
    "MILP": ("opop.model", "MILP"),
    "Variable": ("opop.model", "Variable"),
    "LinearConstraint": ("opop.model", "LinearConstraint"),
    "Objective": ("opop.model", "Objective"),
    "ObjSense": ("opop.model", "ObjSense"),
    "ConstraintSense": ("opop.model", "ConstraintSense"),
    "VarType": ("opop.model", "VarType"),
    # Loop state objects.
    "Phi": ("opop.model", "Phi"),
    "ProblemState": ("opop.model", "ProblemState"),
    "SolveTrace": ("opop.model", "SolveTrace"),
    "ScoreRecord": ("opop.model", "ScoreRecord"),
    "Delta": ("opop.model", "Delta"),
    "DeltaClass": ("opop.model", "DeltaClass"),
    # Problem-class adapters (MIQP/MIQCP/QUBO expansion behind a declared plugin).
    "ProblemClassAdapter": ("opop.model", "ProblemClassAdapter"),
    "AdapterCapabilities": ("opop.model", "AdapterCapabilities"),
    "register_adapter": ("opop.model", "register_adapter"),
    "find_adapter": ("opop.model", "find_adapter"),
    "get_adapter": ("opop.model", "get_adapter"),
    "QUBO": ("opop.model", "QUBO"),
    "Ising": ("opop.model", "Ising"),
    "max_cut_qubo": ("opop.model", "max_cut_qubo"),
    "qubo_to_ir": ("opop.model", "qubo_to_ir"),
    "ir_to_qubo": ("opop.model", "ir_to_qubo"),
}

if TYPE_CHECKING:
    # Static re-exports so type checkers / IDEs resolve ``opop.<name>`` and
    # ``from opop import <name>`` to the real objects. At runtime these are
    # served lazily by ``__getattr__`` below.
    from opop import replay as replay
    from opop import run as run
    from opop.analyzer.api import analyze as analyze
    from opop.bench.registry import BenchmarkRegistry as BenchmarkRegistry
    from opop.config import RunConfig as RunConfig
    from opop.config import load_config as load_config
    from opop.controller.encoder import default_phase1_space as default_phase1_space
    from opop.controller.gp import GaussianProcess as GaussianProcess
    from opop.controller.phase1 import Phase1Controller as Phase1Controller
    from opop.controller.protocol import EI as EI
    from opop.controller.protocol import UCB as UCB
    from opop.controller.protocol import Acquisition as Acquisition
    from opop.controller.protocol import RandomSearch as RandomSearch
    from opop.controller.protocol import Surrogate as Surrogate
    from opop.evaluator import evaluate as evaluate
    from opop.evaluator import scalarize as scalarize
    from opop.experiments.compare import ComparisonReport as ComparisonReport
    from opop.experiments.compare import compare as compare
    from opop.model import MILP as MILP
    from opop.model import QUBO as QUBO
    from opop.model import AdapterCapabilities as AdapterCapabilities
    from opop.model import ConstraintSense as ConstraintSense
    from opop.model import Delta as Delta
    from opop.model import DeltaClass as DeltaClass
    from opop.model import Ising as Ising
    from opop.model import LinearConstraint as LinearConstraint
    from opop.model import Objective as Objective
    from opop.model import ObjSense as ObjSense
    from opop.model import Phi as Phi
    from opop.model import ProblemClassAdapter as ProblemClassAdapter
    from opop.model import ProblemState as ProblemState
    from opop.model import ScoreRecord as ScoreRecord
    from opop.model import SolveTrace as SolveTrace
    from opop.model import VarType as VarType
    from opop.model import Variable as Variable
    from opop.model import find_adapter as find_adapter
    from opop.model import get_adapter as get_adapter
    from opop.model import ir_to_qubo as ir_to_qubo
    from opop.model import max_cut_qubo as max_cut_qubo
    from opop.model import qubo_to_ir as qubo_to_ir
    from opop.model import register_adapter as register_adapter
    from opop.orchestrator import Incumbent as Incumbent
    from opop.orchestrator import RunResult as RunResult
    from opop.orchestrator import run_loop as run_loop
    from opop.proposer.api import propose as propose
    from opop.solver.availability import available_solvers as available_solvers
    from opop.solver.availability import is_solver_available as is_solver_available
    from opop.solver.kernel import SolverKernel as SolverKernel
    from opop.solver.scip import ScipKernel as ScipKernel
    from opop.verify import VerificationReport as VerificationReport
    from opop.verify import verify_delta as verify_delta

__all__ = [
    "Acquisition",
    "AdapterCapabilities",
    "BenchmarkRegistry",
    "ComparisonReport",
    "ConstraintSense",
    "Delta",
    "DeltaClass",
    "EI",
    "GaussianProcess",
    "Incumbent",
    "Ising",
    "LinearConstraint",
    "MILP",
    "ObjSense",
    "Objective",
    "Phase1Controller",
    "Phi",
    "ProblemClassAdapter",
    "ProblemState",
    "QUBO",
    "RandomSearch",
    "RunConfig",
    "RunResult",
    "ScipKernel",
    "ScoreRecord",
    "SolveTrace",
    "SolverKernel",
    "Surrogate",
    "UCB",
    "VarType",
    "Variable",
    "VerificationReport",
    "analyze",
    "available_solvers",
    "compare",
    "default_phase1_space",
    "evaluate",
    "find_adapter",
    "get_adapter",
    "ir_to_qubo",
    "is_solver_available",
    "load_config",
    "max_cut_qubo",
    "propose",
    "qubo_to_ir",
    "register_adapter",
    "replay",
    "run",
    "run_loop",
    "scalarize",
    "verify_delta",
]


def __getattr__(name: str) -> Any:
    """Lazily import and cache a public API symbol (PEP 562)."""
    try:
        module_name, attr = _PUBLIC_API[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    """Include the lazily-exported public names in ``dir(opop)``."""
    return sorted(set(globals()) | set(__all__))
