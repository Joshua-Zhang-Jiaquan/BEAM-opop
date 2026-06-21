"""Baselines 5 & 6 — classic matheuristics and LLM-enhanced CO (plan task 38).

Two *standalone* baselines that share one harness + cost accounting and emit a
schema-identical ``results.parquet`` so they slot straight into
:mod:`opop.experiments.compare`:

* **Baseline 5 — classic matheuristic** (:func:`run_matheuristic_baseline`):
  warm-start a feasible incumbent with a quick truncated SCIP solve, then apply
  exactly ONE neighbourhood core from :mod:`opop.solver.heuristics`
  (``local_branching`` / ``rins`` / ``large_neighborhood_search``). No LLM is
  involved, so the LLM cost columns are all zero.
* **Baseline 6 — LLM-enhanced CO** (:func:`run_llm_enhanced_baseline`): reproduce
  the LLM-LNS / HeurAgenix loop — over several rounds an LLM (the
  :mod:`opop.experiments.heuristic_selector`) *selects/evolves* which core to run
  next from the closed vocabulary ``{local_branching, rins, lns, repair}``; the
  chosen core is then applied to the running incumbent. Per-row LLM token/cost is
  tracked and folded into the honest end-to-end wall time.

Scientific-integrity contract (mirrors :mod:`opop.bench.cost`):

* Both baselines run **standalone** — they NEVER call :func:`opop.analyzer.api.analyze`,
  :func:`opop.verify.gate.verify_delta`, or build a ``Phase1Controller``. The
  ``analyzer_time`` / ``controller_time`` / ``verification_time`` cost columns are
  therefore always exactly ``0.0``, which is itself the evidence of no blending.
* The headline ``time`` column is the **end-to-end** wall time
  (``total_wall_time``), so the LLM-enhanced baseline never reports solver-only
  time. The LLM heuristic-selection wall time is booked under ``proposer_time``
  (the selection *is* the proposal step); this is cost bookkeeping only and
  invokes no opop loop component.
* Matheuristic / LLM-enhanced runs are pure primal heuristics: they never certify
  global optimality. ``solved`` is ``True`` only when a caller-supplied
  ``reference_optimum`` is provably reached; ``censored`` reflects whether any
  inner solve hit its time limit.

Importing this module needs no solver backend (PySCIPOpt is imported lazily by
:func:`opop.model.ir.to_pyscipopt`); only the actual solves require SCIP.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opop.bench.cost import COST_FIELDS, make_event_cost
from opop.evaluator import evaluate
from opop.experiments.heuristic_selector import (
    ALLOWED_HEURISTICS,
    DEFAULT_HEURISTIC,
    select_heuristic,
)
from opop.llm.client import LLMClient
from opop.model.ir import MILP, ObjSense, VarType, to_pyscipopt
from opop.model.state import Phi, ScoreRecord, SolveTrace
from opop.solver.heuristics import (
    DEFAULT_MEMORY_LIMIT_MB,
    HeuristicResult,
    is_solution_feasible,
    large_neighborhood_search,
    local_branching,
    repair_solution,
    rins,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BaselineOutcome",
    "LLM_HEURISTIC_METHOD",
    "MATHEURISTIC_METHOD",
    "RESULT_COLUMNS",
    "main",
    "run_baseline_suite",
    "run_llm_enhanced_baseline",
    "run_matheuristic_baseline",
    "write_results",
]

#: Method tag for the classic matheuristic baseline (Baseline 5).
MATHEURISTIC_METHOD: str = "matheuristic"
#: Method tag for the LLM-enhanced CO baseline (Baseline 6).
LLM_HEURISTIC_METHOD: str = "llm-heuristic"

#: Tolerance for declaring an incumbent equal to a known reference optimum.
_OPT_MATCH_TOL: float = 1e-6
#: Strict-improvement tolerance (mirrors heuristics.OBJ_TOL).
_OBJ_TOL: float = 1e-9

_DISCRETE: frozenset[VarType] = frozenset({VarType.BINARY, VarType.INTEGER})

# Base (non-cost) result columns; the full schema appends every cost column.
_BASE_COLUMNS: tuple[str, ...] = (
    "instance_id",
    "method",
    "seed",
    "heuristic",
    "primal_integral",
    "gap",
    "time",
    "solved",
    "censored",
    "time_limit",
    "n_accepted",
    "n_llm_calls",
)

#: The canonical (stable-order) results schema both baselines emit.
RESULT_COLUMNS: tuple[str, ...] = (*_BASE_COLUMNS, *COST_FIELDS)


# ---------------------------------------------------------------------------
# Outcome record
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    """The result of one baseline run on a single ``(instance, seed)``.

    Attributes:
        method: ``matheuristic`` or ``llm-heuristic``.
        instance_id: The MILP instance name.
        seed: The RNG seed used.
        heuristic: The core that produced the best incumbent (for the LLM
            baseline this is the best of the evolved choices).
        incumbent: Best feasible assignment found, or ``None`` if none was.
        objective: Original-objective value of ``incumbent`` (``nan`` if none).
        improved: Whether the best incumbent strictly beat the warm start.
        feasible: Whether a feasible incumbent was found.
        n_accepted: Number of heuristic applications that strictly improved the
            incumbent (0 or 1 for the matheuristic; per-round count for the LLM).
        n_llm_calls: Number of LLM selection calls (0 for the matheuristic).
        score: The evaluator :class:`~opop.model.state.ScoreRecord`.
        cost: The flat per-row cost dict (see :func:`opop.bench.cost.make_event_cost`).
        time_limit: The per-solve time budget used.
        selection_history: Per-round LLM selection summaries (empty for the
            matheuristic).
    """

    method: str
    instance_id: str
    seed: int
    heuristic: str
    incumbent: dict[str, float] | None
    objective: float
    improved: bool
    feasible: bool
    n_accepted: int
    n_llm_calls: int
    score: ScoreRecord
    cost: dict[str, Any]
    time_limit: float
    selection_history: tuple[dict[str, Any], ...] = ()

    def to_row(self) -> dict[str, Any]:
        """Return the schema-identical ``results.parquet`` row for this outcome."""
        m = self.score.metrics
        row: dict[str, Any] = {
            "instance_id": self.instance_id,
            "method": self.method,
            "seed": int(self.seed),
            "heuristic": self.heuristic,
            "primal_integral": float(m.get("primal_integral", float("nan"))),
            "gap": float(m.get("gap", float("nan"))),
            "time": float(self.cost["total_wall_time"]),
            "solved": bool(m.get("optimal", 0.0)),
            "censored": bool(m.get("censored", 0.0)),
            "time_limit": float(self.time_limit),
            "n_accepted": int(self.n_accepted),
            "n_llm_calls": int(self.n_llm_calls),
        }
        for col in COST_FIELDS:
            row[col] = self.cost[col]
        return row


# ---------------------------------------------------------------------------
# Pure helpers (no solver)
# ---------------------------------------------------------------------------
def _n_discrete(ir: MILP) -> int:
    """Number of BINARY/INTEGER variables in ``ir``."""
    return sum(1 for v in ir.variables if v.vtype in _DISCRETE)


def _objective_value(ir: MILP, assignment: Mapping[str, float]) -> float:
    """Evaluate the IR's original objective at ``assignment`` (includes offset)."""
    total = ir.objective.offset
    for name, coeff in ir.objective.coeffs.items():
        total += coeff * float(assignment[name])
    return total


def _strictly_better(new: float, old: float, sense: ObjSense, tol: float = _OBJ_TOL) -> bool:
    """``True`` iff ``new`` strictly beats ``old`` for the optimisation ``sense``."""
    if sense is ObjSense.MINIMIZE:
        return new < old - tol
    return new > old + tol


def _no_incumbent_primal(sense: ObjSense) -> float:
    """Worst-case primal sentinel for "no incumbent" (``+inf`` MIN / ``-inf`` MAX)."""
    return math.inf if sense is ObjSense.MINIMIZE else -math.inf


def _no_dual(sense: ObjSense) -> float:
    """Unknown-dual sentinel (``-inf`` MIN / ``+inf`` MAX) → gap collapses to 1.0."""
    return -math.inf if sense is ObjSense.MINIMIZE else math.inf


def _finite_or_inf(model: Any, value: float) -> float:
    """Map a SCIP bound (``+-1e20`` sentinel) to a Python float / ``inf``."""
    if model.isInfinity(value):
        return math.inf
    if model.isInfinity(-value):
        return -math.inf
    return float(value)


def _trajectory_trace(
    instance_id: str,
    points: Sequence[tuple[float, float]],
    *,
    dual_bound: float,
    censored: bool,
    status: str,
    solver: str,
) -> SolveTrace:
    """Build a representative :class:`SolveTrace` from an incumbent trajectory.

    ``points`` is a chronological ``(wall_time, objective)`` series; the dual
    bound (a valid bound on the ORIGINAL optimum from the warm-start solve, or a
    non-finite sentinel) is broadcast across the series so the evaluator's gap
    falls back to ``1.0`` when no real bound exists.
    """
    if points:
        times = [float(t) for t, _ in points]
        primal = [float(o) for _, o in points]
        first_feasible = next((t for t, o in points if math.isfinite(o)), math.nan)
    else:
        times = [0.0]
        primal = [math.inf]
        first_feasible = math.nan
    dual_series = [float(dual_bound) for _ in primal]
    return SolveTrace(
        primal_bound_series=primal,
        dual_bound_series=dual_series,
        time_series=times,
        nodes=0,
        lp_iters=0,
        cuts=0,
        first_feasible_time=first_feasible,
        status=status,
        censored=censored,
        memory_peak=math.nan,
        instance_id=instance_id,
        solver=solver,
    )


def _terminal_status(
    *, feasible: bool, censored: bool, objective: float, reference_optimum: float | None
) -> str:
    """Derive a conservative trace status (only claims ``optimal`` if proven)."""
    if not feasible:
        return "no_incumbent"
    if (
        reference_optimum is not None
        and not censored
        and math.isfinite(objective)
        and abs(objective - float(reference_optimum)) <= _OPT_MATCH_TOL
    ):
        return "optimal"
    return "censored" if censored else "feasible"


# ---------------------------------------------------------------------------
# Warm-start incumbent (truncated SCIP "quick heuristic")
# ---------------------------------------------------------------------------
def _quick_incumbent(
    ir: MILP, *, seed: int, time_limit: float, memory_limit_mb: int
) -> tuple[dict[str, float] | None, float, float]:
    """Find a quick feasible incumbent via a first-feasible SCIP solve.

    Solves ``ir`` with ``limits/solutions = 1`` so SCIP stops at the first
    feasible solution its primal heuristics produce — a genuine "quick
    heuristic" warm start. Returns ``(assignment, dual_bound, wall_time)``;
    ``assignment`` is ``None`` when no feasible solution was found within the
    budget, and ``dual_bound`` is the (possibly non-finite) SCIP dual bound at
    the stop point — a valid bound on the ORIGINAL optimum.
    """
    model = to_pyscipopt(ir)
    model.hideOutput()
    model.setIntParam("lp/threads", 1)
    model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/memory", float(memory_limit_mb))
    model.setIntParam("randomization/randomseedshift", int(seed))
    model.setIntParam("limits/solutions", 1)

    start = time.monotonic()
    model.optimize()
    wall = time.monotonic() - start

    dual = _finite_or_inf(model, model.getDualbound())
    sol = model.getBestSol()
    if sol is None:
        return None, dual, wall

    scip_vars = {v.name: v for v in model.getVars()}
    vtype_of = {v.name: v.vtype for v in ir.variables}
    assignment: dict[str, float] = {}
    for name in ir.var_names():
        raw = float(model.getSolVal(sol, scip_vars[name]))
        assignment[name] = float(round(raw)) if vtype_of[name] in _DISCRETE else raw
    return assignment, dual, wall


# ---------------------------------------------------------------------------
# Core dispatch (maps a vocabulary name onto a heuristics core)
# ---------------------------------------------------------------------------
def _run_core(
    ir: MILP,
    incumbent: Mapping[str, float],
    heuristic: str,
    config: Mapping[str, float],
    *,
    seed: int,
    time_limit: float,
    memory_limit_mb: int,
) -> HeuristicResult:
    """Apply one named matheuristic core to ``incumbent`` with a light config."""
    phi = Phi()
    if heuristic == "local_branching":
        n_disc = _n_discrete(ir)
        k = int(config.get("k", min(10, n_disc)))
        k = max(0, min(k, n_disc))
        return local_branching(
            ir, incumbent, k, time_limit, seed, memory_limit_mb=memory_limit_mb, phi=phi
        )
    if heuristic == "rins":
        return rins(ir, incumbent, time_limit, seed, memory_limit_mb=memory_limit_mb, phi=phi)
    if heuristic == "lns":
        destroy_frac = min(1.0, max(0.05, float(config.get("destroy_frac", 0.5))))
        n_iter = max(1, min(50, int(config.get("n_iter", 5))))
        return large_neighborhood_search(
            ir,
            incumbent,
            destroy_frac,
            n_iter,
            time_limit,
            seed,
            memory_limit_mb=memory_limit_mb,
            phi=phi,
        )
    if heuristic == "repair":
        return repair_solution(
            ir, incumbent, time_limit, seed, memory_limit_mb=memory_limit_mb, phi=phi
        )
    raise ValueError(f"unknown heuristic core {heuristic!r}; allowed: {ALLOWED_HEURISTICS}")


def _safe_run_core(
    ir: MILP,
    incumbent: Mapping[str, float],
    heuristic: str,
    config: Mapping[str, float],
    *,
    seed: int,
    time_limit: float,
    memory_limit_mb: int,
) -> HeuristicResult | None:
    """Run a core, returning ``None`` (logged) if it raises (e.g. no discrete vars)."""
    try:
        return _run_core(
            ir,
            incumbent,
            heuristic,
            config,
            seed=seed,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
        )
    except (ValueError, RuntimeError) as exc:
        logger.debug("core %s failed on %s: %s", heuristic, ir.name, exc)
        return None


# ---------------------------------------------------------------------------
# LLM token/cost snapshotting (per-run delta over a possibly-shared tracker)
# ---------------------------------------------------------------------------
def _tracker_snapshot(llm: LLMClient | None) -> tuple[int, int, float]:
    """Read cumulative ``(tokens_in, tokens_out, cost_usd)`` from ``llm.tracker``."""
    if llm is None:
        return (0, 0, 0.0)
    tracker = getattr(llm, "tracker", None)
    if tracker is None:
        return (0, 0, 0.0)
    tokens_in = int(getattr(tracker, "total_tokens_in", 0) or 0)
    tokens_out = int(getattr(tracker, "total_tokens_out", 0) or 0)
    cost_usd = float(getattr(tracker, "total_cost_usd", 0.0) or 0.0)
    return (tokens_in, tokens_out, cost_usd)


# ---------------------------------------------------------------------------
# Baseline 5 — classic matheuristic
# ---------------------------------------------------------------------------
def run_matheuristic_baseline(
    ir: MILP,
    core: str,
    *,
    seed: int,
    time_limit: float,
    initial_incumbent: Mapping[str, float] | None = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    config: Mapping[str, float] | None = None,
    reference_optimum: float | None = None,
) -> BaselineOutcome:
    """Run the classic matheuristic baseline (Baseline 5) on one ``(instance, seed)``.

    Warm-starts a feasible incumbent (a quick first-feasible SCIP solve, unless
    ``initial_incumbent`` is supplied) and applies exactly ONE neighbourhood core
    (``local_branching`` / ``rins`` / ``lns``). No LLM is used, so the LLM cost
    columns are zero. NEVER calls the analyzer / verifier / controller.

    Args:
        ir: The MILP instance.
        core: Which core to apply (a name in :data:`ALLOWED_HEURISTICS`, though
            ``repair`` is unusual standalone — kept for API symmetry).
        seed: RNG seed (determinism).
        time_limit: Per-solve time budget (seconds).
        initial_incumbent: Optional explicit feasible warm start (skips the quick
            SCIP solve); must be feasible for ``ir``.
        memory_limit_mb: Hard memory ceiling for the inner solves.
        config: Optional core configuration (``k`` / ``destroy_frac`` / ``n_iter``).
        reference_optimum: Optional known optimum; enables an honest ``solved``
            flag and a meaningful gap.

    Returns:
        A :class:`BaselineOutcome`.
    """
    return _run_single_core_baseline(
        ir,
        method=MATHEURISTIC_METHOD,
        core=core,
        config=dict(config or {}),
        seed=seed,
        time_limit=time_limit,
        initial_incumbent=initial_incumbent,
        memory_limit_mb=memory_limit_mb,
        reference_optimum=reference_optimum,
    )


def _run_single_core_baseline(
    ir: MILP,
    *,
    method: str,
    core: str,
    config: Mapping[str, float],
    seed: int,
    time_limit: float,
    initial_incumbent: Mapping[str, float] | None,
    memory_limit_mb: int,
    reference_optimum: float | None,
) -> BaselineOutcome:
    """Shared one-core driver (warm start + single core + scoring + cost)."""
    sense = ir.objective.sense
    solver_t = 0.0
    censored = False
    dual_bound = _no_dual(sense)

    incumbent, t_first, solver_t, dual_bound = _warm_start(
        ir,
        initial_incumbent=initial_incumbent,
        seed=seed,
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        sense=sense,
        solver_t=solver_t,
        dual_bound=dual_bound,
    )

    points: list[tuple[float, float]] = []
    if incumbent is not None and t_first > 0.0:
        points.append((0.0, _no_incumbent_primal(sense)))

    best = dict(incumbent) if incumbent is not None else None
    best_obj = _objective_value(ir, incumbent) if incumbent is not None else _no_incumbent_primal(sense)
    if incumbent is not None:
        points.append((t_first, best_obj))

    n_accepted = 0
    cursor = t_first
    if incumbent is not None:
        start = time.monotonic()
        result = _safe_run_core(
            ir,
            best if best is not None else incumbent,
            core,
            config,
            seed=seed,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
        )
        core_wall = time.monotonic() - start
        solver_t += core_wall
        cursor += core_wall
        if result is not None:
            censored = censored or any(t.censored for t in result.traces)
            if (
                result.incumbent is not None
                and result.feasible
                and _strictly_better(result.objective, best_obj, sense)
            ):
                best = dict(result.incumbent)
                best_obj = result.objective
                n_accepted = 1
        points.append((cursor, best_obj))

    return _finalize_outcome(
        ir,
        method=method,
        heuristic=core,
        sense=sense,
        best=best,
        best_obj=best_obj,
        warm_obj=(_objective_value(ir, incumbent) if incumbent is not None else math.nan),
        points=points,
        dual_bound=dual_bound,
        censored=censored,
        solver_t=solver_t,
        proposer_t=0.0,
        n_accepted=n_accepted,
        n_llm_calls=0,
        seed=seed,
        time_limit=time_limit,
        reference_optimum=reference_optimum,
        llm=None,
        token_baseline=(0, 0, 0.0),
        selection_history=(),
    )


# ---------------------------------------------------------------------------
# Baseline 6 — LLM-enhanced CO (selection + evolution)
# ---------------------------------------------------------------------------
def run_llm_enhanced_baseline(
    ir: MILP,
    llm: LLMClient,
    *,
    seed: int,
    time_limit: float,
    n_rounds: int = 2,
    initial_incumbent: Mapping[str, float] | None = None,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    default_heuristic: str = DEFAULT_HEURISTIC,
    reference_optimum: float | None = None,
) -> BaselineOutcome:
    """Run the LLM-enhanced CO baseline (Baseline 6) on one ``(instance, seed)``.

    Reproduces the LLM-LNS / HeurAgenix loop: over ``n_rounds`` the ``llm``
    selects/evolves which core to run next from
    :data:`~opop.experiments.heuristic_selector.ALLOWED_HEURISTICS`, and the
    chosen core is applied to the running incumbent. Per-run LLM token/cost is
    tracked and the LLM selection wall time is booked under ``proposer_time`` so
    the headline ``time`` is honest end-to-end (never solver-only). NEVER calls
    the analyzer / verifier / controller.

    Args:
        ir: The MILP instance.
        llm: The LLM client (a ``FakeLLMClient`` in tests).
        seed: RNG seed (determinism).
        time_limit: Per-solve time budget (seconds).
        n_rounds: Number of select-and-apply rounds (``>= 1``).
        initial_incumbent: Optional explicit feasible warm start.
        memory_limit_mb: Hard memory ceiling for the inner solves.
        default_heuristic: Fallback core when an LLM reply is unusable.
        reference_optimum: Optional known optimum (enables ``solved`` + a gap).

    Returns:
        A :class:`BaselineOutcome` whose ``selection_history`` records each round.
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    sense = ir.objective.sense
    solver_t = 0.0
    proposer_t = 0.0
    censored = False
    dual_bound = _no_dual(sense)
    token_baseline = _tracker_snapshot(llm)

    incumbent, t_first, solver_t, dual_bound = _warm_start(
        ir,
        initial_incumbent=initial_incumbent,
        seed=seed,
        time_limit=time_limit,
        memory_limit_mb=memory_limit_mb,
        sense=sense,
        solver_t=solver_t,
        dual_bound=dual_bound,
    )

    points: list[tuple[float, float]] = []
    if incumbent is not None and t_first > 0.0:
        points.append((0.0, _no_incumbent_primal(sense)))
    best = dict(incumbent) if incumbent is not None else None
    best_obj = _objective_value(ir, incumbent) if incumbent is not None else _no_incumbent_primal(sense)
    warm_obj = best_obj
    if incumbent is not None:
        points.append((t_first, best_obj))

    n_accepted = 0
    n_llm_calls = 0
    cursor = t_first
    best_core = default_heuristic
    history: list[dict[str, Any]] = []

    for round_index in range(n_rounds):
        summary = _instance_summary(ir, best, best_obj, sense, history, round_index=round_index)
        t_llm = time.monotonic()
        choice = select_heuristic(llm, summary, default=default_heuristic)
        proposer_t += time.monotonic() - t_llm
        n_llm_calls += 1
        record: dict[str, Any] = {**choice.to_dict(), "round": round_index, "improved": False}
        history.append(record)

        if best is None:
            break

        start = time.monotonic()
        result = _safe_run_core(
            ir,
            best,
            choice.heuristic,
            choice.config,
            seed=seed + round_index,
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
        )
        core_wall = time.monotonic() - start
        solver_t += core_wall
        cursor += core_wall
        if result is not None:
            censored = censored or any(t.censored for t in result.traces)
            if (
                result.incumbent is not None
                and result.feasible
                and _strictly_better(result.objective, best_obj, sense)
            ):
                best = dict(result.incumbent)
                best_obj = result.objective
                best_core = choice.heuristic
                n_accepted += 1
                record["improved"] = True
        points.append((cursor, best_obj))

    return _finalize_outcome(
        ir,
        method=LLM_HEURISTIC_METHOD,
        heuristic=best_core,
        sense=sense,
        best=best,
        best_obj=best_obj,
        warm_obj=warm_obj,
        points=points,
        dual_bound=dual_bound,
        censored=censored,
        solver_t=solver_t,
        proposer_t=proposer_t,
        n_accepted=n_accepted,
        n_llm_calls=n_llm_calls,
        seed=seed,
        time_limit=time_limit,
        reference_optimum=reference_optimum,
        llm=llm,
        token_baseline=token_baseline,
        selection_history=tuple(history),
    )


# ---------------------------------------------------------------------------
# Shared warm-start + finalisation
# ---------------------------------------------------------------------------
def _warm_start(
    ir: MILP,
    *,
    initial_incumbent: Mapping[str, float] | None,
    seed: int,
    time_limit: float,
    memory_limit_mb: int,
    sense: ObjSense,
    solver_t: float,
    dual_bound: float,
) -> tuple[dict[str, float] | None, float, float, float]:
    """Resolve the warm-start incumbent, returning ``(inc, t_first, solver_t, dual)``."""
    _ = sense
    if initial_incumbent is None:
        inc, dual, wall = _quick_incumbent(
            ir, seed=seed, time_limit=time_limit, memory_limit_mb=memory_limit_mb
        )
        return inc, wall, solver_t + wall, dual
    inc = {str(k): float(v) for k, v in initial_incumbent.items()}
    if not is_solution_feasible(ir, inc):
        raise ValueError("initial_incumbent is infeasible for the given IR")
    return inc, 0.0, solver_t, dual_bound


def _finalize_outcome(
    ir: MILP,
    *,
    method: str,
    heuristic: str,
    sense: ObjSense,
    best: dict[str, float] | None,
    best_obj: float,
    warm_obj: float,
    points: Sequence[tuple[float, float]],
    dual_bound: float,
    censored: bool,
    solver_t: float,
    proposer_t: float,
    n_accepted: int,
    n_llm_calls: int,
    seed: int,
    time_limit: float,
    reference_optimum: float | None,
    llm: LLMClient | None,
    token_baseline: tuple[int, int, float],
    selection_history: tuple[dict[str, Any], ...],
) -> BaselineOutcome:
    """Build the trace, score it, account cost, and assemble the outcome."""
    feasible = best is not None
    objective = best_obj if feasible else math.nan
    status = _terminal_status(
        feasible=feasible,
        censored=censored,
        objective=objective,
        reference_optimum=reference_optimum,
    )
    trace = _trajectory_trace(
        ir.name,
        points,
        dual_bound=dual_bound,
        censored=censored,
        status=status,
        solver=f"opop-{method}",
    )

    eval_start = time.monotonic()
    score = evaluate(trace, reference_optimum=reference_optimum, time_limit=time_limit)
    evaluate_t = time.monotonic() - eval_start

    cur_in, cur_out, cur_cost = _tracker_snapshot(llm)
    d_in = max(0, cur_in - token_baseline[0])
    d_out = max(0, cur_out - token_baseline[1])
    d_cost = max(0.0, cur_cost - token_baseline[2])

    cost = make_event_cost(
        solver_wall_time=solver_t,
        proposer_time=proposer_t,
        evaluate_time=evaluate_t,
        llm_tokens_in=d_in,
        llm_tokens_out=d_out,
        llm_cost_usd=d_cost,
    )

    improved = feasible and (
        not math.isfinite(warm_obj) or _strictly_better(best_obj, warm_obj, sense)
    )

    return BaselineOutcome(
        method=method,
        instance_id=ir.name,
        seed=int(seed),
        heuristic=heuristic,
        incumbent=best,
        objective=objective,
        improved=improved,
        feasible=feasible,
        n_accepted=int(n_accepted),
        n_llm_calls=int(n_llm_calls),
        score=score,
        cost=cost,
        time_limit=float(time_limit),
        selection_history=selection_history,
    )


def _instance_summary(
    ir: MILP,
    incumbent: Mapping[str, float] | None,
    incumbent_obj: float,
    sense: ObjSense,
    history: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
) -> dict[str, Any]:
    """Compact, JSON-serialisable instance + search summary for the LLM prompt."""
    obj: float | None = round(incumbent_obj, 6) if math.isfinite(incumbent_obj) else None
    return {
        "instance_id": ir.name,
        "n_vars": ir.n_vars,
        "n_constraints": ir.n_constraints,
        "n_discrete": _n_discrete(ir),
        "objective_sense": sense.value,
        "has_incumbent": incumbent is not None,
        "incumbent_objective": obj,
        "round": round_index,
        "history": [
            {"heuristic": h.get("heuristic"), "improved": h.get("improved")} for h in history
        ],
    }


# ---------------------------------------------------------------------------
# Harness + results IO
# ---------------------------------------------------------------------------
def run_baseline_suite(
    instances: Sequence[MILP],
    seeds: Sequence[int],
    *,
    llm: LLMClient | None = None,
    cores: Sequence[str] = ("lns",),
    time_limit: float = 5.0,
    n_rounds: int = 2,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    reference_optima: Mapping[str, float] | None = None,
) -> list[BaselineOutcome]:
    """Run both baselines over ``instances`` x ``seeds`` (shared harness).

    For each ``(instance, seed)`` runs the matheuristic baseline once per entry
    in ``cores`` and, when ``llm`` is provided, the LLM-enhanced baseline once.

    Args:
        instances: MILP instances.
        seeds: RNG seeds.
        llm: Optional LLM client; when ``None`` only the matheuristic runs.
        cores: Matheuristic cores to run standalone.
        time_limit: Per-solve time budget.
        n_rounds: Selection rounds for the LLM-enhanced baseline.
        memory_limit_mb: Inner-solve memory ceiling.
        reference_optima: Optional ``instance_id -> known optimum`` mapping.

    Returns:
        A flat list of :class:`BaselineOutcome` (one per produced row).
    """
    outcomes: list[BaselineOutcome] = []
    for ir in instances:
        ref = reference_optima.get(ir.name) if reference_optima else None
        for seed in seeds:
            for core in cores:
                outcomes.append(
                    run_matheuristic_baseline(
                        ir,
                        core,
                        seed=int(seed),
                        time_limit=time_limit,
                        memory_limit_mb=memory_limit_mb,
                        reference_optimum=ref,
                    )
                )
            if llm is not None:
                outcomes.append(
                    run_llm_enhanced_baseline(
                        ir,
                        llm,
                        seed=int(seed),
                        time_limit=time_limit,
                        n_rounds=n_rounds,
                        memory_limit_mb=memory_limit_mb,
                        reference_optimum=ref,
                    )
                )
    return outcomes


def write_results(outcomes: Sequence[BaselineOutcome], out_dir: str | Path) -> Path:
    """Persist outcomes to ``<out_dir>/results.parquet`` (pandas) else JSON fallback.

    Columns are reindexed to the stable :data:`RESULT_COLUMNS` order so the two
    baselines always share an identical schema.
    """
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [o.to_row() for o in outcomes]
    parquet_path = run_dir / "results.parquet"
    try:
        import pandas as pd

        frame = pd.DataFrame(rows).reindex(columns=list(RESULT_COLUMNS))
        frame.to_parquet(parquet_path)
    except ImportError:
        json_path = run_dir / "results.json"
        json_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return json_path
    return parquet_path


# ---------------------------------------------------------------------------
# CLI (offline demo / evidence generator)
# ---------------------------------------------------------------------------
def _demo_instances() -> list[MILP]:
    """A small offline synthetic suite for the CLI demo (no network/files)."""
    from opop.bench.sources.synthetic import generate_knapsack, generate_set_cover

    return [
        generate_knapsack(12, seed=0),
        generate_set_cover(8, 12, 0.4, seed=0),
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI: run both baselines over a small synthetic suite and write results.parquet."""
    parser = argparse.ArgumentParser(
        prog="opop.experiments.baselines_56",
        description="Baselines 5 (classic matheuristic) & 6 (LLM-enhanced CO).",
    )
    parser.add_argument("--out", required=True, type=Path, help="output run directory")
    parser.add_argument("--time-limit", type=float, default=3.0, help="per-solve budget (s)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="RNG seeds")
    parser.add_argument("--rounds", type=int, default=2, help="LLM selection rounds")
    args = parser.parse_args(argv)

    from opop.llm import FakeLLMClient

    demo_reply = (
        '{"heuristic": "lns", "config": {"destroy_frac": 1.0, "n_iter": 1}, '
        + '"rationale": "full reoptimization of the neighbourhood"}'
    )
    llm = FakeLLMClient(response=demo_reply, price_input_1m=0.5, price_output_1m=1.5)
    outcomes = run_baseline_suite(
        _demo_instances(),
        args.seeds,
        llm=llm,
        cores=("lns", "rins", "local_branching"),
        time_limit=args.time_limit,
        n_rounds=args.rounds,
    )
    path = write_results(outcomes, args.out)
    n_math = sum(1 for o in outcomes if o.method == MATHEURISTIC_METHOD)
    n_llm = sum(1 for o in outcomes if o.method == LLM_HEURISTIC_METHOD)
    print(f"baselines_56: wrote {len(outcomes)} rows ({n_math} matheuristic, {n_llm} llm) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
