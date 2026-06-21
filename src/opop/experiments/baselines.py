"""Baseline runners 1--2 for OPOP experiments (plan task 36).

Two baselines that share ONE experiment harness + cost accounting and emit the
SAME ``results.parquet`` schema as :mod:`opop.run`:

1. :class:`DefaultRunner` (baseline 1) — the fixed expert formulation solved
   once with each solver's *default* parameters (SCIP / HiGHS / CP-SAT).
2. :class:`SMACTunedRunner` (baseline 2) — the same formulation with automated
   solver-parameter tuning via SMAC3 ask-tell, reusing the task-28
   :class:`~opop.controller.ladder.SMACSurrogate` under the *same* budget /
   seeds as the opop run.

Both runners route every solve through the *identical* Evaluator metric pipeline
(:func:`opop.evaluator.evaluate` + :func:`opop.evaluator.scalarize`), so the rows
are directly comparable to the opop rows by
:func:`opop.experiments.compare.compare`.  The harness
(:func:`run_baselines`) enforces budget/seed equality against the opop run
(:func:`opop.experiments.fairness.check_budget_fairness`) and refuses to let any
*tuning* runner touch a held-out split.

Schema
------
Each row carries the opop ``results.parquet`` columns
(``instance_id, method, seed, primal_integral, gap, time, solved, censored,
time_limit, n_accepted``) PLUS two cost-accounting columns:

* ``solver_time_sec`` — total measured solver wall-clock for the cell (one solve
  for the default baseline; the sum over all tuning trials for the SMAC baseline).
* ``n_solves`` — number of ``kernel.solve`` invocations (1 vs ``trials``).

so a downstream comparison can cost-normalise the (much cheaper) default baseline
against the (``trials``x more expensive) tuning baseline.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, final, override

from opop.evaluator import evaluate, scalarize
from opop.model.state import Phi

from .fairness import BudgetSpec, assert_tunable_split, check_budget_fairness

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from opop.config import RunConfig
    from opop.model.ir import MILP
    from opop.model.state import ScoreRecord
    from opop.solver.kernel import SolverKernel

__all__ = [
    "CPSAT_PARAM_SPACE",
    "HIGHS_PARAM_SPACE",
    "MEMORY_LIMIT_MB",
    "RESULT_COLUMNS",
    "SCIP_PARAM_SPACE",
    "BaselineRunner",
    "DefaultRunner",
    "ParamSpec",
    "SMACTunedRunner",
    "default_param_space",
    "run_baselines",
    "write_results",
]

#: Per-solve memory ceiling (MiB); mirrors :data:`opop.run.MEMORY_LIMIT_MB`.
MEMORY_LIMIT_MB: int = 4096

#: Reward margin for counting a tuning trial as an incumbent improvement.
_IMPROVE_EPS: float = 1e-12

#: Finite penalty reward fed to SMAC for a non-finite (no-incumbent) trial, so
#: ``cost = -reward`` stays finite and the search steers away from it.
_PENALTY_REWARD: float = -1e18

#: The canonical ``results.parquet`` columns: the opop.run schema (first ten)
#: followed by the two cost-accounting columns. Order is stable for a
#: deterministic frame.
RESULT_COLUMNS: tuple[str, ...] = (
    "instance_id",
    "method",
    "seed",
    "primal_integral",
    "gap",
    "time",
    "solved",
    "censored",
    "time_limit",
    "n_accepted",
    "solver_time_sec",
    "n_solves",
)


def _solver_tag(kernel: SolverKernel) -> str:
    """Normalise a kernel's ``solver_name`` to a ``results.parquet`` method prefix.

    ``SCIP`` -> ``scip``, ``HiGHS`` -> ``highs``, ``CP-SAT`` -> ``cpsat`` (so the
    default tag matches :data:`opop.run.BASELINE_METHOD` == ``"scip-default"``).
    """
    name = getattr(kernel, "solver_name", type(kernel).__name__)
    return str(name).lower().replace("-", "").replace(" ", "")


# --------------------------------------------------------------------------- #
# Tunable solver-parameter spaces (the SMAC search axes; per backend)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable solver parameter mapped onto the SMAC ``[0, 1]`` axis.

    Attributes:
        key: Backend parameter name forwarded via ``phi.p`` to the kernel hook.
        low: Value at unit coordinate ``0.0``.
        high: Value at unit coordinate ``1.0``.
        is_int: Round the decoded value to the nearest integer (count knobs).
    """

    key: str
    low: float
    high: float
    is_int: bool = False

    def decode(self, unit: float) -> float:
        """Map a unit-interval coordinate to the parameter's true value."""
        clamped = min(1.0, max(0.0, float(unit)))
        value = self.low + clamped * (self.high - self.low)
        return float(round(value)) if self.is_int else float(value)


#: SCIP knobs (real param paths; ``separating/gomory`` is class-B whitelisted).
SCIP_PARAM_SPACE: tuple[ParamSpec, ...] = (
    ParamSpec("separating/gomory/freq", 0.0, 10.0, is_int=True),
    ParamSpec("presolving/maxrounds", 0.0, 10.0, is_int=True),
    ParamSpec("branching/scorefactor", 0.0, 1.0),
)

#: HiGHS knobs (all in :data:`opop.solver.highs.HIGHS_WHITELISTED_PARAMS`).
HIGHS_PARAM_SPACE: tuple[ParamSpec, ...] = (
    ParamSpec("mip_heuristic_effort", 0.0, 1.0),
    ParamSpec("mip_rel_gap", 0.0, 0.01),
)

#: CP-SAT knobs (all in :data:`opop.solver.cpsat.KNOWN_CPSAT_PARAMS`).
CPSAT_PARAM_SPACE: tuple[ParamSpec, ...] = (
    ParamSpec("linearization_level", 0.0, 2.0, is_int=True),
    ParamSpec("cp_model_probing_level", 0.0, 2.0, is_int=True),
)

_DEFAULT_PARAM_SPACES: dict[str, tuple[ParamSpec, ...]] = {
    "scip": SCIP_PARAM_SPACE,
    "highs": HIGHS_PARAM_SPACE,
    "cpsat": CPSAT_PARAM_SPACE,
}


def default_param_space(kernel: SolverKernel) -> tuple[ParamSpec, ...]:
    """Return the default SMAC tuning space for ``kernel`` (by ``solver_name``).

    Raises:
        ValueError: If the solver has no built-in tuning space (pass an explicit
            ``param_space=`` to :class:`SMACTunedRunner` instead).
    """
    tag = _solver_tag(kernel)
    space = _DEFAULT_PARAM_SPACES.get(tag)
    if not space:
        raise ValueError(
            f"no default SMAC tuning space for solver {tag!r}; "
            + f"known: {sorted(_DEFAULT_PARAM_SPACES)} — pass param_space= explicitly"
        )
    return space


# --------------------------------------------------------------------------- #
# Result-row assembly (opop.run schema + cost columns)
# --------------------------------------------------------------------------- #


def _result_row(
    *,
    method: str,
    instance_id: str,
    seed: int,
    score: ScoreRecord,
    time_limit: float,
    n_accepted: int,
    solver_time_sec: float,
    n_solves: int,
) -> dict[str, Any]:
    """Assemble one ``results.parquet`` row (opop.run schema + cost columns).

    Reads the SAME ``ScoreRecord.metrics`` keys as :func:`opop.run._baseline_row`
    so the baseline rows are byte-for-byte schema-compatible with the opop rows.
    """
    m = score.metrics
    return {
        "instance_id": instance_id,
        "method": method,
        "seed": int(seed),
        "primal_integral": float(m.get("primal_integral", float("nan"))),
        "gap": float(m.get("gap", 1.0)),
        "time": float(m.get("solve_time", time_limit)),
        "solved": bool(m.get("optimal", 0.0)),
        "censored": bool(m.get("censored", 0.0)),
        "time_limit": float(time_limit),
        "n_accepted": int(n_accepted),
        "solver_time_sec": float(solver_time_sec),
        "n_solves": int(n_solves),
    }


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #


class BaselineRunner:
    """Shared baseline harness: iterate ``(instance, seed)``, build schema rows.

    Subclasses implement :meth:`solve_one` for a single ``(ir, seed)`` cell; the
    base :meth:`run` drives the cartesian product over instances x seeds.  Both
    the fixed-default and SMAC-tuned runners reuse this base so they emit the
    identical row schema and the identical metric pipeline.

    Args:
        kernel: The solver backend (any :class:`~opop.solver.kernel.SolverKernel`).
        method_name: Override the ``method`` tag (defaults to ``<solver>-<variant>``).
        memory_limit_mb: Per-solve memory ceiling (MiB).
    """

    #: Method-name suffix distinguishing the variant ("default" / "smac").
    variant: str = "baseline"
    #: Whether this runner performs hyper-parameter search on the instances.
    is_tuning: bool = False

    def __init__(
        self,
        kernel: SolverKernel,
        *,
        method_name: str | None = None,
        memory_limit_mb: int = MEMORY_LIMIT_MB,
    ) -> None:
        self.kernel: SolverKernel = kernel
        self.memory_limit_mb: int = int(memory_limit_mb)
        self.method_name: str = method_name or f"{_solver_tag(kernel)}-{self.variant}"

    def solve_one(
        self, ir: MILP, seed: int, *, trials: int, time_limit: float
    ) -> dict[str, Any]:
        """Solve one ``(ir, seed)`` cell and return its result row (override)."""
        del ir, seed, trials, time_limit
        raise NotImplementedError

    def run(
        self,
        instances: Sequence[MILP],
        *,
        trials: int,
        time_limit_sec: float,
        seeds: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Run every ``(instance, seed)`` cell; return the unified result rows."""
        rows: list[dict[str, Any]] = []
        for ir in instances:
            for seed in seeds:
                rows.append(
                    self.solve_one(
                        ir, int(seed), trials=int(trials), time_limit=float(time_limit_sec)
                    )
                )
        return rows


@final
class DefaultRunner(BaselineRunner):
    """Baseline 1: solve the fixed formulation once with the solver's defaults.

    A single ``kernel.solve(ir, Phi(), ...)`` per ``(instance, seed)``; no search,
    so ``n_accepted == 0`` and ``n_solves == 1`` (mirroring the ``scip-default``
    baseline row emitted by :mod:`opop.run`).
    """

    variant = "default"
    is_tuning = False

    @override
    def solve_one(
        self, ir: MILP, seed: int, *, trials: int, time_limit: float
    ) -> dict[str, Any]:
        del trials  # the default baseline never searches — one solve only.
        start = time.monotonic()
        trace = self.kernel.solve(
            ir,
            Phi(),
            time_limit=time_limit,
            memory_limit_mb=self.memory_limit_mb,
            seed=int(seed),
        )
        solver_time = time.monotonic() - start
        score = evaluate(trace, time_limit=time_limit)
        return _result_row(
            method=self.method_name,
            instance_id=ir.name,
            seed=int(seed),
            score=score,
            time_limit=time_limit,
            n_accepted=0,
            solver_time_sec=solver_time,
            n_solves=1,
        )


@final
class SMACTunedRunner(BaselineRunner):
    """Baseline 2: tune solver params with SMAC3 ask-tell under the same budget.

    For each ``(instance, seed)`` cell, runs a SMAC3 ask-tell loop
    (:class:`~opop.controller.ladder.SMACSurrogate`) over the solver's
    :class:`ParamSpec` axes for ``trials`` iterations, each solve under the shared
    ``time_limit_sec`` and ``seed``.  The best (highest-reward) trial's metrics
    are reported; ``n_accepted`` counts incumbent improvements, ``n_solves ==
    trials``, and ``solver_time_sec`` is the TOTAL tuning cost.

    ``smac`` is imported lazily inside :meth:`solve_one`, so the default baseline
    never needs the optional dependency.

    Args:
        kernel: The solver backend to tune.
        param_space: Tunable axes (defaults to :func:`default_param_space`).
        method_name: Override the ``method`` tag (defaults to ``<solver>-smac``).
        memory_limit_mb: Per-solve memory ceiling (MiB).
    """

    variant = "smac"
    is_tuning = True

    def __init__(
        self,
        kernel: SolverKernel,
        *,
        param_space: Sequence[ParamSpec] | None = None,
        method_name: str | None = None,
        memory_limit_mb: int = MEMORY_LIMIT_MB,
    ) -> None:
        super().__init__(kernel, method_name=method_name, memory_limit_mb=memory_limit_mb)
        self.param_space: tuple[ParamSpec, ...] = (
            tuple(param_space) if param_space is not None else default_param_space(kernel)
        )

    def _phi_from_unit(self, unit: NDArray[np.float64]) -> Phi:
        """Decode a SMAC ``[0, 1]^d`` vector into a :class:`Phi` with ``p`` set."""
        params = {
            spec.key: spec.decode(float(unit[i])) for i, spec in enumerate(self.param_space)
        }
        return Phi(p=params)

    @override
    def solve_one(
        self, ir: MILP, seed: int, *, trials: int, time_limit: float
    ) -> dict[str, Any]:
        from opop.controller.ladder import SMACSurrogate

        n_trials = max(1, int(trials))
        smac = SMACSurrogate(dim=len(self.param_space), n_trials=n_trials, seed=int(seed))

        best_reward = -math.inf
        best_score: ScoreRecord | None = None
        last_score: ScoreRecord | None = None
        n_accepted = 0
        total_solver_time = 0.0

        for _ in range(n_trials):
            unit = smac.ask()
            phi = self._phi_from_unit(unit)
            start = time.monotonic()
            trace = self.kernel.solve(
                ir,
                phi,
                time_limit=time_limit,
                memory_limit_mb=self.memory_limit_mb,
                seed=int(seed),
            )
            total_solver_time += time.monotonic() - start
            score = evaluate(trace, time_limit=time_limit)
            last_score = score
            reward = scalarize(score)

            # SMAC minimises cost = -reward, so a non-finite reward must be told
            # as a finite penalty (and a censored / no-incumbent trial flagged).
            finite_reward = reward if math.isfinite(reward) else _PENALTY_REWARD
            censored = bool(trace.censored) or not math.isfinite(reward)
            smac.tell(
                unit,
                finite_reward,
                censored=censored,
                time=float(score.metrics.get("solve_time", total_solver_time)),
            )

            if math.isfinite(reward) and (best_score is None or reward > best_reward + _IMPROVE_EPS):
                best_reward = reward
                best_score = score
                n_accepted += 1

        # Fall back to the last trial when no finite reward ever improved the
        # incumbent (e.g. every trial was censored with no usable incumbent).
        chosen = best_score if best_score is not None else last_score
        assert chosen is not None  # n_trials >= 1 guarantees at least one solve

        return _result_row(
            method=self.method_name,
            instance_id=ir.name,
            seed=int(seed),
            score=chosen,
            time_limit=time_limit,
            n_accepted=n_accepted,
            solver_time_sec=total_solver_time,
            n_solves=n_trials,
        )


# --------------------------------------------------------------------------- #
# Harness + persistence
# --------------------------------------------------------------------------- #


def write_results(rows: Sequence[dict[str, Any]], out_dir: str | Path) -> Path:
    """Persist baseline rows to ``<out_dir>/results.parquet`` (pandas), else JSON.

    Mirrors :func:`opop.run._write_results`: writes ``results.parquet`` when
    pandas is importable, otherwise falls back to ``results.json``.  Columns are
    pinned to :data:`RESULT_COLUMNS` for a deterministic frame.
    """
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [{col: r[col] for col in RESULT_COLUMNS} for r in rows]
    parquet_path = run_dir / "results.parquet"
    try:
        import pandas as pd

        pd.DataFrame(records).to_parquet(parquet_path)
    except ImportError:
        json_path = run_dir / "results.json"
        json_path.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return json_path
    return parquet_path


def run_baselines(
    instances: Sequence[MILP],
    runners: Sequence[BaselineRunner],
    *,
    config: RunConfig,
    reference_budget: BudgetSpec | None = None,
    out_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run every baseline runner under ``config``'s budget; emit unified rows.

    Enforces fairness against ``reference_budget`` (the opop run's budget) BEFORE
    any solve, and refuses to let a *tuning* runner touch a held-out split.

    Args:
        instances: The fixed expert formulations to solve.
        runners: The baseline runners (e.g. a :class:`DefaultRunner` and a
            :class:`SMACTunedRunner`).
        config: The baseline run config (its ``budget`` / ``seeds`` / ``split``).
        reference_budget: The opop run's budget; when given, the baseline budget
            must equal it or a :class:`~opop.experiments.fairness.FairnessError`
            is raised.
        out_dir: When given, also write ``results.parquet`` here.

    Returns:
        The combined result rows across all runners (opop.run schema + cost cols).

    Raises:
        FairnessError: On a budget/seed mismatch, or a tuning runner on test/ood.
    """
    spec = BudgetSpec.from_config(config)
    if reference_budget is not None:
        check_budget_fairness(reference_budget, spec)

    split = str(config.split)
    rows: list[dict[str, Any]] = []
    for runner in runners:
        if runner.is_tuning:
            assert_tunable_split(split)
        rows.extend(
            runner.run(
                instances,
                trials=spec.trials,
                time_limit_sec=spec.time_limit_sec,
                seeds=spec.seeds,
            )
        )

    if out_dir is not None:
        write_results(rows, out_dir)
    return rows
