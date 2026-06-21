"""Tests for the reproducibility manifest + strict replay.

Covers the determinism contract enforced by
:mod:`opop.orchestrator.repro` and :mod:`opop.replay`:

* ``test_manifest_has_required_fields`` — a tiny :func:`~opop.orchestrator.loop.run_loop`
  run (deterministic, solver-free fakes) persists a ``repro_manifest.json`` that
  carries every :data:`~opop.orchestrator.repro.REQUIRED_FIELDS`, the nested
  :data:`~opop.orchestrator.repro.REQUIRED_SEED_KEYS`, and
  :data:`~opop.orchestrator.repro.REQUIRED_TOLERANCE_KEYS`.
* ``test_missing_manifest_field_aborts`` — dropping ANY required field makes
  :func:`~opop.orchestrator.repro.validate_manifest` abort with
  :class:`~opop.orchestrator.repro.MissingManifestFieldError` (a run can never
  complete without a full determinism record).
* ``test_strict_replay_reproduces`` — an end-to-end Phase-1 run on a tiny MILP
  with the REAL Phase-1 objects (BO controller / analyzer / proposer / verify
  gate / evaluator / SCIP kernel), then :func:`opop.replay.replay_run` in strict
  mode re-executes it from disk and proves it reproduces (prints ``REPRODUCED``,
  exits ``0``, identical incumbent objective + accepted count).

The fakes mirror :mod:`tests.orchestrator.test_loop` so the manifest tests stay
deterministic and never touch a solver; only the strict-replay integration test
requires SCIP (gated by the ``solver_skip_if_missing`` fixture).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, final

import pytest

from opop.analyzer.report import AnalysisReport
from opop.config import BudgetConfig, RunConfig
from opop.evaluator import evaluate
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)
from opop.model.state import Phi, ProblemState, SolveTrace
from opop.orchestrator import run_loop
from opop.orchestrator.repro import (
    MANIFEST_FILENAME,
    REQUIRED_FIELDS,
    REQUIRED_SEED_KEYS,
    REQUIRED_TOLERANCE_KEYS,
    MissingManifestFieldError,
    build_manifest,
    load_manifest,
    validate_manifest,
)
from opop.proposer.params import make_param_delta
from opop.verify.certificate import VerificationReport

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from opop.model.state import Delta


# ── builders ─────────────────────────────────────────────────────────────────


def _base_ir() -> MILP:
    """A tiny 2-var binary knapsack: ``max x + y s.t. x + y <= 1`` (opt = 1)."""
    return MILP(
        name="tiny_knapsack",
        variables=(
            Variable(name="x", vtype=VarType.BINARY, lower=0.0, upper=1.0),
            Variable(name="y", vtype=VarType.BINARY, lower=0.0, upper=1.0),
        ),
        constraints=(
            LinearConstraint(
                name="c0", coeffs={"x": 1.0, "y": 1.0}, sense=ConstraintSense.LE, rhs=1.0
            ),
        ),
        objective=Objective(coeffs={"x": 1.0, "y": 1.0}, sense=ObjSense.MAXIMIZE, offset=0.0),
    )


def _state(ir: MILP) -> ProblemState:
    """A :class:`ProblemState` carrying the working IR in ``budget_state['ir']``."""
    return ProblemState(instance_id="tiny", task_family="MILP", budget_state={"ir": ir})


def _config(*, trials: int, time_limit: float = 5.0) -> RunConfig:
    return RunConfig(seeds=[0], budget=BudgetConfig(trials=trials, time_limit_sec=time_limit))


def _trace(primal: float, *, t: float = 1.0) -> SolveTrace:
    """A 2-point optimal solve trace pinned at ``primal`` over ``[0, t]``."""
    return SolveTrace(
        primal_bound_series=[primal, primal],
        dual_bound_series=[primal, primal],
        time_series=[0.0, t],
        nodes=1,
        lp_iters=5,
        cuts=0,
        first_feasible_time=0.0,
        status="optimal",
        censored=False,
        memory_peak=1.0,
        instance_id="tiny",
        solver="fake",
    )


# ── fakes (deterministic, solver-free) ────────────────────────────────────────


@final
class _CannedKernel:
    """Fake kernel returning a fixed optimal trace (records every solve call)."""

    solver_name = "fake"

    def __init__(self, primal: float = 12.0) -> None:
        self.primal = primal
        self.calls: list[tuple[MILP, Phi]] = []

    def solve(
        self, ir: MILP, phi: Phi, *, time_limit: float, memory_limit_mb: int, seed: int
    ) -> SolveTrace:
        del time_limit, memory_limit_mb, seed
        self.calls.append((ir, phi))
        return _trace(self.primal)


@final
class _PassVerifier:
    """A gate that passes every delta (no solver needed)."""

    def __call__(
        self, before_ir: MILP, delta: Delta, after_ir: MILP | None = None
    ) -> VerificationReport:
        del before_ir, after_ir
        return VerificationReport(
            status="pass",
            delta_class=delta.declared_class.value,
            feasible_region_integer_preserved=True,
            objective_preserved=True,
            reason="fake -> pass",
        )


@final
class _FakeProposer:
    """Returns a fixed list of deltas every iteration."""

    def __init__(self, deltas: list[Delta]) -> None:
        self._deltas = list(deltas)

    def __call__(
        self, state: ProblemState, report: object, *, llm: object = None, max_deltas: int = 5
    ) -> list[Delta]:
        del state, report, llm, max_deltas
        return list(self._deltas)


@final
class _FixedController:
    """Fixed-Phi ask-tell controller that records every ``tell``."""

    def __init__(self) -> None:
        self._phi = Phi()
        self.tells: list[tuple[Phi, float]] = []

    @property
    def n_observed(self) -> int:
        return len(self.tells)

    def ask(self, candidates: object = None) -> Phi:
        del candidates
        return self._phi

    def tell(self, phi: Phi, reward: float) -> None:
        self.tells.append((phi, reward))


def _analyzer(ir: MILP) -> AnalysisReport:
    """Minimal analyzer stub (the fake proposer ignores the report)."""
    del ir
    return AnalysisReport()


# ── strict-comparison helper ───────────────────────────────────────────────────


def _incumbent_objective(incumbent: Any) -> float | None:
    """Extract ``score.objective`` from a loaded ``incumbent.json`` (``null`` -> ``None``)."""
    if not isinstance(incumbent, dict):
        return None
    score = incumbent.get("score")
    if not isinstance(score, dict):
        return None
    objective = score.get("objective")
    return float(objective) if isinstance(objective, (int, float)) else None


# ── tests ──────────────────────────────────────────────────────────────────────


def test_manifest_has_required_fields(tmp_path: Path) -> None:
    """A finished run persists a manifest carrying every required (sub-)field."""
    ir = _base_ir()
    kernel = _CannedKernel(primal=12.0)
    proposer = _FakeProposer(
        [make_param_delta("separating/gomory/freq", 5.0, rationale="gomory")]
    )

    result = run_loop(
        _state(ir),
        _config(trials=1),
        kernel=kernel,
        proposer=proposer,
        analyzer=_analyzer,
        verifier=_PassVerifier(),
        evaluator=evaluate,
        controller=_FixedController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
    )

    # The manifest exists on disk and is referenced by the run result.
    manifest_path = tmp_path / MANIFEST_FILENAME
    assert manifest_path.is_file()
    assert result.repro_manifest_ref == str(manifest_path)

    manifest = load_manifest(tmp_path)
    # A complete manifest validates cleanly (no MissingManifestFieldError).
    validate_manifest(manifest)

    # Every required top-level field is present.
    for field in REQUIRED_FIELDS:
        assert field in manifest, f"manifest missing required field {field!r}"

    # ALL seeds are recorded (SCIP / numpy / torch / python_random / llm_sampling).
    seeds = manifest["seeds"]
    assert isinstance(seeds, dict)
    for seed_key in REQUIRED_SEED_KEYS:
        assert seed_key in seeds, f"manifest seeds missing {seed_key!r}"

    # Both tolerances are recorded (feasibility / optimality).
    tolerances = manifest["tolerances"]
    assert isinstance(tolerances, dict)
    for tol_key in REQUIRED_TOLERANCE_KEYS:
        assert tol_key in tolerances, f"manifest tolerances missing {tol_key!r}"

    # Threads are pinned to 1 for determinism (validate_manifest enforces this).
    assert manifest["threads"] == 1


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_manifest_field_aborts(field: str) -> None:
    """Dropping ANY required field aborts validation with a clear error."""
    config = _config(trials=1)
    manifest = build_manifest(config=config)

    # Sanity: the freshly built manifest is complete and validates.
    validate_manifest(manifest)

    # Removing a required field must abort (no run completes without a full record).
    del manifest[field]
    with pytest.raises(MissingManifestFieldError):
        validate_manifest(manifest)


def test_strict_replay_reproduces(
    tmp_path: Path,
    solver_skip_if_missing: Callable[[str], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A recorded Phase-1 run strict-replays to an identical incumbent + count."""
    solver_skip_if_missing("scip")

    from opop.analyzer.api import analyze
    from opop.bench.sources.synthetic import generate_knapsack
    from opop.controller.encoder import default_phase1_space
    from opop.controller.phase1 import Phase1Controller
    from opop.proposer.api import propose
    from opop.replay import replay_run
    from opop.solver.scip import ScipKernel
    from opop.verify.gate import verify_delta

    # Tiny deterministic MILP + a 1-trial / 2s budget.
    ir = generate_knapsack(6, seed=0)
    config = RunConfig(seeds=[0], budget=BudgetConfig(trials=1, time_limit_sec=2.0))
    n_trials = int(config.budget.trials)

    # Mirror replay_run's controller construction so the recorded run and its
    # replay are driven by an identical BO controller (same space + seed).
    controller = Phase1Controller.bo(
        default_phase1_space(),
        n_trials=n_trials,
        n_init=min(3, n_trials),
        n_candidates=64,
        time_budget_s=None,
        seed=0,
    )
    state = ProblemState(
        instance_id=ir.name, task_family="MILP", budget_state={"ir": ir}
    )

    result = run_loop(
        state,
        config,
        kernel=ScipKernel(),
        proposer=propose,
        analyzer=analyze,
        verifier=verify_delta,
        evaluator=evaluate,
        controller=controller,
        out_dir=tmp_path,
        reference_optimum=None,
        time_budget_s=None,
        instance_id=ir.name,
    )

    # The loop ran one trial and left a full reproducibility footprint on disk.
    assert result.n_iterations == 1
    assert (tmp_path / MANIFEST_FILENAME).is_file()
    assert (tmp_path / "instance.json").is_file()

    # Strict replay: re-execute entirely from disk and verify reproduction.
    capsys.readouterr()  # discard anything the recorded run printed
    rc = replay_run(tmp_path, strict=True)
    captured = capsys.readouterr()

    assert rc == 0
    assert "REPRODUCED" in captured.out

    # Original vs. replay: the incumbent objective agrees.
    orig_inc = json.loads((tmp_path / "incumbent.json").read_text(encoding="utf-8"))
    replay_inc = json.loads(
        (tmp_path / "replay" / "incumbent.json").read_text(encoding="utf-8")
    )
    assert _incumbent_objective(orig_inc) == _incumbent_objective(replay_inc)

    # Original vs. replay: the accepted-delta count agrees (and matches the run result).
    orig_result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    replay_result = json.loads(
        (tmp_path / "replay" / "result.json").read_text(encoding="utf-8")
    )
    assert orig_result["n_accepted"] == replay_result["n_accepted"]
    assert orig_result["n_accepted"] == result.n_accepted
