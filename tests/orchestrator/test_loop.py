"""Tests for the Phase-1 orchestrator closed loop (:func:`opop.orchestrator.run_loop`).

All fakes are deterministic and network/solver-free:

* ``SeqKernel`` — a fake :class:`~opop.solver.kernel.SolverKernel` returning a
  canned :class:`SolveTrace` per call (and recording ``(ir, phi)`` so we can
  prove a gate-failed delta NEVER reaches the solver).
* ``FakeVerifier`` — a marker-based gate: any delta whose ``target`` contains
  ``"BAD"`` is rejected; everything else passes.
* ``FakeProposer`` / ``FakeController`` — canned deltas / fixed ask-tell.

The real :func:`opop.evaluator.evaluate` (pure numpy) scores the canned traces,
and the real :func:`opop.verify.gate.verify_delta` is exercised in the
integration test (class-C param deltas certify as no-ops without a solver).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import pytest

from opop.analyzer.report import AnalysisReport
from opop.config import BudgetConfig, RunConfig
from opop.evaluator import evaluate
from opop.llm import FakeLLMClient
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
    make_add_constraint_delta,
)
from opop.model.state import Phi, ProblemState, SolveTrace
from opop.orchestrator import OrchestratorError, run_loop
from opop.proposer.params import make_param_delta
from opop.verify.certificate import VerificationReport

if TYPE_CHECKING:
    from opop.model.state import Delta


# ── fixtures / builders ──────────────────────────────────────────────────────


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


def _trace(primal: float, *, t: float = 1.0, status: str = "optimal", nodes: int = 1) -> SolveTrace:
    """A 2-point solve trace pinned at ``primal`` over ``[0, t]``."""
    return SolveTrace(
        primal_bound_series=[primal, primal],
        dual_bound_series=[primal, primal],
        time_series=[0.0, t],
        nodes=nodes,
        lp_iters=5,
        cuts=0,
        first_feasible_time=0.0,
        status=status,
        censored=False,
        memory_peak=1.0,
        instance_id="tiny",
        solver="fake",
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ── fakes ────────────────────────────────────────────────────────────────────


@final
class SeqKernel:
    """Fake kernel returning ``_trace(primals[i])`` for the i-th solve call."""

    solver_name = "fake"

    def __init__(self, primals: list[float], *, ref: float, t: float = 1.0) -> None:
        self._primals = list(primals)
        self.ref = ref
        self.t = t
        self.calls: list[tuple[MILP, Phi]] = []

    def solve(
        self, ir: MILP, phi: Phi, *, time_limit: float, memory_limit_mb: int, seed: int
    ) -> SolveTrace:
        del time_limit, memory_limit_mb, seed
        i = len(self.calls)
        self.calls.append((ir, phi))
        return _trace(self._primals[min(i, len(self._primals) - 1)], t=self.t)


@final
class FakeVerifier:
    """Marker-based gate: ``"BAD"`` in ``delta.target`` -> reject, else pass."""

    def __init__(self, reject_marker: str = "BAD") -> None:
        self.marker = reject_marker
        self.calls: list[Delta] = []

    def __call__(
        self, before_ir: MILP, delta: Delta, after_ir: MILP | None = None
    ) -> VerificationReport:
        del before_ir, after_ir
        self.calls.append(delta)
        cls = delta.declared_class.value
        if self.marker in delta.target:
            return VerificationReport(
                status="reject",
                delta_class=cls,
                feasible_region_integer_preserved=False,
                objective_preserved=True,
                counterexample=None,
                reason=f"marker {self.marker!r} -> reject",
            )
        return VerificationReport(
            status="pass",
            delta_class=cls,
            feasible_region_integer_preserved=True,
            objective_preserved=True,
            reason="marker -> pass",
        )


@final
class FakeProposer:
    """Returns a fixed list of deltas every iteration."""

    def __init__(self, deltas: list[Delta]) -> None:
        self._deltas = list(deltas)
        self.calls = 0

    def __call__(
        self, state: ProblemState, report: object, *, llm: object = None, max_deltas: int = 5
    ) -> list[Delta]:
        del state, report, llm, max_deltas
        self.calls += 1
        return list(self._deltas)


@final
class FakeController:
    """Fixed-Phi ask-tell controller that records every ``tell``."""

    def __init__(self, phi: Phi | None = None) -> None:
        self._phi = phi if phi is not None else Phi()
        self.asks = 0
        self.tells: list[tuple[Phi, float]] = []

    @property
    def n_observed(self) -> int:
        return len(self.tells)

    def ask(self, candidates: object = None) -> Phi:
        del candidates
        self.asks += 1
        return self._phi

    def tell(self, phi: Phi, reward: float) -> None:
        self.tells.append((phi, reward))


def _analyzer(ir: MILP) -> AnalysisReport:
    """Minimal analyzer stub (the fake proposer ignores the report)."""
    del ir
    return AnalysisReport()


# ── tests ────────────────────────────────────────────────────────────────────


def test_loop_runs_n_iters(tmp_path: Path) -> None:
    """The loop executes exactly ``n_trials`` iterations with fakes."""
    ir = _base_ir()
    kernel = SeqKernel([14.0, 12.0, 11.0, 10.0], ref=10.0)
    proposer = FakeProposer([make_param_delta("separating/gomory/freq", 5.0, rationale="gomory")])
    controller = FakeController()

    result = run_loop(
        _state(ir),
        _config(trials=4),
        kernel=kernel,
        proposer=proposer,
        analyzer=_analyzer,
        verifier=FakeVerifier(),
        evaluator=evaluate,
        controller=controller,
        out_dir=tmp_path,
        reference_optimum=10.0,
        stagnation_rounds=10,
    )

    assert result.n_iterations == 4
    assert result.n_accepted == 4
    assert result.n_rejected == 0
    assert result.stopped_reason == "budget"
    assert len(kernel.calls) == 4
    assert len(controller.tells) == 4
    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == 4
    assert all(e["verify_status"] == "pass" for e in events)


def test_gate_failed_delta_never_solved(tmp_path: Path) -> None:
    """A gate-rejected delta is journalled but NEVER reaches the solver."""
    ir = _base_ir()
    good = make_param_delta("separating/gomory/freq", 5.0, rationale="good gomory param")
    bad = make_add_constraint_delta(
        "badcut", {"x": 1.0}, "<=", 0.0, target="BAD cut removes a feasible point"
    )
    kernel = SeqKernel([10.0], ref=10.0)

    result = run_loop(
        _state(ir),
        _config(trials=1),
        kernel=kernel,
        proposer=FakeProposer([good, bad]),
        analyzer=_analyzer,
        verifier=FakeVerifier(reject_marker="BAD"),
        evaluator=evaluate,
        controller=FakeController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
    )

    # Solver invoked exactly once — for the good (param) delta only.
    assert len(kernel.calls) == 1
    solved_ir, _phi = kernel.calls[0]
    assert solved_ir is ir  # a param delta leaves the IR unchanged
    assert all(c.name != "badcut" for c in solved_ir.constraints)
    assert result.n_accepted == 1
    assert result.n_rejected == 1

    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == 2
    bad_events = [e for e in events if "BAD" in str(e["delta_target"])]
    assert len(bad_events) == 1
    assert bad_events[0]["verify_status"] == "reject"
    assert bad_events[0]["accepted"] is False
    assert bad_events[0]["trace_summary"] is None
    good_events = [e for e in events if e["verify_status"] == "pass"]
    assert len(good_events) == 1
    assert good_events[0]["accepted"] is True
    assert good_events[0]["trace_summary"] is not None


def test_events_jsonl_well_formed(tmp_path: Path) -> None:
    """Each journal line is valid JSON with the required schema; incumbent rises."""
    ir = _base_ir()
    kernel = SeqKernel([14.0, 12.0, 11.0, 10.0], ref=10.0)
    proposer = FakeProposer([make_param_delta("separating/clique/freq", 5.0, rationale="clique")])

    result = run_loop(
        _state(ir),
        _config(trials=4),
        kernel=kernel,
        proposer=proposer,
        analyzer=_analyzer,
        verifier=FakeVerifier(),
        evaluator=evaluate,
        controller=FakeController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
        stagnation_rounds=10,
    )

    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == 4
    required = {
        "iter",
        "phi",
        "delta_target",
        "delta_class",
        "verify_status",
        "trace_summary",
        "score",
        "incumbent_so_far",
    }
    for e in events:
        assert required <= set(e.keys())
        assert e["delta_class"] == "C"
        assert isinstance(e["phi"], dict)

    # incumbent_so_far: None-then-finite, non-decreasing among finite values.
    incs = [e["incumbent_so_far"] for e in events]
    finite = [v for v in incs if v is not None]
    assert finite == sorted(finite)

    res = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert res["n_iterations"] == 4
    assert res["stopped_reason"] == "budget"
    inc = json.loads((tmp_path / "incumbent.json").read_text(encoding="utf-8"))
    assert inc is not None
    assert result.incumbent is not None


def test_incumbent_monotonic(tmp_path: Path) -> None:
    """The running incumbent reward improves or holds (never regresses)."""
    ir = _base_ir()
    # Improving then flat: rewards rise to a plateau, never fall.
    kernel = SeqKernel([14.0, 12.0, 10.0, 10.0, 10.0], ref=10.0)
    proposer = FakeProposer([make_param_delta("separating/gomory/freq", 0.0, rationale="gomory0")])

    result = run_loop(
        _state(ir),
        _config(trials=5),
        kernel=kernel,
        proposer=proposer,
        analyzer=_analyzer,
        verifier=FakeVerifier(),
        evaluator=evaluate,
        controller=FakeController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
        stagnation_rounds=10,
    )

    events = _read_events(tmp_path / "events.jsonl")
    rewards = [e["reward"] for e in events if e["reward"] is not None]
    incs = [e["incumbent_so_far"] for e in events if e["incumbent_so_far"] is not None]
    assert incs == sorted(incs)
    assert result.incumbent is not None
    assert result.incumbent.reward == pytest.approx(max(rewards))


def test_stagnation_stops_early(tmp_path: Path) -> None:
    """A flat reward stream stops the loop after ``stagnation_rounds``."""
    ir = _base_ir()
    kernel = SeqKernel([12.0], ref=10.0)  # constant primal -> flat reward
    proposer = FakeProposer([make_param_delta("separating/zerohalf/freq", 5.0, rationale="zh")])

    result = run_loop(
        _state(ir),
        _config(trials=20),
        kernel=kernel,
        proposer=proposer,
        analyzer=_analyzer,
        verifier=FakeVerifier(),
        evaluator=evaluate,
        controller=FakeController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
        stagnation_rounds=3,
    )

    assert result.stopped_reason == "stagnation"
    assert result.n_iterations < 20
    # iter 0 improves (from -inf); iters 1-3 are flat -> stop at 3 stale rounds.
    assert result.n_iterations == 4


def test_apply_error_recorded_not_solved(tmp_path: Path) -> None:
    """A delta that cannot be applied is journalled and never solved."""
    from opop.model.ir import make_rename_delta

    ir = _base_ir()
    # Renaming a non-existent variable fails inside apply_delta.
    broken = make_rename_delta("nonexistent", "z", target="rename ghost")
    kernel = SeqKernel([10.0], ref=10.0)

    result = run_loop(
        _state(ir),
        _config(trials=1),
        kernel=kernel,
        proposer=FakeProposer([broken]),
        analyzer=_analyzer,
        verifier=FakeVerifier(),
        evaluator=evaluate,
        controller=FakeController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
    )

    assert len(kernel.calls) == 0  # never solved
    assert result.n_accepted == 0
    assert result.n_rejected == 1
    assert result.incumbent is None
    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == 1
    assert events[0]["verify_status"] == "apply_error"
    assert events[0]["accepted"] is False


def test_missing_ir_raises(tmp_path: Path) -> None:
    """A state with no working IR fails loudly (no silent success)."""
    state = ProblemState(instance_id="no-ir")
    with pytest.raises(OrchestratorError):
        run_loop(
            state,
            _config(trials=1),
            kernel=SeqKernel([10.0], ref=10.0),
            proposer=FakeProposer([]),
            analyzer=_analyzer,
            verifier=FakeVerifier(),
            evaluator=evaluate,
            controller=FakeController(),
            out_dir=tmp_path,
        )


def test_integration_real_controller_and_verifier(tmp_path: Path) -> None:
    """Real Phase1Controller + real verify_delta certify class-C param no-ops."""
    from opop.controller.encoder import default_phase1_space
    from opop.controller.phase1 import Phase1Controller
    from opop.verify.gate import verify_delta

    ir = _base_ir()
    controller = Phase1Controller.random(default_phase1_space(), n_trials=3, seed=0)
    kernel = SeqKernel([12.0, 11.0, 10.0], ref=10.0)
    proposer = FakeProposer([make_param_delta("separating/gomory/freq", 5.0, rationale="gomory")])

    result = run_loop(
        _state(ir),
        _config(trials=3),
        kernel=kernel,
        proposer=proposer,
        analyzer=_analyzer,
        verifier=verify_delta,
        evaluator=evaluate,
        controller=controller,
        out_dir=tmp_path,
        reference_optimum=10.0,
        stagnation_rounds=10,
        llm=FakeLLMClient(response="{}"),
    )

    assert result.n_iterations == 3
    assert result.n_accepted == 3
    assert result.incumbent is not None
    events = _read_events(tmp_path / "events.jsonl")
    assert all(e["verify_status"] == "pass" for e in events)
    # The real controller observed every (asked) reward.
    assert controller.n_observed == 3
