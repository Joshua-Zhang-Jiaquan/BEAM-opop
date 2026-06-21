"""Tests for the leakage audit + cost accounting (plan task 19).

Two scientific-integrity instruments are covered:

* **Cost columns** — ``test_cost_columns_complete`` runs a tiny solver-free loop
  (deterministic fakes, mirroring :mod:`tests.orchestrator.test_repro`) and proves
  EVERY ``events.jsonl`` row — passed, gate-rejected, AND apply-errored — carries
  the full :data:`~opop.bench.cost.REQUIRED_COST_COLUMNS` set with
  ``total_wall_time >= solver_wall_time``, and that ``result.json`` reports a cost
  summary whose end-to-end wall time is never below the solver-only wall time.
* **Leakage audit** — a planted ``test`` / ``ood_test`` instance id in a run's
  journal must FAIL the audit (nonzero exit); a dev/validation-only run must PASS
  (exit 0). The ``python -m opop.bench.audit_leakage`` entry point is exercised
  end-to-end via a subprocess so the plan's canonical CLI contract stays locked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import pytest
import yaml

from opop.analyzer.report import AnalysisReport
from opop.bench.audit import AuditError, audit_leakage, main
from opop.bench.cost import REQUIRED_COST_COLUMNS
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
    make_add_constraint_delta,
    make_rename_delta,
)
from opop.model.state import Phi, ProblemState, SolveTrace
from opop.orchestrator import run_loop
from opop.proposer.params import make_param_delta
from opop.verify.certificate import VerificationReport

if TYPE_CHECKING:
    from opop.model.state import Delta


# ── loop builders / fakes (deterministic, solver-free) ───────────────────────


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
    return ProblemState(instance_id="tiny", task_family="MILP", budget_state={"ir": ir})


def _config(*, trials: int, time_limit: float = 5.0) -> RunConfig:
    return RunConfig(seeds=[0], budget=BudgetConfig(trials=trials, time_limit_sec=time_limit))


def _trace(primal: float, *, t: float = 1.0) -> SolveTrace:
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


def _read_events(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@final
class _SeqKernel:
    """Fake kernel returning ``_trace(primals[i])`` for the i-th solve call."""

    solver_name = "fake"

    def __init__(self, primals: list[float], *, t: float = 1.0) -> None:
        self._primals = list(primals)
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
class _MarkerVerifier:
    """Marker-based gate: ``"BAD"`` in ``delta.target`` -> reject, else pass."""

    def __init__(self, reject_marker: str = "BAD") -> None:
        self.marker = reject_marker

    def __call__(
        self, before_ir: MILP, delta: Delta, after_ir: MILP | None = None
    ) -> VerificationReport:
        del before_ir, after_ir
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
    del ir
    return AnalysisReport()


# ── audit fixtures ───────────────────────────────────────────────────────────


def _write_registry(tmp_path: Path) -> Path:
    """Write a registry with free (dev/validation) AND held-out (test/ood) splits."""
    data = {
        "benchmarks": [
            {
                "name": "bench_free",
                "problem_type": "MILP",
                "source": "synthetic",
                "split": {
                    "dev": ["dev_001", "dev_002"],
                    "validation": ["val_001"],
                    "test": [],
                    "ood_test": [],
                },
                "license": "MIT",
                "instance_count": 3,
                "time_limit_sec": 30,
                "baseline_set": "scip_default",
                "leakage_group": "group_free",
                "checksum": "sha256:" + "0" * 64,
                "phase": 1,
                "thesis": "T1",
            },
            {
                "name": "bench_held",
                "problem_type": "MILP",
                "source": "synthetic",
                "split": {
                    "dev": [],
                    "validation": [],
                    "test": ["test_001", "test_002"],
                    "ood_test": ["ood_001"],
                },
                "license": "MIT",
                "instance_count": 3,
                "time_limit_sec": 30,
                "baseline_set": "scip_default",
                "leakage_group": "group_held",
                "checksum": "sha256:" + "0" * 64,
                "phase": 1,
                "thesis": "T1",
            },
        ]
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_events(path: Path, instance_ids: list[str]) -> None:
    """Write a minimal ``events.jsonl`` tagging one row per ``instance_id``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"iter": i, "instance_id": inst, "verify_status": "pass"})
        for i, inst in enumerate(instance_ids)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── tests: cost accounting ─────────────────────────────────────────────────────


def test_cost_columns_complete(tmp_path: Path) -> None:
    """Every event row (pass/reject/apply_error) carries all cost columns; total >= solver."""
    ir = _base_ir()
    good = make_param_delta("separating/gomory/freq", 5.0, rationale="gomory")
    bad = make_add_constraint_delta("badcut", {"x": 1.0}, "<=", 0.0, target="BAD cut")
    broken = make_rename_delta("nonexistent", "z", target="rename ghost")
    kernel = _SeqKernel([12.0, 11.0])

    result = run_loop(
        _state(ir),
        _config(trials=2),
        kernel=kernel,
        proposer=_FakeProposer([good, bad, broken]),
        analyzer=_analyzer,
        verifier=_MarkerVerifier(),
        evaluator=evaluate,
        controller=_FixedController(),
        out_dir=tmp_path,
        reference_optimum=10.0,
        stagnation_rounds=10,
    )

    # 2 iterations x 3 deltas = 6 events (1 pass + 1 reject + 1 apply_error each).
    events = _read_events(tmp_path / "events.jsonl")
    assert len(events) == 6
    assert {"pass", "reject", "apply_error"} <= {e["verify_status"] for e in events}
    assert result.n_accepted == 2
    assert result.n_rejected == 4

    for e in events:
        assert set(REQUIRED_COST_COLUMNS) <= set(e.keys())
        assert e["total_wall_time"] >= e["solver_wall_time"]
        assert e["evaluate_time"] >= 0.0
        assert e["llm_tokens_in"] == 0
        assert e["llm_cost_usd"] == 0.0

    # Only solved (pass) rows may carry positive solver time.
    for e in events:
        if e["verify_status"] != "pass":
            assert e["solver_wall_time"] == 0.0

    res = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    summary = res["cost_summary"]
    assert summary["n_events"] == 6
    assert summary["end_to_end_wall_time"] >= summary["solver_only_wall_time"]
    run_total = res["cost_run_total"]
    assert run_total["total_wall_time"] >= run_total["solver_wall_time"]


# ── tests: leakage audit ───────────────────────────────────────────────────────


def test_audit_flags_planted_leakage(tmp_path: Path) -> None:
    """A test/ood_test instance id in the journal fails the audit with a nonzero exit."""
    registry = _write_registry(tmp_path)
    run_dir = tmp_path / "run_leak"
    _write_events(run_dir / "events.jsonl", ["dev_001", "test_001", "ood_001"])

    report = audit_leakage(run_dir, registry)
    assert report["status"] == "fail"
    assert report["test_instances_used_for_tuning"] == ["test_001"]
    assert report["ood_instances_used_for_tuning"] == ["ood_001"]
    assert report["n_violations"] == 2

    rc = main(["--run", str(run_dir), "--registry", str(registry)])
    assert rc == 1

    audit_json = json.loads((run_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit_json["status"] == "fail"
    assert audit_json["n_violations"] == 2


def test_audit_passes_clean_run(tmp_path: Path) -> None:
    """A dev/validation-only run passes the audit (0 violations, exit 0)."""
    registry = _write_registry(tmp_path)
    run_dir = tmp_path / "run_clean"
    _write_events(run_dir / "events.jsonl", ["dev_001", "dev_002", "val_001"])

    report = audit_leakage(run_dir, registry)
    assert report["status"] == "pass"
    assert report["n_violations"] == 0
    assert report["test_instances_used_for_tuning"] == []
    assert report["ood_instances_used_for_tuning"] == []

    out_path = tmp_path / "custom_audit.json"
    rc = main(["--run", str(run_dir), "--registry", str(registry), "--out", str(out_path)])
    assert rc == 0
    audit_json = json.loads(out_path.read_text(encoding="utf-8"))
    assert audit_json["status"] == "pass"


def test_audit_missing_journal_errors(tmp_path: Path) -> None:
    """A missing events.jsonl is a hard error (AuditError; CLI exit 2)."""
    registry = _write_registry(tmp_path)
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    with pytest.raises(AuditError):
        audit_leakage(run_dir, registry)

    rc = main(["--run", str(run_dir), "--registry", str(registry)])
    assert rc == 2


def test_audit_leakage_cli_entrypoint(tmp_path: Path) -> None:
    """``python -m opop.bench.audit_leakage`` flags planted leakage end-to-end."""
    registry = _write_registry(tmp_path)
    run_dir = tmp_path / "run_cli"
    _write_events(run_dir / "events.jsonl", ["dev_001", "test_002"])

    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opop.bench.audit_leakage",
            "--run",
            str(run_dir),
            "--registry",
            str(registry),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    audit_json = json.loads((run_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    assert audit_json["status"] == "fail"
    assert audit_json["test_instances_used_for_tuning"] == ["test_002"]
