"""Phase-1 closed loop: the integrator of every Phase-1 module.

:func:`run_loop` drives one full agent loop per iteration:

    controller.ask -> analyzer.analyze -> proposer.propose
        -> [per delta] verify gate -> solver -> evaluator
        -> controller.tell -> incumbent update -> journal

and repeats until the budget (``n_trials`` or an optional wall-clock
``time_budget_s``) is spent or the incumbent stagnates for
``stagnation_rounds`` consecutive iterations.

Verification is a HARD gate: a delta that does not return ``status == "pass"``
(reject / sandbox) or that cannot even be applied is **recorded and skipped —
NEVER solved**. Class-C SCIP-parameter deltas (``set_param``) do not change the
math model; they are routed into ``Phi.p`` and verified as semantic no-ops
(``after_ir == before_ir``) rather than fed through :func:`apply_delta` (which
only knows the IR ops ``rename_var`` / ``add_constraint`` / ``update_metadata``).

Determinism / safety:

* one ``controller.tell`` per iteration (the best reward observed that
  iteration) so the BO posterior advances in lock-step with the loop;
* incumbent quality is monotonic (best-so-far improves or holds);
* per-delta solver/evaluator exceptions are recorded and the loop continues;
* ``KeyboardInterrupt`` finalises the artifacts and returns
  ``stopped_reason="interrupted"`` rather than aborting silently.

Artifacts written to ``out_dir``: ``events.jsonl`` (one record per proposal),
``incumbent.json`` (the running best, rewritten on each improvement and at the
end), and ``result.json`` (the final :class:`RunResult`).

The reward uses :func:`opop.evaluator.evaluator.scalarize` (pure numpy) — this
keeps the orchestrator import free of the controller's torch dependency; it is
numerically identical to ``opop.controller.phase1.coip_reward`` (locked by a
regression test).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from opop.bench.cost import CostAccountant, cost_summary
from opop.evaluator.evaluator import scalarize
from opop.model.ir import MILP, apply_delta
from opop.orchestrator.events import EventWriter, build_event
from opop.orchestrator.repro import finalize_run
from opop.orchestrator.result import Incumbent, RunResult
from opop.verify.certificate import write_report

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from opop.analyzer.report import AnalysisReport
    from opop.config import RunConfig
    from opop.llm.client import LLMClient
    from opop.model.state import Delta, Phi, ProblemState, ScoreRecord, SolveTrace
    from opop.verify.certificate import VerificationReport

__all__ = ["OrchestratorError", "run_loop"]

logger = logging.getLogger(__name__)

#: Strict-improvement tolerance for incumbent / stagnation accounting.
_IMPROVEMENT_TOL: float = 1e-12

# Verify-status / event tags.
_STATUS_PASS = "pass"
_STATUS_APPLY_ERROR = "apply_error"
_STATUS_SOLVE_ERROR = "solve_error"


class OrchestratorError(RuntimeError):
    """Raised for unrecoverable orchestrator wiring errors (e.g. no IR in state)."""


# ---------------------------------------------------------------------------
# Injected-dependency protocols (documentation + structural typing)
# ---------------------------------------------------------------------------
class KernelProto(Protocol):
    """A solver kernel (see :class:`opop.solver.scip.ScipKernel`)."""

    def solve(
        self,
        ir: MILP,
        phi: Phi,
        *,
        time_limit: float,
        memory_limit_mb: int,
        seed: int,
    ) -> SolveTrace: ...


class ProposerProto(Protocol):
    """A candidate-delta proposer (see :func:`opop.proposer.api.propose`)."""

    def __call__(
        self,
        state: ProblemState,
        report: AnalysisReport,
        *,
        llm: LLMClient | None = ...,
        max_deltas: int = ...,
    ) -> list[Delta]: ...


class AnalyzerProto(Protocol):
    """A structural analyzer (see :func:`opop.analyzer.api.analyze`)."""

    def __call__(self, ir: MILP) -> AnalysisReport: ...


class VerifierProto(Protocol):
    """The verification gate (see :func:`opop.verify.gate.verify_delta`)."""

    def __call__(
        self, before_ir: MILP, delta: Delta, after_ir: MILP | None = ...
    ) -> VerificationReport: ...


class EvaluatorProto(Protocol):
    """The metric evaluator (see :func:`opop.evaluator.evaluator.evaluate`)."""

    def __call__(
        self,
        trace: SolveTrace,
        *,
        reference_optimum: float | None = ...,
        time_limit: float | None = ...,
    ) -> ScoreRecord: ...


class ControllerProto(Protocol):
    """An ask-tell controller (see :class:`opop.controller.phase1.Phase1Controller`)."""

    @property
    def n_observed(self) -> int: ...

    def ask(self, candidates: NDArray[Any] | None = ...) -> Phi: ...

    def tell(self, phi: Phi, reward: float) -> None: ...


# ---------------------------------------------------------------------------
# IR resolution
# ---------------------------------------------------------------------------
def _resolve_ir(state: ProblemState) -> MILP:
    """Resolve the working MILP IR carried by ``state``.

    Phase-1 carries the live :class:`MILP` either in
    ``state.symbolic_model_ref`` (its documented "current symbolic model IR"
    slot) or in ``state.budget_state["ir"]`` (a type-clean dict slot, used by
    tests). Raises :class:`OrchestratorError` if neither is present.
    """
    ref: object = state.symbolic_model_ref
    if isinstance(ref, MILP):
        return ref
    stashed = state.budget_state.get("ir")
    if isinstance(stashed, MILP):
        return stashed
    raise OrchestratorError(
        "run_loop requires the working MILP IR in state.symbolic_model_ref or "
        + "state.budget_state['ir']; neither holds a MILP instance"
    )


def _param_from_delta(delta: Delta) -> tuple[str, float] | None:
    """Return ``(key, value)`` for a class-C ``set_param`` delta, else ``None``.

    Imported lazily so the orchestrator does not hard-depend on the proposer
    package at module load; falls back to ``None`` if it is unavailable.
    """
    try:
        from opop.proposer.params import param_from_delta
    except ImportError:  # pragma: no cover - proposer always present in Phase-1
        return None
    return param_from_delta(delta)


def _write_json(path: Path, payload: dict[str, Any] | None) -> None:
    """Write ``payload`` as deterministic, strictly-valid JSON (or ``null``)."""
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_loop(
    state: ProblemState,
    config: RunConfig,
    *,
    kernel: KernelProto,
    proposer: ProposerProto,
    analyzer: AnalyzerProto,
    verifier: VerifierProto,
    evaluator: EvaluatorProto,
    controller: ControllerProto,
    llm: LLMClient | None = None,
    out_dir: str | Path,
    reference_optimum: float | None = None,
    time_budget_s: float | None = None,
    memory_limit_mb: int = 4096,
    max_deltas: int = 5,
    stagnation_rounds: int = 5,
    instance_id: str = "",
) -> RunResult:
    """Drive the Phase-1 closed loop to budget and return a :class:`RunResult`.

    Args:
        state: The aggregate problem state; must carry the working MILP IR
            (see :func:`_resolve_ir`).
        config: Run configuration; ``config.budget.trials`` is the iteration
            budget, ``config.budget.time_limit_sec`` the per-solve limit, and
            ``config.seeds[0]`` the solver seed.
        kernel: Solver kernel (``solve(ir, phi, ...) -> SolveTrace``).
        proposer: Candidate-delta proposer.
        analyzer: Structural analyzer (run ONCE on the fixed base IR).
        verifier: The HARD verification gate.
        evaluator: Trace -> ScoreRecord evaluator.
        controller: Ask-tell BO/random controller.
        llm: Optional LLM client forwarded to the proposer.
        out_dir: Directory for ``events.jsonl`` / ``incumbent.json`` /
            ``result.json``.
        reference_optimum: Known optimum forwarded to the evaluator.
        time_budget_s: Optional wall-clock budget for the whole loop.
        memory_limit_mb: Per-solve memory ceiling forwarded to the kernel.
        max_deltas: Cap on proposed deltas per iteration.
        stagnation_rounds: Stop after this many consecutive non-improving
            iterations.
        instance_id: Identifier for the benchmark instance; forwarded to every
            journal event as ``instance_id``.

    Returns:
        A :class:`RunResult` summarising the run; its artifacts are persisted
        to ``out_dir``.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    events_path = out_path / "events.jsonl"
    incumbent_path = out_path / "incumbent.json"

    base_ir = _resolve_ir(state)
    time_limit = float(config.budget.time_limit_sec)
    n_trials = int(config.budget.trials)
    seed = int(config.seeds[0]) if config.seeds else 0

    # Cost accounting threads per-phase timings + LLM token deltas so every
    # event row carries the full cost column set (solver-only AND end-to-end).
    acct = CostAccountant(tracker=llm.tracker if llm is not None else None)

    # The base IR is fixed across Phase-1 iterations, so one analysis suffices;
    # the proposer is re-asked each iteration (the LLM path may still vary).
    t_analyze = time.monotonic()
    report = analyzer(base_ir)
    acct.record_analyzer(time.monotonic() - t_analyze)

    current_state = state
    best_reward = float("-inf")
    incumbent: Incumbent | None = None
    stagnation = 0
    n_accepted = 0
    n_rejected = 0
    completed_iters = 0
    stopped_reason = "budget"
    start = time.monotonic()

    writer = EventWriter(events_path)
    try:
        for iteration in range(n_trials):
            if time_budget_s is not None and time.monotonic() - start >= time_budget_s:
                stopped_reason = "time_budget"
                break

            t_ask = time.monotonic()
            phi = controller.ask()
            ask_t = time.monotonic() - t_ask

            t_propose = time.monotonic()
            deltas = proposer(current_state, report, llm=llm, max_deltas=max_deltas)
            proposer_t = time.monotonic() - t_propose

            acct.start_iteration(ask_t=ask_t, proposer_t=proposer_t)

            best_before_iter = best_reward
            iter_rewards: list[float] = []

            for delta_idx, delta in enumerate(deltas):
                param = _param_from_delta(delta)

                # Resolve the post-delta IR + effective phi.
                if param is not None:
                    # class-C param delta: math model unchanged, routed to phi.p.
                    key, value = param
                    after_ir = base_ir
                    eff_phi = replace(phi, p={**phi.p, key: value})
                else:
                    try:
                        after_ir = apply_delta(base_ir, delta)
                    except Exception as exc:  # noqa: BLE001 - record + continue
                        logger.warning("delta apply failed (skipped): %s", exc)
                        n_rejected += 1
                        cost = acct.event_cost(verify_t=0.0, solve_t=0.0, eval_t=0.0)
                        writer.append(
                            build_event(
                                iteration=iteration,
                                phi=phi,
                                delta=delta,
                                verify_status=_STATUS_APPLY_ERROR,
                                incumbent_so_far=_incumbent_value(best_reward),
                                reason=f"{type(exc).__name__}: {exc}",
                                instance_id=instance_id,
                                cost=cost,
                            )
                        )
                        continue
                    eff_phi = phi

                # HARD gate — never solve a delta that does not pass.
                t_verify = time.monotonic()
                report_v = verifier(base_ir, delta, after_ir)
                verify_t = time.monotonic() - t_verify
                if report_v.status != _STATUS_PASS:
                    n_rejected += 1
                    cost = acct.event_cost(verify_t=verify_t, solve_t=0.0, eval_t=0.0)
                    writer.append(
                        build_event(
                            iteration=iteration,
                            phi=eff_phi,
                            delta=delta,
                            verify_status=report_v.status,
                            incumbent_so_far=_incumbent_value(best_reward),
                            reason=report_v.reason,
                            instance_id=instance_id,
                            cost=cost,
                        )
                    )
                    continue

                # Passed the HARD gate: persist its certificate BEFORE solving so
                # the run stays auditable even if the solve later errors out.
                write_report(
                    report_v, out_path, filename=f"report_{iteration}_{delta_idx}.json"
                )

                # Passed: solve + evaluate (record + continue on failure).
                solve_t = 0.0
                eval_t = 0.0
                solved_ok = False
                t_phase = time.monotonic()
                try:
                    trace = kernel.solve(
                        after_ir,
                        eff_phi,
                        time_limit=time_limit,
                        memory_limit_mb=memory_limit_mb,
                        seed=seed,
                    )
                    solve_t = time.monotonic() - t_phase
                    solved_ok = True
                    t_phase = time.monotonic()
                    score = evaluator(
                        trace, reference_optimum=reference_optimum, time_limit=time_limit
                    )
                    eval_t = time.monotonic() - t_phase
                except Exception as exc:  # noqa: BLE001 - record + continue
                    logger.warning("solve/evaluate failed (skipped): %s", exc)
                    n_rejected += 1
                    # Charge the elapsed-until-failure honestly: to the evaluator
                    # if the solve had already returned, else to the solver.
                    elapsed = time.monotonic() - t_phase
                    if solved_ok:
                        eval_t = elapsed
                    else:
                        solve_t = elapsed
                    cost = acct.event_cost(verify_t=verify_t, solve_t=solve_t, eval_t=eval_t)
                    writer.append(
                        build_event(
                            iteration=iteration,
                            phi=eff_phi,
                            delta=delta,
                            verify_status=_STATUS_SOLVE_ERROR,
                            incumbent_so_far=_incumbent_value(best_reward),
                            reason=f"{type(exc).__name__}: {exc}",
                            instance_id=instance_id,
                            cost=cost,
                        )
                    )
                    continue

                reward = float(scalarize(score))
                n_accepted += 1
                iter_rewards.append(reward)

                current_state = replace(
                    current_state,
                    formulation_history=[*current_state.formulation_history, delta],
                    solver_trace_history=[*current_state.solver_trace_history, trace],
                )

                if reward > best_reward + _IMPROVEMENT_TOL:
                    best_reward = reward
                    incumbent = Incumbent(
                        phi=eff_phi,
                        score=score,
                        reward=reward,
                        certificate=report_v.to_dict(),
                        delta_target=delta.target,
                        delta_class=delta.declared_class.value,
                        iteration=iteration,
                    )
                    current_state = replace(
                        current_state,
                        incumbent_solution=dict(score.metrics),
                        incumbent_certificate=report_v.to_dict(),
                    )
                    _write_json(incumbent_path, incumbent.to_dict())

                cost = acct.event_cost(verify_t=verify_t, solve_t=solve_t, eval_t=eval_t)
                writer.append(
                    build_event(
                        iteration=iteration,
                        phi=eff_phi,
                        delta=delta,
                        verify_status=_STATUS_PASS,
                        trace=trace,
                        score=score,
                        incumbent_so_far=_incumbent_value(best_reward),
                        reward=reward,
                        reason=report_v.reason,
                        accepted=True,
                        instance_id=instance_id,
                        cost=cost,
                    )
                )

            completed_iters = iteration + 1

            # One tell per iteration: the best reward observed (if any).
            if iter_rewards:
                t_tell = time.monotonic()
                controller.tell(phi, max(iter_rewards))
                acct.record_tell(time.monotonic() - t_tell)

            # Stagnation: count consecutive non-improving iterations.
            if best_reward > best_before_iter + _IMPROVEMENT_TOL:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= stagnation_rounds:
                    stopped_reason = "stagnation"
                    logger.info("stagnation_stop after %d iterations", completed_iters)
                    break
    except KeyboardInterrupt:
        stopped_reason = "interrupted"
        logger.warning("run_loop interrupted; finalising artifacts")
    finally:
        writer.close()
        manifest_path = finalize_run(
            out_path,
            config=config,
            seeds=config.seeds,
            base_ir=base_ir,
            time_limit=time_limit,
            memory_limit=memory_limit_mb,
            reference_optimum=reference_optimum,
        )

    result = RunResult(
        incumbent=incumbent,
        n_iterations=completed_iters,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        events_path=events_path,
        out_dir=out_path,
        stopped_reason=stopped_reason,
        repro_manifest_ref=str(manifest_path),
    )
    _write_json(incumbent_path, incumbent.to_dict() if incumbent is not None else None)

    # RunResult is frozen + out of edit scope, so cost accounting is merged into
    # result.json directly: ``cost_summary`` aggregates the journal rows;
    # ``cost_run_total`` is the accountant total (carries the final tell too).
    result_payload = result.to_dict()
    result_payload["cost_summary"] = cost_summary(_load_events(events_path))
    result_payload["cost_run_total"] = acct.run_summary()
    _write_json(out_path / "result.json", result_payload)
    return result


def _incumbent_value(best_reward: float) -> float | None:
    """Map the running best reward to a journal value (``-inf`` -> ``None``)."""
    return None if best_reward == float("-inf") else best_reward


def _load_events(path: Path) -> list[dict[str, Any]]:
    """Read back the (closed) ``events.jsonl`` journal as a list of records."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records
