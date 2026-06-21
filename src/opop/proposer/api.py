"""Top-level Phase-1 proposer entry point: ``propose(state, report, *, llm=None)``.

Given a :class:`opop.model.state.ProblemState` and an
:class:`opop.analyzer.report.AnalysisReport`, return a small set of typed
:class:`opop.model.state.Delta` objects restricted to the Phase-1 design space:

1. **class-C SCIP param changes** — a curated knob list (separator frequency,
   branching, presolving rounds, gap limit; :mod:`opop.proposer.params`).
2. **class-B valid inequalities** — ONLY the analyzer-flagged candidate cuts
   (:mod:`opop.proposer.templates`).
3. **class-C decomposition flag** — a dormant Phase-1 stub, emitted only when the
   analyzer reports a non-``NONE`` decomposability.

Selection / ranking is LLM-guided when an :class:`opop.llm.client.LLMClient` is
supplied (the LLM picks from the typed pool — never raw deltas), with a
deterministic rule-based fallback used when no LLM is given or the LLM returns no
usable selection. A final safety envelope drops any class-D delta, de-duplicates,
and truncates to ``max_deltas`` — so the result is always a typed subset of the
legal candidate pool.
"""

from __future__ import annotations

import logging

from opop.analyzer.report import AnalysisReport
from opop.llm.client import LLMClient
from opop.model.ir import MILP
from opop.model.state import Delta, DeltaClass, ProblemState
from opop.proposer.families import family_deltas
from opop.proposer.llm_proposer import select as _llm_select
from opop.proposer.params import curated_param_deltas, decomposition_flag_delta
from opop.proposer.rule_based import rank as _rule_rank
from opop.proposer.stages import DEFAULT_STAGE, Stage, stage_filter
from opop.proposer.templates import cut_deltas_from_report

__all__ = ["build_candidate_pool", "propose"]

logger = logging.getLogger(__name__)


def _resolve_ir(state: ProblemState) -> MILP | None:
    """Return the symbolic IR carried by ``state``, or ``None`` if absent.

    Mirrors the orchestrator's resolution (task 16): ``symbolic_model_ref`` is
    typed ``str | None`` so tests carry a :class:`MILP` in
    ``budget_state['ir']``; both slots are checked, the typed one first.
    """
    ref: object = state.symbolic_model_ref
    if isinstance(ref, MILP):
        return ref
    carried = state.budget_state.get("ir")
    if isinstance(carried, MILP):
        return carried
    return None


def build_candidate_pool(
    state: ProblemState, report: AnalysisReport, *, allow_families: bool = False
) -> list[Delta]:
    """Return the full legal candidate pool for ``report`` (deterministic order).

    The pool is the ONLY set of deltas the proposer may ever emit:

    * class-B cut deltas, one per ``report.candidate_cuts`` entry (analyzer
      whitelist), in report order;
    * class-C curated param deltas, in :data:`opop.proposer.params.CURATED_PARAMS`
      order;
    * the class-C decomposition-flag stub, appended ONLY when
      ``report.decomposability != "NONE"`` (never in Phase-1);
    * when ``allow_families`` and ``state`` carries a recognised IR (e.g. a
      routing TSP), class-A/B formulation-family deltas
      (:func:`opop.proposer.families.family_deltas`), appended last.

    The default (``allow_families=False``) is byte-for-byte the Phase-1 pool, so
    existing callers are unaffected. ``state`` is otherwise accepted for
    interface symmetry; Phase-1 ranking is driven by ``report`` alone.
    """
    pool: list[Delta] = cut_deltas_from_report(report)
    pool.extend(curated_param_deltas())
    if report.decomposability != "NONE":
        pool.append(decomposition_flag_delta())
    if allow_families:
        ir = _resolve_ir(state)
        if ir is not None:
            pool.extend(family_deltas(ir))
    return pool


def _signature(delta: Delta) -> tuple[str, str, str | None]:
    """Identity signature used to de-duplicate deltas (class, target, payload)."""
    return (delta.declared_class.value, delta.target, delta.after_fragment)


def _finalize(deltas: list[Delta], *, max_deltas: int) -> list[Delta]:
    """Apply the safety envelope: drop class-D, de-duplicate, truncate.

    Class-D deltas must NEVER reach the main path; any that appear (they cannot,
    by construction) are dropped with a warning. Order is preserved.
    """
    out: list[Delta] = []
    seen: set[tuple[str, str, str | None]] = set()
    for delta in deltas:
        if delta.declared_class is DeltaClass.D:
            logger.warning("dropping class-D delta from main path: %s", delta.target)
            continue
        sig = _signature(delta)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(delta)
        if len(out) >= max_deltas:
            break
    return out


def propose(
    state: ProblemState,
    report: AnalysisReport,
    *,
    llm: LLMClient | None = None,
    max_deltas: int = 5,
    stage: Stage | str | int = DEFAULT_STAGE,
    allow_families: bool = False,
) -> list[Delta]:
    """Propose <= ``max_deltas`` typed deltas for ``state`` / ``report``.

    Builds the legal candidate pool (optionally including class-A/B
    formulation-family deltas when ``allow_families`` and ``state`` carries a
    recognised IR), restricts it to the staged search space ``stage`` (S0–S4 —
    see :mod:`opop.proposer.stages`), then:

    * if ``llm`` is given, asks it to select/rank from the stage-legal pool
      (typed templates only); a hallucinated / illegal selection is dropped and
      logged, and an empty usable selection falls back to the rule-based ranker;
    * otherwise (or on fallback) ranks deterministically by integrality gap
      (cuts first) plus a few curated param variations.

    Staging is applied to the POOL *before* selection, so the LLM / ranker only
    ever sees stage-legal candidates and the budget is spent on legal deltas
    (e.g. at ``S1`` no ``formulation`` delta is offered, hence none can be
    emitted). ``stage`` defaults to :data:`opop.proposer.stages.DEFAULT_STAGE`
    (``S4`` — the full ladder) so callers that do not opt into staging see the
    complete legal design space.

    The returned list is a typed subset of the pool — every delta carries a
    declared class in ``{A, B, C}``; class-D never enters the main path.
    """
    resolved_stage = stage
    pool = build_candidate_pool(state, report, allow_families=allow_families)
    pool = stage_filter(pool, resolved_stage)
    if not pool:
        return []

    if llm is not None:
        selected = _llm_select(report, pool, llm, max_deltas=max_deltas)
        if selected:
            return _finalize(selected, max_deltas=max_deltas)
        logger.warning("LLM produced no usable selection; using rule-based fallback")

    ranked = _rule_rank(report, pool, max_deltas=max_deltas)
    return _finalize(ranked, max_deltas=max_deltas)
