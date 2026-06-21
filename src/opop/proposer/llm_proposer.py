"""LLM-guided selection over the typed candidate pool (with a safety envelope).

The LLM's role is strictly *selection / ranking*. It is shown a fixed, numbered
list of pre-built, typed candidate deltas (class-B analyzer-flagged cuts and
class-C curated params) plus the analysis features that motivate them (LP
objective, integrality gap, fractional pattern, candidate-cut count). It returns
JSON naming the deltas to keep — by integer index or by candidate id. It can
NEVER emit a new delta, raw solver code, or a reformulation: every returned
delta is mapped back to an existing pool entry, and anything that does not map
(out-of-range index, unknown id, free-form text, non-JSON reply) is dropped and
logged. A non-JSON reply (``LLMParseError``) yields an empty selection, which the
top-level proposer treats as "fall back to the rule-based ranker".

This makes the LLM output provably a SUBSET of the legal pool — the core
guarantee that a hallucinated illegal delta can never reach the main path.
"""

from __future__ import annotations

import json
import logging
from typing import cast

from opop.analyzer.report import AnalysisReport
from opop.llm.client import LLMClient, LLMParseError
from opop.model.state import Delta, DeltaClass
from opop.proposer.params import param_from_delta

__all__ = ["build_prompt", "candidate_id", "select"]

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a combinatorial-optimization formulation-search proposer. You SELECT "
    "from a fixed, numbered list of pre-verified, typed candidate deltas. You MUST NOT "
    "invent deltas, emit solver code, or propose reformulations. Respond with JSON only: "
    '{"selected": [<indices best-first>], "rationale": "<short>"}.'
)


def candidate_id(delta: Delta) -> str:
    """Return a stable human/LLM-facing id for a pool ``delta``.

    Class-C param deltas use ``"key=value"``; class-B cut deltas use the
    constraint name; anything else falls back to ``Delta.target``.
    """
    kv = param_from_delta(delta)
    if kv is not None:
        return f"{kv[0]}={kv[1]:g}"
    if delta.after_fragment:
        try:
            payload = json.loads(delta.after_fragment)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("op") == "add_constraint":
            name = payload.get("name")
            if isinstance(name, str):
                return name
    return delta.target


def _kind(delta: Delta) -> str:
    """Coarse kind tag for the prompt (``cut`` / ``param`` / ``other``)."""
    if delta.declared_class is DeltaClass.B:
        return "cut"
    if delta.declared_class is DeltaClass.C:
        return "param"
    return "other"


def build_prompt(report: AnalysisReport, pool: list[Delta], max_deltas: int) -> tuple[str, str]:
    """Return ``(system, user)`` prompt strings describing ``pool`` + analysis."""
    metrics = report.relaxation_metrics
    fractional = list(metrics.fractional_vars)
    lines = [
        "Analysis features:",
        f"- LP relaxation objective: {metrics.lp_obj}",
        f"- Integrality gap estimate: {report.lp_gap}",
        f"- Fractional variables ({metrics.n_fractional}): {fractional}",
        f"- Analyzer-flagged candidate cuts: {len(report.candidate_cuts)}",
        f"- Decomposability: {report.decomposability}",
        "",
        "Candidate deltas (select by index or id):",
    ]
    for i, delta in enumerate(pool):
        head = f"  [{i}] class={delta.declared_class.value} kind={_kind(delta)}"
        lines.append(f"{head} id={candidate_id(delta)!r} :: {delta.target}")
    lines.append("")
    lines.append(f"Select up to {max_deltas} typed deltas, ranked best-first.")
    return _SYSTEM, "\n".join(lines)


def _resolve_index(item: object, pool_size: int, id_map: dict[str, int]) -> int | None:
    """Map one selection entry to a pool index, or ``None`` if illegal.

    Accepts an integer index, a numeric string, ``"#<i>"``, or a candidate id.
    ``bool`` (an ``int`` subclass) is rejected explicitly.
    """
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return item if 0 <= item < pool_size else None
    if isinstance(item, str):
        if item in id_map:
            return id_map[item]
        stripped = item[1:] if item.startswith("#") else item
        if stripped.lstrip("-").isdigit():
            idx = int(stripped)
            return idx if 0 <= idx < pool_size else None
    return None


def select(
    report: AnalysisReport,
    pool: list[Delta],
    llm: LLMClient,
    *,
    max_deltas: int = 5,
) -> list[Delta]:
    """Ask ``llm`` to select from ``pool``; return a legal sub-list (or ``[]``).

    Returns a de-duplicated, order-preserving sub-list of ``pool`` of length
    ``<= max_deltas``. An empty list signals "no usable selection" (non-JSON
    reply, missing ``selected`` list, or every entry illegal) so the caller can
    fall back to the rule-based ranker. Every dropped entry is logged.
    """
    if not pool or max_deltas <= 0:
        return []

    system, user = build_prompt(report, pool, max_deltas)
    try:
        data = llm.chat_json(user, system=system)
    except LLMParseError:
        logger.warning("LLM reply was not JSON; rejecting (no typed selection)")
        return []

    raw = data.get("selected")
    if raw is None:
        raw = data.get("ranking")
    if not isinstance(raw, list):
        logger.warning("LLM reply missing a 'selected'/'ranking' list; rejecting: %r", data)
        return []

    id_map: dict[str, int] = {}
    for i, delta in enumerate(pool):
        id_map.setdefault(candidate_id(delta), i)
        id_map[f"#{i}"] = i

    chosen: list[Delta] = []
    seen: set[int] = set()
    for item in cast("list[object]", raw):
        idx = _resolve_index(item, len(pool), id_map)
        if idx is None:
            logger.warning("dropping illegal/hallucinated LLM selection: %r", item)
            continue
        if idx in seen:
            continue
        seen.add(idx)
        chosen.append(pool[idx])
        if len(chosen) >= max_deltas:
            break
    return chosen
