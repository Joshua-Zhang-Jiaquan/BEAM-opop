"""Curated SCIP parameter knobs + class-C param deltas for the Phase-1 proposer.

The Phase-1 proposer searches a deliberately NARROW design space. The two
delta families it emits are:

* **class-B valid inequalities** — explicit added constraints, built in
  :mod:`opop.proposer.templates` from analyzer-flagged candidates only.
* **class-C search params** — the SCIP parameter toggles defined here.

A *param delta* never changes the feasible region or the math model; it only
steers SCIP's search path (cut frequency, branching, presolving rounds, gap
limit) and therefore is class-C ("heuristic / search-param") under the
Verification Strategy. Each delta encodes its concrete ``key=value`` change as a
JSON ``{"op": "set_param", ...}`` payload in :attr:`Delta.after_fragment`; the
orchestrator (task 16) routes these into :attr:`Phi.p`, the single parameter
channel the SCIP kernel (task 12) consumes. Param deltas are NOT applied to the
symbolic IR via :func:`opop.model.ir.apply_delta`.

Safety envelope: any ``separating/<name>/...`` knob must name a separator in
:data:`opop.solver.scip.WHITELISTED_SEPARATORS` (every whitelisted separator
emits *globally valid* inequalities). :func:`make_param_delta` rejects a
non-whitelisted separator at construction time (fail-closed), mirroring
``ScipKernel.apply_proposer_hooks`` so an illegal separator can never leak in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from opop.model.state import Delta, DeltaClass
from opop.solver.scip import WHITELISTED_SEPARATORS

__all__ = [
    "CURATED_PARAMS",
    "DECOMP_PARAM_KEY",
    "OP_SET_PARAM",
    "ParamKnob",
    "curated_param_deltas",
    "decomposition_flag_delta",
    "make_param_delta",
    "param_from_delta",
]

#: ``Delta.after_fragment`` op tag identifying a SCIP parameter change. Distinct
#: from the IR ops (``rename_var`` / ``add_constraint`` / ``update_metadata``) in
#: :mod:`opop.model.ir`: a ``set_param`` delta targets ``Phi.p``, never the IR.
OP_SET_PARAM = "set_param"

#: Decomposition flag knob (Phase-1 stub). Phase-1 ``decomposability`` is always
#: ``"NONE"`` so this toggle is dormant — emitted only if a future analyzer
#: reports a non-``NONE`` structure (see :func:`decomposition_flag_delta`). It is
#: a ``decomposition/...`` key (NOT ``separating/...``) so the SCIP kernel passes
#: it through untouched.
DECOMP_PARAM_KEY = "decomposition/applybenders"


# ---------------------------------------------------------------------------
# Curated knob list
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParamKnob:
    """One curated SCIP parameter and the candidate values the proposer may try.

    Attributes:
        key: Full SCIP parameter path (e.g. ``"separating/gomory/freq"``).
        values: Candidate values, each yielding one class-C param delta.
        description: Short human-readable rationale embedded in ``Delta.target``.
    """

    key: str
    values: tuple[float, ...]
    description: str


#: The Phase-1 curated knob list. Separator-frequency knobs name only whitelisted
#: separators (class-B valid-inequality families); the remaining knobs steer
#: branching, presolving, and the gap limit. Ordering is stable and defines the
#: param section of the candidate pool (see :func:`curated_param_deltas`).
CURATED_PARAMS: tuple[ParamKnob, ...] = (
    ParamKnob(
        "separating/gomory/freq",
        (0.0, 5.0),
        "Gomory mixed-integer cut frequency (0=root only, 5=every 5 nodes)",
    ),
    ParamKnob(
        "separating/clique/freq",
        (0.0, 5.0),
        "clique cut frequency (0=root only, 5=every 5 nodes)",
    ),
    ParamKnob(
        "separating/zerohalf/freq",
        (0.0, 5.0),
        "zero-half cut frequency (0=root only, 5=every 5 nodes)",
    ),
    ParamKnob(
        "branching/scorefac",
        (0.0, 0.5),
        "branching score factor (pseudocost vs. inference blend)",
    ),
    ParamKnob(
        "presolving/maxrounds",
        (0.0, 10.0),
        "presolving round cap (0=off, 10=bounded)",
    ),
    ParamKnob(
        "limits/gap",
        (0.0001, 0.01),
        "relative optimality gap stop (0.01%=near-exact, 1%=anytime)",
    ),
)


# ---------------------------------------------------------------------------
# Param-delta constructor / accessor
# ---------------------------------------------------------------------------
def _separator_name(key: str) -> str | None:
    """Return the separator name of a ``separating/<name>/...`` key, else None."""
    if not key.startswith("separating/"):
        return None
    parts = key.split("/")
    return parts[1] if len(parts) >= 2 else ""


def make_param_delta(key: str, value: float, *, rationale: str | None = None) -> Delta:
    """Build a class-C SCIP parameter :class:`Delta` (``key`` set to ``value``).

    The change is encoded as ``{"op": "set_param", "key": key, "value": value}``
    in ``after_fragment``. A ``separating/<name>/...`` key whose ``<name>`` is not
    in :data:`opop.solver.scip.WHITELISTED_SEPARATORS` raises :class:`ValueError`
    (fail-closed) — never silently emitted.
    """
    sep = _separator_name(key)
    if sep is not None and sep not in WHITELISTED_SEPARATORS:
        allowed = sorted(WHITELISTED_SEPARATORS)
        raise ValueError(
            f"separator {sep!r} (param {key!r}) is not class-B whitelisted; allowed: {allowed}"
        )
    payload = {"op": OP_SET_PARAM, "key": key, "value": float(value)}
    return Delta(
        target=rationale or f"set SCIP param {key}={value:g} (class-C search param)",
        after_fragment=json.dumps(payload),
        declared_class=DeltaClass.C,
    )


def param_from_delta(delta: Delta) -> tuple[str, float] | None:
    """Return ``(key, value)`` for a ``set_param`` delta, else ``None``.

    Used by the orchestrator to project class-C param deltas into ``Phi.p``.
    Returns ``None`` for any non-param delta (e.g. a class-B added constraint).
    """
    if not delta.after_fragment:
        return None
    try:
        payload = json.loads(delta.after_fragment)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("op") != OP_SET_PARAM:
        return None
    key = payload.get("key")
    value = payload.get("value")
    if not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return key, float(value)


# ---------------------------------------------------------------------------
# Pool builders
# ---------------------------------------------------------------------------
def curated_param_deltas() -> list[Delta]:
    """Return one class-C delta per ``(knob, value)`` in :data:`CURATED_PARAMS`.

    Order is stable (knob order, then value order) so the param section of the
    candidate pool is deterministic for both LLM selection and the rule-based
    ranker.
    """
    deltas: list[Delta] = []
    for knob in CURATED_PARAMS:
        for value in knob.values:
            rationale = f"SCIP search param {knob.key}={value:g} ({knob.description})"
            deltas.append(make_param_delta(knob.key, value, rationale=rationale))
    return deltas


def decomposition_flag_delta() -> Delta:
    """Return the class-C decomposition-flag toggle (Phase-1 stub).

    Emitted into the candidate pool only when the analyzer reports a non-``NONE``
    decomposability (never in Phase-1). The constructor + wiring exist so Wave-4
    (task 27) can activate it without re-plumbing the proposer.
    """
    return make_param_delta(
        DECOMP_PARAM_KEY,
        1.0,
        rationale=(
            f"decomposition flag (Phase-1 stub) {DECOMP_PARAM_KEY}=1; "
            "dormant unless analyzer detects structure"
        ),
    )
