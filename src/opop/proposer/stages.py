"""Staged search spaces S0–S4 for the OPOP proposer (credit-assignment ladder).

The proposer's design space is unlocked in monotonically growing *stages* so the
ablation matrix can isolate where a method's wins come from (params vs. cuts vs.
heuristics vs. formulation vs. multi-kernel). The ladder, per the plan:

    S0 params → S1 +safe cuts → S2 +heuristics
       → S3 +formulation/decomposition → S4 +multi-kernel/MF/transfer

Each rung is a SUPERSET of the previous one (``allowed_kinds`` is non-decreasing
in the stage index), so a delta legal at stage ``k`` is legal at every ``k' > k``.

Terminology — *kind* vs. *class*
--------------------------------
A delta carries a verification ``DeltaClass`` (A/B/C/D — see
:mod:`opop.verify.gate`). Staging is an ORTHOGONAL axis: it gates a delta by its
*kind* — what design lever it pulls — not by its verification class. The six
kinds are:

* ``param``        — a class-C SCIP search knob (branching/presolving/limits, or a
  separator-frequency knob). Available from **S0**.
* ``cut``          — a class-B analyzer-flagged valid inequality ("safe cut").
  Available from **S1**.
* ``heuristic``    — a class-C primal-heuristic knob (``heuristics/...``).
  Available from **S2**.
* ``formulation``  — a class-A/B formulation-family reformulation
  (:mod:`opop.proposer.families`). Available from **S3**.
* ``decomposition``— a Benders / Dantzig–Wolfe decomposition lever
  (``decomposition/...``). Available from **S3**.
* ``multikernel``  — multi-kernel / multi-fidelity / transfer levers.
  Available from **S4**.

:func:`delta_kind` recovers a delta's kind (an explicit ``"kind"`` tag in the
JSON payload wins; otherwise it is inferred from the op + parameter key), and
:func:`stage_filter` drops any delta whose kind is not yet unlocked at the given
stage. :func:`stage_space` restricts an abstract *space* (a set of available
kinds, or a list of deltas) to the kinds a stage permits — the search-space view
of the same gate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import IntEnum

from opop.model.state import Delta, DeltaClass

__all__ = [
    "ALL_KINDS",
    "KIND_CUT",
    "KIND_DECOMPOSITION",
    "KIND_FORMULATION",
    "KIND_HEURISTIC",
    "KIND_MIN_STAGE",
    "KIND_MULTIKERNEL",
    "KIND_PARAM",
    "Stage",
    "allowed_kinds",
    "delta_kind",
    "parse_stage",
    "stage_allows",
    "stage_filter",
    "stage_space",
]


# ---------------------------------------------------------------------------
# Stage ladder
# ---------------------------------------------------------------------------
class Stage(IntEnum):
    """The S0–S4 staged search spaces (an ``IntEnum`` so ``S1 < S3`` works).

    Higher stages unlock strictly more delta kinds; see :data:`KIND_MIN_STAGE`.
    """

    S0 = 0  #: params only
    S1 = 1  #: + safe cuts
    S2 = 2  #: + heuristics
    S3 = 3  #: + formulation / decomposition
    S4 = 4  #: + multi-kernel / multi-fidelity / transfer


#: Default stage for the proposer: the full ladder (S4) so callers that do not
#: opt into staging see the complete legal design space (backward-compatible).
DEFAULT_STAGE: Stage = Stage.S4


# ---------------------------------------------------------------------------
# Delta kinds + the stage at which each first becomes legal
# ---------------------------------------------------------------------------
KIND_PARAM = "param"  #: class-C SCIP search knob (S0+)
KIND_CUT = "cut"  #: class-B analyzer-flagged valid inequality (S1+)
KIND_HEURISTIC = "heuristic"  #: class-C primal-heuristic knob (S2+)
KIND_FORMULATION = "formulation"  #: class-A/B formulation reformulation (S3+)
KIND_DECOMPOSITION = "decomposition"  #: Benders / DW decomposition lever (S3+)
KIND_MULTIKERNEL = "multikernel"  #: multi-kernel / multi-fidelity / transfer (S4+)

#: The first stage at which each kind becomes legal. Monotone: ``allowed_kinds``
#: is non-decreasing in the stage index because this map's values define a
#: lower-bound gate (``min_stage <= stage``).
KIND_MIN_STAGE: dict[str, Stage] = {
    KIND_PARAM: Stage.S0,
    KIND_CUT: Stage.S1,
    KIND_HEURISTIC: Stage.S2,
    KIND_FORMULATION: Stage.S3,
    KIND_DECOMPOSITION: Stage.S3,
    KIND_MULTIKERNEL: Stage.S4,
}

#: All recognised delta kinds.
ALL_KINDS: frozenset[str] = frozenset(KIND_MIN_STAGE)


# ---------------------------------------------------------------------------
# Stage parsing / queries
# ---------------------------------------------------------------------------
def parse_stage(value: Stage | str | int) -> Stage:
    """Coerce ``value`` to a :class:`Stage` (accepts ``Stage`` / ``"S1"`` / ``1``).

    Raises :class:`ValueError` for an unknown stage name or out-of-range index.
    """
    if isinstance(value, Stage):
        return value
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly.
        raise ValueError(f"invalid stage: {value!r}")
    if isinstance(value, int):
        try:
            return Stage(value)
        except ValueError as exc:
            raise ValueError(f"stage index out of range: {value!r}") from exc
    name = value.strip().upper()
    try:
        return Stage[name]
    except KeyError as exc:
        valid = [s.name for s in Stage]
        raise ValueError(f"unknown stage {value!r}; valid: {valid}") from exc


def allowed_kinds(stage: Stage | str | int) -> frozenset[str]:
    """Return the set of delta kinds legal at ``stage`` (cumulative)."""
    st = parse_stage(stage)
    return frozenset(k for k, min_st in KIND_MIN_STAGE.items() if min_st <= st)


def stage_allows(stage: Stage | str | int, kind: str) -> bool:
    """Return ``True`` iff a delta of ``kind`` is legal at ``stage``.

    An unknown ``kind`` is conservatively gated to :attr:`Stage.S4` (fail-safe:
    an unrecognised lever is treated as the most advanced rung, never leaking
    into an early-stage ablation).
    """
    st = parse_stage(stage)
    return KIND_MIN_STAGE.get(kind, Stage.S4) <= st


# ---------------------------------------------------------------------------
# Delta-kind classification
# ---------------------------------------------------------------------------
def _payload(delta: Delta) -> dict[str, object] | None:
    """Return the decoded ``after_fragment`` JSON object, or ``None``."""
    if not delta.after_fragment:
        return None
    try:
        obj = json.loads(delta.after_fragment)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def delta_kind(delta: Delta) -> str:
    """Classify ``delta`` into one of the :data:`ALL_KINDS` staging kinds.

    Resolution order:

    1. An explicit ``"kind"`` string in the JSON payload (set by the family
       constructors in :mod:`opop.proposer.families`) wins.
    2. Otherwise infer from the op + key:

       * ``add_constraint`` (class B) → :data:`KIND_CUT`;
       * ``set_param`` (class C) → :data:`KIND_HEURISTIC` for a ``heuristics/``
         key, :data:`KIND_DECOMPOSITION` for a ``decomposition/`` key, else
         :data:`KIND_PARAM`;
       * ``rename_var`` (class A) → :data:`KIND_FORMULATION` (a relabel is the
         variable-encoding axis of a formulation family);
       * anything else (e.g. a class-C metadata no-op) → :data:`KIND_PARAM`.
    """
    payload = _payload(delta)
    if payload is not None:
        tagged = payload.get("kind")
        if isinstance(tagged, str) and tagged in ALL_KINDS:
            return tagged
        op = payload.get("op")
        if op == "add_constraint":
            return KIND_CUT
        if op == "set_param":
            key = payload.get("key")
            if isinstance(key, str):
                if key.startswith("heuristics/"):
                    return KIND_HEURISTIC
                if key.startswith("decomposition/"):
                    return KIND_DECOMPOSITION
            return KIND_PARAM
        if op == "rename_var":
            return KIND_FORMULATION

    # No usable payload — fall back to the declared verification class.
    if delta.declared_class is DeltaClass.A:
        return KIND_FORMULATION
    if delta.declared_class is DeltaClass.B:
        return KIND_CUT
    return KIND_PARAM


# ---------------------------------------------------------------------------
# Stage gates
# ---------------------------------------------------------------------------
def stage_filter(deltas: Iterable[Delta], stage: Stage | str | int) -> list[Delta]:
    """Return only the ``deltas`` whose kind is legal at ``stage`` (order kept).

    This is the proposer's stage gate: at S0 it keeps only ``param`` deltas, at
    S1 it additionally keeps ``cut`` deltas, and so on up the ladder. A
    ``formulation``/``decomposition`` delta therefore CANNOT survive below S3,
    and a ``multikernel`` delta CANNOT survive below S4.
    """
    st = parse_stage(stage)
    return [d for d in deltas if stage_allows(st, delta_kind(d))]


def stage_space(
    space: Iterable[str] | Iterable[Delta] | None,
    stage: Stage | str | int,
) -> frozenset[str]:
    """Restrict an abstract design ``space`` to the kinds legal at ``stage``.

    ``space`` may be:

    * ``None`` — the full design space; returns exactly :func:`allowed_kinds`;
    * an iterable of kind strings — returns those also legal at ``stage``;
    * an iterable of :class:`Delta` — returns the legal kinds present among them.

    The result is the search-space view of :func:`stage_filter`: the set of
    delta kinds a controller / proposer is permitted to explore at this rung.
    """
    legal = allowed_kinds(stage)
    if space is None:
        return legal
    kinds: set[str] = set()
    for item in space:
        kind = item if isinstance(item, str) else delta_kind(item)
        if kind in legal:
            kinds.add(kind)
    return frozenset(kinds)
