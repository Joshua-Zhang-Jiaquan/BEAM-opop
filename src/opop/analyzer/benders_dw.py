"""Benders / Dantzig--Wolfe *readiness* classifier for the OPOP analyzer (task 26).

Task 24 (:mod:`opop.analyzer.decompose`) answers the *structural* question — does
the constraint matrix split into independent blocks, and is the border made of
coupling **constraints** (Dantzig--Wolfe) or complicating **variables**
(Benders)? This module sits one level up and answers the *readiness* question the
proposer (task 27) and controller (task 28) actually consume: **which
decomposition strategies are worth trying on this instance, and why?**

It REUSES :func:`opop.analyzer.decompose.detect_decomposition` (never duplicating
the border search) and folds in the variable-domain signal that the purely
graph-structural detector ignores:

* **Dantzig--Wolfe readiness** — the matrix is bordered block-diagonal by
  coupling constraints (verdict ``DW``) or already pure block-diagonal
  (verdict ``BLOCK``): a small master of linking rows over ``>= 2`` independent
  pricing blocks. Column generation / DW reformulation applies.
* **Benders readiness** — either the structural signal (bordered block-diagonal
  by complicating *variables*, verdict ``BENDERS``) **or** a classic two-stage
  shape: fixing the integer/binary (complicating) variables leaves a non-trivial
  **continuous** recourse problem (``n_integer >= 1`` and ``n_continuous >= 1``).
  Fixing the master variables yields an LP subproblem — exactly Benders.

The categorical :attr:`DecompositionReadiness.recommendation` is one of
:data:`READY_DW`, :data:`READY_BENDERS`, :data:`READY_BOTH`, :data:`READY_NONE`.
This module is **pure**: it reads the IR and its bipartite graph only, never
mutates the input, never solves, and never emits constraints or deltas — it is a
*signal* for downstream decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opop.analyzer.decompose import (
    DECOMP_BENDERS,
    DECOMP_BLOCK,
    DECOMP_DW,
    DecompositionReport,
    detect_decomposition,
)
from opop.model.ir import MILP, VarType

__all__ = [
    "READY_BENDERS",
    "READY_BOTH",
    "READY_DW",
    "READY_NONE",
    "DecompositionReadiness",
    "classify_readiness",
]

#: No decomposition strategy is recommended.
READY_NONE = "NONE"
#: Dantzig--Wolfe / column generation is recommended (coupling-constraint border).
READY_DW = "DW"
#: Benders decomposition is recommended (complicating variables / continuous recourse).
READY_BENDERS = "BENDERS"
#: Both Dantzig--Wolfe and Benders structures are present.
READY_BOTH = "BOTH"


# ---------------------------------------------------------------------------
# DecompositionReadiness
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecompositionReadiness:
    """Benders / Dantzig--Wolfe readiness verdict for one MILP.

    Attributes:
        recommendation: Categorical signal — one of :data:`READY_NONE`,
            :data:`READY_DW`, :data:`READY_BENDERS`, :data:`READY_BOTH`.
        dw_ready: ``True`` when the matrix is block-decomposable by coupling
            constraints (or already pure block-diagonal).
        benders_ready: ``True`` when complicating variables decompose the
            constraints, or a continuous recourse exists under integer staging.
        structure: The underlying structural verdict from
            :func:`~opop.analyzer.decompose.detect_decomposition`
            (``"NONE"`` / ``"BLOCK"`` / ``"DW"`` / ``"BENDERS"``).
        n_blocks: Number of independent blocks the structural detector found.
        linking_constraints: Coupling constraints (Dantzig--Wolfe master rows).
        linking_variables: Complicating variables (Benders master columns).
        n_integer: Count of BINARY / INTEGER variables.
        n_continuous: Count of CONTINUOUS variables.
        reasoning: Human-readable explanation of the verdict.
    """

    recommendation: str = READY_NONE
    dw_ready: bool = False
    benders_ready: bool = False
    structure: str = "NONE"
    n_blocks: int = 0
    linking_constraints: tuple[str, ...] = ()
    linking_variables: tuple[str, ...] = ()
    n_integer: int = 0
    n_continuous: int = 0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the readiness verdict."""
        return {
            "recommendation": self.recommendation,
            "dw_ready": self.dw_ready,
            "benders_ready": self.benders_ready,
            "structure": self.structure,
            "n_blocks": self.n_blocks,
            "linking_constraints": list(self.linking_constraints),
            "linking_variables": list(self.linking_variables),
            "n_integer": self.n_integer,
            "n_continuous": self.n_continuous,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def classify_readiness(
    ir: MILP, *, decomposition: DecompositionReport | None = None
) -> DecompositionReadiness:
    """Classify the Benders / Dantzig--Wolfe readiness of ``ir``.

    Args:
        ir: The MILP to analyze (never mutated).
        decomposition: A precomputed structural verdict from
            :func:`~opop.analyzer.decompose.detect_decomposition`. When ``None``
            it is computed here. Pass the shared report to avoid recomputing the
            border search in :func:`opop.analyzer.api.analyze`.

    Returns:
        A :class:`DecompositionReadiness` with the categorical recommendation,
        the contributing structural / variable-domain signals, and reasoning.
    """
    decomp = decomposition if decomposition is not None else detect_decomposition(ir)
    n_integer = sum(1 for v in ir.variables if v.vtype in (VarType.BINARY, VarType.INTEGER))
    n_continuous = sum(1 for v in ir.variables if v.vtype is VarType.CONTINUOUS)

    # Dantzig--Wolfe: coupling-constraint border (DW) or already separable (BLOCK).
    dw_ready = decomp.decomposability in (DECOMP_DW, DECOMP_BLOCK) and decomp.n_blocks >= 2

    # Benders: structural complicating-variable border, OR integer/continuous staging
    # (fixing the integer master leaves a continuous recourse LP).
    benders_structural = decomp.decomposability == DECOMP_BENDERS and decomp.n_blocks >= 2
    benders_staging = n_integer >= 1 and n_continuous >= 1
    benders_ready = benders_structural or benders_staging

    recommendation = _recommend(dw_ready, benders_ready)
    reasoning = _reasoning(
        decomp,
        dw_ready=dw_ready,
        benders_ready=benders_ready,
        benders_structural=benders_structural,
        benders_staging=benders_staging,
        n_integer=n_integer,
        n_continuous=n_continuous,
    )
    return DecompositionReadiness(
        recommendation=recommendation,
        dw_ready=dw_ready,
        benders_ready=benders_ready,
        structure=decomp.decomposability,
        n_blocks=decomp.n_blocks,
        linking_constraints=decomp.linking_constraints,
        linking_variables=decomp.linking_variables,
        n_integer=n_integer,
        n_continuous=n_continuous,
        reasoning=reasoning,
    )


def _recommend(dw_ready: bool, benders_ready: bool) -> str:
    """Map the two boolean readiness signals to the categorical recommendation."""
    if dw_ready and benders_ready:
        return READY_BOTH
    if dw_ready:
        return READY_DW
    if benders_ready:
        return READY_BENDERS
    return READY_NONE


def _reasoning(
    decomp: DecompositionReport,
    *,
    dw_ready: bool,
    benders_ready: bool,
    benders_structural: bool,
    benders_staging: bool,
    n_integer: int,
    n_continuous: int,
) -> str:
    """Build the human-readable explanation of the readiness verdict."""
    parts: list[str] = []
    if dw_ready:
        if decomp.decomposability == DECOMP_BLOCK:
            parts.append(
                f"pure block-diagonal with {decomp.n_blocks} separable blocks "
                + "(Dantzig-Wolfe / column generation applies, empty master)"
            )
        else:
            n_link = len(decomp.linking_constraints)
            parts.append(
                f"bordered block-diagonal: {decomp.n_blocks} blocks coupled by "
                + f"{n_link} linking constraint(s) -> Dantzig-Wolfe master + pricing blocks"
            )
    if benders_ready:
        if benders_structural:
            n_link = len(decomp.linking_variables)
            parts.append(
                f"{decomp.n_blocks} constraint blocks coupled by {n_link} complicating "
                + "variable(s) -> Benders master + subproblems"
            )
        if benders_staging:
            parts.append(
                f"two-stage shape: {n_integer} integer (complicating) and {n_continuous} "
                + "continuous variable(s) -> fixing integers yields a continuous recourse LP"
            )
    if not parts:
        return (
            "no exploitable decomposition: monolithic structure and no integer/continuous "
            + "staging (single-stage or single-domain model)"
        )
    return "; ".join(parts)
