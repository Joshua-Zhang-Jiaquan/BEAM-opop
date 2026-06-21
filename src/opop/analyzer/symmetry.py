"""Symmetry & dominance detection for the OPOP analyzer (task 26).

Two structurally distinct variables that the model treats *identically* create
symmetric search subtrees: a branch-and-bound solver re-explores equivalent
solutions over and over. Detecting these symmetries (and the related
column-dominance signal) lets the proposer (task 27) recommend symmetry handling
and lets the controller (task 28) reason about why a model is hard. This module
only **reports** the signal; it never emits symmetry-breaking constraints — those
are class-B deltas that must be certified by the verification gate (task 11).

Detection runs on the variable--constraint **bipartite graph**
(:func:`opop.model.ir.model_graph`), coloured so a graph automorphism preserves
the math model:

* variable nodes are coloured by ``(vtype, objective coefficient, lower, upper)``,
* constraint nodes by ``(sense, rhs)``,
* edges by their constraint coefficient.

A colour-respecting automorphism of this graph is exactly a symmetry of the MILP
(it permutes variables and constraints, mapping the objective/constraints onto
themselves). The **orbits** of the variable nodes under the automorphism group
are the interchangeable-variable classes. We enumerate automorphisms with
``networkx``'s VF2 :class:`~networkx.algorithms.isomorphism.GraphMatcher`
(node/edge attribute matching), capped to stay cheap on large or
highly-symmetric instances; orbits computed from a capped subset are a *valid
under-approximation* (every reported orbit is genuinely symmetric).

**Dominance** is a complementary, cheaper signal: two variables with identical
domains and identical constraint columns (same coefficient in every constraint)
are interchangeable in the feasible region, so the one with the better objective
coefficient *dominates* the other. Equal columns *and* equal objective makes them
symmetric (also surfaced as an orbit).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from opop.model.ir import MILP, ObjSense, model_graph

__all__ = [
    "SYMMETRY_ANALYZED",
    "SYMMETRY_EMPTY",
    "SYMMETRY_SKIPPED",
    "SymmetryInfo",
    "detect_symmetry",
]

#: Detection ran and the result is exact for the enumerated automorphisms.
SYMMETRY_ANALYZED = "ANALYZED"
#: The model has no variables (nothing to analyze).
SYMMETRY_EMPTY = "EMPTY"
#: The graph exceeded the node cap; detection was skipped (no false negatives claimed).
SYMMETRY_SKIPPED = "SKIPPED_TOO_LARGE"

#: Default cap on graph nodes (vars + constraints) before detection is skipped.
_DEFAULT_MAX_NODES = 120
#: Default cap on enumerated automorphisms (orbits from a subset stay valid).
_DEFAULT_MAX_AUTOMORPHISMS = 20_000
#: Tolerance for rounding coefficients / bounds into colour keys.
_ROUND = 9


# ---------------------------------------------------------------------------
# SymmetryInfo
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SymmetryInfo:
    """Symmetry & dominance signal for one MILP.

    Attributes:
        orbits: Variable-name orbits of size ``>= 2`` under the model's
            automorphism group — each tuple is a class of interchangeable
            variables (sorted within and across orbits for determinism).
        n_automorphisms: Number of automorphisms enumerated (``>= 1``; includes
            the identity). Capped — see :data:`_DEFAULT_MAX_AUTOMORPHISMS`.
        dominance_pairs: ``(dominator, dominated)`` variable pairs sharing an
            identical column and domain but with a strictly better objective
            coefficient for the dominator.
        status: :data:`SYMMETRY_ANALYZED`, :data:`SYMMETRY_EMPTY`, or
            :data:`SYMMETRY_SKIPPED`.
        capped: ``True`` if automorphism enumeration hit the cap (orbits are then
            a valid under-approximation rather than the complete partition).
    """

    orbits: tuple[tuple[str, ...], ...] = ()
    n_automorphisms: int = 0
    dominance_pairs: tuple[tuple[str, str], ...] = ()
    status: str = SYMMETRY_EMPTY
    capped: bool = False

    @property
    def has_symmetry(self) -> bool:
        """``True`` iff at least one non-trivial variable orbit was found."""
        return len(self.orbits) > 0

    @property
    def n_symmetric_vars(self) -> int:
        """Total number of variables lying in a non-trivial orbit."""
        return sum(len(orbit) for orbit in self.orbits)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of the symmetry signal."""
        return {
            "orbits": [list(orbit) for orbit in self.orbits],
            "n_automorphisms": self.n_automorphisms,
            "dominance_pairs": [list(pair) for pair in self.dominance_pairs],
            "status": self.status,
            "capped": self.capped,
            "has_symmetry": self.has_symmetry,
        }


# ---------------------------------------------------------------------------
# Union-find (deterministic orbit accumulation)
# ---------------------------------------------------------------------------
class _UnionFind:
    """A tiny union--find over hashable items (path-compressing)."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent: dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        parent = self._parent
        root = x
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
# Public entry point
# ---------------------------------------------------------------------------
def detect_symmetry(
    ir: MILP,
    *,
    max_nodes: int = _DEFAULT_MAX_NODES,
    max_automorphisms: int = _DEFAULT_MAX_AUTOMORPHISMS,
) -> SymmetryInfo:
    """Detect variable symmetries (orbits) and column dominance in ``ir``.

    Args:
        ir: The MILP to analyze (never mutated).
        max_nodes: Skip automorphism detection when ``n_vars + n_constraints``
            exceeds this (returns :data:`SYMMETRY_SKIPPED`; dominance still runs).
        max_automorphisms: Stop enumerating automorphisms after this many. Orbits
            built from a capped subset remain a valid under-approximation.

    Returns:
        A :class:`SymmetryInfo` with variable orbits, the automorphism count, and
        dominance pairs. Falls back to :data:`SYMMETRY_SKIPPED` / dominance-only
        on large instances and never raises.
    """
    if not ir.variables:
        return SymmetryInfo(status=SYMMETRY_EMPTY)

    dominance = _dominance_pairs(ir)

    n_nodes = ir.n_vars + ir.n_constraints
    if n_nodes > max_nodes:
        return SymmetryInfo(
            dominance_pairs=dominance,
            status=SYMMETRY_SKIPPED,
        )

    orbits, n_automorphisms, capped = _variable_orbits(ir, max_automorphisms)
    return SymmetryInfo(
        orbits=orbits,
        n_automorphisms=n_automorphisms,
        dominance_pairs=dominance,
        status=SYMMETRY_ANALYZED,
        capped=capped,
    )


# ---------------------------------------------------------------------------
# Coloured-automorphism orbits (networkx VF2)
# ---------------------------------------------------------------------------
def _variable_orbits(
    ir: MILP, max_automorphisms: int
) -> tuple[tuple[tuple[str, ...], ...], int, bool]:
    """Return ``(orbits, n_automorphisms, capped)`` from the coloured bipartite graph."""
    from networkx.algorithms.isomorphism import GraphMatcher

    graph: Any = _colored_graph(ir)

    def node_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return a["color"] == b["color"]

    def edge_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return a["weight"] == b["weight"]

    matcher = GraphMatcher(graph, graph, node_match=node_match, edge_match=edge_match)

    var_keys = [("var", v.name) for v in ir.variables]
    uf = _UnionFind([repr(k) for k in var_keys])
    n_automorphisms = 0
    capped = False
    for mapping in matcher.isomorphisms_iter():
        n_automorphisms += 1
        for src, dst in mapping.items():
            if src[0] == "var" and dst[0] == "var":
                uf.union(repr(src), repr(dst))
        if n_automorphisms >= max_automorphisms:
            capped = True
            break

    groups: dict[str, list[str]] = {}
    for var in ir.variables:
        key = repr(("var", var.name))
        groups.setdefault(uf.find(key), []).append(var.name)
    orbits = tuple(
        sorted(tuple(sorted(names)) for names in groups.values() if len(names) >= 2)
    )
    return orbits, n_automorphisms, capped


def _colored_graph(ir: MILP) -> Any:
    """Build a node/edge-coloured ``networkx`` graph of the bipartite model graph.

    Uses :func:`opop.model.ir.model_graph` for the incidence skeleton and the IR
    for the colours (variable domain/objective, constraint sense/rhs, edge
    coefficients) that make an automorphism a true model symmetry.
    """
    import networkx as nx

    skeleton = model_graph(ir)
    obj = ir.objective.coeffs
    graph: Any = nx.Graph()
    for var in ir.variables:
        var_color: tuple[Any, ...] = (
            "var",
            var.vtype.value,
            round(float(obj.get(var.name, 0.0)), _ROUND),
            _round_bound(var.lower),
            _round_bound(var.upper),
        )
        graph.add_node(("var", var.name), color=var_color)
    for con in ir.constraints:
        con_color: tuple[Any, ...] = ("con", con.sense.value, round(float(con.rhs), _ROUND))
        graph.add_node(("con", con.name), color=con_color)
    coeff_by_pair = {
        (vname, con.name): coeff
        for con in ir.constraints
        for vname, coeff in con.coeffs.items()
        if coeff != 0.0
    }
    for vname, cname in skeleton.edges:
        graph.add_edge(
            ("var", vname),
            ("con", cname),
            weight=round(float(coeff_by_pair[(vname, cname)]), _ROUND),
        )
    return graph


def _round_bound(value: float) -> float:
    """Round a bound for colour keys, preserving infinities exactly."""
    if value in (float("inf"), float("-inf")):
        return value
    return round(float(value), _ROUND)


# ---------------------------------------------------------------------------
# Column dominance
# ---------------------------------------------------------------------------
def _dominance_pairs(ir: MILP) -> tuple[tuple[str, str], ...]:
    """Return ``(dominator, dominated)`` pairs sharing a column but not objective.

    Two variables with the same vtype, bounds, and identical constraint column
    (same coefficient in every constraint) are interchangeable; the one with the
    better objective coefficient dominates. For MINIMIZE the lower coefficient
    dominates, for MAXIMIZE the higher.
    """
    obj = ir.objective.coeffs
    columns: dict[str, list[tuple[str, float]]] = {v.name: [] for v in ir.variables}
    for con in ir.constraints:
        for vname, coeff in con.coeffs.items():
            if coeff != 0.0:
                columns[vname].append((con.name, round(float(coeff), _ROUND)))

    groups: dict[tuple[Any, ...], list[str]] = {}
    for var in ir.variables:
        signature = (
            var.vtype.value,
            _round_bound(var.lower),
            _round_bound(var.upper),
            tuple(sorted(columns[var.name])),
        )
        groups.setdefault(signature, []).append(var.name)

    minimize = ir.objective.sense is ObjSense.MINIMIZE

    def obj_coeff(name: str) -> float:
        return round(float(obj.get(name, 0.0)), _ROUND)

    def rank_key(name: str) -> tuple[float, str]:
        # Best objective first: lower coeff for MINIMIZE, higher for MAXIMIZE.
        coeff = obj_coeff(name)
        return (coeff if minimize else -coeff, name)

    pairs: list[tuple[str, str]] = []
    for names in groups.values():
        if len(names) < 2:
            continue
        ranked = sorted(names, key=rank_key)
        best = ranked[0]
        best_obj = obj_coeff(best)
        for other in ranked[1:]:
            if obj_coeff(other) != best_obj:
                pairs.append((best, other))
    return tuple(sorted(pairs))
