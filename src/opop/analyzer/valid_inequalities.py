"""Valid-inequality CANDIDATE generation from a Phase-1 whitelist.

This module proposes structurally-motivated cutting planes but **does not
certify them**. Every returned :class:`LinearConstraint` is a *candidate*; the
verification gate (task 11) is solely responsible for proving validity (class-B:
no feasible integer incumbent removed). The analyzer never adds cuts to the IR.

Two whitelisted families are generated:

* **Cover cuts** — from a 0/1 knapsack row ``sum_j a_j x_j <= b`` (binary
  variables, positive coefficients, positive ``b``). A *cover* ``C`` has
  ``sum_{j in C} a_j > b``; a *minimal* cover additionally satisfies
  ``sum_{j in C} a_j - min_{j in C} a_j <= b``. Each minimal cover yields the
  candidate ``sum_{j in C} x_j <= |C| - 1``.
* **Clique cuts** — from a set-packing conflict graph. Every row of the form
  ``sum_{j in S} x_j <= 1`` (binary, unit coefficients) makes the members of
  ``S`` pairwise conflicting. A maximal clique ``K`` of size >= 3 in that graph
  yields the candidate ``sum_{j in K} x_j <= 1`` (skipped when ``K`` is already
  exactly one existing constraint's support).

Combinatorial work is bounded by term/size/count caps so generation stays cheap.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from opop.model.ir import MILP, ConstraintSense, LinearConstraint, VarType

__all__ = [
    "generate_clique_cuts",
    "generate_cover_cuts",
    "generate_valid_inequalities",
]

_TOL = 1e-9


def generate_valid_inequalities(ir: MILP, *, max_cuts: int = 64) -> list[LinearConstraint]:
    """Return cover + clique candidate cuts for ``ir`` (deduplicated, novel).

    ``max_cuts`` caps each family. Duplicate candidates (same support, sense, and
    rhs) are collapsed, preserving first-seen order, and any candidate identical
    to a constraint already in ``ir`` is dropped — a candidate cut must be new.
    """
    cuts = generate_cover_cuts(ir, max_cuts=max_cuts)
    cuts.extend(generate_clique_cuts(ir, max_cuts=max_cuts))
    existing = {_signature(c.coeffs, c.sense.value, c.rhs) for c in ir.constraints}
    return _dedupe(cuts, existing)


# ---------------------------------------------------------------------------
# Cover cuts
# ---------------------------------------------------------------------------
def generate_cover_cuts(
    ir: MILP,
    *,
    max_terms: int = 16,
    max_cover_size: int = 8,
    max_cuts: int = 64,
) -> list[LinearConstraint]:
    """Generate minimal-cover candidate cuts from 0/1 knapsack rows of ``ir``."""
    binary = {v.name for v in ir.variables if v.vtype is VarType.BINARY}
    cuts: list[LinearConstraint] = []
    serial = 0
    for con in ir.constraints:
        weights = _knapsack_weights(con, binary, max_terms)
        if weights is None:
            continue
        for cover in _minimal_covers(weights, con.rhs, max_cover_size):
            cuts.append(
                LinearConstraint(
                    name=f"cover_{con.name}_{serial}",
                    coeffs={name: 1.0 for name in cover},
                    sense=ConstraintSense.LE,
                    rhs=float(len(cover) - 1),
                )
            )
            serial += 1
            if len(cuts) >= max_cuts:
                return cuts
    return cuts


def _knapsack_weights(
    con: LinearConstraint, binary: set[str], max_terms: int
) -> dict[str, float] | None:
    """Return ``{var: weight}`` if ``con`` is a usable 0/1 knapsack row, else None."""
    if con.sense is not ConstraintSense.LE or con.rhs <= _TOL:
        return None
    weights = {n: c for n, c in con.coeffs.items() if abs(c) > _TOL}
    if not weights or len(weights) > max_terms:
        return None
    if not all(n in binary for n in weights):
        return None
    if not all(c > _TOL for c in weights.values()):
        return None
    return weights


def _minimal_covers(
    weights: dict[str, float], capacity: float, max_cover_size: int
) -> list[tuple[str, ...]]:
    """Enumerate minimal covers (by increasing size) up to ``max_cover_size``."""
    names = sorted(weights, key=lambda n: (-weights[n], n))
    covers: list[tuple[str, ...]] = []
    upper = min(len(names), max_cover_size)
    for size in range(2, upper + 1):
        for combo in combinations(names, size):
            total = sum(weights[n] for n in combo)
            if total <= capacity + _TOL:
                continue
            if total - min(weights[n] for n in combo) <= capacity + _TOL:
                covers.append(tuple(sorted(combo)))
    return covers


# ---------------------------------------------------------------------------
# Clique cuts
# ---------------------------------------------------------------------------
def generate_clique_cuts(
    ir: MILP,
    *,
    min_clique_size: int = 3,
    max_cuts: int = 64,
) -> list[LinearConstraint]:
    """Generate clique candidate cuts from the set-packing conflict graph."""
    import networkx as nx

    binary = {v.name for v in ir.variables if v.vtype is VarType.BINARY}
    graph: Any = nx.Graph()
    existing: set[frozenset[str]] = set()
    for con in ir.constraints:
        members = _set_packing_members(con, binary)
        if members is None:
            continue
        existing.add(frozenset(members))
        for left, right in combinations(members, 2):
            graph.add_edge(left, right)

    cuts: list[LinearConstraint] = []
    serial = 0
    for clique in nx.find_cliques(graph):
        if len(clique) < min_clique_size or frozenset(clique) in existing:
            continue
        members = sorted(clique)
        cuts.append(
            LinearConstraint(
                name=f"clique_{serial}",
                coeffs={name: 1.0 for name in members},
                sense=ConstraintSense.LE,
                rhs=1.0,
            )
        )
        serial += 1
        if len(cuts) >= max_cuts:
            break
    return cuts


def _set_packing_members(con: LinearConstraint, binary: set[str]) -> list[str] | None:
    """Return the members of a set-packing row ``sum x_j <= 1``, else None."""
    if con.sense is not ConstraintSense.LE or abs(con.rhs - 1.0) > _TOL:
        return None
    nz = {n: c for n, c in con.coeffs.items() if abs(c) > _TOL}
    if len(nz) < 2 or not all(n in binary for n in nz):
        return None
    if not all(abs(c - 1.0) <= _TOL for c in nz.values()):
        return None
    return sorted(nz)


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------
_Signature = tuple[tuple[tuple[str, float], ...], str, float]


def _signature(coeffs: dict[str, float], sense: str, rhs: float) -> _Signature:
    """A coefficient-exact signature for cut-vs-cut and cut-vs-constraint matching."""
    support = tuple(sorted((n, round(c, 9)) for n, c in coeffs.items() if abs(c) > _TOL))
    return support, sense, round(rhs, 9)


def _dedupe(
    cuts: list[LinearConstraint], existing: set[_Signature]
) -> list[LinearConstraint]:
    seen: set[_Signature] = set(existing)
    unique: list[LinearConstraint] = []
    for cut in cuts:
        signature = _signature(cut.coeffs, cut.sense.value, cut.rhs)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(cut)
    return unique
