"""Redundancy, trivial-infeasibility, and conflict detection for the analyzer.

Deterministic, solver-free structural checks over the IR:

* **Trivial infeasibility** — a constraint with no variable terms that is still
  violated (``0 <= -1``, ``0 >= 1``, ``0 = 5``), or a variable whose declared
  domain is empty (``lower > upper``).
* **Redundancy** — a constraint that is an exact or *scaled* duplicate of
  another (``2x + 2y <= 4`` duplicates ``x + y <= 2``), or is dominated by a
  tighter same-direction constraint (``x + y <= 5`` is dominated by
  ``x + y <= 2``).
* **Conflict** — two constraints over the same (proportional) left-hand side
  whose implied bounds are incompatible (``x + y <= 1`` versus ``x + y >= 2``;
  ``x + y = 1`` versus ``x + y = 2``).

The core idea is *canonicalisation*: each linear row is reduced to a signed
direction by dividing through by its largest-magnitude coefficient (the
"pivot"), so proportional rows collapse to one key. Rows sharing a key are then
compared by their normalised right-hand side. ``=`` rows act as a simultaneous
``<=`` and ``>=`` at the same value, so the same interval test detects every
cross-sense conflict.
"""

from __future__ import annotations

from collections import defaultdict

from opop.analyzer.report import CONFLICT, REDUNDANT, TRIVIAL_INFEASIBILITY, Flag
from opop.model.ir import MILP, ConstraintSense, LinearConstraint

__all__ = ["detect_redundancy"]

_TOL = 1e-9

# A canonical row: (direction key, sense, normalised rhs).
_Canon = tuple[tuple[tuple[str, float], ...], ConstraintSense, float]


def detect_redundancy(ir: MILP, *, tol: float = _TOL) -> list[Flag]:
    """Return redundancy / trivial-infeasibility / conflict flags for ``ir``.

    Order is stable: trivial issues first (empty rows, empty domains), then
    proportional-family analysis (duplicates, dominance, conflicts). An empty
    list means none were detected.
    """
    flags: list[Flag] = []
    flags.extend(_check_trivial(ir, tol))
    flags.extend(_check_families(ir, tol))
    return flags


# ---------------------------------------------------------------------------
# Trivial infeasibility / always-true empty rows / empty variable domains
# ---------------------------------------------------------------------------
def _check_trivial(ir: MILP, tol: float) -> list[Flag]:
    flags: list[Flag] = []
    for con in ir.constraints:
        if any(abs(c) > tol for c in con.coeffs.values()):
            continue  # not an empty row
        rhs = con.rhs
        sense = con.sense
        infeasible = (
            (sense is ConstraintSense.LE and rhs < -tol)
            or (sense is ConstraintSense.GE and rhs > tol)
            or (sense is ConstraintSense.EQ and abs(rhs) > tol)
        )
        body = f"constraint {con.name!r} reduces to 0 {sense.value} {rhs}"
        if infeasible:
            flags.append(Flag(TRIVIAL_INFEASIBILITY, f"{body} (infeasible)", con.name))
        else:
            flags.append(Flag(REDUNDANT, f"{body} (always satisfied)", con.name))

    for var in ir.variables:
        if var.lower > var.upper + tol:
            flags.append(
                Flag(
                    TRIVIAL_INFEASIBILITY,
                    f"variable {var.name!r} has empty domain [{var.lower}, {var.upper}]",
                    var.name,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# Proportional families: duplicate / dominated / conflict
# ---------------------------------------------------------------------------
def _check_families(ir: MILP, tol: float) -> list[Flag]:
    groups: dict[tuple[tuple[str, float], ...], list[tuple[str, ConstraintSense, float]]] = (
        defaultdict(list)
    )
    for con in ir.constraints:
        canon = _canonical(con, tol)
        if canon is None:
            continue
        key, sense, bound = canon
        groups[key].append((con.name, sense, bound))

    flags: list[Flag] = []
    for rows in groups.values():
        if len(rows) >= 2:
            flags.extend(_analyze_family(rows, tol))
    return flags


def _canonical(con: LinearConstraint, tol: float) -> _Canon | None:
    """Reduce a constraint to ``(direction-key, sense, normalised-rhs)``.

    The pivot is the largest-magnitude coefficient (ties: alphabetically first).
    Dividing by the (signed) pivot fixes its coefficient to ``+1`` and flips the
    sense when the pivot is negative, so two proportional rows share a key.
    Returns ``None`` for an empty row (handled by the trivial check).
    """
    nz = {n: c for n, c in con.coeffs.items() if abs(c) > tol}
    if not nz:
        return None
    pivot = ""
    best = -1.0
    for name in sorted(nz):
        magnitude = abs(nz[name])
        if magnitude > best + tol:
            best = magnitude
            pivot = name
    pivot_coeff = nz[pivot]
    key = tuple(sorted((n, round(c / pivot_coeff, 9)) for n, c in nz.items()))
    bound = round(con.rhs / pivot_coeff, 9)
    sense = con.sense if pivot_coeff > 0 else _flip(con.sense)
    return key, sense, bound


def _flip(sense: ConstraintSense) -> ConstraintSense:
    if sense is ConstraintSense.LE:
        return ConstraintSense.GE
    if sense is ConstraintSense.GE:
        return ConstraintSense.LE
    return ConstraintSense.EQ


def _analyze_family(
    rows: list[tuple[str, ConstraintSense, float]], tol: float
) -> list[Flag]:
    les = [(n, b) for n, s, b in rows if s is ConstraintSense.LE]
    ges = [(n, b) for n, s, b in rows if s is ConstraintSense.GE]
    eqs = [(n, b) for n, s, b in rows if s is ConstraintSense.EQ]

    flags: list[Flag] = []
    flags.extend(_dominance(les, tol, tightest="min"))
    flags.extend(_dominance(ges, tol, tightest="max"))
    flags.extend(_eq_duplicates(eqs, tol))
    flags.extend(_conflict(les, ges, eqs, tol))
    return flags


def _dominance(
    items: list[tuple[str, float]], tol: float, *, tightest: str
) -> list[Flag]:
    """Flag same-direction rows looser than (or equal to) the tightest one."""
    if len(items) < 2:
        return []
    if tightest == "min":
        binding = min(range(len(items)), key=lambda i: items[i][1])
    else:
        binding = max(range(len(items)), key=lambda i: items[i][1])
    binding_name, binding_bound = items[binding]

    flags: list[Flag] = []
    for i, (name, bound) in enumerate(items):
        if i == binding:
            continue
        if abs(bound - binding_bound) <= tol:
            flags.append(Flag(REDUNDANT, f"constraint {name!r} duplicates {binding_name!r}", name))
        else:
            flags.append(
                Flag(REDUNDANT, f"constraint {name!r} is dominated by {binding_name!r}", name)
            )
    return flags


def _eq_duplicates(eqs: list[tuple[str, float]], tol: float) -> list[Flag]:
    """Flag equality rows that repeat the first equality's value (duplicates)."""
    if len(eqs) < 2:
        return []
    first_name, first_bound = eqs[0]
    flags: list[Flag] = []
    for name, bound in eqs[1:]:
        if abs(bound - first_bound) <= tol:
            flags.append(Flag(REDUNDANT, f"constraint {name!r} duplicates {first_name!r}", name))
    return flags


def _conflict(
    les: list[tuple[str, float]],
    ges: list[tuple[str, float]],
    eqs: list[tuple[str, float]],
    tol: float,
) -> list[Flag]:
    """Flag a conflict when the implied lower bound exceeds the implied upper.

    ``=`` rows constrain from both sides, so they join both candidate lists; the
    single interval test then catches LE/GE, EQ/LE, EQ/GE, and EQ/EQ conflicts.
    """
    upper_candidates = les + eqs
    lower_candidates = ges + eqs
    if not upper_candidates or not lower_candidates:
        return []
    up_name, up_bound = min(upper_candidates, key=lambda t: t[1])
    lo_name, lo_bound = max(lower_candidates, key=lambda t: t[1])
    if lo_bound > up_bound + tol:
        message = (
            f"constraints {lo_name!r} (>= {lo_bound}) and {up_name!r} (<= {up_bound}) "
            f"are incompatible"
        )
        return [Flag(CONFLICT, message, up_name)]
    return []
