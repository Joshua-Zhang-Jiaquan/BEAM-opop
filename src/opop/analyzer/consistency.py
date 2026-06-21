"""Dimension / units / index consistency checks for the OPOP analyzer.

These deterministic checks operate purely on the IR (no solver). They surface
*modelling* errors that a numerical solve would otherwise hide or mis-report:

* **Index consistency** — a constraint or variable annotated as corresponding to
  member ``m`` of a named index set ``S`` is flagged when ``S`` is not declared
  in :attr:`MILP.index_sets` or ``m`` is not a member of ``S`` (a constraint
  "referencing a missing set member"). Dangling annotations (naming a constraint
  / variable that does not exist) are flagged too.
* **Dimension consistency** — a constraint whose declared term count disagrees
  with the number of non-zero coefficients it actually carries is flagged.
* **Units consistency** — a constraint that linearly combines variables carrying
  two or more distinct declared units (you cannot add kilograms to metres) is
  flagged; a declared constraint unit that disagrees with its terms' unit is
  flagged as well.

Annotations are read from :attr:`MILP.index_sets` plus an opt-in ``metadata``
contract (all keys optional; absent keys disable the corresponding check):

``metadata["index_annotations"]``
    ``{name: {set_name: member}}`` — ``name`` is a constraint OR variable name.
``metadata["dimension_specs"]``
    ``{constraint_name: expected_non_zero_term_count}``.
``metadata["variable_units"]``
    ``{variable_name: unit_label}``.
``metadata["constraint_units"]``
    ``{constraint_name: expected_unit_label}``.

Because the IR is immutable and the analyzer never mutates it, these checks are
pure functions returning a list of :class:`Flag`.
"""

from __future__ import annotations

from typing import Any

from opop.analyzer.report import DIMENSION_MISMATCH, INDEX_ERROR, UNITS_MISMATCH, Flag
from opop.model.ir import MILP

__all__ = ["check_consistency"]


def check_consistency(ir: MILP) -> list[Flag]:
    """Run all consistency checks on ``ir`` and return the flags found.

    Order is stable: index errors, then dimension mismatches, then units
    mismatches. An empty list means the model is consistent w.r.t. the
    annotations present (a model with no annotations always passes).
    """
    flags: list[Flag] = []
    flags.extend(_check_index(ir))
    flags.extend(_check_dimensions(ir))
    flags.extend(_check_units(ir))
    return flags


# ---------------------------------------------------------------------------
# Index consistency
# ---------------------------------------------------------------------------
def _check_index(ir: MILP) -> list[Flag]:
    annotations = _mapping(ir.metadata.get("index_annotations"))
    if not annotations:
        return []

    known_names = {v.name for v in ir.variables} | {c.name for c in ir.constraints}
    flags: list[Flag] = []
    for name in sorted(annotations):
        if name not in known_names:
            flags.append(
                Flag(
                    INDEX_ERROR,
                    f"index annotation references unknown constraint/variable {name!r}",
                    name,
                )
            )
            continue
        per_set = _mapping(annotations[name])
        for set_name in sorted(per_set):
            member = str(per_set[set_name])
            members = ir.index_sets.get(set_name)
            if members is None:
                flags.append(
                    Flag(
                        INDEX_ERROR,
                        f"{name!r} indexes undeclared set {set_name!r}",
                        name,
                    )
                )
            elif member not in members:
                flags.append(
                    Flag(
                        INDEX_ERROR,
                        f"{name!r} references missing member {member!r} of set {set_name!r}",
                        name,
                    )
                )
    return flags


# ---------------------------------------------------------------------------
# Dimension consistency
# ---------------------------------------------------------------------------
def _check_dimensions(ir: MILP) -> list[Flag]:
    specs = _mapping(ir.metadata.get("dimension_specs"))
    if not specs:
        return []

    by_name = {c.name: c for c in ir.constraints}
    flags: list[Flag] = []
    for con_name in sorted(specs):
        expected = specs[con_name]
        con = by_name.get(con_name)
        if con is None:
            flags.append(
                Flag(
                    DIMENSION_MISMATCH,
                    f"dimension spec references unknown constraint {con_name!r}",
                    con_name,
                )
            )
            continue
        actual = sum(1 for v in con.coeffs.values() if v != 0.0)
        if actual != int(expected):
            flags.append(
                Flag(
                    DIMENSION_MISMATCH,
                    f"constraint {con_name!r} has {actual} terms, expected {int(expected)}",
                    con_name,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# Units consistency
# ---------------------------------------------------------------------------
def _check_units(ir: MILP) -> list[Flag]:
    var_units = _mapping(ir.metadata.get("variable_units"))
    con_units = _mapping(ir.metadata.get("constraint_units"))
    if not var_units and not con_units:
        return []

    flags: list[Flag] = []
    for con in ir.constraints:
        terms = [n for n, c in con.coeffs.items() if c != 0.0]
        units = {str(var_units[n]) for n in terms if n in var_units}
        if len(units) >= 2:
            flags.append(
                Flag(
                    UNITS_MISMATCH,
                    f"constraint {con.name!r} mixes incompatible units {sorted(units)}",
                    con.name,
                )
            )
            continue
        declared = con_units.get(con.name)
        if declared is not None and units and str(declared) not in units:
            flags.append(
                Flag(
                    UNITS_MISMATCH,
                    (
                        f"constraint {con.name!r} declared unit {str(declared)!r} "
                        f"disagrees with term unit {sorted(units)}"
                    ),
                    con.name,
                )
            )
    return flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mapping(value: Any) -> dict[Any, Any]:
    """Return ``value`` if it is a dict, else an empty dict (defensive)."""
    return value if isinstance(value, dict) else {}
