"""MaxSAT (DIMACS CNF / WCNF) loader → generic clause-satisfaction MILP (task 34).

Parses the DIMACS `MaxSAT Evaluations <https://maxsat-evaluations.github.io/>`_
formats and maps them to the OPOP IR via the textbook satisfied-clause-count
linearization (uniform; no per-instance tuning):

* binary literals ``x_v`` (``v = 1 .. nvars``);
* per clause, the satisfaction expression ``L(c) = sum_{l>0} x_l + sum_{l<0} (1 - x_l)``
  (the number of satisfied literal occurrences);
* HARD clauses (weight ``== top`` in WCNF): ``L(c) >= 1`` (must be satisfied);
* SOFT clauses: a binary ``z_c`` with ``L(c) - z_c >= 0`` (``z_c`` may be 1 only
  when the clause is satisfied);
* objective ``maximise sum_{soft c} w_c z_c``.

``p cnf nvars nclauses`` treats every clause as soft with weight 1 (plain
MaxSAT); ``p wcnf nvars nclauses top`` reads a leading per-clause weight and
treats ``weight == top`` as hard (partial / weighted MaxSAT). A missing ``p``
line, an out-of-range variable, or a clause not terminated by ``0`` before EOF
raises :class:`~opop.bench.classic.base.ParseError` with file + line context.
"""

from __future__ import annotations

from pathlib import Path

from opop.bench.classic.base import (
    ClassicAdapter,
    ParseError,
    TokenCursor,
    read_text,
    tag_instance,
)
from opop.model.adapter import register_adapter
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)

__all__ = ["ADAPTER", "load", "loads"]

#: The registered classic-CO adapter for the MaxSAT family.
ADAPTER = ClassicAdapter(family="maxsat", problem_class="MaxSAT")


def _parse_header(lines: list[str], *, source: str) -> tuple[str, int, int, float | None, int]:
    """Find the DIMACS ``p`` line; return (format, nvars, nclauses, top, body_index)."""
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped[0] in ("c", "C"):
            continue
        if stripped[0] in ("p", "P"):
            parts = stripped.split()
            if len(parts) < 4:
                raise ParseError(
                    f"malformed DIMACS 'p' line {stripped!r}", source=source, line=idx + 1
                )
            fmt = parts[1].lower()
            if fmt not in ("cnf", "wcnf"):
                raise ParseError(
                    f"unsupported DIMACS format {parts[1]!r} (cnf|wcnf)",
                    source=source,
                    line=idx + 1,
                )
            try:
                nvars = int(parts[2])
                nclauses = int(parts[3])
                top = float(parts[4]) if fmt == "wcnf" and len(parts) > 4 else None
            except ValueError as exc:
                raise ParseError(
                    f"non-integer counts in 'p' line {stripped!r}",
                    source=source,
                    line=idx + 1,
                ) from exc
            return fmt, nvars, nclauses, top, idx + 1
        raise ParseError(
            f"expected a DIMACS 'p' line before clauses, got {stripped!r}",
            source=source,
            line=idx + 1,
        )
    raise ParseError("missing DIMACS 'p cnf|wcnf' header", source=source)


def loads(text: str, *, name: str = "maxsat", source: str = "<string>") -> MILP:
    """Parse a DIMACS CNF/WCNF ``text`` into a clause-satisfaction :class:`~opop.model.ir.MILP`.

    Raises:
        ParseError: On a missing ``p`` line, an out-of-range variable id, or a
            clause not terminated by ``0`` before end of file.
    """
    lines = text.splitlines()
    fmt, nvars, nclauses, top, body_index = _parse_header(lines, source=source)
    if nvars < 1:
        raise ParseError(f"need nvars >= 1, got {nvars}", source=source)

    # Drop comment lines from the clause body so the token stream is clean.
    body = "\n".join(
        line for line in lines[body_index:] if line.strip()[:1] not in ("c", "C")
    )
    cursor = TokenCursor(body, source=source)

    clauses: list[tuple[float, bool, dict[str, float], float]] = []
    for c in range(nclauses):
        weight = cursor.next_float(f"weight of clause {c + 1}") if fmt == "wcnf" else 1.0
        is_hard = top is not None and weight >= top
        coeffs: dict[str, float] = {}
        neg_count = 0.0
        while True:
            lit = cursor.next_int(f"literal of clause {c + 1}")
            if lit == 0:
                break
            var = abs(lit)
            if not (1 <= var <= nvars):
                raise ParseError(
                    f"clause {c + 1} literal {lit} references variable {var} "
                    + f"out of range [1, {nvars}]",
                    source=source,
                    line=cursor.line,
                )
            vname = f"x_{var}"
            if lit > 0:
                coeffs[vname] = coeffs.get(vname, 0.0) + 1.0
            else:
                coeffs[vname] = coeffs.get(vname, 0.0) - 1.0
                neg_count += 1.0
        clauses.append((weight, is_hard, coeffs, neg_count))

    variables: list[Variable] = [
        Variable(f"x_{v}", VarType.BINARY, 0.0, 1.0) for v in range(1, nvars + 1)
    ]
    constraints: list[LinearConstraint] = []
    obj_coeffs: dict[str, float] = {}
    for c, (weight, is_hard, coeffs, neg_count) in enumerate(clauses):
        clean = {k: v for k, v in coeffs.items() if v != 0.0}
        if is_hard:
            constraints.append(
                LinearConstraint(f"hard_{c}", clean, ConstraintSense.GE, 1.0 - neg_count)
            )
        else:
            zname = f"z_{c}"
            variables.append(Variable(zname, VarType.BINARY, 0.0, 1.0))
            row = dict(clean)
            row[zname] = -1.0
            constraints.append(
                LinearConstraint(f"soft_{c}", row, ConstraintSense.GE, -neg_count)
            )
            obj_coeffs[zname] = weight

    objective = Objective(coeffs=obj_coeffs, sense=ObjSense.MAXIMIZE)
    ir = MILP(
        name=name,
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective=objective,
        metadata={
            "domain": "satisfiability",
            "formulation": "max_sat",
            "n_vars": nvars,
            "n_clauses": nclauses,
            "weighted": fmt == "wcnf",
        },
    )
    return tag_instance(ir, family="maxsat", source="maxsat", instance=name)


def load(path: str) -> MILP:
    """Load a DIMACS ``.cnf`` / ``.wcnf`` file into a MaxSAT :class:`~opop.model.ir.MILP`."""
    return loads(read_text(path), name=Path(path).stem, source=str(path))


register_adapter(ADAPTER)
