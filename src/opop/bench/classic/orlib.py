"""OR-Library set-covering loader → generic set-cover MILP (task 34).

Parses the classic `OR-Library <https://www.brunel.ac.uk/~mastjjb/jeb/info.html>`_
set-covering problem (SCP / ``scp*.txt``) format and maps it to the OPOP IR via
the textbook set-cover binary program (no per-instance tuning):

* binary column variables ``x_j`` (``j = 0 .. n-1``);
* one ``>=`` row per element ``i``: ``sum_{j covers i} x_j >= 1``;
* objective ``minimise sum_j c_j x_j``.

The Beasley SCP layout is a free-form integer stream::

    m n
    c_1 c_2 ... c_n                      # n column costs (may wrap lines)
    k_1  col col ... (k_1 entries)       # for each row i: count then covering cols
    ...

An out-of-range column index, a non-integer token, or a truncated file raises
:class:`~opop.bench.classic.base.ParseError` with file + line context.
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

#: The registered classic-CO adapter for the OR-Library set-covering family.
ADAPTER = ClassicAdapter(family="orlib", problem_class="SCP")


def loads(text: str, *, name: str = "scp", source: str = "<string>") -> MILP:
    """Parse an OR-Library SCP ``text`` into a set-cover :class:`~opop.model.ir.MILP`.

    Raises:
        ParseError: On a non-integer token, an out-of-range column index, or a
            file truncated before all costs / rows are read.
    """
    cursor = TokenCursor(text, source=source)
    n_rows = cursor.next_int("row count m")
    n_cols = cursor.next_int("column count n")
    if n_rows < 1 or n_cols < 1:
        raise ParseError(
            f"set covering needs m >= 1 and n >= 1, got m={n_rows}, n={n_cols}",
            source=source,
            line=cursor.line,
        )

    costs = [float(cursor.next_int(f"cost of column {j + 1}")) for j in range(n_cols)]

    rows: list[list[int]] = []
    for i in range(n_rows):
        k = cursor.next_int(f"covering-column count for row {i + 1}")
        if k < 0:
            raise ParseError(
                f"row {i + 1} has a negative covering count {k}",
                source=source,
                line=cursor.line,
            )
        cover: list[int] = []
        for _ in range(k):
            col = cursor.next_int(f"covering column for row {i + 1}")
            if not (1 <= col <= n_cols):
                raise ParseError(
                    f"row {i + 1} references column {col} out of range [1, {n_cols}]",
                    source=source,
                    line=cursor.line,
                )
            cover.append(col - 1)
        rows.append(cover)

    variables = tuple(Variable(f"x_{j}", VarType.BINARY, 0.0, 1.0) for j in range(n_cols))
    constraints = tuple(
        LinearConstraint(
            f"cover_{i}",
            {f"x_{j}": 1.0 for j in sorted(set(rows[i]))},
            ConstraintSense.GE,
            1.0,
        )
        for i in range(n_rows)
    )
    objective = Objective(
        coeffs={f"x_{j}": costs[j] for j in range(n_cols)},
        sense=ObjSense.MINIMIZE,
    )
    ir = MILP(
        name=name,
        variables=variables,
        constraints=constraints,
        objective=objective,
        metadata={"domain": "covering", "formulation": "set_cover", "n_rows": n_rows, "n_cols": n_cols},
    )
    return tag_instance(ir, family="orlib", source="or-library", instance=name)


def load(path: str) -> MILP:
    """Load an OR-Library SCP ``.txt`` file into a set-cover :class:`~opop.model.ir.MILP`."""
    return loads(read_text(path), name=Path(path).stem, source=str(path))


register_adapter(ADAPTER)
