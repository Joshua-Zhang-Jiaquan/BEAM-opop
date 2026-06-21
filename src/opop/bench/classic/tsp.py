"""TSPLIB loader → generic Miller–Tucker–Zemlin (MTZ) MILP (task 34).

Parses the `TSPLIB <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>`_
symmetric-TSP format and maps it to the OPOP IR through ONE generic compact
formulation — Miller–Tucker–Zemlin subtour elimination — applied uniformly to
every instance (no per-instance hand-tuning):

* ``minimise sum_{i!=j} c_ij x_ij`` over binary arcs ``x_ij``;
* degree rows ``sum_i x_ij == 1`` (in) and ``sum_j x_ij == 1`` (out);
* MTZ rows ``u_i - u_j + (n-1) x_ij <= n-2`` for ``i, j in 1..n-1`` with
  continuous potentials ``u_i in [1, n-1]``.

The feasible integer set is exactly the Hamiltonian tours, so the MILP optimum
is the TSP optimum (a tiny instance solves to the known optimum — locked by the
tests). Supported edge-weight encodings: ``EUC_2D`` (rounded Euclidean from a
``NODE_COORD_SECTION``) and ``EXPLICIT`` + ``FULL_MATRIX`` (an explicit
``EDGE_WEIGHT_SECTION``). Any other encoding, a missing ``DIMENSION``, or a
truncated section raises :class:`~opop.bench.classic.base.ParseError` with file +
line context.
"""

from __future__ import annotations

from opop.bench.classic.base import (
    ClassicAdapter,
    ParseError,
    TokenCursor,
    euclidean_matrix,
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

#: The registered classic-CO adapter for the TSP family.
ADAPTER = ClassicAdapter(family="tsp", problem_class="TSP")

_SECTION_KEYS = frozenset(
    {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION", "EOF"}
)


def _arc(i: int, j: int) -> str:
    return f"x_{i}_{j}"


def _build_mtz(
    n: int, cost: dict[tuple[int, int], float], *, name: str
) -> MILP:
    """Assemble the generic MTZ MILP for ``n`` nodes with arc costs ``cost``."""
    if n < 3:
        raise ParseError(f"TSP needs DIMENSION >= 3, got {n}", source=name)

    variables: list[Variable] = [
        Variable(_arc(i, j), VarType.BINARY, 0.0, 1.0)
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    variables.extend(
        Variable(f"u_{i}", VarType.CONTINUOUS, 1.0, float(n - 1)) for i in range(1, n)
    )

    constraints: list[LinearConstraint] = []
    for j in range(n):
        constraints.append(
            LinearConstraint(
                f"indeg_{j}",
                {_arc(i, j): 1.0 for i in range(n) if i != j},
                ConstraintSense.EQ,
                1.0,
            )
        )
    for i in range(n):
        constraints.append(
            LinearConstraint(
                f"outdeg_{i}",
                {_arc(i, j): 1.0 for j in range(n) if j != i},
                ConstraintSense.EQ,
                1.0,
            )
        )
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints.append(
                    LinearConstraint(
                        f"mtz_{i}_{j}",
                        {f"u_{i}": 1.0, f"u_{j}": -1.0, _arc(i, j): float(n - 1)},
                        ConstraintSense.LE,
                        float(n - 2),
                    )
                )

    objective = Objective(
        coeffs={_arc(i, j): cost[(i, j)] for i in range(n) for j in range(n) if i != j},
        sense=ObjSense.MINIMIZE,
    )
    return MILP(
        name=name,
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective=objective,
        metadata={"domain": "routing", "formulation": "mtz", "n_nodes": n},
    )


def _split_header(line: str) -> tuple[str, str] | None:
    """Split a ``KEY : VALUE`` header line; ``None`` if it is not a header."""
    if ":" not in line:
        return None
    key, _, value = line.partition(":")
    return key.strip().upper(), value.strip()


def _read_coords(
    lines: list[str], start: int, n: int, *, source: str
) -> list[tuple[float, float]]:
    """Read ``n`` ``id x y`` coordinate rows starting at line index ``start``."""
    coords: list[tuple[float, float]] = []
    idx = start
    while idx < len(lines) and len(coords) < n:
        stripped = lines[idx].strip()
        idx += 1
        if not stripped:
            continue
        if stripped.upper() in _SECTION_KEYS:
            break
        parts = stripped.split()
        if len(parts) < 3:
            raise ParseError(
                f"coordinate row needs 'id x y', got {stripped!r}",
                source=source,
                line=idx,
            )
        try:
            coords.append((float(parts[1]), float(parts[2])))
        except ValueError as exc:
            raise ParseError(
                f"non-numeric coordinate in {stripped!r}", source=source, line=idx
            ) from exc
    if len(coords) != n:
        raise ParseError(
            f"expected {n} coordinates, got {len(coords)}", source=source, line=idx
        )
    return coords


def _explicit_matrix(body: str, n: int, *, source: str) -> dict[tuple[int, int], float]:
    """Read an ``EXPLICIT`` + ``FULL_MATRIX`` weight section (``n*n`` numbers)."""
    cursor = TokenCursor(body, source=source)
    cost: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(n):
            value = cursor.next_float(f"weight ({i},{j})")
            if i != j:
                cost[(i, j)] = value
    return cost


def loads(text: str, *, name: str = "tsp", source: str = "<string>") -> MILP:
    """Parse TSPLIB ``text`` into a generic MTZ :class:`~opop.model.ir.MILP`.

    Raises:
        ParseError: On a missing ``DIMENSION``, an unsupported edge-weight
            encoding, or a truncated coordinate / weight section.
    """
    lines = text.splitlines()
    dimension: int | None = None
    weight_type = "EUC_2D"
    weight_format = "FULL_MATRIX"

    idx = 0
    coords: list[tuple[float, float]] | None = None
    explicit_cost: dict[tuple[int, int], float] | None = None

    while idx < len(lines):
        raw = lines[idx].strip()
        idx += 1
        if not raw:
            continue
        upper = raw.upper()
        if upper in _SECTION_KEYS:
            if upper == "EOF":
                break
            if dimension is None:
                raise ParseError(
                    "DIMENSION must precede the data section", source=source, line=idx
                )
            if upper == "NODE_COORD_SECTION":
                coords = _read_coords(lines, idx, dimension, source=source)
                break
            if upper == "EDGE_WEIGHT_SECTION":
                if weight_type != "EXPLICIT":
                    raise ParseError(
                        "EDGE_WEIGHT_SECTION requires EDGE_WEIGHT_TYPE EXPLICIT, "
                        + f"got {weight_type!r}",
                        source=source,
                        line=idx,
                    )
                if weight_format != "FULL_MATRIX":
                    raise ParseError(
                        f"unsupported EDGE_WEIGHT_FORMAT {weight_format!r} "
                        + "(only FULL_MATRIX is supported)",
                        source=source,
                        line=idx,
                    )
                explicit_cost = _explicit_matrix(
                    "\n".join(lines[idx:]), dimension, source=source
                )
                break
            continue
        header = _split_header(raw)
        if header is None:
            continue
        key, value = header
        if key == "DIMENSION":
            try:
                dimension = int(value)
            except ValueError as exc:
                raise ParseError(
                    f"DIMENSION must be an integer, got {value!r}", source=source, line=idx
                ) from exc
        elif key == "EDGE_WEIGHT_TYPE":
            weight_type = value.upper()
        elif key == "EDGE_WEIGHT_FORMAT":
            weight_format = value.upper()

    if dimension is None:
        raise ParseError("missing DIMENSION header", source=source, line=len(lines) or 1)

    if explicit_cost is not None:
        cost = explicit_cost
    elif coords is not None:
        if weight_type != "EUC_2D":
            raise ParseError(
                f"unsupported EDGE_WEIGHT_TYPE {weight_type!r} for coordinates "
                + "(only EUC_2D is supported)",
                source=source,
            )
        cost = euclidean_matrix(coords)
    else:
        raise ParseError(
            "no NODE_COORD_SECTION or EDGE_WEIGHT_SECTION found", source=source
        )

    ir = _build_mtz(dimension, cost, name=name)
    return tag_instance(ir, family="tsp", source="tsplib", instance=name)


def load(path: str) -> MILP:
    """Load a TSPLIB ``.tsp`` file into a generic MTZ :class:`~opop.model.ir.MILP`."""
    from pathlib import Path

    return loads(read_text(path), name=Path(path).stem, source=str(path))


register_adapter(ADAPTER)
