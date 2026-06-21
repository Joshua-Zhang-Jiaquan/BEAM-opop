"""CVRPLIB loader → generic two-index MTZ-capacity CVRP MILP (task 34).

Parses the `CVRPLIB <http://vrp.galgos.inf.puc-rio.br/>`_ capacitated
vehicle-routing format (``DIMENSION`` / ``CAPACITY`` headers + ``NODE_COORD_SECTION``
/ ``DEMAND_SECTION`` / ``DEPOT_SECTION``) and maps it to the OPOP IR via ONE
generic two-index vehicle-flow formulation with Miller–Tucker–Zemlin capacity
elimination (applied uniformly — no per-instance hand-tuning):

* binary arcs ``x_ij`` (``i != j``);
* customer degree rows ``sum_i x_ij == 1`` / ``sum_j x_ij == 1``;
* depot degree rows ``sum_j x_0j == K`` / ``sum_i x_i0 == K`` where ``K`` is the
  minimum vehicle count ``ceil(sum demand / capacity)``;
* MTZ capacity rows ``u_i - u_j + Q x_ij <= Q - d_j`` for customers ``i != j``
  with cumulative-load potentials ``u_i in [d_i, Q]`` (these forbid subtours AND
  enforce the capacity ``Q``);
* objective ``minimise sum_{i!=j} c_ij x_ij`` with rounded-Euclidean costs.

A missing ``DIMENSION`` / ``CAPACITY``, a truncated section, or a bad depot
terminator raises :class:`~opop.bench.classic.base.ParseError` with file + line
context.
"""

from __future__ import annotations

import math
from pathlib import Path

from opop.bench.classic.base import (
    ClassicAdapter,
    ParseError,
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

#: The registered classic-CO adapter for the CVRP family.
ADAPTER = ClassicAdapter(family="cvrp", problem_class="CVRP")

_SECTIONS = frozenset({"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"})


def _arc(i: int, j: int) -> str:
    return f"x_{i}_{j}"


def _split_header(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, _, value = line.partition(":")
    return key.strip().upper(), value.strip()


def _build_cvrp(
    n: int,
    depot: int,
    capacity: float,
    demand: list[float],
    cost: dict[tuple[int, int], float],
    *,
    name: str,
) -> MILP:
    """Assemble the generic two-index MTZ-capacity CVRP MILP."""
    customers = [i for i in range(n) if i != depot]
    if not customers:
        raise ParseError("CVRP needs at least one customer node", source=name)
    total_demand = sum(demand[i] for i in customers)
    n_vehicles = max(1, math.ceil(total_demand / capacity)) if capacity > 0 else 1

    variables: list[Variable] = [
        Variable(_arc(i, j), VarType.BINARY, 0.0, 1.0)
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    variables.extend(
        Variable(f"u_{i}", VarType.CONTINUOUS, demand[i], capacity) for i in customers
    )

    constraints: list[LinearConstraint] = []
    for j in range(n):
        rhs = float(n_vehicles) if j == depot else 1.0
        constraints.append(
            LinearConstraint(
                f"indeg_{j}",
                {_arc(i, j): 1.0 for i in range(n) if i != j},
                ConstraintSense.EQ,
                rhs,
            )
        )
    for i in range(n):
        rhs = float(n_vehicles) if i == depot else 1.0
        constraints.append(
            LinearConstraint(
                f"outdeg_{i}",
                {_arc(i, j): 1.0 for j in range(n) if j != i},
                ConstraintSense.EQ,
                rhs,
            )
        )
    for i in customers:
        for j in customers:
            if i != j:
                constraints.append(
                    LinearConstraint(
                        f"cap_{i}_{j}",
                        {f"u_{i}": 1.0, f"u_{j}": -1.0, _arc(i, j): capacity},
                        ConstraintSense.LE,
                        capacity - demand[j],
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
        metadata={
            "domain": "routing",
            "formulation": "cvrp_mtz",
            "n_nodes": n,
            "n_vehicles": n_vehicles,
            "capacity": capacity,
        },
    )


def _read_pairs(
    lines: list[str], start: int, n: int, what: str, *, source: str
) -> tuple[list[tuple[float, float]], int]:
    """Read ``n`` ``id v1 [v2]`` rows; return (values, next_line_index)."""
    out: list[tuple[float, float]] = []
    idx = start
    while idx < len(lines) and len(out) < n:
        stripped = lines[idx].strip()
        idx += 1
        if not stripped:
            continue
        if stripped.upper() in _SECTIONS:
            break
        parts = stripped.split()
        if len(parts) < 2:
            raise ParseError(
                f"{what} row needs at least 'id value', got {stripped!r}",
                source=source,
                line=idx,
            )
        try:
            second = float(parts[1])
            third = float(parts[2]) if len(parts) > 2 else 0.0
        except ValueError as exc:
            raise ParseError(
                f"non-numeric {what} in {stripped!r}", source=source, line=idx
            ) from exc
        out.append((second, third))
    if len(out) != n:
        raise ParseError(
            f"expected {n} {what} rows, got {len(out)}", source=source, line=idx
        )
    return out, idx


def loads(text: str, *, name: str = "cvrp", source: str = "<string>") -> MILP:
    """Parse CVRPLIB ``text`` into a generic MTZ-capacity :class:`~opop.model.ir.MILP`.

    Raises:
        ParseError: On a missing ``DIMENSION`` / ``CAPACITY``, a truncated
            section, or a malformed coordinate / demand / depot row.
    """
    lines = text.splitlines()
    dimension: int | None = None
    capacity: float | None = None
    weight_type = "EUC_2D"
    coords: list[tuple[float, float]] | None = None
    demands: list[float] | None = None
    depot: int | None = None

    idx = 0
    while idx < len(lines):
        raw = lines[idx].strip()
        idx += 1
        if not raw:
            continue
        upper = raw.upper()
        if upper == "EOF":
            break
        if upper in _SECTIONS:
            if dimension is None:
                raise ParseError(
                    "DIMENSION must precede the data sections", source=source, line=idx
                )
            if upper == "NODE_COORD_SECTION":
                pairs, idx = _read_pairs(lines, idx, dimension, "coordinate", source=source)
                coords = pairs
            elif upper == "DEMAND_SECTION":
                pairs, idx = _read_pairs(lines, idx, dimension, "demand", source=source)
                demands = [d for d, _ in pairs]
            elif upper == "DEPOT_SECTION":
                depot, idx = _read_depot(lines, idx, source=source)
            continue
        header = _split_header(raw)
        if header is None:
            continue
        key, value = header
        if key == "DIMENSION":
            dimension = _parse_int(value, "DIMENSION", source=source, line=idx)
        elif key == "CAPACITY":
            capacity = float(_parse_int(value, "CAPACITY", source=source, line=idx))
        elif key == "EDGE_WEIGHT_TYPE":
            weight_type = value.upper()

    if dimension is None:
        raise ParseError("missing DIMENSION header", source=source, line=len(lines) or 1)
    if capacity is None:
        raise ParseError("missing CAPACITY header", source=source, line=len(lines) or 1)
    if coords is None:
        raise ParseError("missing NODE_COORD_SECTION", source=source)
    if demands is None:
        raise ParseError("missing DEMAND_SECTION", source=source)
    if weight_type != "EUC_2D":
        raise ParseError(
            f"unsupported EDGE_WEIGHT_TYPE {weight_type!r} (only EUC_2D is supported)",
            source=source,
        )
    if depot is None:
        depot = 0

    cost = euclidean_matrix(coords)
    ir = _build_cvrp(dimension, depot, capacity, demands, cost, name=name)
    return tag_instance(ir, family="cvrp", source="cvrplib", instance=name)


def _read_depot(lines: list[str], start: int, *, source: str) -> tuple[int, int]:
    """Read the ``DEPOT_SECTION`` (depot ids until ``-1``); return (depot_index, next)."""
    idx = start
    depot: int | None = None
    while idx < len(lines):
        stripped = lines[idx].strip()
        idx += 1
        if not stripped:
            continue
        if stripped.upper() in _SECTIONS:
            idx -= 1
            break
        token = stripped.split()[0]
        try:
            value = int(token)
        except ValueError as exc:
            raise ParseError(
                f"depot id must be an integer, got {token!r}", source=source, line=idx
            ) from exc
        if value == -1:
            break
        if depot is None:
            depot = value - 1  # CVRPLIB depot ids are 1-based.
    if depot is None:
        raise ParseError("DEPOT_SECTION has no depot id", source=source, line=idx)
    return depot, idx


def _parse_int(value: str, what: str, *, source: str, line: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ParseError(
            f"{what} must be an integer, got {value!r}", source=source, line=line
        ) from exc


def load(path: str) -> MILP:
    """Load a CVRPLIB ``.vrp`` file into a generic CVRP :class:`~opop.model.ir.MILP`."""
    return loads(read_text(path), name=Path(path).stem, source=str(path))


register_adapter(ADAPTER)
