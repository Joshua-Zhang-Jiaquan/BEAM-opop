"""Deterministic synthetic MILP generators for the Phase-1 dev set.

Three classic combinatorial-optimization families, each built **directly** as a
:class:`opop.model.ir.MILP` (no solver needed for generation):

* :func:`generate_set_cover` — minimum-cost set covering (``>=`` rows, MIN).
* :func:`generate_knapsack` — 0/1 knapsack (one ``<=`` row, MAX).
* :func:`generate_facility` — uncapacitated facility location
  (assignment ``=`` rows + ``x <= y`` linking rows, MIN).

Every generator is **deterministic by seed**: it draws all randomness from a
single ``random.Random(seed)`` in a fixed order, so the same ``(params, seed)``
always yields a byte-identical model. :func:`canonical_milp_repr` produces a
stable textual fingerprint of a model (order-independent) used for content
checksums in the registry (see :mod:`opop.bench.sources.phase1_set`).

Size / structure are fully controllable through the generator arguments
(``n_rows``/``n_cols``/``density`` for set cover, ``n_items`` for knapsack,
``n_customers``/``n_facilities`` for facility location).
"""

from __future__ import annotations

import hashlib
import math
import random

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)

__all__ = [
    "canonical_milp_repr",
    "generate_facility",
    "generate_knapsack",
    "generate_set_cover",
    "milp_digest",
]


def _binary(name: str) -> Variable:
    """A 0/1 decision variable."""
    return Variable(name=name, vtype=VarType.BINARY, lower=0.0, upper=1.0)


# ---------------------------------------------------------------------------
# Set covering
# ---------------------------------------------------------------------------
def generate_set_cover(
    n_rows: int,
    n_cols: int,
    density: float,
    seed: int,
    *,
    name: str | None = None,
) -> MILP:
    """Generate a minimum-cost set-covering MILP.

    ``minimize sum_j c_j x_j  s.t.  for each row i: sum_{j covers i} x_j >= 1``,
    with ``x_j in {0, 1}``. Column ``j`` covers row ``i`` independently with
    probability ``density``; any row left uncovered is repaired by assigning one
    random column so the instance is always feasible.

    Args:
        n_rows: Number of elements to cover (``>= 1``) -> ``>=`` constraints.
        n_cols: Number of candidate sets / columns (``>= 1``) -> binary vars.
        density: Per-cell coverage probability in ``(0, 1]``.
        seed: RNG seed; identical seeds yield identical models.
        name: Optional model name (defaults to ``set_cover_{n_rows}x{n_cols}``).

    Returns:
        A :class:`MILP` with ``n_cols`` binary variables and ``n_rows``
        ``>=`` constraints, minimising total column cost.

    Raises:
        ValueError: If ``n_rows``/``n_cols`` ``< 1`` or ``density`` not in
            ``(0, 1]``.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"n_rows and n_cols must be >= 1 (got {n_rows}, {n_cols})")
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1] (got {density})")

    rng = random.Random(seed)
    costs = [rng.randint(1, 100) for _ in range(n_cols)]

    # cols_covering[i] = sorted list of columns covering row i.
    cols_covering: list[set[int]] = [set() for _ in range(n_rows)]
    for j in range(n_cols):
        for i in range(n_rows):
            if rng.random() < density:
                cols_covering[i].add(j)
    for i in range(n_rows):
        if not cols_covering[i]:
            cols_covering[i].add(rng.randrange(n_cols))

    variables = tuple(_binary(f"x_{j}") for j in range(n_cols))
    constraints = tuple(
        LinearConstraint(
            name=f"cover_{i}",
            coeffs={f"x_{j}": 1.0 for j in sorted(cols_covering[i])},
            sense=ConstraintSense.GE,
            rhs=1.0,
        )
        for i in range(n_rows)
    )
    objective = Objective(
        coeffs={f"x_{j}": float(costs[j]) for j in range(n_cols)},
        sense=ObjSense.MINIMIZE,
    )
    return MILP(
        name=name or f"set_cover_{n_rows}x{n_cols}",
        variables=variables,
        constraints=constraints,
        objective=objective,
        metadata={
            "family": "set_cover",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "density": density,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# 0/1 Knapsack
# ---------------------------------------------------------------------------
def generate_knapsack(n_items: int, seed: int, *, name: str | None = None) -> MILP:
    """Generate a 0/1 knapsack MILP.

    ``maximize sum_i v_i x_i  s.t.  sum_i w_i x_i <= capacity``, ``x_i in {0,1}``.
    Weights and values are drawn in ``[1, 100]``; capacity is half the total
    weight (at least 1) so the constraint is binding but feasible.

    Args:
        n_items: Number of items (``>= 1``).
        seed: RNG seed; identical seeds yield identical models.
        name: Optional model name (defaults to ``knapsack_{n_items}``).

    Returns:
        A :class:`MILP` with ``n_items`` binary variables and a single ``<=``
        capacity constraint, maximising total value.

    Raises:
        ValueError: If ``n_items < 1``.
    """
    if n_items < 1:
        raise ValueError(f"n_items must be >= 1 (got {n_items})")

    rng = random.Random(seed)
    weights = [rng.randint(1, 100) for _ in range(n_items)]
    values = [rng.randint(1, 100) for _ in range(n_items)]
    capacity = max(1, sum(weights) // 2)

    variables = tuple(_binary(f"x_{i}") for i in range(n_items))
    constraints = (
        LinearConstraint(
            name="capacity",
            coeffs={f"x_{i}": float(weights[i]) for i in range(n_items)},
            sense=ConstraintSense.LE,
            rhs=float(capacity),
        ),
    )
    objective = Objective(
        coeffs={f"x_{i}": float(values[i]) for i in range(n_items)},
        sense=ObjSense.MAXIMIZE,
    )
    return MILP(
        name=name or f"knapsack_{n_items}",
        variables=variables,
        constraints=constraints,
        objective=objective,
        metadata={"family": "knapsack", "n_items": n_items, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Uncapacitated facility location
# ---------------------------------------------------------------------------
def generate_facility(
    n_customers: int,
    n_facilities: int,
    seed: int,
    *,
    name: str | None = None,
) -> MILP:
    """Generate an uncapacitated facility-location MILP.

    ``minimize sum_f o_f y_f + sum_{c,f} s_{c,f} x_{c,f}`` subject to
    ``sum_f x_{c,f} = 1`` for each customer ``c`` (assigned to exactly one open
    facility) and ``x_{c,f} <= y_f`` (serve only from open facilities), with all
    variables binary. Opening costs ``o_f`` are drawn in ``[10, 100]`` and
    service costs ``s_{c,f}`` in ``[1, 50]``.

    Args:
        n_customers: Number of customers (``>= 1``) -> assignment rows.
        n_facilities: Number of candidate facilities (``>= 1``) -> ``y`` vars.
        seed: RNG seed; identical seeds yield identical models.
        name: Optional model name (defaults to ``facility_{nc}x{nf}``).

    Returns:
        A :class:`MILP` with ``n_facilities + n_customers*n_facilities`` binary
        variables, ``n_customers`` ``=`` rows and ``n_customers*n_facilities``
        linking ``<=`` rows, minimising open + service cost.

    Raises:
        ValueError: If ``n_customers`` or ``n_facilities`` ``< 1``.
    """
    if n_customers < 1 or n_facilities < 1:
        raise ValueError(
            f"n_customers and n_facilities must be >= 1 (got {n_customers}, {n_facilities})"
        )

    rng = random.Random(seed)
    open_cost = [rng.randint(10, 100) for _ in range(n_facilities)]
    serve_cost = [
        [rng.randint(1, 50) for _ in range(n_facilities)] for _ in range(n_customers)
    ]

    y_vars = [_binary(f"y_{f}") for f in range(n_facilities)]
    x_vars = [
        _binary(f"x_{c}_{f}") for c in range(n_customers) for f in range(n_facilities)
    ]
    variables = tuple(y_vars + x_vars)

    assign_rows = tuple(
        LinearConstraint(
            name=f"assign_{c}",
            coeffs={f"x_{c}_{f}": 1.0 for f in range(n_facilities)},
            sense=ConstraintSense.EQ,
            rhs=1.0,
        )
        for c in range(n_customers)
    )
    link_rows = tuple(
        LinearConstraint(
            name=f"link_{c}_{f}",
            coeffs={f"x_{c}_{f}": 1.0, f"y_{f}": -1.0},
            sense=ConstraintSense.LE,
            rhs=0.0,
        )
        for c in range(n_customers)
        for f in range(n_facilities)
    )

    obj_coeffs: dict[str, float] = {f"y_{f}": float(open_cost[f]) for f in range(n_facilities)}
    for c in range(n_customers):
        for f in range(n_facilities):
            obj_coeffs[f"x_{c}_{f}"] = float(serve_cost[c][f])

    return MILP(
        name=name or f"facility_{n_customers}x{n_facilities}",
        variables=variables,
        constraints=assign_rows + link_rows,
        objective=Objective(coeffs=obj_coeffs, sense=ObjSense.MINIMIZE),
        metadata={
            "family": "facility",
            "n_customers": n_customers,
            "n_facilities": n_facilities,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Canonical fingerprint / content digest
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    """Stable float formatting for the canonical representation."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def canonical_milp_repr(milp: MILP) -> str:
    """Return a stable, order-independent textual fingerprint of a model.

    Variables, constraints, and coefficients are emitted in sorted-by-name order
    so the result depends only on the *math model* (objective sense/offset/
    coefficients, variable domains/bounds, constraint senses/rhs/coefficients) —
    never on construction order, ``name``, ``index_sets``, or ``metadata``.
    """
    lines: list[str] = [
        f"OBJ {milp.objective.sense.value} offset={_fmt(milp.objective.offset)}",
    ]
    for vname in sorted(milp.objective.coeffs):
        lines.append(f"OBJC {vname} {_fmt(milp.objective.coeffs[vname])}")
    for var in sorted(milp.variables, key=lambda v: v.name):
        lines.append(f"VAR {var.name} {var.vtype.value} {_fmt(var.lower)} {_fmt(var.upper)}")
    for con in sorted(milp.constraints, key=lambda c: c.name):
        body = " ".join(f"{k}:{_fmt(con.coeffs[k])}" for k in sorted(con.coeffs))
        lines.append(f"CON {con.name} {con.sense.value} {_fmt(con.rhs)} {body}")
    return "\n".join(lines)


def milp_digest(milp: MILP) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_milp_repr` for ``milp``."""
    return hashlib.sha256(canonical_milp_repr(milp).encode("utf-8")).hexdigest()
