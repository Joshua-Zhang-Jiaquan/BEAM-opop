"""Quadratic extension to the OPOP MILP IR: QUBO / Ising / quadratic terms (task 30).

This module is the *pure, solver-free* foundation for the Wave-5 generality
layer. It adds an OPTIONAL quadratic layer on top of the linear MILP IR
(:mod:`opop.model.ir`) WITHOUT touching any linear record:

* :class:`QuadraticTerm` / :class:`QuadraticExtension` — the additive quadratic
  IR records. These are DEFINED in :mod:`opop.model.ir` (they are part of the IR
  and co-locating them keeps the model package's import graph acyclic) and
  re-exported here so callers can treat :mod:`opop.model.quadratic` as the single
  entry point for the quadratic layer.
* :class:`QUBO` / :class:`Ising` — the two canonical unconstrained binary /
  spin energy models, with EXACT :func:`qubo_to_ising` / :func:`ising_to_qubo`
  conversions under the ``x_i = (1 - s_i) / 2`` mapping (spin ``+1`` <-> bit
  ``0``, spin ``-1`` <-> bit ``1``). :func:`qubo_energy` / :func:`ising_energy`
  let callers verify the conversion preserves every assignment's energy.
* IR bridges :func:`qubo_to_ir` / :func:`ir_to_qubo` (a QUBO is a pure-binary
  MILP whose objective carries quadratic terms) and the :func:`max_cut_qubo`
  problem builder.

Nothing here imports a solver: linearization and solver routing live in the
``opop.solver`` layer (:mod:`opop.solver.qubo`, :mod:`opop.solver.miqp`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    QuadraticExtension,
    QuadraticTerm,
    UnsupportedModelError,
    Variable,
    VarType,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

__all__ = [
    "QUBO",
    "Ising",
    "QuadraticExtension",
    "QuadraticTerm",
    "bits_from_spins",
    "ir_to_qubo",
    "ising_energy",
    "ising_to_qubo",
    "linearize_quadratic",
    "max_cut_qubo",
    "qubo_energy",
    "qubo_to_ir",
    "qubo_to_ising",
    "spins_from_bits",
]


# ---------------------------------------------------------------------------
# QUBO / Ising energy models
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QUBO:
    """Quadratic Unconstrained Binary Optimization (minimisation form).

    Energy ``E(x) = offset + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j`` over
    ``x in {0, 1}^n``. The diagonal ``Q_ii`` (since ``x_i^2 == x_i`` for binary
    ``x``) is stored as the *linear* coefficient ``a_i``.

    Attributes:
        linear: Mapping ``variable -> a_i`` (diagonal / linear coefficient).
        quadratic: Mapping ``(min, max) variable pair -> b_ij`` (off-diagonal).
        offset: Constant energy offset.
    """

    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)
    offset: float = 0.0

    def variables(self) -> tuple[str, ...]:
        """Return all variable names (linear + quadratic), sorted."""
        names: set[str] = set(self.linear)
        for i, j in self.quadratic:
            names.add(i)
            names.add(j)
        return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class Ising:
    """Ising spin model (minimisation form).

    Energy ``E(s) = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j`` over
    ``s in {-1, +1}^n``.

    Attributes:
        h: Mapping ``variable -> h_i`` (external field).
        J: Mapping ``(min, max) variable pair -> J_ij`` (coupling).
        offset: Constant energy offset.
    """

    h: dict[str, float] = field(default_factory=dict)
    J: dict[tuple[str, str], float] = field(default_factory=dict)
    offset: float = 0.0

    def variables(self) -> tuple[str, ...]:
        """Return all variable names (field + coupling), sorted."""
        names: set[str] = set(self.h)
        for i, j in self.J:
            names.add(i)
            names.add(j)
        return tuple(sorted(names))


def _canonical_pair(i: str, j: str) -> tuple[str, str]:
    """Return the order-independent ``(min, max)`` pair key."""
    return (i, j) if i <= j else (j, i)


def _incident_sum(pairs: Mapping[tuple[str, str], float], name: str) -> float:
    """Sum the coefficients of every quadratic pair that contains ``name``."""
    total = 0.0
    for (i, j), coeff in pairs.items():
        if i == name or j == name:
            total += coeff
    return total


def qubo_energy(qubo: QUBO, assignment: Mapping[str, float]) -> float:
    """Evaluate the QUBO energy at a binary ``assignment`` (var -> 0/1)."""
    energy = qubo.offset
    for name, coeff in qubo.linear.items():
        energy += coeff * assignment[name]
    for (i, j), coeff in qubo.quadratic.items():
        energy += coeff * assignment[i] * assignment[j]
    return energy


def ising_energy(ising: Ising, spins: Mapping[str, float]) -> float:
    """Evaluate the Ising energy at a spin assignment (var -> -1/+1)."""
    energy = ising.offset
    for name, coeff in ising.h.items():
        energy += coeff * spins[name]
    for (i, j), coeff in ising.J.items():
        energy += coeff * spins[i] * spins[j]
    return energy


def spins_from_bits(bits: Mapping[str, float]) -> dict[str, float]:
    """Map a binary assignment to spins via ``s_i = 1 - 2 x_i`` (x=0->+1, x=1->-1)."""
    return {name: 1.0 - 2.0 * value for name, value in bits.items()}


def bits_from_spins(spins: Mapping[str, float]) -> dict[str, float]:
    """Map spins to a binary assignment via ``x_i = (1 - s_i) / 2`` (+1->0, -1->1)."""
    return {name: (1.0 - value) / 2.0 for name, value in spins.items()}


def qubo_to_ising(qubo: QUBO) -> Ising:
    """Convert a :class:`QUBO` to an energy-equivalent :class:`Ising` model.

    Uses the substitution ``x_i = (1 - s_i) / 2`` (spin ``+1`` <-> bit ``0``,
    spin ``-1`` <-> bit ``1``), which yields::

        J_ij   = b_ij / 4
        h_i    = -a_i / 2 - (1/4) * sum_{j != i} b_ij
        offset = qubo.offset + sum_i a_i / 2 + sum_{i<j} b_ij / 4

    The conversion is exact: ``ising_energy(out, spins_from_bits(x)) ==
    qubo_energy(qubo, x)`` for every binary ``x``.
    """
    coupling: dict[tuple[str, str], float] = {
        _canonical_pair(i, j): coeff / 4.0 for (i, j), coeff in qubo.quadratic.items()
    }
    field_term: dict[str, float] = {}
    for name in qubo.variables():
        a_i = qubo.linear.get(name, 0.0)
        field_term[name] = -a_i / 2.0 - 0.25 * _incident_sum(qubo.quadratic, name)
    offset = (
        qubo.offset
        + sum(value / 2.0 for value in qubo.linear.values())
        + sum(coeff / 4.0 for coeff in qubo.quadratic.values())
    )
    return Ising(h=field_term, J=coupling, offset=offset)


def ising_to_qubo(ising: Ising) -> QUBO:
    """Convert an :class:`Ising` model to an energy-equivalent :class:`QUBO`.

    Inverse of :func:`qubo_to_ising` under ``s_i = 1 - 2 x_i``::

        b_ij   = 4 * J_ij
        a_i    = -2 * h_i - 2 * sum_{j != i} J_ij
        offset = ising.offset + sum_i h_i + sum_{i<j} J_ij

    The conversion is exact: ``qubo_energy(out, bits_from_spins(s)) ==
    ising_energy(ising, s)`` for every spin assignment ``s``.
    """
    quadratic: dict[tuple[str, str], float] = {
        _canonical_pair(i, j): 4.0 * coeff for (i, j), coeff in ising.J.items()
    }
    linear: dict[str, float] = {}
    for name in ising.variables():
        h_i = ising.h.get(name, 0.0)
        linear[name] = -2.0 * h_i - 2.0 * _incident_sum(ising.J, name)
    offset = ising.offset + sum(ising.h.values()) + sum(ising.J.values())
    return QUBO(linear=linear, quadratic=quadratic, offset=offset)


# ---------------------------------------------------------------------------
# QUBO <-> MILP IR bridges
# ---------------------------------------------------------------------------
def qubo_to_ir(
    qubo: QUBO, *, name: str = "qubo", sense: ObjSense = ObjSense.MINIMIZE
) -> MILP:
    """Build a pure-binary :class:`~opop.model.ir.MILP` carrying the QUBO.

    Every variable is BINARY ``[0, 1]``; the linear objective holds the diagonal
    ``a_i`` and the :class:`QuadraticExtension` objective terms hold the
    off-diagonal ``b_ij``. The result is the *unlinearized* quadratic IR — call
    :func:`opop.solver.qubo.QuboAdapter.to_milp` to obtain a plain linear MILP.
    """
    names = qubo.variables()
    variables = tuple(Variable(n, VarType.BINARY, 0.0, 1.0) for n in names)
    linear_coeffs = {n: qubo.linear[n] for n in names if qubo.linear.get(n, 0.0) != 0.0}
    objective = Objective(coeffs=linear_coeffs, sense=sense, offset=qubo.offset)
    obj_terms = tuple(
        QuadraticTerm(i, j, coeff) for (i, j), coeff in qubo.quadratic.items()
    )
    extension = QuadraticExtension(objective_terms=obj_terms)
    return MILP(
        name=name,
        variables=variables,
        constraints=(),
        objective=objective,
        quadratic=extension,
    )


def ir_to_qubo(ir: MILP) -> QUBO:
    """Extract a :class:`QUBO` from a QUBO-shaped :class:`~opop.model.ir.MILP`.

    Requires all-binary variables, a quadratic objective, and NO quadratic
    constraints. Square terms (``x_i^2``) are folded into the linear part since
    ``x_i^2 == x_i`` for binary ``x``. Raises :class:`ValueError` otherwise.
    """
    if ir.quadratic is None or not ir.quadratic.has_objective_terms():
        raise ValueError("ir_to_qubo requires an MILP with quadratic objective terms")
    if ir.quadratic.has_constraint_terms():
        raise ValueError(
            "ir_to_qubo requires an unconstrained-quadratic (QUBO) model; "
            + "the IR carries quadratic constraint terms"
        )
    nonbinary = [v.name for v in ir.variables if v.vtype is not VarType.BINARY]
    if nonbinary:
        raise ValueError(
            f"ir_to_qubo requires all-binary variables; non-binary: {sorted(nonbinary)}"
        )
    linear: dict[str, float] = dict(ir.objective.coeffs)
    quadratic: dict[tuple[str, str], float] = {}
    for term in ir.quadratic.objective_terms:
        if term.is_square:
            linear[term.var1] = linear.get(term.var1, 0.0) + term.coeff
        else:
            pair = term.key()
            quadratic[pair] = quadratic.get(pair, 0.0) + term.coeff
    return QUBO(linear=linear, quadratic=quadratic, offset=ir.objective.offset)


def max_cut_qubo(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float] | None = None,
) -> QUBO:
    """Build the Max-Cut QUBO (minimisation form) for an undirected weighted graph.

    Max-Cut maximises ``sum_e w_e * [x_u != x_v]`` where ``[x_u != x_v] =
    x_u + x_v - 2 x_u x_v``. As a QUBO (minimisation) this is::

        minimise  sum_e w_e * (2 x_u x_v - x_u - x_v)

    so the QUBO minimum equals ``-(max-cut weight)`` (the cut weight is recovered
    as ``-min`` of the QUBO energy). Nodes are named ``x0 .. x{n_nodes-1}``.

    Args:
        n_nodes: Number of nodes (>= 1).
        edges: Undirected edges as ``(u, v)`` index pairs (``0 <= u, v < n_nodes``,
            ``u != v``).
        weights: Per-edge weights (defaults to all ``1.0``).
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    edge_weights: Sequence[float] = [1.0] * len(edges) if weights is None else weights
    if len(edge_weights) != len(edges):
        raise ValueError("weights length must match edges length")
    linear: dict[str, float] = {}
    quadratic: dict[tuple[str, str], float] = {}
    for (u, v), w in zip(edges, edge_weights, strict=True):
        if u == v:
            raise ValueError(f"self-loop edge ({u}, {v}) is not allowed in Max-Cut")
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError(f"edge ({u}, {v}) out of range [0, {n_nodes})")
        name_u, name_v = f"x{u}", f"x{v}"
        linear[name_u] = linear.get(name_u, 0.0) - w
        linear[name_v] = linear.get(name_v, 0.0) - w
        pair = _canonical_pair(name_u, name_v)
        quadratic[pair] = quadratic.get(pair, 0.0) + 2.0 * w
    return QUBO(linear=linear, quadratic=quadratic, offset=0.0)


# ---------------------------------------------------------------------------
# Exact Fortet linearization (binary products) — pure IR -> IR, solver-free
# ---------------------------------------------------------------------------
def _iter_terms(ext: QuadraticExtension) -> Iterator[QuadraticTerm]:
    """Yield every quadratic term (objective + all constraints)."""
    yield from ext.objective_terms
    for terms in ext.constraint_terms.values():
        yield from terms


def _require_binary(term: QuadraticTerm, var_by_name: dict[str, Variable]) -> None:
    """Raise unless both variables of ``term`` are declared and BINARY."""
    for name in {term.var1, term.var2}:
        var = var_by_name.get(name)
        if var is None or var.vtype is not VarType.BINARY:
            vtype = "missing" if var is None else var.vtype.value
            raise UnsupportedModelError(
                "exact linearization requires BINARY variables in quadratic terms; "
                + f"variable {name!r} is {vtype}. Use a native quadratic solve "
                + "(SCIP) for non-binary quadratics."
            )


def linearize_quadratic(ir: MILP) -> MILP:
    """Return an EXACT linear-MILP reformulation of a binary-quadratic ``ir``.

    Each product ``c * x_i * x_j`` of two BINARY variables is replaced by
    ``c * y_ij`` for a fresh binary "edge" variable ``y_ij`` pinned to the product
    by the standard Fortet constraints ``y_ij <= x_i``, ``y_ij <= x_j``,
    ``y_ij >= x_i + x_j - 1``. At every integer ``x`` these force
    ``y_ij == x_i AND x_j`` exactly, so the MILP optimum equals the quadratic
    optimum. A square ``c * x_i^2`` folds into the linear coefficient of ``x_i``
    (``x_i^2 == x_i`` for binary ``x``).

    A purely linear ``ir`` (no extension or an empty one) returns a
    ``quadratic=None`` copy. Raises :class:`~opop.model.ir.UnsupportedModelError`
    if any quadratic term touches a non-binary variable (no exact MILP form
    exists). The input ``ir`` is never mutated.
    """
    ext = ir.quadratic
    if ext is None or ext.is_empty:
        return replace(ir, quadratic=None)

    var_by_name = {v.name: v for v in ir.variables}
    for term in _iter_terms(ext):
        _require_binary(term, var_by_name)

    new_obj = dict(ir.objective.coeffs)
    new_con_coeffs = {c.name: dict(c.coeffs) for c in ir.constraints}
    products: dict[tuple[str, str], str] = {}
    new_vars: list[Variable] = []
    fortet_cons: list[LinearConstraint] = []
    used_var_names = set(var_by_name)
    used_con_names = {c.name for c in ir.constraints}

    def product_var(var1: str, var2: str) -> str:
        pair = (var1, var2) if var1 <= var2 else (var2, var1)
        if pair in products:
            return products[pair]
        low, high = pair
        name = f"_prod_{low}__{high}"
        if name in used_var_names:
            raise UnsupportedModelError(
                f"linearization product variable name {name!r} collides with an "
                + "existing variable"
            )
        products[pair] = name
        used_var_names.add(name)
        new_vars.append(Variable(name, VarType.BINARY, 0.0, 1.0))
        fortet = (
            (f"_fortet_{name}_le1", {name: 1.0, low: -1.0}, ConstraintSense.LE, 0.0),
            (f"_fortet_{name}_le2", {name: 1.0, high: -1.0}, ConstraintSense.LE, 0.0),
            (f"_fortet_{name}_ge", {name: 1.0, low: -1.0, high: -1.0}, ConstraintSense.GE, -1.0),
        )
        for cname, coeffs, sense, rhs in fortet:
            if cname in used_con_names:
                raise UnsupportedModelError(
                    f"linearization constraint name {cname!r} collides with an "
                    + "existing constraint"
                )
            used_con_names.add(cname)
            fortet_cons.append(LinearConstraint(cname, coeffs, sense, rhs))
        return name

    for term in ext.objective_terms:
        if term.is_square:
            new_obj[term.var1] = new_obj.get(term.var1, 0.0) + term.coeff
        else:
            pname = product_var(term.var1, term.var2)
            new_obj[pname] = new_obj.get(pname, 0.0) + term.coeff

    for cname, terms in ext.constraint_terms.items():
        target = new_con_coeffs[cname]
        for term in terms:
            if term.is_square:
                target[term.var1] = target.get(term.var1, 0.0) + term.coeff
            else:
                pname = product_var(term.var1, term.var2)
                target[pname] = target.get(pname, 0.0) + term.coeff

    constraints = tuple(
        LinearConstraint(c.name, new_con_coeffs[c.name], c.sense, c.rhs)
        for c in ir.constraints
    ) + tuple(fortet_cons)
    objective = Objective(
        coeffs=new_obj, sense=ir.objective.sense, offset=ir.objective.offset
    )
    return MILP(
        name=ir.name,
        variables=ir.variables + tuple(new_vars),
        constraints=constraints,
        objective=objective,
        index_sets=ir.index_sets,
        metadata={**ir.metadata, "linearization": "fortet"},
        quadratic=None,
    )
