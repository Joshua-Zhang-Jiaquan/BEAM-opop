"""Symbolic MILP intermediate representation (IR) + MPS/LP I/O for OPOP.

This module is the foundation consumed by the Analyzer, the Verification gate,
and the SCIP solver kernel. It defines:

* An immutable, solver-agnostic MILP IR (:class:`MILP`) made of
  :class:`Variable`, :class:`LinearConstraint`, and :class:`Objective` records,
  plus named index sets and a free-form metadata dict.
* A variable--constraint bipartite incidence view (:func:`model_graph` /
  :class:`ModelGraph`) where ``n_nodes == n_vars + n_constraints`` and
  ``n_edges`` equals the constraint-matrix non-zero count (nnz).
* PySCIPOpt bridges :func:`from_pyscipopt` / :func:`to_pyscipopt` and
  MPS/LP read/write helpers, with a **lossless** ``IR -> MPS -> IR`` round-trip
  for the supported linear subset.
* A *pure* :func:`apply_delta` returning a NEW IR for class-A (equivalent
  reformulation), class-B (valid inequality), and class-C (semantic no-op /
  metadata) deltas; class-D deltas are rejected (sandbox-only).

Supported subset (Phase-1): linear constraints over BINARY / INTEGER /
CONTINUOUS variables with senses ``<=`` / ``>=`` / ``=``. Anything outside this
subset (quadratic / nonlinear handlers, range rows, implicit-integer vtypes)
raises :class:`UnsupportedModelError` rather than being silently dropped.

Quadratic support (Wave 5, task 30) is layered on *additively* via the optional
``MILP.quadratic`` field. The IR-level quadratic records :class:`QuadraticTerm`
and :class:`QuadraticExtension` are defined HERE (they are part of the IR, and
co-locating them keeps the model package's import graph acyclic). The richer
quadratic layer — :class:`~opop.model.quadratic.QUBO` / Ising energy models,
their exact conversions, and QUBO<->IR bridges — lives in
:mod:`opop.model.quadratic`, which imports (and re-exports) these two records.
When ``MILP.quadratic`` is ``None`` (the default) every linear-only behaviour
above is byte-for-byte unchanged. The PySCIPOpt / MPS-LP bridges in this module
remain *linear-only*: they neither read nor write the quadratic extension
(problem-class adapters in :mod:`opop.solver.qubo` / :mod:`opop.solver.miqp` own
quadratic compilation).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from opop.model.state import Delta, DeltaClass

__all__ = [
    "ConstraintSense",
    "LinearConstraint",
    "MILP",
    "ModelGraph",
    "Objective",
    "ObjSense",
    "QuadraticExtension",
    "QuadraticTerm",
    "UnsupportedModelError",
    "Variable",
    "VarType",
    "apply_delta",
    "from_pyscipopt",
    "make_add_constraint_delta",
    "make_metadata_delta",
    "make_rename_delta",
    "milps_equivalent",
    "model_graph",
    "read_lp",
    "read_mps",
    "read_problem",
    "to_pyscipopt",
    "write_lp",
    "write_mps",
    "write_problem",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class UnsupportedModelError(Exception):
    """Raised when a model contains constructs outside the linear MILP subset.

    Examples: quadratic / nonlinear constraint handlers, two-sided range rows,
    implicit-integer variables. The raise is deliberate (never a silent drop)
    so callers can route the model to a future nonlinear path or reject it.
    """


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class VarType(Enum):
    """Supported decision-variable domains."""

    BINARY = "BINARY"
    INTEGER = "INTEGER"
    CONTINUOUS = "CONTINUOUS"


class ConstraintSense(Enum):
    """Linear-constraint relation against the right-hand side."""

    LE = "<="
    GE = ">="
    EQ = "="


class ObjSense(Enum):
    """Objective optimisation direction."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# ---------------------------------------------------------------------------
# Core IR records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Variable:
    """A single decision variable.

    Attributes:
        name: Unique variable identifier (matches MPS column name).
        vtype: Domain (BINARY / INTEGER / CONTINUOUS).
        lower: Lower bound; ``-math.inf`` for unbounded below.
        upper: Upper bound; ``math.inf`` for unbounded above.
    """

    name: str
    vtype: VarType
    lower: float = 0.0
    upper: float = math.inf


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """A single linear constraint ``sum_j a_j x_j  (<=|>=|=)  rhs``.

    Attributes:
        name: Unique constraint identifier (matches MPS row name).
        coeffs: Mapping ``variable name -> coefficient`` (non-zeros only).
        sense: Relation against ``rhs``.
        rhs: Right-hand-side constant.
    """

    name: str
    coeffs: dict[str, float]
    sense: ConstraintSense
    rhs: float


@dataclass(frozen=True, slots=True)
class Objective:
    """The linear objective ``sum_j c_j x_j + offset``.

    Attributes:
        coeffs: Mapping ``variable name -> objective coefficient`` (non-zeros).
        sense: Minimise or maximise.
        offset: Constant objective offset (preserved through MPS round-trip).
    """

    coeffs: dict[str, float] = field(default_factory=dict)
    sense: ObjSense = ObjSense.MINIMIZE
    offset: float = 0.0


# ---------------------------------------------------------------------------
# Optional quadratic extension records (additive; see opop.model.quadratic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QuadraticTerm:
    """A single quadratic monomial ``coeff * var1 * var2``.

    A *square* term has ``var1 == var2`` (i.e. ``coeff * x^2``). For BINARY
    variables ``x^2 == x``, so a square is mathematically a linear term.

    Attributes:
        var1: First variable name.
        var2: Second variable name.
        coeff: Real coefficient of the ``var1 * var2`` product.
    """

    var1: str
    var2: str
    coeff: float

    @property
    def is_square(self) -> bool:
        """``True`` iff this term is ``coeff * var^2`` (``var1 == var2``)."""
        return self.var1 == self.var2

    def key(self) -> tuple[str, str]:
        """Return the order-independent ``(min, max)`` variable-name pair key."""
        return (self.var1, self.var2) if self.var1 <= self.var2 else (self.var2, self.var1)

    def variables(self) -> tuple[str, str]:
        """Return the ``(var1, var2)`` pair as declared."""
        return (self.var1, self.var2)


@dataclass(frozen=True, slots=True)
class QuadraticExtension:
    """The optional quadratic layer carried by a :class:`MILP`.

    Both members are *additive* on top of the MILP's existing linear objective
    and linear constraints; an empty extension is equivalent to no extension.

    Attributes:
        objective_terms: Quadratic terms added to the (linear) objective.
        constraint_terms: Mapping ``constraint name -> quadratic terms`` added to
            the (linear) LHS of that already-declared constraint. The constraint
            keeps its declared sense and rhs; only its LHS gains quadratic terms.
    """

    objective_terms: tuple[QuadraticTerm, ...] = ()
    constraint_terms: dict[str, tuple[QuadraticTerm, ...]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """``True`` iff there are no quadratic objective or constraint terms."""
        return not self.objective_terms and not any(self.constraint_terms.values())

    def has_objective_terms(self) -> bool:
        """``True`` iff the objective carries at least one quadratic term."""
        return bool(self.objective_terms)

    def has_constraint_terms(self) -> bool:
        """``True`` iff at least one constraint carries a quadratic term."""
        return any(self.constraint_terms.values())

    def referenced_variables(self) -> frozenset[str]:
        """Return every variable name appearing in any quadratic term."""
        names: set[str] = set()
        for term in self.objective_terms:
            names.add(term.var1)
            names.add(term.var2)
        for terms in self.constraint_terms.values():
            for term in terms:
                names.add(term.var1)
                names.add(term.var2)
        return frozenset(names)

    def constraint_names(self) -> frozenset[str]:
        """Return the constraint names that carry quadratic terms."""
        return frozenset(self.constraint_terms)


@dataclass(frozen=True, slots=True)
class MILP:
    """An immutable mixed-integer linear program.

    Construction validates referential integrity: variable names are unique,
    constraint names are unique, and every coefficient (constraint + objective)
    references a declared variable. Violations raise :class:`ValueError`.

    Attributes:
        name: Problem name (informational; not part of the math model).
        variables: Ordered tuple of :class:`Variable` (preserves column order).
        constraints: Ordered tuple of :class:`LinearConstraint` (row order).
        objective: The :class:`Objective`.
        index_sets: Named index sets (e.g. ``{"I": ("0", "1", "2")}``) — IR-side
            annotations for the analyzer; not serialised to MPS.
        metadata: Free-form metadata dict; not serialised to MPS.
        quadratic: Optional additive quadratic layer
            (:class:`opop.model.quadratic.QuadraticExtension`). ``None`` (default)
            means a pure linear MILP — all linear behaviour is unchanged. When
            present, its terms must reference declared variables, and any quadratic
            constraint name must match a declared linear constraint (the quadratic
            terms augment that constraint's LHS). Not serialised to MPS.
    """

    name: str = ""
    variables: tuple[Variable, ...] = ()
    constraints: tuple[LinearConstraint, ...] = ()
    objective: Objective = field(default_factory=Objective)
    index_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    quadratic: QuadraticExtension | None = None

    def __post_init__(self) -> None:
        var_names = [v.name for v in self.variables]
        var_set = set(var_names)
        if len(var_names) != len(var_set):
            dupes = sorted({n for n in var_names if var_names.count(n) > 1})
            raise ValueError(f"duplicate variable names: {dupes}")

        con_names = [c.name for c in self.constraints]
        con_set = set(con_names)
        if len(con_names) != len(con_set):
            dupes = sorted({n for n in con_names if con_names.count(n) > 1})
            raise ValueError(f"duplicate constraint names: {dupes}")

        for con in self.constraints:
            unknown = set(con.coeffs) - var_set
            if unknown:
                raise ValueError(
                    f"constraint {con.name!r} references unknown variables: {sorted(unknown)}"
                )
        unknown_obj = set(self.objective.coeffs) - var_set
        if unknown_obj:
            raise ValueError(
                f"objective references unknown variables: {sorted(unknown_obj)}"
            )

        if self.quadratic is not None:
            unknown_quad = self.quadratic.referenced_variables() - var_set
            if unknown_quad:
                raise ValueError(
                    "quadratic extension references unknown variables: "
                    + f"{sorted(unknown_quad)}"
                )
            unknown_quad_con = self.quadratic.constraint_names() - con_set
            if unknown_quad_con:
                raise ValueError(
                    "quadratic extension references unknown constraints: "
                    + f"{sorted(unknown_quad_con)}"
                )

    # -- convenience views --------------------------------------------------
    @property
    def n_vars(self) -> int:
        """Number of decision variables."""
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        """Number of linear constraints."""
        return len(self.constraints)

    @property
    def nnz(self) -> int:
        """Constraint-matrix non-zero count (sum of non-zero coefficients)."""
        return sum(1 for c in self.constraints for v in c.coeffs.values() if v != 0.0)

    def var_names(self) -> tuple[str, ...]:
        """Return variable names in declaration (column) order."""
        return tuple(v.name for v in self.variables)

    def model_graph(self) -> ModelGraph:
        """Return the variable--constraint bipartite incidence view."""
        return model_graph(self)


# ---------------------------------------------------------------------------
# Bipartite model graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelGraph:
    """Variable--constraint bipartite incidence view of a :class:`MILP`.

    ``n_nodes == len(var_nodes) + len(con_nodes) == n_vars + n_constraints``.
    ``n_edges`` equals the constraint-matrix non-zero count (nnz): one edge per
    non-zero ``(variable, constraint)`` incidence. The objective is *not* part
    of the graph.

    Attributes:
        var_nodes: Variable names (one bipartite partition).
        con_nodes: Constraint names (the other partition).
        edges: ``(variable_name, constraint_name)`` incidences (one per nnz).
    """

    var_nodes: tuple[str, ...]
    con_nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    @property
    def n_nodes(self) -> int:
        """Total node count: variables + constraints."""
        return len(self.var_nodes) + len(self.con_nodes)

    @property
    def n_edges(self) -> int:
        """Total edge count: equals the constraint-matrix nnz."""
        return len(self.edges)

    def to_networkx(self) -> Any:
        """Return a ``networkx`` bipartite graph (lazy import).

        Nodes are namespaced tuples ``("var", name)`` / ``("con", name)`` so a
        variable and a constraint that share a name never collide. Each node
        carries a ``bipartite`` attribute (0 = variable, 1 = constraint).
        """
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from((("var", n) for n in self.var_nodes), bipartite=0)
        graph.add_nodes_from((("con", n) for n in self.con_nodes), bipartite=1)
        graph.add_edges_from((("var", v), ("con", c)) for v, c in self.edges)
        return graph


def model_graph(ir: MILP) -> ModelGraph:
    """Build the variable--constraint bipartite incidence view of ``ir``.

    Edges count only non-zero coefficients, so ``ModelGraph.n_edges`` equals the
    constraint-matrix nnz.
    """
    edges = tuple(
        (vname, con.name)
        for con in ir.constraints
        for vname, coeff in con.coeffs.items()
        if coeff != 0.0
    )
    return ModelGraph(
        var_nodes=tuple(v.name for v in ir.variables),
        con_nodes=tuple(c.name for c in ir.constraints),
        edges=edges,
    )


# ---------------------------------------------------------------------------
# PySCIPOpt bridges
# ---------------------------------------------------------------------------
_SUPPORTED_HANDLERS: frozenset[str] = frozenset({"linear"})
_SCIP_TO_VTYPE: dict[str, VarType] = {
    "BINARY": VarType.BINARY,
    "INTEGER": VarType.INTEGER,
    "CONTINUOUS": VarType.CONTINUOUS,
}
_VTYPE_TO_SCIP: dict[VarType, str] = {
    VarType.BINARY: "B",
    VarType.INTEGER: "I",
    VarType.CONTINUOUS: "C",
}
# Tolerance for classifying an MPS row with lhs == rhs as an equality.
_EQ_TOL: float = 1e-9


def _from_scip_bound(model: Any, value: float) -> float:
    """Map a SCIP bound (+-infinity sentinel) to a Python float / math.inf."""
    if model.isInfinity(value):
        return math.inf
    if model.isInfinity(-value):
        return -math.inf
    return float(value)


def _to_scip_bound(model: Any, value: float) -> float:
    """Map a Python bound (math.inf) back to a SCIP infinity sentinel."""
    if value == math.inf:
        return model.infinity()
    if value == -math.inf:
        return -model.infinity()
    return value


def _derive_sense(
    model: Any, lhs: float, rhs: float, con_name: str
) -> tuple[ConstraintSense, float]:
    """Derive ``(sense, rhs)`` from a SCIP linear constraint's lhs/rhs pair."""
    lhs_neg_inf = model.isInfinity(-lhs)
    rhs_pos_inf = model.isInfinity(rhs)
    if lhs_neg_inf and rhs_pos_inf:
        raise UnsupportedModelError(
            f"constraint {con_name!r} is free (no finite bound); not supported"
        )
    if lhs_neg_inf:
        return ConstraintSense.LE, float(rhs)
    if rhs_pos_inf:
        return ConstraintSense.GE, float(lhs)
    if abs(lhs - rhs) <= _EQ_TOL:
        return ConstraintSense.EQ, float(rhs)
    raise UnsupportedModelError(
        f"constraint {con_name!r} has a two-sided range; only <=, >=, = senses are supported"
    )


def from_pyscipopt(model: Any) -> MILP:
    """Build a :class:`MILP` IR from a PySCIPOpt ``Model``.

    Walks ``model.getVars()`` and ``model.getConss()``. Raises
    :class:`UnsupportedModelError` for non-linear constraint handlers (e.g.
    quadratic / nonlinear), implicit-integer variables, free rows, or two-sided
    range rows.
    """
    variables: list[Variable] = []
    for var in model.getVars():
        scip_type = var.vtype()
        vtype = _SCIP_TO_VTYPE.get(scip_type)
        if vtype is None:
            raise UnsupportedModelError(
                f"variable {var.name!r} has unsupported vtype {scip_type!r} (only B/I/C supported)"
            )
        variables.append(
            Variable(
                name=var.name,
                vtype=vtype,
                lower=_from_scip_bound(model, var.getLbOriginal()),
                upper=_from_scip_bound(model, var.getUbOriginal()),
            )
        )

    constraints: list[LinearConstraint] = []
    for con in model.getConss():
        handler = con.getConshdlrName()
        if handler not in _SUPPORTED_HANDLERS:
            raise UnsupportedModelError(
                f"constraint {con.name!r} uses unsupported handler {handler!r} (linear only)"
            )
        coeffs = {str(name): float(coeff) for name, coeff in model.getValsLinear(con).items()}
        sense, rhs = _derive_sense(model, model.getLhs(con), model.getRhs(con), con.name)
        constraints.append(
            LinearConstraint(name=con.name, coeffs=coeffs, sense=sense, rhs=rhs)
        )

    obj_coeffs = {var.name: float(var.getObj()) for var in model.getVars() if var.getObj() != 0.0}
    sense_str = str(model.getObjectiveSense()).lower()
    obj_sense = ObjSense.MAXIMIZE if sense_str.startswith("max") else ObjSense.MINIMIZE
    objective = Objective(
        coeffs=obj_coeffs,
        sense=obj_sense,
        offset=float(model.getObjoffset()),
    )

    return MILP(
        name=str(model.getProbName()),
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective=objective,
        metadata={"source": "pyscipopt"},
    )


def to_pyscipopt(ir: MILP) -> Any:
    """Build a fresh PySCIPOpt ``Model`` from a :class:`MILP` IR.

    The returned model is *not* solved and output is *not* suppressed; the
    caller owns its lifecycle (e.g. the solver kernel sets limits/seed).
    """
    from pyscipopt import Model, quicksum

    model = Model(ir.name or "opop_model")
    scip_vars: dict[str, Any] = {}
    for var in ir.variables:
        scip_vars[var.name] = model.addVar(
            name=var.name,
            vtype=_VTYPE_TO_SCIP[var.vtype],
            lb=_to_scip_bound(model, var.lower),
            ub=_to_scip_bound(model, var.upper),
        )

    for con in ir.constraints:
        expr = quicksum(coeff * scip_vars[name] for name, coeff in con.coeffs.items())
        if con.sense is ConstraintSense.LE:
            model.addCons(expr <= con.rhs, name=con.name)
        elif con.sense is ConstraintSense.GE:
            model.addCons(expr >= con.rhs, name=con.name)
        else:
            model.addCons(expr == con.rhs, name=con.name)

    obj_terms = [coeff * scip_vars[name] for name, coeff in ir.objective.coeffs.items()]
    obj_expr = quicksum(obj_terms) if obj_terms else 0
    model.setObjective(obj_expr, sense=ir.objective.sense.value)
    if ir.objective.offset != 0.0:
        model.addObjoffset(ir.objective.offset)

    return model


# ---------------------------------------------------------------------------
# MPS / LP file I/O
# ---------------------------------------------------------------------------
def read_problem(path: str) -> MILP:
    """Read an MPS or LP file (format inferred by extension) into a :class:`MILP`."""
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.readProblem(str(path))
    return from_pyscipopt(model)


def write_problem(ir: MILP, path: str) -> None:
    """Write a :class:`MILP` to an MPS or LP file (format inferred by extension)."""
    model = to_pyscipopt(ir)
    model.hideOutput()
    model.writeProblem(str(path))


def read_mps(path: str) -> MILP:
    """Read an MPS file into a :class:`MILP`."""
    return read_problem(path)


def write_mps(ir: MILP, path: str) -> None:
    """Write a :class:`MILP` to an MPS file (use a ``.mps`` extension)."""
    write_problem(ir, path)


def read_lp(path: str) -> MILP:
    """Read an LP file into a :class:`MILP`."""
    return read_problem(path)


def write_lp(ir: MILP, path: str) -> None:
    """Write a :class:`MILP` to an LP file (use a ``.lp`` extension)."""
    write_problem(ir, path)


# ---------------------------------------------------------------------------
# Equivalence
# ---------------------------------------------------------------------------
def _close(x: float, y: float, tol: float) -> bool:
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= tol


def _coeff_diffs(
    label: str, a: dict[str, float], b: dict[str, float], tol: float
) -> list[str]:
    diffs: list[str] = []
    for key in set(a) | set(b):
        if not _close(a.get(key, 0.0), b.get(key, 0.0), tol):
            diffs.append(f"{label}: coeff {key!r} differs ({a.get(key, 0.0)} vs {b.get(key, 0.0)})")
    return diffs


def _quad_term_map(terms: tuple[QuadraticTerm, ...]) -> dict[str, float]:
    """Collapse quadratic terms into an order-independent ``"i*j" -> coeff`` map."""
    out: dict[str, float] = {}
    for term in terms:
        i, j = term.key()
        out[f"{i}*{j}"] = out.get(f"{i}*{j}", 0.0) + term.coeff
    return out


def _quadratic_diffs(
    a: QuadraticExtension | None, b: QuadraticExtension | None, tol: float
) -> list[str]:
    """Diff two optional quadratic extensions (``None`` == empty == no diffs)."""
    a_empty = a is None or a.is_empty
    b_empty = b is None or b.is_empty
    if a_empty and b_empty:
        return []
    diffs: list[str] = []
    a_obj = _quad_term_map(a.objective_terms) if a is not None else {}
    b_obj = _quad_term_map(b.objective_terms) if b is not None else {}
    diffs.extend(_coeff_diffs("quadratic objective", a_obj, b_obj, tol))
    a_cons = a.constraint_terms if a is not None else {}
    b_cons = b.constraint_terms if b is not None else {}
    for cname in sorted(set(a_cons) | set(b_cons)):
        diffs.extend(
            _coeff_diffs(
                f"quadratic constraint {cname!r}",
                _quad_term_map(a_cons.get(cname, ())),
                _quad_term_map(b_cons.get(cname, ())),
                tol,
            )
        )
    return diffs


def milp_diffs(a: MILP, b: MILP, tol: float = 1e-9) -> list[str]:
    """Return human-readable differences between two MILPs' *math models*.

    Compares objective sense/offset/coefficients, variable domains/bounds,
    constraint senses/rhs/coefficients, and the optional quadratic extension
    (objective + per-constraint quadratic terms), all matched by name and
    order-independent. A ``None`` quadratic extension is treated as empty, so two
    purely linear models compare exactly as before (no behavioural change).
    Ignores ``name``, ``index_sets``, and ``metadata`` (not part of the math
    model and not serialised to MPS). An empty list means equivalent.
    """
    diffs: list[str] = []

    if a.objective.sense is not b.objective.sense:
        diffs.append(f"objective sense {a.objective.sense} vs {b.objective.sense}")
    if not _close(a.objective.offset, b.objective.offset, tol):
        diffs.append(f"objective offset {a.objective.offset} vs {b.objective.offset}")
    diffs.extend(_coeff_diffs("objective", a.objective.coeffs, b.objective.coeffs, tol))

    va = {v.name: v for v in a.variables}
    vb = {v.name: v for v in b.variables}
    if set(va) != set(vb):
        diffs.append(f"variable set differs: {sorted(set(va) ^ set(vb))}")
    else:
        for name, var_a in va.items():
            var_b = vb[name]
            if var_a.vtype is not var_b.vtype:
                diffs.append(f"variable {name!r} vtype {var_a.vtype} vs {var_b.vtype}")
            if not _close(var_a.lower, var_b.lower, tol):
                diffs.append(f"variable {name!r} lower {var_a.lower} vs {var_b.lower}")
            if not _close(var_a.upper, var_b.upper, tol):
                diffs.append(f"variable {name!r} upper {var_a.upper} vs {var_b.upper}")

    ca = {c.name: c for c in a.constraints}
    cb = {c.name: c for c in b.constraints}
    if set(ca) != set(cb):
        diffs.append(f"constraint set differs: {sorted(set(ca) ^ set(cb))}")
    else:
        for name, con_a in ca.items():
            con_b = cb[name]
            if con_a.sense is not con_b.sense:
                diffs.append(f"constraint {name!r} sense {con_a.sense} vs {con_b.sense}")
            if not _close(con_a.rhs, con_b.rhs, tol):
                diffs.append(f"constraint {name!r} rhs {con_a.rhs} vs {con_b.rhs}")
            diffs.extend(_coeff_diffs(f"constraint {name!r}", con_a.coeffs, con_b.coeffs, tol))

    diffs.extend(_quadratic_diffs(a.quadratic, b.quadratic, tol))

    return diffs


def milps_equivalent(a: MILP, b: MILP, tol: float = 1e-9) -> bool:
    """Return ``True`` iff two MILPs share the same math model within ``tol``."""
    return not milp_diffs(a, b, tol)


# ---------------------------------------------------------------------------
# apply_delta — pure class-A / B / C transformations
# ---------------------------------------------------------------------------
OP_RENAME_VAR = "rename_var"
OP_ADD_CONSTRAINT = "add_constraint"
OP_UPDATE_METADATA = "update_metadata"

# Each op is only valid under its declared verification class.
_OP_CLASS: dict[str, DeltaClass] = {
    OP_RENAME_VAR: DeltaClass.A,
    OP_ADD_CONSTRAINT: DeltaClass.B,
    OP_UPDATE_METADATA: DeltaClass.C,
}


def make_rename_delta(old: str, new: str, target: str | None = None) -> Delta:
    """Build a class-A :class:`Delta` that renames variable ``old`` to ``new``."""
    payload = {"op": OP_RENAME_VAR, "old": old, "new": new}
    return Delta(
        target=target or f"rename variable {old} -> {new}",
        after_fragment=json.dumps(payload),
        declared_class=DeltaClass.A,
    )


def make_add_constraint_delta(
    name: str,
    coeffs: dict[str, float],
    sense: ConstraintSense | str,
    rhs: float,
    target: str | None = None,
) -> Delta:
    """Build a class-B :class:`Delta` that appends a (valid-inequality) constraint."""
    sense_val = sense.value if isinstance(sense, ConstraintSense) else str(sense)
    payload = {
        "op": OP_ADD_CONSTRAINT,
        "name": name,
        "coeffs": {str(k): float(v) for k, v in coeffs.items()},
        "sense": sense_val,
        "rhs": float(rhs),
    }
    return Delta(
        target=target or f"add constraint {name}",
        after_fragment=json.dumps(payload),
        declared_class=DeltaClass.B,
    )


def make_metadata_delta(updates: dict[str, Any], target: str | None = None) -> Delta:
    """Build a class-C :class:`Delta` that merges ``updates`` into IR metadata."""
    payload = {"op": OP_UPDATE_METADATA, "updates": dict(updates)}
    return Delta(
        target=target or "update metadata",
        after_fragment=json.dumps(payload),
        declared_class=DeltaClass.C,
    )


def _apply_rename(ir: MILP, payload: dict[str, Any]) -> MILP:
    old = payload["old"]
    new = payload["new"]
    names = {v.name for v in ir.variables}
    if old not in names:
        raise ValueError(f"rename source {old!r} is not a variable")
    if new in names:
        raise ValueError(f"rename target {new!r} already exists")

    def _rekey(coeffs: dict[str, float]) -> dict[str, float]:
        return {(new if k == old else k): v for k, v in coeffs.items()}

    new_vars = tuple(replace(v, name=new) if v.name == old else v for v in ir.variables)
    new_cons = tuple(replace(c, coeffs=_rekey(c.coeffs)) for c in ir.constraints)
    new_obj = replace(ir.objective, coeffs=_rekey(ir.objective.coeffs))
    return replace(ir, variables=new_vars, constraints=new_cons, objective=new_obj)


def _apply_add_constraint(ir: MILP, payload: dict[str, Any]) -> MILP:
    name = payload["name"]
    if any(c.name == name for c in ir.constraints):
        raise ValueError(f"constraint {name!r} already exists")
    coeffs = {str(k): float(v) for k, v in payload["coeffs"].items()}
    var_names = {v.name for v in ir.variables}
    unknown = set(coeffs) - var_names
    if unknown:
        raise ValueError(f"constraint {name!r} references unknown variables: {sorted(unknown)}")
    con = LinearConstraint(
        name=name,
        coeffs=coeffs,
        sense=ConstraintSense(payload["sense"]),
        rhs=float(payload["rhs"]),
    )
    return replace(ir, constraints=ir.constraints + (con,))


def _apply_update_metadata(ir: MILP, payload: dict[str, Any]) -> MILP:
    updates = payload["updates"]
    if not isinstance(updates, dict):
        raise ValueError("metadata delta 'updates' must be an object")
    return replace(ir, metadata={**ir.metadata, **updates})


def apply_delta(ir: MILP, delta: Delta) -> MILP:
    """Apply a class-A / B / C :class:`Delta` and return a NEW :class:`MILP`.

    Pure: ``ir`` is never mutated. The concrete change is encoded as a JSON
    payload in ``delta.after_fragment`` carrying an ``"op"`` key:

    * ``rename_var`` (class A): ``{"old": ..., "new": ...}`` — equivalent
      reformulation; relabels a variable across vars, constraints, and objective.
    * ``add_constraint`` (class B): ``{"name", "coeffs", "sense", "rhs"}`` —
      appends a (valid-inequality) linear constraint.
    * ``update_metadata`` (class C): ``{"updates": {...}}`` — semantic no-op;
      merges into the metadata dict only.

    Class-D deltas are rejected (sandbox-only, never the main path). A mismatch
    between the op and the delta's ``declared_class``, an unknown op, or a
    missing/invalid payload raises :class:`ValueError`.
    """
    if delta.declared_class is DeltaClass.D:
        raise ValueError("class-D deltas are sandbox-only and cannot be applied to the main IR")
    if not delta.after_fragment:
        raise ValueError("delta.after_fragment must carry a JSON payload describing the change")
    try:
        payload_raw: Any = json.loads(delta.after_fragment)
    except json.JSONDecodeError as exc:
        raise ValueError(f"delta.after_fragment is not valid JSON: {exc}") from exc
    if not isinstance(payload_raw, dict):
        raise ValueError("delta payload must be a JSON object")
    payload: dict[str, Any] = payload_raw

    op = payload.get("op")
    if not isinstance(op, str):
        raise ValueError(f"delta payload missing string 'op'; got {op!r}")
    expected_class = _OP_CLASS.get(op)
    if expected_class is None:
        raise ValueError(f"unknown delta op {op!r}; supported: {sorted(_OP_CLASS)}")
    if delta.declared_class is not expected_class:
        want = expected_class.value
        got = delta.declared_class.value
        raise ValueError(f"op {op!r} requires declared_class {want}, got {got}")

    if op == OP_RENAME_VAR:
        return _apply_rename(ir, payload)
    if op == OP_ADD_CONSTRAINT:
        return _apply_add_constraint(ir, payload)
    return _apply_update_metadata(ir, payload)
