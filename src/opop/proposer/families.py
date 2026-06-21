"""Formulation-family constructors → certified-equivalent reformulation deltas.

The Phase-1 proposer only tuned the *search* (params + analyzer cuts). This
module adds *structure-first* proposals: it knows a small library of
formulation families and emits reformulation :class:`~opop.model.state.Delta`
objects that are each **class A or B** and therefore certifiable by the
verification gate (:func:`opop.verify.gate.verify_delta`). Class-D never enters
the stream.

Families (per the plan, task 27)
--------------------------------
* **routing**   — TSP: ``mtz`` (Miller–Tucker–Zemlin), ``scf`` (single-commodity
  flow), ``mcf`` (multi-commodity flow), ``set_partition`` (set-partition +
  pricing). Fully implemented + gate-certified below.
* **scheduling**— ``time_indexed``, ``disjunctive``, ``arc_flow``.
* **lot_sizing**— ``big_bucket``, ``small_bucket``.

The flagship is the **MTZ ↔ multi-commodity-flow** equivalence on a tiny TSP.

Why the reformulation is encoded as class-B cutset inequalities (the gate seam)
------------------------------------------------------------------------------
The verification gate certifies equivalence by exactly two routes:

* **class A** — a structural relabel: the after-model must equal the before-model
  up to a *single* variable rename (:func:`opop.verify.gate._infer_var_mapping`
  handles only identity / one rename), then a solver confirms the same optimum.
* **class B** — the after-model only *appends* constraints, each proven not to
  remove any feasible integer point of the before-model.

A literal MTZ→MCF swap changes the whole variable set (MTZ's ``u_i`` potentials
vs. MCF's ``f^k_ij`` flows), so it is NOT class-A relabelable and NOT a pure
constraint addition — the gate (correctly, and by design we MUST NOT modify it)
rejects such a swap. The *gate-faithful* certificate of the same equivalence is:

1. The multi-commodity-flow formulation's projection onto the arc variables is
   exactly the **cutset / subtour-elimination polytope**: every directed cut
   ``Σ_{i∈S, j∉S} x_ij ≥ 1`` is implied by max-flow–min-cut between a
   commodity's source and sink. So "import the MCF strength into the MTZ model"
   means *add those cutset inequalities* — each a **class-B PASS** (every
   Hamiltonian tour leaves every proper subset ≥ 1 time, so no feasible tour is
   removed). :func:`mtz_to_flow_reformulation` builds exactly these deltas.
2. The standalone MTZ and standalone MCF models solve to the **same optimum**
   (verified in the tests by solving both). Together these establish "MTZ ↔ MCF,
   certified equivalent, same optimum on both formulations".

A complementary **class-A** family delta (a canonical variable-encoding relabel,
:func:`encoding_relabel_delta`) demonstrates the class-A route on the same model.

All emitted deltas carry an explicit ``"kind"`` tag (see
:mod:`opop.proposer.stages`) so the staged-space filter can gate them: every
reformulation here is kind ``formulation`` (unlocked at stage S3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
    make_add_constraint_delta,
    make_rename_delta,
)
from opop.model.state import Delta
from opop.proposer.stages import KIND_FORMULATION

__all__ = [
    "FAMILIES",
    "FormulationFamily",
    "Reformulation",
    "build_tsp_mcf",
    "build_tsp_mtz",
    "build_tsp_scf",
    "cutset_inequalities",
    "encoding_relabel_delta",
    "family_deltas",
    "mtz_to_flow_reformulation",
]


# ---------------------------------------------------------------------------
# Family registry (the named design space task 27 unlocks at stage S3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FormulationFamily:
    """One named formulation family in a problem domain.

    Attributes:
        name: Stable family id (e.g. ``"mcf"``).
        domain: Coarse problem class (``"routing"`` / ``"scheduling"`` /
            ``"lot_sizing"``).
        description: Short human-readable summary of the formulation.
    """

    name: str
    domain: str
    description: str


#: The formulation-family catalogue (the structural design space). Routing is
#: fully gate-certified below; the scheduling / lot-sizing entries are declared
#: here as the named space the proposer can grow into (consumed by the analyzer
#: signals from task 26 / the controller's ``m`` axis).
FAMILIES: tuple[FormulationFamily, ...] = (
    FormulationFamily("mtz", "routing", "TSP Miller–Tucker–Zemlin compact subtour elimination"),
    FormulationFamily("scf", "routing", "TSP single-commodity flow subtour elimination"),
    FormulationFamily("mcf", "routing", "TSP multi-commodity flow (cutset-tight) subtour elimination"),
    FormulationFamily("set_partition", "routing", "set-partition + column generation (pricing)"),
    FormulationFamily("time_indexed", "scheduling", "time-indexed binary start variables"),
    FormulationFamily("disjunctive", "scheduling", "disjunctive (big-M sequencing) variables"),
    FormulationFamily("arc_flow", "scheduling", "arc-flow / time-expanded network"),
    FormulationFamily("big_bucket", "lot_sizing", "big-bucket (capacitated lot-sizing) periods"),
    FormulationFamily("small_bucket", "lot_sizing", "small-bucket (single-item-per-period) periods"),
)


# ---------------------------------------------------------------------------
# Tiny TSP IR builders (directed, depot = node 0)
# ---------------------------------------------------------------------------
def _arc_name(i: int, j: int) -> str:
    return f"x_{i}_{j}"


def _arcs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def _default_cost(n: int) -> dict[tuple[int, int], float]:
    """A deterministic, asymmetric-ish cost so the optimum is non-trivial."""
    return {(i, j): float(1 + (i * n + j) % 7) for (i, j) in _arcs(n)}


def _arc_variables(n: int) -> list[Variable]:
    return [Variable(_arc_name(i, j), VarType.BINARY, 0.0, 1.0) for (i, j) in _arcs(n)]


def _degree_constraints(n: int) -> list[LinearConstraint]:
    """In-degree == 1 and out-degree == 1 at every node."""
    cons: list[LinearConstraint] = []
    for j in range(n):
        cons.append(
            LinearConstraint(
                f"indeg_{j}", {_arc_name(i, j): 1.0 for i in range(n) if i != j}, ConstraintSense.EQ, 1.0
            )
        )
    for i in range(n):
        cons.append(
            LinearConstraint(
                f"outdeg_{i}", {_arc_name(i, j): 1.0 for j in range(n) if j != i}, ConstraintSense.EQ, 1.0
            )
        )
    return cons


def _objective(n: int, cost: dict[tuple[int, int], float]) -> Objective:
    return Objective(
        coeffs={_arc_name(i, j): cost[(i, j)] for (i, j) in _arcs(n)}, sense=ObjSense.MINIMIZE
    )


def _metadata(n: int, family: str) -> dict[str, object]:
    return {"domain": "routing", "family": family, "n_nodes": n}


def build_tsp_mtz(n: int, cost: dict[tuple[int, int], float] | None = None) -> MILP:
    """Build the directed-TSP Miller–Tucker–Zemlin IR on ``n`` nodes.

    Variables: binary arcs ``x_ij`` plus continuous potentials ``u_i`` for
    ``i ≥ 1`` (``u_i ∈ [1, n-1]``). Constraints: degree rows + MTZ subtour
    elimination ``u_i - u_j + (n-1) x_ij ≤ n-2`` for ``i, j ≥ 1, i ≠ j``. Its
    feasible integer set is exactly the Hamiltonian tours.
    """
    if n < 3:
        raise ValueError(f"TSP needs n >= 3 nodes, got {n}")
    cost = cost or _default_cost(n)
    variables = _arc_variables(n)
    variables.extend(Variable(f"u_{i}", VarType.CONTINUOUS, 1.0, float(n - 1)) for i in range(1, n))
    cons = _degree_constraints(n)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                cons.append(
                    LinearConstraint(
                        f"mtz_{i}_{j}",
                        {f"u_{i}": 1.0, f"u_{j}": -1.0, _arc_name(i, j): float(n - 1)},
                        ConstraintSense.LE,
                        float(n - 2),
                    )
                )
    return MILP(
        name="tsp_mtz",
        variables=tuple(variables),
        constraints=tuple(cons),
        objective=_objective(n, cost),
        metadata=_metadata(n, "mtz"),
    )


def build_tsp_scf(n: int, cost: dict[tuple[int, int], float] | None = None) -> MILP:
    """Build the directed-TSP single-commodity flow IR on ``n`` nodes.

    One commodity: depot 0 ships ``n-1`` units; each other node consumes 1.
    Variables: binary arcs ``x_ij`` + continuous flows ``f_ij ∈ [0, n-1]``.
    Constraints: degree rows, flow conservation, and coupling
    ``f_ij ≤ (n-1) x_ij``.
    """
    if n < 3:
        raise ValueError(f"TSP needs n >= 3 nodes, got {n}")
    cost = cost or _default_cost(n)
    variables = _arc_variables(n)
    variables.extend(
        Variable(f"f_{i}_{j}", VarType.CONTINUOUS, 0.0, float(n - 1)) for (i, j) in _arcs(n)
    )
    cons = _degree_constraints(n)
    for m in range(n):
        coeffs: dict[str, float] = {}
        for i in range(n):
            if i != m:
                coeffs[f"f_{i}_{m}"] = coeffs.get(f"f_{i}_{m}", 0.0) + 1.0
        for j in range(n):
            if j != m:
                coeffs[f"f_{m}_{j}"] = coeffs.get(f"f_{m}_{j}", 0.0) - 1.0
        rhs = -float(n - 1) if m == 0 else 1.0
        cons.append(LinearConstraint(f"flow_{m}", coeffs, ConstraintSense.EQ, rhs))
    for i, j in _arcs(n):
        cons.append(
            LinearConstraint(
                f"cap_{i}_{j}", {f"f_{i}_{j}": 1.0, _arc_name(i, j): -float(n - 1)}, ConstraintSense.LE, 0.0
            )
        )
    return MILP(
        name="tsp_scf",
        variables=tuple(variables),
        constraints=tuple(cons),
        objective=_objective(n, cost),
        metadata=_metadata(n, "scf"),
    )


def build_tsp_mcf(n: int, cost: dict[tuple[int, int], float] | None = None) -> MILP:
    """Build the directed-TSP multi-commodity flow IR on ``n`` nodes.

    One commodity ``k`` per non-depot node: depot 0 ships 1 unit of commodity
    ``k`` to sink ``k``. Variables: binary arcs ``x_ij`` + continuous flows
    ``f^k_ij ∈ [0, 1]``. Constraints: degree rows, per-commodity flow
    conservation, and coupling ``f^k_ij ≤ x_ij``. Its projection onto ``x`` is
    the cutset (subtour-elimination) polytope, so its integer optimum equals the
    MTZ optimum.
    """
    if n < 3:
        raise ValueError(f"TSP needs n >= 3 nodes, got {n}")
    cost = cost or _default_cost(n)
    variables = _arc_variables(n)
    for k in range(1, n):
        variables.extend(
            Variable(f"f_{k}_{i}_{j}", VarType.CONTINUOUS, 0.0, 1.0) for (i, j) in _arcs(n)
        )
    cons = _degree_constraints(n)
    for k in range(1, n):
        for m in range(n):
            coeffs: dict[str, float] = {}
            for i in range(n):
                if i != m:
                    coeffs[f"f_{k}_{i}_{m}"] = coeffs.get(f"f_{k}_{i}_{m}", 0.0) + 1.0
            for j in range(n):
                if j != m:
                    coeffs[f"f_{k}_{m}_{j}"] = coeffs.get(f"f_{k}_{m}_{j}", 0.0) - 1.0
            if m == 0:
                rhs = -1.0
            elif m == k:
                rhs = 1.0
            else:
                rhs = 0.0
            cons.append(LinearConstraint(f"flow_{k}_{m}", coeffs, ConstraintSense.EQ, rhs))
    for k in range(1, n):
        for i, j in _arcs(n):
            cons.append(
                LinearConstraint(
                    f"cap_{k}_{i}_{j}",
                    {f"f_{k}_{i}_{j}": 1.0, _arc_name(i, j): -1.0},
                    ConstraintSense.LE,
                    0.0,
                )
            )
    return MILP(
        name="tsp_mcf",
        variables=tuple(variables),
        constraints=tuple(cons),
        objective=_objective(n, cost),
        metadata=_metadata(n, "mcf"),
    )


# ---------------------------------------------------------------------------
# Cutset (MCF-projection) valid inequalities + the reformulation
# ---------------------------------------------------------------------------
def _subsets(n: int, lo: int, hi: int) -> list[frozenset[int]]:
    """All node subsets ``S ⊆ {0..n-1}`` with ``lo ≤ |S| ≤ hi`` (deterministic)."""
    out: list[frozenset[int]] = []
    nodes = list(range(n))
    for mask in range(1, 1 << n):
        members = [nodes[b] for b in range(n) if mask & (1 << b)]
        if lo <= len(members) <= hi:
            out.append(frozenset(members))
    out.sort(key=lambda s: (len(s), sorted(s)))
    return out


def cutset_inequalities(n: int, *, max_subset_size: int | None = None) -> list[LinearConstraint]:
    """Return the directed cutset / subtour-elimination inequalities for ``n`` nodes.

    For every proper subset ``S`` with ``2 ≤ |S| ≤ n-2`` (sizes 1 and ``n-1`` are
    implied by the degree rows, so they are skipped), emit
    ``Σ_{i∈S, j∉S} x_ij ≥ 1`` — the constraint that at least one arc leaves ``S``.
    This is exactly the projection of the multi-commodity-flow formulation onto
    the arc variables. ``max_subset_size`` caps ``|S|`` to bound the count.
    """
    hi = n - 2 if max_subset_size is None else min(max_subset_size, n - 2)
    cons: list[LinearConstraint] = []
    for s in _subsets(n, 2, hi):
        coeffs = {_arc_name(i, j): 1.0 for (i, j) in _arcs(n) if i in s and j not in s}
        name = "sec_" + "_".join(str(v) for v in sorted(s))
        cons.append(LinearConstraint(name, coeffs, ConstraintSense.GE, 1.0))
    return cons


@dataclass(frozen=True, slots=True)
class Reformulation:
    """A certifiable formulation-family reformulation expressed as IR + deltas.

    Attributes:
        family: Target family id (e.g. ``"mcf"``).
        before: The source IR (e.g. the MTZ model).
        deltas: Ordered class-A/B deltas that, applied in sequence to ``before``,
            realise the reformulation. Each passes :func:`verify_delta`.
    """

    family: str
    before: MILP
    deltas: tuple[Delta, ...]


def _tag_kind(delta: Delta, kind: str) -> Delta:
    """Return ``delta`` with a ``"kind"`` tag merged into its JSON payload.

    The tag is read by :func:`opop.proposer.stages.delta_kind` for staging; the
    IR ops (``add_constraint`` / ``rename_var``) ignore the extra key, so
    :func:`opop.model.ir.apply_delta` and :func:`opop.verify.gate.verify_delta`
    are unaffected (verified: a kind-tagged class-B cut still certifies PASS).
    """
    if not delta.after_fragment:
        return delta
    payload = json.loads(delta.after_fragment)
    if not isinstance(payload, dict):
        return delta
    payload["kind"] = kind
    return replace(delta, after_fragment=json.dumps(payload))


def _cutset_delta(con: LinearConstraint, family: str) -> Delta:
    """Wrap one cutset inequality as a kind-tagged class-B formulation delta."""
    support = "+".join(sorted(con.coeffs))
    rationale = (
        f"formulation reformulation -> {family}: cutset inequality '{con.name}' "
        f"({support} {con.sense.value} {con.rhs:g}); multi-commodity-flow projection"
    )
    delta = make_add_constraint_delta(con.name, con.coeffs, con.sense, con.rhs, target=rationale)
    return _tag_kind(delta, KIND_FORMULATION)


def mtz_to_flow_reformulation(
    mtz_ir: MILP, *, target: str = "mcf", max_subset_size: int | None = None
) -> Reformulation:
    """Reformulate an MTZ TSP toward the (multi-commodity) flow family.

    Returns a :class:`Reformulation` whose ``deltas`` are class-B additions of
    the cutset inequalities that the flow formulation enforces — each certified
    PASS by :func:`opop.verify.gate.verify_delta` (no feasible Hamiltonian tour
    is removed). Applying every delta yields the MTZ model strengthened to the
    flow formulation's LP-projection; it solves to the same integer optimum as
    both the pure MTZ and the pure MCF models.

    ``mtz_ir`` must be a routing-family IR (``metadata['domain'] == 'routing'``)
    carrying ``metadata['n_nodes']``.
    """
    n = _require_routing(mtz_ir)
    deltas = tuple(
        _cutset_delta(con, target) for con in cutset_inequalities(n, max_subset_size=max_subset_size)
    )
    return Reformulation(family=target, before=mtz_ir, deltas=deltas)


def encoding_relabel_delta(ir: MILP, *, family: str = "edge_encoding") -> Delta | None:
    """Return a class-A variable-encoding relabel for a routing IR, else ``None``.

    Renames the first arc variable ``x_0_1`` to a canonical edge-encoding name
    ``e_0_1`` — a genuinely equivalent reformulation (identity feasible region +
    objective, a single rename), certifiable as **class A**. This is the
    variable-encoding axis (Phi ``v``) of a formulation family. Returns ``None``
    if the IR has no such arc variable.
    """
    old = "x_0_1"
    if not any(v.name == old for v in ir.variables):
        return None
    new = "e_0_1"
    if any(v.name == new for v in ir.variables):
        return None
    rationale = f"formulation reformulation -> {family}: canonical edge encoding {old} -> {new}"
    return _tag_kind(make_rename_delta(old, new, target=rationale), KIND_FORMULATION)


# ---------------------------------------------------------------------------
# Pool integration
# ---------------------------------------------------------------------------
def _require_routing(ir: MILP) -> int:
    """Return ``n_nodes`` for a routing IR, raising :class:`ValueError` otherwise."""
    if ir.metadata.get("domain") != "routing":
        raise ValueError("formulation family reformulation requires a routing-domain IR")
    n = ir.metadata.get("n_nodes")
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("routing IR metadata must carry an integer 'n_nodes'")
    return n


def is_routing_ir(ir: MILP) -> bool:
    """Return ``True`` iff ``ir`` is a routing-family IR this module can reformulate."""
    n = ir.metadata.get("n_nodes")
    return ir.metadata.get("domain") == "routing" and isinstance(n, int) and not isinstance(n, bool)


def family_deltas(ir: MILP, *, max_cuts: int = 4) -> list[Delta]:
    """Return a bounded set of class-A/B formulation-family deltas for ``ir``.

    For a routing IR: one class-A encoding relabel (if applicable) followed by up
    to ``max_cuts`` class-B cutset reformulation deltas (the MCF projection). For
    a non-routing IR: an empty list (graceful — no family known). Every returned
    delta is kind ``formulation`` (stage S3+) and class A or B — never class D.
    """
    if not is_routing_ir(ir):
        return []
    out: list[Delta] = []
    relabel = encoding_relabel_delta(ir)
    if relabel is not None:
        out.append(relabel)
    reform = mtz_to_flow_reformulation(ir)
    out.extend(reform.deltas[: max(0, max_cuts)])
    return out
