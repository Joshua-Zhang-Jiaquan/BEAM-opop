"""Tests for formulation families + staged search spaces S0–S4 (task 27).

Two concerns are exercised:

* **Formulation families** (:mod:`opop.proposer.families`) — the flagship TSP
  ``MTZ ↔ multi-commodity-flow`` reformulation, certified equivalent by the
  verification gate (:func:`opop.verify.gate.verify_delta`) with the same
  optimum on both formulations. The verify-gate tests require SCIP and are
  guarded by ``solver_skip_if_missing("scip")``; the pure-IR / staging tests are
  network- and solver-free.
* **Staged search spaces** (:mod:`opop.proposer.stages`) — ``stage_filter`` /
  ``stage_space`` gate deltas by kind: S0 params only, S1 +safe cuts, S2
  +heuristics, S3 +formulation/decomposition, S4 +multi-kernel/transfer. The
  key acceptance: **stage S1 cannot emit a formulation delta.**
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from opop.analyzer.report import AnalysisReport, RelaxationMetrics
from opop.model.ir import (
    MILP,
    ConstraintSense,
    apply_delta,
    make_metadata_delta,
)
from opop.model.state import Delta, DeltaClass, Phi, ProblemState
from opop.proposer import (
    FAMILIES,
    Stage,
    build_candidate_pool,
    build_tsp_mcf,
    build_tsp_mtz,
    build_tsp_scf,
    cutset_inequalities,
    delta_kind,
    encoding_relabel_delta,
    family_deltas,
    make_param_delta,
    mtz_to_flow_reformulation,
    propose,
    stage_filter,
    stage_space,
)
from opop.proposer.stages import (
    KIND_CUT,
    KIND_DECOMPOSITION,
    KIND_FORMULATION,
    KIND_HEURISTIC,
    KIND_MULTIKERNEL,
    KIND_PARAM,
    allowed_kinds,
    parse_stage,
    stage_allows,
)
from opop.proposer.templates import cut_deltas_from_report
from opop.solver.scip import ScipKernel
from opop.verify.certificate import STATUS_PASS
from opop.verify.gate import OBJ_TOL, verify_delta

N = 5  # 5-node TSP, per the QA scenario.


def _optimum(ir: MILP) -> float:
    """Solve ``ir`` to optimality via the public SCIP kernel; return the optimum.

    Uses :meth:`opop.solver.scip.ScipKernel.solve` (public) rather than the
    gate's private solve helper, keeping the test on the supported surface. The
    proven optimum is the final entry of the primal-bound series.
    """
    trace = ScipKernel().solve(ir, Phi(), time_limit=30.0, memory_limit_mb=4096, seed=0)
    assert trace.status == "optimal", f"expected optimal, got {trace.status!r}"
    assert trace.primal_bound_series
    return trace.primal_bound_series[-1]


# ===========================================================================
# IR builders — well-formed, distinct, routing-tagged
# ===========================================================================
def test_tsp_builders_are_routing_tagged_and_well_formed() -> None:
    for build in (build_tsp_mtz, build_tsp_scf, build_tsp_mcf):
        ir = build(N)
        assert ir.metadata["domain"] == "routing"
        assert ir.metadata["n_nodes"] == N
        # n*(n-1) directed arc variables present.
        arc_vars = [v for v in ir.variables if v.name.startswith("x_")]
        assert len(arc_vars) == N * (N - 1)


def test_mtz_and_mcf_have_different_variable_sets() -> None:
    # The literal swap is NOT class-A relabelable (this is WHY we certify via
    # class-B cutset inequalities instead).
    mtz = {v.name for v in build_tsp_mtz(N).variables}
    mcf = {v.name for v in build_tsp_mcf(N).variables}
    assert mtz != mcf
    assert len(mcf) > len(mtz)  # MCF adds per-commodity flow variables.


def test_family_registry_covers_all_three_domains() -> None:
    domains = {f.domain for f in FAMILIES}
    assert {"routing", "scheduling", "lot_sizing"} <= domains
    names = {f.name for f in FAMILIES}
    assert {"mtz", "scf", "mcf"} <= names


# ===========================================================================
# Cutset inequalities (the MCF projection) — structure
# ===========================================================================
def test_cutset_inequalities_are_ge_one_over_arcs() -> None:
    cuts = cutset_inequalities(N)
    assert cuts
    for con in cuts:
        assert con.sense is ConstraintSense.GE
        assert con.rhs == 1.0
        assert all(name.startswith("x_") for name in con.coeffs)
        assert all(coeff == 1.0 for coeff in con.coeffs.values())
    # Names are unique.
    assert len({c.name for c in cuts}) == len(cuts)


# ===========================================================================
# Family deltas are class A/B + kind 'formulation' — NEVER class D
# ===========================================================================
def test_family_deltas_are_class_ab_and_formulation_kind() -> None:
    deltas = family_deltas(build_tsp_mtz(N))
    assert deltas
    for d in deltas:
        assert d.declared_class in {DeltaClass.A, DeltaClass.B}
        assert d.declared_class is not DeltaClass.D
        assert delta_kind(d) == KIND_FORMULATION


def test_family_deltas_empty_for_non_routing_ir() -> None:
    # A bare non-routing IR yields no family deltas (graceful).
    from opop.model.ir import MILP, Objective, Variable, VarType

    ir = MILP(name="plain", variables=(Variable("a", VarType.BINARY, 0.0, 1.0),), objective=Objective())
    assert family_deltas(ir) == []


def test_encoding_relabel_is_class_a_formulation() -> None:
    relabel = encoding_relabel_delta(build_tsp_mtz(N))
    assert relabel is not None
    assert relabel.declared_class is DeltaClass.A
    assert delta_kind(relabel) == KIND_FORMULATION


# ===========================================================================
# FLAGSHIP — MTZ ↔ multi-commodity flow certified equivalent (same optimum)
# ===========================================================================
def test_mtz_and_mcf_same_optimum(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    mtz_opt = _optimum(build_tsp_mtz(N))
    mcf_opt = _optimum(build_tsp_mcf(N))
    # Same optimum on both formulations (the QA expectation).
    assert abs(mtz_opt - mcf_opt) <= OBJ_TOL


def test_mtz_to_flow_reformulation_certified_by_gate(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    mtz = build_tsp_mtz(N)
    reform = mtz_to_flow_reformulation(mtz, target="mcf")
    assert reform.family == "mcf"
    assert reform.deltas  # non-empty set of cutset additions

    base_opt = _optimum(mtz)

    # Each reformulation delta is certified class-B PASS, and applying it keeps
    # the same optimum (no feasible tour removed). Verify a bounded subset to
    # keep runtime tight; every cutset cut is valid by construction.
    after = mtz
    checked = 0
    for delta in reform.deltas:
        assert delta.declared_class is DeltaClass.B
        report = verify_delta(after, delta, time_limit=30.0)
        assert report.status == STATUS_PASS, report.reason
        assert report.delta_class == DeltaClass.B.value
        assert report.feasible_region_integer_preserved is True
        after = apply_delta(after, delta)
        checked += 1
        if checked >= 6:
            break
    assert checked >= 1

    # The strengthened model still solves to the MTZ optimum.
    assert abs(_optimum(after) - base_opt) <= OBJ_TOL


def test_encoding_relabel_certified_class_a(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    mtz = build_tsp_mtz(N)
    relabel = encoding_relabel_delta(mtz)
    assert relabel is not None
    report = verify_delta(mtz, relabel, time_limit=30.0)
    assert report.status == STATUS_PASS, report.reason
    assert report.delta_class == DeltaClass.A.value


# ===========================================================================
# Stage parsing / queries
# ===========================================================================
def test_parse_stage_accepts_enum_name_and_index() -> None:
    assert parse_stage(Stage.S2) is Stage.S2
    assert parse_stage("S3") is Stage.S3
    assert parse_stage("s1") is Stage.S1
    assert parse_stage(4) is Stage.S4
    with pytest.raises(ValueError, match="unknown stage"):
        parse_stage("S9")
    with pytest.raises(ValueError):
        parse_stage(True)


def test_allowed_kinds_are_cumulative() -> None:
    a0 = allowed_kinds(Stage.S0)
    a1 = allowed_kinds(Stage.S1)
    a2 = allowed_kinds(Stage.S2)
    a3 = allowed_kinds(Stage.S3)
    a4 = allowed_kinds(Stage.S4)
    # Monotone growth.
    assert a0 < a1 < a2 < a3 < a4
    assert a0 == {KIND_PARAM}
    assert a1 == {KIND_PARAM, KIND_CUT}
    assert a2 == {KIND_PARAM, KIND_CUT, KIND_HEURISTIC}
    assert a3 == {KIND_PARAM, KIND_CUT, KIND_HEURISTIC, KIND_FORMULATION, KIND_DECOMPOSITION}
    assert KIND_MULTIKERNEL in a4


def test_stage_allows_per_kind() -> None:
    assert stage_allows(Stage.S0, KIND_PARAM)
    assert not stage_allows(Stage.S0, KIND_CUT)
    assert stage_allows(Stage.S1, KIND_CUT)
    assert not stage_allows(Stage.S1, KIND_FORMULATION)
    assert stage_allows(Stage.S3, KIND_FORMULATION)
    assert stage_allows(Stage.S3, KIND_DECOMPOSITION)
    assert not stage_allows(Stage.S3, KIND_MULTIKERNEL)
    assert stage_allows(Stage.S4, KIND_MULTIKERNEL)


# ===========================================================================
# delta_kind classification
# ===========================================================================
def _report() -> AnalysisReport:
    from opop.model.ir import LinearConstraint

    cut = LinearConstraint("cover_c0", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0)
    return AnalysisReport(
        flags=(),
        relaxation_metrics=RelaxationMetrics(lp_obj=12.5, gap=0.25, ip_bound=15.0),
        candidate_cuts=(cut,),
    )


def test_delta_kind_infers_param_cut_heuristic_decomposition() -> None:
    param = make_param_delta("branching/scorefactor", 0.5)
    assert delta_kind(param) == KIND_PARAM
    heuristic = make_param_delta("heuristics/rens/freq", 5.0)
    assert delta_kind(heuristic) == KIND_HEURISTIC
    decomp = make_param_delta("decomposition/applybenders", 1.0)
    assert delta_kind(decomp) == KIND_DECOMPOSITION
    cut = cut_deltas_from_report(_report())[0]
    assert delta_kind(cut) == KIND_CUT


def test_delta_kind_metadata_noop_is_param() -> None:
    meta = make_metadata_delta({"note": "x"})
    assert delta_kind(meta) == KIND_PARAM


def test_delta_kind_explicit_tag_wins() -> None:
    formulation = family_deltas(build_tsp_mtz(N))[-1]  # a kind-tagged cutset cut
    assert delta_kind(formulation) == KIND_FORMULATION


# ===========================================================================
# stage_filter / stage_space gating
# ===========================================================================
def _mixed_deltas() -> list[Delta]:
    param = make_param_delta("branching/scorefactor", 0.5)
    heuristic = make_param_delta("heuristics/rens/freq", 5.0)
    cut = cut_deltas_from_report(_report())[0]
    formulation = family_deltas(build_tsp_mtz(N))[-1]
    return [param, cut, heuristic, formulation]


def test_stage_filter_s0_keeps_only_params() -> None:
    out = stage_filter(_mixed_deltas(), Stage.S0)
    assert out
    assert all(delta_kind(d) == KIND_PARAM for d in out)


def test_stage_filter_s1_keeps_param_and_cut_only() -> None:
    out = stage_filter(_mixed_deltas(), Stage.S1)
    kinds = {delta_kind(d) for d in out}
    assert kinds == {KIND_PARAM, KIND_CUT}
    assert KIND_FORMULATION not in kinds
    assert KIND_HEURISTIC not in kinds


def test_stage_filter_s2_adds_heuristics_not_formulation() -> None:
    kinds = {delta_kind(d) for d in stage_filter(_mixed_deltas(), Stage.S2)}
    assert KIND_HEURISTIC in kinds
    assert KIND_FORMULATION not in kinds


def test_stage_filter_s3_admits_formulation() -> None:
    kinds = {delta_kind(d) for d in stage_filter(_mixed_deltas(), Stage.S3)}
    assert KIND_FORMULATION in kinds


def test_stage_filter_preserves_order() -> None:
    deltas = _mixed_deltas()
    out = stage_filter(deltas, Stage.S4)
    assert out == deltas  # nothing dropped at S4; order intact


def test_stage_space_restricts_kind_set() -> None:
    space = [KIND_PARAM, KIND_CUT, KIND_FORMULATION, KIND_MULTIKERNEL]
    assert stage_space(space, Stage.S1) == {KIND_PARAM, KIND_CUT}
    assert stage_space(space, Stage.S0) == {KIND_PARAM}
    assert stage_space(None, Stage.S0) == {KIND_PARAM}
    # Accepts deltas too.
    assert stage_space(_mixed_deltas(), Stage.S1) == {KIND_PARAM, KIND_CUT}


# ===========================================================================
# propose() integration — stage gating + families in the pool
# ===========================================================================
def _state_with_tsp() -> ProblemState:
    ir = build_tsp_mtz(N)
    return ProblemState(instance_id="tsp", task_family="MILP", budget_state={"ir": ir})


def test_propose_s1_cannot_emit_a_formulation_delta() -> None:
    # The KEY acceptance: at S1, even with families allowed in the pool, NO
    # formulation/decomposition delta can be emitted.
    state = _state_with_tsp()
    report = _report()
    result = propose(state, report, stage=Stage.S1, allow_families=True, max_deltas=99)
    assert result  # still emits param + safe-cut deltas
    kinds = {delta_kind(d) for d in result}
    assert KIND_FORMULATION not in kinds
    assert KIND_DECOMPOSITION not in kinds
    assert kinds <= {KIND_PARAM, KIND_CUT}


def test_propose_s3_emits_a_formulation_delta() -> None:
    state = _state_with_tsp()
    report = _report()
    result = propose(state, report, stage=Stage.S3, allow_families=True, max_deltas=99)
    assert any(delta_kind(d) == KIND_FORMULATION for d in result)


def test_propose_s0_emits_only_params() -> None:
    result = propose(_state_with_tsp(), _report(), stage=Stage.S0, allow_families=True, max_deltas=99)
    assert result
    assert all(delta_kind(d) == KIND_PARAM for d in result)


def test_propose_default_stage_is_full_and_families_off_by_default() -> None:
    # Default propose() == Phase-1 behaviour: no family deltas, full stage.
    state = _state_with_tsp()
    report = _report()
    default = propose(state, report, max_deltas=99)
    assert not any(delta_kind(d) == KIND_FORMULATION for d in default)
    # The pool default also excludes families.
    pool = build_candidate_pool(state, report)
    assert not any(delta_kind(d) == KIND_FORMULATION for d in pool)


def test_propose_never_emits_class_d() -> None:
    for stage in (Stage.S0, Stage.S1, Stage.S2, Stage.S3, Stage.S4):
        result = propose(_state_with_tsp(), _report(), stage=stage, allow_families=True, max_deltas=99)
        assert all(d.declared_class is not DeltaClass.D for d in result)


def test_build_pool_with_families_appends_formulation_deltas() -> None:
    state = _state_with_tsp()
    report = _report()
    with_fam = build_candidate_pool(state, report, allow_families=True)
    without_fam = build_candidate_pool(state, report)
    assert len(with_fam) > len(without_fam)
    assert any(delta_kind(d) == KIND_FORMULATION for d in with_fam)
