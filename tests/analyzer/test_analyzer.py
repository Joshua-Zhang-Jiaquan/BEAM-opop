"""Tests for the Phase-1 OR-analysis layer.

Pure IR checks (consistency, redundancy, cut generation, report structure)
always run. LP-relaxation tests that need SCIP skip cleanly when it is absent.

Reference fixtures with known values:

* ``covering`` — ``min 5x+5y+5z s.t. x+y+z >= 2.5``, binary. LP relaxation
  optimum ``12.5``; integer optimum ``15`` (the sum must reach 3).
* ``triangle`` — ``max x1+x2+x3`` with three pairwise ``x_i+x_j <= 1`` rows,
  binary. LP optimum ``1.5`` at ``x_i = 0.5`` (all fractional); the three
  conflicts form a 3-clique, so ``x1+x2+x3 <= 1`` is a valid clique cut.
* ``knapsack`` — weights ``[5,3,7,4,6]`` with capacity ``12``: ``{i2,i4}``
  (``7+6=13>12``) is a minimal cover, yielding ``x_i2 + x_i4 <= 1``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from opop.analyzer import (
    CONFLICT,
    DIMENSION_MISMATCH,
    INDEX_ERROR,
    REDUNDANT,
    TRIVIAL_INFEASIBILITY,
    UNITS_MISMATCH,
    AnalysisReport,
    analyze,
    check_consistency,
    detect_redundancy,
    generate_clique_cuts,
    generate_cover_cuts,
    generate_valid_inequalities,
    relaxed_ir,
)
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)


# ---------------------------------------------------------------------------
# Fixture builders (hand-built IR; no file I/O)
# ---------------------------------------------------------------------------
def _bin(name: str) -> Variable:
    return Variable(name, VarType.BINARY, 0.0, 1.0)


def covering_milp() -> MILP:
    """min 5x+5y+5z s.t. x+y+z >= 2.5 (LP=12.5, IP=15)."""
    return MILP(
        name="covering",
        variables=(_bin("x"), _bin("y"), _bin("z")),
        constraints=(
            LinearConstraint("cover", {"x": 1.0, "y": 1.0, "z": 1.0}, ConstraintSense.GE, 2.5),
        ),
        objective=Objective({"x": 5.0, "y": 5.0, "z": 5.0}, ObjSense.MINIMIZE),
    )


def triangle_milp() -> MILP:
    """max x1+x2+x3 with pairwise conflicts (LP=1.5, all fractional; 3-clique)."""
    return MILP(
        name="triangle",
        variables=(_bin("x1"), _bin("x2"), _bin("x3")),
        constraints=(
            LinearConstraint("e12", {"x1": 1.0, "x2": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("e23", {"x2": 1.0, "x3": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("e13", {"x1": 1.0, "x3": 1.0}, ConstraintSense.LE, 1.0),
        ),
        objective=Objective({"x1": 1.0, "x2": 1.0, "x3": 1.0}, ObjSense.MAXIMIZE),
    )


def knapsack_milp() -> MILP:
    """max value s.t. 5i0+3i1+7i2+4i3+6i4 <= 12 (cover {i2,i4}: 13 > 12)."""
    weights = {"i0": 5.0, "i1": 3.0, "i2": 7.0, "i3": 4.0, "i4": 6.0}
    values = {"i0": 8.0, "i1": 5.0, "i2": 11.0, "i3": 6.0, "i4": 9.0}
    return MILP(
        name="knapsack",
        variables=tuple(_bin(n) for n in weights),
        constraints=(LinearConstraint("cap", weights, ConstraintSense.LE, 12.0),),
        objective=Objective(values, ObjSense.MAXIMIZE),
    )


# ===========================================================================
# LP-relaxation statistics (SCIP-dependent)
# ===========================================================================
def test_lp_gap_matches_known_fixture(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    report = analyze(covering_milp(), ip_bound=15.0)
    metrics = report.relaxation_metrics
    assert metrics.lp_status == "OPTIMAL"
    assert metrics.lp_obj == pytest.approx(12.5, abs=1e-6)
    assert report.lp_obj == pytest.approx(12.5, abs=1e-6)
    assert report.lp_gap == pytest.approx((15.0 - 12.5) / 15.0, abs=1e-6)
    assert metrics.ip_bound == pytest.approx(15.0)


def test_fractional_pattern_on_triangle(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    metrics = analyze(triangle_milp(), estimate_ip_bound=False).relaxation_metrics
    assert metrics.lp_obj == pytest.approx(1.5, abs=1e-6)
    assert metrics.n_fractional == 3
    assert set(metrics.fractional_vars) == {"x1", "x2", "x3"}


def test_continuous_var_not_counted_fractional(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    # A pure-LP fixture: the continuous var is 0.5 but must NOT count as fractional.
    milp = MILP(
        name="cont",
        variables=(Variable("c", VarType.CONTINUOUS, 0.0, 1.0), _bin("b")),
        constraints=(
            LinearConstraint("r", {"c": 1.0, "b": 1.0}, ConstraintSense.LE, 1.5),
        ),
        objective=Objective({"c": -1.0, "b": -1.0}, ObjSense.MINIMIZE),
    )
    metrics = analyze(milp, estimate_ip_bound=False).relaxation_metrics
    assert "c" not in metrics.fractional_vars


def test_ip_bound_estimate_path(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    # No explicit bound and no metadata: the estimate solves the tiny IP -> 15.
    metrics = analyze(covering_milp(), estimate_ip_bound=True).relaxation_metrics
    assert metrics.ip_bound == pytest.approx(15.0)
    assert metrics.gap == pytest.approx((15.0 - 12.5) / 15.0, abs=1e-6)


def test_ip_bound_from_metadata(solver_skip_if_missing: Callable[[str], None]) -> None:
    solver_skip_if_missing("scip")
    milp = covering_milp()
    milp = MILP(
        name=milp.name,
        variables=milp.variables,
        constraints=milp.constraints,
        objective=milp.objective,
        metadata={"known_optimum": 15.0},
    )
    metrics = analyze(milp, estimate_ip_bound=False).relaxation_metrics
    assert metrics.ip_bound == pytest.approx(15.0)
    assert metrics.gap == pytest.approx((15.0 - 12.5) / 15.0, abs=1e-6)


def test_relaxation_skipped_without_solver() -> None:
    report = analyze(triangle_milp(), solve_relaxation=False)
    assert report.relaxation_metrics.lp_status == "SKIPPED"
    assert report.relaxation_metrics.lp_obj is None
    assert report.lp_gap is None


def test_relaxed_ir_is_pure_and_continuous() -> None:
    milp = covering_milp()
    relaxed = relaxed_ir(milp)
    assert all(v.vtype is VarType.CONTINUOUS for v in relaxed.variables)
    # purity: original untouched
    assert all(v.vtype is VarType.BINARY for v in milp.variables)
    # bounds preserved
    assert {v.name: (v.lower, v.upper) for v in relaxed.variables} == {
        "x": (0.0, 1.0),
        "y": (0.0, 1.0),
        "z": (0.0, 1.0),
    }


# ===========================================================================
# Redundancy / trivial-infeasibility / conflict (pure)
# ===========================================================================
def test_redundant_duplicate_constraint_flagged() -> None:
    milp = MILP(
        name="dup",
        variables=(_bin("x"), _bin("y")),
        constraints=(
            LinearConstraint("c1", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("c1_dup", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
        ),
    )
    report = analyze(milp, solve_relaxation=False)
    assert "c1_dup" in report.locations_by_type(REDUNDANT)
    assert "c1" not in report.locations_by_type(REDUNDANT)


def test_dominated_constraint_flagged() -> None:
    flags = detect_redundancy(
        MILP(
            name="dom",
            variables=(_bin("x"), _bin("y")),
            constraints=(
                LinearConstraint("tight", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
                LinearConstraint("loose", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 5.0),
            ),
        )
    )
    redundant = {f.location for f in flags if f.type == REDUNDANT}
    assert redundant == {"loose"}


def test_scaled_duplicate_flagged() -> None:
    # 2x + 2y <= 4  is the same constraint as  x + y <= 2.
    flags = detect_redundancy(
        MILP(
            name="scaled",
            variables=(_bin("x"), _bin("y")),
            constraints=(
                LinearConstraint("base", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 2.0),
                LinearConstraint("scaled", {"x": 2.0, "y": 2.0}, ConstraintSense.LE, 4.0),
            ),
        )
    )
    assert "scaled" in {f.location for f in flags if f.type == REDUNDANT}


def test_negated_constraint_detected_as_same_family() -> None:
    # -x - y <= -2  is exactly  x + y >= 2; pairing with x + y <= 1 is a conflict.
    flags = detect_redundancy(
        MILP(
            name="neg",
            variables=(_bin("x"), _bin("y")),
            constraints=(
                LinearConstraint("le", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
                LinearConstraint("ge_neg", {"x": -1.0, "y": -1.0}, ConstraintSense.LE, -2.0),
            ),
        )
    )
    assert any(f.type == CONFLICT for f in flags)


def test_trivial_infeasibility_flagged() -> None:
    milp = MILP(
        name="triv",
        variables=(_bin("x"),),
        constraints=(LinearConstraint("bad", {}, ConstraintSense.LE, -1.0),),
    )
    flags = detect_redundancy(milp)
    assert any(f.type == TRIVIAL_INFEASIBILITY and f.location == "bad" for f in flags)


def test_always_true_empty_row_is_redundant_not_infeasible() -> None:
    milp = MILP(
        name="ok_empty",
        variables=(_bin("x"),),
        constraints=(LinearConstraint("ok", {}, ConstraintSense.LE, 3.0),),
    )
    flags = detect_redundancy(milp)
    assert any(f.type == REDUNDANT and f.location == "ok" for f in flags)
    assert not any(f.type == TRIVIAL_INFEASIBILITY for f in flags)


def test_empty_variable_domain_flagged() -> None:
    milp = MILP(
        name="emptydom",
        variables=(Variable("v", VarType.INTEGER, 5.0, 2.0),),
        constraints=(),
    )
    flags = detect_redundancy(milp)
    assert any(f.type == TRIVIAL_INFEASIBILITY and f.location == "v" for f in flags)


def test_conflict_detection() -> None:
    milp = MILP(
        name="conflict",
        variables=(_bin("x"), _bin("y")),
        constraints=(
            LinearConstraint("upper", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("lower", {"x": 1.0, "y": 1.0}, ConstraintSense.GE, 2.0),
        ),
    )
    flags = detect_redundancy(milp)
    assert any(f.type == CONFLICT for f in flags)


def test_clean_model_has_no_redundancy_flags() -> None:
    assert detect_redundancy(triangle_milp()) == []
    assert detect_redundancy(knapsack_milp()) == []


# ===========================================================================
# Consistency: index / dimension / units (pure)
# ===========================================================================
def _annotated_milp(metadata: dict[str, object]) -> MILP:
    return MILP(
        name="annotated",
        variables=(_bin("x"), _bin("y")),
        constraints=(
            LinearConstraint("c_ok", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("c_bad", {"x": 1.0}, ConstraintSense.LE, 1.0),
        ),
        objective=Objective({"x": 1.0}, ObjSense.MAXIMIZE),
        index_sets={"I": ("0", "1")},
        metadata=metadata,
    )


def test_index_error_flagged() -> None:
    milp = _annotated_milp(
        {"index_annotations": {"c_bad": {"I": "5"}, "c_ok": {"I": "0"}}}
    )
    flags = check_consistency(milp)
    index_locs = {f.location for f in flags if f.type == INDEX_ERROR}
    assert index_locs == {"c_bad"}  # member "5" missing; "0" is valid -> c_ok clean


def test_index_undeclared_set_flagged() -> None:
    flags = check_consistency(_annotated_milp({"index_annotations": {"c_ok": {"J": "0"}}}))
    assert any(f.type == INDEX_ERROR and "undeclared set" in f.message for f in flags)


def test_index_dangling_annotation_flagged() -> None:
    flags = check_consistency(_annotated_milp({"index_annotations": {"ghost": {"I": "0"}}}))
    assert any(f.type == INDEX_ERROR and f.location == "ghost" for f in flags)


def test_dimension_mismatch_flagged() -> None:
    flags = check_consistency(_annotated_milp({"dimension_specs": {"c_ok": 3}}))
    assert any(f.type == DIMENSION_MISMATCH and f.location == "c_ok" for f in flags)


def test_dimension_match_is_clean() -> None:
    flags = check_consistency(_annotated_milp({"dimension_specs": {"c_ok": 2, "c_bad": 1}}))
    assert not any(f.type == DIMENSION_MISMATCH for f in flags)


def test_units_mismatch_flagged() -> None:
    flags = check_consistency(_annotated_milp({"variable_units": {"x": "kg", "y": "m"}}))
    assert any(f.type == UNITS_MISMATCH and f.location == "c_ok" for f in flags)


def test_no_annotations_no_flags() -> None:
    assert check_consistency(triangle_milp()) == []


# ===========================================================================
# Valid-inequality candidates: cover + clique (pure)
# ===========================================================================
def test_clique_candidate_on_set_packing() -> None:
    cuts = generate_clique_cuts(triangle_milp())
    cliques = [c for c in cuts if set(c.coeffs) == {"x1", "x2", "x3"}]
    assert len(cliques) >= 1
    cut = cliques[0]
    assert cut.sense is ConstraintSense.LE
    assert cut.rhs == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in cut.coeffs.values())


def test_cover_candidate_on_knapsack() -> None:
    cuts = generate_cover_cuts(knapsack_milp())
    minimal_cover = [c for c in cuts if set(c.coeffs) == {"i2", "i4"}]
    assert len(minimal_cover) == 1
    cut = minimal_cover[0]
    assert cut.sense is ConstraintSense.LE
    assert cut.rhs == pytest.approx(1.0)  # |C| - 1 == 1


def test_set_packing_fixture_yields_at_least_one_candidate() -> None:
    # Acceptance: >= 1 clique/cover candidate on a set-packing fixture.
    assert len(generate_valid_inequalities(triangle_milp())) >= 1


def test_candidate_cuts_exclude_existing_constraints() -> None:
    # The triangle's pairwise rows ARE 2-item covers; those must be filtered as
    # already-present, leaving only the novel 3-clique cut.
    cuts = generate_valid_inequalities(triangle_milp())
    supports = {frozenset(c.coeffs) for c in cuts}
    assert frozenset({"x1", "x2"}) not in supports
    assert frozenset({"x1", "x2", "x3"}) in supports


def test_cover_cuts_ignore_non_knapsack_rows() -> None:
    # An equality row over binaries is not a <= knapsack -> no cover cuts.
    milp = MILP(
        name="eqrow",
        variables=(_bin("a"), _bin("b")),
        constraints=(LinearConstraint("eq", {"a": 1.0, "b": 1.0}, ConstraintSense.EQ, 1.0),),
    )
    assert generate_cover_cuts(milp) == []


# ===========================================================================
# analyze() report structure + purity
# ===========================================================================
def test_analyze_returns_structured_report() -> None:
    report = analyze(triangle_milp(), solve_relaxation=False)
    assert isinstance(report, AnalysisReport)
    assert report.decomposability == "NONE"
    assert isinstance(report.flags, tuple)
    assert isinstance(report.candidate_cuts, tuple)
    assert report.relaxation_metrics is not None


def test_analyze_does_not_mutate_ir() -> None:
    milp = triangle_milp()
    before = (milp.variables, milp.constraints, milp.objective, dict(milp.metadata))
    analyze(milp, solve_relaxation=False)
    assert milp.variables == before[0]
    assert milp.constraints == before[1]
    assert milp.objective == before[2]
    assert dict(milp.metadata) == before[3]


def test_report_to_dict_is_serializable() -> None:
    import json

    report = analyze(knapsack_milp(), solve_relaxation=False)
    payload = report.to_dict()
    text = json.dumps(payload)  # must not raise
    assert "candidate_cuts" in text
    assert payload["decomposability"] == "NONE"
    assert isinstance(payload["flags"], list)
    assert "lp_obj" in payload["relaxation_metrics"]


def test_full_analyze_combines_all_signals(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    solver_skip_if_missing("scip")
    report = analyze(knapsack_milp())
    assert report.relaxation_metrics.lp_status == "OPTIMAL"
    assert report.lp_obj is not None
    assert len(report.candidate_cuts) >= 1
    assert report.decomposability == "NONE"
