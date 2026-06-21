"""Tests for the Phase-1 restricted proposer.

All tests are network-free: the :class:`opop.llm.FakeLLMClient` returns canned
replies and the :class:`opop.analyzer.report.AnalysisReport` is hand-built (no
SCIP solve), so the suite is fully deterministic.

The proposer's contract (task 14):

* return ``<= max_deltas`` typed deltas, each with a declared class in
  ``{A, B, C}`` — NEVER class-D into the main path;
* class-B deltas come ONLY from ``report.candidate_cuts`` (the analyzer
  whitelist); class-C deltas come from the curated SCIP knob list;
* the LLM SELECTS from the typed pool — a hallucinated / illegal selection is
  dropped and logged, and the proposer falls back to the deterministic
  rule-based ranker (which a generic ``FakeLLMClient`` also triggers).
"""

from __future__ import annotations

import json
import logging
from typing import cast

import pytest

from opop.analyzer.report import AnalysisReport, RelaxationMetrics
from opop.llm import FakeLLMClient, LLMClient
from opop.model.ir import ConstraintSense, LinearConstraint
from opop.model.state import Delta, DeltaClass, ProblemState
from opop.proposer import (
    CURATED_PARAMS,
    build_candidate_pool,
    curated_param_deltas,
    cut_deltas_from_report,
    decomposition_flag_delta,
    make_param_delta,
    param_from_delta,
    propose,
    propose_rule_based,
)
from opop.solver.scip import WHITELISTED_SEPARATORS

# ---------------------------------------------------------------------------
# Fixtures / builders (hand-built; no SCIP, no network)
# ---------------------------------------------------------------------------
_CUT1 = LinearConstraint("cover_c0_0", {"x": 1.0, "y": 1.0}, ConstraintSense.LE, 1.0)
_CUT2 = LinearConstraint("cover_c0_1", {"y": 1.0, "z": 1.0}, ConstraintSense.LE, 1.0)


def _report(
    *,
    gap: float | None = 0.25,
    cuts: tuple[LinearConstraint, ...] = (_CUT1, _CUT2),
    decomposability: str = "NONE",
) -> AnalysisReport:
    """Build an AnalysisReport with two cover-cut candidates + an LP gap."""
    return AnalysisReport(
        flags=(),
        relaxation_metrics=RelaxationMetrics(
            lp_obj=12.5,
            gap=gap,
            n_fractional=3,
            fractional_vars=("x", "y", "z"),
            lp_status="OPTIMAL",
            ip_bound=15.0,
        ),
        candidate_cuts=cuts,
        decomposability=decomposability,
    )


def _state() -> ProblemState:
    return ProblemState(instance_id="t0", task_family="MILP")


def _added_constraint_name(delta: Delta) -> str | None:
    """Return the constraint name of a class-B add_constraint delta, else None."""
    if not delta.after_fragment:
        return None
    payload = json.loads(delta.after_fragment)
    if payload.get("op") == "add_constraint":
        name = payload.get("name")
        return name if isinstance(name, str) else None
    return None


# ===========================================================================
# Acceptance: >= 1 typed whitelist delta; never class-D; carries DeltaClass
# ===========================================================================
def test_propose_returns_at_least_one_typed_whitelist_delta() -> None:
    result = propose(_state(), _report())
    assert len(result) >= 1
    assert all(d.declared_class in {DeltaClass.A, DeltaClass.B, DeltaClass.C} for d in result)
    # >= 1 analyzer-flagged cut (class-B) is included.
    assert any(d.declared_class is DeltaClass.B for d in result)


def test_every_delta_carries_a_valid_deltaclass() -> None:
    # Across all three entry paths.
    reports = _report()
    state = _state()
    runs = [
        propose(state, reports),
        propose(state, reports, llm=FakeLLMClient(response=json.dumps({"selected": [0, 1]}))),
        propose(state, reports, llm=FakeLLMClient(response="free-form reformulation idea")),
    ]
    for result in runs:
        for delta in result:
            assert isinstance(delta.declared_class, DeltaClass)
            assert delta.declared_class is not DeltaClass.D


def test_never_emits_class_d_even_with_many_candidates() -> None:
    result = propose(_state(), _report(), max_deltas=99)
    assert result  # non-empty
    assert all(d.declared_class is not DeltaClass.D for d in result)


# ===========================================================================
# Respects analysis candidates (class-B only from report.candidate_cuts)
# ===========================================================================
def test_class_b_deltas_only_from_candidate_cuts() -> None:
    report = _report()
    candidate_names = {c.name for c in report.candidate_cuts}
    result = propose(_state(), report, max_deltas=99)
    emitted_cut_names = {
        _added_constraint_name(d) for d in result if d.declared_class is DeltaClass.B
    }
    emitted_cut_names.discard(None)
    assert emitted_cut_names <= candidate_names
    # All flagged candidates are representable (present in the full pool).
    pool_cut_names = {
        _added_constraint_name(d)
        for d in build_candidate_pool(_state(), report)
        if d.declared_class is DeltaClass.B
    }
    pool_cut_names.discard(None)
    assert pool_cut_names == candidate_names


def test_no_candidate_cuts_means_no_class_b_deltas() -> None:
    report = _report(cuts=())
    result = propose(_state(), report)
    assert result  # still proposes params
    assert all(d.declared_class is DeltaClass.C for d in result)
    assert not any(d.declared_class is DeltaClass.B for d in result)


def test_empty_report_yields_only_curated_params() -> None:
    # No cuts and NONE decomposability -> only class-C curated params.
    report = _report(cuts=(), gap=None)
    pool = build_candidate_pool(_state(), report)
    assert pool
    assert all(d.declared_class is DeltaClass.C for d in pool)


# ===========================================================================
# LLM-guided selection path (typed templates only)
# ===========================================================================
def test_llm_valid_index_selection_returns_pool_subset() -> None:
    report = _report()
    pool = build_candidate_pool(_state(), report)
    llm = FakeLLMClient(response=json.dumps({"selected": [0, 1]}))
    result = propose(_state(), report, llm=llm)
    assert result == pool[:2]
    assert all(d.declared_class is DeltaClass.B for d in result)


def test_llm_selection_by_candidate_id() -> None:
    report = _report()
    # The cut's candidate id is its constraint name.
    llm = FakeLLMClient(response=json.dumps({"selected": ["cover_c0_1"]}))
    result = propose(_state(), report, llm=llm)
    assert len(result) == 1
    assert _added_constraint_name(result[0]) == "cover_c0_1"


def test_llm_ranking_key_is_accepted() -> None:
    report = _report()
    pool = build_candidate_pool(_state(), report)
    llm = FakeLLMClient(response=json.dumps({"ranking": [1, 0]}))
    result = propose(_state(), report, llm=llm)
    assert result == [pool[1], pool[0]]


def test_llm_selection_respects_max_deltas() -> None:
    report = _report()
    llm = FakeLLMClient(response=json.dumps({"selected": [0, 1, 2, 3, 4, 5, 6]}))
    result = propose(_state(), report, llm=llm, max_deltas=3)
    assert len(result) == 3


# ===========================================================================
# Safety envelope: hallucinated / illegal LLM output is filtered + logged
# ===========================================================================
def test_freeform_llm_reply_is_filtered_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _report()
    llm = FakeLLMClient(
        response="I suggest reformulating with a Dantzig-Wolfe decomposition and new lambda vars."
    )
    with caplog.at_level(logging.WARNING):
        result = propose(_state(), report, llm=llm)
    # Output stays within the typed space (fell back to rule-based).
    assert result
    assert all(d.declared_class in {DeltaClass.A, DeltaClass.B, DeltaClass.C} for d in result)
    assert any(d.declared_class is DeltaClass.B for d in result)
    # No free-form content leaked into any delta.
    for delta in result:
        blob = (delta.after_fragment or "") + delta.target
        assert "Dantzig" not in blob
        assert "lambda" not in blob
    # A rejection + fallback was logged.
    assert "not JSON" in caplog.text
    assert "rule-based fallback" in caplog.text


def test_illegal_indices_and_names_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _report()
    pool = build_candidate_pool(_state(), report)
    llm = FakeLLMClient(
        response=json.dumps(
            {"selected": [99, -1, "reformulate_to_qubo", True, {"op": "evil"}]}
        )
    )
    with caplog.at_level(logging.WARNING):
        result = propose(_state(), report, llm=llm)
    # Every illegal entry dropped -> empty selection -> rule-based fallback.
    assert result
    assert all(d in pool for d in result)
    assert "dropping illegal/hallucinated LLM selection" in caplog.text


def test_generic_fake_llm_falls_back_to_rule_based() -> None:
    # A FakeLLMClient with no 'selected'/'ranking' key behaves like the
    # deterministic fallback (matches the task's "if FakeLLMClient, use rule_based").
    report = _report()
    state = _state()
    generic = FakeLLMClient(response=json.dumps({"answer": 42}))
    assert propose(state, report, llm=generic) == propose(state, report)


def test_fake_llm_fixture_falls_back(fake_llm: object) -> None:
    # The shared conftest fixture returns {"answer": 42} -> no usable selection.
    report = _report()
    result = propose(_state(), report, llm=cast("LLMClient", fake_llm))
    assert result
    assert all(d.declared_class is not DeltaClass.D for d in result)


def test_mixed_legal_and_illegal_keeps_only_legal(caplog: pytest.LogCaptureFixture) -> None:
    report = _report()
    pool = build_candidate_pool(_state(), report)
    llm = FakeLLMClient(response=json.dumps({"selected": [0, 999, 1]}))
    with caplog.at_level(logging.WARNING):
        result = propose(_state(), report, llm=llm)
    # 0 and 1 are legal, 999 is dropped; selection non-empty so NO fallback.
    assert result == [pool[0], pool[1]]
    assert "dropping illegal/hallucinated LLM selection: 999" in caplog.text


# ===========================================================================
# Curated SCIP params: class-C, whitelisted separators, extractable
# ===========================================================================
def test_curated_param_deltas_are_class_c_and_extractable() -> None:
    deltas = curated_param_deltas()
    assert deltas
    for delta in deltas:
        assert delta.declared_class is DeltaClass.C
        kv = param_from_delta(delta)
        assert kv is not None
        key, value = kv
        assert isinstance(key, str) and isinstance(value, float)


def test_separator_params_use_only_whitelisted_separators() -> None:
    for knob in CURATED_PARAMS:
        if knob.key.startswith("separating/"):
            sep = knob.key.split("/")[1]
            assert sep in WHITELISTED_SEPARATORS


def test_make_param_delta_rejects_non_whitelisted_separator() -> None:
    with pytest.raises(ValueError, match="not class-B whitelisted"):
        make_param_delta("separating/bogus/freq", 1.0)


def test_make_param_delta_allows_whitelisted_separator() -> None:
    delta = make_param_delta("separating/gomory/freq", 5.0)
    assert delta.declared_class is DeltaClass.C
    assert param_from_delta(delta) == ("separating/gomory/freq", 5.0)


def test_param_from_delta_returns_none_for_cut() -> None:
    cut_delta = cut_deltas_from_report(_report())[0]
    assert param_from_delta(cut_delta) is None


# ===========================================================================
# Decomposition flag stub (dormant in Phase-1)
# ===========================================================================
def test_decomposition_flag_dormant_when_none() -> None:
    pool = build_candidate_pool(_state(), _report(decomposability="NONE"))
    keys = {(param_from_delta(d) or ("", 0.0))[0] for d in pool}
    assert not any(k.startswith("decomposition/") for k in keys)


def test_decomposition_flag_emitted_when_structure_detected() -> None:
    # Phase-1 never sets this, but the stub must wire through if it ever does.
    pool = build_candidate_pool(_state(), _report(decomposability="BENDERS"))
    decomp = [d for d in pool if (param_from_delta(d) or ("", 0.0))[0].startswith("decomposition/")]
    assert len(decomp) == 1
    assert decomp[0].declared_class is DeltaClass.C


def test_decomposition_flag_delta_is_class_c() -> None:
    delta = decomposition_flag_delta()
    assert delta.declared_class is DeltaClass.C
    kv = param_from_delta(delta)
    assert kv is not None
    assert kv[0].startswith("decomposition/")


# ===========================================================================
# Pool ordering, ranking, max_deltas, rationale
# ===========================================================================
def test_pool_is_cuts_then_params() -> None:
    report = _report()
    pool = build_candidate_pool(_state(), report)
    n_cuts = len(report.candidate_cuts)
    assert all(d.declared_class is DeltaClass.B for d in pool[:n_cuts])
    assert all(d.declared_class is DeltaClass.C for d in pool[n_cuts:])
    assert len(pool) == n_cuts + len(curated_param_deltas())


def test_rule_based_includes_cut_and_params() -> None:
    result = propose(_state(), _report())
    classes = {d.declared_class for d in result}
    assert DeltaClass.B in classes  # >= 1 analyzer cut
    assert DeltaClass.C in classes  # >= 1 curated param variation


def test_max_deltas_one_emits_a_single_cut() -> None:
    result = propose(_state(), _report(), max_deltas=1)
    assert len(result) == 1
    assert result[0].declared_class is DeltaClass.B


def test_max_deltas_respected() -> None:
    for k in (1, 2, 3, 5):
        assert len(propose(_state(), _report(), max_deltas=k)) <= k


def test_propose_rule_based_standalone() -> None:
    result = propose_rule_based(_state(), _report())
    assert result
    assert any(d.declared_class is DeltaClass.B for d in result)
    assert all(d.declared_class is not DeltaClass.D for d in result)


def test_every_delta_has_a_rationale_target() -> None:
    result = propose(_state(), _report(), max_deltas=99)
    for delta in result:
        assert delta.target
        assert isinstance(delta.target, str)


def test_low_gap_still_emits_a_cut() -> None:
    # Even a tight relaxation keeps >= 1 analyzer cut.
    result = propose(_state(), _report(gap=0.0), max_deltas=5)
    assert any(d.declared_class is DeltaClass.B for d in result)
