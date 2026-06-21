"""Phase-1 restricted proposer for OPOP.

Given a :class:`opop.model.state.ProblemState` and an
:class:`opop.analyzer.report.AnalysisReport`, :func:`propose` returns a small set
of typed :class:`opop.model.state.Delta` objects restricted to the Phase-1 design
space — curated SCIP search params (class C), analyzer-flagged valid-inequality
candidates (class B), and a dormant decomposition-flag stub (class C). Selection
is LLM-guided (the LLM picks from typed templates only) with a deterministic
rule-based fallback; a safety envelope guarantees the output is always a typed
subset of the legal candidate pool and never contains a class-D delta.
"""

from __future__ import annotations

from opop.proposer.api import build_candidate_pool, propose
from opop.proposer.families import (
    FAMILIES,
    FormulationFamily,
    Reformulation,
    build_tsp_mcf,
    build_tsp_mtz,
    build_tsp_scf,
    cutset_inequalities,
    encoding_relabel_delta,
    family_deltas,
    mtz_to_flow_reformulation,
)
from opop.proposer.llm_proposer import select
from opop.proposer.params import (
    CURATED_PARAMS,
    DECOMP_PARAM_KEY,
    OP_SET_PARAM,
    ParamKnob,
    curated_param_deltas,
    decomposition_flag_delta,
    make_param_delta,
    param_from_delta,
)
from opop.proposer.rule_based import propose_rule_based, rank
from opop.proposer.stages import (
    ALL_KINDS,
    KIND_CUT,
    KIND_DECOMPOSITION,
    KIND_FORMULATION,
    KIND_HEURISTIC,
    KIND_MULTIKERNEL,
    KIND_PARAM,
    Stage,
    allowed_kinds,
    delta_kind,
    parse_stage,
    stage_allows,
    stage_filter,
    stage_space,
)
from opop.proposer.templates import cut_deltas_from_report

__all__ = [
    "ALL_KINDS",
    "CURATED_PARAMS",
    "DECOMP_PARAM_KEY",
    "FAMILIES",
    "KIND_CUT",
    "KIND_DECOMPOSITION",
    "KIND_FORMULATION",
    "KIND_HEURISTIC",
    "KIND_MULTIKERNEL",
    "KIND_PARAM",
    "OP_SET_PARAM",
    "FormulationFamily",
    "ParamKnob",
    "Reformulation",
    "Stage",
    "allowed_kinds",
    "build_candidate_pool",
    "build_tsp_mcf",
    "build_tsp_mtz",
    "build_tsp_scf",
    "curated_param_deltas",
    "cut_deltas_from_report",
    "cutset_inequalities",
    "decomposition_flag_delta",
    "delta_kind",
    "encoding_relabel_delta",
    "family_deltas",
    "make_param_delta",
    "mtz_to_flow_reformulation",
    "param_from_delta",
    "parse_stage",
    "propose",
    "propose_rule_based",
    "rank",
    "select",
    "stage_allows",
    "stage_filter",
    "stage_space",
]
