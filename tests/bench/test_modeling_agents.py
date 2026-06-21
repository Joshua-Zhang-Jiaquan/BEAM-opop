"""Tests for the modeling-agent dataset loader + cleaning path (plan task 35).

Covers: the committed offline NL4Opt / OptiBench fixtures parse into valid IRs
(plain MILP, a MIQP with a quadratic objective, a separable MINLP); malformed
specs raise :class:`~opop.bench.sources.modeling_agents.ModelSpecError`; and the
solver-backed cleaning harness accepts the correct labels and quarantines the
planted wrong label. Solver-backed checks are ``integration`` + SCIP-gated;
nothing needs the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from opop.bench.cleaning import verify_and_clean
from opop.bench.sources.modeling_agents import (
    MODELING_DATASETS,
    ModelSpecError,
    load_modeling_items,
    loads_modeling_items,
)
from opop.model.ir import ObjSense
from opop.model.minlp import NONLINEAR_TERMS_KEY, NonlinearTerm

FIXTURES = Path(__file__).parent / "fixtures" / "modeling"


# ---------------------------------------------------------------------------
# Parsing the committed fixtures into valid IRs (no solver)
# ---------------------------------------------------------------------------
class TestParsing:
    def test_nl4opt_items_are_plain_milp(self) -> None:
        items = {item.id: item for item in load_modeling_items(FIXTURES / "nl4opt.json")}
        assert set(items) == {"nl4opt/production", "nl4opt/diet"}

        production = items["nl4opt/production"]
        assert production.ir.quadratic is None
        assert NONLINEAR_TERMS_KEY not in production.ir.metadata
        assert production.ir.objective.sense is ObjSense.MAXIMIZE
        assert production.ir.n_vars == 2
        assert production.ir.n_constraints == 2
        assert production.labeled_optimum == 36.0
        assert production.source_dataset == "nl4opt"
        assert production.ir.metadata["natural_language"]

        assert items["nl4opt/diet"].ir.objective.sense is ObjSense.MINIMIZE
        assert items["nl4opt/diet"].labeled_optimum == 22.0

    def test_optibench_miqp_has_quadratic_objective(self) -> None:
        items = {item.id: item for item in load_modeling_items(FIXTURES / "optibench.json")}
        ir = items["optibench/portfolio_miqp"].ir
        ext = ir.quadratic
        assert ext is not None
        assert ext.has_objective_terms()
        assert not ext.has_constraint_terms()
        obj_terms = {(t.var1, t.var2): t.coeff for t in ext.objective_terms}
        assert obj_terms == {("x1", "x1"): 1.0, ("x2", "x2"): 1.0}

    def test_optibench_minlp_has_nonlinear_metadata(self) -> None:
        items = {item.id: item for item in load_modeling_items(FIXTURES / "optibench.json")}
        ir = items["optibench/design_minlp"].ir
        assert ir.quadratic is None
        terms = cast("tuple[NonlinearTerm, ...]", ir.metadata[NONLINEAR_TERMS_KEY])
        assert [(t.func, t.var, t.coeff, t.target) for t in terms] == [
            ("square", "x", 1.0, "__objective__")
        ]

    def test_loads_from_text_matches_file(self) -> None:
        text = (FIXTURES / "nl4opt.json").read_text(encoding="utf-8")
        from_text = {item.id for item in loads_modeling_items(text)}
        assert from_text == {"nl4opt/production", "nl4opt/diet"}

    def test_datasets_constant(self) -> None:
        assert MODELING_DATASETS == ("nl4opt", "optibench")


# ---------------------------------------------------------------------------
# Malformed specs raise ModelSpecError (no solver)
# ---------------------------------------------------------------------------
class TestErrors:
    def test_missing_items_raises(self) -> None:
        with pytest.raises(ModelSpecError, match="items"):
            loads_modeling_items('{"datasets": []}')

    def test_unknown_sense_raises(self) -> None:
        text = json.dumps(
            {
                "items": [
                    {
                        "id": "x",
                        "sense": "sideways",
                        "labeled_optimum": 0.0,
                        "model_spec": {"variables": [], "constraints": [], "objective": {}},
                    }
                ]
            }
        )
        with pytest.raises(ModelSpecError, match="unknown sense"):
            loads_modeling_items(text)

    def test_unknown_variable_type_raises(self) -> None:
        text = json.dumps(
            {
                "items": [
                    {
                        "id": "x",
                        "sense": "minimize",
                        "labeled_optimum": 0.0,
                        "model_spec": {
                            "variables": [{"name": "a", "type": "ternary"}],
                            "constraints": [],
                            "objective": {},
                        },
                    }
                ]
            }
        )
        with pytest.raises(ModelSpecError, match="unknown variable type"):
            loads_modeling_items(text)

    def test_bad_quadratic_entry_raises(self) -> None:
        text = json.dumps(
            {
                "items": [
                    {
                        "id": "x",
                        "sense": "minimize",
                        "labeled_optimum": 0.0,
                        "model_spec": {
                            "variables": [{"name": "a", "type": "binary"}],
                            "constraints": [],
                            "objective": {"linear": {}, "quadratic": [["a"]]},
                        },
                    }
                ]
            }
        )
        with pytest.raises(ModelSpecError, match="quadratic entry"):
            loads_modeling_items(text)


# ---------------------------------------------------------------------------
# Solver-backed re-verification (SCIP-gated)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestSolverBackedCleaning:
    def test_clean_labels_accepted(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = load_modeling_items(FIXTURES / "nl4opt.json") + load_modeling_items(
            FIXTURES / "optibench.json"
        )
        report = verify_and_clean(items, solver_name="SCIP", tol=1e-4, time_limit=30.0)
        assert report.quarantined == ()
        assert set(report.clean_ids) == {
            "nl4opt/production",
            "nl4opt/diet",
            "optibench/portfolio_miqp",
            "optibench/design_minlp",
        }
        by_id = {r.id: r for r in report.clean}
        assert by_id["nl4opt/production"].computed == pytest.approx(36.0)
        assert by_id["nl4opt/diet"].computed == pytest.approx(22.0)
        assert by_id["optibench/portfolio_miqp"].computed == pytest.approx(2.0)
        assert by_id["optibench/portfolio_miqp"].problem_type == "MIQP"
        assert by_id["optibench/design_minlp"].computed == pytest.approx(3.0)
        assert by_id["optibench/design_minlp"].problem_type == "MINLP"

    def test_planted_wrong_label_quarantined(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = load_modeling_items(FIXTURES / "planted_wrong.json")
        report = verify_and_clean(items, solver_name="SCIP", time_limit=30.0)
        assert report.clean == ()
        assert report.quarantined_ids == ("optibench/planted_wrong",)
        bad = report.quarantined[0]
        assert bad.computed == pytest.approx(2.0)
        assert bad.labeled == 5.0
        assert "computed=2" in bad.reason
        assert "labeled=5" in bad.reason

    def test_combined_batch_partitions_and_report_schema(
        self, tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = (
            load_modeling_items(FIXTURES / "nl4opt.json")
            + load_modeling_items(FIXTURES / "optibench.json")
            + load_modeling_items(FIXTURES / "planted_wrong.json")
        )
        report = verify_and_clean(items, solver_name="SCIP", time_limit=30.0)
        assert len(report.clean) == 4
        assert report.quarantined_ids == ("optibench/planted_wrong",)

        out = report.to_json(tmp_path / "cleaning_report.json")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert set(loaded) == {
            "solver_name",
            "tol",
            "time_limit",
            "n_items",
            "n_clean",
            "n_quarantined",
            "clean",
            "quarantined",
        }
        assert loaded["n_items"] == 5
        assert loaded["n_clean"] == 4
        assert loaded["n_quarantined"] == 1
