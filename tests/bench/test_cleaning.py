"""Tests for the solver-backed re-verification / cleaning harness (plan task 35).

Covers: a planted wrong label is quarantined with computed-vs-labeled values; a
correct label is accepted as clean; a declared-sense defect quarantines without a
solver; and the :class:`~opop.bench.cleaning.CleaningReport` JSON schema is stable.
Solver-backed checks are ``integration`` + SCIP-gated so the suite stays green
without a backend; nothing here needs the network.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from opop.bench.cleaning import (
    STATUS_CLEAN,
    STATUS_QUARANTINED,
    CleaningItem,
    CleaningReport,
    CleaningResult,
    verify_and_clean,
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

FIXTURES = Path(__file__).parent / "fixtures" / "cleaning"

_VTYPES = {
    "binary": VarType.BINARY,
    "integer": VarType.INTEGER,
    "continuous": VarType.CONTINUOUS,
}


# ---------------------------------------------------------------------------
# Test-local JSON -> CleaningItem builder (a real dataset loader is task-35 ch.3)
# ---------------------------------------------------------------------------
def _build_milp(spec: dict[str, Any]) -> MILP:
    """Build a plain linear :class:`MILP` from a tiny JSON model spec."""
    sense = (
        ObjSense.MAXIMIZE
        if str(spec["sense"]).lower().startswith("max")
        else ObjSense.MINIMIZE
    )
    variables: list[Variable] = []
    for raw in spec.get("variables", []):
        vtype = _VTYPES[str(raw["type"])]
        default_upper = 1.0 if vtype is VarType.BINARY else math.inf
        variables.append(
            Variable(
                name=str(raw["name"]),
                vtype=vtype,
                lower=float(raw.get("lower", 0.0)),
                upper=float(raw.get("upper", default_upper)),
            )
        )
    constraints: list[LinearConstraint] = []
    for raw in spec.get("constraints", []):
        coeffs = {str(k): float(v) for k, v in raw["coeffs"].items()}
        constraints.append(
            LinearConstraint(
                name=str(raw["name"]),
                coeffs=coeffs,
                sense=ConstraintSense(str(raw["sense"])),
                rhs=float(raw["rhs"]),
            )
        )
    obj_coeffs = {str(k): float(v) for k, v in spec.get("objective", {}).items()}
    objective = Objective(coeffs=obj_coeffs, sense=sense, offset=float(spec.get("offset", 0.0)))
    return MILP(
        name=str(spec["id"]),
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective=objective,
    )


def _load_items(path: Path) -> list[CleaningItem]:
    """Load labeled items from a tiny JSON fixture into :class:`CleaningItem`s."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    items: list[CleaningItem] = []
    for spec in data["items"]:
        ir = _build_milp(spec)
        items.append(
            CleaningItem(
                id=str(spec["id"]),
                ir=ir,
                labeled_optimum=float(spec["labeled_optimum"]),
                sense=ir.objective.sense,
                source_dataset=str(spec.get("source_dataset", "fixture")),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Pure (no-solver) report schema
# ---------------------------------------------------------------------------
class TestReportSchema:
    def _report(self) -> CleaningReport:
        clean = (
            CleaningResult(
                id="ok",
                status=STATUS_CLEAN,
                computed=6.0,
                labeled=6.0,
                sense="maximize",
                solver_status="optimal",
                problem_type="MILP",
                source_dataset="fixture",
                reason="computed=6 matches labeled=6 within tol=0.0001",
            ),
        )
        quarantined = (
            CleaningResult(
                id="bad",
                status=STATUS_QUARANTINED,
                computed=2.0,
                labeled=0.0,
                sense="minimize",
                solver_status="optimal",
                problem_type="MILP",
                source_dataset="fixture",
                reason="objective mismatch: computed=2 vs labeled=0 (|delta|=2 > tol=0.0001)",
            ),
        )
        return CleaningReport(
            clean=clean,
            quarantined=quarantined,
            solver_name="SCIP",
            tol=1e-4,
            time_limit=60.0,
        )

    def test_to_dict_schema(self) -> None:
        report = self._report()
        data = report.to_dict()
        assert set(data) == {
            "solver_name",
            "tol",
            "time_limit",
            "n_items",
            "n_clean",
            "n_quarantined",
            "clean",
            "quarantined",
        }
        assert data["n_items"] == 2
        assert data["n_clean"] == 1
        assert data["n_quarantined"] == 1
        item_keys = {
            "id",
            "status",
            "computed",
            "labeled",
            "sense",
            "solver_status",
            "problem_type",
            "source_dataset",
            "reason",
        }
        assert set(report.clean[0].to_dict()) == item_keys
        assert set(report.quarantined[0].to_dict()) == item_keys
        assert isinstance(data["clean"], list)
        assert isinstance(data["quarantined"], list)

    def test_to_json_roundtrips(self, tmp_path: Path) -> None:
        report = self._report()
        out = report.to_json(tmp_path / "cleaning_report.json")
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == report.to_dict()
        assert loaded["quarantined"][0]["computed"] == 2.0
        assert loaded["quarantined"][0]["labeled"] == 0.0

    def test_report_id_helpers(self) -> None:
        report = self._report()
        assert report.clean_ids == ("ok",)
        assert report.quarantined_ids == ("bad",)
        assert report.n_items == 2


# ---------------------------------------------------------------------------
# Sense-defect quarantine (no solver needed)
# ---------------------------------------------------------------------------
class TestSenseDefect:
    def test_declared_sense_mismatch_is_quarantined(self) -> None:
        ir = MILP(
            name="mini",
            variables=(Variable("x", VarType.BINARY, 0.0, 1.0),),
            constraints=(),
            objective=Objective(coeffs={"x": 1.0}, sense=ObjSense.MINIMIZE),
        )
        # Declared MAXIMIZE but the model minimises -> integrity defect.
        item = CleaningItem(id="sense/bad", ir=ir, labeled_optimum=0.0, sense=ObjSense.MAXIMIZE)
        report = verify_and_clean([item], solver_name="SCIP", time_limit=10.0)
        assert report.clean_ids == ()
        assert report.quarantined_ids == ("sense/bad",)
        assert "does not match model objective sense" in report.quarantined[0].reason
        # No solver was invoked, so this verdict holds even without a backend.
        assert report.quarantined[0].solver_status == "not_solved"


# ---------------------------------------------------------------------------
# Solver-backed re-verification (SCIP-gated)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestSolverBackedCleaning:
    def test_cleaning_accepts_clean_label(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = _load_items(FIXTURES / "clean_items.json")
        report = verify_and_clean(items, solver_name="SCIP", tol=1e-4, time_limit=30.0)

        assert report.quarantined == ()
        assert set(report.clean_ids) == {"clean/knapsack_max", "clean/cover_min"}
        by_id = {r.id: r for r in report.clean}
        assert by_id["clean/knapsack_max"].computed == pytest.approx(6.0)
        assert by_id["clean/cover_min"].computed == pytest.approx(3.0)
        for result in report.clean:
            assert result.status == STATUS_CLEAN
            assert result.solver_status == "optimal"
            assert result.problem_type == "MILP"

    def test_cleaning_quarantines_wrong_label(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = _load_items(FIXTURES / "wrong_label.json")
        report = verify_and_clean(items, solver_name="SCIP", tol=1e-4, time_limit=30.0)

        assert report.clean == ()
        assert report.quarantined_ids == ("planted/wrong_min",)
        bad = report.quarantined[0]
        assert bad.status == STATUS_QUARANTINED
        assert bad.computed == pytest.approx(2.0)
        assert bad.labeled == 0.0
        assert "computed=2" in bad.reason
        assert "labeled=0" in bad.reason

    def test_mixed_batch_partitions(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = _load_items(FIXTURES / "clean_items.json") + _load_items(
            FIXTURES / "wrong_label.json"
        )
        report = verify_and_clean(items, solver_name="SCIP", time_limit=30.0)
        assert set(report.clean_ids) == {"clean/knapsack_max", "clean/cover_min"}
        assert report.quarantined_ids == ("planted/wrong_min",)
        assert report.n_items == 3

    def test_report_json_written(
        self, tmp_path: Path, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = _load_items(FIXTURES / "clean_items.json") + _load_items(
            FIXTURES / "wrong_label.json"
        )
        report = verify_and_clean(items, solver_name="SCIP", time_limit=30.0)
        out = report.to_json(tmp_path / "cleaning_report.json")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["n_clean"] == 2
        assert loaded["n_quarantined"] == 1
        assert {r["id"] for r in loaded["quarantined"]} == {"planted/wrong_min"}
