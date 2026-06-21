"""Tests for the minimal QPLIB reader + MIQP/MIQCP cleaning path (plan task 35).

Covers: the committed offline fixtures parse into valid quadratic IRs (a MIQCP
with a quadratic constraint, a MIQP with a quadratic objective); both are claimed
by the task-30 :class:`~opop.solver.miqp.MiqpAdapter`; malformed input raises
:class:`~opop.bench.sources.qplib.QplibParseError` with file/line context; and the
solver-backed cleaning harness re-derives each fixture's reference optimum via
SCIP. Solver-backed checks are ``integration`` + SCIP-gated; nothing needs the
network.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from opop.bench.cleaning import verify_and_clean
from opop.bench.sources.qplib import (
    QPLIB_FIXTURES,
    QplibParseError,
    load_qplib,
    load_qplib_items,
    loads_qplib,
)
from opop.model.adapter import find_adapter
from opop.model.ir import ConstraintSense, ObjSense, VarType
from opop.solver.miqp import MiqpAdapter

FIXTURES = Path(__file__).parent / "fixtures" / "qplib"


# ---------------------------------------------------------------------------
# Parsing the committed fixtures into valid quadratic IRs
# ---------------------------------------------------------------------------
class TestParsing:
    def test_miqcp_loads_to_quadratic_constraint_ir(self) -> None:
        ir = load_qplib(FIXTURES / "ball_miqcp.qplib")
        assert ir.n_vars == 2
        assert all(v.vtype is VarType.INTEGER for v in ir.variables)
        assert all((v.lower, v.upper) == (0.0, 5.0) for v in ir.variables)
        assert ir.objective.sense is ObjSense.MAXIMIZE
        assert ir.objective.coeffs == {"x1": 1.0, "x2": 2.0}

        ext = ir.quadratic
        assert ext is not None
        assert not ext.has_objective_terms()
        assert ext.has_constraint_terms()
        terms = {(t.var1, t.var2): t.coeff for t in ext.constraint_terms["c1"]}
        assert terms == {("x1", "x1"): 1.0, ("x2", "x2"): 1.0}

        (con,) = ir.constraints
        assert con.sense is ConstraintSense.LE
        assert con.rhs == 10.0
        assert con.coeffs == {}

    def test_miqp_loads_with_quadratic_objective(self) -> None:
        ir = load_qplib(FIXTURES / "box_miqp.qplib")
        assert ir.n_vars == 2
        assert all(v.vtype is VarType.INTEGER for v in ir.variables)
        assert all((v.lower, v.upper) == (0.0, 3.0) for v in ir.variables)
        assert ir.objective.sense is ObjSense.MINIMIZE
        assert ir.objective.coeffs == {"x1": -4.0, "x2": -4.0}
        assert ir.objective.offset == 8.0

        ext = ir.quadratic
        assert ext is not None
        assert ext.has_objective_terms()
        assert not ext.has_constraint_terms()
        obj_terms = {(t.var1, t.var2): t.coeff for t in ext.objective_terms}
        assert obj_terms == {("x1", "x1"): 1.0, ("x2", "x2"): 1.0}

        (con,) = ir.constraints
        assert con.coeffs == {"x1": 1.0, "x2": 1.0}
        assert con.sense is ConstraintSense.LE
        assert con.rhs == 3.0

    def test_metadata_records_provenance(self) -> None:
        ir = load_qplib(FIXTURES / "ball_miqcp.qplib")
        assert ir.metadata["source"] == "qplib"
        assert ir.metadata["qplib_type"] == "LQI"
        assert ir.metadata["qplib_name"] == "opop_ball_miqcp"


# ---------------------------------------------------------------------------
# MIQP/MIQCP adapter claims the fixtures (no solver needed)
# ---------------------------------------------------------------------------
class TestAdapterDispatch:
    @pytest.mark.parametrize("filename", ["ball_miqcp.qplib", "box_miqp.qplib"])
    def test_fixture_claimed_by_miqp_adapter(self, filename: str) -> None:
        ir = load_qplib(FIXTURES / filename)
        adapter = find_adapter(ir)
        assert adapter is not None
        assert adapter.name == "miqp"
        assert isinstance(adapter, MiqpAdapter)
        assert "SCIP" in adapter.capabilities.native_kernels


# ---------------------------------------------------------------------------
# Malformed input raises QplibParseError with context
# ---------------------------------------------------------------------------
class TestParseErrors:
    def test_bad_type_code_raises(self) -> None:
        with pytest.raises(QplibParseError, match="3 characters"):
            loads_qplib("bad\nQQ\nMinimize\n1\n", source="<bad>")

    def test_truncated_file_raises_with_context(self) -> None:
        with pytest.raises(QplibParseError) as excinfo:
            loads_qplib("trunc\nQLI\nMinimize\n", source="<trunc>")
        err = excinfo.value
        assert err.source == "<trunc>"
        assert "number of variables" in str(err)

    def test_two_sided_range_constraint_raises(self) -> None:
        text = (
            "rng\nLLI\nMinimize\n1\n1\n"  # name/type/sense/n/m
            "0.0\n0\n0.0\n"  # objective: default b0, nnz b0, q0
            "1\n1 1 1.0\n"  # linear A: nnz, entry (con1, x1, 1)
            "1.0\n0\n5.0\n0\n"  # c_l default 1, c_u default 5 -> range
            "0.0\n0\n3.0\n0\n"  # variable bounds
        )
        with pytest.raises(QplibParseError, match="two-sided range"):
            loads_qplib(text, source="<rng>")

    def test_mixed_integer_type_char_unsupported(self) -> None:
        with pytest.raises(QplibParseError, match="not supported"):
            loads_qplib("mix\nQLM\nMinimize\n1\n", source="<mix>")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
class TestCatalog:
    def test_fixtures_exist_and_are_namespaced(self) -> None:
        assert len(QPLIB_FIXTURES) >= 1
        for fx in QPLIB_FIXTURES:
            assert (FIXTURES / fx.filename).is_file()
        names = [fx.name for fx in QPLIB_FIXTURES]
        assert len(names) == len(set(names))

    def test_load_items_are_labeled(self) -> None:
        items = {item.id: item for item in load_qplib_items(FIXTURES)}
        assert set(items) == {"qplib/ball_miqcp", "qplib/box_miqp"}
        assert items["qplib/ball_miqcp"].labeled_optimum == 7.0
        assert items["qplib/box_miqp"].labeled_optimum == 1.0
        assert items["qplib/ball_miqcp"].sense is ObjSense.MAXIMIZE
        assert all(item.source_dataset == "qplib" for item in items.values())


# ---------------------------------------------------------------------------
# Solver-backed re-verification to the reference optimum (SCIP-gated)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestSolverBackedReference:
    def test_miqcp_solves_to_reference(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        items = [i for i in load_qplib_items(FIXTURES) if i.id == "qplib/ball_miqcp"]
        report = verify_and_clean(items, solver_name="SCIP", tol=1e-4, time_limit=30.0)
        assert report.quarantined == ()
        assert report.clean_ids == ("qplib/ball_miqcp",)
        result = report.clean[0]
        assert result.problem_type == "MIQCP"
        assert result.computed == pytest.approx(7.0)
        assert result.solver_status == "optimal"

    def test_all_fixtures_clean_against_reference(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        report = verify_and_clean(load_qplib_items(FIXTURES), solver_name="SCIP", time_limit=30.0)
        assert report.quarantined == ()
        assert set(report.clean_ids) == {"qplib/ball_miqcp", "qplib/box_miqp"}
        by_id = {r.id: r for r in report.clean}
        assert by_id["qplib/ball_miqcp"].computed == pytest.approx(7.0)
        assert by_id["qplib/box_miqp"].computed == pytest.approx(1.0)

    def test_wrong_reference_is_quarantined(
        self, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("SCIP")
        import dataclasses

        (item,) = [i for i in load_qplib_items(FIXTURES) if i.id == "qplib/ball_miqcp"]
        tampered = dataclasses.replace(item, labeled_optimum=99.0)
        report = verify_and_clean([tampered], solver_name="SCIP", time_limit=30.0)
        assert report.clean == ()
        assert report.quarantined_ids == ("qplib/ball_miqcp",)
        assert report.quarantined[0].computed == pytest.approx(7.0)
        assert "labeled=99" in report.quarantined[0].reason
