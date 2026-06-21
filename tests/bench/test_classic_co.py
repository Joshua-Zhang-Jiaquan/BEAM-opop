"""Tests for the classic CO benchmark loaders (plan task 34).

Covers every family (TSP / CVRP / OR-Library set covering / JSP / MaxSAT /
MaxCut): each loader parses its committed fixtures into a valid ``MILP`` (or
QUBO-shaped) IR; a tiny TSPLIB instance solves to the known optimum; malformed
files raise :class:`~opop.bench.classic.base.ParseError` with file + line
context; the ``ClassicAdapter`` plugins satisfy the
:class:`~opop.model.adapter.ProblemClassAdapter` Protocol and dispatch by the
``co_family`` tag; the committed fixtures are content-locked by the catalog
checksums; and the six families land in the combined registry as ``phase=6`` /
``thesis=T3`` held-out entries. Solver-backed checks are SCIP-gated so the suite
stays green without a backend; nothing needs the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from opop.bench.classic import (
    FAMILIES,
    adapter_for,
    load_instance,
    loads_instance,
)
from opop.bench.classic.base import ParseError
from opop.bench.classic.catalog import (
    CLASSIC_FAMILIES,
    CLASSIC_FIXTURES,
    build_classic_entries,
    family_checksum,
)
from opop.bench.registry import HELD_SPLITS, BenchmarkRegistry
from opop.bench.sources.milp_suites import REGISTRY_PATH, build_all_entries
from opop.model.adapter import ProblemClassAdapter, find_adapter, get_adapter
from opop.model.ir import MILP, ObjSense

FIXTURES = Path(__file__).parent / "fixtures" / "classic"

#: (family, fixture relative path) for every committed valid instance.
VALID_CASES: list[tuple[str, str]] = [
    (fx.family, fx.rel_path) for fx in CLASSIC_FIXTURES
]


def _is_valid_ir(ir: MILP) -> bool:
    """A produced IR is usable iff it has variables and linear or quadratic rows."""
    has_quadratic = ir.quadratic is not None and not ir.quadratic.is_empty
    return ir.n_vars > 0 and (ir.n_constraints > 0 or has_quadratic)


# ---------------------------------------------------------------------------
# Each loader parses >= 2 instances into a valid IR
# ---------------------------------------------------------------------------
class TestLoadersProduceValidIR:
    @pytest.mark.parametrize(("family", "rel_path"), VALID_CASES)
    def test_fixture_loads_to_valid_ir(self, family: str, rel_path: str) -> None:
        ir = load_instance(family, str(FIXTURES / rel_path))
        assert _is_valid_ir(ir)
        assert ir.metadata.get("co_family") == family

    def test_every_family_has_at_least_two_instances(self) -> None:
        for fam in CLASSIC_FAMILIES:
            assert len(fam.fixtures) >= 2

    def test_all_six_families_present(self) -> None:
        assert set(FAMILIES) == {"tsp", "cvrp", "orlib", "jsp", "maxsat", "maxcut"}

    def test_maxcut_is_qubo_shaped(self) -> None:
        ir = load_instance("maxcut", str(FIXTURES / "maxcut/triangle.txt"))
        assert ir.quadratic is not None
        assert ir.quadratic.has_objective_terms()
        assert not ir.quadratic.has_constraint_terms()

    def test_loads_instance_matches_load_instance(self) -> None:
        path = FIXTURES / "orlib/scp_tiny.txt"
        from_file = load_instance("orlib", str(path))
        from_text = loads_instance("orlib", path.read_text(encoding="utf-8"), name="scp_tiny")
        assert from_file.n_vars == from_text.n_vars
        assert from_file.n_constraints == from_text.n_constraints


# ---------------------------------------------------------------------------
# A tiny TSPLIB instance solves to the known optimum (SCIP-gated)
# ---------------------------------------------------------------------------
class TestKnownOptima:
    @pytest.mark.parametrize(
        ("rel_path", "optimum"),
        [("tsp/tiny4.tsp", 40.0), ("tsp/explicit4.tsp", 24.0)],
    )
    def test_tsp_optimum(
        self, rel_path: str, optimum: float, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("scip")
        from opop.solver.scip import ScipKernel

        ir = load_instance("tsp", str(FIXTURES / rel_path))
        trace = adapter_for("tsp").native_solve(ir, ScipKernel(), time_limit=30.0, seed=0)
        assert trace.status == "optimal"
        assert trace.primal_bound_series[-1] == pytest.approx(optimum)

    @pytest.mark.parametrize(
        ("rel_path", "cut_weight"),
        [("maxcut/triangle.txt", 2.0), ("maxcut/square.txt", 4.0)],
    )
    def test_maxcut_cut_weight(
        self, rel_path: str, cut_weight: float, solver_skip_if_missing: Callable[[str], None]
    ) -> None:
        solver_skip_if_missing("scip")
        from opop.solver.scip import ScipKernel

        ir = load_instance("maxcut", str(FIXTURES / rel_path))
        # QUBO is minimised; the cut weight is the negated minimum energy.
        trace = adapter_for("maxcut").native_solve(ir, ScipKernel(), time_limit=30.0, seed=0)
        assert trace.status == "optimal"
        assert -trace.primal_bound_series[-1] == pytest.approx(cut_weight)


# ---------------------------------------------------------------------------
# Malformed files raise ParseError with file + line context
# ---------------------------------------------------------------------------
class TestParseErrors:
    def test_truncated_tsp_file_has_path_and_line(self) -> None:
        path = FIXTURES / "tsp" / "truncated.tsp"
        with pytest.raises(ParseError) as excinfo:
            load_instance("tsp", str(path))
        err = excinfo.value
        assert err.source == str(path)
        assert err.line is not None
        assert str(path) in str(err)
        assert "expected 4 coordinates" in str(err)

    @pytest.mark.parametrize(
        ("family", "text", "needle"),
        [
            ("tsp", "TYPE : TSP\nEDGE_WEIGHT_TYPE : EUC_2D\n", "DIMENSION"),
            (
                "tsp",
                "DIMENSION : 3\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n1 0 0\nEOF\n",
                "expected 3 coordinates",
            ),
            (
                "cvrp",
                "DIMENSION : 3\nEDGE_WEIGHT_TYPE : EUC_2D\n"
                + "NODE_COORD_SECTION\n1 0 0\n2 1 0\n3 0 1\n"
                + "DEMAND_SECTION\n1 0\n2 1\n3 1\nDEPOT_SECTION\n1\n-1\nEOF\n",
                "CAPACITY",
            ),
            ("orlib", "3 4\n1 1 1 1\n2 1 9\n2 2 3\n2 3 4\n", "out of range"),
            ("jsp", "2 2\n0 3 1 2\n5 4 0 1\n", "out of range"),
            ("maxsat", "p wcnf 2 1 5\n5 1 9 0\n", "out of range"),
            ("maxcut", "3 2\n1 2 1\n1 9 1\n", "out of range"),
            ("maxcut", "2 1\n1 1 1\n", "self-loop"),
        ],
    )
    def test_inline_malformed_raises_with_context(
        self, family: str, text: str, needle: str
    ) -> None:
        with pytest.raises(ParseError) as excinfo:
            loads_instance(family, text, name="bad", source="<bad>")
        err = excinfo.value
        assert err.source == "<bad>"
        assert needle in str(err)
        assert "<bad>" in str(err)

    def test_parse_error_is_value_error(self) -> None:
        with pytest.raises(ValueError):
            loads_instance("maxcut", "2 1\n1 1 1\n", source="<bad>")


# ---------------------------------------------------------------------------
# ClassicAdapter satisfies the ProblemClassAdapter Protocol + dispatch
# ---------------------------------------------------------------------------
class TestAdapters:
    def test_all_adapters_registered_and_protocol(self) -> None:
        for family in FAMILIES:
            adapter = adapter_for(family)
            assert isinstance(adapter, ProblemClassAdapter)
            assert adapter.name == f"classic-{family}"
            assert get_adapter(f"classic-{family}") is adapter
            assert adapter.capabilities.exact_linearization

    def test_can_handle_dispatches_on_co_family(self) -> None:
        tsp_ir = load_instance("tsp", str(FIXTURES / "tsp/tiny4.tsp"))
        assert adapter_for("tsp").can_handle(tsp_ir)
        assert not adapter_for("cvrp").can_handle(tsp_ir)

    def test_find_adapter_routes_linear_family(self) -> None:
        tsp_ir = load_instance("tsp", str(FIXTURES / "tsp/tiny4.tsp"))
        found = find_adapter(tsp_ir)
        assert found is not None
        assert found.name == "classic-tsp"

    def test_to_milp_is_linear(self) -> None:
        for family in FAMILIES:
            rel = CLASSIC_FAMILIES[FAMILIES.index(family)].fixtures[0].rel_path
            ir = load_instance(family, str(FIXTURES / rel))
            milp = adapter_for(family).to_milp(ir)
            assert milp.quadratic is None or milp.quadratic.is_empty

    def test_maxcut_to_milp_adds_edge_variables(self) -> None:
        ir = load_instance("maxcut", str(FIXTURES / "maxcut/square.txt"))
        milp = adapter_for("maxcut").to_milp(ir)
        assert milp.quadratic is None
        assert milp.n_vars > ir.n_vars  # Fortet edge variables were introduced.
        assert milp.objective.sense is ObjSense.MINIMIZE


# ---------------------------------------------------------------------------
# Committed fixtures are content-locked by the catalog checksums
# ---------------------------------------------------------------------------
class TestFixtureChecksums:
    def test_committed_fixture_hashes_match_catalog(self) -> None:
        for fx in CLASSIC_FIXTURES:
            path = FIXTURES / fx.rel_path
            assert path.is_file(), f"missing fixture {fx.rel_path}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == fx.sha256, f"fixture drift: {fx.rel_path}"

    def test_family_checksum_matches_registry_entries(self) -> None:
        entries = {e.name: e for e in build_classic_entries()}
        for fam in CLASSIC_FAMILIES:
            assert entries[fam.entry_name].checksum == family_checksum(fam.fixtures)


# ---------------------------------------------------------------------------
# Classic families are registered as held-out (phase 6 / thesis T3) entries
# ---------------------------------------------------------------------------
class TestRegistryIntegration:
    def test_entries_are_phase6_thesis_t3_held_out(self) -> None:
        for entry in build_classic_entries():
            assert entry.phase == 6
            assert entry.thesis == "T3"
            assert set(entry.split).issubset(HELD_SPLITS)
            assert not entry.split.get("dev")
            assert not entry.split.get("validation")
            assert entry.leakage_group == entry.name

    def test_classic_entries_in_committed_registry(self) -> None:
        loaded = {e.name: e for e in BenchmarkRegistry.from_yaml(REGISTRY_PATH).entries}
        for entry in build_classic_entries():
            assert entry.name in loaded
            assert loaded[entry.name] == entry

    def test_classic_entries_in_build_all_entries(self) -> None:
        names = {e.name for e in build_all_entries()}
        assert {e.name for e in build_classic_entries()} <= names

    def test_instance_ids_globally_unique_and_namespaced(self) -> None:
        ids = [fx.id for fx in CLASSIC_FIXTURES]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("classic/") for i in ids)
