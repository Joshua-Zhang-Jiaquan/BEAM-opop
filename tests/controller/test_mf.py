"""Tests for multi-fidelity layers, the ρ-correlation GATE, and cost-aware MFKG.

Covers task 29:

* ``Phi.s`` selects a fidelity layer from the seven-layer ladder.
* :func:`opop.controller.fidelity.fidelity_correlation` computes Spearman ρ
  across methods and gates MFKG at ρ ≥ 0.5 (fail-closed otherwise).
* :class:`opop.controller.fidelity.MFKGController` is always constructible but
  only activates when ρ ≥ 0.5; otherwise it warns and falls back to
  single-fidelity.  The BoTorch MFKG construction is guarded by
  ``pytest.importorskip("botorch")`` and skips cleanly when absent.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from opop.controller.encoder import OrdinalDim, default_phase1_space
from opop.controller.fidelity import (
    FIDELITY_LAYERS,
    FIDELITY_SPECS,
    MFKG_RHO_THRESHOLD,
    FidelityLayer,
    MFKGController,
    fidelity_column,
    fidelity_correlation,
    fidelity_cost,
    fidelity_dim,
    fidelity_phase1_space,
    fidelity_solve,
    layer_for,
    mfkg_available,
    normalized_fidelity,
    resolve_layer,
    should_enable_mfkg,
)
from opop.model.state import Phi

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── fidelity layers ───────────────────────────────────────────────────────────


def test_seven_layers_match_task_spec() -> None:
    """The ladder is exactly the seven task-specified layers, cheapest first."""
    expected = {
        "presolve",
        "lp_relax",
        "root_cuts",
        "short_time",
        "sub_instance",
        "heuristic",
        "full_solve",
    }
    assert {layer.value for layer in FIDELITY_LAYERS} == expected
    assert len(FIDELITY_LAYERS) == 7
    assert FIDELITY_LAYERS[0] is FidelityLayer.PRESOLVE
    assert FIDELITY_LAYERS[-1] is FidelityLayer.FULL_SOLVE


def test_layer_for_indexes_and_clamps() -> None:
    """``Phi.s`` indexes the ladder, clamped into range."""
    assert layer_for(0) is FidelityLayer.PRESOLVE
    assert layer_for(6) is FidelityLayer.FULL_SOLVE
    assert layer_for(99) is FidelityLayer.FULL_SOLVE  # clamp high
    assert layer_for(-5) is FidelityLayer.PRESOLVE  # clamp low
    # The default Phi.s resolves to a valid layer.
    assert layer_for(Phi().s) in FIDELITY_LAYERS


def test_resolve_layer_accepts_enum_name_value_int() -> None:
    """``resolve_layer`` coerces enum / value / member-name / int / bool."""
    assert resolve_layer(FidelityLayer.HEURISTIC) is FidelityLayer.HEURISTIC
    assert resolve_layer("full_solve") is FidelityLayer.FULL_SOLVE
    assert resolve_layer("FULL_SOLVE") is FidelityLayer.FULL_SOLVE
    assert resolve_layer(2) is FidelityLayer.ROOT_CUTS
    assert resolve_layer(True) is FidelityLayer.LP_RELAX  # bool -> int 1
    with pytest.raises(ValueError, match="unknown fidelity layer"):
        resolve_layer("not_a_layer")
    with pytest.raises(TypeError):
        resolve_layer(3.5)


def test_normalized_fidelity_spans_unit_interval() -> None:
    """Normalized fidelity runs 0.0 (presolve) → 1.0 (full_solve), monotone."""
    norms = [normalized_fidelity(layer) for layer in FIDELITY_LAYERS]
    assert norms[0] == 0.0
    assert norms[-1] == 1.0
    assert norms == sorted(norms)
    assert all(0.0 <= n <= 1.0 for n in norms)


def test_fidelity_cost_strictly_increasing() -> None:
    """Evaluation cost increases monotonically with fidelity (full_solve == 1.0)."""
    costs = [fidelity_cost(layer) for layer in FIDELITY_LAYERS]
    assert costs == sorted(costs)
    assert all(b > a for a, b in zip(costs, costs[1:]))
    assert fidelity_cost(FidelityLayer.FULL_SOLVE) == 1.0


def test_fidelity_specs_complete_and_whitelist_safe() -> None:
    """Every layer has a spec; full_solve is unconstrained; no separator knobs."""
    assert set(FIDELITY_SPECS) == set(FIDELITY_LAYERS)
    full = FIDELITY_SPECS[FidelityLayer.FULL_SOLVE]
    assert full.time_fraction == 1.0
    assert full.node_limit is None
    # Reduced layers must never inject ``separating/<name>/...`` knobs (would
    # trip the kernel's class-B separator whitelist).
    for spec in FIDELITY_SPECS.values():
        assert 0.0 < spec.time_fraction <= 1.0
        for key in spec.extra_params:
            assert not key.startswith("separating/")


# ── encoder dimension + fidelity-aware space ─────────────────────────────────


def test_fidelity_dim_is_ordinal_seven_levels() -> None:
    """The fidelity encoder dim mirrors an OrdinalDim over the seven levels."""
    dim = fidelity_dim()
    assert isinstance(dim, OrdinalDim)
    assert dim.field == "s"
    assert dim.levels == tuple(range(7))
    assert dim.width == 1


def test_fidelity_space_appends_s_and_roundtrips() -> None:
    """The fidelity space adds ``s`` last and round-trips every level."""
    space = fidelity_phase1_space()
    col = fidelity_column(space)
    assert col == space.dim - 1  # appended last
    for level in range(7):
        phi = replace(space.base, s=level)
        vec = space.encode(phi)
        assert space.decode(vec).s == level
        assert vec[col] == pytest.approx(normalized_fidelity(layer_for(level)))


def test_default_space_unchanged_has_no_fidelity_dim() -> None:
    """``default_phase1_space`` is NOT mutated (no ``s`` dim) — guards prior tests."""
    default = default_phase1_space()
    fields = {getattr(d, "field", None) for d in default.dims}
    assert "s" not in fields
    with pytest.raises(ValueError, match="no fidelity"):
        fidelity_column(default)


# ── low-fidelity evaluators (reuse the kernel) ───────────────────────────────


def test_fidelity_solve_reuses_kernel_without_mutating_phi(
    solver_skip_if_missing: Callable[[str], None],
) -> None:
    """``fidelity_solve`` drives the real kernel and never mutates ``phi``."""
    solver_skip_if_missing("scip")
    from opop.bench.sources.synthetic import generate_knapsack
    from opop.solver.scip import ScipKernel

    ir = generate_knapsack(8, seed=0)
    kernel = ScipKernel()
    phi = Phi(p={"limits/gap": 0.0}, s=0)  # s=0 -> presolve (lowest fidelity)

    trace = fidelity_solve(kernel, ir, phi, full_time_limit=2.0, seed=0)
    assert trace.instance_id == ir.name
    assert trace.primal_bound_series  # non-empty trajectory
    # phi is immutable input: unchanged after the solve.
    assert phi.p == {"limits/gap": 0.0}
    assert phi.s == 0

    # An explicit layer override solves the full-fidelity problem.
    full = fidelity_solve(kernel, ir, phi, full_time_limit=2.0, seed=0, layer="full_solve")
    assert full.status == "optimal"


# ── fidelity-correlation GATE ─────────────────────────────────────────────────


def _monotone_results() -> dict[str, dict[FidelityLayer, float]]:
    """Four methods whose low- and high-fidelity scores rank identically."""
    return {
        "m1": {FidelityLayer.PRESOLVE: -1.0, FidelityLayer.FULL_SOLVE: -1.1},
        "m2": {FidelityLayer.PRESOLVE: -2.0, FidelityLayer.FULL_SOLVE: -2.2},
        "m3": {FidelityLayer.PRESOLVE: -3.0, FidelityLayer.FULL_SOLVE: -2.9},
        "m4": {FidelityLayer.PRESOLVE: -0.5, FidelityLayer.FULL_SOLVE: -0.4},
    }


def test_correlation_high_rho_enables_mfkg() -> None:
    """Concordant rankings → high ρ → gate opens."""
    report = fidelity_correlation(_monotone_results())
    assert report.rho >= MFKG_RHO_THRESHOLD
    assert report.enable_mfkg is True
    assert report.n_methods == 4
    assert report.low_layer == "presolve"  # cheapest common layer chosen
    assert report.high_layer == "full_solve"
    assert "enable cost-aware MFKG" in report.reason


def test_correlation_anticorrelated_records_negative_result() -> None:
    """Discordant rankings → negative ρ → gate stays closed (recorded)."""
    records = [
        {"method": "a", "fidelity": "short_time", "score": 1.0},
        {"method": "a", "fidelity": "full_solve", "score": 4.0},
        {"method": "b", "fidelity": "short_time", "score": 2.0},
        {"method": "b", "fidelity": "full_solve", "score": 3.0},
        {"method": "c", "fidelity": "short_time", "score": 3.0},
        {"method": "c", "fidelity": "full_solve", "score": 2.0},
        {"method": "d", "fidelity": "short_time", "score": 4.0},
        {"method": "d", "fidelity": "full_solve", "score": 1.0},
    ]
    report = fidelity_correlation(records)
    assert report.rho < 0.5
    assert report.enable_mfkg is False
    assert "keep single-fidelity" in report.reason


def test_correlation_single_method_is_fail_closed() -> None:
    """Fewer than two paired methods → ρ undefined → MFKG disabled."""
    report = fidelity_correlation(
        {"only": {FidelityLayer.PRESOLVE: 1.0, FidelityLayer.FULL_SOLVE: 2.0}}
    )
    assert math.isnan(report.rho)
    assert report.enable_mfkg is False
    assert report.n_methods == 1


def test_correlation_constant_scores_is_fail_closed() -> None:
    """Constant high-fidelity scores → ρ undefined → MFKG disabled."""
    report = fidelity_correlation(
        {
            "m1": {FidelityLayer.PRESOLVE: -1.0, FidelityLayer.FULL_SOLVE: -2.0},
            "m2": {FidelityLayer.PRESOLVE: -2.0, FidelityLayer.FULL_SOLVE: -2.0},
            "m3": {FidelityLayer.PRESOLVE: -3.0, FidelityLayer.FULL_SOLVE: -2.0},
        }
    )
    assert math.isnan(report.rho)
    assert report.enable_mfkg is False


def test_correlation_report_json_roundtrips() -> None:
    """The report serialises to valid JSON carrying the gate decision."""
    report = fidelity_correlation(_monotone_results())
    payload = json.loads(report.to_json())
    assert payload["enable_mfkg"] is True
    assert payload["rho"] == pytest.approx(report.rho)
    assert payload["low_layer"] == "presolve"
    assert set(report.to_dict()) >= {
        "rho",
        "p_value",
        "n_methods",
        "enable_mfkg",
        "threshold",
        "reason",
    }


def test_explicit_low_high_layers_respected() -> None:
    """``low`` / ``high`` overrides select the compared layers."""
    results = {
        "m1": {FidelityLayer.LP_RELAX: -1.0, FidelityLayer.HEURISTIC: -1.0},
        "m2": {FidelityLayer.LP_RELAX: -2.0, FidelityLayer.HEURISTIC: -2.0},
        "m3": {FidelityLayer.LP_RELAX: -3.0, FidelityLayer.HEURISTIC: -3.0},
    }
    report = fidelity_correlation(results, low="lp_relax", high="heuristic")
    assert report.low_layer == "lp_relax"
    assert report.high_layer == "heuristic"
    assert report.n_methods == 3


def test_should_enable_mfkg_threshold_boundary() -> None:
    """Gate is ≥ threshold, fail-closed on nan."""
    assert should_enable_mfkg(0.5) is True
    assert should_enable_mfkg(0.5000001) is True
    assert should_enable_mfkg(0.4999) is False
    assert should_enable_mfkg(float("nan")) is False
    assert should_enable_mfkg(1.0, threshold=1.0) is True


# ── MFKG controller gate (no BoTorch required) ───────────────────────────────


class _RecordingAcq:
    """A stand-in single-fidelity acquisition that records that it was called."""

    def __init__(self) -> None:
        self.called: bool = False

    def __call__(
        self,
        surrogate: object,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        del surrogate, y_best, kappa, seed
        self.called = True
        return np.asarray(X_candidates)[0], 0.0


def test_mfkg_controller_enabled_only_above_threshold() -> None:
    """The controller activates iff ρ is finite and ≥ threshold."""
    space = fidelity_phase1_space()
    assert MFKGController.from_space(space, rho=0.8).enabled is True
    assert MFKGController.from_space(space, rho=0.5).enabled is True
    assert MFKGController.from_space(space, rho=0.49).enabled is False
    assert MFKGController.from_space(space, rho=float("nan")).enabled is False


def test_mfkg_controller_falls_back_and_warns_when_gated_off() -> None:
    """A gated-off controller warns once and delegates to the fallback."""
    fallback = _RecordingAcq()
    ctrl = MFKGController(rho=0.1, dim=5, fidelity_col=4, fallback=fallback)
    assert ctrl.enabled is False
    pool = np.random.default_rng(0).random((6, 5))
    with pytest.warns(UserWarning, match="MFKG gate not met"):
        sel, _ = ctrl(None, pool)
    assert fallback.called is True
    assert np.allclose(sel, pool[0])


def test_mfkg_build_refuses_without_rho_evidence() -> None:
    """Building MFKG without ρ ≥ threshold raises (never enable on no evidence)."""
    ctrl = MFKGController(rho=0.2, dim=5, fidelity_col=4)
    with pytest.raises(RuntimeError, match="refusing to build MFKG"):
        ctrl.build_acquisition(object())


def test_mfkg_from_correlation_inherits_threshold_and_rho() -> None:
    """``from_correlation`` wires the report's ρ + threshold into the gate."""
    space = fidelity_phase1_space()
    report = fidelity_correlation(_monotone_results())
    ctrl = MFKGController.from_correlation(space, report)
    assert ctrl.rho == pytest.approx(report.rho)
    assert ctrl.threshold == report.threshold
    assert ctrl.fidelity_col == fidelity_column(space)
    assert ctrl.dim == space.dim
    assert ctrl.enabled is True


def test_mfkg_enabled_without_botorch_raises_importerror() -> None:
    """When gated on but BoTorch is absent, the MFKG path raises ImportError."""
    if mfkg_available():
        pytest.skip("botorch present; ImportError path not exercised")
    ctrl = MFKGController(rho=0.9, dim=5, fidelity_col=4)
    assert ctrl.enabled is True
    with pytest.raises(ImportError, match="botorch"):
        ctrl.build_acquisition(object())


# ── MFKG BoTorch path (skipped cleanly when BoTorch is absent) ───────────────


def test_mfkg_builds_and_proposes_with_botorch() -> None:
    """With BoTorch installed, MFKG builds and proposes a valid pool member."""
    pytest.importorskip("botorch")
    from opop.controller.botorch_rungs import BoTorchGPSurrogate

    space = fidelity_phase1_space()
    rng = np.random.default_rng(0)
    pool = space.candidate_pool(12, rng)
    y = -np.sum((pool - 0.5) ** 2, axis=1)

    surrogate = BoTorchGPSurrogate()
    surrogate.fit(pool[:8], y[:8])

    ctrl = MFKGController.from_space(
        space, rho=0.9, num_fantasies=4, num_restarts=1, raw_samples=4
    )
    assert ctrl.enabled is True

    acq: Any = ctrl.build_acquisition(surrogate)
    assert acq is not None

    sel, value = ctrl.propose(surrogate, pool)
    assert sel.shape == (space.dim,)
    assert math.isfinite(value)
    assert any(np.allclose(sel, row) for row in pool)


# ── CLI / study driver ────────────────────────────────────────────────────────


def test_run_study_emits_report_and_mfkg_section() -> None:
    """``run_study`` returns a gate report + an mfkg controller section."""
    from opop.config import BudgetConfig, RunConfig
    from opop.eval.fidelity_correlation import run_study

    config = RunConfig(
        name="mf-test",
        split="dev",
        seeds=[0],
        budget=BudgetConfig(trials=1, time_limit_sec=2.0),
    )
    result = run_study(config, n_methods=4, n_instances=1, low="presolve", seed=0)
    payload = result.to_dict()
    assert "rho" in payload
    assert payload["mfkg"]["fidelity_column"] == payload["mfkg"]["encoded_dim"] - 1
    assert payload["solver"] in {"scip", "synthetic"}
    assert payload["n_methods"] == 4
    # enable_mfkg agrees with the controller section.
    assert payload["enable_mfkg"] == payload["mfkg"]["controller_enabled"]
    json.loads(result.to_json())  # serialisable (nan allowed)


def test_cli_main_writes_json(tmp_path: Any) -> None:
    """The CLI writes the gate JSON to --out and a sibling fidelity_correlation.json."""
    from opop.eval.fidelity_correlation import main

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: mf\nsplit: dev\nseeds: [0]\nbudget:\n  trials: 1\n  time_limit_sec: 2\n",
        encoding="utf-8",
    )
    out = tmp_path / "task-29-mfgate.txt"
    code = main(
        ["--config", str(cfg), "--out", str(out), "--n-methods", "4", "--instances", "1"]
    )
    assert code == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "enable_mfkg" in payload
    assert "mfkg" in payload
    assert (out.parent / "fidelity_correlation.json").is_file()
