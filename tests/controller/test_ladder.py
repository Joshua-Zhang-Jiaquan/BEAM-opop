"""Tests for the Wave-4 controller ladder (SMAC/TPE/RF + structured BO router)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from opop.controller.encoder import (
    BoolDim,
    CategoricalDim,
    ContinuousDim,
    OrdinalDim,
    Phase1Space,
    default_phase1_space,
)
from opop.controller.ladder import (
    HIGH_DIM_THRESHOLD,
    BOCSSurrogate,
    COMBOSurrogate,
    DictionaryEmbeddingSurrogate,
    LadderEI,
    MixedGPSurrogate,
    RandomForestSurrogate,
    TPESurrogate,
    analyze_space,
    select_surrogate,
)
from opop.controller.phase1 import Phase1Controller
from opop.controller.protocol import Surrogate
from opop.model.state import Phi

# ── space builders for the router (router reads dim TYPES only; never encodes) ─


def _boolean_space(n: int = 4) -> Phase1Space:
    fields = ("c", "d", "m", "v", "s", "h", "p", "rho")
    dims = tuple(BoolDim(fields[i]) for i in range(n))
    return Phase1Space(base=Phi(), dims=dims)


def _pure_discrete_space() -> Phase1Space:
    return Phase1Space(
        base=Phi(),
        dims=(
            BoolDim("c"),
            CategoricalDim("d", ["none", "benders", "dw"]),
            OrdinalDim("h", [0, 1, 2]),
        ),
    )


def _high_dim_discrete_space(n: int = 25) -> Phase1Space:
    dims = tuple(OrdinalDim("h", [0, 1, 2]) for _ in range(n))
    return Phase1Space(base=Phi(), dims=dims)


def _continuous_space() -> Phase1Space:
    return Phase1Space(
        base=Phi(),
        dims=(ContinuousDim("h", 0.0, 1.0), ContinuousDim("s", 0.0, 1.0)),
    )


# ── router (select_surrogate) ────────────────────────────────────────────────


def test_router_boolean_selects_bocs() -> None:
    choice = select_surrogate(_boolean_space(4), budget=30, noise=False)
    assert choice.name == "bocs"
    assert choice.reason


def test_router_mixed_selects_mixed_gp() -> None:
    choice = select_surrogate(default_phase1_space(), budget=30, noise=False)
    assert choice.name == "mixed_gp"


def test_router_high_dim_discrete_selects_dictionary_embedding() -> None:
    space = _high_dim_discrete_space(25)
    assert space.dim >= HIGH_DIM_THRESHOLD
    choice = select_surrogate(space, budget=30, noise=False)
    assert choice.name == "dictionary_embedding"


def test_router_pure_discrete_low_dim_selects_combo() -> None:
    choice = select_surrogate(_pure_discrete_space(), budget=30, noise=False)
    assert choice.name == "combo"


def test_router_default_is_qlognei() -> None:
    choice = select_surrogate(_continuous_space(), budget=30, noise=False)
    assert choice.name == "qlognei"
    assert choice.requires == ("botorch",)


def test_router_tiny_budget_falls_back_to_tpe() -> None:
    choice = select_surrogate(_boolean_space(4), budget=3, noise=False)
    assert choice.name == "tpe"


def test_router_tiny_budget_noisy_falls_back_to_random_forest() -> None:
    choice = select_surrogate(_boolean_space(4), budget=3, noise=True)
    assert choice.name == "random_forest"


def test_analyze_space_shape_flags() -> None:
    boolean = analyze_space(_boolean_space(4))
    assert boolean.is_boolean and boolean.is_pure_discrete and not boolean.is_mixed
    assert boolean.dim == 4

    mixed = analyze_space(default_phase1_space())
    assert mixed.is_mixed and not mixed.is_pure_discrete
    assert len(mixed.continuous_cols) > 0 and len(mixed.discrete_cols) > 0

    high = analyze_space(_high_dim_discrete_space(25))
    assert high.is_pure_discrete and high.is_high_dim and not high.is_boolean


# ── toy objective driver (finite-pool BO, no Phi/encoder dependency) ──────────


def _bowl(target: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], float]:
    t = np.asarray(target, dtype=np.float64)

    def f(x: NDArray[np.float64]) -> float:
        return -float(np.sum((np.asarray(x, dtype=np.float64) - t) ** 2))

    return f


def _argmatch(pool: NDArray[np.float64], x: NDArray[np.float64]) -> int:
    diffs = np.abs(pool - np.asarray(x, dtype=np.float64)[None, :]).sum(axis=1)
    return int(np.argmin(diffs))


def _bo_best(
    make_surrogate: Callable[[], Any],
    make_acq: Callable[[], Any],
    pool: NDArray[np.float64],
    objective: Callable[[NDArray[np.float64]], float],
    *,
    n_trials: int,
    n_init: int,
    seed: int,
) -> float:
    """Finite-pool BO without replacement; returns best reward observed."""
    rng = np.random.default_rng(seed)
    surrogate = make_surrogate()
    acq = make_acq()
    n = len(pool)
    remaining = list(range(n))
    x_obs: list[NDArray[np.float64]] = []
    y_obs: list[float] = []
    best = -math.inf

    init = rng.choice(np.array(remaining), size=min(n_init, n), replace=False)
    for idx in init:
        i = int(idx)
        x = np.asarray(pool[i], dtype=np.float64)
        y = objective(x)
        x_obs.append(x)
        y_obs.append(y)
        best = max(best, y)
        remaining.remove(i)

    while len(y_obs) < n_trials and remaining:
        surrogate.fit(np.array(x_obs), np.array(y_obs))
        sub = pool[remaining]
        sel, _ = acq(surrogate, sub, y_best=best)
        j = _argmatch(sub, np.asarray(sel, dtype=np.float64))
        i = remaining[j]
        x = np.asarray(pool[i], dtype=np.float64)
        y = objective(x)
        x_obs.append(x)
        y_obs.append(y)
        best = max(best, y)
        remaining.remove(i)
    return best


def _random_best(
    pool: NDArray[np.float64],
    objective: Callable[[NDArray[np.float64]], float],
    *,
    n_trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(pool)
    best = -math.inf
    for _ in range(n_trials):
        idx = int(rng.integers(n))
        best = max(best, objective(np.asarray(pool[idx], dtype=np.float64)))
    return best


def _toy_setup(
    rung: str,
) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], float], Callable[[], Any]]:
    rng = np.random.default_rng(7)
    pool_n = 18
    make: Callable[[], Any]
    if rung in ("tpe", "random_forest"):
        d = 4
        pool = rng.random((pool_n, d))
        target = np.full(d, 0.35)
        pool[0] = target
        make = (
            partial(TPESurrogate, seed=0)
            if rung == "tpe"
            else partial(RandomForestSurrogate, n_estimators=80, seed=0)
        )
    elif rung == "bocs":
        d = 8
        pool = rng.integers(0, 2, (pool_n, d)).astype(np.float64)
        target = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.float64)
        pool[0] = target
        make = partial(BOCSSurrogate)
    elif rung == "combo":
        d = 6
        pool = rng.integers(0, 3, (pool_n, d)).astype(np.float64) / 2.0
        target = np.full(d, 0.5)
        pool[0] = target
        make = partial(COMBOSurrogate, discrete_cols=list(range(d)))
    elif rung == "mixed_gp":
        cont = rng.random((pool_n, 3))
        disc = rng.integers(0, 2, (pool_n, 3)).astype(np.float64)
        pool = np.hstack([cont, disc])
        target = np.array([0.3, 0.6, 0.4, 1.0, 0.0, 1.0], dtype=np.float64)
        pool[0] = target
        make = partial(MixedGPSurrogate, continuous_cols=[0, 1, 2], discrete_cols=[3, 4, 5])
    elif rung == "dictionary_embedding":
        d = 30
        pool = rng.integers(0, 3, (pool_n, d)).astype(np.float64) / 2.0
        target = rng.integers(0, 3, d).astype(np.float64) / 2.0
        pool[0] = target
        make = partial(DictionaryEmbeddingSurrogate, embed_dim=8, seed=0)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown rung {rung!r}")
    return pool, _bowl(target), make


@pytest.mark.parametrize(
    "rung",
    ["tpe", "random_forest", "bocs", "combo", "mixed_gp", "dictionary_embedding"],
)
def test_rung_ties_or_beats_random_on_toy(rung: str) -> None:
    pool, objective, make = _toy_setup(rung)
    best_rung = _bo_best(
        make, LadderEI, pool, objective, n_trials=25, n_init=5, seed=0
    )
    best_rand = _random_best(pool, objective, n_trials=25, seed=0)
    assert best_rung >= best_rand - 1e-9, (
        f"{rung}: {best_rung} lost to random {best_rand}"
    )


# ── surrogate protocol conformance + predict contract ────────────────────────


@pytest.mark.parametrize(
    "make",
    [
        lambda: TPESurrogate(seed=0),
        lambda: RandomForestSurrogate(n_estimators=20, seed=0),
        lambda: BOCSSurrogate(),
        lambda: COMBOSurrogate(discrete_cols=[0, 1, 2]),
        lambda: MixedGPSurrogate(continuous_cols=[0, 1], discrete_cols=[2]),
        lambda: DictionaryEmbeddingSurrogate(embed_dim=2, seed=0),
    ],
)
def test_surrogate_protocol_conformance(make: Callable[[], Any]) -> None:
    surrogate = make()
    assert isinstance(surrogate, Surrogate)
    assert not surrogate.is_fitted()

    rng = np.random.default_rng(0)
    X = rng.random((6, 3))
    y = -np.sum((X - 0.5) ** 2, axis=1)
    surrogate.fit(X, y)
    assert surrogate.is_fitted()

    mean, std = surrogate.predict(X)
    assert mean.shape == (6,)
    assert std.shape == (6,)
    assert bool(np.all(np.isfinite(mean.numpy())))
    assert bool(np.all(std.numpy() >= 0.0))


def test_ladder_ei_works_with_gaussian_process() -> None:
    from opop.controller.gp import GaussianProcess

    gp = GaussianProcess(lengthscale=0.3, noise_var=1e-4)
    rng = np.random.default_rng(0)
    X = rng.random((8, 2))
    y = -np.sum((X - 0.3) ** 2, axis=1)
    gp.fit(X, y)
    pool = rng.random((30, 2))
    acq = LadderEI()
    sel, value = acq(gp, pool, y_best=float(np.max(y)))
    assert sel.shape == (2,)
    assert np.isfinite(value)


# ── Phase1Controller.ladder factory integration (mixed -> mixed_gp, runs) ─────


def _phase1_bowl(space: Phase1Space) -> Callable[[Phi], float]:
    target_phi = replace(
        space.base,
        c="cuts_on",
        d="benders",
        h=2,
        p={
            "separating/gomory/freq": 2.5,
            "separating/clique/freq": 2.5,
            "separating/zerohalf/freq": 2.5,
            "branching/scorefac": 0.25,
            "presolving/maxrounds": 5.0,
            "limits/gap": 0.0001,
        },
    )
    target = space.encode(target_phi)

    def objective(phi: Phi) -> float:
        x = space.encode(phi)
        return -float(np.sum((x - target) ** 2))

    return objective


def test_ladder_factory_wires_mixed_gp_and_beats_random() -> None:
    space = default_phase1_space()
    objective = _phase1_bowl(space)
    pool = space.candidate_pool(60, np.random.default_rng(123))

    ladder = Phase1Controller.ladder(space, budget=30, n_trials=25, n_init=5, seed=0)
    assert isinstance(ladder.surrogate, MixedGPSurrogate)

    rand = Phase1Controller.random(space, n_trials=25, seed=0)

    ladder_res = ladder.run(objective, candidates=pool)
    rand_res = rand.run(objective, candidates=pool)

    assert len(ladder_res.history) == 25
    assert ladder_res.best_reward >= rand_res.best_reward - 1e-9, (
        f"ladder {ladder_res.best_reward} lost to random {rand_res.best_reward}"
    )


def test_ladder_factory_falls_back_to_gp_when_botorch_absent() -> None:
    from opop.controller.botorch_rungs import botorch_available
    from opop.controller.gp import GaussianProcess

    space = _continuous_space_with_real_fields()
    ladder = Phase1Controller.ladder(space, budget=30, n_trials=4, n_init=2, seed=0)
    if botorch_available():
        from opop.controller.botorch_rungs import BoTorchGPSurrogate

        assert isinstance(ladder.surrogate, BoTorchGPSurrogate)
    else:
        assert isinstance(ladder.surrogate, GaussianProcess)
        assert isinstance(ladder.acquisition, LadderEI)


def _continuous_space_with_real_fields() -> Phase1Space:
    base = replace(Phi(), rho={"a": 0.5}, p={"b": 0.5})
    return Phase1Space(
        base=base,
        dims=(ContinuousDim("h", 0.0, 1.0), ContinuousDim("s", 0.0, 1.0)),
    )


# ── SMAC3 rung (censored-aware ask-tell); skipped cleanly without smac ────────


def test_smac_records_censored_timeout_trial() -> None:
    pytest.importorskip("smac")
    pytest.importorskip("ConfigSpace")
    from smac.runhistory.enumerations import (  # pyright: ignore[reportMissingImports]
        StatusType,
    )

    from opop.controller.ladder import SMACSurrogate

    surrogate = SMACSurrogate(dim=2, n_trials=10, seed=0)
    assert isinstance(surrogate, Surrogate)

    x = surrogate.ask()
    surrogate.tell(x, reward=-5.0, censored=True, time=2.0)
    statuses = [surrogate.runhistory[k].status for k in surrogate.runhistory]
    assert StatusType.TIMEOUT in statuses

    x2 = surrogate.ask()
    surrogate.tell(x2, reward=-1.0, censored=False)
    statuses = [surrogate.runhistory[k].status for k in surrogate.runhistory]
    assert StatusType.SUCCESS in statuses
    assert StatusType.TIMEOUT in statuses


def test_smac_ties_or_beats_random_on_toy() -> None:
    pytest.importorskip("smac")
    pytest.importorskip("ConfigSpace")
    from opop.controller.ladder import SMACSurrogate

    d = 2
    target = np.full(d, 0.3)
    objective = _bowl(target)

    surrogate = SMACSurrogate(dim=d, n_trials=50, seed=0)
    best = -math.inf
    for _ in range(50):
        x = surrogate.ask()
        reward = objective(x)
        surrogate.tell(x, reward)
        best = max(best, reward)

    rng = np.random.default_rng(0)
    rand_best = max(objective(rng.random(d)) for _ in range(50))
    assert best >= rand_best - 1e-9


# ── BoTorch rungs (qLogNEI default / qKG batch); skipped cleanly w/o botorch ──


def test_qlognei_ties_or_beats_random_on_toy() -> None:
    pytest.importorskip("botorch")
    from opop.controller.botorch_rungs import BoTorchGPSurrogate, QLogNEIAcquisition

    rng = np.random.default_rng(7)
    pool = rng.random((18, 2))
    target = np.full(2, 0.4)
    pool[0] = target
    objective = _bowl(target)

    best_rung = _bo_best(
        lambda: BoTorchGPSurrogate(),
        lambda: QLogNEIAcquisition(num_samples=64),
        pool,
        objective,
        n_trials=20,
        n_init=4,
        seed=0,
    )
    best_rand = _random_best(pool, objective, n_trials=20, seed=0)
    assert best_rung >= best_rand - 1e-9


def test_qknowledge_gradient_proposes_and_batches() -> None:
    pytest.importorskip("botorch")
    from opop.controller.botorch_rungs import (
        BoTorchGPSurrogate,
        QKnowledgeGradientAcquisition,
    )

    rng = np.random.default_rng(0)
    X = rng.random((10, 2))
    y = -np.sum((X - 0.4) ** 2, axis=1)
    surrogate = BoTorchGPSurrogate()
    surrogate.fit(X, y)

    pool = rng.random((12, 2))
    acq = QKnowledgeGradientAcquisition(
        num_fantasies=8, num_restarts=2, raw_samples=32
    )
    sel, value = acq(surrogate, pool, y_best=float(np.max(y)))
    assert sel.shape == (2,)
    assert np.isfinite(value)
    assert any(np.allclose(sel, row) for row in pool)

    batch, _ = acq.optimize(surrogate, q=2)
    assert batch.shape == (2, 2)


def test_botorch_mixed_gp_surrogate_conformance() -> None:
    pytest.importorskip("botorch")
    from opop.controller.botorch_rungs import BoTorchMixedGPSurrogate

    surrogate = BoTorchMixedGPSurrogate(cat_dims=[2])
    assert isinstance(surrogate, Surrogate)
    rng = np.random.default_rng(0)
    X = rng.random((8, 3))
    X[:, 2] = (X[:, 2] > 0.5).astype(np.float64)
    y = -np.sum((X - 0.3) ** 2, axis=1)
    surrogate.fit(X, y)
    mean, _ = surrogate.predict(X)
    assert mean.shape == (8,)
    assert bool(np.all(np.isfinite(mean.numpy())))
