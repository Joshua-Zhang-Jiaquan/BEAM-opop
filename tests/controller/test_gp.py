"""Tests for the GP + acquisition Bayesian optimization base."""

from __future__ import annotations

import numpy as np
import pytest

from opop.controller.acquisition import (
    ei_acquisition,
    random_acquisition,
    run_bo_trials,
    scalarized_reward,
    ucb_acquisition,
)
from opop.controller.gp import GaussianProcess
from opop.controller.protocol import EI, RandomSearch, Surrogate, UCB


def _nearest_train_distance(X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    return np.min(np.abs(X_test[:, None] - X_train[None, :]), axis=1).ravel()


def test_gp_fits_1d_sine() -> None:
    """A GP with a Matern-5/2 kernel should recover a 1-D sine function."""
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 1, size=(15, 1))
    y_train = np.sin(2 * np.pi * X_train.ravel())

    gp = GaussianProcess(lengthscale=0.2, signal_var=1.0, noise_var=1e-4)
    gp.fit(X_train, y_train)

    assert gp.is_fitted()
    assert gp.n_train == len(X_train)
    assert np.isfinite(gp.log_marginal_likelihood())

    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X_test.ravel())
    mean, std = gp.predict(X_test)
    mean = mean.numpy()
    std = std.numpy()

    rmse = float(np.sqrt(np.mean((mean - y_true) ** 2)))
    assert rmse < 0.35, f"RMSE too large: {rmse}"
    # Predictive uncertainty should be larger away from training data.
    d = _nearest_train_distance(X_test, X_train)
    near_std = std[d == d.min()].mean()
    far_std = std[d == d.max()].mean()
    assert far_std > near_std, "Variance did not expand away from observations"


def test_ucb_selects_candidate() -> None:
    """UCB returns a candidate from the supplied pool."""
    X_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess()
    gp.fit(X_train, y_train)

    X_cand = np.linspace(0, 1, 20).reshape(-1, 1)
    config, value = ucb_acquisition(gp, X_cand, kappa=2.0)
    assert config.shape == (1,)
    assert 0.0 <= config[0] <= 1.0
    assert np.isfinite(value)


def test_ei_selects_candidate() -> None:
    """EI returns a candidate and positive acquisition value."""
    X_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess()
    gp.fit(X_train, y_train)

    X_cand = np.linspace(0, 1, 20).reshape(-1, 1)
    config, value = ei_acquisition(gp, X_cand)
    assert config.shape == (1,)
    assert 0.0 <= config[0] <= 1.0
    assert np.isfinite(value)


def test_ei_beats_random_on_toy() -> None:
    """EI should outperform random search on a simple 1-D quadratic."""

    def objective(x: np.ndarray) -> float:
        return -((x[0] - 0.3) ** 2)

    rng = np.random.default_rng(0)
    X_candidates = rng.uniform(0, 1, size=(200, 1))
    X_init = rng.uniform(0, 1, size=(3, 1))
    y_init = np.array([objective(x) for x in X_init])

    gp_ei = GaussianProcess(lengthscale=0.2, noise_var=1e-4)
    _, _, trace_ei, _ = run_bo_trials(
        gp_ei,
        X_init,
        y_init,
        X_candidates,
        objective,
        n_trials=15,
        acquisition="ei",
    )

    gp_rand = GaussianProcess(lengthscale=0.2, noise_var=1e-4)
    _, _, trace_rand, _ = run_bo_trials(
        gp_rand,
        X_init.copy(),
        y_init.copy(),
        X_candidates,
        objective,
        n_trials=15,
        acquisition="random",
    )

    best_ei = max(trace_ei)
    best_rand = max(trace_rand)
    # With only 15 trials on a noisy-free quadratic, EI should tie or beat random.
    assert best_ei >= best_rand - 1e-6, f"EI {best_ei} lost to random {best_rand}"

    # Posterior variance should shrink near evaluated points.
    gp_ei.fit(np.array(gp_ei.X_train), np.array(gp_ei.y_train))
    mean, std = gp_ei.predict(X_candidates)
    sampled = set(tuple(np.round(x, 6)) for x in gp_ei.X_train.numpy())
    sampled_std = std[
        np.array([tuple(np.round(x, 6)) in sampled for x in X_candidates])
    ].mean()
    unsampled_std = std[
        np.array([tuple(np.round(x, 6)) not in sampled for x in X_candidates])
    ].mean()
    assert sampled_std < unsampled_std, "Variance did not shrink at observed points"


def test_random_acquisition_is_random() -> None:
    """Random acquisition ignores the GP and samples uniformly."""
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.0, 1.0])
    gp = GaussianProcess()
    gp.fit(X_train, y_train)

    X_cand = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    config, value = random_acquisition(X_cand, gp=gp, seed=7)
    assert config.shape == (1,)
    assert value == 0.0


def test_scalarized_reward_matches_hand_value() -> None:
    """scalarized_reward must match a hand-computed weighted sum."""
    metrics = {
        "wns": 0.1,
        "tns": -2.0,
        "total_power_mw": 100.0,
        "area_um2": 50000.0,
        "drc_violations": 3,
        "runtime_seconds": 120.0,
    }
    expected = (
        1.0 * 0.1
        - 0.01 * abs(-2.0)
        - 0.001 * 100.0
        - 1e-6 * 50000.0
        - 0.001 * 3
        - 0.0001 * 120.0
    )
    assert scalarized_reward(metrics) == pytest.approx(expected, abs=1e-9)


def test_protocol_wrappers_accept_gp_surrogate() -> None:
    """UCB/EI/RandomSearch instances satisfy the Acquisition protocol."""
    X_train = np.array([[0.0], [0.5], [1.0]])
    y_train = np.array([0.0, 1.0, 0.2])
    gp = GaussianProcess()
    gp.fit(X_train, y_train)

    X_cand = np.linspace(0, 1, 10).reshape(-1, 1)

    assert isinstance(gp, Surrogate)
    for policy in (UCB(kappa=2.0), EI(), RandomSearch(seed=0)):
        config, value = policy(gp, X_cand)
        assert config.shape == (1,)
        assert np.isfinite(value)


def test_gp_pseudoinverse_fallback() -> None:
    """A GP with duplicate training points should still fit via pinv fallback."""
    X_train = np.array([[0.5], [0.5], [0.5], [0.5]])
    y_train = np.array([1.0, 1.01, 0.99, 1.0])

    gp = GaussianProcess(noise_var=1e-6)
    gp.fit(X_train, y_train)

    mean, std = gp.predict(np.array([[0.5], [0.0]]))
    assert np.all(np.isfinite(mean.numpy()))
    assert np.all(np.isfinite(std.numpy()))
