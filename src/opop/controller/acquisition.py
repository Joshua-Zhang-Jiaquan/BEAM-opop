"""Acquisition functions and Bayesian Optimization trial driver.

Provides:
- :func:`scalarized_reward`: multi-objective trial metrics -> scalar reward.
- :func:`ucb_acquisition`, :func:`ei_acquisition`, :func:`random_acquisition`.
- :func:`run_bo_trials`: sequential BO loop over a finite candidate pool.

All implementations use ``numpy`` + ``torch`` only; they plug into the
:mod:`opop.controller.protocol` abstractions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np
import torch

from .gp import GaussianProcess

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _HasToDict(Protocol):
    def to_dict(self) -> Mapping[str, float]: ...


# ── Reward scalarization ────────────────────────────────────────────────────


def scalarized_reward(
    metrics: Mapping[str, float] | _HasToDict,
    wns_weight: float = 1.0,
    tns_weight: float = 0.01,
    power_weight: float = 0.001,
    area_weight: float = 1e-6,
    drc_weight: float = 0.001,
    runtime_weight: float = 0.0001,
) -> float:
    """Convert multi-objective trial metrics into a scalar reward.

    Higher is better. Default weights mirror the original OpenROAD-style
    reward surface, but the function is used as-is by the GP controller.

    Args:
        metrics: Mapping or dataclass with ``to_dict()`` containing the keys
            ``wns``, ``tns``, ``total_power_mw``, ``area_um2``,
            ``drc_violations``, and ``runtime_seconds``.
        wns_weight: weight for the WNS term.
        tns_weight: penalty weight for ``|tns|``.
        power_weight: penalty weight for power.
        area_weight: penalty weight for area.
        drc_weight: penalty weight for DRC violations.
        runtime_weight: penalty weight for runtime.

    Returns:
        Scalar reward value (higher is better).
    """
    if isinstance(metrics, Mapping):
        m = metrics
    else:
        m = metrics.to_dict()

    wns = float(m.get("wns", 0.0))
    tns = float(m.get("tns", 0.0))
    power = float(m.get("total_power_mw", 0.0))
    area = float(m.get("area_um2", 0.0))
    drc = float(m.get("drc_violations", 0))
    runtime = float(m.get("runtime_seconds", 0.0))

    reward = wns_weight * wns
    reward -= tns_weight * abs(tns)
    reward -= power_weight * power
    reward -= area_weight * area
    reward -= drc_weight * drc
    reward -= runtime_weight * runtime
    return reward


# ── Acquisition Functions ───────────────────────────────────────────────────


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF using ``erfc``."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def ucb_acquisition(
    gp: GaussianProcess,
    X_candidates: NDArray[np.float64] | torch.Tensor,
    kappa: float = 2.0,
) -> tuple[NDArray[np.float64], float]:
    """Upper Confidence Bound: ``mu(x) + kappa * sigma(x)``.

    Args:
        gp: Fitted :class:`GaussianProcess`.
        X_candidates: ``[n_candidates, n_features]`` candidate points.
        kappa: Exploration-exploitation trade-off.

    Returns:
        ``(best_config, best_ucb_value)``.
    """
    X_cand = torch.as_tensor(X_candidates, dtype=gp.dtype)
    mean, std = gp.predict(X_cand)
    ucb_values = mean + kappa * std
    best_idx = int(torch.argmax(ucb_values).item())
    return X_cand[best_idx].numpy(), ucb_values[best_idx].item()


def ei_acquisition(
    gp: GaussianProcess,
    X_candidates: NDArray[np.float64] | torch.Tensor,
    y_best: float | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Expected Improvement: ``E[max(0, f(x) - y_best)]``.

    Args:
        gp: Fitted :class:`GaussianProcess`.
        X_candidates: ``[n_candidates, n_features]`` candidate points.
        y_best: Current best observed value; defaults to ``max(gp.y_train)``.

    Returns:
        ``(best_config, best_ei_value)``.
    """
    X_cand = torch.as_tensor(X_candidates, dtype=gp.dtype)
    mean, std = gp.predict(X_cand)

    if y_best is None:
        if gp.y_train is None:
            raise RuntimeError("GP has no training data; supply y_best.")
        y_best = float(gp.y_train.max().item())

    y_best_t = torch.tensor(y_best, dtype=gp.dtype)
    improvement = mean - y_best_t

    Z = improvement / torch.clamp(std, min=1e-12)
    ei_values = std * (Z * _normal_cdf(Z) + _normal_pdf(Z))

    zero_std_mask = std < 1e-12
    ei_values[zero_std_mask] = torch.clamp(improvement[zero_std_mask], min=0.0)

    best_idx = int(torch.argmax(ei_values).item())
    return X_cand[best_idx].numpy(), float(ei_values[best_idx].item())


def random_acquisition(
    X_candidates: NDArray[np.float64] | torch.Tensor,
    gp: GaussianProcess | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Uniform random acquisition baseline.

    Args:
        X_candidates: ``[n_candidates, n_features]`` candidate points.
        gp: Ignored; present for API compatibility.
        seed: Optional random seed.

    Returns:
        ``(random_config, 0.0)``.
    """
    del gp  # unused baseline
    X_cand = np.asarray(X_candidates)
    rng = random.Random(seed)
    idx = rng.randrange(len(X_cand))
    return X_cand[idx].copy(), 0.0


# ── BO trial driver ─────────────────────────────────────────────────────────


def run_bo_trials(
    gp: GaussianProcess,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_candidates: NDArray[np.float64],
    candidate_evaluator: Callable[[NDArray[np.float64]], float],
    n_trials: int = 50,
    acquisition: str = "ucb",
    kappa: float = 2.0,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float], NDArray[np.float64]]:
    """Run sequential Bayesian Optimization over a finite candidate pool.

    Args:
        gp: Initialized :class:`GaussianProcess`.
        X_train: ``[n_init, n_features]`` initial training inputs.
        y_train: ``[n_init]`` initial training targets.
        X_candidates: ``[n_candidates, n_features]`` pool to select from.
        candidate_evaluator: callable mapping a single candidate to a scalar.
        n_trials: Number of BO iterations.
        acquisition: ``"ucb"``, ``"ei"``, or ``"random"``.
        kappa: UCB exploration parameter.
        verbose: Print progress every 10 trials.

    Returns:
        ``(X_all, y_all, best_trace, best_config)``.
    """
    X_all: list[NDArray[np.float64]] = [np.asarray(x) for x in X_train]
    y_all: list[float] = [float(y) for y in y_train]
    best_trace: list[float] = []
    best_y = float(max(y_all))

    for trial in range(n_trials):
        X_arr = np.array(X_all)
        y_arr = np.array(y_all)

        gp.fit(X_arr, y_arr)

        if acquisition == "ucb":
            next_x, _ = ucb_acquisition(gp, X_candidates, kappa=kappa)
        elif acquisition == "ei":
            next_x, _ = ei_acquisition(gp, X_candidates, y_best=best_y)
        elif acquisition == "random":
            next_x, _ = random_acquisition(X_candidates, seed=trial)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

        next_y = candidate_evaluator(next_x)

        X_all.append(np.asarray(next_x))
        y_all.append(float(next_y))

        if next_y > best_y:
            best_y = float(next_y)

        best_trace.append(best_y)

        if verbose and (trial + 1) % 10 == 0:
            print(f"  BO trial {trial + 1}/{n_trials}: best_y={best_y:.4f}")

    X_arr = np.array(X_all)
    y_arr = np.array(y_all)
    best_idx = int(np.argmax(y_arr))
    return X_arr, y_arr, best_trace, X_arr[best_idx].copy()
