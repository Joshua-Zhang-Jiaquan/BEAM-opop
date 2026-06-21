"""Controller protocols and baseline implementations.

Defines the :class:`Surrogate` and :class:`Acquisition` protocols so that the
Phase-1 GP controller and later SMAC/TPE/BoTorch/structured BO controllers are
interchangeable for callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from .gp import GaussianProcess

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class Surrogate(Protocol):
    """Protocol for a probabilistic surrogate model used by the controller."""

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        """Fit the surrogate to ``(X, y)`` observations."""
        ...

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` posterior predictives for ``X_test``."""
        ...

    def is_fitted(self) -> bool:
        """Return whether the surrogate has been fit."""
        ...

    def log_marginal_likelihood(self) -> float:
        """Return log marginal likelihood of the training data, if available."""
        ...


@runtime_checkable
class Acquisition(Protocol):
    """Protocol for an acquisition policy that selects the next candidate."""

    def __call__(
        self,
        surrogate: Surrogate,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Select the next candidate from ``X_candidates``.

        Args:
            surrogate: Fitted surrogate model (may be ignored by baselines).
            X_candidates: ``[n_candidates, n_features]`` candidate points.
            y_best: Best observed value so far (for improvement-based rules).
            kappa: Exploration parameter (for UCB-style rules).
            seed: Optional reproducibility seed.

        Returns:
            ``(selected_config, acquisition_value)``.
        """
        ...


class RandomSearch:
    """Baseline controller implementing :class:`Acquisition` by uniform sampling.

    The surrogate argument is ignored, matching the protocol so callers can
    substitute ``RandomSearch`` for any acquisition policy.
    """

    rng: np.random.Generator

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        surrogate: Surrogate,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Return a uniformly random candidate from ``X_candidates``."""
        del surrogate, y_best, kappa  # unused baseline
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        idx = int(rng.integers(len(X_candidates)))
        return np.asarray(X_candidates[idx]).copy(), 0.0


# Convenience aliases that adapt the functional acquisitions to the protocol.
# These are thin stateless wrappers so callers can pass an Acquisition instance.


class UCB:
    """UCB acquisition as an :class:`Acquisition` protocol instance."""

    kappa: float

    def __init__(self, kappa: float = 2.0) -> None:
        self.kappa = kappa

    def __call__(
        self,
        surrogate: Surrogate,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        del y_best, seed  # unused
        from .acquisition import ucb_acquisition

        if not isinstance(surrogate, GaussianProcess):
            raise TypeError("UCB currently requires a GaussianProcess surrogate")
        return ucb_acquisition(surrogate, X_candidates, kappa=self.kappa)


class EI:
    """Expected Improvement acquisition as an :class:`Acquisition` instance."""

    def __call__(
        self,
        surrogate: Surrogate,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        del kappa, seed  # unused
        from .acquisition import ei_acquisition

        if not isinstance(surrogate, GaussianProcess):
            raise TypeError("EI currently requires a GaussianProcess surrogate")
        return ei_acquisition(surrogate, X_candidates, y_best=y_best)
