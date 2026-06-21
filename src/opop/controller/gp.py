"""Self-contained Gaussian Process surrogate for Bayesian Optimization.

Provides:
- :class:`GaussianProcess`: Matern 5/2 kernel, Cholesky inference with a
  pseudoinverse fallback, log marginal likelihood, and hyperparameter reset.

No external dependencies beyond ``torch`` (``numpy`` accepted for convenience).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GaussianProcess:
    """Self-contained GP with Matern 5/2 kernel and Cholesky-based inference.

    Supports fallback to pseudoinverse when the kernel matrix is near-singular.
    """

    lengthscale: float
    signal_var: float
    noise_var: float
    dtype: torch.dtype
    X_train: torch.Tensor | None
    y_train: torch.Tensor | None
    _cholesky: torch.Tensor | None
    alpha: torch.Tensor | None

    def __init__(
        self,
        lengthscale: float = 1.0,
        signal_var: float = 1.0,
        noise_var: float = 1e-3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.lengthscale = lengthscale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.dtype = dtype

        self.X_train = None
        self.y_train = None
        self._cholesky = None
        self.alpha = None

    # ── Kernel ──────────────────────────────────────────────────────────

    @staticmethod
    def _matern52_kernel(
        X1: torch.Tensor, X2: torch.Tensor, lengthscale: float
    ) -> torch.Tensor:
        """Matern 5/2 kernel: (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)."""
        dist = torch.cdist(X1 / lengthscale, X2 / lengthscale)
        sqrt5 = math.sqrt(5.0)
        r = sqrt5 * dist
        return (1.0 + r + r**2 / 3.0) * torch.exp(-r)

    # ── Fit ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        """Fit GP to training data (X, y).

        Args:
            X: ``[n_samples, n_features]`` input points.
            y: ``[n_samples]`` target values (scalarized rewards).
        """
        self.X_train = torch.as_tensor(X, dtype=self.dtype)
        self.y_train = torch.as_tensor(y, dtype=self.dtype).reshape(-1)

        n = len(self.X_train)
        kernel = self.signal_var * self._matern52_kernel(
            self.X_train, self.X_train, self.lengthscale
        )
        kernel = kernel + self.noise_var * torch.eye(n, dtype=self.dtype)

        try:
            cholesky = torch.linalg.cholesky(kernel)
            self._cholesky = cholesky
            self.alpha = torch.cholesky_solve(
                self.y_train.unsqueeze(-1), torch.as_tensor(cholesky)
            ).squeeze(-1)
        except RuntimeError:
            # Fallback: pseudoinverse for near-singular kernel matrix.
            self._cholesky = None
            self.alpha = (
                torch.linalg.pinv(kernel) @ self.y_train.unsqueeze(-1)
            ).squeeze(-1)

    def is_fitted(self) -> bool:
        return self.X_train is not None and self.alpha is not None

    @property
    def n_train(self) -> int:
        return len(self.X_train) if self.X_train is not None else 0

    # ── Predict ─────────────────────────────────────────────────────────

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior predictive mean and standard deviation.

        Args:
            X_test: ``[n_test, n_features]`` test points.

        Returns:
            ``(mean, std)`` where each tensor has shape ``[n_test]``.
        """
        if not self.is_fitted():
            raise RuntimeError("GP not fitted. Call fit() first.")

        X_train = self.X_train
        alpha = self.alpha
        assert X_train is not None and alpha is not None

        X_test_t = torch.as_tensor(X_test, dtype=self.dtype)
        K_s = self.signal_var * self._matern52_kernel(
            X_test_t, X_train, self.lengthscale
        )
        mean = K_s @ alpha

        K_ss_diag = self.signal_var * torch.ones(len(X_test_t), dtype=self.dtype)
        K_ss_diag += self.noise_var

        if self._cholesky is not None:
            # v = L^{-1} K_s^T  ->  L v = K_s^T -> solve triangular
            v = torch.linalg.solve_triangular(self._cholesky, K_s.T, upper=False)
            var = K_ss_diag - (v**2).sum(dim=0)
        else:
            # Fallback with pseudoinverse
            kernel = self.signal_var * self._matern52_kernel(
                X_train, X_train, self.lengthscale
            )
            kernel = kernel + self.noise_var * torch.eye(
                len(X_train), dtype=self.dtype
            )
            K_inv = torch.linalg.pinv(kernel)
            var = K_ss_diag - (K_s @ K_inv @ K_s.T).diag()

        std = torch.sqrt(torch.clamp(torch.as_tensor(var), min=1e-12))
        return mean, std

    # ── Log marginal likelihood ─────────────────────────────────────────

    def log_marginal_likelihood(self) -> float:
        """Compute log marginal likelihood: ``log p(y | X)``.

        Returns:
            Scalar log marginal likelihood; higher is better.
        """
        if not self.is_fitted():
            return -float("inf")

        X_train = self.X_train
        y = self.y_train
        assert X_train is not None and y is not None
        n = len(y)

        if self._cholesky is not None:
            # log|K| = 2 * sum(log(diag(L)))
            log_det = 2.0 * torch.sum(torch.log(torch.diag(self._cholesky)))
            # y^T K^{-1} y = || L^{-1} y ||^2
            L_inv_y = torch.linalg.solve_triangular(
                self._cholesky, y.unsqueeze(-1), upper=False
            )
            quad = (L_inv_y**2).sum()
        else:
            kernel = self.signal_var * self._matern52_kernel(
                X_train, X_train, self.lengthscale
            )
            kernel = kernel + self.noise_var * torch.eye(n, dtype=self.dtype)
            eigvals = torch.linalg.eigvalsh(kernel)
            log_det = torch.sum(torch.log(torch.as_tensor(eigvals).clamp(min=1e-12)))
            quad = y @ (torch.linalg.pinv(kernel) @ y)

        lml = -0.5 * (quad + log_det + n * math.log(2.0 * math.pi))
        return lml.item()

    # ── Hyperparameter setting ──────────────────────────────────────────

    def set_hyperparams(
        self,
        lengthscale: float | None = None,
        signal_var: float | None = None,
        noise_var: float | None = None,
    ) -> None:
        """Update GP hyperparameters (requires re-fit)."""
        if lengthscale is not None:
            self.lengthscale = lengthscale
        if signal_var is not None:
            self.signal_var = signal_var
        if noise_var is not None:
            self.noise_var = noise_var
        # Invalidate cached fit
        self.X_train = None
        self.y_train = None
        self._cholesky = None
        self.alpha = None
