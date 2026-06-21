"""Controller ladder: SMAC / TPE / RF + structured-BO surrogate selection.

Adds the Wave-4 controller rungs *behind the task-8
:class:`~opop.controller.protocol.Surrogate` /
:class:`~opop.controller.protocol.Acquisition` protocols*, so they drop into
:class:`~opop.controller.phase1.Phase1Controller` without touching callers.

Rungs
-----
* :class:`SMACSurrogate` — SMAC3 ask-tell (RF surrogate + logEI internally),
  censored/``TIMEOUT``-aware, constructed with ``overwrite=True``.  This is the
  production drop-in for the cheap RF rung; it imports ``smac`` lazily.
* :class:`TPESurrogate` — self-contained Tree-structured Parzen Estimator
  (``numpy`` density-ratio l(x)/g(x)); no optional dependency.
* :class:`RandomForestSurrogate` — ``sklearn`` random forest with tree-variance
  uncertainty (the dependency-free realisation of SMAC's own RF model).
* :class:`BOCSSurrogate` — binary/boolean spaces: sparse **B**ayesian linear
  regression **o**ver **c**ombinatorial **s**tructures (first + second-order
  monomials over ``{0, 1}`` variables).
* :class:`COMBOSurrogate` — pure discrete spaces: a diffusion/RBF-kernel GP over
  the discrete encoding (the continuous relaxation of COMBO's graph kernel).
* :class:`MixedGPSurrogate` — mixed discrete+continuous spaces: a product kernel
  ``Matern(continuous) x Matern(discrete)`` (the CoCaBO / HyBO mixed kernel).
* :class:`DictionaryEmbeddingSurrogate` — high-dim discrete spaces: a fixed
  random *dictionary* embedding into a low-dim space + Matern GP.

The acquisition wrapper :class:`LadderEI` is a generic (surrogate-agnostic)
Expected Improvement that works with *any* surrogate exposing
``predict(X) -> (mean, std)`` — unlike the task-8 :class:`~opop.controller.protocol.EI`,
which is hard-wired to :class:`~opop.controller.gp.GaussianProcess`.

Router
------
:func:`select_surrogate` inspects the encoded :class:`~opop.controller.encoder.Phase1Space`
shape (bool / categorical / ordinal / continuous mix and dimensionality), the
trial ``budget`` and the observation ``noise`` level and returns a
:class:`RungChoice` naming the chosen rung and zero-arg factories.  The ladder is
*evidence-first*: tiny budgets fall back to TPE / RF (the cheap "SMAC/TPE" tier),
structured rungs are selected only when the space shape **and** the budget
justify them, and BoTorch ``qLogNoisyExpectedImprovement`` is the default once a
full GP model is affordable.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, final

import numpy as np
import torch

from .encoder import (
    BoolDim,
    CategoricalDim,
    ContinuousDim,
    OrdinalDim,
    Phase1Space,
)
from .gp import GaussianProcess

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BOCSSurrogate",
    "COMBOSurrogate",
    "DictionaryEmbeddingSurrogate",
    "HIGH_DIM_THRESHOLD",
    "LadderEI",
    "MIN_MODEL_BUDGET",
    "MIN_STRUCTURED_BUDGET",
    "MixedGPSurrogate",
    "RandomForestSurrogate",
    "RungChoice",
    "SMACSurrogate",
    "SpaceShape",
    "TPESurrogate",
    "analyze_space",
    "select_surrogate",
]


# Router thresholds (module constants so callers/tests can reference them).
#: Encoded dimensionality at/above which a discrete space counts as "high-dim".
HIGH_DIM_THRESHOLD = 20
#: Minimum trial budget to fit *any* model rung (below it, use the cheap tier).
MIN_MODEL_BUDGET = 6
#: Minimum trial budget before a structured-BO rung is worth its overhead.
MIN_STRUCTURED_BUDGET = 8


# ── normal pdf/cdf (torch) ──────────────────────────────────────────────────


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via ``erfc`` (matches :mod:`opop.controller.acquisition`)."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _logsumexp(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Numerically stable ``log(sum(exp(a)))`` along ``axis`` (no scipy dep)."""
    arr = np.asarray(a, dtype=np.float64)
    m = np.max(arr, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    summed = np.sum(np.exp(arr - m), axis=axis, keepdims=True)
    out = m + np.log(np.clip(summed, 1e-300, None))
    return np.squeeze(out, axis=axis)


# ── kernel correlation functions (diag == 1; signal_var applied by _KernelGP) ─


def _matern52_corr(
    X1: torch.Tensor, X2: torch.Tensor, lengthscale: float
) -> torch.Tensor:
    """Matern-5/2 correlation: ``(1 + sqrt5 r + 5 r^2 / 3) exp(-sqrt5 r)``."""
    dist = torch.cdist(X1 / lengthscale, X2 / lengthscale)
    r = math.sqrt(5.0) * dist
    return (1.0 + r + r**2 / 3.0) * torch.exp(-r)


def _rbf_corr(X1: torch.Tensor, X2: torch.Tensor, lengthscale: float) -> torch.Tensor:
    """Squared-exponential (diffusion) correlation: ``exp(-||x-x'||^2 / (2 l^2))``."""
    d2 = torch.cdist(X1 / lengthscale, X2 / lengthscale) ** 2
    return torch.exp(-0.5 * d2)


KernelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ── self-contained Cholesky GP over an arbitrary correlation kernel ─────────


@final
class _KernelGP:
    """GP with a pluggable correlation kernel (mirrors :class:`GaussianProcess`).

    The kernel callable must return a correlation matrix with unit diagonal;
    ``signal_var`` and ``noise_var`` are applied here, exactly as in
    :class:`~opop.controller.gp.GaussianProcess`, including the Cholesky path
    with a pseudoinverse fallback for near-singular kernels.
    """

    def __init__(
        self,
        kernel: KernelFn,
        *,
        signal_var: float = 1.0,
        noise_var: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self._kernel = kernel
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.dtype = dtype
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self._cholesky: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        self.X_train = torch.as_tensor(X, dtype=self.dtype)
        self.y_train = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        n = len(self.X_train)
        kernel = self.signal_var * self._kernel(self.X_train, self.X_train)
        kernel = kernel + self.noise_var * torch.eye(n, dtype=self.dtype)
        try:
            cholesky = torch.linalg.cholesky(kernel)
            self._cholesky = cholesky
            self.alpha = torch.cholesky_solve(
                self.y_train.unsqueeze(-1), torch.as_tensor(cholesky)
            ).squeeze(-1)
        except RuntimeError:
            self._cholesky = None
            self.alpha = (
                torch.linalg.pinv(kernel) @ self.y_train.unsqueeze(-1)
            ).squeeze(-1)

    def is_fitted(self) -> bool:
        return self.X_train is not None and self.alpha is not None

    @property
    def n_train(self) -> int:
        return len(self.X_train) if self.X_train is not None else 0

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted():
            raise RuntimeError("kernel GP not fitted. Call fit() first.")
        X_train = self.X_train
        alpha = self.alpha
        assert X_train is not None and alpha is not None
        X_test_t = torch.as_tensor(X_test, dtype=self.dtype)
        K_s = self.signal_var * self._kernel(X_test_t, X_train)
        mean = K_s @ alpha
        K_ss_diag = self.signal_var * torch.ones(len(X_test_t), dtype=self.dtype)
        K_ss_diag = K_ss_diag + self.noise_var
        if self._cholesky is not None:
            v = torch.linalg.solve_triangular(self._cholesky, K_s.T, upper=False)
            var = K_ss_diag - (v**2).sum(dim=0)
        else:
            kernel = self.signal_var * self._kernel(X_train, X_train)
            kernel = kernel + self.noise_var * torch.eye(len(X_train), dtype=self.dtype)
            K_inv = torch.linalg.pinv(kernel)
            var = K_ss_diag - (K_s @ K_inv @ K_s.T).diag()
        std = torch.sqrt(torch.clamp(torch.as_tensor(var), min=1e-12))
        return mean, std

    def log_marginal_likelihood(self) -> float:
        if not self.is_fitted():
            return -math.inf
        X_train = self.X_train
        y = self.y_train
        assert X_train is not None and y is not None
        n = len(y)
        if self._cholesky is not None:
            log_det = 2.0 * torch.sum(torch.log(torch.diag(self._cholesky)))
            L_inv_y = torch.linalg.solve_triangular(
                self._cholesky, y.unsqueeze(-1), upper=False
            )
            quad = (L_inv_y**2).sum()
        else:
            kernel = self.signal_var * self._kernel(X_train, X_train)
            kernel = kernel + self.noise_var * torch.eye(n, dtype=self.dtype)
            eigvals = torch.linalg.eigvalsh(kernel)
            log_det = torch.sum(torch.log(torch.as_tensor(eigvals).clamp(min=1e-12)))
            quad = y @ (torch.linalg.pinv(kernel) @ y)
        lml = -0.5 * (quad + log_det + n * math.log(2.0 * math.pi))
        return lml.item()


# ── generic Expected Improvement over any Surrogate ─────────────────────────


@final
class LadderEI:
    """Expected Improvement over *any* surrogate exposing ``predict -> (mean, std)``.

    Unlike :class:`opop.controller.protocol.EI` (restricted to
    :class:`~opop.controller.gp.GaussianProcess`), this evaluates the analytic EI
    on whatever ``(mean, std)`` the rung's surrogate returns, so it serves every
    ladder rung uniformly.  For density-ratio surrogates (TPE) the std is a
    constant, so EI degenerates to picking the highest density-ratio candidate.
    """

    def __call__(
        self,
        surrogate: object,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        del kappa, seed  # EI uses neither
        predict = getattr(surrogate, "predict", None)
        if predict is None:
            raise TypeError("LadderEI requires a surrogate with a predict() method")
        X_cand = np.asarray(X_candidates, dtype=np.float64)
        mean_t, std_t = predict(X_cand)
        mean = torch.as_tensor(mean_t, dtype=torch.float64).reshape(-1)
        std = torch.as_tensor(std_t, dtype=torch.float64).reshape(-1)
        if y_best is None:
            y_best = float(mean.max().item())
        y_best_t = torch.tensor(float(y_best), dtype=torch.float64)
        improvement = mean - y_best_t
        Z = improvement / torch.clamp(std, min=1e-12)
        ei = std * (Z * _normal_cdf(Z) + _normal_pdf(Z))
        zero_std = std < 1e-12
        ei[zero_std] = torch.clamp(improvement[zero_std], min=0.0)
        best_idx = int(torch.argmax(ei).item())
        return X_cand[best_idx].copy(), float(ei[best_idx].item())


# ── TPE (self-contained Tree-structured Parzen Estimator) ───────────────────


@final
class TPESurrogate:
    """Tree-structured Parzen Estimator surrogate (``numpy`` only).

    Splits observations into a "good" set (top ``gamma`` fraction by reward,
    since the controller *maximises*) and a "bad" set, fits a Gaussian KDE to
    each, and predicts the log density ratio ``log l(x) - log g(x)`` as the mean
    (with a constant std).  Driven by :class:`LadderEI`, the controller then
    selects the candidate maximising the TPE acquisition ``l(x) / g(x)``.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.25,
        min_bandwidth: float = 0.05,
        seed: int | None = 0,
    ) -> None:
        self.gamma = float(gamma)
        self.min_bandwidth = float(min_bandwidth)
        self.seed = seed
        self._good: NDArray[np.float64] | None = None
        self._bad: NDArray[np.float64] | None = None
        self._h_good: NDArray[np.float64] | None = None
        self._h_bad: NDArray[np.float64] | None = None
        self._d: int = 0
        self._n: int = 0

    def _bandwidth(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        n = max(len(A), 1)
        std = np.std(A, axis=0)
        factor = float(n) ** (-1.0 / (self._d + 4))
        h = std * factor
        return np.clip(h, self.min_bandwidth, None)

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        self._d = Xa.shape[1]
        self._n = len(ya)
        k = max(1, int(np.ceil(self.gamma * self._n)))
        order = np.argsort(ya)[::-1]  # descending: best reward first
        good = Xa[order[:k]]
        bad = Xa[order[k:]] if k < self._n else Xa[order[:k]]
        self._good = good
        self._bad = bad
        self._h_good = self._bandwidth(good)
        self._h_bad = self._bandwidth(bad)

    def is_fitted(self) -> bool:
        return self._good is not None

    @property
    def n_train(self) -> int:
        return self._n

    def _log_kde(
        self,
        X: NDArray[np.float64],
        centers: NDArray[np.float64],
        h: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        diff = (X[:, None, :] - centers[None, :, :]) / h[None, None, :]
        log_phi = -0.5 * diff**2 - np.log(h)[None, None, :] - 0.5 * math.log(2 * math.pi)
        log_per_center = np.sum(log_phi, axis=2)  # [m, c]
        return _logsumexp(log_per_center, axis=1) - math.log(centers.shape[0])

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._good is None or self._bad is None:
            raise RuntimeError("TPESurrogate not fitted. Call fit() first.")
        assert self._h_good is not None and self._h_bad is not None
        Xa = np.asarray(X_test, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        log_l = self._log_kde(Xa, self._good, self._h_good)
        log_g = self._log_kde(Xa, self._bad, self._h_bad)
        score = log_l - log_g
        mean = torch.as_tensor(np.asarray(score, dtype=np.float64), dtype=torch.float64)
        std = torch.ones(len(Xa), dtype=torch.float64)
        return mean, std

    def log_marginal_likelihood(self) -> float:
        return -math.inf  # density-ratio model has no closed-form evidence


# ── Random forest surrogate (sklearn; SMAC's own model, dependency-free) ────


@final
class RandomForestSurrogate:
    """Random-forest surrogate with tree-variance uncertainty (``sklearn``).

    Mean is the forest prediction; std is the standard deviation across the
    per-tree predictions (the model SMAC itself uses internally).  Robust to
    noise and to mixed/discrete encodings, which is why the router prefers it for
    noisy low-budget runs.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        seed: int | None = 0,
        min_std: float = 1e-6,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.seed = seed
        self.min_std = float(min_std)
        self._rf: Any = None
        self._n: int = 0

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor  # pyright: ignore[reportMissingTypeStubs]

        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed
        )
        rf.fit(Xa, ya)
        self._rf = rf
        self._n = len(ya)

    def is_fitted(self) -> bool:
        return self._rf is not None

    @property
    def n_train(self) -> int:
        return self._n

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rf is None:
            raise RuntimeError("RandomForestSurrogate not fitted. Call fit() first.")
        Xa = np.asarray(X_test, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        estimators = getattr(self._rf, "estimators_")
        per_tree = np.stack([est.predict(Xa) for est in estimators], axis=0)
        mean = np.mean(per_tree, axis=0)
        std = np.clip(np.std(per_tree, axis=0), self.min_std, None)
        return (
            torch.as_tensor(np.asarray(mean, dtype=np.float64), dtype=torch.float64),
            torch.as_tensor(np.asarray(std, dtype=np.float64), dtype=torch.float64),
        )

    def log_marginal_likelihood(self) -> float:
        return -math.inf  # forests have no closed-form marginal likelihood


# ── BOCS (binary) — sparse Bayesian linear regression over monomials ────────


@final
class BOCSSurrogate:
    """Bayesian Optimisation of Combinatorial Structures surrogate (binary).

    Fits Bayesian linear regression over the monomial features
    ``[1, x_i, x_i x_j (i<j)]`` of a ``{0, 1}`` vector with an isotropic Gaussian
    prior ``N(0, prior_var I)`` and Gaussian noise.  The posterior gives a closed
    form predictive mean/variance, so :class:`LadderEI` performs proper BO.  This
    is the lightweight, dependency-free realisation of the BOCS surrogate (the
    original uses a horseshoe prior + an SDP acquisition; here we use a Gaussian
    prior + EI over the candidate pool).
    """

    def __init__(
        self,
        *,
        order: int = 2,
        prior_var: float = 1.0,
        noise_var: float = 1e-2,
        max_dim_for_pairwise: int = 24,
    ) -> None:
        self.order = int(order)
        self.prior_var = float(prior_var)
        self.noise_var = float(noise_var)
        self.max_dim_for_pairwise = int(max_dim_for_pairwise)
        self._mu: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._d: int = 0
        self._n: int = 0

    def _features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        n, d = Xa.shape
        blocks: list[NDArray[np.float64]] = [np.ones((n, 1), dtype=np.float64), Xa]
        if self.order >= 2 and d <= self.max_dim_for_pairwise:
            pairs = [
                (Xa[:, i] * Xa[:, j]).reshape(-1, 1)
                for i in range(d)
                for j in range(i + 1, d)
            ]
            if pairs:
                blocks.append(np.hstack(pairs))
        return np.hstack(blocks)

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        self._d = Xa.shape[1]
        self._n = len(ya)
        phi = self._features(Xa)
        p = phi.shape[1]
        a_mat = phi.T @ phi / self.noise_var + np.eye(p) / self.prior_var
        sigma = np.linalg.inv(a_mat)
        self._sigma = sigma
        self._mu = sigma @ phi.T @ ya / self.noise_var

    def is_fitted(self) -> bool:
        return self._mu is not None

    @property
    def n_train(self) -> int:
        return self._n

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mu is None or self._sigma is None:
            raise RuntimeError("BOCSSurrogate not fitted. Call fit() first.")
        phi = self._features(np.asarray(X_test, dtype=np.float64))
        mean = phi @ self._mu
        var = np.einsum("ij,jk,ik->i", phi, self._sigma, phi) + self.noise_var
        std = np.sqrt(np.clip(var, 1e-12, None))
        return (
            torch.as_tensor(np.asarray(mean, dtype=np.float64), dtype=torch.float64),
            torch.as_tensor(np.asarray(std, dtype=np.float64), dtype=torch.float64),
        )

    def log_marginal_likelihood(self) -> float:
        return -math.inf


# ── COMBO (pure discrete) — diffusion/RBF kernel GP over the encoding ───────


@final
class COMBOSurrogate:
    """Combinatorial-BO surrogate: a diffusion-style (RBF) kernel GP.

    Wraps :class:`_KernelGP` with a squared-exponential kernel restricted to the
    discrete columns — the continuous relaxation of COMBO's combinatorial-graph
    diffusion kernel.  Suitable for pure-discrete (categorical / ordinal / bool)
    spaces of modest dimensionality.
    """

    def __init__(
        self,
        *,
        discrete_cols: list[int] | None = None,
        lengthscale: float = 0.5,
        signal_var: float = 1.0,
        noise_var: float = 1e-4,
    ) -> None:
        self.discrete_cols = discrete_cols
        self.lengthscale = float(lengthscale)
        cols = discrete_cols
        ls = self.lengthscale

        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if cols is not None:
                a = a[:, cols]
                b = b[:, cols]
            return _rbf_corr(a, b, ls)

        self._gp = _KernelGP(kernel, signal_var=signal_var, noise_var=noise_var)

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        self._gp.fit(X, y)

    def is_fitted(self) -> bool:
        return self._gp.is_fitted()

    @property
    def n_train(self) -> int:
        return self._gp.n_train

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gp.predict(X_test)

    def log_marginal_likelihood(self) -> float:
        return self._gp.log_marginal_likelihood()


# ── CoCaBO / HyBO (mixed) — product Matern(continuous) x Matern(discrete) ────


@final
class MixedGPSurrogate:
    """Mixed discrete+continuous GP with a product kernel (CoCaBO / HyBO).

    The kernel is ``Matern52(x_cont; l_cont) * Matern52(x_disc; l_disc)`` — the
    product of a continuous kernel and a (short-lengthscale) categorical-overlap
    kernel, which is the structure used by CoCaBO and HyBO for mixed spaces.
    Both factors return unit-diagonal correlations, so the product is a valid PSD
    kernel handled by :class:`_KernelGP`.
    """

    def __init__(
        self,
        *,
        continuous_cols: list[int],
        discrete_cols: list[int],
        lengthscale_continuous: float = 0.5,
        lengthscale_discrete: float = 0.3,
        signal_var: float = 1.0,
        noise_var: float = 1e-4,
    ) -> None:
        self.continuous_cols = list(continuous_cols)
        self.discrete_cols = list(discrete_cols)
        self.lengthscale_continuous = float(lengthscale_continuous)
        self.lengthscale_discrete = float(lengthscale_discrete)
        cont = self.continuous_cols
        disc = self.discrete_cols
        ls_c = self.lengthscale_continuous
        ls_d = self.lengthscale_discrete

        def kernel(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
            out = torch.ones((X1.shape[0], X2.shape[0]), dtype=X1.dtype)
            if cont:
                out = out * _matern52_corr(X1[:, cont], X2[:, cont], ls_c)
            if disc:
                out = out * _matern52_corr(X1[:, disc], X2[:, disc], ls_d)
            return out

        self._gp = _KernelGP(kernel, signal_var=signal_var, noise_var=noise_var)

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        self._gp.fit(X, y)

    def is_fitted(self) -> bool:
        return self._gp.is_fitted()

    @property
    def n_train(self) -> int:
        return self._gp.n_train

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gp.predict(X_test)

    def log_marginal_likelihood(self) -> float:
        return self._gp.log_marginal_likelihood()


# ── Dictionary embedding (high-dim discrete) — random embed + Matern GP ─────


@final
class DictionaryEmbeddingSurrogate:
    """High-dimensional discrete surrogate via a fixed random *dictionary* embed.

    Projects the high-dim discrete encoding through a fixed (seeded) Gaussian
    dictionary ``R in R^{d x k}`` into a low-dim space, then fits a Matern-5/2
    :class:`~opop.controller.gp.GaussianProcess` on the embedding.  This is the
    random-embedding approach (HeSBO / dictionary BO) for spaces whose ambient
    dimensionality is too high for a direct GP.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 8,
        seed: int | None = 0,
        lengthscale: float = 0.5,
        signal_var: float = 1.0,
        noise_var: float = 1e-4,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.seed = seed
        self._proj: NDArray[np.float64] | None = None
        self._gp = GaussianProcess(
            lengthscale=lengthscale, signal_var=signal_var, noise_var=noise_var
        )

    def _project(self, X: NDArray[np.float64] | torch.Tensor) -> NDArray[np.float64]:
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        if self._proj is None:
            rng = np.random.default_rng(self.seed)
            k = min(self.embed_dim, Xa.shape[1])
            self._proj = rng.standard_normal((Xa.shape[1], k)) / math.sqrt(max(k, 1))
        return Xa @ self._proj

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        Z = self._project(np.asarray(X, dtype=np.float64))
        self._gp.fit(Z, y)

    def is_fitted(self) -> bool:
        return self._gp.is_fitted()

    @property
    def n_train(self) -> int:
        return self._gp.n_train

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gp.predict(self._project(X_test))

    def log_marginal_likelihood(self) -> float:
        return self._gp.log_marginal_likelihood()


# ── SMAC3 ask-tell wrapper (censored-aware) ─────────────────────────────────


@final
class SMACSurrogate:
    """SMAC3 ask-tell wrapper over the ``[0, 1]^d`` encoded space (censored-aware).

    Builds a :class:`smac.HyperparameterOptimizationFacade` (RF surrogate +
    logEI internally) over ``d`` ``Float(0, 1)`` hyperparameters and exposes a
    native ask-tell loop:

    * :meth:`ask` returns the next encoded vector to evaluate.
    * :meth:`tell` records the (negated, since SMAC minimises) reward.  Pass
      ``censored=True`` to record a right-censored / timed-out trial as a
      :class:`~smac.runhistory.dataclasses.TrialValue` with
      ``status=StatusType.TIMEOUT``.

    The facade is created with ``overwrite=True`` so repeated constructions in a
    fresh output directory never resume a stale run.  ``smac`` and ``ConfigSpace``
    are imported lazily so the module stays importable without them; the matching
    :class:`Surrogate`-protocol methods (:meth:`fit` / :meth:`predict`) are thin
    shims — SMAC maintains its own surrogate via :meth:`tell`.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_trials: int = 50,
        seed: int = 0,
        deterministic: bool = True,
        output_directory: str | None = None,
    ) -> None:
        try:
            from ConfigSpace import (  # pyright: ignore[reportMissingImports]
                ConfigurationSpace,
                Float,
            )
            from smac import (  # pyright: ignore[reportMissingImports]
                HyperparameterOptimizationFacade,
                Scenario,
            )
        except ImportError as exc:  # pragma: no cover - exercised only w/o smac
            raise ImportError(
                "SMACSurrogate requires the optional 'smac' and 'ConfigSpace' "
                + "packages; install smac to use this rung"
            ) from exc

        import tempfile
        from pathlib import Path

        self.dim = int(dim)
        self.seed = int(seed)
        cs: Any = ConfigurationSpace(seed=seed)
        cs.add([Float(f"x{i}", (0.0, 1.0)) for i in range(self.dim)])
        out_dir = output_directory or tempfile.mkdtemp(prefix="opop_smac_")
        scenario = Scenario(
            cs,
            deterministic=deterministic,
            n_trials=n_trials,
            output_directory=Path(out_dir),
            seed=seed,
        )
        self._facade: Any = HyperparameterOptimizationFacade(
            scenario,
            target_function=None,
            overwrite=True,
        )
        self._cs: Any = cs
        self._pending: dict[tuple[float, ...], Any] = {}
        self._n_told = 0

    def _key(self, x: NDArray[np.float64]) -> tuple[float, ...]:
        rounded: list[float] = np.round(np.asarray(x, dtype=np.float64), 12).tolist()
        return tuple(rounded)

    def ask(self) -> NDArray[np.float64]:
        """Ask SMAC for the next trial; return the encoded ``[0, 1]^d`` vector."""
        info = self._facade.ask()
        config = info.config
        x = np.array(
            [float(config[f"x{i}"]) for i in range(self.dim)], dtype=np.float64
        )
        self._pending[self._key(x)] = info
        return x

    def tell(
        self,
        x: NDArray[np.float64],
        reward: float,
        *,
        censored: bool = False,
        time: float = 0.0,
    ) -> None:
        """Tell SMAC the reward for ``x`` (negated: SMAC minimises cost).

        ``censored=True`` records the trial with ``StatusType.TIMEOUT`` so the
        SMAC RF surrogate treats the runtime/objective as right-censored.
        """
        from smac.runhistory.dataclasses import (  # pyright: ignore[reportMissingImports]
            TrialInfo,
            TrialValue,
        )
        from smac.runhistory.enumerations import (  # pyright: ignore[reportMissingImports]
            StatusType,
        )

        info = self._pending.pop(self._key(x), None)
        if info is None:
            from ConfigSpace import Configuration  # pyright: ignore[reportMissingImports]

            cfg = Configuration(
                self._cs,
                values={f"x{i}": float(x[i]) for i in range(self.dim)},
            )
            info = TrialInfo(cfg, seed=self.seed)
        status = StatusType.TIMEOUT if censored else StatusType.SUCCESS
        value = TrialValue(cost=-float(reward), time=float(time), status=status)
        self._facade.tell(info, value)
        self._n_told += 1

    @property
    def runhistory(self) -> Any:
        """The SMAC ``RunHistory`` (trial keys -> ``TrialValue``)."""
        return self._facade.runhistory

    # ── Surrogate-protocol shims (SMAC fits its own model on tell) ──────────

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        """Batch-tell ``(X, y)`` as successful trials (warmstart helper)."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        for row, val in zip(Xa, ya):
            self.tell(row, float(val))

    def is_fitted(self) -> bool:
        return self._n_told > 0

    @property
    def n_train(self) -> int:
        return self._n_told

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del X_test
        raise NotImplementedError(
            "SMACSurrogate uses SMAC's native ask/tell loop; it has no batch "
            + "predict(). Use ask()/tell() directly."
        )

    def log_marginal_likelihood(self) -> float:
        return -math.inf


# ── Space-shape analysis + router ───────────────────────────────────────────


@dataclass(frozen=True)
class SpaceShape:
    """Structural summary of a :class:`~opop.controller.encoder.Phase1Space`.

    Attributes:
        dim: Total encoded vector width.
        n_bool / n_categorical / n_ordinal / n_continuous: per-kind FIELD counts.
        column_kinds: per-column kind tag (length ``dim``).
        continuous_cols / discrete_cols: column indices by kind.
        is_boolean: every searched field is a :class:`BoolDim`.
        is_pure_discrete: has discrete columns and no continuous columns.
        is_mixed: has both discrete and continuous columns.
        is_high_dim: ``dim >= HIGH_DIM_THRESHOLD``.
    """

    dim: int
    n_bool: int
    n_categorical: int
    n_ordinal: int
    n_continuous: int
    column_kinds: tuple[str, ...]
    continuous_cols: tuple[int, ...]
    discrete_cols: tuple[int, ...]
    is_boolean: bool
    is_pure_discrete: bool
    is_mixed: bool
    is_high_dim: bool


def analyze_space(phi_space: Phase1Space) -> SpaceShape:
    """Inspect a :class:`Phase1Space` and summarise its shape for routing."""
    n_bool = n_categorical = n_ordinal = n_continuous = 0
    column_kinds: list[str] = []
    for d in phi_space.dims:
        if isinstance(d, BoolDim):
            n_bool += 1
            column_kinds.append("bool")
        elif isinstance(d, CategoricalDim):
            n_categorical += 1
            column_kinds.extend(["categorical"] * d.width)
        elif isinstance(d, OrdinalDim):
            n_ordinal += 1
            column_kinds.append("ordinal")
        elif isinstance(d, ContinuousDim):
            n_continuous += 1
            column_kinds.append("continuous")
        else:  # ContinuousDictDim (the only remaining Dim variant)
            n_continuous += 1
            column_kinds.extend(["continuous"] * d.width)

    dim = len(column_kinds)
    continuous_cols = tuple(i for i, k in enumerate(column_kinds) if k == "continuous")
    discrete_cols = tuple(i for i, k in enumerate(column_kinds) if k != "continuous")
    has_continuous = len(continuous_cols) > 0
    has_discrete = len(discrete_cols) > 0
    is_boolean = n_bool > 0 and n_categorical == 0 and n_ordinal == 0 and n_continuous == 0
    is_pure_discrete = has_discrete and not has_continuous
    is_mixed = has_discrete and has_continuous
    is_high_dim = dim >= HIGH_DIM_THRESHOLD
    return SpaceShape(
        dim=dim,
        n_bool=n_bool,
        n_categorical=n_categorical,
        n_ordinal=n_ordinal,
        n_continuous=n_continuous,
        column_kinds=tuple(column_kinds),
        continuous_cols=continuous_cols,
        discrete_cols=discrete_cols,
        is_boolean=is_boolean,
        is_pure_discrete=is_pure_discrete,
        is_mixed=is_mixed,
        is_high_dim=is_high_dim,
    )


@dataclass(frozen=True)
class RungChoice:
    """A selected ladder rung: a name, factories and the routing rationale.

    Attributes:
        name: Stable rung identifier (e.g. ``"bocs"``, ``"mixed_gp"``).
        make_surrogate: Zero-arg factory for the rung's surrogate.
        make_acquisition: Zero-arg factory for the rung's acquisition policy.
        reason: Human-readable justification for the choice.
        requires: Optional package dependencies (``()`` for self-contained rungs).
    """

    name: str
    make_surrogate: Callable[[], object]
    make_acquisition: Callable[[], object]
    reason: str
    requires: tuple[str, ...] = field(default=())

    def build(self) -> tuple[object, object]:
        """Instantiate ``(surrogate, acquisition)`` for this rung."""
        return self.make_surrogate(), self.make_acquisition()


def _qlognei_choice(reason: str) -> RungChoice:
    """Default rung: BoTorch ``qLogNoisyExpectedImprovement`` over a SingleTaskGP."""

    def make_surrogate() -> object:
        from .botorch_rungs import BoTorchGPSurrogate

        return BoTorchGPSurrogate()

    def make_acquisition() -> object:
        from .botorch_rungs import QLogNEIAcquisition

        return QLogNEIAcquisition()

    return RungChoice(
        name="qlognei",
        make_surrogate=make_surrogate,
        make_acquisition=make_acquisition,
        reason=reason,
        requires=("botorch",),
    )


def select_surrogate(
    phi_space: Phase1Space,
    budget: int,
    noise: bool | float = False,
) -> RungChoice:
    """Route a :class:`Phase1Space` + budget + noise to a ladder rung.

    Routing policy (evidence-first ladder ``random -> SMAC/TPE -> qLogNEI``,
    structured only when shape **and** budget justify it):

    * ``budget < MIN_MODEL_BUDGET`` -> the cheap tier: ``random_forest`` when the
      observations are noisy (SMAC's own RF model, robust to noise), else
      ``tpe``.
    * ``budget >= MIN_STRUCTURED_BUDGET`` and a structured shape:
        - high-dim pure-discrete -> ``dictionary_embedding``,
        - boolean -> ``bocs``,
        - other pure-discrete -> ``combo``,
        - mixed discrete+continuous -> ``mixed_gp`` (CoCaBO / HyBO).
    * otherwise -> ``qlognei`` (BoTorch ``qLogNoisyExpectedImprovement``).

    Args:
        phi_space: The encoded Phase-1 search space.
        budget: Total trial budget.
        noise: Observation noise (``bool`` or magnitude); ``> 0`` means noisy.

    Returns:
        A :class:`RungChoice` naming the rung with zero-arg factories.
    """
    shape = analyze_space(phi_space)
    budget = int(budget)
    is_noisy = float(noise) > 0.0

    # Cheap tier: not enough budget to amortise a full surrogate model.
    if budget < MIN_MODEL_BUDGET:
        if is_noisy:
            return RungChoice(
                name="random_forest",
                make_surrogate=lambda: RandomForestSurrogate(seed=0),
                make_acquisition=LadderEI,
                reason=(
                    f"budget {budget} < {MIN_MODEL_BUDGET} and noisy: random-forest "
                    "(SMAC-family) rung, robust to noise at low budget"
                ),
            )
        return RungChoice(
            name="tpe",
            make_surrogate=lambda: TPESurrogate(seed=0),
            make_acquisition=LadderEI,
            reason=(
                f"budget {budget} < {MIN_MODEL_BUDGET}: cheap TPE density-ratio rung "
                "before committing to a full GP"
            ),
        )

    if budget >= MIN_STRUCTURED_BUDGET:
        if shape.is_pure_discrete and shape.is_high_dim:
            embed_dim = min(8, max(2, shape.dim // 4))
            return RungChoice(
                name="dictionary_embedding",
                make_surrogate=lambda: DictionaryEmbeddingSurrogate(
                    embed_dim=embed_dim, seed=0
                ),
                make_acquisition=LadderEI,
                reason=(
                    f"high-dim pure-discrete space (dim={shape.dim} >= "
                    f"{HIGH_DIM_THRESHOLD}): random dictionary embedding + GP"
                ),
            )
        if shape.is_boolean:
            return RungChoice(
                name="bocs",
                make_surrogate=lambda: BOCSSurrogate(),
                make_acquisition=LadderEI,
                reason=(
                    f"boolean space ({shape.n_bool} bool fields): BOCS sparse "
                    "Bayesian linear regression over binary monomials"
                ),
            )
        if shape.is_pure_discrete:
            disc = list(shape.discrete_cols)
            return RungChoice(
                name="combo",
                make_surrogate=lambda: COMBOSurrogate(discrete_cols=disc),
                make_acquisition=LadderEI,
                reason=(
                    f"pure-discrete space (dim={shape.dim}): COMBO diffusion-kernel GP"
                ),
            )
        if shape.is_mixed:
            cont = list(shape.continuous_cols)
            disc = list(shape.discrete_cols)
            return RungChoice(
                name="mixed_gp",
                make_surrogate=lambda: MixedGPSurrogate(
                    continuous_cols=cont, discrete_cols=disc
                ),
                make_acquisition=LadderEI,
                reason=(
                    f"mixed space ({len(cont)} continuous + {len(disc)} discrete "
                    "columns): CoCaBO/HyBO product-kernel GP"
                ),
            )

    return _qlognei_choice(
        reason=(
            f"default rung (dim={shape.dim}, budget={budget}, noisy={is_noisy}): "
            "BoTorch qLogNoisyExpectedImprovement over a SingleTaskGP"
        )
    )
