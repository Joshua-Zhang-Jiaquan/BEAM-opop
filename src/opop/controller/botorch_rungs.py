"""BoTorch acquisition rungs for the controller ladder (qLogNEI / qKG).

These wrappers plug BoTorch into the task-8
:class:`~opop.controller.protocol.Surrogate` /
:class:`~opop.controller.protocol.Acquisition` protocols:

* :class:`BoTorchGPSurrogate` — a ``SingleTaskGP`` behind the ``Surrogate``
  protocol (the default model for the qLogNEI / qKG acquisitions).
* :class:`BoTorchMixedGPSurrogate` — a ``MixedSingleTaskGP`` (CategoricalKernel x
  continuous kernel) for mixed spaces, declaring its ``cat_dims``.
* :class:`QLogNEIAcquisition` — the **default** acquisition: batch Log Noisy
  Expected Improvement (``qLogNoisyExpectedImprovement``), evaluated over the
  controller's finite candidate pool.
* :class:`QKnowledgeGradientAcquisition` — the **batch** acquisition: one-shot
  Knowledge Gradient (``qKnowledgeGradient``), optimised with ``optimize_acqf``
  and snapped to the nearest pool candidate (with a native :meth:`optimize`
  for true continuous ``q``-batches).

``botorch`` / ``gpytorch`` are imported lazily inside every method so this
module stays importable without them; instantiating any class raises a clear
:class:`ImportError` when BoTorch is absent.  Tests guard these rungs with
``pytest.importorskip("botorch")`` so they skip cleanly.

Only the *Log* EI variant is used (``qLogNoisyExpectedImprovement``) — never the
numerically-unstable non-log ``qNoisyExpectedImprovement`` / ``qExpectedImprovement``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast, final

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "BoTorchGPSurrogate",
    "BoTorchMixedGPSurrogate",
    "QKnowledgeGradientAcquisition",
    "QLogNEIAcquisition",
    "botorch_available",
]

_DTYPE = torch.float64


def botorch_available() -> bool:
    """Return whether ``botorch`` can be imported (without importing it eagerly)."""
    import importlib.util

    return importlib.util.find_spec("botorch") is not None


def _require_botorch() -> None:
    if not botorch_available():
        raise ImportError(
            "BoTorch rungs require the optional 'botorch' package; "
            + "install botorch (and gpytorch) for qLogNEI / qKnowledgeGradient"
        )


def _model_of(surrogate: object) -> Any:
    """Extract the underlying BoTorch model from a surrogate wrapper."""
    model = getattr(surrogate, "model", None)
    if model is None:
        model = getattr(surrogate, "_model", None)
    if model is None:
        raise TypeError(
            "BoTorch acquisitions require a fitted BoTorch surrogate exposing "
            + f"`.model` (e.g. BoTorchGPSurrogate); got {type(surrogate).__name__}"
        )
    return model


def _baseline_of(surrogate: object) -> torch.Tensor:
    """Extract the observed design matrix (``X_baseline``) from a surrogate."""
    base = getattr(surrogate, "train_X", None)
    if base is None:
        base = getattr(surrogate, "_train_X", None)
    if base is None:
        raise TypeError("BoTorch acquisitions require the surrogate's train_X")
    return torch.as_tensor(base, dtype=_DTYPE)


# ── SingleTaskGP surrogate ──────────────────────────────────────────────────


@final
class BoTorchGPSurrogate:
    """A ``SingleTaskGP`` behind the :class:`Surrogate` protocol.

    Maximisation-oriented: pass rewards directly as ``y`` (higher is better),
    matching BoTorch's convention and the qLogNEI / qKG acquisitions.
    """

    def __init__(self, *, dtype: torch.dtype = _DTYPE) -> None:
        _require_botorch()
        self.dtype = dtype
        self._model: Any = None
        self._train_X: torch.Tensor | None = None
        self._train_Y: torch.Tensor | None = None

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        from botorch.fit import fit_gpytorch_mll  # pyright: ignore[reportMissingImports]
        from botorch.models import SingleTaskGP  # pyright: ignore[reportMissingImports]
        from gpytorch.mlls import (  # pyright: ignore[reportMissingImports]
            ExactMarginalLogLikelihood,
        )

        X_t = torch.as_tensor(X, dtype=self.dtype)
        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        Y_t = torch.as_tensor(y, dtype=self.dtype).reshape(-1, 1)
        model: Any = SingleTaskGP(X_t, Y_t)
        mll: Any = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self._model = model
        self._train_X = X_t
        self._train_Y = Y_t

    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def n_train(self) -> int:
        return 0 if self._train_X is None else int(self._train_X.shape[0])

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("BoTorchGPSurrogate not fitted. Call fit() first.")
        return self._model

    @property
    def train_X(self) -> torch.Tensor:
        if self._train_X is None:
            raise RuntimeError("BoTorchGPSurrogate not fitted. Call fit() first.")
        return self._train_X

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model: Any = self.model
        X_t = torch.as_tensor(X_test, dtype=self.dtype)
        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        model.eval()
        with torch.no_grad():
            posterior: Any = model.posterior(X_t)
            mean = posterior.mean.reshape(-1)
            std = posterior.variance.clamp_min(1e-12).sqrt().reshape(-1)
        return mean, std

    def log_marginal_likelihood(self) -> float:
        if self._model is None or self._train_X is None or self._train_Y is None:
            return -math.inf
        try:
            from gpytorch.mlls import (  # pyright: ignore[reportMissingImports]
                ExactMarginalLogLikelihood,
            )

            model: Any = self._model
            mll: Any = ExactMarginalLogLikelihood(model.likelihood, model)
            model.train()
            output: Any = model(self._train_X)
            value = cast("torch.Tensor", mll(output, self._train_Y.reshape(-1)))
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            return -math.inf


# ── MixedSingleTaskGP surrogate (production mixed rung) ─────────────────────


@final
class BoTorchMixedGPSurrogate:
    """A ``MixedSingleTaskGP`` for mixed spaces (declares its ``cat_dims``).

    Uses BoTorch's ``MixedSingleTaskGP`` (a ``CategoricalKernel`` over the
    categorical columns combined with a continuous kernel), the production
    counterpart of :class:`opop.controller.ladder.MixedGPSurrogate`.  Pair it
    with :func:`botorch.optim.optimize_acqf_mixed` for continuous-batch mixed
    optimisation; over the controller's discrete pool the qLogNEI / qKG
    acquisitions work directly.
    """

    def __init__(self, *, cat_dims: list[int], dtype: torch.dtype = _DTYPE) -> None:
        _require_botorch()
        self.cat_dims = list(cat_dims)
        self.dtype = dtype
        self._model: Any = None
        self._train_X: torch.Tensor | None = None
        self._train_Y: torch.Tensor | None = None

    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        from botorch.fit import fit_gpytorch_mll  # pyright: ignore[reportMissingImports]
        from botorch.models import (  # pyright: ignore[reportMissingImports]
            MixedSingleTaskGP,
        )
        from gpytorch.mlls import (  # pyright: ignore[reportMissingImports]
            ExactMarginalLogLikelihood,
        )

        X_t = torch.as_tensor(X, dtype=self.dtype)
        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        Y_t = torch.as_tensor(y, dtype=self.dtype).reshape(-1, 1)
        model: Any = MixedSingleTaskGP(X_t, Y_t, cat_dims=self.cat_dims)
        mll: Any = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self._model = model
        self._train_X = X_t
        self._train_Y = Y_t

    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def n_train(self) -> int:
        return 0 if self._train_X is None else int(self._train_X.shape[0])

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("BoTorchMixedGPSurrogate not fitted. Call fit() first.")
        return self._model

    @property
    def train_X(self) -> torch.Tensor:
        if self._train_X is None:
            raise RuntimeError("BoTorchMixedGPSurrogate not fitted. Call fit() first.")
        return self._train_X

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model: Any = self.model
        X_t = torch.as_tensor(X_test, dtype=self.dtype)
        if X_t.ndim == 1:
            X_t = X_t.reshape(-1, 1)
        model.eval()
        with torch.no_grad():
            posterior: Any = model.posterior(X_t)
            mean = posterior.mean.reshape(-1)
            std = posterior.variance.clamp_min(1e-12).sqrt().reshape(-1)
        return mean, std

    def log_marginal_likelihood(self) -> float:
        return -math.inf


# ── qLogNoisyExpectedImprovement (default acquisition) ──────────────────────


@final
class QLogNEIAcquisition:
    """Batch Log Noisy EI over the candidate pool (the ladder default).

    Builds ``qLogNoisyExpectedImprovement(model, X_baseline)`` from the fitted
    surrogate and evaluates it on every candidate (as a ``q=1`` batch), returning
    the argmax pool member.  Robust to observation noise and numerically stable
    in the log domain (hence the *Log* variant).
    """

    def __init__(
        self, *, num_samples: int = 128, prune_baseline: bool = True, seed: int = 0
    ) -> None:
        _require_botorch()
        self.num_samples = int(num_samples)
        self.prune_baseline = bool(prune_baseline)
        self.seed = int(seed)

    def __call__(
        self,
        surrogate: object,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        del y_best, kappa  # qLogNEI infers the incumbent from X_baseline
        from botorch.acquisition import (  # pyright: ignore[reportMissingImports]
            qLogNoisyExpectedImprovement,
        )
        from botorch.sampling import (  # pyright: ignore[reportMissingImports]
            SobolQMCNormalSampler,
        )

        model: Any = _model_of(surrogate)
        x_baseline = _baseline_of(surrogate)
        sampler: Any = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.num_samples]),
            seed=self.seed if seed is None else seed,
        )
        acq: Any = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=x_baseline,
            sampler=sampler,
            prune_baseline=self.prune_baseline,
        )
        pool = torch.as_tensor(X_candidates, dtype=_DTYPE)
        if pool.ndim == 1:
            pool = pool.reshape(-1, 1)
        with torch.no_grad():
            values = cast("torch.Tensor", acq(pool.unsqueeze(1)))  # [n, 1, d] -> [n]
        best_idx = int(torch.argmax(values).item())
        return pool[best_idx].cpu().numpy(), float(values[best_idx].item())


# ── qKnowledgeGradient (batch acquisition) ──────────────────────────────────


@final
class QKnowledgeGradientAcquisition:
    """One-shot Knowledge Gradient acquisition (batch-capable).

    For the :class:`Acquisition` protocol (single proposal) it optimises
    ``qKnowledgeGradient`` over the ``[0, 1]^d`` cube with ``optimize_acqf`` and
    snaps the result to the nearest pool candidate (so the controller still
    proposes a valid encoded vector).  :meth:`optimize` exposes the true
    continuous ``q``-batch optimisation.
    """

    def __init__(
        self,
        *,
        num_fantasies: int = 64,
        num_restarts: int = 5,
        raw_samples: int = 128,
        seed: int = 0,
    ) -> None:
        _require_botorch()
        self.num_fantasies = int(num_fantasies)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.seed = int(seed)

    def _acq(self, surrogate: object) -> Any:
        from botorch.acquisition import (  # pyright: ignore[reportMissingImports]
            qKnowledgeGradient,
        )

        model: Any = _model_of(surrogate)
        return qKnowledgeGradient(model=model, num_fantasies=self.num_fantasies)

    def __call__(
        self,
        surrogate: object,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        del y_best, kappa, seed
        from botorch.optim import optimize_acqf  # pyright: ignore[reportMissingImports]

        acq: Any = self._acq(surrogate)
        pool = torch.as_tensor(X_candidates, dtype=_DTYPE)
        if pool.ndim == 1:
            pool = pool.reshape(-1, 1)
        d = int(pool.shape[1])
        bounds = torch.stack([torch.zeros(d, dtype=_DTYPE), torch.ones(d, dtype=_DTYPE)])
        candidate, value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        cand = cast("torch.Tensor", candidate).reshape(1, -1)
        dists = torch.cdist(cand, pool)
        best_idx = int(torch.argmin(dists).item())
        return pool[best_idx].cpu().numpy(), float(cast("torch.Tensor", value).item())

    def optimize(
        self, surrogate: object, *, q: int = 2, bounds: torch.Tensor | None = None
    ) -> tuple[NDArray[np.float64], float]:
        """Optimise qKG for a true continuous batch of ``q`` candidates."""
        from botorch.optim import optimize_acqf  # pyright: ignore[reportMissingImports]

        acq: Any = self._acq(surrogate)
        base = _baseline_of(surrogate)
        d = int(base.shape[1])
        if bounds is None:
            bounds = torch.stack(
                [torch.zeros(d, dtype=_DTYPE), torch.ones(d, dtype=_DTYPE)]
            )
        candidates, value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=int(q),
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        out: Any = candidates.detach().cpu().numpy()
        return out, float(cast("torch.Tensor", value).item())
