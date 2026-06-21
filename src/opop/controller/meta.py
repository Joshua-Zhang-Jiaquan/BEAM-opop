"""Simplified Reptile / MAML meta-tuner for GP-hyperparameter warm-start (task 32).

Meta-learns a shared initialisation of the :class:`~opop.controller.gp.GaussianProcess`
log-hyperparameters (lengthscale / signal variance / noise variance) across a
collection of historical ``(X, y)`` designs, so a fresh controller can start
from a GP whose kernel is already adapted to the task *distribution* rather than
the generic defaults. Pairs with :mod:`opop.controller.transfer`: meta-learn the
hyperparameters, build the GP via :meth:`MetaTuner.build_gp`, hand it to a
:class:`~opop.controller.phase1.Phase1Controller` as its surrogate, then
warm-start the controller's observation buffer from related historical tasks.

The kernel matches :class:`~opop.controller.gp.GaussianProcess` exactly (Matern
5/2), so meta-learned hyperparameters transfer directly. Two update rules are
provided: ``"reptile"`` (first-order, cheap) and ``"maml"`` (second-order through
the inner loop). Both are deterministic given the design order and inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .gp import GaussianProcess

__all__ = ["MetaTuner"]


def _pairwise_dist(points: torch.Tensor) -> torch.Tensor:
    """Euclidean pairwise distances with an in-sqrt epsilon (finite gradient at 0)."""
    diff = points.unsqueeze(0) - points.unsqueeze(1)
    return torch.sqrt((diff**2).sum(dim=-1) + 1e-12)


def _matern52_lml(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    log_lengthscale: torch.Tensor,
    log_signal_var: torch.Tensor,
    log_noise_var: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Matern-5/2 GP log marginal likelihood ``log p(y | X)``.

    Kernel: ``sv * (1 + r + r**2 / 3) * exp(-r)`` with ``r = sqrt(5) * ||x - x'|| /
    lengthscale`` plus ``noise_var`` on the diagonal — identical to
    :meth:`opop.controller.gp.GaussianProcess._matern52_kernel`. A small jitter
    keeps the Cholesky factorisation stable.
    """
    n = inputs.shape[0]
    dtype = inputs.dtype
    lengthscale = torch.exp(log_lengthscale)
    signal_var = torch.exp(log_signal_var)
    noise_var = torch.exp(log_noise_var)

    dist = _pairwise_dist(inputs / lengthscale)
    r = math.sqrt(5.0) * dist
    kernel = signal_var * (1.0 + r + r**2 / 3.0) * torch.exp(-r)
    eye = torch.eye(n, dtype=dtype, device=inputs.device)
    kernel = kernel + (noise_var + 1e-6) * eye

    chol = torch.as_tensor(torch.linalg.cholesky(kernel))
    log_det = 2.0 * torch.sum(torch.log(torch.diag(chol)))
    chol_inv_y = torch.linalg.solve_triangular(chol, targets.unsqueeze(-1), upper=False)
    quad = (chol_inv_y**2).sum()
    return -0.5 * (quad + log_det + n * math.log(2.0 * math.pi))


class MetaTuner:
    """Reptile/MAML meta-learner over GP log-hyperparameters.

    Args:
        lengthscale: Initial kernel lengthscale (``> 0``).
        signal_var: Initial signal variance (``> 0``).
        noise_var: Initial observation noise variance (``> 0``).
        dtype: Tensor dtype (defaults to ``float64`` to match the GP).
    """

    log_lengthscale: nn.Parameter
    log_signal_var: nn.Parameter
    log_noise_var: nn.Parameter
    dtype: torch.dtype

    def __init__(
        self,
        *,
        lengthscale: float = 1.0,
        signal_var: float = 1.0,
        noise_var: float = 1e-3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dtype = dtype
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(lengthscale), dtype=dtype)
        )
        self.log_signal_var = nn.Parameter(
            torch.tensor(math.log(signal_var), dtype=dtype)
        )
        self.log_noise_var = nn.Parameter(
            torch.tensor(math.log(noise_var), dtype=dtype)
        )
        self._losses: list[float] = []

    # ── public API ──────────────────────────────────────────────────────────

    def get_hyperparams(self) -> dict[str, float]:
        """Return the current meta-learned ``lengthscale``/``signal_var``/``noise_var``."""
        with torch.no_grad():
            return {
                "lengthscale": float(torch.exp(self.log_lengthscale).item()),
                "signal_var": float(torch.exp(self.log_signal_var).item()),
                "noise_var": float(torch.exp(self.log_noise_var).item()),
            }

    def build_gp(self) -> GaussianProcess:
        """Construct a :class:`GaussianProcess` initialised with the meta hyperparameters."""
        hp = self.get_hyperparams()
        return GaussianProcess(
            lengthscale=hp["lengthscale"],
            signal_var=hp["signal_var"],
            noise_var=hp["noise_var"],
            dtype=self.dtype,
        )

    def meta_train(
        self,
        designs: Sequence[tuple[Any, Any]],
        *,
        mode: str = "reptile",
        n_inner_steps: int = 5,
        inner_lr: float = 0.01,
        meta_lr: float = 0.1,
    ) -> list[float]:
        """Meta-train over ``designs`` (each a ``(X, y)`` pair); return inner losses.

        Args:
            designs: Historical designs, each an ``[n, d]`` input array and an
                ``[n]`` target array (numpy or torch).
            mode: ``"reptile"`` (first-order) or ``"maml"`` (second-order).
            n_inner_steps: Inner adaptation steps per design.
            inner_lr: Inner-loop step size.
            meta_lr: Outer (meta) step size.
        """
        if mode not in ("reptile", "maml"):
            raise ValueError(f"unknown mode {mode!r}; use 'reptile' or 'maml'")
        self._losses = []
        for inputs_raw, targets_raw in designs:
            inputs = torch.as_tensor(inputs_raw, dtype=self.dtype)
            targets = torch.as_tensor(targets_raw, dtype=self.dtype).reshape(-1)
            if inputs.shape[0] < 2:
                continue
            try:
                if mode == "reptile":
                    self._reptile_step(inputs, targets, n_inner_steps, inner_lr, meta_lr)
                else:
                    self._maml_step(inputs, targets, n_inner_steps, inner_lr, meta_lr)
            except RuntimeError:
                self._losses.append(float("nan"))
        return list(self._losses)

    # ── update rules ──────────────────────────────────────────────────────────

    def _reptile_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_inner_steps: int,
        inner_lr: float,
        meta_lr: float,
    ) -> None:
        ls = self.log_lengthscale.detach().clone().requires_grad_(True)
        sv = self.log_signal_var.detach().clone().requires_grad_(True)
        nv = self.log_noise_var.detach().clone().requires_grad_(True)

        for _ in range(n_inner_steps):
            loss = -_matern52_lml(inputs, targets, ls, sv, nv)
            grads = torch.autograd.grad(loss, [ls, sv, nv])
            with torch.no_grad():
                ls -= inner_lr * grads[0]
                sv -= inner_lr * grads[1]
                nv -= inner_lr * grads[2]
            self._losses.append(float(loss.detach().item()))

        with torch.no_grad():
            self.log_lengthscale.add_(meta_lr * (ls.detach() - self.log_lengthscale))
            self.log_signal_var.add_(meta_lr * (sv.detach() - self.log_signal_var))
            self.log_noise_var.add_(meta_lr * (nv.detach() - self.log_noise_var))

    def _maml_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_inner_steps: int,
        inner_lr: float,
        meta_lr: float,
    ) -> None:
        ls = self.log_lengthscale.clone()
        sv = self.log_signal_var.clone()
        nv = self.log_noise_var.clone()

        for _ in range(n_inner_steps):
            loss = -_matern52_lml(inputs, targets, ls, sv, nv)
            grads = torch.autograd.grad(loss, [ls, sv, nv], create_graph=True)
            ls = ls - inner_lr * grads[0]
            sv = sv - inner_lr * grads[1]
            nv = nv - inner_lr * grads[2]
            self._losses.append(float(loss.detach().item()))

        meta_loss = -_matern52_lml(inputs, targets, ls, sv, nv)
        params = (self.log_lengthscale, self.log_signal_var, self.log_noise_var)
        meta_grads = torch.autograd.grad(meta_loss, list(params))
        with torch.no_grad():
            for param, grad in zip(params, meta_grads):
                param.sub_(meta_lr * grad)
