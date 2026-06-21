# How to add a surrogate / acquisition

The controller climbs a **ladder** of Bayesian-optimization rungs
(random → GP+EI → SMAC/TPE/RF → structured BO) behind two Protocols, so a new
surrogate or acquisition policy drops in without changing any caller. This guide
shows how to implement each Protocol and plug it into `Phase1Controller`.

Both Protocols live in `opop.controller.protocol` and are re-exported as
`opop.Surrogate` and `opop.Acquisition`.

## 1. The `Surrogate` Protocol

A surrogate is a probabilistic model the controller fits to observed
`(encoded Phi, reward)` pairs and queries for posterior predictions:

```python
import numpy as np
import torch
from numpy.typing import NDArray

class MySurrogate:
    def fit(
        self,
        X: NDArray[np.float64] | torch.Tensor | list[float],
        y: NDArray[np.float64] | torch.Tensor | list[float],
    ) -> None:
        """Fit the surrogate to (X, y) observations."""

    def predict(
        self, X_test: NDArray[np.float64] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) posterior predictives for X_test."""

    def is_fitted(self) -> bool:
        """Return whether the surrogate has been fit."""

    def log_marginal_likelihood(self) -> float:
        """Return the training-data log marginal likelihood (if available)."""
```

The reference implementation is `opop.GaussianProcess` (a Matérn-5/2 GP with
Cholesky inference and a pseudoinverse fallback). Because the Protocol is
`@runtime_checkable`, `isinstance(MySurrogate(), opop.Surrogate)` holds once these
methods exist.

## 2. The `Acquisition` Protocol

An acquisition policy scores a finite candidate pool and returns the chosen
candidate:

```python
class MyAcquisition:
    def __call__(
        self,
        surrogate: Surrogate,
        X_candidates: NDArray[np.float64],   # [n_candidates, n_features]
        *,
        y_best: float | None = None,         # best observed value (improvement rules)
        kappa: float = 2.0,                  # exploration weight (UCB-style)
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:  # (selected_config, acquisition_value)
        ...
```

Reference policies: `opop.EI` (Expected Improvement), `opop.UCB`
(Upper-Confidence-Bound), and `opop.RandomSearch` (the baseline — it ignores the
surrogate, which lets the same controller run with no model at all).

> Selection is always over a **finite candidate pool** (the proposer's typed
> candidates), so the controller never proposes an out-of-space configuration.

## 3. Plug into the controller

`Phase1Controller` takes a search space, an acquisition policy, and an optional
surrogate (pure ask-tell; the surrogate is refit after every `tell`):

```python
import opop

space = opop.default_phase1_space()

controller = opop.Phase1Controller(
    space,
    MyAcquisition(),
    surrogate=MySurrogate(),   # omit (None) for a surrogate-free policy like RandomSearch
    n_trials=20,
    n_init=3,
    n_candidates=64,
    seed=0,
)

# Convenience factories for the two Phase-1 rungs:
bo     = opop.Phase1Controller.bo(space, n_trials=20, n_init=3, n_candidates=64, seed=0)
random = opop.Phase1Controller.random(space, n_trials=20, n_init=3, n_candidates=64, seed=0)
```

The ask-tell loop is `ask()` → evaluate → `tell(phi, reward)`; `run(evaluator)`
loops to budget. Encodings are normalised to `[0, 1]`, so a single-lengthscale GP
sees comparable scales across categorical / ordinal / bool / continuous fields.

### Where your rung sits on the ladder
1. `RandomSearch` — always-available baseline.
2. `GaussianProcess` + `EI`/`UCB` — the Phase-1 default (`.bo(...)`).
3. SMAC / TPE / RF — censored-aware SMBO (Wave-4), same Protocols.
4. Structured BO — BOCS (binary) / COMBO (discrete) / CoCaBO/HyBO (mixed) /
   dictionary-embedding (high-dim) / BoTorch `qLogNoisyExpectedImprovement`.

A `select_surrogate(phi_space, budget, noise)` router picks the rung by space
shape and budget; multi-fidelity BO activates **only** after a
fidelity-correlation study passes Spearman ρ ≥ 0.5 (see
[architecture.md](architecture.md)).

## 4. Test it

Under `tests/controller/`: assert your acquisition beats (or ties) random on a
toy objective within a small budget, that the surrogate's posterior variance
shrinks near sampled points, and that `isinstance(MySurrogate(), opop.Surrogate)`
and `isinstance(MyAcquisition(), opop.Acquisition)` hold. Trick for an
apples-to-apples comparison: build **one** fixed candidate pool and pass it to
both your controller and a `RandomSearch` controller. Keep `ruff`, `mypy`, and
`lsp_diagnostics` clean.
