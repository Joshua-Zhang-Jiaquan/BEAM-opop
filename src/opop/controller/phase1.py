"""Phase-1 ask-tell controller over the restricted ``Phi`` subspace.

Wraps the task-8 Bayesian-optimization base (:class:`GaussianProcess` + EI) as
the BO baseline and :class:`RandomSearch` as the random baseline, both behind
the :class:`~opop.controller.protocol.Acquisition` /
:class:`~opop.controller.protocol.Surrogate` protocols, so Wave-4 controllers
(SMAC / TPE / BoTorch / structured surrogates, task 28) can be swapped in
without touching callers.

The controller maintains its posterior by re-fitting the surrogate on the
accumulated ``(X, y)`` observations after every :meth:`Phase1Controller.tell`,
chooses the next configuration via the acquisition value over a finite
candidate pool, and stops at ``n_trials`` (or an optional wall-clock budget).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from opop.model.state import Phi

from .encoder import Phase1Space
from .gp import GaussianProcess
from .protocol import EI, Acquisition, RandomSearch, Surrogate

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ── CO/IP reward scalarization ──────────────────────────────────────────────


class _HasToDict(Protocol):
    def to_dict(self) -> Mapping[str, float]: ...


def coip_reward(
    metrics: Mapping[str, float] | _HasToDict,
    *,
    w_gap: float = 1.0,
    w_time: float = 1e-3,
    w_pi: float = 1.0,
) -> float:
    """Scalarize CO/IP solve metrics into a reward (higher is better).

    ``reward = -w_gap * gap - w_time * time - w_pi * primal_integral``.

    This intentionally replaces the EDA-flavored
    :func:`~opop.controller.acquisition.scalarized_reward` defaults
    (wns / tns / power / area) with combinatorial-optimization metrics.

    Args:
        metrics: Mapping (or object with ``to_dict()``) carrying the keys
            ``gap``, ``time`` (or ``runtime_seconds`` / ``solve_time``), and
            ``primal_integral``.
        w_gap: penalty weight on the optimality gap.
        w_time: penalty weight on wall-clock solve time.
        w_pi: penalty weight on the primal integral.

    Returns:
        Scalar reward (higher is better).
    """
    m = metrics if isinstance(metrics, Mapping) else metrics.to_dict()
    gap = float(m.get("gap", 0.0))
    solve_time = float(
        m.get("time", m.get("runtime_seconds", m.get("solve_time", 0.0)))
    )
    primal_integral = float(m.get("primal_integral", 0.0))
    return -(w_gap * gap) - (w_time * solve_time) - (w_pi * primal_integral)


# ── Result container ────────────────────────────────────────────────────────


@dataclass
class Phase1Result:
    """Outcome of a Phase-1 ask-tell run.

    Attributes:
        best_phi: Configuration with the highest observed reward.
        best_reward: Highest observed reward.
        best_trace: Best-so-far reward after each tell (length == n_trials run).
        history: Ordered ``(phi, reward)`` observations.
        X: Encoded observation matrix, shape ``[n, d]``.
        y: Observed rewards, shape ``[n]``.
    """

    best_phi: Phi
    best_reward: float
    best_trace: list[float]
    history: list[tuple[Phi, float]]
    X: NDArray[np.float64]
    y: NDArray[np.float64]

    @property
    def n_trials(self) -> int:
        return len(self.history)


# ── Controller ──────────────────────────────────────────────────────────────


class Phase1Controller:
    """Ask-tell BO/random controller over a :class:`Phase1Space`.

    Args:
        space: Restricted Phase-1 subspace + its numeric encoding.
        acquisition: Acquisition policy (``EI()`` for BO, ``RandomSearch()`` for
            the random baseline).  Must satisfy the ``Acquisition`` protocol.
        surrogate: Probabilistic surrogate (a :class:`GaussianProcess` for BO);
            ``None`` for the random baseline (no posterior is maintained).
        n_trials: Total ask-tell rounds (the budget).
        n_init: Initial random design size before acquisition kicks in
            (defaults to ``min(5, n_trials)`` for BO, ``1`` when no surrogate).
        n_candidates: Size of the random candidate pool drawn per acquisition
            step when explicit candidates are not supplied.
        time_budget_s: Optional wall-clock budget for :meth:`run`.
        seed: RNG seed for the initial design and candidate pools.
    """

    space: Phase1Space
    acquisition: Acquisition
    surrogate: Surrogate | None
    n_trials: int
    n_init: int
    n_candidates: int
    time_budget_s: float | None

    def __init__(
        self,
        space: Phase1Space,
        acquisition: Acquisition,
        *,
        surrogate: Surrogate | None = None,
        n_trials: int = 25,
        n_init: int | None = None,
        n_candidates: int = 128,
        time_budget_s: float | None = None,
        seed: int | None = 0,
    ) -> None:
        self.space = space
        self.acquisition = acquisition
        self.surrogate = surrogate
        self.n_trials = int(n_trials)
        if n_init is None:
            n_init = min(5, self.n_trials) if surrogate is not None else 1
        self.n_init = max(1, min(int(n_init), self.n_trials))
        self.n_candidates = int(n_candidates)
        self.time_budget_s = time_budget_s
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self._X: list[NDArray[np.float64]] = []
        self._y: list[float] = []
        self._best_y: float = float("-inf")
        self._best_phi: Phi = space.base
        self._best_trace: list[float] = []
        self._history: list[tuple[Phi, float]] = []

    # ── factory helpers ────────────────────────────────────────────────────

    @classmethod
    def bo(
        cls,
        space: Phase1Space,
        *,
        n_trials: int = 25,
        n_init: int | None = None,
        n_candidates: int = 128,
        time_budget_s: float | None = None,
        seed: int | None = 0,
        lengthscale: float = 0.5,
        signal_var: float = 1.0,
        noise_var: float = 1e-4,
    ) -> Phase1Controller:
        """Construct the GP + EI Bayesian-optimization baseline."""
        gp = GaussianProcess(
            lengthscale=lengthscale, signal_var=signal_var, noise_var=noise_var
        )
        return cls(
            space,
            acquisition=EI(),
            surrogate=gp,
            n_trials=n_trials,
            n_init=n_init,
            n_candidates=n_candidates,
            time_budget_s=time_budget_s,
            seed=seed,
        )

    @classmethod
    def random(
        cls,
        space: Phase1Space,
        *,
        n_trials: int = 25,
        n_candidates: int = 128,
        time_budget_s: float | None = None,
        seed: int | None = 0,
    ) -> Phase1Controller:
        """Construct the uniform RandomSearch baseline (no surrogate)."""
        return cls(
            space,
            acquisition=RandomSearch(seed=seed),
            surrogate=None,
            n_trials=n_trials,
            n_init=1,
            n_candidates=n_candidates,
            time_budget_s=time_budget_s,
            seed=seed,
        )

    @classmethod
    def ladder(
        cls,
        space: Phase1Space,
        *,
        budget: int = 25,
        noise: bool | float = False,
        n_trials: int | None = None,
        n_init: int | None = None,
        n_candidates: int = 128,
        time_budget_s: float | None = None,
        seed: int | None = 0,
    ) -> Phase1Controller:
        """Construct the rung chosen by the Wave-4 router (:func:`select_surrogate`).

        Falls back to the self-contained GP + generic-EI baseline when the chosen
        rung needs an optional package that is not installed (e.g. BoTorch).
        """
        from .ladder import LadderEI, select_surrogate

        trials = int(budget) if n_trials is None else int(n_trials)
        choice = select_surrogate(space, budget=budget, noise=noise)
        try:
            surrogate, acquisition = choice.build()
        except ImportError:
            surrogate = GaussianProcess(
                lengthscale=0.5, signal_var=1.0, noise_var=1e-4
            )
            acquisition = LadderEI()
        return cls(
            space,
            acquisition=cast("Acquisition", acquisition),
            surrogate=cast("Surrogate", surrogate),
            n_trials=trials,
            n_init=n_init,
            n_candidates=n_candidates,
            time_budget_s=time_budget_s,
            seed=seed,
        )

    # ── ask / tell ──────────────────────────────────────────────────────────

    @property
    def n_observed(self) -> int:
        return len(self._y)

    @property
    def best_reward(self) -> float:
        return self._best_y

    def ask(
        self, candidates: NDArray[np.float64] | None = None
    ) -> Phi:
        """Propose the next configuration to evaluate.

        The first ``n_init`` proposals form a random initial design; afterwards
        the acquisition policy selects from a finite candidate pool (the
        supplied ``candidates``, else a freshly sampled pool of size
        ``n_candidates``).
        """
        use_acquisition = self.n_observed >= self.n_init and (
            self.surrogate is None or self.surrogate.is_fitted()
        )

        if not use_acquisition:
            if candidates is not None:
                pool = np.asarray(candidates, dtype=np.float64)
                idx = int(self._rng.integers(len(pool)))
                x = pool[idx].copy()
            else:
                x = self.space.sample_vector(self._rng)
            return self.space.decode(x)

        pool = (
            np.asarray(candidates, dtype=np.float64)
            if candidates is not None
            else self.space.candidate_pool(self.n_candidates, self._rng)
        )
        y_best = self._best_y if np.isfinite(self._best_y) else None
        acq_seed = int(self._rng.integers(np.iinfo(np.int32).max))
        # Baselines (RandomSearch) ignore the surrogate; BO passes a fitted GP.
        surrogate = cast("Surrogate", self.surrogate)
        x, _ = self.acquisition(surrogate, pool, y_best=y_best, seed=acq_seed)
        return self.space.decode(np.asarray(x, dtype=np.float64).ravel())

    def tell(self, phi: Phi, reward: float) -> None:
        """Record an observation and refit the surrogate posterior."""
        x = self.space.encode(phi)
        self._X.append(np.asarray(x, dtype=np.float64))
        r = float(reward)
        self._y.append(r)

        if r > self._best_y:
            self._best_y = r
            self._best_phi = phi
        self._best_trace.append(self._best_y)
        self._history.append((phi, r))

        if self.surrogate is not None:
            self.surrogate.fit(np.vstack(self._X), np.asarray(self._y, dtype=np.float64))

    def seed_observations(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | list[float],
    ) -> int:
        """Seed prior encoded ``(X, y)`` into the buffer and refit (warm-start hook).

        Used by :mod:`opop.controller.transfer`. The priors enter the same
        ``_X``/``_y`` buffer the surrogate is refit on, so they survive later
        :meth:`tell` calls (fitting ``self.surrogate`` directly would be wiped by
        the next ``tell``) and let acquisition start on the first :meth:`ask`.
        The trial ``history``/``best_*`` in :meth:`result` are left untouched, so
        they still reflect only trials run on the current task. Returns the count
        seeded; raises :class:`ValueError` on a length or dim mismatch.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if x_arr.size == 0:
            return 0
        x_arr = x_arr.reshape(-1, self.space.dim)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"seed_observations: X has {x_arr.shape[0]} rows but y has "
                + f"{y_arr.shape[0]} entries"
            )
        for row, val in zip(x_arr, y_arr):
            self._X.append(np.asarray(row, dtype=np.float64))
            self._y.append(float(val))
        if self.surrogate is not None:
            self.surrogate.fit(
                np.vstack(self._X), np.asarray(self._y, dtype=np.float64)
            )
        return int(x_arr.shape[0])

    # ── driver ───────────────────────────────────────────────────────────────

    def run(
        self,
        evaluator: Callable[[Phi], float],
        candidates: NDArray[np.float64] | None = None,
    ) -> Phase1Result:
        """Run the ask-tell loop to budget and return the result.

        Args:
            evaluator: Maps a proposed :class:`Phi` to a scalar reward
                (higher is better; e.g. via :func:`coip_reward`).
            candidates: Optional fixed candidate pool shared across iterations.

        Returns:
            A :class:`Phase1Result` summarizing the run.
        """
        start = time.monotonic()
        for _ in range(self.n_trials):
            if (
                self.time_budget_s is not None
                and time.monotonic() - start >= self.time_budget_s
            ):
                break
            phi = self.ask(candidates=candidates)
            reward = float(evaluator(phi))
            self.tell(phi, reward)
        return self.result()

    def result(self) -> Phase1Result:
        """Snapshot the current best/history into a :class:`Phase1Result`."""
        X = (
            np.vstack(self._X)
            if self._X
            else np.empty((0, self.space.dim), dtype=np.float64)
        )
        return Phase1Result(
            best_phi=self._best_phi,
            best_reward=self._best_y,
            best_trace=list(self._best_trace),
            history=list(self._history),
            X=X,
            y=np.asarray(self._y, dtype=np.float64),
        )
