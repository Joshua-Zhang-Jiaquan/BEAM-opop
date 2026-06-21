"""Tests for the Phase-1 ask-tell controller and the Phi <-> vector encoder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from opop.controller.encoder import BoolDim, Phase1Space, default_phase1_space
from opop.controller.gp import GaussianProcess
from opop.controller.phase1 import Phase1Controller, coip_reward
from opop.model.state import Phi


def _zero_objective(phi: Phi) -> float:
    del phi
    return 0.0


def _bowl_objective(space: Phase1Space) -> Callable[[Phi], float]:
    """Smooth reward peaked at a fixed reachable encoded target (max = 0.0)."""
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


def test_encoder_roundtrip() -> None:
    """encode -> decode -> phi' == phi for the restricted Phase-1 subspace."""
    space = default_phase1_space()

    phis = [
        space.base,
        replace(
            space.base,
            c="cuts_on",
            d="benders",
            h=1,
            p={
                "separating/gomory/freq": 2.5,
                "separating/clique/freq": 2.5,
                "separating/zerohalf/freq": 2.5,
                "branching/scorefac": 0.25,
                "presolving/maxrounds": 5.0,
                "limits/gap": 0.0001,
            },
        ),
    ]
    # Exhaustive grid over the categorical / bool / ordinal fields.
    for c in ("cuts_off", "cuts_on"):
        for d in ("none", "benders", "dw"):
            for h in (0, 1, 2):
                phis.append(
                    replace(
                        space.base,
                        c=c,
                        d=d,
                        h=h,
                        p={
                            "separating/gomory/freq": 5.0,
                            "separating/clique/freq": 0.0,
                            "separating/zerohalf/freq": 2.5,
                            "branching/scorefac": 0.5,
                            "presolving/maxrounds": 10.0,
                            "limits/gap": 0.01,
                        },
                    )
                )

    for phi in phis:
        vec = space.encode(phi)
        phi2 = space.decode(vec)
        assert phi2 == phi, f"round-trip failed: {phi} -> {phi2}"

    # Vector-level round-trip for randomly sampled valid candidates.
    rng = np.random.default_rng(0)
    pool = space.candidate_pool(25, rng)
    assert pool.shape == (25, space.dim)
    for x in pool:
        phi = space.decode(x)
        assert np.allclose(space.encode(phi), x)


def test_bool_encoding() -> None:
    """Bool fields encode to 0/1 and round-trip (string flag + plain bool)."""
    space = default_phase1_space()
    on = replace(space.base, c="cuts_on")
    off = replace(space.base, c="cuts_off")

    # The bool ("c") dim is first in the encoding order.
    assert space.encode(on)[0] == 1.0
    assert space.encode(off)[0] == 0.0
    assert space.decode(space.encode(on)).c == "cuts_on"
    assert space.decode(space.encode(off)).c == "cuts_off"

    # Plain True/False BoolDim.
    flag = BoolDim("flag", true_value=True, false_value=False)
    assert flag.encode_value(True) == [1.0]
    assert flag.encode_value(False) == [0.0]
    assert flag.decode_value(np.array([1.0])) is True
    assert flag.decode_value(np.array([0.0])) is False


def test_bo_beats_random() -> None:
    """On a synthetic surrogate objective, BO best >= random best."""
    space = default_phase1_space()
    objective = _bowl_objective(space)

    # Fixed candidate pool shared by both controllers for an apples-to-apples
    # comparison.
    pool = space.candidate_pool(60, np.random.default_rng(123))

    bo = Phase1Controller.bo(space, n_trials=25, n_init=5, seed=0)
    rand = Phase1Controller.random(space, n_trials=25, seed=0)

    bo_res = bo.run(objective, candidates=pool)
    rand_res = rand.run(objective, candidates=pool)

    assert len(bo_res.history) == 25
    assert len(rand_res.history) == 25
    assert bo_res.best_reward >= rand_res.best_reward - 1e-9, (
        f"BO {bo_res.best_reward} lost to random {rand_res.best_reward}"
    )
    # Best-so-far trace is monotonically non-decreasing.
    assert all(
        b >= a - 1e-12 for a, b in zip(bo_res.best_trace, bo_res.best_trace[1:])
    )


def test_posterior_updates() -> None:
    """The surrogate posterior refits (n_train grows) after every tell."""
    space = default_phase1_space()
    objective = _bowl_objective(space)

    bo = Phase1Controller.bo(space, n_trials=8, n_init=3, seed=0)
    assert isinstance(bo.surrogate, GaussianProcess)

    for k in range(1, 6):
        phi = bo.ask()
        bo.tell(phi, objective(phi))
        assert bo.surrogate.n_train == k
        assert bo.surrogate.is_fitted()


def test_ask_tell_budget() -> None:
    """run() executes exactly n_trials ask-tell rounds."""
    space = default_phase1_space()
    bo = Phase1Controller.bo(space, n_trials=10, seed=0)

    res = bo.run(_zero_objective)

    assert res.n_trials == 10
    assert len(res.history) == 10
    assert res.X.shape == (10, space.dim)
    assert res.y.shape == (10,)


def test_time_budget_stops_early() -> None:
    """A zero wall-clock budget stops before any evaluation."""
    space = default_phase1_space()
    bo = Phase1Controller.bo(space, n_trials=10, time_budget_s=0.0, seed=0)

    res = bo.run(_zero_objective)

    assert res.n_trials == 0
    assert res.X.shape == (0, space.dim)


def test_coip_reward_uses_coip_metrics() -> None:
    """coip_reward = -gap - 1e-3*time - primal_integral (not EDA defaults)."""
    metrics = {"gap": 0.1, "time": 50.0, "primal_integral": 2.0}
    expected = -1.0 * 0.1 - 1e-3 * 50.0 - 1.0 * 2.0
    assert coip_reward(metrics) == pytest.approx(expected, abs=1e-12)

    # Missing primal_integral defaults to 0; accepts runtime_seconds alias.
    assert coip_reward({"gap": 0.0, "time": 0.0}) == pytest.approx(0.0, abs=1e-12)
    assert coip_reward({"gap": 0.0, "runtime_seconds": 1000.0}) == pytest.approx(
        -1.0, abs=1e-12
    )
