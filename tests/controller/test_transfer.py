"""Tests for historical-transfer priors / cross-distribution warm start (task 32)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from opop.controller.encoder import Phase1Space, default_phase1_space
from opop.controller.gp import GaussianProcess
from opop.controller.meta import MetaTuner
from opop.controller.phase1 import Phase1Controller
from opop.controller.transfer import (
    InstanceDescriptor,
    LeakageError,
    PosteriorSnapshot,
    PosteriorStore,
    extract_descriptor,
    select_sources,
    warm_start_controller,
    warm_start_from_store,
)
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)
from opop.model.state import Phi


# ── fixtures / builders ─────────────────────────────────────────────────────


def _two_block_milp() -> MILP:
    """4 binaries in two independent blocks (a+b<=1, c+d<=1); max a+b+c+d."""
    return MILP(
        name="two_block",
        variables=tuple(
            Variable(name=n, vtype=VarType.BINARY, lower=0.0, upper=1.0)
            for n in ("a", "b", "c", "d")
        ),
        constraints=(
            LinearConstraint("blkA", {"a": 1.0, "b": 1.0}, ConstraintSense.LE, 1.0),
            LinearConstraint("blkB", {"c": 1.0, "d": 1.0}, ConstraintSense.LE, 1.0),
        ),
        objective=Objective(
            {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, ObjSense.MAXIMIZE, 0.0
        ),
    )


def _dense_milp() -> MILP:
    """2 binaries + 1 continuous in one monolithic row (a+b+c<=2); max a+b+c."""
    return MILP(
        name="dense",
        variables=(
            Variable("a", VarType.BINARY, 0.0, 1.0),
            Variable("b", VarType.BINARY, 0.0, 1.0),
            Variable("c", VarType.CONTINUOUS, 0.0, 1.0),
        ),
        constraints=(
            LinearConstraint("row", {"a": 1.0, "b": 1.0, "c": 1.0}, ConstraintSense.LE, 2.0),
        ),
        objective=Objective({"a": 1.0, "b": 1.0, "c": 1.0}, ObjSense.MAXIMIZE, 0.0),
    )


def _fake_snapshot(
    space: Phase1Space,
    *,
    task_id: str,
    split: str,
    n: int = 6,
    seed: int = 0,
    descriptor: InstanceDescriptor | None = None,
) -> PosteriorSnapshot:
    """A snapshot with fabricated in-space encoded X and random rewards y."""
    rng = np.random.default_rng(seed)
    X = space.candidate_pool(n, rng)
    y = rng.standard_normal(n)
    desc = descriptor or InstanceDescriptor(
        n_vars=10, n_constraints=5, integer_density=0.5, block_structure=1, avg_degree=2.0
    )
    return PosteriorSnapshot(task_id=task_id, split=split, descriptor=desc, X=X, y=y)


def _bowl_objective(space: Phase1Space) -> Callable[[Phi], float]:
    """Smooth reward peaked at a fixed reachable encoded target (max = 0.0)."""
    target = space.encode(space.random_phi(np.random.default_rng(99)))

    def objective(phi: Phi) -> float:
        x = space.encode(phi)
        return -float(np.sum((x - target) ** 2))

    return objective


# ── descriptor ──────────────────────────────────────────────────────────────


def test_extract_descriptor_two_block() -> None:
    desc = extract_descriptor(_two_block_milp())
    assert desc.n_vars == 4
    assert desc.n_constraints == 2
    assert desc.integer_density == pytest.approx(1.0)
    assert desc.avg_degree == pytest.approx(2.0)
    assert desc.block_structure == 2  # two independent blocks


def test_extract_descriptor_dense() -> None:
    desc = extract_descriptor(_dense_milp())
    assert desc.n_vars == 3
    assert desc.n_constraints == 1
    assert desc.integer_density == pytest.approx(2.0 / 3.0)
    assert desc.avg_degree == pytest.approx(3.0)
    assert desc.block_structure == 1  # monolithic -> one block


def test_descriptor_hash_stable_and_distinct() -> None:
    a = extract_descriptor(_two_block_milp())
    b = extract_descriptor(_two_block_milp())
    c = extract_descriptor(_dense_milp())
    assert a.descriptor_hash == b.descriptor_hash
    assert a.descriptor_hash != c.descriptor_hash


def test_descriptor_distance_orders_by_similarity() -> None:
    two_block = extract_descriptor(_two_block_milp())
    dense = extract_descriptor(_dense_milp())
    assert two_block.distance(two_block) == pytest.approx(0.0)
    assert two_block.distance(dense) > 0.0
    # symmetry
    assert two_block.distance(dense) == pytest.approx(dense.distance(two_block))


# ── snapshot serialization ───────────────────────────────────────────────────


def test_snapshot_roundtrip_via_dict() -> None:
    space = default_phase1_space()
    snap = _fake_snapshot(space, task_id="src", split="dev", n=4, seed=1)
    restored = PosteriorSnapshot.from_dict(snap.to_dict())
    assert restored.task_id == snap.task_id
    assert restored.split == snap.split
    assert restored.descriptor == snap.descriptor
    assert restored.dim == space.dim
    assert np.allclose(restored.X, snap.X)
    assert np.allclose(restored.y, snap.y)


def test_snapshot_from_controller_captures_posterior() -> None:
    space = default_phase1_space()
    controller = Phase1Controller.bo(space, n_trials=4, n_init=2, seed=0)
    controller.run(_bowl_objective(space))
    desc = InstanceDescriptor(8, 4, 0.5, 1, 2.0)

    snap = PosteriorSnapshot.from_controller(
        controller, task_id="run1", split="dev", descriptor=desc
    )
    res = controller.result()
    assert snap.n_obs == res.X.shape[0] == 4
    assert np.allclose(snap.X, res.X)
    assert np.allclose(snap.y, res.y)
    assert snap.gp_hyperparams is not None
    assert snap.gp_hyperparams["lengthscale"] > 0.0


# ── store: save/load all splits + leakage guard ──────────────────────────────


def test_store_save_load_roundtrip_free_split(tmp_path: Path) -> None:
    space = default_phase1_space()
    store = PosteriorStore(tmp_path)
    snap = _fake_snapshot(space, task_id="dev_task", split="dev", n=5, seed=2)

    path = store.save(snap)
    assert path.exists()
    loaded = store.load(path)
    assert loaded.task_id == "dev_task"
    assert loaded.split == "dev"
    assert np.allclose(loaded.X, snap.X)


def test_store_saves_every_split_tag(tmp_path: Path) -> None:
    space = default_phase1_space()
    store = PosteriorStore(tmp_path)
    for split in ("dev", "validation", "test", "ood_test"):
        snap = _fake_snapshot(space, task_id=f"t_{split}", split=split, n=3)
        path = store.save(snap)
        assert path.exists()
        # held-out snapshots persist and are readable only with the explicit flag
        loaded = store.load(path, allow_held_out=True)
        assert loaded.split == split


@pytest.mark.parametrize("split", ["test", "ood_test"])
def test_store_load_held_out_raises_leakage(tmp_path: Path, split: str) -> None:
    space = default_phase1_space()
    store = PosteriorStore(tmp_path)
    snap = _fake_snapshot(space, task_id="leak", split=split, n=3)
    path = store.save(snap)

    with pytest.raises(LeakageError):
        store.load(path)


def test_store_warmstart_candidates_raises_on_held_out_file(tmp_path: Path) -> None:
    space = default_phase1_space()
    store = PosteriorStore(tmp_path)
    store.save(_fake_snapshot(space, task_id="ok", split="dev", n=3))
    store.save(_fake_snapshot(space, task_id="bad", split="test", n=3))

    with pytest.raises(LeakageError):
        store.warmstart_candidates(InstanceDescriptor(10, 5, 0.5, 1, 2.0))


# ── warm start ────────────────────────────────────────────────────────────────


def test_warm_start_seeds_gp_posterior() -> None:
    space = default_phase1_space()
    snap = _fake_snapshot(space, task_id="src", split="dev", n=6, seed=3)
    controller = Phase1Controller.bo(space, n_trials=10, n_init=5, seed=0)

    seeded = warm_start_controller(controller, [snap])

    assert seeded == 6
    assert controller.n_observed == 6
    assert isinstance(controller.surrogate, GaussianProcess)
    assert controller.surrogate.is_fitted()
    assert controller.surrogate.n_train == 6
    # trial log stays clean: no real tells yet
    assert controller.result().history == []
    # acquisition is usable immediately (n_observed >= n_init, GP fitted)
    assert isinstance(controller.ask(), Phi)


def test_warm_start_combines_multiple_sources() -> None:
    space = default_phase1_space()
    snap1 = _fake_snapshot(space, task_id="s1", split="dev", n=4, seed=10)
    snap2 = _fake_snapshot(space, task_id="s2", split="validation", n=7, seed=11)
    controller = Phase1Controller.bo(space, n_trials=5, n_init=3, seed=0)

    seeded = warm_start_controller(controller, [snap1, snap2])

    assert seeded == 11  # not overfit to a single source
    assert controller.n_observed == 11
    assert isinstance(controller.surrogate, GaussianProcess)
    assert controller.surrogate.n_train == 11


def test_warm_start_refuses_held_out_source() -> None:
    space = default_phase1_space()
    snap = _fake_snapshot(space, task_id="leak", split="test", n=4)
    controller = Phase1Controller.bo(space, n_trials=5, seed=0)

    with pytest.raises(LeakageError):
        warm_start_controller(controller, [snap])
    # controller untouched by the refusal
    assert controller.n_observed == 0


def test_warm_start_dim_mismatch_raises() -> None:
    space = default_phase1_space()
    bad = PosteriorSnapshot(
        task_id="wrongdim",
        split="dev",
        descriptor=InstanceDescriptor(1, 1, 0.0, 1, 1.0),
        X=np.zeros((3, space.dim + 2), dtype=np.float64),
        y=np.zeros(3, dtype=np.float64),
    )
    controller = Phase1Controller.bo(space, n_trials=5, seed=0)

    with pytest.raises(ValueError, match="dim"):
        warm_start_controller(controller, [bad])


def test_warm_start_empty_sources_is_noop() -> None:
    space = default_phase1_space()
    controller = Phase1Controller.bo(space, n_trials=5, seed=0)
    assert warm_start_controller(controller, []) == 0
    assert controller.n_observed == 0


# ── transfer_off == cold start (deterministic) ────────────────────────────────


def test_transfer_off_reproduces_cold_start() -> None:
    space = default_phase1_space()
    objective = _bowl_objective(space)
    pool = space.candidate_pool(60, np.random.default_rng(123))
    snap = _fake_snapshot(space, task_id="src", split="dev", n=8, seed=5)

    cold = Phase1Controller.bo(space, n_trials=12, n_init=3, seed=0)
    off = Phase1Controller.bo(space, n_trials=12, n_init=3, seed=0)
    seeded = warm_start_controller(off, [snap], transfer_off=True)

    cold_res = cold.run(objective, candidates=pool)
    off_res = off.run(objective, candidates=pool)

    assert seeded == 0
    assert off.n_observed == cold.n_observed == 12
    assert off_res.best_reward == cold_res.best_reward
    assert off_res.best_trace == cold_res.best_trace
    assert np.array_equal(off_res.X, cold_res.X)
    assert np.array_equal(off_res.y, cold_res.y)


def test_transfer_on_changes_gp_training_set() -> None:
    space = default_phase1_space()
    objective = _bowl_objective(space)
    pool = space.candidate_pool(60, np.random.default_rng(123))
    snap = _fake_snapshot(space, task_id="src", split="dev", n=8, seed=5)

    cold = Phase1Controller.bo(space, n_trials=12, n_init=3, seed=0)
    warm = Phase1Controller.bo(space, n_trials=12, n_init=3, seed=0)
    warm_start_controller(warm, [snap], transfer_off=False)

    cold.run(objective, candidates=pool)
    warm.run(objective, candidates=pool)

    assert isinstance(cold.surrogate, GaussianProcess)
    assert isinstance(warm.surrogate, GaussianProcess)
    # warm-start genuinely seeded the posterior: 8 priors + 12 trials.
    assert cold.surrogate.n_train == 12
    assert warm.surrogate.n_train == 20


# ── store-driven warm start + descriptor selection ────────────────────────────


def test_select_sources_orders_and_limits() -> None:
    space = default_phase1_space()
    target = InstanceDescriptor(100, 50, 0.5, 4, 3.0)
    near = _fake_snapshot(
        space, task_id="near", split="dev", descriptor=InstanceDescriptor(98, 49, 0.5, 4, 3.0)
    )
    mid = _fake_snapshot(
        space, task_id="mid", split="dev", descriptor=InstanceDescriptor(50, 25, 0.4, 2, 2.0)
    )
    far = _fake_snapshot(
        space, task_id="far", split="dev", descriptor=InstanceDescriptor(2, 1, 0.0, 1, 1.0)
    )

    ranked = select_sources(target, [far, near, mid])
    assert [s.task_id for s in ranked] == ["near", "mid", "far"]
    assert [s.task_id for s in select_sources(target, [far, near, mid], k=2)] == ["near", "mid"]


def test_warm_start_from_store_selects_similar(tmp_path: Path) -> None:
    space = default_phase1_space()
    store = PosteriorStore(tmp_path)
    target = InstanceDescriptor(100, 50, 0.5, 4, 3.0)
    store.save(
        _fake_snapshot(
            space, task_id="near", split="dev", n=5,
            descriptor=InstanceDescriptor(98, 49, 0.5, 4, 3.0),
        )
    )
    store.save(
        _fake_snapshot(
            space, task_id="far", split="validation", n=6,
            descriptor=InstanceDescriptor(2, 1, 0.0, 1, 1.0),
        )
    )

    controller = Phase1Controller.bo(space, n_trials=8, n_init=3, seed=0)
    seeded = warm_start_from_store(controller, store, target, k=1)

    assert seeded == 5  # only the nearest (dev) source
    assert controller.n_observed == 5


def test_warm_start_from_store_transfer_off_skips_disk(tmp_path: Path) -> None:
    space = default_phase1_space()
    # a held-out file in the store would normally raise on scan; transfer_off
    # must short-circuit before any disk read.
    store = PosteriorStore(tmp_path)
    store.save(_fake_snapshot(space, task_id="leak", split="test", n=3))

    controller = Phase1Controller.bo(space, n_trials=5, seed=0)
    seeded = warm_start_from_store(
        controller, store, InstanceDescriptor(10, 5, 0.5, 1, 2.0), transfer_off=True
    )
    assert seeded == 0
    assert controller.n_observed == 0


# ── meta-tuner ────────────────────────────────────────────────────────────────


def _meta_designs(
    seed: int = 0,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    rng = np.random.default_rng(seed)
    designs: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for _ in range(3):
        X = rng.random((6, 3))
        y = np.sin(X.sum(axis=1))
        designs.append((X, y))
    return designs


def test_meta_tuner_reptile_trains_and_builds_gp() -> None:
    tuner = MetaTuner(lengthscale=1.0, signal_var=1.0, noise_var=1e-3)
    losses = tuner.meta_train(_meta_designs(), mode="reptile", n_inner_steps=3)

    assert losses and all(np.isfinite(loss) for loss in losses)
    hp = tuner.get_hyperparams()
    assert hp["lengthscale"] > 0.0
    assert hp["signal_var"] > 0.0
    assert hp["noise_var"] > 0.0

    gp = tuner.build_gp()
    assert isinstance(gp, GaussianProcess)
    X = np.random.default_rng(1).random((5, 3))
    gp.fit(X, np.sin(X.sum(axis=1)))
    mean, std = gp.predict(X)
    assert bool(np.isfinite(mean.numpy()).all())
    assert bool((std.numpy() >= 0).all())


def test_meta_tuner_maml_runs() -> None:
    tuner = MetaTuner()
    losses = tuner.meta_train(_meta_designs(7), mode="maml", n_inner_steps=2)
    assert losses and all(np.isfinite(loss) for loss in losses)
    hp = tuner.get_hyperparams()
    assert all(v > 0.0 for v in hp.values())


def test_meta_tuner_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        MetaTuner().meta_train(_meta_designs(), mode="bogus")
