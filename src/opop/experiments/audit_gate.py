"""Leakage-audit gate for the OPOP experiment matrix (plan task 39).

Before the matrix driver runs ANY cell on a given split it must pass this gate,
which enforces the two scientific-integrity invariants of the campaign:

1. **Sealed lock** — ``benchmarks/registry.yaml`` must load and its
   ``split_manifest.lock`` must match (:meth:`opop.bench.registry.BenchmarkRegistry.verify_lock`).
   An unsealed / mismatched / unreadable registry means the instance->split
   assignment is not trustworthy, so no run is allowed.
2. **Held-out protection** — the held-out splits
   (:data:`opop.experiments.fairness.HELD_OUT_SPLITS` = ``test`` / ``ood_test``)
   may only be touched in the FINAL one-shot evaluation, i.e. when the caller
   explicitly passes ``one_shot_final=True``.

Both failures raise :class:`MatrixAuditError`. This module is pure policy on top
of :mod:`opop.bench.registry` + :mod:`opop.experiments.fairness` — no solver, no
matrix expansion.
"""

from __future__ import annotations

from pathlib import Path

from opop.bench.registry import SPLITS, BenchmarkRegistry, RegistryError
from opop.bench.sources.phase1_set import REGISTRY_PATH
from opop.experiments.fairness import HELD_OUT_SPLITS

__all__ = ["MatrixAuditError", "assert_can_run_split"]


class MatrixAuditError(RuntimeError):
    """Raised when a matrix run would violate the lock or held-out-split policy."""


def assert_can_run_split(
    split: str,
    *,
    registry_path: str | Path = REGISTRY_PATH,
    one_shot_final: bool = False,
) -> None:
    """Assert the matrix may run on ``split`` (sealed lock + held-out guard).

    Args:
        split: The dataset split a sweep is about to run on.
        registry_path: Registry YAML whose lock must be sealed (defaults to the
            committed ``benchmarks/registry.yaml``; the lock path is inferred).
        one_shot_final: Set ``True`` ONLY for the final one-shot evaluation, which
            is the sole context permitted to touch a held-out split.

    Raises:
        MatrixAuditError: If ``split`` is unknown, the registry lock is
            unsealed / mismatched / unreadable, or a held-out split is requested
            without ``one_shot_final=True``.
    """
    if split not in SPLITS:
        raise MatrixAuditError(f"unknown split {split!r}; valid splits: {sorted(SPLITS)}")

    try:
        BenchmarkRegistry.from_yaml(registry_path).verify_lock()
    except RegistryError as exc:
        raise MatrixAuditError(
            f"registry lock is not sealed/valid for {registry_path}: {exc}"
        ) from exc

    if split in HELD_OUT_SPLITS and not one_shot_final:
        raise MatrixAuditError(
            f"refusing to run on held-out split {split!r} without one_shot_final=True "
            + "(test/ood_test leakage guard)"
        )
