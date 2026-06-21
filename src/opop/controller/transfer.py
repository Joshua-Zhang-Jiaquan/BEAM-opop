"""Historical-transfer priors / cross-distribution warm start (task 32).

Persists per-distribution GP posteriors (the controller's accumulated encoded
``(X, y)`` observations) tagged by an instance *descriptor* and dataset *split*,
and warm-starts a fresh :class:`~opop.controller.phase1.Phase1Controller` from
related historical tasks selected by descriptor similarity (sparsity via average
degree, block structure, integer density, problem size).

Scientific-integrity guard (Metis leakage policy, mirroring
:mod:`opop.bench.audit`): a posterior tagged ``test`` / ``ood_test`` may NEVER
seed a warm-start. Loading such a snapshot for warm-start raises
:class:`LeakageError` (fail-loud), and :func:`warm_start_controller`
re-validates every source as defence in depth.

``transfer_off=True`` makes every warm-start a deterministic no-op, so a
warm-started controller is byte-identical to a cold-start one (the priors are
never read, the RNG is never advanced).

The typed :class:`LeakageError` is defined locally (rather than importing
:class:`opop.bench.registry.LeakageError`, which is a ``RegistryError`` subclass
tied to split-manifest semantics) so the controller layer stays self-contained
and does not depend on the bench layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from opop.model.ir import MILP, VarType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .phase1 import Phase1Controller

__all__ = [
    "DEFAULT_TRANSFER_DIR",
    "FREE_SPLITS",
    "HELD_OUT_SPLITS",
    "InstanceDescriptor",
    "LeakageError",
    "PosteriorSnapshot",
    "PosteriorStore",
    "extract_descriptor",
    "select_sources",
    "warm_start_controller",
    "warm_start_from_store",
]

#: Default on-disk root for persisted posterior snapshots.
DEFAULT_TRANSFER_DIR = ".opop_transfer"

#: Splits that may legitimately seed a warm-start.
FREE_SPLITS = frozenset({"dev", "validation"})
#: Held-out splits that must NEVER seed a warm-start (leakage).
HELD_OUT_SPLITS = frozenset({"test", "ood_test"})


class LeakageError(Exception):
    """Raised when a held-out (``test`` / ``ood_test``) posterior is read for warm-start."""


# ── Instance descriptor ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InstanceDescriptor:
    """Cheap IR-statistics fingerprint keying historical-transfer lookup.

    Attributes:
        n_vars: Number of decision variables.
        n_constraints: Number of linear constraints.
        integer_density: Fraction of variables that are BINARY/INTEGER (``[0, 1]``).
        block_structure: Independent-block signal (``detect_decomposition`` block
            count; ``1`` for a monolithic / dense instance).
        avg_degree: Average constraint degree (nnz per constraint) — the sparsity
            signal (low = sparse).
    """

    n_vars: int = 0
    n_constraints: int = 0
    integer_density: float = 0.0
    block_structure: int = 1
    avg_degree: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping of the descriptor."""
        return {
            "n_vars": int(self.n_vars),
            "n_constraints": int(self.n_constraints),
            "integer_density": float(self.integer_density),
            "block_structure": int(self.block_structure),
            "avg_degree": float(self.avg_degree),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstanceDescriptor:
        """Rebuild a descriptor from its :meth:`to_dict` mapping."""
        return cls(
            n_vars=int(data["n_vars"]),
            n_constraints=int(data["n_constraints"]),
            integer_density=float(data["integer_density"]),
            block_structure=int(data["block_structure"]),
            avg_degree=float(data["avg_degree"]),
        )

    @property
    def descriptor_hash(self) -> str:
        """Stable 16-hex-char digest over the rounded descriptor (filename key)."""
        payload = {
            "n_vars": int(self.n_vars),
            "n_constraints": int(self.n_constraints),
            "integer_density": round(float(self.integer_density), 6),
            "block_structure": int(self.block_structure),
            "avg_degree": round(float(self.avg_degree), 6),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def feature_vector(self) -> NDArray[np.float64]:
        """Log-scaled feature vector for similarity (counts logged, densities raw)."""
        return np.asarray(
            [
                math.log1p(max(self.n_vars, 0)),
                math.log1p(max(self.n_constraints, 0)),
                float(self.integer_density),
                math.log1p(max(self.block_structure, 0)),
                math.log1p(max(self.avg_degree, 0.0)),
            ],
            dtype=np.float64,
        )

    def distance(self, other: InstanceDescriptor) -> float:
        """Euclidean distance between feature vectors (``0`` == identical shape)."""
        return float(np.linalg.norm(self.feature_vector() - other.feature_vector()))


def _block_signal(ir: MILP) -> int:
    """Independent-block count via task-24 ``detect_decomposition`` (lazy import).

    ``NONE`` (``n_blocks == 0``) maps to ``1`` (one monolithic block) so the
    descriptor signal is always ``>= 1``.
    """
    from opop.analyzer.decompose import detect_decomposition

    return max(1, detect_decomposition(ir).n_blocks)


def extract_descriptor(ir: MILP) -> InstanceDescriptor:
    """Extract the :class:`InstanceDescriptor` of a MILP from cheap IR statistics."""
    n_vars = ir.n_vars
    n_cons = ir.n_constraints
    n_int = sum(
        1 for v in ir.variables if v.vtype in (VarType.BINARY, VarType.INTEGER)
    )
    integer_density = (n_int / n_vars) if n_vars else 0.0
    avg_degree = (ir.nnz / n_cons) if n_cons else 0.0
    return InstanceDescriptor(
        n_vars=n_vars,
        n_constraints=n_cons,
        integer_density=integer_density,
        block_structure=_block_signal(ir),
        avg_degree=avg_degree,
    )


def _coerce_hyper(raw: Any) -> dict[str, float] | None:
    """Coerce a deserialised ``gp_hyperparams`` value to ``dict[str, float] | None``."""
    if not isinstance(raw, dict):
        return None
    mapping: dict[str, Any] = raw
    return {str(key): float(value) for key, value in mapping.items()}


# ── Posterior snapshot ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PosteriorSnapshot:
    """A persisted controller posterior: encoded observations + descriptor + split.

    Attributes:
        task_id: Identifier of the historical task this posterior came from.
        split: Dataset split tag (``dev`` / ``validation`` / ``test`` / ``ood_test``).
        descriptor: The source instance's :class:`InstanceDescriptor`.
        X: ``[n, dim]`` encoded observation inputs.
        y: ``[n]`` observed rewards (higher is better).
        gp_hyperparams: Optional GP hyperparameters (``lengthscale`` /
            ``signal_var`` / ``noise_var``) captured at snapshot time.
    """

    task_id: str
    split: str
    descriptor: InstanceDescriptor
    X: NDArray[np.float64]
    y: NDArray[np.float64]
    gp_hyperparams: dict[str, float] | None = None

    @property
    def n_obs(self) -> int:
        """Number of observations stored."""
        return int(self.X.shape[0]) if self.X.ndim == 2 else 0

    @property
    def dim(self) -> int:
        """Encoded input dimensionality."""
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    @property
    def is_held_out(self) -> bool:
        """Whether this snapshot belongs to a held-out (test/ood_test) split."""
        return self.split in HELD_OUT_SPLITS

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping (arrays as nested lists)."""
        return {
            "task_id": self.task_id,
            "split": self.split,
            "descriptor": self.descriptor.to_dict(),
            "dim": self.dim,
            "X": self.X.tolist(),
            "y": self.y.tolist(),
            "gp_hyperparams": self.gp_hyperparams,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PosteriorSnapshot:
        """Rebuild a snapshot from its :meth:`to_dict` mapping."""
        dim = int(data.get("dim", 0))
        x_raw = np.asarray(data.get("X", []), dtype=np.float64)
        x_arr = x_raw.reshape(-1, dim) if dim > 0 else x_raw.reshape(0, 0)
        y_arr = np.asarray(data.get("y", []), dtype=np.float64).reshape(-1)
        return cls(
            task_id=str(data["task_id"]),
            split=str(data["split"]),
            descriptor=InstanceDescriptor.from_dict(data["descriptor"]),
            X=x_arr,
            y=y_arr,
            gp_hyperparams=_coerce_hyper(data.get("gp_hyperparams")),
        )

    @classmethod
    def from_controller(
        cls,
        controller: Phase1Controller,
        *,
        task_id: str,
        split: str,
        descriptor: InstanceDescriptor,
    ) -> PosteriorSnapshot:
        """Capture a controller's accumulated ``(X, y)`` posterior as a snapshot."""
        result = controller.result()
        surrogate = controller.surrogate
        hyper: dict[str, float] | None = None
        if surrogate is not None and all(
            hasattr(surrogate, attr)
            for attr in ("lengthscale", "signal_var", "noise_var")
        ):
            hyper = {
                "lengthscale": float(getattr(surrogate, "lengthscale")),
                "signal_var": float(getattr(surrogate, "signal_var")),
                "noise_var": float(getattr(surrogate, "noise_var")),
            }
        return cls(
            task_id=task_id,
            split=split,
            descriptor=descriptor,
            X=np.asarray(result.X, dtype=np.float64),
            y=np.asarray(result.y, dtype=np.float64),
            gp_hyperparams=hyper,
        )


def _safe_task_id(task_id: str) -> str:
    """Filename-safe rendering of an arbitrary task id."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", task_id).strip("_")
    return cleaned or "task"


# ── Posterior store ─────────────────────────────────────────────────────────


class PosteriorStore:
    """On-disk store of :class:`PosteriorSnapshot` JSON files under a root dir.

    Files are keyed ``{safe_task_id}__{descriptor_hash}.json``. The default
    :meth:`load` enforces the leakage guard (held-out snapshots raise); pass
    ``allow_held_out=True`` only for legitimate inspection (never warm-start).
    """

    root: Path

    def __init__(self, root: str | Path = DEFAULT_TRANSFER_DIR) -> None:
        self.root = Path(root)

    def path_for(self, snapshot: PosteriorSnapshot) -> Path:
        """Return the on-disk path a snapshot saves to (without writing it)."""
        name = f"{_safe_task_id(snapshot.task_id)}__{snapshot.descriptor.descriptor_hash}.json"
        return self.root / name

    def save(self, snapshot: PosteriorSnapshot) -> Path:
        """Persist a snapshot (any split) as deterministic JSON; return its path."""
        path = self.path_for(snapshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot.to_dict(), allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return path

    def load(
        self, path: str | Path, *, allow_held_out: bool = False
    ) -> PosteriorSnapshot:
        """Load one snapshot; raise :class:`LeakageError` on held-out unless allowed."""
        path = Path(path)
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"snapshot {path} is not a JSON object")
        record: dict[str, Any] = data
        split = str(record.get("split", ""))
        if not allow_held_out and split in HELD_OUT_SPLITS:
            raise LeakageError(
                "refusing to load held-out posterior for warm-start: "
                + f"{path} (split={split!r})"
            )
        return PosteriorSnapshot.from_dict(record)

    def iter_snapshots(
        self, *, allow_held_out: bool = False
    ) -> Iterator[PosteriorSnapshot]:
        """Yield every snapshot under :attr:`root` (sorted; honours the leakage guard)."""
        for path in sorted(self.root.glob("*.json")):
            yield self.load(path, allow_held_out=allow_held_out)

    def warmstart_candidates(
        self,
        descriptor: InstanceDescriptor,
        *,
        k: int | None = None,
        max_distance: float | None = None,
    ) -> list[PosteriorSnapshot]:
        """Free-split snapshots nearest ``descriptor`` (fail-loud on any held-out file)."""
        snapshots = list(self.iter_snapshots(allow_held_out=False))
        return select_sources(
            descriptor, snapshots, k=k, max_distance=max_distance
        )


# ── Source selection + warm start ───────────────────────────────────────────


def select_sources(
    descriptor: InstanceDescriptor,
    snapshots: Sequence[PosteriorSnapshot],
    *,
    k: int | None = None,
    max_distance: float | None = None,
) -> list[PosteriorSnapshot]:
    """Rank ``snapshots`` by descriptor distance (nearest first; deterministic).

    Optionally filters to those within ``max_distance`` and/or the ``k`` nearest.
    Ties break on ``task_id`` for reproducibility.
    """
    ranked = sorted(
        snapshots,
        key=lambda s: (s.descriptor.distance(descriptor), s.task_id),
    )
    if max_distance is not None:
        ranked = [
            s for s in ranked if s.descriptor.distance(descriptor) <= max_distance
        ]
    if k is not None:
        ranked = ranked[: max(0, k)]
    return ranked


def warm_start_controller(
    controller: Phase1Controller,
    sources: Sequence[PosteriorSnapshot],
    *,
    transfer_off: bool = False,
) -> int:
    """Seed ``controller`` with historical observations from ``sources``.

    Concatenates every source's ``(X, y)`` and seeds them into the controller's
    surrogate via :meth:`Phase1Controller.seed_observations`. Multiple sources
    are combined (so a warm-start is never overfit to a single source).

    ``transfer_off=True`` is a deterministic no-op: the controller is left
    byte-identical to a cold-start one (returns ``0``, reads nothing).

    Raises:
        LeakageError: if any source is from a held-out split (defence in depth).
        ValueError: if a source's encoded dimensionality differs from the
            controller's :class:`~opop.controller.encoder.Phase1Space`.
    """
    if transfer_off:
        return 0

    dim = controller.space.dim
    blocks_x: list[NDArray[np.float64]] = []
    blocks_y: list[NDArray[np.float64]] = []
    for snap in sources:
        if snap.is_held_out:
            raise LeakageError(
                "refusing to warm-start from held-out posterior "
                + f"{snap.task_id!r} (split={snap.split!r})"
            )
        if snap.n_obs == 0:
            continue
        if snap.dim != dim:
            raise ValueError(
                f"source {snap.task_id!r} encoded dim {snap.dim} != controller "
                + f"space dim {dim}"
            )
        blocks_x.append(np.asarray(snap.X, dtype=np.float64).reshape(-1, dim))
        blocks_y.append(np.asarray(snap.y, dtype=np.float64).reshape(-1))

    if not blocks_x:
        return 0

    x_all = np.vstack(blocks_x)
    y_all = np.concatenate(blocks_y)
    return controller.seed_observations(x_all, y_all)


def warm_start_from_store(
    controller: Phase1Controller,
    store: PosteriorStore,
    descriptor: InstanceDescriptor,
    *,
    k: int | None = None,
    max_distance: float | None = None,
    transfer_off: bool = False,
) -> int:
    """Warm-start ``controller`` from a :class:`PosteriorStore` by descriptor lookup.

    ``transfer_off=True`` short-circuits before any disk read (deterministic
    no-op). Otherwise selects the nearest free-split snapshots (raising
    :class:`LeakageError` if the store contains any held-out file) and seeds them.
    """
    if transfer_off:
        return 0
    candidates = store.warmstart_candidates(
        descriptor, k=k, max_distance=max_distance
    )
    return warm_start_controller(controller, candidates, transfer_off=False)
