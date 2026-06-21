"""Phi <-> numeric-vector encoder for the Phase-1 Bayesian-optimization controller.

Maps the mixed-type :class:`~opop.model.state.Phi` design vector
(categorical / ordinal / bool / continuous, per :meth:`Phi.field_types`) onto a
stable numeric vector in ``[0, 1]^d`` suitable for the Matern-5/2
:class:`~opop.controller.gp.GaussianProcess` surrogate, and back.

Encoding scheme (everything normalized to ``[0, 1]`` so a single-lengthscale
Matern kernel sees comparable scales):

- categorical -> one-hot block (``len(values)`` dims), decode = ``argmax``.
- ordinal     -> single dim ``index / (k - 1)``, decode = nearest level.
- bool        -> single ``0/1`` dim (a two-state flag, e.g. cut on/off).
- continuous  -> single dim ``(x - low) / (high - low)``.
- continuous dict (``p`` / ``rho``) -> one normalized dim per declared key.

Round-trip: ``decode(encode(phi)) == phi`` for any ``phi`` in the restricted
Phase-1 subspace described by a :class:`Phase1Space`.  With ``[0, 1]`` continuous
bounds the round-trip is bit-exact; with general bounds it is exact up to
floating-point normalization error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from opop.model.state import Phi
from opop.proposer.params import CURATED_PARAMS

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _as_float(value: object, field: str) -> float:
    """Coerce a numeric Phi field value to ``float``, failing loudly otherwise."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"field {field!r} expected a number, got {value!r}")
    return float(value)


# ── Per-field encoding dimensions ───────────────────────────────────────────


@dataclass(frozen=True)
class CategoricalDim:
    """One-hot encoded categorical field with a stable, ordered value list."""

    field: str
    values: tuple[str, ...]

    def __init__(self, field: str, values: Sequence[str]) -> None:
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "values", tuple(values))

    @property
    def width(self) -> int:
        return len(self.values)

    def encode_value(self, value: object) -> list[float]:
        vec = [0.0] * len(self.values)
        try:
            idx = self.values.index(value)  # type: ignore[arg-type]
        except ValueError as exc:
            msg = f"{value!r} is not a valid choice for {self.field!r}: {self.values}"
            raise ValueError(msg) from exc
        vec[idx] = 1.0
        return vec

    def decode_value(self, sub: NDArray[np.float64]) -> str:
        return self.values[int(np.argmax(sub))]

    def sample(self, rng: np.random.Generator) -> list[float]:
        vec = [0.0] * len(self.values)
        vec[int(rng.integers(len(self.values)))] = 1.0
        return vec


@dataclass(frozen=True)
class OrdinalDim:
    """Ordinal field encoded as a single normalized index in ``[0, 1]``."""

    field: str
    levels: tuple[int, ...]

    def __init__(self, field: str, levels: Sequence[int]) -> None:
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "levels", tuple(int(x) for x in levels))

    @property
    def width(self) -> int:
        return 1

    @property
    def _denom(self) -> int:
        return max(len(self.levels) - 1, 1)

    def encode_value(self, value: object) -> list[float]:
        try:
            idx = self.levels.index(int(_as_float(value, self.field)))
        except ValueError as exc:
            msg = f"{value!r} is not a valid level for {self.field!r}: {self.levels}"
            raise ValueError(msg) from exc
        return [idx / self._denom]

    def decode_value(self, sub: NDArray[np.float64]) -> int:
        idx = int(round(float(sub[0]) * self._denom))
        idx = min(max(idx, 0), len(self.levels) - 1)
        return self.levels[idx]

    def sample(self, rng: np.random.Generator) -> list[float]:
        return [int(rng.integers(len(self.levels))) / self._denom]


@dataclass(frozen=True)
class BoolDim:
    """Two-state flag encoded as a single ``0/1`` dim (e.g. cut on/off).

    ``true_value`` is the field value mapped to ``1.0``; any other value maps to
    ``0.0`` and decodes back to ``false_value``.
    """

    field: str
    true_value: object = True
    false_value: object = False

    @property
    def width(self) -> int:
        return 1

    def encode_value(self, value: object) -> list[float]:
        return [1.0 if value == self.true_value else 0.0]

    def decode_value(self, sub: NDArray[np.float64]) -> object:
        return self.true_value if float(sub[0]) >= 0.5 else self.false_value

    def sample(self, rng: np.random.Generator) -> list[float]:
        return [float(int(rng.integers(2)))]


@dataclass(frozen=True)
class ContinuousDim:
    """Scalar continuous field min-max normalized to ``[0, 1]``."""

    field: str
    low: float = 0.0
    high: float = 1.0

    @property
    def width(self) -> int:
        return 1

    def _norm(self, value: float) -> float:
        span = self.high - self.low
        u = (float(value) - self.low) / span if span else 0.0
        return float(min(max(u, 0.0), 1.0))

    def _denorm(self, u: float) -> float:
        u = float(min(max(u, 0.0), 1.0))
        return u * (self.high - self.low) + self.low

    def encode_value(self, value: object) -> list[float]:
        return [self._norm(_as_float(value, self.field))]

    def decode_value(self, sub: NDArray[np.float64]) -> float:
        return self._denorm(float(sub[0]))

    def sample(self, rng: np.random.Generator) -> list[float]:
        return [float(rng.random())]


@dataclass(frozen=True)
class ContinuousDictDim:
    """Dict-valued continuous field (e.g. ``p`` solver params) -> one dim per key.

    ``keys`` is an ordered tuple of ``(name, low, high)``; decode reconstructs a
    dict with exactly those keys.
    """

    field: str
    keys: tuple[tuple[str, float, float], ...]

    def __init__(
        self,
        field: str,
        bounds: Mapping[str, tuple[float, float]],
    ) -> None:
        object.__setattr__(self, "field", field)
        object.__setattr__(
            self,
            "keys",
            tuple((name, float(lo), float(hi)) for name, (lo, hi) in bounds.items()),
        )

    @property
    def width(self) -> int:
        return len(self.keys)

    def encode_value(self, value: object) -> list[float]:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"field {self.field!r} must be a mapping, got {type(value).__name__}"
            )
        out: list[float] = []
        for name, low, high in self.keys:
            v = _as_float(value.get(name, low), self.field)
            span = high - low
            u = (v - low) / span if span else 0.0
            out.append(float(min(max(u, 0.0), 1.0)))
        return out

    def decode_value(self, sub: NDArray[np.float64]) -> dict[str, float]:
        out: dict[str, float] = {}
        for i, (name, low, high) in enumerate(self.keys):
            u = float(min(max(float(sub[i]), 0.0), 1.0))
            out[name] = u * (high - low) + low
        return out

    def sample(self, rng: np.random.Generator) -> list[float]:
        return [float(rng.random()) for _ in self.keys]


Dim = CategoricalDim | OrdinalDim | BoolDim | ContinuousDim | ContinuousDictDim


# ── Search space ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Phase1Space:
    """Restricted Phase-1 ``Phi`` subspace + its numeric encoding.

    ``base`` holds the fixed field values (e.g. ``m``/``v``/``s``/``rho``); the
    searched fields are described by ``dims`` (in encoding order).  Decoded
    :class:`Phi` instances are produced via :func:`dataclasses.replace` on
    ``base``, so non-searched fields are always restored exactly.
    """

    base: Phi
    dims: tuple[Dim, ...]

    @property
    def dim(self) -> int:
        """Total length of the encoded numeric vector."""
        return sum(d.width for d in self.dims)

    def encode(self, phi: Phi) -> NDArray[np.float64]:
        """Encode ``phi`` into a stable numeric vector in ``[0, 1]^d``."""
        flat = phi.to_flat_dict()
        out: list[float] = []
        for d in self.dims:
            out.extend(d.encode_value(flat[d.field]))
        return np.asarray(out, dtype=np.float64)

    def decode(self, vec: NDArray[np.float64] | Sequence[float]) -> Phi:
        """Decode a numeric vector back into a :class:`Phi` on ``base``."""
        arr = np.asarray(vec, dtype=np.float64).ravel()
        updates: dict[str, Any] = {}
        i = 0
        for d in self.dims:
            sub = arr[i : i + d.width]
            updates[d.field] = d.decode_value(sub)
            i += d.width
        return replace(self.base, **updates)

    def sample_vector(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Sample one valid encoded vector (valid one-hot/level/bool blocks)."""
        out: list[float] = []
        for d in self.dims:
            out.extend(d.sample(rng))
        return np.asarray(out, dtype=np.float64)

    def candidate_pool(
        self, n: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return ``n`` random valid encoded candidate vectors, shape ``[n, d]``."""
        if n <= 0:
            return np.empty((0, self.dim), dtype=np.float64)
        return np.vstack([self.sample_vector(rng) for _ in range(n)])

    def random_phi(self, rng: np.random.Generator) -> Phi:
        """Sample a random :class:`Phi` from the restricted subspace."""
        return self.decode(self.sample_vector(rng))


def default_phase1_space(base: Phi | None = None) -> Phase1Space:
    """Return the canonical restricted Phase-1 subspace.

    Searched fields (all four encoder kinds exercised):

    - ``c``  -> whitelisted cut on/off (bool ``0/1``).
    - ``d``  -> optional decomposition flag (categorical one-hot).
    - ``h``  -> heuristic intensity (ordinal levels).
    - ``p``  -> SCIP parameter knobs (continuous, ``[0, 1]`` normalized).

    Fields ``m`` / ``v`` / ``s`` / ``rho`` are held fixed at their base values.
    The ``p`` bounds are derived from :data:`opop.proposer.params.CURATED_PARAMS`
    so the decoded keys exactly match real SCIP parameter paths.
    """
    template = base if base is not None else Phi()
    p_bounds: dict[str, tuple[float, float]] = {
        knob.key: (min(knob.values), max(knob.values)) for knob in CURATED_PARAMS
    }
    dims: tuple[Dim, ...] = (
        BoolDim("c", true_value="cuts_on", false_value="cuts_off"),
        CategoricalDim("d", ["none", "benders", "dw"]),
        OrdinalDim("h", [0, 1, 2]),
        ContinuousDictDim("p", p_bounds),
    )
    # Place the searched fields at valid in-space defaults (decode overwrites
    # them; this only matters if ``encode(base)`` is called directly).
    template = replace(
        template,
        c="cuts_off",
        d="none",
        h=0,
        p={knob.key: float(min(knob.values)) for knob in CURATED_PARAMS},
    )
    return Phase1Space(base=template, dims=dims)
