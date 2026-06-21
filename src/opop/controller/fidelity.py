"""Multi-fidelity layers, a fidelity-correlation GATE, and cost-aware MFKG.

This module turns the existing ``Phi.s`` ordinal field into a *fidelity layer*
selector, runs a Spearman-ρ correlation study between a cheap low-fidelity
ranking and the expensive high-fidelity ranking, and exposes a cost-aware
multi-fidelity Knowledge-Gradient controller that **only activates when the
study clears the ρ ≥ 0.5 gate**.

Design
------
``Phi.s`` already exists in :class:`opop.model.state.Phi` as an *ordinal* field
(type tag ``"ordinal"``); per the task constraints ``state.py`` is not modified.
Here ``s`` is interpreted as an index into :data:`FIDELITY_LAYERS`, the ordered
tuple of the seven layers

    presolve < lp_relax < root_cuts < short_time < sub_instance < heuristic
    < full_solve

from cheapest/lowest fidelity to most-expensive/highest fidelity
(``full_solve`` is the target fidelity).

The encoder dimension is added via :func:`fidelity_dim` /
:func:`fidelity_phase1_space` — a *new* space builder that mirrors the existing
:class:`~opop.controller.encoder.OrdinalDim` ``h`` dimension — rather than
mutating :func:`~opop.controller.encoder.default_phase1_space`, so the existing
controller/router tests stay green.

Low-fidelity *evaluators* reuse the existing solver kernels unchanged
(:func:`fidelity_solve`): each layer is a reduced time budget plus pass-through
SCIP limits (node caps, presolve/gap limits).  No ``separating/<name>/...``
knobs are used, so the kernel's class-B separator whitelist is never tripped.

The GATE (:func:`fidelity_correlation`) computes Spearman ρ across methods
between the low- and high-fidelity scores; :class:`MFKGController` is available
but warns and falls back to single-fidelity whenever ρ < 0.5 (or ρ is
undefined).  Enabling MFKG without ρ ≥ 0.5 evidence is impossible by
construction.

The BoTorch MFKG path (``qMultiFidelityKnowledgeGradient`` +
``AffineFidelityCostModel`` + ``InverseCostWeightedUtility`` +
``project_to_target_fidelity``) is imported lazily, mirroring
:mod:`opop.controller.botorch_rungs`, so this module stays importable without
BoTorch and the MFKG tests skip cleanly when it is absent.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import torch
from scipy.stats import spearmanr

from opop.model.state import Phi

from .encoder import OrdinalDim, Phase1Space, default_phase1_space

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from opop.model.ir import MILP
    from opop.model.state import SolveTrace

__all__ = [
    "FIDELITY_LAYERS",
    "FIDELITY_SPECS",
    "MFKG_RHO_THRESHOLD",
    "FidelityCorrelationReport",
    "FidelityKernel",
    "FidelityLayer",
    "FidelitySpec",
    "MFKGController",
    "fidelity_column",
    "fidelity_correlation",
    "fidelity_cost",
    "fidelity_dim",
    "fidelity_index",
    "fidelity_phase1_space",
    "fidelity_solve",
    "layer_for",
    "mfkg_available",
    "normalized_fidelity",
    "resolve_layer",
    "should_enable_mfkg",
]


# ── Fidelity layers ──────────────────────────────────────────────────────────


class FidelityLayer(Enum):
    """A single fidelity layer selected by ``Phi.s``.

    Ordered cheapest → most expensive in :data:`FIDELITY_LAYERS`; ``FULL_SOLVE``
    is the *target* (highest) fidelity that the MFKG ``project`` operator maps
    every candidate to.
    """

    PRESOLVE = "presolve"
    LP_RELAX = "lp_relax"
    ROOT_CUTS = "root_cuts"
    SHORT_TIME = "short_time"
    SUB_INSTANCE = "sub_instance"
    HEURISTIC = "heuristic"
    FULL_SOLVE = "full_solve"


#: The fidelity ladder in increasing cost/fidelity order.  ``Phi.s`` indexes
#: this tuple (clamped); ``FULL_SOLVE`` (last) is the target fidelity.
FIDELITY_LAYERS: tuple[FidelityLayer, ...] = (
    FidelityLayer.PRESOLVE,
    FidelityLayer.LP_RELAX,
    FidelityLayer.ROOT_CUTS,
    FidelityLayer.SHORT_TIME,
    FidelityLayer.SUB_INSTANCE,
    FidelityLayer.HEURISTIC,
    FidelityLayer.FULL_SOLVE,
)

#: The highest / target fidelity layer.
TARGET_LAYER: FidelityLayer = FIDELITY_LAYERS[-1]


@dataclass(frozen=True, slots=True)
class FidelitySpec:
    """How one fidelity layer reduces the full-solve budget.

    Attributes:
        layer: The layer this spec configures.
        time_fraction: Fraction of the full time budget granted to this layer
            (``1.0`` for ``full_solve``).
        node_limit: Optional SCIP ``limits/nodes`` cap (``None`` = no extra cap;
            ``1`` = root node only).
        extra_params: Pass-through SCIP knobs merged into ``phi.p`` for the
            solve (never a ``separating/<name>/...`` key, so the kernel's
            class-B separator whitelist is not tripped).
        cost: Relative evaluation cost (``full_solve == 1.0``), monotonically
            increasing along :data:`FIDELITY_LAYERS`.
    """

    layer: FidelityLayer
    time_fraction: float
    node_limit: int | None
    extra_params: dict[str, float]
    cost: float


def _spec(
    layer: FidelityLayer,
    *,
    time_fraction: float,
    node_limit: int | None,
    cost: float,
    extra: dict[str, float] | None = None,
) -> FidelitySpec:
    params: dict[str, float] = dict(extra or {})
    if node_limit is not None:
        params["limits/nodes"] = float(node_limit)
    return FidelitySpec(
        layer=layer,
        time_fraction=float(time_fraction),
        node_limit=node_limit,
        extra_params=params,
        cost=float(cost),
    )


#: Per-layer budget reductions.  Costs and time fractions increase monotonically
#: with fidelity; the cheap layers cap nodes (root-only / small B&B), the
#: ``heuristic`` layer stops at a loose gap, and ``full_solve`` is unconstrained.
FIDELITY_SPECS: dict[FidelityLayer, FidelitySpec] = {
    FidelityLayer.PRESOLVE: _spec(
        FidelityLayer.PRESOLVE,
        time_fraction=0.05,
        node_limit=1,
        cost=0.02,
        extra={"presolving/maxrounds": -1.0},
    ),
    FidelityLayer.LP_RELAX: _spec(
        FidelityLayer.LP_RELAX,
        time_fraction=0.05,
        node_limit=1,
        cost=0.05,
        extra={"presolving/maxrounds": 0.0},
    ),
    FidelityLayer.ROOT_CUTS: _spec(
        FidelityLayer.ROOT_CUTS,
        time_fraction=0.10,
        node_limit=1,
        cost=0.10,
    ),
    FidelityLayer.SHORT_TIME: _spec(
        FidelityLayer.SHORT_TIME,
        time_fraction=0.20,
        node_limit=None,
        cost=0.20,
    ),
    FidelityLayer.SUB_INSTANCE: _spec(
        FidelityLayer.SUB_INSTANCE,
        time_fraction=0.30,
        node_limit=200,
        cost=0.30,
    ),
    FidelityLayer.HEURISTIC: _spec(
        FidelityLayer.HEURISTIC,
        time_fraction=0.50,
        node_limit=None,
        cost=0.55,
        extra={"limits/gap": 0.05},
    ),
    FidelityLayer.FULL_SOLVE: _spec(
        FidelityLayer.FULL_SOLVE,
        time_fraction=1.0,
        node_limit=None,
        cost=1.0,
    ),
}


def fidelity_index(layer: FidelityLayer) -> int:
    """Return the ladder index of ``layer`` (``0`` cheapest, ``6`` target)."""
    return FIDELITY_LAYERS.index(layer)


def layer_for(s: int) -> FidelityLayer:
    """Map an integer ``Phi.s`` to a :class:`FidelityLayer` (index, clamped).

    ``s`` is clamped into ``[0, len(FIDELITY_LAYERS) - 1]`` so any out-of-range
    fidelity level resolves to a valid layer rather than raising.
    """
    idx = max(0, min(int(s), len(FIDELITY_LAYERS) - 1))
    return FIDELITY_LAYERS[idx]


def resolve_layer(value: object) -> FidelityLayer:
    """Coerce an enum / layer-name / ``Phi.s`` index to a :class:`FidelityLayer`.

    Accepts a :class:`FidelityLayer`, an ``int`` ``Phi.s`` index (clamped), or a
    layer ``.value`` / member name string.  Any other input raises.
    """
    if isinstance(value, FidelityLayer):
        return value
    if isinstance(value, bool):  # guard: bool is an int subclass
        return layer_for(int(value))
    if isinstance(value, int):
        return layer_for(value)
    if isinstance(value, str):
        try:
            return FidelityLayer(value)
        except ValueError:
            pass
        # Accept enum member names too (e.g. "FULL_SOLVE").
        try:
            return FidelityLayer[value.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown fidelity layer {value!r}") from exc
    raise TypeError(f"cannot resolve fidelity layer from {value!r}")


def normalized_fidelity(layer: FidelityLayer) -> float:
    """Return ``layer`` normalized to ``[0, 1]`` (``full_solve == 1.0``).

    This equals the encoder value of the fidelity :class:`OrdinalDim` for the
    layer's level, so it is exactly the column the BoTorch
    ``AffineFidelityCostModel`` / ``project_to_target_fidelity`` operate on.
    """
    denom = max(len(FIDELITY_LAYERS) - 1, 1)
    return fidelity_index(layer) / denom


def fidelity_cost(layer: FidelityLayer) -> float:
    """Return the relative evaluation cost of ``layer`` (``full_solve == 1.0``)."""
    return FIDELITY_SPECS[layer].cost


# ── Encoder dimension + fidelity-aware search space ──────────────────────────


def fidelity_dim() -> OrdinalDim:
    """Return the ``s`` fidelity :class:`OrdinalDim` (levels ``0..6``).

    Mirrors the existing ordinal ``h`` dimension; decode snaps to the nearest
    level and :func:`layer_for` maps the decoded ``Phi.s`` to a layer.
    """
    return OrdinalDim("s", list(range(len(FIDELITY_LAYERS))))


def fidelity_phase1_space(base: Phi | None = None) -> Phase1Space:
    """Return :func:`default_phase1_space` extended with the ``s`` fidelity dim.

    The fidelity dimension is appended last (so its encoded column is
    ``space.dim - 1``; see :func:`fidelity_column`).  A *new* builder is used
    rather than mutating ``default_phase1_space`` so the existing router/encoder
    tests, which assume the canonical mixed space, are unaffected.
    """
    default = default_phase1_space(base)
    new_base = replace(default.base, s=int(default.base.s))
    return Phase1Space(base=new_base, dims=(*default.dims, fidelity_dim()))


def fidelity_column(space: Phase1Space) -> int:
    """Return the encoded column index of the ``s`` fidelity dim in ``space``.

    Raises:
        ValueError: If ``space`` has no ``s`` dimension.
    """
    col = 0
    for dim in space.dims:
        if getattr(dim, "field", None) == "s":
            return col
        col += dim.width
    raise ValueError("space has no fidelity ('s') dimension")


# ── Low-fidelity evaluators (reuse the existing solver kernel) ───────────────


@runtime_checkable
class FidelityKernel(Protocol):
    """Structural type of a solver kernel usable by :func:`fidelity_solve`.

    Satisfied by :class:`opop.solver.scip.ScipKernel` (and the HiGHS/CP-SAT
    kernels) without importing the solver layer into the controller.
    """

    def solve(
        self,
        ir: MILP,
        phi: Phi,
        *,
        time_limit: float,
        memory_limit_mb: int,
        seed: int,
    ) -> SolveTrace:
        """Solve ``ir`` under ``phi`` + budget, returning a trace."""
        ...


def fidelity_solve(
    kernel: FidelityKernel,
    ir: MILP,
    phi: Phi,
    *,
    full_time_limit: float,
    memory_limit_mb: int = 4096,
    seed: int = 0,
    layer: FidelityLayer | str | int | None = None,
    min_time_limit: float = 0.1,
) -> SolveTrace:
    """Solve ``ir`` at a reduced fidelity by reusing ``kernel`` unchanged.

    The fidelity layer is taken from ``layer`` when given, else from ``phi.s``.
    The layer's :class:`FidelitySpec` is applied as (a) an effective time limit
    ``max(min_time_limit, full_time_limit * time_fraction)`` and (b) pass-through
    SCIP knobs merged into a *copy* of ``phi.p`` (``phi`` is never mutated).

    Args:
        kernel: Any :class:`FidelityKernel` (e.g. ``ScipKernel``).
        ir: The MILP instance to solve.
        phi: Design vector; its ``s`` selects the layer unless ``layer`` is set.
        full_time_limit: The full-solve time budget (seconds).
        memory_limit_mb: Per-solve memory ceiling (MiB).
        seed: Solver seed.
        layer: Optional explicit layer override.
        min_time_limit: Floor on the reduced time budget (seconds).

    Returns:
        The :class:`~opop.model.state.SolveTrace` from ``kernel.solve``.
    """
    resolved = resolve_layer(layer) if layer is not None else layer_for(phi.s)
    spec = FIDELITY_SPECS[resolved]
    eff_time = (
        float(full_time_limit)
        if resolved is TARGET_LAYER
        else max(float(min_time_limit), float(full_time_limit) * spec.time_fraction)
    )
    merged_p: dict[str, float] = {**phi.p, **spec.extra_params}
    phi_eff = replace(phi, p=merged_p, s=fidelity_index(resolved))
    return kernel.solve(
        ir,
        phi_eff,
        time_limit=eff_time,
        memory_limit_mb=memory_limit_mb,
        seed=seed,
    )


# ── Fidelity-correlation GATE (Spearman ρ across methods) ────────────────────


#: Minimum Spearman ρ between low- and high-fidelity rankings to enable MFKG.
MFKG_RHO_THRESHOLD: float = 0.5


def should_enable_mfkg(rho: float, threshold: float = MFKG_RHO_THRESHOLD) -> bool:
    """Return ``True`` iff ``rho`` is finite and clears ``threshold`` (≥).

    Fail-closed: a ``nan`` / non-finite ρ (too few methods, constant scores)
    never enables MFKG.
    """
    return bool(math.isfinite(rho) and rho >= threshold)


@dataclass(frozen=True, slots=True)
class FidelityCorrelationReport:
    """Result of one low-vs-high fidelity Spearman-ρ study.

    Attributes:
        rho: Spearman rank correlation across methods (``nan`` if undefined).
        p_value: Two-sided p-value from :func:`scipy.stats.spearmanr`.
        n_methods: Number of methods paired across both fidelities.
        low_layer / high_layer: The compared layers (by ``.value`` name).
        threshold: The ρ gate applied.
        enable_mfkg: ``True`` iff ρ ≥ ``threshold`` (fail-closed on ``nan``).
        methods: Ordered method names (paired).
        low_scores / high_scores: Per-method scores, index-aligned to ``methods``.
        reason: Human-readable gate rationale.
    """

    rho: float
    p_value: float
    n_methods: int
    low_layer: str
    high_layer: str
    threshold: float
    enable_mfkg: bool
    methods: tuple[str, ...] = ()
    low_scores: tuple[float, ...] = ()
    high_scores: tuple[float, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dict of all fields."""
        return {
            "rho": self.rho,
            "p_value": self.p_value,
            "n_methods": self.n_methods,
            "low_layer": self.low_layer,
            "high_layer": self.high_layer,
            "threshold": self.threshold,
            "enable_mfkg": self.enable_mfkg,
            "methods": list(self.methods),
            "low_scores": list(self.low_scores),
            "high_scores": list(self.high_scores),
            "reason": self.reason,
        }

    def to_json(self) -> str:
        """Serialise to pretty, key-sorted JSON (no trailing newline)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=True)


# ``dev_results`` accepted shapes:
#  * Mapping[method, Mapping[layer_key, score]]
#  * Sequence[Mapping] of records with method + fidelity (s / fidelity / layer)
#    + score (score / reward).
_DevResults = (
    Mapping[str, Mapping[Any, float]] | Sequence[Mapping[str, Any]]
)


def _normalize_dev_results(
    dev_results: _DevResults,
) -> dict[str, dict[FidelityLayer, float]]:
    """Coerce supported ``dev_results`` containers to ``{method: {layer: score}}``.

    The top-level ``isinstance`` discriminates the ``Mapping`` form
    (``{method: {layer: score}}``) from the ``Sequence`` form (a list of
    ``{method, fidelity, score}`` records); both element shapes are guaranteed
    by the ``_DevResults`` type, so no further runtime guards are needed.
    """
    out: dict[str, dict[FidelityLayer, float]] = {}
    if isinstance(dev_results, Mapping):
        for method, per_layer in dev_results.items():
            bucket = out.setdefault(str(method), {})
            for layer_key, score in per_layer.items():
                bucket[resolve_layer(layer_key)] = float(score)
        return out
    for rec in dev_results:
        method = str(rec.get("method", ""))
        fidelity = rec.get("fidelity", rec.get("layer", rec.get("s")))
        if fidelity is None:
            raise ValueError(f"record missing fidelity/layer/s: {dict(rec)!r}")
        raw_score = rec.get("score", rec.get("reward"))
        if raw_score is None:
            raise ValueError(f"record missing score/reward: {dict(rec)!r}")
        out.setdefault(method, {})[resolve_layer(fidelity)] = float(raw_score)
    return out


def _default_low_layer(
    per_method: Mapping[str, Mapping[FidelityLayer, float]], high: FidelityLayer
) -> FidelityLayer | None:
    """Pick the cheapest layer (other than ``high``) present in every method."""
    common: set[FidelityLayer] | None = None
    for layers in per_method.values():
        present = set(layers.keys())
        common = present if common is None else (common & present)
    if not common:
        return None
    for layer in FIDELITY_LAYERS:  # cheapest first
        if layer is not high and layer in common:
            return layer
    return None


def fidelity_correlation(
    dev_results: _DevResults,
    *,
    low: FidelityLayer | str | int | None = None,
    high: FidelityLayer | str | int = FidelityLayer.FULL_SOLVE,
    threshold: float = MFKG_RHO_THRESHOLD,
) -> FidelityCorrelationReport:
    """Spearman-ρ study between low- and high-fidelity rankings across methods.

    For every method that has BOTH a low- and high-fidelity score, the paired
    ``(low, high)`` scores are collected and Spearman ρ is computed across
    methods.  A high ρ means the cheap proxy ranks configurations like the
    expensive target fidelity, so multi-fidelity acquisition is trustworthy.

    Both fidelities MUST share orientation (e.g. both rewards where higher is
    better, or both primal integrals where lower is better) for the sign of ρ
    to be meaningful.

    Args:
        dev_results: ``{method: {layer: score}}`` or a sequence of
            ``{method, fidelity, score}`` records.
        low: Low-fidelity layer; defaults to the cheapest layer common to all
            methods (excluding ``high``).
        high: High / target fidelity layer (default ``full_solve``).
        threshold: ρ gate for enabling MFKG.

    Returns:
        A :class:`FidelityCorrelationReport`.  ``enable_mfkg`` is fail-closed:
        ``False`` whenever ρ is undefined (``< 2`` methods or constant scores).
    """
    high_layer = resolve_layer(high)
    per_method = _normalize_dev_results(dev_results)
    low_layer = (
        resolve_layer(low) if low is not None else _default_low_layer(per_method, high_layer)
    )

    methods: list[str] = []
    low_scores: list[float] = []
    high_scores: list[float] = []
    if low_layer is not None:
        for method in sorted(per_method):
            layers = per_method[method]
            if low_layer in layers and high_layer in layers:
                methods.append(method)
                low_scores.append(layers[low_layer])
                high_scores.append(layers[high_layer])

    low_name = low_layer.value if low_layer is not None else "none"
    n = len(methods)
    if low_layer is None:
        rho, p_value, reason = (
            math.nan,
            math.nan,
            "no common low-fidelity layer across methods; MFKG disabled (fail-closed)",
        )
    elif n < 2:
        rho, p_value, reason = (
            math.nan,
            math.nan,
            f"only {n} paired method(s) for {low_name} vs {high_layer.value}; "
            + "ρ undefined, MFKG disabled (fail-closed)",
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant-input → nan (handled below)
            stat, pval = cast(
                "tuple[float, float]",
                spearmanr(np.asarray(low_scores), np.asarray(high_scores)),
            )
        rho = float(stat)
        p_value = float(pval)
        if not math.isfinite(rho):
            reason = (
                f"Spearman ρ undefined (constant scores) for {low_name} vs "
                + f"{high_layer.value}; MFKG disabled (fail-closed)"
            )
        elif rho >= threshold:
            reason = (
                f"ρ={rho:.3f} ≥ {threshold:g} across {n} methods "
                + f"({low_name}→{high_layer.value}): enable cost-aware MFKG"
            )
        else:
            reason = (
                f"ρ={rho:.3f} < {threshold:g} across {n} methods "
                + f"({low_name}→{high_layer.value}): keep single-fidelity "
                + "(negative result recorded)"
            )

    return FidelityCorrelationReport(
        rho=rho,
        p_value=p_value,
        n_methods=n,
        low_layer=low_name,
        high_layer=high_layer.value,
        threshold=float(threshold),
        enable_mfkg=should_enable_mfkg(rho, threshold),
        methods=tuple(methods),
        low_scores=tuple(low_scores),
        high_scores=tuple(high_scores),
        reason=reason,
    )


# ── Cost-aware MFKG controller (gated by ρ ≥ 0.5) ────────────────────────────


def mfkg_available() -> bool:
    """Return whether the BoTorch MFKG stack can be imported (lazily)."""
    from .botorch_rungs import botorch_available

    return botorch_available()


def _extract_model(surrogate: object) -> Any:
    """Extract the fitted BoTorch model from a surrogate wrapper (``.model``).

    Mirrors :func:`opop.controller.botorch_rungs._model_of` without importing a
    private symbol; satisfied by :class:`~opop.controller.botorch_rungs.BoTorchGPSurrogate`.
    """
    model = getattr(surrogate, "model", None)
    if model is None:
        model = getattr(surrogate, "_model", None)
    if model is None:
        raise TypeError(
            "MFKG requires a fitted BoTorch surrogate exposing `.model` "
            + f"(e.g. BoTorchGPSurrogate); got {type(surrogate).__name__}"
        )
    return model


@dataclass
class MFKGController:
    """Cost-aware multi-fidelity Knowledge-Gradient controller (ρ-gated).

    The controller is *always constructible* but only **activates** when the
    fidelity-correlation gate passes (``rho >= threshold`` and finite).  When
    the gate fails it emits a :class:`UserWarning` and delegates to a
    single-fidelity ``fallback`` acquisition (default
    :class:`~opop.controller.ladder.LadderEI`), so it satisfies the
    :class:`~opop.controller.protocol.Acquisition` protocol either way.

    The BoTorch acquisition
    (``qMultiFidelityKnowledgeGradient`` + ``AffineFidelityCostModel`` +
    ``InverseCostWeightedUtility`` + ``project_to_target_fidelity``) is built
    lazily, so importing/constructing this controller never requires BoTorch;
    only :meth:`build_acquisition` / :meth:`propose` (the *enabled* path) do.

    Attributes:
        rho: Spearman ρ from the correlation study (drives the gate).
        dim: Encoded design dimensionality (includes the fidelity column).
        fidelity_col: Encoded column index of the fidelity dimension.
        threshold: ρ gate (default :data:`MFKG_RHO_THRESHOLD`).
        fixed_cost: ``AffineFidelityCostModel`` fixed cost.
        num_fantasies / num_restarts / raw_samples / seed: MFKG/optimiser knobs.
        fallback: Single-fidelity acquisition used when the gate fails.
    """

    rho: float
    dim: int
    fidelity_col: int
    threshold: float = MFKG_RHO_THRESHOLD
    fixed_cost: float = 5.0
    num_fantasies: int = 64
    num_restarts: int = 5
    raw_samples: int = 128
    seed: int = 0
    fallback: Any = None
    _warned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fallback is None:
            from .ladder import LadderEI

            self.fallback = LadderEI()

    # ── factories ───────────────────────────────────────────────────────────

    @classmethod
    def from_space(
        cls, space: Phase1Space, *, rho: float, **kwargs: Any
    ) -> MFKGController:
        """Build a controller for ``space`` (derives ``dim`` + fidelity column)."""
        return cls(
            rho=float(rho),
            dim=space.dim,
            fidelity_col=fidelity_column(space),
            **kwargs,
        )

    @classmethod
    def from_correlation(
        cls,
        space: Phase1Space,
        report: FidelityCorrelationReport,
        **kwargs: Any,
    ) -> MFKGController:
        """Build a controller from a :class:`FidelityCorrelationReport`.

        Inherits the report's ρ and threshold so the gate matches the study.
        """
        kwargs.setdefault("threshold", report.threshold)
        return cls.from_space(space, rho=report.rho, **kwargs)

    # ── gate ──────────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """``True`` iff the ρ gate passes (finite ρ ≥ threshold)."""
        return should_enable_mfkg(self.rho, self.threshold)

    def _warn_fallback(self) -> None:
        if not self._warned:
            warnings.warn(
                f"MFKG gate not met (ρ={self.rho:.3f} < {self.threshold:g} or "
                + "undefined); falling back to single-fidelity acquisition.",
                UserWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "_warned", True)

    # ── BoTorch MFKG construction (lazy; enabled path only) ──────────────────

    def _project(self) -> Any:
        from botorch.acquisition.utils import (  # pyright: ignore[reportMissingImports]
            project_to_target_fidelity,
        )

        target = {self.fidelity_col: 1.0}

        def project(X: Any) -> Any:
            return project_to_target_fidelity(X=X, target_fidelities=target)

        return project

    def _cost_aware_utility(self) -> Any:
        from botorch.acquisition.cost_aware import (  # pyright: ignore[reportMissingImports]
            InverseCostWeightedUtility,
        )
        from botorch.models import (  # pyright: ignore[reportMissingImports]
            AffineFidelityCostModel,
        )

        cost_model = AffineFidelityCostModel(
            fidelity_weights={self.fidelity_col: 1.0}, fixed_cost=self.fixed_cost
        )
        return InverseCostWeightedUtility(cost_model=cost_model)

    def _current_value(self, model: Any) -> Any:
        """Best-effort current target-fidelity posterior-mean value (or ``None``)."""
        try:
            from botorch.acquisition import (  # pyright: ignore[reportMissingImports]
                PosteriorMean,
            )
            from botorch.acquisition.fixed_feature import (  # pyright: ignore[reportMissingImports]
                FixedFeatureAcquisitionFunction,
            )
            from botorch.optim import optimize_acqf  # pyright: ignore[reportMissingImports]

            curr_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(model),
                d=self.dim,
                columns=[self.fidelity_col],
                values=[1.0],
            )
            non_fid = self.dim - 1
            lo = torch.zeros(non_fid, dtype=torch.float64)
            hi = torch.ones(non_fid, dtype=torch.float64)
            _, value = optimize_acqf(
                acq_function=curr_acqf,
                bounds=torch.stack([lo, hi]),
                q=1,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
            return value
        except Exception:  # pragma: no cover - defensive (KG value still valid w/o it)
            return None

    def build_acquisition(self, surrogate: object) -> Any:
        """Build the ``qMultiFidelityKnowledgeGradient`` acquisition.

        Raises:
            RuntimeError: If the ρ gate is not met (enabling MFKG without
                ρ ≥ threshold evidence is forbidden).
            ImportError: If BoTorch is not installed.
        """
        if not self.enabled:
            raise RuntimeError(
                f"refusing to build MFKG: ρ={self.rho} does not clear "
                + f"threshold {self.threshold} (need ρ ≥ {self.threshold})"
            )
        if not mfkg_available():
            raise ImportError(
                "MFKG requires the optional 'botorch' package; install botorch "
                + "(and gpytorch) for qMultiFidelityKnowledgeGradient"
            )
        from botorch.acquisition.knowledge_gradient import (  # pyright: ignore[reportMissingImports]
            qMultiFidelityKnowledgeGradient,
        )

        model = _extract_model(surrogate)
        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=self.num_fantasies,
            current_value=self._current_value(model),
            cost_aware_utility=self._cost_aware_utility(),
            project=self._project(),
        )

    # ── Acquisition protocol ─────────────────────────────────────────────────

    def propose(
        self, surrogate: object, X_candidates: NDArray[np.float64], *, seed: int | None = None
    ) -> tuple[NDArray[np.float64], float]:
        """Optimise MFKG over ``[0, 1]^dim`` and snap to the nearest candidate."""
        del seed  # optimize_acqf manages its own restarts; seed unused here
        from botorch.optim import optimize_acqf  # pyright: ignore[reportMissingImports]

        acq = self.build_acquisition(surrogate)
        pool = torch.as_tensor(X_candidates, dtype=torch.float64)
        if pool.ndim == 1:
            pool = pool.reshape(-1, 1)
        d = int(pool.shape[1])
        bounds = torch.stack(
            [torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)]
        )
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
        out: NDArray[np.float64] = pool[best_idx].cpu().numpy()
        return out, float(cast("torch.Tensor", value).item())

    def __call__(
        self,
        surrogate: object,
        X_candidates: NDArray[np.float64],
        *,
        y_best: float | None = None,
        kappa: float = 2.0,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Acquisition entry point: MFKG when gated on, else the fallback.

        When the ρ gate fails, warns once and delegates to ``self.fallback``
        (single-fidelity), so MFKG can never run without ρ ≥ threshold evidence.
        """
        if not self.enabled:
            self._warn_fallback()
            return self.fallback(
                surrogate, X_candidates, y_best=y_best, kappa=kappa, seed=seed
            )
        return self.propose(surrogate, X_candidates, seed=seed)
