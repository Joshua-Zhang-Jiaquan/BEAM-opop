"""Modeling-agent dataset loader: structured JSON model spec -> symbolic IR (task 35).

Modeling-agent benchmarks (NL4Opt, NLP4LP, MAMO, IndustryOR, OptiBench, ...) pair a
natural-language optimisation problem with a *labeled* optimum and a structured
model. This module reads a small, offline, dataset-agnostic JSON encoding of those
items and converts each into an OPOP :class:`~opop.model.ir.MILP` (with the task-30
quadratic extension and/or the task-31 separable-nonlinear metadata when present),
wrapped as a :class:`~opop.bench.cleaning.CleaningItem` so the labels can be
re-verified by the solver-backed cleaning harness
(:func:`opop.bench.cleaning.verify_and_clean`).

JSON schema (one file is ``{"items": [ <item>, ... ]}``)
--------------------------------------------------------
Each ``<item>`` is::

    {
      "id": "nl4opt/production",          # globally unique item id
      "dataset": "nl4opt",                # originating dataset tag
      "natural_language": "...",          # informational (kept in IR metadata)
      "sense": "maximize",                # "minimize" | "maximize"
      "labeled_optimum": 36.0,            # the optimum to re-verify
      "model_spec": {
        "variables":   [{"name": "a", "type": "continuous", "lower": 0.0, "upper": 10.0}, ...],
        "constraints": [{"name": "c0", "linear": {"a": 1.0}, "sense": "<=", "rhs": 14.0,
                         "quadratic": [["a", "a", 1.0]],            # optional
                         "nonlinear": [{"func": "square", "var": "a", "coeff": 1.0}]}],  # optional
        "objective":   {"linear": {"a": 3.0}, "offset": 0.0,
                        "quadratic": [["a", "b", 2.0]],             # optional
                        "nonlinear": [{"func": "square", "var": "a", "coeff": 1.0}]}     # optional
      }
    }

Variable ``type`` is ``binary`` / ``integer`` / ``continuous`` (binary defaults to
``[0, 1]``, others to ``[0, +inf)``). A ``quadratic`` entry ``[v1, v2, c]`` adds
``c * v1 * v2`` (to the objective or that constraint's LHS); a ``nonlinear`` entry
adds a separable :class:`~opop.model.minlp.NonlinearTerm` (``square``/``exp``
convex, ``log``/``sqrt`` concave). The produced IR routes through the capability
registry exactly like any other instance: plain MILP solves directly, a quadratic
spec is claimed by the MIQP/MIQCP adapter, and a separable-nonlinear spec by the
structured-MINLP adapter.

This module is offline + stdlib-only (no dataset download). It represents two
datasets here (:data:`MODELING_DATASETS` = NL4Opt + OptiBench) via committed
fixtures; the loader itself is dataset-agnostic, so adding a dataset is just
another fixture file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opop.bench.cleaning import CleaningItem
from opop.bench.registry import BenchmarkEntry
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    QuadraticExtension,
    QuadraticTerm,
    Variable,
    VarType,
)
from opop.model.minlp import NONLINEAR_TERMS_KEY, OBJECTIVE_TARGET, NonlinearTerm

__all__ = [
    "MODELING_CATALOG",
    "MODELING_DATASETS",
    "ModelSpecError",
    "ModelingDataset",
    "build_entries",
    "load_modeling_items",
    "loads_modeling_items",
]

#: Representative modeling-agent datasets shipped as offline fixtures.
MODELING_DATASETS: tuple[str, ...] = ("nl4opt", "optibench")

_REGISTRY_TIME_LIMIT_SEC = 300.0
_REGISTRY_PHASE = 6
_REGISTRY_THESIS = "T3"
_REGISTRY_BASELINE = "scip_default"
_REGISTRY_LICENSE = "MIT"

_VTYPE_BY_NAME: dict[str, VarType] = {
    "binary": VarType.BINARY,
    "integer": VarType.INTEGER,
    "continuous": VarType.CONTINUOUS,
}

_SENSE_BY_NAME: dict[str, ObjSense] = {
    "minimize": ObjSense.MINIMIZE,
    "maximize": ObjSense.MAXIMIZE,
}


class ModelSpecError(ValueError):
    """Raised when a modeling-agent item's ``model_spec`` is malformed/unsupported."""


def _quadratic_term(entry: Any, item_id: str) -> QuadraticTerm:
    """Build a :class:`QuadraticTerm` from a ``[var1, var2, coeff]`` JSON entry."""
    if len(entry) != 3:
        raise ModelSpecError(
            f"{item_id}: quadratic entry must be [var1, var2, coeff], got {entry!r}"
        )
    return QuadraticTerm(str(entry[0]), str(entry[1]), float(entry[2]))


def _nonlinear_term(entry: Any, target: str, item_id: str) -> NonlinearTerm:
    """Build a :class:`NonlinearTerm` from a ``{func, var, coeff}`` JSON entry."""
    if "func" not in entry or "var" not in entry:
        raise ModelSpecError(
            f"{item_id}: nonlinear entry needs 'func' and 'var', got {entry!r}"
        )
    return NonlinearTerm(
        func=str(entry["func"]),
        var=str(entry["var"]),
        coeff=float(entry.get("coeff", 1.0)),
        target=target,
    )


def _build_variable(raw: Any, item_id: str) -> Variable:
    """Build a :class:`Variable` from a ``{name, type, lower, upper}`` JSON entry."""
    vtype = _VTYPE_BY_NAME.get(str(raw["type"]).lower())
    if vtype is None:
        raise ModelSpecError(
            f"{item_id}: unknown variable type {raw['type']!r} "
            + f"(expected one of {sorted(_VTYPE_BY_NAME)})"
        )
    default_upper = 1.0 if vtype is VarType.BINARY else math.inf
    return Variable(
        name=str(raw["name"]),
        vtype=vtype,
        lower=float(raw.get("lower", 0.0)),
        upper=float(raw.get("upper", default_upper)),
    )


def _build_milp(item_id: str, sense: ObjSense, spec: dict[str, Any], *, metadata: dict[str, Any]) -> MILP:
    """Convert a ``model_spec`` mapping into a :class:`~opop.model.ir.MILP`."""
    variables = tuple(_build_variable(raw, item_id) for raw in spec.get("variables", []))

    constraints: list[LinearConstraint] = []
    constraint_quadratic: dict[str, list[QuadraticTerm]] = {}
    nonlinear_terms: list[NonlinearTerm] = []
    for raw in spec.get("constraints", []):
        cname = str(raw["name"])
        coeffs = {str(k): float(v) for k, v in raw.get("linear", {}).items()}
        constraints.append(
            LinearConstraint(cname, coeffs, ConstraintSense(str(raw["sense"])), float(raw["rhs"]))
        )
        for entry in raw.get("quadratic", []):
            constraint_quadratic.setdefault(cname, []).append(_quadratic_term(entry, item_id))
        for entry in raw.get("nonlinear", []):
            nonlinear_terms.append(_nonlinear_term(entry, cname, item_id))

    obj = spec.get("objective", {})
    obj_coeffs = {str(k): float(v) for k, v in obj.get("linear", {}).items()}
    objective = Objective(coeffs=obj_coeffs, sense=sense, offset=float(obj.get("offset", 0.0)))
    objective_terms = [_quadratic_term(entry, item_id) for entry in obj.get("quadratic", [])]
    for entry in obj.get("nonlinear", []):
        nonlinear_terms.append(_nonlinear_term(entry, OBJECTIVE_TARGET, item_id))

    constraint_terms = {name: tuple(terms) for name, terms in constraint_quadratic.items() if terms}
    extension: QuadraticExtension | None = None
    if objective_terms or constraint_terms:
        extension = QuadraticExtension(
            objective_terms=tuple(objective_terms), constraint_terms=constraint_terms
        )

    full_metadata = dict(metadata)
    if nonlinear_terms:
        full_metadata[NONLINEAR_TERMS_KEY] = tuple(nonlinear_terms)

    return MILP(
        name=item_id,
        variables=variables,
        constraints=tuple(constraints),
        objective=objective,
        metadata=full_metadata,
        quadratic=extension,
    )


def _item_from_raw(raw: Any) -> CleaningItem:
    """Convert one JSON item into a labeled :class:`CleaningItem`."""
    item_id = str(raw["id"])
    sense = _SENSE_BY_NAME.get(str(raw["sense"]).lower())
    if sense is None:
        raise ModelSpecError(
            f"{item_id}: unknown sense {raw['sense']!r} (expected 'minimize' or 'maximize')"
        )
    dataset = str(raw.get("dataset", ""))
    metadata: dict[str, Any] = {
        "source": "modeling_agent",
        "dataset": dataset,
        "natural_language": str(raw.get("natural_language", "")),
    }
    ir = _build_milp(item_id, sense, raw["model_spec"], metadata=metadata)
    return CleaningItem(
        id=item_id,
        ir=ir,
        labeled_optimum=float(raw["labeled_optimum"]),
        sense=ir.objective.sense,
        source_dataset=dataset,
    )


def _items_from_data(data: Any) -> list[CleaningItem]:
    """Convert a parsed ``{"items": [...]}`` payload into labeled items."""
    if "items" not in data:
        raise ModelSpecError("modeling-agent fixture must have a top-level 'items' list")
    return [_item_from_raw(raw) for raw in data["items"]]


def loads_modeling_items(text: str) -> list[CleaningItem]:
    """Parse modeling-agent JSON ``text`` into labeled :class:`CleaningItem`s."""
    data: Any = json.loads(text)
    return _items_from_data(data)


def load_modeling_items(path: str | Path) -> list[CleaningItem]:
    """Load a modeling-agent JSON fixture at ``path`` into labeled :class:`CleaningItem`s.

    The returned items feed directly into
    :func:`opop.bench.cleaning.verify_and_clean`: each ``labeled_optimum`` is the
    dataset's provided optimum, to be re-confirmed (or quarantined) by a solver.
    """
    return loads_modeling_items(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class ModelingDataset:
    """One cleaned modeling-agent dataset as a held-out registry family.

    Attributes:
        entry_name: Registry entry name / leakage_group (``modeling_<dataset>_cleaned``).
        dataset: Originating dataset tag (``nl4opt`` / ``optibench``).
        problem_type: Coarse class tag for the family (``MILP`` / ``MINLP``).
        filename: The committed clean-items fixture file name.
        item_ids: Globally unique ids of the clean (verified) items.
        sha256: Hex SHA-256 of the committed fixture file (content lock).
    """

    entry_name: str
    dataset: str
    problem_type: str
    filename: str
    item_ids: tuple[str, ...]
    sha256: str


#: The committed *clean* modeling-agent datasets (planted-wrong fixture excluded).
MODELING_CATALOG: tuple[ModelingDataset, ...] = (
    ModelingDataset(
        "modeling_nl4opt_cleaned",
        "nl4opt",
        "MILP",
        "nl4opt.json",
        ("nl4opt/production", "nl4opt/diet"),
        "b6aa74a56a261736a3ba9a9305316c3c4f4a780eac16fbc72cd0df58380291b3",
    ),
    ModelingDataset(
        "modeling_optibench_cleaned",
        "optibench",
        "MINLP",
        "optibench.json",
        ("optibench/portfolio_miqp", "optibench/design_minlp"),
        "d387bae21894c3291baa405d37bde7bb509cf03bb5809ec7f94f7468a7e1cd52",
    ),
)


def build_entries() -> list[BenchmarkEntry]:
    """Return one held-out registry entry per cleaned modeling-agent dataset.

    Each dataset is a ``phase=6`` / ``thesis=T3`` family whose solver-verified
    (clean) items sit in the immutable ``test`` split as its own ``leakage_group``,
    with a ``sha256:`` checksum over the committed fixture file. The planted-wrong
    fixture is deliberately NOT referenced here.
    """
    return [
        BenchmarkEntry(
            name=dataset.entry_name,
            problem_type=dataset.problem_type,
            source="modeling_agent",
            split={"test": dataset.item_ids},
            license=_REGISTRY_LICENSE,
            instance_count=len(dataset.item_ids),
            time_limit_sec=_REGISTRY_TIME_LIMIT_SEC,
            baseline_set=_REGISTRY_BASELINE,
            leakage_group=dataset.entry_name,
            checksum="sha256:" + dataset.sha256,
            phase=_REGISTRY_PHASE,
            thesis=_REGISTRY_THESIS,
        )
        for dataset in MODELING_CATALOG
    ]
