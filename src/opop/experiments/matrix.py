"""Experiment-matrix expansion + resume logic for the OPOP sweep (plan task 39).

The final research campaign runs a Cartesian sweep over five factors —
``instance`` x ``method`` x ``ablation`` x ``seed`` x ``time_limit`` — for every
combination of benchmark instance, comparison method, ablation row, random seed,
and per-solve budget. This module is the pure (solver-free, I/O-light) model of
that sweep:

* :class:`AblationRow` — the canonical ablation vocabulary (the five named rows
  ``scip_default`` … ``full_opop`` plus the staged ``S0``–``S4`` ladder), used as
  the ``ablation`` factor levels.
* :class:`MatrixCell` — one fully-specified sweep cell (the unit of work), with a
  filesystem-safe :attr:`MatrixCell.slug` identity and an optional ``payload`` for
  runtime context.
* :class:`ExperimentMatrix` / :func:`expand_matrix` — the factor lists and their
  Cartesian expansion into a deterministic tuple of cells.
* :func:`is_cell_done` / :func:`write_cell_marker` / :class:`MatrixStatus` — the
  resume layer: a cell is *done* once its ``cell_done.json`` marker records
  ``"status": "ok"``, so a re-run skips completed cells.

The driver / CLI / runner integration lives in :mod:`opop.experiments.runner`
(this module never executes a solver).
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "ALL_ABLATIONS",
    "CANONICAL_ABLATIONS",
    "CELL_MARKER_NAME",
    "STAGED_ABLATIONS",
    "AblationRow",
    "ExperimentMatrix",
    "MatrixCell",
    "MatrixStatus",
    "as_cells",
    "cell_marker_path",
    "cell_out_dir",
    "expand_matrix",
    "is_cell_done",
    "write_cell_marker",
]

#: Per-cell completion marker file name (lives in the cell's output directory).
CELL_MARKER_NAME = "cell_done.json"

_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


class AblationRow(StrEnum):
    """The canonical ablation rows used as the ``ablation`` matrix factor.

    The five named rows isolate each capability layer (default solver → params →
    analyzer cuts → params+cuts → full opop); the ``S0``–``S4`` members mirror the
    proposer's staged search ladder (:class:`opop.proposer.stages.Stage`) for the
    credit-assignment ablation. Members are plain strings (``StrEnum``) so they
    drop straight into a :class:`MatrixCell` and serialise as their value.
    """

    SCIP_DEFAULT = "scip_default"
    PARAMS_ONLY = "params_only"
    ANALYZER_CUTS_ONLY = "analyzer_cuts_only"
    PARAMS_PLUS_CUTS = "params_plus_cuts"
    FULL_OPOP = "full_opop"
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


#: The five named ablation rows (capability-isolation comparison).
CANONICAL_ABLATIONS: tuple[AblationRow, ...] = (
    AblationRow.SCIP_DEFAULT,
    AblationRow.PARAMS_ONLY,
    AblationRow.ANALYZER_CUTS_ONLY,
    AblationRow.PARAMS_PLUS_CUTS,
    AblationRow.FULL_OPOP,
)

#: The staged search-ladder ablation rows (S0–S4).
STAGED_ABLATIONS: tuple[AblationRow, ...] = (
    AblationRow.S0,
    AblationRow.S1,
    AblationRow.S2,
    AblationRow.S3,
    AblationRow.S4,
)

#: Every recognised ablation row (named + staged).
ALL_ABLATIONS: tuple[AblationRow, ...] = CANONICAL_ABLATIONS + STAGED_ABLATIONS


def _slugify(text: str) -> str:
    """Collapse any non ``[A-Za-z0-9._-]`` run to a single underscore."""
    return _SAFE_SLUG.sub("_", text).strip("_") or "_"


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One fully-specified sweep cell (the unit of work).

    Attributes:
        instance_id: Benchmark instance id (registry namespace, may contain ``/``).
        method: Comparison method tag (e.g. ``opop`` / ``scip-default``).
        ablation: Ablation row (an :class:`AblationRow` value or a raw string).
        seed: Random seed.
        time_limit: Per-solve wall-clock budget in seconds.
        payload: Optional runtime context (never part of the cell's identity).
    """

    instance_id: str
    method: str
    ablation: str
    seed: int
    time_limit: float
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Filesystem-safe identity over the five factors (excludes ``payload``)."""
        return "__".join(
            (
                _slugify(self.instance_id),
                _slugify(self.method),
                _slugify(str(self.ablation)),
                f"seed{int(self.seed)}",
                f"tl{float(self.time_limit):g}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of this cell."""
        return {
            "instance_id": self.instance_id,
            "method": self.method,
            "ablation": str(self.ablation),
            "seed": int(self.seed),
            "time_limit": float(self.time_limit),
            "payload": dict(self.payload),
            "slug": self.slug,
        }


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    """The five factor lists whose Cartesian product defines the sweep.

    Sequences are normalised to tuples on construction so the matrix is immutable
    and its :meth:`expand` order is deterministic (instances outermost,
    time-limits innermost).

    Attributes:
        instances: Benchmark instance ids.
        methods: Comparison method tags.
        ablations: Ablation rows (:class:`AblationRow` values or raw strings).
        seeds: Random seeds.
        time_limits: Per-solve budgets (seconds).
    """

    instances: Sequence[str]
    methods: Sequence[str]
    ablations: Sequence[str]
    seeds: Sequence[int]
    time_limits: Sequence[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "ablations", tuple(str(a) for a in self.ablations))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        object.__setattr__(self, "time_limits", tuple(float(t) for t in self.time_limits))

    @property
    def n_cells(self) -> int:
        """The number of cells the matrix expands to."""
        return (
            len(self.instances)
            * len(self.methods)
            * len(self.ablations)
            * len(self.seeds)
            * len(self.time_limits)
        )

    def expand(self) -> tuple[MatrixCell, ...]:
        """Return the Cartesian product of the factors as a tuple of cells."""
        return tuple(
            MatrixCell(
                instance_id=instance,
                method=method,
                ablation=ablation,
                seed=int(seed),
                time_limit=float(time_limit),
            )
            for instance, method, ablation, seed, time_limit in itertools.product(
                self.instances, self.methods, self.ablations, self.seeds, self.time_limits
            )
        )


def expand_matrix(matrix: ExperimentMatrix) -> tuple[MatrixCell, ...]:
    """Expand ``matrix`` into its deterministic tuple of :class:`MatrixCell`."""
    return matrix.expand()


def as_cells(cells: ExperimentMatrix | Sequence[MatrixCell]) -> tuple[MatrixCell, ...]:
    """Normalise a matrix OR a cell sequence to a tuple of :class:`MatrixCell`."""
    if isinstance(cells, ExperimentMatrix):
        return cells.expand()
    return tuple(cells)


# ---------------------------------------------------------------------------
# Resume layer
# ---------------------------------------------------------------------------
def cell_out_dir(cell: MatrixCell, out_dir: str | Path) -> Path:
    """Return the per-cell output directory ``<out_dir>/cells/<slug>``."""
    return Path(out_dir) / "cells" / cell.slug


def cell_marker_path(cell: MatrixCell, out_dir: str | Path) -> Path:
    """Return the per-cell completion marker path (``cell_done.json``)."""
    return cell_out_dir(cell, out_dir) / CELL_MARKER_NAME


def is_cell_done(cell: MatrixCell, out_dir: str | Path) -> bool:
    """Return ``True`` iff ``cell`` has a marker recording ``"status": "ok"``.

    A missing, unreadable, or non-``ok`` marker counts as not-done (so a partial
    or failed cell is re-run on resume).
    """
    marker = cell_marker_path(cell, out_dir)
    if not marker.is_file():
        return False
    try:
        data: Any = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("status") == "ok"


def write_cell_marker(
    cell: MatrixCell,
    out_dir: str | Path,
    *,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    runner: str = "",
) -> Path:
    """Write a cell's ``cell_done.json`` marker; return its path.

    ``status == "ok"`` marks the cell complete (skipped on resume); any other
    status (e.g. ``"error"``) is treated as not-done by :func:`is_cell_done`.
    """
    marker = cell_marker_path(cell, out_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"status": status, "runner": runner, "cell": cell.to_dict()}
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


@dataclass(frozen=True, slots=True)
class MatrixStatus:
    """Resume snapshot of a matrix against an output directory.

    Attributes:
        done: Cells whose marker records ``"status": "ok"``.
        pending: Cells that still need to run.
    """

    done: tuple[MatrixCell, ...]
    pending: tuple[MatrixCell, ...]

    @property
    def total(self) -> int:
        """Total cell count (done + pending)."""
        return len(self.done) + len(self.pending)

    @property
    def n_done(self) -> int:
        """Number of completed cells."""
        return len(self.done)

    @property
    def n_pending(self) -> int:
        """Number of cells still to run."""
        return len(self.pending)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary (slugs only, for compactness)."""
        return {
            "total": self.total,
            "n_done": self.n_done,
            "n_pending": self.n_pending,
            "done": [c.slug for c in self.done],
            "pending": [c.slug for c in self.pending],
        }

    @classmethod
    def scan(
        cls, cells: ExperimentMatrix | Sequence[MatrixCell], out_dir: str | Path
    ) -> MatrixStatus:
        """Classify every cell as done / pending against ``out_dir``."""
        done: list[MatrixCell] = []
        pending: list[MatrixCell] = []
        for cell in as_cells(cells):
            (done if is_cell_done(cell, out_dir) else pending).append(cell)
        return cls(done=tuple(done), pending=tuple(pending))
