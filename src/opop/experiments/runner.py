"""Pluggable job runners for the OPOP experiment matrix (plan task 39).

A :class:`Runner` turns a sweep (an :class:`~opop.experiments.matrix.ExperimentMatrix`
or a sequence of :class:`~opop.experiments.matrix.MatrixCell`) into :class:`Job`
records. Every runner can :meth:`Runner.plan_jobs` (a side-effect-free listing of
``(cell, command)`` jobs — the "dry-run listing") and :meth:`Runner.submit_jobs`
(the act):

* :class:`LocalRunner` — runs each cell in-process via a ``work_fn`` callable,
  sequentially, **resume-safe** (skips any cell whose ``cell_done.json`` marker
  records ``"status": "ok"``) and error-tolerant (a raising cell is recorded as
  ``"error"`` and the sweep continues).
* :class:`DryRunRunner` — :meth:`submit_jobs` simply returns the listing; it never
  executes a ``work_fn`` or writes a marker.
* :class:`SlurmRunner` / :class:`QzRunner` — cluster stubs: they satisfy the
  :class:`Runner` Protocol and support dry-run listing via :meth:`plan_jobs`, but
  :meth:`submit_jobs` raises :class:`NotImplementedError` (actual scheduler
  submission is wired in chunk 2).

:func:`runner_for` is the ``kind`` -> runner factory (``"local"`` / ``"slurm"`` /
``"qz"`` / ``"dry-run"``). This module executes NO solver itself — the per-cell
work is entirely the caller's ``work_fn``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, final, runtime_checkable

from opop.experiments.matrix import (
    ExperimentMatrix,
    MatrixCell,
    as_cells,
    is_cell_done,
    write_cell_marker,
)

__all__ = [
    "SUPPORTED_RUNNERS",
    "DryRunRunner",
    "Job",
    "LocalRunner",
    "QzRunner",
    "Runner",
    "SlurmRunner",
    "WorkFn",
    "runner_for",
]

#: A per-cell work function: takes a cell, returns a JSON-friendly result row.
WorkFn = Callable[[MatrixCell], dict[str, object]]

#: Recognised runner kinds (the :func:`runner_for` factory keys).
SUPPORTED_RUNNERS: tuple[str, ...] = ("local", "slurm", "qz", "dry-run")

#: Job statuses.
STATUS_PLANNED = "planned"
STATUS_DRY_RUN = "dry-run"
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Job:
    """One runner job: a cell paired with its command/tag and outcome status.

    Attributes:
        cell: The :class:`~opop.experiments.matrix.MatrixCell` this job runs.
        command: The command or tag that would run / did run the cell.
        runner_kind: The runner that produced the job (``local`` / ``slurm`` / …).
        status: ``planned`` / ``dry-run`` / ``ok`` / ``error`` / ``skipped``.
        detail: Optional human-readable note (e.g. an error message).
    """

    cell: MatrixCell
    command: str
    runner_kind: str
    status: str = STATUS_PLANNED
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping of this job."""
        return {
            "cell": self.cell.to_dict(),
            "command": self.command,
            "runner_kind": self.runner_kind,
            "status": self.status,
            "detail": self.detail,
        }


def _command_for(kind: str, cell: MatrixCell) -> str:
    """Return the command/tag a ``kind`` runner would use for ``cell``.

    A descriptive tag keyed on the cell slug; chunk 2 materialises the concrete
    submission command (``sbatch`` / ``qz train``) for the cluster runners.
    """
    return f"{kind}:{cell.slug}"


@runtime_checkable
class Runner(Protocol):
    """Contract every job runner satisfies.

    ``plan_jobs`` is a pure listing (no execution, no I/O); ``submit_jobs`` is the
    act — executing (local), listing (dry-run), or submitting (cluster). Cells may
    be passed as an :class:`~opop.experiments.matrix.ExperimentMatrix` or an
    already-expanded sequence.
    """

    kind: str

    def plan_jobs(self, cells: ExperimentMatrix | Sequence[MatrixCell]) -> list[Job]:
        """Return the ``(cell, command)`` job listing without side effects."""
        ...

    def submit_jobs(
        self,
        cells: ExperimentMatrix | Sequence[MatrixCell],
        *,
        work_fn: WorkFn | None = None,
        out_dir: str | Path | None = None,
    ) -> list[Job]:
        """Act on the cells (execute / list / submit) and return the jobs."""
        ...


@final
class LocalRunner:
    """Run each cell in-process, sequentially, resume-safe and error-tolerant."""

    kind: str = "local"

    def plan_jobs(self, cells: ExperimentMatrix | Sequence[MatrixCell]) -> list[Job]:
        """List the local jobs without executing any ``work_fn``."""
        return [
            Job(cell, _command_for(self.kind, cell), self.kind, status=STATUS_PLANNED)
            for cell in as_cells(cells)
        ]

    def submit_jobs(
        self,
        cells: ExperimentMatrix | Sequence[MatrixCell],
        *,
        work_fn: WorkFn | None = None,
        out_dir: str | Path | None = None,
    ) -> list[Job]:
        """Run every cell via ``work_fn`` sequentially; skip completed cells.

        A cell with an ``ok`` ``cell_done.json`` marker (under ``out_dir``) is
        skipped. A cell whose ``work_fn`` raises is recorded as ``error`` (marker
        written when ``out_dir`` is given) and the sweep continues. Requires
        ``work_fn`` (the local runner executes the work itself).

        Raises:
            ValueError: If ``work_fn`` is ``None``.
        """
        if work_fn is None:
            raise ValueError("LocalRunner.submit_jobs requires a work_fn to execute each cell")

        jobs: list[Job] = []
        for cell in as_cells(cells):
            command = _command_for(self.kind, cell)
            if out_dir is not None and is_cell_done(cell, out_dir):
                jobs.append(
                    Job(cell, command, self.kind, status=STATUS_SKIPPED, detail="cell_done.json ok")
                )
                continue
            try:
                result = dict(work_fn(cell))
            except Exception as exc:  # noqa: BLE001 (record + continue; one bad cell never aborts the sweep)
                detail = f"{type(exc).__name__}: {exc}"
                if out_dir is not None:
                    write_cell_marker(cell, out_dir, status=STATUS_ERROR, error=detail, runner=self.kind)
                jobs.append(Job(cell, command, self.kind, status=STATUS_ERROR, detail=detail))
                continue
            if out_dir is not None:
                write_cell_marker(cell, out_dir, status=STATUS_OK, result=result, runner=self.kind)
            jobs.append(Job(cell, command, self.kind, status=STATUS_OK))
        return jobs


@final
class DryRunRunner:
    """List the sweep's jobs without executing or submitting anything."""

    kind: str = "dry-run"

    def plan_jobs(self, cells: ExperimentMatrix | Sequence[MatrixCell]) -> list[Job]:
        """Return the ``(cell, command)`` listing for every cell."""
        return [
            Job(cell, _command_for(self.kind, cell), self.kind, status=STATUS_DRY_RUN)
            for cell in as_cells(cells)
        ]

    def submit_jobs(
        self,
        cells: ExperimentMatrix | Sequence[MatrixCell],
        *,
        work_fn: WorkFn | None = None,
        out_dir: str | Path | None = None,
    ) -> list[Job]:
        """Return the dry-run listing; never executes ``work_fn`` or writes I/O."""
        del work_fn, out_dir
        return self.plan_jobs(cells)


class _ClusterRunner:
    """Shared base for cluster stubs: dry-run listing works; submission raises."""

    kind: str = "cluster"

    def plan_jobs(self, cells: ExperimentMatrix | Sequence[MatrixCell]) -> list[Job]:
        """List the submission jobs without contacting a scheduler."""
        return [
            Job(cell, _command_for(self.kind, cell), self.kind, status=STATUS_PLANNED)
            for cell in as_cells(cells)
        ]

    def submit_jobs(
        self,
        cells: ExperimentMatrix | Sequence[MatrixCell],
        *,
        work_fn: WorkFn | None = None,
        out_dir: str | Path | None = None,
    ) -> list[Job]:
        """Reject actual submission (not wired in this chunk)."""
        del cells, work_fn, out_dir
        raise NotImplementedError(
            f"{type(self).__name__} submission is not implemented in this chunk; "
            + "use plan_jobs() for the dry-run listing or runner_for('dry-run')"
        )


@final
class SlurmRunner(_ClusterRunner):
    """Slurm cluster stub (dry-run listing only; submission raises)."""

    kind: str = "slurm"


@final
class QzRunner(_ClusterRunner):
    """Inspire/SII ``qz`` cluster stub (dry-run listing only; submission raises)."""

    kind: str = "qz"


def runner_for(kind: str) -> Runner:
    """Return a fresh :class:`Runner` for ``kind``.

    Args:
        kind: One of :data:`SUPPORTED_RUNNERS` (``local`` / ``slurm`` / ``qz`` /
            ``dry-run``).

    Raises:
        ValueError: If ``kind`` is not a recognised runner kind.
    """
    if kind == "local":
        return LocalRunner()
    if kind == "dry-run":
        return DryRunRunner()
    if kind == "slurm":
        return SlurmRunner()
    if kind == "qz":
        return QzRunner()
    raise ValueError(f"unknown runner kind {kind!r}; supported: {list(SUPPORTED_RUNNERS)}")
