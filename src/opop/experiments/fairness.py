"""Fairness guards for OPOP baseline experiments (plan task 36).

A baseline comparison is only scientifically valid when every method runs under
an *identical* resource budget — the same number of search ``trials``, the same
per-solve ``time_limit_sec``, and the same set of ``seeds`` — and when no
held-out split is ever used to tune solver hyper-parameters.  This module is the
small, pure guard layer the baseline harness (:mod:`opop.experiments.baselines`)
calls before it runs anything:

* :class:`FairnessError` — raised on ANY budget / seed / split violation.
* :class:`BudgetSpec` — the immutable ``(trials, time_limit_sec, seeds)`` a
  method is permitted to consume, with :meth:`BudgetSpec.from_config`.
* :func:`check_budget_fairness` — assert a candidate budget equals the reference
  (the opop run); raise :class:`FairnessError` otherwise.
* :func:`assert_tunable_split` — refuse to tune on ``test`` / ``ood_test``.

No solver, no I/O, no optional dependencies — just the equality contract, so it
is always importable and always cheap to call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opop.config import RunConfig

__all__ = [
    "HELD_OUT_SPLITS",
    "TUNABLE_SPLITS",
    "BudgetSpec",
    "FairnessError",
    "assert_tunable_split",
    "check_budget_fairness",
]

#: Splits a baseline is allowed to TUNE on (free / non-held-out).
TUNABLE_SPLITS: frozenset[str] = frozenset({"dev", "validation"})

#: Held-out splits that must NEVER be used for hyper-parameter tuning (leakage).
HELD_OUT_SPLITS: frozenset[str] = frozenset({"test", "ood_test"})


class FairnessError(RuntimeError):
    """Raised when two methods are not run under an identical, fair budget.

    Covers a mismatch in ``trials`` / ``time_limit_sec`` / ``seeds`` between a
    baseline and the opop run it is compared against, and any attempt to tune a
    baseline on a held-out (``test`` / ``ood_test``) split.
    """


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """The fair resource budget a single method is permitted to consume.

    Two methods are budget-comparable iff their :class:`BudgetSpec` are equal:
    same search-``trials`` count, same per-solve ``time_limit_sec``, and the same
    ordered ``seeds``.

    Attributes:
        trials: Search-trial budget (BO iterations / tuning trials / 1 for default).
        time_limit_sec: Hard per-solve wall-clock ceiling in seconds.
        seeds: The exact seeds every method must run, in order.
    """

    trials: int
    time_limit_sec: float
    seeds: tuple[int, ...]

    @classmethod
    def from_config(cls, config: RunConfig) -> BudgetSpec:
        """Extract the budget spec from a :class:`~opop.config.RunConfig`."""
        return cls(
            trials=int(config.budget.trials),
            time_limit_sec=float(config.budget.time_limit_sec),
            seeds=tuple(int(s) for s in config.seeds),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable mapping of the budget."""
        return {
            "trials": self.trials,
            "time_limit_sec": self.time_limit_sec,
            "seeds": list(self.seeds),
        }


def check_budget_fairness(
    reference: BudgetSpec,
    candidate: BudgetSpec,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> None:
    """Assert ``candidate`` matches the ``reference`` budget exactly.

    The ``trials`` and ``seeds`` must be identical; ``time_limit_sec`` must match
    within ``rel_tol`` / ``abs_tol`` (float-safe equality).  All violations are
    collected and reported together.

    Args:
        reference: The authoritative budget (the opop run's).
        candidate: The baseline budget to validate.
        rel_tol: Relative tolerance for the time-limit comparison.
        abs_tol: Absolute tolerance for the time-limit comparison.

    Raises:
        FairnessError: If any of ``trials`` / ``time_limit_sec`` / ``seeds`` differ.
    """
    problems: list[str] = []
    if reference.trials != candidate.trials:
        problems.append(
            f"trials differ (opop={reference.trials} != baseline={candidate.trials})"
        )
    if not math.isclose(
        reference.time_limit_sec, candidate.time_limit_sec, rel_tol=rel_tol, abs_tol=abs_tol
    ):
        problems.append(
            "time_limit_sec differ "
            + f"(opop={reference.time_limit_sec} != baseline={candidate.time_limit_sec})"
        )
    if tuple(reference.seeds) != tuple(candidate.seeds):
        problems.append(
            f"seeds differ (opop={list(reference.seeds)} != baseline={list(candidate.seeds)})"
        )
    if problems:
        raise FairnessError(
            "baseline budget is not equal to the opop run's budget: " + "; ".join(problems)
        )


def assert_tunable_split(split: str) -> None:
    """Refuse to tune solver parameters on a held-out split.

    Tuning on ``test`` / ``ood_test`` leaks held-out information into the search,
    which would invalidate the comparison.  Only :data:`TUNABLE_SPLITS` are
    permitted.

    Args:
        split: The dataset split a tuning baseline is about to run on.

    Raises:
        FairnessError: If ``split`` is a held-out split.
    """
    if split in HELD_OUT_SPLITS:
        raise FairnessError(
            f"refusing to tune solver parameters on held-out split {split!r}; "
            + f"tuning is permitted only on {sorted(TUNABLE_SPLITS)} (no test/ood leakage)"
        )
