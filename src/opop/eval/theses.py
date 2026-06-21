"""``python -m opop.eval.theses`` -> the falsifiable T1-T4 thesis evaluator (task 40).

Reads a consolidated ``results.parquet`` (and optional ``events.jsonl`` for the
T2 solver-invocation counts) produced by the task-39 matrix driver and emits a
``thesis_report.json`` with a per-thesis verdict for each of the four locked,
falsifiable theses:

``T1`` — anytime / cross-distribution superiority
    ``opop`` must WIN vs ``scip-default`` **AND** vs ``opop-params-only`` on
    ``primal_integral`` (significant + >= 10% reduction) at an equal end-to-end
    budget on held-out instances.
``T2`` — sample / compute efficiency
    ``opop`` must use **>= 30% fewer** full-solve evaluations than
    ``scip-default`` to reach baseline-best quality (overhead included). Solve
    counts come from ``events.jsonl`` solve rows per cell, falling back to an
    ``n_solves`` column on the result records when no events exist.
``T3`` — generality
    ``opop`` must WIN vs ``scip-default`` on ``primal_integral`` on **each**
    problem type present in the results (MILP, QUBO, MIQP, MIQCP, ...), mapping
    ``instance_id`` -> ``problem_type`` through the benchmark registry.
``T4`` — method novelty
    ``opop`` must WIN vs ``opop-params-only`` **AND** vs ``modeling-agent`` on
    ``primal_integral`` (significant + >= 10% reduction), showing the
    analyzer-certified deltas add value beyond params-only-BO and the
    modeling-agent-only baseline.

A **WIN** is exactly the locked Win Definition of
:func:`opop.experiments.compare.compare`: a comparison is a win iff it is both
statistically significant (paired Wilcoxon signed-rank, ``alpha = 0.05``) AND it
clears the per-metric min-effect threshold (>= 10% primal-integral reduction).
T1/T3/T4 reuse :func:`~opop.experiments.compare.compare` directly; T2 uses a
small paired-Wilcoxon helper on the solve counts (which ``compare`` cannot
extract from the records).

The evaluator NEVER suppresses a negative result: every thesis appears in the
report with its actual ``verdict`` and the full underlying comparison(s). A
one-shot guard refuses to touch the held-out ``test`` / ``ood_test`` splits
unless ``one_shot_final=True`` is passed explicitly.

``PYTHONPATH=src python -m opop.eval.theses --results runs/final_eval --out
thesis_report.json`` writes the report (``--results`` may be a directory holding
``results.parquet`` + ``events.jsonl`` or a direct results file).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, final

import numpy as np
from scipy.stats import wilcoxon

from opop.bench.registry import BenchmarkRegistry, RegistryError
from opop.bench.sources.phase1_set import REGISTRY_PATH
from opop.experiments.compare import (
    ComparisonReport,
    build_min_effect,
    compare,
    load_results,
)

__all__ = [
    "PRIMAL_INTEGRAL",
    "ThesisError",
    "ThesisEvaluator",
    "ThesisReport",
    "ThesisVerdict",
    "build_problem_type_map",
    "evaluate_theses",
    "main",
]

# --------------------------------------------------------------------------- #
# Locked constants
# --------------------------------------------------------------------------- #

#: The single metric every thesis is decided on (lower is better).
PRIMAL_INTEGRAL = "primal_integral"

#: Min-effect threshold for a primal-integral WIN (>= 10% reduction).
PI_MIN_EFFECT = 0.10

#: Min-effect threshold for the T2 solve-count reduction (>= 30% fewer solves).
N_SOLVES_MIN_EFFECT = 0.30

#: Significance level for every paired Wilcoxon signed-rank test.
ALPHA = 0.05

#: Held-out splits that require an explicit ``one_shot_final=True`` to evaluate.
HELD_OUT_SPLITS = frozenset({"test", "ood_test"})

#: Canonical method tags emitted by the task-39 matrix driver / baselines.
OPOP = "opop"
SCIP_DEFAULT = "scip-default"
PARAMS_ONLY = "opop-params-only"
MODELING_AGENT = "modeling-agent"

#: ``event_type`` values in ``events.jsonl`` that denote a full-solve invocation.
SOLVE_EVENT_TYPES = frozenset({"solve"})

#: Guard floor for the relative-improvement denominator (mirrors ``compare``).
_REL_EPS = 1e-12

#: Human-readable thesis claims (verbatim in the report).
_CLAIMS: dict[str, str] = {
    "T1": (
        "opop beats scip-default AND opop-params-only on primal_integral "
        "(significant + >=10% reduction) at equal end-to-end budget on held-out "
        "instances (anytime / cross-distribution superiority)."
    ),
    "T2": (
        "opop reaches baseline-best quality using >=30% fewer full-solve "
        "evaluations than scip-default, overhead included (sample/compute "
        "efficiency)."
    ),
    "T3": (
        "opop beats scip-default on primal_integral on EVERY problem type "
        "present in the results (generality)."
    ),
    "T4": (
        "opop beats opop-params-only AND modeling-agent on primal_integral "
        "(significant + >=10% reduction): analyzer-certified deltas add value "
        "beyond params-only-BO and modeling-agent-only (method novelty)."
    ),
}


class ThesisError(RuntimeError):
    """Raised when a thesis evaluation is refused (e.g. the one-shot guard)."""


# --------------------------------------------------------------------------- #
# Report dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ThesisVerdict:
    """Machine-readable verdict for one thesis.

    Attributes:
        claim: The verbatim thesis statement being tested.
        metric: The metric the verdict is decided on (``primal_integral`` /
            ``n_solves``).
        baseline: The baseline method (T2/T3) or the list of baselines that must
            ALL be beaten (T1/T4).
        significant: ``True`` iff every required comparison is statistically
            significant (paired Wilcoxon, ``alpha = 0.05``).
        effect: The binding (minimum) relative improvement across the required
            comparisons; for T2 the median solve-count reduction.
        clears_threshold: ``True`` iff every required comparison clears its
            min-effect threshold.
        verdict: The falsifiable outcome — ``True`` iff EVERY required
            comparison is a WIN (significant AND clears the threshold).
        details: The full underlying comparison(s) and any diagnostic notes
            (never elided, so negative results stay visible).
    """

    claim: str
    metric: str
    baseline: str | list[str]
    significant: bool
    effect: float
    clears_threshold: bool
    verdict: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dict of all fields."""
        baseline: Any = list(self.baseline) if isinstance(self.baseline, list) else self.baseline
        return {
            "claim": self.claim,
            "metric": self.metric,
            "baseline": baseline,
            "significant": self.significant,
            "effect": self.effect,
            "clears_threshold": self.clears_threshold,
            "verdict": self.verdict,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ThesisReport:
    """The full ``thesis_report.json`` payload: T1-T4 verdicts + provenance."""

    verdicts: Mapping[str, ThesisVerdict]
    split: str
    one_shot_final: bool
    n_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Flatten to ``{"T1": ..., ..., "T4": ..., "meta": {...}}``."""
        payload: dict[str, Any] = {name: v.to_dict() for name, v in self.verdicts.items()}
        payload["meta"] = {
            "split": self.split,
            "one_shot_final": self.one_shot_final,
            "n_records": int(self.n_records),
            "theses": sorted(self.verdicts),
            "all_pass": all(v.verdict for v in self.verdicts.values()),
        }
        return payload

    def to_json(self) -> str:
        """Deterministic JSON: key-sorted, ``allow_nan=False`` (no trailing newline)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)

    def write(self, path: str | Path) -> Path:
        """Write ``thesis_report.json`` (key-sorted, ``allow_nan=False``, trailing newline)."""
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p


# --------------------------------------------------------------------------- #
# Record / registry helpers
# --------------------------------------------------------------------------- #


def _as_records(results: Any) -> list[dict[str, Any]]:
    """Coerce supported result containers to a ``list[dict]`` (cf. ``compare``)."""
    if isinstance(results, Mapping):
        for key in ("records", "results"):
            if key in results:
                inner = cast("Sequence[Mapping[str, Any]]", results[key])
                return [dict(r) for r in inner]
        raise ValueError("dict results must carry a 'records' or 'results' list")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        rows = cast("Sequence[Mapping[str, Any]]", results)
        return [dict(r) for r in rows]
    to_dict = getattr(results, "to_dict", None)
    if callable(to_dict):
        rows = cast("Sequence[Mapping[str, Any]]", to_dict("records"))
        return [dict(r) for r in rows]
    raise TypeError(f"unsupported results container: {type(results)!r}")


def _finite_pi_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only records whose ``primal_integral`` is present and finite.

    The driver writes ``primal_integral = NaN`` when a method finds no incumbent;
    such rows cannot be paired meaningfully and would break ``allow_nan=False``
    serialisation, so they are dropped before the comparison (the pairing in
    ``compare`` already discards any half-pair that loses its partner).
    """
    finite: list[dict[str, Any]] = []
    for rec in records:
        raw = rec.get(PRIMAL_INTEGRAL)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(dict(rec))
    return finite


def build_problem_type_map(registry: BenchmarkRegistry) -> dict[str, str]:
    """Map every registered ``instance_id`` to its ``problem_type``.

    Iterates every benchmark entry and every split it declares, so the map
    covers free *and* held-out instances (the caller decides which split the
    records belong to).
    """
    mapping: dict[str, str] = {}
    for entry in registry.entries:
        for ids in entry.split.values():
            for inst in ids:
                mapping[str(inst)] = entry.problem_type
    return mapping


# --------------------------------------------------------------------------- #
# T2 solve-count helpers (events.jsonl + n_solves fallback)
# --------------------------------------------------------------------------- #


def _load_events(events: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Load events from an ``events.jsonl`` path or an in-memory iterable."""
    if isinstance(events, (str, Path)):
        rows: list[dict[str, Any]] = []
        with Path(events).open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(dict(json.loads(stripped)))
        return rows
    return [dict(r) for r in events]


def _is_solve_event(row: Mapping[str, Any]) -> bool:
    """Decide whether one journal row represents a full-solve invocation.

    Prefers an explicit ``event_type`` (``"solve"``); otherwise falls back to the
    closed-loop journal schema, where a solved/passed proposal carries a
    ``verify_status == "pass"`` and a non-null ``score`` / ``trace_summary``.
    """
    event_type = row.get("event_type")
    if event_type is not None:
        return event_type in SOLVE_EVENT_TYPES
    if row.get("verify_status") == "pass":
        return True
    return row.get("score") is not None or row.get("trace_summary") is not None


def _solve_counts_from_events(
    events: str | Path | Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], int]:
    """Count solve rows grouped by ``(instance_id, seed, method)``.

    Rows that do not carry a ``method`` (e.g. the raw closed-loop journal, which
    is not method-tagged) cannot be attributed and are skipped, so the caller
    transparently falls back to the ``n_solves`` column.
    """
    counts: dict[tuple[str, str, str], int] = {}
    for row in _load_events(events):
        if not _is_solve_event(row):
            continue
        method = row.get("method")
        if method is None:
            continue
        key = (str(row.get("instance_id", "")), str(row.get("seed", "")), str(method))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _solve_counts_from_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], int]:
    """Read per-cell solve counts from an ``n_solves`` column when present."""
    counts: dict[tuple[str, str, str], int] = {}
    for rec in records:
        raw = rec.get("n_solves")
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        key = (
            str(rec.get("instance_id", "")),
            str(rec.get("seed", "")),
            str(rec.get("method", "")),
        )
        counts[key] = value
    return counts


def _paired_wilcoxon(baseline: Sequence[float], method: Sequence[float]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank ``(statistic, p_value)`` (cf. ``compare``).

    Returns ``(0.0, 1.0)`` when every paired difference is zero (scipy cannot run
    then) and silences scipy's ties/zeros approximation warning so a legitimate
    tie in solve counts is a data condition, not a failure.
    """
    b = np.asarray(baseline, dtype=np.float64)
    m = np.asarray(method, dtype=np.float64)
    if not bool(np.any((b - m) != 0.0)):
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = cast("tuple[float, float]", wilcoxon(b, m))
    return float(statistic), float(p_value)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """Compact summary of a value distribution (n / median / mean / min / max + raw)."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    return {
        "n": n,
        "median": float(np.median(arr)) if n else 0.0,
        "mean": float(arr.mean()) if n else 0.0,
        "min": float(arr.min()) if n else 0.0,
        "max": float(arr.max()) if n else 0.0,
        "values": [float(x) for x in arr.tolist()],
    }


# --------------------------------------------------------------------------- #
# ThesisEvaluator
# --------------------------------------------------------------------------- #


@final
class ThesisEvaluator:
    """Evaluate the four falsifiable theses (T1-T4) over consolidated results.

    Args:
        registry: A pre-loaded :class:`BenchmarkRegistry` for the T3
            ``instance_id`` -> ``problem_type`` map. Ignored if ``problem_types``
            is given.
        problem_types: An explicit ``instance_id`` -> ``problem_type`` map
            (overrides the registry; primarily for tests).
        registry_path: Path to ``registry.yaml`` loaded lazily for T3 when
            neither ``problem_types`` nor ``registry`` is supplied.
        alpha: Significance level for every paired Wilcoxon test.
    """

    def __init__(
        self,
        *,
        registry: BenchmarkRegistry | None = None,
        problem_types: Mapping[str, str] | None = None,
        registry_path: str | Path = REGISTRY_PATH,
        alpha: float = ALPHA,
    ) -> None:
        self._registry = registry
        self._problem_types = dict(problem_types) if problem_types is not None else None
        self._registry_path = registry_path
        self.alpha = float(alpha)

    # -- public API ---------------------------------------------------------
    def evaluate(
        self,
        records: Any,
        events: str | Path | Iterable[Mapping[str, Any]] | None = None,
        *,
        split: str,
        one_shot_final: bool = False,
    ) -> ThesisReport:
        """Evaluate T1-T4; return a :class:`ThesisReport`.

        Enforces the one-shot guard FIRST: a held-out (``test`` / ``ood_test``)
        split raises :class:`ThesisError` unless ``one_shot_final=True``.
        """
        if split in HELD_OUT_SPLITS and not one_shot_final:
            raise ThesisError(
                f"split {split!r} is held-out; pass one_shot_final=True to evaluate it "
                + "(refusing to touch the test/ood_test split without the one-shot flag)"
            )
        rows = _as_records(records)
        verdicts = {
            "T1": self._evaluate_t1(rows),
            "T2": self._evaluate_t2(rows, events),
            "T3": self._evaluate_t3(rows),
            "T4": self._evaluate_t4(rows),
        }
        return ThesisReport(
            verdicts=verdicts,
            split=split,
            one_shot_final=one_shot_final,
            n_records=len(rows),
        )

    # -- comparison primitive ----------------------------------------------
    def _safe_compare(
        self, records: Sequence[Mapping[str, Any]], *, baseline: str, method: str
    ) -> tuple[ComparisonReport | None, str | None]:
        """Run the locked primal-integral comparison; return ``(report, error)``.

        Drops non-finite primal integrals first and converts the
        ``no paired observations`` / ``unknown metric`` ``ValueError`` from
        :func:`compare` into a recorded error string (so a missing baseline is a
        reported non-win, never a crash).
        """
        finite = _finite_pi_records(records)
        try:
            report = compare(
                finite,
                baseline=baseline,
                method=method,
                metric=PRIMAL_INTEGRAL,
                alpha=self.alpha,
                min_effect=build_min_effect(PRIMAL_INTEGRAL, PI_MIN_EFFECT),
            )
        except ValueError as exc:
            return None, str(exc)
        return report, None

    def _win_over_all(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        baselines: Sequence[str],
        labels: Mapping[str, str] | None = None,
    ) -> tuple[bool, bool, bool, float, dict[str, Any]]:
        """AND a set of ``opop``-vs-baseline wins into one aggregate verdict.

        Returns ``(significant, clears_threshold, verdict, effect, comparisons)``
        where ``effect`` is the binding (minimum) relative improvement and a
        missing/empty comparison contributes a hard failure (effect ``0.0``).
        """
        comparisons: dict[str, Any] = {}
        errors: dict[str, str] = {}
        effects: list[float] = []
        significant = True
        clears = True
        verdict = True
        for baseline in baselines:
            label = labels.get(baseline, baseline) if labels else baseline
            report, error = self._safe_compare(records, baseline=baseline, method=OPOP)
            if report is None:
                errors[label] = error or "no comparison"
                significant = clears = verdict = False
                effects.append(0.0)
                continue
            comparisons[label] = report.to_dict()
            effects.append(report.relative_improvement)
            significant = significant and report.significant
            clears = clears and report.clears_min_effect
            verdict = verdict and report.is_win
        details: dict[str, Any] = {"comparisons": comparisons}
        if errors:
            details["errors"] = errors
        effect = min(effects) if effects else 0.0
        return significant, clears, verdict, effect, details

    # -- per-thesis ---------------------------------------------------------
    def _evaluate_t1(self, records: Sequence[Mapping[str, Any]]) -> ThesisVerdict:
        baselines = [SCIP_DEFAULT, PARAMS_ONLY]
        significant, clears, verdict, effect, details = self._win_over_all(
            records, baselines=baselines
        )
        return ThesisVerdict(
            claim=_CLAIMS["T1"],
            metric=PRIMAL_INTEGRAL,
            baseline=list(baselines),
            significant=significant,
            effect=effect,
            clears_threshold=clears,
            verdict=verdict,
            details=details,
        )

    def _evaluate_t4(self, records: Sequence[Mapping[str, Any]]) -> ThesisVerdict:
        baselines = [PARAMS_ONLY, MODELING_AGENT]
        significant, clears, verdict, effect, details = self._win_over_all(
            records, baselines=baselines
        )
        return ThesisVerdict(
            claim=_CLAIMS["T4"],
            metric=PRIMAL_INTEGRAL,
            baseline=list(baselines),
            significant=significant,
            effect=effect,
            clears_threshold=clears,
            verdict=verdict,
            details=details,
        )

    def _evaluate_t3(self, records: Sequence[Mapping[str, Any]]) -> ThesisVerdict:
        try:
            type_map = self._problem_type_map()
        except RegistryError as exc:
            return ThesisVerdict(
                claim=_CLAIMS["T3"],
                metric=PRIMAL_INTEGRAL,
                baseline=SCIP_DEFAULT,
                significant=False,
                effect=0.0,
                clears_threshold=False,
                verdict=False,
                details={"note": f"could not load problem-type registry: {exc}"},
            )

        by_type: dict[str, list[Mapping[str, Any]]] = {}
        unmapped: set[str] = set()
        for rec in records:
            inst = str(rec.get("instance_id", ""))
            problem_type = type_map.get(inst)
            if problem_type is None:
                unmapped.add(inst)
                continue
            by_type.setdefault(problem_type, []).append(rec)

        per_type: dict[str, Any] = {}
        errors: dict[str, str] = {}
        effects: list[float] = []
        significant = True
        clears = True
        verdict = bool(by_type)  # no recognised problem type -> cannot be a win
        for problem_type in sorted(by_type):
            report, error = self._safe_compare(
                by_type[problem_type], baseline=SCIP_DEFAULT, method=OPOP
            )
            if report is None:
                errors[problem_type] = error or "no comparison"
                significant = clears = verdict = False
                effects.append(0.0)
                continue
            per_type[problem_type] = report.to_dict()
            effects.append(report.relative_improvement)
            significant = significant and report.significant
            clears = clears and report.clears_min_effect
            verdict = verdict and report.is_win
        if not by_type:
            significant = clears = False

        details: dict[str, Any] = {
            "per_problem_type": per_type,
            "problem_types": sorted(by_type),
        }
        if unmapped:
            details["unmapped_instances"] = sorted(unmapped)
        if errors:
            details["errors"] = errors
        return ThesisVerdict(
            claim=_CLAIMS["T3"],
            metric=PRIMAL_INTEGRAL,
            baseline=SCIP_DEFAULT,
            significant=significant,
            effect=min(effects) if effects else 0.0,
            clears_threshold=clears,
            verdict=verdict,
            details=details,
        )

    def _evaluate_t2(
        self,
        records: Sequence[Mapping[str, Any]],
        events: str | Path | Iterable[Mapping[str, Any]] | None,
    ) -> ThesisVerdict:
        source: str | None = None
        counts: dict[tuple[str, str, str], int] = {}
        if events is not None:
            counts = _solve_counts_from_events(events)
            if counts:
                source = "events"
        if not counts:
            counts = _solve_counts_from_records(records)
            if counts:
                source = "n_solves"

        if not counts:
            return self._t2_no_data(
                "no solve-count data (no events.jsonl solve rows and no n_solves column)"
            )

        opop_by: dict[tuple[str, str], int] = {}
        scip_by: dict[tuple[str, str], int] = {}
        for (inst, seed, method), value in counts.items():
            if method == OPOP:
                opop_by[(inst, seed)] = value
            elif method == SCIP_DEFAULT:
                scip_by[(inst, seed)] = value

        keys = sorted(opop_by.keys() & scip_by.keys())
        if not keys:
            return self._t2_no_data(
                f"no paired (instance, seed) solve counts for {OPOP!r} vs {SCIP_DEFAULT!r}",
                source=source,
            )

        scip_vals = [float(scip_by[k]) for k in keys]
        opop_vals = [float(opop_by[k]) for k in keys]
        median_scip = float(np.median(scip_vals))
        median_opop = float(np.median(opop_vals))
        effect = (
            (median_scip - median_opop) / median_scip if abs(median_scip) > _REL_EPS else 0.0
        )
        statistic, p_value = _paired_wilcoxon(scip_vals, opop_vals)
        significant = bool(p_value < self.alpha)
        clears = bool(effect >= N_SOLVES_MIN_EFFECT)
        details = {
            "source": source,
            "baseline": SCIP_DEFAULT,
            "method": OPOP,
            "n_pairs": len(keys),
            "baseline_median": median_scip,
            "method_median": median_opop,
            "p_value": p_value,
            "statistic": statistic,
            "alpha": self.alpha,
            "min_effect_threshold": N_SOLVES_MIN_EFFECT,
            "baseline_distribution": _distribution(scip_vals),
            "method_distribution": _distribution(opop_vals),
        }
        return ThesisVerdict(
            claim=_CLAIMS["T2"],
            metric="n_solves",
            baseline=SCIP_DEFAULT,
            significant=significant,
            effect=effect,
            clears_threshold=clears,
            verdict=significant and clears,
            details=details,
        )

    @staticmethod
    def _t2_no_data(note: str, *, source: str | None = None) -> ThesisVerdict:
        """A T2 verdict honestly reporting the absence of usable solve counts."""
        return ThesisVerdict(
            claim=_CLAIMS["T2"],
            metric="n_solves",
            baseline=SCIP_DEFAULT,
            significant=False,
            effect=0.0,
            clears_threshold=False,
            verdict=False,
            details={"note": note, "source": source},
        )

    # -- registry -----------------------------------------------------------
    def _problem_type_map(self) -> dict[str, str]:
        if self._problem_types is not None:
            return dict(self._problem_types)
        registry = self._registry or BenchmarkRegistry.from_yaml(self._registry_path)
        return build_problem_type_map(registry)


# --------------------------------------------------------------------------- #
# High-level helper + CLI
# --------------------------------------------------------------------------- #


def _resolve_inputs(
    results: str | Path,
    events: str | Path | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Resolve a results file/dir + optional events into (records, events_path)."""
    raw = Path(results)
    if raw.is_dir():
        results_dir = raw
        results_path: Path | None = None
        for name in ("results.parquet", "results.json", "results.jsonl"):
            candidate = raw / name
            if candidate.is_file():
                results_path = candidate
                break
        if results_path is None:
            raise ValueError(f"no results.parquet/.json/.jsonl in directory {raw}")
    else:
        results_path = raw
        results_dir = raw.parent
    records = load_results(results_path)
    events_path = str(events) if events is not None else None
    if events_path is None:
        candidate = results_dir / "events.jsonl"
        if candidate.is_file():
            events_path = str(candidate)
    return records, events_path


def evaluate_theses(
    results: Any,
    *,
    events: str | Path | Iterable[Mapping[str, Any]] | None = None,
    split: str = "validation",
    one_shot_final: bool = False,
    registry: BenchmarkRegistry | None = None,
    problem_types: Mapping[str, str] | None = None,
    registry_path: str | Path = REGISTRY_PATH,
    alpha: float = ALPHA,
) -> ThesisReport:
    """Evaluate T1-T4 and return the :class:`ThesisReport`.

    ``results`` may be in-memory records (list / dict / DataFrame-like) or a path
    to a results file or a task-39 run directory (``results.parquet`` +
    ``events.jsonl`` are auto-discovered when ``results`` is a directory).
    """
    resolved_events: str | Path | Iterable[Mapping[str, Any]] | None
    if isinstance(results, (str, Path)):
        # An in-memory events iterable is kept as-is; a path/None defers to
        # _resolve_inputs (explicit events path, else auto-discovered events.jsonl).
        if events is None or isinstance(events, (str, Path)):
            records, resolved_events = _resolve_inputs(results, events)
        else:
            records, _ = _resolve_inputs(results, None)
            resolved_events = events
    else:
        records = _as_records(results)
        resolved_events = events
    evaluator = ThesisEvaluator(
        registry=registry,
        problem_types=problem_types,
        registry_path=registry_path,
        alpha=alpha,
    )
    return evaluator.evaluate(
        records, resolved_events, split=split, one_shot_final=one_shot_final
    )


def _format_summary(report: ThesisReport) -> str:
    """Render a short human-readable summary of the four verdicts."""
    lines = [
        "=" * 72,
        " OPOP thesis evaluation (T1-T4)",
        f" split: {report.split}   one_shot_final: {report.one_shot_final}"
        + f"   records: {report.n_records}",
        "-" * 72,
    ]
    for name in sorted(report.verdicts):
        verdict = report.verdicts[name]
        flag = "PASS" if verdict.verdict else "fail"
        lines.append(
            f"  {name} [{flag}]  metric={verdict.metric}  "
            + f"effect={verdict.effect * 100:+.2f}%  "
            + f"significant={verdict.significant}  clears={verdict.clears_threshold}"
        )
    lines.append("-" * 72)
    lines.append(f"  ALL PASS: {all(v.verdict for v in report.verdicts.values())}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opop.eval.theses",
        description="Falsifiable T1-T4 thesis evaluator over consolidated matrix results.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="results file (.parquet/.json/.jsonl) or a task-39 run directory",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="optional events.jsonl for T2 solve counts (auto-discovered in a run dir)",
    )
    parser.add_argument("--split", default="validation", help="dataset split the records belong to")
    parser.add_argument("--out", default="thesis_report.json", help="output JSON path")
    parser.add_argument(
        "--registry", default=str(REGISTRY_PATH), help="registry.yaml for the T3 problem-type map"
    )
    parser.add_argument(
        "--one-shot-final",
        action="store_true",
        help="permit evaluating a held-out (test/ood_test) split",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: load results, evaluate T1-T4, print a summary, write thesis_report.json."""
    args = _build_parser().parse_args(argv)
    try:
        report = evaluate_theses(
            args.results,
            events=args.events,
            split=args.split,
            one_shot_final=args.one_shot_final,
            registry_path=args.registry,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"theses failed: {exc}", file=sys.stderr)
        return 1
    print(_format_summary(report))
    out = report.write(args.out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
