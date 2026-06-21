"""Comparison report + statistical tests for OPOP experiment results.

Loads per-(instance, seed) result records for two named methods (a *baseline*
and a candidate *method*), computes per-method aggregates, runs a paired
Wilcoxon signed-rank test, and decides whether the candidate is a **win** under
the locked Win Definition (Verification Strategy of the plan):

* **Statistical test**: Wilcoxon signed-rank, ``alpha = 0.05`` (two-sided), on
  the metric values paired by ``(instance_id, seed)``.
* **Min effect for a win**: >= 10% primal-integral reduction, OR >= 20%
  shifted-geometric-mean time reduction, OR >= 5 pp solved-rate gain.
* **is_win = significant AND clears_min_effect** — never a win below the
  min-effect threshold, and never a win without significance.

Metrics (``--metric``):

``primal_integral``
    Lower is better.  Aggregate = mean across the paired values (the full
    distribution is also reported).  Relative improvement = ``(b - m) / b``.
``time``
    End-to-end wall-clock, lower is better.  Aggregate = **shifted geometric
    mean** (shift ``s = 10``, standard OR convention).  Relative improvement =
    ``(b - m) / b``.  Right-censored runtimes are handled censored-aware: a
    censored runtime is a LOWER BOUND, so for the shifted geomean it is treated
    as the time limit (``time_limit`` field when present, else the recorded
    censored runtime which already sits at the limit).
``solved_rate``
    Fraction of instances solved (not censored), higher is better.  Aggregate =
    mean of the per-record solved indicator.  Relative improvement is the
    ABSOLUTE difference ``m - b`` (a fraction, so ``0.05`` == 5 pp).

A *result record* is a mapping carrying at least ``method`` plus the fields the
chosen metric needs::

    {instance_id, method, seed, primal_integral, gap, time, solved, censored}
    # optional: time_limit (used for censored-aware shifted-geomean time)

Records may be supplied in memory (``list``/``dict``) to :func:`compare`, or
loaded from a ``.json`` / ``.jsonl`` / ``.parquet`` file via
:func:`load_results` (parquet needs ``pandas`` + ``pyarrow``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import wilcoxon

# --------------------------------------------------------------------------- #
# Locked constants (Win Definition)
# --------------------------------------------------------------------------- #

#: Shift for the shifted geometric mean of runtimes (standard OR convention).
SHIFT = 10.0

#: Minimum number of seeds for a credible win (reported, not part of is_win).
SEED_FLOOR = 5

#: Default per-metric min-effect thresholds (fractions; solved_rate is pp).
DEFAULT_MIN_EFFECT: dict[str, float] = {
    "primal_integral": 0.10,  # >= 10% reduction
    "time": 0.20,  # >= 20% reduction (shifted geomean)
    "solved_rate": 0.05,  # >= 5 pp gain
}

#: Whether a smaller value of the metric is better.
METRIC_LOWER_IS_BETTER: dict[str, bool] = {
    "primal_integral": True,
    "time": True,
    "solved_rate": False,
}

#: The metrics this module knows how to compare.
VALID_METRICS = frozenset(METRIC_LOWER_IS_BETTER)

# Guard floor for the relative-improvement denominator.
_REL_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Pure numeric helpers
# --------------------------------------------------------------------------- #


def shifted_geometric_mean(values: Sequence[float], shift: float = SHIFT) -> float:
    """Shifted geometric mean ``exp(mean(ln(v + s))) - s`` (Achterberg).

    The shift ``s`` (default :data:`SHIFT` = 10) damps the influence of very
    small values, the standard convention for aggregating solver runtimes.  An
    empty sequence returns ``0.0``.  Raises :class:`ValueError` if any
    ``value + shift <= 0`` (runtimes are non-negative, so ``v + 10 > 0`` always
    holds for valid input).
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    shifted = arr + shift
    if bool(np.any(shifted <= 0.0)):
        raise ValueError(f"shifted_geometric_mean requires every value > -{shift}")
    return float(np.exp(np.mean(np.log(shifted))) - shift)


def _truthy(value: Any) -> bool:
    """Coerce a record flag (bool / 0|1 / numpy bool) to ``bool``."""
    return bool(value)


def _record_time(rec: Mapping[str, Any]) -> float:
    """Censored-aware end-to-end runtime for one record.

    A censored runtime is a LOWER BOUND on the true solve time; per the spec we
    treat it as the time limit for the shifted-geomean aggregate.  When a
    ``time_limit`` is present we lift the censored value to it; otherwise the
    recorded censored runtime (which already sits at the limit) is used as-is.
    """
    raw = rec.get("time")
    if raw is None:
        raise ValueError(f"record missing 'time': {dict(rec)!r}")
    value = float(raw)
    if _truthy(rec.get("censored")):
        limit = rec.get("time_limit")
        if limit is not None:
            return max(value, float(limit))
    return value


def _metric_value(rec: Mapping[str, Any], metric: str) -> float:
    """Extract a single record's value for ``metric`` (censored-aware time)."""
    if metric == "primal_integral":
        raw = rec.get("primal_integral")
        if raw is None:
            raise ValueError(f"record missing 'primal_integral': {dict(rec)!r}")
        return float(raw)
    if metric == "time":
        return _record_time(rec)
    if metric == "solved_rate":
        return 1.0 if _truthy(rec.get("solved")) else 0.0
    raise ValueError(f"unknown metric {metric!r}; valid: {sorted(VALID_METRICS)}")


def _aggregate(values: Sequence[float], metric: str) -> float:
    """Per-method point aggregate: shifted geomean for time, else the mean."""
    if not values:
        return 0.0
    if metric == "time":
        return shifted_geometric_mean(values)
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _relative_improvement(baseline: float, method: float, *, lower_is_better: bool) -> float:
    """Signed relative improvement (positive == method better).

    Lower-is-better metrics use the fractional reduction ``(b - m) / b``;
    higher-is-better metrics (solved_rate) use the absolute gain ``m - b`` (a
    fraction, so ``0.05`` == 5 pp).  A near-zero baseline for a ratio metric
    yields ``0.0`` (no meaningful relative improvement is definable).
    """
    if lower_is_better:
        if abs(baseline) < _REL_EPS:
            return 0.0
        return (baseline - method) / baseline
    return method - baseline


def _wilcoxon(baseline_vals: Sequence[float], method_vals: Sequence[float]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank ``(statistic, p_value)``.

    Delegates to :func:`scipy.stats.wilcoxon` (default two-sided, ``wilcox``
    zero-handling) so the p-value matches scipy exactly.  When every paired
    difference is zero scipy cannot run; we return ``(0.0, 1.0)`` — no evidence
    of a difference — instead of raising.
    """
    b = np.asarray(baseline_vals, dtype=np.float64)
    m = np.asarray(method_vals, dtype=np.float64)
    if not bool(np.any((b - m) != 0.0)):
        return 0.0, 1.0
    statistic, p_value = cast("tuple[float, float]", wilcoxon(b, m))
    return float(statistic), float(p_value)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    """Summary of a value distribution (n / mean / median / std / min / max + raw)."""
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    return {
        "n": n,
        "mean": float(arr.mean()) if n else 0.0,
        "median": float(np.median(arr)) if n else 0.0,
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()) if n else 0.0,
        "max": float(arr.max()) if n else 0.0,
        "values": [float(x) for x in arr.tolist()],
    }


# --------------------------------------------------------------------------- #
# ComparisonReport
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Machine-readable result of one baseline-vs-method comparison.

    The first eleven fields are the required schema; the remainder add
    transparency (test name, statistic, paired-sample count, the threshold that
    was applied, the seed-floor flag, and the full per-method distributions so
    consumers see distributions, not just point estimates).
    """

    baseline: str
    method: str
    metric: str
    significant: bool
    p_value: float
    relative_improvement: float
    clears_min_effect: bool
    is_win: bool
    n_seeds: int
    baseline_value: float
    method_value: float
    alpha: float = 0.05
    test: str = "wilcoxon"
    statistic: float = 0.0
    n_pairs: int = 0
    min_effect_threshold: float = 0.0
    lower_is_better: bool = True
    meets_seed_floor: bool = False
    baseline_distribution: dict[str, Any] = field(default_factory=dict)
    method_distribution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable dict of all fields."""
        return {
            "baseline": self.baseline,
            "method": self.method,
            "metric": self.metric,
            "significant": self.significant,
            "p_value": self.p_value,
            "relative_improvement": self.relative_improvement,
            "clears_min_effect": self.clears_min_effect,
            "is_win": self.is_win,
            "n_seeds": self.n_seeds,
            "baseline_value": self.baseline_value,
            "method_value": self.method_value,
            "alpha": self.alpha,
            "test": self.test,
            "statistic": self.statistic,
            "n_pairs": self.n_pairs,
            "min_effect_threshold": self.min_effect_threshold,
            "lower_is_better": self.lower_is_better,
            "meets_seed_floor": self.meets_seed_floor,
            "baseline_distribution": dict(self.baseline_distribution),
            "method_distribution": dict(self.method_distribution),
        }

    def to_json(self) -> str:
        """Serialise to pretty, key-sorted JSON (no trailing newline)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# Record loading / normalisation
# --------------------------------------------------------------------------- #


def _normalize_records(results: Any) -> list[dict[str, Any]]:
    """Coerce supported result containers to a ``list[dict]`` of records.

    Accepts a list of record mappings, a dict wrapping them under ``records``
    or ``results``, or any object exposing ``to_dict("records")`` (e.g. a
    pandas ``DataFrame``).
    """
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


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    """Load records from a parquet file (requires pandas + pyarrow)."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise RuntimeError(
            "reading .parquet results requires pandas (and pyarrow); "
            + "install them or supply a .json file"
        ) from exc
    frame = pd.read_parquet(path)
    rows = cast("Sequence[Mapping[str, Any]]", frame.to_dict("records"))
    return [dict(r) for r in rows]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSON record per non-blank line."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(dict(json.loads(line)))
    return records


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load result records from ``.json`` / ``.jsonl`` / ``.parquet``.

    JSON may be a bare list of records or a dict wrapping them under
    ``records`` / ``results``.  Parquet support is optional (pandas + pyarrow).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return _load_parquet(p)
    if suffix == ".jsonl":
        return _load_jsonl(p)
    if suffix == ".json":
        with p.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return _normalize_records(data)
    raise ValueError(f"unsupported results extension {p.suffix!r}; use .json/.jsonl/.parquet")


# --------------------------------------------------------------------------- #
# Pairing + comparison
# --------------------------------------------------------------------------- #


def _paired(
    records: Sequence[Mapping[str, Any]], baseline: str, method: str, metric: str
) -> tuple[list[tuple[str, str]], list[float], list[float]]:
    """Pair baseline and method metric values by ``(instance_id, seed)``.

    Returns sorted keys plus the index-aligned baseline and method value lists.
    Only keys present for BOTH methods are kept (paired observations).
    """
    b_map: dict[tuple[str, str], float] = {}
    m_map: dict[tuple[str, str], float] = {}
    for rec in records:
        name = rec.get("method")
        if name == baseline:
            target = b_map
        elif name == method:
            target = m_map
        else:
            continue
        key = (str(rec.get("instance_id", "")), str(rec.get("seed", "")))
        target[key] = _metric_value(rec, metric)
    keys = sorted(b_map.keys() & m_map.keys())
    b_vals = [b_map[k] for k in keys]
    m_vals = [m_map[k] for k in keys]
    return keys, b_vals, m_vals


def compare(
    results: Any,
    *,
    baseline: str,
    method: str,
    metric: str,
    alpha: float = 0.05,
    min_effect: Mapping[str, float] | None = None,
) -> ComparisonReport:
    """Compare ``method`` against ``baseline`` on ``metric`` -> :class:`ComparisonReport`.

    Args:
        results: In-memory records (list / dict) or a DataFrame-like object.
        baseline: Method name treated as the reference.
        method: Candidate method name.
        metric: One of :data:`VALID_METRICS`.
        alpha: Significance level for the two-sided Wilcoxon test.
        min_effect: Per-metric min-effect thresholds; missing metrics fall back
            to :data:`DEFAULT_MIN_EFFECT`.

    Returns:
        A :class:`ComparisonReport`.  ``is_win`` is ``True`` iff the result is
        both statistically significant AND clears the min-effect threshold.

    Raises:
        ValueError: Unknown metric, baseline == method, or no paired
            observations exist.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"unknown metric {metric!r}; valid: {sorted(VALID_METRICS)}")
    if baseline == method:
        raise ValueError("baseline and method must be different names")

    records = _normalize_records(results)
    effect = dict(DEFAULT_MIN_EFFECT)
    if min_effect:
        effect.update(min_effect)
    threshold = float(effect[metric])

    keys, b_vals, m_vals = _paired(records, baseline, method, metric)
    if not keys:
        raise ValueError(
            "no paired (instance, seed) observations for "
            + f"baseline={baseline!r} method={method!r} on metric={metric!r}"
        )

    lower_is_better = METRIC_LOWER_IS_BETTER[metric]
    baseline_value = _aggregate(b_vals, metric)
    method_value = _aggregate(m_vals, metric)
    rel = _relative_improvement(baseline_value, method_value, lower_is_better=lower_is_better)
    statistic, p_value = _wilcoxon(b_vals, m_vals)

    significant = bool(p_value < alpha)
    clears = bool(rel >= threshold)
    is_win = significant and clears

    n_seeds = len({k[1] for k in keys})
    return ComparisonReport(
        baseline=baseline,
        method=method,
        metric=metric,
        significant=significant,
        p_value=p_value,
        relative_improvement=rel,
        clears_min_effect=clears,
        is_win=is_win,
        n_seeds=n_seeds,
        baseline_value=baseline_value,
        method_value=method_value,
        alpha=float(alpha),
        test="wilcoxon",
        statistic=statistic,
        n_pairs=len(keys),
        min_effect_threshold=threshold,
        lower_is_better=lower_is_better,
        meets_seed_floor=bool(n_seeds >= SEED_FLOOR),
        baseline_distribution=_distribution(b_vals),
        method_distribution=_distribution(m_vals),
    )


# --------------------------------------------------------------------------- #
# Reporting + CLI
# --------------------------------------------------------------------------- #


def build_min_effect(metric: str, value: float | None = None) -> dict[str, float]:
    """Default min-effect thresholds, optionally overriding ``metric``."""
    effect = dict(DEFAULT_MIN_EFFECT)
    if value is not None:
        effect[metric] = float(value)
    return effect


def format_report(report: ComparisonReport) -> str:
    """Render a human-readable table for one comparison."""
    direction = "lower" if report.lower_is_better else "higher"
    unit = "" if report.lower_is_better else " pp"
    seed_note = "" if report.meets_seed_floor else f"  (< {SEED_FLOOR} seeds: under floor)"
    verdict = "WIN" if report.is_win else "no-win"
    lines = [
        "=" * 64,
        f" {report.method}  vs  baseline {report.baseline}",
        f" metric: {report.metric}  ({direction} is better, test={report.test})",
        "-" * 64,
        f"  baseline value       : {report.baseline_value:.6g}",
        f"  method value         : {report.method_value:.6g}",
        f"  relative improvement : {report.relative_improvement * 100:+.2f}%{unit}",
        f"  min-effect threshold : {report.min_effect_threshold * 100:.2f}%{unit}",
        f"  clears min-effect    : {report.clears_min_effect}",
        f"  p-value              : {report.p_value:.4g}  (alpha={report.alpha})",
        f"  significant          : {report.significant}",
        f"  n_seeds / n_pairs    : {report.n_seeds} / {report.n_pairs}{seed_note}",
        "-" * 64,
        f"  VERDICT              : {verdict}",
        "=" * 64,
    ]
    return "\n".join(lines)


def write_report(report: ComparisonReport, path: str | Path) -> Path:
    """Write ``comparison_report.json`` (pretty, key-sorted, trailing newline)."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.to_json() + "\n", encoding="utf-8")
    return p


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opop.eval.compare",
        description="Comparison report + Wilcoxon signed-rank test with min-effect gating.",
    )
    parser.add_argument("--results", required=True, help="results file (.json/.jsonl/.parquet)")
    parser.add_argument("--baseline", required=True, help="baseline method name")
    parser.add_argument("--method", required=True, help="candidate method name")
    parser.add_argument(
        "--metric", required=True, choices=sorted(VALID_METRICS), help="metric to compare"
    )
    parser.add_argument(
        "--test", default="wilcoxon", choices=["wilcoxon"], help="paired test (Phase-1: wilcoxon)"
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level")
    parser.add_argument(
        "--min-effect",
        dest="min_effect",
        type=float,
        default=None,
        help="override the min-effect threshold for the chosen metric (fraction, e.g. 0.10)",
    )
    parser.add_argument(
        "--out", default="comparison_report.json", help="output JSON path"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI: load results, compare, print the table, write comparison_report.json."""
    args = _build_parser().parse_args(argv)
    try:
        records = load_results(args.results)
        effect = build_min_effect(args.metric, args.min_effect)
        report = compare(
            records,
            baseline=args.baseline,
            method=args.method,
            metric=args.metric,
            alpha=args.alpha,
            min_effect=effect,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    out = write_report(report, args.out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
