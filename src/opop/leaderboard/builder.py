"""Leaderboard builder: aggregate results into a static HTML + markdown page.

Reads ``results.parquet`` (via :func:`opop.experiments.compare.load_results`) and
``thesis_report.json`` from a run directory, aggregates per-method metrics with
95% bootstrap confidence intervals, and emits a self-contained static site.

The headline table contains ONLY ``test`` / ``ood_test`` results (or all results
clearly labeled by split); ``dev`` / ``validation`` rows are never presented as
headline results.
"""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from opop.experiments.compare import load_results, shifted_geometric_mean

__all__ = ["LeaderboardBuilder", "LeaderboardData"]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Splits considered "headline" (held-out evaluation).
_HEADLINE_SPLITS = frozenset({"test", "ood_test"})

#: Number of bootstrap resamples for confidence intervals.
_N_BOOTSTRAP = 2000

#: Confidence level for the interval.
_CI_LEVEL = 0.95


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _MethodRow:
    """One row in the aggregated leaderboard table."""

    method: str
    split: str
    n_instances: int
    n_seeds: int
    primal_integral_mean: float
    primal_integral_median: float
    solved_rate: float
    time_sgm: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True, slots=True)
class LeaderboardData:
    """Fully aggregated leaderboard payload ready for rendering."""

    rows: tuple[_MethodRow, ...]
    thesis_verdicts: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""

    def headline_rows(self) -> tuple[_MethodRow, ...]:
        """Return only rows from held-out (test/ood_test) splits."""
        return tuple(r for r in self.rows if r.split in _HEADLINE_SPLITS)

    def all_rows(self) -> tuple[_MethodRow, ...]:
        """Return every row (all splits, clearly labeled)."""
        return self.rows


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def _bootstrap_ci(
    values: list[float],
    *,
    n_resamples: int = _N_BOOTSTRAP,
    level: float = _CI_LEVEL,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI for the mean of ``values``.

    Returns ``(lower, upper)``.  With fewer than 2 values the CI collapses to
    the point estimate (no resampling possible).
    """
    if len(values) < 2:
        point = float(np.mean(values)) if values else 0.0
        return point, point
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.empty(n_resamples, dtype=np.float64)
    n = arr.size
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(np.mean(sample))
    alpha = 1.0 - level
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _infer_split(records: list[dict[str, Any]]) -> str:
    """Best-effort split inference from result records.

    Checks for an explicit ``split`` column; falls back to ``"unknown"``.
    """
    splits: set[str] = set()
    for rec in records:
        s = rec.get("split")
        if isinstance(s, str) and s:
            splits.add(s)
    if len(splits) == 1:
        return splits.pop()
    if splits:
        # Multiple splits present — return the "highest" held-out if any.
        for candidate in ("ood_test", "test"):
            if candidate in splits:
                return candidate
        return sorted(splits)[0]
    return "unknown"


def _aggregate_records(records: list[dict[str, Any]]) -> list[_MethodRow]:
    """Group records by ``(method, split)`` and compute aggregate metrics."""
    # Group by (method, split, instance_id, seed) to deduplicate, then by (method, split).
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "pi_values": [],
            "solved_flags": [],
            "time_values": [],
            "instances": set(),
            "seeds": set(),
        }
    )
    for rec in records:
        method = str(rec.get("method", "unknown"))
        split = str(rec.get("split", "unknown"))
        key = (method, split)
        g = groups[key]

        pi = rec.get("primal_integral")
        if pi is not None:
            try:
                pi_f = float(pi)
                if math.isfinite(pi_f):
                    g["pi_values"].append(pi_f)
            except (TypeError, ValueError):
                pass

        solved = rec.get("solved")
        if solved is not None:
            g["solved_flags"].append(1.0 if bool(solved) else 0.0)

        time_val = rec.get("time")
        if time_val is not None:
            try:
                g["time_values"].append(float(time_val))
            except (TypeError, ValueError):
                pass

        inst = rec.get("instance_id")
        if inst is not None:
            g["instances"].add(str(inst))
        seed = rec.get("seed")
        if seed is not None:
            g["seeds"].add(str(seed))

    rows: list[_MethodRow] = []
    for (method, split), g in sorted(groups.items()):
        pi_vals = g["pi_values"]
        solved_flags = g["solved_flags"]
        time_vals = g["time_values"]

        pi_mean = float(np.mean(pi_vals)) if pi_vals else 0.0
        pi_median = float(np.median(pi_vals)) if pi_vals else 0.0
        solved_rate = float(np.mean(solved_flags)) if solved_flags else 0.0
        time_sgm = shifted_geometric_mean(time_vals) if time_vals else 0.0
        ci_lo, ci_hi = _bootstrap_ci(pi_vals)

        rows.append(
            _MethodRow(
                method=method,
                split=split,
                n_instances=len(g["instances"]),
                n_seeds=len(g["seeds"]),
                primal_integral_mean=pi_mean,
                primal_integral_median=pi_median,
                solved_rate=solved_rate,
                time_sgm=time_sgm,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

_CSS = """\
:root {
  --bg: #0e1117;
  --fg: #e6edf3;
  --accent: #58a6ff;
  --border: #30363d;
  --surface: #161b22;
  --pass: #3fb950;
  --fail: #f85149;
  --muted: #8b949e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'IBM Plex Mono', 'JetBrains Mono', 'Fira Code', monospace;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}
h1 { font-size: 1.8rem; margin-bottom: 0.5rem; color: var(--accent); }
h2 { font-size: 1.3rem; margin: 2rem 0 0.8rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
h3 { font-size: 1.1rem; margin: 1.5rem 0 0.5rem; color: var(--muted); }
p, li { font-size: 0.9rem; color: var(--muted); }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.85rem;
}
th, td {
  padding: 0.5rem 0.75rem;
  text-align: left;
  border: 1px solid var(--border);
}
th {
  background: var(--surface);
  color: var(--accent);
  font-weight: 600;
}
tr:nth-child(even) td { background: rgba(22,27,34,0.5); }
.pass { color: var(--pass); font-weight: 700; }
.fail { color: var(--fail); font-weight: 700; }
.tag {
  display: inline-block;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  font-size: 0.75rem;
  font-weight: 600;
}
.tag-test { background: rgba(88,166,255,0.15); color: var(--accent); }
.tag-dev { background: rgba(139,148,158,0.15); color: var(--muted); }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--muted); }
"""


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(text, quote=True)


def _fmt_float(value: float, *, decimals: int = 4) -> str:
    """Format a float for table display."""
    if value == 0.0:
        return "0.0000"
    return f"{value:.{decimals}f}"


def _render_table(rows: tuple[_MethodRow, ...] | list[_MethodRow], *, caption: str) -> str:
    """Render an HTML table for a set of leaderboard rows."""
    if not rows:
        return f"<p>No data for: {_esc(caption)}</p>"
    header = (
        "<tr>"
        "<th>Method</th>"
        "<th>Split</th>"
        "<th>Instances</th>"
        "<th>Seeds</th>"
        "<th>Primal Int. (mean)</th>"
        "<th>Primal Int. (median)</th>"
        "<th>95% CI</th>"
        "<th>Solved Rate</th>"
        "<th>Time (SGM)</th>"
        "</tr>"
    )
    body_lines: list[str] = []
    for r in rows:
        split_tag = (
            f'<span class="tag tag-test">{_esc(r.split)}</span>'
            if r.split in _HEADLINE_SPLITS
            else f'<span class="tag tag-dev">{_esc(r.split)}</span>'
        )
        ci_str = f"[{_fmt_float(r.ci_lower)}, {_fmt_float(r.ci_upper)}]"
        body_lines.append(
            "<tr>"
            f"<td>{_esc(r.method)}</td>"
            f"<td>{split_tag}</td>"
            f"<td>{r.n_instances}</td>"
            f"<td>{r.n_seeds}</td>"
            f"<td>{_fmt_float(r.primal_integral_mean)}</td>"
            f"<td>{_fmt_float(r.primal_integral_median)}</td>"
            f"<td>{ci_str}</td>"
            f"<td>{_fmt_float(r.solved_rate, decimals=2)}</td>"
            f"<td>{_fmt_float(r.time_sgm, decimals=2)}</td>"
            "</tr>"
        )
    return f"<table>\n<caption>{_esc(caption)}</caption>\n{header}\n" + "\n".join(body_lines) + "\n</table>"


def _render_thesis_panel(verdicts: dict[str, Any]) -> str:
    """Render the thesis verdict summary panel."""
    if not verdicts:
        return "<p>No thesis report available.</p>"
    lines: list[str] = ["<table>", "<tr><th>Thesis</th><th>Verdict</th><th>Effect</th><th>Claim</th></tr>"]
    for name in sorted(verdicts):
        if name == "meta":
            continue
        v = verdicts[name]
        if not isinstance(v, dict):
            continue
        passed = v.get("verdict", False)
        cls = "pass" if passed else "fail"
        flag = "PASS" if passed else "FAIL"
        effect = v.get("effect", 0.0)
        claim = v.get("claim", "")
        lines.append(
            f"<tr>"
            f"<td>{_esc(name)}</td>"
            f'<td class="{cls}">{flag}</td>'
            f"<td>{effect * 100:+.2f}%</td>"
            f"<td>{_esc(str(claim))}</td>"
            f"</tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def _render_html(data: LeaderboardData) -> str:
    """Render the full ``index.html`` page."""
    headline = data.headline_rows()
    all_rows = data.all_rows()
    non_headline = tuple(r for r in all_rows if r.split not in _HEADLINE_SPLITS)

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>OPOP Leaderboard</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<h1>OPOP Leaderboard</h1>",
        f"<p>Run: {_esc(data.run_dir)}</p>",
    ]

    # Headline results
    parts.append("<h2>Headline Results (Test / OOD)</h2>")
    parts.append(
        "<p>These results are from held-out test/ood_test splits. "
        "Dev/validation results are NOT included here.</p>"
    )
    parts.append(_render_table(headline, caption="Headline: held-out test/ood results"))

    # Thesis panel
    parts.append("<h2>Thesis Verdicts (T1-T4)</h2>")
    parts.append(_render_thesis_panel(data.thesis_verdicts))

    # All results (labeled)
    if non_headline:
        parts.append("<h2>All Results (by Split)</h2>")
        parts.append(
            "<p>Every row is labeled with its dataset split. "
            "Dev/validation rows are for development only.</p>"
        )
        parts.append(_render_table(all_rows, caption="All results, labeled by split"))

    # Methodology
    parts.append("<h2>Methodology</h2>")
    parts.append(
        "<ul>"
        "<li><strong>Primal Integral</strong>: area under the incumbent objective curve; "
        "lower is better. Aggregated as the mean across (instance, seed) pairs.</li>"
        "<li><strong>Solved Rate</strong>: fraction of instances solved to optimality "
        "(not censored); higher is better.</li>"
        "<li><strong>Time (SGM)</strong>: shifted geometric mean of end-to-end wall-clock "
        "runtimes (shift=10, standard OR convention); lower is better.</li>"
        "<li><strong>95% CI</strong>: bootstrap confidence interval for the mean primal "
        f"integral ({_N_BOOTSTRAP} resamples, seed=42).</li>"
        "<li><strong>Thesis verdicts</strong>: T1-T4 evaluated per the locked Win Definition "
        "(Wilcoxon signed-rank, alpha=0.05, min-effect gating).</li>"
        "</ul>"
    )

    # Limitations
    parts.append("<h2>Limitations / Splits &amp; Leakage Policy</h2>")
    parts.append(
        "<ul>"
        "<li>Headline results contain ONLY held-out <code>test</code> / <code>ood_test</code> "
        "instances. Dev/validation numbers are shown separately and must NOT be cited as "
        "headline performance.</li>"
        "<li>Submissions require a <code>repro_manifest.json</code>, a passing "
        "<code>leakage_audit.json</code>, and a sealed registry lock. "
        "Submissions missing any artifact are rejected.</li>"
        "<li>No test/ood_test instance may enter a tuning/proposal path (Metis leakage "
        "policy). The leakage audit cross-references the run's events journal against "
        "the registry's held-out split manifest.</li>"
        "<li>Confidence intervals are bootstrap-based and assume i.i.d. sampling of "
        "instances; they do not account for instance-set bias.</li>"
        "<li>Results are from synthetic benchmarks (Phase 1); generalisation to "
        "real-world MILP suites is an open question.</li>"
        "</ul>"
    )

    parts.append(
        "<footer>"
        "Generated by <code>opop.leaderboard</code>. "
        "Static HTML, no external dependencies."
        "</footer>"
    )
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _render_markdown(data: LeaderboardData) -> str:
    """Render a markdown fallback of the leaderboard."""
    lines: list[str] = [
        "# OPOP Leaderboard",
        "",
        f"Run: `{data.run_dir}`",
        "",
    ]

    headline = data.headline_rows()
    if headline:
        lines.append("## Headline Results (Test / OOD)")
        lines.append("")
        lines.append("| Method | Split | Instances | Seeds | PI (mean) | PI (median) | 95% CI | Solved Rate | Time (SGM) |")
        lines.append("|--------|-------|-----------|-------|-----------|-------------|--------|-------------|------------|")
        for r in headline:
            ci = f"[{_fmt_float(r.ci_lower)}, {_fmt_float(r.ci_upper)}]"
            lines.append(
                f"| {r.method} | {r.split} | {r.n_instances} | {r.n_seeds} "
                f"| {_fmt_float(r.primal_integral_mean)} | {_fmt_float(r.primal_integral_median)} "
                f"| {ci} | {_fmt_float(r.solved_rate, decimals=2)} | {_fmt_float(r.time_sgm, decimals=2)} |"
            )
        lines.append("")

    # Thesis verdicts
    if data.thesis_verdicts:
        lines.append("## Thesis Verdicts (T1-T4)")
        lines.append("")
        lines.append("| Thesis | Verdict | Effect | Claim |")
        lines.append("|--------|---------|--------|-------|")
        for name in sorted(data.thesis_verdicts):
            if name == "meta":
                continue
            v = data.thesis_verdicts[name]
            if not isinstance(v, dict):
                continue
            flag = "PASS" if v.get("verdict", False) else "FAIL"
            effect = v.get("effect", 0.0)
            claim = str(v.get("claim", ""))
            lines.append(f"| {name} | {flag} | {effect * 100:+.2f}% | {claim} |")
        lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Primal Integral**: area under the incumbent objective curve; lower is better.")
    lines.append("- **Solved Rate**: fraction of instances solved to optimality; higher is better.")
    lines.append("- **Time (SGM)**: shifted geometric mean of runtimes (shift=10); lower is better.")
    lines.append(f"- **95% CI**: bootstrap CI for mean primal integral ({_N_BOOTSTRAP} resamples).")
    lines.append("")

    # Limitations
    lines.append("## Limitations / Splits & Leakage Policy")
    lines.append("")
    lines.append("- Headline results contain ONLY held-out test/ood_test instances.")
    lines.append("- Submissions require repro_manifest.json, leakage_audit.json, and a sealed registry lock.")
    lines.append("- No test/ood_test instance may enter a tuning/proposal path (Metis leakage policy).")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LeaderboardBuilder
# --------------------------------------------------------------------------- #


class LeaderboardBuilder:
    """Build a static leaderboard site from a run directory.

    Args:
        results_path: Path to ``results.parquet`` / ``.json`` / ``.jsonl``.
        thesis_path: Optional path to ``thesis_report.json``.
        split: Explicit split label (overrides auto-inference from records).
    """

    def __init__(
        self,
        results_path: str | Path,
        *,
        thesis_path: str | Path | None = None,
        split: str | None = None,
    ) -> None:
        self._results_path = Path(results_path)
        self._thesis_path = Path(thesis_path) if thesis_path is not None else None
        self._split = split

    def build(self) -> LeaderboardData:
        """Load results + thesis report and return aggregated :class:`LeaderboardData`."""
        records = load_results(self._results_path)

        # Tag records with split if not already present.
        split_label = self._split or _infer_split(records)
        for rec in records:
            if "split" not in rec or not rec["split"]:
                rec["split"] = split_label

        rows = _aggregate_records(records)

        # Load thesis report if available.
        verdicts: dict[str, Any] = {}
        if self._thesis_path is not None and self._thesis_path.is_file():
            try:
                verdicts = json.loads(self._thesis_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        run_dir = str(self._results_path.parent)
        return LeaderboardData(
            rows=tuple(rows),
            thesis_verdicts=verdicts,
            run_dir=run_dir,
        )

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        """Build the leaderboard and write ``index.html`` + ``leaderboard.md``.

        Returns ``(html_path, md_path)``.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        data = self.build()
        html_content = _render_html(data)
        md_content = _render_markdown(data)

        html_path = out / "index.html"
        md_path = out / "leaderboard.md"
        html_path.write_text(html_content, encoding="utf-8")
        md_path.write_text(md_content, encoding="utf-8")
        return html_path, md_path
