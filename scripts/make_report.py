#!/usr/bin/env python3
"""Regenerate figures and tables for the OPOP technical report from experiment artifacts.

Usage::

    python scripts/make_report.py --results runs/final_eval --out docs/tech-report

Reads ``results.parquet`` (or ``.json``/``.jsonl``), ``thesis_report.json``, and
``comparison_report.json`` from the run directory, then produces:

- ``figures/per_method_primal_integral.png`` — per-method primal integral distribution.
- ``figures/per_problem_type_win_rate.png`` — win rate vs scip-default by problem type.
- ``tables/thesis_verdicts.md`` — T1–T4 verdict summary.
- ``tables/ablation_cross.md`` — ablation row × baseline win matrix.
- ``tables/cross_distribution.md`` — per-problem-type comparison table.

Inserts Markdown references into ``results.md`` in-place (idempotent).
Requires ``matplotlib``; falls back to ASCII tables if unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load result records from ``results.parquet``, ``.json``, or ``.jsonl``."""
    for name in ("results.parquet", "results.json", "results.jsonl"):
        candidate = results_dir / name
        if candidate.is_file():
            if candidate.suffix == ".parquet":
                try:
                    import pandas as pd

                    frame = pd.read_parquet(candidate)
                    rows: list[Any] = list(frame.to_dict("records"))  # type: ignore[arg-type]
                    return [dict(r) for r in rows]
                except ImportError:
                    pass
            elif candidate.suffix == ".jsonl":
                records: list[dict[str, Any]] = []
                with candidate.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(dict(json.loads(line)))
                return records
            elif candidate.suffix == ".json":
                with candidate.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return [dict(r) for r in data]
                for key in ("records", "results"):
                    if key in data:
                        return [dict(r) for r in data[key]]
                raise ValueError(
                    "JSON results must be a list or carry 'records'/'results'; "
                    + f"got keys {sorted(data)}"
                )
    raise FileNotFoundError(f"no results.parquet/.json/.jsonl in {results_dir}")


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if missing."""
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            return dict(json.load(fh))
    return None


def _load_thesis_report(results_dir: Path) -> dict[str, Any] | None:
    return _load_json(results_dir / "thesis_report.json")


def _load_comparison_report(results_dir: Path) -> dict[str, Any] | None:
    return _load_json(results_dir / "comparison_report.json")


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

_has_matplotlib: bool
plt: Any = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    plt = _plt
    _has_matplotlib = True
except ImportError:  # pragma: no cover
    _has_matplotlib = False


def _gen_primal_integral_figure(
    records: list[dict[str, Any]], out_path: Path
) -> bool:
    """Generate ``per_method_primal_integral.png``."""
    if not _has_matplotlib or plt is None:  # pragma: no cover
        print("  (matplotlib unavailable; skipping figure)", file=sys.stderr)
        return False

    by_method: dict[str, list[float]] = {}
    for rec in records:
        pi = rec.get("primal_integral")
        if pi is None or not math.isfinite(float(pi)):
            continue
        method = str(rec.get("method", "unknown"))
        by_method.setdefault(method, []).append(float(pi))

    if not by_method:
        print("  (no primal-integral data for figure)", file=sys.stderr)
        return False

    methods = sorted(
        by_method, key=lambda m: float(np.median(by_method[m])) if by_method[m] else 0.0
    )
    data = [by_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=True)
    ax.set_xticklabels(methods)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    ax.set_ylabel("Primal Integral (lower is better)")
    ax.set_title("Per-Method Primal Integral Distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _gen_win_rate_figure(
    records: list[dict[str, Any]], problem_types: dict[str, str], out_path: Path
) -> bool:
    """Generate ``per_problem_type_win_rate.png``."""
    if not _has_matplotlib or plt is None:  # pragma: no cover
        print("  (matplotlib unavailable; skipping figure)", file=sys.stderr)
        return False

    by_type: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        inst = str(rec.get("instance_id", ""))
        ptype = problem_types.get(inst, "unknown")
        method = str(rec.get("method", "unknown"))
        pi = rec.get("primal_integral")
        if pi is None or not math.isfinite(float(pi)):
            continue
        by_type.setdefault(ptype, {}).setdefault(method, []).append(float(pi))

    types = sorted(by_type)
    win_rates: list[float] = []
    type_labels: list[str] = []
    for ptype in types:
        opop_vals = by_type[ptype].get("opop", [])
        scip_vals = by_type[ptype].get("scip-default", [])
        if opop_vals and scip_vals:
            opop_by_inst: dict[str, list[float]] = {}
            scip_by_inst: dict[str, list[float]] = {}
            for rec in records:
                iid = str(rec.get("instance_id", ""))
                if problem_types.get(iid) != ptype:
                    continue
                pi = rec.get("primal_integral")
                if pi is None or not math.isfinite(float(pi)):
                    continue
                method = str(rec.get("method", ""))
                if method == "opop":
                    opop_by_inst.setdefault(iid, []).append(float(pi))
                elif method == "scip-default":
                    scip_by_inst.setdefault(iid, []).append(float(pi))
            common = sorted(set(opop_by_inst) & set(scip_by_inst))
            wins = 0
            for iid in common:
                opop_med = float(np.median(opop_by_inst[iid]))
                scip_med = float(np.median(scip_by_inst[iid]))
                if opop_med < scip_med:
                    wins += 1
            total = len(common)
            rate = wins / total if total > 0 else 0.0
            win_rates.append(rate)
            type_labels.append(f"{ptype}\n(n={total})")
        elif opop_vals:
            win_rates.append(0.5)
            type_labels.append(f"{ptype}\n(no scip)")
        elif scip_vals:
            win_rates.append(0.0)
            type_labels.append(f"{ptype}\n(no opop)")

    if not win_rates:
        print("  (no problem-type win-rate data)", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(max(6, len(win_rates) * 1.2), 5))
    colors = ["#4C72B0" if r >= 0.5 else "#C44E52" for r in win_rates]
    ax.bar(
        range(len(win_rates)), win_rates, tick_label=type_labels,
        color=colors, alpha=0.8,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Win Rate (fraction of instances)")
    ax.set_title("Per-Problem-Type Win Rate (OPOP vs scip-default)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _fmt_p(value: float) -> str:
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def _generate_thesis_table(
    thesis_report: dict[str, Any] | None, out_path: Path
) -> str:
    lines: list[str] = []
    lines.append(
        "| Thesis | Verdict | Metric | Effect | Significant | Clears Threshold |"
    )
    lines.append(
        "|--------|---------|--------|--------|-------------|------------------|"
    )

    if thesis_report is None:
        lines.append("| — | — | — | — | — | — |")
        lines.append("\n*No thesis report found.*")
    else:
        for name in ("T1", "T2", "T3", "T4"):
            verdict_data = thesis_report.get(name)
            if not isinstance(verdict_data, dict):
                lines.append(f"| {name} | — | — | — | — | — |")
                continue
            verdict_flag = "PASS" if verdict_data.get("verdict") else "fail"
            metric = verdict_data.get("metric", "—")
            effect = _fmt_pct(float(verdict_data.get("effect", 0.0)))
            significant = str(verdict_data.get("significant", "—"))
            clears = str(verdict_data.get("clears_threshold", "—"))
            lines.append(
                f"| {name} | **{verdict_flag}** | {metric} | {effect} | "
                + f"{significant} | {clears} |"
            )

        meta = thesis_report.get("meta", {})
        all_pass = meta.get("all_pass", False)
        lines.append("")
        lines.append(f"**All theses pass**: {all_pass}")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return content


def _generate_ablation_cross_table(
    records: list[dict[str, Any]],
    comparison_report: dict[str, Any] | None,
    out_path: Path,
) -> str:
    lines: list[str] = []
    lines.append(
        "| Ablation Row | vs scip-default | vs scip-tuned | "
        + "vs params-only | vs cuts-only | vs modeling-agent |"
    )
    lines.append(
        "|--------------|-----------------|---------------|"
        + "----------------|--------------|-------------------|"
    )

    opop_vs_scip = "—"
    if comparison_report is not None and isinstance(comparison_report, dict):
        opop_vs_scip = "WIN" if comparison_report.get("is_win") else "no-win"

    methods = sorted(
        {str(r.get("method", "")) for r in records if r.get("method")}
    )
    expected = [
        "scip-default", "scip-tuned", "opop-params-only",
        "cuts-only", "params+cuts", "modeling-agent",
    ]
    present = [b for b in expected if b in methods]
    if not present:
        present = [b for b in methods if b != "opop"]

    cells = [opop_vs_scip]
    for bl in present[1:]:
        try:
            from opop.experiments.compare import compare as _compare

            comp = _compare(records, baseline=bl, method="opop", metric="primal_integral")
            cells.append("WIN" if comp.is_win else "no-win")
        except Exception:
            cells.append("—")

    row = "| full-opop | " + " | ".join(cells) + " |"
    lines.append(row)

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return content


def _generate_cross_distribution_table(
    records: list[dict[str, Any]],
    problem_types: dict[str, str],
    out_path: Path,
) -> str:
    lines: list[str] = []
    lines.append(
        "| Problem Type | n Instances | scip-default PI (median) | "
        + "opop PI (median) | Rel. Improvement | p-value | Win |"
    )
    lines.append(
        "|--------------|-------------|--------------------------|"
        + "------------------|------------------|---------|-----|"
    )

    by_type: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        inst = str(rec.get("instance_id", ""))
        ptype = problem_types.get(inst, "unknown")
        method = str(rec.get("method", "unknown"))
        pi = rec.get("primal_integral")
        if pi is None or not math.isfinite(float(pi)):
            continue
        by_type.setdefault(ptype, {}).setdefault(method, []).append(float(pi))

    for ptype in sorted(by_type):
        entries = by_type[ptype]
        opop_vals = entries.get("opop", [])
        scip_vals = entries.get("scip-default", [])

        inst_ids: set[str] = set()
        for rec in records:
            iid = str(rec.get("instance_id", ""))
            if problem_types.get(iid) == ptype:
                inst_ids.add(iid)
        n_inst = len(inst_ids)

        if opop_vals and scip_vals:
            scip_med = float(np.median(scip_vals))
            opop_med = float(np.median(opop_vals))
            rel_imp = (
                (scip_med - opop_med) / scip_med if abs(scip_med) > 1e-12 else 0.0
            )
            try:
                from opop.experiments.compare import compare as _compare

                typed = [
                    r for r in records
                    if problem_types.get(str(r.get("instance_id", ""))) == ptype
                ]
                comp = _compare(
                    typed,
                    baseline="scip-default",
                    method="opop",
                    metric="primal_integral",
                )
                p_val = _fmt_p(comp.p_value)
                win = "WIN" if comp.is_win else "no-win"
            except Exception:
                p_val = "—"
                win = "—"

            lines.append(
                f"| {ptype} | {n_inst} | {scip_med:.2f} | {opop_med:.2f} | "
                + f"{_fmt_pct(rel_imp)} | {p_val} | {win} |"
            )
        else:
            lines.append(f"| {ptype} | {n_inst} | — | — | — | — | — |")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return content


# ---------------------------------------------------------------------------
# Markdown injection
# ---------------------------------------------------------------------------


def _inject_tables(results_md_path: Path) -> None:
    """Replace placeholder comments in ``results.md`` with actual table/figure references."""
    if not results_md_path.is_file():
        print(
            f"  results.md not found at {results_md_path}; skipping injection",
            file=sys.stderr,
        )
        return

    content = results_md_path.read_text(encoding="utf-8")

    replacements: dict[str, str] = {
        "<!-- THESIS_VERDICTS_TABLE -->": (
            "[T1-T4 Thesis Verdicts](tables/thesis_verdicts.md)\n\n"
            + "{{% include \"tables/thesis_verdicts.md\" %}}"
        ),
        "<!-- FIGURE: per_method_primal_integral -->": (
            "![Per-Method Primal Integral Distribution]"
            + "(figures/per_method_primal_integral.png)"
        ),
        "<!-- FIGURE: per_problem_type_win_rate -->": (
            "![Per-Problem-Type Win Rate]"
            + "(figures/per_problem_type_win_rate.png)"
        ),
        "<!-- ABLATION_CROSS_TABLE -->": (
            "[Ablation Cross-Table](tables/ablation_cross.md)\n\n"
            + "{{% include \"tables/ablation_cross.md\" %}}"
        ),
        "<!-- CROSS_DISTRIBUTION_TABLE -->": (
            "[Cross-Distribution Table](tables/cross_distribution.md)\n\n"
            + "{{% include \"tables/cross_distribution.md\" %}}"
        ),
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    results_md_path.write_text(content, encoding="utf-8")
    print(f"  injected references into {results_md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate tech-report figures and tables from experiment artifacts.",
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Run directory containing results.parquet + thesis_report.json + comparison_report.json",
    )
    parser.add_argument(
        "--out",
        default="docs/tech-report",
        type=Path,
        help="Output directory for figures/ and tables/ (default: docs/tech-report)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results_dir = args.results.resolve()
    out_dir = args.out.resolve()

    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"

    if not results_dir.is_dir():
        print(
            f"error: --results must be a directory; got {results_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        records = _load_results(results_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error loading results from {results_dir}: {exc}", file=sys.stderr)
        return 1

    thesis_report = _load_thesis_report(results_dir)
    comparison_report = _load_comparison_report(results_dir)

    n = len(records)
    print(f"loaded {n} records from {results_dir}")

    problem_types: dict[str, str] = {}
    for rec in records:
        iid = str(rec.get("instance_id", ""))
        if iid:
            problem_types.setdefault(iid, "MILP")

    try:
        from opop.bench.registry import BenchmarkRegistry
        from opop.bench.sources.phase1_set import REGISTRY_PATH

        registry = BenchmarkRegistry.from_yaml(REGISTRY_PATH)
        for entry in registry.entries:
            for split_ids in entry.split.values():
                for inst_id in split_ids:
                    problem_types[str(inst_id)] = entry.problem_type
    except Exception:
        pass

    print("generating figures...")
    fig_ok = _gen_primal_integral_figure(
        records, figures_dir / "per_method_primal_integral.png"
    )
    if fig_ok:
        print(f"  wrote {figures_dir / 'per_method_primal_integral.png'}")
    fig_ok2 = _gen_win_rate_figure(
        records, problem_types, figures_dir / "per_problem_type_win_rate.png"
    )
    if fig_ok2:
        print(f"  wrote {figures_dir / 'per_problem_type_win_rate.png'}")

    print("generating tables...")
    _generate_thesis_table(thesis_report, tables_dir / "thesis_verdicts.md")
    print(f"  wrote {tables_dir / 'thesis_verdicts.md'}")
    _generate_ablation_cross_table(
        records, comparison_report, tables_dir / "ablation_cross.md"
    )
    print(f"  wrote {tables_dir / 'ablation_cross.md'}")
    _generate_cross_distribution_table(
        records, problem_types, tables_dir / "cross_distribution.md"
    )
    print(f"  wrote {tables_dir / 'cross_distribution.md'}")

    results_md = out_dir / "results.md"
    if results_md.is_file():
        _inject_tables(results_md)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
