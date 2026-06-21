#!/usr/bin/env python3
"""Regenerate figures and tables for the OPOP conference paper from experiment artifacts.

Usage::

    python scripts/make_paper.py --results runs/final_eval --out docs/paper

Reads ``results.parquet`` (or ``.json``/``.jsonl``), ``thesis_report.json``, and
``comparison_report.json`` from the run directory, then produces:

- ``figures/anytime_primal_integral.png`` — anytime primal-integral curves.
- ``figures/ablation_bar.png`` — ablation bar chart.
- ``figures/cross_distribution_heatmap.png`` — cross-distribution win-rate heatmap.
- ``tables/thesis_verdicts.md`` — T1–T4 verdict summary.
- ``tables/ablation_cross.md`` — ablation row x baseline win matrix.
- ``tables/cross_distribution.md`` — per-problem-type comparison table.

Inserts Markdown references into ``paper.md`` in-place (idempotent).
Requires ``matplotlib``; falls back to ASCII tables if unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

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

                    frame: Any = pd.read_parquet(candidate)
                    rows = cast("list[dict[str, Any]]", frame.to_dict("records"))
                    return rows
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
                    data_list: list[Any] = data
                    return cast("list[dict[str, Any]]", [dict(r) for r in data_list])
                for key in ("records", "results"):
                    if key in data:
                        inner_list: list[Any] = data[key]
                        return cast("list[dict[str, Any]]", [dict(r) for r in inner_list])
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
# Matplotlib guard
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


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _gen_anytime_curves(
    records: list[dict[str, Any]], out_path: Path
) -> bool:
    """Generate ``anytime_primal_integral.png`` — primal-integral distribution comparison."""
    if not _has_matplotlib or plt is None:  # pragma: no cover
        print("  (matplotlib unavailable; skipping anytime curves)", file=sys.stderr)
        return False

    by_method: dict[str, list[float]] = {}
    for rec in records:
        pi = rec.get("primal_integral")
        if pi is None or not math.isfinite(float(pi)):
            continue
        method = str(rec.get("method", "unknown"))
        by_method.setdefault(method, []).append(float(pi))

    if not by_method:
        print("  (no primal-integral data for anytime curves)", file=sys.stderr)
        return False

    methods = sorted(
        by_method, key=lambda m: float(np.median(by_method[m])) if by_method[m] else 0.0
    )
    target_methods = [m for m in ("opop", "scip-default", "opop-params-only", "modeling-agent")
                      if m in methods]
    if not target_methods:
        target_methods = methods[:4]

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(len(target_methods))
    data = [by_method[m] for m in target_methods]
    bp = ax.boxplot(data, positions=positions, patch_artist=True, showfliers=True, widths=0.5)
    colors = ["#2C7BB6", "#D7191C", "#FDAE61", "#ABD9E9"]
    for patch, color in zip(bp["boxes"], colors[: len(target_methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(target_methods)
    ax.set_ylabel("Primal Integral (lower is better)")
    ax.set_title("Anytime Primal-Integral Distribution by Method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _gen_ablation_bar(
    records: list[dict[str, Any]], out_path: Path
) -> bool:
    """Generate ``ablation_bar.png`` — per-ablation-stage relative improvement over scip-default."""
    if not _has_matplotlib or plt is None:  # pragma: no cover
        print("  (matplotlib unavailable; skipping ablation bar)", file=sys.stderr)
        return False

    methods_present = sorted({str(r.get("method", "")) for r in records if r.get("method")})
    ablation_order = [
        "scip-default", "opop-params-only", "cuts-only", "params+cuts", "opop",
        "modeling-agent",
    ]
    ablated = [m for m in ablation_order if m in methods_present]
    if not ablated:
        ablated = methods_present

    def _median_pi(method: str) -> float:
        vals = [float(r["primal_integral"]) for r in records
                if r.get("method") == method
                and r.get("primal_integral") is not None
                and math.isfinite(float(r["primal_integral"]))]
        return float(np.median(vals)) if vals else 0.0

    scip_pi = _median_pi("scip-default")
    labels: list[str] = []
    improvements: list[float] = []
    colors: list[str] = []
    for method in ablated:
        pi = _median_pi(method)
        rel_imp = (
            (scip_pi - pi) / scip_pi if abs(scip_pi) > 1e-12 and pi > 0 else 0.0
        )
        labels.append(method)
        improvements.append(rel_imp * 100)
        colors.append("#2C7BB6" if rel_imp >= 0.10 else "#D7191C")

    if not improvements:
        print("  (no ablation data for bar chart)", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(max(7, len(improvements) * 1.3), 5))
    bars = ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.8)
    ax.axhline(10.0, color="gray", linestyle="--", linewidth=0.8, label="10% threshold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Relative Improvement (%)")
    ax.set_title("Ablation: Relative Improvement over scip-default")
    ax.legend()
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{imp:+.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _gen_cross_distribution_heatmap(
    records: list[dict[str, Any]],
    problem_types: dict[str, str],
    out_path: Path,
) -> bool:
    """Generate ``cross_distribution_heatmap.png`` — win-rate heatmap across problem types x baselines."""
    if not _has_matplotlib or plt is None:  # pragma: no cover
        print("  (matplotlib unavailable; skipping heatmap)", file=sys.stderr)
        return False

    baseline_methods = ["scip-default", "opop-params-only", "modeling-agent"]
    available_baselines = [b for b in baseline_methods
                           if any(r.get("method") == b for r in records)]

    by_type_bl: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        inst = str(rec.get("instance_id", ""))
        ptype = problem_types.get(inst, "MILP")
        method = str(rec.get("method", ""))
        pi = rec.get("primal_integral")
        if pi is None or not math.isfinite(float(pi)):
            continue
        by_type_bl.setdefault(ptype, {})
        by_type_bl[ptype].setdefault(method, []).append(float(pi))

    problem_type_list = sorted(by_type_bl)
    if len(problem_type_list) < 1:
        print("  (insufficient data for heatmap)", file=sys.stderr)
        return False

    heatmap: list[list[float]] = []
    for ptype in problem_type_list:
        row: list[float] = []
        opop_vals: list[float] = by_type_bl[ptype].get("opop", [])
        for bl in available_baselines:
            bl_vals: list[float] = by_type_bl[ptype].get(bl, [])
            if opop_vals and bl_vals:
                # Simple win rate: fraction where opop median < baseline median per instance
                opop_by_inst: dict[str, list[float]] = {}
                bl_by_inst: dict[str, list[float]] = {}
                for rec in records:
                    iid = str(rec.get("instance_id", ""))
                    if problem_types.get(iid, "MILP") != ptype:
                        continue
                    pi = rec.get("primal_integral")
                    if pi is None or not math.isfinite(float(pi)):
                        continue
                    method = str(rec.get("method", ""))
                    if method == "opop":
                        opop_by_inst.setdefault(iid, []).append(float(pi))
                    elif method == bl:
                        bl_by_inst.setdefault(iid, []).append(float(pi))
                common = sorted(set(opop_by_inst) & set(bl_by_inst))
                wins = sum(1 for iid in common
                           if float(np.median(opop_by_inst[iid])) < float(np.median(bl_by_inst[iid])))
                row.append(wins / len(common) if common else 0.0)
            else:
                row.append(0.0)
        heatmap.append(row)

    if not heatmap:
        print("  (no heatmap data)", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(max(6, len(available_baselines) * 1.8),
                                    max(3, len(problem_type_list) * 0.6)))
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(available_baselines)))
    ax.set_xticklabels(available_baselines, rotation=30, ha="right")
    ax.set_yticks(range(len(problem_type_list)))
    ax.set_yticklabels(problem_type_list)
    ax.set_title("Win Rate Heatmap (OPOP vs Baselines, by Problem Type)")

    # Annotate cells
    for i in range(len(problem_type_list)):
        for j in range(len(available_baselines)):
            val = heatmap[i][j]
            text_color = "white" if val < 0.3 or val > 0.7 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Win Rate")
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
    """Generate ``tables/thesis_verdicts.md``."""
    lines: list[str] = []
    lines.append(
        "| Thesis | Verdict | Metric | Effect | Significant | Clears Threshold |"
    )
    lines.append(
        "|--------|---------|--------|--------|-------------|------------------|"
    )

    if thesis_report is None:
        lines.append("| — | — | — | — | — | — |")
        lines.append("")
        lines.append("*No thesis report found. Run the experiment matrix first.*")
    else:
        all_verdicts: list[bool] = []
        for name in ("T1", "T2", "T3", "T4"):
            verdict_data = thesis_report.get(name)
            if not isinstance(verdict_data, dict):
                lines.append(f"| {name} | — | — | — | — | — |")
                continue
            vd: dict[str, Any] = cast("dict[str, Any]", verdict_data)
            verdict_flag = "PASS" if vd.get("verdict") else "fail"
            effect_val: Any = vd.get("effect", 0.0)
            sig_val: Any = vd.get("significant", "—")
            clears_val: Any = vd.get("clears_threshold", "—")
            verdict_val: Any = vd.get("verdict", False)
            metric = str(vd.get("metric", "—"))
            effect = _fmt_pct(float(effect_val))
            significant = str(sig_val)
            clears = str(clears_val)
            lines.append(
                f"| {name} | **{verdict_flag}** | {metric} | {effect} | "
                + f"{significant} | {clears} |"
            )
            all_verdicts.append(bool(verdict_val))

        meta = thesis_report.get("meta", {})
        all_pass = meta.get("all_pass", all(all_verdicts) if all_verdicts else False)
        lines.append("")
        lines.append(f"**All theses pass**: {all_pass}")
        lines.append(f"*Split: {meta.get('split', 'unknown')}, "
                      + f"n_records: {meta.get('n_records', 0)}*")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return content


def _generate_ablation_cross_table(
    records: list[dict[str, Any]],
    comparison_report: dict[str, Any] | None,
    out_path: Path,
) -> str:
    """Generate ``tables/ablation_cross.md`` — ablation row x baseline win matrix."""
    lines: list[str] = []
    baseline_cols = [
        "scip-default", "scip-tuned", "opop-params-only",
        "cuts-only", "params+cuts", "modeling-agent",
    ]
    present_cols = [b for b in baseline_cols
                    if any(r.get("method") == b for r in records)]
    if not present_cols:
        present_cols = baseline_cols[:3]

    header = "| Ablation Row | " + " | ".join(f"vs {b}" for b in present_cols) + " |"
    sep = "|--------------|" + "|".join("---------------" for _ in present_cols) + "|"
    lines.append(header)
    lines.append(sep)

    _compare: Any = None
    try:
        from opop.experiments.compare import compare as _compare_imported
        _compare = _compare_imported
        _has_compare = True
    except ImportError:
        _has_compare = False

    def _cell(row_method: str, bl: str) -> str:
        if not _has_compare:
            return "—"
        try:
            comp = _compare(
                records, baseline=bl, method=row_method, metric="primal_integral"
            )
            return "WIN" if comp.is_win else "no-win"
        except Exception:
            return "—"

    ablation_methods = ["scip-default", "opop-params-only", "cuts-only",
                        "params+cuts", "opop"]
    present_rows = [m for m in ablation_methods
                    if any(r.get("method") == m for r in records)]
    if not present_rows:
        present_rows = ["opop"]

    for row_method in present_rows:
        cells = " | ".join(_cell(row_method, bl) for bl in present_cols)
        lines.append(f"| {row_method} | {cells} |")

    # Add comparison report summary if available
    if comparison_report is not None:
        lines.append("")
        ver = "WIN" if comparison_report.get("is_win") else "no-win"
        lines.append(f"*comparison_report.json: opop vs scip-default → **{ver}** "
                      + f"(p={comparison_report.get('p_value', '?'):.4g}, "
                      + f"rel_imp={float(comparison_report.get('relative_improvement', 0)) * 100:+.1f}%)*")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return content


def _generate_cross_distribution_table(
    records: list[dict[str, Any]],
    problem_types: dict[str, str],
    out_path: Path,
) -> str:
    """Generate ``tables/cross_distribution.md`` — per-problem-type comparison."""
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
        ptype = problem_types.get(inst, "MILP")
        method = str(rec.get("method", "unknown"))
        pi = rec.get("primal_integral")
        if pi is None or not math.isfinite(float(pi)):
            continue
        by_type.setdefault(ptype, {}).setdefault(method, []).append(float(pi))

    if not by_type:
        lines.append("| — | — | — | — | — | — | — |")
        content = "\n".join(lines) + "\n"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return content

    _compare2: Any = None
    try:
        from opop.experiments.compare import compare as _compare_imported2
        _compare2 = _compare_imported2
        _has_compare = True
    except ImportError:
        _has_compare = False

    for ptype in sorted(by_type):
        entries = by_type[ptype]
        opop_vals = entries.get("opop", [])
        scip_vals = entries.get("scip-default", [])

        inst_ids: set[str] = set()
        for rec in records:
            iid = str(rec.get("instance_id", ""))
            if problem_types.get(iid, "MILP") == ptype:
                inst_ids.add(iid)
        n_inst = len(inst_ids)

        if opop_vals and scip_vals:
            scip_med = float(np.median(scip_vals))
            opop_med = float(np.median(opop_vals))
            rel_imp = (
                (scip_med - opop_med) / scip_med if abs(scip_med) > 1e-12 else 0.0
            )
            if _has_compare:
                try:
                    typed = [
                        r for r in records
                        if problem_types.get(str(r.get("instance_id", "")), "MILP") == ptype
                    ]
                    comp = _compare2(
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
            else:
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
# Markdown injection into paper.md
# ---------------------------------------------------------------------------


def _inject_references(paper_md_path: Path) -> None:
    """Replace placeholder comments in ``paper.md`` with actual table/figure references."""
    if not paper_md_path.is_file():
        print(
            f"  paper.md not found at {paper_md_path}; skipping injection",
            file=sys.stderr,
        )
        return

    content = paper_md_path.read_text(encoding="utf-8")

    replacements: dict[str, str] = {
        "<!-- THESIS_VERDICTS_TABLE -->": (
            "[T1-T4 Thesis Verdicts](tables/thesis_verdicts.md)\n\n"
            + "{{% include \"tables/thesis_verdicts.md\" %}}"
        ),
        "<!-- FIGURE: anytime_primal_integral -->": (
            "![Anytime Primal-Integral Distribution]"
            + "(figures/anytime_primal_integral.png)"
        ),
        "<!-- FIGURE: ablation_bar -->": (
            "![Ablation Bar Chart](figures/ablation_bar.png)"
        ),
        "<!-- FIGURE: cross_distribution_heatmap -->": (
            "![Cross-Distribution Win-Rate Heatmap]"
            + "(figures/cross_distribution_heatmap.png)"
        ),
        "<!-- ABLATION_CROSS_TABLE -->": (
            "[Ablation Cross-Table](tables/ablation_cross.md)\n\n"
            + "{{% include \"tables/ablation_cross.md\" %}}"
        ),
        "<!-- CROSS_DISTRIBUTION_TABLE -->": (
            "[Cross-Distribution Table](tables/cross_distribution.md)\n\n"
            + "{{% include \"tables/cross_distribution.md\" %}}"
        ),
        "<!-- NEGATIVE_RESULTS -->": (
            "Negative results and non-wins are documented in the thesis "
            + "report (`thesis_report.json`). See the thesis verdicts table "
            + "above for per-thesis outcomes. Every non-win comparison "
            + "appears in the ablation cross-table with its exact status."
        ),
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    paper_md_path.write_text(content, encoding="utf-8")
    print(f"  injected references into {paper_md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate conference paper figures and tables from experiment artifacts.",
    )
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        help="Run directory containing results.parquet + thesis_report.json + comparison_report.json",
    )
    parser.add_argument(
        "--out",
        default="docs/paper",
        type=Path,
        help="Output directory for figures/ and tables/ (default: docs/paper)",
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

    # Build problem-type map; default to "MILP" if not available
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

    # --- Figures ---
    print("generating figures...")
    n_figs = 0
    if _gen_anytime_curves(records, figures_dir / "anytime_primal_integral.png"):
        print(f"  wrote {figures_dir / 'anytime_primal_integral.png'}")
        n_figs += 1
    if _gen_ablation_bar(records, figures_dir / "ablation_bar.png"):
        print(f"  wrote {figures_dir / 'ablation_bar.png'}")
        n_figs += 1
    if _gen_cross_distribution_heatmap(
        records, problem_types, figures_dir / "cross_distribution_heatmap.png"
    ):
        print(f"  wrote {figures_dir / 'cross_distribution_heatmap.png'}")
        n_figs += 1
    if n_figs == 0:
        print("  (no figures generated; matplotlib may be missing or no data)")

    # --- Tables ---
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

    # --- Inject references into paper.md ---
    paper_md = out_dir / "paper.md"
    if paper_md.is_file():
        _inject_references(paper_md)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
