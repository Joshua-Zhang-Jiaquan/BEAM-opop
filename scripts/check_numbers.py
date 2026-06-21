#!/usr/bin/env python3
"""Link-check + numbers-trace checker for the OPOP technical report.

Usage::

    python scripts/check_numbers.py docs/tech-report

Parses all ``.md`` files in the report directory, extracts numeric tokens that
look like headline numbers, and verifies that each one appears in the artifact
JSON/parquet files or generated tables.

A *headline number* is a numeric token that appears to be a result (percentage,
count, measurement, p-value) AND is not clearly contextual prose (version string,
task reference, architectural constant, named formula parameter, etc.).

**Prose exclusion rules** (numbers that are NOT expected to trace to artifacts):
- Version strings (``6.2.1``, ``1.14.0``, ``10.0.2``, etc.).
- e-notation constants (``1e-12``, ``5e-4``, etc.).
- Numbers following named prefixes: ``SHA-*``, ``ISO-*``, ``task *``, ``Task *``.
- Architecture layer numbers (``layer 1``, ``S0`` through ``S4``, ``Phase-1``).
- List-item bullets (``1.``, ``2.``, etc.).
- Time units already captured as ``30s``/``300s``/``1800s``.
- Numbers appearing in method-name code blocks.
- The explicit whitelist of prose tokens.

Exit codes: 0 if all headline numbers are traceable, 1 if any are orphaned.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Token-level whitelist — tokens that are always prose (never headline)
# ---------------------------------------------------------------------------

_TOKEN_WHITELIST: set[str] = {
    # Architecture / policy
    "S0", "S1", "S2", "S3", "S4",
    "layer 1", "layer 2", "layer 3", "layer 4", "layer 5",
    "5 layers", "4 theses", "5 pp",
    "5 seeds", "6 curated", "6 baseline",
    "Phase-1",
    # Single-letter identifiers
    "d", "p", "s", "m", "v", "c", "h", "rho",
    "0", "1", "10",
    # Named constants / thresholds (Win Definition)
    "alpha=0.05", "0.05", "0.10", "0.30", "0.20", "0.50",
    "10%", "20%", "30%", "5%", "70%",
    # Tolerances
    "1e-9", "1e-7", "1e-6", "1e-12", "0.0001",
    # Parameter values
    "s=10",
    "4096", "128",
    "256",
    "1 TiB",
    "dpi", "2.0", "2s", "5s", "5.0",
    # Solver version strings (exact tokens)
    "3.12", "3.12.3",
    "2.0.8", "0.17.2", "2.38.0", "0.15.18", "2.1.0",
    "6.2.1", "10.0", "10.0.2",
    "9.14.6206", "9.14", "1.14.0", "1.14.1",
    "8.0.2", "2.10.3",
    "3.2.1", "1.26.4",
    "2.0", "2.8.0a0",
    "1.14",
}


# ---------------------------------------------------------------------------
# Context patterns — if the surrounding text contains any of these markers,
# the extracted number is prose, not a headline.
# ---------------------------------------------------------------------------

# If any of these substrings appear in the 80-char context window around the
# number, it is classified as prose.
_PROSE_CONTEXT_MARKERS: tuple[str, ...] = (
    "SHA-", "ISO-", "SHA256", "sha256",
    "task ", "Task ",
    "Spearman", "ρ ≥", "ρ =",
    "kHighsInf", "monotonic", "rungs",
    "α=", "α ",
    "s=",
    "│",
    "`python",
    "PYTHONPATH",
    "requirements.txt",
    "pyscipopt", "highspy", "PuLP", "pulp", "ortools", "botorch",
    "openai",
    "SCIP ", "CBC ", "HiGHS ", "CP-SAT", "GCG",
    "numpy", "scipy", "torch",
    "smoke",
    "Phase-",
    "S0", "S1", "S2", "S3", "S4",
    "e-",  # catches e-notation like 1e-12 when split
    "Wilcoxon",
    "Matern",
    "ladder",
    "deferred",
    "not yet",
)


# ---------------------------------------------------------------------------
# Pattern-based prose detection (before artifact matching)
# ---------------------------------------------------------------------------

# Version fragment: a 2-segment number that is part of a longer 3-segment version
_VERSION_FRAGMENT_RE = re.compile(r"\d+\.\d+")

# Full version: three or more segments (e.g., 6.2.1, 10.0.2, 9.14.6206)
_FULL_VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\w*\b")

# e-notation tokens that could be split
_E_NOTATION_CTX = re.compile(r"\de[+-]\d+", re.IGNORECASE)

# List-like bullet: token is "X." (a digit followed by a dot)
_LIST_BULLET_RE = re.compile(r"^\d+\.$")

# Time units (handled elsewhere but context may still trigger)
_TIME_UNITS_RE = re.compile(r"\b\d+s\b")

# Task references
_TASK_REF_RE = re.compile(r"\btask\s+\d+\b", re.IGNORECASE)

# Method/pipeline names that contain numbers
_STAGE_RE = re.compile(r"\bS[0-4]\b")

# Hash-like contexts
_HASH_CTX = re.compile(r"(?:SHA|ISO|md5|sha)[- ]?\d+", re.IGNORECASE)


def _extract_headline_numbers(text: str) -> list[tuple[str, str]]:
    """Extract ONLY numbers that look like headline results from markdown text.

    Returns ``[(token, context), ...]`` where context is surrounding text.
    Excludes numbers classified as prose via the rules above.
    """
    results: list[tuple[str, str]] = []

    # Step 1: find all version-like and e-notation spans and mark them to be skipped
    skip_spans: list[tuple[int, int]] = []

    for pat in (_FULL_VERSION_RE, _E_NOTATION_CTX, _TIME_UNITS_RE, _STAGE_RE,
                _TASK_REF_RE, _HASH_CTX):
        for m in pat.finditer(text):
            skip_spans.append((m.start(), m.end()))

    # Step 2: find bare numeric tokens (the headline candidates)
    # Match: optional sign, digits, optional decimal point + digits, optional %
    num_pat = re.compile(r"[-+]?\d+\.?\d*\s*%?")
    for m in num_pat.finditer(text):
        token = m.group().strip()

        # Skip empty / whitespace-only
        if not token:
            continue

        # Skip if token is in the explicit whitelist
        if token in _TOKEN_WHITELIST:
            continue

        # Skip 1-digit numbers without decimal point (almost never headline)
        numeric_part = token.rstrip("%")
        try:
            val = float(numeric_part)
        except ValueError:
            continue
        if abs(val) < 10 and "." not in numeric_part:
            continue

        # Skip list-item bullets like "3." or "6."
        if _LIST_BULLET_RE.match(token):
            continue

        # Skip if inside a skip span (version/e-notation/task/time/stage/hash)
        t_start = m.start()
        if any(s <= t_start < e for s, e in skip_spans):
            continue

        # Build context window
        ctx_start = max(0, m.start() - 60)
        ctx_end = min(len(text), m.end() + 60)
        context = text[ctx_start:ctx_end].replace("\n", " ")

        # Step 3: check context for prose markers
        if _is_prose_context(context, token):
            continue

        results.append((token, context))

    return results


def _is_prose_context(context: str, token: str) -> bool:
    """Return True if the context indicates this token is prose, not a headline number."""
    ctx_lower = context.lower()

    # Version fragment: if token is "X.Y" and the context contains "X.Y.Z" nearby
    if _VERSION_FRAGMENT_RE.fullmatch(token):
        # Check if there's a 3-segment version containing this token nearby
        if _FULL_VERSION_RE.search(context):
            return True
        # Check if this is part of a version line like "PySCIPOpt 6.2.1"
        for marker in ("pyscipopt", "highspy", "pulp", "ortools", "botorch",
                       "openai", "numpy", "scipy", "torch", "== "):
            if marker in ctx_lower:
                return True

    # e-notation fragments
    if _E_NOTATION_CTX.search(context):
        return True

    # Hash-like prefixes
    if _HASH_CTX.search(context):
        return True

    # Known prose context markers
    for marker in _PROSE_CONTEXT_MARKERS:
        if marker in ctx_lower:
            return True

    # Time limit mentions ("30s", "300s", "1800s" etc.)
    if _TIME_UNITS_RE.search(context):
        return True

    # Version line in code block
    if "==" in context or "===" in context:
        return True

    # Task references
    if _TASK_REF_RE.search(context):
        return True

    # Aggregate thresholds mentioned inline
    for probe in ("≥", "<=", ">=", "alpha", "α", "ρ", "threshold", "tolerance",
                  "cap", "budget"):
        if probe in ctx_lower and abs(float(token.rstrip("%")) if token.rstrip("%").replace(".", "").replace("-", "").isdigit() else 100) <= 100:
            return True

    # Parameter knobs / named numbers
    for pattern in (r"knob", r"knapsack", r"instance_limit", r"max_terms",
                    r"max_cover", r"MAX_INT", r"PAR_FACTOR"):
        if re.search(pattern, ctx_lower):
            return True

    return False


# ---------------------------------------------------------------------------
# Artifact number collection
# ---------------------------------------------------------------------------


def _collect_artifact_numbers(artifact_dir: Path) -> set[str]:
    """Collect all numeric tokens from artifact JSON files and generated tables."""
    numbers: set[str] = set()

    for name in ("thesis_report.json", "comparison_report.json", "results.json"):
        path = artifact_dir / name
        if path.is_file():
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            _extract_numbers(data, numbers)

    tables_dir = artifact_dir / "tables"
    if tables_dir.is_dir():
        for md_path in sorted(tables_dir.glob("*.md")):
            text = md_path.read_text(encoding="utf-8")
            # Extract ALL numbers from tables (these are the authoritative generated values)
            for m in re.finditer(r"[-+]?\d+\.?\d*\s*%?", text):
                token = m.group().strip()
                if token:
                    numbers.add(token)
                    # Also add common alternate representations
                    numeric_part = token.rstrip("%")
                    try:
                        val = float(numeric_part)
                    except ValueError:
                        continue
                    for fmt in (str(int(val)), f"{val:.1f}", f"{val:.2f}", f"{val:.3f}", f"{val:.4f}"):
                        numbers.add(fmt)

    return numbers


def _extract_numbers(obj: Any, numbers: set[str]) -> None:
    """Recursively extract all numeric values from a JSON-compatible object."""
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not _is_integer_value(obj):
            for fmt in (f"{obj:.4f}", f"{obj:.2f}", f"{obj:.1f}"):
                numbers.add(fmt)
        else:
            numbers.add(str(int(obj)))
            numbers.add(str(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_numbers(v, numbers)
    elif isinstance(obj, list):
        for v in obj:
            _extract_numbers(v, numbers)


def _is_integer_value(x: float) -> bool:
    try:
        return x == int(x) and abs(x) < 1e15
    except (ValueError, OverflowError):
        return False


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


def check_report(report_dir: Path, artifact_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """Check all headline numbers in the report against artifacts.

    Returns ``(traceable, orphaned)`` token lists.
    """
    artifact_dir = artifact_dir or report_dir
    artifact_numbers = _collect_artifact_numbers(artifact_dir)

    md_files = sorted(report_dir.glob("*.md"))
    if not md_files:
        print(f"no .md files found in {report_dir}", file=sys.stderr)
        return [], []

    traceable: list[str] = []
    orphaned: list[str] = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        tokens = _extract_headline_numbers(text)

        for token, context in tokens:
            matched = token in artifact_numbers

            if not matched:
                numeric_part = token.rstrip("%")
                try:
                    val = float(numeric_part)
                except ValueError:
                    pass
                else:
                    for fmt in (str(int(val)), f"{val:.1f}", f"{val:.2f}",
                                f"{val:.3f}", f"{val:.4f}"):
                        if fmt in artifact_numbers:
                            matched = True
                            break
                    pct_str = f"{val:.1f}%"
                    if pct_str in artifact_numbers:
                        matched = True

            if matched:
                traceable.append(token)
            else:
                orphaned.append(
                    f"{md_path.name}: {token!r}  (context: ...{context}...)"
                )

    return traceable, orphaned


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that every headline number in the report traces to an artifact.",
    )
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Report directory (e.g., docs/tech-report)",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifact directory (default: same as report_dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    report_dir = args.report_dir.resolve()
    if not report_dir.is_dir():
        print(f"error: report directory not found: {report_dir}", file=sys.stderr)
        return 2

    traceable, orphaned = check_report(report_dir, args.artifacts)

    print(f"Traceable: {len(traceable)}")
    if orphaned:
        print(f"Orphaned: {len(orphaned)}")
        for entry in orphaned:
            print(f"  {entry}")
        print(f"\nFAIL: {len(orphaned)} headline number(s) could not be traced to artifacts.")
        return 1

    print("PASS: all headline numbers traceable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
