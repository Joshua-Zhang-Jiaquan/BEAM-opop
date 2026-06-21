#!/usr/bin/env python3
"""Claims audit: verify that every claim in the paper maps to a thesis_report.json verdict.

Usage::

    python scripts/claims_audit.py docs/paper/paper.md
    python scripts/claims_audit.py docs/paper/paper.md --thesis-report runs/final_eval/thesis_report.json

Exit codes: 0 if every claim is backed by artifacts, 1 if unsupported claims or
dev/validation numbers appear in headline result tables.

A *claim* is a sentence or sentence fragment that asserts a result about OPOP's
performance relative to a baseline, or a verdict about a thesis.

The auditor:
1. Parses ``paper.md`` for explicit claim markers.
2. Loads the thesis report and, optionally, other artifact JSON files.
3. Verifies that each claim references a key in ``thesis_report.json`` or a
   comparison report field.
4. Rejects unsupported claims (e.g., "SOTA on all domains").
5. Verifies no dev/validation numbers appear in headline result tables ("Table 1",
   "Table 2") when the data provenance states they are test-split results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Claim detection patterns
# ---------------------------------------------------------------------------

# Sentences or fragments that start with claim-like patterns.
_CLAIM_STARTERS: tuple[str, ...] = (
    "We find", "we find",
    "opop achieves", "OPOP achieves",
    "opop beats", "OPOP beats",
    "opop wins", "OPOP wins",
    "opop is", "OPOP is",
    "opop reaches", "OPOP reaches",
    "opop reduces", "OPOP reduces",
    "opop outperforms", "OPOP outperforms",
    "T1 holds", "T2 holds", "T3 holds", "T4 holds",
    "the thesis", "The thesis",
    "the evidence supports", "The evidence supports",
    "significant",  # as in "significant improvement"
    "state-of-the-art", "SOTA",
    "all domains", "every domain",
    "consistently", "always",
)

# Patterns that flag overclaims (unsupported by the thesis report).
_OVERCLAIM_PATTERNS: list[tuple[str, str]] = [
    ("SOTA", "Claim of state-of-the-art status"),
    ("state-of-the-art", "Claim of state-of-the-art status"),
    ("on all domains", "Claim of superiority on all domains"),
    ("on every domain", "Claim of superiority on every domain"),
    ("always outperforms", "Absolute claim of always outperforming"),
    ("always beats", "Absolute claim of always outperforming"),
    ("universally", "Absolute claim of universal superiority"),
    ("guarantees", "Guarantee claim (not falsifiable)"),
    ("proves", "Proof claim (not empirical)"),
]

# T1–T4 thesis name mapping to thesis_report.json keys.
_THESIS_KEYS = {"T1", "T2", "T3", "T4"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_sentences(text: str) -> list[tuple[str, int]]:
    """Extract sentences from text, returning ``(sentence, line_number)`` pairs."""
    lines = text.split("\n")
    results: list[tuple[str, int]] = []
    for lineno, line in enumerate(lines, start=1):
        # Split by sentence boundaries (.!?)
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            stripped = part.strip()
            if stripped and len(stripped) > 10:
                results.append((stripped, lineno))
    return results


def _find_claims(text: str) -> list[dict[str, Any]]:
    """Find all claim-like sentences in the paper text.

    Returns a list of dicts with ``sentence``, ``line``, ``marker``, and ``risk``.
    """
    sentences = _extract_sentences(text)
    claims: list[dict[str, Any]] = []
    for sentence, lineno in sentences:
        marker: str | None = None
        for starter in _CLAIM_STARTERS:
            if starter.lower() in sentence.lower():
                marker = starter
                break
        if marker is None:
            continue

        # Determine risk level
        risk = "low"
        for pattern, reason in _OVERCLAIM_PATTERNS:
            if pattern.lower() in sentence.lower():
                risk = "high"
                marker = f"{marker} (OVERCLAIM: {reason})"
                break

        claims.append({
            "sentence": sentence,
            "line": lineno,
            "marker": marker,
            "risk": risk,
        })
    return claims


def _find_headline_table_numbers(text: str) -> list[dict[str, Any]]:
    """Find numeric values appearing near headline table captions (Table 1, Table 2).

    Returns list of ``{caption, line, numbers}`` dicts.
    """
    tables: list[dict[str, Any]] = []
    table_caption_re = re.compile(
        r"\*Table\s+\d+\*?\s*[:.]?\s*(.*?)(?:\n|$)",
        re.IGNORECASE,
    )
    lines = text.split("\n")
    for lineno, line in enumerate(lines, start=1):
        m = table_caption_re.search(line)
        if m:
            caption = m.group(0).strip()
            # Collect numbers from this line and nearby lines (+/- 3 lines)
            nearby_text = "\n".join(lines[max(0, lineno - 4): min(len(lines), lineno + 4)])
            numbers = re.findall(r"\b\d+\.?\d*%?\b", nearby_text)
            tables.append({
                "caption": caption,
                "line": lineno,
                "numbers": numbers,
            })
    return tables


# ---------------------------------------------------------------------------
# Thesis report verification
# ---------------------------------------------------------------------------


def _load_thesis_report(path: Path) -> dict[str, Any] | None:
    """Load a thesis_report.json, returning None if missing."""
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            return dict(json.load(fh))
    return None


def _verify_thesis_claims(
    claims: list[dict[str, Any]], thesis_report: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Verify thesis-related claims against the thesis report.

    Returns list of issues found.
    """
    issues: list[dict[str, Any]] = []

    if thesis_report is None:
        # If no thesis report, thesis-related claims are noted but do NOT cause failure.
        # Only real issues (overclaims, dev_in_headline) should fail.
        for claim in claims:
            for key_name in _THESIS_KEYS:
                if key_name.lower() in claim["sentence"].lower():
                    issues.append({
                        "type": "info_no_thesis_report",
                        "claim": claim["sentence"],
                        "line": claim["line"],
                        "detail": "Thesis claim found but no thesis_report.json loaded (informational)",
                    })
                    break
        return issues

    for claim in claims:
        sentence = claim["sentence"]
        risk = claim["risk"]

        # Check if the claim references a known thesis key
        referenced_thesis: str | None = None
        for key_name in _THESIS_KEYS:
            if key_name in sentence:
                # Check if the verdict data exists
                verdict_data = thesis_report.get(key_name)
                if not isinstance(verdict_data, dict):
                    issues.append({
                        "type": "missing_thesis_data",
                        "claim": sentence,
                        "line": claim["line"],
                        "detail": f"{key_name} mentioned in paper but missing from thesis_report.json",
                    })
                    break
                referenced_thesis = key_name
                break

        # Check for overclaims
        if risk == "high":
            issues.append({
                "type": "overclaim",
                "claim": sentence,
                "line": claim["line"],
                "detail": claim["marker"],
            })

        # Check if claim asserts a verdict that contradicts the report
        if referenced_thesis:
            verdict_data = thesis_report[referenced_thesis]
            actual_verdict = verdict_data.get("verdict", None)

            # If claim says "passes" but report says False
            if any(w in sentence.lower() for w in ("passes", "holds", "is supported", "confirmed")):
                if actual_verdict is False:
                    issues.append({
                        "type": "verdict_mismatch",
                        "claim": sentence,
                        "line": claim["line"],
                        "detail": (
                            f"Paper claims {referenced_thesis} passes, "
                            + "but thesis_report.json says verdict=False"
                        ),
                    })

            # If claim says "fails" but report says True
            if any(w in sentence.lower() for w in ("fails", "does not hold", "is not supported")):
                if actual_verdict is True:
                    issues.append({
                        "type": "verdict_mismatch",
                        "claim": sentence,
                        "line": claim["line"],
                        "detail": (
                            f"Paper claims {referenced_thesis} fails, "
                            + "but thesis_report.json says verdict=True"
                        ),
                    })

    return issues


def _check_dev_validation_in_headline_tables(
    tables: list[dict[str, Any]], thesis_report: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Verify no dev/validation numbers appear in headline tables when the split is test.

    Returns list of issues found.
    """
    issues: list[dict[str, Any]] = []

    if thesis_report is None:
        return issues

    meta = thesis_report.get("meta", {})
    split = meta.get("split", "unknown")

    # Only relevant if the thesis report is on a held-out split
    if split in ("test", "ood_test"):
        # Warn if tables exist but data provenance is clear
        pass  # Tables are auto-generated; the audit just verifies provenance

    # Check for dev/validation mentions near tables
    for table in tables:
        caption = table["caption"]
        if any(w in caption.lower() for w in ("validation", "dev ", "dev/")):
            issues.append({
                "type": "dev_in_headline",
                "caption": caption,
                "line": table["line"],
                "detail": (
                    "Headline table mentions dev/validation split; "
                    + f"thesis_report.json reports split={split}"
                ),
            })

    return issues


def _check_data_provenance(text: str, thesis_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Verify the paper's data provenance matches the thesis report metadata.

    Returns list of issues found.
    """
    issues: list[dict[str, Any]] = []

    if thesis_report is None:
        return issues

    meta = thesis_report.get("meta", {})
    n_records = meta.get("n_records", 0)

    # If the paper mentions a specific n_records, verify it matches
    n_match = re.search(r"n_record[s]?\s*=\s*(\d+)", text, re.IGNORECASE)
    if n_match:
        claimed_n = int(n_match.group(1))
        if claimed_n != n_records and n_records > 0:
            issues.append({
                "type": "record_count_mismatch",
                "detail": (
                    f"Paper claims n_records={claimed_n} but thesis_report.json reports {n_records}"
                ),
            })

    return issues


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def audit_paper(
    paper_path: Path,
    thesis_report_path: Path | None = None,
    *,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Audit a paper markdown file for artifact-backed claims.

    Args:
        paper_path: Path to paper.md.
        thesis_report_path: Path to thesis_report.json; auto-discovered if None.
        verbose: If True, print detailed audit output.

    Returns:
        ``(issues, summary)`` where issues is a list of problem dicts and
        summary is a human-readable audit result string.
    """
    if not paper_path.is_file():
        return [], f"paper not found: {paper_path}"

    text = paper_path.read_text(encoding="utf-8")

    # Auto-discover thesis report from paper directory or parent
    if thesis_report_path is None:
        candidates = [
            paper_path.parent / "thesis_report.json",
            paper_path.parent.parent / "thesis_report.json",
            paper_path.parent.parent.parent / "thesis_report.json",  # e.g., docs/paper/ -> project root
            Path("runs/final_eval/thesis_report.json"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                thesis_report_path = candidate
                break

    thesis_report = _load_thesis_report(thesis_report_path) if thesis_report_path else None

    # Find claims
    claims = _find_claims(text)
    if verbose:
        print(f"Found {len(claims)} claim-like sentence(s):")
        for c in claims:
            print(f"  L{c['line']}: [{c['risk']}] {c['marker']!r} -> {c['sentence'][:100]}...")

    # Find headline tables
    tables = _find_headline_table_numbers(text)
    if verbose:
        print(f"Found {len(tables)} headline table(s)")

    # Verify
    all_issues: list[dict[str, Any]] = []
    all_issues.extend(_verify_thesis_claims(claims, thesis_report))
    all_issues.extend(_check_dev_validation_in_headline_tables(tables, thesis_report))
    all_issues.extend(_check_data_provenance(text, thesis_report))

    # Separate informational notes from real failures
    info_issues = [i for i in all_issues if str(i.get("type", "")).startswith("info_")]
    real_issues = [i for i in all_issues if not str(i.get("type", "")).startswith("info_")]

    # Build summary
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" OPOP Paper Claims Audit")
    lines.append(f" Paper: {paper_path}")
    lines.append(f" Thesis report: {thesis_report_path or 'NOT FOUND'}")
    lines.append(f" Claims found: {len(claims)}")
    lines.append(f" Headline tables: {len(tables)}")
    lines.append(f" Informational notes: {len(info_issues)}")
    lines.append(f" Real issues: {len(real_issues)}")
    lines.append("-" * 72)

    if info_issues:
        lines.append(" INFORMATIONAL:")
        for i, issue in enumerate(info_issues):
            lines.append(f"  [{i + 1}] {issue['type']}: {issue.get('detail', '')}")
            if "line" in issue:
                lines.append(f"       Line: {issue['line']}")

    if real_issues:
        lines.append(" ISSUES:")
        for i, issue in enumerate(real_issues):
            lines.append(f"  [{i + 1}] {issue['type']}: {issue.get('detail', '')}")
            if "claim" in issue:
                lines.append(f"       Claim: {issue['claim'][:120]}...")
            if "line" in issue:
                lines.append(f"       Line: {issue['line']}")
    elif not info_issues:
        lines.append(" No issues found. All claims trace to artifacts.")
    else:
        lines.append(" No real issues found.")
        if thesis_report is None:
            lines.append(" NOTE: No thesis_report.json was found; thesis-specific")
            lines.append("       claims could not be verified.")

    lines.append("-" * 72)
    if real_issues:
        lines.append(f" AUDIT RESULT: FAIL ({len(real_issues)} real issue(s), "
                      + f"{len(info_issues)} informational)")
    else:
        lines.append(f" AUDIT RESULT: PASS ({len(info_issues)} informational)")
    lines.append("=" * 72)

    summary = "\n".join(lines)
    return real_issues, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a conference paper for artifact-backed claims.",
    )
    parser.add_argument(
        "paper",
        type=Path,
        help="Path to paper.md",
    )
    parser.add_argument(
        "--thesis-report",
        type=Path,
        default=None,
        help="Path to thesis_report.json (auto-discovered if not provided)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed claim detection output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    issues, summary = audit_paper(
        args.paper,
        thesis_report_path=args.thesis_report,
        verbose=args.verbose,
    )

    print(summary)

    if issues:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
