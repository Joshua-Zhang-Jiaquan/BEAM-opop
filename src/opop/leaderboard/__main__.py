"""CLI entry point: ``python -m opop.leaderboard build|submit``.

Subcommands:

``build``
    Build a static leaderboard site from a run directory.
    ``python -m opop.leaderboard build --results <run_dir> --out site/``

``submit``
    Validate a run directory for leaderboard submission integrity.
    ``python -m opop.leaderboard submit --run <run_dir>``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opop.leaderboard.builder import LeaderboardBuilder
from opop.leaderboard.submit import SubmissionValidator


def _resolve_results_path(run_dir: Path) -> Path:
    """Find the results file in a run directory."""
    for name in ("results.parquet", "results.json", "results.jsonl"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    # If run_dir is itself a results file, use it directly.
    if run_dir.is_file() and run_dir.suffix in (".parquet", ".json", ".jsonl"):
        return run_dir
    raise FileNotFoundError(
        f"no results.parquet/.json/.jsonl found in {run_dir}"
    )


def _resolve_thesis_path(run_dir: Path) -> Path | None:
    """Find thesis_report.json in a run directory (optional)."""
    candidate = run_dir / "thesis_report.json"
    if candidate.is_file():
        return candidate
    return None


def _cmd_build(args: argparse.Namespace) -> int:
    """Handle the ``build`` subcommand."""
    run_dir = Path(args.results)
    try:
        results_path = _resolve_results_path(run_dir)
    except FileNotFoundError as exc:
        print(f"build error: {exc}", file=sys.stderr)
        return 1

    thesis_path = _resolve_thesis_path(run_dir) if run_dir.is_dir() else None

    builder = LeaderboardBuilder(
        results_path,
        thesis_path=thesis_path,
        split=args.split,
    )
    try:
        html_path, md_path = builder.write(args.out)
    except (OSError, ValueError) as exc:
        print(f"build error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {html_path}")
    print(f"wrote {md_path}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    """Handle the ``submit`` subcommand."""
    validator = SubmissionValidator(registry_path=args.registry)
    result = validator.validate(args.run)

    status = "ACCEPTED" if result.accepted else "REJECTED"
    print(f"submission {status}: {result.reason}")
    print(f"  run_dir: {result.run_dir}")
    print(f"  checked: {', '.join(result.artifacts_checked)}")
    print(f"  found:   {', '.join(result.artifacts_found)}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {out_path}")

    return 0 if result.accepted else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m opop.leaderboard",
        description="OPOP leaderboard builder and submission validator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = sub.add_parser("build", help="Build a static leaderboard site.")
    build_p.add_argument(
        "--results",
        required=True,
        help="run directory (containing results.parquet) or direct results file",
    )
    build_p.add_argument(
        "--out",
        default="site",
        help="output directory for the static site (default: site/)",
    )
    build_p.add_argument(
        "--split",
        default=None,
        help="explicit split label (overrides auto-inference from records)",
    )

    # submit
    submit_p = sub.add_parser("submit", help="Validate a run for leaderboard submission.")
    submit_p.add_argument(
        "--run",
        required=True,
        help="run directory to validate",
    )
    submit_p.add_argument(
        "--registry",
        default=None,
        help="path to registry.yaml for lock verification",
    )
    submit_p.add_argument(
        "--json-out",
        default=None,
        help="optional path to write the validation result as JSON",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatch."""
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "submit":
        return _cmd_submit(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
