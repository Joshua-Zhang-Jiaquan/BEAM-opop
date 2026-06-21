"""OPOP CLI entry point."""

import argparse


def _run_command(args: argparse.Namespace) -> int:
    from opop.run import main as run_main

    return run_main(["--config", args.config, "--out", args.out])


def _add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run", help="execute a CO/IP search run")
    p.add_argument("--config", required=True, help="path to a run config (.yaml/.json)")
    p.add_argument("--out", required=True, help="output run directory")
    p.set_defaults(func=_run_command)


def _add_replay_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("replay", help="replay a previous run")
    p.set_defaults(func=lambda _args: print("replay: not implemented"))


def _add_bench_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("bench", help="run benchmarks")
    p.set_defaults(func=lambda _args: print("bench: not implemented"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="opop", description="OPOP: CO/IP formulation-and-search engine")
    sub = parser.add_subparsers(dest="command")
    _add_run_subparser(sub)
    _add_replay_subparser(sub)
    _add_bench_subparser(sub)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        print("OPOP CLI — use: opop {run,replay,bench}")


if __name__ == "__main__":
    main()
