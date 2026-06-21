"""``python -m opop.eval.fidelity_correlation`` — the fidelity-correlation GATE CLI.

Runs a Spearman-ρ correlation study between a cheap *low-fidelity* ranking and
the expensive *high-fidelity* (``full_solve``) ranking of a handful of candidate
configurations, then decides whether cost-aware MFKG may be enabled
(ρ ≥ 0.5) or whether the negative result is recorded and single-fidelity is
kept.

Usage::

    python -m opop.eval.fidelity_correlation \\
        --config configs/phase1_smoke.yaml \\
        --out .omo/evidence/task-29-mfgate.txt

The command loads the run config (for the time budget + seed), materialises a
small set of dev-split synthetic instances (offline; falls back to freshly
generated set-cover instances when the dev set is empty), samples ``--n-methods``
configurations from the canonical Phase-1 space, scores each at the low and high
fidelity layers with the real SCIP kernel via
:func:`opop.controller.fidelity.fidelity_solve` (synthetic deterministic scoring
when no solver is available), and writes a JSON report to ``--out`` **and** to a
sibling ``fidelity_correlation.json``.

The JSON carries the :class:`~opop.controller.fidelity.FidelityCorrelationReport`
fields plus an ``mfkg`` section describing the gate decision, BoTorch
availability, and the cost-aware-MFKG controller configuration that *would* be
used at the target fidelity — never enabling MFKG unless ρ ≥ 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from opop.config import load_config
from opop.controller.encoder import default_phase1_space
from opop.controller.fidelity import (
    MFKG_RHO_THRESHOLD,
    FidelityCorrelationReport,
    FidelityKernel,
    MFKGController,
    fidelity_column,
    fidelity_correlation,
    fidelity_phase1_space,
    fidelity_solve,
    mfkg_available,
    normalized_fidelity,
    resolve_layer,
)
from opop.model.state import Phi

if TYPE_CHECKING:
    from opop.config import RunConfig
    from opop.model.ir import MILP

__all__ = ["StudyResult", "main", "run_study"]

#: Per-solve memory ceiling (MiB); mirrors ``opop.run``.
MEMORY_LIMIT_MB: int = 4096
#: Default low / high fidelity layers for the study.
DEFAULT_LOW = "short_time"
DEFAULT_HIGH = "full_solve"


@dataclass(frozen=True, slots=True)
class StudyResult:
    """Container for a completed fidelity-correlation study.

    Attributes:
        report: The :class:`FidelityCorrelationReport` (ρ + gate decision).
        details: A JSON-serialisable dict of study metadata + per-method scores
            + the ``mfkg`` controller section.
    """

    report: FidelityCorrelationReport
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Merge the report fields with the study/mfkg details."""
        merged = dict(self.report.to_dict())
        merged.update(self.details)
        return merged

    def to_json(self) -> str:
        """Pretty, key-sorted JSON (``nan`` allowed for an undefined ρ)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=True)


# ── instance + method construction ───────────────────────────────────────────


def _study_instances(config: RunConfig, n_instances: int) -> list[MILP]:
    """Return up to ``n_instances`` offline instances for the study.

    Prefers the sealed dev-split synthetic instances; falls back to freshly
    generated set-cover instances when the dev set cannot be loaded or is empty.
    """
    instances: list[MILP] = []
    try:
        from opop.bench.sources.phase1_set import get_phase1_instances

        instances = get_phase1_instances(config.split, sources=("synthetic",))
    except Exception:  # pragma: no cover - registry/IO issues → synthetic fallback
        instances = []
    if not instances:
        from opop.bench.sources.synthetic import generate_set_cover

        instances = [
            generate_set_cover(n_rows=18, n_cols=36, density=0.3, seed=seed)
            for seed in range(max(1, n_instances))
        ]
    return instances[:n_instances]


def _sample_methods(n_methods: int, seed: int) -> list[tuple[str, Phi]]:
    """Sample ``n_methods`` distinct configurations from the Phase-1 space."""
    space = default_phase1_space()
    rng = np.random.default_rng(seed)
    methods: list[tuple[str, Phi]] = []
    seen: set[str] = set()
    attempts = 0
    while len(methods) < n_methods and attempts < n_methods * 50:
        attempts += 1
        phi = space.decode(space.sample_vector(rng))
        key = json.dumps(phi.to_flat_dict(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        methods.append((f"m{len(methods)}", phi))
    return methods


# ── scoring (real solver; deterministic synthetic fallback) ──────────────────


def _make_kernel() -> FidelityKernel | None:
    """Return a SCIP kernel if available, else ``None`` (synthetic scoring)."""
    try:
        from opop.solver.availability import is_solver_available

        if not is_solver_available("scip"):
            return None
        from opop.solver.scip import ScipKernel

        return ScipKernel()
    except Exception:  # pragma: no cover - import/availability issues
        return None


def _synthetic_score(phi: Phi, layer_value: str, instance_id: str, fidelity_norm: float) -> float:
    """Deterministic correlated proxy score (used only when no solver exists).

    The low- and high-fidelity scores share a common per-(method, instance)
    "truth" with a small fidelity-dependent perturbation, so the proxy is
    naturally rank-correlated (documented fallback; the real path uses SCIP).
    """
    blob = json.dumps(phi.to_flat_dict(), sort_keys=True, default=str) + "|" + instance_id
    truth = (abs(hash(blob)) % 1000) / 1000.0
    noise = ((abs(hash(blob + layer_value)) % 100) / 100.0 - 0.5) * (1.0 - fidelity_norm) * 0.2
    return -(truth + noise)


def _score_method(
    kernel: FidelityKernel | None,
    instances: list[MILP],
    phi: Phi,
    layer: str,
    *,
    time_limit: float,
    seed: int,
) -> float:
    """Mean scalarized reward (higher is better) of ``phi`` at ``layer``."""
    resolved = resolve_layer(layer)
    rewards: list[float] = []
    if kernel is None:
        for ir in instances:
            rewards.append(
                _synthetic_score(phi, layer, ir.name, normalized_fidelity(resolved))
            )
        return float(np.mean(rewards)) if rewards else 0.0

    from opop.evaluator import evaluate, scalarize

    for ir in instances:
        trace = fidelity_solve(
            kernel,
            ir,
            phi,
            full_time_limit=time_limit,
            memory_limit_mb=MEMORY_LIMIT_MB,
            seed=seed,
            layer=resolved,
        )
        rewards.append(scalarize(evaluate(trace, time_limit=time_limit)))
    return float(np.mean(rewards)) if rewards else 0.0


# ── study driver ──────────────────────────────────────────────────────────────


def run_study(
    config: RunConfig,
    *,
    n_methods: int = 6,
    n_instances: int = 3,
    low: str = DEFAULT_LOW,
    high: str = DEFAULT_HIGH,
    threshold: float = MFKG_RHO_THRESHOLD,
    seed: int | None = None,
    time_limit: float | None = None,
) -> StudyResult:
    """Run the low-vs-high fidelity correlation study and build the gate report.

    Args:
        config: The loaded run config (supplies the default seed + time budget).
        n_methods: Number of candidate configurations to rank.
        n_instances: Number of dev instances to aggregate each score over.
        low / high: Fidelity layer names (``high`` is the target fidelity).
        threshold: ρ gate for enabling MFKG.
        seed: Study seed (defaults to ``config.seeds[0]``).
        time_limit: Full-solve time budget (defaults to ``config.budget``).

    Returns:
        A :class:`StudyResult` with the report + serialisable details.
    """
    eff_seed = int(config.seeds[0]) if seed is None and config.seeds else int(seed or 0)
    eff_tl = float(config.budget.time_limit_sec) if time_limit is None else float(time_limit)
    low_layer = resolve_layer(low)
    high_layer = resolve_layer(high)

    kernel = _make_kernel()
    instances = _study_instances(config, n_instances)
    methods = _sample_methods(n_methods, eff_seed)

    dev_results: dict[str, dict[str, float]] = {}
    for name, phi in methods:
        dev_results[name] = {
            low_layer.value: _score_method(
                kernel, instances, phi, low_layer.value, time_limit=eff_tl, seed=eff_seed
            ),
            high_layer.value: _score_method(
                kernel, instances, phi, high_layer.value, time_limit=eff_tl, seed=eff_seed
            ),
        }

    report = fidelity_correlation(
        dev_results, low=low_layer, high=high_layer, threshold=threshold
    )

    space = fidelity_phase1_space()
    controller = MFKGController.from_correlation(space, report)
    mfkg_section: dict[str, Any] = {
        "botorch_available": mfkg_available(),
        "controller_enabled": controller.enabled,
        "fidelity_column": fidelity_column(space),
        "encoded_dim": space.dim,
        "target_fidelity": 1.0,
        "fixed_cost": controller.fixed_cost,
        "num_fantasies": controller.num_fantasies,
        "decision": (
            "enable cost-aware MFKG (qMultiFidelityKnowledgeGradient + "
            + "AffineFidelityCostModel + InverseCostWeightedUtility)"
            if controller.enabled
            else "keep single-fidelity (MFKG gated off; negative result recorded)"
        ),
    }
    details: dict[str, Any] = {
        "config_name": config.name,
        "split": config.split,
        "seed": eff_seed,
        "time_limit_sec": eff_tl,
        "n_instances": len(instances),
        "instance_ids": [ir.name for ir in instances],
        "n_methods": len(methods),
        "solver": "scip" if kernel is not None else "synthetic",
        "per_method_scores": dev_results,
        "mfkg": mfkg_section,
    }
    return StudyResult(report, details)


# ── reporting + CLI ───────────────────────────────────────────────────────────


def _format_summary(result: StudyResult) -> str:
    """Render a short human-readable summary of the gate decision."""
    rep = result.report
    mfkg = result.details["mfkg"]
    lines = [
        "=" * 64,
        " fidelity-correlation GATE (low vs high fidelity)",
        "-" * 64,
        f"  low → high          : {rep.low_layer} → {rep.high_layer}",
        f"  methods (paired)    : {rep.n_methods}",
        f"  solver              : {result.details['solver']}",
        f"  Spearman ρ          : {rep.rho:.4f}  (p={rep.p_value:.4g})",
        f"  threshold           : {rep.threshold:g}",
        f"  enable MFKG         : {rep.enable_mfkg}",
        f"  botorch available   : {mfkg['botorch_available']}",
        "-" * 64,
        f"  {rep.reason}",
        f"  MFKG: {mfkg['decision']}",
        "=" * 64,
    ]
    return "\n".join(lines)


def _write_outputs(result: StudyResult, out: Path) -> list[Path]:
    """Write the JSON report to ``out`` and a sibling ``fidelity_correlation.json``."""
    payload = result.to_json() + "\n"
    written: list[Path] = []
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    written.append(out)
    canonical = out.parent / "fidelity_correlation.json"
    if canonical != out:
        canonical.write_text(payload, encoding="utf-8")
        written.append(canonical)
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opop.eval.fidelity_correlation",
        description=(
            "Spearman-ρ fidelity-correlation study + cost-aware MFKG gate "
            "(ρ ≥ 0.5 to enable)."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="run config (.yaml/.json)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fidelity_correlation.json"),
        help="output JSON path (also writes a sibling fidelity_correlation.json)",
    )
    parser.add_argument("--n-methods", type=int, default=6, help="configurations to rank")
    parser.add_argument("--instances", type=int, default=3, help="dev instances per score")
    parser.add_argument("--low", default=DEFAULT_LOW, help="low-fidelity layer name")
    parser.add_argument("--high", default=DEFAULT_HIGH, help="high (target) fidelity layer name")
    parser.add_argument(
        "--threshold", type=float, default=MFKG_RHO_THRESHOLD, help="ρ gate to enable MFKG"
    )
    parser.add_argument("--seed", type=int, default=None, help="study seed (default config seed)")
    parser.add_argument(
        "--time-limit", type=float, default=None, help="full-solve seconds (default config budget)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the study, print a summary, write the JSON report."""
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_study(
            config,
            n_methods=args.n_methods,
            n_instances=args.instances,
            low=args.low,
            high=args.high,
            threshold=args.threshold,
            seed=args.seed,
            time_limit=args.time_limit,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"fidelity_correlation failed: {exc}", file=sys.stderr)
        return 1
    print(_format_summary(result))
    written = _write_outputs(result, args.out)
    print("\nwrote " + ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
