"""Replay a persisted Phase-1 closed loop from its reproducibility artifacts.

``python -m opop.replay --run <dir> [--strict]`` re-executes the Phase-1 closed
loop recorded in ``<dir>`` entirely from disk:

* :func:`opop.orchestrator.repro.load_manifest` loads ``repro_manifest.json``
  (the full determinism fingerprint: seeds, the :class:`~opop.config.RunConfig`
  snapshot, tolerances, ...);
* :func:`opop.orchestrator.repro.read_instance` loads the working MILP IR from
  ``instance.json``;
* the recorded seeds are re-applied (:func:`set_seeds`), the real Phase-1
  objects are rebuilt (controller / analyzer / proposer / verifier / evaluator /
  SCIP kernel), and :func:`opop.orchestrator.loop.run_loop` is driven into a
  ``replay/`` sub-directory so the original artifacts are never clobbered.

With ``--strict`` the replay is *verified*: the replayed incumbent objective is
compared with the original ``incumbent.json`` (to ``1e-9``) and the replayed
``n_accepted`` with the original ``result.json``. A match prints ``REPRODUCED``
and exits ``0``; any divergence prints a mismatch report and exits ``1``.

Replaying a recorded run is a real solve, so a missing SCIP backend
(``pyscipopt``) is a loud :class:`ReplayError`, never a silent no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from opop.analyzer.api import analyze
from opop.controller.encoder import default_phase1_space
from opop.evaluator import evaluate
from opop.model.state import ProblemState
from opop.orchestrator.loop import run_loop
from opop.orchestrator.repro import config_from_dict, load_manifest, read_instance
from opop.proposer.api import propose
from opop.solver.scip import ScipKernel
from opop.verify.gate import verify_delta

if TYPE_CHECKING:
    from opop.orchestrator.result import RunResult

__all__ = ["OBJECTIVE_TOLERANCE", "ReplayError", "main", "replay_run", "set_seeds"]

#: Objective-equality tolerance for the strict reproduction check.
OBJECTIVE_TOLERANCE: float = 1e-9
#: Sub-directory (under the run dir) the replay writes its artifacts to.
REPLAY_SUBDIR: str = "replay"


class ReplayError(RuntimeError):
    """Raised when a run cannot be replayed (e.g. the SCIP backend is absent)."""


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seeds(seeds: dict[str, Any]) -> None:
    """Re-apply every RNG seed recorded in the manifest's ``seeds`` mapping.

    Seeds the Python :mod:`random` module, NumPy's legacy global RNG, and — only
    when :mod:`torch` is importable — the torch CPU RNG. ``torch`` is an optional
    dependency, so a missing import is tolerated (its seed is simply not applied).
    """
    random.seed(int(seeds["python_random"]))
    np.random.seed(int(seeds["numpy"]))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seeds["torch"]))


# ---------------------------------------------------------------------------
# SCIP kernel (real-run replay requires it)
# ---------------------------------------------------------------------------
def _build_scip_kernel() -> ScipKernel:
    """Return a real :class:`~opop.solver.scip.ScipKernel`, or fail loudly.

    Importing :class:`ScipKernel` is solver-free, so the actual SCIP backend
    (``pyscipopt``) is probed explicitly here. Replaying a recorded run is a real
    solve, so a missing backend is a hard :class:`ReplayError` rather than a
    silent degradation.
    """
    if importlib.util.find_spec("pyscipopt") is None:
        raise ReplayError(
            "replay requires the SCIP backend (pyscipopt), which is not "
            + "installed; a recorded run cannot be re-executed without it"
        )
    return ScipKernel()


# ---------------------------------------------------------------------------
# JSON helpers / strict comparison
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    """Load a JSON artifact, returning ``None`` when the file is absent."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _incumbent_objective(incumbent: Any) -> float:
    """Extract ``score.objective`` from an incumbent dict (absent -> ``NaN``).

    The persisted ``incumbent.json`` JSON-sanitises non-finite metrics to
    ``null`` (``None``), so a missing / null objective maps to ``NaN`` — the
    canonical "no objective" marker used by the strict comparison.
    """
    if not isinstance(incumbent, dict):
        return math.nan
    inc: dict[str, Any] = incumbent
    score = inc.get("score")
    if not isinstance(score, dict):
        return math.nan
    metrics: dict[str, Any] = score
    objective = metrics.get("objective")
    if objective is None:
        return math.nan
    return float(objective)


def _objectives_match(
    original: float, replayed: float, *, tol: float = OBJECTIVE_TOLERANCE
) -> bool:
    """Compare two (possibly ``NaN``) objectives; two ``NaN``s count as equal."""
    if math.isnan(original) and math.isnan(replayed):
        return True
    if math.isnan(original) or math.isnan(replayed):
        return False
    return abs(original - replayed) <= tol


def _verify_strict(run_dir: Path, replay_dir: Path, result: RunResult) -> int:
    """Compare the replay against the original artifacts; print + return a code.

    A reproduction (objective within ``tol`` AND identical ``n_accepted``) prints
    ``REPRODUCED`` and returns ``0``; any divergence prints a diff and returns
    ``1``. Both objectives are read from their respective ``incumbent.json`` so
    they pass through the identical JSON-sanitisation path.
    """
    original_objective = _incumbent_objective(_load_json(run_dir / "incumbent.json"))
    replay_objective = _incumbent_objective(_load_json(replay_dir / "incumbent.json"))
    objective_ok = _objectives_match(original_objective, replay_objective)

    original_result = _load_json(run_dir / "result.json")
    original_n_accepted = (
        original_result.get("n_accepted") if isinstance(original_result, dict) else None
    )
    n_accepted_ok = original_n_accepted == result.n_accepted

    if objective_ok and n_accepted_ok:
        print("REPRODUCED")
        return 0

    o_obj, r_obj = original_objective, replay_objective
    o_acc, r_acc = original_n_accepted, result.n_accepted
    print("MISMATCH: replay diverged from the recorded run")
    print(f"  objective: original={o_obj} replay={r_obj} (match={objective_ok})")
    print(f"  n_accepted: original={o_acc} replay={r_acc} (match={n_accepted_ok})")
    return 1


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------
def replay_run(run_dir: str | Path, *, strict: bool = False) -> int:
    """Re-execute the Phase-1 loop recorded in ``run_dir``; return an exit code.

    Loads the manifest + IR, re-applies the recorded seeds, rebuilds the real
    Phase-1 objects, and drives :func:`~opop.orchestrator.loop.run_loop` into a
    ``replay/`` sub-directory of ``run_dir``. In ``strict`` mode the replay is
    verified against the original artifacts (see :func:`_verify_strict`),
    returning ``0`` (reproduced) or ``1`` (mismatch); otherwise it returns ``0``
    once the replay completes.
    """
    # Deferred so ``import opop.replay`` stays torch-free; the GP controller's torch
    # dependency ships in the `bo` extra and is only needed for the actual replay.
    from opop.controller.phase1 import Phase1Controller

    run_path = Path(run_dir)
    manifest = load_manifest(run_path)
    ir = read_instance(run_path)
    config = config_from_dict(manifest["config"])
    seeds = manifest["seeds"]

    set_seeds(seeds)
    scip_seed = int(seeds["scip"])

    # ``run_loop`` derives its solver seed from ``config.seeds[0]``; pin it to the
    # manifest's SCIP seed so the replayed solve uses exactly that seed.
    config = replace(config, seeds=[scip_seed, *config.seeds[1:]])

    n_trials = int(config.budget.trials)
    controller = Phase1Controller.bo(
        default_phase1_space(),
        n_trials=n_trials,
        n_init=min(3, n_trials),
        n_candidates=64,
        time_budget_s=None,
        seed=scip_seed,
    )

    kernel = _build_scip_kernel()
    state = ProblemState(
        instance_id=ir.name,
        task_family="MILP",
        budget_state={"ir": ir},
    )

    replay_dir = run_path / REPLAY_SUBDIR
    result = run_loop(
        state,
        config,
        kernel=kernel,
        proposer=propose,
        analyzer=analyze,
        verifier=verify_delta,
        evaluator=evaluate,
        controller=controller,
        out_dir=replay_dir,
        reference_optimum=manifest.get("reference_optimum"),
        time_budget_s=None,
        instance_id=ir.name,
    )

    if strict:
        return _verify_strict(run_path, replay_dir, result)

    objective = _incumbent_objective(_load_json(replay_dir / "incumbent.json"))
    n_it, n_acc = result.n_iterations, result.n_accepted
    print(f"replay complete (artifacts in {replay_dir})")
    print(f"  iterations={n_it} accepted={n_acc} incumbent_objective={objective}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse ``--run`` / ``--strict`` and replay the run; return an exit code."""
    parser = argparse.ArgumentParser(
        prog="opop.replay",
        description="Re-execute a recorded Phase-1 closed loop from its manifest.",
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="run directory containing repro_manifest.json + instance.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="verify the replay reproduces the original incumbent + n_accepted",
    )
    args = parser.parse_args(argv)
    return replay_run(args.run, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
