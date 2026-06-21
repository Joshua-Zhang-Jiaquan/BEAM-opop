"""Reproducibility manifest for the Phase-1 closed loop.

:func:`write_manifest` / :func:`finalize_run` persist ``repro_manifest.json`` per
run — the full determinism fingerprint required to reproduce an experimental run:

* ``git_commit`` (best-effort ``git rev-parse HEAD``; ``""`` outside a git repo —
  this project is NOT a git repo, so it is normally empty) and ``container_digest``;
* ``python_version`` (``sys.version``) and ``platform``;
* ``solver_versions`` (SCIP / CP-SAT / HiGHS / CBC via
  :func:`opop.solver.availability.available_solvers`);
* ``hardware`` (``os.cpu_count`` + :mod:`platform`);
* ``threads`` (pinned to ``1`` — never a non-deterministic default);
* ``seeds`` — ALL of them: SCIP, numpy, torch, Python ``random``, and LLM sampling;
* ``time_limit`` / ``memory_limit``;
* ``tolerances`` (feasibility ``1e-7``, optimality ``1e-6``).

A run is NEVER allowed to complete without a complete manifest:
:func:`validate_manifest` raises :class:`MissingManifestFieldError` if any required
field is missing (or empty, except the legitimately-empty ``git_commit`` /
``container_digest``), and :func:`finalize_run` validates before the run summary is
written.

The manifest also embeds a snapshot of the :class:`~opop.config.RunConfig`, and
:func:`finalize_run` persists the working IR to ``instance.json`` so
:mod:`opop.replay` can re-execute the run entirely from disk.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opop.config import BudgetConfig, ControllerConfig, RunConfig, SolverConfig
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)

__all__ = [
    "FEASIBILITY_TOLERANCE",
    "INSTANCE_FILENAME",
    "MANIFEST_FILENAME",
    "OPTIMALITY_TOLERANCE",
    "REQUIRED_FIELDS",
    "THREADS",
    "MissingManifestFieldError",
    "build_manifest",
    "config_from_dict",
    "config_to_dict",
    "default_tolerances",
    "finalize_run",
    "git_commit",
    "host_hardware",
    "ir_from_dict",
    "ir_to_dict",
    "load_manifest",
    "read_instance",
    "seed_mapping",
    "solver_versions",
    "validate_manifest",
    "write_instance",
    "write_manifest",
]

#: Feasibility tolerance recorded in every manifest (spec-pinned).
FEASIBILITY_TOLERANCE: float = 1e-7
#: Optimality tolerance recorded in every manifest (spec-pinned); replay uses it.
OPTIMALITY_TOLERANCE: float = 1e-6
#: Threads are pinned to 1 for determinism — never a non-deterministic default.
THREADS: int = 1
#: Default per-solve memory ceiling (MiB); mirrors ``run_loop``'s default.
DEFAULT_MEMORY_LIMIT_MB: int = 4096

MANIFEST_FILENAME: str = "repro_manifest.json"
INSTANCE_FILENAME: str = "instance.json"

#: Top-level keys every manifest MUST carry.
REQUIRED_FIELDS: tuple[str, ...] = (
    "git_commit",
    "container_digest",
    "python_version",
    "platform",
    "solver_versions",
    "hardware",
    "threads",
    "seeds",
    "time_limit",
    "memory_limit",
    "tolerances",
    "config",
)

#: Fields that may legitimately be empty (no git repo / no container).
_NONEMPTY_EXEMPT: frozenset[str] = frozenset({"git_commit", "container_digest"})

#: Seed sub-keys every manifest's ``seeds`` mapping MUST carry.
REQUIRED_SEED_KEYS: tuple[str, ...] = (
    "scip",
    "numpy",
    "torch",
    "python_random",
    "llm_sampling",
)

#: Tolerance sub-keys every manifest's ``tolerances`` mapping MUST carry.
REQUIRED_TOLERANCE_KEYS: tuple[str, ...] = ("feasibility", "optimality")


class MissingManifestFieldError(RuntimeError):
    """Raised when a reproducibility manifest is missing a required field.

    Finalising a run with an incomplete manifest aborts with this error so that
    no experimental run can ever complete without a full determinism record.
    """


# ---------------------------------------------------------------------------
# Field collectors
# ---------------------------------------------------------------------------
def git_commit() -> str:
    """Return ``git rev-parse HEAD`` (best-effort); ``""`` if not a git repo.

    All failures (git missing, not a repository, timeout) collapse to ``""`` —
    this project is NOT a git repo, so the recorded commit is normally empty.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def container_digest() -> str:
    """Return the container image digest if discoverable, else ``""``.

    Best-effort: reads ``OPOP_CONTAINER_DIGEST`` (set by the launcher when a
    pinned image is used). Empty is a valid value (bare-metal / venv run).
    """
    return os.environ.get("OPOP_CONTAINER_DIGEST", "").strip()


def host_hardware() -> dict[str, Any]:
    """Return a hardware fingerprint: CPU count + platform identifiers."""
    return {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
    }


def solver_versions() -> list[dict[str, Any]]:
    """Return ``[{name, version, available, detail}, ...]`` for every solver.

    Delegates to :func:`opop.solver.availability.available_solvers` (imported
    lazily so importing this module stays solver-free).
    """
    from opop.solver.availability import available_solvers

    return [dict(info) for info in available_solvers()]


def default_tolerances() -> dict[str, float]:
    """Return the spec-pinned tolerance pair (feasibility / optimality)."""
    return {"feasibility": FEASIBILITY_TOLERANCE, "optimality": OPTIMALITY_TOLERANCE}


def _master_seed(config: RunConfig) -> int:
    """The master seed = ``config.seeds[0]`` (or ``0`` when no seeds declared)."""
    return int(config.seeds[0]) if config.seeds else 0


def seed_mapping(
    config: RunConfig,
    seeds: Sequence[int] | Mapping[str, Any] | None = None,
    *,
    llm_sampling: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical ALL-seeds mapping for the manifest.

    A single master seed drives SCIP, numpy, torch, and Python ``random`` in
    Phase-1 (one knob → one reproducible run). ``seeds`` may be an explicit
    per-stream mapping or a sequence whose first entry is the master seed; if
    omitted it is taken from ``config.seeds``. ``llm_sampling`` overrides the
    deterministic default (``temperature=0``) recorded for the LLM channel.
    """
    explicit: dict[str, Any] = {}
    master = _master_seed(config)
    if isinstance(seeds, Mapping):
        explicit = dict(seeds)
        if "scip" in explicit:
            master = int(explicit["scip"])
    elif seeds is not None:
        seq = list(seeds)
        if seq:
            master = int(seq[0])

    mapping: dict[str, Any] = {
        "scip": int(explicit.get("scip", master)),
        "numpy": int(explicit.get("numpy", master)),
        "torch": int(explicit.get("torch", master)),
        "python_random": int(explicit.get("python_random", master)),
    }
    sampling: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "seed": master}
    raw_sampling = explicit.get("llm_sampling")
    if isinstance(raw_sampling, Mapping):
        sampling.update(raw_sampling)
    if isinstance(llm_sampling, Mapping):
        sampling.update(llm_sampling)
    mapping["llm_sampling"] = sampling
    return mapping


# ---------------------------------------------------------------------------
# Config snapshot (round-trips through the manifest)
# ---------------------------------------------------------------------------
def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Serialise a :class:`~opop.config.RunConfig` to a plain JSON-safe dict."""
    return dataclasses.asdict(config)


def config_from_dict(data: Mapping[str, Any]) -> RunConfig:
    """Reconstruct a :class:`~opop.config.RunConfig` from :func:`config_to_dict`."""
    raw = dict(data)
    solver = SolverConfig(**dict(raw.pop("solver", {})))
    controller = ControllerConfig(**dict(raw.pop("controller", {})))
    budget = BudgetConfig(**dict(raw.pop("budget", {})))
    return RunConfig(solver=solver, controller=controller, budget=budget, **raw)


# ---------------------------------------------------------------------------
# IR snapshot (instance.json) — JSON with inf-aware numeric encoding
# ---------------------------------------------------------------------------
def _encode_number(x: float) -> float | str:
    """Map a float to JSON (``inf`` / ``-inf`` / ``nan`` -> string tokens)."""
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "nan"
    return float(x)


def _decode_number(x: float | int | str) -> float:
    """Inverse of :func:`_encode_number`."""
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}.get(x, float(x))
    return float(x)


def _encode_coeffs(coeffs: Mapping[str, float]) -> dict[str, float | str]:
    return {str(k): _encode_number(float(v)) for k, v in coeffs.items()}


def _decode_coeffs(coeffs: Mapping[str, Any]) -> dict[str, float]:
    return {str(k): _decode_number(v) for k, v in coeffs.items()}


def ir_to_dict(ir: MILP) -> dict[str, Any]:
    """Serialise a :class:`~opop.model.ir.MILP` to a JSON-safe dict.

    Captures the full math model plus ``name`` / ``index_sets`` / ``metadata``
    so :func:`ir_from_dict` reconstructs an equivalent IR.
    """
    return {
        "name": ir.name,
        "variables": [
            {
                "name": v.name,
                "vtype": v.vtype.value,
                "lower": _encode_number(v.lower),
                "upper": _encode_number(v.upper),
            }
            for v in ir.variables
        ],
        "constraints": [
            {
                "name": c.name,
                "coeffs": _encode_coeffs(c.coeffs),
                "sense": c.sense.value,
                "rhs": _encode_number(c.rhs),
            }
            for c in ir.constraints
        ],
        "objective": {
            "coeffs": _encode_coeffs(ir.objective.coeffs),
            "sense": ir.objective.sense.value,
            "offset": _encode_number(ir.objective.offset),
        },
        "index_sets": {k: list(v) for k, v in ir.index_sets.items()},
        "metadata": dict(ir.metadata),
    }


def ir_from_dict(data: Mapping[str, Any]) -> MILP:
    """Reconstruct a :class:`~opop.model.ir.MILP` from :func:`ir_to_dict`."""
    variables = tuple(
        Variable(
            name=str(v["name"]),
            vtype=VarType(v["vtype"]),
            lower=_decode_number(v["lower"]),
            upper=_decode_number(v["upper"]),
        )
        for v in data["variables"]
    )
    constraints = tuple(
        LinearConstraint(
            name=str(c["name"]),
            coeffs=_decode_coeffs(c["coeffs"]),
            sense=ConstraintSense(c["sense"]),
            rhs=_decode_number(c["rhs"]),
        )
        for c in data["constraints"]
    )
    obj = data["objective"]
    objective = Objective(
        coeffs=_decode_coeffs(obj["coeffs"]),
        sense=ObjSense(obj["sense"]),
        offset=_decode_number(obj["offset"]),
    )
    index_sets = {str(k): tuple(str(x) for x in v) for k, v in data.get("index_sets", {}).items()}
    return MILP(
        name=str(data.get("name", "")),
        variables=variables,
        constraints=constraints,
        objective=objective,
        index_sets=index_sets,
        metadata=dict(data.get("metadata", {})),
    )


def write_instance(out_dir: str | Path, ir: MILP) -> Path:
    """Persist ``ir`` to ``<out_dir>/instance.json`` (for strict replay)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / INSTANCE_FILENAME
    text = json.dumps(ir_to_dict(ir), allow_nan=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_instance(run_dir: str | Path) -> MILP:
    """Load the persisted IR from ``<run_dir>/instance.json``."""
    path = Path(run_dir) / INSTANCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no persisted instance to replay: {path}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"instance.json must be a JSON object, got {type(data).__name__}")
    return ir_from_dict(data)


# ---------------------------------------------------------------------------
# Manifest build / validate / write
# ---------------------------------------------------------------------------
def build_manifest(
    *,
    config: RunConfig,
    seeds: Sequence[int] | Mapping[str, Any] | None = None,
    solver_versions: list[dict[str, Any]] | None = None,
    hardware: Mapping[str, Any] | None = None,
    tolerances: Mapping[str, float] | None = None,
    time_limit: float | None = None,
    memory_limit: int | None = None,
    container_digest: str | None = None,
    git_commit: str | None = None,
    llm_sampling: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the full reproducibility manifest dict (not yet validated).

    Every argument auto-derives from the environment / ``config`` when omitted,
    so the canonical call is ``build_manifest(config=config)``. Unknown keyword
    arguments are recorded verbatim (e.g. ``reference_optimum``) as non-required
    extras.
    """
    manifest: dict[str, Any] = {
        "git_commit": _git_commit() if git_commit is None else git_commit,
        "container_digest": _container_digest() if container_digest is None else container_digest,
        "python_version": sys.version,
        "platform": platform.platform(),
        "solver_versions": _solver_versions() if solver_versions is None else solver_versions,
        "hardware": dict(host_hardware() if hardware is None else hardware),
        "threads": THREADS,
        "seeds": seed_mapping(config, seeds, llm_sampling=llm_sampling),
        "time_limit": (
            float(config.budget.time_limit_sec) if time_limit is None else float(time_limit)
        ),
        "memory_limit": (DEFAULT_MEMORY_LIMIT_MB if memory_limit is None else int(memory_limit)),
        "tolerances": dict(default_tolerances() if tolerances is None else tolerances),
        "config": config_to_dict(config),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for key, value in extra.items():
        manifest[key] = value
    return manifest


def _is_empty(value: Any) -> bool:
    """``True`` for ``None`` / empty string / empty collection (NOT numeric 0)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        return len(value) == 0
    return False


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Raise :class:`MissingManifestFieldError` if any required field is absent/empty.

    ``git_commit`` and ``container_digest`` may be empty (no git repo / no
    container); every other required field must be present and non-empty, and
    the nested ``seeds`` / ``tolerances`` mappings must carry their sub-keys.
    """
    for key in REQUIRED_FIELDS:
        if key not in manifest:
            raise MissingManifestFieldError(f"required manifest field missing: {key!r}")
        if key not in _NONEMPTY_EXEMPT and _is_empty(manifest[key]):
            raise MissingManifestFieldError(f"required manifest field empty: {key!r}")

    if manifest["threads"] != THREADS:
        raise MissingManifestFieldError(
            f"threads must be pinned to {THREADS} for determinism, got {manifest['threads']!r}"
        )

    seeds = manifest["seeds"]
    if not isinstance(seeds, Mapping):
        raise MissingManifestFieldError("manifest field 'seeds' must be a mapping")
    for seed_key in REQUIRED_SEED_KEYS:
        if seed_key not in seeds or _is_empty(seeds[seed_key]):
            raise MissingManifestFieldError(f"required seed missing/empty: seeds.{seed_key}")

    tolerances = manifest["tolerances"]
    if not isinstance(tolerances, Mapping):
        raise MissingManifestFieldError("manifest field 'tolerances' must be a mapping")
    for tol_key in REQUIRED_TOLERANCE_KEYS:
        if tol_key not in tolerances:
            raise MissingManifestFieldError(f"required tolerance missing: tolerances.{tol_key}")


def write_manifest(
    out_dir: str | Path,
    *,
    config: RunConfig,
    seeds: Sequence[int] | Mapping[str, Any] | None = None,
    solver_versions: list[dict[str, Any]] | None = None,
    hardware: Mapping[str, Any] | None = None,
    tolerances: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> Path:
    """Build, validate, and write ``repro_manifest.json``; return its path.

    Validation happens BEFORE the file is written, so an incomplete manifest is
    never persisted (the run aborts with :class:`MissingManifestFieldError`).
    """
    manifest = build_manifest(
        config=config,
        seeds=seeds,
        solver_versions=solver_versions,
        hardware=hardware,
        tolerances=tolerances,
        **kwargs,
    )
    validate_manifest(manifest)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST_FILENAME
    text = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Load and return ``<run_dir>/repro_manifest.json`` as a dict."""
    path = Path(run_dir) / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no reproducibility manifest at {path}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{MANIFEST_FILENAME} must be a JSON object, got {type(data).__name__}")
    return data


def finalize_run(
    out_dir: str | Path,
    *,
    config: RunConfig,
    seeds: Sequence[int] | Mapping[str, Any] | None = None,
    base_ir: MILP | None = None,
    time_limit: float | None = None,
    memory_limit: int | None = None,
    llm_sampling: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    """Finalise a run: persist the IR snapshot + write the validated manifest.

    Called by :func:`opop.orchestrator.loop.run_loop` at the end of every run so
    that NO run completes without a complete manifest. Persists ``base_ir`` to
    ``instance.json`` (so :mod:`opop.replay` can re-execute from disk) and writes
    ``repro_manifest.json``; returns the manifest path. Aborts with
    :class:`MissingManifestFieldError` on an incomplete manifest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if base_ir is not None:
        write_instance(out, base_ir)
    return write_manifest(
        out,
        config=config,
        seeds=seeds,
        time_limit=time_limit,
        memory_limit=memory_limit,
        llm_sampling=llm_sampling,
        **kwargs,
    )


# Private aliases so the public collectors can be overridden as keyword args
# (``build_manifest(git_commit=...)``) without shadowing the module functions.
_git_commit = git_commit
_container_digest = container_digest
_solver_versions = solver_versions
