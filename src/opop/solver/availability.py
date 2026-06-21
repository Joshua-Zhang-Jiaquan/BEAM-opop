"""Open-solver availability probing for OPOP.

Detects which open-source MILP / CP / LP backends are importable in the *active*
environment and reports a normalized ``{name, version, available}`` record for
each. Import or initialization failures are CAPTURED (never silently swallowed)
and surfaced via the ``detail`` field and the ``__main__`` CLI — per the task-3
requirement that a broken or absent solver must surface clearly rather than be
masked.

Solvers probed (open-only; no Gurobi):
  - ``SCIP``   via ``pyscipopt`` (wheel bundles the SCIP C library + SoPlex)
  - ``CP-SAT`` via ``ortools`` (``ortools.sat.python.cp_model``)
  - ``HiGHS``  via ``highspy``
  - ``CBC``    via ``pulp`` (PuLP bundles a CBC binary)
  - ``GCG``    via ``pygcgopt`` (Dantzig--Wolfe branch-price-and-cut on SCIP) —
    an *optional* extended backend (task 24). It is probeable via
    :func:`is_solver_available` (e.g. ``is_solver_available("gcg")``) but is NOT
    part of :data:`SOLVER_NAMES` (the canonical four with cross-solver smoke
    agreement), so adding it never perturbs the task-3 agreement table.

This module performs NO solving; see ``opop.solver.smoke`` for the tiny-MILP
agreement check used by the task-3 acceptance tests.

Run as a script to print the capability table::

    python -m opop.solver.availability
"""
from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import subprocess
import sys
from collections.abc import Callable

__all__ = [
    "SolverInfo",
    "SOLVER_NAMES",
    "solver_infos",
    "available_solvers",
    "is_solver_available",
    "main",
]

#: Canonical solver names, in the order reported by every public helper.
SOLVER_NAMES: tuple[str, ...] = ("SCIP", "CP-SAT", "HiGHS", "CBC")


@dataclasses.dataclass(frozen=True)
class SolverInfo:
    """Normalized availability record for a single solver backend.

    ``version`` is the underlying *solver* version where it differs from the
    Python binding (e.g. the SCIP / CBC engine version, not the wrapper); the
    binding/provider version is recorded in ``detail``.
    """

    name: str
    version: str | None
    available: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return the public ``{name, version, available, detail}`` mapping."""
        return {
            "name": self.name,
            "version": self.version,
            "available": self.available,
            "detail": self.detail,
        }


def _fail(name: str, exc: BaseException, where: str = "import") -> SolverInfo:
    """Build an *unavailable* record that surfaces the captured failure."""
    return SolverInfo(name, None, False, f"{where} failed: {type(exc).__name__}: {exc}")


@functools.lru_cache(maxsize=1)
def _detect_scip() -> SolverInfo:
    name = "SCIP"
    try:
        import pyscipopt
        from pyscipopt import Model
    except Exception as exc:  # absent or broken (e.g. SCIP ABI mismatch)
        return _fail(name, exc)
    binding = getattr(pyscipopt, "__version__", "?")
    try:
        # Model().version() returns the SCIP engine version (e.g. 10.0).
        scip_version = str(Model().version())
    except Exception as exc:
        return _fail(name, exc, where="SCIP init")
    detail = f"pyscipopt {binding}; SCIP {scip_version} (bundled engine + SoPlex)"
    return SolverInfo(name, scip_version, True, detail)


@functools.lru_cache(maxsize=1)
def _detect_cpsat() -> SolverInfo:
    name = "CP-SAT"
    try:
        import ortools
        from ortools.sat.python import cp_model
    except Exception as exc:
        return _fail(name, exc)
    version = getattr(ortools, "__version__", None)
    cpsat_ok = hasattr(cp_model, "CpModel")
    detail = f"ortools {version} (OR-Tools CP-SAT; CpModel={'ok' if cpsat_ok else 'MISSING'})"
    return SolverInfo(name, version, cpsat_ok, detail)


@functools.lru_cache(maxsize=1)
def _detect_highs() -> SolverInfo:
    name = "HiGHS"
    try:
        import highspy
    except Exception as exc:
        return _fail(name, exc)
    # highspy exposes no module-level __version__; query a Highs instance.
    version: str | None = getattr(highspy, "__version__", None)
    via = ""
    if version is None:
        try:
            version = str(highspy.Highs().version())
            via = " (version via Highs().version())"
        except Exception as exc:
            return _fail(name, exc, where="HiGHS version query")
    return SolverInfo(name, version, True, f"highspy {version}{via}")


@functools.lru_cache(maxsize=1)
def _detect_cbc() -> SolverInfo:
    name = "CBC"
    try:
        import pulp
    except Exception as exc:
        return _fail(name, exc)
    try:
        cbc = pulp.PULP_CBC_CMD(msg=False)
        available = bool(cbc.available())
    except Exception as exc:
        return _fail(name, exc, where="PuLP CBC probe")
    try:
        provider = f"PuLP {importlib.metadata.version('pulp')}"
    except importlib.metadata.PackageNotFoundError:
        provider = f"PuLP {getattr(pulp, '__version__', '?')}"
    if not available:
        return SolverInfo(name, None, False, f"{provider} present but bundled CBC binary not found")
    binary_version = _cbc_binary_version(getattr(cbc, "path", None))
    detail = f"CBC {binary_version or '?'} via {provider}"
    return SolverInfo(name, binary_version, True, detail)


def _cbc_binary_version(path: str | None) -> str | None:
    """Best-effort parse of the bundled CBC banner (``Version: X.Y.Z``)."""
    if not path:
        return None
    try:
        proc = subprocess.run(
            [path],
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("version:"):
            return stripped.split(":", 1)[1].strip()
    return None


@functools.lru_cache(maxsize=1)
def _detect_gcg() -> SolverInfo:
    name = "GCG"
    # importlib (not ``import pygcgopt``): the optional backend may be absent, so
    # a static import would be unresolved to the type checker.
    try:
        pygcgopt = importlib.import_module("pygcgopt")
    except Exception as exc:  # absent or broken (e.g. GCG/SCIP ABI mismatch)
        return _fail(name, exc)
    binding = getattr(pygcgopt, "__version__", "?")
    model_cls = getattr(pygcgopt, "Model", None)
    if model_cls is None:
        return SolverInfo(name, None, False, "pygcgopt present but exposes no Model")
    try:
        # Model().version() reports the underlying SCIP engine version; GCG runs
        # its Dantzig-Wolfe branch-price-and-cut on top of that engine.
        engine_version = str(model_cls().version())
    except Exception as exc:
        return _fail(name, exc, where="GCG init")
    detail = f"pygcgopt {binding}; GCG branch-price-and-cut on SCIP {engine_version}"
    return SolverInfo(name, engine_version, True, detail)


_DETECTORS: dict[str, Callable[[], SolverInfo]] = {
    "SCIP": _detect_scip,
    "CP-SAT": _detect_cpsat,
    "HiGHS": _detect_highs,
    "CBC": _detect_cbc,
    # GCG is probeable but intentionally NOT in SOLVER_NAMES (see module docstring).
    "GCG": _detect_gcg,
}

_ALIASES: dict[str, str] = {
    "scip": "SCIP",
    "pyscipopt": "SCIP",
    "cp-sat": "CP-SAT",
    "cpsat": "CP-SAT",
    "ortools": "CP-SAT",
    "or-tools": "CP-SAT",
    "highs": "HiGHS",
    "highspy": "HiGHS",
    "cbc": "CBC",
    "pulp": "CBC",
    "gcg": "GCG",
    "pygcgopt": "GCG",
}


def _canonical(name: str) -> str:
    return _ALIASES.get(name.strip().lower().replace("_", "-"), name)


def solver_infos() -> list[SolverInfo]:
    """Probe every known solver; one :class:`SolverInfo` each (order=SOLVER_NAMES)."""
    return [_DETECTORS[name]() for name in SOLVER_NAMES]


def available_solvers() -> list[dict[str, object]]:
    """Return ``[{name, version, available, detail}, ...]`` for all known solvers."""
    return [info.to_dict() for info in solver_infos()]


def is_solver_available(name: str) -> bool:
    """True iff the named solver (case-insensitive, aliases ok) is usable now."""
    detector = _DETECTORS.get(_canonical(name))
    return detector().available if detector is not None else False


def main() -> int:
    """Print the solver capability table; exit 0 (missing solvers -> stderr warn)."""
    infos = solver_infos()
    name_w = max(len(i.name) for i in infos)
    ver_w = max(len(str(i.version)) for i in infos)
    header = f"{'SOLVER':<{name_w}}  {'VERSION':<{ver_w}}  AVAILABLE  DETAIL"
    print(header)
    print("-" * len(header))
    for info in infos:
        flag = "yes" if info.available else "NO"
        print(f"{info.name:<{name_w}}  {str(info.version):<{ver_w}}  {flag:<9}  {info.detail}")
    missing = [i.name for i in infos if not i.available]
    if missing:
        print(f"\n[warn] unavailable: {', '.join(missing)} (see DETAIL)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
