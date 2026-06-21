"""Classic combinatorial-optimization benchmark loaders (plan task 34).

A unified, capability-driven loader layer for six classic CO families, each
behind a :class:`~opop.model.adapter.ProblemClassAdapter` and mapped to the OPOP
IR by ONE generic formulation (no per-instance hand-tuning):

============  =================================  ============================
Family        Native format                      Generic IR formulation
============  =================================  ============================
``tsp``       TSPLIB (EUC_2D / EXPLICIT)          Miller–Tucker–Zemlin MILP
``cvrp``      CVRPLIB                            two-index MTZ-capacity MILP
``orlib``     OR-Library set covering (SCP)       set-cover MILP
``jsp``       JSPLIB job shop                    disjunctive makespan MILP
``maxsat``    DIMACS CNF / WCNF                  clause-satisfaction MILP
``maxcut``    Biq Mac / Gset graph               QUBO-shaped IR (quadratic)
============  =================================  ============================

Importing this package registers all six ``classic-<family>`` adapters in the
process-wide adapter registry (:func:`opop.model.adapter.register_adapter`), so
``find_adapter(ir)`` routes a tagged classic IR to its adapter. The committed
small fixtures + their Wave-6 held-out registry entries live in
:mod:`opop.bench.classic.catalog` (consumed by the combined-registry generator
:mod:`opop.bench.sources.milp_suites`); malformed instances raise
:class:`~opop.bench.classic.base.ParseError` with file + line context.

Public entry points: :func:`load_instance` (by family id + path),
:func:`loads_instance` (by family id + text), and the per-family
``<module>.load`` / ``<module>.loads`` functions.
"""

from __future__ import annotations

from collections.abc import Callable

from opop.bench.classic import cvrp, jsp, maxcut, maxsat, orlib, tsp
from opop.bench.classic.base import ClassicAdapter, ParseError, TokenCursor, co_family
from opop.bench.classic.catalog import (
    CLASSIC_FAMILIES,
    CLASSIC_FIXTURES,
    ClassicFamily,
    ClassicFixture,
    build_classic_entries,
    family_checksum,
)
from opop.model.adapter import ProblemClassAdapter
from opop.model.ir import MILP

__all__ = [
    "ADAPTERS",
    "CLASSIC_FAMILIES",
    "CLASSIC_FIXTURES",
    "FAMILIES",
    "LOADERS",
    "ClassicAdapter",
    "ClassicFamily",
    "ClassicFixture",
    "ParseError",
    "TokenCursor",
    "adapter_for",
    "build_classic_entries",
    "co_family",
    "cvrp",
    "family_checksum",
    "jsp",
    "load_instance",
    "loads_instance",
    "maxcut",
    "maxsat",
    "orlib",
    "tsp",
]

#: Per-family file loaders (``family id -> load(path) -> MILP``).
LOADERS: dict[str, Callable[[str], MILP]] = {
    "tsp": tsp.load,
    "cvrp": cvrp.load,
    "orlib": orlib.load,
    "jsp": jsp.load,
    "maxsat": maxsat.load,
    "maxcut": maxcut.load,
}

#: Per-family text loaders (``family id -> loads(text, *, name, source) -> MILP``).
_LOADS: dict[str, Callable[..., MILP]] = {
    "tsp": tsp.loads,
    "cvrp": cvrp.loads,
    "orlib": orlib.loads,
    "jsp": jsp.loads,
    "maxsat": maxsat.loads,
    "maxcut": maxcut.loads,
}

#: Registered classic-CO adapters (``family id -> ClassicAdapter``).
ADAPTERS: dict[str, ClassicAdapter] = {
    "tsp": tsp.ADAPTER,
    "cvrp": cvrp.ADAPTER,
    "orlib": orlib.ADAPTER,
    "jsp": jsp.ADAPTER,
    "maxsat": maxsat.ADAPTER,
    "maxcut": maxcut.ADAPTER,
}

#: The recognised classic-CO family ids.
FAMILIES: tuple[str, ...] = tuple(LOADERS)


def _require_family(family: str) -> None:
    if family not in LOADERS:
        raise ValueError(
            f"unknown classic-CO family {family!r}; known: {sorted(LOADERS)}"
        )


def load_instance(family: str, path: str) -> MILP:
    """Load a classic-CO ``family`` instance file at ``path`` into a :class:`MILP`.

    Raises:
        ValueError: If ``family`` is not a recognised classic-CO family.
        ParseError: If the file is malformed (with file + line context).
    """
    _require_family(family)
    return LOADERS[family](path)


def loads_instance(
    family: str, text: str, *, name: str = "instance", source: str = "<string>"
) -> MILP:
    """Parse classic-CO ``family`` instance ``text`` into a :class:`MILP`.

    Raises:
        ValueError: If ``family`` is not a recognised classic-CO family.
        ParseError: If the text is malformed (with source + line context).
    """
    _require_family(family)
    return _LOADS[family](text, name=name, source=source)


def adapter_for(family: str) -> ProblemClassAdapter:
    """Return the registered :class:`ProblemClassAdapter` for a classic ``family``."""
    _require_family(family)
    return ADAPTERS[family]
