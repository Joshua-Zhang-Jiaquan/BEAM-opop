"""Shared primitives for the classic CO loaders (task 34).

This module holds everything the six family loaders
(:mod:`opop.bench.classic.tsp` … :mod:`opop.bench.classic.maxcut`) share:

* :class:`ParseError` — a :class:`ValueError` subclass that always carries a
  ``source`` (file / origin) and a ``line`` so a malformed instance fails loudly
  with context (``<source>:<line>: <message>``) rather than a silent partial load.
* :class:`TokenCursor` — a whitespace token stream with per-token line tracking,
  used by the free-form numeric formats (OR-Library set covering, JSPLIB,
  MaxCut, DIMACS) where counts and values span lines arbitrarily.
* :class:`ClassicAdapter` — ONE generic
  :class:`~opop.model.adapter.ProblemClassAdapter` implementation, instantiated
  once per family (``classic-tsp``, ``classic-cvrp``, …). It dispatches on the
  ``co_family`` metadata tag the loaders stamp onto every produced IR, and its
  :meth:`ClassicAdapter.to_milp` returns an exact linear MILP — identity for the
  already-linear families and the Fortet linearization
  (:func:`opop.model.quadratic.linearize_quadratic`) for the QUBO-shaped MaxCut
  family.

The layering is deliberate: this package depends ONLY on the ``opop.model``
layer (IR + quadratic + adapter Protocol). The generic MILP/QUBO formulations
live with each loader; no formulation is hand-tuned per instance.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, final

from opop.model.adapter import AdapterCapabilities
from opop.model.ir import MILP
from opop.model.quadratic import linearize_quadratic
from opop.model.state import Phi, SolveTrace

if TYPE_CHECKING:
    from opop.solver.kernel import SolverKernel

__all__ = [
    "ClassicAdapter",
    "ParseError",
    "TokenCursor",
    "co_family",
    "euclidean_matrix",
    "nint",
    "read_text",
    "tag_instance",
]

#: Metadata key carrying a produced IR's classic-CO family id (the adapter key).
CO_FAMILY_KEY = "co_family"

#: Linear kernels that can solve any classic IR after :meth:`ClassicAdapter.to_milp`.
_LINEAR_KERNELS: tuple[str, ...] = ("CP-SAT", "SCIP", "HiGHS")


class ParseError(ValueError):
    """A malformed classic-CO instance file, with file + line context.

    A :class:`ValueError` subclass so generic ``except ValueError`` handlers keep
    working, while ``except ParseError`` callers get the structured ``source`` /
    ``line`` attributes. The rendered message is prefixed with the location so a
    truncated / inconsistent file never loads partially in silence.
    """

    def __init__(self, message: str, *, source: str | None = None, line: int | None = None) -> None:
        self.source: str | None = source
        self.line: int | None = line
        if source is not None and line is not None:
            located = f"{source}:{line}: {message}"
        elif source is not None:
            located = f"{source}: {message}"
        elif line is not None:
            located = f"line {line}: {message}"
        else:
            located = message
        super().__init__(located)


@final
class TokenCursor:
    """A whitespace-delimited token stream with per-token line tracking.

    Built once from the full file text; each ``next_*`` consumes one token and
    raises a contextual :class:`ParseError` (with the offending line) on EOF or a
    non-numeric token. ``#`` and ``c`` DIMACS-style comment handling is left to
    the caller (it is format-specific); :class:`TokenCursor` only tokenises and
    converts.
    """

    def __init__(self, text: str, *, source: str = "<string>") -> None:
        self.source = source
        self._tokens: list[str] = []
        self._lines: list[int] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            for tok in raw.split():
                self._tokens.append(tok)
                self._lines.append(lineno)
        self._index = 0

    @property
    def eof(self) -> bool:
        """``True`` iff every token has been consumed."""
        return self._index >= len(self._tokens)

    @property
    def line(self) -> int:
        """Line number of the most recently consumed token (1 if none yet)."""
        if self._index == 0:
            return self._lines[0] if self._lines else 1
        return self._lines[self._index - 1]

    @property
    def peek_line(self) -> int:
        """Line number of the NEXT token (last line at EOF)."""
        if self.eof:
            return self._lines[-1] if self._lines else 1
        return self._lines[self._index]

    def next_token(self, what: str) -> str:
        """Consume and return the next raw token, or raise :class:`ParseError`."""
        if self.eof:
            raise ParseError(
                f"unexpected end of file while reading {what}",
                source=self.source,
                line=self.peek_line,
            )
        tok = self._tokens[self._index]
        self._index += 1
        return tok

    def next_int(self, what: str) -> int:
        """Consume the next token as an ``int`` (raises with context on failure)."""
        tok = self.next_token(what)
        try:
            return int(tok)
        except ValueError as exc:
            raise ParseError(
                f"expected integer for {what}, got {tok!r}",
                source=self.source,
                line=self.line,
            ) from exc

    def next_float(self, what: str) -> float:
        """Consume the next token as a ``float`` (raises with context on failure)."""
        tok = self.next_token(what)
        try:
            return float(tok)
        except ValueError as exc:
            raise ParseError(
                f"expected number for {what}, got {tok!r}",
                source=self.source,
                line=self.line,
            ) from exc


def co_family(ir: MILP) -> str | None:
    """Return the classic-CO family tag of ``ir`` (``None`` if untagged)."""
    value = ir.metadata.get(CO_FAMILY_KEY)
    return value if isinstance(value, str) else None


def tag_instance(
    ir: MILP,
    *,
    family: str,
    source: str,
    instance: str,
    extra: dict[str, Any] | None = None,
) -> MILP:
    """Return ``ir`` renamed to ``instance`` and tagged with its classic-CO family.

    The ``co_family`` tag is what :meth:`ClassicAdapter.can_handle` dispatches on,
    so every loader funnels its result through here. ``ir`` is never mutated
    (frozen dataclass ``replace``).
    """
    metadata: dict[str, Any] = {
        **ir.metadata,
        CO_FAMILY_KEY: family,
        "source": source,
        "instance": instance,
    }
    if extra:
        metadata.update(extra)
    return replace(ir, name=instance, metadata=metadata)


@final
class ClassicAdapter:
    """Generic :class:`~opop.model.adapter.ProblemClassAdapter` for a CO family.

    One instance is registered per family (``classic-tsp`` … ``classic-maxcut``).
    Dispatch is by the ``co_family`` metadata tag (set by :func:`tag_instance`),
    NOT by structural inspection, so a classic IR routes to its own adapter even
    when (as for MaxCut) a structural QUBO adapter would also claim it.

    ``to_milp`` is exact: identity for the already-linear families and the Fortet
    linearization for a QUBO-shaped (quadratic) IR. ``native_solve`` is the
    linearize-then-solve convenience (classic families all have an exact linear
    MILP), so it routes to any linear kernel passed in.
    """

    def __init__(self, *, family: str, problem_class: str, quadratic: bool = False) -> None:
        self._family = family
        self._problem_class = problem_class
        self._quadratic = quadratic

    @property
    def family(self) -> str:
        """The classic-CO family id this adapter handles (e.g. ``"tsp"``)."""
        return self._family

    @property
    def name(self) -> str:
        """Unique registry key (``classic-<family>``)."""
        return f"classic-{self._family}"

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declared capabilities (exact linear MILP via :meth:`to_milp`)."""
        return AdapterCapabilities(
            name=self.name,
            problem_class=self._problem_class,
            handles_quadratic_objective=self._quadratic,
            handles_quadratic_constraints=False,
            exact_linearization=True,
            native_kernels=(),
            linear_kernels=_LINEAR_KERNELS,
        )

    def can_handle(self, ir: MILP) -> bool:
        """Handle an IR this loader produced (matched by the ``co_family`` tag)."""
        return co_family(ir) == self._family

    def to_milp(self, ir: MILP) -> MILP:
        """Return an exact linear MILP: identity if linear, Fortet if quadratic."""
        ext = ir.quadratic
        if ext is not None and not ext.is_empty:
            return linearize_quadratic(ir)
        if ext is not None:
            return replace(ir, quadratic=None)
        return ir

    def native_solve(
        self,
        ir: MILP,
        kernel: SolverKernel,
        *,
        phi: Phi | None = None,
        time_limit: float = 60.0,
        memory_limit_mb: int = 4096,
        seed: int = 0,
    ) -> SolveTrace:
        """Solve the exact linear MILP of ``ir`` on ``kernel`` (linearize-then-solve)."""
        return kernel.solve(
            self.to_milp(ir),
            phi if phi is not None else Phi(),
            time_limit=time_limit,
            memory_limit_mb=memory_limit_mb,
            seed=seed,
        )


def read_text(path: str | Path) -> str:
    """Read a fixture file as UTF-8 text (small helper shared by the loaders)."""
    return Path(path).read_text(encoding="utf-8")


def nint(value: float) -> float:
    """TSPLIB/CVRPLIB nearest-integer rounding ``nint(x) = floor(x + 0.5)``."""
    return float(math.floor(value + 0.5))


def euclidean_matrix(coords: list[tuple[float, float]]) -> dict[tuple[int, int], float]:
    """Build the rounded-Euclidean (``EUC_2D``) cost matrix from node coordinates.

    ``cost[(i, j)] = nint(sqrt((xi-xj)^2 + (yi-yj)^2))`` for every ``i != j``;
    the matrix is symmetric and excludes the zero diagonal.
    """
    n = len(coords)
    cost: dict[tuple[int, int], float] = {}
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            cost[(i, j)] = nint(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
    return cost
