"""MaxCut graph loader → QUBO-shaped IR (task 34).

Parses an undirected weighted graph (the Biq Mac / `Gset
<https://web.stanford.edu/~yyye/yyye/Gset/>`_ "rudy" layout, plus a tolerated
DIMACS ``p edge`` / ``e`` variant) and maps it to the OPOP IR as a genuine
**QUBO** via the model-layer quadratic builders — NOT a hand-rolled MILP:

* :func:`opop.model.quadratic.max_cut_qubo` builds the Max-Cut QUBO
  ``minimise sum_e w_e (2 x_u x_v - x_u - x_v)`` (so ``-min`` is the cut weight);
* :func:`opop.model.quadratic.qubo_to_ir` wraps it as a pure-binary
  :class:`~opop.model.ir.MILP` carrying a :class:`~opop.model.ir.QuadraticExtension`.

The QUBO-shaped IR is solved through any linear kernel after the exact Fortet
linearization (:meth:`opop.bench.classic.base.ClassicAdapter.to_milp`), which is
the textbook edge-variable Max-Cut MILP. The committed
:class:`~opop.solver.qubo.QuboAdapter` would also claim this IR structurally; the
``classic-maxcut`` adapter claims it by its ``co_family`` tag — both linearize
identically.

Layout (``c`` / ``#`` comment lines ignored)::

    n m              # nodes, edges  (or 'p edge n m')
    u v [w]          # 1-based endpoints, optional weight (default 1.0)

A non-integer token, an endpoint out of ``[1, n]``, a self-loop, or fewer than
``m`` edges raises :class:`~opop.bench.classic.base.ParseError` with file + line
context.
"""

from __future__ import annotations

from pathlib import Path

from opop.bench.classic.base import ClassicAdapter, ParseError, read_text, tag_instance
from opop.model.adapter import register_adapter
from opop.model.ir import MILP
from opop.model.quadratic import max_cut_qubo, qubo_to_ir

__all__ = ["ADAPTER", "load", "loads"]

#: The registered classic-CO adapter for the MaxCut family (QUBO-shaped IR).
ADAPTER = ClassicAdapter(family="maxcut", problem_class="MaxCut", quadratic=True)


def _data_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(lineno, stripped)`` for non-blank, non-comment lines."""
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped[0] in ("c", "C", "#"):
            continue
        out.append((lineno, stripped))
    return out


def _int(token: str, what: str, *, source: str, line: int) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise ParseError(
            f"expected integer for {what}, got {token!r}", source=source, line=line
        ) from exc


def _float(token: str, what: str, *, source: str, line: int) -> float:
    try:
        return float(token)
    except ValueError as exc:
        raise ParseError(
            f"expected number for {what}, got {token!r}", source=source, line=line
        ) from exc


def loads(text: str, *, name: str = "maxcut", source: str = "<string>") -> MILP:
    """Parse a MaxCut graph ``text`` into a QUBO-shaped :class:`~opop.model.ir.MILP`.

    Raises:
        ParseError: On a non-integer token, an out-of-range / self-loop edge, or
            fewer than the declared ``m`` edges.
    """
    data = _data_lines(text)
    if not data:
        raise ParseError("empty MaxCut graph (no header)", source=source)

    header_line, header = data[0]
    tokens = header.split()
    if tokens and tokens[0].lower() == "p":
        if len(tokens) < 4:
            raise ParseError(
                f"malformed DIMACS header {header!r} (expected 'p edge n m')",
                source=source,
                line=header_line,
            )
        n_nodes = _int(tokens[2], "node count", source=source, line=header_line)
        n_edges = _int(tokens[3], "edge count", source=source, line=header_line)
    else:
        if len(tokens) < 2:
            raise ParseError(
                f"malformed header {header!r} (expected 'n m')",
                source=source,
                line=header_line,
            )
        n_nodes = _int(tokens[0], "node count", source=source, line=header_line)
        n_edges = _int(tokens[1], "edge count", source=source, line=header_line)
    if n_nodes < 1:
        raise ParseError(f"need n >= 1 nodes, got {n_nodes}", source=source, line=header_line)

    edge_rows = data[1:]
    if len(edge_rows) < n_edges:
        raise ParseError(
            f"expected {n_edges} edges, got {len(edge_rows)}",
            source=source,
            line=data[-1][0] if data else header_line,
        )

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for lineno, row in edge_rows[:n_edges]:
        parts = row.split()
        if parts and parts[0].lower() == "e":
            parts = parts[1:]
        if len(parts) < 2:
            raise ParseError(
                f"edge row needs at least 'u v', got {row!r}", source=source, line=lineno
            )
        u = _int(parts[0], "edge endpoint u", source=source, line=lineno)
        v = _int(parts[1], "edge endpoint v", source=source, line=lineno)
        weight = _float(parts[2], "edge weight", source=source, line=lineno) if len(parts) > 2 else 1.0
        if not (1 <= u <= n_nodes and 1 <= v <= n_nodes):
            raise ParseError(
                f"edge ({u}, {v}) endpoint out of range [1, {n_nodes}]",
                source=source,
                line=lineno,
            )
        if u == v:
            raise ParseError(
                f"self-loop edge ({u}, {v}) is not allowed in MaxCut",
                source=source,
                line=lineno,
            )
        edges.append((u - 1, v - 1))
        weights.append(weight)

    qubo = max_cut_qubo(n_nodes, edges, weights)
    ir = qubo_to_ir(qubo, name=name)
    return tag_instance(
        ir,
        family="maxcut",
        source="maxcut",
        instance=name,
        extra={"domain": "graph", "formulation": "qubo_maxcut", "n_nodes": n_nodes, "n_edges": n_edges},
    )


def load(path: str) -> MILP:
    """Load a MaxCut ``.txt`` graph file into a QUBO-shaped :class:`~opop.model.ir.MILP`."""
    return loads(read_text(path), name=Path(path).stem, source=str(path))


register_adapter(ADAPTER)
