"""Minimal, offline-safe QPLIB reader for small MIQP / MIQCP instances (task 35).

`QPLIB <https://qplib.zib.de>`_ is the reference library of quadratic programming
instances. Its ASCII format encodes

    optimise   1/2 x^T Q0 x + b0^T x + q0
    subject to c_l <= 1/2 x^T Qc x + a^T x <= c_u
    with       x_l <= x <= x_u ,  x of declared type (continuous/integer/binary)

one value per line (the rest of each line is free-form description, ignored) with
``!`` comment lines. This module implements the **subset** needed for the small,
hand-written MIQP/MIQCP fixtures under ``tests/bench/fixtures/qplib/`` — enough to
exercise the task-30 :class:`~opop.solver.miqp.MiqpAdapter` and the task-35
solver-backed cleaning harness, with no third-party dependency.

Supported subset (everything the committed fixtures use)
--------------------------------------------------------
* Header: problem name, 3-char type code, objective sense, ``n`` variables, and
  (when the constraint type char is not ``N``/``B``) ``m`` constraints.
* Objective: a lower-triangle Hessian ``Q0`` (present iff the objective type char
  is not ``L``), a sparse linear vector ``b0`` (default + overrides), and the
  constant ``q0``. The standard ``1/2`` Hessian factor is applied: a diagonal
  ``Q0[i,i]`` becomes ``coeff = Q0[i,i]/2`` on ``x_i^2`` and an off-diagonal
  ``Q0[i,j]`` (``i>j``) becomes ``coeff = Q0[i,j]`` on ``x_i x_j``.
* Constraints: a sparse quadratic part ``Qc`` (present iff the constraint type
  char is in ``{D,C,Q}``, same ``1/2`` rule), a sparse linear matrix ``A``, and
  the one-sided bounds ``c_l`` / ``c_u`` (default + overrides). Each row maps to a
  single :class:`~opop.model.ir.LinearConstraint` sense: ``c_l==c_u`` -> ``=``,
  ``c_l == -inf`` -> ``<=``, ``c_u == +inf`` -> ``>=`` (two-sided range rows are
  rejected).
* Variables: sparse lower/upper bounds (default + overrides); the variable type
  char maps ``C`` -> continuous, ``B`` -> binary, ``I``/``G`` -> integer. The
  mixed-integer per-variable type list (``M``) is **not** supported (it raises) —
  an explicit extension point.

Bounds with ``|value| >= 1e19`` are treated as infinite. Trailing sections of the
real format (initial points, variable/constraint names) are simply not read.
Malformed input raises :class:`QplibParseError` with file + line context rather
than loading a partial model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import final

from opop.bench.cleaning import CleaningItem
from opop.bench.registry import BenchmarkEntry
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    QuadraticExtension,
    QuadraticTerm,
    Variable,
    VarType,
)

__all__ = [
    "QPLIB_FIXTURES",
    "QplibFixture",
    "QplibParseError",
    "build_entries",
    "load_qplib",
    "load_qplib_items",
    "loads_qplib",
]

_REGISTRY_TIME_LIMIT_SEC = 300.0
_REGISTRY_PHASE = 6
_REGISTRY_THESIS = "T3"
_REGISTRY_BASELINE = "scip_default"
_REGISTRY_LICENSE = "MIT"

#: Magnitude at or above which a QPLIB bound value denotes infinity.
_INFINITY = 1e19

#: Constraint type chars that carry a quadratic (Hessian) part.
_QUADRATIC_CONSTRAINT_CHARS = frozenset({"D", "C", "Q"})

#: Constraint type chars meaning "no general constraints" (so ``m`` is omitted).
_NO_CONSTRAINT_CHARS = frozenset({"N", "B"})

_VTYPE_BY_CHAR: dict[str, VarType] = {
    "C": VarType.CONTINUOUS,
    "B": VarType.BINARY,
    "I": VarType.INTEGER,
    "G": VarType.INTEGER,
}


class QplibParseError(ValueError):
    """A malformed QPLIB instance, carrying file + line context.

    A :class:`ValueError` subclass (so generic ``except ValueError`` keeps
    working) whose message is prefixed with ``<source>:<line>:`` when known.
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
class _Reader:
    """Cursor over QPLIB *value lines* (non-blank, non-``!`` lines).

    Each value line is split into whitespace tokens; readers consume one line at a
    time and take the leading tokens they need (trailing description is ignored).
    """

    def __init__(self, text: str, *, source: str) -> None:
        self.source = source
        self._rows: list[tuple[int, list[str]]] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("!"):
                continue
            self._rows.append((lineno, stripped.split()))
        self._index = 0

    def _pop(self, what: str) -> tuple[int, list[str]]:
        if self._index >= len(self._rows):
            last = self._rows[-1][0] if self._rows else 1
            raise QplibParseError(
                f"unexpected end of file while reading {what}", source=self.source, line=last
            )
        row = self._rows[self._index]
        self._index += 1
        return row

    def _int_at(self, tokens: list[str], idx: int, what: str, lineno: int) -> int:
        if idx >= len(tokens):
            raise QplibParseError(
                f"expected an integer for {what}", source=self.source, line=lineno
            )
        try:
            return int(tokens[idx])
        except ValueError as exc:
            raise QplibParseError(
                f"expected an integer for {what}, got {tokens[idx]!r}",
                source=self.source,
                line=lineno,
            ) from exc

    def _float_at(self, tokens: list[str], idx: int, what: str, lineno: int) -> float:
        if idx >= len(tokens):
            raise QplibParseError(
                f"expected a number for {what}", source=self.source, line=lineno
            )
        try:
            return float(tokens[idx])
        except ValueError as exc:
            raise QplibParseError(
                f"expected a number for {what}, got {tokens[idx]!r}",
                source=self.source,
                line=lineno,
            ) from exc

    def token(self, what: str) -> str:
        """Consume the next value line and return its leading token."""
        lineno, tokens = self._pop(what)
        if not tokens:
            raise QplibParseError(f"expected a value for {what}", source=self.source, line=lineno)
        return tokens[0]

    def integer(self, what: str) -> int:
        """Consume the next value line as a single integer."""
        lineno, tokens = self._pop(what)
        return self._int_at(tokens, 0, what, lineno)

    def number(self, what: str) -> float:
        """Consume the next value line as a single number."""
        lineno, tokens = self._pop(what)
        return self._float_at(tokens, 0, what, lineno)

    def index_value(self, what: str) -> tuple[int, float]:
        """Consume an ``index value`` entry line."""
        lineno, tokens = self._pop(what)
        return self._int_at(tokens, 0, what, lineno), self._float_at(tokens, 1, what, lineno)

    def matrix_entry(self, what: str) -> tuple[int, int, float]:
        """Consume an ``i j value`` Hessian entry line."""
        lineno, tokens = self._pop(what)
        return (
            self._int_at(tokens, 0, what, lineno),
            self._int_at(tokens, 1, what, lineno),
            self._float_at(tokens, 2, what, lineno),
        )

    def constraint_matrix_entry(self, what: str) -> tuple[int, int, int, float]:
        """Consume a ``con i j value`` constraint-Hessian entry line."""
        lineno, tokens = self._pop(what)
        return (
            self._int_at(tokens, 0, what, lineno),
            self._int_at(tokens, 1, what, lineno),
            self._int_at(tokens, 2, what, lineno),
            self._float_at(tokens, 3, what, lineno),
        )


def _bound(value: float) -> float:
    """Map a QPLIB bound to a Python float, collapsing ``|v| >= 1e19`` to +-inf."""
    if value >= _INFINITY:
        return math.inf
    if value <= -_INFINITY:
        return -math.inf
    return value


def _sparse_vector(reader: _Reader, size: int, what: str) -> list[float]:
    """Read a ``default``/``count``/``index value`` sparse vector of length ``size``."""
    default = reader.number(f"default {what}")
    count = reader.integer(f"number of non-default {what}")
    values = [default] * size
    for _ in range(count):
        idx, value = reader.index_value(f"{what} entry")
        if not 1 <= idx <= size:
            raise QplibParseError(
                f"{what} index {idx} out of range [1, {size}]", source=reader.source
            )
        values[idx - 1] = value
    return values


def _quadratic_term(i: int, j: int, value: float) -> QuadraticTerm:
    """Build a :class:`QuadraticTerm` for a lower-triangle Hessian entry.

    The QPLIB objective/constraint carry ``1/2 x^T Q x``, so a diagonal entry
    contributes ``value/2`` on ``x_i^2`` and an off-diagonal (``i>j``) entry
    contributes ``value`` on ``x_i x_j``.
    """
    if i == j:
        return QuadraticTerm(f"x{i}", f"x{i}", value / 2.0)
    return QuadraticTerm(f"x{i}", f"x{j}", value)


def _constraint_sense(lower: float, upper: float, name: str, source: str) -> tuple[ConstraintSense, float]:
    """Map a QPLIB one-sided ``[lower, upper]`` row to an IR sense + rhs."""
    lower_inf = lower == -math.inf
    upper_inf = upper == math.inf
    if lower_inf and upper_inf:
        raise QplibParseError(f"constraint {name!r} is free (no finite bound)", source=source)
    if not lower_inf and not upper_inf:
        if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-9):
            return ConstraintSense.EQ, upper
        raise QplibParseError(
            f"constraint {name!r} has a two-sided range [{lower}, {upper}]; not supported",
            source=source,
        )
    if lower_inf:
        return ConstraintSense.LE, upper
    return ConstraintSense.GE, lower


def loads_qplib(text: str, *, name: str = "qplib", source: str = "<string>") -> MILP:
    """Parse QPLIB ``text`` into a :class:`~opop.model.ir.MILP` (MIQP / MIQCP).

    Args:
        text: The QPLIB instance text.
        name: Name for the produced IR (the file's declared name is kept in
            ``metadata['qplib_name']``).
        source: Source label used in :class:`QplibParseError` messages.

    Raises:
        QplibParseError: On malformed input or an unsupported construct.
    """
    reader = _Reader(text, source=source)

    qplib_name = reader.token("problem name")
    type_code = reader.token("problem type").upper()
    if len(type_code) != 3:
        raise QplibParseError(
            f"problem type must be 3 characters, got {type_code!r}", source=source
        )
    obj_char, con_char, var_char = type_code[0], type_code[1], type_code[2]

    vtype = _VTYPE_BY_CHAR.get(var_char)
    if vtype is None:
        if var_char == "M":
            raise QplibParseError(
                "mixed-integer variable-type list (type char 'M') is not supported",
                source=source,
            )
        raise QplibParseError(f"unsupported variable type char {var_char!r}", source=source)

    sense = (
        ObjSense.MAXIMIZE
        if reader.token("objective sense").lower().startswith("max")
        else ObjSense.MINIMIZE
    )

    n = reader.integer("number of variables")
    if n < 1:
        raise QplibParseError(f"need at least 1 variable, got {n}", source=source)
    m = 0 if con_char in _NO_CONSTRAINT_CHARS else reader.integer("number of constraints")
    if m < 0:
        raise QplibParseError(f"number of constraints must be >= 0, got {m}", source=source)

    obj_terms: list[QuadraticTerm] = []
    if obj_char != "L":
        nnz_q0 = reader.integer("number of objective Hessian entries")
        for _ in range(nnz_q0):
            i, j, value = reader.matrix_entry("objective Hessian entry")
            _check_var(i, n, source)
            _check_var(j, n, source)
            obj_terms.append(_quadratic_term(i, j, value))

    b0 = _sparse_vector(reader, n, "objective linear coefficient")
    offset = reader.number("objective constant")

    con_quad: dict[int, list[QuadraticTerm]] = {}
    con_linear: dict[int, dict[str, float]] = {}
    con_sense: dict[int, ConstraintSense] = {}
    con_rhs: dict[int, float] = {}
    if m > 0:
        if con_char in _QUADRATIC_CONSTRAINT_CHARS:
            nnz_qc = reader.integer("number of quadratic constraint entries")
            for _ in range(nnz_qc):
                k, i, j, value = reader.constraint_matrix_entry("quadratic constraint entry")
                _check_con(k, m, source)
                _check_var(i, n, source)
                _check_var(j, n, source)
                con_quad.setdefault(k, []).append(_quadratic_term(i, j, value))

        nnz_a = reader.integer("number of linear constraint entries")
        for _ in range(nnz_a):
            k, i, value = reader.matrix_entry("linear constraint entry")
            _check_con(k, m, source)
            _check_var(i, n, source)
            con_linear.setdefault(k, {})[f"x{i}"] = value

        lowers = _sparse_vector(reader, m, "constraint lower bound")
        uppers = _sparse_vector(reader, m, "constraint upper bound")
        for k in range(1, m + 1):
            sense_k, rhs_k = _constraint_sense(
                _bound(lowers[k - 1]), _bound(uppers[k - 1]), f"c{k}", source
            )
            con_sense[k] = sense_k
            con_rhs[k] = rhs_k

    var_lowers = _sparse_vector(reader, n, "variable lower bound")
    var_uppers = _sparse_vector(reader, n, "variable upper bound")

    variables = tuple(
        Variable(f"x{i}", vtype, _bound(var_lowers[i - 1]), _bound(var_uppers[i - 1]))
        for i in range(1, n + 1)
    )
    constraints = tuple(
        LinearConstraint(f"c{k}", con_linear.get(k, {}), con_sense[k], con_rhs[k])
        for k in range(1, m + 1)
    )
    obj_coeffs = {f"x{i}": b0[i - 1] for i in range(1, n + 1) if b0[i - 1] != 0.0}
    objective = Objective(coeffs=obj_coeffs, sense=sense, offset=offset)

    constraint_terms = {f"c{k}": tuple(terms) for k, terms in con_quad.items() if terms}
    extension: QuadraticExtension | None = None
    if obj_terms or constraint_terms:
        extension = QuadraticExtension(
            objective_terms=tuple(obj_terms), constraint_terms=constraint_terms
        )

    return MILP(
        name=name,
        variables=variables,
        constraints=constraints,
        objective=objective,
        metadata={
            "source": "qplib",
            "qplib_name": qplib_name,
            "qplib_type": type_code,
            "instance": name,
        },
        quadratic=extension,
    )


def _check_var(idx: int, n: int, source: str) -> None:
    if not 1 <= idx <= n:
        raise QplibParseError(f"variable index {idx} out of range [1, {n}]", source=source)


def _check_con(idx: int, m: int, source: str) -> None:
    if not 1 <= idx <= m:
        raise QplibParseError(f"constraint index {idx} out of range [1, {m}]", source=source)


def load_qplib(path: str | Path) -> MILP:
    """Load a QPLIB ``.qplib`` file at ``path`` into a :class:`~opop.model.ir.MILP`."""
    text = Path(path).read_text(encoding="utf-8")
    return loads_qplib(text, name=Path(path).stem, source=str(path))


@dataclass(frozen=True, slots=True)
class QplibFixture:
    """One committed QPLIB fixture with its hand-verified reference optimum.

    Attributes:
        name: Instance name stem (the registry id is ``qplib/<name>``).
        filename: File name under the QPLIB fixtures directory.
        problem_type: Human-readable class tag (``MIQP`` / ``MIQCP``).
        reference_optimum: The known optimal objective value (SCIP-confirmed).
        sha256: Hex SHA-256 of the committed fixture file (content lock).
        entry_name: Registry entry name / leakage_group for this fixture.
    """

    name: str
    filename: str
    problem_type: str
    reference_optimum: float
    sha256: str
    entry_name: str

    @property
    def id(self) -> str:
        """Globally unique registry instance id (``qplib/<name>``)."""
        return f"qplib/{self.name}"


#: The committed QPLIB fixtures (small, offline, hand-verified optima).
QPLIB_FIXTURES: tuple[QplibFixture, ...] = (
    QplibFixture(
        "ball_miqcp",
        "ball_miqcp.qplib",
        "MIQCP",
        7.0,
        "e31c9123c760c8936cd1fe660fcd391c5eeb867894a994268259eeb7524da1fd",
        "qplib_miqcp_tiny",
    ),
    QplibFixture(
        "box_miqp",
        "box_miqp.qplib",
        "MIQP",
        1.0,
        "3984964c44fc6d586b9ca55f3d31d3ce1bee81f3eb59f99cfcf5abe0cdcb00f0",
        "qplib_miqp_tiny",
    ),
)


def load_qplib_items(
    fixtures_dir: str | Path,
    *,
    fixtures: tuple[QplibFixture, ...] = QPLIB_FIXTURES,
) -> list[CleaningItem]:
    """Load QPLIB ``fixtures`` from ``fixtures_dir`` as labeled :class:`CleaningItem`s.

    The returned items feed directly into
    :func:`opop.bench.cleaning.verify_and_clean`: the ``labeled_optimum`` is each
    fixture's reference optimum, to be re-confirmed by a solver.
    """
    base = Path(fixtures_dir)
    items: list[CleaningItem] = []
    for fixture in fixtures:
        ir = load_qplib(base / fixture.filename)
        items.append(
            CleaningItem(
                id=fixture.id,
                ir=ir,
                labeled_optimum=fixture.reference_optimum,
                sense=ir.objective.sense,
                source_dataset="qplib",
            )
        )
    return items


def build_entries() -> list[BenchmarkEntry]:
    """Return one held-out registry entry per QPLIB fixture.

    Each fixture is a ``phase=6`` / ``thesis=T3`` generality benchmark (like the
    classic-CO families): it sits alone in the immutable ``test`` split as its own
    ``leakage_group`` (so no group spans a free and a held-out split), with a
    ``sha256:`` checksum over the committed fixture file.
    """
    return [
        BenchmarkEntry(
            name=fixture.entry_name,
            problem_type=fixture.problem_type,
            source="qplib",
            split={"test": (fixture.id,)},
            license=_REGISTRY_LICENSE,
            instance_count=1,
            time_limit_sec=_REGISTRY_TIME_LIMIT_SEC,
            baseline_set=_REGISTRY_BASELINE,
            leakage_group=fixture.entry_name,
            checksum="sha256:" + fixture.sha256,
            phase=_REGISTRY_PHASE,
            thesis=_REGISTRY_THESIS,
        )
        for fixture in QPLIB_FIXTURES
    ]
