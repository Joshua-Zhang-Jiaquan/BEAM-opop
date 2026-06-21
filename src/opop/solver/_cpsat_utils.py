"""Deterministic rational-to-integer scaling helpers for the CP-SAT kernel (task 22).

OR-Tools CP-SAT is an *integer* solver: every variable domain and every linear
constraint / objective coefficient must be an integer. The OPOP MILP IR, by
contrast, carries ``float`` coefficients and right-hand sides. This module turns
a row of floats into an exactly-equivalent row of integers by multiplying the
whole row by a common positive scale factor.

Why exactness matters
----------------------
Multiplying a single linear constraint ``sum a_j x_j (<=|>=|=) rhs`` (or the
objective ``sum c_j x_j``) by a positive constant ``K`` is an *equivalence*: the
feasible region and the arg-optimum are unchanged. So if every ``a_j`` and
``rhs`` is a rational ``p/q``, choosing ``K = lcm(denominators)`` yields integer
coefficients ``a_j * K`` with **no rounding**. The objective scale ``K`` and the
constant ``offset`` are recorded so the reported objective value can be mapped
back to the true (unscaled) space via ``true = cpsat_value / K + offset``.

The hard part is recovering ``p/q`` from a binary ``float``: ``Fraction(0.1)`` is
the exact (huge-denominator) binary value, not ``1/10``. We use
``Fraction(value).limit_denominator(max_denominator)`` — the standard best-
rational-approximation — and then **verify** the recovered fraction reproduces
the float within ``tol``. If it does not (an irrational-looking coefficient, or
one needing a denominator beyond ``max_denominator``), we raise
:class:`~opop.model.ir.UnsupportedModelError` rather than silently scaling to a
*wrong* optimum. Scaled magnitudes are likewise bounded by
:data:`MAX_INT_MAGNITUDE` so an exploding ``lcm`` can never overflow CP-SAT's
safe integer range undetected.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from opop.model.ir import UnsupportedModelError

__all__ = [
    "DEFAULT_MAX_DENOMINATOR",
    "DEFAULT_SCALE_TOL",
    "MAX_INT_MAGNITUDE",
    "scale_row_to_integers",
    "to_exact_fraction",
]

#: Largest denominator tried when recovering a rational from a float coefficient.
#: ``Fraction.limit_denominator`` recovers "nice" fractions (``1/3``, ``0.1``)
#: well below this; the cap bounds how large the per-row ``lcm`` scale can grow.
DEFAULT_MAX_DENOMINATOR: int = 1_000_000

#: Absolute tolerance for accepting ``limit_denominator``'s approximation as the
#: coefficient's true value. ``1e-9`` accepts genuine rationals (error ~1e-16)
#: while rejecting coefficients that only fit with a denominator past the cap.
DEFAULT_SCALE_TOL: float = 1e-9

#: Conservative ceiling for a scaled integer coefficient / rhs / bound and for
#: the scale factor itself. ``2**53`` is the largest integer exactly
#: representable as a float and stays well inside CP-SAT's int64 arithmetic, so
#: products/sums formed internally do not overflow or lose precision.
MAX_INT_MAGNITUDE: int = 1 << 53


def to_exact_fraction(
    value: float,
    *,
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
    tol: float = DEFAULT_SCALE_TOL,
    what: str = "coefficient",
) -> Fraction:
    """Recover the exact rational behind ``value`` or raise (never approximate).

    Uses ``Fraction(value).limit_denominator(max_denominator)`` and verifies the
    result reproduces ``value`` within ``tol``. Non-finite values and values that
    only fit with a denominator beyond ``max_denominator`` raise
    :class:`~opop.model.ir.UnsupportedModelError` so the caller fails closed
    instead of scaling to a silently wrong optimum.

    Args:
        value: The float coefficient / rhs / bound to convert.
        max_denominator: Largest denominator tried (see :data:`DEFAULT_MAX_DENOMINATOR`).
        tol: Absolute error tolerance for accepting the approximation.
        what: Human-readable label for error messages (e.g. ``"constraint 'c0'"``).

    Returns:
        A :class:`fractions.Fraction` exactly equal (within ``tol``) to ``value``.
    """
    if not math.isfinite(value):
        raise UnsupportedModelError(
            f"{what}: non-finite value {value!r} cannot be scaled to an integer for CP-SAT"
        )
    frac = Fraction(value).limit_denominator(max_denominator)
    if abs(float(frac) - value) > tol:
        raise UnsupportedModelError(
            f"{what}: value {value!r} is not exactly representable as a rational with "
            + f"denominator <= {max_denominator} within tol {tol} "
            + f"(closest is {frac} = {float(frac)!r}). CP-SAT is integer-only; refusing to "
            + "scale it to avoid a silently wrong optimum (use the SCIP/HiGHS kernel instead)."
        )
    return frac


def scale_row_to_integers(
    values: Sequence[float],
    *,
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
    tol: float = DEFAULT_SCALE_TOL,
    what: str = "row",
) -> tuple[int, list[int]]:
    """Scale a row of floats to integers by a common positive factor (exact).

    Returns ``(scale, ints)`` where ``scale = lcm(denominators) >= 1`` and
    ``ints[i] == values[i] * scale`` exactly (integer arithmetic, no rounding).
    Multiplying a constraint or objective by ``scale`` is sense-preserving, so
    the integer row is mathematically equivalent to the float row.

    An empty ``values`` yields ``(1, [])``. Raises
    :class:`~opop.model.ir.UnsupportedModelError` if any value is not exactly
    representable (see :func:`to_exact_fraction`) or if ``scale`` / any scaled
    coefficient would exceed :data:`MAX_INT_MAGNITUDE`.
    """
    fracs = [
        to_exact_fraction(v, max_denominator=max_denominator, tol=tol, what=what) for v in values
    ]
    scale = math.lcm(*(f.denominator for f in fracs)) if fracs else 1
    if scale > MAX_INT_MAGNITUDE:
        raise UnsupportedModelError(
            f"{what}: common denominator {scale} exceeds CP-SAT's safe integer magnitude "
            + f"{MAX_INT_MAGNITUDE}; coefficients need too much precision to scale exactly"
        )
    ints: list[int] = []
    for frac in fracs:
        scaled = frac.numerator * (scale // frac.denominator)
        if abs(scaled) > MAX_INT_MAGNITUDE:
            raise UnsupportedModelError(
                f"{what}: scaled integer {scaled} (= {frac} * {scale}) exceeds CP-SAT's safe "
                + f"integer magnitude {MAX_INT_MAGNITUDE}; reduce coefficient precision/range"
            )
        ints.append(scaled)
    return scale, ints
