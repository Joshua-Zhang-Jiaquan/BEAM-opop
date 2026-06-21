"""Evaluation CLI namespace.

Thin façade exposing the comparison-report tooling (implemented in
:mod:`opop.experiments.compare`) under the ``opop.eval`` name so the plan's
``python -m opop.eval.compare`` entry point resolves.  No logic lives here.
"""

from __future__ import annotations

from opop.experiments.compare import ComparisonReport, compare, load_results

__all__ = ["ComparisonReport", "compare", "load_results"]
