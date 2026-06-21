"""Experiment tooling: comparison reports + statistical tests.

Public API for the Phase-1 comparison report (Wilcoxon signed-rank, shifted
geometric-mean time, min-effect gating).  CLI entry points:

* ``python -m opop.experiments`` (this package's ``__main__``)
* ``python -m opop.eval.compare`` (thin shim onto :func:`compare`)
"""

from __future__ import annotations

from .baselines import (
    RESULT_COLUMNS,
    BaselineRunner,
    DefaultRunner,
    ParamSpec,
    SMACTunedRunner,
    default_param_space,
    run_baselines,
    write_results,
)
from .compare import (
    DEFAULT_MIN_EFFECT,
    METRIC_LOWER_IS_BETTER,
    SEED_FLOOR,
    SHIFT,
    VALID_METRICS,
    ComparisonReport,
    build_min_effect,
    compare,
    format_report,
    load_results,
    main,
    shifted_geometric_mean,
    write_report,
)
from .fairness import (
    BudgetSpec,
    FairnessError,
    assert_tunable_split,
    check_budget_fairness,
)

__all__ = [
    "DEFAULT_MIN_EFFECT",
    "METRIC_LOWER_IS_BETTER",
    "RESULT_COLUMNS",
    "SEED_FLOOR",
    "SHIFT",
    "VALID_METRICS",
    "BaselineRunner",
    "BudgetSpec",
    "ComparisonReport",
    "DefaultRunner",
    "FairnessError",
    "ParamSpec",
    "SMACTunedRunner",
    "assert_tunable_split",
    "build_min_effect",
    "check_budget_fairness",
    "compare",
    "default_param_space",
    "format_report",
    "load_results",
    "main",
    "run_baselines",
    "shifted_geometric_mean",
    "write_report",
    "write_results",
]
