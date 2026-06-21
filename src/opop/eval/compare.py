"""``python -m opop.eval.compare`` -> the comparison-report CLI.

This is a thin re-export shim; the implementation lives in
:mod:`opop.experiments.compare`.  Importing from here gives the same public
API, and running the module executes the same CLI.
"""

from __future__ import annotations

import sys

from opop.experiments.compare import (
    ComparisonReport,
    compare,
    format_report,
    load_results,
    main,
    shifted_geometric_mean,
    write_report,
)

__all__ = [
    "ComparisonReport",
    "compare",
    "format_report",
    "load_results",
    "main",
    "shifted_geometric_mean",
    "write_report",
]


if __name__ == "__main__":
    sys.exit(main())
