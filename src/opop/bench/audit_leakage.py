"""CLI entry point so ``python -m opop.bench.audit_leakage`` runs the leakage audit.

The audit logic lives in :mod:`opop.bench.audit`; this module only wires the
canonical module path (referenced across the plan + the ``events.jsonl`` schema)
to :func:`opop.bench.audit.main`.
"""

from __future__ import annotations

import sys

from opop.bench.audit import main

if __name__ == "__main__":
    sys.exit(main())
