"""``python -m opop.experiments`` -> the comparison-report CLI."""

from __future__ import annotations

import sys

from .compare import main

if __name__ == "__main__":
    sys.exit(main())
