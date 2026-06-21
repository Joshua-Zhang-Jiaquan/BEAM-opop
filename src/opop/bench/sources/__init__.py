"""Phase-1 benchmark sources: synthetic generators + MIPLIB 2017 subset.

Public entry points:

* :func:`get_phase1_instances` / :func:`make_phase1_splits` — load and seal the
  Phase-1 dev/validation set (see :mod:`opop.bench.sources.phase1_set`).
* :func:`generate_set_cover` / :func:`generate_knapsack` / :func:`generate_facility`
  — deterministic synthetic MILP generators.
* :func:`download_miplib_subset` / :func:`load_miplib_instance` — best-effort,
  checksum-verified MIPLIB 2017 acquisition.
"""

from __future__ import annotations

from opop.bench.sources.miplib import (
    MIPLIB_PHASE1_SUBSET,
    MiplibChecksumError,
    MiplibDownloadError,
    MiplibError,
    MiplibInstance,
    download_miplib_instance,
    download_miplib_subset,
    load_miplib_instance,
    network_available,
    subset_manifest_checksum,
    verify_checksum,
)
from opop.bench.sources.phase1_set import (
    PHASE1_CATALOG,
    Phase1Error,
    Recipe,
    build_phase1_registry,
    build_registry_entries,
    get_phase1_instances,
    make_phase1_splits,
    materialize,
    write_registry_yaml,
)
from opop.bench.sources.synthetic import (
    canonical_milp_repr,
    generate_facility,
    generate_knapsack,
    generate_set_cover,
    milp_digest,
)

__all__ = [
    "MIPLIB_PHASE1_SUBSET",
    "PHASE1_CATALOG",
    "MiplibChecksumError",
    "MiplibDownloadError",
    "MiplibError",
    "MiplibInstance",
    "Phase1Error",
    "Recipe",
    "build_phase1_registry",
    "build_registry_entries",
    "canonical_milp_repr",
    "download_miplib_instance",
    "download_miplib_subset",
    "generate_facility",
    "generate_knapsack",
    "generate_set_cover",
    "get_phase1_instances",
    "load_miplib_instance",
    "make_phase1_splits",
    "materialize",
    "milp_digest",
    "network_available",
    "subset_manifest_checksum",
    "verify_checksum",
    "write_registry_yaml",
]
