"""MILPBench downloader + checksum-verified loader.

A curated, **held-out** slice of `MILPBench <https://github.com/thuiar/MILPBench>`_
(Ye et al., GECCO 2025), a large-scale MILP benchmark of 100k instances across 60
classes. The full categorized datasets are distributed as Google-Drive archives
(see :data:`MILPBENCH_GDRIVE`); fetching those requires ``gdown`` and is left to
the user. What we register + verify here is the small, real, in-repository
knapsack test subset (the baseline-library ``LP_test`` instances), each pinned by
name + SHA-256 and fetched on demand from ``raw.githubusercontent.com`` into a
git-ignored cache (``benchmarks/_cache/milpbench`` by default).

These instances populate the ``ood_test`` split: MILPBench's knapsack distribution
is a distinct generator from the synthetic Phase-1 knapsack dev family (different
scale/structure), so it is its own ``leakage_group`` and is genuinely
out-of-distribution relative to the free splits — it is NOT a near-duplicate of
any dev/validation instance. Files load into :class:`opop.model.ir.MILP` via SCIP
(LP format); the registry/checksum/split machinery is fully offline.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from opop.bench.registry import BenchmarkEntry
from opop.bench.sources.miplib import sha256_bytes, verify_checksum
from opop.model.ir import MILP, read_problem

__all__ = [
    "MILPBENCH_GDRIVE",
    "MILPBENCH_LICENSE",
    "MILPBENCH_RAW_BASE",
    "MILPBENCH_SUBSET",
    "MilpbenchChecksumError",
    "MilpbenchDownloadError",
    "MilpbenchError",
    "MilpbenchInstance",
    "build_entries",
    "default_cache_dir",
    "download_instance",
    "download_subset",
    "ensure_instance",
    "gdrive_url",
    "instance_by_id",
    "load_instance",
    "milpbench_url",
    "network_available",
    "subset_manifest_checksum",
]

MILPBENCH_RAW_BASE = "https://raw.githubusercontent.com/thuiar/MILPBench/main/"
MILPBENCH_LICENSE = "Apache-2.0"
_USER_AGENT = "opop-bench/0.1 (Wave-6 held-out acquisition)"
_KNAPSACK_DIR = "Baseline Library/GNN_MILP/Code/instances/knapsack/LP_test"

_TIME_LIMIT_SEC = 900.0
_PHASE = 6
_THESIS = "T1"
_BASELINE = "scip_default"

# Canonical full-dataset Google-Drive file ids (per ./Benchmark Datasets/README.md).
# These large archives are the authoritative MILPBench source; fetching them needs
# `gdown` (not a project dependency) and is out of scope for the verified subset.
MILPBENCH_GDRIVE: dict[str, str] = {
    "MIS_easy": "1slfuVvma5R5qwoFtIw1I3wLeIzg5EGvM",
    "MIS_medium": "1DOSR3rZ3ezwaMJAB-5aHtwKWoOngUQzH",
    "MIS_hard": "15ZkWUq5dysm-3D9VAL2kb1nr-Sgwjain",
    "MVC_easy": "10CCgHflKtxO4XOXZZCkD-pLU7vh81GZ0",
    "MVC_medium": "11Frntl0fDKf0bnbgvrZun_vHJxbTKxbu",
    "MVC_hard": "1y80fAwcty8e3yR93dR5whD6xx39_QLXE",
    "SC_easy": "1Oa9NiP6I1XpOkneLETGfKgTeYMDybVJX",
    "SC_medium": "1OOEiav-07UmCtCKOfnpWraN5Rxz980bk",
    "SC_hard": "1uJFOUz6Xr_qgrmXhZcisWUG0hw_fnCSV",
    "CAT_easy": "1sWsUkQdKYi50HYAutunFieRMcmckXqHr",
    "CAT_medium": "136Kte9O3-VslVJBHvYHt2ew-squRCFEV",
    "CAT_hard": "1lRRwC09rK5p8hH3adJiju4iL4cODTpOh",
    "CFL_easy": "1z6oNG1ja6CwlsRYViXIzBj0j8Ch6sxdt",
    "CFL_medium": "181Evo5Q6otZRq6EBeQXFcCYlC4kM8zaH",
    "CFL_hard": "13NS9YTTyNsiV6Dth3qsQ7lWWNQs4Pek0",
}


class MilpbenchError(Exception):
    """Base class for MILPBench acquisition failures."""


class MilpbenchDownloadError(MilpbenchError):
    """Raised when an instance cannot be fetched (network/HTTP failure)."""


class MilpbenchChecksumError(MilpbenchError):
    """Raised when a downloaded file does not match its recorded checksum."""


@dataclass(frozen=True, slots=True)
class MilpbenchInstance:
    """One MILPBench instance hosted in the repository.

    Attributes:
        problem_class: Problem class (currently ``knapsack``).
        name: Instance name (the ``<name>`` in ``<name>.lp``).
        repo_path: Path of the ``.lp`` file under the repository root.
        sha256: Hex SHA-256 of the ``.lp`` file content.
        n_bytes: Size of the ``.lp`` file in bytes.
    """

    problem_class: str
    name: str
    repo_path: str
    sha256: str
    n_bytes: int

    @property
    def id(self) -> str:
        """Globally unique registry instance id."""
        return f"milpbench/{self.problem_class}/{self.name}"

    @property
    def filename(self) -> str:
        """Flattened local cache filename."""
        return f"{self.problem_class}__{self.name}.lp"


# Real in-repo knapsack LP_test instances (content SHA-256 of the raw GitHub blob).
_RAW_KNAPSACK: tuple[tuple[str, str, int], ...] = (
    ("instance_46", "f532225aa1f1198f9bd31f1899d66de90da80468a39d89ccc9a171077284d208", 22645),
    ("instance_152", "e10b7ee3a1bd5dd14af0492f7d89569948fe036515c8f7c33c8e9545787a63e2", 22645),
    ("instance_270", "14fdbfe115a040fe661668d58f08d4f9dc86e6ec0d1e96dbca416a2ceb7570c4", 22645),
    ("instance_864", "3cd27c2ac92ebf5f9c1d104911d3567857e67fd2d8d1860287a24c7f186c006a", 22645),
    ("instance_875", "d8b8fe1e5a5d3c93be8cc1482853bd783248dc7a0300c2476e1d993f19c85d12", 22645),
)
MILPBENCH_SUBSET: tuple[MilpbenchInstance, ...] = tuple(
    MilpbenchInstance("knapsack", name, f"{_KNAPSACK_DIR}/{name}.lp", sha256, n_bytes)
    for name, sha256, n_bytes in _RAW_KNAPSACK
)


def default_cache_dir() -> Path:
    """Return the default MILPBench cache directory (``<repo>/benchmarks/_cache/milpbench``)."""
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "benchmarks" / "_cache" / "milpbench"


def milpbench_url(instance: MilpbenchInstance) -> str:
    """Return the raw GitHub URL for ``instance`` (path components URL-encoded)."""
    return MILPBENCH_RAW_BASE + urllib.parse.quote(instance.repo_path)


def gdrive_url(dataset: str) -> str:
    """Return the Google-Drive share URL for a full-dataset archive in :data:`MILPBENCH_GDRIVE`."""
    if dataset not in MILPBENCH_GDRIVE:
        known = ", ".join(sorted(MILPBENCH_GDRIVE))
        raise MilpbenchError(f"unknown MILPBench dataset {dataset!r}; documented: {known}")
    return f"https://drive.google.com/file/d/{MILPBENCH_GDRIVE[dataset]}/view?usp=sharing"


def instance_by_id(instance_id: str) -> MilpbenchInstance:
    """Return the curated :class:`MilpbenchInstance` for ``instance_id``.

    Raises:
        MilpbenchError: If ``instance_id`` is not in :data:`MILPBENCH_SUBSET`.
    """
    for inst in MILPBENCH_SUBSET:
        if inst.id == instance_id:
            return inst
    known = ", ".join(i.id for i in MILPBENCH_SUBSET)
    raise MilpbenchError(f"unknown MILPBench instance {instance_id!r}; curated subset: {known}")


def network_available(*, timeout: float = 5.0) -> bool:
    """Probe whether the raw GitHub host is reachable for the curated subset."""
    probe = milpbench_url(MILPBENCH_SUBSET[0])
    request = urllib.request.Request(probe, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _fetch(url: str, *, timeout: float) -> bytes:
    """Fetch ``url`` over HTTPS, returning its bytes (raises on failure)."""
    if not url.startswith("https://"):
        raise MilpbenchDownloadError(f"refusing non-HTTPS URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
            return response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise MilpbenchDownloadError(f"failed to download {url}: {exc}") from exc


def download_instance(
    instance: MilpbenchInstance | str,
    dest_dir: str | Path,
    *,
    timeout: float = 30.0,
    verify: bool = True,
) -> Path:
    """Download one curated instance into ``dest_dir`` and verify its checksum.

    Raises:
        MilpbenchDownloadError: On any network/HTTP failure.
        MilpbenchChecksumError: If ``verify`` and the checksum does not match.
    """
    inst = instance if isinstance(instance, MilpbenchInstance) else instance_by_id(instance)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / inst.filename

    data = _fetch(milpbench_url(inst), timeout=timeout)
    actual = sha256_bytes(data)
    if verify and actual.lower() != inst.sha256.lower():
        raise MilpbenchChecksumError(
            f"checksum mismatch for {inst.id}: expected {inst.sha256}, got {actual}"
        )
    target.write_bytes(data)
    return target


def download_subset(
    dest_dir: str | Path | None = None,
    *,
    ids: tuple[str, ...] | None = None,
    timeout: float = 30.0,
    verify: bool = True,
) -> list[Path]:
    """Download the curated subset (best effort), returning paths that succeeded.

    Per-instance network failures are swallowed; a checksum mismatch is never.
    """
    dest = Path(dest_dir) if dest_dir is not None else default_cache_dir()
    selected = MILPBENCH_SUBSET if ids is None else tuple(instance_by_id(i) for i in ids)
    downloaded: list[Path] = []
    for inst in selected:
        try:
            downloaded.append(download_instance(inst, dest, timeout=timeout, verify=verify))
        except MilpbenchDownloadError:
            continue
    return downloaded


def ensure_instance(
    instance: MilpbenchInstance | str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Return a checksum-verified path to ``instance``, downloading if needed.

    Raises:
        MilpbenchError: If the instance is absent and ``allow_download`` is False,
            or a download/checksum failure occurs.
    """
    inst = instance if isinstance(instance, MilpbenchInstance) else instance_by_id(instance)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    target = cache / inst.filename

    if target.exists() and verify_checksum(target, inst.sha256):
        return target
    if not allow_download:
        raise MilpbenchError(
            f"MILPBench instance {inst.id!r} not cached at {target} and downloads disabled; "
            + "run download_subset() with network access"
        )
    return download_instance(inst, cache, timeout=timeout, verify=True)


def load_instance(
    instance: MilpbenchInstance | str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> MILP:
    """Load a curated MILPBench instance into a :class:`MILP` (downloading if needed)."""
    path = ensure_instance(instance, cache_dir=cache_dir, allow_download=allow_download)
    return read_problem(str(path))


def subset_manifest_checksum(
    subset: tuple[MilpbenchInstance, ...] = MILPBENCH_SUBSET,
) -> str:
    """Return a ``sha256:`` manifest checksum over the subset's ``id=sha256`` lines.

    Locks both *which* instances are included and *their content hashes*.
    """
    lines = "\n".join(f"{i.id}={i.sha256}" for i in sorted(subset, key=lambda i: i.id))
    return "sha256:" + hashlib.sha256(lines.encode("utf-8")).hexdigest()


def build_entries() -> list[BenchmarkEntry]:
    """Return the registry entry for the held-out MILPBench knapsack subset.

    A single ``ood_test`` entry (its own ``leakage_group``); the per-entry checksum
    locks the included instances and their content hashes.
    """
    ids = tuple(inst.id for inst in MILPBENCH_SUBSET)
    return [
        BenchmarkEntry(
            name="milpbench_knapsack",
            problem_type="MILP",
            source="milpbench",
            split={"ood_test": ids},
            license=MILPBENCH_LICENSE,
            instance_count=len(ids),
            time_limit_sec=_TIME_LIMIT_SEC,
            baseline_set=_BASELINE,
            leakage_group="milpbench_knapsack",
            checksum=subset_manifest_checksum(),
            phase=_PHASE,
            thesis=_THESIS,
        )
    ]
