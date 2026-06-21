"""Distributional MIPLIB (D-MIPLIB) downloader + checksum-verified loader.

A curated, **held-out** slice of `Distributional MIPLIB
<https://huggingface.co/datasets/weiminhu/D-MIPLIB>`_ (Huang et al., 2024), the
multi-domain library of MILP *distributions* at graded hardness levels. We never
commit the raw instance data — only the published per-file SHA-256 checksums and
the download recipe live here. Each registered unit is one *distribution* (the
``<domain>/<hardness>/test.csv`` standardized test set), fetched on demand into a
git-ignored cache (``benchmarks/_cache/dmiplib`` by default) and verified against
its recorded checksum before use.

Provenance: the recorded ``sha256`` of every distribution is the Git-LFS object id
published by the Hugging Face dataset (``lfs.oid``), captured WITHOUT downloading
the (multi-GB) files. A downloaded file is accepted only if its content SHA-256
equals that id.

Splits follow the *hardness shift* convention: ``easy``/``medium`` distributions
form the ``test`` split, ``hard``/``very-hard``/``ext-hard`` form ``ood_test``.
Each domain is its own ``leakage_group`` so no domain spans a free split.

A D-MIPLIB CSV stores instances as text strings; :func:`load_distribution_instance`
materializes one row into a :class:`opop.model.ir.MILP` (SCIP reads the embedded
LP/MPS text). This path needs the cached file + SCIP; the registry/checksum/split
machinery is fully offline.
"""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from opop.bench.registry import BenchmarkEntry
from opop.bench.sources.miplib import sha256_file, verify_checksum
from opop.model.ir import MILP, read_problem

__all__ = [
    "DMIPLIB_BASE_URL",
    "DMIPLIB_LICENSE",
    "DMIPLIB_SUBSET",
    "DMiplibChecksumError",
    "DMiplibDistribution",
    "DMiplibDownloadError",
    "DMiplibError",
    "build_entries",
    "default_cache_dir",
    "distribution_by_id",
    "dmiplib_url",
    "download_distribution",
    "download_subset",
    "ensure_distribution",
    "load_distribution_instance",
    "network_available",
    "subset_manifest_checksum",
]

DMIPLIB_BASE_URL = "https://huggingface.co/datasets/weiminhu/D-MIPLIB/resolve/main/"
DMIPLIB_LICENSE = "CC-BY-4.0"
_USER_AGENT = "opop-bench/0.1 (Wave-6 held-out acquisition)"

_TIME_LIMIT_SEC = 600.0
_PHASE = 6
_THESIS = "T1"
_BASELINE = "scip_default"


class DMiplibError(Exception):
    """Base class for D-MIPLIB acquisition failures."""


class DMiplibDownloadError(DMiplibError):
    """Raised when a distribution cannot be fetched (network/HTTP failure)."""


class DMiplibChecksumError(DMiplibError):
    """Raised when a downloaded file does not match its recorded checksum."""


@dataclass(frozen=True, slots=True)
class DMiplibDistribution:
    """One D-MIPLIB distribution (a standardized ``test.csv`` test set).

    Attributes:
        domain: Problem domain (e.g. ``MIS``, ``MVC``, ``SC``, ``CA``, ``GISP``).
        hardness: Hardness level (``easy``/``medium``/``hard``/``very-hard``/``ext-hard``).
        opop_split: The OPOP held-out split this distribution belongs to
            (``test`` or ``ood_test``).
        sha256: Hex SHA-256 of the ``test.csv`` file (the published LFS object id).
        n_bytes: Size of the ``test.csv`` file in bytes.
    """

    domain: str
    hardness: str
    opop_split: str
    sha256: str
    n_bytes: int

    @property
    def id(self) -> str:
        """Globally unique registry instance id."""
        return f"dmiplib/{self.domain}/{self.hardness}"

    @property
    def remote_path(self) -> str:
        """Path of the ``test.csv`` under the dataset root."""
        return f"{self.domain}/{self.hardness}/test.csv"

    @property
    def filename(self) -> str:
        """Flattened local cache filename."""
        return f"{self.domain}__{self.hardness}__test.csv"


# Per-distribution SHA-256 = the Hugging Face Git-LFS object id of each test.csv,
# captured from the dataset tree API. Hardness drives the held-out split:
# easy/medium -> test, hard/very-hard/ext-hard -> ood_test.
_RAW_DISTRIBUTIONS: tuple[tuple[str, str, str, str, int], ...] = (
    ("MIS", "easy", "test", "159be90f6a7eca06802b453e6faef215cf97091c483a6251d98406e346921fc8", 11645302),
    ("MIS", "medium", "test", "7880b0d7444d6adb791555aff13e31ef3484f3f64c19312aea173e3cc802e742", 17986498),
    ("MIS", "very-hard", "ood_test", "2308316d86777a1217483163fde1a8553654f16c414b73f6206b13ab70b4b593", 401956700),
    ("MVC", "easy", "test", "e9997f68e4d8e1d55ee04defbef7193c175bb266f5a51192ca5f99d7dc74c382", 23890825),
    ("MVC", "medium", "test", "b8a7e59e11d30878e34f169e44848d7e6fbf72617f8a3e513a37a03fe1fac273", 40724149),
    ("MVC", "hard", "ood_test", "a5d692ce2a9a5482dafb8f38ac4b7c9669441f1b44897a92951a2080f8948b87", 95870514),
    ("MVC", "very-hard", "ood_test", "343afaac1cf8459a1ea3d915160c0e481c8b577e9b84326ce77ed65b58cef0e7", 942948134),
    ("SC", "easy", "test", "c03209d687179217edf1d6960fc528e1754ef9771a07604383b3fe58b44b1759", 21697178),
    ("SC", "medium", "test", "df30e7f46363859e5f09ab0b01f6205f78bc65f4389d7c54473d750631000b49", 42029689),
    ("SC", "hard", "ood_test", "3c8fce63b549f62f2611a4f02b215e02d0f121328850ab4c06ebb8473b48ae22", 82794614),
    ("SC", "very-hard", "ood_test", "e8b6b1d31adefd39d10b259644f71359c49d18e49c5317a7085be8f8e891ca23", 884862959),
    ("CA", "medium", "test", "90a16110803e33ea3f7cf2c1263966635401f9f5bfcabe7e878ee548fcab3dcc", 11775967),
    ("CA", "very-hard", "ood_test", "9b2a0eb28e165a2b7aa97b56ab0749038f31f9f0a1c4deb4eda6c0d3a2b856fb", 33601974),
    ("GISP", "easy", "test", "77a681032152926a0d52b2bd1c6ddd9b42d08d0fbfe80cd0759d35479a73b48c", 10996940),
    ("GISP", "medium", "test", "434e528d72ad26f372504408c49243afa52d8504e6565c25172df2f8e9d32e9b", 69977994),
    ("GISP", "hard", "ood_test", "3f1eabc9d33fafa2160d2411e35852f2ab30924e5fb3c9eab89531facea27106", 95906745),
    ("GISP", "very-hard", "ood_test", "f61d24b94e3498a82c345d1c7059957aadda07a1890820241b20baf8d214a59d", 32333037),
    ("GISP", "ext-hard", "ood_test", "27af16d8fc502041d43e9c97c03783f7d9bf17c96cfd591ac7863eab38b90595", 1605466345),
    ("IP", "very-hard", "ood_test", "4abb06ad76d723defbd3826bdbb48bef8f8dcabdbbaae9f6c626c2a62ed39c0c", 40122675),
)
DMIPLIB_SUBSET: tuple[DMiplibDistribution, ...] = tuple(
    DMiplibDistribution(domain, hardness, opop_split, sha256, n_bytes)
    for domain, hardness, opop_split, sha256, n_bytes in _RAW_DISTRIBUTIONS
)


def default_cache_dir() -> Path:
    """Return the default D-MIPLIB cache directory (``<repo>/benchmarks/_cache/dmiplib``)."""
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "benchmarks" / "_cache" / "dmiplib"


def dmiplib_url(distribution: DMiplibDistribution) -> str:
    """Return the Hugging Face resolve URL for ``distribution``'s ``test.csv``."""
    return f"{DMIPLIB_BASE_URL}{distribution.remote_path}"


def distribution_by_id(instance_id: str) -> DMiplibDistribution:
    """Return the curated :class:`DMiplibDistribution` for ``instance_id``.

    Raises:
        DMiplibError: If ``instance_id`` is not in :data:`DMIPLIB_SUBSET`.
    """
    for dist in DMIPLIB_SUBSET:
        if dist.id == instance_id:
            return dist
    known = ", ".join(d.id for d in DMIPLIB_SUBSET)
    raise DMiplibError(f"unknown D-MIPLIB distribution {instance_id!r}; curated subset: {known}")


def network_available(*, timeout: float = 5.0) -> bool:
    """Probe whether the Hugging Face dataset mirror is reachable."""
    probe = dmiplib_url(DMIPLIB_SUBSET[0])
    request = urllib.request.Request(probe, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _fetch(url: str, dest: Path, *, timeout: float) -> None:
    """Stream ``url`` over HTTPS into ``dest`` (raises on failure)."""
    if not url.startswith("https://"):
        raise DMiplibDownloadError(f"refusing non-HTTPS URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
            with open(dest, "wb") as handle:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    handle.write(chunk)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        dest.unlink(missing_ok=True)
        raise DMiplibDownloadError(f"failed to download {url}: {exc}") from exc


def download_distribution(
    distribution: DMiplibDistribution | str,
    dest_dir: str | Path,
    *,
    timeout: float = 120.0,
    verify: bool = True,
) -> Path:
    """Download one distribution's ``test.csv`` into ``dest_dir`` and verify its checksum.

    Raises:
        DMiplibDownloadError: On any network/HTTP failure.
        DMiplibChecksumError: If ``verify`` and the checksum does not match.
    """
    dist = (
        distribution
        if isinstance(distribution, DMiplibDistribution)
        else distribution_by_id(distribution)
    )
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / dist.filename

    _fetch(dmiplib_url(dist), target, timeout=timeout)
    if verify and not verify_checksum(target, dist.sha256):
        actual = sha256_file(target)
        target.unlink(missing_ok=True)
        raise DMiplibChecksumError(
            f"checksum mismatch for {dist.id}: expected {dist.sha256}, got {actual}"
        )
    return target


def download_subset(
    dest_dir: str | Path | None = None,
    *,
    ids: tuple[str, ...] | None = None,
    timeout: float = 120.0,
    verify: bool = True,
) -> list[Path]:
    """Download the curated subset (best effort), returning paths that succeeded.

    Per-distribution network failures are swallowed (a partial mirror still yields
    what is reachable); a checksum mismatch is never swallowed.
    """
    dest = Path(dest_dir) if dest_dir is not None else default_cache_dir()
    selected = (
        DMIPLIB_SUBSET if ids is None else tuple(distribution_by_id(i) for i in ids)
    )
    downloaded: list[Path] = []
    for dist in selected:
        try:
            downloaded.append(
                download_distribution(dist, dest, timeout=timeout, verify=verify)
            )
        except DMiplibDownloadError:
            continue
    return downloaded


def ensure_distribution(
    distribution: DMiplibDistribution | str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    timeout: float = 120.0,
) -> Path:
    """Return a checksum-verified path to ``distribution``'s ``test.csv``.

    Raises:
        DMiplibError: If the file is absent and ``allow_download`` is False, or a
            download/checksum failure occurs.
    """
    dist = (
        distribution
        if isinstance(distribution, DMiplibDistribution)
        else distribution_by_id(distribution)
    )
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    target = cache / dist.filename

    if target.exists() and verify_checksum(target, dist.sha256):
        return target
    if not allow_download:
        raise DMiplibError(
            f"D-MIPLIB distribution {dist.id!r} not cached at {target} and downloads disabled; "
            + "run download_subset() with network access"
        )
    return download_distribution(dist, cache, timeout=timeout, verify=True)


def _decode_milp_cell(cell: str) -> str:
    """Decode a D-MIPLIB ``MILP`` cell (a ``b'...'`` byte-repr) into instance text."""
    text = cell
    if text[:2] in ("b'", 'b"') and text[-1:] in ("'", '"'):
        text = text[2:-1]
    return text.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def load_distribution_instance(
    distribution: DMiplibDistribution | str,
    *,
    index: int = 0,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> MILP:
    """Materialize one instance (row ``index``) of ``distribution`` into a :class:`MILP`.

    Downloads + verifies the ``test.csv`` if needed, extracts the ``index``-th
    instance text, and reads it through SCIP.

    Raises:
        DMiplibError: If the row/column is missing or the instance text is empty.
    """
    dist = (
        distribution
        if isinstance(distribution, DMiplibDistribution)
        else distribution_by_id(distribution)
    )
    path = ensure_distribution(dist, cache_dir=cache_dir, allow_download=allow_download)

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    row: dict[str, str] | None = None
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, record in enumerate(reader):
            if i == index:
                row = {str(k): str(v) for k, v in record.items()}
                break
    if row is None:
        raise DMiplibError(f"{dist.id}: no instance at row index {index}")

    milp_text = _decode_milp_cell(row.get("MILP", ""))
    if not milp_text.strip():
        raise DMiplibError(f"{dist.id}: empty MILP text at row index {index}")
    fmt = row.get("format", "lp").strip().lower() or "lp"
    suffix = ".mps" if fmt == "mps" else ".lp"

    with tempfile.TemporaryDirectory() as tmp:
        instance_path = Path(tmp) / f"{dist.domain}_{dist.hardness}_{index}{suffix}"
        instance_path.write_text(milp_text, encoding="utf-8")
        return read_problem(str(instance_path))


def subset_manifest_checksum(
    subset: tuple[DMiplibDistribution, ...] = DMIPLIB_SUBSET,
) -> str:
    """Return a ``sha256:`` manifest checksum over the subset's ``id=sha256`` lines.

    Locks both *which* distributions are included and *their content hashes*.
    """
    lines = "\n".join(f"{d.id}={d.sha256}" for d in sorted(subset, key=lambda d: d.id))
    return "sha256:" + hashlib.sha256(lines.encode("utf-8")).hexdigest()


def build_entries() -> list[BenchmarkEntry]:
    """Return one registry entry per D-MIPLIB domain (each its own leakage_group).

    Within a domain the held-out splits are populated by hardness
    (easy/medium -> ``test``, harder -> ``ood_test``); the per-entry checksum locks
    the domain's distributions and their content hashes.
    """
    by_domain: dict[str, list[DMiplibDistribution]] = {}
    order: list[str] = []
    for dist in DMIPLIB_SUBSET:
        if dist.domain not in by_domain:
            by_domain[dist.domain] = []
            order.append(dist.domain)
        by_domain[dist.domain].append(dist)

    entries: list[BenchmarkEntry] = []
    for domain in order:
        dists = by_domain[domain]
        split: dict[str, list[str]] = {}
        for dist in dists:
            split.setdefault(dist.opop_split, []).append(dist.id)
        entries.append(
            BenchmarkEntry(
                name=f"dmiplib_{domain}",
                problem_type="MILP",
                source="dmiplib",
                split={name: tuple(ids) for name, ids in split.items()},
                license=DMIPLIB_LICENSE,
                instance_count=len(dists),
                time_limit_sec=_TIME_LIMIT_SEC,
                baseline_set=_BASELINE,
                leakage_group=f"dmiplib_{domain}",
                checksum=subset_manifest_checksum(tuple(dists)),
                phase=_PHASE,
                thesis=_THESIS,
            )
        )
    return entries
