"""Best-effort MIPLIB 2017 subset downloader + checksum-verified loader.

A *small curated subset* (12 instances) of the public MIPLIB 2017 collection,
identified **by name + SHA-256 checksum**. We never commit the raw instance
blobs — only the download script and the published checksums live here. Files
are fetched on demand into a local cache (``benchmarks/_cache/miplib2017`` by
default, which is git-ignored) and verified against their recorded checksum
before use.

Network is **best effort**: if downloads are unavailable, the synthetic
generators in :mod:`opop.bench.sources.synthetic` provide a fully offline,
deterministic Phase-1 dev set on their own. Loading an instance returns a
:class:`opop.model.ir.MILP` via :func:`opop.model.ir.read_mps` (SCIP reads the
gzipped ``.mps.gz`` directly).

The recorded ``sha256`` values were captured from the official mirror
``https://miplib.zib.de/WebData/instances/<name>.mps.gz`` and every listed
instance round-trips into the linear MILP IR.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from opop.bench.registry import BenchmarkEntry
from opop.model.ir import MILP, read_mps

__all__ = [
    "MIPLIB_BASE_URL",
    "MIPLIB_HELDOUT_OOD",
    "MIPLIB_HELDOUT_SUBSET",
    "MIPLIB_HELDOUT_TEST",
    "MIPLIB_LICENSE",
    "MIPLIB_PHASE1_SUBSET",
    "MiplibChecksumError",
    "MiplibDownloadError",
    "MiplibError",
    "MiplibInstance",
    "build_heldout_entries",
    "default_cache_dir",
    "download_miplib_instance",
    "download_miplib_subset",
    "ensure_miplib_instance",
    "instance_by_name",
    "load_miplib_instance",
    "miplib_url",
    "network_available",
    "sha256_bytes",
    "sha256_file",
    "subset_manifest_checksum",
    "verify_checksum",
]

MIPLIB_BASE_URL = "https://miplib.zib.de/WebData/instances/"
MIPLIB_LICENSE = "MIPLIB2017-public"
_USER_AGENT = "opop-bench/0.1 (Phase-1 dev set acquisition)"


class MiplibError(Exception):
    """Base class for MIPLIB acquisition failures."""


class MiplibDownloadError(MiplibError):
    """Raised when an instance cannot be fetched (network/HTTP failure)."""


class MiplibChecksumError(MiplibError):
    """Raised when a downloaded file does not match its recorded checksum."""


@dataclass(frozen=True, slots=True)
class MiplibInstance:
    """One curated MIPLIB 2017 instance.

    Attributes:
        name: Instance name (the ``<name>`` in ``<name>.mps.gz``); also its
            registry instance id.
        sha256: Hex SHA-256 of the gzipped ``.mps.gz`` file (no ``sha256:`` prefix).
        n_bytes: Size of the gzipped file in bytes (informational).
    """

    name: str
    sha256: str
    n_bytes: int

    @property
    def filename(self) -> str:
        """The on-disk / remote filename (``<name>.mps.gz``)."""
        return f"{self.name}.mps.gz"


# Captured from https://miplib.zib.de/WebData/instances/ — small instances that
# round-trip cleanly into the linear MILP IR (no range rows / nonlinear terms).
_RAW_SUBSET: tuple[tuple[str, str, int], ...] = (
    ("flugpl", "d0b817caf496c95336f64815a740d6a54db1e37924785fcfd3c5da3ee6a15640", 889),
    ("markshare_4_0", "1a1feb04f5637db9f69be53bd6224fcd9fe5b8d750870d9b2002b535d25f4353", 797),
    ("markshare_5_0", "5309e16f917ec25e9f70402a8c039d4595a91c9499f682b1483babfff5f55b4a", 1174),
    ("enlight_hard", "942168c2126a2a91ae3ec1ededea59bc1af0cad55f94223edf4c03d20e831f66", 3872),
    ("noswot", "f4ac4805801d06bd12e7f67204101928413f7df27fa347bfca1c918e29bca86b", 4492),
    ("pk1", "6c64530538b254a8ae648f8f8f188847366ae52da560fdec0ad290c642973ba0", 4005),
    ("neos5", "2888e9aeb10a4cf16d3b9ad221302fd3de815ce1454b31e660ebd060589c1072", 5686),
    ("gen-ip054", "b5089ebc97c43bd583fcadf30955b8800861044d632d747c0b7ac73f19b9e6f3", 5884),
    ("gen-ip016", "ba1599099bccf0cf78cbc7e4c181f03a3af877b1ba596dac67ad296c81220bee", 7444),
    ("mas76", "91f624659e181538020e0c627910f02ffa6b355d6df708ef324e697a69485a74", 9186),
    ("gen-ip002", "cb37e723fa2187947981215bf9c8ba40d452984d7b9be3467e9f9d18f0343f78", 10577),
    ("gen-ip036", "cdb97b0176d4e46c12755eba12568048d9650cc26d09b93478bf64bbf5682cf7", 14459),
)
MIPLIB_PHASE1_SUBSET: tuple[MiplibInstance, ...] = tuple(
    MiplibInstance(name, sha256, n_bytes) for name, sha256, n_bytes in _RAW_SUBSET
)


# Held-out MIPLIB 2017 instances (name + SHA-256 from the ZIB mirror), disjoint in
# name AND family from MIPLIB_PHASE1_SUBSET. mas74 is omitted on purpose: it shares
# the mas family with Phase-1's free-split mas76, which would leak across splits.
_RAW_HELDOUT: tuple[tuple[str, str, int], ...] = (
    ("22433", "bf1d9bf9427c43e02b1fd93e7e0c8298f71b5c7a6beaa440b58eeb725b6b86f2", 13997),
    ("assign1-5-8", "305c26c9eee8da7242f2f42504cb0988c9503f65972859ad5f26d9615086cde3", 12566),
    ("b-ball", "67c5cb2dc43704d11f456537f6a951cb8d5bfb6d74c90fb1827a3bc34abf3586", 1381),
    ("dcmulti", "83627542fa87eb04e4eb22ce805d3bcde64e2a58f3caa3eafac7004816c470df", 8983),
    ("gr4x6", "ecb539938284a6453b8b94f47dfdd88d635e47cfd2977365f7246bb62ed562c7", 925),
    ("ran14x18-disj-8", "2a063d948b69e080e8712c449000bf8f26b05ff4f619552a35b06ca5b6eb38c8", 58880),
    ("timtab1", "5867d1a9c1067a849211c1a61bdb8839fb0025255dde09989d0ed333e0674d7a", 7077),
)
MIPLIB_HELDOUT_SUBSET: tuple[MiplibInstance, ...] = tuple(
    MiplibInstance(name, sha256, n_bytes) for name, sha256, n_bytes in _RAW_HELDOUT
)

MIPLIB_HELDOUT_TEST: tuple[str, ...] = ("22433", "b-ball", "dcmulti", "gr4x6")
MIPLIB_HELDOUT_OOD: tuple[str, ...] = ("assign1-5-8", "ran14x18-disj-8", "timtab1")

_HELDOUT_TIME_LIMIT_SEC = 300.0
_HELDOUT_PHASE = 6
_HELDOUT_THESIS = "T1"
_HELDOUT_BASELINE = "scip_default"


def default_cache_dir() -> Path:
    """Return the default MIPLIB cache directory (``<repo>/benchmarks/_cache/miplib2017``)."""
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "benchmarks" / "_cache" / "miplib2017"


def miplib_url(name: str) -> str:
    """Return the official download URL for instance ``name``."""
    return f"{MIPLIB_BASE_URL}{name}.mps.gz"


def instance_by_name(name: str) -> MiplibInstance:
    """Return the curated :class:`MiplibInstance` for ``name``.

    Searches both the Phase-1 dev/validation subset and the Wave-6 held-out
    collection (the two are name-disjoint).

    Raises:
        MiplibError: If ``name`` is in neither curated subset.
    """
    for inst in (*MIPLIB_PHASE1_SUBSET, *MIPLIB_HELDOUT_SUBSET):
        if inst.name == name:
            return inst
    known = ", ".join(i.name for i in (*MIPLIB_PHASE1_SUBSET, *MIPLIB_HELDOUT_SUBSET))
    raise MiplibError(f"unknown MIPLIB instance {name!r}; curated subset: {known}")


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path`` (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: str | Path, expected_sha256: str) -> bool:
    """Return ``True`` iff the file at ``path`` matches ``expected_sha256``.

    A leading ``sha256:`` prefix on ``expected_sha256`` is tolerated.
    """
    expected = expected_sha256.split(":", 1)[-1].strip().lower()
    return sha256_file(path).lower() == expected


def network_available(*, url: str | None = None, timeout: float = 5.0) -> bool:
    """Probe whether the MIPLIB mirror is reachable (no exception on failure)."""
    probe = url or miplib_url(MIPLIB_PHASE1_SUBSET[0].name)
    request = urllib.request.Request(probe, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _fetch(url: str, *, timeout: float) -> bytes:
    """Fetch ``url`` over HTTPS, returning its bytes (raises on failure)."""
    if not url.startswith("https://"):
        raise MiplibDownloadError(f"refusing non-HTTPS URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
            return response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise MiplibDownloadError(f"failed to download {url}: {exc}") from exc


def download_miplib_instance(
    instance: MiplibInstance | str,
    dest_dir: str | Path,
    *,
    timeout: float = 30.0,
    verify: bool = True,
) -> Path:
    """Download one curated instance into ``dest_dir`` and verify its checksum.

    Args:
        instance: A :class:`MiplibInstance` or a curated instance name.
        dest_dir: Directory to write ``<name>.mps.gz`` into (created if absent).
        timeout: Per-request timeout in seconds.
        verify: Verify the SHA-256 after download (raises on mismatch).

    Returns:
        Path to the downloaded ``.mps.gz`` file.

    Raises:
        MiplibDownloadError: On any network/HTTP failure.
        MiplibChecksumError: If ``verify`` and the checksum does not match.
    """
    inst = instance if isinstance(instance, MiplibInstance) else instance_by_name(instance)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / inst.filename

    data = _fetch(miplib_url(inst.name), timeout=timeout)
    actual = sha256_bytes(data)
    if verify and actual.lower() != inst.sha256.lower():
        raise MiplibChecksumError(
            f"checksum mismatch for {inst.name}: expected {inst.sha256}, got {actual}"
        )
    target.write_bytes(data)
    return target


def download_miplib_subset(
    dest_dir: str | Path | None = None,
    *,
    names: tuple[str, ...] | None = None,
    timeout: float = 30.0,
    verify: bool = True,
) -> list[Path]:
    """Download the curated subset (best effort), returning paths that succeeded.

    Network failures are swallowed per instance (so a partial mirror still
    yields whatever is reachable); a checksum mismatch is *not* swallowed
    (data integrity is non-negotiable).

    Args:
        dest_dir: Cache directory (defaults to :func:`default_cache_dir`).
        names: Optional subset of instance names (defaults to all curated).
        timeout: Per-request timeout in seconds.
        verify: Verify each download's checksum.

    Returns:
        Paths to successfully downloaded files (may be empty if offline).

    Raises:
        MiplibChecksumError: If a fetched file fails checksum verification.
    """
    dest = Path(dest_dir) if dest_dir is not None else default_cache_dir()
    selected = (
        MIPLIB_PHASE1_SUBSET
        if names is None
        else tuple(instance_by_name(n) for n in names)
    )
    downloaded: list[Path] = []
    for inst in selected:
        try:
            downloaded.append(
                download_miplib_instance(inst, dest, timeout=timeout, verify=verify)
            )
        except MiplibDownloadError:
            continue
    return downloaded


def ensure_miplib_instance(
    instance: MiplibInstance | str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    timeout: float = 30.0,
) -> Path:
    """Return a checksum-verified path to ``instance``, downloading if needed.

    If a cached file exists and matches its checksum it is reused. A cached file
    that fails verification is treated as corrupt and re-downloaded (when
    ``allow_download``).

    Raises:
        MiplibError: If the instance is absent and ``allow_download`` is False,
            or a download/checksum failure occurs.
    """
    inst = instance if isinstance(instance, MiplibInstance) else instance_by_name(instance)
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    target = cache / inst.filename

    if target.exists() and verify_checksum(target, inst.sha256):
        return target
    if not allow_download:
        raise MiplibError(
            f"MIPLIB instance {inst.name!r} not cached at {target} and downloads disabled; "
            + "run download_miplib_subset() with network access"
        )
    return download_miplib_instance(inst, cache, timeout=timeout, verify=True)


def load_miplib_instance(
    instance: MiplibInstance | str,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> MILP:
    """Load a curated MIPLIB instance into a :class:`MILP` (downloading if needed)."""
    path = ensure_miplib_instance(
        instance, cache_dir=cache_dir, allow_download=allow_download
    )
    return read_mps(str(path))


def subset_manifest_checksum(
    subset: tuple[MiplibInstance, ...] = MIPLIB_PHASE1_SUBSET,
) -> str:
    """Return a ``sha256:`` manifest checksum over the subset's ``name=sha256`` lines.

    This is the per-entry registry checksum for the MIPLIB family: it locks both
    *which* instances are included and *their content hashes*.
    """
    lines = "\n".join(f"{inst.name}={inst.sha256}" for inst in sorted(subset, key=lambda i: i.name))
    return "sha256:" + hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _heldout_entry(name: str, split: str, instance_names: tuple[str, ...]) -> BenchmarkEntry:
    instances = tuple(instance_by_name(n) for n in instance_names)
    return BenchmarkEntry(
        name=name,
        problem_type="MILP",
        source="miplib2017",
        split={split: instance_names},
        license=MIPLIB_LICENSE,
        instance_count=len(instance_names),
        time_limit_sec=_HELDOUT_TIME_LIMIT_SEC,
        baseline_set=_HELDOUT_BASELINE,
        leakage_group=name,
        checksum=subset_manifest_checksum(instances),
        phase=_HELDOUT_PHASE,
        thesis=_HELDOUT_THESIS,
    )


def build_heldout_entries() -> list[BenchmarkEntry]:
    """Return the registry entries for the held-out MIPLIB 2017 collection.

    Two entries, each its own ``leakage_group`` and confined to a single held-out
    split (``test`` / ``ood_test``); the per-entry checksum locks the included
    instances and their content hashes.
    """
    return [
        _heldout_entry("miplib2017_collection_test", "test", MIPLIB_HELDOUT_TEST),
        _heldout_entry("miplib2017_collection_ood", "ood_test", MIPLIB_HELDOUT_OOD),
    ]
