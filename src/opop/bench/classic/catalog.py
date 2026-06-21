"""Classic-CO registry catalog: committed fixtures → held-out registry entries (task 34).

This is the single source of truth that turns the small committed instance files
under ``tests/bench/fixtures/classic/`` into Wave-6 held-out
:class:`~opop.bench.registry.BenchmarkEntry` objects, consumed by the combined
registry generator (:func:`opop.bench.sources.milp_suites.build_all_entries`).

Each of the six classic families (TSP / CVRP / set covering / job shop / MaxSAT /
MaxCut) is ONE registry entry:

* tagged ``phase: 6`` and ``thesis: T3`` (generality breadth — the T3 thesis);
* confined to the immutable ``test`` held-out split (never dev/validation), with
  its own ``leakage_group`` so no group spans a free and a held-out split;
* checksummed by a ``sha256:`` manifest over its instances' file hashes (the
  ``CLASSIC_FIXTURES`` constants), exactly like the MIPLIB / MILPBench suites —
  so the committed fixtures are content-locked and a drift is caught by the
  checksum-integrity test.

The committed fixtures are small, public-domain, hand-crafted instances in each
library's native text format; the loaders in this package parse the SAME formats
for full-size downloads. This module is pure (registry + hashlib only); it never
imports the parsers/solvers, so registry generation stays light.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from opop.bench.registry import BenchmarkEntry

__all__ = [
    "CLASSIC_FAMILIES",
    "CLASSIC_FIXTURES",
    "ClassicFamily",
    "ClassicFixture",
    "build_classic_entries",
    "family_checksum",
]

#: Held-out time budget (seconds) for the classic-CO suites.
TIME_LIMIT_SEC = 300.0
#: Baseline tag — the open-solver default (no Gurobi).
BASELINE_SET = "scip_default"
#: Wave-6 phase tag.
PHASE = 6
#: Generality thesis (T3): breadth across CO problem classes.
THESIS = "T3"


@dataclass(frozen=True, slots=True)
class ClassicFixture:
    """One committed classic-CO instance file (content-locked by ``sha256``).

    Attributes:
        family: Classic-CO family id (``tsp`` / ``cvrp`` / ``orlib`` / ``jsp`` /
            ``maxsat`` / ``maxcut``).
        name: Instance name (the file stem).
        rel_path: Path relative to ``tests/bench/fixtures/classic/``.
        sha256: Hex SHA-256 of the committed file content.
        split: Held-out split the instance belongs to (``test`` / ``ood_test``).
    """

    family: str
    name: str
    rel_path: str
    sha256: str
    split: str

    @property
    def id(self) -> str:
        """Globally unique registry instance id (``classic/<family>/<name>``)."""
        return f"classic/{self.family}/{self.name}"


@dataclass(frozen=True, slots=True)
class ClassicFamily:
    """One classic-CO registry family (a single :class:`BenchmarkEntry`).

    Attributes:
        family: Family id (matches :attr:`ClassicFixture.family` and the
            ``classic-<family>`` adapter name).
        entry_name: Registry entry name / leakage_group (``classic_<family>``).
        problem_type: Human-readable CO class tag (``TSP`` / ``CVRP`` / ``SCP`` /
            ``JSP`` / ``MaxSAT`` / ``MaxCut``).
        source: Upstream library / format id.
        license: Provenance / license tag for the family.
        fixtures: The committed instances of this family.
    """

    family: str
    entry_name: str
    problem_type: str
    source: str
    license: str
    fixtures: tuple[ClassicFixture, ...]


_FAMILY_SUFFIX = {"tsp": "tsp", "cvrp": "vrp", "orlib": "txt", "jsp": "txt", "maxcut": "txt"}
_MAXSAT_SUFFIX = {"tiny": "wcnf", "small": "cnf"}


def _fx(family: str, name: str, sha256: str) -> ClassicFixture:
    """Build a ``test``-split :class:`ClassicFixture` with its standard rel path."""
    suffix = _MAXSAT_SUFFIX[name] if family == "maxsat" else _FAMILY_SUFFIX[family]
    return ClassicFixture(
        family=family,
        name=name,
        rel_path=f"{family}/{name}.{suffix}",
        sha256=sha256,
        split="test",
    )


#: The committed classic-CO instances, content-locked by their file SHA-256.
CLASSIC_FAMILIES: tuple[ClassicFamily, ...] = (
    ClassicFamily(
        family="tsp",
        entry_name="classic_tsp",
        problem_type="TSP",
        source="tsplib",
        license="TSPLIB",
        fixtures=(
            _fx("tsp", "tiny4", "34ce54db45763d853c080e35b9037435d0f01aa1d9cffbca59bf73b482e7c22a"),
            _fx("tsp", "explicit4", "b64070f89bbcd4acca8b383d5fb2cf0abc822745ab800cceb8e8d432c57e2618"),
        ),
    ),
    ClassicFamily(
        family="cvrp",
        entry_name="classic_cvrp",
        problem_type="CVRP",
        source="cvrplib",
        license="CVRPLIB",
        fixtures=(
            _fx("cvrp", "tiny5", "ab7ad8f1c2f57f9050905f76264e990803052f68140defa0a75ddfad31103574"),
            _fx("cvrp", "small6", "a3d84b82bdccc2110d0d1998c359502b63d7fad9eca22b46d66e7e285349482e"),
        ),
    ),
    ClassicFamily(
        family="orlib",
        entry_name="classic_orlib_scp",
        problem_type="SCP",
        source="or-library",
        license="OR-Library",
        fixtures=(
            _fx("orlib", "scp_tiny", "06b4a50b702606405e8150e0098a39980d4b24fdeb8edd019033e34d5cc2adb7"),
            _fx("orlib", "scp_small", "b7b1d66b180e5eff020ebe3815ae828f9222df24060b4591f9e671b4b00d71c8"),
        ),
    ),
    ClassicFamily(
        family="jsp",
        entry_name="classic_jsp",
        problem_type="JSP",
        source="jsplib",
        license="JSPLIB",
        fixtures=(
            _fx("jsp", "jsp2x2", "583dcda78f9f6c3cb35b01b2ed0dc3f8fabad6db1d7a6243528d619ea02af61a"),
            _fx("jsp", "jsp3x3", "1557146e81f770e0352f7671297fdbc0088f1384359ea3493bf3ffd4f0459437"),
        ),
    ),
    ClassicFamily(
        family="maxsat",
        entry_name="classic_maxsat",
        problem_type="MaxSAT",
        source="maxsat",
        license="MaxSAT-Evaluations",
        fixtures=(
            _fx("maxsat", "tiny", "e5dc818496bf778368e6d5e54b98074d9e3829cc5966dd815b2943e1b082a65c"),
            _fx("maxsat", "small", "bc54257114a0a76fcd8e345ad39693dcce1cabacf87acde0edfd594043bce504"),
        ),
    ),
    ClassicFamily(
        family="maxcut",
        entry_name="classic_maxcut",
        problem_type="MaxCut",
        source="maxcut",
        license="Gset/BiqMac",
        fixtures=(
            _fx("maxcut", "triangle", "b556e04c13957e905f01c08c77b59383e06075dde4c63929ad808cfc3df37b29"),
            _fx("maxcut", "square", "0dae027754a92c7a9ab78c5005e1c6ac96aada68058cb4cdcd0046bfb9a569bc"),
        ),
    ),
)

#: Flat tuple of every committed classic-CO fixture (used by the checksum test).
CLASSIC_FIXTURES: tuple[ClassicFixture, ...] = tuple(
    fx for fam in CLASSIC_FAMILIES for fx in fam.fixtures
)


def family_checksum(fixtures: tuple[ClassicFixture, ...]) -> str:
    """Return a ``sha256:`` manifest checksum over the fixtures' ``id=sha256`` lines.

    Locks both *which* instances are included and *their content hashes*
    (identical convention to the MIPLIB / MILPBench held-out suites).
    """
    lines = "\n".join(f"{fx.id}={fx.sha256}" for fx in sorted(fixtures, key=lambda f: f.id))
    return "sha256:" + hashlib.sha256(lines.encode("utf-8")).hexdigest()


def build_classic_entries() -> list[BenchmarkEntry]:
    """Return one held-out :class:`BenchmarkEntry` per classic-CO family.

    Every entry carries ``phase=6`` / ``thesis="T3"``, its own ``leakage_group``,
    and a single held-out split; the per-entry checksum locks the committed
    instances and their content hashes.
    """
    entries: list[BenchmarkEntry] = []
    for fam in CLASSIC_FAMILIES:
        split: dict[str, list[str]] = {}
        for fx in fam.fixtures:
            split.setdefault(fx.split, []).append(fx.id)
        entries.append(
            BenchmarkEntry(
                name=fam.entry_name,
                problem_type=fam.problem_type,
                source=fam.source,
                split={name: tuple(ids) for name, ids in split.items()},
                license=fam.license,
                instance_count=len(fam.fixtures),
                time_limit_sec=TIME_LIMIT_SEC,
                baseline_set=BASELINE_SET,
                leakage_group=fam.entry_name,
                checksum=family_checksum(fam.fixtures),
                phase=PHASE,
                thesis=THESIS,
            )
        )
    return entries
