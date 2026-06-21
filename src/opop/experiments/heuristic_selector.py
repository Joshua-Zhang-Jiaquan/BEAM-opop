"""LLM-based heuristic selection/evolution for the LLM-enhanced CO baseline (task 38).

Reproduces the LLM-LNS / HeurAgenix-style *heuristic selection* step: given a
compact summary of a MILP instance and the search history so far, an LLM is
asked to pick ONE matheuristic core to run next (plus a light configuration)
from a FIXED, safe vocabulary::

    {local_branching, rins, lns, repair}

The LLM only ever *selects a name from this closed set* — exactly as the typed
proposer selection does (task 14). Anything outside the set, a malformed
payload, or a parse failure deterministically FALLS BACK to a default core, so a
hallucinated or broken reply can never inject an arbitrary heuristic. The
companion runner in :mod:`opop.experiments.baselines_56` maps the chosen name to
the corresponding :mod:`opop.solver.heuristics` core and applies it; iterating
the selection over several rounds (the choice may change as the incumbent
improves) is the "evolution" half of the baseline.

This module is pure + network-free at import and call time: it talks only to an
injected :class:`~opop.llm.client.LLMClient` (a
:class:`~opop.llm.client.FakeLLMClient` in tests), never to opop's analyzer /
verifier / controller.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from opop.llm.client import LLMClient, LLMParseError

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_HEURISTICS",
    "DEFAULT_HEURISTIC",
    "HeuristicChoice",
    "build_selection_prompt",
    "normalize_heuristic_name",
    "sanitize_config",
    "select_heuristic",
]

#: The closed, safe vocabulary the LLM may select from.
ALLOWED_HEURISTICS: tuple[str, ...] = ("local_branching", "rins", "lns", "repair")

#: The deterministic fallback used when the LLM reply is unusable.
DEFAULT_HEURISTIC: str = "lns"

# Common LLM spellings mapped onto the canonical vocabulary.
_ALIASES: dict[str, str] = {
    "local_branching": "local_branching",
    "localbranching": "local_branching",
    "local_branch": "local_branching",
    "lb": "local_branching",
    "rins": "rins",
    "relaxation_induced_neighborhood_search": "rins",
    "lns": "lns",
    "large_neighborhood_search": "lns",
    "large_neighbourhood_search": "lns",
    "repair": "repair",
    "repair_solution": "repair",
}

# Config keys the cores understand; every other key is dropped.
_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset({"k", "destroy_frac", "n_iter", "agreement_tol"})

# System prompt describing the strict JSON contract (single line — no adjacent
# string literals, to keep the basedpyright zero-diagnostic bar).
_SYSTEM_PROMPT: str = (
    "You are a matheuristic selection policy for mixed-integer programming."
    " Given an instance summary and the search history, choose exactly ONE"
    " neighbourhood heuristic to run next and a light configuration for it."
    " Reply with a single JSON object of the form"
    ' {"heuristic": "local_branching|rins|lns|repair",'
    ' "config": {"k": <int>, "destroy_frac": <float>, "n_iter": <int>},'
    ' "rationale": "<short string>"}.'
)


def normalize_heuristic_name(name: Any) -> str | None:
    """Map a raw heuristic name onto the canonical vocabulary, or ``None``.

    Lower-cases, trims, and collapses hyphens/spaces to underscores before an
    alias lookup. Returns ``None`` for a non-string or an unrecognised name (the
    caller then falls back to its default).
    """
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return _ALIASES.get(key)


def sanitize_config(config: Any) -> dict[str, float]:
    """Keep only known numeric config keys (coerced to ``float``); drop the rest.

    Booleans are rejected (``bool`` is an ``int`` subclass but never a valid
    numeric config value), and any non-mapping input yields an empty config.
    """
    if not isinstance(config, Mapping):
        return {}
    raw: Mapping[str, Any] = config
    clean: dict[str, float] = {}
    for key, value in raw.items():
        if (
            key in _KNOWN_CONFIG_KEYS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            clean[str(key)] = float(value)
    return clean


@dataclass(frozen=True, slots=True)
class HeuristicChoice:
    """One LLM heuristic-selection decision.

    Attributes:
        heuristic: The chosen core name (ALWAYS in :data:`ALLOWED_HEURISTICS`).
        config: Sanitised numeric configuration for the core (may be empty).
        rationale: The LLM's free-text justification (``""`` if absent/invalid).
        fell_back: ``True`` iff the LLM reply was unusable and the default core
            was substituted.
        raw: The raw parsed LLM payload (for audit), or ``{}`` on a parse error.
    """

    heuristic: str
    config: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    fell_back: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary (excludes the raw payload)."""
        return {
            "heuristic": self.heuristic,
            "config": dict(self.config),
            "rationale": self.rationale,
            "fell_back": self.fell_back,
        }


def build_selection_prompt(instance_summary: Mapping[str, Any]) -> str:
    """Render a deterministic selection prompt from an instance + search summary.

    The summary is serialised as sorted-key JSON so identical inputs always yield
    an identical prompt (and thus a deterministic FakeLLM token estimate).
    """
    summary_json = json.dumps(dict(instance_summary), sort_keys=True, default=str)
    allowed = ", ".join(ALLOWED_HEURISTICS)
    lines = [
        "Select the next matheuristic core to improve the incumbent.",
        f"Allowed heuristics: {allowed}.",
        f"Instance and search summary (JSON): {summary_json}",
        "Respond with the JSON object described in the system message.",
    ]
    return "\n".join(lines)


def select_heuristic(
    llm: LLMClient,
    instance_summary: Mapping[str, Any],
    *,
    default: str = DEFAULT_HEURISTIC,
    temperature: float = 0.0,
) -> HeuristicChoice:
    """Ask ``llm`` to pick one core from :data:`ALLOWED_HEURISTICS`.

    Builds a deterministic prompt, calls ``llm.chat_json``, and validates the
    reply. The chosen ``heuristic`` is ALWAYS in the allowed set: an unknown
    name, a malformed payload, or a parse failure falls back to ``default``
    (``fell_back=True``) so a hallucinated reply can never inject an arbitrary
    core.

    Args:
        llm: Any :class:`~opop.llm.client.LLMClient` (a ``FakeLLMClient`` in tests).
        instance_summary: Compact instance + search-history features for the prompt.
        default: Fallback core name when the reply is unusable.
        temperature: Sampling temperature forwarded to the client.

    Returns:
        A validated :class:`HeuristicChoice`.
    """
    safe_default = normalize_heuristic_name(default) or DEFAULT_HEURISTIC
    prompt = build_selection_prompt(instance_summary)
    try:
        payload = llm.chat_json(prompt, system=_SYSTEM_PROMPT, temperature=temperature)
    except LLMParseError as exc:
        logger.debug("heuristic selection parse error: %s", exc)
        return HeuristicChoice(heuristic=safe_default, fell_back=True)

    raw: dict[str, Any] = dict(payload)
    chosen = normalize_heuristic_name(raw.get("heuristic"))
    if chosen is None:
        logger.debug("heuristic selection returned no valid name: %r", raw.get("heuristic"))
        return HeuristicChoice(heuristic=safe_default, fell_back=True, raw=raw)

    config = sanitize_config(raw.get("config"))
    rationale_obj = raw.get("rationale", "")
    rationale = rationale_obj if isinstance(rationale_obj, str) else ""
    return HeuristicChoice(heuristic=chosen, config=config, rationale=rationale, raw=raw)
