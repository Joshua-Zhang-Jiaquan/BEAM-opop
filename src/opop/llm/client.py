"""Swappable LLM client layer for OPOP.

Provides a protocol-backed adapter over OpenAI-compatible chat-completion
endpoints (remote API or local vLLM) plus a deterministic fake client for tests.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, ClassVar, Protocol, override

import requests

logger = logging.getLogger(__name__)


class LLMParseError(ValueError):
    """Raised when ``chat_json`` cannot parse a JSON payload from the reply."""

    raw: str
    cause: Exception | None

    def __init__(self, message: str, raw: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.cause = cause

    @override
    def __str__(self) -> str:
        preview = self.raw[:200].replace("\n", " ")
        return f"{self.args[0]} (raw: {preview!r})"


class TokenTracker:
    """Records per-call LLM token usage and cumulative totals.

    Costs are computed from per-1M-token prices supplied at construction.
    """

    price_input_1m: float
    price_output_1m: float
    records: list[dict[str, Any]]
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    calls: int

    def __init__(
        self,
        price_input_1m: float = 0.0,
        price_output_1m: float = 0.0,
    ) -> None:
        self.price_input_1m = price_input_1m
        self.price_output_1m = price_output_1m
        self.records = []
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost_usd = 0.0
        self.calls = 0

    def record(self, tokens_in: int, tokens_out: int) -> dict[str, Any]:
        """Append one call and update cumulative counters."""
        cost = (
            tokens_in * self.price_input_1m + tokens_out * self.price_output_1m
        ) / 1e6
        rec: dict[str, Any] = {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost,
        }
        self.records.append(rec)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_cost_usd += cost
        self.calls += 1
        return rec

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of cumulative usage."""
        return {
            "calls": self.calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "total_cost_usd": self.total_cost_usd,
        }


class LLMClient(Protocol):
    """Protocol for swappable LLM backends."""

    @property
    def tracker(self) -> TokenTracker:
        """Token/cost tracker attached to this client."""
        ...

    def chat(
        self,
        message: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Send a chat message and return the response text."""
        ...

    def chat_json(
        self,
        message: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat message and parse the JSON payload of the reply."""
        ...


def _env_config() -> dict[str, str | None]:
    """Read OPOP / OPENAI env vars with the required fallback chain."""
    return {
        "api_key": os.environ.get("OPOP_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPOP_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        "model": os.environ.get("OPOP_MODEL") or os.environ.get("OPENAI_MODEL"),
    }


def _build_messages(message: str, system: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    return messages


def _parse_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from ``text``.

    Tries direct parsing, then a markdown ``json`` block, then the outermost
    ``{...}`` span. Raises :class:`LLMParseError` on failure.
    """
    # Direct parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as exc:
        logger.debug("Direct JSON parse failed: %s", exc)

    # Markdown code block.
    marker = "```json"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.find("```", start)
        if end != -1:
            try:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as exc:
                logger.debug("JSON-block parse failed: %s", exc)

    # Outermost braces.
    if "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as exc:
            logger.debug("Brace-span parse failed: %s", exc)

    raise LLMParseError("Could not parse JSON from LLM response", raw=text)


def _estimate_tokens(text: str) -> int:
    """Naive word-count estimator for fake/testing clients.

    Not intended to match a real tokenizer; only gives deterministic counts.
    """
    return len(text.split()) if text else 0


class OpenAICompatClient:
    """OpenAI-compatible chat-completion client using ``requests``.

    Configuration is resolved from explicit arguments first, then environment
    variables with the fallback chain
    ``OPOP_API_KEY -> OPENAI_API_KEY``,
    ``OPOP_BASE_URL -> OPENAI_BASE_URL``,
    ``OPOP_MODEL -> OPENAI_MODEL``,
    and finally the class defaults.
    """

    DEFAULT_BASE_URL: ClassVar[str] = "https://api.openai.com/v1"
    DEFAULT_MODEL: ClassVar[str] = "gpt-4o-mini"
    DEFAULT_PRICE_INPUT_1M: ClassVar[float] = 2.5
    DEFAULT_PRICE_OUTPUT_1M: ClassVar[float] = 10.0

    api_key: str | None
    base_url: str
    model: str
    timeout: float
    tracker: TokenTracker

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        tracker: TokenTracker | None = None,
        price_input_1m: float | None = None,
        price_output_1m: float | None = None,
        timeout: float = 120.0,
    ) -> None:
        cfg = _env_config()
        self.api_key = api_key if api_key is not None else cfg["api_key"]
        self.base_url = (
            base_url if base_url is not None else cfg["base_url"] or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model if model is not None else cfg["model"] or self.DEFAULT_MODEL
        self.timeout = timeout
        self.tracker = tracker or TokenTracker(
            price_input_1m=price_input_1m if price_input_1m is not None else self.DEFAULT_PRICE_INPUT_1M,
            price_output_1m=price_output_1m if price_output_1m is not None else self.DEFAULT_PRICE_OUTPUT_1M,
        )

    def chat(
        self,
        message: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Send a chat request and return the assistant's text reply."""
        if not self.api_key:
            raise RuntimeError(
                "OpenAICompatClient requires an API key. Set OPOP_API_KEY or OPENAI_API_KEY, or pass api_key=."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(message, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        content: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        _ = self.tracker.record(int(prompt_tokens or 0), int(completion_tokens or 0))
        return content

    def chat_json(
        self,
        message: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request and parse the JSON in the reply."""
        text = self.chat(
            message,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return _parse_json(text)


class VLLMClient(OpenAICompatClient):
    """Thin specialization of :class:`OpenAICompatClient` for a local vLLM server.

    vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so the
    request path is identical; only the default ``base_url`` changes.
    """

    DEFAULT_BASE_URL: ClassVar[str] = "http://localhost:8000/v1"

    def __init__(
        self,
        api_key: str | None = "not-needed-for-vllm",
        base_url: str | None = None,
        model: str | None = None,
        tracker: TokenTracker | None = None,
        price_input_1m: float | None = None,
        price_output_1m: float | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            tracker=tracker,
            price_input_1m=price_input_1m if price_input_1m is not None else 0.0,
            price_output_1m=price_output_1m if price_output_1m is not None else 0.0,
            timeout=timeout,
        )


class FakeLLMClient:
    """Deterministic, network-free LLM client for tests and offline development.

    The canned ``response`` is returned verbatim by :meth:`chat`.
    :meth:`chat_json` parses the same response as JSON. Token counts are
    estimated with a simple word-count heuristic so that ``TokenTracker``
    accounting can be exercised without a real tokenizer.
    """

    response: str | Callable[[str], str]
    model: str
    tracker: TokenTracker

    def __init__(
        self,
        response: str | Callable[[str], str] = "",
        model: str = "fake",
        tracker: TokenTracker | None = None,
        price_input_1m: float = 0.0,
        price_output_1m: float = 0.0,
    ) -> None:
        self.response = response
        self.model = model
        self.tracker = tracker or TokenTracker(
            price_input_1m=price_input_1m,
            price_output_1m=price_output_1m,
        )

    def _render(self, message: str) -> str:
        if callable(self.response):
            return self.response(message)
        return self.response

    def chat(
        self,
        message: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **_kwargs: Any,
    ) -> str:
        """Return the canned response and record token usage."""
        _ = (temperature, max_tokens, _kwargs)
        text = self._render(message)
        tokens_in = _estimate_tokens(system) + _estimate_tokens(message)
        tokens_out = _estimate_tokens(text)
        _ = self.tracker.record(tokens_in, tokens_out)
        return text

    def chat_json(
        self,
        message: str,
        *,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Return the canned response parsed as JSON."""
        text = self.chat(
            message,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **_kwargs,
        )
        return _parse_json(text)
