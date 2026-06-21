"""Tests for ``opop.llm.client``.

All tests are network-free: the real backends are only instantiated and
configured, never called. Chat round-trips use :class:`FakeLLMClient`.
"""

import json

import pytest

from opop.llm.client import (
    FakeLLMClient,
    LLMParseError,
    OpenAICompatClient,
    TokenTracker,
    VLLMClient,
)


class TestTokenTracker:
    """Unit tests for ``TokenTracker`` accounting."""

    def test_per_call_record(self) -> None:
        tracker = TokenTracker(price_input_1m=1.0, price_output_1m=2.0)
        rec = tracker.record(tokens_in=1000, tokens_out=500)

        assert rec["tokens_in"] == 1000
        assert rec["tokens_out"] == 500
        # cost = (1000*1 + 500*2) / 1e6
        assert rec["cost_usd"] == pytest.approx(0.002)

    def test_cumulative_totals(self) -> None:
        tracker = TokenTracker(price_input_1m=1.0, price_output_1m=1.0)
        _ = tracker.record(100, 50)
        _ = tracker.record(200, 100)

        assert tracker.calls == 2
        assert tracker.total_tokens_in == 300
        assert tracker.total_tokens_out == 150
        assert tracker.total_cost_usd == pytest.approx(0.00045)

    def test_summary(self) -> None:
        tracker = TokenTracker(price_input_1m=0.0, price_output_1m=0.0)
        _ = tracker.record(10, 20)
        summary = tracker.summary()

        assert summary["calls"] == 1
        assert summary["total_tokens_in"] == 10
        assert summary["total_tokens_out"] == 20
        assert summary["total_tokens"] == 30
        assert summary["total_cost_usd"] == 0.0


class TestParseJson:
    """Tests for JSON extraction via ``FakeLLMClient.chat_json``."""

    def _client(self, response: str) -> FakeLLMClient:
        return FakeLLMClient(response=response)

    def test_direct_parse(self) -> None:
        assert self._client('{"answer": 42}').chat_json("q") == {"answer": 42}

    def test_markdown_code_block(self) -> None:
        text = "Some explanation\n```json\n{\"x\": 1}\n```"
        assert self._client(text).chat_json("q") == {"x": 1}

    def test_outer_braces(self) -> None:
        text = "prefix {\"a\": 1} suffix"
        assert self._client(text).chat_json("q") == {"a": 1}

    def test_malformed_raises_llm_parse_error(self) -> None:
        with pytest.raises(LLMParseError):
            _ = self._client("not json at all").chat_json("q")

    def test_llm_parse_error_is_value_error(self) -> None:
        """Backward-compatible: callers catching ``ValueError`` still work."""
        with pytest.raises(ValueError):
            _ = self._client("broken").chat_json("q")


class TestFakeLLMClient:
    """Round-trip tests using the deterministic fake client."""

    def test_chat_returns_canned_response(self) -> None:
        client = FakeLLMClient(response="hello back")
        assert client.chat("hi") == "hello back"

    def test_chat_uses_callable_response(self) -> None:
        client = FakeLLMClient(response=lambda msg: f"echo: {msg}")
        assert client.chat("ping") == "echo: ping"

    def test_chat_json_parses_response(self) -> None:
        payload = {"delta": "add_cover_cut", "reason": "LP gap is large"}
        client = FakeLLMClient(response=json.dumps(payload))
        assert client.chat_json("propose") == payload

    def test_chat_json_parses_markdown_block(self) -> None:
        payload = {"x": 1}
        client = FakeLLMClient(response=f"```json\n{json.dumps(payload)}\n```")
        assert client.chat_json("propose") == payload

    def test_chat_json_malformed_raises_typed_error(self) -> None:
        client = FakeLLMClient(response="this is not json")
        with pytest.raises(LLMParseError):
            client.chat_json("propose")

    def test_tracker_accounting_after_chat(self) -> None:
        client = FakeLLMClient(
            response="one two three",
            price_input_1m=1.0,
            price_output_1m=2.0,
        )
        _ = client.chat("hello world", system="sys")

        # Word counts: system=1, message=2, response=3
        assert client.tracker.total_tokens_in == 3
        assert client.tracker.total_tokens_out == 3
        assert client.tracker.total_cost_usd == pytest.approx(9e-06)
        assert client.tracker.calls == 1


class TestOpenAICompatClient:
    """Configuration tests that never hit the network."""

    def test_env_fallback_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "OPOP_API_KEY",
            "OPENAI_API_KEY",
            "OPOP_BASE_URL",
            "OPENAI_BASE_URL",
            "OPOP_MODEL",
            "OPENAI_MODEL",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("OPENAI_MODEL", "fallback-model")

        client = OpenAICompatClient()
        assert client.api_key == "fallback-key"
        assert client.base_url == "https://api.example.com/v1"
        assert client.model == "fallback-model"

    def test_opop_env_overrides_openai_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "OPOP_API_KEY",
            "OPENAI_API_KEY",
            "OPOP_BASE_URL",
            "OPENAI_BASE_URL",
            "OPOP_MODEL",
            "OPENAI_MODEL",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("OPOP_API_KEY", "opop-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPOP_MODEL", "custom-model")

        client = OpenAICompatClient()
        assert client.api_key == "opop-key"
        assert client.model == "custom-model"

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPOP_API_KEY", "env-key")
        client = OpenAICompatClient(api_key="arg-key", model="arg-model")
        assert client.api_key == "arg-key"
        assert client.model == "arg-model"

    def test_missing_api_key_raises_on_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPOP_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = OpenAICompatClient()
        with pytest.raises(RuntimeError):
            _ = client.chat("hi")


class TestVLLMClient:
    """Smoke tests for the vLLM thin specialization."""

    def test_default_local_base_url(self) -> None:
        client = VLLMClient(model="llama-3-8b")
        assert client.base_url == "http://localhost:8000/v1"
        assert client.model == "llama-3-8b"

    def test_opop_base_url_env_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPOP_BASE_URL", "http://vllm.cluster:8000/v1")
        client = VLLMClient()
        assert client.base_url == "http://vllm.cluster:8000/v1"
