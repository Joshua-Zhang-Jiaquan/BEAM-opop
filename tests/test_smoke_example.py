"""Example smoke test proving the global pytest harness works."""

from __future__ import annotations

from pathlib import Path

import pytest

import opop
from opop.llm import FakeLLMClient


@pytest.mark.smoke
def test_harness_works(
    fake_llm: FakeLLMClient,
    tmp_run_dir: Path,
    tiny_milp_fixture: dict[str, object],
) -> None:
    """A trivial passing test exercising the shared fixtures."""
    assert opop is not None
    assert fake_llm.chat("hello") == '{"answer": 42}'
    assert fake_llm.tracker.calls >= 1
    assert tmp_run_dir.exists()
    assert tiny_milp_fixture["known_optimum"] == 1.0
