"""Tests for opop.config dataclass loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from opop.config import ConfigError, RunConfig, load_config


MINIMAL_CFG: dict[str, object] = {
    "name": "test-run",
    "split": "dev",
    "seeds": [0, 1],
    "solver": {"name": "scip", "threads": 2},
    "controller": {"method": "random", "n_init": 3},
    "budget": {"trials": 5, "time_limit_sec": 60.0},
}


def test_json_yaml_equivalence(tmp_path: Path) -> None:
    """Equivalent JSON and YAML files produce identical RunConfig objects."""
    json_path = tmp_path / "cfg.json"
    yaml_path = tmp_path / "cfg.yaml"

    json_path.write_text(json.dumps(MINIMAL_CFG), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(MINIMAL_CFG), encoding="utf-8")

    cfg_json = load_config(json_path)
    cfg_yaml = load_config(yaml_path)

    assert isinstance(cfg_json, RunConfig)
    assert isinstance(cfg_yaml, RunConfig)
    assert cfg_json == cfg_yaml


def test_env_override_applies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OPOP_<SECTION>_<FIELD> env vars override loaded config values."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(MINIMAL_CFG), encoding="utf-8")

    monkeypatch.setenv("OPOP_BUDGET_TRIALS", "7")
    monkeypatch.setenv("OPOP_SOLVER_THREADS", "4")

    cfg = load_config(cfg_path)

    assert cfg.budget.trials == 7
    assert cfg.solver.threads == 4


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    """An unknown top-level key raises ConfigError naming the bad key."""
    cfg_path = tmp_path / "cfg.yaml"
    bad_cfg = dict(MINIMAL_CFG)
    bad_cfg["bogus_field"] = 123
    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")

    with pytest.raises(ConfigError, match="bogus_field"):
        load_config(cfg_path)


def test_unknown_nested_key_raises(tmp_path: Path) -> None:
    """An unknown key inside a nested section raises ConfigError naming the key."""
    cfg_path = tmp_path / "cfg.yaml"
    bad_cfg = dict(MINIMAL_CFG)
    bad_cfg["solver"] = {"name": "scip", "unknown_solver_param": 42}
    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown_solver_param"):
        load_config(cfg_path)
