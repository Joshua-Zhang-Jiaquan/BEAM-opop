"""Typed dataclass config + JSON/YAML loader with env-var overrides."""

from __future__ import annotations

import json
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints


class ConfigError(ValueError):
    """Raised when config validation fails (unknown key, bad type, etc.)."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """Solver backend selection and parameters."""

    name: str = "scip"
    threads: int = 1


@dataclass
class ControllerConfig:
    """Bayesian controller parameters."""

    method: str = "random"
    n_init: int = 5


@dataclass
class BudgetConfig:
    """Resource budget for a single experiment run."""

    trials: int = 10
    time_limit_sec: float = 30.0


@dataclass
class RunConfig:
    """Top-level experiment configuration."""

    name: str = "opop-run"
    description: str = ""
    split: str = "dev"
    seeds: list[int] = field(default_factory=lambda: [0])
    output_dir: str = "runs"
    solver: SolverConfig = field(default_factory=SolverConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    #: Cap on the number of instances to run (first N); ``None`` means all.
    instance_limit: int | None = None


# ---------------------------------------------------------------------------
# Field metadata for typed env-var coercion
# ---------------------------------------------------------------------------

def _field_map() -> dict[str, tuple[str, Any, Any]]:
    """Build a flat mapping: env-key → (dataclass_attr, field_type, default)."""
    mapping: dict[str, tuple[str, Any, Any]] = {}
    for cls, prefix in [
        (SolverConfig, "SOLVER"),
        (ControllerConfig, "CONTROLLER"),
        (BudgetConfig, "BUDGET"),
    ]:
        hints = get_type_hints(cls)
        for f in fields(cls):
            env_key = f"OPOP_{prefix}_{f.name.upper()}"
            mapping[env_key] = (
                f"{prefix.lower()}.{f.name}",
                hints[f.name],
                _default_for_field(f),
            )
    # Top-level RunConfig fields (except nested dataclasses)
    run_hints = get_type_hints(RunConfig)
    for f in fields(RunConfig):
        if f.name in ("solver", "controller", "budget"):
            continue
        env_key = f"OPOP_{f.name.upper()}"
        mapping[env_key] = (f.name, run_hints[f.name], _default_for_field(f))
    return mapping


def _default_for_field(fld: Any) -> Any:
    if fld.default is not MISSING:
        return fld.default
    if fld.default_factory is not MISSING:
        return fld.default_factory()
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> RunConfig:
    """Load a RunConfig from a .json or .yaml file, applying OPOP_* env overrides.

    Raises ConfigError for unknown keys, missing files, or bad formats.
    """
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"config file not found: {source}")

    raw = _load_mapping(source)
    _check_unknown_keys(source, raw)

    # Apply env overrides
    raw = _apply_env_overrides(raw)

    # Build nested configs
    solver = SolverConfig(**_nested(raw, "solver", SolverConfig))
    controller = ControllerConfig(**_nested(raw, "controller", ControllerConfig))
    budget = BudgetConfig(**_nested(raw, "budget", BudgetConfig))

    top = {k: v for k, v in raw.items() if k not in ("solver", "controller", "budget")}
    return RunConfig(solver=solver, controller=controller, budget=budget, **top)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ConfigError(
                "YAML configs require PyYAML. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ConfigError(f"unsupported config format: {suffix} (use .json, .yaml, or .yml)")

    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def _nested(raw: dict[str, Any], key: str, cls: type) -> dict[str, Any]:
    """Extract and validate a nested section dict, falling back to defaults."""
    if key not in raw:
        return {}  # rely on dataclass defaults
    value = raw[key]
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(
            f"'{key}' must be a mapping (dict), got {type(value).__name__}"
        )
    section: dict[str, Any] = value
    _check_unknown_keys_in_section(key, section, cls)
    return dict(section)


def _all_known_keys(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _check_unknown_keys(path: Path, raw: dict[str, Any]) -> None:
    known = _all_known_keys(RunConfig)
    for k in raw:
        if k not in known:
            raise ConfigError(f"unknown top-level key '{k}' in {path.name}")


def _check_unknown_keys_in_section(section: str, raw: dict[str, Any], cls: type) -> None:
    known = _all_known_keys(cls)
    for k in raw:
        if k not in known:
            raise ConfigError(f"unknown key '{k}' in section '{section}'")


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply OPOP_<SECTION>_<FIELD> env-var overrides onto the raw dict."""
    fmap = _field_map()
    result = dict(raw)  # shallow copy
    for env_key, (attr_path, field_type, _default) in fmap.items():
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue
        coerced = _coerce_env(env_key, env_val, field_type)
        _set_nested(result, attr_path, coerced)
    return result


def _set_nested(d: dict[str, Any], attr_path: str, value: Any) -> None:
    parts = attr_path.split(".")
    current: Any = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _is_optional(tp: Any) -> bool:
    """``True`` if ``tp`` is an ``Optional[...]`` / ``X | None`` annotation."""
    args = getattr(tp, "__args__", None)
    return args is not None and type(None) in args


def _resolve_type(tp: Any) -> type:
    """Extract the concrete type from a generic / Optional annotation (list[int], int | None)."""
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        return list
    args = getattr(tp, "__args__", None)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_type(non_none[0])
    return tp if isinstance(tp, type) else origin or str


def _coerce_env(env_key: str, raw_val: str, field_type: Any) -> Any:
    """Coerce a string env-var value to *field_type* (int/float/list[int]/str/Optional)."""
    if _is_optional(field_type) and raw_val.strip().lower() in ("", "none", "null"):
        return None
    concrete = _resolve_type(field_type)
    try:
        if concrete is int:
            return int(raw_val)
        if concrete is float:
            return float(raw_val)
        if concrete is list:
            parsed = json.loads(raw_val)
            if not isinstance(parsed, list):
                raise ValueError("not a JSON array")
            typed: list[Any] = parsed
            # coerce inner elements if we know the type arg
            inner = _list_inner_type(field_type)
            if inner is int:
                return [int(x) for x in typed]
            if inner is float:
                return [float(x) for x in typed]
            return typed
        if concrete is bool:
            return raw_val.lower() in ("1", "true", "yes")
        return raw_val
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"cannot coerce {env_key}={raw_val!r} to {concrete.__name__}: {exc}") from exc


def _list_inner_type(tp: Any) -> Any:
    """Return the inner type of list[X] if available."""
    args = getattr(tp, "__args__", None)
    if args and len(args) == 1:
        return args[0]
    return str
