"""Single source of truth for repository paths and shared runtime defaults.

Configuration is read from ``config/project.json`` (or
``WM_DYNAMICS_CONFIG``). Environment variables declared by that file are
machine-local overrides, so absolute paths never need to be committed.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV = "WM_DYNAMICS_CONFIG"


class ConfigurationError(RuntimeError):
    """Raised when required project configuration is absent or malformed."""


def _config_path() -> Path:
    value = os.environ.get(CONFIG_ENV)
    return Path(value).expanduser() if value else REPO_ROOT / "config" / "project.json"


@lru_cache(maxsize=1)
def load_project_config() -> dict[str, Any]:
    path = _config_path()
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Project config not found: {path}") from exc
    if config.get("schema_version") != "1.0.0":
        raise ConfigurationError(f"Unsupported project config schema in {path}")
    return config


def _resolved_path(value: str | Path, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else relative_to / path


def project_path(name: str) -> Path:
    """Resolve a repository-local configured path, honoring its env override."""
    config = load_project_config()
    try:
        value = config["paths"][name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown configured path: {name}") from exc
    env_name = config.get("environment", {}).get(name)
    if env_name and os.environ.get(env_name):
        value = os.environ[env_name]
    if not value:
        suffix = f" (or set {env_name})" if env_name else ""
        raise ConfigurationError(f"Configure paths.{name} in {_config_path()}{suffix}")
    return _resolved_path(value, relative_to=REPO_ROOT)


def data_root(*, required: bool = True) -> Path | None:
    """Return the external data root; optionally return ``None`` if unset."""
    try:
        return project_path("data_root")
    except ConfigurationError:
        if required:
            raise
        return None


@lru_cache(maxsize=1)
def load_dataset_registry() -> dict[str, Any]:
    return json.loads(project_path("datasets_registry").read_text())


def dataset_path(name: str, *parts: str, required: bool = True) -> Path | None:
    """Resolve a dataset key from the registry below the configured data root."""
    root = data_root(required=required)
    if root is None:
        return None
    registry = load_dataset_registry()
    try:
        relative = registry["datasets"][name]["local_path"]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown dataset registry key: {name}") from exc
    return root / relative / Path(*parts)


def data_asset_path(name: str, *, required: bool = True) -> Path | None:
    """Resolve a configured non-dataset asset below the external data root."""
    root = data_root(required=required)
    if root is None:
        return None
    try:
        relative = load_project_config()["paths"][name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown configured data asset: {name}") from exc
    return root / relative


def executable(name: str, *, required: bool = False) -> str | None:
    """Resolve a configured auxiliary interpreter or executable."""
    config = load_project_config()
    env_name = config.get("environment", {}).get(name)
    value = os.environ.get(env_name, "") if env_name else ""
    value = value or config.get("executables", {}).get(name)
    if required and not value:
        raise ConfigurationError(f"Configure executables.{name} in {_config_path()}")
    return value


def default(name: str) -> Any:
    """Read a shared runtime default by name."""
    try:
        return load_project_config()["defaults"][name]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown configured default: {name}") from exc
