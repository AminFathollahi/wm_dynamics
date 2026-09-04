import json
from pathlib import Path

import pytest

from src import project_config


@pytest.fixture(autouse=True)
def clear_config_caches():
    project_config.load_project_config.cache_clear()
    project_config.load_dataset_registry.cache_clear()
    yield
    project_config.load_project_config.cache_clear()
    project_config.load_dataset_registry.cache_clear()


def test_repo_local_paths_resolve_from_config():
    assert project_config.project_path("results") == project_config.REPO_ROOT / "results"
    assert project_config.project_path("datasets_registry").is_file()


def test_data_root_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WM_DYNAMICS_DATA_ROOT", str(tmp_path))
    assert project_config.data_root() == tmp_path
    assert project_config.dataset_path("dandi_000469") == tmp_path / "000469"


def test_missing_optional_data_root_is_explicit(monkeypatch, tmp_path):
    config = json.loads((project_config.REPO_ROOT / "config" / "project.json").read_text())
    config["paths"]["data_root"] = None
    custom = tmp_path / "project.json"
    custom.write_text(json.dumps(config))
    monkeypatch.setenv("WM_DYNAMICS_CONFIG", str(custom))
    monkeypatch.delenv("WM_DYNAMICS_DATA_ROOT", raising=False)
    assert project_config.data_root(required=False) is None
    with pytest.raises(project_config.ConfigurationError, match="data_root"):
        project_config.data_root()


def test_production_code_has_no_machine_specific_data_paths():
    forbidden = "/media/amin/EXTERNAL_USB"
    for directory in (project_config.REPO_ROOT / "src", project_config.REPO_ROOT / "scripts"):
        for path in directory.glob("*.py"):
            assert forbidden not in path.read_text(), path
