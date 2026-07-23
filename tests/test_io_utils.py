"""Tests for src/io_utils.py."""

import json
import multiprocessing
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from io_utils import locked_json_update


def test_creates_file_if_missing(tmp_path):
    target = tmp_path / "stats.json"
    with locked_json_update(target) as stats:
        stats["a"] = 1
    assert json.loads(target.read_text()) == {"a": 1}


def test_preserves_existing_keys(tmp_path):
    target = tmp_path / "stats.json"
    target.write_text(json.dumps({"existing": "value"}))
    with locked_json_update(target) as stats:
        stats["new_key"] = 2
    result = json.loads(target.read_text())
    assert result == {"existing": "value", "new_key": 2}


def test_yields_mutable_dict_reflected_on_write(tmp_path):
    target = tmp_path / "stats.json"
    with locked_json_update(target) as stats:
        stats["nested"] = {"x": [1, 2, 3]}
    assert json.loads(target.read_text())["nested"]["x"] == [1, 2, 3]


def _writer(path_str: str, key: str, n_iters: int) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from io_utils import locked_json_update
    for i in range(n_iters):
        with locked_json_update(Path(path_str)) as stats:
            stats[key] = stats.get(key, 0) + 1


def test_concurrent_writers_do_not_lose_updates(tmp_path):
    target = tmp_path / "stats.json"
    target.write_text(json.dumps({}))
    n_iters = 20
    procs = [
        multiprocessing.Process(target=_writer, args=(str(target), f"counter_{k}", n_iters))
        for k in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    result = json.loads(target.read_text())
    for k in range(4):
        assert result[f"counter_{k}"] == n_iters
