"""Reproduction test for scripts/build_maintenance_stimulation_prediction.py:
its output must equal the delivered results/maintenance_stimulation_prediction.json
on every number, plus carry a producing_script field the stored file lacks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_maintenance_stimulation_prediction import PRODUCING_SCRIPT, build  # noqa: E402

RESULTS = ROOT / "results"
STORED_PATH = RESULTS / "maintenance_stimulation_prediction.json"


def _assert_equal_on_every_number(rebuilt, stored, path=""):
    if isinstance(stored, dict):
        for key, stored_value in stored.items():
            assert key in rebuilt, f"missing key {path}.{key}"
            _assert_equal_on_every_number(rebuilt[key], stored_value, f"{path}.{key}")
    elif isinstance(stored, list):
        assert len(rebuilt) == len(stored), f"length mismatch at {path}"
        for i, (rebuilt_item, stored_item) in enumerate(zip(rebuilt, stored)):
            _assert_equal_on_every_number(rebuilt_item, stored_item, f"{path}[{i}]")
    elif isinstance(stored, (int, float)) and not isinstance(stored, bool):
        assert math.isclose(rebuilt, stored, rel_tol=1e-12, abs_tol=1e-12), \
            f"{path}: rebuilt {rebuilt!r} != stored {stored!r}"
    else:
        assert rebuilt == stored, f"{path}: rebuilt {rebuilt!r} != stored {stored!r}"


def test_rebuilt_record_equals_stored_file_on_every_number():
    if not STORED_PATH.exists():
        pytest.skip(f"{STORED_PATH} not delivered")
    stored = json.loads(STORED_PATH.read_text())
    rebuilt = build(RESULTS)
    _assert_equal_on_every_number(rebuilt, stored)
    assert rebuilt["producing_script"] == PRODUCING_SCRIPT
    assert "producing_script" not in stored
