"""Tests for scripts/build_session_admission_accuracy_census.py.

Verifies the floor/patient aggregation arithmetic on synthetic session
accuracies, with _session_accuracy monkeypatched so no real NWB file or
data root is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_session_admission_accuracy_census as census  # noqa: E402


def _fake_registry():
    return [
        {"release": "000469", "patient": "P1", "session": "sub-1_ses-2", "path": "000469/sub-1/sub-1_ses-2.nwb"},
        {"release": "000469", "patient": "P1", "session": "sub-1_ses-1", "path": "000469/sub-1/sub-1_ses-1.nwb"},
        {"release": "000469", "patient": "P2", "session": "sub-2_ses-2", "path": "000469/sub-2/sub-2_ses-2.nwb"},
        {"release": "000469", "patient": "P3", "session": "sub-3_ses-2", "path": "000469/sub-3/sub-3_ses-2.nwb"},
    ]


def test_zero_below_floor_when_all_above(monkeypatch):
    monkeypatch.setattr(census, "_session_accuracy",
                         lambda root, row, field, group: {"status": "computed", "n_trials": 40, "accuracy": 0.9})
    result = census._census_corpus(Path("/fake"), "dandi_000469", _fake_registry())
    assert result["n_sessions_total"] == 3
    assert result["n_sessions_below_floor"] == 0
    assert result["patients_with_at_least_one_session_below_floor"] == []
    assert result["zero_drop_reconciliation"]["reconciles"] is True
    assert result["n_sessions_excluded_as_non_working_memory_task"] == 1


def test_below_floor_session_names_its_patient(monkeypatch):
    accuracies = {"sub-1_ses-2": 0.9, "sub-2_ses-2": 0.4, "sub-3_ses-2": 0.9}

    def fake_accuracy(root, row, field, group):
        return {"status": "computed", "n_trials": 40, "accuracy": accuracies[row["session"]]}

    monkeypatch.setattr(census, "_session_accuracy", fake_accuracy)
    result = census._census_corpus(Path("/fake"), "dandi_000469", _fake_registry())
    assert result["n_sessions_below_floor"] == 1
    assert result["n_patients_with_at_least_one_session_below_floor"] == 1
    assert result["patients_with_at_least_one_session_below_floor"] == ["P2"]
    assert result["zero_drop_reconciliation"]["reconciles"] is True
