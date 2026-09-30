from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_dandi_000004_recognition_generalization import _participant_rows
from corpus_sessions import recognition_correct


def test_recognition_response_mapping():
    labels = np.array(["0", "0", "1", "1"])
    responses = np.array([36, 31, 31, 35])
    assert recognition_correct(labels, responses).tolist() == [True, False, True, False]


def test_participant_rows_keep_repeated_sessions_in_one_inference_unit():
    sessions = [
        {"participant": "p1", "regions": {"hippocampus": {
            "status": "computed", "n_trials": 20, "raw_r": 0.1, "joint_partial_r": 0.2}}},
        {"participant": "p1", "regions": {"hippocampus": {
            "status": "computed", "n_trials": 40, "raw_r": 0.4, "joint_partial_r": 0.5}}},
        {"participant": "p2", "regions": {"hippocampus": {
            "status": "refused", "reason": "few units"}}},
    ]
    rows = _participant_rows(sessions, "hippocampus")
    assert list(rows) == ["p1"]
    assert rows["p1"]["n_sessions"] == 2
    assert rows["p1"]["n_trials"] == 60
    assert np.isclose(rows["p1"]["joint_partial_r"], 0.4)
