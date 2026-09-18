from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_human_behavioral_value import evaluate_records, _history_features  # noqa: E402


def _rows(seed: int = 4) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(6):
        previous = []
        for trial in range(30):
            state = rng.normal()
            load = 1 + (trial % 3)
            rate = rng.poisson(8 + load)
            p = 1.0 / (1.0 + np.exp(-(0.8 * state - 0.5 * load)))
            y = float(rng.random() < p)
            history = [previous[-1] if previous else 0.0, previous[-2] if len(previous) > 1 else 0.0,
                       float(bool(previous)), float(len(previous) > 1)]
            rows.append({"participant": f"p{participant}", "corpus": f"c{participant % 2}",
                         "session": f"s{participant}", "y": y,
                         "state": state, "load": load, "rate": float(rate), "history": history})
            previous.append(y)
    return rows


def test_participant_held_out_evaluation_reports_requested_models_and_matched_effects():
    result = evaluate_records(_rows())
    assert result["status"] == "complete"
    assert result["eligibility"]["n_participants_eligible"] == 6
    assert set(result["metrics"]) == {"state", "task_load", "rate", "trial_history", "combined_nuisance", "combined_state"}
    assert set(result["matched_prediction_differences"]) == {"state", "task_load", "rate", "trial_history", "combined_nuisance"}
    assert (result["matched_prediction_differences"]["combined_nuisance"]["estimand"]
            == "baseline_log_loss_minus_combined_state_log_loss")
    assert result["native_units"]["log_loss"] == "natural log loss per trial"


def test_history_is_past_only_and_first_trial_is_explicitly_missing():
    features = _history_features({"is_correct": np.array([1.0, 0.0, 1.0])})
    assert np.allclose(features[0], [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(features[2], [0.0, 1.0, 1.0, 1.0])


def test_fails_closed_when_requested_baseline_labels_are_missing():
    rows = _rows()[:60]
    for row in rows:
        row.pop("load")
    result = evaluate_records(rows)
    assert result["status"] == "blocked"
    assert "labels" in result["reason"] or "folds" in result["reason"]
