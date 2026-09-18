"""Guard the prediction text against treating correlations as probability changes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_human_maintenance_behaviour_link import refresh_stored_prediction, run_human_stimulation_prediction


def test_human_stimulation_prediction_keeps_correlation_scale_and_omits_probability_dose() -> None:
    result = run_human_stimulation_prediction(
        {"pooled": {"raw_outcome_vs_deviation": {"mean_value": -0.0974, "p_value": 0.0035},
                    "joint_partial_controlling_spike_count_and_trial_index": {"mean_value": -0.0696, "p_value": 0.0206}}},
        None,
        {"combined_within_load_then_meta_analysed": {"joint_partial": {"status": "computed", "p_value": 0.8}}},
    )
    prediction = result["prediction"].lower()
    assert "correlation" in prediction
    assert "probability of an error" not in prediction
    assert "base-rate" in prediction


def test_refresh_stored_prediction_preserves_results_and_replaces_scale_error(tmp_path: Path) -> None:
    artifact_path = tmp_path / "result.json"
    reference_path = tmp_path / "reference.json"
    subspace_path = tmp_path / "missing.json"
    artifact_path.write_text(
        '{"status":"complete","block_b":{"combined_within_load_then_meta_analysed":'
        '{"joint_partial":{"status":"computed","p_value":0.8}}},'
        '"block_f":{"prediction":"probability of an error"}}'
    )
    reference_path.write_text(
        '{"pooled":{"raw_outcome_vs_deviation":{"mean_value":-0.0974,"p_value":0.0035},'
        '"joint_partial_controlling_spike_count_and_trial_index":{"mean_value":-0.0696,"p_value":0.0206}}}'
    )

    refreshed = refresh_stored_prediction(artifact_path, reference_path, subspace_path)

    assert refreshed["status"] == "complete"
    assert "probability of an error" not in refreshed["block_f"]["prediction"].lower()
    assert "correlations" in refreshed["block_f"]["prediction"].lower()


def test_human_prediction_uses_human_sign_and_values_together() -> None:
    result = run_human_stimulation_prediction(
        {"pooled": {"raw_outcome_vs_deviation": {"mean_value": -0.0974, "p_value": 0.0035},
                    "joint_partial_controlling_spike_count_and_trial_index": {"mean_value": -0.0696, "p_value": 0.0206}}},
        None,
        {"combined_within_load_then_meta_analysed": {
            "raw": {"status": "computed", "pooled": 0.12, "p_value": 0.03},
            "joint_partial": {"status": "computed", "pooled": 0.08, "p_value": 0.04},
        }},
    )

    assert result["sign_of_per_trial_component_change_vs_accuracy_change"] == "positive"
    assert result["prediction_correlation_values"] == {"joint_partial_r": 0.08, "raw_r": 0.12}
    assert "positive direction" in result["prediction"]
    assert "corresponding accuracy increase" in result["prediction"]
    assert "corresponding accuracy decrease" not in result["prediction"]
