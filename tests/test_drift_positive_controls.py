import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_drift_positive_controls",
    ROOT / "scripts" / "run_drift_positive_controls.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_positive_controls_reuse_source_spine_fold_seeds():
    assert MODULE.source_fold_seed("000469", "sub-12") == 20260743
    assert MODULE.source_fold_seed("000574", "sub-08_ses-05") == 20261536
    assert MODULE.PERMUTED_AXIS_SEED_OFFSET == 2000
    assert MODULE.COMPLEMENT_AXIS_SEED_OFFSET == 4000
    assert MODULE.SWITCHING_SEED_OFFSET == 6000


def test_permuted_axis_retains_true_conditioning_labels(monkeypatch):
    train_labels = np.arange(12) % 3
    test_labels = np.arange(6) % 3
    captured = {}

    def fake_direction(_window, labels):
        captured["axis_labels"] = labels.copy()
        return np.array([1.0, 0.0])

    def fake_scores(_train, _test, fit_labels, held_out_labels, _direction, _dt):
        captured["fit_labels"] = fit_labels.copy()
        captured["held_out_labels"] = held_out_labels.copy()
        return {"status": "estimable"}, np.empty((0, 0)), np.empty((0, 0))

    monkeypatch.setattr(MODULE, "discriminant_direction", fake_direction)
    monkeypatch.setattr(MODULE, "projected_axis_scores", fake_scores)
    MODULE.permuted_axis_scores(
        np.zeros((12, 5, 2)), np.zeros((6, 5, 2)), train_labels, test_labels,
        np.zeros((12, 2)), 0.1, 7,
    )

    assert not np.array_equal(captured["axis_labels"], train_labels)
    assert np.array_equal(captured["fit_labels"], train_labels)
    assert np.array_equal(captured["held_out_labels"], test_labels)


def test_partial_fold_metric_remains_in_patient_summary():
    sessions = {
        "sub-01": {
            "status": "complete",
            "folds": [
                {"raw": {"always": 1.0, "sometimes": 2.0}},
                {"raw": {"always": 3.0}},
            ],
        }
    }

    summary = MODULE.summarize_sessions("000469", sessions)

    assert summary["entity_metrics"]["sub-01"]["raw.always"] == 2.0
    assert summary["entity_metrics"]["sub-01"]["raw.sometimes"] == 2.0
    assert summary["group"]["raw.sometimes"]["n_entities"] == 1


def test_adjudication_rejects_confined_dynamics_when_matched_interval_crosses_zero():
    def metric(mean, lower):
        return {"mean": mean, "patient_bootstrap_interval_95": [lower, mean + 0.1]}

    group = {
        "raw.content_axis.m2_minus_heteroscedastic_m0_nats_per_transition": metric(-0.01, -0.02),
        "raw.content_axis.m2_minus_free_variance_ar1_m0_nats_per_transition": metric(0.001, -0.001),
        "raw.neighbouring_trial_prediction.own_minus_neighbour_r2_advantage": metric(0.03, 0.01),
        "variance_stabilized.content_axis.m2_minus_heteroscedastic_m0_nats_per_transition": metric(0.001, -0.01),
        "variance_stabilized.content_axis.m2_minus_free_variance_ar1_m0_nats_per_transition": metric(0.002, -0.001),
        "variance_stabilized.neighbouring_trial_prediction.own_minus_neighbour_r2_advantage": metric(0.03, 0.01),
        "trial_order_detrended.neighbouring_trial_prediction.own_minus_neighbour_r2_advantage": metric(0.03, 0.01),
        "raw.content_axis.m2_minus_m0_nats_per_observation": metric(0.01, 0.001),
        "raw.content_axis.own_trial_prediction.held_out_r2_advantage": metric(0.02, 0.001),
        "raw.permuted_label_axis.m2_minus_m0_nats_per_observation": metric(0.02, 0.001),
        "raw.permuted_label_axis.own_trial_prediction.held_out_r2_advantage": metric(0.03, 0.001),
        "raw.signal_matched_complement_axis.m2_minus_m0_nats_per_observation": metric(0.03, 0.001),
        "raw.signal_matched_complement_axis.own_trial_prediction.held_out_r2_advantage": metric(0.04, 0.001),
        "raw.content_minus_permuted_label_axis.m2_minus_m0_nats_per_observation": metric(-0.01, -0.02),
        "raw.content_minus_permuted_label_axis.own_trial_prediction_r2_advantage": metric(-0.01, -0.02),
        "raw.content_minus_signal_matched_complement_axis.m2_minus_m0_nats_per_observation": metric(-0.02, -0.03),
        "raw.content_minus_signal_matched_complement_axis.own_trial_prediction_r2_advantage": metric(-0.02, -0.03),
    }

    result = MODULE.adjudicate_positive_controls({"000469": {"group": group}})

    assert result["overall"] == "original_confined_dynamics_interpretation_reversed"
    assert result["datasets"]["000469"]["confined_dynamics_supported"] is False
    assert result["datasets"]["000469"]["content_specificity_supported"] is False
    assert result["datasets"]["000469"]["off_axis_effects_positive"] is True
