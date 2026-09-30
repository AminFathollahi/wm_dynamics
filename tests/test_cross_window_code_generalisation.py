"""Tests for scripts/run_cross_window_code_generalisation.py -- the pieces that
could silently break this analysis without erroring: (1) a rotating code must
produce a chance-level cross-window quadrant while both within-window blocks
decode well, (2) a stable code must produce a strong cross-window quadrant,
(3) the subject-cluster bootstrap must recover a planted effect and
straddle zero on null data, (4) a NaN estimator failure must be distinguished
from a successful estimate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_cross_window_code_generalisation import cross_window_ctg, cross_window_ctg_multiclass, quadrant_stats, subject_cluster_bootstrap, MIN_DIAG_AUC
from statistics import subject_cluster_bootstrap_paired, cell_status
from statistics import fdr_bh  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "cross_window_code_generalisation.json"


def _make_population(n_trials, n_units, n_bins_a, n_bins_b, direction_a, direction_b,
                      amplitude=1.2, noise=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    y = np.concatenate([np.zeros(n_trials // 2), np.ones(n_trials - n_trials // 2)]).astype(int)
    rng.shuffle(y)
    signed = (2 * y - 1).astype(float)
    psth_a = rng.normal(0.0, noise, size=(n_trials, n_units, n_bins_a))
    psth_b = rng.normal(0.0, noise, size=(n_trials, n_units, n_bins_b))
    psth_a += amplitude * signed[:, None, None] * direction_a[None, :, None]
    psth_b += amplitude * signed[:, None, None] * direction_b[None, :, None]
    return psth_a, psth_b, y


def test_rotating_code_kills_cross_window_but_not_within_window():
    rng = np.random.default_rng(1)
    n_units = 24
    direction_a = rng.normal(size=n_units)
    direction_a /= np.linalg.norm(direction_a)
    direction_b = rng.normal(size=n_units)
    direction_b -= direction_b @ direction_a * direction_a  # orthogonal to direction_a
    direction_b /= np.linalg.norm(direction_b)

    psth_a, psth_b, y = _make_population(200, n_units, 5, 5, direction_a, direction_b, rng=rng)
    auc_mat, n_a = cross_window_ctg(psth_a, psth_b, y, n_components=6, n_splits=5,
                                    rng=np.random.default_rng(2), joint_scaler=False)
    qs = quadrant_stats(auc_mat, n_a)

    assert qs["diag_a_mean_auc"] > 0.8
    assert qs["diag_b_mean_auc"] > 0.8
    assert abs(qs["train_a_test_b_offdiag_effect"]) < 0.12
    assert abs(qs["train_b_test_a_offdiag_effect"]) < 0.12


def test_stable_code_clears_cross_window():
    rng = np.random.default_rng(3)
    n_units = 24
    direction = rng.normal(size=n_units)
    direction /= np.linalg.norm(direction)

    psth_a, psth_b, y = _make_population(200, n_units, 5, 5, direction, direction, rng=rng)
    auc_mat, n_a = cross_window_ctg(psth_a, psth_b, y, n_components=6, n_splits=5,
                                    rng=np.random.default_rng(4), joint_scaler=False)
    qs = quadrant_stats(auc_mat, n_a)

    assert qs["diag_a_mean_auc"] > 0.8
    assert qs["diag_b_mean_auc"] > 0.8
    assert qs["train_a_test_b_offdiag_effect"] > 0.2
    assert qs["train_b_test_a_offdiag_effect"] > 0.2
    assert qs["diag_a_mean_auc"] >= MIN_DIAG_AUC


def test_subject_cluster_bootstrap_recovers_planted_effect():
    rng = np.random.default_rng(5)
    values = rng.normal(0.15, 0.05, size=30)
    result = subject_cluster_bootstrap(values, n_boot=2000, rng=np.random.default_rng(6))
    assert result["ci_lower"] > 0.0
    assert result["mdd"] > 0.0


def test_subject_cluster_bootstrap_straddles_zero_on_null():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.05, size=30)
    result = subject_cluster_bootstrap(values, n_boot=2000, rng=np.random.default_rng(50))
    assert result["ci_lower"] < 0.0 < result["ci_upper"]


def test_cell_status_distinguishes_nan_from_estimate():
    assert cell_status(is_nan=True) == "failure_nan"
    assert cell_status(is_nan=False) == "estimated"


def test_delivered_artifact_has_no_clearance_label():
    artifact = json.loads(RESULTS.read_text())
    banned = ("clear", "inconclusive_not_cleared", "not_cleared")

    def walk(obj):
        if isinstance(obj, dict):
            assert "verdict" not in obj
            if "cell_status" in obj:
                value = obj["cell_status"]
                if value is not None:
                    assert not any(b in value for b in banned), value
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(artifact)


def test_predeclared_rules_and_cell_status_values():
    artifact = json.loads(RESULTS.read_text())
    rules = artifact["predeclared_rules"]
    assert "clearing_rule" in rules
    assert "clearing_rule_status" in rules
    permitted = {"estimated", "failure_nan", "descriptive_not_in_fdr_family"}

    def walk(obj):
        if isinstance(obj, dict):
            if "cell_status" in obj and obj["cell_status"] is not None:
                assert obj["cell_status"] in permitted
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(artifact)


def test_standardisation_arms_comparison_matches_stored_numbers():
    artifact = json.loads(RESULTS.read_text())
    dataset_key = next(k for k, v in artifact["corpora"].items()
                       if isinstance(v, dict) and "standardisation_arms_comparison" in v)
    record = artifact["corpora"][dataset_key]
    label_name = next(iter(record["standardisation_arms_comparison"]))
    comparison = record["standardisation_arms_comparison"][label_name]
    arm_record = record["label_arms"][label_name]
    for direction, entry in comparison.items():
        a = arm_record["arms"]["per_window"][direction]
        b = arm_record["arms"]["joint_scaler"][direction]
        assert entry["per_window"]["estimand"] == a["estimand"]
        assert entry["per_window"]["ci_lower"] == a["ci_lower"]
        assert entry["per_window"]["ci_upper"] == a["ci_upper"]
        assert entry["per_window"]["q_value"] == a["q_value"]
        assert entry["joint_scaler"]["estimand"] == b["estimand"]
        if a["estimand"] is not None and b["estimand"] is not None:
            assert entry["estimand_difference_per_window_minus_joint_scaler"] == a["estimand"] - b["estimand"]


def test_transfer_ratio_is_null_when_diagonal_is_not_distinguishable_from_chance():
    rng = np.random.default_rng(20)
    n = 30
    offdiag = rng.normal(0.05, 0.01, size=n)

    diag_decodable = rng.normal(0.7, 0.02, size=n)
    result_decodable = subject_cluster_bootstrap_paired(offdiag, diag_decodable, n_boot=2000,
                                                        rng=np.random.default_rng(21))
    assert result_decodable["diag_distinguishable_from_half"] is True
    assert np.isfinite(result_decodable["transfer_ratio"])
    assert result_decodable["diag_ci_lower"] < result_decodable["diag_observed"] < result_decodable["diag_ci_upper"]

    diag_near_chance = np.random.default_rng(99).normal(0.5, 0.1, size=n)
    result_near_chance = subject_cluster_bootstrap_paired(offdiag, diag_near_chance, n_boot=2000,
                                                          rng=np.random.default_rng(22))
    assert result_near_chance["diag_distinguishable_from_half"] is False
    assert result_near_chance["transfer_ratio"] is None
    assert result_near_chance["transfer_ratio_ci_lower"] is None
    assert result_near_chance["transfer_ratio_ci_upper"] is None


def test_fdr_bh_family_separation_sanity():
    # A strong family and a weak family should not contaminate each other's q-values.
    strong = np.array([0.001, 0.002, 0.003])
    weak = np.array([0.4, 0.6, 0.8])
    q_strong = fdr_bh(strong, alpha=0.05)
    q_weak = fdr_bh(weak, alpha=0.05)
    assert q_strong["n_reject"] == 3
    assert q_weak["n_reject"] == 0


def _make_multiclass_population(n_trials, n_units, n_bins_a, n_bins_b, direction_a, direction_b,
                                 n_classes=5, amplitude=1.4, noise=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    y = rng.integers(0, n_classes, size=n_trials)
    onehot_centered = (np.eye(n_classes)[y] - 1.0 / n_classes)  # (n_trials, n_classes), sums to 0 per class
    psth_a = rng.normal(0.0, noise, size=(n_trials, n_units, n_bins_a))
    psth_b = rng.normal(0.0, noise, size=(n_trials, n_units, n_bins_b))
    coeff_a = onehot_centered @ direction_a  # (n_trials, n_units)
    coeff_b = onehot_centered @ direction_b
    psth_a += amplitude * coeff_a[:, :, None]
    psth_b += amplitude * coeff_b[:, :, None]
    return psth_a, psth_b, y


def test_multiclass_stable_content_clears_cross_window():
    rng = np.random.default_rng(9)
    n_units, n_classes = 24, 5
    direction = rng.normal(size=(n_classes, n_units))
    psth_a, psth_b, y = _make_multiclass_population(300, n_units, 5, 5, direction, direction,
                                                     n_classes=n_classes, rng=rng)
    auc_mat, n_a = cross_window_ctg_multiclass(psth_a, psth_b, y, n_components=8, n_splits=5,
                                               rng=np.random.default_rng(10), joint_scaler=False)
    qs = quadrant_stats(auc_mat, n_a)
    assert qs["diag_a_mean_auc"] > 0.7
    assert qs["diag_b_mean_auc"] > 0.7
    assert qs["train_a_test_b_offdiag_effect"] > 0.1
    assert qs["train_b_test_a_offdiag_effect"] > 0.1


def test_multiclass_rotating_content_kills_cross_window():
    rng = np.random.default_rng(11)
    n_units, n_classes = 24, 5
    direction_a = rng.normal(size=(n_classes, n_units))
    direction_b = rng.normal(size=(n_classes, n_units))
    psth_a, psth_b, y = _make_multiclass_population(300, n_units, 5, 5, direction_a, direction_b,
                                                     n_classes=n_classes, rng=rng)
    auc_mat, n_a = cross_window_ctg_multiclass(psth_a, psth_b, y, n_components=8, n_splits=5,
                                               rng=np.random.default_rng(12), joint_scaler=False)
    qs = quadrant_stats(auc_mat, n_a)
    assert qs["diag_a_mean_auc"] > 0.7
    assert qs["diag_b_mean_auc"] > 0.7
    assert abs(qs["train_a_test_b_offdiag_effect"]) < 0.15
    assert abs(qs["train_b_test_a_offdiag_effect"]) < 0.15
