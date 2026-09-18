from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_human_behavior_load_cluster_sensitivity import (
    build_sensitivity,
    equal_patient_cluster_bootstrap,
    participant_cluster_bootstrap,
    refit_weight_participant_cluster_bootstrap,
)


def _arms(repeated: bool = True) -> dict:
    values = {"corpus::a": -0.3, "corpus::b": 0.1, "corpus::c": 0.4, "corpus::d": 0.0}
    arm = {"values": values, "frozen_standard_error": 0.1, "original_estimate": 0.05, "original_n_patients": 4}
    return {"corpus": {"1": arm, **({"2": {**arm, "values": dict(values)}} if repeated else {})}}


def test_perfectly_repeated_load_cells_do_not_narrow_cluster_interval() -> None:
    single = participant_cluster_bootstrap(_arms(False), n_draws=3000, seed_tag="same")
    repeated = participant_cluster_bootstrap(_arms(True), n_draws=3000, seed_tag="same")
    assert np.allclose(single["cluster_bootstrap_ci_95pct"], repeated["cluster_bootstrap_ci_95pct"])


def test_missing_load_cells_remain_eligible_without_complete_case_drop() -> None:
    arms = _arms(True)
    del arms["corpus"]["2"]["values"]["corpus::d"]
    result = participant_cluster_bootstrap(arms, n_draws=100, seed_tag="missing")
    assert result["status"] == "computed"
    assert result["n_unique_patients_by_corpus"] == {"corpus": 4}


def test_sampling_is_reproducible() -> None:
    first = participant_cluster_bootstrap(_arms(True), n_draws=400, seed_tag="repeatable")
    second = participant_cluster_bootstrap(_arms(True), n_draws=400, seed_tag="repeatable")
    assert first == second


def test_refit_weight_bootstrap_reports_valid_draws_and_precision() -> None:
    result = refit_weight_participant_cluster_bootstrap(_arms(True), n_draws=200, seed_tag="refit")
    assert result["status"] == "computed"
    assert result["n_draws_valid"] > 0
    assert result["n_draws_valid"] + result["n_draws_invalid"] == 200
    assert result["conditional_weighting"] is False
    assert result["cluster_bootstrap_se"] > 0.0
    assert result["n_arms_observed"] == 2
    assert result["refit_scope"] == [
        "arm means",
        "arm standard errors",
        "between-arm heterogeneity",
        "meta-analysis weights",
    ]
    assert result["trial_level_patient_estimates_refit"] is False
    assert set(result["arm_inclusion_fraction"]) == {"corpus_load1", "corpus_load2"}


def test_refit_preserves_missing_load_cells() -> None:
    arms = _arms(True)
    del arms["corpus"]["2"]["values"]["corpus::c"]
    result = refit_weight_participant_cluster_bootstrap(arms, n_draws=300, seed_tag="missing-refit")
    assert result["status"] == "computed"
    assert result["n_unique_patients_by_corpus"] == {"corpus": 4}
    assert result["n_arms_per_valid_draw"]["max"] == 2
    assert 0.0 < result["arm_inclusion_fraction"]["corpus_load2"] <= 1.0


def test_same_patient_label_in_two_corpora_remains_two_clusters() -> None:
    left = _arms(False)["corpus"]
    right = _arms(False)["corpus"]
    arms = {"left": left, "right": right}
    result = refit_weight_participant_cluster_bootstrap(arms, n_draws=100, seed_tag="qualified")
    assert result["n_unique_patients_by_corpus"] == {"left": 4, "right": 4}
    assert set(result["arm_inclusion_fraction"]) == {"left_load1", "right_load1"}


def test_equal_patient_estimand_does_not_duplicate_perfect_load_cells() -> None:
    single = equal_patient_cluster_bootstrap(_arms(False), n_draws=500, seed_tag="equal")
    repeated = equal_patient_cluster_bootstrap(_arms(True), n_draws=500, seed_tag="equal")
    assert single == repeated


def test_capability_statement_limits_refit_to_stored_patient_load_estimates() -> None:
    values = (-0.2, 0.0, 0.2, 0.4)
    patients = {
        f"corpus::p{i}": {
            "status": "computed",
            "raw": {"status": "computed", "r": value},
            "joint_partial": {"status": "computed", "r": value / 2.0},
        }
        for i, value in enumerate(values)
    }
    arm = {
        "pooled_raw": {"status": "tested", "mean_value": 0.1, "ci_lower": -0.1, "ci_upper": 0.3},
        "pooled_joint_partial": {"status": "tested", "mean_value": 0.05, "ci_lower": -0.1, "ci_upper": 0.2},
        "n_patients_computed": 4,
        "patients": patients,
    }
    data = {"block_b": {"per_corpus": {"corpus": {"by_load": {"1": arm}}}}}
    result = build_sensitivity(data, n_draws=20)
    assert result["refit_capability"]["patient_load_estimates_available"] is True
    assert result["refit_capability"]["trial_arrays_available_in_source_artifact"] is False
