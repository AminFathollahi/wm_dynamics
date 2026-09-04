"""Tests for scripts/run_multi_object_interference_and_locus_within_item_count.py --
the trial-count-weighted combination formula, the within-item-count-level decisive
partial's level-splitting logic, and both blocks' decision-rule classifiers against
every named cell, including the third Block A cell a prior analysis mislabelled."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_multi_object_interference_and_locus_within_item_count import (  # noqa: E402
    _aggregate_level_trial_counts, _block_a_branch_within_item_count, _block_b_branch_within_item_count,
    _close, _decisive_partial_within_item_count, _trial_count_weighted,
)


# ---------------------------------------------------------------------------------------------------
# Trial-count-weighted combination
# ---------------------------------------------------------------------------------------------------

def test_trial_count_weighted_matches_hand_computed_average():
    # Level with 30 trials at r=0.2, level with 10 trials at r=0.6 -> weighted mean = (30*0.2+10*0.6)/40 = 0.3
    tested = [(30, 0.2), (10, 0.6)]
    assert _trial_count_weighted(tested) == pytest.approx(0.3)


def test_trial_count_weighted_empty_is_none():
    assert _trial_count_weighted([]) is None


def test_trial_count_weighted_single_level_returns_that_level():
    assert _trial_count_weighted([(50, -0.4)]) == pytest.approx(-0.4)


# ---------------------------------------------------------------------------------------------------
# Reproduction-gate closeness check
# ---------------------------------------------------------------------------------------------------

def test_close_within_tolerance():
    assert _close(0.019673378706534905, 0.019673378706534905, tol=1e-6)
    assert not _close(0.02, 0.019673378706534905, tol=1e-6)
    assert not _close(None, 0.02)


# ---------------------------------------------------------------------------------------------------
# Decisive partial -- level splitting on a synthetic bundle
# ---------------------------------------------------------------------------------------------------

def _synthetic_bundle(n_trials: int, n_units: int = 6, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    activity_by_unit = np.abs(rng.normal(size=(n_trials, n_units))) + 1.0
    item_count = np.array([1.0, 2.0, 3.0] * (n_trials // 3) + [1.0] * (n_trials % 3))
    return {
        "session": "synthetic", "activity_by_unit": activity_by_unit,
        "report_error": rng.normal(size=n_trials), "deviation": rng.normal(size=n_trials),
        "spike_count": activity_by_unit.sum(axis=1), "trial_index": np.arange(n_trials, dtype=float),
        "item_count": item_count,
    }


def test_decisive_partial_too_few_trials_for_lag_profile():
    bundle = _synthetic_bundle(n_trials=10)
    result = _decisive_partial_within_item_count(bundle, window=51, seed_prefix="test")
    assert result["status"] == "too_few_trials_for_lag_profile"


def test_decisive_partial_computed_splits_by_item_count_level():
    bundle = _synthetic_bundle(n_trials=300, seed=1)
    result = _decisive_partial_within_item_count(bundle, window=51, seed_prefix="test")
    assert result["status"] == "computed"
    assert result["item_count_levels_present"] == [1, 2, 3]
    assert result["n_levels_tested"] >= 1
    assert set(result["within_item_count_level"]) == {
        "raw", "partial_controlling_lag1_alignment",
        "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index",
    }
    for level_entry in result["per_level"].values():
        if level_entry["status"] == "computed":
            assert {"raw", "partial_controlling_lag1_alignment",
                    "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index"} <= set(level_entry)


# ---------------------------------------------------------------------------------------------------
# _aggregate_level_trial_counts
# ---------------------------------------------------------------------------------------------------

def test_aggregate_level_trial_counts_sums_across_sessions():
    sessions = [
        {"status": "computed", "per_level": {"1": {"status": "computed", "n_trials": 20},
                                              "2": {"status": "too_few_trials_at_this_item_count", "n_trials": 5}}},
        {"status": "computed", "per_level": {"1": {"status": "computed", "n_trials": 30}}},
        {"status": "too_few_trials_for_lag_profile"},
    ]
    agg = _aggregate_level_trial_counts(sessions)
    assert agg["total_trials_by_item_count_level"] == {"1": 50}
    assert agg["n_sessions_contributing_by_item_count_level"] == {"1": 2}


# ---------------------------------------------------------------------------------------------------
# Block A classifier -- every named cell, including the disclosed third one
# ---------------------------------------------------------------------------------------------------

def _tested(mean_value: float, significant: bool) -> dict:
    return {"status": "tested", "mean_value": mean_value, "significant": significant}


def test_block_a_behaviour_survives_same_sign_significant_joint():
    raw = _tested(0.02, True)
    joint = _tested(0.015, True)
    result = _block_a_branch_within_item_count(raw, joint, established_positive=True)
    assert result["branch"] == "interference_from_the_preceding_trial_is_present_and_separable_from_the_accuracy_predicting_component"


def test_block_a_behaviour_does_not_survive_but_raw_significant_and_shrinks_is_interference():
    raw = _tested(0.02, True)
    joint = _tested(0.005, False)
    result = _block_a_branch_within_item_count(raw, joint, established_positive=True)
    assert result["branch"] == "accuracy_predicting_component_is_interference_from_the_preceding_trial"


def test_block_a_raw_not_significant_falls_to_third_cell():
    # This is the configuration a prior analysis mislabelled as interference: raw link not present at
    # the full epoch, so interference cannot be adjudicated.
    raw = _tested(0.01, False)
    joint = _tested(0.008, False)
    result = _block_a_branch_within_item_count(raw, joint, established_positive=True)
    assert result["branch"] == "within_level_link_not_present_at_full_epoch_so_interference_cannot_be_adjudicated"


def test_block_a_raw_significant_but_joint_grows_falls_to_third_cell():
    raw = _tested(0.01, True)
    joint = _tested(0.03, False)
    result = _block_a_branch_within_item_count(raw, joint, established_positive=True)
    assert result["branch"] == "within_level_link_not_present_at_full_epoch_so_interference_cannot_be_adjudicated"


def test_block_a_raw_significant_wrong_sign_falls_to_third_cell():
    raw = _tested(-0.02, True)  # significant but opposite the corpus's established positive direction
    joint = _tested(-0.005, False)
    result = _block_a_branch_within_item_count(raw, joint, established_positive=True)
    assert result["branch"] == "within_level_link_not_present_at_full_epoch_so_interference_cannot_be_adjudicated"


def test_block_a_not_computable_when_either_pooled_result_untested():
    result = _block_a_branch_within_item_count({"status": "not_computed"}, _tested(0.01, True), True)
    assert result["branch"] == "not_computable"


# ---------------------------------------------------------------------------------------------------
# Block B classifier
# ---------------------------------------------------------------------------------------------------

def _window(void: bool, significant: bool | None) -> dict:
    pooled = {"status": "not_computed"} if significant is None else {"status": "tested", "significant": significant}
    return {"void": void, "raw_vs_behaviour_pooled_within_item_count_level": pooled}


def test_block_b_all_void_is_unreachable():
    split = {"windows": [_window(True, None), _window(True, None)],
             "ordering_test_latest_minus_earliest_surviving": None}
    assert _block_b_branch_within_item_count(split, True)["branch"] == "temporal_locus_unreachable_at_this_count_per_window"


def test_block_b_present_throughout_when_ordering_not_significant():
    split = {"windows": [_window(False, True), _window(False, True)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": False, "mean_value": 0.01}}
    assert _block_b_branch_within_item_count(split, True)["branch"] == "accuracy_predicting_component_is_present_throughout_the_delay"


def test_block_b_grows_across_the_delay_when_ordering_significant_and_positive():
    split = {"windows": [_window(False, True), _window(False, True)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": True, "mean_value": 0.05}}
    assert _block_b_branch_within_item_count(split, True)["branch"] == "accuracy_predicting_component_grows_across_the_delay"


def test_block_b_strongest_at_onset_when_ordering_significant_and_negative():
    split = {"windows": [_window(False, True), _window(False, True)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": True, "mean_value": -0.05}}
    assert _block_b_branch_within_item_count(split, True)["branch"] == "accuracy_predicting_component_is_strongest_at_delay_onset"


def test_block_b_absent_everywhere_but_present_over_full_epoch():
    split = {"windows": [_window(False, False), _window(False, False)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": False, "mean_value": 0.0}}
    assert _block_b_branch_within_item_count(split, True)["branch"] == "accuracy_predicting_component_requires_the_full_epoch_to_be_detected"


def test_block_b_mixed_presence_is_the_disclosed_rule_gap_not_a_forced_label():
    split = {"windows": [_window(False, True), _window(False, False)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": False, "mean_value": 0.0}}
    result = _block_b_branch_within_item_count(split, True)
    assert result["branch"] == "block_b_outcome_not_covered_by_the_pre_declared_rule"
    assert result["sub_label"] == "behaviour_link_present_in_some_but_not_all_surviving_sub_windows"


def test_block_b_single_surviving_window_no_ordering_possible():
    split = {"windows": [_window(False, True), _window(True, None)],
             "ordering_test_latest_minus_earliest_surviving": None}
    assert _block_b_branch_within_item_count(split, True)["branch"] == "accuracy_predicting_component_is_present_throughout_the_delay"
