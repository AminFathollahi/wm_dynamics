"""Tests for scripts/run_deviation_serial_dependence_and_temporal_locus.py --
the pieces that could silently break this analysis without erroring: the
lag-cosine and detrending arithmetic, the neighbour-excluding reference
variant against a hand-computed example, the sign-to-worse-behaviour flip,
and both blocks' decision-rule classifiers against every named cell."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    _adjacency_statistic, _block_a_branch, _block_b_branch, _cosine_at_lag, _detrend,
    _sign_to_worse_behaviour, rate_free_state_deviation_neighbour_excluded, unit_direction_vectors,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------------------------------

def test_unit_direction_vectors_matches_the_delivered_estimators_own_normalisation():
    activity = np.array([[3.0, 4.0], [0.0, 5.0], [1.0, 0.0]])
    vectors = unit_direction_vectors(activity)
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.0, 1.0])
    assert vectors[2] == pytest.approx([1.0, 0.0])


def test_cosine_at_lag_hand_computed():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
    lag1 = _cosine_at_lag(vectors, 1)
    assert lag1 == pytest.approx([0.0, 0.0, -1.0])
    lag2 = _cosine_at_lag(vectors, 2)
    # pairs (0,2)=([1,0],[1,0]) cos=1; (1,3)=([0,1],[-1,0]) cos=0
    assert lag2 == pytest.approx([1.0, 0.0])


def test_cosine_at_lag_returns_empty_when_lag_exceeds_length():
    assert _cosine_at_lag(np.ones((3, 2)), 5).shape == (0,)


def test_adjacency_statistic_zero_when_all_pairs_identical():
    # Every trial points the same direction -> every lag's cosine is 1 -> adjacency statistic is exactly 0.
    vectors = np.tile(np.array([1.0, 0.0]), (30, 1))
    stat = _adjacency_statistic(vectors)
    assert stat == pytest.approx(0.0, abs=1e-9)


def test_detrend_removes_a_pure_linear_drift_at_a_narrow_window():
    # A trend so slow relative to the window that a centred moving average tracks it almost exactly;
    # the detrended residual should be close to zero everywhere except the very edges.
    n = 400
    t = np.arange(n, dtype=float)
    drift = np.stack([t / n, 1.0 - t / n], axis=1)
    vectors = drift / np.linalg.norm(drift, axis=1, keepdims=True)
    detrended = _detrend(vectors, window=21)
    interior = detrended[40:-40]
    assert np.max(np.abs(interior)) < 0.05


def test_detrend_preserves_a_lag1_only_interference_signal_a_slow_trend_would_hide():
    # Superpose a slow session-wide drift (which a raw, undetrended lag-1 profile cannot distinguish from
    # real adjacency) with a genuine, narrow lag-1 pull: even trials pulled toward odd trials. After
    # detrending, the observed lag-1 adjacency should survive while the drift's contribution is removed.
    rng = np.random.default_rng(0)
    n = 300
    base = rng.normal(size=(n, 8))
    t = np.arange(n, dtype=float)
    drift = np.outer(t / n, np.ones(8))  # slow additive drift shared by every unit
    pulled = base.copy()
    pulled[1::2] = 0.7 * base[1::2] + 0.3 * base[0:-1:2]  # odd trials pulled toward the preceding even trial
    activity = pulled + drift
    vectors = unit_direction_vectors(activity)
    raw_stat = _adjacency_statistic(vectors)
    detrended_stat = _adjacency_statistic(_detrend(vectors, window=21))
    # The pull is real and local (lag 1 only), so it should survive detrending and remain clearly positive.
    assert detrended_stat > 0.02
    # A pure drift with no real interference would give an undetrended profile dominated by the trend;
    # detrending should not have destroyed the genuine signal that raw_stat also (partly) reflects.
    assert detrended_stat > 0.0


# ---------------------------------------------------------------------------------------------------
# Neighbour-excluded reference variant
# ---------------------------------------------------------------------------------------------------

def test_neighbour_excluded_reference_excludes_more_than_the_delivered_estimator():
    # Four trials: 0 and 2 point one way, 1 and 3 point another. Trial 1's DELIVERED leave-one-out
    # reference is the mean of trials {0, 2, 3}; its NEIGHBOUR-EXCLUDED reference additionally drops
    # trial 0 (its left neighbour) and trial 2 (its right neighbour), leaving only trial 3.
    activity = np.array([
        [1.0, 0.0],   # trial 0
        [0.0, 1.0],   # trial 1
        [1.0, 0.0],   # trial 2
        [0.0, 1.0],   # trial 3
    ])
    delivered = rate_free_state_deviation(activity)
    alt = rate_free_state_deviation_neighbour_excluded(activity)
    # Trial 1's neighbour-excluded reference is trial 3 alone = [0, 1] (already unit norm) -> cosine 1 -> deviation 0.
    assert alt[1] == pytest.approx(0.0, abs=1e-9)
    # Trial 1's delivered reference is mean([1,0],[1,0],[0,1]) = [2/3, 1/3], normalised [0.894, 0.447];
    # cosine([0,1], that) = 0.447 -> deviation = 1 - 0.447 = 0.553, clearly different from the alt value.
    assert delivered[1] == pytest.approx(1.0 - 0.4472135955, abs=1e-6)
    assert delivered[1] != pytest.approx(alt[1], abs=1e-3)


def test_neighbour_excluded_reference_returns_nan_when_too_few_trials_remain():
    # With only 3 trials, excluding trial i and both neighbours can leave zero eligible reference trials.
    activity = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    alt = rate_free_state_deviation_neighbour_excluded(activity)
    assert np.isnan(alt[1])  # trial 1's neighbours are 0 and 2 -- excluding all three leaves nothing


# ---------------------------------------------------------------------------------------------------
# Sign-to-worse-behaviour flip
# ---------------------------------------------------------------------------------------------------

def test_sign_flip_negates_r_and_swaps_ci_and_direction_flags():
    result = {
        "status": "computed", "r": 0.3, "p_value": 0.01,
        "ci_lower": -0.1, "ci_upper": 0.5,
        "significant_negative": False, "significant_positive": True,
    }
    flipped = _sign_to_worse_behaviour(result, -1.0)
    assert flipped["r"] == pytest.approx(-0.3)
    assert flipped["p_value"] == 0.01  # unchanged: a sign flip does not change how extreme a value is
    assert flipped["ci_lower"] == pytest.approx(-0.5)
    assert flipped["ci_upper"] == pytest.approx(0.1)
    assert flipped["significant_negative"] is True
    assert flipped["significant_positive"] is False
    # Original untouched (packaging boundary only, never mutates the estimator's own return value).
    assert result["r"] == 0.3


def test_sign_flip_is_a_no_op_at_multiplier_one():
    result = {"status": "computed", "r": 0.3, "p_value": 0.01}
    assert _sign_to_worse_behaviour(result, 1.0) == result


# ---------------------------------------------------------------------------------------------------
# Block A classifier -- every named cell
# ---------------------------------------------------------------------------------------------------

def test_block_a_adjacency_absent_behaviour_survives():
    assert _block_a_branch(False, True) == "accuracy_predicting_component_is_not_interference_from_the_preceding_trial"


def test_block_a_adjacency_absent_behaviour_does_not_survive_is_a_disclosed_defect():
    assert _block_a_branch(False, False) == "partial_control_removes_variance_the_adjacency_statistic_does_not_explain"


def test_block_a_adjacency_present_behaviour_does_not_survive():
    assert _block_a_branch(True, False) == "accuracy_predicting_component_is_interference_from_the_preceding_trial"


def test_block_a_adjacency_present_behaviour_survives():
    assert _block_a_branch(True, True) == (
        "interference_from_the_preceding_trial_is_present_and_separable_from_the_accuracy_predicting_component"
    )


def test_block_a_not_computable_on_missing_inputs():
    assert _block_a_branch(None, True) == "not_computable"
    assert _block_a_branch(True, None) == "not_computable"


# ---------------------------------------------------------------------------------------------------
# Block B classifier
# ---------------------------------------------------------------------------------------------------

def _window(void: bool, significant: bool | None) -> dict:
    pooled = {"status": "not_computed"} if significant is None else {"status": "tested", "significant": significant}
    return {"void": void, "raw_vs_behaviour_pooled": pooled}


def test_block_b_all_void_is_unreachable():
    split = {"windows": [_window(True, None), _window(True, None)],
             "ordering_test_latest_minus_earliest_surviving": None}
    assert _block_b_branch(split, True)["branch"] == "temporal_locus_unreachable_at_this_count_per_window"


def test_block_b_present_throughout_when_ordering_not_significant():
    split = {"windows": [_window(False, True), _window(False, True)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": False, "mean_value": 0.01}}
    assert _block_b_branch(split, True)["branch"] == "accuracy_predicting_component_is_present_throughout_the_delay"


def test_block_b_grows_across_the_delay_when_ordering_significant_and_positive():
    split = {"windows": [_window(False, True), _window(False, True)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": True, "mean_value": 0.05}}
    assert _block_b_branch(split, True)["branch"] == "accuracy_predicting_component_grows_across_the_delay"


def test_block_b_strongest_at_onset_when_ordering_significant_and_negative():
    split = {"windows": [_window(False, True), _window(False, True)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": True, "mean_value": -0.05}}
    assert _block_b_branch(split, True)["branch"] == "accuracy_predicting_component_is_strongest_at_delay_onset"


def test_block_b_absent_everywhere_but_present_over_full_epoch():
    split = {"windows": [_window(False, False), _window(False, False)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": False, "mean_value": 0.0}}
    assert _block_b_branch(split, True)["branch"] == "accuracy_predicting_component_requires_the_full_epoch_to_be_detected"


def test_block_b_mixed_presence_is_the_disclosed_rule_gap_not_a_forced_label():
    # One surviving window present, the other absent -- a reachable outcome the pre-declared rule does
    # not name (it only names "every surviving window" agreeing one way or the other).
    split = {"windows": [_window(False, True), _window(False, False)],
             "ordering_test_latest_minus_earliest_surviving": {"status": "tested", "significant": False, "mean_value": 0.0}}
    result = _block_b_branch(split, True)
    assert result["branch"] == "block_b_outcome_not_covered_by_the_pre_declared_rule"
    assert result["sub_label"] == "behaviour_link_present_in_some_but_not_all_surviving_sub_windows"


def test_block_b_single_surviving_window_no_ordering_possible():
    split = {"windows": [_window(False, True), _window(True, None)],
             "ordering_test_latest_minus_earliest_surviving": None}
    # Only one window survives -> vacuously "present in every surviving window", no ordering test exists.
    assert _block_b_branch(split, True)["branch"] == "accuracy_predicting_component_is_present_throughout_the_delay"
