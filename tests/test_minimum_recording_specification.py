"""Tests for the recording specification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_minimum_recording_specification import (  # noqa: E402
    BRANCH_HIGH_FIDELITY_NOT_REACHED, BRANCH_LOW_FIDELITY_NOT_REACHED, BRANCH_NOT_DETERMINABLE,
    BRANCH_RECOVERABLE_SMALL, BRANCH_REQUIRES_MORE_THAN_SMALL,
    NATIVE_LABEL, _channel_rung_draws, _deviation_against_direction, _reference_direction,
    bootstrap_ci_for_crossing_q, bootstrap_ci_for_joint_crossing, classify_corpus_branch,
    common_channel_rungs, compute_fidelity_targets, compute_joint_fidelity_specification,
    draw_index_subset, independent_reference_cells, pool_cell, random_axis, recovery_score,
    reference_pair_score_split, reference_score_split, run_corpus_recovery,
    session_joint_cells, session_rungs,
    smallest_q_reaching_target,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# _reference_direction / _deviation_against_direction, including bit-for-bit equivalence to
# rate_free_state_deviation via the leave-one-out stacking identity.
# ---------------------------------------------------------------------------------------------------

def test_reference_direction_is_unit_length():
    rng = np.random.default_rng(0)
    activity = rng.poisson(5.0, size=(20, 6)).astype(float)
    direction = _reference_direction(activity)
    assert direction is not None
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_reference_direction_none_when_all_zero():
    assert _reference_direction(np.zeros((5, 4))) is None


def test_deviation_against_direction_nan_for_zero_activity_trial():
    activity = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    direction = np.array([1.0, 0.0, 0.0])
    out = _deviation_against_direction(activity, direction)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_held_out_reference_matches_rate_free_state_deviation_stacking_identity():
    """rate_free_state_deviation's own leave-one-out mean, for a single trial stacked onto a REFERENCE
    block it is not itself a member of, reduces to exactly the fixed external-reference cosine
    _deviation_against_direction computes -- proving the held-out construction is the SAME quantity
    rate_free_state_deviation reduces to under this project's own unchanged formula, not a new one."""
    rng = np.random.default_rng(1)
    reference = rng.poisson(8.0, size=(12, 5)).astype(float)
    score_trials = rng.poisson(8.0, size=(3, 5)).astype(float)
    direction = _reference_direction(reference)
    expected = _deviation_against_direction(score_trials, direction)
    for i in range(score_trials.shape[0]):
        stacked = np.vstack([score_trials[i : i + 1], reference])
        via_unchanged_function = rate_free_state_deviation(stacked)[0]
        assert via_unchanged_function == pytest.approx(expected[i], abs=1e-10)


def test_recovery_score_perfect_when_identical():
    x = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
    assert recovery_score(x, x) == pytest.approx(1.0)


def test_recovery_score_none_below_minimum_finite_pairs():
    assert recovery_score(np.array([0.1, np.nan, np.nan]), np.array([0.1, 0.2, np.nan])) is None


def test_recovery_score_none_for_zero_variance():
    x = np.array([0.5, 0.5, 0.5, 0.5])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    assert recovery_score(x, y) is None


# ---------------------------------------------------------------------------------------------------
# rung ladders, subset/split draws
# ---------------------------------------------------------------------------------------------------

def test_session_rungs_caps_at_full_count_and_appends_native_value():
    rungs = session_rungs(20, (2, 4, 8, 16, 32))
    assert rungs == [2, 4, 8, 16, 20]


def test_session_rungs_exact_universal_value_not_duplicated():
    rungs = session_rungs(16, (2, 4, 8, 16, 32))
    assert rungs == [2, 4, 8, 16]


def test_common_channel_rungs_use_only_universal_counts_reached_by_every_session():
    assert common_channel_rungs([8, 12, 31], (2, 4, 8, 16, 32)) == [2, 4, 8]


def test_draw_index_subset_deterministic_and_correct_size():
    a = draw_index_subset(100, 10, "tag")
    b = draw_index_subset(100, 10, "tag")
    assert np.array_equal(a, b)
    assert len(a) == 10
    assert len(set(a.tolist())) == 10


def test_draw_index_subset_native_returns_everything():
    out = draw_index_subset(10, 10, "tag")
    assert np.array_equal(out, np.arange(10))


def test_reference_score_split_disjoint_and_covers_all_trials():
    idx = np.arange(17)
    ref, score = reference_score_split(idx, "tag")
    assert set(ref.tolist()).isdisjoint(set(score.tolist()))
    assert len(ref) + len(score) == 17
    assert len(ref) == 9  # odd count -> reference half gets the extra trial


def test_reference_pair_score_split_is_disjoint_and_complete():
    idx = np.arange(17)
    target, candidate, score = reference_pair_score_split(idx, "tag")
    assert set(target).isdisjoint(candidate)
    assert set(target).isdisjoint(score)
    assert set(candidate).isdisjoint(score)
    assert sorted(np.concatenate([target, candidate, score]).tolist()) == idx.tolist()


def test_random_axis_is_unit_length_and_deterministic():
    a = random_axis(8, "tag")
    b = random_axis(8, "tag")
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


# ---------------------------------------------------------------------------------------------------
# pool_cell / smallest_q_reaching_target / bootstrap_ci_for_crossing_q
# ---------------------------------------------------------------------------------------------------

def test_pool_cell_reports_median_and_iqr():
    result = pool_cell([1.0, 2.0, 3.0, 4.0, np.nan])
    assert result["status"] == "computed"
    assert result["n"] == 4
    assert result["median"] == pytest.approx(2.5)


def test_pool_cell_not_computable_when_all_nan():
    assert pool_cell([np.nan, np.nan])["status"] == "not_computable"


def test_smallest_q_reaching_target_basic():
    curve = {2: 0.1, 4: 0.4, 8: 0.6, 16: 0.9}
    assert smallest_q_reaching_target([2, 4, 8, 16], curve, 0.5) == 8
    assert smallest_q_reaching_target([2, 4, 8, 16], curve, 0.95) is None


def test_smallest_q_requires_a_sustained_crossing():
    curve = {2: 0.1, 4: 0.7, 8: 0.4, 16: 0.8}
    assert smallest_q_reaching_target([2, 4, 8, 16], curve, 0.5) == 16


def test_bootstrap_ci_for_crossing_q_contains_point_estimate_structure():
    summaries = ([{"session": f"high_{i}", "by_q_debiased_median": {4: 0.3, 8: 0.6, 16: 0.9},
                   "native_debiased_median": 1.0} for i in range(4)]
                 + [{"session": f"low_{i}", "by_q_debiased_median": {4: 0.3, 8: 0.4, 16: 0.9},
                     "native_debiased_median": 1.0} for i in range(4)])
    result = bootstrap_ci_for_crossing_q(
        8, summaries, 0.5, [4, 8, 16], n_boot=200, seed_tag="tag"
    )
    assert result["status"] == "computed"
    assert result["ci_lower"] <= result["point_estimate_q"] <= result["ci_upper"]
    assert result["ci_lower"] < result["ci_upper"]


def test_bootstrap_ci_not_computable_below_two_sessions():
    assert bootstrap_ci_for_crossing_q(8, [{4: 0.3}], 0.5, [4, 8], 200, "tag")["status"] == "not_computable"


def test_compute_fidelity_targets_reports_not_reached_when_curve_never_clears_target():
    curve = {2: 0.05, 4: 0.1}
    summaries = [{"session": str(i), "by_q_debiased_median": curve, "native_debiased_median": 1.0}
                 for i in range(5)]
    out = compute_fidelity_targets(curve, native_anchor_value=1.0, numeric_qs_sorted=[2, 4],
                                    targets=(0.5,), session_summaries=summaries, seed_tag="tag")
    assert out["0.5"]["status"] == "right_censored"
    assert out["0.5"]["crossing_status"] == "not_reached_at_maximum_available_channel_count"
    assert out["0.5"]["maximum_available_channel_count"] == 4


def test_bootstrap_censoring_never_returns_a_finite_interval():
    summaries = ([{"session": f"high_{i}", "by_q_debiased_median": {4: 0.2, 8: 1.0},
                   "native_debiased_median": 1.0} for i in range(2)]
                 + [{"session": f"low_{i}", "by_q_debiased_median": {4: 0.2, 8: 0.0},
                     "native_debiased_median": 1.0} for i in range(2)])
    result = bootstrap_ci_for_crossing_q(
        8, summaries, 0.5, [4, 8], n_boot=200, seed_tag="censored"
    )
    assert result["status"] == "right_censored"
    assert result["n_bootstrap_right_censored"] > 0
    assert result["ci_upper"] is None
    assert result["ci_upper_status"] == "right_censored"


# ---------------------------------------------------------------------------------------------------
# classify_corpus_branch -- all four named outcomes
# ---------------------------------------------------------------------------------------------------

def test_branch_not_determinable_below_session_floor():
    out = classify_corpus_branch(2, {4: 0.9}, 1.0, 16.0, [4], (0.5, 0.7, 0.9), 0.25, 4)
    assert out["branch"] == BRANCH_NOT_DETERMINABLE


def test_branch_low_fidelity_not_reached_on_common_grid():
    curve = {2: 0.01, 4: 0.02, 8: 0.03}
    out = classify_corpus_branch(10, curve, 1.0, 32.0, [2, 4, 8], (0.5, 0.7, 0.9), 0.25, 4)
    assert out["branch"] == BRANCH_LOW_FIDELITY_NOT_REACHED
    assert out["maximum_common_channel_count"] == 8


def test_branch_high_fidelity_not_reached_on_common_grid():
    curve = {2: 0.3, 4: 0.6, 8: 0.8}
    out = classify_corpus_branch(10, curve, 1.0, 64.0, [2, 4, 8], (0.5, 0.7, 0.9), 0.25, 4)
    assert out["branch"] == BRANCH_HIGH_FIDELITY_NOT_REACHED
    assert out["maximum_common_channel_count"] == 8


def test_branch_recoverable_from_small_channel_count():
    # native median q = 64; 0.25*64 = 16 -> reaching the top target at q=8 is "small"
    curve = {2: 0.4, 4: 0.6, 8: 0.95}
    out = classify_corpus_branch(10, curve, 1.0, 64.0, [2, 4, 8], (0.5, 0.7, 0.9), 0.25, 4)
    assert out["branch"] == BRANCH_RECOVERABLE_SMALL


def test_branch_high_fidelity_requires_more_than_small_channel_count():
    # native median q = 8; 0.25*8 = 2 -> reaching the top target only at q=8 is NOT "small"
    curve = {2: 0.4, 4: 0.6, 8: 0.95}
    out = classify_corpus_branch(10, curve, 1.0, 8.0, [2, 4, 8], (0.5, 0.7, 0.9), 0.25, 4)
    assert out["branch"] == BRANCH_REQUIRES_MORE_THAN_SMALL


def test_joint_thresholds_are_bidirectional_and_right_censored():
    joint = {
        "16": {"2": 0.2, "4": 0.4, "native": 0.6},
        "32": {"2": 0.4, "4": 0.7, "native": 0.95},
        "native": {"2": 0.4, "4": 0.8, "native": 1.0},
    }
    sessions = [
        {"session": f"s{i}", "native_debiased_median": 1.0, "joint_debiased_median": joint}
        for i in range(4)
    ]
    result = compute_joint_fidelity_specification(sessions, [16, 32], [2, 4], (0.5,), "joint_test")
    target = result["fidelity_targets"]["0.5"]
    assert "native" not in target["minimum_calibration_trials_by_channel_count"]
    assert "native" not in target["minimum_channels_by_trial_count"]
    assert target["minimum_calibration_trials_by_channel_count"]["2"]["status"] == "right_censored"
    q4 = target["minimum_calibration_trials_by_channel_count"]["4"]
    assert q4["smallest_calibration_trial_count_reaching_target"] == 32
    assert q4["cluster_bootstrap_ci"]["ci_lower"] == 32
    assert target["minimum_channels_by_trial_count"]["32"]["smallest_channel_count_reaching_target"] == 4


def test_joint_bootstrap_keeps_finite_interval_when_censoring_is_below_upper_percentile():
    summaries = []
    for i in range(12):
        value = 0.6 if i < 10 else 0.0
        summaries.append({
            "session": f"s{i}",
            "native_debiased_median": 1.0,
            "joint_debiased_median": {"16": {"2": value}},
        })
    result = bootstrap_ci_for_joint_crossing(
        16, summaries, 0.5, [16], 2, True, 2000, "low_censor_test")
    assert 0 < result["n_bootstrap_right_censored"] < 50
    assert result["status"] == "computed"
    assert result["ci_upper"] == 16


# ---------------------------------------------------------------------------------------------------
# End-to-end synthetic checks: the ceiling case and the pure-noise null case.
# ---------------------------------------------------------------------------------------------------

def _planted_direction_activity(n_trials: int, n_units: int, seed: int) -> np.ndarray:
    """Trials with a REAL shared direction plus per-trial noise -- enough structure for a channel subset
    to partially recover the full-population direction."""
    rng = np.random.default_rng(seed)
    base_direction = np.abs(rng.standard_normal(n_units)) + 1.0
    base_direction /= np.linalg.norm(base_direction)
    trial_scale = rng.uniform(20.0, 60.0, size=n_trials)
    lambdas = trial_scale[:, None] * base_direction[None, :]
    return rng.poisson(lambdas).astype(float)


def test_full_population_recovery_is_at_the_ceiling():
    activity = _planted_direction_activity(n_trials=80, n_units=16, seed=42)
    cells = session_joint_cells(activity, channel_rungs=[NATIVE_LABEL], trial_rungs=[NATIVE_LABEL],
                                 n_unit_draws=50, n_trial_draws=20, seed_tag="ceiling_test")
    observed = cells[(NATIVE_LABEL, NATIVE_LABEL)]["observed"]
    assert observed.size == 50
    assert observed == pytest.approx(np.ones(50), abs=1e-9)


def test_calibration_ladder_keeps_the_score_set_and_target_fixed():
    activity = _planted_direction_activity(n_trials=80, n_units=16, seed=44)
    cells = session_joint_cells(
        activity, channel_rungs=[NATIVE_LABEL], trial_rungs=[8, NATIVE_LABEL],
        n_unit_draws=5, n_trial_draws=3, seed_tag="fixed_score_test")
    ref_idx, score_idx = reference_score_split(np.arange(80), "fixed_score_test|fixed_split")
    local_idx = draw_index_subset(len(ref_idx), 8, "fixed_score_test|T=8|draw=0|select")
    full_direction = _reference_direction(activity[ref_idx])
    calibration_direction = _reference_direction(activity[ref_idx[local_idx]])
    target = _deviation_against_direction(activity[score_idx], full_direction)
    candidate = _deviation_against_direction(activity[score_idx], calibration_direction)
    assert cells[(8, NATIVE_LABEL)]["observed"][0] == pytest.approx(recovery_score(target, candidate))


def test_pure_noise_recovery_is_lower_than_planted_structure():
    rng = np.random.default_rng(7)
    noise_activity = rng.standard_normal((400, 200))  # zero mean: no true direction of any kind
    noise_cells = session_joint_cells(noise_activity, channel_rungs=[4, NATIVE_LABEL], trial_rungs=[NATIVE_LABEL],
                                       n_unit_draws=50, n_trial_draws=20, seed_tag="noise_test")
    obs, null = noise_cells[(NATIVE_LABEL, 4)]["observed"], noise_cells[(NATIVE_LABEL, 4)]["null"]
    finite = np.isfinite(obs) & np.isfinite(null)
    noise_debiased_median = np.median(obs[finite] - null[finite])
    assert abs(noise_debiased_median) < 0.2, f"debiased median {noise_debiased_median} not near zero"

    planted_activity = _planted_direction_activity(n_trials=400, n_units=200, seed=99)
    planted_cells = session_joint_cells(planted_activity, channel_rungs=[4, NATIVE_LABEL], trial_rungs=[NATIVE_LABEL],
                                         n_unit_draws=50, n_trial_draws=20, seed_tag="planted_test")
    obs_p, null_p = planted_cells[(NATIVE_LABEL, 4)]["observed"], planted_cells[(NATIVE_LABEL, 4)]["null"]
    finite_p = np.isfinite(obs_p) & np.isfinite(null_p)
    planted_debiased_median = np.median(obs_p[finite_p] - null_p[finite_p])
    assert planted_debiased_median > noise_debiased_median + 0.2, (
        "real planted structure should be clearly more recoverable than pure noise at the same channel count")


def test_independent_reference_recovery_separates_structure_from_noise():
    rng = np.random.default_rng(17)
    noise = independent_reference_cells(rng.standard_normal((600, 120)), [8], 100, "noise")
    planted = independent_reference_cells(_planted_direction_activity(600, 120, 18), [8], 100, "planted")
    noise_delta = noise[8]["observed"] - noise[8]["null"]
    planted_delta = planted[8]["observed"] - planted[8]["null"]
    noise_delta = noise_delta[np.isfinite(noise_delta)]
    planted_delta = planted_delta[np.isfinite(planted_delta)]
    assert abs(float(np.median(noise_delta))) < 0.15
    assert float(np.median(planted_delta)) > float(np.median(noise_delta)) + 0.15


def test_channel_rung_draws_native_observation_is_deterministic_with_matched_null_draws():
    activity = _planted_direction_activity(n_trials=40, n_units=10, seed=3)
    ref, score = activity[:20], activity[20:]
    direction = _reference_direction(ref)
    d_full = _deviation_against_direction(score, direction)
    result = _channel_rung_draws(ref, score, d_full, NATIVE_LABEL, n_units_full=10, n_draws=50, seed_tag="tag")
    assert result["observed"].size == 50
    assert result["null"].size == 50
    assert result["observed"] == pytest.approx(np.ones(50), abs=1e-9)


def test_corpus_curve_uses_a_fixed_complete_session_cohort_and_common_grid():
    sessions = [
        {"session": "eight_a", "activity_by_unit": _planted_direction_activity(64, 8, 10)},
        {"session": "eight_b", "activity_by_unit": _planted_direction_activity(64, 8, 11)},
        {"session": "twelve_a", "activity_by_unit": _planted_direction_activity(64, 12, 12)},
        {"session": "twelve_b", "activity_by_unit": _planted_direction_activity(64, 12, 13)},
        {"session": "excluded_hundred", "activity_by_unit": np.zeros((64, 100))},
    ]
    result = run_corpus_recovery(sessions, "heterogeneous_synthetic", compute_axis=False)
    assert result["common_numeric_channel_grid"] == [2, 4, 8]
    assert result["common_numeric_calibration_trial_grid"] == [8, 16, 32]
    assert set(result["joint_channel_calibration_trial_table"]) == {"8", "16", "32", "native"}
    assert result["joint_fidelity_specification"]["status"] == "computed"
    assert result["n_sessions_complete_joint_grid"] == 4
    assert result["independent_reference_diagnostic"]["status"] == "computed"
    assert set(result["channel_only_recovery_curve_debiased_median_by_q"]) == {"2", "4", "8", "native"}
    for q in ("2", "4", "8", "native"):
        accounting = result["channel_curve_session_accounting_by_q"][q]
        assert accounting["n_sessions_denominator"] == 5
        assert accounting["n_sessions_eligible"] == 5
        assert accounting["n_sessions_used"] == 4
        assert accounting["n_sessions_not_used"] == 1
    assert result["native_full_unit_count"] == {"median": 10.0, "min": 8, "max": 12}
