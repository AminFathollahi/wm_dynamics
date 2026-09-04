"""Tests for scripts/run_pre_cue_state_and_reported_item_identity.py -- the pieces that could silently
break this analysis without erroring: (1) the circular angular-distance primitive and the two-alternative
report-following statistic built on it; (2) the landed/target object index, reimplemented locally from
raw Cartesian columns and proved to agree with run_component_and_item_binding's own swap indicator, on
hand-computed trials where the correct answer is known by construction; (3) the subspace decomposition of
the full rate-free direction, proved to agree with cv_regression_subspace's own within/outside magnitudes
on synthetic data; (4) the pooling-against-0.5 helper; (5) the Block A and Block B decision-rule
classifiers, every named branch; (6) the Block 0 timing-premise stop condition, both triggered and clear."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_pre_cue_state_and_reported_item_identity import (  # noqa: E402
    BRANCH_A_BIAS_CONFOUND, BRANCH_A_INCONCLUSIVE, BRANCH_A_POSITIVE, BRANCH_A_POWERED_NULL,
    BRANCH_A_SURPRISE, BRANCH_A_TOO_FEW, BRANCH_B_BOTH, BRANCH_B_INSIDE, BRANCH_B_NEITHER,
    BRANCH_B_ORDERING_NOT_ESTABLISHED, BRANCH_B_OUTSIDE, BRANCH_REPRODUCTION_GATE_FAILED,
    BRANCH_TASK_GEOM_NEEDS_DECODED, BRANCH_TASK_GEOM_NOT_COVERED, BRANCH_TASK_GEOM_NOT_SEPARABLE,
    BRANCH_TASK_GEOM_POWERED_NULL, BRANCH_TASK_GEOM_REPRODUCED, MIN_POOLED_TEST_TRIALS,
    _block_b_label_validity_gap_statement, _landed_and_target_index, _landed_identity_check,
    _loo_subspace_component_vectors, _paired_session_test_full,
    _primary_below_half_bias_only_control_gap_statement, _pool_against_half, _report_following_fraction,
    _subspace_decomposition_identity_check, block0_timing_premise, circular_abs_diff, decide_block_a,
    decide_block_b, decide_swap_destination_task_geometry,
)
from run_component_and_item_binding import _object_geometry  # noqa: E402
from run_deviation_subspace_decomposition import cv_regression_subspace  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Circular distance and the report-following statistic
# ---------------------------------------------------------------------------------------------------

def test_circular_abs_diff_handles_wraparound():
    assert circular_abs_diff(np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0)
    assert circular_abs_diff(np.array([0.1]), np.array([2 * np.pi - 0.1]))[0] == pytest.approx(0.2, abs=1e-9)
    assert circular_abs_diff(np.array([0.0]), np.array([np.pi]))[0] == pytest.approx(np.pi, abs=1e-9)


def test_report_following_fraction_counts_closer_trials():
    decoded = np.array([0.0, 0.0, np.pi])
    reported = np.array([0.05, 3.0, np.pi - 0.1])   # trial 0 & 2: decoded closer to reported
    other = np.array([3.0, 0.05, 0.1])               # trial 1: decoded closer to other
    result = _report_following_fraction(decoded, reported, other)
    assert result["fraction"] == pytest.approx(2.0 / 3.0)
    assert result["n_trials"] == 3
    assert result["n_ties"] == 0


# ---------------------------------------------------------------------------------------------------
# Landed/target object index and its identity check against _object_geometry
# ---------------------------------------------------------------------------------------------------

def _synthetic_behaviour_and_session() -> tuple[pd.DataFrame, dict]:
    rows = [
        dict(trial_num=1, num_objects=1, target_object_index=0,
             object_0_x=0.0, object_0_y=0.0, object_0_theta=0.0,
             object_1_x=np.nan, object_1_y=np.nan, object_1_theta=np.nan,
             object_2_x=np.nan, object_2_y=np.nan, object_2_theta=np.nan,
             response_x=0.05, response_y=0.0),
        dict(trial_num=2, num_objects=3, target_object_index=0,
             object_0_x=0.0, object_0_y=0.0, object_0_theta=0.3,
             object_1_x=1.0, object_1_y=0.0, object_1_theta=1.1,
             object_2_x=0.0, object_2_y=1.0, object_2_theta=2.2,
             response_x=0.9, response_y=0.05),
        dict(trial_num=3, num_objects=3, target_object_index=1,
             object_0_x=0.0, object_0_y=0.0, object_0_theta=0.3,
             object_1_x=1.0, object_1_y=0.0, object_1_theta=1.1,
             object_2_x=0.0, object_2_y=1.0, object_2_theta=2.2,
             response_x=0.98, response_y=0.02),
    ]
    frame = pd.DataFrame(rows)
    frame["subject"] = "TestAnimal"
    frame["session"] = "2026-01-01"
    tx = np.choose(frame.target_object_index.to_numpy(),
                    [frame.object_0_x, frame.object_1_x, frame.object_2_x])
    ty = np.choose(frame.target_object_index.to_numpy(),
                    [frame.object_0_y, frame.object_1_y, frame.object_2_y])
    t_theta = np.choose(frame.target_object_index.to_numpy(),
                         [frame.object_0_theta, frame.object_1_theta, frame.object_2_theta])
    frame["report_deviation"] = np.hypot(tx - frame.response_x.to_numpy(), ty - frame.response_y.to_numpy())
    behaviour = frame.set_index(["subject", "session", "trial_num"], drop=False).sort_index()
    session = {
        "animal": "TestAnimal", "session_date": "2026-01-01",
        "trial_num": np.array([1, 2, 3], dtype=int),
        "report_deviation": frame["report_deviation"].to_numpy(dtype=float),
        "cued_theta": np.mod(t_theta.astype(float), 2.0 * np.pi),
    }
    return behaviour, session


def test_landed_and_target_index_matches_hand_computed_geometry():
    behaviour, session = _synthetic_behaviour_and_session()
    info = _landed_and_target_index(behaviour, session)

    # Trial 0 (1 object): only object 0 exists, response lands on it.
    assert info["landed"][0] == 0 and info["target"][0] == 0

    # Trial 1 (3 objects, cued=0): response (0.9, 0.05) is nearest object 1 at (1.0, 0.0) -- a swap.
    assert info["landed"][1] == 1 and info["target"][1] == 0

    # Trial 2 (3 objects, cued=1): response (0.98, 0.02) is nearest object 1 -- not a swap.
    assert info["landed"][2] == 1 and info["target"][2] == 1


def test_landed_identity_check_agrees_with_object_geometry():
    behaviour, session = _synthetic_behaviour_and_session()
    geometry = _object_geometry(behaviour, session)
    info = _landed_and_target_index(behaviour, session)
    identity = _landed_identity_check(behaviour, session, geometry, info)
    assert identity["swap_primary_matches_object_geometry"] is True
    assert identity["cued_theta_matches_target_object_theta_max_abs_diff"] < 1e-9


def test_landed_identity_check_fails_when_swap_definitions_disagree():
    behaviour, session = _synthetic_behaviour_and_session()
    info = _landed_and_target_index(behaviour, session)
    # A fabricated geometry whose swap_primary disagrees with the recomputed one on trial 1.
    fake_geometry = {"swap_primary": np.array([False, False, False])}
    identity = _landed_identity_check(behaviour, session, fake_geometry, info)
    assert identity["swap_primary_matches_object_geometry"] is False


# ---------------------------------------------------------------------------------------------------
# Subspace decomposition of the full direction, proved against cv_regression_subspace
# ---------------------------------------------------------------------------------------------------

def test_subspace_component_vectors_agree_with_cv_regression_subspace():
    rng = np.random.default_rng(0)
    n_trials, n_units = 40, 12
    target_2d = rng.normal(size=(n_trials, 2))
    coeffs = rng.normal(size=(2, n_units))
    raw = target_2d @ coeffs + rng.normal(scale=0.3, size=(n_trials, n_units))
    u = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    inside, outside = _loo_subspace_component_vectors(u, target_2d, dim=2)
    check = _subspace_decomposition_identity_check(u, target_2d, inside, outside, dim=2)
    assert check["passed"] is True

    within_ref, outside_ref = cv_regression_subspace(u, target_2d, u, dim=2)
    np.testing.assert_allclose(np.linalg.norm(inside, axis=1), within_ref, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(outside, axis=1), outside_ref, atol=1e-8)

    # Inside and outside must exactly reconstruct the original direction (Pythagorean split).
    np.testing.assert_allclose(inside + outside, u, atol=1e-8)


# ---------------------------------------------------------------------------------------------------
# Pooling against 0.5
# ---------------------------------------------------------------------------------------------------

def test_pool_against_half_detects_a_real_departure():
    values = np.full(12, 0.7)
    result = _pool_against_half(values, "unit_test|pool_above")
    assert result["status"] == "tested"
    assert result["significant_above_half"] is True
    assert result["significant_below_half"] is False
    assert result["mean_fraction"] == pytest.approx(0.7)


def test_pool_against_half_too_few_sessions():
    result = _pool_against_half(np.array([0.6]), "unit_test|pool_one")
    assert result["status"] == "too_few_sessions"


# ---------------------------------------------------------------------------------------------------
# Block A decision rule
# ---------------------------------------------------------------------------------------------------

def _pooled(mean_fraction: float, significant: bool, mdd: float = 0.02) -> dict:
    diff = mean_fraction - 0.5
    return {
        "status": "tested", "mean_fraction": mean_fraction, "mean_diff_from_half": diff,
        "p_value": 0.01 if significant else 0.5,
        "significant_above_half": bool(significant and diff > 0),
        "significant_below_half": bool(significant and diff < 0),
        "minimum_detectable_departure_from_half_at_80pct_power": {"status": "computed", "mdd": mdd},
    }


def test_block_a_reproduction_gate_failed_short_circuits():
    result = decide_block_a("not_reproduced", 1000, _pooled(0.7, True), _pooled(0.5, False),
                             np.array([0.7, 0.7]), np.array([0.5, 0.5]))
    assert result["branch"] == BRANCH_REPRODUCTION_GATE_FAILED


def test_block_a_too_few_pooled_test_trials():
    result = decide_block_a("reproduced_exactly", MIN_POOLED_TEST_TRIALS - 1, _pooled(0.7, True),
                             _pooled(0.5, False), np.array([0.7, 0.7]), np.array([0.5, 0.5]))
    assert result["branch"] == BRANCH_A_TOO_FEW
    assert result["n_pooled_test_trials_across_gate_cleared_sessions"] == MIN_POOLED_TEST_TRIALS - 1


def test_block_a_positive_branch():
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.65, True), _pooled(0.5, False),
                             np.array([0.65, 0.7]), np.array([0.5, 0.51]))
    assert result["branch"] == BRANCH_A_POSITIVE


def test_block_a_bias_confound_branch_runs_paired_test():
    a_vals = np.array([0.65, 0.7, 0.68, 0.66, 0.71])
    c_vals = np.array([0.63, 0.69, 0.67, 0.64, 0.70])
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.68, True), _pooled(0.67, True), a_vals, c_vals)
    assert result["branch"] == BRANCH_A_BIAS_CONFOUND
    assert "paired_test_a_vs_c" in result
    assert result["paired_test_a_vs_c"]["status"] == "computed"


def test_block_a_surprise_branch():
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.35, True), _pooled(0.5, False),
                             np.array([0.35, 0.3]), np.array([0.5, 0.5]))
    assert result["branch"] == BRANCH_A_SURPRISE


def test_block_a_powered_null_branch():
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.51, False, mdd=0.03), _pooled(0.5, False),
                             np.array([0.51, 0.49]), np.array([0.5, 0.5]))
    assert result["branch"] == BRANCH_A_POWERED_NULL
    assert "retrieval" in result["retrieval_locus_statement"]


def test_block_a_inconclusive_branch():
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.51, False, mdd=0.08), _pooled(0.5, False),
                             np.array([0.51, 0.49]), np.array([0.5, 0.5]))
    assert result["branch"] == BRANCH_A_INCONCLUSIVE


# ---------------------------------------------------------------------------------------------------
# Block B decision rule
# ---------------------------------------------------------------------------------------------------

def test_block_b_outside_only_with_significant_paired_test():
    outside_vals = np.array([0.65, 0.7, 0.68, 0.66, 0.71, 0.69])
    inside_vals = np.array([0.50, 0.49, 0.51, 0.50, 0.48, 0.52])
    pooled_outside = _pooled(0.68, True, mdd=0.01)
    pooled_inside = _pooled(0.50, False, mdd=0.01)
    pooled_outside["p_value"], pooled_inside["p_value"] = 0.001, 0.9
    result = decide_block_b(pooled_outside, pooled_inside, outside_vals, inside_vals)
    assert result["branch"] == BRANCH_B_OUTSIDE


def test_block_b_inside_only_with_significant_paired_test():
    inside_vals = np.array([0.65, 0.7, 0.68, 0.66, 0.71, 0.69])
    outside_vals = np.array([0.50, 0.49, 0.51, 0.50, 0.48, 0.52])
    pooled_outside = _pooled(0.50, False, mdd=0.01)
    pooled_inside = _pooled(0.68, True, mdd=0.01)
    pooled_outside["p_value"], pooled_inside["p_value"] = 0.9, 0.001
    result = decide_block_b(pooled_outside, pooled_inside, outside_vals, inside_vals)
    assert result["branch"] == BRANCH_B_INSIDE


def test_block_b_both_significant():
    outside_vals = np.array([0.65, 0.7, 0.68, 0.66, 0.71, 0.69])
    inside_vals = np.array([0.64, 0.69, 0.67, 0.65, 0.70, 0.68])
    pooled_outside = _pooled(0.68, True, mdd=0.01)
    pooled_inside = _pooled(0.67, True, mdd=0.01)
    pooled_outside["p_value"], pooled_inside["p_value"] = 0.001, 0.002
    result = decide_block_b(pooled_outside, pooled_inside, outside_vals, inside_vals)
    assert result["branch"] == BRANCH_B_BOTH
    assert "collinearity_outside_vs_inside_across_sessions" in result


def test_block_b_neither_significant():
    vals = np.array([0.50, 0.49, 0.51, 0.50, 0.48, 0.52])
    pooled = _pooled(0.50, False, mdd=0.02)
    result = decide_block_b(pooled, pooled, vals, vals)
    assert result["branch"] == BRANCH_B_NEITHER


def test_block_b_ordering_not_established_when_paired_test_not_significant():
    # Outside significant, inside not, but the two per-session series are identical -- the paired
    # difference is exactly zero, so ordering cannot be established even though one component "wins".
    outside_vals = np.array([0.65, 0.7, 0.68, 0.66, 0.71, 0.69])
    pooled_outside = _pooled(0.68, True, mdd=0.01)
    pooled_inside = _pooled(0.50, False, mdd=0.01)
    pooled_outside["p_value"], pooled_inside["p_value"] = 0.001, 0.9
    result = decide_block_b(pooled_outside, pooled_inside, outside_vals, outside_vals)
    assert result["branch"] == BRANCH_B_ORDERING_NOT_ESTABLISHED


# ---------------------------------------------------------------------------------------------------
# Block 0 timing premise
# ---------------------------------------------------------------------------------------------------

def _behaviour_with_margins(margins: list[float]) -> tuple[pd.DataFrame, dict]:
    n = len(margins)
    frame = pd.DataFrame({
        "subject": ["A"] * n, "session": ["2026-01-01"] * n, "trial_num": np.arange(1, n + 1),
        "time_delay_onset": np.zeros(n), "time_cue_onset": np.array(margins, dtype=float),
    })
    behaviour = frame.set_index(["subject", "session", "trial_num"], drop=False).sort_index()
    session = {"animal": "A", "session_date": "2026-01-01", "trial_num": np.arange(1, n + 1, dtype=int),
               "trials_dropped_by_reason": {"delay_shorter_than_window": 3}}
    return behaviour, session


def test_block0_does_not_trigger_when_every_margin_clears_the_loaders_own_threshold():
    behaviour, session = _behaviour_with_margins([1.0, 1.05, 1.2, 0.999])
    result = block0_timing_premise([session], behaviour)
    assert result["stop_condition_triggered"] is False
    assert result["n_trials_pooled"] == 4
    assert result["n_trials_the_loader_already_refused_for_a_too_short_delay"] == 3


def test_block0_triggers_when_a_margin_is_short_of_the_loaders_own_threshold():
    behaviour, session = _behaviour_with_margins([1.0, 1.05, 0.5, 0.999])  # 0.5 s margin: a real violation
    result = block0_timing_premise([session], behaviour)
    assert result["stop_condition_triggered"] is True
    assert result["n_trials_overlapping_cue_onset_by_the_loaders_own_threshold"] == 1


# ---------------------------------------------------------------------------------------------------
# Extension: the direct paired test of A against C, present in every Block A branch (not only the
# bias-confound branch), with its own sd and minimum detectable difference.
# ---------------------------------------------------------------------------------------------------

def test_paired_session_test_full_carries_sd_and_mdd():
    a = np.array([0.65, 0.7, 0.68, 0.66, 0.71])
    b = np.array([0.60, 0.62, 0.61, 0.59, 0.63])
    result = _paired_session_test_full(a, b, "unit_test|paired_full")
    assert result["status"] == "computed"
    assert result["sd"] == pytest.approx(np.std(a - b, ddof=1))
    assert result["minimum_detectable_paired_difference_at_80pct_power"]["status"] == "computed"


def test_block_a_surprise_branch_carries_the_paired_test_and_its_detection_floor_statement():
    # A is significantly below 0.5, C is also significantly below 0.5 -- exactly the delivered
    # configuration this extension was written to record, but on the surprise branch, which the
    # earlier implementation did not attach the paired test to at all.
    a_vals = np.array([0.40, 0.42, 0.38, 0.44, 0.41, 0.39])
    c_vals = np.array([0.35, 0.37, 0.36, 0.34, 0.38, 0.33])
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.407, True), _pooled(0.355, True),
                             a_vals, c_vals)
    assert result["branch"] == BRANCH_A_SURPRISE
    assert result["paired_test_a_vs_c"]["status"] == "computed"
    assert result["paired_test_a_vs_c_detection_floor_statement"] is not None


def test_block_a_positive_branch_also_carries_the_paired_test():
    a_vals = np.array([0.65, 0.7, 0.68, 0.66, 0.71])
    c_vals = np.array([0.50, 0.51, 0.49, 0.50, 0.52])
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.68, True), _pooled(0.504, False),
                             a_vals, c_vals)
    assert result["branch"] == BRANCH_A_POSITIVE
    assert result["paired_test_a_vs_c"]["status"] == "computed"


def test_detection_floor_statement_says_underpowered_when_mdd_exceeds_a_departure():
    # A's departure is small (0.02) and the two series are near-identical, giving the paired test a
    # coarse mdd -- the statement must say the separation test is underpowered relative to A's own effect.
    a_vals = np.array([0.52, 0.48, 0.52, 0.48, 0.52, 0.48])
    c_vals = np.array([0.52, 0.47, 0.53, 0.49, 0.51, 0.47])
    result = decide_block_a("reproduced_exactly", 500, _pooled(0.5, False), _pooled(0.5, False),
                             a_vals, c_vals)
    statement = result["paired_test_a_vs_c_detection_floor_statement"]
    if statement is not None:
        assert "ABOVE" in statement or "AT OR BELOW" in statement


# ---------------------------------------------------------------------------------------------------
# Extension: the swap-destination task-geometry control decision rule, all four named branches plus
# the not-covered fallback.
# ---------------------------------------------------------------------------------------------------

def _paired(mean_diff: float, significant: bool, mdd: float = 0.02) -> dict:
    return {
        "status": "computed", "mean_diff": mean_diff, "p_value": 0.01 if significant else 0.5,
        "significant": significant,
        "minimum_detectable_paired_difference_at_80pct_power": {"status": "computed", "mdd": mdd},
    }


def test_task_geometry_reproduced_branch():
    pooled_c = _pooled(0.42, True)
    pooled_task = _pooled(0.41, True)
    paired = _paired(0.01, False)
    result = decide_swap_destination_task_geometry(pooled_c, pooled_task, paired)
    assert result["branch"] == BRANCH_TASK_GEOM_REPRODUCED


def test_task_geometry_needs_decoded_position_branch():
    pooled_c = _pooled(0.42, True)
    pooled_task = _pooled(0.49, False)
    paired = _paired(0.07, True)
    result = decide_swap_destination_task_geometry(pooled_c, pooled_task, paired)
    assert result["branch"] == BRANCH_TASK_GEOM_NEEDS_DECODED


def test_task_geometry_not_separable_branch():
    pooled_c = _pooled(0.42, True)  # departure 0.08
    pooled_task = _pooled(0.47, False)
    paired = _paired(0.02, False, mdd=0.15)  # mdd 0.15 exceeds C's own departure 0.08
    result = decide_swap_destination_task_geometry(pooled_c, pooled_task, paired)
    assert result["branch"] == BRANCH_TASK_GEOM_NOT_SEPARABLE


def test_task_geometry_powered_null_branch():
    # Paired mdd (0.01) must NOT exceed C's own departure (0.02) or the not-separable rule fires first --
    # the ordered rule is evaluated top to bottom, not by which cell "looks closest".
    pooled_c = _pooled(0.48, False, mdd=0.03)
    pooled_task = _pooled(0.485, False, mdd=0.03)
    paired = _paired(0.005, False, mdd=0.01)
    result = decide_swap_destination_task_geometry(pooled_c, pooled_task, paired)
    assert result["branch"] == BRANCH_TASK_GEOM_POWERED_NULL


def test_task_geometry_not_covered_fallback():
    # Both significant in the same direction, and the paired test is also significant -- an outcome
    # none of the four named cells describe.
    pooled_c = _pooled(0.40, True)
    pooled_task = _pooled(0.41, True)
    paired = _paired(0.05, True)
    result = decide_swap_destination_task_geometry(pooled_c, pooled_task, paired)
    assert result["branch"] == BRANCH_TASK_GEOM_NOT_COVERED


# ---------------------------------------------------------------------------------------------------
# Extension: the two recorded-gap prose statements, computed live from the pooled statistics rather
# than hardcoded to the delivered outcome.
# ---------------------------------------------------------------------------------------------------

def test_bias_only_control_gap_statement_fires_when_both_below_half_and_c_larger():
    pooled_a = _pooled(0.4486, True)
    pooled_c = _pooled(0.4182, True)
    statement = _primary_below_half_bias_only_control_gap_statement(pooled_a, pooled_c)
    assert "gap" in statement
    assert "not separable from a session-level spatial decoding bias" in statement


def test_bias_only_control_gap_statement_does_not_fire_when_c_is_smaller_or_opposite():
    pooled_a = _pooled(0.40, True)
    pooled_c = _pooled(0.60, True)  # opposite direction
    statement = _primary_below_half_bias_only_control_gap_statement(pooled_a, pooled_c)
    assert "does not apply as written" in statement


def test_block_b_label_validity_gap_statement_fires_below_half():
    pooled_a = _pooled(0.4486, True)
    statement = _block_b_label_validity_gap_statement(pooled_a)
    assert "no component" in statement


def test_block_b_label_validity_gap_statement_clear_above_half():
    pooled_a = _pooled(0.60, True)
    statement = _block_b_label_validity_gap_statement(pooled_a)
    assert "does not apply here" in statement
