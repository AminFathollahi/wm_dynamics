"""Tests for scripts/run_rate_free_state_geometry_behavior_link.py: the
leave-one-out construction of the rate-free deviation observable (the
subtlest part -- a trial must not contribute to its own reference), and the
pre-declared branch logic including the orthogonality gate's void branch and
the disclosed gap branch for a raw-significant result that does not survive
the joint control."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, _classify, rate_free_state_deviation,
)


def test_leave_one_out_excludes_trial_from_its_own_reference():
    # Trial 0 points opposite trials 1 and 2 (which point together). The CORRECT leave-one-out reference
    # for trial 0 is the mean of trials 1 and 2 ONLY -- if trial 0 wrongly contributed to its own
    # reference (the bug this test exists to catch), the reference would be pulled toward trial 0 and its
    # deviation would come out lower than the true leave-one-out value computed here by hand.
    activity = np.array([
        [1.0, 0.0],   # trial 0: unit vector [1, 0]
        [0.0, 1.0],   # trial 1: unit vector [0, 1]
        [0.0, 1.0],   # trial 2: unit vector [0, 1]
    ])
    deviation = rate_free_state_deviation(activity)

    # Correct LOO reference for trial 0 = mean([0,1],[0,1]) = [0,1] (already unit norm).
    # cosine([1,0], [0,1]) = 0 -> deviation = 1 - 0 = 1.0.
    assert deviation[0] == pytest.approx(1.0, abs=1e-9)

    # The bug this test guards against: if trial 0 were (wrongly) included in its own reference, the
    # reference would be mean([1,0],[0,1],[0,1]) = [1/3, 2/3], normalised to [0.4472, 0.8944], and
    # cosine([1,0], that) = 0.4472 -> a biased deviation of 0.5528, NOT 1.0.
    biased_self_inclusive_reference = np.array([1 / 3, 2 / 3])
    biased_self_inclusive_reference /= np.linalg.norm(biased_self_inclusive_reference)
    biased_deviation_0 = 1.0 - np.dot(np.array([1.0, 0.0]), biased_self_inclusive_reference)
    assert deviation[0] != pytest.approx(biased_deviation_0, abs=1e-6)

    # By symmetry, trials 1 and 2 have the same LOO reference structure: for trial 1, LOO reference is
    # mean(trial0, trial2) = mean([1,0],[0,1]) = [0.5,0.5], normalised [0.7071, 0.7071].
    # cosine([0,1], that) = 0.7071 -> deviation = 1 - 0.7071 = 0.2929.
    assert deviation[1] == pytest.approx(1.0 - 1 / np.sqrt(2), abs=1e-6)
    assert deviation[2] == pytest.approx(1.0 - 1 / np.sqrt(2), abs=1e-6)


def test_leave_one_out_excludes_zero_activity_trials_from_others_reference():
    # A trial with zero total activity has no defined direction (NaN) and must not pull down the
    # leave-one-out denominator or contribute a spurious zero vector to another trial's reference.
    activity = np.array([
        [1.0, 0.0],
        [0.0, 0.0],   # undefined direction
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    deviation = rate_free_state_deviation(activity)
    assert np.isnan(deviation[1])
    # Trials 0, 2, 3 all point the same direction; each one's LOO reference (the other two, both [1,0])
    # is also [1,0], so their deviation should be exactly 0 -- the zero-activity trial contributed nothing.
    assert deviation[0] == pytest.approx(0.0, abs=1e-9)
    assert deviation[2] == pytest.approx(0.0, abs=1e-9)
    assert deviation[3] == pytest.approx(0.0, abs=1e-9)


def _tested(mean_value: float, significant: bool) -> dict:
    return {"status": "tested", "mean_value": mean_value, "significant": significant}


def test_orthogonality_gate_failure_voids_the_analysis_before_any_behavioural_branch():
    gate = _tested(mean_value=0.3, significant=True)
    raw = _tested(mean_value=0.2, significant=True)
    joint = _tested(mean_value=0.2, significant=True)
    assert _classify(gate, raw, joint, mdd=0.01) == "rate_free_state_observable_is_not_rate_free_and_this_analysis_is_void"


def test_raw_significant_and_joint_agrees_predicts_accuracy():
    gate = _tested(mean_value=0.02, significant=False)
    raw = _tested(mean_value=0.25, significant=True)
    joint = _tested(mean_value=0.20, significant=True)
    assert _classify(gate, raw, joint, mdd=0.05) == "rate_free_state_geometry_predicts_accuracy"


def test_raw_significant_but_joint_control_kills_it_is_its_own_disclosed_branch():
    # The gap the pre-declared rule itself names but does not assign a branch to: raw is significant, but
    # the joint partial is not significant (or flips sign) -- this must NOT be forced into the positive
    # branch, and must NOT silently fall through to a null branch either.
    gate = _tested(mean_value=0.02, significant=False)
    raw = _tested(mean_value=0.25, significant=True)
    joint_not_significant = _tested(mean_value=0.10, significant=False)
    assert _classify(gate, raw, joint_not_significant, mdd=0.05) == (
        "raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index"
    )
    joint_flips_sign = _tested(mean_value=-0.20, significant=True)
    assert _classify(gate, raw, joint_flips_sign, mdd=0.05) == (
        "raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index"
    )


def test_raw_not_significant_with_low_mdd_is_a_bounded_null():
    gate = _tested(mean_value=0.01, significant=False)
    raw = _tested(mean_value=0.03, significant=False)
    joint = _tested(mean_value=0.02, significant=False)
    mdd = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS - 0.01
    assert _classify(gate, raw, joint, mdd) == "no_rate_free_state_geometry_link_to_accuracy_above_the_reported_bound"


def test_raw_not_significant_with_high_mdd_is_underpowered():
    gate = _tested(mean_value=0.01, significant=False)
    raw = _tested(mean_value=0.03, significant=False)
    joint = _tested(mean_value=0.02, significant=False)
    mdd = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS + 0.05
    assert _classify(gate, raw, joint, mdd) == "underpowered_to_ask"


def test_not_computable_when_gate_or_raw_is_underpowered_by_construction():
    underpowered = {"status": "underpowered_by_construction", "n_sessions": 2}
    raw = _tested(mean_value=0.1, significant=False)
    assert _classify(underpowered, raw, raw, mdd=0.05) == "not_computable"
    gate = _tested(mean_value=0.01, significant=False)
    assert _classify(gate, underpowered, underpowered, mdd=0.05) == "not_computable"
