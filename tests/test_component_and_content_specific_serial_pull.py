"""Tests for scripts/run_component_and_content_specific_serial_pull.py -- the pieces that could
silently break this leg without erroring: the per-trial content-specific-pull geometry against a
hand-computed example, the bias-only session-offset control and its exact void name, both decision-cell
classifiers against every named cell, the within-item-count-level combination's refusal to fall back to
a pooled-across-level estimator, the reproduction gate's session-limit exemption, and the two planted-data
cases (recovery and independence) required to show the deviation and the content-specific pull are
genuinely separable quantities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_component_and_content_specific_serial_pull as m  # noqa: E402
from run_component_and_content_specific_serial_pull import (  # noqa: E402
    _bias_only_branch, _bias_only_content_pull, _bias_only_reproduces, _content_pull_survival_branch,
    _content_specific_pull_per_trial, _deviation_survival_branch, _gate_result_for_run, _level_split_stats,
    _trial_count_weighted,
)
from provenance import _json_safe, restore_checkpoint  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Per-trial content-specific pull, hand-computed
# ---------------------------------------------------------------------------------------------------

def test_content_specific_pull_hand_computed():
    # 13 trials, 3 units, 3 classes of >=4 members each so every class is eligible.
    activity = np.zeros((13, 3))
    activity[0:4] = [1.0, 0.0, 0.0]      # class 0
    activity[4:8] = [0.0, 1.0, 0.0]      # class 1
    activity[8:12] = [0.0, 0.0, 1.0]     # class 2
    activity[12] = [1.0, 1.0, 2.0]       # trial 12: own class 0, predecessor (trial 11) is class 2
    label = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0], dtype=float)

    pull = _content_specific_pull_per_trial(activity, label)

    # Trial 12: prev class = 2 (mean direction [0,0,1]), other eligible class = {1} (mean [0,1,0]).
    # Unit direction of trial 12 = [1,1,2]/sqrt(6). pull_prev = 2/sqrt(6), pull_other = 1/sqrt(6).
    expected = 2.0 / np.sqrt(6.0) - 1.0 / np.sqrt(6.0)
    assert pull[12] == pytest.approx(expected, abs=1e-9)

    # Trial 1: own class == predecessor's class (both 0) -> undefined by construction.
    assert np.isnan(pull[1])


def test_content_specific_pull_undefined_when_too_few_classes():
    activity = np.tile([1.0, 0.0], (10, 1))
    label = np.array([0.0] * 5 + [1.0] * 5)  # only 2 classes -- SHARP_TEST_MIN_CLASSES is 3
    pull = _content_specific_pull_per_trial(activity, label)
    assert np.all(np.isnan(pull))


# ---------------------------------------------------------------------------------------------------
# Bias-only session-offset control
# ---------------------------------------------------------------------------------------------------

def test_bias_only_content_pull_collapses_to_the_session_mean():
    content_pull = np.array([1.0, 2.0, np.nan, 3.0])
    bias = _bias_only_content_pull(content_pull)
    assert bias[0] == bias[1] == bias[3] == pytest.approx(2.0)
    assert np.isnan(bias[2])


def test_bias_only_reproduces_when_significance_and_sign_match():
    real = {"status": "tested", "significant": True, "mean_value": 0.05}
    bias = {"status": "tested", "significant": True, "mean_value": 0.02}
    assert _bias_only_reproduces(real, bias) is True


def test_bias_only_does_not_reproduce_on_sign_disagreement():
    real = {"status": "tested", "significant": True, "mean_value": 0.05}
    bias = {"status": "tested", "significant": True, "mean_value": -0.02}
    assert _bias_only_reproduces(real, bias) is False


def test_bias_only_does_not_reproduce_on_significance_disagreement():
    real = {"status": "tested", "significant": True, "mean_value": 0.05}
    bias = {"status": "tested", "significant": False, "mean_value": 0.001}
    assert _bias_only_reproduces(real, bias) is False


def test_bias_only_reproduces_trivially_when_both_flat():
    real = {"status": "tested", "significant": False, "mean_value": 0.001}
    bias = {"status": "tested", "significant": False, "mean_value": -0.0005}
    assert _bias_only_reproduces(real, bias) is True


def test_bias_only_voiding_fires_by_its_exact_pre_declared_name():
    # Synthesised case where all the signal reproduces under the session-level-offset replacement.
    real = {"status": "tested", "significant": True, "mean_value": 0.08}
    bias = {"status": "tested", "significant": True, "mean_value": 0.06}
    reproduces = _bias_only_reproduces(real, bias)
    assert reproduces is True
    assert _bias_only_branch(reproduces) == "content_pull_control_not_separable_from_a_session_level_offset"


def test_bias_only_branch_when_not_a_session_level_offset():
    assert _bias_only_branch(False) == "content_pull_control_is_not_a_session_level_offset"


# ---------------------------------------------------------------------------------------------------
# Deviation-survival classifier -- every named cell
# ---------------------------------------------------------------------------------------------------

def _tested(mean_value: float, significant: bool, n=20) -> dict:
    return {"status": "tested", "mean_value": mean_value, "significant": significant, "n_sessions": n}


def _mdd(value: float) -> dict:
    return {"minimum_detectable_paired_difference_at_80pct_power": {"status": "computed", "mdd": value}}


def test_deviation_survival_component_survives_content_pull():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.08, True),
              "partial_controlling_consecutive_trial_alignment": _tested(0.09, True)}
    paired = {"status": "tested", "significant": False}
    assert _deviation_survival_branch(pooled, paired)["branch"] == \
        "accuracy_predicting_component_survives_the_content_specific_serial_pull"


def test_deviation_survival_component_is_the_content_specific_pull():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.01, False),
              "partial_controlling_consecutive_trial_alignment": _tested(0.09, True)}
    paired = {"status": "tested", "significant": True, **_mdd(0.02)}
    assert _deviation_survival_branch(pooled, paired)["branch"] == \
        "accuracy_predicting_component_is_the_content_specific_serial_pull"


def test_deviation_survival_no_ordering_when_paired_test_not_significant():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.01, False),
              "partial_controlling_consecutive_trial_alignment": _tested(0.09, True)}
    paired = {"status": "tested", "significant": False, **_mdd(0.2)}
    assert _deviation_survival_branch(pooled, paired)["branch"] == \
        "content_specific_pull_control_removes_the_link_but_not_more_than_consecutive_trial_alignment_does"


def test_deviation_survival_powered_null():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(-0.08, True),
              "partial_controlling_consecutive_trial_alignment": _tested(-0.07, True)}
    paired = {"status": "tested", "significant": False, **_mdd(0.02)}  # mdd 0.02 < raw_effect 0.1
    assert _deviation_survival_branch(pooled, paired)["branch"] == "powered_null_the_two_controls_are_not_distinguishable"


def test_deviation_survival_inconclusive_below_detection_floor():
    # cp_state must land on "opposite_sign_significant" (not "not_significant") to reach this cell --
    # the "not significant" cases are claimed earlier in the first-match-wins order by the two cells above.
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(-0.05, True),
              "partial_controlling_consecutive_trial_alignment": _tested(0.01, False)}
    paired = {"status": "tested", "significant": False, **_mdd(0.5)}  # mdd 0.5 >= raw_effect 0.1
    assert _deviation_survival_branch(pooled, paired)["branch"] == "inconclusive_below_detection_floor"


def test_deviation_survival_not_computable():
    pooled = {"raw": {"status": "not_computed"}, "partial_controlling_content_specific_pull": {"status": "not_computed"},
              "partial_controlling_consecutive_trial_alignment": {"status": "not_computed"}}
    assert _deviation_survival_branch(pooled, {"status": "not_computed"})["branch"] == "not_computable"


# ---------------------------------------------------------------------------------------------------
# Content-pull-survival classifier -- every named cell
# ---------------------------------------------------------------------------------------------------

def test_content_pull_survival_deviation_carries_it():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.08, True),
              "raw_content_pull_to_behaviour": _tested(0.07, True), "partial_controlling_the_deviation": _tested(0.01, False)}
    paired = {"status": "tested", "significant": True}
    assert _content_pull_survival_branch(pooled, paired)["branch"] == \
        "the_deviation_carries_the_behavioural_link_and_the_serial_pull_does_not"


def test_content_pull_survival_serial_pull_carries_it():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.01, False),
              "raw_content_pull_to_behaviour": _tested(0.07, True), "partial_controlling_the_deviation": _tested(0.06, True)}
    paired = {"status": "tested", "significant": True}
    assert _content_pull_survival_branch(pooled, paired)["branch"] == \
        "the_serial_pull_carries_the_behavioural_link_and_the_deviation_does_not"


def test_content_pull_survival_both_independent():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.08, True),
              "raw_content_pull_to_behaviour": _tested(0.07, True), "partial_controlling_the_deviation": _tested(0.06, True)}
    paired = {"status": "tested", "significant": True}
    assert _content_pull_survival_branch(pooled, paired)["branch"] == "both_observables_carry_it_independently"


def test_content_pull_survival_neither_survives():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.01, False),
              "raw_content_pull_to_behaviour": _tested(0.07, True), "partial_controlling_the_deviation": _tested(0.01, False)}
    paired = {"status": "tested", "significant": True}
    assert _content_pull_survival_branch(pooled, paired)["branch"] == "neither_observable_survives_the_other"


def test_content_pull_survival_ordering_not_established():
    pooled = {"raw": _tested(0.1, True), "partial_controlling_content_specific_pull": _tested(0.08, True),
              "raw_content_pull_to_behaviour": _tested(0.07, True), "partial_controlling_the_deviation": _tested(0.01, False)}
    paired = {"status": "tested", "significant": False}
    assert _content_pull_survival_branch(pooled, paired)["branch"] == "ordering_not_established_by_a_paired_test"


def test_content_pull_survival_not_computable():
    pooled = {"raw": {"status": "not_computed"}, "raw_content_pull_to_behaviour": {"status": "not_computed"},
              "partial_controlling_content_specific_pull": {"status": "not_computed"},
              "partial_controlling_the_deviation": {"status": "not_computed"}}
    assert _content_pull_survival_branch(pooled, {"status": "not_computed"})["branch"] == "not_computable"


# ---------------------------------------------------------------------------------------------------
# Within-item-count-level combination never falls back to the pooled-across-level estimator
# ---------------------------------------------------------------------------------------------------

def test_within_level_combination_survives_a_simpsons_reversal():
    rng = np.random.default_rng(0)
    n_per_level = 150

    # Two item-count levels, each with a genuine POSITIVE within-level deviation-outcome slope, but
    # offset so far apart between levels (high deviation / low outcome in level 1, the reverse in
    # level 2) that the naive pooled-across-level correlation reverses sign -- the exact structure the
    # mandate names (+0.019673 within-level vs -0.012147 pooled, same sessions, same variable).
    dev1 = rng.normal(10.0, 0.5, n_per_level)
    out1 = -10.0 + 2.0 * (dev1 - 10.0) + rng.normal(0.0, 0.3, n_per_level)
    dev2 = rng.normal(-10.0, 0.5, n_per_level)
    out2 = 10.0 + 2.0 * (dev2 + 10.0) + rng.normal(0.0, 0.3, n_per_level)

    deviation = np.concatenate([dev1, dev2])
    outcome = np.concatenate([out1, out2])
    item_count = np.concatenate([np.full(n_per_level, 1.0), np.full(n_per_level, 2.0)])
    n = len(outcome)
    content_pull = rng.normal(size=n)  # unrelated nuisance covariate, defined everywhere
    lag1_align = rng.normal(size=n)
    spike_count = rng.normal(size=n)
    trial_index = np.arange(n, dtype=float)

    naive_pooled_r = float(np.corrcoef(deviation, outcome)[0, 1])
    assert naive_pooled_r < 0.0  # the reversal is really there in this synthetic dataset

    old_n_perm = m.N_PERM
    m.N_PERM = 200
    try:
        result = _level_split_stats(outcome, deviation, content_pull, lag1_align, spike_count, trial_index,
                                     item_count, sign=1.0, seed_prefix="simpson_test")
    finally:
        m.N_PERM = old_n_perm

    within_raw = result["within_item_count_level"]["raw"]
    assert within_raw > 0.0  # the genuine within-level slope, not the reversed pooled one
    assert within_raw != pytest.approx(naive_pooled_r, abs=0.1)
    assert result["n_levels_tested"] == 2

    # And the weighted-average arithmetic itself: match a hand computation from the two levels' raw r's.
    r1 = float(np.corrcoef(dev1, out1)[0, 1])
    r2 = float(np.corrcoef(dev2, out2)[0, 1])
    hand_weighted = _trial_count_weighted([(n_per_level, r1), (n_per_level, r2)])
    assert within_raw == pytest.approx(hand_weighted, abs=1e-6)


# ---------------------------------------------------------------------------------------------------
# The reproduction gate under a session limit -- not a smoke-test-safe comparison against a delivered
# reference fit on the full corpus, so it must not be evaluated (and must never report non-reproduction)
# when a session limit is set; it runs for real, unmodified, only when no limit is set.
# ---------------------------------------------------------------------------------------------------

def test_gate_under_a_session_limit_is_not_evaluated_even_if_the_computed_gate_failed():
    computed = {"status": "not_reproduced", "checks": {"watters_deviation_gate_r": False}}
    result = _gate_result_for_run(computed, session_limit=3, watters_n_loaded=3, watters_n_delivered=41)
    assert result["status"] == "not_evaluated_under_a_session_limit"
    assert result["status"] != "not_reproduced"
    assert result["session_limit"] == 3
    assert result["watters_n_sessions_loaded_under_the_limit"] == 3
    assert result["watters_n_sessions_the_delivered_reference_values_were_computed_on"] == 41
    assert result["diagnostic_gate_result_computed_under_the_limit_not_a_verdict"] is computed


def test_gate_with_no_session_limit_passes_the_computed_result_through_unmodified():
    computed = {"status": "reproduced_exactly", "checks": {"watters_deviation_gate_r": True}}
    result = _gate_result_for_run(computed, session_limit=None, watters_n_loaded=41, watters_n_delivered=41)
    assert result is computed
    assert result["status"] == "reproduced_exactly"


def test_restore_arrays_round_trips_a_checkpointed_session_bundle():
    # A stale checkpoint entry re-read from disk is plain JSON: numpy ndarrays flattened to nested
    # lists, NaN written as null, session-name strings and status strings untouched.
    # provenance.restore_checkpoint must undo exactly that, so a cached bundle behaves like a
    # freshly-computed one to every downstream consumer that indexes, reshapes or vectorises its
    # fields. This module no longer keeps its own copy of that repair (previously named
    # _restore_arrays here) -- every checkpointing analysis script now shares the one in provenance.py.
    original = {
        "session": "s01",
        "activity_by_unit": np.array([[1.0, 2.0], [3.0, np.nan]]),
        "outcome_raw": np.array([1.0, 0.0, 1.0]),
        "usable": np.array([True, False, True]),
        "trial_index": np.arange(3, dtype=float),
        "counts": np.arange(12, dtype=float).reshape(3, 2, 2),
        "n_trials_total": 3,
    }
    round_tripped = json.loads(json.dumps(_json_safe(original), allow_nan=False))
    restored = restore_checkpoint(round_tripped)

    assert restored["session"] == "s01"
    assert restored["n_trials_total"] == 3
    np.testing.assert_array_equal(restored["outcome_raw"], original["outcome_raw"])
    assert restored["outcome_raw"].dtype == np.float64
    np.testing.assert_array_equal(restored["usable"], original["usable"])
    assert restored["usable"].dtype == bool
    np.testing.assert_array_equal(restored["counts"], original["counts"])
    assert restored["counts"].shape == (3, 2, 2)
    assert restored["activity_by_unit"][1, 0] == pytest.approx(3.0)
    assert np.isnan(restored["activity_by_unit"][1, 1])
    # the restored fields index the way a fresh bundle's fields do
    assert restored["outcome_raw"][restored["usable"]].tolist() == [1.0, 1.0]


def test_gate_with_no_session_limit_still_reports_genuine_non_reproduction():
    computed = {"status": "not_reproduced", "checks": {"watters_deviation_gate_r": False}}
    result = _gate_result_for_run(computed, session_limit=None, watters_n_loaded=41, watters_n_delivered=41)
    assert result is computed
    assert result["status"] == "not_reproduced"


# ---------------------------------------------------------------------------------------------------
# Planted recovery / planted independence (mandate's required synthetic tests)
# ---------------------------------------------------------------------------------------------------

def _synthetic_trial_arrays(rng, n, deviation_is_content_pull=True):
    content_pull = rng.normal(size=n)
    if deviation_is_content_pull:
        deviation = content_pull + rng.normal(0.0, 0.05, n)  # deviation ~= content_pull, tightly
    else:
        deviation = rng.normal(size=n)  # independent of content_pull
    outcome = -content_pull + rng.normal(0.0, 0.3, n)  # behaviour driven only by content_pull
    if not deviation_is_content_pull:
        outcome = outcome - deviation  # planted-independence case: outcome also driven by deviation, separately
    lag1_align = rng.normal(size=n)
    spike_count = rng.normal(size=n)
    trial_index = np.arange(n, dtype=float)
    return outcome, deviation, content_pull, lag1_align, spike_count, trial_index


def test_planted_recovery_deviation_link_is_absorbed_by_the_content_pull_partial():
    rng = np.random.default_rng(1)
    n = 300
    outcome, deviation, content_pull, lag1_align, spike_count, trial_index = _synthetic_trial_arrays(
        rng, n, deviation_is_content_pull=True)

    old_n_perm = m.N_PERM
    m.N_PERM = 500
    try:
        result = _level_split_stats(outcome, deviation, content_pull, lag1_align, spike_count, trial_index,
                                     item_count=None, sign=1.0, seed_prefix="planted_recovery")
    finally:
        m.N_PERM = old_n_perm

    level = result["per_level"]["all"]
    assert level["raw"]["status"] == "computed"
    assert level["raw"]["p_value"] <= 0.05  # deviation ~= content_pull, so raw deviation-outcome link is real
    # once content_pull is partialled out, deviation's residual is just noise -> link absorbed
    assert level["partial_controlling_content_specific_pull"]["p_value"] > 0.05
    assert abs(level["partial_controlling_content_specific_pull"]["r"]) < abs(level["raw"]["r"])


def test_planted_independence_neither_partial_absorbs_the_other():
    rng = np.random.default_rng(2)
    n = 300
    outcome, deviation, content_pull, lag1_align, spike_count, trial_index = _synthetic_trial_arrays(
        rng, n, deviation_is_content_pull=False)

    old_n_perm = m.N_PERM
    m.N_PERM = 500
    try:
        result = _level_split_stats(outcome, deviation, content_pull, lag1_align, spike_count, trial_index,
                                     item_count=None, sign=1.0, seed_prefix="planted_independence")
    finally:
        m.N_PERM = old_n_perm

    level = result["per_level"]["all"]
    # Both observables independently drive outcome -> both raw links are real and neither control absorbs
    # the other's link.
    assert level["raw"]["p_value"] <= 0.05
    assert level["raw_content_pull_to_behaviour"]["p_value"] <= 0.05
    assert level["partial_controlling_content_specific_pull"]["p_value"] <= 0.05
    assert level["partial_controlling_the_deviation"]["p_value"] <= 0.05
