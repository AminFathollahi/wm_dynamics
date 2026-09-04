"""Tests for scripts/run_component_effect_size_and_anatomy.py: the new glue
this module introduces (decile contrast, spike-count-matched decile
contrast, contiguous-fold cross-validated discrimination, the trial-count-
weighted combiner) and its two pre-declared branch classifiers (Block A's
cross-validated-discrimination branch, Block B's cross-group localisation
branch), all against known small examples worked out by hand or with an
unambiguous ground truth. No real corpus data is touched -- the reused
estimators (rate_free_state_deviation, partial_correlation_permutation_test,
slope_across_sessions_test, and so on) are already covered by their own
modules' tests and are not re-tested here."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_component_effect_size_and_anatomy import (  # noqa: E402
    _block_a_cv_branch, _contiguous_folds, _cv_discrimination_binary, _cv_discrimination_continuous,
    _decile_contrast, _localisation_branch, _matched_contrast_reachability, _pairwise_cell_reachability,
    _pairwise_comparison_note, _pairwise_predictor_tests, _spike_count_matched_contrast, _trial_count_weighted,
)


# --------------------------------------------------------------------------------------------------------
# Decile contrast
# --------------------------------------------------------------------------------------------------------

def test_decile_contrast_matches_hand_computed_top_and_bottom_groups():
    # 20 trials, deviation = 0..19 (already sorted), worse_behaviour = deviation / 19 (perfectly monotone).
    # fraction=0.10 -> k = 2: bottom = deviation {0,1} -> worse {0, 1/19}; top = deviation {18,19} -> worse
    # {18/19, 1}. contrast = mean(top) - mean(bottom) = (18/19+1)/2 - (0+1/19)/2 = (37/19)/2 - (1/19)/2 = 36/38.
    deviation = np.arange(20, dtype=float)
    worse = deviation / 19.0
    result = _decile_contrast(deviation, worse, 0.10)
    assert result["status"] == "computed"
    assert result["n_per_group"] == 2
    assert result["contrast"] == pytest.approx(36.0 / 38.0, abs=1e-9)


def test_decile_contrast_too_few_trials_is_named_not_silently_skipped():
    result = _decile_contrast(np.arange(4, dtype=float), np.arange(4, dtype=float), 0.10)
    assert result["status"] == "too_few_trials"
    assert result["n_trials"] == 4


def test_decile_contrast_is_invariant_to_deviation_scale_and_offset():
    # The contrast depends only on the RANK of deviation, not its scale -- a monotone rescaling must give
    # the identical top/bottom trial sets and therefore the identical contrast.
    rng = np.random.default_rng(0)
    deviation = rng.normal(size=40)
    worse = rng.normal(size=40)
    a = _decile_contrast(deviation, worse, 0.10)
    b = _decile_contrast(deviation * 3.0 + 7.0, worse, 0.10)
    assert a["contrast"] == pytest.approx(b["contrast"])


# --------------------------------------------------------------------------------------------------------
# Spike-count-matched decile contrast
# --------------------------------------------------------------------------------------------------------

def test_matched_decile_contrast_discards_trials_outside_common_support():
    # 20 trials. deviation is rank order 0..19. spike_count for the bottom decile (idx 0,1) is huge
    # (10000, 10001) -- outside the top decile's own spike-count range (idx 18,19 have spike_count 0,1) --
    # so there is NO overlap and the matched contrast must report no_common_support, not a fabricated pair.
    deviation = np.arange(20, dtype=float)
    worse = np.arange(20, dtype=float)
    spike_count = np.arange(20, dtype=float)
    spike_count[0], spike_count[1] = 10000.0, 10001.0  # bottom-decile trials now have huge spike counts
    spike_count[18], spike_count[19] = 0.0, 1.0  # top-decile trials now have tiny spike counts
    result = _spike_count_matched_contrast(deviation, worse, spike_count, 0.10)
    assert result["status"] == "no_common_support"


def test_matched_decile_contrast_pairs_and_discards_correctly_with_partial_overlap():
    # 20 trials, deviation rank order 0..19. Bottom decile = idx {0,1}, top decile = idx {18,19}.
    # Give the bottom decile spike counts {5, 6} and the top decile spike counts {6, 50}: the common
    # support range is [max(5,6), min(6,50)] = [6, 6], so only spike_count==6 trials survive on each side
    # -- one from the bottom (idx 1) and one from the top (idx 18) -- a single matched pair, one discard
    # per side.
    deviation = np.arange(20, dtype=float)
    worse = np.arange(20, dtype=float).astype(float)
    spike_count = np.zeros(20)
    spike_count[0], spike_count[1] = 5.0, 6.0
    spike_count[18], spike_count[19] = 6.0, 50.0
    result = _spike_count_matched_contrast(deviation, worse, spike_count, 0.10)
    assert result["status"] == "too_few_matched_pairs"  # only 1 pair, floor is 2
    assert result["n_matched_pairs"] == 1


def test_matched_decile_contrast_computed_with_full_overlap():
    rng = np.random.default_rng(1)
    n = 60
    deviation = np.arange(n, dtype=float)
    worse = deviation.copy()
    spike_count = rng.uniform(100, 200, size=n)  # same range for every trial -> full common support
    result = _spike_count_matched_contrast(deviation, worse, spike_count, 0.10)
    assert result["status"] == "computed"
    assert result["n_matched_pairs"] >= 2
    assert result["n_top_discarded"] >= 0 and result["n_bottom_discarded"] >= 0


# --------------------------------------------------------------------------------------------------------
# Contiguous folds and cross-validated discrimination
# --------------------------------------------------------------------------------------------------------

def test_contiguous_folds_are_ordered_blocks_not_a_random_split():
    folds = _contiguous_folds(20, 5)
    assert list(folds) == sorted(folds)  # non-decreasing: each fold is a contiguous block in the given order
    assert len(set(folds)) == 5
    assert np.bincount(folds).tolist() == [4, 4, 4, 4, 4]


def test_cv_discrimination_binary_recovers_a_perfectly_informative_feature():
    rng = np.random.default_rng(2)
    n = 100
    y = rng.integers(0, 2, size=n)
    feature = y.astype(float) + rng.normal(scale=0.01, size=n)  # near-perfectly separates the two classes
    result = _cv_discrimination_binary(feature, y)
    assert result["status"] == "computed"
    assert result["auc"] > 0.9


def test_cv_discrimination_binary_is_near_chance_for_an_uninformative_feature():
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, size=n)
    feature = rng.normal(size=n)  # independent of y
    result = _cv_discrimination_binary(feature, y)
    assert result["status"] == "computed"
    assert 0.3 < result["auc"] < 0.7


def test_cv_discrimination_continuous_recovers_a_linear_relationship():
    rng = np.random.default_rng(4)
    n = 100
    feature = rng.normal(size=n)
    target = 2.0 * feature + rng.normal(scale=0.05, size=n)
    result = _cv_discrimination_continuous(feature, target)
    assert result["status"] == "computed"
    assert result["r"] > 0.9


def test_cv_discrimination_too_few_trials_is_named():
    result = _cv_discrimination_binary(np.zeros(4), np.array([0, 1, 0, 1]))
    assert result["status"] == "too_few_trials"


def test_cv_discrimination_continuous_between_fold_offset_inflates_pooled_but_not_per_fold():
    # feature and target both carry the same fold-level offset (a stand-in for a session-time drift shared
    # by every predictor) but are pure independent noise WITHIN a fold. Each fold's model is trained on the
    # OTHER four folds, where target = -feature holds exactly at the offset level, so it learns slope -1;
    # applied to the held-out fold that slope flips the fold's own negative offset back to a positive
    # prediction that tracks the fold's actual (positive) offset -- the across-fold-pooled statistic ends up
    # STRONGLY POSITIVELY correlated with target (a spurious relationship built entirely from between-fold
    # structure the model never earned within a fold). The per-fold statistic, computed only on each fold's
    # own held-out trials (which carry no cross-fold offset contrast to exploit), stays near zero.
    rng = np.random.default_rng(7)
    k, n_per_fold = 5, 20
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    fold_id = np.repeat(np.arange(k), n_per_fold)
    feature = -offsets[fold_id] + rng.normal(scale=0.05, size=k * n_per_fold)
    target = offsets[fold_id] + rng.normal(scale=0.05, size=k * n_per_fold)
    result = _cv_discrimination_continuous(feature, target, k=k)
    assert result["status"] == "computed"
    assert result["r"] > 0.9
    assert abs(result["r_per_fold_not_pooled_across_folds"]) < 0.3


# --------------------------------------------------------------------------------------------------------
# Spike-count-matched contrast reachability against its own detection floor
# --------------------------------------------------------------------------------------------------------

def _tested(mean_value: float, mdd: float) -> dict:
    return {"status": "tested", "mean_value": mean_value,
            "minimum_detectable_paired_difference_at_80pct_power": {"status": "computed", "mdd": mdd}}


def test_matched_contrast_reachability_below_detection_floor_is_not_a_failure():
    # mdd (0.0121) exceeds the unmatched effect (0.0072) it would have to detect -- exactly the multi-object
    # corpus's own numbers -- so a non-significant matched contrast here must read as underpowered, not failed.
    decile = _tested(0.007172248184959188, mdd=None)
    matched = _tested(0.003709428226044531, mdd=0.012079886791352513)
    result = _matched_contrast_reachability(decile, matched)
    assert result["status"] == "computed"
    assert result["matched_comparison_is_powered_to_detect_the_unmatched_effect_size"] is False
    assert "not evidence" in result["reading"]


def test_matched_contrast_reachability_powered_comparison_survives():
    decile = _tested(0.16682661757647319, mdd=None)
    matched = _tested(0.16020967992735347, mdd=0.12490045886576351)
    result = _matched_contrast_reachability(decile, matched)
    assert result["matched_comparison_is_powered_to_detect_the_unmatched_effect_size"] is True
    assert "speaks to whether" in result["reading"]


# --------------------------------------------------------------------------------------------------------
# Trial-count-weighted combiner
# --------------------------------------------------------------------------------------------------------

def test_trial_count_weighted_matches_hand_computed_average():
    # level A: n=10, value=1.0; level B: n=30, value=0.0 -> weighted mean = (10*1 + 30*0) / 40 = 0.25
    result = _trial_count_weighted([(10, 1.0), (30, 0.0)])
    assert result == pytest.approx(0.25)


def test_trial_count_weighted_skips_none_values():
    result = _trial_count_weighted([(10, None), (30, 0.5)])
    assert result == pytest.approx(0.5)


def test_trial_count_weighted_all_none_returns_none():
    assert _trial_count_weighted([(10, None), (30, None)]) is None


# --------------------------------------------------------------------------------------------------------
# Block A cross-validated-discrimination branch classifier
# --------------------------------------------------------------------------------------------------------

def _pooled(significant_positive: bool, status: str = "tested") -> dict:
    return {"status": status, "significant_positive": significant_positive}


def test_block_a_branch_deviation_above_chance_amplitude_not():
    branch = _block_a_cv_branch(_pooled(True), _pooled(False))
    assert branch == ("accuracy_predicting_component_carries_held_out_single_trial_information_and_the_"
                       "dominant_amplitude_does_not")


def test_block_a_branch_both_above_chance():
    assert _block_a_cv_branch(_pooled(True), _pooled(True)) == "both_observables_carry_held_out_information"


def test_block_a_branch_neither_above_chance():
    assert _block_a_cv_branch(_pooled(False), _pooled(False)) == \
        "no_observable_reaches_held_out_single_trial_discrimination_at_this_power"


def test_block_a_branch_amplitude_only():
    assert _block_a_cv_branch(_pooled(False), _pooled(True)) == \
        "dominant_amplitude_outpredicts_the_component_in_held_out_data"


def test_block_a_branch_not_tested_counts_as_not_above_chance():
    assert _block_a_cv_branch({"status": "not_computable"}, _pooled(True)) == \
        "dominant_amplitude_outpredicts_the_component_in_held_out_data"


# --------------------------------------------------------------------------------------------------------
# Block B localisation branch classifier
# --------------------------------------------------------------------------------------------------------

def _cell(status: str, r_values: list[float] | None = None) -> dict:
    return {"status": status, "per_session_worse_behaviour_r": r_values or []}


def test_localisation_all_void_is_unreachable():
    cells = {"a": _cell("void_due_to_gate_failure"), "b": _cell("void_no_reachable_sessions")}
    result = _localisation_branch(cells, "area")
    assert result["branch"] == "anatomical_localisation_unreachable_at_this_unit_count_per_area"


def test_localisation_none_computed_but_some_underpowered_is_named_distinctly_from_unreachable():
    cells = {"a": _cell("underpowered_by_construction"), "b": _cell("void_no_reachable_sessions")}
    result = _localisation_branch(cells, "area")
    assert result["branch"] == \
        "no_group_reaches_a_powered_behavioural_association_some_cells_reachable_but_underpowered"


def test_localisation_exactly_one_survivor_has_no_cross_group_comparison():
    cells = {"a": _cell("computed", [0.1, 0.2, 0.3, 0.15]), "b": _cell("void_no_reachable_sessions")}
    result = _localisation_branch(cells, "animal")
    assert result["branch"] == "only_one_group_survives_no_cross_group_comparison_possible"


def test_localisation_two_survivors_indistinguishable_groups_is_not_localised():
    rng = np.random.default_rng(5)
    shared = rng.normal(loc=0.1, scale=0.02, size=6).tolist()
    cells = {"a": _cell("computed", shared[:4]), "b": _cell("computed", shared)}
    result = _localisation_branch(cells, "area")
    assert result["branch"] == "accuracy_predicting_component_is_not_localised_to_one_recorded_area"
    assert result["surviving_groups"] == ["a", "b"]
    mdd = result["pairwise_tests"]["a_vs_b"]["minimum_detectable_paired_difference_at_80pct_power"]
    assert mdd["status"] == "computed"
    assert mdd["mdd"] > 0


def test_localisation_two_survivors_clearly_separated_groups_is_stronger_in_one():
    cells = {
        "a": _cell("computed", [0.5, 0.55, 0.52, 0.48, 0.53, 0.51]),
        "b": _cell("computed", [-0.5, -0.48, -0.52, -0.55, -0.51, -0.49]),
    }
    result = _localisation_branch(cells, "area")
    assert result["branch"] == "accuracy_predicting_component_is_stronger_in_one_area"


# --------------------------------------------------------------------------------------------------------
# Direct paired predictor-vs-predictor tests (the ordering an above/below-chance branch alone cannot show)
# --------------------------------------------------------------------------------------------------------

def _fake_session_records(key: str, deviation: list[float], amplitude: list[float],
                           spike_count: list[float]) -> list[dict]:
    return [{"combined": {key: {"deviation": d, "amplitude": a, "spike_count": s}}}
            for d, a, s in zip(deviation, amplitude, spike_count)]


def test_pairwise_predictor_tests_recovers_a_known_paired_offset_with_bh_correction():
    # 11 sessions (matches the single-item corpus's own n): amplitude is a fixed +0.15 above deviation in
    # every session (small session-to-session noise on top), spike_count is independent noise centred at
    # the same level as amplitude -- so deviation-vs-amplitude should come out a clear, significant paired
    # difference and amplitude-vs-spike_count should not.
    rng = np.random.default_rng(11)
    n = 11
    deviation = rng.normal(0.0, 0.02, size=n)
    amplitude = deviation + 0.15 + rng.normal(0.0, 0.01, size=n)
    spike_count = rng.normal(0.15, 0.05, size=n)
    key = "cross_validated_discrimination_centred_on_chance"
    records = _fake_session_records(key, deviation.tolist(), amplitude.tolist(), spike_count.tolist())
    cells = _pairwise_predictor_tests(records, key)
    assert set(cells) == {"deviation_minus_amplitude", "deviation_minus_spike_count", "amplitude_minus_spike_count"}
    dev_amp = cells["deviation_minus_amplitude"]
    assert dev_amp["status"] == "tested"
    assert dev_amp["mean_value"] == pytest.approx(-0.15, abs=0.03)
    assert dev_amp["significant"] is True
    # every tested cell gets a BH q-value alongside its raw p-value, and the reachability object is only
    # populated (beyond "not_applicable") for cells that did NOT reach significance.
    assert "benjamini_hochberg_q_value" in dev_amp
    assert dev_amp["benjamini_hochberg_q_value"] >= dev_amp["two_sided_p_value"]
    assert dev_amp["reachability"]["status"] == "not_applicable"


def test_pairwise_predictor_tests_pairs_only_sessions_where_both_predictors_are_defined():
    # one session has no amplitude value (None) -- it must be dropped from deviation-vs-amplitude but kept
    # for deviation-vs-spike_count, since that pair does not need amplitude.
    key = "cross_validated_discrimination_per_fold_not_pooled_across_folds"
    records = [
        {"combined": {key: {"deviation": 0.1, "amplitude": None, "spike_count": 0.05}}},
        {"combined": {key: {"deviation": 0.12, "amplitude": 0.2, "spike_count": 0.06}}},
        {"combined": {key: {"deviation": 0.09, "amplitude": 0.18, "spike_count": 0.04}}},
        {"combined": {key: {"deviation": 0.11, "amplitude": 0.21, "spike_count": 0.05}}},
    ]
    cells = _pairwise_predictor_tests(records, key)
    assert cells["deviation_minus_amplitude"]["n_sessions"] == 3
    assert cells["deviation_minus_spike_count"]["n_sessions"] == 4


# --------------------------------------------------------------------------------------------------------
# Pairwise-cell reachability against its own detection floor
# --------------------------------------------------------------------------------------------------------

def test_pairwise_cell_reachability_below_detection_floor():
    # matches the single-item corpus's own within-block deviation-vs-amplitude cell: not significant, and
    # its mdd (0.088459) exceeds the observed difference it would have to detect (0.044311).
    cell = _tested(-0.044311, mdd=0.088459)
    cell["significant"] = False
    result = _pairwise_cell_reachability(cell)
    assert result["status"] == "computed"
    assert result["label"] == "below_its_own_detection_floor"


def test_pairwise_cell_reachability_powered_null():
    cell = _tested(-0.02, mdd=0.01)
    cell["significant"] = False
    result = _pairwise_cell_reachability(cell)
    assert result["label"] == "powered_null"


def test_pairwise_cell_reachability_not_applicable_when_significant():
    cell = _tested(-0.15, mdd=0.05)
    cell["significant"] = True
    assert _pairwise_cell_reachability(cell)["status"] == "not_applicable"


# --------------------------------------------------------------------------------------------------------
# Pairwise comparison note: ordering branches name a pair, non-ordering branches name none
# --------------------------------------------------------------------------------------------------------

def test_pairwise_comparison_note_names_the_pair_for_an_ordering_branch():
    across = {"deviation_minus_amplitude": {"significant": True, "mean_value": -0.1, "two_sided_p_value": 0.004}}
    within = {"deviation_minus_amplitude": {"significant": False, "mean_value": -0.04, "two_sided_p_value": 0.19}}
    note = _pairwise_comparison_note("dominant_amplitude_outpredicts_the_component_in_held_out_data", across, within)
    assert "deviation" in note and "amplitude" in note
    assert "established by a direct paired test under the across-block version" in note
    assert "not established" in note  # within-block clause


def test_pairwise_comparison_note_states_no_ordering_claim_for_a_non_ordering_branch():
    note = _pairwise_comparison_note("both_observables_carry_held_out_information", {}, {})
    assert "makes no ordering claim" in note
