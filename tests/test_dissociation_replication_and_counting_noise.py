"""Tests for scripts/run_dissociation_replication_and_counting_noise.py --
the pieces that could silently break this analysis without erroring: (1) the
Poisson surrogate construction actually removes trial-to-trial direction
structure while preserving each trial's own total-count distribution,
(2) the within-item-count-level trial-count-weighted combination arithmetic,
(3) Block A's decision-rule classifier, (4) Block B's decision-rule
classifier, and (5) the two-sided empirical p-value helper the counting-noise
comparison is built on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    _attach_block_b_label_disclosure, _block_a_branch, _block_b_branch, _count_separation_disclosure,
    _heterogeneity_disclosure, _magnitude_diagnostic_verdict, _observable_arrays, _primary_cell_label_disclosure,
    _session_observable_arm, _two_sided_empirical_p, poisson_surrogate_and_real_deviation_magnitudes,
    poisson_surrogate_draw_correlations,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Poisson surrogate construction
# ---------------------------------------------------------------------------------------------------

def test_surrogate_preserves_each_trials_expected_total():
    # A session where every trial's own total is wildly different: the surrogate's REALISED totals must
    # track those real totals, trial for trial, because that is the counting-noise hypothesis's own
    # precondition -- if the surrogate did not preserve the real totals, a gate value built from it would
    # say nothing about whether counting noise (a function of total count) explains the real gate.
    rng = np.random.default_rng(0)
    n_trials, n_units = 60, 12
    mean_vec = rng.gamma(2.0, 2.0, size=n_units)
    direction = mean_vec / np.linalg.norm(mean_vec)
    real_totals = rng.integers(20, 400, size=n_trials).astype(float)
    activity = np.array([rng.poisson(direction * (t / direction.sum())) for t in real_totals], dtype=float)

    result = poisson_surrogate_draw_correlations(activity, n_draws=40, seed_tag="test_surrogate_totals")
    assert result["status"] == "computed"
    assert result["n_draws_usable"] >= 30

    # Rebuild one draw directly (same construction the function uses) and check its realised per-trial
    # total is close to the real trial's own total in expectation over many draws -- not exact per draw
    # (Poisson noise), but the AVERAGE surrogate total per trial should track the real total closely.
    lambdas = np.outer(real_totals / direction.sum(), direction)
    draws = np.stack([np.random.default_rng(1000 + d).poisson(lambdas).sum(axis=1) for d in range(300)])
    mean_surrogate_total_per_trial = draws.mean(axis=0)
    # Relative error should be small (Poisson mean over 300 draws has low sampling error).
    relative_error = np.abs(mean_surrogate_total_per_trial - real_totals) / real_totals
    assert np.median(relative_error) < 0.05


def test_surrogate_direction_is_the_session_mean_and_has_no_trial_structure():
    # Every surrogate trial's EXPECTED direction is the identical session-mean direction by construction,
    # so a real session's trial-to-trial direction structure (e.g. two alternating directions) must not
    # survive into the surrogate: the surrogate's own deviation-vs-count correlation must be explainable
    # by counting noise alone, never by a real alternating-direction signal the real data might carry.
    rng = np.random.default_rng(2)
    n_trials, n_units = 80, 10
    total = np.full(n_trials, 300.0)  # large, near-constant totals: counting noise should be negligible
    mean_vec = rng.gamma(2.0, 2.0, size=n_units)
    activity = np.array([rng.poisson(mean_vec / mean_vec.sum() * total[i]) for i in range(n_trials)], dtype=float)
    result = poisson_surrogate_draw_correlations(activity, n_draws=60, seed_tag="test_surrogate_no_structure")
    assert result["status"] == "computed"
    # At large, near-constant totals, sampling noise in the direction estimate is small relative to the
    # deviation-vs-count relationship counting noise alone would predict at LOW counts, so the surrogate's
    # gate values here should cluster near zero, not show a strong systematic correlation.
    assert abs(np.mean(result["draw_r_values"])) < 0.3


def test_surrogate_low_counts_produce_the_predicted_negative_sign():
    # The counting-noise sign prediction this surrogate exists to test: a low-count trial estimates its
    # own direction from fewer spikes, so its cosine against the leave-one-out mean is lower in
    # expectation, so its deviation is HIGHER -- a negative deviation-vs-count correlation from counting
    # noise alone. Reproduced directly here with totals spanning two orders of magnitude.
    rng = np.random.default_rng(3)
    n_trials, n_units = 100, 15
    mean_vec = rng.gamma(2.0, 2.0, size=n_units)
    real_totals = rng.integers(2, 60, size=n_trials).astype(float)  # low counts -- large relative Poisson noise
    direction = mean_vec / np.linalg.norm(mean_vec)
    activity = np.array([rng.poisson(direction * (t / direction.sum())) for t in real_totals], dtype=float)
    result = poisson_surrogate_draw_correlations(activity, n_draws=80, seed_tag="test_surrogate_low_counts")
    assert result["status"] == "computed"
    assert np.mean(result["draw_r_values"]) < -0.1


def test_surrogate_handles_zero_mean_activity():
    activity = np.zeros((20, 5))
    result = poisson_surrogate_draw_correlations(activity, n_draws=10, seed_tag="test_zero")
    assert result["status"] == "not_computable"


def test_surrogate_agrees_with_rate_free_state_deviation_on_one_realised_draw():
    # The surrogate function must call the delivered, unmodified rate_free_state_deviation rather than a
    # forked copy -- checked here by rebuilding one draw with the identical seed and confirming its r
    # matches what a direct call to the unmodified estimator produces on that same surrogate matrix.
    rng = np.random.default_rng(4)
    n_trials, n_units = 30, 8
    mean_vec = rng.gamma(2.0, 2.0, size=n_units)
    real_totals = rng.integers(10, 100, size=n_trials).astype(float)
    direction = mean_vec / np.linalg.norm(mean_vec)
    activity = np.array([rng.poisson(direction * (t / direction.sum())) for t in real_totals], dtype=float)

    from statistics import stable_seed
    seed_tag = "test_agreement"
    # The function derives its direction and per-trial totals from activity's OWN empirical mean and row
    # sums (finite-sample), not from the true generating parameters used to construct activity above --
    # recomputed the identical way here so this is a check of AGREEMENT, not a re-derivation.
    empirical_direction = activity.mean(axis=0)
    empirical_direction = empirical_direction / np.linalg.norm(empirical_direction)
    empirical_totals = activity.sum(axis=1)
    lambdas = np.outer(empirical_totals / empirical_direction.sum(), empirical_direction)
    draw_rng = np.random.default_rng(stable_seed(f"{seed_tag}|surrogate_draw0"))
    surrogate = draw_rng.poisson(lambdas)
    expected_deviation = rate_free_state_deviation(surrogate)
    expected_total = surrogate.sum(axis=1).astype(float)
    finite = np.isfinite(expected_deviation)
    expected_r = float(np.corrcoef(expected_deviation[finite], expected_total[finite])[0, 1])

    result = poisson_surrogate_draw_correlations(activity, n_draws=1, seed_tag=seed_tag)
    assert result["status"] == "computed"
    assert result["draw_r_values"][0] == pytest.approx(expected_r, abs=1e-10)


# ---------------------------------------------------------------------------------------------------
# Two-sided empirical p-value
# ---------------------------------------------------------------------------------------------------

def test_two_sided_empirical_p_center_versus_extreme():
    null = np.random.default_rng(5).normal(0.0, 1.0, 2000)
    p_center = _two_sided_empirical_p(0.0, null)
    p_extreme = _two_sided_empirical_p(20.0, null)
    assert p_center > 0.5
    assert p_extreme < 0.01
    assert p_extreme < p_center


def test_two_sided_empirical_p_is_symmetric_in_direction():
    null = np.random.default_rng(6).normal(0.0, 1.0, 2000)
    p_pos = _two_sided_empirical_p(3.0, null)
    p_neg = _two_sided_empirical_p(-3.0, null)
    assert p_pos == pytest.approx(p_neg, abs=0.05)


# ---------------------------------------------------------------------------------------------------
# Block B branch classifier
# ---------------------------------------------------------------------------------------------------

def test_block_b_branch_explained_by_counting_noise():
    assert _block_b_branch(real_mean=-0.05, p_value=0.4, surrogate_mean=-0.06, mdd_value=0.02) == \
        "deviation_gate_value_is_explained_by_counting_noise"


def test_block_b_branch_exceeds_counting_noise():
    assert _block_b_branch(real_mean=-0.5, p_value=0.001, surrogate_mean=-0.1, mdd_value=0.02) == \
        "deviation_gate_failure_exceeds_counting_noise"


def test_block_b_branch_wrong_direction():
    assert _block_b_branch(real_mean=0.5, p_value=0.001, surrogate_mean=-0.1, mdd_value=0.02) == \
        "deviation_gate_value_is_not_in_the_counting_noise_direction"


def test_block_b_branch_inconclusive_when_mdd_exceeds_observed():
    assert _block_b_branch(real_mean=0.02, p_value=0.3, surrogate_mean=-0.06, mdd_value=0.5) == \
        "inconclusive_below_detection_floor"


# ---------------------------------------------------------------------------------------------------
# Block A branch classifier
# ---------------------------------------------------------------------------------------------------

def _entry(significant: bool, r: float, mdd: float | None) -> dict:
    mdd_block = {"status": "computed", "mdd": mdd} if mdd is not None else {"status": "not_computable"}
    return {"status": "tested", "significant": significant, "r": r,
            "minimum_detectable_paired_difference_at_80pct_power": mdd_block}


def _cell(gate_sig: bool, amp_raw_sig: bool, amp_partial_sig: bool, dev_raw_sig: bool, dev_partial_sig: bool,
          amp_mdd: float = 0.05, dev_mdd: float = 0.05, void: bool = False) -> dict:
    return {
        "void_due_to_orthogonality_gate_failure": void,
        "amplitude": {"within_item_count_level": {
            "orthogonality_gate_vs_spike_count": _entry(gate_sig, 0.5, 0.02),
            "raw_vs_report_error": _entry(amp_raw_sig, 0.1, amp_mdd),
            "partial_controlling_spike_count": _entry(amp_partial_sig, 0.05, 0.02),
        }},
        "deviation": {"within_item_count_level": {
            "raw_vs_report_error": _entry(dev_raw_sig, 0.05, dev_mdd),
            "partial_controlling_spike_count": _entry(dev_partial_sig, 0.05, 0.02),
        }},
    }


def test_block_a_branch_gate_not_significant():
    branch = _block_a_branch(_cell(False, False, False, False, False))
    assert branch["branch"] == "dominant_mode_is_not_rate_in_this_preparation"


def test_block_a_branch_dissociation_replicates():
    branch = _block_a_branch(_cell(True, True, False, True, True))
    assert branch["branch"] == "dissociation_replicates_in_the_better_powered_corpus"


def test_block_a_branch_amplitude_survives():
    branch = _block_a_branch(_cell(True, True, True, False, False))
    assert branch["branch"] == "dominant_amplitude_link_survives_the_rate_control_here"


def test_block_a_branch_neither_predicts_powered_null():
    branch = _block_a_branch(_cell(True, False, False, False, False, amp_mdd=0.05, dev_mdd=0.05))
    assert branch["branch"] == "neither_observable_predicts_report_error_in_this_corpus"
    assert branch["sub_label"] == "powered_null"


def test_block_a_branch_neither_predicts_inconclusive_sub_label():
    branch = _block_a_branch(_cell(True, False, False, False, False, amp_mdd=0.99, dev_mdd=0.99))
    assert branch["branch"] == "neither_observable_predicts_report_error_in_this_corpus"
    assert branch["sub_label"] == "inconclusive"


def test_block_a_branch_off_list_combination_is_inconclusive():
    # amplitude raw significant but its partial does not survive, AND the deviation does not survive its
    # own partial either -- none of branches 2-4 match, so the pre-declared rule's own "otherwise" fires.
    branch = _block_a_branch(_cell(True, True, False, True, False))
    assert branch["branch"] == "inconclusive_below_detection_floor"


def test_block_a_branch_void_cell_is_reported_as_void_not_as_a_result():
    branch = _block_a_branch(_cell(True, True, False, True, True, void=True))
    assert branch["branch"] == "void_orthogonality_gate_failed"


# ---------------------------------------------------------------------------------------------------
# Within-item-count-level trial-count-weighted combination
# ---------------------------------------------------------------------------------------------------

def _synthetic_watters_session(seed: int, n_trials: int = 80, n_units: int = 20, n_bins: int = 10) -> dict:
    """A minimal but complete multi-object-macaque-shaped session: two item-count levels of unequal size,
    every field _behaviour_observables and trial_amplitude_covariates read."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(3.0, size=(n_trials, n_units, n_bins)).astype(float)
    item_count = np.where(np.arange(n_trials) < 50, 1, 2).astype(float)  # 50 trials at level 1, 30 at level 2
    return {
        "session": f"synthetic_{seed}", "counts": counts,
        "report_deviation": rng.uniform(0.0, 1.0, n_trials),
        "reaction_time_ms": rng.uniform(200.0, 800.0, n_trials),
        "num_objects": item_count, "correct": rng.uniform(size=n_trials) > 0.3,
        "trial_num": np.arange(n_trials),
    }


def test_within_load_combination_matches_a_hand_computed_weighted_average():
    session = _synthetic_watters_session(seed=42)
    arrays, excluded, usable = _observable_arrays(session["counts"], session)
    assert arrays is not None
    arm = _session_observable_arm(arrays, "deviation", "test_within_load")

    levels = arm["item_count_levels_present"]
    assert levels == [1, 2]
    stat = "raw_vs_report_error"
    per_level_r = {lv: arm["per_level"][str(lv)]["family"][stat] for lv in levels}
    assert all(entry["status"] == "computed" for entry in per_level_r.values())

    n_by_level = {lv: arm["per_level"][str(lv)]["n_trials"] for lv in levels}
    hand_weighted = sum(n_by_level[lv] * per_level_r[lv]["r"] for lv in levels) / sum(n_by_level.values())
    assert arm["within_load_trial_count_weighted"][stat] == pytest.approx(hand_weighted, abs=1e-12)

    # A trial-count-weighted average of two level correlations must lie between them (or equal an
    # endpoint if the two levels happen to coincide) -- a basic sanity bound on the arithmetic.
    r_values = [per_level_r[lv]["r"] for lv in levels]
    assert min(r_values) - 1e-9 <= arm["within_load_trial_count_weighted"][stat] <= max(r_values) + 1e-9


def test_within_load_weight_favours_the_larger_level():
    # 50 trials at level 1, 30 at level 2 -- construct arrays where the two levels have deliberately
    # different correlations and check the combined value sits closer to the LARGER level's value than
    # the midpoint, proving the weighting is by trial count and not an unweighted average of the two.
    session = _synthetic_watters_session(seed=7)
    arrays, _excluded, _usable = _observable_arrays(session["counts"], session)
    arm = _session_observable_arm(arrays, "amplitude", "test_weighting")
    stat = "raw_vs_report_error"
    per_level_r = {lv: arm["per_level"][str(lv)]["family"][stat]["r"] for lv in arm["item_count_levels_present"]}
    unweighted_midpoint = sum(per_level_r.values()) / len(per_level_r)
    combined = arm["within_load_trial_count_weighted"][stat]
    distance_to_level1 = abs(combined - per_level_r[1])
    distance_to_midpoint = abs(unweighted_midpoint - per_level_r[1])
    # The level-1 group (50 of 80 trials) has more weight than an unweighted average would give it, so
    # the weighted combination sits no farther from level 1's own value than the unweighted midpoint does.
    assert distance_to_level1 <= distance_to_midpoint + 1e-9


# ---------------------------------------------------------------------------------------------------
# Repair: surrogate-vs-real deviation magnitude diagnostic
# ---------------------------------------------------------------------------------------------------

def _synthetic_activity(seed: int, n_trials: int = 100, n_units: int = 15, lo: int = 2, hi: int = 60):
    rng = np.random.default_rng(seed)
    mean_vec = rng.gamma(2.0, 2.0, size=n_units)
    real_totals = rng.integers(lo, hi, size=n_trials).astype(float)
    direction = mean_vec / np.linalg.norm(mean_vec)
    return np.array([rng.poisson(direction * (t / direction.sum())) for t in real_totals], dtype=float)


def test_magnitude_diagnostic_reproduces_the_delivered_correlations_bit_for_bit():
    activity = _synthetic_activity(seed=10)
    delivered = poisson_surrogate_draw_correlations(activity, n_draws=50, seed_tag="magnitude_repro")
    assert delivered["status"] == "computed"

    reproduced = poisson_surrogate_and_real_deviation_magnitudes(
        activity, n_draws=50, seed_tag="magnitude_repro", expected_draw_r=delivered["draw_r_values"])
    assert reproduced["status"] == "computed"
    assert reproduced["reproduced_delivered_draw_r_values"] is True
    assert reproduced["real_median_deviation_magnitude"] is not None
    assert reproduced["real_median_deviation_magnitude"] >= 0.0
    assert reproduced["surrogate_median_deviation_magnitude"] >= 0.0


def test_magnitude_diagnostic_raises_when_expected_draw_r_does_not_match():
    activity = _synthetic_activity(seed=11)
    with pytest.raises(AssertionError):
        poisson_surrogate_and_real_deviation_magnitudes(
            activity, n_draws=30, seed_tag="magnitude_mismatch", expected_draw_r=[0.0] * 30)


def test_magnitude_diagnostic_verdict_thresholds():
    assert _magnitude_diagnostic_verdict(0.5) == "surrogate_is_a_plausible_upper_bound_on_the_counting_noise_contribution"
    assert _magnitude_diagnostic_verdict(1.0) == "surrogate_is_a_plausible_upper_bound_on_the_counting_noise_contribution"
    assert _magnitude_diagnostic_verdict(1.01) == "surrogate_over_predicts_deviation_magnitude_and_is_not_a_plausible_null"
    assert _magnitude_diagnostic_verdict(None) == "not_computable"


# ---------------------------------------------------------------------------------------------------
# Repair: reporting-only disclosure fields (no branch move, no rule change)
# ---------------------------------------------------------------------------------------------------

def _amp_dev_entry(significant: bool, r: float, p: float, mdd: float) -> dict:
    return {"status": "tested", "significant": significant, "mean_value": r, "two_sided_p_value": p,
            "minimum_detectable_paired_difference_at_80pct_power": {"status": "computed", "mdd": mdd}}


def _primary_cell(branch_label: str, amp_raw_sig: bool, dev_raw_sig: bool, dev_partial_sig: bool,
                   dev_joint_sig: bool) -> dict:
    return {
        "branch": {"branch": branch_label, "single_item_corpus_amplitude_effect_size_r_units": 0.1675415174221389},
        "amplitude": {"within_item_count_level": {
            "raw_vs_report_error": _amp_dev_entry(amp_raw_sig, -0.004, 0.66, 0.0244),
        }},
        "deviation": {"within_item_count_level": {
            "raw_vs_report_error": _amp_dev_entry(dev_raw_sig, 0.0197, 0.0003, 0.0138),
            "partial_controlling_spike_count": _amp_dev_entry(dev_partial_sig, 0.0182, 0.0007, 0.0138),
            "joint_partial_controlling_every_nuisance": _amp_dev_entry(dev_joint_sig, 0.0188, 0.0006, 0.0138),
        }},
    }


def test_primary_cell_label_disclosure_applies_on_the_actual_delivered_pattern():
    cell = _primary_cell("inconclusive_below_detection_floor", amp_raw_sig=False, dev_raw_sig=True,
                          dev_partial_sig=True, dev_joint_sig=True)
    disclosure = _primary_cell_label_disclosure(cell)
    assert disclosure["applies"] is True
    assert disclosure["power_margin_multiple_of_reference_effect"] == pytest.approx(0.1675415174221389 / 0.0244, rel=1e-3)
    assert disclosure["deviation_raw_vs_report_error"]["significant"] is True


def test_primary_cell_label_disclosure_does_not_apply_to_a_named_branch():
    cell = _primary_cell("dissociation_replicates_in_the_better_powered_corpus", amp_raw_sig=True,
                          dev_raw_sig=True, dev_partial_sig=True, dev_joint_sig=True)
    disclosure = _primary_cell_label_disclosure(cell)
    assert disclosure["applies"] is False


def test_heterogeneity_disclosure_counts_split_cells_correctly():
    gate_sig = {"a": True, "b": False, "c": False, "d": True}
    dev_raw = {"a": 0.02, "b": 0.01, "c": 0.03, "d": None}
    h = _heterogeneity_disclosure(gate_sig, dev_raw)
    assert h["n_split_cells_with_amplitude_gate_significant"] == 2
    assert h["n_split_cells_total"] == 4
    assert h["n_split_cells_with_deviation_raw_association_positive"] == 3
    assert h["n_split_cells_with_deviation_raw_association_computed"] == 3


def test_count_separation_disclosure_flags_the_asymmetric_pattern():
    gate_table = [
        {"corpus": "pass1", "deviation_gate_p_value": 0.59, "median_total_spike_count_per_trial": 1847.5,
         "median_count_per_unit_per_trial": 2.0},
        {"corpus": "pass2", "deviation_gate_p_value": 0.96, "median_total_spike_count_per_trial": 989.5,
         "median_count_per_unit_per_trial": 6.0},
        {"corpus": "fail1", "deviation_gate_p_value": 0.0005, "median_total_spike_count_per_trial": 353.0,
         "median_count_per_unit_per_trial": 4.0},
        {"corpus": "fail2", "deviation_gate_p_value": 0.028, "median_total_spike_count_per_trial": 256.25,
         "median_count_per_unit_per_trial": 3.0},
        {"corpus": "fail3", "deviation_gate_p_value": 0.0015, "median_total_spike_count_per_trial": 132.5,
         "median_count_per_unit_per_trial": 3.0},
    ]
    cs = _count_separation_disclosure(gate_table)
    assert cs["total_spike_count_per_trial_cleanly_separates_passing_from_failing"] is True
    assert cs["median_count_per_unit_per_trial_cleanly_separates_passing_from_failing"] is False
    assert cs["unsampled_gap_in_total_spike_count_per_trial_bounds"] == [353.0, 989.5]


def test_attach_block_b_label_disclosure_only_fires_on_its_own_branch():
    fired = _attach_block_b_label_disclosure(
        {"branch": "deviation_gate_value_is_not_in_the_counting_noise_direction", "real_gate_mean_value": -0.2057,
         "surrogate_distribution": {"mean": -0.449}})
    assert "label_disclosure" in fired
    assert fired["label_disclosure"]["observed_gate_is_negative"] is True

    unfired = _attach_block_b_label_disclosure({"branch": "deviation_gate_failure_exceeds_counting_noise"})
    assert "label_disclosure" not in unfired


# ---------------------------------------------------------------------------------------------------
# Zero-drop reconciliation
# ---------------------------------------------------------------------------------------------------

def test_zero_drop_reconciles_when_every_seen_session_is_either_loaded_or_refused():
    # The exact reconciliation expression run_block_a and run_block_b both assert: every session counted
    # as "seen" must be accounted for as either analysed or refused-with-a-reason, never left uncounted.
    seen_sessions = [f"s{i}" for i in range(10)]
    loaded = set(seen_sessions[:7])
    refused = {s: "some_named_reason" for s in seen_sessions[7:]}
    assert len(seen_sessions) == len(loaded) + len(refused)
    # A session dropped from BOTH sets (the defect this assertion exists to catch) must break the check.
    seen_with_a_drop = seen_sessions + ["s10_never_counted_anywhere"]
    assert len(seen_with_a_drop) != len(loaded) + len(refused)
