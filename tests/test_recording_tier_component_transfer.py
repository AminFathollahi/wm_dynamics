"""Tests for scripts/run_recording_tier_component_transfer.py.

Covers the parts of this module with no counterpart elsewhere in the project and no NWB/EDF/mat file
dependency: the trial-alignment assertion two independent reads of the same NWB file must pass before any
cross-tier statistic is trusted, the magnitude-matched rotation null's sensitivity to real vs. absent
direction structure, the bias-only control's ability to both catch a pure session/patient-level offset and
leave a genuine trial-level effect alone, the band-selection and mains-notch logic against measured
sample rates, and the refusal paths (too few trials, too few patients, no channels in a region, structurally
untestable behaviour for the tier with no accuracy label)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_recording_tier_component_transfer import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, MIN_PATIENTS_FOR_TEST, _bias_only_values, _classify_block_c,
    _patient_clustered_test, block_a_tier, block_b_tier, block_c_pair_patient_level, block_c_pair_trial_wise,
    classify_depth_channels, highest_available_band, needs_mains_notch, rate_free_state_deviation,
    rotation_null_variance_test, trial_tables_agree,
)


# ---------------------------------------------------------------------------------------------------
# Trial-alignment assertion (the bridge Block C depends on)
# ---------------------------------------------------------------------------------------------------

def _table(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "artifact": np.zeros(n, dtype=bool),
        "correct": rng.integers(0, 2, size=n).astype(bool),
        "set_size": rng.choice([4, 6, 8], size=n),
        "start_time": np.arange(n, dtype=float) * 10.0,
    }


def test_trial_tables_agree_passes_on_identical_reads():
    table = _table()
    ieeg = {"set_sizes": table["set_size"].copy(), "correct": table["correct"].copy()}
    eeg = {"set_sizes": table["set_size"].copy(), "correct": table["correct"].copy()}
    ok, reason = trial_tables_agree(table, ieeg, eeg)
    assert ok is True
    assert reason is None


def test_trial_tables_agree_catches_a_trial_count_mismatch():
    table = _table(n=10)
    ieeg = {"set_sizes": table["set_size"].copy(), "correct": table["correct"].copy()}
    eeg = {"set_sizes": table["set_size"][:9], "correct": table["correct"][:9]}  # one fewer trial
    ok, reason = trial_tables_agree(table, ieeg, eeg)
    assert ok is False
    assert reason == "trial_count_mismatch_between_signals_in_same_file"


def test_trial_tables_agree_catches_a_set_size_mismatch_even_with_equal_length():
    table = _table(n=10)
    ieeg = {"set_sizes": table["set_size"].copy(), "correct": table["correct"].copy()}
    scrambled = table["set_size"].copy()
    scrambled[0] = 4 if scrambled[0] != 4 else 6  # same length, one entry disagrees
    eeg = {"set_sizes": scrambled, "correct": table["correct"].copy()}
    ok, reason = trial_tables_agree(table, ieeg, eeg)
    assert ok is False
    assert reason == "set_size_mismatch_between_signals_in_same_file"


def test_trial_tables_agree_catches_an_accuracy_mismatch():
    table = _table(n=10)
    ieeg = {"set_sizes": table["set_size"].copy(), "correct": table["correct"].copy()}
    flipped = table["correct"].copy()
    flipped[0] = ~flipped[0]
    eeg = {"set_sizes": table["set_size"].copy(), "correct": flipped}
    ok, reason = trial_tables_agree(table, ieeg, eeg)
    assert ok is False
    assert reason == "accuracy_mismatch_between_signals_in_same_file"


# ---------------------------------------------------------------------------------------------------
# Magnitude-matched rotation null (Block A's existence test)
# ---------------------------------------------------------------------------------------------------

def test_rotation_null_flags_real_direction_structure_as_distinguishable():
    # Every trial's activity is a near-fixed direction plus tiny noise, at random magnitudes -- real
    # shared direction structure, magnitudes uninformative. The observed cosine-deviation variance should
    # be small and reliably distinguishable from the rotation null's (much larger, isotropic) variance.
    rng = np.random.default_rng(0)
    n_trials, n_features = 60, 12
    center = rng.normal(size=n_features)
    center /= np.linalg.norm(center)
    directions = center[None, :] + rng.normal(size=(n_trials, n_features)) * 0.05
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = rng.uniform(1.0, 50.0, size=(n_trials, 1))
    activity = directions * magnitudes

    result = rotation_null_variance_test(activity, n_draws=300, rng=np.random.default_rng(1))
    assert result["status"] == "computed"
    assert result["observed_variance"] < result["null_mean_variance"]
    assert result["p_value"] <= 0.05


def test_rotation_null_does_not_flag_activity_with_no_shared_direction():
    # Every trial's direction is already independent and uniformly random -- the observed data IS a draw
    # from the same generative process the null itself uses, so the test should not reliably reject.
    rng = np.random.default_rng(2)
    n_trials, n_features = 60, 12
    directions = rng.normal(size=(n_trials, n_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = rng.uniform(1.0, 50.0, size=(n_trials, 1))
    activity = directions * magnitudes

    result = rotation_null_variance_test(activity, n_draws=300, rng=np.random.default_rng(3))
    assert result["status"] == "computed"
    assert result["p_value"] > 0.01  # not a tight bound -- this is a negative result, not an exact one


def test_rotation_null_reuses_the_estimator_and_a_global_rotation_would_not_have_worked():
    # Sanity check on the estimator this null is built from: a global rotation (the same orthogonal
    # transform applied to every trial) leaves every pairwise cosine unchanged, which is exactly why the
    # module's own docstring insists on an INDEPENDENT per-trial randomisation instead.
    rng = np.random.default_rng(4)
    activity = rng.normal(size=(20, 5)) + 3.0  # arbitrary, non-degenerate
    deviation_before = rate_free_state_deviation(activity)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    deviation_after = rate_free_state_deviation(activity @ q)
    assert np.allclose(deviation_before, deviation_after, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------------------------------
# Bias-only control
# ---------------------------------------------------------------------------------------------------

def test_bias_only_values_are_the_leave_one_out_mean_of_every_other_trial():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    out = _bias_only_values(values)
    # trial 0's bias-only value excludes itself: mean(2,3,4) = 3.0
    assert out[0] == pytest.approx(3.0)
    assert out[1] == pytest.approx((1 + 3 + 4) / 3)
    assert out[3] == pytest.approx((1 + 2 + 3) / 3)


def test_bias_only_values_skip_nan_trials_on_both_sides():
    values = np.array([1.0, np.nan, 3.0, 5.0])
    out = _bias_only_values(values)
    assert np.isnan(out[1])  # a trial with no defined value gets no bias-only value either
    assert out[0] == pytest.approx((3.0 + 5.0) / 2)  # excludes itself AND the NaN trial


def _synthetic_patient_sessions(n_patients, n_sessions_per_patient, trials_per_session_per_cell,
                                 primary_signal, session_block_offset_scale, seed):
    """Builds per_tier_sessions-shaped records with a controllable mixture of a genuine trial-level
    accuracy signal and a within-patient, ACROSS-SESSION offset in the component value (one session of a
    patient running systematically higher than another, with a correlated session-level accuracy shift) --
    the confound block_b_tier's trial-pooled-per-patient design is actually exposed to, since it pools a
    patient's OWN sessions together before computing one correlation per patient. A confound that varies
    only BETWEEN patients (not within one patient's own sessions) is already absorbed by the sign-flip
    test's own per-patient clustering and is not what the bias-only control exists to catch here."""
    rng = np.random.default_rng(seed)
    sessions = []
    for p in range(n_patients):
        patient = f"sub-{p:02d}"
        for s_idx in range(n_sessions_per_patient):
            session_offset = rng.normal() * session_block_offset_scale
            session_accuracy_rate = float(np.clip(0.5 + 0.35 * np.sign(session_offset), 0.05, 0.95))
            deviation, correct, set_size = [], [], []
            for level in (4, 6, 8):
                n = trials_per_session_per_cell
                y = (rng.uniform(size=n) < session_accuracy_rate).astype(float)
                x = session_offset + primary_signal * (y - 0.5) + rng.normal(size=n) * 0.5
                deviation.append(x); correct.append(y); set_size.append(np.full(n, level))
            sessions.append({
                "patient": patient, "session_key": f"{patient}_ses-{s_idx:02d}",
                "deviation": np.concatenate(deviation), "correct": np.concatenate(correct),
                "set_size": np.concatenate(set_size), "gate": {"status": "computed", "signed_effect": 0.0},
            })
    return sessions


def test_bias_only_control_fires_the_void_branch_for_a_within_patient_session_offset():
    # No trial-level signal at all (primary_signal=0) -- accuracy correlates with the component value only
    # because one of a patient's sessions runs at a different baseline than another. Pooling that patient's
    # trials mixes the two sessions together, so both the primary statistic and the bias-only (a
    # leave-one-out mean over the same mixed pool) pick up the same session-block structure, and the void
    # branch must fire.
    sessions = _synthetic_patient_sessions(n_patients=8, n_sessions_per_patient=3,
                                            trials_per_session_per_cell=20, primary_signal=0.0,
                                            session_block_offset_scale=3.0, seed=10)
    result = block_b_tier(sessions, MEANINGFUL_EFFECT_THRESHOLD_R_UNITS)
    assert result["branch"] == "behaviour_link_not_separable_from_a_session_level_offset"


def test_bias_only_control_does_not_void_a_genuine_trial_level_effect():
    # A strong genuine trial-to-trial effect (accuracy predicts the component value within a session,
    # independent of which session it came from) with NO session-block offset at all. The bias-only
    # statistic (a leave-one-out mean over many trials) collapses toward a near-constant value per patient
    # and should not reproduce this, so the tier's result must not be voided.
    sessions = _synthetic_patient_sessions(n_patients=8, n_sessions_per_patient=2,
                                            trials_per_session_per_cell=25, primary_signal=4.0,
                                            session_block_offset_scale=0.0, seed=11)
    result = block_b_tier(sessions, MEANINGFUL_EFFECT_THRESHOLD_R_UNITS)
    assert result["branch"] != "behaviour_link_not_separable_from_a_session_level_offset"
    assert result["primary_within_set_size_test"]["significant"] is True


def test_block_b_reports_the_trial_and_error_counts_the_estimator_needs_disclosed():
    sessions = _synthetic_patient_sessions(n_patients=6, n_sessions_per_patient=1,
                                            trials_per_session_per_cell=15, primary_signal=0.0,
                                            session_block_offset_scale=0.0, seed=12)
    result = block_b_tier(sessions, MEANINGFUL_EFFECT_THRESHOLD_R_UNITS)
    assert result["n_trials_entering_estimate"] == 6 * 15 * 3
    assert result["n_patients_total"] == 6
    assert len(result["per_session_error_distribution"]) == 6
    assert all("n_errors" in row and "n_trials" in row for row in result["per_session_error_distribution"])


# ---------------------------------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------------------------------

def test_patient_clustered_test_refuses_below_the_structural_floor():
    # Below MIN_PATIENTS_FOR_TEST the sign-flip test cannot structurally reach p<=0.05 -- this must be
    # caught before ever calling the test, not discovered as a coincidentally-non-significant result.
    values = {f"sub-{i:02d}": 0.9 for i in range(MIN_PATIENTS_FOR_TEST - 1)}
    result = _patient_clustered_test(values)
    assert result["status"] == "underpowered_by_construction"
    assert result["n_patients"] == MIN_PATIENTS_FOR_TEST - 1


def test_patient_clustered_test_runs_at_the_floor():
    rng = np.random.default_rng(5)
    values = {f"sub-{i:02d}": float(0.5 + rng.normal() * 0.05) for i in range(MIN_PATIENTS_FOR_TEST)}
    result = _patient_clustered_test(values)
    assert result["status"] == "tested"
    assert result["n_patients"] == MIN_PATIENTS_FOR_TEST


def test_block_a_tier_underpowered_below_the_patient_floor():
    sessions = [{"patient": f"sub-{i:02d}", "gate": {"status": "computed", "signed_effect": 0.1},
                 "deviation": np.array([0.1, 0.2, 0.3])} for i in range(2)]
    result = block_a_tier(sessions, reference_effect=None)
    assert result["branch"] == "underpowered_to_ask_at_this_tier"


def test_block_c_trial_wise_refuses_below_the_patient_floor():
    session_records = [
        {"patient": "sub-01", "session_key": "s1",
         "tiers": {"single_unit": np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.19, 0.21, 0.17,
                                             0.16, 0.24]),
                   "depth_mtl": np.array([0.11, 0.19, 0.16, 0.29, 0.24, 0.2, 0.23, 0.18, 0.2, 0.16,
                                          0.15, 0.25])}},
    ]
    result = block_c_pair_trial_wise(session_records, "single_unit", "depth_mtl", reference_effect=None)
    assert result["regime"] == "trial_wise"
    assert result["branch"] == "cross_tier_agreement_not_testable_at_matched_trials"
    assert result["n_patients_contributing"] < MIN_PATIENTS_FOR_TEST


def test_block_c_patient_level_below_floor_is_not_testable():
    result = block_c_pair_patient_level({"sub-01": 0.1, "sub-02": 0.2}, {"sub-01": 0.05, "sub-02": 0.09},
                                         reference_effect=None)
    assert result["regime"] == "patient_level_only"
    assert result["branch"] == "cross_tier_agreement_not_testable_at_matched_trials"


def test_block_c_patient_level_detects_real_agreement():
    rng = np.random.default_rng(6)
    n = 9
    a = {f"sub-{i:02d}": float(i) + rng.normal() * 0.05 for i in range(n)}
    b = {f"sub-{i:02d}": float(i) * 2.0 + rng.normal() * 0.05 for i in range(n)}  # monotone in a
    result = block_c_pair_patient_level(a, b, reference_effect=None)
    assert result["regime"] == "patient_level_only"
    assert result["branch"] == "the_two_tiers_track_the_same_per_trial_quantity"


def test_classify_block_c_uses_reference_effect_to_distinguish_null_from_not_testable():
    powered_null = {"status": "tested", "significant": False, "mdd": 0.05, "n_patients": 9}
    underpowered_null = {"status": "tested", "significant": False, "mdd": 0.5, "n_patients": 9}
    assert _classify_block_c(powered_null, reference_effect=0.14) == "no_cross_tier_agreement_above_the_reported_bound"
    assert _classify_block_c(underpowered_null, reference_effect=0.14) == "cross_tier_agreement_not_testable_at_matched_trials"
    assert _classify_block_c(underpowered_null, reference_effect=None) == "cross_tier_agreement_not_testable_at_matched_trials"


# ---------------------------------------------------------------------------------------------------
# Band selection and mains-notch logic (measured sample rates, not the docstring-nominal ones)
# ---------------------------------------------------------------------------------------------------

def test_highest_available_band_picks_hgp_for_depth_macro_contact_sample_rate():
    name, lo, hi = highest_available_band(1398.0)
    assert (name, lo, hi) == ("hgp", 70.0, 150.0)


def test_highest_available_band_falls_back_to_beta_for_the_measured_eeg_sample_rate():
    # 137.1 Hz is what this project's own loader measures for 000574's EEG signal (not the 140 Hz nominal
    # figure in its docstring) -- Nyquist ~68.5 Hz rules out both hgp (150) and gamma (70 essentially at
    # the ceiling), leaving beta (30) as the highest band that clears it.
    name, lo, hi = highest_available_band(137.1)
    assert (name, lo, hi) == ("beta", 13.0, 30.0)


def test_highest_available_band_picks_gamma_for_the_beamformed_derivative_sample_rate():
    name, lo, hi = highest_available_band(200.0)
    assert (name, lo, hi) == ("gamma", 30.0, 70.0)


def test_highest_available_band_returns_none_when_nothing_fits():
    assert highest_available_band(6.0) is None


def test_needs_mains_notch_true_for_hgp_and_gamma_false_for_beta():
    assert needs_mains_notch(70.0, 150.0) is True   # 100 Hz harmonic falls inside
    assert needs_mains_notch(30.0, 70.0) is True    # 50 Hz fundamental falls inside
    assert needs_mains_notch(13.0, 30.0) is False   # no mains harmonic falls inside


# ---------------------------------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------------------------------

def test_classify_depth_channels_separates_mtl_from_cortical_and_drops_unlabelled():
    locs = [
        "Hipp, Left Hippocampus rHipp, rostral hippocampus",
        "Amyg, Left Amygdala mAmyg, medial amygdala",
        "PHG, Right ParaHippocampal Gyrus",  # entorhinal_parahippocampal
        "MTG, Left Middle Temporal Gyrus aSTS, anterior superior temporal sulcus",
        "unspecific",
        "no_label_found",
    ]
    result = classify_depth_channels(locs)
    assert result["depth_mtl"] == [0, 1, 2]
    assert result["depth_cortical"] == [3]
    assert 4 not in result["depth_mtl"] and 4 not in result["depth_cortical"]
    assert 5 not in result["depth_mtl"] and 5 not in result["depth_cortical"]


# ---------------------------------------------------------------------------------------------------
# The delivered artifact itself: existence, internal consistency, the power bound on every fired null
# branch, and zero-drop bookkeeping. Everything above tests the module's functions in isolation; this
# section is the check that the actual results/recording_tier_component_transfer.json this run produced
# obeys its own decision rules.
# ---------------------------------------------------------------------------------------------------

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "results" / "recording_tier_component_transfer.json"

NULL_BRANCHES = {
    "component_is_not_distinguishable_from_a_magnitude_matched_rotation_null",
    "no_behaviour_link_at_this_tier_above_the_reported_bound",
    "no_cross_tier_agreement_above_the_reported_bound",
}
POSITIVE_BRANCHES = {
    "component_is_present_at_this_recording_tier",
    "component_predicts_accuracy_at_this_recording_tier",
    "the_two_tiers_track_the_same_per_trial_quantity",
}


def _load_artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


def _iter_branches(doc: dict):
    """Yields (label, branch, detectable_effect, reference_effect, significance_flag_or_None) for every
    tested cell across Blocks A, B and C -- the one place both the power check and the significance check
    below draw from, so the two regimes Block C mixes (trial-wise mdd vs. patient-level ci_half_width)
    are only reconciled once."""
    for tier, rec in doc["block_a"].items():
        p = rec.get("pooled_patient_test", {})
        yield f"block_a/{tier}", rec["branch"], p.get("mdd"), rec.get("reference_effect_used"), p.get("significant")
    for tier, rec in doc["block_b"].items():
        p = rec.get("primary_within_set_size_test", {})
        yield f"block_b/{tier}", rec["branch"], p.get("mdd"), rec.get("reference_effect_used"), p.get("significant")
    for pair, rec in doc["block_c"].items():
        if rec["regime"] == "trial_wise":
            p = rec.get("pooled_patient_test", {})
            yield f"block_c/{pair}", rec["branch"], p.get("mdd"), rec.get("reference_effect_used"), p.get("significant")
        else:
            yield (f"block_c/{pair}", rec["branch"], rec.get("ci_half_width"), rec.get("reference_effect_used"),
                   rec.get("excludes_zero"))


def test_artifact_exists_and_is_complete():
    assert ARTIFACT_PATH.exists()
    doc = _load_artifact()
    assert doc["analysis_id"] == "recording_tier_component_transfer"
    assert doc["status"] == "complete"


def test_zero_drop_bookkeeping_adds_up():
    doc = _load_artifact()
    scope = doc["scope"]
    assert scope["n_000574_sessions_computed"] + scope["n_000574_sessions_refused"] == scope["n_000574_sessions_seen"]
    assert (scope["n_beamformed_patients_computed"] + scope["n_beamformed_patients_refused"]
            == scope["n_beamformed_patients_seen"])
    n_000574_status = sum(1 for k in doc["session_status"] if k.startswith("000574/"))
    n_beamformed_status = sum(1 for k in doc["session_status"] if k.startswith("ds004752_beamformed/"))
    assert n_000574_status == scope["n_000574_sessions_seen"]
    assert n_beamformed_status == scope["n_beamformed_patients_seen"]
    for key, rec in doc["session_status"].items():
        if rec["status"] == "refused":
            assert rec.get("reason"), f"{key} refused with no machine-readable reason"


def test_pooled_sign_flip_tests_are_internally_consistent():
    doc = _load_artifact()
    for block_name, test_key in (("block_a", "pooled_patient_test"), ("block_b", "primary_within_set_size_test")):
        for tier, rec in doc[block_name].items():
            p = rec.get(test_key)
            if not p:
                continue
            if p["status"] == "underpowered_by_construction":
                assert p["n_patients"] < MIN_PATIENTS_FOR_TEST, f"{block_name}/{tier}"
            elif p["status"] == "tested":
                assert p["n_patients"] >= MIN_PATIENTS_FOR_TEST, f"{block_name}/{tier}"
                assert p["significant"] == (p["p_value"] <= 0.05), f"{block_name}/{tier}"
                assert p["ci_lower"] <= p["mean_value"] <= p["ci_upper"], f"{block_name}/{tier}"
                if p.get("mdd") is not None:
                    assert p["mdd"] >= 0.0, f"{block_name}/{tier}"


def test_every_fired_null_branch_is_powered_below_its_reference_effect():
    # This is the artifact-level regression for the power check this leg's verification pass ran by hand:
    # a branch asserting "no link/agreement above the reported bound" is only licensed when that cell's
    # own minimum detectable difference (or, in Block C's patient-level regime, its CI half-width) is
    # strictly below the reference effect it was judged against -- otherwise the honest branch is the
    # underpowered one, not a null result.
    doc = _load_artifact()
    branches = list(_iter_branches(doc))
    fired = [b for _, b, *_ in branches if b in NULL_BRANCHES]
    assert fired, "expected at least one null branch to fire in this artifact"
    for label, branch, detectable, ref, _sig in branches:
        if branch in NULL_BRANCHES:
            assert ref is not None, label
            assert detectable is not None, label
            assert detectable < ref, (label, detectable, ref)


def test_positive_branches_agree_with_their_own_significance_flag():
    doc = _load_artifact()
    for label, branch, _detectable, _ref, significant in _iter_branches(doc):
        if branch in POSITIVE_BRANCHES:
            assert significant is True, label


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
