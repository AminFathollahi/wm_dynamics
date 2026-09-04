"""Tests for scripts/run_human_stimulation_component_response.py.

Three things are checked directly against synthetic data, per the module's
own non-negotiables: (1) a known directional shift injected into synthetic
stimulated trials is recovered by the displacement computation, (2) no
displacement is reported when stimulated and control trials are drawn from
the identical distribution, and (3) the reference used to score stimulated
trials is built from control trials only -- perturbing a stimulated trial's
own values must never change either the reference direction or any control
trial's own leave-one-out deviation. A fourth block checks the channel/shank
exclusion logic and the subject-clustering helpers, since those are the
parts of this module with no counterpart in the estimator's home module."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_human_stimulation_component_response import (  # noqa: E402
    ALPHA,
    MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT,
    _bipolar_channel_shanks,
    _bootstrap_pooled_mdd_displacement,
    _classify_block_a,
    _classify_block_b,
    _classify_pretask_titration,
    _classify_two_arm_meta,
    _contact_shank,
    _displacement_scale_arm,
    _dose_scaling_interpretability_note,
    _exhaustive_sign_flip_check,
    _heterogeneity_guard_capacity,
    _ladder_rung,
    _pretask_titration_dose_arm,
    _pretask_titration_subject_level_slopes,
    _sign_flip_null_capacity,
    _spontaneous_control_sd_direct,
    _subjects_with_amplitude_variation,
    _task_period_dose_arm,
    _task_period_subject_level_slopes,
    _verify_commensurable_normalisation,
    channel_condition_masks,
    compute_block_b_displacement,
    _dose_quantities,
    find_pretask_titration_series,
    minimum_detectable_correlation,
    subject_aggregated_correlation,
    subject_clustered_mean_test,
)


def _random_unit_vectors(rng: np.random.Generator, n: int, d: int, center: np.ndarray, concentration: float) -> np.ndarray:
    """n vectors clustered around `center` (already unit norm): a noisy copy
    of `center` plus isotropic Gaussian noise scaled by `concentration`
    (smaller = tighter cluster around `center`), each row then renormalised
    -- the same generative shape rate_free_state_deviation expects."""
    noise = rng.normal(size=(n, d)) * concentration
    vectors = center[None, :] + noise
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------------
# Planted-displacement recovery
# ---------------------------------------------------------------------------------------------------

def test_planted_displacement_is_recovered():
    rng = np.random.default_rng(0)
    d = 12
    control_center = np.zeros(d)
    control_center[0] = 1.0
    # Stimulated trials point in a KNOWN, different direction -- a 90-degree rotation in the (0, 1)
    # plane, so the true planted displacement is exactly 1 - cos(90 deg) = 1.0 above the control arm's
    # own near-zero self-consistent deviation.
    stim_center = np.zeros(d)
    stim_center[1] = 1.0

    n_ctrl, n_stim = 200, 80
    control_activity = _random_unit_vectors(rng, n_ctrl, d, control_center, concentration=0.05) * 5.0
    stim_activity = _random_unit_vectors(rng, n_stim, d, stim_center, concentration=0.05) * 5.0

    activity = np.vstack([control_activity, stim_activity])
    stim_flag = np.array([0] * n_ctrl + [1] * n_stim)

    out = compute_block_b_displacement(activity, stim_flag)
    control_deviation = out["control_deviation"]
    stim_deviation = out["stim_deviation"]

    # Control trials should show a small deviation (tightly clustered around their own true direction).
    assert np.nanmean(control_deviation) == pytest.approx(0.0, abs=0.05)
    # Stimulated trials, scored against the control-only reference, should show a deviation close to the
    # planted 1 - cos(90 deg) = 1.0.
    assert np.nanmean(stim_deviation) == pytest.approx(1.0, abs=0.1)

    displacement = float(np.nanmean(stim_deviation) - np.nanmean(control_deviation))
    assert displacement == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------------------------------
# Matched-distribution null
# ---------------------------------------------------------------------------------------------------

def test_matched_distribution_produces_no_displacement():
    rng = np.random.default_rng(1)
    d = 12
    center = np.zeros(d)
    center[3] = 1.0

    n_ctrl, n_stim = 200, 80
    # Both arms drawn from the IDENTICAL generative distribution -- no true displacement exists.
    control_activity = _random_unit_vectors(rng, n_ctrl, d, center, concentration=0.3) * 4.0
    stim_activity = _random_unit_vectors(rng, n_stim, d, center, concentration=0.3) * 4.0

    activity = np.vstack([control_activity, stim_activity])
    stim_flag = np.array([0] * n_ctrl + [1] * n_stim)

    out = compute_block_b_displacement(activity, stim_flag)
    displacement = float(np.nanmean(out["stim_deviation"]) - np.nanmean(out["control_deviation"]))
    assert displacement == pytest.approx(0.0, abs=0.08)


# ---------------------------------------------------------------------------------------------------
# Reference construction: control trials only
# ---------------------------------------------------------------------------------------------------

def test_reference_and_control_deviation_are_built_from_control_trials_only():
    rng = np.random.default_rng(2)
    d = 10
    center = np.zeros(d)
    center[0] = 1.0
    n_ctrl, n_stim = 150, 50
    control_activity = _random_unit_vectors(rng, n_ctrl, d, center, concentration=0.1) * 3.0
    stim_activity_a = _random_unit_vectors(rng, n_stim, d, center, concentration=0.1) * 3.0
    # A second, wildly different set of "stimulated" values -- huge magnitude, orthogonal direction.
    orthogonal_center = np.zeros(d)
    orthogonal_center[5] = 1.0
    stim_activity_b = _random_unit_vectors(rng, n_stim, d, orthogonal_center, concentration=0.1) * 500.0

    stim_flag = np.array([0] * n_ctrl + [1] * n_stim)
    activity_a = np.vstack([control_activity, stim_activity_a])
    activity_b = np.vstack([control_activity, stim_activity_b])

    out_a = compute_block_b_displacement(activity_a, stim_flag)
    out_b = compute_block_b_displacement(activity_b, stim_flag)

    # The reference direction and every control trial's own leave-one-out deviation must be IDENTICAL
    # across the two runs: only the stimulated rows differ between activity_a and activity_b, and
    # stimulated trials must never enter the reference or any control trial's own comparison.
    np.testing.assert_allclose(out_a["reference_direction"], out_b["reference_direction"], atol=1e-10)
    np.testing.assert_allclose(out_a["control_deviation"], out_b["control_deviation"], atol=1e-10, equal_nan=True)

    # The stimulated deviations, by contrast, must differ (they were scored against the same reference
    # but from very different raw values).
    assert not np.allclose(out_a["stim_deviation"], out_b["stim_deviation"], atol=1e-6)


# ---------------------------------------------------------------------------------------------------
# Channel / shank exclusion
# ---------------------------------------------------------------------------------------------------

def test_contact_shank_parses_lead_prefix():
    assert _contact_shank("LAH12") == "LAH"
    assert _contact_shank("RTG1") == "RTG"


def test_bipolar_channel_shanks_returns_both_leads():
    assert _bipolar_channel_shanks("LAH1-LAH2") == {"LAH"}
    assert _bipolar_channel_shanks("LAH1-RTG3") == {"LAH", "RTG"}


def test_channel_condition_masks_shank_exclusion_is_a_superset_of_pair_exclusion():
    ch_names = ["LAH1-LAH2", "LAH2-LAH3", "RTG1-RTG2", "LAH3-RTG1"]
    masks = channel_condition_masks(ch_names, anode="LAH1", cathode="LAH2", stim_ch="LAH1-LAH2")
    assert masks["full_channel_set"].sum() == 4
    # Only the exact stimulated pair is dropped.
    assert masks["excluding_stimulated_pair"].tolist() == [False, True, True, True]
    # Every channel touching the LAH lead (including the cross-lead LAH3-RTG1) is dropped too.
    assert masks["excluding_stimulated_shank"].tolist() == [False, False, True, False]
    assert masks["excluding_stimulated_shank"].sum() <= masks["excluding_stimulated_pair"].sum()


# ---------------------------------------------------------------------------------------------------
# Subject clustering
# ---------------------------------------------------------------------------------------------------

def test_subject_clustered_mean_test_collapses_multi_session_subjects_first():
    # Subject A contributes 3 sessions all near +1; subject B contributes 1 session near -1. An
    # unclustered pool would be dominated 3:1 by subject A; the subject-clustered mean must instead be
    # roughly the midpoint of the two subjects' own means, not the pooled session mean.
    values = np.array([1.0, 1.1, 0.9, -1.0])
    subjects = ["A", "A", "A", "B"]
    out = subject_clustered_mean_test(values, subjects)
    assert out["status"] == "computed"
    assert out["n_sessions"] == 4
    assert out["n_subjects"] == 2
    assert out["mean_value"] == pytest.approx((1.0 + -1.0) / 2, abs=0.15)


def test_subject_aggregated_correlation_needs_at_least_four_subjects():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    out = subject_aggregated_correlation(x, y, ["A", "B", "C"])
    assert out["status"] == "not_computable"


def test_minimum_detectable_correlation_shrinks_with_more_subjects():
    small = minimum_detectable_correlation(6)
    large = minimum_detectable_correlation(40)
    assert small["status"] == "computed" and large["status"] == "computed"
    assert large["mdd"] < small["mdd"]


# ---------------------------------------------------------------------------------------------------
# Pre-declared branch classifiers
# ---------------------------------------------------------------------------------------------------

def test_classify_block_a_void_control_overrides_positive_branch():
    main_test = {"status": "computed", "mean_value": 0.2, "p_value": 0.01}
    void_test = {"status": "computed", "r": 0.25, "p_value": 0.02}
    assert _classify_block_a(main_test, {"status": "computed", "mdd": 0.05}, void_test) == \
        "component_recall_link_not_separable_from_a_session_level_offset"


def test_classify_block_a_positive_without_void_reproduction():
    main_test = {"status": "computed", "mean_value": 0.2, "p_value": 0.01}
    void_test = {"status": "computed", "r": 0.01, "p_value": 0.9}
    assert _classify_block_a(main_test, {"status": "computed", "mdd": 0.05}, void_test) == \
        "component_predicts_recall_failure_in_human_intracranial_recording"


def test_classify_block_b_artifact_only_when_full_significant_but_shank_excluded_is_not():
    pooled = {
        "full_channel_set": {"status": "computed", "mean_value": 1.5, "p_value": 0.01,
                             "mdd": {"status": "computed", "mdd": 0.3}},
        "excluding_stimulated_shank": {"status": "computed", "mean_value": 0.1, "p_value": 0.6,
                                       "mdd": {"status": "computed", "mdd": 0.3}},
    }
    assert _classify_block_b(pooled) == "stimulation_displacement_not_separable_from_recording_artifact"


def test_classify_block_b_survives_when_shank_excluded_still_significant_same_sign():
    pooled = {
        "full_channel_set": {"status": "computed", "mean_value": 1.5, "p_value": 0.01,
                             "mdd": {"status": "computed", "mdd": 0.3}},
        "excluding_stimulated_shank": {"status": "computed", "mean_value": 1.2, "p_value": 0.02,
                                       "mdd": {"status": "computed", "mdd": 0.3}},
    }
    assert _classify_block_b(pooled) == "stimulation_displaces_the_component"


# ---------------------------------------------------------------------------------------------------
# Pre-task amplitude titration arm: series discovery and its branch classifier
# ---------------------------------------------------------------------------------------------------

def _write_pretask_events_tsv(tmp_path: Path) -> Path:
    """A synthetic session with three STIM_ON/STIM_OFF groups: a genuine two-level pre-task
    titration at pair A1-A2, a single-amplitude pre-task probe at a different pair B1-B2 (no dose
    axis -- must be dropped), and an in-task stimulation event (list != -999) that must never be
    treated as pre-task at all."""
    ieeg_json = tmp_path / "sub-X_ses-0_task-FR3_acq-bipolar_ieeg.json"
    events_tsv = tmp_path / "sub-X_ses-0_task-FR3_events.tsv"
    common = {"pulse_width": "300", "n_pulses": "10"}
    rows = [
        {"onset": "0.0", "duration": "0.0", "trial_type": "STIM_ON", "list": "-999", "stim_list": "0",
         "amplitude": "250", "anode_label": "A1", "cathode_label": "A2", **common},
        {"onset": "0.5", "duration": "5.0", "trial_type": "STIM_OFF", "list": "-999", "stim_list": "0",
         "amplitude": "250", "anode_label": "A1", "cathode_label": "A2", **common},
        {"onset": "5.5", "duration": "0.0", "trial_type": "STIM_ON", "list": "-999", "stim_list": "0",
         "amplitude": "500", "anode_label": "A1", "cathode_label": "A2", **common},
        {"onset": "6.0", "duration": "5.0", "trial_type": "STIM_OFF", "list": "-999", "stim_list": "0",
         "amplitude": "500", "anode_label": "A1", "cathode_label": "A2", **common},
        {"onset": "11.0", "duration": "0.0", "trial_type": "STIM_ON", "list": "-999", "stim_list": "0",
         "amplitude": "750", "anode_label": "B1", "cathode_label": "B2", **common},
        {"onset": "11.5", "duration": "5.0", "trial_type": "STIM_OFF", "list": "-999", "stim_list": "0",
         "amplitude": "750", "anode_label": "B1", "cathode_label": "B2", **common},
        {"onset": "100.0", "duration": "0.5", "trial_type": "STIM_ON", "list": "3", "stim_list": "1",
         "amplitude": "1000", "anode_label": "A1", "cathode_label": "A2", **common},
    ]
    fieldnames = ["onset", "duration", "trial_type", "list", "stim_list", "amplitude",
                 "anode_label", "cathode_label", "pulse_width", "n_pulses"]
    with open(events_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return ieeg_json


def test_find_pretask_titration_series_filters_single_level_and_ignores_in_task_events(tmp_path):
    series = find_pretask_titration_series(_write_pretask_events_tsv(tmp_path))
    # Only the genuine two-level A1-A2 group survives -- the single-amplitude B1-B2 probe has no
    # dose axis, and the list=3 in-task STIM_ON is never grouped in at all.
    assert len(series) == 1
    assert (series[0]["anode"], series[0]["cathode"]) == ("A1", "A2")
    assert len(series[0]["events"]) == 2
    assert {e["amplitude_uA"] for e in series[0]["events"]} == {250.0, 500.0}


def test_pretask_charge_uses_microampere_microsecond_units():
    charge, total = _dose_quantities(
        np.array([250.0, 1000.0]), np.array([300.0, 300.0]), np.array([5.0, 100.0]))
    np.testing.assert_allclose(charge, [75000.0, 300000.0])
    np.testing.assert_allclose(total, [375000.0, 30000000.0])


def test_classify_pretask_titration_fires_positive_branch_only_with_shuffle_agreement():
    pooled = {"status": "computed", "mean_value": 0.01, "p_value": 0.01,
             "mdd": {"status": "computed", "mdd": 0.02}}
    assert _classify_pretask_titration(pooled, shuffle_p=0.02, n_series_clearing_displacement_threshold=1,
                                        n_series_with_displacement_mdd=1) == \
        "displacement_scales_with_amplitude_within_session"
    # Same pooled result, but the shuffle null does not agree -- must not fire the positive branch.
    assert _classify_pretask_titration(pooled, shuffle_p=0.5, n_series_clearing_displacement_threshold=1,
                                        n_series_with_displacement_mdd=1) != \
        "displacement_scales_with_amplitude_within_session"


def test_classify_pretask_titration_powered_null_only_when_every_series_clears_displacement_threshold():
    # A true displacement-scale powered null: every series' own realised-range mdd clears the threshold.
    pooled = {"status": "computed", "mean_value": 0.0001, "p_value": 0.9,
             "mdd": {"status": "computed", "mdd": 0.05}}
    assert _classify_pretask_titration(pooled, shuffle_p=0.8, n_series_clearing_displacement_threshold=3,
                                        n_series_with_displacement_mdd=3) == "no_scaling_above_the_reported_bound"


def test_classify_pretask_titration_underpowered_when_no_series_clears_displacement_threshold():
    # Reproduces the reported defect's actual numbers: a slope-scale mdd of ~0.00435 per microampere is
    # NOT directly comparable to the level-scale threshold of 1.0 -- converted onto the displacement
    # scale over each of the 13 series' own realised amplitude range, none of them clear it, so the arm
    # must classify as underpowered, not as a powered null.
    pooled = {"status": "computed", "mean_value": -0.00046422595072984015, "p_value": 1.0,
             "mdd": {"status": "computed", "mdd": 0.004350590791976703}}
    assert _classify_pretask_titration(pooled, shuffle_p=0.962037962037962,
                                        n_series_clearing_displacement_threshold=0,
                                        n_series_with_displacement_mdd=13) == "underpowered_to_ask"


def test_classify_pretask_titration_underpowered_when_only_some_series_clear_displacement_threshold():
    # A mixed arm -- some series individually powered on the displacement scale, some not -- must not
    # be called a powered null either: a powered null requires the bound to hold everywhere, not on
    # average.
    pooled = {"status": "computed", "mean_value": 0.0001, "p_value": 0.9,
             "mdd": {"status": "computed", "mdd": 0.05}}
    assert _classify_pretask_titration(pooled, shuffle_p=0.8, n_series_clearing_displacement_threshold=2,
                                        n_series_with_displacement_mdd=3) == "underpowered_to_ask"


def test_classify_pretask_titration_underpowered_when_mdd_above_threshold():
    pooled = {"status": "computed", "mean_value": 0.0001, "p_value": 0.9,
             "mdd": {"status": "computed", "mdd": 5.0}}
    assert _classify_pretask_titration(pooled, shuffle_p=0.8, n_series_clearing_displacement_threshold=0,
                                        n_series_with_displacement_mdd=1) == "underpowered_to_ask"


def test_classify_pretask_titration_not_computable_when_pooled_absent():
    assert _classify_pretask_titration({"status": "not_computable"}, shuffle_p=None,
                                        n_series_clearing_displacement_threshold=None,
                                        n_series_with_displacement_mdd=None) == \
        "not_computable_from_this_recording"


# ---------------------------------------------------------------------------------------------------
# Dose-variation attrition ladder: within-pair vs. any-pair, and the rung accounting itself
# ---------------------------------------------------------------------------------------------------

def test_subjects_with_amplitude_variation_distinguishes_within_pair_from_any_pair():
    # Subject A varies amplitude at ONE pair (a genuine within-pair contrast). Subject B only varies
    # amplitude by also switching electrode pair (a confounded contrast that must not count as
    # within-pair). Subject C never varies at all.
    rows = [
        {"subject": "A", "session_key": "s0", "pair": "X1-X2", "amplitude_uA": 250.0, "is_pretask_titration_event": False},
        {"subject": "A", "session_key": "s1", "pair": "X1-X2", "amplitude_uA": 500.0, "is_pretask_titration_event": False},
        {"subject": "B", "session_key": "s2", "pair": "Y1-Y2", "amplitude_uA": 250.0, "is_pretask_titration_event": False},
        {"subject": "B", "session_key": "s3", "pair": "Y2-Y3", "amplitude_uA": 500.0, "is_pretask_titration_event": False},
        {"subject": "C", "session_key": "s4", "pair": "Z1-Z2", "amplitude_uA": 250.0, "is_pretask_titration_event": False},
    ]
    any_pair, within_pair = _subjects_with_amplitude_variation(rows)
    assert any_pair == {"A", "B"}
    assert within_pair == {"A"}


def test_subjects_with_amplitude_variation_task_period_only_excludes_pretask_events():
    # Subject A's only within-pair variation is confined to a pre-task titration event; with the
    # task-period-only restriction it must not count.
    rows = [
        {"subject": "A", "session_key": "s0", "pair": "X1-X2", "amplitude_uA": 250.0, "is_pretask_titration_event": True},
        {"subject": "A", "session_key": "s1", "pair": "X1-X2", "amplitude_uA": 500.0, "is_pretask_titration_event": False},
    ]
    _, within_pair_all = _subjects_with_amplitude_variation(rows)
    _, within_pair_task_only = _subjects_with_amplitude_variation(rows, task_period_only=True)
    assert within_pair_all == {"A"}
    assert within_pair_task_only == set()


def test_subjects_with_amplitude_variation_session_keys_filter_can_remove_a_subject():
    rows = [
        {"subject": "A", "session_key": "s0", "pair": "X1-X2", "amplitude_uA": 250.0, "is_pretask_titration_event": False},
        {"subject": "A", "session_key": "s1", "pair": "X1-X2", "amplitude_uA": 500.0, "is_pretask_titration_event": False},
    ]
    _, within_pair_full = _subjects_with_amplitude_variation(rows, session_keys={"s0", "s1"})
    _, within_pair_restricted = _subjects_with_amplitude_variation(rows, session_keys={"s0"})
    assert within_pair_full == {"A"}
    assert within_pair_restricted == set()  # only one session's amplitude survives -- no contrast left


def test_ladder_rung_accounting():
    seen = {"A", "B", "C", "D"}
    retained = {"A", "C"}
    rung = _ladder_rung("some_step", "some reason", seen, retained)
    assert rung["n_seen"] == 4
    assert rung["n_retained"] == 2
    assert rung["n_lost"] == 2
    assert rung["subjects_lost"] == ["B", "D"]
    assert rung["n_seen"] == rung["n_retained"] + rung["n_lost"]


def test_ladder_rung_no_loss_when_retained_equals_seen():
    seen = {"A", "B"}
    rung = _ladder_rung("some_step", "some reason", seen, seen)
    assert rung["n_lost"] == 0
    assert rung["subjects_lost"] == []


# ---------------------------------------------------------------------------------------------------
# Two-arm dose-scaling meta-analysis
# ---------------------------------------------------------------------------------------------------

def _block_d_row(subject: str, pair: str, amplitude: float, displacement: float) -> dict:
    return {"subject": subject, "stim_channel": pair, "amplitude_uA": amplitude,
           "normalised_displacement": displacement}


def test_task_period_dose_arm_only_pools_subjects_with_amplitude_variation():
    rows = [
        _block_d_row("A", "P1-P2", 1000.0, 0.1), _block_d_row("A", "P1-P2", 1500.0, 0.3),
        _block_d_row("B", "P3-P4", 500.0, -0.2), _block_d_row("B", "P3-P4", 1000.0, 0.0),
        # Subject C has two sessions but a CONSTANT amplitude -- no dose axis, must be dropped.
        _block_d_row("C", "P5-P6", 1000.0, 0.05), _block_d_row("C", "P5-P6", 1000.0, 0.4),
    ]
    pooled, range_summary, subjects = _task_period_dose_arm(rows, exclude_subjects=set())
    assert subjects == {"A", "B"}
    assert pooled["status"] == "computed" and pooled["n_subjects"] == 2
    assert range_summary["n_subjects"] == 2
    assert range_summary["median_uA"] == pytest.approx(500.0)
    # Subject A's own slope: (0.3-0.1)/(1500-1000) = 0.0004; subject B's: (0.0 - -0.2)/(1000-500) = 0.0004.
    assert pooled["mean_value"] == pytest.approx(0.0004, abs=1e-9)


def test_task_period_dose_arm_exclude_subjects_removes_them_from_pooling():
    rows = [
        _block_d_row("A", "P1-P2", 1000.0, 0.1), _block_d_row("A", "P1-P2", 1500.0, 0.3),
        _block_d_row("B", "P3-P4", 500.0, -0.2), _block_d_row("B", "P3-P4", 1000.0, 0.0),
    ]
    pooled, range_summary, subjects = _task_period_dose_arm(rows, exclude_subjects={"B"})
    assert subjects == {"A"}
    # subject_clustered_mean_test refuses to pool a single subject.
    assert pooled["status"] == "not_computable"


def test_pretask_titration_dose_arm_reuses_precomputed_when_nothing_excluded():
    pretask = {
        "pooled_amplitude_slope_subject_clustered": {"status": "computed", "mean_value": -0.001, "n_subjects": 2},
        "realised_amplitude_range_uA_summary": {"median_uA": 400.0},
        "per_series": {
            "s1": {"status": "computed", "subject": "X", "amplitude_uA_slope": -0.001,
                  "events": [{"amplitude_uA": 250.0}, {"amplitude_uA": 500.0}]},
            "s2": {"status": "computed", "subject": "Y", "amplitude_uA_slope": -0.001,
                  "events": [{"amplitude_uA": 300.0}, {"amplitude_uA": 700.0}]},
        },
    }
    pooled, range_summary, subjects = _pretask_titration_dose_arm(pretask, exclude_subjects=set())
    # Identity, not a recompute: reuses the exact precomputed objects when there is nothing to drop.
    assert pooled is pretask["pooled_amplitude_slope_subject_clustered"]
    assert range_summary is pretask["realised_amplitude_range_uA_summary"]
    assert subjects == {"X", "Y"}


def test_pretask_titration_dose_arm_excludes_overlap_subject_and_rebuilds():
    pretask = {
        "pooled_amplitude_slope_subject_clustered": {"status": "computed", "mean_value": -0.001, "n_subjects": 2},
        "realised_amplitude_range_uA_summary": {"median_uA": 400.0},
        "per_series": {
            "s1": {"status": "computed", "subject": "X", "amplitude_uA_slope": -0.001,
                  "events": [{"amplitude_uA": 250.0}, {"amplitude_uA": 500.0}]},
            "s2": {"status": "computed", "subject": "Y", "amplitude_uA_slope": 0.002,
                  "events": [{"amplitude_uA": 300.0}, {"amplitude_uA": 700.0}]},
        },
    }
    pooled, range_summary, subjects = _pretask_titration_dose_arm(pretask, exclude_subjects={"X"})
    assert subjects == {"Y"}
    assert range_summary["n_series"] == 1
    assert range_summary["median_uA"] == pytest.approx(400.0)  # 700 - 300
    assert pooled["status"] == "not_computable"  # subject_clustered_mean_test needs >= 2 subjects


def test_displacement_scale_arm_multiplies_slope_and_se_by_realised_range():
    pooled_slope = {"status": "computed", "n_subjects": 5, "mean_value": 0.002, "p_value": 0.3,
                    "ci_lower": -0.001, "ci_upper": 0.005,
                    "mdd": {"status": "computed", "mdd": 0.0028015852181129683}}
    range_summary = {"median_uA": 500.0}
    z_factor = 2.8015852181129683
    out = _displacement_scale_arm(pooled_slope, range_summary, "some_arm", z_factor)
    assert out["status"] == "computed"
    assert out["displacement_estimate"] == pytest.approx(0.002 * 500.0)
    # se_slope = mdd / z_factor = 0.001 exactly, by construction of the mdd above.
    assert out["displacement_se"] == pytest.approx(0.001 * 500.0)
    assert out["displacement_ci_lower"] == pytest.approx(-0.001 * 500.0)
    assert out["displacement_ci_upper"] == pytest.approx(0.005 * 500.0)


def test_displacement_scale_arm_not_computable_when_range_missing():
    pooled_slope = {"status": "computed", "n_subjects": 5, "mean_value": 0.002,
                    "mdd": {"status": "computed", "mdd": 0.001}}
    out = _displacement_scale_arm(pooled_slope, {"median_uA": None}, "some_arm", 2.8)
    assert out["status"] == "not_computable"


def _synthetic_session_arrays(rng: np.random.Generator, n_ctrl: int = 30, n_stim: int = 10) -> dict:
    """A minimal synthetic session with the exact fields _spontaneous_control_sd_direct and
    _bin_averaged both read: one bin, three channels, one bipolar stim pair on its own shank so
    excluding_stimulated_shank drops only that one channel."""
    ch_names = np.array(["LA1-LA2", "LB1-LB2", "LC1-LC2"])
    n = n_ctrl + n_stim
    epochs = rng.normal(size=(n, 1, 3)) + np.array([1.0, 1.0, 1.0])
    stim_flag = np.array([0] * n_ctrl + [1] * n_stim)
    return {"epochs_log": epochs.astype(np.float32), "ch_names": ch_names,
           "anode": np.array("LA1"), "cathode": np.array("LA2"), "stim_channel": np.array("LA1-LA2"),
           "stim_flag": stim_flag}


def test_spontaneous_control_sd_direct_matches_manual_computation():
    rng = np.random.default_rng(7)
    arrays = _synthetic_session_arrays(rng)
    sd = _spontaneous_control_sd_direct(arrays)
    assert sd is not None and sd > 0
    # Re-running on the identical arrays must reproduce the identical number (pure function of its input).
    assert _spontaneous_control_sd_direct(arrays) == pytest.approx(sd)


def test_verify_commensurable_normalisation_detects_agreement_and_mismatch():
    rng = np.random.default_rng(11)
    arrays = _synthetic_session_arrays(rng)
    true_sd = _spontaneous_control_sd_direct(arrays)
    pretask = {"per_series": {"s1": {"status": "computed", "session_key": "sess1", "subject": "X"}}}
    closedloop_records = [{"session_key": "sess1", "arrays": arrays}]

    matching_block_b = {"per_session": {"sess1": {"conditions": {"excluding_stimulated_shank": {
        "status": "computed", "spontaneous_control_sd": true_sd}}}}}
    out = _verify_commensurable_normalisation(pretask, matching_block_b, closedloop_records)
    assert out["commensurable"] is True and out["n_sessions_checked"] == 1

    mismatched_block_b = {"per_session": {"sess1": {"conditions": {"excluding_stimulated_shank": {
        "status": "computed", "spontaneous_control_sd": true_sd + 1.0}}}}}
    out_bad = _verify_commensurable_normalisation(pretask, mismatched_block_b, closedloop_records)
    assert out_bad["commensurable"] is False


# ---------------------------------------------------------------------------------------------------
# Two-arm meta-analysis decision rule (pre-declared; see TWO_ARM_META_DECISION_RULE_TEXT)
# ---------------------------------------------------------------------------------------------------

_TASK_ARM_POS = {"status": "computed", "displacement_estimate": 0.5}
_PRETASK_ARM_POS = {"status": "computed", "displacement_estimate": 0.3}
_PRETASK_ARM_NEG = {"status": "computed", "displacement_estimate": -0.3}
_LOW_HETEROGENEITY = {"i_squared": 0.0, "Q_df": 1, "Q_p": 0.8, "p_value": 0.9}
_HIGH_HETEROGENEITY = {"i_squared": 80.0, "Q_df": 1, "Q_p": 0.9, "p_value": 0.9}
_SIGNIFICANT = {"i_squared": 0.0, "Q_df": 1, "Q_p": 0.8, "p_value": 0.001}


def test_classify_two_arm_meta_not_combinable_when_incommensurable():
    assert _classify_two_arm_meta(False, True, _TASK_ARM_POS, _PRETASK_ARM_POS, None, None) == \
        "arms_not_combinable_incommensurable_normalisation"


def test_classify_two_arm_meta_not_combinable_when_overlap_unresolved():
    assert _classify_two_arm_meta(True, False, _TASK_ARM_POS, _PRETASK_ARM_POS, None, None) == \
        "arms_not_combinable_subject_overlap_unresolved"


def test_classify_two_arm_meta_uninterpretable_when_signs_disagree():
    assert _classify_two_arm_meta(True, True, _TASK_ARM_POS, _PRETASK_ARM_NEG,
                                  _LOW_HETEROGENEITY, 0.1) == "pooled_estimate_uninterpretable_arm_disagreement"


def test_classify_two_arm_meta_uninterpretable_when_heterogeneity_substantial():
    assert _classify_two_arm_meta(True, True, _TASK_ARM_POS, _PRETASK_ARM_POS,
                                  _HIGH_HETEROGENEITY, 0.1) == "pooled_estimate_uninterpretable_arm_disagreement"


def test_classify_two_arm_meta_positive_when_significant_and_agreeing():
    assert _classify_two_arm_meta(True, True, _TASK_ARM_POS, _PRETASK_ARM_POS,
                                  _SIGNIFICANT, 0.1) == "dose_scaling_across_both_arms_pooled"


def test_classify_two_arm_meta_powered_null_when_mdd_below_threshold():
    assert _classify_two_arm_meta(True, True, _TASK_ARM_POS, _PRETASK_ARM_POS,
                                  _LOW_HETEROGENEITY, 0.5) == \
        "no_dose_scaling_above_the_reported_bound_pooled_across_arms"


def test_classify_two_arm_meta_underpowered_when_mdd_above_threshold():
    assert _classify_two_arm_meta(True, True, _TASK_ARM_POS, _PRETASK_ARM_POS,
                                  _LOW_HETEROGENEITY, 5.0) == "underpowered_to_ask"


# ---------------------------------------------------------------------------------------------------
# Interpretability note must not claim the two-arm heterogeneity check as independent evidence of
# combinability -- with exactly two arms (Q_df=1) that check cannot fail regardless of the data.
# ---------------------------------------------------------------------------------------------------

def test_interpretability_note_uninterpretable_branches_say_not_the_answer():
    note = _dose_scaling_interpretability_note("pooled_estimate_uninterpretable_arm_disagreement", False)
    assert "must not be read as the answer" in note
    generic_note = _dose_scaling_interpretability_note("underpowered_to_ask", False)
    assert "must not be read as the answer" in generic_note


def test_interpretability_note_when_combinable_does_not_overclaim_the_heterogeneity_check():
    note = _dose_scaling_interpretability_note("dose_scaling_across_both_arms_pooled", True)
    assert "agree in sign" in note
    # The defect this note fixes: claiming the heterogeneity check is itself positive evidence of
    # combinability, when with two arms it structurally cannot return anything else.
    assert "could not have flagged disagreement" in note
    assert "not independent evidence" in note


# ---------------------------------------------------------------------------------------------------
# Subject-level slope extraction for the two-arm meta-analysis's own bootstrap and null-capacity checks
# ---------------------------------------------------------------------------------------------------

def test_task_period_subject_level_slopes_matches_pooled_mean():
    rows = [
        _block_d_row("A", "P1-P2", 1000.0, 0.1), _block_d_row("A", "P1-P2", 1500.0, 0.3),
        _block_d_row("B", "P3-P4", 500.0, -0.2), _block_d_row("B", "P3-P4", 1000.0, 0.0),
    ]
    pooled, _, _ = _task_period_dose_arm(rows, exclude_subjects=set())
    vals = _task_period_subject_level_slopes(rows, set())
    assert len(vals) == pooled["n_subjects"]
    assert vals.mean() == pytest.approx(pooled["mean_value"])


def test_pretask_titration_subject_level_slopes_collapses_multi_series_subjects_first():
    pretask = {
        "per_series": {
            "s1": {"status": "computed", "subject": "X", "amplitude_uA_slope": -0.001},
            "s2": {"status": "computed", "subject": "Y", "amplitude_uA_slope": 0.003},
            "s3": {"status": "computed", "subject": "X", "amplitude_uA_slope": 0.005},
            "s4": {"status": "not_computable_from_this_recording", "subject": "Z", "amplitude_uA_slope": 9.0},
        },
    }
    vals = _pretask_titration_subject_level_slopes(pretask, set())
    # X's own two series collapse to their mean (0.002) first; the not-computed series is dropped.
    assert sorted(vals.tolist()) == pytest.approx([0.002, 0.003])


# ---------------------------------------------------------------------------------------------------
# Sign-flip null capacity (item 1): why own_slope_p_value is exactly 1.0 for both arms
# ---------------------------------------------------------------------------------------------------

def test_sign_flip_null_capacity_matches_known_floor_at_n3_and_n7():
    n3 = _sign_flip_null_capacity(3)
    assert n3["n_distinct_sign_arrangements"] == 8
    assert n3["smallest_attainable_two_sided_p"] == pytest.approx(0.25)
    assert n3["capable_of_significance_at_alpha"] is False

    n7 = _sign_flip_null_capacity(7)
    assert n7["n_distinct_sign_arrangements"] == 128
    assert n7["smallest_attainable_two_sided_p"] == pytest.approx(0.015625)
    assert n7["capable_of_significance_at_alpha"] is True


def test_sign_flip_null_capacity_boundary_between_n5_and_n6():
    # 2/2**5 = 0.0625 > 0.05 (incapable); 2/2**6 = 0.03125 <= 0.05 (capable) -- the floor crosses
    # alpha between n=5 and n=6, independently of ALPHA's own default value.
    assert _sign_flip_null_capacity(5, alpha=ALPHA)["capable_of_significance_at_alpha"] is False
    assert _sign_flip_null_capacity(6, alpha=ALPHA)["capable_of_significance_at_alpha"] is True


def test_exhaustive_sign_flip_check_detects_global_minimum():
    # Two near-cancelling values plus a small residual: every possible sign flip of these three
    # values produces a larger-or-equal-magnitude sum than the actually-observed one.
    out = _exhaustive_sign_flip_check(np.array([5.0, -5.0, 0.1]))
    assert out["n_arrangements_enumerated"] == 8
    assert out["observed_statistic_is_global_minimum_magnitude"] is True
    assert out["exact_two_sided_p"] == pytest.approx(1.0)


def test_exhaustive_sign_flip_check_rejects_non_minimum_observed():
    # All three values share a sign: the observed sum is the unique (up to global negation) MAXIMUM
    # magnitude arrangement, not the minimum, so only 2 of the 8 arrangements tie it.
    out = _exhaustive_sign_flip_check(np.array([5.0, 5.0, 5.0]))
    assert out["observed_statistic_is_global_minimum_magnitude"] is False
    assert out["exact_two_sided_p"] == pytest.approx(2.0 / 8.0)


# ---------------------------------------------------------------------------------------------------
# Bootstrap CI on the pooled minimum detectable displacement (item 2)
# ---------------------------------------------------------------------------------------------------

_BOOT_TASK_VALS = np.array([-0.01, 0.02, 0.005])
_BOOT_PRETASK_VALS = np.array([-0.001, 0.002, 0.0005, 0.003, -0.002, 0.001, 0.0004])
_BOOT_Z = 2.8015852181129683


def test_bootstrap_pooled_mdd_displacement_scales_linearly_with_range():
    # The bootstrap's own RNG is seeded from a fixed string, not from the input values, so doubling
    # both arms' realised amplitude range (the only place range enters) must exactly double every
    # sampled displacement and hence every reported percentile -- an exact, not approximate, check.
    out1 = _bootstrap_pooled_mdd_displacement(_BOOT_TASK_VALS, _BOOT_PRETASK_VALS, 100.0, 100.0, _BOOT_Z, n_boot=300)
    out2 = _bootstrap_pooled_mdd_displacement(_BOOT_TASK_VALS, _BOOT_PRETASK_VALS, 200.0, 200.0, _BOOT_Z, n_boot=300)
    assert out1["n_boot_usable"] == out2["n_boot_usable"]
    assert out2["ci_lower_2.5pct"] == pytest.approx(2 * out1["ci_lower_2.5pct"])
    assert out2["ci_upper_97.5pct"] == pytest.approx(2 * out1["ci_upper_97.5pct"])


def test_bootstrap_pooled_mdd_displacement_threshold_flag_matches_ci_upper():
    out = _bootstrap_pooled_mdd_displacement(_BOOT_TASK_VALS, _BOOT_PRETASK_VALS, 500.0, 500.0, _BOOT_Z, n_boot=500)
    assert out["n_boot_usable"] + out["n_boot_dropped_zero_variance_resample"] == 500
    assert out["ci_lower_2.5pct"] <= out["ci_upper_97.5pct"]
    expected = out["ci_upper_97.5pct"] < MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT
    assert out["upper_bound_below_meaningful_effect_threshold"] == expected


# ---------------------------------------------------------------------------------------------------
# Heterogeneity guard capacity (item 5)
# ---------------------------------------------------------------------------------------------------

def test_heterogeneity_guard_capacity_matches_closed_form_Q_round_trip():
    task_arm = {"label": "task_period", "displacement_estimate": 0.0, "displacement_se": 1.0}
    pretask_arm = {"label": "pretask_titration", "displacement_estimate": 0.5, "displacement_se": 2.0}
    w1, w2 = 1.0 / 1.0 ** 2, 1.0 / 2.0 ** 2
    Q = w1 * w2 / (w1 + w2) * (0.0 - 0.5) ** 2
    forest = {"Q": Q, "Q_df": 1}

    out = _heterogeneity_guard_capacity(task_arm, pretask_arm, forest, i_squared_target=50.0)
    assert out["status"] == "computed"
    # SE=2.0 (pretask) is the larger SE, i.e. the lower-weight arm.
    assert out["lower_weight_arm_label"] == "pretask_titration"
    assert out["Q_needed_for_i_squared_target"] == pytest.approx(2.0)

    needed = out["lower_weight_arm_displacement_estimate_needed"]
    Q_check = w1 * w2 / (w1 + w2) * (task_arm["displacement_estimate"] - needed) ** 2
    assert Q_check == pytest.approx(2.0)


def test_heterogeneity_guard_capacity_identifies_lower_weight_arm_regardless_of_argument_order():
    # Same two arms as above but with the larger-SE (lower-weight) one passed as task_arm instead of
    # pretask_arm -- the lower-weight identification must follow the SE, not the argument position.
    task_arm = {"label": "task_period", "displacement_estimate": 0.5, "displacement_se": 2.0}
    pretask_arm = {"label": "pretask_titration", "displacement_estimate": 0.0, "displacement_se": 1.0}
    w1, w2 = 1.0 / 2.0 ** 2, 1.0 / 1.0 ** 2
    Q = w1 * w2 / (w1 + w2) * (0.5 - 0.0) ** 2
    forest = {"Q": Q, "Q_df": 1}
    out = _heterogeneity_guard_capacity(task_arm, pretask_arm, forest, i_squared_target=50.0)
    assert out["lower_weight_arm_label"] == "task_period"


def test_heterogeneity_guard_capacity_not_computable_at_100pct_target():
    task_arm = {"label": "task_period", "displacement_estimate": 0.0, "displacement_se": 1.0}
    pretask_arm = {"label": "pretask_titration", "displacement_estimate": 0.5, "displacement_se": 2.0}
    out = _heterogeneity_guard_capacity(task_arm, pretask_arm, {"Q": 0.0, "Q_df": 1}, i_squared_target=100.0)
    assert out["status"] == "not_computable"
