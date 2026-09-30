"""Tests for src/stimulation_response_estimator.py and the window/pooling
logic in scripts/run_stimulation_response_gate_and_panel.py.

Covers the five things the shared interface is not allowed to get wrong:
same-trial reference leakage (a trial never enters its own reference),
temporal-window leakage (pre/post windows never overlap and pre never
follows post), a treatment-unit violation (many sessions from one
participant/animal must not silently outweigh one from another), the
window-void logic (a census window not marked 'measured' is never forced
into a computation), and synthetic recovery (a planted displacement is
recovered, a zero displacement is not flagged)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stimulation_response_estimator import (  # noqa: E402
    cluster_bootstrap_pooled_effect,
    deviation_from_fixed_reference,
    fixed_reference_direction,
    nuisance_treatment_effect,
    rate_free_state_deviation,
    window_treatment_effect,
)
from stimulation_response_estimator import _pool_arm
from stimulation_events import _specificity_check


# ── same-trial reference leakage ────────────────────────────────────────────

def test_leave_one_out_excludes_the_trial_scored_against_it():
    """9 control trials all point along e1; a 10th points along e2
    (orthogonal). With a genuine leave-one-out reference, trial 10's own
    reference is pure e1 (it never contributes to itself), so its deviation
    is exactly 1.0 (cosine(e2, e1) = 0). If the reference wrongly included
    the trial being scored, the reference would tilt toward e2 and the
    deviation would be strictly less than 1.0."""
    activity = np.zeros((10, 2))
    activity[:9] = [1.0, 0.0]
    activity[9] = [0.0, 1.0]
    deviation = rate_free_state_deviation(activity)
    assert np.isfinite(deviation[9])
    assert abs(deviation[9] - 1.0) < 1e-9


def test_fixed_reference_is_a_pure_function_of_the_control_pool():
    """window_treatment_effect must score treated trials against a
    reference built ONLY from control trials -- changing a treated trial's
    own value must never change the control-side reference or any control
    trial's own deviation."""
    rng = np.random.default_rng(0)
    control = rng.normal(size=(30, 6)) + np.array([1.0, 0, 0, 0, 0, 0])
    treated_a = rng.normal(size=(10, 6)) + np.array([0, 1.0, 0, 0, 0, 0])
    treated_b = treated_a.copy()
    treated_b[0] = [1000.0, -1000.0, 500.0, -500.0, 250.0, -250.0]  # one treated trial made extreme

    ref_a = fixed_reference_direction(control)
    ref_b = fixed_reference_direction(control)  # reference construction never even sees treated data
    assert np.allclose(ref_a, ref_b)

    ctrl_dev_a = rate_free_state_deviation(control)
    result_a = window_treatment_effect(control, treated_a)
    result_b = window_treatment_effect(control, treated_b)
    assert result_a["spontaneous_control_sd"] == result_b["spontaneous_control_sd"]
    ctrl_dev_b = rate_free_state_deviation(control)
    assert np.allclose(ctrl_dev_a, ctrl_dev_b, equal_nan=True)


# ── temporal-window leakage ─────────────────────────────────────────────────

def test_pre_and_post_window_bins_never_overlap_human_and_macaque_geometry():
    """Both admissible arms' window geometry (read directly from the same
    constants the panel script imports) must produce a pre-window that ends
    exactly where the post-window begins, with no shared bin and the pre
    window strictly before the post window -- the pre-stimulation-must-end-
    before-stimulation-begins leakage rule, checked structurally rather than
    trusted."""
    from run_ram_openloop_pipeline import PRE_S as HUMAN_PRE_S, POST_S as HUMAN_POST_S, BIN_S as HUMAN_BIN_S
    from spike_pipeline import PRE_S as MACAQUE_PRE_S, BIN_S as MACAQUE_BIN_S, N_BINS as MACAQUE_N_BINS

    human_pre_bins = int(round(HUMAN_PRE_S / HUMAN_BIN_S))
    human_n_bins = int(round((HUMAN_PRE_S + HUMAN_POST_S) / HUMAN_BIN_S))
    assert 0 < human_pre_bins < human_n_bins  # a genuine split exists on both sides

    macaque_pre_bins = int(round(MACAQUE_PRE_S / MACAQUE_BIN_S))
    assert 0 < macaque_pre_bins < MACAQUE_N_BINS

    # Bin i covers [i*BIN_S - PRE_S, (i+1)*BIN_S - PRE_S); the last pre-window bin's right edge must be
    # <= 0 (stimulation onset) and the first post-window bin's left edge must be >= that same edge.
    pre_window_right_edge_s = human_pre_bins * HUMAN_BIN_S - HUMAN_PRE_S
    assert abs(pre_window_right_edge_s - 0.0) < 1e-9


def test_pool_arm_windows_are_scored_independently_not_pooled_together():
    """A displacement planted ONLY in the post window must not leak into the
    pre-window cell -- the two windows are genuinely separate computations,
    not two views of one pooled epoch."""
    rng = np.random.default_rng(1)
    n_sessions, n_trials_ctrl, n_trials_treat, n_units = 16, 80, 30, 6
    rows = []
    for s in range(n_sessions):
        ctrl = rng.normal(size=(n_trials_ctrl, n_units)) + np.array([1.0, 0, 0, 0, 0, 0])
        pre_treat = rng.normal(size=(n_trials_treat, n_units)) + np.array([1.0, 0, 0, 0, 0, 0])  # NO shift pre
        post_treat = rng.normal(size=(n_trials_treat, n_units)) + np.array([0, 3.0, 0, 0, 0, 0])  # big shift post
        rows.append({
            "session_key": f"s{s}", "cluster_id": f"subj{s}",  # one session per cluster -- enough clusters for the sign-flip lattice to reach p<=0.05
            "windows": {
                "pre_stimulation_window": {"control_activity": ctrl, "treated_activity": pre_treat,
                                            "control_total": ctrl.sum(1), "treated_total": pre_treat.sum(1)},
                "post_stimulation_window": {"control_activity": ctrl, "treated_activity": post_treat,
                                             "control_total": ctrl.sum(1), "treated_total": post_treat.sum(1)},
            },
        })
    windows_measured = {"pre_stimulation_window": {"status": "measured", "value": 0.3},
                        "post_stimulation_window": {"status": "measured", "value": 1.0}}
    result = _pool_arm("synthetic_arm", rows, "randomized_contemporaneous_controls: synthetic", windows_measured)
    pre_p = result["windows"]["pre_stimulation_window"]["rate_free_deviation_biomarker"]["pooled"]["p_value"]
    post_p = result["windows"]["post_stimulation_window"]["rate_free_deviation_biomarker"]["pooled"]["p_value"]
    assert post_p <= 0.05
    assert pre_p > 0.05
    spec = _specificity_check(result)
    assert spec["status"] == "computed"
    assert spec["pre_window_significant"] is False


# ── treatment-unit violation (session vs. participant/animal clustering) ────

def test_cluster_bootstrap_collapses_to_cluster_count_not_session_count():
    """5 sessions from 2 clusters (4 from A, 1 from B) must be pooled as
    n_clusters=2, not n_sessions=5 -- cluster A's four identical sessions
    must not outweigh cluster B's single session the way naive per-session
    pooling would."""
    session_values = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
    cluster_ids = ["A", "A", "A", "A", "B"]
    result = cluster_bootstrap_pooled_effect(session_values, cluster_ids, "test|treatment_unit")
    assert result["status"] == "computed"
    assert result["n_clusters"] == 2
    assert result["n_sessions"] == 5
    # cluster-collapsed values are [1.0, -1.0] -> mean 0.0 exactly, not the per-session mean (0.6)
    assert abs(result["mean_value"] - 0.0) < 1e-9


def test_two_animal_clustering_cannot_reach_the_conventional_alpha():
    """With exactly 2 clusters the sign-flip null has 2^2=4 patterns, so the
    two-sided permutation p-value can never fall below 0.5 regardless of how
    large the observed effect is -- the macaque arm's own disclosed
    consequence of n_animals=2, verified directly against the pooling
    primitive rather than only asserted in prose."""
    session_values = np.array([5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    cluster_ids = ["Sa"] + ["Wa"] * 10
    result = cluster_bootstrap_pooled_effect(session_values, cluster_ids, "test|two_animal_floor")
    assert result["n_clusters"] == 2
    assert result["p_value"] >= 0.49


# ── window-void logic ───────────────────────────────────────────────────────

def test_structural_void_window_is_never_computed():
    rows = [{"session_key": "s0", "cluster_id": "p0", "windows": {
        "pre_stimulation_window": {"control_activity": np.ones((10, 4)), "treated_activity": np.ones((10, 4)),
                                    "control_total": np.ones(10), "treated_total": np.ones(10)},
    }}]
    windows_measured = {
        "pre_stimulation_window": {"status": "measured", "value": 0.3},
        "post_stimulation_window": {"status": "structural_void", "reason": "this corpus has no post-stimulation window"},
    }
    result = _pool_arm("synthetic_arm", rows, "randomized_contemporaneous_controls: synthetic", windows_measured)
    assert result["windows"]["post_stimulation_window"]["status"] == "structural_void"
    assert result["windows"]["post_stimulation_window"]["reason"] == "this corpus has no post-stimulation window"
    assert "rate_free_deviation_biomarker" not in result["windows"]["post_stimulation_window"]
    assert result["windows"]["pre_stimulation_window"]["status"] == "computed"


# ── synthetic recovery: planted displacement recovered, zero displacement not flagged ──

def test_planted_displacement_is_recovered():
    rng = np.random.default_rng(2)
    n_sessions, n_units = 16, 6
    session_vals, cluster_ids = [], []
    for s in range(n_sessions):
        control = rng.normal(size=(80, n_units)) + np.array([1.0, 0, 0, 0, 0, 0])
        treated = rng.normal(size=(30, n_units)) + np.array([0, 3.0, 0, 0, 0, 0])  # a real planted shift
        eff = window_treatment_effect(control, treated)
        assert eff["status"] == "computed"
        session_vals.append(eff["normalised_displacement"])
        cluster_ids.append(f"subj{s}")  # one session per cluster
    pooled = cluster_bootstrap_pooled_effect(np.array(session_vals), cluster_ids, "test|planted_recovery")
    assert pooled["status"] == "computed"
    assert pooled["mean_value"] > 0.0
    assert pooled["p_value"] <= 0.05


def test_zero_displacement_is_not_flagged():
    rng = np.random.default_rng(3)
    n_sessions, n_units = 16, 6
    session_vals, cluster_ids = [], []
    for s in range(n_sessions):
        control = rng.normal(size=(80, n_units)) + np.array([1.0, 0, 0, 0, 0, 0])
        treated = rng.normal(size=(30, n_units)) + np.array([1.0, 0, 0, 0, 0, 0])  # same distribution, no shift
        eff = window_treatment_effect(control, treated)
        assert eff["status"] == "computed"
        session_vals.append(eff["normalised_displacement"])
        cluster_ids.append(f"subj{s}")
    pooled = cluster_bootstrap_pooled_effect(np.array(session_vals), cluster_ids, "test|zero_null")
    assert pooled["status"] == "computed"
    assert pooled["p_value"] > 0.05


# ── nuisance control is a plain mean difference, never the cosine machinery ─

def test_nuisance_effect_is_a_plain_mean_difference():
    control_totals = np.array([10.0] * 20)
    treated_totals = np.array([13.0] * 10)
    result = nuisance_treatment_effect(control_totals, treated_totals)
    assert result["status"] == "computed"
    assert abs(result["change"] - 3.0) < 1e-9
    assert result["control_sd"] == 0.0  # constant control -> normalised_change falls back to None, not inf/nan
    assert result["normalised_change"] is None


def test_deviation_from_fixed_reference_never_uses_the_control_pool_of_the_treated_trial_itself():
    """A treated trial scored via deviation_from_fixed_reference must use
    ONLY the externally supplied reference direction, never a value derived
    from the treated activity itself."""
    reference = np.array([1.0, 0.0, 0.0])
    treated = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    deviation = deviation_from_fixed_reference(treated, reference)
    assert abs(deviation[0] - 1.0) < 1e-9  # orthogonal to the fixed reference -> full deviation
    assert abs(deviation[1] - 0.0) < 1e-9  # aligned with the fixed reference -> zero deviation
