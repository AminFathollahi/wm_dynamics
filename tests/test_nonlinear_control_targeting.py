"""Tests for scripts/run_nonlinear_control_targeting.py.

Six things are checked directly: (1) the stimulation input direction projection and its
missing-channel refusal, (2) the pre-declared branch classifier against the reference
threshold, (3) the bias-only control's sign-and-significance voiding rule (never a
magnitude comparison), (4) the leave-one-subject-out mean substitution the bias-only
control is built from, (5) the reproduction gate's exact/mismatch/missing detection
against a delivered artifact's own displacement values, and (6) the subject-majority
admissibility gate and the family choice built on top of it -- including the case where
no family clears its own held-out shuffle."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_nonlinear_control_targeting import (  # noqa: E402
    _cv_r2_shuffle_null,
    admissibility_gate,
    bias_only_voids,
    choose_family,
    classify_branch,
    leave_one_subject_out_bias_only,
    reproduction_gate,
    stimulation_input_latent_direction,
)


# ---------------------------------------------------------------------------------------------------
# (1) Stimulation input direction
# ---------------------------------------------------------------------------------------------------

def test_stimulation_input_latent_direction_projects_the_one_hot_channel():
    ch_names = ["A1-A2", "B1-B2", "C1-C2"]
    V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (n_ch, k)
    direction = stimulation_input_latent_direction(V, ch_names, "B1-B2")
    assert direction == pytest.approx([0.0, 1.0])


def test_stimulation_input_latent_direction_refuses_unknown_channel():
    ch_names = ["A1-A2", "B1-B2"]
    V = np.eye(2)
    assert stimulation_input_latent_direction(V, ch_names, "Z1-Z2") is None


# ---------------------------------------------------------------------------------------------------
# (2) Branch classifier
# ---------------------------------------------------------------------------------------------------

def test_classify_branch_fires_positive_when_significant_and_ci_excludes_zero():
    corr = {"status": "computed", "p_value": 0.01, "ci_lower": 0.1, "ci_upper": 0.5,
           "mdd": {"status": "computed", "mdd": 0.2}}
    assert classify_branch(corr) == "nonlinear_controller_recovers_a_targeting_quantity"


def test_classify_branch_fires_powered_retirement_when_mdd_clears_reference():
    corr = {"status": "computed", "p_value": 0.9, "ci_lower": -0.3, "ci_upper": 0.3,
           "mdd": {"status": "computed", "mdd": 0.10}}
    assert classify_branch(corr, mdd_reference=0.14) == \
        "targeting_claim_retired_no_controller_predicts_its_own_intervention"


def test_classify_branch_fires_inconclusive_when_underpowered():
    corr = {"status": "computed", "p_value": 0.9, "ci_lower": -0.3, "ci_upper": 0.3,
           "mdd": {"status": "computed", "mdd": 0.45}}
    assert classify_branch(corr, mdd_reference=0.14) == "inconclusive_below_detection_floor"


def test_classify_branch_significant_p_but_ci_crossing_zero_is_not_positive():
    # A cluster-robust interval that still crosses zero must never fire the positive branch,
    # even if the point-estimate p-value alone clears alpha.
    corr = {"status": "computed", "p_value": 0.04, "ci_lower": -0.05, "ci_upper": 0.4,
           "mdd": {"status": "computed", "mdd": 0.45}}
    assert classify_branch(corr, mdd_reference=0.14) == "inconclusive_below_detection_floor"


# ---------------------------------------------------------------------------------------------------
# (3) Bias-only voiding rule
# ---------------------------------------------------------------------------------------------------

def test_bias_only_voids_requires_same_sign_and_both_significant():
    real = {"status": "computed", "r": 0.4, "p_value": 0.01}
    bias_same_sign_significant = {"status": "computed", "r": 0.35, "p_value": 0.02}
    bias_wrong_sign = {"status": "computed", "r": -0.35, "p_value": 0.02}
    bias_not_significant = {"status": "computed", "r": 0.35, "p_value": 0.5}
    assert bias_only_voids(real, bias_same_sign_significant) is True
    assert bias_only_voids(real, bias_wrong_sign) is False
    assert bias_only_voids(real, bias_not_significant) is False


def test_bias_only_voids_never_compares_magnitude():
    # A bias-only effect FAR smaller in magnitude than the real one still voids, as long as it
    # keeps the same sign and clears significance -- voiding is on sign and significance only.
    real = {"status": "computed", "r": 0.4, "p_value": 0.01}
    bias_much_smaller = {"status": "computed", "r": 0.02, "p_value": 0.03}
    assert bias_only_voids(real, bias_much_smaller) is True


# ---------------------------------------------------------------------------------------------------
# (4) Leave-one-subject-out mean substitution
# ---------------------------------------------------------------------------------------------------

def test_leave_one_subject_out_bias_only_excludes_the_subjects_own_sessions():
    values = np.array([1.0, 3.0, 10.0, 20.0])
    subjects = ["S1", "S1", "S2", "S2"]
    out = leave_one_subject_out_bias_only(values, subjects)
    # S1's two sessions both get the mean of S2's own sessions (10, 20) -> 15.
    assert out[0] == pytest.approx(15.0)
    assert out[1] == pytest.approx(15.0)
    # S2's two sessions both get the mean of S1's own sessions (1, 3) -> 2.
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------------------
# (5) Reproduction gate
# ---------------------------------------------------------------------------------------------------

def _session(key, displacement, condition="excluding_stimulated_shank"):
    return {"session_key": key, "subject": "sub-X",
            "displacement_conditions": {condition: {"status": "computed", "displacement": displacement}}}


def test_reproduction_gate_reports_exact_when_values_match():
    sessions = [_session("s1", 0.123), _session("s2", -0.456)]
    component = {"block_b": {"per_session": {
        "s1": {"conditions": {"excluding_stimulated_shank": {"status": "computed", "displacement": 0.123}}},
        "s2": {"conditions": {"excluding_stimulated_shank": {"status": "computed", "displacement": -0.456}}},
    }}}
    out = reproduction_gate(sessions, component)
    assert out == {"n_exact": 2, "n_mismatched": 0, "mismatched_sessions": [], "n_missing_in_source": 0,
                   "outcome": "exact"}


def test_reproduction_gate_flags_a_mismatch_by_name_and_never_masks_it_as_exact():
    sessions = [_session("s1", 0.123)]
    component = {"block_b": {"per_session": {
        "s1": {"conditions": {"excluding_stimulated_shank": {"status": "computed", "displacement": 0.999}}},
    }}}
    out = reproduction_gate(sessions, component)
    assert out["outcome"] == "not_exact"
    assert out["mismatched_sessions"] == ["s1"]


def test_reproduction_gate_flags_a_session_missing_from_the_source_artifact():
    sessions = [_session("s1", 0.123)]
    component = {"block_b": {"per_session": {}}}
    out = reproduction_gate(sessions, component)
    assert out["outcome"] == "not_exact"
    assert out["n_missing_in_source"] == 1


# ---------------------------------------------------------------------------------------------------
# (6) Admissibility gate and family choice
# ---------------------------------------------------------------------------------------------------

def _dyn(gbr_pass, krr_pass, gbr_delta=0.1, krr_delta=0.1):
    return {
        "status": "computed",
        "admissibility": {
            "gbr": {"clears_shuffle": gbr_pass, "r2_cv": gbr_delta, "r2_null": 0.0},
            "krr": {"clears_shuffle": krr_pass, "r2_cv": krr_delta, "r2_null": 0.0},
        },
    }


def test_admissibility_gate_passes_when_majority_of_a_subjects_own_sessions_clear():
    per_session = {
        "s1a": _dyn(True, False), "s1b": _dyn(True, False),   # subject S1: 2/2 gbr -> passes
        "s2a": _dyn(False, False),                            # subject S2: 0/1 gbr -> fails
    }
    subject_of = {"s1a": "S1", "s1b": "S1", "s2a": "S2"}
    gate = admissibility_gate(per_session, subject_of, "gbr")
    assert gate["n_subjects"] == 2
    assert gate["n_subjects_passed"] == 1
    assert gate["admissible"] is False  # 1/2 == 0.5, strictly greater than 0.5 required


def test_choose_family_picks_the_admissible_family_with_the_larger_mean_delta():
    per_session = {
        "s1": _dyn(True, True, gbr_delta=0.9, krr_delta=0.3),
        "s2": _dyn(True, True, gbr_delta=0.8, krr_delta=0.2),
    }
    subject_of = {"s1": "S1", "s2": "S2"}
    gate_gbr = admissibility_gate(per_session, subject_of, "gbr")
    gate_krr = admissibility_gate(per_session, subject_of, "krr")
    choice = choose_family(gate_gbr, gate_krr, per_session)
    assert choice["chosen"] == "gbr"


def test_choose_family_retires_when_neither_family_is_admissible():
    per_session = {"s1": _dyn(False, False), "s2": _dyn(False, False)}
    subject_of = {"s1": "S1", "s2": "S2"}
    gate_gbr = admissibility_gate(per_session, subject_of, "gbr")
    gate_krr = admissibility_gate(per_session, subject_of, "krr")
    choice = choose_family(gate_gbr, gate_krr, per_session)
    assert choice["chosen"] is None
    assert choice["reason"] == "no_admissible_nonlinear_forward_model"


# ---------------------------------------------------------------------------------------------------
# (7) Held-out shuffle null -- plumbing check on a tiny, fast linear model
# ---------------------------------------------------------------------------------------------------

def test_cv_r2_shuffle_null_returns_a_finite_score_on_a_deterministic_linear_system():
    rng_data = np.random.default_rng(0)
    N, T, d = 12, 6, 2
    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    Z = np.empty((N, T, d))
    for i in range(N):
        Z[i, 0] = rng_data.normal(size=d)
        for t in range(1, T):
            Z[i, t] = Z[i, t - 1] @ A.T + 0.01 * rng_data.normal(size=d)

    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    score = _cv_r2_shuffle_null(Z, LinearRegression, rng)
    assert np.isfinite(score)
