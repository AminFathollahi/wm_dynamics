from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_haslacher_phase_omega import PHASE_CONDITIONS  # noqa: E402
from run_phase_locked_scalp_stimulation_component import (  # noqa: E402
    REFUSED_IMPLEMENTATION_FAILURE,
    _harmonic_permutation_test,
    circular_distance_deg,
    synthesize_output,
)

# Every branch name any test in this corpus may pre-declare -- read directly off the source so this list
# cannot silently drift from the decision rules themselves.
PRE_DECLARED_BRANCH_NAMES = {
    "inconclusive_below_detection_floor",
    "the_component_is_present_in_the_non_invasive_timing_corpus",
    "the_component_is_absent_in_the_non_invasive_timing_corpus",
    "component_behaviour_link_not_separable_from_a_participant_level_offset",
    "the_component_predicts_accuracy_in_a_non_invasive_human_maintenance_delay",
    "the_component_does_not_predict_accuracy_in_a_non_invasive_human_maintenance_delay",
    "phase_modulation_is_present_in_both_groups",
    "the_stimulation_phase_modulates_the_component_in_the_targeted_group_only",
    "the_stimulation_phase_does_not_modulate_the_component",
    "the_predictor_predicts_the_outcome_across_participants",
    "no_link_above_the_reported_bound",
    "underpowered_to_ask",
}


def _computed_stub(subject: str, group: str, signed_effect: float) -> dict:
    """The minimal shape of a 'computed' participant record that the group-level synthesis in
    synthesize_output reads from -- enough fields for it to run its dict comprehensions without needing
    real EEG data behind them."""
    return {
        "status": "computed", "subject": subject, "group": group,
        "presence_test": {"gate": {"status": "computed", "signed_effect": signed_effect}},
        "behaviour_link": {"status": "too_few_trials_with_outcome", "n_errors": 0},
        "phase_modulation_primary": {"status": "excluded"},
        "phase_modulation_off_band_control": {"status": "excluded"},
        "phase_modulation_stimulation_off_control": {"status": "excluded"},
        "behavioural_benefit_modulation": {"depth": None},
        "component_displacement_between_blocks": None,
        "circular_distance_component_min_to_accuracy_max_deg": None,
    }


def _synthetic_phase_labelled_trials(depth: float, preferred_phase_deg: float, n_per_condition: int,
                                      noise_sd: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A per-trial deviation array whose per-condition mean is a known circular harmonic of the
    six phase-condition trigger codes, plus per-trial noise -- the same construction the phase-modulation
    test itself fits, so recovering the planted depth and phase back out of it is a direct check of the
    harmonic-plus-permutation ladder, not a check of anything about real EEG data."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c]), n_per_condition)
    phase_deg = np.array([PHASE_CONDITIONS[c] for c in codes], dtype=float)
    mean_level = 0.5
    signal = mean_level + depth * np.cos(np.deg2rad(phase_deg) - np.deg2rad(preferred_phase_deg))
    deviation = signal + rng.normal(scale=noise_sd, size=len(codes))
    return deviation, codes


def test_planted_phase_modulation_recovered_in_depth_and_phase():
    depth, preferred_phase = 0.35, 200.0
    deviation, codes = _synthetic_phase_labelled_trials(
        depth=depth, preferred_phase_deg=preferred_phase, n_per_condition=40, noise_sd=0.03, seed=1)
    result = _harmonic_permutation_test(deviation, codes, "test_planted_modulation", n_permutations=1000)
    assert result is not None
    assert abs(result["harmonic"]["amplitude"] - depth) < 0.05
    phase_error = circular_distance_deg(result["harmonic"]["optimal_phase_deg"], preferred_phase)
    assert phase_error < 5.0
    assert result["p_value"] <= 0.05


def test_no_structure_null_is_not_significant():
    rng = np.random.default_rng(2)
    codes = np.repeat(sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c]), 40)
    deviation = 0.5 + rng.normal(scale=0.05, size=len(codes))  # phase-independent by construction
    result = _harmonic_permutation_test(deviation, codes, "test_no_structure_null", n_permutations=1000)
    assert result is not None
    assert result["p_value"] > 0.05


def test_circular_distance_wraps_correctly():
    assert circular_distance_deg(10.0, 350.0) == 20.0
    assert circular_distance_deg(0.0, 180.0) == 180.0
    assert circular_distance_deg(45.0, 45.0) == 0.0


def test_all_participants_refused_by_exception_produces_no_pre_declared_branch():
    """Reproduces the failure this test is guarding against: two participants both crashed with an
    uncaught TypeError inside process_participant, and the artifact reported that as
    'inconclusive_below_detection_floor' -- a scientific verdict that never touched any data. A crash
    must be reported as a crash."""
    participants = [("SX1", "active"), ("SX2", "control")]
    exception_reason = "exception:TypeError:'>' not supported between instances of 'NoneType' and 'float'"
    records = {
        "SX1": {"status": "refused", "subject": "SX1", "group": "active", "reason": exception_reason},
        "SX2": {"status": "refused", "subject": "SX2", "group": "control", "reason": exception_reason},
    }
    gate = {"status": "matched", "max_abs_diff": 0.0}

    output = synthesize_output(records, participants, ["SX1"], ["SX2"], gate, t0=0.0)

    assert output["presence_test"]["branch"] == REFUSED_IMPLEMENTATION_FAILURE
    assert output["behaviour_link"]["branch"] == REFUSED_IMPLEMENTATION_FAILURE
    assert output["presence_test"]["branch"] not in PRE_DECLARED_BRANCH_NAMES
    assert output["behaviour_link"]["branch"] not in PRE_DECLARED_BRANCH_NAMES
    assert output["status"] != "complete"
    assert output["phase_modulation"]["status"] == "not_run"
    assert output["benefit_prediction"]["status"] == "not_run"
    assert len(output["presence_test"]["exception_refusals"]) == 2
    assert all(r["reason"].startswith("exception:") for r in output["presence_test"]["exception_refusals"])

    # No pre-declared branch name may appear anywhere a branch is reported for this bundle -- a crash
    # is a crash at every level of the artifact, not just the two tests checked above by name.
    reported_branches = {
        output["presence_test"]["branch"], output["behaviour_link"]["branch"],
        output["phase_modulation"].get("branch"), output["benefit_prediction"].get("branch"),
    }
    assert not (reported_branches & PRE_DECLARED_BRANCH_NAMES)


def test_presence_absence_blocks_phase_and_prediction_tests():
    """The presence test gates the phase-modulation and benefit-prediction tests by rule: if the
    component is absent, those tests must not run at all -- not run and come back empty."""
    active_ids = [f"SA{i}" for i in range(6)]
    participants = [(s, "active") for s in active_ids]
    records = {s: _computed_stub(s, "active", signed_effect=0.001 * i) for i, s in enumerate(active_ids)}
    gate = {"status": "matched", "max_abs_diff": 0.0}

    with patch("run_phase_locked_scalp_stimulation_component._classify_presence",
               return_value="the_component_is_absent_in_the_non_invasive_timing_corpus"):
        output = synthesize_output(records, participants, active_ids, [], gate, t0=0.0)

    assert output["presence_test"]["branch"] == "the_component_is_absent_in_the_non_invasive_timing_corpus"
    assert output["phase_modulation"]["status"] == "not_run"
    assert output["benefit_prediction"]["status"] == "not_run"
