"""Tests for scripts/run_train_overlap_decontamination.py's own logic: the
event-table train-overlap geometry and the pre-declared decision rule.
Reuses (never re-implements) the pooling/estimator primitives already
covered by tests/test_stimulation_response_gate_and_panel.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_train_overlap_decontamination import apply_decision_rule, _session_word_contamination  # noqa: E402
from run_stimulation_timing_and_parameter_structure import build_trains_openloop
from stimulation_events import overlaps


# ── train/word overlap geometry (event tables only) ────────────────────────

def test_overlaps_matches_neighbouring_train_that_outlasts_its_own_word():
    """A train starting at word A's onset and lasting 4.6 s outlasts the
    2.5 s gap to word B -- B's pre-window ([onset_B - 0.3, onset_B)) must
    register as overlapping A's train."""
    onset_a, train_duration, spacing = 100.0, 4.6, 2.5
    onset_b = onset_a + spacing
    train_start, train_end = onset_a, onset_a + train_duration
    pre_b0, pre_b1 = onset_b - 0.3, onset_b
    assert overlaps(pre_b0, pre_b1, train_start, train_end)


def test_own_train_started_before_word_onset_would_falsely_contaminate_without_exclusion():
    """The measured real-world case: STIM_ON precedes its own WORD onset by
    ~0.2 s (trigger latency), which puts the own train's start inside the
    0.3 s pre-window -- exactly why _session_word_contamination must
    exclude a word's own train (the one overlapping its on-screen window)
    before checking the pre-window, or every stimulated word would read as
    'contaminated' by construction, regardless of any real neighbour."""
    onset = 147.224
    own_train_start, own_train_end = 147.019, 147.019 + 4.6
    pre0, pre1 = onset - 0.3, onset
    assert overlaps(pre0, pre1, own_train_start, own_train_end)  # the naive (bugged) check would fire


def test_session_word_contamination_excludes_the_owning_train(tmp_path):
    """Two stimulated words 2.5 s apart, each with its own 4.6 s train
    starting ~0.2 s before its own onset (the measured real pattern).
    Word 1 has no other train nearby -> uncontaminated once its own train
    is excluded. Word 2's pre-window is still covered by word 1's
    still-running train -> genuinely contaminated by a NEIGHBOUR."""
    events_tsv = tmp_path / "sub-X_ses-0_task-FR2_events.tsv"
    header = ["onset", "duration", "trial_type", "list", "serialpos", "stimulation"]
    rows = [
        ["99.981", "4.6", "STIM_ON", "1", "-999", "1"],
        ["100.181", "1.6", "WORD", "1", "1", "1"],
        ["102.481", "4.6", "STIM_ON", "1", "-999", "1"],
        ["102.681", "1.6", "WORD", "1", "2", "1"],
    ]
    with open(events_tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    import run_train_overlap_decontamination as mod
    orig = mod._events_path_for_session
    mod._events_path_for_session = lambda session_key: events_tsv
    try:
        result = _session_word_contamination("fake_session", pre_window_s=0.3)
    finally:
        mod._events_path_for_session = orig

    assert result["status"] == "computed"
    word1 = result["by_list_serialpos"][(1, 1)]
    word2 = result["by_list_serialpos"][(1, 2)]
    assert word1["pre_window_contaminated"] is False
    assert word2["pre_window_contaminated"] is True


def test_build_trains_openloop_reads_stim_on_duration_directly():
    rows = [
        {"trial_type": "STIM_ON", "_onset": 10.0, "_duration": 4.6, "amplitude": "1000", "pulse_freq": "50",
         "pulse_width": "300", "stim_duration": "4600", "anode_label": "A1", "cathode_label": "A2"},
        {"trial_type": "STIM_ON", "_onset": 20.0, "_duration": float("nan")},  # malformed, must be skipped
        {"trial_type": "WORD", "_onset": 10.2, "_duration": 1.6},
    ]
    trains = build_trains_openloop(rows)
    assert len(trains) == 1
    assert trains[0]["start"] == 10.0 and trains[0]["end"] == 14.6


# ── decision rule: supported / refuted / inconclusive, exactly as declared ─

def _pre_post_result(pre_p, pre_mean, pre_mdd, post_p, post_mean):
    return {"windows": {
        "pre_stimulation_window": {"status": "computed", "rate_free_deviation_biomarker": {"pooled": {
            "status": "computed", "p_value": pre_p, "mean_value": pre_mean,
            "mdd": {"status": "computed", "mdd": pre_mdd},
        }}},
        "post_stimulation_window": {"status": "computed", "rate_free_deviation_biomarker": {"pooled": {
            "status": "computed", "p_value": post_p, "mean_value": post_mean,
            "mdd": {"status": "computed", "mdd": 0.01},
        }}},
    }}


def test_decision_rule_refuted_when_pre_window_remains_significant():
    result = _pre_post_result(pre_p=0.01, pre_mean=0.08, pre_mdd=0.02, post_p=0.01, post_mean=0.09)
    verdict = apply_decision_rule(result, reference_effect=0.0673)
    assert verdict["verdict"] == "contamination_account_refuted"


def test_decision_rule_supported_when_pre_falls_to_a_powered_null_and_post_survives():
    result = _pre_post_result(pre_p=0.40, pre_mean=0.01, pre_mdd=0.03, post_p=0.02, post_mean=0.09)
    verdict = apply_decision_rule(result, reference_effect=0.0673)
    assert verdict["verdict"] == "contamination_account_supported"


def test_decision_rule_inconclusive_when_uncontaminated_subset_is_underpowered():
    """pre p > 0.05 (looks like a null) but the subset's own mdd (0.20) cannot
    even reach the effect size (0.0673) it would need to rule out -- an
    infeasibility, not a null."""
    result = _pre_post_result(pre_p=0.40, pre_mean=0.01, pre_mdd=0.20, post_p=0.02, post_mean=0.09)
    verdict = apply_decision_rule(result, reference_effect=0.0673)
    assert verdict["verdict"] == "inconclusive_underpowered"


def test_decision_rule_flags_when_post_window_also_fails_in_reduced_subset():
    result = _pre_post_result(pre_p=0.40, pre_mean=0.01, pre_mdd=0.03, post_p=0.30, post_mean=0.02)
    verdict = apply_decision_rule(result, reference_effect=0.0673)
    assert verdict["verdict"] == "contamination_account_partially_supported_post_window_not_significant_in_reduced_subset"
