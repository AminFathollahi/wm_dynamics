from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_ram_stimulation_response_latency import (  # noqa: E402
    _censored_first_recall,
    _is_zero_recall_list,
    censored_first_recall_contrast,
    contrast_summary,
    per_subject_contrast,
)


def _entry(stim_list, rec_word_rt, recall_period_s=30.0):
    return {"stim_list": stim_list, "rec_word_rt": rec_word_rt, "rec_all_rt": list(rec_word_rt),
            "n_recalled": len(rec_word_rt), "recall_period_s": recall_period_s}


def test_is_zero_recall_list_flags_empty_rec_word_rt():
    assert _is_zero_recall_list(_entry(True, [])) == 1.0
    assert _is_zero_recall_list(_entry(True, [3.2])) == 0.0


def test_censored_first_recall_returns_event_or_censored_pair():
    recalled = _censored_first_recall(_entry(True, [4.0, 9.0]))
    assert recalled == (4.0, 1)
    unrecalled = _censored_first_recall(_entry(True, []))
    assert unrecalled == (30.0, 0)
    no_period = _censored_first_recall({"rec_word_rt": [], "recall_period_s": None})
    assert no_period is None


def test_censored_first_recall_contrast_shorter_stim_latency_gives_negative_mean_diff():
    by_subject_lists = {}
    for subject in range(6):
        lists = {}
        for i in range(8):
            lists[(subject, "stim", i)] = _entry(True, [2.0 + 0.1 * i])
        for i in range(8):
            lists[(subject, "unstim", i)] = _entry(False, [8.0 + 0.1 * i])
        by_subject_lists[f"sub-{subject}"] = lists
    contrasts = censored_first_recall_contrast(by_subject_lists)
    assert len(contrasts) == 6
    summary = contrast_summary(contrasts, "unit_test_censored_first_recall")
    assert summary["status"] == "computed"
    assert summary["mean_diff"] < 0
    assert summary["mean_stim"] < summary["mean_unstim"]


def test_censored_first_recall_contrast_drops_subjects_below_min_lists():
    by_subject_lists = {"sub-0": {("a",): _entry(True, [2.0]), ("b",): _entry(False, [3.0])}}
    contrasts = censored_first_recall_contrast(by_subject_lists, min_lists=3)
    assert contrasts == {}


def test_fraction_zero_recall_per_subject_contrast_matches_hand_count():
    by_subject_lists = {
        "sub-0": {
            ("s", 0): _entry(True, []), ("s", 1): _entry(True, [1.0]),
            ("u", 0): _entry(False, []), ("u", 1): _entry(False, []),
        },
    }
    contrasts = per_subject_contrast(by_subject_lists, _is_zero_recall_list, np.mean)
    assert contrasts["sub-0"]["stim"] == 0.5
    assert contrasts["sub-0"]["unstim"] == 1.0
