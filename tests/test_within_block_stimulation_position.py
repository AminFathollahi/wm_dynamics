"""Tests for scripts/run_within_block_stimulation_position.py's own logic:
the event-table within-block position geometry and the pre-declared
decision rule. Reuses (never re-implements) the pooling/estimator
primitives already covered by tests/test_stimulation_response_gate_and_panel.py
and tests/test_train_overlap_decontamination.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_within_block_stimulation_position import (  # noqa: E402
    apply_decision_rule, _session_word_positions,
)


# ── within-block position geometry (event tables only) ─────────────────────

def _write_events(tmp_path, rows):
    events_tsv = tmp_path / "sub-X_ses-0_task-FR2_events.tsv"
    header = ["onset", "duration", "trial_type", "list", "serialpos", "stimulation"]
    with open(events_tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return events_tsv


def test_two_word_block_ranks_first_and_second_in_pair(tmp_path):
    """The measured real-world pattern (spot-checked ds005489 session): one
    4.6 s train starting ~0.2 s before the first of two stimulated words,
    spanning both on-screen presentations 2.5 s apart. Word 1 must rank
    first_in_pair, word 2 second_in_pair, sharing one train_group_size=2."""
    rows = [
        ["147.019", "4.6", "STIM_ON", "1", "-999", "1"],
        ["147.224", "1.6", "WORD", "1", "1", "1"],
        ["149.808", "1.6", "WORD", "1", "2", "1"],
        ["152.292", "1.6", "WORD", "1", "3", "0"],
    ]
    events_tsv = _write_events(tmp_path, rows)

    import run_within_block_stimulation_position as mod
    orig = mod._events_path_for_session
    mod._events_path_for_session = lambda session_key: events_tsv
    try:
        result = _session_word_positions("fake_session", pre_window_s=0.3)
    finally:
        mod._events_path_for_session = orig

    assert result["status"] == "computed"
    assert result["train_group_size_histogram"] == {"2": 1}
    first, second = result["by_list_serialpos"][(1, 1)], result["by_list_serialpos"][(1, 2)]
    assert first["within_block_position"] == "first_in_pair"
    assert second["within_block_position"] == "second_in_pair"


def test_first_in_pair_pre_window_only_partially_covered_by_own_train():
    """First-in-pair coverage is the small trigger-latency lead-in only
    (own train starts ~0.2 s before its own word, inside the 0.3 s
    pre-window but nowhere near filling it) -- qualitatively different from
    a second word whose entire pre-window sits inside an already-running
    train. This is the empirical distinction the decision rule leans on."""
    rows = [
        ["147.019", "4.6", "STIM_ON", "1", "-999", "1"],
        ["147.224", "1.6", "WORD", "1", "1", "1"],
        ["149.808", "1.6", "WORD", "1", "2", "1"],
    ]

    def _run(tmp_path):
        events_tsv = _write_events(tmp_path, rows)
        import run_within_block_stimulation_position as mod
        orig = mod._events_path_for_session
        mod._events_path_for_session = lambda session_key: events_tsv
        try:
            return _session_word_positions("fake_session", pre_window_s=0.3)
        finally:
            mod._events_path_for_session = orig

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result = _run(Path(d))

    first, second = result["by_list_serialpos"][(1, 1)], result["by_list_serialpos"][(1, 2)]
    assert 0.0 < first["own_train_pre_window_coverage_fraction"] < 1.0
    assert second["own_train_pre_window_coverage_fraction"] == pytest.approx(1.0)


def test_word_with_no_enclosing_train_is_reported_not_dropped_silently():
    rows = [
        ["100.0", "1.6", "WORD", "1", "1", "1"],  # stimulation=1 but no STIM_ON row at all
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        events_tsv = _write_events(Path(d), rows)
        import run_within_block_stimulation_position as mod
        orig = mod._events_path_for_session
        mod._events_path_for_session = lambda session_key: events_tsv
        try:
            result = _session_word_positions("fake_session", pre_window_s=0.3)
        finally:
            mod._events_path_for_session = orig

    assert result["status"] == "computed"
    assert result["n_stim_words_no_own_train_identified"] == 1
    assert (1, 1) not in result["by_list_serialpos"]


def test_three_word_group_ranks_beyond_second_reported_not_assumed(tmp_path):
    """A hypothetical (not the measured pattern) three-word group must not
    silently collapse into first/second -- rank 2 gets its own label so a
    deviation from the two-word structure would be visible, not hidden."""
    rows = [
        ["100.0", "7.0", "STIM_ON", "1", "-999", "1"],
        ["100.2", "1.6", "WORD", "1", "1", "1"],
        ["102.7", "1.6", "WORD", "1", "2", "1"],
        ["105.2", "1.6", "WORD", "1", "3", "1"],
    ]
    events_tsv = _write_events(tmp_path, rows)
    import run_within_block_stimulation_position as mod
    orig = mod._events_path_for_session
    mod._events_path_for_session = lambda session_key: events_tsv
    try:
        result = _session_word_positions("fake_session", pre_window_s=0.3)
    finally:
        mod._events_path_for_session = orig

    assert result["train_group_size_histogram"] == {"3": 1}
    assert result["by_list_serialpos"][(1, 3)]["within_block_position"] == "third_or_later_in_group"


# ── decision rule: account_b / account_a / inconclusive / neither ──────────

def _arm(pre_p, pre_mean, pre_mdd):
    return {"windows": {"pre_stimulation_window": {"status": "computed", "rate_free_deviation_biomarker": {
        "pooled": {"status": "computed", "p_value": pre_p, "mean_value": pre_mean,
                   "mdd": {"status": "computed", "mdd": pre_mdd}}}}}}


def test_account_b_confirmed_when_first_in_pair_alone_is_significant():
    first = _arm(pre_p=0.02, pre_mean=0.06, pre_mdd=0.03)
    second = _arm(pre_p=0.01, pre_mean=0.09, pre_mdd=0.03)
    verdict = apply_decision_rule(first, second)
    assert verdict["verdict"] == "account_b_confirmed"


def test_account_b_confirmed_takes_priority_even_if_second_in_pair_is_null():
    """account_b is checked first and does not depend on the second_in_pair
    reading at all -- a first-in-pair effect is diagnostic on its own."""
    first = _arm(pre_p=0.03, pre_mean=0.05, pre_mdd=0.02)
    second = _arm(pre_p=0.60, pre_mean=0.01, pre_mdd=0.03)
    verdict = apply_decision_rule(first, second)
    assert verdict["verdict"] == "account_b_confirmed"


def test_account_a_confirmed_when_first_is_a_powered_null_and_second_is_positive():
    first = _arm(pre_p=0.40, pre_mean=0.01, pre_mdd=0.20)  # powered null (mdd < 1.0)
    second = _arm(pre_p=0.02, pre_mean=0.09, pre_mdd=0.30)
    verdict = apply_decision_rule(first, second)
    assert verdict["verdict"] == "account_a_confirmed"


def test_inconclusive_underpowered_when_first_null_but_not_powered():
    first = _arm(pre_p=0.40, pre_mean=0.01, pre_mdd=1.50)  # mdd does not clear internal reference
    second = _arm(pre_p=0.02, pre_mean=0.09, pre_mdd=0.30)
    verdict = apply_decision_rule(first, second)
    assert verdict["verdict"] == "inconclusive_underpowered"


def test_neither_position_significant_despite_adequate_power():
    first = _arm(pre_p=0.40, pre_mean=0.01, pre_mdd=0.20)
    second = _arm(pre_p=0.50, pre_mean=0.01, pre_mdd=0.20)
    verdict = apply_decision_rule(first, second)
    assert verdict["verdict"] == "neither_position_significant_despite_adequate_power"


if __name__ == "__main__":
    # ponytail: no test framework required for a self-check; pytest is the real suite.
    import traceback
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                import inspect
                if "tmp_path" in inspect.signature(fn).parameters:
                    continue  # needs pytest's fixture; covered by the pytest run
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
