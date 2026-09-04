"""Tests for scripts/run_stimulation_timing_and_parameter_structure.py.

Covers the train-to-item overlap geometry, the classifier-triggered
corpus's derivation of stimulated-item status from STIM_ON/STIM_OFF timing
(including the empirically-observed case of a matched pulse train that
lands after its owning item's on-screen window has already ended), the
session-level refusal path for an unpaired STIM_ON/STIM_OFF count, and the
position-interaction estimator (both its exact-recovery behaviour on a
noise-free synthetic dataset and its own trial-count refusal path).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_stimulation_timing_and_parameter_structure import (  # noqa: E402
    MIN_TRIALS_PER_ARM_PER_SUBJECT,
    block_a_neighbor_coverage,
    block_a_session,
    build_trains_closedloop,
    build_trains_openloop,
    fit_subject_interaction,
    group_words_by_list,
    match_train_owner,
    overlaps,
    process_closedloop_session,
)


def make_row(trial_type: str, onset: float, duration: float = 0.0, **kw) -> dict:
    """Builds one event-table row exactly as run_stimulation_timing_and_parameter_structure's
    own read_events() would hand it to a processor: every schema column present as a
    string, plus the numeric _onset/_duration fields read_events() adds."""
    row = {
        "onset": str(onset), "duration": str(duration), "sample": "0",
        "trial_type": trial_type, "response_time": "n/a", "stim_file": "n/a",
        "item_name": "n/a", "serialpos": "-999", "recalled": "0", "list": "1",
        "test": "[0, 0, 0]", "answer": "n/a", "stimulation": "0", "stim_list": "1",
        "stim_duration": "0", "anode_label": "n/a", "cathode_label": "n/a",
        "amplitude": "0.0", "pulse_freq": "0", "pulse_width": "0", "n_pulses": "0",
        "experiment": "FR3", "session": "0", "subject": "R0000X",
    }
    row.update({k: str(v) for k, v in kw.items()})
    row["_onset"] = float(onset)
    row["_duration"] = float(duration)
    return row


class TestOverlapGeometry:
    def test_disjoint(self):
        assert overlaps(0.0, 1.0, 2.0, 3.0) is False

    def test_touching_endpoints_not_overlapping(self):
        # Half-open convention: a pulse train that starts exactly when an
        # item's window ends does not count as landing during it.
        assert overlaps(0.0, 1.0, 1.0, 2.0) is False

    def test_real_overlap(self):
        assert overlaps(0.0, 2.0, 1.0, 3.0) is True

    def test_nested(self):
        assert overlaps(0.0, 5.0, 1.0, 2.0) is True


class TestTrainToItemOverlap:
    """block_a_session's items_per_train count -- the statistic Block A's
    attributability branch is decided from."""

    def _words(self):
        # Four items, 2.5 s apart, each on screen for 1.6 s (matches the
        # measured spacing/duration for both real corpora).
        return [make_row("WORD", onset=o, duration=1.6, list="1", serialpos=str(i + 1))
                for i, o in enumerate([0.0, 2.5, 5.0, 7.5])]

    def test_long_train_spans_two_items(self):
        # A 4.6 s open-loop-style train starting just before item 1 overlaps
        # both item 1 [0, 1.6] is too early -- start it so it truly spans two.
        words = self._words()
        trains = [{"start": 2.4, "end": 2.4 + 4.6}]  # overlaps item[1] and item[2]
        result = block_a_session(words, trains)
        assert result["items_per_train"] == [2]

    def test_short_train_confined_to_one_item(self):
        words = self._words()
        trains = [{"start": 2.9, "end": 3.3}]  # inside item[1]'s [2.5, 4.1] window
        result = block_a_session(words, trains)
        assert result["items_per_train"] == [1]

    def test_train_landing_in_the_gap_overlaps_nothing(self):
        words = self._words()
        trains = [{"start": 9.4, "end": 9.7}]  # after item[3]'s [7.5, 9.1] window,
                                                # before the (nonexistent) next item
        result = block_a_session(words, trains)
        assert result["items_per_train"] == [0]


class TestNeighborCoverage:
    def test_pair_block_design_covers_the_shared_neighbor(self):
        # Two adjacent stim items sharing one long train: each has exactly
        # one neighbour covered (the other stim item), matching the
        # confirmed real ds005489 pattern (mean items-per-train == 2).
        words = [
            make_row("WORD", onset=0.0, duration=1.6, list="1", serialpos="1", stimulation="0"),
            make_row("WORD", onset=2.5, duration=1.6, list="1", serialpos="2", stimulation="1"),
            make_row("WORD", onset=5.0, duration=1.6, list="1", serialpos="3", stimulation="1"),
            make_row("WORD", onset=7.5, duration=1.6, list="1", serialpos="4", stimulation="0"),
        ]
        by_list = group_words_by_list(words)
        trains = [{"start": 2.3, "end": 2.3 + 4.6}]  # overlaps items 2 and 3 only
        cov = block_a_neighbor_coverage(by_list, trains, lambda w: w["stimulation"] == "1")
        assert cov["n_stim_items"] == 2
        assert cov["n_next_covered"] == 1  # the first stim item (serialpos 2)
        assert cov["n_prev_covered"] == 1  # the second stim item (serialpos 3)
        assert cov["n_any_covered"] == 2
        assert cov["n_train_unmatched_to_owning_item"] == 0

    def test_unmatched_stim_item_is_counted_not_silently_dropped(self):
        words = [make_row("WORD", onset=0.0, duration=1.6, list="1", serialpos="1", stimulation="1")]
        by_list = group_words_by_list(words)
        cov = block_a_neighbor_coverage(by_list, trains=[], is_stim_flag=lambda w: w["stimulation"] == "1")
        assert cov["n_stim_items"] == 1
        assert cov["n_train_unmatched_to_owning_item"] == 1
        assert cov["n_any_covered"] == 0


class TestMatchTrainOwner:
    def test_matches_nearest_preceding_word_within_window(self):
        onsets = [0.0, 2.0, 4.0, 6.0]
        assert match_train_owner(2.3, onsets, window_s=2.0) == 1

    def test_outside_window_is_unmatched(self):
        onsets = [0.0, 2.0, 4.0, 6.0]
        assert match_train_owner(100.0, onsets, window_s=2.0) is None

    def test_never_matches_a_following_word(self):
        # train strictly before every word onset
        onsets = [5.0, 7.0]
        assert match_train_owner(1.0, onsets, window_s=2.0) is None


class TestBuildTrains:
    def test_openloop_train_from_stim_on_duration(self):
        rows = [make_row("STIM_ON", onset=10.0, duration=4.6, amplitude="1000.0",
                         pulse_freq="50", pulse_width="300", stim_duration="4600",
                         anode_label="A1", cathode_label="A2")]
        trains = build_trains_openloop(rows)
        assert len(trains) == 1
        assert trains[0]["start"] == pytest.approx(10.0)
        assert trains[0]["end"] == pytest.approx(14.6)
        assert trains[0]["amplitude"] == pytest.approx(1000.0)

    def test_openloop_zero_duration_row_is_dropped(self):
        rows = [make_row("STIM_ON", onset=10.0, duration=0.0)]
        assert build_trains_openloop(rows) == []

    def test_closedloop_pairs_on_off_sequentially(self):
        rows = [
            make_row("STIM_ON", onset=1.0, amplitude="500.0"),
            make_row("STIM_OFF", onset=1.5),
            make_row("STIM_ON", onset=3.0, amplitude="750.0"),
            make_row("STIM_OFF", onset=3.4),
        ]
        trains, mismatch = build_trains_closedloop(rows)
        assert mismatch == 0
        assert len(trains) == 2
        assert (trains[0]["start"], trains[0]["end"]) == pytest.approx((1.0, 1.5))
        assert (trains[1]["start"], trains[1]["end"]) == pytest.approx((3.0, 3.4))

    def test_closedloop_count_mismatch_is_flagged(self):
        rows = [
            make_row("STIM_ON", onset=1.0),
            make_row("STIM_ON", onset=3.0),
            make_row("STIM_OFF", onset=1.5),
        ]
        trains, mismatch = build_trains_closedloop(rows)
        assert trains == []
        assert mismatch == 1


class TestClosedLoopStimulationDerivation:
    """process_closedloop_session end to end: WORD rows' own `stimulation`
    field is always '0' in this corpus (as in the real data), so stimulated-
    item status has to be derived purely from STIM_ON/STIM_OFF timing
    against item onsets -- this is the derivation checked below."""

    def _rows(self):
        words = [
            make_row("WORD", onset=0.0, duration=1.6, list="1", serialpos="1", recalled="1"),
            make_row("WORD", onset=2.5, duration=1.6, list="1", serialpos="2", recalled="0"),
            make_row("WORD", onset=5.0, duration=1.6, list="1", serialpos="3", recalled="1"),
            make_row("WORD", onset=7.5, duration=1.6, list="1", serialpos="4", recalled="0"),
        ]
        # Train A: owned by item[1] (onset 2.5, within the 2.0 s match window of
        # 2.9) and its interval [2.9, 3.3] also overlaps item[1]'s own on-screen
        # window [2.5, 4.1] -- the clean case.
        # Train B: owned by item[3] (onset 7.5, match window reaches 9.4) but its
        # interval [9.4, 9.7] no longer overlaps item[3]'s on-screen window
        # [7.5, 9.1] -- the case actually found in the real ds005557 data, where a
        # matched pulse lands after the word has already left the screen.
        stim = [
            make_row("STIM_ON", onset=2.9, amplitude="500.0", anode_label="L1", cathode_label="L2"),
            make_row("STIM_OFF", onset=3.3),
            make_row("STIM_ON", onset=9.4, amplitude="500.0", anode_label="L1", cathode_label="L2"),
            make_row("STIM_OFF", onset=9.7),
        ]
        return words + stim

    def test_derives_two_stimulated_items(self):
        record = process_closedloop_session(self._rows())
        assert record["status"] == "included"
        assert record["n_stim_items_derived"] == 2

    def test_items_per_train_reflects_the_late_pulse(self):
        record = process_closedloop_session(self._rows())
        assert sorted(record["items_per_train"]) == [0, 1]

    def test_neighbor_coverage_flags_the_unmatched_owning_item(self):
        record = process_closedloop_session(self._rows())
        cov = record["neighbor_coverage"]
        assert cov["n_stim_items"] == 2
        assert cov["n_train_unmatched_to_owning_item"] == 1
        assert cov["n_any_covered"] == 0  # items are 2.5 s apart, trains are ~0.4 s long

    def test_amplitude_captured_from_stim_on_row_not_word_row(self):
        record = process_closedloop_session(self._rows())
        assert record["amplitudes"] == [pytest.approx(500.0), pytest.approx(500.0)]


class TestClosedLoopRefusalPath:
    def test_stim_on_off_count_mismatch_is_excluded_not_guessed_at(self):
        rows = [
            make_row("WORD", onset=0.0, duration=1.6, list="1", serialpos="1"),
            make_row("STIM_ON", onset=0.5),
            make_row("STIM_ON", onset=5.0),
            make_row("STIM_OFF", onset=0.9),
        ]
        record = process_closedloop_session(rows)
        assert record["status"] == "excluded"
        assert "mismatch" in record["reason"]


class TestPositionInteractionEstimator:
    def _synthetic_items(self, stim_main=0.1, position_main=-0.02, interaction=0.05, n_reps=3):
        """Noise-free items: 'recalled' is set to the exact linear-model value
        (not a stochastic draw), so OLS must recover the planted coefficients
        to numerical precision -- an exact-recovery test of the estimator
        itself, not of a fitting procedure's robustness to noise."""
        items = []
        for pos in range(1, 13):
            pos_c = pos - 6.5
            for stim in (0, 1):
                value = 0.5 + stim_main * stim + position_main * pos_c + interaction * stim * pos_c
                for _ in range(n_reps):
                    items.append({"stim": stim, "serialpos": pos, "recalled": value})
        return items

    def test_recovers_planted_coefficients_exactly(self):
        items = self._synthetic_items(stim_main=0.1, position_main=-0.02, interaction=0.05)
        fit = fit_subject_interaction(items)
        assert fit is not None
        assert fit["stim_main_effect"] == pytest.approx(0.1, abs=1e-9)
        assert fit["position_main_effect"] == pytest.approx(-0.02, abs=1e-9)
        assert fit["interaction"] == pytest.approx(0.05, abs=1e-9)
        assert fit["n_stim"] == 36
        assert fit["n_control"] == 36

    def test_zero_interaction_recovered_as_zero(self):
        items = self._synthetic_items(interaction=0.0)
        fit = fit_subject_interaction(items)
        assert fit["interaction"] == pytest.approx(0.0, abs=1e-9)

    def test_refuses_below_min_trials_per_arm(self):
        n = MIN_TRIALS_PER_ARM_PER_SUBJECT - 1
        items = ([{"stim": 1, "serialpos": p % 12 + 1, "recalled": 1} for p in range(n)] +
                 [{"stim": 0, "serialpos": p % 12 + 1, "recalled": 0} for p in range(20)])
        assert fit_subject_interaction(items) is None

    def test_refuses_when_stim_arm_has_no_position_variance(self):
        items = ([{"stim": 1, "serialpos": 6, "recalled": 1} for _ in range(20)] +
                 [{"stim": 0, "serialpos": p % 12 + 1, "recalled": 0} for p in range(20)])
        assert fit_subject_interaction(items) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
