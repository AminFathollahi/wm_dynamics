"""Tests for scripts/run_ram_stimulation_state_excursion.py's own logic:
event-table word/train geometry, sham construction, window cut-short rules,
the cross-fit distance-to-reference pipeline, and the artifact/recovery bin
bookkeeping (excursion, time to return). Synthetic throughout -- no
data-root dependency, except the one end-to-end epoching test, which
monkeypatches mne.io.read_raw_edf with an in-memory synthetic recording."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_ram_stimulation_state_excursion import (  # noqa: E402
    ARTIFACT_MARGIN_S, BIN_S, EXCURSION_BIN_COUNT, N_FOLDS, PRE_BIN_COUNT,
    _build_events, _distance_pipeline, _epoch_events, _event_window, _fold_assignment,
    _per_event_excursion, _session_geometry, _spread_p95, _time_course, _word_key,
    artifact_bin_count,
)


def _events_row(onset, duration, trial_type, list_id="1", serialpos="-999", **extra):
    row = {"onset": str(onset), "duration": str(duration), "trial_type": trial_type,
           "list": list_id, "serialpos": serialpos}
    row.update({k: str(v) for k, v in extra.items()})
    return row


def _with_numeric(rows):
    for r in rows:
        r["_onset"] = float(r["onset"])
        r["_duration"] = float(r["duration"])
    return rows


# ── _word_key ────────────────────────────────────────────────────────────

def test_word_key_parses_list_and_serialpos():
    assert _word_key({"list": "3", "serialpos": "5"}) == (3, 5)


def test_word_key_none_for_missing_list():
    assert _word_key({"list": "-1", "serialpos": "5"}) is None


# ── session geometry: open-loop and closed-loop train ownership ────────────

def test_openloop_geometry_matches_own_enclosing_train():
    events = _with_numeric([
        _events_row(99.8, 1.0, "STIM_ON", amplitude="500"),
        _events_row(100.0, 1.6, "WORD", serialpos="1", stimulation="1", stim_list="1"),
        _events_row(105.0, 1.6, "WORD", serialpos="2", stimulation="0", stim_list="0"),
    ])
    geometry = _session_geometry(events, is_openloop=True)
    assert geometry is not None
    assert geometry["onset_by_key"][(1, 1)] == 100.0
    assert geometry["train_by_key"][(1, 1)] == (99.8, 99.8 + 1.0)
    assert geometry["train_starts_sorted"] == [99.8]


def test_closedloop_geometry_owns_train_by_nearest_preceding_word():
    events = _with_numeric([
        _events_row(100.0, 1.6, "WORD", serialpos="1", stim_list="1"),
        _events_row(100.2, 0.0, "STIM_ON", amplitude="500"),
        _events_row(103.0, 0.0, "STIM_OFF"),
    ])
    geometry = _session_geometry(events, is_openloop=False)
    assert geometry is not None
    assert geometry["train_by_key"][(1, 1)] == (100.2, 103.0)


def test_closedloop_geometry_flags_stim_on_off_count_mismatch():
    events = _with_numeric([
        _events_row(100.0, 1.6, "WORD", serialpos="1"),
        _events_row(100.2, 0.0, "STIM_ON", amplitude="500"),
    ])
    assert _session_geometry(events, is_openloop=False) is None


# ── sham construction ───────────────────────────────────────────────────────

def _openloop_session_with_shams():
    """Six stimulated words (list 1, serialpos 1-6, at or above
    MIN_REAL_EVENTS_FOR_LAG so a session-median lag/duration is computable),
    each with its own 1 s train exactly 0.2 s before word onset, and one
    fully non-stimulated list (list 2) sharing serial positions 1-6 plus one
    extra position (7) no stimulated word ever occupies."""
    rows = []
    for sp in range(1, 7):
        onset = 100.0 + 3.0 * sp
        rows.append(_events_row(onset - 0.2, 1.0, "STIM_ON"))
        rows.append(_events_row(onset, 1.6, "WORD", list_id="1", serialpos=str(sp),
                                stimulation="1", stim_list="1", recalled=str(sp % 2)))
    for sp in range(1, 8):
        onset = 300.0 + 3.0 * sp
        rows.append(_events_row(onset, 1.6, "WORD", list_id="2", serialpos=str(sp),
                                stimulation="0", stim_list="0", recalled=str(sp % 2)))
    return _with_numeric(rows)


def test_sham_events_drawn_only_from_non_stimulated_lists_at_matched_serialpositions():
    events = _openloop_session_with_shams()
    geometry = _session_geometry(events, is_openloop=True)
    built = _build_events(geometry)
    assert built is not None
    real_events, sham_events = built
    assert {e["key"] for e in real_events} == {(1, sp) for sp in range(1, 7)}
    sham_keys = {e["key"] for e in sham_events}
    assert sham_keys == {(2, sp) for sp in range(1, 7)}  # serialpos 7 has no matching stimulated word


def test_sham_timing_uses_session_median_lag_and_duration():
    events = _openloop_session_with_shams()
    geometry = _session_geometry(events, is_openloop=True)
    real_events, sham_events = _build_events(geometry)
    median_lag = sham_events[0]["lag_and_duration_source"]["median_lag_s"]
    median_duration = sham_events[0]["lag_and_duration_source"]["median_duration_s"]
    assert abs(median_lag - (-0.2)) < 1e-9
    assert abs(median_duration - 1.0) < 1e-9
    sham_21 = next(e for e in sham_events if e["key"] == (2, 1))
    word_onset_sham_21 = 300.0 + 3.0 * 1
    assert abs(sham_21["train_start_abs"] - (word_onset_sham_21 + median_lag)) < 1e-9
    assert abs(sham_21["train_duration"] - median_duration) < 1e-9


def test_two_word_train_yields_one_event_anchored_on_the_first_word():
    """One train overlapping two consecutive words' on-screen windows (the
    corpus's own documented 'one train spans ~2 items' pattern) must
    collapse to a single real event, anchored at the first word's own
    onset -- not two events, and not the second word's much larger
    negative lag."""
    rows = []
    for sp in range(1, 6):  # five ordinary, single-word trains
        onset = 100.0 + 20.0 * sp
        rows.append(_events_row(onset - 0.2, 1.0, "STIM_ON"))
        rows.append(_events_row(onset, 1.6, "WORD", list_id="1", serialpos=str(sp),
                                stimulation="1", stim_list="1", recalled="1"))
    word6_onset, word7_onset = 300.0, 302.0  # a pair sharing one long train
    rows.append(_events_row(word6_onset - 0.2, 4.0, "STIM_ON"))
    rows.append(_events_row(word6_onset, 1.6, "WORD", list_id="1", serialpos="6",
                            stimulation="1", stim_list="1", recalled="1"))
    rows.append(_events_row(word7_onset, 1.6, "WORD", list_id="1", serialpos="7",
                            stimulation="1", stim_list="1", recalled="0"))
    events = _with_numeric(rows)
    geometry = _session_geometry(events, is_openloop=True)
    real_events, _ = _build_events(geometry)

    assert len(real_events) == 6  # 5 singles + 1 deduplicated pair, never 7
    pair_event = next(e for e in real_events if e["key"] == (1, 6))
    assert pair_event["member_keys"] == [(1, 6), (1, 7)]
    assert abs(pair_event["word_onset_abs"] - word6_onset) < 1e-9
    lag = pair_event["train_start_abs"] - pair_event["word_onset_abs"]
    assert abs(lag - (-0.2)) < 1e-9  # first-word lag, not the second word's much larger one
    assert not any(e["key"] == (1, 7) for e in real_events)  # word 7 never gets its own event


# ── window cut-short: next train, list end ──────────────────────────────────

def test_event_window_cut_short_by_next_train_onset():
    event = {"train_start_abs": 100.0, "train_duration": 4.6, "list_id": 1}
    geometry = {"train_starts_sorted": [100.0, 106.0], "list_end_by_list": {1: 1000.0}}
    window_start, n_bins = _event_window(event, geometry)
    assert window_start == 99.0
    assert abs(window_start + n_bins * BIN_S - 106.0) < 1e-9


def test_event_window_cut_short_by_list_end():
    event = {"train_start_abs": 100.0, "train_duration": 4.6, "list_id": 1}
    geometry = {"train_starts_sorted": [100.0], "list_end_by_list": {1: 103.0}}
    window_start, n_bins = _event_window(event, geometry)
    assert window_start == 99.0
    assert abs(window_start + n_bins * BIN_S - 103.0) < 1e-9


def test_artifact_bin_count_covers_train_plus_margin():
    event = {"train_duration": 1.0}
    expected = PRE_BIN_COUNT + round((1.0 + ARTIFACT_MARGIN_S) / BIN_S)
    assert artifact_bin_count(event) == expected


# ── cross-fit distance pipeline recovers a planted displacement ────────────

def test_distance_pipeline_recovers_planted_displacement():
    rng = np.random.default_rng(0)
    n_sham, n_stim, n_feat, n_bins, magnitude = 40, 10, 6, 15, 8.0
    n_events = n_sham + n_stim
    state = rng.normal(scale=0.1, size=(n_events, n_bins, n_feat))
    sham_idx = np.arange(n_sham)
    stim_idx = np.arange(n_sham, n_events)
    state[stim_idx, :, 0] += magnitude

    fold_of = _fold_assignment(sham_idx, rng)
    assert set(fold_of.values()) <= set(range(N_FOLDS))

    dist = _distance_pipeline(state, sham_idx, stim_idx, fold_of, pca_k=None)
    assert dist is not None
    sham_mean_dist = np.nanmean(dist[sham_idx])
    stim_mean_dist = np.nanmean(dist[stim_idx])
    assert sham_mean_dist < 5.0 * np.sqrt(n_feat)
    assert stim_mean_dist > 10.0 * sham_mean_dist


def test_distance_pipeline_tolerates_variable_length_nan_padded_windows():
    """Different events have different valid window lengths (the whole point
    of the train-locked, cut-short-at-the-next-train design) -- later bins
    are NaN for a shorter event, and neither the scaling fit nor the
    reference may propagate NaN into bins other events do have."""
    rng = np.random.default_rng(2)
    n_sham, n_feat, n_bins = 20, 4, 10
    state = rng.normal(scale=0.1, size=(n_sham, n_bins, n_feat))
    state[5:, 7:, :] = np.nan  # most sham events cut short before bin 7
    sham_idx = np.arange(n_sham)
    stim_idx = np.array([], dtype=int)
    fold_of = _fold_assignment(sham_idx, rng)
    dist = _distance_pipeline(state, sham_idx, stim_idx, fold_of, pca_k=None)
    assert dist is not None
    assert np.isfinite(dist[:5, 7:]).all()   # the long events still get a real distance at late bins
    assert np.isnan(dist[5:, 7:]).all()      # the short events have no data there at all


# ── excursion / artifact exclusion / time-to-return ─────────────────────────

def test_per_event_excursion_excludes_artifact_bins_and_finds_return():
    n_bins = 40
    events = [{"train_duration": 1.0}]
    start = artifact_bin_count(events[0])
    dist = np.full((1, n_bins), 0.05)
    dist[0, :start] = 50.0  # pre + artifact bins: large, must never enter the excursion or the return search
    spread_p95 = np.full(n_bins, 0.1)
    stim_idx = np.array([0])

    per_event = _per_event_excursion(dist, events, stim_idx, spread_p95)
    record = per_event[0]
    assert record["raw_excursion"] < 1.0
    assert record["time_to_return_status"] == 1
    assert record["time_to_return_s"] == 0.0  # first recovery bin already within spread


def test_per_event_excursion_right_censors_when_never_within_spread():
    n_bins = 40
    events = [{"train_duration": 1.0}]
    start = artifact_bin_count(events[0])
    dist = np.full((1, n_bins), 100.0)  # never returns, including during recovery
    spread_p95 = np.full(n_bins, 0.1)
    stim_idx = np.array([0])

    per_event = _per_event_excursion(dist, events, stim_idx, spread_p95)
    record = per_event[0]
    assert record["time_to_return_status"] == 0
    assert record["raw_excursion"] > 10.0
    assert record["time_to_return_s"] == float((n_bins - 1 - start) * BIN_S)


def test_time_course_aligns_by_recovery_bin_ordinal_not_absolute_index():
    """Two events with different train durations (and so different
    artifact_bin_count) must still land in the SAME time_course row for
    'the first bin after the margin', since recovery time is offset-relative."""
    n_bins = 40
    events = [{"train_duration": 1.0}, {"train_duration": 2.0}]
    dist = np.full((2, n_bins), np.nan)
    start0, start1 = artifact_bin_count(events[0]), artifact_bin_count(events[1])
    dist[0, start0] = 3.0
    dist[1, start1] = 7.0
    stim_idx = np.array([0, 1])
    sham_idx = np.array([], dtype=int)
    course = _time_course(dist, events, stim_idx, sham_idx)
    first_row = course[0]
    assert first_row["recovery_bin_index"] == 0
    assert first_row["n_stimulated"] == 2
    assert abs(first_row["stimulated_mean"] - 5.0) < 1e-9


def test_spread_p95_uses_only_sham_events():
    n_bins = 5
    dist = np.zeros((3, n_bins))
    dist[0] = 1.0   # a stim event with a huge distance
    dist[1] = 0.1
    dist[2] = 0.2
    sham_idx = np.array([1, 2])
    spread = _spread_p95(dist, sham_idx)
    assert np.all(spread < 1.0)  # the stim outlier never enters the sham spread


# ── end-to-end epoching on a synthetic continuous recording ────────────────

def test_epoch_events_end_to_end_on_synthetic_continuous_recording(monkeypatch):
    import mne

    srate = 500.0
    ch_names = ["A1-A2", "B1-B2", "C1-C2"]
    rng = np.random.default_rng(3)
    n_samples = int(20.0 * srate)
    data = rng.normal(scale=1e-5, size=(len(ch_names), n_samples))

    word_onset, lag, train_duration = 8.0, 0.2, 1.0
    train_start = word_onset + lag
    i0, i1 = int(train_start * srate), int((train_start + train_duration) * srate)
    data[0, i0:i1] += 5e-4  # artifact deflection on one channel during the train

    info = mne.create_info(ch_names, srate, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *a, **k: raw)

    events = [{"key": (1, 1), "kind": "stim", "word_onset_abs": word_onset, "train_start_abs": train_start,
              "train_duration": train_duration, "list_id": 1}]
    geometry = {"train_starts_sorted": [train_start], "list_end_by_list": {1: 20.0}}

    out = _epoch_events(Path("fake.edf"), events, geometry)
    assert out is not None
    band_state, ch_names_out = out
    assert ch_names_out == ch_names
    assert band_state.shape[0] == 1
    n_bins_event = band_state.shape[1]
    assert n_bins_event >= artifact_bin_count(events[0]) + EXCURSION_BIN_COUNT
    assert np.isfinite(band_state[0]).all()


if __name__ == "__main__":
    import inspect

    module = sys.modules[__name__]
    for name, fn in sorted(inspect.getmembers(module, inspect.isfunction)):
        if name.startswith("test_"):
            fn() if "monkeypatch" not in inspect.signature(fn).parameters else print(f"skip {name} (needs pytest)")
    print("all non-monkeypatch tests passed")
