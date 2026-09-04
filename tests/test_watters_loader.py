"""Checks on the multi-object macaque corpus loader for the two joins that
would otherwise fail silently: the trial-number join between the per-unit
spike cache and the behaviour table, and the alignment of the delay window
inside each cached trial's 10 ms bin vector.

Both are re-derived here from the raw pickle and CSV rather than from the
loader's own bookkeeping, so a loader that consistently mis-joins would still
fail these.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_sessions import (  # noqa: E402
    WATTERS_CACHE_ORIGIN_BEFORE_STIMULUS_S,
    WATTERS_DELAY_WINDOW_S,
    WATTERS_RAW_BIN_MS,
    load_watters_session,
    watters_behaviour,
    watters_directories,
    watters_session_dates,
)

# The fixed-offset delay window an earlier analysis of this corpus in this
# repository arrived at independently (scripts/run_watters_item_count_drift.py):
# raw bins 120:220 of each cached trial. The per-trial timestamp cut must agree
# with it on the modal trial, or one of the two conventions is wrong.
FIXED_DELAY_RAW_BINS = (120, 220)


def _root() -> Path:
    value = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not value:
        pytest.skip("WM_DYNAMICS_DATA_ROOT is not set")
    return Path(value)


@pytest.fixture(scope="module")
def session():
    root = _root()
    spikes_dir, _ = watters_directories(root)
    if not spikes_dir.is_dir():
        pytest.skip("multi-object macaque spike cache is not staged locally")
    behaviour = watters_behaviour(root)
    for animal, date, _variant in watters_session_dates(root):
        loaded = load_watters_session(root, animal, date, behaviour)
        if loaded["status"] == "loaded":
            return root, behaviour, loaded
    pytest.skip("no session of the multi-object macaque corpus loaded")


def test_trial_number_join_is_by_trial_number_not_position(session):
    """Re-reads one unit's pickle pair directly and rebuilds the loader's own
    row for several trials. A loader that zipped the two lists positionally,
    or that indexed the cache by behaviour-table row order, would disagree
    here on any session where a unit's trial list has gaps -- which is every
    session in this corpus, since units come and go mid-session."""
    root, _behaviour, loaded = session
    spikes_dir, _ = watters_directories(root)
    directory = spikes_dir / loaded["animal"] / loaded["session_date"]
    factor = int(round(loaded["bin_ms"] / WATTERS_RAW_BIN_MS))
    n_raw = int(round(WATTERS_DELAY_WINDOW_S * 1000.0 / WATTERS_RAW_BIN_MS))

    rng = np.random.default_rng(0)
    unit_indices = rng.choice(loaded["n_units"], size=min(3, loaded["n_units"]), replace=False)
    trial_indices = rng.choice(len(loaded["trial_num"]), size=min(5, len(loaded["trial_num"])), replace=False)

    for u in unit_indices:
        probe, quality = loaded["unit_probe"][u], loaded["unit_quality"][u]
        tier = directory / probe / quality
        candidates = sorted(tier.glob("*_trials.pkl"))
        # The loader keeps only the units spanning every analysed trial, so a
        # positional index into the tier's files is not the loader's unit index;
        # the unit is identified by its own trial list containing every trial.
        for path in candidates:
            unit_id = path.name[: -len("_trials.pkl")]
            with open(path, "rb") as handle:
                trials = np.asarray(pickle.load(handle), dtype=int)
            if not set(loaded["trial_num"].tolist()).issubset(set(trials.tolist())):
                continue
            with open(tier / f"{unit_id}_spike_counts.pkl", "rb") as handle:
                per_trial = pickle.load(handle)
            row_for_trial = {int(t): i for i, t in enumerate(trials)}
            rebuilt = np.stack([
                np.asarray(per_trial[row_for_trial[int(loaded["trial_num"][t])]]
                           [loaded["delay_onset_raw_bin"][t]:loaded["delay_onset_raw_bin"][t] + n_raw],
                           dtype=float).reshape(-1, factor).sum(axis=1)
                for t in trial_indices
            ])
            matches = [np.allclose(loaded["counts"][trial_indices, v], rebuilt) for v in range(loaded["n_units"])]
            assert any(matches), f"no loader column reproduces unit {probe}/{quality}/{unit_id}"
            break


def test_delay_window_alignment_matches_the_fixed_bin_convention(session):
    """The per-trial timestamp cut must land on the fixed 120:220 raw-bin
    window for the modal trial, and must never run past cue onset."""
    _root_path, behaviour, loaded = session
    assert int(np.median(loaded["delay_onset_raw_bin"])) == FIXED_DELAY_RAW_BINS[0]

    rows = behaviour.loc[(loaded["animal"], loaded["session_date"])].loc[loaded["trial_num"].tolist()]
    origin = rows.time_stimulus_onset.to_numpy(float) - WATTERS_CACHE_ORIGIN_BEFORE_STIMULUS_S
    n_raw = int(round(WATTERS_DELAY_WINDOW_S * 1000.0 / WATTERS_RAW_BIN_MS))
    tolerance_s = WATTERS_RAW_BIN_MS / 2000.0
    window_end_s = origin + (loaded["delay_onset_raw_bin"] + n_raw) * WATTERS_RAW_BIN_MS / 1000.0
    assert np.all(window_end_s <= rows.time_cue_onset.to_numpy(float) + tolerance_s)

    window_start_s = origin + loaded["delay_onset_raw_bin"] * WATTERS_RAW_BIN_MS / 1000.0
    assert np.allclose(window_start_s, rows.time_delay_onset.to_numpy(float), atol=WATTERS_RAW_BIN_MS / 2000.0)


def test_tensor_shape_and_labels_line_up(session):
    _root_path, _behaviour, loaded = session
    n_bins = int(round(WATTERS_DELAY_WINDOW_S * 1000.0 / loaded["bin_ms"]))
    assert loaded["counts"].shape == (len(loaded["trial_num"]), loaded["n_units"], n_bins)
    for field in ("num_objects", "correct", "cued_theta", "reaction_time_ms", "delay_onset_raw_bin"):
        assert len(loaded[field]) == len(loaded["trial_num"])
    assert len(loaded["unit_quality"]) == loaded["n_units"]
    assert np.all(loaded["counts"] >= 0)
    assert np.all(np.isfinite(loaded["cued_theta"]))
