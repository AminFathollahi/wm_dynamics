#!/usr/bin/env python3
"""Channel-space band-power distance to an attractor and recovery after
encoding-period electrical stimulation, in the two human intracranial
free-recall corpora.

State: a channels x bands log-power vector per 0.1 s bin (theta, alpha,
beta, gamma, high-gamma -- src/preprocessing.py's own BAND_DEFINITIONS),
epoched from the continuous recording relative to each stimulation train:
train onset - 1.0 s through train offset + 3.0 s, cut short at the next
train's own onset or at the end of that word's list, whichever comes first.
Bins from train onset through train offset + 0.3 s cannot be separated from
the electrical artifact and are flagged; they are never used for recovery.

Sham events give the non-stimulated comparison: for every non-stimulated
list (BIDS field stim_list == 0), every word at a serial position that some
real stimulated word in the same session also occupies gets a synthetic
train, onset = that word's own onset plus the session's own median
train-onset lag (real train start minus its word's onset), offset = that
onset plus the session's own median train duration -- both measured
per-session from the session's own real stimulated words, separately for
each corpus, because the closed-loop classifier's own lag and pulse
duration are read from its events and differ from the open-loop design.

Reference: the mean, per bin, over the session's own sham pool, cross-fit
so a held-out sham never sees its own contribution (5-fold, block-assigned
after a session-local shuffle) while a real stimulated event -- never a
member of the sham pool -- is scored against the reference built from the
full sham pool. A parallel version projects onto a PCA basis fit on the
training fold alone, at the rank selected by held-out probabilistic-PCA
score (sklearn PCA(k).fit(train).score(test), argmax over k).

Distance is plain Euclidean, per bin. Excursion = the mean distance over
the first 1.0 s of recovery bins (train offset + 0.3 s onward), reported
raw and corrected by that same event's own pre-onset distance (the 1.0 s of
bins before train onset). Time to return = the first recovery bin whose
distance falls within the 95th percentile of the held-out sham pool's
distances at that same bin, right-censored at the end of the available
window.

alagapan_phase_stimulation is not analysed here: every trial in its
stimulation-condition recordings is stimulated (continuous phase-locked
delivery through the whole retention period, one condition per session, not
interleaved with unstimulated trials); the only unstimulated comparison is
a separate baseline SESSION, not a same-session, same-trial-structure
counterfactual, so it has no stimulated/non-stimulated trial-level contrast
with a post-stimulation recovery window for this design to use.

Outputs:
  results/ram_stimulation_state_excursion.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_ram_stimulation_state_excursion.py [--smoke N]
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from project_config import data_root  # noqa: E402
from provenance import canonical_json, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, paired_sign_flip_test,
    partial_correlation_permutation_test, stable_seed,
)
from subspace_identity import block_folds  # noqa: E402
from preprocessing import band_power, line_noise_notch  # noqa: E402
from stimulation_response_estimator import cluster_bootstrap_pooled_effect  # noqa: E402

from run_ram_openloop_pipeline import BIN_S, DATA as OPENLOOP_DATA  # noqa: E402
from run_ram_closedloop_pipeline import DATA as CLOSEDLOOP_DATA  # noqa: E402
from run_human_stimulation_component_response import channel_condition_masks, load_corpus  # noqa: E402
from run_stimulation_timing_and_parameter_structure import (  # noqa: E402
    CLOSEDLOOP_OWNER_MATCH_WINDOW_S, build_trains_closedloop, build_trains_openloop,
    match_train_owner, overlaps, read_events,
)
from run_ram_stimulation_response_latency import (  # noqa: E402
    _censored_first_recall, contrast_summary, load_corpus as load_recall_latency_corpus,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "ram_stimulation_state_excursion.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_ram_stimulation_state_excursion"
SCHEMA_VERSION = 4  # v3 double-counted a two-word open-loop train as two events and mixed first/second-word lags

ALL_BANDS = ("theta", "alpha", "beta", "gamma", "hgp")
BAND_SMOOTH_MS = {"theta": 200.0, "alpha": 200.0, "beta": 200.0, "gamma": 100.0, "hgp": 50.0}
WINDOW_PRE_S = 1.0
WINDOW_POST_S = 3.0
ARTIFACT_MARGIN_S = 0.3
EXCURSION_WINDOW_S = 1.0
PRE_BIN_COUNT = int(round(WINDOW_PRE_S / BIN_S))
EXCURSION_BIN_COUNT = int(round(EXCURSION_WINDOW_S / BIN_S))
MIN_REAL_EVENTS_FOR_LAG = 6
MIN_SHAM_FOR_REFERENCE = 2 * 5  # cross-fit needs enough sham per fold
N_FOLDS = 5
PCA_MAX_RANK = 10
ALPHA = 0.05
POWER = 0.80
N_PERM = 10000
N_BOOT = 5000

CORPORA = {
    "ram_ds005489_openloop": {"data_dir": OPENLOOP_DATA, "derive_stim_from_stim_on": False, "is_openloop": True},
    "ram_ds005557_closedloop": {"data_dir": CLOSEDLOOP_DATA, "derive_stim_from_stim_on": True, "is_openloop": False},
}


# ── Checkpointing (per session, schema-versioned) ───────────────────────────

def _checkpoint_path(unit: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_schema_version") != SCHEMA_VERSION:
        return None
    return restore_checkpoint(data["record"])


def save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_schema_version": SCHEMA_VERSION, "record": checkpoint_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ── Event-table word/train geometry ─────────────────────────────────────────

def _safe_float(value) -> float:
    try:
        if value in (None, "", "n/a"):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _word_key(w: dict) -> tuple[int, int] | None:
    list_raw = w.get("list")
    try:
        list_id = int(list_raw) if list_raw not in (None, "", "n/a", "-1") else None
        serialpos = int(w.get("serialpos"))
    except (TypeError, ValueError):
        return None
    if list_id is None:
        return None
    return (list_id, serialpos)


def _session_geometry(events: list[dict], is_openloop: bool) -> dict | None:
    """words_sorted, each word's own onset and own stimulation train (start,
    end) if any, and every delivered train's own onset (real trains only,
    including any pre-task ones -- those never own a word and so never
    enter train_by_key, but still bound a nearby event's recovery window)."""
    words = [e for e in events if e["trial_type"] == "WORD"]
    words_sorted = sorted(words, key=lambda w: w["_onset"])
    onset_by_key: dict[tuple[int, int], float] = {}
    train_by_key: dict[tuple[int, int], tuple[float, float] | None] = {}

    if is_openloop:
        trains = build_trains_openloop(events)
        for w in words_sorted:
            key = _word_key(w)
            if key is None:
                continue
            onset_by_key[key] = w["_onset"]
            if w.get("stimulation") != "1":
                continue
            w0, w1 = w["_onset"], w["_onset"] + w["_duration"]
            enclosing = [t for t in trains if overlaps(t["start"], t["end"], w0, w1)]
            train_by_key[key] = (enclosing[0]["start"], enclosing[0]["end"]) if enclosing else None
    else:
        trains, mismatch = build_trains_closedloop(events)
        if mismatch:
            return None
        word_onsets_sorted = [w["_onset"] for w in words_sorted]
        owner_by_train = [match_train_owner(t["start"], word_onsets_sorted, CLOSEDLOOP_OWNER_MATCH_WINDOW_S)
                          for t in trains]
        train_by_word_index: dict[int, dict] = {}
        for idx, t in zip(owner_by_train, trains):
            if idx is not None:
                train_by_word_index[idx] = t
        for i, w in enumerate(words_sorted):
            key = _word_key(w)
            if key is None:
                continue
            onset_by_key[key] = w["_onset"]
            t = train_by_word_index.get(i)
            train_by_key[key] = (t["start"], t["end"]) if t else None

    train_starts_sorted = sorted(t["start"] for t in trains)
    list_end_by_list: dict[int, float] = {}
    for w in words_sorted:
        key = _word_key(w)
        if key is None:
            continue
        end = w["_onset"] + w["_duration"]
        list_end_by_list[key[0]] = max(list_end_by_list.get(key[0], end), end)

    word_onsets_arr = np.array([w["_onset"] for w in words_sorted])
    word_recalled_arr = np.array([_safe_float(w.get("recalled")) for w in words_sorted])
    return {
        "words_sorted": words_sorted, "onset_by_key": onset_by_key, "train_by_key": train_by_key,
        "train_starts_sorted": train_starts_sorted, "list_end_by_list": list_end_by_list,
        "word_onsets_arr": word_onsets_arr, "word_recalled_arr": word_recalled_arr,
    }


def _next_train_onset_after(train_starts_sorted: list[float], t: float) -> float:
    i = bisect.bisect_right(train_starts_sorted, t)
    return train_starts_sorted[i] if i < len(train_starts_sorted) else float("inf")


def _next_word_recalled_after(word_onsets_arr: np.ndarray, word_recalled_arr: np.ndarray, t: float) -> float | None:
    i = bisect.bisect_right(word_onsets_arr, t)
    if i >= len(word_onsets_arr):
        return None
    value = float(word_recalled_arr[i])
    return value if np.isfinite(value) else None


def _median_lag_and_duration(real_events: list[dict]) -> tuple[float, float, int] | None:
    lags = [e["train_start_abs"] - e["word_onset_abs"] for e in real_events]
    durations = [e["train_duration"] for e in real_events]
    if len(lags) < MIN_REAL_EVENTS_FOR_LAG:
        return None
    return float(np.median(lags)), float(np.median(durations)), len(lags)


def _build_events(geometry: dict) -> tuple[list[dict], list[dict]] | None:
    """Real stimulated events (own resolved train) and sham events (matched
    serial position, non-stimulated list, synthetic train from this
    session's own median lag/duration)."""
    words_sorted = geometry["words_sorted"]
    onset_by_key = geometry["onset_by_key"]
    train_by_key = geometry["train_by_key"]

    # A word counts as "really stimulated" if it owns a resolved train in train_by_key -- not by
    # its own WORD-row "stimulation" field, which run_ram_openloop_pipeline.py's own docstring notes
    # is unreliable (always "0") in the closed-loop corpus. train_by_key is already corpus-aware:
    # _session_geometry only ever populates it for open-loop words whose own "stimulation" field is
    # "1", and for closed-loop words a real STIM_ON/STIM_OFF pair was actually matched to.
    #
    # One open-loop train commonly outlasts a single item and overlaps TWO consecutive words' own
    # on-screen windows (this project's own item_overlap finding); both words then map to the same
    # physical train, and treating them as two events would double-count that train and pull the
    # lag statistic toward the trailing word's own, much larger, negative lag. One event per
    # physical train instead, grouped by the train's own (start, end) tuple and anchored to the
    # FIRST word (earliest onset) it overlaps -- that word's own onset is what "lag" is measured
    # from. Every word the train overlaps is kept as a member, for the recall associations.
    words_by_train: dict[tuple[float, float], list[tuple[int, int]]] = {}
    for w in words_sorted:
        key = _word_key(w)
        if key is None:
            continue
        train = train_by_key.get(key)
        if train is None:
            continue
        words_by_train.setdefault(train, []).append(key)

    real_events = []
    for train, member_keys in words_by_train.items():
        member_keys_sorted = sorted(member_keys, key=lambda k: onset_by_key[k])
        anchor_key = member_keys_sorted[0]
        real_events.append({"key": anchor_key, "member_keys": member_keys_sorted, "kind": "stim",
                            "word_onset_abs": onset_by_key[anchor_key],
                            "train_start_abs": train[0], "train_duration": train[1] - train[0],
                            "list_id": anchor_key[0]})

    lag_duration = _median_lag_and_duration(real_events)
    if lag_duration is None:
        return None
    median_lag, median_duration, n_lag_events = lag_duration
    stim_serialpositions = {e["key"][1] for e in real_events}

    sham_events = []
    for w in words_sorted:
        if w.get("stim_list") == "1" or w.get("stimulation") == "1":
            continue
        key = _word_key(w)
        if key is None or key[1] not in stim_serialpositions:
            continue
        onset = onset_by_key[key]
        train_start = onset + median_lag
        sham_events.append({"key": key, "kind": "sham", "word_onset_abs": onset,
                            "train_start_abs": train_start, "train_duration": median_duration,
                            "list_id": key[0]})

    for e in real_events + sham_events:
        e["lag_and_duration_source"] = {"median_lag_s": median_lag, "median_duration_s": median_duration,
                                        "n_real_events_used": n_lag_events}
    return real_events, sham_events


def _event_window(event: dict, geometry: dict) -> tuple[float, int]:
    """window_start_abs, n_bins available before any recording-length cap."""
    window_start = event["train_start_abs"] - WINDOW_PRE_S
    train_end = event["train_start_abs"] + event["train_duration"]
    ceiling = min(
        train_end + WINDOW_POST_S,
        _next_train_onset_after(geometry["train_starts_sorted"], event["train_start_abs"]),
        geometry["list_end_by_list"].get(event["list_id"], float("inf")),
    )
    n_bins = max(0, int((ceiling - window_start) / BIN_S))
    return window_start, n_bins


def artifact_bin_count(event: dict) -> int:
    return PRE_BIN_COUNT + int(round((event["train_duration"] + ARTIFACT_MARGIN_S) / BIN_S))


# ── Continuous re-epoching, all five bands ──────────────────────────────────

def _epoch_events(edf_path: Path, events: list[dict], geometry: dict) -> tuple[np.ndarray, list[str]] | None:
    """(n_events, max_bins, n_bands, n_ch) log-power, NaN-padded past each
    event's own available window. One raw read per event window; band_power
    (src/preprocessing.py) computes all five bands on the same notched
    segment, mirroring run_ram_openloop_pipeline.build_session_features's
    own per-word convention (notch, band-power Hilbert envelope, bin-average,
    log1p) but anchored to the stimulation train instead of the word."""
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    srate = raw.info["sfreq"]
    ch_names = raw.ch_names
    n_ch = len(ch_names)
    n_samples_total = raw.n_times
    pad = int(0.3 * srate)
    samp_per_bin = int(round(BIN_S * srate))

    windows = []
    for event in events:
        window_start, n_bins = _event_window(event, geometry)
        i0 = int(round(window_start * srate)) - pad
        max_bins_available = max(0, (n_samples_total - 2 * pad - i0) // samp_per_bin) if i0 >= 0 else 0
        n_bins = min(n_bins, int(max_bins_available))
        windows.append((i0, n_bins))

    max_bins = max((n for _, n in windows), default=0)
    if max_bins < PRE_BIN_COUNT + 1:
        return None
    out = np.full((len(events), max_bins, len(ALL_BANDS), n_ch), np.nan, dtype=np.float32)
    for ei, (i0, n_bins) in enumerate(windows):
        if n_bins < PRE_BIN_COUNT + 1:
            continue
        i1 = i0 + n_bins * samp_per_bin + 2 * pad
        raw_seg = raw.get_data(start=i0, stop=i1)
        raw_seg_notched = line_noise_notch(raw_seg.T, srate, fundamental=60.0, n_harmonics=3)
        for bi, band in enumerate(ALL_BANDS):
            p = band_power(raw_seg_notched, band, srate=srate, smooth_ms=BAND_SMOOTH_MS[band])
            p = p[pad:-pad]
            binned = p[: n_bins * samp_per_bin].reshape(n_bins, samp_per_bin, n_ch).mean(axis=1)
            out[ei, :n_bins, bi, :] = np.log1p(np.clip(binned, 0, None))
    return out, ch_names


# ── Cross-fit distance-to-reference, NaN-tolerant (variable window length) ──

def _fold_assignment(idx: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    shuffled = idx[rng.permutation(len(idx))]
    folds = block_folds(len(shuffled), N_FOLDS)
    return {int(i): int(f) for i, f in zip(shuffled, folds)}


def _complete_rows(flat: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.all(np.isfinite(flat), axis=1))


def _select_pca_rank(state: np.ndarray, sham_idx: np.ndarray, fold_of: dict) -> int | None:
    train_idx = np.array([i for i in sham_idx if fold_of[i] != 0])
    test_idx = np.array([i for i in sham_idx if fold_of[i] == 0])
    if len(train_idx) < 8 or len(test_idx) < 4:
        return None
    train_flat = state[train_idx].reshape(-1, state.shape[-1])
    test_flat = state[test_idx].reshape(-1, state.shape[-1])
    mu, sd = np.nanmean(train_flat, axis=0), np.nanstd(train_flat, axis=0) + 1e-8
    train_z, test_z = (train_flat - mu) / sd, (test_flat - mu) / sd
    train_complete, test_complete = train_z[_complete_rows(train_z)], test_z[_complete_rows(test_z)]
    max_k = min(PCA_MAX_RANK, train_complete.shape[1] - 1, train_complete.shape[0] - 2)
    if max_k < 1 or len(test_complete) < 4:
        return None
    best_k, best_score = 1, -np.inf
    for k in range(1, max_k + 1):
        pca = PCA(n_components=k, random_state=0)
        pca.fit(train_complete)
        score = pca.score(test_complete)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _distance_pipeline(state: np.ndarray, sham_idx: np.ndarray, stim_idx: np.ndarray,
                        fold_of: dict, pca_k: int | None) -> np.ndarray | None:
    """Per-event, per-bin Euclidean distance to the cross-fit sham
    reference: real events scored against the reference built from the full
    sham pool; each held-out sham scored against the reference built with
    its own fold excluded. NaN-tolerant throughout (event windows have
    different lengths); PCA is fit only on complete (fully-observed) rows."""
    n_events, n_bins, n_feat = state.shape

    def _fit(train_idx: np.ndarray):
        if len(train_idx) < 4:
            return None
        flat = state[train_idx].reshape(-1, n_feat)
        mu, sd = np.nanmean(flat, axis=0), np.nanstd(flat, axis=0) + 1e-8
        components = None
        if pca_k is not None:
            z = (flat - mu) / sd
            complete = z[_complete_rows(z)]
            if len(complete) < pca_k + 2:
                return None
            pca = PCA(n_components=pca_k, random_state=0)
            pca.fit(complete)
            components = pca.components_
        z_trials = (state[train_idx] - mu) / sd
        if components is not None:
            z_trials = z_trials @ components.T
        with np.errstate(invalid="ignore"):
            ref = np.nanmean(z_trials, axis=0)
        return mu, sd, components, ref

    def _project(idx: np.ndarray, mu, sd, components):
        z = (state[idx] - mu) / sd
        return z @ components.T if components is not None else z

    full = _fit(sham_idx)
    if full is None:
        return None
    mu_full, sd_full, comp_full, ref_full = full
    dist = np.full((n_events, n_bins), np.nan)
    for i in stim_idx:
        z = _project(np.array([i]), mu_full, sd_full, comp_full)[0]
        dist[i] = np.linalg.norm(z - ref_full, axis=-1)

    per_fold = {}
    for i in sham_idx:
        k = fold_of[i]
        if k not in per_fold:
            train_k = np.array([j for j in sham_idx if fold_of[j] != k])
            per_fold[k] = _fit(train_k)
        cached = per_fold[k]
        if cached is None:
            continue
        mu_k, sd_k, comp_k, ref_k = cached
        z = _project(np.array([i]), mu_k, sd_k, comp_k)[0]
        dist[i] = np.linalg.norm(z - ref_k, axis=-1)
    return dist


# ── Time course, excursion, time to return ──────────────────────────────────

def _time_course(dist: np.ndarray, events: list[dict], stim_idx: np.ndarray, sham_idx: np.ndarray) -> list[dict]:
    """Per recovery-bin ORDINAL position j (0 = first bin after train
    offset + margin, regardless of where that falls in absolute bin index
    for this event's own train duration)."""
    by_j: dict[int, dict] = {}
    for group_name, idx_arr in (("stimulated", stim_idx), ("sham", sham_idx)):
        for i in idx_arr:
            i = int(i)
            start = artifact_bin_count(events[i])
            row = dist[i]
            for j, b in enumerate(range(start, row.shape[0])):
                if not np.isfinite(row[b]):
                    continue
                bucket = by_j.setdefault(j, {"stimulated": [], "sham": []})
                bucket[group_name].append(float(row[b]))
    course = []
    for j in sorted(by_j):
        s, c = by_j[j]["stimulated"], by_j[j]["sham"]
        course.append({
            "recovery_bin_index": j, "time_since_offset_s": ARTIFACT_MARGIN_S + j * BIN_S,
            "stimulated_mean": float(np.mean(s)) if s else None, "n_stimulated": len(s),
            "sham_mean": float(np.mean(c)) if c else None, "n_sham": len(c),
        })
    return course


def _per_event_pre_distance(dist: np.ndarray, i: int) -> float | None:
    d = dist[i][:PRE_BIN_COUNT]
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else None


def _spread_p95(dist: np.ndarray, sham_idx: np.ndarray) -> np.ndarray:
    n_bins = dist.shape[1]
    spread = np.full(n_bins, np.nan)
    sham_dist = dist[sham_idx]
    for b in range(n_bins):
        col = sham_dist[:, b]
        col = col[np.isfinite(col)]
        if len(col):
            spread[b] = np.percentile(col, 95)
    return spread


def _per_event_excursion(dist: np.ndarray, events: list[dict], stim_idx: np.ndarray, spread_p95: np.ndarray) -> dict:
    per_event = {}
    for i in stim_idx:
        i = int(i)
        d = dist[i]
        pre_val = _per_event_pre_distance(dist, i)
        start = artifact_bin_count(events[i])
        record = {"pre_onset_distance": pre_val, "raw_excursion": None, "corrected_excursion": None,
                  "n_recovery_bins": max(0, d.shape[0] - start), "time_to_return_s": None,
                  "time_to_return_status": None}
        excursion_vals = d[start:start + EXCURSION_BIN_COUNT]
        excursion_vals = excursion_vals[np.isfinite(excursion_vals)]
        if len(excursion_vals):
            record["raw_excursion"] = float(excursion_vals.mean())
            if pre_val is not None:
                record["corrected_excursion"] = record["raw_excursion"] - pre_val
        recovery_bins = np.arange(start, d.shape[0])
        returned = False
        for j, b in enumerate(recovery_bins):
            if np.isfinite(d[b]) and np.isfinite(spread_p95[b]) and d[b] <= spread_p95[b]:
                record["time_to_return_s"] = float(j * BIN_S)
                record["time_to_return_status"] = 1
                returned = True
                break
        if not returned and len(recovery_bins):
            last_finite = [j for j, b in enumerate(recovery_bins) if np.isfinite(d[b])]
            if last_finite:
                record["time_to_return_s"] = float(last_finite[-1] * BIN_S)
                record["time_to_return_status"] = 0
        per_event[i] = record
    return per_event


# ── Per-session processing ───────────────────────────────────────────────────

def _process_session(rec: dict, is_openloop: bool) -> dict:
    arrays = rec["arrays"]
    ch_names_meta = arrays["ch_names"].tolist()
    channel_mask = channel_condition_masks(
        ch_names_meta, str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"]),
    )["excluding_stimulated_shank"]
    if channel_mask.sum() < 1:
        return {"status": "excluded", "reason": "no_channels_survive_shank_exclusion"}

    ieeg_json = Path(rec["ieeg_json"])
    stem = str(ieeg_json).replace("_ieeg.json", "")
    events_tsv = Path(stem.replace("_acq-bipolar", "") + "_events.tsv")
    edf_path = Path(stem + "_ieeg.edf")
    if not events_tsv.exists() or not edf_path.exists():
        return {"status": "excluded", "reason": "events_or_edf_file_missing"}

    events = read_events(events_tsv)
    geometry = _session_geometry(events, is_openloop)
    if geometry is None:
        return {"status": "excluded", "reason": "stim_on_off_count_mismatch_train_geometry_unavailable"}

    built = _build_events(geometry)
    if built is None:
        return {"status": "excluded", "reason": "fewer_than_min_real_events_with_resolved_train_timing"}
    real_events, sham_events = built
    if len(sham_events) < MIN_SHAM_FOR_REFERENCE:
        return {"status": "excluded", "reason": "too_few_sham_events_for_cross_fit_reference",
                "n_sham_events": len(sham_events)}

    all_events = real_events + sham_events
    epoched = _epoch_events(edf_path, all_events, geometry)
    if epoched is None:
        return {"status": "excluded", "reason": "no_event_reaches_the_minimum_usable_window"}
    band_state, band_ch_names = epoched
    if band_ch_names != ch_names_meta:
        return {"status": "excluded", "reason": "channel_order_mismatch_between_metadata_and_band_reread"}

    n_ch_kept = int(channel_mask.sum())
    state = band_state[:, :, :, channel_mask].reshape(band_state.shape[0], band_state.shape[1], 5 * n_ch_kept)
    finite_event = np.any(np.isfinite(state).reshape(state.shape[0], -1), axis=1)
    stim_idx = np.array([i for i in range(len(real_events)) if finite_event[i]])
    sham_idx = np.array([len(real_events) + i for i in range(len(sham_events)) if finite_event[len(real_events) + i]])
    if len(sham_idx) < MIN_SHAM_FOR_REFERENCE or len(stim_idx) < 4:
        return {"status": "excluded", "reason": "too_few_usable_events_after_windowing",
                "n_stim_usable": int(len(stim_idx)), "n_sham_usable": int(len(sham_idx))}

    rng = np.random.default_rng(stable_seed(f"ram_state_excursion|{rec['corpus']}|{rec['session_key']}"))
    fold_of = _fold_assignment(sham_idx, rng)

    dist_raw = _distance_pipeline(state, sham_idx, stim_idx, fold_of, pca_k=None)
    pca_k = _select_pca_rank(state, sham_idx, fold_of)
    dist_pca = _distance_pipeline(state, sham_idx, stim_idx, fold_of, pca_k=pca_k) if pca_k is not None else None

    recall_by_key = {}
    for lst, sp, rc in zip(arrays["list_number"].tolist(), arrays["serialpos"].tolist(), arrays["recalled"].tolist()):
        recall_by_key[(int(lst), int(sp))] = float(rc)

    def _event_behaviour(i: int, event: dict) -> dict:
        member_recalls = [recall_by_key.get(k) for k in event.get("member_keys", [event["key"]])]
        member_recalls = [r for r in member_recalls if r is not None]
        own_recalled_mean = float(np.mean(member_recalls)) if member_recalls else None
        own_recalled_any = float(any(r == 1.0 for r in member_recalls)) if member_recalls else None
        offset_abs = event["train_start_abs"] + event["train_duration"]
        next_recall = _next_word_recalled_after(geometry["word_onsets_arr"], geometry["word_recalled_arr"], offset_abs)
        return {"own_word_recalled_mean": own_recalled_mean, "own_word_recalled_any": own_recalled_any,
                "next_word_recalled": next_recall, "list_id": event["list_id"]}

    result = {
        "status": "computed",
        "n_stim_events": int(len(stim_idx)), "n_sham_events": int(len(sham_idx)),
        "n_channels_kept": n_ch_kept, "pca_rank_selected": pca_k,
        "median_lag_s": all_events[0]["lag_and_duration_source"]["median_lag_s"],
        "median_duration_s": all_events[0]["lag_and_duration_source"]["median_duration_s"],
        "n_real_events_for_lag_duration": all_events[0]["lag_and_duration_source"]["n_real_events_used"],
    }
    spread_raw = _spread_p95(dist_raw, sham_idx)
    result["time_course_raw"] = _time_course(dist_raw, all_events, stim_idx, sham_idx)
    result["excursion_raw"] = _per_event_excursion(dist_raw, all_events, stim_idx, spread_raw)
    if dist_pca is not None:
        spread_pca = _spread_p95(dist_pca, sham_idx)
        result["time_course_pca"] = _time_course(dist_pca, all_events, stim_idx, sham_idx)
        result["excursion_pca"] = _per_event_excursion(dist_pca, all_events, stim_idx, spread_pca)

    result["stim_behaviour"] = {int(i): _event_behaviour(int(i), all_events[int(i)]) for i in stim_idx}
    result["stim_pre_onset_distance_raw"] = {
        int(i): _per_event_pre_distance(dist_raw, int(i)) for i in stim_idx
    }
    result["sham_pre_onset_distance_raw"] = {
        int(i): _per_event_pre_distance(dist_raw, int(i)) for i in sham_idx
    }
    return result


# ── Behaviour associations ──────────────────────────────────────────────────

def _word_level_recall_association(per_session_records: list[dict]) -> dict:
    """Point-biserial correlation, within session, between corrected
    excursion and (a) recall of the stimulated word(s) that own the event's
    train -- mean recall across both members of a two-word open-loop pair,
    and recalled-if-either -- and (b) the recall of the next word presented
    after that event's train offset; pooled across patients by
    cluster_bootstrap_pooled_effect."""
    out = {}
    for field, label in (("own_word_recalled_mean", "own_word_mean"),
                         ("own_word_recalled_any", "own_word_any"),
                         ("next_word_recalled", "next_word")):
        session_r, subjects = [], []
        for rec in per_session_records:
            if rec.get("status") != "computed":
                continue
            excursion = rec["excursion_raw"]
            pairs = []
            for i, v in excursion.items():
                behaviour = rec["stim_behaviour"].get(i)
                if behaviour is None or v["corrected_excursion"] is None:
                    continue
                y = behaviour.get(field)
                if y is None:
                    continue
                pairs.append((v["corrected_excursion"], y))
            if len(pairs) < 8:
                continue
            e_arr = np.array([p[0] for p in pairs])
            y_arr = np.array([p[1] for p in pairs])
            if np.std(e_arr) == 0 or np.std(y_arr) == 0:
                continue
            rng = np.random.default_rng(stable_seed(f"word_recall_{label}|{rec['session_key']}"))
            corr = partial_correlation_permutation_test(y_arr, e_arr, [], N_PERM, rng)
            if corr.get("status") == "computed":
                session_r.append(corr["r"])
                subjects.append(rec["subject"])
        out[label] = cluster_bootstrap_pooled_effect(np.array(session_r), subjects, f"word_recall_{label}",
                                                      n_perm=N_PERM, n_boot=N_BOOT, alpha=ALPHA, power=POWER) \
            if session_r else {"status": "not_computable"}
    return out


def _list_level_latency_cox_association(per_session_records: list[dict], recall_latency_by_session: dict) -> dict:
    """Session-level Cox regression (statsmodels PHReg) of each list's
    censored time to first recall on that list's own continuous mean
    corrected excursion -- the continuous-predictor counterpart to the
    censored-time construction run_ram_stimulation_response_latency.py
    already uses for its own stim/unstim contrast; its per-list censored
    (time, status) pair is reused unchanged here. Session coefficients are
    pooled across patients by cluster_bootstrap_pooled_effect."""
    from statsmodels.duration.hazard_regression import PHReg

    coefs, subjects = [], []
    for rec in per_session_records:
        if rec.get("status") != "computed":
            continue
        latency_lists = recall_latency_by_session.get(rec["session_key"])
        if not latency_lists:
            continue
        excursion = rec["excursion_raw"]
        by_list: dict[int, list[float]] = {}
        for i, v in excursion.items():
            if v["corrected_excursion"] is None:
                continue
            behaviour = rec["stim_behaviour"].get(i)
            if behaviour is None:
                continue
            by_list.setdefault(int(behaviour["list_id"]), []).append(v["corrected_excursion"])
        times, statuses, xs = [], [], []
        for list_no, vals in by_list.items():
            entry = latency_lists.get(list_no)
            if entry is None or not vals:
                continue
            pair = _censored_first_recall(entry)
            if pair is None:
                continue
            times.append(pair[0])
            statuses.append(pair[1])
            xs.append(float(np.mean(vals)))
        if len(times) < 6 or np.std(xs) == 0:
            continue
        try:
            model = PHReg(np.array(times), np.array(xs).reshape(-1, 1), status=np.array(statuses))
            fit = model.fit(disp=False)
        except Exception:  # noqa: BLE001 -- a single session's Cox fit must not crash the run
            continue
        coef = float(fit.params[0])
        if not np.isfinite(coef):
            continue
        coefs.append(coef)
        subjects.append(rec["subject"])
    if not coefs:
        return {"status": "not_computable"}
    return cluster_bootstrap_pooled_effect(np.array(coefs), subjects, "list_excursion_cox_first_recall_latency",
                                            n_perm=N_PERM, n_boot=N_BOOT, alpha=ALPHA, power=POWER)


def _closedloop_pre_stim_distance(per_session_records: list[dict]) -> dict:
    session_diff, subjects = [], []
    for rec in per_session_records:
        if rec.get("status") != "computed":
            continue
        stim_vals = [v for v in rec["stim_pre_onset_distance_raw"].values() if v is not None]
        sham_vals = [v for v in rec["sham_pre_onset_distance_raw"].values() if v is not None]
        if len(stim_vals) < 4 or len(sham_vals) < 4:
            continue
        session_diff.append(float(np.mean(stim_vals) - np.mean(sham_vals)))
        subjects.append(rec["subject"])
    if not session_diff:
        return {"status": "not_computable"}
    return cluster_bootstrap_pooled_effect(np.array(session_diff), subjects, "closedloop_pre_stim_distance",
                                            n_perm=N_PERM, n_boot=N_BOOT, alpha=ALPHA, power=POWER)


def _closedloop_excursion_covariate(per_session_records: list[dict]) -> dict:
    slopes, subjects = [], []
    for rec in per_session_records:
        if rec.get("status") != "computed":
            continue
        excursion = rec["excursion_raw"]
        pairs = [(v["pre_onset_distance"], v["raw_excursion"]) for v in excursion.values()
                 if v["pre_onset_distance"] is not None and v["raw_excursion"] is not None]
        if len(pairs) < 8:
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        if np.std(x) == 0:
            continue
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(float(slope))
        subjects.append(rec["subject"])
    if not slopes:
        return {"status": "not_computable"}
    return cluster_bootstrap_pooled_effect(np.array(slopes), subjects, "closedloop_excursion_covariate",
                                            n_perm=N_PERM, n_boot=N_BOOT, alpha=ALPHA, power=POWER)


def _km_median_per_patient(per_session_records: list[dict]) -> dict:
    from statsmodels.duration.survfunc import SurvfuncRight

    by_subject: dict[str, list[tuple[float, int]]] = {}
    for rec in per_session_records:
        if rec.get("status") != "computed":
            continue
        excursion = rec["excursion_raw"]
        for v in excursion.values():
            if v["time_to_return_s"] is None:
                continue
            by_subject.setdefault(rec["subject"], []).append((v["time_to_return_s"], v["time_to_return_status"]))

    per_patient = {}
    for subject, pairs in by_subject.items():
        if len(pairs) < 4:
            continue
        time_arr, status_arr = (np.array(x) for x in zip(*pairs))
        median = SurvfuncRight(time_arr, status_arr).quantile(0.5)
        per_patient[subject] = {
            "median_s": float(median) if np.isfinite(median) else None,
            "n_events": int(len(pairs)), "n_returned": int(status_arr.sum()),
            "n_right_censored": int((status_arr == 0).sum()),
        }
    finite_medians = [v["median_s"] for v in per_patient.values() if v["median_s"] is not None]
    if not finite_medians:
        return {"status": "not_computable", "per_patient": per_patient}
    rng = np.random.default_rng(stable_seed("km_median_bootstrap"))
    boot = [float(np.median(rng.choice(finite_medians, size=len(finite_medians), replace=True)))
            for _ in range(N_BOOT)]
    return {
        "status": "computed", "per_patient": per_patient, "n_patients": len(finite_medians),
        "median_of_patient_medians_s": float(np.median(finite_medians)),
        "ci_lower": float(np.percentile(boot, 2.5)), "ci_upper": float(np.percentile(boot, 97.5)),
        "p_value": None,
        "note": "descriptive patient-cluster bootstrap interval on the median of per-patient Kaplan-Meier "
                "medians; no natural zero-null exists for a return time, so no p-value is reported here",
    }


def _pooled_time_course(per_session_records: list[dict], field: str) -> list[dict]:
    by_j: dict[int, dict] = {}
    for rec in per_session_records:
        if rec.get("status") != "computed" or field not in rec:
            continue
        for row in rec[field]:
            bucket = by_j.setdefault(row["recovery_bin_index"], {"s": [], "c": []})
            if row["stimulated_mean"] is not None:
                bucket["s"].append(row["stimulated_mean"])
            if row["sham_mean"] is not None:
                bucket["c"].append(row["sham_mean"])
    course = []
    for j in sorted(by_j):
        s, c = by_j[j]["s"], by_j[j]["c"]
        course.append({
            "recovery_bin_index": j, "time_since_offset_s": ARTIFACT_MARGIN_S + j * BIN_S,
            "stimulated_mean": float(np.mean(s)) if s else None, "n_sessions_stimulated": len(s),
            "sham_mean": float(np.mean(c)) if c else None, "n_sessions_sham": len(c),
        })
    return course


def _pooled_excursion(per_session_records: list[dict], field: str, key: str) -> dict:
    session_vals, subjects = [], []
    for rec in per_session_records:
        if rec.get("status") != "computed" or field not in rec:
            continue
        excursion = rec[field]
        vals = [v[key] for v in excursion.values() if v[key] is not None]
        if not vals:
            continue
        session_vals.append(float(np.mean(vals)))
        subjects.append(rec["subject"])
    if not session_vals:
        return {"status": "not_computable"}
    return cluster_bootstrap_pooled_effect(np.array(session_vals), subjects, f"{field}_{key}",
                                            n_perm=N_PERM, n_boot=N_BOOT, alpha=ALPHA, power=POWER)


def _lag_duration_summary(per_session_records: list[dict]) -> dict:
    lags = [rec["median_lag_s"] for rec in per_session_records if rec.get("status") == "computed"]
    durations = [rec["median_duration_s"] for rec in per_session_records if rec.get("status") == "computed"]
    if not lags:
        return {"status": "not_computable"}
    return {"status": "computed", "n_sessions": len(lags),
            "median_lag_s": float(np.median(lags)), "median_duration_s": float(np.median(durations))}


def _recall_latency_lists_by_session(dataset: str, data_dir: Path) -> dict[str, dict[int, dict]]:
    root = data_root()
    loaded = load_recall_latency_corpus(data_dir.name, root)
    out: dict[str, dict[int, dict]] = {}
    for subject, lists in loaded["by_subject_lists"].items():
        for (events_path, list_id), entry in lists.items():
            try:
                session_key = str(Path(events_path).relative_to(data_dir)).replace(
                    "_events.tsv", "_acq-bipolar_ieeg.json")
            except ValueError:
                continue
            try:
                list_no = int(list_id)
            except (TypeError, ValueError):
                continue
            out.setdefault(session_key, {})[list_no] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None)
    args = parser.parse_args()
    t0 = time.time()

    per_corpus_records: dict[str, list[dict]] = {}
    exclusions: dict[str, dict] = {}
    for dataset, spec in CORPORA.items():
        loaded = load_corpus(dataset, spec["data_dir"], spec["derive_stim_from_stim_on"], args.smoke)
        exclusions[dataset] = dict(loaded["exclusions"])
        records = []
        for rec in loaded["records"]:
            unit = f"{dataset}__{rec['session_key']}"
            cached = load_checkpoint(unit)
            if cached is None:
                cached = _process_session(rec, spec["is_openloop"])
                save_checkpoint(unit, cached)
            record = {"session_key": rec["session_key"], "subject": f"{dataset}:{rec['subject_id']}", **cached}
            records.append(record)
            if record.get("status") != "computed":
                exclusions[dataset].setdefault(record.get("reason", record.get("status", "excluded")), []).append(
                    rec["session_key"])
        per_corpus_records[dataset] = records

    recall_latency_by_dataset = {dataset: _recall_latency_lists_by_session(dataset, spec["data_dir"])
                                  for dataset, spec in CORPORA.items()}

    def _corpus_summary(dataset: str, records: list[dict], exclusions_for_dataset: dict,
                        recall_latency_lookup: dict) -> dict:
        computed = [r for r in records if r.get("status") == "computed"]
        summary = {
            "n_sessions_seen": len(records), "n_sessions_computed": len(computed),
            "exclusions": exclusions_for_dataset,
            "lag_and_duration": _lag_duration_summary(computed),
            "time_course_raw": _pooled_time_course(computed, "time_course_raw"),
            "excursion_raw": _pooled_excursion(computed, "excursion_raw", "raw_excursion"),
            "excursion_corrected": _pooled_excursion(computed, "excursion_raw", "corrected_excursion"),
            "time_to_return_km_median_per_patient": _km_median_per_patient(computed),
            "word_level_recall_association": _word_level_recall_association(computed),
            "list_level_first_recall_latency_association": _list_level_latency_cox_association(
                computed, recall_latency_lookup),
        }
        if any(r.get("time_course_pca") for r in computed):
            summary["time_course_pca"] = _pooled_time_course(computed, "time_course_pca")
            summary["excursion_pca_raw"] = _pooled_excursion(computed, "excursion_pca", "raw_excursion")
            summary["excursion_pca_corrected"] = _pooled_excursion(computed, "excursion_pca", "corrected_excursion")
        if dataset == "ram_ds005557_closedloop":
            summary["pre_stim_distance_stim_vs_sham"] = _closedloop_pre_stim_distance(computed)
            summary["excursion_corrected_by_pre_stim_covariate"] = _closedloop_excursion_covariate(computed)
            summary["classifier_flagged_unstimulated_words"] = {
                "status": "not_available",
                "reason": "ds005557's own events.tsv carries no classifier-score or flagged-but-unstimulated "
                          "field; the matched comparison used throughout is a sham event built from "
                          "non-stimulated lists, not a classifier-flagged subset",
            }
        return summary

    per_corpus_output = {
        dataset: _corpus_summary(dataset, records, exclusions[dataset], recall_latency_by_dataset[dataset])
        for dataset, records in per_corpus_records.items()
    }
    pooled_records = [r for records in per_corpus_records.values() for r in records]
    pooled_recall_latency = {k: v for d in recall_latency_by_dataset.values() for k, v in d.items()}
    pooled_summary = _corpus_summary("pooled", pooled_records, {}, pooled_recall_latency)
    pooled_summary.pop("exclusions", None)

    output = {
        "version": "2026-09-24b",
        "labelling": "channel-space band-power distance throughout; band power is never named or treated as firing rate",
        "state_construction": {
            "bands": list(ALL_BANDS), "bin_width_s": BIN_S,
            "window_pre_onset_s": WINDOW_PRE_S, "window_post_offset_s": WINDOW_POST_S,
            "window_cut_short_at": "the next delivered train's own onset, or the end of that word's list",
            "artifact_margin_s": ARTIFACT_MARGIN_S, "excursion_window_s": EXCURSION_WINDOW_S,
            "pre_onset_window": f"{WINDOW_PRE_S} s of bins before train onset, real and sham events alike",
            "sham_construction": "non-stimulated lists (stim_list==0), serial positions that some real "
                                 "stimulated word in the same session also occupies; synthetic train onset "
                                 "= matched word onset + this session's own median real train-onset lag, "
                                 "offset = onset + this session's own median real train duration",
            "cross_fit_folds": N_FOLDS,
            "pca_probabilistic_rank_selection": "sklearn PCA(k).fit(train).score(test), argmax over k, "
                                                "1 vs 4 fold split of the sham pool",
        },
        "alagapan_phase_stimulation": {
            "status": "excluded",
            "reason": "every trial in a stimulation-condition recording is stimulated (continuous phase-locked "
                      "delivery through the whole retention period; scripts/run_alagapan_stimulation_geometry.py's "
                      "own docstring): stimulated vs non-stimulated is a between-SESSION contrast (baseline "
                      "recording vs. a separate stimulation recording), not an interleaved, same-session "
                      "trial-level design, so it has no post-stimulation recovery window of the kind this "
                      "analysis needs",
        },
        "per_corpus": per_corpus_output,
        "pooled": pooled_summary,
        "wall_clock_s": time.time() - t0,
        "code_commit": git_commit(ROOT, __file__),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json(output)
    fd, tmp_name = tempfile.mkstemp(dir=str(RESULTS), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(rendered)
        os.replace(tmp_name, OUTPUT_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    print(f"Wrote {OUTPUT_PATH} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
