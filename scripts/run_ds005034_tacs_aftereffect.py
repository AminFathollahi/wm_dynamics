#!/usr/bin/env python3
"""Paired open-data test of the ds005034 theta-tACS aftereffect."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import detrend, resample_poly, windows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from preprocessing import bandpass_filter, common_average_reference  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from stimulation_response_estimator import rate_free_state_deviation
from statistics import fdr_bh, paired_sign_flip_test, stable_seed  # noqa: E402
from corpus_sessions import SESSIONS  # noqa: E402
from preprocessing import BANDS, BASELINE_WINDOW, CELLS, DELAY_WINDOW, MAX_GLOBAL_BAD_FRACTION, POSTERIOR_ROI  # noqa: E402
from corpus_sessions import paired_inventory  # noqa: E402
from preprocessing import TARGET_SFREQ, THETA_ROI, _global_bad_channels, _interpolate, _mne_session_context, load_events, periodogram_band_power  # noqa: E402

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/wm_dynamics_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wm_dynamics_matplotlib")

ANALYSIS_ID = "ds005034_tacs_aftereffect"
ANALYSIS_VERSION = "2026-09-08"
READ_WINDOW = (-5.0, 7.0)
MIN_CLEAN_PER_CELL = 12
MAX_TRANSIENT_BAD_CHANNELS = 5






def robust_high_outliers(values: np.ndarray, z: float = 6.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    if not np.isfinite(mad) or mad == 0.0:
        return x > med
    return x > med + z * 1.4826 * mad


def transient_bad_channels(epoch_uv: np.ndarray) -> np.ndarray:
    peak_to_peak = np.ptp(epoch_uv, axis=1)
    relative = robust_high_outliers(np.log(np.maximum(peak_to_peak, np.finfo(float).tiny)))
    return np.flatnonzero(relative & (peak_to_peak > 150.0))




def spectral_features(
    epoch_uv: np.ndarray, original_sfreq: float, csd_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    downsampled = resample_poly(epoch_uv, int(TARGET_SFREQ), int(original_sfreq), axis=1)
    filtered = bandpass_filter(downsampled.T, 1.0, 45.0, TARGET_SFREQ).T
    filtered = common_average_reference(filtered.T).T
    times = np.arange(filtered.shape[1]) / TARGET_SFREQ + READ_WINDOW[0]
    baseline = periodogram_band_power(filtered, TARGET_SFREQ, BASELINE_WINDOW, times, csd_transform)
    delay = periodogram_band_power(filtered, TARGET_SFREQ, DELAY_WINDOW, times, csd_transform)
    eps = np.finfo(float).tiny
    return np.log(np.maximum(delay, eps) / np.maximum(baseline, eps)), delay


def condition_summary(features: np.ndarray, tasks: np.ndarray, loads: np.ndarray) -> dict:
    deviations = []
    centroids = []
    counts = {}
    for task, load in CELLS:
        mask = (tasks == task) & (loads == load)
        cell = features[mask]
        counts[f"{task}_load{load}"] = int(mask.sum())
        if len(cell) < MIN_CLEAN_PER_CELL:
            raise ValueError(f"{task}/load{load} has {len(cell)} clean trials")
        deviation = rate_free_state_deviation(cell)
        if np.isfinite(deviation).sum() < MIN_CLEAN_PER_CELL:
            raise ValueError(f"{task}/load{load} has too few defined spectral directions")
        deviations.append(float(np.nanmean(deviation)))
        norms = np.linalg.norm(cell, axis=1, keepdims=True)
        unit = cell / np.where(norms > 0, norms, 1.0)
        centroid = unit.mean(axis=0)
        if not np.isfinite(centroid).all() or np.linalg.norm(centroid) == 0.0:
            raise ValueError(f"{task}/load{load} has no defined centroid")
        centroids.append(centroid / np.linalg.norm(centroid))
    return {
        "equal_cell_mean_deviation": float(np.mean(deviations)),
        "equal_cell_centroid": np.mean(centroids, axis=0),
        "cell_counts": counts,
    }


def balanced_condition_summary(
    features: np.ndarray, tasks: np.ndarray, loads: np.ndarray, n_per_cell: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    selected = []
    for task, load in CELLS:
        index = np.flatnonzero((tasks == task) & (loads == load))
        if len(index) < n_per_cell:
            raise ValueError(f"{task}/load{load} cannot supply {n_per_cell} trials")
        selected.extend(rng.choice(index, n_per_cell, replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    return condition_summary(features[selected], tasks[selected], loads[selected])


def exact_vector_sign_flip(vectors: np.ndarray) -> dict:
    values = np.asarray(vectors, dtype=float)
    observed = float(np.linalg.norm(values.mean(axis=0)))
    n = len(values)
    null = np.empty(1 << n, dtype=float)
    for start in range(0, len(null), 4096):
        codes = np.arange(start, min(start + 4096, len(null)), dtype=np.uint64)
        bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        signs = 2.0 * bits - 1.0
        null[start:start + len(codes)] = np.linalg.norm(signs @ values / n, axis=1)
    return {
        "n_pairs": n,
        "mean_vector_norm": observed,
        "mean_participant_vector_norm": float(np.mean(np.linalg.norm(values, axis=1))),
        "standardized_vector_norm": float(observed / np.mean(np.linalg.norm(values, axis=1))),
        "exact_sign_flip_p_value": float(np.mean(null >= observed)),
        "null_quantiles": dict(zip(("q025", "q50", "q975"), np.quantile(null, (0.025, 0.5, 0.975)).tolist())),
    }


def paired_test(a: np.ndarray, b: np.ndarray, tag: str, alternative: str = "two-sided") -> dict:
    result = paired_sign_flip_test(
        np.asarray(a), np.asarray(b), n_perm=100_000, n_boot=10_000,
        alternative=alternative, rng=np.random.default_rng(stable_seed(f"{ANALYSIS_ID}|{tag}")),
    )
    result.pop("null", None)
    return result








def analyze_session(set_path: Path, events_path: Path, cache_path: Path, force: bool = False) -> dict:
    if cache_path.is_file() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["analysis_version"]) == ANALYSIS_VERSION:
            return {key: cached[key] for key in cached.files if key != "analysis_version"}

    raw, info, csd_transform = _mne_session_context(set_path)
    events = load_events(events_path)
    payload = loadmat(set_path, variable_names=("data", "srate"), squeeze_me=True)
    data_uv = np.asarray(payload["data"], dtype=np.float32)
    sfreq = float(payload["srate"])
    if sfreq != 1000.0 or sfreq != float(raw.info["sfreq"]):
        raise ValueError(f"unexpected sampling frequency {sfreq}")
    global_bads = _global_bad_channels(data_uv, info, stable_seed(f"{ANALYSIS_ID}|{set_path}"))
    if len(global_bads) > int(np.floor(MAX_GLOBAL_BAD_FRACTION * data_uv.shape[0])):
        raise ValueError(f"{len(global_bads)} globally bad channels exceeds 20% gate")

    rows = []
    for trial_index, event in enumerate(events):
        start = int(round((event["onset"] + READ_WINDOW[0]) * sfreq))
        stop = int(round((event["onset"] + READ_WINDOW[1]) * sfreq))
        if start < 0 or stop > data_uv.shape[1]:
            continue
        epoch = np.asarray(data_uv[:, start:stop], dtype=float)
        transient_idx = transient_bad_channels(epoch)
        if len(transient_idx) > MAX_TRANSIENT_BAD_CHANNELS:
            continue
        bad_names = sorted(set(global_bads) | {raw.ch_names[i] for i in transient_idx})
        epoch = _interpolate(epoch, info, bad_names)
        log_ratio, delay_power = spectral_features(epoch, sfreq, csd_transform)
        rows.append({
            **event, "trial_index": trial_index, "transient_bad_count": len(transient_idx),
            "median_peak_to_peak_uv": float(np.median(np.ptp(epoch, axis=1))),
            "log_ratio": log_ratio.reshape(-1), "delay_power": delay_power.reshape(-1),
        })
    if not rows:
        raise ValueError("no trials survived signal processing")

    trial_scale = np.array([row["median_peak_to_peak_uv"] for row in rows])
    keep = ~robust_high_outliers(np.log(np.maximum(trial_scale, np.finfo(float).tiny)))
    rows = [row for row, accepted in zip(rows, keep) if accepted]
    features = np.stack([row["log_ratio"] for row in rows])
    delay = np.stack([row["delay_power"] for row in rows])
    tasks = np.array([row["task"] for row in rows])
    loads = np.array([row["load"] for row in rows])
    trial_index = np.array([row["trial_index"] for row in rows], dtype=int)
    transient_counts = np.array([row["transient_bad_count"] for row in rows], dtype=int)
    summary = condition_summary(features, tasks, loads)
    absolute_summary = condition_summary(delay, tasks, loads)
    roi = _roi_summary(features, raw.ch_names, tasks, loads)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, analysis_version=ANALYSIS_VERSION, features=features, delay_power=delay,
        tasks=tasks, loads=loads, trial_index=trial_index, transient_bad_counts=transient_counts,
        global_bads=np.asarray(global_bads), equal_cell_mean_deviation=summary["equal_cell_mean_deviation"],
        equal_cell_centroid=summary["equal_cell_centroid"], cell_count_keys=np.asarray(list(summary["cell_counts"])),
        cell_count_values=np.asarray(list(summary["cell_counts"].values())),
        absolute_equal_cell_mean_deviation=absolute_summary["equal_cell_mean_deviation"],
        absolute_equal_cell_centroid=absolute_summary["equal_cell_centroid"],
        roi_json=json.dumps(roi, sort_keys=True), n_events=len(events), n_retained=len(rows),
    )
    return analyze_session(set_path, events_path, cache_path, force=False)


def _roi_summary(features: np.ndarray, ch_names: list[str], tasks: np.ndarray, loads: np.ndarray) -> dict:
    cube = features.reshape(len(features), len(ch_names), len(BANDS))
    band_index = {band: i for i, band in enumerate(BANDS)}
    channel_index = {name: i for i, name in enumerate(ch_names)}
    roi_defs = {"frontal_theta": (THETA_ROI, "theta"), "posterior_alpha": (POSTERIOR_ROI, "alpha"),
                "posterior_beta": (POSTERIOR_ROI, "beta")}
    result = {}
    for name, (channels, band) in roi_defs.items():
        indices = [channel_index[channel] for channel in channels]
        trial_values = cube[:, indices, band_index[band]].mean(axis=1)
        cells = {}
        for task, load in CELLS:
            mask = (tasks == task) & (loads == load)
            cells[f"{task}_load{load}"] = float(np.mean(trial_values[mask]))
        result[name] = {"equal_cell_mean_log_power_ratio": float(np.mean(list(cells.values()))), "cells": cells}
    return result


def _cached_session_view(cached: dict) -> dict:
    keys = [str(x) for x in cached["cell_count_keys"]]
    counts = [int(x) for x in cached["cell_count_values"]]
    return {
        "n_events": int(cached["n_events"]), "n_retained": int(cached["n_retained"]),
        "global_bad_channels": [str(x) for x in cached["global_bads"]],
        "cell_counts": dict(zip(keys, counts)),
        "baseline_normalized_equal_cell_mean_deviation": float(cached["equal_cell_mean_deviation"]),
        "absolute_power_equal_cell_mean_deviation": float(cached["absolute_equal_cell_mean_deviation"]),
        "roi": json.loads(str(cached["roi_json"])),
    }


def synthesize(participants: list[str], sessions: dict[str, dict], caches: dict[str, dict]) -> dict:
    admitted = [p for p in participants if all(sessions[p][s].get("status") == "complete" for s in SESSIONS)]
    if len(admitted) < 8:
        return {"status": "insufficient_complete_pairs", "n_admitted_pairs": len(admitted)}
    balanced = {}
    for p in admitted:
        counts = [
            int(value) for session in SESSIONS
            for value in caches[p][session]["cell_count_values"]
        ]
        n_per_cell = min(counts)
        balanced[p] = {"n_per_cell": n_per_cell}
        for session in SESSIONS:
            cache = caches[p][session]
            balanced[p][session] = {
                "baseline_normalized": balanced_condition_summary(
                    cache["features"], cache["tasks"], cache["loads"], n_per_cell,
                    stable_seed(f"{ANALYSIS_ID}|balanced|{p}|{session}|normalized"),
                ),
                "absolute": balanced_condition_summary(
                    cache["delay_power"], cache["tasks"], cache["loads"], n_per_cell,
                    stable_seed(f"{ANALYSIS_ID}|balanced|{p}|{session}|absolute"),
                ),
            }
    feature_deltas = np.stack([
        balanced[p]["verum"]["baseline_normalized"]["equal_cell_centroid"]
        - balanced[p]["sham"]["baseline_normalized"]["equal_cell_centroid"] for p in admitted
    ])
    absolute_deltas = np.stack([
        balanced[p]["verum"]["absolute"]["equal_cell_centroid"]
        - balanced[p]["sham"]["absolute"]["equal_cell_centroid"] for p in admitted
    ])
    rate_free = paired_test(
        [balanced[p]["verum"]["baseline_normalized"]["equal_cell_mean_deviation"] for p in admitted],
        [balanced[p]["sham"]["baseline_normalized"]["equal_cell_mean_deviation"] for p in admitted],
        "balanced_rate_free_dispersion",
    )
    absolute_rate_free = paired_test(
        [balanced[p]["verum"]["absolute"]["equal_cell_mean_deviation"] for p in admitted],
        [balanced[p]["sham"]["absolute"]["equal_cell_mean_deviation"] for p in admitted],
        "balanced_absolute_rate_free_dispersion",
    )
    all_clean_rate_free = paired_test(
        [caches[p]["verum"]["equal_cell_mean_deviation"] for p in admitted],
        [caches[p]["sham"]["equal_cell_mean_deviation"] for p in admitted],
        "all_clean_rate_free_dispersion",
    )
    band_shifts = {}
    for index, band in enumerate(BANDS):
        band_shifts[band] = exact_vector_sign_flip(feature_deltas.reshape(len(admitted), 129, 3)[:, :, index])
    roi_tests = {}
    roi_cells = {}
    for p in admitted:
        roi_cells[p] = {
            session: json.loads(str(caches[p][session]["roi_json"])) for session in SESSIONS
        }
    for roi in ("frontal_theta", "posterior_alpha", "posterior_beta"):
        verum = [roi_cells[p]["verum"][roi] for p in admitted]
        sham = [roi_cells[p]["sham"][roi] for p in admitted]
        stimulation = paired_test(
            [row["equal_cell_mean_log_power_ratio"] for row in verum],
            [row["equal_cell_mean_log_power_ratio"] for row in sham], f"roi|{roi}|stimulation",
        )
        load_diffs = []
        task_diffs = []
        for p in admitted:
            session_load = []
            session_task = []
            for session in SESSIONS:
                cells = roi_cells[p][session][roi]["cells"]
                load6 = np.mean([cells[f"{task}_load6"] for task, _ in CELLS[::2]])
                load4 = np.mean([cells[f"{task}_load4"] for task, _ in CELLS[::2]])
                session_load.append(load6 - load4)
                if roi == "frontal_theta":
                    session_task.append(np.mean([cells["alphabetical_load4"], cells["alphabetical_load6"]])
                                        - np.mean([cells["forward_load4"], cells["forward_load6"]]))
                else:
                    session_task.append(np.mean([cells["backward_load4"], cells["backward_load6"]])
                                        - np.mean([cells["alphabetical_load4"], cells["alphabetical_load6"]]))
            load_diffs.append(float(np.mean(session_load)))
            task_diffs.append(float(np.mean(session_task)))
        roi_tests[roi] = {
            "verum_minus_sham": stimulation,
            "load6_minus_load4": paired_test(load_diffs, np.zeros(len(load_diffs)), f"roi|{roi}|load", "greater"),
            "source_task_direction": paired_test(task_diffs, np.zeros(len(task_diffs)), f"roi|{roi}|task", "greater"),
        }
    test_refs = [(roi, contrast) for roi in roi_tests for contrast in roi_tests[roi]]
    correction = fdr_bh(np.array([roi_tests[roi][contrast]["p_value"] for roi, contrast in test_refs]))
    for index, (roi, contrast) in enumerate(test_refs):
        roi_tests[roi][contrast]["q_value_nine_source_control_family"] = float(correction["q_values"][index])
    return {
        "status": "complete", "n_admitted_pairs": len(admitted), "admitted_participants": admitted,
        "balanced_trials_per_cell_by_participant": {p: balanced[p]["n_per_cell"] for p in admitted},
        "primary_baseline_normalized_multivariate_shift": exact_vector_sign_flip(feature_deltas),
        "baseline_normalized_band_specific_shifts": band_shifts,
        "primary_rate_free_within_cell_dispersion_change": rate_free,
        "all_clean_trial_rate_free_dispersion_sensitivity": all_clean_rate_free,
        "absolute_delay_power_sensitivities": {
            "multivariate_shift": exact_vector_sign_flip(absolute_deltas), "rate_free_dispersion_change": absolute_rate_free,
        },
        "source_paper_roi_stimulation_controls": roi_tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("WM_DYNAMICS_DATA_ROOT", str(Path.home() / "data"))),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / f"{ANALYSIS_ID}.json")
    parser.add_argument("--participants", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dataset_root = args.data_root / "ds005034" if (args.data_root / "ds005034").is_dir() else args.data_root
    inventory = paired_inventory(dataset_root)
    participants = inventory["complete_pairs"]
    if args.participants:
        participants = [p for p in participants if p in set(args.participants)]
    output = {
        "analysis_id": ANALYSIS_ID, "analysis_version": ANALYSIS_VERSION, "status": "running",
        "started_unix": time.time(), "code_commit": git_commit(ROOT),
        "design": {
            "estimand": "participant-paired verum-minus-sham aftereffect on delay-period scalp spectral state",
            "independent_unit": "participant", "randomization_inference": "within-participant sign flips",
            "source": "Journal of Cognitive Neuroscience, doi 10.1162/jocn_a_02269",
            "windows_seconds_from_delay_onset": {"baseline": BASELINE_WINDOW, "delay": DELAY_WINDOW},
            "bands_hz": BANDS, "minimum_clean_trials_per_task_load_cell": MIN_CLEAN_PER_CELL,
            "minimum_admitted_complete_pairs": 8,
            "preprocessing": (
                "read 12-second buffered delay epochs; polyphase downsample 1000 to 250 Hz; 1-45 Hz "
                "zero-phase Butterworth filter; PREP noisy-channel detection on 120 sampled seconds; "
                "spherical interpolation of global and at most five transient bad channels; average reference; "
                "frequency-domain spherical-spline CSD (m=4, lambda2=1e-5, 50 Legendre terms)"
            ),
            "primary_feature": (
                "channel-by-theta/alpha/beta log delay-to-baseline CSD power ratio, balanced within each "
                "participant across the six task-by-load cells and both sessions"
            ),
            "limitations": [
                "EEG begins tens of minutes after stimulation; this is an aftereffect, not online maintenance stimulation.",
                "The public BIDS release lacks trial behavior and realized session order.",
                "Automated PREP-style channel and epoch QC cannot reproduce the source paper's manual AMICA and trial review.",
            ],
        },
        "inventory": {key: value for key, value in inventory.items() if key != "paths"}, "sessions": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(output))
    checkpoint_root = ROOT / "results" / ".checkpoints" / ANALYSIS_ID
    session_caches = {}
    for participant in participants:
        output["sessions"][participant] = {}
        session_caches[participant] = {}
        for session in SESSIONS:
            set_path = inventory["paths"][participant][session]
            events_path = set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))
            cache_path = checkpoint_root / f"{participant}_ses-{session}.npz"
            print(f"{participant} {session}", flush=True)
            try:
                cached = analyze_session(set_path, events_path, cache_path, force=args.force)
                session_caches[participant][session] = cached
                output["sessions"][participant][session] = {
                    "status": "complete", "set_path": str(set_path), "summary": _cached_session_view(cached),
                }
            except Exception as exc:
                output["sessions"][participant][session] = {"status": "refused", "reason": f"{type(exc).__name__}: {exc}"}
            args.output.write_text(canonical_json(output))
    output["inference"] = synthesize(participants, output["sessions"], session_caches)
    output["status"] = "complete" if output["inference"]["status"] == "complete" else "incomplete"
    output["duration_seconds"] = time.time() - output["started_unix"]
    args.output.write_text(canonical_json(output))


if __name__ == "__main__":
    main()
