from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test,
    spearman_permutation_test, stable_seed,
)
from io_utils import locked_json_update  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import DATA, PRE_S, SESSIONS, load_macaque_pfc_microstimulation_session  # noqa: E402
from run_macaque_pfc_microstimulation_design_corrected import BIN_MS, N_BINS, N_COMPONENTS, ONSET_BIN, bin_spiketrain  # noqa: E402

PRE_STIM_BASELINE_BINS = (0, 10)
CONTROL_CV_FOLDS = 5
MIN_FOLD_ATTRACTOR_CONTROLS = 4
MIN_FOLD_CONTROL_SPREAD_CONTROLS = 5

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints"
SCHEMA_VERSION = "recovery_latency_v6"

SACCADE_MAD_K = 8.0
SACCADE_MIN_SUSTAIN_S = 0.020
SACCADE_SEARCH_START_S = 0.0
SACCADE_BASELINE_WINDOW_S = (-0.6, -0.1)
SACCADE_MIN_LATENCY_S = 0.05
SACCADE_MAX_LATENCY_S = 1.6
SACCADE_SMOOTH_WINDOW_SAMPLES = 15
MIN_EYEDAT_SAMPLES = 200
BLANKING_MARGIN_BINS = 1


def frozen_mdd(values: np.ndarray) -> dict:
    result = minimum_detectable_paired_difference(values)
    if result.get("status") == "computed":
        result = {**result, "z_factor": Z_80_POWER,
                  "mdd": Z_80_POWER * result["sd"] / np.sqrt(result["n"])}
    return result


def _smooth_columns(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.vstack([np.convolve(values[:, i], kernel, mode="same") for i in range(values.shape[1])]).T


def _valid_eyedat(xy: np.ndarray) -> bool:
    return xy.ndim == 2 and xy.shape[0] >= MIN_EYEDAT_SAMPLES and xy.shape[1] == 2 and np.isfinite(xy).all()


def _behavior_trials_scipy(path: Path) -> dict | None:
    data = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    if "behavior" not in data:
        return None
    beh = data["behavior"]
    if beh.ndim != 3:
        return None
    n_cond, n_angle, n_trial = beh.shape
    out: dict[tuple[int, int], list[dict]] = {}
    for c in range(n_cond):
        for a in range(n_angle):
            trials = []
            for k in range(n_trial):
                trial = beh[c, a, k]
                try:
                    eyedat = np.asarray(trial.eyedat, dtype=float)
                    times = np.asarray(trial.times, dtype=float).reshape(-1)
                except AttributeError:
                    continue
                if eyedat.ndim == 2 and eyedat.shape[0] in (2, 3) and eyedat.shape[1] == len(times):
                    xy = eyedat[:2].T
                elif eyedat.ndim == 2 and eyedat.shape[1] in (2, 3) and eyedat.shape[0] == len(times):
                    xy = eyedat[:, :2]
                else:
                    continue
                if not _valid_eyedat(xy):
                    continue
                trials.append({"eyedat_xy": xy, "times": times})
            out[(c, a)] = trials
    return out


def _behavior_trials_h5py(path: Path) -> dict | None:
    with h5py.File(str(path), "r") as f:
        if "behavior" not in f:
            return None
        beh = f["behavior"]
        if beh.ndim != 4:
            return None
        n_trial, n_angle, n_amp, n_cond = beh.shape
        out: dict[tuple[int, int], list[dict]] = {}
        for c in range(n_cond):
            for a in range(n_angle):
                trials = []
                for tr in range(n_trial):
                    ref = beh[tr, a, 0, c]
                    grp = f[ref]
                    if not hasattr(grp, "keys") or "eyedat" not in grp or "times" not in grp:
                        continue
                    eyedat = np.asarray(grp["eyedat"], dtype=float)
                    times = np.asarray(grp["times"], dtype=float).reshape(-1)
                    if eyedat.ndim == 2 and eyedat.shape[0] == len(times) and eyedat.shape[1] >= 2:
                        xy = eyedat[:, :2]
                    elif eyedat.ndim == 2 and eyedat.shape[1] == len(times) and eyedat.shape[0] >= 2:
                        xy = eyedat[:2].T
                    else:
                        continue
                    if not _valid_eyedat(xy):
                        continue
                    trials.append({"eyedat_xy": xy, "times": times})
                out[(c, a)] = trials
    return out


def load_behavior_trials(prefix: str, correct: bool) -> dict | None:
    folder = "correct" if correct else "error"
    fname = f"{prefix}.mat" if correct else f"{prefix}_err.mat"
    path = DATA / folder / fname
    if not path.exists():
        return None
    try:
        with h5py.File(str(path), "r"):
            is_h5 = True
    except OSError:
        is_h5 = False
    try:
        return _behavior_trials_h5py(path) if is_h5 else _behavior_trials_scipy(path)
    except Exception:
        return None


def stim_train_duration_ms(prefix: str) -> float | None:
    path = DATA / "correct" / f"{prefix}.mat"
    if not path.exists():
        return None
    try:
        with h5py.File(str(path), "r"):
            is_h5 = True
    except OSError:
        is_h5 = False
    try:
        if is_h5:
            with h5py.File(str(path), "r") as f:
                if "behavior" not in f:
                    return None
                beh = f["behavior"]
                if beh.ndim != 4:
                    return None
                n_trial, n_angle, n_amp, n_cond = beh.shape
                for c in range(n_cond):
                    grp = f[beh[0, 0, 0, c]]
                    if not hasattr(grp, "keys") or "trialinfo" not in grp:
                        continue
                    ti = grp["trialinfo"]
                    if "xippmexStimDur" in ti:
                        return float(np.asarray(ti["xippmexStimDur"]).ravel()[0])
            return None
        data = loadmat(str(path), struct_as_record=False, squeeze_me=True)
        if "behavior" not in data:
            return None
        beh = data["behavior"]
        if beh.ndim != 3:
            return None
        for c in range(beh.shape[0]):
            trial = beh[c, 0, 0]
            ti = getattr(trial, "trialinfo", None)
            if ti is not None and hasattr(ti, "xippmexStimDur"):
                return float(ti.xippmexStimDur)
        return None
    except Exception:
        return None


def blanking_exclusion_bins(train_duration_ms: float | None) -> int:
    if train_duration_ms is None:
        return 0
    return int(np.ceil(train_duration_ms / BIN_MS)) + BLANKING_MARGIN_BINS


def detect_saccade_onset(times: np.ndarray, xy: np.ndarray) -> float | None:
    dt = np.diff(times)
    if np.median(dt) <= 0:
        return None
    xy_smooth = _smooth_columns(xy, SACCADE_SMOOTH_WINDOW_SAMPLES)
    velocity = np.linalg.norm(np.diff(xy_smooth, axis=0), axis=1) / np.maximum(dt, 1e-6)
    velocity_times = times[1:]
    baseline_mask = (velocity_times >= SACCADE_BASELINE_WINDOW_S[0]) & (velocity_times <= SACCADE_BASELINE_WINDOW_S[1])
    if baseline_mask.sum() < 10:
        return None
    baseline = velocity[baseline_mask]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median))) + 1e-9
    threshold = median + SACCADE_MAD_K * 1.4826 * mad
    min_sustain_samples = max(1, int(round(SACCADE_MIN_SUSTAIN_S / np.median(dt))))
    search_idx = np.where(velocity_times >= SACCADE_SEARCH_START_S)[0]
    run = 0
    for i in search_idx:
        if velocity[i] > threshold:
            run += 1
            if run >= min_sustain_samples:
                onset_i = i - min_sustain_samples + 1
                onset_time = float(velocity_times[onset_i])
                if SACCADE_MIN_LATENCY_S <= onset_time <= SACCADE_MAX_LATENCY_S:
                    return onset_time
                return None
        else:
            run = 0
    return None


def validate_saccade_detection(rows: list[dict]) -> dict:
    usable = [(i, r["saccade_latency"]) for i, r in enumerate(rows) if r["saccade_latency"] is not None]
    n_total = len(rows)
    n_usable = len(usable)
    if n_usable < 8:
        return {"n_total_trials": n_total, "n_usable_latency": n_usable,
                "n_unusable": n_total - n_usable, "status": "too_few_usable_trials_to_validate"}
    latencies = np.array([lat for _, lat in usable])
    order = np.array([i for i, _ in usable])
    half = len(latencies) // 2
    first_half = latencies[np.argsort(order)][:half]
    second_half = latencies[np.argsort(order)][half:]
    return {
        "n_total_trials": n_total, "n_usable_latency": n_usable, "n_unusable": n_total - n_usable,
        "median_latency_s": float(np.median(latencies)),
        "first_half_median_s": float(np.median(first_half)) if len(first_half) else None,
        "second_half_median_s": float(np.median(second_half)) if len(second_half) else None,
        "split_half_abs_difference_s": (float(abs(np.median(first_half) - np.median(second_half)))
                                        if len(first_half) and len(second_half) else None),
        "status": "validated",
    }


def _pool_rows_with_behavior(prefix: str, correct: bool) -> tuple[list[dict], dict]:
    neural = load_macaque_pfc_microstimulation_session(prefix, correct=correct, neural_field="spiketrain")
    if neural is None:
        return [], {}
    behavior = load_behavior_trials(prefix, correct)
    occurrences: dict[tuple[int, int], int] = {}
    rows = []
    for trial in neural["trials"]:
        key = (int(trial["stim_cond"]), int(trial["angle_idx"]))
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        counts = bin_spiketrain(trial["spikerate"])
        if counts is None:
            continue
        eyedat_xy, times = None, None
        if behavior is not None:
            cell = behavior.get(key)
            if cell is not None and occurrence < len(cell):
                eyedat_xy = cell[occurrence]["eyedat_xy"]
                times = cell[occurrence]["times"]
        saccade_latency = detect_saccade_onset(times, eyedat_xy) if times is not None else None
        rows.append({"stim_cond": key[0], "angle_idx": key[1], "correct": int(correct),
                     "counts": counts, "saccade_latency": saccade_latency, "occurrence": occurrence})
    meta = {"channel_ids": neural["channel_ids"], "stim_channels": neural["stim_channels"],
            "control_idx": neural["control_idx"], "behavior_available": behavior is not None}
    return rows, meta


def _fit_control_projection(control_counts: np.ndarray, unit_mask: np.ndarray | None) -> tuple:
    if unit_mask is not None:
        control_counts = control_counts[:, :, unit_mask]
    center = control_counts.reshape(-1, control_counts.shape[-1]).mean(axis=0)
    scale = control_counts.reshape(-1, control_counts.shape[-1]).std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (control_counts - center) / scale
    pca = PCA(n_components=min(N_COMPONENTS, standardized.shape[-1]))
    pca.fit(standardized.reshape(-1, standardized.shape[-1]))
    return center, scale, pca


def _project_counts(counts: np.ndarray, center: np.ndarray, scale: np.ndarray, pca: PCA,
                    unit_mask: np.ndarray | None) -> np.ndarray:
    if unit_mask is not None:
        counts = counts[:, :, unit_mask]
    standardized = (counts - center) / scale
    return pca.transform(standardized.reshape(-1, standardized.shape[-1])).reshape(len(counts), N_BINS, -1)


def fit_cross_fold_projections(rows: list[dict], control_idx: int, prefix: str,
                               unit_mask: np.ndarray | None = None,
                               n_folds: int = CONTROL_CV_FOLDS) -> dict | None:
    control_rows_idx = [i for i, r in enumerate(rows) if r["stim_cond"] == control_idx]
    if len(control_rows_idx) < 20:
        return None
    rng = np.random.default_rng(stable_seed(f"mac_cv_folds_{prefix}"))
    order = rng.permutation(len(control_rows_idx))
    fold_of = {control_rows_idx[pos]: int(rank % n_folds) for rank, pos in enumerate(order)}
    stim_rows_idx = [i for i, r in enumerate(rows) if r["stim_cond"] != control_idx]
    all_counts = np.stack([r["counts"] for r in rows])

    control_latent: dict[int, np.ndarray] = {}
    stim_latent_by_fold: dict[int, dict[int, np.ndarray]] = {}
    for k in range(n_folds):
        train_idx = [i for i in control_rows_idx if fold_of[i] != k]
        test_idx = [i for i in control_rows_idx if fold_of[i] == k]
        if len(train_idx) < 10 or not test_idx:
            continue
        center, scale, pca = _fit_control_projection(all_counts[train_idx], unit_mask)
        test_latent = _project_counts(all_counts[test_idx], center, scale, pca, unit_mask)
        for j, i in enumerate(test_idx):
            control_latent[i] = test_latent[j]
        if stim_rows_idx:
            stim_latent = _project_counts(all_counts[stim_rows_idx], center, scale, pca, unit_mask)
            stim_latent_by_fold[k] = {i: stim_latent[j] for j, i in enumerate(stim_rows_idx)}

    if not control_latent:
        return None
    return {"control_latent": control_latent, "stim_latent_by_fold": stim_latent_by_fold,
            "fold_of_control": fold_of, "control_rows_idx": list(control_latent.keys())}


def condition_recovery(rows: list[dict], condition: int, control_idx: int,
                       analysis_start_bin: int, fit: dict) -> dict | None:
    stim_idx = [i for i, r in enumerate(rows) if r["stim_cond"] == condition]
    if not stim_idx:
        return None

    control_latent = fit["control_latent"]
    stim_latent_by_fold = fit["stim_latent_by_fold"]
    fold_of_control = fit["fold_of_control"]

    control_by_angle_fold: dict[tuple[int, int], list[int]] = {}
    for i in fit["control_rows_idx"]:
        key = (rows[i]["angle_idx"], fold_of_control[i])
        control_by_angle_fold.setdefault(key, []).append(i)

    def fold_attractor(angle: int, fold: int, exclude: int | None = None) -> np.ndarray | None:
        members = control_by_angle_fold.get((angle, fold), [])
        if exclude is not None:
            members = [c for c in members if c != exclude]
        min_n = MIN_FOLD_CONTROL_SPREAD_CONTROLS if exclude is not None else MIN_FOLD_ATTRACTOR_CONTROLS
        if len(members) < min_n:
            return None
        return np.mean([control_latent[c][:ONSET_BIN].mean(axis=0) for c in members], axis=0)

    own_baseline_dist, attractor_dist, control_attractor_dist = [], [], []
    used_stim_idx, used_angle = [], []
    for i in stim_idx:
        angle = rows[i]["angle_idx"]
        fold_dists, fold_own = [], []
        for k, latents in stim_latent_by_fold.items():
            if i not in latents:
                continue
            attractor = fold_attractor(angle, k)
            if attractor is None:
                continue
            latent_k = latents[i]
            fold_dists.append(np.linalg.norm(latent_k - attractor, axis=1))
            own_pre_k = latent_k[:ONSET_BIN].mean(axis=0)
            fold_own.append(np.linalg.norm(latent_k - own_pre_k, axis=1))
        if not fold_dists:
            continue
        attractor_dist.append(np.mean(fold_dists, axis=0))
        own_baseline_dist.append(np.mean(fold_own, axis=0))
        used_stim_idx.append(i)
        used_angle.append(angle)

    for (angle, fold), members in control_by_angle_fold.items():
        for i in members:
            attractor = fold_attractor(angle, fold, exclude=i)
            if attractor is None:
                continue
            control_attractor_dist.append(np.linalg.norm(control_latent[i] - attractor, axis=1))

    if not attractor_dist:
        return None
    attractor_dist = np.array(attractor_dist)
    own_baseline_dist = np.array(own_baseline_dist)
    control_attractor_dist = np.array(control_attractor_dist)
    used_angle = np.array(used_angle)

    mean_attractor = attractor_dist.mean(axis=0)
    mean_own = own_baseline_dist.mean(axis=0)
    mean_control = control_attractor_dist.mean(axis=0)

    fixed_lo, fixed_hi = analysis_start_bin, N_BINS
    post = mean_attractor[fixed_lo:fixed_hi]
    control_post = mean_control[fixed_lo:fixed_hi]
    spread = float(np.percentile(control_attractor_dist[:, fixed_lo:fixed_hi], 95))
    time_axis = (np.arange(fixed_lo, fixed_hi) - ONSET_BIN) * (BIN_MS / 1000.0)

    peak_idx = int(np.argmax(post))
    peak_value = float(post[peak_idx])
    per_trial_max = attractor_dist[:, fixed_lo:fixed_hi].max(axis=1)
    per_trial_maximum_mean = float(per_trial_max.mean())

    excluded_bins_course = mean_attractor[ONSET_BIN:fixed_lo]
    within_train_excursion_ratio = (float(excluded_bins_course.max() / post.mean())
                                    if len(excluded_bins_course) and post.mean() != 0 else None)

    sustain_bins = 3
    return_idx = None
    for j in range(len(post)):
        window = post[j:j + sustain_bins]
        if len(window) == sustain_bins and np.all(window <= spread):
            return_idx = j
            break
    time_to_return_s = float(time_axis[return_idx]) if return_idx is not None else None

    pre_lo, pre_hi = PRE_STIM_BASELINE_BINS
    pre_trial = attractor_dist[:, pre_lo:pre_hi].mean(axis=1)
    post_trial = attractor_dist[:, fixed_lo:fixed_hi].mean(axis=1)
    control_pre_mean = float(mean_control[pre_lo:pre_hi].mean())
    control_post_mean = float(control_post.mean())

    pre_stim_per_trial = pre_trial - control_pre_mean
    uncorrected_per_trial = post_trial - control_post_mean
    corrected_per_trial = uncorrected_per_trial - pre_stim_per_trial

    rng = np.random.default_rng(stable_seed(f"mac_level_{condition}"))
    pre_stim_offset_ci = bootstrap_ci(pre_stim_per_trial, lambda x: float(np.mean(x)), rng=rng)
    uncorrected_ci = bootstrap_ci(uncorrected_per_trial, lambda x: float(np.mean(x)), rng=rng)
    corrected_ci = bootstrap_ci(corrected_per_trial, lambda x: float(np.mean(x)), rng=rng)

    by_angle = {}
    for angle in sorted(set(used_angle.tolist())):
        mask = used_angle == angle
        by_angle[str(angle)] = {"pre_stimulation_offset": float(pre_stim_per_trial[mask].mean()),
                                "n_stim_trials": int(mask.sum())}

    return {
        "condition": condition, "n_stim_trials_used": int(attractor_dist.shape[0]),
        "n_control_trials_used": int(control_attractor_dist.shape[0]),
        "analysis_start_bin": int(analysis_start_bin),
        "post_stimulation_window_bins": [int(fixed_lo), int(fixed_hi - 1)],
        "post_stimulation_window_s": [float(time_axis[0]), float(time_axis[-1])],
        "attractor_distance_timecourse": mean_attractor.tolist(),
        "own_baseline_distance_timecourse": mean_own.tolist(),
        "control_attractor_distance_timecourse": mean_control.tolist(),
        "control_attractor_distance_spread_95pct": spread,
        "peak_attractor_distance": peak_value,
        "per_trial_maximum_attractor_distance_mean": per_trial_maximum_mean,
        "within_train_excursion_ratio": within_train_excursion_ratio,
        "time_to_return_within_control_spread_s": time_to_return_s,
        "not_returned_within_window": return_idx is None,
        "pre_stimulation_offset": pre_stim_offset_ci[0], "pre_stimulation_offset_ci": [pre_stim_offset_ci[1], pre_stim_offset_ci[2]],
        "level_shift_uncorrected": uncorrected_ci[0], "level_shift_uncorrected_ci": [uncorrected_ci[1], uncorrected_ci[2]],
        "level_shift_baseline_corrected": corrected_ci[0], "level_shift_baseline_corrected_ci": [corrected_ci[1], corrected_ci[2]],
        "pre_stimulation_offset_by_angle": by_angle,
        "mean_occurrence_index": float(np.mean([rows[i]["occurrence"] for i in used_stim_idx])),
        "trial_saccade_latency_s": [rows[i]["saccade_latency"] for i in used_stim_idx],
        "trial_peak_attractor_distance": per_trial_max.tolist(),
        "trial_index": used_stim_idx,
    }


def analyze_session(prefix: str) -> dict:
    correct_rows, correct_meta = _pool_rows_with_behavior(prefix, True)
    error_rows, error_meta = _pool_rows_with_behavior(prefix, False)
    if not correct_rows:
        return {"session": prefix, "status": "excluded", "reason": "no usable correct-trial pool"}
    rows = correct_rows + (error_rows if error_rows else [])
    control_idx = correct_meta["control_idx"]
    if control_idx is None:
        return {"session": prefix, "status": "excluded", "reason": "no control condition identified"}

    fit = fit_cross_fold_projections(rows, control_idx, prefix)
    if fit is None:
        return {"session": prefix, "status": "excluded", "reason": "fewer than 20 control trials with valid counts"}
    control_rows_used = [r for r in rows if r["stim_cond"] == control_idx]

    train_duration_ms = stim_train_duration_ms(prefix)
    excluded_bins = blanking_exclusion_bins(train_duration_ms)
    analysis_start_bin = min(ONSET_BIN + excluded_bins, N_BINS - 2)

    eligible = sorted({r["stim_cond"] for r in rows if r["stim_cond"] != control_idx})
    conditions = {}
    for condition in eligible:
        result = condition_recovery(rows, condition, control_idx, analysis_start_bin, fit)
        if result is not None:
            conditions[str(condition)] = result

    saccade_rows = [r for r in rows if r["stim_cond"] != control_idx]
    validation = validate_saccade_detection(saccade_rows)

    latency_vs_recovery = None
    latencies, peaks = [], []
    for cond_result in conditions.values():
        for lat, peak in zip(cond_result["trial_saccade_latency_s"], cond_result["trial_peak_attractor_distance"]):
            if lat is not None:
                latencies.append(lat)
                peaks.append(peak)
    if len(latencies) >= 8:
        corr = spearman_permutation_test(np.array(latencies), np.array(peaks),
                                         rng=np.random.default_rng(stable_seed(f"mac_latency_recovery_{prefix}")))
        latency_vs_recovery = {"n": len(latencies), "rho": corr["rho"], "p_value": corr["p_value"]}

    return {
        "session": prefix, "animal": prefix[:2], "status": "complete",
        "n_trials_pooled": len(rows), "n_control_trials": len(control_rows_used),
        "behavior_available": correct_meta.get("behavior_available", False),
        "stim_train_duration_ms": train_duration_ms,
        "stim_train_duration_source": "trialinfo.xippmexStimDur, read from the first trial of each "
            "stimulation condition in the session's correct-trial file",
        "blanking_excluded_bins": excluded_bins,
        "blanking_excluded_window_s": [0.0, round(excluded_bins * BIN_MS / 1000.0, 3)],
        "blanking_exclusion_method": "ceil(stim_train_duration_ms / bin_ms) bins covering the "
            f"delivered pulse train, plus a {BLANKING_MARGIN_BINS}-bin margin for amplifier "
            "settling; statistics below start at analysis_start_bin, the raw stored time courses "
            "are not truncated",
        "within_train_excursion_note": "the excluded bins cannot be separated from the electrical "
            "stimulation artifact at 150 ms, 350 Hz, 250 us pulses; within_train_excursion_ratio per "
            "condition is the maximum of those bins' attractor distance divided by the mean of the "
            "post-train window. every other statistic below describes the post-train course only.",
        "bin_width_s": BIN_MS / 1000.0,
        "bin_zero_time_relative_to_stim_onset_s": -PRE_S,
        "eligible_conditions": eligible, "conditions": conditions,
        "saccade_detection_validation": validation,
        "latency_vs_peak_recovery_distance": latency_vs_recovery,
    }


def _checkpoint_path(prefix: str, schema_version: str = SCHEMA_VERSION) -> Path:
    return CHECKPOINT_DIR / f"macaque_recovery_latency_{schema_version}_{prefix}.json"


def load_checkpoint(prefix: str) -> dict | None:
    path = _checkpoint_path(prefix)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    return restore_checkpoint(payload["result"])


def save_checkpoint(prefix: str, result: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "result": checkpoint_safe(result)}
    _checkpoint_path(prefix).write_text(json.dumps(payload))


def pooled_latency_vs_recovery(per_session: list[dict]) -> dict:
    rhos, ns, animals = [], [], []
    by_animal: dict[str, list[float]] = {}
    for session in per_session:
        if session.get("status") != "complete":
            continue
        within = session.get("latency_vs_peak_recovery_distance")
        if within is None:
            continue
        rhos.append(within["rho"])
        ns.append(within["n"])
        animals.append(session["animal"])
        by_animal.setdefault(session["animal"], []).append(within["rho"])
    if len(rhos) < 2:
        return {"status": "not_computable", "n_sessions": len(rhos)}
    rhos_arr = np.array(rhos)
    sign_flip = paired_sign_flip_test(rhos_arr, np.zeros_like(rhos_arr), alternative="two-sided",
                                      rng=np.random.default_rng(stable_seed("mac_pooled_latency_recovery_sign_flip")))
    per_animal = {animal: {"n_sessions": len(vals), "mean_rho": float(np.mean(vals)), "rho_per_session": vals}
                  for animal, vals in sorted(by_animal.items())}
    return {
        "status": "computed", "n_sessions": len(rhos), "session_rho": rhos, "session_n_trials": ns,
        "mean_within_session_rho": float(np.mean(rhos_arr)),
        "sign_flip_test": {"mean_diff": sign_flip["mean_diff"], "p_value": sign_flip["p_value"],
                           "ci_lower": sign_flip["ci_lower"], "ci_upper": sign_flip["ci_upper"],
                           "n": sign_flip["n"]},
        "per_animal": per_animal,
        "note": "each session_rho is the within-session Spearman rho between trial saccade latency "
                "and trial peak attractor distance, pooled across conditions and angles within that "
                "session; the cross-session estimate is the sign-flip test on the per-session rho "
                "values against zero",
    }


def pooled_recovery_stats(per_session: list[dict]) -> dict:
    peaks, per_trial_max_means = [], []
    pre_offset, uncorrected, corrected = [], [], []
    not_returned = []
    for session in per_session:
        if session.get("status") != "complete":
            continue
        for cond in session["conditions"].values():
            peaks.append(cond["peak_attractor_distance"])
            per_trial_max_means.append(cond["per_trial_maximum_attractor_distance_mean"])
            pre_offset.append(cond["pre_stimulation_offset"])
            uncorrected.append(cond["level_shift_uncorrected"])
            corrected.append(cond["level_shift_baseline_corrected"])
            not_returned.append(cond["not_returned_within_window"])
    if len(peaks) < 2:
        return {"status": "not_computable", "n": len(peaks)}
    peaks = np.array(peaks)
    per_trial_max_means = np.array(per_trial_max_means)
    pre_offset = np.array(pre_offset)
    uncorrected = np.array(uncorrected)
    corrected = np.array(corrected)
    rng = np.random.default_rng(stable_seed("mac_pooled_recovery"))
    peak_ci = bootstrap_ci(peaks, lambda x: float(np.mean(x)), rng=rng)
    pre_ci = bootstrap_ci(pre_offset, lambda x: float(np.mean(x)), rng=rng)
    uncorrected_ci = bootstrap_ci(uncorrected, lambda x: float(np.mean(x)), rng=rng)
    corrected_ci = bootstrap_ci(corrected, lambda x: float(np.mean(x)), rng=rng)
    return {
        "n_conditions": len(peaks), "mean_peak_attractor_distance": peak_ci[0],
        "peak_ci": [peak_ci[1], peak_ci[2]],
        "peak_attractor_distance_definition": "maximum of the trial-averaged (mean across trials) "
            "distance time course within the fixed post-train window; the within-train excursion "
            "is not measurable (see within_train_excursion_note) so this is typically the window's "
            "first bin on a decaying course, not a physiological peak",
        "mean_per_trial_maximum_attractor_distance": float(per_trial_max_means.mean()),
        "per_trial_maximum_note": "mean of each trial's own maximum distance, a noise-inflated "
            "secondary statistic (the expected maximum of a noisy series exceeds the maximum of "
            "its mean); not compared against the level-shift statistics below",
        "n_not_returned_within_window": int(np.sum(not_returned)),
        "fraction_not_returned_within_window": float(np.mean(not_returned)),
        "pre_stimulation_offset_mean": pre_ci[0], "pre_stimulation_offset_ci": [pre_ci[1], pre_ci[2]],
        "pre_stimulation_offset_n_positive": int(np.sum(pre_offset > 0)),
        "level_shift_uncorrected_mean": uncorrected_ci[0], "level_shift_uncorrected_ci": [uncorrected_ci[1], uncorrected_ci[2]],
        "level_shift_uncorrected_n_positive": int(np.sum(uncorrected > 0)),
        "level_shift_baseline_corrected_mean": corrected_ci[0], "level_shift_baseline_corrected_ci": [corrected_ci[1], corrected_ci[2]],
        "level_shift_baseline_corrected_n_positive": int(np.sum(corrected > 0)),
        "level_shift_note": "level_shift_baseline_corrected (primary) removes each condition's own "
            "pre_stimulation_offset from level_shift_uncorrected, both computed on the fixed "
            "post-train window; every one of the n_conditions conditions is included in every "
            "statistic above, none excluded for not returning within the window.",
        "minimum_detectable_difference_peak": frozen_mdd(peaks),
        "minimum_detectable_difference_level_shift_uncorrected": frozen_mdd(uncorrected),
        "minimum_detectable_difference_level_shift_baseline_corrected": frozen_mdd(corrected),
    }


def pre_stimulation_offset_diagnostics(per_session: list[dict]) -> dict:
    offsets, n_stim, n_control, occurrence = [], [], [], []
    by_angle: dict[str, list[tuple[float, int]]] = {}
    for session in per_session:
        if session.get("status") != "complete":
            continue
        for cond in session["conditions"].values():
            offsets.append(cond["pre_stimulation_offset"])
            n_stim.append(cond["n_stim_trials_used"])
            n_control.append(cond["n_control_trials_used"])
            occurrence.append(cond["mean_occurrence_index"])
            for angle, entry in cond["pre_stimulation_offset_by_angle"].items():
                by_angle.setdefault(angle, []).append((entry["pre_stimulation_offset"], entry["n_stim_trials"]))
    offsets = np.array(offsets)
    if len(offsets) < 8:
        return {"status": "not_computable", "n": len(offsets)}
    rng = np.random.default_rng(stable_seed("mac_pre_stim_offset_diag"))
    trial_count_corr = spearman_permutation_test(offsets, np.array(n_stim), rng=rng)
    control_count_corr = spearman_permutation_test(offsets, np.array(n_control), rng=rng)
    occurrence_corr = spearman_permutation_test(offsets, np.array(occurrence), rng=rng)
    angle_table = {
        angle: {"mean_pre_stimulation_offset": float(np.mean([o for o, _ in rows])),
               "n_conditions": len(rows), "n_trials_total": int(sum(n for _, n in rows))}
        for angle, rows in sorted(by_angle.items())
    }
    return {
        "n_conditions": len(offsets),
        "trial_count_correlation": {"rho": trial_count_corr["rho"], "p_value": trial_count_corr["p_value"],
                                    "against": "n_stim_trials_used"},
        "control_trial_count_correlation": {"rho": control_count_corr["rho"], "p_value": control_count_corr["p_value"],
                                            "against": "n_control_trials_used"},
        "session_order_correlation": {"rho": occurrence_corr["rho"], "p_value": occurrence_corr["p_value"],
                                      "against": "mean_occurrence_index",
                                      "note": "mean_occurrence_index is each condition's mean position "
                                          "within its own (stim_cond, angle) trial stream, not a "
                                          "verified session-wide clock -- this corpus's correct/error "
                                          "file split has no shared original trial index or timestamp, "
                                          "so this is a proxy for order, not a reconstruction of it"},
        "by_angle": angle_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=None)
    args = parser.parse_args()
    if "__WM_DYNAMICS_DATA_ROOT_NOT_SET__" in str(DATA) or not DATA.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT; macaque microstimulation data directory unavailable.")

    chosen = SESSIONS[:args.sessions] if args.sessions else SESSIONS
    per_session = []
    for prefix in chosen:
        cached = load_checkpoint(prefix)
        if cached is not None:
            print(f"checkpoint hit {prefix}", flush=True)
            per_session.append(cached)
            continue
        print(f"analyzing {prefix}", flush=True)
        result = analyze_session(prefix)
        save_checkpoint(prefix, result)
        per_session.append(result)

    complete = [s for s in per_session if s.get("status") == "complete"]
    output = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(ROOT),
        "gated_unit_selection": "not applied -- all surviving channels after shorted-channel exclusion "
            "and PCA are used (unit_mask=None); fit_cross_fold_projections accepts a boolean unit_mask "
            "so a memorandum-selective subset can be passed in once that selection routine exists "
            "elsewhere in this project, without re-fitting this module",
        "corpus_scope_note": "single anatomical structure (dlPFC) by design, two animals (Wa: 10 "
            "sessions, amplitude 50; Sa: 1 session, amplitude 125, non-commensurable montage). This "
            "arm speaks to a within-session recovery time course only, never to a regional difference "
            "and never to how any effect scales with stimulation intensity.",
        "bin_width_s": BIN_MS / 1000.0,
        "bin_zero_time_relative_to_stim_onset_s": -PRE_S,
        "n_sessions_complete": len(complete), "n_sessions_excluded": len(per_session) - len(complete),
        "per_session": per_session,
        "pooled": pooled_recovery_stats(per_session),
        "pre_stimulation_offset_diagnostics": pre_stimulation_offset_diagnostics(per_session),
        "pooled_latency_vs_peak_recovery_distance": pooled_latency_vs_recovery(per_session),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "macaque_pfc_microstimulation_recovery_latency.json").write_text(json.dumps(_json_safe(output), indent=2))
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["macaque_pfc_microstimulation_recovery_latency"] = output
    print(json.dumps({"n_sessions_complete": len(complete), "pooled": _json_safe(output["pooled"])}, indent=2))


if __name__ == "__main__":
    main()
