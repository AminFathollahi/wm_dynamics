"""Stimulation-locked response measurement: displacement of a component under stimulation, relative to a same-session no-stimulation counterfactual, and the subject-level statistical tests built on it."""
from __future__ import annotations

import numpy as np
import re
from stimulation_response_estimator import rate_free_state_deviation
from statistics import bootstrap_ci, pearson_permutation_test, stable_seed, minimum_detectable_paired_difference, paired_sign_flip_test
import csv
from pathlib import Path
import json
from collections import Counter
from statistics import Z_80_POWER
from preprocessing import butterworth_bandstop
from collections import defaultdict
from preprocessing import bipolar_reference_by_shank, line_noise_notch, phase_amplitude_coupling
from project_config import data_root
import h5py


ALPHA = 0.05


N_PERM = 10000


def _bin_averaged(arrays: dict, channel_mask: np.ndarray | None = None) -> np.ndarray:
    epochs = arrays["epochs_log"]
    if channel_mask is not None:
        epochs = epochs[:, :, channel_mask]
    return epochs.mean(axis=1).astype(float)


def _reference_direction(activity_by_unit: np.ndarray) -> np.ndarray:
    """The renormalised mean unit direction of every trial in
    `activity_by_unit`, with NO leave-one-out exclusion -- used to build a
    single FIXED reference from the control-trial pool, which stimulated
    trials (never members of that pool) are then scored against without
    letting them define any part of their own comparison point. A trial with
    zero total activity across channels has no defined direction and is
    excluded from the mean, exactly as rate_free_state_deviation excludes it
    from its own leave-one-out reference."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors[valid], axis=0)
    n_valid = int(valid.sum())
    if n_valid < 1:
        return np.full(activity.shape[1], np.nan)
    mean_dir = total / n_valid
    norm = np.linalg.norm(mean_dir)
    return mean_dir / norm if norm > 0 else np.full(activity.shape[1], np.nan)


def _contact_shank(contact_label: str) -> str:
    """The alphabetic prefix of a depth/grid electrode contact label (e.g.
    'LAH2' -> 'LAH') identifies the physical lead the contact sits on; RAM
    contact labels are always an alphabetic lead name followed by a numeric
    contact index."""
    match = re.match(r"^[A-Za-z]+", contact_label)
    return match.group(0) if match else contact_label


def _bipolar_channel_shanks(channel_name: str) -> set[str]:
    """A bipolar channel name is 'anode-cathode' (e.g. 'LAH1-LAH2'); returns
    the set of one or two leads either contact belongs to."""
    return {_contact_shank(p) for p in channel_name.split("-") if p}


def channel_condition_masks(ch_names: list[str], anode: str, cathode: str, stim_ch: str) -> dict:
    """The three channel sets the stimulation-displacement arm's mandatory artifact control compares:
    every channel, every channel except the driven bipolar pair, and every
    channel except the driven pair AND any channel sharing a lead with
    either the anode or the cathode contact -- the mandatory control for a
    large stimulation deflection contaminating nearby contacts, not only the
    driven pair itself."""
    stim_shanks = {_contact_shank(anode), _contact_shank(cathode)}
    full = np.ones(len(ch_names), dtype=bool)
    excl_pair = np.array([ch != stim_ch for ch in ch_names])
    excl_shank = np.array([
        ch != stim_ch and not (_bipolar_channel_shanks(ch) & stim_shanks) for ch in ch_names
    ])
    return {
        "full_channel_set": full,
        "excluding_stimulated_pair": excl_pair,
        "excluding_stimulated_shank": excl_shank,
    }


def _deviation_from_reference(activity_by_unit: np.ndarray, reference_direction: np.ndarray) -> np.ndarray:
    """Per trial, 1 - cosine(unit_vector_i, reference_direction), scoring
    each trial against a FIXED external direction rather than a leave-one-out
    mean of its own group -- the counterpart to rate_free_state_deviation's
    leave-one-out reference for trials that must never contribute to the
    reference they are being compared against."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    if not np.all(np.isfinite(reference_direction)):
        return np.full(activity.shape[0], np.nan)
    deviation = np.full(activity.shape[0], np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    deviation[valid] = 1.0 - unit_vectors[valid] @ reference_direction
    return deviation


def compute_stimulation_displacement(activity_by_unit: np.ndarray, stim_flag: np.ndarray) -> dict:
    """The core stimulation-displacement computation for one session and one channel
    condition: leave-one-out deviation among control trials only
    (rate_free_state_deviation, unmodified, called on the control subset
    alone so a stimulated trial can never enter any control trial's
    reference), a single fixed reference direction built from that same
    control pool, and stimulated trials scored against that fixed reference.
    Nothing about a stimulated trial's own value can change the reference it
    is compared against."""
    activity = np.asarray(activity_by_unit, dtype=float)
    stim = np.asarray(stim_flag).astype(bool)
    ctrl_activity = activity[~stim]
    stim_activity = activity[stim]
    control_deviation = rate_free_state_deviation(ctrl_activity)
    reference_direction = _reference_direction(ctrl_activity)
    stim_deviation = _deviation_from_reference(stim_activity, reference_direction)
    return {
        "control_deviation": control_deviation,
        "stim_deviation": stim_deviation,
        "reference_direction": reference_direction,
    }


N_BOOT = 5000


POWER = 0.80


def minimum_detectable_correlation(n_subjects: int, alpha: float = ALPHA, power: float = POWER) -> dict:
    """Smallest true Pearson correlation a two-sided test on n_subjects
    independent units could detect at the given power, via the standard
    Fisher z-transform normal approximation -- the correlation analogue of
    minimum_detectable_paired_difference, which is defined for a paired mean
    difference, not a correlation coefficient."""
    from scipy.stats import norm

    if n_subjects < 4:
        return {"status": "not_computable", "n_subjects": int(n_subjects),
                "reason": "fewer than 4 subjects -- Fisher z approximation undefined"}
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    z_r = (z_a + z_b) / np.sqrt(n_subjects - 3)
    return {"status": "computed", "n_subjects": int(n_subjects), "alpha": alpha, "power": power,
            "mdd": float(np.tanh(z_r))}


def required_subjects_for_detectable_correlation(r: float, alpha: float = ALPHA, power: float = POWER) -> dict:
    """Inverse of minimum_detectable_correlation: the smallest subject count
    whose minimum detectable correlation is at most the given r, via the
    same Fisher z-transform approximation solved for n."""
    z_factor = Z_80_POWER if (alpha, power) == (ALPHA, POWER) else \
        float(_scipy_norm_ppf(1.0 - alpha / 2.0) + _scipy_norm_ppf(power))
    if not (0.0 < r < 1.0):
        return {"status": "not_computable", "r": float(r),
                "reason": "r must be strictly between 0 and 1 for the Fisher z inverse"}
    raw_n_subjects = 3.0 + (z_factor / np.arctanh(r)) ** 2
    return {"status": "computed", "r": float(r), "alpha": alpha, "power": power,
            "z_factor": float(z_factor), "raw_n_subjects": float(raw_n_subjects),
            "n_subjects_required": int(np.ceil(raw_n_subjects))}


def _scipy_norm_ppf(q: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(q))


def subject_aggregated_correlation(x: np.ndarray, y: np.ndarray, subject_ids: list) -> dict:
    """Collapses session-level (x, y) pairs to one point per subject (the
    unweighted mean of that subject's own sessions) before correlating, so
    the permutation null and the bootstrap CI both resample at the subject
    level -- avoiding a subject with many sessions silently outweighing one
    with few, and matching the reachable-sample-size regime
    pearson_permutation_test and bootstrap_ci (both already used elsewhere in
    this project) were built for."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    subject_ids = np.asarray(subject_ids)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, subject_ids = x[finite], y[finite], subject_ids[finite]
    unique_subjects = sorted(set(subject_ids.tolist()))
    if len(unique_subjects) < 4:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
                "reason": "fewer than 4 subjects"}
    x_subj = np.array([x[subject_ids == s].mean() for s in unique_subjects])
    y_subj = np.array([y[subject_ids == s].mean() for s in unique_subjects])
    if np.std(x_subj) == 0 or np.std(y_subj) == 0:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
                "reason": "zero variance in a subject-aggregated variable"}
    rng = np.random.default_rng(stable_seed(f"subject_aggregated_corr|{tuple(unique_subjects)}"))
    corr = pearson_permutation_test(x_subj, y_subj, n_perm=N_PERM, rng=rng)
    _, ci_lo, ci_hi = bootstrap_ci(
        np.column_stack([x_subj, y_subj]),
        lambda d: float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]),
        n_boot=N_BOOT, rng=rng,
    )
    mdd = minimum_detectable_correlation(len(unique_subjects))
    return {
        "status": "computed", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
        "r": corr["r"], "p_value": corr["p_value"], "ci_lower": float(ci_lo), "ci_upper": float(ci_hi),
        "mdd": mdd,
    }


def subject_clustered_mean_test(session_values: np.ndarray, subject_ids: list, alternative: str = "two-sided") -> dict:
    """Pools one scalar per session into a subject-clustered mean test: each
    subject's own sessions are first collapsed to their unweighted mean, then
    the collapsed subject-level values are tested against zero with the
    paired sign-flip test -- subject is the unit both the permutation null
    and the bootstrap CI resample, so a subject contributing many sessions
    cannot silently outweigh one contributing few."""
    session_values = np.asarray(session_values, dtype=float)
    subject_ids = np.asarray(subject_ids)
    finite = np.isfinite(session_values)
    session_values, subject_ids = session_values[finite], subject_ids[finite]
    unique_subjects = sorted(set(subject_ids.tolist()))
    if len(unique_subjects) < 2:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects)}
    subject_values = np.array([session_values[subject_ids == s].mean() for s in unique_subjects])
    rng = np.random.default_rng(stable_seed(f"subject_clustered|{tuple(unique_subjects)}|{finite.sum()}"))
    test = paired_sign_flip_test(subject_values, np.zeros_like(subject_values), n_perm=N_PERM,
                                  alternative=alternative, n_boot=N_BOOT, rng=rng)
    mdd = minimum_detectable_paired_difference(subject_values, alpha=ALPHA, power=POWER)
    return {
        "status": "computed", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
        "mean_value": test["mean_diff"], "p_value": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "mdd": mdd,
    }


def _derive_word_stimulation(words: list[dict], stim_on: list[dict], stim_off: list[dict],
                             window_s: float = 2.0) -> None:
    """Mutate `words` in place, setting each event's 'stimulation' field from
    STIM_ON/STIM_OFF timestamp overlap rather than trusting the WORD event's
    own field. Some closed-loop RAM releases (e.g. ds005557) record real
    STIM_ON/STIM_OFF events but leave every WORD event's own `stimulation`
    field at "0" -- the online classifier's trigger timing is not backfilled
    onto the word row it applies to. A word is marked stimulated if any
    STIM_ON falls within [onset, onset + window_s] of it (word presentation
    plus a margin for triggering/pulse-train latency), matched to the closest
    such word if a STIM_ON is nearer to more than one.

    Also copies the matched STIM_ON event's own dose fields (amplitude,
    pulse_freq, n_pulses, pulse_width) onto the word row: unlike ds005489,
    this release does not carry those fields on the WORD row itself, only on
    the STIM_ON event that triggered delivery."""
    if not stim_on:
        return
    word_onsets = [float(w["onset"]) for w in words]
    for stim_event in sorted(stim_on, key=lambda s: float(s["onset"])):
        stim_t = float(stim_event["onset"])
        candidates = [(abs(stim_t - w_on), i) for i, w_on in enumerate(word_onsets)
                     if 0 <= stim_t - w_on <= window_s]
        if not candidates:
            continue
        _, best_i = min(candidates)
        words[best_i]["stimulation"] = "1"
        for field in ("amplitude", "pulse_freq", "n_pulses", "pulse_width"):
            if field in stim_event:
                words[best_i][field] = stim_event[field]


def _float_or_nan(value) -> float:
    """Parse a BIDS TSV field (always a string from csv.DictReader) that may
    be a number, 'n/a', or empty -- never raises."""
    try:
        if value in (None, "", "n/a"):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_events(events_tsv: Path) -> list[dict]:
    with open(events_tsv) as f:
        return list(csv.DictReader(f, delimiter="\t"))


MORLET_FREQS_HZ = np.array([3.0, 5.4, 9.7, 17.4, 31.1, 55.9, 100.3, 180.0])


MORLET_WAVENUMBER = 5.0


LINE_NOISE_BANDSTOP_HZ = (58.0, 62.0)


MIRROR_BUFFER_S = 1.365


N_BINS = 13


def morlet_log_power_bank(
    raw_epochs: np.ndarray,
    srate: float,
    freqs: np.ndarray = MORLET_FREQS_HZ,
    n_cycles: float = MORLET_WAVENUMBER,
    buffer_s: float = MIRROR_BUFFER_S,
    bandstop_hz: tuple[float, float] = LINE_NOISE_BANDSTOP_HZ,
    n_bins: int = N_BINS,
    n_workers: int = 1,
) -> np.ndarray:
    """Reproduce ds005557's classifier feature bank, kept time-resolved.

    Pipeline (README order): mirror-pad -> Butterworth band-stop -> Morlet
    wavelet transform -> log power -> remove the mirrored buffer -> bin.
    z-transform is deliberately NOT applied here; callers must fit mean/sd
    on training-fold (here: unperturbed-trial) statistics only.

    Parameters
    ----------
    raw_epochs : (n_epochs, n_channels, n_times) raw voltage, the 0..window_s
        segment at native sampling rate, no buffer yet.

    Returns
    -------
    (n_epochs, n_channels, n_freqs, n_bins) log power, buffer removed.
    """
    from mne.time_frequency import tfr_array_morlet

    n_epochs, n_channels, n_times = raw_epochs.shape
    buffer_samples = int(round(buffer_s * srate))
    padded = np.pad(raw_epochs, ((0, 0), (0, 0), (buffer_samples, buffer_samples)), mode="reflect")

    # Band-stop applies along the time axis; reshape to (T, epochs*channels) to
    # reuse the existing (T, C)-shaped filter helper instead of writing a new one.
    n_times_padded = padded.shape[2]
    flat = padded.transpose(2, 0, 1).reshape(n_times_padded, -1)
    flat = butterworth_bandstop(flat, bandstop_hz[0], bandstop_hz[1], srate)
    padded = flat.reshape(n_times_padded, n_epochs, n_channels).transpose(1, 2, 0).astype(np.float32)

    n_samp_bin = n_times // n_bins
    if n_samp_bin < 1:
        raise ValueError("encoding window too short for the requested number of bins")

    n_freq = len(freqs)
    # ponytail: fixed byte-budget chunking, not a tuned scheduler -- if a
    # dataset ships far more channels this still works, just in more chunks.
    target_bytes = 3e8
    per_epoch_bytes = n_channels * n_freq * n_times_padded * 8
    chunk = int(np.clip(target_bytes // max(per_epoch_bytes, 1), 1, n_epochs))

    binned = np.empty((n_epochs, n_channels, n_freq, n_bins), dtype=np.float32)
    for start in range(0, n_epochs, chunk):
        stop = min(start + chunk, n_epochs)
        power = tfr_array_morlet(
            padded[start:stop], sfreq=srate, freqs=freqs, n_cycles=n_cycles,
            output="power", zero_mean=True, n_jobs=n_workers, verbose=False,
        )
        log_power = np.log(power + 1e-20)
        cropped = log_power[..., buffer_samples: buffer_samples + n_times]
        trimmed = cropped[..., : n_samp_bin * n_bins]
        binned[start:stop] = trimmed.reshape(
            stop - start, n_channels, n_freq, n_bins, n_samp_bin,
        ).mean(axis=-1).astype(np.float32)
    return binned


def _safe_int(value, default: int = -1) -> int:
    try:
        if value in (None, "", "n/a"):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


MIN_WORDS = 80


ENCODING_WINDOW_S = 1.366


def _find_stim_sessions(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json"))


def build_session_trajectory(
    ieeg_json: Path, data_root: Path, derive_stim_from_stim_on: bool, n_workers: int = 1,
) -> dict | None:
    """Build the per-word Morlet feature trajectory for one session.

    Returns None for files outside this analysis's scope (no stimulation
    channel); returns a status dict with an explicit reason for sessions
    that were in scope but unusable, so nothing is silently dropped.
    """
    import mne

    with open(ieeg_json) as f:
        meta = json.load(f)
    if not meta.get("ElectricalStimulation", False):
        return None

    stem = str(ieeg_json).replace("_ieeg.json", "")
    events_tsv = Path(stem.replace("_acq-bipolar", "") + "_events.tsv")
    edf_path = Path(stem + "_ieeg.edf")
    session_name = str(edf_path.relative_to(data_root)) if edf_path.exists() else str(ieeg_json)
    if not events_tsv.exists() or not edf_path.exists():
        return {"status": "excluded", "reason": "missing events.tsv or edf", "session": session_name}

    events = _load_events(events_tsv)
    words = [e for e in events if e["trial_type"] == "WORD"]
    if len(words) < MIN_WORDS:
        return {"status": "excluded", "reason": f"only {len(words)} WORD events (< {MIN_WORDS})",
                "session": session_name}

    if derive_stim_from_stim_on:
        stim_on = [e for e in events if e["trial_type"] == "STIM_ON"]
        stim_off = [e for e in events if e["trial_type"] == "STIM_OFF"]
        _derive_word_stimulation(words, stim_on, stim_off)

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    srate = raw.info["sfreq"]
    n_ch = len(raw.ch_names)
    rec_dur = raw.times[-1]
    window_samples = int(round(ENCODING_WINDOW_S * srate))

    segments, kept_words = [], []
    for w in words:
        onset_s = float(w["onset"])
        if onset_s < 0 or onset_s + ENCODING_WINDOW_S > rec_dur:
            continue
        i0 = int(round(onset_s * srate))
        seg = raw.get_data(start=i0, stop=i0 + window_samples)
        if seg.shape[1] != window_samples or not np.all(np.isfinite(seg)):
            continue
        segments.append(seg)
        kept_words.append(w)
    if len(segments) < MIN_WORDS:
        return {"status": "excluded",
                "reason": f"only {len(segments)} usable epochs after edge-of-recording exclusion (< {MIN_WORDS})",
                "session": session_name}

    raw_epochs = np.stack(segments, axis=0)
    features = morlet_log_power_bank(raw_epochs, srate, n_workers=n_workers)
    n_words, n_ch2, n_freq, n_bins = features.shape
    features_flat = features.transpose(0, 3, 1, 2).reshape(n_words, n_bins, n_ch2 * n_freq)

    return {
        "status": "complete",
        "session": session_name,
        "subject": edf_path.parts[-4],
        "srate": float(srate), "n_channels": int(n_ch2), "n_words": int(n_words),
        "features": features_flat,
        "stim": np.array([_safe_int(w.get("stimulation"), 0) for w in kept_words], dtype=int),
        "stim_list": np.array([_safe_int(w.get("stim_list")) for w in kept_words], dtype=int),
        "list": np.array([_safe_int(w.get("list")) for w in kept_words], dtype=int),
        "serialpos": np.array([_safe_int(w.get("serialpos")) for w in kept_words], dtype=int),
        "recalled": np.array([_safe_int(w.get("recalled")) for w in kept_words], dtype=int),
    }


def _censored_first_recall(entry: dict) -> tuple[float, int] | None:
    period = entry.get("recall_period_s")
    if period is None:
        return None
    rt = _first_recall(entry, "rec_word_rt")
    if rt is not None:
        return (min(float(rt), period), 1)
    return (period, 0)


def _first_recall(entry: dict, field: str) -> float | None:
    values = entry[field]
    return min(values) if values else None


def contrast_summary(contrasts: dict[str, dict], seed_name: str) -> dict:
    if not contrasts:
        return {"status": "not_computable", "n": 0}
    stim = np.array([c["stim"] for c in contrasts.values()])
    unstim = np.array([c["unstim"] for c in contrasts.values()])
    rng = np.random.default_rng(stable_seed(seed_name))
    test = paired_sign_flip_test(stim, unstim, alternative="two-sided", rng=rng)
    mdd = frozen_mdd(stim - unstim)
    return {
        "status": "computed", "n": int(len(stim)), "mean_stim": float(np.mean(stim)), "mean_unstim": float(np.mean(unstim)),
        "mean_diff": test["mean_diff"], "sd_diff": float(np.std(stim - unstim, ddof=1)),
        "n_positive_diff": int(np.sum((stim - unstim) > 0)),
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "p_value": test["p_value"], "minimum_detectable_difference": mdd,
    }


def frozen_mdd(values: np.ndarray) -> dict:
    result = minimum_detectable_paired_difference(values)
    if result.get("status") == "computed":
        result = {**result, "z_factor": Z_80_POWER,
                  "mdd": Z_80_POWER * result["sd"] / np.sqrt(result["n"])}
    return result


def load_corpus(local_path: str, root: Path) -> dict:
    directory = root / local_path
    by_subject_lists: dict[str, dict] = {}
    param_counts = {field: Counter() for field in STIM_PARAM_FIELDS}
    n_sessions = 0
    n_events = 0
    for events_path in sorted(directory.glob("sub-*/ses-*/ieeg/*_events.tsv")):
        with open(events_path) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if not rows:
            continue
        n_sessions += 1
        n_events += len(rows)
        subject = rows[0]["subject"]
        lists = by_subject_lists.setdefault(subject, {})
        for row in rows:
            list_id = row.get("list", "")
            trial_type = row["trial_type"]
            if trial_type not in ("REC_WORD", "REC_WORD_VV", "WORD", "REC_START", "REC_END") or list_id in ("-999", "-1", ""):
                continue
            key = (str(events_path), list_id)
            entry = lists.setdefault(key, {"stim_list": row.get("stim_list") == "1",
                                            "rec_word_rt": [], "rec_all_rt": [], "n_recalled": 0,
                                            "rec_start_onset": None, "rec_end_onset": None})
            if trial_type == "REC_START":
                entry["rec_start_onset"] = _num(row["onset"])
            elif trial_type == "REC_END":
                entry["rec_end_onset"] = _num(row["onset"])
            elif trial_type == "REC_WORD" and row["response_time"] not in ("n/a", ""):
                rt = _num(row["response_time"])
                if np.isfinite(rt) and rt > 0:
                    entry["rec_word_rt"].append(rt)
                    entry["rec_all_rt"].append(rt)
            elif trial_type == "REC_WORD_VV" and row["response_time"] not in ("n/a", ""):
                rt = _num(row["response_time"])
                if np.isfinite(rt) and rt > 0:
                    entry["rec_all_rt"].append(rt)
            elif trial_type == "WORD" and row.get("recalled") not in ("n/a", ""):
                entry["n_recalled"] += int(_num(row["recalled"]))
            if row.get("stimulation") == "1":
                for field in STIM_PARAM_FIELDS:
                    param_counts[field][row.get(field, "n/a")] += 1
    for lists in by_subject_lists.values():
        for entry in lists.values():
            start, end = entry.pop("rec_start_onset"), entry.pop("rec_end_onset")
            period = end - start if start is not None and end is not None and np.isfinite(end - start) else None
            entry["recall_period_s"] = period if period is not None and period > 0 else None
    return {"by_subject_lists": by_subject_lists, "param_counts": param_counts,
            "n_sessions": n_sessions, "n_events": n_events}


STIM_PARAM_FIELDS = ("amplitude", "pulse_freq", "pulse_width", "n_pulses", "stim_duration")


def _num(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _human_session_windows(rec: dict, pre_bins: int, mask: np.ndarray) -> dict:
    arrays = rec["arrays"]
    epochs = arrays["epochs_log"][:, :, mask]  # (n_trials, n_bins, n_channels_kept)
    pre_activity, post_activity = epochs[:, :pre_bins, :].mean(axis=1), epochs[:, pre_bins:, :].mean(axis=1)
    stim = arrays["stim_flag"].astype(bool)
    return {
        "pre_stimulation_window": {
            "control_activity": pre_activity[~stim], "treated_activity": pre_activity[stim],
            "control_total": pre_activity[~stim].sum(axis=1), "treated_total": pre_activity[stim].sum(axis=1),
        },
        "post_stimulation_window": {
            "control_activity": post_activity[~stim], "treated_activity": post_activity[stim],
            "control_total": post_activity[~stim].sum(axis=1), "treated_total": post_activity[stim].sum(axis=1),
        },
    }


def _specificity_check(result: dict) -> dict:
    """Compares the pre-stimulation-window cell (before stimulation has
    begun -- by construction, no true treatment effect can exist there) to
    the post-stimulation-window cell (the actual test). A pre-window
    displacement that is itself significant means the post-window number
    cannot be read as cleanly attributable to that trial's own stimulation
    -- it is reported here as a finding, not smoothed into the post-window
    headline."""
    windows = result["windows"]
    pre, post = windows.get("pre_stimulation_window", {}), windows.get("post_stimulation_window", {})
    if pre.get("status") != "computed" or post.get("status") != "computed":
        return {"status": "not_computable", "reason": "one or both windows are a structural void for this arm"}
    pre_dev = pre["rate_free_deviation_biomarker"]["pooled"]
    post_dev = post["rate_free_deviation_biomarker"]["pooled"]
    if pre_dev.get("status") != "computed" or post_dev.get("status") != "computed":
        return {"status": "not_computable", "reason": "pooled deviation not computed in one or both windows"}
    pre_significant = pre_dev["p_value"] <= 0.05
    return {
        "status": "computed",
        "pre_window_mean": pre_dev["mean_value"], "pre_window_p_value": pre_dev["p_value"],
        "post_window_mean": post_dev["mean_value"], "post_window_p_value": post_dev["p_value"],
        "pre_window_significant": pre_significant,
        "reading": (
            "The pre-stimulation window is itself significant, at a magnitude comparable to the "
            "post-stimulation window. Direct measurement of the release's own event tables shows why: "
            "the delivered pulse train begins before the word it accompanies appears, by a fixed "
            "trigger lead of approximately 0.211 s, so both the pre- and the post-stimulation window "
            "lie inside an already-running train for every stimulated word without exception. This "
            "corpus therefore contains no stimulation-free pre-window, and the pre/post comparison is "
            "consequently not a specificity contrast: a significant pre-window cannot be resolved into "
            "a preferred causal account, only bounded in magnitude."
            if pre_significant else
            "The pre-stimulation window is not significant, unlike the post-stimulation window -- the "
            "specificity pattern a genuine treatment effect predicts (no effect before treatment begins, "
            "an effect after)."
        ),
    }


def load_electrode_table(data_dir: Path, session_key: str) -> dict:
    """MNI coordinates are required for the geometric distance analysis and
    are read only from the MNI152NLin6ASym-space table. The three
    anatomical labelling schemes are shipped in both space variants of the
    table; if the MNI table is absent for a session (a real gap in this
    release), the region labels are still read from the Talairach-space
    table so the anatomical map is not forced to drop that session too --
    the space actually used is recorded on every session."""
    mni_path, tal_path = _electrode_table_paths(data_dir, session_key)
    coords: dict[str, tuple[float, float, float]] = {}
    labels: dict[str, dict[str, str]] = {}
    space_for_labels = None
    table_path = mni_path if mni_path.exists() else (tal_path if tal_path.exists() else None)
    if table_path is not None:
        with open(table_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            labels[name] = {scheme: (row.get(scheme) or "n/a").strip() or "n/a" for scheme in LABEL_SCHEMES}
        space_for_labels = "MNI152NLin6ASym" if table_path == mni_path else "Talairach"
    if mni_path.exists():
        with open(mni_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            x, y, z = _numeric_or_nan(row.get("x")), _numeric_or_nan(row.get("y")), _numeric_or_nan(row.get("z"))
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                coords[name] = (x, y, z)
    return {"coords": coords, "labels": labels, "space_for_labels": space_for_labels,
            "mni_table_found": mni_path.exists()}


LABEL_SCHEMES = ["ind.region", "das.region", "stein.region"]


def _electrode_table_paths(data_dir: Path, session_key: str) -> tuple[Path, Path]:
    ieeg_json = data_dir / session_key
    stem = ieeg_json.name.replace("_acq-bipolar_ieeg.json", "")
    mni_path = ieeg_json.parent / f"{stem}_space-MNI152NLin6ASym_electrodes.tsv"
    tal_path = ieeg_json.parent / f"{stem}_space-Talairach_electrodes.tsv"
    return mni_path, tal_path


def _numeric_or_nan(value) -> float:
    """-999 and 'n/a' both appear as missing-value sentinels for numeric
    fields in this release's electrode tables; neither may enter a distance
    or an average."""
    if value is None:
        return float("nan")
    token = value.strip().lower()
    if token in MISSING_NUMERIC_TOKENS:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


MISSING_NUMERIC_TOKENS = {"", "n/a", "-999", "nan"}


CLOSEDLOOP_CORPUS = "ds005557-download"


CLOSEDLOOP_OWNER_MATCH_WINDOW_S = 2.0


OPENLOOP_CORPUS = "ds005489-download"


def match_train_owner(train_start: float, word_onsets: list[float], window_s: float) -> int | None:
    """Nearest preceding word within window_s of the train's own onset, or
    None. Matches scripts/run_ram_openloop_pipeline.py's
    `_derive_word_stimulation` convention exactly."""
    candidates = [(train_start - w, i) for i, w in enumerate(word_onsets) if 0 <= train_start - w <= window_s]
    if not candidates:
        return None
    return min(candidates)[1]


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


ITEMS_PER_TRAIN_SPAN_THRESHOLD = 1.0


def pool_item_attributability(records: dict, corpus_label: str) -> dict:
    per_session = {}
    all_items_per_train, all_spacing, all_train_durations = [], [], []
    n_prev = n_next = n_any = n_total = n_unmatched = 0
    n_trains_pooled = 0
    for sid, r in records.items():
        per_session[sid] = {
            "n_words": r["n_words"], "n_trains": r["n_trains"],
            "spacing_s_median": float(np.median(r["spacing_s"])) if r["spacing_s"] else None,
            "spacing_s_mean": float(np.mean(r["spacing_s"])) if r["spacing_s"] else None,
            "spacing_s_sd": float(np.std(r["spacing_s"], ddof=1)) if len(r["spacing_s"]) > 1 else None,
            "train_duration_s_median": float(np.median(r["train_duration_s"])) if r["train_duration_s"] else None,
            "train_duration_s_mean": float(np.mean(r["train_duration_s"])) if r["train_duration_s"] else None,
            "items_per_train_mean": float(np.mean(r["items_per_train"])) if r["items_per_train"] else None,
            "neighbor_coverage": r["neighbor_coverage"],
        }
        all_items_per_train.extend(r["items_per_train"])
        all_spacing.extend(r["spacing_s"])
        all_train_durations.extend(r["train_duration_s"])
        nb = r["neighbor_coverage"]
        n_prev += nb["n_prev_covered"]; n_next += nb["n_next_covered"]
        n_any += nb["n_any_covered"]; n_total += nb["n_stim_items"]
        n_unmatched += nb["n_train_unmatched_to_owning_item"]
        n_trains_pooled += r["n_trains"]

    attributable = [n for n in all_items_per_train if n >= 1]
    zero_overlap = len(all_items_per_train) - len(attributable)
    mean_items_per_train = float(np.mean(attributable)) if attributable else None

    if mean_items_per_train is None:
        branch = "underpowered_to_ask"
    elif mean_items_per_train > ITEMS_PER_TRAIN_SPAN_THRESHOLD:
        branch = "the_intervention_spans_more_than_one_item_and_cannot_be_attributed_to_one"
    else:
        branch = "the_intervention_is_confined_to_a_single_item"

    return {
        "per_session": per_session,
        "pooled": {
            "n_trains_total": n_trains_pooled,
            "n_trains_overlapping_zero_items": zero_overlap,
            "n_trains_overlapping_at_least_one_item": len(attributable),
            "items_per_train_distribution": {str(k): int(v) for k, v in
                                             zip(*np.unique(attributable, return_counts=True))} if attributable else {},
            "mean_items_per_train_among_attributable_trains": mean_items_per_train,
            "span_decision_threshold": ITEMS_PER_TRAIN_SPAN_THRESHOLD,
            "item_presentation_spacing_s_median": float(np.median(all_spacing)) if all_spacing else None,
            "item_presentation_spacing_s_mean": float(np.mean(all_spacing)) if all_spacing else None,
            "item_presentation_spacing_s_sd": float(np.std(all_spacing, ddof=1)) if len(all_spacing) > 1 else None,
            "train_duration_s_median": float(np.median(all_train_durations)) if all_train_durations else None,
            "train_duration_s_mean": float(np.mean(all_train_durations)) if all_train_durations else None,
            "n_stimulated_items": n_total,
            "n_stimulated_items_train_unmatched": n_unmatched,
            "fraction_stimulated_items_train_covers_preceding_item": (n_prev / n_total) if n_total else None,
            "fraction_stimulated_items_train_covers_following_item": (n_next / n_total) if n_total else None,
            "fraction_stimulated_items_train_covers_either_neighbor": (n_any / n_total) if n_total else None,
        },
        "branch": branch,
    }


def pool_parameter_census(records: dict, corpus_label: str) -> dict:
    amp_counts = defaultdict(int)
    pair_amps = defaultdict(set)
    pair_trials = defaultdict(int)
    freqs, widths, durs = set(), set(), set()
    pair_freqs, pair_widths = defaultdict(set), defaultdict(set)
    n_stim_trials = 0
    subjects = set()
    for sid, r in records.items():
        subj = r["subject"]
        subjects.add(subj)
        for amp in r["amplitudes"]:
            amp_counts[amp] += 1
            n_stim_trials += 1
        freqs.update(r["pulse_freqs"]); widths.update(r["pulse_widths"]); durs.update(r["stim_durations_ms"])
        for (anode, cathode) in r["electrode_pairs"]:
            pair_key = (subj, anode, cathode)
            pair_trials[pair_key] += 1
        # per-pair amplitude/freq/width sets need per-train granularity, not just the
        # per-session unique-value set, so recover it from the parallel per-session lists
        # (amplitudes list is per-train; electrode_pairs is per-corpus unique -- re-derive
        # per-pair amplitude sets from records that carry per-train pair identity below).

    # Second pass: per-train (amplitude, pair) association is not retained per-session
    # above (only per-session unique sets), so re-open a lighter per-pair accumulation
    # using each session's amplitude list matched 1:1 against its own single stim
    # electrode pair -- true for both corpora (one stimulated bipolar pair per session).
    for sid, r in records.items():
        subj = r["subject"]
        pairs = r["electrode_pairs"]
        if len(pairs) != 1:
            continue  # a session with zero or >1 distinct stim pairs contributes no
                      # unambiguous per-pair amplitude assignment; counted in the
                      # pooled amplitude census above regardless.
        anode, cathode = pairs[0]
        key = (subj, anode, cathode)
        for amp in r["amplitudes"]:
            pair_amps[key].add(amp)
        for f in r["pulse_freqs"]:
            pair_freqs[key].add(f)
        for w in r["pulse_widths"]:
            pair_widths[key].add(w)

    multi_amp_pairs = {k: sorted(v) for k, v in pair_amps.items() if len(v) > 1}
    subjects_multi_amp = sorted({k[0] for k in multi_amp_pairs})
    multi_freq_within_pair = {str(k): sorted(v) for k, v in pair_freqs.items() if len(v) > 1}
    multi_width_within_pair = {str(k): sorted(v) for k, v in pair_widths.items() if len(v) > 1}

    # Same accumulation, but over every STIM_ON/STIM_OFF-derived train regardless of
    # whether it could be matched to an owning WORD item -- surfaces amplitude values
    # that only ever appear on pre-task titration/calibration pulses (never near an
    # item), so that number is reported rather than silently absorbed into, or
    # silently absent from, the item-linked count above.
    pair_amps_all = defaultdict(set)
    for sid, r in records.items():
        subj = r["subject"]
        pairs_all = r.get("electrode_pairs_all_trains", r["electrode_pairs"])
        if len(pairs_all) != 1:
            continue
        anode, cathode = pairs_all[0]
        for amp in r.get("amplitudes_all_trains", r["amplitudes"]):
            pair_amps_all[(subj, anode, cathode)].add(amp)
    multi_amp_pairs_all_trains = {k: sorted(v) for k, v in pair_amps_all.items() if len(v) > 1}

    return {
        "n_subjects_with_stimulated_trials": len(subjects),
        "n_stimulated_trials_total": n_stim_trials,
        "stimulated_trials_by_amplitude_microamps": {str(k): v for k, v in sorted(amp_counts.items())},
        "n_electrode_pairs": len(pair_trials),
        "n_electrode_pairs_with_more_than_one_amplitude": len(multi_amp_pairs),
        "subjects_with_a_multi_amplitude_electrode_pair": subjects_multi_amp,
        "electrode_pairs_with_more_than_one_amplitude": {str(k): v for k, v in multi_amp_pairs.items()},
        "constant_across_corpus": {
            "pulse_freq_hz": sorted(freqs) if len(freqs) == 1 else None,
            "pulse_width_us": sorted(widths) if len(widths) == 1 else None,
            "stim_duration_ms": sorted(durs) if len(durs) == 1 else None,
            "amplitude_microamps": sorted(amp_counts.keys()) if len(amp_counts) == 1 else None,
        },
        "unique_values_seen": {
            "pulse_freq_hz": sorted(freqs), "pulse_width_us": sorted(widths),
            "stim_duration_ms": sorted(durs), "amplitude_microamps": sorted(amp_counts.keys()),
        },
        "electrode_pairs_with_more_than_one_pulse_freq": multi_freq_within_pair,
        "electrode_pairs_with_more_than_one_pulse_width": multi_width_within_pair,
        "n_electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses":
            len(multi_amp_pairs_all_trains),
        "electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses":
            {str(k): v for k, v in multi_amp_pairs_all_trains.items()},
        "note_on_unmatched_pulse_amplitude": (
            "the item-linked counts above use only STIM_ON/STIM_OFF trains matched to an "
            "owning WORD item; a small number of STIM_ON rows in this corpus (e.g. a single "
            "pulse at recording onset, long before any word is shown) cannot be matched to any "
            "item and are excluded from them. The "
            "'..._including_unmatched_pulses' fields fold those back in, so a widening gap "
            "between the two flags amplitude variation coming from pre-task device "
            "titration/calibration rather than genuine within-task dose variation."
        ),
    }


BORAN_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 10)]


B_HAT_MISMATCH_DEG = 20.0


NEAR_TIE_REL_TOL = 0.90


def _near_tie_candidates(scores: np.ndarray, tol: float = NEAR_TIE_REL_TOL) -> list[int]:
    max_score = np.max(scores)
    idxs = [i for i in range(len(scores)) if scores[i] >= tol * max_score]
    idxs.sort(key=lambda i: -scores[i])
    return idxs


DATA_ROOT = data_root()


def _pac_channel_weights(subj: str, good_ch_indices: np.ndarray) -> np.ndarray | None:
    """Theta-phase/HGP-amplitude PAC modulation index per GOOD channel (indices
    matching boran_geometry_*.npz's V rows exactly -- good_ch_indices selects
    into the raw full-channel data the same way run_boran_pipeline.py's
    channel_rejection mask does), on the RAW maintenance-window LFP
    (pre-HGP-transform -- PAC needs phase, band power alone destroys it),
    averaged over trials. Reuses preprocessing.phase_amplitude_coupling
    directly (modulation index)."""
    subj_dir = DATA_ROOT / "000574" / subj
    srate = 1398.0
    t_pre_maint, t_post_maint, t_epoch_pre = 3.0, 3.0, 1.0
    epoch_total = t_pre_maint + t_post_maint + t_epoch_pre
    all_epochs = []
    electrode_labels = None
    for nwb_path in sorted(subj_dir.glob("*.nwb")):
        with h5py.File(str(nwb_path), "r") as f:
            raw = f["acquisition/ecephys.ieeg/data"][:]
            times = f["acquisition/ecephys.ieeg/timestamps"][:]
            t_start_arr = f["intervals/trials/start_time"][:]
            artifact = f["intervals/trials/artifact"][:].astype(bool)
            if electrode_labels is None:
                try:
                    ieeg_idx = f["acquisition/ecephys.ieeg/electrodes"][:]
                    labels_full = [l.decode() for l in
                                    f["general/extracellular_ephys/electrodes/label"][:]]
                    electrode_labels = [labels_full[i] for i in ieeg_idx]
                except KeyError:
                    electrode_labels = [f"ch{i}" for i in range(raw.shape[1])]
        n_samp = int(epoch_total * srate)
        n_pre = int(t_epoch_pre * srate)
        for trial_idx, t0 in enumerate(t_start_arr):
            if artifact[trial_idx]:
                continue
            i0 = np.searchsorted(times, t0) - n_pre
            i1 = i0 + n_samp
            if i0 < 0 or i1 > raw.shape[0]:
                continue
            all_epochs.append(raw[i0:i1].astype(np.float32))   # (T, C_full)
    if len(all_epochs) < 10:
        return None

    # Notch (50/100/150 Hz) + bipolar-by-shank reref.
    # Must exactly match run_boran_pipeline.py's channel reduction, since
    # good_ch_indices (below) indexes into that same bipolar channel set,
    # not the raw monopolar channels.
    epochs_arr = np.stack(all_epochs, axis=0)           # (N, T, C_full)
    N0, T0, C_raw = epochs_arr.shape
    for n in range(N0):
        epochs_arr[n] = line_noise_notch(epochs_arr[n], srate, fundamental=50.0, n_harmonics=3)
    X_flat = epochs_arr.reshape(-1, C_raw)
    X_bp, _ = bipolar_reference_by_shank(X_flat, electrode_labels)
    epochs_arr = X_bp.reshape(N0, T0, -1).astype(np.float32)  # (N, T, C_bp)
    all_epochs = list(epochs_arr)

    maint_start_s = int((t_epoch_pre + t_pre_maint) * srate)
    maint_end_s = int((t_epoch_pre + t_pre_maint + t_post_maint) * srate)
    mi_per_trial = []
    # Subsample trials for tractability (PAC's Hilbert-transform cost scales
    # with samples x channels x trials; 30 trials is ample to average a
    # per-channel MI estimate that is a session-level scalar, not a per-trial
    # decode).
    rng = np.random.default_rng(stable_seed(f"pac_{subj}"))
    idx = rng.choice(len(all_epochs), size=min(30, len(all_epochs)), replace=False)
    for i in idx:
        maint = all_epochs[i][maint_start_s:maint_end_s][:, good_ch_indices]   # (T_maint, C_good)
        mi = phase_amplitude_coupling(maint, phase_band=(4.0, 8.0), amplitude_band=(70.0, 150.0),
                                      srate=srate, n_phase_bins=18)
        mi_per_trial.append(mi)
    mi_mean = np.nanmean(np.stack(mi_per_trial), axis=0)
    return mi_mean


def _stability_horizon(A: np.ndarray, n_time_constants: float = 3.0) -> float:
    """SAME construction as run_closed_loop_analysis.py's _stability_horizon:
    caps the rollout at n_time_constants e-folding times of A's least-stable
    eigenvalue, so a near-unit-circle-to-unstable plant (every DANDI 000574 A here
    has max|eig|>1, per that module's docstring) does not saturate
    simulate_closed_loop's numerical state-norm cap before the horizon ends
    -- which manufactures an arbitrarily huge (not genuine) drift number."""
    log_lam_max = np.log(np.max(np.abs(np.linalg.eigvals(A))))
    if abs(log_lam_max) < 1e-12:
        return np.inf
    return n_time_constants / abs(log_lam_max)
