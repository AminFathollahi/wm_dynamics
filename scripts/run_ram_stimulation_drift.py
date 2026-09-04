#!/usr/bin/env python3
"""RAM ds005489 (open-loop) and ds005557 (closed-loop) -- human intracranial
stimulation at encoding, analysed with the author-defined spectral feature
bank and this project's confined-drift model.

Both datasets are Kahana/DARPA RAM delayed free-recall releases with
electrical stimulation during list encoding. `scripts/run_ram_openloop_pipeline.py`
and `scripts/run_ram_closedloop_pipeline.py` fit an ad hoc high-gamma-power
+ DMD-eigenvector feature; this script replaces that feature with the
spectral decomposition the datasets' own README defines (ds005557's
"Classifier Details": bipolar montage, 58-62 Hz 4th-order Butterworth
band-stop, 8 log-spaced Morlet wavelets at wavenumber 5 across 3-180 Hz,
1365 ms mirrored buffers, log power, within-session z-transform), then fits
`src/drift_dynamics.py`'s confined-diffusion model on the resulting latent
trajectory instead of computing a single classifier-style decision scalar --
the drift model needs a trajectory across the encoding window, not the one
number the online classifier collapses it to. That is the only departure
from the released recipe; the spectral decomposition itself (filter,
wavelets, log power) is reproduced verbatim.

Two designs, two estimands:

  ds005489 (open-loop): item/pair-level stimulation is genuinely randomized
  (block order random, per the dataset README) inside each session's stim
  lists, licensing a causal (G3) test: does stimulation displace the latent
  state relative to the endogenous confinement/diffusion scale measured on
  that session's own unstimulated words?

  ds005557 (closed-loop): only WHICH LISTS are stim lists is randomized;
  within a stim list, an item is stimulated only when a trained classifier's
  own encoding-window output falls below the running median -- propensity-
  selected on the very state this script measures, not randomized. The
  list-level comparison (confinement estimated separately on stim-list vs
  no-stim-list trials) is therefore reported as a design-correct causal
  test; the item-level classifier-triggered pattern is reported purely
  descriptively, explicitly labeled non-causal.

Hard limit, true of every result in this file: stimulation here is
delivered during episodic ENCODING, not during a working-memory
MAINTENANCE/delay period. These results speak to whether human intracranial
stimulation displaces a memory-relevant state in a way that predicts a
memory outcome; they do not support any WM-maintenance claim.

Output: results/ram_stimulation_drift.json

Run:
    conda run -n wm_dynamics python scripts/run_ram_stimulation_drift.py \\
        --openloop-max-subjects 38 --closedloop-max-subjects 35
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drift_dynamics import (  # noqa: E402
    fit_gaussian_state_space,
    fit_ou_moments,
    leave_one_out_condition_residuals,
)
from geometry import pca_decompose  # noqa: E402
from preprocessing import butterworth_bandstop  # noqa: E402
from provenance import git_commit, sha256_file  # noqa: E402
from statistics import (  # noqa: E402
    linear_mixed_effects_test,
    paired_sign_flip_test,
    stable_seed,
)
from run_ram_openloop_pipeline import _derive_word_stimulation, _load_events  # noqa: E402

RESULTS = ROOT / "results"

# ── Author-defined feature bank (ds005557 README, "Classifier Details") ────────
MORLET_FREQS_HZ = np.array([3.0, 5.4, 9.7, 17.4, 31.1, 55.9, 100.3, 180.0])
MORLET_WAVENUMBER = 5.0
LINE_NOISE_BANDSTOP_HZ = (58.0, 62.0)
ENCODING_WINDOW_S = 1.366
MIRROR_BUFFER_S = 1.365
BIN_S = 0.1                # non-overlapping, unsmoothed bins (drift_dynamics.py's contract)
N_BINS = 13                 # floor(1.366 / 0.1); the trailing 66 ms is dropped, not smoothed over
N_PC = 8
MIN_WORDS = 80

LIMITATIONS = [
    "Stimulation in both ds005489 and ds005557 is delivered during episodic ENCODING, not "
    "during a working-memory MAINTENANCE/delay period. These results support a claim about "
    "whether human intracranial stimulation displaces a memory-relevant state in a way that "
    "predicts a memory outcome at encoding; they do NOT support any WM-maintenance claim.",
    "The closed-loop (ds005557) item-level, classifier-triggered comparison is descriptive, "
    "not causal: which items get stimulated is propensity-selected on the very neural state "
    "being measured (the classifier's own encoding-window decision variable), not randomized.",
    "The Morlet feature bank keeps the classifier's own within-window time bins instead of "
    "collapsing them to the single scalar the online classifier uses, so the drift model has "
    "a trajectory to fit; the spectral decomposition itself (filter, wavelets, log power) is "
    "otherwise the released recipe verbatim.",
    "Displacement is measured in the leading PC of each session's own unperturbed-word "
    "feature space -- an unsupervised, session-specific direction (no repeated-item content "
    "label exists in free recall to define a supervised content axis, unlike DANDI 000469).",
]


def _safe_int(value, default: int = -1) -> int:
    try:
        if value in (None, "", "n/a"):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def configured_data_path(dataset_key: str) -> Path:
    """Resolve a staged release from the configured external data root."""
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    try:
        path = Path(root) / config["datasets"][dataset_key]["local_path"]
    except KeyError as exc:
        raise SystemExit(f"Dataset configuration is incomplete for {dataset_key}.") from exc
    if not path.is_dir():
        raise SystemExit(f"Dataset is not staged at configured path: {path}")
    return path


# ── Author-defined spectral feature bank ────────────────────────────────────────

def morlet_log_power_bank(
    raw_epochs: np.ndarray,
    srate: float,
    freqs: np.ndarray = MORLET_FREQS_HZ,
    n_cycles: float = MORLET_WAVENUMBER,
    buffer_s: float = MIRROR_BUFFER_S,
    bandstop_hz: tuple[float, float] = LINE_NOISE_BANDSTOP_HZ,
    n_bins: int = N_BINS,
    n_jobs: int = 1,
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
    # ponytail: fixed byte-budget batching, not a tuned scheduler -- if a
    # dataset ships far more channels this still works, just in more batches.
    target_bytes = 3e8
    per_epoch_bytes = n_channels * n_freq * n_times_padded * 8
    batch = int(np.clip(target_bytes // max(per_epoch_bytes, 1), 1, n_epochs))

    binned = np.empty((n_epochs, n_channels, n_freq, n_bins), dtype=np.float32)
    for start in range(0, n_epochs, batch):
        stop = min(start + batch, n_epochs)
        power = tfr_array_morlet(
            padded[start:stop], sfreq=srate, freqs=freqs, n_cycles=n_cycles,
            output="power", zero_mean=True, n_jobs=n_jobs, verbose=False,
        )
        log_power = np.log(power + 1e-20)
        cropped = log_power[..., buffer_samples: buffer_samples + n_times]
        trimmed = cropped[..., : n_samp_bin * n_bins]
        binned[start:stop] = trimmed.reshape(
            stop - start, n_channels, n_freq, n_bins, n_samp_bin,
        ).mean(axis=-1).astype(np.float32)
    return binned


# ── BIDS loading (bipolar montage; reuses ds005489 pipeline's event helpers) ───

def build_session_trajectory(
    ieeg_json: Path, data_root: Path, derive_stim_from_stim_on: bool, n_jobs: int = 1,
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
    features = morlet_log_power_bank(raw_epochs, srate, n_jobs=n_jobs)
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


# ── Confined-drift plant + displacement ─────────────────────────────────────────

def fit_group_drift(
    features: np.ndarray, bin_s: float, plant_mask: np.ndarray,
    n_pc: int = N_PC, n_boot: int = 150, seed: int = 0,
) -> dict:
    """Fit PCA + the confined-diffusion drift model on `plant_mask` trials
    (the endogenous/unperturbed data), then score every trial's RMS
    deviation from that plant's own encoding trajectory.

    Trials in `plant_mask` are scored against a leave-one-out group mean (so
    a trial never contributes to its own reference); trials outside the
    mask (e.g. stimulated words) are scored against the full plant mean,
    which never includes them, so there is no leakage either way.
    """
    n_trials, n_bins, n_feat = features.shape
    mask = np.asarray(plant_mask, dtype=bool)
    if mask.sum() < 3:
        return {"status": "excluded", "reason": "fewer than 3 unperturbed trials to fit the plant",
                "n_plant_trials": int(mask.sum())}

    train2d = features[mask].reshape(-1, n_feat)
    mu, sd = train2d.mean(axis=0), train2d.std(axis=0) + 1e-8
    z = (features - mu) / sd
    z_train2d = z[mask].reshape(-1, n_feat)
    z_train_mean = z_train2d.mean(axis=0)

    _, components, var_ratio = pca_decompose(z_train2d, n_pc)
    proj = ((z.reshape(-1, n_feat) - z_train_mean) @ components).reshape(
        n_trials, n_bins, components.shape[1],
    )
    pc1 = proj[:, :, 0]

    condition = np.zeros(int(mask.sum()))
    residuals, _ = leave_one_out_condition_residuals(pc1[mask], condition)
    state_estimate = fit_gaussian_state_space(residuals, bin_s)
    moment_estimate = fit_ou_moments(residuals, bin_s, n_boot=n_boot, rng=np.random.default_rng(seed))

    reference_mean = pc1[mask].mean(axis=0)
    displacement = np.full(n_trials, np.nan)
    displacement[mask] = np.sqrt(np.nanmean(residuals ** 2, axis=1))
    displacement[~mask] = np.sqrt(np.mean((pc1[~mask] - reference_mean[None, :]) ** 2, axis=1))

    scale, scale_source = None, None
    if state_estimate.status == "identifiable" and state_estimate.stationary_variance:
        scale, scale_source = float(np.sqrt(state_estimate.stationary_variance)), "gaussian_lgssm"
    elif moment_estimate.status == "identifiable" and moment_estimate.stationary_variance:
        scale, scale_source = float(np.sqrt(moment_estimate.stationary_variance)), "ou_moments"
    normalized_displacement = displacement / scale if scale else np.full(n_trials, np.nan)

    return {
        "status": "complete",
        "n_plant_trials": int(mask.sum()), "n_pc": int(components.shape[1]),
        "var_explained": float(var_ratio.sum()),
        "pc1": pc1, "displacement": displacement,
        "normalized_displacement": normalized_displacement,
        "confinement_scale": scale, "confinement_scale_source": scale_source,
        "state_space": state_estimate.to_dict(), "moment": moment_estimate.to_dict(),
    }


# ── ds005489 open-loop: design-correct causal test ──────────────────────────────

def openloop_causal_rows(session: dict, seed: int) -> dict:
    """Displacement of stim vs control words, restricted to stim-list words
    (`stim_list == 1`) -- the dataset's own genuinely item/pair-randomized
    subset. The plant is fit on ALL never-stimulated words in the session
    (a larger, still-unperturbed sample) for better-identified confinement.
    """
    features, stim = session["features"], session["stim"]
    fit = fit_group_drift(features, BIN_S, stim == 0, seed=seed)
    if fit["status"] != "complete":
        return {"status": fit["status"], "reason": fit.get("reason"), "session": session["session"]}

    idx = np.flatnonzero(session["stim_list"] == 1)
    rows = [{
        "session": session["session"], "subject": session["subject"],
        "displacement": float(fit["displacement"][i]),
        "normalized_displacement": (
            float(fit["normalized_displacement"][i])
            if np.isfinite(fit["normalized_displacement"][i]) else None
        ),
        "stim": int(stim[i]), "serialpos": int(session["serialpos"][i]), "list": int(session["list"][i]),
    } for i in idx]
    return {
        "status": "complete", "session": session["session"], "subject": session["subject"],
        "n_words": session["n_words"], "n_plant_trials": fit["n_plant_trials"],
        "n_causal_rows": len(rows), "var_explained": fit["var_explained"],
        "confinement_scale": fit["confinement_scale"], "confinement_scale_source": fit["confinement_scale_source"],
        "state_space": fit["state_space"], "moment": fit["moment"], "rows": rows,
    }


# ── ds005557 closed-loop: list-level causal + item-level descriptive ───────────

ITEM_LEVEL_NONCAUSAL_REASON = (
    "Item-level stimulation in ds005557 is triggered only when the classifier's own "
    "encoding-window output falls below the running median -- assignment is propensity-"
    "selected on the very state measured here, not randomized. This comparison describes "
    "the closed-loop system's own selection pattern; it is not a causal contrast."
)


def closedloop_session_analysis(session: dict, seed: int) -> dict:
    features, stim, stim_list = session["features"], session["stim"], session["stim_list"]
    fit = fit_group_drift(features, BIN_S, stim == 0, seed=seed)
    if fit["status"] != "complete":
        return {"status": fit["status"], "reason": fit.get("reason"), "session": session["session"]}
    pc1 = fit["pc1"]

    group0, group1 = np.flatnonzero(stim_list == 0), np.flatnonzero(stim_list == 1)
    list_level: dict = {
        "status": "not_estimable",
        "reason": f"only {len(group0)} no-stim-list and {len(group1)} stim-list words",
    }
    if len(group0) >= 3 and len(group1) >= 3:
        res0, _ = leave_one_out_condition_residuals(pc1[group0], np.zeros(len(group0)))
        res1, _ = leave_one_out_condition_residuals(pc1[group1], np.zeros(len(group1)))
        est0, est1 = fit_gaussian_state_space(res0, BIN_S), fit_gaussian_state_space(res1, BIN_S)
        list_level = {
            "status": "complete",
            "no_stim_list": est0.to_dict(), "stim_list": est1.to_dict(),
            "n_no_stim_list_words": int(len(group0)), "n_stim_list_words": int(len(group1)),
        }

    triggered = np.flatnonzero((stim_list == 1) & (stim == 1))
    nontriggered = np.flatnonzero((stim_list == 1) & (stim == 0))
    item_level: dict = {
        "status": "not_estimable",
        "reason": f"only {len(triggered)} triggered and {len(nontriggered)} non-triggered items",
    }
    if len(triggered) >= 3 and len(nontriggered) >= 3:
        item_level = {
            "status": "complete", "causal": False, "descriptive_only": True,
            "why_noncausal": ITEM_LEVEL_NONCAUSAL_REASON,
            "mean_state_triggered": float(pc1[triggered].mean()),
            "mean_state_nontriggered": float(pc1[nontriggered].mean()),
            "n_triggered": int(len(triggered)), "n_nontriggered": int(len(nontriggered)),
        }

    return {
        "status": "complete", "session": session["session"], "subject": session["subject"],
        "n_words": session["n_words"], "n_plant_trials": fit["n_plant_trials"],
        "var_explained": fit["var_explained"],
        "confinement_scale": fit["confinement_scale"], "confinement_scale_source": fit["confinement_scale_source"],
        "plant_state_space": fit["state_space"], "plant_moment": fit["moment"],
        "list_level": list_level, "item_level": item_level,
    }


# ── Aggregation ──────────────────────────────────────────────────────────────

def _find_stim_sessions(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json"))


def _process_dataset(data_root: Path, max_subjects: int, derive_stim: bool, n_jobs: int,
                      analyze_fn, label: str) -> dict:
    sessions = _find_stim_sessions(data_root)
    print(f"[{label}] {len(sessions)} candidate bipolar+stim sessions on disk", flush=True)
    results, done_subjects = {}, set()
    for ieeg_json in sessions:
        subj = ieeg_json.parts[-4]
        if len(done_subjects) >= max_subjects and subj not in done_subjects:
            continue
        t0 = time.time()
        session_fallback_id = str(ieeg_json.relative_to(data_root))
        try:
            session = build_session_trajectory(ieeg_json, data_root, derive_stim, n_jobs=n_jobs)
        except Exception as exc:
            # zero-drop: a processing failure is recorded with a reason, not silently skipped.
            results[session_fallback_id] = {
                "status": "excluded", "reason": f"processing error: {exc}",
                "session": session_fallback_id,
            }
            print(f"  {session_fallback_id}: FAILED {exc}", flush=True)
            continue
        if session is None:
            continue
        elapsed = time.time() - t0
        if session["status"] != "complete":
            results[session["session"]] = session
            print(f"  {session['session']}: EXCLUDED ({session['reason']}) in {elapsed:.1f}s", flush=True)
            continue
        done_subjects.add(subj)
        result = analyze_fn(session, seed=stable_seed(session["session"]))
        results[session["session"]] = result
        print(f"  {session['session']}: {session['n_words']} words, status={result['status']} "
              f"in {elapsed:.1f}s (n_subjects_so_far={len(done_subjects)})", flush=True)
    return results


def _aggregate_openloop(results: dict) -> dict:
    complete = {k: v for k, v in results.items() if v.get("status") == "complete"}
    rows = [row for sess in complete.values() for row in sess["rows"]]
    n_sessions_excluded = sum(1 for v in results.values() if v.get("status") != "complete")
    if not rows:
        return {"status": "not_estimable", "reason": "no session produced causal rows",
                "n_sessions_complete": len(complete), "n_sessions_excluded": n_sessions_excluded}

    session_id = np.array([r["session"] for r in rows])
    stim = np.array([r["stim"] for r in rows], dtype=float)
    covariates = np.array([[r["serialpos"], r["list"]] for r in rows], dtype=float)
    raw_displacement = np.array([r["displacement"] for r in rows], dtype=float)
    normalized = np.array([
        r["normalized_displacement"] if r["normalized_displacement"] is not None else np.nan
        for r in rows
    ])

    raw_test = linear_mixed_effects_test(raw_displacement, stim, session_id, covariates=covariates)
    finite = np.isfinite(normalized)
    if finite.sum() >= 20 and len(np.unique(session_id[finite])) >= 2:
        normalized_test = linear_mixed_effects_test(
            normalized[finite], stim[finite], session_id[finite], covariates=covariates[finite],
        )
    else:
        normalized_test = {
            "converged": False,
            "reason": f"only {int(finite.sum())} rows had an identifiable confinement scale to normalize by",
        }

    return {
        "status": "complete",
        "n_sessions_complete": len(complete), "n_sessions_excluded": n_sessions_excluded,
        "n_rows_pooled": len(rows), "n_stim": int(stim.sum()), "n_control": int((1 - stim).sum()),
        "n_rows_with_identified_confinement_scale": int(finite.sum()),
        "primary_estimand": "normalized_displacement (RMS PC1 deviation / endogenous confinement scale)",
        "primary_test": normalized_test,
        "legacy_estimand": "raw_displacement (RMS PC1 deviation, PC-score units, not cross-session-comparable)",
        "legacy_test": raw_test,
    }


def _aggregate_closedloop(results: dict) -> dict:
    complete = {k: v for k, v in results.items() if v.get("status") == "complete"}
    n_sessions_excluded = sum(1 for v in results.values() if v.get("status") != "complete")

    paired_stim, paired_nostim, list_sessions = [], [], []
    for key, res in complete.items():
        ll = res["list_level"]
        if ll.get("status") == "complete" and ll["stim_list"]["status"] == "identifiable" \
                and ll["no_stim_list"]["status"] == "identifiable":
            paired_stim.append(ll["stim_list"]["lambda_rate"])
            paired_nostim.append(ll["no_stim_list"]["lambda_rate"])
            list_sessions.append(key)
    if len(paired_stim) >= 3:
        list_level_group = paired_sign_flip_test(
            np.array(paired_stim), np.array(paired_nostim), alternative="two-sided",
            rng=np.random.default_rng(stable_seed("ram_closedloop_list_level")),
        )
        list_level_group.pop("null", None)
        list_level_group["n_sessions"] = len(paired_stim)
        list_level_group["status"] = "complete"
    else:
        list_level_group = {
            "status": "not_estimable",
            "reason": f"only {len(paired_stim)} sessions had identifiable confinement in both list groups",
            "n_sessions": len(paired_stim),
        }

    triggered_means, nontriggered_means, item_sessions = [], [], []
    for key, res in complete.items():
        il = res["item_level"]
        if il.get("status") == "complete":
            triggered_means.append(il["mean_state_triggered"])
            nontriggered_means.append(il["mean_state_nontriggered"])
            item_sessions.append(key)
    if len(triggered_means) >= 3:
        item_level_group = paired_sign_flip_test(
            np.array(triggered_means), np.array(nontriggered_means), alternative="two-sided",
            rng=np.random.default_rng(stable_seed("ram_closedloop_item_level")),
        )
        item_level_group.pop("null", None)
        item_level_group["n_sessions"] = len(triggered_means)
        item_level_group["status"] = "complete"
    else:
        item_level_group = {
            "status": "not_estimable",
            "reason": f"only {len(triggered_means)} sessions had both triggered and non-triggered items",
            "n_sessions": len(triggered_means),
        }
    item_level_group["causal"] = False
    item_level_group["descriptive_only"] = True
    item_level_group["why_noncausal"] = ITEM_LEVEL_NONCAUSAL_REASON

    return {
        "status": "complete",
        "n_sessions_complete": len(complete), "n_sessions_excluded": n_sessions_excluded,
        "list_level_causal_test": list_level_group,
        "item_level_descriptive_pattern": item_level_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openloop-max-subjects", type=int, default=38)
    parser.add_argument("--closedloop-max-subjects", type=int, default=35)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    t_start = time.time()
    openloop_results = _process_dataset(
        configured_data_path("ram_ds005489_openloop"), args.openloop_max_subjects,
        derive_stim=False, n_jobs=args.n_jobs,
        analyze_fn=openloop_causal_rows, label="ds005489 open-loop",
    )
    t_openloop = time.time()
    closedloop_results = _process_dataset(
        configured_data_path("ram_ds005557_closedloop"), args.closedloop_max_subjects,
        derive_stim=True, n_jobs=args.n_jobs,
        analyze_fn=closedloop_session_analysis, label="ds005557 closed-loop",
    )
    t_closedloop = time.time()

    openloop_group = _aggregate_openloop(openloop_results)
    closedloop_group = _aggregate_closedloop(closedloop_results)

    def _strip_arrays(sessions: dict) -> dict:
        # Session result dicts hold only JSON-safe scalars/lists already (pc1
        # and displacement arrays stay internal to fit_group_drift's caller);
        # this is a shallow copy so the caller can't mutate `results` in place.
        return {key: dict(val) for key, val in sessions.items()}

    output = {
        "schema_version": "1.0.0", "analysis_id": "ram_stimulation_drift",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "feature_bank": {
            "source": "ds005557 OpenNeuro README, Classifier Details",
            "freqs_hz": MORLET_FREQS_HZ.tolist(), "wavenumber": MORLET_WAVENUMBER,
            "bandstop_hz": list(LINE_NOISE_BANDSTOP_HZ), "mirror_buffer_s": MIRROR_BUFFER_S,
            "encoding_window_s": ENCODING_WINDOW_S, "bin_s": BIN_S, "n_bins": N_BINS, "n_pc": N_PC,
        },
        "openloop": {
            "dataset": "ds005489", "design": "item/pair-level randomized (stim_list & within-list block order)",
            "gate": "G3 causal", "max_subjects_requested": args.openloop_max_subjects,
            "runtime_seconds": t_openloop - t_start,
            "sessions": _strip_arrays(openloop_results), "group": openloop_group,
        },
        "closedloop": {
            "dataset": "ds005557", "design": "list-level randomized; item-level classifier-triggered",
            "gate": "list-level: G3 causal; item-level: descriptive only",
            "max_subjects_requested": args.closedloop_max_subjects,
            "runtime_seconds": t_closedloop - t_openloop,
            "sessions": _strip_arrays(closedloop_results), "group": closedloop_group,
        },
        "limitations": LIMITATIONS,
    }

    def _default(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    destination = RESULTS / "ram_stimulation_drift.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, default=_default) + "\n")
    print(f"\nSaved {destination}", flush=True)
    print(json.dumps({"openloop_group": openloop_group, "closedloop_group": closedloop_group}, default=_default, indent=2))


if __name__ == "__main__":
    main()
