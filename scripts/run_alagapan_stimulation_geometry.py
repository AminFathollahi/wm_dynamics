#!/usr/bin/env python3
"""Alagapan et al. (2019) stimulation-response geometry: does electrical
stimulation reshape the working-memory retention-period manifold, does the
stimulated cortex align with this project's own fitted dynamics (DMD's
leading mode v*) and controllability structure, and does the size of that
reshaping track each patient's own behavioral benefit?

This is deliberately NOT a replication of the source paper's own phase-lag
hypothesis (that lives in run_alagapan_phase_omega.py, band-matched to each
patient's stimulation frequency). This script instead applies this project's
OWN framework -- broadband PCA geometry, ensemble-DMD-fitted dynamics
(v*/v_stable), and LQR-style controllability (src/control.py) -- to ask
whether THOSE quantities respond to, and are predictive of, this causal
stimulation, using each patient's own known stimulation electrodes (Codes/
Preprocessing.m's hardcoded `stimElectrodes`) as the input direction.

Citation: Alagapan S, Riddle J, Huang WA, Hadar E, Shin HW, Froehlich F.
"Network-Targeted, Multi-site Direct Cortical Stimulation Enhances Working
Memory by Modulating Phase Lag of Low-Frequency Oscillations." Cell Reports
2019;29(9):2590-2598. PMC6901101.

This complements scripts/run_alagapan_phase_omega.py, which only ever reads
the BASELINE (no-stimulation) recording. That script cannot say anything
about how stimulation changes neural geometry, because it never looks at the
stimulation-session recording at all. This script does.

Why not just analyze the raw stimulation-session recording directly: the
on-disk `iEEG Data/Stimulation/{P}_SMS_Stimulation.set` is UNCLEANED. The
original authors' artifact-removal step (Codes/StimArtifactRemoval_ICA.m)
runs interactive ICA with a manual `listdlg` component-rejection dialog, and
its cleaned output was never included in the shared dataset (no
`*_Stimulation_Epoched.set` file exists on disk, only the raw continuous
recording). Analyzing the stimulation-burst window directly would measure
electrical artifact, not physiology.

The retention period is the workaround: stimulation is delivered only during
encoding (`stimDuration == encodingDuration` in the authors' own
Codes/Preprocessing.m), so retention always starts after the stimulation
burst ends. Retention boundaries are given directly by event codes rather
than assumed timing constants (verified by inspecting
P1_SMS_Baseline_Epoched.set's own EEG.epoch fields via pymatreader): DIN1 =
trial onset, DIN2 (xN) = encoding items, DIN4 = retention onset (encoding
end), DIN5 = retention offset (probe onset). The same DIN4/DIN5 codes are
present in the raw continuous Stimulation.set. A short buffer after DIN4
guards against any stimulation-artifact decay tail.

Evidentiary strength (stated once, applies everywhere a number from this
script appears, same caveat as run_alagapan_phase_omega.py): n=3 patients.
Reported as descriptive per-patient geometry and 3-point sign agreement with
behavior, never as a p-value.

Output: results/alagapan_stimulation_geometry.json

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_alagapan_stimulation_geometry.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mne  # noqa: E402
import pymatreader  # noqa: E402
from scipy.signal import welch  # noqa: E402

from control import stimulation_input_alignment  # noqa: E402
from dynamics import fit_retention_dynamics  # noqa: E402
from geometry import (  # noqa: E402
    pca_decompose,
    pca_participation_ratio,
    select_latent_dim,
    subspace_overlap,
)
from run_alagapan_phase_omega import (  # noqa: E402
    DATA_DIR,
    PATIENTS,
    _load_mapping,
    _load_seizure_electrodes,
    load_baseline_data,
    load_behavior,
)
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
RETENTION_ONSET_BUFFER_S = 0.25  # guards against a stimulation-artifact decay tail
CONDITIONS = ("In Phase", "Anti Phase", "Sham")
GRAMIAN_HORIZON = 20
N_RANDOM_DIRS = 20

# Each patient's own 2-site stimulation electrodes, resolved from the
# authors' own Codes/Preprocessing.m (hardcoded `stimElectrodes` per patient,
# e.g. "stimElectrodes = {'E33','E34','E51','E52'}; % For P1") to this
# project's descriptive channel labels via the same electrode-mapping CSV
# _load_mapping already reads (row 33 -> LFA1, 34 -> LFA2, 51 -> LPA1,
# 52 -> LPA2 for P1, confirmed identical index->label in both the baseline
# and stimulation mapping CSVs). Both sites are pooled into one input
# direction: the causal manipulation here is the PHASE LAG between them
# (already tested by the per-condition geometry below), so this direction
# instead asks whether the stimulated cortex overall aligns with the
# patient's own fitted dynamics.
STIM_SITES = {
    "P1": [["LFA1", "LFA2"], ["LPA1", "LPA2"]],
    "P2": [["LAF5", "LAF6"], ["LAP5", "LAP6"]],
    "P3": [["RAF6", "RAF7"], ["RSP5", "RSP6"]],
}


def _stimulation_channel_weight(sites: list[list[str]], labels: list[str]) -> dict | None:
    """(C,) averaged indicator over the stimulation electrodes that survive
    this patient's channel exclusion (seizure electrodes, baseline/stim
    montage intersection), or None if none survived."""
    label_idx = {lb: i for i, lb in enumerate(labels)}
    idx = [label_idx[ch] for site in sites for ch in site if ch in label_idx]
    if not idx:
        return None
    weight = np.zeros(len(labels))
    weight[idx] = 1.0 / len(idx)
    n_expected = sum(len(site) for site in sites)
    return {"weight": weight, "n_electrodes_found": len(idx), "n_electrodes_expected": n_expected}


def _trial_conditions(patient: str) -> list[str]:
    """Ordered per-trial condition labels, in the same row order as the
    Task Performance CSV. Column position differs across patients' CSVs, so
    this reads by header name (DictReader), same as load_behavior."""
    path = DATA_DIR / "Task Performance" / f"{patient}_SternbergStimulation_Summary.csv"
    with open(path) as f:
        return [row["Condition"] for row in csv.DictReader(f)]


def _retention_onset_offset_ms(types: np.ndarray, latencies: np.ndarray) -> tuple[float, float] | None:
    """Retention onset/offset within one trial, from its own event codes.

    Onset: DIN4 (explicit retention-onset marker) if present; some patients'
    task-code version never emits DIN4 (verified: 0/80 epochs for P2 and P3,
    vs. 60/60 for P1), in which case the last DIN2 (final encoding-item
    marker) is used instead -- encoding necessarily ends, and retention
    necessarily begins, at that point regardless of set size.
    Offset: DIN5 (retention-offset / probe-onset marker). Not every epoch
    captures it (some patients' longer encoding periods for larger set
    sizes push the probe past the epoch's fixed window) -- returns None if
    absent rather than guessing a duration.
    """
    din4 = np.where(types == "DIN4")[0]
    din2 = np.where(types == "DIN2")[0]
    din5 = np.where(types == "DIN5")[0]
    if len(din4) > 0:
        onset_ms = float(latencies[din4[0]])
    elif len(din2) > 0:
        onset_ms = float(latencies[din2[-1]])
    else:
        return None
    if len(din5) == 0:
        return None
    offset_candidates = latencies[din5]
    offset_candidates = offset_candidates[offset_candidates > onset_ms]
    if len(offset_candidates) == 0:
        return None
    return onset_ms, float(offset_candidates.min())


def _baseline_retention_trials(patient: str) -> tuple[np.ndarray, list[str], float]:
    """Per-trial retention-window baseline data (see
    _retention_onset_offset_ms for the onset/offset convention), cropped to
    a common length. Returns ((N, C, T) array, kept channel labels, srate)."""
    set_path = DATA_DIR / "iEEG Data" / "Baseline" / f"{patient}_SMS_Baseline_Epoched.set"
    d = pymatreader.read_mat(str(set_path))
    eeg = d["EEG"] if "EEG" in d else d
    srate = float(eeg["srate"])
    epoch = eeg["epoch"]
    n_epochs = len(epoch["eventlatency"])

    data, _, labels = load_baseline_data(patient)  # (N, C, T_full), already channel-filtered
    windows = []
    for i in range(n_epochs):
        lat = np.atleast_1d(epoch["eventlatency"][i])
        typ = np.atleast_1d(epoch["eventtype"][i])
        bounds = _retention_onset_offset_ms(typ, lat)
        if bounds is None:
            continue
        onset_ms, offset_ms = bounds
        onset_ms += RETENTION_ONSET_BUFFER_S * 1000.0
        if offset_ms <= onset_ms:
            continue
        s0 = int(round(onset_ms / 1000.0 * srate))
        s1 = int(round(offset_ms / 1000.0 * srate))
        windows.append((i, s0, s1))

    common_t = min(s1 - s0 for _, s0, s1 in windows)
    trials = np.stack([data[i, :, s0:s0 + common_t] for i, s0, _ in windows], axis=0)
    return trials, labels, srate


def _stimulation_retention_trials(patient: str) -> tuple[np.ndarray, list[str]] | None:
    """Per-trial retention-window (DIN4->DIN5) stimulation-session data,
    cropped to a common length, in the same channel order as
    _baseline_retention_trials. Returns None if trial-count integrity check
    fails (do not silently misalign trial order to condition labels)."""
    conditions = _trial_conditions(patient)
    set_path = DATA_DIR / "iEEG Data" / "Stimulation" / f"{patient}_SMS_Stimulation.set"
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    mapping = _load_mapping(patient)
    seizure = _load_seizure_electrodes(patient)
    keep_idx = [i for i in range(len(raw.ch_names))
                if (i + 1) in mapping and (i + 1) not in seizure
                and not mapping[i + 1].upper().startswith("EKG")]
    labels = [mapping[i + 1] for i in keep_idx]
    data_full = raw.get_data()[keep_idx]  # (C, T_total)
    srate = float(raw.info["sfreq"])

    ann = raw.annotations
    onsets = np.array(ann.onset)
    descs = np.array(ann.description)
    trial_starts = np.sort(onsets[descs == "DIN1"])

    if len(trial_starts) != len(conditions):
        print(f"  SKIP {patient} stimulation geometry -- DIN1 trial count "
              f"({len(trial_starts)}) != behavior CSV row count ({len(conditions)}); "
              f"trial-order alignment cannot be trusted")
        return None

    trial_bounds = list(trial_starts) + [onsets.max() + 1.0]
    windows = []
    for k in range(len(trial_starts)):
        t0, t1 = trial_bounds[k], trial_bounds[k + 1]
        in_trial = (onsets >= t0) & (onsets < t1)
        bounds = _retention_onset_offset_ms(descs[in_trial], onsets[in_trial] * 1000.0)
        if bounds is None:
            continue
        onset_ms, offset_ms = bounds
        onset_s = onset_ms / 1000.0 + RETENTION_ONSET_BUFFER_S
        offset_s = offset_ms / 1000.0
        if offset_s <= onset_s:
            continue
        s0 = int(round(onset_s * srate))
        s1 = int(round(offset_s * srate))
        if s1 <= s0:
            continue
        windows.append((k, s0, s1))

    if not windows:
        return None
    common_t = min(s1 - s0 for _, s0, s1 in windows)
    trials = np.stack([data_full[:, s0:s0 + common_t] for _, s0, _ in windows], axis=0)
    kept_conditions = [conditions[k] for k, _, _ in windows]
    return trials, labels, kept_conditions


def _spectral_sanity_check(baseline_pooled: np.ndarray, condition_pooled: np.ndarray,
                            srate: float) -> dict:
    """Compare power spectra of stimulation-session vs baseline retention data
    as a coarse check against residual stimulation-artifact contamination
    (mirrors the SASS-validation logic in run_haslacher_phase_omega.py's
    dataset documentation): a plausible neural difference should not look like
    a broadband gain change."""
    f_b, p_b = welch(baseline_pooled, fs=srate, axis=-1, nperseg=min(256, baseline_pooled.shape[-1]))
    f_c, p_c = welch(condition_pooled, fs=srate, axis=-1, nperseg=min(256, condition_pooled.shape[-1]))
    ratio = np.mean(p_c) / np.mean(p_b) if np.mean(p_b) > 0 else float("nan")
    return {"mean_power_ratio_condition_over_baseline": float(ratio),
            "note": ("ratio far from 1 across the whole spectrum is more consistent with "
                     "residual broadband artifact than a band-specific neural effect; "
                     "reported as a caveat, not used to suppress the result")}


def _geometry_vs_baseline(baseline_trials: np.ndarray, condition_trials: np.ndarray,
                           k: int) -> dict:
    """subspace_overlap, participation-ratio change, and dispersion shift (in
    the baseline PCA frame) of condition_trials relative to baseline_trials.
    Both are (N, C, T) with matching C (channel) axis.

    Dispersion shift, not a raw centroid-distance: iEEG voltage in an
    oscillation band is zero-mean by construction, so the *mean* channel
    vector over a multi-cycle retention window is ~0 for both baseline and
    any stimulation condition regardless of a real amplitude/geometry
    change -- comparing mean vectors is a near-tautological null. Comparing
    the mean *distance* from the baseline centroid (norm taken before
    averaging) is not: it tracks a genuine change in how far, on average, a
    moment of retention-period activity sits from the baseline manifold's
    center, which does respond to an amplitude or subspace change.
    """
    baseline_pooled = baseline_trials.transpose(0, 2, 1).reshape(-1, baseline_trials.shape[1])
    condition_pooled = condition_trials.transpose(0, 2, 1).reshape(-1, condition_trials.shape[1])

    baseline_scores, baseline_components, _ = pca_decompose(baseline_pooled, k)
    _, condition_components, _ = pca_decompose(condition_pooled, k)

    baseline_mean = baseline_pooled.mean(axis=0)
    condition_scores_in_baseline_frame = (condition_pooled - baseline_mean) @ baseline_components
    baseline_dispersion = float(np.linalg.norm(baseline_scores, axis=1).mean())
    condition_dispersion = float(np.linalg.norm(condition_scores_in_baseline_frame, axis=1).mean())

    dispersion_shift = condition_dispersion - baseline_dispersion
    return {
        "subspace_overlap": subspace_overlap(baseline_components, condition_components),
        "participation_ratio_baseline": pca_participation_ratio(baseline_pooled),
        "participation_ratio_condition": pca_participation_ratio(condition_pooled),
        "dispersion_baseline": baseline_dispersion,
        "dispersion_condition": condition_dispersion,
        "dispersion_shift_in_baseline_pca_frame": dispersion_shift,
        "dispersion_shift_pct": 100.0 * dispersion_shift / baseline_dispersion
                                if baseline_dispersion > 0 else float("nan"),
    }


def main():
    out = {}
    for patient in PATIENTS:
        print(f"{patient} ...")
        baseline_trials_full, labels_full, srate = _baseline_retention_trials(patient)

        stim_result = _stimulation_retention_trials(patient)
        if stim_result is None:
            out[patient] = {"status": "skipped", "reason": "DIN1/behavior trial-count mismatch"}
            continue
        stim_trials_full, stim_labels, kept_conditions = stim_result

        # The stimulation-session recording does not always record every
        # channel the baseline session does (e.g. P1's baseline montage
        # includes right-hemisphere depth contacts absent from its
        # stimulation recording). Restrict to the channel set common to
        # both sessions rather than assuming the two montages match.
        stim_label_set = set(stim_labels)
        common_idx_baseline = [i for i, lb in enumerate(labels_full) if lb in stim_label_set]
        common_labels = [labels_full[i] for i in common_idx_baseline]
        stim_idx_by_label = {lb: i for i, lb in enumerate(stim_labels)}
        common_idx_stim = [stim_idx_by_label[lb] for lb in common_labels]
        if len(common_labels) < len(labels_full) or len(common_labels) < len(stim_labels):
            print(f"  {patient}: baseline has {len(labels_full)} channels, stimulation session "
                  f"has {len(stim_labels)}; restricting to {len(common_labels)} common channels")

        baseline_trials = baseline_trials_full[:, common_idx_baseline, :]
        stim_trials = stim_trials_full[:, common_idx_stim, :]
        labels = common_labels

        rng = np.random.default_rng(stable_seed(f"alagapan_stim_geometry_{patient}"))
        k_info = select_latent_dim(baseline_trials, method="cv_pr", rng=rng)
        k = k_info["k"]
        print(f"  baseline retention: {baseline_trials.shape[0]} trials, "
              f"{baseline_trials.shape[1]} channels, k={k} (cv_pr={k_info['cv_pr']:.2f})")

        dyn = fit_retention_dynamics(baseline_trials, srate, k, rng)
        print(f"  dynamics: max_real_eig={dyn['max_real_eig']:.3f} r2_cv={dyn['r2_cv']:.3f} "
              f"r2_null={dyn['r2_null']:.3f} identifiable={dyn['identifiable']}")

        chan_weight = _stimulation_channel_weight(STIM_SITES[patient], labels)
        stim_input = None
        if chan_weight is not None:
            align_rng = np.random.default_rng(stable_seed(f"alagapan_random_dirs_{patient}"))
            stim_input = stimulation_input_alignment(
                dyn["A"], dyn["components"], chan_weight["weight"],
                dyn["v_star"], dyn["v_stable"], align_rng,
                gramian_horizon=GRAMIAN_HORIZON, n_random_dirs=N_RANDOM_DIRS,
            )
            stim_input["n_stim_electrodes_found"] = chan_weight["n_electrodes_found"]
            stim_input["n_stim_electrodes_expected"] = chan_weight["n_electrodes_expected"]
            print(f"  stimulation-input geometry: alignment_to_vstar="
                  f"{stim_input['alignment_to_vstar']:.3f} "
                  f"(random_null={stim_input['random_direction_alignment']:.3f}) "
                  f"gramian_trace={stim_input['gramian_trace']:.3e} "
                  f"[{chan_weight['n_electrodes_found']}/{chan_weight['n_electrodes_expected']} "
                  f"stim electrodes survived exclusion]")
        else:
            print("  stimulation-input geometry: skipped (no stimulation electrodes "
                  "survived channel exclusion)")

        behavior = load_behavior(patient)
        per_condition = {}
        for cond in CONDITIONS:
            idx = [i for i, c in enumerate(kept_conditions) if c == cond]
            if len(idx) < 2:
                per_condition[cond] = {"status": "skipped", "reason": "fewer than 2 trials"}
                continue
            cond_trials = stim_trials[idx]
            geom = _geometry_vs_baseline(baseline_trials, cond_trials, k)
            geom["spectral_sanity_check"] = _spectral_sanity_check(
                baseline_trials.reshape(-1, baseline_trials.shape[-1]),
                cond_trials.reshape(-1, cond_trials.shape[-1]),
                srate=1000.0,
            )
            geom["n_trials"] = len(idx)
            per_condition[cond] = geom
            print(f"  {cond}: n={len(idx)} subspace_overlap={geom['subspace_overlap']:.3f} "
                  f"dispersion_shift={geom['dispersion_shift_pct']:+.1f}% "
                  f"PR base->cond={geom['participation_ratio_baseline']:.2f}->"
                  f"{geom['participation_ratio_condition']:.2f}")

        in_phase = per_condition.get("In Phase", {})
        sham = per_condition.get("Sham", {})
        geometric_perturbation_larger_in_phase = None
        sign_agrees_with_behavior = None
        if "dispersion_shift_in_baseline_pca_frame" in in_phase and \
                "dispersion_shift_in_baseline_pca_frame" in sham:
            geometric_perturbation_larger_in_phase = bool(
                in_phase["dispersion_shift_in_baseline_pca_frame"]
                > sham["dispersion_shift_in_baseline_pca_frame"])
            behavioral_benefit = behavior.get("in_phase_minus_sham")
            if behavioral_benefit is not None:
                sign_agrees_with_behavior = bool(
                    geometric_perturbation_larger_in_phase and behavioral_benefit > 0)

        out[patient] = {
            "k_latent_dim": k_info, "n_baseline_trials": int(baseline_trials.shape[0]),
            "dynamics": {k_: v_ for k_, v_ in dyn.items() if k_ not in ("A", "components", "mean",
                                                                         "v_star", "v_stable")},
            "stimulation_input_alignment": stim_input,
            "per_condition": per_condition, "behavior": behavior,
            "geometric_perturbation_larger_in_phase_than_sham": geometric_perturbation_larger_in_phase,
            "sign_agrees_with_behavior": sign_agrees_with_behavior,
        }

    n_valid = sum(1 for v in out.values() if v.get("sign_agrees_with_behavior") is not None)
    n_agree = sum(1 for v in out.values() if v.get("sign_agrees_with_behavior"))
    print(f"\nSign agreement (geometric perturbation tracks behavioral benefit) across "
          f"n={n_valid} evaluable patients: {n_agree}/{n_valid} "
          f"(reported as sign agreement, not a p-value)")

    out["_meta"] = {
        "citation": "Alagapan et al. 2019, Cell Reports, PMC6901101",
        "n_patients": len(PATIENTS), "n_evaluable": n_valid, "n_sign_agree": n_agree,
        "evidentiary_strength": "reanalysis of public raw data, n=3 patients, descriptive only",
        "note": ("Retention-period-only geometry, computed to avoid the unremovable "
                 "electrical stimulation artifact present in the raw, uncleaned "
                 "stimulation-burst window (see module docstring)."),
    }
    with open(RESULTS / "alagapan_stimulation_geometry.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/alagapan_stimulation_geometry.json")


if __name__ == "__main__":
    main()
