#!/usr/bin/env python3
"""Haslacher et al. (2024) CLAM-tACS stimulation-response geometry: does
phase-tuned tACS reshape the working-memory retention-period manifold, does
each participant's own stimulation site align with this project's own fitted
dynamics (DMD's leading mode v*) and controllability structure, is any of
this specific to the active (occipital) group vs. the control (frontal)
group, and does its size predict each participant's own behavioral phase-
modulation depth?

Citation: Haslacher D, Cavallo A, Reber P, Kattein A, Thiele M, Nasr K,
Hashemi K, Sokoliuk R, Thut G, Soekadar SR. "Working memory enhancement
using real-time phase-tuned transcranial alternating current stimulation."
Brain Stimulation 17 (2024) 850-859. https://doi.org/10.1016/j.brs.2024.07.007

This complements scripts/run_haslacher_phase_omega.py, which reads the
`stim` recording with `preload=False` for event markers only and never
analyzes the EEG signal itself, and reproduces the source paper's own
Pz-centered-Laplacian, single-alpha-band phase-locking analysis. THIS script
is deliberately NOT that replication: it applies this project's OWN
framework -- broadband PCA geometry, ensemble-DMD-fitted dynamics (v*/
v_stable), and LQR-style controllability (src/control.py) -- to the full
retained channel set, asking whether those quantities respond to, and
predict, this causal stimulation. Two consequences follow from that choice,
both departures from the dataset's own recommended recipe (Data/README.md
sections 5-8, which targets the single Pz Laplacian):
  1. No band restriction. Filtering to alpha (or any single band) before
     PCA/DMD would collapse a narrowband oscillation to a single spatial
     line (the covariance of A*cos(theta(t)+phi) across channels is close to
     rank-1 whenever phi is near-constant across nearby electrodes, as
     scalp alpha typically is) -- trivializing subspace_overlap and
     preempting exactly the multi-dimensional geometry this project's
     framework is built to characterise. A standard broadband EEG passband
     (BROADBAND_HZ) is used instead.
  2. The actual stimulation electrodes are kept, not dropped. The README's
     own NOT_OF_INTEREST list (still followed by run_haslacher_phase_omega.py
     for the Pz-Laplacian target) excludes O1/O2 (active) and Fpz/Cz
     (control) -- i.e. exactly the channels this script needs to define the
     stimulation input direction B. Saturated-channel rejection + SASS
     (README section 5-6, reused with attribution) still apply to them: a
     hardware-railed channel is still dropped, and SASS's projection now
     also sees the near-stimulation channels it is meant to clean, which the
     original channel-drop order never let it do.

Output: results/haslacher_stimulation_geometry.json

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_haslacher_stimulation_geometry.py
"""
from __future__ import annotations

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
from scipy import linalg  # noqa: E402
from scipy.signal import welch  # noqa: E402

from control import stimulation_input_alignment  # noqa: E402
from dynamics import fit_retention_dynamics  # noqa: E402
from geometry import (  # noqa: E402
    pca_decompose,
    pca_participation_ratio,
    select_latent_dim,
    subspace_overlap,
)
from run_haslacher_phase_omega import (  # noqa: E402
    ACTIVE_SUBJECTS,
    CONTROL_SUBJECTS,
    DATA_DIR,
    PHASE_CONDITIONS,
    RETENTION_TMAX,
    RETENTION_TMIN,
    _modulation,
)
from statistics import pearson_permutation_test, permutation_pvalue, stable_seed  # noqa: E402

RESULTS = ROOT / "results"
SFREQ_ANALYSIS = 200.0
SATURATION_THRESHOLD = 0.418  # dataset README's own hardware-rail indicator, volts
BROADBAND_HZ = (1.0, 40.0)  # standard EEG passband: removes slow drift and muscle/
                            # line-adjacent noise without restricting to one oscillatory
                            # sub-band (contrast run_haslacher_phase_omega.py's ALPHA_BAND)
N_PERM_MODULATION = 200  # smaller than the 2000 used for the cheap accuracy statistic
                         # in run_haslacher_phase_omega.py: this null requires only
                         # re-averaging precomputed per-trial scalars, but many
                         # participants x conditions makes 2000 unnecessarily slow
GRAMIAN_HORIZON = 20
N_RANDOM_DIRS = 20
AUX_CHANNELS = ["envelope", "stim"]
PROTECT = ["Pz", "PO7", "PO8", "P3", "P4"]  # the source paper's own analysis target;
                                            # never dropped by saturated-channel rejection
# The dataset's own documented stimulation montage (Data/README.md's
# participant table): occipital electrodes for the active group, frontal for
# control. These are exactly the channels the source paper's own
# NOT_OF_INTEREST list drops (they are not needed for the Pz-Laplacian
# target) -- but they are what this script needs to build the stimulation
# input direction B, so they are kept through preprocessing here.
STIM_ELECTRODES = {"active": ["O1", "O2"], "control": ["Fpz", "Cz"]}
NOT_OF_INTEREST = {
    "active": ["Fp1", "Fpz", "Fp2", "FC1", "Fz", "FC2", "C1", "Cz", "C2", "CP1",
               "CPz", "CP2", "F9", "F10", "FT9", "FT10", "TP9", "TP10", "O1", "O2"],
    "control": ["Fp1", "Fpz", "Fp2", "FC1", "C5", "FC2", "C1", "Cz", "C2", "CP1",
                "CPz", "CP2", "F9", "F10", "FT9", "FT10", "TP9", "TP10", "O1", "O2"],
}


def _stimulation_channel_weight(group: str, ch_names: list[str]) -> dict | None:
    """(C,) averaged indicator over this group's stimulation electrodes that
    survived saturated-channel rejection, or None if none survived."""
    idx = [ch_names.index(ch) for ch in STIM_ELECTRODES[group] if ch in ch_names]
    if not idx:
        return None
    weight = np.zeros(len(ch_names))
    weight[idx] = 1.0 / len(idx)
    return {"weight": weight, "n_electrodes_found": len(idx),
            "n_electrodes_expected": len(STIM_ELECTRODES[group])}


def _sass(no_stim: "mne.io.Raw", stim: "mne.io.Raw") -> int:
    """Project the tACS artifact out of `stim` in place, using `no_stim` as
    the artifact-free reference (Haslacher et al., NeuroImage 2021,
    228:117571). Reused directly from the dataset's own Data/README.md
    section 6, with attribution."""
    picks = [ch for ch in stim.ch_names if ch not in AUX_CHANNELS]
    ix = [stim.ch_names.index(ch) for ch in picks]

    c_stim = np.cov(stim.get_data(picks))
    c_nostim = np.cov(no_stim.get_data(picks))

    eigvals, eigvecs = linalg.eig(c_stim, c_nostim)
    order = np.argsort(eigvals.real)[::-1]
    d = eigvecs.real[:, order].T
    m = linalg.pinv(d)

    dists = []
    for k in range(len(picks)):
        keep = np.ones(m.shape[0])
        keep[:k] = 0
        p = m @ np.diag(keep) @ d
        dists.append(np.linalg.norm(c_nostim - p @ c_stim @ p.T, ord="nuc"))
    k = int(np.argmin(dists))

    keep = np.ones(m.shape[0])
    keep[:k] = 0
    p = m @ np.diag(keep) @ d
    stim._data[ix] = p @ stim._data[ix]
    return k


def _preprocess(subject: str, group: str) -> tuple["mne.io.Raw", "mne.io.Raw", int]:
    """Saturated-channel rejection + resample + broadband filter + SASS, per
    Data/README.md section 5-6 -- WITHOUT that README's own NOT_OF_INTEREST
    channel drop or alpha-band restriction (see module docstring for why: this
    script needs the actual stimulation electrodes and a multi-dimensional,
    not band-collapsed, state space). Returns (no_stim, stim,
    n_sass_components)."""
    no_stim = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "no_stim.vhdr"),
                                           preload=True, verbose="ERROR")
    stim = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "stim.vhdr"),
                                        preload=True, verbose="ERROR")

    sprobe = stim.copy().drop_channels([c for c in AUX_CHANNELS if c in stim.ch_names])
    maxv = np.abs(sprobe.get_data()).max(axis=-1)
    bad_saturated = [ch for ch, v in zip(sprobe.ch_names, maxv) if v > SATURATION_THRESHOLD]

    to_drop = (set(bad_saturated) | {"stim"}) - set(PROTECT)
    to_drop = [c for c in to_drop if c in no_stim.ch_names]
    no_stim.drop_channels(to_drop)
    stim.drop_channels(to_drop)

    no_stim.resample(SFREQ_ANALYSIS)
    stim.resample(SFREQ_ANALYSIS)
    no_stim.filter(*BROADBAND_HZ, verbose="ERROR")
    stim.filter(*BROADBAND_HZ, verbose="ERROR")

    n_sass = _sass(no_stim, stim)
    # The envelope is an auxiliary stimulation-control trace, not neural EEG.
    # It is required while estimating SASS but must never enter PCA/dynamics.
    for raw in (no_stim, stim):
        raw.drop_channels([channel for channel in AUX_CHANNELS if channel in raw.ch_names])
    return no_stim, stim, n_sass


def _target_alpha_power(raw: "mne.io.Raw") -> float:
    """Mean 8--14 Hz power of the README-defined Pz surface Laplacian."""
    required = ["Pz", "PO7", "PO8", "P3", "P4"]
    if any(channel not in raw.ch_names for channel in required):
        return float("nan")
    center = raw.get_data(picks=["Pz"])[0]
    ring = raw.get_data(picks=["PO7", "PO8", "P3", "P4"]).mean(axis=0)
    frequency, power = welch(center - ring, fs=raw.info["sfreq"],
                             nperseg=min(int(3 * raw.info["sfreq"]), len(center)))
    selected = (frequency >= 8.0) & (frequency <= 14.0)
    return float(np.mean(power[selected]))


def _sass_sanity(baseline_power: float, before_power: float, after_power: float) -> dict:
    """Operationalize the README's post-SASS target-spectrum sanity check."""
    values = np.asarray([baseline_power, before_power, after_power], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        return {"power_ratio": None, "reduced_log_distance": False, "passed": False}
    before_distance = float(abs(np.log(before_power / baseline_power)))
    after_distance = float(abs(np.log(after_power / baseline_power)))
    ratio = float(after_power / baseline_power)
    return {"power_ratio": ratio, "reduced_log_distance": after_distance < before_distance,
            "passed": bool(0.25 <= ratio <= 4.0 and after_distance < before_distance)}


def _preprocess_author_native(subject: str, group: str) -> tuple["mne.io.Raw", "mne.io.Raw", dict]:
    """Apply the release README's ordered QC, alpha filter, SASS, and reference.

    This path is used for phase-diffusion inference.  Unlike the exploratory
    broadband geometry path, it follows the source paper's neural-signal
    preprocessing and uses pyprep bad-channel detection as performed in the
    paper.  Average reference is applied after SASS because diffusion is a
    whole-scalp measure rather than the single Pz Laplacian.
    """
    from pyprep.find_noisy_channels import NoisyChannels

    no_stim = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "no_stim.vhdr"),
                                           preload=True, verbose="ERROR")
    stim = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "stim.vhdr"),
                                        preload=True, verbose="ERROR")
    drop_hint = NOT_OF_INTEREST[group]

    probe = no_stim.copy().drop_channels([
        channel for channel in AUX_CHANNELS + drop_hint if channel in no_stim.ch_names
    ])
    noisy = NoisyChannels(probe, random_state=stable_seed(f"haslacher_pyprep_{subject}"))
    noisy.find_all_bads(ransac=False)
    bad_noisy = noisy.get_bads()

    stimulation_probe = stim.copy().drop_channels([
        channel for channel in AUX_CHANNELS + drop_hint if channel in stim.ch_names
    ])
    maximum = np.abs(stimulation_probe.get_data()).max(axis=-1)
    bad_saturated = [channel for channel, value in zip(stimulation_probe.ch_names, maximum)
                     if value > SATURATION_THRESHOLD]
    requested = (set(bad_noisy) | set(bad_saturated) | set(drop_hint) | {"stim"}) - set(PROTECT)
    dropped = sorted(channel for channel in requested
                     if channel in no_stim.ch_names and channel in stim.ch_names)
    no_stim.drop_channels(dropped)
    stim.drop_channels(dropped)

    no_stim.resample(SFREQ_ANALYSIS)
    stim.resample(SFREQ_ANALYSIS)
    no_stim.filter(8.0, 14.0, verbose="ERROR")
    stim.filter(8.0, 14.0, verbose="ERROR")
    baseline_power = _target_alpha_power(no_stim)
    before_power = _target_alpha_power(stim)
    n_sass = _sass(no_stim, stim)
    after_power = _target_alpha_power(stim)

    for raw in (no_stim, stim):
        raw.drop_channels([channel for channel in AUX_CHANNELS if channel in raw.ch_names])
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    sanity = _sass_sanity(baseline_power, before_power, after_power)
    metadata = {
        "bad_noisy_pyprep": bad_noisy,
        "bad_saturated": bad_saturated,
        "channels_dropped": dropped,
        "n_sass_components_removed": n_sass,
        "target_alpha_power_baseline": baseline_power,
        "target_alpha_power_before_sass": before_power,
        "target_alpha_power_after_sass": after_power,
        "target_alpha_power_after_to_baseline_ratio": sanity["power_ratio"],
        "sass_reduced_target_log_power_distance": sanity["reduced_log_distance"],
        "sass_sanity_pass": sanity["passed"],
        "sass_sanity_rule": "after/base target alpha power within [0.25,4] and closer on log scale than before/base",
    }
    return no_stim, stim, metadata


def _retention_trials(raw: "mne.io.Raw", codes: list[int] | None = None) -> dict[int, np.ndarray]:
    """Per-phase-condition (N, C, T) retention-window trials. `codes=None`
    pools all six conditions (used for the no_stim baseline, where the
    phase-condition label is not behaviorally meaningful)."""
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    event_id = list(PHASE_CONDITIONS) if codes is None else codes
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=RETENTION_TMIN, tmax=RETENTION_TMAX,
                        baseline=None, preload=True, on_missing="ignore", verbose="ERROR")
    data = epochs.get_data(copy=True)  # (N, C, T)
    trial_codes = epochs.events[:, 2]
    if codes is None:
        return {0: data}
    return {c: data[trial_codes == c] for c in codes}


def _geometry_vs_baseline(baseline_pooled: np.ndarray, condition_pooled: np.ndarray,
                           baseline_scores: np.ndarray, baseline_components: np.ndarray,
                           baseline_mean: np.ndarray, k: int) -> dict:
    """subspace_overlap, participation-ratio, and dispersion shift (in the
    baseline PCA frame) of condition_pooled relative to the baseline PCA
    basis. Pooled arrays are (observations, C).

    Dispersion shift, not a raw centroid distance: high-pass-filtered EEG is
    ~zero-mean over any window much longer than the passband's low cutoff,
    so the *mean* channel vector is ~0 for both baseline and any stimulation
    condition regardless of a real change -- comparing mean vectors is a
    near-tautological null. Comparing the mean *distance* from the baseline
    centroid (norm taken before averaging over observations) is not.
    """
    _, condition_components, _ = pca_decompose(condition_pooled, k)
    condition_scores_in_baseline_frame = (condition_pooled - baseline_mean) @ baseline_components
    baseline_dispersion = float(np.linalg.norm(baseline_scores, axis=1).mean())
    condition_dispersion = float(np.linalg.norm(condition_scores_in_baseline_frame, axis=1).mean())
    return {
        "subspace_overlap": subspace_overlap(baseline_components, condition_components),
        "participation_ratio_condition": pca_participation_ratio(condition_pooled),
        "dispersion_baseline": baseline_dispersion,
        "dispersion_condition": condition_dispersion,
        "dispersion_shift_in_baseline_pca_frame": condition_dispersion - baseline_dispersion,
    }


def _per_trial_drift(condition_trials: np.ndarray, baseline_components: np.ndarray,
                      baseline_mean: np.ndarray, baseline_scores_mean: np.ndarray) -> np.ndarray:
    """Per-trial scalar dispersion from the baseline centroid, in the
    baseline PCA frame. Uses the per-TIMEPOINT projected norm, averaged over
    the retention window within each trial -- not the norm of the
    time-averaged vector, which would be near-zero for oscillatory data
    regardless of a real amplitude change (same reasoning as
    _geometry_vs_baseline). One scalar per trial, so the existing cheap
    trial-label permutation scheme (as in modulation_from_outcomes) applies
    directly."""
    n, c, t = condition_trials.shape
    flat = condition_trials.transpose(0, 2, 1).reshape(-1, c)  # (n*t, C)
    scores = (flat - baseline_mean) @ baseline_components
    norms = np.linalg.norm(scores - baseline_scores_mean, axis=1).reshape(n, t)
    return norms.mean(axis=1)


def _geometric_modulation(per_trial_drift_by_code: dict[int, np.ndarray],
                           rng: np.random.Generator) -> dict:
    """Single-cycle-DFT modulation depth of per-trial geometric drift across
    the six phase conditions, with a trial-label-shuffle permutation null --
    mirrors modulation_from_outcomes in run_haslacher_phase_omega.py, applied
    to a geometric quantity instead of behavioral accuracy."""
    codes_ordered = sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c])
    if any(len(per_trial_drift_by_code.get(c, [])) == 0 for c in codes_ordered):
        return {"depth": None, "optimal_phase_deg": None, "p_value": None,
                "reason": "at least one phase condition has zero trials"}

    means_obs = np.array([per_trial_drift_by_code[c].mean() for c in codes_ordered])
    depth_obs, phase_obs = _modulation(means_obs)

    all_values = np.concatenate([per_trial_drift_by_code[c] for c in codes_ordered])
    all_codes = np.concatenate([np.full(len(per_trial_drift_by_code[c]), c)
                                 for c in codes_ordered])
    null = np.empty(N_PERM_MODULATION)
    for p in range(N_PERM_MODULATION):
        shuffled = rng.permutation(all_codes)
        means_p = np.array([all_values[shuffled == c].mean() if np.any(shuffled == c) else np.nan
                            for c in codes_ordered])
        null[p] = _modulation(means_p)[0] if not np.any(np.isnan(means_p)) else np.nan
    valid = null[~np.isnan(null)]
    p_value = permutation_pvalue(valid >= depth_obs) if len(valid) else float("nan")
    return {"depth": float(depth_obs), "optimal_phase_deg": float(np.degrees(phase_obs)),
            "p_value": float(p_value), "n_trials": int(len(all_values))}


def run_participant(subject: str, group: str) -> dict:
    try:
        no_stim, stim, n_sass = _preprocess(subject, group)
    except (FileNotFoundError, OSError) as e:
        return {"status": "error", "reason": str(e)}

    baseline_trials = _retention_trials(no_stim)[0]  # (N, C, T)
    condition_trials = _retention_trials(stim, codes=list(PHASE_CONDITIONS))

    rng = np.random.default_rng(stable_seed(f"haslacher_stim_geometry_{subject}"))
    # parallel_analysis, not cv_pr: scalp EEG's per-timepoint spatial
    # covariance is dominated by a handful of large, volume-conduction-
    # blurred generators regardless of passband (confirmed here: this
    # persisted after switching from alpha-only to broadband filtering), so
    # the magnitude-weighted participation-ratio statistic still rounds to
    # k=1 -- trivializing subspace_overlap and v*/alignment (a 1-D subspace
    # trivially "overlaps" and "aligns" with anything). Alagapan's iEEG does
    # not show this (contacts are close to sources, cv_pr there gives
    # k=16-21) -- this is a property of the scalp-EEG modality, not of any
    # one preprocessing choice. parallel_analysis (how many components clear
    # a shuffle noise floor, not magnitude-weighted) is the selector that
    # answers "how many dimensions does this subspace comparison need".
    k_info = select_latent_dim(baseline_trials, method="parallel_analysis", rng=rng)
    k = k_info["k"]

    baseline_pooled = baseline_trials.transpose(0, 2, 1).reshape(-1, baseline_trials.shape[1])
    baseline_scores, baseline_components, _ = pca_decompose(baseline_pooled, k)
    baseline_mean = baseline_pooled.mean(axis=0)
    baseline_scores_mean = baseline_scores.mean(axis=0)
    pr_baseline = pca_participation_ratio(baseline_pooled)

    stim_pooled = np.concatenate([v for v in condition_trials.values() if len(v)], axis=0)
    stim_pooled_flat = stim_pooled.transpose(0, 2, 1).reshape(-1, stim_pooled.shape[1])
    overall = _geometry_vs_baseline(baseline_pooled, stim_pooled_flat, baseline_scores,
                                     baseline_components, baseline_mean, k)
    overall["participation_ratio_baseline"] = pr_baseline

    per_trial_drift_by_code = {
        c: _per_trial_drift(v, baseline_components, baseline_mean, baseline_scores_mean)
        for c, v in condition_trials.items() if len(v)
    }
    modulation = _geometric_modulation(per_trial_drift_by_code, rng)

    srate = float(no_stim.info["sfreq"])
    dyn = fit_retention_dynamics(baseline_trials, srate, k, rng)

    chan_weight = _stimulation_channel_weight(group, no_stim.ch_names)
    stim_input = None
    if chan_weight is not None:
        align_rng = np.random.default_rng(stable_seed(f"haslacher_random_dirs_{subject}"))
        stim_input = stimulation_input_alignment(
            dyn["A"], dyn["components"], chan_weight["weight"],
            dyn["v_star"], dyn["v_stable"], align_rng,
            gramian_horizon=GRAMIAN_HORIZON, n_random_dirs=N_RANDOM_DIRS,
        )
        stim_input["n_stim_electrodes_found"] = chan_weight["n_electrodes_found"]
        stim_input["n_stim_electrodes_expected"] = chan_weight["n_electrodes_expected"]

    return {
        "status": "ok", "group": group, "n_sass_components_removed": n_sass,
        "k_latent_dim": k_info, "n_baseline_trials": int(baseline_trials.shape[0]),
        "n_stim_trials": int(stim_pooled.shape[0]),
        "geometry_stim_pooled_vs_baseline": overall,
        "geometric_modulation": modulation,
        "dynamics": {k_: v_ for k_, v_ in dyn.items() if k_ not in ("A", "components", "mean",
                                                                     "v_star", "v_stable")},
        "stimulation_input_alignment": stim_input,
    }


def run_group(subjects: list[str], group_name: str) -> dict:
    out = {}
    for subject in subjects:
        print(f"  {subject} ({group_name}) ...")
        out[subject] = run_participant(subject, group_name)
        r = out[subject]
        if r.get("status") == "ok":
            mod = r["geometric_modulation"]
            dyn = r["dynamics"]
            si = r["stimulation_input_alignment"]
            si_str = (f"alignment_to_vstar={si['alignment_to_vstar']:.3f} "
                      f"(null={si['random_direction_alignment']:.3f}) "
                      f"gramian_trace={si['gramian_trace']:.3e}") if si else "no stim electrodes survived"
            print(f"    subspace_overlap={r['geometry_stim_pooled_vs_baseline']['subspace_overlap']:.3f} "
                  f"dispersion_shift={r['geometry_stim_pooled_vs_baseline']['dispersion_shift_in_baseline_pca_frame']:.3f} "
                  f"modulation depth={mod.get('depth')} p={mod.get('p_value')}")
            print(f"      dynamics: max_real_eig={dyn['max_real_eig']:.3f} r2_cv={dyn['r2_cv']:.3f} "
                  f"r2_null={dyn['r2_null']:.3f} identifiable={dyn['identifiable']} | {si_str}")
        else:
            print(f"    {r.get('reason')}")
    n_significant = sum(1 for v in out.values()
                        if v.get("geometric_modulation", {}).get("p_value") is not None
                        and v["geometric_modulation"]["p_value"] < 0.05)
    return {"per_subject": out, "n_subjects": len(subjects),
            "n_significant_geometric_modulation": n_significant}


def _modifier_value(geom: dict, key: str) -> float | None:
    if key == "geometric_modulation_depth":
        return geom.get("geometric_modulation", {}).get("depth")
    si = geom.get("stimulation_input_alignment")
    return si.get(key) if si else None


def _brain_behavior_link(geometry_by_subject: dict, phase_omega_path: Path) -> dict:
    """Correlate each participant's own geometry/DMD/LQR quantities against
    their own behavioral-modulation depth (results/haslacher_phase_omega.json),
    properly powered here (n up to 46) unlike the Alagapan n=3 case:
      geometric_modulation_depth -- per-trial dispersion-along-v* modulation
      alignment_to_vstar         -- does this participant's fixed stimulation
                                     site align with their own dynamics' v*
      gramian_trace              -- controllability energy the fitted plant
                                     affords along that stimulation input
    """
    if not phase_omega_path.exists():
        return {"status": "skipped", "reason": "results/haslacher_phase_omega.json not found; "
                "run scripts/run_haslacher_phase_omega.py first"}
    with open(phase_omega_path) as f:
        behavior = json.load(f)

    results = {}
    for modifier_key in ("geometric_modulation_depth", "alignment_to_vstar", "gramian_trace"):
        modifier_vals, behavioral_depths = [], []
        for group_name in ("active", "control"):
            for subject, geom in geometry_by_subject.get(group_name, {}).get("per_subject", {}).items():
                m_val = _modifier_value(geom, modifier_key)
                b_depth = behavior.get(group_name, {}).get("per_subject", {}) \
                                  .get(subject, {}).get("modulation", {}).get("depth")
                if m_val is not None and b_depth is not None:
                    modifier_vals.append(m_val)
                    behavioral_depths.append(b_depth)
        if len(modifier_vals) < 3:
            results[modifier_key] = {"status": "skipped",
                                      "reason": "fewer than 3 participants with both values"}
            continue
        result = pearson_permutation_test(np.array(modifier_vals), np.array(behavioral_depths))
        result["n"] = len(modifier_vals)
        results[modifier_key] = result
    return results


def main():
    print(f"Active group (n={len(ACTIVE_SUBJECTS)}) ...")
    active = run_group(ACTIVE_SUBJECTS, "active")
    print(f"\nControl group (n={len(CONTROL_SUBJECTS)}) ...")
    control = run_group(CONTROL_SUBJECTS, "control")

    print(f"\nActive: {active['n_significant_geometric_modulation']}/{active['n_subjects']} "
          f"significant geometric modulation")
    print(f"Control: {control['n_significant_geometric_modulation']}/{control['n_subjects']} "
          f"significant geometric modulation")

    link = _brain_behavior_link({"active": active, "control": control},
                                 RESULTS / "haslacher_phase_omega.json")
    for modifier_key, result in link.items():
        print(f"\n{modifier_key} vs. behavioral-modulation-depth: {result}")

    out = {"active": active, "control": control, "brain_behavior_link": link,
          "_meta": {"citation": "Haslacher et al. 2024, Brain Stimulation, "
                    "doi:10.1016/j.brs.2024.07.007",
                    "broadband_hz": list(BROADBAND_HZ),
                    "retention_window_s": [RETENTION_TMIN, RETENTION_TMAX],
                    "preprocessing": "saturated-channel rejection + SASS, per the dataset's "
                    "own Data/README.md (pyprep noisy-channel detection skipped: README "
                    "marks it optional, not installed in this environment); no band "
                    "restriction and stimulation electrodes kept (see module docstring)"}}
    with open(RESULTS / "haslacher_stimulation_geometry.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/haslacher_stimulation_geometry.json")


if __name__ == "__main__":
    main()
