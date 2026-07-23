#!/usr/bin/env python3
"""Behavioral performance-decoding CTG (Round-7, STEP B): "when during the
delay is trial outcome (correct vs error) predictable from the maintenance
population state?"

Mirrors the load/item CTG pipeline (spike_pipeline.load_vs_load_ctg /
item_identity_ctg) but the decoded label is TRIAL OUTCOME. Uses the SAME
underlying machinery (geometry.ctg_label_permutation_null -> ctg_nested_cv,
which folds PCA into cross-validation and scores AUC via the Mann-Whitney U
statistic == ROC-AUC, on StratifiedKFold folds) so decodability, the
permutation null, and the tau computation are identical in kind to every
other CTG result in the paper -- nothing new is implemented here, this
script only supplies a different label to existing functions.

Datasets with real, non-fabricated response_accuracy in the data available
to this project: Boran iEEG, Boran units (DANDI 000574), DANDI 000469, 001187,
000673. Two datasets in the Round-7 spec's inventory turned out NOT to have a
usable label on inspection (STOP-and-report, not fabricated):
  - Miller N-back: the raw MAT files (stim/task/target) encode task condition
    only (whether a response was REQUIRED on that trial), not whether the
    subject's response was correct -- no hit/miss/FA/CR field exists in the
    data mounted for this project. Excluded, not a fabricated proxy.
  - CRCNS pfc-3: no correct/error field wired into this project's existing
    pfc-3 pipeline, and it is a non-simultaneous pseudo-population (the same
    property that already makes PR/load-CTG not meaningful for it per
    DATASET_ANALYSIS_MATRIX.md footnote 6) -- spec marks it optional/
    secondary; skipped rather than force a weak decode on invalid trial-unit
    pairing.

Because outcome decoding needs a leakage-free per-fold PCA fit (unlike the
cached post-PCA Z arrays the geometry npz files store, which were fit on ALL
trials), this re-derives the z-scored PSTH/HGP per session directly from the
NWB files, reusing each cohort's own field names / windowing constants from
its existing pipeline script (not re-implementing them).

Outputs: results/behavior_ctg.json
Updates: results/all_statistics.json -- "behavior_ctg" key

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_behavior_ctg.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import h5py
import scipy.signal as sig

from spike_pipeline import load_spike_times, build_psth, low_rate_unit_mask, MIN_SESSION_ACCURACY
from geometry import ctg_label_permutation_null, temporal_stability_tau
from preprocessing import bandpass_filter, line_noise_notch, bipolar_reference_by_shank
from statistics import stable_seed
from io_utils import locked_json_update

RESULTS = ROOT / "results"
DATA_ROOT = Path("/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data")

MIN_ERROR_TRIALS = 15    # spec B2: dataset-level POOLED (across sessions) error-trial floor
MIN_ERROR_PER_SESSION = 4   # per-session floor to even attempt a fit: sessions differ in
                            # units, so features cannot be pooled across sessions into one
                            # matrix (same reason every sibling CTG in this project treats a
                            # session as its own analysis unit) -- this only needs enough
                            # error trials for CTG_N_SPLITS-fold StratifiedKFold to keep >=1
                            # error trial per fold; the MIN_ERROR_TRIALS floor above is what
                            # actually gates whether the DATASET is reported as underpowered,
                            # applied to the trial count pooled across all of a dataset's
                            # sessions (spec B2: "< 15 error trials pooled").
CTG_STEP = 3
CTG_N_SPLITS = 4         # imbalanced label -> fewer splits so each fold keeps >=1 error trial
CTG_N_PERM = 500
N_PC = 8


def _outcome_ctg(psth_z: np.ndarray, correct: np.ndarray, times: np.ndarray, rng) -> dict | None:
    """Shared core: outcome-decoding CTG + tau, on an already-built (N,U,T)
    z-scored PSTH/HGP and a (N,) bool correct/error label. Gated on the
    PER-SESSION floor only (MIN_ERROR_PER_SESSION) -- the dataset-level
    MIN_ERROR_TRIALS pooled floor is applied later in _pool_dataset."""
    n_correct, n_error = int(correct.sum()), int((~correct).sum())
    if n_error < MIN_ERROR_PER_SESSION:
        return {"underpowered": True, "n_trials": int(len(correct)),
                "n_correct": n_correct, "n_error": n_error}
    y = (~correct).astype(int)   # 1 = error, so "diagonal predictable" reads as "error detectable"
    t_idx = np.arange(0, psth_z.shape[2], CTG_STEP)
    n_comp = min(N_PC, psth_z.shape[1] - 2)
    if n_comp < 2:
        return {"underpowered": True, "n_trials": int(len(correct)),
                "n_correct": n_correct, "n_error": n_error}
    res = ctg_label_permutation_null(psth_z, y, t_idx, n_components=n_comp,
                                     n_splits=CTG_N_SPLITS, n_perm=CTG_N_PERM, rng=rng)
    tau_info = temporal_stability_tau(res["auc_mat"])
    auc_mat = res["auc_mat"]
    diag = np.diag(auc_mat)
    peak_i = int(np.nanargmax(diag))
    return {
        "underpowered": False,
        "diag_auc_peak": float(diag[peak_i]),
        "peak_time_s": float(times[t_idx[peak_i]]) if peak_i < len(t_idx) else float("nan"),
        "offdiag_auc": tau_info["mean_offdiag_auc"],
        "tau": tau_info["tau"],
        "tau_interpretable": tau_info["interpretable"],
        "p_perm": res["p_value"],
        "n_trials": int(len(correct)), "n_correct": n_correct, "n_error": n_error,
        "ctg_matrix": auc_mat.tolist(),
        "t_idx_times": times[t_idx].tolist(),
    }


# ── Spike-based cohorts (000469, 001187, 000673, Boran units) ────────────────

def _spike_session_outcome_ctg(f, trials_group: str, loads_field: str, maint_win: float,
                               min_units: int, rng) -> dict | None:
    if trials_group not in f.get("intervals", {}) or "units" not in f:
        return None
    n_units_raw = int(f["units/id"].shape[0])
    if n_units_raw < min_units:
        return None
    trials = f[f"intervals/{trials_group}"]
    t_maint = trials["timestamps_Maintenance"][:]
    response_acc = trials["response_accuracy"][:].astype(bool)
    spike_lists = load_spike_times(f)

    if response_acc.mean() < MIN_SESSION_ACCURACY:
        return None
    rate_mask = low_rate_unit_mask(spike_lists, t_maint, maint_win)
    n_units = int(rate_mask.sum())
    if n_units < min_units:
        return None
    spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]

    times = np.arange(50, maint_win * 1000, 100) / 1000.0
    psth = build_psth(spike_lists, t_maint, bin_ms=100, smooth_ms=200, window_s=maint_win)
    mu = psth.mean(axis=0, keepdims=True)
    sd = psth.std(axis=0, keepdims=True) + 1e-8
    psth_z = (psth - mu) / sd
    return _outcome_ctg(psth_z, response_acc, times, rng)


def run_rutishauser_lineage(dataset_key: str, glob_pattern: str, trials_group: str,
                            maint_win: float, min_units: int) -> dict:
    data_dir = DATA_ROOT / dataset_key.replace("dandi", "")
    out = {}
    files = sorted(data_dir.glob(glob_pattern))
    for fp in files:
        key = fp.stem
        with h5py.File(str(fp), "r") as f:
            try:
                row = _spike_session_outcome_ctg(f, trials_group, "loads", maint_win, min_units,
                                                 np.random.default_rng(stable_seed(f"behctg_{key}")))
            except (KeyError, ValueError) as e:
                print(f"    SKIP {key}: {e}")
                continue
        if row is None:
            continue
        out[key] = row
        _print_row(f"{dataset_key}/{key}", row)
    return out


def run_boran_units() -> dict:
    data_dir = DATA_ROOT / "000574"
    out = {}
    for subj_dir in sorted(data_dir.glob("sub-*")):
        for fp in sorted(subj_dir.glob("*.nwb")):
            key = fp.stem
            with h5py.File(str(fp), "r") as f:
                if "units" not in f or f["units/id"].shape[0] < 8:
                    continue
                n_units = int(f["units/id"].shape[0])
                trials = f["intervals/trials"]
                start_time = trials["start_time"][:]
                correct = trials["correct"][:].astype(bool)
                artifact = trials["artifact"][:].astype(bool)
                spike_lists = load_spike_times(f)
            good = ~artifact
            start_time, correct = start_time[good], correct[good]
            maint_onsets = start_time + 3.0
            if len(maint_onsets) < 20:
                continue
            times = np.arange(50, 3000, 100) / 1000.0
            psth = build_psth(spike_lists, maint_onsets, bin_ms=100, smooth_ms=200, window_s=3.0)
            mu = psth.mean(axis=0, keepdims=True)
            sd = psth.std(axis=0, keepdims=True) + 1e-8
            psth_z = (psth - mu) / sd
            row = _outcome_ctg(psth_z, correct, times,
                               np.random.default_rng(stable_seed(f"behctg_boranunits_{key}")))
            out[key] = row
            _print_row(f"boran_units/{key}", row)
    return out


# ── Boran iEEG (HGP; small self-contained copy of run_boran_pipeline's
#    loading steps -- that module runs its full 9-subject pipeline at import
#    time, so importing it here would trigger an expensive unwanted re-run) ──

def run_boran_ieeg() -> dict:
    data_dir = DATA_ROOT / "000574"
    srate = 1398.0
    t_pre_maint, t_post_maint, t_epoch_pre = 3.0, 3.0, 1.0
    epoch_total = t_pre_maint + t_post_maint + t_epoch_pre
    out = {}
    for subj_dir in sorted(data_dir.glob("sub-*")):
        subj = subj_dir.name
        all_epochs, all_correct = [], []
        electrode_labels = None
        for nwb_path in sorted(subj_dir.glob("*.nwb")):
            with h5py.File(str(nwb_path), "r") as f:
                raw = f["acquisition/ecephys.ieeg/data"][:]
                times = f["acquisition/ecephys.ieeg/timestamps"][:]
                t_start_arr = f["intervals/trials/start_time"][:]
                correct = f["intervals/trials/correct"][:].astype(bool)
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
                all_epochs.append(raw[i0:i1].T.astype(np.float32))
                all_correct.append(correct[trial_idx])
        if len(all_epochs) < 20:
            continue
        epochs = np.stack(all_epochs, axis=0)
        correct = np.array(all_correct, dtype=bool)
        n_epoch_samp = int(epoch_total * srate)
        if epochs.shape[2] != n_epoch_samp:
            from scipy.signal import resample
            epochs = resample(epochs, n_epoch_samp, axis=2)

        # Notch (50/100/150 Hz) + bipolar-by-shank reref -- Round-8 7A/7B fix,
        # replacing the prior median-CAR (see run_boran_pipeline.py for the
        # canonical version of this step).
        N0, C_raw, T0 = epochs.shape
        for n in range(N0):
            epochs[n] = line_noise_notch(epochs[n].T, srate, fundamental=50.0, n_harmonics=3).T
        X_flat = epochs.transpose(0, 2, 1).reshape(-1, C_raw)
        X_bp, _ = bipolar_reference_by_shank(X_flat, electrode_labels)
        epochs = X_bp.reshape(N0, T0, -1).transpose(0, 2, 1).astype(np.float32)
        N, C, T = epochs.shape
        hgp = np.zeros_like(epochs, dtype=np.float32)
        smooth_s = int(50e-3 * srate)
        kernel = sig.windows.gaussian(smooth_s * 6 + 1, std=smooth_s)
        kernel /= kernel.sum()
        for n in range(N):
            filtered = bandpass_filter(epochs[n].T, 70.0, 150.0, srate)
            power = np.abs(sig.hilbert(filtered, axis=0)) ** 2
            for c in range(C):
                power[:, c] = np.convolve(power[:, c], kernel, mode="same")
            hgp[n] = power.T.astype(np.float32)

        bl_samps = int(t_epoch_pre * srate)
        bl = hgp[:, :, :bl_samps]
        mu = bl.mean(axis=(0, 2), keepdims=True)
        sd = bl.std(axis=(0, 2), keepdims=True) + 1e-10
        hgp = (hgp - mu) / sd

        flat = hgp.reshape(C, -1)
        mad = np.median(np.abs(flat - np.median(flat, axis=1, keepdims=True)), axis=1)
        good_ch = mad <= 5 * 1.4826 * np.median(mad)
        hgp = hgp[:, good_ch, :]

        maint_start_s = int((t_epoch_pre + t_pre_maint) * srate)
        maint_end_s = int((t_epoch_pre + t_pre_maint + t_post_maint) * srate)
        hgp_maint = hgp[:, :, maint_start_s:maint_end_s]
        # Downsample the CTG time axis to a manageable stride at native 1398Hz
        # (same ~0.2s stride run_boran_pipeline.py uses for its own CTG_STEP).
        step_native = 280
        t_idx_native = np.arange(0, hgp_maint.shape[2], step_native)
        hgp_ds = hgp_maint[:, :, t_idx_native]
        times_ds = np.linspace(0.0, t_post_maint, hgp_maint.shape[2])[t_idx_native]

        row = _outcome_ctg_native(hgp_ds, correct, times_ds,
                                  np.random.default_rng(stable_seed(f"behctg_boranieeg_{subj}")))
        out[subj] = row
        _print_row(f"boran_ieeg/{subj}", row)
    return out


def _outcome_ctg_native(hgp_ds: np.ndarray, correct: np.ndarray, times_ds: np.ndarray, rng) -> dict:
    """Like _outcome_ctg but the time axis is ALREADY downsampled (Boran iEEG's
    native rate is far too fine to stride post-hoc the way spike-PSTH bins are),
    so t_idx here is every column, not a further CTG_STEP subsample."""
    n_correct, n_error = int(correct.sum()), int((~correct).sum())
    if n_error < MIN_ERROR_PER_SESSION:
        return {"underpowered": True, "n_trials": int(len(correct)),
                "n_correct": n_correct, "n_error": n_error}
    y = (~correct).astype(int)
    t_idx = np.arange(hgp_ds.shape[2])
    n_comp = min(N_PC, hgp_ds.shape[1] - 2)
    res = ctg_label_permutation_null(hgp_ds, y, t_idx, n_components=n_comp,
                                     n_splits=CTG_N_SPLITS, n_perm=CTG_N_PERM, rng=rng)
    tau_info = temporal_stability_tau(res["auc_mat"])
    auc_mat = res["auc_mat"]
    diag = np.diag(auc_mat)
    peak_i = int(np.nanargmax(diag))
    return {
        "underpowered": False,
        "diag_auc_peak": float(diag[peak_i]),
        "peak_time_s": float(times_ds[peak_i]),
        "offdiag_auc": tau_info["mean_offdiag_auc"],
        "tau": tau_info["tau"],
        "tau_interpretable": tau_info["interpretable"],
        "p_perm": res["p_value"],
        "n_trials": int(len(correct)), "n_correct": n_correct, "n_error": n_error,
        "ctg_matrix": auc_mat.tolist(),
        "t_idx_times": times_ds.tolist(),
    }


def _print_row(label: str, row: dict) -> None:
    if row.get("underpowered"):
        print(f"    {label}: skip (n_error={row['n_error']} < {MIN_ERROR_PER_SESSION} per-session floor)")
    else:
        print(f"    {label}: diag_auc_peak={row['diag_auc_peak']:.3f} @t={row['peak_time_s']:.2f}s "
              f"tau={row['tau']:.3f} p={row['p_perm']:.4f} "
              f"(n_correct={row['n_correct']}, n_error={row['n_error']})")


def _pool_dataset(per_session: dict, label: str) -> dict:
    """Pool a dataset's per-session outcome-CTG into one dataset-level row
    (spec B3 schema is dataset-keyed, not session-keyed) via a trial-weighted
    mean of the diagonal/offdiag effects and Stouffer-combined significance,
    matching the pooling style already used elsewhere in this project
    (content_ctg_pooled / stouffer_combine).

    Sessions differ in units/channels, so per-trial FEATURES cannot be pooled
    across sessions into one decoder (same reason every sibling CTG in this
    project is a per-session analysis unit) -- each session is fit separately
    at a low per-session floor (MIN_ERROR_PER_SESSION), and the spec's B2
    dataset-level "< 15 error trials pooled" floor is applied here to the
    error-trial count POOLED ACROSS all of a dataset's sessions."""
    from statistics import stouffer_combine

    powered = {k: v for k, v in per_session.items() if not v.get("underpowered")}
    n_correct_total = sum(v["n_correct"] for v in per_session.values())
    n_error_total = sum(v["n_error"] for v in per_session.values())
    n_trials_total = sum(v["n_trials"] for v in per_session.values())
    if not powered or n_error_total < MIN_ERROR_TRIALS:
        return {"underpowered": True, "n_sessions": len(per_session), "n_sessions_powered": len(powered),
                "n_trials": n_trials_total, "n_correct": n_correct_total, "n_error": n_error_total}

    weights = np.array([v["n_trials"] for v in powered.values()], dtype=float)
    diag = np.array([v["diag_auc_peak"] for v in powered.values()])
    offdiag = np.array([v["offdiag_auc"] for v in powered.values()])
    tau = np.array([v["tau"] for v in powered.values() if np.isfinite(v["tau"])])
    peak_t = np.array([v["peak_time_s"] for v in powered.values()])
    p_vals = np.array([v["p_perm"] for v in powered.values()])
    stouffer = stouffer_combine(p_vals)

    # Representative CTG matrix (largest-N session) for the figure, per spec B3
    # "ctg_matrix (optional, for figure)".
    rep_key = max(powered, key=lambda k: powered[k]["n_trials"])

    return {
        "underpowered": False,
        "diag_auc_peak": float(np.average(diag, weights=weights)),
        "peak_time_s": float(np.average(peak_t, weights=weights)),
        "offdiag_auc": float(np.average(offdiag, weights=weights)),
        "tau": float(np.mean(tau)) if len(tau) else float("nan"),
        "p_perm": float(stouffer["p_combined"]),
        "n_trials": n_trials_total, "n_correct": n_correct_total, "n_error": n_error_total,
        "n_sessions": len(per_session), "n_sessions_powered": len(powered),
        "ctg_matrix": powered[rep_key]["ctg_matrix"], "ctg_matrix_session": rep_key,
        "label": label,
    }


def main():
    out_per_session = {}
    out = {}

    print("Boran iEEG...")
    boran_ieeg = run_boran_ieeg()
    out_per_session["boran_ieeg"] = boran_ieeg
    out["boran_ieeg"] = _pool_dataset(boran_ieeg, "Boran iEEG")

    print("Boran units (DANDI 000574)...")
    boran_units = run_boran_units()
    out_per_session["boran_units"] = boran_units
    out["boran_units"] = _pool_dataset(boran_units, "Boran units")

    print("DANDI 000469...")
    d469 = run_rutishauser_lineage("dandi000469", "sub-*/*_ses-2_ecephys+image.nwb",
                                   "trials", 2.3, 15)
    out_per_session["dandi000469"] = d469
    out["dandi000469"] = _pool_dataset(d469, "DANDI 000469")

    print("DANDI 001187...")
    d1187 = run_rutishauser_lineage("dandi001187", "sub-*/*_ecephys*.nwb", "WM_trials", 2.3, 15)
    out_per_session["dandi001187"] = d1187
    out["dandi001187"] = _pool_dataset(d1187, "DANDI 001187")

    print("DANDI 000673...")
    d0673 = run_rutishauser_lineage("dandi000673", "sub-*/*_ecephys*.nwb", "trials", 2.3, 15)
    out_per_session["dandi000673"] = d0673
    out["dandi000673"] = _pool_dataset(d0673, "DANDI 000673")

    print("\nMiller N-back: EXCLUDED -- no hit/miss/FA/CR field in the raw MAT "
          "files (stim/task/target encode task condition, not response accuracy); "
          "STOP-and-report, not fabricated.")
    print("CRCNS pfc-3: SKIPPED (optional/secondary per spec; non-simultaneous "
          "pseudo-population, no correct/error field wired into this project's "
          "existing pfc-3 pipeline).")

    print("\n=== Dataset-pooled summary ===")
    for ds, row in out.items():
        if row.get("underpowered"):
            print(f"  {ds}: UNDERPOWERED (n_error={row['n_error']} pooled < {MIN_ERROR_TRIALS}, "
                  f"n_sessions_powered={row.get('n_sessions_powered', 0)})")
        else:
            print(f"  {ds}: diag_auc_peak={row['diag_auc_peak']:.3f} tau={row['tau']:.3f} "
                  f"p={row['p_perm']:.4g} n_error={row['n_error']} "
                  f"({row['n_sessions_powered']}/{row['n_sessions']} sessions powered)")

    with open(RESULTS / "behavior_ctg.json", "w") as f:
        import json
        json.dump(out, f, indent=2)
    with open(RESULTS / "behavior_ctg_per_session.json", "w") as f:
        import json
        json.dump(out_per_session, f, indent=2)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["behavior_ctg"] = out
    print("\nSaved results/behavior_ctg.json, results/behavior_ctg_per_session.json, "
          "updated all_statistics.json")


if __name__ == "__main__":
    main()
