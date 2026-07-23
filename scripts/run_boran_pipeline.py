#!/usr/bin/env python3
"""Full Boran Sternberg iEEG pipeline — replicates all Miller analyses.

Outputs:
  results/boran_geometry_{subj}.npz  — Z (N,T,8), set_sizes, times, correct
  results/boran_ctg_{subj}.npz       — CTG matrix (set4 vs set8), t_idx
  results/boran_summary.json         — τ, PR, phase stats per subject

Run:
    conda run -n wm_dynamics python scripts/run_boran_pipeline.py
"""

import sys, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import h5py
import scipy.signal as sig
from preprocessing import (bandpass_filter, line_noise_notch,
                           bipolar_reference_by_shank, shank_bipolar_index_pairs)
from geometry import (ctg_label_permutation_null, temporal_stability_tau,
                      spatiotemporal_participation_ratio, geometric_drift)
from statistics import linear_mixed_effects_test, fdr_bh, stable_seed, paired_sign_flip_test
from causal import dml_partial_linear, e_value

BORAN_DIR = Path("/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/000574")
RESULTS   = ROOT / "results"
SUBJECTS  = [f"sub-0{i}" for i in range(1, 10)]
SRATE     = 1398.0   # Hz

# Epoch: relative to TRIAL START.  Maintenance = [3.0, 6.0]s; Probe = 6.0s.
# We extract [-0.5, 6.5]s around trial start to include baseline + full delay + early probe.
T_PRE_MAINT = 3.0   # seconds before maintenance onset to keep (encoding + fixation)
T_POST_MAINT = 3.0  # maintenance window length
T_EPOCH_PRE  = 1.0  # include 1s pre-fixation as baseline for normalization
EPOCH_TOTAL  = T_PRE_MAINT + T_POST_MAINT + T_EPOCH_PRE  # 7 s total epoch

N_COMPONENTS = 8
CTG_STEP     = 280   # ~0.2 s at 1398 Hz → ~15 timepoints across 3s maintenance


def load_subject_sessions(subj: str) -> dict:
    """Load and concatenate all sessions for one Boran subject.

    Also extracts iEEG electrode MNI coordinates from the first NWB file
    (electrode positions are consistent across sessions for one subject).
    """
    nwbs = sorted((BORAN_DIR / subj).glob("*.nwb"))
    print(f"  {subj}: {len(nwbs)} sessions")

    all_epochs, all_sizes, all_correct, all_rt = [], [], [], []
    electrode_mni_all = None   # (C_ieeg, 3) from first session
    electrode_labels_all = None  # list[str] len C_ieeg, for bipolar shank parsing

    for nwb_path in nwbs:
        with h5py.File(str(nwb_path), "r") as f:
            raw   = f["acquisition/ecephys.ieeg/data"][:]         # (T_total, C)
            times = f["acquisition/ecephys.ieeg/timestamps"][:]   # (T_total,)

            t_start_arr = f["intervals/trials/start_time"][:]
            set_sizes   = f["intervals/trials/set_size"][:].astype(int)
            correct     = f["intervals/trials/correct"][:].astype(bool)
            artifact    = f["intervals/trials/artifact"][:].astype(bool)
            # response_time is an absolute NWB timestamp, not a trial-relative
            # duration — subtract trial start to get response latency (RT).
            response_time = f["intervals/trials/response_time"][:] - t_start_arr

            # Extract electrode MNI coords + labels once
            if electrode_mni_all is None:
                try:
                    x = f["general/extracellular_ephys/electrodes/x"][:]
                    y = f["general/extracellular_ephys/electrodes/y"][:]
                    z = f["general/extracellular_ephys/electrodes/z"][:]
                    ieeg_idx = f["acquisition/ecephys.ieeg/electrodes"][:]
                    electrode_mni_all = np.column_stack(
                        [x[ieeg_idx], y[ieeg_idx], z[ieeg_idx]]
                    ).astype(np.float32)   # (C_ieeg, 3)
                    labels_full = [l.decode() for l in
                                    f["general/extracellular_ephys/electrodes/label"][:]]
                    electrode_labels_all = [labels_full[i] for i in ieeg_idx]
                except KeyError:
                    electrode_mni_all = np.full((raw.shape[1], 3), np.nan,
                                                dtype=np.float32)
                    electrode_labels_all = [f"ch{i}" for i in range(raw.shape[1])]

        srate_actual = 1.0 / np.median(np.diff(times))
        n_samp = int(EPOCH_TOTAL * srate_actual)
        n_pre  = int(T_EPOCH_PRE * srate_actual)

        for trial_idx, t0 in enumerate(t_start_arr):
            if artifact[trial_idx]:
                continue
            i0 = np.searchsorted(times, t0) - n_pre
            i1 = i0 + n_samp
            if i0 < 0 or i1 > raw.shape[0]:
                continue
            epoch = raw[i0:i1].T.astype(np.float32)  # (C, T)
            all_epochs.append(epoch)
            all_sizes.append(set_sizes[trial_idx])
            all_correct.append(correct[trial_idx])
            all_rt.append(response_time[trial_idx])

    if not all_epochs:
        return None

    epochs = np.stack(all_epochs, axis=0)   # (N, C, T)
    n_epoch_samp = int(EPOCH_TOTAL * SRATE)
    if epochs.shape[2] != n_epoch_samp:
        from scipy.signal import resample
        epochs = resample(epochs, n_epoch_samp, axis=2)

    return {
        "epochs":         epochs,
        "set_sizes":      np.array(all_sizes, dtype=int),
        "correct":        np.array(all_correct, dtype=bool),
        "response_time":  np.array(all_rt, dtype=float),
        "electrode_mni":  electrode_mni_all,  # (C_ieeg, 3) all iEEG channels
        "electrode_labels": electrode_labels_all,  # list[str] len C_ieeg
    }


def compute_hgp_epochs(epochs: np.ndarray) -> np.ndarray:
    """High-gamma power (70-150 Hz, 50ms smooth) for Boran epoched data.

    Parameters
    ----------
    epochs : (N, C, T) raw iEEG at SRATE

    Returns
    -------
    hgp : (N, C, T) smoothed HGP
    """
    N, C, T = epochs.shape
    hgp = np.zeros_like(epochs, dtype=np.float32)
    smooth_s = int(50e-3 * SRATE)
    kernel = sig.windows.gaussian(smooth_s * 6 + 1, std=smooth_s)
    kernel /= kernel.sum()

    for n in range(N):
        filtered = bandpass_filter(epochs[n].T, 70.0, 150.0, SRATE)  # (T, C)
        analytic = sig.hilbert(filtered, axis=0)
        power = np.abs(analytic) ** 2  # (T, C)
        for c in range(C):
            power[:, c] = np.convolve(power[:, c], kernel, mode="same")
        hgp[n] = power.T.astype(np.float32)

    return hgp


def baseline_normalize(hgp: np.ndarray, baseline_samps: int) -> np.ndarray:
    """Z-score each channel using the first baseline_samps samples."""
    bl = hgp[:, :, :baseline_samps]
    mu = bl.mean(axis=(0, 2), keepdims=True)
    sd = bl.std(axis=(0, 2), keepdims=True) + 1e-10
    return (hgp - mu) / sd


def channel_rejection(hgp: np.ndarray) -> np.ndarray:
    """Return boolean mask of good channels (MAD threshold = 5)."""
    flat = hgp.reshape(hgp.shape[1], -1)  # (C, N*T)
    mad = np.median(np.abs(flat - np.median(flat, axis=1, keepdims=True)), axis=1)
    threshold = 5 * 1.4826 * np.median(mad)
    return mad <= threshold


def fit_pca(hgp_maint: np.ndarray, n_comp: int = N_COMPONENTS):
    """PCA on maintenance-window HGP.

    Parameters
    ----------
    hgp_maint : (N, C, T_maint) — maintenance window only

    Returns
    -------
    Z   : (N, T_maint, n_comp)
    V   : (C, n_comp) — loading vectors
    """
    N, C, T = hgp_maint.shape
    X = hgp_maint.transpose(0, 2, 1).reshape(-1, C)  # (N*T, C)
    X -= X.mean(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:n_comp].T  # (C, n_comp)
    Z_flat = X @ V     # (N*T, n_comp)
    Z = Z_flat.reshape(N, T, n_comp)
    return Z, V


def rayleigh_test(phases: np.ndarray):
    N = len(phases)
    R = float(np.abs(np.mean(np.exp(1j * phases))))
    Z = N * R ** 2
    p = float(np.exp(-Z) * (1 + (2*Z - Z**2)/(4*N)))
    mu = float(np.angle(np.mean(np.exp(1j * phases))) % (2 * np.pi))
    return {"R": R, "Z_stat": Z, "p_value": p, "mean_direction": mu, "N": N}


# ─── Main pipeline ────────────────────────────────────────────────────────────

summary = {}

CTG_N_SPLITS = 5
CTG_N_PERM   = 100
PR_MIN_TRIALS_PER_GROUP = 8

pooled_drift, pooled_correct, pooled_subj, pooled_set_size, pooled_rt = [], [], [], [], []

for subj in SUBJECTS:
    print(f"\n{'='*55}")
    print(f"  Processing {subj}")
    print(f"{'='*55}")

    data = load_subject_sessions(subj)
    if data is None:
        print(f"  SKIP {subj} — no valid epochs")
        continue

    epochs        = data["epochs"]        # (N, C, T_epoch)
    set_sizes     = data["set_sizes"]
    correct       = data["correct"]
    response_time = data["response_time"]
    electrode_mni = data["electrode_mni"]       # (C_ieeg, 3)
    electrode_labels = data["electrode_labels"] # list[str] len C_ieeg

    N, C_raw, T_total = epochs.shape
    print(f"  Epochs: {N} trials, {C_raw} channels, {T_total} samples")
    print(f"  Set sizes: 4={((set_sizes==4).sum())}, 6={((set_sizes==6).sum())}, 8={((set_sizes==8).sum())}")
    print(f"  Accuracy: {100*correct.mean():.1f}%")

    # 1. Line-noise notch (50 Hz Zurich mains + harmonics), per trial -- a
    #    temporal filter, must not blend across trial boundaries (Round-8 7A
    #    fix: this HGP path -- 70-150 Hz -- previously had no notch at all,
    #    so the 100/150 Hz mains harmonics fell straight inside the band).
    print("  Notch (50/100/150 Hz)...")
    for n in range(N):
        epochs[n] = line_noise_notch(epochs[n].T, SRATE, fundamental=50.0, n_harmonics=3).T

    # 2. Bipolar re-reference along each depth shank -- a spatial operation,
    #    safe to vectorize across all trials/timepoints at once (Round-8 7B
    #    fix: previously only a global per-trial median-CAR, not bipolar,
    #    which is the sEEG field standard for depth electrodes).
    print("  Bipolar re-reference by shank...")
    X_flat = epochs.transpose(0, 2, 1).reshape(-1, C_raw)          # (N*T, C_raw)
    X_bp, bp_labels = bipolar_reference_by_shank(X_flat, electrode_labels)
    epochs = X_bp.reshape(N, T_total, -1).transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
    bp_pairs, bp_orphans = shank_bipolar_index_pairs(electrode_labels)
    electrode_mni_bp = np.array(
        [(electrode_mni[i0] + electrode_mni[i1]) / 2 for i0, i1, _ in bp_pairs]
        + [electrode_mni[i] for i in bp_orphans], dtype=np.float32
    )
    C = epochs.shape[1]
    print(f"  Channels after bipolar reref: {C} (from {C_raw}, {len(bp_pairs)} bipolar pairs "
          f"+ {len(bp_orphans)} CAR orphans)")

    # 3. HGP
    print("  Computing HGP...")
    hgp = compute_hgp_epochs(epochs)   # (N, C, T)

    # 4. Baseline normalize (first 1s = T_EPOCH_PRE)
    bl_samps = int(T_EPOCH_PRE * SRATE)
    hgp = baseline_normalize(hgp, bl_samps)

    # 5. Channel rejection
    good_ch = channel_rejection(hgp)
    n_rej = int((~good_ch).sum())
    print(f"  Channels: {good_ch.sum()}/{C} kept ({n_rej} rejected)")
    hgp = hgp[:, good_ch, :]
    good_ch_indices = np.where(good_ch)[0]  # indices of kept channels (into bipolar channel set)
    # Filter (bipolar-reduced) electrode MNI to good channels
    if len(electrode_mni_bp) == C:
        electrode_mni_good = electrode_mni_bp[good_ch_indices]
    else:
        electrode_mni_good = np.full((int(good_ch.sum()), 3), np.nan, dtype=np.float32)

    # 6. Maintenance window: t ∈ [T_EPOCH_PRE + T_PRE_MAINT, T_EPOCH_PRE + T_PRE_MAINT + T_POST_MAINT]
    #    = [1.0 + 3.0, 1.0 + 3.0 + 3.0]s = [4.0, 7.0]s from epoch start
    maint_start_s = int((T_EPOCH_PRE + T_PRE_MAINT) * SRATE)
    maint_end_s   = int((T_EPOCH_PRE + T_PRE_MAINT + T_POST_MAINT) * SRATE)
    hgp_maint = hgp[:, :, maint_start_s:maint_end_s]   # (N, C_good, T_maint)

    # Times relative to maintenance onset (for consistency with Miller t=0 at stimulus)
    T_maint = hgp_maint.shape[2]
    times_maint = np.linspace(0.0, T_POST_MAINT, T_maint)

    # Encoding-period window, same duration as maintenance — comparison
    # epoch (letters are being presented sequentially here; imperfect as a
    # negative control since set size trivially covaries with letter count
    # during encoding, but still informative to compare against maintenance).
    enc_start_s = int(T_EPOCH_PRE * SRATE)
    hgp_enc = hgp[:, :, enc_start_s:enc_start_s + T_maint]

    # Pre-stimulus baseline window — clean negative control: by task design
    # no information about the upcoming set size can exist yet, so both
    # diagonal and off-diagonal AUC should collapse to chance here. This
    # directly rules out "the pipeline always returns a square matrix
    # regardless of the underlying code."
    hgp_baseline = hgp[:, :, :bl_samps]

    # 6. PCA (fixed global basis — used for downstream geometry/drift/TES1,
    #    NOT for the CTG significance test, which folds PCA into CV below)
    print("  Fitting PCA...")
    Z, V = fit_pca(hgp_maint, n_comp=N_COMPONENTS)
    print(f"  Z shape: {Z.shape}")

    N_tr, T_m, D = Z.shape
    X_full = hgp_maint.transpose(0, 2, 1).reshape(-1, hgp_maint.shape[1])
    X_full -= X_full.mean(0)
    total_var = np.linalg.norm(X_full) ** 2
    proj_var  = np.linalg.norm(X_full @ V) ** 2
    var_ratio = float(proj_var / (total_var + 1e-10))
    print(f"  Variance in 8 PCs: {100*var_ratio:.1f}%")

    # 7. PR per set size — cross-validated spatiotemporal PR in the native
    #    (pre-PCA) channel space, the same method used for Miller, rather
    #    than PR of the pooled 8-D latent (which caps PR at 8 by construction).
    pr_per_set = {}
    for ss in [4, 6, 8]:
        mask = set_sizes == ss
        if mask.sum() < 5:
            pr_per_set[ss] = {"pr_cv": np.nan}
            continue
        pr_per_set[ss] = spatiotemporal_participation_ratio(
            hgp_maint[mask], n_splits=2, rng=np.random.default_rng(stable_seed(f"{subj}_pr_set{ss}"))
        )
    pr_cv_str = ", ".join(f"{k}: {v.get('pr_cv', float('nan')):.2f}" for k, v in pr_per_set.items())
    print(f"  PR (cv, native {hgp_maint.shape[1]}-channel space): {{{pr_cv_str}}}")

    # 8. CTG (set4 vs set8) with PCA folded into CV and a label-shuffle
    #    permutation null (valid: refits the whole matrix per permutation,
    #    not a t-test/bootstrap over non-independent off-diagonal cells).
    mask_ctg = (set_sizes == 4) | (set_sizes == 8)
    X_ctg = hgp_maint[mask_ctg]
    y_ctg = (set_sizes[mask_ctg] == 8).astype(int)
    t_idx = np.arange(0, T_maint, CTG_STEP)
    print(f"  CTG: {mask_ctg.sum()} trials ({(y_ctg==0).sum()} set4, {(y_ctg==1).sum()} set8), "
          f"{len(t_idx)} timepoints, nested-CV + {CTG_N_PERM}-perm label-shuffle null...")
    ctg_rng = np.random.default_rng(stable_seed(subj))
    ctg_res = ctg_label_permutation_null(
        X_ctg, y_ctg, t_idx, n_components=N_COMPONENTS,
        n_splits=CTG_N_SPLITS, n_perm=CTG_N_PERM, rng=ctg_rng,
    )
    auc_mat = ctg_res["auc_mat"]
    tau_info = temporal_stability_tau(auc_mat)
    print(f"  τ={tau_info['tau']:.4f} (interpretable={tau_info['interpretable']}), "
          f"offdiag_effect={ctg_res['mean_offdiag_auc_minus_chance']:.4f}, "
          f"p(label-shuffle)={ctg_res['p_value']:.4f}")

    # Dynamic-code negative control: identical CTG pipeline, same labels,
    # applied to the encoding-period window instead of maintenance.
    X_enc = hgp_enc[mask_ctg]
    enc_ctg_rng = np.random.default_rng(stable_seed(subj + "_1"))
    enc_res = ctg_label_permutation_null(
        X_enc, y_ctg, t_idx, n_components=N_COMPONENTS,
        n_splits=CTG_N_SPLITS, n_perm=50, rng=enc_ctg_rng,
    )
    enc_tau_info = temporal_stability_tau(enc_res["auc_mat"])
    print(f"  Comparison epoch (encoding period): τ={enc_tau_info['tau']:.4f}, "
          f"offdiag_effect={enc_res['mean_offdiag_auc_minus_chance']:.4f}, "
          f"p={enc_res['p_value']:.4f}")

    # Clean negative control: pre-stimulus baseline (chance expected)
    X_bl = hgp_baseline[mask_ctg]
    t_idx_bl = np.arange(0, hgp_baseline.shape[2], max(1, CTG_STEP // 2))
    bl_ctg_rng = np.random.default_rng(stable_seed(subj + "_2"))
    bl_res = ctg_label_permutation_null(
        X_bl, y_ctg, t_idx_bl, n_components=N_COMPONENTS,
        n_splits=CTG_N_SPLITS, n_perm=50, rng=bl_ctg_rng,
    )
    bl_tau_info = temporal_stability_tau(bl_res["auc_mat"])
    print(f"  Negative control (pre-stimulus baseline): "
          f"offdiag_effect={bl_res['mean_offdiag_auc_minus_chance']:.4f}, "
          f"p={bl_res['p_value']:.4f} (expect ~chance)")

    # Does the load/context code itself destabilize on error trials? Splits
    # the same set4-vs-8 CTG by trial outcome (correct-only vs error-only),
    # a direct test of code stability rather than distance-to-centroid.
    correct_ctg = data["correct"][mask_ctg]
    code_stability_outcome = None
    for outcome_name, outcome_mask in [("correct", correct_ctg), ("error", ~correct_ctg)]:
        y_o = y_ctg[outcome_mask]
        if min((y_o == 0).sum(), (y_o == 1).sum()) < 5:
            code_stability_outcome = None
            break
    else:
        oc_rng = np.random.default_rng(stable_seed(subj + "_correct"))
        oc_res = ctg_label_permutation_null(
            X_ctg[correct_ctg], y_ctg[correct_ctg], t_idx, n_components=N_COMPONENTS,
            n_splits=min(CTG_N_SPLITS, int(correct_ctg.sum()) // 2), n_perm=50, rng=oc_rng,
        )
        oe_rng = np.random.default_rng(stable_seed(subj + "_error"))
        oe_res = ctg_label_permutation_null(
            X_ctg[~correct_ctg], y_ctg[~correct_ctg], t_idx, n_components=N_COMPONENTS,
            n_splits=min(CTG_N_SPLITS, int((~correct_ctg).sum()) // 2), n_perm=50, rng=oe_rng,
        )
        code_stability_outcome = {
            "offdiag_effect_correct": oc_res["mean_offdiag_auc_minus_chance"],
            "offdiag_effect_error": oe_res["mean_offdiag_auc_minus_chance"],
            "n_correct": int(correct_ctg.sum()), "n_error": int((~correct_ctg).sum()),
        }
        print(f"  Code stability by outcome: offdiag_effect correct={oc_res['mean_offdiag_auc_minus_chance']:.4f} "
              f"(N={int(correct_ctg.sum())}) vs error={oe_res['mean_offdiag_auc_minus_chance']:.4f} "
              f"(N={int((~correct_ctg).sum())})")

    # 9. Phase distribution (2D endpoint at maintenance end)
    mask_set8 = set_sizes == 8
    t_end_idx = -1
    Z_end = Z[mask_set8][:, t_end_idx, :2]
    phases = np.arctan2(Z_end[:, 1], Z_end[:, 0]) % (2 * np.pi)
    rayleigh = rayleigh_test(phases)
    print(f"  Rayleigh: R={rayleigh['R']:.3f}, p={rayleigh['p_value']:.4f}")

    # 10. Correct vs. error: per-trial drift (pooled across subjects for an
    #     LME test below) and per-subject spatiotemporal PR split by outcome.
    drift = geometric_drift(Z, set_sizes, times_maint, maint_window=(0.0, T_POST_MAINT))
    pooled_drift.extend(drift.tolist())
    pooled_correct.extend(correct.astype(int).tolist())
    pooled_subj.extend([subj] * len(drift))
    pooled_set_size.extend(set_sizes.tolist())
    pooled_rt.extend(response_time.tolist())

    pr_correct_error = {"n_correct": int(correct.sum()), "n_error": int((~correct).sum())}
    if correct.sum() >= PR_MIN_TRIALS_PER_GROUP and (~correct).sum() >= PR_MIN_TRIALS_PER_GROUP:
        pr_correct_error["pr_correct"] = spatiotemporal_participation_ratio(
            hgp_maint[correct], n_splits=2, rng=np.random.default_rng(stable_seed(subj + "_pr_correct"))
        )
        pr_correct_error["pr_error"] = spatiotemporal_participation_ratio(
            hgp_maint[~correct], n_splits=2, rng=np.random.default_rng(stable_seed(subj + "_pr_error"))
        )
        print(f"  PR correct={pr_correct_error['pr_correct']['pr_cv']:.2f} vs "
              f"error={pr_correct_error['pr_error']['pr_cv']:.2f} "
              f"(n={pr_correct_error['n_correct']}/{pr_correct_error['n_error']})")
    else:
        print(f"  Correct/error PR skipped — too few error trials "
              f"(n_error={pr_correct_error['n_error']})")

    # 11. Save geometry file (includes V for TES1 B-matrix projection)
    np.savez_compressed(
        RESULTS / f"boran_geometry_{subj}.npz",
        Z=Z.astype(np.float32),
        V=V.astype(np.float32),              # (C_good, 8) — for B_latent = V.T @ B_electrode
        electrode_mni=electrode_mni_good,     # (C_good, 3) MNI coords of good channels
        good_ch_indices=good_ch_indices,
        set_sizes=set_sizes,
        correct=correct,
        response_time=response_time,
        drift=drift,
        times=times_maint,
        var_ratio=var_ratio,
        pr_per_set=pr_per_set,
        n_trials=N,
        n_channels_good=int(good_ch.sum()),
    )

    # 12. Save CTG file
    np.savez_compressed(
        RESULTS / f"boran_ctg_{subj}.npz",
        auc_mat=auc_mat,
        t_idx=t_idx,
        times_ctg=times_maint[t_idx],
        y_labels=y_ctg,
        null=ctg_res["null"],
        auc_mat_encoding_control=enc_res["auc_mat"],
        auc_mat_baseline_control=bl_res["auc_mat"],
    )

    summary[subj] = {
        "n_trials":    int(N),
        "n_channels":  int(good_ch.sum()),
        "accuracy":    float(correct.mean()),
        "var_ratio":   float(var_ratio),
        "pr_per_set":  {int(k): v for k, v in pr_per_set.items()},
        "tau":         tau_info["tau"],
        "tau_interpretable":  tau_info["interpretable"],
        "mean_diag_auc":      tau_info["mean_diag_auc"],
        "mean_offdiag_auc":   tau_info["mean_offdiag_auc"],
        "offdiag_effect":     ctg_res["mean_offdiag_auc_minus_chance"],
        "p_offdiag_vs_chance": ctg_res["p_value"],
        "comparison_epoch_encoding": {
            "tau": enc_tau_info["tau"],
            "offdiag_effect": enc_res["mean_offdiag_auc_minus_chance"],
            "p_value": enc_res["p_value"],
        },
        "negative_control_baseline": {
            "offdiag_effect": bl_res["mean_offdiag_auc_minus_chance"],
            "p_value": bl_res["p_value"],
        },
        "pr_correct_error": pr_correct_error,
        "code_stability_by_outcome": code_stability_outcome,
        "rayleigh":    rayleigh,
    }
    print(f"  Saved boran_geometry_{subj}.npz, boran_ctg_{subj}.npz")

# Save summary
with open(RESULTS / "boran_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# FDR correction across the 9-subject CTG-vs-chance test family
p_vals = np.array([v["p_offdiag_vs_chance"] for v in summary.values()])
fdr = fdr_bh(p_vals, alpha=0.05)
for sub, q in zip(summary.keys(), fdr["q_values"]):
    summary[sub]["q_value_fdr"] = float(q)
print(f"\n  FDR (BH): {fdr['n_reject']}/{len(p_vals)} subjects survive q<0.05")

# Pooled correct-vs-error drift test (LME-style permutation, subject as random effect)
drift_arr = np.array(pooled_drift)
correct_arr = np.array(pooled_correct, dtype=float)
subj_arr = np.array(pooled_subj)
lme_drift = linear_mixed_effects_test(drift_arr, correct_arr, subj_arr, n_perm=5000,
                                       rng=np.random.default_rng(0))
print(f"\n  Correct-vs-error drift (pooled N={len(drift_arr)}, LME-perm): "
      f"beta={lme_drift['beta']:.4f}, p={lme_drift['p_value']:.4f} "
      f"(negative beta = error trials drift further from centroid)")

# Adjust for set size and response time, since error trials may be confounded
# with harder (larger set-size) or slower trials rather than "error" per se.
set_size_arr = np.array(pooled_set_size, dtype=float)
rt_arr = np.array(pooled_rt, dtype=float)
lme_drift_adjusted = linear_mixed_effects_test(
    drift_arr, correct_arr, subj_arr, n_perm=5000, rng=np.random.default_rng(0),
    covariates=np.column_stack([set_size_arr, rt_arr]),
)
print(f"  Correct-vs-error drift, set-size+RT-adjusted (pooled N={len(drift_arr)}): "
      f"beta={lme_drift_adjusted['beta']:.4f}, p={lme_drift_adjusted['p_value']:.4f}")

# Q6 DML upgrade (WP-CAUSAL-4ii): a DML estimate of the maintenance-state (drift)
# -> behavior (correct) effect, non-parametrically controlling set size, RT, and
# subject identity, with a mandatory E-value sensitivity analysis. This is a
# causal-FLAVORED association on purely observational data -- drift is measured,
# not manipulated, so there is no positivity argument for a "drift effect" the way
# there is for a genuine stimulation effect; report it as such, not as a causal claim.
_, subj_idx = np.unique(subj_arr, return_inverse=True)
X_confounders = np.column_stack([set_size_arr, rt_arr, subj_idx.astype(float)])
dml_res = dml_partial_linear(correct_arr, drift_arr, X_confounders, n_folds=5,
                             rng=np.random.default_rng(0))
dml_ev = e_value(dml_res["theta"], dml_res["se"], y_sd=float(correct_arr.std()),
                 d_sd=float(drift_arr.std()))
print(f"  Q6 DML (drift -> correct, controlling set-size/RT/subject, N={dml_res['n']}): "
      f"theta={dml_res['theta']:.4f} [{dml_res['ci_lo']:.4f}, {dml_res['ci_hi']:.4f}] "
      f"p={dml_res['p_value']:.4f}; E-value={dml_ev['e_value']:.2f} "
      f"(CI-edge {dml_ev['e_value_ci']:.2f})")

# Update all_statistics.json
with open(RESULTS / "all_statistics.json") as f:
    stats = json.load(f)
stats["boran_ctg"] = {
    sub: {
        "tau":                 v["tau"],
        "tau_interpretable":   v["tau_interpretable"],
        "mean_offdiag_auc":    v["mean_offdiag_auc"],
        "mean_diag_auc":       v["mean_diag_auc"],
        "offdiag_effect":      v["offdiag_effect"],
        "p_offdiag_vs_chance": v["p_offdiag_vs_chance"],
        "q_value_fdr":         v["q_value_fdr"],
        "comparison_epoch_encoding": v["comparison_epoch_encoding"],
        "negative_control_baseline": v["negative_control_baseline"],
        "pr_per_set":          v["pr_per_set"],
        "pr_correct_error":    v["pr_correct_error"],
        "code_stability_by_outcome": v["code_stability_by_outcome"],
        "rayleigh":            v["rayleigh"],
    }
    for sub, v in summary.items()
}
stats["boran_correct_error_drift"] = {
    "beta": lme_drift["beta"], "p_value": lme_drift["p_value"],
    "r_squared": lme_drift["r_squared"], "n_trials": int(len(drift_arr)),
    "set_size_rt_adjusted": {
        "beta": lme_drift_adjusted["beta"], "p_value": lme_drift_adjusted["p_value"],
        "r_squared": lme_drift_adjusted["r_squared"],
    },
    "dml_causal_flavored": {
        "theta": dml_res["theta"], "ci_lo": dml_res["ci_lo"], "ci_hi": dml_res["ci_hi"],
        "p_value": dml_res["p_value"], "e_value": dml_ev["e_value"],
        "e_value_ci": dml_ev["e_value_ci"], "n": dml_res["n"],
        "confounders": ["set_size", "response_time", "subject_id"],
        "label": "causal-flavored association on observational data (drift is "
                 "measured, not manipulated -- no positivity for a genuine effect)",
    },
}

# Paired test, across subjects with both outcomes powered enough to run CTG
# separately: is the load/context code less stable on error trials?
cso = {sub: v["code_stability_by_outcome"] for sub, v in summary.items()
       if v["code_stability_by_outcome"] is not None}
if len(cso) >= 4:
    oc = np.array([v["offdiag_effect_correct"] for v in cso.values()])
    oe = np.array([v["offdiag_effect_error"] for v in cso.values()])
    # H1: code is LESS stable (lower offdiag effect) on error trials
    res = paired_sign_flip_test(oe, oc, n_perm=10000, alternative="less", rng=np.random.default_rng(3))
    stats["boran_code_stability_by_outcome"] = {
        "per_subject": cso, "n_subjects": len(cso),
        "mean_gap_error_minus_correct": res["mean_diff"],
        "ci": [res["ci_lower"], res["ci_upper"]], "p_value": res["p_value"],
    }
    print(f"\n  Code stability by outcome (N={len(cso)} subjects): "
          f"mean(error-correct offdiag effect)={res['mean_diff']:.4f} "
          f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], p={res['p_value']:.4f}")
else:
    stats["boran_code_stability_by_outcome"] = {"per_subject": cso, "n_subjects": len(cso),
                                                "note": "fewer than 4 subjects with both outcomes powered"}

with open(RESULTS / "all_statistics.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n" + "="*55)
print("BORAN PIPELINE COMPLETE")
print("="*55)
for sub, v in summary.items():
    print(f"  {sub}: N={v['n_trials']}, τ={v['tau']:.4f}, "
          f"offdiag_effect={v['offdiag_effect']:.4f}, "
          f"p={v['p_offdiag_vs_chance']:.3f}, q={v['q_value_fdr']:.3f}")
