#!/usr/bin/env python3
"""Multiband replication of the core WM-geometry findings — Miller ECoG + Boran iEEG.

A reviewer who sees that theta decodes at least as well as HGP will ask why
HGP was used at all. The answer is not "HGP wins" (it doesn't) — it is that
HGP is the only band directly comparable to the Rutishauser single-unit
spike dataset (Ray & Maunsell 2011), so it is retained as the substrate for
the rest of the paper's pipeline for a methodological, not a performance,
reason. To make that defensible, every core geometric/dynamical claim is
checked here in ALL FIVE bands, not just CTG τ:

  1. CTG τ (activity-maintained code, τ < 1)          — compute_ctg
  2. PR null (dimensionality flat across load)         — pr_by_band + LME permutation
  3. Ring-attractor phase concentration (Rayleigh R)    — rayleigh_test
  4. Flow divergence ∇·v (DMD, contracting vs expanding) — dmd_reconstruction_error

If all four replicate qualitatively across bands (as they do), the paper's
claims are shown to be about the population code, not an artifact of
choosing HGP.

Five frequency bands:
  theta  4–8  Hz    beta   13–30 Hz   hgp   70–150 Hz
  alpha  8–13 Hz    gamma  30–70 Hz

Pipeline per band (Miller):
  1. Load raw voltage (Miller MAT) → CAR + notch.
  2. Bandpass → Hilbert envelope² → Gaussian smooth → epoch.
  3. Baseline z-score per channel.
  4. PCA (top-8 PCs) → CTG (0-back vs 2-back, stride=40); PR per load; Rayleigh
     at 2-back maintenance end; DMD divergence on mean 2-back trajectory.

Pipeline per band (Boran, iEEG LFP — continuous field, so the same
band-power decomposition applies):
  Same CTG steps, using the good-channel selection already fixed by the main
  HGP-based Boran pipeline (boran_geometry_*.npz), set4 vs set8 CTG.

Rutishauser (single-unit spike times) has no continuous voltage field, so
band-power decomposition does not apply — this is a stated scope limit,
not an oversight. TES1/LQR personalisation is likewise not re-derived per
band: it is an engineering controllability demonstration conditioned on
having *an* adequate manifold, not a claim about which band carries the
code, so it is demonstrated once (HGP) and that choice is stated explicitly.

Phase-amplitude coupling (PAC):
  Theta phase × HGP amplitude, KL-divergence modulation index (Tort et al. 2010).
  Computed per channel × trial; projected to population level via mean over channels.

Saves: results/multiband_ctg.npz

Run from project root:
    conda run -n wm_dynamics python scripts/run_multiband_analysis.py
"""
from __future__ import annotations
import sys, warnings
import numpy as np
import scipy.signal as sig
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (
    DATA_DIR, load_subject, preprocess,
    bandpass_filter, find_stimulus_onsets, baseline_normalize,
    line_noise_notch, bipolar_reference_by_shank,
)
from statistics import linear_mixed_effects_test, rayleigh_test
from dynamics import dmd_reconstruction_error
from geometry import ctg_label_permutation_null

RESULTS = ROOT / "results"
SUBJECTS = ["al", "ca", "cc", "ug"]

SRATE = 1000  # Miller 2019 Nat Hum Behav Methods: "sampled at 1000 Hz" (not 1200)
PRE_MS  = 200.0
POST_MS = 1500.0

BANDS = {
    "theta": (4,   8,   200),   # (lo_hz, hi_hz, smooth_ms)
    "alpha": (8,   13,  200),
    "beta":  (13,  30,  150),
    "gamma": (30,  70,  100),
    "hgp":   (70,  150,  50),
}

# Maintenance window
MAINT_T0 = 0.30
MAINT_T1 = 1.40
N_PC     = 8
CTG_STEP = 40   # stride for CTG matrix (matches main analysis)
PAC_N_BINS = 18
# Nested-CV, label-permutation null (geometry.ctg_label_permutation_null) — the
# same estimator as the headline CTG (run_miller_ctg_corrected.py), so multiband
# results are directly comparable rather than inflated by a single non-nested
# train/test split (audit item 1c-i). n_perm is lower than the headline's 10000
# purely for compute budget (5 bands x 13 subjects here vs. 1 band x 4 there).
N_PERM_CTG = 1000

# ── Boran iEEG (mirrors scripts/run_boran_pipeline.py epoching) ────────────────
import h5py

BORAN_DIR      = Path("/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/000574")
BORAN_SUBJECTS = [f"sub-0{i}" for i in range(1, 10)]
BORAN_SRATE    = 1398.0
T_PRE_MAINT    = 3.0
T_POST_MAINT   = 3.0
T_EPOCH_PRE    = 1.0
EPOCH_TOTAL    = T_PRE_MAINT + T_POST_MAINT + T_EPOCH_PRE
BORAN_CTG_STEP = 280


# ── Signal helpers ─────────────────────────────────────────────────────────────

def band_power_epoch(
    data_clean: np.ndarray,
    stim: np.ndarray,
    task: np.ndarray,
    target: np.ndarray,
    lo: float, hi: float, smooth_ms: float,
    srate: float = SRATE,
    pre_ms: float = PRE_MS,
    post_ms: float = POST_MS,
) -> dict:
    """Extract bandpass power epochs for a single frequency band."""
    bp = bandpass_filter(data_clean, lo, hi, srate)
    analytic = sig.hilbert(bp, axis=0)
    power = np.abs(analytic) ** 2

    smooth_s = int(smooth_ms * srate / 1000)
    if smooth_s > 1:
        kernel = sig.windows.gaussian(smooth_s * 6 + 1, std=smooth_s)
        kernel /= kernel.sum()
        power = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), 0, power
        )

    pre  = int(pre_ms * srate / 1000)
    post = int(post_ms * srate / 1000)
    onsets = find_stimulus_onsets(stim)
    valid = (onsets >= pre) & (onsets + post <= len(power))
    onsets = onsets[valid]

    epochs = np.stack([power[o - pre : o + post] for o in onsets]).astype(np.float32)
    times  = np.linspace(-pre_ms / 1000, post_ms / 1000, pre + post)
    return {
        "epochs":  epochs,
        "times":   times,
        "task_id": task[onsets],
        "tgt_id":  target[onsets],
    }


def compute_ctg(X_ct: np.ndarray, task_id: np.ndarray, step: int = CTG_STEP,
                n_perm: int = N_PERM_CTG, rng: np.random.Generator | None = None
                ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """CTG matrix (0-back vs 2-back) via the shared nested-CV, label-permutation
    pipeline (geometry.ctg_label_permutation_null) — the same estimator as the
    headline CTG, so multiband results are comparable to it (audit item 1c-i).

    Parameters
    ----------
    X_ct : (N, C, T) — raw per-trial channel/unit responses over the
           maintenance window (PCA is fit per-fold inside ctg_label_permutation_null,
           not on X_ct as a whole, so no leakage into held-out trials).

    Returns: auc_mat, t_idx, tau, mean_offdiag, p_value
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mask = (task_id == 0) | (task_id == 2)
    X_sub = X_ct[mask]
    y = (task_id[mask] == 2).astype(int)

    T = X_sub.shape[2]
    t_idx = np.arange(0, T, step)
    res = ctg_label_permutation_null(
        X_sub, y, t_idx, n_components=N_PC, n_splits=5, n_perm=n_perm, rng=rng
    )
    return res["auc_mat"], t_idx, res["tau"], res["mean_offdiag_auc_minus_chance"], res["p_value"]


def compute_pr(Z_flat: np.ndarray) -> float:
    """Participation ratio from a (n_obs, d) latent matrix."""
    cov = np.cov(Z_flat.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def band_dmd_divergence(Z_mean: np.ndarray, dt: float, r: int = 8) -> dict:
    """Flow divergence ∇·v = trace(A−I)/dt via DMD on a mean maintenance trajectory.

    r must equal the full latent dimensionality (matches run_divergence_analysis.py's
    DMD_RANK=8): truncating below full rank on an 8-D system produces a rank-deficient
    reconstruction of A whose trace is dominated by numerical noise, inflating
    trace(A-I)/dt by ~300x even though R² still looks acceptable (e.g. r=6 gives
    div=-2402 s⁻¹ vs the correct r=8 value of -7.6 s⁻¹ for the same trajectory).
    """
    T, d = Z_mean.shape
    r_use = min(r, d, T - 2)
    res = dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)
    A = res["A"]
    eigs = np.linalg.eigvals(A)
    return {
        "div_scalar": float(np.trace(A - np.eye(d))) / dt,
        "max_re_eig": float(eigs.real.max()),
        "r2":         float(res["r_squared"]),
    }


def compute_pac_mi(phase_signal: np.ndarray, amp_signal: np.ndarray,
                   n_bins: int = PAC_N_BINS) -> float:
    """KL-divergence modulation index (Tort et al. 2010).

    Parameters
    ----------
    phase_signal : (T,) — instantaneous phase of the phase-providing band
    amp_signal   : (T,) — envelope of the amplitude-providing band

    Returns
    -------
    MI : float — modulation index (0=uniform, >0=coupled)
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase_signal, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    amp_sum = np.zeros(n_bins)
    for k in range(n_bins):
        amp_sum[k] = amp_signal[bin_idx == k].mean() if (bin_idx == k).any() else 0.0

    p = amp_sum / (amp_sum.sum() + 1e-12)
    p = np.maximum(p, 1e-12)
    kl_div = np.sum(p * np.log(p * n_bins))        # KL from uniform
    mi = kl_div / np.log(n_bins)
    return float(mi)


# ── Boran iEEG helpers (mirrors run_boran_pipeline.load_subject_sessions) ──────

def boran_load_subject_sessions(subj: str) -> dict | None:
    """Load and concatenate all sessions for one Boran subject (epochs, set_sizes)."""
    nwbs = sorted((BORAN_DIR / subj).glob("*.nwb"))
    all_epochs, all_sizes = [], []
    electrode_labels = None

    for nwb_path in nwbs:
        with h5py.File(str(nwb_path), "r") as f:
            raw   = f["acquisition/ecephys.ieeg/data"][:]
            times = f["acquisition/ecephys.ieeg/timestamps"][:]
            t_start_arr = f["intervals/trials/start_time"][:]
            set_sizes   = f["intervals/trials/set_size"][:].astype(int)
            artifact    = f["intervals/trials/artifact"][:].astype(bool)

            if electrode_labels is None:
                try:
                    ieeg_idx = f["acquisition/ecephys.ieeg/electrodes"][:]
                    labels_full = [l.decode() for l in
                                    f["general/extracellular_ephys/electrodes/label"][:]]
                    electrode_labels = [labels_full[i] for i in ieeg_idx]
                except KeyError:
                    electrode_labels = [f"ch{i}" for i in range(raw.shape[1])]

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
            all_epochs.append(raw[i0:i1].T.astype(np.float32))  # (C, T)
            all_sizes.append(set_sizes[trial_idx])

    if not all_epochs:
        return None

    epochs = np.stack(all_epochs, axis=0)  # (N, C, T)
    n_epoch_samp = int(EPOCH_TOTAL * BORAN_SRATE)
    if epochs.shape[2] != n_epoch_samp:
        from scipy.signal import resample
        epochs = resample(epochs, n_epoch_samp, axis=2)

    return {"epochs": epochs, "set_sizes": np.array(all_sizes, dtype=int),
            "electrode_labels": electrode_labels}


def boran_band_power_epochs(epochs: np.ndarray, lo: float, hi: float,
                             smooth_ms: float, srate: float = BORAN_SRATE) -> np.ndarray:
    """Generic band-power extraction per trial for Boran epochs (N, C, T)."""
    N, C, T = epochs.shape
    out = np.zeros_like(epochs, dtype=np.float32)
    smooth_s = int(smooth_ms / 1000 * srate)
    kernel = sig.windows.gaussian(smooth_s * 6 + 1, std=smooth_s)
    kernel /= kernel.sum()

    for n in range(N):
        filtered = bandpass_filter(epochs[n].T, lo, hi, srate)  # (T, C)
        analytic = sig.hilbert(filtered, axis=0)
        power = np.abs(analytic) ** 2
        for c in range(C):
            power[:, c] = np.convolve(power[:, c], kernel, mode="same")
        out[n] = power.T.astype(np.float32)
    return out


def boran_baseline_normalize(band_power: np.ndarray, baseline_samps: int) -> np.ndarray:
    bl = band_power[:, :, :baseline_samps]
    mu = bl.mean(axis=(0, 2), keepdims=True)
    sd = bl.std(axis=(0, 2), keepdims=True) + 1e-10
    return (band_power - mu) / sd


def process_boran_multiband(out: dict):
    """Multiband CTG (set4 vs set8) for Boran iEEG, reusing HGP-pipeline good channels."""
    print("\n  Boran iEEG (N=9):")
    maint_start_s = int((T_EPOCH_PRE + T_PRE_MAINT) * BORAN_SRATE)
    maint_end_s   = int((T_EPOCH_PRE + T_PRE_MAINT + T_POST_MAINT) * BORAN_SRATE)
    bl_samps      = int(T_EPOCH_PRE * BORAN_SRATE)

    for subj in BORAN_SUBJECTS:
        geo_path = RESULTS / f"boran_geometry_{subj}.npz"
        if not geo_path.exists():
            print(f"    SKIP {subj} — run run_boran_pipeline.py first")
            continue

        print(f"  {subj}:", flush=True)
        data = boran_load_subject_sessions(subj)
        if data is None:
            print(f"    SKIP {subj} — no valid epochs")
            continue

        # Notch (50/100/150 Hz) + bipolar-by-shank reref — Round-8 7A/7B fix,
        # matches run_boran_pipeline.py exactly so good_ch_indices (computed
        # there on the same bipolar channel ordering) stay valid here.
        epochs = data["epochs"]
        N_e, C_raw, T_e = epochs.shape
        for n in range(N_e):
            epochs[n] = line_noise_notch(epochs[n].T, BORAN_SRATE, fundamental=50.0,
                                         n_harmonics=3).T
        X_flat = epochs.transpose(0, 2, 1).reshape(-1, C_raw)
        X_bp, _ = bipolar_reference_by_shank(X_flat, data["electrode_labels"])
        epochs = X_bp.reshape(N_e, T_e, -1).transpose(0, 2, 1).astype(np.float32)
        set_sizes = data["set_sizes"]

        geo = np.load(geo_path, allow_pickle=True)
        good_ch = geo["good_ch_indices"]
        epochs = epochs[:, good_ch, :]

        mask_ctg = (set_sizes == 4) | (set_sizes == 8)
        task_id_ctg = np.where(set_sizes[mask_ctg] == 8, 2, 0)  # reuse compute_ctg's 0/2 convention

        rng_ctg = np.random.default_rng(0)
        for band_name, (lo, hi, smooth_ms) in BANDS.items():
            print(f"    {band_name} ({lo}–{hi} Hz)...", end=" ", flush=True)
            bp = boran_band_power_epochs(epochs, lo, hi, smooth_ms, BORAN_SRATE)
            bp = boran_baseline_normalize(bp, bl_samps)
            bp_maint = bp[:, :, maint_start_s:maint_end_s]      # (N, C, T_maint)
            times_maint = np.linspace(0.0, T_POST_MAINT, bp_maint.shape[2])

            X_ctg = bp_maint[mask_ctg]                          # (N_ctg, C, T_maint) — raw
            # channel space; ctg_label_permutation_null fits PCA per-fold internally.
            auc_mat, t_idx, tau, mean_od, p_val = compute_ctg(
                X_ctg, task_id_ctg, step=BORAN_CTG_STEP, rng=rng_ctg
            )
            print(f"τ={tau:.3f}  AUC_od={mean_od:.3f}  p={p_val:.4f}")

            out[f"boran_{subj}_{band_name}_tau"]          = np.array(tau)
            out[f"boran_{subj}_{band_name}_mean_offdiag"] = np.array(mean_od)
            out[f"boran_{subj}_{band_name}_p_value"]      = np.array(p_val)
            out[f"boran_{subj}_{band_name}_auc_mat"]      = auc_mat
            out[f"boran_{subj}_{band_name}_t_times"]      = times_maint[t_idx]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out = {}

    for subj in SUBJECTS:
        print(f"\n  {subj}:", flush=True)
        raw = load_subject(subj)
        data_clean = preprocess(raw["data"], srate=SRATE)
        stim   = raw["stim"]
        task   = raw["task"]
        target = raw["target"]

        # Good channels (same as main pipeline)
        ep0 = np.load(RESULTS / f"01_epochs_{subj}.npz", allow_pickle=True)
        good_ch = ep0["good_channels"]

        # Theta phase for PAC (computed from clean data directly)
        theta_bp = bandpass_filter(data_clean[:, good_ch], 4, 8, SRATE)
        theta_phase_cont = np.angle(sig.hilbert(theta_bp, axis=0))  # (T, n_ch)

        band_results = {}
        pac_mi_per_trial = []   # PAC theta×HGP per trial

        for band_name, (lo, hi, smooth_ms) in BANDS.items():
            print(f"    {band_name} ({lo}–{hi} Hz)...", end=" ", flush=True)
            ep = band_power_epoch(
                data_clean[:, good_ch], stim, task, target,
                lo, hi, smooth_ms
            )
            epochs  = ep["epochs"]      # (N, T, n_ch)
            times   = ep["times"]
            task_id = ep["task_id"]

            # Baseline z-score
            epochs_z = baseline_normalize(epochs, times)

            # Maintenance window
            maint = (times >= MAINT_T0) & (times <= MAINT_T1)
            X_maint = epochs_z[:, maint, :]
            N, T_m, n_ch = X_maint.shape

            # CTG: shared nested-CV pipeline on raw channel-space data (PCA fit
            # per-fold internally) — comparable to the headline CTG estimator.
            rng_ctg = np.random.default_rng(0)
            auc_mat, t_idx, tau, mean_od, p_val = compute_ctg(
                X_maint.transpose(0, 2, 1), task_id, CTG_STEP, rng=rng_ctg
            )

            # PR/Rayleigh/DMD still use a single global PCA projection (Z):
            # those are descriptive geometry statistics, not label-supervised
            # decoders, so they carry no train/test leakage risk from a
            # whole-dataset PCA fit.
            X_flat = X_maint.reshape(-1, n_ch)
            X_flat -= X_flat.mean(0)
            _, _, Vt = np.linalg.svd(X_flat, full_matrices=False)
            V = Vt[:N_PC].T
            Z = np.einsum("ntc,cd->ntd", epochs_z[:, maint, :], V)  # (N, T_m, N_PC)

            # PR per load — dimensionality null, checked in every band
            pr_per_load = {}
            for load in [0, 1, 2]:
                m_load = task_id == load
                pr_per_load[load] = compute_pr(Z[m_load].reshape(-1, N_PC)) if m_load.sum() >= 5 else np.nan

            # Ring-attractor phase concentration at 2-back maintenance end (PC1-PC2)
            mask_2back = task_id == 2
            Z_end = Z[mask_2back][:, -1, :2]
            phases = np.arctan2(Z_end[:, 1], Z_end[:, 0]) % (2 * np.pi)
            ray = rayleigh_test(phases)

            # DMD flow divergence on the mean 2-back maintenance trajectory
            Z_mean_2back = Z[mask_2back].mean(0)         # (T_m, N_PC)
            dt_band = float(np.median(np.diff(times[maint])))
            div = band_dmd_divergence(Z_mean_2back, dt_band)

            print(f"τ={tau:.3f}  AUC_od={mean_od:.3f}  p={p_val:.4f}  "
                  f"PR(0,1,2)=({pr_per_load[0]:.2f},{pr_per_load[1]:.2f},{pr_per_load[2]:.2f})  "
                  f"R={ray['R']:.3f}  div={div['div_scalar']:.2f}s⁻¹")

            band_results[band_name] = {
                "tau":           tau,
                "mean_offdiag":  mean_od,
                "p_value":       p_val,
                "auc_mat":       auc_mat,
                "t_times":       times[maint][t_idx],
                "pr_per_load":   pr_per_load,
                "rayleigh_R":    ray["R"],
                "rayleigh_p":    ray["p_value"],
                "div_scalar":    div["div_scalar"],
                "max_re_eig":    div["max_re_eig"],
                "dmd_r2":        div["r2"],
            }

            # PAC: theta phase × HGP amplitude (only for HGP band)
            if band_name == "hgp":
                # Epoch theta phase at the same onsets
                onsets_ep = find_stimulus_onsets(stim)
                pre  = int(PRE_MS * SRATE / 1000)
                post = int(POST_MS * SRATE / 1000)
                valid = (onsets_ep >= pre) & (onsets_ep + post <= len(theta_phase_cont))
                onsets_ep = onsets_ep[valid]

                theta_epochs = np.stack([
                    theta_phase_cont[o - pre : o + post]
                    for o in onsets_ep
                ])  # (N, T, n_ch)

                # Align with HGP epochs (they use the same onset logic so shapes match)
                n_trials = min(len(theta_epochs), len(epochs_z))
                theta_epochs = theta_epochs[:n_trials]
                hgp_epochs   = epochs_z[:n_trials]
                task_id_pac  = task_id[:n_trials]

                # Compute MI per trial × channel, then average over channels
                mi_trials = []
                for n in range(n_trials):
                    mi_ch = []
                    for ch in range(n_ch):
                        ph = theta_epochs[n, :, ch]
                        am = hgp_epochs[n, :, ch]
                        mi_ch.append(compute_pac_mi(ph, am))
                    mi_trials.append(float(np.mean(mi_ch)))
                pac_mi_per_trial = mi_trials
                pac_task_id = task_id_pac.tolist()

        # Summarise PAC by condition
        pac_arr = np.array(pac_mi_per_trial)
        pac_tid = np.array(pac_task_id)
        pac_0back = float(pac_arr[pac_tid == 0].mean()) if (pac_tid == 0).any() else np.nan
        pac_2back = float(pac_arr[pac_tid == 2].mean()) if (pac_tid == 2).any() else np.nan
        print(f"    PAC (theta×HGP): 0-back={pac_0back:.4f}  2-back={pac_2back:.4f}  "
              f"Δ={pac_2back - pac_0back:.4f}")

        # Store
        for band_name, br in band_results.items():
            out[f"{subj}_{band_name}_tau"]          = np.array(br["tau"])
            out[f"{subj}_{band_name}_mean_offdiag"] = np.array(br["mean_offdiag"])
            out[f"{subj}_{band_name}_p_value"]      = np.array(br["p_value"])
            out[f"{subj}_{band_name}_auc_mat"]      = br["auc_mat"]
            out[f"{subj}_{band_name}_t_times"]      = br["t_times"]
            out[f"{subj}_{band_name}_pr_load0"]     = np.array(br["pr_per_load"][0])
            out[f"{subj}_{band_name}_pr_load1"]     = np.array(br["pr_per_load"][1])
            out[f"{subj}_{band_name}_pr_load2"]     = np.array(br["pr_per_load"][2])
            out[f"{subj}_{band_name}_rayleigh_R"]   = np.array(br["rayleigh_R"])
            out[f"{subj}_{band_name}_rayleigh_p"]   = np.array(br["rayleigh_p"])
            out[f"{subj}_{band_name}_div_scalar"]   = np.array(br["div_scalar"])
            out[f"{subj}_{band_name}_max_re_eig"]   = np.array(br["max_re_eig"])
            out[f"{subj}_{band_name}_dmd_r2"]       = np.array(br["dmd_r2"])

        out[f"{subj}_pac_mi_per_trial"] = np.array(pac_mi_per_trial)
        out[f"{subj}_pac_task_id"]      = np.array(pac_task_id)
        out[f"{subj}_pac_0back"]        = np.array(pac_0back)
        out[f"{subj}_pac_2back"]        = np.array(pac_2back)

    process_boran_multiband(out)

    # ── Cross-subject band-replication summary (Miller) ─────────────────────────
    print("\n  Miller — band replication summary (mean ± SD across 4 subjects):")
    print(f"  {'band':6s} {'tau':>14s} {'PR(0,1,2)':>22s} {'PR-null p':>10s} "
          f"{'Rayleigh R':>11s} {'div (s⁻¹)':>12s}")
    rng_lme = np.random.default_rng(0)
    for band_name in BANDS:
        taus = np.array([float(out[f"{s}_{band_name}_tau"]) for s in SUBJECTS])
        pr0 = np.array([float(out[f"{s}_{band_name}_pr_load0"]) for s in SUBJECTS])
        pr1 = np.array([float(out[f"{s}_{band_name}_pr_load1"]) for s in SUBJECTS])
        pr2 = np.array([float(out[f"{s}_{band_name}_pr_load2"]) for s in SUBJECTS])
        R   = np.array([float(out[f"{s}_{band_name}_rayleigh_R"]) for s in SUBJECTS])
        div = np.array([float(out[f"{s}_{band_name}_div_scalar"]) for s in SUBJECTS])

        # Subject-level permutation test: PR ~ load, subject as random intercept
        metric    = np.concatenate([pr0, pr1, pr2])
        condition = np.concatenate([np.zeros(4), np.ones(4), 2 * np.ones(4)])
        subject   = np.tile(np.arange(4), 3)
        lme = linear_mixed_effects_test(metric, condition, subject, n_perm=2000, rng=rng_lme)

        out[f"band_summary_{band_name}_lme_beta"]    = np.array(lme["beta"])
        out[f"band_summary_{band_name}_lme_p"]       = np.array(lme["p_value"])
        out[f"band_summary_{band_name}_tau_mean"]    = np.array(taus.mean())
        out[f"band_summary_{band_name}_rayleigh_mean"] = np.array(R.mean())
        out[f"band_summary_{band_name}_div_mean"]    = np.array(div.mean())

        print(f"  {band_name:6s} {taus.mean():.3f}±{taus.std():.3f}   "
              f"({pr0.mean():.2f},{pr1.mean():.2f},{pr2.mean():.2f})  "
              f"p={lme['p_value']:.3f}       "
              f"{R.mean():.3f}±{R.std():.3f}   "
              f"{div.mean():7.2f}±{div.std():.2f}")

    np.savez(RESULTS / "multiband_ctg.npz", **out)
    print("\n  Saved: results/multiband_ctg.npz")

    boran_done = [s for s in BORAN_SUBJECTS if f"{s}_hgp_tau" in out]
    if boran_done:
        print("\n  Boran τ per band (mean ± SD across "
              f"{len(boran_done)} subjects):")
        for band_name in BANDS:
            taus = [float(out[f"{s}_{band_name}_tau"]) for s in boran_done]
            print(f"    {band_name:5s}: {np.mean(taus):.3f} ± {np.std(taus):.3f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("Multiband CTG + PAC analysis (Miller ECoG + Boran iEEG)...")
    main()
    print("Done.")
