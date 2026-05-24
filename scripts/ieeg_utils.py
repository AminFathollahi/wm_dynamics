"""
ieeg_utils.py — Shared utilities for the WM Dynamics project.

This module contains the preprocessing and analysis functions used across
all notebooks. The goal is clean, readable code where every function
does exactly one thing and its docstring explains the WHY, not just the WHAT.

Understanding these functions is part of your education. Before using any
function, read the docstring and the implementation. If you can't explain
what it does mathematically, you don't own it yet.
"""

import numpy as np
import scipy.signal as sig
import scipy.io as sio
from pathlib import Path


# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("/home/amin/Research/Representation/Working Memory/dynamical systems/kai miller/memory_nback/memory_nback/data")
SUBJECTS  = ['al', 'ca', 'cc', 'ug']
SRATE     = 1200   # Hz — verified from block durations (100s × 1200 = 120040 samples)

# Condition codes (from `task` vector)
TASK_CODES = {-1: 'rest', 0: 'zero_back', 1: 'one_back', 2: 'two_back'}

# Target codes (from `target` vector)
TARGET_CODES = {0: 'no_stim', 1: 'non_target', 2: 'target'}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_subject(subj: str) -> dict:
    """
    Load one subject's N-back iEEG data.

    Returns a dict with keys:
      data   : (n_samples, n_channels) float64 — raw ECoG voltage, already
               bandpass-filtered 1-300 Hz by Miller lab (see ns_1k_1_300_filt.mat)
      stim   : (n_samples,) uint8 — letter identity at each sample (0=no stim, 1-40=letter)
      task   : (n_samples,) int8  — condition (-1=rest, 0/1/2=N-back load)
      target : (n_samples,) uint8 — trial type (0=no stim, 1=non-target, 2=target)
      srate  : int — sampling rate in Hz
      subject: str

    WHY this structure: The Miller dataset encodes trial information as
    sample-by-sample vectors rather than an event table. This means you can
    directly index into the raw data using boolean masks — no epoch lookup table needed.
    """
    path = DATA_DIR / f"{subj}_nback.mat"
    raw  = sio.loadmat(str(path))
    return {
        'data'   : raw['data'],               # (T, C)
        'stim'   : raw['stim'].ravel(),       # (T,)
        'task'   : raw['task'].ravel().astype(np.int8),
        'target' : raw['target'].ravel(),     # (T,)
        'srate'  : SRATE,
        'subject': subj,
    }


# ─── Preprocessing ────────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lo: float, hi: float,
                    srate: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    WHY zero-phase (filtfilt): A causal filter introduces a frequency-dependent
    phase delay — your 80 Hz high-gamma peak would be shifted in time relative
    to your 10 Hz LFP. For ERP/event-locked analysis this corrupts latency
    estimates. filtfilt applies the filter forward then backward, canceling
    the phase shift while doubling the effective filter order.

    WHY Butterworth: Maximally flat passband. No ripple in the frequencies
    you want. Chebyshev would give steeper rolloff but introduces ripple.

    Args:
        data  : (T, C) or (T,) array — raw signal
        lo    : lower cutoff frequency in Hz
        hi    : upper cutoff frequency in Hz
        srate : sampling rate in Hz
        order : filter order (4 is standard; higher = steeper rolloff, more ringing)

    Returns: filtered array, same shape as input
    """
    nyq  = srate / 2.0
    sos  = sig.butter(order, [lo / nyq, hi / nyq], btype='band', output='sos')
    if data.ndim == 1:
        return sig.sosfiltfilt(sos, data)
    return np.apply_along_axis(lambda x: sig.sosfiltfilt(sos, x), 0, data)


def notch_filter(data: np.ndarray, freq: float, srate: float,
                 q: float = 30.0) -> np.ndarray:
    """
    Notch filter to remove line noise and harmonics.

    WHY: Power line noise at 60 Hz (US) and harmonics (120, 180 Hz) corrupts
    the high-gamma band (70-150 Hz). The notch removes a narrow band around
    each harmonic. Q factor controls bandwidth: Q=30 → bandwidth ≈ 2 Hz.

    Args:
        freq : line noise frequency (60 Hz for US, 50 Hz for Europe)
    """
    nyq = srate / 2.0
    b, a = sig.iirnotch(freq / nyq, q)
    if data.ndim == 1:
        return sig.filtfilt(b, a, data)
    return np.apply_along_axis(lambda x: sig.filtfilt(b, a, x), 0, data)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Re-reference each sample to the mean across all electrodes.

    WHY: ECoG recordings pick up a shared voltage reference that is not
    neural activity. By subtracting the mean across all channels at each
    time point, you remove common-mode signals (volume conduction, movement)
    and improve spatial specificity.

    WHY NOT a specific reference electrode: A bad reference electrode would
    corrupt every channel. CAR distributes any reference noise equally.

    Args:
        data : (T, C) — raw voltage
    Returns: (T, C) — re-referenced voltage
    """
    return data - data.mean(axis=1, keepdims=True)


def reject_bad_channels(data: np.ndarray, threshold_sd: float = 3.0) -> np.ndarray:
    """
    Return a boolean mask of good channels.

    WHY: Broken electrodes, reference channels, or channels over scar tissue
    have abnormally high or low variance. Including them corrupts your PCA
    and high-gamma estimates. The heuristic: reject channels whose variance
    deviates >threshold_sd SDs from the median channel variance.

    Args:
        data : (T, C)
        threshold_sd : rejection threshold in standard deviations

    Returns: (C,) boolean mask — True = keep
    """
    ch_var   = data.var(axis=0)
    med      = np.median(ch_var)
    mad      = np.median(np.abs(ch_var - med))  # median absolute deviation — robust SD
    robust_z = np.abs(ch_var - med) / (1.4826 * mad + 1e-12)
    return robust_z < threshold_sd


def preprocess(data: np.ndarray, srate: float = SRATE,
               notch_freq: float = 60.0) -> np.ndarray:
    """
    Full preprocessing pipeline: CAR → notch → no additional bandpass
    (data is pre-filtered 1-300 Hz by Miller lab).

    Returns: (T, C) preprocessed signal
    """
    data = common_average_reference(data)
    # Notch 60 Hz + harmonics up to 300 Hz
    for harmonic in [60, 120, 180, 240]:
        if harmonic < srate / 2:
            data = notch_filter(data, harmonic, srate)
    return data


# ─── High-Gamma Power ─────────────────────────────────────────────────────────

def high_gamma_power(data: np.ndarray, srate: float = SRATE,
                     lo: float = 70.0, hi: float = 150.0,
                     smooth_ms: float = 50.0) -> np.ndarray:
    """
    Extract high-gamma band power (70-150 Hz) via Hilbert transform.

    WHY high-gamma for cognitive states: High-gamma (broadband, 70-150 Hz)
    reflects local cortical processing — the aggregate firing of neurons
    near the electrode. It is tightly coupled to MUA (multi-unit activity)
    and tracks cognitive load, attention, and WM maintenance with millisecond
    precision. It is NOT an oscillation — it is broadband noise that increases
    when neurons in the vicinity are active.

    Pipeline:
      1. Bandpass to [lo, hi] Hz
      2. Hilbert transform → analytic signal (real + imaginary)
      3. Compute envelope: |z(t)| = amplitude of the analytic signal
      4. Square for power: |z(t)|²
      5. Smooth with Gaussian kernel to reduce trial noise

    WHY Hilbert (not short-time FFT): The Hilbert transform gives you
    instantaneous amplitude at every sample point. STFT gives you amplitude
    averaged over a window. For trial-by-trial analysis you need the
    instantaneous estimate.

    Args:
        smooth_ms : Gaussian smoothing kernel width in milliseconds

    Returns: (T, C) high-gamma power
    """
    # Step 1: bandpass
    hg = bandpass_filter(data, lo, hi, srate)
    # Step 2-4: analytic signal → power
    analytic = sig.hilbert(hg, axis=0)
    power    = np.abs(analytic) ** 2
    # Step 5: Gaussian smoothing
    smooth_samples = int(smooth_ms * srate / 1000)
    if smooth_samples > 1:
        kernel = sig.windows.gaussian(smooth_samples * 6, std=smooth_samples)
        kernel /= kernel.sum()
        power = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 0, power
        )
    return power


# ─── Epoching ─────────────────────────────────────────────────────────────────

def find_stimulus_onsets(stim: np.ndarray) -> np.ndarray:
    """
    Find sample indices where a new stimulus appears (stim goes from 0 → nonzero).

    Returns: (n_trials,) integer array of onset sample indices
    """
    # Rising edge: stim was 0 at t-1 and nonzero at t
    onset_mask = (stim[:-1] == 0) & (stim[1:] != 0)
    return np.where(onset_mask)[0] + 1


def epoch_data(power: np.ndarray, stim: np.ndarray, task: np.ndarray,
               target: np.ndarray, pre_ms: float = 200.0, post_ms: float = 1500.0,
               srate: float = SRATE) -> dict:
    """
    Cut continuous power signal into stimulus-locked epochs.

    WHY epoch: The continuous recording contains rest periods, transitions,
    and all conditions interleaved. Epoching aligns all trials to the stimulus
    onset, giving you a [trials × time × channels] tensor where the first
    axis indexes over events and the second over time relative to the event.

    Args:
        power   : (T, C) high-gamma power
        pre_ms  : ms before stimulus onset to include (baseline period)
        post_ms : ms after stimulus onset to include (response period)

    Returns dict with:
      epochs  : (n_trials, n_times, n_channels) float32
      times   : (n_times,) in seconds relative to onset (negative = pre-stim)
      task_id : (n_trials,) int — condition (0/1/2)
      tgt_id  : (n_trials,) int — target type (1=non-target, 2=target)
      stim_id : (n_trials,) int — letter identity
    """
    pre  = int(pre_ms  * srate / 1000)
    post = int(post_ms * srate / 1000)
    n_t  = pre + post

    onsets = find_stimulus_onsets(stim)

    # Filter to onsets where we have enough data
    valid = (onsets >= pre) & (onsets + post <= len(power))
    onsets = onsets[valid]

    epochs  = np.stack([power[o - pre : o + post] for o in onsets], axis=0).astype(np.float32)
    times   = np.linspace(-pre_ms / 1000, post_ms / 1000, n_t)

    return {
        'epochs' : epochs,           # (n_trials, n_times, n_channels)
        'times'  : times,
        'task_id': task[onsets],
        'tgt_id' : target[onsets],
        'stim_id': stim[onsets],
        'onsets' : onsets,
    }


# ─── Baseline Normalization ────────────────────────────────────────────────────

def baseline_normalize(epochs: np.ndarray, times: np.ndarray,
                        baseline_window: tuple = (-0.2, 0.0)) -> np.ndarray:
    """
    Z-score each channel relative to its pre-stimulus baseline.

    WHY: Absolute power varies enormously across electrodes and subjects.
    Normalizing to baseline (the 200 ms before stimulus onset) puts all
    channels on a common scale — units of "standard deviations above baseline."

    Args:
        epochs   : (n_trials, n_times, n_channels)
        baseline_window : (start_s, end_s) relative to onset
    Returns: normalized epochs, same shape
    """
    t0, t1      = baseline_window
    bl_mask     = (times >= t0) & (times <= t1)
    bl          = epochs[:, bl_mask, :]           # (n_trials, n_bl, C)
    bl_mean     = bl.mean(axis=(0, 1), keepdims=True)   # across trials and time
    bl_std      = bl.std(axis=(0, 1), keepdims=True) + 1e-10
    return (epochs - bl_mean) / bl_std


# ─── Geometry ─────────────────────────────────────────────────────────────────

def pca_project(X: np.ndarray, n_components: int = 10) -> tuple:
    """
    Project data onto its top principal components via SVD.

    WHY SVD not sklearn PCA: Identical result, but SVD makes the math explicit.
    X = U Σ Vᵀ where:
      - V columns = principal directions (axes of maximum variance)
      - Σ diagonal = variance explained by each direction
      - U Σ = projection of X onto principal directions

    WHY dimensionality reduction before geometry analysis:
    Principal angles and tangling are computed in the latent space.
    With 40-64 electrodes, the data lives in a very high-dimensional space
    but the neural manifold is intrinsically low-dimensional (typically 3-15D).
    Reducing first removes noise directions and makes geometry computation stable.

    Args:
        X            : (n_samples, n_features) — e.g. (n_trials × n_times, n_channels)
        n_components : number of PCs to retain

    Returns: (scores, components, variance_explained)
      scores    : (n_samples, n_components) — projected data
      components: (n_features, n_components) — principal directions
      var_exp   : (n_components,) — fraction of variance per PC
    """
    X_c = X - X.mean(axis=0)   # center
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    scores    = U[:, :n_components] * s[:n_components]
    var_exp   = (s ** 2) / (s ** 2).sum()
    return scores, Vt[:n_components].T, var_exp[:n_components]


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces spanned by columns of A and B.

    MATH: If A = QₐRₐ and B = QᵦRᵦ are QR decompositions (orthonormal bases),
    then the cosines of the principal angles are the singular values of QₐᵀQᵦ.
    θᵢ = arccos(σᵢ),  where σᵢ ∈ [0, 1]

    WHY this matters for WM: During maintenance, neural population activity
    for each WM item lives in a subspace of the full state space. If these
    subspaces are nearly parallel (low θ), items interfere with each other
    — the brain cannot distinguish them. This is the geometric signature of
    representational interference and predicts WM failure.

    Low θ_min → subspaces overlap → representations interfere → failure

    Args:
        A : (n, k) — k basis vectors for subspace A
        B : (n, m) — m basis vectors for subspace B

    Returns: (min(k, m),) principal angles in radians ∈ [0, π/2]
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    # Singular values of QaᵀQb are cosines of principal angles
    _, sv, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    sv = np.clip(sv, -1, 1)   # numerical safety
    return np.arccos(sv)


def trajectory_tangling(Z: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """
    Compute trajectory tangling metric Q(t) from Russo et al. (2018).

    MATH:
        Q(t) = max_{t'} [ ‖ż(t) - ż(t')‖² / (‖z(t) - z(t')‖² + ε) ]

    WHERE:
        ż(t) ≈ (z(t+1) - z(t-1)) / 2Δt   (central finite difference)
        z(t)  = latent state at time t

    INTERPRETATION:
    Q(t) is high when there exist two time points with SIMILAR states but
    VERY DIFFERENT velocities. This is the signature of dynamical instability:
    from the same position, the system can go to very different places.
    A noise-robust system (Russo et al.: motor cortex) has low Q everywhere.

    YOUR HYPOTHESIS: PFC working memory maintenance has low Q during successful
    maintenance, and Q spikes precede failure (target misses, high-load breakdown).

    Args:
        Z       : (T, d) — latent trajectory, T time points, d dimensions
        epsilon : regularizer preventing division by zero when states coincide

    Returns: (T,) tangling values (0 at endpoints due to finite difference)
    """
    T = len(Z)
    # Finite difference velocity (central, zero-padded at endpoints)
    Zdot = np.zeros_like(Z)
    Zdot[1:-1] = (Z[2:] - Z[:-2]) / 2.0

    Q = np.zeros(T)
    for t in range(1, T - 1):
        dstate = Z - Z[t]                          # (T, d) — state differences
        dvel   = Zdot - Zdot[t]                    # (T, d) — velocity differences
        num    = (dvel ** 2).sum(axis=1)           # ‖ṡ‖²
        den    = (dstate ** 2).sum(axis=1) + epsilon
        Q[t]   = (num / den).max()
    return Q


# ─── Convenience ──────────────────────────────────────────────────────────────

def condition_mask(task: np.ndarray, condition: int) -> np.ndarray:
    """Boolean mask for samples in a given task condition."""
    return task == condition


def trial_mask(task_id: np.ndarray, tgt_id: np.ndarray,
               task_cond: int = None, target_type: int = None) -> np.ndarray:
    """Boolean mask over epochs for a given condition / target type."""
    mask = np.ones(len(task_id), dtype=bool)
    if task_cond is not None:
        mask &= (task_id == task_cond)
    if target_type is not None:
        mask &= (tgt_id == target_type)
    return mask
