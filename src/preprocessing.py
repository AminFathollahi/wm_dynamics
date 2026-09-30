"""
preprocessing.py — multi-dataset iEEG signal processing pipeline.

Datasets supported:
  1. ECoG n-back corpus N-back ECoG (Nat Hum Behav library, "memory_nback")
       4 subjects (al, ca, cc, ug); ECoG; 1000 Hz (Synamps2, scalp/mastoid ref,
       0.15-200 Hz instrument bandpass -- the dataset's own Methods, "Recordings");
       0/1/2-back house-repetition detection (NOT verbal -- stimuli are house
       pictures; see the dataset's own README_memory_nback_dataset_notes.docx).
       MAT format. Functions: load_subject, load_miller_nback, compute_hgp

  2. DANDI 000574 — Sternberg WM
       9 subjects; iEEG + EEG + LFP + single units (MTL: hippocampus, amygdala,
       temporal cortex); NWB format; Sternberg set sizes 4/6/8
       Source: doi 10.1038/s41597-020-0364-3, Sci Data 7:30; recorded
       in Zurich
       Functions: load_boran_nwb, compute_boran_hgp

  3. TES1 — doi 10.7554/eLife.18834 transcranial electrical stimulation
       17 subjects; intracranial voltage induced by 1 mA tES; MNI coordinates
       Provides the input matrix B for LQR control (tES → brain field mapping)
       Functions: load_tes1_stimulation, build_tes1_input_matrix


Signal processing methods:
  high-gamma extraction (standard broadband conventions)
  common average reference rationale (published re-referencing analysis)
  AR(1) deconvolution framework (published calcium-deconvolution method)
  broadband HGP as MUA proxy (published spike-band relationship)
"""

from __future__ import annotations

import re
import zipfile
from collections import defaultdict
import numpy as np
import scipy.signal as sig
import scipy.io as sio
from pathlib import Path
from scipy.signal import welch
import itertools
from info_decoding import MIN_TRIALS_PER_CLASS, MIN_TRIALS_PER_CLASS_COMPARISON
from statistics import _choose_n_splits, stable_seed
from geometry import _fit_pca_fold, _project_fold, _ctg_score_fold, _ctg_splits
from statistics import _standardize_train_test
import csv
from scipy.signal import detrend, resample_poly, windows
from drift_dynamics import fit_gaussian_state_space, leave_one_out_condition_residuals
from statistics import permutation_pvalue
from scipy import linalg
import mne
from scipy.signal import butter, hilbert, sosfiltfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
try:  # package import in tests; flat import in directly-run legacy scripts
    from .project_config import data_asset_path, data_root, dataset_path
except ImportError:  # pragma: no cover - exercised by script entry points
    from project_config import data_asset_path, data_root, dataset_path

_EXT_DATA = data_root(required=False)
DATA_DIR = dataset_path("kai_miller_nback", "data", required=False)
MILLER_DATA_DIR = (
    dataset_path("kai_miller_nback", required=False).parents[1]
    if _EXT_DATA is not None else None
)
BORAN_DATA_DIR = dataset_path("dandi_000574", required=False)
TES1_ZIP_PATH = data_asset_path("tes1_zip", required=False)

SUBJECTS = ["al", "ca", "cc", "ug"]
SRATE = 1000  # Hz (ECoG n-back corpus) -- confirmed from ECoG n-back corpus 2019 Nat Hum Behav Methods
              # ("Electrical potentials were sampled at 1000 Hz"), NOT 1200 Hz.
BORAN_SRATE_IEEG = 1398  # Hz (DANDI 000574 iEEG)
BORAN_SRATE_LFP = 22370  # Hz (DANDI 000574 raw LFP)

TASK_CODES = {-1: "rest", 0: "zero_back", 1: "one_back", 2: "two_back"}
TARGET_CODES = {0: "no_stim", 1: "non_target", 2: "target"}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_subject(subj: str, data_dir: Path = DATA_DIR) -> dict:
    """Load one subject's N-back iEEG data from the ECoG n-back corpus dataset.

    Parameters
    ----------
    subj : one of {'al', 'ca', 'cc', 'ug'}

    Returns
    -------
    dict with keys:
      data    : (T, C) float64 — ECoG voltage, instrument-bandpassed 0.15-200 Hz
      stim    : (T,)  uint8  — house-picture identity (0=ISI, 1-40=house)
      task    : (T,)  int8   — condition (-1=rest, 0/1/2=N-back load)
      target  : (T,)  uint8  — trial type (0=ISI, 1=non-target, 2=target)
      srate   : int          — sampling rate in Hz
      subject : str
    """
    path = data_dir / f"{subj}_nback.mat"
    raw = sio.loadmat(str(path))
    return {
        "data": raw["data"].astype(np.float64),
        "stim": raw["stim"].ravel().astype(np.uint8),
        "task": raw["task"].ravel().astype(np.int8),
        "target": raw["target"].ravel().astype(np.uint8),
        "srate": SRATE,
        "subject": subj,
    }


# ── Signal processing ──────────────────────────────────────────────────────────

class PassbandExceedsNyquistError(ValueError):
    """Raised by bandpass_filter when the requested passband is not representable
    at the signal's own sampling rate: a Butterworth bandpass needs its low and
    high edges strictly between 0 Hz and the Nyquist frequency (srate / 2). Without
    this check, scipy.signal.butter fails later with an unnamed "Digital filter
    critical frequencies must be 0 < Wn < 1" ValueError that reports neither the
    sampling rate nor which requested band caused it. The request is never
    clamped to Nyquist and never narrowed to a band that would fit: either would
    silently substitute a different measurement for the one asked for."""


def bandpass_filter(
    data: np.ndarray, lo: float, hi: float, srate: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (sosfiltfilt for numerical stability).

    Parameters
    ----------
    data  : (T,) or (T, C)
    lo, hi : passband edges in Hz
    order : filter order (effective 2*order after forward-backward pass)

    Raises
    ------
    PassbandExceedsNyquistError
        If the requested band is not entirely representable at this sampling
        rate (0 < lo < hi < nyquist fails). This is a hard error, not a
        warning: a filter fit on a coerced band would produce a number that
        looks like a measurement and is not one.
    """
    nyq = srate / 2.0
    if not (0.0 < lo < hi < nyq):
        raise PassbandExceedsNyquistError(
            f"requested passband {lo:g}-{hi:g} Hz is not representable at a "
            f"{srate:g} Hz sampling rate (Nyquist frequency {nyq:g} Hz): a "
            "Butterworth bandpass filter requires 0 < lo < hi < nyquist. This "
            "passband is not clamped to Nyquist and no narrower band is "
            "substituted for it."
        )
    sos = sig.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    if data.ndim == 1:
        return sig.sosfiltfilt(sos, data)
    return np.apply_along_axis(lambda x: sig.sosfiltfilt(sos, x), 0, data)


def notch_filter(
    data: np.ndarray, freq: float, srate: float, q: float = 30.0
) -> np.ndarray:
    """IIR notch filter at `freq` Hz with quality factor Q (bandwidth ≈ freq/Q)."""
    nyq = srate / 2.0
    b, a = sig.iirnotch(freq / nyq, q)
    if data.ndim == 1:
        return sig.filtfilt(b, a, data)
    return np.apply_along_axis(lambda x: sig.filtfilt(b, a, x), 0, data)


def butterworth_bandstop(
    data: np.ndarray, lo: float, hi: float, srate: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth band-stop filter (sosfiltfilt).

    Distinct from `notch_filter` (a narrow IIR notch at one frequency): this
    is a wide band-stop over [lo, hi], matching source datasets that specify
    their own line-noise removal as a Butterworth band-stop rather than a
    notch (e.g. RAM's 58-62 Hz, 4th-order recipe for the Morlet classifier
    feature bank -- reproduce the released recipe exactly, don't substitute
    a differently-shaped filter).

    Parameters
    ----------
    data   : (T,) or (T, C)
    lo, hi : stopband edges in Hz
    order  : filter order (effective 2*order after forward-backward pass)
    """
    nyq = srate / 2.0
    sos = sig.butter(order, [lo / nyq, hi / nyq], btype="bandstop", output="sos")
    if data.ndim == 1:
        return sig.sosfiltfilt(sos, data)
    return np.apply_along_axis(lambda x: sig.sosfiltfilt(sos, x), 0, data)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the cross-electrode mean at each time point.

    Removes volume-conducted and reference-electrode artefacts while
    preserving spatially local signals (published local-reference analysis).

    Parameters
    ----------
    data : (T, C)

    Returns
    -------
    data_car : (T, C)
    """
    return data - data.mean(axis=1, keepdims=True)


def line_noise_notch(
    data: np.ndarray, srate: float, fundamental: float = 50.0, n_harmonics: int = 3,
) -> np.ndarray:
    """Notch out mains frequency + harmonics (e.g. 50/100/150 Hz for EU sites).

    Required for any high-gamma (70-150 Hz) path at a 50 Hz-mains site (e.g.
    DANDI 000574/Zurich): the 2nd/3rd mains harmonics (100, 150 Hz) otherwise fall
    inside the HGP band uncorrected. ECoG n-back corpus (US, 60 Hz) already goes through
    preprocess(), which does notch -- this covers 50 Hz sites.

    Parameters
    ----------
    fundamental : mains frequency (50 EU, 60 US) -- verify per site, don't hard-code blindly.
    n_harmonics : number of harmonics to notch, including the fundamental.
    """
    for h in range(1, n_harmonics + 1):
        f = fundamental * h
        if f < srate / 2:
            data = notch_filter(data, f, srate)
    return data


def shank_bipolar_index_pairs(
    labels: list[str],
) -> tuple[list[tuple[int, int, str]], list[int]]:
    """Parse electrode labels into adjacent-contact bipolar (idx0, idx1, pair_label)
    triples plus a list of orphan channel indices (no adjacent contact to pair with).

    Electrode labels are parsed as <shank letters><contact number>, e.g.
    'AHL3' -> shank 'AHL', contact 3 (matches DANDI 000574's naming). Shared by
    bipolar_reference_by_shank (for the signal) and any per-channel metadata
    (e.g. MNI coords) that needs the identical channel reduction.
    """
    shank_pat = re.compile(r"^([A-Za-z]+)(\d+)$")
    shanks: dict[str, list[tuple[int, int]]] = defaultdict(list)
    orphans: list[int] = []
    for i, lab in enumerate(labels):
        m = shank_pat.match(lab)
        if m:
            shanks[m.group(1)].append((int(m.group(2)), i))
        else:
            orphans.append(i)

    pairs: list[tuple[int, int, str]] = []
    for shank, contacts in shanks.items():
        contacts.sort()
        if len(contacts) < 2:
            orphans.extend(i for _, i in contacts)
            continue
        for (n0, i0), (n1, i1) in zip(contacts[:-1], contacts[1:]):
            pairs.append((i0, i1, f"{shank}{n0}-{shank}{n1}"))
    return pairs, orphans


def bipolar_reference_by_shank(
    data: np.ndarray, labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Bipolar (sequential adjacent-contact) re-reference for depth/sEEG shanks.

    Bipolar-along-shank is the sEEG field standard (shared-reference /
    volume-conduction artifacts are far better removed by differencing
    adjacent contacts a few mm apart than by a whole-head common-average
    reference, which mixes signal across shanks/regions). Channels that
    don't parse (e.g. scalp EEG
    'F3', 'Cz') or belong to a single-contact shank fall back to a
    common-average reference among themselves.

    Parameters
    ----------
    data   : (T, C)
    labels : length-C electrode labels

    Returns
    -------
    data_bp   : (T, C_bp) -- C_bp < C (each shank loses its last contact)
    labels_bp : list[str], e.g. 'AHL3-AHL4' for bipolar pairs, '<label>_CAR' for orphans
    """
    pairs, orphans = shank_bipolar_index_pairs(labels)
    out_cols = [data[:, i0] - data[:, i1] for i0, i1, _ in pairs]
    out_labels = [lab for _, _, lab in pairs]

    if orphans:
        sub = data[:, orphans]
        car = sub - sub.mean(axis=1, keepdims=True)
        for j, i in enumerate(orphans):
            out_cols.append(car[:, j])
            out_labels.append(f"{labels[i]}_CAR")

    return np.stack(out_cols, axis=1), out_labels


def _mad_outlier_mask(values: np.ndarray, threshold_mad: float) -> np.ndarray:
    """True where `values` is a robust-z outlier above the population median."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    robust_z = np.abs(values - med) / (1.4826 * mad + 1e-12)
    return robust_z >= threshold_mad


def line_noise_power_ratio(
    data: np.ndarray,
    srate: float,
    mains_hz: float = 60.0,
    n_harmonics: int = 3,
    bandwidth_hz: float = 2.0,
) -> np.ndarray:
    """Fraction of each channel's broadband power concentrated at mains harmonics.

    A channel contaminated by narrowband line noise (mains hum, a stimulation
    artifact) does not necessarily have unusual total variance -- the
    contamination can sit almost entirely inside a couple of Hz-wide peaks,
    which a broadband variance/MAD test cannot see. This measures that
    concentration directly from a Welch PSD.

    Parameters
    ----------
    data   : (T, C)
    mains_hz : site line frequency (60 US, 50 EU) -- do not assume one site.

    Returns
    -------
    ratio : (C,) float in [0, 1] -- line-band power / total power
    """
    nperseg = int(min(data.shape[0], max(srate, 256)))
    freqs, psd = sig.welch(data, fs=srate, axis=0, nperseg=nperseg)
    total_power = np.trapezoid(psd, freqs, axis=0)
    line_power = np.zeros(data.shape[1])
    for h in range(1, n_harmonics + 1):
        f0 = mains_hz * h
        if f0 >= srate / 2:
            break
        band = (freqs >= f0 - bandwidth_hz / 2) & (freqs <= f0 + bandwidth_hz / 2)
        if np.any(band):
            line_power += np.trapezoid(psd[band], freqs[band], axis=0)
    return line_power / (total_power + 1e-12)


def reject_bad_channels(
    data: np.ndarray,
    threshold_mad: float = 3.0,
    srate: float | None = None,
    mains_hz: float = 60.0,
    line_noise_threshold_mad: float = 3.0,
) -> np.ndarray:
    """Return boolean mask of channels passing broadband-variance AND line-noise QC.

    Two independent, additive criteria; a channel is rejected if it fails
    EITHER (additive, not substitutive -- both criteria are checked, neither replaces the other):
      1. broadband variance is a MAD-outlier relative to the other channels
         (the original criterion).
      2. its power is abnormally concentrated at the mains frequency and its
         harmonics (`line_noise_power_ratio`), relative to the other
         channels. This is required because criterion 1 alone is a known
         false-negative source in ECoG/iEEG QC: narrowband contamination
         (mains hum, stimulation artifact) can leave broadband variance
         unremarkable while still corrupting a specific band, e.g. the
         70-150 Hz high-gamma band used throughout this project.

    Criterion 2 only runs when `srate` is given (it needs a sampling rate to
    define frequency bins); passing `srate=None` reproduces the original
    variance-only behavior exactly, for callers that have not been migrated.

    Parameters
    ----------
    threshold_mad : rejection z-score in robust MAD units (3.0 = 3 MAD)
    srate         : sampling rate in Hz; enables the line-noise criterion.
    mains_hz      : site line frequency (60 US, 50 EU) -- verify per site,
                    do not assume one site for every dataset.

    Returns
    -------
    good : (C,) bool  — True = keep
    """
    variance_bad = _mad_outlier_mask(data.var(axis=0), threshold_mad)
    if srate is None:
        return ~variance_bad
    line_ratio = line_noise_power_ratio(data, srate, mains_hz=mains_hz)
    line_noise_bad = _mad_outlier_mask(line_ratio, line_noise_threshold_mad)
    return ~(variance_bad | line_noise_bad)


def preprocess(
    data: np.ndarray,
    srate: float = SRATE,
    notch_freq: float = 60.0,
    n_harmonics: int = 4,
) -> np.ndarray:
    """Full preprocessing pipeline: CAR → notch (line noise + harmonics).

    The ECoG n-back corpus dataset has only an instrument-imposed 0.15-200 Hz bandpass;
    CAR and line-noise removal are what this function adds.

    Parameters
    ----------
    notch_freq  : fundamental line frequency (60 Hz US, 50 Hz EU)
    n_harmonics : number of harmonics to notch (including fundamental)

    Returns
    -------
    data_clean : (T, C)
    """
    data = common_average_reference(data)
    for h in range(1, n_harmonics + 1):
        f = notch_freq * h
        if f < srate / 2:
            data = notch_filter(data, f, srate)
    return data


# ── High-gamma power ───────────────────────────────────────────────────────────

def high_gamma_power(
    data: np.ndarray,
    srate: float = SRATE,
    lo: float = 70.0,
    hi: float = 150.0,
    smooth_ms: float = 50.0,
) -> np.ndarray:
    """Extract broadband high-gamma (70-150 Hz) power via Hilbert envelope.

    Pipeline:
      1. Bandpass [lo, hi] Hz (Butterworth, zero-phase)
      2. Hilbert transform → analytic signal
      3. Squared envelope = instantaneous power
      4. Gaussian smoothing (σ = smooth_ms) to reduce trial-to-trial noise

    High-gamma is broadband, tracks local MUA, and is the best non-invasive
    correlate of cognitive state in ECoG (published high-gamma studies).

    Parameters
    ----------
    smooth_ms : Gaussian σ in milliseconds (50 ms = 50 samples at 1000 Hz)

    Returns
    -------
    power : (T, C) — instantaneous high-gamma power, smoothed
    """
    hg = bandpass_filter(data, lo, hi, srate)
    analytic = sig.hilbert(hg, axis=0)
    power = np.abs(analytic) ** 2

    smooth_s = int(smooth_ms * srate / 1000)
    if smooth_s > 1:
        kernel = sig.windows.gaussian(smooth_s * 6 + 1, std=smooth_s)
        kernel /= kernel.sum()
        power = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), 0, power
        )
    return power


# ── Epoching ───────────────────────────────────────────────────────────────────

def find_stimulus_onsets(stim: np.ndarray) -> np.ndarray:
    """Sample indices at which a new stimulus appears (0 → nonzero transition)."""
    rising = (stim[:-1] == 0) & (stim[1:] != 0)
    return np.where(rising)[0] + 1


def epoch_data(
    power: np.ndarray,
    stim: np.ndarray,
    task: np.ndarray,
    target: np.ndarray,
    pre_ms: float = 200.0,
    post_ms: float = 1500.0,
    srate: float = SRATE,
) -> dict:
    """Cut continuous power into stimulus-locked epochs.

    Parameters
    ----------
    power   : (T, C)
    pre_ms  : baseline window before onset (ms)
    post_ms : analysis window after onset (ms)

    Returns
    -------
    dict:
      epochs  : (N, n_times, C) float32
      times   : (n_times,) s, relative to onset
      task_id : (N,) int
      tgt_id  : (N,) int
      stim_id : (N,) int — house-picture identity at onset
      onsets  : (N,) int — sample indices of stimulus onset
    """
    pre = int(pre_ms * srate / 1000)
    post = int(post_ms * srate / 1000)
    onsets = find_stimulus_onsets(stim)

    valid = (onsets >= pre) & (onsets + post <= len(power))
    onsets = onsets[valid]

    epochs = np.stack(
        [power[o - pre : o + post] for o in onsets], axis=0
    ).astype(np.float32)
    times = np.linspace(-pre_ms / 1000.0, post_ms / 1000.0, pre + post)

    return {
        "epochs": epochs,
        "times": times,
        "task_id": task[onsets],
        "tgt_id": target[onsets],
        "stim_id": stim[onsets],
        "onsets": onsets,
    }


def baseline_normalize(
    epochs: np.ndarray,
    times: np.ndarray,
    baseline_window: tuple[float, float] = (-0.2, 0.0),
) -> np.ndarray:
    """Z-score each channel relative to pre-stimulus baseline.

    Baseline is pooled across trials (grand-average baseline), matching
    the published high-gamma baseline convention.

    Parameters
    ----------
    baseline_window : (t_start, t_end) in seconds (must be ≤ 0)

    Returns
    -------
    epochs_z : (N, n_times, C) — units = SD above baseline
    """
    t0, t1 = baseline_window
    bl_mask = (times >= t0) & (times <= t1)
    bl = epochs[:, bl_mask, :]
    mu = bl.mean(axis=(0, 1), keepdims=True)
    sd = bl.std(axis=(0, 1), keepdims=True) + 1e-10
    return (epochs - mu) / sd


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    subj: str,
    pre_ms: float = 200.0,
    post_ms: float = 1500.0,
    bad_ch_threshold: float = 3.0,
) -> dict:
    """End-to-end pipeline for one subject.

    Returns the normalized epoch tensor and metadata ready for geometry analysis.
    """
    d = load_subject(subj)
    good = reject_bad_channels(d["data"], threshold_mad=bad_ch_threshold, srate=SRATE, mains_hz=60.0)
    data_clean = preprocess(d["data"][:, good])
    hgp = high_gamma_power(data_clean)
    ep = epoch_data(hgp, d["stim"], d["task"], d["target"], pre_ms, post_ms)
    ep["epochs"] = baseline_normalize(ep["epochs"], ep["times"])
    ep["good_channels"] = np.where(good)[0]
    ep["subject"] = subj
    return ep


# ── ECoG n-back corpus convenience wrappers (used by notebooks 07/08) ─────────────────────

def load_miller_nback(
    mat_path: str,
    pre_ms: float = 200.0,
    post_ms: float = 1500.0,
    bad_ch_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one ECoG n-back corpus N-back MAT file and return epoch tensor.

    Convenience wrapper over load_subject → preprocess → high_gamma_power →
    epoch_data for use in production notebooks.

    Parameters
    ----------
    mat_path : path to subject MAT file (e.g. '.../al_nback.mat')

    Returns
    -------
    epochs : (N_trials, N_ch, T)  float32  — baseline-z-scored HGP
    times  : (T,)  float64  — seconds relative to stimulus onset
    labels : (N_trials,) int8  — WM load (0/1/2)
    """
    import scipy.io as _sio
    raw = _sio.loadmat(str(mat_path))
    data = raw["data"].astype(np.float64)
    stim = raw["stim"].ravel().astype(np.uint8)
    task = raw["task"].ravel().astype(np.int8)
    target = raw["target"].ravel().astype(np.uint8)

    good = reject_bad_channels(data, threshold_mad=bad_ch_threshold, srate=SRATE, mains_hz=60.0)
    data_clean = preprocess(data[:, good])
    hgp = high_gamma_power(data_clean)  # (T, C)

    ep = epoch_data(hgp, stim, task, target, pre_ms, post_ms)
    epochs_z = baseline_normalize(ep["epochs"], ep["times"])  # (N, T, C)
    epochs_out = epochs_z.transpose(0, 2, 1)  # → (N, C, T)

    return epochs_out, ep["times"], ep["task_id"]


def compute_hgp(
    epochs: np.ndarray,
    srate: float = SRATE,
    lo: float = 70.0,
    hi: float = 150.0,
    smooth_ms: float = 50.0,
) -> np.ndarray:
    """Compute high-gamma power for already-epoched data.

    Parameters
    ----------
    epochs : (N_trials, N_ch, T)

    Returns
    -------
    hgp : (N_trials, N_ch, T)
    """
    N, C, T = epochs.shape
    out = np.zeros_like(epochs, dtype=np.float32)
    for n in range(N):
        out[n] = high_gamma_power(epochs[n].T, srate, lo, hi, smooth_ms).T
    return out


# ── DANDI 000574 — DANDI 000574 Sternberg WM ──────────────────────────────────

def load_boran_nwb(
    nwb_path: str,
    signal: str = "ieeg",
    epoch_win: tuple[float, float] = (-6.5, 2.5),
    reject_channels: bool = False,
    bad_channel_mad_threshold: float = 3.0,
    mains_hz: float = 50.0,
) -> dict:
    """Load DANDI 000574 Sternberg WM task from NWB file (DANDI 000574).

    Task structure (time relative to probe onset at t=0):
      Fixation:    [-6, -5] s
      Encoding:    [-5, -3] s  (2s, 4/6/8 letters)
      Maintenance: [-3,  0] s  (3s delay)
      Probe:       [ 0, +2] s

    Uses h5py to avoid requiring pynwb. Electrode brain areas include MTL
    (hippocampus CA1/CA3, amygdala, entorhinal), superior/middle temporal
    gyrus, and scalp EEG — enabling multi-region geometry analysis not
    possible with the ECoG n-back corpus dataset.

    Parameters
    ----------
    nwb_path : path to .nwb file
    signal   : 'ieeg' (1398 Hz, 48 ch) | 'eeg' (140 Hz, 19 ch) | 'lfp' (raw)
    epoch_win: (t_pre, t_post) in seconds relative to probe onset
    reject_channels : if True, run `reject_bad_channels` (variance-MAD plus
                       line-noise power-ratio) on the continuous recording
                       before epoching, and drop failing channels from
                       `epochs`/`electrode_labels`/`electrode_locs`. Off by
                       default so existing callers keep the full raw channel
                       set; new analyses that need clean channels should
                       opt in explicitly.
    mains_hz : site line frequency for the line-noise criterion. This
               release is recorded in Zurich (50 Hz mains), not a US site —
               do not reuse a 60 Hz default here.

    Returns
    -------
    dict with keys:
      epochs       : (N_trials, N_ch, T)  float32
      times        : (T,) float64  — s relative to probe onset
      set_sizes    : (N_trials,) int
      correct      : (N_trials,) bool
      artifact     : (N_trials,) bool
      electrode_labels : list[str]
      electrode_locs   : list[str]  — brain area per electrode
      good_channels    : (N_ch,) bool — QC mask already applied above
                          (all True when reject_channels=False)
      srate        : float
    """
    import h5py

    with h5py.File(str(nwb_path), "r") as f:
        sig_key = f"ecephys.{signal}"
        raw_data = f["acquisition"][sig_key]["data"][:]       # (T_total, C)
        timestamps = f["acquisition"][sig_key]["timestamps"][:]  # (T_total,)
        srate = 1.0 / np.diff(timestamps).mean()

        # Electrode metadata
        elec = f["general/extracellular_ephys/electrodes"]
        elec_labels = [x.decode() for x in elec["label"][:]]
        elec_locs = [x.decode() for x in elec["location"][:]]
        elec_group = [x.decode() for x in elec["group_name"][:]]

        # Select electrodes for this signal type
        grp_name = "ieeg" if signal in ("ieeg", "lfp") else "eeg"
        ch_mask = np.array([g == grp_name for g in elec_group])
        if signal == "lfp":
            ch_mask = np.ones(len(elec_group), dtype=bool)

        # Trials
        trials = f["intervals/trials"]
        trial_starts = trials["start_time"][:]    # probe - 6s (fixation start)
        set_sizes = trials["set_size"][:]
        correct = trials["correct"][:].astype(bool)
        artifact = trials["artifact"][:].astype(bool)

        # Probe time = trial_start + 6.0 s (fixation [-6,-5] + encoding [-5,-3] + maint [-3,0])
        probe_times = trial_starts + 6.0

        t_pre, t_post = epoch_win
        pre_samp = int(abs(t_pre) * srate)
        post_samp = int(t_post * srate)
        n_samp = pre_samp + post_samp

        times = np.linspace(t_pre, t_post, n_samp)

        epochs_list = []
        valid_mask = []
        for pt in probe_times:
            center_idx = np.searchsorted(timestamps, pt)
            i0 = center_idx - pre_samp
            i1 = center_idx + post_samp
            if i0 < 0 or i1 > raw_data.shape[0]:
                valid_mask.append(False)
                epochs_list.append(np.zeros((raw_data.shape[1], n_samp), dtype=np.float32))
            else:
                epoch = raw_data[i0:i1, :].T.astype(np.float32)  # (C, T)
                epochs_list.append(epoch)
                valid_mask.append(True)

    epochs_arr = np.stack(epochs_list, axis=0)  # (N, C, T)
    valid_mask = np.array(valid_mask)

    # Filter electrode metadata to only the channels present in epochs
    elec_labels_sel = [elec_labels[i] for i in range(len(elec_labels)) if ch_mask[i]]
    elec_locs_sel   = [elec_locs[i]   for i in range(len(elec_locs))   if ch_mask[i]]

    if reject_channels:
        good_channels = reject_bad_channels(
            raw_data, threshold_mad=bad_channel_mad_threshold, srate=srate, mains_hz=mains_hz,
        )
    else:
        good_channels = np.ones(raw_data.shape[1], dtype=bool)
    epochs_arr = epochs_arr[:, good_channels, :]
    elec_labels_sel = [lab for lab, keep in zip(elec_labels_sel, good_channels) if keep]
    elec_locs_sel = [loc for loc, keep in zip(elec_locs_sel, good_channels) if keep]

    return {
        "epochs": epochs_arr,
        "times": times,
        "set_sizes": set_sizes,
        "correct": correct,
        "artifact": artifact,
        "valid": valid_mask,
        "electrode_labels": elec_labels_sel,
        "electrode_locs": elec_locs_sel,
        "good_channels": good_channels,
        "srate": srate,
        "signal": signal,
    }


def compute_boran_hgp(
    epochs: np.ndarray,
    srate: float = BORAN_SRATE_IEEG,
    lo: float = 70.0,
    hi: float = 150.0,
    smooth_ms: float = 50.0,
) -> np.ndarray:
    """High-gamma power for DANDI 000574 Sternberg epochs.

    Parameters
    ----------
    epochs : (N, C, T) — raw iEEG epochs

    Returns
    -------
    hgp : (N, C, T) float32
    """
    return compute_hgp(epochs, srate=srate, lo=lo, hi=hi, smooth_ms=smooth_ms)


def boran_baseline_normalize(
    epochs: np.ndarray,
    times: np.ndarray,
    baseline_window: tuple[float, float] = (-6.5, -6.0),
) -> np.ndarray:
    """Z-score DANDI 000574 epochs relative to pre-fixation baseline.

    Uses far pre-fixation [-6.5, -6.0]s to avoid contamination by
    anticipatory activity (which begins at fixation onset ~-6s).

    Parameters
    ----------
    epochs : (N, C, T)
    times  : (T,) relative to probe onset

    Returns
    -------
    epochs_z : (N, C, T)
    """
    t0, t1 = baseline_window
    bl_mask = (times >= t0) & (times <= t1)
    bl = epochs[:, :, bl_mask]                            # (N, C, T_bl)
    mu = bl.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]  # (1, C, 1)
    sd = bl.std(axis=(0, 2))[np.newaxis, :, np.newaxis] + 1e-10
    return (epochs - mu) / sd


# ── TES1 — published eLife tES stimulation field ──────────────────────

def load_tes1_stimulation(
    zip_path: str | Path | None = None,
    subject: str | None = None,
) -> dict | list[dict]:
    """Load tES-induced intracranial voltages from the published eLife TES1 release.

    TES1 (CRCNS) provides the voltage at each intracranial electrode induced
    by 1 mA transcranial stimulation. This gives the input matrix B for our
    LQR controller: B[i] = voltage induced at electrode i per mA of current.

    Format per file: electrode_name, MNI_x, MNI_y, MNI_z, voltage_mV
    (voltage calibrated to 1 mA; NaN = high impedance or unlocalized electrode)

    Parameters
    ----------
    zip_path : path to HuangLiu2016dataset.zip; defaults to the configured asset
    subject  : subject ID string (e.g. 'P03'). If None, returns list of all.

    Returns
    -------
    dict (or list of dicts) with keys:
      subject      : str
      names        : (N_elec,) list[str]
      mni_coords   : (N_elec, 3) float — MNI x,y,z in mm (NaN if unlocalized)
      voltage_mV   : (N_elec,) float — induced voltage per mA (NaN if rejected)
    """
    if zip_path is None:
        zip_path = data_asset_path("tes1_zip")

    def _parse_file(z: zipfile.ZipFile, fname: str, subj_id: str) -> dict:
        with z.open(fname) as fh:
            lines = fh.read().decode("utf-8", errors="replace").strip().split("\n")
        names, coords, volts = [], [], []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            names.append(parts[0])
            if len(parts) >= 4:
                try:
                    x, y, z_ = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    x, y, z_ = np.nan, np.nan, np.nan
                coords.append([x, y, z_])
                volt_idx = 4
            else:
                coords.append([np.nan, np.nan, np.nan])
                volt_idx = 1
            try:
                volts.append(float(parts[volt_idx]))
            except (ValueError, IndexError):
                volts.append(np.nan)
        return {
            "subject": subj_id,
            "names": names,
            "mni_coords": np.array(coords, dtype=float),
            "voltage_mV": np.array(volts, dtype=float),
        }

    with zipfile.ZipFile(str(zip_path), "r") as z:
        txt_files = [
            n for n in z.namelist()
            if n.endswith(".txt") and "README" not in n
        ]

        if subject is not None:
            target = f"HuangLiu2016dataset/{subject}.txt"
            return _parse_file(z, target, subject)

        results = []
        for fname in txt_files:
            subj_id = fname.split("/")[-1].replace(".txt", "")
            results.append(_parse_file(z, fname, subj_id))
        return results


def build_tes1_input_matrix(
    tes1_data: dict,
    target_mni: np.ndarray,
    n_sources: int = 1,
    sigma_mm: float = 20.0,
) -> np.ndarray:
    """Build the LQR input matrix B from TES1 stimulation field data.

    Maps tES surface stimulation (1 mA per source) to intracranial voltage at
    target electrode locations. Uses Gaussian spatial interpolation to map
    available TES1 MNI coordinates to arbitrary target electrode locations.

    This gives the B matrix in: x_{t+1} = Ax_t + Bu_t
    where u_t is the stimulation current vector (amps) and B maps current to
    the neural state change (via induced voltage field).

    The TES1 release (eLife 6:e18834) reports no spatial-smoothness constant
    for the field and, if anything, argue AGAINST transporting one subject's
    field onto another's anatomy without a subject-specific model (their
    cross-subject prediction is significantly worse than subject-specific
    FEM, p=1e-7). sigma_mm is therefore a coarse, literature-unconstrained
    smoothing choice, not a calibrated physical scale -- treat B as a rough
    proxy, not a validated field estimate. Matches the value used to
    generate the manuscript's actual TES1 results (scripts/run_tes1_analysis.py).

    Parameters
    ----------
    tes1_data     : output of load_tes1_stimulation for one subject
    target_mni    : (N_target, 3) MNI coordinates of target electrodes
    n_sources     : number of stimulation montages to include as columns of B
    sigma_mm      : Gaussian width for MNI-space interpolation (mm)

    Returns
    -------
    B : (N_target, n_sources) float — voltage (mV) per mA at target locations
    """
    src_mni = tes1_data["mni_coords"]     # (N_src, 3)
    src_volt = tes1_data["voltage_mV"]    # (N_src,)

    # Remove NaN electrodes
    valid = ~(np.isnan(src_mni).any(axis=1) | np.isnan(src_volt))
    src_mni = src_mni[valid]
    src_volt = src_volt[valid]

    if n_sources != 1:
        raise NotImplementedError(
            "TES1 provides a single stimulation field; B supports n_sources=1 only. "
            "Distinct source columns would need per-montage field maps, which this "
            "dataset does not contain (the prior code silently left extra columns zero)."
        )

    N_target = target_mni.shape[0]
    B = np.zeros((N_target, n_sources))

    for i, tgt in enumerate(target_mni):
        dists = np.linalg.norm(src_mni - tgt, axis=1)
        weights = np.exp(-dists**2 / (2 * sigma_mm**2))
        weights /= weights.sum() + 1e-10
        B[i, 0] = np.dot(weights, src_volt)

    return B


# ── Multi-band feature extraction ──────────────────────────────────────────────

BAND_DEFINITIONS = {
    "theta":  (4.0,  8.0),
    "alpha":  (8.0,  13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 70.0),
    "hgp":    (70.0, 150.0),
}


def band_power(
    data: np.ndarray,
    band: str | tuple[float, float],
    srate: float = SRATE,
    smooth_ms: float = 200.0,
) -> np.ndarray:
    """Instantaneous band power via Hilbert envelope, Gaussian-smoothed.

    For low-frequency bands (theta, alpha, beta), a longer smoothing window
    (200 ms default) is appropriate because instantaneous power estimates are
    noisy at low frequencies.  HGP uses the existing high_gamma_power function.

    Parameters
    ----------
    data      : (T, C) — continuous iEEG signal
    band      : name key from BAND_DEFINITIONS or (lo, hi) tuple in Hz
    srate     : sampling rate (Hz)
    smooth_ms : Gaussian σ (ms)

    Returns
    -------
    power : (T, C) — instantaneous band power, smoothed
    """
    if isinstance(band, str):
        lo, hi = BAND_DEFINITIONS[band]
    else:
        lo, hi = band

    filtered = bandpass_filter(data, lo, hi, srate)
    analytic = sig.hilbert(filtered, axis=0)
    power = np.abs(analytic) ** 2

    smooth_s = int(smooth_ms * srate / 1000)
    if smooth_s > 1:
        kernel = sig.windows.gaussian(smooth_s * 6 + 1, std=smooth_s)
        kernel /= kernel.sum()
        power = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), 0, power
        )
    return power


def multiband_features(
    data: np.ndarray,
    srate: float = SRATE,
    bands: list[str] | None = None,
    smooth_ms_map: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Extract band-specific power features for all WM-relevant frequency bands.

    Theta-HGP decomposition enables:
    - Band-specific CTG (which band carries a time-stable WM code?)
    - PAC (phase-amplitude coupling between theta and HGP)
    - Multi-band manifold (richer attractor geometry)

    Parameters
    ----------
    bands        : list of band names; defaults to all BAND_DEFINITIONS keys
    smooth_ms_map: per-band smoothing (ms); defaults: theta/alpha/beta 200ms, gamma/hgp 50ms

    Returns
    -------
    dict mapping band name → (T, C) power array
    """
    if bands is None:
        bands = list(BAND_DEFINITIONS.keys())
    default_smooth = {"theta": 200.0, "alpha": 200.0, "beta": 200.0,
                      "gamma": 100.0, "hgp": 50.0}
    if smooth_ms_map is not None:
        default_smooth.update(smooth_ms_map)

    return {band: band_power(data, band, srate, default_smooth.get(band, 100.0))
            for band in bands}


def phase_amplitude_coupling(
    data: np.ndarray,
    phase_band: str | tuple[float, float] = "theta",
    amplitude_band: str | tuple[float, float] = "hgp",
    srate: float = SRATE,
    n_phase_bins: int = 18,
) -> np.ndarray:
    """Modulation index (MI) PAC: theta phase × HGP amplitude coupling.

    Uses the KL-divergence modulation index: for each channel, compute
    the distribution of HGP amplitude across theta phase bins.  MI measures
    how non-uniform this distribution is (KL divergence from uniform).

    High MI → theta phase organises HGP bursts → multiplexed WM maintenance.

    Parameters
    ----------
    data            : (T, C) — continuous iEEG
    phase_band      : frequency band for the phase signal
    amplitude_band  : frequency band for the amplitude signal
    n_phase_bins    : number of phase bins (default 18 = 20° each)

    Returns
    -------
    mi : (C,) — modulation index per channel
    """
    lo_ph, hi_ph = (BAND_DEFINITIONS[phase_band] if isinstance(phase_band, str)
                    else phase_band)
    lo_amp, hi_amp = (BAND_DEFINITIONS[amplitude_band] if isinstance(amplitude_band, str)
                      else amplitude_band)

    phase_signal = np.angle(sig.hilbert(bandpass_filter(data, lo_ph, hi_ph, srate), axis=0))
    amp_signal   = np.abs(sig.hilbert(bandpass_filter(data, lo_amp, hi_amp, srate), axis=0))

    bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    n_channels = data.shape[1]
    mi = np.zeros(n_channels)

    for c in range(n_channels):
        amp_per_bin = np.zeros(n_phase_bins)
        for b in range(n_phase_bins):
            in_bin = (phase_signal[:, c] >= bins[b]) & (phase_signal[:, c] < bins[b + 1])
            if in_bin.sum() > 0:
                amp_per_bin[b] = amp_signal[in_bin, c].mean()
        total = amp_per_bin.sum()
        if total > 0:
            p = amp_per_bin / total
            uniform = np.ones(n_phase_bins) / n_phase_bins
            mi[c] = float(np.sum(p * np.log(p / uniform + 1e-10)) / np.log(n_phase_bins))

    return mi


def time_resolved_pac(
    data: np.ndarray,
    phase_band: str | tuple[float, float] = "theta",
    amplitude_band: str | tuple[float, float] = "hgp",
    srate: float = SRATE,
    window_ms: float = 400.0,
    step_ms: float = 50.0,
    n_phase_bins: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-resolved PAC modulation index in sliding windows (trPAC).

    KL-divergence MI computed in overlapping windows.  During WM
    maintenance, theta-HGP trPAC should be elevated above baseline and above
    chance — directly testing the theta-gamma working-memory model.

    Parameters
    ----------
    data        : (T, C) — continuous iEEG (FULL recording, not epoched)
    window_ms   : analysis window width in ms
    step_ms     : step between windows in ms

    Returns
    -------
    mi_t   : (n_windows, C) — PAC modulation index per window per channel
    t_cent : (n_windows,)   — centre time of each window in samples
    bins   : (n_phase_bins,) — phase bin centres (radians)
    """
    lo_ph, hi_ph = (BAND_DEFINITIONS[phase_band] if isinstance(phase_band, str)
                    else phase_band)
    lo_amp, hi_amp = (BAND_DEFINITIONS[amplitude_band] if isinstance(amplitude_band, str)
                      else amplitude_band)

    phase_sig = np.angle(sig.hilbert(bandpass_filter(data, lo_ph, hi_ph, srate), axis=0))
    amp_sig   = np.abs(sig.hilbert(bandpass_filter(data, lo_amp, hi_amp, srate), axis=0))

    win_s   = int(window_ms * srate / 1000)
    step_s  = int(step_ms * srate / 1000)
    T, C    = data.shape
    starts  = np.arange(0, T - win_s, step_s)
    bins    = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    bin_ctrs = (bins[:-1] + bins[1:]) / 2
    mi_t    = np.zeros((len(starts), C))

    for wi, t0 in enumerate(starts):
        ph_win  = phase_sig[t0:t0 + win_s]
        amp_win = amp_sig[t0:t0 + win_s]
        for c in range(C):
            amp_per_bin = np.zeros(n_phase_bins)
            for b in range(n_phase_bins):
                mask_b = (ph_win[:, c] >= bins[b]) & (ph_win[:, c] < bins[b + 1])
                if mask_b.sum() > 0:
                    amp_per_bin[b] = amp_win[mask_b, c].mean()
            total = amp_per_bin.sum()
            if total > 0:
                p = amp_per_bin / total
                uniform = np.ones(n_phase_bins) / n_phase_bins
                mi_t[wi, c] = float(np.sum(p * np.log(p / uniform + 1e-10)) /
                                    np.log(n_phase_bins))

    t_cent = starts + win_s // 2
    return mi_t, t_cent, bin_ctrs


PATIENTS = {
    "P1": {"band": "theta", "band_range": (4.0, 8.0), "stim_freq_hz": 4.0},
    "P2": {"band": "alpha", "band_range": (8.0, 12.0), "stim_freq_hz": 10.0},
    "P3": {"band": "alpha", "band_range": (8.0, 12.0), "stim_freq_hz": 10.0},
}


CONDITIONS = ("In Phase", "Anti Phase", "Sham")


RETENTION_ONSET_BUFFER_S = 0.25


STIM_SITES = {
    "P1": [["LFA1", "LFA2"], ["LPA1", "LPA2"]],
    "P2": [["LAF5", "LAF6"], ["LAP5", "LAP6"]],
    "P3": [["RAF6", "RAF7"], ["RSP5", "RSP6"]],
}


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


BAD_CHANNEL_MAD_THRESHOLD = 3.0


BORAN_MAINS_HZ = 50.0


MAINT_WIN = 3.0


MIN_BIPOLAR_CHANNELS = 4


MIN_TRIALS = 20


def lfp_maintenance_tensor(
    ieeg: dict, trial_mask: np.ndarray, n_bins: int, lo: float = 70.0, hi: float = 150.0,
) -> np.ndarray:
    """Notch -> bipolar-reference -> band-envelope power -> maintenance-window bin.

    Matches the convention already established in scripts/run_boran_pipeline.py
    (filter/envelope computed on the full pre-probe-to-post-probe epoch, THEN
    the maintenance sub-window is sliced out, to keep zero-phase filter edge
    effects away from the analysis window boundary). ``lo``/``hi`` default to
    this project's standard high-gamma band; passing a different band (e.g.
    8-45 Hz) reuses the identical filter-and-envelope path with only the
    passband changed, which is the intended way to compare bands at fixed
    sensor and referencing.
    """
    epochs = ieeg["epochs"][trial_mask]
    srate = ieeg["srate"]
    labels = ieeg["electrode_labels"]
    n_trials, n_channels, n_samples = epochs.shape
    notched = np.empty_like(epochs)
    for trial in range(n_trials):
        notched[trial] = line_noise_notch(
            epochs[trial].T, srate, fundamental=BORAN_MAINS_HZ, n_harmonics=3,
        ).T
    flat = notched.transpose(0, 2, 1).reshape(-1, n_channels)
    bipolar_flat, _ = bipolar_reference_by_shank(flat, labels)
    bipolar = bipolar_flat.reshape(n_trials, n_samples, -1).transpose(0, 2, 1).astype(np.float32)
    # smooth_ms=0: an estimator that reports a persistence or confinement
    # contrast must never see a smoothed signal -- a temporal kernel
    # manufactures autocorrelation indistinguishable from genuine confinement.
    # compute_boran_hgp's default (50 ms Gaussian, used by the CTG/geometry
    # pipelines) is correct for those purposes but wrong here; this arm must
    # match the spike arm's build_psth(..., smooth_ms=0, ...) convention
    # exactly.
    band_power = compute_boran_hgp(bipolar, srate=srate, lo=lo, hi=hi, smooth_ms=0.0)
    maint_mask = (ieeg["times"] >= -MAINT_WIN) & (ieeg["times"] < 0)
    band_maint = band_power[:, :, maint_mask]
    edges = np.linspace(0, band_maint.shape[2], n_bins + 1).astype(int)
    return np.stack(
        [band_maint[:, :, edges[k]:edges[k + 1]].mean(axis=2) for k in range(n_bins)], axis=2,
    )


FIELD_BANDS = ("theta", "alpha", "beta", "gamma", "hgp")


FIELD_NYQUIST_MARGIN = 0.8


def multiband_maintenance_tensor(
    epochs: np.ndarray, times: np.ndarray, labels: list[str], srate: float,
    window_s: tuple[float, float], n_bins: int, referencing: str = "bipolar",
    bands: tuple[str, ...] = FIELD_BANDS, mains_hz: float = BORAN_MAINS_HZ,
) -> dict:
    """Notch -> reference once -> per-band Hilbert-envelope power (band_power) ->
    maintenance-window bin, bands stacked as a trailing channel x band feature axis.

    Referencing and the mains notch happen exactly once regardless of how many
    bands are requested (unlike calling a single-band tensor function once per
    band, which would redo both for every band). A band whose high edge reaches
    FIELD_NYQUIST_MARGIN x this signal's own Nyquist frequency is dropped and
    named in ``dropped_bands`` rather than handed to band_power's bandpass_filter,
    which raises past the true Nyquist.

    Parameters
    ----------
    epochs   : (trials, channels, samples) raw signal, already cut to cover window_s
    times    : (samples,) seconds, same time origin as window_s
    labels   : length-channels electrode labels
    srate    : sampling rate, Hz
    window_s : (t0, t1) maintenance sub-window to bin, in ``times``' own time base
    n_bins   : number of equal-width maintenance bins
    referencing : 'bipolar' (bipolar_reference_by_shank, depth contacts) or
                  'as_released' (identity, no re-referencing beyond the notch)

    Returns
    -------
    dict:
      tensor         : (trials, n_bins, n_channels_out * n_kept_bands) float32,
                        log10 power, channel-major (every kept band of channel 0,
                        then channel 1, ...)
      channel_labels : list[str], length n_channels_out
      kept_bands     : list[str], bands actually stacked, BAND_DEFINITIONS order
      dropped_bands  : list[str], bands dropped for reaching the Nyquist margin
    """
    n_trials, n_channels, n_samples = epochs.shape
    notched = np.empty_like(epochs, dtype=np.float64)
    for trial in range(n_trials):
        notched[trial] = line_noise_notch(
            epochs[trial].T, srate, fundamental=mains_hz, n_harmonics=3,
        ).T
    flat = notched.transpose(0, 2, 1).reshape(-1, n_channels)
    if referencing == "bipolar":
        referenced_flat, out_labels = bipolar_reference_by_shank(flat, labels)
    elif referencing == "as_released":
        referenced_flat, out_labels = flat, list(labels)
    else:
        raise ValueError(f"unknown referencing: {referencing!r}")
    n_out = referenced_flat.shape[1]
    referenced = referenced_flat.reshape(n_trials, n_samples, n_out).transpose(0, 2, 1)

    nyquist = srate / 2.0
    kept_bands, dropped_bands, band_power_by_band = [], [], []
    for band_name in bands:
        lo, hi = BAND_DEFINITIONS[band_name]
        if hi >= FIELD_NYQUIST_MARGIN * nyquist:
            dropped_bands.append(band_name)
            continue
        kept_bands.append(band_name)
        power = np.empty_like(referenced, dtype=np.float64)
        for trial in range(n_trials):
            power[trial] = band_power(referenced[trial].T, (lo, hi), srate=srate, smooth_ms=0.0).T
        band_power_by_band.append(power)
    if not kept_bands:
        raise ValueError(
            f"every requested band's high edge reaches {FIELD_NYQUIST_MARGIN:g} x Nyquist "
            f"({nyquist:g} Hz) at a {srate:g} Hz sampling rate"
        )

    win_mask = (times >= window_s[0]) & (times < window_s[1])
    edges = np.linspace(0, int(win_mask.sum()), n_bins + 1).astype(int)
    binned_by_band = []
    for power in band_power_by_band:
        win_power = power[:, :, win_mask]
        binned = np.stack(
            [np.log10(np.clip(win_power[:, :, edges[k]:edges[k + 1]].mean(axis=2), 1e-12, None))
             for k in range(n_bins)], axis=2,
        )  # (trials, channels_out, n_bins)
        binned_by_band.append(binned)
    stacked = np.stack(binned_by_band, axis=-1)  # (trials, channels_out, n_bins, n_kept_bands)
    tensor = stacked.transpose(0, 2, 1, 3).reshape(n_trials, n_bins, n_out * len(kept_bands))
    return {
        "tensor": tensor.astype(np.float32), "channel_labels": out_labels,
        "kept_bands": kept_bands, "dropped_bands": dropped_bands,
    }


MASTOID_LABELS = ("A1", "A2")


def scalp_reference_excluding_mastoids(data: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Common-average reference across scalp channels, excluding the two
    mastoids from both the average and the analysis -- pre-declared before
    any scalp session is fit, not improvised per session. The depth path's
    bipolar-by-shank scheme (shank_bipolar_index_pairs) has no meaning for a
    scalp 10-20 montage, which has no shank structure to difference along, so
    it is not reused here.

    Parameters
    ----------
    data   : (T, C)
    labels : length-C electrode labels (10-20 names plus the two mastoids)

    Returns
    -------
    referenced  : (T, C - n_mastoids_present)
    kept_labels : list[str], mastoid labels removed, order otherwise preserved
    """
    keep = [i for i, lab in enumerate(labels) if lab not in MASTOID_LABELS]
    referenced = common_average_reference(data[:, keep])
    kept_labels = [labels[i] for i in keep]
    return referenced, kept_labels


def scalp_low_band_maintenance_tensor(
    eeg: dict, trial_mask: np.ndarray, n_bins: int, lo: float = 8.0, hi: float = 45.0,
) -> tuple[np.ndarray, list[str]]:
    """Notch -> common-average-reference-excluding-mastoids -> low-band
    envelope power -> maintenance-window bin. Mirrors lfp_maintenance_tensor's
    structure exactly (same mains notch, same maintenance window, same
    binning convention, same unsmoothed envelope); the referencing step
    differs because a scalp montage has no shank structure to bipolar-
    reference along (scalp_reference_excluding_mastoids). The 8-45 Hz default
    is fixed by two physical constraints -- below the 50 Hz mains notch and
    its harmonics, and below the scalp series' own Nyquist frequency with
    margin -- not tuned per call.

    Returns
    -------
    tensor      : (trials, channels, n_bins) float32
    kept_labels : list[str], the channel labels surviving mastoid exclusion
    """
    epochs = eeg["epochs"][trial_mask]
    srate = eeg["srate"]
    labels = eeg["electrode_labels"]
    n_trials, n_channels, n_samples = epochs.shape
    notched = np.empty_like(epochs)
    for trial in range(n_trials):
        notched[trial] = line_noise_notch(
            epochs[trial].T, srate, fundamental=BORAN_MAINS_HZ, n_harmonics=3,
        ).T
    flat = notched.transpose(0, 2, 1).reshape(-1, n_channels)
    referenced_flat, kept_labels = scalp_reference_excluding_mastoids(flat, labels)
    referenced = referenced_flat.reshape(n_trials, n_samples, -1).transpose(0, 2, 1).astype(np.float32)
    band_power = compute_boran_hgp(referenced, srate=srate, lo=lo, hi=hi, smooth_ms=0.0)
    maint_mask = (eeg["times"] >= -MAINT_WIN) & (eeg["times"] < 0)
    band_maint = band_power[:, :, maint_mask]
    edges = np.linspace(0, band_maint.shape[2], n_bins + 1).astype(int)
    tensor = np.stack(
        [band_maint[:, :, edges[k]:edges[k + 1]].mean(axis=2) for k in range(n_bins)], axis=2,
    )
    return tensor, kept_labels


def session_cells(sess):
    tc, load = sess["task_condition"], sess["load"]
    psth = sess["counts"]
    out = {}

    for op_a, op_b in OPERATION_PAIRS:
        name = f"operation_{op_a}_vs_{op_b}"
        mask = np.isin(tc, (op_a, op_b))
        y = (tc[mask] == op_b).astype(float)
        tag = f"ds005034_{sess['session']}_{name}"
        out[name] = fit_cell(tag, psth[mask], y)

    name = "operation_forward_vs_transform"
    y = (tc != "forward").astype(float)
    out[name] = fit_cell(f"ds005034_{sess['session']}_{name}", psth, y)

    for op in OPERATION_LEVELS:
        name = f"load_within_{op}"
        mask = tc == op
        y = (load[mask] == 6).astype(float)
        out[name] = fit_cell(f"ds005034_{sess['session']}_{name}", psth[mask], y)

    return out


OPERATION_LEVELS = ("forward", "backward", "alphabetical")


OPERATION_PAIRS = (("forward", "backward"), ("forward", "alphabetical"), ("backward", "alphabetical"))


def fit_cell(tag, psth, y_raw):
    y = np.asarray(y_raw)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < MIN_TRIALS_PER_CLASS:
        return None
    n_units = psth.shape[1]
    n_comp = max(2, min(N_PC, n_units - 1))
    n_splits = _choose_n_splits(y)
    rng = np.random.default_rng(stable_seed(tag))
    auc_mat = within_window_ctg(psth, y, n_comp, n_splits, rng)
    stats = window_stats(auc_mat)
    stats["min_class_count"] = int(counts.min())
    stats["admitted_at_comparison_floor"] = bool(counts.min() >= MIN_TRIALS_PER_CLASS_COMPARISON)
    return stats


N_PC = 8


BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 20.0)}


BASELINE_WINDOW = (-3.7, -2.7)


CELLS = tuple(itertools.product(("forward", "backward", "alphabetical"), (4, 6)))


DELAY_WINDOW = (0.5, 6.5)


MAX_GLOBAL_BAD_FRACTION = 0.20


POSTERIOR_ROI = (
    "E60", "E62", "E85", "E59", "E67", "E77", "E91", "E58", "E66",
    "E72", "E84", "E96", "E65", "E90", "E70", "E75", "E83",
)


def within_window_ctg(psth, y, n_components, n_splits, rng):
    y = np.asarray(y)
    t_idx = np.arange(psth.shape[2])
    fold_mats = []
    for tr_idx, te_idx in _ctg_splits(y, n_splits, rng, None):
        X_tr, X_te = _standardize_train_test(psth[tr_idx], psth[te_idx], psth.shape[2], joint=True)
        mu, V = _fit_pca_fold(X_tr, n_components)
        Z_tr, Z_te = _project_fold(X_tr, mu, V), _project_fold(X_te, mu, V)
        fold_mats.append(_ctg_score_fold(Z_tr, y[tr_idx], Z_te, y[te_idx], t_idx))
    return np.nanmean(np.stack(fold_mats), axis=0)


def window_stats(auc_mat):
    n = auc_mat.shape[0]
    offdiag_mask = ~np.eye(n, dtype=bool)
    return {
        "offdiag_effect": float(np.nanmean(auc_mat[offdiag_mask]) - 0.5),
        "diag_mean_auc": float(np.nanmean(np.diag(auc_mat))),
    }


TARGET_SFREQ = 250.0


THETA_ROI = ("E15", "E18", "E10", "E11", "E16")


def _global_bad_channels(data_uv: np.ndarray, info, seed: int) -> list[str]:
    import mne
    from pyprep.find_noisy_channels import NoisyChannels

    n_times = data_uv.shape[1]
    width = int(2 * 1000)
    starts = np.linspace(0, max(n_times - width, 0), 60).astype(int)
    sampled = np.concatenate([data_uv[:, start:start + width] for start in starts], axis=1)
    sampled = resample_poly(sampled, int(TARGET_SFREQ), 1000, axis=1) / 1e6
    sampled = bandpass_filter(sampled.T, 1.0, 45.0, TARGET_SFREQ).T
    sample_raw = mne.io.RawArray(sampled, info, verbose="ERROR")
    detector = NoisyChannels(sample_raw, random_state=seed, ransac=True)
    detector.find_all_bads(ransac=True, channel_wise=True, max_chunk_size=30)
    return sorted(detector.get_bads())


def _interpolate(data_uv: np.ndarray, info, bad_names: list[str]) -> np.ndarray:
    if not bad_names:
        return data_uv
    import mne

    raw = mne.io.RawArray(data_uv / 1e6, info, verbose="ERROR")
    raw.info["bads"] = bad_names
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    return raw.get_data() * 1e6


def _mne_session_context(set_path: Path):
    import mne

    raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="ERROR")
    montage = raw.get_montage()
    info = mne.create_info(raw.ch_names, TARGET_SFREQ, ch_types="eeg")
    info.set_montage(montage, on_missing="raise")
    identity = mne.io.RawArray(np.eye(len(raw.ch_names)), info, verbose="ERROR")
    csd = mne.preprocessing.compute_current_source_density(
        identity, lambda2=1e-5, stiffness=4, n_legendre_terms=50, copy=True,
    )
    return raw, info, csd.get_data()


def load_events(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    delays = []
    for row in rows:
        if row["value"] not in EVENT_MAP:
            continue
        task, load = EVENT_MAP[row["value"]]
        delays.append({"onset": float(row["onset"]), "task": task, "load": load, "code": row["value"]})
    return delays


EVENT_MAP = {
    "DIN104": ("forward", 4), "DIN106": ("forward", 6),
    "DIN114": ("backward", 4), "DIN116": ("backward", 6),
    "DIN124": ("alphabetical", 4), "DIN126": ("alphabetical", 6),
}


def periodogram_band_power(
    data: np.ndarray, sfreq: float, interval: tuple[float, float], times: np.ndarray,
    csd_transform: np.ndarray,
) -> np.ndarray:
    mask = (times >= interval[0]) & (times < interval[1])
    segment = detrend(data[:, mask], axis=1, type="linear")
    taper = windows.hann(segment.shape[1], sym=False)
    spectrum = np.fft.rfft(segment * taper[None, :], axis=1)
    spectrum = csd_transform @ spectrum
    frequencies = np.fft.rfftfreq(segment.shape[1], 1.0 / sfreq)
    scale = sfreq * np.sum(taper**2)
    power = np.abs(spectrum) ** 2 / scale
    return np.stack([
        power[:, (frequencies >= lo) & (frequencies < hi)].mean(axis=1)
        for lo, hi in BANDS.values()
    ], axis=1)


BIN_MS = 100


N_PERM = 5000


def _phase_diffusion(trials: np.ndarray, center: np.ndarray, scale: np.ndarray,
                     pca: PCA, sampling_rate: float) -> dict:
    binned = bin_analog_trials(trials, sampling_rate)
    latent = pca.transform(((binned.transpose(0, 2, 1) - center) / scale).reshape(-1, len(center)))
    latent = latent.reshape(len(binned), binned.shape[-1], -1)
    residuals, _ = leave_one_out_condition_residuals(latent, np.zeros(len(latent), dtype=int))
    state_rows = []
    increment_diffusion = []
    dt = BIN_MS / 1000.0
    for component in range(latent.shape[-1]):
        values = residuals[..., component]
        estimate = fit_gaussian_state_space(values, dt)
        state_rows.append(estimate.to_dict())
        increment_diffusion.append(float(np.nanmean(np.diff(values, axis=1) ** 2) / (2.0 * dt)))
    process = [row["diffusion"] for row in state_rows if row.get("diffusion") is not None
               and np.isfinite(row["diffusion"])]
    return {
        "state_space_total_diffusion": float(np.sum(process)) if process else None,
        "legacy_increment_total_diffusion": float(np.sum(increment_diffusion)),
        "state_space_components": state_rows,
        "n_trials": int(len(trials)),
        "n_components_estimable": len(process),
        "n_lambda_precision_identified": sum(
            row.get("lambda_ci") is not None and row["lambda_ci"][0] > 0 for row in state_rows
        ),
    }


def active_control_difference(rows: list[dict], key: str, n_perm: int = N_PERM) -> dict | None:
    groups = {}
    for group in ("active", "control"):
        vectors = []
        for row in rows:
            harmonic = row.get(key)
            if row.get("group") == group and harmonic is not None:
                vectors.append([harmonic["cosine"], harmonic["sine"]])
        groups[group] = np.asarray(vectors, dtype=float)
    if min(len(groups["active"]), len(groups["control"])) < 3:
        return None
    observed = groups["active"].mean(0) - groups["control"].mean(0)
    pooled = np.vstack((groups["active"], groups["control"]))
    rng = np.random.default_rng(stable_seed(f"haslacher_active_control_{key}"))
    null = np.empty(n_perm)
    n_active = len(groups["active"])
    for index in range(n_perm):
        permuted = pooled[rng.permutation(len(pooled))]
        null[index] = np.linalg.norm(permuted[:n_active].mean(0) - permuted[n_active:].mean(0))
    magnitude = float(np.linalg.norm(observed))
    return {"difference_cosine": float(observed[0]), "difference_sine": float(observed[1]),
            "difference_amplitude": magnitude,
            "participant_label_permutation_p_value": float((1 + np.sum(null >= magnitude)) / (n_perm + 1)),
            "n_active": len(groups["active"]), "n_control": len(groups["control"])}


def bin_analog_trials(trials: np.ndarray, sampling_rate: float, bin_ms: int = BIN_MS) -> np.ndarray:
    """Average analog samples into nonoverlapping bins without smoothing."""
    values = np.asarray(trials, dtype=float)
    samples = int(round(sampling_rate * bin_ms / 1000.0))
    n_bins = values.shape[-1] // samples
    if values.ndim != 3 or samples < 1 or n_bins < 4:
        raise ValueError("trials must be (trial, channel, time) with at least four bins")
    return values[..., :n_bins * samples].reshape(values.shape[0], values.shape[1], n_bins, samples).mean(-1)


def group_vector_test(rows: list[dict], key_path: tuple[str, ...], seed: str,
                      n_perm: int = N_PERM) -> dict | None:
    """Population circular-vector test with participant-level phase rotations."""
    vectors = []
    for row in rows:
        value = row
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            vectors.append([value["cosine"], value["sine"]])
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) < 3:
        return None
    observed = vectors.mean(axis=0)
    rng = np.random.default_rng(stable_seed(seed))
    null = np.empty(n_perm)
    angles = np.arange(6) * np.pi / 3.0
    for index in range(n_perm):
        rotations = rng.choice(angles, size=len(vectors))
        cosine, sine = np.cos(rotations), np.sin(rotations)
        rotated = np.column_stack((vectors[:, 0] * cosine - vectors[:, 1] * sine,
                                   vectors[:, 0] * sine + vectors[:, 1] * cosine))
        null[index] = np.linalg.norm(rotated.mean(axis=0))
    bootstrap = np.array([vectors[rng.integers(0, len(vectors), len(vectors))].mean(axis=0)
                          for _ in range(2000)])
    magnitude = float(np.linalg.norm(observed))
    return {"mean_cosine": float(observed[0]), "mean_sine": float(observed[1]),
            "population_amplitude": magnitude,
            "optimal_phase_deg": float(np.degrees(np.arctan2(observed[1], observed[0])) % 360.0),
            "participant_bootstrap_cosine_ci": np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
            "participant_bootstrap_sine_ci": np.quantile(bootstrap[:, 1], [0.025, 0.975]).tolist(),
            "circular_rotation_p_value": float((1 + np.sum(null >= magnitude)) / (n_perm + 1)),
            "n_participants": int(len(vectors))}


ACTIVE_SUBJECTS = ([f"PA{i}" for i in range(1, 17)] + ["PA18", "PA19", "PA20", "PA22", "PA23"])


CONTROL_SUBJECTS = (["PA17", "PA21"] + [f"PA{i}" for i in range(24, 47)])


PHASE_CONDITIONS = {3: 30, 4: 90, 5: 150, 6: 210, 1: 270, 2: 330}


def _modulation(values_in_phase_order: np.ndarray) -> tuple[float, float]:
    """Single-cycle DFT modulation depth and optimal phase (rad) -- the
    dataset README's own method (sec 10), reused directly with attribution."""
    x = np.asarray(values_in_phase_order, float)
    n = len(x)
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = (x * np.exp(-1j * phases)).sum() * 2 / n
    wrapped = (np.angle(c) + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(c)), float(wrapped)


def modulation_from_outcomes(outcomes: list[tuple[int, int]], rng: np.random.Generator) -> dict:
    codes_ordered = sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c])
    by_code = {c: [] for c in codes_ordered}
    for code, correct in outcomes:
        by_code[code].append(correct)
    accuracy = np.array([np.mean(by_code[c]) if by_code[c] else np.nan for c in codes_ordered])
    if np.any(np.isnan(accuracy)):
        return {"depth": None, "optimal_phase_deg": None, "p_value": None,
                "n_trials": len(outcomes), "reason": "at least one phase condition has zero trials"}

    depth_obs, phase_obs = _modulation(accuracy)
    codes_arr = np.array([c for c, _ in outcomes])
    correct_arr = np.array([v for _, v in outcomes])
    null = np.empty(N_PERM_MODULATION)
    for p in range(N_PERM_MODULATION):
        shuffled_codes = rng.permutation(codes_arr)
        acc_p = np.array([correct_arr[shuffled_codes == c].mean() if np.any(shuffled_codes == c)
                          else np.nan for c in codes_ordered])
        d_p, _ = _modulation(acc_p) if not np.any(np.isnan(acc_p)) else (np.nan, 0.0)
        null[p] = d_p
    valid = null[~np.isnan(null)]
    p_value = permutation_pvalue(valid >= depth_obs) if len(valid) else float("nan")
    return {"depth": depth_obs, "optimal_phase_deg": float(np.degrees(phase_obs)),
            "p_value": float(p_value), "n_trials": len(outcomes),
            "accuracy_by_phase_deg": {PHASE_CONDITIONS[c]: float(accuracy[i])
                                      for i, c in enumerate(codes_ordered)}}


N_PERM_MODULATION = 2000


def harmonic_coefficients(values: dict[int, float]) -> dict | None:
    """Fit one circular harmonic to six phase-condition values."""
    ordered = sorted(PHASE_CONDITIONS, key=lambda code: PHASE_CONDITIONS[code])
    if any(code not in values or not np.isfinite(values[code]) for code in ordered):
        return None
    phase = np.deg2rad([PHASE_CONDITIONS[code] for code in ordered])
    y = np.array([values[code] for code in ordered], dtype=float)
    design = np.column_stack((np.ones(len(y)), np.cos(phase), np.sin(phase)))
    intercept, cosine, sine = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"intercept": float(intercept), "cosine": float(cosine), "sine": float(sine),
            "amplitude": float(np.hypot(cosine, sine)),
            "optimal_phase_deg": float(np.degrees(np.arctan2(sine, cosine)) % 360.0)}


AUX_CHANNELS = ["envelope", "stim"]


GRAMIAN_HORIZON = 20


NOT_OF_INTEREST = {
    "active": ["Fp1", "Fpz", "Fp2", "FC1", "Fz", "FC2", "C1", "Cz", "C2", "CP1",
               "CPz", "CP2", "F9", "F10", "FT9", "FT10", "TP9", "TP10", "O1", "O2"],
    "control": ["Fp1", "Fpz", "Fp2", "FC1", "C5", "FC2", "C1", "Cz", "C2", "CP1",
                "CPz", "CP2", "F9", "F10", "FT9", "FT10", "TP9", "TP10", "O1", "O2"],
}


N_RANDOM_DIRS = 20


PROTECT = ["Pz", "PO7", "PO8", "P3", "P4"]


SATURATION_THRESHOLD = 0.418


SFREQ_ANALYSIS = 200.0


RETENTION_TMIN, RETENTION_TMAX = 0.6, 3.6  # dataset's own retention window (sec 3/7)


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


def _sass(no_stim: "mne.io.Raw", stim: "mne.io.Raw") -> int:
    """Project the tACS artifact out of `stim` in place, using `no_stim` as
    the artifact-free reference (published SASS method, NeuroImage 2021,
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


STIM_ELECTRODES = {"active": ["O1", "O2"], "control": ["Fpz", "Cz"]}


AXIS_WINDOW = (0.2, 1.0)


POST_MS = 1500.0


PRE_MS = 200.0


def bin_time_axis(epochs_ct: np.ndarray, times: np.ndarray, bin_ms: float, srate: float) -> tuple[np.ndarray, np.ndarray]:
    """Block-average the trailing time axis into non-overlapping bins.

    Unlike a smoothing kernel, non-overlapping block averaging does not
    introduce cross-bin autocorrelation, so it stays compatible with
    drift_dynamics.py's unsmoothed-observation requirement.
    """
    bin_samples = max(int(round(bin_ms * srate / 1000.0)), 1)
    n_bins = epochs_ct.shape[-1] // bin_samples
    trimmed = epochs_ct[..., : n_bins * bin_samples]
    binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
    time_trimmed = times[: n_bins * bin_samples]
    bin_times = time_trimmed.reshape(n_bins, bin_samples).mean(axis=-1)
    return binned, bin_times


def iid_log_likelihood(test: np.ndarray, train: np.ndarray) -> float:
    variance = max(float(np.nanvar(train)), 1e-10)
    values = test[np.isfinite(test)]
    return float(np.sum(-0.5 * (np.log(2 * np.pi * variance) + values * values / variance)))


TARGET_HZ = 100.0


def prepare_epoch(epoch, measure: str) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.asarray(epoch.trial, dtype=float)
    time = np.asarray(epoch.time, dtype=float)
    native_hz = float(1.0 / np.median(np.diff(time)))
    data = data - data.mean(axis=1, keepdims=True)
    if measure == "alpha_power":
        sos = butter(4, [8.0, 12.0], btype="bandpass", fs=native_hz, output="sos")
        data = np.log(np.abs(hilbert(sosfiltfilt(sos, data, axis=2), axis=2)) ** 2 + 1e-12)
        data = data - data.mean(axis=1, keepdims=True)
    elif measure != "voltage":
        raise ValueError(f"unknown measure: {measure}")
    stride = max(1, int(round(native_hz / TARGET_HZ)))
    return data[:, :, ::stride], time[::stride], native_hz / stride


def valid_mask(epoch, n_trials: int) -> np.ndarray:
    mask = np.ones(n_trials, dtype=bool)
    bad = np.atleast_1d(epoch.bad_trials)
    if bad.size and np.all(np.isfinite(bad)):
        mask[bad.astype(int) - 1] = False
    return mask
