"""
preprocessing.py — multi-dataset iEEG signal processing pipeline.

Datasets supported:
  1. Kai Miller N-back iEEG (Miller et al. 2007, 2016)
       4 subjects (al, ca, cc, ug); ECoG; 1200 Hz; 0/1/2-back verbal N-back
       MAT format. Functions: load_subject, load_miller_nback, compute_hgp

  2. DANDI 000574 — Boran et al. (2025) Sternberg WM
       9 subjects; iEEG + EEG + LFP + single units (MTL: hippocampus, amygdala,
       temporal cortex); NWB format; Sternberg set sizes 4/6/8
       References: Boran et al. 2025; Sarnthein lab Zurich
       Functions: load_boran_nwb, compute_boran_hgp

  3. TES1 — Huang et al. (2017) eLife transcranial electrical stimulation
       17 subjects; intracranial voltage induced by 1 mA tES; MNI coordinates
       Provides the input matrix B for LQR control (tES → brain field mapping)
       Functions: load_tes1_stimulation, build_tes1_input_matrix


Signal processing methods:
  Crone et al. (1998) for high-gamma extraction
  Engel et al. (2005) for common average reference rationale
  Vogelstein et al. (2010) for AR(1) deconvolution framework
  Ray & Maunsell (2011) for broadband HGP as MUA proxy
"""

from __future__ import annotations

import io
import zipfile
import numpy as np
import scipy.signal as sig
import scipy.io as sio
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_EXT_DATA = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data"
)
DATA_DIR = _EXT_DATA / "kai miller" / "memory_nback" / "memory_nback" / "data"
MILLER_DATA_DIR = _EXT_DATA / "kai miller"
BORAN_DATA_DIR = _EXT_DATA / "000574"
TES1_ZIP_PATH = _EXT_DATA / "Tes1" / "HuangLiu2016dataset.zip"

SUBJECTS = ["al", "ca", "cc", "ug"]
SRATE = 1200  # Hz  (Miller)
BORAN_SRATE_IEEG = 1398  # Hz (DANDI 000574 iEEG)
BORAN_SRATE_LFP = 22370  # Hz (DANDI 000574 raw LFP)

TASK_CODES = {-1: "rest", 0: "zero_back", 1: "one_back", 2: "two_back"}
TARGET_CODES = {0: "no_stim", 1: "non_target", 2: "target"}


# ── Loading ────────────────────────────────────────────────────────────────────

def load_subject(subj: str, data_dir: Path = DATA_DIR) -> dict:
    """Load one subject's N-back iEEG data from the Miller dataset.

    Parameters
    ----------
    subj : one of {'al', 'ca', 'cc', 'ug'}

    Returns
    -------
    dict with keys:
      data    : (T, C) float64 — ECoG voltage, pre-filtered 1-300 Hz
      stim    : (T,)  uint8  — letter identity (0=ISI, 1-40=letter)
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

def bandpass_filter(
    data: np.ndarray, lo: float, hi: float, srate: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (sosfiltfilt for numerical stability).

    Parameters
    ----------
    data  : (T,) or (T, C)
    lo, hi : passband edges in Hz
    order : filter order (effective 2*order after forward-backward pass)
    """
    nyq = srate / 2.0
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


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the cross-electrode mean at each time point.

    Removes volume-conducted and reference-electrode artefacts while
    preserving spatially local signals (Engel et al. 2005, Clin. Neurophysiol.).

    Parameters
    ----------
    data : (T, C)

    Returns
    -------
    data_car : (T, C)
    """
    return data - data.mean(axis=1, keepdims=True)


def reject_bad_channels(
    data: np.ndarray, threshold_mad: float = 3.0
) -> np.ndarray:
    """Return boolean mask of channels whose variance is within threshold MADs of median.

    Uses the median absolute deviation (MAD) scaled to match SD under normality
    (scale factor 1.4826) for robustness to outlier channels.

    Parameters
    ----------
    threshold_mad : rejection z-score in robust MAD units (3.0 = 3 MAD)

    Returns
    -------
    good : (C,) bool  — True = keep
    """
    ch_var = data.var(axis=0)
    med = np.median(ch_var)
    mad = np.median(np.abs(ch_var - med))
    robust_z = np.abs(ch_var - med) / (1.4826 * mad + 1e-12)
    return robust_z < threshold_mad


def preprocess(
    data: np.ndarray,
    srate: float = SRATE,
    notch_freq: float = 60.0,
    n_harmonics: int = 4,
) -> np.ndarray:
    """Full preprocessing pipeline: CAR → notch (line noise + harmonics).

    The Miller dataset is pre-filtered 1-300 Hz; only CAR and line-noise
    removal are needed.

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
    correlate of cognitive state in ECoG (Ray & Maunsell 2011; Nir et al. 2007).

    Parameters
    ----------
    smooth_ms : Gaussian σ in milliseconds (50 ms = 60 samples at 1200 Hz)

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
      stim_id : (N,) int — letter identity at onset
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
    the convention in Crone et al. 1998 and Miller et al. 2007.

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
    good = reject_bad_channels(d["data"], threshold_mad=bad_ch_threshold)
    data_clean = preprocess(d["data"][:, good])
    hgp = high_gamma_power(data_clean)
    ep = epoch_data(hgp, d["stim"], d["task"], d["target"], pre_ms, post_ms)
    ep["epochs"] = baseline_normalize(ep["epochs"], ep["times"])
    ep["good_channels"] = np.where(good)[0]
    ep["subject"] = subj
    return ep


# ── Miller convenience wrappers (used by notebooks 07/08) ─────────────────────

def load_miller_nback(
    mat_path: str,
    pre_ms: float = 200.0,
    post_ms: float = 1500.0,
    bad_ch_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one Miller N-back MAT file and return epoch tensor.

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

    good = reject_bad_channels(data, threshold_mad=bad_ch_threshold)
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


# ── DANDI 000574 — Boran et al. Sternberg WM ──────────────────────────────────

def load_boran_nwb(
    nwb_path: str,
    signal: str = "ieeg",
    epoch_win: tuple[float, float] = (-6.5, 2.5),
) -> dict:
    """Load Boran et al. Sternberg WM task from NWB file (DANDI 000574).

    Task structure (time relative to probe onset at t=0):
      Fixation:    [-6, -5] s
      Encoding:    [-5, -3] s  (2s, 4/6/8 letters)
      Maintenance: [-3,  0] s  (3s delay)
      Probe:       [ 0, +2] s

    Uses h5py to avoid requiring pynwb. Electrode brain areas include MTL
    (hippocampus CA1/CA3, amygdala, entorhinal), superior/middle temporal
    gyrus, and scalp EEG — enabling multi-region geometry analysis not
    possible with the Miller dataset.

    Parameters
    ----------
    nwb_path : path to .nwb file
    signal   : 'ieeg' (1398 Hz, 48 ch) | 'eeg' (140 Hz, 19 ch) | 'lfp' (raw)
    epoch_win: (t_pre, t_post) in seconds relative to probe onset

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
      srate        : float
    """
    import h5py

    f = h5py.File(str(nwb_path), "r")

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

    f.close()

    epochs_arr = np.stack(epochs_list, axis=0)  # (N, C, T)
    valid_mask = np.array(valid_mask)

    # Filter electrode metadata to only the channels present in epochs
    elec_labels_sel = [elec_labels[i] for i in range(len(elec_labels)) if ch_mask[i]]
    elec_locs_sel   = [elec_locs[i]   for i in range(len(elec_locs))   if ch_mask[i]]

    return {
        "epochs": epochs_arr,
        "times": times,
        "set_sizes": set_sizes,
        "correct": correct,
        "artifact": artifact,
        "valid": valid_mask,
        "electrode_labels": elec_labels_sel,
        "electrode_locs": elec_locs_sel,
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
    """High-gamma power for Boran Sternberg epochs.

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
    """Z-score Boran epochs relative to pre-fixation baseline.

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


# ── TES1 — Huang et al. 2017 eLife tES stimulation field ──────────────────────

def load_tes1_stimulation(
    zip_path: str = str(TES1_ZIP_PATH),
    subject: str | None = None,
) -> dict | list[dict]:
    """Load tES-induced intracranial voltages from Huang et al. 2017 eLife.

    TES1 (CRCNS) provides the voltage at each intracranial electrode induced
    by 1 mA transcranial stimulation. This gives the input matrix B for our
    LQR controller: B[i] = voltage induced at electrode i per mA of current.

    Format per file: electrode_name, MNI_x, MNI_y, MNI_z, voltage_mV
    (voltage calibrated to 1 mA; NaN = high impedance or unlocalized electrode)

    Parameters
    ----------
    zip_path : path to HuangLiu2016dataset.zip
    subject  : subject ID string (e.g. 'P03'). If None, returns list of all.

    Returns
    -------
    dict (or list of dicts) with keys:
      subject      : str
      names        : (N_elec,) list[str]
      mni_coords   : (N_elec, 3) float — MNI x,y,z in mm (NaN if unlocalized)
      voltage_mV   : (N_elec,) float — induced voltage per mA (NaN if rejected)
    """
    z = zipfile.ZipFile(str(zip_path), "r")
    txt_files = [
        n for n in z.namelist()
        if n.endswith(".txt") and "README" not in n
    ]

    def _parse_file(fname: str, subj_id: str) -> dict:
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

    if subject is not None:
        target = f"HuangLiu2016dataset/{subject}.txt"
        result = _parse_file(target, subject)
        z.close()
        return result

    results = []
    for fname in txt_files:
        subj_id = fname.split("/")[-1].replace(".txt", "")
        results.append(_parse_file(fname, subj_id))
    z.close()
    return results


def build_tes1_input_matrix(
    tes1_data: dict,
    target_mni: np.ndarray,
    n_sources: int = 1,
    sigma_mm: float = 10.0,
) -> np.ndarray:
    """Build the LQR input matrix B from TES1 stimulation field data.

    Maps tES surface stimulation (1 mA per source) to intracranial voltage at
    target electrode locations. Uses Gaussian spatial interpolation to map
    available TES1 MNI coordinates to arbitrary target electrode locations.

    This gives the B matrix in: x_{t+1} = Ax_t + Bu_t
    where u_t is the stimulation current vector (amps) and B maps current to
    the neural state change (via induced voltage field).

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

    N_target = target_mni.shape[0]
    B = np.zeros((N_target, n_sources))

    for col in range(min(n_sources, 1)):
        for i, tgt in enumerate(target_mni):
            dists = np.linalg.norm(src_mni - tgt, axis=1)
            weights = np.exp(-dists**2 / (2 * sigma_mm**2))
            weights /= weights.sum() + 1e-10
            B[i, col] = np.dot(weights, src_volt)

    return B
