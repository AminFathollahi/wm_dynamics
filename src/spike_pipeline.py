"""spike_pipeline.py — shared single-unit Sternberg WM pipeline.

Rutishauser-lab Sternberg single-unit datasets (DANDI 000469, 001187, 000673,
and — via Boran's co-located microwires — 000574) share the same underlying
analysis: bin spike times into a firing-rate population vector, PCA-project,
run cross-temporal generalisation (load and item-identity) with a
label-permutation null, cross-validated participation ratio, and a
correct-vs-error drift comparison. NWB field/group names differ slightly
across releases (e.g. 000469's ``trials`` vs 001187's ``WM_trials``, and
``loadsEnc1_PicIDs`` vs ``PicIDs_Encoding1``), so each dataset gets a thin
driver script that extracts these into a common array representation and
calls the functions here — the numerical pipeline itself is written once.
"""

from __future__ import annotations

import sys
import os
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

# Allow `from geometry import ...` to work whether src/ is on sys.path or not.
_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from geometry import (ctg_label_permutation_null, ctg_content_permutation_null,
                      temporal_stability_tau, spatiotemporal_participation_ratio,
                      geometric_drift)

N_PC_DEFAULT = 8
BIN_MS_DEFAULT = 100
SMOOTH_MS_DEFAULT = 200

# QC floors documented in Daume et al. 2024 (Neuron; DANDI 001187 source
# paper): 2/48 sessions excluded for behavioural accuracy below this floor,
# 67/883 neurons (7.1%) excluded for firing rate below this floor. Applied
# uniformly across all three Rutishauser-lineage Sternberg cohorts sharing
# this pipeline (000469, 001187, 000673) for cross-cohort QC consistency,
# even where a given cohort's own paper does not separately restate them.
MIN_SESSION_ACCURACY = 0.55
MIN_UNIT_FIRING_RATE_HZ = 0.1


def load_spike_times(f) -> list[NDArray]:
    """Return list of per-unit spike-time arrays from an NWB ``units`` table."""
    times_flat = f["units/spike_times"][:]
    index = f["units/spike_times_index"][:]
    units, prev = [], 0
    for idx in index:
        units.append(times_flat[prev:idx])
        prev = idx
    return units


def low_rate_unit_mask(
    spike_lists: list[NDArray],
    onsets: NDArray,
    window_s: float,
    min_rate_hz: float = MIN_UNIT_FIRING_RATE_HZ,
) -> NDArray:
    """Boolean keep-mask: True where a unit's mean firing rate across all
    trial windows (onset, onset+window_s) is >= min_rate_hz."""
    onsets = np.asarray(onsets)
    total_time = len(onsets) * window_s
    keep = np.zeros(len(spike_lists), dtype=bool)
    if total_time <= 0:
        return keep
    for u, spk in enumerate(spike_lists):
        n_spikes = sum(int(np.sum((spk >= t0) & (spk < t0 + window_s))) for t0 in onsets)
        keep[u] = (n_spikes / total_time) >= min_rate_hz
    return keep


def build_psth(
    spike_lists: list[NDArray],
    onsets: NDArray,
    bin_ms: float = BIN_MS_DEFAULT,
    smooth_ms: float = SMOOTH_MS_DEFAULT,
    window_s: float = 3.0,
) -> NDArray:
    """Smoothed per-trial firing-rate matrix aligned to `onsets`.

    Returns (n_trials, n_units, n_bins) in spikes/s.
    """
    bin_s = bin_ms / 1000.0
    edges = np.arange(0.0, window_s + bin_s, bin_s)
    n_bins = len(edges) - 1
    psth = np.zeros((len(onsets), len(spike_lists), n_bins), dtype=np.float32)
    for tr, t0 in enumerate(onsets):
        for u, spk in enumerate(spike_lists):
            counts, _ = np.histogram(spk - t0, bins=edges)
            psth[tr, u, :] = counts / bin_s
    if smooth_ms > 0:
        psth = gaussian_filter1d(psth, sigma=smooth_ms / bin_ms, axis=2)
    return psth


def zscore_psth(psth: NDArray) -> NDArray:
    """Z-score each unit/bin across trials."""
    mu = psth.mean(axis=0, keepdims=True)
    sd = psth.std(axis=0, keepdims=True) + 1e-8
    return (psth - mu) / sd


def fit_pca_psth(psth: NDArray, n_comp: int = N_PC_DEFAULT) -> tuple[NDArray, NDArray, float]:
    """PCA on (n_trials, n_units, n_bins) -> (n_trials, n_bins, n_comp) latent + loadings."""
    N, U, T = psth.shape
    X = psth.transpose(0, 2, 1).reshape(-1, U)
    X = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:n_comp].T
    Z_flat = X @ V
    Z = Z_flat.reshape(N, T, n_comp)
    var_total = np.linalg.norm(X) ** 2
    var_proj = np.linalg.norm(Z_flat) ** 2
    return Z, V, float(var_proj / (var_total + 1e-10))


def load_vs_load_ctg(
    psth_z: NDArray,
    loads: NDArray,
    low: int,
    high: int,
    n_components: int = N_PC_DEFAULT,
    ctg_step: int = 3,
    n_splits: int = 5,
    n_perm: int = 100,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Load-`low`-vs-load-`high` CTG with label-permutation null, or None if underpowered."""
    if rng is None:
        rng = np.random.default_rng(0)
    mask = (loads == low) | (loads == high)
    if min((loads[mask] == low).sum(), (loads[mask] == high).sum()) < 5:
        return None
    X = psth_z[mask]
    y = (loads[mask] == high).astype(int)
    t_idx = np.arange(0, psth_z.shape[2], ctg_step)
    res = ctg_label_permutation_null(
        X, y, t_idx, n_components=n_components, n_splits=n_splits, n_perm=n_perm, rng=rng
    )
    res["tau_info"] = temporal_stability_tau(res["auc_mat"])
    res["t_idx"] = t_idx
    return res


def item_identity_ctg(
    psth_z: NDArray,
    loads: NDArray,
    item_ids: NDArray,
    target_load: int = 1,
    min_trials: int = 20,
    n_components: int = N_PC_DEFAULT,
    ctg_step: int = 3,
    n_splits: int = 3,
    n_perm: int = 100,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """Item-identity (content) CTG within a fixed load, or None if underpowered."""
    if rng is None:
        rng = np.random.default_rng(0)
    mask = loads == target_load
    if mask.sum() < min_trials:
        return None
    y = item_ids[mask]
    if len(np.unique(y)) < 2:
        return None
    class_counts = np.bincount(y - y.min())
    min_class = class_counts[class_counts > 0].min()
    n_splits_use = max(2, min(n_splits, int(min_class)))
    if min_class < 2:
        return None
    t_idx = np.arange(0, psth_z.shape[2], ctg_step)
    res = ctg_content_permutation_null(
        psth_z[mask], y, t_idx, n_components=min(n_components, mask.sum() - 2),
        n_splits=n_splits_use, n_perm=n_perm, rng=rng,
    )
    return res


def correct_error_drift(
    Z: NDArray,
    loads: NDArray,
    times: NDArray,
    window_s: float,
) -> NDArray:
    """Per-trial drift from the load-conditioned centroid (for pooled correct-vs-error tests).

    Returns drift for every trial; callers pair it with their own correct/error
    labels downstream (e.g. in a correct-vs-error LME), so no response-accuracy
    array is needed here.
    """
    return geometric_drift(Z, loads, times, maint_window=(0.0, window_s))


def pr_by_load(
    psth_z: NDArray,
    loads: NDArray,
    load_values: tuple[int, ...] = (1, 2, 3),
    min_trials: int = 5,
    n_splits: int = 2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Cross-validated spatiotemporal PR (native unit space) per load level."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for ld in load_values:
        mask = loads == ld
        if mask.sum() < min_trials:
            continue
        out[ld] = spatiotemporal_participation_ratio(psth_z[mask], n_splits=n_splits, rng=rng)
    return out
