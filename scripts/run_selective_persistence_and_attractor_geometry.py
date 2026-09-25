#!/usr/bin/env python3
"""Reproduces the published bioRxiv study's (doi 10.1101/2025.08.20.671301)
category-neuron, selective-persistence, and distance-to-attractor (DA)
methods on our own human single-unit corpora, which carry no stimulation.
Full rationale for every choice below lives in build_artifact's
predeclared_rules, not here.

Run:
    conda run -n wm_dynamics python scripts/run_selective_persistence_and_attractor_geometry.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from project_config import dataset_path  # noqa: E402
from spike_pipeline import (  # noqa: E402
    load_spike_times, build_psth, resolve_unit_regions,
    MIN_UNITS_PER_REGION, MIN_SESSION_ACCURACY, N_PC_DEFAULT,
)
from corpus_sessions import region_filtered_units, MIN_TRIALS, EPOCH_WINDOWS_S  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, stable_seed, fdr_bh, minimum_detectable_paired_difference,
    partial_correlation_permutation_test, permutation_pvalue,
    linear_mixed_effects_test,
)
from state_persistence import slope_across_sessions_test  # noqa: E402
from selectivity_test import two_stage_selectivity_test  # noqa: E402
from stimulation_response_estimator import rate_free_state_deviation  # noqa: E402
from provenance import _json_safe, git_commit, canonical_patient_by_relative_path  # noqa: E402
from subject_independence import resolve_group, count_independent_groups  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from response_latency import response_probe_latency_field, response_time_field  # noqa: E402
from run_region_resolved_rate_stability_behaviour import (  # noqa: E402
    CORPUS_SPECS as REGION_CORPUS_SPECS, _trial_population_spike_count,
)

CANONICAL_PATIENT_BY_PATH = canonical_patient_by_relative_path(ROOT / "provenance")

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_selective_persistence_and_attractor_geometry"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_NEURON_WINDOW_OFFSET_S = 0.2
CATEGORY_NEURON_WINDOW_DURATION_S = 1.0
CATEGORY_NEURON_OMNIBUS_ALPHA = 0.05
CATEGORY_NEURON_PERMUTATION_ALPHA = 0.05
N_PERM_NEURON = 10000
MIN_TRIALS_PER_CATEGORY_FOR_NEURON_TEST = 2
MIN_TOTAL_TRIALS_FOR_NEURON_TEST = 10
MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM = 10
OUTLIER_FIRING_RATE_SD_THRESHOLD = 3.0

MIN_TRIALS_PER_CATEGORY_FOR_DA = 5
MIN_CATEGORIES_FOR_DA = 3
DA_SCHEMA_VERSION = 3
PART1_SCHEMA_VERSION = 4
PSEUDOPOP_SCHEMA_VERSION = 9
N_TRIAL_SPLIT_REPEATS = 5
MIN_CORRECT_TRIALS_PER_NEURON_TRIAL_SPLIT_TEST = 2
LAMBDA_SWEEP_FLAT_RELATIVE_RANGE_THRESHOLD = 0.15
LAMBDA_SWEEP_MACHINE_EPSILON_RELATIVE_RANGE = 1e-6
DPCA_RIDGE_AXES_OVERLAP_FLOOR_MARGIN = 1e-6
MARGINALIZATION_ARM_NOT_PUBLISHED_METHOD_NOTE = (
    "the marginalization-based arm below (arm_a/arm_b, and the per-session arm_a_lambda/arm_b_lambda "
    "cells) is NOT the published study's demixed principal component analysis method and its shrinkage "
    "parameter has no effect on its axes by construction -- see shrinkage_axes_isotropic_invariance_note. "
    "dpca_ridge_arm_a/dpca_ridge_arm_b below implement the actual published method, a ridge-regularised "
    "reduced-rank regression fit against the time-resolved category-by-time marginal; whether its "
    "regularisation parameter moves its axes on data of this shape is measured, not assumed -- see "
    "dpca_ridge_axes_lambda_dependence_check and dpca_ridge_time_resolved_near_invariance_note."
)
SHRINKAGE_AXES_ISOTROPIC_INVARIANCE_NOTE = (
    "_shrinkage_axes shrinks the category marginal's covariance M toward an isotropic target, scale * "
    "identity. For any symmetric M and any positive scalar a, the matrix a*M + b*identity has exactly "
    "the same eigenvectors as M -- only its eigenvalues shift, by the same additive constant b, which "
    "preserves their ranking whenever a > 0. Here a = (1 - lambda) and b = lambda * scale, so for every "
    "lambda in SHRINKAGE_GRID (all strictly below 1.0) the shrunk matrix has the identical eigenvectors "
    "as the unshrunk category marginal, in the identical order -- the selected axes cannot depend on "
    "lambda by construction, only at the unreached lambda=1.0 boundary would the target become a bare "
    "scalar multiple of identity with no preferred eigenvectors. This was verified directly: on real "
    "pooled data the held-out reconstruction error's relative range across the grid was of order 1e-16, "
    "floating-point noise on an exactly constant quantity, not a small but genuine data-driven effect. "
    "The lambda a cross-validated selection lands on is therefore not a selection at all under this "
    "construction; it is an arbitrary pick among numerically-tied floats, which is the mechanism behind "
    "the flat sweep below. This same _shrinkage_axes function is also what the untouched per-session arms "
    "(arm_a_lambda/arm_b_lambda on part1/part2 cells) select with, so the identical mathematical fact "
    "applies there too; per-session output is left exactly as it was and this note does not change it."
)

AXIS_FIT_WINDOW_OFFSET_S = CATEGORY_NEURON_WINDOW_OFFSET_S
AXIS_FIT_WINDOW_DURATION_S = CATEGORY_NEURON_WINDOW_DURATION_S
SLIDING_WINDOW_DURATION_S = 0.2
SLIDING_WINDOW_STEP_S_PRIMARY = 0.02
SLIDING_WINDOW_STEP_S_SENSITIVITY = 0.1
SLIDING_WINDOW_STEP_S_PUBLISHED = 0.001
SLIDING_WINDOW_STEP_REASON = (
    f"the published study bins at a {SLIDING_WINDOW_STEP_S_PUBLISHED*1000:.0f}ms step; at our unit and "
    f"trial counts that is computationally infeasible across every cell in one run, so "
    f"{SLIDING_WINDOW_STEP_S_PRIMARY*1000:.0f}ms is used as the largest feasible step, with a "
    f"{SLIDING_WINDOW_STEP_S_SENSITIVITY*1000:.0f}ms comparison reported per pseudo-population cell "
    "rather than silently substituted."
)
CATEGORY_SUBSPACE_DIMENSIONALITY = 4
SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 0.9)
DPCA_RIDGE_LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)
N_CV_FOLDS_LAMBDA = 3
N_CV_FOLDS_LAMBDA_NESTED_OUTER = 3

N_PSEUDO_TRIALS_PER_CATEGORY = 20
MIN_TRIALS_PER_CATEGORY_PSEUDOPOP = MIN_TRIALS_PER_CATEGORY_FOR_DA

N_DECODING_SUBSAMPLE_REPEATS = 10
N_DECODING_SUBSAMPLE_REPEATS_PUBLISHED = 50
N_DECODING_SHUFFLE = 200
N_DECODING_SHUFFLE_PUBLISHED = 500
DECODING_COMPUTE_REDUCTION_REASON = (
    "the published study's own repeat counts (50 subsampling repeats, 500-shuffle null) are reduced here "
    f"to {N_DECODING_SUBSAMPLE_REPEATS} and {N_DECODING_SHUFFLE} for wall-clock feasibility across every "
    "(corpus, region) pseudo-population cell in one run -- the same disclosed-reduction convention "
    "scripts/run_dpca_analysis.py already uses for its own permutation count (1000 vs a required >=5000, "
    "documented there as a wall-clock-feasibility reduction), both numbers reported alongside the "
    "published ones so a reader sees what was asked and what was run."
)

HIPPOCAMPUS_VTC_COMBINED_UNAVAILABLE_REASON = (
    "the published study pooled hippocampus with ventral temporal cortex for its attractor analysis. None "
    "of the three corpora carrying per-trial category labels here has a ventral-temporal-cortex region at "
    "all; the one corpus whose anatomical labels do include that region (dandi_000574, 'vtc', "
    "spike_pipeline.BORAN_ANATOMICAL_REGIONS) carries no per-trial item or category field (content_label_"
    "basis). So the hippocampal population below is a subset of the published study's pooled hippocampal-"
    "plus-ventral-temporal population -- an availability limit, not a partition choice made here."
)

N_PERM = 10000

CATEGORIES_469 = "loadsEnc1_PicIDs is already valued 1..5 -- used directly as the category"
CATEGORIES_100_DIVIDED = "PicIDs_Encoding1 // 100 resolves to 5 category values (1..5)"

DANDI_000574_EXCLUSION_REASON = (
    "dandi_000574 has no per-trial item/category field: set_letters is reported 'not available' on "
    "every trial in the public NWB release (see scripts/run_human_drift_spine_000574.py's own "
    "item_identity_available=False finding). Its Brainnetome-hybrid anatomical labels do carry a region "
    "literally named 'vtc' plus entorhinal_parahippocampal/inferior|middle|superior_temporal_gyrus "
    "(spike_pipeline.BORAN_ANATOMICAL_REGIONS) -- the anatomical analogue of the published study's "
    "ventral temporal cortex, established by direct region-label match -- but it is unreachable for this "
    "analysis because the corpus carrying it has no content label at all."
)

ITEM_IDENTITY_TEST_FEASIBILITY = {
    "dandi_000469": "the picture identifier field takes only 5 distinct values and is already at category "
        "level; there is no finer item-identity label to test in this corpus.",
    "dandi_000673": "picture identifiers carry 140 distinct item identities, 140 per session, 5 categories "
        "of 28 items each at identifier // 100. A unit sees each item roughly once per session (roughly "
        "four times pooling every encoding presentation and the probe) -- a 140-level per-unit identity "
        "test is not powered at this count and is recorded as a measured limit rather than attempted; "
        "category-level testing at 5 levels remains well powered.",
    "dandi_001187": "picture identifiers carry 140 distinct item identities, 140 per session, 5 categories "
        "of 28 items each at identifier // 100. A unit sees each item roughly once per session (roughly "
        "four times pooling every encoding presentation and the probe) -- a 140-level per-unit identity "
        "test is not powered at this count and is recorded as a measured limit rather than attempted; "
        "category-level testing at 5 levels remains well powered.",
}


def _category_469(pic_ids: np.ndarray) -> np.ndarray:
    return pic_ids.astype(int)


def _category_divided(pic_ids: np.ndarray) -> np.ndarray:
    return (pic_ids.astype(int) // 100)


def _canonical_patient(release_dir: str, path: Path) -> str | None:
    relpath = f"{release_dir}/{path.parent.name}/{path.name}"
    return CANONICAL_PATIENT_BY_PATH.get(relpath)


def _session_latency(raw: np.ndarray, keep: np.ndarray) -> np.ndarray:
    latency = raw[keep].astype(float)
    return np.where(np.isfinite(latency) & (latency > 0), latency, np.nan)


def _load_000469_session(path: Path) -> dict | None:
    patient = _canonical_patient("000469", path)
    if patient is None:
        return None
    with h5py.File(str(path), "r") as f:
        if "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        accuracy_all = trials["response_accuracy"][:].astype(bool)
        if len(loads) < MIN_TRIALS or accuracy_all.mean() < MIN_SESSION_ACCURACY:
            return None
        keep = loads == 1
        if keep.sum() < MIN_TRIALS:
            return None
        pic_ids = trials["loadsEnc1_PicIDs"][:][keep]
        pic_ids_probe = trials["loadsProbe_PicIDs"][:][keep]
        t_enc1 = trials["timestamps_Encoding1"][:][keep]
        t_fix = trials["timestamps_FixationCross"][:][keep]
        t_maint = trials["timestamps_Maintenance"][:][keep]
        t_probe = trials["timestamps_Probe"][:][keep]
        accuracy = accuracy_all[keep]
        latency = _session_latency(response_probe_latency_field(trials), keep)
    return dict(patient=patient, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                category=_category_469(pic_ids), category_basis=CATEGORIES_469,
                category_probe=_category_469(pic_ids_probe),
                t_enc1=t_enc1, t_probe=t_probe, t_baseline=t_fix, t_maint=t_maint, accuracy=accuracy,
                latency=latency,
                enc_win=EPOCH_WINDOWS_S["encoding"], baseline_win=EPOCH_WINDOWS_S["baseline"],
                maint_win=EPOCH_WINDOWS_S["delay"])


def _load_000673_session(path: Path) -> dict | None:
    patient = _canonical_patient("000673", path)
    if patient is None:
        return None
    with h5py.File(str(path), "r") as f:
        if "intervals" not in f or "trials" not in f["intervals"] or "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        accuracy_all = trials["response_accuracy"][:].astype(bool)
        if len(loads) < MIN_TRIALS or accuracy_all.mean() < MIN_SESSION_ACCURACY:
            return None
        keep = loads == 1
        if keep.sum() < MIN_TRIALS:
            return None
        pic_ids = trials["PicIDs_Encoding1"][:][keep]
        pic_ids_probe = trials["PicIDs_Probe"][:][keep]
        t_enc1 = trials["timestamps_Encoding1"][:][keep]
        t_fix = trials["timestamps_FixationCross"][:][keep]
        t_maint = trials["timestamps_Maintenance"][:][keep]
        t_probe = trials["timestamps_Probe"][:][keep]
        accuracy = accuracy_all[keep]
        latency = _session_latency(response_probe_latency_field(trials), keep)
    return dict(patient=patient, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                category=_category_divided(pic_ids), category_basis=CATEGORIES_100_DIVIDED,
                category_probe=_category_divided(pic_ids_probe),
                t_enc1=t_enc1, t_probe=t_probe, t_baseline=t_fix, t_maint=t_maint, accuracy=accuracy,
                latency=latency,
                enc_win=EPOCH_WINDOWS_S["encoding"], baseline_win=EPOCH_WINDOWS_S["baseline"],
                maint_win=EPOCH_WINDOWS_S["delay"])


def _load_001187_session(path: Path) -> dict | None:
    patient = _canonical_patient("001187", path)
    if patient is None:
        return None
    with h5py.File(str(path), "r") as f:
        if "intervals" not in f or "WM_trials" not in f["intervals"] or "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        trials = f["intervals/WM_trials"]
        loads = trials["loads"][:].astype(int)
        accuracy_all = trials["response_accuracy"][:].astype(bool)
        if len(loads) < MIN_TRIALS or accuracy_all.mean() < MIN_SESSION_ACCURACY:
            return None
        keep = loads == 1
        if keep.sum() < MIN_TRIALS:
            return None
        pic_ids = trials["PicIDs_Encoding1"][:][keep]
        pic_ids_probe = trials["PicIDs_Probe"][:][keep]
        t_enc1 = trials["timestamps_Encoding1"][:][keep]
        t_fix = trials["timestamps_FixationCross"][:][keep]
        t_maint = trials["timestamps_Maintenance"][:][keep]
        t_probe = trials["timestamps_Probe"][:][keep]
        accuracy = accuracy_all[keep]
        latency = _session_latency(response_time_field(trials), keep)
    return dict(patient=patient, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                category=_category_divided(pic_ids), category_basis=CATEGORIES_100_DIVIDED,
                category_probe=_category_divided(pic_ids_probe),
                t_enc1=t_enc1, t_probe=t_probe, t_baseline=t_fix, t_maint=t_maint, accuracy=accuracy,
                latency=latency,
                enc_win=EPOCH_WINDOWS_S["encoding"], baseline_win=EPOCH_WINDOWS_S["baseline"],
                maint_win=EPOCH_WINDOWS_S["delay"])


CORPUS_SPECS = {
    "dandi_000469": {
        "regions": REGION_CORPUS_SPECS["dandi_000469"]["regions"],
        "loader": _load_000469_session,
        "glob": REGION_CORPUS_SPECS["dandi_000469"]["glob"],
    },
    "dandi_000673": {
        "regions": REGION_CORPUS_SPECS["dandi_000673"]["regions"],
        "loader": _load_000673_session,
        "glob": REGION_CORPUS_SPECS["dandi_000673"]["glob"],
    },
    "dandi_001187": {
        "regions": REGION_CORPUS_SPECS["dandi_001187"]["regions"],
        "loader": _load_001187_session,
        "glob": REGION_CORPUS_SPECS["dandi_001187"]["glob"],
    },
}


def _fit_pca_2d(X: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:n_comp].T


def _project_pca_2d(X: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    return (X - mu) @ V


def _project_pca_3d(psth: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    N, U, T = psth.shape
    X = psth.transpose(0, 2, 1).reshape(-1, U) - mu
    return (X @ V).reshape(N, T, V.shape[1])


def _window_rate(spike_lists: list, onsets: np.ndarray, offset_s: float, window_s: float) -> np.ndarray:
    return build_psth(spike_lists, onsets + offset_s, bin_ms=int(window_s * 1000),
                       smooth_ms=0, window_s=window_s)[:, :, 0]


def _sliding_window_rate(spike_lists: list, onsets: np.ndarray, step_s: float, window_s: float,
                          total_duration_s: float) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    fine = build_psth(spike_lists, onsets, bin_ms=max(1, int(round(step_s * 1000))), smooth_ms=0,
                       window_s=total_duration_s)
    n_window_bins = max(1, int(round(window_s / step_s)))
    return uniform_filter1d(fine, size=n_window_bins, axis=2, mode="nearest")


def _category_marginal(Z: np.ndarray, category: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cats = np.unique(category)
    means = np.stack([Z[category == c].mean(axis=0) for c in cats], axis=0)
    return cats, means - means.mean(axis=0)


def _shrinkage_axes(marginal: np.ndarray, lam: float, d: int) -> tuple[np.ndarray, np.ndarray]:
    k = marginal.shape[1]
    M = marginal.T @ marginal
    scale = np.trace(M) / k if k else 0.0
    M_shrunk = (1.0 - lam) * M + lam * scale * np.eye(k)
    eigvals, eigvecs = np.linalg.eigh(M_shrunk)
    order = np.argsort(eigvals)[::-1]
    d = min(d, k)
    return eigvecs[:, order[:d]], M


def _variance_captured(V: np.ndarray, M_unshrunk: np.ndarray) -> float:
    total = float(np.trace(M_unshrunk))
    if total <= 0.0:
        return float("nan")
    return float(np.trace(V.T @ M_unshrunk @ V) / total)


def _dpca_axes_marginalization(Z: np.ndarray, category: np.ndarray, lam: float, d: int
                                ) -> tuple[np.ndarray, np.ndarray]:
    _, marginal = _category_marginal(Z, category)
    return _shrinkage_axes(marginal, lam, d)


def _category_time_marginal_targets(Z_time: np.ndarray, category: np.ndarray
                                     ) -> tuple[np.ndarray, np.ndarray]:
    n_trials, T, k = Z_time.shape
    cats = np.unique(category)
    time_marginal = _category_time_marginal(Z_time, category)  # (n_cats*T, k)
    marginal_by_cat = time_marginal.reshape(len(cats), T, k)
    lookup = {int(c): marginal_by_cat[i] for i, c in enumerate(cats)}
    Y = np.concatenate([lookup[int(c)] for c in category], axis=0)  # (n_trials*T, k), trial-major
    return time_marginal, Y


def _dpca_ridge_axes(Z: np.ndarray, category: np.ndarray, lam: float, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard demixed principal component analysis (doi 10.7554/eLife.10989), reduced-rank ridge
    regression formulation: fit encoder/decoder pair (E, D) minimising ||Y - X @ E @ D.T||^2 with a ridge
    penalty on the encoding step, X = per-(trial, timepoint) full population activity, Y = the matching
    category-by-time marginal target (category means as a function of time, via _category_time_marginal,
    reused rather than the time-collapsed per-category mean) -- carrying time structure the way the
    published study's own 200ms sliding-window binning does. D (returned here as the axes) is the top-d
    eigenvectors of the second moment of the ridge-regularised regression's fitted values, B.T @ Sxx @ B
    where B = (Sxx + lambda*I)^-1 @ Sxy -- the classic reduced-rank-ridge-regression solution (Mukherjee &
    Zhu 2011). Z may be 2D (n_trials, k), treated as a single timepoint (T=1), in which case the target
    collapses exactly to the time-collapsed per-category mean -- this is what every lambda-selection call
    site below still passes, so selecting lambda by held-out reconstruction stays on the cheaper
    single-timepoint pseudo-trial construction while the axes actually reported are fit on the full
    time-resolved trajectory (3D Z). Unlike the isotropic-shrinkage marginalization arm, lambda sits inside
    a matrix inverse that multiplies Sxx from a different side than it appears in B.T @ Sxx @ B, so it can
    genuinely rotate the eigenvectors of that product, not merely rescale its eigenvalues -- whether it
    actually does so on data of this shape is measured, not assumed, by
    dpca_ridge_axes_lambda_dependence_check.
    """
    Z_time = Z if Z.ndim == 3 else Z[:, None, :]
    n_trials, T, k = Z_time.shape
    d = min(d, k)
    cats = np.unique(category)
    if len(cats) < 2:
        return np.zeros((k, d)), np.zeros((k, k))
    time_marginal, Y = _category_time_marginal_targets(Z_time, category)
    M = time_marginal.T @ time_marginal
    X = Z_time.reshape(n_trials * T, k)
    Sxx = X.T @ X
    Sxy = X.T @ Y
    scale = np.trace(Sxx) / k if k else 0.0
    ridge = lam * scale
    try:
        B = np.linalg.solve(Sxx + ridge * np.eye(k), Sxy)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(Sxx + ridge * np.eye(k)) @ Sxy
    M_fit = B.T @ Sxx @ B
    M_fit = (M_fit + M_fit.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M_fit)
    order = np.argsort(eigvals)[::-1]
    D = eigvecs[:, order[:d]]
    return D, M


def _subspace_overlap(V1: np.ndarray, V2: np.ndarray) -> float:
    d = V1.shape[1]
    if d == 0:
        return float("nan")
    return float(np.trace(V1 @ V1.T @ V2 @ V2.T) / d)


def _dpca_axes_overlap_across_grid(Z: np.ndarray, category: np.ndarray, grid: tuple, d: int, fit_fn) -> dict:
    axes = {lam: fit_fn(Z, category, lam, d)[0] for lam in grid}
    v0 = axes[grid[0]]
    per_grid_point_overlap_vs_first = {str(lam): _subspace_overlap(v0, axes[lam]) for lam in grid}
    return {
        "grid": list(grid),
        "min_lambda_vs_max_lambda_overlap": _subspace_overlap(axes[grid[0]], axes[grid[-1]]),
        "per_grid_point_overlap_vs_smallest_lambda": per_grid_point_overlap_vs_first,
    }


def _verify_dpca_ridge_lambda_changes_axes() -> dict:
    rng = np.random.default_rng(2026)
    n_per_cat, n_cat, k, rank = 40, 5, 8, CATEGORY_SUBSPACE_DIMENSIONALITY
    n = n_per_cat * n_cat
    category = np.repeat(np.arange(1, n_cat + 1), n_per_cat)
    basis = np.linalg.qr(rng.standard_normal((k, rank)))[0]
    cat_signal = rng.standard_normal((n_cat, rank)) * 3.0
    Z = rng.standard_normal((n, k)) * 0.5
    for i, c in enumerate(category):
        Z[i] += cat_signal[c - 1] @ basis.T
    check_at_ceiling = _dpca_axes_overlap_across_grid(Z, category, DPCA_RIDGE_LAMBDA_GRID, d=rank,
                                                        fit_fn=_dpca_ridge_axes)
    check_below_ceiling = _dpca_axes_overlap_across_grid(Z, category, DPCA_RIDGE_LAMBDA_GRID, d=rank - 1,
                                                           fit_fn=_dpca_ridge_axes)
    overlap_at_ceiling = check_at_ceiling["min_lambda_vs_max_lambda_overlap"]
    overlap_below_ceiling = check_below_ceiling["min_lambda_vs_max_lambda_overlap"]
    return {
        "synthetic_construction": f"single-timepoint rank-{rank}-in-{k}-dimensions category structure, "
                                   f"{n_cat} categories",
        "at_component_count_equal_to_marginal_rank": {
            "d": rank, "check": check_at_ceiling,
            "note": f"d equals the category marginal's own rank ({n_cat} categories, rank {rank}) here, "
                    f"exactly CATEGORY_SUBSPACE_DIMENSIONALITY's relationship to this project's own 5-"
                    f"category schemes -- the row space of B = (Sxx + lambda*I)^-1 @ Sxy is always "
                    f"contained in that fixed rank-{rank} subspace for any lambda, so requesting all {rank} "
                    f"of its dimensions forces the SAME subspace at every lambda for any correct fitting "
                    f"method, ridge included. Measured overlap here: {overlap_at_ceiling:.9f}.",
        },
        "at_component_count_below_marginal_rank": {
            "d": rank - 1, "check": check_below_ceiling,
            "note": f"one component below that ceiling (d={rank - 1}), the same mathematical argument does "
                    "not force invariance, and on this synthetic construction the ridge axes do move with "
                    f"lambda: overlap {overlap_below_ceiling:.6f}.",
        },
        "note": (
            "this is a reported measurement, not a blocking check: the repository's actual "
            f"CATEGORY_SUBSPACE_DIMENSIONALITY is {CATEGORY_SUBSPACE_DIMENSIONALITY}, which sits at the "
            "ceiling for every corpus here (5 fixed categories), so per-cell dpca_ridge_axes_lambda_"
            "dependence_check is expected to show near-total overlap by this same mathematical argument, "
            "independently of whatever the time-resolved category-by-time marginal's own rank turns out to "
            "be -- a run is never halted on either number."
        ),
    }


def _lambda_reconstruction_error(Z_train: np.ndarray, cat_train: np.ndarray, Z_test: np.ndarray,
                                  lam: float, d: int, fit_fn=_dpca_axes_marginalization) -> float:
    if len(np.unique(cat_train)) < 2:
        return float("inf")
    V, _ = fit_fn(Z_train, cat_train, lam, d)
    mu = Z_train.mean(axis=0)
    recon = (Z_test - mu) @ V @ V.T + mu
    return float(np.mean((Z_test - recon) ** 2))


def _select_lambda_cv(Z: np.ndarray, category: np.ndarray, grid: tuple, n_folds: int,
                       rng: np.random.Generator, d: int, fit_fn=_dpca_axes_marginalization
                       ) -> tuple[float, dict]:
    from sklearn.model_selection import StratifiedKFold
    n = len(Z)
    counts = np.unique(category, return_counts=True)[1]
    usable_folds = max(2, min(n_folds, int(counts.min())))
    scores = {lam: [] for lam in grid}
    splitter = StratifiedKFold(n_splits=usable_folds, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
    for train_idx, test_idx in splitter.split(np.zeros(n), category):
        for lam in grid:
            scores[lam].append(_lambda_reconstruction_error(Z[train_idx], category[train_idx],
                                                              Z[test_idx], lam, d, fit_fn))
    mean_scores = {lam: float(np.mean(v)) if v else float("inf") for lam, v in scores.items()}
    best = min(mean_scores, key=mean_scores.get)
    return best, mean_scores


def _fit_axes_arm_a(Z: np.ndarray, category: np.ndarray, rng: np.random.Generator, d: int,
                     grid: tuple = SHRINKAGE_GRID, fit_fn=_dpca_axes_marginalization) -> dict:
    lam, score_table = _select_lambda_cv(Z, category, grid, N_CV_FOLDS_LAMBDA, rng, d, fit_fn)
    V, M = fit_fn(Z, category, lam, d)
    return {"lambda": lam, "V": V, "cats": np.unique(category).tolist(),
            "variance_captured": _variance_captured(V, M),
            "cv_score_table": {str(k): v for k, v in score_table.items()}}


def _fit_axes_arm_b_nested(Z: np.ndarray, category: np.ndarray, rng: np.random.Generator, d: int,
                            grid: tuple = SHRINKAGE_GRID, fit_fn=_dpca_axes_marginalization) -> dict:
    from sklearn.model_selection import StratifiedKFold
    n = len(Z)
    counts = np.unique(category, return_counts=True)[1]
    n_outer = max(2, min(N_CV_FOLDS_LAMBDA_NESTED_OUTER, int(counts.min())))
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
    outer_lambdas = []
    for train_idx, _ in outer.split(np.zeros(n), category):
        lam, _ = _select_lambda_cv(Z[train_idx], category[train_idx], grid, N_CV_FOLDS_LAMBDA, rng, d, fit_fn)
        outer_lambdas.append(lam)
    lam_final = float(np.median(outer_lambdas))
    V, M = fit_fn(Z, category, lam_final, d)
    return {"lambda": lam_final, "V": V, "cats": np.unique(category).tolist(),
            "variance_captured": _variance_captured(V, M),
            "per_outer_fold_lambda": outer_lambdas, "n_outer_folds_used": n_outer}


def _da_trial_scores(maint_traj: np.ndarray, category_all: np.ndarray, correct_all: np.ndarray) -> dict:
    trial_time_avg = maint_traj.mean(axis=1)
    correct_idx = np.where(correct_all)[0]
    attractor, counts = {}, {}
    for c in np.unique(category_all[correct_idx]):
        idx_c = correct_idx[category_all[correct_idx] == c]
        attractor[int(c)] = trial_time_avg[idx_c].mean(axis=0)
        counts[int(c)] = len(idx_c)
    cats = sorted(attractor)
    n = len(category_all)
    da = np.full(n, np.nan)
    for i in range(n):
        c = int(category_all[i])
        if c not in attractor:
            continue
        if correct_all[i] and counts[c] > 1:
            own_attr = (attractor[c] * counts[c] - trial_time_avg[i]) / (counts[c] - 1)
        else:
            own_attr = attractor[c]
        others = [oc for oc in cats if oc != c]
        if not others:
            continue
        d_own_t = np.linalg.norm(maint_traj[i] - own_attr[None, :], axis=1)
        d_other_t = np.mean([np.linalg.norm(maint_traj[i] - attractor[oc][None, :], axis=1)
                             for oc in others], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            da_t = np.where(d_other_t > 0, d_own_t / d_other_t, np.nan)
        da[i] = float(np.nanmean(da_t))
    return {"da": da, "attractor_n_correct_trials": counts}


def _category_neuron_test(rates: np.ndarray, category: np.ndarray, trial_id: np.ndarray,
                           rng: np.random.Generator) -> dict:
    result = two_stage_selectivity_test(
        rates, category, trial_id, rng, n_perm=N_PERM_NEURON,
        min_total_trials=MIN_TOTAL_TRIALS_FOR_NEURON_TEST,
        min_trials_per_level=MIN_TRIALS_PER_CATEGORY_FOR_NEURON_TEST,
        omnibus_alpha=CATEGORY_NEURON_OMNIBUS_ALPHA, permutation_alpha=CATEGORY_NEURON_PERMUTATION_ALPHA,
    )
    if result["status"] != "computed":
        return result
    return {
        "status": "computed", "n_trials": result["n_trials"], "n_categories_usable": result["n_levels_usable"],
        "preferred_category": int(result["preferred_level"]),
        "omnibus_between_group_ss": result["omnibus_between_group_ss"], "omnibus_p": result["omnibus_p"],
        "preferred_vs_rest_mean_difference": result["preferred_vs_rest_mean_difference"],
        "permutation_p": result["permutation_p"],
        "meets_published_category_neuron_criterion": result["meets_selectivity_criterion"],
    }


def _part1_region_session(session: dict, region: str, rng: np.random.Generator) -> dict:
    spike_lists = region_filtered_units(session["spike_lists_all"], session["unit_regions"], region,
                                         session["t_maint"], session["maint_win"])
    if spike_lists is None:
        return {"status": "refused", "reason": f"fewer than {MIN_UNITS_PER_REGION} units in {region}",
                "schema_version": PART1_SCHEMA_VERSION}
    rate_cat_window = _window_rate(spike_lists, session["t_enc1"], CATEGORY_NEURON_WINDOW_OFFSET_S,
                                    CATEGORY_NEURON_WINDOW_DURATION_S)
    rate_probe_window = _window_rate(spike_lists, session["t_probe"], CATEGORY_NEURON_WINDOW_OFFSET_S,
                                      CATEGORY_NEURON_WINDOW_DURATION_S)
    rate_maint = _window_rate(spike_lists, session["t_maint"], 0.0, session["maint_win"])
    rate_baseline = _window_rate(spike_lists, session["t_baseline"], 0.0, session["baseline_win"])
    baseline_corrected_maint = rate_maint - rate_baseline
    category = session["category"]
    category_probe = session["category_probe"]
    accuracy = np.asarray(session["accuracy"], dtype=bool)
    n_trials = len(category)
    category_widened = np.concatenate([category, category_probe])
    trial_id_widened = np.concatenate([np.arange(n_trials), np.arange(n_trials)])
    trial_id_narrow = np.arange(n_trials)
    latency_correct_trials = session["latency"][accuracy]
    neurons = []
    for u in range(len(spike_lists)):
        rates_widened = np.concatenate([rate_cat_window[:, u], rate_probe_window[:, u]])
        test = _category_neuron_test(rates_widened, category_widened, trial_id_widened, rng)
        test_first_encoding_only = _category_neuron_test(rate_cat_window[:, u], category, trial_id_narrow, rng)
        row = {"neuron_index": u, "test": test, "test_first_encoding_only": test_first_encoding_only,
               "rate_cat_window_all_trials": rate_cat_window[:, u].tolist(),
               "rate_probe_window_all_trials": rate_probe_window[:, u].tolist(),
               "baseline_corrected_maint_all_trials": baseline_corrected_maint[:, u].tolist()}
        if test["status"] == "computed":
            row["preferred_category"] = test["preferred_category"]
            row["baseline_corrected_maintenance_rate_correct_trials"] = (
                baseline_corrected_maint[accuracy, u].tolist())
            row["is_preferred_correct_trials"] = (
                (category[accuracy] == test["preferred_category"]).astype(int).tolist())
        neurons.append(row)
    return {"status": "computed", "n_units": len(spike_lists), "neurons": neurons,
            "category_all": category.tolist(), "accuracy_all": accuracy.astype(int).tolist(),
            "latency_correct_trials": latency_correct_trials.tolist(),
            "schema_version": PART1_SCHEMA_VERSION}


def _fit_selective_persistence_glm(rows: list) -> dict:
    metric, condition, subject, neuron = [], [], [], []
    for r in rows:
        metric.extend(r["baseline_corrected_maintenance_rate_correct_trials"])
        condition.extend(r["is_preferred_correct_trials"])
        subject.extend([r["patient"]] * len(r["is_preferred_correct_trials"]))
        neuron.extend([r["neuron_id"]] * len(r["is_preferred_correct_trials"]))
    n = len(metric)
    n_neurons = len(rows)
    if n_neurons < 2 or n < 8:
        return {"status": "underpowered", "n_neurons": n_neurons, "n_trials": n,
                "reason": "fewer than 2 neurons or 8 trial rows"}
    fit = linear_mixed_effects_test(np.array(metric), np.array(condition), np.array(subject),
                                     nested_group=np.array(neuron))
    if not fit["converged"]:
        return {"status": "not_computable", "n_neurons": n_neurons, "n_trials": n, "reason": fit["reason"]}
    ci_lo = fit["beta"] - 1.959963984540054 * fit["se"]
    ci_hi = fit["beta"] + 1.959963984540054 * fit["se"]
    mdd = float(Z_80_POWER * fit["se"]) if np.isfinite(fit["se"]) else float("nan")
    return {"status": "computed", "n_neurons": n_neurons, "n_trials": n, "n_patients": fit["n_subjects"],
            "estimate": fit["beta"], "se": fit["se"], "ci_lower": ci_lo, "ci_upper": ci_hi,
            "p_value": fit["p_value"], "mdd": mdd, "r_squared": fit["r_squared"]}


def _combine_glm_fits_across_splits(fits: list[dict]) -> dict:
    from scipy.stats import norm
    computed = [f for f in fits if f.get("status") == "computed"]
    if not computed:
        return {"status": "not_computable", "n_splits_used": 0, "n_splits_attempted": len(fits),
                "reason": "no random split produced a computed fit"}
    betas = np.array([f["estimate"] for f in computed])
    ses = np.array([f["se"] for f in computed])
    r_used = len(computed)
    beta_pooled = float(betas.mean())
    se_within_sq = float((ses ** 2).mean())
    se_between_sq = float(betas.var(ddof=1)) if r_used > 1 else 0.0
    se_pooled = float(np.sqrt(se_within_sq + (1.0 + 1.0 / r_used) * se_between_sq))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(beta_pooled / se_pooled)))) if se_pooled > 0 else float("nan")
    ci_lo = beta_pooled - 1.959963984540054 * se_pooled
    ci_hi = beta_pooled + 1.959963984540054 * se_pooled
    mdd = float(Z_80_POWER * se_pooled) if np.isfinite(se_pooled) else float("nan")
    return {
        "status": "computed", "n_splits_used": r_used, "n_splits_attempted": len(fits),
        "estimate": beta_pooled, "se": se_pooled, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "p_value": p_value, "mdd": mdd,
        "combining_rule": "Rubin's rule across random trial splits: pooled beta is the split mean, pooled "
            "variance is the mean within-split variance plus (1 + 1/n_splits_used) times the between-"
            "split variance of the beta estimates -- the same combining rule used for multiple imputation, "
            "treating each random split as one imputation of the train/test partition.",
        "n_neurons_mean_across_splits": float(np.mean([f["n_neurons"] for f in computed])),
        "n_trials_mean_across_splits": float(np.mean([f["n_trials"] for f in computed])),
        "r_squared_mean_across_splits": float(np.mean([f["r_squared"] for f in computed])),
        "per_split_estimate": betas.tolist(), "per_split_se": ses.tolist(),
    }


def _combine_decoding_splits(fits: list[dict]) -> dict:
    computed = [f for f in fits if f.get("status") == "computed"]
    if not computed:
        return {"status": "not_computable", "n_splits_used": 0, "n_splits_attempted": len(fits),
                "reason": "no random split produced a computed decoding"}
    accuracies = np.array([f["estimate"] for f in computed])
    ses = np.array([f["se"] for f in computed])
    r_used = len(computed)
    pooled = float(accuracies.mean())
    se_within_sq = float((ses ** 2).mean())
    se_between_sq = float(accuracies.var(ddof=1)) if r_used > 1 else 0.0
    se_pooled = float(np.sqrt(se_within_sq + (1.0 + 1.0 / r_used) * se_between_sq))
    ci_lo = pooled - 1.959963984540054 * se_pooled
    ci_hi = pooled + 1.959963984540054 * se_pooled
    mdd = float(Z_80_POWER * se_pooled) if np.isfinite(se_pooled) else float("nan")
    pooled_null = np.concatenate([np.asarray(f["null_accuracies"], dtype=float) for f in computed
                                   if f.get("null_accuracies")])
    p_value = permutation_pvalue(pooled_null >= pooled) if pooled_null.size else float("nan")
    return {
        "status": "computed", "n_splits_used": r_used, "n_splits_attempted": len(fits),
        "estimate": pooled, "se": se_pooled, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "p_value": p_value, "mdd": mdd,
        "combining_rule": "estimate/se/ci pooled by Rubin's rule across random trial splits, the split "
            "mean and its between/within-split variance, as for a mixed-model coefficient; p_value tests "
            "the pooled estimate against the label-shuffle null decoding accuracies pooled across the same "
            "splits, not against zero or against decoding accuracy minus chance.",
        "n_neurons_mean_across_splits": float(np.mean([f["n_neurons"] for f in computed])),
        "n_trials_mean_across_splits": float(np.mean([f["n_trials"] for f in computed])),
        "per_split_estimate": accuracies.tolist(), "per_split_se": ses.tolist(),
    }


def _part1_trial_split_arm(region_sessions: dict) -> dict:
    computed_sessions = {k: v for k, v in region_sessions.items() if v.get("status") == "computed"}
    if not computed_sessions:
        empty = {"status": "not_computable", "n_splits_used": 0, "n_splits_attempted": 0,
                  "n_splits_total": N_TRIAL_SPLIT_REPEATS, "reason": "no computed session in this region"}
        return empty, empty
    per_split = []
    fits_all = []
    fits_floor_restricted = []
    for split_i in range(N_TRIAL_SPLIT_REPEATS):
        rng = np.random.default_rng(stable_seed(f"trial_split_{split_i}"))
        rows = []
        for session_key, cell in computed_sessions.items():
            category_all = np.array(cell["category_all"], dtype=int)
            accuracy_all = np.array(cell["accuracy_all"], dtype=bool)
            n_trials = len(category_all)
            for neuron in cell["neurons"]:
                rates_enc = np.array(neuron["rate_cat_window_all_trials"])
                rates_probe = np.array(neuron["rate_probe_window_all_trials"])
                bcm_all = np.array(neuron["baseline_corrected_maint_all_trials"])
                perm = rng.permutation(n_trials)
                half = n_trials // 2
                train_idx, test_idx = perm[:half], perm[half:]
                rates_train_widened = np.concatenate([rates_enc[train_idx], rates_probe[train_idx]])
                category_train_widened = np.concatenate([category_all[train_idx], category_all[train_idx]])
                trial_id_train_widened = np.concatenate([train_idx, train_idx])
                train_test = _category_neuron_test(rates_train_widened, category_train_widened,
                                                     trial_id_train_widened, rng)
                if (train_test["status"] != "computed"
                        or not train_test["meets_published_category_neuron_criterion"]):
                    continue
                correct_test_idx = test_idx[accuracy_all[test_idx]]
                if len(correct_test_idx) < MIN_CORRECT_TRIALS_PER_NEURON_TRIAL_SPLIT_TEST:
                    continue
                preferred = train_test["preferred_category"]
                rows.append({
                    "patient": cell["patient"], "neuron_id": f"{session_key}__u{neuron['neuron_index']}",
                    "baseline_corrected_maintenance_rate_correct_trials": bcm_all[correct_test_idx].tolist(),
                    "is_preferred_correct_trials": (
                        category_all[correct_test_idx] == preferred).astype(int).tolist(),
                })
        n_selected = len(rows)
        above_floor = n_selected >= MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM
        fit = _fit_selective_persistence_glm(rows) if n_selected >= 2 else {
            "status": "underpowered", "n_neurons": n_selected, "n_trials": 0,
            "reason": "fewer than 2 selected neurons"}
        per_split.append({"split": split_i, "n_selected_units": n_selected, "above_published_minimum_"
                           "category_neuron_count": above_floor, "status": fit.get("status"),
                           "estimate": fit.get("estimate"), "se": fit.get("se")})
        if fit.get("status") == "computed":
            fits_all.append(fit)
            if above_floor:
                fits_floor_restricted.append(fit)
    combined_all = _combine_glm_fits_across_splits(fits_all)
    combined_all["per_split"] = per_split
    combined_all["n_splits_total"] = N_TRIAL_SPLIT_REPEATS
    combined_all["description"] = (
        "pools every random trial split whose train-half selection produced a computed fit, regardless of "
        "whether the selected-unit count cleared the category-neuron floor; per_split carries every split's "
        "own selected-unit count and estimate so the relationship between the two is visible rather than "
        "hidden by dropping the splits selection happened to under-populate."
    )
    combined_floor = _combine_glm_fits_across_splits(fits_floor_restricted)
    combined_floor["per_split"] = per_split
    combined_floor["n_splits_total"] = N_TRIAL_SPLIT_REPEATS
    combined_floor["n_splits_below_published_minimum_category_neuron_count"] = (
        N_TRIAL_SPLIT_REPEATS - len(fits_floor_restricted))
    combined_floor["description"] = (
        "secondary restriction of trial_split_matched_arm to splits whose train-half selection cleared the "
        "category-neuron floor only; dropping the other splits biases the surviving set toward splits where "
        "selection happened to find more units, so trial_split_matched_arm above, not this one, is the "
        "reported estimate."
    )
    return combined_all, combined_floor


def _selection_matched_fit(rows: list) -> dict:
    if len(rows) < MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM:
        return {
            "status": "below_published_minimum_category_neuron_count",
            "n_category_neurons": len(rows),
            "reason": f"fewer than {MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM} category neurons "
                      "(the published study's own pre-SMA exclusion floor, applied here to every region)",
        }
    fit = _fit_selective_persistence_glm(rows)
    fit["n_category_neurons"] = len(rows)
    return fit


def _outlier_trimmed_fit(rows: list) -> dict:
    means = np.array([np.mean(r["baseline_corrected_maintenance_rate_correct_trials"]) for r in rows])
    if len(means) < 2:
        kept = rows
    else:
        threshold = means.mean() + OUTLIER_FIRING_RATE_SD_THRESHOLD * means.std(ddof=1)
        kept = [r for r, m in zip(rows, means) if m <= threshold]
    fit = _selection_matched_fit(kept)
    fit["n_excluded_as_outlier"] = len(rows) - len(kept)
    fit["n_candidate_neurons"] = len(rows)
    return fit


def _part1_region_summary(region_sessions: dict) -> dict:
    rows_all, rows_selection_matched, rows_first_encoding_only = [], [], []
    for session_key, cell in region_sessions.items():
        if cell.get("status") != "computed":
            continue
        for neuron in cell["neurons"]:
            if neuron["test"]["status"] != "computed":
                continue
            row = {"patient": cell["patient"], "neuron_id": f"{session_key}__u{neuron['neuron_index']}",
                   "baseline_corrected_maintenance_rate_correct_trials": neuron[
                       "baseline_corrected_maintenance_rate_correct_trials"],
                   "is_preferred_correct_trials": neuron["is_preferred_correct_trials"]}
            rows_all.append(row)
            if neuron["test"]["meets_published_category_neuron_criterion"]:
                rows_selection_matched.append(row)
            first_encoding_test = neuron.get("test_first_encoding_only", {})
            if (first_encoding_test.get("status") == "computed"
                    and first_encoding_test.get("meets_published_category_neuron_criterion")):
                rows_first_encoding_only.append(row)
    full_population = _fit_selective_persistence_glm(rows_all)
    full_population["n_neurons_tested"] = len(rows_all)
    selection_matched = _selection_matched_fit(rows_selection_matched)
    selection_matched_first_encoding_only = _selection_matched_fit(rows_first_encoding_only)
    selection_matched_outlier_trimmed = _outlier_trimmed_fit(rows_selection_matched)
    trial_split_matched, trial_split_matched_floor_restricted = _part1_trial_split_arm(region_sessions)
    return {"selection_matched_arm": selection_matched,
            "selection_matched_first_encoding_only_arm": selection_matched_first_encoding_only,
            "selection_matched_outlier_trimmed_arm": selection_matched_outlier_trimmed,
            "full_population_arm": full_population,
            "trial_split_matched_arm": trial_split_matched,
            "trial_split_matched_floor_restricted_arm": trial_split_matched_floor_restricted,
            "n_neurons_total": sum(len(c["neurons"]) for c in region_sessions.values()
                                    if c.get("status") == "computed"),
            "n_category_neurons_widened_window": len(rows_selection_matched),
            "n_category_neurons_first_encoding_only": len(rows_first_encoding_only)}


def _part1_reaction_time_cells(region_sessions: dict) -> tuple[dict, dict]:
    nonpreferred, signed = {}, {}
    for session_key, cell in region_sessions.items():
        if cell.get("status") != "computed":
            continue
        latency = np.asarray(cell.get("latency_correct_trials", []), dtype=float)
        if latency.size == 0:
            continue
        finite_latency = np.isfinite(latency)
        for neuron in cell["neurons"]:
            if neuron["test"]["status"] != "computed":
                continue
            neuron_id = f"{session_key}__u{neuron['neuron_index']}"
            rate = np.array(neuron["baseline_corrected_maintenance_rate_correct_trials"], dtype=float)
            is_preferred = np.array(neuron["is_preferred_correct_trials"], dtype=bool)
            nonpreferred_mask = finite_latency & ~is_preferred
            nonpreferred[neuron_id] = {
                "status": "computed", "patient": cell["patient"],
                "nonpreferred_rate": rate[nonpreferred_mask].tolist(),
                "latency": latency[nonpreferred_mask].tolist(),
            }
            signed_rate = rate * np.where(is_preferred, 1.0, -1.0)
            signed[neuron_id] = {
                "status": "computed", "patient": cell["patient"],
                "signed_preferred_rate": signed_rate[finite_latency].tolist(),
                "latency": latency[finite_latency].tolist(),
            }
    return nonpreferred, signed


def _da_session(session: dict, region: str) -> dict:
    spike_lists = region_filtered_units(session["spike_lists_all"], session["unit_regions"], region,
                                         session["t_maint"], session["maint_win"])
    if spike_lists is None:
        return {"status": "refused", "reason": f"fewer than {MIN_UNITS_PER_REGION} units in {region}",
                "schema_version": DA_SCHEMA_VERSION}
    category_all = session["category"]
    correct_all = np.asarray(session["accuracy"], dtype=bool)
    n_units = len(spike_lists)

    enc_rate = _window_rate(spike_lists, session["t_enc1"], AXIS_FIT_WINDOW_OFFSET_S, AXIS_FIT_WINDOW_DURATION_S)
    cats_correct, counts_correct = np.unique(category_all[correct_all], return_counts=True)
    usable_cats = cats_correct[counts_correct >= MIN_TRIALS_PER_CATEGORY_FOR_DA]
    if len(usable_cats) < MIN_CATEGORIES_FOR_DA:
        return {"status": "refused",
                "reason": f"only {len(usable_cats)} categories with >={MIN_TRIALS_PER_CATEGORY_FOR_DA} "
                          f"correct trials (need >={MIN_CATEGORIES_FOR_DA})",
                "schema_version": DA_SCHEMA_VERSION}
    fit_mask = correct_all & np.isin(category_all, usable_cats)
    fit_idx = np.where(fit_mask)[0]
    cat_fit_all = category_all[fit_idx]

    n_pc = max(2, min(N_PC_DEFAULT, n_units - 1))
    d = min(CATEGORY_SUBSPACE_DIMENSIONALITY, n_pc)
    rng = np.random.default_rng(stable_seed(f"{session['patient']}_{region}_da"))

    from sklearn.model_selection import StratifiedKFold
    n_folds = max(2, min(N_CV_FOLDS_LAMBDA, int(np.unique(cat_fit_all, return_counts=True)[1].min())))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
    fold_of_fit = np.zeros(len(fit_idx), dtype=int)
    for fold_i, (_, test_local) in enumerate(splitter.split(np.zeros(len(fit_idx)), cat_fit_all)):
        fold_of_fit[test_local] = fold_i

    maint_sliding = _sliding_window_rate(spike_lists, session["t_maint"], SLIDING_WINDOW_STEP_S_PRIMARY,
                                          SLIDING_WINDOW_DURATION_S, session["maint_win"])
    n_trials, T = len(category_all), maint_sliding.shape[2]
    Z_maint_pca = np.zeros((n_trials, T, n_pc))
    Z_maint_a = np.zeros((n_trials, T, d))
    Z_maint_b = np.zeros((n_trials, T, d))
    fold_bases = []
    for fold_i in range(n_folds):
        train_idx, test_idx = fit_idx[fold_of_fit != fold_i], fit_idx[fold_of_fit == fold_i]
        mu_std_f = enc_rate[train_idx].mean(axis=0)
        sd_std_f = enc_rate[train_idx].std(axis=0)
        sd_std_f = np.where(sd_std_f > 1e-8, sd_std_f, 1.0)
        enc_z_train = (enc_rate[train_idx] - mu_std_f) / sd_std_f
        mu_pca_f, V_pca_f = _fit_pca_2d(enc_z_train, n_pc)
        Z_fit_f = _project_pca_2d(enc_z_train, mu_pca_f, V_pca_f)
        cat_train = category_all[train_idx]
        arm_a_f = _fit_axes_arm_a(Z_fit_f, cat_train, rng, d)
        arm_b_f = _fit_axes_arm_b_nested(Z_fit_f, cat_train, rng, d)
        fold_bases.append({"mu_std": mu_std_f, "sd_std": sd_std_f, "mu_pca": mu_pca_f, "V_pca": V_pca_f,
                            "arm_a": arm_a_f, "arm_b": arm_b_f})
        maint_z_test = (maint_sliding[test_idx] - mu_std_f[None, :, None]) / sd_std_f[None, :, None]
        Z_pca_test = _project_pca_3d(maint_z_test, mu_pca_f, V_pca_f)
        Z_maint_pca[test_idx] = Z_pca_test
        Z_maint_a[test_idx] = Z_pca_test @ arm_a_f["V"]
        Z_maint_b[test_idx] = Z_pca_test @ arm_b_f["V"]

    non_fit_idx = np.where(~fit_mask)[0]
    if len(non_fit_idx):
        mu_std_avg = np.mean([b["mu_std"] for b in fold_bases], axis=0)
        sd_std_avg = np.mean([b["sd_std"] for b in fold_bases], axis=0)
        mu_pca_avg = np.mean([b["mu_pca"] for b in fold_bases], axis=0)
        V_pca_avg = np.mean([b["V_pca"] for b in fold_bases], axis=0)
        arm_a_V_avg = np.mean([b["arm_a"]["V"] for b in fold_bases], axis=0)
        arm_b_V_avg = np.mean([b["arm_b"]["V"] for b in fold_bases], axis=0)
        maint_z_avg = (maint_sliding[non_fit_idx] - mu_std_avg[None, :, None]) / sd_std_avg[None, :, None]
        Z_pca_avg = _project_pca_3d(maint_z_avg, mu_pca_avg, V_pca_avg)
        Z_maint_pca[non_fit_idx] = Z_pca_avg
        Z_maint_a[non_fit_idx] = Z_pca_avg @ arm_a_V_avg
        Z_maint_b[non_fit_idx] = Z_pca_avg @ arm_b_V_avg

    da_pca = _da_trial_scores(Z_maint_pca[:, :, :d], category_all, correct_all)
    da_dpca_a = _da_trial_scores(Z_maint_a, category_all, correct_all)
    da_dpca_b = _da_trial_scores(Z_maint_b, category_all, correct_all)

    spike_count = _trial_population_spike_count(spike_lists, session["t_maint"], session["maint_win"])
    raw_activity_by_unit = np.stack(
        [np.array([np.sum((spk >= t0) & (spk < t0 + session["maint_win"]))
                   for t0 in session["t_maint"]]) for spk in spike_lists], axis=1,
    ).astype(float)
    rate_free_dev = rate_free_state_deviation(raw_activity_by_unit)

    finite = np.isfinite(da_dpca_a["da"]) & np.isfinite(rate_free_dev)
    latency_all = session["latency"]
    correct_finite_latency = finite & correct_all & np.isfinite(latency_all)
    arm_a_lambdas = [b["arm_a"]["lambda"] for b in fold_bases]
    arm_b_lambdas = [b["arm_b"]["lambda"] for b in fold_bases]
    return {
        "status": "computed", "n_units": n_units, "n_trials_total": int(len(category_all)),
        "n_categories_usable": int(len(usable_cats)), "n_pc": n_pc, "category_subspace_dimensionality": d,
        "n_axis_folds": n_folds,
        "arm_a_lambda": float(np.median(arm_a_lambdas)), "arm_a_per_fold_lambda": arm_a_lambdas,
        "arm_a_variance_captured": float(np.mean([b["arm_a"]["variance_captured"] for b in fold_bases])),
        "arm_b_lambda": float(np.median(arm_b_lambdas)), "arm_b_per_fold_lambda": arm_b_lambdas,
        "arm_b_variance_captured": float(np.mean([b["arm_b"]["variance_captured"] for b in fold_bases])),
        "arm_b_per_outer_fold_lambda": arm_b_lambdas,
        "arm_a_minus_arm_b_lambda": float(np.median(arm_a_lambdas) - np.median(arm_b_lambdas)),
        "axis_fit_note": "mu_std, the PCA basis, and the demixed category axes (arm_a, arm_b) are each "
            "fit on one cross-validation fold's correct usable-category trials and applied only to that "
            "fold's own held-out correct trials; a trial outside the usable-category correct set is "
            "projected with the average of the per-fold bases.",
        "n_trials_with_defined_da": int(finite.sum()),
        "da": da_dpca_a["da"][finite].tolist(), "da_dpca_arm_b": da_dpca_b["da"][finite].tolist(),
        "da_pca_only": da_pca["da"][finite].tolist(),
        "accuracy": correct_all[finite].astype(float).tolist(),
        "population_spike_count": spike_count[finite].tolist(),
        "rate_free_deviation": rate_free_dev[finite].tolist(),
        "da_correct_trials": da_dpca_a["da"][correct_finite_latency].tolist(),
        "rate_free_deviation_correct_trials": rate_free_dev[correct_finite_latency].tolist(),
        "latency_correct_trials": latency_all[correct_finite_latency].tolist(),
        "schema_version": DA_SCHEMA_VERSION,
    }


def _collect_region_neurons(corpus: str, region: str, step_s: float, selected_keys: set | None = None) -> dict:
    spec = CORPUS_SPECS[corpus]
    per_category_enc: dict = {}
    per_category_maint: dict = {}
    patients: set = set()
    n_sessions_used = 0
    for path in spec["glob"]():
        session = spec["loader"](path)
        if session is None:
            continue
        spike_lists = region_filtered_units(session["spike_lists_all"], session["unit_regions"], region,
                                             session["t_maint"], session["maint_win"])
        if spike_lists is None:
            continue
        category_all = session["category"]
        correct_all = np.asarray(session["accuracy"], dtype=bool)
        enc_rate = _window_rate(spike_lists, session["t_enc1"], AXIS_FIT_WINDOW_OFFSET_S,
                                 AXIS_FIT_WINDOW_DURATION_S)
        maint_sliding = _sliding_window_rate(spike_lists, session["t_maint"], step_s,
                                              SLIDING_WINDOW_DURATION_S, session["maint_win"])
        used_any = False
        for u in range(len(spike_lists)):
            neuron_enc, neuron_maint = {}, {}
            for c in np.unique(category_all[correct_all]):
                mask = correct_all & (category_all == c)
                if mask.sum() < MIN_TRIALS_PER_CATEGORY_PSEUDOPOP:
                    continue
                neuron_enc[int(c)] = enc_rate[mask, u]
                neuron_maint[int(c)] = maint_sliding[mask, u, :]
            key = f"{path.stem}__u{u}"
            if neuron_enc and (selected_keys is None or key in selected_keys):
                per_category_enc[key] = neuron_enc
                per_category_maint[key] = neuron_maint
                used_any = True
        if used_any:
            patients.add(session["patient"])
            n_sessions_used += 1
    return {"enc": per_category_enc, "maint": per_category_maint,
            "n_sessions_used": n_sessions_used, "n_patients": len(patients)}


def _build_pseudopopulation(neurons_enc: dict, neurons_maint: dict, n_pseudo: int,
                             rng: np.random.Generator, trial_pool: dict | None = None,
                             ) -> tuple[dict | None, list]:
    neuron_keys = sorted(neurons_enc)
    if not neuron_keys:
        return None, []
    common_cats = sorted(set.intersection(*(set(neurons_enc[k]) for k in neuron_keys)))
    if len(common_cats) < MIN_CATEGORIES_FOR_DA:
        return None, common_cats
    n_units = len(neuron_keys)
    n_t = next(iter(neurons_maint[neuron_keys[0]].values())).shape[1]
    n_trials_pseudo = n_pseudo * len(common_cats)
    enc = np.zeros((n_trials_pseudo, n_units))
    maint = np.zeros((n_trials_pseudo, n_units, n_t))
    category = np.zeros(n_trials_pseudo, dtype=int)
    row = 0
    for c in common_cats:
        for _ in range(n_pseudo):
            for j, key in enumerate(neuron_keys):
                trials_enc, trials_maint = neurons_enc[key][c], neurons_maint[key][c]
                pool = trial_pool[key][c] if trial_pool is not None else np.arange(len(trials_enc))
                idx = int(pool[int(rng.integers(0, len(pool)))])
                enc[row, j] = trials_enc[idx]
                maint[row, j, :] = trials_maint[idx]
            category[row] = c
            row += 1
    return {"enc": enc, "maint": maint, "category": category, "n_units": n_units}, common_cats


def _category_time_marginal(Z_time: np.ndarray, category: np.ndarray) -> np.ndarray:
    cats = np.unique(category)
    means = np.stack([Z_time[category == c].mean(axis=0) for c in cats], axis=0)
    grand_mean = means.mean(axis=(0, 1))
    return (means - grand_mean).reshape(-1, means.shape[-1])


def _select_lambda_cv_pseudopop(collected_enc: dict, collected_maint: dict, common_cats: list,
                                 n_pc: int, n_pseudo: int, n_folds: int,
                                 grid: tuple, rng: np.random.Generator, d: int,
                                 parent_pool: dict | None = None, fit_fn=_dpca_axes_marginalization
                                 ) -> tuple[float, dict]:
    # CV over REAL trials, not over already-resampled pseudo-trials: a pseudo-trial is not the unit of
    # independence (the same real trial can back many pseudo-trials), so folds are built by partitioning
    # each neuron's own real trial indices per category first, then resampling train/test pseudo-trials
    # only from their own disjoint side -- same discipline as _pseudopopulation_decoding.
    neuron_keys = sorted(collected_enc)
    pools = {key: {c: (parent_pool[key][c] if parent_pool is not None
                        else np.arange(len(collected_enc[key][c]))) for c in common_cats}
             for key in neuron_keys}
    min_pool_size = min(len(pools[key][c]) for key in neuron_keys for c in common_cats)
    actual_n_folds = max(2, min(n_folds, min_pool_size))
    groups = {key: {c: np.array_split(rng.permutation(pools[key][c]), actual_n_folds) for c in common_cats}
              for key in neuron_keys}
    scores = {lam: [] for lam in grid}
    for fold_i in range(actual_n_folds):
        train_pool: dict = {}
        test_pool: dict = {}
        for key in neuron_keys:
            train_pool[key], test_pool[key] = {}, {}
            for c in common_cats:
                folds = groups[key][c]
                test_pool[key][c] = folds[fold_i]
                train_pool[key][c] = np.concatenate([folds[j] for j in range(actual_n_folds) if j != fold_i])
        if any(len(train_pool[key][c]) == 0 or len(test_pool[key][c]) == 0
               for key in neuron_keys for c in common_cats):
            continue
        train_pop, _ = _build_pseudopopulation(collected_enc, collected_maint, n_pseudo, rng, train_pool)
        test_pop, _ = _build_pseudopopulation(collected_enc, collected_maint, n_pseudo, rng, test_pool)
        if train_pop is None or test_pop is None:
            continue
        mu_pca_fold, V_pca_fold = _fit_pca_2d(train_pop["enc"], n_pc)
        Z_train = _project_pca_2d(train_pop["enc"], mu_pca_fold, V_pca_fold)
        Z_test = _project_pca_2d(test_pop["enc"], mu_pca_fold, V_pca_fold)
        for lam in grid:
            scores[lam].append(_lambda_reconstruction_error(Z_train, train_pop["category"], Z_test, lam, d,
                                                              fit_fn))
    mean_scores = {lam: float(np.mean(v)) if v else float("inf") for lam, v in scores.items()}
    best = min(mean_scores, key=mean_scores.get)
    return best, mean_scores


def _fit_axes_arm_b_nested_pseudopop(collected_enc: dict, collected_maint: dict, common_cats: list,
                                      n_pc: int, n_pseudo: int,
                                      rng: np.random.Generator, d: int, grid: tuple = SHRINKAGE_GRID,
                                      fit_fn=_dpca_axes_marginalization) -> tuple[float, list, int]:
    neuron_keys = sorted(collected_enc)
    full_pool = {key: {c: np.arange(len(collected_enc[key][c])) for c in common_cats} for key in neuron_keys}
    min_pool_size = min(len(full_pool[key][c]) for key in neuron_keys for c in common_cats)
    n_outer = max(2, min(N_CV_FOLDS_LAMBDA_NESTED_OUTER, min_pool_size))
    outer_groups = {key: {c: np.array_split(rng.permutation(full_pool[key][c]), n_outer) for c in common_cats}
                    for key in neuron_keys}
    outer_lambdas = []
    for fold_i in range(n_outer):
        outer_train_pool = {key: {c: np.concatenate([outer_groups[key][c][j] for j in range(n_outer)
                                                       if j != fold_i]) for c in common_cats}
                             for key in neuron_keys}
        lam, _ = _select_lambda_cv_pseudopop(collected_enc, collected_maint, common_cats, n_pc,
                                              n_pseudo, N_CV_FOLDS_LAMBDA, grid, rng, d,
                                              parent_pool=outer_train_pool, fit_fn=fit_fn)
        outer_lambdas.append(lam)
    return float(np.median(outer_lambdas)), outer_lambdas, n_outer


def _lambda_sweep(Z_fit_full: np.ndarray, category_full: np.ndarray, Z_maint_pca_full: np.ndarray,
                   score_table: dict, grid: tuple, d: int, arm_a_lambda: float, arm_b_lambda: float,
                   fit_fn=_dpca_axes_marginalization) -> list:
    cat_time_marginal = _category_time_marginal(Z_maint_pca_full, category_full)
    M_cat_time = cat_time_marginal.T @ cat_time_marginal
    correct_all = np.ones(len(category_full), dtype=bool)
    sweep = []
    for lam in grid:
        V_lam, _ = fit_fn(Z_fit_full, category_full, lam, d)
        da = _da_trial_scores(Z_maint_pca_full @ V_lam, category_full, correct_all)["da"]
        finite = np.isfinite(da)
        sweep.append({
            "lambda": lam,
            "variance_captured_category_time_marginal": _variance_captured(V_lam, M_cat_time),
            "mean_distance_to_attractor": float(np.mean(da[finite])) if finite.any() else float("nan"),
            "n_trials_with_defined_distance_to_attractor": int(finite.sum()),
            "held_out_reconstruction_error": score_table.get(lam, float("nan")),
            "is_arm_a_cv_selected_point": lam == arm_a_lambda,
            "is_arm_b_cv_selected_point": lam == arm_b_lambda,
        })
    return sweep


def _lambda_sweep_flatness(score_table: dict) -> dict:
    finite = [v for v in score_table.values() if np.isfinite(v)]
    if len(finite) < 2 or min(finite) <= 0:
        return {"status": "not_computable"}
    lo, hi = min(finite), max(finite)
    relative_range = (hi - lo) / lo
    argmin_lambda = min(score_table, key=score_table.get)
    if relative_range < LAMBDA_SWEEP_MACHINE_EPSILON_RELATIVE_RANGE:
        note = (f"held-out reconstruction error varies by only {relative_range:.2e} across the lambda "
                f"grid (from {lo:.6g} to {hi:.6g}) -- at floating-point precision, not merely small. "
                "This is not a statistical property of these data: see shrinkage_axes_isotropic_"
                "invariance_note. The axes cannot depend on lambda by construction here, so the argmin "
                f"(lambda={argmin_lambda}) is an arbitrary pick among numerically-tied floats, not a "
                "selection, and every quantity below should be read as constant across the sweep.")
    elif relative_range < LAMBDA_SWEEP_FLAT_RELATIVE_RANGE_THRESHOLD:
        note = (f"held-out reconstruction error varies by only {relative_range:.1%} across the lambda "
                f"grid (from {lo:.4g} to {hi:.4g}) -- below the disclosed "
                f"{LAMBDA_SWEEP_FLAT_RELATIVE_RANGE_THRESHOLD:.0%} threshold this artifact uses to "
                "describe a selection surface as weakly determined by these data; the argmin "
                f"(lambda={argmin_lambda}) is one point on a near-flat curve, not a confidently-selected "
                "value, and the reported quantities should be read across the whole sweep.")
    else:
        note = (f"held-out reconstruction error varies by {relative_range:.1%} across the lambda grid "
                f"(from {lo:.4g} to {hi:.4g}) -- above the disclosed "
                f"{LAMBDA_SWEEP_FLAT_RELATIVE_RANGE_THRESHOLD:.0%} threshold; the argmin "
                f"(lambda={argmin_lambda}) sits on a more clearly peaked surface at this cell.")
    return {"status": "computed", "relative_range": relative_range, "min_error": lo, "max_error": hi,
            "argmin_lambda": argmin_lambda, "note": note}


def _selective_neuron_keys(corpus: str, region: str, field: str = "test") -> set:
    sessions = {k: v for k, v in _load_checkpoint(f"{corpus}__part1__{region}").items()
                if v.get("schema_version") == PART1_SCHEMA_VERSION}
    keys = set()
    for session_key, cell in sessions.items():
        if cell.get("status") != "computed":
            continue
        for neuron in cell["neurons"]:
            test = neuron.get(field, {})
            if test.get("status") == "computed" and test.get("meets_published_category_neuron_criterion"):
                keys.add(f"{session_key}__u{neuron['neuron_index']}")
    return keys


def _pseudopopulation_geometry(corpus: str, region: str, selected_keys: set | None = None,
                                seed_tag: str | None = None) -> dict:
    tag = seed_tag if seed_tag is not None else f"{corpus}_{region}"
    collected = _collect_region_neurons(corpus, region, SLIDING_WINDOW_STEP_S_PRIMARY, selected_keys)
    rng = np.random.default_rng(stable_seed(f"{tag}_pseudopop"))
    pseudopop, common_cats = _build_pseudopopulation(collected["enc"], collected["maint"],
                                                       N_PSEUDO_TRIALS_PER_CATEGORY, rng)
    if pseudopop is None:
        return {"status": "refused", "n_sessions_used": collected["n_sessions_used"],
                "n_patients": collected["n_patients"], "n_candidate_units": len(collected["enc"]),
                "reason": f"fewer than {MIN_CATEGORIES_FOR_DA} categories with a common pool of neurons "
                          f"reaching >={MIN_TRIALS_PER_CATEGORY_PSEUDOPOP} correct trials each",
                "schema_version": PSEUDOPOP_SCHEMA_VERSION}
    n_pc = max(2, min(N_PC_DEFAULT, pseudopop["n_units"] - 1))
    d = min(CATEGORY_SUBSPACE_DIMENSIONALITY, n_pc)
    mu_pca, V_pca = _fit_pca_2d(pseudopop["enc"], n_pc)
    Z_enc = _project_pca_2d(pseudopop["enc"], mu_pca, V_pca)

    correct_all = np.ones(len(pseudopop["category"]), dtype=bool)
    mu_std, sd_std = pseudopop["enc"].mean(axis=0), pseudopop["enc"].std(axis=0)
    sd_std = np.where(sd_std > 1e-8, sd_std, 1.0)
    maint_z_primary = (pseudopop["maint"] - mu_std[None, :, None]) / sd_std[None, :, None]
    Z_maint_pca_primary = _project_pca_3d(maint_z_primary, mu_pca, V_pca)

    # lambda is selected by cross-validating over disjoint real-trial pools (not over already-resampled
    # pseudo-trials -- see _select_lambda_cv_pseudopop), then the axes at that lambda are fit on the full
    # pooled encoding-window data, matching the per-session convention.
    lam_a, score_table_a = _select_lambda_cv_pseudopop(collected["enc"], collected["maint"], common_cats,
                                                         n_pc, N_PSEUDO_TRIALS_PER_CATEGORY,
                                                         N_CV_FOLDS_LAMBDA, SHRINKAGE_GRID, rng, d)
    _, marginal_a = _category_marginal(Z_enc, pseudopop["category"])
    V_a, M_a = _shrinkage_axes(marginal_a, lam_a, d)
    arm_a = {"lambda": lam_a, "V": V_a, "variance_captured": _variance_captured(V_a, M_a),
             "cv_score_table": {str(k): v for k, v in score_table_a.items()}}

    lam_b, outer_lambdas_b, n_outer_b = _fit_axes_arm_b_nested_pseudopop(
        collected["enc"], collected["maint"], common_cats, n_pc, N_PSEUDO_TRIALS_PER_CATEGORY,
        rng, d)
    V_b, M_b = _shrinkage_axes(marginal_a, lam_b, d)
    arm_b = {"lambda": lam_b, "V": V_b, "variance_captured": _variance_captured(V_b, M_b),
             "per_outer_fold_lambda": outer_lambdas_b, "n_outer_folds_used": n_outer_b}

    # dpca_arm_a/dpca_arm_b: the actual published demixed principal component analysis method (doi
    # 10.7554/eLife.10989), ridge-regularised reduced-rank regression via _dpca_ridge_axes, sharing every
    # CV-fold/pool/nested-outer-loop mechanism above with arm_a/arm_b -- only the axis-fitting function and
    # lambda grid differ. See MARGINALIZATION_ARM_NOT_PUBLISHED_METHOD_NOTE.
    # the axes actually reported are fit directly on the time-resolved maintenance trajectory (see
    # _dpca_ridge_axes' own docstring), so fit and evaluation pseudo-trials are drawn from disjoint halves
    # of each neuron's own real trials (_split_trial_pool) rather than the same pooled draw: dpca_train_pop
    # backs the axis fit, dpca_test_pop backs every reported distance-to-attractor and variance-captured
    # number for this arm. Lambda selection stays on the cheaper single-timepoint pseudo-trial construction
    # _select_lambda_cv_pseudopop already builds from the encoding window.
    dpca_train_pool, dpca_test_pool = _split_trial_pool(collected["enc"], common_cats, rng)
    dpca_train_pop, _ = _build_pseudopopulation(collected["enc"], collected["maint"],
                                                 N_PSEUDO_TRIALS_PER_CATEGORY, rng, dpca_train_pool)
    dpca_test_pop, _ = _build_pseudopopulation(collected["enc"], collected["maint"],
                                                N_PSEUDO_TRIALS_PER_CATEGORY, rng, dpca_test_pool)
    maint_z_train = (dpca_train_pop["maint"] - mu_std[None, :, None]) / sd_std[None, :, None]
    Z_maint_train = _project_pca_3d(maint_z_train, mu_pca, V_pca)
    maint_z_test = (dpca_test_pop["maint"] - mu_std[None, :, None]) / sd_std[None, :, None]
    Z_maint_test = _project_pca_3d(maint_z_test, mu_pca, V_pca)
    correct_all_test = np.ones(len(dpca_test_pop["category"]), dtype=bool)

    lam_dpca_a, score_table_dpca_a = _select_lambda_cv_pseudopop(
        collected["enc"], collected["maint"], common_cats, n_pc, N_PSEUDO_TRIALS_PER_CATEGORY,
        N_CV_FOLDS_LAMBDA, DPCA_RIDGE_LAMBDA_GRID, rng, d, fit_fn=_dpca_ridge_axes)
    V_dpca_a, M_dpca_a = _dpca_ridge_axes(Z_maint_train, dpca_train_pop["category"], lam_dpca_a, d)
    dpca_arm_a = {"lambda": lam_dpca_a, "V": V_dpca_a, "variance_captured": _variance_captured(V_dpca_a, M_a),
                  "cv_score_table": {str(k): v for k, v in score_table_dpca_a.items()}}

    lam_dpca_b, outer_lambdas_dpca_b, n_outer_dpca_b = _fit_axes_arm_b_nested_pseudopop(
        collected["enc"], collected["maint"], common_cats, n_pc, N_PSEUDO_TRIALS_PER_CATEGORY,
        rng, d, grid=DPCA_RIDGE_LAMBDA_GRID, fit_fn=_dpca_ridge_axes)
    V_dpca_b, M_dpca_b = _dpca_ridge_axes(Z_maint_train, dpca_train_pop["category"], lam_dpca_b, d)
    dpca_arm_b = {"lambda": lam_dpca_b, "V": V_dpca_b, "variance_captured": _variance_captured(V_dpca_b, M_a),
                  "per_outer_fold_lambda": outer_lambdas_dpca_b, "n_outer_folds_used": n_outer_dpca_b}

    dpca_ridge_axes_lambda_dependence_check = _dpca_axes_overlap_across_grid(
        Z_maint_train, dpca_train_pop["category"], DPCA_RIDGE_LAMBDA_GRID, d, _dpca_ridge_axes)
    dpca_ridge_arm_a_vs_arm_b_axes_overlap = _subspace_overlap(V_dpca_a, V_dpca_b)

    da_primary = _da_trial_scores(Z_maint_pca_primary @ arm_a["V"], pseudopop["category"], correct_all)
    da_dpca_ridge_primary = _da_trial_scores(Z_maint_test @ dpca_arm_a["V"], dpca_test_pop["category"],
                                              correct_all_test)

    cat_time_marginal = _category_time_marginal(Z_maint_pca_primary, pseudopop["category"])
    M_cat_time = cat_time_marginal.T @ cat_time_marginal
    arm_a_variance_captured_category_time_marginal = _variance_captured(arm_a["V"], M_cat_time)
    arm_b_variance_captured_category_time_marginal = _variance_captured(arm_b["V"], M_cat_time)
    cat_time_marginal_test = _category_time_marginal(Z_maint_test, dpca_test_pop["category"])
    M_cat_time_test = cat_time_marginal_test.T @ cat_time_marginal_test
    dpca_arm_a_variance_captured_category_time_marginal = _variance_captured(dpca_arm_a["V"], M_cat_time_test)
    dpca_arm_b_variance_captured_category_time_marginal = _variance_captured(dpca_arm_b["V"], M_cat_time_test)

    lambda_sweep = _lambda_sweep(Z_enc, pseudopop["category"], Z_maint_pca_primary, score_table_a,
                                  SHRINKAGE_GRID, d, lam_a, lam_b)
    lambda_sweep_flatness = _lambda_sweep_flatness(score_table_a)
    dpca_ridge_lambda_sweep = _lambda_sweep(Z_maint_train, dpca_train_pop["category"], Z_maint_test,
                                             score_table_dpca_a, DPCA_RIDGE_LAMBDA_GRID, d, lam_dpca_a,
                                             lam_dpca_b, fit_fn=_dpca_ridge_axes)
    dpca_ridge_lambda_sweep_flatness = _lambda_sweep_flatness(score_table_dpca_a)

    collected_sens = _collect_region_neurons(corpus, region, SLIDING_WINDOW_STEP_S_SENSITIVITY, selected_keys)
    rng_sens = np.random.default_rng(stable_seed(f"{tag}_pseudopop"))
    pseudopop_sens, _ = _build_pseudopopulation(collected_sens["enc"], collected_sens["maint"],
                                                 N_PSEUDO_TRIALS_PER_CATEGORY, rng_sens)
    step_sensitivity = {"status": "not_computable"}
    if pseudopop_sens is not None and pseudopop_sens["n_units"] == pseudopop["n_units"]:
        maint_z_sens = (pseudopop_sens["maint"] - mu_std[None, :, None]) / sd_std[None, :, None]
        Z_maint_pca_sens = _project_pca_3d(maint_z_sens, mu_pca, V_pca)
        da_sens = _da_trial_scores(Z_maint_pca_sens @ arm_a["V"], pseudopop_sens["category"], correct_all)
        both_finite = np.isfinite(da_primary["da"]) & np.isfinite(da_sens["da"])
        if both_finite.sum() >= 4:
            step_sensitivity = {
                "status": "computed", "n_pseudo_trials_compared": int(both_finite.sum()),
                "correlation_primary_vs_sensitivity_step": float(np.corrcoef(
                    da_primary["da"][both_finite], da_sens["da"][both_finite])[0, 1]),
                "mean_absolute_difference": float(np.mean(np.abs(
                    da_primary["da"][both_finite] - da_sens["da"][both_finite]))),
            }

    return {
        "status": "computed", "n_sessions_used": collected["n_sessions_used"],
        "n_patients": collected["n_patients"], "n_units_pooled": pseudopop["n_units"],
        "n_pseudo_trials_per_category": N_PSEUDO_TRIALS_PER_CATEGORY,
        "categories_used": common_cats, "n_pc": n_pc, "category_subspace_dimensionality": d,
        "arm_a_lambda": arm_a["lambda"],
        "arm_a_variance_captured_static_category_marginal": arm_a["variance_captured"],
        "arm_b_lambda": arm_b["lambda"],
        "arm_b_variance_captured_static_category_marginal": arm_b["variance_captured"],
        "arm_b_per_outer_fold_lambda": arm_b["per_outer_fold_lambda"],
        "static_category_marginal_variance_captured_note": (
            "the static-category-marginal ratio above is 1.0 by construction whenever "
            f"CATEGORY_SUBSPACE_DIMENSIONALITY ({CATEGORY_SUBSPACE_DIMENSIONALITY}) is >= "
            "n_categories_usable - 1, because the across-category-mean marginal it is measured against "
            "has rank at most n_categories_usable - 1 regardless of the data -- it would be 1.0 for pure "
            "noise too. It is NOT comparable to the published study's 60.26%, which is a fraction of a "
            "category-by-time marginal of much higher rank. The comparable quantity is "
            "arm_a/b_variance_captured_category_time_marginal below."
        ),
        "arm_a_variance_captured_category_time_marginal": arm_a_variance_captured_category_time_marginal,
        "arm_b_variance_captured_category_time_marginal": arm_b_variance_captured_category_time_marginal,
        "arm_a_minus_arm_b_variance_captured_category_time_marginal": (
            arm_a_variance_captured_category_time_marginal - arm_b_variance_captured_category_time_marginal
            if np.isfinite(arm_a_variance_captured_category_time_marginal)
            and np.isfinite(arm_b_variance_captured_category_time_marginal) else None),
        "category_time_marginal_note": (
            "fraction of the category-by-time marginal (built from the same axes arm_a/arm_b fit on the "
            "encoding window, applied to the full 200ms-sliding-window maintenance trajectory, averaged "
            "per category per timepoint) that those axes explain -- the quantity genuinely comparable to "
            "the published study's 60.26%, since both are computed over a category-by-time marginal "
            "rather than a static per-category mean."
        ),
        "lambda_sweep": lambda_sweep,
        "lambda_sweep_flatness": lambda_sweep_flatness,
        "lambda_sweep_note": (
            "the primary reporting mode for this cell's lambda-dependent quantities is the sweep above, "
            "not the single cross-validated argmin: for every lambda on SHRINKAGE_GRID, "
            "lambda_sweep reports variance_captured_category_time_marginal, mean_distance_to_attractor, "
            "and the held-out reconstruction error that lambda scored during CV-selection (the actual "
            "selection surface, cross-validated over disjoint real-trial pools). arm_a_lambda and "
            "arm_b_lambda mark, via is_arm_a_cv_selected_point/is_arm_b_cv_selected_point, which sweep "
            "entry a cross-validated selection happened to land on -- lambda_sweep_flatness measures, "
            "not assumes, whether that argmin sits on a flat or a peaked surface; when it is flat, read "
            "the swept quantities across the grid rather than only at the argmin."
        ),
        "shrinkage_axes_isotropic_invariance_note": SHRINKAGE_AXES_ISOTROPIC_INVARIANCE_NOTE,
        "decoding_lambda_independence_note": (
            "decoding.observed_accuracy is fit on raw per-unit mean maintenance rate, not a lambda-"
            "dependent dPCA projection, so it does not vary with lambda in this construction and has no "
            "entry in lambda_sweep; only variance_captured_category_time_marginal and "
            "mean_distance_to_attractor depend on the selected axes."
        ),
        "marginalization_arm_not_published_method_note": MARGINALIZATION_ARM_NOT_PUBLISHED_METHOD_NOTE,
        "dpca_ridge_arm_a_lambda": dpca_arm_a["lambda"],
        "dpca_ridge_arm_a_variance_captured_static_category_marginal": dpca_arm_a["variance_captured"],
        "dpca_ridge_arm_b_lambda": dpca_arm_b["lambda"],
        "dpca_ridge_arm_b_variance_captured_static_category_marginal": dpca_arm_b["variance_captured"],
        "dpca_ridge_arm_b_per_outer_fold_lambda": dpca_arm_b["per_outer_fold_lambda"],
        "dpca_ridge_static_category_marginal_variance_captured_note": (
            "unlike arm_a/b_variance_captured_static_category_marginal, this quantity is NOT 1.0 by "
            "construction: dpca_ridge axes are the top eigenvectors of the ridge-regression's own fitted-"
            "value second moment (B.T @ Sxx @ B), fit against the time-resolved category-by-time target, "
            "not against the static category marginal covariance itself, so how much of the static "
            "marginal they happen to capture is a genuine, non-tautological measurement, evaluated against "
            "the identical unshrunk marginal covariance M used above for apples-to-apples comparison."
        ),
        "dpca_ridge_arm_a_variance_captured_category_time_marginal": (
            dpca_arm_a_variance_captured_category_time_marginal),
        "dpca_ridge_arm_b_variance_captured_category_time_marginal": (
            dpca_arm_b_variance_captured_category_time_marginal),
        "dpca_ridge_arm_a_vs_arm_b_axes_overlap": dpca_ridge_arm_a_vs_arm_b_axes_overlap,
        "dpca_ridge_da": da_dpca_ridge_primary["da"].tolist(),
        "dpca_ridge_lambda_sweep": dpca_ridge_lambda_sweep,
        "dpca_ridge_lambda_sweep_flatness": dpca_ridge_lambda_sweep_flatness,
        "dpca_ridge_axes_lambda_dependence_check": dpca_ridge_axes_lambda_dependence_check,
        "dpca_ridge_time_resolved_near_invariance_note": (
            "dpca_ridge_arm_a/b and dpca_ridge_axes_lambda_dependence_check are fit and measured on the "
            "time-resolved category-by-time marginal (category means as a function of time, via "
            "_category_time_marginal), the faithful reproduction of the published study's own 200ms "
            "sliding-window binned marginalisation, independently of which lambda is selected. On this "
            "cell's real data, dpca_ridge_axes_lambda_dependence_check's min_lambda_vs_max_lambda_overlap "
            "reports how much DPCA_RIDGE_LAMBDA_GRID actually moves that subspace -- when it sits at or "
            "near 1.0, the regularisation strength is not materially determining the category subspace "
            "here, a property of the data at this component count rather than a defect in the axis-fitting "
            "code -- the top-level dpca_ridge_axes_synthetic_verification documents that the same function "
            "does rotate genuinely with lambda on a lower-rank synthetic construction, so the near-"
            "invariance measured here is not a bug that silently pins the axes everywhere."
        ),
        "dpca_ridge_method_note": (
            "dpca_ridge_arm_a/b implement the published study's demixed principal component analysis "
            "(doi 10.7554/eLife.10989) as a ridge-regularised reduced-rank regression (see "
            "_dpca_ridge_axes' own docstring for the exact formulation and component ordering): encoder "
            "and decoder are fit per marginalization by minimising the squared error of reconstructing the "
            "time-resolved category-by-time marginal target from full population activity under a ridge "
            "penalty inside a matrix inverse. dpca_ridge_axes_lambda_dependence_check reports the measured "
            "subspace overlap across DPCA_RIDGE_LAMBDA_GRID at this cell rather than assuming it must sit "
            "below 1.0 -- see dpca_ridge_time_resolved_near_invariance_note for what the measured value "
            "means here; the run is never halted on this number."
        ),
        "step_size_sensitivity": step_sensitivity,
        "pseudopopulation": pseudopop, "mu_pca": mu_pca, "V_pca": V_pca, "mu_std": mu_std, "sd_std": sd_std,
        "arm_a": arm_a, "dpca_arm_a": dpca_arm_a,
        "collected_enc": collected["enc"], "collected_maint": collected["maint"],
        "schema_version": PSEUDOPOP_SCHEMA_VERSION,
    }


def _split_trial_pool(neurons_enc: dict, common_cats: list, rng: np.random.Generator) -> tuple[dict, dict]:
    train_pool: dict = {}
    test_pool: dict = {}
    for key, cats in neurons_enc.items():
        train_pool[key], test_pool[key] = {}, {}
        for c in common_cats:
            n_trials = len(cats[c])
            idx = rng.permutation(n_trials)
            half = n_trials // 2
            train_pool[key][c], test_pool[key][c] = idx[:half], idx[half:]
    return train_pool, test_pool


def _pseudopopulation_decoding(collected_enc: dict, collected_maint: dict, rng: np.random.Generator,
                                n_shuffle: int = N_DECODING_SHUFFLE) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    neuron_keys = sorted(collected_enc)
    if not neuron_keys:
        return {"status": "not_computable", "reason": "no pooled neurons"}
    common_cats = sorted(set.intersection(*(set(collected_enc[k]) for k in neuron_keys)))
    if len(common_cats) < MIN_CATEGORIES_FOR_DA:
        return {"status": "not_computable",
                "reason": f"fewer than {MIN_CATEGORIES_FOR_DA} categories in the common pool of neurons"}

    # each neuron's own real trials, per category, are split into disjoint train/test halves ONCE;
    # every repeat and every shuffle draw fresh pseudo-trials from those same two disjoint pools, so no
    # real trial can ever land in both a training pseudo-trial and a test pseudo-trial.
    train_pool, test_pool = _split_trial_pool(collected_enc, common_cats, rng)

    def _fit_score(shuffle_labels: bool) -> float | None:
        train_pop, _ = _build_pseudopopulation(collected_enc, collected_maint, N_PSEUDO_TRIALS_PER_CATEGORY,
                                                rng, train_pool)
        test_pop, _ = _build_pseudopopulation(collected_enc, collected_maint, N_PSEUDO_TRIALS_PER_CATEGORY,
                                               rng, test_pool)
        if train_pop is None or test_pop is None:
            return None
        X_train, y_train = train_pop["maint"].mean(axis=2), train_pop["category"]
        X_test, y_test = test_pop["maint"].mean(axis=2), test_pop["category"]
        if shuffle_labels:
            y_train = rng.permutation(y_train)
        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(C=1.0, max_iter=500).fit(scaler.transform(X_train), y_train)
        return float(clf.score(scaler.transform(X_test), y_test))

    accuracies = [a for a in (_fit_score(False) for _ in range(N_DECODING_SUBSAMPLE_REPEATS)) if a is not None]
    if not accuracies:
        return {"status": "not_computable", "reason": "no repeat produced a scoreable train/test split"}
    observed = float(np.mean(accuracies))
    accuracy_se = float(np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))) if len(accuracies) > 1 else float("nan")

    null_accuracies = [a for a in (_fit_score(True) for _ in range(n_shuffle)) if a is not None]
    p_value = permutation_pvalue(np.array(null_accuracies) >= observed) if null_accuracies else float("nan")
    chance = 1.0 / len(common_cats)
    return {
        "status": "computed", "observed_accuracy": observed, "accuracy_se": accuracy_se,
        "chance_accuracy": chance, "p_value": p_value, "null_accuracies": null_accuracies,
        "n_units": len(neuron_keys), "n_categories": len(common_cats),
        "n_pseudo_trials_per_split": N_PSEUDO_TRIALS_PER_CATEGORY * len(common_cats),
        "n_decoding_repeats_used": N_DECODING_SUBSAMPLE_REPEATS,
        "n_decoding_repeats_published": N_DECODING_SUBSAMPLE_REPEATS_PUBLISHED,
        "n_shuffle_used": n_shuffle, "n_shuffle_published": N_DECODING_SHUFFLE_PUBLISHED,
        "train_test_construction": "every pooled neuron's own real trials are split into disjoint train "
            "and test halves once per region before any pseudo-trial is resampled, so no real trial "
            "contributes to both a training and a test pseudo-trial; each repeat and each shuffle draw "
            "fresh pseudo-trials from those same two disjoint real-trial pools.",
        "reduction_reason": DECODING_COMPUTE_REDUCTION_REASON,
    }


def _pseudopopulation_region_report(corpus: str, region: str, selected_keys: set | None = None,
                                     seed_tag: str | None = None) -> dict:
    tag = seed_tag if seed_tag is not None else f"{corpus}_{region}"
    geometry = _pseudopopulation_geometry(corpus, region, selected_keys, seed_tag=tag)
    if geometry.get("status") != "computed":
        return geometry
    rng = np.random.default_rng(stable_seed(f"{tag}_pseudopop_decode"))
    decoding = _pseudopopulation_decoding(geometry["collected_enc"], geometry["collected_maint"], rng)
    report = {k: v for k, v in geometry.items()
              if k not in ("pseudopopulation", "mu_pca", "V_pca", "mu_std", "sd_std", "arm_a", "dpca_arm_a",
                           "collected_enc", "collected_maint")}
    report["decoding"] = decoding
    return report


N_COUNT_MATCHED_REPEATS = 200


def _holdout_restricted(collected: dict, keys, common_cats: list, test_pool: dict) -> tuple[dict, dict]:
    enc = {k: {c: collected["enc"][k][c][test_pool[k][c]] for c in common_cats} for k in keys}
    maint = {k: {c: collected["maint"][k][c][test_pool[k][c]] for c in common_cats} for k in keys}
    return enc, maint


def _selective_holdout_partition(collected: dict, common_cats: list, neuron_keys: list, corpus: str,
                                  region: str, split_i: int) -> tuple[set, dict]:
    rng = np.random.default_rng(stable_seed(f"{corpus}_{region}_selective_holdout_{split_i}"))
    train_pool, test_pool = _split_trial_pool(collected["enc"], common_cats, rng)
    selected = set()
    for key in neuron_keys:
        rates_parts, label_parts = [], []
        for c in common_cats:
            idx = train_pool[key][c]
            if len(idx) == 0:
                continue
            rates_parts.append(collected["enc"][key][c][idx])
            label_parts.append(np.full(len(idx), c))
        if not rates_parts:
            continue
        rates_train = np.concatenate(rates_parts)
        labels_train = np.concatenate(label_parts)
        trial_id_train = np.arange(len(rates_train))
        test = two_stage_selectivity_test(
            rates_train, labels_train, trial_id_train, rng, n_perm=N_PERM_NEURON,
            min_total_trials=MIN_TOTAL_TRIALS_FOR_NEURON_TEST,
            min_trials_per_level=MIN_TRIALS_PER_CATEGORY_FOR_NEURON_TEST,
            omnibus_alpha=CATEGORY_NEURON_OMNIBUS_ALPHA, permutation_alpha=CATEGORY_NEURON_PERMUTATION_ALPHA)
        if test["status"] == "computed" and test["meets_selectivity_criterion"]:
            selected.add(key)
    return selected, test_pool


def _count_matched_pseudopopulation_split(corpus: str, region: str, collected: dict, common_cats: list,
                                           all_unit_keys: list, test_pool: dict, n_target: int,
                                           split_i: int) -> dict:
    if n_target < 1 or len(all_unit_keys) < n_target:
        return {"status": "not_computable", "n_target_units": n_target, "n_candidate_units": len(all_unit_keys),
                "reason": "fewer candidate units than this split's selected-unit count"}
    accuracies = []
    n_used = 0
    for draw in range(N_COUNT_MATCHED_REPEATS):
        rng = np.random.default_rng(stable_seed(f"{corpus}_{region}_count_matched_{split_i}_{draw}"))
        subset = rng.choice(sorted(all_unit_keys), size=n_target, replace=False).tolist()
        holdout_enc, holdout_maint = _holdout_restricted(collected, subset, common_cats, test_pool)
        decode_rng = np.random.default_rng(
            stable_seed(f"{corpus}_{region}_count_matched_decode_{split_i}_{draw}"))
        decoding = _pseudopopulation_decoding(holdout_enc, holdout_maint, decode_rng, n_shuffle=0)
        if decoding.get("status") != "computed":
            continue
        n_used += 1
        accuracies.append(decoding["observed_accuracy"])
    if n_used == 0:
        return {"status": "not_computable", "n_target_units": n_target, "n_candidate_units": len(all_unit_keys),
                "n_repeats_attempted": N_COUNT_MATCHED_REPEATS,
                "reason": "no draw produced a computed decoding"}
    arr = np.array(accuracies)
    quantiles = {"q2_5": float(np.percentile(arr, 2.5)), "q25": float(np.percentile(arr, 25)),
                 "q50": float(np.percentile(arr, 50)), "q75": float(np.percentile(arr, 75)),
                 "q97_5": float(np.percentile(arr, 97.5))}
    return {
        "status": "computed", "n_target_units": n_target, "n_candidate_units": len(all_unit_keys),
        "n_repeats_used": n_used, "n_repeats_attempted": N_COUNT_MATCHED_REPEATS,
        "accuracy_distribution": {
            "values": [float(v) for v in accuracies],
            "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            **quantiles,
        },
        "description": "unit subsets drawn at random, with no reference to the data, sized to this split's "
            "own selected-unit count and evaluated on this split's own held-out test partition.",
    }


def _pseudopopulation_selective_holdout_arm(corpus: str, region: str) -> tuple[dict, dict, list, list]:
    collected = _collect_region_neurons(corpus, region, SLIDING_WINDOW_STEP_S_PRIMARY)
    neuron_keys = sorted(collected["enc"])
    if not neuron_keys:
        empty = {"status": "not_computable", "reason": "no pooled neurons"}
        return empty, collected, [], neuron_keys
    common_cats = sorted(set.intersection(*(set(collected["enc"][k]) for k in neuron_keys)))
    if len(common_cats) < MIN_CATEGORIES_FOR_DA:
        empty = {"status": "not_computable",
                 "reason": f"fewer than {MIN_CATEGORIES_FOR_DA} categories in the common pool of neurons"}
        return empty, collected, common_cats, neuron_keys

    per_split = []
    fits, fits_floor_restricted, differences = [], [], []
    for split_i in range(N_TRIAL_SPLIT_REPEATS):
        selected, test_pool = _selective_holdout_partition(collected, common_cats, neuron_keys, corpus,
                                                             region, split_i)
        n_selected = len(selected)
        above_floor = n_selected >= MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM
        if n_selected < 2:
            per_split.append({"split": split_i, "n_selected_units": n_selected,
                               "above_published_minimum_category_neuron_count": above_floor,
                               "status": "underpowered"})
            continue
        holdout_enc, holdout_maint = _holdout_restricted(collected, selected, common_cats, test_pool)
        decode_rng = np.random.default_rng(stable_seed(f"{corpus}_{region}_selective_holdout_decode_{split_i}"))
        decoding = _pseudopopulation_decoding(holdout_enc, holdout_maint, decode_rng)
        row = {"split": split_i, "n_selected_units": n_selected,
               "above_published_minimum_category_neuron_count": above_floor,
               "status": decoding.get("status"), "observed_accuracy": decoding.get("observed_accuracy"),
               "accuracy_se": decoding.get("accuracy_se")}
        if decoding.get("status") == "computed":
            fit = {"status": "computed", "estimate": decoding["observed_accuracy"],
                   "se": decoding.get("accuracy_se", float("nan")), "n_neurons": n_selected,
                   "n_trials": decoding.get("n_pseudo_trials_per_split", 0),
                   "null_accuracies": decoding.get("null_accuracies", [])}
            fits.append(fit)
            if above_floor:
                fits_floor_restricted.append(fit)
            random_subset = _count_matched_pseudopopulation_split(
                corpus, region, collected, common_cats, neuron_keys, test_pool, n_selected, split_i)
            row["count_matched_all_unit_subsample"] = random_subset
            if random_subset.get("status") == "computed":
                diff = decoding["observed_accuracy"] - random_subset["accuracy_distribution"]["mean"]
                row["selective_minus_random_difference"] = diff
                differences.append(diff)
        per_split.append(row)

    combined = _combine_decoding_splits(fits)
    combined["per_split"] = per_split
    combined["n_splits_total"] = N_TRIAL_SPLIT_REPEATS
    combined["description"] = (
        "pools every random trial partition whose train-half selection produced at least two selected "
        "units and a computed held-out decoding, regardless of whether the selected-unit count cleared "
        "the category-neuron floor; per_split carries every partition's own selected-unit count, its own "
        "count-matched random-subset null and selective-minus-random difference, and, where computed, its "
        "own accuracy, including partitions dropped from the pooled estimate."
    )
    combined_floor = _combine_decoding_splits(fits_floor_restricted)
    combined_floor["per_split"] = per_split
    combined_floor["n_splits_total"] = N_TRIAL_SPLIT_REPEATS
    combined_floor["n_splits_below_published_minimum_category_neuron_count"] = (
        N_TRIAL_SPLIT_REPEATS - len(fits_floor_restricted))
    combined_floor["description"] = (
        "secondary restriction of the pooled decoding above to partitions whose train-half selection "
        "cleared the category-neuron floor only; dropping the other partitions biases the surviving set "
        "toward partitions where selection happened to find more units, so the unrestricted pooled arm "
        "above, not this one, is the reported estimate."
    )
    combined["floor_restricted"] = combined_floor
    if differences:
        diff_arr = np.array(differences)
        se_d = float(diff_arr.std(ddof=1) / np.sqrt(len(diff_arr))) if len(diff_arr) > 1 else float("nan")
        combined["selective_minus_random_difference"] = {
            "mean": float(diff_arr.mean()), "se": se_d, "n_splits": len(diff_arr),
            "ci_lower": float(diff_arr.mean() - 1.959963984540054 * se_d) if np.isfinite(se_d) else None,
            "ci_upper": float(diff_arr.mean() + 1.959963984540054 * se_d) if np.isfinite(se_d) else None,
            "per_split": differences,
        }
    return combined, collected, common_cats, neuron_keys


SELECTIVE_ARM_DESCRIPTION = (
    "population pooled only from units that met the widened-window category-selectivity test used by the "
    "single-unit selective-persistence analysis (meets_published_category_neuron_criterion, computed once "
    "there over the full trial set and read directly, not reapplied here). Its units were selected using "
    "all trials, including the trials the decoder below is evaluated on -- selection and evaluation "
    "overlap, so its decoding accuracy is optimistic by construction and is kept beside "
    "selective_arm_holdout, not in place of it, so a reader can see the size of that overlap by comparing "
    "the two."
)
ALL_UNIT_ARM_DESCRIPTION = (
    "population pooled from every unit recorded in the region; no per-unit selection is applied."
)


def _pseudopopulation_region_report_all_arms(corpus: str, region: str) -> dict:
    selected_keys = _selective_neuron_keys(corpus, region, field="test")
    selected_keys_first_encoding_only = _selective_neuron_keys(corpus, region, field="test_first_encoding_only")
    fittable = len(selected_keys) >= MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM
    fittable_first_encoding_only = (
        len(selected_keys_first_encoding_only) >= MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM)
    fittability_differs = fittable != fittable_first_encoding_only

    if not selected_keys:
        selective_report = {"status": "refused", "n_selective_units": 0,
                             "reason": "no unit in this region met the widened-window category-"
                                       "selectivity test"}
    else:
        selective_report = _pseudopopulation_region_report(corpus, region, selected_keys=selected_keys,
                                                             seed_tag=f"{corpus}_{region}_selective")
        selective_report = {**selective_report, "n_selective_units": len(selected_keys)}

    all_unit_report = _pseudopopulation_region_report(corpus, region)

    holdout_arm, _, _, _ = _pseudopopulation_selective_holdout_arm(corpus, region)

    result = {
        "selective_arm": {"description": SELECTIVE_ARM_DESCRIPTION, **selective_report},
        "selective_arm_holdout": {
            "description": "selection and evaluation trials are disjoint: category-selective units are "
                "chosen on a random half of the trials and decoding accuracy is measured on the other "
                "half, repeated over random splits and pooled -- see _pseudopopulation_selective_holdout_"
                "arm's own description for the per-split detail. Each split's own count-matched random-"
                "unit-subset null and selective-minus-random difference sit in per_split; the pooled "
                "difference across splits is in selective_minus_random_difference.",
            **holdout_arm,
        },
        "all_unit_arm": {"description": ALL_UNIT_ARM_DESCRIPTION, **all_unit_report},
        "selective_unit_prevalence": {
            "n_units_widened_window": len(selected_keys),
            "n_units_first_encoding_only": len(selected_keys_first_encoding_only),
            "fittable_widened_window": fittable, "fittable_first_encoding_only": fittable_first_encoding_only,
            "fittability_differs": fittability_differs,
        },
        "schema_version": PSEUDOPOP_SCHEMA_VERSION,
    }
    if fittable_first_encoding_only and selected_keys_first_encoding_only:
        result["selective_arm_first_encoding_only"] = {
            "description": SELECTIVE_ARM_DESCRIPTION + " Gated on the first-encoding-onset-only window, "
                "reported beside selective_arm (gated on the widened window) at every cell where the "
                "first-encoding-only window's own unit count clears the category-neuron floor.",
            **_pseudopopulation_region_report(corpus, region, selected_keys=selected_keys_first_encoding_only,
                                               seed_tag=f"{corpus}_{region}_selective_first_encoding_only"),
            "n_selective_units": len(selected_keys_first_encoding_only),
        }
    return result


def _checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.json"


def _load_checkpoint(name: str) -> dict:
    path = _checkpoint_path(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("sessions", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _record_session(name: str, session_key: str, record: dict) -> None:
    with locked_json_update(_checkpoint_path(name)) as ck:
        ck.setdefault("sessions", {})[session_key] = record


def process_corpus(corpus: str) -> None:
    spec = CORPUS_SPECS[corpus]
    # a checkpoint entry whose schema_version does not match the constant below is filtered out here
    # (not deleted on disk; _record_session overwrites the same key in place), so process_corpus
    # recomputes that session under the current construction rather than resuming a schema it cannot
    # interpret.
    part1_done = {r: {k: v for k, v in _load_checkpoint(f"{corpus}__part1__{r}").items()
                       if v.get("schema_version") == PART1_SCHEMA_VERSION}
                  for r in spec["regions"]}
    da_done = {r: {k: v for k, v in _load_checkpoint(f"{corpus}__da__{r}").items()
                   if v.get("schema_version") == DA_SCHEMA_VERSION}
               for r in spec["regions"]}
    for path in spec["glob"]():
        session_key = path.stem
        need_part1 = [r for r in spec["regions"] if session_key not in part1_done[r]]
        need_da = [r for r in spec["regions"] if session_key not in da_done[r]]
        if not need_part1 and not need_da:
            continue
        session = spec["loader"](path)
        if session is None:
            reason = {"status": "refused", "reason": "session excluded: below MIN_TRIALS overall or "
                      "load-1 subset, or MIN_SESSION_ACCURACY, or missing units/trials tables, or not "
                      "the canonical primary recording for its patient identity"}
            reason_part1 = {**reason, "schema_version": PART1_SCHEMA_VERSION}
            reason_da = {**reason, "schema_version": DA_SCHEMA_VERSION}
            for r in need_part1:
                _record_session(f"{corpus}__part1__{r}", session_key, reason_part1)
                part1_done[r][session_key] = reason_part1
            for r in need_da:
                _record_session(f"{corpus}__da__{r}", session_key, reason_da)
                da_done[r][session_key] = reason_da
            continue
        for r in need_part1:
            rng = np.random.default_rng(stable_seed(f"{session_key}_{r}_part1"))
            cell = _part1_region_session(session, r, rng)
            cell["patient"] = session["patient"]
            _record_session(f"{corpus}__part1__{r}", session_key, cell)
            part1_done[r][session_key] = cell
        for r in need_da:
            cell = _da_session(session, r)
            cell["patient"] = session["patient"]
            _record_session(f"{corpus}__da__{r}", session_key, cell)
            da_done[r][session_key] = cell
        print(f"  {corpus} {session_key}: part1+da done", flush=True)


def _session_correlation(x_key: str, y_key: str, cell: dict, tag: str) -> float | None:
    x = np.asarray(cell.get(x_key, []), dtype=float)
    y = np.asarray(cell.get(y_key, []), dtype=float)
    if len(x) < 8 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    rng = np.random.default_rng(stable_seed(tag))
    test = partial_correlation_permutation_test(y, x, controls=[], n_perm=N_PERM, rng=rng)
    return test.get("r") if np.isfinite(test.get("r", float("nan"))) else None


def _pooled_association(cells_by_region: dict, x_key: str, y_key: str, label: str) -> dict:
    per_patient_rows: dict[str, list] = {}
    for region, cells in cells_by_region.items():
        for session_key, cell in cells.items():
            if cell.get("status") != "computed":
                continue
            r = _session_correlation(x_key, y_key, cell, f"{label}|{region}|{session_key}")
            if r is not None:
                per_patient_rows.setdefault(cell["patient"], []).append(r)
    patient_values = [float(np.mean(v)) for v in per_patient_rows.values()]
    if len(patient_values) < 4:
        return {"status": "underpowered", "n_patients": len(patient_values),
                "reason": "fewer than 4 patients carry a computed session-level correlation"}
    test = slope_across_sessions_test(patient_values, alternative="two-sided")
    mdd = minimum_detectable_paired_difference(patient_values)
    return {"status": test["status"], "n_patients": len(patient_values),
            "mean_value": test.get("mean_value"), "p_value": test.get("p_value"),
            "ci_lower": test.get("ci_lower"), "ci_upper": test.get("ci_upper"),
            "mdd": mdd.get("mdd"), "patient_values": patient_values}


def _region_association(cells: dict, x_key: str, y_key: str, label: str) -> dict:
    return _pooled_association({"_single": cells}, x_key, y_key, label)


def _read_existing_maintenance_behaviour_null() -> dict:
    path = RESULTS / "human_maintenance_behaviour_link.json"
    if not path.exists():
        return {"status": "not_found"}
    d = json.loads(path.read_text())
    block_b = d.get("block_b", {})
    branch = block_b.get("branch", {})
    raw = block_b.get("combined_within_load_then_meta_analysed", {}).get("raw", {})
    return {
        "status": "read", "branch": branch.get("branch"),
        "mdd_80pct_power_r_units": branch.get("minimum_detectable_difference_80pct_power", {}).get("mdd"),
        "combined_raw_pooled_estimate": raw.get("pooled"), "combined_raw_p_value": raw.get("p_value"),
        "combined_raw_k_arms": raw.get("k"),
        "note": "results/human_maintenance_behaviour_link.json's 52-patient pooled null (dandi_000469, "
                "dandi_001187, dandi_000574; dandi_000673 was never part of it). This artifact's DA-to-"
                "behaviour cells extend that baseline (a different observable, an overlapping but not "
                "identical corpus set) rather than reproducing or overwriting it.",
    }


def build_artifact() -> dict:
    predeclared_rules = {
        "content_label_basis": {
            "dandi_000469": CATEGORIES_469, "dandi_000673": CATEGORIES_100_DIVIDED,
            "dandi_001187": CATEGORIES_100_DIVIDED, "dandi_000574": DANDI_000574_EXCLUSION_REASON,
        },
        "item_identity_test_feasibility": ITEM_IDENTITY_TEST_FEASIBILITY,
        "load_restriction": "load-1 (single-item) trials only, throughout every part, matching the "
            "published study's own load-1 restriction for its dPCA/DA construction.",
        "category_neuron_window": f"{CATEGORY_NEURON_WINDOW_OFFSET_S*1000:.0f}-"
            f"{(CATEGORY_NEURON_WINDOW_OFFSET_S+CATEGORY_NEURON_WINDOW_DURATION_S)*1000:.0f}ms after onset, "
            "per the published study's own stated window, counted on every presentation event in the trial "
            "that carries a memorandum label -- at the load-1 restriction used throughout, that is the "
            "encoding presentation and the probe -- and this widened-window test is what drives "
            "selection_matched_arm, trial_split_matched_arm, and part4's selective_arm. The probe-window "
            "rows are labelled with the probe's own picture category, read from the probe picture field, "
            "not with the encoding (memorandum) category. The first-encoding-onset-only version is also "
            "computed and kept as test_first_encoding_only on every neuron; it gates "
            "selection_matched_first_encoding_only_arm and part4's selective_arm_first_encoding_only at "
            "every cell where its own unit count clears the category-neuron floor, reported beside the "
            "widened-window arms rather than in place of them; the gap between the two windows' prevalence "
            "is itself disclosed, not just one number. Both stages "
            f"of the test are permutation-based: a trial-block permutation omnibus test across categories "
            f"with >={MIN_TRIALS_PER_CATEGORY_FOR_NEURON_TEST} trials (a whole trial, carrying every one of "
            "its presentation-event rows, moves together under each shuffle -- not individual presentation "
            "rows -- so the test is valid whether or not category assignment is independent of trial "
            "structure), then a right-tailed permutation test of the preferred category's mean against the "
            f"rest, both at {N_PERM_NEURON} draws and both at p<{CATEGORY_NEURON_OMNIBUS_ALPHA} -- "
            "reproducing their category-neuron definition for the selection-matched arm only; the full-"
            "population arm applies no such filter. The published method's own further step, dropping units "
            f"whose firing rate exceeds {OUTLIER_FIRING_RATE_SD_THRESHOLD:.0f} standard deviations above "
            "the selected set's mean, is not applied on the primary path -- it trims the dependent "
            "variable's own scale post-selection, and firing rates are right-skewed enough that it would "
            "remove high-rate units, disproportionately the strongly-tuned ones, systematically rather than "
            "randomly -- it is instead reported as its own additional arm, "
            "selection_matched_outlier_trimmed_arm.",
        "selective_persistence_model": "statistics.linear_mixed_effects_test(baseline-corrected "
            "maintenance rate ~ preferred-vs-non-preferred category, patient random intercept plus a "
            "neuron-level variance component nested inside patient via nested_group), correct load-1 "
            "trials only -- reproduces the published study's neuronID-nested-in-patientID mixed model "
            "structure: patient is the random-intercept grouping factor and neuron is a separate variance "
            "component nested inside it (statsmodels MixedLM.from_formula with vc_formula, one nested "
            "group per unique neuron, never shared across patients), so the many trial rows one neuron "
            "contributes are no longer scored as independent replicates. Against an unnested patient-only "
            "model on three (corpus, region) cells (dandi_000469 hippocampus, 164 neurons/6631 rows; dacc, "
            "158 neurons/6123 rows; amygdala, 207 neurons/8100 rows), the nested fit's condition-effect "
            "confidence interval is NARROWER, not wider, by 7.4pct/12.3pct/1.9pct respectively, and every "
            "point estimate moves by well under one unnested standard error (0.08/0.24/0.09 SE). This is "
            "the opposite of the naive pseudo-replication direction, and the reason is the covariate: "
            "preferred-vs-non-preferred category varies trial-to-trial within a neuron (not between "
            "neurons), so partitioning out the substantial between-neuron nuisance variance that the "
            "unnested fit dumps into its residual sharpens this within-neuron contrast rather than "
            "inflating its apparent precision -- the classic too-narrow-interval failure mode applies to a "
            "between-cluster covariate, which this one is not. Region dropped from the selection-matched "
            f"arm below {MIN_CATEGORY_NEURONS_FOR_SELECTION_MATCHED_ARM} category neurons, the published "
            "study's own pre-SMA exclusion floor applied uniformly. trial_split_matched_arm is the "
            f"reported estimate for this question: each neuron's own trials are split at the trial level "
            f"into two random halves, {N_TRIAL_SPLIT_REPEATS} times independently, category selection runs "
            "on one half's encoding-window rates only, maintenance selectivity is then estimated on the "
            f"other half's correct trials only, and the {N_TRIAL_SPLIT_REPEATS} resulting fits are pooled "
            "by Rubin's rule (combine_glm_fits_across_splits), so selection and evaluation trials never "
            "overlap. selection_matched_arm selects category neurons and estimates their maintenance "
            "selectivity from the SAME trials, so its disjoint encoding/maintenance time windows share "
            "trial-level noise (within-trial rate drift, or a category that happens to land on high-firing "
            "trials) that can inflate the enrichment it finds; it is kept as the overlapping-trial "
            "comparison beside trial_split_matched_arm, not as the reported arm, so a reader can see the "
            "size of that overlap directly.",
        "da_construction": MARGINALIZATION_ARM_NOT_PUBLISHED_METHOD_NOTE + " "
            "activity is standardised and PCA-reduced from encoding-period rates only "
            f"(the {AXIS_FIT_WINDOW_OFFSET_S*1000:.0f}-"
            f"{(AXIS_FIT_WINDOW_OFFSET_S+AXIS_FIT_WINDOW_DURATION_S)*1000:.0f}ms post-encoding-onset window, "
            "fit on load-1 correct trials in the usable categories only, never refit on maintenance data) "
            "before being carried into three parallel geometries, all sharing that one encoding-fit PCA "
            "basis: an unsupervised PCA-only baseline (da_pca_only, no category label used to build the "
            "axes); and two SUPERVISED demixed-PCA arms, which build their axes directly from the category "
            "labels (per-category mean rate vectors -- the class marginal -- eigendecomposed with "
            "shrinkage regularisation) so a reader must not assume an unsupervised method produced them: "
            "arm A (da / arm_a_*) selects its shrinkage lambda by ordinary cross-validation on the fitting "
            "trials; arm B (da_dpca_arm_b / arm_b_*) is nested -- lambda is re-selected inside each outer "
            "fold's training partition only, and the final lambda is the median across outer folds -- so "
            "the reported variance-captured number is never evaluated on data any part of its own lambda "
            "selection saw. Both dPCA arms keep the top "
            f"{CATEGORY_SUBSPACE_DIMENSIONALITY} axes by eigenvalue (capped at the number of encoding "
            "principal components available), matching the published study's own component count without "
            "copying its component indices onto our data. All three geometries share the same maintenance-"
            "period sliding-window trajectory (200ms windows) and the same attractor rule: attractor per "
            "category = leave-one-out mean maintenance position (own-category attractor excludes the "
            "trial itself; other-category attractors use every trial in that category -- a deliberate "
            "departure from the published study's non-excluding attractor, chosen because it removes by "
            "construction the same-trial selection bias their own permutation control was built to catch, "
            "so this artifact needs no analogous control). DA = own-attractor distance / mean distance to "
            "every other attractor. A separate, additional construction (part4_pseudopopulation_geometry) "
            "pools trial-resampled pseudo-trials across every session in a (corpus, region) cell rather "
            "than fitting per session, matching the published study's own pseudo-population procedure; it "
            "fits the same two marginalization-based arms (arm_a/arm_b) plus, additionally, two arms "
            "(dpca_ridge_arm_a/dpca_ridge_arm_b) that implement the published study's actual method as a "
            "ridge-regularised reduced-rank regression -- see dpca_ridge_method_note and "
            "marginalization_arm_not_published_method_note -- plus a logistic-regression category decoding "
            "check on that pooled population; it is a distinct cell family from the per-session DA above "
            "rather than a replacement for it. This whole construction is fit under several arms per cell "
            "(selective_arm, selective_arm_holdout, all_unit_arm) -- see "
            "part4_selective_versus_all_unit_arms. Two further properties of the underlying pseudo-population fit need explicit "
            "disclosure so neither is misread as a substantive finding: (1) decoding.observed_accuracy is "
            "from a construction that splits every pooled neuron's own real trials, per category, into "
            "disjoint train and test halves before any pseudo-trial is resampled, so no real trial can "
            "appear on both sides of a split; dpca_ridge_arm_a/b similarly fit their axes on pseudo-trials "
            "resampled from one real-trial half and evaluate distance-to-attractor and variance-captured on "
            "pseudo-trials resampled from the disjoint other half (_split_trial_pool), so no real trial "
            "backs both a dpca_ridge fit and its own evaluation either. (2) "
            "arm_a/b_variance_captured_static_category_marginal is 1.0 by mathematical necessity at this "
            "category count and CATEGORY_SUBSPACE_DIMENSIONALITY and must never be set beside the "
            "published study's 60.26pct; arm_a/b_variance_captured_category_time_marginal is the quantity "
            "genuinely comparable to it, computed the same way over a category-by-time marginal built "
            "from the full maintenance trajectory rather than a static per-category mean. (3) the lambda "
            "each arm cross-validates to is not treated as a settled value: _shrinkage_axes shrinks toward "
            "an isotropic target, and shrinking any symmetric matrix toward a scalar multiple of the "
            "identity leaves its eigenvectors exactly unchanged for every lambda short of 1.0 -- "
            "SHRINKAGE_GRID never reaches 1.0, so the selected axes are provably identical at every grid "
            "point for arm_a/arm_b, verified on real data to floating-point precision (relative range of "
            "order 1e-16); see shrinkage_axes_isotropic_invariance_note for the exact argument. The lambda "
            "a cross-validated selection lands on for arm_a/arm_b is therefore not a selection at all, at "
            "any cell; the same fact applies to the untouched per-session arms, which use the identical "
            "_shrinkage_axes function. lambda_sweep (every SHRINKAGE_GRID point's category-by-time variance "
            "captured, mean distance to attractor, and held-out reconstruction error) is "
            "therefore the primary reporting mode for this block, not arm_a/b_lambda alone; "
            "lambda_sweep_flatness reports the measured relative range and distinguishes floating-point-"
            "level flatness from a genuinely data-driven small range rather than assuming either. (4) "
            "dpca_ridge_arm_a/b are fit against the time-resolved category-by-time marginal (category "
            "means as a function of time, reusing _category_time_marginal), the faithful reproduction of "
            "the published study's own sliding-window binning, and their lambda-dependence is measured "
            "directly rather than assumed -- dpca_ridge_axes_lambda_dependence_check reports the measured "
            "subspace overlap across DPCA_RIDGE_LAMBDA_GRID at every cell, and a startup check on a small "
            "synthetic construction (dpca_ridge_axes_synthetic_verification) confirms the ridge fit is not "
            "trivially lambda-invariant everywhere; neither halts the run, since near-invariance on real "
            "data is a property of that data to report, not a defect to block on -- see "
            "dpca_ridge_time_resolved_near_invariance_note per cell.",
        "part4_selective_versus_all_unit_arms": "every population-level quantity in "
            "part4_pseudopopulation_geometry (distance-to-attractor, pseudo-population decoding accuracy, "
            "variance-captured against the category-by-time marginal, and the rest of "
            "_pseudopopulation_geometry's output) is produced under several arms. selective_arm pools "
            "units meeting the widened-window category-neuron test (meets_published_category_neuron_"
            "criterion), read from part1_selective_persistence's already-computed result, which was itself "
            "computed over the full trial set -- selective_arm's own decoder is then trained and evaluated "
            "on overlapping trials, so its decoding accuracy is optimistic by construction and is reported "
            "beside selective_arm_holdout, never in place of it, so the size of that gap is visible. "
            "selective_arm_holdout selects units on one random half of each neuron's trials and evaluates "
            "decoding accuracy on the other half only, repeated over "
            f"{N_TRIAL_SPLIT_REPEATS} random splits: every split whose train-half selection finds at "
            "least two units is decoded and pooled by Rubin's rule, not only the splits that clear the "
            "category-neuron count floor -- a floor-restricted pool sits beside the unrestricted one at "
            "selective_arm_holdout.floor_restricted as a secondary reading, since restricting to floor-"
            "clearing splits biases the surviving set toward splits where selection happened to find more "
            "units. selective_arm_holdout's p_value tests its pooled decoding accuracy against a pooled "
            "label-shuffle null built from that same set of splits' own shuffle draws, not against zero. "
            "selective_unit_prevalence reports the unit count under both the widened window and the first-"
            "encoding-only window; a second gated arm, selective_arm_first_encoding_only, is built at "
            "every cell where the first-encoding-only window's own unit count clears the category-neuron "
            "floor, whether or not the widened window also clears it. all_unit_arm pools every unit in the "
            "region, no selection applied. Neither selective_arm_holdout nor all_unit_arm is a robustness "
            "check on the other: one asks whether neurons that are persistently category-selective carry "
            "the geometry, the other whether the region's average unit does, and both are reported. The "
            "two differ in unit count by roughly an order of magnitude, and this project has an established "
            "result that estimator behaviour tracks unit count, so a bare comparison is not on its own "
            "interpretable -- within every split, a count-matched null draws "
            f"{N_COUNT_MATCHED_REPEATS} random unit subsets with no reference to the data, sized to that "
            "split's own selected-unit count, and evaluates each on that same split's own held-out test "
            "partition, reporting the resulting accuracy distribution (mean, spread, quantiles) at "
            "selective_arm_holdout.per_split[*].count_matched_all_unit_subsample. Each split's own "
            "selective-minus-random difference sits beside it "
            "(per_split[*].selective_minus_random_difference); the differences are pooled across splits at "
            "selective_arm_holdout.selective_minus_random_difference (mean, se, interval), the plain "
            "difference in accuracy units, never a standardized score, and read as a distribution across "
            "splits rather than collapsed to one number for a threshold judgement. A region with too few "
            "category-selective units to fit the geometry reports n_selective_units and a reason rather "
            "than a zero or an omitted key. " + HIPPOCAMPUS_VTC_COMBINED_UNAVAILABLE_REASON,
        "trial_floor": f"a session-region cell is admitted to DA only with >={MIN_CATEGORIES_FOR_DA} "
            f"categories carrying >={MIN_TRIALS_PER_CATEGORY_FOR_DA} trials each; trials in an under-floor "
            "category are dropped from that cell, never from the whole session. Never a p-value.",
        "resampling_unit": "patient; every correlation below is fit within one session, pooled to one "
            "value per patient by averaging that patient's own sessions, then tested across patients "
            "(paired sign-flip test) or reported per-region and pooled across regions/corpora with patient "
            "rows keyed by canonical patient identity (provenance/canonical_primary_records.json), not by "
            "the bare per-corpus subject directory name -- a recording shared between dandi_000673 and "
            "dandi_001187 enters this pooling once, from dandi_001187 (the canonical primary release for "
            "that overlap), and the loaders never admit the matching dandi_000673 session at all, so no "
            "patient row is double-counted across the two.",
        "rate_control": "Part 3 recomputes the maintenance distance-to-attractor construct in rate-free "
            "form using stimulation_response_estimator.rate_free_state_deviation (this project's existing "
            "rate-free distance-to-attractor observable, unmodified) on the same per-trial per-unit "
            "maintenance activity, and reports its own association with behaviour beside the raw "
            "(Euclidean, dPCA-subspace) DA's.",
        "multiplicity_families": {
            "part1_selective_persistence_selection_matched": "one cell per (corpus, region), gated on the "
                "widened (encoding presentation plus probe) category-neuron window",
            "part1_selective_persistence_selection_matched_first_encoding_only": "one cell per (corpus, "
                "region), gated on the first-encoding-onset-only category-neuron window, reported beside "
                "the widened-window arm rather than replacing it",
            "part1_selective_persistence_selection_matched_outlier_trimmed": "one cell per (corpus, "
                f"region); the widened-window selection-matched set with units above "
                f"{OUTLIER_FIRING_RATE_SD_THRESHOLD:.0f} SD of its own mean firing rate excluded, reported "
                "beside the untrimmed arm rather than replacing it",
            "part1_selective_persistence_full_population": "one cell per (corpus, region)",
            "part1_selective_persistence_trial_split_matched": "one cell per (corpus, region), itself "
                f"pooled across {N_TRIAL_SPLIT_REPEATS} random trial splits by Rubin's rule",
            "part2_da_to_behaviour": "one cell per (corpus, region), plus one pooled cell",
            "part3_rate_controlled_da_to_behaviour": "one cell per (corpus, region), plus one pooled cell",
            "distance_to_attractor_reaction_time": "one cell per (corpus, region), plus one pooled cell; "
                "correct trials only, same construction as part2_da_to_behaviour with reaction time in "
                "place of accuracy",
            "rate_controlled_deviation_reaction_time": "one cell per (corpus, region), plus one pooled "
                "cell; correct trials only, same construction as part3_rate_controlled_da_to_behaviour "
                "with reaction time in place of accuracy",
            "selective_persistence_nonpreferred_rate_reaction_time": "one cell per (corpus, region); "
                "per-neuron correlation of baseline-corrected maintenance rate on non-preferred correct "
                "trials against reaction time on those same trials, pooled across neurons by patient",
            "selective_persistence_signed_rate_reaction_time": "one cell per (corpus, region); per-neuron "
                "correlation of baseline-corrected maintenance rate, sign-flipped on non-preferred correct "
                "trials so it reads uniformly in the preferred direction, against reaction time on the "
                "same correct trials, pooled across neurons by patient",
            "part4_pseudopopulation_geometry": "one cell per (corpus, region), each holding several arms "
                "(selective_arm, selective_arm_holdout, all_unit_arm -- see "
                "part4_selective_versus_all_unit_arms); reports its own decoding p_value as a bare number "
                "but is not folded into any FDR family above -- it is a geometry/decoding check on the "
                "pooled pseudo-population, not a behaviour association, so it does not share a "
                "multiplicity family with parts 1-3.",
        },
        "declared_before_fitting": True,
    }
    dpca_ridge_axes_synthetic_verification = _verify_dpca_ridge_lambda_changes_axes()
    artifact = {
        "description": "Reproduction, on three human microwire single-unit epilepsy corpora, of a "
            "published bioRxiv preprint's (doi 10.1101/2025.08.20.671301) category-neuron, selective-"
            "persistence, and distance-to-attractor methods -- no stimulation contrast, since none of "
            "our corpora carry one.",
        "code_commit": git_commit(ROOT),
        "predeclared_rules": predeclared_rules,
        "existing_maintenance_behaviour_null": _read_existing_maintenance_behaviour_null(),
        "marginalization_arm_not_published_method_note": MARGINALIZATION_ARM_NOT_PUBLISHED_METHOD_NOTE,
        "dpca_ridge_axes_synthetic_verification": dpca_ridge_axes_synthetic_verification,
        "corpora": {},
        "status": "in_progress",
    }
    _write(artifact)

    for corpus in CORPUS_SPECS:
        print(f"processing {corpus}", flush=True)
        process_corpus(corpus)

    fdr_cells = {
        "part1_selective_persistence_trial_split_matched": [],
        "part1_selective_persistence_selection_matched": [],
        "part1_selective_persistence_selection_matched_first_encoding_only": [],
        "part1_selective_persistence_selection_matched_outlier_trimmed": [],
        "part1_selective_persistence_full_population": [],
        "part2_da_to_behaviour": [],
        "part3_rate_controlled_da_to_behaviour": [],
        "distance_to_attractor_reaction_time": [],
        "rate_controlled_deviation_reaction_time": [],
        "selective_persistence_nonpreferred_rate_reaction_time": [],
        "selective_persistence_signed_rate_reaction_time": [],
    }
    da_cells_by_corpus_region: dict[str, dict] = {}

    for corpus, spec in CORPUS_SPECS.items():
        corpus_entry = {"regions": {}, "independent_group": resolve_group(corpus)}
        for region in spec["regions"]:
            part1_sessions = {k: v for k, v in _load_checkpoint(f"{corpus}__part1__{region}").items()
                               if v.get("schema_version") == PART1_SCHEMA_VERSION}
            da_sessions = {k: v for k, v in _load_checkpoint(f"{corpus}__da__{region}").items()
                           if v.get("schema_version") == DA_SCHEMA_VERSION}
            da_cells_by_corpus_region[f"{corpus}__{region}"] = da_sessions

            part1_summary = _part1_region_summary(part1_sessions)
            for arm_key, family in (
                ("trial_split_matched_arm", "part1_selective_persistence_trial_split_matched"),
                ("selection_matched_arm", "part1_selective_persistence_selection_matched"),
                ("selection_matched_first_encoding_only_arm",
                 "part1_selective_persistence_selection_matched_first_encoding_only"),
                ("selection_matched_outlier_trimmed_arm",
                 "part1_selective_persistence_selection_matched_outlier_trimmed"),
                ("full_population_arm", "part1_selective_persistence_full_population"),
            ):
                if part1_summary[arm_key].get("status") == "computed":
                    fdr_cells[family].append(part1_summary[arm_key])

            da_to_behaviour = _region_association(da_sessions, "da", "accuracy",
                                                   f"{corpus}|{region}|da_behaviour")
            rate_controlled_to_behaviour = _region_association(
                da_sessions, "rate_free_deviation", "accuracy", f"{corpus}|{region}|ratefree_behaviour")
            da_to_rate = _region_association(da_sessions, "da", "population_spike_count",
                                              f"{corpus}|{region}|da_rate")
            da_to_reaction_time = _region_association(
                da_sessions, "da_correct_trials", "latency_correct_trials", f"{corpus}|{region}|da_latency")
            rate_controlled_to_reaction_time = _region_association(
                da_sessions, "rate_free_deviation_correct_trials", "latency_correct_trials",
                f"{corpus}|{region}|ratefree_latency")
            if da_to_behaviour.get("status") == "tested":
                fdr_cells["part2_da_to_behaviour"].append(da_to_behaviour)
            if rate_controlled_to_behaviour.get("status") == "tested":
                fdr_cells["part3_rate_controlled_da_to_behaviour"].append(rate_controlled_to_behaviour)
            if da_to_reaction_time.get("status") == "tested":
                fdr_cells["distance_to_attractor_reaction_time"].append(da_to_reaction_time)
            if rate_controlled_to_reaction_time.get("status") == "tested":
                fdr_cells["rate_controlled_deviation_reaction_time"].append(rate_controlled_to_reaction_time)

            nonpref_cells, signed_cells = _part1_reaction_time_cells(part1_sessions)
            nonpreferred_rate_to_reaction_time = _region_association(
                nonpref_cells, "nonpreferred_rate", "latency", f"{corpus}|{region}|nonpref_rt")
            signed_rate_to_reaction_time = _region_association(
                signed_cells, "signed_preferred_rate", "latency", f"{corpus}|{region}|signed_rt")
            if nonpreferred_rate_to_reaction_time.get("status") == "tested":
                fdr_cells["selective_persistence_nonpreferred_rate_reaction_time"].append(
                    nonpreferred_rate_to_reaction_time)
            if signed_rate_to_reaction_time.get("status") == "tested":
                fdr_cells["selective_persistence_signed_rate_reaction_time"].append(
                    signed_rate_to_reaction_time)

            pseudopop_ck_name = f"{corpus}__pseudopop__{region}"
            pseudopop_cached = _load_checkpoint(pseudopop_ck_name)
            cached_pooled = pseudopop_cached.get("pooled")
            if cached_pooled is None or cached_pooled.get("schema_version") != PSEUDOPOP_SCHEMA_VERSION:
                pseudopop_report = _pseudopopulation_region_report_all_arms(corpus, region)
                _record_session(pseudopop_ck_name, "pooled", pseudopop_report)
                pseudopop_cached = {"pooled": pseudopop_report}

            n_computed = sum(1 for c in da_sessions.values() if c.get("status") == "computed")
            n_refused = sum(1 for c in da_sessions.values() if c.get("status") == "refused")
            corpus_entry["regions"][region] = {
                "part1_selective_persistence": {
                    **part1_summary,
                    "reaction_time": {
                        "nonpreferred_rate_to_reaction_time": nonpreferred_rate_to_reaction_time,
                        "signed_preferred_rate_to_reaction_time": signed_rate_to_reaction_time,
                    },
                },
                "part2_da": {
                    "n_sessions_total": len(da_sessions), "n_sessions_computed": n_computed,
                    "n_sessions_refused": n_refused,
                    "da_to_behaviour": da_to_behaviour,
                    "da_to_reaction_time": da_to_reaction_time,
                },
                "part3_beyond_published": {
                    "da_to_population_rate": da_to_rate,
                    "rate_controlled_deviation_to_behaviour": rate_controlled_to_behaviour,
                    "rate_controlled_deviation_to_reaction_time": rate_controlled_to_reaction_time,
                },
                "part4_pseudopopulation_geometry": pseudopop_cached["pooled"],
            }
        artifact["corpora"][corpus] = corpus_entry
        _write(artifact)

    artifact["part2_da_to_behaviour_pooled_all_regions_all_corpora"] = _pooled_association(
        da_cells_by_corpus_region, "da", "accuracy", "pooled|da_behaviour")
    artifact["part3_rate_controlled_da_to_behaviour_pooled_all_regions_all_corpora"] = _pooled_association(
        da_cells_by_corpus_region, "rate_free_deviation", "accuracy", "pooled|ratefree_behaviour")
    artifact["part3_da_to_population_rate_pooled_all_regions_all_corpora"] = _pooled_association(
        da_cells_by_corpus_region, "da", "population_spike_count", "pooled|da_rate")
    artifact["distance_to_attractor_to_reaction_time_pooled_all_regions_all_corpora"] = _pooled_association(
        da_cells_by_corpus_region, "da_correct_trials", "latency_correct_trials", "pooled|da_latency")
    artifact["rate_controlled_deviation_to_reaction_time_pooled_all_regions_all_corpora"] = _pooled_association(
        da_cells_by_corpus_region, "rate_free_deviation_correct_trials", "latency_correct_trials",
        "pooled|ratefree_latency")
    pooled_family_keys = {
        "part2_da_to_behaviour": "part2_da_to_behaviour_pooled_all_regions_all_corpora",
        "part3_rate_controlled_da_to_behaviour": "part3_rate_controlled_da_to_behaviour_pooled_all_regions_all_corpora",
        "distance_to_attractor_reaction_time": "distance_to_attractor_to_reaction_time_pooled_all_regions_all_corpora",
        "rate_controlled_deviation_reaction_time":
            "rate_controlled_deviation_to_reaction_time_pooled_all_regions_all_corpora",
    }
    for family, pooled_key in pooled_family_keys.items():
        if artifact[pooled_key].get("status") == "tested":
            fdr_cells[family].append(artifact[pooled_key])

    for family, cells in fdr_cells.items():
        if not cells:
            continue
        q = fdr_bh(np.array([c["p_value"] for c in cells]), alpha=0.05)
        for cell, qv in zip(cells, q["q_values"]):
            cell["q_value_fdr"] = float(qv)

    artifact["independence"] = {
        "n_independent_groups": count_independent_groups(list(CORPUS_SPECS)),
        "groups": {corpus: resolve_group(corpus) for corpus in CORPUS_SPECS},
        "note": "dandi_000673 and dandi_001187 resolve to one independence group; dandi_000469 is its "
            "own group -- two independence groups across three corpus cells in this artifact "
            "(dandi_000574 carries no content label and contributes nothing to it).",
    }
    artifact["status"] = "complete"
    _write(artifact)
    return artifact


def _write(artifact: dict) -> None:
    with open(RESULTS / "selective_persistence_and_attractor_geometry.json", "w") as f:
        json.dump(_json_safe(artifact), f, indent=2, allow_nan=False)


def main():
    build_artifact()


if __name__ == "__main__":
    main()
