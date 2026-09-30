from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from statistics import stable_seed
from statistics import permutation_pvalue, stable_seed, _trial_count_weighted
from subspace_identity import N_CV_FOLDS, _contiguous_folds, leading_eigenvector, _orthonormal_basis
from statistics import SHARP_TEST_MIN_CLASSES, SHARP_TEST_MIN_PER_CLASS
from statistics import MIN_TRIALS_WITH_DEFINED_DIRECTION, unit_direction_vectors
from stimulation_response_estimator import rate_free_state_deviation
from statistics import partial_correlation_permutation_test
from types import SimpleNamespace
from pathlib import Path
import hashlib
from provenance import canonical_json
import os
import tempfile
import neo
import quantities as pq
from geometry import participation_ratio
from statistics import Z_80_POWER
from scipy import stats
from statistics import minimum_detectable_paired_difference, paired_sign_flip_test
from subspace_identity import block_folds, class_basis, regression_basis
from geometry import principal_angles, subspace_overlap
from geometry import content_decoding_dropping_latent
from statistics import CONTENT_N_PERM_FULL
from spike_pipeline import FrozenPSTHTransform, BIN_MS
from observability import _leading_latent_projection
from state_persistence import _window_means_at_width, rank1_gain_and_residual, rank1_gain_share, temporal_profile_sign_crossings
from control import energy_accuracy_pareto, stimulation_input_alignment
from geometry import pca_decompose
from spike_pipeline import crop_trial
from run_macaque_pfc_microstimulation_pipeline import load_macaque_pfc_microstimulation_session as load_causal_microstim_session


Decoder = Literal["linear", "nonlinear"]
Fold = tuple[NDArray[np.int_], NDArray[np.int_]]


def aggregate_seed_records(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        if record.get("status") != "computed":
            continue
        key = tuple(record.get(name, "") for name in ("corpus", "session", "level", "candidate", "decoder"))
        groups.setdefault(key, []).append(record)
    aggregated = []
    for key, group in sorted(groups.items()):
        row = {name: value for name, value in zip(("corpus", "session", "level", "candidate", "decoder"), key)}
        row["status"] = "computed"
        row["n_seeds"] = len(group)
        for metric in ("same_time", "cross_temporal"):
            values = np.asarray([r[metric]["auc"] for r in group if metric in r and r[metric].get("auc") is not None], dtype=float)
            nulls = np.asarray([r[metric]["null_auc"] for r in group if metric in r and r[metric].get("null_auc") is not None], dtype=float)
            if values.size:
                row[metric] = {"status": "computed", "auc": float(values.mean()), "null_auc": float(nulls.mean()) if nulls.size else None,
                               "auc_above_null": float(values.mean() - nulls.mean()) if nulls.size else None}
            else:
                row[metric] = {"status": "not_computable"}
        aggregated.append(row)
    return aggregated


def matched_method_contrast(
    records: Sequence[dict], candidate: str, reference: str = "native_full_rank",
    *, metric: str = "cross_temporal", decoder: str | None = None, n_bootstrap: int = 5000, seed: int = 0,
) -> dict:
    cells = aggregate_seed_records(records)
    if decoder is not None:
        cells = [r for r in cells if r["decoder"] == decoder]
    indexed = {(r["corpus"], r["session"], r["level"], r["decoder"], r["candidate"]): r for r in cells}
    keys = sorted({k[:4] for k in indexed if k[4] == candidate} & {k[:4] for k in indexed if k[4] == reference})
    cell_differences = np.asarray([indexed[(*k, candidate)][metric]["auc"] - indexed[(*k, reference)][metric]["auc"] for k in keys], dtype=float)
    sessions = sorted({k[:2] for k in keys})
    differences = np.asarray([
        cell_differences[[k[:2] == session for k in keys]].mean()
        for session in sessions
    ], dtype=float)
    if not differences.size:
        return {"status": "not_computable", "candidate": candidate, "reference": reference, "n_cells": 0}
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, differences.size, size=(max(1, n_bootstrap), differences.size))].mean(axis=1)
    return {"status": "computed", "candidate": candidate, "reference": reference, "metric": metric,
            "estimate": float(differences.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_cells": int(cell_differences.size), "n_sessions": int(differences.size), "cell_keys": [list(k) for k in keys], "cell_differences": cell_differences.tolist(),
            "session_keys": [list(session) for session in sessions],
            "session_differences": differences.tolist(), "independent_unit": "recording_session", "n_bootstrap": int(max(1, n_bootstrap)),
            "decoder": decoder}


def make_decoder(kind: Decoder = "linear", seed: int = 0) -> Pipeline:
    if kind == "linear":
        model = LogisticRegression(
            C=1.0, max_iter=2_000, solver="lbfgs", random_state=seed
        )
    elif kind == "nonlinear":
        model = SVC(
            C=1.0,
            kernel="rbf",
            decision_function_shape="ovr",
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown decoder: {kind}")
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def make_folds(labels: ArrayLike, n_splits: int = 5, seed: int = 0) -> tuple[Fold, ...]:
    y = _labels(labels)
    counts = np.unique(y, return_counts=True)[1]
    if n_splits < 2 or n_splits > counts.min():
        raise ValueError("n_splits must be between 2 and the smallest class count")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return tuple((train, test) for train, test in splitter.split(np.zeros(y.size), y))


def split_auc(
    train_activity: ArrayLike,
    train_labels: ArrayLike,
    test_activity: ArrayLike,
    test_labels: ArrayLike,
    *,
    decoder: Decoder = "linear",
    seed: int = 0,
) -> NDArray[np.float64]:
    train_x, train_y, test_x, test_y = _split_inputs(
        train_activity, train_labels, test_activity, test_labels
    )
    return _split_score(train_x, train_y, test_x, test_y, decoder, seed, False)


def temporal_auc(
    activity: ArrayLike,
    labels: ArrayLike,
    *,
    decoder: Decoder = "linear",
    folds: Sequence[Fold] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> NDArray[np.float64]:
    x, y, fixed_folds = _inputs(activity, labels, folds, n_splits, seed)
    return _score(x, y, fixed_folds, decoder, seed, same_time=False)


def same_time_auc(
    activity: ArrayLike,
    labels: ArrayLike,
    *,
    decoder: Decoder = "linear",
    folds: Sequence[Fold] | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> NDArray[np.float64]:
    x, y, fixed_folds = _inputs(activity, labels, folds, n_splits, seed)
    return _score(x, y, fixed_folds, decoder, seed, same_time=True)


def permutation_null(
    activity: ArrayLike,
    labels: ArrayLike,
    *,
    decoder: Decoder = "linear",
    folds: Sequence[Fold] | None = None,
    n_splits: int = 5,
    n_permutations: int = 100,
    seed: int = 0,
    same_time: bool = False,
) -> NDArray[np.float64]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    x, y, fixed_folds = _inputs(activity, labels, folds, n_splits, seed)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_permutations):
        shuffled = _stratified_permutation(y, fixed_folds, rng)
        permuted = tuple(
            (shuffled[train], shuffled[test])
            for train, test in fixed_folds
        )
        scores.append(_score(x, y, fixed_folds, decoder, seed, same_time, permuted))
    return np.stack(scores)


def _stratified_permutation(
    labels: NDArray,
    folds: Sequence[Fold],
    rng: np.random.Generator,
) -> NDArray:
    shuffled = np.empty_like(labels)
    for _, test in folds:
        shuffled[test] = rng.permutation(labels[test])
    return shuffled


def _inputs(
    activity: ArrayLike,
    labels: ArrayLike,
    folds: Sequence[Fold] | None,
    n_splits: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray, tuple[Fold, ...]]:
    x = np.asarray(activity, dtype=float)
    y = _labels(labels)
    if x.ndim != 3:
        raise ValueError("activity must have shape (trials, time, features)")
    if x.shape[0] != y.size:
        raise ValueError("activity and labels must have the same number of trials")
    if x.shape[1] < 1 or x.shape[2] < 1 or not np.isfinite(x).all():
        raise ValueError("activity must be non-empty and finite")
    fixed_folds = make_folds(y, n_splits, seed) if folds is None else _folds(folds, y)
    return x, y, fixed_folds


def _split_inputs(
    train_activity: ArrayLike,
    train_labels: ArrayLike,
    test_activity: ArrayLike,
    test_labels: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray, NDArray[np.float64], NDArray]:
    train_x = np.asarray(train_activity, dtype=float)
    test_x = np.asarray(test_activity, dtype=float)
    train_y = _labels(train_labels)
    test_y = _labels(test_labels)
    if train_x.ndim != 3 or test_x.ndim != 3:
        raise ValueError("activity must have shape (trials, time, features)")
    if train_x.shape[0] != train_y.size or test_x.shape[0] != test_y.size:
        raise ValueError("activity and labels must have the same number of trials")
    if train_x.shape[1:] != test_x.shape[1:]:
        raise ValueError("train and test activity must share time and feature dimensions")
    if train_x.shape[1] < 1 or train_x.shape[2] < 1:
        raise ValueError("activity must be non-empty")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("activity must be finite")
    if not np.array_equal(np.unique(train_y), np.unique(test_y)):
        raise ValueError("train and test labels must contain the same classes")
    return train_x, train_y, test_x, test_y


def _labels(labels: ArrayLike) -> NDArray:
    y = np.asarray(labels)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("labels must be a one-dimensional array")
    if np.unique(y).size < 2:
        raise ValueError("labels must contain at least two classes")
    return y


def _folds(folds: Sequence[Fold], labels: NDArray) -> tuple[Fold, ...]:
    checked = tuple(
        (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
        for train, test in folds
    )
    if len(checked) < 2:
        raise ValueError("folds must contain at least two train-test pairs")
    seen = np.zeros(labels.size, dtype=int)
    classes = np.unique(labels)
    for train, test in checked:
        if train.ndim != 1 or test.ndim != 1 or train.size == 0 or test.size == 0:
            raise ValueError("fold indices must be non-empty one-dimensional arrays")
        if (
            train.min() < 0
            or test.min() < 0
            or train.max() >= labels.size
            or test.max() >= labels.size
        ):
            raise ValueError("fold index is out of bounds")
        if np.intersect1d(train, test).size:
            raise ValueError("train and test indices must not overlap")
        combined = np.concatenate((train, test))
        if combined.size != labels.size or np.unique(combined).size != labels.size:
            raise ValueError("each train-test pair must partition the trials")
        if not np.array_equal(np.unique(labels[train]), classes):
            raise ValueError("each training fold must contain every class")
        if not np.array_equal(np.unique(labels[test]), classes):
            raise ValueError("each test fold must contain every class")
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("test folds must partition the trials")
    return checked


def _score(
    activity: NDArray[np.float64],
    labels: NDArray,
    folds: tuple[Fold, ...],
    decoder: Decoder,
    seed: int,
    same_time: bool,
    fold_labels: tuple[tuple[NDArray, NDArray], ...] | None = None,
) -> NDArray[np.float64]:
    fold_scores = []
    base = make_decoder(decoder, seed) if same_time else None
    for fold_id, (train, test) in enumerate(folds):
        y_train, y_test = (
            (labels[train], labels[test]) if fold_labels is None else fold_labels[fold_id]
        )
        if same_time:
            fold_scores.append(
                _split_score(
                    activity[train],
                    y_train,
                    activity[test],
                    y_test,
                    decoder,
                    seed,
                    True,
                    base,
                )
            )
        else:
            fold_scores.append(
                split_auc(
                    activity[train],
                    y_train,
                    activity[test],
                    y_test,
                    decoder=decoder,
                    seed=seed,
                )
            )
    return np.mean(fold_scores, axis=0)


def _split_score(
    train_activity: NDArray[np.float64],
    train_labels: NDArray,
    test_activity: NDArray[np.float64],
    test_labels: NDArray,
    decoder: Decoder,
    seed: int,
    same_time: bool,
    base: Pipeline | None = None,
) -> NDArray[np.float64]:
    n_times = train_activity.shape[1]
    scores = np.empty(n_times if same_time else (n_times, n_times), dtype=float)
    base = make_decoder(decoder, seed) if base is None else base
    for train_time in range(n_times):
        model = clone(base).fit(train_activity[:, train_time], train_labels)
        if same_time:
            scores[train_time] = _auc(
                model, test_activity[:, train_time], test_labels
            )
        else:
            for test_time in range(n_times):
                scores[train_time, test_time] = _auc(
                    model, test_activity[:, test_time], test_labels
                )
    return scores


def _auc(model: Pipeline, features: NDArray[np.float64], labels: NDArray) -> float:
    scores = model.decision_function(features)
    classes = model.classes_
    if classes.size == 2:
        return float(roc_auc_score(labels, scores, labels=classes))
    targets = label_binarize(labels, classes=classes)
    return float(roc_auc_score(targets, scores, average="macro"))


REPRODUCTION_TOLERANCE = 1e-9


CANDIDATE_KEYS = (
    "memorandum_content", "upcoming_response", "gain_total_spike_count", "time_in_trial_within_delay",
    "previous_trial_content", "field_potential_low_frequency_band_power_ieeg",
    "field_potential_low_frequency_band_power_eeg", "field_potential_aperiodic_slope_ieeg",
    "field_potential_aperiodic_slope_eeg",
)


CANDIDATE_SUPPORT_MATRIX = {
    "panichello_2024_macaque_lPFC": {
        "memorandum_content": ("present", "cueAngIdx, the discretised cued-position label, native to this corpus's raw session file"),
        "upcoming_response": ("absent", "single-item delayed-saccade task: the response target IS the remembered location (cueAngIdx), already tested as memorandum_content; no separable response label exists"),
        "gain_total_spike_count": ("present", None), "time_in_trial_within_delay": ("present", None),
        "previous_trial_content": ("present", "positional shift of cueAngIdx"),
        "field_potential_low_frequency_band_power_ieeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_low_frequency_band_power_eeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_ieeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_eeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
    },
    "watters_2026_macaque_multi_object": {
        "memorandum_content": ("present", "cued_theta [cos, sin], the continuous cued position, native to this corpus"),
        "upcoming_response": ("absent", "the report is a continuous saccade whose target is the memorandum itself by task design (same cued position determines both); no decision epoch separable from the delay-epoch tensor this atlas uses exists in the cached data"),
        "gain_total_spike_count": ("present", None), "time_in_trial_within_delay": ("present", None),
        "previous_trial_content": ("present", "positional shift of cued_theta"),
        "field_potential_low_frequency_band_power_ieeg": ("absent", "this corpus ships single-unit and multi-unit spikes only, no co-registered field potential"),
        "field_potential_low_frequency_band_power_eeg": ("absent", "this corpus ships single-unit and multi-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_ieeg": ("absent", "this corpus ships single-unit and multi-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_eeg": ("absent", "this corpus ships single-unit and multi-unit spikes only, no co-registered field potential"),
    },
    "inagaki_alm5_mouse_ALM": {
        "memorandum_content": ("absent", "motor-planning task: the instructed lick direction IS the memorandum (no stimulus identity distinct from the planned response), so this candidate would be a byte-for-byte duplicate of upcoming_response; reported once, under upcoming_response, the name used for this corpus's response subspace specifically"),
        "upcoming_response": ("present", "the trial's instructed lick direction (0=left/1=right), native condition label this project's own iter_alm already carries"),
        "gain_total_spike_count": ("present", None), "time_in_trial_within_delay": ("present", None),
        "previous_trial_content": ("present", "positional shift of the instructed lick direction, over the control-trial sequence only"),
        "field_potential_low_frequency_band_power_ieeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_low_frequency_band_power_eeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_ieeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_eeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
    },
}


MAX_SESSIONS_ENV_VAR = "COMPONENT_IDENTITY_ATLAS_MAX_SESSIONS"


N_BOOT_SESSION_CLUSTER = 2000


def _previous_label(full_labels: np.ndarray) -> np.ndarray:
    """Positional shift by one row over this session's OWN trial order (before any kept-trial
    subsetting) -- trial i's 'previous-trial content' is trial (i-1)'s own label. For every human
    corpus and the mouse ALM corpus this is the previous ADMITTED trial (the corpus's own admission
    mask may have already excluded some trials), not necessarily the literal preceding trial number;
    stated once here rather than at every call site."""
    full_labels = np.asarray(full_labels, dtype=float)
    prev = np.full_like(full_labels, np.nan)
    if len(full_labels) > 1:
        prev[1:] = full_labels[:-1]
    return prev


MIN_TRIALS_PER_CLASS = 5


MIN_TRIALS_PER_CLASS_COMPARISON = 15


def _linear_detrend_activity(activity_by_unit: np.ndarray, trial_index: np.ndarray) -> np.ndarray:
    n = activity_by_unit.shape[0]
    t_centered = trial_index - trial_index.mean()
    design = np.column_stack([np.ones(n), t_centered])
    coef, *_ = np.linalg.lstsq(design, activity_by_unit, rcond=None)
    slope = coef[1]
    return activity_by_unit - np.outer(t_centered, slope)


def _vector_reference_alignment_with_draws(a: np.ndarray, ref: np.ndarray | None, n_draws: int, seed_tag: str) -> dict | None:
    if ref is None:
        return None
    n_units = a.shape[0]
    observed = abs(float(np.dot(a, ref)))
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.abs(g @ ref)
    return {"observed": observed, "draws": draws}


CORPORA = ("panichello_2024_macaque_lPFC_single_item", "watters_2026_macaque_multi_object")


MIN_FOLD_TRIALS = 8


N_PERM = 10000


N_RANDOM_AXIS_DRAWS = 200


N_ROTATION_DRAWS = 1000


RESIDUAL_NORM_FLOOR = 1e-8


def _cv_pca_rank(U: np.ndarray, max_k: int, seed_tag: str) -> dict:
    n = U.shape[0]
    if max_k < 1 or n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "not_estimable"}
    folds = _contiguous_folds(n, N_CV_FOLDS)
    candidate_ks = list(range(1, max_k + 1))
    errors = np.zeros(len(candidate_ks))
    n_folds_used = 0
    for f in range(N_CV_FOLDS):
        train, test = folds != f, folds == f
        if int(train.sum()) < 4 or not test.any():
            continue
        mean_train = U[train].mean(axis=0)
        _u_svd, _s_svd, vt_svd = np.linalg.svd(U[train] - mean_train, full_matrices=False)
        test_c = U[test] - mean_train
        n_folds_used += 1
        for i, k in enumerate(candidate_ks):
            k_eff = min(k, vt_svd.shape[0])
            basis = vt_svd[:k_eff].T
            proj = test_c @ basis @ basis.T
            errors[i] += float(np.sum((test_c - proj) ** 2))
    if n_folds_used == 0:
        return {"status": "not_estimable"}
    best_idx = int(np.argmin(errors))
    return {"status": "computed", "n_folds_used": n_folds_used, "candidate_ks": candidate_ks,
            "cv_reconstruction_error": errors.tolist(), "best_k": candidate_ks[best_idx]}


def _axis_stability(R: np.ndarray, seed_tag: str) -> dict:
    n, n_units = R.shape
    if n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "too_few_trials", "n_trials": n}
    folds = _contiguous_folds(n, N_CV_FOLDS)
    fold_axes = [leading_eigenvector(R[folds == f]) if int((folds == f).sum()) >= MIN_FOLD_TRIALS else None
                 for f in range(N_CV_FOLDS)]
    pairs = [(i, j) for i in range(N_CV_FOLDS) for j in range(i + 1, N_CV_FOLDS)
             if fold_axes[i] is not None and fold_axes[j] is not None]
    if len(pairs) < 3:
        return {"status": "too_few_fold_pairs", "n_fold_pairs": len(pairs)}
    observed = float(np.mean([abs(float(np.dot(fold_axes[i], fold_axes[j]))) for i, j in pairs]))
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.empty(N_ROTATION_DRAWS)
    for d in range(N_ROTATION_DRAWS):
        vecs = rng.standard_normal((2 * len(pairs), n_units))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        draws[d] = float(np.mean(np.abs(np.sum(vecs[0::2] * vecs[1::2], axis=1))))
    null_mean = float(np.mean(draws))
    p = float(permutation_pvalue(draws >= observed))
    return {
        "status": "computed", "observed_mean_abs_cosine": observed, "n_fold_pairs": len(pairs),
        "null_mean": null_mean, "null_sd": float(np.std(draws)),
        "one_sided_p_value_stability_above_chance": p, "stable": bool(p <= 0.05),
    }


def _empirical_two_sided(observed: float, draws: np.ndarray) -> dict:
    finite = draws[np.isfinite(draws)]
    if finite.size < 10:
        return {"status": "not_computable"}
    null_mean = float(np.mean(finite))
    p = float(permutation_pvalue(np.abs(finite - null_mean) >= abs(observed - null_mean)))
    return {
        "status": "computed", "observed": observed, "null_mean": null_mean, "null_sd": float(np.std(finite)),
        "n_draws": int(finite.size), "two_sided_p_value": p, "significant": bool(p <= 0.05 and observed > null_mean),
    }


def _occupied_space_decomposition(U: np.ndarray, axis: np.ndarray, k: int, n_draws: int, seed_tag: str) -> dict:
    n, p = U.shape
    k_eff = min(k, min(n, p))
    if k_eff < 1 or np.sum(axis ** 2) <= 0.0:
        return {"status": "not_computable"}
    mean_u = U.mean(axis=0)
    _u_svd, _s_svd, vt = np.linalg.svd(U - mean_u, full_matrices=False)
    k_eff = min(k_eff, vt.shape[0])
    basis = vt[:k_eff].T
    within = basis @ (basis.T @ axis)
    within_frac = float(np.sum(within ** 2) / np.sum(axis ** 2))
    off_frac = 1.0 - within_frac
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, p))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    w = g @ basis
    null_off = 1.0 - np.sum(w ** 2, axis=1)
    null_mean = float(np.mean(null_off))
    p_value = float(permutation_pvalue(np.abs(null_off - null_mean) >= abs(off_frac - null_mean)))
    null_within = 1.0 - null_off
    null_within_ci95 = [float(v) for v in np.percentile(null_within, [2.5, 97.5])]
    return {
        "status": "computed", "k": k_eff, "within_fraction": within_frac, "off_fraction": off_frac,
        "null_off_fraction_mean": null_mean, "null_off_fraction_sd": float(np.std(null_off)),
        "null_within_fraction_mean": float(np.mean(null_within)),
        "null_within_fraction_ci95pct": null_within_ci95,
        "within_fraction_minus_null_mean": float(within_frac - np.mean(null_within)),
        "two_sided_p_value": p_value, "off_fraction_above_null": bool(p_value <= 0.05 and off_frac > null_mean),
        # kept for pooling across sessions; stripped from the stored per-level furniture
        "null_off_fraction_draws": null_off,
    }


def _regression_direction(U: np.ndarray, covariate: np.ndarray) -> np.ndarray | None:
    n = U.shape[0]
    design = np.column_stack([np.ones(n), covariate])
    coef, *_ = np.linalg.lstsq(design, U, rcond=None)
    beta = coef[1]
    norm = np.linalg.norm(beta)
    return (beta / norm) if norm > 0 else None


def _unit_residual_matrix(rows: dict) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(rows["keep"])
    R = rows["residual"][idx] / rows["residual_norm"][idx, None]
    return R, idx


def _weighted_combine_draws(entries: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    """Trial-count-weighted combination of per-level null-draw arrays into one session-level array, the
    same weighting _trial_count_weighted applies to scalars, applied here draw-index by draw-index."""
    if not entries:
        return None
    n_arr = np.array([n for n, _ in entries], dtype=float)
    stack = np.array([d for _, d in entries], dtype=float)
    valid = np.isfinite(stack)
    weights = n_arr[:, None] * valid
    denom = weights.sum(axis=0)
    numer = np.nansum(np.where(valid, stack, 0.0) * n_arr[:, None], axis=0)
    return np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)


def _worse_behaviour(bundle: dict, sign: float) -> np.ndarray:
    """A literal worse-coded outcome array (higher = worse), for the signed-displacement test's median-split sign rule --
    distinct from the correlation-coefficient sign flip SIGN_TO_WORSE_BEHAVIOUR applies elsewhere."""
    return bundle["outcome_raw"] if sign == 1.0 else (1.0 - bundle["outcome_raw"])


def classify_occupied_space_branch(comparison: dict) -> str:
    """Pre-declared mapping from the pooled observed-versus-matched-null comparison to one of the three
    named branches; see OCCUPIED_SPACE_DECISION_RULE_DECLARED_BEFORE_FITTING."""
    if comparison.get("status") != "computed":
        return "not_separable_at_the_available_dimensionality"
    if comparison["significant_above_null"]:
        return "the_axis_lies_outside_the_occupied_state_space"
    return "the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace"


def pooled_off_fraction_against_matched_null(computed_sessions: list[dict]) -> dict:
    """Pools each session's own matched-random-axis null draws -- per session first combined across
    item-count levels by trial-count weighting, draw-index aligned; across sessions by their mean,
    draw-index aligned (the same pooling convention the anisotropy block applies to its rotation nulls)
    -- and compares the pooled observed off-fraction (mean across sessions) against that pooled null's
    central mass by a two-sided empirical percentile test. The observed off-fraction is never tested
    against zero: a random axis of the same dimension has a nonzero off-occupied fraction by
    construction, so only the matched null separates 'leaves the occupied space' from 'does not'."""
    off_fracs = [s["off_fraction"] for s in computed_sessions if s.get("off_fraction") is not None]
    draws_list = [np.asarray(s["null_draws"], dtype=float) for s in computed_sessions
                  if s.get("null_draws") is not None]
    out = {"status": "not_computable", "n_sessions": len(off_fracs),
           "n_sessions_with_a_null_distribution": len(draws_list),
           "pooled_observed_off_fraction": None, "pooled_null_mean": None, "pooled_null_sd": None,
           "two_sided_empirical_p_value": None, "above_null": None,
           "significant_above_null": False, "significant_below_null": False}
    if not draws_list or not off_fracs or len(off_fracs) != len(draws_list):
        return out
    pooled_null = np.nanmean(np.stack(draws_list), axis=0)
    finite = pooled_null[np.isfinite(pooled_null)]
    if finite.size < 10:
        return out
    null_centre = float(np.mean(finite))
    observed_pooled = float(np.mean(off_fracs))
    p_value = float(permutation_pvalue(np.abs(finite - null_centre) >= abs(observed_pooled - null_centre)))
    above = observed_pooled > null_centre
    out.update({
        "status": "computed", "pooled_observed_off_fraction": observed_pooled,
        "pooled_null_mean": null_centre, "pooled_null_sd": float(np.std(finite)),
        "two_sided_empirical_p_value": p_value, "above_null": bool(above),
        "significant_above_null": bool(p_value <= 0.05 and above),
        "significant_below_null": bool(p_value <= 0.05 and not above),
    })
    return out


def _combine_restated_cells_across_levels(level_cells: list[tuple[int, dict]], n_perm: int) -> dict:
    """Trial-count-weighted combination of per-item-count-level restatement cells into one
    session-level cell. The predictable fraction and its null mean are weighted-averaged the same way
    this corpus's own primary behavioural estimator combines any within-level statistic across levels;
    the null distribution itself is combined the identical draw-index-weighted way the delivered
    rotation-null machinery already uses, so the combined p-value is judged against a null built the
    same way as the combined effect -- never a p-value picked from whichever level happened to have
    the smallest one."""
    computed = [(n, c) for n, c in level_cells if c.get("status") == "computed"]
    if not computed:
        return {"status": "not_computable", "reason": "no item-count level reached the trial floor"}
    frac = _trial_count_weighted([(n, c["predictable_fraction"]) for n, c in computed])
    null_mean = _trial_count_weighted([(n, c["null_mean"]) for n, c in computed])
    padded = []
    for n, c in computed:
        vals = list(c.get("null_values", []))
        vals = vals + [float("nan")] * (n_perm - len(vals))
        padded.append((n, np.asarray(vals[:n_perm], dtype=float)))
    pooled_null = _weighted_combine_draws(padded)
    pooled_null = pooled_null[np.isfinite(pooled_null)] if pooled_null is not None else np.array([])
    p_value = permutation_pvalue(pooled_null >= frac) if len(pooled_null) else None
    return {"status": "computed", "predictable_fraction": float(frac), "null_mean": float(null_mean),
            "effect_size": float(frac - null_mean), "p_value": p_value,
            "n_pooled_null_draws": int(len(pooled_null)), "n_levels_combined": len(computed),
            "n_trials": int(sum(n for n, _ in computed))}


def zero_drop(records_by_candidate: dict[str, list[dict]]) -> dict:
    per_candidate = {}
    for candidate, recs in sorted(records_by_candidate.items()):
        statuses: dict[str, int] = {}
        exclusions = []
        for rec in recs:
            st = rec.get("status", "missing")
            statuses[st] = statuses.get(st, 0) + 1
            if st in ("excluded", "fit_failed"):
                exclusions.append({"session_key": rec.get("session_key"), "reason": rec.get("reason")})
        per_candidate[candidate] = {
            "n_seen": len(recs), "statuses": statuses,
            "seen_equals_tested_plus_excluded": len(recs) == sum(statuses.values()),
            "exclusions_with_reasons": exclusions}
    return per_candidate


def _class_mean_dict(U: np.ndarray, labels: np.ndarray) -> dict | None:
    finite_label = np.isfinite(labels)
    if not finite_label.any():
        return None
    classes = np.unique(labels[finite_label])
    class_members = {c: np.flatnonzero((labels == c) & finite_label) for c in classes}
    class_mean = {c: U[idx].mean(axis=0) for c, idx in class_members.items() if len(idx) >= SHARP_TEST_MIN_PER_CLASS}
    return class_mean if len(class_mean) >= SHARP_TEST_MIN_CLASSES else None


IDENTITY_TOLERANCE = 1e-8


WATTERS_RECOVERABILITY_K_CLASSES = 8


WATTERS_REGRESSION_DIM = 2


def _leave_one_out_unit_directions(activity_by_unit: np.ndarray) -> dict:
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials = activity.shape[0]
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors, axis=0)
    n_valid = int(valid.sum())

    loo_mean_normalized = np.full_like(unit_vectors, np.nan)
    for i in range(n_trials):
        if not valid[i]:
            continue
        n_other = n_valid - 1
        if n_other < 1:
            continue
        loo_mean = (total - unit_vectors[i]) / n_other
        loo_norm = np.linalg.norm(loo_mean)
        if loo_norm == 0.0:
            continue
        loo_mean_normalized[i] = loo_mean / loo_norm
    return {"unit_vectors": unit_vectors, "loo_mean_normalized": loo_mean_normalized, "valid": valid}


def cv_regression_subspace(u: np.ndarray, target_2d: np.ndarray, residual: np.ndarray, dim: int = WATTERS_REGRESSION_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Multi-object arm: for each trial, fit the dim-dimensional regression
    subspace (each unit's direction regressed on the target's coordinates)
    from every OTHER trial, project that trial's own residual onto it."""
    n = u.shape[0]
    within = np.full(n, np.nan)
    outside = np.full(n, np.nan)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        x_held_in, u_held_in = target_2d[keep], u[keep]
        x_centred = x_held_in - x_held_in.mean(axis=0)
        u_centred = u_held_in - u_held_in.mean(axis=0)
        coefficients, *_ = np.linalg.lstsq(x_centred, u_centred, rcond=None)  # (dim, units)
        basis = _orthonormal_basis(coefficients, dim)
        proj = basis @ (basis.T @ residual[i])
        within[i] = np.linalg.norm(proj)
        outside[i] = np.linalg.norm(residual[i] - proj)
    return within, outside


def residual_decomposition_and_identity_check(activity_by_unit: np.ndarray) -> dict:
    """Residual r_i for every trial, plus the decomposition-identity proof:
    the deviation recomputed from this module's own u_i and m_i must match
    rate_free_state_deviation's delivered output to floating-point tolerance
    on every finite trial of this session."""
    directions = _leave_one_out_unit_directions(activity_by_unit)
    unit_vectors, loo_mean, valid = directions["unit_vectors"], directions["loo_mean_normalized"], directions["valid"]
    delivered = rate_free_state_deviation(activity_by_unit)
    finite_delivered = np.isfinite(delivered)

    with np.errstate(invalid="ignore"):
        cosine = np.einsum("ij,ij->i", unit_vectors, loo_mean)
    recomputed = 1.0 - cosine
    finite_recomputed = np.isfinite(recomputed)
    same_finite_mask = bool(np.array_equal(finite_delivered, finite_recomputed))
    diffs = (np.abs(delivered[finite_delivered] - recomputed[finite_delivered])
             if finite_delivered.any() else np.array([0.0]))
    max_abs_diff = float(np.max(diffs)) if diffs.size else 0.0

    with np.errstate(invalid="ignore"):
        residual = unit_vectors - cosine[:, None] * loo_mean

    return {
        "residual": residual, "deviation_recomputed": recomputed, "deviation_delivered": delivered,
        "cosine": cosine, "finite": finite_delivered,
        "identity_same_finite_mask": same_finite_mask, "identity_max_abs_diff": max_abs_diff,
        "identity_passed": bool(same_finite_mask and max_abs_diff < IDENTITY_TOLERANCE),
    }


def _residual_rows(activity_by_unit: np.ndarray, floor: float = RESIDUAL_NORM_FLOOR) -> dict:
    identity = residual_decomposition_and_identity_check(activity_by_unit)
    loo_mean = _leave_one_out_unit_directions(activity_by_unit)["loo_mean_normalized"]
    finite = identity["finite"]
    residual = identity["residual"]
    residual_norm = np.linalg.norm(residual, axis=1)
    with np.errstate(invalid="ignore"):
        above_floor = finite & (residual_norm >= floor)
    return {
        "identity": identity, "loo_mean": loo_mean, "residual": residual, "residual_norm": residual_norm,
        "keep": above_floor, "n_trials_with_defined_direction": int(finite.sum()),
        "n_trials_excluded_by_residual_floor": int((finite & ~above_floor).sum()),
        "n_kept": int(above_floor.sum()),
    }


def _collect_axis_entries(bundles: list[dict], corpus_key: str) -> dict:
    """Rebuilds the identical (R, U) pair the anisotropy test used, per session (per item-count level for
    the multi-object corpus), for the occupied-state-space decomposition -- recomputed rather than threaded
    through the checkpoint, since these arrays are not JSON-safe and the anisotropy test's own checkpoint
    intentionally keeps only its scalar summaries."""
    out: dict[str, list[dict]] = {}
    for bundle in bundles:
        session = bundle["session"]
        entries = []
        if corpus_key == CORPORA[0]:
            rows = _residual_rows(bundle["activity_by_unit"])
            if rows["n_kept"] >= MIN_TRIALS_WITH_DEFINED_DIRECTION:
                R, idx = _unit_residual_matrix(rows)
                U = unit_direction_vectors(bundle["activity_by_unit"])[idx]
                entries.append({"n_trials": int(R.shape[0]), "R": R, "U": U, "level": "all"})
        else:
            item_count = bundle["item_count"]
            for level in sorted({int(v) for v in item_count.tolist()}):
                mask = item_count == float(level)
                if int(mask.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
                    continue
                rows = _residual_rows(bundle["activity_by_unit"][mask])
                if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
                    continue
                R, idx = _unit_residual_matrix(rows)
                U = unit_direction_vectors(bundle["activity_by_unit"][mask])[idx]
                entries.append({"n_trials": int(R.shape[0]), "R": R, "U": U, "level": str(level)})
        out[session] = entries
    return out


def _session_core(activity_by_unit: np.ndarray) -> dict | None:
    """Recovers this session's deviation scalar, axis, and kept-trial unit-direction matrix, using
    only imported, unmodified project functions (see DEVIATION_AXIS_SOURCE_QUOTE)."""
    rows = _residual_rows(activity_by_unit)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, idx = _unit_residual_matrix(rows)
    if R.shape[1] < 2:
        return None
    axis = leading_eigenvector(R)
    directions = _leave_one_out_unit_directions(activity_by_unit)
    U = directions["unit_vectors"][idx]
    deviation = rate_free_state_deviation(activity_by_unit)
    spike_count = np.nansum(activity_by_unit, axis=1)
    return {
        "idx": idx, "R": R, "axis": axis, "U": U, "deviation": deviation, "spike_count": spike_count,
        "identity_passed": bool(rows["identity"]["identity_passed"]),
        "identity_max_abs_diff": rows["identity"]["identity_max_abs_diff"],
        "n_kept": int(rows["n_kept"]), "n_trials_total": int(activity_by_unit.shape[0]),
        "n_units": int(activity_by_unit.shape[1]),
    }


def _fold_combined_and_pooled(feature: np.ndarray, outcome: np.ndarray, folds: np.ndarray, seed_tag: str) -> dict:
    valid = np.isfinite(feature) & np.isfinite(outcome)
    if int(valid.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials"}
    pooled = partial_correlation_permutation_test(
        outcome[valid], feature[valid], [], N_PERM, np.random.default_rng(stable_seed(f"{seed_tag}|pooled")))
    per_fold = []
    for f in range(N_CV_FOLDS):
        test = (folds == f) & valid
        if int(test.sum()) < 4:
            continue
        r = _pearson_r(feature[test], outcome[test])
        if r is not None:
            per_fold.append((int(test.sum()), r))
    within_fold = _trial_count_weighted(per_fold) if per_fold else None
    return {
        "status": "computed", "pooled_across_fold": pooled,
        "within_fold_trial_count_weighted_r": within_fold, "n_folds_contributing": len(per_fold),
    }


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 4 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def _decoder_api() -> SimpleNamespace:
    from info_decoding import _stratified_permutation, make_folds, split_auc

    return SimpleNamespace(
        make_folds=make_folds,
        split_auc=split_auc,
        stratified_permutation=_stratified_permutation,
    )


def _seeded_folds(labels: np.ndarray, n_splits: int, seed: int):
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    usable = min(n_splits, int(counts.min())) if counts.size else 0
    if usable < 2:
        return None
    return _decoder_api().make_folds(labels, n_splits=usable, seed=seed)


CANDIDATES = (
    "native_full_rank",
    "principal_components",
    "factor_analysis",
    "gaussian_process_factor_analysis",
    "time_contrastive_embedding",
    "temporal_diffusion_embedding",
    "sequential_autoencoder",
)


DECODERS = ("linear", "nonlinear")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def fold_signature(folds) -> str:
    values = [
        [np.asarray(train, dtype=int).tolist(), np.asarray(test, dtype=int).tolist()]
        for train, test in folds
    ]
    return hashlib.sha256(canonical_json(values).encode()).hexdigest()[:12]


def score_null_draw(
    fits: list[dict],
    labels: np.ndarray,
    decoder: str,
    folds,
    shuffle_seed: int,
    decoder_seed: int,
    time_step: int = 1,
) -> dict:
    api = _decoder_api()
    shuffled = api.stratified_permutation(
        labels, folds, np.random.default_rng(shuffle_seed)
    )
    fold_scores = []
    for fold_id, (train, test) in enumerate(folds):
        fit = fits[fold_id]
        fold_scores.append(
            api.split_auc(
                np.asarray(fit["latent_train"], dtype=float)[:, ::time_step],
                shuffled[train],
                np.asarray(fit["latent_test"], dtype=float)[:, ::time_step],
                shuffled[test],
                decoder=decoder,
                seed=decoder_seed + fold_id,
            )
        )
    return {"status": "computed", "temporal_auc": np.nanmean(fold_scores, axis=0)}


def score_observed(
    fits: list[dict],
    labels: np.ndarray,
    decoder: str,
    folds,
    seed: int,
    time_step: int = 1,
) -> dict:
    api = _decoder_api()
    temporal_folds = []
    for fold_id, (train, test) in enumerate(folds):
        fit = fits[fold_id]
        latent_train = np.asarray(fit["latent_train"], dtype=float)[:, ::time_step]
        latent_test = np.asarray(fit["latent_test"], dtype=float)[:, ::time_step]
        temporal_folds.append(
            api.split_auc(
                latent_train,
                labels[train],
                latent_test,
                labels[test],
                decoder=decoder,
                seed=seed + fold_id,
            )
        )
    return {"status": "computed", "temporal_auc": np.nanmean(temporal_folds, axis=0)}


def summarize_scores(temporal: np.ndarray, temporal_null: np.ndarray, decoder: str) -> dict:
    temporal = np.asarray(temporal, dtype=float)
    temporal_null = np.asarray(temporal_null, dtype=float)
    same_time = np.diag(temporal)
    same_time_null = np.diagonal(temporal_null, axis1=1, axis2=2)
    off_diag = ~np.eye(temporal.shape[0], dtype=bool)
    return {
        "status": "computed",
        "decoder": decoder,
        "same_time": _metric(same_time, same_time_null),
        "cross_temporal": _metric(temporal, temporal_null, off_diag),
        "same_time_auc": np.asarray(same_time),
        "temporal_auc": np.asarray(temporal),
        "same_time_null_mean": np.nanmean(same_time_null, axis=0),
        "temporal_null_mean": np.nanmean(temporal_null, axis=0),
    }


def _metric(observed: np.ndarray, null: np.ndarray, mask: np.ndarray | None = None) -> dict:
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if mask is not None:
        observed = observed[mask]
        null = null[:, mask]
    score = _mean(observed)
    null_scores = np.asarray([_mean(draw) for draw in null], dtype=object)
    null_scores = np.asarray([value for value in null_scores if value is not None], dtype=float)
    if score is None or not null_scores.size:
        return {"status": "not_computable"}
    null_mean = float(null_scores.mean())
    p_value = float((1 + np.sum(null_scores >= score)) / (null_scores.size + 1))
    return {
        "status": "computed",
        "auc": score,
        "null_auc": null_mean,
        "auc_above_null": float(score - null_mean),
        "p_value": p_value,
        "n_permutations": int(null_scores.size),
    }


def _mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def mean_squared_displacement(latent: np.ndarray) -> np.ndarray:
    """latent: (n_trials, n_bins, k) -> msd[lag-1] for lag = 1..n_bins-1."""
    n_bins = latent.shape[1]
    max_lag = n_bins - 1
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        diffs = latent[:, lag:, :] - latent[:, :-lag, :]
        msd[lag - 1] = np.mean(np.sum(diffs**2, axis=2))
    return msd


def summarize_msd(msd: np.ndarray) -> dict:
    max_lag = len(msd)
    half = max(1, max_lag // 2)
    saturation_ratio = float(msd[-1] / msd[half - 1]) if msd[half - 1] > 0 else None
    lags = np.arange(1, max_lag + 1)
    valid = msd > 0
    if valid.sum() >= 3:
        slope, _ = np.polyfit(np.log(lags[valid]), np.log(msd[valid]), 1)
        log_log_slope = float(slope)
    else:
        log_log_slope = None
    return {
        "saturation_ratio": saturation_ratio,
        "log_log_slope": log_log_slope,
        "msd_at_K_state_units_sq": float(msd[-1]),
        "msd_curve": msd.tolist(),
    }


LATENT_DIM = 6


MIN_TEST_TRIALS = 6


MIN_TRIALS = 20


def anscombe_counts(counts: np.ndarray) -> np.ndarray:
    """Variance-stabilizing transform matching the project's existing
    Anscombe convention (scripts/run_drift_positive_controls.py)."""
    return 2.0 * np.sqrt(np.maximum(counts, 0.0) + 3.0 / 8.0)


def counts_to_spiketrains(counts: np.ndarray, bin_size_ms: float) -> list[list[neo.SpikeTrain]]:
    """Reconstruct one synthetic spike per unit count, placed uniformly within
    its bin -- elephant's GPFA re-bins at the same width on input, so this
    round-trips the original integer counts exactly."""
    n_trials, n_units, n_bins = counts.shape
    t_stop = n_bins * bin_size_ms
    trials = []
    for tr in range(n_trials):
        units = []
        for u in range(n_units):
            spikes: list[float] = []
            for b in range(n_bins):
                n = int(counts[tr, u, b])
                if n > 0:
                    offsets = (np.arange(n) + 0.5) / n * bin_size_ms
                    spikes.extend(b * bin_size_ms + offsets)
            units.append(neo.SpikeTrain(sorted(spikes) * pq.ms, t_stop=t_stop * pq.ms))
        trials.append(units)
    return trials


TRAIN_FRACTION = 0.6


def split_trials(n_trials: int, rng: np.random.Generator, train_fraction: float = TRAIN_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_trials)
    n_train = max(2, int(round(n_trials * train_fraction)))
    n_train = min(n_train, n_trials - 2)
    return perm[:n_train], perm[n_train:]


def dimensionality_and_noise_term(counts: np.ndarray, is_point_process: bool = True) -> dict:
    """Participation ratio of the full population covariance spectrum
    (PCA arm: no noise correction) versus the same spectrum with the
    factor model's own estimated diagonal observation-noise variance
    subtracted before the eigendecomposition (factor-analysis arm), plus
    the factor model's own noise-variance share of total variance -- the
    quantity compared against the census's nugget fraction below. Computed
    on every available trial (a covariance property, not a held-out
    predictive one). ``is_point_process`` selects the preprocessing this
    project already uses for the two physically different measurements it
    feeds through this same estimator: Anscombe-variance-stabilized spike
    counts (scripts/run_latent_model_comparison.py's convention) for a
    point-process grain, or plain mean-centering with no count-specific
    transform for a continuous grain (e.g. bipolar-referenced LFP high-
    gamma power, which is not a count and for which the Anscombe transform
    has no justification)."""
    from sklearn.decomposition import FactorAnalysis
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    n_units = counts.shape[1]
    x = anscombe_counts(counts) if is_point_process else counts
    flat = x.transpose(0, 2, 1).reshape(-1, n_units)
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = np.cov(flat, rowvar=False)
    pca_eigenvalues = np.linalg.eigvalsh(cov)
    pca_pr = float(participation_ratio(pca_eigenvalues))

    k = max(1, min(LATENT_DIM, n_units - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = FactorAnalysis(n_components=k, random_state=0, max_iter=1000)
            model.fit(flat)
    except (ConvergenceWarning, ValueError, np.linalg.LinAlgError):
        return {
            "pca": {"participation_ratio": pca_pr, "k_used_for_noise_estimate": k},
            "factor_analysis": {"status": "factor_model_did_not_converge_or_degenerate"},
        }
    noise_variance = model.noise_variance_
    if not np.all(np.isfinite(noise_variance)) or np.sum(noise_variance) <= 0:
        return {
            "pca": {"participation_ratio": pca_pr, "k_used_for_noise_estimate": k},
            "factor_analysis": {"status": "factor_model_did_not_converge_or_degenerate"},
        }
    denoised_cov = cov - np.diag(noise_variance)
    denoised_eigenvalues = np.clip(np.linalg.eigvalsh(denoised_cov), 0.0, None)
    fa_pr = float(participation_ratio(denoised_eigenvalues))
    noise_variance_fraction = float(np.sum(noise_variance) / np.trace(cov))
    return {
        "pca": {"participation_ratio": pca_pr, "k_used_for_noise_estimate": k},
        "factor_analysis": {
            "status": "fitted", "participation_ratio": fa_pr, "k_used": k,
            "observation_noise_variance_fraction": noise_variance_fraction,
        },
    }


def _whole_session_cluster_bootstrap(values: list[float], seed_tag: str) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = int(arr.size)
    if n < MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP:
        return {"status": "not_computable", "n_sessions": n,
                "reason": f"fewer than {MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP} sessions have a detected subspace to pool"}
    rng = np.random.default_rng(stable_seed(seed_tag))
    boot_means = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(N_BOOT_SESSION_CLUSTER)])
    se = float(np.std(boot_means, ddof=1))
    ci_low, ci_high = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))
    return {
        "status": "computed", "n_sessions": n, "n_bootstrap_draws": N_BOOT_SESSION_CLUSTER,
        "pooled_mean": float(arr.mean()), "cluster_bootstrap_ci_95pct": [ci_low, ci_high],
        "cluster_bootstrap_se": se, "z_multiplier_80pct_power": Z_80_POWER,
        "minimum_detectable_difference_at_80pct_power": Z_80_POWER * se,
    }


MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP = 4


ESTIMATOR_NAMES = ("entry_holdout_bicross_validation", "permutation_eigenvalue_threshold")


def _gate_status_for_corpus(corpus_result: dict, gate: dict) -> dict:
    """Validation is SCALE-CONDITIONAL, not a single global pass/fail: `minimum_validated_ambient_unit_count`
    is already the smallest tested ambient size at which an estimator passed the low-noise gate, so a corpus
    whose own median ambient unit count sits at or above that size is validated for that estimator even if
    the estimator failed the gate at a SMALLER tested size (`passes_gate_overall` is reported alongside for
    transparency, but requiring it in addition would make the per-size minimum meaningless -- an estimator
    that only works above some scale is still usable on cells above that scale). Extrapolation beyond the
    largest tested size (600 ambient units) is not attempted; recovery only ever improved with ambient size
    in the synthetic gate here, so this is a disclosed, one-directional assumption, not a measured fact above
    that ceiling."""
    median_units = corpus_result["median_ambient_unit_count_across_cells"]
    out = {}
    for est_name in ESTIMATOR_NAMES:
        g = gate["gate_by_estimator"][est_name]
        min_p = g["minimum_validated_ambient_unit_count"]
        validated = bool(min_p is not None and median_units is not None and median_units >= min_p)
        out[est_name] = {
            "passes_gate_overall": g["passes_gate_overall"], "minimum_validated_ambient_unit_count": min_p,
            "corpus_median_ambient_unit_count": median_units, "validated_for_this_corpus": validated,
        }
    return out


def _one_sample_patient_stats(patient_values: dict[str, float]) -> dict:
    values = np.array(list(patient_values.values()), dtype=float)
    n = len(values)
    if n < 2:
        return {"status": "not_computable", "n_patients": n, "reason": "fewer than 2 patients with a fitted median"}
    t = stats.ttest_1samp(values, 0.0)
    w = stats.wilcoxon(values) if n >= 1 and np.any(values != 0) else None
    return {
        "status": "computed",
        "n_patients": n,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "n_positive": int((values > 0).sum()),
        "t_test_p_value": float(t.pvalue),
        "wilcoxon_p_value": float(w.pvalue) if w is not None else None,
        "per_patient_median": dict(sorted(patient_values.items())),
    }


def _paired_patient_stats(interest: dict[str, float], reference: dict[str, float], seed_parts: tuple) -> dict:
    shared = sorted(set(interest) & set(reference))
    n = len(shared)
    if n < 2:
        return {"status": "not_computable", "n_patients": n, "reason": "fewer than 2 patients with both arms fitted"}
    interest_arr = np.array([interest[p] for p in shared], dtype=float)
    reference_arr = np.array([reference[p] for p in shared], dtype=float)
    diffs = interest_arr - reference_arr
    test = paired_sign_flip_test(interest_arr, reference_arr, alternative="two-sided", rng=_seed(*seed_parts))
    mdd = minimum_detectable_paired_difference(diffs)
    return {
        "status": "computed",
        "n_patients": n,
        "patients": shared,
        "mean_difference_interest_minus_reference": float(diffs.mean()),
        "median_difference_interest_minus_reference": float(np.median(diffs)),
        "n_positive": int((diffs > 0).sum()),
        "p_value": test["p_value"],
        "ci_lower_mean_difference": test["ci_lower"],
        "ci_upper_mean_difference": test["ci_upper"],
        "minimum_detectable_paired_difference_80pct_power": mdd,
    }


def _patient_median(values_by_patient_session: dict[tuple[str, str], float]) -> dict[str, float]:
    by_patient: dict[str, list[float]] = {}
    for (patient, _session), value in values_by_patient_session.items():
        by_patient.setdefault(patient, []).append(value)
    return {patient: float(np.median(vals)) for patient, vals in by_patient.items()}


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


SEED = 20260813


FDR_ALPHA = 0.05


MIN_INDEPENDENT_UNITS = 4


N_BOOT = 2000


N_FOLDS = 2


def _rotation(directions: np.ndarray, target: np.ndarray, kind: str) -> dict:
    n = len(target)
    folds = block_folds(n, N_FOLDS)
    fit_basis = class_basis if kind == "categorical" else regression_basis
    early_basis = fit_basis(directions[folds == 0], target[folds == 0])
    late_basis = fit_basis(directions[folds == 1], target[folds == 1])
    if early_basis is None or late_basis is None:
        return {"status": "not_computable", "reason": "label subspace could not be fit in one or both halves"}
    angles = principal_angles(early_basis, late_basis)
    return {
        "status": "computed",
        "n_trials_early": int((folds == 0).sum()),
        "n_trials_late": int((folds == 1).sum()),
        "principal_angles_radians": [float(a) for a in angles],
        "mean_principal_angle_radians": float(np.mean(angles)),
        "subspace_overlap": float(subspace_overlap(early_basis, late_basis)),
    }


def _combine_levels(levels: list[tuple[int, dict]]) -> dict:
    usable = [(weight, cell) for weight, cell in levels if cell.get("status") == "computed"]
    if not usable:
        return {"status": "not_computable", "reason": "no item-count level was computable"}
    weights = np.asarray([weight for weight, _ in usable], dtype=float)
    observed = np.average([cell["alignment"] for _, cell in usable], weights=weights)
    n_draws = min(len(cell["null_draws"]) for _, cell in usable)
    draws = np.average(
        np.stack([np.asarray(cell["null_draws"])[:n_draws] for _, cell in usable]), axis=0, weights=weights,
    )
    return {
        "status": "computed",
        "alignment": float(observed),
        "null_mean": float(draws.mean()),
        "alignment_above_null": float(observed - draws.mean()),
        "p_value": float((1 + np.sum(draws >= observed)) / (len(draws) + 1)),
        "n_permutations": len(draws),
        "null_interval_95pct": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "null_draws": draws,
        "n_levels": len(usable),
        "n_trials": int(weights.sum()),
        "target_kind": usable[0][1].get("target_kind"),
        "seed_ids": [cell.get("seed_id") for _, cell in usable],
    }


def _finite(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    if target.ndim == 1:
        return np.isfinite(target)
    return np.isfinite(target).all(axis=tuple(range(1, target.ndim)))


def _prepare_trials(activity: np.ndarray, target: np.ndarray):
    """Mirrors run_rank_free_component_identity._cell's own trial-validity filter exactly (same
    _finite import, same N_FOLDS-derived floor), so the interleaved-fold and rotation cells here see
    the identical trial set the delivered block-fold cell used internally."""
    activity = np.asarray(activity, dtype=float)
    target = np.asarray(target)
    norms = np.linalg.norm(activity, axis=1)
    valid = np.isfinite(activity).all(axis=1) & (norms > 0) & _finite(target)
    if int(valid.sum()) < 2 * max(3, N_FOLDS):
        return None
    return activity[valid] / norms[valid, None], target[valid]


CATEGORIES_100_DIVIDED = "PicIDs_Encoding1 // 100 resolves to 5 category values (1..5)"


CATEGORIES_469 = "loadsEnc1_PicIDs is already valued 1..5 -- used directly as the category"


DPCA_RIDGE_LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0)


N_CV_FOLDS_LAMBDA = 3


def _category_469(pic_ids: np.ndarray) -> np.ndarray:
    return pic_ids.astype(int)


def _category_divided(pic_ids: np.ndarray) -> np.ndarray:
    return (pic_ids.astype(int) // 100)


MIN_CLASSES = 2


CONTENT_N_COMPONENTS_MAX = 8


CONTENT_N_HALF_SPLITS = 12


CONTENT_N_PERM_LATENT1 = 30


def session_subtractive_test(X: np.ndarray, labels: np.ndarray, seed: int,
                              k_max: int = CONTENT_N_COMPONENTS_MAX, n_half_splits: int = CONTENT_N_HALF_SPLITS,
                              with_permutation_nulls: bool = True) -> dict:
    """Leave-one-latent-out ablation cost of every one of the k PCA latents,
    and the leading latent's rank among them by that cost.

    ``with_permutation_nulls=False`` returns the ablation-cost ranking and
    the full-model decoding accuracy but skips the three label-permutation
    null fits, which dominate the cost. Callers that need only the ranking
    -- for instance one resampling the label sequence many times per session
    to build its own null over the pooled ranking -- should set it False;
    the permutation p-values of a resampled label sequence answer a
    different question and would be discarded anyway. Every returned field
    that does not depend on the nulls is identical either way."""
    k = min(k_max, X.shape[1] - 2, max(2, X.shape[0] // 8))
    if k < 3:
        return {"status": "too_few_units_or_trials_for_k_latents", "k_attempted": int(k)}
    t_idx = np.array([0])

    full_aucs: list[float] = []
    minus_aucs: dict[int, list[float]] = {j: [] for j in range(k)}
    for s in range(n_half_splits):
        rng = np.random.default_rng(seed + s)
        full = content_decoding_dropping_latent(X, labels, t_idx, k, None, n_splits=2, n_perm=0, rng=rng)
        full_aucs.append(float(full["auc_per_t"][0]))
        for j in range(k):
            rng_j = np.random.default_rng(seed + 1000 * (s + 1) + j)
            m = content_decoding_dropping_latent(X, labels, t_idx, k, j, n_splits=2, n_perm=0, rng=rng_j)
            minus_aucs[j].append(float(m["auc_per_t"][0]))

    a_full = float(np.nanmedian(full_aucs))
    a_minus = {j: float(np.nanmedian(minus_aucs[j])) for j in range(k)}
    cost = {j: a_full - a_minus[j] for j in range(k)}
    rank_from_top = 1 + sum(1 for j in range(1, k) if cost[j] > cost[0])
    fractional_rank = (rank_from_top - 1) / (k - 1) if k > 1 else 0.0

    ranking = {
        "status": "tested", "n_trials": int(X.shape[0]), "n_units": int(X.shape[1]),
        "n_classes": int(len(np.unique(labels))), "k_latents": int(k),
        "a_full": a_full, "a_minus1": a_minus[0], "a_minus_j": a_minus, "cost_j": cost,
        "leading_latent_cost": cost[0], "leading_latent_rank_from_top": rank_from_top,
        "leading_latent_fractional_rank": fractional_rank,
    }
    if not with_permutation_nulls:
        return ranking

    null_rng = np.random.default_rng(seed + 55555)
    full_with_null = content_decoding_dropping_latent(X, labels, t_idx, k, None, n_splits=3, n_perm=CONTENT_N_PERM_FULL, rng=null_rng)
    minus1_rng = np.random.default_rng(seed + 66666)
    minus1_with_null = content_decoding_dropping_latent(X, labels, t_idx, k, 0, n_splits=3, n_perm=CONTENT_N_PERM_FULL, rng=minus1_rng)
    latent1_rng = np.random.default_rng(seed + 77777)
    latent1 = content_decoding_dropping_latent(X, labels, t_idx, 1, None, n_splits=3, n_perm=CONTENT_N_PERM_LATENT1, rng=latent1_rng)

    return {
        **ranking,
        "a_full_p_value": float(full_with_null["p_per_t"][0]),
        "a_full_clears_own_null": bool(full_with_null["p_per_t"][0] <= 0.05),
        "a_minus1_p_value": float(minus1_with_null["p_per_t"][0]),
        "a_minus1_clears_own_null": bool(minus1_with_null["p_per_t"][0] <= 0.05),
        "a_latent1_alone": float(latent1["auc_per_t"][0]), "a_latent1_p_value": float(latent1["p_per_t"][0]),
    }


N_SHUFFLES_RANK1 = 200


def session_rank1_and_residual(counts: np.ndarray, width_bins: int, seed: int) -> dict | None:
    rng = np.random.default_rng(seed)
    n_trials = counts.shape[0]
    perm = rng.permutation(n_trials)
    half = n_trials // 2
    train_idx, test_idx = perm[:half], perm[half:]
    if len(train_idx) < 6 or len(test_idx) < 6:
        return None
    transform = FrozenPSTHTransform().fit(counts[train_idx])
    z_train, z_test = transform.transform(counts[train_idx]), transform.transform(counts[test_idx])
    latent = _leading_latent_projection(z_train, z_test)
    window_means = _window_means_at_width(latent, width_bins)
    if window_means is None or window_means.shape[1] < 4:
        return None
    rank1 = rank1_gain_share(window_means, n_shuffles=N_SHUFFLES_RANK1, rng=rng)
    gain, h_profile, residual = rank1_gain_and_residual(window_means)
    crossings = temporal_profile_sign_crossings(h_profile)
    position_corr = [float(np.corrcoef(gain, window_means[:, j])[0, 1]) if window_means[:, j].std() > 0 else None
                      for j in range(window_means.shape[1])]
    return {"test_idx": test_idx, "gain": gain, "rank1": rank1, "residual": residual,
            "temporal_profile_sign_crossings": crossings,
            "position_corr": [c for c in position_corr if c is not None]}


BIN_WIDTH_S = BIN_MS / 1000.0


HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")


PANICHELLO_DELAY_WINDOW_MS = (300.0, 1450.0)


PANICHELLO_LAG_N_NULL_REPLICATES = 10


PANICHELLO_LAG_N_SPLITS = 10


def _to_int_keyed(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def _lag_lists(rows: list[dict], width: int) -> tuple[list[dict], list[dict], list[dict]]:
    profiles, pois, perm = [], [], []
    for r in rows:
        if r.get("width_bins") != width:
            continue
        if r["profile"].get("status") != "fitted" or r.get("null_poisson") is None or r.get("null_permutation") is None:
            continue
        profiles.append(_to_int_keyed(r["profile"]["lags"]))
        pois.append(_to_int_keyed(r["null_poisson"]["lags"]))
        perm.append(_to_int_keyed(r["null_permutation"]["lags"]))
    return profiles, pois, perm


CAUSAL_MICROSTIM_ENERGY_ACCURACY_Q = (0.01, 0.1, 1.0, 10.0, 100.0)


CAUSAL_MICROSTIM_GRAMIAN_HORIZON = 20


CTG_N_SPLITS = 4


CTG_STEP = 2


def render_summary(
    records: list[dict],
    complete: bool,
    time_step: int = CTG_STEP,
    failed: bool = False,
) -> str:
    status = "failed" if failed else "complete" if complete else "running"
    lines = [
        "# Matched information benchmark",
        "",
        f"Status: {status}",
        "",
        "All decoder cells use the same stratified folds within each session and item-count level. "
        "Representations are fit on training trials at full temporal resolution. Each score is "
        "compared with label permutations evaluated on the same held-out folds.",
        f"Decoder temporal grids use every {time_step} time bin(s).",
        "",
        "| Corpus | Session | Level | Representation | Decoder | Status | Same-time AUC | Same-time null | Same-time ΔAUC | Same-time p | Cross-time AUC | Cross-time null | Cross-time ΔAUC | Cross-time p |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        same = record.get("same_time", {}).get("auc_above_null")
        cross = record.get("cross_temporal", {}).get("auc_above_null")
        same_auc = record.get("same_time", {}).get("auc")
        cross_auc = record.get("cross_temporal", {}).get("auc")
        same_null = record.get("same_time", {}).get("null_auc")
        cross_null = record.get("cross_temporal", {}).get("null_auc")
        same_p = record.get("same_time", {}).get("p_value")
        cross_p = record.get("cross_temporal", {}).get("p_value")
        lines.append(
            "| {corpus} | {session} | {level} | {candidate} | {decoder} | {status} | {same_auc} | {same_null} | {same} | {same_p} | {cross_auc} | {cross_null} | {cross} | {cross_p} |".format(
                corpus=record.get("corpus", ""),
                session=record.get("session", ""),
                level=record.get("level", ""),
                candidate=record.get("candidate", ""),
                decoder=record.get("decoder", ""),
                status=record.get("status", ""),
                same_auc="" if same_auc is None else f"{same_auc:.3f}",
                same_null="" if same_null is None else f"{same_null:.3f}",
                same="" if same is None else f"{same:.3f}",
                same_p="" if same_p is None else f"{same_p:.3g}",
                cross_auc="" if cross_auc is None else f"{cross_auc:.3f}",
                cross_null="" if cross_null is None else f"{cross_null:.3f}",
                cross="" if cross is None else f"{cross:.3f}",
                cross_p="" if cross_p is None else f"{cross_p:.3g}",
            )
        )
    return "\n".join(lines) + "\n"


DELIVERED_RANK = 8


HUMAN_BIN_MS = 200.0


def _energy_error_slope(energies: np.ndarray, errors: np.ndarray) -> float | None:
    if len(energies) < 2 or np.std(energies) < 1e-12:
        return None
    return float(np.polyfit(energies, errors, 1)[0])


def claim_control_model(fit: dict, cond_info: dict, rng: np.random.Generator) -> dict:
    """Controllability, the stimulation-input-alignment targeting quantity, and the energy-accuracy
    trade-off, at the rank ``fit`` was estimated at. The energy-accuracy sweep targets each real
    stimulation direction scaled to unit displacement in the fitted latent space with a fully
    actuatable (identity) input matrix -- this asks how the trade-off itself shifts with rank, not a
    reproduction of any single delivered causal-displacement measurement."""
    A, components, v_star, v_stable = fit["A"], fit["components"], fit["v_star"], fit["v_stable"]
    k = A.shape[0]
    per_condition = {}
    x0_list, xf_list = [], []
    for c, info in cond_info.items():
        alignment = stimulation_input_alignment(A, components, info["b_chan"], v_star, v_stable, rng,
                                                  gramian_horizon=CAUSAL_MICROSTIM_GRAMIAN_HORIZON)
        per_condition[str(c)] = {**alignment, "accuracy": info["accuracy"],
                                  "n_correct": info["n_correct"], "n_error": info["n_error"]}
        b_lat = components.T @ info["b_chan"]
        b_hat = b_lat / (np.linalg.norm(b_lat) + 1e-12)
        x0_list.append(np.zeros(k))
        xf_list.append(b_hat)

    energy_accuracy = None
    if x0_list:
        B_identity = np.eye(k)
        pareto = energy_accuracy_pareto(A, B_identity, x0_list, xf_list,
                                         np.array(CAUSAL_MICROSTIM_ENERGY_ACCURACY_Q), T=20)
        energy_accuracy = {
            "q_values": pareto["q_values"].tolist(), "energies": pareto["energies"].tolist(),
            "errors": pareto["errors"].tolist(),
            "energy_error_slope": _energy_error_slope(pareto["energies"], pareto["errors"]),
        }
    return {"status": "computed", "rank": int(k), "n_conditions": int(len(cond_info)),
            "per_condition": per_condition, "energy_accuracy": energy_accuracy}


def condition_mean_subspace(X: np.ndarray, labels: np.ndarray, rank: int) -> np.ndarray | None:
    """Top ``rank`` PCA axes of the between-condition mean matrix -- the linear subspace a
    memorandum-content label occupies."""
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    means = np.stack([X[labels == c].mean(axis=0) for c in classes])
    means_c = means - means.mean(axis=0)
    d = int(min(rank, means_c.shape[0] - 1, means_c.shape[1]))
    if d < 1:
        return None
    _, _, Vt = np.linalg.svd(means_c, full_matrices=False)
    return Vt[:d].T


def claim_memorandum_subspace(X: np.ndarray, labels: np.ndarray, deviation: np.ndarray, w: np.ndarray,
                               rank: int, rng: np.random.Generator, n_perm: int) -> dict:
    """Does the deviation direction lie inside the memorandum's coding subspace at this rank?
    Effect size: the observed projection fraction minus its label-permutation null."""
    S = condition_mean_subspace(X, labels, rank)
    if S is None:
        return {"status": "not_computable", "rank": rank}
    P = S @ S.T
    w_norm = np.linalg.norm(w) + 1e-12
    within = float(np.linalg.norm(P @ w) / w_norm)
    null_vals = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        Sp = condition_mean_subspace(X, perm_labels, rank)
        if Sp is None:
            continue
        null_vals.append(float(np.linalg.norm((Sp @ Sp.T) @ w) / w_norm))
    null_vals = np.asarray(null_vals)
    p_value = permutation_pvalue(null_vals >= within) if len(null_vals) else float("nan")
    null_mean = float(np.mean(null_vals)) if len(null_vals) else float("nan")
    return {"status": "computed", "rank": int(S.shape[1]), "within_frac": within,
            "null_mean_within_frac": null_mean, "effect_size": within - null_mean,
            "p_value": p_value, "n_null": int(len(null_vals)), "n_trials": int(X.shape[0])}


def fit_linear_representation(X: np.ndarray, rank: int, labels: np.ndarray | None = None) -> dict:
    """PCA representation of an (n_trials, n_features) array at the given rank.

    The one function every claim's verdict routine consumes: swapping this for a differently fitted
    (including nonlinear) representation of the same ``scores``/``components`` shape changes no
    downstream claim code. That is exactly why the label refusal belongs here rather than only in
    prose -- this is the single fitting entry point a later nonlinear embedding is reached through,
    and any embedding trained with a label or behavioural outcome as its fitting objective is
    circular for every behaviour claim that consumes it. ``labels`` exists only so that passing one
    raises; this project forbids label- or behaviour-conditioned representation fitting everywhere.
    """
    if labels is not None:
        raise ValueError(
            "fit_linear_representation refuses a non-null label argument -- label- or "
            "behaviour-conditioned representation fitting is forbidden project-wide for any "
            "behaviour claim")
    rank_eff = int(max(1, min(rank, X.shape[0] - 1, X.shape[1])))
    scores, components, var_ratio = pca_decompose(X, rank_eff)
    return {"kind": "pca", "rank": rank_eff, "scores": scores, "components": components,
            "var_ratio": var_ratio, "fitting_objective": "unsupervised_pca_reconstruction_no_labels"}


def _require_linear_representation(representation: dict) -> None:
    """A subspace-angle projector (components @ components.T) has no meaning without a canonical
    orthonormal basis, which a nonlinear embedding does not have. Every code path that builds such a
    projector must call this first, so a future nonlinear representation cannot be fed to it by
    accident -- the estimator-invariant restatement (``cross_validated_predictable_fraction``) is the
    only route to a claim's verdict once the representation is nonlinear."""
    if representation.get("kind") != "pca":
        raise ValueError(
            f"subspace-angle projector requested on a non-linear representation (kind="
            f"{representation.get('kind')!r}); a subspace angle has no canonical basis in a "
            "nonlinear embedding -- use cross_validated_predictable_fraction instead")


def claim_occupied_manifold(X: np.ndarray, w: np.ndarray, rank: int, rng: np.random.Generator,
                             n_perm: int) -> dict:
    """Decomposition of the deviation direction against the top-``rank`` occupied manifold, against
    a random-direction null in the same ambient feature space -- the same within/outside-fraction
    construction the delivered causal-microstimulation manifold-constraint analysis uses, generalised
    from a fixed rank to a swept one."""
    representation = fit_linear_representation(X, rank)
    _require_linear_representation(representation)
    comps = representation["components"]
    r_eff = comps.shape[1]
    C = X.shape[1]
    P = comps @ comps.T
    w_norm = np.linalg.norm(w) + 1e-12
    within = float(np.linalg.norm(P @ w) / w_norm)
    outside = float(np.linalg.norm((np.eye(C) - P) @ w) / w_norm)
    rand = rng.standard_normal((n_perm, C))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12
    null_within = np.linalg.norm(rand @ P, axis=1)
    p_value = permutation_pvalue(null_within >= within)
    null_mean = float(np.mean(null_within))
    return {"status": "computed", "rank": int(r_eff), "within_frac": within, "outside_frac": outside,
            "null_mean_within_frac": null_mean, "effect_size": within - null_mean,
            "p_value": p_value, "n_null": int(n_perm), "n_trials": int(X.shape[0]),
            "fitting_objective": representation["fitting_objective"]}


def component_direction(X: np.ndarray, deviation: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Unit-norm ridge-regression direction in feature space along which trial-to-trial variation
    best predicts the leave-one-out cosine deviation -- a concrete axis for the two subspace-
    projection claims (the predictable-fraction restatement needs no such direction, only the
    scalar deviation itself)."""
    mask = np.isfinite(deviation)
    Xc = X[mask] - X[mask].mean(axis=0)
    yc = deviation[mask] - deviation[mask].mean()
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(Xc, yc)
    w = model.coef_
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w


CV_FOLDS = 5


def cross_validated_predictable_fraction(y: np.ndarray, Z: np.ndarray, n_splits: int = CV_FOLDS,
                                          alpha: float = 1.0, rng: np.random.Generator | None = None) -> dict:
    """Cross-validated fraction of scalar ``y``'s variance predictable by ridge regression on
    representation coordinates ``Z``.

    Defined identically whether ``Z`` came from a linear projection or a nonlinear embedding -- ridge
    regression from a fixed set of per-trial coordinates onto a scalar target makes no reference to
    how those coordinates were produced. Held-out predictions are concatenated across folds before
    the single R^2 is computed, rather than averaging per-fold R^2, so a fold with little residual
    variance cannot dominate the summary.
    """
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y, Z = y[mask], Z[mask]
    n = len(y)
    if n < max(6, n_splits + 1):
        return {"status": "not_computable", "reason": "fewer trials than folds require", "n_trials": n}
    n_splits_eff = min(n_splits, n)
    kf = KFold(n_splits=n_splits_eff, shuffle=True,
               random_state=int(rng.integers(0, 2**31 - 1)) if rng is not None else 0)
    y_true_held, y_pred_held = [], []
    for train_idx, test_idx in kf.split(Z):
        model = Ridge(alpha=alpha)
        model.fit(Z[train_idx], y[train_idx])
        y_pred_held.append(model.predict(Z[test_idx]))
        y_true_held.append(y[test_idx])
    y_true_held = np.concatenate(y_true_held)
    y_pred_held = np.concatenate(y_pred_held)
    ss_res = float(np.sum((y_true_held - y_pred_held) ** 2))
    ss_tot = float(np.sum((y_true_held - y_true_held.mean()) ** 2))
    fraction = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"status": "computed", "predictable_fraction": fraction, "n_trials": n,
            "n_dims": int(Z.shape[1]), "alpha": alpha, "n_splits": n_splits_eff}


def in_sample_linear_fraction(y: np.ndarray, Z: np.ndarray) -> dict:
    """Non-cross-validated (in-sample, ordinary-least-squares) fraction of ``y``'s variance
    explained by ``Z`` -- the naive linear quantity this project's existing subspace-projection
    metrics are instances of, reported beside the cross-validated version for comparison."""
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y, Z = y[mask], Z[mask]
    n = len(y)
    if n < Z.shape[1] + 2:
        return {"status": "not_computable", "n_trials": n}
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    coef, *_ = np.linalg.lstsq(Zc, yc, rcond=None)
    fitted = Zc @ coef
    ss_res = float(np.sum((yc - fitted) ** 2))
    ss_tot = float(np.sum(yc ** 2))
    fraction = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"status": "computed", "linear_fraction": fraction, "n_trials": n, "n_dims": int(Z.shape[1])}


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return X / norm


def leave_one_out_cosine_deviation(X: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """Per-trial deviation score needing no dimensionality reduction: one minus the cosine
    similarity between a trial's L2-normalised feature vector and the L2-normalised leave-one-out
    mean of its own condition. NaN for trials whose condition has fewer than two members."""
    Xn = l2_normalize_rows(np.asarray(X, dtype=float))
    labels = np.asarray(condition)
    deviation = np.full(Xn.shape[0], np.nan)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        total = Xn[idx].sum(axis=0)
        for i in idx:
            loo_mean = (total - Xn[i]) / (len(idx) - 1)
            loo_norm = np.linalg.norm(loo_mean)
            if loo_norm < 1e-12:
                continue
            deviation[i] = 1.0 - float(Xn[i] @ loo_mean) / loo_norm
    return deviation


def load_causal_microstim_dynamics_inputs(prefix: str) -> dict | None:
    corr = load_causal_microstim_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    err = load_causal_microstim_session(prefix, correct=False)
    control_idx = corr["control_idx"]
    channel_ids = corr["channel_ids"]
    C = len(channel_ids)

    ctrl_epochs = [crop_trial(tr["spikerate"]) for tr in corr["trials"] if tr["stim_cond"] == control_idx]
    ctrl_epochs = [e for e in ctrl_epochs if e is not None]
    if len(ctrl_epochs) < 10:
        return None
    trials = np.stack(ctrl_epochs, axis=0).transpose(0, 2, 1)  # (n_trials, C, n_bins)

    n_correct_control = len(ctrl_epochs)
    n_error_control = sum(1 for tr in err["trials"] if tr["stim_cond"] == control_idx) if err else 0

    cond_info = {}
    for c in range(len(corr["stim_channels"])):
        if c == control_idx:
            continue
        chan_ids = corr["stim_channels"][c]
        idx = [i for i, cid in enumerate(channel_ids) if cid in chan_ids]
        if len(idx) != len(chan_ids):
            continue
        b_chan = np.zeros(C)
        b_chan[idx] = 1.0 / len(idx)
        n_correct = sum(1 for tr in corr["trials"] if tr["stim_cond"] == c)
        n_error = sum(1 for tr in err["trials"] if tr["stim_cond"] == c) if err else 0
        accuracy = n_correct / (n_correct + n_error) if (n_correct + n_error) > 0 else None
        cond_info[c] = {"b_chan": b_chan, "n_correct": n_correct, "n_error": n_error, "accuracy": accuracy}

    return {"trials": trials, "n_channels": int(C), "cond_info": cond_info,
            "control_accuracy": {
                "n_correct": n_correct_control, "n_error": n_error_control,
                "accuracy": n_correct_control / (n_correct_control + n_error_control)
                if (n_correct_control + n_error_control) > 0 else None}}


def restatement_reduction_synthetic_check(rng: np.random.Generator) -> dict:
    """Numeric verification, run as part of this artifact rather than only in the test suite, that
    ``cross_validated_predictable_fraction`` and ``in_sample_linear_fraction`` reduce to the delivered
    subspace-projection quantity ``(||P_S w|| / ||w||) ** 2`` in a noiseless linear scenario: ``y = X @
    w`` exactly, ``X`` isotropic, ``S`` an orthonormal basis unrelated to ``w``. Reported with the
    exact discrepancy numbers, not only a pass/fail assertion."""
    n, d, k = 4000, 10, 3
    X = rng.standard_normal((n, d))
    S, _ = np.linalg.qr(rng.standard_normal((d, k)))
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    y = X @ w
    Z = X @ S
    cv = cross_validated_predictable_fraction(y, Z, alpha=1e-6, rng=rng)
    naive = in_sample_linear_fraction(y, Z)
    P = S @ S.T
    delivered = float((np.linalg.norm(P @ w) / np.linalg.norm(w)) ** 2)
    return {
        "scenario": "noiseless linear generator y = X @ w, isotropic X (n=4000, d=10), rank-3 "
                    "orthonormal basis S unrelated to w, delivered quantity = (||P_S w||/||w||)**2",
        "delivered_subspace_quantity": delivered,
        "cross_validated_predictable_fraction": cv["predictable_fraction"],
        "in_sample_linear_fraction": naive["linear_fraction"],
        "discrepancy_cross_validated_vs_subspace": abs(cv["predictable_fraction"] - delivered),
        "discrepancy_in_sample_vs_subspace": abs(naive["linear_fraction"] - delivered),
    }


FIELD_BAND_HI_HZ = 40.0


FIELD_BAND_LO_HZ = 1.0


FIELD_MAINTENANCE_WINDOW_S = (-3.0, 0.0)


def _flatten(X):
    n, b, f = X.shape
    return X.reshape(n * b, f), n, b


MAJORITY_SIGNIFICANCE_THRESHOLD = 0.5


MICROSTIM_CANDIDATES = ("native_full_rank", "principal_components", "factor_analysis",
                        "gaussian_process_factor_analysis", "temporal_diffusion_embedding",
                        "time_contrastive_embedding")


OPERATING_RANK = DELIVERED_RANK


STATUS_VOCABULARY = {
    "settled_robust": "verdict agreed across every admissible estimator run; the claim stands "
                      "robust to the estimation choice, with the range of effect sizes stated",
    "escalated": "verdict disagreed across estimators and the claim completed its escalation run "
                 "under the pre-declared session budget",
    "requires_refit_budget_decision": "verdict disagreed and the sizing rule's required sample "
                                      "exceeds this pass's session budget cap; a human decides "
                                      "whether to raise the cap and spend it",
    "pending_cached_fit_reuse": "waiting on a cached fit covering the needed split (none existed "
                                "on this leg: admissibility caches hold scores, not latents)",
    "inconclusive_below_detection_floor": "an escalation sample whose minimum detectable "
                                          "difference exceeds the effect it audits; both numbers "
                                          "reported, never read as agreement or disagreement",
    "escalation_attempt_failed_resource_limit": "the escalation run was triggered and started but "
                                                "its fits did not complete on this machine; "
                                                "recorded as a resource limit, never as a "
                                                "statement about the sessions",
    "escalation_undefined_for_claim_quantity": "the claim's quantity has no definition under the "
                                               "tier-three estimator's representation, for every "
                                               "session; no sample size is computed and no budget "
                                               "cap, however large, would make this claim's "
                                               "escalation computable",
    "escalation_exceeds_available_sessions": "the sizing rule's required sample size is within "
                                             "this pass's session budget cap but exceeds the "
                                             "number of sessions in the corpus that have a "
                                             "computable native reference cell to draw from; no "
                                             "larger sample without replacement is possible, so "
                                             "raising the cap further would not help",
}


def class_mean_coordinates(latent_trial: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Orthonormal basis of the centred between-class mean structure inside any representation's
    coordinate space -- the coding-subspace structure expressed as regression coordinates rather
    than an ambient-angle projector, so one definition serves a linear projection and a nonlinear
    embedding alike."""
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    means = np.stack([latent_trial[labels == c].mean(axis=0) for c in classes])
    means_c = means - means.mean(axis=0)
    _, s, vt = np.linalg.svd(means_c, full_matrices=False)
    rank = int(np.sum(s > 1e-10))
    if rank < 1:
        return None
    return vt[:rank].T


def decide_claim_standing(verdict_keys: dict[str, str]) -> dict:
    """Pre-declared escalation trigger: a claim whose verdict is stable across every admissible
    estimator is settled and is not escalated; a claim whose verdict differs between estimators
    escalates, that claim alone. Every disagreement names the estimators on each side, and each
    estimator cell carries its own effect size beside its verdict."""
    computable = {name: key for name, key in verdict_keys.items()
                  if key not in (None, "not_applicable", "not_computable")}
    if not computable:
        return {"branch": "no_computable_estimator_cells", "unique_verdict_keys": [],
                "estimators_by_side": {}}
    unique = sorted(set(computable.values()))
    if len(unique) == 1:
        return {"branch": "verdict_confirmed_across_estimators",
                "unique_verdict_keys": unique, "estimators_by_side": {}}
    sides: dict[str, list[str]] = {}
    for name, key in sorted(computable.items()):
        sides.setdefault(key, []).append(name)
    return {"branch": "estimation_dependent_rung_three_escalation",
            "unique_verdict_keys": unique, "estimators_by_side": sides}


def _interval_sign(interval: list[float]) -> str:
    lo, hi = interval
    if lo > 0.0:
        return "positive"
    if hi < 0.0:
        return "negative"
    return "spans_zero"


def interval_agreement(estimators: dict[str, dict]) -> dict:
    """Cross-estimator agreement read off each estimator's own pooled-effect session-cluster
    interval, never off its verdict key: whether every computed interval shares a sign, and the
    pairwise overlap between every pair of estimator intervals."""
    intervals = {
        name: agg["pooled_effect_session_cluster_interval_95pct"]["interval_95pct"]
        for name, agg in estimators.items()
        if agg.get("pooled_effect_session_cluster_interval_95pct", {}).get("status") == "computed"
    }
    if len(intervals) < 2:
        return {"status": "not_computable", "n_estimators_with_interval": len(intervals)}
    signs = {name: _interval_sign(iv) for name, iv in intervals.items()}
    all_share_a_sign = len(set(signs.values())) == 1 and next(iter(signs.values())) != "spans_zero"
    names = sorted(intervals)
    pairwise_overlap = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = intervals[names[i]], intervals[names[j]]
            pairwise_overlap[f"{names[i]}__{names[j]}"] = bool(max(a[0], b[0]) <= min(a[1], b[1]))
    return {"status": "computed", "n_estimators_with_interval": len(intervals), "signs": signs,
            "all_intervals_share_a_sign": all_share_a_sign, "pairwise_interval_overlap": pairwise_overlap}


def restated_claim_cell(y: np.ndarray, coords: np.ndarray | None, labels: np.ndarray,
                        latent_trial: np.ndarray, null_kind: str,
                        rng: np.random.Generator, n_perm: int) -> dict:
    """Estimator-invariant restatement cell for one linear-geometry claim: the cross-validated
    predictable fraction of the deviation component's variance from ``latent_trial @ coords``.

    ``null_kind='label_permutation'`` mirrors the coding-subspace claim's delivered null: the
    class-mean coordinate structure is rebuilt under permuted item labels while the deviation
    target stays fixed. ``null_kind='y_shuffle'`` mirrors the occupied-manifold claim's
    random-direction baseline: the trial-to-state correspondence is broken while the
    representation stays fixed."""
    if coords is None or coords.shape[1] < 1:
        return {"status": "not_computable", "reason": "no usable coordinate structure"}
    base_z = latent_trial @ coords
    observed = cross_validated_predictable_fraction(y, base_z, rng=rng)
    if observed.get("status") != "computed":
        return {"status": observed.get("status", "not_computable"),
                "reason": observed.get("reason")}
    null_vals = []
    for _ in range(n_perm):
        if null_kind == "label_permutation":
            null_coords = class_mean_coordinates(latent_trial, rng.permutation(labels))
            if null_coords is None:
                continue
            null_z, null_y = latent_trial @ null_coords, y
        elif null_kind == "y_shuffle":
            null_z, null_y = base_z, rng.permutation(y)
        else:
            raise ValueError(f"unknown null_kind {null_kind!r}")
        entry = cross_validated_predictable_fraction(null_y, null_z, rng=rng)
        if entry.get("status") == "computed":
            null_vals.append(entry["predictable_fraction"])
    if not null_vals:
        return {"status": "not_computable", "reason": "empty null"}
    frac = observed["predictable_fraction"]
    null_mean = float(np.mean(null_vals))
    return {"status": "computed", "predictable_fraction": frac, "null_mean": null_mean,
            "effect_size": float(frac - null_mean),
            "p_value": permutation_pvalue(np.asarray(null_vals) >= frac),
            "null_kind": null_kind, "n_null": len(null_vals),
            "null_values": [float(v) for v in null_vals],  # raw draws, for cross-level pooling only
            "n_dims": int(coords.shape[1]), "n_trials": int(len(y)),
            "in_sample_linear_beside": in_sample_linear_fraction(y, base_z)}


def _blocks_for_levels(arrays: dict, levels: list[int], outcome_key: str, controls: tuple[str, ...]) -> list[dict]:
    blocks = []
    for level in levels:
        mask = arrays["item_count"] == float(level)
        n = int(mask.sum())
        blocks.append({
            "n": n, "predictor": arrays["deviation"][mask], "outcome": arrays[outcome_key][mask],
            "controls": [arrays[c][mask] for c in controls],
        })
    return blocks


def _cheap_partial_r(outcome: np.ndarray, covariate: np.ndarray, controls: list[np.ndarray]) -> float | None:
    """Point-estimate-only partial Pearson correlation by the residual method (OLS on
    [intercept, *controls], then Pearson on the two residual series) -- the identical formula
    statistics.partial_correlation_permutation_test uses for its own observed statistic, without that
    function's inner permutation loop, because this helper is called once per session on every one of
    N_PERM_WITHIN_SESSION outer draws, where re-running a full permutation test that many times nested
    inside another permutation test is not tractable."""
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    if controls:
        design = np.column_stack([np.ones(len(outcome)), *[np.asarray(c, dtype=float) for c in controls]])
        y_resid = outcome - design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
        x_resid = covariate - design @ np.linalg.lstsq(design, covariate, rcond=None)[0]
    else:
        y_resid, x_resid = outcome - outcome.mean(), covariate - covariate.mean()
    if np.std(y_resid) == 0.0 or np.std(x_resid) == 0.0:
        return None
    return float(np.corrcoef(y_resid, x_resid)[0, 1])


def _circular_outcome_shift(rng: np.random.Generator, block: dict) -> dict:
    shift = int(rng.integers(1, block["n"]))
    return {**block, "outcome": np.roll(block["outcome"], shift)}


def _circular_residual_shift(rng: np.random.Generator, block: dict) -> dict:
    if not block["controls"]:
        return _circular_outcome_shift(rng, block)
    design = np.column_stack([np.ones(block["n"]), *block["controls"]])
    fitted = design @ np.linalg.lstsq(design, block["outcome"], rcond=None)[0]
    residual = block["outcome"] - fitted
    shift = int(rng.integers(1, block["n"]))
    return {**block, "outcome": fitted + np.roll(residual, shift)}
