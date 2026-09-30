from __future__ import annotations

from collections.abc import Callable

import numpy as np


BasisFit = Callable[[np.ndarray, np.ndarray], np.ndarray | None]


def block_folds(n: int, n_folds: int) -> np.ndarray:
    if n_folds < 2 or n < n_folds:
        raise ValueError("n_folds must be between 2 and n")
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds = np.empty(n, dtype=int)
    for fold, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        folds[start:stop] = fold
    return folds


def residual_axis(reference_directions: np.ndarray, evaluation_directions: np.ndarray) -> np.ndarray | None:
    reference_directions = np.asarray(reference_directions, dtype=float)
    directions = np.asarray(evaluation_directions, dtype=float)
    if (
        reference_directions.ndim != 2
        or directions.ndim != 2
        or reference_directions.shape[1] != directions.shape[1]
        or reference_directions.shape[0] < 2
        or directions.shape[0] < 3
    ):
        return None
    reference = reference_directions.mean(axis=0)
    norm = np.linalg.norm(reference)
    if not np.isfinite(reference).all() or norm <= 0:
        return None
    reference = reference / norm
    valid = np.isfinite(directions).all(axis=1)
    rows = directions[valid] - np.sum(directions[valid] * reference, axis=1, keepdims=True) * reference
    row_norms = np.linalg.norm(rows, axis=1)
    rows = rows[row_norms > 1e-12]
    row_norms = row_norms[row_norms > 1e-12]
    if rows.shape[0] < 3:
        return None
    rows = rows / row_norms[:, None]
    values, vectors = np.linalg.eigh(rows.T @ rows)
    if values[-1] <= 0:
        return None
    return vectors[:, -1]


def class_basis(directions: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return None
    centred = directions - directions.mean(axis=0)
    means = np.stack([centred[labels == label].mean(axis=0) for label in classes])
    return _basis(means, min(classes.size - 1, directions.shape[1]))


def regression_basis(directions: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    target = target - target.mean(axis=0)
    directions = directions - directions.mean(axis=0)
    if np.linalg.matrix_rank(target) == 0:
        return None
    coef, *_ = np.linalg.lstsq(target, directions, rcond=None)
    return _basis(coef, min(target.shape[1], directions.shape[1]))


def crossfit_alignment(
    directions: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    fit_basis: BasisFit,
    axes: dict[int | float | str, np.ndarray | None] | None = None,
) -> dict:
    directions = np.asarray(directions, dtype=float)
    target = np.asarray(target)
    folds = np.asarray(folds)
    if directions.ndim != 2 or len(directions) != len(target) or len(folds) != len(target):
        raise ValueError("directions, target, and folds must share the trial axis")
    scores = []
    weights = []
    unique_folds = np.unique(folds)
    for fold in unique_folds:
        test = folds == fold
        train = ~test
        axis = residual_axis(directions[train], directions[test]) if axes is None else axes.get(fold)
        basis = fit_basis(directions[train], target[train])
        if axis is None or basis is None:
            continue
        score = float(np.sum((basis.T @ axis) ** 2))
        scores.append(min(max(score, 0.0), 1.0))
        weights.append(int(test.sum()))
    if len(scores) != len(unique_folds):
        return {
            "status": "not_computable",
            "n_folds": len(scores),
            "n_folds_requested": len(unique_folds),
            "reason": "at least one fold could not define both the residual axis and label basis",
        }
    score = float(np.average(scores, weights=weights))
    return {
        "status": "computed",
        "alignment": score,
        "n_folds": len(scores),
        "n_folds_requested": len(unique_folds),
        "fold_alignment": scores,
        "fold_weight": weights,
    }


def permutation_alignment(
    directions: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    fit_basis: BasisFit,
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    directions = np.asarray(directions, dtype=float)
    target = np.asarray(target)
    folds = np.asarray(folds)
    axes = {}
    for fold in np.unique(folds):
        test = folds == fold
        axes[fold] = residual_axis(directions[~test], directions[test])
    observed = crossfit_alignment(directions, target, folds, fit_basis, axes)
    if observed["status"] != "computed":
        return observed
    draws = np.empty(n_perm, dtype=float)
    valid = np.ones(n_perm, dtype=bool)
    unique_folds = np.unique(folds)
    for draw in range(n_perm):
        permuted = np.array(target, copy=True)
        for fold in unique_folds:
            indices = np.flatnonzero(folds == fold)
            permuted[indices] = target[indices[rng.permutation(len(indices))]]
        result = crossfit_alignment(directions, permuted, folds, fit_basis, axes)
        if result["status"] == "computed":
            draws[draw] = result["alignment"]
        else:
            valid[draw] = False
    draws = draws[valid]
    if draws.size == 0:
        return {"status": "not_computable", "n_folds": observed["n_folds"]}
    score = observed["alignment"]
    return {
        **observed,
        "null_mean": float(draws.mean()),
        "alignment_above_null": float(score - draws.mean()),
        "p_value": float((1 + np.sum(draws >= score)) / (draws.size + 1)),
        "n_permutations": int(draws.size),
        "null_interval_95pct": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
        "null_draws": draws,
    }


def _basis(rows: np.ndarray, dim: int) -> np.ndarray | None:
    if rows.size == 0 or dim < 1:
        return None
    _, singular, vectors = np.linalg.svd(rows, full_matrices=False)
    keep = min(dim, int(np.sum(singular > 1e-10)))
    return vectors[:keep].T if keep else None


N_CV_FOLDS = 5


def _contiguous_folds(n: int, k: int) -> np.ndarray:
    edges = np.linspace(0, n, k + 1).astype(int)
    fold = np.empty(n, dtype=int)
    for f in range(k):
        fold[edges[f]:edges[f + 1]] = f
    return fold


def leading_eigenvector(R: np.ndarray) -> np.ndarray:
    _w, v = np.linalg.eigh(R.T @ R)
    return v[:, -1]


def _orthonormal_basis(row_vectors: np.ndarray, dim: int) -> np.ndarray:
    """(units,) orthonormal basis of the row space of ``row_vectors`` (k,
    units), top ``dim`` left singular vectors of its transpose."""
    u_svd, _, _ = np.linalg.svd(row_vectors.T, full_matrices=False)
    return u_svd[:, :dim]


def _class_mean_subspace_basis(U: np.ndarray, labels: np.ndarray, dim: int) -> np.ndarray | None:
    classes = np.unique(labels)
    if len(classes) < dim + 1:
        return None
    centred = U - U.mean(axis=0)
    class_means = np.stack([centred[labels == c].mean(axis=0) for c in classes])
    return _orthonormal_basis(class_means, dim)


def _regression_subspace_basis(U: np.ndarray, target_2d: np.ndarray, dim: int) -> np.ndarray | None:
    x_c = target_2d - target_2d.mean(axis=0)
    u_c = U - U.mean(axis=0)
    coef, *_ = np.linalg.lstsq(x_c, u_c, rcond=None)
    return _orthonormal_basis(coef, dim)
