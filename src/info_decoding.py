from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


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
