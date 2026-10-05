"""Per-subject decoding of a remembered orientation (and the cued item) from scalp-EEG channel features with the
memorandum decoding study's linear support vector machine and nested cross-validation inside each subject.

A subject is one recording. Features are channel means over windows of the preprocessed epochs. Representations are
the channels themselves, the channels selected by the category-selectivity test on training trials, principal
components and demixed principal components fitted on the encoding epoch of the training trials."""
from __future__ import annotations

import numpy as np

import memorandum_decoding as md

DECODER = "linear_svm"
EXPOSURE = 1.0
LATENT_RANK = 4
REPRESENTATIONS = ("native", "native_selected", "principal_components", "demixed_principal_components")
COURSE_WIDTHS_MS = (100, 250)
MIN_TRIAL_FOLDS = 2


def course_starts(duration_s: float, width_s: float) -> np.ndarray:
    """Window starts from the epoch's time zero, stepped by the study's window step, that fit inside the epoch."""
    return np.arange(0.0, duration_s - width_s + 1e-6, md.WINDOW_STEP_S)


def window_means(trials: np.ndarray, time: np.ndarray, starts: np.ndarray, width_s: float) -> np.ndarray:
    """(trials, windows, channels) means of (trials, channels, samples) over [start, start + width) of ``time``."""
    cumulative = np.concatenate([np.zeros(trials.shape[:2] + (1,)), np.cumsum(trials, axis=2)], axis=2)
    low = np.searchsorted(time, starts - 1e-9)
    high = np.searchsorted(time, starts + width_s - 1e-9)
    return ((cumulative[:, :, high] - cumulative[:, :, low]) / (high - low)).transpose(0, 2, 1)


def fit_transforms(encoding_course: np.ndarray, encoding_window: np.ndarray, labels: np.ndarray, rows: np.ndarray,
                   representations, rng: np.random.Generator, n_permutations: int | None = None) -> dict:
    """(mean, weights) of every representation, fitted on the given trial rows of the encoding epoch only.
    ``encoding_course`` is (trials, 100 ms bins, channels), ``encoding_window`` the (trials, channels) encoding-window
    means used for channel selection. Features are mapped as (x - mean) @ weights."""
    course, y = encoding_course[rows], labels[rows]
    n_channels = course.shape[2]
    mean = course.mean(axis=(0, 1))
    out = {}
    for name in representations:
        if name == "native":
            out[name] = (np.zeros(n_channels), np.eye(n_channels))
        elif name == "native_selected":
            keep = md.category_selective_mask(encoding_window[rows], y, rng, n_permutations)
            out[name] = (np.zeros(n_channels), np.eye(n_channels)[:, keep])
        elif name == "principal_components":
            _, _, vt = np.linalg.svd((course - mean).reshape(-1, n_channels), full_matrices=False)
            out[name] = (mean, vt[:LATENT_RANK].T)
        elif name == "demixed_principal_components":
            fit = md.fit_demixed_axes(course, y, rng, LATENT_RANK,
                                      fit_fn=lambda z, c, lam, d: md.ridge_marginal_axes(z, c, lam, d, "memorandum"))
            out[name] = (mean, fit["decoder"])
        else:
            raise ValueError(f"unknown representation {name}")
    return out


def apply_transform(x: np.ndarray, transform: tuple) -> np.ndarray:
    """(trials, channels, windows) to (trials, dimensions, windows)."""
    mean, weights = transform
    return np.einsum("mcw,cd->mdw", x - mean[None, :, None], weights)


def score_readout(train_x: np.ndarray, test_x: np.ndarray, valid: np.ndarray, y: np.ndarray, fold_ids: np.ndarray,
                  transforms: list, classes: np.ndarray, tuned: np.ndarray | None = None, seed: int = 0,
                  keep_scores: bool = False) -> dict:
    """Outer-fold balanced accuracy and one-versus-rest area under the curve per window. ``train_x`` and ``test_x``
    are (trials, channels, windows); the same windows are used for training and testing unless ``train_x`` has one
    window, which is then the training window for every test window. Hyperparameters are chosen on each outer
    fold's training rows only unless ``tuned`` (folds, windows) supplies them. ``transforms`` holds one fitted
    transform per outer fold."""
    n_folds, n_windows = len(transforms), test_x.shape[2]
    transfer = train_x.shape[2] == 1 and n_windows > 1
    n_picks = 1 if transfer else n_windows
    picks = np.zeros((n_folds, n_picks), dtype=int)
    collected, truth, fold_scores = [[] for _ in range(n_windows)], [], []
    for f in range(n_folds):
        train, test = valid & (fold_ids != f), valid & (fold_ids == f)
        xt, xe = apply_transform(train_x[train], transforms[f]), apply_transform(test_x[test], transforms[f])
        truth.append(y[test])
        scores_f = []
        for p in range(n_picks):
            if tuned is None:
                picks[f, p] = md.choose_hyperparameter(DECODER, xt[:, :, p], y[train], EXPOSURE, classes,
                                                       seed + 1000 * f + p)
            else:
                picks[f, p] = tuned[f, p]
        if transfer:
            flat = xe.transpose(0, 2, 1).reshape(xe.shape[0] * n_windows, xe.shape[1])
            stacked = md._fit_scores(DECODER, md.HYPER_GRIDS[DECODER][picks[f, 0]], xt[:, :, 0], y[train], flat,
                                     EXPOSURE, EXPOSURE, classes).reshape(xe.shape[0], n_windows, len(classes))
            scores_f = [stacked[:, w] for w in range(n_windows)]
        else:
            for w in range(n_windows):
                scores_f.append(md._fit_scores(DECODER, md.HYPER_GRIDS[DECODER][picks[f, w]], xt[:, :, w], y[train],
                                               xe[:, :, w], EXPOSURE, EXPOSURE, classes))
        for w in range(n_windows):
            collected[w].append(scores_f[w])
        fold_scores.append(scores_f)
    truth = np.concatenate(truth)
    balanced, auc = np.empty(n_windows), np.empty(n_windows)
    for w in range(n_windows):
        stacked = np.concatenate(collected[w])
        balanced[w] = md.balanced_accuracy(truth, classes[stacked.argmax(axis=1)], classes)
        auc[w] = md.one_vs_rest_auc(truth, stacked, classes)
    out = {"balanced_accuracy": balanced, "auc": auc, "picks": picks}
    if keep_scores:
        out["fold_scores"] = fold_scores
    return out


def build_readouts(bundle: dict, spec: dict, groups=None) -> list[dict]:
    """Read-outs of one subject. A bundle holds ``labels`` (name to per-trial integer vector) and ``epochs`` (name to
    ``valid`` mask, ``fixed`` (trials, channels) window means and ``course`` (trials, windows, channels) sliding
    window means by width in ms with their ``course_starts``). ``spec`` names the encoding epoch and window, the
    fixed windows as (epoch, window), and for every label family which label vector belongs to which epoch."""
    epochs, readouts = bundle["epochs"], []
    enc_epoch, enc_window = spec["encoding_window"]
    enc = epochs[enc_epoch]
    for family, vectors in spec["label_vectors"].items():
        def add(group, kind, name, epoch, train, test, valid, starts, width):
            readouts.append({"id": f"{family}|{kind}|{name}", "family": family, "group": group, "kind": kind,
                             "epoch": epoch, "label_vector": vectors[epoch], "train": train, "test": test,
                             "valid": valid, "window_start_s": starts, "window_width_s": width})
        for name, (epoch, key) in spec["fixed"].items():
            feature = epochs[epoch]["fixed"][key][:, :, None]
            add("window_means", "fixed", name, epoch, feature, feature, epochs[epoch]["valid"], [0.0], None)
            if epoch != enc_epoch:
                add("window_means", "transfer_fixed", name, epoch, enc["fixed"][enc_window][:, :, None], feature,
                    enc["valid"] & epochs[epoch]["valid"], [0.0], None)
        for width in COURSE_WIDTHS_MS:
            for epoch in spec["course_epochs"]:
                course = epochs[epoch]["course"][width].transpose(0, 2, 1)
                starts = epochs[epoch]["course_starts"][width].tolist()
                add(f"course_{width}ms", "course", epoch, epoch, course, course, epochs[epoch]["valid"], starts,
                    width / 1000)
                if epoch != enc_epoch:
                    add(f"transfer_{width}ms", "transfer_course", epoch, epoch, enc["fixed"][enc_window][:, :, None],
                        course, enc["valid"] & epochs[epoch]["valid"], starts, width / 1000)
    return [r for r in readouts if groups is None or r["group"] in groups]


def representations_for(family: str, spec: dict) -> tuple[str, ...]:
    return tuple(r for r in REPRESENTATIONS if r in spec["representations"][family])


def decode_group(bundle: dict, spec: dict, group: str, n_shuffles: int, seed: int, n_permutations: int | None = None) -> dict:
    """Observed scores and the within-subject label-shuffle null of every read-out of one group, for every
    representation of its label family. Fits and hyperparameters of the observed labels are reused in the shuffled
    runs; the shuffled labels are permutations of each label vector across all of the subject's trials."""
    epochs = bundle["epochs"]
    enc_epoch, enc_window = spec["encoding_window"]
    readouts = build_readouts(bundle, spec, (group,))
    out, transforms, fold_ids = {}, {}, {}
    names = sorted({r["label_vector"] for r in readouts})
    for name in names:
        y = bundle["labels"][name]
        n_folds = md.feasible_fold_count(y[epochs[enc_epoch]["valid"]])
        fold_ids[name] = md.stratified_fold_ids(y, n_folds, np.random.default_rng([seed, 1, names.index(name)]))
        families = {r["family"] for r in readouts if r["label_vector"] == name}
        wanted = sorted({rep for fam in families for rep in representations_for(fam, spec)}, key=REPRESENTATIONS.index)
        transforms[name] = {rep: [] for rep in wanted}
        for f in range(n_folds):
            rows = np.flatnonzero(epochs[enc_epoch]["valid"] & (fold_ids[name] != f))
            fitted = fit_transforms(epochs[enc_epoch]["course"][100], epochs[enc_epoch]["fixed"][enc_window], y, rows,
                                    wanted, np.random.default_rng([seed, 2, names.index(name), f]), n_permutations)
            for rep in wanted:
                transforms[name][rep].append(fitted[rep])
    permuted = {name: [np.random.default_rng([seed, 10_000 + s, names.index(name)]).permutation(bundle["labels"][name])
                       for s in range(n_shuffles)] for name in names}
    for ro in readouts:
        name = ro["label_vector"]
        y = bundle["labels"][name]
        classes = np.unique(y)
        cell = {"family": ro["family"], "group": group, "kind": ro["kind"], "epoch": ro["epoch"],
                "label_vector": name, "window_start_s": ro["window_start_s"], "window_width_s": ro["window_width_s"],
                "n_trials": int(ro["valid"].sum()), "n_classes": int(len(classes)), "n_folds": int(len(set(fold_ids[name]))), "representations": {}}
        for rep in representations_for(ro["family"], spec):
            fits = transforms[name][rep]
            observed = score_readout(ro["train"], ro["test"], ro["valid"], y, fold_ids[name], fits, classes,
                                     seed=seed)
            null = [score_readout(ro["train"], ro["test"], ro["valid"], permuted[name][s], fold_ids[name], fits,
                                  classes, tuned=observed["picks"], seed=seed) for s in range(n_shuffles)]
            cell["representations"][rep] = {
                "observed": {m: observed[m] for m in ("balanced_accuracy", "auc")},
                "null": {m: np.stack([n[m] for n in null]) if null else None for m in ("balanced_accuracy", "auc")},
                "mean_dimensions": float(np.mean([t[1].shape[1] for t in fits]))}
        out[ro["id"]] = cell
    return out
