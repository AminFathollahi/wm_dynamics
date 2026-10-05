"""Memorandum decoding from human medial temporal single units: session loading, category-selective
unit selection, pseudo-population assembly and nested cross-validated read-outs."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC

from corpus_sessions import (
    DANDI_000004_RESPONSE_RANGE, EPOCH_WINDOWS_S, MIN_UNITS_POOLED, _trial_group, recognition_correct,
)
from info_decoding import _category_469, _category_divided
from info_decoding import DPCA_RIDGE_LAMBDA_GRID, N_CV_FOLDS_LAMBDA, anscombe_counts
from project_config import dataset_path
from provenance import canonical_patient_by_relative_path, linked_duplicate_000673_sessions
from spike_pipeline import MIN_UNITS_PER_REGION, load_spike_times, low_rate_unit_mask, resolve_unit_regions

REGIONS = ("hippocampus", "amygdala", "pooled")
UNIT_REGIONS = ("hippocampus", "amygdala")
TASK_RATE_FLOOR_HZ = 0.05
SELECTION_LEVEL = 0.05
SELECTION_PERMUTATIONS = 1000
MAINTENANCE_S = float(EPOCH_WINDOWS_S["delay"])
ENCODING_SEGMENT_S = 1.2
WINDOW_STEP_S = 0.1
SELECTION_WINDOW_S = (0.2, 1.2)
INNER_FOLDS = 3
MIN_CLASS_TRIALS = 5
PSEUDO_TRAIN_PER_CATEGORY = 60
PSEUDO_TEST_PER_CATEGORY = 20
MIN_TEST_TRIALS_PER_CATEGORY = 1
MIN_TRAIN_TRIALS_PER_CATEGORY = 2
PSEUDO_FOLDS = 5

RELEASES = {
    "001187": {"dataset": "dandi_001187", "glob": "sub-*/*_ecephys*.nwb", "pictures": "PicIDs_Encoding1",
               "category": _category_divided},
    "000673": {"dataset": "dandi_000673", "glob": "sub-*/*_ecephys*.nwb", "pictures": "PicIDs_Encoding1",
               "category": _category_divided},
    "000469": {"dataset": "dandi_000469", "glob": "sub-*/*_ses-2_ecephys+image.nwb", "pictures": "loadsEnc1_PicIDs",
               "category": _category_469},
}


def _window_starts(length_s: float, width_s: float) -> np.ndarray:
    return np.arange(0.0, length_s - width_s + 1e-9, WINDOW_STEP_S)


FEATURES = {
    "selection_window": {"align": "picture", "starts": np.array([SELECTION_WINDOW_S[0]]),
                         "width": round(SELECTION_WINDOW_S[1] - SELECTION_WINDOW_S[0], 6)},
    "maintenance_whole": {"align": "maintenance", "starts": np.array([0.0]), "width": MAINTENANCE_S},
    **{f"maintenance_{ms}ms": {"align": "maintenance", "starts": _window_starts(MAINTENANCE_S, ms / 1000),
                               "width": ms / 1000} for ms in (100, 250, 500)},
    **{f"encoding_{ms}ms": {"align": "picture", "starts": _window_starts(ENCODING_SEGMENT_S, ms / 1000),
                            "width": ms / 1000} for ms in (100, 250)},
}
KINDS = {
    "maintenance_whole": ("maintenance_whole", "maintenance_whole"),
    "maintenance_100ms": ("maintenance_100ms", "maintenance_100ms"),
    "maintenance_250ms": ("maintenance_250ms", "maintenance_250ms"),
    "maintenance_500ms": ("maintenance_500ms", "maintenance_500ms"),
    "encoding_100ms": ("encoding_100ms", "encoding_100ms"),
    "encoding_250ms": ("encoding_250ms", "encoding_250ms"),
    "transfer_100ms": ("selection_window", "maintenance_100ms"),
    "transfer_250ms": ("selection_window", "maintenance_250ms"),
}
GRID_KINDS = ("maintenance_whole", "maintenance_100ms", "maintenance_250ms", "maintenance_500ms")

DECODERS = ("linear_svm", "poisson_naive_bayes", "radial_svm")
HYPER_GRIDS = {
    "linear_svm": [0.001, 0.01, 0.1, 1.0],
    "poisson_naive_bayes": [0.01, 0.1, 1.0],
    "radial_svm": [(c, g) for c in (1.0, 10.0) for g in (0.3, 1.0, 3.0)],
}


def window_counts(spikes: list[np.ndarray], onsets: np.ndarray, starts: np.ndarray, width: float) -> np.ndarray:
    """Spike counts in [onset + start, onset + start + width) for every trial, unit and start."""
    low = onsets[:, None] + starts[None, :]
    out = np.empty((len(onsets), len(spikes), len(starts)))
    for unit, times in enumerate(spikes):
        out[:, unit, :] = np.searchsorted(times, low + width) - np.searchsorted(times, low)
    return out


def session_manifest(provenance_dir: str | Path) -> list[dict]:
    patient_by_path = canonical_patient_by_relative_path(provenance_dir)
    duplicates = linked_duplicate_000673_sessions(provenance_dir)
    rows, claimed = [], {}
    for release, spec in RELEASES.items():
        for path in sorted(dataset_path(spec["dataset"]).glob(spec["glob"])):
            patient = patient_by_path.get(f"{release}/{path.parent.name}/{path.name}")
            row = {"release": release, "session": path.stem, "path": str(path), "patient": patient}
            if release == "000673" and path.stem in duplicates:
                row["status"] = "linked_duplicate_000673_session"
            elif patient is None:
                row["status"] = "no_canonical_patient"
            elif claimed.setdefault(patient, release) != release:
                row["status"] = f"patient_listed_under_{claimed[patient]}"
            else:
                row["status"] = "listed"
            rows.append(row)
    return rows


def load_session(path: str | Path, release: str, patient: str) -> tuple[dict | None, str]:
    spec = RELEASES[release]
    with h5py.File(str(path), "r") as handle:
        if "units" not in handle or "intervals" not in handle:
            return None, "missing_units_or_trials"
        trials = _trial_group(handle, release)
        spikes = [np.sort(times) for times in load_spike_times(handle)]
        regions = np.asarray(resolve_unit_regions(handle)["region"])
        pictures = trials[spec["pictures"]][:]
        loads = trials["loads"][:].astype(int)
        correct = trials["response_accuracy"][:].astype(bool)
        t_picture = trials["timestamps_Encoding1"][:].astype(float)
        t_maintenance = trials["timestamps_Maintenance"][:].astype(float)
        task_start = float(np.nanmin(trials["timestamps_FixationCross"][:]))
        task_end = float(np.nanmax(trials["timestamps_Response"][:]))
    if not task_end > task_start:
        return None, "empty_task_span"
    task_rate_hz = np.array([(np.searchsorted(t, task_end) - np.searchsorted(t, task_start)) / (task_end - task_start)
                             for t in spikes])
    return {"patient": patient, "release": release, "session": Path(path).stem, "spikes": spikes,
            "regions": regions, "loads": loads, "correct": correct, "category": spec["category"](pictures),
            "t_picture": t_picture, "t_maintenance": t_maintenance, "task_rate_hz": task_rate_hz}, "loaded"


def unit_inclusion_counts(session: dict, load: int) -> dict:
    in_load = session["loads"] == load
    out = {}
    for region in REGIONS:
        in_region = np.isin(session["regions"], UNIT_REGIONS if region == "pooled" else (region,))
        new_rule = in_region & (session["task_rate_hz"] >= TASK_RATE_FLOOR_HZ)
        spikes = [t for t, keep in zip(session["spikes"], in_region) if keep]
        old_mask = low_rate_unit_mask(spikes, session["t_maintenance"][in_load], MAINTENANCE_S) if spikes else []
        old_units = int(np.sum(old_mask))
        floor = MIN_UNITS_POOLED if region == "pooled" else MIN_UNITS_PER_REGION
        out[region] = {"units_in_region": int(in_region.sum()), "units_task_rate_floor": int(new_rule.sum()),
                       "units_previous_rule": old_units if old_units >= floor else 0}
    return out


def build_entry(session: dict, load: int, feature_keys=tuple(FEATURES)) -> dict | None:
    keep = (session["loads"] == load) & session["correct"]
    units = np.flatnonzero(np.isin(session["regions"], UNIT_REGIONS) & (session["task_rate_hz"] >= TASK_RATE_FLOOR_HZ))
    if keep.sum() == 0 or len(units) == 0:
        return None
    spikes = [session["spikes"][u] for u in units]
    onsets = {"picture": session["t_picture"][keep], "maintenance": session["t_maintenance"][keep]}
    features = {key: window_counts(spikes, onsets[FEATURES[key]["align"]], FEATURES[key]["starts"],
                                   FEATURES[key]["width"]) for key in feature_keys}
    return {"patient": session["patient"], "release": session["release"], "session": session["session"],
            "load": load, "labels": session["category"][keep], "unit_region": session["regions"][units],
            "features": features}


def feasible_fold_count(labels: np.ndarray) -> int | None:
    """10 folds when every class has at least 10 trials, 5 when every class has at least 5. A stratified
    split cannot place a class in more folds than it has trials, so a smaller class is infeasible."""
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return None
    if counts.min() >= 10:
        return 10
    return 5 if counts.min() >= MIN_CLASS_TRIALS else None


def region_columns(entry: dict, region: str) -> np.ndarray:
    return np.flatnonzero(np.isin(entry["unit_region"], UNIT_REGIONS if region == "pooled" else (region,)))


def stratified_fold_ids(labels: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    ids = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        ids[members] = (np.arange(len(members)) + rng.integers(n_folds)) % n_folds
    return ids


def category_selective_mask(rates: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                            n_perm: int | None = None, level: float | None = None) -> np.ndarray:
    """Per unit: permutation one-way analysis of variance across categories, and a right-tailed permutation
    test of the preferred category's mean against the rest. Both use the same label permutations."""
    n_perm = SELECTION_PERMUTATIONS if n_perm is None else n_perm
    level = SELECTION_LEVEL if level is None else level
    levels, counts = np.unique(labels, return_counts=True)
    usable = levels[counts >= 2]
    if len(usable) < 2:
        return np.zeros(rates.shape[1], dtype=bool)
    keep = np.isin(labels, usable)
    values, labels = rates[keep], labels[keep]
    n_trials = len(labels)
    onehot = (labels[:, None] == usable[None, :]).astype(float)
    sizes = onehot.sum(axis=0)
    total = values.sum(axis=0)
    grand = total / n_trials

    def statistics(sums):
        means = sums / sizes[:, None]
        between = (sizes[:, None] * (means - grand) ** 2).sum(axis=-2)
        best = means.argmax(axis=-2)
        best_sum = np.take_along_axis(sums, best[..., None, :], axis=-2)[..., 0, :]
        return between, best_sum / sizes[best] - (total - best_sum) / (n_trials - sizes[best])

    observed_between, observed_diff = statistics(onehot.T @ values)
    permuted = onehot[np.argsort(rng.random((n_perm, n_trials)), axis=1)]
    null_between, null_diff = statistics(np.einsum("pnc,nu->pcu", permuted, values, optimize=True))
    p_between = (1 + (null_between >= observed_between).sum(axis=0)) / (n_perm + 1)
    p_preferred = (1 + (null_diff >= observed_diff).sum(axis=0)) / (n_perm + 1)
    return (p_between < level) & (p_preferred < level)


def balanced_accuracy(truth: np.ndarray, predicted: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([np.mean(predicted[truth == c] == c) for c in classes]))


def one_vs_rest_auc(truth: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> float:
    aucs = []
    for k, c in enumerate(classes):
        positive = truth == c
        n_pos, n_neg = positive.sum(), (~positive).sum()
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(scores[:, k])
        aucs.append((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else float("nan")


def _fit_scores(decoder: str, hyper, train: np.ndarray, labels: np.ndarray, test: np.ndarray,
                train_exposure: float, test_exposure: float, classes: np.ndarray) -> np.ndarray:
    n_features = train.shape[1]
    if n_features == 0:
        return np.zeros((len(test), len(classes)))
    if decoder == "poisson_naive_bayes":
        sums = np.stack([train[labels == c].sum(axis=0) for c in classes])
        sizes = np.array([(labels == c).sum() for c in classes])[:, None]
        expected = (sums + hyper) / (sizes * train_exposure) * test_exposure
        return test @ np.log(expected).T - expected.sum(axis=1)
    train, test = train / train_exposure, test / test_exposure
    mean, spread = train.mean(axis=0), train.std(axis=0)
    spread[spread < 1e-12] = 1.0
    train, test = (train - mean) / spread, (test - mean) / spread
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if decoder == "linear_svm":
            model = LinearSVC(C=hyper, dual="auto", max_iter=2000).fit(train, labels)
        else:
            model = SVC(C=hyper[0], gamma=hyper[1] / n_features, kernel="rbf").fit(train, labels)
    scores = model.decision_function(test)
    return np.stack([-scores, scores], axis=1) if scores.ndim == 1 else scores


def choose_hyperparameter(decoder: str, train: np.ndarray, labels: np.ndarray, train_exposure: float,
                          classes: np.ndarray, seed: int) -> int:
    """Index into the decoder's grid with the best mean inner balanced accuracy; only the rows passed in
    are ever read."""
    inner = stratified_fold_ids(labels, INNER_FOLDS, np.random.default_rng(seed))
    best, best_score = 0, -np.inf
    for index, hyper in enumerate(HYPER_GRIDS[decoder]):
        scores = []
        for fold in range(INNER_FOLDS):
            fit, held = inner != fold, inner == fold
            predicted = classes[_fit_scores(decoder, hyper, train[fit], labels[fit], train[held],
                                            train_exposure, train_exposure, classes).argmax(axis=1)]
            scores.append(balanced_accuracy(labels[held], predicted, classes))
        if np.mean(scores) > best_score + 1e-12:
            best, best_score = index, float(np.mean(scores))
    return best


def evaluate_folds(folds: list[dict], kinds, decoders, classes: np.ndarray, tuned: dict | None = None,
                   seed: int = 0) -> tuple[dict, dict]:
    """Outer-fold scores for every decoder, time feature and window. Each fold holds ``y_train``, ``y_test``
    and ``train``/``test`` dictionaries of (rows, units, windows) count arrays. Hyperparameters are chosen
    on the fold's training rows only unless ``tuned`` supplies them."""
    scores, chosen = {}, {}
    for decoder in decoders:
        scores[decoder], chosen[decoder] = {}, {}
        for kind in kinds:
            train_key, test_key = KINDS[kind]
            train_exposure, test_exposure = FEATURES[train_key]["width"], FEATURES[test_key]["width"]
            n_windows = folds[0]["test"][test_key].shape[2]
            transfer = train_key != test_key
            picks = np.zeros((len(folds), n_windows if not transfer else 1), dtype=int)
            collected = [[] for _ in range(n_windows)]
            for f, fold in enumerate(folds):
                for w in range(n_windows):
                    if transfer and w > 0:
                        pick = picks[f, 0]
                    else:
                        train = fold["train"][train_key][:, :, 0 if transfer else w]
                        if tuned is not None:
                            pick = tuned[decoder][kind][f, 0 if transfer else w]
                        else:
                            pick = choose_hyperparameter(decoder, train, fold["y_train"], train_exposure, classes,
                                                         seed + 1000 * f + w)
                        picks[f, 0 if transfer else w] = pick
                    train = fold["train"][train_key][:, :, 0 if transfer else w]
                    collected[w].append(_fit_scores(
                        decoder, HYPER_GRIDS[decoder][pick], train, fold["y_train"],
                        fold["test"][test_key][:, :, w], train_exposure, test_exposure, classes))
            truth = np.concatenate([fold["y_test"] for fold in folds])
            balanced, auc = np.empty(n_windows), np.empty(n_windows)
            for w in range(n_windows):
                stacked = np.concatenate(collected[w])
                balanced[w] = balanced_accuracy(truth, classes[stacked.argmax(axis=1)], classes)
                auc[w] = one_vs_rest_auc(truth, stacked, classes)
            scores[decoder][kind] = {"balanced_accuracy": balanced, "auc": auc}
            chosen[decoder][kind] = picks
    return scores, chosen


def _required_keys(kinds, unit_set: str) -> list[str]:
    keys = {key for kind in kinds for key in KINDS[kind]}
    if unit_set == "selective":
        keys.add("selection_window")
    return sorted(keys)


def session_folds(entry: dict, region: str, unit_set: str, kinds, n_folds: int, rng: np.random.Generator,
                  labels: np.ndarray | None = None) -> tuple[list[dict], np.ndarray]:
    labels = entry["labels"] if labels is None else labels
    columns = region_columns(entry, region)
    fold_ids = stratified_fold_ids(labels, n_folds, rng)
    keys = _required_keys(kinds, unit_set)
    folds = []
    for f in range(n_folds):
        train, test = np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f)
        used = columns
        if unit_set == "selective":
            rates = entry["features"]["selection_window"][train][:, columns, 0]
            used = columns[category_selective_mask(rates, labels[train], rng)]
        folds.append({"y_train": labels[train], "y_test": labels[test], "index_train": train, "index_test": test,
                      "n_units": len(used),
                      "train": {k: entry["features"][k][train][:, used, :] for k in keys},
                      "test": {k: entry["features"][k][test][:, used, :] for k in keys}})
    return folds, fold_ids


def pseudo_population_folds(entries: list[dict], region: str, unit_set: str, kinds, n_folds: int,
                            rng: np.random.Generator, labels_by_entry: list[np.ndarray] | None = None,
                            max_units: int | None = None, max_train_per_class: int | None = None) -> list[dict]:
    """Inside each fold every entry's trials of each category are split into training and test trials; the
    entry's units are dropped from the fold when some category has too few test or training trials. Pseudo-trials
    are then drawn with replacement, category-matched, independently for every unit: PSEUDO_TRAIN_PER_CATEGORY
    from the unit's own training trials and PSEUDO_TEST_PER_CATEGORY from its own test trials. Selection uses
    the entry's training trials only."""
    labels_by_entry = [e["labels"] for e in entries] if labels_by_entry is None else labels_by_entry
    classes = np.unique(np.concatenate(labels_by_entry))
    keys = _required_keys(kinds, unit_set)
    fold_ids = [stratified_fold_ids(lab, n_folds, rng) for lab in labels_by_entry]
    columns = [region_columns(e, region) for e in entries]
    counts = {"train": PSEUDO_TRAIN_PER_CATEGORY, "test": PSEUDO_TEST_PER_CATEGORY}
    n_windows = {k: entries[0]["features"][k].shape[2] for k in keys}
    folds = []
    for f in range(n_folds):
        dropped = {"too_few_test_trials": 0, "too_few_training_trials": 0}
        parts = {"train": {k: [] for k in keys}, "test": {k: [] for k in keys}}
        for e, entry in enumerate(entries):
            labels, ids = labels_by_entry[e], fold_ids[e]
            used = columns[e]
            if not len(used):
                continue
            pools = {"train": [np.flatnonzero((labels == c) & (ids != f)) for c in classes],
                     "test": [np.flatnonzero((labels == c) & (ids == f)) for c in classes]}
            if min(len(p) for p in pools["test"]) < MIN_TEST_TRIALS_PER_CATEGORY:
                dropped["too_few_test_trials"] += len(used)
                continue
            if min(len(p) for p in pools["train"]) < MIN_TRAIN_TRIALS_PER_CATEGORY:
                dropped["too_few_training_trials"] += len(used)
                continue
            if unit_set == "selective":
                train_rows = np.flatnonzero(ids != f)
                rates = entry["features"]["selection_window"][train_rows][:, used, 0]
                used = used[category_selective_mask(rates, labels[train_rows], rng)]
            for part in ("train", "test"):
                pool_list = pools[part]
                if part == "train" and max_train_per_class is not None:
                    pool_list = [rng.permutation(p)[:max_train_per_class] for p in pool_list]
                rows = np.concatenate([rng.choice(p, size=(counts[part], len(used)), replace=True) for p in pool_list])
                for key in keys:
                    parts[part][key].append(entry["features"][key][rows, used[None, :], :])
        fold = {"y_train": np.repeat(classes, counts["train"]), "y_test": np.repeat(classes, counts["test"]),
                "train": {}, "test": {}, "dropped_units": dropped}
        for part in ("train", "test"):
            for key in keys:
                fold[part][key] = (np.concatenate(parts[part][key], axis=1) if parts[part][key]
                                   else np.zeros((len(fold["y_" + part]), 0, n_windows[key])))
        n_units = fold["train"][keys[0]].shape[1]
        if max_units is not None and n_units > max_units:
            keep = np.sort(rng.choice(n_units, max_units, replace=False))
            for part in ("train", "test"):
                fold[part] = {key: value[:, keep, :] for key, value in fold[part].items()}
            n_units = max_units
        fold["n_units"] = n_units
        folds.append(fold)
    return folds


def bin_window_members(kind: str, n_bins: int, bin_s: float = WINDOW_STEP_S) -> list[np.ndarray]:
    """Bins whose centre lies inside each window of a maintenance time feature."""
    spec = FEATURES[KINDS[kind][1]]
    centres = (np.arange(n_bins) + 0.5) * bin_s
    return [np.flatnonzero((centres >= s) & (centres < s + spec["width"])) for s in spec["starts"]]


def aggregate_bins(series: np.ndarray, kind: str) -> np.ndarray:
    """(rows, bins, dimensions) trajectories to (rows, dimensions, windows) window means."""
    return np.stack([series[:, members, :].mean(axis=1) for members in bin_window_members(kind, series.shape[1])],
                    axis=2)


def stack_scores(runs: list[dict]) -> dict:
    return {decoder: {kind: {metric: np.stack([run[decoder][kind][metric] for run in runs])
                             for metric in ("balanced_accuracy", "auc")} for kind in runs[0][decoder]}
            for decoder in runs[0]}


def observed_runs(make_folds, kinds, decoders, classes: np.ndarray, n_runs: int, seed: int) -> dict:
    """Score ``n_runs`` independent resamples. Hyperparameters are chosen inside each run's training folds;
    the first run's choices are returned for reuse by the shuffled-label runs."""
    runs, tuned, n_units, dropped = [], None, [], []
    for r in range(n_runs):
        folds = make_folds(r, np.random.default_rng([seed, r]))
        scores, picks = evaluate_folds(folds, kinds, decoders, classes, seed=seed + r)
        tuned = picks if tuned is None else tuned
        runs.append(scores)
        n_units.append(float(np.mean([fold["n_units"] for fold in folds])))
        dropped.append({reason: float(np.mean([fold.get("dropped_units", {}).get(reason, 0) for fold in folds]))
                        for reason in ("too_few_test_trials", "too_few_training_trials")})
    return {"scores": stack_scores(runs), "tuned": tuned, "mean_units": n_units, "mean_dropped_units": dropped}


def shuffled_runs(make_folds, kinds, decoders, classes: np.ndarray, tuned: dict, indices, seed: int) -> dict:
    """Score label-shuffled resamples with the hyperparameters chosen on the observed labels."""
    runs = []
    for s in indices:
        folds = make_folds(s, np.random.default_rng([seed, 10_000 + s]))
        runs.append(evaluate_folds(folds, kinds, decoders, classes, tuned=tuned, seed=seed + s)[0])
    return stack_scores(runs)


def session_make_folds(entry: dict, region: str, unit_set: str, kinds, n_folds: int, shuffled: bool):
    def make(index, rng):
        labels = rng.permutation(entry["labels"]) if shuffled else entry["labels"]
        return session_folds(entry, region, unit_set, kinds, n_folds, rng, labels=labels)[0]
    return make


def pseudo_make_folds(entries: list[dict], region: str, unit_set: str, kinds, n_folds: int, shuffled: bool, **options):
    def make(index, rng):
        labels = [rng.permutation(e["labels"]) if shuffled else e["labels"] for e in entries]
        return pseudo_population_folds(entries, region, unit_set, kinds, n_folds, rng, labels_by_entry=labels,
                                       **options)
    return make


def latent_folds(folds: list[dict], representation: str, fits: dict, kind: str, seed: int) -> list[dict] | str:
    """Replace each fold's count features by the representation's window-averaged latent state. Returns a
    reason string when any fit fails."""
    key = KINDS[kind][0]
    out = []
    for f, fold in enumerate(folds):
        train = fold["train"]["maintenance_100ms"].transpose(0, 2, 1)
        test = fold["test"]["maintenance_100ms"].transpose(0, 2, 1)
        if representation == "native":
            latent_train, latent_test = train, test
        else:
            if train.shape[2] < 3:
                return "fewer_than_3_units_in_fold"
            fit = fits[representation](train, test, fold["y_train"], seed + f)
            if fit.get("status") != "fitted":
                return str(fit.get("reason") or fit.get("status"))
            latent_train, latent_test = np.asarray(fit["latent_train"]), np.asarray(fit["latent_test"])
            if latent_train.shape[1] != train.shape[1] or latent_test.shape[1] != test.shape[1]:
                return "latent_time_axis_differs_from_input"
        out.append({**fold, "train": {key: aggregate_bins(latent_train, kind)},
                    "test": {key: aggregate_bins(latent_test, kind)}})
    return out


def latent_cell(entry: dict, region: str, unit_set: str, representations, fits: dict, kind: str, decoder: str,
                n_folds: int, n_shuffles: int, seed: int) -> dict:
    """Latent-state read-out for one entry: the same folds, unit selection and label shuffles for every
    representation, so representations are paired against native counts. Shuffles reuse the fits and the
    observed hyperparameters."""
    base, _ = session_folds(entry, region, unit_set, ("maintenance_100ms",), n_folds, np.random.default_rng(seed))
    classes = np.unique(entry["labels"])
    shuffles = [np.random.default_rng([seed, 10_000 + s]).permutation(entry["labels"]) for s in range(n_shuffles)]
    out = {}
    for representation in representations:
        folds = latent_folds(base, representation, fits, kind, seed)
        if isinstance(folds, str):
            out[representation] = {"status": "not_estimable", "reason": folds}
            continue
        observed, tuned = evaluate_folds(folds, (kind,), (decoder,), classes, seed=seed)
        null = [evaluate_folds([{**fold, "y_train": perm[fold["index_train"]], "y_test": perm[fold["index_test"]]}
                                for fold in folds], (kind,), (decoder,), classes, tuned=tuned, seed=seed)[0]
                for perm in shuffles]
        out[representation] = {"status": "computed", "observed": stack_scores([observed]),
                               "null": stack_scores(null) if null else None}
    return out


def pseudo_latent_cell(entries: list[dict], region: str, unit_set: str, representations, fits: dict, kind: str,
                       decoder: str, n_folds: int, n_resamples: int, n_shuffles: int, seed: int) -> dict:
    """Principal and demixed principal components fitted on training pseudo-trials. Shuffles permute the
    pseudo-trial labels of the first resample's fits."""
    classes = np.unique(np.concatenate([e["labels"] for e in entries]))
    out = {}
    for representation in representations:
        runs, first, tuned = [], None, None
        for r in range(n_resamples):
            base = pseudo_population_folds(entries, region, unit_set, ("maintenance_100ms",), n_folds,
                                           np.random.default_rng([seed, r]))
            folds = latent_folds(base, representation, fits, kind, seed + r)
            if isinstance(folds, str):
                out[representation] = {"status": "not_estimable", "reason": folds}
                break
            scores, picks = evaluate_folds(folds, (kind,), (decoder,), classes, seed=seed + r)
            runs.append(scores)
            first, tuned = (folds, picks) if first is None else (first, tuned)
        else:
            rng = np.random.default_rng([seed, 99])
            null = [evaluate_folds([{**fold, "y_train": rng.permutation(fold["y_train"]),
                                     "y_test": rng.permutation(fold["y_test"])} for fold in first],
                                   (kind,), (decoder,), classes, tuned=tuned, seed=seed)[0]
                    for _ in range(n_shuffles)]
            out[representation] = {"status": "computed", "observed": stack_scores(runs),
                                   "null": stack_scores(null) if null else None}
    return out


def fold_row_pools(labels: np.ndarray, fold_ids: np.ndarray, fold: int, classes: np.ndarray) -> tuple[dict, str | None]:
    """Training and test trial indices of every category in one outer fold, and the reason the session is
    dropped from the fold when some category has too few test or training trials."""
    pools = {"train": [np.flatnonzero((labels == c) & (fold_ids != fold)) for c in classes],
             "test": [np.flatnonzero((labels == c) & (fold_ids == fold)) for c in classes]}
    if min(len(p) for p in pools["test"]) < MIN_TEST_TRIALS_PER_CATEGORY:
        return pools, "too_few_test_trials"
    if min(len(p) for p in pools["train"]) < MIN_TRAIN_TRIALS_PER_CATEGORY:
        return pools, "too_few_training_trials"
    return pools, None


def fold_latent_series(counts: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray, fold: int,
                       representation: str, fits: dict, seed: int) -> dict:
    """State of every trial of one session, (trials, bins, dimensions), for one outer fold: the representation is
    fitted on the fold's training trials only and the test trials are projected. ``counts`` is
    (trials, bins, units); native counts are returned unchanged."""
    classes = np.unique(labels)
    _, reason = fold_row_pools(labels, fold_ids, fold, classes)
    if reason:
        return {"status": "fold_dropped", "reason": reason}
    if representation == "native":
        return {"status": "fitted", "series": counts}
    if counts.shape[2] < 3:
        return {"status": "not_estimable", "reason": "fewer_than_3_units"}
    train, test = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
    try:
        fit = fits[representation](counts[train], counts[test], labels[train], seed + fold)
    except Exception as error:
        return {"status": "not_estimable", "reason": f"fit_raised: {error}"[:200]}
    if fit.get("status") != "fitted":
        return {"status": "not_estimable", "reason": str(fit.get("reason") or fit.get("status"))[:200]}
    latent_train, latent_test = np.asarray(fit["latent_train"]), np.asarray(fit["latent_test"])
    if latent_train.shape[1] != counts.shape[1] or latent_test.shape[1] != counts.shape[1]:
        return {"status": "not_estimable", "reason": "latent_time_axis_differs_from_input"}
    series = np.empty((len(counts), counts.shape[1], latent_train.shape[2]))
    series[train], series[test] = latent_train, latent_test
    return {"status": "fitted", "series": series}


def representation_session_set(fold_records: dict[str, list[dict]]) -> tuple[list[str], dict[str, str]]:
    """Sessions whose fits succeeded in every fold they were not dropped from, and the reason for each other
    session. A session dropped from every fold has no usable fold."""
    kept, excluded = [], {}
    for key, records in fold_records.items():
        failed = [r["reason"] for r in records if r["status"] == "not_estimable"]
        if failed:
            excluded[key] = failed[0]
        elif all(r["status"] == "fold_dropped" for r in records):
            excluded[key] = "no_fold_meets_the_trial_count_rule"
        else:
            kept.append(key)
    return kept, excluded


def selected_dimension_columns(series: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray, fold: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Dimensions (or units) of one session that are category-selective on the fold's training trials,
    judged on their whole-maintenance mean."""
    train = fold_ids != fold
    whole = aggregate_bins(series[train], "maintenance_whole")[:, :, 0]
    return np.flatnonzero(category_selective_mask(whole, labels[train], rng))


def session_blocked_folds(blocks: list[dict], kind: str, classes: np.ndarray, n_folds: int, seed: int,
                          resample: int) -> list[dict]:
    """Pooled pseudo-trials in which one real trial per session supplies all of that session's dimensions.

    Each block holds ``key``, ``copy``, ``labels``, ``fold_ids`` and ``series`` (one (trials, bins, dimensions)
    array per outer fold, None for a dropped fold) and optionally ``columns`` (selected dimensions per fold).
    Within a fold every block draws, with replacement and category-matched, PSEUDO_TRAIN_PER_CATEGORY training
    and PSEUDO_TEST_PER_CATEGORY test trials from its own trials. The draw depends only on the seed, the
    resample, the fold and the block's key and copy, so blocks holding different representations of the same
    sessions receive identical trial draws. Sessions are dropped from a fold by the same rule as for
    pseudo-populations."""
    key = KINDS[kind][0]
    n_windows = len(FEATURES[KINDS[kind][1]]["starts"])
    counts = {"train": PSEUDO_TRAIN_PER_CATEGORY, "test": PSEUDO_TEST_PER_CATEGORY}
    folds = []
    for f in range(n_folds):
        dropped = {"too_few_test_trials": 0, "too_few_training_trials": 0}
        parts = {"train": [], "test": []}
        for block in blocks:
            pools, reason = fold_row_pools(block["labels"], block["fold_ids"], f, classes)
            if reason:
                dropped[reason] += 1
                continue
            rng = np.random.default_rng([seed, resample, f, block["key"], block["copy"]])
            series = block["series"][f]
            if block.get("columns") is not None:
                series = series[:, :, block["columns"][f]]
            for part in ("train", "test"):
                rows = np.concatenate([rng.choice(p, size=counts[part], replace=True) for p in pools[part]])
                parts[part].append(aggregate_bins(series[rows], kind))
        fold = {"y_train": np.repeat(classes, counts["train"]), "y_test": np.repeat(classes, counts["test"]),
                "train": {}, "test": {}, "dropped_sessions": dropped, "n_sessions": len(parts["train"])}
        for part in ("train", "test"):
            fold[part][key] = (np.concatenate(parts[part], axis=1) if parts[part]
                               else np.zeros((len(fold["y_" + part]), 0, n_windows)))
        fold["n_units"] = fold["train"][key].shape[1]
        folds.append(fold)
    return folds


def patient_bootstrap_blocks(blocks: list[dict], rng: np.random.Generator) -> list[dict]:
    """Draw patients with replacement; a drawn patient brings all of its sessions, and a patient drawn
    more than once brings them again as further copies, each assembled independently."""
    patients = sorted({b["patient"] for b in blocks})
    seen, out = {}, []
    for index in rng.choice(len(patients), size=len(patients), replace=True):
        patient = patients[index]
        copy = seen.get(patient, 0)
        seen[patient] = copy + 1
        out += [{**b, "copy": copy} for b in blocks if b["patient"] == patient]
    return out


def _mean_dropped(folds: list[dict]) -> dict:
    return {r: float(np.mean([f["dropped_sessions"][r] for f in folds])) for r in folds[0]["dropped_sessions"]}


def pooled_observed(blocks: list[dict], kind: str, decoders, classes: np.ndarray, n_folds: int, n_resamples: int,
                    n_shuffles: int, seed: int) -> dict:
    """Scores of ``n_resamples`` trial draws without patient redraw, and of ``n_shuffles`` label permutations of
    the first draw's assembled pseudo-trials (training and test permuted separately) that reuse its fits and
    its hyperparameters."""
    runs, first, tuned, seconds, dimensions, dropped = [], None, None, [], [], []
    for r in range(n_resamples):
        started = time.perf_counter()
        folds = session_blocked_folds(blocks, kind, classes, n_folds, seed, r)
        scores, picks = evaluate_folds(folds, (kind,), decoders, classes, seed=seed + r)
        seconds.append(time.perf_counter() - started)
        runs.append(scores)
        dimensions.append(float(np.mean([f["n_units"] for f in folds])))
        dropped.append(_mean_dropped(folds))
        first, tuned = (folds, picks) if first is None else (first, tuned)
    rng = np.random.default_rng([seed, 99])
    started = time.perf_counter()
    null = [evaluate_folds([{**fold, "y_train": rng.permutation(fold["y_train"]),
                             "y_test": rng.permutation(fold["y_test"])} for fold in first],
                           (kind,), decoders, classes, tuned=tuned, seed=seed)[0] for _ in range(n_shuffles)]
    return {"scores": stack_scores(runs), "null": stack_scores(null) if null else None,
            "resample_seconds": seconds, "null_seconds": time.perf_counter() - started,
            "mean_dimensions": dimensions, "mean_dropped_sessions": dropped}


def pooled_bootstrap(blocks: list[dict], kind: str, decoders, classes: np.ndarray, n_folds: int, indices,
                     seed: int) -> dict:
    """Scores of patient-bootstrap resamples: patients drawn with replacement, then folds assembled."""
    runs, seconds = [], []
    for b in indices:
        started = time.perf_counter()
        drawn = patient_bootstrap_blocks(blocks, np.random.default_rng([seed, 5_000, b]))
        folds = session_blocked_folds(drawn, kind, classes, n_folds, seed, 10_000 + b)
        runs.append(evaluate_folds(folds, (kind,), decoders, classes, seed=seed + b)[0])
        seconds.append(time.perf_counter() - started)
    return {"scores": stack_scores(runs), "resample_seconds": seconds}


def paired_bootstrap_difference(observed_a: np.ndarray, observed_b: np.ndarray, bootstrap_a: np.ndarray,
                                bootstrap_b: np.ndarray) -> dict:
    """A minus B per window. The estimate is the difference of the means over trial draws without patient
    redraw; the interval is the 2.5 and 97.5 percentiles of the patient-bootstrap differences; p is twice the
    smaller share of those differences at or beyond zero."""
    spread = bootstrap_a - bootstrap_b
    tails = np.minimum((spread <= 0).mean(axis=0), (spread >= 0).mean(axis=0))
    return {"estimate": (observed_a.mean(axis=0) - observed_b.mean(axis=0)).tolist(),
            "interval_95": np.percentile(spread, [2.5, 97.5], axis=0).tolist(),
            "p": np.minimum(1.0, 2 * tails).tolist(), "n_bootstrap": int(len(spread))}


RIDGE_MARGINALS = ("memorandum", "time")


def marginal_target(means: np.ndarray, marginal: str) -> np.ndarray:
    """Target of one marginal from the (categories, bins, units) training-trial means: the memorandum marginal
    removes the mean over categories at every bin; the time marginal is that mean minus its grand mean and is
    the same for every category."""
    over_categories = means.mean(axis=0, keepdims=True)
    if marginal == "memorandum":
        return means - over_categories
    if marginal == "time":
        return np.broadcast_to(over_categories - means.mean(axis=(0, 1)), means.shape)
    raise ValueError(f"marginal must be one of {RIDGE_MARGINALS}")


def ridge_marginal_axes(Z: np.ndarray, category: np.ndarray, lam: float, d: int, marginal: str) -> dict:
    """Reduced-rank ridge regression of one marginal target on the activity, (trials, bins, units) centred on its
    mean over trials and bins: B = (Sxx + lam * scale * I)^-1 Sxy with Sxx and Sxy taken over every (trial, bin)
    row. The encoder is the top-d eigenvectors of B' Sxx B (how each component appears in the units); the decoder
    is B times the encoder and computes the components, (activity - mean) @ decoder. Also returns the second
    moment of the target and the marginal."""
    n_trials, n_bins, k = Z.shape
    d = min(d, k)
    categories = np.unique(category)
    if len(categories) < 2:
        empty = np.zeros((k, d))
        return {"encoder": empty, "decoder": empty, "target_second_moment": np.zeros((k, k)), "marginal": marginal}
    means = np.stack([Z[category == c].mean(axis=0) for c in categories])
    target = marginal_target(means, marginal)
    Y = target[np.searchsorted(categories, category)].reshape(n_trials * n_bins, k)
    X = (Z - Z.mean(axis=(0, 1))).reshape(n_trials * n_bins, k)
    Sxx, Sxy = X.T @ X, X.T @ Y
    penalty = lam * np.trace(Sxx) / k * np.eye(k)
    try:
        B = np.linalg.solve(Sxx + penalty, Sxy)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(Sxx + penalty) @ Sxy
    fitted = B.T @ Sxx @ B
    values, vectors = np.linalg.eigh((fitted + fitted.T) / 2.0)
    encoder = vectors[:, np.argsort(values)[::-1][:d]]
    flat = target.reshape(-1, k)
    return {"encoder": encoder, "decoder": B @ encoder, "target_second_moment": flat.T @ flat, "marginal": marginal}


def memorandum_ridge_axes(Z: np.ndarray, category: np.ndarray, lam: float, d: int) -> dict:
    return ridge_marginal_axes(Z, category, lam, d, "memorandum")


def marginal_reconstruction_error(fit: dict, Z: np.ndarray, labels: np.ndarray, mean: np.ndarray) -> float:
    """Squared error of the marginal target reconstructed from the activity through the decoder and encoder,
    (Z - mean) @ decoder @ encoder', divided by the target's squared norm. The target is the marginal of Z's own
    category means; NaN when Z holds fewer than two categories or the target is zero."""
    categories = np.unique(labels)
    if len(categories) < 2:
        return float("nan")
    means = np.stack([Z[labels == c].mean(axis=0) for c in categories])
    target = marginal_target(means, fit["marginal"])[np.searchsorted(categories, labels)]
    total = float(np.sum(target ** 2))
    if total <= 0.0:
        return float("nan")
    return float(np.sum((target - ((Z - mean) @ fit["decoder"]) @ fit["encoder"].T) ** 2) / total)


def held_out_reconstruction_error(Z_train: np.ndarray, labels_train: np.ndarray, Z_test: np.ndarray,
                                  labels_test: np.ndarray, lam: float, d: int, fit_fn=memorandum_ridge_axes) -> float:
    """Marginal reconstruction error of held-out trials under axes fitted to the training trials, around the
    training mean."""
    if len(np.unique(labels_train)) < 2:
        return float("inf")
    fit = fit_fn(Z_train, labels_train, lam, d)
    return marginal_reconstruction_error(fit, Z_test, labels_test, Z_train.mean(axis=(0, 1)))


def mean_finite_score(values: list) -> float:
    kept = [v for v in values if not np.isnan(v)]
    return float(np.mean(kept)) if kept else float("inf")


def select_ridge_penalty(Z: np.ndarray, labels: np.ndarray, grid: tuple, n_folds: int, rng: np.random.Generator,
                         d: int, fit_fn=memorandum_ridge_axes) -> tuple[float, dict]:
    """Penalty with the smallest mean held-out marginal reconstruction error over stratified folds of the given
    trials."""
    counts = np.unique(labels, return_counts=True)[1]
    splitter = StratifiedKFold(n_splits=max(2, min(n_folds, int(counts.min()))), shuffle=True,
                               random_state=int(rng.integers(0, 1_000_000)))
    scores = {lam: [] for lam in grid}
    for train, test in splitter.split(np.zeros(len(Z)), labels):
        for lam in grid:
            scores[lam].append(held_out_reconstruction_error(Z[train], labels[train], Z[test], labels[test], lam, d,
                                                             fit_fn))
    mean_scores = {lam: mean_finite_score(v) for lam, v in scores.items()}
    return min(mean_scores, key=mean_scores.get), mean_scores


def fit_demixed_axes(Z: np.ndarray, labels: np.ndarray, rng: np.random.Generator, d: int,
                     grid: tuple = DPCA_RIDGE_LAMBDA_GRID, fit_fn=memorandum_ridge_axes) -> dict:
    """Demixed principal component axes of the label marginal (the label effect with the label-independent time
    course removed) of (trials, bins, units) activity, with the ridge penalty chosen from ``grid`` by held-out
    marginal reconstruction inside the given trials. Two-dimensional (trials, units) activity is one bin. The
    components are (activity - mean) @ decoder, with the mean over trials and bins; the encoder is recorded
    beside it. ``variance_captured`` is the share of the marginal target reconstructed on the given trials."""
    if Z.ndim == 2:
        Z = Z[:, None, :]
    penalty, scores = select_ridge_penalty(Z, labels, grid, N_CV_FOLDS_LAMBDA, rng, d, fit_fn)
    fit = fit_fn(Z, labels, penalty, d)
    error = marginal_reconstruction_error(fit, Z, labels, Z.mean(axis=(0, 1)))
    return {"lambda": penalty, "encoder": fit["encoder"], "decoder": fit["decoder"], "cats": np.unique(labels).tolist(),
            "variance_captured": 1.0 - error, "cv_score_table": {str(k): v for k, v in scores.items()}}


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of the squared cosines of the principal angles between two column spaces divided by the smaller
    dimension: 1 for the same subspace, 0 for orthogonal ones, about dimension / units for random ones."""
    qa, qb = np.linalg.qr(a)[0], np.linalg.qr(b)[0]
    return float(np.sum((qa.T @ qb) ** 2) / min(qa.shape[1], qb.shape[1]))


def encoding_fit_bins() -> np.ndarray:
    """Indices of the 100 ms encoding bins inside the selection window (200-1200 ms after picture onset)."""
    spec = FEATURES["encoding_100ms"]
    return np.flatnonzero((spec["starts"] >= SELECTION_WINDOW_S[0] - 1e-9)
                          & (spec["starts"] + spec["width"] <= SELECTION_WINDOW_S[1] + 1e-9))


def fit_projection_axes(fit_counts: np.ndarray, labels: np.ndarray, fit_axes, marginal: str, d: int, seed: int) -> dict:
    """Axes fitted to anscombe-transformed (trials, bins, units) counts. ``fit_axes(Z, labels, rng, d, marginal)``
    selects the ridge penalty by cross-validation and returns ``{"decoder", "encoder", "lambda"}``. ``axes`` are
    the decoder axes that compute the components; ``encoder_axes`` are recorded beside them."""
    transformed = anscombe_counts(fit_counts)
    fit = fit_axes(transformed, labels, np.random.default_rng(seed), d, marginal)
    return {"axes": np.asarray(fit["decoder"]), "encoder_axes": np.asarray(fit["encoder"]), "mean": transformed.reshape(-1, transformed.shape[-1]).mean(axis=0),
            "ridge_penalty": float(fit["lambda"])}


def project_counts(counts: np.ndarray, axes: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (anscombe_counts(counts) - mean) @ axes


def fold_axes_series(counts: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray, fold: int, fit_axes,
                     marginal: str, d: int, seed: int, fit_counts: np.ndarray | None = None) -> dict:
    """State of every trial of one session for one outer fold, (trials, bins, dimensions): axes are fitted on the
    fold's training trials of ``fit_counts`` (``counts`` when absent) and ``counts`` of all trials is projected.
    With ``fit_counts`` the projection of those counts is returned as ``encoding_series``."""
    _, reason = fold_row_pools(labels, fold_ids, fold, np.unique(labels))
    if reason:
        return {"status": "fold_dropped", "reason": reason}
    train = np.flatnonzero(fold_ids != fold)
    source = counts if fit_counts is None else fit_counts
    try:
        fit = fit_projection_axes(source[train], labels[train], fit_axes, marginal, d, seed + fold)
        out = {"status": "fitted", "series": project_counts(counts, fit["axes"], fit["mean"]), **fit}
        if fit_counts is not None:
            out["encoding_series"] = project_counts(fit_counts, fit["axes"], fit["mean"])
    except Exception as error:
        return {"status": "not_estimable", "reason": f"fit_raised: {error}"[:200]}
    if not all(np.isfinite(v).all() for k, v in out.items() if k.endswith("series")):
        return {"status": "not_estimable", "reason": "non-finite latent values"}
    return out


SPAN_FIELDS = ("axes", "encoder_axes")


def permuted_label_overlaps(encoding_train: np.ndarray, maintenance_train: np.ndarray, labels: np.ndarray, fit_axes,
                            d: int, n_permutations: int, seed: int) -> dict[str, np.ndarray]:
    """Overlap between memorandum axes fitted to the encoding window and to the maintenance window of the same
    training trials after permuting their labels, once per permutation, for decoder spans (``axes``) and encoder
    spans (``encoder_axes``)."""
    rng = np.random.default_rng(seed)
    out = {field: np.empty(n_permutations) for field in SPAN_FIELDS}
    for p in range(n_permutations):
        permuted = rng.permutation(labels)
        fits = [fit_projection_axes(counts, permuted, fit_axes, "memorandum", d, int(rng.integers(2**31)))
                for counts in (encoding_train, maintenance_train)]
        for field in SPAN_FIELDS:
            out[field][p] = subspace_overlap(*(fit[field] for fit in fits))
    return out


def session_blocked_transfer_folds(blocks: list[dict], kind: str, classes: np.ndarray, n_folds: int, seed: int,
                                   resample: int) -> list[dict]:
    """Pooled pseudo-trials for a decoder trained on the encoding-window projection and tested on each window of
    a maintenance time feature. Blocks hold ``series`` (maintenance projection) and ``encoding_series`` per outer
    fold. The two assemblies share every trial draw. Test values are multiplied by the window width so that the
    decoder's division by exposure leaves the projection unscaled, as it does for the training side."""
    train_key, test_key = KINDS[kind]
    width = FEATURES[test_key]["width"]
    encoding_blocks = [{**b, "series": b["encoding_series"]} for b in blocks]
    train_folds = session_blocked_folds(encoding_blocks, "maintenance_whole", classes, n_folds, seed, resample)
    test_folds = session_blocked_folds(blocks, test_key, classes, n_folds, seed, resample)
    folds = []
    for train_fold, test_fold in zip(train_folds, test_folds):
        folds.append({**test_fold, "train": {train_key: train_fold["train"]["maintenance_whole"]},
                      "test": {test_key: test_fold["test"][test_key] * width},
                      "n_units": train_fold["train"]["maintenance_whole"].shape[1]})
    return folds


def pooled_observed_assembled(blocks: list[dict], kind: str, decoders, classes: np.ndarray, n_folds: int,
                              n_resamples: int, n_shuffles: int, seed: int, assemble=session_blocked_folds) -> dict:
    """``pooled_observed`` with the pseudo-trial assembly supplied by the caller."""
    runs, first, tuned, seconds, dimensions, dropped = [], None, None, [], [], []
    for r in range(n_resamples):
        started = time.perf_counter()
        folds = assemble(blocks, kind, classes, n_folds, seed, r)
        scores, picks = evaluate_folds(folds, (kind,), decoders, classes, seed=seed + r)
        seconds.append(time.perf_counter() - started)
        runs.append(scores)
        dimensions.append(float(np.mean([f["n_units"] for f in folds])))
        dropped.append(_mean_dropped(folds))
        first, tuned = (folds, picks) if first is None else (first, tuned)
    rng = np.random.default_rng([seed, 99])
    started = time.perf_counter()
    null = [evaluate_folds([{**fold, "y_train": rng.permutation(fold["y_train"]),
                             "y_test": rng.permutation(fold["y_test"])} for fold in first],
                           (kind,), decoders, classes, tuned=tuned, seed=seed)[0] for _ in range(n_shuffles)]
    return {"scores": stack_scores(runs), "null": stack_scores(null) if null else None,
            "resample_seconds": seconds, "null_seconds": time.perf_counter() - started,
            "mean_dimensions": dimensions, "mean_dropped_sessions": dropped}


def pooled_bootstrap_assembled(blocks: list[dict], kind: str, decoders, classes: np.ndarray, n_folds: int, indices,
                               seed: int, assemble=session_blocked_folds) -> dict:
    """``pooled_bootstrap`` with the pseudo-trial assembly supplied by the caller."""
    runs, seconds = [], []
    for b in indices:
        started = time.perf_counter()
        drawn = patient_bootstrap_blocks(blocks, np.random.default_rng([seed, 5_000, b]))
        folds = assemble(drawn, kind, classes, n_folds, seed, 10_000 + b)
        runs.append(evaluate_folds(folds, (kind,), decoders, classes, seed=seed + b)[0])
        seconds.append(time.perf_counter() - started)
    return {"scores": stack_scores(runs), "resample_seconds": seconds}


def configure_window_geometry(maintenance_s: float, encoding_s: float) -> None:
    """Rebuild the maintenance and encoding window specifications in place for a corpus whose segments differ in
    length from the defaults; every function reads FEATURES and MAINTENANCE_S when it is called."""
    global MAINTENANCE_S, ENCODING_SEGMENT_S
    MAINTENANCE_S, ENCODING_SEGMENT_S = maintenance_s, encoding_s
    FEATURES["maintenance_whole"] = {"align": "maintenance", "starts": np.array([0.0]), "width": maintenance_s}
    for ms in (100, 250, 500):
        FEATURES[f"maintenance_{ms}ms"] = {"align": "maintenance", "starts": _window_starts(maintenance_s, ms / 1000),
                                           "width": ms / 1000}
    for ms in (100, 250):
        FEATURES[f"encoding_{ms}ms"] = {"align": "picture", "starts": _window_starts(encoding_s, ms / 1000),
                                        "width": ms / 1000}


RECOGNITION_RELEASE = "000004"
RECOGNITION_MAINTENANCE_S = 0.5
RECOGNITION_COLUMNS = ("stim_phase", "new_old_labels_recog", "response_value", "delay1_time", "stim_on_time",
                       "stim_off_time", "stimCategory", "start_time", "stop_time")


def select_recognition_trials(trials: dict, blank_s: float = RECOGNITION_MAINTENANCE_S) -> dict:
    """Trials of the new/old recognition phase that enter the memorandum analyses, with the count removed at each
    step. A trial needs a recognition-phase label, a truth label and a button response, finite picture-onset,
    picture-offset and question-screen times, and a blank between picture offset (stim_off_time) and the question
    screen (delay1_time) of at least ``blank_s``; it is correct when the response side matches the truth label.
    ``trials`` holds the NWB trial columns by name. ``blank_s`` of the result is the blank length of every trial."""
    steps = {}
    keep = np.asarray(trials["stim_phase"]) == "recog"
    steps["not_recognition_phase"] = int((~keep).sum())
    valid = np.isin(np.asarray(trials["new_old_labels_recog"]), ("0", "1")) & np.isin(
        np.asarray(trials["response_value"], dtype=float), DANDI_000004_RESPONSE_RANGE)
    steps["invalid_truth_label_or_response"] = int((keep & ~valid).sum())
    keep &= valid
    times = np.stack([np.asarray(trials[k], dtype=float) for k in ("stim_on_time", "stim_off_time", "delay1_time")])
    finite = np.isfinite(times).all(axis=0)
    steps["non_finite_times"] = int((keep & ~finite).sum())
    keep &= finite
    blank = times[2] - times[1]
    with np.errstate(invalid="ignore"):
        long_enough = blank >= blank_s
    steps["post_picture_blank_shorter_than_window"] = int((keep & ~long_enough).sum())
    correct_before_blank_rule = np.zeros(len(keep), dtype=bool)
    correct_before_blank_rule[keep] = recognition_correct(
        np.asarray(trials["new_old_labels_recog"])[keep], np.asarray(trials["response_value"], dtype=float)[keep])
    keep &= long_enough
    correct = keep & correct_before_blank_rule
    steps["error_trials"] = int((keep & ~correct).sum())
    return {"candidate": keep, "correct": correct, "removed_by_step": steps, "blank_s": blank,
            "correct_before_blank_rule": correct_before_blank_rule & np.isfinite(blank)}


def recognition_session_manifest() -> list[dict]:
    rows = []
    for path in sorted(dataset_path("dandi_000004").glob("sub-*/*.nwb")):
        rows.append({"release": RECOGNITION_RELEASE, "session": path.stem, "path": str(path),
                     "patient": path.parent.name, "status": "listed"})
    return rows


def load_recognition_session(path: str | Path, patient: str) -> tuple[dict | None, str]:
    """One recognition-corpus session as the dictionary ``build_entry`` reads: trials restricted to the
    recognition-phase candidates, picture onset = stim_on_time, maintenance onset = stim_off_time (the blank before
    the question screen), label = stimCategory, one condition (load 1). The task span for the rate floor runs from
    the first trial start to the last trial stop of the whole table (learning and recognition phases)."""
    with h5py.File(str(path), "r") as handle:
        if "units" not in handle or "intervals/trials" not in handle:
            return None, "missing_units_or_trials"
        table = handle["intervals/trials"]
        if not all(column in table for column in RECOGNITION_COLUMNS):
            return None, "missing_required_trial_columns"
        trials = {column: table[column][:] for column in RECOGNITION_COLUMNS}
        for column in ("stim_phase", "new_old_labels_recog"):
            trials[column] = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in trials[column]])
        spikes = [np.sort(times) for times in load_spike_times(handle)]
        regions = np.asarray(resolve_unit_regions(handle, "nwb_hemisphere_prefixed_structure")["region"])
    task_start, task_end = float(np.nanmin(trials["start_time"])), float(np.nanmax(trials["stop_time"]))
    if not task_end > task_start:
        return None, "empty_task_span"
    chosen = select_recognition_trials(trials)
    candidate = chosen["candidate"]
    task_rate_hz = np.array([(np.searchsorted(t, task_end) - np.searchsorted(t, task_start)) / (task_end - task_start)
                             for t in spikes])
    picture_s = np.asarray(trials["stim_off_time"] - trials["stim_on_time"], dtype=float)[candidate]
    kept = chosen["correct"][candidate]
    timing = {"n_trials": int(len(candidate)), "removed_by_step": chosen["removed_by_step"],
              "n_correct_recognition_trials": int(kept.sum()),
              "n_correct_trials_with_picture_at_least_encoding_segment": int((kept & (picture_s >= ENCODING_SEGMENT_S)).sum()),
              "blank_s_of_correct_trials_before_the_blank_rule": np.round(
                  chosen["blank_s"][chosen["correct_before_blank_rule"]], 4).tolist()}
    return {"patient": patient, "release": RECOGNITION_RELEASE, "session": Path(path).stem, "spikes": spikes,
            "regions": regions, "loads": np.ones(int(candidate.sum()), dtype=int), "correct": kept,
            "category": np.asarray(trials["stimCategory"])[candidate].astype(int),
            "t_picture": np.asarray(trials["stim_on_time"], dtype=float)[candidate],
            "t_maintenance": np.asarray(trials["stim_off_time"], dtype=float)[candidate],
            "picture_s": picture_s, "task_rate_hz": task_rate_hz, "timing": timing}, "loaded"
