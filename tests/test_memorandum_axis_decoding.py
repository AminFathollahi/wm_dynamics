import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import memorandum_decoding as md  # noqa: E402
from info_decoding import anscombe_counts  # noqa: E402

CLASSES = np.arange(1, 6)
KIND = "maintenance_whole"
N_FOLDS = 5


def fixed_penalty_fit(Z, labels, rng, d, marginal):
    fit = md.ridge_marginal_axes(Z, labels, 0.1, d, marginal)
    return {"decoder": fit["decoder"], "encoder": fit["encoder"], "lambda": 0.1}


def ramp_and_category_population(rng, per_class=60, n_units=12, n_bins=10, signal=2.5):
    """Counts with a large category-independent ramp along one direction and a small category effect in a
    four-dimensional subspace orthogonal to it."""
    ramp_direction = np.abs(rng.normal(size=n_units))
    ramp_direction /= np.linalg.norm(ramp_direction)
    free = rng.normal(size=(n_units, 4))
    free -= np.outer(ramp_direction, ramp_direction @ free)
    subspace = np.linalg.qr(free)[0]
    effects = np.linalg.qr(rng.normal(size=(len(CLASSES), 4)) - 0.0)[0] * np.sqrt(len(CLASSES))
    effects -= effects.mean(axis=0)
    labels = np.repeat(CLASSES, per_class)
    ramp = np.linspace(0, 1, n_bins)[None, :, None] * 12 * ramp_direction[None, None, :]
    rate = np.maximum(4.0 + ramp + signal * (effects @ subspace.T)[labels - 1][:, None, :], 0.3)
    return rng.poisson(rate).astype(float), labels, subspace, ramp_direction


def axis_decoding(counts, labels, fold_ids, marginal, d):
    predicted = np.empty_like(labels)
    for f in range(N_FOLDS):
        series = md.fold_axes_series(counts, labels, fold_ids, f, fixed_penalty_fit, marginal, d, 0)["series"]
        train, test = fold_ids != f, fold_ids == f
        scores = md._fit_scores("linear_svm", 0.1, series[train].mean(axis=1), labels[train], series[test].mean(axis=1),
                                1.0, 1.0, CLASSES)
        predicted[test] = CLASSES[scores.argmax(axis=1)]
    return md.balanced_accuracy(labels, predicted, CLASSES)


def test_memorandum_axes_recover_the_category_subspace_and_avoid_the_ramp():
    rng = np.random.default_rng(0)
    counts, labels, subspace, _ = ramp_and_category_population(rng)
    transformed = anscombe_counts(counts)
    ramp = transformed[:, -2:].mean(axis=(0, 1)) - transformed[:, :2].mean(axis=(0, 1))
    ramp = ramp[:, None] / np.linalg.norm(ramp)
    memorandum = md.ridge_marginal_axes(transformed, labels, 0.1, 4, "memorandum")["decoder"]
    assert md.subspace_overlap(memorandum, subspace) > 0.9
    assert md.subspace_overlap(memorandum, ramp) < 0.3
    fold_ids = md.stratified_fold_ids(labels, N_FOLDS, rng)
    assert axis_decoding(counts, labels, fold_ids, "memorandum", 4) > 0.6
    assert axis_decoding(counts, labels, fold_ids, "time", 1) < 0.3


def test_marginal_targets_average_to_zero_over_categories_and_the_time_target_is_category_free():
    means = np.random.default_rng(1).normal(size=(5, 7, 6)) + np.linspace(0, 3, 7)[None, :, None]
    memorandum = md.marginal_target(means, "memorandum")
    assert np.allclose(memorandum.mean(axis=0), 0)
    time_target = md.marginal_target(means, "time")
    assert np.allclose(time_target, time_target[:1]) and np.allclose(time_target.mean(axis=(0, 1)), 0)
    assert np.allclose(memorandum + time_target, means - means.mean(axis=(0, 1)))


def test_subspace_overlap_is_one_for_the_same_subspace_zero_for_orthogonal_and_dimension_over_units_for_random():
    rng = np.random.default_rng(2)
    a = np.linalg.qr(rng.normal(size=(40, 4)))[0]
    rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    assert abs(md.subspace_overlap(a, a @ rotation) - 1) < 1e-9
    other = rng.normal(size=(40, 4))
    other -= a @ (a.T @ other)
    assert md.subspace_overlap(a, other) < 1e-12
    randoms = [md.subspace_overlap(a, rng.normal(size=(40, 4))) for _ in range(400)]
    assert abs(np.mean(randoms) - 4 / 40) < 0.01


def test_shuffled_labels_give_chance_from_memorandum_axes_and_assembled_scoring_matches_the_pooled_scoring():
    rng = np.random.default_rng(3)
    sessions = []
    for k in range(4):
        counts, labels, _, _ = ramp_and_category_population(rng, per_class=15, n_units=10, n_bins=4, signal=3.0)
        sessions.append({"key": k, "patient": f"p{k}", "counts": counts, "labels": labels,
                         "fold_ids": md.stratified_fold_ids(labels, N_FOLDS, rng)})

    def blocks(shuffle):
        out = []
        for s in sessions:
            labels = rng.permutation(s["labels"]) if shuffle else s["labels"]
            fold_ids = md.stratified_fold_ids(labels, N_FOLDS, rng) if shuffle else s["fold_ids"]
            series = [md.fold_axes_series(s["counts"], labels, fold_ids, f, fixed_penalty_fit, "memorandum", 3, 0)
                      ["series"] for f in range(N_FOLDS)]
            out.append({"key": s["key"], "copy": 0, "patient": s["patient"], "labels": labels,
                        "fold_ids": fold_ids, "series": series})
        return out

    score = lambda b: md.pooled_observed(b, KIND, ("linear_svm",), CLASSES, N_FOLDS, 2, 0, 5)["scores"]["linear_svm"][KIND]
    assert score(blocks(False))["balanced_accuracy"].mean() > 0.5
    assert abs(score(blocks(True))["balanced_accuracy"].mean() - 0.2) < 0.06
    real = blocks(False)
    again = md.pooled_observed_assembled(real, KIND, ("linear_svm",), CLASSES, N_FOLDS, 2, 0, 5)["scores"]["linear_svm"][KIND]
    assert np.array_equal(again["balanced_accuracy"], score(real)["balanced_accuracy"])


def noisy_population(seed, per_class=20, n_units=40, n_bins=3, noise_directions=4, noise_sd=3.0, signal=1.5, n_classes=4):
    """Large trial-level noise in a few directions, a category signal in four other directions, and a category-free
    time course in two more; returns the activity, labels and the planted category subspace."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(n_units, n_units)))[0]
    noise, subspace = basis[:, :noise_directions], basis[:, noise_directions:noise_directions + 4]
    time_directions = basis[:, noise_directions + 4:noise_directions + 4 + n_bins - 1]
    effects = rng.normal(size=(n_classes, n_bins, 4)) * signal
    effects -= effects.mean(axis=0)
    labels = np.repeat(np.arange(n_classes), per_class)
    course = (np.linalg.qr(rng.normal(size=(n_bins, n_bins)))[0][:, :n_bins - 1] * 3.0) @ time_directions.T
    activity = (course + np.einsum("ibs,ks->ibk", effects[labels], subspace)
                + (rng.normal(size=(len(labels), 1, noise_directions)) * noise_sd) @ noise.T
                + rng.normal(size=(len(labels), n_bins, n_units)))
    return activity, labels, subspace


def test_decoder_projection_recovers_the_planted_category_subspace_better_than_the_encoder_and_time_axes_decode_at_chance():
    from sklearn.linear_model import LogisticRegression

    def accuracy(train, test, y_train, y_test):
        model = LogisticRegression(max_iter=3000).fit(train.reshape(len(train), -1), y_train)
        return model.score(test.reshape(len(test), -1), y_test)

    overlap, score, time_score = {"encoder": [], "decoder": []}, {"encoder": [], "decoder": []}, []
    for seed in range(12):
        activity, labels, subspace = noisy_population(seed)
        order = np.random.default_rng(seed).permutation(len(labels))
        train, test = order[:len(order) // 2], order[len(order) // 2:]
        centre = activity[train].mean(axis=(0, 1))
        fit = md.fit_demixed_axes(activity[train], labels[train], np.random.default_rng(seed), 4)
        for name in overlap:
            overlap[name].append(md.subspace_overlap(fit[name], subspace))
            score[name].append(accuracy((activity[train] - centre) @ fit[name], (activity[test] - centre) @ fit[name],
                                        labels[train], labels[test]))
        time_fit = md.fit_demixed_axes(activity[train], labels[train], np.random.default_rng(seed), 2,
                                       fit_fn=lambda z, c, lam, k: md.ridge_marginal_axes(z, c, lam, k, "time"))
        time_score.append(accuracy((activity[train] - centre) @ time_fit["decoder"],
                                   (activity[test] - centre) @ time_fit["decoder"], labels[train], labels[test]))
    assert np.mean(overlap["decoder"]) > np.mean(overlap["encoder"]) + 0.02
    assert np.mean(score["decoder"]) >= np.mean(score["encoder"])
    assert np.mean(time_score) < 0.35


def test_penalty_selection_reads_only_the_trials_it_is_given():
    activity, labels, _ = noisy_population(0, per_class=12, n_units=10)
    activity[:, 0, 0] = np.arange(len(labels))
    given = np.arange(0, len(labels), 2)
    seen = []

    def spy_fit(Z, category, lam, d):
        seen.append(Z[:, 0, 0])
        return md.memorandum_ridge_axes(Z, category, lam, d)

    md.select_ridge_penalty(activity[given], labels[given], (0.0, 0.1, 1.0), 3, np.random.default_rng(0), 2, spy_fit)
    assert seen and set(np.concatenate(seen)) <= set(given.astype(float))
    held_out = md.held_out_reconstruction_error(activity[given], labels[given], activity[1::2], labels[1::2], 0.1, 2)
    assert np.isfinite(held_out) and md.held_out_reconstruction_error(
        activity[given], np.zeros(len(given)), activity[1::2], labels[1::2], 0.1, 2) == float("inf")
