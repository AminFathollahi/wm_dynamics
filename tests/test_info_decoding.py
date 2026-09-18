import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import src.info_decoding as decoding
from src.info_decoding import (
    aggregate_seed_records,
    matched_method_contrast,
    make_decoder,
    make_folds,
    permutation_null,
    same_time_auc,
    split_auc,
    temporal_auc,
)


def test_seed_aggregation_averages_seeds_within_session_cells():
    row = lambda seed, candidate, value: {"status": "computed", "seed": seed, "corpus": "c", "session": "s", "level": "2", "candidate": candidate, "decoder": "linear", "same_time": {"auc": value, "null_auc": .5}, "cross_temporal": {"auc": value, "null_auc": .5}}
    cells = aggregate_seed_records([row(1, "native_full_rank", .6), row(2, "native_full_rank", .8)])
    assert cells[0]["n_seeds"] == 2
    assert cells[0]["cross_temporal"]["auc"] == .7


def test_method_contrast_bootstraps_session_cells_not_permutations():
    def row(session, candidate, value):
        return {"status": "computed", "corpus": "c", "session": session, "level": "all", "candidate": candidate, "decoder": "linear", "cross_temporal": {"auc": value, "null_auc": .5}, "same_time": {"auc": value, "null_auc": .5}}
    result = matched_method_contrast([row("s1", "native_full_rank", .5), row("s1", "pca", .7), row("s2", "native_full_rank", .5), row("s2", "pca", .9)], "pca", n_bootstrap=100, seed=3)
    assert result["n_cells"] == 2
    assert result["independent_unit"] == "recording_session"
    assert result["estimate"] == .3


def test_folds_are_deterministic_stratified_partitions():
    labels = np.repeat(np.arange(3), 12)

    first = make_folds(labels, n_splits=4, seed=8)
    second = make_folds(labels, n_splits=4, seed=8)

    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        np.testing.assert_array_equal(np.unique(labels[test_a]), np.arange(3))
    tests = np.concatenate([test for _, test in first])
    np.testing.assert_array_equal(np.sort(tests), np.arange(36))


def test_decoder_pipeline_scales_only_fit_data():
    train = np.array([[0.0, 1.0], [2.0, 5.0], [4.0, 9.0], [6.0, 13.0]])
    labels = np.array([0, 0, 1, 1])
    held_out = np.full((3, 2), 10_000.0)

    decoder = make_decoder("linear", seed=2).fit(train, labels)
    decoder.predict_proba(held_out)

    assert isinstance(decoder, Pipeline)
    np.testing.assert_allclose(decoder.named_steps["scale"].mean_, train.mean(axis=0))


def test_multiclass_same_time_matches_temporal_diagonal():
    rng = np.random.default_rng(10)
    labels = np.repeat(np.arange(3), 30)
    signal = np.eye(3)[labels]
    activity = np.stack(
        [signal + rng.normal(scale=0.25, size=signal.shape) for _ in range(3)], axis=1
    )
    folds = make_folds(labels, n_splits=3, seed=5)

    matrix = temporal_auc(activity, labels, folds=folds, seed=5)
    diagonal = same_time_auc(activity, labels, folds=folds, seed=5)

    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(diagonal, np.diag(matrix))
    assert diagonal.min() > 0.95


def test_split_auc_accepts_distinct_train_and_test_sets(monkeypatch):
    rng = np.random.default_rng(11)
    train_labels = np.repeat(np.arange(3), 12)
    test_labels = np.repeat(np.arange(3), 7)
    train_signal = np.eye(3)[train_labels]
    test_signal = np.eye(3)[test_labels]
    train = np.stack((train_signal, train_signal), axis=1)
    test = np.stack((test_signal, test_signal), axis=1)
    train += rng.normal(scale=0.1, size=train.shape)
    test += rng.normal(scale=0.1, size=test.shape)

    fit_means = []

    class TrackingScaler(StandardScaler):
        def fit(self, features, labels=None, sample_weight=None):
            result = super().fit(features, labels, sample_weight=sample_weight)
            fit_means.append(self.mean_.copy())
            return result

    monkeypatch.setattr(decoding, "StandardScaler", TrackingScaler)
    matrix = split_auc(train, train_labels, test, test_labels, seed=6)

    assert matrix.shape == (2, 2)
    assert matrix.min() > 0.95
    np.testing.assert_allclose(fit_means, train.mean(axis=0))


def test_nonlinear_decoder_recovers_xor():
    rng = np.random.default_rng(12)
    activity = rng.normal(size=(400, 1, 2))
    labels = (activity[:, 0, 0] * activity[:, 0, 1] > 0).astype(int)
    folds = make_folds(labels, n_splits=4, seed=3)

    linear = same_time_auc(activity, labels, decoder="linear", folds=folds, seed=3)[0]
    nonlinear = same_time_auc(
        activity, labels, decoder="nonlinear", folds=folds, seed=3
    )[0]

    assert linear < 0.65
    assert nonlinear > 0.9


def test_nonlinear_decoder_supports_four_classes():
    rng = np.random.default_rng(13)
    labels = np.repeat(np.arange(4), 20)
    centers = np.array(((-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)))
    activity = centers[labels, None, :] + rng.normal(scale=0.3, size=(80, 1, 2))

    score = same_time_auc(
        activity, labels, decoder="nonlinear", n_splits=4, seed=2
    )

    assert score[0] > 0.95


def test_permutation_null_is_deterministic_and_near_chance(monkeypatch):
    rng = np.random.default_rng(14)
    labels = np.repeat(np.arange(3), 24)
    activity = rng.normal(size=(labels.size, 2, 4))
    folds = make_folds(labels, n_splits=3, seed=7)
    monkeypatch.setattr(
        decoding,
        "make_folds",
        lambda *args, **kwargs: pytest.fail("fixed folds were not reused"),
    )

    first = permutation_null(
        activity,
        labels,
        folds=folds,
        n_permutations=12,
        seed=9,
        same_time=True,
    )
    second = permutation_null(
        activity,
        labels,
        folds=folds,
        n_permutations=12,
        seed=9,
        same_time=True,
    )

    assert first.shape == (12, 2)
    np.testing.assert_allclose(first, second)
    assert abs(first.mean() - 0.5) < 0.1


@pytest.mark.parametrize("kind", ["linear", "nonlinear"])
def test_decoders_are_seed_deterministic(kind):
    rng = np.random.default_rng(16)
    labels = np.repeat([0, 1], 30)
    activity = rng.normal(size=(60, 2, 3))

    first = temporal_auc(activity, labels, decoder=kind, n_splits=3, seed=4)
    second = temporal_auc(activity, labels, decoder=kind, n_splits=3, seed=4)

    np.testing.assert_allclose(first, second)
