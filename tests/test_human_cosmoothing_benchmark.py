from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / _subdir))

import run_human_cosmoothing_benchmark as mod  # noqa: E402
import run_info_benchmark  # noqa: E402


def test_neuron_holdout_is_a_quarter_and_deterministic():
    held_in, held_out = mod.neuron_holdout(12, seed=7)
    assert len(held_out) == 3
    assert len(held_in) == 9
    assert set(held_in.tolist()) | set(held_out.tolist()) == set(range(12))

    held_in2, held_out2 = mod.neuron_holdout(12, seed=7)
    np.testing.assert_array_equal(held_out, held_out2)
    np.testing.assert_array_equal(held_in, held_in2)

    _, tiny_held_out = mod.neuron_holdout(2, seed=1)
    assert len(tiny_held_out) == 1


def test_poisson_log_likelihood_matches_hand_computation():
    y = np.array([[[1.0, 0.0], [2.0, 1.0]], [[0.0, 3.0], [1.0, 2.0]]])
    rate = np.array([[[0.5, 0.2], [1.0, 0.9]], [[0.3, 2.5], [0.8, 1.5]]])
    expected = 0.0
    for trial in range(2):
        for time_bin in range(2):
            for neuron in range(2):
                yy = y[trial, time_bin, neuron]
                rr = rate[trial, time_bin, neuron]
                expected += yy * math.log(rr) - rr - math.lgamma(yy + 1.0)
    assert mod.poisson_log_likelihood(y, rate) == pytest.approx(expected)


def test_bits_per_spike_zero_when_predictions_equal_null():
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, size=(5, 4, 3)).astype(float)
    rate = np.broadcast_to(y.mean(axis=0), y.shape)
    ll_model = mod.poisson_log_likelihood(y, rate)
    ll_null = mod.poisson_log_likelihood(y, rate)
    assert mod.bits_per_spike(ll_model, ll_null, y.sum()) == 0.0


def test_bits_per_spike_positive_for_true_rate_with_low_dimensional_latent():
    rng = np.random.default_rng(0)
    n_trials, n_bins = 200, 4
    trial_gain = rng.uniform(0.3, 3.0, size=n_trials)
    bin_shape = np.array([1.0, 2.0, 1.5, 0.5])
    neuron_scale = np.array([1.0, 2.0, 0.5])
    true_rate = trial_gain[:, None, None] * bin_shape[None, :, None] * neuron_scale[None, None, :]
    y = rng.poisson(true_rate).astype(float)
    null_rate = np.broadcast_to(y.mean(axis=(0, 1), keepdims=True), y.shape)
    assert np.all(null_rate == null_rate[:, :1, :])
    ll_model = mod.poisson_log_likelihood(y, true_rate)
    ll_null = mod.poisson_log_likelihood(y, null_rate)
    assert mod.bits_per_spike(ll_model, ll_null, y.sum()) > 0.0


def test_held_out_neurons_never_reach_the_fit(monkeypatch):
    n_trials, n_bins, n_neurons = 12, 3, 8
    activity = np.zeros((n_trials, n_bins, n_neurons))
    for neuron in range(n_neurons):
        activity[:, :, neuron] = float(neuron)
    labels = np.array([0, 1] * (n_trials // 2))
    train_idx = np.arange(0, 8)
    test_idx = np.arange(8, 12)
    held_in, held_out = mod.neuron_holdout(n_neurons, seed=123)

    captured = {}

    def fake_fit_candidate(train_x, test_x, candidate, seed):
        captured["train_x"] = train_x
        captured["test_x"] = test_x
        return {"status": "fitted", "candidate": candidate, "k_used": train_x.shape[-1],
                "latent_train": train_x, "latent_test": test_x}

    monkeypatch.setattr(mod.benchmark_core, "fit_candidate", fake_fit_candidate)
    mod._evaluate_fold("principal_components", activity, labels, train_idx, test_idx, held_in, held_out,
                        "hippocampus", "s1", "load1_maintenance", 0)

    seen = set(np.unique(captured["train_x"]).tolist()) | set(np.unique(captured["test_x"]).tolist())
    assert seen == set(held_in.tolist())
    assert seen.isdisjoint(set(held_out.tolist()))


def test_trial_folds_equal_content_scores_folds(monkeypatch, tmp_path):
    captured = {}
    original = run_info_benchmark._seeded_folds

    def _spy(labels, n_splits, seed):
        result = original(labels, n_splits, seed)
        captured["folds"] = result
        return result

    monkeypatch.setattr(run_info_benchmark, "_seeded_folds", _spy)

    rng = np.random.default_rng(0)
    n_trials, n_bins, n_neurons = 40, 6, 10
    activity = rng.poisson(2.0, size=(n_trials, n_bins, n_neurons)).astype(float)
    labels = np.array([0, 1, 2, 3] * (n_trials // 4))

    run_info_benchmark.evaluate_block(
        "hippocampus", "s1", "load1_maintenance", activity, labels, (), ("linear",),
        5, 2, 1, lambda partial: None, {"probe": True}, 0, tmp_path, "schema_v1",
    )

    ours = mod._seeded_folds(
        labels, mod.N_SPLITS, mod.stable_seed("info_benchmark|seed0|hippocampus|s1|load1_maintenance|folds")
    )
    theirs = captured["folds"]
    assert len(ours) == len(theirs)
    for (train_a, test_a), (train_b, test_b) in zip(ours, theirs):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_penalty_selected_on_training_trials_only():
    rng = np.random.default_rng(0)
    n_train_trials, n_bins, k = 20, 3, 3
    groups = np.repeat(np.arange(n_train_trials), n_bins)
    x_train = rng.normal(size=(n_train_trials * n_bins, k))
    true_w = rng.normal(size=k)
    rate = np.exp(0.1 * x_train @ true_w)
    y_train = rng.poisson(rate).astype(float)

    x_test_small = rng.normal(size=(5, k))
    x_test_huge = rng.normal(size=(500, k)) * 1000.0

    _, alpha_small = mod.fit_readout(x_train, y_train, x_test_small, groups)
    _, alpha_huge = mod.fit_readout(x_train, y_train, x_test_huge, groups)
    assert alpha_small == alpha_huge
