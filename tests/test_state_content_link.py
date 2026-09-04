from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_state_content_link import (  # noqa: E402
    session_subtractive_test,
    session_trial_resolved_test,
    usable_label,
)

BIN_WIDTH_S = 0.1


def _planted_window_means(n_trials: int, n_units: int, labels: np.ndarray, signal_unit: int,
                           signal_amplitude: float, noise_sd: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=noise_sd, size=(n_trials, n_units))
    X[:, signal_unit] += signal_amplitude * labels
    return X[:, :, None]


def test_usable_label_rejects_missing_and_low_count_labels():
    assert usable_label(None) == (False, "no_label_field", None)

    labels = np.array([0, 0, 0, 1, 1, 1, 2])  # class 2 has only one trial
    ok, reason, mask = usable_label(labels, min_classes=2, min_per_class=3)
    assert ok
    assert mask.tolist() == [True, True, True, True, True, True, False]

    ok, reason, mask = usable_label(np.array([0, 0, 1]), min_classes=2, min_per_class=4)
    assert not ok
    assert "fewer_than_2_classes" in reason


def test_content_in_leading_latent_ranks_first_by_removal_cost():
    """Content signal lives on the unit with by far the largest variance, so
    PCA's leading component IS the content axis: removing it must cost the
    most of any single-latent removal, landing leading_latent_rank_from_top
    at 1 -- the content_in_the_state signature."""
    rng = np.random.default_rng(10)
    n_trials, n_units = 80, 8
    labels = np.array([i % 2 for i in range(n_trials)])
    rng.shuffle(labels)
    X = _planted_window_means(n_trials, n_units, labels, signal_unit=0,
                               signal_amplitude=6.0, noise_sd=1.0, seed=11)
    result = session_subtractive_test(X, labels, seed=1, k_max=5, n_half_splits=6)
    assert result["status"] == "tested"
    assert result["leading_latent_rank_from_top"] == 1
    assert result["a_full"] > 0.85
    assert result["leading_latent_cost"] > 0.15


def test_content_orthogonal_to_leading_latent_does_not_rank_first():
    """The leading (highest-variance) latent is unrelated noise; the content
    signal sits on a lower-variance unit. Removing the leading latent should
    barely touch decoding -- content_beside_the_state's signature."""
    rng = np.random.default_rng(20)
    n_trials, n_units = 80, 8
    labels = np.array([i % 2 for i in range(n_trials)])
    rng.shuffle(labels)
    X = rng.normal(scale=1.0, size=(n_trials, n_units))
    X[:, 0] += rng.normal(scale=8.0, size=n_trials)  # huge, label-independent variance
    X[:, 5] += 4.0 * labels  # content signal, low-variance unit
    X = X[:, :, None]
    result = session_subtractive_test(X, labels, seed=2, k_max=5, n_half_splits=6)
    assert result["status"] == "tested"
    assert result["leading_latent_rank_from_top"] > 1
    assert result["a_full"] > 0.85


def test_trial_resolved_margin_split_matches_trial_count_and_flags_too_few_trials():
    rng = np.random.default_rng(30)
    n_trials, n_units, n_bins = 30, 6, 10
    labels = np.array([i % 2 for i in range(n_trials)])
    rng.shuffle(labels)
    counts = rng.poisson(lam=1.0, size=(n_trials, n_units, n_bins)).astype(float)
    X = counts.mean(axis=2, keepdims=True)
    result = session_trial_resolved_test(counts, X, labels, k=3, clearing_lags=[3, 4],
                                          breakpoint_bins=6, seed=3)
    assert result["status"] == "tested"
    assert result["n_target_trials_per_half"] == n_trials // 2

    small_counts = counts[:10]
    small_X = X[:10]
    small_labels = labels[:10]
    too_few = session_trial_resolved_test(small_counts, small_X, small_labels, k=3,
                                           clearing_lags=[3, 4], breakpoint_bins=6, seed=4)
    assert too_few["status"] == "too_few_trials_to_split"
