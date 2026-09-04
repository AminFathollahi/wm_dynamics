"""Lightweight synthetic-input regression test for the Miller drift-spine fit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_dynamics import simulate_confined_diffusion  # noqa: E402
from run_miller_drift_spine import bin_time_axis, fit_condition_drift  # noqa: E402

_VALID_STATUSES = {"identifiable", "not_identifiable", "unconfined", "nonconverged", "not_estimable"}


def _synthetic_condition_epochs(rng: np.random.Generator):
    """Three-condition synthetic epochs with a confined-drift signal on channel 0."""
    n_per_condition = 30
    n_time = 20
    n_channels = 6
    dt = 0.1
    labels = np.repeat([0, 1, 2], n_per_condition)
    n_trials = len(labels)
    condition_offset = {0: -1.0, 1: 0.0, 2: 1.0}

    epochs = rng.normal(scale=0.5, size=(n_trials, n_channels, n_time))
    for label, offset in condition_offset.items():
        rows = np.flatnonzero(labels == label)
        _, observations = simulate_confined_diffusion(
            len(rows), n_time, dt, lambda_rate=1.5, diffusion=0.4,
            equilibrium=offset, observation_sd=0.15, rng=rng,
        )
        epochs[rows, 0, :] = observations
    return epochs, labels, dt


def test_fit_condition_drift_runs_end_to_end_on_synthetic_data():
    rng = np.random.default_rng(0)
    epochs, labels, dt = _synthetic_condition_epochs(rng)
    axis_bins = np.zeros(epochs.shape[-1], dtype=bool)
    axis_bins[epochs.shape[-1] // 4:] = True

    result = fit_condition_drift(epochs, labels, axis_bins, dt, seed=1, n_splits=3, n_components=4)

    assert result["n_trials"] == epochs.shape[0]
    assert len(result["folds"]) == 3
    for row in result["folds"]:
        assert row["state_space"]["status"] in _VALID_STATUSES
        assert row["moment"]["status"] in _VALID_STATUSES

    summary = result["summary"]
    assert summary["moment_lambda_identified_mean"] is None or summary["moment_lambda_identified_mean"] > 0

    descriptive = result["descriptive_in_sample_bootstrap"]["estimate"]
    assert descriptive["status"] in _VALID_STATUSES


def test_bin_time_axis_block_averages_without_leftover_bleed():
    epochs = np.arange(2 * 1 * 10, dtype=float).reshape(2, 1, 10)
    times = np.linspace(-0.1, 0.8, 10)
    binned, bin_times = bin_time_axis(epochs, times, bin_ms=200, srate=10.0)
    assert binned.shape == (2, 1, 5)
    assert bin_times.shape == (5,)
    np.testing.assert_allclose(binned[0, 0], epochs[0, 0].reshape(5, 2).mean(axis=1))
