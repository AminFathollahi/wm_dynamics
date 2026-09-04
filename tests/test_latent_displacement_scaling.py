"""Tests for the MSD/displacement-scaling primitives in
scripts/run_latent_displacement_scaling.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_latent_displacement_scaling import mean_squared_displacement, summarize_msd


def simulate_ou(n_trials, n_bins, lam, diffusion, rng):
    """Exact OU discretization, initialized from the stationary distribution
    (not a deterministic x0=0) so every bin -- including the first -- already
    has the process's steady-state variance; a deterministic start would
    otherwise impose an early-bin warm-up transient that biases MSD low."""
    stationary_var = diffusion / (2 * lam)
    latent = np.zeros((n_trials, n_bins, 1))
    latent[:, 0, 0] = rng.normal(0, np.sqrt(stationary_var), n_trials)
    dt = 1.0
    for t in range(1, n_bins):
        latent[:, t, 0] = latent[:, t - 1, 0] * np.exp(-lam * dt) + rng.normal(
            0, np.sqrt(diffusion * (1 - np.exp(-2 * lam * dt)) / (2 * lam)), n_trials,
        )
    return latent


def simulate_random_walk(n_trials, n_bins, step_sd, rng):
    steps = rng.normal(0, step_sd, size=(n_trials, n_bins - 1, 1))
    latent = np.concatenate([np.zeros((n_trials, 1, 1)), np.cumsum(steps, axis=1)], axis=1)
    return latent


def simulate_linear_drift(n_trials, n_bins, rate, noise_sd, rng):
    t = np.arange(n_bins)[None, :, None]
    latent = rate * t + rng.normal(0, noise_sd, size=(n_trials, n_bins, 1))
    return latent


class TestMSDSummaries:
    def test_confined_ou_gives_ratio_near_one_and_slope_near_zero(self, rng):
        latent = simulate_ou(n_trials=300, n_bins=31, lam=0.5, diffusion=1.0, rng=rng)
        summary = summarize_msd(mean_squared_displacement(latent))
        assert abs(summary["saturation_ratio"] - 1.0) < 0.3
        assert abs(summary["log_log_slope"] - 0.0) < 0.3

    def test_random_walk_gives_ratio_near_two_and_slope_near_one(self, rng):
        latent = simulate_random_walk(n_trials=300, n_bins=31, step_sd=1.0, rng=rng)
        summary = summarize_msd(mean_squared_displacement(latent))
        assert abs(summary["saturation_ratio"] - 2.0) < 0.3
        assert abs(summary["log_log_slope"] - 1.0) < 0.3

    def test_linear_drift_gives_slope_near_two(self, rng):
        latent = simulate_linear_drift(n_trials=300, n_bins=31, rate=1.0, noise_sd=0.05, rng=rng)
        summary = summarize_msd(mean_squared_displacement(latent))
        assert abs(summary["log_log_slope"] - 2.0) < 0.3

    def test_denominator_equals_attempted_on_well_formed_input(self, rng):
        latent = simulate_ou(n_trials=50, n_bins=20, lam=0.5, diffusion=1.0, rng=rng)
        msd = mean_squared_displacement(latent)
        assert len(msd) == latent.shape[1] - 1
        assert np.isfinite(msd).all()
