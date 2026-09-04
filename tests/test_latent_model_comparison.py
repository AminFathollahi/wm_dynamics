"""Scientific regression tests for the PCA-vs-GPFA latent model comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from observability import median_nugget_from_held_out_latent  # noqa: E402
from run_latent_model_comparison import fit_gpfa_arm, fit_pca_arm  # noqa: E402

BIN_MS = 100.0


def _simulate_counts(rng: np.random.Generator, n_trials: int, n_units: int, n_bins: int,
                      x_dim: int, rho: float, noise_sd: float, mean_level: float = 8.0) -> np.ndarray:
    """Counts generated in the Anscombe (variance-stabilized) domain, matching
    the internal representation both GPFA and this project's PCA arm actually
    fit: x_dim independent AR(1) latents loaded onto n_units with a fixed
    random loading, plus independent per-unit-per-bin Gaussian noise, inverted
    back to non-negative integer counts."""
    latents = np.empty((x_dim, n_trials, n_bins))
    for d in range(x_dim):
        latents[d, :, 0] = rng.normal(0.0, 1.0, size=n_trials)
        innovation = np.sqrt(max(1.0 - rho**2, 1e-6))
        for t in range(1, n_bins):
            latents[d, :, t] = rho * latents[d, :, t - 1] + rng.normal(0.0, innovation, size=n_trials)
    loading = rng.normal(0.0, 1.0, size=(n_units, x_dim))
    shared = np.einsum("uk,ktb->utb", loading, latents).transpose(1, 0, 2)
    per_unit_noise = rng.normal(0.0, noise_sd, size=(n_trials, n_units, n_bins))
    sqrt_counts = mean_level + shared + per_unit_noise
    counts = np.maximum((sqrt_counts / 2.0) ** 2 - 3.0 / 8.0, 0.0)
    return np.rint(counts).astype(int)


def _split(counts: np.ndarray, rng: np.random.Generator, train_fraction: float = 0.6):
    n_trials = counts.shape[0]
    perm = rng.permutation(n_trials)
    n_train = max(2, int(round(n_trials * train_fraction)))
    return counts[perm[:n_train]], counts[perm[n_train:]]


def test_gpfa_recovers_lower_nugget_than_pca_under_large_noise():
    rng = np.random.default_rng(2026)
    counts = _simulate_counts(rng, n_trials=100, n_units=20, n_bins=25, x_dim=2,
                               rho=0.9, noise_sd=8.0)
    train, test = _split(counts, np.random.default_rng(1))

    pca = fit_pca_arm(train, test, k=2)
    gpfa = fit_gpfa_arm(train, test, BIN_MS, k=2)

    pca_nugget = median_nugget_from_held_out_latent(
        pca["latent"][:, :, 0], bin_width_s=BIN_MS / 1000.0, n_splits=16, rng=np.random.default_rng(3))
    gpfa_nugget = median_nugget_from_held_out_latent(
        gpfa["latent"][:, :, 0], bin_width_s=BIN_MS / 1000.0, n_splits=16, rng=np.random.default_rng(4))

    assert pca_nugget["status"] == "fitted"
    assert gpfa_nugget["status"] == "fitted"
    assert gpfa_nugget["median_nugget_fraction"] < pca_nugget["median_nugget_fraction"]


def test_both_methods_recover_planted_dimensionality_order_of_magnitude():
    from geometry import twonn_dimension

    rng = np.random.default_rng(11)
    counts = _simulate_counts(rng, n_trials=100, n_units=25, n_bins=25, x_dim=2,
                               rho=0.9, noise_sd=1.0)
    train, test = _split(counts, np.random.default_rng(5))

    pca = fit_pca_arm(train, test, k=4)
    gpfa = fit_gpfa_arm(train, test, BIN_MS, k=4)

    pca_dim = twonn_dimension(pca["latent"].reshape(-1, pca["latent"].shape[-1]))
    gpfa_dim = twonn_dimension(gpfa["latent"].reshape(-1, gpfa["latent"].shape[-1]))

    # A planted 2-dimensional trajectory should not read back as high-single-digit
    # or double-digit ambient dimensionality under either method.
    assert 0.5 <= pca_dim <= 5.0
    assert 0.5 <= gpfa_dim <= 5.0


def test_methods_agree_on_noiseless_data():
    rng = np.random.default_rng(2026)
    counts = _simulate_counts(rng, n_trials=100, n_units=20, n_bins=25, x_dim=2,
                               rho=0.9, noise_sd=0.01)
    train, test = _split(counts, np.random.default_rng(1))

    pca = fit_pca_arm(train, test, k=2)
    gpfa = fit_gpfa_arm(train, test, BIN_MS, k=2)

    pca_nugget = median_nugget_from_held_out_latent(
        pca["latent"][:, :, 0], bin_width_s=BIN_MS / 1000.0, n_splits=16, rng=np.random.default_rng(3))
    gpfa_nugget = median_nugget_from_held_out_latent(
        gpfa["latent"][:, :, 0], bin_width_s=BIN_MS / 1000.0, n_splits=16, rng=np.random.default_rng(4))

    assert pca_nugget["status"] == "fitted"
    assert gpfa_nugget["status"] == "fitted"
    assert abs(pca_nugget["median_nugget_fraction"] - gpfa_nugget["median_nugget_fraction"]) < 0.15
