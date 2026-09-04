"""Scientific regression tests for the observability nugget-fraction estimator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observability import _leading_latent_projection, _leading_latent_projection_via_factor_analysis, nugget_fraction  # noqa: E402


def _simulate_ar1_plus_white(
    rng: np.random.Generator,
    n_trials: int,
    n_bins: int,
    rho: float,
    s2_slow: float,
    s2_nugget: float,
) -> np.ndarray:
    """(trials, 1, bins) single-channel AR(1)-plus-white-noise series.

    A single channel is used deliberately: z-scoring by one channel's own
    scale and PCA-fitting a leading direction from a single dimension are
    both scale/identity no-ops, so the recovered nugget fraction is a direct
    check of the autocovariance decomposition itself, independent of the
    multi-channel averaging benefit a real multi-unit population gets.
    """
    latent = np.empty((n_trials, n_bins))
    latent[:, 0] = rng.normal(0.0, np.sqrt(s2_slow), size=n_trials)
    innovation_var = s2_slow * (1.0 - rho**2)
    for t in range(1, n_bins):
        latent[:, t] = rho * latent[:, t - 1] + rng.normal(0.0, np.sqrt(innovation_var), size=n_trials)
    observed = latent + rng.normal(0.0, np.sqrt(s2_nugget), size=(n_trials, n_bins))
    return observed[:, None, :]


@pytest.mark.parametrize("target_nugget", [0.1, 0.5, 0.9])
def test_recovers_planted_nugget_fraction(target_nugget):
    # A long window (150 bins) keeps the per-trial temporal-mean-removal
    # step from eating into the slow component's own variance, which
    # otherwise biases the decomposition at short windows -- the estimator
    # is exercised at a scale where its assumptions hold, not confounded
    # with a short-window artifact this test isn't targeting.
    rng = np.random.default_rng(1234)
    s2_slow = 1.0
    s2_nugget = s2_slow * target_nugget / (1.0 - target_nugget)
    counts = _simulate_ar1_plus_white(rng, n_trials=3000, n_bins=150, rho=0.9,
                                       s2_slow=s2_slow, s2_nugget=s2_nugget)
    result = nugget_fraction(counts, n_splits=16, rng=np.random.default_rng(7))
    assert result["status"] == "fitted"
    assert result["median_nugget_fraction"] is not None
    assert abs(result["median_nugget_fraction"] - target_nugget) < 0.15


def test_pure_white_noise_returns_nugget_near_one():
    rng = np.random.default_rng(0)
    counts = rng.normal(0.0, 1.0, size=(40, 1, 23))
    result = nugget_fraction(counts, n_splits=40, rng=np.random.default_rng(100))
    assert result["status"] == "fitted"
    assert result["median_nugget_fraction"] is not None
    assert result["median_nugget_fraction"] > 0.75


def test_noiseless_ar1_returns_nugget_near_zero():
    rng = np.random.default_rng(9)
    counts = _simulate_ar1_plus_white(rng, n_trials=400, n_bins=25, rho=0.8,
                                       s2_slow=1.0, s2_nugget=0.0)
    result = nugget_fraction(counts, n_splits=12, rng=np.random.default_rng(13))
    assert result["median_nugget_fraction"] is not None
    assert result["median_nugget_fraction"] < 0.2


def test_cross_validated_exceeds_in_sample_under_basis_overfitting():
    """A weak shared signal spread across many more units than trials: the
    leading PCA direction fit and scored on the same trials chases that
    trial set's own noise, understating the nugget in-sample relative to
    scoring the fitted direction on held-out trials."""
    rng = np.random.default_rng(1002)
    n_trials, n_units, n_bins = 16, 30, 30
    rho, s2_slow = 0.9, 0.05
    latent = np.empty((n_trials, n_bins))
    latent[:, 0] = rng.normal(0.0, np.sqrt(s2_slow), size=n_trials)
    innovation_var = s2_slow * (1.0 - rho**2)
    for t in range(1, n_bins):
        latent[:, t] = rho * latent[:, t - 1] + rng.normal(0.0, np.sqrt(innovation_var), size=n_trials)
    loading = np.ones(n_units)
    shared = latent[:, None, :] * loading[None, :, None]
    noise = rng.normal(0.0, 1.0, size=(n_trials, n_units, n_bins))
    counts = shared + noise
    result = nugget_fraction(counts, n_splits=30, rng=np.random.default_rng(2002))
    assert result["status"] == "fitted"
    assert result["median_nugget_fraction"] is not None
    assert result["in_sample_nugget_fraction"] is not None
    assert result["in_sample_nugget_fraction"] < result["median_nugget_fraction"]


def test_too_few_bins_for_three_lags_reports_status_not_missing():
    rng = np.random.default_rng(0)
    counts = rng.normal(0.0, 1.0, size=(50, 4, 3))  # lag_cap = min(2, 8) = 2 < 3
    result = nugget_fraction(counts, n_splits=8, rng=rng)
    assert result["status"] == "fewer_than_three_positive_lags"
    assert result["median_nugget_fraction"] is None


class TestLeadingLatentProjectionViaFactorAnalysis:
    """The factor-analysis analogue of _leading_latent_projection must
    honour the same (psth_fit, psth_apply) -> (trials, bins) call contract,
    and must return None -- never raise -- on a fit too small or degenerate
    to support one factor."""

    def test_matches_pca_projection_shape(self):
        rng = np.random.default_rng(0)
        psth_fit = rng.normal(size=(30, 12, 8))
        psth_apply = rng.normal(size=(10, 12, 8))
        pca_latent = _leading_latent_projection(psth_fit, psth_apply)
        fa_latent = _leading_latent_projection_via_factor_analysis(psth_fit, psth_apply)
        assert fa_latent is not None
        assert fa_latent.shape == pca_latent.shape == (10, 8)
        assert np.all(np.isfinite(fa_latent))

    def test_single_unit_input_returns_none_not_an_exception(self):
        rng = np.random.default_rng(1)
        psth_fit = rng.normal(size=(20, 1, 6))
        psth_apply = rng.normal(size=(5, 1, 6))
        assert _leading_latent_projection_via_factor_analysis(psth_fit, psth_apply) is None

    def test_recovers_a_planted_shared_factor_better_than_pure_noise(self):
        """Not a numerical-agreement check against PCA (the two bases differ
        by construction) -- a sanity check that the factor-analysis arm
        recovers real shared structure at all, on a population with a
        strong shared signal and weak independent noise."""
        rng = np.random.default_rng(2)
        n_trials, n_units, n_bins = 60, 15, 10
        shared = rng.normal(size=(n_trials, n_bins))
        loadings = rng.normal(size=n_units)
        signal = shared[:, None, :] * loadings[None, :, None]
        noise = rng.normal(scale=0.1, size=(n_trials, n_units, n_bins))
        data = signal + noise
        fit_idx, apply_idx = np.arange(40), np.arange(40, 60)
        latent = _leading_latent_projection_via_factor_analysis(data[fit_idx], data[apply_idx])
        assert latent is not None
        correlations = [abs(np.corrcoef(latent[:, b], shared[apply_idx, b])[0, 1]) for b in range(n_bins)]
        assert np.mean(correlations) > 0.8
