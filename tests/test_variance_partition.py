"""Scientific regression tests for the four-way single-trial variance split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from variance_partition import partition_single_trial_variance  # noqa: E402

SHARE_KEYS = ("cond", "static", "slow", "white")


def _simulate(rng: np.random.Generator, n_trials: int, n_units: int, n_bins: int, cond_amp: float,
              static_sd: float, slow_sd: float, slow_rho: float, white_sd: float, unit_noise_sd: float) -> np.ndarray:
    """Counts built from a single shared latent direction with known,
    independently-controllable cond/static/slow/white shares: a fixed
    across-trial time course, a per-trial constant offset, a per-trial
    AR(1) excursion, and per-trial-per-bin white noise, loaded onto
    n_units with a fixed random loading plus independent per-unit noise."""
    bins = np.arange(n_bins)
    cond_curve = cond_amp * np.sin(2 * np.pi * (bins + 0.5) / n_bins)
    static = rng.normal(0.0, static_sd, size=n_trials)
    slow = np.empty((n_trials, n_bins))
    innovation = np.sqrt(max(1.0 - slow_rho**2, 1e-6))
    slow[:, 0] = rng.normal(0.0, slow_sd, size=n_trials)
    for t in range(1, n_bins):
        slow[:, t] = slow_rho * slow[:, t - 1] + rng.normal(0.0, slow_sd * innovation, size=n_trials)
    white = rng.normal(0.0, white_sd, size=(n_trials, n_bins))
    latent = cond_curve[None, :] + static[:, None] + slow + white
    loading = rng.normal(0.0, 1.0, size=n_units)
    per_unit_noise = rng.normal(0.0, unit_noise_sd, size=(n_trials, n_units, n_bins))
    return latent[:, None, :] * loading[None, :, None] + per_unit_noise


@pytest.mark.parametrize("cond_amp,static_sd,slow_sd,slow_rho,seed", [
    (2.0, 0.3, 0.3, 0.85, 100),   # cond-dominant
    (0.3, 2.0, 0.3, 0.85, 101),   # static-dominant
    (0.3, 0.3, 2.0, 0.60, 102),   # slow-dominant
])
def test_recovers_planted_shares_across_settings(cond_amp, static_sd, slow_sd, slow_rho, seed):
    rng = np.random.default_rng(seed)
    counts = _simulate(rng, n_trials=300, n_units=30, n_bins=24, cond_amp=cond_amp, static_sd=static_sd,
                        slow_sd=slow_sd, slow_rho=slow_rho, white_sd=0.3, unit_noise_sd=0.3)
    expected_var = np.mean((cond_amp * np.sin(2 * np.pi * (np.arange(24) + 0.5) / 24)) ** 2) + \
        static_sd**2 + slow_sd**2 + 0.3**2
    expected = {"cond": np.mean((cond_amp * np.sin(2 * np.pi * (np.arange(24) + 0.5) / 24)) ** 2) / expected_var,
                "static": static_sd**2 / expected_var, "slow": slow_sd**2 / expected_var, "white": 0.3**2 / expected_var}

    result = partition_single_trial_variance(counts, n_splits=12, rng=np.random.default_rng(seed + 1000), bin_width_s=0.1)
    assert result["status"] == "fitted"
    for key in SHARE_KEYS:
        assert result[f"{key}_fraction_median"] == pytest.approx(expected[key], abs=0.2)


def test_pure_white_noise_returns_white_near_one():
    rng = np.random.default_rng(700)
    counts = _simulate(rng, n_trials=300, n_units=30, n_bins=24, cond_amp=0.0, static_sd=0.0,
                        slow_sd=0.0, slow_rho=0.6, white_sd=1.0, unit_noise_sd=0.3)
    result = partition_single_trial_variance(counts, n_splits=12, rng=np.random.default_rng(701), bin_width_s=0.1)
    assert result["white_fraction_median"] > 0.85
    assert result["cond_fraction_median"] < 0.1
    assert result["static_fraction_median"] < 0.1
    assert result["slow_fraction_median"] < 0.1
    assert abs(result["held_position_correlation_median"]) < 0.2


def test_static_offset_and_ar1_components_are_separable():
    rng = np.random.default_rng(1100)
    static_only = _simulate(rng, n_trials=300, n_units=30, n_bins=24, cond_amp=0.3, static_sd=1.5,
                             slow_sd=0.0, slow_rho=0.6, white_sd=0.3, unit_noise_sd=0.3)
    rng = np.random.default_rng(1200)
    ar1_only = _simulate(rng, n_trials=300, n_units=30, n_bins=24, cond_amp=0.3, static_sd=0.0,
                          slow_sd=1.5, slow_rho=0.6, white_sd=0.3, unit_noise_sd=0.3)

    r_static = partition_single_trial_variance(static_only, n_splits=12, rng=np.random.default_rng(1101), bin_width_s=0.1)
    r_ar1 = partition_single_trial_variance(ar1_only, n_splits=12, rng=np.random.default_rng(1201), bin_width_s=0.1)

    assert r_static["static_fraction_median"] > 0.8
    assert r_static["slow_fraction_median"] < 0.1
    assert r_ar1["slow_fraction_median"] > 0.7
    assert r_ar1["static_fraction_median"] < r_static["static_fraction_median"]
    assert r_static["held_position_correlation_median"] > 0.8  # a static offset predicts itself across halves


def test_cross_validated_white_share_exceeds_in_sample():
    rng = np.random.default_rng(1002)
    counts = _simulate(rng, n_trials=16, n_units=30, n_bins=24, cond_amp=0.1, static_sd=0.05,
                        slow_sd=0.05, slow_rho=0.5, white_sd=1.0, unit_noise_sd=0.3)
    result = partition_single_trial_variance(counts, n_splits=16, rng=np.random.default_rng(1003), bin_width_s=0.1)
    assert result["in_sample"]["white_fraction"] < result["white_fraction_median"]


def test_too_few_bins_returns_zero_slow_with_documented_status_not_none_or_raise():
    rng = np.random.default_rng(900)
    counts = _simulate(rng, n_trials=40, n_units=20, n_bins=3, cond_amp=0.5, static_sd=0.5,
                        slow_sd=0.5, slow_rho=0.6, white_sd=0.5, unit_noise_sd=0.3)
    result = partition_single_trial_variance(counts, n_splits=8, rng=np.random.default_rng(901), bin_width_s=0.1)
    assert result["slow_fraction_median"] == 0.0
    assert result["n_splits_with_fittable_decay"] == 0
    assert result["status"] == "fitted"  # the shares themselves are still computed; only the decay fit failed
