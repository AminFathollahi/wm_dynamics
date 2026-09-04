"""Scientific regression tests for the achievable-control-ceiling estimator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from control import scalar_steady_state_kalman_error  # noqa: E402
from run_control_ceiling import achievable_reduction_curve, ceiling_from_lambda_diffusion  # noqa: E402


def _analytic_p(a: float, q: float, r: float) -> tuple[float, float]:
    b = r * (1 - a**2) - q
    c = -q * r
    p = (-b + np.sqrt(b**2 - 4 * c)) / 2.0
    p_post = p * r / (p + r)
    return p, p_post


@pytest.mark.parametrize("a,q,r", [(0.9, 0.1, 0.5), (0.5, 1.0, 2.0), (0.99, 0.01, 10.0), (0.3, 2.0, 0.01)])
def test_p_matches_analytic_solution(a, q, r):
    p, p_post = scalar_steady_state_kalman_error(a, q, r)
    p_expected, p_post_expected = _analytic_p(a, q, r)
    assert p == pytest.approx(p_expected, rel=1e-6)
    assert p_post == pytest.approx(p_post_expected, rel=1e-6)


def test_achievable_reduction_vanishes_as_observation_noise_grows():
    a, q = 0.9, 0.1
    _, p_post_small_r = scalar_steady_state_kalman_error(a, q, 1e-3)
    _, p_post_large_r = scalar_steady_state_kalman_error(a, q, 1e6)
    stationary_variance = q / (1 - a**2)
    reduction_small_r = 1.0 - p_post_small_r / stationary_variance
    reduction_large_r = 1.0 - p_post_large_r / stationary_variance
    assert reduction_large_r < 0.01
    assert reduction_small_r > reduction_large_r


def test_achievable_reduction_approaches_noiseless_bound_as_observation_noise_vanishes():
    a, q, dt = 0.9, 0.1, 0.1
    lambda_rate = -np.log(a) / dt
    stationary_variance = q / (1 - a**2)
    diffusion = lambda_rate * stationary_variance
    _, p_post = scalar_steady_state_kalman_error(a, q, 1e-8)
    ceiling = ceiling_from_lambda_diffusion(lambda_rate, diffusion, stationary_variance, p_post)
    curve = achievable_reduction_curve(lambda_rate, diffusion, p_post, stationary_variance, gains=[1e6])
    # With near-zero observation noise the estimation floor is negligible and
    # the achievable reduction at a very large gain approaches the fully
    # observed noiseless bound (variance -> 0, reduction -> 1).
    assert ceiling["ceiling_fraction"] < 0.01
    assert curve[0]["reduction"] > 0.99


def test_ceiling_fraction_is_bounded_in_unit_interval():
    for lambda_rate, q, r in [(2.0, 0.05, 0.5), (0.5, 1.0, 0.01), (5.0, 0.2, 10.0)]:
        dt = 0.1
        a = np.exp(-lambda_rate * dt)
        stationary_variance = q / (1 - a**2)
        diffusion = lambda_rate * stationary_variance
        _, p_post = scalar_steady_state_kalman_error(a, q, r)
        ceiling = ceiling_from_lambda_diffusion(lambda_rate, diffusion, stationary_variance, p_post)
        assert 0.0 <= ceiling["ceiling_fraction"] <= 1.0
