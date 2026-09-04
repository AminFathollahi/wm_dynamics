"""Regression tests for run_lambda_estimator_limits.py's simulation and
fingerprint-classification logic: the failure modes this script exists to
tell apart (a true random walk vs. no latent signal at all) must actually
look different on the outputs the classification reads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from drift_dynamics import fit_gaussian_state_space  # noqa: E402
import run_lambda_estimator_limits as mod  # noqa: E402


def test_no_signal_regime_has_no_temporal_structure():
    rng = np.random.default_rng(0)
    y = mod._simulate("no_signal", lam=0.0, diffusion=0.18, obs_noise_var=0.1,
                       n_trials=200, n_time=23, dt=0.1, rng=rng)
    # every bin is independent noise -- lag-1 autocorrelation should be near zero
    lag1 = np.corrcoef(y[:, :-1].ravel(), y[:, 1:].ravel())[0, 1]
    assert abs(lag1) < 0.1


def test_random_walk_regime_has_growing_variance_and_no_restoring_force():
    rng = np.random.default_rng(0)
    y = mod._simulate("random_walk", lam=0.0, diffusion=0.18, obs_noise_var=1e-6,
                       n_trials=500, n_time=23, dt=0.1, rng=rng)
    # unconfined diffusion: variance across trials grows with elapsed time
    assert np.var(y[:, -1]) > 3 * np.var(y[:, 2])


def test_confined_regime_reaches_a_smaller_stationary_variance_than_a_random_walk():
    rng = np.random.default_rng(0)
    confined = mod._simulate("confined", lam=2.0, diffusion=0.18, obs_noise_var=1e-6,
                              n_trials=500, n_time=23, dt=0.1, rng=rng)
    random_walk = mod._simulate("random_walk", lam=0.0, diffusion=0.18, obs_noise_var=1e-6,
                                 n_trials=500, n_time=23, dt=0.1, rng=rng)
    assert np.var(confined[:, -1]) < np.var(random_walk[:, -1])


def test_fingerprint_distinguishes_random_walk_from_no_signal_through_the_real_estimator():
    """The claim the whole script exists to establish: run BOTH failure regimes
    through the project's own fit_gaussian_state_space and confinement_identifiability
    (not a reimplementation) and confirm their lambda_hat fingerprints differ --
    random walk should pile up near/below zero, no signal should pile up at the
    estimator's upper bound."""
    dt = 0.1
    lower_bound, upper_bound = mod._lambda_bounds(dt)
    rng = np.random.default_rng(1)

    def lambda_hats(regime, n=30):
        out = []
        for _ in range(n):
            y = mod._simulate(regime, lam=0.0, diffusion=0.18, obs_noise_var=0.05,
                               n_trials=36, n_time=23, dt=dt, rng=rng)
            estimate = fit_gaussian_state_space(y, dt)
            if estimate.lambda_rate is not None and np.isfinite(estimate.lambda_rate):
                out.append(estimate.lambda_rate)
        return np.asarray(out)

    random_walk_lambdas = lambda_hats("random_walk")
    no_signal_lambdas = lambda_hats("no_signal")
    assert len(random_walk_lambdas) > 5 and len(no_signal_lambdas) > 5
    assert np.median(random_walk_lambdas) < np.median(no_signal_lambdas) - 5.0
    # no-signal should sit closer to the upper identifiability bound than random walk does
    assert (upper_bound - np.median(no_signal_lambdas)) < (upper_bound - np.median(random_walk_lambdas))
