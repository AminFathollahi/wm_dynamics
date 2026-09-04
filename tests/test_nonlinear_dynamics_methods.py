"""Self-checks for the small pure-math helpers behind the DPAD/DFINE/NoMAD
dynamics scripts (scripts/run_dpad_dynamics_dissociation.py,
run_dfine_nonlinear_dynamics_check.py, run_nomad_cross_session_alignment.py).

These scripts import DPAD/TensorFlow, which live in the isolated .venv_dpad/
env (numpy==1.26.4 pin), not the main wm_dynamics env -- so this file skips
itself under any env without those packages, and is meant to be run via:
    ./.venv_dpad/bin/python -m pytest tests/test_nonlinear_dynamics_methods.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("DPAD")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_dpad_dynamics_dissociation import _dpad_relevant_direction
from run_dfine_nonlinear_dynamics_check import _r2
from run_nomad_cross_session_alignment import _gaussian_kl


def test_dpad_relevant_direction_recovers_known_direction():
    rng = np.random.default_rng(0)
    d, n_trials, T = 5, 30, 10
    true_w = rng.normal(size=d)
    true_w /= np.linalg.norm(true_w)
    Y = rng.normal(size=(n_trials, T, d))
    x1 = Y @ true_w  # noiseless: stage-1 latent IS the projection onto true_w
    w_hat = _dpad_relevant_direction(Y, x1)
    assert abs(np.dot(w_hat, true_w)) > 0.999


def test_r2_perfect_and_mean_baseline():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(50, 3))
    assert _r2(true, true) == pytest.approx(1.0)
    mean_pred = np.tile(true.mean(axis=0, keepdims=True), (50, 1))
    assert _r2(true, mean_pred) == pytest.approx(0.0, abs=1e-8)


def test_gaussian_kl_zero_for_identical_distributions():
    k = 4
    mu = tf.constant(np.zeros(k), dtype=tf.float32)
    cov = tf.constant(np.eye(k), dtype=tf.float32)
    kl = _gaussian_kl(mu, cov, mu, cov)
    assert float(kl.numpy()) == pytest.approx(0.0, abs=1e-4)


def test_gaussian_kl_matches_closed_form_shifted_means():
    k = 3
    mu_p = tf.constant(np.array([1.0, 0.5, -0.5]), dtype=tf.float32)
    mu_q = tf.constant(np.zeros(k), dtype=tf.float32)
    cov = tf.constant(np.eye(k), dtype=tf.float32)
    kl = _gaussian_kl(mu_p, cov, mu_q, cov)
    expected = 0.5 * np.sum((mu_p.numpy() - mu_q.numpy()) ** 2)  # equal isotropic covs -> KL = 0.5*||mu_p-mu_q||^2
    assert float(kl.numpy()) == pytest.approx(expected, rel=1e-3)


if __name__ == "__main__":
    test_dpad_relevant_direction_recovers_known_direction()
    test_r2_perfect_and_mean_baseline()
    test_gaussian_kl_zero_for_identical_distributions()
    test_gaussian_kl_matches_closed_form_shifted_means()
    print("All nonlinear-dynamics-methods self-checks passed.")
