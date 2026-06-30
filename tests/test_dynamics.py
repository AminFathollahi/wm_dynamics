"""Tests for src/dynamics.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dynamics import (
    velocity_field,
    trajectory_tangling,
    trial_tangling,
    exact_dmd,
    trial_dmd,
    eigenspectrum_stability,
    maintenance_eigenspectra,
    velocity_autocorrelation,
    ring_attractor_phase,
    local_linear_stability,
)


class TestVelocityField:
    def test_shape(self, rng):
        Z = rng.standard_normal((50, 4))
        Zdot = velocity_field(Z)
        assert Zdot.shape == Z.shape

    def test_endpoints_zero(self, rng):
        Z = rng.standard_normal((50, 4))
        Zdot = velocity_field(Z)
        np.testing.assert_array_equal(Zdot[0], np.zeros(4))
        np.testing.assert_array_equal(Zdot[-1], np.zeros(4))

    def test_central_difference_accuracy(self):
        t = np.linspace(0, 1, 1000)
        dt = t[1] - t[0]
        Z = np.sin(2 * np.pi * t)[:, None]
        Zdot = velocity_field(Z, dt=dt)
        expected = 2 * np.pi * np.cos(2 * np.pi * t)
        # Interior points should be accurate
        np.testing.assert_allclose(Zdot[50:-50, 0], expected[50:-50], atol=1e-4)


class TestTrajectorTangling:
    def test_output_shape(self, rng):
        Z = rng.standard_normal((80, 4))
        Q = trajectory_tangling(Z)
        assert Q.shape == (80,)

    def test_nonnegative(self, rng):
        Z = rng.standard_normal((60, 3))
        Q = trajectory_tangling(Z)
        assert np.all(Q >= 0)

    def test_endpoints_zero(self, rng):
        Z = rng.standard_normal((60, 4))
        Q = trajectory_tangling(Z)
        assert Q[0] == 0.0
        assert Q[-1] == 0.0

    def test_stable_limit_cycle_low_Q(self):
        # Perfect limit cycle: same velocity everywhere at same state
        t = np.linspace(0, 2 * np.pi, 200)
        Z = np.stack([np.cos(t), np.sin(t)], axis=1)
        Q = trajectory_tangling(Z, epsilon=1e-3)
        # For a pure cycle, tangling should be low (not zero due to discretisation)
        assert np.median(Q[5:-5]) < 10.0

    def test_tangling_increases_with_divergence(self, rng):
        # Tangled: same position, different velocity
        T, d = 50, 3
        Z = rng.standard_normal((T, d)) * 0.1  # states clustered near origin
        Q = trajectory_tangling(Z, epsilon=1e-6)
        Q_stable = trajectory_tangling(np.outer(np.linspace(0, 1, T), np.ones(d)))
        assert Q[5:-5].mean() > Q_stable[5:-5].mean()


class TestTrialTangling:
    def test_output_shape(self, rng):
        N, T, d = 10, 50, 4
        Z_trials = rng.standard_normal((N, T, d))
        Q = trial_tangling(Z_trials)
        assert Q.shape == (N, T)


class TestExactDMD:
    def test_recovers_system_matrix(self):
        # Rotation + decay: n distinct eigenvalues, not degenerate
        n = 4
        t = 200
        # Build A with distinct real eigenvalues to avoid rank degeneracy
        lam_true = np.array([0.95, 0.90, 0.85, 0.80])
        V = np.linalg.qr(np.random.default_rng(7).standard_normal((n, n)))[0]
        A_true = V @ np.diag(lam_true) @ V.T

        X = np.zeros((n, t))
        X[:, 0] = np.ones(n)
        for k in range(1, t):
            X[:, k] = A_true @ X[:, k - 1]

        dmd = exact_dmd(X, r=n)
        lam_recovered = np.sort(np.abs(dmd["eigenvalues"]))
        lam_expected = np.sort(lam_true)
        np.testing.assert_allclose(lam_recovered, lam_expected, atol=1e-6)

    def test_output_keys(self, rng):
        X = rng.standard_normal((8, 50))
        result = exact_dmd(X, r=4)
        for key in ["eigenvalues", "eigenvalues_ct", "modes", "amplitudes", "rank"]:
            assert key in result

    def test_rank_truncation(self, rng):
        X = rng.standard_normal((10, 40))
        result = exact_dmd(X, r=3)
        assert result["rank"] == 3
        assert len(result["eigenvalues"]) == 3

    def test_modes_shape(self, rng):
        n, t = 8, 60
        X = rng.standard_normal((n, t))
        result = exact_dmd(X, r=4)
        assert result["modes"].shape[0] == n


class TestEigenspectrumStability:
    def test_stable_system(self):
        lam = np.array([0.95, 0.9, 0.85, 0.8], dtype=complex)
        stats = eigenspectrum_stability(lam)
        assert stats["max_growth_rate"] < 0.0
        assert stats["n_stable_modes"] == 4
        assert stats["unit_circle_dist"] > 0

    def test_unstable_mode(self):
        lam = np.array([1.1, 0.9, 0.8], dtype=complex)
        stats = eigenspectrum_stability(lam)
        assert stats["max_growth_rate"] > 0.0

    def test_unit_circle_distance_zero_for_marginally_stable(self):
        lam = np.array([1.0 + 0j, -1.0 + 0j])
        stats = eigenspectrum_stability(lam)
        np.testing.assert_allclose(stats["unit_circle_dist"], 0.0, atol=1e-10)


class TestVelocityAutocorrelation:
    def test_output_shape(self, rng):
        Z = rng.standard_normal((100, 4))
        ac = velocity_autocorrelation(Z, max_lag=20)
        assert ac.shape == (21,)

    def test_lag_zero_near_one(self, rng):
        Z = rng.standard_normal((80, 3))
        ac = velocity_autocorrelation(Z, max_lag=10)
        np.testing.assert_allclose(ac[0], 1.0, atol=1e-8)

    def test_smooth_trajectory_slow_decay(self):
        t = np.linspace(0, 4 * np.pi, 200)
        Z = np.stack([np.sin(t), np.cos(t)], axis=1)
        ac = velocity_autocorrelation(Z, max_lag=30)
        # Smooth limit cycle: ac[10] should still be strongly correlated
        assert ac[10] > 0.5

    def test_random_trajectory_faster_decay(self, rng):
        Z_smooth = np.stack([np.sin(np.linspace(0, 4 * np.pi, 200)),
                              np.cos(np.linspace(0, 4 * np.pi, 200))], axis=1)
        Z_random = rng.standard_normal((200, 2))
        ac_smooth = velocity_autocorrelation(Z_smooth, max_lag=20)
        ac_random = velocity_autocorrelation(Z_random, max_lag=20)
        assert ac_smooth[10] > ac_random[10]


class TestRingAttractorPhase:
    def test_output_shape(self):
        t = np.linspace(0, 2 * np.pi, 100)
        Z = np.stack([np.cos(t), np.sin(t)], axis=1)
        phase = ring_attractor_phase(Z, smooth_sigma=0.0)
        assert phase.shape == (100,)

    def test_range(self, rng):
        Z = rng.standard_normal((80, 4))
        phase = ring_attractor_phase(Z, smooth_sigma=0.0)
        assert np.all(phase >= -np.pi - 1e-6)
        assert np.all(phase <= np.pi + 1e-6)

    def test_circle_monotone_phase(self):
        t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        Z = np.stack([np.cos(t), np.sin(t)], axis=1)
        phase = ring_attractor_phase(Z, smooth_sigma=0.0)
        # Phase should span from -pi to +pi (wraps once)
        phase_span = phase.max() - phase.min()
        assert phase_span > np.pi  # spans most of the circle

    def test_works_on_high_d(self, rng):
        Z = rng.standard_normal((60, 8))
        phase = ring_attractor_phase(Z)
        assert phase.shape == (60,)


class TestLocalLinearStability:
    def test_output_keys(self, rng):
        Z = rng.standard_normal((50, 3))
        result = local_linear_stability(Z, n_neighbors=10)
        assert "eigenvalues" in result
        assert "max_real" in result
        assert "mean_real" in result

    def test_output_shapes(self, rng):
        T, d = 60, 3
        Z = rng.standard_normal((T, d))
        result = local_linear_stability(Z, n_neighbors=10)
        assert result["eigenvalues"].shape == (T, d)
        assert result["max_real"].shape == (T,)

    def test_stable_system_negative_real(self):
        # Linear decay: Z[t+1] = 0.9 * Z[t]; Jacobian eigenvalues ~ -0.1 (discrete → cont)
        T, d = 80, 2
        Z = np.zeros((T, d))
        Z[0] = np.array([1.0, 0.5])
        for t in range(1, T):
            Z[t] = 0.9 * Z[t - 1]
        result = local_linear_stability(Z, n_neighbors=15)
        interior = result["max_real"][5:-5]
        valid = ~np.isnan(interior)
        if valid.sum() > 5:
            assert interior[valid].mean() < 0
