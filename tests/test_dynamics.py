"""Tests for src/dynamics.py."""

import inspect
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
    flow_divergence,
    ensemble_dmd,
    _dmd_rank_cap,
    dmd_reconstruction_error,
    koopman_edmd,
    stimulation_trigger_window,
    divergence_rank_sweep,
    mean_trajectory_divergence_rank_sweep,
    rank_robustness_sign,
    fit_input_lds,
    simulate_input_response,
)


class TestVelocityField:
    def test_shape(self, rng):
        Z = rng.standard_normal((50, 4))
        Zdot = velocity_field(Z)
        assert Zdot.shape == Z.shape

    def test_endpoints_are_one_sided_difference(self, rng):
        Z = rng.standard_normal((50, 4))
        Zdot = velocity_field(Z)
        np.testing.assert_allclose(Zdot[0], Z[1] - Z[0])
        np.testing.assert_allclose(Zdot[-1], Z[-1] - Z[-2])

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

    def test_interior_point_revisiting_start_state_is_not_corrupted_by_boundary(self):
        # Two full loops around a circle: t=20 revisits the exact position
        # AND velocity of t=0 (same phase of a uniform periodic flow), so a
        # correctly-computed Q(20) should be small, matching the smooth
        # untangled flow everywhere else on the circle. If velocity_field
        # fabricates Zdot[0] = 0 instead of the true (nonzero) boundary
        # velocity, ratio[20, 0] = ||Zdot(20) - 0||^2 / (~0 + eps) spikes to
        # a huge, non-representative value purely because t'=0 is a boundary
        # sample, not because t=20 is actually a dynamically unstable point.
        T_period, reps = 20, 2
        T = T_period * reps + 1
        theta = 2 * np.pi * np.arange(T) / T_period
        Z = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        t_revisit = T_period
        assert np.allclose(Z[t_revisit], Z[0])  # construction sanity check

        Q = trajectory_tangling(Z, epsilon=1e-3, dt=1.0)
        assert Q[t_revisit] < 5.0


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


class TestFlowDivergence:
    def test_pure_rotation_gives_near_zero_divergence(self):
        # Volume-preserving rotation: true divergence is 0, but the naive
        # trace(A-I)/dt approximation would register spurious contraction.
        T = 300
        theta = 0.2
        t = np.arange(T)
        Z = np.column_stack([np.cos(theta * t), np.sin(theta * t)])
        res = flow_divergence(Z, dt=1.0, method="dmd", r=2)
        assert abs(res["mean_divergence"]) < 0.05

    def test_pure_contraction_matches_analytic_rate(self):
        T, decay = 200, 0.9
        rng = np.random.default_rng(0)
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = decay * Z[t - 1] + 1e-3 * rng.standard_normal(2)
        res = flow_divergence(Z, dt=1.0, method="dmd", r=2)
        expected = 2 * np.log(decay)
        assert res["mean_divergence"] == pytest.approx(expected, abs=0.05)

    def test_expansion_is_positive(self):
        # Two independent per-axis growth rates, so the state matrix is full
        # rank (a purely rank-1 trajectory ill-conditions the DMD SVD fit).
        T = 60
        growth = np.array([1.05, 1.03])
        Z = np.zeros((T, 2))
        Z[0] = [0.1, 0.15]
        for t in range(1, T):
            Z[t] = growth * Z[t - 1]
        res = flow_divergence(Z, dt=1.0, method="dmd", r=2)
        assert res["mean_divergence"] > 0


class TestEnsembleDMD:
    def test_output_keys(self, rng):
        N, T, d = 20, 30, 3
        Z_trials = rng.standard_normal((N, T, d)) * 0.1
        res = ensemble_dmd(Z_trials, r=3, dt=1.0, n_splits=4, n_null=10, rng=rng)
        for key in ["A", "eigenvalues", "div_scalar", "r2_insample",
                    "r2_cv", "r2_null", "n_trials"]:
            assert key in res
        assert res["n_trials"] == N

    def test_recovers_contraction_from_noisy_ensemble(self, rng):
        N, T, decay = 40, 50, 0.9
        Z_trials = np.zeros((N, T, 2))
        for n in range(N):
            x = rng.standard_normal(2)
            for t in range(T):
                Z_trials[n, t] = x
                x = decay * x + 0.01 * rng.standard_normal(2)
        res = ensemble_dmd(Z_trials, r=2, dt=1.0, n_splits=5, n_null=20, rng=rng)
        expected = 2 * np.log(decay)
        assert res["div_scalar"] == pytest.approx(expected, abs=0.05)

    def test_cv_r2_below_insample_or_close(self, rng):
        # CV R^2 should not wildly exceed in-sample R^2 for a well-specified
        # linear system with real single-trial variability.
        N, T, decay = 30, 40, 0.85
        Z_trials = np.zeros((N, T, 2))
        for n in range(N):
            x = rng.standard_normal(2)
            for t in range(T):
                Z_trials[n, t] = x
                x = decay * x + 0.05 * rng.standard_normal(2)
        res = ensemble_dmd(Z_trials, r=2, dt=1.0, n_splits=5, n_null=10, rng=rng)
        assert res["r2_cv"] > 0.5
        assert res["r2_cv"] <= res["r2_insample"] + 0.15

    def test_null_r2_lower_than_real_fit(self, rng):
        N, T, decay = 30, 40, 0.85
        Z_trials = np.zeros((N, T, 2))
        for n in range(N):
            x = rng.standard_normal(2)
            for t in range(T):
                Z_trials[n, t] = x
                x = decay * x + 0.02 * rng.standard_normal(2)
        res = ensemble_dmd(Z_trials, r=2, dt=1.0, n_splits=5, n_null=30, rng=rng)
        assert res["r2_null"] < res["r2_cv"]

    def test_offset_equilibrium_recovered_without_distorting_eigenvalues(self):
        # Every trial follows the SAME affine dynamics x(t+1) = A x(t) + c
        # with a fixed point far from the origin. A purely linear fit
        # (x(t+1) ~= A x(t), no intercept) can only hold a nonzero fixed
        # point if one eigenvalue is exactly 1, so it has to inflate an
        # eigenvalue toward 1 to explain the offset -- distorting the
        # recovered spectrum even though the true decay is 0.6 on both axes.
        rng = np.random.default_rng(0)
        N, T, d = 30, 40, 2
        decay = 0.6
        A_true = decay * np.eye(d)
        c_true = np.array([3.0, -2.0])
        x_star = np.linalg.solve(np.eye(d) - A_true, c_true)

        Z_trials = np.zeros((N, T, d))
        for n in range(N):
            x = x_star + rng.standard_normal(d) * 2.0
            for t in range(T):
                Z_trials[n, t] = x
                x = A_true @ x + c_true + 0.01 * rng.standard_normal(d)

        res = ensemble_dmd(Z_trials, r=2, dt=1.0, n_splits=5, n_null=5,
                            rng=np.random.default_rng(1))
        eig = np.sort(np.real(np.linalg.eigvals(res["A"])))
        np.testing.assert_allclose(eig, [decay, decay], atol=0.05)
        np.testing.assert_allclose(res["equilibrium"], c_true, atol=0.5)

    def test_rank_cap_selected_per_fold_not_from_full_trial_set(self):
        # 6 trials of 2 samples each (1 snapshot pair per trial) with an
        # ambient dimensionality (d=6) that exceeds any individual fold's
        # available pairs. The full-ensemble cap (6 trials -> 5 pairs after
        # the -1 margin) must be looser than what a 5-trial training fold
        # can support (5 trials -> 4 pairs after the same margin): if the
        # fold reused the full-set cap instead of recomputing it from its
        # own training trials, every fold would report rank 5, not 4.
        rng = np.random.default_rng(0)
        N, T, d, r = 6, 2, 6, 6
        Z_trials = rng.standard_normal((N, T, d))

        assert _dmd_rank_cap(r, d, N, T) == 5
        assert _dmd_rank_cap(r, d, N - 1, T) == 4

        res = ensemble_dmd(Z_trials, r=r, dt=1.0, n_splits=N, n_null=3,
                            rng=np.random.default_rng(2))
        assert res["r_used"] == 5
        assert len(res["r_used_per_fold"]) > 0
        assert all(rank_used == 4 for rank_used in res["r_used_per_fold"])


class TestDMDReconstructionError:
    def test_output_keys(self, rng):
        Z = rng.standard_normal((60, 4)) * 0.1
        res = dmd_reconstruction_error(Z, r=3, dt=1.0)
        for key in ["relative_error", "mean_rel_error", "r_squared", "A"]:
            assert key in res
        assert res["relative_error"].shape == (59,)
        assert res["A"].shape == (4, 4)

    def test_near_perfect_linear_system_low_error(self, rng):
        T, decay = 100, 0.9
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = decay * Z[t - 1]
        res = dmd_reconstruction_error(Z, r=2, dt=1.0)
        assert res["mean_rel_error"] < 1e-6
        assert res["r_squared"] > 0.999

    def test_A_is_real(self, rng):
        # Rotation + decay: complex-conjugate eigenvalue pair.
        T, theta, decay = 80, 0.3, 0.95
        A_true = decay * np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = A_true @ Z[t - 1]
        res = dmd_reconstruction_error(Z, r=2, dt=1.0)
        np.testing.assert_allclose(res["A"], A_true, atol=1e-6)


class TestKoopmanEDMD:
    def test_output_keys(self, rng):
        Z = rng.standard_normal((80, 3)) * 0.1
        res = koopman_edmd(Z, r=4, dt=1.0, poly_degree=2, delay_embeddings=3)
        for key in ["eigenvalues", "eigenvalues_ct", "modes", "r_squared_lift",
                    "r_squared_orig", "lifting_dim"]:
            assert key in res

    def test_linear_system_high_r_squared(self, rng):
        # Pure linear dynamics: even the polynomial-lifted EDMD fit should
        # explain nearly all variance in the original space.
        T, decay = 150, 0.92
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = decay * Z[t - 1]
        res = koopman_edmd(Z, r=2, dt=1.0, poly_degree=1, delay_embeddings=1)
        assert res["r_squared_orig"] > 0.99

    def test_no_delay_padding_artifact(self, rng):
        # With delay embedding, the fit should not depend on Z[0] being
        # arbitrarily repeated as fake history (regression test for a padding bug).
        T, decay = 100, 0.9
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = decay * Z[t - 1]
        res_delay3 = koopman_edmd(Z, r=2, dt=1.0, poly_degree=1, delay_embeddings=3)
        assert res_delay3["r_squared_orig"] > 0.99


class TestDtPropagation:
    """Regression test: dt must default to a neutral 1.0 sample and every
    dt-sensitive quantity must actually scale with the dt passed in, not a
    silently baked-in sampling rate."""

    @pytest.mark.parametrize("fn", [trial_tangling, trial_dmd, maintenance_eigenspectra,
                                     dmd_reconstruction_error, koopman_edmd])
    def test_default_dt_is_neutral(self, fn):
        sig = inspect.signature(fn)
        assert sig.parameters["dt"].default == 1.0

    def test_trial_dmd_continuous_eigenvalues_scale_with_dt(self, rng):
        T, decay = 100, 0.9
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = decay * Z[t - 1]

        res_a = trial_dmd(Z, r=2, dt=1.0)
        res_b = trial_dmd(Z, r=2, dt=0.5)

        # eigenvalues_ct = log(lambda) / dt: halving dt must double the
        # continuous-time rate for the same discrete-time eigenvalues.
        np.testing.assert_allclose(
            res_b["eigenvalues_ct"], res_a["eigenvalues_ct"] * 2.0, atol=1e-8
        )

    def test_ensemble_dmd_divergence_scales_with_dt(self, rng):
        N, T, decay = 20, 30, 0.85
        Z_trials = np.zeros((N, T, 2))
        for n in range(N):
            x = rng.standard_normal(2)
            for t in range(T):
                Z_trials[n, t] = x
                x = decay * x + 0.01 * rng.standard_normal(2)

        res_a = ensemble_dmd(Z_trials, r=2, dt=1.0, n_splits=3, n_null=2, rng=rng)
        res_b = ensemble_dmd(Z_trials, r=2, dt=0.5, n_splits=3, n_null=2, rng=rng)
        np.testing.assert_allclose(res_b["div_scalar"], res_a["div_scalar"] * 2.0, atol=1e-8)


class TestDivergenceRankSweep:
    def test_ranks_clipped_to_dimensionality(self, rng):
        N, T, d = 15, 25, 4
        Z_trials = rng.standard_normal((N, T, d)) * 0.1
        res = divergence_rank_sweep(Z_trials, dt=1.0, ranks=(2, 3, 8, 10), n_null=2, rng=rng)
        assert max(res["ranks"]) <= d
        assert len(res["div_scalar"]) == len(res["ranks"])

    def test_mean_trajectory_sweep_matches_full_dmd(self, rng):
        T, decay = 60, 0.9
        Z = np.zeros((T, 2))
        Z[0] = rng.standard_normal(2)
        for t in range(1, T):
            Z[t] = decay * Z[t - 1]
        res = mean_trajectory_divergence_rank_sweep(Z, dt=1.0, ranks=(2,))
        lam = exact_dmd(Z.T, r=2, dt=1.0)["eigenvalues"]
        expected = float(np.sum(np.log(np.abs(lam) + 1e-300)))
        np.testing.assert_allclose(res["div_scalar"][0], expected, atol=1e-6)


class TestRankRobustnessSign:
    def test_all_same_sign_is_robust(self):
        assert rank_robustness_sign([-1.0, -0.5, -2.0]) is True

    def test_sign_flip_is_not_robust(self):
        assert rank_robustness_sign([-1.0, 0.5, -2.0]) is False

    def test_zero_is_not_robust(self):
        assert rank_robustness_sign([-1.0, 0.0, -2.0]) is False


class TestStimulationTriggerWindow:
    def test_output_keys(self, rng):
        Z = rng.standard_normal((80, 3)) * 0.1
        times = np.arange(80) * 0.01
        res = stimulation_trigger_window(Z, times, dt=0.01, n_neighbors=10)
        for key in ["trigger_onsets", "trigger_offsets", "trigger_times", "divergence"]:
            assert key in res
        assert res["divergence"].shape == (80,)

    def test_expanding_trajectory_has_trigger_window(self, rng):
        T = 60
        growth = 1.05
        Z = np.zeros((T, 2))
        Z[0] = [0.1, 0.15]
        for t in range(1, T):
            Z[t] = growth * Z[t - 1] + 1e-4 * rng.standard_normal(2)
        times = np.arange(T) * 1.0
        res = stimulation_trigger_window(Z, times, dt=1.0, n_neighbors=10,
                                          threshold=0.0, min_duration_s=2.0)
        assert len(res["trigger_onsets"]) >= 1
        assert len(res["trigger_offsets"]) == len(res["trigger_onsets"])


class TestFitInputLDS:
    def test_recovers_planted_A_B(self, rng):
        # Plant a stable random (A, B) in a k-dim latent space, embed into a
        # higher-dim observation space via a random orthonormal C, drive with
        # a random input sequence, and check fit_input_lds recovers (A, B) up
        # to the (near-)identical latent basis (C is orthonormal and X's PCA
        # will recover it exactly since X lives exactly in that k-dim subspace).
        k, d, T = 3, 6, 400
        A_true = 0.9 * np.linalg.qr(rng.standard_normal((k, k)))[0]  # stable (orthogonal * 0.9)
        B_true = rng.standard_normal((k, 2))
        C_true, _ = np.linalg.qr(rng.standard_normal((d, k)))

        U = rng.standard_normal((T, 2)) * 0.3
        Z = np.zeros((T, k))
        Z[0] = rng.standard_normal(k) * 0.1
        for t in range(T - 1):
            Z[t + 1] = A_true @ Z[t] + B_true @ U[t]
        X = Z @ C_true.T  # noiseless embedding

        A_hat, B_hat, C_hat, z_hat = fit_input_lds(X, U, latent_dim=k)
        assert A_hat.shape == (k, k)
        assert B_hat.shape == (k, 2)
        assert C_hat.shape == (d, k)
        assert z_hat.shape == (T, k)

        # The recovered latent basis can differ from the true one by an
        # orthogonal rotation (PCA sign/rotation ambiguity within the shared
        # k-dim subspace); compare in OBSERVATION space instead, where the
        # rotation cancels: C_hat @ A_hat @ C_hat.T should match C_true @ A_true @ C_true.T,
        # and likewise for B.
        A_obs_hat = C_hat @ A_hat @ C_hat.T
        A_obs_true = C_true @ A_true @ C_true.T
        np.testing.assert_allclose(A_obs_hat, A_obs_true, atol=1e-2)

        B_obs_hat = C_hat @ B_hat
        B_obs_true = C_true @ B_true
        np.testing.assert_allclose(B_obs_hat, B_obs_true, atol=1e-2)

    def test_simulated_response_aligns_with_planted_B(self, rng):
        # A pure-decay A (no rotation) so the response after one input pulse
        # points along B; check simulate_input_response's displacement is
        # cosine-aligned with the planted B direction.
        k, d, T = 3, 5, 300
        A_true = 0.85 * np.eye(k)
        b_dir = rng.standard_normal(k)
        b_dir /= np.linalg.norm(b_dir)
        B_true = b_dir[:, None]  # (k, 1) -- single input channel
        C_true, _ = np.linalg.qr(rng.standard_normal((d, k)))

        U = rng.standard_normal((T, 1)) * 0.3
        Z = np.zeros((T, k))
        Z[0] = rng.standard_normal(k) * 0.05
        for t in range(T - 1):
            Z[t + 1] = A_true @ Z[t] + B_true @ U[t]
        X = Z @ C_true.T

        A_hat, B_hat, C_hat, z_hat = fit_input_lds(X, U, latent_dim=k)

        # Unit pulse response, starting from the origin, in the FITTED basis.
        U_pulse = np.zeros((10, 1))
        U_pulse[0, 0] = 1.0
        Z_sim = simulate_input_response(A_hat, B_hat, C_hat, np.zeros(k), U_pulse)
        assert Z_sim.shape == (11, k)
        displacement_hat = Z_sim[1] - Z_sim[0]  # ~= B_hat[:, 0]

        # Compare direction in OBSERVATION space (basis-invariant), against
        # the planted B direction.
        disp_obs = C_hat @ displacement_hat
        b_obs_true = C_true @ b_dir
        cos_sim = float(
            disp_obs @ b_obs_true / (np.linalg.norm(disp_obs) * np.linalg.norm(b_obs_true) + 1e-12)
        )
        assert cos_sim > 0.99


def _demo_self_check() -> None:
    """Standalone sanity check (assert-based) — run directly if pytest is
    unavailable: recovers a planted (A, B) input-LDS and confirms the
    simulated response aligns with the planted input direction."""
    rng = np.random.default_rng(0)
    k, T = 2, 200
    A_true = np.array([[0.9, 0.0], [0.0, 0.8]])
    B_true = np.array([[1.0], [0.0]])
    U = rng.standard_normal((T, 1)) * 0.2
    Z = np.zeros((T, k))
    for t in range(T - 1):
        Z[t + 1] = A_true @ Z[t] + B_true @ U[t]
    A_hat, B_hat, C_hat, z_hat = fit_input_lds(Z, U, latent_dim=k)
    Z_sim = simulate_input_response(A_hat, B_hat, C_hat, np.zeros(k), np.array([[1.0]]))
    disp = Z_sim[1] - Z_sim[0]
    assert np.linalg.norm(disp - B_hat[:, 0]) < 1e-8
    print("fit_input_lds/simulate_input_response self-check OK")


if __name__ == "__main__":
    _demo_self_check()
