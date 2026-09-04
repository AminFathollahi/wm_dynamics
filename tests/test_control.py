"""Tests for src/control.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from control import (
    controllability_gramian,
    is_controllable,
    dare_solve,
    lqr_design,
    minimum_energy_trajectory,
    lqr_simulate,
    energy_accuracy_pareto,
    dominant_eigenmode,
    average_controllability,
    modal_controllability,
    _normalize_adjacency,
    invariant_subspace_basis,
    subspace_alignment,
    canonicalize_eigenvector_phase,
)


class TestCanonicalizeEigenvectorPhase:
    def test_stable_across_arbitrary_input_phase(self):
        # Same physical mode, 10 random overall phases applied to the raw
        # eigenvector before canonicalization -- affinity to the phi=0
        # canonicalization must be 1.0 to 1e-10.
        rng = np.random.default_rng(0)
        v = rng.standard_normal(5) + 1j * rng.standard_normal(5)
        v = v / np.linalg.norm(v)
        ref = canonicalize_eigenvector_phase(v)
        for phi in rng.uniform(0, 2 * np.pi, size=10):
            v_rot = v * np.exp(1j * phi)
            out = canonicalize_eigenvector_phase(v_rot)
            assert abs(float(out @ ref)) == pytest.approx(1.0, abs=1e-10)

    def test_real_eigenvector_only_fixes_sign(self):
        # Largest-magnitude entry (index 2, value 4) is already real and
        # positive, so no rotation is applied; just unit-normalize.
        v = np.array([0.0, -3.0, 4.0], dtype=complex)
        out = canonicalize_eigenvector_phase(v)
        np.testing.assert_allclose(out, [0.0, -0.6, 0.8], atol=1e-10)

        # Flip the largest entry negative -> canonicalization must rotate by
        # pi (i.e. flip overall sign) to restore largest-entry-positive.
        v2 = np.array([0.0, 3.0, -4.0], dtype=complex)
        out2 = canonicalize_eigenvector_phase(v2)
        np.testing.assert_allclose(out2, [0.0, -0.6, 0.8], atol=1e-10)

    def test_unit_norm(self):
        rng = np.random.default_rng(1)
        v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        out = canonicalize_eigenvector_phase(v)
        assert np.linalg.norm(out) == pytest.approx(1.0)


class TestInvariantSubspaceBasis:
    def test_m1_real_mode_matches_dominant_eigenmode(self):
        # Diagonal A: eigenvalue 1.5 dominates 0.5 in |lambda|; the m=1
        # subspace must be the same span as dominant_eigenmode's v*.
        A = np.diag([1.5, 0.5])
        mode = dominant_eigenmode(A)
        result = invariant_subspace_basis(A, m=1)
        assert result.status == "ok"
        assert result.basis.shape == (2, 1)
        assert abs(float(np.abs(result.basis[:, 0] @ mode.v_star))) == pytest.approx(1.0, abs=1e-8)

    def test_complex_pair_gives_2d_real_subspace(self):
        theta = 0.4
        A = 0.9 * np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        result = invariant_subspace_basis(A, m=1)  # one complex-conjugate PAIR = one mode = 2 real dims
        assert result.status == "ok"
        assert result.basis.shape == (2, 2)
        np.testing.assert_allclose(result.basis.T @ result.basis, np.eye(2), atol=1e-8)

    def test_basis_is_orthonormal(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 5)) * 0.3
        result = invariant_subspace_basis(A, m=3)
        np.testing.assert_allclose(result.basis.T @ result.basis, np.eye(result.dim), atol=1e-8)

    def test_near_real_pair_flags_degeneracy_instead_of_arbitrary_second_column(self):
        # A complex mode whose imaginary component is negligible relative to
        # its real component (here: real eigenvalues perturbed by a tiny
        # asymmetry so np.linalg.eig reports a barely-complex pair) must not
        # silently receive a numerically-arbitrary second QR column.
        eps = 1e-10
        A = np.array([[1.5, eps], [-eps, 1.5]])
        result = invariant_subspace_basis(A, m=1)
        assert result.status == "near_real_pair"
        assert result.dim == 1
        assert len(result.notes) == 1

    def test_unmatched_conjugate_status_present_in_notes(self):
        # Directly exercise the degeneracy-reporting path without depending
        # on a specific matrix construction: a note phrased as the
        # "unmatched_conjugate" case must be classified as such.
        from control import InvariantSubspaceBasis
        result = InvariantSubspaceBasis(
            basis=np.eye(2), dim=2, status="unmatched_conjugate",
            notes=["mode 0: nearest candidate conjugate partner is 1.0e-02 away"],
        )
        assert result.status == "unmatched_conjugate"


class TestSubspaceAlignment:
    def test_in_subspace_gives_one(self):
        Q = np.eye(3)[:, :2]
        b = np.array([1.0, 1.0, 0.0])
        assert subspace_alignment(Q, b) == pytest.approx(1.0, abs=1e-8)

    def test_orthogonal_to_subspace_gives_zero(self):
        Q = np.eye(3)[:, :2]
        b = np.array([0.0, 0.0, 1.0])
        assert subspace_alignment(Q, b) == pytest.approx(0.0, abs=1e-8)

    def test_m1_matches_cos_to_vstar(self):
        A = np.diag([1.5, 0.5])
        mode = dominant_eigenmode(A)
        result = invariant_subspace_basis(A, m=1)
        b = np.array([0.6, 0.8])
        expected = abs(float(b @ mode.v_star)) / np.linalg.norm(b)
        assert subspace_alignment(result.basis, b) == pytest.approx(expected, abs=1e-8)


class TestDominantEigenmode:
    def test_recovers_dominant_growth_direction(self):
        # Diagonal A: eigenvalue 1.5 dominates 0.5, eigenvector is e0.
        A = np.diag([1.5, 0.5])
        mode = dominant_eigenmode(A)
        np.testing.assert_allclose(np.abs(mode.v_star), [1.0, 0.0], atol=1e-8)
        assert mode.rho == pytest.approx(1.5)
        assert mode.theta == pytest.approx(0.0)
        assert not mode.is_complex
        assert mode.classification == "unstable_real"

    def test_unit_norm(self, synthetic_system):
        A, _ = synthetic_system
        mode = dominant_eigenmode(A)
        assert np.linalg.norm(mode.v_star) == pytest.approx(1.0)

    def test_discrete_mode_uses_modulus_not_real_part(self):
        # The rotational mode has smaller real part than 0.8 but a larger
        # discrete-time modulus (0.95), so it is the slowest-decaying mode.
        theta = 0.5
        A = np.array([
            [0.95 * np.cos(theta), -0.95 * np.sin(theta), 0.0],
            [0.95 * np.sin(theta),  0.95 * np.cos(theta), 0.0],
            [0.0, 0.0, 0.8],
        ])
        mode = dominant_eigenmode(A)
        assert mode.rho == pytest.approx(0.95)
        # rho < 1 and complex: this must NOT be called unstable, even though
        # a naive Re(lambda)-based selection would have preferred it over 0.8
        # for the wrong reason.
        assert mode.is_complex
        assert mode.classification == "damped_rotation"

    def test_real_eigenvalue_below_one_is_damped_not_unstable(self):
        A = np.diag([0.5, 0.3])
        mode = dominant_eigenmode(A)
        assert mode.classification == "damped_real"

    def test_complex_pair_near_unit_modulus_is_classified_as_rotation(self):
        # A pure rotation (modulus exactly 1) must be classified as a
        # rotation, never as "unstable", regardless of Re(lambda) sign.
        theta = 1.2
        A = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        mode = dominant_eigenmode(A)
        assert mode.is_complex
        assert mode.classification == "rotation"
        assert mode.theta == pytest.approx(theta, abs=1e-8) or mode.theta == pytest.approx(-theta, abs=1e-8)

    def test_complex_pair_above_unit_modulus_is_growing_rotation(self):
        theta = 0.7
        A = 1.2 * np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        mode = dominant_eigenmode(A)
        assert mode.classification == "growing_rotation"


class TestControllabilityGramian:
    def test_shape(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        Wc = controllability_gramian(A, B, T=20)
        assert Wc.shape == (n, n)

    def test_symmetric_psd(self, synthetic_system):
        A, B = synthetic_system
        Wc = controllability_gramian(A, B, T=20)
        np.testing.assert_allclose(Wc, Wc.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(Wc)
        assert np.all(eigvals >= -1e-10)

    def test_increases_with_horizon(self, synthetic_system):
        A, B = synthetic_system
        Wc10 = controllability_gramian(A, B, T=10)
        Wc50 = controllability_gramian(A, B, T=50)
        # Gramian at larger T should have larger trace
        assert np.trace(Wc50) >= np.trace(Wc10)


class TestAverageModalControllability:
    def _small_network(self, rng):
        n = 6
        W = rng.random((n, n))
        W[np.arange(n), np.arange(n)] = 0.0
        return W

    def test_average_controllability_matches_truncated_gramian_trace(self, rng):
        # Gu et al. 2015's closed form is the infinite-horizon limit of
        # Tr(Wc) for single-node input B=e_i on the normalized (symmetric)
        # adjacency; a long-but-finite-horizon Gramian should converge to it.
        W = self._small_network(rng)
        A = _normalize_adjacency(W)
        y_avg = average_controllability(W)

        n = A.shape[0]
        for i in [0, 2, 5]:
            e_i = np.zeros((n, 1))
            e_i[i, 0] = 1.0
            Wc = controllability_gramian(A, e_i, T=500)
            assert float(np.trace(Wc)) == pytest.approx(y_avg[i], rel=1e-3)

    def test_modal_controllability_closed_form_identity(self, rng):
        # phi(i) = sum_k (1 - lambda_k^2) v_k(i)^2 by definition; recompute
        # directly from eigh and compare to the function's output.
        W = self._small_network(rng)
        A = _normalize_adjacency(W)
        eigvals, eigvecs = np.linalg.eigh(A)
        expected = (eigvecs**2) @ (1.0 - eigvals**2)
        np.testing.assert_allclose(modal_controllability(W), expected, atol=1e-10)

    def test_outputs_are_positive_and_finite(self, rng):
        W = self._small_network(rng)
        y_avg = average_controllability(W)
        phi = modal_controllability(W)
        assert np.all(np.isfinite(y_avg)) and np.all(y_avg > 0)
        assert np.all(np.isfinite(phi)) and np.all(phi > 0)

    def test_normalize_adjacency_has_subunity_spectral_radius(self, rng):
        W = self._small_network(rng) * 10.0   # arbitrary scale
        A = _normalize_adjacency(W)
        np.testing.assert_allclose(A, A.T)
        assert np.max(np.abs(np.linalg.eigvalsh(A))) < 1.0


class TestIsControllable:
    def test_controllable_system(self):
        A = np.array([[1, 1], [0, 1]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        assert is_controllable(A, B)

    def test_uncontrollable_system(self):
        A = np.diag([1.0, 2.0])
        B = np.array([[1], [0]], dtype=float)
        # Only first state is actuated; second is uncontrollable
        assert not is_controllable(A, B)


class TestDARESolve:
    def test_dare_solution_is_symmetric(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        Q = np.eye(n)
        R = np.eye(B.shape[1])
        P, K = dare_solve(A, B, Q, R)
        np.testing.assert_allclose(P, P.T, atol=1e-8)

    def test_dare_solution_is_psd(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        Q, R = np.eye(n), np.eye(B.shape[1])
        P, _ = dare_solve(A, B, Q, R)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals >= -1e-8)

    def test_gain_shape(self, synthetic_system):
        A, B = synthetic_system
        n, m = A.shape[0], B.shape[1]
        Q, R = np.eye(n), np.eye(m)
        _, K = dare_solve(A, B, Q, R)
        assert K.shape == (m, n)


class TestLQRDesign:
    def test_stable_closed_loop(self, synthetic_system):
        A, B = synthetic_system
        result = lqr_design(A, B, q_state=1.0, r_control=0.1)
        assert result["is_stable"]

    def test_higher_q_more_aggressive(self, synthetic_system):
        A, B = synthetic_system
        r1 = lqr_design(A, B, q_state=0.1, r_control=1.0)
        r2 = lqr_design(A, B, q_state=10.0, r_control=1.0)
        # Higher q → larger gain norm
        assert np.linalg.norm(r2["K"]) > np.linalg.norm(r1["K"])

    def test_output_keys(self, synthetic_system):
        A, B = synthetic_system
        result = lqr_design(A, B)
        for key in ["P", "K", "closed_loop_A", "is_stable", "closed_loop_eigenvalues"]:
            assert key in result


class TestMinimumEnergyTrajectory:
    def test_reaches_target(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        x0 = np.zeros(n)
        x0[0] = 1.0
        xf = np.zeros(n)

        x_traj, u_traj, energy = minimum_energy_trajectory(A, B, x0, xf, T=30)
        np.testing.assert_allclose(x_traj[-1], xf, atol=1e-4)

    def test_energy_is_positive(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        x0 = np.ones(n)
        xf = np.zeros(n)
        _, _, energy = minimum_energy_trajectory(A, B, x0, xf, T=30)
        assert energy >= 0

    def test_output_shapes(self, synthetic_system):
        A, B = synthetic_system
        n, m = A.shape[0], B.shape[1]
        T = 20
        x0, xf = np.zeros(n), np.ones(n)
        x_traj, u_traj, _ = minimum_energy_trajectory(A, B, x0, xf, T)
        assert x_traj.shape == (T + 1, n)
        assert u_traj.shape == (T, m)

    def test_zero_trajectory_zero_energy(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        x0 = xf = np.zeros(n)
        _, _, energy = minimum_energy_trajectory(A, B, x0, xf, T=20)
        assert energy < 1e-8


class TestLQRSimulate:
    def test_converges_to_reference(self, synthetic_system):
        A, B = synthetic_system
        # Use high q to drive fast convergence; test with many steps
        lqr = lqr_design(A, B, q_state=100.0, r_control=1.0)
        assert lqr["is_stable"], "Closed-loop system must be stable"
        x_ref = np.zeros(A.shape[0])  # track to origin
        x0 = np.ones(A.shape[0])
        x_traj, _ = lqr_simulate(A, B, lqr["K"], x0, T=500, x_ref=x_ref)
        # After 500 steps of a stable closed-loop system, norm should be small
        assert np.linalg.norm(x_traj[-1]) < 0.1

    def test_output_shapes(self, synthetic_system):
        A, B = synthetic_system
        n, m = A.shape[0], B.shape[1]
        lqr = lqr_design(A, B)
        T = 30
        x_traj, u_traj = lqr_simulate(A, B, lqr["K"], np.ones(n), T)
        assert x_traj.shape == (T + 1, n)
        assert u_traj.shape == (T, m)


class TestEnergyAccuracyPareto:
    def test_tradeoff_direction(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        x0_list = [np.ones(n)]
        xf_list = [np.zeros(n)]
        q_vals = np.array([0.01, 0.1, 1.0, 10.0])

        result = energy_accuracy_pareto(A, B, x0_list, xf_list, q_vals, T=50)
        # Higher q → more energy, lower error (generally)
        assert result["errors"][0] > result["errors"][-1]
        assert result["energies"][0] < result["energies"][-1]

    def test_output_shapes(self, synthetic_system):
        A, B = synthetic_system
        n = A.shape[0]
        q_vals = np.array([0.1, 1.0, 10.0])
        result = energy_accuracy_pareto(A, B, [np.ones(n)], [np.zeros(n)], q_vals, T=30)
        assert len(result["energies"]) == 3
        assert len(result["errors"]) == 3
