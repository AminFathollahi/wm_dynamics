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
)


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
