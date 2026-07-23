"""Tests for src/normative.py (STEP C normative demo)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from normative import optimize_maintenance_operator, embed_maintenance_operator, _loss


class TestOptimizeMaintenanceOperator:
    def test_rotation_beats_static_memorandum(self):
        # At a horizon short enough that interference is unambiguous, the
        # optimum should prefer SOME rotation over none (theta=0 is the
        # unique global maximum of self-overlap, i.e. worst-case interference).
        res = optimize_maintenance_operator(T=20)
        assert res["theta"] > 0.05
        static_loss = _loss(np.array([1e-6, res["lambda_c"]]), T=20, energy_eps=0.01)
        assert res["loss"] < static_loss

    def test_context_axis_is_retained(self):
        res = optimize_maintenance_operator(T=20)
        assert res["lambda_c"] > 0.9


class TestEmbedMaintenanceOperator:
    def _setup(self, k=5):
        rng = np.random.default_rng(1)
        c_axis = rng.standard_normal(k)
        x0_mem = rng.standard_normal(k)
        return c_axis, x0_mem, k

    def test_context_axis_is_an_eigenvector(self):
        c_axis, x0_mem, k = self._setup()
        M = embed_maintenance_operator(theta=0.3, lambda_c=0.95, c_axis=c_axis,
                                       x0_mem=x0_mem, k=k, lambda_rest=0.2)
        b1 = c_axis / np.linalg.norm(c_axis)
        np.testing.assert_allclose(M @ b1, 0.95 * b1, atol=1e-8)

    def test_memorandum_subspace_preserves_norm(self):
        c_axis, x0_mem, k = self._setup()
        M = embed_maintenance_operator(theta=0.4, lambda_c=0.9, c_axis=c_axis,
                                       x0_mem=x0_mem, k=k, lambda_rest=0.1)
        b1 = c_axis / np.linalg.norm(c_axis)
        resid = x0_mem - (x0_mem @ b1) * b1
        b2 = resid / np.linalg.norm(resid)
        assert np.linalg.norm(M @ b2) == pytest.approx(1.0, abs=1e-6)

    def test_planted_theta_lambda_c_recovered_as_eigenstructure(self):
        # The literal spec requirement: an optimizer solution, once embedded,
        # has the PLANTED (theta, lambda_c) as its actual eigenstructure --
        # one real eigenvalue = lambda_c (context), one conjugate pair with
        # |lambda|=1 and angle=theta (the rotating memorandum).
        theta_true, lambda_true = 0.35, 0.92
        c_axis, x0_mem, k = self._setup(k=6)
        M = embed_maintenance_operator(theta=theta_true, lambda_c=lambda_true,
                                       c_axis=c_axis, x0_mem=x0_mem, k=k, lambda_rest=0.15)
        eigs = np.linalg.eigvals(M)
        real_eigs = eigs[np.abs(eigs.imag) < 1e-8].real
        complex_eigs = eigs[np.abs(eigs.imag) >= 1e-8]

        assert np.any(np.isclose(real_eigs, lambda_true, atol=1e-6))
        assert len(complex_eigs) == 2
        np.testing.assert_allclose(np.abs(complex_eigs), 1.0, atol=1e-6)
        angles = np.sort(np.abs(np.angle(complex_eigs)))
        assert angles[0] == pytest.approx(theta_true, abs=1e-6)

    def test_rejects_k_below_three(self):
        with pytest.raises(ValueError):
            embed_maintenance_operator(theta=0.3, lambda_c=0.9, c_axis=np.array([1.0, 0.0]),
                                       x0_mem=np.array([0.0, 1.0]), k=2)
