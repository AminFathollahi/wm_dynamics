"""Tests for src/neuroai.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from neuroai import (
    centered_kernel_alignment,
    cka_timecourse,
    procrustes_align,
    noise_ceiling_rsa,
    cross_subject_rsa,
    geometry_summary,
)


class TestCenteredKernelAlignment:
    def test_identical_returns_one(self, rng):
        X = rng.standard_normal((30, 10))
        cka = centered_kernel_alignment(X, X)
        np.testing.assert_allclose(cka, 1.0, atol=1e-6)

    def test_range(self, rng):
        X = rng.standard_normal((30, 8))
        Y = rng.standard_normal((30, 6))
        cka = centered_kernel_alignment(X, Y)
        assert 0.0 <= cka <= 1.0 + 1e-8

    def test_orthogonal_transform_invariant(self, rng):
        X = rng.standard_normal((40, 6))
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        cka_orig = centered_kernel_alignment(X, X)
        cka_rot = centered_kernel_alignment(X, X @ Q)
        np.testing.assert_allclose(cka_orig, cka_rot, atol=1e-8)

    def test_symmetric(self, rng):
        X = rng.standard_normal((25, 5))
        Y = rng.standard_normal((25, 4))
        np.testing.assert_allclose(
            centered_kernel_alignment(X, Y),
            centered_kernel_alignment(Y, X),
            atol=1e-8,
        )

    def test_random_vs_identical_ordering(self, rng):
        X = rng.standard_normal((30, 6))
        Y = rng.standard_normal((30, 6))
        cka_same = centered_kernel_alignment(X, X)
        cka_diff = centered_kernel_alignment(X, Y)
        assert cka_same > cka_diff


class TestCKATimecourse:
    def test_output_shape(self, rng):
        N, T, d = 20, 15, 4
        Z = rng.standard_normal((N, T, d))
        Y = rng.standard_normal((N, T, d))
        cka_t = cka_timecourse(Z, Y)
        assert cka_t.shape == (T,)

    def test_identical_trajectories_all_one(self, rng):
        N, T, d = 20, 10, 4
        Z = rng.standard_normal((N, T, d))
        cka_t = cka_timecourse(Z, Z)
        np.testing.assert_allclose(cka_t, np.ones(T), atol=1e-6)


class TestProcrustesAlign:
    def test_output_keys(self, rng):
        X = rng.standard_normal((20, 5))
        Y = rng.standard_normal((20, 5))
        result = procrustes_align(X, Y)
        assert "R" in result
        assert "Y_aligned" in result
        assert "disparity" in result

    def test_R_is_orthogonal(self, rng):
        X = rng.standard_normal((20, 4))
        Y = rng.standard_normal((20, 4))
        R = procrustes_align(X, Y)["R"]
        np.testing.assert_allclose(R @ R.T, np.eye(4), atol=1e-10)

    def test_identical_inputs_zero_disparity(self, rng):
        X = rng.standard_normal((20, 4))
        result = procrustes_align(X, X)
        np.testing.assert_allclose(result["disparity"], 0.0, atol=1e-8)

    def test_rotation_recovered(self, rng):
        X = rng.standard_normal((30, 4))
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        Y = X @ Q
        result = procrustes_align(X, Y)
        # After alignment, Y_aligned should match X
        np.testing.assert_allclose(result["Y_aligned"], X, atol=1e-8)

    def test_disparity_in_range(self, rng):
        X = rng.standard_normal((20, 4))
        Y = rng.standard_normal((20, 4))
        d = procrustes_align(X, Y)["disparity"]
        assert 0.0 <= d


class TestNoiseCeilingRSA:
    def test_output_bounds_valid(self, rng):
        N_subj, n_items = 5, 12
        rdms = np.abs(rng.standard_normal((N_subj, n_items, n_items)))
        for i in range(N_subj):
            rdms[i] = (rdms[i] + rdms[i].T) / 2
            np.fill_diagonal(rdms[i], 0.0)
        lower, upper = noise_ceiling_rsa(rdms)
        assert lower <= upper + 1e-8
        assert -1.0 <= lower <= 1.0
        assert -1.0 <= upper <= 1.0

    def test_identical_subjects_ceiling_near_one(self, rng):
        X = rng.standard_normal((10, 6))
        from geometry import representational_dissimilarity_matrix
        rdm = representational_dissimilarity_matrix(X)
        rdms = np.stack([rdm] * 5)
        lower, upper = noise_ceiling_rsa(rdms)
        np.testing.assert_allclose(upper, 1.0, atol=1e-6)


class TestCrossSubjectRSA:
    def test_output_keys(self, rng):
        from geometry import representational_dissimilarity_matrix
        subjects = ["s1", "s2", "s3"]
        rdms = {s: representational_dissimilarity_matrix(rng.standard_normal((10, 6)))
                for s in subjects}
        result = cross_subject_rsa(rdms)
        assert "pairwise_r" in result
        assert "mean_pairwise_r" in result

    def test_pairwise_diagonal_one(self, rng):
        from geometry import representational_dissimilarity_matrix
        subjects = ["a", "b", "c"]
        rdms = {s: representational_dissimilarity_matrix(rng.standard_normal((8, 5)))
                for s in subjects}
        result = cross_subject_rsa(rdms)
        np.testing.assert_allclose(np.diag(result["pairwise_r"]), np.ones(3), atol=1e-8)

    def test_with_model_rdm(self, rng):
        from geometry import representational_dissimilarity_matrix
        subjects = ["a", "b"]
        rdms = {s: representational_dissimilarity_matrix(rng.standard_normal((8, 5)))
                for s in subjects}
        model_rdm = representational_dissimilarity_matrix(rng.standard_normal((8, 3)))
        result = cross_subject_rsa(rdms, model_rdm=model_rdm)
        assert result["model_r"] is not None
        assert len(result["model_r"]) == 2
        assert isinstance(result["mean_model_r"], float)


class TestGeometrySummary:
    def test_output_keys(self, rng):
        N, d = 20, 4
        Z_neural = rng.standard_normal((N, d))
        Z_model = rng.standard_normal((N, d))
        labels = np.array([0] * 10 + [1] * 10)
        result = geometry_summary(Z_neural, Z_model, labels)
        for key in ("cka", "rsa_r", "neural_rdm", "model_rdm"):
            assert key in result

    def test_cka_range(self, rng):
        Z_neural = rng.standard_normal((20, 4))
        Z_model = rng.standard_normal((20, 4))
        labels = np.array([0] * 10 + [1] * 10)
        result = geometry_summary(Z_neural, Z_model, labels)
        assert 0.0 <= result["cka"] <= 1.0 + 1e-8
