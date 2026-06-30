"""Tests for src/geometry.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from geometry import (
    pca_decompose,
    participation_ratio,
    pca_participation_ratio,
    principal_angles,
    latent_trajectories,
    time_resolved_principal_angles,
    time_resolved_pr,
    electrode_capacity_curve,
    representational_dissimilarity_matrix,
    rsa_compare,
    cross_temporal_generalization,
    subspace_overlap,
    geometric_drift,
)


class TestPCADecompose:
    def test_output_shapes(self, rng):
        N, D, k = 100, 20, 5
        X = rng.standard_normal((N, D))
        scores, comps, var = pca_decompose(X, k)
        assert scores.shape == (N, k)
        assert comps.shape == (D, k)
        assert len(var) == k

    def test_variance_sums_leq_one(self, rng):
        X = rng.standard_normal((80, 15))
        _, _, var = pca_decompose(X, 5)
        assert var.sum() <= 1.0 + 1e-8
        assert np.all(var >= 0)

    def test_variance_monotone_decreasing(self, rng):
        X = rng.standard_normal((80, 15))
        _, _, var = pca_decompose(X, 8)
        assert np.all(np.diff(var) <= 1e-10)

    def test_components_orthonormal(self, rng):
        X = rng.standard_normal((100, 20))
        _, comps, _ = pca_decompose(X, 5)
        # Columns should be orthonormal
        np.testing.assert_allclose(comps.T @ comps, np.eye(5), atol=1e-10)

    def test_reconstruction_quality(self, rng):
        X = rng.standard_normal((100, 5))  # only 5 true dims
        scores, comps, _ = pca_decompose(X, 5)
        Xc = X - X.mean(0)
        X_rec = scores @ comps.T
        np.testing.assert_allclose(X_rec, Xc, atol=1e-8)


class TestParticipationRatio:
    def test_uniform_spectrum_equals_ndim(self):
        lam = np.ones(10)
        pr = participation_ratio(lam)
        np.testing.assert_allclose(pr, 10.0, rtol=1e-10)

    def test_single_dominant_equals_one(self):
        lam = np.array([100.0, 1e-8, 1e-8, 1e-8])
        pr = participation_ratio(lam)
        np.testing.assert_allclose(pr, 1.0, atol=1e-4)

    def test_range(self, rng):
        lam = np.abs(rng.standard_normal(12)) ** 2
        pr = participation_ratio(lam)
        assert 1.0 <= pr <= len(lam)

    def test_ignores_zeros(self):
        lam = np.array([1.0, 1.0, 0.0, 0.0])
        pr = participation_ratio(lam)
        np.testing.assert_allclose(pr, 2.0, rtol=1e-10)

    def test_monotone_with_uniformity(self):
        # More uniform spectrum → higher PR
        concentrated = np.array([100.0, 1.0, 1.0, 1.0])
        uniform = np.array([25.25, 25.25, 25.25, 25.25])
        assert participation_ratio(concentrated) < participation_ratio(uniform)


class TestPrincipalAngles:
    def test_identical_subspaces_zero_angle(self, rng):
        A = rng.standard_normal((20, 3))
        angles = principal_angles(A, A)
        # arccos(1) has ~1e-6 numerical error near the boundary
        np.testing.assert_allclose(np.degrees(angles), np.zeros(3), atol=1e-4)

    def test_orthogonal_subspaces_90_degrees(self):
        A = np.eye(10)[:, :3]
        B = np.eye(10)[:, 5:8]
        angles = principal_angles(A, B)
        np.testing.assert_allclose(np.degrees(angles), 90 * np.ones(3), atol=1e-8)

    def test_angles_in_valid_range(self, rng):
        A = rng.standard_normal((20, 4))
        B = rng.standard_normal((20, 4))
        angles = principal_angles(A, B)
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi / 2 + 1e-10)

    def test_output_length_is_min_k_m(self, rng):
        A = rng.standard_normal((20, 4))
        B = rng.standard_normal((20, 3))
        angles = principal_angles(A, B)
        assert len(angles) == 3

    def test_angles_sorted_ascending(self, rng):
        A = rng.standard_normal((20, 4))
        B = rng.standard_normal((20, 4))
        angles = principal_angles(A, B)
        assert np.all(np.diff(angles) >= -1e-10)


class TestLatentTrajectories:
    def test_output_shapes(self, synthetic_epochs):
        epochs, times, task_id, tgt_id = synthetic_epochs
        Z, comps, var = latent_trajectories(epochs, n_components=6)
        N, T, C = epochs.shape
        assert Z.shape == (N, T, 6)
        assert comps.shape == (C, 6)

    def test_var_ratio_valid(self, synthetic_epochs):
        epochs, times, task_id, tgt_id = synthetic_epochs
        _, _, var = latent_trajectories(epochs, n_components=5)
        assert var.sum() <= 1.0 + 1e-8
        assert np.all(var >= 0)


class TestTimeResolvedPrincipalAngles:
    def test_output_shape(self, synthetic_epochs):
        epochs, times, task_id, tgt_id = synthetic_epochs
        Z, _, _ = latent_trajectories(epochs, 6)
        mask_a = task_id == 0
        mask_b = task_id == 2
        theta = time_resolved_principal_angles(Z, mask_a, mask_b, n_dims=3)
        assert theta.shape == (epochs.shape[1], 3)

    def test_values_in_range(self, synthetic_epochs):
        epochs, times, task_id, tgt_id = synthetic_epochs
        Z, _, _ = latent_trajectories(epochs, 4)
        mask_a, mask_b = task_id == 0, task_id == 1
        theta = time_resolved_principal_angles(Z, mask_a, mask_b, n_dims=2)
        valid = ~np.isnan(theta)
        assert np.all(theta[valid] >= 0)
        assert np.all(theta[valid] <= np.pi / 2 + 1e-10)


class TestElectrodeCapacityCurve:
    def test_output_shapes(self, synthetic_epochs, rng):
        epochs, times, task_id, tgt_id = synthetic_epochs
        result = electrode_capacity_curve(
            epochs, task_id,
            channel_counts=[2, 4, 8],
            n_bootstrap=5,
            rng=rng,
        )
        assert len(result["n_channels"]) == 3
        assert len(result["pr_contrast_mean"]) == 3
        assert len(result["pr_contrast_std"]) == 3


class TestRepresentationalDissimilarityMatrix:
    def test_shape(self, rng):
        X = rng.standard_normal((20, 10))
        rdm = representational_dissimilarity_matrix(X)
        assert rdm.shape == (20, 20)

    def test_zero_diagonal(self, rng):
        X = rng.standard_normal((15, 8))
        rdm = representational_dissimilarity_matrix(X)
        np.testing.assert_allclose(np.diag(rdm), np.zeros(15), atol=1e-8)

    def test_symmetric(self, rng):
        X = rng.standard_normal((12, 6))
        rdm = representational_dissimilarity_matrix(X)
        np.testing.assert_allclose(rdm, rdm.T, atol=1e-10)

    def test_values_in_range_correlation(self, rng):
        X = rng.standard_normal((10, 8))
        rdm = representational_dissimilarity_matrix(X, metric="correlation")
        assert np.all(rdm >= -1e-8)
        assert np.all(rdm <= 2.0 + 1e-8)

    def test_euclidean_metric(self, rng):
        X = rng.standard_normal((8, 5))
        rdm = representational_dissimilarity_matrix(X, metric="euclidean")
        assert np.all(rdm >= 0)

    def test_identical_rows_zero_distance(self):
        X = np.ones((5, 8))
        rdm = representational_dissimilarity_matrix(X, metric="euclidean")
        np.testing.assert_allclose(rdm, np.zeros((5, 5)), atol=1e-10)


class TestRSACompare:
    def test_identical_rdms_returns_one(self, rng):
        X = rng.standard_normal((15, 8))
        rdm = representational_dissimilarity_matrix(X)
        r = rsa_compare(rdm, rdm)
        np.testing.assert_allclose(r, 1.0, atol=1e-8)

    def test_returns_scalar(self, rng):
        X = rng.standard_normal((10, 6))
        rdm1 = representational_dissimilarity_matrix(X)
        rdm2 = representational_dissimilarity_matrix(rng.standard_normal((10, 4)))
        r = rsa_compare(rdm1, rdm2)
        assert isinstance(r, float)

    def test_range(self, rng):
        X = rng.standard_normal((12, 6))
        Y = rng.standard_normal((12, 4))
        rdm1 = representational_dissimilarity_matrix(X)
        rdm2 = representational_dissimilarity_matrix(Y)
        r = rsa_compare(rdm1, rdm2)
        assert -1.0 <= r <= 1.0

    def test_antisymmetric_rdm_negative(self, rng):
        X = rng.standard_normal((10, 5))
        rdm1 = representational_dissimilarity_matrix(X)
        rdm2 = 1.0 - rdm1  # inverted dissimilarity
        np.fill_diagonal(rdm2, 0.0)
        r = rsa_compare(rdm1, rdm2)
        assert r < 0


class TestCrossTemporalGeneralization:
    def test_output_shape(self, synthetic_epochs, rng):
        epochs, times, task_id, tgt_id = synthetic_epochs
        Z, _, _ = latent_trajectories(epochs, 4)
        binary_labels = (task_id == 2).astype(int)
        # Use small T slice for speed
        auc = cross_temporal_generalization(Z[:, :10, :], binary_labels,
                                            n_splits=3, rng=rng)
        assert auc.shape == (10, 10)

    def test_diagonal_near_chance_for_random(self, rng):
        N, T, k = 40, 10, 4
        Z = rng.standard_normal((N, T, k))
        labels = np.array([0] * 20 + [1] * 20)
        auc = cross_temporal_generalization(Z, labels, n_splits=3, rng=rng)
        # With random data, mean diagonal AUC should be near 0.5
        diag = np.diag(auc)
        assert 0.3 < diag.mean() < 0.7


class TestSubspaceOverlap:
    def test_identical_subspaces_unity(self, rng):
        A = rng.standard_normal((20, 3))
        overlap = subspace_overlap(A, A)
        np.testing.assert_allclose(overlap, 1.0, atol=1e-8)

    def test_orthogonal_subspaces_zero(self):
        A = np.eye(10)[:, :3]
        B = np.eye(10)[:, 5:8]
        overlap = subspace_overlap(A, B)
        np.testing.assert_allclose(overlap, 0.0, atol=1e-8)

    def test_range(self, rng):
        A = rng.standard_normal((20, 4))
        B = rng.standard_normal((20, 4))
        overlap = subspace_overlap(A, B)
        assert 0.0 <= overlap <= 1.0 + 1e-8


class TestGeometricDrift:
    def test_output_shape(self, synthetic_epochs):
        epochs, times, task_id, _ = synthetic_epochs
        Z, _, _ = latent_trajectories(epochs, 6)
        drift = geometric_drift(Z, task_id, times)
        assert drift.shape == (len(epochs),)

    def test_nonnegative(self, synthetic_epochs):
        epochs, times, task_id, _ = synthetic_epochs
        Z, _, _ = latent_trajectories(epochs, 6)
        drift = geometric_drift(Z, task_id, times)
        assert np.all(drift >= 0)

    def test_centroid_trial_has_low_drift(self, rng):
        N, T, k = 30, 40, 4
        times = np.linspace(-0.2, 1.5, T)
        task_id = np.array([0] * 15 + [2] * 15)
        Z = rng.standard_normal((N, T, k))
        # Replace first trial of each group with the group mean
        Z[0] = Z[:15].mean(axis=0)
        Z[15] = Z[15:].mean(axis=0)
        drift = geometric_drift(Z, task_id, times)
        # The centroid trial (index 0 and 15) should have near-zero drift
        assert drift[0] < drift[1:15].mean()
        assert drift[15] < drift[16:].mean()
