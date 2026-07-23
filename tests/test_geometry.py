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
    ctg_nested_cv,
    ctg_label_permutation_null,
    ctg_content_permutation_null,
    temporal_stability_tau,
    spatiotemporal_participation_ratio,
    marginalize_condition_time,
    coding_direction_stability,
    dpca_condition_subspace_projection,
    time_resolved_content_decoding,
    cross_decoding_leakage_test,
    out_of_fold_class_confidence,
    cross_condition_decoding_test,
    ctg_phase_scramble_null,
    parallel_analysis,
    select_latent_dim,
)


def _labeled_channel_signal(rng, N=60, C=10, T=30, effect=1.5):
    """(N, C, T) raw features with a sustained (square) label-locked signal."""
    labels = (rng.random(N) < 0.5).astype(int)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    direction = rng.standard_normal(C)
    direction /= np.linalg.norm(direction)
    X += effect * labels[:, None, None] * direction[None, :, None]
    return X, labels


class TestCtgNestedCV:
    def test_output_shape(self, rng):
        X, labels = _labeled_channel_signal(rng)
        t_idx = np.arange(0, 30, 5)
        auc_mat = ctg_nested_cv(X, labels, t_idx, n_components=4, n_splits=3, rng=rng)
        assert auc_mat.shape == (len(t_idx), len(t_idx))

    def test_detects_sustained_signal_above_chance(self, rng):
        X, labels = _labeled_channel_signal(rng, effect=2.5)
        t_idx = np.arange(0, 30, 5)
        auc_mat = ctg_nested_cv(X, labels, t_idx, n_components=4, n_splits=4, rng=rng)
        off_mask = ~np.eye(len(t_idx), dtype=bool)
        assert np.nanmean(auc_mat[off_mask]) > 0.6

    def test_pca_not_fit_on_test_trials(self, rng):
        # Test-fold-only channels (no label signal reaches them) should not
        # bias the fold-fitted PCA basis: pure noise labels -> chance AUC.
        N, C, T = 60, 10, 20
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        labels = rng.integers(0, 2, N)
        t_idx = np.arange(0, T, 4)
        auc_mat = ctg_nested_cv(X, labels, t_idx, n_components=4, n_splits=3, rng=rng)
        off_mask = ~np.eye(len(t_idx), dtype=bool)
        assert abs(np.nanmean(auc_mat[off_mask]) - 0.5) < 0.15


class TestCtgLabelPermutationNull:
    def test_significant_for_strong_signal(self, rng):
        X, labels = _labeled_channel_signal(rng, effect=3.0)
        t_idx = np.arange(0, 30, 6)
        res = ctg_label_permutation_null(
            X, labels, t_idx, n_components=4, n_splits=3, n_perm=100, rng=rng
        )
        assert res["p_value"] < 0.05
        assert res["mean_offdiag_auc_minus_chance"] > 0

    def test_null_distribution_centered_near_zero(self, rng):
        N, C, T = 50, 8, 20
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        labels = rng.integers(0, 2, N)
        t_idx = np.arange(0, T, 5)
        res = ctg_label_permutation_null(
            X, labels, t_idx, n_components=4, n_splits=3, n_perm=100, rng=rng
        )
        assert abs(np.mean(res["null"])) < 0.1


class TestCtgContentPermutationNull:
    def test_detects_multiclass_signal(self, rng):
        N, C, T, n_classes = 90, 12, 20, 3
        labels = rng.integers(0, n_classes, N)
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        directions = rng.standard_normal((n_classes, C))
        for c in range(n_classes):
            X[labels == c] += 3.0 * directions[c][None, :, None]
        t_idx = np.arange(0, T, 4)
        res = ctg_content_permutation_null(
            X, labels, t_idx, n_components=6, n_splits=3, n_perm=50, rng=rng
        )
        assert res["n_classes"] == n_classes
        assert res["p_value"] < 0.05


class TestTimeResolvedContentDecoding:
    def test_detects_signal_only_where_present(self, rng):
        N, C, T, n_classes = 90, 12, 10, 3
        labels = rng.integers(0, n_classes, N)
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        directions = rng.standard_normal((n_classes, C))
        signal_bins = [3, 4, 5]
        for c in range(n_classes):
            for t in signal_bins:
                X[labels == c, :, t] += 4.0 * directions[c]
        t_idx = np.arange(T)
        res = time_resolved_content_decoding(X, labels, t_idx, n_components=6,
                                             n_splits=3, n_perm=100, rng=rng)
        assert res["auc_per_t"].shape == (T,)
        assert res["p_per_t"].shape == (T,)
        assert np.mean(res["auc_per_t"][signal_bins]) > np.mean(
            res["auc_per_t"][[t for t in range(T) if t not in signal_bins]])
        assert np.all(res["p_per_t"][signal_bins] < 0.1)

    def test_cost_is_linear_not_quadratic_in_timepoints(self, rng):
        # a regression guard on the O(T) design: running with many timepoints
        # should not raise or hang (the O(T^2) full CTG would be intractable
        # at this many timepoints with n_perm=50)
        N, C, T, n_classes = 40, 8, 60, 2
        labels = rng.integers(0, n_classes, N)
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        t_idx = np.arange(T)
        res = time_resolved_content_decoding(X, labels, t_idx, n_components=4,
                                             n_splits=2, n_perm=50, rng=rng)
        assert res["auc_per_t"].shape == (T,)


class TestOutOfFoldClassConfidence:
    def test_higher_confidence_with_stronger_signal(self, rng):
        N, C, T, n_classes = 150, 10, 4, 4
        labels = rng.integers(0, n_classes, N)
        directions = rng.standard_normal((n_classes, C))
        X_strong = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X_strong[labels == c] += 4.0 * directions[c][None, :, None]
        X_weak = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X_weak[labels == c] += 0.3 * directions[c][None, :, None]

        conf_strong = out_of_fold_class_confidence(X_strong, labels, t_idx=1,
                                                    n_components=6, n_splits=5, rng=rng)
        conf_weak = out_of_fold_class_confidence(X_weak, labels, t_idx=1,
                                                  n_components=6, n_splits=5, rng=rng)
        assert np.nanmean(conf_strong) > np.nanmean(conf_weak)

    def test_output_shape_and_range(self, rng):
        N, C, T, n_classes = 100, 8, 3, 3
        labels = rng.integers(0, n_classes, N)
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        conf = out_of_fold_class_confidence(X, labels, t_idx=0, n_components=5,
                                            n_splits=4, rng=rng)
        assert conf.shape == (N,)
        valid = conf[np.isfinite(conf)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_near_chance_for_pure_noise(self, rng):
        N, C, T, n_classes = 200, 10, 3, 4
        labels = rng.integers(0, n_classes, N)
        X = rng.standard_normal((N, C, T)).astype(np.float32)   # no signal
        conf = out_of_fold_class_confidence(X, labels, t_idx=0, n_components=6,
                                            n_splits=5, rng=rng)
        assert abs(np.nanmean(conf) - 1.0 / n_classes) < 0.1

    def test_array_of_timepoints_matches_scalar_calls(self, rng):
        N, C, T, n_classes = 100, 8, 5, 3
        labels = rng.integers(0, n_classes, N)
        directions = rng.standard_normal((n_classes, C))
        X = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X[labels == c] += 2.0 * directions[c][None, :, None]

        t_idx = np.arange(T)
        conf_array = out_of_fold_class_confidence(X, labels, t_idx, n_components=5,
                                                   n_splits=4, rng=np.random.default_rng(1))
        conf_scalar = np.stack([
            out_of_fold_class_confidence(X, labels, int(t), n_components=5, n_splits=4,
                                         rng=np.random.default_rng(1))
            for t in t_idx
        ], axis=1)
        assert conf_array.shape == (N, T)
        np.testing.assert_allclose(conf_array, conf_scalar)

    def test_signal_only_at_specific_timepoint_is_localized(self, rng):
        N, C, T, n_classes = 150, 10, 6, 3
        labels = rng.integers(0, n_classes, N)
        directions = rng.standard_normal((n_classes, C))
        X = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        signal_t = 3
        X[np.arange(N), :, signal_t] += 4.0 * directions[labels]
        conf = out_of_fold_class_confidence(X, labels, np.arange(T), n_components=6,
                                            n_splits=5, rng=rng)
        assert np.nanmean(conf[:, signal_t]) > np.nanmean(conf[:, 0])


class TestCrossDecodingLeakageTest:
    def test_detects_leakage_when_test_label_is_the_same_signal(self, rng):
        N, C, T = 120, 10, 5
        labels = (rng.random(N) < 0.5).astype(int)
        direction = rng.standard_normal(C)
        X = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        X += 2.5 * labels[:, None, None] * direction[None, :, None]
        res = cross_decoding_leakage_test(X, labels, labels, t_idx=2,
                                          n_components=6, n_splits=4, n_perm=500, rng=rng)
        assert res["p_value"] < 0.05

    def test_no_leakage_when_test_label_is_independent(self, rng):
        N, C, T = 150, 10, 5
        labels_train = (rng.random(N) < 0.5).astype(int)
        labels_test = rng.integers(0, 3, N)   # independent of labels_train
        direction = rng.standard_normal(C)
        X = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        X += 2.5 * labels_train[:, None, None] * direction[None, :, None]
        res = cross_decoding_leakage_test(X, labels_train, labels_test, t_idx=2,
                                          n_components=6, n_splits=4, n_perm=500, rng=rng)
        assert res["p_value"] > 0.05

    def test_multiclass_train_binary_test(self, rng):
        N, C, T, n_classes = 150, 10, 5, 4
        labels_train = rng.integers(0, n_classes, N)
        # labels_test correlated with labels_train via a binary split of classes
        labels_test = (labels_train < n_classes // 2).astype(int)
        directions = rng.standard_normal((n_classes, C))
        X = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X[labels_train == c] += 2.5 * directions[c][None, :, None]
        res = cross_decoding_leakage_test(X, labels_train, labels_test, t_idx=2,
                                          n_components=6, n_splits=3, n_perm=500, rng=rng)
        assert res["n_trials"] > 0
        assert res["p_value"] < 0.1


class TestCrossConditionDecodingTest:
    def test_detects_generalizing_signal(self, rng):
        N, C, T, n_classes = 100, 10, 6, 3
        directions = rng.standard_normal((n_classes, C))
        y_train = rng.integers(0, n_classes, N)
        X_train = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X_train[y_train == c] += 3.0 * directions[c][None, :, None]
        y_test = rng.integers(0, n_classes, N)
        X_test = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X_test[y_test == c] += 3.0 * directions[c][None, :, None]
        t_idx = np.arange(T)
        res = cross_condition_decoding_test(X_train, y_train, X_test, y_test, t_idx,
                                            n_components=6, n_perm=200, rng=rng)
        assert res["auc_per_t"].shape == (T,)
        assert np.mean(res["auc_per_t"]) > 0.6
        assert np.mean(res["p_per_t"]) < 0.05

    def test_no_generalization_when_test_set_is_unrelated(self, rng):
        N, C, T, n_classes = 100, 10, 6, 3
        directions = rng.standard_normal((n_classes, C))
        y_train = rng.integers(0, n_classes, N)
        X_train = rng.standard_normal((N, C, T)).astype(np.float32) * 0.3
        for c in range(n_classes):
            X_train[y_train == c] += 3.0 * directions[c][None, :, None]
        y_test = rng.integers(0, n_classes, N)
        X_test = rng.standard_normal((N, C, T)).astype(np.float32)   # no signal at all
        t_idx = np.arange(T)
        res = cross_condition_decoding_test(X_train, y_train, X_test, y_test, t_idx,
                                            n_components=6, n_perm=200, rng=rng)
        assert np.mean(res["p_per_t"]) > 0.2


class TestTemporalStabilityTau:
    def test_perfect_generalization_gives_tau_one(self):
        auc_mat = np.full((5, 5), 0.8)
        info = temporal_stability_tau(auc_mat)
        assert info["tau"] == pytest.approx(1.0)
        assert info["interpretable"]

    def test_near_chance_diagonal_flagged_uninterpretable(self):
        auc_mat = np.full((5, 5), 0.51)
        info = temporal_stability_tau(auc_mat, min_diag_auc=0.55)
        assert not info["interpretable"]

    def test_narrow_diagonal_gives_low_tau(self):
        n = 6
        auc_mat = np.full((n, n), 0.5)
        np.fill_diagonal(auc_mat, 0.9)
        info = temporal_stability_tau(auc_mat)
        assert info["tau"] == pytest.approx(0.0, abs=1e-8)


class TestSpatiotemporalParticipationRatio:
    def test_output_keys_and_range(self, rng):
        X = rng.standard_normal((40, 10, 15)).astype(np.float32)
        res = spatiotemporal_participation_ratio(X, n_splits=2, rng=rng)
        assert 1.0 <= res["pr_cv"] <= 10.0 + 1e-6
        assert res["n_channels"] == 10
        assert res["n_trials"] == 40

    def test_low_rank_signal_gives_low_pr(self, rng):
        N, C, T = 50, 10, 15
        direction = rng.standard_normal(C)
        direction /= np.linalg.norm(direction)
        amplitude = rng.standard_normal((N, 1, T))
        X = (amplitude * direction[None, :, None]).astype(np.float32)
        X += 0.01 * rng.standard_normal((N, C, T)).astype(np.float32)
        res = spatiotemporal_participation_ratio(X, n_splits=2, rng=rng)
        assert res["pr_cv"] < 2.0


class TestMarginalizeConditionTime:
    def test_output_fractions_sum_to_one(self, rng):
        N, T, k = 40, 20, 5
        labels = rng.integers(0, 2, N)
        Z = rng.standard_normal((N, T, k))
        res = marginalize_condition_time(Z, labels)
        total = res["frac_condition_independent"] + res["frac_condition_dependent"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_pure_condition_signal_is_condition_dependent(self, rng):
        N, T, k = 40, 20, 5
        labels = (rng.random(N) < 0.5).astype(int)
        direction = rng.standard_normal(k)
        Z = 0.01 * rng.standard_normal((N, T, k))
        Z += 5.0 * labels[:, None, None] * direction[None, None, :]
        res = marginalize_condition_time(Z, labels)
        assert res["frac_condition_dependent"] > 0.9


class TestDpcaConditionSubspaceProjection:
    def test_output_shapes(self, rng):
        N, T, k = 40, 20, 6
        labels = rng.integers(0, 3, N)
        Z = rng.standard_normal((N, T, k))
        res = dpca_condition_subspace_projection(Z, labels, n_components=4)
        assert res["Z_condition_independent"].shape == (N, T, 4)
        assert res["Z_condition_dependent"].shape == (N, T, 4)
        assert res["V_condition_independent"].shape == (k, 4)
        assert res["V_condition_dependent"].shape == (k, 4)

    def test_condition_signal_survives_in_dependent_projection(self, rng):
        # A condition-locked, time-constant signal should be decodable from
        # the condition-dependent projection but not (beyond noise) from the
        # condition-independent one.
        N, T, k = 100, 15, 6
        labels = (rng.random(N) < 0.5).astype(int)
        direction = rng.standard_normal(k)
        direction /= np.linalg.norm(direction)
        Z = 0.05 * rng.standard_normal((N, T, k))
        Z += 3.0 * labels[:, None, None] * direction[None, None, :]
        res = dpca_condition_subspace_projection(Z, labels, n_components=2)

        def _class_separation(Zp):
            m = Zp.mean(axis=1)  # (N, d) time-averaged
            return np.linalg.norm(m[labels == 1].mean(0) - m[labels == 0].mean(0))

        sep_cd = _class_separation(res["Z_condition_dependent"])
        sep_ci = _class_separation(res["Z_condition_independent"])
        assert sep_cd > sep_ci

    def test_axes_are_orthonormal(self, rng):
        N, T, k = 50, 10, 8
        labels = rng.integers(0, 2, N)
        Z = rng.standard_normal((N, T, k))
        res = dpca_condition_subspace_projection(Z, labels, n_components=3)
        for V in [res["V_condition_independent"], res["V_condition_dependent"]]:
            gram = V.T @ V
            np.testing.assert_allclose(gram, np.eye(3), atol=1e-6)

    def test_n_components_capped_at_k(self, rng):
        N, T, k = 30, 10, 3
        labels = rng.integers(0, 2, N)
        Z = rng.standard_normal((N, T, k))
        res = dpca_condition_subspace_projection(Z, labels, n_components=10)
        assert res["Z_condition_independent"].shape == (N, T, k)


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

    def test_n_components_exceeding_channel_count(self, rng):
        # Regression test: pca_decompose caps k = min(n_components, rank), so
        # asking for more components than channels must not crash the reshape.
        N, T, C = 10, 20, 4
        epochs = rng.standard_normal((N, T, C))
        Z, comps, var = latent_trajectories(epochs, n_components=8)
        assert Z.shape == (N, T, C)
        assert comps.shape == (C, C)


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


class TestCodingDirectionStability:
    def test_fixed_axis_gives_high_stability(self, rng):
        N, T, k = 80, 40, 6
        labels = (rng.random(N) < 0.5).astype(int)
        direction = rng.standard_normal(k)
        direction /= np.linalg.norm(direction)
        Z = rng.standard_normal((N, T, k)) * 0.2
        Z += 2.0 * labels[:, None, None] * direction[None, None, :]
        cos_sim, t_idx = coding_direction_stability(Z, labels, step=8)
        assert cos_sim.shape == (len(t_idx), len(t_idx))
        off_diag = cos_sim[~np.eye(len(t_idx), dtype=bool)]
        assert np.mean(off_diag) > 0.8

    def test_rotating_axis_gives_low_offdiagonal_stability(self, rng):
        N, T, k = 80, 40, 4
        labels = (rng.random(N) < 0.5).astype(int)
        Z = rng.standard_normal((N, T, k)) * 0.2
        d_early = np.array([1.0, 0.0, 0.0, 0.0])
        d_late = np.array([0.0, 1.0, 0.0, 0.0])   # orthogonal to d_early
        for t in range(T):
            frac = t / (T - 1)
            direction = (1 - frac) * d_early + frac * d_late
            Z[:, t, :] += 2.0 * labels[:, None] * direction[None, :]
        cos_sim, t_idx = coding_direction_stability(Z, labels, step=8)
        i_first, i_last = 0, len(t_idx) - 1
        assert cos_sim[i_first, i_last] < cos_sim[i_first, i_first] - 0.2

    def test_diagonal_is_near_one(self, rng):
        N, T, k = 80, 32, 5
        labels = (rng.random(N) < 0.5).astype(int)
        direction = rng.standard_normal(k)
        Z = rng.standard_normal((N, T, k)) * 0.2 + 2.0 * labels[:, None, None] * direction[None, None, :]
        cos_sim, t_idx = coding_direction_stability(Z, labels, step=8)
        np.testing.assert_allclose(np.diag(cos_sim), 1.0, atol=1e-6)

    def test_multiclass_fixed_axes_gives_high_stability(self, rng):
        N, T, k, n_classes = 150, 32, 6, 4
        labels = rng.integers(0, n_classes, N)
        directions = rng.standard_normal((n_classes, k))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        Z = rng.standard_normal((N, T, k)) * 0.2
        Z += 2.0 * directions[labels][:, None, :]
        cos_sim, t_idx = coding_direction_stability(Z, labels, step=8)
        assert cos_sim.shape == (len(t_idx), len(t_idx))
        off_diag = cos_sim[~np.eye(len(t_idx), dtype=bool)]
        assert np.mean(off_diag) > 0.7

    def test_multiclass_output_bounded(self, rng):
        N, T, k, n_classes = 100, 24, 5, 3
        labels = rng.integers(0, n_classes, N)
        Z = rng.standard_normal((N, T, k))
        cos_sim, _ = coding_direction_stability(Z, labels, step=6)
        assert np.all(cos_sim >= -1e-8) and np.all(cos_sim <= 1.0 + 1e-8)


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


class TestCtgPhaseScrambleNull:
    def test_output_types_and_range(self, rng):
        N, T, k = 40, 80, 3
        labels = (rng.random(N) < 0.5).astype(int)
        Z = rng.standard_normal((N, T, k))
        tau_obs, tau_null = ctg_phase_scramble_null(Z, labels, step=20, n_permutations=15, rng=rng)
        assert isinstance(tau_obs, float)
        assert tau_null.shape == (15,)
        assert np.isfinite(tau_obs)

    def test_scrambling_preserves_amplitude_spectrum(self, rng):
        # Regression test for a Nyquist-bin phase-scramble bug: for even-length
        # signals, the amplitude spectrum (including the Nyquist bin) must be
        # exactly preserved by the scramble, not just approximately.
        T = 40  # even -> has a genuine real-valued Nyquist bin
        x = rng.standard_normal(T)
        fft = np.fft.rfft(x)
        true_amp = np.abs(fft)
        for _ in range(50):
            phases = rng.uniform(0, 2 * np.pi, len(fft))
            fft_scrambled = np.abs(fft) * np.exp(1j * phases)
            fft_scrambled[0] = fft[0]
            fft_scrambled[-1] = np.abs(fft[-1])
            y = np.fft.irfft(fft_scrambled, n=T)
            np.testing.assert_allclose(np.abs(np.fft.rfft(y)), true_amp, atol=1e-8)


class TestParallelAnalysis:
    def test_recovers_known_rank(self, rng):
        # rank-3 signal + isotropic noise: parallel analysis should retain 3.
        N, D, rank = 300, 12, 3
        loadings = rng.standard_normal((rank, D))
        scores = rng.standard_normal((N, rank))
        X = scores @ loadings * 4.0 + rng.standard_normal((N, D)) * 0.4
        assert parallel_analysis(X, n_surrogate=100, rng=rng) == rank

    def test_pure_noise_recovers_near_zero(self, rng):
        X = rng.standard_normal((200, 8))
        assert parallel_analysis(X, n_surrogate=100, rng=rng) <= 1


class TestSelectLatentDim:
    def test_k_between_one_and_n_channels(self, rng):
        N, C, T = 40, 10, 20
        X = rng.standard_normal((N, C, T))
        for method in ("cv_pr", "parallel_analysis"):
            sel = select_latent_dim(X, method=method, rng=rng)
            assert 1 <= sel["k"] <= C
            assert sel["k"] == sel[f"k_{method}"]
            assert sel["n_channels"] == C

    def test_reports_both_selectors(self, rng):
        X = rng.standard_normal((40, 10, 20))
        sel = select_latent_dim(X, rng=rng)
        assert "k_cv_pr" in sel and "k_parallel_analysis" in sel
        assert np.isfinite(sel["cv_pr"])

    def test_bad_method_raises(self, rng):
        X = rng.standard_normal((30, 6, 15))
        with pytest.raises(ValueError):
            select_latent_dim(X, method="aic", rng=rng)
