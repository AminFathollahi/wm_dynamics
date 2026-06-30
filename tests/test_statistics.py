"""Tests for src/statistics.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from statistics import (
    bootstrap_ci,
    bootstrap_ci_timecourse,
    permutation_test_twosample,
    temporal_cluster_permutation,
    auroc,
    permutation_test_auroc,
    cohens_d,
    hedges_g,
    linear_mixed_effects_test,
)


class TestBootstrapCI:
    def test_ci_contains_true_mean(self, rng):
        data = rng.standard_normal(100) + 3.0
        obs, lo, hi = bootstrap_ci(data, np.mean, n_boot=500, ci=0.95, rng=rng)
        assert lo < 3.0 < hi

    def test_ci_wider_for_smaller_sample(self, rng):
        data_large = rng.standard_normal(1000)
        data_small = rng.standard_normal(20)
        _, lo_l, hi_l = bootstrap_ci(data_large, np.mean, n_boot=200, rng=rng)
        _, lo_s, hi_s = bootstrap_ci(data_small, np.mean, n_boot=200, rng=rng)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_returns_three_values(self, rng):
        data = rng.standard_normal(50)
        result = bootstrap_ci(data, np.mean, rng=rng)
        assert len(result) == 3

    def test_observed_stat_matches(self, rng):
        data = rng.standard_normal(80)
        obs, _, _ = bootstrap_ci(data, np.mean, rng=rng)
        np.testing.assert_allclose(obs, np.mean(data), rtol=1e-10)


class TestPermutationTest:
    def test_detects_real_difference(self, rng):
        x = rng.standard_normal(50) + 2.0  # shifted
        y = rng.standard_normal(50)
        stat, p = permutation_test_twosample(x, y, n_perm=2000, rng=rng)
        assert p < 0.05
        assert stat > 0

    def test_null_pvalue_uniform(self, rng):
        # Under H0, p-values should be approximately uniform
        ps = []
        for _ in range(200):
            x = rng.standard_normal(20)
            y = rng.standard_normal(20)
            _, p = permutation_test_twosample(x, y, n_perm=500, rng=rng)
            ps.append(p)
        # Under H0, <5% should be significant at alpha=0.05 (binomial check)
        assert np.mean(np.array(ps) < 0.05) < 0.15

    def test_two_sided_vs_one_sided(self, rng):
        x = rng.standard_normal(40) + 1.5
        y = rng.standard_normal(40)
        _, p_two = permutation_test_twosample(x, y, n_perm=1000, alternative="two-sided", rng=rng)
        _, p_gt = permutation_test_twosample(x, y, n_perm=1000, alternative="greater", rng=rng)
        # One-sided in the right direction should give smaller or equal p
        assert p_gt <= p_two + 0.05


class TestTemporalClusterPermutation:
    def test_finds_significant_cluster(self, rng):
        T = 100
        # Signal present from t=30 to t=70
        x = rng.standard_normal((30, T))
        y = rng.standard_normal((30, T))
        x[:, 30:70] += 3.0  # large effect in this window
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation(x, y, times, n_perm=500, rng=rng)
        sig = result["significant"]
        assert len(sig) > 0

    def test_no_cluster_under_null(self, rng):
        T = 80
        x = rng.standard_normal((20, T))
        y = rng.standard_normal((20, T))
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation(x, y, times, n_perm=500, rng=rng)
        # Under H0, should mostly not find significant clusters
        sig = result["significant"]
        assert len(sig) == 0 or all(c["p_value"] > 0.001 for c in sig)

    def test_t_stat_shape(self, rng):
        T = 60
        x = rng.standard_normal((15, T))
        y = rng.standard_normal((15, T))
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation(x, y, times, n_perm=100, rng=rng)
        assert result["t_stat"].shape == (T,)


class TestAUROC:
    def test_perfect_classifier(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert auroc(y, scores) == pytest.approx(1.0)

    def test_random_classifier(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        scores = rng.random(200)
        auc = auroc(y, scores)
        # Random classifier should be near 0.5
        assert abs(auc - 0.5) < 0.1

    def test_worst_classifier(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])  # inverted
        assert auroc(y, scores) == pytest.approx(0.0)

    def test_returns_nan_for_single_class(self):
        y = np.array([1, 1, 1])
        scores = np.array([0.5, 0.6, 0.7])
        assert np.isnan(auroc(y, scores))

    def test_range_01(self, rng):
        y = rng.integers(0, 2, 100)
        scores = rng.random(100)
        auc = auroc(y, scores)
        assert 0.0 <= auc <= 1.0


class TestPermutationAUROC:
    def test_significant_for_informative_scores(self, rng):
        y = np.array([0] * 50 + [1] * 50)
        scores = np.concatenate([rng.standard_normal(50), rng.standard_normal(50) + 2.0])
        auc, p = permutation_test_auroc(y, scores, n_perm=500, rng=rng)
        assert p < 0.05
        assert auc > 0.7


class TestCohensD:
    def test_zero_for_identical(self, rng):
        x = rng.standard_normal(100)
        assert abs(cohens_d(x, x)) < 1e-10

    def test_positive_when_x_larger(self):
        x = np.ones(50) * 3.0
        y = np.ones(50) * 1.0
        assert cohens_d(x, y) > 0

    def test_large_effect(self):
        x = np.ones(100) + np.random.randn(100) * 0.1
        y = np.zeros(100) + np.random.randn(100) * 0.1
        assert abs(cohens_d(x, y)) > 5.0  # mean diff = 1, SD ≈ 0.1


class TestHedgesG:
    def test_smaller_than_cohens_d_small_n(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(10) + 1.0
        y = rng.standard_normal(10)
        assert abs(hedges_g(x, y)) < abs(cohens_d(x, y))

    def test_sign_matches_cohens_d(self, rng):
        x = rng.standard_normal(30) + 2.0
        y = rng.standard_normal(30)
        assert np.sign(hedges_g(x, y)) == np.sign(cohens_d(x, y))

    def test_large_n_converges_to_cohens_d(self, rng):
        x = rng.standard_normal(10000) + 0.5
        y = rng.standard_normal(10000)
        np.testing.assert_allclose(hedges_g(x, y), cohens_d(x, y), rtol=1e-3)


class TestLinearMixedEffectsTest:
    def test_detects_load_effect(self, rng):
        # PR increases with load across subjects
        n_subj, n_trials = 4, 30
        subjects = np.repeat(np.arange(n_subj), n_trials)
        conditions = np.tile(np.array([0]*10 + [1]*10 + [2]*10), n_subj)
        metric = conditions * 1.5 + rng.standard_normal(n_subj * n_trials) * 0.5
        result = linear_mixed_effects_test(metric, conditions, subjects, n_perm=500, rng=rng)
        assert result["beta"] > 0
        assert result["p_value"] < 0.05

    def test_null_is_uniform(self, rng):
        n_subj, n_trials = 4, 20
        subjects = np.repeat(np.arange(n_subj), n_trials)
        conditions = np.tile(np.arange(n_trials), n_subj)
        metric = rng.standard_normal(n_subj * n_trials)
        result = linear_mixed_effects_test(metric, conditions, subjects, n_perm=500, rng=rng)
        assert result["p_value"] > 0.01  # should not be significant

    def test_returns_required_keys(self, rng):
        x = rng.standard_normal(40)
        c = np.tile([0, 1, 2, 3], 10)
        s = np.repeat(np.arange(10), 4)
        result = linear_mixed_effects_test(x, c, s, n_perm=100, rng=rng)
        assert set(result.keys()) >= {"beta", "p_value", "r_squared"}
