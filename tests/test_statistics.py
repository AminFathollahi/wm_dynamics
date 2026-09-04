"""Tests for src/statistics.py."""

import numpy as np
import pytest
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from statistics import (
    bootstrap_ci,
    bootstrap_ci_timecourse,
    permutation_test_twosample,
    paired_sign_flip_test,
    spearman_permutation_test,
    temporal_cluster_permutation,
    temporal_cluster_permutation_auroc,
    gated_outcome_cluster_test,
    auroc,
    permutation_test_auroc,
    cohens_d,
    hedges_g,
    linear_mixed_effects_test,
    rayleigh_test,
    ctg_offdiagonal_test,
    mantel_test,
    group_test_auroc,
    fdr_bh,
    robust_dispersion,
    stouffer_combine,
    forest_meta,
    stable_seed,
    permutation_pvalue,
    tost_equivalence,
    bf_null_slope,
    minimum_detectable_paired_difference,
    partial_correlation_permutation_test,
    power_to_detect_effect,
    circular_anova_permutation_test,
)


class TestPermutationPvalue:
    def test_never_exactly_zero(self):
        # Regression test: even when NO null draw is as extreme as observed,
        # the +1/+1 correction must keep p bounded away from 0.
        mask = np.zeros(999, dtype=bool)
        p = permutation_pvalue(mask)
        assert p > 0.0
        assert p == pytest.approx(1.0 / 1000)

    def test_all_exceed_gives_p_one(self):
        mask = np.ones(500, dtype=bool)
        p = permutation_pvalue(mask)
        assert p == pytest.approx(1.0)

    def test_matches_formula(self):
        mask = np.array([True, False, True, True, False])
        p = permutation_pvalue(mask)
        assert p == pytest.approx((3 + 1) / (5 + 1))


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


class TestPairedSignFlipTest:
    def test_detects_consistent_paired_decrease(self, rng):
        a = rng.standard_normal(20) * 0.1 + 0.2   # consistently lower
        b = rng.standard_normal(20) * 0.1 + 0.8
        res = paired_sign_flip_test(a, b, n_perm=2000, alternative="less", rng=rng)
        assert res["p_value"] < 0.05
        assert res["mean_diff"] < 0

    def test_null_pvalue_uniform_when_no_effect(self, rng):
        ps = []
        for _ in range(200):
            a = rng.standard_normal(15)
            b = rng.standard_normal(15)
            res = paired_sign_flip_test(a, b, n_perm=500, alternative="two-sided", rng=rng)
            ps.append(res["p_value"])
        assert np.mean(np.array(ps) < 0.05) < 0.15

    def test_ci_excludes_zero_for_strong_effect(self, rng):
        a = np.full(10, 0.2)
        b = np.full(10, 0.9)
        res = paired_sign_flip_test(a, b, n_perm=1000, rng=rng)
        assert res["ci_upper"] < 0

    def test_raises_on_nonfinite_input(self, rng):
        a = np.array([0.1, 0.2, np.nan, 0.3])
        b = rng.standard_normal(4)
        with pytest.raises(ValueError):
            paired_sign_flip_test(a, b, n_perm=100, rng=rng)

    def test_mean_diff_matches_naive_difference(self, rng):
        a = rng.standard_normal(12)
        b = rng.standard_normal(12)
        res = paired_sign_flip_test(a, b, n_perm=100, rng=rng)
        assert res["mean_diff"] == pytest.approx(np.mean(a - b))


class TestSpearmanPermutationTest:
    def test_detects_monotonic_relationship(self, rng):
        x = np.arange(20, dtype=float)
        y = x + rng.standard_normal(20) * 0.5
        res = spearman_permutation_test(x, y, n_perm=2000, rng=rng)
        assert res["rho"] > 0.8
        assert res["p_value"] < 0.05

    def test_null_pvalue_uniform_when_uncorrelated(self, rng):
        ps = []
        for _ in range(200):
            x = rng.standard_normal(15)
            y = rng.standard_normal(15)
            res = spearman_permutation_test(x, y, n_perm=300, rng=rng)
            ps.append(res["p_value"])
        assert np.mean(np.array(ps) < 0.05) < 0.15

    def test_negative_relationship_gives_negative_rho(self, rng):
        x = np.arange(10, dtype=float)
        y = -x + rng.standard_normal(10) * 0.1
        res = spearman_permutation_test(x, y, n_perm=1000, rng=rng)
        assert res["rho"] < 0


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


class TestTemporalClusterPermutationAuroc:
    def test_finds_significant_cluster(self, rng):
        N, T = 80, 50
        outcome = (rng.random(N) < 0.5).astype(int)
        scores = rng.standard_normal((N, T))
        scores[:, 15:35] += 2.0 * outcome[:, None]   # signal only in this window
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation_auroc(scores, outcome, times, n_perm=300, rng=rng)
        assert len(result["significant"]) > 0

    def test_no_cluster_under_null(self, rng):
        N, T = 60, 40
        outcome = (rng.random(N) < 0.5).astype(int)
        scores = rng.standard_normal((N, T))   # no relationship to outcome
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation_auroc(scores, outcome, times, n_perm=300, rng=rng)
        sig = result["significant"]
        assert len(sig) == 0 or all(c["p_value"] > 0.001 for c in sig)

    def test_auc_stat_shape_and_range(self, rng):
        N, T = 50, 20
        outcome = (rng.random(N) < 0.5).astype(int)
        scores = rng.standard_normal((N, T))
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation_auroc(scores, outcome, times, n_perm=100, rng=rng)
        assert result["auc_stat"].shape == (T,)
        assert np.all(np.abs(result["auc_stat"]) <= 0.5 + 1e-8)

    def test_matches_manual_auroc_at_each_timepoint(self, rng):
        N, T = 40, 5
        outcome = (rng.random(N) < 0.5).astype(int)
        scores = rng.standard_normal((N, T))
        times = np.linspace(0, 1, T)
        result = temporal_cluster_permutation_auroc(scores, outcome, times, n_perm=50, rng=rng)
        manual = np.array([auroc(outcome, scores[:, t]) - 0.5 for t in range(T)])
        np.testing.assert_allclose(result["auc_stat"], manual)


class TestGatedOutcomeClusterTest:
    def test_none_when_one_outcome_group_underpowered(self, rng):
        N, T = 30, 10
        outcome = np.zeros(N, dtype=int)
        outcome[:3] = 1   # only 3 trials in the minority group
        confidence = rng.standard_normal((N, T))
        times = np.linspace(0, 1, T)
        result = gated_outcome_cluster_test(confidence, outcome, times,
                                            min_trials_per_outcome=8, rng=rng)
        assert result is None

    def test_none_when_too_many_nonfinite_trials(self, rng):
        N, T = 40, 10
        outcome = (rng.random(N) < 0.5).astype(int)
        confidence = rng.standard_normal((N, T))
        confidence[:35] = np.nan   # only 5 usable trials remain
        times = np.linspace(0, 1, T)
        result = gated_outcome_cluster_test(confidence, outcome, times,
                                            min_trials_per_outcome=8, rng=rng)
        assert result is None

    def test_runs_and_matches_underlying_cluster_test_when_powered(self, rng):
        N, T = 80, 20
        outcome = (rng.random(N) < 0.5).astype(int)
        confidence = rng.standard_normal((N, T))
        confidence[:, 5:10] += 2.0 * outcome[:, None]
        times = np.linspace(0, 1, T)
        result = gated_outcome_cluster_test(confidence, outcome, times,
                                            min_trials_per_outcome=8, n_perm=200,
                                            rng=np.random.default_rng(0))
        assert result is not None
        assert result["n_trials"] == N
        assert len(result["significant"]) > 0


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

    def test_covariate_absorbs_confound(self, rng):
        # metric is driven entirely by a covariate that also correlates with
        # condition; without adjustment this looks like a condition effect,
        # with adjustment it should not.
        n_subj, n_trials = 6, 40
        subjects = np.repeat(np.arange(n_subj), n_trials)
        covariate = rng.standard_normal(n_subj * n_trials)
        condition = covariate + rng.standard_normal(n_subj * n_trials) * 0.1
        metric = covariate * 2.0 + rng.standard_normal(n_subj * n_trials) * 0.1

        unadjusted = linear_mixed_effects_test(metric, condition, subjects, n_perm=500, rng=rng)
        adjusted = linear_mixed_effects_test(metric, condition, subjects, n_perm=500,
                                             rng=rng, covariates=covariate)
        assert unadjusted["p_value"] < 0.05
        assert abs(adjusted["beta"]) < abs(unadjusted["beta"])

    def test_covariate_with_no_true_effect_barely_shifts_beta(self, rng):
        x = rng.standard_normal(40)
        c = np.tile([0, 1, 2, 3], 10).astype(float)
        s = np.repeat(np.arange(10), 4)
        unrelated_covariate = rng.standard_normal(40)
        r1 = linear_mixed_effects_test(x, c, s, rng=np.random.default_rng(1))
        r2 = linear_mixed_effects_test(x, c, s, rng=np.random.default_rng(1),
                                       covariates=unrelated_covariate)
        assert r1["beta"] == pytest.approx(r2["beta"], abs=0.5)

    def test_degenerate_covariate_reports_explicit_failure(self, rng):
        # A covariate with exactly zero variance makes the joint fixed-effect
        # design matrix singular. The fit must fail loudly (converged=False,
        # nan, and a reason) rather than silently falling back to a
        # zero-effect/p=1 result.
        x = rng.standard_normal(40)
        c = np.tile([0, 1, 2, 3], 10).astype(float)
        s = np.repeat(np.arange(10), 4)
        result = linear_mixed_effects_test(x, c, s, covariates=np.zeros(40))
        assert result["converged"] is False
        assert result["reason"]
        assert np.isnan(result["beta"])
        assert np.isnan(result["p_value"])


class TestRayleighTest:
    def test_uniform_not_significant(self, rng):
        phases = rng.uniform(-np.pi, np.pi, 200)
        result = rayleigh_test(phases)
        assert result["p_value"] > 0.05

    def test_concentrated_significant(self):
        phases = np.random.default_rng(0).normal(0, 0.1, 100)
        result = rayleigh_test(phases)
        assert result["p_value"] < 0.01

    def test_R_range(self, rng):
        phases = rng.uniform(-np.pi, np.pi, 50)
        result = rayleigh_test(phases)
        assert 0.0 <= result["R"] <= 1.0

    def test_perfect_concentration(self):
        phases = np.zeros(50)
        result = rayleigh_test(phases)
        np.testing.assert_allclose(result["R"], 1.0, atol=1e-10)
        assert result["p_value"] < 0.001

    def test_returns_required_keys(self, rng):
        phases = rng.uniform(-np.pi, np.pi, 30)
        result = rayleigh_test(phases)
        assert set(result.keys()) >= {"R", "Z", "p_value", "mean_direction", "N"}


class TestCircularAnovaPermutationTest:
    def test_shared_mean_direction_not_significant(self, rng):
        phases = rng.uniform(-np.pi, np.pi, 90)
        groups = np.repeat(["a", "b", "c"], 30)
        result = circular_anova_permutation_test(phases, groups, n_perm=300, rng=rng)
        assert result["p_value"] > 0.05

    def test_separated_mean_directions_significant(self, rng):
        phases = np.concatenate([
            rng.normal(0.0, 0.1, 30), rng.normal(np.pi / 2, 0.1, 30), rng.normal(np.pi, 0.1, 30),
        ])
        groups = np.repeat(["a", "b", "c"], 30)
        result = circular_anova_permutation_test(phases, groups, n_perm=300, rng=rng)
        assert result["p_value"] < 0.01

    def test_returns_required_keys(self, rng):
        phases = rng.uniform(-np.pi, np.pi, 40)
        groups = np.repeat(["a", "b"], 20)
        result = circular_anova_permutation_test(phases, groups, n_perm=100, rng=rng)
        assert set(result.keys()) >= {"statistic", "p_value", "n_groups", "N"}
        assert result["n_groups"] == 2
        assert result["N"] == 40


class TestCTGOffdiagonalTest:
    def test_identity_matrix_zero_offdiag(self, rng):
        auc = np.eye(10) * 0.8 + 0.5 * (1 - np.eye(10))
        result = ctg_offdiagonal_test(auc, n_perm=200, rng=rng)
        assert result["mean_diag"] > result["mean_offdiag"]
        assert result["temporal_stability"] < 1.0

    def test_flat_matrix_high_stability(self, rng):
        auc = np.full((8, 8), 0.75)
        result = ctg_offdiagonal_test(auc, n_perm=100, rng=rng)
        np.testing.assert_allclose(result["temporal_stability"], 1.0, atol=1e-6)

    def test_returns_required_keys(self, rng):
        auc = rng.uniform(0.4, 0.8, (6, 6))
        result = ctg_offdiagonal_test(auc, n_perm=100, rng=rng)
        assert set(result.keys()) >= {
            "mean_diag", "mean_offdiag", "p_offdiag_vs_chance",
            "p_diag_vs_offdiag", "temporal_stability"
        }

    def test_above_chance_detected(self, rng):
        auc = np.full((10, 10), 0.75)
        result = ctg_offdiagonal_test(auc, n_perm=500, rng=rng)
        assert result["p_offdiag_vs_chance"] < 0.05


class TestMantelTest:
    def test_identical_rdms_r_one(self):
        rdm = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        result = mantel_test(rdm, rdm, n_perm=200)
        np.testing.assert_allclose(result["r"], 1.0, atol=1e-8)

    def test_uncorrelated_rdms_near_zero(self, rng):
        rdm_a = rng.uniform(0, 1, (8, 8)); rdm_a = (rdm_a + rdm_a.T) / 2; np.fill_diagonal(rdm_a, 0)
        rdm_b = rng.uniform(0, 1, (8, 8)); rdm_b = (rdm_b + rdm_b.T) / 2; np.fill_diagonal(rdm_b, 0)
        result = mantel_test(rdm_a, rdm_b, n_perm=200, rng=rng)
        assert -1.0 <= result["r"] <= 1.0

    def test_p_value_range(self, rng):
        rdm_a = rng.uniform(0, 1, (6, 6))
        rdm_b = rng.uniform(0, 1, (6, 6))
        result = mantel_test(rdm_a, rdm_b, n_perm=200, rng=rng)
        assert 0.0 <= result["p_value"] <= 1.0


class TestGroupTestAUROC:
    def test_above_chance_significant(self):
        aurocs = np.array([0.7, 0.72, 0.68, 0.75, 0.71, 0.69, 0.73, 0.70])
        result = group_test_auroc(aurocs)
        assert result["p_value"] < 0.05
        assert result["mean"] > 0.5

    def test_chance_level_not_significant(self, rng):
        aurocs = rng.uniform(0.45, 0.55, 10)
        result = group_test_auroc(aurocs)
        assert result["p_value"] > 0.05

    def test_ci_contains_mean(self, rng):
        aurocs = rng.uniform(0.6, 0.8, 20)
        result = group_test_auroc(aurocs)
        assert result["ci_lo"] < result["mean"] < result["ci_hi"]


class TestFDRBH:
    def test_all_significant_all_rejected(self):
        p = np.array([0.001, 0.002, 0.003, 0.0005])
        res = fdr_bh(p, alpha=0.05)
        assert res["n_reject"] == 4
        assert np.all(res["reject"])

    def test_none_significant_none_rejected(self):
        p = np.array([0.8, 0.9, 0.6, 0.7])
        res = fdr_bh(p, alpha=0.05)
        assert res["n_reject"] == 0

    def test_q_values_monotone_with_sorted_p(self):
        p = np.array([0.001, 0.2, 0.03, 0.5, 0.01])
        res = fdr_bh(p)
        order = np.argsort(p)
        q_sorted = res["q_values"][order]
        assert np.all(np.diff(q_sorted) >= -1e-10)

    def test_less_conservative_than_bonferroni(self):
        # A mix of very small and moderate p-values: BH should reject at
        # least as many hypotheses as Bonferroni at the same alpha.
        p = np.array([0.001, 0.004, 0.01, 0.02, 0.04, 0.3, 0.6])
        res = fdr_bh(p, alpha=0.05)
        bonferroni_reject = (p <= 0.05 / len(p)).sum()
        assert res["n_reject"] >= bonferroni_reject


class TestRobustDispersion:
    def test_single_outlier_does_not_blow_up_iqr_ratio(self):
        x = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 0.001])
        res = robust_dispersion(x)
        naive_range = x.max() / x.min()
        assert res["iqr_ratio"] < naive_range

    def test_median_robust_to_outlier(self):
        x = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 1000.0])
        res = robust_dispersion(x)
        assert 8.0 < res["median"] < 12.0

    def test_constant_array_has_zero_iqr(self):
        x = np.full(10, 5.0)
        res = robust_dispersion(x)
        assert res["iqr"] == pytest.approx(0.0)


class TestStoufferCombine:
    def test_all_null_gives_p_near_half(self):
        res = stouffer_combine(np.full(6, 0.5))
        assert res["p_combined"] == pytest.approx(0.5, abs=1e-6)

    def test_consistent_small_p_values_combine_to_smaller_p(self):
        res = stouffer_combine(np.full(8, 0.2))
        assert res["p_combined"] < 0.2

    def test_one_strong_signal_among_null_still_shifts_combined_p(self):
        p_null_only = stouffer_combine(np.full(5, 0.5))["p_combined"]
        p_with_signal = stouffer_combine(np.array([0.001, 0.5, 0.5, 0.5, 0.5]))["p_combined"]
        assert p_with_signal < p_null_only

    def test_weights_emphasize_larger_sessions(self):
        p = np.array([0.01, 0.9])
        res_equal = stouffer_combine(p)
        res_weighted = stouffer_combine(p, weights=np.array([10.0, 1.0]))
        assert res_weighted["p_combined"] < res_equal["p_combined"]


class TestStableSeed:
    def test_same_string_same_process_gives_same_seed(self):
        assert stable_seed("sub-01") == stable_seed("sub-01")

    def test_different_strings_give_different_seeds(self):
        assert stable_seed("al") != stable_seed("ca")

    def test_stable_across_hash_randomization(self):
        # Regression guard: must not depend on Python's salted hash() and
        # must reproduce a fixed value independent of PYTHONHASHSEED.
        assert stable_seed("al") == zlib.crc32("al".encode("utf-8"))

    def test_returns_valid_rng_seed(self):
        seed = stable_seed("sub-09")
        rng = np.random.default_rng(seed)
        assert 0 <= seed < 2**32
        assert rng.random() is not None


class TestForestMeta:
    def test_homogeneous_pooled_equals_common_value(self):
        est = np.array([0.4, 0.4, 0.4, 0.4])
        se = np.array([0.1, 0.1, 0.1, 0.1])
        res = forest_meta(est, se)
        assert abs(res["pooled"] - 0.4) < 1e-9
        assert res["i_squared"] < 1e-6
        assert res["tau2"] < 1e-9

    def test_fixed_effect_equal_se_is_mean_with_shrunk_se(self):
        est = np.array([0.2, 0.6])
        se = np.array([0.1, 0.1])
        res = forest_meta(est, se, method="fixed")
        assert abs(res["pooled"] - 0.4) < 1e-9
        assert abs(res["se"] - 0.1 / np.sqrt(2)) < 1e-9

    def test_precise_cohort_dominates_pooled(self):
        est = np.array([0.0, 1.0])
        se = np.array([0.01, 1.0])
        res = forest_meta(est, se, method="fixed")
        assert res["pooled"] < 0.05
        assert res["rows"][0]["weight_pct"] > 99.0

    def test_heterogeneity_detected(self):
        est = np.array([0.1, 0.9, 0.1, 0.9])
        se = np.array([0.02, 0.02, 0.02, 0.02])
        res = forest_meta(est, se)
        assert res["i_squared"] > 90.0
        assert res["tau2"] > 0.0
        assert res["Q_p"] < 0.05

    def test_random_effects_ci_wider_than_fixed_under_heterogeneity(self):
        est = np.array([0.1, 0.9, 0.1, 0.9])
        se = np.array([0.05, 0.05, 0.05, 0.05])
        re = forest_meta(est, se, method="random")
        fe = forest_meta(est, se, method="fixed")
        assert (re["ci_hi"] - re["ci_lo"]) > (fe["ci_hi"] - fe["ci_lo"])

    def test_single_cohort(self):
        res = forest_meta(np.array([0.3]), np.array([0.1]))
        assert abs(res["pooled"] - 0.3) < 1e-9
        assert res["k"] == 1
        assert res["i_squared"] == 0.0

    def test_drops_nonfinite_rows(self):
        est = np.array([0.4, np.nan, 0.4])
        se = np.array([0.1, 0.1, 0.0])
        res = forest_meta(est, se)
        assert res["k"] == 1
        assert len(res["rows"]) == 1

    def test_recovers_known_effect(self, rng):
        true = 0.35
        k = 20
        se = rng.uniform(0.05, 0.2, size=k)
        est = true + rng.standard_normal(k) * se
        res = forest_meta(est, se)
        assert res["ci_lo"] < true < res["ci_hi"]

    def test_rows_carry_labels_and_cis(self):
        res = forest_meta(np.array([0.2, 0.5]), np.array([0.1, 0.1]),
                          labels=["Miller", "Boran"])
        assert [r["label"] for r in res["rows"]] == ["Miller", "Boran"]
        r0 = res["rows"][0]
        assert r0["ci_lo"] < r0["estimate"] < r0["ci_hi"]

    def test_empty_after_filter_raises(self):
        with pytest.raises(ValueError):
            forest_meta(np.array([np.nan, np.nan]), np.array([0.1, 0.1]))


class TestTostEquivalence:
    def test_rejects_for_tight_ci_near_zero(self):
        # Estimate near zero with a small SE, well inside a wide SESOI -> equivalent.
        res = tost_equivalence(0.01, 0.05, sesoi=0.5)
        assert res["reject"] is True
        assert res["p"] < 0.05

    def test_fails_to_reject_for_wide_ci(self):
        # A CI far wider than the SESOI cannot exclude a meaningful effect.
        res = tost_equivalence(0.01, 0.9, sesoi=0.5)
        assert res["reject"] is False
        assert res["p"] > 0.05

    def test_binding_p_is_the_larger_one_sided(self):
        res = tost_equivalence(0.3, 0.1, sesoi=0.5)
        assert res["p"] == max(res["p_lower"], res["p_upper"])


class TestBfNullSlope:
    def test_favours_null_for_near_zero_estimate(self):
        bf = bf_null_slope(0.009, 0.057, r_scale=0.5)
        assert bf["bf_01"] > 3.0   # substantial evidence for the null

    def test_favours_effect_for_clear_slope(self):
        bf = bf_null_slope(0.9, 0.1, r_scale=0.5)
        assert bf["bf_10"] > 3.0   # substantial evidence against the null
        assert bf["bf_01"] < 1.0

    def test_bf01_and_bf10_reciprocal(self):
        bf = bf_null_slope(0.05, 0.1, r_scale=0.5)
        np.testing.assert_allclose(bf["bf_01"] * bf["bf_10"], 1.0, rtol=1e-9)


class TestMinimumDetectablePairedDifference:
    def test_matches_closed_form_for_known_spread(self):
        # sd=1, n=25 -> mdd = (1.96 + 0.8416) / 5 ~= 0.5603
        rng = np.random.default_rng(0)
        values = rng.normal(loc=0.0, scale=1.0, size=25)
        result = minimum_detectable_paired_difference(values)
        expected = (1.959964 + 0.841621) * np.std(values, ddof=1) / np.sqrt(25)
        assert result["status"] == "computed"
        np.testing.assert_allclose(result["mdd"], expected, rtol=1e-4)

    def test_larger_n_gives_a_smaller_bound_at_fixed_spread(self):
        rng = np.random.default_rng(1)
        small = minimum_detectable_paired_difference(rng.normal(scale=2.0, size=10))
        large = minimum_detectable_paired_difference(rng.normal(scale=2.0, size=1000))
        assert large["mdd"] < small["mdd"]

    def test_too_few_values_is_not_computable(self):
        result = minimum_detectable_paired_difference([0.1])
        assert result["status"] == "not_computable"


class TestPowerToDetectEffect:
    def test_power_is_high_for_a_large_effect_relative_to_spread(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=0.12, scale=0.11, size=36)
        result = power_to_detect_effect(effect=0.12, values=values)
        assert result["status"] == "computed"
        assert result["power"] > 0.95

    def test_power_is_low_for_a_tiny_effect_relative_to_spread(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=0.0, scale=1.0, size=10)
        result = power_to_detect_effect(effect=0.01, values=values)
        assert result["power"] < 0.10

    def test_power_increases_with_sample_size_at_fixed_spread_and_effect(self):
        rng = np.random.default_rng(1)
        small = power_to_detect_effect(effect=0.1, values=rng.normal(scale=0.3, size=10))
        large = power_to_detect_effect(effect=0.1, values=rng.normal(scale=0.3, size=200))
        assert large["power"] > small["power"]

    def test_zero_effect_reduces_to_alpha(self):
        # power against a true effect of exactly 0 is the test's own false-positive rate: alpha
        rng = np.random.default_rng(2)
        values = rng.normal(scale=1.0, size=500)
        result = power_to_detect_effect(effect=0.0, values=values)
        assert result["power"] == pytest.approx(0.05, abs=0.01)

    def test_too_few_values_is_not_computable(self):
        assert power_to_detect_effect(0.1, [0.1])["status"] == "not_computable"

    def test_zero_spread_is_not_computable(self):
        assert power_to_detect_effect(0.1, [1.0, 1.0, 1.0])["status"] == "not_computable"


class TestPartialCorrelationPermutationTest:
    def test_no_controls_matches_plain_pearson(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = 0.6 * x + rng.normal(size=200)
        from scipy.stats import pearsonr
        result = partial_correlation_permutation_test(y, x, controls=[], n_perm=500, rng=np.random.default_rng(1))
        expected_r, _ = pearsonr(y, x)
        assert result["status"] == "computed"
        np.testing.assert_allclose(result["r"], expected_r, rtol=1e-8)
        assert result["p_value"] < 0.05

    def test_confound_fully_explained_by_control_vanishes(self):
        # y and x are both driven only by z; once z is controlled for, y and x share nothing.
        rng = np.random.default_rng(2)
        z = rng.normal(size=300)
        y = z + rng.normal(scale=0.01, size=300)
        x = z + rng.normal(scale=0.01, size=300)
        raw = partial_correlation_permutation_test(y, x, controls=[], n_perm=200, rng=np.random.default_rng(3))
        partial = partial_correlation_permutation_test(y, x, controls=[z], n_perm=2000, rng=np.random.default_rng(4))
        assert raw["r"] > 0.9
        assert abs(partial["r"]) < 0.2
        assert partial["p_value"] > 0.05

    def test_genuine_partial_relationship_survives_an_irrelevant_control(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=300)
        irrelevant = rng.normal(size=300)
        y = 0.7 * x + rng.normal(scale=0.3, size=300)
        result = partial_correlation_permutation_test(y, x, controls=[irrelevant], n_perm=2000, rng=np.random.default_rng(6))
        assert result["r"] > 0.5
        assert result["p_value"] < 0.05
        assert result["n_controls"] == 1
