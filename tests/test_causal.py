"""Tests for src/causal.py — doubly-robust ATE and CATE-by-modifier estimation.

Each test builds a data-generating process with a KNOWN average effect and a
KNOWN modifier slope, then checks the estimator recovers it. Outcome nuisances
are passed a LinearRegression so the (linear) DGP is fit exactly and the tests
are fast and deterministic; the doubly-robust guarantees are still exercised
because the propensity is either known (randomised) or estimated separately.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from causal import (
    crossfit_nuisances,
    aipw_pseudo_outcome,
    aipw_ate,
    cate_vs_modifier_slope,
    benchmark_modifiers,
    dr_learner_cate,
    dml_partial_linear,
    e_value,
)
from sklearn.linear_model import LinearRegression


def _dgp(rng, n=3000, tau0=0.5, tau1=0.8, confounded=False):
    """X (n,3) confounders; modifier m = X[:,0]; tau(m)=tau0+tau1*m.

    Returns y, t, X, m, true_ate (= tau0, since E[m]=0), true_slope (= tau1),
    and the known propensity (0.5 if randomised, else None-signalled by prop).
    """
    X = rng.standard_normal((n, 3))
    m = X[:, 0]
    if confounded:
        logits = 0.9 * X[:, 0] + 0.7 * X[:, 1]
        e = 1.0 / (1.0 + np.exp(-logits))
    else:
        e = np.full(n, 0.5)
    t = (rng.random(n) < e).astype(int)

    y0 = 1.0 + X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(n) * 0.3
    tau = tau0 + tau1 * m
    y1 = y0 + tau
    y = np.where(t == 1, y1, y0)
    return y, t, X, m, e


class TestCrossfitNuisances:
    def test_known_propensity_is_passed_through(self, rng):
        y, t, X, m, e = _dgp(rng, n=500)
        nu = crossfit_nuisances(y, t, X, n_folds=3, propensity=e,
                                outcome_learner=LinearRegression(), rng=rng)
        np.testing.assert_allclose(nu["e_hat"], np.clip(e, 1e-2, 1 - 1e-2))

    def test_outputs_are_full_length_and_finite(self, rng):
        y, t, X, m, e = _dgp(rng, n=600)
        nu = crossfit_nuisances(y, t, X, n_folds=4,
                                outcome_learner=LinearRegression(), rng=rng)
        for key in ("e_hat", "mu0_hat", "mu1_hat"):
            assert nu[key].shape == (600,)
            assert np.all(np.isfinite(nu[key]))


class TestAIPWATE:
    def test_recovers_ate_randomised(self, rng):
        y, t, X, m, e = _dgp(rng, n=4000, tau0=0.5, tau1=0.8)
        res = aipw_ate(y, t, X, n_folds=4, propensity=e,
                       outcome_learner=LinearRegression(), rng=rng)
        assert res["ci_lo"] < 0.5 < res["ci_hi"]

    def test_recovers_ate_under_confounding_estimated_propensity(self, rng):
        # Naive difference-in-means is biased here; the doubly-robust estimator
        # with an estimated propensity should still recover the true ATE.
        y, t, X, m, e = _dgp(rng, n=5000, tau0=0.5, tau1=0.8, confounded=True)
        naive = y[t == 1].mean() - y[t == 0].mean()
        res = aipw_ate(y, t, X, n_folds=5, outcome_learner=LinearRegression(), rng=rng)
        assert res["ci_lo"] < 0.5 < res["ci_hi"]
        assert abs(res["ate"] - 0.5) < abs(naive - 0.5)

    def test_pseudo_outcome_mean_is_ate(self, rng):
        y, t, X, m, e = _dgp(rng, n=2000)
        nu = crossfit_nuisances(y, t, X, n_folds=3, propensity=e,
                                outcome_learner=LinearRegression(), rng=rng)
        phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])
        assert abs(phi.mean() - 0.5) < 0.1


class TestCATEvsModifierSlope:
    def test_recovers_positive_slope(self, rng):
        y, t, X, m, e = _dgp(rng, n=4000, tau0=0.5, tau1=0.8)
        res = cate_vs_modifier_slope(y, t, X, m, n_folds=4, propensity=e,
                                     outcome_learner=LinearRegression(),
                                     n_perm=1000, rng=rng)
        assert res["slope_ci_lo"] < 0.8 < res["slope_ci_hi"]
        assert res["slope"] > 0
        assert res["p_value"] < 0.05

    def test_null_slope_not_significant(self, rng):
        y, t, X, m, e = _dgp(rng, n=3000, tau0=0.5, tau1=0.0)
        res = cate_vs_modifier_slope(y, t, X, m, n_folds=4, propensity=e,
                                     outcome_learner=LinearRegression(),
                                     n_perm=1000, rng=rng)
        assert res["slope_ci_lo"] < 0.0 < res["slope_ci_hi"]
        assert res["p_value"] > 0.05

    def test_recovers_slope_under_confounding(self, rng):
        y, t, X, m, e = _dgp(rng, n=5000, tau0=0.5, tau1=0.8, confounded=True)
        res = cate_vs_modifier_slope(y, t, X, m, n_folds=5,
                                     outcome_learner=LinearRegression(),
                                     n_perm=800, rng=rng)
        assert res["slope_ci_lo"] < 0.8 < res["slope_ci_hi"]
        assert res["p_value"] < 0.05


class TestBenchmarkModifiers:
    def _benchmark_dgp(self, rng, n=4000, tau0=0.5, tau1=0.9):
        """One TRUE modifier drives tau(m); two DECOYS are correlated with it
        (so they show a spurious marginal effect but should attenuate once
        the true modifier is also in the model); one is pure noise (null)."""
        X = rng.standard_normal((n, 3))
        true_mod = X[:, 0]
        decoy1 = 0.7 * true_mod + 0.3 * rng.standard_normal(n)
        decoy2 = 0.5 * true_mod + 0.5 * rng.standard_normal(n)
        noise_mod = rng.standard_normal(n)

        e = np.full(n, 0.5)
        t = (rng.random(n) < e).astype(int)
        y0 = 1.0 + X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(n) * 0.3
        tau = tau0 + tau1 * true_mod
        y1 = y0 + tau
        y = np.where(t == 1, y1, y0)

        modifiers = {"true_mod": true_mod, "decoy1": decoy1, "decoy2": decoy2, "noise_mod": noise_mod}
        return y, t, X, e, modifiers

    def test_recovers_true_modifier_as_winner(self, rng):
        y, t, X, e, modifiers = self._benchmark_dgp(rng)
        res = benchmark_modifiers(y, t, X, modifiers, propensity=e,
                                  outcome_learner=LinearRegression(),
                                  n_perm=500, rng=rng)
        assert res["winner"] == "true_mod"

        lb = res["leaderboard"]
        assert lb["true_mod"]["slope"] == max(v["slope"] for v in lb.values())
        assert lb["true_mod"]["p_value"] < 0.05

        # Noise modifier is a null: CI should include zero.
        assert lb["noise_mod"]["slope_ci_lo"] < 0.0 < lb["noise_mod"]["slope_ci_hi"]

        # Joint model: decoys attenuate once the true modifier is present;
        # the true modifier is the only reliable survivor.
        joint = res["joint"]
        assert joint["true_mod"]["p_value"] < 0.05
        assert abs(joint["true_mod"]["coef"]) > abs(joint["decoy1"]["coef"])
        assert abs(joint["true_mod"]["coef"]) > abs(joint["decoy2"]["coef"])

        # Nested: adding pure noise to a model already containing the true
        # modifier contributes negligible incremental variance (at large N a
        # permutation test on dR2 can flag a tiny nonzero sample correlation
        # as "significant" even under a true null, so check magnitude here
        # rather than the p-value alone).
        nested = res["nested"]
        assert abs(nested["noise_mod"]["dR2"]) < 0.005

    def test_single_modifier_matches_cate_vs_modifier_slope_up_to_scale(self, rng):
        # benchmark_modifiers z-scores; cate_vs_modifier_slope does not — the
        # two should agree once put back on the same scale, since both share
        # the _dr_slope core.
        # m scaled well away from unit SD -- catches an unstandardization
        # bug (dividing vs. multiplying by SD) that unit-SD data would mask.
        X = rng.standard_normal((2000, 3))
        m = 5.0 * X[:, 0]
        e = np.full(2000, 0.5)
        t = (rng.random(2000) < e).astype(int)
        y0 = 1.0 + X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(2000) * 0.3
        y1 = y0 + 0.5 + 0.8 * m
        y = np.where(t == 1, y1, y0)

        rng1 = np.random.default_rng(42)
        gate = cate_vs_modifier_slope(y, t, X, m, n_folds=4, propensity=e,
                                      outcome_learner=LinearRegression(),
                                      n_perm=200, rng=rng1)
        rng2 = np.random.default_rng(42)
        res = benchmark_modifiers(y, t, X, {"m": m}, n_folds=4, propensity=e,
                                  outcome_learner=LinearRegression(),
                                  n_perm=200, rng=rng2)
        raw_slope_recovered = res["leaderboard"]["m"]["slope"] / m.std()
        assert raw_slope_recovered == pytest.approx(gate["slope"], rel=0.05)

    def test_constant_modifier_is_excluded_not_scored_as_null(self, rng):
        # A modifier broadcast as the SAME scalar to every row has zero
        # within-arena variance; regressing phi on a constant gives a fake
        # slope==0/CI==[0,0]/p==1.0 that must NOT be filed as a trustworthy
        # null.
        y, t, X, e, modifiers = self._benchmark_dgp(rng)
        modifiers = dict(modifiers)
        modifiers["constant_mod"] = np.full(len(y), 3.14)
        res = benchmark_modifiers(y, t, X, modifiers, propensity=e,
                                  outcome_learner=LinearRegression(),
                                  n_perm=500, rng=rng)
        assert "constant_mod" not in res["leaderboard"]
        assert "constant_mod" not in res["joint"]
        assert "constant_mod" not in res["z_modifiers"]
        assert res["excluded"]["constant_mod"]["eligible"] is False


class TestDRLearnerCATE:
    def test_predicted_cate_tracks_true_cate(self, rng):
        y, t, X, m, e = _dgp(rng, n=5000, tau0=0.5, tau1=0.8)
        res = dr_learner_cate(y, t, X, m, n_folds=5, propensity=e,
                              outcome_learner=LinearRegression(),
                              cate_learner=LinearRegression(), rng=rng)
        true_cate = 0.5 + 0.8 * m
        corr = np.corrcoef(res["cate"], true_cate)[0, 1]
        assert corr > 0.5
        assert res["cate"].shape == (5000,)


def _plr_dgp(rng, n=3000, theta=0.6):
    """X (n,3) confounders; D = m(X) + noise; Y = theta*D + g(X) + noise."""
    X = rng.standard_normal((n, 3))
    m_x = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    d = m_x + rng.standard_normal(n) * 0.5
    g_x = 1.0 + X @ np.array([0.4, -0.2, 0.3])
    y = theta * d + g_x + rng.standard_normal(n) * 0.3
    return y, d, X


class TestDMLPartialLinear:
    def test_recovers_known_theta(self, rng):
        y, d, X = _plr_dgp(rng, n=4000, theta=0.6)
        res = dml_partial_linear(y, d, X, n_folds=4,
                                 outcome_learner=LinearRegression(),
                                 exposure_learner=LinearRegression(), rng=rng)
        assert res["ci_lo"] < 0.6 < res["ci_hi"]
        assert abs(res["theta"] - 0.6) < 0.05

    def test_null_effect_not_significant(self, rng):
        y, d, X = _plr_dgp(rng, n=3000, theta=0.0)
        res = dml_partial_linear(y, d, X, n_folds=4,
                                 outcome_learner=LinearRegression(),
                                 exposure_learner=LinearRegression(), rng=rng)
        assert res["ci_lo"] < 0.0 < res["ci_hi"]
        assert res["p_value"] > 0.05

    def test_residuals_are_full_length_and_finite(self, rng):
        y, d, X = _plr_dgp(rng, n=800)
        res = dml_partial_linear(y, d, X, n_folds=4,
                                 outcome_learner=LinearRegression(),
                                 exposure_learner=LinearRegression(), rng=rng)
        assert res["y_resid"].shape == (800,)
        assert res["d_resid"].shape == (800,)
        assert np.all(np.isfinite(res["y_resid"]))
        assert np.all(np.isfinite(res["d_resid"]))
        assert res["n"] == 800


class TestEValue:
    def test_larger_effect_gives_larger_e_value(self):
        small = e_value(estimate=0.05, se=0.02, y_sd=1.0, d_sd=1.0)
        large = e_value(estimate=0.5, se=0.02, y_sd=1.0, d_sd=1.0)
        assert large["e_value"] > small["e_value"]

    def test_null_estimate_gives_e_value_near_one(self):
        res = e_value(estimate=0.0, se=0.1, y_sd=1.0, d_sd=1.0)
        assert res["e_value"] == pytest.approx(1.0, abs=1e-6)

    def test_e_value_at_least_one(self, rng):
        for _ in range(20):
            est = rng.uniform(-2, 2)
            se = rng.uniform(0.01, 1.0)
            res = e_value(estimate=est, se=se, y_sd=1.0, d_sd=1.0)
            assert res["e_value"] >= 1.0
            assert res["e_value_ci"] >= 1.0

    def test_ci_e_value_not_larger_than_point(self, rng):
        # The CI-edge E-value (closer to the null) should never exceed the
        # point-estimate E-value.
        for _ in range(20):
            est = rng.uniform(0.1, 2)
            se = rng.uniform(0.01, 0.3)
            res = e_value(estimate=est, se=se, y_sd=1.0, d_sd=1.0)
            assert res["e_value_ci"] <= res["e_value"] + 1e-9
