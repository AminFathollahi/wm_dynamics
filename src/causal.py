"""
causal.py — Doubly-robust causal effect estimation for the interventional arm.

The observational geometry arm (src/geometry.py, src/dynamics.py) fits the plant
and derives predictions about WHEN, WHERE, and in WHICH DIRECTION a stimulation
input should most change the working-memory state. This module tests those
predictions on the interventional datasets (delay-period stimulation) as
geometry-conditioned treatment effects.

The central, falsifiable prediction is a HETEROGENEITY statement, not an average
one: the effect of stimulation should grow with the alignment of the stimulation
input to the unstable eigenvector v* (and during a contracting-flow window). So
the workhorse here is not an ATE but a CATE estimated against a theory-specified
geometric modifier. We use the doubly-robust (AIPW) pseudo-outcome so the
estimate stays valid if either the propensity or the outcome model is right, and
cross-fitting so flexible nuisance learners do not bias the second stage
(Chernozhukov 2018; Kennedy 2023).

Where treatment is experimentally assigned (macaque PFC microstimulation, RAM
open-loop), the propensity is KNOWN — pass it in via `propensity=` rather than
estimating it; that puts the doubly-robust estimator in its ideal regime with
guaranteed overlap.

References
----------
Chernozhukov V et al. (2018) Double/debiased machine learning. Econom. J. 21(1).
Kennedy EH (2023) Towards optimal doubly robust estimation of heterogeneous
  causal effects. Electron. J. Stat. 17(2).
"""

from __future__ import annotations

import sys
import os
import numpy as np
from numpy.typing import NDArray

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from statistics import permutation_pvalue


def _default_outcome_learner():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(random_state=0)


def _default_propensity_learner():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000)


def crossfit_nuisances(
    y: NDArray,
    t: NDArray,
    X: NDArray,
    n_folds: int = 5,
    propensity: NDArray | None = None,
    outcome_learner=None,
    propensity_learner=None,
    clip: float = 1e-2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Cross-fitted nuisance estimates e(x)=P(T=1|x), mu0(x)=E[Y|T=0,x], mu1(x).

    Each observation's nuisances are predicted by a model trained on the OTHER
    folds only, so the second-stage estimator inherits no overfitting bias from
    the first stage (the reason cross-fitting is required for valid DML/DR
    inference with flexible learners).

    Parameters
    ----------
    y, t, X    : outcome (N,), binary treatment (N,), features (N, d)
    propensity : (N,) known e(x) for a randomised/assigned design; if given, it
                 is used directly (only the outcome models are cross-fitted)
    clip       : propensity is clipped to [clip, 1-clip] to keep IPW weights finite

    Returns
    -------
    dict: e_hat, mu0_hat, mu1_hat — each (N,)
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    if rng is None:
        rng = np.random.default_rng(0)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    N = len(y)

    outcome_learner = outcome_learner or _default_outcome_learner()
    propensity_learner = propensity_learner or _default_propensity_learner()

    e_hat = np.full(N, np.nan)
    mu0_hat = np.full(N, np.nan)
    mu1_hat = np.full(N, np.nan)

    seed = int(rng.integers(0, 2**31 - 1))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in kf.split(X):
        tr_t = t[train]

        if propensity is None:
            if len(np.unique(tr_t)) < 2:
                e_hat[test] = tr_t.mean()
            else:
                pm = clone(propensity_learner).fit(X[train], tr_t)
                e_hat[test] = pm.predict_proba(X[test])[:, 1]

        tr0 = train[tr_t == 0]
        tr1 = train[tr_t == 1]
        m0 = clone(outcome_learner).fit(X[tr0], y[tr0])
        m1 = clone(outcome_learner).fit(X[tr1], y[tr1])
        mu0_hat[test] = m0.predict(X[test])
        mu1_hat[test] = m1.predict(X[test])

    if propensity is not None:
        e_hat = np.asarray(propensity, dtype=float)

    e_hat = np.clip(e_hat, clip, 1 - clip)
    return {"e_hat": e_hat, "mu0_hat": mu0_hat, "mu1_hat": mu1_hat}


def aipw_pseudo_outcome(
    y: NDArray, t: NDArray, e_hat: NDArray, mu0_hat: NDArray, mu1_hat: NDArray
) -> NDArray:
    """Doubly-robust (AIPW) pseudo-outcome whose conditional mean given X is the
    CATE tau(x)=E[Y(1)-Y(0)|x], and whose overall mean is the ATE.

        phi = (mu1 - mu0)
              + t/e       * (y - mu1)
              - (1-t)/(1-e) * (y - mu0)
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    return (
        (mu1_hat - mu0_hat)
        + t / e_hat * (y - mu1_hat)
        - (1 - t) / (1 - e_hat) * (y - mu0_hat)
    )


def _bootstrap_pvalue(
    stat_fn, n: int, n_boot: int, rng: np.random.Generator
) -> dict:
    """Nonparametric bootstrap CI + two-sided p-value for a statistic that is
    zero under the null (ATE, DML theta, ...), used instead of a normal-theory
    Wald test so validity does not depend on enough independent units for the
    CLT to hold -- exactly what breaks with the small session/subject counts
    common across these datasets (a normal-theory SE from ~5 cross-fit folds
    can produce implausible p-values like 1e-193 that are a small-n artifact,
    not evidence). p-value via the fraction of the bootstrap distribution
    crossing zero (Davison & Hinkley 1997 percentile-based test).

    stat_fn(idx) : recompute the statistic on a resampled index array.
    """
    boot = np.array([stat_fn(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(permutation_pvalue(boot <= 0), permutation_pvalue(boot >= 0))
    return {"ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": float(min(p, 1.0)),
            "se": float(np.std(boot, ddof=1))}


def _analytic_wald(point: float, se: float) -> dict:
    """Normal-theory Wald CI/p-value, reported ALONGSIDE the bootstrap result
    (not instead of it) -- when the unit count is large enough for the CLT to
    hold this should closely match the bootstrap, and the two are shown
    side-by-side (same dual-reporting convention as the trial-level-vs-
    cluster-robust comparison in run_macaque_pfc_microstimulation_headline_robustness.py) so a
    reader can see agreement or, at small n, the analytic test's known
    unreliability (e.g. implausible p~1e-193 from ~5 cross-fit folds).
    """
    from scipy.stats import norm

    z = point / se if se > 0 else float("inf")
    return {"se": float(se), "ci_lo": float(point - 1.96 * se), "ci_hi": float(point + 1.96 * se),
            "p_value": float(2.0 * norm.sf(abs(z)))}


def aipw_ate(
    y: NDArray,
    t: NDArray,
    X: NDArray,
    n_folds: int = 5,
    propensity: NDArray | None = None,
    outcome_learner=None,
    propensity_learner=None,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Cross-fitted doubly-robust average treatment effect (AIPW).

    Returns
    -------
    dict: ate, se, ci_lo, ci_hi, p_value -- primary, bootstrap (see
          _bootstrap_pvalue); se_analytic, ci_lo_analytic, ci_hi_analytic,
          p_value_analytic -- normal-theory Wald, reported alongside for
          comparison (see _analytic_wald); n
    """
    if rng is None:
        rng = np.random.default_rng(0)

    nu = crossfit_nuisances(
        y, t, X, n_folds=n_folds, propensity=propensity,
        outcome_learner=outcome_learner, propensity_learner=propensity_learner, rng=rng,
    )
    phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])
    ate = float(np.mean(phi))
    n = len(phi)
    boot_res = _bootstrap_pvalue(lambda idx: float(phi[idx].mean()), n, n_boot, rng)
    analytic = _analytic_wald(ate, float(np.std(phi, ddof=1) / np.sqrt(n)))
    return {
        "ate": ate,
        "se": boot_res["se"],
        "ci_lo": boot_res["ci_lo"],
        "ci_hi": boot_res["ci_hi"],
        "p_value": boot_res["p_value"],
        "se_analytic": analytic["se"],
        "ci_lo_analytic": analytic["ci_lo"],
        "ci_hi_analytic": analytic["ci_hi"],
        "p_value_analytic": analytic["p_value"],
        "n": int(n),
    }


def _dr_slope(phi: NDArray, modifier: NDArray, n_perm: int, rng: np.random.Generator) -> dict:
    """Shared core: OLS slope of a (already finite-filtered) doubly-robust
    pseudo-outcome on ONE modifier, with a permutation p-value (permuting the
    modifier, which breaks any modifier-effect relationship while preserving
    the pseudo-outcome distribution — so significance does not lean on the
    OLS normal-error assumption) and a bootstrap CI. Used by both
    cate_vs_modifier_slope (raw-scale, single modifier) and
    benchmark_modifiers (z-scored, one call per candidate modifier) so the
    two never fork.

    When the modifier has ~zero variance over the rows handed in (no rows at
    all, or every row sharing one constant value), the closed-form slope has
    no defined denominator; `_fit` below falls back to slope=0.0 so the
    caller's arithmetic never divides by zero, but that 0.0 is a numerical
    guard firing, not a measured null. The returned `degenerate` flag says
    which one happened so a caller can tell a real zero-slope fit apart from
    "there was nothing to fit" before serializing either into a result.
    """
    n = len(phi)

    def _fit(mm: NDArray, pp: NDArray) -> tuple[float, float]:
        mc = mm - mm.mean()
        denom = float((mc**2).sum())
        if denom < 1e-15:
            return 0.0, float(pp.mean())
        b = float((mc * (pp - pp.mean())).sum() / denom)
        a = float(pp.mean() - b * mm.mean())
        return b, a

    degenerate = n == 0 or float(((modifier - modifier.mean()) ** 2).sum()) < 1e-15

    slope, intercept = _fit(modifier, phi)

    null = np.array([_fit(rng.permutation(modifier), phi)[0] for _ in range(n_perm)])
    p = permutation_pvalue(np.abs(null) >= abs(slope))

    boot = np.empty(2000)
    for i in range(2000):
        idx = rng.integers(0, n, size=n)
        boot[i] = _fit(modifier[idx], phi[idx])[0]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_ci_lo": float(ci_lo),
        "slope_ci_hi": float(ci_hi),
        "p_value": p,
        "ate": float(phi.mean()),
        "n": int(n),
        "null": null,
        "degenerate": bool(degenerate),
        "status": "no_regressor_variance" if degenerate else "fitted",
    }


def cate_vs_modifier_slope(
    y: NDArray,
    t: NDArray,
    X: NDArray,
    modifier: NDArray,
    n_folds: int = 5,
    propensity: NDArray | None = None,
    outcome_learner=None,
    propensity_learner=None,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Test whether the treatment effect varies with a single, theory-specified
    geometric modifier (e.g. alignment of the stimulation input to v*).

    This is the small-N-safe form of the DR-Learner: instead of estimating a
    full nonparametric CATE surface (data-hungry, and easy to over-fit / fish
    on a handful of stimulated sites), we cross-fit the doubly-robust
    pseudo-outcome and regress it linearly on ONE pre-registered modifier. The
    framework predicts a POSITIVE slope. See `_dr_slope` for the shared core
    (also used by `benchmark_modifiers`, this function's multi-modifier
    generalization); this is benchmark_modifiers with one modifier, on the
    raw (non-z-scored) scale.

    Parameters
    ----------
    modifier : (N,) the pre-specified effect modifier (e.g. |cos(B, v*)|)

    Returns
    -------
    dict: slope, intercept, slope_ci_lo, slope_ci_hi (bootstrap), p_value
          (permutation, two-sided), ate (mean pseudo-outcome), n
    """
    if rng is None:
        rng = np.random.default_rng(0)
    m = np.asarray(modifier, dtype=float)

    nu = crossfit_nuisances(
        y, t, X, n_folds=n_folds, propensity=propensity,
        outcome_learner=outcome_learner, propensity_learner=propensity_learner, rng=rng,
    )
    phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])

    finite = np.isfinite(phi) & np.isfinite(m)
    phi, m = phi[finite], m[finite]

    result = _dr_slope(phi, m, n_perm, rng)
    result["phi"] = phi
    result["modifier"] = m
    return result


def _ols_fit(Zcols: NDArray, y: NDArray) -> NDArray:
    """OLS coefficients (intercept first) for y ~ 1 + Zcols."""
    Zc = np.column_stack([np.ones(len(y)), Zcols])
    beta, *_ = np.linalg.lstsq(Zc, y, rcond=None)
    return beta


def _r2(Zcols: NDArray, y: NDArray) -> float:
    beta = _ols_fit(Zcols, y)
    Zc = np.column_stack([np.ones(len(y)), Zcols])
    pred = Zc @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def benchmark_modifiers(
    y: NDArray,
    t: NDArray,
    X: NDArray,
    modifiers: dict[str, NDArray],
    n_folds: int = 5,
    propensity: NDArray | None = None,
    outcome_learner=None,
    propensity_learner=None,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Falsifiable model comparison (Brain-Score logic): score several
    candidate theories of the causally-relevant controllable direction
    (e.g. the unstable-mode alignment vs. anatomical controllability vs. the
    stable-context direction vs. random directions vs. raw coupling
    magnitude) against ONE held-out causal criterion — the SAME cross-fit
    doubly-robust pseudo-outcome, computed ONCE so every modifier is scored
    on identical nuisance estimates. Each modifier is z-scored so slopes are
    per-SD and comparable across modifiers with different native scales.

    Any one candidate is exactly `cate_vs_modifier_slope` (see `_dr_slope`,
    the shared core); this generalizes it to a leaderboard, a joint model,
    and a nested-comparison falsification test.

    Parameters
    ----------
    modifiers : {name: (N,) array} candidate effect modifiers

    Returns
    -------
    dict:
      leaderboard : {name: {slope, slope_ci_lo, slope_ci_hi, p_value, n}} —
        per-SD marginal DR-interaction slope for each modifier alone.
      joint       : {name: {coef, p_value}} — partial coefficient (and
        permutation p) when ALL modifiers are entered together; the theory's
        a-priori prediction is that only the winning modifier survives.
      nested      : {name: {dR2, p_value}} — for each non-winning modifier,
        the R^2 gain from adding it to a model already containing the
        marginal winner (permutation p on that gain); the a-priori
        prediction is no reliable gain.
      winner : name of the largest POSITIVE marginal slope (None if no
        modifier has a positive slope).
      n : rows used (after dropping any row non-finite in phi or ANY modifier).
      excluded : {name: {eligible: False, reason}} — modifiers with ~zero
        within-arena variance (np.nanstd < 1e-12 over the finite rows), e.g. a
        constant scalar broadcast to every row. Regressing the pseudo-outcome
        on a constant gives a fake slope==0/CI==[0,0]/p==1.0 that reads as a
        "trustworthy null" rather than a structurally-undefined competitor, so
        these are excluded from leaderboard/joint/nested/z_modifiers entirely
        rather than silently scored.
      phi, z_modifiers : the cross-fit pseudo-outcome and {name: z-scored
        modifier} actually used above — not a new statistical claim (the
        leaderboard/joint/nested numbers already are), kept only so a caller
        can plot the real scatter/leaderboard instead of a schematic (same
        role as cate_vs_modifier_slope's phi/modifier/null fields).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    m_raw = {k: np.asarray(v, dtype=float) for k, v in modifiers.items()}

    nu = crossfit_nuisances(
        y, t, X, n_folds=n_folds, propensity=propensity,
        outcome_learner=outcome_learner, propensity_learner=propensity_learner, rng=rng,
    )
    phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])

    finite = np.isfinite(phi)
    for v in m_raw.values():
        finite &= np.isfinite(v)
    phi = phi[finite]
    n = int(len(phi))

    excluded = {}
    names = []
    mz = {}
    for k, v in m_raw.items():
        vf = v[finite]
        sd = float(np.nanstd(vf))
        if sd < 1e-12:
            excluded[k] = {"eligible": False, "reason": "constant modifier (zero within-arena variance)"}
            continue
        names.append(k)
        mz[k] = (vf - vf.mean()) / sd

    if not names:
        # Every candidate modifier was constant on this row set -- nothing to
        # benchmark; report that plainly instead of crashing on an empty stack.
        return {"leaderboard": {}, "joint": {}, "nested": {}, "winner": None, "n": n,
                "excluded": excluded, "phi": phi, "z_modifiers": {},
                "status": "no_eligible_modifiers"}

    leaderboard = {}
    for k in names:
        r = _dr_slope(phi, mz[k], n_perm, rng)
        leaderboard[k] = {"slope": r["slope"], "slope_ci_lo": r["slope_ci_lo"],
                          "slope_ci_hi": r["slope_ci_hi"], "p_value": r["p_value"], "n": r["n"]}

    positive = {k: v["slope"] for k, v in leaderboard.items() if v["slope"] > 0}
    winner = max(positive, key=positive.get) if positive else None

    z_mat = np.column_stack([mz[k] for k in names])
    joint = {}
    beta_full = _ols_fit(z_mat, phi)
    coefs = beta_full[1:]
    for i, k in enumerate(names):
        obs = float(coefs[i])
        null = np.empty(n_perm)
        z_perm = z_mat.copy()
        col = z_mat[:, i].copy()
        for b in range(n_perm):
            z_perm[:, i] = rng.permutation(col)
            null[b] = _ols_fit(z_perm, phi)[1 + i]
        p = permutation_pvalue(np.abs(null) >= abs(obs))
        joint[k] = {"coef": obs, "p_value": p}

    nested = {}
    if winner is not None:
        base_r2 = _r2(mz[winner][:, None], phi)
        for k in names:
            if k == winner:
                continue
            full_r2 = _r2(np.column_stack([mz[winner], mz[k]]), phi)
            dr2_obs = full_r2 - base_r2
            null_dr2 = np.empty(n_perm)
            for b in range(n_perm):
                perm_k = rng.permutation(mz[k])
                null_dr2[b] = _r2(np.column_stack([mz[winner], perm_k]), phi) - base_r2
            p = permutation_pvalue(null_dr2 >= dr2_obs)
            nested[k] = {"dR2": float(dr2_obs), "p_value": p}

    return {"leaderboard": leaderboard, "joint": joint, "nested": nested, "winner": winner, "n": n,
            "excluded": excluded, "phi": phi, "z_modifiers": mz}


def dr_learner_cate(
    y: NDArray,
    t: NDArray,
    X: NDArray,
    modifiers: NDArray,
    n_folds: int = 5,
    propensity: NDArray | None = None,
    outcome_learner=None,
    propensity_learner=None,
    cate_learner=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Full DR-Learner: cross-fit the doubly-robust pseudo-outcome, then fit a
    second-stage learner of tau(m) on the effect modifiers.

    Use this only where N supports a flexible surface (e.g. macaque PFC microstimulation trial-level,
    RAM thousands of events). For small interventional sets, prefer
    cate_vs_modifier_slope with a single theory-specified modifier.

    Returns
    -------
    dict: model (fitted second stage), cate (in-sample predictions, N,),
          pseudo_outcome (N,)
    """
    from sklearn.base import clone

    M = np.asarray(modifiers, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    cate_learner = cate_learner or _default_outcome_learner()

    nu = crossfit_nuisances(
        y, t, X, n_folds=n_folds, propensity=propensity,
        outcome_learner=outcome_learner, propensity_learner=propensity_learner, rng=rng,
    )
    phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])

    model = clone(cate_learner).fit(M, phi)
    return {"model": model, "cate": model.predict(M), "pseudo_outcome": phi}


def dml_partial_linear(
    y: NDArray,
    d: NDArray,
    X: NDArray,
    n_folds: int = 5,
    outcome_learner=None,
    exposure_learner=None,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Double/debiased ML estimate of a CONTINUOUS exposure's effect (Chernozhukov
    et al. 2018 partially-linear model): Y = theta*D + g(X) + U, D = m(X) + V.

    Unlike cate_vs_modifier_slope/aipw_ate (which need a BINARY, typically
    experimentally-assigned treatment), this is for observational data where
    the "exposure" is a continuous measured quantity -- e.g. Q6's maintenance-
    state metric (drift from centroid, decoder confidence) predicting behavior
    -- and confounders (set size, RT, session, unit count) must be partialled
    out non-parametrically. This is a causal-FLAVORED association: there is no
    experimental manipulation of D, so theta is only a valid causal effect
    under the (untestable) assumption that X captures all confounding of the
    D->Y relationship. Always accompany with e_value() to quantify how strong
    unmeasured confounding would have to be to explain away the estimate away.

    Cross-fitting: g_hat and m_hat are each fit on K-1 folds and used to
    residualize the held-out fold, so flexible ML nuisance fits do not bias
    the second-stage theta (same rationale as crossfit_nuisances).

    Parameters
    ----------
    y, d, X : outcome (N,), continuous exposure (N,), confounders (N, p)

    Returns
    -------
    dict: theta (effect of one unit of D on Y, controlling for X), se, ci_lo,
          ci_hi, p_value -- primary, bootstrap (see _bootstrap_pvalue);
          se_analytic, ci_lo_analytic, ci_hi_analytic, p_value_analytic --
          DML1/PLR asymptotic-variance Wald test, reported alongside for
          comparison (see _analytic_wald) -- at the small per-fold n typical
          here (n<~10) this analytic test is a known unreliable artifact
          (e.g. p~1e-193), so treat the bootstrap fields as authoritative and
          the analytic ones as a labeled reference point, not confirmatory;
          y_resid, d_resid (for diagnostics/plotting), n
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    if rng is None:
        rng = np.random.default_rng(0)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)

    outcome_learner = outcome_learner or _default_outcome_learner()
    exposure_learner = exposure_learner or _default_outcome_learner()

    y_resid = np.full(n, np.nan)
    d_resid = np.full(n, np.nan)
    seed = int(rng.integers(0, 2**31 - 1))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in kf.split(X):
        g_hat = clone(outcome_learner).fit(X[train], y[train])
        m_hat = clone(exposure_learner).fit(X[train], d[train])
        y_resid[test] = y[test] - g_hat.predict(X[test])
        d_resid[test] = d[test] - m_hat.predict(X[test])

    def _theta(dd: NDArray, yy: NDArray) -> float:
        dn = float((dd**2).sum())
        return float((dd * yy).sum() / dn) if dn > 1e-15 else 0.0

    theta = _theta(d_resid, y_resid)
    boot_res = _bootstrap_pvalue(
        lambda idx: _theta(d_resid[idx], y_resid[idx]), n, n_boot, rng
    )

    u_resid = y_resid - theta * d_resid
    ev2 = float(np.mean(d_resid**2))
    eu2v2 = float(np.mean((d_resid**2) * (u_resid**2)))
    se_analytic_val = float(np.sqrt(eu2v2 / (n * ev2**2))) if ev2 > 1e-15 else float("inf")
    analytic = _analytic_wald(theta, se_analytic_val)

    return {
        "theta": theta,
        "se": boot_res["se"],
        "ci_lo": boot_res["ci_lo"],
        "ci_hi": boot_res["ci_hi"],
        "p_value": boot_res["p_value"],
        "se_analytic": analytic["se"],
        "ci_lo_analytic": analytic["ci_lo"],
        "ci_hi_analytic": analytic["ci_hi"],
        "p_value_analytic": analytic["p_value"],
        "y_resid": y_resid,
        "d_resid": d_resid,
        "n": int(n),
    }


def e_value(estimate: float, se: float, y_sd: float, d_sd: float) -> dict:
    """E-value (VanderWeele & Ding 2017): how strong an unmeasured confounder's
    association with BOTH the exposure and the outcome would need to be to
    fully explain away an observed (non-experimental) effect estimate.

    Converts the continuous DML coefficient to an approximate risk-ratio scale
    via the standardized effect size (theta*d_sd/y_sd), then applies VanderWeele
    & Ding's RR->E-value formula. A large E-value (e.g. >2) means only a strong
    unmeasured confounder could null the result; an E-value near 1 means even
    weak unmeasured confounding could. Report alongside every DML estimate used
    causally: this project's standing rule that every DML estimate carries a
    mandatory sensitivity analysis.

    Parameters
    ----------
    estimate : theta from dml_partial_linear (or any linear coefficient)
    se       : its standard error
    y_sd     : SD of the outcome (for standardizing to an effect size)
    d_sd     : SD of the exposure

    Returns
    -------
    dict: rr_approx (approximate risk ratio), e_value (for the point estimate),
          e_value_ci (for the CI bound closer to the null)
    """
    def _rr_from_smd(smd: float) -> float:
        # VanderWeele & Ding 2017 eq 2: OR ~= exp(0.91 * SMD) for a
        # standardized mean difference; approximate RR by the OR for a
        # not-too-common outcome (standard approximation in this literature).
        return float(np.exp(0.91 * abs(smd)))

    def _ev_from_rr(rr: float) -> float:
        if rr <= 1.0:
            return 1.0
        return float(rr + np.sqrt(rr * (rr - 1.0)))

    smd = estimate * d_sd / (y_sd + 1e-15)
    rr = _rr_from_smd(smd)
    ev_point = _ev_from_rr(rr)

    ci_bound = estimate - np.sign(estimate) * 1.96 * se  # CI edge nearer the null
    smd_ci = ci_bound * d_sd / (y_sd + 1e-15)
    rr_ci = _rr_from_smd(smd_ci) if np.sign(ci_bound) == np.sign(estimate) else 1.0
    ev_ci = _ev_from_rr(rr_ci)

    return {"rr_approx": rr, "e_value": ev_point, "e_value_ci": ev_ci}
