"""
statistics.py — Permutation tests, bootstrap CI, and predictive modelling.

Publication-grade statistical inference for neural geometry biomarkers.

All tests are non-parametric (permutation or bootstrap) to avoid distributional
assumptions on neural data. Effect sizes and confidence intervals are always
reported alongside p-values (following Lakens 2013; Wasserstein & Lazar 2016).

References
----------
Maris E & Oostenveld R (2007) Nonparametric statistical testing of EEG- and
  MEG-data. J Neurosci Methods 164(1):177-90.
Efron B & Tibshirani RJ (1993) An Introduction to the Bootstrap. Chapman & Hall.
Laber EB et al. — scikit-learn's cross_val_score for AUROC.
"""

from __future__ import annotations

import zlib
import numpy as np
from numpy.typing import NDArray
from typing import Callable


def stable_seed(name: str) -> int:
    """Deterministic RNG seed from a string, stable across processes.

    Python's built-in hash() is salted per-process (PEP 456) unless
    PYTHONHASHSEED is fixed, so seeding np.random.default_rng(hash(name))
    silently makes every pipeline run non-reproducible. crc32 is stable.
    """
    return zlib.crc32(name.encode("utf-8"))


def permutation_pvalue(exceed_mask: NDArray) -> float:
    """Monte Carlo permutation p-value with the standard +1/+1 correction.

    p = (#{null at least as extreme as observed} + 1) / (n_perm + 1).
    The observed statistic is itself one exchangeable draw under H0, so it
    belongs in both the numerator and denominator (Davison & Hinkley 1997;
    North, Curtis & Sham 2002). Without the +1 correction, a finite Monte
    Carlo test can report p=0.0 exactly, which is never a valid claim (the
    true p can only be bounded below by 1/(n_perm+1)).

    Parameters
    ----------
    exceed_mask : (n_perm,) bool — whether each null draw is >= (or <=, or
        |null|>=|obs|, depending on the caller's tail) the observed statistic.
    """
    exceed_mask = np.asarray(exceed_mask)
    return float((exceed_mask.sum() + 1) / (len(exceed_mask) + 1))


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: NDArray,
    stat_fn: Callable[[NDArray], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    Parameters
    ----------
    data    : 1D or 2D array; resampling is along axis 0
    stat_fn : function mapping data → scalar statistic
    n_boot  : number of bootstrap resamples
    ci      : confidence level (0.95 = 95%)

    Returns
    -------
    (stat, lower, upper) — observed statistic and CI bounds
    """
    if rng is None:
        rng = np.random.default_rng(0)

    observed = stat_fn(data)
    n = len(data)
    boot_stats = np.array(
        [stat_fn(data[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lower, upper = np.nanpercentile(boot_stats, [100 * alpha, 100 * (1 - alpha)])
    return float(observed), float(lower), float(upper)


def bootstrap_ci_timecourse(
    data: NDArray,
    stat_fn: Callable[[NDArray], NDArray],
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Bootstrap CI for a time-series statistic.

    Parameters
    ----------
    data    : (N, T) — N observations, T time points
    stat_fn : function mapping (N, T) → (T,) statistic (e.g. np.mean(axis=0))

    Returns
    -------
    (observed, lower, upper) each (T,)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    observed = stat_fn(data)
    T = len(observed)
    n = data.shape[0]
    boot = np.array(
        [stat_fn(data[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lower = np.nanpercentile(boot, 100 * alpha, axis=0)
    upper = np.nanpercentile(boot, 100 * (1 - alpha), axis=0)
    return observed, lower, upper


# ── Permutation tests ──────────────────────────────────────────────────────────

def permutation_test_twosample(
    x: NDArray,
    y: NDArray,
    stat_fn: Callable[[NDArray, NDArray], float] | None = None,
    n_perm: int = 5000,
    alternative: str = "two-sided",
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Permutation test for two-sample comparison.

    Under H₀: labels are exchangeable (observations drawn from the same
    distribution). Shuffles labels n_perm times, computes the statistic
    each time to build the null distribution.

    Parameters
    ----------
    stat_fn     : function mapping (x, y) → scalar; defaults to mean difference
    alternative : 'two-sided', 'greater', 'less'

    Returns
    -------
    (observed_stat, p_value)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if stat_fn is None:
        stat_fn = lambda a, b: np.mean(a) - np.mean(b)

    pooled = np.concatenate([x, y])
    n_x = len(x)
    observed = stat_fn(x, y)

    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        null[i] = stat_fn(pooled[perm[:n_x]], pooled[perm[n_x:]])

    if alternative == "two-sided":
        p = permutation_pvalue(np.abs(null) >= np.abs(observed))
    elif alternative == "greater":
        p = permutation_pvalue(null >= observed)
    else:
        p = permutation_pvalue(null <= observed)

    return float(observed), float(p)


def paired_sign_flip_test(
    a: NDArray,
    b: NDArray,
    n_perm: int = 10000,
    alternative: str = "less",
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Within-subject sign-flip permutation test on paired differences (a - b).

    Under H0 (exchangeability within each pair), the sign of each subject's
    (a-b) difference is arbitrary, so the null is built by randomly flipping
    signs rather than shuffling across subjects (which would break pairing).
    Used for paired within-subject contrasts (e.g. content_tau vs
    context_tau, same subjects) where a two-sample permutation would be
    invalid.

    Parameters
    ----------
    a, b        : (n_subjects,) paired observations
    alternative : 'less' (a<b), 'greater' (a>b), or 'two-sided'

    Returns
    -------
    dict: mean_diff, p_value, ci_lower, ci_upper (percentile bootstrap on
          mean_diff), null (n_perm,)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    diffs = np.asarray(a) - np.asarray(b)
    if not np.all(np.isfinite(diffs)):
        raise ValueError(
            "paired_sign_flip_test received non-finite paired differences; "
            "NaN silently propagates through every permutation and can make "
            "an undefined comparison look like p=0.0. Filter non-finite "
            "pairs before calling."
        )
    n = len(diffs)
    observed = float(np.mean(diffs))

    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * diffs[None, :]).mean(axis=1)

    if alternative == "less":
        p = permutation_pvalue(null <= observed)
    elif alternative == "greater":
        p = permutation_pvalue(null >= observed)
    else:
        p = permutation_pvalue(np.abs(null) >= np.abs(observed))

    boot_means = np.array(
        [np.mean(diffs[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    )
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_diff": observed,
        "p_value": p,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "null": null,
        "n": int(n),
    }


def spearman_permutation_test(
    x: NDArray,
    y: NDArray,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Spearman rank correlation with a label-permutation null (small-N safe).

    Preferred over the asymptotic Spearman p-value for the small,
    cross-subject sample sizes typical of this project (e.g. correlating a
    per-subject contraction rate with a per-subject behavioral or geometric
    statistic across ~4-20 subjects), where the asymptotic approximation is
    unreliable.

    Returns
    -------
    dict: rho (observed Spearman correlation), p_value, null (n_perm,)
    """
    from scipy.stats import spearmanr

    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = float(spearmanr(x, y).statistic)

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(y)).statistic

    p = permutation_pvalue(np.abs(null) >= np.abs(obs))
    return {"rho": obs, "p_value": p, "null": null, "n": int(len(x))}


def pearson_permutation_test(
    x: NDArray,
    y: NDArray,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Pearson correlation with a label-permutation null (small-N safe),
    reported alongside scipy's analytic pearsonr for comparison -- the
    analytic p-value assumes bivariate normality and enough points for its
    t-distribution approximation to hold, which is unreliable for the small
    per-subject electrode counts typical of this project.

    Returns
    -------
    dict: r (observed Pearson correlation), p_value, r_analytic, p_analytic, n
    """
    from scipy.stats import pearsonr

    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs, p_analytic = pearsonr(x, y)
    obs = float(obs)

    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = pearsonr(x, rng.permutation(y))[0]

    p = permutation_pvalue(np.abs(null) >= np.abs(obs))
    return {"r": obs, "p_value": p, "r_analytic": obs, "p_analytic": float(p_analytic), "n": int(len(x))}


def partial_correlation_permutation_test(
    outcome: NDArray,
    covariate: NDArray,
    controls: list[NDArray] | tuple[NDArray, ...] = (),
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Partial Pearson correlation between ``outcome`` and ``covariate``,
    controlling for zero or more ``controls``, by the residual method:
    regress both ``outcome`` and ``covariate`` on [intercept, *controls] by
    ordinary least squares, then Pearson-correlate the two residual series.
    With no controls this is the ordinary Pearson correlation (and, for a
    0/1 ``outcome``, the ordinary point-biserial correlation -- scipy's
    pointbiserialr is itself a Pearson correlation on a binary variable, so
    no separate binary-specific code path is needed).

    Significance is a permutation test that shuffles ``outcome`` only,
    leaving ``covariate`` and every control fixed: this breaks any
    relationship ``outcome`` has with ``covariate`` or the controls while
    preserving the covariate structure the partial correlation is
    conditioning on, which is the appropriate null for "does outcome relate
    to covariate after conditioning on controls" (Kennedy 1995, one of
    several valid permutation schemes for a partial correlation). An
    analytic reference p-value (Student's t on the partial correlation,
    df = n - 2 - len(controls)) is reported alongside for comparison, per
    this module's standing rule against reporting only the small-sample-
    unsafe analytic figure.

    Returns
    -------
    dict: r (observed partial correlation), p_value (permutation, two-
    sided), r_analytic (same r, named for symmetry with the other
    permutation tests in this module), p_analytic, n, n_controls.
    """
    from scipy.stats import t as t_distribution

    if rng is None:
        rng = np.random.default_rng(0)
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    control_list = [np.asarray(c, dtype=float) for c in controls]
    n = len(outcome)

    def _partial_r(y: NDArray) -> float:
        if control_list:
            design = np.column_stack([np.ones(n), *control_list])
            y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
            x_resid = covariate - design @ np.linalg.lstsq(design, covariate, rcond=None)[0]
        else:
            y_resid, x_resid = y - y.mean(), covariate - covariate.mean()
        if np.std(y_resid) == 0.0 or np.std(x_resid) == 0.0:
            return float("nan")
        return float(np.corrcoef(y_resid, x_resid)[0, 1])

    obs = _partial_r(outcome)
    null = np.array([_partial_r(rng.permutation(outcome)) for _ in range(n_perm)])
    if np.isnan(obs):
        return {"status": "not_computable", "reason": "zero-variance residual", "n": int(n), "n_controls": len(control_list)}
    p = permutation_pvalue(np.abs(null[~np.isnan(null)]) >= np.abs(obs))

    df = n - 2 - len(control_list)
    if df > 0 and abs(obs) < 1.0:
        t_stat = obs * np.sqrt(df / (1.0 - obs ** 2))
        p_analytic = float(2.0 * t_distribution.sf(np.abs(t_stat), df))
    else:
        p_analytic = float("nan")

    return {"status": "computed", "r": obs, "p_value": p, "r_analytic": obs, "p_analytic": p_analytic,
            "n": int(n), "n_controls": len(control_list)}


def temporal_cluster_permutation(
    x_time: NDArray,
    y_time: NDArray,
    times: NDArray,
    n_perm: int = 2000,
    alpha_threshold: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Cluster-based permutation test for time-series comparisons (Maris & Oostenveld 2007).

    Identifies temporal clusters where two conditions differ significantly,
    while controlling the family-wise error rate across time points.

    Algorithm:
      1. Compute point-wise t-statistics: t(t) = (mean_x - mean_y) / pooled_SE
      2. Threshold at ±t_{alpha/2}: identify contiguous clusters above threshold
      3. Cluster statistic = sum of t-values within cluster
      4. Permute labels n_perm times, find max cluster stat each time → null distribution
      5. Observed cluster is significant if its stat > 95th percentile of null

    Parameters
    ----------
    x_time : (N_x, T)
    y_time : (N_y, T)
    times  : (T,) in seconds

    Returns
    -------
    dict:
      t_stat       : (T,) observed t-statistics
      clusters     : list of (start_idx, end_idx, cluster_stat, p_value)
      significant  : list of significant cluster time ranges
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = x_time.shape[1]
    nx, ny = len(x_time), len(y_time)
    pooled_time = np.vstack([x_time, y_time])  # (N, T)

    def _tstat(a, b):
        mu_a, mu_b = a.mean(0), b.mean(0)
        se = np.sqrt(a.var(0) / len(a) + b.var(0) / len(b) + 1e-15)
        return (mu_a - mu_b) / se

    def _clusters(t_vec, threshold):
        """Return sign-homogeneous supra-threshold runs.

        Opposite-sign neighbouring samples must never be merged: summing them
        cancels the cluster mass and invalidates the max-cluster null.
        """
        clusters = []
        for sign, active in ((1, t_vec > threshold), (-1, t_vec < -threshold)):
            start = None
            for i, is_active in enumerate(active):
                if is_active and start is None:
                    start = i
                elif not is_active and start is not None:
                    clusters.append((start, i, sign))
                    start = None
            if start is not None:
                clusters.append((start, T, sign))
        return sorted(clusters, key=lambda item: item[0])

    # Critical t-value (df ≈ nx + ny - 2, two-tailed)
    from scipy.stats import t as t_dist
    df = nx + ny - 2
    t_crit = t_dist.ppf(1 - alpha_threshold / 2, df)

    t_obs = _tstat(x_time, y_time)
    obs_clusters = _clusters(t_obs, t_crit)

    if not obs_clusters:
        return {
            "t_stat": t_obs,
            "clusters": [],
            "significant": [],
            "times": times,
        }

    obs_cluster_stats = [abs(t_obs[s:e].sum()) for s, e, _ in obs_clusters]

    # Null distribution
    null_max_stats = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(nx + ny)
        a_p = pooled_time[perm[:nx]]
        b_p = pooled_time[perm[nx:]]
        t_p = _tstat(a_p, b_p)
        clust_p = _clusters(t_p, t_crit)
        if clust_p:
            null_max_stats[i] = max(abs(t_p[s:e].sum()) for s, e, _ in clust_p)

    results_clusters = []
    for (s, e, sign), stat in zip(obs_clusters, obs_cluster_stats):
        p = permutation_pvalue(null_max_stats >= stat)
        results_clusters.append({
            "start_s": float(times[s]),
            "end_s": float(times[e - 1]),
            "cluster_stat": float(sign * stat),
            "sign": "positive" if sign > 0 else "negative",
            "p_value": float(p),
        })

    significant = [c for c in results_clusters if c["p_value"] < alpha_threshold]

    return {
        "t_stat": t_obs,
        "clusters": results_clusters,
        "significant": significant,
        "times": times,
        "t_critical": float(t_crit),
    }


def temporal_cluster_permutation_auroc(
    scores_time: NDArray,
    outcome: NDArray,
    times: NDArray,
    n_perm: int = 2000,
    alpha_threshold: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Cluster-based permutation test (Maris & Oostenveld 2007) using AUROC as
    the per-timepoint statistic, for a single continuous score observed at
    every timepoint together with one binary per-trial outcome label.

    Same algorithm as temporal_cluster_permutation, but suited to a
    within-trial-set design (e.g. decoder confidence at each timepoint versus
    correct/error outcome) rather than two separate groups: the null is built
    by permuting the outcome label across trials (as in
    geometry.ctg_label_permutation_null) rather than by resplitting trials
    into two groups. AUROC-0.5 deviation is used in place of a t-statistic so
    the effect size is on the same AUC-0.5 scale used throughout this project.

    Parameters
    ----------
    scores_time : (N, T) — per-trial score at each timepoint (e.g. decoder
        confidence in the trial's own true class)
    outcome     : (N,) — binary outcome label (e.g. correct/error)
    times       : (T,) in seconds

    Returns
    -------
    dict: auc_stat (T,) AUC-0.5 at each timepoint, clusters, significant,
          times, auc_threshold (the data-driven cluster-forming threshold)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N, T = scores_time.shape
    outcome = np.asarray(outcome)

    def _auc_minus_chance(t: int, y: NDArray) -> float:
        return auroc(y, scores_time[:, t]) - 0.5

    def _clusters(vec: NDArray, threshold: float) -> list[tuple[int, int, int]]:
        """Return sign-homogeneous supra-threshold runs.

        Opposite-sign neighbouring samples must never be merged: summing them
        cancels the cluster mass and invalidates the max-cluster null (see
        temporal_cluster_permutation, same fix).
        """
        clusters = []
        for sign, active in ((1, vec > threshold), (-1, vec < -threshold)):
            start = None
            for i, is_active in enumerate(active):
                if is_active and start is None:
                    start = i
                elif not is_active and start is not None:
                    clusters.append((start, i, sign))
                    start = None
            if start is not None:
                clusters.append((start, T, sign))
        return sorted(clusters, key=lambda item: item[0])

    auc_obs = np.array([_auc_minus_chance(t, outcome) for t in range(T)])

    null_auc = np.empty((n_perm, T))
    for p in range(n_perm):
        y_perm = rng.permutation(outcome)
        null_auc[p] = np.array([_auc_minus_chance(t, y_perm) for t in range(T)])

    auc_threshold = float(np.percentile(np.abs(null_auc), 100 * (1 - alpha_threshold)))
    obs_clusters = _clusters(auc_obs, auc_threshold)

    null_max_stats = np.array([
        max((abs(null_auc[p, s:e].sum()) for s, e, _ in _clusters(null_auc[p], auc_threshold)),
            default=0.0)
        for p in range(n_perm)
    ])

    if not obs_clusters:
        return {"auc_stat": auc_obs, "clusters": [], "significant": [], "times": times,
                "auc_threshold": auc_threshold}

    results_clusters = []
    for s, e, sign in obs_clusters:
        stat = abs(float(auc_obs[s:e].sum()))
        p_value = permutation_pvalue(null_max_stats >= stat)
        results_clusters.append({
            "start_s": float(times[s]), "end_s": float(times[e - 1]),
            "cluster_stat": float(sign * stat),
            "sign": "positive" if sign > 0 else "negative",
            "p_value": p_value,
        })

    significant = [c for c in results_clusters if c["p_value"] < alpha_threshold]

    return {"auc_stat": auc_obs, "clusters": results_clusters, "significant": significant,
            "times": times, "auc_threshold": auc_threshold}


def gated_outcome_cluster_test(
    confidence: NDArray,
    outcome: NDArray,
    times: NDArray,
    min_trials_per_outcome: int = 8,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """temporal_cluster_permutation_auroc, gated on a minimum trial count in
    each outcome group after dropping trials with a non-finite confidence
    value at any timepoint (e.g. a cross-validation fold that never held out
    that trial). Returns None rather than an unreliable result when either
    outcome group is underpowered.

    Parameters
    ----------
    confidence : (N, T) — per-trial score at each timepoint
    outcome    : (N,) — binary outcome label (e.g. correct/error)
    times      : (T,) in seconds

    Returns
    -------
    dict (see temporal_cluster_permutation_auroc) with a JSON-serializable
    "clusters"/"significant" payload, or None if underpowered.
    """
    valid = np.all(np.isfinite(confidence), axis=1)
    outcome_valid = np.asarray(outcome)[valid].astype(int)
    if valid.sum() < 2 * min_trials_per_outcome or len(np.unique(outcome_valid)) < 2:
        return None
    if min((outcome_valid == 0).sum(), (outcome_valid == 1).sum()) < min_trials_per_outcome:
        return None
    res = temporal_cluster_permutation_auroc(confidence[valid], outcome_valid, times,
                                             n_perm=n_perm, rng=rng)
    return {"auc_stat": res["auc_stat"].tolist(), "clusters": res["clusters"],
            "significant": res["significant"], "n_trials": int(valid.sum())}


# ── AUROC and decoding accuracy ────────────────────────────────────────────────

def auroc(y_true: NDArray, scores: NDArray) -> float:
    """Area under the ROC curve.

    Delegates to sklearn's roc_auc_score, which handles tied scores via the
    Mann-Whitney U equivalence (a tied positive/negative pair contributes 0.5,
    not 0 or 1). A hand-rolled trapezoidal integral over an argsort of the
    scores is not order-independent under ties -- the tied entries land on
    either side of the threshold depending on argsort's arbitrary tie-break,
    so the same data can score 0.0 or 1.0 for a single tied pair.

    Parameters
    ----------
    y_true  : (N,) binary labels (0 or 1)
    scores  : (N,) decision scores (higher = more likely class 1)

    Returns
    -------
    auc : float in [0, 1], or nan if either class is absent
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    n_pos = y_true.sum()
    if n_pos == 0 or n_pos == len(y_true):
        return np.nan
    return float(roc_auc_score(y_true, scores))


def permutation_test_auroc(
    y_true: NDArray,
    scores: NDArray,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Permutation test for AUROC significance.

    Returns
    -------
    (observed_auc, p_value)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    obs = auroc(y_true, scores)
    null = np.array([auroc(rng.permutation(y_true), scores) for _ in range(n_perm)])
    p = permutation_pvalue(null >= obs)
    return float(obs), float(p)


def cross_validated_auroc(
    y: NDArray,
    X: NDArray,
    groups: NDArray,
) -> dict:
    """Leave-one-group-out cross-validated AUROC.

    Each unique group (e.g. subject) is used as the test set once,
    with all other groups used for training. Requires scikit-learn.

    Parameters
    ----------
    y      : (N,) binary labels
    X      : (N, d) feature matrix
    groups : (N,) group labels (e.g. subject IDs)

    Returns
    -------
    dict: aucs (per fold), mean, std
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    logo = LeaveOneGroupOut()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])

    aucs = []
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_te)) < 2:
            continue
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, proba))

    return {
        "aucs": np.array(aucs),
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
    }


# ── Effect size ────────────────────────────────────────────────────────────────

def cohens_d(x: NDArray, y: NDArray) -> float:
    """Cohen's d effect size for two independent samples.

    d = (μ_x - μ_y) / s_pooled
    where s_pooled = sqrt(((n_x-1)s_x² + (n_y-1)s_y²) / (n_x+n_y-2))
    """
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / (sp + 1e-15))


def hedges_g(x: NDArray, y: NDArray) -> float:
    """Hedge's g — bias-corrected Cohen's d for small samples.

    Applies the correction factor J ≈ 1 - 3/(4*df - 1) (Hedges 1981).
    Preferred over Cohen's d when n < 20 per group.
    """
    nx, ny = len(x), len(y)
    df = nx + ny - 2
    d = cohens_d(x, y)
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * j)


def linear_mixed_effects_test(
    metric: NDArray,
    condition: NDArray,
    subject: NDArray,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
    covariates: NDArray | None = None,
) -> dict:
    """Mixed-effects test for a condition effect, subject as random intercept.

    Fits metric ~ condition [+ covariates] by restricted maximum likelihood
    (statsmodels MixedLM) with a per-subject random intercept, and reports the
    model-based Wald statistics for the condition coefficient. This replaces
    an earlier version of this function that only subject-demeaned the data
    and permuted residuals -- that procedure never fit a mixed-effects model
    at all, despite the name.

    `n_perm` and `rng` are accepted for backward compatibility with existing
    call sites but are no longer used: a fitted mixed model has an analytic
    sampling distribution for its coefficients, so there is nothing left to
    permute.

    Parameters
    ----------
    metric     : (N,) — dependent variable (e.g., PR per trial)
    condition  : (N,) — numeric condition code (e.g., 0, 1, 2 for N-back load)
    subject    : (N,) — subject identifier (random-effect grouping variable)
    covariates : (N,) or (N, k) — optional nuisance regressors (e.g. set size,
                 response time) added as additional fixed effects alongside
                 condition, so the reported beta is adjusted for them.

    Returns
    -------
    dict:
      beta       : condition fixed-effect coefficient (nan if not converged)
      se         : its standard error (nan if not converged)
      p_value    : Wald test p-value, H0: beta == 0 (nan if not converged)
      r_squared  : squared correlation between the fixed-effects-predicted
                   values and metric (nan if not converged)
      converged  : whether MixedLM reported a stable fit
      reason     : None on success; a human-readable explanation of the
                   failure when converged is False (e.g. too few subject
                   groups, or the optimizer's Hessian was not positive
                   definite) -- a failed fit is never silently reported as a
                   zero effect
      n          : number of trials used in the fit
      n_subjects : number of unique subject groups
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    metric = np.asarray(metric, dtype=float)
    condition = np.asarray(condition, dtype=float)
    subject = np.asarray(subject)
    n = len(metric)
    n_subjects = len(np.unique(subject))

    def _failure(reason: str) -> dict:
        return {
            "beta": float("nan"), "se": float("nan"), "p_value": float("nan"),
            "r_squared": float("nan"), "converged": False, "reason": reason,
            "n": n, "n_subjects": n_subjects,
        }

    if n_subjects < 2:
        return _failure("fewer than 2 subject groups: no random-effect structure to estimate")

    X_cols = [np.ones(n), condition]
    if covariates is not None:
        C = np.atleast_2d(np.asarray(covariates, dtype=float))
        if C.shape[0] != n:
            C = C.T
        if C.shape[0] != n:
            return _failure("covariate rows do not match the outcome length")
        scales = np.std(C, axis=0)
        if np.any(~np.isfinite(scales)) or np.any(scales < 1e-12):
            return _failure("covariate design contains a non-finite or zero-variance column")
        C = (C - np.mean(C, axis=0)) / scales
        X_cols.extend(C[:, j] for j in range(C.shape[1]))
    X = np.column_stack(X_cols)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(metric)):
        return _failure("outcome or fixed-effect design contains non-finite values")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return _failure("fixed-effect design matrix is rank deficient")

    # A random-effect variance estimated at (or near) zero is a normal,
    # valid MLE outcome -- statsmodels raises ConvergenceWarning for that
    # boundary solution too, so a bare "did a warning fire" check would
    # reject perfectly good fits. What actually matters for this function is
    # whether the *condition* coefficient itself was identified: check
    # result.converged and that its own SE/p-value came out finite, rather
    # than reacting to every warning the variance-component estimate emits.
    fit_errors: list[str] = []
    valid_results = []
    for method in ("lbfgs", "powell", "bfgs", "cg"):
        try:
            candidate = MixedLM(metric, X, groups=subject).fit(
                reml=True, method=method, disp=False,
            )
        except Exception as exc:
            fit_errors.append(f"{method}: {exc}")
            continue
        coefficient_is_finite = (
            np.isfinite(candidate.params[1])
            and np.isfinite(candidate.bse[1])
            and np.isfinite(candidate.pvalues[1])
        )
        if candidate.converged and coefficient_is_finite:
            valid_results.append(candidate)
            break
        fit_errors.append(
            f"{method}: converged={candidate.converged}, "
            f"condition_coefficient_finite={coefficient_is_finite}"
        )

    if not valid_results:
        detail = "; ".join(fit_errors[:4])
        return _failure(f"MixedLM did not produce an identified converged fit ({detail})")
    result = valid_results[0]

    beta = float(result.params[1])
    se = float(result.bse[1])
    p_value = float(result.pvalues[1])

    # result.fittedvalues additionally predicts the random effects, which
    # requires inverting the random-effect covariance and raises whenever
    # that covariance is singular -- a near-zero between-subject variance
    # that has no bearing on whether the condition effect itself was
    # identified (checked above). Use the fixed-effects-only prediction
    # instead: it only needs the part of the fit we already validated.
    fixed_effects_fitted = X @ np.asarray(result.params[:X.shape[1]])
    if np.std(fixed_effects_fitted) < 1e-12 or np.std(metric) < 1e-12:
        r_squared = 0.0
    else:
        r_squared = float(np.corrcoef(fixed_effects_fitted, metric)[0, 1] ** 2)

    return {
        "beta": beta, "se": se, "p_value": p_value, "r_squared": r_squared,
        "converged": True, "reason": None, "n": n, "n_subjects": n_subjects,
    }


# ── Circular statistics ────────────────────────────────────────────────────────

def rayleigh_test(phases: NDArray) -> dict:
    """Rayleigh test for non-uniformity of circular distribution.

    Tests H₀: phases are uniformly distributed on the circle.
    A significant result (p < 0.05) indicates directional clustering —
    i.e., the population tends to occupy a preferred phase angle.

    Parameters
    ----------
    phases : (N,) array of angles in radians

    Returns
    -------
    dict:
      R      : mean resultant length (0 = uniform, 1 = perfectly concentrated)
      Z      : Rayleigh Z statistic = N * R²
      p_value: approximation valid for N ≥ 10 (Mardia & Jupp 2000, eq. 6.3.2)
      mean_direction : mean circular direction (radians)
    """
    phases = np.asarray(phases, dtype=float)
    N = len(phases)
    if N < 10:
        import warnings
        warnings.warn(
            f"rayleigh_test: N={N} < 10; the Mardia & Jupp asymptotic p-value "
            "approximation is not guaranteed accurate at this sample size.",
            RuntimeWarning,
        )
    C = np.cos(phases).mean()
    S = np.sin(phases).mean()
    R = np.sqrt(C**2 + S**2)          # mean resultant length
    Z = N * R**2                       # Rayleigh Z
    # p-value approximation (Mardia & Jupp 2000 Eq 6.3.3)
    p = float(np.exp(-Z) * (1 + (2*Z - Z**2) / (4*N) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*N**2)))
    p = float(np.clip(p, 0.0, 1.0))
    mean_dir = float(np.arctan2(S, C))
    return {"R": float(R), "Z": float(Z), "p_value": p, "mean_direction": mean_dir, "N": N}


def circular_anova_permutation_test(
    phases: NDArray,
    groups: NDArray,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Permutation-based multi-sample test for a common circular mean across groups.

    Uses the classical Watson-Williams statistic (common-kappa one-way circular
    ANOVA: pooled within-group resultant length vs. the total resultant length)
    as the test statistic, but calibrates its null by permuting group labels
    rather than relying on the classical F-distribution approximation (which
    assumes a von Mises distribution with kappa >= ~1) -- consistent with this
    project's general preference for permutation over parametric approximation.

    Parameters
    ----------
    phases : (N,) angles in radians
    groups : (N,) group labels
    n_perm : number of label permutations
    rng    : random number generator

    Returns
    -------
    dict: statistic (observed), p_value, n_groups, N
    """
    if rng is None:
        rng = np.random.default_rng(0)
    phases = np.asarray(phases, dtype=float)
    groups = np.asarray(groups)
    N = len(phases)

    def _stat(ph, gr):
        labels_u = np.unique(gr)
        R_sum = 0.0
        for g in labels_u:
            m = gr == g
            n_g = m.sum()
            R_sum += n_g * np.sqrt(np.cos(ph[m]).mean() ** 2 + np.sin(ph[m]).mean() ** 2)
        R_total = N * np.sqrt(np.cos(ph).mean() ** 2 + np.sin(ph).mean() ** 2)
        return float(R_sum - R_total)  # higher = groups more separated in mean direction

    obs = _stat(phases, groups)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _stat(phases, rng.permutation(groups))
    p_value = permutation_pvalue(null >= obs)

    return {"statistic": obs, "p_value": p_value, "n_groups": len(np.unique(groups)), "N": N}


def ctg_offdiagonal_test(
    auc_matrix: NDArray,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Test whether mean off-diagonal CTG AUC significantly exceeds chance (0.5).

    Also tests whether off-diagonal AUC is significantly lower than diagonal
    AUC (which would indicate time-specific rather than time-general coding).

    Parameters
    ----------
    auc_matrix : (T, T) cross-temporal generalisation matrix

    Returns
    -------
    dict:
      mean_diag        : mean diagonal AUC
      mean_offdiag     : mean off-diagonal AUC
      p_offdiag_vs_chance : p-value (off-diagonal > 0.5, bootstrap)
      p_diag_vs_offdiag   : p-value (diagonal > off-diagonal, permutation)
      temporal_stability  : mean_offdiag / mean_diag (1 = perfectly stable)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T = auc_matrix.shape[0]
    diag = np.diag(auc_matrix)
    mask_off = ~np.eye(T, dtype=bool)
    offdiag = auc_matrix[mask_off]

    mean_diag = float(diag.mean())
    mean_off = float(offdiag.mean())

    # Bootstrap: is mean_offdiag > 0.5?
    boot = np.array([
        offdiag[rng.integers(0, len(offdiag), len(offdiag))].mean()
        for _ in range(n_perm)
    ])
    p_vs_chance = float((boot <= 0.5).mean())

    # Permutation: diagonal > off-diagonal?
    obs_diff = mean_diag - mean_off
    all_vals = auc_matrix.flatten()
    null_diffs = np.array([
        np.diag(rng.permutation(all_vals).reshape(T, T)).mean() -
        rng.permutation(all_vals).reshape(T, T)[mask_off].mean()
        for _ in range(n_perm)
    ])
    p_diag_vs_off = permutation_pvalue(null_diffs >= obs_diff)

    stability = mean_off / (mean_diag + 1e-8)
    return {
        "mean_diag": mean_diag,
        "mean_offdiag": mean_off,
        "diag_minus_offdiag": float(obs_diff),
        "p_offdiag_vs_chance": p_vs_chance,
        "p_diag_vs_offdiag": p_diag_vs_off,
        "temporal_stability": float(stability),
    }


def mantel_test(
    rdm_a: NDArray,
    rdm_b: NDArray,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Mantel test: Spearman correlation between lower triangles of two RDMs.

    Parameters
    ----------
    rdm_a, rdm_b : (N, N) symmetric distance matrices

    Returns
    -------
    dict: r (Spearman ρ), p_value, null_95th_pct
    """
    from scipy.stats import spearmanr

    if rng is None:
        rng = np.random.default_rng(0)

    idx = np.tril_indices(rdm_a.shape[0], k=-1)
    a_vec = rdm_a[idx]
    b_vec = rdm_b[idx]
    obs_r, _ = spearmanr(a_vec, b_vec)

    # A valid Mantel null jointly permutes row/column labels of one complete
    # RDM.  Permuting only its lower triangle treats the dependent distances as
    # exchangeable observations and grossly understates uncertainty.
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(rdm_a.shape[0])
        b_perm = rdm_b[np.ix_(perm, perm)]
        null[i] = spearmanr(a_vec, b_perm[idx])[0]
    p = permutation_pvalue(np.abs(null) >= abs(obs_r))
    return {
        "r": float(obs_r),
        "p_value": p,
        "null_95th_pct": float(np.percentile(null, 95)),
        "n_pairs": len(a_vec),
    }


def fisher_z_transform(r: float) -> float:
    """Fisher's z-transform: z = 0.5 * ln((1+r)/(1-r))."""
    return float(0.5 * np.log((1 + r + 1e-8) / (1 - r + 1e-8)))


def group_test_auroc(
    aurocs: NDArray, chance: float = 0.5, n_boot: int = 2000, rng: np.random.Generator | None = None
) -> dict:
    """Group-level AUROC summary with bootstrap CI and sign-flip p-value.

    Parameters
    ----------
    aurocs : (N,) AUROCs, one per subject/fold
    chance : null hypothesis value (0.5 for binary classification)

    Returns
    -------
      dict: mean, std, ci_lo, ci_hi, p_value -- sign-flip randomization;
          t_stat, ci_lo_analytic, ci_hi_analytic, p_value_analytic -- one-sample
          t-test against chance, reported alongside for comparison; n
    """
    from scipy.stats import ttest_1samp

    if rng is None:
        rng = np.random.default_rng(0)
    aurocs = np.asarray(aurocs, dtype=float)
    n = len(aurocs)
    centered = aurocs - chance

    t, p_analytic = ttest_1samp(aurocs, chance)
    ci_lo_analytic, ci_hi_analytic = (
        aurocs.mean() - 1.96 * aurocs.std(ddof=1) / np.sqrt(n),
        aurocs.mean() + 1.96 * aurocs.std(ddof=1) / np.sqrt(n),
    )

    if n < 2:
        raise ValueError("group_test_auroc requires at least two independent units")
    boot = np.array([centered[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) + chance
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = (signs * centered[None, :]).mean(axis=1)
    p_value = permutation_pvalue(np.abs(null) >= abs(float(centered.mean())))

    return {
        "mean": float(aurocs.mean()),
        "std": float(aurocs.std()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_value),
        "t_stat": float(t),
        "ci_lo_analytic": float(ci_lo_analytic),
        "ci_hi_analytic": float(ci_hi_analytic),
        "p_value_analytic": float(p_analytic),
        "n": n,
    }


def stouffer_combine(p_values: NDArray, weights: NDArray | None = None) -> dict:
    """Stouffer's Z-score method for combining independent p-values.

    WARNING -- INDEPENDENCE ASSUMPTION: this method is valid only when every
    p-value in `p_values` comes from a STATISTICALLY INDEPENDENT unit. It
    must not be called across dependent units -- e.g. multiple sessions from
    the same patient, or the same recording appearing under two release IDs
    -- without first collapsing each dependent cluster to a single row (one
    p-value per independent unit). Feeding it dependent p-values overstates
    the combined evidence: the same true effect gets counted once per
    duplicate, so z_combined grows and p_combined shrinks even though no new
    independent information was added. This function does not detect or
    correct for that on its own; the caller is responsible for the input
    being one row per independent unit.

    Used to pool per-session significance tests (e.g. item-identity CTG
    computed independently in many low-trial-count sessions) into one
    meta-analytic statistic, rather than treating each session's marginal
    p-value as if it settled the question alone.

    Parameters
    ----------
    p_values : (K,) — independent one-sided p-values (small p = evidence for
               the effect); values are clipped away from 0/1 to avoid
               infinite z-scores.
    weights  : (K,) optional weights (e.g. sqrt(n_trials) per session, for
               effective-sample-size weighting); defaults to equal weighting.

    Returns
    -------
    dict: z_combined, p_combined, k (number of units combined)
    """
    import warnings

    from scipy.stats import norm

    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1 - 1e-12)
    if len(p) < 2:
        warnings.warn(
            "stouffer_combine called with fewer than 2 p-values; there is "
            "nothing to combine and the result is just the input restated "
            "as a z-score.", stacklevel=2,
        )
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    z = norm.isf(p)  # inverse survival function: z s.t. P(Z>z)=p
    z_combined = float(np.sum(w * z) / np.sqrt(np.sum(w**2)))
    p_combined = float(norm.sf(z_combined))
    return {"z_combined": z_combined, "p_combined": p_combined, "k": int(len(p))}


# ── Multiple comparisons ───────────────────────────────────────────────────────

def fdr_bh(p_values: NDArray, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg false discovery rate correction.

    With many test families reported across a paper (PR, CTG, τ, Rayleigh,
    DMD/divergence, alignment, location, correct-vs-error — each run per
    subject per dataset), uncorrected per-test p<0.05 claims accumulate a
    substantial forking-paths exposure. BH controls the expected proportion
    of false discoveries among rejected hypotheses.

    Parameters
    ----------
    p_values : (M,) uncorrected p-values across a test family
    alpha    : target FDR level

    Returns
    -------
    dict:
      q_values  : (M,) BH-adjusted p-values (q-values), same order as input
      reject    : (M,) bool — significant at the given alpha after correction
      n_reject  : int
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity (BH step-up)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_values = np.empty(m)
    q_values[order] = q_sorted
    reject = q_values <= alpha

    return {
        "q_values": q_values,
        "reject": reject,
        "n_reject": int(reject.sum()),
        "alpha": alpha,
    }


def robust_dispersion(x: NDArray) -> dict:
    """Robust dispersion summary, resistant to single-outlier-donor blowup.

    A raw min/max fold-range (e.g. max(x)/min(x)) is dominated by whichever
    single value happens to be smallest — for near-zero denominators the
    ratio can explode and grows with sample count even under a fixed
    underlying distribution. IQR-based measures don't share that pathology.

    Parameters
    ----------
    x : (N,) values (e.g. per-donor B-matrix magnitudes)

    Returns
    -------
    dict: median, q25, q75, iqr, iqr_ratio (q75/q25), mad (median abs deviation),
          range_naive (max/min, reported alongside for comparison only)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    mad = float(np.median(np.abs(x - med)))
    x_pos = x[x > 0]
    range_naive = float(x_pos.max() / x_pos.min()) if len(x_pos) > 1 else float("nan")
    return {
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "iqr_ratio": float(q75 / q25) if q25 > 1e-12 else float("nan"),
        "mad": mad,
        "range_naive": range_naive,
        "n": int(len(x)),
    }


def forest_meta(
    estimates: NDArray,
    ses: NDArray,
    labels: list[str] | None = None,
    method: str = "random",
) -> dict:
    """Inverse-variance meta-analysis of per-dataset effect estimates.

    The paper reports the same effect (e.g. context-minus-content temporal
    stability, or correct-minus-error drift) in several cohorts that differ in
    N and precision. Averaging the point estimates unweighted throws away that
    precision and treats a 4-subject ECoG cohort like a 34-session single-unit
    one; a raw pool across subjects would instead ignore between-cohort
    heterogeneity and understate the CI. Inverse-variance pooling with a
    DerSimonian-Laird random-effects variance does both correctly and yields
    the pooled estimate + CI that each constraint's forest plot is built on.

    Parameters
    ----------
    estimates : (K,) per-cohort effect estimates (any additive scale)
    ses       : (K,) their standard errors (must be > 0; non-finite rows and
                rows with non-finite estimates are dropped)
    labels    : (K,) cohort names, for the returned plotting table
    method    : 'random' (DerSimonian-Laird, default) or 'fixed'

    Returns
    -------
    dict:
      pooled, se, ci_lo, ci_hi, z, p_value : pooled effect and its inference
      tau2, Q, Q_df, Q_p, i_squared         : between-cohort heterogeneity
      method, k                             : method used, n cohorts pooled
      rows : list of per-cohort dicts (label, estimate, se, ci_lo, ci_hi,
             weight_pct) — one forest-plot row each, ordered as input
    """
    from scipy.stats import norm, chi2

    est = np.asarray(estimates, dtype=float)
    se = np.asarray(ses, dtype=float)
    if labels is None:
        labels = [f"cohort{i}" for i in range(len(est))]
    labels = list(labels)

    keep = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[keep], se[keep]
    kept_labels = [lab for lab, k in zip(labels, keep) if k]
    k = len(est)
    if k == 0:
        raise ValueError("forest_meta: no cohort has a finite estimate and a positive SE.")

    v = se**2
    w_fixed = 1.0 / v
    pooled_fixed = float(np.sum(w_fixed * est) / np.sum(w_fixed))

    # Cochran's Q and DerSimonian-Laird tau^2 (both computed from fixed weights)
    Q = float(np.sum(w_fixed * (est - pooled_fixed) ** 2))
    Q_df = k - 1
    if Q_df > 0:
        c = float(np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed))
        tau2 = max(0.0, (Q - Q_df) / c) if c > 0 else 0.0
        i_squared = max(0.0, (Q - Q_df) / Q) * 100.0 if Q > 0 else 0.0
        Q_p = float(chi2.sf(Q, Q_df))
    else:
        tau2, i_squared, Q_p = 0.0, 0.0, float("nan")

    if method == "random":
        w = 1.0 / (v + tau2)
    elif method == "fixed":
        w = w_fixed
    else:
        raise ValueError("method must be 'random' or 'fixed'.")

    pooled = float(np.sum(w * est) / np.sum(w))
    se_pooled = float(np.sqrt(1.0 / np.sum(w)))
    z = pooled / se_pooled if se_pooled > 0 else float("inf")
    p = float(2.0 * norm.sf(abs(z)))

    weight_pct = 100.0 * w / np.sum(w)
    rows = [
        {
            "label": lab,
            "estimate": float(e),
            "se": float(s),
            "ci_lo": float(e - 1.96 * s),
            "ci_hi": float(e + 1.96 * s),
            "weight_pct": float(wp),
        }
        for lab, e, s, wp in zip(kept_labels, est, se, weight_pct)
    ]

    return {
        "pooled": pooled,
        "se": se_pooled,
        "ci_lo": float(pooled - 1.96 * se_pooled),
        "ci_hi": float(pooled + 1.96 * se_pooled),
        "z": float(z),
        "p_value": p,
        "tau2": float(tau2),
        "Q": Q,
        "Q_df": int(Q_df),
        "Q_p": Q_p,
        "i_squared": float(i_squared),
        "method": method,
        "k": int(k),
        "rows": rows,
    }


def tost_equivalence(estimate: float, se: float, sesoi: float) -> dict:
    """Two one-sided tests (TOST) for statistical equivalence to zero.

    A non-significant slope does not establish that an effect is absent; TOST
    reframes the question as "is the effect small enough to not matter" by
    testing the composite null that the true value lies outside +/- sesoi
    (Schuirmann 1987; Lakens 2017). Rejecting BOTH one-sided nulls (upper p is
    the larger of the two) licenses the accept-the-null claim that the estimate
    is equivalent to zero within the smallest effect of interest.

    Parameters
    ----------
    estimate : point estimate (e.g. a pooled PR-vs-load slope)
    se       : its standard error (> 0)
    sesoi    : smallest effect size of interest (equivalence bound, > 0); the
               test is against the interval [-sesoi, +sesoi]

    Returns
    -------
    dict: p_lower (H0: theta <= -sesoi), p_upper (H0: theta >= +sesoi), p (the
          binding max of the two), reject (bool, equivalence at 0.05), sesoi
    """
    from scipy.stats import norm

    sesoi = abs(float(sesoi))
    z_lower = (estimate + sesoi) / se   # test theta <= -sesoi (want estimate >> -sesoi)
    z_upper = (sesoi - estimate) / se   # test theta >= +sesoi (want estimate << +sesoi)
    p_lower = float(norm.sf(z_lower))
    p_upper = float(norm.sf(z_upper))
    p = max(p_lower, p_upper)
    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p": p,
        "reject": bool(p < 0.05),
        "sesoi": sesoi,
    }


def bf_null_slope(estimate: float, se: float, r_scale: float | None = None) -> dict:
    """JZS (unit-information Cauchy) Bayes factor for a regression slope null.

    Complements TOST with an evidential measure: BF01 is how much more likely
    the data are under H0 (slope = 0) than under a Jeffreys-Zellner-Siow
    alternative placing a Cauchy(0, r_scale) prior on the slope (Rouder et al.
    2009; Rouder & Morey 2012). Computed by Savage-Dickey: BF01 is the ratio
    of the posterior to the prior density at slope = 0.

    A Cauchy prior combined with a Gaussian likelihood does not have a closed-
    form Normal posterior, so the posterior density at 0 is obtained from the
    Savage-Dickey identity in its likelihood/marginal-likelihood form instead
    of approximating the posterior directly:

        BF01 = p(0 | data) / p(0)  =  L(data | theta=0) / m(data)

    where L(data | theta) is the Gaussian sampling density of `estimate`
    around theta with standard error `se`, and m(data) is L(data | theta)
    averaged over the Cauchy(0, r_scale) prior on theta, obtained by
    numerical integration. (An earlier version of this function treated
    Normal(estimate, se) itself as the posterior, i.e. ignored the prior
    entirely when forming the ratio -- that is not a valid Bayes factor.)
    BF01 > 1 favours the null.

    Parameters
    ----------
    estimate : slope point estimate
    se       : its standard error (> 0)
    r_scale  : Cauchy prior scale on the slope. Defaults to se (unit-information),
               but callers should pass the SESOI so H1's plausible-effect range is
               the smallest effect worth caring about rather than the (possibly
               tiny) sampling SE, which otherwise degenerates the ratio at 0.

    Returns
    -------
    dict: bf_01 (evidence for null over effect), bf_10, r_scale
    """
    from scipy.integrate import quad
    from scipy.stats import cauchy, norm

    r = se if r_scale is None else float(r_scale)

    def _likelihood(theta: float) -> float:
        return norm.pdf(estimate, loc=theta, scale=se)

    marginal, _ = quad(lambda theta: _likelihood(theta) * cauchy.pdf(theta, loc=0.0, scale=r),
                        -np.inf, np.inf)
    likelihood_at_null = _likelihood(0.0)
    bf_01 = float(likelihood_at_null / marginal)
    return {"bf_01": bf_01, "bf_10": float(1.0 / bf_01), "r_scale": r}


def minimum_detectable_paired_difference(
    values: NDArray | list[float], alpha: float = 0.05, power: float = 0.80,
) -> dict:
    """Smallest true mean paired difference a one-sample (paired) two-sided
    test could have detected at the given power, from the OBSERVED
    between-unit spread of ``values`` (e.g. one difference per session) and
    the number of units actually available -- the complement to a
    non-significant p-value, which on its own says nothing about how large an
    effect the design could have ruled out (Lakens 2013). Uses the standard
    normal approximation to the paired t-test's power (adequate once n is
    more than a handful of units; this project's paired sign-flip test's own
    exact small-sample behaviour is not reproduced here, so this bound is
    reported as an approximation, not a substitute for the sign-flip test's
    own p-value and CI).

        mdd = (z_(1-alpha/2) + z_power) * sd(values) / sqrt(n)

    Parameters
    ----------
    values : one difference (or any paired scalar) per unit; only its count
             and sample standard deviation are used
    alpha  : two-sided test size (default 0.05)
    power  : target power (default 0.80)

    Returns
    -------
    dict: n, sd (sample standard deviation, ddof=1), mdd, alpha, power, or a
          ``not_computable`` status if fewer than 2 values are supplied.
    """
    from scipy.stats import norm

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return {"status": "not_computable", "n": n, "reason": "fewer than 2 values -- no spread to estimate"}
    sd = float(np.std(arr, ddof=1))
    z = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    mdd = z * sd / np.sqrt(n)
    return {"status": "computed", "n": n, "sd": sd, "alpha": alpha, "power": power, "z_factor": z, "mdd": float(mdd)}


def power_to_detect_effect(
    effect: float, values: NDArray | list[float], alpha: float = 0.05,
) -> dict:
    """Power of a one-sample (or paired) two-sided t-test, run on ``values``,
    to detect a true effect of the given magnitude -- the complement question
    to :func:`minimum_detectable_paired_difference` ("what effect could this
    design detect at 80% power" versus "what is this design's power against a
    SPECIFIC effect size"). Exact, via the noncentral t distribution (unlike
    :func:`minimum_detectable_paired_difference`'s normal approximation):
    under a true mean ``effect`` with the observed standard deviation
    ``sd(values)`` and ``n = len(values)``, the test statistic is
    noncentral-t distributed with ``n - 1`` degrees of freedom and
    noncentrality ``effect / (sd / sqrt(n))``; power is the probability that
    statistic falls outside the central test's two-sided critical region.

    A non-significant result on ``values`` is otherwise ambiguous between "no
    effect" and "underpowered to see the effect the comparison arm shows" --
    this quantity distinguishes them by naming a concrete effect size (e.g.
    another arm's observed mean) and reporting the design's power against it.

    Parameters
    ----------
    effect : the true effect size to compute power against (e.g. another
        arm's observed mean difference)
    values : the sample whose count and standard deviation set this test's
        sensitivity (one value per unit -- session, subject, etc.)
    alpha  : two-sided test size (default 0.05)

    Returns
    -------
    dict: n, sd (ddof=1), effect, alpha, noncentrality, power, or a
          ``not_computable`` status if fewer than 2 values are supplied or
          the sample has zero spread.
    """
    from scipy.stats import nct, t as t_dist

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return {"status": "not_computable", "n": n, "reason": "fewer than 2 values -- no spread to estimate"}
    sd = float(np.std(arr, ddof=1))
    if sd <= 0:
        return {"status": "not_computable", "n": n, "sd": sd, "reason": "zero spread -- noncentrality is undefined"}
    df = n - 1
    se = sd / np.sqrt(n)
    ncp = float(effect / se)
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df))
    power = float(nct.sf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp))
    return {
        "status": "computed", "n": n, "sd": sd, "effect": float(effect), "alpha": alpha,
        "df": df, "noncentrality": ncp, "power": power,
    }


# ── Geometric biomarker summary table ─────────────────────────────────────────

def biomarker_summary(
    pr_by_condition: dict[str, NDArray],
    theta_by_condition: dict[str, NDArray],
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Produce a summary table row for each pairwise comparison.

    Returns
    -------
    List of dicts with: comparison, metric, effect_d, p_value, mean_a, mean_b
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []

    def _row(name, metric, a, b):
        _, p = permutation_test_twosample(a, b, rng=rng)
        return {
            "comparison": name,
            "metric": metric,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "effect_d": cohens_d(a, b),
            "effect_g": hedges_g(a, b),
            "p_value": p,
        }

    keys = list(pr_by_condition.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k_a, k_b = keys[i], keys[j]
            rows.append(_row(f"{k_a} vs {k_b}", "PR",
                             pr_by_condition[k_a], pr_by_condition[k_b]))

    return rows
