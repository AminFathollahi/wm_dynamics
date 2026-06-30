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

import numpy as np
from numpy.typing import NDArray
from typing import Callable


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
        p = (np.abs(null) >= np.abs(observed)).mean()
    elif alternative == "greater":
        p = (null >= observed).mean()
    else:
        p = (null <= observed).mean()

    return float(observed), float(p)


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
        above = np.abs(t_vec) > threshold
        clusters = []
        in_cluster = False
        for i, a in enumerate(above):
            if a and not in_cluster:
                start = i
                in_cluster = True
            elif not a and in_cluster:
                clusters.append((start, i))
                in_cluster = False
        if in_cluster:
            clusters.append((start, T))
        return clusters

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

    obs_cluster_stats = [t_obs[s:e].sum() for s, e in obs_clusters]

    # Null distribution
    null_max_stats = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(nx + ny)
        a_p = pooled_time[perm[:nx]]
        b_p = pooled_time[perm[nx:]]
        t_p = _tstat(a_p, b_p)
        clust_p = _clusters(t_p, t_crit)
        if clust_p:
            null_max_stats[i] = max(abs(t_p[s:e].sum()) for s, e in clust_p)

    results_clusters = []
    for (s, e), stat in zip(obs_clusters, obs_cluster_stats):
        p = (null_max_stats >= abs(stat)).mean()
        results_clusters.append({
            "start_s": float(times[s]),
            "end_s": float(times[min(e, T - 1)]),
            "cluster_stat": float(stat),
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


# ── AUROC and decoding accuracy ────────────────────────────────────────────────

def auroc(y_true: NDArray, scores: NDArray) -> float:
    """Area under the ROC curve (trapezoidal rule).

    Parameters
    ----------
    y_true  : (N,) binary labels (0 or 1)
    scores  : (N,) decision scores (higher = more likely class 1)

    Returns
    -------
    auc : float in [0, 1]
    """
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    return float(np.trapz(tpr, fpr))


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
    p = (null >= obs).mean()
    return float(obs), float(p)


def cross_validated_auroc(
    y: NDArray,
    X: NDArray,
    groups: NDArray,
    scorer: str = "roc_auc",
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
) -> dict:
    """Permutation-based test for a condition effect accounting for subject.

    Approximates a linear mixed-effects model (condition as fixed, subject as
    random) via permutation: within each subject, condition labels are permuted
    independently so the null distribution preserves subject-level structure.

    Parameters
    ----------
    metric    : (N,) — dependent variable (e.g., PR per trial)
    condition : (N,) — numeric condition code (e.g., 0, 1, 2 for N-back load)
    subject   : (N,) — subject identifier

    Returns
    -------
    dict:
      beta       : observed slope (regression of metric on condition)
      p_value    : permutation p-value
      r_squared  : fraction of between-subject-residualised variance explained
    """
    if rng is None:
        rng = np.random.default_rng(0)

    metric = np.asarray(metric, dtype=float)
    condition = np.asarray(condition, dtype=float)
    subject = np.asarray(subject)

    # Remove subject means (partial out random intercept)
    metric_resid = metric.copy()
    cond_resid = condition.copy()
    for s in np.unique(subject):
        mask = subject == s
        metric_resid[mask] -= metric[mask].mean()
        cond_resid[mask] -= condition[mask].mean()

    def _beta(m, c):
        denom = (c**2).sum()
        return float((m * c).sum() / denom) if denom > 1e-15 else 0.0

    obs = _beta(metric_resid, cond_resid)

    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm_m = metric_resid.copy()
        for s in np.unique(subject):
            mask = subject == s
            perm_m[mask] = rng.permutation(metric_resid[mask])
        null[i] = _beta(perm_m, cond_resid)

    p = float((np.abs(null) >= np.abs(obs)).mean())

    # R² on residualised data
    ss_tot = ((metric_resid - metric_resid.mean()) ** 2).sum()
    predicted = cond_resid * obs
    ss_res = ((metric_resid - predicted) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))

    return {"beta": obs, "p_value": p, "r_squared": r2}


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
