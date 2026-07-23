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
        p = permutation_pvalue(null_max_stats >= abs(stat))
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

    def _clusters(vec: NDArray, threshold: float) -> list[tuple[int, int]]:
        above = np.abs(vec) > threshold
        clusters, in_cluster, start = [], False, 0
        for i, a in enumerate(above):
            if a and not in_cluster:
                start, in_cluster = i, True
            elif not a and in_cluster:
                clusters.append((start, i))
                in_cluster = False
        if in_cluster:
            clusters.append((start, T))
        return clusters

    auc_obs = np.array([_auc_minus_chance(t, outcome) for t in range(T)])

    null_auc = np.empty((n_perm, T))
    for p in range(n_perm):
        y_perm = rng.permutation(outcome)
        null_auc[p] = np.array([_auc_minus_chance(t, y_perm) for t in range(T)])

    auc_threshold = float(np.percentile(np.abs(null_auc), 100 * (1 - alpha_threshold)))
    obs_clusters = _clusters(auc_obs, auc_threshold)

    null_max_stats = np.array([
        max((abs(null_auc[p, s:e].sum()) for s, e in _clusters(null_auc[p], auc_threshold)),
            default=0.0)
        for p in range(n_perm)
    ])

    if not obs_clusters:
        return {"auc_stat": auc_obs, "clusters": [], "significant": [], "times": times,
                "auc_threshold": auc_threshold}

    results_clusters = []
    for s, e in obs_clusters:
        stat = float(auc_obs[s:e].sum())
        p_value = permutation_pvalue(null_max_stats >= abs(stat))
        results_clusters.append({
            "start_s": float(times[s]), "end_s": float(times[min(e, T - 1)]),
            "cluster_stat": stat, "p_value": p_value,
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
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(tpr, fpr))


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
    """Permutation-based test for a condition effect accounting for subject.

    Approximates a linear mixed-effects model (condition as fixed, subject as
    random) via permutation: within each subject, condition labels are permuted
    independently so the null distribution preserves subject-level structure.

    Parameters
    ----------
    metric     : (N,) — dependent variable (e.g., PR per trial)
    condition  : (N,) — numeric condition code (e.g., 0, 1, 2 for N-back load)
    subject    : (N,) — subject identifier
    covariates : (N,) or (N, k) — optional nuisance regressors (e.g. set size,
                 response time) partialled out of both metric and condition via
                 OLS (Frisch-Waugh-Lovell) before subject-demeaning, so the
                 reported beta is the condition effect adjusted for them.

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

    if covariates is not None:
        C = np.atleast_2d(np.asarray(covariates, dtype=float))
        if C.shape[0] != len(metric):
            C = C.T
        C = np.column_stack([np.ones(len(metric)), C])

        def _residualize(v: NDArray) -> NDArray:
            beta_cov, *_ = np.linalg.lstsq(C, v, rcond=None)
            return v - C @ beta_cov

        metric = _residualize(metric)
        condition = _residualize(condition)

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

    p = permutation_pvalue(np.abs(null) >= np.abs(obs))

    # R² on residualised data
    ss_tot = ((metric_resid - metric_resid.mean()) ** 2).sum()
    predicted = cond_resid * obs
    ss_res = ((metric_resid - predicted) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-15))

    return {"beta": obs, "p_value": p, "r_squared": r2}


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

    null = np.array([
        spearmanr(rng.permutation(a_vec), b_vec)[0]
        for _ in range(n_perm)
    ])
    p = permutation_pvalue(null >= obs_r)
    return {
        "r": float(obs_r),
        "p_value": p,
        "null_95th_pct": float(np.percentile(null, 95)),
        "n_pairs": len(a_vec),
    }


def fisher_z_transform(r: float) -> float:
    """Fisher's z-transform: z = 0.5 * ln((1+r)/(1-r))."""
    return float(0.5 * np.log((1 + r + 1e-8) / (1 - r + 1e-8)))


def group_test_auroc(aurocs: NDArray, chance: float = 0.5) -> dict:
    """One-sample t-test against chance and bootstrap CI for a vector of AUROCs.

    Parameters
    ----------
    aurocs : (N,) AUROCs, one per subject/fold
    chance : null hypothesis value (0.5 for binary classification)

    Returns
    -------
    dict: mean, ci_lo, ci_hi, t_stat, p_value
    """
    from scipy.stats import ttest_1samp
    t, p = ttest_1samp(aurocs, chance)
    ci_lo, ci_hi = (
        aurocs.mean() - 1.96 * aurocs.std() / np.sqrt(len(aurocs)),
        aurocs.mean() + 1.96 * aurocs.std() / np.sqrt(len(aurocs)),
    )
    return {
        "mean": float(aurocs.mean()),
        "std": float(aurocs.std()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "t_stat": float(t),
        "p_value": float(p),
        "n": len(aurocs),
    }


def stouffer_combine(p_values: NDArray, weights: NDArray | None = None) -> dict:
    """Stouffer's Z-score method for combining independent p-values.

    Used to pool per-session significance tests (e.g. item-identity CTG
    computed independently in many low-trial-count sessions) into one
    meta-analytic statistic, rather than treating each session's marginal
    p-value as if it settled the question alone.

    Parameters
    ----------
    p_values : (K,) — independent one-sided p-values (small p = evidence for
               the effect); values are clipped away from 0/1 to avoid
               infinite z-scores.
    weights  : (K,) optional weights (e.g. sqrt(n_trials) per session);
               defaults to equal weighting.

    Returns
    -------
    dict: z_combined, p_combined
    """
    from scipy.stats import norm

    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1 - 1e-12)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)
    z = norm.isf(p)  # inverse survival function: z s.t. P(Z>z)=p
    z_combined = float(np.sum(w * z) / np.sqrt(np.sum(w**2)))
    p_combined = float(norm.sf(z_combined))
    return {"z_combined": z_combined, "p_combined": p_combined}


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
    alternative placing a Cauchy(0, r_scale) prior on the standardized slope
    (Rouder et al. 2009; Rouder & Morey 2012). Computed by Savage-Dickey: the
    prior/posterior density ratio at 0, with the posterior approximated as
    Normal(estimate, se) — exact in the large-sample / known-variance limit
    this pooled estimate lives in. BF01 > 1 favours the null.

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
    from scipy.stats import cauchy, norm

    r = se if r_scale is None else float(r_scale)
    prior_at_0 = cauchy.pdf(0.0, loc=0.0, scale=r)
    post_at_0 = norm.pdf(0.0, loc=estimate, scale=se)
    bf_01 = float(post_at_0 / prior_at_0)   # Savage-Dickey: posterior/prior at 0
    return {"bf_01": bf_01, "bf_10": float(1.0 / bf_01), "r_scale": r}


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
