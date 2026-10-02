"""Common-scale effect size, minimum detectable correlation and random-effects meta-analysis."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from statistics import Z_80_POWER, bootstrap_ci, fisher_z_transform, forest_meta, stable_seed

POOLABLE_LINKS = frozenset({4, 5, 6, 7, 10})
POOLING_KEYS = ("species", "recording_type", "task_class")
MIN_CLUSTERS = 3
MIN_CLUSTERS_FOR_DETECTABLE_CORRELATION = 4
MIN_STUDIES_FOR_META_ANALYSIS = 3
DEFAULT_SEED = stable_seed("within_cluster_partial_correlation")


def _not_estimable(reason: str, **counts) -> dict:
    return {"status": "not_estimable", "reason": reason, "r": None, "z": None, "ci_95_r": None,
            "se_z": None, **counts}


def _demean_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=float, copy=True)
    for group in np.unique(groups):
        mask = groups == group
        out[mask] -= out[mask].mean(axis=0)
    return out


def _residual_correlation(predictor: np.ndarray, outcome: np.ndarray, covariates: np.ndarray | None) -> float:
    if covariates is not None and covariates.shape[1]:
        design = covariates
        predictor = predictor - design @ np.linalg.lstsq(design, predictor, rcond=None)[0]
        outcome = outcome - design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
    if predictor.std() == 0 or outcome.std() == 0:
        return float("nan")
    return float(np.corrcoef(predictor, outcome)[0, 1])


def within_cluster_partial_correlation(
    predictor, outcome, session, cluster, covariates=None, n_boot: int = 5000, seed: int = DEFAULT_SEED
) -> dict:
    predictor, outcome = np.asarray(predictor, dtype=float), np.asarray(outcome, dtype=float)
    session, cluster = np.asarray(session), np.asarray(cluster)
    design = None
    if covariates is not None:
        design = np.asarray(covariates, dtype=float)
        design = design.reshape(len(predictor), -1)
    finite = np.isfinite(predictor) & np.isfinite(outcome)
    if design is not None:
        finite &= np.isfinite(design).all(axis=1)
    predictor, outcome, session, cluster = predictor[finite], outcome[finite], session[finite], cluster[finite]
    design = None if design is None else design[finite]
    counts = {"n_trials": int(len(predictor)), "n_sessions": int(len(np.unique(session))),
              "n_clusters": int(len(np.unique(cluster)))}
    if counts["n_clusters"] < MIN_CLUSTERS:
        return _not_estimable(f"fewer than {MIN_CLUSTERS} clusters", **counts)

    predictor, outcome = _demean_within(predictor, session), _demean_within(outcome, session)
    if design is not None:
        design = _demean_within(design, session)
    r = _residual_correlation(predictor, outcome, design)
    if not np.isfinite(r):
        return _not_estimable("predictor or outcome has no variance within sessions", **counts)

    cluster_ids = np.unique(cluster)
    rows_of = [np.flatnonzero(cluster == c) for c in cluster_ids]
    boot_z: list[float] = []

    def resampled_r(drawn: np.ndarray) -> float:
        rows = np.concatenate([rows_of[i] for i in drawn.astype(int)])
        value = _residual_correlation(predictor[rows], outcome[rows], None if design is None else design[rows])
        boot_z.append(fisher_z_transform(value) if np.isfinite(value) else float("nan"))
        return value

    _, low, high = bootstrap_ci(np.arange(len(cluster_ids)), resampled_r, n_boot=n_boot,
                                rng=np.random.default_rng(seed))
    draws = np.asarray(boot_z[1:], dtype=float)
    return {"status": "estimable", "reason": None, "r": r, "z": fisher_z_transform(r),
            "ci_95_r": [low, high], "se_z": float(np.nanstd(draws, ddof=1)), "n_boot": n_boot, **counts}


def _bound(n: int | None) -> float | None:
    if n is None or int(n) < MIN_CLUSTERS_FOR_DETECTABLE_CORRELATION:
        return None
    return float(np.tanh(Z_80_POWER / np.sqrt(int(n) - 3)))


def minimum_detectable_correlation(n_clusters: int, n_trials: int | None = None) -> dict:
    """Bracket on the smallest detectable correlation: the cluster bound (n = clusters) and the trial bound (trials treated as independent)."""
    cluster_bound, trial_bound = _bound(n_clusters), _bound(n_trials)
    return {"status": "estimable" if cluster_bound is not None else "not_estimable", "n_clusters": int(n_clusters),
            "n_trials": None if n_trials is None else int(n_trials), "cluster_bound": cluster_bound, "trial_bound": trial_bound}


def random_effects_meta_analysis(z, se, labels) -> dict:
    z, se = np.asarray(z, dtype=float), np.asarray(se, dtype=float)
    labels = list(labels)
    usable = np.isfinite(z) & np.isfinite(se) & (se > 0)
    k = int(usable.sum())
    if k < MIN_STUDIES_FOR_META_ANALYSIS:
        return {"status": "not_estimable", "reason": f"fewer than {MIN_STUDIES_FOR_META_ANALYSIS} corpora",
                "k": k, "labels": [lab for lab, ok in zip(labels, usable) if ok]}
    z, se = z[usable], se[usable]
    base = forest_meta(z, se, [lab for lab, ok in zip(labels, usable) if ok], method="random")
    tau2 = base["tau2"]
    weights = 1.0 / (se**2 + tau2)
    pooled = float(np.sum(weights * z) / np.sum(weights))
    variance = float(np.sum(weights * (z - pooled) ** 2) / ((k - 1) * np.sum(weights)))
    standard_error = float(np.sqrt(variance))
    t_quantile = float(student_t.ppf(0.975, k - 1))
    ci = [pooled - t_quantile * standard_error, pooled + t_quantile * standard_error]
    half_prediction = float(student_t.ppf(0.975, k - 2) * np.sqrt(tau2 + variance))
    prediction = [pooled - half_prediction, pooled + half_prediction]
    return {
        "status": "estimable", "reason": None, "k": k, "labels": [row["label"] for row in base["rows"]],
        "pooled_z": pooled, "se_z": standard_error, "ci_95_z": ci,
        "pooled_r": float(np.tanh(pooled)), "ci_95_r": [float(np.tanh(v)) for v in ci],
        "p_value": float(2.0 * student_t.sf(abs(pooled / standard_error), k - 1)) if standard_error > 0 else None,
        "tau2": tau2, "i_squared": base["i_squared"], "q": base["Q"], "q_df": base["Q_df"],
        "prediction_interval_z": prediction, "prediction_interval_r": [float(np.tanh(v)) for v in prediction],
        "adjustment": "Hartung-Knapp", "degrees_of_freedom": k - 1,
    }


def poolable(cells) -> bool:
    cells = list(cells)
    if len(cells) < 2:
        return False
    if any(cell["link"] not in POOLABLE_LINKS for cell in cells):
        return False
    if len({cell["link"] for cell in cells}) != 1:
        return False
    return all(len({cell[key] for cell in cells}) == 1 for key in POOLING_KEYS)
