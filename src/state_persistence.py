"""state_persistence.py -- r(gap): the across-trial correlation of a
population's leading latent position between two equal-length sub-windows of
an epoch, reported as a function of the gap between the windows, without
fitting a decay rate to get there.

The estimand, per cross-validated half-split of trials: fit the leading PCA
direction on training trials with :func:`observability._leading_latent_
projection`, project held-out trials onto it, divide the epoch's bins into
``n_windows`` equal windows, and take each held-out trial's mean latent per
window. Centre each window's values across trials -- this removes the
condition-locked (across-trial-average) time course and is the only
demeaning this estimator performs; unlike a per-trial temporal-mean removal,
it does not touch a component whose timescale is comparable to or longer
than the epoch, which is what makes this observable free of the failure mode
documented for the lagged-autocovariance decay fit. Correlate across trials
for every pair of windows and average the pairs sharing a gap. Report the
median over repeated half-splits, scored on held-out trials only.

Read the result off the SHAPE across gaps, never the level: flat means a
timescale long relative to the epoch (a fixed point is the limiting case);
decaying means drift, with the decay read off the calibration ladder rather
than a fitted rate; crossing zero means rotation in the leading plane.
Because the shape is a ratio across gaps, a constant attenuation factor --
noise level, unit count, firing rate -- cannot manufacture or destroy it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.typing import NDArray

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from observability import _leading_latent_projection  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from statistics import fdr_bh, paired_sign_flip_test, permutation_pvalue, spearman_permutation_test  # noqa: E402

N_WINDOWS_DEFAULT = 4
N_SPLITS_DEFAULT = 12
N_NULL_REPLICATES_DEFAULT = 20
NULL_SPLITS_PER_REPLICATE_DEFAULT = 6

# Reference calibration regimes: tau=None means a pure static offset (no
# decaying component at all).
CALIBRATION_REGIMES: list[tuple[str, dict]] = [
    ("pure_offset", dict(tau_s=None, share_static=1.0, share_slow=0.0)),
    ("ar1_tau_0.3s", dict(tau_s=0.3, share_static=0.0, share_slow=1.0)),
    ("ar1_tau_0.6s", dict(tau_s=0.6, share_static=0.0, share_slow=1.0)),
    ("ar1_tau_1.2s", dict(tau_s=1.2, share_static=0.0, share_slow=1.0)),
    ("ar1_tau_2.5s", dict(tau_s=2.5, share_static=0.0, share_slow=1.0)),
    ("ar1_tau_10s", dict(tau_s=10.0, share_static=0.0, share_slow=1.0)),
    ("offset_ar1_mixture_tau_0.6s", dict(tau_s=0.6, share_static=0.5, share_slow=0.5)),
]
ROTATION_REGIME = ("rotation_1.5pi", dict(tau_s=None, share_static=1.0, share_slow=0.0,
                                           rotate=True, rotation_radians=1.5 * np.pi))


def _window_means(latent: NDArray, n_windows: int) -> NDArray | None:
    n_bins = latent.shape[1]
    width = n_bins // n_windows
    if width < 1:
        return None
    means = np.stack([latent[:, i * width:(i + 1) * width].mean(axis=1) for i in range(n_windows)], axis=1)
    return means - means.mean(axis=0, keepdims=True)  # removes the condition-locked time course only


def _r_gap_one_split(counts: NDArray, train_idx: NDArray, test_idx: NDArray, n_windows: int) -> dict[int, float] | None:
    transform = FrozenPSTHTransform().fit(counts[train_idx])
    z_train = transform.transform(counts[train_idx])
    z_test = transform.transform(counts[test_idx])
    latent = _leading_latent_projection(z_train, z_test)
    window_means = _window_means(latent, n_windows)
    if window_means is None:
        return None
    per_gap: dict[int, list[float]] = {}
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            if window_means[:, i].std() == 0.0 or window_means[:, j].std() == 0.0:
                continue
            per_gap.setdefault(j - i, []).append(float(np.corrcoef(window_means[:, i], window_means[:, j])[0, 1]))
    return {gap: float(np.mean(values)) for gap, values in per_gap.items()} if per_gap else None


def r_gap_profile(
    counts: NDArray,
    n_splits: int = N_SPLITS_DEFAULT,
    rng: np.random.Generator | None = None,
    n_windows: int = N_WINDOWS_DEFAULT,
) -> dict:
    """Cross-validated r(gap): median across-trial correlation at each window
    gap, over repeated random half-splits, scored on held-out trials only.

    Returns a dict with ``status`` ("fitted" if at least 3 splits produced a
    usable profile, else a documented non-exception status), and ``gaps``
    mapping each integer gap to its median, IQR, and the number of splits
    that contributed to it.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (trials, units, bins)")
    n_trials = counts.shape[0]

    per_gap: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        train_idx, test_idx = perm[:half], perm[half:]
        if len(train_idx) < 3 or len(test_idx) < 3:
            continue
        result = _r_gap_one_split(counts, train_idx, test_idx, n_windows)
        if result is None:
            continue
        n_fitted += 1
        for gap, value in result.items():
            per_gap.setdefault(gap, []).append(value)

    status = "fitted" if n_fitted >= 3 else "fewer_than_three_fitted_splits"
    gaps: dict[int, dict] = {}
    if status == "fitted":
        for gap, values in sorted(per_gap.items()):
            gaps[gap] = {
                "r_median": float(np.median(values)),
                "r_iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
                "n_splits_fitted": len(values),
            }
    return {
        "status": status, "n_splits_requested": int(n_splits), "n_splits_fitted": int(n_fitted),
        "n_windows": int(n_windows), "n_trials": int(n_trials), "n_units": int(counts.shape[1]),
        "n_bins": int(counts.shape[2]), "gaps": gaps,
    }


def _lag_pairs(n_bins: int, width: int, lag: int) -> list[tuple[int, int]]:
    """The (start, start + lag) window-start pairs at fixed width and lag,
    both windows of ``width`` bins wholly inside [0, n_bins) and never
    overlapping (lag >= width is required upstream). Empty whenever
    lag + width > n_bins. The pair count this returns, n_bins - width -
    lag + 1, is the same estimator-asymmetry the gap-count profile has --
    more pairs are averaged at short lags than long ones -- carried into
    the fixed-width parameterisation, which is why every contrast on this
    estimator is still run against a null through the identical pair set
    rather than against zero."""
    if lag < width:
        return []
    n_pairs = n_bins - width - lag + 1
    if n_pairs < 1:
        return []
    return [(s, s + lag) for s in range(n_pairs)]


def _window_means_at_width(latent: NDArray, width: int) -> NDArray | None:
    n_bins = latent.shape[1]
    n_starts = n_bins - width + 1
    if n_starts < 1:
        return None
    means = np.stack([latent[:, s:s + width].mean(axis=1) for s in range(n_starts)], axis=1)
    return means - means.mean(axis=0, keepdims=True)  # removes the condition-locked time course only


def _r_lag_per_pair_one_split(
    counts: NDArray, train_idx: NDArray, test_idx: NDArray, width: int,
    projection_fn=_leading_latent_projection,
) -> dict[tuple[int, int], float] | None:
    """Same fit as :func:`_r_lag_one_split` (train-fold basis fit, held-out
    projection, per-width window means, across-trial correlation) but keyed
    by the individual (start, lag) pair rather than averaged within a lag --
    what the position-versus-lag decomposition needs, since averaging over
    starts at a fixed lag is exactly the step that discards the information
    needed to ask whether r depends on where in the epoch a pair sits.

    ``projection_fn`` fits the leading latent on the training fold and
    projects the held-out fold (same (psth_fit, psth_apply) -> (trials,
    bins) contract as :func:`observability._leading_latent_projection`,
    which is the default). Passing an alternative -- e.g.
    :func:`observability._leading_latent_projection_via_factor_analysis` --
    is the only change needed to ask every downstream statistic under a
    different noise model with the split, null and window logic held
    identical. A projection that returns None (a degenerate or
    non-converged fit) is treated exactly like a split with no usable
    window means: this split contributes nothing, not an error."""
    transform = FrozenPSTHTransform().fit(counts[train_idx])
    z_train = transform.transform(counts[train_idx])
    z_test = transform.transform(counts[test_idx])
    latent = projection_fn(z_train, z_test)
    if latent is None:
        return None
    window_means = _window_means_at_width(latent, width)
    if window_means is None:
        return None
    n_bins = latent.shape[1]
    pairs: dict[tuple[int, int], float] = {}
    for lag in range(width, n_bins - width + 1):
        for s, t in _lag_pairs(n_bins, width, lag):
            a, b = window_means[:, s], window_means[:, t]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            pairs[(s, lag)] = float(np.corrcoef(a, b)[0, 1])
    return pairs if pairs else None


def _r_lag_one_split(
    counts: NDArray, train_idx: NDArray, test_idx: NDArray, width: int,
    projection_fn=_leading_latent_projection,
) -> dict[int, float] | None:
    pairs = _r_lag_per_pair_one_split(counts, train_idx, test_idx, width, projection_fn=projection_fn)
    if pairs is None:
        return None
    per_lag: dict[int, list[float]] = {}
    for (_s, lag), r in pairs.items():
        per_lag.setdefault(lag, []).append(r)
    return {lag: float(np.mean(values)) for lag, values in per_lag.items()}


def r_lag_profile_per_pair(
    counts: NDArray, width_bins: int, n_splits: int = N_SPLITS_DEFAULT, rng: np.random.Generator | None = None,
) -> dict:
    """:func:`r_lag_profile` with the estimate kept PER (start, lag) PAIR
    instead of averaged to one value per lag: identical splits, identical
    per-split fit, identical seeds -- the only change is what gets written
    out. ``pairs`` is a flat list (JSON has no tuple keys) of records with
    the start bin, the lag in bins, the median and IQR across the splits
    that produced that pair, and the number of splits that did."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (trials, units, bins)")
    n_trials = counts.shape[0]

    per_pair: dict[tuple[int, int], list[float]] = {}
    n_fitted = 0
    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        train_idx, test_idx = perm[:half], perm[half:]
        if len(train_idx) < 3 or len(test_idx) < 3:
            continue
        result = _r_lag_per_pair_one_split(counts, train_idx, test_idx, width_bins)
        if result is None:
            continue
        n_fitted += 1
        for key, value in result.items():
            per_pair.setdefault(key, []).append(value)

    status = "fitted" if n_fitted >= 3 else "fewer_than_three_fitted_splits"
    pairs: list[dict] = []
    if status == "fitted":
        for (s, lag), values in sorted(per_pair.items()):
            pairs.append({
                "s": int(s), "lag": int(lag), "r_median": float(np.median(values)),
                "r_iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
                "n_splits_fitted": len(values),
            })
    return {
        "status": status, "n_splits_requested": int(n_splits), "n_splits_fitted": int(n_fitted),
        "width_bins": int(width_bins), "n_trials": int(n_trials), "n_units": int(counts.shape[1]),
        "n_bins": int(counts.shape[2]), "pairs": pairs,
    }


def per_unit_permutation_null_r_lag_profile_per_pair(
    counts: NDArray, width_bins: int, n_replicates: int = N_NULL_REPLICATES_DEFAULT,
    n_splits_per_replicate: int = NULL_SPLITS_PER_REPLICATE_DEFAULT, rng: np.random.Generator | None = None,
) -> dict:
    """The per-pair analogue of :func:`per_unit_permutation_null_r_lag_profile`:
    the same per-unit trial-label permutation, scored per (start, lag) pair
    rather than averaged to one value per lag, so d_perm(s, L) exists."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)

    replicate_pairs: dict[tuple[int, int], list[float]] = {}
    n_fitted = 0
    for _ in range(n_replicates):
        permuted = _permute_counts_independently_per_unit(counts, rng)
        result = r_lag_profile_per_pair(permuted, width_bins, n_splits=n_splits_per_replicate, rng=rng)
        if result["status"] != "fitted":
            continue
        n_fitted += 1
        for p in result["pairs"]:
            replicate_pairs.setdefault((p["s"], p["lag"]), []).append(p["r_median"])

    pairs = [
        {"s": int(s), "lag": int(lag), "r_null_median": float(np.median(values)), "r_null_p95": float(np.percentile(values, 95))}
        for (s, lag), values in sorted(replicate_pairs.items())
    ]
    return {"n_replicates_requested": int(n_replicates), "n_replicates_fitted": int(n_fitted), "pairs": pairs}


def pair_midpoint_seconds(s: int, lag: int, width_bins: int, bin_width_s: float) -> float:
    """The pair's midpoint position in the epoch, in seconds: (start + lag/2
    + width/2) * bin_width_s -- what the position-versus-lag decomposition
    needs to test whether r depends on WHERE a pair sits rather than on how
    far apart its two windows are."""
    return (s + lag / 2.0 + width_bins / 2.0) * bin_width_s


def r_lag_profile(
    counts: NDArray,
    width_bins: int,
    n_splits: int = N_SPLITS_DEFAULT,
    rng: np.random.Generator | None = None,
    projection_fn=_leading_latent_projection,
) -> dict:
    """Cross-validated r as a function of LAG IN BINS at a FIXED window
    width: the window-count profile's own estimator (fit the leading latent
    on training trials, project held-out trials, centre each window's mean
    across trials only) with the pair set redefined so that width and lag
    are independent knobs. At (width, lag), the pairs are every (s, s+lag)
    with both windows of ``width`` bins inside the epoch and lag >= width
    (see :func:`_lag_pairs`); r is corr(mean(latent[:, s:s+width]),
    mean(latent[:, s+lag:s+lag+width])) averaged over those pairs. This
    replaces window-count as the free parameter for anything that needs to
    ask whether r falls with lag or merely with narrower windows -- a
    window-count sweep ties the two together, since window count sets both
    at once.

    ``projection_fn`` selects the leading-latent basis fit per split; see
    :func:`_r_lag_per_pair_one_split` for the contract and its use to swap
    PCA for an alternative noise model without touching the split, null or
    window logic.

    Returns a dict with ``status`` ("fitted" if at least 3 splits produced a
    usable profile), and ``lags`` mapping each integer lag (in bins) to its
    median, IQR, splits-fitted count, and analytic pair count.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (trials, units, bins)")
    n_trials, n_units, n_bins = counts.shape

    per_lag: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        train_idx, test_idx = perm[:half], perm[half:]
        if len(train_idx) < 3 or len(test_idx) < 3:
            continue
        result = _r_lag_one_split(counts, train_idx, test_idx, width_bins, projection_fn=projection_fn)
        if result is None:
            continue
        n_fitted += 1
        for lag, value in result.items():
            per_lag.setdefault(lag, []).append(value)

    status = "fitted" if n_fitted >= 3 else "fewer_than_three_fitted_splits"
    lags: dict[int, dict] = {}
    if status == "fitted":
        for lag, values in sorted(per_lag.items()):
            lags[lag] = {
                "r_median": float(np.median(values)),
                "r_iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
                "n_splits_fitted": len(values),
                "n_pairs": len(_lag_pairs(n_bins, width_bins, lag)),
            }
    return {
        "status": status, "n_splits_requested": int(n_splits), "n_splits_fitted": int(n_fitted),
        "width_bins": int(width_bins), "n_trials": int(n_trials), "n_units": int(n_units),
        "n_bins": int(n_bins), "lags": lags,
    }


def poisson_null_r_gap_profile(
    counts: NDArray,
    n_replicates: int = N_NULL_REPLICATES_DEFAULT,
    n_splits_per_replicate: int = NULL_SPLITS_PER_REPLICATE_DEFAULT,
    rng: np.random.Generator | None = None,
    n_windows: int = N_WINDOWS_DEFAULT,
) -> dict:
    """Per-session negative control: r(gap) recomputed on ``n_replicates``
    Poisson populations drawn from this session's own per-unit per-bin PSTH
    -- an identical rate profile on every simulated trial, so no
    trial-specific structure exists by construction -- through the identical
    estimator. A row without this null is not a row."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    n_trials = counts.shape[0]
    psth = np.clip(counts.mean(axis=0), 1e-6, None)  # (units, bins)

    replicate_gaps: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(n_replicates):
        simulated = rng.poisson(lam=np.broadcast_to(psth, counts.shape)).astype(float)
        result = r_gap_profile(simulated, n_splits=n_splits_per_replicate, rng=rng, n_windows=n_windows)
        if result["status"] != "fitted":
            continue
        n_fitted += 1
        for gap, g in result["gaps"].items():
            replicate_gaps.setdefault(gap, []).append(g["r_median"])

    gaps = {
        gap: {"r_null_median": float(np.median(values)), "r_null_p95": float(np.percentile(values, 95))}
        for gap, values in sorted(replicate_gaps.items())
    }
    return {"n_replicates_requested": int(n_replicates), "n_replicates_fitted": int(n_fitted), "gaps": gaps}


def binomial_thin(counts: NDArray, keep_probability: float, rng: np.random.Generator) -> NDArray:
    """Thin spike counts by independent binomial retention of each count --
    the operation that degrades firing rate without touching the underlying
    temporal structure, used to match a high-yield recording's effect size
    to a low-yield one."""
    if keep_probability >= 1.0:
        return counts.astype(float)
    return rng.binomial(counts.astype(int), keep_probability).astype(float)


def firing_rate_hz(counts: NDArray, window_s: float) -> float:
    return float(counts.sum() / (counts.shape[0] * counts.shape[1] * window_s))


def reachability_note() -> dict:
    """r(gap) is not floored: r can fall below its own null in either
    direction, and the gap effect r(1) - r(3) can be negative (rotation) as
    well as zero. Both contrasts a paired mean-difference test is run
    against are therefore reachable in both directions, unlike the floored
    `slow` share this observable replaces."""
    return {
        "statistic_is_floored": False,
        "reason": "r(gap) is a correlation, unbounded below by construction; a mean-difference "
                  "sign-flip test against the null is valid because the null hypothesis (no "
                  "difference) is reachable from both directions, not guaranteed to fail.",
    }


def paired_vs_null_contrast(observed: list[float], null: list[float], direction: str = "greater") -> dict:
    """Paired sign-flip test of ``observed`` against its matched ``null``
    value, session by session -- the shared shape of contrasts A, B and C.
    Reachability is checked once, in :func:`reachability_note`, since it
    does not depend on which quantity is being contrasted: r(gap) and any
    difference of r(gap) values is unbounded, so this test can fail in
    either direction.

    ``direction`` ("greater", "less", or "two-sided") is the ONE-SIDED
    alternative actually used to decide significance and is echoed back in
    the ``alternative`` field so a reader never has to guess it from the
    p-value alone. ``two_sided_p_value`` is always reported alongside it,
    computed from the same null draws at no extra permutation cost, because
    a one-sided p is easy to misread as two-sided and a null result used to
    retire a claim is exactly the case where that misreading matters most."""
    n_pairs = len(observed)
    min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs else 1.0
    if n_pairs < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
                "alternative": direction}
    test = paired_sign_flip_test(np.array(observed), np.array(null), alternative=direction)
    two_sided_p = permutation_pvalue(np.abs(test["null"]) >= np.abs(test["mean_diff"]))
    return {
        "status": "tested", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
        "alternative": direction,
        "mean_diff": test["mean_diff"], "p_value": test["p_value"], "two_sided_p_value": two_sided_p,
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "significant": bool(test["p_value"] <= 0.05),
    }


def profile_adjacent_steps(gaps_dicts: list[dict], w_max: int) -> dict:
    """Per-adjacent-step r(k) - r(k+1) test across a population of profiles
    (sessions or synthetic replicates), for k = 1 .. w_max - 2, each with
    its own paired sign-flip test and sign, plus a monotone-non-increasing
    flag that is False only if some step is a resampling-robust INCREASE (a
    significant negative difference). Decision rules in this module are
    stated on the whole profile -- every adjacent step -- and never on an
    endpoint ratio or a single summary scalar, because a ratio can land in a
    calibration regime that the discarded interior points would exclude."""
    steps: dict[str, dict] = {}
    any_significant_increase = False
    for k in range(1, w_max - 1):
        a = [g[k]["r_median"] for g in gaps_dicts if k in g and (k + 1) in g]
        b = [g[k + 1]["r_median"] for g in gaps_dicts if k in g and (k + 1) in g]
        n_pairs = len(a)
        min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs else 1.0
        if n_pairs < 4 or min_attainable_p > 0.05:
            steps[f"{k}_to_{k + 1}"] = {
                "status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
            }
            continue
        test = paired_sign_flip_test(np.array(a), np.array(b), alternative="two-sided")
        significant = bool(test["p_value"] <= 0.05)
        sign = "positive" if test["mean_diff"] > 0 else ("negative" if test["mean_diff"] < 0 else "flat")
        if significant and sign == "negative":
            any_significant_increase = True
        steps[f"{k}_to_{k + 1}"] = {
            "status": "tested", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
            "mean_diff": test["mean_diff"], "p_value": test["p_value"],
            "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
            "significant": significant, "sign": sign,
        }
    return {"steps": steps, "monotone_non_increasing": not any_significant_increase}


def _rotation_check(gaps_dicts: list[dict], w_max: int) -> dict:
    """Any gap k >= 2 whose pooled r(k) is significantly below zero (a
    sign-flip test against zero, not against a null population) is a
    rotation signature: the leading plane has come back around rather than
    merely decayed."""
    for k in range(2, w_max):
        values = [g[k]["r_median"] for g in gaps_dicts if k in g]
        n = len(values)
        min_attainable_p = 1.0 / (2 ** n) if n else 1.0
        if n < 4 or min_attainable_p > 0.05:
            continue
        test = paired_sign_flip_test(np.array(values), np.zeros(n), alternative="less")
        if test["p_value"] <= 0.05 and test["mean_diff"] < 0.0:
            return {"detected": True, "gap": k, "mean_r": test["mean_diff"], "p_value": test["p_value"]}
    return {"detected": False}


def classify_gap_profile(
    profiles: list[dict],
    poisson_nulls: list[dict],
    permutation_nulls: list[dict],
    w_max: int,
) -> dict:
    """The predeclared profile decision rule: classify a population of
    r(gap) profiles -- real sessions or synthetic replicates, each
    contributing one ``gaps`` dict from :func:`r_gap_profile`, matched
    one-to-one with a Poisson-null ``gaps`` dict and a per-unit-permutation-
    null ``gaps`` dict -- into one of four named branches.

    Contrast A (existence, nuisance-immune): r(k) for every k >= 2, against
    the Poisson null. Contrast B (decay beyond the shortest lag): the double
    difference r(2) - r(k_max), observed minus the same quantity computed on
    the Poisson null -- a guard against the estimator's own gap-count
    asymmetry (it averages more window pairs at short gaps than long ones,
    so even a null population's own r(2) - r(k_max) need not be exactly
    zero). Contrast C (the shortest-lag step): the double difference
    r(1) - r(2), observed minus the same quantity computed on the PER-UNIT
    PERMUTATION null, because the Poisson null -- white by construction --
    cannot speak to within-unit temporal autocorrelation.

    Priority, evaluated in this order: rotation first (a qualitatively
    different signature than a decay question); then whether contrast C
    clears the permutation null at all, because a negative answer here is
    the most consequential outcome this comparison can return and is
    reported first regardless of what B shows; then contrast B for a positive graded
    decay; the residual case is the fast-component-plus-floor branch.
    """
    k_max = w_max - 1
    step_report = profile_adjacent_steps(profiles, w_max)
    rotation = _rotation_check(profiles, w_max)

    contrast_a: dict[str, dict] = {}
    for k in range(2, w_max):
        obs, nul = [], []
        for g, n in zip(profiles, poisson_nulls):
            if k in g and k in n:
                obs.append(g[k]["r_median"])
                nul.append(n[k]["r_null_median"])
        contrast_a[str(k)] = paired_vs_null_contrast(obs, nul, direction="greater")

    obs_b, nul_b = [], []
    for g, n in zip(profiles, poisson_nulls):
        if 2 in g and k_max in g and 2 in n and k_max in n:
            obs_b.append(g[2]["r_median"] - g[k_max]["r_median"])
            nul_b.append(n[2]["r_null_median"] - n[k_max]["r_null_median"])
    contrast_b = paired_vs_null_contrast(obs_b, nul_b, direction="greater")

    obs_c, nul_c = [], []
    for g, n in zip(profiles, permutation_nulls):
        if 1 in g and 2 in g and 1 in n and 2 in n:
            obs_c.append(g[1]["r_median"] - g[2]["r_median"])
            nul_c.append(n[1]["r_null_median"] - n[2]["r_null_median"])
    contrast_c = paired_vs_null_contrast(obs_c, nul_c, direction="greater")

    b_positive_significant = bool(contrast_b.get("significant", False) and contrast_b.get("mean_diff", 0.0) > 0.0)
    c_significant = bool(contrast_c.get("significant", False))

    if rotation["detected"]:
        branch = "rotation"
    elif not c_significant:
        branch = "shortest_lag_is_nuisance"
    elif b_positive_significant:
        branch = "graded_decay"
    else:
        branch = "fast_component_plus_stable_floor"

    return {
        "reachability": reachability_note(),
        "w_max": w_max, "k_max": k_max,
        "adjacent_steps": step_report,
        "rotation_check": rotation,
        "contrast_a_existence_by_gap": contrast_a,
        "contrast_b_decay_beyond_shortest_lag": contrast_b,
        "contrast_c_shortest_lag_step_vs_permutation_null": contrast_c,
        "branch": branch,
        "branch_priority_note": "Evaluated in this order: rotation (a qualitatively different signature) "
                                 "first; then contrast C's significance against the permutation null, "
                                 "reported first regardless of B because a null result here is the most "
                                 "consequential outcome available; then contrast B for a positive graded "
                                 "decay; the residual case (C significant, B not) is the fast-component-"
                                 "plus-stable-floor branch.",
    }


def lag_reachability_note() -> dict:
    """d_perm(L) at any single lag can come out positive, zero or negative,
    and the SLOPE of d_perm across lags can come out negative, zero or
    positive -- both are unbounded differences of correlations, not floored
    statistics -- so every branch in :func:`classify_lag_profile` is
    reachable and a mean-difference or slope sign-flip test against zero is
    valid in every direction it is used."""
    return {
        "statistic_is_floored": False,
        "reason": "d_perm(L) is a difference of two correlations and the slope of that difference across "
                  "lags is an ordinary least-squares coefficient; neither is bounded below or above by "
                  "construction, so a sign-flip test against zero is reachable in both directions.",
    }


def _ols_slope(x: list[float] | NDArray, y: list[float] | NDArray) -> float | None:
    """Ordinary least-squares slope of y on x; None if fewer than 2 points
    or x carries no variance to regress against."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0.0:
        return None
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def slope_across_sessions_test(values: list[float], alternative: str = "two-sided") -> dict:
    """Paired sign-flip test of a per-session scalar (a slope or an endpoint
    difference) against zero, across sessions -- ONE test on the pooled
    per-session statistic, not one test per lag. Reports the one-sided
    p-value at the requested ``alternative`` plus the two-sided p from the
    same null draws, per the sidedness-disclosure rule every contrast in
    this module follows.

    ``significant_negative``/``significant_positive`` are always derived
    from the TWO-SIDED test and the sign of the mean, regardless of which
    ``alternative`` was requested for the headline p-value: a branch
    condition gated on "alternative == 'less'" is exactly the defect this
    module's decision rules were found to carry (a significantly positive
    slope satisfied "not significantly negative" and fell into a branch
    meant for a flat one). Every branch built on these two fields is
    two-sided by construction."""
    n = len(values)
    min_attainable_p = 1.0 / (2 ** n) if n else 1.0
    if n < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_sessions": n, "min_attainable_p": min_attainable_p,
                "alternative": alternative}
    values_arr = np.array(values)
    test = paired_sign_flip_test(values_arr, np.zeros(n), alternative=alternative)
    two_sided_test = test if alternative == "two-sided" else paired_sign_flip_test(values_arr, np.zeros(n), alternative="two-sided")
    two_sided_p = two_sided_test["p_value"]
    significant = bool(test["p_value"] <= 0.05)
    two_sided_significant = bool(two_sided_p <= 0.05)
    return {
        "status": "tested", "n_sessions": n, "min_attainable_p": min_attainable_p, "alternative": alternative,
        "mean_value": test["mean_diff"], "p_value": test["p_value"], "two_sided_p_value": two_sided_p,
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "significant_negative": bool(two_sided_significant and test["mean_diff"] < 0.0),
        "significant_positive": bool(two_sided_significant and test["mean_diff"] > 0.0),
        "significant": significant,
    }


def attenuation_constancy_check(profiles_narrow: list[dict], profiles_wide: list[dict], bin_width_s: float) -> dict:
    """Tests the attenuation model that licenses comparing r(gap) shapes
    across window widths: if a narrower window only attenuates r by a
    single scalar factor -- the same factor at every lag -- then the ratio
    r(w_narrow, L) / r(w_wide, L) is CONSTANT in lag at every lag both
    widths reach. Regresses each session's own ratio against lag in
    seconds and tests the pooled per-session slope against zero, two-sided,
    since either a rising or a falling ratio disproves constancy equally.
    If constancy holds, the width effect is measurement error and shapes
    may be compared across widths after scaling; if it does not, that is a
    finding about the estimator and every shape claim stays confined to a
    single width."""
    per_session_slope: list[float] = []
    ratio_samples: list[float] = []
    for g_narrow, g_wide in zip(profiles_narrow, profiles_wide):
        common = sorted(set(g_narrow) & set(g_wide))
        ratios, lags_s = [], []
        for lag in common:
            r_narrow = g_narrow[lag]["r_median"]
            r_wide = g_wide[lag]["r_median"]
            if r_wide == 0.0:
                continue
            ratios.append(r_narrow / r_wide)
            lags_s.append(lag * bin_width_s)
        if len(ratios) >= 2:
            slope = _ols_slope(lags_s, ratios)
            if slope is not None:
                per_session_slope.append(slope)
            ratio_samples.extend(ratios)
    test = slope_across_sessions_test(per_session_slope, alternative="two-sided") if per_session_slope else \
        {"status": "not_computed", "reason": "no sessions reach a lag common to both widths"}
    return {
        "n_sessions_with_common_lags": len(per_session_slope),
        "mean_ratio": float(np.mean(ratio_samples)) if ratio_samples else None,
        "ratio_slope_vs_lag_seconds_test": test,
        "constant_across_lag": bool(test.get("status") == "tested" and not test.get("significant", False)),
    }


def _lag_existence_and_rotation(
    profiles: list[dict], permutation_nulls: list[dict],
) -> tuple[dict, dict, int, int, bool, bool]:
    """Shared by every decision rule in this module: contrast A (existence
    per lag against the permutation null, FDR-corrected across the lag
    axis) and the rotation check (the raw observed r(lag), not d_perm,
    significantly negative at some lag -- the permutation null has a known
    small downward bias on d_perm, so testing d_perm against zero would
    mistake that bias for a rotation signature). Returns (contrast_a,
    rotation, n_lags_tested, n_lags_clearing, a_clears_any, a_clears_majority)."""
    lags_present = sorted(set().union(*[set(g) for g in profiles])) if profiles else []

    contrast_a: dict[str, dict] = {}
    a_pvalues, a_lag_order = [], []
    for lag in lags_present:
        obs, nul = [], []
        for g, n in zip(profiles, permutation_nulls):
            if lag in g and lag in n:
                obs.append(g[lag]["r_median"])
                nul.append(n[lag]["r_null_median"])
        result = paired_vs_null_contrast(obs, nul, direction="greater")
        contrast_a[str(lag)] = result
        if result.get("status") == "tested":
            a_pvalues.append(result["p_value"])
            a_lag_order.append(lag)
    if a_pvalues:
        fdr = fdr_bh(np.array(a_pvalues), alpha=0.05)
        for lag, q, rej in zip(a_lag_order, fdr["q_values"], fdr["reject"]):
            contrast_a[str(lag)]["fdr_q_value"] = float(q)
            contrast_a[str(lag)]["fdr_significant"] = bool(rej)
    n_lags_tested = len(a_lag_order)
    n_lags_clearing = sum(1 for lag in a_lag_order if contrast_a[str(lag)].get("fdr_significant"))
    a_clears_any = n_lags_clearing >= 1
    a_clears_majority = n_lags_tested > 0 and n_lags_clearing > n_lags_tested / 2.0

    rotation: dict = {"detected": False}
    for lag in lags_present:
        raw_r = [g[lag]["r_median"] for g in profiles if lag in g]
        n_pairs = len(raw_r)
        min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs else 1.0
        if n_pairs < 4 or min_attainable_p > 0.05:
            continue
        test = paired_sign_flip_test(np.array(raw_r), np.zeros(n_pairs), alternative="less")
        if test["p_value"] <= 0.05 and test["mean_diff"] < 0.0:
            rotation = {"detected": True, "lag_bins": lag, "mean_r": test["mean_diff"], "p_value": test["p_value"]}
            break

    return contrast_a, rotation, n_lags_tested, n_lags_clearing, a_clears_any, a_clears_majority


def classify_lag_profile(
    profiles: list[dict],
    poisson_nulls: list[dict],
    permutation_nulls: list[dict],
    width_bins: int,
    bin_width_s: float,
) -> dict:
    """The predeclared decision rule on the SHAPE of d_perm(L) -- the
    observed r minus the per-unit permutation null, at fixed window width --
    across lag in seconds, per session then pooled by paired sign-flip test.

    Contrast A (existence, per lag, against the permutation null): tested at
    every lag reached by at least 4 sessions, FDR-corrected across the lag
    axis. Contrast B (shape): the per-session OLS slope of d_perm on lag in
    seconds, tested twice -- including and excluding L = width, the single
    lag where the two windows touch and the only point a within-window
    boundary effect could contaminate -- so that a branch resting on a
    multi-lag decay (``graded_decay``) is distinguished from one resting on
    that single adjacent point alone (``fast_component_plus_flat_floor``).
    Contrast C is that adjacency point reported on its own, plus the
    endpoint difference d_perm(L_min) - d_perm(L_max) as a secondary.

    Priority: rotation (d_perm significantly negative at any lag, a
    qualitatively different signature) first; then whether contrast A
    clears at NO lag at all, the most consequential null result available;
    then whether the slope EXCLUDING the adjacency point is itself
    significantly negative (a decay that survives dropping the one point a
    boundary effect could fake); then whether the slope INCLUDING it is
    significant while excluding it is not (the fast-component-then-flat
    signature); then whether existence clears a majority of lags with no
    significant slope either way (a flat, non-decaying, above-floor state);
    anything else is reported off the branch list rather than forced onto
    the nearest one, per this module's standing rule on predeclared tests.
    """
    adjacency_lag = width_bins
    contrast_a, rotation, n_lags_tested, n_lags_clearing, a_clears_any, a_clears_majority = \
        _lag_existence_and_rotation(profiles, permutation_nulls)

    per_session_slope_incl, per_session_slope_excl, endpoint_diffs = [], [], []
    for g, n in zip(profiles, permutation_nulls):
        common_lags = sorted(set(g) & set(n))
        if len(common_lags) >= 2:
            x = [lag * bin_width_s for lag in common_lags]
            y = [g[lag]["r_median"] - n[lag]["r_null_median"] for lag in common_lags]
            slope = _ols_slope(x, y)
            if slope is not None:
                per_session_slope_incl.append(slope)
            endpoint_diffs.append(y[0] - y[-1])
        common_excl = [lag for lag in common_lags if lag != adjacency_lag]
        if len(common_excl) >= 2:
            x = [lag * bin_width_s for lag in common_excl]
            y = [g[lag]["r_median"] - n[lag]["r_null_median"] for lag in common_excl]
            slope = _ols_slope(x, y)
            if slope is not None:
                per_session_slope_excl.append(slope)

    contrast_b_including_adjacency = slope_across_sessions_test(per_session_slope_incl, alternative="less")
    contrast_b_excluding_adjacency = slope_across_sessions_test(per_session_slope_excl, alternative="less")
    contrast_c_endpoint_difference = slope_across_sessions_test(endpoint_diffs, alternative="greater") if endpoint_diffs else \
        {"status": "not_computed"}

    b_excl_significant_negative = bool(contrast_b_excluding_adjacency.get("significant_negative", False))
    b_incl_significant_negative = bool(contrast_b_including_adjacency.get("significant_negative", False))
    verdict_depends_on_adjacent_lag = b_incl_significant_negative != b_excl_significant_negative

    if rotation["detected"]:
        branch = "rotation"
    elif not a_clears_any:
        branch = "no_state_above_the_within_unit_floor"
    elif b_excl_significant_negative:
        branch = "graded_decay"
    elif b_incl_significant_negative:
        branch = "fast_component_plus_flat_floor"
    elif a_clears_majority:
        branch = "flat_cross_unit_state"
    else:
        branch = "unattributed_off_branch_list"

    return {
        "reachability": lag_reachability_note(),
        "width_bins": int(width_bins), "bin_width_s": bin_width_s, "adjacency_lag_bins": int(adjacency_lag),
        "n_lags_tested": n_lags_tested, "n_lags_clearing_fdr": n_lags_clearing,
        "contrast_a_existence_by_lag": contrast_a,
        "rotation_check": rotation,
        "contrast_b_slope_including_adjacency": contrast_b_including_adjacency,
        "contrast_b_slope_excluding_adjacency": contrast_b_excluding_adjacency,
        "contrast_c_endpoint_difference": contrast_c_endpoint_difference,
        "verdict_depends_on_adjacent_lag": verdict_depends_on_adjacent_lag,
        "branch": branch,
        "branch_priority_note": "Evaluated in this order: rotation first (a qualitatively different "
                                 "signature); then whether existence clears at NO lag at all (the most "
                                 "consequential null result available); then whether the slope EXCLUDING "
                                 "the single adjacency lag is itself significantly negative (a decay that "
                                 "survives dropping the one point a window-boundary effect could fake, "
                                 "'graded_decay'); then whether the slope INCLUDING it is significant while "
                                 "excluding it is not (the decay is carried entirely by that one point, "
                                 "'fast_component_plus_flat_floor'); then whether existence clears a "
                                 "majority of lags with no significant slope either way "
                                 "('flat_cross_unit_state'); anything else is reported off the branch list "
                                 "explicitly rather than forced onto the nearest one.",
    }


# ── Patient-clustered lag inference ─────────────────────────────────────────

def _patient_cluster_key(row: dict) -> tuple[str, str]:
    """A patient identifier is only unique within a released dataset.

    The human deposits use independent patient namespaces.  Keeping the
    dataset in the cluster key prevents a coincidental re-use of an ID from
    joining two unrelated people when results from deposits are pooled.
    """
    return str(row["dataset"]), str(row["patient"])


def _cluster_label(cluster: tuple[str, str]) -> str:
    return f"{cluster[0]}:{cluster[1]}"


def _clustered_test(values: dict[tuple[str, str], float], alternative: str, seed: int) -> dict:
    """Sign-flip and bootstrap inference with *patients* as the unit.

    Session estimates are first reduced to an equally weighted within-patient
    mean.  The sign-flip test and its percentile bootstrap CI then resample
    those patient means, never the individual sessions.  This is appropriate
    for the present nested design, where sessions from a person are not
    independent biological replicates.  The returned per-patient estimates
    make the reduction auditable without exposing any trial-level data.
    """
    ordered = sorted(values)
    x = np.asarray([values[p] for p in ordered], dtype=float)
    n = len(x)
    min_attainable = 1.0 / (2 ** n) if n else 1.0
    if n < 4 or min_attainable > 0.05:
        return {
            "status": "underpowered_by_construction", "n_patients": n,
            "min_attainable_p_exact_sign_flip": min_attainable,
            "alternative": alternative,
            "per_patient_equal_weight_mean": {_cluster_label(p): float(values[p]) for p in ordered},
        }
    test = paired_sign_flip_test(
        x, np.zeros(n), alternative=alternative, rng=np.random.default_rng(seed))
    two_sided = test if alternative == "two-sided" else paired_sign_flip_test(
        x, np.zeros(n), alternative="two-sided", rng=np.random.default_rng(seed + 1))
    two_sided_p = float(two_sided["p_value"])
    return {
        "status": "tested", "n_patients": n,
        "min_attainable_p_exact_sign_flip": min_attainable,
        "alternative": alternative,
        "mean_value": float(test["mean_diff"]), "p_value": float(test["p_value"]),
        "two_sided_p_value": two_sided_p,
        "ci_lower": float(test["ci_lower"]), "ci_upper": float(test["ci_upper"]),
        "significant": bool(test["p_value"] <= 0.05),
        "significant_negative": bool(two_sided_p <= 0.05 and test["mean_diff"] < 0.0),
        "significant_positive": bool(two_sided_p <= 0.05 and test["mean_diff"] > 0.0),
        "per_patient_equal_weight_mean": {_cluster_label(p): float(values[p]) for p in ordered},
    }


def _patient_series_from_rows(rows: list[dict], width_bins: int, null_name: str) -> tuple[dict, dict]:
    """Reduce fitted session rows to equally weighted patient lag profiles.

    ``null_name`` is either ``permutation`` or ``poisson``.  The result is a
    patient -> {lag: mean session-level d(L)} mapping and bookkeeping that
    records how many session rows each patient contributed at every lag.
    """
    null_field = "null_permutation" if null_name == "permutation" else "null_poisson"
    values: dict[tuple[str, str], dict[int, list[float]]] = {}
    sessions: dict[tuple[str, str], dict[int, int]] = {}
    for row in rows:
        if row.get("width_bins") != width_bins or row.get("profile", {}).get("status") != "fitted":
            continue
        null = row.get(null_field)
        if not null:
            continue
        profile_lags = row["profile"].get("lags", {})
        null_lags = null.get("lags", {})
        patient = _patient_cluster_key(row)
        for raw_lag in set(profile_lags) & set(null_lags):
            lag = int(raw_lag)
            observed = profile_lags[raw_lag]["r_median"]
            reference = null_lags[raw_lag]["r_null_median"]
            if not np.isfinite(observed) or not np.isfinite(reference):
                continue
            values.setdefault(patient, {}).setdefault(lag, []).append(float(observed - reference))
            sessions.setdefault(patient, {}).setdefault(lag, 0)
            sessions[patient][lag] += 1
    reduced = {patient: {lag: float(np.mean(v)) for lag, v in by_lag.items()}
               for patient, by_lag in values.items()}
    return reduced, sessions


def _patient_raw_r_from_rows(rows: list[dict], width_bins: int) -> dict:
    """Patient means of observed r(L), used only for the rotation guard."""
    values: dict[tuple[str, str], dict[int, list[float]]] = {}
    for row in rows:
        if row.get("width_bins") != width_bins or row.get("profile", {}).get("status") != "fitted":
            continue
        patient = _patient_cluster_key(row)
        for raw_lag, item in row["profile"].get("lags", {}).items():
            lag = int(raw_lag)
            value = item["r_median"]
            if np.isfinite(value):
                values.setdefault(patient, {}).setdefault(lag, []).append(float(value))
    return {patient: {lag: float(np.mean(v)) for lag, v in by_lag.items()}
            for patient, by_lag in values.items()}


def patient_clustered_lag_inference(
    rows: list[dict], width_bins: int, bin_width_s: float, seed: int = 0,
) -> dict:
    """Patient-clustered counterpart of :func:`classify_lag_profile`.

    This is deliberately a separate output rather than silently replacing
    historical session profiles.  The profiles remain useful descriptive
    data, but all F1 existence p-values, CIs and BH correction in this
    companion use one equal-weighted observation per patient.
    """
    d_perm, session_counts = _patient_series_from_rows(rows, width_bins, "permutation")
    raw_r = _patient_raw_r_from_rows(rows, width_bins)
    lags = sorted(set().union(*(set(v) for v in d_perm.values()))) if d_perm else []
    contrast_a, p_values, p_lags = {}, [], []
    for lag in lags:
        patient_values = {p: series[lag] for p, series in d_perm.items() if lag in series}
        test = _clustered_test(patient_values, "greater", seed + 1000 + lag)
        test["n_session_rows_contributing"] = int(sum(session_counts[p].get(lag, 0) for p in patient_values))
        contrast_a[str(lag)] = test
        if test.get("status") == "tested":
            p_values.append(test["p_value"])
            p_lags.append(lag)
    if p_values:
        fdr = fdr_bh(np.asarray(p_values), alpha=0.05)
        for lag, q, rejected in zip(p_lags, fdr["q_values"], fdr["reject"]):
            contrast_a[str(lag)]["fdr_q_value"] = float(q)
            contrast_a[str(lag)]["fdr_significant"] = bool(rejected)

    rotation_by_lag, rotation_ps, rotation_lags = {}, [], []
    for lag in sorted(set().union(*(set(v) for v in raw_r.values()))) if raw_r else []:
        test = _clustered_test({p: series[lag] for p, series in raw_r.items() if lag in series}, "less", seed + 2000 + lag)
        rotation_by_lag[str(lag)] = test
        if test.get("status") == "tested":
            rotation_ps.append(test["p_value"])
            rotation_lags.append(lag)
    if rotation_ps:
        fdr = fdr_bh(np.asarray(rotation_ps), alpha=0.05)
        for lag, q, rejected in zip(rotation_lags, fdr["q_values"], fdr["reject"]):
            rotation_by_lag[str(lag)]["fdr_q_value"] = float(q)
            rotation_by_lag[str(lag)]["fdr_significant"] = bool(rejected)
    rotation_lag = next((lag for lag in rotation_lags if rotation_by_lag[str(lag)].get("fdr_significant") and
                         rotation_by_lag[str(lag)].get("mean_value", 0.0) < 0.0), None)

    def _slopes(exclude: int | None) -> dict:
        values = {}
        for patient, series in d_perm.items():
            available = sorted(lag for lag in series if lag != exclude)
            slope = _ols_slope([lag * bin_width_s for lag in available], [series[lag] for lag in available])
            if slope is not None:
                values[patient] = slope
        return values

    slopes_including = _slopes(None)
    slopes_excluding = _slopes(width_bins)
    endpoint = {}
    for patient, series in d_perm.items():
        available = sorted(series)
        if len(available) >= 2:
            endpoint[patient] = series[available[0]] - series[available[-1]]
    slope_including = _clustered_test(slopes_including, "less", seed + 3000)
    slope_excluding = _clustered_test(slopes_excluding, "less", seed + 3001)
    endpoint_test = _clustered_test(endpoint, "greater", seed + 3002)
    n_tested = len(p_lags)
    n_clearing = sum(bool(contrast_a[str(lag)].get("fdr_significant")) for lag in p_lags)
    any_existence = n_clearing > 0
    majority_existence = n_tested > 0 and n_clearing > n_tested / 2.0
    if rotation_lag is not None:
        branch = "rotation"
    elif not any_existence:
        branch = "no_state_above_the_within_unit_floor"
    elif slope_excluding.get("significant_negative", False):
        branch = "graded_decay"
    elif slope_including.get("significant_negative", False):
        branch = "fast_component_plus_flat_floor"
    elif majority_existence:
        branch = "flat_cross_unit_state"
    else:
        branch = "unattributed_off_branch_list"
    rows_fitted = [r for r in rows if r.get("width_bins") == width_bins and
                   r.get("profile", {}).get("status") == "fitted" and r.get("null_permutation")]
    patients = sorted(d_perm)
    return {
        "method": (
            "For each lag, session-level observed-minus-permutation-null values are first averaged within "
            "each patient (dataset-qualified patient identity), then every patient receives equal weight. "
            "One-sample sign flips and percentile bootstrap confidence intervals resample those patient "
            "means; Benjamini--Hochberg correction is applied once across all reachable lags at this width. "
            "Session rows remain descriptive only and are not independent inferential observations."
        ),
        "inference_unit": "patient", "width_bins": int(width_bins), "bin_width_s": float(bin_width_s),
        "n_patients": len(patients), "n_session_rows_descriptive": len(rows_fitted),
        "patient_session_count": {
            "min": int(min(sum(session_counts.get(p, {}).values()) / max(1, len(session_counts.get(p, {}))) for p in patients)) if patients else 0,
            "median": float(np.median([sum(session_counts.get(p, {}).values()) / max(1, len(session_counts.get(p, {}))) for p in patients])) if patients else 0.0,
            "max": int(max(sum(session_counts.get(p, {}).values()) / max(1, len(session_counts.get(p, {}))) for p in patients)) if patients else 0,
        },
        "contrast_a_existence_by_lag": contrast_a,
        "rotation_check": {"detected": rotation_lag is not None, "fdr_corrected_by_lag": rotation_by_lag,
                           "first_detected_lag_bins": rotation_lag},
        "contrast_b_slope_including_adjacency": slope_including,
        "contrast_b_slope_excluding_adjacency": slope_excluding,
        "contrast_c_endpoint_difference": endpoint_test,
        "n_lags_tested": n_tested, "n_lags_clearing_fdr": n_clearing,
        "verdict_depends_on_adjacent_lag": bool(slope_including.get("significant_negative", False)) !=
                                         bool(slope_excluding.get("significant_negative", False)),
        "branch": branch,
    }


def patient_clustered_segmented_inference(
    rows: list[dict], width_bins: int, bin_width_s: float, breakpoint_bins: int, seed: int = 0,
) -> dict:
    """Patient-clustered F2 tests at a pre-specified breakpoint.

    The breakpoint is supplied by the caller rather than selected during the
    test.  That keeps this clustered confirmation from using the same patient
    data to choose and then test a segment boundary.  It is therefore a
    confirmation of the already declared 0.8-s descriptive boundary, not a
    new claim that the breakpoint itself has been estimated precisely.
    """
    base = patient_clustered_lag_inference(rows, width_bins, bin_width_s, seed)
    d_perm, _ = _patient_series_from_rows(rows, width_bins, "permutation")
    d_pois, _ = _patient_series_from_rows(rows, width_bins, "poisson")
    all_lags = sorted(set().union(*(set(v) for v in d_perm.values()))) if d_perm else []
    if not all_lags or breakpoint_bins not in all_lags:
        return {**base, "status": "not_computable", "reason": "pre-specified breakpoint is not reachable"}
    lag_min, lag_max = all_lags[0], all_lags[-1]

    def _segment_test(series_by_patient: dict, lo: int, hi: int, exclude: int | None,
                      alternative: str, local_seed: int) -> dict:
        values = {}
        for patient, series in series_by_patient.items():
            lags = sorted(lag for lag in series if lo <= lag <= hi and lag != exclude)
            slope = _ols_slope([lag * bin_width_s for lag in lags], [series[lag] for lag in lags])
            if slope is not None:
                values[patient] = slope
        result = _clustered_test(values, alternative, local_seed)
        return {"range_bins": [int(lo), int(hi)], "range_seconds": [lo * bin_width_s, hi * bin_width_s],
                "excluded_lag_bins": exclude, "n_patients_contributing": len(values), "test": result}

    by_statistic = {}
    for name, series in (("d_perm", d_perm), ("d_pois", d_pois)):
        early_incl = _segment_test(series, lag_min, breakpoint_bins, None, "less", seed + 4100)
        early_excl = _segment_test(series, lag_min, breakpoint_bins, width_bins, "less", seed + 4101)
        late = _segment_test(series, breakpoint_bins, lag_max, None, "less", seed + 4102)
        whole = _segment_test(series, lag_min, lag_max, None, "less", seed + 4103)
        endpoints = {p: s[lag_min] - s[breakpoint_bins] for p, s in series.items()
                     if lag_min in s and breakpoint_bins in s}
        by_statistic[name] = {
            "early_segment_including_adjacency": early_incl,
            "early_segment_excluding_adjacency": early_excl,
            "late_segment": late, "whole_range_slope_dilution_comparison": whole,
            "endpoint_drop_lag_min_minus_breakpoint": _clustered_test(endpoints, "greater", seed + 4104),
        }
    floor_values = {p: s[breakpoint_bins] - s[lag_max] for p, s in d_perm.items()
                    if breakpoint_bins in s and lag_max in s}
    floor = _clustered_test(floor_values, "greater", seed + 4200)
    early = by_statistic["d_perm"]["early_segment_excluding_adjacency"]["test"]
    early_incl = by_statistic["d_perm"]["early_segment_including_adjacency"]["test"]
    late = by_statistic["d_perm"]["late_segment"]["test"]
    if base["rotation_check"]["detected"]:
        branch = "rotation"
    elif base["n_lags_clearing_fdr"] == 0:
        branch = "no_state_above_the_within_unit_floor"
    elif early.get("significant_positive") or early_incl.get("significant_positive") or late.get("significant_positive"):
        branch = "unattributed_off_branch_list"
    elif early.get("significant_negative"):
        branch = "graded_decay" if late.get("significant_negative") else "fast_component_plus_flat_floor"
    elif late.get("significant_negative"):
        branch = "graded_decay"
    elif base["n_lags_clearing_fdr"] > base["n_lags_tested"] / 2.0:
        branch = "flat_cross_unit_state"
    else:
        branch = "unattributed_off_branch_list"
    return {
        **base, "status": "tested", "pre_specified_breakpoint_bins": int(breakpoint_bins),
        "pre_specified_breakpoint_s": float(breakpoint_bins * bin_width_s),
        "breakpoint_scope_note": (
            "The 0.8-s boundary was the live report's already-declared descriptive breakpoint. It is held "
            "fixed here, so patient-clustered CIs and p-values do not reselect a breakpoint on the same data; "
            "this analysis does not claim a precise breakpoint estimate."
        ),
        "segmented_contrasts": {
            "breakpoint_bins": int(breakpoint_bins), "breakpoint_s": float(breakpoint_bins * bin_width_s),
            "lag_min_bins": int(lag_min), "lag_max_bins": int(lag_max), "adjacency_lag_bins": int(width_bins),
            "by_statistic": by_statistic,
            "double_difference_floor_vs_persistent_component": floor,
        },
        "early_segment_significant_negative": bool(early.get("significant_negative")),
        "early_segment_significant_positive": bool(early.get("significant_positive") or early_incl.get("significant_positive")),
        "late_segment_significant_negative": bool(late.get("significant_negative")),
        "late_segment_significant_positive": bool(late.get("significant_positive")),
        "branch": branch,
    }


def _d_series(profiles: list[dict], nulls: list[dict], null_value_key: str) -> list[dict[int, float]]:
    """Per-session d(L) = r_obs(L) - r_null(L) at every lag common to both,
    for either null (``null_value_key`` is "r_null_median" for both the
    Poisson and the per-unit permutation null -- the two dicts have the
    same shape)."""
    return [
        {lag: g[lag]["r_median"] - n[lag][null_value_key] for lag in sorted(set(g) & set(n))}
        for g, n in zip(profiles, nulls)
    ]


def component_series(profiles: list[dict], nulls: list[dict], null_value_key: str) -> tuple[list[dict[int, float]], list[dict[int, float]]]:
    """The two terms :func:`_d_series` differences, kept SEPARATE rather
    than only their difference: per-session r_obs(L) and per-session
    r_null(L), at the same lags :func:`_d_series` would use for the same
    ``profiles``/``nulls``/``null_value_key`` -- so that a d_perm slope's
    two component slopes (see :func:`per_session_slopes_in_range`) can be
    reported alongside it rather than only their arithmetic difference,
    which by construction carries no information about which term moved."""
    r_obs = [{lag: g[lag]["r_median"] for lag in sorted(set(g) & set(n))} for g, n in zip(profiles, nulls)]
    r_null = [{lag: n[lag][null_value_key] for lag in sorted(set(g) & set(n))} for g, n in zip(profiles, nulls)]
    return r_obs, r_null


def per_session_slopes_in_range(
    series_per_session: list[dict[int, float]],
    bin_width_s: float,
    lag_range_bins: tuple[int, int],
    exclude_lag_bins: int | None = None,
) -> list[float]:
    """The per-session OLS slopes :func:`segmented_slope_test` pools into
    its cross-session test, returned as the raw list rather than only the
    pooled summary -- what a between-session correlation or regression
    against another per-session quantity needs (e.g. correlating a d_perm
    slope with the r_null slope that is one of its two components), which
    the pooled test alone does not expose. Restricted to the CLOSED
    interval ``lag_range_bins``, optionally dropping one lag."""
    lo, hi = lag_range_bins
    per_session_slope = []
    for series in series_per_session:
        lags = [lag for lag in series if lo <= lag <= hi and lag != exclude_lag_bins]
        if len(lags) < 2:
            continue
        x = [lag * bin_width_s for lag in lags]
        y = [series[lag] for lag in lags]
        slope = _ols_slope(x, y)
        if slope is not None:
            per_session_slope.append(slope)
    return per_session_slope


def segmented_slope_test(
    series_per_session: list[dict[int, float]],
    bin_width_s: float,
    lag_range_bins: tuple[int, int],
    exclude_lag_bins: int | None = None,
    alternative: str = "less",
) -> dict:
    """Per-session OLS slope of a d(L) series on lag in seconds, restricted
    to the CLOSED interval ``lag_range_bins`` and optionally dropping one
    lag (the adjacency lag) -- a slope is a claim about a range, so the
    range in seconds is part of the return value, not left to be
    reconstructed from the caller's bookkeeping. Pooled across sessions by
    :func:`slope_across_sessions_test`, which reports both the requested
    one-sided p and the two-sided p, and derives significant_negative/
    significant_positive from the two-sided test regardless of
    ``alternative`` (see that function's docstring)."""
    lo, hi = lag_range_bins
    per_session_slope = per_session_slopes_in_range(series_per_session, bin_width_s, lag_range_bins, exclude_lag_bins)
    test = slope_across_sessions_test(per_session_slope, alternative=alternative) if per_session_slope else \
        {"status": "not_computed", "reason": "fewer than 2 usable lags in range for any session"}
    return {
        "range_bins": [int(lo), int(hi)], "range_seconds": [lo * bin_width_s, hi * bin_width_s],
        "excluded_lag_bins": exclude_lag_bins, "n_sessions_contributing": len(per_session_slope),
        "test": test,
    }


def fit_segmented_breakpoint(lags_bins: list[int], values: dict[int, float], bin_width_s: float) -> dict | None:
    """Continuous two-slope (hinge) regression of a single d(L) series on
    lag in seconds, breakpoint chosen by grid search over the interior lags
    to minimise total squared error: y = a + b*x + c*max(0, x - breakpoint),
    fit by ordinary least squares for every candidate breakpoint, keeping
    the one with the lowest SSE. Continuity is enforced by construction (the
    hinge term is zero at the breakpoint from both sides), unlike fitting
    two independent lines and only comparing their SSE. Returns None if
    fewer than 5 lags are available (need at least 2 candidate breakpoints
    with 2 points on each side)."""
    lags_sorted = sorted(lags_bins)
    if len(lags_sorted) < 5:
        return None
    x = np.array([lag * bin_width_s for lag in lags_sorted])
    y = np.array([values[lag] for lag in lags_sorted])
    best = None
    for bp_bin in lags_sorted[2:-2]:
        bp = bp_bin * bin_width_s
        hinge = np.maximum(0.0, x - bp)
        design = np.stack([np.ones_like(x), x, hinge], axis=1)
        coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 3:
            continue
        pred = design @ coef
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {"breakpoint_bins": int(bp_bin), "breakpoint_s": float(bp),
                    "early_slope": float(coef[1]), "late_slope": float(coef[1] + coef[2]), "sse": sse}
    return best


def segmented_breakpoint_summary(series_per_session: list[dict[int, float]], bin_width_s: float) -> dict:
    """Per-session breakpoint fits (:func:`fit_segmented_breakpoint`) pooled
    into a median breakpoint and its spread across sessions, plus the median
    of each session's own early/late slope -- the breakpoint ESTIMATED
    rather than read off a pooled profile by eye."""
    fits = []
    for series in series_per_session:
        fit = fit_segmented_breakpoint(sorted(series), series, bin_width_s)
        if fit is not None:
            fits.append(fit)
    if not fits:
        return {"status": "not_computed", "n_sessions": 0}
    breakpoints_s = [f["breakpoint_s"] for f in fits]
    return {
        "status": "tested", "n_sessions": len(fits),
        "median_breakpoint_s": float(np.median(breakpoints_s)),
        "breakpoint_iqr_s": [float(np.percentile(breakpoints_s, 25)), float(np.percentile(breakpoints_s, 75))],
        "median_early_slope": float(np.median([f["early_slope"] for f in fits])),
        "median_late_slope": float(np.median([f["late_slope"] for f in fits])),
        "per_session_fits": fits,
    }


def fit_breakpoint_from_pooled_profile(series_per_session: list[dict[int, float]], bin_width_s: float) -> dict | None:
    """Pools per-session d(L) series into a single across-session MEDIAN
    profile (dividing single-session noise by roughly sqrt(n_sessions))
    before applying :func:`fit_segmented_breakpoint` once. See
    :func:`breakpoint_estimator_calibration` for why this is the preferred
    breakpoint estimate at this project's typical session count and noise
    level: an unpenalised SSE grid search over a short, noisy SINGLE-session
    profile is biased toward late candidate breakpoints, because a late
    "flat" segment fit to only a couple of noisy edge points can spuriously
    lower the SSE; pooling first removes most of that noise before the same
    search is run."""
    if not series_per_session:
        return None
    lags = sorted(set().union(*[set(s) for s in series_per_session]))
    pooled = {lag: float(np.median([s[lag] for s in series_per_session if lag in s]))
              for lag in lags if any(lag in s for s in series_per_session)}
    return fit_segmented_breakpoint(lags, pooled, bin_width_s)


def breakpoint_estimator_calibration(
    n_sessions: int, noise_sem: float, lags_bins: list[int], bin_width_s: float,
    true_breakpoint_bins: int, early_slope_r_per_s: float, floor_value: float,
    n_trials_sim: int = 200, seed: int = 400,
) -> dict:
    """Validates the breakpoint estimator on a planted two-piece profile at
    THIS ARM'S OWN session count and empirical per-session noise level,
    before it is trusted to declare a breakpoint for anything: compares the
    per-session-fit-then-median approach (:func:`segmented_breakpoint_
    summary`, literally what 1.2(a) asks for) against fitting once on the
    across-session pooled profile (:func:`fit_breakpoint_from_pooled_
    profile`). Whichever recovers the planted breakpoint with the smaller
    absolute bias is the one used as this artifact's declared breakpoint;
    the other stays in the artifact as a characterisation of single-session
    noise, not silently dropped."""
    rng = np.random.default_rng(seed)
    lags_sorted = sorted(lags_bins)
    bp_s = true_breakpoint_bins * bin_width_s

    def true_val(lag: int) -> float:
        t = lag * bin_width_s
        return floor_value + early_slope_r_per_s * (bp_s - t) if t <= bp_s else floor_value

    per_session_median_recovered, pooled_recovered = [], []
    for _ in range(n_trials_sim):
        sessions = [{lag: true_val(lag) + rng.normal(scale=noise_sem) for lag in lags_sorted} for _ in range(n_sessions)]
        per_session_fits = [f for f in (fit_segmented_breakpoint(lags_sorted, s, bin_width_s) for s in sessions) if f is not None]
        if per_session_fits:
            per_session_median_recovered.append(float(np.median([f["breakpoint_bins"] for f in per_session_fits])))
        pooled_fit = fit_breakpoint_from_pooled_profile(sessions, bin_width_s)
        if pooled_fit is not None:
            pooled_recovered.append(pooled_fit["breakpoint_bins"])

    def summarize(vals: list[float]) -> dict:
        if not vals:
            return {"status": "not_computed"}
        arr = np.array(vals, dtype=float)
        return {
            "status": "tested", "n_trials": len(arr), "median_recovered_bins": float(np.median(arr)),
            "bias_bins": float(np.median(arr) - true_breakpoint_bins),
            "iqr_bins": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        }

    per_session_summary = summarize(per_session_median_recovered)
    pooled_summary = summarize(pooled_recovered)
    pooled_preferred = bool(
        pooled_summary.get("status") == "tested" and
        (per_session_summary.get("status") != "tested"
         or abs(pooled_summary["bias_bins"]) <= abs(per_session_summary.get("bias_bins", np.inf)))
    )
    return {
        "true_breakpoint_bins": true_breakpoint_bins, "n_sessions": n_sessions, "noise_sem": noise_sem,
        "per_session_fit_then_median": per_session_summary,
        "pooled_profile_fit": pooled_summary,
        "pooled_profile_preferred": pooled_preferred,
        "criterion": "the method with the smaller absolute bias (median recovered breakpoint minus the "
                     "planted one, in bins) at this arm's own session count and empirical per-session noise "
                     "level is used as the declared breakpoint fed into every other segmented contrast; the "
                     "rejected method's spread is still reported, never silently dropped.",
    }


def segmented_shape_contrasts(
    profiles: list[dict],
    poisson_nulls: list[dict],
    permutation_nulls: list[dict],
    bin_width_s: float,
    breakpoint_bins: int,
    adjacency_lag_bins: int,
) -> dict:
    """The full 1.1 battery at a DECLARED breakpoint: on both d_perm and
    d_pois, the early-segment slope (including and excluding the adjacency
    lag), the late-segment slope, the whole-range slope as a dilution
    comparison, the endpoint drop d(L_min) - d(breakpoint), and the double
    difference [r_obs(breakpoint) - r_obs(L_max)] - [r_perm(breakpoint) -
    r_perm(L_max)] that would license calling the late segment a persistent
    component rather than a bound on its decay."""
    lags_present = sorted(set().union(*[set(g) for g in profiles])) if profiles else []
    if not lags_present:
        return {"status": "not_computed"}
    lag_min, lag_max = lags_present[0], lags_present[-1]

    by_statistic = {}
    for stat_name, nulls, null_key in (("d_perm", permutation_nulls, "r_null_median"),
                                        ("d_pois", poisson_nulls, "r_null_median")):
        series = _d_series(profiles, nulls, null_key)
        early_incl = segmented_slope_test(series, bin_width_s, (lag_min, breakpoint_bins), None, "less")
        early_excl = segmented_slope_test(series, bin_width_s, (lag_min, breakpoint_bins), adjacency_lag_bins, "less")
        late = segmented_slope_test(series, bin_width_s, (breakpoint_bins, lag_max), None, "less")
        whole = segmented_slope_test(series, bin_width_s, (lag_min, lag_max), None, "less")

        endpoint_drop = [s[lag_min] - s[breakpoint_bins] for s in series if lag_min in s and breakpoint_bins in s]
        endpoint_drop_test = slope_across_sessions_test(endpoint_drop, alternative="greater") if len(endpoint_drop) >= 4 else \
            {"status": "underpowered_by_construction", "n_sessions": len(endpoint_drop)}

        by_statistic[stat_name] = {
            "early_segment_including_adjacency": early_incl,
            "early_segment_excluding_adjacency": early_excl,
            "late_segment": late,
            "whole_range_slope_dilution_comparison": whole,
            "endpoint_drop_lag_min_minus_breakpoint": endpoint_drop_test,
        }

    double_diff = []
    for g, n in zip(profiles, permutation_nulls):
        if breakpoint_bins in g and lag_max in g and breakpoint_bins in n and lag_max in n:
            obs_diff = g[breakpoint_bins]["r_median"] - g[lag_max]["r_median"]
            null_diff = n[breakpoint_bins]["r_null_median"] - n[lag_max]["r_null_median"]
            double_diff.append(obs_diff - null_diff)
    double_diff_test = slope_across_sessions_test(double_diff, alternative="greater") if len(double_diff) >= 4 else \
        {"status": "underpowered_by_construction", "n_sessions": len(double_diff)}

    return {
        "breakpoint_bins": int(breakpoint_bins), "breakpoint_s": breakpoint_bins * bin_width_s,
        "lag_min_bins": int(lag_min), "lag_max_bins": int(lag_max), "adjacency_lag_bins": int(adjacency_lag_bins),
        "by_statistic": by_statistic,
        "double_difference_floor_vs_persistent_component": double_diff_test,
        "double_difference_note": "[r_obs(breakpoint) - r_obs(L_max)] - [r_perm(breakpoint) - r_perm(L_max)]; "
                                   "significant and positive is what would license calling the late segment a "
                                   "persistent component rather than a bound on its decay.",
    }


def two_component_fit(lags_s: list[float], values: list[float]) -> dict:
    """Nonlinear least-squares fit of d(L) = A * exp(-L / tau) + C to a
    single profile, tau/A/C free. Reports parameter standard errors from
    the covariance matrix so a caller can judge identifiability from the
    relative standard error, not merely from whether the optimiser
    converged."""
    from scipy.optimize import curve_fit
    x = np.asarray(lags_s, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 4:
        return {"status": "not_enough_points", "n_points": len(x)}

    def model(lag, amplitude, tau, floor):
        return amplitude * np.exp(-lag / tau) + floor

    try:
        p0 = [float(y[0] - y[-1]) or 0.05, max(float(x[1] - x[0]), 0.1) * 3.0, float(y[-1])]
        bounds = ([-1.0, 1e-3, -1.0], [1.0, 100.0, 1.0])
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(3, np.nan)
        return {
            "status": "fitted", "n_points": len(x),
            "A": float(popt[0]), "tau_s": float(popt[1]), "C": float(popt[2]),
            "A_se": float(perr[0]), "tau_se": float(perr[1]), "C_se": float(perr[2]),
            "tau_relative_se": float(perr[1] / popt[1]) if popt[1] != 0 and np.isfinite(perr[1]) else None,
            "at_lower_bound": bool(abs(popt[1] - 1e-3) < 1e-4), "at_upper_bound": bool(abs(popt[1] - 100.0) < 1e-2),
        }
    except Exception as exc:
        return {"status": "fit_failed", "reason": str(exc)}


TWO_COMPONENT_LADDER_TAUS_S = [0.3, 0.6, 1.2, 2.5, 5.0]


def two_component_identifiability_ladder(
    n_trials: int, n_units: int, n_bins: int, bin_width_s: float, width_bins: int,
    rate_hz: float = 0.15, share_static: float = 0.5, share_slow: float = 0.5, seed: int = 300,
) -> dict:
    """The profile-form calibration ladder this module's timescale claims are
    read against, reported as a whole shape rather than a single summary
    number: for each of :data:`TWO_COMPONENT_LADDER_TAUS_S`, plant a
    population whose latent is a static-offset-plus-AR(1) mixture at that
    time constant (an A * exp(-L/tau) + C generative model in expectation,
    matching the two-component fit), run it through :func:`r_lag_profile`
    and the per-unit permutation null at the recording regime supplied by
    the caller, and fit
    :func:`two_component_fit` to the recovered d_perm(L). Identifiability
    is judged by whether recovered tau tracks planted tau (a significant
    positive Spearman correlation across the ladder) and by how many
    recovered values are pegged at the fit's own bounds -- NOT by whether
    any single fit converged, since a fit can converge to a meaningless
    value inside a wide bound."""
    ladder = []
    s = seed
    for tau_s in TWO_COMPONENT_LADDER_TAUS_S:
        counts = simulate_planted_population(
            n_trials, n_units, n_bins, bin_width_s, tau_s=tau_s, share_static=share_static, share_slow=share_slow,
            rate_hz=rate_hz, rng=np.random.default_rng(s))
        profile = r_lag_profile(counts, width_bins, n_splits=12, rng=np.random.default_rng(s + 1))
        if profile["status"] != "fitted":
            ladder.append({"planted_tau_s": tau_s, "status": "not_fitted"})
            s += 10
            continue
        null = per_unit_permutation_null_r_lag_profile(counts, width_bins, n_replicates=20, rng=np.random.default_rng(s + 2))
        common = sorted(set(profile["lags"]) & set(null["lags"]))
        lags_s = [lag * bin_width_s for lag in common]
        d_perm = [profile["lags"][lag]["r_median"] - null["lags"][lag]["r_null_median"] for lag in common]
        fit = two_component_fit(lags_s, d_perm)
        ladder.append({"planted_tau_s": tau_s, "recovered": fit, "lags_s": lags_s, "d_perm_profile": d_perm})
        s += 10

    fitted = [row for row in ladder if row.get("recovered", {}).get("status") == "fitted"]
    planted = [row["planted_tau_s"] for row in fitted]
    recovered = [row["recovered"]["tau_s"] for row in fitted]
    correlation = None
    identifiable = False
    if len(planted) >= 3 and len(set(recovered)) > 1:
        corr = spearman_permutation_test(np.array(planted), np.array(recovered), rng=np.random.default_rng(s + 3))
        correlation = {"rho": corr["rho"], "p_value": corr["p_value"]}
        identifiable = bool(corr["p_value"] <= 0.05 and corr["rho"] > 0.5)
    n_pegged = sum(1 for row in fitted if row["recovered"].get("at_lower_bound") or row["recovered"].get("at_upper_bound"))

    return {
        "planted_taus_s": TWO_COMPONENT_LADDER_TAUS_S, "ladder": ladder,
        "n_fitted": len(fitted), "n_pegged_at_fit_bound": n_pegged,
        "correlation_planted_vs_recovered_tau": correlation,
        "identifiable": identifiable,
        "criterion": "identifiable only if recovered tau has a significant (p<=0.05) positive Spearman "
                     "correlation (rho>0.5) with planted tau across the ladder; a fit converging for every "
                     "planted value is not sufficient, since the optimiser can converge inside a wide bound "
                     "to a value uninformative about the planted timescale.",
    }


def amplitude_ratio_report(series_per_session: list[dict[int, float]], lag_min_bins: int, breakpoint_bins: int) -> dict:
    """The fallback report when the two-component fit's tau is not
    identifiable (1.2(c)): the fast component described as a RATE (the
    early-segment slope, computed elsewhere) and an amplitude ratio --
    d(breakpoint) / d(lag_min), pooled across sessions -- rather than as a
    time constant no measurement here can support."""
    ratios = [s[breakpoint_bins] / s[lag_min_bins] for s in series_per_session
              if lag_min_bins in s and breakpoint_bins in s and s[lag_min_bins] != 0]
    early_vals = [s[lag_min_bins] for s in series_per_session if lag_min_bins in s]
    late_vals = [s[breakpoint_bins] for s in series_per_session if breakpoint_bins in s]
    if not ratios:
        return {"status": "not_computed"}
    return {
        "status": "tested", "n_sessions": len(ratios),
        "median_value_at_lag_min": float(np.median(early_vals)) if early_vals else None,
        "median_value_at_breakpoint": float(np.median(late_vals)) if late_vals else None,
        "median_amplitude_ratio": float(np.median(ratios)),
        "amplitude_ratio_iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
    }


def classify_lag_profile_segmented(
    profiles: list[dict],
    poisson_nulls: list[dict],
    permutation_nulls: list[dict],
    width_bins: int,
    bin_width_s: float,
    breakpoint_bins: int,
) -> dict:
    """The two-sided segmented decision rule: same five
    branches as :func:`classify_lag_profile` plus the off-list bucket, now
    decided on the SEGMENTED slopes at a declared breakpoint rather than a
    single whole-range slope, and with a two-sided condition on every
    branch per this module's standing rule that no branch may be defined by
    the absence of one sign -- a significantly POSITIVE early or late
    segment slope routes to the off-list bucket with its number rather than
    being absorbed into a flat branch.

    Priority: rotation first; then whether existence (contrast A) clears at
    no lag at all; then whether either segment is significantly positive
    (off-list, since no branch below describes a state whose cross-unit
    correlation RISES with separation); then whether the early segment
    (excluding the adjacency lag) is significantly negative -- graded_decay
    if the late segment is also significantly negative (the decay
    continues past the breakpoint), fast_component_plus_flat_floor if not
    (the decay is confined to the early segment and the late segment is a
    floor); then whether the late segment alone is significantly negative
    (still graded_decay -- a genuine multi-lag decay, merely positioned
    late rather than early); then whether existence clears a majority of
    lags with no significant slope anywhere (flat_cross_unit_state);
    anything else is the off-list bucket."""
    contrast_a, rotation, n_lags_tested, n_lags_clearing, a_clears_any, a_clears_majority = \
        _lag_existence_and_rotation(profiles, permutation_nulls)
    adjacency_lag = width_bins
    segmented = segmented_shape_contrasts(profiles, poisson_nulls, permutation_nulls, bin_width_s,
                                           breakpoint_bins, adjacency_lag)

    d_perm = segmented.get("by_statistic", {}).get("d_perm", {})
    early_excl = d_perm.get("early_segment_excluding_adjacency", {}).get("test", {})
    early_incl = d_perm.get("early_segment_including_adjacency", {}).get("test", {})
    late = d_perm.get("late_segment", {}).get("test", {})

    early_significant_negative = bool(early_excl.get("significant_negative", False))
    early_significant_positive = bool(early_incl.get("significant_positive", False) or early_excl.get("significant_positive", False))
    late_significant_negative = bool(late.get("significant_negative", False))
    late_significant_positive = bool(late.get("significant_positive", False))

    if rotation["detected"]:
        branch = "rotation"
    elif not a_clears_any:
        branch = "no_state_above_the_within_unit_floor"
    elif early_significant_positive or late_significant_positive:
        branch = "unattributed_off_branch_list"
    elif early_significant_negative:
        branch = "graded_decay" if late_significant_negative else "fast_component_plus_flat_floor"
    elif late_significant_negative:
        branch = "graded_decay"
    elif a_clears_majority:
        branch = "flat_cross_unit_state"
    else:
        branch = "unattributed_off_branch_list"

    return {
        "reachability": lag_reachability_note(),
        "width_bins": int(width_bins), "bin_width_s": bin_width_s, "adjacency_lag_bins": int(adjacency_lag),
        "n_lags_tested": n_lags_tested, "n_lags_clearing_fdr": n_lags_clearing,
        "contrast_a_existence_by_lag": contrast_a,
        "rotation_check": rotation,
        "segmented_contrasts": segmented,
        "early_segment_significant_negative": early_significant_negative,
        "early_segment_significant_positive": early_significant_positive,
        "late_segment_significant_negative": late_significant_negative,
        "late_segment_significant_positive": late_significant_positive,
        "branch": branch,
        "branch_priority_note": "Evaluated in this order: rotation first; then whether existence clears at "
                                 "NO lag at all; then whether either segment (early excluding the adjacency "
                                 "lag, or late) is significantly POSITIVE -- off-list, per the rule that no "
                                 "branch may be defined by the absence of one sign; then whether the early "
                                 "segment is significantly negative ('graded_decay' if the late segment is "
                                 "too, else 'fast_component_plus_flat_floor'); then whether the late segment "
                                 "alone is significantly negative ('graded_decay'); then whether existence "
                                 "clears a majority of lags with no significant slope anywhere "
                                 "('flat_cross_unit_state'); anything else is the off-list bucket.",
    }


def simulate_planted_population(
    n_trials: int,
    n_units: int,
    n_bins: int,
    bin_width_s: float,
    tau_s: float | None,
    share_static: float,
    share_slow: float,
    rate_hz: float = 1.0,
    rotate: bool = False,
    rotation_radians: float = 0.0,
    loading_scale: float = 0.35,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Poisson population generated from a known scalar latent, for the
    calibration ladder and for tests. ``tau_s=None`` means the latent has no
    decaying component at all (a pure static offset); otherwise the latent's
    slow component is an AR(1) process with that time constant. ``rotate``
    replaces the additive offset-plus-AR(1) latent with a two-dimensional
    rotation of the offset through ``rotation_radians`` across the epoch."""
    if rng is None:
        rng = np.random.default_rng(0)
    rho = float(np.exp(-bin_width_s / tau_s)) if tau_s else 0.0
    baseline = rate_hz * (1.0 + 0.3 * np.sin(np.linspace(0.0, np.pi, n_bins)))[None, :]
    offset = rng.normal(size=(n_trials, 1)) * np.sqrt(share_static)
    ar1 = np.zeros((n_trials, n_bins))
    ar1[:, 0] = rng.normal(size=n_trials)
    for k in range(1, n_bins):
        ar1[:, k] = rho * ar1[:, k - 1] + rng.normal(size=n_trials) * np.sqrt(max(1.0 - rho ** 2, 0.0))
    latent = offset + ar1 * np.sqrt(share_slow)
    if rotate:
        angle = np.linspace(0.0, rotation_radians, n_bins)[None, :]
        latent = offset * np.cos(angle) + rng.normal(size=(n_trials, 1)) * np.sin(angle)
    loading = rng.normal(size=n_units) * loading_scale
    rate = np.clip(baseline[None] * np.exp(latent[:, None, :] * loading[None, :, None]), 1e-6, None)
    return rng.poisson(rate).astype(float)


def simulate_no_shared_state_population(
    n_trials: int,
    n_units: int,
    n_bins: int,
    bin_width_s: float,
    tau_s: float,
    rate_hz: float = 1.0,
    loading_scale: float = 0.35,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Poisson population with NO cross-unit shared latent: each unit's rate
    is modulated by its own independent per-trial AR(1) process, drawn fresh
    per unit per trial. Unit-to-unit correlation is exactly zero by
    construction, but each unit's own value in one window still covaries
    with its value in a nearby window of the SAME trial, because the AR(1)
    process autocorrelates in time -- the within-unit diagonal term a
    shared-state permutation null must be sensitive to, or it cannot tell a
    real fast-decaying shared state from within-unit spike-train noise."""
    if rng is None:
        rng = np.random.default_rng(0)
    rho = float(np.exp(-bin_width_s / tau_s))
    baseline = rate_hz * (1.0 + 0.3 * np.sin(np.linspace(0.0, np.pi, n_bins)))[None, None, :]
    ar1 = np.zeros((n_trials, n_units, n_bins))
    ar1[:, :, 0] = rng.normal(size=(n_trials, n_units))
    for k in range(1, n_bins):
        ar1[:, :, k] = rho * ar1[:, :, k - 1] + rng.normal(size=(n_trials, n_units)) * np.sqrt(max(1.0 - rho ** 2, 0.0))
    rate = np.clip(baseline * np.exp(ar1 * loading_scale), 1e-6, None)
    return rng.poisson(rate).astype(float)


def _permute_counts_independently_per_unit(counts: NDArray, rng: np.random.Generator) -> NDArray:
    """Permute the trial axis independently for each unit: one permutation
    per unit, applied to all of that unit's bins at once. Every unit's own
    rate profile, count distribution, burstiness and refractory structure is
    untouched -- no unit's time series is altered at all, only which trial
    it is filed under changes -- so this destroys cross-unit (shared-state)
    covariance while preserving each unit's own within-unit, across-window
    covariance exactly."""
    n_trials = counts.shape[0]
    n_units = counts.shape[1]
    permuted = np.empty_like(counts)
    for u in range(n_units):
        perm = rng.permutation(n_trials)
        permuted[:, u, :] = counts[perm, u, :]
    return permuted


def per_unit_permutation_null_r_gap_profile(
    counts: NDArray,
    n_replicates: int = N_NULL_REPLICATES_DEFAULT,
    n_splits_per_replicate: int = NULL_SPLITS_PER_REPLICATE_DEFAULT,
    rng: np.random.Generator | None = None,
    n_windows: int = N_WINDOWS_DEFAULT,
) -> dict:
    """Second per-session negative control, alongside (not instead of) the
    Poisson null: destroys the shared trial-specific state by permuting
    trial labels independently per unit, while preserving each unit's own
    within-trial temporal autocorrelation. The Poisson null is white by
    construction and cannot be sensitive to bursting, refractoriness or rate
    wander within a unit's own spike train; this null can. See
    :func:`_permute_counts_independently_per_unit` for what it does and does
    not preserve."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)

    replicate_gaps: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(n_replicates):
        permuted = _permute_counts_independently_per_unit(counts, rng)
        result = r_gap_profile(permuted, n_splits=n_splits_per_replicate, rng=rng, n_windows=n_windows)
        if result["status"] != "fitted":
            continue
        n_fitted += 1
        for gap, g in result["gaps"].items():
            replicate_gaps.setdefault(gap, []).append(g["r_median"])

    gaps = {
        gap: {"r_null_median": float(np.median(values)), "r_null_p95": float(np.percentile(values, 95))}
        for gap, values in sorted(replicate_gaps.items())
    }
    return {"n_replicates_requested": int(n_replicates), "n_replicates_fitted": int(n_fitted), "gaps": gaps}


def poisson_null_r_lag_profile(
    counts: NDArray,
    width_bins: int,
    n_replicates: int = N_NULL_REPLICATES_DEFAULT,
    n_splits_per_replicate: int = NULL_SPLITS_PER_REPLICATE_DEFAULT,
    rng: np.random.Generator | None = None,
) -> dict:
    """The fixed-width analogue of :func:`poisson_null_r_gap_profile`: a
    rate-matched population with an identical time course on every
    simulated trial -- no trial-specific structure exists in it by
    construction -- run through :func:`r_lag_profile` at the same width."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    psth = np.clip(counts.mean(axis=0), 1e-6, None)  # (units, bins)

    replicate_lags: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(n_replicates):
        simulated = rng.poisson(lam=np.broadcast_to(psth, counts.shape)).astype(float)
        result = r_lag_profile(simulated, width_bins, n_splits=n_splits_per_replicate, rng=rng)
        if result["status"] != "fitted":
            continue
        n_fitted += 1
        for lag, g in result["lags"].items():
            replicate_lags.setdefault(lag, []).append(g["r_median"])

    lags = {
        lag: {"r_null_median": float(np.median(values)), "r_null_p95": float(np.percentile(values, 95))}
        for lag, values in sorted(replicate_lags.items())
    }
    return {"n_replicates_requested": int(n_replicates), "n_replicates_fitted": int(n_fitted), "lags": lags}


def per_unit_permutation_null_r_lag_profile(
    counts: NDArray,
    width_bins: int,
    n_replicates: int = N_NULL_REPLICATES_DEFAULT,
    n_splits_per_replicate: int = NULL_SPLITS_PER_REPLICATE_DEFAULT,
    rng: np.random.Generator | None = None,
    projection_fn=_leading_latent_projection,
) -> dict:
    """The fixed-width analogue of :func:`per_unit_permutation_null_r_gap_
    profile`: destroys cross-unit shared-state covariance while preserving
    each unit's own within-trial temporal autocorrelation exactly, run
    through :func:`r_lag_profile` at the same width. This is the null
    d_perm is computed against at every lag. ``projection_fn`` is passed
    through to :func:`r_lag_profile` unchanged -- the null must be scored
    under the SAME basis-fitting method as the observed profile it is
    subtracted from, or d_perm would be comparing two different noise
    models rather than one model's signal against its own null."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)

    replicate_lags: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(n_replicates):
        permuted = _permute_counts_independently_per_unit(counts, rng)
        result = r_lag_profile(permuted, width_bins, n_splits=n_splits_per_replicate, rng=rng, projection_fn=projection_fn)
        if result["status"] != "fitted":
            continue
        n_fitted += 1
        for lag, g in result["lags"].items():
            replicate_lags.setdefault(lag, []).append(g["r_median"])

    lags = {
        lag: {"r_null_median": float(np.median(values)), "r_null_p95": float(np.percentile(values, 95))}
        for lag, values in sorted(replicate_lags.items())
    }
    return {"n_replicates_requested": int(n_replicates), "n_replicates_fitted": int(n_fitted), "lags": lags}


def calibration_ladder(
    n_trials: int,
    n_units: int,
    n_bins: int,
    bin_width_s: float,
    rate_hz: float = 1.0,
    n_splits: int = N_SPLITS_DEFAULT,
    n_windows: int = N_WINDOWS_DEFAULT,
    seed: int = 0,
) -> dict:
    """Recovered r(gap) profile for every regime in CALIBRATION_REGIMES plus
    the rotation regime, at a given (trials, units, bins, rate) recording
    regime. This is part of the deliverable, not a test: every timescale
    statement about real data is read against this table."""
    ladder = {}
    for name, params in CALIBRATION_REGIMES + [ROTATION_REGIME]:
        counts = simulate_planted_population(
            n_trials, n_units, n_bins, bin_width_s, rate_hz=rate_hz,
            rng=np.random.default_rng(seed), **params,
        )
        profile = r_gap_profile(counts, n_splits=n_splits, rng=np.random.default_rng(seed + 1), n_windows=n_windows)
        ladder[name] = profile
    return ladder


def ols_with_two_predictors(x1: NDArray, x2: NDArray, y: NDArray) -> dict | None:
    """Ordinary least squares of ``y`` on an intercept, ``x1`` and ``x2``
    jointly. Returns None if fewer than 5 points or either predictor carries
    no variance left after accounting for the other (a rank-deficient
    design, e.g. every pair at a given lag sharing one start position)."""
    x1, x2, y = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float), np.asarray(y, dtype=float)
    if len(y) < 5:
        return None
    design = np.stack([np.ones_like(x1), x1, x2], axis=1)
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 3:
        return None
    return {"intercept": float(coef[0]), "coef_x1": float(coef[1]), "coef_x2": float(coef[2]), "n_points": int(len(y))}


def rank1_gain_share(window_means: NDArray, n_shuffles: int = 200, rng: np.random.Generator | None = None) -> dict:
    """The fraction of a (trials x windows) matrix's total squared
    Frobenius norm carried by its first singular value, against a
    matched-noise reference built by permuting each TRIAL'S OWN row
    independently -- preserving that trial's own marginal (its own set of
    window values, hence its own variance) while destroying which window
    carries which value, and therefore destroying any consistent across-trial
    temporal alignment. A rank-1 gain on a shared temporal profile does not
    survive this shuffle (each row's specific ordering is exactly what ties
    it to every other row's), so a share well above this reference is
    evidence of real cross-trial structure, not merely of trial-to-trial
    variance heterogeneity."""
    if rng is None:
        rng = np.random.default_rng(0)
    window_means = np.asarray(window_means, dtype=float)

    def _share(matrix: NDArray) -> float:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        total = float(np.sum(singular_values ** 2))
        return float(singular_values[0] ** 2 / total) if total > 0 else 0.0

    observed_share = _share(window_means)
    null_shares = np.array([
        _share(np.array([rng.permutation(row) for row in window_means])) for _ in range(n_shuffles)
    ])
    p_value = permutation_pvalue(null_shares >= observed_share)
    return {
        "observed_share": observed_share, "null_share_median": float(np.median(null_shares)),
        "null_share_p95": float(np.percentile(null_shares, 95)), "p_value": p_value,
        "significantly_above_reference": bool(p_value <= 0.05), "n_shuffles": int(n_shuffles),
    }


def rank1_gain_and_residual(window_means: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """The first left singular vector scaled by its singular value (the
    per-trial 'gain', one scalar per trial), the first right singular vector
    (the shared temporal profile h(t) the gain multiplies -- under exact
    rank 1, z_i(t) = g_i * h(t)), and the residual matrix after removing that
    rank-1 component.

    h(t) matters beyond the variance share it explains: under exact rank 1
    the correlation between windows s and s+L is sign(h(s) * h(s+L)) with
    magnitude 1, independent of lag -- a flat profile is what a pure
    per-trial gain predicts. But if h(t) changes sign inside the epoch,
    pairs on the same side of the crossing give +1 and pairs straddling it
    give -1; since longer lags straddle a mid-epoch crossing more often than
    short ones, a sign-changing h(t) manufactures an apparent monotone decay
    with lag out of a structure that has no dynamics in it at all. Per-window
    centring does not prevent this -- it constrains the mean over trials, not
    the sign of h over time. Callers that report a lag-dependent slope must
    also report h(t)'s sign-crossing structure alongside it."""
    window_means = np.asarray(window_means, dtype=float)
    u, s, vt = np.linalg.svd(window_means, full_matrices=False)
    gain = u[:, 0] * s[0]
    h_profile = vt[0, :]
    residual = window_means - np.outer(u[:, 0] * s[0], h_profile)
    return gain, h_profile, residual


def temporal_profile_sign_crossings(h_profile: NDArray) -> dict:
    """Number and location of sign changes of a temporal profile h(t) across
    windows, plus the sign convention used (the leading singular vector's
    overall sign is arbitrary, so 'positive' and 'negative' here are only
    meaningful relative to each other within one profile, not comparable
    across sessions)."""
    h = np.asarray(h_profile, dtype=float)
    signs = np.sign(h)
    crossings = [int(i) for i in range(1, len(signs)) if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]]
    return {
        "n_windows": int(len(h)), "n_sign_crossings": len(crossings), "crossing_window_indices": crossings,
        "h_profile": [float(v) for v in h],
    }


def residual_pair_correlations(residual: NDArray, width_bins: int) -> dict[int, float]:
    """r(lag) on the RESIDUAL (trials x windows) matrix after the rank-1
    gain is removed, averaged over start positions at each lag -- the same
    pairing :func:`_r_lag_one_split` uses, applied to a matrix that already
    has a leading latent projected and its rank-1 component subtracted
    rather than to raw counts."""
    n_bins = residual.shape[1]
    per_lag: dict[int, list[float]] = {}
    for lag in range(width_bins, n_bins - width_bins + 1):
        for s, t in _lag_pairs(n_bins, width_bins, lag):
            a, b = residual[:, s], residual[:, t]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            per_lag.setdefault(lag, []).append(float(np.corrcoef(a, b)[0, 1]))
    return {lag: float(np.mean(values)) for lag, values in per_lag.items()}


def breakpoint_bootstrap_ci(
    series_per_session: list[dict[int, float]], bin_width_s: float,
    n_boot: int = 500, rng: np.random.Generator | None = None,
) -> dict:
    """Session-resampled 95% interval on the pooled-profile breakpoint
    (:func:`fit_breakpoint_from_pooled_profile`), in seconds: resample
    sessions with replacement, refit the breakpoint on each resample, and
    take the 2.5/97.5 percentiles of the recovered breakpoint. A resample
    that yields fewer than 5 lags, or a run with fewer than 20 successful
    fits out of ``n_boot``, is reported as ``not_estimable`` rather than a
    percentile interval built on too few draws to mean anything."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(series_per_session)
    point = fit_breakpoint_from_pooled_profile(series_per_session, bin_width_s) if n >= 4 else None
    if point is None:
        return {"status": "not_estimable", "n_sessions": n, "reason": "no point-estimate breakpoint fit on the full cohort"}
    boot_s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = [series_per_session[i] for i in idx]
        fit = fit_breakpoint_from_pooled_profile(resample, bin_width_s)
        if fit is not None:
            boot_s.append(fit["breakpoint_s"])
    if len(boot_s) < 20:
        return {"status": "not_estimable", "n_sessions": n, "n_boot_fitted": len(boot_s),
                "reason": "fewer than 20 of the n_boot resamples produced a fittable breakpoint"}
    return {
        "status": "tested", "n_sessions": n, "n_boot_requested": int(n_boot), "n_boot_fitted": len(boot_s),
        "point_estimate_s": point["breakpoint_s"],
        "ci_lower_s": float(np.percentile(boot_s, 2.5)), "ci_upper_s": float(np.percentile(boot_s, 97.5)),
    }


def geometry_vs_clock_verdict(ci_a: dict, window_a_s: float, ci_b: dict, window_b_s: float) -> dict:
    """Three-branch verdict comparing two cohorts' breakpoint bootstrap
    intervals (:func:`breakpoint_bootstrap_ci`) that differ in window length:
    overlap the two intervals in SECONDS (the fixed-timescale/clock
    prediction -- a single seconds-value consistent with both cohorts) and
    separately overlap them as a FRACTION of each cohort's own window (the
    geometry-scaled prediction -- a single fraction consistent with both).
    Exactly one of the two overlaps holding is what adjudicates; both or
    neither holding means the intervals are too wide to separate the two
    hypotheses. This is symmetric in the two hypotheses (seconds-space for
    the clock, fraction-space for geometry) rather than requiring a single
    externally chosen prediction value to test against."""
    if ci_a.get("status") != "tested" or ci_b.get("status") != "tested":
        return {"verdict": "underpowered_to_adjudicate",
                "reason": "bootstrap breakpoint interval not estimable in at least one cohort",
                "seconds_overlap": None, "fraction_overlap": None}
    seconds_overlap = bool(ci_a["ci_lower_s"] <= ci_b["ci_upper_s"] and ci_b["ci_lower_s"] <= ci_a["ci_upper_s"])
    frac_a = [ci_a["ci_lower_s"] / window_a_s, ci_a["ci_upper_s"] / window_a_s]
    frac_b = [ci_b["ci_lower_s"] / window_b_s, ci_b["ci_upper_s"] / window_b_s]
    fraction_overlap = bool(frac_a[0] <= frac_b[1] and frac_b[0] <= frac_a[1])
    if seconds_overlap and not fraction_overlap:
        verdict = "stayed_on_the_clock"
    elif fraction_overlap and not seconds_overlap:
        verdict = "moved_with_the_geometry"
    else:
        verdict = "underpowered_to_adjudicate"
    return {
        "verdict": verdict, "seconds_overlap": seconds_overlap, "fraction_overlap": fraction_overlap,
        "ci_seconds_cohort_a": [ci_a["ci_lower_s"], ci_a["ci_upper_s"]],
        "ci_seconds_cohort_b": [ci_b["ci_lower_s"], ci_b["ci_upper_s"]],
        "ci_fraction_cohort_a": frac_a, "ci_fraction_cohort_b": frac_b,
        "verdict_note": "stayed_on_the_clock: the seconds intervals overlap (a single seconds-value fits both "
                         "cohorts) while the fraction-of-window intervals do not (no single fraction fits both), "
                         "so equality-in-seconds is the reading the data can support. moved_with_the_geometry: "
                         "the reverse. underpowered_to_adjudicate: both overlap (either hypothesis fits) or "
                         "neither does (neither hypothesis fits) -- the bootstrap intervals are too wide, or too "
                         "displaced, to separate the two predictions.",
    }


def permutation_null_validation(seed: int = 200) -> dict:
    """Validates the per-unit permutation null on synthetic data before it is
    used to adjudicate anything: (a) on a planted SHARED latent riding on a
    population where each unit's own loading is weak relative to the
    cross-unit consensus PCA relies on (many units, small per-unit loading),
    the null must destroy most of the signal -- a single unit's own
    autocorrelation cannot substitute for the cross-unit averaging that
    detects a real shared state, so its own diagonal contribution is a small
    fraction of the un-permuted r; (b) on a planted population with NO
    shared state but strong within-unit temporal autocorrelation (few units,
    each with a large, fast-decaying private AR(1) loading), the null must
    REPRODUCE the direct fit's shortest-lag inflation almost fully, since
    that inflation never depended on cross-unit structure to begin with. A
    null that fails (b) cannot be used to clear the step."""
    shared = simulate_planted_population(
        n_trials=80, n_units=80, n_bins=24, bin_width_s=0.1, tau_s=None, share_static=1.0, share_slow=0.0,
        rate_hz=0.3, loading_scale=0.15, rng=np.random.default_rng(seed))
    shared_direct = r_gap_profile(shared, n_splits=20, rng=np.random.default_rng(seed + 1))
    shared_null = per_unit_permutation_null_r_gap_profile(shared, n_replicates=20, rng=np.random.default_rng(seed + 2))

    no_shared = simulate_no_shared_state_population(
        n_trials=80, n_units=10, n_bins=24, bin_width_s=0.1, tau_s=0.3, rate_hz=5.0, loading_scale=1.0,
        rng=np.random.default_rng(seed + 3))
    no_shared_direct = r_gap_profile(no_shared, n_splits=20, rng=np.random.default_rng(seed + 4))
    no_shared_null = per_unit_permutation_null_r_gap_profile(no_shared, n_replicates=20, rng=np.random.default_rng(seed + 5))

    slow_wander = simulate_no_shared_state_population(
        n_trials=80, n_units=10, n_bins=40, bin_width_s=0.1, tau_s=2.0, rate_hz=5.0, loading_scale=1.0,
        rng=np.random.default_rng(seed + 6))
    slow_direct = r_lag_profile(slow_wander, width_bins=3, n_splits=20, rng=np.random.default_rng(seed + 7))
    slow_null = per_unit_permutation_null_r_lag_profile(slow_wander, width_bins=3, n_replicates=20, rng=np.random.default_rng(seed + 8))
    slow_lags_direct = sorted(slow_direct["lags"]) if slow_direct["status"] == "fitted" else []
    check_c_passes = False
    slow_short = slow_mid = slow_long = slow_short_null = slow_mid_null = slow_long_null = None
    if slow_lags_direct and len(slow_lags_direct) >= 3:
        short_lag, mid_lag, long_lag = slow_lags_direct[0], slow_lags_direct[len(slow_lags_direct) // 2], slow_lags_direct[-1]
        slow_short = slow_direct["lags"][short_lag]["r_median"]
        slow_mid = slow_direct["lags"][mid_lag]["r_median"]
        slow_long = slow_direct["lags"][long_lag]["r_median"]
        slow_short_null = slow_null["lags"].get(short_lag, {}).get("r_null_median")
        slow_mid_null = slow_null["lags"].get(mid_lag, {}).get("r_null_median")
        slow_long_null = slow_null["lags"].get(long_lag, {}).get("r_null_median")
        if None not in (slow_short_null, slow_mid_null, slow_long_null):
            check_c_passes = bool(
                slow_short > slow_mid > slow_long
                and slow_short_null > slow_mid_null > slow_long_null
            )

    shared_r1_direct = shared_direct["gaps"].get(1, {}).get("r_median") if shared_direct["status"] == "fitted" else None
    shared_r1_null = shared_null["gaps"].get(1, {}).get("r_null_median")
    check_a_passes = bool(
        shared_r1_direct is not None and shared_r1_direct > 0.3
        and shared_r1_null is not None and shared_r1_null < 0.4 * shared_r1_direct
    )

    gap1_direct = no_shared_direct["gaps"].get(1, {}).get("r_median") if no_shared_direct["status"] == "fitted" else None
    gap3_direct = no_shared_direct["gaps"].get(3, {}).get("r_median") if no_shared_direct["status"] == "fitted" else None
    gap1_null = no_shared_null["gaps"].get(1, {}).get("r_null_median")
    gap3_null = no_shared_null["gaps"].get(3, {}).get("r_null_median")
    check_b_passes = bool(
        gap1_direct is not None and gap3_direct is not None and gap1_direct > gap3_direct
        and gap1_null is not None and gap3_null is not None and gap1_null > gap3_null
        and gap1_null > 0.5 * gap1_direct
    )

    usable = bool(check_a_passes and check_b_passes and check_c_passes)
    return {
        "check_a_destroys_planted_shared_signal": {
            "regime": "80 units, weak per-unit loading (0.15), a genuinely shared static offset",
            "direct_r_gap1": shared_r1_direct, "null_r_gap1_median": shared_r1_null, "passes": check_a_passes,
            "criterion": "direct r(1) > 0.3 (a real signal exists to destroy), and null r(1) < 0.4 * direct r(1)",
        },
        "check_b_reproduces_shortest_lag_inflation_with_no_shared_state": {
            "regime": "10 units, strong independent per-unit loading (1.0), tau=0.3s, no shared latent",
            "direct_r_gap1": gap1_direct, "direct_r_gap3": gap3_direct,
            "null_r_gap1_median": gap1_null, "null_r_gap3_median": gap3_null, "passes": check_b_passes,
            "criterion": "direct r(1) > direct r(3) (a real shortest-lag inflation with no shared state), "
                         "null r(1) > null r(3) (the null reproduces the shape), and null r(1) > 0.5 * direct "
                         "r(1) (the null preserves most of the magnitude, not just the sign)",
        },
        "check_c_reproduces_decline_across_the_whole_lag_range": {
            "regime": "10 units, strong independent per-unit loading (1.0), tau=2.0s rate wander, no shared "
                      "latent, width=3 bins, n_bins=40 -- a slower nuisance than check (b)'s, chosen because "
                      "the real data's own nuisance floor turns out to decay over seconds rather than tens "
                      "of milliseconds (see the deciding contrast's null-profile field)",
            "short_lag_bins": slow_lags_direct[0] if slow_lags_direct else None,
            "mid_lag_bins": slow_lags_direct[len(slow_lags_direct) // 2] if len(slow_lags_direct) >= 3 else None,
            "long_lag_bins": slow_lags_direct[-1] if slow_lags_direct else None,
            "direct_r_short": slow_short, "direct_r_mid": slow_mid, "direct_r_long": slow_long,
            "null_r_short": slow_short_null, "null_r_mid": slow_mid_null, "null_r_long": slow_long_null,
            "passes": check_c_passes,
            "criterion": "direct r declines monotonically short > mid > long lag, AND the null reproduces "
                         "that same three-point decline -- not only a shortest-lag inflation, since a nuisance "
                         "that decays over seconds must be visible in the null across the whole range or it "
                         "cannot be used to subtract that range's floor",
        },
        "usable": usable,
        "verdict": (
            "the per-unit permutation null is validated and usable to adjudicate the shortest-lag step"
            if usable else
            "the per-unit permutation null FAILED validation and cannot be used to clear or convict the "
            "shortest-lag step -- this is a completed deliverable, not a failure, and the round stops here"
        ),
    }
