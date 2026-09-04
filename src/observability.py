"""observability.py -- how much of a fitted latent's held-out variance is
white per-trial noise rather than slow, temporally structured signal.

The estimand: fit the leading PCA direction of a binned population matrix on
training trials, project held-out trials onto it, and decompose the
across-trial autocovariance of that held-out latent time series into a slow
component decaying geometrically across lags (``c(k) = s2_slow * rho**k`` for
``k >= 1``) and a white nugget sitting only at lag zero
(``c(0) = s2_slow + s2_nugget``). The nugget fraction is the share of lag-zero
variance the slow component cannot account for.

Every reported cross-validated statistic is computed by fitting the basis on
one half of the trials and scoring the autocovariance decomposition on the
other half only; the in-sample counterpart (basis fit and scored on the same
trials) is reported alongside it because in-sample estimation is known to
understate the nugget.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.typing import NDArray

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from spike_pipeline import FrozenPSTHTransform  # noqa: E402

MAX_LAG_DEFAULT = 8
N_SPLITS_DEFAULT = 8
MIN_POSITIVE_LAGS = 3


def _leading_latent_projection(psth_fit: NDArray, psth_apply: NDArray) -> NDArray:
    """Fit the leading PCA direction on ``psth_fit`` (trials, units, bins),
    already unit-standardized, and project ``psth_apply`` onto it.

    Returns the projected (trials, bins) latent time series.
    """
    n_units = psth_fit.shape[1]
    fit_flat = psth_fit.transpose(0, 2, 1).reshape(-1, n_units)
    fit_flat = fit_flat - fit_flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(fit_flat, full_matrices=False)
    direction = vt[0]
    apply_flat = psth_apply.transpose(0, 2, 1).reshape(-1, n_units)
    projected = apply_flat @ direction
    return projected.reshape(psth_apply.shape[0], psth_apply.shape[2])


def _leading_latent_projection_via_factor_analysis(psth_fit: NDArray, psth_apply: NDArray) -> NDArray | None:
    """The factor-analysis analogue of :func:`_leading_latent_projection`:
    same call contract (fit on ``psth_fit``, project ``psth_apply``, return
    a (trials, bins) leading-latent time series), but the fitted basis is
    the single leading factor of a linear-Gaussian factor model with an
    explicit per-unit diagonal observation-noise variance (``sklearn``'s EM
    solver), rather than a PCA direction that has no observation-noise term
    at all. Returns None -- instead of raising -- if the EM fit does not
    converge or degenerates (a unit count too small for one factor, or a
    noise variance that collapses to a degenerate near-zero or non-finite
    value for every unit), so a caller iterating over cross-validation
    splits can skip that split exactly as it already skips a split with no
    usable window means.
    """
    from sklearn.decomposition import FactorAnalysis
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    n_units = psth_fit.shape[1]
    if n_units < 2:
        return None
    fit_flat = psth_fit.transpose(0, 2, 1).reshape(-1, n_units)
    apply_flat = psth_apply.transpose(0, 2, 1).reshape(-1, n_units)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = FactorAnalysis(n_components=1, random_state=0, max_iter=1000)
            model.fit(fit_flat)
    except (ConvergenceWarning, ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(model.noise_variance_)) or not np.all(np.isfinite(model.components_)):
        return None
    projected = model.transform(apply_flat)[:, 0]
    if not np.all(np.isfinite(projected)):
        return None
    return projected.reshape(psth_apply.shape[0], psth_apply.shape[2])


def _lagged_autocovariance(latent: NDArray, max_lag: int) -> NDArray:
    """c(k) for k = 0..max_lag: across-trial mean of the lagged product of
    each trial's own-temporal-mean-removed latent time series."""
    demeaned = latent - latent.mean(axis=1, keepdims=True)
    values = np.empty(max_lag + 1)
    values[0] = float(np.mean(demeaned * demeaned))
    for lag in range(1, max_lag + 1):
        values[lag] = float(np.mean(demeaned[:, :-lag] * demeaned[:, lag:]))
    return values


def _decompose_autocovariance(c: NDArray, bin_width_s: float) -> dict:
    """Regress log c(k) on k over positive k>=1 lags; extrapolate to k=0 for
    the slow variance, and read the decay rate off the slope."""
    lags = np.arange(1, len(c))
    positive = c[1:] > 0
    n_positive = int(positive.sum())
    if n_positive < MIN_POSITIVE_LAGS:
        return {
            "status": "fewer_than_three_positive_lags",
            "n_lags_used": n_positive,
            "nugget_fraction": None,
            "slow_timescale_s": None,
            "rho": None,
            "s2_slow": None,
            "c0": float(c[0]),
        }
    k = lags[positive].astype(float)
    log_c = np.log(c[1:][positive])
    slope, intercept = np.polyfit(k, log_c, 1)
    rho = float(np.exp(slope))
    s2_slow = float(np.exp(intercept))
    c0 = float(c[0])
    nugget_fraction = float(max(c0 - s2_slow, 0.0) / c0) if c0 > 0 else None
    slow_timescale_s = float(-bin_width_s / np.log(rho)) if 0.0 < rho < 1.0 else None
    return {
        "status": "fitted",
        "n_lags_used": n_positive,
        "nugget_fraction": nugget_fraction,
        "slow_timescale_s": slow_timescale_s,
        "rho": rho,
        "s2_slow": s2_slow,
        "c0": c0,
    }


def decompose_held_out_latent(latent: NDArray, bin_width_s: float, max_lag: int = MAX_LAG_DEFAULT) -> dict:
    """Nugget-fraction decomposition of an already-fit, already-projected
    held-out latent time series ``(trials, bins)``.

    Shared by :func:`nugget_fraction` (whose latent comes from a PCA
    direction fit on training trials) and by any other latent-fitting method
    that produces a comparable held-out ``(trials, bins)`` series -- the
    autocovariance decomposition itself does not depend on how the latent
    direction was obtained.
    """
    n_bins = latent.shape[1]
    lag_cap = max(1, min(n_bins - 1, max_lag))
    return _decompose_autocovariance(_lagged_autocovariance(latent, lag_cap), bin_width_s)


def median_nugget_from_held_out_latent(
    latent: NDArray,
    bin_width_s: float,
    n_splits: int = N_SPLITS_DEFAULT,
    rng: np.random.Generator | None = None,
    max_lag: int = MAX_LAG_DEFAULT,
) -> dict:
    """Stable median nugget fraction from an already-fit, already-projected
    held-out latent, by repeatedly decomposing random half-subsets of its
    trials rather than by refitting the latent basis.

    For a latent-fitting method expensive enough that refitting it on many
    random training splits (:func:`nugget_fraction`'s protocol) is not
    affordable, this recovers the same "at least three subsamples must
    decompose" stability guarantee from resampling the one held-out
    evaluation set instead: a single sample autocovariance is noisy enough
    that individual half-subsets often land on fewer than three positive
    lags purely by chance, and repeated resampling is what makes the median
    reportable at all (see :func:`nugget_fraction`, which resamples via
    fresh training splits rather than held-out subsets, but the same
    "several attempts, take the median of the ones that decomposed" logic
    applies).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_trials = latent.shape[0]
    per_split = []
    for split_index in range(n_splits):
        subset = rng.permutation(n_trials)[: max(2, n_trials // 2)]
        result = decompose_held_out_latent(latent[subset], bin_width_s, max_lag=max_lag)
        result["split"] = split_index
        per_split.append(result)
    fitted = [r for r in per_split if r["status"] == "fitted"]
    n_fitted = len(fitted)
    status = "fitted" if n_fitted >= MIN_POSITIVE_LAGS else "fewer_than_three_positive_lags"
    if fitted:
        nuggets = np.array([r["nugget_fraction"] for r in fitted])
        median_nugget = float(np.median(nuggets))
        iqr = [float(np.percentile(nuggets, 25)), float(np.percentile(nuggets, 75))]
    else:
        median_nugget = None
        iqr = None
    return {
        "status": status, "n_splits_requested": int(n_splits), "n_splits_fitted": int(n_fitted),
        "median_nugget_fraction": median_nugget, "nugget_fraction_iqr": iqr,
    }


def nugget_fraction(
    counts: NDArray,
    n_splits: int = N_SPLITS_DEFAULT,
    rng: np.random.Generator | None = None,
    bin_width_s: float = 0.1,
    max_lag: int = MAX_LAG_DEFAULT,
) -> dict:
    """Cross-validated observation-noise fraction of a population's leading latent.

    Parameters
    ----------
    counts : (trials, units, bins) binned population matrix (spike counts or
        band power; any modality that reduces to a binned population matrix).
    n_splits : number of random half-splits scored (>= 8 recommended).
    bin_width_s : bin width in seconds, used to convert the fitted decay rate
        to a timescale.
    max_lag : the autocovariance is computed out to ``min(n_bins - 1, max_lag)``.

    Returns
    -------
    dict with the median and IQR of the cross-validated nugget fraction over
    fitted splits, the in-sample counterpart, and per-split diagnostics. Each
    half-split fits a :class:`FrozenPSTHTransform` and the leading PCA
    direction on the training half only, projects the held-out half onto it,
    removes each held-out trial's own temporal mean, and decomposes the
    resulting autocovariance (:func:`_decompose_autocovariance`). A split
    whose autocovariance has fewer than three positive lags at k>=1 cannot be
    decomposed and is recorded with status "fewer_than_three_positive_lags"
    rather than folded into the median as a missing value. The session-level
    status is "fitted" only when at least three of the requested splits
    produced a decomposable autocovariance (the same floor the per-split fit
    applies to positive lags, applied one level up to splits); otherwise it
    is "fewer_than_three_positive_lags" and the median/IQR are reported as
    null, since one or two lucky splits are too noisy a basis for a
    reported point estimate.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (trials, units, bins)")
    n_trials, n_units, n_bins = counts.shape
    lag_cap = max(1, min(n_bins - 1, max_lag))

    transform_all = FrozenPSTHTransform().fit(counts)
    z_all = transform_all.transform(counts)
    latent_all = _leading_latent_projection(z_all, z_all)
    in_sample = _decompose_autocovariance(_lagged_autocovariance(latent_all, lag_cap), bin_width_s)

    per_split = []
    for split_index in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        train_idx, test_idx = perm[:half], perm[half:]
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue
        transform = FrozenPSTHTransform().fit(counts[train_idx])
        z_train = transform.transform(counts[train_idx])
        z_test = transform.transform(counts[test_idx])
        latent_test = _leading_latent_projection(z_train, z_test)
        c = _lagged_autocovariance(latent_test, lag_cap)
        result = _decompose_autocovariance(c, bin_width_s)
        result["split"] = split_index
        per_split.append(result)

    n_total = len(per_split)
    fitted = [r for r in per_split if r["status"] == "fitted"]
    n_fitted = len(fitted)
    # A session-level median needs a floor on how many splits actually
    # produced a decomposable autocovariance -- the same "at least three"
    # bar the per-split fit applies to positive lags, applied one level up
    # to splits. Below it, the few fitted splits are too noisy a basis for
    # a reported median and the session is reported as unfittable rather
    # than as a point estimate built from 1-2 lucky splits.
    session_status = "fitted" if n_fitted >= MIN_POSITIVE_LAGS else "fewer_than_three_positive_lags"

    if fitted:
        nuggets = np.array([r["nugget_fraction"] for r in fitted])
        timescales = np.array([t for t in (r["slow_timescale_s"] for r in fitted) if t is not None])
        n_lags = np.array([r["n_lags_used"] for r in fitted])
        median_nugget = float(np.median(nuggets))
        iqr = [float(np.percentile(nuggets, 25)), float(np.percentile(nuggets, 75))]
        median_timescale = float(np.median(timescales)) if len(timescales) else None
        median_n_lags = float(np.median(n_lags))
    else:
        median_nugget = None
        iqr = None
        median_timescale = None
        median_n_lags = float(np.median([r["n_lags_used"] for r in per_split])) if per_split else 0.0

    return {
        "status": session_status,
        "n_splits_requested": int(n_splits),
        "n_splits_total": int(n_total),
        "n_splits_fitted": int(n_fitted),
        "median_nugget_fraction": median_nugget,
        "nugget_fraction_iqr": iqr,
        "median_slow_timescale_s": median_timescale,
        "n_lags_used": median_n_lags,
        "in_sample_nugget_fraction": in_sample["nugget_fraction"],
        "in_sample_status": in_sample["status"],
        "n_trials": int(n_trials),
        "n_units": int(n_units),
        "n_bins": int(n_bins),
        "lag_cap": int(lag_cap),
        "bin_width_s": float(bin_width_s),
        "per_split": per_split,
    }
