"""variance_partition.py -- four-way split of a population's held-out
single-trial latent variance, and a held-position test that needs no decay
fit.

The estimand: fit the leading PCA direction of a binned population matrix on
training trials, project held-out trials onto it, and split the held-out
latent's variance into four shares that sum (up to estimation noise) to the
total:

    cond    the across-trial-average time course       survives trial averaging
    static  a per-trial constant offset                 single-trial position
    slow    a per-trial decaying component               single-trial dynamics
    white   temporally uncorrelated residual             counting noise floor

``cond`` is estimated as the cross-product of two independent estimates of
the same average time course (one from the training trials, one from the
held-out trials) rather than the square of either alone, because squaring a
single noisy estimate adds its own sampling variance to the signal term.
``static`` is the mean square of each held-out trial's temporal mean once
the training-trial average time course has been subtracted. ``slow`` and
``white`` come from regressing the log lagged autocovariance of the
remaining (doubly demeaned) residual on lag, exactly as in
:mod:`observability`, whose autocovariance decomposition this module reuses
rather than reimplementing.

Every reported share is the median over repeated random half-splits of the
trials, scored on the held-out half only; the in-sample counterpart (basis
fit and scored on the same trials) is reported alongside it because in-sample
estimation is known to understate the white share.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.typing import NDArray

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from observability import _decompose_autocovariance, _lagged_autocovariance, _leading_latent_projection  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402

MAX_LAG_DEFAULT = 8
N_SPLITS_DEFAULT = 8
N_NULL_REPLICATES_DEFAULT = 20
SHARE_KEYS = ("cond", "static", "slow", "white")


def _partition_one_split(counts: NDArray, train_idx: NDArray, test_idx: NDArray,
                          bin_width_s: float, max_lag: int) -> dict:
    transform = FrozenPSTHTransform().fit(counts[train_idx])
    z_train = transform.transform(counts[train_idx])
    z_test = transform.transform(counts[test_idx])
    latent_train = _leading_latent_projection(z_train, z_train)
    latent_test = _leading_latent_projection(z_train, z_test)

    var_total = float(np.mean(latent_test ** 2))
    train_time_course = latent_train.mean(axis=0)
    held_out_time_course = latent_test.mean(axis=0)
    cond = float(np.mean(train_time_course * held_out_time_course))

    resid = latent_test - train_time_course[None, :]
    static = float(np.mean(resid.mean(axis=1) ** 2))

    lag_cap = max(1, min(resid.shape[1] - 2, max_lag))
    c = _lagged_autocovariance(resid, lag_cap)
    decomposition = _decompose_autocovariance(c, bin_width_s)
    if decomposition["status"] == "fitted":
        slow = min(decomposition["s2_slow"], decomposition["c0"])
        white = max(decomposition["c0"] - slow, 0.0)
    else:
        slow = 0.0
        white = decomposition["c0"]

    return {
        "var_total": var_total, "cond": cond, "static": static, "slow": slow, "white": white,
        "status": decomposition["status"], "n_lags_used": decomposition["n_lags_used"],
        "rho": decomposition["rho"], "slow_timescale_s": decomposition["slow_timescale_s"],
    }


def _held_position_one_split(counts: NDArray, train_idx: NDArray, test_idx: NDArray) -> float | None:
    """Correlation, across held-out trials, between a trial's mean latent
    position in the first half of the epoch's bins and its mean position in
    the second half, each half centred across trials to remove the
    across-trial-average time course. Needs no decay fit and therefore
    carries no identifiability band."""
    transform = FrozenPSTHTransform().fit(counts[train_idx])
    z_train = transform.transform(counts[train_idx])
    z_test = transform.transform(counts[test_idx])
    latent_test = _leading_latent_projection(z_train, z_test)
    n_bins = latent_test.shape[1]
    if n_bins < 2:
        return None
    mid = n_bins // 2
    first_half = latent_test[:, :mid].mean(axis=1)
    second_half = latent_test[:, mid:].mean(axis=1)
    first_half = first_half - first_half.mean()
    second_half = second_half - second_half.mean()
    if first_half.std() == 0.0 or second_half.std() == 0.0:
        return None
    return float(np.corrcoef(first_half, second_half)[0, 1])


def partition_single_trial_variance(
    counts: NDArray,
    n_splits: int = N_SPLITS_DEFAULT,
    rng: np.random.Generator | None = None,
    bin_width_s: float = 0.1,
    max_lag: int = MAX_LAG_DEFAULT,
) -> dict:
    """Cross-validated four-way variance split plus the held-position
    correlation of a population's leading latent.

    Parameters
    ----------
    counts : (trials, units, bins) binned population matrix (spike counts or
        band power; any modality that reduces to a binned population matrix).
    n_splits : number of random half-splits scored (>= 8 required by spec).
    bin_width_s : bin width in seconds, used to convert the fitted decay rate
        to a timescale.
    max_lag : the residual autocovariance is computed out to
        ``min(n_bins - 2, max_lag)``.

    Returns
    -------
    dict with the median and IQR of each of the four cross-validated shares
    (as a fraction of held-out total variance) over fitted splits, the
    in-sample counterpart, the held-position correlation, and per-split
    diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError("counts must have shape (trials, units, bins)")
    n_trials, n_units, n_bins = counts.shape
    all_idx = np.arange(n_trials)

    in_sample = _partition_one_split(counts, all_idx, all_idx, bin_width_s, max_lag)

    per_split = []
    held_position_per_split = []
    for split_index in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        train_idx, test_idx = perm[:half], perm[half:]
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue
        result = _partition_one_split(counts, train_idx, test_idx, bin_width_s, max_lag)
        result["split"] = split_index
        per_split.append(result)
        r = _held_position_one_split(counts, train_idx, test_idx)
        if r is not None:
            held_position_per_split.append(r)

    n_total = len(per_split)
    fitted = [r for r in per_split if r["var_total"] > 0]
    n_fitted = len(fitted)
    status = "fitted" if n_fitted >= 3 else "fewer_than_three_fitted_splits"

    shares: dict[str, float | list | None] = {}
    if fitted:
        for key in SHARE_KEYS:
            fractions = np.array([r[key] / r["var_total"] for r in fitted])
            shares[f"{key}_fraction_median"] = float(np.median(fractions))
            shares[f"{key}_fraction_iqr"] = [float(np.percentile(fractions, 25)), float(np.percentile(fractions, 75))]
        n_lags = np.array([r["n_lags_used"] for r in fitted])
        shares["n_lags_used_median"] = float(np.median(n_lags))
        n_slow_fitted = sum(1 for r in fitted if r["status"] == "fitted")
        shares["n_splits_with_fittable_decay"] = int(n_slow_fitted)
    else:
        for key in SHARE_KEYS:
            shares[f"{key}_fraction_median"] = None
            shares[f"{key}_fraction_iqr"] = None
        shares["n_lags_used_median"] = None
        shares["n_splits_with_fittable_decay"] = 0

    in_sample_shares = {
        f"{key}_fraction": (in_sample[key] / in_sample["var_total"]) if in_sample["var_total"] > 0 else None
        for key in SHARE_KEYS
    }

    return {
        "status": status,
        "n_splits_requested": int(n_splits),
        "n_splits_total": int(n_total),
        "n_splits_fitted": int(n_fitted),
        **shares,
        "in_sample": in_sample_shares,
        "held_position_correlation_median": float(np.median(held_position_per_split)) if held_position_per_split else None,
        "held_position_n_splits_fitted": len(held_position_per_split),
        "n_trials": int(n_trials),
        "n_units": int(n_units),
        "n_bins": int(n_bins),
        "bin_width_s": float(bin_width_s),
        "per_split": per_split,
    }


def poisson_null_from_counts(counts: NDArray, n_replicates: int = N_NULL_REPLICATES_DEFAULT,
                              n_splits: int = N_SPLITS_DEFAULT, rng: np.random.Generator | None = None,
                              bin_width_s: float = 0.1, max_lag: int = MAX_LAG_DEFAULT) -> dict:
    """Per-session negative control: simulate ``n_replicates`` Poisson
    populations from this session's own per-unit per-bin PSTH -- a rate
    profile identical on every trial by construction, so no trial-specific
    structure exists in the replicate -- and run the identical partition on
    each. Reports the null distribution's median and 95th percentile for
    each share and for the held-position correlation, which is what
    separates a statement about the code from a statement about firing
    rates and unit counts alone."""
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.asarray(counts, dtype=float)
    n_trials = counts.shape[0]
    psth = counts.mean(axis=0)  # (units, bins), identical rate profile for every simulated trial
    psth = np.clip(psth, 1e-6, None)

    replicate_results = []
    for _ in range(n_replicates):
        simulated = rng.poisson(lam=psth[None, :, :], size=(n_trials,) + psth.shape).astype(float)
        replicate_results.append(
            partition_single_trial_variance(simulated, n_splits=n_splits, rng=rng, bin_width_s=bin_width_s, max_lag=max_lag)
        )

    fitted = [r for r in replicate_results if r["status"] == "fitted"]
    null: dict[str, float | None] = {}
    for key in SHARE_KEYS:
        values = np.array([r[f"{key}_fraction_median"] for r in fitted if r[f"{key}_fraction_median"] is not None])
        null[f"{key}_null_median"] = float(np.median(values)) if len(values) else None
        null[f"{key}_null_p95"] = float(np.percentile(values, 95)) if len(values) else None
    held_position = np.array([r["held_position_correlation_median"] for r in fitted
                               if r["held_position_correlation_median"] is not None])
    null["held_position_null_median"] = float(np.median(held_position)) if len(held_position) else None
    null["held_position_null_p95"] = float(np.percentile(held_position, 95)) if len(held_position) else None
    null["n_replicates_requested"] = int(n_replicates)
    null["n_replicates_fitted"] = len(fitted)
    return null
