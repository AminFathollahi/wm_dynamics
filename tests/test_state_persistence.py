import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from observability import _leading_latent_projection  # noqa: E402
from state_persistence import (  # noqa: E402
    _d_series,
    _lag_pairs,
    attenuation_constancy_check,
    binomial_thin,
    classify_gap_profile,
    classify_lag_profile,
    classify_lag_profile_segmented,
    fit_breakpoint_from_pooled_profile,
    fit_segmented_breakpoint,
    per_unit_permutation_null_r_gap_profile,
    per_unit_permutation_null_r_lag_profile,
    permutation_null_validation,
    poisson_null_r_gap_profile,
    poisson_null_r_lag_profile,
    r_gap_profile,
    r_lag_profile,
    segmented_breakpoint_summary,
    simulate_no_shared_state_population,
    simulate_planted_population,
)

N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S = 150, 45, 23, 0.1


def _ratio(profile: dict) -> float:
    gaps = profile["gaps"]
    return gaps[3]["r_median"] / gaps[1]["r_median"]


def test_pure_offset_gives_a_flat_profile():
    counts = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=None, share_static=1.0, share_slow=0.0,
        rng=np.random.default_rng(1),
    )
    profile = r_gap_profile(counts, rng=np.random.default_rng(2))
    assert profile["status"] == "fitted"
    assert _ratio(profile) > 0.85


def test_ar1_decay_ratio_orders_with_time_constant():
    fast = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=0.3, share_static=0.0, share_slow=1.0,
        rng=np.random.default_rng(3),
    )
    slow = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=10.0, share_static=0.0, share_slow=1.0,
        rng=np.random.default_rng(4),
    )
    fast_profile = r_gap_profile(fast, rng=np.random.default_rng(5))
    slow_profile = r_gap_profile(slow, rng=np.random.default_rng(6))
    assert fast_profile["status"] == "fitted" and slow_profile["status"] == "fitted"
    fast_gaps = fast_profile["gaps"]
    assert fast_gaps[1]["r_median"] > fast_gaps[3]["r_median"]
    assert _ratio(fast_profile) < _ratio(slow_profile)
    assert _ratio(fast_profile) < 0.3
    assert _ratio(slow_profile) > 0.7


def test_mixture_ratio_is_intermediate_between_offset_and_ar1():
    offset = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=None, share_static=1.0, share_slow=0.0,
        rng=np.random.default_rng(7),
    )
    ar1 = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=0.6, share_static=0.0, share_slow=1.0,
        rng=np.random.default_rng(8),
    )
    mixture = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=0.6, share_static=0.5, share_slow=0.5,
        rng=np.random.default_rng(9),
    )
    offset_ratio = _ratio(r_gap_profile(offset, rng=np.random.default_rng(10)))
    ar1_ratio = _ratio(r_gap_profile(ar1, rng=np.random.default_rng(11)))
    mixture_ratio = _ratio(r_gap_profile(mixture, rng=np.random.default_rng(12)))
    assert ar1_ratio < mixture_ratio < offset_ratio


def test_rotation_crosses_zero():
    counts = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=None, share_static=1.0, share_slow=0.0,
        rotate=True, rotation_radians=1.5 * np.pi, rng=np.random.default_rng(13),
    )
    profile = r_gap_profile(counts, rng=np.random.default_rng(14))
    gaps = profile["gaps"]
    assert gaps[1]["r_median"] > 0.0
    assert gaps[3]["r_median"] < 0.0
    assert gaps[3]["r_median"] < gaps[1]["r_median"]


def test_invariant_to_a_global_attenuation():
    """The invariance property this module relies on: thinning changes the LEVEL of r(gap)
    but not its SHAPE (the ratio across gaps), because a constant
    attenuation factor scales every window equally."""
    counts = simulate_planted_population(
        N_TRIALS, N_UNITS, N_BINS, BIN_WIDTH_S, tau_s=1.2, share_static=0.0, share_slow=1.0,
        rng=np.random.default_rng(15),
    )
    full = r_gap_profile(counts, rng=np.random.default_rng(16))
    thinned = binomial_thin(counts, keep_probability=0.2, rng=np.random.default_rng(17))
    thinned_profile = r_gap_profile(thinned, rng=np.random.default_rng(18))
    assert full["status"] == "fitted" and thinned_profile["status"] == "fitted"
    assert thinned_profile["gaps"][1]["r_median"] < full["gaps"][1]["r_median"] - 0.05
    assert abs(_ratio(thinned_profile) - _ratio(full)) < 0.35


def test_pure_poisson_replicate_gives_r_near_zero():
    rng = np.random.default_rng(19)
    psth = np.clip(0.3 + 0.1 * rng.normal(size=(N_UNITS, N_BINS)), 0.05, None)
    counts = rng.poisson(np.broadcast_to(psth, (N_TRIALS, N_UNITS, N_BINS))).astype(float)
    profile = r_gap_profile(counts, rng=np.random.default_rng(20))
    assert profile["status"] == "fitted"
    for gap_result in profile["gaps"].values():
        assert abs(gap_result["r_median"]) < 0.2


def test_null_profile_and_reachability_hold_for_poisson_replicate():
    counts = simulate_planted_population(
        40, 30, 23, BIN_WIDTH_S, tau_s=1.2, share_static=0.0, share_slow=1.0, rate_hz=0.3,
        rng=np.random.default_rng(21),
    )
    null = poisson_null_r_gap_profile(counts, n_replicates=8, rng=np.random.default_rng(22))
    assert null["n_replicates_fitted"] >= 5
    for gap_null in null["gaps"].values():
        assert abs(gap_null["r_null_median"]) < 0.3


def test_too_few_bins_returns_documented_status_not_exception():
    counts = np.random.default_rng(23).poisson(1.0, size=(20, 10, 2)).astype(float)
    profile = r_gap_profile(counts, n_windows=4, rng=np.random.default_rng(24))
    assert profile["status"] == "fewer_than_three_fitted_splits"
    assert profile["gaps"] == {}


def test_permutation_null_destroys_a_planted_shared_signal():
    """On a planted shared latent riding on many units with weak per-unit
    loading, the per-unit permutation null must remove most of the signal
    -- a single unit's own autocorrelation cannot substitute for the
    cross-unit averaging a real shared state needs."""
    shared = simulate_planted_population(
        n_trials=80, n_units=80, n_bins=24, bin_width_s=0.1, tau_s=None, share_static=1.0, share_slow=0.0,
        rate_hz=0.3, loading_scale=0.15, rng=np.random.default_rng(200),
    )
    direct = r_gap_profile(shared, n_splits=20, rng=np.random.default_rng(201))
    null = per_unit_permutation_null_r_gap_profile(shared, n_replicates=20, rng=np.random.default_rng(202))
    assert direct["status"] == "fitted"
    direct_r1 = direct["gaps"][1]["r_median"]
    null_r1 = null["gaps"][1]["r_null_median"]
    assert direct_r1 > 0.3  # a real signal exists to destroy
    assert abs(null_r1) < 0.25 and null_r1 < 0.4 * direct_r1  # near zero, and far below the direct fit


def test_permutation_null_reproduces_shortest_lag_step_with_no_shared_state():
    """On a population with NO shared latent but strong within-unit
    autocorrelation, the null must reproduce the direct fit's
    shortest-lag inflation almost fully, since that inflation never depended
    on cross-unit structure. A null that failed this could not be used to
    clear the shortest-lag step."""
    no_shared = simulate_no_shared_state_population(
        n_trials=80, n_units=10, n_bins=24, bin_width_s=0.1, tau_s=0.3, rate_hz=5.0, loading_scale=1.0,
        rng=np.random.default_rng(203),
    )
    direct = r_gap_profile(no_shared, n_splits=20, rng=np.random.default_rng(204))
    null = per_unit_permutation_null_r_gap_profile(no_shared, n_replicates=20, rng=np.random.default_rng(205))
    assert direct["status"] == "fitted"
    assert direct["gaps"][1]["r_median"] > direct["gaps"][3]["r_median"]
    assert null["gaps"][1]["r_null_median"] > null["gaps"][3]["r_null_median"]
    assert null["gaps"][1]["r_null_median"] > 0.5 * direct["gaps"][1]["r_median"]


def test_permutation_null_validation_passes_on_its_reference_regimes():
    result = permutation_null_validation()
    assert result["check_a_destroys_planted_shared_signal"]["passes"]
    assert result["check_b_reproduces_shortest_lag_inflation_with_no_shared_state"]["passes"]
    assert result["usable"]


def _replicate_gaps(regime_fn, n_replicates, n_windows, seed0):
    profiles, poisson_nulls, permutation_nulls = [], [], []
    for i in range(n_replicates):
        counts = regime_fn(np.random.default_rng(seed0 + i))
        profile = r_gap_profile(counts, n_splits=10, n_windows=n_windows, rng=np.random.default_rng(seed0 + 1000 + i))
        if profile["status"] != "fitted":
            continue
        poisson_null = poisson_null_r_gap_profile(
            counts, n_replicates=8, n_splits_per_replicate=4, n_windows=n_windows, rng=np.random.default_rng(seed0 + 2000 + i))
        permutation_null = per_unit_permutation_null_r_gap_profile(
            counts, n_replicates=8, n_splits_per_replicate=4, n_windows=n_windows, rng=np.random.default_rng(seed0 + 3000 + i))
        profiles.append(profile["gaps"])
        poisson_nulls.append(poisson_null["gaps"])
        permutation_nulls.append(permutation_null["gaps"])
    return profiles, poisson_nulls, permutation_nulls


def test_multi_lag_decay_and_fast_transient_are_separated_by_the_profile_rule():
    """A planted multi-lag AR(1) at W=8 decays across every lag (monotone
    non-increasing, and the decay survives past
    the shortest lag), while a planted static-floor-plus-fast-transient
    steps once then goes flat. The predeclared rule must tell these apart
    from the profiles and nulls alone, with no number read by a human."""
    n_trials, n_units, n_bins, bin_width_s = 60, 30, 40, 0.1

    def multi_lag(rng):
        return simulate_planted_population(
            n_trials, n_units, n_bins, bin_width_s, tau_s=1.2, share_static=0.0, share_slow=1.0,
            rate_hz=0.6, rng=rng)

    def fast_transient_plus_floor(rng):
        return simulate_planted_population(
            n_trials, n_units, n_bins, bin_width_s, tau_s=0.2, share_static=0.7, share_slow=0.3,
            rate_hz=0.6, rng=rng)

    decay_profiles, decay_poisson, decay_perm = _replicate_gaps(multi_lag, 12, 8, seed0=300)
    decay = classify_gap_profile(decay_profiles, decay_poisson, decay_perm, w_max=8)
    assert decay["adjacent_steps"]["monotone_non_increasing"]
    assert decay["branch"] == "graded_decay"

    step_profiles, step_poisson, step_perm = _replicate_gaps(fast_transient_plus_floor, 12, 8, seed0=400)
    step = classify_gap_profile(step_profiles, step_poisson, step_perm, w_max=8)
    assert step["branch"] == "fast_component_plus_stable_floor"


def test_lag_pairs_hand_checked():
    """n_bins=10, width=3, lag=3: five (start, start+lag) pairs, s=0..4 --
    n_bins - width - lag + 1 = 10 - 3 - 3 + 1 = 5. Empty when lag + width
    exceeds n_bins, and empty when lag < width (the two windows would
    overlap)."""
    assert _lag_pairs(10, 3, 3) == [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
    assert _lag_pairs(10, 3, 8) == []
    assert _lag_pairs(10, 3, 2) == []


def _replicate_lags(regime_fn, n_replicates, width_bins, seed0):
    profiles, poisson_nulls, permutation_nulls = [], [], []
    for i in range(n_replicates):
        counts = regime_fn(np.random.default_rng(seed0 + i))
        profile = r_lag_profile(counts, width_bins, n_splits=8, rng=np.random.default_rng(seed0 + 1000 + i))
        if profile["status"] != "fitted":
            continue
        poisson_null = poisson_null_r_lag_profile(
            counts, width_bins, n_replicates=10, n_splits_per_replicate=4, rng=np.random.default_rng(seed0 + 2000 + i))
        permutation_null = per_unit_permutation_null_r_lag_profile(
            counts, width_bins, n_replicates=10, n_splits_per_replicate=4, rng=np.random.default_rng(seed0 + 3000 + i))
        profiles.append(profile["lags"])
        poisson_nulls.append(poisson_null["lags"])
        permutation_nulls.append(permutation_null["lags"])
    return profiles, poisson_nulls, permutation_nulls


def test_flat_static_state_gives_flat_positive_d_perm():
    def flat_regime(rng):
        return simulate_planted_population(
            60, 30, 30, 0.1, tau_s=None, share_static=1.0, share_slow=0.0, rate_hz=0.6, loading_scale=0.35, rng=rng)

    profiles, pois, perm = _replicate_lags(flat_regime, 12, 3, seed0=500)
    result = classify_lag_profile(profiles, pois, perm, width_bins=3, bin_width_s=0.1)
    assert result["branch"] == "flat_cross_unit_state"
    assert not result["contrast_b_slope_excluding_adjacency"].get("significant_negative", False)
    assert not result["contrast_b_slope_including_adjacency"].get("significant_negative", False)
    assert result["n_lags_clearing_fdr"] > result["n_lags_tested"] / 2


def test_ar1_state_gives_graded_decay_and_recovers_planted_timescale():
    """A planted AR(1) shared state at tau=0.8s: the slope of d_perm
    excluding the adjacency point is significantly negative, the rule
    returns graded_decay, and a simple two-point estimate off the two
    shortest lags recovers the planted tau to a loose tolerance -- a proper
    calibration-ladder fit is explicitly out of scope here, so this
    checks direction and rough magnitude, not calibration-grade precision."""
    def ar1_regime(rng):
        return simulate_planted_population(
            60, 30, 30, 0.1, tau_s=0.8, share_static=0.0, share_slow=1.0, rate_hz=0.6, loading_scale=0.35, rng=rng)

    profiles, pois, perm = _replicate_lags(ar1_regime, 12, 3, seed0=600)
    result = classify_lag_profile(profiles, pois, perm, width_bins=3, bin_width_s=0.1)
    assert result["branch"] == "graded_decay"
    assert result["contrast_b_slope_excluding_adjacency"]["significant_negative"]

    lags_sorted = sorted(set().union(*[set(g) for g in profiles]))
    lag1, lag2 = lags_sorted[0], lags_sorted[1]
    r1 = float(np.median([g[lag1]["r_median"] for g in profiles if lag1 in g]))
    r2 = float(np.median([g[lag2]["r_median"] for g in profiles if lag2 in g]))
    assert r1 > r2 > 0.0
    tau_hat = -0.1 / np.log(r2 / r1)
    assert abs(tau_hat - 0.8) < 0.5


def test_no_shared_state_rate_wander_gives_no_state_above_the_within_unit_floor():
    """No shared latent, per-unit AR(1) rate wander at tau=2s: the raw r
    declines across the whole lag range and the permutation null reproduces
    that decline, so d_perm has no significant slope and does not clear
    existence at a majority of lags -- the rule must return
    no_state_above_the_within_unit_floor on data built to have exactly that
    answer, not mistake the nuisance floor's own decline for a real state."""
    def no_shared_regime(rng):
        return simulate_no_shared_state_population(
            60, 15, 30, 0.1, tau_s=2.0, rate_hz=3.0, loading_scale=0.8, rng=rng)

    profiles, pois, perm = _replicate_lags(no_shared_regime, 12, 3, seed0=700)
    result = classify_lag_profile(profiles, pois, perm, width_bins=3, bin_width_s=0.1)

    lags_sorted = sorted(set().union(*[set(g) for g in profiles]))
    r_first = float(np.median([g[lags_sorted[0]]["r_median"] for g in profiles if lags_sorted[0] in g]))
    r_last = float(np.median([g[lags_sorted[-1]]["r_median"] for g in profiles if lags_sorted[-1] in g]))
    null_first = float(np.median([n[lags_sorted[0]]["r_null_median"] for n in perm if lags_sorted[0] in n]))
    null_last = float(np.median([n[lags_sorted[-1]]["r_null_median"] for n in perm if lags_sorted[-1] in n]))
    assert r_first > r_last
    assert null_first > null_last

    assert result["branch"] == "no_state_above_the_within_unit_floor"
    assert result["n_lags_clearing_fdr"] == 0


def test_attenuation_constancy_check_passes_when_attenuation_is_the_only_width_effect():
    """The same underlying process measured at two widths: any r difference
    between widths is pure measurement-noise attenuation (narrower windows
    average fewer bins into a noisier per-trial mean), so the ratio
    r(w_narrow, L) / r(w_wide, L) must be constant across lag."""
    def regime(rng):
        return simulate_planted_population(
            80, 40, 40, 0.1, tau_s=1.0, share_static=0.3, share_slow=0.7, rate_hz=0.8, loading_scale=0.35, rng=rng)

    profiles_w2, profiles_w5 = [], []
    for i in range(14):
        counts = regime(np.random.default_rng(9000 + i))
        p2 = r_lag_profile(counts, width_bins=2, n_splits=10, rng=np.random.default_rng(9100 + i))
        p5 = r_lag_profile(counts, width_bins=5, n_splits=10, rng=np.random.default_rng(9200 + i))
        if p2["status"] == "fitted" and p5["status"] == "fitted":
            profiles_w2.append(p2["lags"])
            profiles_w5.append(p5["lags"])

    result = attenuation_constancy_check(profiles_w2, profiles_w5, bin_width_s=0.1)
    assert result["constant_across_lag"]


class TestProjectionFnParameterIsThreadedThrough:
    """r_lag_profile and per_unit_permutation_null_r_lag_profile must accept
    an alternative leading-latent basis fit (projection_fn) and use it
    instead of the PCA default -- what lets a caller ask this project's own
    persistence-contrast machinery under a different noise model without
    duplicating the split/null/window logic."""

    def test_default_projection_fn_matches_explicit_pca(self):
        counts = simulate_planted_population(60, 20, 12, 0.1, tau_s=1.0, share_static=0.3, share_slow=0.7,
                                               rate_hz=0.8, loading_scale=0.35, rng=np.random.default_rng(0))
        default_profile = r_lag_profile(counts, width_bins=3, n_splits=8, rng=np.random.default_rng(1))
        explicit_pca_profile = r_lag_profile(counts, width_bins=3, n_splits=8, rng=np.random.default_rng(1),
                                              projection_fn=_leading_latent_projection)
        assert default_profile["lags"] == explicit_pca_profile["lags"]

    def test_a_projection_fn_that_always_returns_none_yields_fewer_than_three_fitted_splits(self):
        counts = simulate_planted_population(60, 20, 12, 0.1, tau_s=1.0, share_static=0.3, share_slow=0.7,
                                               rate_hz=0.8, loading_scale=0.35, rng=np.random.default_rng(2))
        profile = r_lag_profile(counts, width_bins=3, n_splits=8, rng=np.random.default_rng(3),
                                 projection_fn=lambda fit, apply: None)
        assert profile["status"] == "fewer_than_three_fitted_splits"
        assert profile["n_splits_fitted"] == 0

    def test_permutation_null_passes_projection_fn_through_to_r_lag_profile(self):
        calls = []

        def _tracking_projection(psth_fit, psth_apply):
            calls.append(1)
            return _leading_latent_projection(psth_fit, psth_apply)

        counts = simulate_planted_population(60, 20, 12, 0.1, tau_s=1.0, share_static=0.3, share_slow=0.7,
                                               rate_hz=0.8, loading_scale=0.35, rng=np.random.default_rng(4))
        per_unit_permutation_null_r_lag_profile(
            counts, width_bins=3, n_replicates=2, n_splits_per_replicate=3, rng=np.random.default_rng(5),
            projection_fn=_tracking_projection,
        )
        assert len(calls) > 0
