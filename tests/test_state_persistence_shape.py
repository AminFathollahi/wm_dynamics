from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_state_persistence_shape import cohort_control  # noqa: E402
from state_persistence import (  # noqa: E402
    _d_series,
    breakpoint_bootstrap_ci,
    classify_lag_profile_segmented,
    fit_breakpoint_from_pooled_profile,
    geometry_vs_clock_verdict,
    segmented_slope_test,
)

BIN_WIDTH_S = 0.1


def _synthetic_series(lags: list[int], value_fn, n_sessions: int, noise_sem: float, seed: int) -> list[dict[int, float]]:
    rng = np.random.default_rng(seed)
    return [{lag: value_fn(lag) + rng.normal(scale=noise_sem) for lag in lags} for _ in range(n_sessions)]


def _profiles_from_d_perm(series_list: list[dict[int, float]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Builds minimal (profile, poisson_null, permutation_null) dicts
    directly from a planted d_perm(L) series -- what
    :func:`classify_lag_profile_segmented` and :func:`segmented_shape_
    contrasts` need (``r_median`` on the profile, ``r_null_median`` on both
    nulls) -- with the null held at zero, so d_perm equals the planted
    value exactly by construction. This tests the decision rule and the
    breakpoint fit directly against a KNOWN shape, rather than through the
    extra noise a full population simulation would add."""
    profiles, pois, perm = [], [], []
    for series in series_list:
        profiles.append({lag: {"r_median": v} for lag, v in series.items()})
        pois.append({lag: {"r_null_median": 0.0} for lag in series})
        perm.append({lag: {"r_null_median": 0.0} for lag in series})
    return profiles, pois, perm


def test_two_component_profile_gives_fast_component_plus_flat_floor_with_breakpoint_recovery():
    """A planted fast decay from lag=3 to the breakpoint at lag=10 (1.0 s),
    then a flat floor to lag=20: the two-sided segmented rule must return
    fast_component_plus_flat_floor, and the pooled-profile breakpoint fit
    (found unbiased at realistic session counts and noise levels by direct
    calibration against planted profiles, unlike the per-session-then-median
    approach it replaces for this purpose) must land within one lag of the
    planted breakpoint."""
    lags = list(range(3, 21))
    true_bp_bin = 10
    floor = 0.02

    def value_fn(lag):
        t = lag * BIN_WIDTH_S
        bp_t = true_bp_bin * BIN_WIDTH_S
        return floor + 0.09 * (bp_t - t) if t <= bp_t else floor

    series = _synthetic_series(lags, value_fn, n_sessions=46, noise_sem=0.06, seed=1)
    fit = fit_breakpoint_from_pooled_profile(series, BIN_WIDTH_S)
    assert fit is not None
    assert abs(fit["breakpoint_bins"] - true_bp_bin) <= 1

    profiles, pois, perm = _profiles_from_d_perm(series)
    result = classify_lag_profile_segmented(profiles, pois, perm, width_bins=3, bin_width_s=BIN_WIDTH_S,
                                             breakpoint_bins=fit["breakpoint_bins"])
    assert result["branch"] == "fast_component_plus_flat_floor"
    assert result["early_segment_significant_negative"]
    assert not result["late_segment_significant_negative"]
    assert not result["late_segment_significant_positive"]


def test_single_exponential_with_no_floor_gives_graded_decay():
    """A planted decay that keeps falling across the whole reachable range
    (no floor placed inside it): the late segment (split at the
    pooled-profile breakpoint) must ALSO be significantly negative, so the
    rule returns graded_decay rather than fast_component_plus_flat_floor,
    and the whole-range slope agrees in sign with the early-segment slope."""
    lags = list(range(3, 21))

    def value_fn(lag):
        t = lag * BIN_WIDTH_S
        return 0.5 - 0.09 * t

    series = _synthetic_series(lags, value_fn, n_sessions=46, noise_sem=0.03, seed=2)
    fit = fit_breakpoint_from_pooled_profile(series, BIN_WIDTH_S)
    assert fit is not None

    profiles, pois, perm = _profiles_from_d_perm(series)
    result = classify_lag_profile_segmented(profiles, pois, perm, width_bins=3, bin_width_s=BIN_WIDTH_S,
                                             breakpoint_bins=fit["breakpoint_bins"])
    assert result["branch"] == "graded_decay"
    whole = result["segmented_contrasts"]["by_statistic"]["d_perm"]["whole_range_slope_dilution_comparison"]["test"]
    early = result["segmented_contrasts"]["by_statistic"]["d_perm"]["early_segment_excluding_adjacency"]["test"]
    assert whole["mean_value"] < 0.0 and early["mean_value"] < 0.0
    assert abs(whole["mean_value"] - early["mean_value"]) < 0.06


def test_significantly_positive_slope_routes_to_off_list_not_flat():
    """The regression test for the round's own headline bug: a profile
    whose d_perm RISES significantly with lag (the macaque signature) must
    not satisfy 'not significantly negative' and fall into
    flat_cross_unit_state -- it must land in the explicit off-list bucket."""
    lags = list(range(3, 10))

    def value_fn(lag):
        t = lag * BIN_WIDTH_S
        return 0.02 + 0.09 * t

    series = _synthetic_series(lags, value_fn, n_sessions=25, noise_sem=0.03, seed=3)
    fit = fit_breakpoint_from_pooled_profile(series, BIN_WIDTH_S)
    assert fit is not None

    profiles, pois, perm = _profiles_from_d_perm(series)
    result = classify_lag_profile_segmented(profiles, pois, perm, width_bins=3, bin_width_s=BIN_WIDTH_S,
                                             breakpoint_bins=fit["breakpoint_bins"])
    assert result["branch"] == "unattributed_off_branch_list"
    assert result["branch"] != "flat_cross_unit_state"
    assert result["late_segment_significant_positive"] or result["early_segment_significant_positive"]


def _fake_lag_row(dataset: str, session: str, window_s: float, n_bins: int, width_bins: int,
                   value_fn, noise_sem: float, rng: np.random.Generator) -> dict:
    lags = list(range(width_bins, n_bins - width_bins + 1))
    profile_lags = {lag: {"r_median": value_fn(lag) + rng.normal(scale=noise_sem), "n_pairs": 10} for lag in lags}
    null_lags = {lag: {"r_null_median": 0.0, "r_null_p95": 0.05} for lag in lags}
    return {
        "dataset": dataset, "patient": session, "session": session, "structure": "pooled", "epoch": "delay",
        "bin_ms": BIN_WIDTH_S * 1000.0, "window_mode": "native", "window_s": window_s,
        "n_trials": 40, "n_units": 30, "n_bins": n_bins, "width_bins": width_bins,
        "profile": {"status": "fitted", "lags": profile_lags},
        "null_poisson": {"lags": null_lags},
        "null_permutation": {"lags": null_lags},
    }


def test_cohort_control_detects_a_breakpoint_that_moves_with_a_fixed_fraction_of_the_window():
    rng = np.random.default_rng(4)
    rows = []
    for window_s, n_bins, dataset in ((2.0, 20, "dataset_a"), (3.0, 30, "dataset_b")):
        breakpoint_s = 0.4 * window_s  # a fixed FRACTION of the window -- an estimator/geometry property

        def value_fn(lag, breakpoint_s=breakpoint_s):
            t = lag * BIN_WIDTH_S
            return 0.02 + 0.08 * (breakpoint_s - t) if t <= breakpoint_s else 0.02

        for i in range(10):
            rows.append(_fake_lag_row(dataset, f"{dataset}_s{i}", window_s, n_bins, 3, value_fn, 0.03, rng))

    result = cohort_control(rows, width=3)
    assert result["verdict"] == "moved_with_the_geometry"


def test_cohort_control_detects_a_breakpoint_that_stays_on_the_clock():
    rng = np.random.default_rng(5)
    rows = []
    breakpoint_s = 0.8  # a fixed number of SECONDS regardless of window length
    for window_s, n_bins, dataset in ((2.0, 20, "dataset_c"), (3.0, 30, "dataset_d")):

        def value_fn(lag, breakpoint_s=breakpoint_s):
            t = lag * BIN_WIDTH_S
            return 0.02 + 0.08 * (breakpoint_s - t) if t <= breakpoint_s else 0.02

        for i in range(10):
            rows.append(_fake_lag_row(dataset, f"{dataset}_s{i}", window_s, n_bins, 3, value_fn, 0.03, rng))

    result = cohort_control(rows, width=3)
    assert result["verdict"] == "stayed_on_the_clock"


def _synthetic_two_component_series(n_sessions: int, noise_sem: float, seed: int, n_lags: int = 18) -> list[dict[int, float]]:
    """The same planted fast-component-then-floor shape as the tests above
    (floor 0.02, early slope -0.09 r/s, true breakpoint at 8 bins = 0.8 s),
    parameterised by session count and per-session noise so two 'arms' with
    an IDENTICAL true shape can differ only in how precisely their own
    breakpoint can be located."""
    lags = list(range(3, 3 + n_lags))
    true_bp_s = 8 * BIN_WIDTH_S

    def value_fn(lag):
        t = lag * BIN_WIDTH_S
        return 0.02 + 0.09 * (true_bp_s - t) if t <= true_bp_s else 0.02

    return _synthetic_series(lags, value_fn, n_sessions, noise_sem, seed)


def test_common_range_comparison_finds_equal_arms_but_own_fitted_boundary_finds_them_different():
    """Regression test for holding a cross-arm comparison's range fixed in
    seconds: two arms with the IDENTICAL true fast component (same floor,
    same early rate, same true breakpoint) but different session counts and
    noise levels. Compared over the SAME declared range in seconds, their
    early-segment slopes must overlap (indistinguishable -- correctly reads
    them as the same shape). Compared each over its OWN noisily fitted
    breakpoint as the range endpoint, the two ranges differ enough that the
    resulting slope estimates no longer overlap -- an artifact of the
    per-arm fit rather than of the underlying shape, which is exactly the
    failure that arises when a between-arm comparison is built on each arm's
    own separately fitted boundary instead of on a range held fixed in
    seconds across every arm being compared."""
    arm_a = _synthetic_two_component_series(n_sessions=72, noise_sem=0.03, seed=71)
    arm_b = _synthetic_two_component_series(n_sessions=23, noise_sem=0.15, seed=72)

    fit_a = fit_breakpoint_from_pooled_profile(arm_a, BIN_WIDTH_S)
    fit_b = fit_breakpoint_from_pooled_profile(arm_b, BIN_WIDTH_S)
    assert fit_a is not None and fit_b is not None
    assert fit_a["breakpoint_bins"] != fit_b["breakpoint_bins"], \
        "test fixture requires the two noisy fits to actually diverge"

    common_a = segmented_slope_test(arm_a, BIN_WIDTH_S, (3, 8), exclude_lag_bins=3, alternative="two-sided")["test"]
    common_b = segmented_slope_test(arm_b, BIN_WIDTH_S, (3, 8), exclude_lag_bins=3, alternative="two-sided")["test"]
    common_overlap = common_a["ci_lower"] <= common_b["ci_upper"] and common_b["ci_lower"] <= common_a["ci_upper"]
    assert common_overlap, "the common-range comparison must find the two identical-shape arms equal"

    own_a = segmented_slope_test(arm_a, BIN_WIDTH_S, (3, fit_a["breakpoint_bins"]), exclude_lag_bins=3, alternative="two-sided")["test"]
    own_b = segmented_slope_test(arm_b, BIN_WIDTH_S, (3, fit_b["breakpoint_bins"]), exclude_lag_bins=3, alternative="two-sided")["test"]
    own_overlap = own_a["ci_lower"] <= own_b["ci_upper"] and own_b["ci_lower"] <= own_a["ci_upper"]
    assert not own_overlap, "the per-arm-fitted-boundary comparison must find the two arms different"


def test_breakpoint_bootstrap_ci_excludes_geometry_prediction_at_high_session_count():
    """1.3's regression test: at a high session count and low per-session
    noise, the bootstrap interval on each cohort's breakpoint is tight
    enough that a TRUE clock (breakpoint fixed in seconds regardless of
    window length) gives seconds-intervals that overlap and fraction-of-
    window intervals that do not -- excluding the geometry-scaled
    prediction and returning 'stayed_on_the_clock', not 'underpowered'."""
    true_bp_s = 0.8

    def value_fn(lag):
        t = lag * BIN_WIDTH_S
        return 0.02 + 0.09 * (true_bp_s - t) if t <= true_bp_s else 0.02

    series_a = _synthetic_series(list(range(3, 21)), value_fn, n_sessions=200, noise_sem=0.02, seed=501)
    series_b = _synthetic_series(list(range(3, 41)), value_fn, n_sessions=200, noise_sem=0.02, seed=502)

    ci_a = breakpoint_bootstrap_ci(series_a, BIN_WIDTH_S, n_boot=400, rng=np.random.default_rng(1))
    ci_b = breakpoint_bootstrap_ci(series_b, BIN_WIDTH_S, n_boot=400, rng=np.random.default_rng(2))
    assert ci_a["status"] == "tested" and ci_b["status"] == "tested"

    result = geometry_vs_clock_verdict(ci_a, 2.3, ci_b, 4.0)
    assert result["verdict"] == "stayed_on_the_clock"


def test_breakpoint_bootstrap_ci_is_underpowered_at_low_session_count():
    """The same true clock shape as above, but only 6 noisy sessions per
    cohort: the bootstrap interval must be too wide to separate the clock
    and geometry predictions, and the verdict must be
    'underpowered_to_adjudicate' rather than either substantive branch."""
    true_bp_s = 0.8

    def value_fn(lag):
        t = lag * BIN_WIDTH_S
        return 0.02 + 0.09 * (true_bp_s - t) if t <= true_bp_s else 0.02

    series_a = _synthetic_series(list(range(3, 21)), value_fn, n_sessions=6, noise_sem=0.12, seed=601)
    series_b = _synthetic_series(list(range(3, 41)), value_fn, n_sessions=6, noise_sem=0.12, seed=602)

    ci_a = breakpoint_bootstrap_ci(series_a, BIN_WIDTH_S, n_boot=300, rng=np.random.default_rng(3))
    ci_b = breakpoint_bootstrap_ci(series_b, BIN_WIDTH_S, n_boot=300, rng=np.random.default_rng(4))

    result = geometry_vs_clock_verdict(ci_a, 2.3, ci_b, 4.0)
    assert result["verdict"] == "underpowered_to_adjudicate"
