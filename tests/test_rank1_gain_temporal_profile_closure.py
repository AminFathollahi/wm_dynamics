"""Test for scripts/run_rank1_gain_temporal_profile_closure.py's
sign-crossing-conditioned slope test: a synthetic fixture where the planted
decay is IDENTICAL in the crossing-free and crossing sessions, which must
recover a significant negative slope in the crossing-free group alone and
an indistinguishable between-group comparison -- the exact pattern that
rules out a sign-crossing mechanism as necessary for the decay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_rank1_gain_temporal_profile_closure import sign_crossing_conditioned_slope_test  # noqa: E402


def _synthetic_row(session: str, seed: int, slope: float = -0.1) -> dict:
    rng = np.random.default_rng(seed)
    lags = [3, 4, 5, 6, 7, 8]
    r_obs = {str(lag): {"r_median": 0.3 + slope * (lag * 0.1) + rng.normal(scale=0.005)} for lag in lags}
    r_null = {str(lag): {"r_null_median": 0.1 + rng.normal(scale=0.005)} for lag in lags}
    return {
        "session": session, "epoch": "delay", "structure": "pooled", "width_bins": 3,
        "n_trials": 100, "n_units": 30,
        "profile": {"status": "fitted", "lags": r_obs},
        "null_permutation": {"lags": r_null},
    }


def test_identical_decay_in_both_groups_is_not_flagged_as_crossing_dependent():
    lag = {"human_lag_rows": []}
    crossings = {"session_rows": []}
    for i in range(15):
        session = f"no_cross_{i}"
        lag["human_lag_rows"].append(_synthetic_row(session, seed=i))
        crossings["session_rows"].append({"session": session, "status": "tested", "n_sign_crossings": 0})
    for i in range(15):
        session = f"cross_{i}"
        lag["human_lag_rows"].append(_synthetic_row(session, seed=100 + i))
        crossings["session_rows"].append({"session": session, "status": "tested", "n_sign_crossings": 3})

    result = sign_crossing_conditioned_slope_test(lag["human_lag_rows"], crossings["session_rows"])
    assert result["join"]["n_joined_to_a_crossing_flag"] == 30
    assert result["by_group"]["no_crossing"]["n_sessions"] == 15
    assert result["by_group"]["at_least_one_crossing"]["n_sessions"] == 15
    assert result["by_group"]["no_crossing"]["d_perm_slope"]["mean_slope"] < 0
    assert result["by_group"]["no_crossing"]["d_perm_slope"]["p_value"] < 0.05
    assert result["between_group_welch_test"]["d_perm_slope"]["welch_p_value"] > 0.05
    assert result["verdict"] == "sign_crossing_is_not_required_for_the_decay"


def test_too_few_lags_are_excluded_from_the_slope_and_counted():
    lag = {"human_lag_rows": [{
        "session": "short", "epoch": "delay", "structure": "pooled", "width_bins": 3,
        "n_trials": 50, "n_units": 10,
        "profile": {"status": "fitted", "lags": {"3": {"r_median": 0.3}, "4": {"r_median": 0.28}}},
        "null_permutation": {"lags": {"3": {"r_null_median": 0.1}, "4": {"r_null_median": 0.1}}},
    }]}
    crossings = {"session_rows": [{"session": "short", "status": "tested", "n_sign_crossings": 0}]}
    result = sign_crossing_conditioned_slope_test(lag["human_lag_rows"], crossings["session_rows"])
    assert result["join"]["n_joined_but_below_min_lags_for_slope"] == 1
    assert result["by_group"]["no_crossing"]["n_sessions"] == 0


def test_width_bins_selects_only_matching_rows():
    """The width_bins parameter (added to extend this test beyond the
    deciding width) must filter lag rows by width_bins, not silently
    accept every width -- a row planted at width_bins=5 must be picked up
    when width_bins=5 is requested and ignored at width_bins=3."""
    row_w5 = _synthetic_row("only_at_w5", seed=1)
    row_w5["width_bins"] = 5
    lag_rows = [row_w5]
    crossing_rows = [{"session": "only_at_w5", "status": "tested", "n_sign_crossings": 0}]

    at_w3 = sign_crossing_conditioned_slope_test(lag_rows, crossing_rows, width_bins=3)
    assert at_w3["join"]["n_left_rows"] == 0

    at_w5 = sign_crossing_conditioned_slope_test(lag_rows, crossing_rows, width_bins=5)
    assert at_w5["join"]["n_left_rows"] == 1
    assert at_w5["width_bins"] == 5
