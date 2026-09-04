"""Smoke tests for scripts/run_state_behavior_link.py's core statistics
(the trial-count-matching machinery and the deciding contrast), run against
small synthetic populations rather than the real Panichello .mat files so
they run in well under a second."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_state_behavior_link import (  # noqa: E402
    LAG_RANGE_BINS, cheap_first_look, full_statistic, matched_correct_draws,
)
from state_persistence import simulate_planted_population  # noqa: E402

BIN_WIDTH_S = 0.1
N_BINS = 12  # matches the macaque delay epoch: 300-1450 ms at 100 ms bins


def _synthetic_counts(n_trials: int, n_units: int, seed: int) -> np.ndarray:
    return simulate_planted_population(
        n_trials, n_units, N_BINS, BIN_WIDTH_S, tau_s=0.6, share_static=0.5, share_slow=0.5,
        rate_hz=1.0, rng=np.random.default_rng(seed))


def test_full_statistic_reports_per_lag_components_and_identity_holds():
    counts = _synthetic_counts(n_trials=60, n_units=20, seed=0)
    result = full_statistic(counts, n_splits=4, n_null_replicates=4, n_null_splits_per_replicate=3, seed=1)
    assert result["status"] == "fitted"
    lo, hi = LAG_RANGE_BINS
    assert set(int(lag) for lag in result["per_lag"]) == set(range(lo, hi + 1))
    for record in result["per_lag"].values():
        assert abs(record["r_obs"] - record["r_null"] - record["d_perm"]) < 1e-9
    assert abs(result["identity_check_r_obs_minus_r_null_minus_d_perm"]) < 1e-9


def test_matched_correct_draws_honours_declared_draw_count_and_matches_trial_count():
    rng = np.random.default_rng(2)
    counts_all = _synthetic_counts(n_trials=80, n_units=15, seed=3)
    n_error = 20
    correct_idx = rng.choice(80, size=60, replace=False)
    result = matched_correct_draws(counts_all, correct_idx, n_error, "unit-test-session", n_draws=12)
    assert result["status"] == "fitted"
    assert result["n_draws_requested"] == 12
    assert result["n_draws_fitted"] <= 12
    assert len(result["d_perm_level_all_draws"]) == result["n_draws_fitted"]
    # median-of-draws must lie within the observed spread it is summarising
    lo, hi = result["d_perm_level_iqr"]
    assert lo <= result["d_perm_level_median"] <= hi or result["n_draws_fitted"] < 4


def test_cheap_first_look_reports_all_three_correlates():
    rng = np.random.default_rng(4)
    counts = _synthetic_counts(n_trials=50, n_units=10, seed=5)
    is_corr = rng.integers(0, 2, size=50).astype(bool)
    result = cheap_first_look(counts, is_corr)
    assert result["status"] == "computed"
    for key in ("leading_component_score_gain", "total_spike_count", "trial_index"):
        assert result[key]["status"] == "computed"
        assert -1.0 <= result[key]["r"] <= 1.0


def test_cheap_first_look_declines_on_too_few_trials():
    counts = _synthetic_counts(n_trials=5, n_units=5, seed=6)
    is_corr = np.array([True, False, True, False, True])
    result = cheap_first_look(counts, is_corr)
    assert result["status"] == "not_computable"
