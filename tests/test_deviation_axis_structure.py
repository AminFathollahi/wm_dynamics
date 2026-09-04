"""Tests for scripts/run_deviation_axis_structure.py's occupied-state-space classifier: the
observed off-occupied fraction must be judged against its own matched-dimension random-axis null,
never against zero, and the three named branches must fire on planted synthetic cases -- an axis
lying inside the occupied subspace (off-fraction far below its null) lands in the within branch,
an axis lying outside it (off-fraction far above its null) lands in the outside branch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_deviation_axis_structure import (  # noqa: E402
    _occupied_space_decomposition, classify_occupied_space_branch,
    pooled_off_fraction_against_matched_null,
)

WITHIN_BRANCH = "the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace"
OUTSIDE_BRANCH = "the_axis_lies_outside_the_occupied_state_space"
NOT_SEPARABLE_BRANCH = "not_separable_at_the_available_dimensionality"


def _synthetic_occupied_space(seed: int = 7, n_trials: int = 200, n_units: int = 30):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_trials, n_units))
    _u, _s, vt = np.linalg.svd(U - U.mean(axis=0), full_matrices=False)
    return U, vt[0], vt[-1]


def test_planted_axis_inside_the_occupied_subspace_sits_below_its_matched_null():
    U, axis_within, _axis_out = _synthetic_occupied_space()
    result = _occupied_space_decomposition(U, axis_within, 25, 200, "tests|axis|within")
    assert result["status"] == "computed"
    assert result["within_fraction"] > 0.999
    assert result["off_fraction"] < result["null_off_fraction_mean"]
    assert result["off_fraction_above_null"] is False


def test_planted_axis_outside_the_occupied_subspace_sits_above_its_matched_null():
    U, _axis_within, axis_out = _synthetic_occupied_space()
    result = _occupied_space_decomposition(U, axis_out, 25, 200, "tests|axis|outside")
    assert result["status"] == "computed"
    assert result["off_fraction"] > 0.99
    assert result["off_fraction"] > result["null_off_fraction_mean"]
    assert result["two_sided_p_value"] <= 0.05
    assert result["off_fraction_above_null"] is True


def test_classifier_fires_within_when_off_fraction_is_below_the_null_central_mass():
    U, axis_within, _axis_out = _synthetic_occupied_space()
    result = _occupied_space_decomposition(U, axis_within, 25, 200, "tests|axis|classify-within")
    sessions = [{"off_fraction": result["off_fraction"], "null_draws": result["null_off_fraction_draws"]}]
    comparison = pooled_off_fraction_against_matched_null(sessions)
    assert comparison["status"] == "computed"
    assert comparison["significant_above_null"] is False
    assert classify_occupied_space_branch(comparison) == WITHIN_BRANCH


def test_classifier_fires_outside_when_off_fraction_is_above_the_null_central_mass():
    U, _axis_within, axis_out = _synthetic_occupied_space()
    result = _occupied_space_decomposition(U, axis_out, 25, 200, "tests|axis|classify-outside")
    sessions = [{"off_fraction": result["off_fraction"], "null_draws": result["null_off_fraction_draws"]}]
    comparison = pooled_off_fraction_against_matched_null(sessions)
    assert comparison["status"] == "computed"
    assert comparison["significant_above_null"] is True
    assert comparison["above_null"] is True
    assert classify_occupied_space_branch(comparison) == OUTSIDE_BRANCH


def test_pooled_comparison_means_sessions_then_levels_and_carries_effect_sizes():
    U, axis_within, axis_out = _synthetic_occupied_space()
    res_in = _occupied_space_decomposition(U, axis_within, 25, 200, "tests|axis|pool-in")
    res_out = _occupied_space_decomposition(U, axis_out, 25, 200, "tests|axis|pool-out")
    # two sessions whose observed values bracket their own null centre: the pooled observed value
    # must be the plain mean across sessions, not a test against zero
    sessions = [
        {"off_fraction": res_in["off_fraction"], "null_draws": res_in["null_off_fraction_draws"]},
        {"off_fraction": res_out["off_fraction"], "null_draws": res_out["null_off_fraction_draws"]},
    ]
    comparison = pooled_off_fraction_against_matched_null(sessions)
    assert comparison["status"] == "computed"
    expected_observed = (res_in["off_fraction"] + res_out["off_fraction"]) / 2.0
    expected_null = float(np.mean(np.concatenate([res_in["null_off_fraction_draws"],
                                                  res_out["null_off_fraction_draws"]])))
    assert abs(comparison["pooled_observed_off_fraction"] - expected_observed) < 1e-12
    assert abs(comparison["pooled_null_mean"] - expected_null) < 1e-12
    assert comparison["pooled_observed_off_fraction"] != 0.0


def test_uncomputable_comparison_lands_in_the_not_separable_branch():
    assert classify_occupied_space_branch({"status": "not_computable"}) == NOT_SEPARABLE_BRANCH
    empty = pooled_off_fraction_against_matched_null([])
    assert empty["status"] == "not_computable"
    assert classify_occupied_space_branch(empty) == NOT_SEPARABLE_BRANCH
