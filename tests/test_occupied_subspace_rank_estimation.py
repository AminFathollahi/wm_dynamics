"""Tests for scripts/run_occupied_subspace_rank_estimation.py.

The single most important test here pins the property the delivered `_cv_pca_rank` selector lacks: a
genuine interior minimum. `entry_holdout_rank` must recover a known low true rank from synthetic data
(never trivially return the ambient ceiling the way the degenerate selector does), and its held-out-entry
error curve must not be monotone non-increasing -- if it were, it would share the same defect this module
exists to repair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_deviation_axis_structure import _cv_pca_rank, _occupied_space_decomposition  # noqa: E402
from run_occupied_subspace_rank_estimation import (  # noqa: E402
    _gate_status_for_corpus, _within_subspace_null_decomposition, classify_final_branch, entry_holdout_rank,
    permutation_eigenvalue_rank, _whole_session_cluster_bootstrap_mdd,
)


def _low_rank_synthetic(n: int, p: int, true_rank: int, noise_relative: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, true_rank))
    loadings = rng.standard_normal((true_rank, p))
    signal = latent @ loadings
    return signal + rng.standard_normal((n, p)) * noise_relative * float(np.std(signal))


def test_entry_holdout_rank_recovers_known_low_rank_synthetic():
    """The mandatory positive control: a genuine rank-3 signal in a 200x40 matrix at low noise must be
    recovered within a small tolerance -- not pushed to the ambient ceiling (40) or to 1."""
    U = _low_rank_synthetic(n=200, p=40, true_rank=3, noise_relative=0.1, seed=0)
    result = entry_holdout_rank(U, "test|entry_holdout|low_rank")
    assert result["status"] == "computed"
    assert abs(result["best_k"] - 3) <= 2
    assert result["best_k"] < min(40, 200 - 1)  # never the degenerate top-of-range answer


def test_entry_holdout_rank_is_not_monotonic_unlike_the_degenerate_selector():
    """Contrast with the defect this module replaces: the degenerate `_cv_pca_rank` selector's held-out
    error is monotone non-increasing by construction and always selects the top of its candidate range.
    entry_holdout_rank's held-out-ENTRY error must NOT be monotone non-increasing -- inverting small
    singular values past the true rank must make the held-out prediction worse, not better or flat."""
    U = _low_rank_synthetic(n=200, p=40, true_rank=3, noise_relative=0.1, seed=0)
    eh = entry_holdout_rank(U, "test|entry_holdout|monotonicity")
    errors = np.asarray(eh["held_out_entry_reconstruction_error"])
    assert eh["status"] == "computed"
    assert np.any(np.diff(errors) > 0), "held-out-entry error never rises -- same degeneracy as the old selector"

    max_k = min(40, 200 - 1)
    old = _cv_pca_rank(U, max_k, "test|old_selector|monotonicity")
    old_errors = np.asarray(old["cv_reconstruction_error"])
    assert np.all(np.diff(old_errors) <= 1e-9), "sanity: the imported old selector should still be monotone"
    assert old["best_k"] == max_k, "sanity: the imported old selector should still return the ambient ceiling"


def test_entry_holdout_rank_not_estimable_when_blocks_too_small():
    """With only 2 rows and 2 columns, a 5x5 row/column fold split leaves every training block with fewer
    than 2 usable rows or columns after holding a fold out -- no candidate rank >= 1 is testable."""
    U = np.random.default_rng(1).standard_normal((2, 2))
    result = entry_holdout_rank(U, "test|entry_holdout|too_small", max_candidate_k=25)
    assert result["status"] == "not_estimable"


def test_entry_holdout_rank_candidate_range_shrinks_with_small_blocks_but_still_computes():
    """A modest matrix (12 rows x 8 columns) does not collapse to not_estimable -- its candidate range is
    simply capped below 25 by the smallest training block across the 25 row/column fold combinations."""
    U = np.random.default_rng(1).standard_normal((12, 8))
    result = entry_holdout_rank(U, "test|entry_holdout|small_but_computable", max_candidate_k=25)
    assert result["status"] == "computed"
    assert result["candidate_ks"][-1] < 25


def test_entry_holdout_rank_selects_zero_for_pure_noise():
    U = np.random.default_rng(7).standard_normal((200, 40))
    result = entry_holdout_rank(U, "test|entry_holdout|rank_zero")
    assert result["status"] == "computed"
    assert result["best_k"] == 0


def test_entry_holdout_rank_does_not_force_a_rank_at_its_search_ceiling():
    U = _low_rank_synthetic(n=150, p=30, true_rank=3, noise_relative=0.0, seed=8)
    result = entry_holdout_rank(U, "test|entry_holdout|search_ceiling", max_candidate_k=1)
    assert result["status"] == "nonidentified_at_search_ceiling"
    assert result["best_k"] is None
    assert result["selected_k_at_search_ceiling"] == 1


def test_permutation_eigenvalue_rank_recovers_known_low_rank_synthetic():
    U = _low_rank_synthetic(n=200, p=40, true_rank=3, noise_relative=0.1, seed=2)
    result = permutation_eigenvalue_rank(U, "test|perm_eig|low_rank", n_perm=100)
    assert result["status"] == "computed"
    assert abs(result["best_k"] - 3) <= 2


def test_permutation_eigenvalue_rank_on_pure_noise_returns_a_small_rank():
    """Negative control: pure noise (no low-dimensional structure at all) must not be called high rank --
    the permutation null should swallow almost every observed eigenvalue."""
    rng = np.random.default_rng(3)
    U = rng.standard_normal((200, 40))
    result = permutation_eigenvalue_rank(U, "test|perm_eig|pure_noise", n_perm=100)
    assert result["status"] == "computed"
    assert result["best_k"] <= 3


def test_within_subspace_null_matches_ambient_decomposition_off_fraction():
    """The harder null recomputes the basis independently (deterministic SVD, no randomness in the basis
    itself) so its observed within/off fraction must be bit-identical to the delivered ambient-null
    decomposition's own within/off fraction for the same (U, axis, k)."""
    rng = np.random.default_rng(4)
    U = _low_rank_synthetic(n=150, p=25, true_rank=4, noise_relative=0.2, seed=4)
    axis = rng.standard_normal(25)
    axis /= np.linalg.norm(axis)
    k = 4
    ambient = _occupied_space_decomposition(U, axis, k, 50, "test|ambient")
    harder = _within_subspace_null_decomposition(U, axis, k, 50, "test|harder")
    assert ambient["status"] == "computed" and harder["status"] == "computed"
    assert abs(ambient["off_fraction"] - harder["off_fraction"]) < 1e-9


def test_within_subspace_null_is_degenerate_near_zero():
    """By construction, a random direction confined to the estimated occupied subspace has off-fraction
    ~0 relative to that same subspace, up to floating-point round-off -- the null this test pins is a
    floor, not a classical hypothesis test."""
    rng = np.random.default_rng(5)
    U = _low_rank_synthetic(n=150, p=25, true_rank=4, noise_relative=0.2, seed=5)
    axis = rng.standard_normal(25)
    axis /= np.linalg.norm(axis)
    harder = _within_subspace_null_decomposition(U, axis, 4, 200, "test|harder|degenerate")
    assert harder["status"] == "computed"
    assert harder["null_off_fraction_mean"] < 1e-6
    assert harder["null_off_fraction_sd"] < 1e-6


def test_whole_session_cluster_bootstrap_mdd_is_positive_and_scales_down_with_more_sessions():
    rng = np.random.default_rng(6)
    few = list(rng.normal(0.01, 0.02, size=6))
    many = list(rng.normal(0.01, 0.02, size=60))
    mdd_few = _whole_session_cluster_bootstrap_mdd(few, "test|mdd|few")
    mdd_many = _whole_session_cluster_bootstrap_mdd(many, "test|mdd|many")
    assert mdd_few["status"] == "computed" and mdd_many["status"] == "computed"
    assert mdd_few["minimum_detectable_off_fraction_difference_at_80pct_power"] > 0
    assert (mdd_many["minimum_detectable_off_fraction_difference_at_80pct_power"]
            < mdd_few["minimum_detectable_off_fraction_difference_at_80pct_power"])


def test_whole_session_cluster_bootstrap_mdd_not_computable_below_four_sessions():
    result = _whole_session_cluster_bootstrap_mdd([0.1, 0.2, 0.3], "test|mdd|too_few")
    assert result["status"] == "not_computable"


def _fake_gate(min_p_a: int | None, min_p_b: int | None, passes_a: bool = True, passes_b: bool = True) -> dict:
    return {"gate_by_estimator": {
        "entry_holdout_bicross_validation": {"passes_gate_overall": passes_a, "minimum_validated_ambient_unit_count": min_p_a},
        "permutation_eigenvalue_threshold": {"passes_gate_overall": passes_b, "minimum_validated_ambient_unit_count": min_p_b},
    }}


def _fake_corpus_result(median_units: float, amb_sig_above: dict, hard_sig_above: dict) -> dict:
    def _comparison(sig_above):
        return {"status": "computed", "significant_above_null": sig_above}
    return {
        "median_ambient_unit_count_across_cells": median_units,
        "per_session": [{"levels": [{
            est: {"status": "computed", "best_k": 3, "best_k_is_interior": True}
            for est in ("entry_holdout_bicross_validation", "permutation_eigenvalue_threshold")
        }]}],
        "pooled_by_estimator_and_null": {
            est: {
                "ambient_null": {"pooled_observed_vs_matched_null": _comparison(amb_sig_above[est])},
                "harder_null": {"pooled_observed_vs_matched_null": _comparison(hard_sig_above[est])},
            }
            for est in ("entry_holdout_bicross_validation", "permutation_eigenvalue_threshold")
        },
    }


def test_classify_final_branch_not_determinable_when_gate_fails_at_corpus_scale():
    gate = _fake_gate(min_p_a=50, min_p_b=200)  # corpus median (30) below both
    corpus_result = _fake_corpus_result(30, {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False},
                                         {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False})
    gate_status = _gate_status_for_corpus(corpus_result, gate)
    assert classify_final_branch(corpus_result, gate_status) == "not_determinable_because_an_estimator_failed_its_synthetic_gate"


def test_classify_final_branch_closer_than_ambient_under_both_estimators():
    gate = _fake_gate(min_p_a=10, min_p_b=10)
    corpus_result = _fake_corpus_result(100, {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False},
                                         {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False})
    gate_status = _gate_status_for_corpus(corpus_result, gate)
    assert classify_final_branch(corpus_result, gate_status) == (
        "the_axis_is_closer_to_the_occupied_space_than_ambient_directions_under_both_estimators")


def test_classify_final_branch_closer_than_ambient_under_one_estimator_only():
    gate = _fake_gate(min_p_a=10, min_p_b=10)
    corpus_result = _fake_corpus_result(100, {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": True},
                                         {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False})
    gate_status = _gate_status_for_corpus(corpus_result, gate)
    assert classify_final_branch(corpus_result, gate_status) == (
        "the_axis_is_closer_to_the_occupied_space_than_ambient_directions_under_one_estimator_only")


def test_classify_final_branch_not_closer_than_ambient_under_either_estimator():
    gate = _fake_gate(min_p_a=10, min_p_b=10)
    corpus_result = _fake_corpus_result(100, {"entry_holdout_bicross_validation": True, "permutation_eigenvalue_threshold": True},
                                         {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False})
    gate_status = _gate_status_for_corpus(corpus_result, gate)
    assert classify_final_branch(corpus_result, gate_status) == (
        "the_axis_is_not_closer_to_the_occupied_space_than_ambient_directions_under_either_estimator")


def test_classify_final_branch_stops_when_rank_is_not_identified():
    gate = _fake_gate(min_p_a=10, min_p_b=10)
    corpus_result = _fake_corpus_result(
        100,
        {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False},
        {"entry_holdout_bicross_validation": False, "permutation_eigenvalue_threshold": False},
    )
    corpus_result["per_session"][0]["levels"][0]["permutation_eigenvalue_threshold"] = {
        "status": "zero_rank_no_occupied_subspace_to_test", "best_k": 0,
    }
    gate_status = _gate_status_for_corpus(corpus_result, gate)
    assert classify_final_branch(corpus_result, gate_status) == (
        "not_determinable_because_rank_selection_is_nonidentified_or_disagrees")
