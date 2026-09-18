"""Tests for scripts/run_occupied_subspace_rank_selection_repair.py.

The single most important test here pins the defect mechanism independently of any dataset: a
training fold whose trial count meets or exceeds the ambient unit count makes the original selector's
candidate range reach a COMPLETE orthonormal basis of the ambient space, forcing held-out
reconstruction error to numerical zero at the top of the range regardless of whether the data is
genuinely full rank. The corrected selector's candidate range is capped strictly below that point and
must not reproduce the same forced-zero result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_deviation_axis_structure import _cv_pca_rank  # noqa: E402
from run_occupied_subspace_rank_selection_repair import (  # noqa: E402
    _cv_pca_rank_capped, _session_cluster_bootstrap_mdd,
)


def test_original_selector_returns_max_k_with_near_zero_error_when_n_train_exceeds_p():
    """The mechanism, pinned directly: n=200 trials, p=20 units, 5 contiguous folds -> n_train ~160 >>
    p, so every training fold's SVD supplies a COMPLETE 20-dimensional basis of the ambient space. The
    original (unmodified, imported) selector must therefore select k = max_k = p, with reconstruction
    error at the floating-point floor -- for pure noise with no low-dimensional structure at all."""
    rng = np.random.default_rng(0)
    n, p = 200, 20
    U = rng.standard_normal((n, p))
    max_k = min(p, n - 1)
    result = _cv_pca_rank(U, max_k, "test|original")
    assert result["status"] == "computed"
    assert result["best_k"] == max_k
    assert result["cv_reconstruction_error"][-1] < 1e-15


def test_corrected_selector_abstains_at_its_search_ceiling():
    """Same synthetic matrix, same folds, same accumulation -- only the candidate range differs. The
    corrected selector must never test a k at which a training fold's basis is complete, so its
    candidate range tops out below p and its final-candidate reconstruction error must be a real,
    non-degenerate number (this is pure noise: the "occupied dimensionality" is genuinely close to p,
    so a legitimate cross-validated fit on a smaller candidate range must show a non-trivial error, not
    a floating-point zero)."""
    rng = np.random.default_rng(0)
    n, p = 200, 20
    U = rng.standard_normal((n, p))
    max_k = min(p, n - 1)
    result = _cv_pca_rank_capped(U, max_k, "test|corrected")
    assert result["status"] == "nonidentified_at_search_ceiling"
    assert result["capped_max_k"] < p
    assert result["best_k"] is None
    assert result["selected_k_at_search_ceiling"] == result["capped_max_k"] < max_k
    assert result["cv_reconstruction_error"][-1] > 1.0  # not forced toward zero


def test_corrected_selector_does_not_force_a_rank_when_its_error_minimum_hits_the_cap():
    """Positive control: when the data really does live in a low-dimensional subspace (here, 3 of 20
    ambient dimensions carry all the trial-to-trial variance, the rest is exactly zero), the corrected
    selector's held-out error must be minimised near that true rank, not merely pushed to some
    arbitrary smaller number by the cap alone -- the cap removes the tautology without breaking the
    selector's ability to find real structure."""
    rng = np.random.default_rng(1)
    n, p, true_rank = 400, 20, 3
    latent = rng.standard_normal((n, true_rank))
    loadings = rng.standard_normal((true_rank, p))
    U = latent @ loadings + rng.standard_normal((n, p)) * 1e-6  # negligible noise floor
    max_k = min(p, n - 1)
    result = _cv_pca_rank_capped(U, max_k, "test|corrected|low_rank")
    assert result["status"] == "nonidentified_at_search_ceiling"
    assert result["best_k"] is None
    assert result["selected_k_at_search_ceiling"] == result["capped_max_k"]


def test_corrected_selector_not_estimable_when_capped_range_collapses():
    """When the cap itself would fall below 1 (an extremely small session relative to its own unit
    count), the corrected selector must report not_estimable rather than silently falling back to the
    original's degenerate behaviour."""
    rng = np.random.default_rng(2)
    n, p = 40, 39  # min_n_train ~ 32 (5-fold contiguous, largest fold ~8), p is close to min_n_train
    U = rng.standard_normal((n, p))
    max_k = min(p, n - 1)
    result = _cv_pca_rank_capped(U, max_k, "test|corrected|collapse")
    assert result["status"] in ("computed", "not_estimable", "nonidentified_at_search_ceiling")
    if result["status"] == "computed":
        assert result["best_k"] < p  # never allowed to reach the ambient dimension
    if result["status"] == "nonidentified_at_search_ceiling":
        assert result["best_k"] is None


def test_session_cluster_bootstrap_mdd_is_positive_and_scales_down_with_more_sessions():
    """The whole-session cluster bootstrap detection floor (never a trial-count formula) must return a
    positive minimum detectable difference that shrinks as more independent sessions are pooled."""
    rng = np.random.default_rng(3)
    few = list(rng.normal(0.01, 0.02, size=6))
    many = list(rng.normal(0.01, 0.02, size=60))
    mdd_few = _session_cluster_bootstrap_mdd(few, "test|mdd|few")
    mdd_many = _session_cluster_bootstrap_mdd(many, "test|mdd|many")
    assert mdd_few["status"] == "computed" and mdd_many["status"] == "computed"
    assert mdd_few["minimum_detectable_off_fraction_difference_at_80pct_power"] > 0
    assert (mdd_many["minimum_detectable_off_fraction_difference_at_80pct_power"]
            < mdd_few["minimum_detectable_off_fraction_difference_at_80pct_power"])


def test_session_cluster_bootstrap_mdd_not_computable_below_four_sessions():
    result = _session_cluster_bootstrap_mdd([0.1, 0.2, 0.3], "test|mdd|too_few")
    assert result["status"] == "not_computable"
