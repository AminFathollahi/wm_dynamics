"""Tests for run_occupied_subspace_membership_split_half_circularity_check.py.

Covers: the verdict rule's five pre-declared outcomes; the column-shuffle control preserves each unit's
own marginal distribution while destroying row (trial) identity; and the two calibrating cases the split-
half design exists to distinguish -- genuine, split-independent low-rank structure (split-half contrast
should be large and clearly above the no-structure control) versus pure noise (split-half contrast should
collapse to approximately zero, matching the pre-declared 'withdrawn as unmeasured' condition).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_occupied_subspace_membership_split_half_circularity_check as mod  # noqa: E402


def test_verdict_not_determinable_when_contrast_not_computable():
    assert mod._verdict({"status": "not_computable"}, {"status": "not_computable"}) == \
        "not_determinable_insufficient_sessions_with_a_valid_split"


def test_verdict_below_chance():
    contrast = {"status": "computed", "cluster_bootstrap_ci_95pct": [-0.3, -0.1]}
    assert mod._verdict(contrast, {"status": "not_computable"}) == "split_half_contrast_is_below_chance"


def test_verdict_withdrawn_when_ci_includes_zero():
    contrast = {"status": "computed", "cluster_bootstrap_ci_95pct": [-0.05, 0.2]}
    assert mod._verdict(contrast, {"status": "not_computable"}) == \
        "the_reversal_is_withdrawn_as_unmeasured_under_split_half_estimation"


def test_verdict_inconclusive_when_not_separated_from_control():
    contrast = {"status": "computed", "cluster_bootstrap_ci_95pct": [0.05, 0.3]}
    control = {"status": "computed", "cluster_bootstrap_ci_95pct": [0.02, 0.1]}  # control upper (0.1) >= 0.05
    assert mod._verdict(contrast, control) == \
        "inconclusive_split_half_effect_not_clearly_separated_from_the_no_structure_control"


def test_verdict_survives_when_clearly_above_control():
    contrast = {"status": "computed", "cluster_bootstrap_ci_95pct": [0.6, 0.9]}
    control = {"status": "computed", "cluster_bootstrap_ci_95pct": [0.05, 0.2]}
    assert mod._verdict(contrast, control) == "the_reversal_survives_the_split_half_circularity_check"


def test_verdict_survives_when_control_not_computable():
    contrast = {"status": "computed", "cluster_bootstrap_ci_95pct": [0.6, 0.9]}
    assert mod._verdict(contrast, {"status": "not_computable"}) == \
        "the_reversal_survives_the_split_half_circularity_check"


def test_column_shuffle_preserves_marginals_destroys_row_identity():
    rng = np.random.default_rng(0)
    activity = rng.standard_normal((50, 6))
    shuffled = mod._column_shuffled(activity, np.random.default_rng(1))
    for j in range(activity.shape[1]):
        assert np.array_equal(np.sort(activity[:, j]), np.sort(shuffled[:, j]))
    assert not np.array_equal(activity, shuffled)


def test_split_half_detects_genuine_shared_low_rank_structure_above_control():
    rng = np.random.default_rng(2)
    n, p, k_true = 300, 50, 4
    latent = rng.standard_normal((n, k_true))
    loadings = rng.standard_normal((k_true, p))
    activity = np.abs(latent @ loadings + rng.standard_normal((n, p)) * 0.3) + 0.01

    rows_full = mod._residual_rows(activity)
    kept_idx = np.flatnonzero(rows_full["keep"])
    out = mod._split_half_cell(activity, kept_idx, "test|genuine_structure")

    assert out["n_splits_valid"] >= 15  # most of the 20 splits should be usable
    assert out["mean_contrast"] > 0.3
    # the real, held-out axis must sit further inside the held-out basis than the no-structure control does
    assert out["mean_contrast"] > out["mean_control_contrast"]


def test_split_half_collapses_toward_zero_on_pure_noise():
    rng = np.random.default_rng(3)
    n, p = 300, 50
    activity = np.abs(rng.standard_normal((n, p))) + 0.01  # no shared structure across trials at all

    rows_full = mod._residual_rows(activity)
    kept_idx = np.flatnonzero(rows_full["keep"])
    out = mod._split_half_cell(activity, kept_idx, "test|pure_noise")

    assert abs(out["mean_contrast"]) < 0.1
    assert abs(out["mean_control_contrast"]) < 0.1
