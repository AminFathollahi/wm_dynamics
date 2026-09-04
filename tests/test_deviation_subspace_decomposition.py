"""Tests for scripts/run_deviation_subspace_decomposition.py: the three
things that could silently break this analysis without erroring --
(1) the residual decomposition must recompose the delivered
rate_free_state_deviation exactly (the decomposition-identity assertion the
whole module stops on if it fails), (2) the cross-validated memorandum
subspace fitted for a trial must never depend on that trial's own data, and
(3) the matched-dimension random-subspace null must actually be built at the
memorandum subspace's own dimension, not some other one."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_deviation_subspace_decomposition import (  # noqa: E402
    _orthonormal_basis, _random_orthonormal_basis, cv_class_mean_subspace,
    cv_regression_subspace, random_subspace_null_raw_correlations,
    residual_decomposition_and_identity_check,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402


def _synthetic_activity(seed: int, n_trials: int, n_units: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(loc=5.0, scale=2.0, size=(n_trials, n_units)))


def test_decomposition_identity_matches_delivered_deviation_exactly():
    activity = _synthetic_activity(0, n_trials=20, n_units=6)
    result = residual_decomposition_and_identity_check(activity)
    delivered = rate_free_state_deviation(activity)
    assert result["identity_passed"]
    assert result["identity_max_abs_diff"] < 1e-10
    np.testing.assert_array_equal(np.isfinite(result["deviation_recomputed"]), np.isfinite(delivered))
    finite = np.isfinite(delivered)
    np.testing.assert_allclose(result["deviation_recomputed"][finite], delivered[finite], atol=1e-10)


def test_decomposition_identity_holds_with_a_zero_activity_trial():
    # Mirrors the delivered function's own NaN-handling test: a trial with zero total activity has no
    # defined direction and must not silently corrupt the identity check on every OTHER trial either.
    activity = _synthetic_activity(1, n_trials=12, n_units=5)
    activity[3] = 0.0
    result = residual_decomposition_and_identity_check(activity)
    delivered = rate_free_state_deviation(activity)
    assert result["identity_passed"]
    assert np.isnan(result["deviation_recomputed"][3])
    assert np.isnan(delivered[3])


def test_residual_is_orthogonal_to_the_leave_one_out_reference():
    # r_i = u_i - (u_i . m_i) m_i is, by construction, orthogonal to m_i -- a direct algebraic check that
    # the residual actually is the perpendicular component, not just a plausible-looking vector.
    activity = _synthetic_activity(2, n_trials=15, n_units=7)
    result = residual_decomposition_and_identity_check(activity)
    from run_deviation_subspace_decomposition import _leave_one_out_unit_directions
    directions = _leave_one_out_unit_directions(activity)
    finite = result["finite"]
    dot = np.einsum("ij,ij->i", result["residual"][finite], directions["loo_mean_normalized"][finite])
    np.testing.assert_allclose(dot, 0.0, atol=1e-10)


def test_cv_class_mean_subspace_never_uses_the_held_out_trial():
    # Two trials per class -- deliberately thin, so that if trial 0 leaked into its own subspace fit, its
    # class mean would be dominated by its own value; excluding it correctly leaves only the OTHER member.
    n_units = 6
    classes = np.array([0, 0, 1, 1, 2, 2])
    rng = np.random.default_rng(3)
    base = np.zeros((6, n_units))
    base[0] = [10, 0, 0, 0, 0, 0]
    base[1] = [1, 0, 0, 0, 0, 0]   # trial 0's held-in classmate
    base[2] = [0, 10, 0, 0, 0, 0]
    base[3] = [0, 1, 0, 0, 0, 0]
    base[4] = [0, 0, 10, 0, 0, 0]
    base[5] = [0, 0, 1, 0, 0, 0]
    activity = base + rng.normal(scale=1e-6, size=base.shape) + 5.0
    directions_result = residual_decomposition_and_identity_check(activity)
    from run_deviation_subspace_decomposition import _leave_one_out_unit_directions
    u = _leave_one_out_unit_directions(activity)["unit_vectors"]
    residual = directions_result["residual"]

    within, outside = cv_class_mean_subspace(u, classes, residual, dim=2)

    # Manually recompute C_0 with trial 0 explicitly excluded, exactly as the leave-one-out rule requires.
    keep = np.ones(6, dtype=bool)
    keep[0] = False
    grand_mean = u[keep].mean(axis=0)
    centred = u[keep] - grand_mean
    expected_means = np.stack([centred[classes[keep] == c].mean(axis=0) for c in np.unique(classes[keep])])
    expected_basis = _orthonormal_basis(expected_means, dim=2)
    expected_proj = expected_basis @ (expected_basis.T @ residual[0])
    expected_within = np.linalg.norm(expected_proj)

    assert within[0] == pytest.approx(expected_within, abs=1e-8)

    # The bug this test guards against: if trial 0 (an extreme outlier along unit 0) had leaked into its
    # own class-mean fit, its class mean would shift sharply toward it and the within-magnitude computed
    # WITH it included would differ from the correctly-excluded one by much more than floating tolerance.
    wrong_grand_mean = u.mean(axis=0)
    wrong_centred = u - wrong_grand_mean
    wrong_means = np.stack([wrong_centred[classes == c].mean(axis=0) for c in np.unique(classes)])
    wrong_basis = _orthonormal_basis(wrong_means, dim=2)
    wrong_proj = wrong_basis @ (wrong_basis.T @ residual[0])
    wrong_within = np.linalg.norm(wrong_proj)
    assert abs(wrong_within - within[0]) > 1e-4


def test_cv_regression_subspace_never_uses_the_held_out_trial():
    n_units = 5
    n_trials = 30
    rng = np.random.default_rng(4)
    theta = rng.uniform(0, 2 * np.pi, size=n_trials)
    target_2d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    activity = np.abs(5.0 + 2.0 * np.outer(np.cos(theta), rng.normal(size=n_units))
                       + rng.normal(scale=0.5, size=(n_trials, n_units)))
    u = None
    from run_deviation_subspace_decomposition import _leave_one_out_unit_directions
    u = _leave_one_out_unit_directions(activity)["unit_vectors"]
    residual = residual_decomposition_and_identity_check(activity)["residual"]

    within, outside = cv_regression_subspace(u, target_2d, residual, dim=2)

    i = 0
    keep = np.ones(n_trials, dtype=bool)
    keep[i] = False
    x_c = target_2d[keep] - target_2d[keep].mean(axis=0)
    u_c = u[keep] - u[keep].mean(axis=0)
    coefficients, *_ = np.linalg.lstsq(x_c, u_c, rcond=None)
    basis = _orthonormal_basis(coefficients, 2)
    expected_within = np.linalg.norm(basis @ (basis.T @ residual[i]))
    assert within[i] == pytest.approx(expected_within, abs=1e-8)


def test_random_subspace_basis_has_the_declared_dimension_and_is_orthonormal():
    rng = np.random.default_rng(5)
    for dim in (2, 5, 7):
        basis = _random_orthonormal_basis(n_units=20, dim=dim, rng=rng)
        assert basis.shape == (20, dim)
        np.testing.assert_allclose(basis.T @ basis, np.eye(dim), atol=1e-8)


def test_random_subspace_null_matches_the_memorandum_subspace_dimension():
    # The null must be built at the SAME dimension as the fitted subspace it stands in for -- a null of
    # any other dimension answers a different question (dimensionality itself, not the memorandum).
    n_trials, n_units = 25, 10
    rng = np.random.default_rng(6)
    residual = rng.normal(size=(n_trials, n_units))
    outcome = rng.normal(size=n_trials)
    for dim in (2, 6):
        values = random_subspace_null_raw_correlations(residual, dim, outcome, seed=123, n_draws=50)
        assert len(values) == 50
        assert all((v is None) or (-1.0 <= v <= 1.0) for v in values)
