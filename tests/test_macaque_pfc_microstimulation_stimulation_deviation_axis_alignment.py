"""Tests for scripts/run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.py -- the pieces that could
silently break this analysis without erroring: (1) a stimulation-induced displacement planted along a
session's own residual axis is recovered by the alignment statistic, (2) one planted orthogonal to that
axis is not, (3) the axis-estimation entry point refuses anything other than control-only trials, and
(4) the bias-only voiding rule fires on a synthetic pooled result built to reproduce the real result's
significance and direction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment import (  # noqa: E402
    _bias_only_voids, _classify_arm, displacement_vector, estimate_axis,
)


def _synthetic_control(rng: np.random.Generator, n_trials: int, n_units: int, axis_direction: np.ndarray,
                        base: np.ndarray, noise_sd: float = 1.0, base_scale: float = 8.0) -> np.ndarray:
    """Every trial is the SAME fixed base vector plus noise confined to ONE direction (axis_direction),
    so the residual covariance's leading eigenvector recovers that direction to high fidelity -- a
    session with a real, single-axis deviation structure, planted rather than fitted from noise. `base`
    is passed in (not sampled here) so control and stimulated trials share the identical mean direction
    and only the planted shift tells them apart."""
    coefs = rng.standard_normal(n_trials) * noise_sd
    return base[None, :] * base_scale + coefs[:, None] * axis_direction[None, :]


def _orthonormal_pair(rng: np.random.Generator, n_units: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.standard_normal(n_units)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(n_units)
    b -= np.dot(b, a) * a
    b /= np.linalg.norm(b)
    return a, b


# ---------------------------------------------------------------------------------------------------
# Planted displacement recovery
# ---------------------------------------------------------------------------------------------------

def test_along_axis_displacement_is_recovered():
    rng = np.random.default_rng(0)
    n_units = 24
    axis_direction, _ = _orthonormal_pair(rng, n_units)
    base = rng.standard_normal(n_units)
    base /= np.linalg.norm(base)
    control = _synthetic_control(rng, 200, n_units, axis_direction, base)

    axis_fit = estimate_axis(control, np.arange(control.shape[0], dtype=float), source="control_only",
                              detrend=False, seed_tag="test|along|stability")
    assert axis_fit["status"] == "computed"
    fitted_axis = axis_fit["axis"]
    assert abs(float(np.dot(fitted_axis, axis_direction))) > 0.9

    # Stimulated trials share the SAME base direction and are shifted along the SAME planted axis.
    stim = _synthetic_control(rng, 150, n_units, axis_direction, base) + 3.0 * axis_direction[None, :]
    disp = displacement_vector(control, stim)
    assert disp is not None and disp["direction_unit"] is not None
    cosine_to_fitted_axis = abs(float(np.dot(disp["direction_unit"], fitted_axis)))
    assert cosine_to_fitted_axis > 0.85


def test_orthogonal_displacement_is_rejected():
    rng = np.random.default_rng(1)
    n_units = 24
    axis_direction, orthogonal_direction = _orthonormal_pair(rng, n_units)
    base = rng.standard_normal(n_units)
    base -= np.dot(base, axis_direction) * axis_direction
    base -= np.dot(base, orthogonal_direction) * orthogonal_direction
    base /= np.linalg.norm(base)
    control = _synthetic_control(rng, 200, n_units, axis_direction, base)

    axis_fit = estimate_axis(control, np.arange(control.shape[0], dtype=float), source="control_only",
                              detrend=False, seed_tag="test|orthogonal|stability")
    assert axis_fit["status"] == "computed"
    fitted_axis = axis_fit["axis"]

    # Stimulated trials share the SAME base direction and are shifted along a direction ORTHOGONAL to
    # the planted axis.
    stim = _synthetic_control(rng, 150, n_units, axis_direction, base) + 3.0 * orthogonal_direction[None, :]
    disp = displacement_vector(control, stim)
    assert disp is not None and disp["direction_unit"] is not None
    cosine_to_fitted_axis = abs(float(np.dot(disp["direction_unit"], fitted_axis)))
    assert cosine_to_fitted_axis < 0.3


# ---------------------------------------------------------------------------------------------------
# Circularity guard
# ---------------------------------------------------------------------------------------------------

def test_circularity_guard_rejects_anything_but_control_only():
    rng = np.random.default_rng(2)
    n_units = 12
    axis_direction, _ = _orthonormal_pair(rng, n_units)
    base = rng.standard_normal(n_units)
    base /= np.linalg.norm(base)
    activity = _synthetic_control(rng, 60, n_units, axis_direction, base)
    trial_index = np.arange(activity.shape[0], dtype=float)

    # The correct, only-permitted call succeeds.
    ok = estimate_axis(activity, trial_index, source="control_only", detrend=False, seed_tag="test|guard|ok")
    assert ok["status"] == "computed"

    # Any other source string -- standing in for a stimulated trial reaching this fit -- fails loudly.
    with pytest.raises(ValueError, match="control_only"):
        estimate_axis(activity, trial_index, source="includes_stimulated_trials", detrend=False,
                       seed_tag="test|guard|bad")


# ---------------------------------------------------------------------------------------------------
# Bias-only voiding rule
# ---------------------------------------------------------------------------------------------------

def _pooled_rotation_result(mean_value: float, mdd: float, significant: bool, below_null: bool) -> dict:
    return {
        "n_sessions": 9,
        "real_pooled": {"status": "tested", "mean_value": mean_value},
        "minimum_detectable_difference_80pct_power": {"status": "computed", "mdd": mdd},
        "pooled_null_mean": 0.09, "pooled_null_sd": 0.02,
        "two_sided_empirical_p_value": 0.001 if significant else 0.5,
        "significant": significant, "below_null": below_null,
    }


def test_offset_control_branch_reachable_on_synthetic_pooled_data():
    real_pooled = _pooled_rotation_result(mean_value=0.20, mdd=0.05, significant=True, below_null=False)
    bias_pooled = _pooled_rotation_result(mean_value=0.18, mdd=0.05, significant=True, below_null=False)
    assert _bias_only_voids(real_pooled, bias_pooled) is True
    classification = _classify_arm(real_pooled, bias_pooled)
    assert classification["branch"] == "displacement_direction_not_separable_from_a_unit_level_offset"


def test_offset_control_does_not_fire_when_bias_alignment_is_not_significant():
    real_pooled = _pooled_rotation_result(mean_value=0.20, mdd=0.05, significant=True, below_null=False)
    bias_pooled = _pooled_rotation_result(mean_value=0.10, mdd=0.05, significant=False, below_null=False)
    assert _bias_only_voids(real_pooled, bias_pooled) is False
    classification = _classify_arm(real_pooled, bias_pooled)
    assert classification["branch"] == "stimulation_pushes_along_the_deviation_axis"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
