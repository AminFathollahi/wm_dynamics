"""Checks for the serial-dependence permutation candidates."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for subdirectory in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / subdirectory))

import run_trial_exchangeability_calibration as calibration  # noqa: E402


def test_circular_shift_preserves_values_and_uses_nonzero_lag() -> None:
    block = calibration._block(np.arange(12.0), np.arange(12.0) ** 2)
    shifted = calibration.current_engine._circular_outcome_shift(np.random.default_rng(4), block)
    assert np.array_equal(np.sort(shifted["outcome"]), np.sort(block["outcome"]))
    assert not np.array_equal(shifted["outcome"], block["outcome"])
    assert np.array_equal(shifted["predictor"], block["predictor"])


def test_circular_residual_shift_retains_reduced_fit_component() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(size=80)
    outcome = 2.0 * z + rng.normal(size=80)
    block = calibration._block(rng.normal(size=80), outcome, [z])
    shifted = calibration.current_engine._circular_residual_shift(rng, block)
    design = np.column_stack([np.ones(80), z])
    original_fitted = design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
    shifted_residual = shifted["outcome"] - original_fitted
    assert np.allclose(np.sort(shifted_residual), np.sort(outcome - original_fitted))
