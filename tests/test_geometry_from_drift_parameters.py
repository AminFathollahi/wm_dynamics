"""Regression test for patient-level geometry calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_geometry_from_drift_parameters_000469 import calibration


def test_calibration_reports_identity_for_parameter_free_match():
    rows = [
        {"patient": f"p{index}", "predicted": value, "observed": value}
        for index, value in enumerate((0.3, 0.5, 0.8, 1.1, 1.5, 2.0))
    ]
    result = calibration(rows, np.random.default_rng(3))
    assert result["status"] == "estimable"
    assert result["pearson_correlation"] == 1.0
    assert np.isclose(result["calibration_slope"], 1.0)
