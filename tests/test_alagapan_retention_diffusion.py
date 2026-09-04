from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from run_alagapan_retention_diffusion import common_average_reference, condition_contrasts  # noqa: E402


def test_condition_contrasts_are_log_ratios_to_shared_sham():
    rows = {
        "Sham": {"state_space_total_diffusion": 2.0, "legacy_increment_total_diffusion": 4.0},
        "In Phase": {"state_space_total_diffusion": 4.0, "legacy_increment_total_diffusion": 2.0},
        "Anti Phase": {"state_space_total_diffusion": 1.0, "legacy_increment_total_diffusion": 8.0},
    }
    result = condition_contrasts(rows)
    assert result["state_space_total_diffusion"]["In Phase_minus_Sham_log_ratio"] > 0
    assert result["state_space_total_diffusion"]["Anti Phase_minus_Sham_log_ratio"] < 0
    assert result["legacy_increment_total_diffusion"]["In Phase_minus_Sham_log_ratio"] < 0


def test_common_average_reference_is_zero_mean_across_contacts():
    values = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
    referenced = common_average_reference(values)
    np.testing.assert_allclose(referenced.mean(axis=1), 0.0, atol=1e-12)
