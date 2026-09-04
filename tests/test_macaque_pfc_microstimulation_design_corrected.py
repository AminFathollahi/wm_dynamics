from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_macaque_pfc_microstimulation_design_corrected import (  # noqa: E402
    _log_odds_effect,
    bin_spiketrain,
    fit_recovery,
    recovery_agreement,
)


def test_bin_spiketrain_uses_nonoverlapping_1ms_counts():
    spikes = np.zeros((1500, 2), dtype=np.uint8)
    spikes[[0, 49, 50, 1499], 0] = 1
    spikes[100:150, 1] = 1
    counts = bin_spiketrain(spikes)
    assert counts.shape == (30, 2)
    assert counts[0, 0] == 2
    assert counts[1, 0] == 1
    assert counts[-1, 0] == 1
    assert counts[2, 1] == 50


def test_angle_stratified_behavior_effect_uses_each_row_once():
    rows = []
    for angle in (0, 1):
        for correct in (1, 1, 1, 0):
            rows.append({"stim_cond": 1, "angle_idx": angle, "correct": correct})
        for correct in (1, 0, 0, 0):
            rows.append({"stim_cond": 0, "angle_idx": angle, "correct": correct})
    result = _log_odds_effect(rows, stim_cond=1, control_idx=0)
    assert result is not None
    assert result["estimate"] > 0
    assert len(result["angle_tables"]) == 2
    assert sum(sum(table[key] for key in ("stim_correct", "stim_error", "control_correct", "control_error"))
               for table in result["angle_tables"]) == len(rows)


def test_signed_exponential_recovery_recovers_rate():
    time = np.arange(14) * 0.05
    post = 0.2 + 1.4 * np.exp(-3.0 * time)
    displacement = np.r_[np.zeros(16), post]
    result = fit_recovery(displacement)
    assert result is not None
    assert abs(result["lambda_rate_per_s"] - 3.0) < 1e-4


def test_recovery_agreement_excludes_endogenous_intervals_crossing_zero():
    sessions = [{
        "session": "wide",
        "endogenous_control_drift": {"lambda_rate": 2.0, "lambda_ci": [-1.0, 5.0]},
        "patterns": [{"recovery": {"lambda_rate_per_s": 2.0},
                      "recovery_lambda_bootstrap_ci": [1.0, 3.0]}],
    }, {
        "session": "resolved",
        "endogenous_control_drift": {"lambda_rate": 4.0, "lambda_ci": [3.0, 5.0]},
        "patterns": [{"recovery": {"lambda_rate_per_s": 4.5},
                      "recovery_lambda_bootstrap_ci": [4.0, 6.0]}],
    }]
    result = recovery_agreement(sessions, n_boot=20)
    assert result["n_recovery_patterns"] == 2
    assert result["n_patterns_with_precision_identified_endogenous_lambda"] == 1
    assert result["interval_overlap_count"] == 1
    assert result["verdict"] == "agreement_not_established"
