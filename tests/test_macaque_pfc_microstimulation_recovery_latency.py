from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_macaque_pfc_microstimulation_recovery_latency import (  # noqa: E402
    N_BINS,
    ONSET_BIN,
    condition_recovery,
    fit_cross_fold_projections,
    pooled_latency_vs_recovery,
    pre_stimulation_offset_diagnostics,
)
from provenance import checkpoint_safe, restore_checkpoint  # noqa: E402


def _make_rows(rng, n_units=10, n_control_per_angle=60, n_stim_per_angle=15):
    rows = []
    for angle in range(3):
        for _ in range(n_control_per_angle):
            counts = rng.poisson(3.0, size=(N_BINS, n_units)).astype(float)
            rows.append({"stim_cond": 0, "angle_idx": angle, "counts": counts,
                        "saccade_latency": None, "occurrence": 0})
    for cond in (1, 2):
        for angle in range(3):
            for _ in range(n_stim_per_angle):
                counts = rng.poisson(3.0, size=(N_BINS, n_units)).astype(float)
                counts[ONSET_BIN:ONSET_BIN + 4] += 5.0
                counts[ONSET_BIN + 4:] += 1.5
                rows.append({"stim_cond": cond, "angle_idx": angle, "counts": counts,
                            "saccade_latency": float(rng.uniform(0.1, 0.5)), "occurrence": 0})
    return rows


def test_cross_fold_fit_never_projects_a_control_trial_through_its_own_fold(rng):
    rows = _make_rows(rng)
    fit = fit_cross_fold_projections(rows, control_idx=0, prefix="unit_test_session")
    assert fit is not None
    control_rows_idx = [i for i, r in enumerate(rows) if r["stim_cond"] == 0]
    assert set(fit["control_latent"].keys()) == set(control_rows_idx)
    for fold, latents in fit["stim_latent_by_fold"].items():
        stim_rows_idx = {i for i, r in enumerate(rows) if r["stim_cond"] != 0}
        assert set(latents.keys()) == stim_rows_idx


def test_condition_recovery_reports_small_pre_stimulation_offset_when_no_leakage(rng):
    rows = _make_rows(rng)
    fit = fit_cross_fold_projections(rows, control_idx=0, prefix="unit_test_session")
    result = condition_recovery(rows, condition=1, control_idx=0,
                                analysis_start_bin=ONSET_BIN + 4, fit=fit)
    assert result is not None
    assert abs(result["pre_stimulation_offset"]) < 0.3
    assert result["level_shift_uncorrected"] > 0.5
    assert result["within_train_excursion_ratio"] > 1.0


def test_pooled_latency_vs_recovery_runs_sign_flip_and_reports_per_animal():
    per_session = [
        {"status": "complete", "animal": "Wa", "latency_vs_peak_recovery_distance": {"rho": 0.3, "n": 20}},
        {"status": "complete", "animal": "Wa", "latency_vs_peak_recovery_distance": {"rho": 0.5, "n": 25}},
        {"status": "complete", "animal": "Sa", "latency_vs_peak_recovery_distance": {"rho": -0.2, "n": 10}},
        {"status": "excluded"},
    ]
    result = pooled_latency_vs_recovery(per_session)
    assert result["status"] == "computed"
    assert result["n_sessions"] == 3
    assert 0.0 <= result["sign_flip_test"]["p_value"] <= 1.0
    assert result["per_animal"]["Wa"]["n_sessions"] == 2
    assert result["per_animal"]["Sa"]["n_sessions"] == 1


def test_pooled_latency_vs_recovery_not_computable_below_two_sessions():
    result = pooled_latency_vs_recovery([{"status": "complete", "animal": "Wa",
                                          "latency_vs_peak_recovery_distance": {"rho": 0.3, "n": 20}}])
    assert result["status"] == "not_computable"


def _condition(offset, angle_offsets):
    return {
        "pre_stimulation_offset": offset, "n_stim_trials_used": 15, "n_control_trials_used": 60,
        "mean_occurrence_index": 2.0,
        "pre_stimulation_offset_by_angle": {
            str(angle): {"pre_stimulation_offset": value, "n_stim_trials": 5}
            for angle, value in angle_offsets.items()
        },
    }


def test_pre_stimulation_offset_diagnostics_pools_a_resumed_sessions_by_angle_table():
    """analyze_session builds pre_stimulation_offset_by_angle keyed by str(condition/angle). A
    session served from load_checkpoint on a resumed run comes back through restore_checkpoint before
    this pooling loop sees it, and the loop keys a single shared by_angle dict off whatever type
    "angle" happens to be -- pre-fix, a resumed session's int keys and a fresh session's str keys
    split the same angle into two entries. Round-trip one session (simulating a resumed one) and pool
    it together with a freshly-built session (still str-keyed): both must land under the same key."""
    fresh_session = {"status": "complete", "conditions": {
        f"c{i}": _condition(0.1 * i, {0: 0.1 * i, 45: 0.2 * i}) for i in range(4)
    }}
    resumed_session_raw = {"status": "complete", "conditions": {
        f"c{i}": _condition(0.1 * i, {0: 0.1 * i, 45: 0.2 * i}) for i in range(4, 8)
    }}
    resumed_session = restore_checkpoint(json.loads(json.dumps(checkpoint_safe(resumed_session_raw))))

    result = pre_stimulation_offset_diagnostics([fresh_session, resumed_session])
    assert result["n_conditions"] == 8
    assert set(result["by_angle"]) == {"0", "45"}
