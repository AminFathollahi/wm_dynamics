"""Tests for scripts/run_watters_load_decomposition.py's reporting-only
additions: the per-item-count-level session-mean summary table (the correct
way to see whether a between-load slope's size comes from its numerator or
its denominator moving), the mean/median pairing and units disclosure carried
on every between-load slope block, and that a within-load correlation block
is left untouched by that disclosure (it is not a slope and must not carry a
slope-units caveat)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_watters_load_decomposition import (  # noqa: E402
    BETWEEN_LOAD_SLOPE_UNITS_AND_COMPARABILITY_NOTE,
    PRIMARY_QUALITY_TIER,
    _between_load_level_summary,
    _between_versus_within_sign,
    _cell_verdict,
)


def _fake_row(session_id: str, animal: str, task_variant: str, per_level: dict, between_slope: float) -> dict:
    within_trial = sum(e["r"] * e["n_trials"] for e in per_level.values()) / sum(
        e["n_trials"] for e in per_level.values())
    return {
        "animal": animal, "session_date": "2026-01-01", "session": session_id, "task_variant": task_variant,
        "by_tier": {
            PRIMARY_QUALITY_TIER: {
                "status": "computed",
                "per_level": {str(k): {**v, "status": "computed"} for k, v in per_level.items()},
                "within_load_trial_count_weighted": within_trial,
                "within_load_equal_weighted": sum(e["r"] for e in per_level.values()) / len(per_level),
                "between_load_slope_on_level_means": between_slope,
            }
        },
    }


def test_between_load_level_summary_computes_across_session_means():
    rows = [
        _fake_row("s1", "A", "ring", {
            1: {"r": 0.1, "n_trials": 20, "mean_report_error": 0.05, "mean_state_deviation": 0.10},
            2: {"r": 0.2, "n_trials": 20, "mean_report_error": 0.15, "mean_state_deviation": 0.08},
        }, between_slope=5.0),
        _fake_row("s2", "B", "triangle", {
            1: {"r": 0.3, "n_trials": 20, "mean_report_error": 0.09, "mean_state_deviation": 0.12},
        }, between_slope=None),
    ]
    summary = _between_load_level_summary(rows, PRIMARY_QUALITY_TIER)
    assert set(summary) == {"1", "2"}
    assert summary["1"]["n_sessions"] == 2
    assert summary["1"]["mean_report_error_across_sessions"] == (0.05 + 0.09) / 2
    assert summary["1"]["mean_state_deviation_across_sessions"] == (0.10 + 0.12) / 2
    assert summary["2"]["n_sessions"] == 1
    assert summary["2"]["mean_report_error_across_sessions"] == 0.15


def test_between_versus_within_sign_carries_median_and_units_note():
    pooled_between = {"status": "tested", "mean_value": -17.69, "median_value": -11.34}
    pooled_within = {"status": "tested", "mean_value": 0.0197}
    result = _between_versus_within_sign(pooled_between, pooled_within, delivered_raw_mean=-0.0117)
    assert result["between_load_mean_slope"] == -17.69
    assert result["between_load_median_slope"] == -11.34
    assert result["between_load_slope_units_and_comparability"] == BETWEEN_LOAD_SLOPE_UNITS_AND_COMPARABILITY_NOTE
    assert result["between_load_opposes_within_load"] is True


def test_cell_verdict_between_load_block_carries_units_note_within_load_does_not():
    rows = [
        _fake_row("s1", "A", "ring", {
            1: {"r": 0.1, "n_trials": 20, "mean_report_error": 0.05, "mean_state_deviation": 0.10},
            2: {"r": 0.2, "n_trials": 20, "mean_report_error": 0.15, "mean_state_deviation": 0.08},
        }, between_slope=5.0),
        _fake_row("s2", "B", "triangle", {
            1: {"r": 0.15, "n_trials": 20, "mean_report_error": 0.06, "mean_state_deviation": 0.11},
            2: {"r": 0.25, "n_trials": 20, "mean_report_error": 0.16, "mean_state_deviation": 0.09},
        }, between_slope=6.0),
    ]
    verdict = _cell_verdict(PRIMARY_QUALITY_TIER, "pooled", rows, gate_lookup={},
                             reference_magnitude=0.05, delivered_raw_mean=-0.01)
    assert verdict["between_load_on_level_means"]["units_and_comparability"] == \
        BETWEEN_LOAD_SLOPE_UNITS_AND_COMPARABILITY_NOTE
    assert "units_and_comparability" not in verdict["within_load_trial_count_weighted"]
    assert "units_and_comparability" not in verdict["within_load_equal_weighted"]


if __name__ == "__main__":
    test_between_load_level_summary_computes_across_session_means()
    test_between_versus_within_sign_carries_median_and_units_note()
    test_cell_verdict_between_load_block_carries_units_note_within_load_does_not()
    print("ok")
