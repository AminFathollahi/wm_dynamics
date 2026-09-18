import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_observability_matched_modality_test.py"
SPEC = importlib.util.spec_from_file_location("observability_matched_modality_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _session(patient, session, bin_ms, status, trials=20, nugget=0.2):
    return {
        "patient": patient, "session": session, "bin_ms": bin_ms,
        "status": status, "n_trials": trials, "median_nugget_fraction": nugget,
        "n_splits_fitted": 8 if status == "fitted" else 2,
        "n_units_or_channels": 20, "median_slow_timescale_s": 1.0,
    }


def test_matched_test_joins_only_same_patient_and_session_and_tracks_counts():
    unit = [_session("p1", "s1", 200, "fitted"), _session("p2", "s2", 200, "not_fitted")]
    lfp = [_session("p1", "s1", 200, "not_fitted"), _session("p2", "s2", 200, "fitted"),
           _session("p3", "s3", 200, "fitted")]
    result = MODULE.matched_modality_test(unit, lfp, 200)
    assert result["n_sessions_with_both_grains"] == 2
    assert result["pairing_verification"]["n_trial_count_mismatches"] == 0
    assert result["counts"]["present_at_lfp_absent_at_unit"] == 1
    assert result["counts"]["present_at_unit_absent_at_lfp"] == 1


def test_matched_test_surfaces_trial_count_mismatch():
    result = MODULE.matched_modality_test(
        [_session("p1", "s1", 100, "fitted", trials=20)],
        [_session("p1", "s1", 100, "fitted", trials=19)], 100,
    )
    assert result["pairing_verification"]["n_trial_count_mismatches"] == 1
    assert result["sessions"][0]["same_trial_count"] is False


def test_discordance_test_uses_only_directional_disagreements():
    matched = {"bin100": {"bin_ms": 100, "counts": {
        "present_at_lfp_absent_at_unit": 1,
        "present_at_unit_absent_at_lfp": 1,
        "present_at_both": 0, "absent_at_both": 0,
    }}, "bin200": {"bin_ms": 200, "counts": {
        "present_at_lfp_absent_at_unit": 2,
        "present_at_unit_absent_at_lfp": 0,
        "present_at_both": 5, "absent_at_both": 3,
    }}}
    result = MODULE.compute_grain_discordance_paired_test(matched)["by_bin_width"]["bin200"]
    assert result["n_discordant"] == 2
    assert result["two_sided_exact_binomial_p_value"] == 0.5


def test_participant_test_clusters_repeated_sessions():
    sessions = [
        {"patient": "p1", "unit_status": "not_fitted", "lfp_status": "fitted"},
        {"patient": "p1", "unit_status": "not_fitted", "lfp_status": "fitted"},
        {"patient": "p2", "unit_status": "fitted", "lfp_status": "not_fitted"},
    ]
    matched = {
        "bin100": {"bin_ms": 100, "sessions": sessions},
        "bin200": {"bin_ms": 200, "sessions": sessions},
    }
    result = MODULE.compute_grain_discordance_participant_test(matched)
    cell = result["by_bin_width"]["bin200"]
    assert cell["n_participants"] == 2
    assert cell["mean_difference"] == 0.0
    assert cell["participant_mean_lfp_minus_unit_fittability"] == {"p1": 1.0, "p2": -1.0}
