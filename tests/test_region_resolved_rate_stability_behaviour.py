"""Tests for scripts/run_region_resolved_rate_stability_behaviour.py.

Covers the pieces that could silently break the region-resolved fit without
erroring: (1) per-patient row pooling across sessions (never the unit, never
the trial); (2) the partial-correlation cell (raw vs spike-count-controlled,
underpowered/not-computable edge cases); (3) the paired region-contrast cell;
(4) region-summary min/median/max arithmetic; (5) checkpoint round-trip;
(6) the count-matched between-region draw, including the unit-count-matching
guarantee itself (never comparing raw unmatched counts); (7) an end-to-end
synthetic-session check of _region_cell and the underlying CTG off-diagonal
fit; (8) that the region admission floor is the same MIN_UNITS_PER_REGION
constant spike_pipeline.py already defines, not a second unexplained one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_region_resolved_rate_stability_behaviour import CORPUS_SPECS, _correlation_cell, _fit_ctg_offdiag, _matched_pair_session, _paired_contrast, _partial_r_stat, _patient_rows, _region_cell, _region_estimate_summary
from statistics import _trial_population_spike_count
import run_region_resolved_rate_stability_behaviour as target_module  # noqa: E402
from spike_pipeline import MIN_UNITS_PER_REGION, MIN_UNIT_FIRING_RATE_HZ  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# region admission floor
# ---------------------------------------------------------------------------------------------------

def test_region_admission_floor_reuses_spike_pipeline_constant():
    assert target_module.MIN_UNITS_PER_REGION is MIN_UNITS_PER_REGION


# ---------------------------------------------------------------------------------------------------
# _partial_r_stat
# ---------------------------------------------------------------------------------------------------

def test_partial_r_stat_no_controls_equals_plain_correlation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(30)
    y = 2.0 * x + rng.standard_normal(30) * 0.1
    r = _partial_r_stat(y, x, [])
    assert r == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_partial_r_stat_removes_confound_shared_with_both():
    rng = np.random.default_rng(1)
    n = 400
    confound = rng.standard_normal(n)
    x = confound + rng.standard_normal(n) * 0.05
    y = confound + rng.standard_normal(n) * 0.05
    raw = _partial_r_stat(y, x, [])
    controlled = _partial_r_stat(y, x, [confound])
    assert raw > 0.9
    assert abs(controlled) < abs(raw)


def test_partial_r_stat_zero_variance_residual_is_nan():
    y = np.ones(10)
    x = np.arange(10, dtype=float)
    assert np.isnan(_partial_r_stat(y, x, []))


# ---------------------------------------------------------------------------------------------------
# _correlation_cell
# ---------------------------------------------------------------------------------------------------

def test_correlation_cell_underpowered_below_four_patients():
    out = _correlation_cell([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [], "test|underpowered")
    assert out["status"] == "underpowered"
    assert out["n"] == 3


def test_correlation_cell_computed_carries_estimate_ci_mdd_and_q_precursor():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(12)
    y = 0.8 * x + rng.standard_normal(12) * 0.3
    out = _correlation_cell(list(x), list(y), [], "test|computed")
    assert out["status"] == "computed"
    for key in ("estimate", "ci_lower", "ci_upper", "mdd", "p_value", "n"):
        assert key in out
    assert out["ci_lower"] <= out["estimate"] <= out["ci_upper"]
    assert out["mdd"] >= 0.0


def test_correlation_cell_with_control_matches_partial_r_stat():
    rng = np.random.default_rng(3)
    n = 20
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    control = rng.standard_normal(n)
    out = _correlation_cell(list(x), list(y), [control], "test|controlled")
    assert out["status"] == "computed"
    assert out["n_controls"] == 1
    assert out["estimate"] == pytest.approx(_partial_r_stat(y, x, [control]))


def test_correlation_cell_deterministic_across_repeat_calls():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [1.1, 1.9, 3.2, 3.8, 5.1, 6.2]
    a = _correlation_cell(x, y, [], "same_seed_tag")
    b = _correlation_cell(x, y, [], "same_seed_tag")
    assert a == b


# ---------------------------------------------------------------------------------------------------
# _region_estimate_summary
# ---------------------------------------------------------------------------------------------------

def test_region_estimate_summary_not_computable_below_two_values():
    assert _region_estimate_summary([1.0])["status"] == "not_computable"
    assert _region_estimate_summary([])["status"] == "not_computable"


def test_region_estimate_summary_computes_median_min_max():
    out = _region_estimate_summary([1.0, 2.0, 3.0, None, float("nan")])
    assert out["status"] == "computed"
    assert out["n"] == 3
    assert out["median"] == 2.0
    assert out["min"] == 1.0
    assert out["max"] == 3.0


# ---------------------------------------------------------------------------------------------------
# _paired_contrast
# ---------------------------------------------------------------------------------------------------

def test_paired_contrast_underpowered_below_four_patients():
    rows = [{"a": 1.0, "b": 0.5}, {"a": 2.0, "b": 1.0}]
    out = _paired_contrast(rows, "a", "b")
    assert out["status"] == "underpowered"
    assert out["n_patients"] == 2


def test_paired_contrast_ignores_rows_with_a_missing_value():
    rows = [
        {"a": 1.0, "b": 0.5}, {"a": 2.0, "b": 1.0}, {"a": None, "b": 1.0},
        {"a": 3.0, "b": 1.5}, {"a": 4.0, "b": 2.0},
    ]
    out = _paired_contrast(rows, "a", "b")
    assert out["n_patients"] == 4
    assert len(out["values_a"]) == 4


# ---------------------------------------------------------------------------------------------------
# _patient_rows: resampling unit is the patient, sessions of one patient pool first
# ---------------------------------------------------------------------------------------------------

def test_patient_rows_pools_sessions_within_patient_and_skips_non_computed():
    region_sessions = {
        "s1": {"patient": "sub-1", "status": "computed", "n_units": 10, "n_trials": 40,
               "n_correct": 30, "median_unit_rate_hz": 2.0, "mean_trial_spike_count": 5.0,
               "ctg": {"offdiag_effect": 0.1}},
        "s2": {"patient": "sub-1", "status": "computed", "n_units": 12, "n_trials": 60,
               "n_correct": 50, "median_unit_rate_hz": 4.0, "mean_trial_spike_count": 7.0,
               "ctg": {"offdiag_effect": 0.3}},
        "s3": {"patient": "sub-2", "status": "refused", "reason": "x"},
    }
    rows = _patient_rows(region_sessions)
    assert len(rows) == 1
    row = rows[0]
    assert row["patient"] == "sub-1"
    assert row["n_sessions"] == 2
    assert row["rate_hz"] == pytest.approx(3.0)
    assert row["stability_offdiag_effect"] == pytest.approx(0.2)
    assert row["n_trials"] == 100
    assert row["accuracy"] == pytest.approx(80.0 / 100.0)


# ---------------------------------------------------------------------------------------------------
# _trial_population_spike_count
# ---------------------------------------------------------------------------------------------------

def test_trial_population_spike_count_sums_across_units():
    spike_lists = [np.array([0.1, 0.5, 1.2]), np.array([0.2, 0.4])]
    onsets = np.array([0.0, 1.0])
    counts = _trial_population_spike_count(spike_lists, onsets, window_s=1.0)
    # trial at t=0: unit1 has 0.1,0.5 in [0,1); unit2 has 0.2,0.4 in [0,1) -> 4
    # trial at t=1: unit1 has 1.2 in [1,2); unit2 has none -> 1
    assert counts.tolist() == [4.0, 1.0]


# ---------------------------------------------------------------------------------------------------
# _fit_ctg_offdiag underpowered gate
# ---------------------------------------------------------------------------------------------------

def _synthetic_psth(n_trials, n_units, n_bins, rng):
    return rng.standard_normal((n_trials, n_units, n_bins)).astype(np.float32)


def test_fit_ctg_offdiag_returns_none_below_five_per_class():
    rng = np.random.default_rng(4)
    psth = _synthetic_psth(8, 6, 10, rng)
    condition = np.array([1] * 4 + [3] * 4)
    out = _fit_ctg_offdiag(psth, condition, low=1, high=3, n_components=3, rng=rng)
    assert out is None


def test_fit_ctg_offdiag_computes_with_enough_trials_per_class():
    rng = np.random.default_rng(5)
    psth = _synthetic_psth(30, 8, 12, rng)
    condition = np.array([1] * 15 + [3] * 15)
    out = _fit_ctg_offdiag(psth, condition, low=1, high=3, n_components=4, rng=rng)
    assert out is not None
    for key in ("offdiag_effect", "mean_diag_auc", "tau", "interpretable"):
        assert key in out


# ---------------------------------------------------------------------------------------------------
# end-to-end synthetic _region_cell and count-matched _matched_pair_session
# ---------------------------------------------------------------------------------------------------

def _synthetic_session(rng, n_units_hippocampus=20, n_units_amygdala=6, n_trials=40, window_s=2.3):
    n_units_total = n_units_hippocampus + n_units_amygdala
    unit_regions = np.array(["hippocampus"] * n_units_hippocampus + ["amygdala"] * n_units_amygdala)
    onsets = np.arange(n_trials, dtype=float) * (window_s + 1.0)
    spike_lists_all = []
    for _ in range(n_units_total):
        spikes = []
        for t0 in onsets:
            n_spk = rng.poisson(2.0 * window_s)
            spikes.append(t0 + rng.uniform(0, window_s, size=n_spk))
        spike_lists_all.append(np.sort(np.concatenate(spikes)) if spikes else np.array([]))
    condition = np.array([1] * (n_trials // 2) + [3] * (n_trials - n_trials // 2))
    accuracy = rng.uniform(size=n_trials) > 0.3
    return dict(patient="sub-synthetic", spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                t_onset=onsets, maint_win=window_s, condition=condition, low=1, high=3, accuracy=accuracy)


def test_region_cell_end_to_end_on_synthetic_session_above_floor():
    rng = np.random.default_rng(6)
    session = _synthetic_session(rng, n_units_hippocampus=20, n_units_amygdala=6)
    cell = _region_cell(session, "hippocampus")
    assert cell["status"] == "computed"
    assert cell["n_units"] == 20
    assert cell["n_trials"] == 40
    assert cell["median_unit_rate_hz"] > 0.0
    assert cell["ctg_status"] == "computed"
    assert cell["ctg"]["interpretable"] in (True, False)


def test_region_cell_refused_below_unit_floor():
    rng = np.random.default_rng(7)
    session = _synthetic_session(rng, n_units_hippocampus=20, n_units_amygdala=6)
    cell = _region_cell(session, "amygdala")
    assert cell["status"] == "refused"
    assert f"{MIN_UNITS_PER_REGION}" in cell["reason"]
    assert f"{MIN_UNIT_FIRING_RATE_HZ}" in cell["reason"]


def test_matched_pair_session_is_count_matched_not_raw_unmatched():
    rng = np.random.default_rng(8)
    session = _synthetic_session(rng, n_units_hippocampus=20, n_units_amygdala=10)
    out = _matched_pair_session(session, "hippocampus", "amygdala", rng)
    assert out["status"] == "computed"
    # both arms fit at the SAME subsampled count, never the raw 20-vs-10 split
    assert out["target_n_units"] == 10
    assert out["target_n_units"] < 20


def test_matched_pair_session_refused_when_smaller_region_below_floor():
    rng = np.random.default_rng(9)
    session = _synthetic_session(rng, n_units_hippocampus=20, n_units_amygdala=3)
    out = _matched_pair_session(session, "hippocampus", "amygdala", rng)
    assert out["status"] == "refused"


# ---------------------------------------------------------------------------------------------------
# checkpoint round-trip
# ---------------------------------------------------------------------------------------------------

def test_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(target_module, "CHECKPOINT_DIR", tmp_path)
    name = "dandi_test__hippocampus"
    assert target_module._load_checkpoint(name) == {}
    target_module._record_session(name, "sub-1_ses-1", {"status": "computed", "n_units": 9})
    reloaded = target_module._load_checkpoint(name)
    assert reloaded["sub-1_ses-1"]["n_units"] == 9
    ck_file = tmp_path / f"{name}.json"
    assert ck_file.exists()
    on_disk = json.loads(ck_file.read_text())
    assert on_disk["sessions"]["sub-1_ses-1"]["status"] == "computed"


# ---------------------------------------------------------------------------------------------------
# corpus spec sanity: every declared matched pair references a declared region
# ---------------------------------------------------------------------------------------------------

def test_matched_pairs_reference_declared_regions():
    for corpus, spec in CORPUS_SPECS.items():
        for region_a, region_b in spec["matched_pairs"]:
            assert region_a in spec["regions"], (corpus, region_a)
            assert region_b in spec["regions"], (corpus, region_b)
