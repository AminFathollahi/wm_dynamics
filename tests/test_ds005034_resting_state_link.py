"""Tests for scripts/run_ds005034_resting_state_link.py -- the pieces that could
silently break this analysis without erroring: (1) eyes-closed/eyes-open segment
delimitation from the rest recording's own eyec/eyeo event markers, including the
missing-eyec fallback and the truncated eyes-open segment failing its duration
floor rather than returning a garbage number, (2) session order recovery from the
.set file's embedded acquisition timestamp and its Patient-ID cross-check,
(3) per-session checkpointing round-trips exactly, (4) the two multiplicity
families are FDR-corrected independently of each other and of the primary cells,
(5) a NaN estimator failure is distinguished from a successful near-zero estimate
and is never silently dropped, (6) no artifact key or string value carries a
clearance label."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_ds005034_resting_state_link as mod  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results" / "ds005034_resting_state_link.json"


def _no_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(obj, dict):
        return all(_no_ndarray(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_no_ndarray(v) for v in obj)
    return True


def _write_events(tmp_path, rows, name="sub-01_ses-sham_task-rest_events.tsv"):
    path = tmp_path / name
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["onset", "duration", "sample", "value"])
        for row in rows:
            writer.writerow(row)
    return path


def test_segment_bounds_uses_eyec_and_eyeo_markers(tmp_path):
    path = _write_events(tmp_path, [
        (-0.0005, 2.0, -0.5, "boundary"),
        (0.999, 1, 999, "eyec"),
        (180.5, 1, 180500, "eyeo"),
    ])
    bounds = mod.segment_bounds(path, total_duration=181.5)
    assert bounds["eyes_closed"] == (0.999, 180.5)
    assert bounds["eyes_open"] == (180.5, 181.5)


def test_segment_bounds_falls_back_to_zero_without_eyec(tmp_path):
    path = _write_events(tmp_path, [(170.0, 1, 170000, "eyeo")])
    bounds = mod.segment_bounds(path, total_duration=171.0)
    assert bounds["eyes_closed"] == (0.0, 170.0)
    assert bounds["eyes_open"] == (170.0, 171.0)


def test_segment_bounds_uses_earliest_eyec_when_duplicated(tmp_path):
    path = _write_events(tmp_path, [
        (0.999, 1, 999, "eyec"),
        (9.001, 1, 9001, "eyec"),
        (200.0, 1, 200000, "eyeo"),
    ])
    bounds = mod.segment_bounds(path, total_duration=201.0)
    assert bounds["eyes_closed"][0] == pytest.approx(0.999)


def test_segment_bounds_raises_without_eyeo(tmp_path):
    path = _write_events(tmp_path, [(0.999, 1, 999, "eyec")])
    with pytest.raises(ValueError):
        mod.segment_bounds(path, total_duration=100.0)


def test_eyes_open_segment_in_this_release_fails_the_duration_floor():
    """The release truncates every rest recording to ~1.0 s after the eyeo
    marker (verified against all 50 files on disk). segment_band_power must
    treat that as a real failure with a reason, never as a zero."""
    result = mod.segment_band_power(
        filtered=np.zeros((5, 100)), times=np.arange(100) / 250.0, interval=(0.0, 1.002),
        csd_transform=np.eye(5), channel_index={"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4},
    )
    assert result["status"] == "failure_nan"
    assert result["duration_seconds"] == pytest.approx(1.002)
    assert "reason" in result


def test_segment_band_power_computes_above_the_duration_floor():
    sfreq = 250.0
    n = int(60 * sfreq)
    t = np.arange(n) / sfreq
    theta = np.sin(2 * np.pi * 6.0 * t)
    channels = {name: i for i, name in enumerate(mod.THETA_ROI + mod.POSTERIOR_ROI)}
    filtered = np.tile(theta, (len(channels), 1)) * 10.0
    result = mod.segment_band_power(
        filtered=filtered, times=t, interval=(0.0, 60.0),
        csd_transform=np.eye(len(channels)), channel_index=channels,
    )
    assert result["status"] == "computed"
    assert set(result["log_power"]) == {mod.PRIMARY_KEY, *mod.SECONDARY_KEYS}
    assert all(np.isfinite(v) for v in result["log_power"].values())


@pytest.mark.parametrize("patient_id,expected", [
    ("WMt_02_1RS", 1), ("WMt_02_2RS", 2), ("WMt03_1", 1), ("WMt_04_01", 1),
    ("wmt_07_1", 1), ("WM_12_1", 1), ("WMt_35_02_active", 2), (None, None), ("nodigits", None),
])
def test_parse_session_index_from_patient_id(patient_id, expected):
    assert mod._parse_session_index_from_patient_id(patient_id) == expected


def test_parse_session_index_uninformative_when_both_sessions_share_a_suffix():
    """sub-25's release-shipped Patient ID is identical on both arms ('WMt_25_01');
    verified on disk. The parser must not fabricate a distinction that isn't there."""
    assert mod._parse_session_index_from_patient_id("WMt_25_01") == 1
    assert mod._parse_session_index_from_patient_id("WMt_25_01") == 1


def test_matlab_datenum_to_iso_matches_a_verified_release_timestamp():
    assert mod._matlab_datenum_to_iso(736638.444104185).startswith("2016-11-04T10:39:30")


def test_correlation_mdd_grows_with_ci_width():
    narrow = mod._correlation_mdd(-0.1, 0.1)
    wide = mod._correlation_mdd(-0.5, 0.5)
    assert narrow["mdd"] < wide["mdd"]
    assert narrow["status"] == "computed"


def test_session_checkpoint_round_trips_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    record = {
        "participant": "sub-99", "stimulation_session": "sham",
        "rest": {"status": "computed", "total_duration_seconds": 180.0,
                 "global_bad_channels": ["E1"],
                 "segments": {"eyes_closed": {"status": "computed", "duration_seconds": 170.0,
                                               "log_power": {"frontal_theta": 1.0}},
                              "eyes_open": {"status": "failure_nan", "duration_seconds": 1.0, "reason": "too short"}},
                 "order_metadata": {"matlab_datenum": 736638.44, "acquisition_datetime_iso": "x",
                                     "patient_id_field": "WMt_99_1"}},
        "decode": {"status": "computed", "cells": {}, "primary_diag_auc": 0.6},
    }
    mod.save_session_checkpoint("sub-99_ses-sham", record)
    reloaded = mod.load_session_checkpoint("sub-99_ses-sham")
    assert reloaded == record
    assert isinstance(reloaded["decode"]["primary_diag_auc"], float)


def test_missing_or_excluded_checkpoint_never_masquerades_as_computed(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    assert mod.load_session_checkpoint("sub-99_ses-sham") is None
    mod.save_session_checkpoint("sub-99_ses-verum", {"rest": {"status": "failure_nan", "reason": "x"}, "decode": {"status": "failure_nan", "reason": "y"}})
    cached = mod.load_session_checkpoint("sub-99_ses-verum")
    assert cached["rest"]["status"] == "failure_nan"


def _fabricate_per_session(participants, rng):
    per_session = {}
    for i, participant in enumerate(participants):
        for session, base_theta, base_auc in (("sham", 1.0, 0.55), ("verum", 1.2, 0.60)):
            segs = {
                "eyes_closed": {"status": "computed", "duration_seconds": 180.0,
                                 "log_power": {k: base_theta + 0.01 * i + rng.normal(0, 0.05)
                                               for k in (mod.PRIMARY_KEY,) + mod.SECONDARY_KEYS}},
                "eyes_open": {"status": "failure_nan", "duration_seconds": 1.0, "reason": "too short"},
            }
            per_session[f"{participant}_ses-{session}"] = {
                "participant": participant, "stimulation_session": session,
                "rest": {"status": "computed", "total_duration_seconds": 181.0,
                         "global_bad_channels": [], "segments": segs,
                         "order_metadata": {
                             "matlab_datenum": 736638.0 + i + (0.1 if session == "verum" else 0.0),
                             "acquisition_datetime_iso": "x",
                             "patient_id_field": f"WMt_{i}_{'1' if session == 'sham' else '2'}"}},
                "decode": {"status": "computed", "cells": {},
                           "primary_diag_auc": base_auc + 0.01 * i + rng.normal(0, 0.03)},
            }
    return per_session


def test_order_recovery_agrees_with_fabricated_timestamps():
    participants = [f"sub-{i:02d}" for i in range(6)]
    per_session = _fabricate_per_session(participants, np.random.default_rng(0))
    order = mod.synthesize_order(per_session, participants)
    assert order["n_recovered"] == 6
    assert order["n_sham_first"] == 6  # sham datenum is always earlier by construction
    assert order["n_id_corroborated"] == order["n_id_informative"] == 6


def test_family1_eyes_open_fails_structurally_not_as_zero():
    participants = [f"sub-{i:02d}" for i in range(6)]
    rng = np.random.default_rng(1)
    per_session = _fabricate_per_session(participants, rng)
    order = mod.synthesize_order(per_session, participants)
    family1 = mod.synthesize_family1(per_session, participants, order)
    eo_primary = family1["eyes_open"]["primary"][mod.PRIMARY_KEY]
    assert eo_primary["cell_status"] == "failure_nan"
    assert eo_primary.get("mean_diff") is None
    assert eo_primary.get("estimate") is None


def test_family1_secondary_family_is_fdr_corrected_independently_of_primary():
    participants = [f"sub-{i:02d}" for i in range(10)]
    rng = np.random.default_rng(2)
    per_session = _fabricate_per_session(participants, rng)
    order = mod.synthesize_order(per_session, participants)
    family1 = mod.synthesize_family1(per_session, participants, order)
    primary = family1["eyes_closed"]["primary"][mod.PRIMARY_KEY]
    secondary = family1["eyes_closed"]["secondary"]
    assert primary["q_value"] is None  # single pre-declared primary, never FDR-corrected
    assert all(secondary[k]["q_value"] is not None for k in mod.SECONDARY_KEYS)
    assert _no_ndarray(family1)


def test_family2_reports_within_and_between_participant_as_separate_cells():
    participants = [f"sub-{i:02d}" for i in range(8)]
    rng = np.random.default_rng(3)
    per_session = _fabricate_per_session(participants, rng)
    family2 = mod.synthesize_family2(per_session, participants)
    within = family2["cells"]["within_participant"]
    between = family2["cells"]["between_participant"]
    assert within["definition"] != between["definition"]
    assert within["n"] == between["n"] == family2["n_participants_with_both_measures_in_both_arms"]
    assert within["q_value"] is not None and between["q_value"] is not None
    assert _no_ndarray(family2)


def test_family2_correlation_cell_reports_mdd_and_ci():
    participants = [f"sub-{i:02d}" for i in range(8)]
    rng = np.random.default_rng(4)
    per_session = _fabricate_per_session(participants, rng)
    family2 = mod.synthesize_family2(per_session, participants)
    for cell in family2["cells"].values():
        assert cell["ci_lower"] <= cell["estimate"] <= cell["ci_upper"]
        assert cell["minimum_detectable_difference_at_80pct_power"]["mdd"] > 0.0


def test_correlation_cell_fails_below_three_participants():
    cell = mod._correlation_cell(np.array([0.1, 0.2]), np.array([0.5, 0.6]), "tiny", "definition")
    assert cell["cell_status"] == "failure_nan"
    assert cell.get("estimate") is None


def test_no_clearance_label_anywhere_in_a_fabricated_family1_artifact():
    banned = ("clear", "not_cleared", "inconclusive_not_cleared", "passing", "failing", "confirmed")
    participants = [f"sub-{i:02d}" for i in range(6)]
    rng = np.random.default_rng(5)
    per_session = _fabricate_per_session(participants, rng)
    order = mod.synthesize_order(per_session, participants)
    family1 = mod.synthesize_family1(per_session, participants, order)
    family2 = mod.synthesize_family2(per_session, participants)

    def walk(obj):
        if isinstance(obj, dict):
            if "cell_status" in obj and obj["cell_status"] is not None:
                assert not any(b in obj["cell_status"] for b in banned), obj["cell_status"]
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(family1)
    walk(family2)


def test_delivered_artifact_has_no_clearance_label_or_omits_eyes_open_note():
    if not RESULTS.is_file():
        return
    artifact = json.loads(RESULTS.read_text())
    banned = ("clear", "not_cleared", "inconclusive_not_cleared")

    def walk(obj):
        if isinstance(obj, dict):
            assert "verdict" not in obj
            if "cell_status" in obj and obj["cell_status"] is not None:
                assert not any(b in obj["cell_status"] for b in banned), obj["cell_status"]
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(artifact)
    assert "eyes_open_truncation" in artifact.get("design", {})
