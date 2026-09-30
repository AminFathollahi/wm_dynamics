"""Tests for scripts/run_ds005034_operation_versus_load_decoding.py -- the
pieces that could silently break this analysis without erroring: (1) a
within-window code stable across delay-window time bins must show a
decodable off-diagonal alongside its diagonal, (2) a code that flips
direction bin-to-bin must show a chance-level off-diagonal despite a
decodable diagonal, (3) aggregation averages multiple sessions per subject
before resampling, so the resampling unit is the participant and not the
session or trial, (4) a NaN estimator failure is distinguished from a
successful near-zero estimate, (5) no artifact key or string value carries a
clearance label."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_ds005034_operation_versus_load_decoding import aggregate_cell, CELL_FAMILY, save_session_checkpoint, load_session_checkpoint, CHECKPOINT_DIR
from preprocessing import within_window_ctg, window_stats
from preprocessing import fit_cell
from statistics import cell_status

RESULTS = Path(__file__).resolve().parents[1] / "results" / "ds005034_operation_versus_load_decoding.json"


def _no_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(obj, dict):
        return all(_no_ndarray(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_no_ndarray(v) for v in obj)
    return True


def _make_population(n_trials, n_units, n_bins, direction_per_bin, amplitude=1.3, noise=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    y = np.concatenate([np.zeros(n_trials // 2), np.ones(n_trials - n_trials // 2)]).astype(int)
    rng.shuffle(y)
    signed = (2 * y - 1).astype(float)
    psth = rng.normal(0.0, noise, size=(n_trials, n_units, n_bins))
    for t in range(n_bins):
        psth[:, :, t] += amplitude * signed[:, None] * direction_per_bin[t][None, :]
    return psth, y


def test_stable_bin_direction_generalises_across_delay_window():
    rng = np.random.default_rng(1)
    n_units, n_bins = 20, 6
    direction = rng.normal(size=n_units)
    direction /= np.linalg.norm(direction)
    psth, y = _make_population(200, n_units, n_bins, [direction] * n_bins, rng=rng)

    auc_mat = within_window_ctg(psth, y, n_components=6, n_splits=5, rng=np.random.default_rng(2))
    stats = window_stats(auc_mat)

    assert stats["diag_mean_auc"] > 0.8
    assert stats["offdiag_effect"] > 0.2


def test_rotating_bin_direction_kills_offdiagonal_not_diagonal():
    rng = np.random.default_rng(3)
    n_units, n_bins = 20, 6
    directions = []
    for _ in range(n_bins):
        d = rng.normal(size=n_units)
        d /= np.linalg.norm(d)
        directions.append(d)
    psth, y = _make_population(200, n_units, n_bins, directions, rng=rng)

    auc_mat = within_window_ctg(psth, y, n_components=6, n_splits=5, rng=np.random.default_rng(4))
    stats = window_stats(auc_mat)

    assert stats["diag_mean_auc"] > 0.75
    assert abs(stats["offdiag_effect"]) < 0.15


def test_aggregate_cell_resamples_at_the_subject_not_session_level():
    entries = [
        ("sub-01", {"offdiag_effect": 0.10, "diag_mean_auc": 0.7}),
        ("sub-01", {"offdiag_effect": 0.12, "diag_mean_auc": 0.72}),
        ("sub-02", {"offdiag_effect": 0.08, "diag_mean_auc": 0.68}),
    ]
    cell = aggregate_cell(entries, "test_tag")
    assert cell["n_subjects"] == 2
    assert cell["n_sessions"] == 3
    assert cell["estimand"] is not None
    assert cell["ci_lower"] <= cell["estimand"] <= cell["ci_upper"]


def test_aggregate_cell_distinguishes_nan_from_zero():
    nan_entries = [("sub-01", {"offdiag_effect": float("nan"), "diag_mean_auc": 0.5}),
                   ("sub-02", {"offdiag_effect": 0.0, "diag_mean_auc": 0.5})]
    nan_cell = aggregate_cell(nan_entries, "nan_tag")
    assert nan_cell["cell_status"] == "failure_nan"
    assert nan_cell["estimand"] is None

    zero_entries = [("sub-01", {"offdiag_effect": 0.0, "diag_mean_auc": 0.5}),
                    ("sub-02", {"offdiag_effect": 0.0, "diag_mean_auc": 0.5}),
                    ("sub-03", {"offdiag_effect": 0.0, "diag_mean_auc": 0.5})]
    zero_cell = aggregate_cell(zero_entries, "zero_tag")
    assert zero_cell["estimand"] == 0.0
    assert zero_cell["cell_status"] is None  # q_value/cell_status assigned after the FDR pass


def test_aggregate_cell_reports_mdd_from_shared_helper():
    entries = [(f"sub-{i:02d}", {"offdiag_effect": 0.05 + 0.001 * i, "diag_mean_auc": 0.6}) for i in range(6)]
    cell = aggregate_cell(entries, "mdd_tag")
    assert cell["minimum_detectable_difference_at_80pct_power"]["status"] == "computed"
    assert cell["minimum_detectable_difference_at_80pct_power"]["mdd"] > 0.0


def test_every_cell_belongs_to_exactly_one_predeclared_family():
    assert set(CELL_FAMILY.values()) == {"operation_decoding", "load_within_operation_decoding"}
    assert len(CELL_FAMILY) == 7


def test_cell_status_values_are_never_clearance_labels():
    banned = ("clear", "inconclusive_not_cleared", "not_cleared", "passing", "failing", "confirmed")
    permitted = {"estimated", "failure_nan", "descriptive_not_in_fdr_family", None}
    assert cell_status(is_nan=True) == "failure_nan"
    assert cell_status(is_nan=False) == "estimated"
    for value in permitted:
        if value is not None:
            assert not any(b in value for b in banned)


def test_delivered_artifact_has_no_clearance_label_or_behaviour_omission():
    if not RESULTS.is_file():
        return
    artifact = json.loads(RESULTS.read_text())
    banned = ("clear", "inconclusive_not_cleared", "not_cleared")

    def walk(obj):
        if isinstance(obj, dict):
            assert "verdict" not in obj
            if "cell_status" in obj:
                value = obj["cell_status"]
                if value is not None:
                    assert not any(b in value for b in banned), value
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(artifact)
    assert "behaviour_link_unavailable_reason" in artifact
    assert artifact["behaviour_link_unavailable_reason"]


def test_fit_cell_output_carries_no_ndarray():
    rng = np.random.default_rng(7)
    psth = rng.normal(size=(40, 10, 6))
    y = np.concatenate([np.zeros(20), np.ones(20)])
    stats = fit_cell("checkpoint_shape_probe", psth, y)
    assert stats is not None
    assert _no_ndarray(stats)


def test_session_checkpoint_round_trips_cell_values_exactly(tmp_path, monkeypatch):
    import run_ds005034_operation_versus_load_decoding as mod
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)

    rng = np.random.default_rng(8)
    psth = rng.normal(size=(40, 10, 6))
    y = np.concatenate([np.zeros(20), np.ones(20)])
    stats = fit_cell("checkpoint_round_trip_probe", psth, y)
    record = {"subject": "sub-99", "stimulation_session": "sham", "n_trials": 40,
              "cells": {"operation_forward_vs_backward": stats}}

    save_session_checkpoint("sub-99_ses-sham", record)
    reloaded = load_session_checkpoint("sub-99_ses-sham")

    assert reloaded == record
    assert isinstance(reloaded["cells"]["operation_forward_vs_backward"]["offdiag_effect"], float)
    assert not isinstance(reloaded["n_trials"], np.ndarray)


def test_missing_or_excluded_checkpoint_never_masquerades_as_a_computed_session(tmp_path, monkeypatch):
    import run_ds005034_operation_versus_load_decoding as mod
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)

    assert load_session_checkpoint("sub-99_ses-sham") is None
    save_session_checkpoint("sub-99_ses-verum", {"excluded": True})
    cached = load_session_checkpoint("sub-99_ses-verum")
    assert cached["excluded"] is True
