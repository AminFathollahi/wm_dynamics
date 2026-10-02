import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / _subdir))

import info_decoding  # noqa: E402
import run_info_benchmark  # noqa: E402
import run_readout_timing_followup as followup  # noqa: E402
from info_decoding import CTG_STEP  # noqa: E402
from spike_pipeline import BIN_MS  # noqa: E402

IMPLEMENTATION = {"test": "fixed"}
ENTRY = {"region": "hippocampus", "session_key": "P1_ses-1", "level": "load1_maintenance"}


def _planted_entry(seed=0, n_per_class=15, n_classes=3, n_bins=12, n_units=10):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    rng.shuffle(labels)
    tuning = rng.normal(0.0, 0.8, size=(n_classes, n_units))
    rate = np.exp(0.2 + tuning[labels])[:, None, :] * np.ones((1, n_bins, 1))
    return rng.poisson(rate).astype(float), labels


def test_whole_trial_stratified_cell_equals_benchmark_evaluation(tmp_path, monkeypatch):
    activity, labels = _planted_entry()
    folds, _ = followup.build_splits(ENTRY["region"], ENTRY["session_key"], ENTRY["level"], labels, (), 0)[
        followup.STRATIFIED]
    identity = followup.block_identity(activity, labels, folds, IMPLEMENTATION)
    delivered = run_info_benchmark.evaluate_block(
        ENTRY["region"], ENTRY["session_key"], ENTRY["level"], activity, labels, ("principal_components",),
        ("linear",), followup.N_SPLITS, 3, CTG_STEP, lambda records: None, IMPLEMENTATION, 0, tmp_path / "benchmark",
        "test_followup_cell_one")
    monkeypatch.setattr(info_decoding, "_decoder_api", info_decoding._decoder_api)
    monkeypatch.setattr(followup.pilot, "split_auc", followup.pilot.split_auc)
    followup.use_same_time_scoring()
    cell = followup.score_decoding(ENTRY, activity, labels, folds, followup.STRATIFIED, "principal_components",
                                   followup.WHOLE, 3, tmp_path / "followup", identity)
    assert cell["status"] == "computed" and delivered[0]["status"] == "computed"
    assert abs(cell["same_time"]["auc"] - delivered[0]["same_time"]["auc"]) < 1e-9
    assert cell["same_time"]["auc"] > 0.6


def test_causal_latent_at_bin_t_ignores_later_bins_and_whole_trial_latent_does_not():
    pytest.importorskip("elephant")
    rng = np.random.default_rng(1)
    train = rng.poisson(2.0, size=(24, 8, 6)).astype(float)
    test = rng.poisson(2.0, size=(6, 8, 6)).astype(float)
    altered = test.copy()
    altered[:, 4:, :] = rng.poisson(6.0, size=altered[:, 4:, :].shape)

    def latent(inputs, causal):
        fit = followup.pilot._fit_gpfa(train, inputs, 3, np.random.default_rng(0), BIN_MS, causal)
        assert fit["status"] == "fitted"
        return fit["latent_test"]

    np.testing.assert_allclose(latent(test, True)[:, :4], latent(altered, True)[:, :4], atol=1e-10)
    assert not np.allclose(latent(test, False)[:, :4], latent(altered, False)[:, :4])


def test_chronological_split_never_places_a_test_trial_in_calibration():
    labels = np.tile(np.arange(3), 20)
    for fraction in (0.667, 0.75):
        folds, info = followup.chronological_split(labels, fraction)
        (calibration, test), = folds
        assert set(calibration.tolist()).isdisjoint(test.tolist())
        assert calibration.max() < test.min()
        np.testing.assert_array_equal(np.concatenate([calibration, test]), np.arange(len(labels)))
        assert info["n_calibration"] == len(calibration) and info["n_test"] == len(test)


def test_chronological_test_set_with_one_class_is_not_estimable_with_counts():
    labels = np.array([0, 1, 2] * 10 + [1] * 10)
    folds, info = followup.chronological_split(labels, 0.75)
    assert folds is None
    assert info["test_class_counts"] == {"1": 10}
    assert "fewer than two classes" in info["reason"]
    labels = np.array([0] * 10 + [1] * 10 + [2, 0, 2, 0])
    folds, info = followup.chronological_split(labels, 0.75)
    assert folds is None and "same classes" in info["reason"]


def test_summary_with_one_patient_is_not_estimable():
    row = followup.estimate_values([0.1, 0.2, 0.3], ["P1", "P1", "P1"])
    assert row["status"] == "not_estimable" and row["n_patients"] == 1 and row["n_entries"] == 3
    assert followup.estimate_values([], [])["status"] == "not_estimable"
    assert followup.estimate_values([0.1, 0.2, 0.3], ["P1", "P2", "P3"])["status"] == "estimable"


def _loaded(region, patient, level, values):
    cells = {followup.cell_id(r, i, s): {"status": "computed", "representation": r, "inference": i, "split": s,
                                          "same_time": {"auc_above_null": v}} for (r, i, s), v in values.items()}
    return {"status": "loaded", "level": level, "region": region, "patient": patient, "session": "ses-1",
            "cells": {"decoding": cells}}


def test_contrasts_average_within_patient_and_pool_regions():
    level, strat = "load1_maintenance", followup.STRATIFIED
    model, reference = "gaussian_process_factor_analysis", "principal_components"
    entries = []
    for patient, (gap_hip, gap_amy) in {"P1": (0.1, 0.3), "P2": (0.2, 0.2), "P3": (0.0, 0.0), "P4": (0.4, 0.0)}.items():
        for region, gap in (("hippocampus", gap_hip), ("amygdala", gap_amy)):
            entries.append(_loaded(region, patient, level, {
                (reference, followup.WHOLE, strat): 0.5, (model, followup.WHOLE, strat): 0.5 + gap,
                (model, followup.CAUSAL, strat): 0.5 + gap / 2}))
    _, contrasts = followup.summarise(entries, (model,), [strat])

    def pick(kind, minuend_inference, subtrahend_representation, region_cell):
        return next(c for c in contrasts if c["kind"] == kind and c["region_cell"] == region_cell
                    and c["minuend"]["inference"] == minuend_inference
                    and c["subtrahend"]["representation"] == subtrahend_representation
                    and c["subtrahend"]["inference"] == followup.WHOLE)

    pooled = pick("candidate_minus_reference", followup.WHOLE, reference, followup.POOLED)
    assert pooled["n_patients"] == 4 and pooled["n_entries"] == 8
    assert pooled["mean"] == pytest.approx(np.mean([0.2, 0.2, 0.0, 0.2]))
    causal = pick("causal_minus_whole_trial", followup.CAUSAL, model, "hippocampus")
    assert causal["mean"] == pytest.approx(-np.mean([0.1, 0.2, 0.0, 0.4]) / 2)


def _stored_artifact(gains):
    model = "gaussian_process_factor_analysis"
    reference = followup.REFERENCES[0]
    level = followup.LEVELS[0]
    entries = []
    for i, gain in enumerate(gains):
        cells = {}
        for representation, value in ((reference, 0.0), (model, gain)):
            cid = followup.cell_id(representation, followup.WHOLE, followup.STRATIFIED)
            cells[cid] = {"representation": representation, "inference": followup.WHOLE,
                          "split": followup.STRATIFIED, "status": "computed",
                          "same_time": {"auc_above_null": value}}
        entries.append({"level": level, "region": "hippocampus", "patient": f"p{i}", "session": "ses-1",
                        "session_key": f"p{i}_ses-1", "status": "loaded", "cells": {"decoding": cells}})
    return {"status": "complete", "statement": followup.STATEMENT, "scope": {
        "candidates": [model], "train_fractions": [], "scores": ["decoding"]}, "entries": entries,
        "absolute": [], "contrasts": []}


def test_summaries_only_recomputes_from_stored_entries_and_reports_a_robust_median(tmp_path):
    existing = tmp_path / "followup.json"
    existing.write_text(json.dumps(_stored_artifact([0.05] * 9 + [5.0])))
    output = tmp_path / "recomputed.json"
    followup.recompute_summaries(existing, output)
    artifact = json.loads(output.read_text())
    rows = [r for r in artifact["contrasts"] if r["kind"] == "candidate_minus_reference"
            and r["region_cell"] == followup.POOLED and r["minuend"]["inference"] == followup.WHOLE
            and r["subtrahend"]["representation"] == followup.REFERENCES[0]]
    assert rows and rows[0]["status"] == "estimable"
    assert rows[0]["median"] == pytest.approx(0.05) and rows[0]["mean"] > 0.5
    assert rows[0]["median_ci_95"][1] < rows[0]["mean"]
    assert artifact["summaries_recomputed"]["source_file"] == "followup.json"
    assert output.with_suffix(".md").exists()
    assert json.loads(existing.read_text())["contrasts"] == []
    with pytest.raises(SystemExit):
        followup.recompute_summaries(existing, existing)
