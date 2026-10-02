from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_representation_benchmark_co_primary as mod  # noqa: E402

PCA, DPCA, FA = "principal_components", "demixed_principal_components", "factor_analysis"
SPEC = {s["name"]: s for s in mod.INPUT_SPECS}
LEVELS = {"load1": mod.LOAD1, "load3": mod.LOAD3}


def loaded(name, payload, status="complete"):
    return {"spec": SPEC[name], "path": Path(f"/nonexistent/{name}.json"), "payload": payload,
            "status": status if payload is not None else "missing"}


def unit_inputs(patient_entries, improvement, status_load1="complete"):
    """patient_entries: {patient: [entry suffixes]}; the candidate beats both references by
    improvement(patient_index, entry_index) on both scores."""
    payloads = {}
    for tag, level in LEVELS.items():
        sessions = [{"patient": p, "session": s} for p, entries in patient_entries.items() for s in entries]
        decoding = {"status": "complete", "sessions": sessions, "records": []}
        reconstruction = {"status": "complete", "sessions": sessions, "records": []}
        for i, (patient, entries) in enumerate(patient_entries.items()):
            for j, suffix in enumerate(entries):
                session = f"{patient}_{suffix}"
                for candidate, bump in ((PCA, 0.0), (DPCA, 0.0), (FA, improvement(i, j))):
                    decoding["records"].append({
                        "candidate": candidate, "corpus": "hippocampus", "level": level, "decoder": "linear",
                        "session": session, "status": "computed", "same_time": {"auc_above_null": 0.02 + bump}})
                    reconstruction["records"].append({
                        "candidate": candidate, "region": "hippocampus", "level": level, "session": session,
                        "status": "computed", "bits_per_spike": 0.1 + 2 * bump})
        payloads[tag] = (decoding, reconstruction)
    return [
        loaded("unit_decoding_load1", payloads["load1"][0], status_load1),
        loaded("unit_decoding_load3", payloads["load3"][0]),
        loaded("unit_cosmoothing_load1", payloads["load1"][1]),
        loaded("unit_cosmoothing_load3", payloads["load3"][1]),
        loaded("unit_decoding_switching_and_transformer_load1", None),
        loaded("unit_decoding_switching_and_transformer_load3", None),
        loaded("unit_cosmoothing_time_contrastive_load1", None),
        loaded("unit_cosmoothing_time_contrastive_load3", None),
        loaded("field_potentials", None),
        loaded("field_potentials_time_contrastive", None),
    ]


def row(artifact, cell, candidate=FA, reference=PCA, level=None):
    key = "co_primary" if level is None else "paired_differences"
    for r in artifact[key]:
        if (r["region_cell"], r["candidate"], r["reference"]) == (cell, candidate, reference) and (
                level is None or r["level"] == level):
            return r
    raise AssertionError("row absent")


def test_planted_improvement_meets_the_rule_at_both_loads():
    entries = {f"P{i}": ["ses-1"] for i in range(6)}
    artifact = mod.build_artifact(unit_inputs(entries, lambda i, j: 0.05 + 0.001 * i))
    result = row(artifact, "hippocampus")
    assert result["status"] == "estimable"
    assert result["co_primary"] is True
    assert [lv["meets_interval_rule"] for lv in result["levels"]] == [True, True]
    paired = row(artifact, "hippocampus", level=mod.LOAD1)
    assert paired["decoding_difference"]["ci_95"][0] > 0 and paired["reconstruction_difference"]["ci_95"][0] > 0
    assert paired["decoding_difference"]["n_patients"] == 6


def test_one_patient_cell_is_not_estimable_and_carries_no_flag():
    artifact = mod.build_artifact(unit_inputs({"P0": ["ses-1", "ses-2", "ses-3"]}, lambda i, j: 0.05))
    paired = row(artifact, "hippocampus", level=mod.LOAD1)
    assert paired["status"] == "not_estimable"
    assert paired["decoding_difference"]["ci_95"] is None and paired["decoding_difference"]["sign_flip_p"] is None
    assert paired["meets_interval_rule"] is None
    assert row(artifact, "hippocampus")["co_primary"] is None
    assert paired["decoding_difference"]["n_patients"] == 1 and paired["decoding_difference"]["n_entries"] == 3


def test_incomplete_input_marks_only_the_cells_that_depend_on_it():
    entries = {f"P{i}": ["ses-1"] for i in range(5)}
    artifact = mod.build_artifact(unit_inputs(entries, lambda i, j: 0.05, status_load1="running"))
    assert artifact["status"] == "inputs_incomplete"
    assert "unit_decoding_load1" in artifact["scope"]["incomplete_inputs"]
    assert row(artifact, "hippocampus", level=mod.LOAD1)["status"] == "inputs_incomplete"
    assert row(artifact, "hippocampus", level=mod.LOAD1)["meets_interval_rule"] is None
    assert row(artifact, "hippocampus", level=mod.LOAD3)["status"] == "estimable"
    assert row(artifact, "hippocampus")["co_primary"] is None
    assert row(artifact, "hippocampus")["status"] == "inputs_incomplete"


def test_missing_file_is_recorded_as_missing(tmp_path):
    result = mod.load_input(SPEC["field_potentials"], tmp_path / "absent.json")
    assert result["status"] == "missing" and not mod.is_complete(result)


def test_means_are_exact_patient_averages_of_paired_differences():
    entries = {"P0": ["ses-1", "ses-2"], "P1": ["ses-1"], "P2": ["ses-1", "ses-2", "ses-3"], "P3": ["ses-1"]}
    bumps = {("P0", 0): 0.10, ("P0", 1): 0.30, ("P1", 0): 0.05, ("P2", 0): 0.01, ("P2", 1): 0.02,
             ("P2", 2): 0.06, ("P3", 0): 0.07}
    names = list(entries)
    artifact = mod.build_artifact(unit_inputs(entries, lambda i, j: bumps[(names[i], j)]))
    expected = np.mean([np.mean([0.10, 0.30]), 0.05, np.mean([0.01, 0.02, 0.06]), 0.07])
    paired = row(artifact, "hippocampus", level=mod.LOAD1)
    assert abs(paired["decoding_difference"]["mean"] - expected) < 1e-12
    assert abs(paired["reconstruction_difference"]["mean"] - 2 * expected) < 1e-12
    assert paired["decoding_difference"]["n_entries"] == 7 and paired["decoding_difference"]["n_patients"] == 4


def test_estimate_adds_a_patient_median_beside_an_unchanged_mean():
    from statistics import bootstrap_ci

    patients = [f"p{i}" for i in range(10)]
    values = [0.1] * 9 + [10.0]
    result = mod.estimate(values, patients)
    expected_mean = bootstrap_ci(np.asarray(values), np.mean, n_boot=mod.N_BOOT, rng=np.random.default_rng(mod.SEED))
    assert [result["mean"], *result["ci_95"]] == list(expected_mean)
    assert result["median"] == 0.1
    assert result["median_ci_95"][0] <= 0.1 <= result["median_ci_95"][1]
    assert result["median_ci_95"][1] < 1.0 < result["mean"]


def test_estimate_with_too_few_patients_still_reports_the_median():
    result = mod.estimate([0.2, 0.4, 3.0], ["p", "p", "q"])
    assert result["status"] == "not_estimable"
    assert result["median"] == np.median([np.mean([0.2, 0.4]), 3.0]) and result["median_ci_95"] is None
