from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_ds006065_feasibility import assess  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_assess_marks_unavailable_root_as_not_inspectable(tmp_path):
    payload = assess(tmp_path / "missing")
    assert payload["status"] == "not_inspectable"
    assert "scientific_interpretation" in payload


def test_assess_inventories_bids_metadata_without_reading_signal_samples(tmp_path):
    dataset = tmp_path / "ds006065"
    _write(dataset / "dataset_description.json", json.dumps({
        "Name": "TSS_iEEG", "BIDSVersion": "1.8", "DatasetType": "raw", "License": "CC0",
    }))
    _write(dataset / "participants.tsv", "participant_id\nsub-p01\nsub-p02\n")
    for subject in ("sub-p01", "sub-p02"):
        for task in ("cl", "clcontrol", "eppre", "eppost", "restpre"):
            stem = dataset / subject / "ieeg" / f"{subject}_task-{task}_ieeg"
            _write(stem.with_suffix(".vhdr"), "Brain Vision Data Exchange Header File Version 1.0")
            _write(stem.with_suffix(".eeg"), "not opened by assessment")
            _write(stem.with_suffix(".json"), json.dumps({
                "SamplingFrequency": 500, "ElectricalStimulation": True,
                "PowerLineFrequency": 60, "iEEGReference": "bipolar",
            }))
            _write(stem.with_name(stem.name.replace("_ieeg", "_events")).with_suffix(".tsv"),
                   "onset\tduration\ttrial_type\ttheta_phase\tblinded_condition\tartifact_flag\n"
                   "0\t1\tSTIM\t0\tclosed\t0\n1\t1\tSTIM\t180\tcontrol\t0\n")
        _write(dataset / subject / "ieeg" / f"{subject}_electrodes.tsv",
               "name\tx\ty\tz\nA1\t1\t2\t3\n")
        _write(dataset / subject / "ieeg" / f"{subject}_coordsystem.json",
               json.dumps({"iEEGCoordinateSystem": "MNI152NLin2009cAsym", "iEEGCoordinateUnits": "mm"}))

    payload = assess(tmp_path)

    assert payload["status"] == "complete"
    assert payload["participants"]["n_participants_tsv"] == 2
    assert payload["files"]["tasks_from_ieeg_filenames"]["cl"] == 2
    assert payload["signals"]["formats_inferred_from_files"] == ["vhdr", "eeg"]
    assert payload["events"]["phase_levels"]["theta_phase"] == ["0", "180"]
    assert payload["events"]["subjects_with_cl_and_clcontrol"] == ["sub-p01", "sub-p02"]
    assert payload["feasibility"]["person_level_phase_comparison"]["status"] == "candidate"
    assert payload["feasibility"]["neural_input_response_estimation"]["status"] == "candidate"
