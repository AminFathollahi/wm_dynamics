#!/usr/bin/env python3
"""Measure DANDI 000469 content stability with crossnobis distance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import crossnobis_content_matrix, crossnobis_decay_timescale  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import (  # noqa: E402
    FrozenPSTHTransform,
    MIN_SESSION_ACCURACY,
    build_psth,
    load_spike_times,
    low_rate_unit_mask,
)

BIN_MS = 100
WINDOW_S = 2.3
N_SPLITS = 5
MIN_UNITS = 15


def data_directory() -> Path:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    data_root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not data_root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    return Path(data_root) / config["datasets"]["dandi_000469"]["local_path"]


def analyze_patient(path: Path, seed: int) -> dict:
    with h5py.File(path, "r") as handle:
        spikes = load_spike_times(handle)
        trials = handle["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        labels = trials["loadsEnc1_PicIDs"][:].astype(int)
        accuracy = trials["response_accuracy"][:].astype(bool)
        onsets = trials["timestamps_Maintenance"][:]
    unit_mask = low_rate_unit_mask(spikes, onsets, WINDOW_S)
    spikes = [unit for unit, keep in zip(spikes, unit_mask) if keep]
    if len(spikes) < MIN_UNITS or float(np.mean(accuracy)) < MIN_SESSION_ACCURACY:
        return {"status": "excluded", "reason": "prospective unit-count or accuracy QC failed"}
    keep = loads == 1
    labels, onsets = labels[keep], onsets[keep]
    counts = [int(np.sum(labels == label)) for label in np.unique(labels)]
    if len(counts) < 3 or min(counts) < N_SPLITS:
        return {"status": "excluded", "reason": "repeated-item counts cannot support five folds"}
    rates = build_psth(spikes, onsets, BIN_MS, 0, WINDOW_S)
    counts = rates * (BIN_MS / 1000.0)
    stabilized = 2.0 * np.sqrt(np.maximum(counts, 0.0) + 3.0 / 8.0)
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    matrices = []
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        transform = FrozenPSTHTransform().fit(stabilized[train])
        states = transform.transform(stabilized).transpose(0, 2, 1)
        result = crossnobis_content_matrix(states, labels, [(train, test)])
        if result["status"] == "estimable":
            matrices.append(np.asarray(result["matrix"], dtype=float))
    if len(matrices) < N_SPLITS:
        return {"status": "not_estimable", "reason": "at least one cross-validation fold failed"}
    matrix = np.mean(matrices, axis=0)
    timescale = crossnobis_decay_timescale(matrix, BIN_MS / 1000.0)
    return {
        "status": "estimable" if timescale["status"] == "estimable" else "not_estimable",
        "reason": timescale.get("reason"),
        "n_trials": int(len(labels)), "n_units": int(len(spikes)),
        "n_items": int(len(np.unique(labels))),
        "crossnobis_matrix": matrix.tolist(),
        "mean_diagonal_distance": float(np.mean(np.diag(matrix))),
        "mean_off_diagonal_distance": float(np.mean(matrix[~np.eye(len(matrix), dtype=bool)])),
        "decay": timescale,
    }


def main() -> None:
    patients = {}
    for path in sorted(data_directory().glob("sub-*/sub-*_ses-2_ecephys+image.nwb")):
        patient = path.parent.name
        print(f"crossnobis {patient}", flush=True)
        patients[patient] = analyze_patient(path, 20260802 + int(patient.split("-")[1]))
    output = {
        "schema_version": "1.0.0", "analysis_id": "crossnobis_content_dandi000469",
        "dataset": "DANDI 000469", "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "measurement": (
            "five-fold crossnobis item-identity matrix on unsmoothed Anscombe-transformed counts; "
            "fold-frozen scaling and training-only shrinkage precision"
        ),
        "patients": patients,
        "n_interpretable_patients": int(sum(
            row.get("status") == "estimable" for row in patients.values()
        )),
    }
    destination = ROOT / "results" / "crossnobis_content_000469.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"output": str(destination), "n_interpretable": output["n_interpretable_patients"]}, indent=2))


if __name__ == "__main__":
    main()
