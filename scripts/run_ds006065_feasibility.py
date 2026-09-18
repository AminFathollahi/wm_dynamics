#!/usr/bin/env python3
"""Inventory the staged ds006065 BIDS metadata without reading signal samples."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_config import data_root
from provenance import canonical_json, git_commit

RESULT_PATH = ROOT / "results" / "ds006065_feasibility.json"
DATASET_ID = "ds006065"
TASK_RE = re.compile(r"(?:^|_)task-([^_]+)")
PHASE_TOKENS = ("phase", "theta", "angle")
BLIND_TOKENS = ("blind", "mask", "sham", "control")
ARTIFACT_TOKENS = ("artifact", "artefact", "bad", "reject", "saturat")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames or [], list(reader)


def _find_dataset_root(root: Path) -> Path | None:
    candidates = [root / DATASET_ID, root / f"{DATASET_ID}-download", root]
    candidates.extend(sorted(path for path in root.glob(f"*{DATASET_ID}*") if path.is_dir()))
    for candidate in candidates:
        if (candidate / "dataset_description.json").is_file():
            return candidate
    return None


def _metadata_values(sidecars: list[Path], key: str) -> list[Any]:
    values = []
    for path in sidecars:
        try:
            value = json.loads(path.read_text()).get(key)
        except json.JSONDecodeError:
            continue
        if value is not None and value not in values:
            values.append(value)
    return values


def _task_name(path: Path) -> str:
    match = TASK_RE.search(path.name)
    return match.group(1) if match else "unspecified"


def _token_fields(fields: list[str], tokens: tuple[str, ...]) -> list[str]:
    return [field for field in fields if any(token in field.lower() for token in tokens)]


def _event_inventory(event_files: list[Path]) -> dict[str, Any]:
    field_counts: Counter[str] = Counter()
    trial_types: Counter[str] = Counter()
    marker_values: dict[str, set[str]] = defaultdict(set)
    per_subject_tasks: dict[str, set[str]] = defaultdict(set)
    n_rows = 0
    for path in event_files:
        fields, rows = _read_tsv(path)
        field_counts.update(fields)
        task = _task_name(path)
        subject = next((part for part in path.parts if part.startswith("sub-")), "unknown")
        per_subject_tasks[subject].add(task)
        n_rows += len(rows)
        for row in rows:
            if row.get("trial_type"):
                trial_types[row["trial_type"]] += 1
            for field in _token_fields(fields, PHASE_TOKENS + BLIND_TOKENS + ARTIFACT_TOKENS):
                value = row.get(field, "")
                if value not in ("", "n/a"):
                    marker_values[field].add(value)

    fields = sorted(field_counts)
    phase_fields = _token_fields(fields, PHASE_TOKENS)
    blind_fields = _token_fields(fields, BLIND_TOKENS)
    artifact_fields = _token_fields(fields, ARTIFACT_TOKENS)
    phase_levels = {field: sorted(marker_values[field]) for field in phase_fields}
    paired = sorted(subject for subject, tasks in per_subject_tasks.items()
                    if {"cl", "clcontrol"}.issubset(tasks))
    return {
        "n_event_files": len(event_files),
        "n_event_rows": n_rows,
        "fields": fields,
        "timing_fields": [field for field in fields if field.lower() in {"onset", "duration", "sample"}],
        "trial_type_counts": dict(sorted(trial_types.items())),
        "phase_fields": phase_fields,
        "phase_levels": phase_levels,
        "blinding_or_control_fields": blind_fields,
        "artifact_marker_fields": artifact_fields,
        "artifact_marker_values": {field: sorted(marker_values[field]) for field in artifact_fields},
        "subjects_with_cl_and_clcontrol": paired,
        "per_subject_tasks": {subject: sorted(tasks) for subject, tasks in sorted(per_subject_tasks.items())},
    }


def assess(data_root_path: Path) -> dict[str, Any]:
    root = Path(data_root_path)
    base = {
        "schema_version": "1.0.0",
        "analysis_id": "ds006065_metadata_feasibility",
        "code_commit": git_commit(ROOT),
        "assessment_type": "BIDS metadata and event-table inventory; raw signal samples were not read",
        "requested_data_root": str(root),
        "dataset_id": DATASET_ID,
    }
    if not root.is_dir():
        return {
            **base,
            "status": "not_inspectable",
            "reason": "configured external data root is not mounted or is unavailable",
            "scientific_interpretation": "no feasibility conclusion is drawn from unavailable local files",
        }
    dataset_root = _find_dataset_root(root)
    if dataset_root is None:
        return {
            **base,
            "status": "not_staged",
            "reason": "no ds006065 directory containing dataset_description.json was found under the configured data root",
            "scientific_interpretation": "no feasibility conclusion is drawn from absent staged files",
        }

    description = json.loads((dataset_root / "dataset_description.json").read_text())
    participant_path = dataset_root / "participants.tsv"
    participant_fields, participant_rows = _read_tsv(participant_path) if participant_path.exists() else ([], [])
    subject_dirs = sorted(path.name for path in dataset_root.glob("sub-*") if path.is_dir())
    sidecars = sorted(dataset_root.glob("sub-*/**/*_ieeg.json"))
    event_files = sorted(dataset_root.glob("sub-*/**/*_events.tsv"))
    electrode_files = sorted(dataset_root.glob("sub-*/**/*_electrodes.tsv"))
    coordsystem_files = sorted(dataset_root.glob("sub-*/**/*_coordsystem.json"))
    signal_files = {
        suffix.lstrip("."): len(list(dataset_root.glob(f"sub-*/**/*{suffix}")))
        for suffix in (".vhdr", ".vmrk", ".eeg", ".edf", ".set", ".fif")
    }
    task_counts = Counter(_task_name(path) for path in sidecars)
    events = _event_inventory(event_files)
    electrode_fields: Counter[str] = Counter()
    electrode_rows = 0
    for path in electrode_files:
        fields, rows = _read_tsv(path)
        electrode_fields.update(fields)
        electrode_rows += len(rows)
    phase_comparison = bool(events["phase_fields"]) and any(
        len(levels) >= 2 for levels in events["phase_levels"].values()
    )
    paired_control = len(events["subjects_with_cl_and_clcontrol"]) >= 2
    evoked_tasks = {task for task in task_counts if task.startswith("ep")}
    return {
        **base,
        "status": "complete",
        "dataset_root": str(dataset_root),
        "dataset": {
            "name": description.get("Name"), "doi": description.get("DatasetDOI"),
            "license": description.get("License"), "bids_version": description.get("BIDSVersion"),
            "dataset_type": description.get("DatasetType"),
        },
        "participants": {
            "participants_tsv_present": participant_path.exists(),
            "participant_fields": participant_fields,
            "participant_ids": [row.get("participant_id") for row in participant_rows if row.get("participant_id")],
            "subject_directories": subject_dirs,
            "n_participants_tsv": len(participant_rows), "n_subject_directories": len(subject_dirs),
        },
        "files": {"signal_files_by_extension": signal_files, "n_ieeg_sidecars": len(sidecars),
                  "tasks_from_ieeg_filenames": dict(sorted(task_counts.items()))},
        "signals": {
            "formats_inferred_from_files": [suffix for suffix, count in signal_files.items() if count],
            "sampling_frequency_hz": _metadata_values(sidecars, "SamplingFrequency"),
            "electrical_stimulation": _metadata_values(sidecars, "ElectricalStimulation"),
            "power_line_frequency_hz": _metadata_values(sidecars, "PowerLineFrequency"),
            "ieeg_reference": _metadata_values(sidecars, "iEEGReference"),
        },
        "anatomy": {
            "n_electrode_files": len(electrode_files), "n_electrode_rows": electrode_rows,
            "electrode_fields": sorted(electrode_fields), "n_coordsystem_files": len(coordsystem_files),
            "coordinate_systems": _metadata_values(coordsystem_files, "iEEGCoordinateSystem"),
            "coordinate_units": _metadata_values(coordsystem_files, "iEEGCoordinateUnits"),
        },
        "events": events,
        "feasibility": {
            "resting_state_scope": "rest tasks are identifiable from task-restpre/restpost/restcontrolpre/restcontrolpost filenames when staged; this is not a working-memory-benefit dataset",
            "person_level_phase_comparison": {
                "status": "candidate" if phase_comparison and paired_control else "not_estimable_from_staged_metadata",
                "requires": "a phase field with at least two observed levels plus within-person cl and clcontrol recordings",
                "phase_field_requirement_met": phase_comparison,
                "within_person_closed_loop_control_requirement_met": paired_control,
            },
            "neural_input_response_estimation": {
                "status": "candidate" if evoked_tasks and len(sidecars) and len(electrode_files) else "not_estimable_from_staged_metadata",
                "requires": "pre/post evoked recordings, readable iEEG, stimulation-event timing, and electrode locations",
                "evoked_tasks": sorted(evoked_tasks),
                "has_ieeg_sidecars": bool(sidecars), "has_electrode_locations": bool(electrode_files),
            },
        },
    }


def main() -> None:
    configured_root = data_root(required=False)
    payload = assess(configured_root if configured_root is not None else Path("/__wm_dynamics_data_root_unset__"))
    RESULT_PATH.write_text(canonical_json(payload))
    print(f"wrote {RESULT_PATH}: {payload['status']}")


if __name__ == "__main__":
    main()
