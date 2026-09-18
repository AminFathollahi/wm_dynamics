#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

DATASET_ID = "ds006848"
RESULT_PATH = ROOT / "results" / "ds006848_feasibility.json"
TASK_RE = re.compile(r"(?:^|_)task-([^_]+)")
CONDITIONS = {"28": "Simultaneous", "40": "Fast", "60": "Fast+delay", "100": "Slow"}
SEQUENTIAL = {"Fast", "Fast+delay", "Slow"}


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames or [], list(reader)


def find_dataset(root: Path) -> Path | None:
    candidates = [root / DATASET_ID, root / f"{DATASET_ID}-download", root]
    candidates.extend(sorted(path for path in root.glob(f"*{DATASET_ID}*") if path.is_dir()))
    return next((path for path in candidates if (path / "dataset_description.json").is_file()), None)


def task_name(path: Path) -> str:
    match = TASK_RE.search(path.name)
    return match.group(1) if match else "unspecified"


def subject_name(path: Path) -> str:
    return next((part for part in path.parts if part.startswith("sub-")), "unknown")


def value_counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(field, "") or "<missing>" for row in rows).items()))


def json_values(paths: list[Path], field: str) -> list[Any]:
    values = []
    for path in paths:
        value = json.loads(path.read_text()).get(field)
        if value is not None and value not in values:
            values.append(value)
    return values


def parse_header(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in {"DataFile", "MarkerFile"}:
                values[key] = value.strip()
    return values


def referenced_state(path: Path) -> str:
    if path.is_symlink():
        return "annex_present" if path.exists() else "annex_missing"
    if path.is_file():
        return "regular_present"
    if any(path.parent.glob(f"{path.name}.*")):
        return "partial"
    return "missing"


def payload_inventory(dataset: Path, headers: list[Path]) -> dict[str, Any]:
    records = []
    for header in headers:
        refs = parse_header(header)
        data_path = header.parent / refs.get("DataFile", header.with_suffix(".eeg").name)
        marker_path = header.parent / refs.get("MarkerFile", header.with_suffix(".vmrk").name)
        records.append({
            "subject": subject_name(header),
            "task": task_name(header),
            "data_state": referenced_state(data_path),
            "marker_state": referenced_state(marker_path),
            "data_path": str(data_path.relative_to(dataset)),
        })
    data_states = Counter(record["data_state"] for record in records)
    marker_states = Counter(record["marker_state"] for record in records)
    task_states: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        task_states[record["task"]][record["data_state"]] += 1
    absent = [record for record in records if record["data_state"] not in {"regular_present", "annex_present"}]
    attributes = (dataset / ".gitattributes").read_text(errors="replace") if (dataset / ".gitattributes").is_file() else ""
    return {
        "datalad_marker_present": (dataset / ".datalad").is_dir(),
        "git_metadata_present": (dataset / ".git").exists(),
        "annex_rules_present": "annex.largefiles" in attributes or "annex.backend" in attributes,
        "annex_queryable": (dataset / ".git").exists(),
        "n_headers": len(headers),
        "data_states": dict(sorted(data_states.items())),
        "marker_states": dict(sorted(marker_states.items())),
        "data_states_by_task": {
            task: dict(sorted(counts.items())) for task, counts in sorted(task_states.items())
        },
        "unavailable_data": absent,
    }


def channel_inventory(paths: list[Path]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    per_file = Counter()
    names: dict[str, set[str]] = defaultdict(set)
    fields = set()
    for path in paths:
        file_fields, rows = read_tsv(path)
        fields.update(file_fields)
        counts = Counter(row.get("type", "<missing>") for row in rows)
        per_file[tuple(sorted(counts.items()))] += 1
        for row in rows:
            channel_type = row.get("type", "<missing>")
            channel_name = row.get("name", "")
            type_counts[channel_type] += 1
            status_counts[f"{channel_type}:{row.get('status', '<missing>')}"] += 1
            names[channel_type].add(channel_name)
    patterns = [
        {"counts": dict(pattern), "n_files": count}
        for pattern, count in sorted(per_file.items(), key=lambda item: item[0])
    ]
    ppg_names = sorted(name for channel_names in names.values() for name in channel_names if "ppg" in name.lower())
    return {
        "n_files": len(paths),
        "fields": sorted(fields),
        "type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "per_recording_type_patterns": patterns,
        "ecg_names": sorted(names.get("ECG", set())),
        "ppg_names": ppg_names,
        "ppg_bids_types": sorted(channel_type for channel_type, channel_names in names.items()
                                 if any("ppg" in name.lower() for name in channel_names)),
    }


def behavior_inventory(paths: list[Path]) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    rows_by_subject = {}
    all_rows = []
    schemas = Counter()
    for path in paths:
        fields, rows = read_tsv(path)
        schemas[tuple(fields)] += 1
        subject = subject_name(path)
        rows_by_subject[subject] = rows
        all_rows.extend(rows)
    outcome_fields = [field for field in ("response", "NCorrect", "partialScore", "triggerCorrect")
                      if all(field in schema for schema in schemas)]
    missing = {
        field: sum(row.get(field, "").strip().lower() in {"", "n/a", "nan"} for row in all_rows)
        for field in outcome_fields
    }
    numeric_ranges = {}
    for field in ("NCorrect", "partialScore"):
        values = [float(row[field]) for row in all_rows if row.get(field, "").strip().lower() not in {"", "n/a", "nan"}]
        if values:
            numeric_ranges[field] = [min(values), max(values)]
    keys = [(row.get("participant_id"), row.get("condition"), row.get("trial")) for row in all_rows]
    return {
        "n_files": len(paths),
        "n_rows": len(all_rows),
        "schemas": [{"fields": list(schema), "n_files": count} for schema, count in sorted(schemas.items())],
        "rows_per_subject": value_counts(
            [{"n": str(len(rows))} for rows in rows_by_subject.values()], "n"
        ),
        "condition_counts": value_counts(all_rows, "condition"),
        "event_id_counts": value_counts(all_rows, "event_id"),
        "outcome_fields": outcome_fields,
        "outcome_missing_counts": missing,
        "outcome_ranges": numeric_ranges,
        "duplicate_subject_condition_trial_keys": len(keys) - len(set(keys)),
    }, rows_by_subject


def event_trials(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    digits = []
    trials = []
    for row in rows:
        trial_type = row.get("trial_type", "")
        if trial_type == "Baseline_2s":
            digits = []
        elif trial_type.startswith("Encoding_DigitValue_"):
            value = row.get("value", "")
            digits.append("0" if value == "10" else value)
        elif row.get("value") in CONDITIONS:
            condition = CONDITIONS[row["value"]]
            trials.append({
                "condition": condition,
                "sequence": "".join(digits[-7:]) if condition in SEQUENTIAL and len(digits) >= 7 else None,
                "onset": float(row["onset"]),
                "retrieval_delay": None,
            })
            digits = []
        elif trial_type == "Digits_Retrieval" and trials and trials[-1]["retrieval_delay"] is None:
            trials[-1]["retrieval_delay"] = float(row["onset"]) - trials[-1]["onset"]
    return trials


def event_inventory(paths: list[Path]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rows_by_subject = {}
    all_rows = []
    trials_by_subject = {}
    schemas = Counter()
    for path in paths:
        fields, rows = read_tsv(path)
        schemas[tuple(fields)] += 1
        subject = subject_name(path)
        rows_by_subject[subject] = rows
        all_rows.extend(rows)
        if task_name(path) == "verbalwm":
            trials_by_subject[subject] = event_trials(rows)
    delays = [trial["retrieval_delay"] for trials in trials_by_subject.values() for trial in trials
              if trial["retrieval_delay"] is not None]
    trial_counts = Counter(trial["condition"] for trials in trials_by_subject.values() for trial in trials)
    return {
        "n_files": len(paths),
        "n_rows": len(all_rows),
        "schemas": [{"fields": list(schema), "n_files": count} for schema, count in sorted(schemas.items())],
        "task_file_counts": dict(sorted(Counter(task_name(path) for path in paths).items())),
        "trial_type_counts": value_counts(all_rows, "trial_type"),
        "retention_condition_counts": dict(sorted(trial_counts.items())),
        "retention_to_retrieval_seconds": {
            "n_with_retrieval": len(delays),
            "n_near_six_seconds": sum(abs(value - 6.0) <= 0.05 for value in delays),
            "minimum": min(delays) if delays else None,
            "median": sorted(delays)[len(delays) // 2] if delays else None,
            "maximum": max(delays) if delays else None,
        },
    }, trials_by_subject


def linkage_inventory(
    behavior: dict[str, list[dict[str, str]]], events: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    exact = []
    cell_mismatches = []
    matched_sequences = Counter()
    event_sequences = Counter()
    for subject in sorted(set(behavior) | set(events)):
        behavior_counts = Counter(row.get("condition") for row in behavior.get(subject, []))
        event_counts = Counter(row["condition"] for row in events.get(subject, []))
        if all(behavior_counts[name] == event_counts[name] for name in CONDITIONS.values()):
            exact.append(subject)
        for condition in CONDITIONS.values():
            if behavior_counts[condition] != event_counts[condition]:
                cell_mismatches.append({
                    "subject": subject,
                    "condition": condition,
                    "behavior_trials": behavior_counts[condition],
                    "event_trials": event_counts[condition],
                })
        for condition in SEQUENTIAL:
            behavior_seq = Counter(
                row.get("sequence", "").zfill(7) for row in behavior.get(subject, [])
                if row.get("condition") == condition
            )
            event_seq = Counter(
                trial["sequence"] for trial in events.get(subject, [])
                if trial["condition"] == condition and trial["sequence"]
            )
            matched_sequences[condition] += sum((behavior_seq & event_seq).values())
            event_sequences[condition] += sum(event_seq.values())
    return {
        "subjects_with_exact_condition_counts": exact,
        "n_subjects_with_exact_condition_counts": len(exact),
        "condition_count_mismatches": cell_mismatches,
        "sequential_event_trials_with_sequence": dict(sorted(event_sequences.items())),
        "sequential_sequences_matched_to_behavior": dict(sorted(matched_sequences.items())),
        "simultaneous_linkage": "ordinal only because digit identities are absent from simultaneous event rows",
        "outcome_linkage": "sequence-level for matched sequential trials; ordinal within complete simultaneous cells",
    }


def participant_inventory(dataset: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    path = dataset / "participants.tsv"
    fields, rows = read_tsv(path) if path.is_file() else ([], [])
    exclusion_fields = [field for field in fields if field.lower().endswith("excluded")]
    exclusions = {
        field: {
            "counts": value_counts(rows, field),
            "excluded_subjects": sorted(row.get("participant_id", "") for row in rows
                                        if row.get(field, "").lower() == "yes"),
        }
        for field in exclusion_fields
    }
    return {
        "participants_tsv_present": path.is_file(),
        "fields": fields,
        "n_rows": len(rows),
        "n_subject_directories": len(list(dataset.glob("sub-*"))),
        "exclusions": exclusions,
    }, rows


def assess(root: Path) -> dict[str, Any]:
    requested = Path(root)
    base = {
        "schema_version": "1.0.0",
        "analysis_id": "ds006848_metadata_feasibility",
        "dataset_id": DATASET_ID,
        "code_commit": git_commit(ROOT),
        "assessment_scope": "BIDS metadata and tabular records; signal payloads were not opened",
        "requested_data_root": str(requested),
    }
    if not requested.is_dir():
        return {**base, "status": "not_inspectable", "reason": "configured data root is unavailable"}
    dataset = find_dataset(requested)
    if dataset is None:
        return {**base, "status": "not_staged", "reason": "ds006848 metadata were not found"}

    description = json.loads((dataset / "dataset_description.json").read_text())
    participants, participant_rows = participant_inventory(dataset)
    behavior, behavior_rows = behavior_inventory(sorted(dataset.glob("sub-*/beh/*_beh.tsv")))
    events, event_trials_by_subject = event_inventory(sorted(dataset.glob("sub-*/eeg/*_events.tsv")))
    channel_paths = sorted(dataset.glob("sub-*/eeg/*_channels.tsv"))
    eeg_sidecars = sorted(dataset.glob("sub-*/eeg/*_eeg.json"))
    headers = sorted(dataset.glob("sub-*/eeg/*_eeg.vhdr"))
    channels = channel_inventory(channel_paths)
    payloads = payload_inventory(dataset, headers)
    linkage = linkage_inventory(behavior_rows, event_trials_by_subject)
    subject_dirs = {path.name for path in dataset.glob("sub-*") if path.is_dir()}
    task_subjects = {subject_name(path) for path in headers if task_name(path) == "verbalwm"}
    rest_subjects = {subject_name(path) for path in headers if task_name(path) == "rest"}
    excluded = {
        field: {row["participant_id"] for row in participant_rows if row.get(field, "").lower() == "yes"}
        for field in ("EEG_excluded", "RS_excluded", "behavior_excluded")
    }
    unavailable_task = {
        record["subject"] for record in payloads["unavailable_data"]
        if record["task"] == "verbalwm"
    }
    payload_ready = task_subjects - unavailable_task
    admitted_payloads = payload_ready - excluded.get("EEG_excluded", set()) - excluded.get("behavior_excluded", set())
    exact_linkage = set(linkage["subjects_with_exact_condition_counts"])
    linkage_ready = admitted_payloads & exact_linkage
    return {
        **base,
        "status": "complete",
        "dataset_root": str(dataset),
        "dataset": {
            "name": description.get("Name"),
            "doi": description.get("DatasetDOI"),
            "license": description.get("License"),
            "bids_version": description.get("BIDSVersion"),
            "dataset_type": description.get("DatasetType"),
        },
        "participants": participants,
        "recording_coverage": {
            "subject_directories": len(subject_dirs),
            "working_memory_subjects": len(task_subjects),
            "rest_subjects": len(rest_subjects),
            "rest_missing_subjects": sorted(subject_dirs - rest_subjects),
            "rest_exclusion_matches_missing": sorted(excluded.get("RS_excluded", set())) == sorted(subject_dirs - rest_subjects),
        },
        "payload_availability": payloads,
        "signals": {
            "n_eeg_sidecars": len(eeg_sidecars),
            "sampling_frequency_hz": json_values(eeg_sidecars, "SamplingFrequency"),
            "eeg_channel_count": json_values(eeg_sidecars, "EEGChannelCount"),
            "ecg_channel_count": json_values(eeg_sidecars, "ECGChannelCount"),
            "misc_channel_count": json_values(eeg_sidecars, "MiscChannelCount"),
            "recording_types": json_values(eeg_sidecars, "RecordingType"),
        },
        "channels": channels,
        "events": events,
        "behavior": behavior,
        "trial_outcome_linkage": linkage,
        "feasibility": {
            "working_memory_autonomic_control": {
                "status": "candidate" if len(linkage_ready) == len(task_subjects) else "partially_staged",
                "n_payload_ready_subjects": len(admitted_payloads),
                "n_linkage_ready_subjects": len(linkage_ready),
                "linkage_ready_subjects": sorted(linkage_ready),
                "n_task_subjects": len(task_subjects),
                "requirements_met": {
                    "trial_level_outcomes": behavior["n_rows"] > 0 and not any(behavior["outcome_missing_counts"].values()),
                    "retention_timing": events["retention_to_retrieval_seconds"]["n_near_six_seconds"] > 0,
                    "ecg_channels": bool(channels["ecg_names"]),
                    "ppg_channels": bool(channels["ppg_names"]),
                    "all_task_payloads_available": len(admitted_payloads) == len(task_subjects),
                    "exact_condition_count_linkage": len(exact_linkage) == len(task_subjects),
                },
            },
            "scope": "observational autonomic nuisance-control and behavioral association; no stimulation or causal contrast",
            "remaining_limits": [
                "complete unavailable or partial working-memory signal payloads",
                "resolve event-count mismatches before participant-level trial analysis",
                "simultaneous trials lack digit identities in the event table",
            ],
        },
    }


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    root = args.data_root or data_root(required=False) or Path("/__wm_dynamics_data_root_unset__")
    payload = assess(root)
    write_result(args.output, payload)
    print(f"wrote {args.output}: {payload['status']}")


if __name__ == "__main__":
    main()
