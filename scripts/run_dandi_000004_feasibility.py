#!/usr/bin/env python3
"""Metadata feasibility census for DANDI 000004."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project_config import dataset_path
from spike_pipeline import resolve_unit_regions

LABEL_CONVENTION = "nwb_hemisphere_prefixed_structure"
EVENT_INTERVALS = (
    ("stim_off_minus_stim_on", "stim_off_time", "stim_on_time"),
    ("delay1_minus_stim_off", "delay1_time", "stim_off_time"),
    ("response_minus_delay1", "response_time", "delay1_time"),
    ("delay2_minus_response", "delay2_time", "response_time"),
    ("delay2_minus_delay1", "delay2_time", "delay1_time"),
)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numeric_summary(values: np.ndarray) -> dict[str, Any] | None:
    if not np.issubdtype(values.dtype, np.number):
        return None
    numeric = values.astype(float)
    finite = np.isfinite(numeric)
    summary: dict[str, Any] = {
        "n_values": int(numeric.size),
        "n_finite": int(finite.sum()),
        "n_nonfinite": int((~finite).sum()),
    }
    if finite.any():
        summary.update({"min": float(numeric[finite].min()), "max": float(numeric[finite].max())})
    return summary


def _field_summary(values: np.ndarray) -> dict[str, Any]:
    summary = {"dtype": str(values.dtype), "n_values": int(values.size)}
    numeric = _numeric_summary(values)
    if numeric is not None:
        summary.update(numeric)
        unique = np.unique(values[np.isfinite(values.astype(float))])
        if unique.size <= 32:
            summary["value_counts"] = {
                str(_json_value(value)): int(count)
                for value, count in Counter(values[np.isfinite(values.astype(float))]).items()
            }
    else:
        counts = Counter(values)
        if len(counts) <= 32:
            summary["value_counts"] = {_text(value): int(count) for value, count in counts.items()}
    return summary


def _interval_summary(later: np.ndarray, earlier: np.ndarray) -> dict[str, Any]:
    values = np.asarray(later - earlier, dtype=float)
    summary = _numeric_summary(values) or {}
    finite = values[np.isfinite(values)]
    summary["n_nonpositive"] = int((finite <= 0).sum())
    if finite.size:
        summary["quantiles"] = {
            str(q): float(value)
            for q, value in zip((0.01, 0.05, 0.5, 0.95, 0.99), np.quantile(finite, (0.01, 0.05, 0.5, 0.95, 0.99)))
        }
    return summary


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_files(data_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    complete = sorted(data_dir.rglob("*.nwb"))
    incomplete = []
    for directory in sorted(data_dir.rglob("*.nwb.dandidownload")):
        incomplete.append({
            "path": _relative(directory, data_dir),
            "entries": [
                {"path": _relative(path, data_dir), "size_bytes": path.stat().st_size}
                for path in sorted(directory.rglob("*")) if path.is_file()
            ],
        })
    return complete, incomplete


def _combined_interval_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [summary for summary in summaries if summary["n_finite"]]
    return {
        "n_values": sum(summary["n_values"] for summary in summaries),
        "n_finite": sum(summary["n_finite"] for summary in summaries),
        "n_nonfinite": sum(summary["n_nonfinite"] for summary in summaries),
        "n_nonpositive": sum(summary["n_nonpositive"] for summary in summaries),
        **({"min": min(summary["min"] for summary in finite),
            "max": max(summary["max"] for summary in finite)} if finite else {}),
    }


def inspect_file(path: Path, data_dir: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        trials = handle["intervals/trials"]
        fields = {name: trials[name][:] for name in trials}
        trial_summary = {name: _field_summary(values) for name, values in fields.items()}
        intervals = {
            name: _interval_summary(fields[later], fields[earlier])
            for name, later, earlier in EVENT_INTERVALS
            if later in fields and earlier in fields
        }
        regions = resolve_unit_regions(handle, LABEL_CONVENTION)
        electrode_locations = [_text(value) for value in handle[
            "general/extracellular_ephys/electrodes/location"
        ][:]]
        subject = _text(handle["general/subject/subject_id"][()])
        identifier = _text(handle["identifier"][()])
    return {
        "path": _relative(path, data_dir),
        "size_bytes": path.stat().st_size,
        "participant": subject,
        "nwb_identifier": identifier,
        "n_trials": int(next(iter(fields.values())).size),
        "n_units": int(regions["region"].size),
        "trial_fields": trial_summary,
        "named_inter_event_intervals_seconds": intervals,
        "raw_electrode_location_counts": dict(sorted(Counter(electrode_locations).items())),
        "resolved_unit_region_counts": dict(sorted(Counter(regions["region"]).items())),
        "resolved_unit_hemisphere_counts": dict(sorted(Counter(regions["hemisphere"]).items())),
        "resolved_unit_raw_location_counts": dict(sorted(Counter(regions["raw"]).items())),
    }


def build_feasibility(data_dir: Path) -> dict[str, Any]:
    started = perf_counter()
    complete, incomplete = discover_files(data_dir)
    inspected, failures = [], []
    for path in complete:
        try:
            inspected.append(inspect_file(path, data_dir))
        except (OSError, KeyError, RuntimeError, ValueError) as exc:
            failures.append({"path": _relative(path, data_dir), "error": str(exc)})

    participants = sorted({row["participant"] for row in inspected})
    participant_directories = sorted(path.name.removeprefix("sub-") for path in data_dir.glob("sub-*"))
    identifiers = sorted({row["nwb_identifier"] for row in inspected})
    all_fields = Counter()
    field_presence = Counter()
    unit_regions, unit_hemispheres, electrode_locations = Counter(), Counter(), Counter()
    outcomes = defaultdict(Counter)
    task_values = defaultdict(Counter)
    interval_rows = defaultdict(list)
    for row in inspected:
        for field, summary in row["trial_fields"].items():
            field_presence[field] += 1
            all_fields[field] += summary["n_values"]
            if field in {"new_old_labels_recog", "response_value"}:
                outcomes[field].update(summary.get("value_counts", {}))
            if "value_counts" in summary:
                task_values[field].update(summary["value_counts"])
        unit_regions.update(row["resolved_unit_region_counts"])
        unit_hemispheres.update(row["resolved_unit_hemisphere_counts"])
        electrode_locations.update(row["raw_electrode_location_counts"])
        for name, summary in row["named_inter_event_intervals_seconds"].items():
            interval_rows[name].append(summary)

    elapsed = perf_counter() - started
    return {
        "analysis_id": "dandi_000004_metadata_feasibility",
        "schema_version": "1.0.0",
        "status": "complete",
        "dataset": "dandi_000004",
        "data_directory": str(data_dir),
        "label_convention": LABEL_CONVENTION,
        "runtime_seconds": elapsed,
        "file_inventory": {
            "n_complete_nwb_files_discovered": len(complete),
            "n_complete_nwb_files_inspected": len(inspected),
            "n_complete_nwb_files_dropped": len(failures),
            "complete_nwb_files": [
                {"path": row["path"], "size_bytes": row["size_bytes"]} for row in inspected
            ],
            "failed_complete_nwb_files": failures,
            "n_incomplete_download_directories": len(incomplete),
            "incomplete_download_directories": incomplete,
        },
        "identity_inventory": {
            "n_participant_directories": len(participant_directories),
            "participant_directories": participant_directories,
            "n_unique_participants": len(participants),
            "participants": participants,
            "participant_directories_without_complete_nwb": sorted(set(participant_directories) - set(participants)),
            "n_unique_nwb_identifiers": len(identifiers),
            "nwb_identifiers": identifiers,
            "n_unique_nominal_participant_date_stems": len({
                row["path"].split("_obj-", 1)[0] for row in inspected
            }),
            "nominal_participant_date_stem_collisions": {
                stem: count for stem, count in sorted(Counter(
                    row["path"].split("_obj-", 1)[0] for row in inspected
                ).items()) if count > 1
            },
        },
        "trial_inventory": {
            "fields": sorted(field_presence),
            "file_presence_count_by_field": dict(sorted(field_presence.items())),
            "total_values_by_field": dict(sorted(all_fields.items())),
            "trial_outcome_value_counts": {
                field: dict(sorted(counts.items())) for field, counts in sorted(outcomes.items())
            },
            "value_counts_for_low_cardinality_fields": {
                field: dict(sorted(counts.items())) for field, counts in sorted(task_values.items())
            },
            "named_inter_event_interval_file_summaries_seconds": dict(sorted(interval_rows.items())),
            "named_inter_event_interval_corpus_summaries_seconds": {
                name: _combined_interval_summary(summaries)
                for name, summaries in sorted(interval_rows.items())
            },
        },
        "unit_inventory": {
            "total_units": int(sum(unit_regions.values())),
            "resolved_unit_region_counts": dict(sorted(unit_regions.items())),
            "resolved_unit_hemisphere_counts": dict(sorted(unit_hemispheres.items())),
            "raw_electrode_location_counts": dict(sorted(electrode_locations.items())),
            "raw_electrode_locations_with_hemisphere_prefix": {
                label: count for label, count in sorted(electrode_locations.items())
                if label.lower().startswith(("left ", "right "))
            },
        },
        "zero_drop_audit": {
            "complete_files_excluded": len(failures),
            "exclusion_reasons": failures,
            "frozen_scientific_observable_runs": 0,
            "pooled_region_analyses": 0,
            "maintenance_interval_inferences": 0,
        },
        "scope": {
            "task_interpretation": "recognition-task metadata only; timing differences are descriptive and include order-violation counts without assigning a maintenance construct",
            "next_step": "A frozen observable may analyze hippocampus and amygdala separately after defining phase-specific timing and behavioral-validity gates.",
        },
        "files": inspected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "dandi_000004_feasibility.json")
    args = parser.parse_args()
    data_dir = args.data_dir or dataset_path("dandi_000004")
    result = build_feasibility(data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "complete_inspected": result["file_inventory"]["n_complete_nwb_files_inspected"],
        "complete_dropped": result["file_inventory"]["n_complete_nwb_files_dropped"],
        "participants": result["identity_inventory"]["n_unique_participants"],
        "nwb_identifiers": result["identity_inventory"]["n_unique_nwb_identifiers"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
