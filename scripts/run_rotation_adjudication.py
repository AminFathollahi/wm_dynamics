"""Aggregate held-out rotation scores and matched stationary-code floors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from drift_dynamics import planted_rotation_recovery  # noqa: E402
from spike_pipeline import ANATOMICAL_REGIONS  # noqa: E402

PLANTED_ROTATION_RATES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def summarize(records: list[tuple[str, dict[str, Any]]], rng: np.random.Generator) -> dict[str, Any]:
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for entity, result in records:
        by_entity.setdefault(entity, []).append(result)
    entity_rows = {}
    for entity, rows in sorted(by_entity.items()):
        entity_rows[entity] = {
            "observed_rotation_rate": float(np.mean([
                row["mean_axis_rotation_rate_radians_per_second"] for row in rows
            ])),
            "stationary_floor_rotation_rate": float(np.mean([
                row["stationary_code_floor"]["mean_apparent_rotation_rate_radians_per_second"]
                for row in rows
            ])),
            "m1_minus_m0_nats_per_observation": float(np.mean([
                row["m1_minus_m0_nats_per_observation"] for row in rows
            ])),
            "m3_minus_m2_nats_per_transition": float(np.mean([
                row["m3_minus_m2_nats_per_transition"] for row in rows
            ])),
            "counter_rotation_accuracy_recovery": float(np.mean([
                row["counter_rotation"]["accuracy_recovery"] for row in rows
            ])),
            "real_decoding_accuracy": float(np.mean([
                row["stationary_code_floor"].get("real_time_resolved_axis_accuracy", np.nan)
                for row in rows
            ])),
            "stationary_decoding_accuracy": float(np.mean([
                row["stationary_code_floor"].get("stationary_time_resolved_axis_accuracy", np.nan)
                for row in rows
            ])),
            "real_residual_snr": float(np.mean([
                row["stationary_code_floor"].get("real_residual_snr", np.nan) for row in rows
            ])),
            "stationary_residual_snr": float(np.mean([
                row["stationary_code_floor"].get("stationary_residual_snr", np.nan) for row in rows
            ])),
        }
    differences = np.asarray([
        row["observed_rotation_rate"] - row["stationary_floor_rotation_rate"]
        for row in entity_rows.values()
    ])
    counter = np.asarray([
        row["counter_rotation_accuracy_recovery"] for row in entity_rows.values()
    ])
    bootstrap_difference = np.mean(
        differences[rng.integers(0, len(differences), size=(5000, len(differences)))], axis=1
    )
    bootstrap_counter = np.mean(
        counter[rng.integers(0, len(counter), size=(5000, len(counter)))], axis=1
    )
    supported = bool(
        np.percentile(bootstrap_difference, 2.5) > 0.0
        and np.percentile(bootstrap_counter, 2.5) > 0.0
    )
    wrong_direction_floor = bool(np.percentile(bootstrap_difference, 97.5) < 0.0)
    calibration_available = bool(all(
        np.isfinite(row[field])
        for row in entity_rows.values()
        for field in (
            "real_decoding_accuracy", "stationary_decoding_accuracy",
            "real_residual_snr", "stationary_residual_snr",
        )
    ))
    return {
        "n_entities": len(entity_rows),
        "n_folds": len(records),
        "entities": entity_rows,
        "observed_minus_stationary_floor_mean": float(np.mean(differences)),
        "observed_minus_stationary_floor_patient_bootstrap_interval_95": list(map(
            float, np.percentile(bootstrap_difference, [2.5, 97.5])
        )),
        "counter_rotation_accuracy_recovery_mean": float(np.mean(counter)),
        "counter_rotation_accuracy_recovery_patient_bootstrap_interval_95": list(map(
            float, np.percentile(bootstrap_counter, [2.5, 97.5])
        )),
        "deterministic_rotation_supported": supported,
        "stationary_floor_comparison_status": (
            "non_identified" if wrong_direction_floor or not calibration_available else "identified"
        ),
        "stationary_floor_non_identification_reason": (
            "real data rotate significantly less than the matched stationary simulation"
            if wrong_direction_floor else (
                None if calibration_available else
                "decoding-accuracy and residual-SNR calibration diagnostics are unavailable"
            )
        ),
        "floor_calibration": {
            "real_decoding_accuracy_mean": float(np.nanmean([
                row["real_decoding_accuracy"] for row in entity_rows.values()
            ])),
            "stationary_decoding_accuracy_mean": float(np.nanmean([
                row["stationary_decoding_accuracy"] for row in entity_rows.values()
            ])),
            "real_residual_snr_mean": float(np.nanmean([
                row["real_residual_snr"] for row in entity_rows.values()
            ])),
            "stationary_residual_snr_mean": float(np.nanmean([
                row["stationary_residual_snr"] for row in entity_rows.values()
            ])),
        },
        "decision_rule": "observed rotation exceeds the matched stationary floor and counter-rotation recovery is positive, both with patient-bootstrap intervals above zero",
    }


def planted_recovery(
    records: list[tuple[str, dict[str, Any]]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Estimate cohort power over planted rates at every fold's design and SNR."""
    rate_rows: dict[str, Any] = {}
    detectable = None
    for rate in PLANTED_ROTATION_RATES:
        by_entity: dict[str, list[float]] = {}
        for entity, row in records:
            floor = row["stationary_code_floor"]
            snr = floor.get("real_residual_snr")
            if snr is None or not np.isfinite(snr):
                continue
            recovered = planted_rotation_recovery(
                int(floor["matched_train_trials"]), int(floor["matched_test_trials"]),
                int(floor["matched_time_bins"]), int(floor["matched_dimensions"]),
                int(floor["matched_classes"]), float(row["_dt"]), float(snr), float(rate), rng,
            )
            by_entity.setdefault(entity, []).append(
                recovered["counter_rotation_accuracy_recovery"]
            )
        entity_values = np.asarray([
            np.mean(values) for values in by_entity.values()
        ], dtype=float)
        if not len(entity_values):
            rate_rows[str(rate)] = {"status": "not_estimable"}
            continue
        draws = np.mean(entity_values[
            rng.integers(0, len(entity_values), size=(5000, len(entity_values)))
        ], axis=1)
        interval = list(map(float, np.percentile(draws, [2.5, 97.5])))
        rate_rows[str(rate)] = {
            "status": "estimable", "n_entities": len(entity_values),
            "mean_counter_rotation_accuracy_recovery": float(np.mean(entity_values)),
            "patient_bootstrap_interval_95": interval,
        }
        if rate > 0.0 and detectable is None and interval[0] > 0.0:
            detectable = float(rate)
    return {
        "rates": rate_rows,
        "minimum_detectable_rotation_rate_radians_per_second": detectable,
        "criterion": "smallest planted rate whose patient-bootstrap recovery-gain interval is above zero",
        "status": "identified" if detectable is not None else "non_identified",
    }


def records_from_sessions(
    sessions: dict[str, Any], name: str
) -> list[tuple[str, dict[str, Any]]]:
    """Build (entity, rotation_comparison) records from one sessions-keyed block."""
    records: list[tuple[str, dict[str, Any]]] = []
    for entity, value in sorted(sessions.items()):
        if value.get("status") != "complete":
            continue
        independent_entity = entity.split("_ses-")[0] if name == "DANDI 000574" else entity
        for fold in value["folds"]:
            if "rotation_comparison" not in fold:
                raise SystemExit(f"{name} has not been rerun with the rotation comparison")
            rotation = dict(fold["rotation_comparison"])
            rotation["_dt"] = float(value.get("dt", value.get("bin_ms", 100) / 1000.0))
            records.append((independent_entity, rotation))
    return records


def adjudicate_rotation(
    name: str, records: list[tuple[str, dict[str, Any]]], seed_index: int
) -> dict[str, Any]:
    summary = summarize(records, np.random.default_rng(20260801 + seed_index))
    recovery = planted_recovery(records, np.random.default_rng(20261801 + seed_index))
    summary["planted_rotation_recovery"] = recovery
    if (
        summary["stationary_floor_comparison_status"] != "identified"
        or recovery["status"] != "identified"
    ):
        summary["deterministic_rotation_supported"] = None
        summary["rotation_verdict"] = "non_identified"
    elif summary["deterministic_rotation_supported"]:
        summary["rotation_verdict"] = "deterministic_rotation_supported"
    else:
        summary["rotation_verdict"] = (
            "no_deterministic_rotation_above_"
            f"{recovery['minimum_detectable_rotation_rate_radians_per_second']}_radians_per_second_detected"
        )
    return summary


def run_region_stratified() -> dict[str, Any]:
    """Re-run the rotation-floor adjudication per anatomical region on DANDI 000469 only
    -- 000574 and Miller's stationary floors are already invalid and are
    not re-run per region.
    """
    artifact = json.loads((ROOT / "results" / "region_stratified_drift_000469.json").read_text())
    regions: dict[str, Any] = {}
    for seed_index, region in enumerate(ANATOMICAL_REGIONS):
        sessions = artifact["regions"][region]["sessions"]
        records = records_from_sessions(sessions, "DANDI 000469")
        if not records:
            regions[region] = {
                "status": "non_identified",
                "reason": "no complete region-session fold with a rotation_comparison in the region-stratified refit",
            }
            continue
        summary = adjudicate_rotation(f"DANDI 000469 :: {region}", records, seed_index)
        summary["status"] = "estimable"
        regions[region] = summary
    return regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-stratified", action="store_true")
    args = parser.parse_args()

    destination = ROOT / "results" / "rotation_estimator_floor.json"
    if args.region_stratified:
        region_stratified = run_region_stratified()
        output = json.loads(destination.read_text())
        output["region_stratified_000469"] = region_stratified
        destination.write_text(canonical_json(output))

        crack_path = ROOT / "results" / "crack_register.json"
        cracks = json.loads(crack_path.read_text())
        entries = cracks["entries"] if isinstance(cracks, dict) else cracks
        estimable = [
            region for region, entry in region_stratified.items()
            if entry.get("status") == "estimable"
        ]
        detectable = {
            region: entry["planted_rotation_recovery"].get(
                "minimum_detectable_rotation_rate_radians_per_second"
            )
            for region, entry in region_stratified.items()
            if entry.get("status") == "estimable"
        }
        entries.append({
            "artifact": "results/rotation_estimator_floor.json",
            "crack_id": "region_resolved_rotation_floor",
            "trigger": (
                "Re-run the rotation estimator floor per anatomical region on "
                "DANDI 000469 only, where the floor was at least calibrated at the pooled level."
            ),
            "chase": (
                "Reused summarize/planted_recovery unchanged against each region's session block "
                "from region_stratified_drift_000469.json; no new estimator was written."
            ),
            "resolution": (
                f"{len(estimable)}/5 regions estimable. Minimum detectable planted rotation rate "
                f"per estimable region (rad/s): {detectable}. Does not close the rotation crack: "
                f"a per-region floor at a coarser detection bound (or non-identified) is expected "
                f"given the reduced per-region unit count, not evidence for or against rotation."
            ),
            "status": "resolved_region_resolved_extension_not_closing_rotation_crack",
        })
        crack_path.write_text(canonical_json(cracks))
        print(json.dumps({"output": str(destination), "regions": region_stratified}, indent=2))
        return

    configurations = (
        ("DANDI 000469", "human_drift_spine_000469.json", "sessions"),
        ("DANDI 000574", "human_drift_spine_000574.json", "sessions"),
        ("Miller N-back", "miller_drift_spine.json", "patients"),
    )
    datasets = {}
    for index, (name, filename, container) in enumerate(configurations):
        artifact = json.loads((ROOT / "results" / filename).read_text())
        if container == "sessions":
            records = records_from_sessions(artifact[container], name)
        else:
            records = []
            for entity, value in sorted(artifact[container].items()):
                if value.get("status") != "complete":
                    continue
                fit = value["drift"]
                for fold in fit["folds"]:
                    if "rotation_comparison" not in fold:
                        raise SystemExit(f"{name} has not been rerun with the rotation comparison")
                    rotation = dict(fold["rotation_comparison"])
                    rotation["_dt"] = float(fit.get("dt", value.get("bin_ms", 100) / 1000.0))
                    records.append((entity, rotation))
        datasets[name] = adjudicate_rotation(name, records, index)
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "rotation_estimator_floor",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "frozen_scaling_audit": {
            "status": "already_fixed_before_this_adjudication",
            "finding": "_fit_axis_weights uses one StandardScaler fitted on pooled training-fold timepoints and selects logistic C within the training fold",
        },
        "datasets": datasets,
    }
    destination.write_text(canonical_json(output))
    print(json.dumps({"output": str(destination), "datasets": datasets}, indent=2))


if __name__ == "__main__":
    main()
