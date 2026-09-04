"""Adjudicate switching dynamics against confined-drift and noise-scale controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import (  # noqa: E402
    fit_gaussian_state_space,
    fit_switching_ar_hmm,
    gaussian_state_space_conditional_log_likelihood,
    simulate_confined_diffusion,
    simulate_heteroscedastic_confined_diffusion,
    simulate_switching_ar_hmm,
    summarize_switching_decompositions,
    switching_ar_hmm_log_likelihood,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import ANATOMICAL_REGIONS  # noqa: E402


DATASETS = {
    "DANDI 000469": "human_drift_spine_000469.json",
    "DANDI 000574": "human_drift_spine_000574.json",
    "DANDI 001187/000673": "human_drift_spine_001187_000673.json",
    "Miller N-back": "miller_drift_spine.json",
    "Panichello 2024": "panichello_2024_drift_switching.json",
}


def _record(entity: str, fold: dict[str, Any], dt: float) -> dict[str, Any]:
    state = fold["state_space"]
    switching = fold.get("switching_two_state", fold.get("switching_two"))
    n_time = int(state.get("diagnostics", {}).get("n_time", 0))
    return {
        "entity": entity,
        "fold": fold,
        "state": state,
        "switching": switching,
        "dt": float(dt),
        "n_time": n_time,
        "n_train": int(fold["n_train"]),
        "n_test": int(fold["n_test"]),
    }


def extract_folds(name: str, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if name in {"DANDI 000469", "DANDI 000574"}:
        for entity, session in sorted(artifact["sessions"].items()):
            if session.get("status") == "complete":
                independent_entity = (
                    entity.split("_ses-", maxsplit=1)[0]
                    if name == "DANDI 000574"
                    else entity
                )
                dt = float(session.get("dt", artifact.get("bin_ms", 50) / 1000.0))
                rows.extend(
                    _record(independent_entity, fold, dt) for fold in session["folds"]
                )
    elif name == "DANDI 001187/000673":
        for entity, session in sorted(artifact["sessions"].items()):
            independent_entity = str(session.get("patient", entity))
            for view in (
                session["unit_based_primary_fit"],
                session.get("lfp_linked_sensitivity_fit"),
            ):
                if view is None or view.get("status") != "complete":
                    continue
                for load, fit in sorted(view["by_load"].items()):
                    if fit.get("status") == "complete":
                        rows.extend(
                            _record(independent_entity, fold, float(fit["dt"]))
                            for fold in fit["folds"]
                        )
    elif name == "Miller N-back":
        for entity, patient in sorted(artifact["patients"].items()):
            if patient.get("status") == "complete":
                fit = patient["drift"]
                rows.extend(_record(entity, fold, float(fit["dt"])) for fold in fit["folds"])
    elif name == "Panichello 2024":
        for session_name, session in sorted(artifact["sessions"].items()):
            if session.get("status") == "complete":
                entity = str(session.get("animal", session_name))
                rows.extend(_record(entity, fold, 0.05) for fold in session["folds"])
    return rows


def _finite_number(value: Any) -> bool:
    return value is not None and np.isfinite(value)


def _observed_delta(fold: dict[str, Any]) -> float:
    if "m4_minus_m2_nats_per_transition" in fold:
        value = fold["m4_minus_m2_nats_per_transition"]
        return float(value) if _finite_number(value) else float("nan")
    first = fold.get("M4_two_state_log_likelihood_per_transition")
    second = fold.get("M2_log_likelihood_per_transition_conditional")
    return float(first - second) if _finite_number(first) and _finite_number(second) else float("nan")


def _control_delta(fold: dict[str, Any], field: str) -> float:
    m2 = fold.get(
        "m2_log_likelihood_per_transition_conditional",
        fold.get("M2_log_likelihood_per_transition_conditional"),
    )
    value = fold.get(field)
    return float(value - m2) if _finite_number(value) and _finite_number(m2) else float("nan")


def _heteroscedastic_delta(fold: dict[str, Any]) -> float:
    value = fold.get(
        "m4_minus_heteroscedastic_drift_nats_per_transition",
        fold.get("M4_two_minus_heteroscedastic_drift"),
    )
    return float(value) if _finite_number(value) else float("nan")


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"status": "not_estimable", "n": 0}
    return {
        "status": "estimable",
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "fold_value_percentile_range_95": list(map(float, np.percentile(array, [2.5, 97.5]))),
        "fraction_positive": float(np.mean(array > 0.0)),
    }


def _entity_summary(rows: list[dict[str, Any]], value_getter=_observed_delta) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["switching"].get("status") == "non_identified":
            continue
        value = float(value_getter(row["fold"]))
        if np.isfinite(value):
            grouped.setdefault(row["entity"], []).append(value)
    return {entity: float(np.mean(values)) for entity, values in sorted(grouped.items())}


def _entity_median_inference(
    values: dict[str, float],
    rng: np.random.Generator,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Cluster-bootstrap an entity median using entities as resampling units."""
    array = np.asarray([value for value in values.values() if np.isfinite(value)], dtype=float)
    if not len(array):
        return {"status": "not_estimable", "n_entities": 0, "reason": "no finite entity values"}
    draws = np.median(array[rng.integers(0, len(array), size=(n_boot, len(array)))], axis=1)
    return {
        "status": "estimable",
        "estimate": float(np.median(array)),
        "patient_or_session_bootstrap_interval_95": list(map(
            float, np.percentile(draws, [2.5, 97.5])
        )),
        "n_entities": int(len(array)),
        "bootstrap_replicates": int(n_boot),
    }


def _score_simulation(
    train: np.ndarray,
    test: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> float:
    m2 = fit_gaussian_state_space(train, dt)
    m4 = fit_switching_ar_hmm(train, n_states=2, n_restarts=4, rng=rng)
    denominator = max(test[:, 1:].size, 1)
    return float(
        (
            switching_ar_hmm_log_likelihood(test, m4)
            - gaussian_state_space_conditional_log_likelihood(test, m2, dt)
        )
        / denominator
    )


def model_recovery(
    rows: list[dict[str, Any]],
    n_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    m2_rows = [
        row for row in rows
        if row["n_time"] >= 5
        and all(_finite_number(row["state"].get(field)) for field in (
            "lambda_rate", "diffusion", "equilibrium", "observation_variance"
        ))
        and float(row["state"]["diffusion"]) >= 0.0
        and float(row["state"]["lambda_rate"]) > 0.0
        and float(row["state"]["observation_variance"]) >= 0.0
    ]
    switching_rows = [
        row for row in rows
        if row["n_time"] >= 5 and row["switching"].get("status") == "complete"
    ]
    heteroscedastic_rows = [
        row for row in rows
        if row["n_time"] >= 5
        and row["fold"].get("heteroscedastic_drift", {}).get("status") == "complete"
    ]
    if not m2_rows or not switching_rows or not heteroscedastic_rows:
        return {
            "status": "not_estimable",
            "reason": "no valid fitted fold for at least one generating model",
        }
    null_values: list[float] = []
    heteroscedastic_null_values: list[float] = []
    reverse_values: list[float] = []
    null_by_entity: dict[str, list[float]] = {}
    heteroscedastic_null_by_entity: dict[str, list[float]] = {}
    reverse_by_entity: dict[str, list[float]] = {}
    excluded = {"pure_confined_diffusion": 0, "heteroscedastic_confined_diffusion": 0,
                "fitted_switching": 0}
    for replicate in range(n_replicates):
        null_row = m2_rows[replicate % len(m2_rows)]
        state = null_row["state"]
        null_train = simulate_confined_diffusion(
            null_row["n_train"], null_row["n_time"], null_row["dt"],
            float(state["lambda_rate"]), float(state["diffusion"]),
            equilibrium=float(state["equilibrium"]),
            observation_sd=float(np.sqrt(state["observation_variance"])), rng=rng,
        )[1]
        null_test = simulate_confined_diffusion(
            null_row["n_test"], null_row["n_time"], null_row["dt"],
            float(state["lambda_rate"]), float(state["diffusion"]),
            equilibrium=float(state["equilibrium"]),
            observation_sd=float(np.sqrt(state["observation_variance"])), rng=rng,
        )[1]
        null_value = _score_simulation(null_train, null_test, null_row["dt"], rng)
        if np.isfinite(null_value):
            null_values.append(null_value)
            null_by_entity.setdefault(null_row["entity"], []).append(null_value)
        else:
            excluded["pure_confined_diffusion"] += 1

        heteroscedastic_row = heteroscedastic_rows[replicate % len(heteroscedastic_rows)]
        heteroscedastic_estimate = heteroscedastic_row["fold"]["heteroscedastic_drift"]
        heteroscedastic_train = simulate_heteroscedastic_confined_diffusion(
            heteroscedastic_row["n_train"], heteroscedastic_row["n_time"],
            heteroscedastic_estimate, rng,
        )
        heteroscedastic_test = simulate_heteroscedastic_confined_diffusion(
            heteroscedastic_row["n_test"], heteroscedastic_row["n_time"],
            heteroscedastic_estimate, rng,
        )
        heteroscedastic_value = _score_simulation(
            heteroscedastic_train, heteroscedastic_test, heteroscedastic_row["dt"], rng
        )
        if np.isfinite(heteroscedastic_value):
            heteroscedastic_null_values.append(heteroscedastic_value)
            heteroscedastic_null_by_entity.setdefault(
                heteroscedastic_row["entity"], []
            ).append(heteroscedastic_value)
        else:
            excluded["heteroscedastic_confined_diffusion"] += 1

        reverse_row = switching_rows[replicate % len(switching_rows)]
        reverse_train = simulate_switching_ar_hmm(
            reverse_row["n_train"], reverse_row["n_time"], reverse_row["switching"], rng,
        )
        reverse_test = simulate_switching_ar_hmm(
            reverse_row["n_test"], reverse_row["n_time"], reverse_row["switching"], rng,
        )
        reverse_value = _score_simulation(reverse_train, reverse_test, reverse_row["dt"], rng)
        if np.isfinite(reverse_value):
            reverse_values.append(reverse_value)
            reverse_by_entity.setdefault(reverse_row["entity"], []).append(reverse_value)
        else:
            excluded["fitted_switching"] += 1
    null_entity_means = {
        entity: float(np.mean(values)) for entity, values in sorted(null_by_entity.items())
    }
    reverse_entity_means = {
        entity: float(np.mean(values)) for entity, values in sorted(reverse_by_entity.items())
    }
    heteroscedastic_null_entity_means = {
        entity: float(np.mean(values))
        for entity, values in sorted(heteroscedastic_null_by_entity.items())
    }
    return {
        "status": "complete",
        "n_replicates_per_leg": int(n_replicates),
        "pure_confined_diffusion": _summary(null_values),
        "heteroscedastic_confined_diffusion": _summary(heteroscedastic_null_values),
        "fitted_switching": _summary(reverse_values),
        "pure_confined_diffusion_entity_means": null_entity_means,
        "fitted_switching_entity_means": reverse_entity_means,
        "heteroscedastic_confined_diffusion_entity_means": heteroscedastic_null_entity_means,
        "pure_confined_diffusion_entity_summary": _summary(list(null_entity_means.values())),
        "fitted_switching_entity_summary": _summary(list(reverse_entity_means.values())),
        "heteroscedastic_confined_diffusion_entity_summary": _summary(
            list(heteroscedastic_null_entity_means.values())
        ),
        "false_positive_gate": bool(
            np.median(list(null_entity_means.values())) <= 0.0
        ),
        "false_positive_gate_direction": "one_sided_null_median_must_not_be_positive",
        "switching_recovery_gate": bool(
            _entity_median_inference(reverse_entity_means, rng)
            .get("patient_or_session_bootstrap_interval_95", [-np.inf])[0] > 0.0
        ),
        "variance_floor_exclusions": excluded,
        "null_values": list(map(float, null_values)),
        "heteroscedastic_null_values": list(map(float, heteroscedastic_null_values)),
        "reverse_values": list(map(float, reverse_values)),
    }


def adjudicate_dataset(
    name: str,
    rows: list[dict[str, Any]],
    min_replicates: int,
    seed_index: int,
    artifact_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full switching-vs-drift adjudication on one pre-extracted set of folds."""
    observed = [_observed_delta(row["fold"]) for row in rows]
    tied = [_control_delta(row["fold"], "m4_tied_log_likelihood_per_transition")
            if "m4_tied_log_likelihood_per_transition" in row["fold"] else
            _control_delta(row["fold"], "M4_tied_log_likelihood_per_transition") for row in rows]
    heteroscedastic = [_heteroscedastic_delta(row["fold"]) for row in rows]
    dataset_replicates = max(min_replicates, len(rows))
    recovery = model_recovery(
        rows, dataset_replicates, np.random.default_rng(20260801 + seed_index)
    )
    recovery["requested_minimum_replicates"] = int(min_replicates)
    recovery["coverage_rule"] = (
        "at least 200 replicates and at least one round-robin draw per fitted fold"
    )
    observed_entities = _entity_summary(rows)
    tied_entities = _entity_summary(
        rows,
        lambda fold: _control_delta(
            fold,
            "m4_tied_log_likelihood_per_transition"
            if "m4_tied_log_likelihood_per_transition" in fold
            else "M4_tied_log_likelihood_per_transition",
        ),
    )
    heteroscedastic_entities = _entity_summary(
        rows,
        _heteroscedastic_delta,
    )
    entity_rng = np.random.default_rng(20261801 + seed_index)
    entity_inference = {
        "free_switching_minus_m2": _entity_median_inference(observed_entities, entity_rng),
        "tied_variance_switching_minus_m2": _entity_median_inference(tied_entities, entity_rng),
        "free_switching_minus_heteroscedastic_drift": _entity_median_inference(
            heteroscedastic_entities, entity_rng
        ),
    }
    observed_median = entity_inference["free_switching_minus_m2"].get("estimate")
    tied_median = entity_inference["tied_variance_switching_minus_m2"].get("estimate")
    heteroscedastic_median = entity_inference[
        "free_switching_minus_heteroscedastic_drift"
    ].get("estimate")
    null_values = np.asarray(
        list(recovery.get("pure_confined_diffusion_entity_means", {}).values()),
        dtype=float,
    )
    percentile = (
        float(100.0 * np.mean(null_values <= observed_median))
        if len(null_values) and observed_median is not None else None
    )
    heteroscedastic_null_values = np.asarray(
        list(recovery.get("heteroscedastic_confined_diffusion_entity_means", {}).values()),
        dtype=float,
    )
    heteroscedastic_percentile = (
        float(100.0 * np.mean(heteroscedastic_null_values <= observed_median))
        if len(heteroscedastic_null_values) and observed_median is not None else None
    )
    lower_bounds = [
        entity_inference[key].get("patient_or_session_bootstrap_interval_95", [np.nan])[0]
        for key in (
            "free_switching_minus_m2",
            "tied_variance_switching_minus_m2",
            "free_switching_minus_heteroscedastic_drift",
        )
    ]
    supported = bool(
        all(np.isfinite(lower) and lower > 0.0 for lower in lower_bounds)
        and heteroscedastic_percentile is not None
        and heteroscedastic_percentile >= 95.0
        and recovery.get("false_positive_gate", False)
        and recovery.get("switching_recovery_gate", False)
    )
    floor_excluded_rows = [
        row for row in rows if row["switching"].get("status") == "non_identified"
    ]
    tied_floor_excluded_rows = [
        row for row in rows
        if row["fold"].get("switching_tied_variance", {}).get("status")
        == "non_identified"
    ]
    dataset_entry = {
        "artifact": artifact_label,
        "n_folds": len(rows),
        "n_entities": len(set(row["entity"] for row in rows)),
        "variance_floor": {
            "fraction_of_pooled_target_variance": 1e-3,
            "n_folds_excluded": len(floor_excluded_rows),
            "n_folds_eligible": len(rows) - len(floor_excluded_rows),
            "n_tied_variance_folds_at_floor": len(tied_floor_excluded_rows),
            "excluded_folds": [
                {"entity": row["entity"], "reason": row["switching"].get("reason")}
                for row in floor_excluded_rows
            ],
        },
        "switching_decomposition": summarize_switching_decompositions(
            [row["fold"] for row in rows]
        ),
        "free_switching_minus_m2": _summary(observed),
        "tied_variance_switching_minus_m2": _summary(tied),
        "free_switching_minus_heteroscedastic_drift": _summary(heteroscedastic),
        "observed_entity_means": observed_entities,
        "tied_variance_entity_means": tied_entities,
        "heteroscedastic_control_entity_means": heteroscedastic_entities,
        "entity_medians": {
            "free_switching_minus_m2": observed_median,
            "tied_variance_switching_minus_m2": tied_median,
            "free_switching_minus_heteroscedastic_drift": heteroscedastic_median,
        },
        "entity_median_inference": entity_inference,
        "observed_entity_median_percentile_of_fitted_m2_null": percentile,
        "observed_entity_median_percentile_of_heteroscedastic_diffusion_null": heteroscedastic_percentile,
        "interpretation": (
            "dynamics_switching_supported"
            if supported
            else "noise_scale_or_estimator_null_not_excluded"
        ),
        "model_recovery": recovery,
    }
    recovery_entry = {
        "pure_confined_diffusion_entity_median": float(np.median(list(
            recovery.get("pure_confined_diffusion_entity_means", {}).values()
        ))),
        "fitted_switching_entity_median": float(np.median(list(
            recovery.get("fitted_switching_entity_means", {}).values()
        ))),
        "heteroscedastic_confined_diffusion_entity_median": (
            float(np.median(list(
                recovery.get("heteroscedastic_confined_diffusion_entity_means", {}).values()
            )))
            if recovery.get("heteroscedastic_confined_diffusion_entity_means") else None
        ),
        "pure_confined_diffusion_fold_median": recovery.get("pure_confined_diffusion", {}).get("median"),
        "fitted_switching_fold_median": recovery.get("fitted_switching", {}).get("median"),
        "heteroscedastic_confined_diffusion_fold_median": recovery.get(
            "heteroscedastic_confined_diffusion", {}
        ).get("median"),
        "n_replicates_per_leg": recovery.get("n_replicates_per_leg"),
    }
    return dataset_entry, recovery_entry


REGION_STRATIFIED_GROUPS = (
    ("DANDI 000469", "region_stratified_drift_000469.json"),
    ("DANDI 001187/000673", "region_stratified_drift_001187_000673.json"),
)


def run_region_stratified(min_replicates: int) -> dict[str, Any]:
    """Re-run the switching-vs-drift adjudication per anatomical region.

    Underpowered per-region outcomes are expected and recorded as non_identified rather
    than treated as a negative -- the point is to test whether the pooled null is a
    genuine null or a region-mixture artifact, not to rescue switching.
    """
    region_stratified: dict[str, Any] = {}
    for dandi_name, filename in REGION_STRATIFIED_GROUPS:
        path = ROOT / "results" / filename
        artifact = json.loads(path.read_text())
        group: dict[str, Any] = {}
        for seed_index, region in enumerate(ANATOMICAL_REGIONS):
            sessions = artifact["regions"][region]["sessions"]
            rows = extract_folds(dandi_name, {"sessions": sessions})
            if not rows:
                group[region] = {
                    "status": "non_identified",
                    "reason": "no complete region-session fold in the region-stratified refit",
                }
                continue
            if any("switching_decomposition" not in row["fold"] for row in rows):
                group[region] = {
                    "status": "non_identified",
                    "reason": "switching_decomposition missing on at least one fold",
                }
                continue
            dataset_entry, recovery_entry = adjudicate_dataset(
                f"{dandi_name} :: {region}",
                rows,
                min_replicates,
                seed_index=hash((dandi_name, region)) % 1000,
                artifact_label=f"results/{filename}#regions.{region}",
            )
            dataset_entry["status"] = "estimable"
            dataset_entry["two_cell_model_recovery"] = recovery_entry
            group[region] = dataset_entry
        region_stratified[dandi_name] = group
    return region_stratified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--region-stratified", action="store_true")
    args = parser.parse_args()
    if args.replicates < 200:
        raise SystemExit("at least 200 replicates per dataset are required")

    if args.region_stratified:
        started = time.time()
        region_stratified = run_region_stratified(args.replicates)
        destination = ROOT / "results" / "switching_adjudication.json"
        output = json.loads(destination.read_text())
        output["region_stratified"] = region_stratified
        output["region_stratified_runtime_seconds"] = float(time.time() - started)
        destination.write_text(canonical_json(output))

        crack_path = ROOT / "results" / "crack_register.json"
        cracks = json.loads(crack_path.read_text())
        entries = cracks["entries"] if isinstance(cracks, dict) else cracks
        estimable = [
            f"{dandi_name}/{region}"
            for dandi_name, group in region_stratified.items()
            for region, entry in group.items()
            if entry.get("status") == "estimable"
        ]
        supported = [
            f"{dandi_name}/{region}"
            for dandi_name, group in region_stratified.items()
            for region, entry in group.items()
            if entry.get("interpretation") == "dynamics_switching_supported"
        ]
        entries.append({
            "artifact": "results/switching_adjudication.json",
            "crack_id": "region_resolved_switching_adjudication",
            "trigger": (
                "Re-run the switching-vs-drift adjudication per anatomical "
                "region on DANDI 000469 and the linked 001187/000673 view, to test whether the "
                "pooled CRACK-7 null is a genuine null or a region-mixture artifact."
            ),
            "chase": (
                "Reused extract_folds and the full symmetric-rule adjudication (variance floor, "
                "cluster bootstrap, heteroscedastic null leg, one-sided false-positive gate) "
                "unchanged against each region's session block from the region-stratified "
                "artifacts; no new estimator was written."
            ),
            "resolution": (
                f"{len(estimable)}/10 region x dataset-group cells were estimable "
                f"(non-identified cells lacked a complete region-session fold, expected given "
                f"MIN_UNITS_PER_REGION sparsity). Controlled dynamics-switching support: "
                f"{', '.join(supported) or 'none'}. This does not close CRACK-7: an "
                f"underpowered per-region null is not evidence against the pooled result, "
                f"only a report of what is identifiable at this n. See "
                f"results/switching_adjudication.json's region_stratified block for full "
                f"per-region numbers."
            ),
            "status": "resolved_region_resolved_extension_not_closing_crack7",
        })
        crack_path.write_text(canonical_json(cracks))
        print(json.dumps({
            "output": str(destination),
            "runtime_seconds": output["region_stratified_runtime_seconds"],
            "estimable_cells": estimable,
        }, indent=2))
        return

    started = time.time()
    decision_rule = {
        "effect_threshold": None,
        "criterion": (
            "patient_or_session_bootstrap lower bound exceeds zero for free M4-M2, "
            "tied-variance M4-M2, and M4-heteroscedastic drift; observed median is at "
            "or above the 95th percentile of the heteroscedastic confined-diffusion null; "
            "the homoscedastic null median is nonpositive; and switching recovery lower bound exceeds zero"
        ),
        "threshold_rationale": "The former 0.01-nat threshold was removed because it was not an independently justified scientific SESOI.",
    }
    decision_rule_hash = hashlib.sha256(canonical_json(decision_rule).encode()).hexdigest()
    datasets: dict[str, Any] = {}
    recovery_table: dict[str, Any] = {}
    for index, (name, filename) in enumerate(DATASETS.items()):
        path = ROOT / "results" / filename
        artifact = json.loads(path.read_text())
        rows = extract_folds(name, artifact)
        if not rows or any("switching_decomposition" not in row["fold"] for row in rows):
            raise SystemExit(f"{name} has not been rerun with switching decomposition fields")
        dataset_entry, recovery_entry = adjudicate_dataset(
            name, rows, args.replicates, seed_index=index, artifact_label=f"results/{filename}"
        )
        datasets[name] = dataset_entry
        recovery_table[name] = recovery_entry
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "switching_adjudication",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "decision_rule": decision_rule,
        "decision_rule_hash": decision_rule_hash,
        "runtime_seconds": float(time.time() - started),
        "datasets": datasets,
        "two_cell_model_recovery": recovery_table,
    }
    destination = ROOT / "results" / "switching_adjudication.json"
    destination.write_text(canonical_json(output))

    gate_path = ROOT / "results" / "drift_simulation_gate.json"
    gate = json.loads(gate_path.read_text())
    gate["switching_model_recovery"] = {
        "source_artifact": "results/switching_adjudication.json",
        "datasets": recovery_table,
        "all_false_positive_gates_pass": all(
            row["model_recovery"].get("false_positive_gate", False)
            for row in datasets.values()
        ),
        "all_switching_recovery_gates_pass": all(
            row["model_recovery"].get("switching_recovery_gate", False)
            for row in datasets.values()
        ),
    }
    gate_path.write_text(canonical_json(gate))

    crack_path = ROOT / "results" / "crack_register.json"
    cracks = json.loads(crack_path.read_text())
    entries = cracks["entries"] if isinstance(cracks, dict) else cracks
    supported = [
        name for name, row in datasets.items()
        if row["interpretation"] == "dynamics_switching_supported"
    ]
    unsupported = [name for name in datasets if name not in supported]
    for crack in entries:
        if crack.get("crack_id") != "CRACK-7":
            continue
        crack.update({
            "artifact": "results/switching_adjudication.json; results/drift_simulation_gate.json; results/dynamax_dependency_audit.json",
            "chase": "Decomposed both regimes into AR-coefficient, observation-variance, and occupancy separation; compared free switching with a tied-variance fit and a one-dynamics/two-observation-variance drift control on identical folds; then ran at least 200 fitted-M2 and fitted-switching recovery simulations per dataset.",
            "resolution": (
                "Controlled dynamics-switching support is dataset-specific. "
                f"Supported after both variance controls and fitted-M2 calibration: {', '.join(supported) or 'none'}. "
                f"Noise-scale or estimator-null explanations remain viable: {', '.join(unsupported) or 'none'}. "
                "The installed Dynamax API still cannot train the requested Poisson rSLDS, so no biological regime-switching claim is licensed."
            ),
            "status": "resolved_as_controlled_dataset_heterogeneity",
        })
        break
    crack_path.write_text(canonical_json(cracks))
    print(json.dumps({"output": str(destination), "runtime_seconds": output["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
