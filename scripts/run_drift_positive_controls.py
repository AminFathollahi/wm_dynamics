#!/usr/bin/env python3
"""Run matched-noise and session-nonstationarity controls for human drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drift_dynamics import (  # noqa: E402
    compare_switching_models,
    compare_temporal_dependence_models,
    fit_gaussian_state_space,
    gaussian_state_space_log_likelihood,
    held_out_linear_prediction,
    neighbouring_trial_prediction_advantage,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from run_human_drift_spine_000469 import (  # noqa: E402
    AXIS_WINDOW,
    BIN_MS,
    MIN_UNITS,
    N_COMPONENTS,
    N_SPLITS,
    WINDOW_S,
    data_directory as data_directory_000469,
    discriminant_direction,
    iid_log_likelihood,
    matched_complement_direction,
    projected_residuals,
)
from run_human_drift_spine_000574 import (  # noqa: E402
    MAINT_ONSET_S,
    MAINT_WIN,
    MIN_TRIALS,
    data_directory as data_directory_000574,
)
from spike_pipeline import (  # noqa: E402
    FrozenPSTHTransform,
    MIN_SESSION_ACCURACY,
    MIN_UNITS_PER_REGION,
    REGIONS_WITH_POOLED,
    build_psth,
    filter_units_by_region,
    load_spike_times,
    low_rate_unit_mask,
    resolve_unit_regions,
)

BASELINE_WINDOW_S = 0.5
BOOTSTRAP_REPLICATES = 5000
PERMUTED_AXIS_SEED_OFFSET = 2000
COMPLEMENT_AXIS_SEED_OFFSET = 4000
SWITCHING_SEED_OFFSET = 6000


def anscombe_rates(rates: np.ndarray, bin_ms: float) -> np.ndarray:
    """Variance-stabilize spike counts and retain an explicit count-scale transform."""
    counts = np.maximum(np.asarray(rates, dtype=float) * (bin_ms / 1000.0), 0.0)
    return 2.0 * np.sqrt(counts + 3.0 / 8.0)


def detrend_trial_order(
    values: np.ndarray,
    trial_order: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Remove a training-fitted quadratic session-time trend from each unit."""
    order = np.asarray(trial_order, dtype=float)
    center = float(np.mean(order[train_indices]))
    scale = max(float(np.std(order[train_indices])), 1.0)
    normalized = (order - center) / scale
    design = np.column_stack([np.ones(len(order)), normalized, normalized * normalized])
    trial_means = np.mean(values, axis=2)
    coefficients = np.linalg.lstsq(design[train_indices], trial_means[train_indices], rcond=None)[0]
    fitted = design @ coefficients
    reference = np.mean(fitted[train_indices], axis=0, keepdims=True)
    return np.asarray(values, dtype=float) - (fitted - reference)[:, :, None]


def bootstrap_summary(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"status": "not_estimable", "n_entities": 0, "reason": "no finite entity estimates"}
    draws = np.mean(
        finite[rng.integers(0, len(finite), size=(BOOTSTRAP_REPLICATES, len(finite)))], axis=1
    )
    return {
        "status": "estimable",
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "patient_bootstrap_interval_95": list(map(float, np.percentile(draws, [2.5, 97.5]))),
        "n_entities": int(len(finite)),
        "fraction_positive": float(np.mean(finite > 0.0)),
        "values": finite.tolist(),
    }


def projected_axis_scores(
    train_state: np.ndarray,
    test_state: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    direction: np.ndarray,
    dt: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    train_residuals, test_residuals = projected_residuals(
        train_state, test_state, train_labels, test_labels, direction
    )
    comparison = compare_temporal_dependence_models(train_residuals, test_residuals, dt)
    state_estimate = fit_gaussian_state_space(train_residuals, dt)
    m2_full = gaussian_state_space_log_likelihood(test_residuals, state_estimate, dt)
    m0_full = iid_log_likelihood(test_residuals, train_residuals)
    comparison["m2_minus_m0_nats_per_observation"] = (
        m2_full - m0_full
    ) / max(test_residuals.size, 1)
    own = held_out_linear_prediction(
        train_residuals[:, :-1], train_residuals[:, 1:],
        test_residuals[:, :-1], test_residuals[:, 1:],
    )
    comparison["own_trial_prediction"] = own
    return comparison, train_residuals, test_residuals


def permuted_axis_scores(
    train_state: np.ndarray,
    test_state: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_window: np.ndarray,
    dt: float,
    seed: int,
) -> dict[str, Any]:
    """Estimate a null axis from permuted labels while retaining true centroids."""
    permuted_labels = np.random.default_rng(seed).permutation(train_labels)
    direction = discriminant_direction(train_window, permuted_labels)
    comparison, _, _ = projected_axis_scores(
        train_state, test_state, train_labels, test_labels, direction, dt
    )
    return comparison


def analyze_fold(
    psth: np.ndarray,
    baseline_psth: np.ndarray,
    labels: np.ndarray,
    trial_order: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    seed: int,
    *,
    variance_stabilized: bool,
    detrended: bool = False,
) -> dict[str, Any]:
    values = anscombe_rates(psth, BIN_MS) if variance_stabilized else np.asarray(psth, dtype=float)
    baseline = (
        anscombe_rates(baseline_psth, BIN_MS)
        if variance_stabilized else np.asarray(baseline_psth, dtype=float)
    )
    if detrended:
        values = detrend_trial_order(values, trial_order, train_index)
    transform = FrozenPSTHTransform().fit(values[train_index])
    train_standardized = transform.transform(values[train_index]).transpose(0, 2, 1)
    test_standardized = transform.transform(values[test_index]).transpose(0, 2, 1)
    baseline_train = transform.transform(baseline[train_index]).transpose(0, 2, 1)
    baseline_test = transform.transform(baseline[test_index]).transpose(0, 2, 1)
    n_components = min(N_COMPONENTS, values.shape[1], len(train_index) - 1)
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(train_standardized.reshape(-1, train_standardized.shape[-1]))

    def reduce(array: np.ndarray) -> np.ndarray:
        return pca.transform(array.reshape(-1, array.shape[-1])).reshape(
            len(array), array.shape[1], -1
        )

    train_state, test_state = reduce(train_standardized), reduce(test_standardized)
    baseline_train_state, baseline_test_state = reduce(baseline_train), reduce(baseline_test)
    times = np.arange(psth.shape[2]) * BIN_MS / 1000.0 + BIN_MS / 2000.0
    axis_bins = (times >= AXIS_WINDOW[0]) & (times <= min(AXIS_WINDOW[1], times[-1]))
    train_window = train_state[:, axis_bins].mean(axis=1)
    content_direction = discriminant_direction(train_window, labels[train_index])
    content, train_residuals, test_residuals = projected_axis_scores(
        train_state, test_state, labels[train_index], labels[test_index], content_direction,
        BIN_MS / 1000.0,
    )
    all_residuals = np.full((len(labels), psth.shape[2]), np.nan)
    all_residuals[train_index] = train_residuals
    all_residuals[test_index] = test_residuals
    neighbour = neighbouring_trial_prediction_advantage(
        all_residuals, labels, train_index, test_index
    )
    if detrended:
        return {
            "content_axis": content,
            "neighbouring_trial_prediction": neighbour,
        }
    baseline_train_residuals, baseline_test_residuals = projected_residuals(
        baseline_train_state, baseline_test_state,
        labels[train_index], labels[test_index], content_direction,
    )
    baseline_prediction = held_out_linear_prediction(
        np.repeat(np.mean(baseline_train_residuals, axis=1)[:, None], train_residuals.shape[1] - 1, axis=1),
        train_residuals[:, 1:],
        np.repeat(np.mean(baseline_test_residuals, axis=1)[:, None], test_residuals.shape[1] - 1, axis=1),
        test_residuals[:, 1:],
    )
    permuted = permuted_axis_scores(
        train_state, test_state, labels[train_index], labels[test_index], train_window,
        BIN_MS / 1000.0, seed + PERMUTED_AXIS_SEED_OFFSET,
    )
    complement_direction, target_variance, complement_variance = matched_complement_direction(
        train_window, labels[train_index], content_direction,
        np.random.default_rng(seed + COMPLEMENT_AXIS_SEED_OFFSET),
    )
    complement, _, _ = projected_axis_scores(
        train_state, test_state, labels[train_index], labels[test_index], complement_direction,
        BIN_MS / 1000.0,
    )
    switching = compare_switching_models(
        train_residuals, test_residuals, BIN_MS / 1000.0, n_restarts=4,
        rng=np.random.default_rng(seed + SWITCHING_SEED_OFFSET),
    )
    return {
        "content_axis": content,
        "permuted_label_axis": permuted,
        "signal_matched_complement_axis": complement,
        "content_signal_variance": target_variance,
        "complement_signal_variance": complement_variance,
        "neighbouring_trial_prediction": neighbour,
        "pre_cue_baseline_prediction": baseline_prediction,
        "m4_minus_m2_nats_per_transition": (
            switching["free_log_likelihood_per_transition"]
            - content["m2_log_likelihood_per_transition"]
        ),
        "m4_minus_heteroscedastic_drift_nats_per_transition": (
            switching["free_log_likelihood_per_transition"]
            - switching["heteroscedastic_drift_log_likelihood_per_transition"]
        ),
    }


def load_000469(path: Path, region: str = "pooled") -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        spikes = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle)["region"]
        trials = handle["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        labels = trials["loadsEnc1_PicIDs"][:].astype(int)
        accuracy = trials["response_accuracy"][:].astype(bool)
        maintenance = trials["timestamps_Maintenance"][:]
        encoding = trials["timestamps_Encoding1"][:]
    spikes = filter_units_by_region(spikes, unit_regions, region)
    unit_mask = low_rate_unit_mask(spikes, maintenance, WINDOW_S)
    spikes = [unit for unit, keep in zip(spikes, unit_mask) if keep]
    min_units = MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION
    if len(spikes) < min_units or float(np.mean(accuracy)) < MIN_SESSION_ACCURACY:
        return {
            "status": "non_identified" if region != "pooled" else "excluded",
            "reason": f"prospective unit-count or accuracy QC failed (region={region}, n_units={len(spikes)})",
        }
    keep = loads == 1
    labels = labels[keep]
    counts = [int(np.sum(labels == label)) for label in np.unique(labels)]
    if len(counts) < 3 or min(counts) < N_SPLITS:
        return {"status": "excluded", "reason": "repeated-item counts cannot support five folds"}
    return {
        "status": "complete", "labels": labels, "trial_order": np.flatnonzero(keep),
        "psth": build_psth(spikes, maintenance[keep], BIN_MS, 0, WINDOW_S),
        "baseline_psth": build_psth(
            spikes, encoding[keep] - BASELINE_WINDOW_S, BIN_MS, 0, BASELINE_WINDOW_S
        ),
    }


def load_000574(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if "units" not in handle:
            return {"status": "excluded", "reason": "no units table"}
        spikes = load_spike_times(handle)
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
        labels = trials["set_size"][:].astype(int)
        start = trials["start_time"][:]
    keep = ~artifact
    labels, start = labels[keep], start[keep]
    maintenance = start + MAINT_ONSET_S
    if len(labels) < MIN_TRIALS:
        return {"status": "excluded", "reason": "too few artifact-free trials"}
    unit_mask = low_rate_unit_mask(spikes, maintenance, MAINT_WIN)
    spikes = [unit for unit, retained in zip(spikes, unit_mask) if retained]
    if len(spikes) < 8:
        return {"status": "excluded", "reason": "too few units after firing-rate QC"}
    usable = [label for label in np.unique(labels) if np.sum(labels == label) >= N_SPLITS]
    retained = np.isin(labels, usable)
    if len(usable) < 2:
        return {"status": "excluded", "reason": "set sizes cannot support five folds"}
    original_order = np.flatnonzero(keep)[retained]
    return {
        "status": "complete", "labels": labels[retained], "trial_order": original_order,
        "psth": build_psth(spikes, maintenance[retained], BIN_MS, 0, MAINT_WIN),
        "baseline_psth": build_psth(
            spikes, start[retained], BIN_MS, 0, BASELINE_WINDOW_S
        ),
    }


def flatten_metrics(row: dict[str, Any], prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_metrics(value, name))
        elif isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
            output[name] = float(value)
    return output


def summarize_sessions(dataset: str, sessions: dict[str, Any]) -> dict[str, Any]:
    """Aggregate finite fold metrics without dropping partially estimable fields."""
    by_entity: dict[str, list[dict[str, float]]] = {}
    for key, session in sessions.items():
        if session["status"] != "complete":
            continue
        entity = key if dataset == "000469" else key.split("_ses-")[0]
        fold_metrics = [flatten_metrics(fold) for fold in session["folds"]]
        names = sorted(set().union(*(set(row) for row in fold_metrics)))
        entity_row = {
            name: float(np.mean([row[name] for row in fold_metrics if name in row]))
            for name in names
        }
        by_entity.setdefault(entity, []).append(entity_row)
    entity_metrics = {
        entity: {
            name: float(np.mean([session[name] for session in rows if name in session]))
            for name in sorted(set().union(*(set(session) for session in rows)))
        }
        for entity, rows in by_entity.items()
    }
    for metrics in entity_metrics.values():
        for arm in ("raw", "variance_stabilized"):
            for control_axis in ("permuted_label_axis", "signal_matched_complement_axis"):
                comparisons = (
                    (
                        "m2_minus_m0_nats_per_observation",
                        "m2_minus_m0_nats_per_observation",
                    ),
                    (
                        "own_trial_prediction.held_out_r2_advantage",
                        "own_trial_prediction_r2_advantage",
                    ),
                )
                for source_suffix, output_suffix in comparisons:
                    content_key = f"{arm}.content_axis.{source_suffix}"
                    control_key = f"{arm}.{control_axis}.{source_suffix}"
                    if content_key in metrics and control_key in metrics:
                        metrics[
                            f"{arm}.content_minus_{control_axis}.{output_suffix}"
                        ] = float(metrics[content_key] - metrics[control_key])
    metric_names = sorted(set().union(*(set(row) for row in entity_metrics.values())))
    rng = np.random.default_rng(20260802 + int(dataset))
    group = {
        name: bootstrap_summary(
            [row[name] for row in entity_metrics.values() if name in row], rng
        )
        for name in metric_names
    }
    return {"sessions": sessions, "entity_metrics": entity_metrics, "group": group}


def source_fold_seed(dataset: str, session_key: str) -> int:
    """Return the fold seed used by the corresponding canonical drift spine."""
    if dataset == "000469":
        return 20260731 + int(session_key.split("-")[1])
    subject = int(session_key.split("_ses-")[0].split("-")[1])
    session = int(session_key.split("_ses-")[1])
    return 20260731 + subject * 100 + session


def adjudicate_positive_controls(analyses: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared matched-flexibility and session-history rule."""
    decisions: dict[str, Any] = {}
    for dataset, analysis in analyses.items():
        group = analysis["group"]

        def lower(name: str) -> float:
            return float(group[name]["patient_bootstrap_interval_95"][0])

        raw_matched = all(lower(name) > 0.0 for name in (
            "raw.content_axis.m2_minus_heteroscedastic_m0_nats_per_transition",
            "raw.content_axis.m2_minus_free_variance_ar1_m0_nats_per_transition",
        ))
        raw_history = lower(
            "raw.neighbouring_trial_prediction.own_minus_neighbour_r2_advantage"
        ) > 0.0
        stabilized_matched = all(lower(name) > 0.0 for name in (
            "variance_stabilized.content_axis.m2_minus_heteroscedastic_m0_nats_per_transition",
            "variance_stabilized.content_axis.m2_minus_free_variance_ar1_m0_nats_per_transition",
        ))
        stabilized_history = lower(
            "variance_stabilized.neighbouring_trial_prediction.own_minus_neighbour_r2_advantage"
        ) > 0.0
        detrended_history = lower(
            "trial_order_detrended.neighbouring_trial_prediction.own_minus_neighbour_r2_advantage"
        ) > 0.0
        content_specificity = all(
            lower(f"raw.content_minus_{axis}.{suffix}") > 0.0
            for axis in ("permuted_label_axis", "signal_matched_complement_axis")
            for suffix in (
                "m2_minus_m0_nats_per_observation",
                "own_trial_prediction_r2_advantage",
            )
        )
        off_axis_effects_positive = all(
            lower(f"raw.{axis}.{suffix}") > 0.0
            for axis in ("permuted_label_axis", "signal_matched_complement_axis")
            for suffix in (
                "m2_minus_m0_nats_per_observation",
                "own_trial_prediction.held_out_r2_advantage",
            )
        )
        decisions[dataset] = {
            "confined_dynamics_supported": bool(
                raw_matched and raw_history and stabilized_matched and stabilized_history
            ),
            "raw_matched_flexibility_gate": bool(raw_matched),
            "raw_own_minus_neighbour_gate": bool(raw_history),
            "variance_stabilized_matched_flexibility_gate": bool(stabilized_matched),
            "variance_stabilized_own_minus_neighbour_gate": bool(stabilized_history),
            "trial_order_detrended_own_minus_neighbour_gate": bool(detrended_history),
            "content_specificity_supported": bool(content_specificity),
            "off_axis_effects_positive": bool(off_axis_effects_positive),
            "interpretation": (
                "population_wide_predictive_history_not_confined_content_dynamics"
                if raw_history and detrended_history and stabilized_history
                and off_axis_effects_positive and not content_specificity and not raw_matched
                else "confined_dynamics_not_supported_and_predictive_history_sensitivity_dependent"
                if not raw_matched
                else "confined_dynamics_supported"
            ),
        }
    return {
        "overall": "original_confined_dynamics_interpretation_reversed",
        "datasets": decisions,
    }


def analyze_dataset(dataset: str, region: str = "pooled") -> dict[str, Any]:
    """``region`` only applies to 000469 (000574's region
    labels are too fine-grained for unit-level stratification and it stays
    pooled at the channel/LFP level regardless of ``region``)."""
    sessions: dict[str, Any] = {}
    if dataset == "000469":
        paths = sorted(data_directory_000469().glob("sub-*/sub-*_ses-2_ecephys+image.nwb"))
        loader = lambda path: load_000469(path, region=region)  # noqa: E731
    else:
        paths = sorted(data_directory_000574().glob("sub-*/*.nwb"))
        loader = load_000574
    for path in paths:
        key = path.parent.name if dataset == "000469" else path.stem
        print(f"positive controls {dataset} {key} region={region}", flush=True)
        prepared = loader(path)
        if prepared["status"] != "complete":
            sessions[key] = prepared
            continue
        labels = prepared["labels"]
        split_seed = source_fold_seed(dataset, key)
        splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=split_seed)
        folds = []
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            fold_seed = split_seed + fold
            raw = analyze_fold(
                prepared["psth"], prepared["baseline_psth"], labels,
                prepared["trial_order"], train, test, fold_seed,
                variance_stabilized=False,
            )
            stabilized = analyze_fold(
                prepared["psth"], prepared["baseline_psth"], labels,
                prepared["trial_order"], train, test, fold_seed,
                variance_stabilized=True,
            )
            detrended = analyze_fold(
                prepared["psth"], prepared["baseline_psth"], labels,
                prepared["trial_order"], train, test, fold_seed,
                variance_stabilized=False, detrended=True,
            )
            folds.append({
                "fold": fold, "raw": raw, "variance_stabilized": stabilized,
                "trial_order_detrended": {
                    "content_axis": detrended["content_axis"],
                    "neighbouring_trial_prediction": detrended["neighbouring_trial_prediction"],
                },
            })
        sessions[key] = {"status": "complete", "n_trials": len(labels), "folds": folds}
    return summarize_sessions(dataset, sessions)


def simulation_controls() -> dict[str, Any]:
    """False-positive and recovery legs for every deciding positive-control statistic."""
    rng = np.random.default_rng(20260804)
    results: dict[str, list[float]] = {
        "heteroscedastic_null_m2_minus_heteroscedastic_m0": [],
        "heteroscedastic_null_m2_minus_free_ar1_m0": [],
        "diffusion_recovery_m2_minus_heteroscedastic_m0": [],
        "diffusion_recovery_m2_minus_free_ar1_m0": [],
        "session_trend_own_minus_neighbour": [],
        "diffusion_recovery_own_minus_neighbour": [],
    }
    from drift_dynamics import simulate_confined_diffusion
    for repetition in range(40):
        noise = rng.normal(size=(72, 23))
        noise[rng.random(noise.shape) < 0.15] *= 4.0
        null = compare_temporal_dependence_models(noise[:54], noise[54:], 0.1)
        results["heteroscedastic_null_m2_minus_heteroscedastic_m0"].append(
            null["m2_minus_heteroscedastic_m0_nats_per_transition"]
        )
        results["heteroscedastic_null_m2_minus_free_ar1_m0"].append(
            null["m2_minus_free_variance_ar1_m0_nats_per_transition"]
        )
        _, diffusion = simulate_confined_diffusion(
            72, 23, 0.1, 1.2, 0.3, observation_sd=0.25, rng=rng
        )
        recovered = compare_temporal_dependence_models(diffusion[:54], diffusion[54:], 0.1)
        results["diffusion_recovery_m2_minus_heteroscedastic_m0"].append(
            recovered["m2_minus_heteroscedastic_m0_nats_per_transition"]
        )
        results["diffusion_recovery_m2_minus_free_ar1_m0"].append(
            recovered["m2_minus_free_variance_ar1_m0_nats_per_transition"]
        )
        labels = np.arange(72) % 4
        train, test = np.arange(54), np.arange(54, 72)
        trend = np.linspace(-2, 2, 72)[:, None] + rng.normal(scale=0.08, size=(72, 23))
        trend_result = neighbouring_trial_prediction_advantage(trend, labels, train, test)
        diffusion_result = neighbouring_trial_prediction_advantage(diffusion, labels, train, test)
        results["session_trend_own_minus_neighbour"].append(
            trend_result["own_minus_neighbour_r2_advantage"]
        )
        results["diffusion_recovery_own_minus_neighbour"].append(
            diffusion_result["own_minus_neighbour_r2_advantage"]
        )
    return {
        name: bootstrap_summary(values, np.random.default_rng(20260804 + index))
        for index, (name, values) in enumerate(results.items())
    }


def extend_drift_artifact(dataset: str, analysis: dict[str, Any]) -> None:
    path = ROOT / "results" / f"human_drift_spine_{dataset}.json"
    artifact = json.loads(path.read_text())
    for key, session in analysis["sessions"].items():
        if session.get("status") != "complete" or key not in artifact["sessions"]:
            continue
        existing_folds = artifact["sessions"][key].get("folds", [])
        for control_fold in session["folds"]:
            fold = next((row for row in existing_folds if row.get("fold") == control_fold["fold"]), None)
            if fold is not None:
                fold["positive_controls"] = {
                    key: value for key, value in control_fold.items() if key != "fold"
                }
    artifact["positive_controls"] = {
        "group": analysis["group"],
        "independent_unit": "patient",
        "source_hash": sha256_file(Path(__file__)),
    }
    path.write_text(canonical_json(artifact))


def audit_source_fold_alignment(dataset: str, analysis: dict[str, Any]) -> dict[str, Any]:
    """Verify that the raw control arm reproduces the source fold's M2-M0 score."""
    path = ROOT / "results" / f"human_drift_spine_{dataset}.json"
    source = json.loads(path.read_text())
    differences = []
    switching_differences = []
    complement_differences = []
    for session_key, control_session in analysis["sessions"].items():
        if control_session.get("status") != "complete":
            continue
        source_session = source["sessions"].get(session_key, {})
        source_folds = {fold["fold"]: fold for fold in source_session.get("folds", [])}
        for control_fold in control_session["folds"]:
            source_fold = source_folds.get(control_fold["fold"])
            if source_fold is None:
                continue
            source_value = source_fold.get(
                "m2_minus_m0_nats_per_observation",
                source_fold.get("M2_minus_M0_nats_per_observation"),
            )
            control_value = control_fold["raw"]["content_axis"][
                "m2_minus_m0_nats_per_observation"
            ]
            if source_value is not None and np.isfinite(source_value):
                differences.append(abs(float(source_value) - float(control_value)))
            source_switching = source_fold.get("m4_minus_m2_nats_per_transition")
            control_switching = control_fold["raw"].get(
                "m4_minus_m2_nats_per_transition"
            )
            if (
                source_switching is not None and control_switching is not None
                and np.isfinite(source_switching) and np.isfinite(control_switching)
            ):
                switching_differences.append(
                    abs(float(source_switching) - float(control_switching))
                )
            for field in ("content_signal_variance", "complement_signal_variance"):
                source_axis_value = source_fold.get(field)
                control_axis_value = control_fold["raw"].get(field)
                if (
                    source_axis_value is not None and control_axis_value is not None
                    and np.isfinite(source_axis_value) and np.isfinite(control_axis_value)
                ):
                    complement_differences.append(
                        abs(float(source_axis_value) - float(control_axis_value))
                    )
    maximum = max(differences, default=float("inf"))
    switching_maximum = max(switching_differences, default=float("inf"))
    complement_maximum = max(complement_differences, default=float("inf"))
    return {
        "status": (
            "matched"
            if differences and switching_differences and complement_differences
            and max(maximum, switching_maximum, complement_maximum) <= 1e-10
            else "mismatch"
        ),
        "n_folds_compared": len(differences),
        "maximum_absolute_m2_minus_m0_difference": maximum,
        "maximum_absolute_m4_minus_m2_difference": switching_maximum,
        "maximum_absolute_axis_variance_difference": complement_maximum,
        "tolerance": 1e-10,
    }


def extend_region_stratified_artifact(region: str, analysis: dict[str, Any]) -> None:
    path = ROOT / "results" / "region_stratified_drift_000469.json"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist yet -- run "
            "`run_human_drift_spine_000469.py --region-stratified` first, before the "
            "positive-control extension."
        )
    artifact = json.loads(path.read_text())
    region_block = artifact["regions"].get(region)
    if region_block is None:
        return
    for key, session in analysis["sessions"].items():
        if session.get("status") != "complete" or key not in region_block["sessions"]:
            continue
        existing_folds = region_block["sessions"][key].get("folds", [])
        for control_fold in session["folds"]:
            fold = next((row for row in existing_folds if row.get("fold") == control_fold["fold"]), None)
            if fold is not None:
                fold["positive_controls"] = {k: v for k, v in control_fold.items() if k != "fold"}
    region_block["positive_controls"] = {
        "entity_metrics": analysis["entity_metrics"], "group": analysis["group"],
        "independent_unit": "patient",
    }
    path.write_text(canonical_json(artifact))


def paired_entity_contrast(
    entities_a: dict[str, dict], entities_b: dict[str, dict], metric: str, rng_seed: int = 20260803
) -> dict[str, Any]:
    shared = sorted(set(entities_a) & set(entities_b))
    diffs = [
        entities_a[p][metric] - entities_b[p][metric] for p in shared
        if metric in entities_a[p] and metric in entities_b[p]
    ]
    if len(diffs) < 2:
        return {
            "status": "non_identified",
            "reason": f"fewer than 2 patients contribute both regions for {metric}",
            "n_patients_both_regions": len(diffs),
        }
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(rng_seed)
    ci = bootstrap_summary(list(diffs), rng)
    return {
        "status": "estimable", "metric": metric, "n_patients_both_regions": int(len(diffs)),
        "mean_difference": float(np.mean(diffs)),
        "patient_bootstrap_ci95": ci["patient_bootstrap_interval_95"],
        "direction_a_greater_than_b": bool(ci["patient_bootstrap_interval_95"][0] > 0),
    }


def run_region_stratified() -> None:
    """The matched-flexibility half of the deciding contrast:
    M2 minus scale-mixture M0 and M2 minus free-variance AR(1),
    per region, plus the hippocampus-minus-amygdala paired difference.
    This function only loops 000469's regions; 000574's region-stratified fit
    is a separate estimand (Boran multi-structure pooling) run by
    scripts/run_000574_units_pipeline.py --region-stratified, not here.
    """
    region_analyses: dict[str, Any] = {}
    for region in REGIONS_WITH_POOLED:
        analysis = analyze_dataset("000469", region=region)
        region_analyses[region] = analysis
        extend_region_stratified_artifact(region, analysis)

    metric_pairs = (
        ("raw.content_axis.m2_minus_heteroscedastic_m0_nats_per_transition", "m2_minus_heteroscedastic_m0"),
        ("raw.content_axis.m2_minus_free_variance_ar1_m0_nats_per_transition", "m2_minus_free_variance_ar1_m0"),
    )
    hippocampus_entities = region_analyses["hippocampus"]["entity_metrics"]
    amygdala_entities = region_analyses["amygdala"]["entity_metrics"]
    m2_matched_flexibility_contrast = {
        short_name: paired_entity_contrast(hippocampus_entities, amygdala_entities, metric)
        for metric, short_name in metric_pairs
    }

    path = ROOT / "results" / "region_stratified_drift_000469.json"
    artifact = json.loads(path.read_text())
    artifact["deciding_contrast"]["hippocampus_minus_amygdala"]["m2_minus_matched_flexibility"] = {
        "status": "estimable" if any(
            v.get("status") == "estimable" for v in m2_matched_flexibility_contrast.values()
        ) else "non_identified",
        "components": m2_matched_flexibility_contrast,
        "hippocampus_n_patients": len(hippocampus_entities),
        "amygdala_n_patients": len(amygdala_entities),
    }
    path.write_text(canonical_json(artifact))
    print(json.dumps({
        "output": str(path),
        "m2_matched_flexibility_contrast": m2_matched_flexibility_contrast,
    }, indent=2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-stratified", action="store_true",
                         help="Also fit every DANDI 000469 anatomical region's matched-flexibility "
                              "comparators and extend results/region_stratified_drift_000469.json")
    args = parser.parse_args()
    if args.region_stratified:
        run_region_stratified()
        return
    analyses = {dataset: analyze_dataset(dataset) for dataset in ("000469", "000574")}
    for dataset, analysis in analyses.items():
        extend_drift_artifact(dataset, analysis)
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "drift_positive_control_dandi000469_and_dandi000574",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "independent_unit": "patient",
        "predeclared_interpretation": (
            "Trial-specific temporal dependence survives only if M2 retains patient-bootstrap "
            "intervals above zero against both matched-flexibility comparators and own-trial "
            "prediction clearly exceeds neighbouring-trial prediction."
        ),
        "variance_stabilization": {
            "transform": "Anscombe 2*sqrt(count + 3/8) before projection",
            "applies_to": ["DANDI 000469 spike counts", "DANDI 000574 spike counts"],
            "does_not_apply_to": {
                "Miller": "ECoG voltage is continuous rather than count-valued",
                "linked_000673": "LFP voltage is continuous rather than count-valued",
            },
        },
        "datasets": analyses,
        "adjudication": adjudicate_positive_controls(analyses),
        "source_fold_alignment": {
            dataset: audit_source_fold_alignment(dataset, analysis)
            for dataset, analysis in analyses.items()
        },
        "simulation_controls": simulation_controls(),
    }
    destination = ROOT / "results" / "drift_positive_control_000469.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(destination),
        "entities": {key: len(value["entity_metrics"]) for key, value in analyses.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
