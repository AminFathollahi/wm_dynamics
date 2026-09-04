#!/usr/bin/env python3
"""Leakage-free confined-drift analysis for DANDI 000574 (Boran verbal Sternberg).

000574's public NWB release has ``set_letters == "not available"`` on every
trial (see ``scripts/run_boran_pipeline.py`` and
``scripts/run_000574_units_pipeline.py``), so the memorandum identity of each
trial cannot be recovered and the per-item drift/equilibrium analysis run on
DANDI 000469 (``run_human_drift_spine_000469.py``) is not estimable here.
What *is* estimable, and what this script runs, is the same held-out
confinement/diffusion spine conditioned on the one trial-level factor that is
recoverable from every session: Sternberg set size (4/6/8 letters), per the
task's own ``general/experiment_description`` field
(fixation [-6,-5] s, encoding [-5,-3] s, maintenance [-3,0] s relative to the
probe -- i.e. maintenance onset = trial start + 3.0 s for 3.0 s, matching the
window already used by the sibling 000574/Boran pipelines in this repo).

Methodologically this mirrors the 000469 spine exactly: a multiclass
discriminant direction fit only on outer-training trials, leave-one-out
condition-by-time residuals along that direction, and held-out negative log
likelihood of a Gaussian state-space model (M2) against an i.i.d. baseline
(M0) and a two-state switching AR-HMM (M4), scored on the identical held-out
transitions. Sessions are nested within patient; patient is the independent
inferential unit (below this project's own small-N population-inference floor), so session-level fits are first
averaged within each patient before the group-level patient-cluster
bootstrap.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import (  # noqa: E402
    compare_switching_models,
    discriminant_direction,
    fit_gaussian_state_space,
    fit_ou_moments,
    fit_rotation_drift_comparison,
    gaussian_state_space_conditional_log_likelihood,
    gaussian_state_space_log_likelihood,
    iid_log_likelihood,
    leave_one_out_condition_residuals,
    matched_complement_direction,
    projected_residuals,
    summarize_switching_decompositions,
    trial_prediction_advantage,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import (  # noqa: E402
    BORAN_ANATOMICAL_REGIONS,
    BORAN_REGIONS_WITH_POOLED,
    FrozenPSTHTransform,
    MIN_UNIT_FIRING_RATE_HZ,
    MIN_UNITS_PER_REGION,
    build_psth,
    filter_units_by_region,
    load_spike_times,
    low_rate_unit_mask,
    resolve_unit_regions,
)

BIN_MS = 100
MAINT_ONSET_S = 3.0
MAINT_WIN = 3.0
N_COMPONENTS = 8
N_SPLITS = 5
MIN_UNITS = 8  # Boran microwire yield is lower than the Rutishauser bundles (000469 uses 15)
MIN_TRIALS = 20
LABEL_CONVENTION = "nwb_boran_brainnetome_hybrid"
RULE_PATH = ROOT / "preregistration" / "rotation_drift_decision_rule.json"
RULE_HASH = "c9505c80aed6b6c82494e472991a519c46a60a00bd8bfab7e6375f0706dc0ecd"


def data_directory() -> Path:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    data_root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not data_root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = Path(data_root) / config["datasets"]["dandi_000574"]["local_path"]
    if not path.is_dir():
        raise SystemExit(f"DANDI 000574 is not staged at configured path: {path}")
    return path


def analyze_session(path: Path, seed: int, region: str = "pooled") -> dict:
    """Fit the confined-drift spine for one 000574 session, conditioned on set size."""
    with h5py.File(path, "r") as handle:
        if "units" not in handle:
            return {"status": "excluded", "reason": "no units table in this NWB file"}
        spike_lists = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle, LABEL_CONVENTION)["region"]
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
        set_size = trials["set_size"][:].astype(int)
        correct = trials["correct"][:].astype(bool)
        start_time = trials["start_time"][:]
    good = ~artifact
    set_size, correct, start_time = set_size[good], correct[good], start_time[good]
    maint_onsets = start_time + MAINT_ONSET_S
    n_trials = len(maint_onsets)
    if n_trials < MIN_TRIALS:
        return {"status": "excluded", "reason": f"only {n_trials} artifact-free trials"}
    n_units_in_region = int(np.sum(unit_regions == region)) if region != "pooled" else len(spike_lists)
    spike_lists = filter_units_by_region(spike_lists, unit_regions, region)
    rate_mask = low_rate_unit_mask(spike_lists, maint_onsets, MAINT_WIN)
    spike_lists = [spikes for spikes, keep in zip(spike_lists, rate_mask) if keep]
    min_units = MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION
    if len(spike_lists) < min_units:
        return {
            "status": "non_identified" if region != "pooled" else "excluded",
            "reason": f"only {len(spike_lists)} units after firing-rate QC (region={region})",
            "region": region, "n_units_in_region_before_rate_qc": n_units_in_region,
        }
    counts = {int(size): int(np.sum(set_size == size)) for size in np.unique(set_size)}
    usable_classes = [size for size, count in counts.items() if count >= N_SPLITS]
    if len(usable_classes) < 2:
        return {"status": "excluded", "reason": "fewer than two set-size levels support five folds", "counts": counts}
    keep = np.isin(set_size, usable_classes)
    labels = set_size[keep]
    accuracy = correct[keep]
    onsets = maint_onsets[keep]

    psth = build_psth(spike_lists, onsets, bin_ms=BIN_MS, smooth_ms=0, window_s=MAINT_WIN)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_rows = []
    out_of_fold_probe = np.full(len(labels), np.nan)
    out_of_fold_onset = np.full(len(labels), np.nan)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        transform = FrozenPSTHTransform().fit(psth[train_index])
        train_standardized = transform.transform(psth[train_index]).transpose(0, 2, 1)
        test_standardized = transform.transform(psth[test_index]).transpose(0, 2, 1)
        pca = PCA(n_components=min(N_COMPONENTS, len(spike_lists) - 1, len(train_index) - 1), svd_solver="full")
        pca.fit(train_standardized.reshape(-1, train_standardized.shape[-1]))
        train_state = pca.transform(train_standardized.reshape(-1, train_standardized.shape[-1])).reshape(
            len(train_index), psth.shape[2], -1,
        )
        test_state = pca.transform(test_standardized.reshape(-1, test_standardized.shape[-1])).reshape(
            len(test_index), psth.shape[2], -1,
        )
        train_window = train_state.mean(axis=1)
        direction = discriminant_direction(train_window, labels[train_index])
        train_residuals, test_residuals = projected_residuals(
            train_state, test_state, labels[train_index], labels[test_index], direction,
        )
        state_estimate = fit_gaussian_state_space(train_residuals, BIN_MS / 1000.0)
        moment_estimate = fit_ou_moments(
            train_residuals, BIN_MS / 1000.0, n_boot=120,
            rng=np.random.default_rng(seed + fold_index + 1000),
        )
        m2_log_likelihood = gaussian_state_space_log_likelihood(
            test_residuals, state_estimate, BIN_MS / 1000.0,
        )
        m2_transition_log_likelihood = gaussian_state_space_conditional_log_likelihood(
            test_residuals, state_estimate, BIN_MS / 1000.0,
        )
        switching_comparison = compare_switching_models(
            train_residuals, test_residuals, BIN_MS / 1000.0, n_restarts=4,
            rng=np.random.default_rng(seed + fold_index + 6000),
        )
        m4_transition_score = switching_comparison["free_log_likelihood_per_transition"]
        m0_log_likelihood = iid_log_likelihood(test_residuals, train_residuals)
        rotation_comparison = fit_rotation_drift_comparison(
            train_state, test_state, labels[train_index], labels[test_index], direction,
            train_residuals, test_residuals,
            m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1),
            BIN_MS / 1000.0,
            rng=np.random.default_rng(seed + fold_index + 7000),
        )
        prediction = trial_prediction_advantage(train_residuals, test_residuals)
        out_of_fold_probe[test_index] = np.abs(test_residuals[:, -1])
        out_of_fold_onset[test_index] = np.abs(test_residuals[:, 0])

        permutation_rng = np.random.default_rng(seed + fold_index + 2000)
        permuted_labels = permutation_rng.permutation(labels[train_index])
        permuted_direction = discriminant_direction(train_window, permuted_labels)
        permuted_train, _ = projected_residuals(
            train_state, test_state, permuted_labels, labels[test_index], permuted_direction,
        )
        permuted_estimate = fit_ou_moments(
            permuted_train, BIN_MS / 1000.0, n_boot=80,
            rng=np.random.default_rng(seed + fold_index + 3000),
        )
        complement, target_signal_variance, complement_signal_variance = matched_complement_direction(
            train_window, labels[train_index], direction,
            np.random.default_rng(seed + fold_index + 4000),
        )
        complement_train, _ = projected_residuals(
            train_state, test_state, labels[train_index], labels[test_index], complement,
        )
        complement_estimate = fit_ou_moments(
            complement_train, BIN_MS / 1000.0, n_boot=80,
            rng=np.random.default_rng(seed + fold_index + 5000),
        )
        fold_rows.append({
            "fold": fold_index,
            "n_train": int(len(train_index)), "n_test": int(len(test_index)),
            "state_space": state_estimate.to_dict(), "moment": moment_estimate.to_dict(),
            "permuted_axis_moment": permuted_estimate.to_dict(),
            "matched_complement_moment": complement_estimate.to_dict(),
            "content_signal_variance": target_signal_variance,
            "complement_signal_variance": complement_signal_variance,
            "m0_log_likelihood_per_observation": m0_log_likelihood / test_residuals.size,
            "m2_log_likelihood_per_observation": m2_log_likelihood / test_residuals.size,
            "m2_minus_m0_nats_per_observation": (m2_log_likelihood - m0_log_likelihood) / test_residuals.size,
            "m2_log_likelihood_per_transition_conditional": (
                m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1)
            ),
            "m4_log_likelihood_per_transition": (
                m4_transition_score
            ),
            "m4_minus_m2_nats_per_transition": (
                m4_transition_score
                - m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1)
            ),
            "switching_two_state": switching_comparison["free"],
            "rotation_comparison": rotation_comparison,
            "m1_minus_m0_nats_per_observation": rotation_comparison[
                "m1_minus_m0_nats_per_observation"
            ],
            "m3_minus_m2_nats_per_transition": rotation_comparison[
                "m3_minus_m2_nats_per_transition"
            ],
            "switching_decomposition": switching_comparison["free_decomposition"],
            "switching_tied_variance": switching_comparison["tied_variance"],
            "m4_tied_log_likelihood_per_transition": switching_comparison[
                "tied_log_likelihood_per_transition"
            ],
            "heteroscedastic_drift": switching_comparison["heteroscedastic_drift"],
            "heteroscedastic_drift_log_likelihood_per_transition": switching_comparison[
                "heteroscedastic_drift_log_likelihood_per_transition"
            ],
            "m4_minus_heteroscedastic_drift_nats_per_transition": (
                m4_transition_score
                - switching_comparison["heteroscedastic_drift_log_likelihood_per_transition"]
            ),
            "trial_prediction": prediction,
        })
    state_lambdas_all = np.array([
        row["state_space"]["lambda_rate"] for row in fold_rows
        if row["state_space"]["lambda_rate"] is not None
    ])
    moment_lambdas_all = np.array([
        row["moment"]["lambda_rate"] for row in fold_rows
        if row["moment"]["lambda_rate"] is not None
    ])
    state_lambdas = np.array([
        row["state_space"]["lambda_rate"] for row in fold_rows
        if row["state_space"]["status"] == "identifiable"
    ])
    moment_lambdas = np.array([
        row["moment"]["lambda_rate"] for row in fold_rows
        if row["moment"]["status"] == "identifiable"
    ])
    content_minus_permuted = np.array([
        row["moment"]["lambda_rate"] - row["permuted_axis_moment"]["lambda_rate"]
        for row in fold_rows
        if row["moment"]["status"] == "identifiable"
        and row["permuted_axis_moment"]["status"] == "identifiable"
    ])
    content_minus_complement = np.array([
        row["moment"]["lambda_rate"] - row["matched_complement_moment"]["lambda_rate"]
        for row in fold_rows
        if row["moment"]["status"] == "identifiable"
        and row["matched_complement_moment"]["status"] == "identifiable"
    ])
    both_identifiable = [
        row for row in fold_rows
        if row["state_space"]["status"] == "identifiable"
        and row["moment"]["status"] == "identifiable"
    ]
    interval_agreements = []
    for row in both_identifiable:
        state_ci = row["state_space"]["lambda_ci"]
        moment_ci = row["moment"]["lambda_ci"]
        if state_ci is not None and moment_ci is not None:
            interval_agreements.append(max(state_ci[0], moment_ci[0]) <= min(state_ci[1], moment_ci[1]))
    errors = int(np.sum(~accuracy))
    return {
        "status": "complete",
        "n_trials": int(len(labels)), "n_units": int(len(spike_lists)),
        "n_set_size_levels": int(len(usable_classes)), "set_size_counts": counts,
        "n_errors": errors, "accuracy": float(np.mean(accuracy)), "bin_ms": BIN_MS,
        "smoothed": False, "condition": "set_size",
        "folds": fold_rows,
        "summary": {
            "state_space_lambda_identified_mean": (
                float(np.mean(state_lambdas)) if len(state_lambdas) else None
            ),
            "moment_lambda_identified_mean": (
                float(np.mean(moment_lambdas)) if len(moment_lambdas) else None
            ),
            "state_space_identifiable_folds": int(len(state_lambdas)),
            "moment_identifiable_folds": int(len(moment_lambdas)),
            "all_fit_state_space_lambda_mean_not_for_inference": float(np.mean(state_lambdas_all)),
            "all_fit_moment_lambda_mean_not_for_inference": float(np.mean(moment_lambdas_all)),
            "lambda_content_minus_permuted_identified_mean": (
                float(np.mean(content_minus_permuted)) if len(content_minus_permuted) else None
            ),
            "lambda_content_minus_permuted_identified_folds": int(len(content_minus_permuted)),
            "lambda_content_minus_complement_identified_mean": (
                float(np.mean(content_minus_complement)) if len(content_minus_complement) else None
            ),
            "lambda_content_minus_complement_identified_folds": int(len(content_minus_complement)),
            "state_moment_both_identifiable_folds": int(len(both_identifiable)),
            "state_moment_interval_agreement_folds": int(np.sum(interval_agreements)),
            "state_moment_intervals_compared": int(len(interval_agreements)),
            "m2_minus_m0_nats_per_observation": float(np.mean([
                row["m2_minus_m0_nats_per_observation"] for row in fold_rows
            ])),
            "m4_minus_m2_nats_per_transition": float(np.mean([
                row["m4_minus_m2_nats_per_transition"] for row in fold_rows
            ])),
            "m1_minus_m0_nats_per_observation": float(np.mean([
                row["m1_minus_m0_nats_per_observation"] for row in fold_rows
            ])),
            "m3_minus_m2_nats_per_transition": float(np.mean([
                row["m3_minus_m2_nats_per_transition"] for row in fold_rows
            ])),
            "counter_rotation_accuracy_recovery": float(np.mean([
                row["rotation_comparison"]["counter_rotation"]["accuracy_recovery"]
                for row in fold_rows
            ])),
            "m4_tied_minus_m2_nats_per_transition": float(np.mean([
                row["m4_tied_log_likelihood_per_transition"]
                - row["m2_log_likelihood_per_transition_conditional"] for row in fold_rows
            ])),
            "m4_minus_heteroscedastic_drift_nats_per_transition": float(np.mean([
                row["m4_minus_heteroscedastic_drift_nats_per_transition"] for row in fold_rows
            ])),
            "switching_complete_folds": int(sum(
                row["switching_two_state"]["status"] == "complete" for row in fold_rows
            )),
            "trial_prediction_r2_advantage": float(np.mean([
                row["trial_prediction"]["held_out_r2_advantage"] for row in fold_rows
            ])),
        },
        "behavior": {
            "delay_onset_absolute_residual": out_of_fold_onset.tolist(),
            "probe_absolute_residual": out_of_fold_probe.tolist(),
            "accuracy": accuracy.astype(int).tolist(),
            "status": "estimable" if errors >= 5 else "underpowered",
            "reason": None if errors >= 5 else f"only {errors} errors among usable trials",
        },
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draws = np.mean(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def patient_level_means(sessions: dict, metric_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Average each metric within patient before any cross-patient inference (advisor N4/10.4)."""
    by_patient: dict[str, list[dict]] = {}
    for key, row in sessions.items():
        if row["status"] != "complete":
            continue
        patient = key.split("_ses-")[0]
        by_patient.setdefault(patient, []).append(row)
    patient_metrics: dict[str, dict[str, float]] = {}
    for patient, rows in by_patient.items():
        patient_metrics[patient] = {}
        for name in metric_names:
            values = [row["summary"][name] for row in rows if row["summary"][name] is not None]
            patient_metrics[patient][name] = float(np.mean(values)) if values else None
    return patient_metrics


METRIC_NAMES = (
    "state_space_lambda_identified_mean", "moment_lambda_identified_mean",
    "lambda_content_minus_permuted_identified_mean",
    "lambda_content_minus_complement_identified_mean",
    "m2_minus_m0_nats_per_observation", "m4_minus_m2_nats_per_transition",
    "m1_minus_m0_nats_per_observation", "m3_minus_m2_nats_per_transition",
    "counter_rotation_accuracy_recovery",
    "m4_tied_minus_m2_nats_per_transition",
    "m4_minus_heteroscedastic_drift_nats_per_transition",
    "trial_prediction_r2_advantage",
)


def fit_region(directory: Path, region: str, rng_seed: int = 20260731) -> tuple[dict, dict, dict]:
    """Fit every DANDI 000574 session for one region (or ``"pooled"``).

    Same sessions, same per-session seed mapping, same folds regardless of
    region -- only which units feed the population vector changes.
    Returns ``(sessions, patient_metrics, group)``.
    """
    sessions = {}
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            key = path.stem
            print(f"fitting {key} region={region}", flush=True)
            seed = 20260731 + int(subject_dir.name.split("-")[1]) * 100 + int(path.stem.split("-")[-1])
            sessions[key] = analyze_session(path, seed=seed, region=region)
    complete = {key: row for key, row in sessions.items() if row["status"] == "complete"}
    patient_metrics = patient_level_means(sessions, METRIC_NAMES)
    if not complete:
        return sessions, patient_metrics, {
            "status": "non_identified",
            "reason": f"no DANDI 000574 session-region cleared the minimum unit count for region={region}",
        }
    group = {}
    rng = np.random.default_rng(rng_seed)
    for name in METRIC_NAMES:
        values = np.array([
            row[name] for row in patient_metrics.values() if row[name] is not None
        ])
        if len(values) < 2:
            group[name] = {
                "status": "not_estimable", "n_patients": int(len(values)),
                "reason": f"fewer than 2 patients have an identified estimate ({len(values)})",
            }
            continue
        group[name] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "ci": bootstrap_mean(values, rng), "n_patients": int(len(values)),
            "values": values.tolist(),
        }
    threshold = json.loads(RULE_PATH.read_text())["minimum_effect"]["negative_log_likelihood_nats_per_observation"]
    m2 = group["m2_minus_m0_nats_per_observation"]
    m2["passes_practical_threshold"] = bool(
        m2.get("n_patients", 0) >= 2 and m2["mean"] >= threshold and m2["ci"][0] > 0
    )
    m4 = group["m4_minus_m2_nats_per_transition"]
    m4["passes_practical_threshold"] = bool(
        m4.get("n_patients", 0) >= 2 and m4["mean"] >= threshold and m4["ci"][0] > 0
        and np.mean(np.asarray(m4["values"]) > 0) >= 0.60
    )
    group["switching_decomposition"] = summarize_switching_decompositions([
        fold for session in complete.values() for fold in session["folds"]
    ])
    group["identifiability"] = {
        "state_space_folds_identifiable": int(sum(
            row["summary"]["state_space_identifiable_folds"] for row in complete.values()
        )),
        "moment_folds_identifiable": int(sum(
            row["summary"]["moment_identifiable_folds"] for row in complete.values()
        )),
        "total_folds": int(N_SPLITS * len(complete)),
        "patients_with_state_space_identification": int(sum(
            any(row["summary"]["state_space_identifiable_folds"] > 0
                for key, row in complete.items() if key.split("_ses-")[0] == patient)
            for patient in patient_metrics
        )),
        "n_patients": len(patient_metrics),
    }
    return sessions, patient_metrics, group


def write_pooled_artifact(sessions: dict, patient_metrics: dict, group: dict) -> Path:
    complete = {key: row for key, row in sessions.items() if row["status"] == "complete"}
    output = {
        "schema_version": "1.0.0", "analysis_id": "human_drift_spine_dandi000574",
        "dataset": "DANDI 000574 (Boran verbal Sternberg)",
        "canonical_role": "replication of the confinement/diffusion spine at scale",
        "item_identity_available": False,
        "item_identity_reason": (
            "set_letters is 'not available' on every trial in the public NWB release; "
            "per-item drift/equilibria cannot be estimated for this dataset"
        ),
        "condition_used": "set_size (4/6/8 letters)",
        "independent_unit": "patient, with sessions nested within patient",
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "coordinate": "FrozenPSTHTransform and PCA fit within each outer-training fold",
        "smoothing": "none",
        "maintenance_window": {"onset_s_from_trial_start": MAINT_ONSET_S, "duration_s": MAINT_WIN,
                               "source": "general/experiment_description in the NWB file (maintenance = [-3, 0] s relative to probe)"},
        "sessions": sessions, "patient_level_metrics": patient_metrics, "group": group,
        "model_status": {
            "M0": "scored", "M1": "scored as a training-fold axis-rotation operator",
            "M2": "scored", "M3": "scored as rotation plus confined residual drift",
            "M4": "scored as a probabilistic two-state Gaussian AR-HMM; not a full Poisson rSLDS",
        },
        "pooled_result_superseded_note": (
            "This fit pools every recorded single unit in a session into one population vector "
            "regardless of anatomical structure -- hippocampus, amygdala, entorhinal/parahippocampal "
            "cortex, inferior/middle/superior temporal gyrus and unspecific-depth channels together "
            "(the anatomical census, results/anatomical_census.json, resolves all of these as "
            "distinct structures in this corpus). That is the chimeric-pool N10 violation this project's "
            "own anatomical-pooling prohibition flags. Superseded by "
            "results/region_stratified_drift_000574.json as of 2026-08-04: "
            "the group.state_space_lambda_identified_mean.median reported in THIS file (including the "
            "3.67 s^-1 figure quoted in any manuscript draft referencing "
            "'the DANDI 000574 confinement rate') describes a mixture over multiple anatomical "
            "structures, not any single structure's dynamics, and is retained here only as the "
            "pre-stratification baseline."
        ),
    }
    destination = ROOT / "results" / "human_drift_spine_000574.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"n_complete_sessions": len(complete), "n_patients": len(patient_metrics),
                      "group": group, "output": str(destination)}, indent=2))
    return destination


def paired_region_contrast(regions_group: dict[str, dict], region_a: str, region_b: str, metric: str) -> dict:
    """Within-patient region_a-minus-region_b difference, restricted to
    patients contributing both regions (mirrors the 000469
    deciding contrast in run_human_drift_spine_000469.py)."""
    metrics_a = regions_group[region_a]["patient_level_metrics"]
    metrics_b = regions_group[region_b]["patient_level_metrics"]
    shared_patients = sorted(set(metrics_a) & set(metrics_b))
    diffs = [
        metrics_a[p][metric] - metrics_b[p][metric]
        for p in shared_patients
        if metrics_a[p].get(metric) is not None and metrics_b[p].get(metric) is not None
    ]
    if len(diffs) < 2:
        return {
            "status": "non_identified",
            "reason": f"fewer than 2 patients contribute both {region_a} and {region_b}",
            "n_patients_both_regions": len(diffs),
        }
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(20260804)
    ci = bootstrap_mean(diffs, rng)
    return {
        "status": "estimable",
        "region_a": region_a, "region_b": region_b, "metric": metric,
        "n_patients_both_regions": int(len(diffs)),
        "mean_difference": float(np.mean(diffs)),
        "median_difference": float(np.median(diffs)),
        "patient_bootstrap_ci95": ci,
        "direction_a_greater_than_b": bool(ci[0] > 0),
    }


def run_region_stratified(directory: Path, pooled: tuple[dict, dict, dict] | None = None) -> Path:
    """Refit DANDI 000574 per anatomical structure instead
    of pooling every unit in a session into one state."""
    regions_out: dict[str, dict] = {}
    for region in BORAN_REGIONS_WITH_POOLED:
        if region == "pooled" and pooled is not None:
            sessions, patient_metrics, group = pooled
        else:
            sessions, patient_metrics, group = fit_region(directory, region=region)
        n_session_regions_identified = sum(1 for row in sessions.values() if row["status"] == "complete")
        n_session_regions_non_identified = sum(
            1 for row in sessions.values() if row["status"] in ("non_identified", "excluded")
        )
        regions_out[region] = {
            "sessions": sessions, "patient_level_metrics": patient_metrics, "group": group,
            "n_session_regions_identified": n_session_regions_identified,
            "n_session_regions_non_identified": n_session_regions_non_identified,
            "n_patients_identified": len(patient_metrics),
            "min_units_threshold": MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION,
        }

    deciding_contrast = {
        "hippocampus_minus_amygdala": paired_region_contrast(
            regions_out, "hippocampus", "amygdala", "trial_prediction_r2_advantage",
        ),
        "predeclared_direction": (
            "hippocampus > amygdala on held-out trial-prediction R2 advantage, mirroring the "
            "DANDI 000469 deciding contrast"
        ),
    }

    region_lambda_medians = {}
    for region in BORAN_ANATOMICAL_REGIONS:
        group = regions_out[region]["group"]
        state_space = group.get("state_space_lambda_identified_mean") if isinstance(group, dict) else None
        # A single-patient point estimate must never rank in this ordering --
        # its bootstrap CI is numerically well-defined at n=1 but not decisive.
        if isinstance(state_space, dict) and state_space.get("median") is not None and state_space.get("n_patients", 0) >= 2:
            region_lambda_medians[region] = state_space["median"]
    if len(region_lambda_medians) >= 2:
        ordered = sorted(region_lambda_medians, key=region_lambda_medians.get, reverse=True)
        lambda_ordering = {
            "status": "estimable",
            "region_lambda_state_space_median": region_lambda_medians,
            "region_order_fastest_to_slowest": ordered,
            "note": (
                "Rank-based ordering of per-region state-space lambda medians (patient n varies by "
                "region, see each region's group.state_space_lambda_identified_mean.n_patients). Not "
                "a fitted hierarchy model."
            ),
        }
    else:
        lambda_ordering = {
            "status": "non_identified",
            "reason": "fewer than 2 regions had an identifiable state-space lambda median",
        }

    output = {
        "schema_version": "1.0.0", "analysis_id": "region_stratified_drift_dandi000574",
        "dataset": "DANDI 000574 (Boran verbal Sternberg)",
        "condition_used": "set_size (4/6/8 letters)",
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "min_units_per_region_threshold": MIN_UNITS_PER_REGION,
        "estimands_present": [
            "state_space_lambda_identified_mean (M2)", "moment_lambda_identified_mean",
            "lambda_content_minus_permuted_identified_mean",
            "lambda_content_minus_complement_identified_mean",
            "m2_minus_m0_nats_per_observation", "m4_minus_m2_nats_per_transition (switching)",
            "m1_minus_m0_nats_per_observation (rotation)",
            "m3_minus_m2_nats_per_transition (rotation plus confined drift)",
            "counter_rotation_accuracy_recovery", "trial_prediction_r2_advantage",
            "switching_decomposition",
        ],
        "regions": regions_out,
        "deciding_contrast": deciding_contrast,
        "lambda_regional_ordering": lambda_ordering,
        "pooled_result_superseded_note": (
            "regions.pooled reproduces results/human_drift_spine_000574.json exactly (same code path, "
            "region='pooled' is a no-op filter) and is retained inline as the pre-stratification "
            "baseline -- not removed, only superseded by the per-structure headline below."
        ),
    }
    destination = ROOT / "results" / "region_stratified_drift_000574.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(destination),
        "regions": {r: regions_out[r]["n_patients_identified"] for r in BORAN_REGIONS_WITH_POOLED},
        "deciding_contrast": deciding_contrast,
        "lambda_regional_ordering": lambda_ordering,
    }, indent=2))
    return destination


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-stratified", action="store_true",
                         help="Also fit every anatomical region and write results/region_stratified_drift_000574.json")
    args = parser.parse_args()
    if sha256_file(RULE_PATH) != RULE_HASH:
        raise SystemExit("frozen adjudication rule hash mismatch; refusing to fit real data")
    directory = data_directory()
    sessions, patient_metrics, group = fit_region(directory, region="pooled")
    if not any(row["status"] == "complete" for row in sessions.values()):
        raise SystemExit("no DANDI 000574 session completed the drift fit")
    write_pooled_artifact(sessions, patient_metrics, group)
    if args.region_stratified:
        run_region_stratified(directory, pooled=(sessions, patient_metrics, group))


if __name__ == "__main__":
    main()
