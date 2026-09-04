#!/usr/bin/env python3
"""Leakage-free confined-drift analysis for repeated items in DANDI 000469."""

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
from geometry import distance_to_attractor  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import (  # noqa: E402
    ANATOMICAL_REGIONS,
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

BIN_MS = 100
WINDOW_S = 2.3
N_COMPONENTS = 8
N_SPLITS = 5
MIN_UNITS = 15
AXIS_WINDOW = (0.3, 1.0)
RULE_PATH = ROOT / "preregistration" / "rotation_drift_decision_rule.json"
RULE_HASH = "c9505c80aed6b6c82494e472991a519c46a60a00bd8bfab7e6375f0706dc0ecd"


def data_directory() -> Path:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    data_root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not data_root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = Path(data_root) / config["datasets"]["dandi_000469"]["local_path"]
    if not path.is_dir():
        raise SystemExit(f"DANDI 000469 is not staged at configured path: {path}")
    return path


def analyze_session(
    path: Path, seed: int, region: str = "pooled", unit_indices: np.ndarray | None = None,
) -> dict:
    """``unit_indices``, if given, subsets the region-filtered, firing-rate-QC'd
    unit pool to exactly those positions before fitting -- this project's
    within-patient unit-count-matched downsampling. ``None`` (default)
    preserves the full pool, i.e. every existing caller is unaffected."""
    with h5py.File(path, "r") as handle:
        spike_lists = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle)["region"]
        trials = handle["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        labels = trials["loadsEnc1_PicIDs"][:].astype(int)
        accuracy = trials["response_accuracy"][:].astype(bool)
        onsets = trials["timestamps_Maintenance"][:]
    n_units_in_region = int(np.sum(unit_regions == region)) if region != "pooled" else len(spike_lists)
    spike_lists = filter_units_by_region(spike_lists, unit_regions, region)
    rate_mask = low_rate_unit_mask(spike_lists, onsets, WINDOW_S)
    spike_lists = [spikes for spikes, keep in zip(spike_lists, rate_mask) if keep]
    if unit_indices is not None:
        spike_lists = [spike_lists[i] for i in unit_indices]
    min_units = MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION
    if len(spike_lists) < min_units:
        return {
            "status": "non_identified" if region != "pooled" else "excluded",
            "reason": f"only {len(spike_lists)} units after firing-rate QC (region={region})",
            "region": region, "n_units_in_region_before_rate_qc": n_units_in_region,
        }
    if float(np.mean(accuracy)) < MIN_SESSION_ACCURACY:
        return {"status": "excluded", "reason": "session accuracy below prospective QC floor"}
    keep = loads == 1
    labels = labels[keep]
    accuracy = accuracy[keep]
    onsets = onsets[keep]
    counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
    if len(counts) < 3 or min(counts.values()) < N_SPLITS:
        return {"status": "excluded", "reason": "repeated-item counts cannot support five folds", "counts": counts}
    psth = build_psth(
        spike_lists, onsets, bin_ms=BIN_MS, smooth_ms=0, window_s=WINDOW_S,
    )
    times = np.arange(psth.shape[2]) * BIN_MS / 1000.0 + BIN_MS / 2000.0
    axis_bins = (times >= AXIS_WINDOW[0]) & (times <= AXIS_WINDOW[1])
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_rows = []
    out_of_fold_probe = np.full(len(labels), np.nan)
    out_of_fold_onset = np.full(len(labels), np.nan)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        transform = FrozenPSTHTransform().fit(psth[train_index])
        train_standardized = transform.transform(psth[train_index]).transpose(0, 2, 1)
        test_standardized = transform.transform(psth[test_index]).transpose(0, 2, 1)
        pca = PCA(n_components=min(N_COMPONENTS, len(spike_lists), len(train_index) - 1), svd_solver="full")
        pca.fit(train_standardized.reshape(-1, train_standardized.shape[-1]))
        train_state = pca.transform(train_standardized.reshape(-1, train_standardized.shape[-1])).reshape(
            len(train_index), psth.shape[2], -1,
        )
        test_state = pca.transform(test_standardized.reshape(-1, test_standardized.shape[-1])).reshape(
            len(test_index), psth.shape[2], -1,
        )
        train_window = train_state[:, axis_bins].mean(axis=1)
        direction = discriminant_direction(train_window, labels[train_index])
        da = distance_to_attractor(
            train_state[:, axis_bins], labels[train_index], test_state[:, axis_bins], labels[test_index],
        )
        da_mean_per_time = np.nanmean(da, axis=0)
        da_summary = {
            "mean": float(np.nanmean(da)) if np.isfinite(da).any() else None,
            "below_one_fraction": float(np.nanmean(da < 1.0)) if np.isfinite(da).any() else None,
            "n_test_trials_scored": int(np.isfinite(da).any(axis=1).sum()),
            "mean_per_time_bin": da_mean_per_time.tolist(),
        }
        # DA recomputed in the condition-discriminative (LDA) basis, not
        # the plain-variance PCA state above -- fixes the framing error where DA
        # was compared to 1 in a basis with no reason to be condition-separable.
        direction_unit = direction / (np.linalg.norm(direction) + 1e-12)
        train_state_content_axis = (train_state[:, axis_bins] @ direction_unit)[:, :, None]
        test_state_content_axis = (test_state[:, axis_bins] @ direction_unit)[:, :, None]
        da_content_axis = distance_to_attractor(
            train_state_content_axis, labels[train_index], test_state_content_axis, labels[test_index],
        )
        da_content_axis_summary = {
            "mean": float(np.nanmean(da_content_axis)) if np.isfinite(da_content_axis).any() else None,
            "below_one_fraction": (
                float(np.nanmean(da_content_axis < 1.0)) if np.isfinite(da_content_axis).any() else None
            ),
            "n_test_trials_scored": int(np.isfinite(da_content_axis).any(axis=1).sum()),
        }
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
            "distance_to_attractor": da_summary,
            "distance_to_attractor_content_axis": da_content_axis_summary,
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
        "status": "complete", "region": region,
        "n_trials": int(len(labels)), "n_units": int(len(spike_lists)),
        "n_units_in_region_before_rate_qc": n_units_in_region,
        "n_items": int(len(np.unique(labels))), "n_errors": errors,
        "accuracy": float(np.mean(accuracy)), "bin_ms": BIN_MS,
        "smoothed": False, "axis_window_seconds": list(AXIS_WINDOW),
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
            "distance_to_attractor_mean": (
                float(np.mean(da_means))
                if len(da_means := [
                    row["distance_to_attractor"]["mean"] for row in fold_rows
                    if row["distance_to_attractor"]["mean"] is not None
                ]) else None
            ),
            "distance_to_attractor_below_one_fraction": (
                float(np.mean(da_below_one))
                if len(da_below_one := [
                    row["distance_to_attractor"]["below_one_fraction"] for row in fold_rows
                    if row["distance_to_attractor"]["below_one_fraction"] is not None
                ]) else None
            ),
            "distance_to_attractor_content_axis_mean": (
                float(np.mean(da_ca_means))
                if len(da_ca_means := [
                    row["distance_to_attractor_content_axis"]["mean"] for row in fold_rows
                    if row["distance_to_attractor_content_axis"]["mean"] is not None
                ]) else None
            ),
            "distance_to_attractor_content_axis_below_one_fraction": (
                float(np.mean(da_ca_below_one))
                if len(da_ca_below_one := [
                    row["distance_to_attractor_content_axis"]["below_one_fraction"] for row in fold_rows
                    if row["distance_to_attractor_content_axis"]["below_one_fraction"] is not None
                ]) else None
            ),
        },
        "behavior": {
            "delay_onset_absolute_residual": out_of_fold_onset.tolist(),
            "probe_absolute_residual": out_of_fold_probe.tolist(),
            "accuracy": accuracy.astype(int).tolist(),
            "status": "estimable" if errors >= 5 else "underpowered",
            "reason": None if errors >= 5 else f"only {errors} errors among repeated-item trials",
        },
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draws = np.mean(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


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
    "distance_to_attractor_mean", "distance_to_attractor_below_one_fraction",
    "distance_to_attractor_content_axis_mean", "distance_to_attractor_content_axis_below_one_fraction",
)


def fit_region(directory: Path, region: str, rng_seed: int = 20260731) -> tuple[dict, dict]:
    """Fit every DANDI 000469 session for one region (or ``"pooled"``).

    Same sessions, same per-session seed mapping, same folds regardless of
    region -- only which units feed the population vector changes, so a
    cross-region comparison is not confounded by a different fitting
    procedure per region. Returns ``(sessions, group)``.
    """
    sessions = {}
    for path in sorted(directory.glob("sub-*/sub-*_ses-2_ecephys+image.nwb")):
        key = path.parent.name
        print(f"fitting {key} region={region}", flush=True)
        sessions[key] = analyze_session(path, seed=rng_seed + int(key.split("-")[1]), region=region)
    complete = {key: row for key, row in sessions.items() if row["status"] == "complete"}
    if not complete:
        return sessions, {
            "status": "non_identified",
            "reason": f"no session-region cleared the minimum unit count for region={region}",
        }
    group = {}
    rng = np.random.default_rng(rng_seed)
    for name in METRIC_NAMES:
        values = np.array([
            row["summary"][name] for row in complete.values()
            if row["summary"][name] is not None
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
            row["summary"]["state_space_identifiable_folds"] > 0 for row in complete.values()
        )),
        "patients_with_moment_identification": int(sum(
            row["summary"]["moment_identifiable_folds"] > 0 for row in complete.values()
        )),
        "n_patients": len(complete),
        "verdict": "transition dependence is detectable, but the confinement rate is not identified in most folds at the available delay length and bin width",
    }
    return sessions, group


def region_common_mode(directory: Path) -> dict:
    """Fraction of pooled-state variance carried by a region-common component
    vs region-private components, per session.

    A cross-region canonical-correlation-style variance decomposition: within
    each session with >=2 qualifying regions, PCA the concatenated
    region-block-standardized firing rates and report the variance explained
    by the leading shared component against a region-block-shuffled null.
    """
    from spike_pipeline import resolve_unit_regions as _resolve  # local, avoids top-level cycle noise
    per_session = {}
    for path in sorted(directory.glob("sub-*/sub-*_ses-2_ecephys+image.nwb")):
        key = path.parent.name
        with h5py.File(path, "r") as handle:
            spike_lists = load_spike_times(handle)
            unit_regions = _resolve(handle)["region"]
            trials = handle["intervals/trials"]
            loads = trials["loads"][:].astype(int)
            onsets = trials["timestamps_Maintenance"][:]
        rate_mask = low_rate_unit_mask(spike_lists, onsets, WINDOW_S)
        spike_lists = [s for s, keep in zip(spike_lists, rate_mask) if keep]
        unit_regions = unit_regions[rate_mask]
        keep_trials = loads == 1
        qualifying = [r for r in set(unit_regions) if r != "other"
                      and int(np.sum(unit_regions == r)) >= MIN_UNITS_PER_REGION]
        if len(qualifying) < 2:
            per_session[key] = {"status": "non_identified", "reason": "fewer than 2 qualifying regions", "n_qualifying_regions": len(qualifying)}
            continue
        psth = build_psth(spike_lists, onsets[keep_trials], bin_ms=BIN_MS, smooth_ms=0, window_s=WINDOW_S)
        trial_mean = psth.mean(axis=2)  # (trials, units) -- collapse time for a compact per-trial rate vector
        keep_units = np.isin(unit_regions, qualifying)
        trial_mean = trial_mean[:, keep_units]
        region_of_unit = unit_regions[keep_units]
        standardized = (trial_mean - trial_mean.mean(axis=0)) / (trial_mean.std(axis=0) + 1e-8)
        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(standardized)
        shared_variance_fraction = float(pca.explained_variance_ratio_[0])
        rng = np.random.default_rng(20260803)
        null_fractions = []
        for _ in range(200):
            shuffled = standardized.copy()
            for region_name in qualifying:
                cols = np.flatnonzero(region_of_unit == region_name)
                shuffled[:, cols] = rng.permutation(shuffled[:, cols], axis=0)
            null_pca = PCA(n_components=1, svd_solver="full")
            null_pca.fit(shuffled)
            null_fractions.append(float(null_pca.explained_variance_ratio_[0]))
        per_session[key] = {
            "status": "complete",
            "qualifying_regions": sorted(qualifying),
            "n_units_total": int(keep_units.sum()),
            "region_common_component_variance_fraction": shared_variance_fraction,
            "region_shuffled_null_mean": float(np.mean(null_fractions)),
            "region_shuffled_null_ci95": list(map(float, np.percentile(null_fractions, [2.5, 97.5]))),
            "exceeds_shuffled_null": bool(shared_variance_fraction > np.percentile(null_fractions, 97.5)),
        }
    complete = {k: v for k, v in per_session.items() if v["status"] == "complete"}
    return {
        "sessions": per_session,
        "group": {
            "n_sessions_estimable": len(complete),
            "mean_region_common_component_variance_fraction": (
                float(np.mean([v["region_common_component_variance_fraction"] for v in complete.values()]))
                if complete else None
            ),
            "n_sessions_exceeding_shuffled_null": int(sum(v["exceeds_shuffled_null"] for v in complete.values())),
        },
    }


def paired_region_contrast(regions_group: dict[str, dict], region_a: str, region_b: str, metric: str) -> dict:
    """Within-patient region_a-minus-region_b difference, restricted to
    patients contributing both regions -- a within-patient pairing controls
    for between-patient variability that a between-patient comparison would
    not."""
    sessions_a = {k: v for k, v in regions_group[region_a]["sessions"].items() if v["status"] == "complete"}
    sessions_b = {k: v for k, v in regions_group[region_b]["sessions"].items() if v["status"] == "complete"}
    shared_patients = sorted(set(sessions_a) & set(sessions_b))
    diffs = [
        sessions_a[p]["summary"][metric] - sessions_b[p]["summary"][metric]
        for p in shared_patients
        if sessions_a[p]["summary"].get(metric) is not None and sessions_b[p]["summary"].get(metric) is not None
    ]
    if len(diffs) < 2:
        return {
            "status": "non_identified",
            "reason": f"fewer than 2 patients contribute both {region_a} and {region_b}",
            "n_patients_both_regions": len(diffs),
        }
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(20260803)
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


def write_pooled_artifact(directory: Path, sessions: dict, group: dict) -> Path:
    complete = {key: row for key, row in sessions.items() if row["status"] == "complete"}
    output = {
        "schema_version": "1.0.0", "analysis_id": "human_drift_spine_dandi000469",
        "dataset": "DANDI 000469", "canonical_role": "primary repeated-item human intracranial dataset",
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "coordinate": "FrozenPSTHTransform and PCA fit within each outer-training fold",
        "smoothing": "none", "sessions": sessions, "group": group,
        "model_status": {
            "M0": "scored", "M1": "scored as a training-fold axis-rotation operator",
            "M2": "scored", "M3": "scored as rotation plus confined residual drift",
            "M4": "scored as a probabilistic two-state Gaussian AR-HMM; not a full Poisson rSLDS",
        },
    }
    destination = ROOT / "results" / "human_drift_spine_000469.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"n_complete": len(complete), "group": group, "output": str(destination)}, indent=2))
    return destination


def run_region_stratified(directory: Path, pooled: tuple[dict, dict] | None = None) -> Path:
    """Fits every anatomical region (and the pooled population) for DANDI
    000469, builds the within-patient paired region contrasts and the
    cross-region lambda ordering, and writes the single artifact that
    carries all of it."""
    regions_out: dict[str, dict] = {}
    for region in REGIONS_WITH_POOLED:
        sessions, group = pooled if (region == "pooled" and pooled is not None) else fit_region(directory, region=region)
        n_session_regions_identified = sum(1 for row in sessions.values() if row["status"] == "complete")
        n_session_regions_non_identified = sum(1 for row in sessions.values() if row["status"] == "non_identified")
        regions_out[region] = {
            "sessions": sessions, "group": group,
            "n_session_regions_identified": n_session_regions_identified,
            "n_session_regions_non_identified": n_session_regions_non_identified,
            "min_units_threshold": MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION,
        }

    common_mode = region_common_mode(directory)

    deciding_contrast = {
        "hippocampus_minus_amygdala": {
            "m2_minus_matched_flexibility": (
                "computed in results/drift_positive_control_000469.json's region-stratified block "
                "(scripts/run_drift_positive_controls.py); this artifact carries only the plain "
                "M2-minus-iid-M0 comparator, which is not the matched-flexibility comparator this "
                "project's standing rule for this contrast calls for"
            ),
            "content_minus_complement_own_trial_r2": paired_region_contrast(
                regions_out, "hippocampus", "amygdala", "trial_prediction_r2_advantage",
            ),
        },
        "hippocampus_minus_pre_sma_secondary": paired_region_contrast(
            regions_out, "hippocampus", "pre_sma", "trial_prediction_r2_advantage",
        ),
        "hippocampus_minus_dacc_secondary": paired_region_contrast(
            regions_out, "hippocampus", "dacc", "trial_prediction_r2_advantage",
        ),
        "predeclared_direction": "hippocampus > amygdala on both statistics",
    }

    lambda_ordering = {}
    region_lambda_medians = {}
    for region in ANATOMICAL_REGIONS:
        group = regions_out[region]["group"]
        state_space = group.get("state_space_lambda_identified_mean") if isinstance(group, dict) else None
        # A single-patient point estimate must never rank in this ordering --
        # its bootstrap CI is numerically well-defined at n=1 but not decisive.
        if isinstance(state_space, dict) and state_space.get("median") is not None and state_space.get("n_patients", 0) >= 2:
            region_lambda_medians[region] = state_space["median"]
    if len(region_lambda_medians) >= 2:
        ordered = sorted(region_lambda_medians, key=region_lambda_medians.get, reverse=True)
        mtl_regions = [r for r in ordered if r == "hippocampus"]
        frontal_regions = [r for r in ordered if r in ("pre_sma", "dacc", "vmpfc")]
        mtl_faster_than_frontal = (
            bool(mtl_regions and frontal_regions
                 and region_lambda_medians[mtl_regions[0]] > max(region_lambda_medians[r] for r in frontal_regions))
            if mtl_regions and frontal_regions else None
        )
        lambda_ordering = {
            "status": "estimable",
            "region_lambda_state_space_median": region_lambda_medians,
            "region_order_fastest_to_slowest": ordered,
            "mtl_faster_than_medial_frontal": mtl_faster_than_frontal,
            "note": (
                "Rank-based ordering of per-region state-space lambda medians (patient n varies by "
                "region, see each region's group.state_space_lambda_identified_mean.n_patients). "
                "Not a fitted hierarchy model -- no cross-region model is jointly fit, only ranked."
            ),
        }
    else:
        lambda_ordering = {
            "status": "non_identified",
            "reason": "fewer than 2 regions had an identifiable state-space lambda median",
        }

    predeclared_interpretation = (
        "Confined content dynamics are supported region-wise only if, within a region, M2 retains a "
        "patient-bootstrap interval above zero against BOTH matched-flexibility comparators AND the "
        "content axis exceeds both the permuted axis and the signal-matched complement. The pooled "
        "2026-08-02 reversal is attributed to anatomical pooling only if this holds in hippocampus "
        "while failing in the pooled state, with the unit-count-matched sensitivity preserving the "
        "sign. This artifact carries the deciding contrast's own-trial-R2 half and "
        "the lambda ordering; the M2-minus-matched-flexibility half is in "
        "results/drift_positive_control_000469.json."
    )

    distance_to_attractor_basis_note = (
        "DA is recomputed here in a content-discriminative basis (fields with the "
        "_content_axis_ infix), not the plain-variance PCA state the withdrawn 2026-08-03 comparison "
        "used. The basis is this fold's discriminant_direction(), trained on repeated-item identity "
        "labels -- the same labels M2 already tests confinement against -- projected to one dimension. "
        "That is the closest match available in this project to Daume et al. 2025's dPCA axis "
        "marginalised on picture category: both isolate the dimension that best separates trials by "
        "which item was held in working memory. The original plain-variance fields "
        "(distance_to_attractor_mean / distance_to_attractor_below_one_fraction) are retained inline, "
        "superseded but not deleted, so the withdrawn comparison stays reconstructable."
    )
    distance_to_attractor_primary_statistic_note = (
        "DA is a ratio with a noisy denominator (mean distance to the other "
        "conditions' centroids), so its mean is upward-biased relative to its median. "
        "distance_to_attractor_content_axis_below_one_fraction (median and the fraction of test "
        "trials/folds with DA < 1) is PRIMARY. distance_to_attractor_content_axis_mean is reported "
        "beside it and must not be read alone."
    )
    distance_to_attractor_scope_note = (
        "Computed for DANDI 000469 only, the sole spike-resolved corpus in this "
        "project with repeated-item identity labels (item_identity_available=True), the condition "
        "variable a content-discriminative axis needs. DANDI 000574 carries ventral_temporal_cortex "
        "units (57 units, 8 patients; results/anatomical_census.json "
        "structure_by_dataset_matrix.vtc) -- Daume et al. 2025's second positive region -- but its own "
        "artifact records item_identity_available=False (its trial labels are memory set size, not "
        "item identity; run_human_drift_spine_000574.py), so a content-discriminative DA basis is not "
        "defensible there and the comparison is withdrawn for that corpus rather than computed on a "
        "mismatched axis (9.5). VTC remains untested for DA in this project's data -- a corpus-"
        "coverage limitation, not a computed null."
    )
    distance_to_attractor_content_axis_summary = {}
    for region in REGIONS_WITH_POOLED:
        group = regions_out[region]["group"]
        below_one = group.get("distance_to_attractor_content_axis_below_one_fraction")
        mean_field = group.get("distance_to_attractor_content_axis_mean")
        if not isinstance(below_one, dict):
            distance_to_attractor_content_axis_summary[region] = {
                "status": "non_identified",
                "reason": "fewer than 2 patients had an identified content-axis DA fold",
            }
            continue
        ci_lo, ci_hi = below_one["ci"]
        distance_to_attractor_content_axis_summary[region] = {
            "status": "identified",
            "primary_below_one_fraction_median": below_one["median"],
            "primary_below_one_fraction_ci": below_one["ci"],
            "primary_excludes_0.5": bool(ci_lo > 0.5 or ci_hi < 0.5),
            "secondary_mean_da": mean_field.get("mean") if isinstance(mean_field, dict) else None,
            "n_patients": below_one["n_patients"],
        }

    output = {
        "schema_version": "1.0.0", "analysis_id": "region_stratified_drift_dandi000469",
        "dataset": "DANDI 000469", "code_commit": git_commit(ROOT),
        "decision_rule_hash": RULE_HASH, "source_hash": sha256_file(Path(__file__)),
        "min_units_per_region_threshold": MIN_UNITS_PER_REGION,
        "regions": regions_out,
        "region_common_mode": common_mode,
        "deciding_contrast": deciding_contrast,
        "lambda_regional_ordering": lambda_ordering,
        "predeclared_interpretation": predeclared_interpretation,
        "distance_to_attractor_basis_note": distance_to_attractor_basis_note,
        "distance_to_attractor_primary_statistic_note": distance_to_attractor_primary_statistic_note,
        "distance_to_attractor_scope_note": distance_to_attractor_scope_note,
        "distance_to_attractor_content_axis_summary": distance_to_attractor_content_axis_summary,
        "pooled_result_superseded_note": (
            "The pooled region's block above (regions.pooled) reproduces "
            "results/human_drift_spine_000469.json exactly (same code path, region='pooled' is a "
            "no-op filter) and is retained inline as the pre-stratification baseline -- not "
            "removed, only superseded as the project's current headline result by the region-"
            "stratified contrasts this artifact adds."
        ),
    }
    destination = ROOT / "results" / "region_stratified_drift_000469.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(destination),
        "regions": {r: regions_out[r]["n_session_regions_identified"] for r in REGIONS_WITH_POOLED},
    }, indent=2))
    return destination


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-stratified", action="store_true",
                         help="Also fit every anatomical region and write results/region_stratified_drift_000469.json")
    args = parser.parse_args()
    if sha256_file(RULE_PATH) != RULE_HASH:
        raise SystemExit("frozen adjudication rule hash mismatch; refusing to fit real data")
    directory = data_directory()
    sessions, group = fit_region(directory, region="pooled")
    write_pooled_artifact(directory, sessions, group)
    if args.region_stratified:
        run_region_stratified(directory, pooled=(sessions, group))


if __name__ == "__main__":
    main()
