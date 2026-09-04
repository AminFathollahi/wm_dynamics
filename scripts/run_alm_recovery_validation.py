#!/usr/bin/env python3
"""Validate the confinement estimator against ALM optogenetic recovery.

Only unstimulated delayed-response trials are used to fit endogenous
confinement.  Bilateral early-delay silencing trials are then used once, as a
known large perturbation, to measure recovery of the population choice code.
This validates method sensitivity in mouse motor preparation; it is not a
working-memory replication.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import fit_gaussian_state_space, fit_ou_moments, leave_one_out_condition_residuals  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402

BIN_MS = 100
WINDOW_S = 2.0
MIN_UNITS = 15
MIN_TRIALS_PER_ARM_CONDITION = 8
SEED = 20260731


def data_directory() -> Path:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = Path(root) / config["datasets"]["inagaki_alm5"]["local_path"] / "RandomDelayTask" / "withPerturbation"
    if not path.is_dir():
        raise SystemExit(f"ALM perturbation sessions not staged at {path}")
    return path


def trial_condition(trial_types: np.ndarray) -> np.ndarray:
    return np.array([0 if str(value).lower().startswith("l") else 1 for value in trial_types], dtype=int)


def build_counts(units: np.ndarray, delay_start: np.ndarray, trial_indices: np.ndarray) -> np.ndarray:
    starts = np.arange(0.0, WINDOW_S, BIN_MS / 1000.0)
    counts = np.zeros((len(trial_indices), len(units), len(starts)), dtype=float)
    row_for_trial = {int(trial): row for row, trial in enumerate(trial_indices)}
    for unit_index, unit in enumerate(units):
        spike_times = np.asarray(unit.SpikeTimes, dtype=float).reshape(-1)
        spike_trials = np.asarray(unit.Trial_idx_of_spike, dtype=int).reshape(-1) - 1
        for trial in trial_indices:
            row = row_for_trial[int(trial)]
            relative = spike_times[spike_trials == trial] - delay_start[trial]
            counts[row, unit_index], _ = np.histogram(relative, bins=np.append(starts, WINDOW_S))
    return counts


def fit_recovery_rate(time: np.ndarray, deficit: np.ndarray) -> dict:
    """Fit decay of the post-silencing coding deficit after the 1-s offset."""
    mask = (time >= 1.0) & (time <= 1.9) & np.isfinite(deficit)
    x = time[mask] - 1.0
    y = deficit[mask]

    def model(value, offset, amplitude, rate):
        return offset + amplitude * np.exp(-rate * value)

    if len(x) < 5:
        return {"status": "not_estimable", "reason": "fewer than five post-offset bins"}
    try:
        parameters, covariance = curve_fit(
            model, x, y,
            p0=(float(y[-1]), float(y[0] - y[-1]), 3.0),
            bounds=((-np.inf, -np.inf, 1e-6), (np.inf, np.inf, 100.0)),
            maxfev=20000,
        )
        se = float(np.sqrt(max(covariance[2, 2], 0.0)))
        return {
            "status": "complete",
            "lambda_rate": float(parameters[2]),
            "lambda_ci": [float(parameters[2] - 1.96 * se), float(parameters[2] + 1.96 * se)],
            "amplitude": float(parameters[1]),
            "offset": float(parameters[0]),
        }
    except Exception as exc:
        return {"status": "nonconverged", "reason": str(exc)}


def analyze_session(path: Path) -> dict:
    units = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["unit"])
    behavior = units[0].Behavior
    trial_info = units[0].Trial_info
    trial_type = np.asarray(behavior.Trial_types_of_response_vector, dtype=int).reshape(-1)
    stimulation = np.asarray(behavior.stim_trial_vector, dtype=int).reshape(-1)
    delay_duration = np.asarray(behavior.delay_dur, dtype=float).reshape(-1)
    delay_start = np.asarray(behavior.Delay_start, dtype=float).reshape(-1)
    condition = trial_condition(np.asarray(trial_info.Trial_types).reshape(-1))
    start, stop = np.asarray(trial_info.Trial_range_to_analyze, dtype=int).reshape(-1) - 1
    eligible = np.arange(start, stop + 1)
    eligible = eligible[(trial_type[eligible] < 5) & (delay_duration[eligible] >= WINDOW_S)]
    control_trials = eligible[stimulation[eligible] == 0]
    perturb_trials = eligible[stimulation[eligible] > 1]
    base = {
        "mouse": path.stem.split("_")[0],
        "n_units_released": int(len(units)),
        "n_control_trials": int(len(control_trials)),
        "n_perturb_trials": int(len(perturb_trials)),
    }
    arm_counts = {
        "control": [int(np.sum(condition[control_trials] == value)) for value in (0, 1)],
        "perturb": [int(np.sum(condition[perturb_trials] == value)) for value in (0, 1)],
    }
    if min(*arm_counts["control"], *arm_counts["perturb"], len(units)) < MIN_TRIALS_PER_ARM_CONDITION:
        return {**base, "status": "excluded", "reason": "insufficient trials or units", "arm_condition_counts": arm_counts}
    all_trials = np.concatenate((control_trials, perturb_trials))
    counts = build_counts(units, delay_start, all_trials)
    rates = counts[:len(control_trials)].sum(axis=(0, 2)) / (len(control_trials) * WINDOW_S)
    unit_mask = rates >= 0.2
    counts = counts[:, unit_mask]
    base["n_units_after_rate_qc"] = int(np.sum(unit_mask))
    if np.sum(unit_mask) < MIN_UNITS:
        return {**base, "status": "excluded", "reason": "fewer than 15 units after rate QC", "arm_condition_counts": arm_counts}

    transform = FrozenPSTHTransform().fit(counts[:len(control_trials)])
    standardized = transform.transform(counts).transpose(0, 2, 1)
    pca = PCA(n_components=min(8, standardized.shape[2], len(control_trials) - 1), svd_solver="full")
    pca.fit(standardized[:len(control_trials)].reshape(-1, standardized.shape[2]))
    state = pca.transform(standardized.reshape(-1, standardized.shape[2])).reshape(len(all_trials), counts.shape[2], -1)
    control_state = state[:len(control_trials)]
    perturb_state = state[len(control_trials):]
    control_labels = condition[control_trials]
    perturb_labels = condition[perturb_trials]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(control_state[:, :5].mean(axis=1), control_labels)
    direction = np.asarray(lda.coef_[0], dtype=float)
    direction /= max(np.linalg.norm(direction), 1e-12)
    control_projection = control_state @ direction
    perturb_projection = perturb_state @ direction
    residual, _ = leave_one_out_condition_residuals(control_projection, control_labels)
    state_estimate = fit_gaussian_state_space(residual, BIN_MS / 1000.0)
    moment_estimate = fit_ou_moments(
        residual, BIN_MS / 1000.0, n_boot=100,
        rng=np.random.default_rng(SEED + sum(map(ord, path.stem))),
    )
    separation_control = control_projection[control_labels == 1].mean(axis=0) - control_projection[control_labels == 0].mean(axis=0)
    separation_perturb = perturb_projection[perturb_labels == 1].mean(axis=0) - perturb_projection[perturb_labels == 0].mean(axis=0)
    sign = 1.0 if np.mean(separation_control[:5]) >= 0 else -1.0
    coding_deficit = sign * (separation_control - separation_perturb)
    time = np.arange(counts.shape[2]) * BIN_MS / 1000.0 + BIN_MS / 2000.0
    recovery = fit_recovery_rate(time, coding_deficit)
    if state_estimate.lambda_ci and recovery.get("lambda_ci"):
        overlap = max(state_estimate.lambda_ci[0], recovery["lambda_ci"][0]) <= min(state_estimate.lambda_ci[1], recovery["lambda_ci"][1])
    else:
        overlap = None
    return {
        **base,
        "status": "complete",
        "arm_condition_counts": arm_counts,
        "endogenous_state_space": state_estimate.to_dict(),
        "endogenous_moments": moment_estimate.to_dict(),
        "observed_recovery": recovery,
        "lambda_interval_overlap": overlap,
        "coding_deficit": coding_deficit.tolist(),
        "time_seconds": time.tolist(),
    }


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draws = np.mean(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def main() -> None:
    directory = data_directory()
    sessions = {}
    for path in sorted(directory.glob("*_units.mat")):
        print(f"fitting ALM {path.stem}", flush=True)
        try:
            sessions[path.stem] = analyze_session(path)
        except Exception as exc:
            sessions[path.stem] = {"status": "failed", "reason": str(exc), "mouse": path.stem.split("_")[0]}
    complete = [row for row in sessions.values() if row["status"] == "complete"]
    paired = [
        row for row in complete
        if row["endogenous_state_space"]["status"] == "identifiable"
        and row["observed_recovery"].get("status") == "complete"
    ]
    differences = np.array([
        row["observed_recovery"]["lambda_rate"] - row["endogenous_state_space"]["lambda_rate"]
        for row in paired
    ])
    rng = np.random.default_rng(SEED)
    group = {
        "n_files": int(len(sessions)),
        "n_complete_sessions": int(len(complete)),
        "n_paired_identifiable_sessions": int(len(paired)),
        "n_mice_complete": int(len({row["mouse"] for row in complete})),
        "ping_minus_endogenous_lambda_mean": float(np.mean(differences)) if len(differences) else None,
        "ping_minus_endogenous_lambda_ci": bootstrap_interval(differences, rng) if len(differences) >= 2 else None,
        "interval_overlap_sessions": int(sum(row["lambda_interval_overlap"] is True for row in paired)),
        "verdict": (
            "recovery_rate_agrees_with_endogenous_confinement" if len(paired) >= 3 and bootstrap_interval(differences, rng)[0] <= 0 <= bootstrap_interval(differences, rng)[1]
            else "agreement_not_established"
        ),
    }
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "alm_unperturbed_confinement_predicts_opto_recovery",
        "dataset": "Inagaki ALM silicon-probe perturbation release, Figshare 7489253",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "metadata_decision": {
            "construct": "recovery of ALM choice code after bilateral early-delay silencing",
            "independent_unit": "session nested in mouse",
            "simultaneous_population": True,
            "intervention": True,
            "go_no_go": "go",
        },
        "sessions": sessions,
        "group": group,
        "claim_limit": "mouse motor preparation method validation only; not a human working-memory replication",
    }
    destination = ROOT / "results" / "alm_recovery_validation.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"group": group, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
