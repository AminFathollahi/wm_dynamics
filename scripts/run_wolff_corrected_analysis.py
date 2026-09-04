#!/usr/bin/env python3
"""Continuous circular Wolff EEG analysis addressing both published readings.

This replaces the legacy six-class/window shortcut with cross-validated
circular Mahalanobis tuning on the doubled orientation angle.  Voltage and
alpha-power measures are analyzed separately.  The pre-ping signal is tested
against a label-matched noise floor, and ping-evoked decay is compared with an
endogenous pre-ping confinement estimate in the same participants.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import (  # noqa: E402
    fit_gaussian_state_space,
    fit_ou_moments,
    leave_one_out_condition_residuals,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402

N_ANGLE_BINS = 12
N_SPLITS = 5
TARGET_HZ = 100.0
SEED = 20260731


def data_directory() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the external data root.")
    path = Path(root) / "Wolff" / "data"
    if not path.is_dir():
        raise SystemExit(f"Wolff data not staged at {path}")
    return path


def valid_mask(epoch, n_trials: int) -> np.ndarray:
    mask = np.ones(n_trials, dtype=bool)
    bad = np.atleast_1d(epoch.bad_trials)
    if bad.size and np.all(np.isfinite(bad)):
        mask[bad.astype(int) - 1] = False
    return mask


def prepare_epoch(epoch, measure: str) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.asarray(epoch.trial, dtype=float)
    time = np.asarray(epoch.time, dtype=float)
    native_hz = float(1.0 / np.median(np.diff(time)))
    data = data - data.mean(axis=1, keepdims=True)
    if measure == "alpha_power":
        sos = butter(4, [8.0, 12.0], btype="bandpass", fs=native_hz, output="sos")
        data = np.log(np.abs(hilbert(sosfiltfilt(sos, data, axis=2), axis=2)) ** 2 + 1e-12)
        data = data - data.mean(axis=1, keepdims=True)
    elif measure != "voltage":
        raise ValueError(f"unknown measure: {measure}")
    stride = max(1, int(round(native_hz / TARGET_HZ)))
    return data[:, :, ::stride], time[::stride], native_hz / stride


def angle_bins(theta: np.ndarray) -> np.ndarray:
    wrapped = np.mod(theta + np.pi, 2.0 * np.pi)
    return np.floor(wrapped / (2.0 * np.pi) * N_ANGLE_BINS + 1e-8).astype(int) % N_ANGLE_BINS


def circular_mahalanobis_scores(
    data: np.ndarray,
    theta: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trial-level circular tuning scores and a matched permuted-label floor."""
    labels = angle_bins(theta)
    counts = np.bincount(labels, minlength=N_ANGLE_BINS)
    n_splits = min(N_SPLITS, int(np.min(counts[counts > 0])))
    if n_splits < 2 or np.any(counts == 0):
        raise ValueError("all circular bins need at least two trials")
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    scores = np.full((len(labels), data.shape[2]), np.nan)
    null_scores = np.full_like(scores, np.nan)
    offsets = np.arange(N_ANGLE_BINS)
    cosine = np.cos(2.0 * np.pi * offsets / N_ANGLE_BINS)
    rng = np.random.default_rng(seed + 1)
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        null_labels = rng.permutation(labels[test])
        for time_index in range(data.shape[2]):
            train_values = data[train, :, time_index]
            test_values = data[test, :, time_index]
            covariance = LedoitWolf().fit(train_values).precision_
            means = np.vstack([
                train_values[labels[train] == label].mean(axis=0)
                for label in range(N_ANGLE_BINS)
            ])
            delta = test_values[:, None, :] - means[None, :, :]
            distances = np.einsum("nki,ij,nkj->nk", delta, covariance, delta, optimize=True)
            class_index = (labels[test, None] - offsets[None, :]) % N_ANGLE_BINS
            null_index = (null_labels[:, None] - offsets[None, :]) % N_ANGLE_BINS
            ordered = np.take_along_axis(distances, class_index, axis=1)
            ordered_null = np.take_along_axis(distances, null_index, axis=1)
            scores[test, time_index] = -np.mean(ordered * cosine[None, :], axis=1)
            null_scores[test, time_index] = -np.mean(ordered_null * cosine[None, :], axis=1)
    return scores, null_scores


def bootstrap_mean_interval(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> list[float]:
    values = values[np.isfinite(values)]
    draws = np.mean(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def fit_impulse_decay(scores: np.ndarray, time: np.ndarray) -> dict:
    baseline = np.mean(scores[:, (time >= -0.08) & (time <= -0.01)], axis=1)
    response = np.mean(scores - baseline[:, None], axis=0)
    fit_mask = (time >= 0.10) & (time <= 0.48)
    fit_time = time[fit_mask] - 0.10
    fit_values = response[fit_mask]

    def model(t, offset, amplitude, rate):
        return offset + amplitude * np.exp(-rate * t)

    try:
        parameters, covariance = curve_fit(
            model, fit_time, fit_values,
            p0=(float(fit_values[-1]), float(fit_values[0] - fit_values[-1]), 4.0),
            bounds=((-np.inf, -np.inf, 1e-6), (np.inf, np.inf, 100.0)),
            maxfev=20000,
        )
        standard_error = float(np.sqrt(max(covariance[2, 2], 0.0)))
        return {
            "status": "complete",
            "lambda_rate": float(parameters[2]),
            "lambda_ci": [float(parameters[2] - 1.96 * standard_error), float(parameters[2] + 1.96 * standard_error)],
            "amplitude": float(parameters[1]),
            "offset": float(parameters[0]),
            "fit_window_seconds": [0.10, 0.48],
        }
    except Exception as exc:
        return {"status": "nonconverged", "reason": str(exc)}


def analyze_measure(data, results: np.ndarray, measure: str, subject_seed: int) -> dict:
    remembered = np.where(results[:, 2] == 1, results[:, 0], results[:, 1])
    theta = 2.0 * remembered
    cue_valid = valid_mask(data.EEG_cue, len(results))
    impulse_valid = valid_mask(data.EEG_impulse, len(results))
    common = cue_valid & impulse_valid
    cue_values, cue_time, cue_hz = prepare_epoch(data.EEG_cue, measure)
    impulse_values, impulse_time, impulse_hz = prepare_epoch(data.EEG_impulse, measure)
    pre_data = cue_values[common]
    post_data = impulse_values[common]
    labels = theta[common]
    pre_scores, pre_null = circular_mahalanobis_scores(pre_data, labels, subject_seed)
    post_scores, post_null = circular_mahalanobis_scores(post_data, labels, subject_seed + 100)

    pre_window = (cue_time >= 0.70) & (cue_time <= 1.09)
    post_window = (impulse_time >= 0.10) & (impulse_time <= 0.30)
    trial_pre = np.mean(pre_scores[:, pre_window], axis=1)
    trial_pre_null = np.mean(pre_null[:, pre_window], axis=1)
    trial_post = np.mean(post_scores[:, post_window], axis=1)
    rng = np.random.default_rng(subject_seed + 200)
    pre_difference = trial_pre - trial_pre_null
    post_difference = trial_post - np.mean(post_null[:, post_window], axis=1)

    endogenous_residual, _ = leave_one_out_condition_residuals(
        pre_scores[:, pre_window], np.zeros(len(pre_scores), dtype=int),
    )
    dt = float(1.0 / cue_hz)
    state = fit_gaussian_state_space(endogenous_residual, dt)
    moment = fit_ou_moments(
        endogenous_residual, dt, n_boot=100,
        rng=np.random.default_rng(subject_seed + 300),
    )
    ping_decay = fit_impulse_decay(post_scores, impulse_time)
    return {
        "n_trials": int(np.sum(common)),
        "sampling_hz": {"pre": cue_hz, "post": impulse_hz},
        "pre_ping_effect": float(np.mean(pre_difference)),
        "pre_ping_trial_bootstrap_ci": bootstrap_mean_interval(pre_difference, rng),
        "post_ping_effect": float(np.mean(post_difference)),
        "post_ping_trial_bootstrap_ci": bootstrap_mean_interval(post_difference, rng),
        "noise_floor_adjudication": (
            "above_noise_floor" if bootstrap_mean_interval(pre_difference, rng)[0] > 0
            else "below_or_unresolved_noise_floor"
        ),
        "endogenous_state_space": state.to_dict(),
        "endogenous_moments": moment.to_dict(),
        "ping_decay": ping_decay,
    }


def analyze_subject(path: Path) -> dict:
    data = loadmat(path, struct_as_record=False, squeeze_me=True)["exp1_data"]
    results = np.asarray(data.Results, dtype=float)
    subject = int(path.stem.rsplit("_", 1)[1])
    output = {"subject": subject, "status": "complete", "measures": {}}
    for measure_index, measure in enumerate(("voltage", "alpha_power")):
        try:
            output["measures"][measure] = analyze_measure(
                data, results, measure, SEED + subject * 1000 + measure_index * 400,
            )
        except Exception as exc:
            output["measures"][measure] = {"status": "failed", "reason": str(exc)}
    return output


def two_ping_linearity(path: Path) -> dict:
    data = loadmat(path, struct_as_record=False, squeeze_me=True)["exp2_data"]
    first, second = [], []
    for session in (1, 2):
        epoch1 = getattr(data, f"EEG_impulse1_sess{session}")
        epoch2 = getattr(data, f"EEG_impulse2_sess{session}")
        n_trials = np.asarray(getattr(data, f"Results_sess{session}")).shape[0]
        common = valid_mask(epoch1, n_trials) & valid_mask(epoch2, n_trials)
        for epoch, target in ((epoch1, first), (epoch2, second)):
            values, time, _hz = prepare_epoch(epoch, "voltage")
            baseline = values[common][:, :, (time >= -0.08) & (time <= -0.01)].mean(axis=2)
            response = values[common][:, :, (time >= 0.10) & (time <= 0.30)].mean(axis=2)
            target.extend(np.linalg.norm(response - baseline, axis=1).tolist())
    first_array = np.asarray(first)
    second_array = np.asarray(second)
    return {
        "n_trial_sessions": int(len(first_array)),
        "second_minus_first_rms": float(np.mean(second_array - first_array)),
        "second_to_first_rms_ratio": float(np.mean(second_array) / max(np.mean(first_array), 1e-12)),
        "superposition_status": "not_identifiable",
        "reason": "the release has sequential first- and second-ping epochs but no isolated second-input or summed-input condition, so equality to a linear sum cannot be tested",
    }


def group_summary(subjects: list[dict], measure: str) -> dict:
    usable = [row["measures"][measure] for row in subjects if row["measures"][measure].get("n_trials")]
    pre = np.array([row["pre_ping_effect"] for row in usable])
    post = np.array([row["post_ping_effect"] for row in usable])
    rng = np.random.default_rng(SEED + (0 if measure == "voltage" else 1))
    state_lambda = np.array([
        row["endogenous_state_space"]["lambda_rate"] for row in usable
        if row["endogenous_state_space"]["status"] == "identifiable"
    ])
    ping_lambda = np.array([
        row["ping_decay"]["lambda_rate"] for row in usable
        if row["ping_decay"].get("status") == "complete"
    ])
    paired = [
        (row["endogenous_state_space"]["lambda_rate"], row["ping_decay"]["lambda_rate"])
        for row in usable
        if row["endogenous_state_space"]["status"] == "identifiable"
        and row["ping_decay"].get("status") == "complete"
    ]
    return {
        "n_participants": int(len(usable)),
        "pre_ping_mean": float(np.mean(pre)),
        "pre_ping_participant_bootstrap_ci": bootstrap_mean_interval(pre, rng),
        "post_ping_mean": float(np.mean(post)),
        "post_ping_participant_bootstrap_ci": bootstrap_mean_interval(post, rng),
        "participants_above_pre_ping_noise_floor": int(sum(row["noise_floor_adjudication"] == "above_noise_floor" for row in usable)),
        "endogenous_lambda_identifiable_participants": int(len(state_lambda)),
        "endogenous_lambda_mean": float(np.mean(state_lambda)) if len(state_lambda) else None,
        "ping_decay_complete_participants": int(len(ping_lambda)),
        "ping_lambda_mean": float(np.mean(ping_lambda)) if len(ping_lambda) else None,
        "paired_lambda_participants": int(len(paired)),
        "ping_minus_endogenous_lambda_mean": float(np.mean([b - a for a, b in paired])) if paired else None,
        "lambda_agreement_status": "estimable" if len(paired) >= 10 else "not_estimable_fewer_than_10_paired_identifiable_participants",
    }


def main() -> None:
    directory = data_directory()
    exp1_files = sorted(directory.glob("Dynamic_hidden_states_exp1_*.mat"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    exp2_files = sorted(directory.glob("Dynamic_hidden_states_exp2_*.mat"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    subjects = []
    for path in exp1_files:
        print(f"fitting Wolff experiment 1 subject {path.stem.rsplit('_', 1)[1]}", flush=True)
        subjects.append(analyze_subject(path))
    two_ping = []
    for path in exp2_files:
        print(f"checking Wolff experiment 2 subject {path.stem.rsplit('_', 1)[1]}", flush=True)
        two_ping.append({"subject": int(path.stem.rsplit("_", 1)[1]), **two_ping_linearity(path)})
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "wolff_continuous_circular_impulse",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "dataset": "Wolff et al. 2017 experiments 1 and 2",
        "method": "continuous doubled-orientation 12-bin circular Mahalanobis tuning; five-fold participant-local CV; voltage and 8-12 Hz alpha power",
        "participants": subjects,
        "group": {
            "voltage": group_summary(subjects, "voltage"),
            "alpha_power": group_summary(subjects, "alpha_power"),
        },
        "two_ping_linearity": two_ping,
        "published_reading_adjudication": "decided from pre-ping participant-level voltage and alpha effects relative to matched permuted-label floors; neither reading is imposed in advance",
        "limitations": [
            "scalp EEG cannot localize or be equated quantitatively with intracranial single-unit coordinates",
            "five-fold covariance estimation replaces the source leave-one-trial-out implementation for tractable participant-level uncertainty",
            "alpha power is derived from the released preprocessed voltage and may retain preprocessing dependencies",
        ],
    }
    destination = ROOT / "results" / "wolff_corrected_impulse.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"group": output["group"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
