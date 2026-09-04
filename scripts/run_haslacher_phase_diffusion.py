#!/usr/bin/env python3
"""Participant-level phase modulation of diffusion in CLAM-tACS WM EEG.

The coordinate system is fitted on each participant's stimulation-off
baseline without behavioral outcomes.  Active stimulation trials are then
projected into that frozen frame, phase-condition trial means are removed
leave-one-out, and total process diffusion is estimated across the leading
three PCs with a scalar LGSSM per component.  A transparent increment
estimator is retained beside the state-space estimate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drift_dynamics import fit_gaussian_state_space, leave_one_out_condition_residuals  # noqa: E402
from provenance import git_commit  # noqa: E402
from statistics import stable_seed  # noqa: E402
from run_haslacher_phase_omega import (  # noqa: E402
    ACTIVE_SUBJECTS,
    CONTROL_SUBJECTS,
    DATA_DIR,
    PHASE_CONDITIONS,
    trial_outcomes,
)
from run_haslacher_stimulation_geometry import (  # noqa: E402
    _preprocess_author_native,
    _retention_trials,
)

BIN_MS = 100
N_COMPONENTS = 3
N_PERM = 5000


def bin_analog_trials(trials: np.ndarray, sampling_rate: float, bin_ms: int = BIN_MS) -> np.ndarray:
    """Average analog samples into nonoverlapping bins without smoothing."""
    values = np.asarray(trials, dtype=float)
    samples = int(round(sampling_rate * bin_ms / 1000.0))
    n_bins = values.shape[-1] // samples
    if values.ndim != 3 or samples < 1 or n_bins < 4:
        raise ValueError("trials must be (trial, channel, time) with at least four bins")
    return values[..., :n_bins * samples].reshape(values.shape[0], values.shape[1], n_bins, samples).mean(-1)


def harmonic_coefficients(values: dict[int, float]) -> dict | None:
    """Fit one circular harmonic to six phase-condition values."""
    ordered = sorted(PHASE_CONDITIONS, key=lambda code: PHASE_CONDITIONS[code])
    if any(code not in values or not np.isfinite(values[code]) for code in ordered):
        return None
    phase = np.deg2rad([PHASE_CONDITIONS[code] for code in ordered])
    y = np.array([values[code] for code in ordered], dtype=float)
    design = np.column_stack((np.ones(len(y)), np.cos(phase), np.sin(phase)))
    intercept, cosine, sine = np.linalg.lstsq(design, y, rcond=None)[0]
    return {"intercept": float(intercept), "cosine": float(cosine), "sine": float(sine),
            "amplitude": float(np.hypot(cosine, sine)),
            "optimal_phase_deg": float(np.degrees(np.arctan2(sine, cosine)) % 360.0)}


def _phase_diffusion(trials: np.ndarray, center: np.ndarray, scale: np.ndarray,
                     pca: PCA, sampling_rate: float) -> dict:
    binned = bin_analog_trials(trials, sampling_rate)
    latent = pca.transform(((binned.transpose(0, 2, 1) - center) / scale).reshape(-1, len(center)))
    latent = latent.reshape(len(binned), binned.shape[-1], -1)
    residuals, _ = leave_one_out_condition_residuals(latent, np.zeros(len(latent), dtype=int))
    state_rows = []
    increment_diffusion = []
    dt = BIN_MS / 1000.0
    for component in range(latent.shape[-1]):
        values = residuals[..., component]
        estimate = fit_gaussian_state_space(values, dt)
        state_rows.append(estimate.to_dict())
        increment_diffusion.append(float(np.nanmean(np.diff(values, axis=1) ** 2) / (2.0 * dt)))
    process = [row["diffusion"] for row in state_rows if row.get("diffusion") is not None
               and np.isfinite(row["diffusion"])]
    return {
        "state_space_total_diffusion": float(np.sum(process)) if process else None,
        "legacy_increment_total_diffusion": float(np.sum(increment_diffusion)),
        "state_space_components": state_rows,
        "n_trials": int(len(trials)),
        "n_components_estimable": len(process),
        "n_lambda_precision_identified": sum(
            row.get("lambda_ci") is not None and row["lambda_ci"][0] > 0 for row in state_rows
        ),
    }


def _behavior_harmonic(subject: str) -> dict | None:
    outcomes = trial_outcomes(subject)
    log_odds = {}
    counts = {}
    for code in PHASE_CONDITIONS:
        values = [correct for trial_code, correct in outcomes if trial_code == code]
        if not values:
            continue
        successes = sum(values)
        failures = len(values) - successes
        log_odds[code] = float(np.log((successes + 0.5) / (failures + 0.5)))
        counts[code] = {"correct": successes, "error": failures}
    harmonic = harmonic_coefficients(log_odds)
    return None if harmonic is None else {"harmonic_log_odds": harmonic, "counts": counts}


def analyze_subject(subject: str, group: str) -> dict:
    try:
        no_stim, stim, preprocessing = _preprocess_author_native(subject, group)
        baseline = _retention_trials(no_stim)[0]
        by_phase = _retention_trials(stim, codes=list(PHASE_CONDITIONS))
    except (FileNotFoundError, OSError, ValueError) as error:
        return {"status": "excluded", "reason": str(error), "subject": subject, "group": group}
    behavior = _behavior_harmonic(subject)
    phase_counts = {str(code): int(len(by_phase.get(code, []))) for code in PHASE_CONDITIONS}
    if not preprocessing["sass_sanity_pass"]:
        return {"status": "artifact_qc_failed", "subject": subject, "group": group,
                "preprocessing_qc": preprocessing, "phase_trial_counts": phase_counts,
                "behavior": behavior,
                "reason": "post-SASS target alpha spectrum did not pass the predeclared baseline-similarity rule"}
    if any(count < 5 for count in phase_counts.values()):
        return {"status": "excluded", "subject": subject, "group": group,
                "preprocessing_qc": preprocessing, "phase_trial_counts": phase_counts,
                "behavior": behavior, "reason": "at least one phase condition has fewer than five trials"}
    sampling_rate = float(no_stim.info["sfreq"])
    baseline_binned = bin_analog_trials(baseline, sampling_rate)
    observations = baseline_binned.transpose(0, 2, 1).reshape(-1, baseline_binned.shape[1])
    center = observations.mean(axis=0)
    scale = observations.std(axis=0)
    scale[scale < 1e-10] = 1.0
    pca = PCA(n_components=min(N_COMPONENTS, observations.shape[1]))
    pca.fit((observations - center) / scale)

    phase_rows = {}
    for code in sorted(PHASE_CONDITIONS):
        if len(by_phase.get(code, [])) < 5:
            continue
        phase_rows[code] = _phase_diffusion(by_phase[code], center, scale, pca, sampling_rate)
    state_values = {code: np.log(row["state_space_total_diffusion"])
                    for code, row in phase_rows.items()
                    if row["state_space_total_diffusion"] is not None
                    and row["state_space_total_diffusion"] > 0}
    legacy_values = {code: np.log(row["legacy_increment_total_diffusion"])
                     for code, row in phase_rows.items() if row["legacy_increment_total_diffusion"] > 0}
    return {
        "status": "complete",
        "subject": subject,
        "group": group,
        "preprocessing_qc": preprocessing,
        "phase_trial_counts": phase_counts,
        "n_eeg_channels": int(baseline.shape[1]),
        "n_baseline_trials": int(len(baseline)),
        "baseline_pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "phase_diffusion": {str(code): row for code, row in phase_rows.items()},
        "state_space_log_diffusion_harmonic": harmonic_coefficients(state_values),
        "legacy_log_diffusion_harmonic": harmonic_coefficients(legacy_values),
        "behavior": behavior,
    }


def group_vector_test(rows: list[dict], key_path: tuple[str, ...], seed: str,
                      n_perm: int = N_PERM) -> dict | None:
    """Population circular-vector test with participant-level phase rotations."""
    vectors = []
    for row in rows:
        value = row
        for key in key_path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            vectors.append([value["cosine"], value["sine"]])
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) < 3:
        return None
    observed = vectors.mean(axis=0)
    rng = np.random.default_rng(stable_seed(seed))
    null = np.empty(n_perm)
    angles = np.arange(6) * np.pi / 3.0
    for index in range(n_perm):
        rotations = rng.choice(angles, size=len(vectors))
        cosine, sine = np.cos(rotations), np.sin(rotations)
        rotated = np.column_stack((vectors[:, 0] * cosine - vectors[:, 1] * sine,
                                   vectors[:, 0] * sine + vectors[:, 1] * cosine))
        null[index] = np.linalg.norm(rotated.mean(axis=0))
    bootstrap = np.array([vectors[rng.integers(0, len(vectors), len(vectors))].mean(axis=0)
                          for _ in range(2000)])
    magnitude = float(np.linalg.norm(observed))
    return {"mean_cosine": float(observed[0]), "mean_sine": float(observed[1]),
            "population_amplitude": magnitude,
            "optimal_phase_deg": float(np.degrees(np.arctan2(observed[1], observed[0])) % 360.0),
            "participant_bootstrap_cosine_ci": np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
            "participant_bootstrap_sine_ci": np.quantile(bootstrap[:, 1], [0.025, 0.975]).tolist(),
            "circular_rotation_p_value": float((1 + np.sum(null >= magnitude)) / (n_perm + 1)),
            "n_participants": int(len(vectors))}


def active_control_difference(rows: list[dict], key: str, n_perm: int = N_PERM) -> dict | None:
    groups = {}
    for group in ("active", "control"):
        vectors = []
        for row in rows:
            harmonic = row.get(key)
            if row.get("group") == group and harmonic is not None:
                vectors.append([harmonic["cosine"], harmonic["sine"]])
        groups[group] = np.asarray(vectors, dtype=float)
    if min(len(groups["active"]), len(groups["control"])) < 3:
        return None
    observed = groups["active"].mean(0) - groups["control"].mean(0)
    pooled = np.vstack((groups["active"], groups["control"]))
    rng = np.random.default_rng(stable_seed(f"haslacher_active_control_{key}"))
    null = np.empty(n_perm)
    n_active = len(groups["active"])
    for index in range(n_perm):
        permuted = pooled[rng.permutation(len(pooled))]
        null[index] = np.linalg.norm(permuted[:n_active].mean(0) - permuted[n_active:].mean(0))
    magnitude = float(np.linalg.norm(observed))
    return {"difference_cosine": float(observed[0]), "difference_sine": float(observed[1]),
            "difference_amplitude": magnitude,
            "participant_label_permutation_p_value": float((1 + np.sum(null >= magnitude)) / (n_perm + 1)),
            "n_active": len(groups["active"]), "n_control": len(groups["control"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", type=int, default=None, help="First N active participants")
    parser.add_argument("--control", type=int, default=None, help="First N control participants")
    parser.add_argument("--permutations", type=int, default=N_PERM)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "haslacher_phase_diffusion.json")
    args = parser.parse_args()
    if "__WM_DYNAMICS_DATA_ROOT_NOT_SET__" in str(DATA_DIR) or not DATA_DIR.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT; configured Haslacher data directory is unavailable.")
    active = ACTIVE_SUBJECTS[:args.active] if args.active is not None else ACTIVE_SUBJECTS
    control = CONTROL_SUBJECTS[:args.control] if args.control is not None else CONTROL_SUBJECTS
    rows = []
    for group, subjects in (("active", active), ("control", control)):
        for subject in subjects:
            print(f"fitting Haslacher phase diffusion {group}:{subject}", flush=True)
            rows.append(analyze_subject(subject, group))
    complete = [row for row in rows if row.get("status") == "complete"]
    groups = {}
    for group in ("active", "control"):
        selected = [row for row in complete if row["group"] == group]
        groups[group] = {
            "state_space_diffusion": group_vector_test(
                selected, ("state_space_log_diffusion_harmonic",),
                f"haslacher_state_space_{group}", args.permutations),
            "legacy_increment_diffusion": group_vector_test(
                selected, ("legacy_log_diffusion_harmonic",),
                f"haslacher_legacy_{group}", args.permutations),
            "behavior_log_odds": group_vector_test(
                selected, ("behavior", "harmonic_log_odds"),
                f"haslacher_behavior_{group}", args.permutations),
        }
    output = {
        "analysis": "Haslacher CLAM-tACS participant-level phase modulation of diffusion",
        "git_commit": git_commit(ROOT),
        "parameters": {"bin_ms": BIN_MS, "n_components": N_COMPONENTS,
                       "permutations": args.permutations, "preprocessing":
                       "README-ordered pyprep noisy-channel and saturation rejection, 8-14 Hz filter, SASS, post-SASS average reference; auxiliary channels excluded; baseline-frozen channel z-score/PCA"},
        "evidence": {"n_participants_complete": len(complete),
                     "n_active": sum(row["group"] == "active" for row in complete),
                     "n_control": sum(row["group"] == "control" for row in complete),
                     "n_artifact_qc_failed": sum(row.get("status") == "artifact_qc_failed" for row in rows)},
        "per_participant": rows,
        "population": groups,
        "active_control_state_space_difference": active_control_difference(
            complete, "state_space_log_diffusion_harmonic", args.permutations),
        "active_control_legacy_difference": active_control_difference(
            complete, "legacy_log_diffusion_harmonic", args.permutations),
        "claim_gate": {"G3": "candidate_only_pending_artifact_sensitivity",
                       "reason": "Concurrent scalp tACS is vulnerable to residual phase-locked artifact even after SASS; active-vs-control and auxiliary-channel exclusions are required negative controls."},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(args.output), "evidence": output["evidence"],
                      "population": output["population"],
                      "active_control_state_space_difference": output["active_control_state_space_difference"]}, indent=2))


if __name__ == "__main__":
    main()
