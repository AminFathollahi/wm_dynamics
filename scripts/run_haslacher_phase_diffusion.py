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
from run_haslacher_phase_omega import DATA_DIR, trial_outcomes
from preprocessing import ACTIVE_SUBJECTS, CONTROL_SUBJECTS
from preprocessing import PHASE_CONDITIONS
from run_haslacher_stimulation_geometry import _preprocess_author_native
from preprocessing import _retention_trials
from preprocessing import _phase_diffusion, active_control_difference, bin_analog_trials, group_vector_test, BIN_MS, N_PERM  # noqa: E402
from preprocessing import harmonic_coefficients  # noqa: E402

N_COMPONENTS = 3








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






def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", type=int, default=None, help="First N active participants")
    parser.add_argument("--control", type=int, default=None, help="First N control participants")
    parser.add_argument("--permutations", type=int, default=N_PERM)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "haslacher_phase_diffusion.json")
    args = parser.parse_args()
    if "__WM_DYNAMICS_DATA_ROOT_NOT_SET__" in str(DATA_DIR) or not DATA_DIR.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT; configured closed-loop transcranial alternating-current stimulation scalp-EEG corpus (doi:10.1016/j.brs.2024.07.007) data directory is unavailable.")
    active = ACTIVE_SUBJECTS[:args.active] if args.active is not None else ACTIVE_SUBJECTS
    control = CONTROL_SUBJECTS[:args.control] if args.control is not None else CONTROL_SUBJECTS
    rows = []
    for group, subjects in (("active", active), ("control", control)):
        for subject in subjects:
            print(f"fitting closed-loop tACS scalp-EEG corpus phase diffusion {group}:{subject}", flush=True)
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
        "analysis": "Closed-loop tACS scalp-EEG participant-level phase modulation of diffusion",
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
