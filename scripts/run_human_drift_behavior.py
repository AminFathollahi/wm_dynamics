#!/usr/bin/env python3
"""Hierarchical error model for out-of-fold probe-time drift in DANDI 000469."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402

INPUT = ROOT / "results" / "human_drift_spine_000469.json"
OUTPUT = ROOT / "results" / "human_drift_behavior_000469.json"
ERROR_CRITERION = 0.25


def main() -> None:
    source = json.loads(INPUT.read_text())
    rows = []
    patient_scale = {}
    for patient, session in source["sessions"].items():
        if session.get("status") != "complete":
            continue
        onset = np.asarray(session["behavior"]["delay_onset_absolute_residual"], dtype=float)
        drift = np.asarray(session["behavior"]["probe_absolute_residual"], dtype=float)
        accuracy = np.asarray(session["behavior"]["accuracy"], dtype=int)
        finite = np.isfinite(drift) & np.isfinite(onset)
        onset = onset[finite]
        drift = drift[finite]
        accuracy = accuracy[finite]
        center = float(np.mean(drift))
        scale = float(np.std(drift, ddof=1))
        if not np.isfinite(scale) or scale < 1e-10:
            continue
        onset_center = float(np.mean(onset))
        onset_scale = float(np.std(onset, ddof=1))
        change = drift - onset
        change_center = float(np.mean(change))
        change_scale = float(np.std(change, ddof=1))
        if onset_scale < 1e-10 or change_scale < 1e-10:
            continue
        patient_scale[patient] = {
            "probe_mean": center, "probe_sd": scale,
            "onset_mean": onset_center, "onset_sd": onset_scale,
            "change_mean": change_center, "change_sd": change_scale,
        }
        for probe_value, onset_value, change_value, correct in zip(
            (drift - center) / scale,
            (onset - onset_center) / onset_scale,
            (change - change_center) / change_scale,
            accuracy,
        ):
            rows.append({
                "patient": patient, "drift_z": probe_value,
                "onset_z": onset_value, "change_z": change_value,
                "error": 1 - int(correct),
            })
    frame = pd.DataFrame(rows)
    n_errors = int(frame["error"].sum())
    n_patients_with_error = int(frame.loc[frame["error"] == 1, "patient"].nunique())
    if n_errors < 20 or n_patients_with_error < 4:
        output = {
            "schema_version": "1.0.0", "status": "underpowered",
            "reason": "fewer than 20 errors or fewer than 4 patients with an error",
            "n_trials": int(len(frame)), "n_errors": n_errors,
            "n_patients": int(frame["patient"].nunique()),
            "n_patients_with_error": n_patients_with_error,
            "source_hash": sha256_file(INPUT), "code_commit": git_commit(ROOT),
        }
        OUTPUT.write_text(canonical_json(output))
        print(json.dumps(output, indent=2))
        return
    model = BinomialBayesMixedGLM.from_formula(
        "error ~ drift_z", {"patient_intercept": "0 + C(patient)"}, frame,
    )
    result = model.fit_vb()
    intercept, slope = map(float, result.fe_mean[:2])
    intercept_sd, slope_sd = map(float, result.fe_sd[:2])
    slope_interval = [slope - 1.96 * slope_sd, slope + 1.96 * slope_sd]
    random_effects = result.random_effects()
    tolerances = {}
    threshold_status = "identified" if slope > 0 and slope_interval[0] > 0 else "not_identified"
    if threshold_status == "identified":
        for patient in sorted(patient_scale):
            label = f"C(patient)[{patient}]"
            random_intercept = float(random_effects.loc[label, "Mean"]) if label in random_effects.index else 0.0
            threshold_z = float((logit(ERROR_CRITERION) - intercept - random_intercept) / slope)
            tolerances[patient] = {
                "threshold_z": threshold_z,
                "threshold_native_absolute_residual": (
                    patient_scale[patient]["probe_mean"]
                    + threshold_z * patient_scale[patient]["probe_sd"]
                ),
            }
    failure_model = BinomialBayesMixedGLM.from_formula(
        "error ~ onset_z + change_z", {"patient_intercept": "0 + C(patient)"}, frame,
    ).fit_vb()
    failure_effects = {}
    for index, name in ((1, "delay_onset"), (2, "maintenance_change")):
        mean = float(failure_model.fe_mean[index])
        sd = float(failure_model.fe_sd[index])
        failure_effects[name] = {
            "log_odds_slope": mean, "sd": sd,
            "interval_95pct": [mean - 1.96 * sd, mean + 1.96 * sd],
            "odds_ratio": float(np.exp(mean)),
        }
    onset_positive = failure_effects["delay_onset"]["interval_95pct"][0] > 0
    maintenance_positive = failure_effects["maintenance_change"]["interval_95pct"][0] > 0
    if onset_positive and not maintenance_positive:
        failure_stage = "encoding_or_delay_entry"
    elif maintenance_positive and not onset_positive:
        failure_stage = "maintenance"
    elif onset_positive and maintenance_positive:
        failure_stage = "both"
    else:
        failure_stage = "not_resolved"
    output = {
        "schema_version": "1.0.0", "analysis_id": "human_drift_behavior_dandi000469",
        "status": "complete", "code_commit": git_commit(ROOT),
        "source_artifact": str(INPUT.relative_to(ROOT)), "source_hash": sha256_file(INPUT),
        "outcome": "memory error", "predictor": "out-of-fold absolute content-axis residual at probe, standardized within patient",
        "model": "hierarchical logistic regression with patient random intercept; variational Bayes posterior",
        "n_trials": int(len(frame)), "n_errors": n_errors,
        "n_patients": int(frame["patient"].nunique()),
        "n_patients_with_error": n_patients_with_error,
        "fixed_effects": {
            "intercept": intercept, "intercept_sd": intercept_sd,
            "drift_log_odds_slope": slope, "drift_slope_sd": slope_sd,
            "drift_slope_95pct_interval": slope_interval,
            "drift_odds_ratio": float(np.exp(slope)),
        },
        "failure_stage_chase": {
            "model": "hierarchical logistic error ~ delay-onset residual + probe-minus-onset residual, patient random intercept",
            "effects": failure_effects,
            "verdict": failure_stage,
        },
        "tolerance_criterion_error_probability": ERROR_CRITERION,
        "tolerance_status": threshold_status,
        "tolerances": tolerances,
        "limitations": [
            "accuracy is near ceiling in most patients",
            "the variational posterior is an approximation and requires sensitivity against a likelihood or MCMC fit",
            "within-patient standardization makes the group slope comparable but patient thresholds depend on fold-varying latent coordinates",
        ],
    }
    OUTPUT.write_text(canonical_json(output))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
