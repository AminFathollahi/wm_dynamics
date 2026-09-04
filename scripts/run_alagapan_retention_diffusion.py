#!/usr/bin/env python3
"""Descriptive retention diffusion after Alagapan encoding stimulation.

Stimulation ends before the analyzed retention window.  The analysis uses
the recording's own sampling metadata and event-defined retention bounds,
then estimates total process diffusion in a baseline-frozen PCA frame for
In Phase, Anti Phase, and Sham trials.  With three patients, all contrasts
remain descriptive case evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import git_commit  # noqa: E402
from run_alagapan_phase_omega import DATA_DIR, PATIENTS, load_behavior  # noqa: E402
from run_alagapan_stimulation_geometry import (  # noqa: E402
    CONDITIONS,
    RETENTION_ONSET_BUFFER_S,
    _baseline_retention_trials,
    _spectral_sanity_check,
    _stimulation_retention_trials,
    STIM_SITES,
)
from run_haslacher_phase_diffusion import _phase_diffusion, bin_analog_trials  # noqa: E402

N_COMPONENTS = 3


def common_average_reference(trials: np.ndarray) -> np.ndarray:
    """Apply the authors' common-average reference over retained contacts."""
    values = np.asarray(trials, dtype=float)
    return values - values.mean(axis=1, keepdims=True)


def condition_contrasts(condition_rows: dict[str, dict]) -> dict:
    """Log-diffusion contrasts relative to sham for both estimators."""
    output = {}
    for estimator in ("state_space_total_diffusion", "legacy_increment_total_diffusion"):
        sham = condition_rows.get("Sham", {}).get(estimator)
        contrasts = {}
        for condition in ("In Phase", "Anti Phase"):
            value = condition_rows.get(condition, {}).get(estimator)
            contrasts[f"{condition}_minus_Sham_log_ratio"] = (
                float(np.log(value / sham))
                if value is not None and sham is not None and value > 0 and sham > 0 else None
            )
        output[estimator] = contrasts
    return output


def analyze_patient(patient: str) -> dict:
    baseline_full, baseline_labels, sampling_rate = _baseline_retention_trials(patient)
    stimulation = _stimulation_retention_trials(patient)
    if stimulation is None:
        return {"status": "excluded", "reason": "trial-order alignment failed", "patient": patient}
    stim_full, stim_labels, trial_conditions = stimulation
    stimulation_contacts = {label for site in STIM_SITES[patient] for label in site}
    common = [label for label in baseline_labels
              if label in set(stim_labels) and label not in stimulation_contacts]
    if len(common) < N_COMPONENTS:
        return {"status": "excluded", "reason": "fewer than three common neural contacts", "patient": patient}
    baseline_index = [baseline_labels.index(label) for label in common]
    stim_index = [stim_labels.index(label) for label in common]
    baseline = common_average_reference(baseline_full[:, baseline_index])
    stim = common_average_reference(stim_full[:, stim_index])

    baseline_binned = bin_analog_trials(baseline, sampling_rate)
    observations = baseline_binned.transpose(0, 2, 1).reshape(-1, len(common))
    center = observations.mean(axis=0)
    scale = observations.std(axis=0)
    scale[scale < 1e-10] = 1.0
    pca = PCA(n_components=min(N_COMPONENTS, len(common)))
    pca.fit((observations - center) / scale)

    condition_rows = {}
    labels = np.asarray(trial_conditions)
    for condition in CONDITIONS:
        selected = stim[labels == condition]
        if len(selected) >= 5:
            condition_rows[condition] = _phase_diffusion(
                selected, center, scale, pca, sampling_rate)
            condition_rows[condition]["spectral_sanity"] = _spectral_sanity_check(
                baseline, selected, sampling_rate)
    ratios = [row["spectral_sanity"]["mean_power_ratio_condition_over_baseline"]
              for row in condition_rows.values()]
    artifact_qc_pass = bool(ratios and all(np.isfinite(ratio) and 0.25 <= ratio <= 4.0
                                           for ratio in ratios))
    return {
        "status": "complete" if artifact_qc_pass else "artifact_qc_failed",
        "patient": patient,
        "sampling_rate_hz": float(sampling_rate),
        "n_common_contacts": len(common),
        "n_baseline_trials": int(len(baseline)),
        "n_stimulation_trials": int(len(stim)),
        "baseline_pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "conditions": condition_rows,
        "diffusion_contrasts": condition_contrasts(condition_rows),
        "behavior": load_behavior(patient),
        "preprocessing_qc": {
            "seizure_contacts_removed_by_source_loader": True,
            "stimulation_contacts_removed": sorted(stimulation_contacts),
            "common_average_reference_applied": True,
            "concurrent_stimulation_ica": "not_required_for_retention_window_but_manual_components_not_released",
            "spectral_power_ratio_rule": "each condition/baseline broadband mean-power ratio within [0.25,4]",
            "artifact_qc_pass": artifact_qc_pass,
        },
    }


def main() -> None:
    if "__WM_DYNAMICS_DATA_ROOT_NOT_SET__" in str(DATA_DIR) or not DATA_DIR.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT; configured Alagapan data directory is unavailable.")
    rows = []
    for patient in PATIENTS:
        print(f"fitting Alagapan retention diffusion {patient}", flush=True)
        rows.append(analyze_patient(patient))
    complete = [row for row in rows if row.get("status") == "complete"]
    output = {
        "analysis": "Alagapan post-encoding-stimulation retention diffusion",
        "git_commit": git_commit(ROOT),
        "evidence": {"n_patients": len(complete),
                     "n_artifact_qc_failed": sum(row.get("status") == "artifact_qc_failed" for row in rows),
                     "inference_level": "descriptive per-patient case evidence"},
        "parameters": {"bin_ms": 100, "n_components": N_COMPONENTS,
                       "retention_onset_buffer_s": RETENTION_ONSET_BUFFER_S,
                       "sampling_rate": "read from each recording",
                       "author_preprocessing": "remove stimulation and seizure contacts, then common-average reference"},
        "per_patient": rows,
        "limitations": [
            "n=3 does not support a population claim",
            "stimulation occurred during encoding, so retention estimates are aftereffects rather than concurrent maintenance inputs",
            "a 250-ms retention-onset buffer does not establish freedom from residual stimulation artifact",
            "session and montage are confounded within patient",
        ],
        "claim_gate": {"G3": "descriptive_only", "G4": "not_tested"},
    }
    path = ROOT / "results" / "alagapan_retention_diffusion.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(path), "evidence": output["evidence"],
                      "contrasts": {row["patient"]: row.get("diffusion_contrasts") for row in complete}}, indent=2))


if __name__ == "__main__":
    main()
