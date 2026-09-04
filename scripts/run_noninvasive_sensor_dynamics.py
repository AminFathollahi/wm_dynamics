#!/usr/bin/env python3
"""Participant-level confinement rate achievable from our own scalp EEG.

Re-reads the already-staged Wolff (impulse, delay task) and
Haslacher (visual WM, tACS) EEG lambda fits at the sensor level -- no new
download, no new fit -- and reports lambda with participant intervals. This is
the non-invasive power calibration: what confinement rate can scalp EEG
resolve at all, before any claim is made about recovering an intracranial
structure with it.

Both datasets already carry a per-participant confined-drift fit
(``fit_gaussian_state_space`` in src/drift_dynamics.py) computed directly on
sensor-level voltage/PCA components by scripts/run_wolff_corrected_analysis.py
and scripts/run_haslacher_phase_diffusion.py. This script only aggregates
those existing fits with a participant-level bootstrap interval; it does not
refit anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402

SEED = 20260803


def _wolff_lambda(participants: list[dict[str, Any]], measure: str) -> np.ndarray:
    values = []
    for row in participants:
        entry = row.get("measures", {}).get(measure, {})
        state = entry.get("endogenous_state_space", {})
        if state.get("status") == "identifiable" and state.get("lambda_rate") is not None:
            values.append(float(state["lambda_rate"]))
    return np.asarray(values, dtype=float)


def _haslacher_lambda(participants: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in participants:
        if row.get("status") != "complete":
            continue
        component_lambda = [
            component["lambda_rate"]
            for phase in row.get("phase_diffusion", {}).values()
            for component in phase.get("state_space_components", [])
            if component.get("status") == "identifiable" and component.get("lambda_rate") is not None
        ]
        if component_lambda:
            values.append(float(np.mean(component_lambda)))
    return np.asarray(values, dtype=float)


def _summarize(name: str, lambda_values: np.ndarray, n_total_participants: int) -> dict[str, Any]:
    if len(lambda_values) < 5:
        return {
            "name": name,
            "status": "non_identified",
            "reason": "fewer than five participants with an identifiable sensor-level confinement estimate",
            "n_identifiable_participants": int(len(lambda_values)),
            "n_total_participants": int(n_total_participants),
        }
    rng = np.random.default_rng(SEED)
    mean, lower, upper = bootstrap_ci(lambda_values, np.mean, rng=rng)
    return {
        "name": name,
        "status": "estimable",
        "n_identifiable_participants": int(len(lambda_values)),
        "n_total_participants": int(n_total_participants),
        "lambda_rate_per_second_mean": mean,
        "lambda_rate_per_second_participant_bootstrap_ci": [lower, upper],
        "implied_time_constant_seconds_mean": float(1.0 / mean) if mean > 0 else None,
    }


def main() -> None:
    wolff_path = ROOT / "results" / "wolff_corrected_impulse.json"
    haslacher_path = ROOT / "results" / "haslacher_phase_diffusion.json"
    if not wolff_path.is_file() or not haslacher_path.is_file():
        raise SystemExit(
            "Run scripts/run_wolff_corrected_analysis.py and "
            "scripts/run_haslacher_phase_diffusion.py first; this script only aggregates their output."
        )
    wolff = json.loads(wolff_path.read_text())
    haslacher = json.loads(haslacher_path.read_text())

    wolff_participants = wolff["participants"]
    haslacher_participants = haslacher["per_participant"]

    datasets = {
        "wolff_2017_impulse_voltage": _summarize(
            "Wolff et al. 2017, impulse task, scalp voltage",
            _wolff_lambda(wolff_participants, "voltage"),
            len(wolff_participants),
        ),
        "wolff_2017_impulse_alpha_power": _summarize(
            "Wolff et al. 2017, impulse task, 8-12 Hz alpha power",
            _wolff_lambda(wolff_participants, "alpha_power"),
            len(wolff_participants),
        ),
        "haslacher_clam_tacs_pca": _summarize(
            "Haslacher CLAM-tACS, visual WM, sensor-level PCA components",
            _haslacher_lambda(haslacher_participants),
            len(haslacher_participants),
        ),
    }

    output = {
        "schema_version": "1.0.0",
        "analysis_id": "noninvasive_sensor_dynamics",
        "trigger": "non-invasive power calibration for the confinement-rate estimator",
        "code_commit": git_commit(ROOT),
        "method": (
            "No new fit. Aggregates the participant-level confined-drift lambda "
            "already estimated directly on sensor-level EEG (Wolff: 12-bin circular "
            "Mahalanobis tuning score, cue-to-impulse endogenous window; Haslacher: "
            "leading 3 PCA components of the stimulation-off-frozen sensor coordinate "
            "frame) by the fitted scalar Gaussian state-space model in "
            "src/drift_dynamics.py. Participant is the resampling unit."
        ),
        "datasets": datasets,
        "honest_limit": (
            "No scalp measurement resolves hippocampus or amygdala. "
            "This is a whole-head sensor mixture with no anatomical resolution at all -- "
            "not even the cortex-only resolution a beamformed virtual sensor would carry. "
            "A lambda recovered here answers only 'is a confinement rate estimable from scalp EEG "
            "at all', not 'which structure does it come from'."
        ),
        "interpretation": (
            "This is the achievable non-invasive effect size, not a validation of any "
            "intracranial region estimate. See results/switching_adjudication.json and "
            "results/rotation_estimator_floor.json region_stratified blocks, and "
            "results/region_stratified_drift_000469.json, for the intracranial reference "
            "this does not yet calibrate against -- that calibration is 2C.2/2C.3 "
            "(results/cross_modality_calibration.json, pending ds004752)."
        ),
    }
    destination = ROOT / "results" / "noninvasive_sensor_dynamics.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"output": str(destination), "datasets": datasets}, indent=2))


if __name__ == "__main__":
    main()
