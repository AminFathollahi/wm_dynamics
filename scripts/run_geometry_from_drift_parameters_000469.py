"""Test whether fitted DANDI 000469 drift parameters predict measured geometry."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402


def calibration(rows: list[dict[str, float]], rng: np.random.Generator) -> dict[str, Any]:
    if len(rows) < 4:
        return {
            "status": "not_estimable",
            "n_patients": len(rows),
            "reason": "fewer than four patients have paired predicted and observed values",
        }
    predicted = np.asarray([row["predicted"] for row in rows], dtype=float)
    observed = np.asarray([row["observed"] for row in rows], dtype=float)

    def statistics(indices: np.ndarray) -> tuple[float, float]:
        x, y = predicted[indices], observed[indices]
        correlation = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        design = np.column_stack([np.ones(len(x)), x])
        slope = float(np.linalg.lstsq(design, y, rcond=None)[0][1])
        return correlation, slope

    correlation, slope = statistics(np.arange(len(rows)))
    bootstrap = np.asarray([
        statistics(rng.integers(0, len(rows), size=len(rows)))
        for _ in range(5000)
    ])
    finite_correlation = bootstrap[np.isfinite(bootstrap[:, 0]), 0]
    finite_slope = bootstrap[np.isfinite(bootstrap[:, 1]), 1]
    return {
        "status": "estimable",
        "n_patients": len(rows),
        "pearson_correlation": correlation,
        "pearson_correlation_patient_bootstrap_interval_95": (
            list(map(float, np.percentile(finite_correlation, [2.5, 97.5])))
            if len(finite_correlation) else None
        ),
        "calibration_slope": slope,
        "calibration_slope_patient_bootstrap_interval_95": list(map(
            float, np.percentile(finite_slope, [2.5, 97.5])
        )),
        "identity_slope": 1.0,
    }


def main() -> None:
    hierarchy = json.loads((ROOT / "results" / "hierarchical_confinement_000469.json").read_text())
    drift = json.loads((ROOT / "results" / "human_drift_spine_000469.json").read_text())
    geometry = json.loads((ROOT / "results" / "dandi000469_summary.json").read_text())
    crossnobis_path = ROOT / "results" / "crossnobis_content_000469.json"
    crossnobis = json.loads(crossnobis_path.read_text()) if crossnobis_path.exists() else {"patients": {}}
    patient_lambdas = hierarchy["log_scale"]["moment_estimator"].get("patients", {})
    guard = hierarchy["divergence_guard"]
    excluded_patients = {
        row["patient"] for row in guard["excluded_folds"] if row["estimator"] == "moment"
    }
    ctg_rows = []
    crossnobis_rows = []
    variance_rows = []
    patient_rows = {}
    for patient, pooled in sorted(patient_lambdas.items()):
        lambda_rate = float(pooled["partial_pooled_geometric_mean_per_second"])
        session = drift["sessions"].get(patient)
        measured = geometry.get(patient)
        if session is None or measured is None or lambda_rate <= 0.0 or patient in excluded_patients:
            continue
        diffusion_values = [
            fold["moment"]["diffusion"] for fold in session["folds"]
            if fold["moment"].get("diffusion") is not None
            and np.isfinite(fold["moment"]["diffusion"])
            and fold["moment"]["diffusion"] >= 0.0
            and fold["moment"]["diffusion"] <= float(guard["maximum_diffusion"])
        ]
        diffusion = float(np.mean(diffusion_values)) if diffusion_values else None
        predicted_tau = 1.0 / lambda_rate
        content_ctg = measured.get("content_ctg", {})
        observed_tau = (
            float(content_ctg["tau"])
            if content_ctg.get("interpretable") and content_ctg.get("tau") is not None else None
        )
        if observed_tau is not None:
            ctg_rows.append({"patient": patient, "predicted": predicted_tau, "observed": observed_tau})
        crossnobis_patient = crossnobis.get("patients", {}).get(patient, {})
        observed_crossnobis_tau = (
            float(crossnobis_patient["decay"]["timescale_seconds"])
            if crossnobis_patient.get("status") == "estimable" else None
        )
        if observed_crossnobis_tau is not None:
            crossnobis_rows.append({
                "patient": patient, "predicted": predicted_tau,
                "observed": observed_crossnobis_tau,
            })
        absolute_probe = np.asarray(session["behavior"]["probe_absolute_residual"], dtype=float)
        observed_probe_variance = float(np.nanmean(absolute_probe ** 2))
        predicted_stationary_variance = (
            float(diffusion / lambda_rate) if diffusion is not None else None
        )
        if predicted_stationary_variance is not None:
            variance_rows.append({
                "patient": patient,
                "predicted": predicted_stationary_variance,
                "observed": observed_probe_variance,
            })
        patient_rows[patient] = {
            "partial_pooled_lambda_per_second": lambda_rate,
            "mean_fold_diffusion": diffusion,
            "predicted_ctg_timescale_seconds": predicted_tau,
            "observed_content_ctg_timescale_seconds": observed_tau,
            "content_ctg_measurement_interpretable": bool(content_ctg.get("interpretable", False)),
            "observed_crossnobis_timescale_seconds": observed_crossnobis_tau,
            "crossnobis_measurement_interpretable": observed_crossnobis_tau is not None,
            "predicted_stationary_variance": predicted_stationary_variance,
            "observed_probe_mean_squared_content_residual": observed_probe_variance,
        }
    rng = np.random.default_rng(20260801)
    ctg_calibration = calibration(ctg_rows, rng)
    crossnobis_calibration = calibration(crossnobis_rows, rng)
    variance_calibration = calibration(variance_rows, rng)
    estimable = bool(
        crossnobis_calibration.get("status") == "estimable"
        and variance_calibration.get("status") == "estimable"
    )
    quantitative_agreement = bool(
        estimable
        and crossnobis_calibration["calibration_slope_patient_bootstrap_interval_95"][0] <= 1.0
        <= crossnobis_calibration["calibration_slope_patient_bootstrap_interval_95"][1]
        and variance_calibration["calibration_slope_patient_bootstrap_interval_95"][0] <= 1.0
        <= variance_calibration["calibration_slope_patient_bootstrap_interval_95"][1]
    )
    identified = bool(
        estimable
        and (
            crossnobis_calibration["calibration_slope_patient_bootstrap_interval_95"][1] < 1.0
            or crossnobis_calibration["calibration_slope_patient_bootstrap_interval_95"][0] > 1.0
        )
        and not (
            crossnobis_calibration["pearson_correlation_patient_bootstrap_interval_95"][0]
            <= 0.0 <=
            crossnobis_calibration["pearson_correlation_patient_bootstrap_interval_95"][1]
        )
    )
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "geometry_from_drift_parameters_dandi000469",
        "dataset": "DANDI 000469",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "prediction_rule": (
            "No geometry parameter is refit: tau is 1/lambda and stationary variance is D/lambda."
        ),
        "patients": patient_rows,
        "cross_temporal_generalization": {
            "pairs": ctg_rows,
            "calibration": ctg_calibration,
            "measurement": "corrected content-decoding lag decay; uninterpretable near-chance fits are excluded explicitly",
            "role": "legacy",
        },
        "crossnobis_content_distance": {
            "pairs": crossnobis_rows,
            "calibration": crossnobis_calibration,
            "measurement": "cross-validated Mahalanobis item-identity distance decay",
            "role": "primary",
            "n_interpretable_before_divergence_guard": crossnobis.get(
                "n_interpretable_patients", 0
            ),
        },
        "probe_dispersion": {
            "pairs": variance_rows,
            "calibration": variance_calibration,
            "measurement": "out-of-fold squared content-axis residual at probe",
        },
        "participation_ratio": {
            "status": "not_estimable",
            "reason": (
                "The available per-patient PR values use the superseded pooled trials-by-time estimator. "
                "A scalar diffusion fit also does not determine full-covariance PR without additional parameters; "
                "introducing a fitted scale would void this prediction test."
            ),
        },
        "overall": {
            "status": "identified" if identified else "non_identified",
            "quantitative_agreement": quantitative_agreement if identified else None,
            "interpretation": (
                "drift_parameters_quantitatively_predict_measured_geometry"
                if quantitative_agreement else "drift_parameters_do_not_quantitatively_predict_all_measured_geometry"
            ) if identified else "available_patient_sample_does_not_identify_agreement_or_disagreement",
            "reason": (
                None if identified else
                "The primary crossnobis calibration does not jointly identify correlation and "
                "calibration away from their decision boundaries at the available patient count."
            ),
        },
        "divergence_exclusions": {
            "patients": sorted(excluded_patients),
            "reason": "at least one moment fold exceeded the declared hierarchy divergence guard",
        },
    }
    destination = ROOT / "results" / "geometry_from_drift_parameters_000469.json"
    destination.write_text(canonical_json(output))

    if not identified or not quantitative_agreement:
        crack_path = ROOT / "results" / "crack_register.json"
        cracks = json.loads(crack_path.read_text())
        entry = next((
            row for row in cracks["entries"]
            if row.get("crack_id") == "geometry-not-generated-by-confined-drift"
        ), None)
        record = {
            "crack_id": "geometry-not-generated-by-confined-drift",
            "status": "non_identified",
            "trigger": (
                f"The crossnobis calibration uses {crossnobis_calibration.get('n_patients', 0)} "
                "guarded patients, but its patient-bootstrap slope and correlation intervals "
                "identify neither agreement nor disagreement."
            ),
            "chase": "Replaced near-chance decoding-timescale fits with native-unit crossnobis decay, excluded sub-12 under the preregistered divergence guard, and retained correlation and calibration as separate patient-bootstrap quantities.",
            "resolution": (
                "The former model-mismatch conclusion remains withdrawn. Crossnobis yielded "
                f"{crossnobis.get('n_interpretable_patients', 0)} bounded, interpretable decay "
                "measurements; the uncertainty still spans both directions, so H4 is "
                "non-identified. Participation ratio remains not estimable without prohibited "
                "extra parameters."
            ),
            "artifact": "results/geometry_from_drift_parameters_000469.json",
        }
        if entry is None:
            cracks["entries"].append(record)
        else:
            entry.update(record)
        crack_path.write_text(canonical_json(cracks))
    print(json.dumps({"output": str(destination), "overall": output["overall"]}, indent=2))


if __name__ == "__main__":
    main()
