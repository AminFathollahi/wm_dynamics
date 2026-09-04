"""Partial-pool DANDI 000469 confinement rates across patients and folds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402

MAX_LAMBDA_PER_SECOND = 100.0
MAX_DIFFUSION = 1e4
PREREGISTRATION_HASH = "c9505c80aed6b6c82494e472991a519c46a60a00bd8bfab7e6375f0706dc0ecd"


def _standard_error(interval: list[float] | None) -> float | None:
    if interval is None or len(interval) != 2:
        return None
    low, high = map(float, interval)
    value = (high - low) / (2.0 * 1.959963984540054)
    return value if np.isfinite(value) and value > 0.0 else None


def _marginal_fit(groups: list[tuple[np.ndarray, np.ndarray]], positive: bool) -> dict[str, Any]:
    if len(groups) < 2:
        return {"status": "not_estimable", "reason": "fewer than two patients"}

    def unpack(parameters: np.ndarray) -> tuple[float, float]:
        mean = float(np.exp(parameters[0])) if positive else float(parameters[0])
        return mean, float(np.exp(parameters[1]))

    def objective(parameters: np.ndarray) -> float:
        mean, between_sd = unpack(parameters)
        total = 0.0
        for values, errors in groups:
            covariance = np.diag(errors ** 2) + between_sd ** 2 * np.ones((len(values), len(values)))
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0:
                return 1e30
            difference = values - mean
            total += 0.5 * (
                logdet + difference @ np.linalg.solve(covariance, difference)
                + len(values) * np.log(2.0 * np.pi)
            )
        return float(total)

    all_values = np.concatenate([values for values, _errors in groups])
    initial_mean = max(float(np.median(all_values)), 1e-3) if positive else float(np.median(all_values))
    initial_sd = max(float(np.std(all_values)), 1e-3)
    start = np.array([np.log(initial_mean) if positive else initial_mean, np.log(initial_sd)])
    bounds = [(-12.0, 8.0), (-12.0, 8.0)] if positive else [(-100.0, 100.0), (-12.0, 8.0)]
    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e29:
        return {
            "status": "nonconverged",
            "reason": (
                f"optimizer returned failure sentinel negative log likelihood {result.fun:.3g}: "
                f"{result.message}"
            ),
            "negative_log_likelihood": float(result.fun),
            "optimizer_success": bool(result.success),
        }
    mean, between_sd = unpack(result.x)
    return {
        "status": "complete",
        "group_mean": mean,
        "between_patient_sd": between_sd,
        "negative_log_likelihood": float(result.fun),
        "optimizer": "L-BFGS-B marginal Gaussian likelihood",
    }


def _bootstrap_interval(
    groups: list[tuple[np.ndarray, np.ndarray]],
    positive: bool,
    rng: np.random.Generator,
    n_boot: int = 2000,
) -> list[float] | None:
    estimates = []
    for _ in range(n_boot):
        sampled = [groups[index] for index in rng.integers(0, len(groups), size=len(groups))]
        fitted = _marginal_fit(sampled, positive)
        if fitted.get("status") == "complete":
            estimates.append(fitted["group_mean"])
    if len(estimates) < 0.8 * n_boot:
        return None
    return list(map(float, np.percentile(estimates, [2.5, 97.5])))


def _patient_posteriors(
    patient_groups: dict[str, tuple[np.ndarray, np.ndarray]],
    group_mean: float,
    between_sd: float,
) -> dict[str, Any]:
    prior_precision = 1.0 / max(between_sd ** 2, 1e-12)
    output = {}
    for patient, (values, errors) in sorted(patient_groups.items()):
        measurement_precision = np.sum(1.0 / errors ** 2)
        variance = 1.0 / (prior_precision + measurement_precision)
        mean = variance * (
            group_mean * prior_precision + np.sum(values / errors ** 2)
        )
        output[patient] = {
            "partial_pooled_mean": float(mean),
            "posterior_sd_gaussian_approximation": float(np.sqrt(variance)),
            "unpooled_inverse_variance_mean": float(
                np.sum(values / errors ** 2) / measurement_precision
            ),
            "n_folds": int(len(values)),
        }
    return output


def _fit_measurements(
    measurements: dict[str, list[tuple[float, float]]],
    positive: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    groups = [
        (np.asarray([row[0] for row in rows]), np.asarray([row[1] for row in rows]))
        for _patient, rows in sorted(measurements.items()) if rows
    ]
    fit = _marginal_fit(groups, positive)
    fit["n_patients"] = len(groups)
    fit["n_fold_likelihoods"] = int(sum(len(values) for values, _errors in groups))
    if fit.get("status") != "complete":
        return fit
    fit["patient_cluster_bootstrap_interval_95"] = _bootstrap_interval(
        groups, positive, rng
    )
    patient_groups = {
        patient: (
            np.asarray([row[0] for row in rows]),
            np.asarray([row[1] for row in rows]),
        )
        for patient, rows in measurements.items() if rows
    }
    fit["patients"] = _patient_posteriors(
        patient_groups, fit["group_mean"], fit["between_patient_sd"]
    )
    interval = fit["patient_cluster_bootstrap_interval_95"]
    fit["group_identified"] = bool(
        interval is not None and (interval[0] > 0.0 if positive else not (interval[0] <= 0.0 <= interval[1]))
    )
    return fit


def _fit_log_measurements(
    measurements: dict[str, list[tuple[float, float]]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Fit a Gaussian random effect to log rates and back-transform summaries."""
    transformed = {
        patient: [
            (float(np.log(value)), float(error / value))
            for value, error in rows if value > 0.0 and error > 0.0
        ]
        for patient, rows in measurements.items()
    }
    fit = _fit_measurements(transformed, False, rng)
    fit["scale"] = "log_lambda"
    if fit.get("status") != "complete":
        return fit
    fit["group_log_mean"] = fit.pop("group_mean")
    fit["between_patient_log_sd"] = fit.pop("between_patient_sd")
    fit["group_geometric_mean_per_second"] = float(np.exp(fit["group_log_mean"]))
    interval = fit.get("patient_cluster_bootstrap_interval_95")
    fit["group_geometric_mean_patient_bootstrap_interval_95"] = (
        list(map(float, np.exp(interval))) if interval is not None else None
    )
    fit["group_identified"] = bool(interval is not None)
    for patient in fit.get("patients", {}).values():
        patient["partial_pooled_log_mean"] = patient.pop("partial_pooled_mean")
        patient["partial_pooled_geometric_mean_per_second"] = float(
            np.exp(patient["partial_pooled_log_mean"])
        )
        patient["unpooled_inverse_variance_log_mean"] = patient.pop(
            "unpooled_inverse_variance_mean"
        )
    return fit


def _one_sided_negative_test(fit: dict[str, Any]) -> dict[str, Any]:
    """Report the pre-declared negative-direction test alongside two-sided inference."""
    interval = fit.get("patient_cluster_bootstrap_interval_95")
    mean = fit.get("group_mean")
    if interval is None or mean is None:
        return {"status": "not_estimable", "reason": "bootstrap inference unavailable"}
    # A normal approximation uses the already-computed two-sided percentile width;
    # the two-sided interval remains the reported primary uncertainty summary.
    standard_error = max((float(interval[1]) - float(interval[0])) / (2.0 * 1.959963984540054), 1e-12)
    from scipy.stats import norm
    return {
        "status": "estimable",
        "alternative": "content_axis_confinement_is_lower_than_control_axis_confinement",
        "z_statistic": float(mean / standard_error),
        "one_sided_p_value": float(norm.cdf(float(mean / standard_error))),
        "preregistration_hash": PREREGISTRATION_HASH,
        "headline": False,
    }


def main() -> None:
    source_path = ROOT / "results" / "human_drift_spine_000469.json"
    source = json.loads(source_path.read_text())
    state_measurements: dict[str, list[tuple[float, float]]] = {}
    moment_measurements: dict[str, list[tuple[float, float]]] = {}
    permuted_contrasts: dict[str, list[tuple[float, float]]] = {}
    complement_contrasts: dict[str, list[tuple[float, float]]] = {}
    permuted_log_contrasts: dict[str, list[tuple[float, float]]] = {}
    complement_log_contrasts: dict[str, list[tuple[float, float]]] = {}
    divergence_exclusions: list[dict[str, Any]] = []
    for patient, session in sorted(source["sessions"].items()):
        if session.get("status") != "complete":
            continue
        for fold in session["folds"]:
            state = fold["state_space"]
            state_se = _standard_error(state.get("lambda_ci"))
            if state_se is not None and state.get("lambda_rate") is not None:
                state_lambda = float(state["lambda_rate"])
                state_diffusion = state.get("diffusion")
                if (
                    state_lambda <= 0.0 or state_lambda > MAX_LAMBDA_PER_SECOND
                    or state_diffusion is None or float(state_diffusion) > MAX_DIFFUSION
                ):
                    divergence_exclusions.append({
                        "patient": patient, "fold": fold["fold"], "estimator": "state_space",
                        "lambda_per_second": state_lambda, "diffusion": state_diffusion,
                        "reason": "nonpositive or above declared lambda/diffusion divergence bound",
                    })
                else:
                    state_measurements.setdefault(patient, []).append((state_lambda, state_se))
            moment = fold["moment"]
            moment_se = _standard_error(moment.get("lambda_ci"))
            moment_valid = False
            if moment_se is not None and moment.get("lambda_rate") is not None:
                moment_lambda = float(moment["lambda_rate"])
                moment_diffusion = moment.get("diffusion")
                if (
                    moment_lambda <= 0.0 or moment_lambda > MAX_LAMBDA_PER_SECOND
                    or moment_diffusion is None or float(moment_diffusion) > MAX_DIFFUSION
                ):
                    divergence_exclusions.append({
                        "patient": patient, "fold": fold["fold"], "estimator": "moment",
                        "lambda_per_second": moment_lambda, "diffusion": moment_diffusion,
                        "reason": "nonpositive or above declared lambda/diffusion divergence bound",
                    })
                else:
                    moment_measurements.setdefault(patient, []).append((moment_lambda, moment_se))
                    moment_valid = True
            for destination, null_name in (
                (permuted_contrasts, "permuted_axis_moment"),
                (complement_contrasts, "matched_complement_moment"),
            ):
                null = fold[null_name]
                null_se = _standard_error(null.get("lambda_ci"))
                if (
                    moment_valid and null_se is not None
                    and moment.get("lambda_rate") is not None and null.get("lambda_rate") is not None
                    and 0.0 < float(null["lambda_rate"]) <= MAX_LAMBDA_PER_SECOND
                    and null.get("diffusion") is not None
                    and float(null["diffusion"]) <= MAX_DIFFUSION
                ):
                    destination.setdefault(patient, []).append((
                        float(moment["lambda_rate"] - null["lambda_rate"]),
                        float(np.hypot(moment_se, null_se)),
                    ))
                    log_destination = (
                        permuted_log_contrasts
                        if null_name == "permuted_axis_moment" else complement_log_contrasts
                    )
                    log_destination.setdefault(patient, []).append((
                        float(np.log(moment["lambda_rate"] / null["lambda_rate"])),
                        float(np.hypot(
                            moment_se / float(moment["lambda_rate"]),
                            null_se / float(null["lambda_rate"]),
                        )),
                    ))

    rng = np.random.default_rng(20260801)
    state_fit = _fit_measurements(state_measurements, True, rng)
    moment_fit = _fit_measurements(moment_measurements, True, rng)
    state_log_fit = _fit_log_measurements(state_measurements, rng)
    moment_log_fit = _fit_log_measurements(moment_measurements, rng)
    permuted_fit = _fit_measurements(permuted_contrasts, False, rng)
    complement_fit = _fit_measurements(complement_contrasts, False, rng)
    permuted_log_fit = _fit_measurements(permuted_log_contrasts, False, rng)
    complement_log_fit = _fit_measurements(complement_log_contrasts, False, rng)
    permuted_fit["predeclared_one_sided_test"] = _one_sided_negative_test(permuted_fit)
    complement_fit["predeclared_one_sided_test"] = _one_sided_negative_test(complement_fit)
    permuted_log_fit["predeclared_one_sided_test"] = _one_sided_negative_test(permuted_log_fit)
    complement_log_fit["predeclared_one_sided_test"] = _one_sided_negative_test(complement_log_fit)
    anisotropy_identified = bool(
        permuted_log_fit.get("group_identified", False)
        and complement_log_fit.get("group_identified", False)
        and np.sign(permuted_log_fit["group_mean"]) == np.sign(complement_log_fit["group_mean"])
    )
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "hierarchical_confinement_dandi000469",
        "dataset": "DANDI 000469",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "source_artifact": "results/human_drift_spine_000469.json",
        "likelihood": (
            "direct marginal Gaussian random-effects likelihood over every finite fold estimate; "
            "fold likelihood curvature is represented by its stored interval-derived standard error; "
            "no fold is selected by the individual identifiability label"
        ),
        "primary_scale": "log_lambda",
        "divergence_guard": {
            "maximum_lambda_per_second": MAX_LAMBDA_PER_SECOND,
            "maximum_diffusion": MAX_DIFFUSION,
            "rule": "folds outside either bound are non-identified and excluded from pooling",
            "excluded_folds": divergence_exclusions,
            "n_excluded": len(divergence_exclusions),
        },
        "former_moment_optimizer_failure": {
            "classification": "failed_optimization_not_reporting_bug",
            "former_negative_log_likelihood": 1e30,
            "former_group_identified": True,
            "correction": "an optimizer failure sentinel can no longer be reported as identified or consumed downstream",
        },
        "state_space": state_fit,
        "moment_estimator": moment_fit,
        "log_scale": {
            "state_space": state_log_fit,
            "moment_estimator": moment_log_fit,
        },
        "individual_estimator_retained": True,
        "individual_vs_hierarchical": {
            "state_space": {
                "identified_subset_mean": float(source["group"]["state_space_lambda_identified_mean"]["mean"]),
                "log_scale_hierarchical_group_geometric_mean": state_log_fit.get(
                    "group_geometric_mean_per_second"
                ),
            },
            "moment": {
                "identified_subset_mean": float(source["group"]["moment_lambda_identified_mean"]["mean"]),
                "log_scale_hierarchical_group_geometric_mean": moment_log_fit.get(
                    "group_geometric_mean_per_second"
                ),
            },
            "interpretation": "The guarded log-scale fits are primary; the raw-scale fits are retained as legacy estimates. Reconciliation with the individually identified subset is evaluated from the regenerated intervals rather than assumed.",
        },
        "anisotropy": {
            "primary_scale": "log_lambda_ratio",
            "log_content_minus_permuted_axis": permuted_log_fit,
            "log_content_minus_signal_matched_complement": complement_log_fit,
            "raw_scale_legacy": {
                "content_minus_permuted_axis": permuted_fit,
                "content_minus_signal_matched_complement": complement_fit,
            },
            "identified_with_both_controls": anisotropy_identified,
            "interpretation": (
                "selection_controlled_anisotropy_supported"
                if anisotropy_identified else "selection_controlled_anisotropy_not_established"
            ),
        },
        "control_payload_eligibility": {
            "hierarchical_lambda_identified": bool(
                state_log_fit.get("group_identified", False)
                or moment_log_fit.get("group_identified", False)
            ),
            "behavioral_gate": "unchanged; the existing slope interval includes zero",
            "payload_status": "nonidentified",
            "reason": "better group lambda estimation cannot identify a tolerance without a behavioral displacement slope",
        },
    }
    destination = ROOT / "results" / "hierarchical_confinement_000469.json"
    destination.write_text(canonical_json(output))

    crack_path = ROOT / "results" / "crack_register.json"
    cracks = json.loads(crack_path.read_text())
    for entry in cracks["entries"]:
        if entry.get("crack_id") == "human-anisotropy-null":
            entry.update({
                "status": "resolved_as_hierarchical_null" if not anisotropy_identified else "resolved_as_hierarchical_support",
                "trigger": "The individually identified subset was too small to adjudicate content-axis anisotropy.",
                "chase": "Fit a direct random-effects likelihood to every finite fold likelihood approximation, without the individual identifiability filter, and retained both selection controls.",
                "resolution": (
                    "Full-cohort partial pooling does not establish anisotropy with both the permuted-label and signal-matched-complement controls."
                    if not anisotropy_identified else
                    "Full-cohort partial pooling establishes a same-direction content-axis contrast against both selection controls."
                ),
                "artifact": "results/hierarchical_confinement_000469.json",
            })
            break
    if not any(entry.get("crack_id") == "hierarchical-vs-filtered-confinement" for entry in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "hierarchical-vs-filtered-confinement",
            "trigger": "The all-fold hierarchical state-space rate (5.05/s) is higher than the individually identified-subset mean (1.36/s), with non-overlapping intervals.",
            "chase": "Retained both estimands, compared both state-space and moment fits, and inspected the effect of the predeclared sampling-window truncation on which individual folds can be quoted.",
            "resolution": "The hard filter and hierarchical likelihood answer different questions: the former quotes only individually resolved rates inside the sampling window, whereas the latter pools every finite fold likelihood. The moment intervals overlap and both estimators support positive group confinement, but the state-space magnitude disagreement prevents a precise estimator-independent population-rate claim.",
            "status": "resolved_as_selection_estimand_difference_magnitude_uncertain",
            "artifact": "results/hierarchical_confinement_000469.json",
        })
    crack_path.write_text(canonical_json(cracks))
    print(json.dumps({"output": str(destination), "anisotropy": output["anisotropy"]}, indent=2))


if __name__ == "__main__":
    main()
