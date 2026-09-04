#!/usr/bin/env python3
"""How much of the standing closed-loop control claim survives once the
state is only imperfectly observed?

The project's control payload assumes the confinement rate lambda can be
driven to lambda+g by feedback, cutting stationary variance from D/lambda to
D/(lambda+g). That assumes the controller can read the state exactly. With
observation noise it cannot: the feedback can only act on a Kalman-filtered
estimate, and the achievable variance is bounded below by the filter's own
steady-state estimation error, which no feedback gain can shrink further.

Per structure with at least three patients, using each patient's own
confinement fit (lambda, diffusion, stationary variance -- the existing
region-stratified drift machinery) and each patient's own observation
variance (implied by the nugget fraction computed here on the same latent's
variance scale), this solves the steady-state scalar Kalman filter and
reports the resulting ceiling: the fraction of state variance that cannot be
estimated, and therefore cannot be corrected by any feedback controller.

Deciding contrast: does that ceiling separate structures more than the
confinement rate itself already does (the 1/lambda spread on record in
results/structure_registry.json)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control import scalar_steady_state_kalman_error  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import stable_seed  # noqa: E402

SEED = 20260809
BIN_S = 0.1
MIN_PATIENTS_PER_STRUCTURE = 3
N_BOOTSTRAP = 2000
ALM_HYPOTHETICAL_NUGGET = 0.03
SATURATION_FRACTION = 0.95

REGION_DRIFT_ARTIFACTS = {
    "dandi_000469": ROOT / "results" / "region_stratified_drift_000469.json",
}
OBSERVABILITY_CENSUS = ROOT / "results" / "observability_census.json"
STRUCTURE_REGISTRY = ROOT / "results" / "structure_registry.json"
OUTPUT_PATH = ROOT / "results" / "control_ceiling.json"


def ceiling_from_lambda_diffusion(lambda_rate: float, diffusion: float, stationary_variance: float, p_post: float) -> dict:
    ceiling_fraction = float(p_post / stationary_variance) if stationary_variance > 0 else float("nan")
    return {"ceiling_fraction": ceiling_fraction, "max_achievable_reduction": 1.0 - ceiling_fraction}


def achievable_reduction_curve(lambda_rate: float, diffusion: float, p_post: float, stationary_variance: float,
                                gains: list[float]) -> list[dict]:
    rows = []
    for g in gains:
        var_g = diffusion / (lambda_rate + g) + p_post
        reduction = 1.0 - var_g / stationary_variance if stationary_variance > 0 else float("nan")
        rows.append({"gain": float(g), "closed_loop_variance": float(var_g), "reduction": float(reduction)})
    return rows


def gain_at_saturation(lambda_rate: float, diffusion: float, stationary_variance: float, p_post: float,
                        fraction_of_max: float = SATURATION_FRACTION) -> float:
    """Feedback gain at which the achievable reduction first reaches
    ``fraction_of_max`` of its asymptotic (g -> infinity) ceiling-bounded
    maximum, solved in closed form from the reduction curve's own algebra."""
    residual = stationary_variance - p_post
    if residual <= 0:
        return float("nan")
    target_gap = (1.0 - fraction_of_max) * residual
    if target_gap <= 0:
        return float("inf")
    return float(diffusion / target_gap - lambda_rate)


def patient_state_space_fit(sessions: dict) -> dict[str, dict]:
    """Mean (lambda_rate, diffusion, process_variance, stationary_variance)
    over identified ``gaussian_lgssm`` folds, per patient."""
    out = {}
    for patient, session in sessions.items():
        folds = session.get("folds")
        if not folds:
            continue
        identified = [f["state_space"] for f in folds if f.get("state_space", {}).get("status") == "identifiable"]
        if not identified:
            continue
        out[patient] = {
            "lambda_rate": float(np.mean([f["lambda_rate"] for f in identified])),
            "diffusion": float(np.mean([f["diffusion"] for f in identified])),
            "process_variance": float(np.mean([f["process_variance"] for f in identified])),
            "stationary_variance": float(np.mean([f["stationary_variance"] for f in identified])),
            "n_folds_identified": len(identified),
        }
    return out


def load_nugget_by_patient(census_rows: list[dict], dataset: str, structure: str) -> dict[str, float]:
    out = {}
    for row in census_rows:
        if (row["dataset"] == dataset and row["structure"] == structure and row["epoch"] == "delay"
                and row["bin_ms"] == 100 and row["status"] == "fitted"):
            out[row["patient"]] = row["median_nugget_fraction"]
    return out


def patient_ceiling(confinement: dict, nugget_fraction: float, hypothetical_nugget: float | None = None) -> dict:
    lambda_rate = confinement["lambda_rate"]
    diffusion = confinement["diffusion"]
    stationary_variance = confinement["stationary_variance"]
    a = float(np.exp(-lambda_rate * BIN_S))
    process_variance = confinement["process_variance"]

    def _solve(nugget: float) -> dict:
        nugget = min(max(nugget, 1e-6), 1.0 - 1e-6)
        r_obs = nugget / (1.0 - nugget) * stationary_variance
        _, p_post = scalar_steady_state_kalman_error(a, process_variance, r_obs)
        ceiling = ceiling_from_lambda_diffusion(lambda_rate, diffusion, stationary_variance, p_post)
        gains = list(np.geomspace(max(lambda_rate * 1e-3, 1e-4), lambda_rate * 50 + 50, 40))
        curve = achievable_reduction_curve(lambda_rate, diffusion, p_post, stationary_variance, gains)
        saturation_gain = gain_at_saturation(lambda_rate, diffusion, stationary_variance, p_post)
        return {
            "observation_variance": r_obs, "p_post": p_post,
            "ceiling_fraction": ceiling["ceiling_fraction"],
            "max_achievable_reduction": ceiling["max_achievable_reduction"],
            "saturation_gain": saturation_gain, "reduction_curve": curve,
        }

    result = {
        "lambda_rate": lambda_rate, "diffusion": diffusion, "stationary_variance": stationary_variance,
        "process_variance": process_variance, "nugget_fraction": nugget_fraction,
        "measured": _solve(nugget_fraction),
    }
    if hypothetical_nugget is not None:
        result["hypothetical_alm_nugget"] = _solve(hypothetical_nugget)
    return result


def bootstrap_structure_spread(structure_patients: dict[str, list[float]], rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Resample patients within each structure; return n_boot draws of
    max-over-min structure-median achievable reduction."""
    draws = np.empty(n_boot)
    structures = list(structure_patients.keys())
    for b in range(n_boot):
        medians = []
        for structure in structures:
            values = np.array(structure_patients[structure])
            resample = values[rng.integers(0, len(values), size=len(values))]
            medians.append(np.median(resample))
        medians = np.array(medians)
        draws[b] = medians.max() / max(medians.min(), 1e-12)
    return draws


def lambda_spread_reference() -> dict:
    """Recompute the across-structure 1/lambda spread from
    results/structure_registry.json rather than hard-coding it, so this
    stays the exact figure on record when that artifact is regenerated."""
    if not STRUCTURE_REGISTRY.exists():
        return {"status": "not_estimable", "reason": "results/structure_registry.json is missing"}
    registry = json.loads(STRUCTURE_REGISTRY.read_text())
    medians = []
    for name, s in registry["structures"].items():
        c = s.get("confinement_rate_lambda_pooled_across_datasets", {})
        if c.get("status") == "estimable" and c.get("n_patients", 0) >= MIN_PATIENTS_PER_STRUCTURE:
            medians.append((name, c["median"]))
    if len(medians) < MIN_PATIENTS_PER_STRUCTURE:
        return {"status": "not_estimable", "reason": "fewer than 3 structures identified in structure_registry.json"}
    lambdas = np.array([m for _, m in medians])
    spread = float(lambdas.max() / lambdas.min())
    return {"status": "estimable", "spread": spread, "n_structures": len(medians),
            "structures": [m[0] for m in medians]}


def main() -> None:
    census = json.loads(OBSERVABILITY_CENSUS.read_text())
    census_rows = census["rows"]

    per_structure_patients: dict[str, dict[str, dict]] = {}
    for dataset, path in REGION_DRIFT_ARTIFACTS.items():
        if not path.exists():
            continue
        artifact = json.loads(path.read_text())
        for structure, region_data in artifact.get("regions", {}).items():
            if structure == "pooled":
                continue
            confinement_by_patient = patient_state_space_fit(region_data["sessions"])
            nugget_by_patient = load_nugget_by_patient(census_rows, dataset, structure)
            per_structure_patients.setdefault(structure, {})
            for patient, confinement in confinement_by_patient.items():
                if patient not in nugget_by_patient:
                    continue
                per_structure_patients[structure][f"{dataset}:{patient}"] = patient_ceiling(
                    confinement, nugget_by_patient[patient], hypothetical_nugget=ALM_HYPOTHETICAL_NUGGET,
                )

    structures_output = {}
    structure_median_reduction: dict[str, list[float]] = {}
    for structure, patients in per_structure_patients.items():
        n_patients = len(patients)
        status = "estimable" if n_patients >= MIN_PATIENTS_PER_STRUCTURE else "non_identified"
        reductions = [p["measured"]["max_achievable_reduction"] for p in patients.values()]
        ceilings = [p["measured"]["ceiling_fraction"] for p in patients.values()]
        hypothetical_reductions = [p["hypothetical_alm_nugget"]["max_achievable_reduction"] for p in patients.values()]
        structures_output[structure] = {
            "n_patients": n_patients, "status": status,
            "median_ceiling_fraction": float(np.median(ceilings)) if ceilings else None,
            "median_max_achievable_reduction": float(np.median(reductions)) if reductions else None,
            "median_hypothetical_alm_nugget_reduction": float(np.median(hypothetical_reductions)) if hypothetical_reductions else None,
            "patients": patients,
        }
        if status == "estimable":
            structure_median_reduction[structure] = reductions

    reference = lambda_spread_reference()
    eligible_structures = list(structure_median_reduction.keys())
    if len(eligible_structures) < MIN_PATIENTS_PER_STRUCTURE or reference["status"] != "estimable":
        decision = {
            "status": "ceiling_not_estimable",
            "reason": f"only {len(eligible_structures)} structures have both a nugget and a confinement estimate at n>=3 patients",
            "n_structures_eligible": len(eligible_structures),
        }
    else:
        rng = np.random.default_rng(stable_seed("control_ceiling_bootstrap") ^ SEED)
        medians = {s: float(np.median(v)) for s, v in structure_median_reduction.items()}
        observed_spread = max(medians.values()) / max(min(medians.values()), 1e-12)
        boot_spread = bootstrap_structure_spread(structure_median_reduction, rng, N_BOOTSTRAP)
        boot_ratio = boot_spread / reference["spread"]
        ci_lo, ci_hi = np.percentile(boot_ratio, [2.5, 97.5])
        if observed_spread > reference["spread"] and ci_lo > 1.0:
            branch = "ceiling_orders_structures"
        else:
            branch = "ceiling_uniform"
        decision = {
            "status": branch,
            "observed_ceiling_spread": observed_spread,
            "lambda_spread_reference": reference["spread"],
            "ratio_bootstrap_ci95": [float(ci_lo), float(ci_hi)],
            "n_bootstrap": N_BOOTSTRAP,
            "structure_medians": medians,
            "fastest_structure": max(medians, key=medians.get),
            "slowest_structure": min(medians, key=medians.get),
        }

    payload = {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "bin_s": BIN_S,
        "alm_hypothetical_nugget": ALM_HYPOTHETICAL_NUGGET,
        "saturation_fraction": SATURATION_FRACTION,
        "scope": ("DANDI 000469 only: the only corpus whose region-stratified confinement "
                  "fit (results/region_stratified_drift_000469.json) exposes per-patient, "
                  "per-fold gaussian_lgssm state-space parameters (lambda, diffusion, process "
                  "variance, stationary variance) alongside the per-patient observability census "
                  "computed here on matching patient identifiers. Extending to "
                  "001187/000673 (whose region-stratified artifact does not carry the same "
                  "state-space battery) and 000574 (session-keyed, not patient-keyed, in its "
                  "region-stratified artifact) is deferred."),
        "observation_variance_assumption": (
            "R is inferred by applying the session-level nugget fraction -- a "
            "scale-invariant ratio -- to the confinement fit's own stationary_variance, i.e. "
            "R = nugget/(1-nugget) * stationary_variance. This assumes the same fractional "
            "observation-noise contamination measured on the leading PCA latent applies to the "
            "discriminant-axis latent the confinement fit itself uses; it is not an independent "
            "per-axis noise measurement, and is stated here as an explicit modeling assumption."),
        "reference_lambda_spread": reference,
        "structures": structures_output,
        "predeclared_decision": decision,
    }
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
