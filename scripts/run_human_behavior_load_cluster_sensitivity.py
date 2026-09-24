from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statistics import Z_80_POWER, forest_meta, stable_seed  # noqa: E402


N_BOOTSTRAP = 8000
MEANINGFUL_EFFECT_R = 0.14
SOURCE_ARTIFACT = ROOT / "results" / "human_maintenance_behaviour_link.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _computed_r(patient: dict[str, Any], statistic: str) -> float | None:
    result = patient.get(statistic, {})
    value = result.get("r")
    if patient.get("status") != "computed" or result.get("status") != "computed":
        return None
    return float(value) if value is not None and np.isfinite(value) else None


def extract_arms(data: dict[str, Any], statistic: str) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for corpus, detail in data["block_b"]["per_corpus"].items():
        corpus_arms = {}
        for load, arm in detail["by_load"].items():
            pooled = arm[f"pooled_{statistic}"]
            if pooled.get("status") != "tested":
                continue
            values = {
                patient_id: value
                for patient_id, patient in arm["patients"].items()
                if (value := _computed_r(patient, statistic)) is not None
            }
            if not values:
                continue
            ci_lower, ci_upper = float(pooled["ci_lower"]), float(pooled["ci_upper"])
            standard_error = (ci_upper - ci_lower) / (2.0 * 1.959963984540054)
            corpus_arms[str(load)] = {
                "values": values,
                "frozen_standard_error": standard_error,
                "original_estimate": float(pooled["mean_value"]),
                "original_n_patients": int(arm["n_patients_computed"]),
            }
        if corpus_arms:
            output[corpus] = corpus_arms
    return output


def _corpus_patient_ids(arms: dict[str, dict[str, Any]]) -> list[str]:
    return sorted({patient for arm in arms.values() for patient in arm["values"]})


def _meta_from_selected_patients(
    arms_by_corpus: dict[str, dict[str, dict[str, Any]]],
    selected: dict[str, np.ndarray],
    *,
    refit_arm_standard_errors: bool = False,
) -> float | None:
    fit = _fit_selected_patients(
        arms_by_corpus,
        selected,
        refit_arm_standard_errors=refit_arm_standard_errors,
    )
    return None if fit is None else float(fit["pooled"])


def _fit_selected_patients(
    arms_by_corpus: dict[str, dict[str, dict[str, Any]]],
    selected: dict[str, np.ndarray],
    *,
    refit_arm_standard_errors: bool,
) -> dict[str, Any] | None:
    estimates, ses, labels = [], [], []
    for corpus, arms in arms_by_corpus.items():
        for load, arm in arms.items():
            values = [arm["values"][patient] for patient in selected[corpus] if patient in arm["values"]]
            if not values:
                continue
            if refit_arm_standard_errors:
                if len(values) < 2:
                    continue
                standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if not np.isfinite(standard_error) or standard_error <= 0.0:
                    continue
                ses.append(standard_error)
            else:
                ses.append(arm["frozen_standard_error"])
            estimates.append(float(np.mean(values)))
            labels.append(f"{corpus}_load{load}")
    if not estimates:
        return None
    return forest_meta(np.asarray(estimates), np.asarray(ses), labels)


def _bootstrap_summary(draws: np.ndarray, point: float, n_requested: int) -> dict[str, Any]:
    valid = draws[np.isfinite(draws)]
    if len(valid) < 2:
        return {
            "status": "not_computable",
            "n_draws_requested": n_requested,
            "n_draws_valid": int(len(valid)),
            "n_draws_invalid": int(n_requested - len(valid)),
        }
    standard_error = float(np.std(valid, ddof=1))
    z_power = Z_80_POWER
    return {
        "status": "computed",
        "n_draws_requested": n_requested,
        "n_draws_valid": int(len(valid)),
        "n_draws_invalid": int(n_requested - len(valid)),
        "point_estimate": point,
        "cluster_bootstrap_ci_95pct": [float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))],
        "cluster_bootstrap_se": standard_error,
        "normal_approximation_mdd_80pct_power": z_power * standard_error,
        "mdd_reference_r_units": MEANINGFUL_EFFECT_R,
        "mdd_below_reference": bool(z_power * standard_error < MEANINGFUL_EFFECT_R),
    }


def participant_cluster_bootstrap(
    arms_by_corpus: dict[str, dict[str, dict[str, Any]]],
    n_draws: int = N_BOOTSTRAP,
    seed_tag: str = "human_behavior_load_cluster_sensitivity",
) -> dict[str, Any]:
    if n_draws < 2:
        raise ValueError("n_draws must be at least two")
    patient_ids = {corpus: _corpus_patient_ids(arms) for corpus, arms in arms_by_corpus.items()}
    if any(len(ids) == 0 for ids in patient_ids.values()):
        raise ValueError("each corpus needs at least one computed patient")
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.full(n_draws, np.nan, dtype=float)
    for draw in range(n_draws):
        selected = {
            corpus: rng.choice(ids, size=len(ids), replace=True)
            for corpus, ids in patient_ids.items()
        }
        estimate = _meta_from_selected_patients(arms_by_corpus, selected)
        if estimate is not None:
            draws[draw] = estimate
    point = _meta_from_selected_patients(
        arms_by_corpus, {corpus: np.asarray(ids) for corpus, ids in patient_ids.items()}
    )
    return {
        **_bootstrap_summary(draws, float(point), n_draws),
        "conditional_weighting": True,
        "seed_tag": seed_tag,
        "n_unique_patients_by_corpus": {corpus: len(ids) for corpus, ids in patient_ids.items()},
    }


def refit_weight_participant_cluster_bootstrap(
    arms_by_corpus: dict[str, dict[str, dict[str, Any]]],
    n_draws: int = N_BOOTSTRAP,
    seed_tag: str = "human_behavior_load_cluster_sensitivity_refit_weights",
) -> dict[str, Any]:
    if n_draws < 2:
        raise ValueError("n_draws must be at least two")
    patient_ids = {corpus: _corpus_patient_ids(arms) for corpus, arms in arms_by_corpus.items()}
    if any(len(ids) == 0 for ids in patient_ids.values()):
        raise ValueError("each corpus needs at least one computed patient")
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.full(n_draws, np.nan, dtype=float)
    arm_counts = np.zeros(n_draws, dtype=int)
    arm_hits = {f"{corpus}_load{load}": 0 for corpus, arms in arms_by_corpus.items() for load in arms}
    for draw in range(n_draws):
        selected = {
            corpus: rng.choice(ids, size=len(ids), replace=True)
            for corpus, ids in patient_ids.items()
        }
        fit = _fit_selected_patients(
            arms_by_corpus, selected, refit_arm_standard_errors=True
        )
        if fit is not None:
            draws[draw] = fit["pooled"]
            arm_counts[draw] = fit["k"]
            for row in fit["rows"]:
                arm_hits[row["label"]] += 1
    point_fit = _fit_selected_patients(
        arms_by_corpus,
        {corpus: np.asarray(ids) for corpus, ids in patient_ids.items()},
        refit_arm_standard_errors=True,
    )
    if point_fit is None:
        raise RuntimeError("observed sample did not produce a refitted-weight meta estimate")
    valid_counts = arm_counts[np.isfinite(draws)]
    return {
        **_bootstrap_summary(draws, float(point_fit["pooled"]), n_draws),
        "conditional_weighting": False,
        "estimand": "random-effects mean of load-specific patient means with resample-specific precision",
        "arm_precision_refit": "sample standard deviation divided by sqrt(n) in every resampled arm",
        "refit_scope": ["arm means", "arm standard errors", "between-arm heterogeneity", "meta-analysis weights"],
        "trial_level_patient_estimates_refit": False,
        "n_arms_observed": int(point_fit["k"]),
        "n_arms_per_valid_draw": {
            "min": int(np.min(valid_counts)) if len(valid_counts) else None,
            "median": float(np.median(valid_counts)) if len(valid_counts) else None,
            "max": int(np.max(valid_counts)) if len(valid_counts) else None,
        },
        "arm_inclusion_fraction": {
            label: hits / n_draws for label, hits in arm_hits.items()
        },
        "seed_tag": seed_tag,
        "n_unique_patients_by_corpus": {corpus: len(ids) for corpus, ids in patient_ids.items()},
    }


def equal_patient_cluster_bootstrap(
    arms_by_corpus: dict[str, dict[str, dict[str, Any]]],
    n_draws: int = N_BOOTSTRAP,
    seed_tag: str = "human_behavior_load_cluster_sensitivity_equal_patient",
) -> dict[str, Any]:
    patient_ids = {corpus: _corpus_patient_ids(arms) for corpus, arms in arms_by_corpus.items()}

    def estimate(selected: dict[str, np.ndarray]) -> float:
        corpus_means = []
        for corpus, arms in arms_by_corpus.items():
            patient_means = [
                float(np.mean([arm["values"][patient] for arm in arms.values() if patient in arm["values"]]))
                for patient in selected[corpus]
            ]
            corpus_means.append(float(np.mean(patient_means)))
        return float(np.mean(corpus_means))

    observed_selection = {corpus: np.asarray(ids) for corpus, ids in patient_ids.items()}
    point = estimate(observed_selection)
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.asarray([
        estimate({corpus: rng.choice(ids, size=len(ids), replace=True) for corpus, ids in patient_ids.items()})
        for _ in range(n_draws)
    ])
    return {
        **_bootstrap_summary(draws, point, n_draws),
        "estimand": "equal weight per patient across available loads within corpus, then equal weight per corpus",
        "seed_tag": seed_tag,
        "n_unique_patients_by_corpus": {corpus: len(ids) for corpus, ids in patient_ids.items()},
    }


def equal_patient_load_averages(arms_by_corpus: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_corpus = {}
    for corpus, arms in arms_by_corpus.items():
        patient_ids = _corpus_patient_ids(arms)
        values = {
            patient: float(np.mean([arm["values"][patient] for arm in arms.values() if patient in arm["values"]]))
            for patient in patient_ids
        }
        by_corpus[corpus] = {
            "n_patients": len(values),
            "patient_mean_across_available_loads": values,
            "corpus_mean": float(np.mean(list(values.values()))),
        }
    corpus_means = [entry["corpus_mean"] for entry in by_corpus.values()]
    return {
        "status": "descriptive_only",
        "by_corpus": by_corpus,
        "unweighted_mean_of_corpus_means": float(np.mean(corpus_means)),
        "caveat": "This equal-patient summary is descriptive and does not replace the delivered seven-arm estimand.",
    }


def coverage(arms_by_corpus: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        corpus: {
            load: {
                "n_patients_with_computed_statistic": len(arm["values"]),
                "n_patients_in_delivered_arm": arm["original_n_patients"],
                "canonical_patient_ids": sorted(arm["values"]),
            }
            for load, arm in arms.items()
        }
        for corpus, arms in arms_by_corpus.items()
    }


def build_sensitivity(data: dict[str, Any], n_draws: int = N_BOOTSTRAP) -> dict[str, Any]:
    readings = {}
    for statistic in ("raw", "joint_partial"):
        arms = extract_arms(data, statistic)
        readings[statistic] = {
            "participant_load_coverage": coverage(arms),
            "conditional_weight_participant_cluster_bootstrap": participant_cluster_bootstrap(
                arms, n_draws=n_draws, seed_tag=f"human_behavior_load_cluster_sensitivity|{statistic}"
            ),
            "refit_weight_participant_cluster_bootstrap": refit_weight_participant_cluster_bootstrap(
                arms, n_draws=n_draws, seed_tag=f"human_behavior_load_cluster_sensitivity|refit|{statistic}"
            ),
            "equal_patient_cluster_bootstrap": equal_patient_cluster_bootstrap(
                arms, n_draws=n_draws, seed_tag=f"human_behavior_load_cluster_sensitivity|equal|{statistic}"
            ),
            "equal_patient_within_corpus_load_average": equal_patient_load_averages(arms),
        }
    return {
        "status": "sensitivity_complete_not_a_primary_replacement",
        "source_artifact": "results/human_maintenance_behaviour_link.json",
        "reference_meaningful_effect_r_units": MEANINGFUL_EFFECT_R,
        "refit_capability": {
            "patient_load_estimates_available": True,
            "trial_arrays_available_in_source_artifact": False,
            "supported_outer_refit": [
                "load-arm means",
                "load-arm standard errors",
                "between-arm heterogeneity",
                "random-effects meta-analysis weights",
            ],
            "unsupported_outer_refit": "within-patient trial-level correlations and nuisance models",
        },
        "cross_corpus_identity_status": (
            "unresolved: patient identifiers are canonical only within their corpus namespaces; no cross-corpus "
            "identity map was found in the source artifact, so same-looking identifiers were not merged."
        ),
        "caveats": [
            "The bootstrap keeps all available load cells of a resampled patient together but uses the delivered arm standard errors as fixed weights.",
            "The conditional analysis does not refit arm precision; the refit-weight analysis recomputes arm standard errors and random-effects weights from each patient resample.",
            "Within-patient trial-level statistics are not refit because this additive sensitivity reads patient estimates from the delivered artifact.",
            "The selected finite cohort and missing patient-load cells remain part of the estimand.",
            "Bootstrap MDD values use a stated normal approximation; no p-value-derived branch or silent primary-result replacement is produced.",
        ],
        "readings": readings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify participant-cluster uncertainty across load arms.")
    parser.add_argument("--input", type=Path, default=SOURCE_ARTIFACT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/human_behavior_load_cluster_sensitivity.json")
    parser.add_argument("--n-draws", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    output = build_sensitivity(data, args.n_draws)
    output["source_hashes"] = {
        "results/human_maintenance_behaviour_link.json": sha256_file(args.input),
        "scripts/run_human_behavior_load_cluster_sensitivity.py": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
