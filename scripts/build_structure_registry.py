#!/usr/bin/env python3
"""Structure-as-unit synthesis across the full corpus.

scripts/build_structure_paired_contrasts.py answers the within-patient paired question. This
script answers the separate pooled-at-scale question: for each anatomical structure, pool every
patient's estimate across every dataset that has a region-stratified lambda/D fit, with dataset
kept as an explicit stratum (never silently averaged away) and task recorded as a stratum.
Species is not a stratum here -- every eligible dataset is human intracranial; that is stated,
not assumed.

Eligible datasets (same six as build_structure_paired_contrasts.py):
    DANDI 000469, DANDI 001187+000673 (content_axis_battery), DANDI 000574, ds004752, ds005489
    (RAM open-loop), ds005557 (RAM closed-loop). Every other census dataset is listed under
    `datasets_not_eligible` with its reason -- not silently dropped.

Gated on results/patient_identity_audit.json: 000469 has zero verified cross-release overlap
with 001187/000673, so pooling their patients within a structure never double-counts a patient.

Run (after every region-stratified/LFP structure artifact below exists):
    conda run -n wm_dynamics python scripts/build_structure_registry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402
from build_structure_paired_contrasts import (  # noqa: E402
    extract_000469, extract_000574, extract_001187_000673, extract_lfp,
)

MIN_STRUCTURE_PATIENTS = 3
RESULTS = ROOT / "results"

# Task is recorded as a stratum, not collapsed.
DATASET_TASK = {
    "dandi_000469": "Sternberg repeated-item, load 1",
    "dandi_001187_000673_content_axis_battery": "Sternberg novel-picture, load 1 (of 1 vs 3 manipulation)",
    "dandi_000574": "Boran verbal Sternberg (set_size 4/6/8)",
    "ds004752": "Verbal Sternberg maintenance window",
    "ds005489_openloop": "RAM free recall, math-distractor retention interval (not WM maintenance)",
    "ds005557_closedloop": "RAM free recall, math-distractor retention interval (not WM maintenance)",
}
DATASET_MODALITY = {
    "dandi_000469": "single_unit", "dandi_001187_000673_content_axis_battery": "single_unit",
    "dandi_000574": "single_unit",
    "ds004752": "depth_lfp_high_gamma", "ds005489_openloop": "depth_lfp_high_gamma",
    "ds005557_closedloop": "depth_lfp_high_gamma",
}
NOT_ELIGIBLE_DATASETS = {
    "panichello_2024": "not_staged -- source .mat carries no area field",
    "watters_2026": "single_structure_by_design",
    "inagaki_alm5": "single_structure_by_design (ALM only)",
    "macaque_pfc_microstimulation": "single_structure_by_design (calibration target only)",
    "pfc3": "single_structure_by_design (PFC only)",
    "haslacher_clam_tacs": "no_anatomy_available (scalp EEG/tACS)",
    "wolff_eeg_impulse": "no_anatomy_available (scalp EEG)",
    "alagapan_phase_stimulation": "coordinates_only, not region-stratified",
    "kai_miller_nback": "coordinates_only, no atlas lookup performed",
}


def bootstrap_summary(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(finite) < 2:
        return {"status": "non_identified", "n_patients": int(len(finite)),
                "reason": f"fewer than 2 finite patient estimates ({len(finite)})"}
    _, lo, hi = bootstrap_ci(finite, np.mean, n_boot=5000, rng=rng)
    return {
        "status": "estimable", "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "patient_bootstrap_ci95": [float(lo), float(hi)], "n_patients": int(len(finite)),
    }


def collect_lambda_by_dataset(region: str, artifacts: dict[str, dict]) -> dict[str, dict[str, float]]:
    """dataset_key -> {patient: lambda}, only for datasets that carry this region."""
    out: dict[str, dict[str, float]] = {}
    art_469, art_1187, art_574, art_lfp = (
        artifacts["dandi_000469"], artifacts["dandi_001187_000673"], artifacts["dandi_000574"], artifacts["lfp"],
    )
    if region in art_469.get("regions", {}):
        lam, _ = extract_000469(art_469, region)
        if lam:
            out["dandi_000469"] = lam
    if region in art_1187.get("content_axis_battery", {}).get("regions", {}):
        lam, _ = extract_001187_000673(art_1187, region)
        if lam:
            out["dandi_001187_000673_content_axis_battery"] = lam
    if region in art_574.get("regions", {}):
        lam, _ = extract_000574(art_574, region)
        if lam:
            out["dandi_000574"] = lam
    for dataset_key, out_key in (
        ("ds004752", "ds004752"), ("ds005489_openloop", "ds005489_openloop"),
        ("ds005557_closedloop", "ds005557_closedloop"),
    ):
        if region in art_lfp.get("datasets", {}).get(dataset_key, {}).get("structures", {}):
            lam, _ = extract_lfp(art_lfp, dataset_key, region)
            if lam:
                out[out_key] = lam
    return out


def heterogeneity_statistic(per_dataset_means: dict[str, float | None], per_dataset_n: dict[str, int]) -> dict[str, Any]:
    """Unweighted-vs-weighted spread check, not a formal I^2 (too few strata for that to be
    meaningful) -- reports whether per-dataset point estimates disagree, flagged qualitatively."""
    finite = {k: v for k, v in per_dataset_means.items() if v is not None and np.isfinite(v)}
    if len(finite) < 2:
        return {"status": "not_applicable", "reason": "fewer than 2 datasets contribute a finite estimate"}
    values = list(finite.values())
    return {
        "status": "reported", "per_dataset_mean": finite, "per_dataset_n": {k: per_dataset_n.get(k) for k in finite},
        "max_minus_min": float(max(values) - min(values)),
        "note": "qualitative spread only; not a formal between-study heterogeneity test at this n_datasets",
    }


def rank_estimable_structures(
    structures: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict], list[str], dict[str, int]]:
    """An ordering key must be built only from structures whose pooled lambda is itself
    `estimable`, never from a fallback sentinel (e.g. +inf for "no estimate") that could
    out-rank a real estimate and silently put a non-identified structure first."""
    estimable = {
        r: s for r, s in structures.items()
        if s["status"] == "identified" and s["confinement_rate_lambda_pooled_across_datasets"]["status"] == "estimable"
    }
    ranked = sorted(
        estimable, key=lambda r: estimable[r]["confinement_rate_lambda_pooled_across_datasets"]["mean"], reverse=True,
    )
    non_identified = {
        r: (structures[r].get("n_patients_total", 0) if structures[r]["status"] != "identified"
            else structures[r]["confinement_rate_lambda_pooled_across_datasets"].get("n_patients", 0))
        for r in structures if r not in estimable
    }
    return estimable, ranked, non_identified


def build_structure(region: str, artifacts: dict[str, dict], census_matrix: dict, rng: np.random.Generator) -> dict[str, Any]:
    by_dataset = collect_lambda_by_dataset(region, artifacts)
    n_by_dataset = {k: len(v) for k, v in by_dataset.items()}
    n_total = sum(n_by_dataset.values())
    all_values = [v for lam in by_dataset.values() for v in lam.values()]
    pooled_lambda = bootstrap_summary(all_values, rng)
    per_dataset_component = {k: bootstrap_summary(list(v.values()), rng) for k, v in by_dataset.items()}

    single_unit_datasets = [k for k in by_dataset if DATASET_MODALITY[k] == "single_unit"]
    lfp_datasets = [k for k in by_dataset if DATASET_MODALITY[k] != "single_unit"]
    single_unit_values = [v for k in single_unit_datasets for v in by_dataset[k].values()]
    pooled_lambda_single_unit_only = bootstrap_summary(single_unit_values, rng)
    modality_caveat = (
        (
            f"pooled_across_datasets mixes single-unit ({', '.join(single_unit_datasets) or 'none'}) and "
            f"depth-LFP-high-gamma ({', '.join(lfp_datasets)}) confinement-rate estimates, which are "
            "different observables (spike-count state-space fits versus high-gamma-power state-space "
            "fits) sharing a lambda unit by convention, not by construction; "
            "confinement_rate_lambda_pooled_single_unit_only isolates the single-unit-only estimate."
        )
        if single_unit_datasets and lfp_datasets else None
    )

    census_present_key_map = {
        "dandi_000469": ["dandi_000469"], "dandi_001187_000673_content_axis_battery": ["dandi_001187", "dandi_000673"],
        "dandi_000574": ["dandi_000574"], "ds004752": ["ds004752"],
        "ds005489_openloop": ["ram_ds005489_openloop"], "ds005557_closedloop": ["ram_ds005557_closedloop"],
    }
    denominators = {}
    for dataset_key in DATASET_TASK:
        present = sum(
            census_matrix.get(region, {}).get(k, {}).get("n_patients", 0)
            for k in census_present_key_map[dataset_key]
        )
        identified = n_by_dataset.get(dataset_key, 0)
        denominators[dataset_key] = {
            "patients_present_per_census": present,
            "patients_with_identified_lambda": identified,
            "identified_fraction": (identified / present) if present else None,
        }

    # Gate on the pooled bootstrap's OWN finite count, not the raw n_total collected from
    # by_dataset -- these can disagree (a value collected as "identified" upstream can still be
    # dropped as non-finite by bootstrap_summary's own np.isfinite filter), and gating on the raw
    # count previously let two single-dataset non-identified singletons pool into a reported
    # "estimable" entry that entered the cross-structure ordering below this registry's own
    # declared threshold.
    pooled_n = pooled_lambda.get("n_patients", 0) if pooled_lambda.get("status") == "estimable" else 0
    if pooled_n < MIN_STRUCTURE_PATIENTS:
        return {
            "status": "non_identified",
            "reason": (
                f"only {pooled_n} patients survive the pooled bootstrap's finite-value filter "
                f"(<{MIN_STRUCTURE_PATIENTS}); raw n_total collected across datasets was {n_total}"
            ),
            "n_patients_total": n_total, "n_patients_pooled_finite": pooled_n,
            "n_patients_by_dataset": n_by_dataset, "denominators": denominators,
        }
    return {
        "status": "identified",
        "n_patients_total": n_total, "n_patients_by_dataset": n_by_dataset,
        "denominators": denominators,
        "tasks": {k: DATASET_TASK[k] for k in by_dataset},
        "recording_modality_by_dataset": {k: DATASET_MODALITY[k] for k in by_dataset},
        "species": "human (every eligible dataset is human intracranial; not a stratum here)",
        "confinement_rate_lambda_pooled_across_datasets": pooled_lambda,
        "confinement_rate_lambda_pooled_single_unit_only": pooled_lambda_single_unit_only,
        "modality_caveat": modality_caveat,
        "confinement_rate_lambda_per_dataset": per_dataset_component,
        "heterogeneity": heterogeneity_statistic(
            {k: v.get("mean") for k, v in per_dataset_component.items()}, n_by_dataset,
        ),
        "participation_ratio": "not_estimable",
        "control_cost": "see results/structure_control_observables.json",
    }


def main() -> None:
    census_path = RESULTS / "anatomical_census.json"
    path_469 = RESULTS / "region_stratified_drift_000469.json"
    path_1187 = RESULTS / "region_stratified_drift_001187_000673.json"
    path_574 = RESULTS / "region_stratified_drift_000574.json"
    path_lfp = RESULTS / "lfp_structure_dynamics.json"
    path_identity = RESULTS / "patient_identity_audit.json"
    required = (census_path, path_469, path_1187, path_574, path_lfp, path_identity)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required upstream artifacts: {[str(p) for p in missing]}")

    census = json.loads(census_path.read_text())
    census_matrix = census["structure_by_dataset_matrix"]
    artifacts = {
        "dandi_000469": json.loads(path_469.read_text()),
        "dandi_001187_000673": json.loads(path_1187.read_text()),
        "dandi_000574": json.loads(path_574.read_text()),
        "lfp": json.loads(path_lfp.read_text()),
    }
    identity = json.loads(path_identity.read_text())
    if identity.get("cross_release_000469_overlap_found", True):
        raise SystemExit(
            "results/patient_identity_audit.json reports a 000469 cross-release overlap; pooling "
            "000469 patients with 001187/000673 patients would risk double-counting and is refused "
            "until that is resolved."
        )

    all_regions_with_pooled = (
        set(artifacts["dandi_000469"].get("regions", {}))
        | set(artifacts["dandi_001187_000673"].get("content_axis_battery", {}).get("regions", {}))
        | set(artifacts["dandi_000574"].get("regions", {}))
        | {r for ds in artifacts["lfp"].get("datasets", {}).values() for r in ds.get("structures", {})}
    )
    # Python's `-` binds tighter than `|`, so "A | B | C | D - {'pooled'}" only ever removed
    # "pooled" from D -- it leaked back in from A/B/C's own "pooled" region keys and entered the
    # ordering as if it were a peer anatomical structure (the same defect
    # fix_lambda_ordering_underpowered_leakage.py already fixed once for a different artifact).
    NON_ANATOMICAL_LABELS = {"unlabelled", "unspecific"}
    all_regions = sorted(all_regions_with_pooled - {"pooled"} - NON_ANATOMICAL_LABELS)
    non_anatomical_present = sorted(all_regions_with_pooled & NON_ANATOMICAL_LABELS)

    rng = np.random.default_rng(20260806)
    structures = {region: build_structure(region, artifacts, census_matrix, rng) for region in all_regions}
    estimable, ranked, non_identified = rank_estimable_structures(structures)
    # "unlabelled"/"unspecific" are QC/coverage bookkeeping values, not anatomical objects, and must
    # never appear in an anatomical ordering -- but they are not deleted either (zero-drop): their
    # own values are computed and retained under a separate, clearly-non-anatomical key.
    labels_excluded_as_non_anatomical = {
        region: build_structure(region, artifacts, census_matrix, rng) for region in non_anatomical_present
    }

    registry = {
        "schema_version": "1.0.0", "analysis_id": "structure_registry",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "minimum_patient_count_threshold": MIN_STRUCTURE_PATIENTS,
        "source_artifacts": {
            "anatomical_census": str(census_path.relative_to(ROOT)),
            "dandi_000469": str(path_469.relative_to(ROOT)),
            "dandi_001187_000673": str(path_1187.relative_to(ROOT)),
            "dandi_000574": str(path_574.relative_to(ROOT)),
            "lfp_structure_dynamics": str(path_lfp.relative_to(ROOT)),
            "identity_audit": str(path_identity.relative_to(ROOT)),
        },
        "structures": structures,
        "across_structure_lambda_ordering_fastest_to_slowest_estimable_only": ranked,
        "non_identified_structures_n": non_identified,
        "labels_excluded_as_non_anatomical": labels_excluded_as_non_anatomical,
        "datasets_not_eligible": NOT_ELIGIBLE_DATASETS,
        "note_both_grains_reported": (
            "This artifact is the pooled-at-scale structure grain, built over the full corpus "
            "the census admits: DANDI 000469, DANDI 001187/000673, DANDI 000574, ds004752, "
            "ds005489, ds005557. The within-patient paired contrast per dataset lives in "
            "results/structure_paired_contrasts.json and is not reproduced here -- the two "
            "answer different questions and neither replaces the other."
        ),
    }
    dynamics = {
        "schema_version": "1.0.0", "analysis_id": "structure_pooled_dynamics",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "structures": {r: s for r, s in structures.items()},
    }
    (RESULTS / "structure_registry.json").write_text(canonical_json(registry))
    (RESULTS / "structure_pooled_dynamics.json").write_text(canonical_json(dynamics))
    print(json.dumps({
        "structures_identified": list(estimable),
        "structures_non_identified": list(non_identified),
        "lambda_ordering_estimable_only": ranked,
    }, indent=2))


if __name__ == "__main__":
    main()
