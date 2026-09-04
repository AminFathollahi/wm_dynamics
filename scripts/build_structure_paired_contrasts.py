#!/usr/bin/env python3
"""Within-patient paired structure contrast, for every eligible dataset.

results/structure_registry.json's lambda ordering is built from unpaired, unevenly-weighted
samples: pre-SMA's estimate is 000469-only, amygdala's is 001187/000673-only. Where two
structures are co-recorded in the same patient, the correct test is the within-patient paired
difference. This script computes that difference for lambda, diffusion D, and stationary
variance D/lambda, for every ordered structure pair co-recorded in a patient, separately in every
dataset whose region-stratified lambda/D fit already exists -- it does not refit anything.

Eligible datasets (region-stratified lambda/D already fit):
    DANDI 000469, DANDI 001187+000673 (content_axis_battery), DANDI 000574,
    ds004752, ds005489 (RAM open-loop), ds005557 (RAM closed-loop).
Every other dataset in config/datasets.json is single-structure-by-design or has no anatomy
(results/anatomical_census.json anatomy_status) and is listed with that status, not silently
dropped.

This is the within-dataset paired grain. The pooled-across-datasets structure grain, with
dataset/species/task as explicit strata, is the separate question
scripts/build_structure_registry.py answers -- neither replaces the other.

Run (after every region-stratified/LFP structure artifact below exists):
    conda run -n wm_dynamics python scripts/build_structure_paired_contrasts.py
"""

from __future__ import annotations

import json
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import bootstrap_ci, paired_sign_flip_test  # noqa: E402
from spike_pipeline import load_spike_times, low_rate_unit_mask, resolve_unit_regions  # noqa: E402
from build_structure_control_observables import _identifiable_session_field  # noqa: E402
import run_human_drift_spine_000469 as spine469  # noqa: E402

RESULTS = ROOT / "results"
SEED = 20260806
MIN_PATIENTS_BOTH_STRUCTURES = 3
HEMISPHERE_SENSITIVITY_WINDOW_S = 2.3
HEMISPHERE_SENSITIVITY_MIN_UNITS = 4

# Every dataset in the census gets a status here, even the ones this script cannot pair -- so
# "computed on N of 13 datasets" is never silently reported as "every dataset".
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


def _session_to_patient_identity(key: str) -> str:
    return key


def _session_to_patient_by_ses_suffix(key: str) -> str:
    return key.split("_ses-")[0]


def per_patient_lambda_diffusion_from_sessions(
    sessions: dict, session_to_patient: Callable[[str], str],
) -> tuple[dict[str, float], dict[str, float]]:
    """One (lambda, diffusion) per patient: mean over a session's identifiable folds
    (the same filter results/structure_control_observables.json uses), then mean over a
    patient's sessions for corpora with more than one session per patient.
    Handles both a plain ``sessions[key]["folds"]`` row (000469, 000574) and the
    ``sessions[key]["content_axis_fit"]["folds"]`` row 001187/000673's content-axis battery uses."""
    session_lambda: dict[str, float] = {}
    session_diffusion: dict[str, float] = {}
    for key, row in sessions.items():
        if "folds" in row:
            if row.get("status") != "complete":
                continue
            folds = row["folds"]
        else:
            fit = row.get("content_axis_fit", {})
            if fit.get("status") != "complete":
                continue
            folds = fit.get("folds", [])
        lam = _identifiable_session_field(folds, "lambda_rate")
        dif = _identifiable_session_field(folds, "diffusion")
        if lam is not None:
            session_lambda[key] = lam
        if dif is not None:
            session_diffusion[key] = dif
    patient_lambda: dict[str, list[float]] = {}
    patient_diffusion: dict[str, list[float]] = {}
    for key, value in session_lambda.items():
        patient_lambda.setdefault(session_to_patient(key), []).append(value)
    for key, value in session_diffusion.items():
        patient_diffusion.setdefault(session_to_patient(key), []).append(value)
    return (
        {p: float(np.mean(v)) for p, v in patient_lambda.items()},
        {p: float(np.mean(v)) for p, v in patient_diffusion.items()},
    )


def extract_000469(artifact: dict, region: str) -> tuple[dict[str, float], dict[str, float]]:
    sessions = artifact.get("regions", {}).get(region, {}).get("sessions", {})
    return per_patient_lambda_diffusion_from_sessions(sessions, _session_to_patient_identity)


def extract_000574(artifact: dict, region: str) -> tuple[dict[str, float], dict[str, float]]:
    sessions = artifact.get("regions", {}).get(region, {}).get("sessions", {})
    return per_patient_lambda_diffusion_from_sessions(sessions, _session_to_patient_by_ses_suffix)


def extract_001187_000673(artifact: dict, region: str) -> tuple[dict[str, float], dict[str, float]]:
    sessions = artifact.get("content_axis_battery", {}).get("regions", {}).get(region, {}).get("sessions", {})
    return per_patient_lambda_diffusion_from_sessions(sessions, _session_to_patient_by_ses_suffix)


def extract_lfp(artifact: dict, dataset_key: str, region: str) -> tuple[dict[str, float], dict[str, float]]:
    block = artifact.get("datasets", {}).get(dataset_key, {}).get("structures", {}).get(region, {})
    return dict(block.get("per_patient_lambda", {})), dict(block.get("per_patient_diffusion", {}))


def bootstrap_paired_difference(diffs: np.ndarray, patient_ids: list[str], rng: np.random.Generator) -> dict[str, Any]:
    if len(diffs) < MIN_PATIENTS_BOTH_STRUCTURES:
        return {
            "status": "non_identified", "n_patients_both_structures": int(len(diffs)),
            "patient_ids": list(patient_ids),
            "reason": f"fewer than {MIN_PATIENTS_BOTH_STRUCTURES} patients have an identified estimate in both structures",
        }
    _, lo, hi = bootstrap_ci(diffs, np.mean, n_boot=5000, rng=rng)
    sign_flip = paired_sign_flip_test(diffs, np.zeros_like(diffs), n_perm=10000, alternative="two-sided", rng=rng)
    return {
        "status": "estimable", "n_patients_both_structures": int(len(diffs)), "patient_ids": list(patient_ids),
        "mean_difference": float(np.mean(diffs)), "median_difference": float(np.median(diffs)),
        "patient_bootstrap_ci95": [float(lo), float(hi)],
        "sign_flip_p_value": sign_flip["p_value"],
        "a_greater_than_b": bool(lo > 0), "b_greater_than_a": bool(hi < 0),
    }


def paired_contrast_one_pair(
    region_a: str, region_b: str,
    lambda_a: dict[str, float], diffusion_a: dict[str, float],
    lambda_b: dict[str, float], diffusion_b: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    shared_lambda = sorted(set(lambda_a) & set(lambda_b))
    lambda_diffs = np.array([lambda_a[p] - lambda_b[p] for p in shared_lambda], dtype=float)

    shared_diffusion = sorted(set(diffusion_a) & set(diffusion_b))
    diffusion_diffs = np.array([diffusion_a[p] - diffusion_b[p] for p in shared_diffusion], dtype=float)

    shared_variance = sorted(
        p for p in shared_lambda if p in diffusion_a and p in diffusion_b
        and lambda_a[p] > 0 and lambda_b[p] > 0
    )
    variance_diffs = np.array(
        [(diffusion_a[p] / lambda_a[p]) - (diffusion_b[p] / lambda_b[p]) for p in shared_variance],
        dtype=float,
    )

    return {
        "region_a": region_a, "region_b": region_b,
        "lambda_s_per_s": bootstrap_paired_difference(lambda_diffs, shared_lambda, rng),
        "diffusion_state_units_sq_per_s": bootstrap_paired_difference(diffusion_diffs, shared_diffusion, rng),
        "stationary_variance_D_over_lambda": bootstrap_paired_difference(variance_diffs, shared_variance, rng),
    }


def denominator_table(
    dataset_label: str, structure_dataset_matrix: dict[str, dict[str, Any]],
    census_dataset_keys: list[str], per_region_lambda: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Patients present (census), patients with an identified paired-contrast lambda estimate,
    and the identified fraction -- for every region this dataset contributes."""
    table = {}
    for region, lambda_by_patient in per_region_lambda.items():
        present = sum(
            structure_dataset_matrix.get(region, {}).get(k, {}).get("n_patients", 0)
            for k in census_dataset_keys
        )
        identified = len(lambda_by_patient)
        table[region] = {
            "patients_present_per_census": present,
            "patients_with_identified_lambda": identified,
            "identified_fraction": (identified / present) if present else None,
        }
    return table


def build_dataset_block(
    dataset_label: str, regions_present: list[str],
    extract: Callable[[str], tuple[dict[str, float], dict[str, float]]],
    structure_dataset_matrix: dict[str, dict[str, Any]], census_dataset_keys: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    per_region_lambda: dict[str, dict[str, float]] = {}
    per_region_diffusion: dict[str, dict[str, float]] = {}
    for region in regions_present:
        lam, dif = extract(region)
        per_region_lambda[region] = lam
        per_region_diffusion[region] = dif

    pair_matrix = []
    for region_a, region_b in permutations(regions_present, 2):
        pair_matrix.append(paired_contrast_one_pair(
            region_a, region_b,
            per_region_lambda[region_a], per_region_diffusion[region_a],
            per_region_lambda[region_b], per_region_diffusion[region_b],
            rng,
        ))

    return {
        "status": "estimable" if regions_present else "non_identified",
        "regions_present": regions_present,
        "denominators": denominator_table(
            dataset_label, structure_dataset_matrix, census_dataset_keys, per_region_lambda,
        ),
        "pair_matrix": pair_matrix,
    }


def unpaired_vs_paired_pre_sma_vs_hippocampus(registry: dict, pair_matrix_000469: list[dict]) -> dict:
    """The unpaired difference of pooled means beside the paired within-patient difference
    for the deciding contrast, so a sign reversal (or agreement) between the two forms is
    visible rather than only ever reporting one of them."""
    structures = registry.get("structures", {})
    pre_sma = structures.get("pre_sma", {}).get("confinement_rate_lambda_pooled_across_datasets", {})
    hippocampus = structures.get("hippocampus", {}).get("confinement_rate_lambda_pooled_across_datasets", {})
    if pre_sma.get("status") != "estimable" or hippocampus.get("status") != "estimable":
        return {
            "status": "non_identified",
            "reason": "pooled pre_sma or hippocampus lambda is not estimable in the (fixed) structure registry",
        }
    unpaired_diff = pre_sma["mean"] - hippocampus["mean"]

    paired = next(
        (p for p in pair_matrix_000469 if p["region_a"] == "pre_sma" and p["region_b"] == "hippocampus"),
        None,
    )
    paired_lambda = paired["lambda_s_per_s"] if paired else {"status": "non_identified", "reason": "pair not found"}
    paired_mean = paired_lambda.get("mean_difference")

    sign_reversal = None
    if paired_lambda.get("status") == "estimable" and paired_mean is not None:
        sign_reversal = bool(np.sign(unpaired_diff) != np.sign(paired_mean) and unpaired_diff != 0 and paired_mean != 0)

    return {
        "status": "estimable",
        "unpaired_pooled_mean_difference_pre_sma_minus_hippocampus": float(unpaired_diff),
        "unpaired_pre_sma_pooled_mean": pre_sma["mean"], "unpaired_hippocampus_pooled_mean": hippocampus["mean"],
        "paired_within_patient_difference_000469": paired_lambda,
        "sign_reversal_paired_vs_unpaired": sign_reversal,
        "note": (
            "if sign_reversal_paired_vs_unpaired is true, that is the finding, reported as such, "
            "not suppressed."
        ),
    }


def _hemisphere_unit_indices(path: Path, region: str, window_s: float) -> dict[str, np.ndarray]:
    """Index arrays, in the SAME position space run_human_drift_spine_000469.analyze_session's
    `unit_indices` argument expects (post region-filter, post firing-rate QC), split by
    hemisphere."""
    with h5py.File(path, "r") as handle:
        spike_lists = load_spike_times(handle)
        resolved = resolve_unit_regions(handle)
        onsets = handle["intervals/trials"]["timestamps_Maintenance"][:]
    region_mask = resolved["region"] == region
    region_spike_lists = [s for s, keep in zip(spike_lists, region_mask) if keep]
    region_hemisphere = resolved["hemisphere"][region_mask]
    rate_mask = low_rate_unit_mask(region_spike_lists, onsets, window_s)
    post_qc_hemisphere = region_hemisphere[rate_mask]
    return {
        "left": np.where(post_qc_hemisphere == "left")[0],
        "right": np.where(post_qc_hemisphere == "right")[0],
    }


def hemisphere_sensitivity_pre_sma_vs_hippocampus_000469(rng: np.random.Generator) -> dict:
    """Mandatory sensitivity, declared reduced scope: the deciding pre-SMA-vs-hippocampus pair
    only, in DANDI 000469 only (the corpus both structures are co-recorded in). A hemisphere-
    split refit costs about as much as the matched-draw refits elsewhere in this project; the full
    pair grid across all three corpora would require refitting machinery that two of the three
    datasets' session-fit functions don't have, so this scope is declared before running rather
    than silently narrowed."""
    artifact = json.loads((RESULTS / "region_stratified_drift_000469.json").read_text())
    hipp_complete = {p for p, r in artifact["regions"]["hippocampus"]["sessions"].items() if r.get("status") == "complete"}
    sma_complete = {p for p, r in artifact["regions"]["pre_sma"]["sessions"].items() if r.get("status") == "complete"}
    patients = sorted(hipp_complete & sma_complete)

    per_hemisphere_lambda: dict[str, dict[str, dict[str, float]]] = {
        "left": {"pre_sma": {}, "hippocampus": {}}, "right": {"pre_sma": {}, "hippocampus": {}},
    }
    n_refits_attempted = 0
    t0 = time.time()
    for patient in patients:
        path = spine469.data_directory() / patient / f"{patient}_ses-2_ecephys+image.nwb"
        if not path.is_file():
            continue
        for region in ("pre_sma", "hippocampus"):
            hemisphere_indices = _hemisphere_unit_indices(path, region, HEMISPHERE_SENSITIVITY_WINDOW_S)
            for hemisphere, indices in hemisphere_indices.items():
                if len(indices) < HEMISPHERE_SENSITIVITY_MIN_UNITS:
                    continue
                n_refits_attempted += 1
                fit = spine469.analyze_session(
                    path, seed=SEED + n_refits_attempted, region=region, unit_indices=indices,
                )
                if fit.get("status") == "complete":
                    lam = fit["summary"].get("state_space_lambda_identified_mean")
                    if lam is not None:
                        per_hemisphere_lambda[hemisphere][region][patient] = lam
    wall_clock_s = time.time() - t0

    by_hemisphere = {}
    for hemisphere, per_region in per_hemisphere_lambda.items():
        shared = sorted(set(per_region["pre_sma"]) & set(per_region["hippocampus"]))
        diffs = np.array([per_region["pre_sma"][p] - per_region["hippocampus"][p] for p in shared])
        by_hemisphere[hemisphere] = bootstrap_paired_difference(diffs, shared, rng)

    return {
        "status": "estimable",
        "declared_reduced_scope": (
            "pre-SMA-vs-hippocampus pair only, DANDI 000469 only -- declared before running, "
            "since a full pair grid across every corpus is not currently affordable."
        ),
        "n_candidate_patients": len(patients), "n_refits_attempted": n_refits_attempted,
        "measured_wall_clock_s": wall_clock_s,
        "min_units_per_hemisphere": HEMISPHERE_SENSITIVITY_MIN_UNITS,
        "paired_difference_by_hemisphere": by_hemisphere,
    }


def main() -> None:
    census_path = RESULTS / "anatomical_census.json"
    path_469 = RESULTS / "region_stratified_drift_000469.json"
    path_1187 = RESULTS / "region_stratified_drift_001187_000673.json"
    path_574 = RESULTS / "region_stratified_drift_000574.json"
    path_lfp = RESULTS / "lfp_structure_dynamics.json"
    path_registry = RESULTS / "structure_registry.json"
    missing = [p for p in (census_path, path_469, path_1187, path_574, path_lfp, path_registry) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required upstream artifacts: {[str(p) for p in missing]}")

    census = json.loads(census_path.read_text())
    matrix = census["structure_by_dataset_matrix"]
    art_469 = json.loads(path_469.read_text())
    art_1187 = json.loads(path_1187.read_text())
    art_574 = json.loads(path_574.read_text())
    art_lfp = json.loads(path_lfp.read_text())

    rng = np.random.default_rng(SEED)

    regions_469 = sorted(r for r in art_469.get("regions", {}) if r != "pooled")
    regions_1187 = sorted(r for r in art_1187.get("content_axis_battery", {}).get("regions", {}) if r != "pooled")
    regions_574 = sorted(r for r in art_574.get("regions", {}) if r != "pooled")

    datasets_out = {
        "dandi_000469": build_dataset_block(
            "dandi_000469", regions_469, lambda r: extract_000469(art_469, r),
            matrix, ["dandi_000469"], rng,
        ),
        "dandi_001187_000673_content_axis_battery": build_dataset_block(
            "dandi_001187_000673", regions_1187, lambda r: extract_001187_000673(art_1187, r),
            matrix, ["dandi_001187", "dandi_000673"], rng,
        ),
        "dandi_000574": build_dataset_block(
            "dandi_000574", regions_574, lambda r: extract_000574(art_574, r),
            matrix, ["dandi_000574"], rng,
        ),
    }
    for dataset_key, census_key in (
        ("ds004752", "ds004752"),
        ("ds005489_openloop", "ram_ds005489_openloop"),
        ("ds005557_closedloop", "ram_ds005557_closedloop"),
    ):
        lfp_block = art_lfp.get("datasets", {}).get(dataset_key, {})
        regions_lfp = sorted(lfp_block.get("structures", {}))
        datasets_out[dataset_key] = build_dataset_block(
            dataset_key, regions_lfp, lambda r, dk=dataset_key: extract_lfp(art_lfp, dk, r),
            matrix, [census_key], rng,
        )

    registry = json.loads(path_registry.read_text())
    unpaired_vs_paired = unpaired_vs_paired_pre_sma_vs_hippocampus(
        registry, datasets_out["dandi_000469"]["pair_matrix"],
    )
    hemisphere_sensitivity = hemisphere_sensitivity_pre_sma_vs_hippocampus_000469(rng)

    output = {
        "schema_version": "1.0.0", "analysis_id": "structure_paired_contrasts",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "seed": SEED, "minimum_patients_both_structures": MIN_PATIENTS_BOTH_STRUCTURES,
        "method_note": (
            "Within-patient paired difference (region_a minus region_b), restricted to patients "
            "with an identified estimate in BOTH regions of the pair, patient-level bootstrap "
            "(n_boot=5000). Reported for lambda, diffusion D, and the derived stationary variance "
            "D/lambda -- computed per patient from that patient's own (D, lambda) pair, not from "
            "dataset-level D and lambda separately, to avoid the ratio-of-means bias a pooled "
            "division would introduce. This is the within-dataset paired grain; the "
            "pooled-across-datasets structure grain is results/structure_registry.json -- "
            "neither replaces the other."
        ),
        "source_artifacts": {
            "anatomical_census": str(census_path.relative_to(ROOT)),
            "dandi_000469": str(path_469.relative_to(ROOT)),
            "dandi_001187_000673": str(path_1187.relative_to(ROOT)),
            "dandi_000574": str(path_574.relative_to(ROOT)),
            "lfp_structure_dynamics": str(path_lfp.relative_to(ROOT)),
        },
        "datasets": datasets_out,
        "datasets_not_eligible": NOT_ELIGIBLE_DATASETS,
        "unpaired_vs_paired_pre_sma_vs_hippocampus": unpaired_vs_paired,
        "hemisphere_sensitivity_pre_sma_vs_hippocampus_000469": hemisphere_sensitivity,
    }
    (RESULTS / "structure_paired_contrasts.json").write_text(canonical_json(output))
    print(json.dumps({
        "datasets_eligible": {k: v["status"] for k, v in datasets_out.items()},
        "n_pairs_per_dataset": {k: len(v["pair_matrix"]) for k, v in datasets_out.items()},
        "unpaired_vs_paired_status": unpaired_vs_paired.get("status"),
        "sign_reversal": unpaired_vs_paired.get("sign_reversal_paired_vs_unpaired"),
        "hemisphere_sensitivity_wall_clock_s": hemisphere_sensitivity.get("measured_wall_clock_s"),
    }, indent=2))


if __name__ == "__main__":
    main()
