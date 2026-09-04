#!/usr/bin/env python3
"""Band-free geometry: intrinsic dimensionality of the population point cloud,
free of any confinement-rate identifiability gate.

Four independent estimators (twonn, Levina-Bickel MLE, correlation
dimension, participation ratio) on the same (dataset, structure, session,
epoch) population point cloud. Estimator agreement -- not any single
estimator's number -- is the result: dimensionality estimators have
different failure modes, so agreement across them is the criterion.

Epochs (baseline, encoding, delay, probe) per corpus, all anchored to the
NWB trial-timing fields each corpus's existing pipeline already uses:
  - DANDI 000469 / 001187 (canonical_sessions() dedup of 001187/000673):
    timestamps_FixationCross, timestamps_Encoding1, timestamps_Maintenance,
    timestamps_Probe are all present in both releases' trial tables
    (verified directly against the NWB files, not assumed from 000469 alone).
  - DANDI 000574 (Boran): no named per-epoch timestamp fields exist in its
    trial table; run_human_drift_spine_000574.py's own docstring documents
    the task's fixed relative structure (fixation [-6,-5] s, encoding
    [-5,-3] s, maintenance [-3,0] s relative to the probe, i.e. maintenance
    onset = trial start_time + 3.0 s) -- reused here to derive all four
    epoch onsets from start_time.
  - DANDI 000004 (Chandravadia) is EXCLUDED from this section: its
    electrode `location` field uses a region-label convention
    ("Right Hippocampus") this project has no parser for yet, and it is not
    yet registered in config/datasets.json. Not staged for lack of a real
    delay epoch (it has one) -- see `excluded_corpora` in the output
    artifact for the exact reason and src/corpus_sessions.py for detail.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_sessions import data_root, iter_all_corpora  # noqa: E402
from geometry import (  # noqa: E402
    correlation_dimension,
    levina_bickel_mle_dimension,
    participation_ratio,
    twonn_dimension,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform, build_psth  # noqa: E402
from statistics import fdr_bh, paired_sign_flip_test, spearman_permutation_test, stable_seed  # noqa: E402

SEED = 20260808
BIN_MS = 100
ESTIMATORS = ("twonn", "levina_bickel_mle", "correlation_dimension", "participation_ratio")
N_BOOT = 30
MAX_POINTS_FOR_DISTANCE_ESTIMATORS = 300  # twonn/MLE/correlation_dimension are O(n^2) in pairwise distances
N_CALIBRATION_MATCHED_DRAWS = 20
UNIT_COUNT_LEVELS_FRACTIONS = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
DECIDING_CONTRAST = ("hippocampus", "pre_sma")


def data_root() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    return Path(root)


def compute_estimators(cloud: np.ndarray, rng: np.random.Generator | None = None) -> dict:
    """twonn/MLE/correlation_dimension are all O(n^2) in pairwise distances, so a
    dense point cloud is subsampled to a fixed cap before computing them;
    participation_ratio (a covariance-spectrum statistic, not distance-based)
    always uses the full cloud."""
    _, s, _ = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
    distance_cloud = cloud
    if len(cloud) > MAX_POINTS_FOR_DISTANCE_ESTIMATORS:
        sampler = rng if rng is not None else np.random.default_rng(0)
        idx = sampler.choice(len(cloud), size=MAX_POINTS_FOR_DISTANCE_ESTIMATORS, replace=False)
        distance_cloud = cloud[idx]
    return {
        "twonn": twonn_dimension(distance_cloud),
        "levina_bickel_mle": levina_bickel_mle_dimension(distance_cloud),
        "correlation_dimension": correlation_dimension(distance_cloud),
        "participation_ratio": participation_ratio(s**2),
    }


def bootstrap_estimators(psth: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """psth: (n_trials, n_units, n_bins) firing-rate counts for one epoch."""
    transform = FrozenPSTHTransform().fit(psth)
    standardized = transform.transform(psth).transpose(0, 2, 1)  # (trials, bins, units)
    n_trials = standardized.shape[0]
    cloud = standardized.reshape(-1, standardized.shape[-1])
    point_estimate = compute_estimators(cloud, rng)

    boot_values = {name: [] for name in ESTIMATORS}
    for _ in range(n_boot):
        trial_idx = rng.integers(0, n_trials, size=n_trials)
        boot_cloud = standardized[trial_idx].reshape(-1, standardized.shape[-1])
        estimates = compute_estimators(boot_cloud, rng)
        for name in ESTIMATORS:
            if np.isfinite(estimates[name]):
                boot_values[name].append(estimates[name])
    intervals = {}
    for name in ESTIMATORS:
        values = np.array(boot_values[name])
        if len(values) >= 10:
            intervals[name] = {
                "point_estimate": point_estimate[name],
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
                "n_boot_finite": int(len(values)),
            }
        else:
            intervals[name] = {"point_estimate": point_estimate[name], "ci_lower": None, "ci_upper": None, "n_boot_finite": int(len(values))}
    return intervals


def n_dependence_curve(psth: np.ndarray, rng: np.random.Generator) -> dict:
    n_units = psth.shape[1]
    levels = sorted({max(5, int(round(n_units * frac))) for frac in UNIT_COUNT_LEVELS_FRACTIONS if int(round(n_units * frac)) <= n_units})
    curve = {}
    for level in levels:
        unit_idx = rng.choice(n_units, size=level, replace=False)
        subset = psth[:, unit_idx, :]
        transform = FrozenPSTHTransform().fit(subset)
        standardized = transform.transform(subset).transpose(0, 2, 1)
        cloud = standardized.reshape(-1, standardized.shape[-1])
        curve[str(level)] = compute_estimators(cloud, rng)
    return curve


def epoch_psth(spike_lists: list, onset: np.ndarray, window_s: float) -> np.ndarray:
    return build_psth(spike_lists, onset, bin_ms=BIN_MS, smooth_ms=0, window_s=window_s)


def analyze_unit_pool(spike_lists: list, epoch_onsets: dict, epoch_windows: dict, rng: np.random.Generator) -> dict:
    result = {}
    for epoch, onset in epoch_onsets.items():
        window = epoch_windows[epoch]
        psth = epoch_psth(spike_lists, onset, window)
        if psth.shape[2] < 2:
            result[epoch] = {"status": "excluded", "reason": "fewer than 2 bins in this epoch window"}
            continue
        estimators = bootstrap_estimators(psth, rng)
        result[epoch] = {
            "status": "complete",
            "n_trials": int(psth.shape[0]),
            "n_units": int(psth.shape[1]),
            "n_bins": int(psth.shape[2]),
            "estimators": estimators,
        }
        if epoch == "delay":
            result[epoch]["n_dependence_curve"] = n_dependence_curve(psth, rng)
            result[epoch]["_psth_for_matched_count"] = psth  # stripped before serialization
    return result


def collect_rows(root: Path, rng: np.random.Generator) -> dict:
    rows = {}
    for item in iter_all_corpora(root):
        key = f"{item['dataset']}/{item['session']}/{item['structure']}"
        rows[key] = {
            "dataset": item["dataset"], "patient": item["patient"], "session": item["session"], "structure": item["structure"],
            "epochs": analyze_unit_pool(item["spike_lists"], item["epoch_onsets"], item["epoch_windows"], rng),
        }
    return rows


# ── Estimator agreement, unit-count-matched contrast, predeclared decision ──


def structure_order(rows: dict, dataset_patient_filter=None) -> dict:
    """Per-structure median delay-epoch estimate, per estimator, for ordering agreement."""
    by_structure = {name: {} for name in ESTIMATORS}
    for row in rows.values():
        delay = row["epochs"].get("delay", {})
        if delay.get("status") != "complete":
            continue
        structure = row["structure"]
        for name in ESTIMATORS:
            point = delay["estimators"][name]["point_estimate"]
            if point is not None and np.isfinite(point):
                by_structure[name].setdefault(structure, []).append(point)
    medians = {name: {s: float(np.median(v)) for s, v in vals.items() if len(v) > 0} for name, vals in by_structure.items()}
    return medians


def cross_estimator_agreement(medians: dict, rng: np.random.Generator) -> dict:
    common_structures = set.intersection(*[set(v.keys()) for v in medians.values()]) if all(medians.values()) else set()
    common_structures = sorted(common_structures)
    result = {}
    if len(common_structures) < 3:
        return {"status": "not_estimable", "reason": "fewer than 3 structures with all estimators defined", "n_structures": len(common_structures)}
    pairs = [(a, b) for i, a in enumerate(ESTIMATORS) for b in ESTIMATORS[i + 1:]]
    for a, b in pairs:
        x = np.array([medians[a][s] for s in common_structures])
        y = np.array([medians[b][s] for s in common_structures])
        test = spearman_permutation_test(x, y, n_perm=2000, rng=rng)
        result[f"{a}_vs_{b}"] = {"rho": test["rho"], "p_value": test["p_value"], "n_structures": len(common_structures)}
    return {"status": "complete", "n_structures": len(common_structures), "structures": common_structures, "pairwise": result}


def matched_count_comparison(rows: dict, structure_a: str, structure_b: str, rng: np.random.Generator) -> dict:
    """Within-patient matched unit count, both structures, N_CALIBRATION_MATCHED_DRAWS resamples each."""
    by_patient_structure: dict[tuple[str, str, str], dict] = {}
    for row in rows.values():
        delay = row["epochs"].get("delay", {})
        if delay.get("status") != "complete" or "_psth_for_matched_count" not in delay:
            continue
        key = (row["dataset"], row["patient"], row["structure"])
        by_patient_structure[key] = delay

    paired_rows = []
    patients_seen = set((d, p) for d, p, s in by_patient_structure)
    for dataset, patient in patients_seen:
        key_a = (dataset, patient, structure_a)
        key_b = (dataset, patient, structure_b)
        if key_a not in by_patient_structure or key_b not in by_patient_structure:
            continue
        psth_a = by_patient_structure[key_a]["_psth_for_matched_count"]
        psth_b = by_patient_structure[key_b]["_psth_for_matched_count"]
        n_match = min(psth_a.shape[1], psth_b.shape[1])
        if n_match < 5:
            continue
        draws = {name: {"a": [], "b": []} for name in ESTIMATORS}
        for _ in range(N_CALIBRATION_MATCHED_DRAWS):
            idx_a = rng.choice(psth_a.shape[1], size=n_match, replace=False)
            idx_b = rng.choice(psth_b.shape[1], size=n_match, replace=False)
            est_a = compute_estimators(FrozenPSTHTransform().fit_transform(psth_a[:, idx_a, :]).transpose(0, 2, 1).reshape(-1, n_match), rng)
            est_b = compute_estimators(FrozenPSTHTransform().fit_transform(psth_b[:, idx_b, :]).transpose(0, 2, 1).reshape(-1, n_match), rng)
            for name in ESTIMATORS:
                draws[name]["a"].append(est_a[name])
                draws[name]["b"].append(est_b[name])
        paired_rows.append({
            "dataset": dataset, "patient": patient, "n_matched_units": int(n_match),
            "mean_a": {name: float(np.nanmean(draws[name]["a"])) for name in ESTIMATORS},
            "mean_b": {name: float(np.nanmean(draws[name]["b"])) for name in ESTIMATORS},
        })
    return {
        "structure_a": structure_a, "structure_b": structure_b,
        "n_attempted_patients": len(patients_seen),
        "n_paired_patients": len(paired_rows),
        "fraction_with_a_value": (len(paired_rows) / len(patients_seen)) if patients_seen else None,
        "rows": paired_rows,
    }


def predeclared_decision(matched: dict, rng: np.random.Generator) -> dict:
    structure_a, structure_b = DECIDING_CONTRAST
    n_paired = matched["n_paired_patients"]
    if n_paired < 6:
        return {
            "deciding_contrast": f"{structure_a} minus {structure_b}, delay epoch, matched unit count, within patient",
            "verdict": "underpowered_by_construction",
            "n_paired_patients": n_paired,
            "reason": "fewer than 6 paired patients; minimum attainable sign-flip p exceeds 0.05 by construction",
        }
    signs = {}
    p_values = {}
    for name in ESTIMATORS:
        a = np.array([row["mean_a"][name] for row in matched["rows"]])
        b = np.array([row["mean_b"][name] for row in matched["rows"]])
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() < 6:
            signs[name] = None
            p_values[name] = None
            continue
        test = paired_sign_flip_test(a[finite], b[finite], alternative="two-sided", rng=rng)
        signs[name] = "positive" if test["mean_diff"] > 0 else "negative"
        p_values[name] = {"p": test["p_value"], "mean_diff": test["mean_diff"], "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"]}
    sign_counts = {}
    for name, sign in signs.items():
        if sign is not None:
            sign_counts[sign] = sign_counts.get(sign, 0) + 1
    n_agreeing = max(sign_counts.values()) if sign_counts else 0
    majority_sign = max(sign_counts, key=sign_counts.get) if sign_counts else None

    deciding_estimator = "twonn"
    deciding_p = p_values.get(deciding_estimator, {}).get("p") if p_values.get(deciding_estimator) else None
    if deciding_p is not None and deciding_p < 0.05 and n_agreeing >= 3:
        verdict = "dimensionality_differs_by_structure"
    elif deciding_p is not None and deciding_p >= 0.05 and n_agreeing >= 3:
        verdict = "no_dimensionality_difference"
    else:
        verdict = "estimator_disagreement"
    return {
        "deciding_contrast": f"{structure_a} minus {structure_b}, delay epoch, matched unit count, within patient, twonn as the named deciding estimator",
        "n_paired_patients": n_paired,
        "per_estimator_sign": signs,
        "per_estimator_test": p_values,
        "n_estimators_agreeing_on_sign": n_agreeing,
        "majority_sign": majority_sign,
        "verdict": verdict,
    }


def strip_internal_fields(rows: dict) -> dict:
    for row in rows.values():
        for epoch in row["epochs"].values():
            epoch.pop("_psth_for_matched_count", None)
    return rows


def main() -> None:
    root = data_root()
    rng = np.random.default_rng(SEED)
    rows = collect_rows(root, rng)

    medians = structure_order(rows, None)
    agreement = cross_estimator_agreement(medians, np.random.default_rng(SEED + 1))
    matched = matched_count_comparison(rows, DECIDING_CONTRAST[0], DECIDING_CONTRAST[1], np.random.default_rng(SEED + 2))
    decision = predeclared_decision(matched, np.random.default_rng(SEED + 3))

    all_structures = sorted({s for v in medians.values() for s in v.keys()})
    family_pairs = [(a, b) for i, a in enumerate(all_structures) for b in all_structures[i + 1:]]
    family_results = []
    for a, b in family_pairs:
        m = matched_count_comparison(rows, a, b, np.random.default_rng(stable_seed(f"{a}_{b}")))
        d = predeclared_decision(m, np.random.default_rng(stable_seed(f"{a}_{b}_test")))
        family_results.append({"pair": [a, b], "n_paired_patients": m["n_paired_patients"], "decision": d})
    finite_p = [
        r["decision"]["per_estimator_test"]["twonn"]["p"]
        for r in family_results
        if r["decision"].get("per_estimator_test", {}).get("twonn")
    ]
    if finite_p:
        fdr = fdr_bh(np.array(finite_p))
    else:
        fdr = {"status": "not_estimable"}

    output = {
        "schema_version": "1.0.0",
        "analysis_id": "intrinsic_dimensionality",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "estimators": list(ESTIMATORS),
        "epochs": ["baseline", "encoding", "delay", "probe"],
        "excluded_corpora": {
            "dandi_000004": (
                "not a task-design limitation -- it has a genuine ~2.2s maintenance interval "
                "(delay1_time to delay2_time) after a ~1.0s encoding period. Excluded because its "
                "electrode location field uses a region-label convention (\"Right Hippocampus\") "
                "this project has no parser for yet, and it is not yet registered in "
                "config/datasets.json; both are a scoped follow-up, not yet attempted."
            ),
        },
        "n_dataset_structure_session_rows": len(rows),
        "structure_delay_median_by_estimator": medians,
        "cross_estimator_ordering_agreement": agreement,
        "deciding_contrast_matched_count_comparison": matched,
        "predeclared_decision": decision,
        "family_wise_ordered_pairs": family_results,
        "family_wise_fdr_bh_on_twonn_p": fdr,
        "rows": strip_internal_fields(rows),
    }
    destination = ROOT / "results" / "intrinsic_dimensionality.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "n_rows": len(rows),
        "agreement_status": agreement.get("status"),
        "decision_verdict": decision["verdict"],
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
