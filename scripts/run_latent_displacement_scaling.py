#!/usr/bin/env python3
"""Band-free geometry: latent displacement scaling.

The confinement-rate (lambda) estimator has an identifiability band and
cannot resolve every structure; mean squared displacement (MSD) as a
function of lag has no such gate and runs on the full denominator. This is
the observable that decides whether prior anatomical null results were a
property of the underlying biology or an artifact of the estimator's
identifiability band.

MSD(k) = mean over held-out trials of ||z(t0+k) - z(t0)||^2, k = 1..K bins,
in a leakage-free PCA latent (fit on train folds only, matching every other
leave-one-fold-out fit in this project). Three summaries: saturation_ratio =
MSD(K)/MSD(K/2) (the primary, dimensionless observable: 1.0 fully confined,
2.0 an unconfined random walk, >2 directed drift), log_log_slope (OLS slope
of log MSD on log lag: 0 confined, 1 random walk, 2 ballistic), and
msd_at_K_state_units_sq (per-structure only, never compared across
structures since PCA units are not shared across fits).

Corpora: DANDI 000469, the canonical 001187/000673 dedup, and DANDI 000574
(Boran) via src/corpus_sessions.py -- see that module's docstring for why
DANDI 000004 is not yet included.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import attractor_analysis as aa  # noqa: E402
from corpus_sessions import data_root, iter_all_corpora  # noqa: E402
from drift_dynamics import confinement_identifiability, fit_ou_moments  # noqa: E402
from geometry import phase_scrambled_null  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform, build_psth  # noqa: E402
from statistics import fdr_bh, paired_sign_flip_test, stable_seed  # noqa: E402

SEED = 20260808
BIN_MS = 100
N_SPLITS = 5
N_COMPONENTS = 8
MIN_TEST_TRIALS = 5
N_NULL_DRAWS = 200
DECIDING_CONTRAST = ("pre_sma", "hippocampus")
N_CALIBRATION_MATCHED_DRAWS = 20


def fold_latents(psth: np.ndarray, seed: int) -> list[np.ndarray | None]:
    """Leakage-free per-fold held-out latent trajectories: PCA fit on train
    trials only, test trials projected into that basis. One entry per fold;
    None marks a fold with too few test trials to be usable."""
    n_trials, n_units, n_bins = psth.shape
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in splitter.split(np.arange(n_trials)):
        if len(test_idx) < MIN_TEST_TRIALS:
            folds.append(None)
            continue
        k = min(N_COMPONENTS, n_units, len(train_idx) - 1)
        if k < 2:
            folds.append(None)
            continue
        transform = FrozenPSTHTransform().fit(psth[train_idx])
        train_std = transform.transform(psth[train_idx]).transpose(0, 2, 1)
        test_std = transform.transform(psth[test_idx]).transpose(0, 2, 1)
        pca = PCA(n_components=k, svd_solver="full").fit(train_std.reshape(-1, train_std.shape[-1]))
        test_latent = pca.transform(test_std.reshape(-1, test_std.shape[-1])).reshape(len(test_idx), n_bins, k)
        folds.append(test_latent)
    return folds


def mean_squared_displacement(latent: np.ndarray) -> np.ndarray:
    """latent: (n_trials, n_bins, k) -> msd[lag-1] for lag = 1..n_bins-1."""
    n_bins = latent.shape[1]
    max_lag = n_bins - 1
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        diffs = latent[:, lag:, :] - latent[:, :-lag, :]
        msd[lag - 1] = np.mean(np.sum(diffs**2, axis=2))
    return msd


def summarize_msd(msd: np.ndarray) -> dict:
    max_lag = len(msd)
    half = max(1, max_lag // 2)
    saturation_ratio = float(msd[-1] / msd[half - 1]) if msd[half - 1] > 0 else None
    lags = np.arange(1, max_lag + 1)
    valid = msd > 0
    if valid.sum() >= 3:
        slope, _ = np.polyfit(np.log(lags[valid]), np.log(msd[valid]), 1)
        log_log_slope = float(slope)
    else:
        log_log_slope = None
    return {
        "saturation_ratio": saturation_ratio,
        "log_log_slope": log_log_slope,
        "msd_at_K_state_units_sq": float(msd[-1]),
        "msd_curve": msd.tolist(),
    }


def analyze_fold(latent: np.ndarray, rng: np.random.Generator) -> dict:
    if latent.shape[1] < 4:
        return {"status": "excluded", "reason": "fewer than 4 bins; cannot form MSD(K/2) distinctly from MSD(K)"}
    observed_msd = mean_squared_displacement(latent)
    observed = summarize_msd(observed_msd)

    shuffle_ratios, shuffle_slopes = [], []
    phase_ratios, phase_slopes = [], []
    for _ in range(N_NULL_DRAWS):
        shuffled = aa.time_shuffled_null(latent, rng)
        s = summarize_msd(mean_squared_displacement(shuffled))
        if s["saturation_ratio"] is not None:
            shuffle_ratios.append(s["saturation_ratio"])
        if s["log_log_slope"] is not None:
            shuffle_slopes.append(s["log_log_slope"])
    for _ in range(N_NULL_DRAWS):
        scrambled = phase_scrambled_null(latent, rng)
        s = summarize_msd(mean_squared_displacement(scrambled))
        if s["saturation_ratio"] is not None:
            phase_ratios.append(s["saturation_ratio"])
        if s["log_log_slope"] is not None:
            phase_slopes.append(s["log_log_slope"])

    percentiles = {}
    if observed["saturation_ratio"] is not None:
        if shuffle_ratios:
            percentiles["saturation_ratio_vs_shuffle_null_percentile"] = aa.null_percentile(observed["saturation_ratio"], shuffle_ratios)
        if phase_ratios:
            percentiles["saturation_ratio_vs_phase_scrambled_null_percentile"] = aa.null_percentile(observed["saturation_ratio"], phase_ratios)

    # OU identifiability on the leading-PC projection, alongside the same fold's MSD, per comments' request
    # to sit both denominators (fraction_of_folds_with_a_value vs fraction_identified) in one table.
    leading_pc_residual = latent[:, :, 0] - latent[:, :1, 0]
    ou_estimate = fit_ou_moments(leading_pc_residual, BIN_MS / 1000.0, n_boot=50, rng=rng)
    duration_s = latent.shape[1] * BIN_MS / 1000.0
    lambda_rate = ou_estimate.lambda_rate if ou_estimate.lambda_rate is not None else float("nan")
    identifiability = confinement_identifiability(lambda_rate, BIN_MS / 1000.0, duration_s)

    return {
        "status": "complete",
        "n_test_trials": int(latent.shape[0]),
        "n_bins": int(latent.shape[1]),
        "observed": observed,
        "null_percentiles": percentiles,
        "ou_identifiability_status": identifiability.status,
        "ou_identifiability_reason": identifiability.reason,
        "ou_lambda_dt": identifiability.lambda_dt,
        "ou_time_constants_spanned": identifiability.delay_time_constants,
    }


def analyze_unit_pool(spike_lists: list, epoch_onsets: dict, epoch_windows: dict, seed: int, rng: np.random.Generator) -> dict:
    onset, window = epoch_onsets["delay"], epoch_windows["delay"]
    psth = build_psth(spike_lists, onset, bin_ms=BIN_MS, smooth_ms=0, window_s=window)
    folds = fold_latents(psth, seed)
    fold_results = []
    for fold in folds:
        if fold is None:
            fold_results.append({"status": "excluded", "reason": "fewer than 5 test trials or fewer than 2 usable components"})
            continue
        fold_results.append(analyze_fold(fold, rng))
    n_attempted = len(fold_results)
    n_computed = sum(f["status"] == "complete" for f in fold_results)
    n_ou_identified = sum(f.get("ou_identifiability_status") == "identifiable" for f in fold_results if f["status"] == "complete")
    return {
        "n_folds_attempted": n_attempted,
        "n_folds_computed": n_computed,
        "fraction_of_folds_with_a_value": (n_computed / n_attempted) if n_attempted else None,
        "n_folds_ou_identified": n_ou_identified,
        "fraction_identified": (n_ou_identified / n_computed) if n_computed else None,
        "folds": fold_results,
        "_psth_for_matched_count": psth,
    }


def collect_rows(root: Path, rng: np.random.Generator) -> dict:
    rows = {}
    for item in iter_all_corpora(root):
        key = f"{item['dataset']}/{item['session']}/{item['structure']}"
        seed = stable_seed(key)
        rows[key] = {
            "dataset": item["dataset"], "patient": item["patient"], "session": item["session"], "structure": item["structure"],
            "delay": analyze_unit_pool(item["spike_lists"], item["epoch_onsets"], item["epoch_windows"], seed, rng),
        }
    return rows


def session_saturation_ratio(row: dict) -> float | None:
    values = [f["observed"]["saturation_ratio"] for f in row["delay"]["folds"] if f["status"] == "complete" and f["observed"]["saturation_ratio"] is not None]
    return float(np.median(values)) if values else None


def matched_count_comparison(rows: dict, structure_a: str, structure_b: str, rng: np.random.Generator) -> dict:
    by_patient_structure = {}
    for row in rows.values():
        if "_psth_for_matched_count" not in row["delay"]:
            continue
        key = (row["dataset"], row["patient"], row["structure"])
        by_patient_structure[key] = row["delay"]["_psth_for_matched_count"]

    patients_seen = set((d, p) for d, p, s in by_patient_structure)
    paired_rows = []
    for dataset, patient in patients_seen:
        key_a, key_b = (dataset, patient, structure_a), (dataset, patient, structure_b)
        if key_a not in by_patient_structure or key_b not in by_patient_structure:
            continue
        psth_a, psth_b = by_patient_structure[key_a], by_patient_structure[key_b]
        n_match = min(psth_a.shape[1], psth_b.shape[1])
        if n_match < 5:
            continue
        ratios_a, ratios_b = [], []
        for draw in range(N_CALIBRATION_MATCHED_DRAWS):
            idx_a = rng.choice(psth_a.shape[1], size=n_match, replace=False)
            idx_b = rng.choice(psth_b.shape[1], size=n_match, replace=False)
            seed_a = stable_seed(f"{dataset}_{patient}_{structure_a}_{draw}")
            seed_b = stable_seed(f"{dataset}_{patient}_{structure_b}_{draw}")
            folds_a = fold_latents(psth_a[:, idx_a, :], seed_a)
            folds_b = fold_latents(psth_b[:, idx_b, :], seed_b)
            for fold in folds_a:
                if fold is not None and fold.shape[1] >= 4:
                    s = summarize_msd(mean_squared_displacement(fold))
                    if s["saturation_ratio"] is not None:
                        ratios_a.append(s["saturation_ratio"])
            for fold in folds_b:
                if fold is not None and fold.shape[1] >= 4:
                    s = summarize_msd(mean_squared_displacement(fold))
                    if s["saturation_ratio"] is not None:
                        ratios_b.append(s["saturation_ratio"])
        if ratios_a and ratios_b:
            paired_rows.append({
                "dataset": dataset, "patient": patient, "n_matched_units": int(n_match),
                "mean_saturation_ratio_a": float(np.mean(ratios_a)), "mean_saturation_ratio_b": float(np.mean(ratios_b)),
            })
    return {
        "structure_a": structure_a, "structure_b": structure_b,
        "n_attempted_patients": len(patients_seen),
        "n_paired_patients": len(paired_rows),
        "fraction_with_a_value": (len(paired_rows) / len(patients_seen)) if patients_seen else None,
        "rows": paired_rows,
    }


def predeclared_decision(matched: dict, rows: dict, rng: np.random.Generator) -> dict:
    structure_a, structure_b = DECIDING_CONTRAST
    n_paired = matched["n_paired_patients"]
    if n_paired < 6:
        return {
            "deciding_contrast": f"{structure_a} minus {structure_b} on saturation_ratio, within patient, delay epoch",
            "verdict": "underpowered_by_construction",
            "n_paired_patients": n_paired,
            "reason": "fewer than 6 paired patients; minimum attainable sign-flip p exceeds 0.05 by construction",
        }
    a = np.array([row["mean_saturation_ratio_a"] for row in matched["rows"]])
    b = np.array([row["mean_saturation_ratio_b"] for row in matched["rows"]])
    test = paired_sign_flip_test(a, b, alternative="two-sided", rng=rng)

    structure_null_clearance = {}
    for structure in (structure_a, structure_b):
        percentiles = [
            f["null_percentiles"].get("saturation_ratio_vs_shuffle_null_percentile")
            for row in rows.values() if row["structure"] == structure
            for f in row["delay"]["folds"] if f["status"] == "complete"
        ]
        percentiles = [p for p in percentiles if p is not None]
        clears = (np.median(percentiles) > 95.0) if percentiles else False
        structure_null_clearance[structure] = {"n_folds": len(percentiles), "median_percentile_vs_shuffle_null": float(np.median(percentiles)) if percentiles else None, "clears_null": bool(clears)}

    both_clear_null = all(structure_null_clearance[s]["clears_null"] for s in (structure_a, structure_b))
    if test["p_value"] < 0.05 and both_clear_null:
        verdict = "confinement_differs_by_structure"
    elif test["p_value"] >= 0.05 and both_clear_null:
        verdict = "no_structure_difference_band_free"
    else:
        verdict = "observable_has_no_signal"
    return {
        "deciding_contrast": f"{structure_a} minus {structure_b} on saturation_ratio, within patient, delay epoch, exact paired sign-flip test",
        "n_paired_patients": n_paired,
        "mean_diff": test["mean_diff"], "p_value": test["p_value"], "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "structure_null_clearance": structure_null_clearance,
        "verdict": verdict,
    }


def strip_internal_fields(rows: dict) -> dict:
    for row in rows.values():
        row["delay"].pop("_psth_for_matched_count", None)
    return rows


def main() -> None:
    root = data_root()
    rng = np.random.default_rng(SEED)
    rows = collect_rows(root, rng)

    matched = matched_count_comparison(rows, DECIDING_CONTRAST[0], DECIDING_CONTRAST[1], np.random.default_rng(SEED + 1))
    decision = predeclared_decision(matched, rows, np.random.default_rng(SEED + 2))

    all_structures = sorted({row["structure"] for row in rows.values()})
    family_pairs = [(a, b) for i, a in enumerate(all_structures) for b in all_structures[i + 1:]]
    family_results = []
    for a, b in family_pairs:
        m = matched_count_comparison(rows, a, b, np.random.default_rng(stable_seed(f"{a}_{b}_msd")))
        d = predeclared_decision(m, rows, np.random.default_rng(stable_seed(f"{a}_{b}_msd_test")))
        family_results.append({"pair": [a, b], "n_paired_patients": m["n_paired_patients"], "decision": d})
    finite_p = [r["decision"]["p_value"] for r in family_results if "p_value" in r["decision"]]
    fdr = fdr_bh(np.array(finite_p)) if finite_p else {"status": "not_estimable"}

    all_denominators = [row["delay"] for row in rows.values()]
    n_folds_attempted_total = sum(r["n_folds_attempted"] for r in all_denominators)
    n_folds_computed_total = sum(r["n_folds_computed"] for r in all_denominators)
    n_folds_ou_identified_total = sum(r["n_folds_ou_identified"] for r in all_denominators)

    output = {
        "schema_version": "1.0.0",
        "analysis_id": "latent_displacement_scaling",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
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
        "denominator_summary": {
            "n_folds_attempted": n_folds_attempted_total,
            "n_folds_computed": n_folds_computed_total,
            "fraction_of_folds_with_a_value": (n_folds_computed_total / n_folds_attempted_total) if n_folds_attempted_total else None,
            "n_folds_ou_identified": n_folds_ou_identified_total,
            "fraction_identified": (n_folds_ou_identified_total / n_folds_computed_total) if n_folds_computed_total else None,
        },
        "deciding_contrast_matched_count_comparison": matched,
        "predeclared_decision": decision,
        "family_wise_ordered_pairs": family_results,
        "family_wise_fdr_bh": fdr,
        "rows": strip_internal_fields(rows),
    }
    destination = ROOT / "results" / "latent_displacement_scaling.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "n_rows": len(rows),
        "fraction_of_folds_with_a_value": output["denominator_summary"]["fraction_of_folds_with_a_value"],
        "fraction_identified": output["denominator_summary"]["fraction_identified"],
        "decision_verdict": decision["verdict"],
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
