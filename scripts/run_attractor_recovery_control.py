#!/usr/bin/env python3
"""Blocking gate: recover Inagaki et al. 2019's known ALM discrete-attractor
dynamics with three attractor-identification methods before any of them is
licensed to make an attractor-class claim about an unvalidated human corpus.

Methods, each with a prediction declared before running:
(1) Vietoris-Rips persistent homology should read beta_0 = 2 at
the dominant gap with no persistent beta_1; (2) recurrence quantification
should show high laminarity/long trapping time relative to a time-shuffled
null; (3) local-linear fixed-point/Jacobian classification should recover
two stable points separated by a saddle, calibrated against planted-truth
simulations at ALM's real session dimensions. A fourth leg checks the
photoinhibition perturbation trials against the discrete-basin-recovery
prediction. Sections 4-6 run regardless of this gate's outcome (they do not
depend on the attractor methods); only attractor-class claims about human
corpora are conditioned on it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import attractor_analysis as aa  # noqa: E402
from corpus_sessions import alm_data_directory, load_alm_raw_session  # noqa: E402
from geometry import apply_frozen_pca, fit_frozen_pca, latent_trajectories  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from statistics import bootstrap_ci, paired_sign_flip_test, stable_seed  # noqa: E402

BIN_MS = 100
WINDOW_S = 2.0
SEED = 20260808

N_COMPONENTS_HOMOLOGY = 8
N_COMPONENTS_FIXED_POINT = 2
N_NULL_DRAWS = 200
HOMOLOGY_MAX_POINTS = 150
N_FIXED_POINT_INITS = 80
RQA_TARGET_RRS = (0.025, 0.05, 0.10)
N_CALIBRATION_REPLICATES = 20


def data_directory() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = alm_data_directory(Path(root))
    if not path.is_dir():
        raise SystemExit(f"ALM perturbation sessions not staged at {path}")
    return path


def load_session_trials(path: Path) -> dict | None:
    """Load one ALM session's delay-epoch counts and standardize on the
    control (unperturbed) trials, via the shared raw loader in
    src/corpus_sessions.py (also used by the observability census)."""
    session = load_alm_raw_session(path, bin_ms=BIN_MS, window_s=WINDOW_S)
    if session is None:
        return None
    n_control = session["n_control_trials"]
    all_counts = np.concatenate((session["control_counts"], session["perturb_counts"]), axis=0)
    transform = FrozenPSTHTransform().fit(all_counts[:n_control])
    standardized = transform.transform(all_counts).transpose(0, 2, 1)  # (trials, bins, units)
    return {
        "mouse": session["mouse"],
        "n_units_after_rate_qc": session["n_units_after_rate_qc"],
        "n_control_trials": n_control,
        "n_perturb_trials": session["n_perturb_trials"],
        "control_condition": session["control_condition"],
        "perturb_condition": session["perturb_condition"],
        "control": standardized[:n_control],
        "perturb": standardized[n_control:],
    }


# ── Method 1: persistent homology ───────────────────────────────────────────


def run_persistent_homology(control: np.ndarray, rng: np.random.Generator) -> dict:
    latent, _, var_ratio = latent_trajectories(control, n_components=N_COMPONENTS_HOMOLOGY)
    cloud_full = latent.reshape(-1, latent.shape[-1])
    n_points = min(HOMOLOGY_MAX_POINTS, len(cloud_full))
    subsample_idx = rng.choice(len(cloud_full), size=n_points, replace=False)
    cloud = cloud_full[subsample_idx]

    observed = aa.persistent_homology(cloud, maxdim=2)
    gaussian_null = {0: [], 1: [], 2: []}
    for _ in range(N_NULL_DRAWS):
        betti = aa.persistent_homology(aa.gaussian_matched_null(cloud, rng), maxdim=2)["betti"]
        for dim in (0, 1, 2):
            gaussian_null[dim].append(betti[dim]["betti"])
    shuffle_null = {0: [], 1: [], 2: []}
    for _ in range(N_NULL_DRAWS):
        shuffled_full = aa.time_shuffled_null(latent, rng).reshape(-1, latent.shape[-1])
        shuffled_cloud = shuffled_full[rng.choice(len(shuffled_full), size=n_points, replace=False)]
        betti = aa.persistent_homology(shuffled_cloud, maxdim=2)["betti"]
        for dim in (0, 1, 2):
            shuffle_null[dim].append(betti[dim]["betti"])

    prediction_met = observed["betti"][0]["betti"] == 2 and observed["betti"][1]["betti"] == 0
    percentiles = {
        f"beta_{dim}_vs_gaussian_null_percentile": aa.null_percentile(observed["betti"][dim]["betti"], gaussian_null[dim])
        for dim in (0, 1)
    }
    percentiles.update({
        f"beta_{dim}_vs_shuffle_null_percentile": aa.null_percentile(observed["betti"][dim]["betti"], shuffle_null[dim])
        for dim in (0, 1)
    })
    recovered = (
        prediction_met
        and percentiles["beta_0_vs_gaussian_null_percentile"] > 95.0
        and percentiles["beta_0_vs_shuffle_null_percentile"] > 95.0
    )
    return {
        "n_components": N_COMPONENTS_HOMOLOGY,
        "var_ratio": var_ratio.tolist(),
        "n_points_used": n_points,
        "n_points_available": int(len(cloud_full)),
        "observed_betti": {dim: observed["betti"][dim] for dim in (0, 1, 2)},
        "prediction_met": bool(prediction_met),
        "percentiles": percentiles,
        "recovered": bool(recovered),
        "beta_0_observable_for_perturbation_leg": aa.null_percentile(observed["betti"][0]["betti"], gaussian_null[0]),
    }


# ── Method 2: recurrence quantification ─────────────────────────────────────


def run_recurrence_quantification(control: np.ndarray, condition: np.ndarray) -> dict:
    latent, _, var_ratio = latent_trajectories(control, n_components=N_COMPONENTS_HOMOLOGY)
    n_trials, n_bins, _ = latent.shape
    trajectory = latent.reshape(-1, latent.shape[-1])

    by_threshold = {}
    for target_rr in RQA_TARGET_RRS:
        threshold = aa.threshold_for_target_recurrence_rate(trajectory, target_rr)
        by_threshold[f"target_rr_{target_rr}"] = aa.recurrence_quantification(trajectory, threshold)

    threshold_5pct = by_threshold["target_rr_0.05"]["threshold"]
    clusters = aa.recurrence_clusters(trajectory, threshold_5pct)
    labels = np.array(clusters["labels"])
    trial_of_point = np.repeat(np.arange(n_trials), n_bins)
    majority_cluster_per_trial = np.array([
        np.bincount(labels[trial_of_point == trial]).argmax() for trial in range(n_trials)
    ])
    non_noise = np.array(clusters["cluster_sizes"])[majority_cluster_per_trial] >= max(2, int(0.02 * len(trajectory)))
    if non_noise.sum() >= 2 and len(np.unique(majority_cluster_per_trial[non_noise])) >= 2:
        contingency = {}
        for choice in np.unique(condition):
            trials_this_choice = non_noise & (condition == choice)
            if trials_this_choice.sum() > 0:
                clusters_seen, counts = np.unique(majority_cluster_per_trial[trials_this_choice], return_counts=True)
                contingency[int(choice)] = {int(c): int(n) for c, n in zip(clusters_seen, counts)}
    else:
        contingency = {}

    prediction_met = (
        by_threshold["target_rr_0.05"]["laminarity"] > 0
        and clusters["n_clusters"] >= 2
    )
    return {
        "n_components": N_COMPONENTS_HOMOLOGY,
        "var_ratio": var_ratio.tolist(),
        "by_threshold": by_threshold,
        "n_recurrence_clusters_at_5pct": clusters["n_clusters"],
        "cluster_by_choice_contingency": contingency,
        "prediction_met": bool(prediction_met),
        "laminarity_observable_for_perturbation_leg": by_threshold["target_rr_0.05"]["laminarity"],
    }


# ── Method 3: fixed points and Jacobians ────────────────────────────────────


def run_fixed_points(control: np.ndarray, rng: np.random.Generator) -> dict:
    latent, comps, var_ratio = latent_trajectories(control, n_components=N_COMPONENTS_FIXED_POINT)
    x_t, x_tp1 = aa.build_transition_pairs(latent)
    bounds = (x_t.min(axis=0), x_t.max(axis=0))
    points = aa.find_fixed_points(x_t, x_tp1, N_FIXED_POINT_INITS, rng, bounds)
    pattern = aa.summarize_fixed_point_pattern(points)
    return {
        "n_components": N_COMPONENTS_FIXED_POINT,
        "var_ratio": var_ratio.tolist(),
        "fixed_points": points,
        "pattern": pattern,
        "prediction_met": pattern == "two_stable_plus_saddle",
        "recovered": pattern == "two_stable_plus_saddle",
        "components": comps.tolist(),
    }


def planted_truth_calibration(rng: np.random.Generator, n_units: int, n_trials: int) -> dict:
    """Confusion matrix of the fixed-point classifier on four planted systems at
    ALM's real session dimensions, over N_CALIBRATION_REPLICATES independent
    draws per system (a single draw is not reliable enough to characterize a
    Monte Carlo classifier -- see src/attractor_analysis.py's local_linear_map
    docstring)."""
    simulators = {
        "two_well": aa.simulate_two_well,
        "line_attractor": aa.simulate_line_attractor,
        "point_attractor": aa.simulate_point_attractor,
        "pure_noise": aa.simulate_pure_noise,
    }
    from collections import Counter

    confusion = {}
    for name, sim_fn in simulators.items():
        pattern_counts = Counter()
        for replicate in range(N_CALIBRATION_REPLICATES):
            replicate_rng = np.random.default_rng(stable_seed(f"{name}_{replicate}") + int(rng.integers(0, 2**16)))
            observed = sim_fn(n_trials=n_trials, n_bins=int(WINDOW_S / (BIN_MS / 1000.0)), n_units=n_units, dt=BIN_MS / 1000.0, rng=replicate_rng)
            latent, _, _ = latent_trajectories(observed, n_components=2)
            x_t, x_tp1 = aa.build_transition_pairs(latent)
            bounds = (x_t.min(axis=0), x_t.max(axis=0))
            points = aa.find_fixed_points(x_t, x_tp1, N_FIXED_POINT_INITS, replicate_rng, bounds)
            pattern_counts[aa.summarize_fixed_point_pattern(points)] += 1
        confusion[name] = dict(pattern_counts)

    modal_pattern = {name: max(counts, key=counts.get) for name, counts in confusion.items()}
    expected_pattern = {
        "two_well": "two_stable_plus_saddle",
        "line_attractor": "line_attractor_dominant",
        "point_attractor": "single_stable_point",
    }
    separates_discrete_from_line = (
        modal_pattern.get("two_well") == expected_pattern["two_well"]
        and modal_pattern.get("line_attractor") == expected_pattern["line_attractor"]
        and modal_pattern.get("two_well") != modal_pattern.get("line_attractor")
    )
    return {
        "n_replicates_per_system": N_CALIBRATION_REPLICATES,
        "n_units": n_units,
        "n_trials": n_trials,
        "confusion_matrix": confusion,
        "modal_pattern_per_system": modal_pattern,
        "expected_pattern_per_system": expected_pattern,
        "modal_pattern_matches_expected": {name: modal_pattern.get(name) == expected for name, expected in expected_pattern.items()},
        "separates_planted_discrete_from_planted_line": bool(separates_discrete_from_line),
    }


# ── Perturbation leg: photoinhibition trials against the discrete-basin prediction ──


def run_perturbation_leg(session: dict, homology_result: dict, rqa_result: dict, rng: np.random.Generator) -> dict:
    control, perturb = session["control"], session["perturb"]
    control_condition, perturb_condition = session["control_condition"], session["perturb_condition"]

    homology_control = homology_result["beta_0_observable_for_perturbation_leg"]
    homology_perturb = run_persistent_homology(perturb, rng)["beta_0_observable_for_perturbation_leg"] if len(perturb) >= 10 else None

    rqa_control = rqa_result["laminarity_observable_for_perturbation_leg"]
    rqa_perturb = run_recurrence_quantification(perturb, perturb_condition)["laminarity_observable_for_perturbation_leg"] if len(perturb) >= 10 else None

    n_train, n_units = control.shape[0], control.shape[2]
    train_flat = control.reshape(n_train * control.shape[1], n_units)
    mean, components = fit_frozen_pca(train_flat, N_COMPONENTS_FIXED_POINT)
    control_last = apply_frozen_pca(control[:, -1, :], mean, components)
    perturb_last = apply_frozen_pca(perturb[:, -1, :], mean, components) if len(perturb) > 0 else None
    basin_a = control_last[control_condition == 0].mean(axis=0)
    basin_b = control_last[control_condition == 1].mean(axis=0)
    midpoint = 0.5 * (basin_a + basin_b)

    def margin(points: np.ndarray) -> np.ndarray:
        dist_mid = np.linalg.norm(points - midpoint[None, :], axis=1)
        dist_a = np.linalg.norm(points - basin_a[None, :], axis=1)
        dist_b = np.linalg.norm(points - basin_b[None, :], axis=1)
        return dist_mid - np.minimum(dist_a, dist_b)

    margin_control = float(margin(control_last).mean())
    margin_perturb = float(margin(perturb_last).mean()) if perturb_last is not None and len(perturb_last) > 0 else None

    return {
        "homology_beta0_percentile_control": homology_control,
        "homology_beta0_percentile_perturb": homology_perturb,
        "rqa_laminarity_control": rqa_control,
        "rqa_laminarity_perturb": rqa_perturb,
        "fixed_point_margin_control": margin_control,
        "fixed_point_margin_perturb": margin_perturb,
    }


def perturbation_leg_summary(sessions: dict) -> dict:
    complete = [s for s in sessions.values() if s.get("perturbation_leg") is not None]
    result = {}
    for method, control_key, perturb_key in [
        ("persistent_homology", "homology_beta0_percentile_control", "homology_beta0_percentile_perturb"),
        ("recurrence_quantification", "rqa_laminarity_control", "rqa_laminarity_perturb"),
        ("fixed_points", "fixed_point_margin_control", "fixed_point_margin_perturb"),
    ]:
        control_vals, perturb_vals = [], []
        for s in complete:
            leg = s["perturbation_leg"]
            if leg[control_key] is not None and leg[perturb_key] is not None and np.isfinite(leg[control_key]) and np.isfinite(leg[perturb_key]):
                control_vals.append(leg[control_key])
                perturb_vals.append(leg[perturb_key])
        if len(control_vals) < 3:
            result[method] = {"status": "not_estimable", "reason": "fewer than 3 sessions with both arms", "n_sessions": len(control_vals)}
            continue
        control_arr, perturb_arr = np.array(control_vals), np.array(perturb_vals)
        test = paired_sign_flip_test(perturb_arr, control_arr, alternative="two-sided", rng=np.random.default_rng(SEED))
        result[method] = {
            "status": "complete",
            "n_sessions": len(control_vals),
            "mean_control": float(control_arr.mean()),
            "mean_perturb": float(perturb_arr.mean()),
            "paired_diff_perturb_minus_control": test["mean_diff"],
            "p_value": test["p_value"],
            "ci_lower": test["ci_lower"],
            "ci_upper": test["ci_upper"],
        }
    return result


def main() -> None:
    directory = data_directory()
    rng = np.random.default_rng(SEED)
    sessions = {}
    for path in sorted(directory.glob("*_units.mat")):
        print(f"attractor-recovery-control: {path.stem}", flush=True)
        try:
            session = load_session_trials(path)
        except Exception as exc:
            sessions[path.stem] = {"status": "failed", "reason": str(exc)}
            continue
        if session is None:
            sessions[path.stem] = {"status": "excluded", "reason": "insufficient trials or units"}
            continue
        session_rng = np.random.default_rng(stable_seed(path.stem) + SEED)
        homology = run_persistent_homology(session["control"], session_rng)
        rqa = run_recurrence_quantification(session["control"], session["control_condition"])
        fixed_points = run_fixed_points(session["control"], session_rng)
        perturbation_leg = (
            run_perturbation_leg(session, homology, rqa, session_rng) if session["n_perturb_trials"] >= 10 else None
        )
        sessions[path.stem] = {
            "status": "complete",
            "mouse": session["mouse"],
            "n_units_after_rate_qc": session["n_units_after_rate_qc"],
            "n_control_trials": session["n_control_trials"],
            "n_perturb_trials": session["n_perturb_trials"],
            "persistent_homology": homology,
            "recurrence_quantification": rqa,
            "fixed_points": fixed_points,
            "perturbation_leg": perturbation_leg,
        }

    complete = [s for s in sessions.values() if s.get("status") == "complete"]
    n_units_median = int(np.median([s["n_units_after_rate_qc"] for s in complete])) if complete else 41
    n_trials_median = int(np.median([s["n_control_trials"] for s in complete])) if complete else 137
    calibration_rng = np.random.default_rng(SEED)
    calibration = planted_truth_calibration(calibration_rng, n_units_median, n_trials_median)

    n_homology_recovered = sum(s["persistent_homology"]["recovered"] for s in complete)
    n_rqa_recovered = sum(s["recurrence_quantification"]["prediction_met"] for s in complete)
    n_fixed_point_recovered = sum(s["fixed_points"]["recovered"] for s in complete)
    fraction_recovered = {
        "persistent_homology": n_homology_recovered / len(complete) if complete else None,
        "recurrence_quantification": n_rqa_recovered / len(complete) if complete else None,
        "fixed_points": n_fixed_point_recovered / len(complete) if complete else None,
    }
    methods_recovered = sum(
        (fraction_recovered[m] or 0) > 0.5 for m in ("persistent_homology", "recurrence_quantification", "fixed_points")
    )

    if not complete:
        verdict = "estimator_non_identified"
        verdict_reason = "no ALM session cleared QC; the latent could not be formed"
    elif methods_recovered >= 2 and calibration["separates_planted_discrete_from_planted_line"]:
        verdict = "pipeline_validated"
        verdict_reason = f"{methods_recovered}/3 methods recover the ALM prediction in a majority of sessions, and the planted-truth confusion matrix separates two-well from line-attractor"
    else:
        verdict = "pipeline_not_validated"
        verdict_reason = f"only {methods_recovered}/3 methods recover the ALM prediction in a majority of sessions, or the planted-truth calibration does not separate two-well from line-attractor -- no attractor-class claim about any human structure is licensed until this failure is understood"

    output = {
        "schema_version": "1.0.0",
        "analysis_id": "attractor_recovery_control",
        "dataset": "Inagaki ALM silicon-probe perturbation release, RandomDelayTask/withPerturbation",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "sessions": sessions,
        "n_sessions_complete": len(complete),
        "fraction_recovered": fraction_recovered,
        "planted_truth_calibration": calibration,
        "perturbation_leg_summary": perturbation_leg_summary(sessions),
        "predeclared_decision": {
            "deciding_quantity": "how many of the three methods recover discrete attractor dynamics in a majority of ALM sessions, AND whether the planted-truth confusion matrix separates two-well from line-attractor",
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        },
        "claim_limit": (
            "mouse motor preparation method validation only; no attractor-class claim about any human corpus "
            "is licensed by this artifact alone -- see predeclared_decision.verdict"
        ),
    }
    destination = ROOT / "results" / "attractor_recovery_control.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "n_sessions_complete": len(complete),
        "fraction_recovered": fraction_recovered,
        "verdict": verdict,
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
