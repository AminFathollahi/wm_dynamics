#!/usr/bin/env python3
"""Design-gated raw-spike recovery analysis for the macaque uStim corpus.

The release separates correct and error trials into different files.  Each
file retains its within-file ``trialsequence``, but no shared original trial
index or timestamp exists with which to reconstruct their interleaving or the
published randomization blocks.  Consequently this script estimates neural
recovery and descriptive behavior associations, while explicitly blocking a
design-correct causal claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from drift_dynamics import fit_gaussian_state_space, leave_one_out_condition_residuals  # noqa: E402
from provenance import git_commit  # noqa: E402
from statistics import stable_seed  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import DATA, SESSIONS, load_macaque_pfc_microstimulation_session  # noqa: E402

BIN_MS = 50
N_BINS = 30
ONSET_BIN = 16
N_COMPONENTS = 8
N_BOOT = 500


def bin_spiketrain(spikes: np.ndarray, bin_ms: int = BIN_MS, n_bins: int = N_BINS) -> np.ndarray | None:
    """Convert the released unsmoothed 1-ms binary train to nonoverlapping counts."""
    values = np.asarray(spikes)
    required = bin_ms * n_bins
    if values.ndim != 2 or values.shape[0] < required:
        return None
    return values[:required].reshape(n_bins, bin_ms, values.shape[1]).sum(axis=1).astype(np.float32)


def discriminant_direction(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the dominant multiclass shrinkage-LDA coefficient direction."""
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(values, labels)
    _, _, right = np.linalg.svd(model.coef_, full_matrices=False)
    direction = right[0]
    class_means = np.array([np.mean(values[labels == label] @ direction) for label in model.classes_])
    if len(class_means) > 1 and np.corrcoef(np.arange(len(class_means)), class_means)[0, 1] < 0:
        direction = -direction
    return direction / max(np.linalg.norm(direction), 1e-12)


def _pool_rows(prefix: str, correct: bool) -> tuple[list[dict], dict]:
    loaded = load_macaque_pfc_microstimulation_session(prefix, correct=correct, neural_field="spiketrain")
    if loaded is None:
        return [], {}
    outcome = "correct" if correct else "error"
    occurrences: dict[tuple[int, int], int] = {}
    rows = []
    for trial in loaded["trials"]:
        key = (int(trial["stim_cond"]), int(trial["angle_idx"]))
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        counts = bin_spiketrain(trial["spikerate"])
        if counts is None:
            continue
        rows.append({
            "trial_id": f"{prefix}:{outcome}:{key[0]}:{key[1]}:{occurrence}",
            "correct": int(correct),
            "stim_cond": key[0],
            "angle_idx": key[1],
            "counts": counts,
        })
    metadata = {
        "channel_ids": np.asarray(loaded["channel_ids"], dtype=int),
        "stim_channels": loaded["stim_channels"],
        "control_idx": int(loaded["control_idx"]),
        "trialsequence_present": bool(loaded.get("trialsequence_present", False)),
        "n_trialsequence": int(loaded.get("n_trialsequence", 0)),
        "n_channels_dropped_shorted": int(loaded.get("n_channels_dropped_shorted", 0)),
    }
    return rows, metadata


def _log_odds_effect(rows: list[dict], stim_cond: int, control_idx: int) -> dict | None:
    """Angle-stratified descriptive log-odds contrast with Jeffreys correction."""
    angles = sorted({r["angle_idx"] for r in rows if r["stim_cond"] in (stim_cond, control_idx)})
    effects, variances, tables = [], [], []
    for angle in angles:
        cells = {}
        for condition in (stim_cond, control_idx):
            correct = sum(r["correct"] for r in rows if r["stim_cond"] == condition and r["angle_idx"] == angle)
            total = sum(1 for r in rows if r["stim_cond"] == condition and r["angle_idx"] == angle)
            cells[condition] = (correct, total - correct)
        sc, se = cells[stim_cond]
        cc, ce = cells[control_idx]
        if sc + se == 0 or cc + ce == 0:
            continue
        effect = np.log((sc + 0.5) / (se + 0.5)) - np.log((cc + 0.5) / (ce + 0.5))
        variance = sum(1.0 / (value + 0.5) for value in (sc, se, cc, ce))
        effects.append(float(effect))
        variances.append(float(variance))
        tables.append({"angle": angle, "stim_correct": sc, "stim_error": se,
                       "control_correct": cc, "control_error": ce})
    if not effects:
        return None
    weights = 1.0 / np.asarray(variances)
    estimate = float(np.sum(weights * effects) / np.sum(weights))
    standard_error = float(np.sqrt(1.0 / np.sum(weights)))
    return {"estimate": estimate, "ci": [estimate - 1.96 * standard_error,
                                           estimate + 1.96 * standard_error],
            "standard_error": standard_error, "angle_tables": tables}


def _recovery_curve(time: np.ndarray, offset: float, amplitude: float, rate: float) -> np.ndarray:
    return offset + amplitude * np.exp(-rate * time)


def fit_recovery(displacement: np.ndarray) -> dict | None:
    """Fit a signed post-stim offset-plus-exponential recovery curve."""
    values = np.asarray(displacement, dtype=float)[ONSET_BIN:]
    time = np.arange(len(values), dtype=float) * BIN_MS / 1000.0
    finite = np.isfinite(values)
    if finite.sum() < 6 or np.ptp(values[finite]) < 1e-10:
        return None
    try:
        parameters, _ = curve_fit(
            _recovery_curve, time[finite], values[finite],
            p0=(values[finite][-1], values[finite][0] - values[finite][-1], 2.0),
            bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 50.0]),
            maxfev=20_000,
        )
    except (RuntimeError, ValueError):
        return None
    prediction = _recovery_curve(time[finite], *parameters)
    denominator = np.sum((values[finite] - np.mean(values[finite])) ** 2)
    r2 = 1.0 - np.sum((values[finite] - prediction) ** 2) / max(denominator, 1e-12)
    return {"offset": float(parameters[0]), "amplitude": float(parameters[1]),
            "lambda_rate_per_s": float(parameters[2]), "r2": float(r2)}


def _condition_curve(states: np.ndarray, rows: list[dict], condition: int, control_idx: int) -> np.ndarray:
    curves = []
    for angle in sorted({r["angle_idx"] for r in rows}):
        stim = [index for index, row in enumerate(rows)
                if row["stim_cond"] == condition and row["angle_idx"] == angle]
        control = [index for index, row in enumerate(rows)
                   if row["stim_cond"] == control_idx and row["angle_idx"] == angle]
        if stim and control:
            curves.append(states[stim].mean(axis=0) - states[control].mean(axis=0))
    return np.mean(curves, axis=0) if curves else np.full(states.shape[1], np.nan)


def _bootstrap_recovery(states: np.ndarray, rows: list[dict], condition: int,
                        control_idx: int, rng: np.random.Generator, n_boot: int) -> list[float]:
    by_cell = {}
    for angle in sorted({r["angle_idx"] for r in rows}):
        for cond in (condition, control_idx):
            by_cell[(angle, cond)] = np.array([
                index for index, row in enumerate(rows)
                if row["angle_idx"] == angle and row["stim_cond"] == cond
            ], dtype=int)
    values = []
    for _ in range(n_boot):
        curves = []
        for angle in sorted({key[0] for key in by_cell}):
            stim, control = by_cell[(angle, condition)], by_cell[(angle, control_idx)]
            if len(stim) and len(control):
                s = rng.choice(stim, size=len(stim), replace=True)
                c = rng.choice(control, size=len(control), replace=True)
                curves.append(states[s].mean(axis=0) - states[c].mean(axis=0))
        if curves:
            fit = fit_recovery(np.mean(curves, axis=0))
            if fit is not None and np.isfinite(fit["lambda_rate_per_s"]):
                values.append(fit["lambda_rate_per_s"])
    return values


def analyze_session(prefix: str, n_boot: int = N_BOOT) -> dict:
    correct_rows, correct_meta = _pool_rows(prefix, True)
    error_rows, error_meta = _pool_rows(prefix, False)
    if not correct_rows or not error_rows:
        return {"session": prefix, "status": "excluded", "reason": "missing correct/error raw-spike pool"}
    if not np.array_equal(correct_meta["channel_ids"], error_meta["channel_ids"]):
        return {"session": prefix, "status": "excluded", "reason": "correct/error channel bases differ"}
    rows = correct_rows + error_rows
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise RuntimeError(f"{prefix}: duplicate trial identifiers")
    control_idx = correct_meta["control_idx"]
    channel_ids = set(correct_meta["channel_ids"].tolist())
    eligible = [condition for condition, channels in enumerate(correct_meta["stim_channels"])
                if condition != control_idx and channels and all(channel in channel_ids for channel in channels)]
    control_rows = [row for row in rows if row["stim_cond"] == control_idx]
    if len(control_rows) < 20:
        return {"session": prefix, "status": "excluded", "reason": "fewer than 20 shared control trials"}

    control_counts = np.stack([row["counts"] for row in control_rows])
    center = control_counts.reshape(-1, control_counts.shape[-1]).mean(axis=0)
    scale = control_counts.reshape(-1, control_counts.shape[-1]).std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized_control = (control_counts - center) / scale
    pca = PCA(n_components=min(N_COMPONENTS, standardized_control.shape[-1]))
    pca.fit(standardized_control.reshape(-1, standardized_control.shape[-1]))
    all_counts = np.stack([row["counts"] for row in rows])
    latent = pca.transform(((all_counts - center) / scale).reshape(-1, all_counts.shape[-1])).reshape(
        len(rows), N_BINS, -1)
    control_index = np.array([index for index, row in enumerate(rows) if row["stim_cond"] == control_idx])
    labels = np.array([rows[index]["angle_idx"] for index in control_index])
    pre_values = latent[control_index, :ONSET_BIN].mean(axis=1)
    direction = discriminant_direction(pre_values, labels)
    states = latent @ direction

    control_states = states[control_index]
    residuals, _ = leave_one_out_condition_residuals(control_states, labels)
    endogenous = fit_gaussian_state_space(residuals, BIN_MS / 1000.0).to_dict()

    pattern_rows = []
    rng = np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_design_corrected_{prefix}"))
    for condition in eligible:
        curve = _condition_curve(states, rows, condition, control_idx)
        recovery = fit_recovery(curve)
        bootstrap = _bootstrap_recovery(states, rows, condition, control_idx, rng, n_boot)
        recovery_ci = (np.quantile(bootstrap, [0.025, 0.975]).tolist()
                       if len(bootstrap) >= max(30, n_boot // 4) else None)
        behavior = _log_odds_effect(rows, condition, control_idx)
        post_early = curve[ONSET_BIN:ONSET_BIN + 4]
        pattern_rows.append({
            "condition": condition,
            "stim_channels": correct_meta["stim_channels"][condition],
            "n_trials": sum(row["stim_cond"] == condition for row in rows),
            "n_correct": sum(row["stim_cond"] == condition and row["correct"] for row in rows),
            "early_signed_displacement": float(np.mean(post_early)),
            "early_absolute_displacement": float(np.mean(np.abs(post_early))),
            "poststim_displacement": curve[ONSET_BIN:].tolist(),
            "recovery": recovery,
            "recovery_lambda_bootstrap_ci": recovery_ci,
            "n_recovery_bootstrap": len(bootstrap),
            "behavior_log_odds_effect": behavior,
        })

    return {
        "session": prefix,
        "animal": prefix[:2],
        "status": "complete",
        "n_trials_unique": len(rows),
        "n_correct": len(correct_rows),
        "n_error": len(error_rows),
        "n_control_trials_unique": len(control_rows),
        "n_control_correct": sum(row["correct"] for row in control_rows),
        "n_control_error": sum(1 - row["correct"] for row in control_rows),
        "n_channels": len(channel_ids),
        "n_channels_dropped_shorted": correct_meta["n_channels_dropped_shorted"],
        "trialsequence": {
            "correct_present": correct_meta["trialsequence_present"],
            "correct_length": correct_meta["n_trialsequence"],
            "error_present": error_meta["trialsequence_present"],
            "error_length": error_meta["n_trialsequence"],
        },
        "control_idx": control_idx,
        "eligible_stim_conditions": eligible,
        "ineligible_stim_conditions": [condition for condition in range(len(correct_meta["stim_channels"]))
                                        if condition != control_idx and condition not in eligible],
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "endogenous_control_drift": endogenous,
        "patterns": pattern_rows,
        "trial_id_sha256": hashlib.sha256("\n".join(sorted(trial_ids)).encode()).hexdigest(),
    }


def fixed_effect_slope(rows: list[dict], animal: str | None, n_boot: int = 2000) -> dict | None:
    selected = [row for row in rows if animal is None or row["animal"] == animal]
    sessions = sorted({row["session"] for row in selected})
    if len(selected) < 4 or len(sessions) < 2:
        return None

    def estimate(sampled_sessions: list[str]) -> float:
        xs, ys = [], []
        for session in sampled_sessions:
            group = [row for row in selected if row["session"] == session]
            if len(group) < 2:
                continue
            x = np.array([row["x"] for row in group])
            y = np.array([row["y"] for row in group])
            xs.extend((x - x.mean()).tolist())
            ys.extend((y - y.mean()).tolist())
        if len(xs) < 2 or np.dot(xs, xs) < 1e-12:
            return np.nan
        return float(np.dot(xs, ys) / np.dot(xs, xs))

    point = estimate(sessions)
    rng = np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_pattern_regression_{animal}"))
    draws = np.array([estimate(rng.choice(sessions, size=len(sessions), replace=True).tolist())
                      for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    return {"slope_log_odds_per_absolute_displacement": point,
            "session_bootstrap_ci": np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) else None,
            "n_patterns": len(selected), "n_sessions": len(sessions),
            "n_bootstrap_finite": len(draws), "animal": animal or "pooled_with_session_fixed_effects"}


def recovery_agreement(per_session: list[dict], n_boot: int = 2000) -> dict:
    """Compare recovery and endogenous rates only where endogenous precision is resolved."""
    session_values = {}
    interval_overlaps = []
    n_patterns_total = 0
    for session in per_session:
        endogenous = session.get("endogenous_control_drift") or {}
        estimate = endogenous.get("lambda_rate")
        interval = endogenous.get("lambda_ci")
        for pattern in session.get("patterns", []):
            recovery = pattern.get("recovery") or {}
            rate = recovery.get("lambda_rate_per_s")
            if rate is not None:
                n_patterns_total += 1
            precision_identified = (estimate is not None and estimate > 0 and interval is not None
                                    and np.all(np.isfinite(interval)) and interval[0] > 0)
            if rate is None or rate <= 0 or not precision_identified:
                continue
            session_values.setdefault(session["session"], []).append(float(np.log(rate / estimate)))
            recovery_interval = pattern.get("recovery_lambda_bootstrap_ci")
            if recovery_interval is not None:
                interval_overlaps.append(max(interval[0], recovery_interval[0])
                                         <= min(interval[1], recovery_interval[1]))
    session_means = {key: float(np.mean(value)) for key, value in session_values.items()}
    values = np.array(list(session_means.values()), dtype=float)
    if len(values) >= 2:
        rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_recovery_agreement"))
        draws = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
        group_ci = np.quantile(draws, [0.025, 0.975]).tolist()
    else:
        group_ci = None
    return {
        "n_recovery_patterns": n_patterns_total,
        "n_patterns_with_precision_identified_endogenous_lambda": int(sum(len(v) for v in session_values.values())),
        "n_sessions_with_precision_identified_endogenous_lambda": len(session_values),
        "mean_log_recovery_to_endogenous_ratio": float(values.mean()) if len(values) else None,
        "session_bootstrap_ci": group_ci,
        "interval_overlap_count": int(sum(interval_overlaps)),
        "interval_comparison_count": len(interval_overlaps),
        "verdict": ("agreement_not_established" if len(session_values) < 3 or group_ci is None
                    else ("agreement_compatible" if group_ci[0] <= 0 <= group_ci[1]
                          else "rates_differ")),
        "reason": "Most endogenous state-space Wald intervals cross zero; rate agreement is restricted to sessions whose endogenous interval is wholly positive.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=None, help="Run the first N sessions for a smoke test")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "macaque_pfc_microstimulation_design_corrected.json")
    args = parser.parse_args()
    if "__WM_DYNAMICS_DATA_ROOT_NOT_SET__" in str(DATA) or not DATA.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT; configured macaque PFC microstimulation data directory is unavailable.")
    chosen = SESSIONS[:args.sessions] if args.sessions else SESSIONS
    per_session = []
    for prefix in chosen:
        print(f"fitting macaque PFC microstimulation raw-spike recovery {prefix}", flush=True)
        per_session.append(analyze_session(prefix, n_boot=args.bootstrap))

    regression_rows = []
    for session in per_session:
        if session.get("status") != "complete":
            continue
        for pattern in session["patterns"]:
            behavior = pattern["behavior_log_odds_effect"]
            if behavior is not None:
                regression_rows.append({"session": session["session"], "animal": session["animal"],
                                        "x": pattern["early_absolute_displacement"],
                                        "y": behavior["estimate"]})
    regressions = {"pooled": fixed_effect_slope(regression_rows, None)}
    for animal in sorted({row["animal"] for row in regression_rows}):
        regressions[animal] = fixed_effect_slope(regression_rows, animal)

    complete = [row for row in per_session if row.get("status") == "complete"]
    output = {
        "analysis": "Macaque PFC microstimulation raw-spike perturbation recovery with design gate",
        "git_commit": git_commit(ROOT),
        "parameters": {"bin_ms": BIN_MS, "n_bins": N_BINS, "window_s": [-0.8, 0.7],
                       "stim_onset_bin": ONSET_BIN, "n_components": N_COMPONENTS,
                       "recovery_bootstrap": args.bootstrap},
        "design_gate": {
            "block_randomization_declared_in_source": True,
            "correct_and_error_trialsequence_present": all(
                row["trialsequence"]["correct_present"] and row["trialsequence"]["error_present"]
                for row in complete),
            "shared_original_trial_index_or_timestamp_present": False,
            "cross_outcome_interleaving_recoverable": False,
            "within_block_randomization_inference": "not_identifiable",
            "G3_perturbation_claim": "no_go",
            "descriptive_recovery_analysis": "go",
            "reason": "correct and error outcomes are stored in separate files without a shared original-trial key, so the published randomization blocks cannot be reconstructed",
        },
        "open_question_resolution": {
            "Q1_original_blocks_reconstructable": {
                "answer": "no",
                "reason": "the correct and error files preserve separate within-file trialsequence arrays but expose no shared original-trial index or timestamp, so their interleaving, block IDs, and target-angle allocation cannot be recovered",
            },
            "Q2_Sa_absent_from_primary_result": {
                "answer": "Sa has no eligible stimulation pattern after electrical short-channel quality control",
                "reason": "105 of 192 Sa channels are removed by the release-provided short-channel masks, and every bipolar stimulation condition includes at least one contact absent from the retained recording basis; the eligibility rule requires every stimulation contact to survive",
                "outcome_independent_rule": True,
                "rule_timing": "channel eligibility is determined from release-provided electrical short metadata before behavioral outcomes are modeled",
                "descriptive_data_retained": "Sa control and outcome trials remain in the endogenous recovery description, but no Sa stimulation pattern enters a targeting comparison",
            },
        },
        "evidence": {"n_sessions": len(complete), "n_animals": len({row["animal"] for row in complete}),
                     "n_unique_trials": sum(row["n_trials_unique"] for row in complete),
                     "n_unique_control_trials": sum(row["n_control_trials_unique"] for row in complete),
                     "n_eligible_patterns": sum(len(row["patterns"]) for row in complete)},
        "per_session": per_session,
        "recovery_endogenous_agreement": recovery_agreement(per_session),
        "behavior_displacement_regressions": regressions,
        "interpretation": "Neural recovery and behavior-displacement associations are descriptive; this artifact does not identify a causal stimulation effect or validate an input map.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(args.output), "evidence": output["evidence"],
                      "design_gate": output["design_gate"], "regressions": regressions}, indent=2))


if __name__ == "__main__":
    main()
