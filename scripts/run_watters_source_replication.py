#!/usr/bin/env python3
"""Reproduce Watters et al. source model and latent-gain comparisons.

The OSF ``data_for_figures`` release contains the held-out per-unit likelihoods
and per-trial latent attention values from all ten random seeds.  Reusing those
released fits is the exact source-defined comparison; retraining them would add
optimizer variation without changing the estimand.  Trial-level behavior is
collapsed across seeds before session- and animal-level inference so seeds are
not treated as independent animals or sessions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402

SEED = 20260731
N_BOOT_SOURCE = 1000
SOURCE_BOOTSTRAP_OBSERVATIONS = 100
# Normalized screen units between the reported and the cued object position,
# below which a trial counts as correct.
CORRECT_REPORT_DISTANCE_THRESHOLD = 0.2
MODEL_NAMES = {
    "gain": "Gain",
    "slot_partition": "Slot",
    "switching": "Switching",
}


def data_directories() -> tuple[Path, Path]:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    data_root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not data_root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    configured = Path(data_root) / config["datasets"]["watters_2026"]["local_path"]
    watters = configured.parent if configured.name == "data_for_modeling" else configured
    figures = watters / "data_for_figures" / "data_for_figures"
    modeling = figures / "modeling" / "main"
    behavior = figures / "behavior_processing"
    if not modeling.is_dir() or not behavior.is_dir():
        raise SystemExit(f"Watters processed figure cache is incomplete under {figures}")
    return modeling, behavior


def flatten_likelihood(payload: dict) -> np.ndarray:
    return np.asarray([value for unit in payload["ll_per_unit"] for value in unit], dtype=float)


def source_bootstrap_probabilities(
    likelihood: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT_SOURCE,
) -> np.ndarray:
    """Return source-defined model probabilities for model × observation LL."""
    if likelihood.ndim != 2 or likelihood.shape[0] != 3:
        raise ValueError("likelihood must have three aligned model rows")
    draw = rng.integers(0, likelihood.shape[1], size=(n_boot, SOURCE_BOOTSTRAP_OBSERVATIONS))
    totals = likelihood[:, draw].sum(axis=2).T
    return softmax(totals, axis=1)


def compare_models(modeling: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for task in ("triangle", "ring"):
        directories = {
            key: modeling / f"{task}_{key}"
            for key in MODEL_NAMES
        }
        sessions = sorted(
            (subject.name, session.name)
            for subject in directories["gain"].iterdir()
            if subject.is_dir() and not subject.name.startswith(".")
            for session in subject.iterdir()
            if session.is_dir() and not session.name.startswith(".")
        )
        for subject, session in sessions:
            seed_sets = [
                {int(path.name) for path in (directory / subject / session).iterdir() if path.name.isdigit()}
                for directory in directories.values()
            ]
            seeds = sorted(set.intersection(*seed_sets))
            for seed in seeds:
                payloads = {}
                num_neurons = {}
                for key, directory in directories.items():
                    run = directory / subject / session / str(seed)
                    payloads[key] = json.loads((run / "stop_step" / "eval_stats.json").read_text())
                    num_neurons[key] = int(json.loads((run / "test_metrics.json").read_text())[0]["num_neurons"])
                trial_maps = [json.dumps(payloads[key]["trials_per_unit"], separators=(",", ":")) for key in MODEL_NAMES]
                if len(set(trial_maps)) != 1:
                    raise ValueError(f"held-out trial mapping mismatch for {task} {subject} {session} seed {seed}")
                flattened = [flatten_likelihood(payloads[key]) for key in MODEL_NAMES]
                if len({len(values) for values in flattened}) != 1:
                    raise ValueError("likelihood vectors are not aligned")
                matrix = np.stack(flattened)
                rng = np.random.default_rng(SEED + seed + sum(map(ord, f"{task}{subject}{session}")))
                probabilities = source_bootstrap_probabilities(matrix, rng)
                mean_ll = matrix.mean(axis=1)
                for index, key in enumerate(MODEL_NAMES):
                    rows.append({
                        "task": task,
                        "animal": subject,
                        "session": session,
                        "seed": int(seed),
                        "model": MODEL_NAMES[key],
                        "source_probability": float(probabilities[:, index].mean()),
                        "mean_log_likelihood_per_unit_trial": float(mean_ll[index]),
                        "n_aligned_unit_trials": int(matrix.shape[1]),
                        "n_neurons": num_neurons[key],
                    })
    frame = pd.DataFrame(rows)
    session = (
        frame.groupby(["task", "animal", "session", "model"], as_index=False)
        .agg(source_probability=("source_probability", "mean"),
             mean_log_likelihood_per_unit_trial=("mean_log_likelihood_per_unit_trial", "mean"),
             n_neurons=("n_neurons", "first"))
    )
    winners = session.loc[session.groupby(["task", "animal", "session"])["source_probability"].idxmax()]
    summary = {
        task: {
            animal: {
                "n_sessions": int(group["session"].nunique()),
                "mean_probability_by_model": {
                    model: float(group.loc[group.model == model, "source_probability"].mean())
                    for model in ("Slot", "Switching", "Gain")
                },
                "sessions_won_by_model": {
                    model: int(np.sum((winners.task == task) & (winners.animal == animal) & (winners.model == model)))
                    for model in ("Slot", "Switching", "Gain")
                },
            }
            for animal, group in session.loc[session.task == task].groupby("animal")
        }
        for task in ("triangle", "ring")
    }
    return rows, {"by_task_and_animal": summary, "n_session_seed_comparisons": int(len(frame) // 3)}


def add_behavior_columns(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    frame = frame.loc[frame.completed.astype(bool)].copy()
    targets_x = np.empty(len(frame))
    targets_y = np.empty(len(frame))
    for row_index, (_, row) in enumerate(frame.iterrows()):
        target = int(row.target_object_index)
        radius = float(row[f"object_{target}_r"])
        theta = float(row[f"object_{target}_theta"])
        targets_x[row_index] = radius * np.cos(theta)
        targets_y[row_index] = radius * np.sin(theta)
    response_x = frame.response_r.to_numpy(float) * np.cos(frame.response_theta.to_numpy(float))
    response_y = frame.response_r.to_numpy(float) * np.sin(frame.response_theta.to_numpy(float))
    # The saccadic report is continuous, so the distance between the reported and
    # the cued position is the graded quantity and the correctness label is a
    # threshold on it. Both are returned: an analysis that can use the graded
    # form is not restricted to the trials where the animal failed.
    frame["report_deviation"] = np.hypot(response_x - targets_x, response_y - targets_y)
    frame["correct"] = frame["report_deviation"] < CORRECT_REPORT_DISTANCE_THRESHOLD
    frame["reaction_time_ms"] = 1000.0 * (frame.response_time - frame.time_cue_onset)
    frame["task"] = task
    return frame


def load_gain_trials(modeling: Path, behavior: Path, task: str) -> pd.DataFrame:
    source = add_behavior_columns(pd.read_csv(behavior / f"{task}.csv"), task)
    source = source.set_index(["subject", "session", "trial_num"], drop=False)
    records: list[dict] = []
    directory = modeling / f"{task}_gain"
    for subject_dir in sorted(path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            per_seed = []
            trial_reference = None
            for seed_dir in sorted((path for path in session_dir.iterdir() if path.name.isdigit()), key=lambda path: int(path.name)):
                stop = seed_dir / "stop_step"
                trials = np.load(stop / "dataset_cache" / "trial_num.npy").astype(int)
                attention = np.load(stop / "model_cache" / "attention.npy")
                if trial_reference is None:
                    trial_reference = trials
                elif not np.array_equal(trial_reference, trials):
                    raise ValueError(f"seed trial mismatch in {task} {subject_dir.name} {session_dir.name}")
                per_seed.append(attention)
            mean_attention = np.mean(np.stack(per_seed), axis=0)
            for row_index, trial in enumerate(trial_reference):
                key = (subject_dir.name, session_dir.name, int(trial))
                if key not in source.index:
                    raise ValueError(f"behavior trial absent: {key}")
                row = source.loc[key]
                target = int(row.target_object_index)
                records.append({
                    "task": task,
                    "animal": subject_dir.name,
                    "session": session_dir.name,
                    "trial_num": int(trial),
                    "num_objects": int(row.num_objects),
                    "correct": bool(row.correct),
                    "reaction_time_ms": float(row.reaction_time_ms),
                    "target_gain": float(mean_attention[row_index, target]),
                    "n_seeds_averaged": int(len(per_seed)),
                })
    return pd.DataFrame(records)


def session_slopes(frame: pd.DataFrame, max_items: int) -> list[dict]:
    frame = frame.loc[frame.num_objects == max_items].copy()
    output = []
    for (task, animal, session), group in frame.groupby(["task", "animal", "session"]):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_gain", "reaction_time_ms"])
        gain_sd = float(group.target_gain.std(ddof=0))
        base = {"task": task, "animal": animal, "session": session, "n_trials": int(len(group))}
        if len(group) < 30 or gain_sd <= 1e-10 or group.correct.nunique() < 2:
            output.append({**base, "status": "not_estimable", "reason": "insufficient trials, gain variance, or outcome variation"})
            continue
        z = (group.target_gain.to_numpy(float) - float(group.target_gain.mean())) / gain_sd
        design = sm.add_constant(z)
        try:
            logistic = sm.GLM(group.correct.astype(int).to_numpy(), design, family=sm.families.Binomial()).fit()
            correct = group.correct.to_numpy(bool)
            rt = sm.OLS(group.loc[correct, "reaction_time_ms"].to_numpy(float), sm.add_constant(z[correct])).fit()
            output.append({
                **base,
                "status": "complete",
                "performance_log_odds_per_gain_sd": float(logistic.params[1]),
                "performance_standard_error": float(logistic.bse[1]),
                "correct_rt_ms_per_gain_sd": float(rt.params[1]),
                "correct_rt_standard_error": float(rt.bse[1]),
                "n_correct_rt_trials": int(np.sum(correct)),
            })
        except Exception as exc:
            output.append({**base, "status": "nonconverged", "reason": str(exc)})
    return output


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draw = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return list(map(float, np.percentile(draw, [2.5, 97.5])))


def summarize_behavior(slopes: list[dict]) -> dict:
    complete = pd.DataFrame([row for row in slopes if row["status"] == "complete"])
    rng = np.random.default_rng(SEED)
    summary = {}
    for task in ("triangle", "ring"):
        summary[task] = {}
        for animal in sorted(complete.loc[complete.task == task, "animal"].unique()):
            group = complete[(complete.task == task) & (complete.animal == animal)]
            effects = {}
            for column in ("performance_log_odds_per_gain_sd", "correct_rt_ms_per_gain_sd"):
                values = group[column].to_numpy(float)
                effects[column] = {
                    "mean_session_slope": float(np.mean(values)),
                    "session_bootstrap_ci": bootstrap_mean(values, rng),
                    "n_sessions": int(len(values)),
                    "sessions_positive": int(np.sum(values > 0)),
                }
            summary[task][animal] = effects
    return summary


def main() -> None:
    modeling, behavior = data_directories()
    print("reproducing released held-out Gain/Slot/Switching comparisons", flush=True)
    model_rows, model_summary = compare_models(modeling)
    print("testing released latent gain against errors and correct-trial RT", flush=True)
    behavior_trials = pd.concat([
        load_gain_trials(modeling, behavior, "triangle"),
        load_gain_trials(modeling, behavior, "ring"),
    ], ignore_index=True)
    slopes = session_slopes(behavior_trials[behavior_trials.task == "triangle"], 3)
    slopes += session_slopes(behavior_trials[behavior_trials.task == "ring"], 2)
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "watters_2026_source_replication",
        "dataset": "Watters et al. 2026 OSF vyw49 processed cache",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "metadata_decision": {
            "processed_cache_only": True,
            "raw_dandi_000620_downloaded": False,
            "animals": ["Elgar", "Perle"],
            "independent_unit": "session nested in animal; ten optimizer seeds averaged",
            "spatial_configuration_replication": ["triangle", "ring"],
            "go_no_go": "go",
        },
        "model_comparison": {
            "method": "released held-out per-unit/trial Poisson log likelihood; source-defined 100-observation bootstrap model probability",
            "rows": model_rows,
            "summary": model_summary,
        },
        "latent_gain_behavior": {
            "method": "released target gain averaged across ten seeds; within-session gain standardization; session-specific logistic error and correct-trial OLS RT slopes",
            "trial_counts": {
                task: int(np.sum(behavior_trials.task == task)) for task in ("triangle", "ring")
            },
            "session_slopes": slopes,
            "summary": summarize_behavior(slopes),
        },
        "limitations": [
            "The exact released trained fits are re-evaluated; stochastic optimizer retraining is intentionally not substituted for the released source comparison.",
            "Two monkeys are reported separately, so no universal primate effect is claimed.",
            "The behavior check is hierarchy-corrected and therefore need not numerically equal source notebook correlations that repeat optimizer seeds as rows.",
            "This artifact reproduces the source models and behavior link only; the project-specific item-count diffusion test is a separate artifact.",
        ],
    }
    destination = ROOT / "results" / "watters_2026_source_replication.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "model_summary": model_summary,
        "behavior_summary": output["latent_gain_behavior"]["summary"],
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
