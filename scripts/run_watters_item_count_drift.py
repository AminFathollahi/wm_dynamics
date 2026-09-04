#!/usr/bin/env python3
"""Test whether Watters multi-object diffusion scales with item count.

The analysis uses the exact source-included trials and source-filtered units
from the released Gain fits, but estimates dynamics from the staged raw
10-ms spike-count cache.  Coordinates are fit on the opposite stratified
fold.  Scalar LGSSMs separate process and observation noise in each retained
PC; total process diffusion, diffusion per item, and the effective number of
diffusive dimensions are summarized per session and animal.  Triangle and
ring configurations are kept separate as an independent configuration check.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import fit_gaussian_state_space  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402

RAW_BIN_MS = 10
DELAY_RAW_BINS = (120, 220)
ANALYSIS_BIN_MS = (100, 50)
N_COMPONENTS = 4
N_SPLITS = 2
MIN_STABLE_UNITS = 15
MIN_TRIALS_PER_ITEM_FOLD = 20
SEED = 20260731


def data_directories() -> tuple[Path, Path, Path]:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    data_root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not data_root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    configured = Path(data_root) / config["datasets"]["watters_2026"]["local_path"]
    watters = configured.parent if configured.name == "data_for_modeling" else configured
    spikes = watters / "data_for_modeling" / "data_for_modeling" / "spikes_per_trial"
    figures = watters / "data_for_figures" / "data_for_figures"
    models = figures / "modeling" / "main"
    behavior = figures / "behavior_processing"
    if not spikes.is_dir() or not models.is_dir() or not behavior.is_dir():
        raise SystemExit(f"Watters processed caches are incomplete under {watters}")
    return spikes, models, behavior


def aggregate_delay_bins(values: list, bin_ms: int) -> np.ndarray:
    """Aggregate one released 10-ms trial into unsmoothed delay counts."""
    array = np.asarray(values, dtype=float)[slice(*DELAY_RAW_BINS)]
    factor = bin_ms // RAW_BIN_MS
    if bin_ms % RAW_BIN_MS or len(array) % factor:
        raise ValueError("analysis bins must exactly partition the released delay window")
    return array.reshape(-1, factor).sum(axis=1)


def behavior_for_task(path: Path, task: str) -> pd.DataFrame:
    frame = pd.read_csv(path / f"{task}.csv")
    frame = frame.loc[frame.completed.astype(bool)].copy()
    if task == "triangle":
        frame = frame.loc[frame.on_triangle.astype(bool)]
    else:
        left_right = (
            (frame.num_objects == 2)
            & (
                (np.isclose(frame.object_0_theta, 0) & np.isclose(frame.object_1_theta, np.pi))
                | (np.isclose(frame.object_1_theta, 0) & np.isclose(frame.object_0_theta, np.pi))
            )
        )
        frame = frame.loc[~left_right]
    return frame.set_index(["subject", "session", "trial_num"], drop=False)


def source_sessions(models: Path, task: str) -> list[tuple[str, str, Path]]:
    directory = models / f"{task}_gain"
    return sorted(
        (subject.name, session.name, session / "2")
        for subject in directory.iterdir()
        if subject.is_dir() and not subject.name.startswith(".")
        for session in subject.iterdir()
        if session.is_dir() and not session.name.startswith(".")
    )


def load_session_counts(
    spikes: Path,
    run: Path,
    subject: str,
    session: str,
    bin_ms: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    trials = np.load(run / "stop_step" / "dataset_cache" / "trial_num.npy").astype(int)
    units = pd.read_csv(run / "df_units.csv")
    stable = []
    for _, unit in units.iterrows():
        directory = spikes / subject / session / str(unit.probe) / str(unit.quality)
        trial_path = directory / f"{int(unit.unit)}_trials.pkl"
        count_path = directory / f"{int(unit.unit)}_spike_counts.pkl"
        unit_trials = np.asarray(pickle.load(open(trial_path, "rb")), dtype=int)
        index = {int(trial): row for row, trial in enumerate(unit_trials)}
        if not all(int(trial) in index for trial in trials):
            continue
        unit_counts = pickle.load(open(count_path, "rb"))
        stable.append(np.stack([aggregate_delay_bins(unit_counts[index[int(trial)]], bin_ms) for trial in trials]))
    metadata = {
        "n_source_trials": int(len(trials)),
        "n_source_units": int(len(units)),
        "n_units_present_on_every_source_trial": int(len(stable)),
    }
    if len(stable) < MIN_STABLE_UNITS:
        return np.empty((len(trials), 0, 0)), trials, metadata
    counts = np.stack(stable, axis=1)
    return counts, trials, metadata


def fold_item_metrics(
    counts: np.ndarray,
    item_count: np.ndarray,
    bin_ms: int,
    seed: int,
) -> list[dict]:
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(item_count)), item_count)):
        transform = FrozenPSTHTransform().fit(counts[train])
        train_values = transform.transform(counts[train]).transpose(0, 2, 1)
        test_values = transform.transform(counts[test]).transpose(0, 2, 1)
        n_components = min(N_COMPONENTS, train_values.shape[2], len(train) - 1)
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(train_values.reshape(-1, train_values.shape[2]))
        train_state = pca.transform(train_values.reshape(-1, train_values.shape[2])).reshape(len(train), counts.shape[2], -1)
        test_state = pca.transform(test_values.reshape(-1, test_values.shape[2])).reshape(len(test), counts.shape[2], -1)
        for n_items in sorted(np.unique(item_count)):
            train_mask = item_count[train] == n_items
            test_mask = item_count[test] == n_items
            if np.sum(train_mask) < MIN_TRIALS_PER_ITEM_FOLD or np.sum(test_mask) < MIN_TRIALS_PER_ITEM_FOLD:
                rows.append({
                    "fold": int(fold), "num_objects": int(n_items), "status": "not_estimable",
                    "reason": "fewer than 20 trials in a train or estimation fold",
                })
                continue
            centroid = train_state[train_mask].mean(axis=0)
            residual = test_state[test_mask] - centroid[None, :, :]
            estimates = [
                fit_gaussian_state_space(residual[:, :, component], bin_ms / 1000.0)
                for component in range(n_components)
            ]
            usable = [estimate for estimate in estimates if estimate.converged and estimate.diffusion is not None]
            if len(usable) != n_components:
                rows.append({
                    "fold": int(fold), "num_objects": int(n_items), "status": "not_estimable",
                    "reason": "one or more component LGSSMs did not converge",
                    "component_estimates": [estimate.to_dict() for estimate in estimates],
                })
                continue
            diffusion = np.asarray([estimate.diffusion for estimate in usable], dtype=float)
            total = float(np.sum(diffusion))
            effective = float(total * total / max(np.sum(diffusion * diffusion), 1e-12))
            rows.append({
                "fold": int(fold),
                "num_objects": int(n_items),
                "status": "complete",
                "n_estimation_trials": int(np.sum(test_mask)),
                "total_diffusion_standardized_latent_variance_per_s": total,
                "diffusion_per_item": float(total / n_items),
                "effective_diffusive_dimensions": effective,
                "identifiable_components": int(sum(
                    estimate.identifiability is not None
                    and estimate.identifiability.status == "identifiable"
                    for estimate in usable
                )),
                "component_estimates": [estimate.to_dict() for estimate in estimates],
            })
    return rows


def session_summary(folds: list[dict]) -> dict:
    complete = pd.DataFrame([row for row in folds if row["status"] == "complete"])
    if complete.empty:
        return {"status": "not_estimable", "reason": "no complete item-count fold estimates"}
    metrics = (
        complete.groupby("num_objects", as_index=False)
        .agg(
            total_diffusion=("total_diffusion_standardized_latent_variance_per_s", "mean"),
            diffusion_per_item=("diffusion_per_item", "mean"),
            effective_diffusive_dimensions=("effective_diffusive_dimensions", "mean"),
            identifiable_components=("identifiable_components", "sum"),
            n_folds=("fold", "nunique"),
        )
    )
    if len(metrics) < 2:
        return {"status": "not_estimable", "reason": "fewer than two item counts completed", "by_item_count": metrics.to_dict("records")}
    x = metrics.num_objects.to_numpy(float)
    slopes = {
        column: float(np.polyfit(x, metrics[column].to_numpy(float), 1)[0])
        for column in ("total_diffusion", "diffusion_per_item", "effective_diffusive_dimensions")
    }
    return {"status": "complete", "by_item_count": metrics.to_dict("records"), "slope_per_added_item": slopes}


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def group_summary(sessions: dict) -> dict:
    rng = np.random.default_rng(SEED)
    output = {}
    for bin_key in ("100ms", "50ms"):
        output[bin_key] = {}
        for task in ("triangle", "ring"):
            output[bin_key][task] = {}
            for animal in ("Elgar", "Perle"):
                rows = []
                for row in sessions.values():
                    if row["task"] != task or row["animal"] != animal:
                        continue
                    summary = row["analyses"][bin_key].get("summary", {})
                    if summary.get("status") == "complete":
                        rows.append(summary)
                effects = {}
                for metric in ("total_diffusion", "diffusion_per_item", "effective_diffusive_dimensions"):
                    values = np.asarray([row["slope_per_added_item"][metric] for row in rows], dtype=float)
                    effects[metric] = {
                        "mean_session_slope_per_added_item": float(np.mean(values)) if len(values) else None,
                        "session_bootstrap_ci": bootstrap_mean(values, rng) if len(values) else None,
                        "n_sessions": int(len(values)),
                        "sessions_positive": int(np.sum(values > 0)),
                    }
                output[bin_key][task][animal] = effects
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-sessions", type=int, default=None)
    args = parser.parse_args()
    spikes, models, behavior_dir = data_directories()
    behavior = {task: behavior_for_task(behavior_dir, task) for task in ("triangle", "ring")}
    entries = [(task, *entry) for task in ("triangle", "ring") for entry in source_sessions(models, task)]
    if args.limit_sessions is not None:
        entries = entries[:args.limit_sessions]
    sessions = {}
    for task, animal, session, run in entries:
        key = f"{task}:{animal}:{session}"
        print(f"fitting Watters item-count drift {key}", flush=True)
        row = {"task": task, "animal": animal, "session": session, "analyses": {}}
        for bin_ms in ANALYSIS_BIN_MS:
            counts, trials, metadata = load_session_counts(spikes, run, animal, session, bin_ms)
            labels = np.asarray([
                int(behavior[task].loc[(animal, session, int(trial)), "num_objects"])
                for trial in trials
            ])
            if counts.shape[1] < MIN_STABLE_UNITS:
                analysis = {"status": "excluded", "reason": "fewer than 15 source units span every source trial", "metadata": metadata}
            else:
                folds = fold_item_metrics(counts, labels, bin_ms, SEED + sum(map(ord, key)) + bin_ms)
                analysis = {
                    "status": "complete",
                    "metadata": {**metadata, "item_count_trials": {str(value): int(np.sum(labels == value)) for value in np.unique(labels)}},
                    "folds": folds,
                    "summary": session_summary(folds),
                }
            row["analyses"][f"{bin_ms}ms"] = analysis
        sessions[key] = row
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "watters_2026_item_count_drift",
        "dataset": "Watters et al. 2026 OSF vyw49 processed spike cache",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "metadata_decision": {
            "raw_dandi_000620_downloaded": False,
            "source_inclusion_reused": True,
            "independent_unit": "session nested in animal, triangle and ring analyzed separately",
            "simultaneous_population": True,
            "go_no_go": "go",
        },
        "preprocessing": "source trials and significant DMFC/FEF good+mua units; units spanning every source trial; unsmoothed non-overlapping bins; opposite-fold fixed normalization and PCA",
        "estimand": {
            "total_diffusion": "sum of process diffusion across four cross-fitted PCs, standardized latent variance/s",
            "diffusion_per_item": "total diffusion divided by remembered item count",
            "effective_diffusive_dimensions": "participation ratio of positive component diffusion values",
        },
        "sessions": sessions,
        "group": group_summary(sessions),
        "limitations": [
            "Only two monkeys are available; every effect is animal-specific.",
            "PC diffusion is in within-session standardized neural coordinates and is not compared in absolute units across animals.",
            "This is a project-specific extension, not a statistic reported by the source paper.",
            "Component diffusion is reported when the state-space fit converges; confinement identifiability is recorded separately and is not silently required for process-variance estimation.",
        ],
    }
    suffix = "" if args.limit_sessions is None else f"_smoke{args.limit_sessions}"
    destination = ROOT / "results" / f"watters_2026_item_count_drift{suffix}.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"n_sessions": len(sessions), "group": output["group"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
