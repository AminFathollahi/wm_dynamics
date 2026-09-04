#!/usr/bin/env python3
"""Panichello et al. 2024 source-style reproduction and drift/switching test.

The public Dryad release contains 25 simultaneous lPFC sessions.  This runner
first records the released inclusion counts and transparent operational
replications of intermittent above-chance cue confidence and cue-selective
units.  It then compares confined drift with a probabilistic two-state
switching AR-HMM on identical held-out trials.  The switching model has a
learned Markov transition matrix; it is not mislabeled as a Poisson rSLDS.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import (  # noqa: E402
    compare_switching_models,
    fit_gaussian_state_space,
    fit_ou_moments,
    fit_switching_ar_hmm,
    gaussian_state_space_conditional_log_likelihood,
    gaussian_state_space_log_likelihood,
    leave_one_out_condition_residuals,
    summarize_switching_decompositions,
    switching_ar_hmm_log_likelihood,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402

BIN_MS = 50
DELAY_WINDOW_MS = (300, 1450)
N_COMPONENTS = 8
N_SPLITS = 5
MIN_UNITS = 15
MIN_TRIALS_PER_CUE = 5
SEED = 20260731


def data_directory() -> Path:
    config = json.loads((ROOT / "config" / "datasets.json").read_text())
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = Path(root) / config["datasets"]["panichello_2024"]["local_path"]
    if not path.is_dir():
        raise SystemExit(f"Panichello 2024 data not staged at {path}")
    return path


def monkey_for_session(stem: str) -> str:
    year = int(stem[:2])
    return {21: "A", 22: "H", 24: "J"}.get(year, "unknown")


def bin_spikes(spikes: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    """Return unsmoothed non-overlapping trial × unit × time counts."""
    starts = np.arange(DELAY_WINDOW_MS[0], DELAY_WINDOW_MS[1], BIN_MS)
    rows = []
    for start in starts:
        mask = (time_ms >= start) & (time_ms < start + BIN_MS)
        rows.append(np.sum(spikes[:, mask, :], axis=1))
    return np.stack(rows, axis=2).astype(float)


def discriminant_direction(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(values, labels)
    _left, _singular, right = np.linalg.svd(model.coef_, full_matrices=False)
    direction = right[0]
    return direction / max(np.linalg.norm(direction), 1e-12)


def train_test_residuals(
    train_state: np.ndarray,
    test_state: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_projection = train_state @ direction
    test_projection = test_state @ direction
    train_residual, _ = leave_one_out_condition_residuals(train_projection, train_labels)
    test_residual = np.full_like(test_projection, np.nan)
    for label in np.unique(test_labels):
        train_mask = train_labels == label
        test_mask = test_labels == label
        if np.any(train_mask):
            test_residual[test_mask] = test_projection[test_mask] - train_projection[train_mask].mean(axis=0)
    return train_residual, test_residual


def iid_log_likelihood(test: np.ndarray, train: np.ndarray) -> float:
    variance = max(float(np.var(train)), 1e-10)
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + test * test / variance)))


def cue_selective_units(counts: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    """Permutation-screened delay-rate cue selectivity, reported as a source-style check."""
    rates = counts.mean(axis=2)
    cue_values = np.unique(labels)

    def eta_squared(values: np.ndarray, grouping: np.ndarray) -> float:
        grand = float(np.mean(values))
        between = sum(np.sum(grouping == cue) * (np.mean(values[grouping == cue]) - grand) ** 2 for cue in cue_values)
        total = float(np.sum((values - grand) ** 2))
        return float(between / total) if total > 0 else 0.0

    observed = np.array([eta_squared(rates[:, unit], labels) for unit in range(rates.shape[1])])
    null = np.empty((200, rates.shape[1]))
    for permutation in range(len(null)):
        shuffled = rng.permutation(labels)
        null[permutation] = [eta_squared(rates[:, unit], shuffled) for unit in range(rates.shape[1])]
    threshold = np.percentile(null, 95.0, axis=0)
    selective = observed > threshold
    return {
        "n_units": int(rates.shape[1]),
        "n_cue_selective_units": int(np.sum(selective)),
        "cue_selective_fraction": float(np.mean(selective)),
        "method": "delay-rate eta-squared above the unit-specific 95th label-permutation percentile",
    }


def out_of_fold_cue_confidence(counts: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """Operational reproduction of intermittent single-trial cue-confidence epochs."""
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    confidence = np.full((len(labels), counts.shape[2]), np.nan)
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        transform = FrozenPSTHTransform().fit(counts[train])
        train_values = transform.transform(counts[train])
        test_values = transform.transform(counts[test])
        for time_index in range(counts.shape[2]):
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model.fit(train_values[:, :, time_index], labels[train])
            probabilities = model.predict_proba(test_values[:, :, time_index])
            class_index = {value: index for index, value in enumerate(model.classes_)}
            confidence[test, time_index] = [
                probabilities[row, class_index[label]]
                for row, label in enumerate(labels[test])
            ]
    chance = 1.0 / len(np.unique(labels))
    above = confidence > chance
    transitions = np.diff(above.astype(int), axis=1)
    return {
        "mean_true_cue_probability": float(np.nanmean(confidence)),
        "chance_probability": float(chance),
        "fraction_trial_bins_above_chance": float(np.nanmean(above)),
        "fraction_trials_with_on_and_off_bins": float(np.mean(np.any(above, axis=1) & np.any(~above, axis=1))),
        "mean_on_transitions_per_trial": float(np.mean(np.sum(transitions == 1, axis=1))),
        "mean_off_transitions_per_trial": float(np.mean(np.sum(transitions == -1, axis=1))),
        "method": "five-fold out-of-fold shrinkage-LDA true-cue probability in unsmoothed 50-ms bins",
        "reproduction_scope": "operational reproduction of intermittent confidence; the public source repository supplies cluster correction but not the complete decoder wrapper",
    }


def analyze_session(path: Path) -> dict:
    raw = loadmat(path, squeeze_me=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    labels = np.asarray(raw["cueAngIdx"], dtype=int).reshape(-1)
    correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
    counts_all = bin_spikes(spikes, time_ms)
    firing_rate_hz = counts_all.sum(axis=(0, 2)) / (len(counts_all) * counts_all.shape[2] * BIN_MS / 1000.0)
    unit_mask = firing_rate_hz >= 0.2
    counts_all = counts_all[:, unit_mask]
    keep = correct
    counts = counts_all[keep]
    labels_kept = labels[keep]
    cue_counts = {int(cue): int(np.sum(labels_kept == cue)) for cue in np.unique(labels_kept)}
    base = {
        "animal": monkey_for_session(path.stem),
        "n_trials_released": int(len(labels)),
        "n_correct_trials": int(np.sum(correct)),
        "accuracy": float(np.mean(correct)),
        "n_units_released": int(spikes.shape[2]),
        "n_units_after_rate_qc": int(np.sum(unit_mask)),
        "cue_counts_correct": cue_counts,
    }
    if np.sum(unit_mask) < MIN_UNITS:
        return {**base, "status": "excluded", "reason": "fewer than 15 units after prospective rate QC"}
    if len(cue_counts) < 8 or min(cue_counts.values()) < MIN_TRIALS_PER_CUE:
        return {**base, "status": "excluded", "reason": "insufficient correct trials per cue for five folds"}

    seed = SEED + int(path.stem)
    source_checks = {
        "intermittent_confidence": out_of_fold_cue_confidence(counts, labels_kept, seed),
        "cue_specific_ensembles": cue_selective_units(counts, labels_kept, np.random.default_rng(seed)),
    }
    splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels_kept)), labels_kept)):
        transform = FrozenPSTHTransform().fit(counts[train])
        train_standard = transform.transform(counts[train]).transpose(0, 2, 1)
        test_standard = transform.transform(counts[test]).transpose(0, 2, 1)
        pca = PCA(
            n_components=min(N_COMPONENTS, train_standard.shape[2], len(train) - 1),
            svd_solver="full",
        )
        pca.fit(train_standard.reshape(-1, train_standard.shape[2]))
        train_state = pca.transform(train_standard.reshape(-1, train_standard.shape[2])).reshape(len(train), counts.shape[2], -1)
        test_state = pca.transform(test_standard.reshape(-1, test_standard.shape[2])).reshape(len(test), counts.shape[2], -1)
        direction = discriminant_direction(train_state.mean(axis=1), labels_kept[train])
        train_residual, test_residual = train_test_residuals(
            train_state, test_state, labels_kept[train], labels_kept[test], direction,
        )
        state_space = fit_gaussian_state_space(train_residual, BIN_MS / 1000.0)
        moments = fit_ou_moments(
            train_residual, BIN_MS / 1000.0, n_boot=60,
            rng=np.random.default_rng(seed + 1000 + fold),
        )
        switching_one = fit_switching_ar_hmm(
            train_residual, n_states=1, n_restarts=2,
            rng=np.random.default_rng(seed + 2000 + fold),
        )
        switching_comparison = compare_switching_models(
            train_residual, test_residual, BIN_MS / 1000.0, n_restarts=4,
            rng=np.random.default_rng(seed + 3000 + fold),
        )
        switching_two = switching_comparison["free"]
        n_observations = int(test_residual.size)
        m0 = iid_log_likelihood(test_residual, train_residual) / n_observations
        m2 = gaussian_state_space_log_likelihood(test_residual, state_space, BIN_MS / 1000.0) / n_observations
        m2_transition = gaussian_state_space_conditional_log_likelihood(
            test_residual, state_space, BIN_MS / 1000.0,
        ) / max(test_residual[:, 1:].size, 1)
        m4_one = switching_ar_hmm_log_likelihood(test_residual, switching_one) / max(test_residual[:, 1:].size, 1)
        m4_two = switching_comparison["free_log_likelihood_per_transition"]
        folds.append({
            "fold": fold,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "M0_log_likelihood_per_observation": m0,
            "M2_log_likelihood_per_observation": m2,
            "M2_log_likelihood_per_transition_conditional": m2_transition,
            "M4_one_state_log_likelihood_per_transition": m4_one,
            "M4_two_state_log_likelihood_per_transition": m4_two,
            "M2_minus_M0": m2 - m0,
            "M4_two_minus_one": m4_two - m4_one,
            "state_space": state_space.to_dict(),
            "moments": moments.to_dict(),
            "switching_one": switching_one,
            "switching_two": switching_two,
            "switching_decomposition": switching_comparison["free_decomposition"],
            "switching_tied_variance": switching_comparison["tied_variance"],
            "M4_tied_log_likelihood_per_transition": switching_comparison[
                "tied_log_likelihood_per_transition"
            ],
            "heteroscedastic_drift": switching_comparison["heteroscedastic_drift"],
            "heteroscedastic_drift_log_likelihood_per_transition": switching_comparison[
                "heteroscedastic_drift_log_likelihood_per_transition"
            ],
            "M4_two_minus_heteroscedastic_drift": (
                m4_two
                - switching_comparison["heteroscedastic_drift_log_likelihood_per_transition"]
            ),
        })
    return {
        **base,
        "status": "complete",
        "source_reproduction": source_checks,
        "folds": folds,
        "summary": {
            "M2_minus_M0_nats_per_observation": float(np.mean([row["M2_minus_M0"] for row in folds])),
            "M4_two_minus_one_nats_per_transition": float(np.mean([row["M4_two_minus_one"] for row in folds])),
            "M4_two_minus_M2_nats_per_transition": float(np.mean([
                row["M4_two_state_log_likelihood_per_transition"]
                - row["M2_log_likelihood_per_transition_conditional"]
                for row in folds
            ])),
            "M4_tied_minus_M2_nats_per_transition": float(np.mean([
                row["M4_tied_log_likelihood_per_transition"]
                - row["M2_log_likelihood_per_transition_conditional"] for row in folds
            ])),
            "M4_two_minus_heteroscedastic_drift_nats_per_transition": float(np.mean([
                row["M4_two_minus_heteroscedastic_drift"] for row in folds
            ])),
            "state_space_identifiable_folds": int(sum(row["state_space"]["status"] == "identifiable" for row in folds)),
            "moment_identifiable_folds": int(sum(row["moments"]["status"] == "identifiable" for row in folds)),
            "switching_two_state_complete_folds": int(sum(row["switching_two"]["status"] == "complete" for row in folds)),
        },
    }


def bootstrap(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draws = np.mean(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def main() -> None:
    directory = data_directory()
    files = sorted(directory.glob("*.mat"))
    sessions = {}
    for path in files:
        print(f"fitting Panichello {path.stem}", flush=True)
        sessions[path.stem] = analyze_session(path)
    complete = {key: row for key, row in sessions.items() if row["status"] == "complete"}
    if not complete:
        raise SystemExit("no Panichello session completed")
    rng = np.random.default_rng(SEED)
    metrics = {}
    for name in (
        "M2_minus_M0_nats_per_observation",
        "M4_two_minus_one_nats_per_transition",
        "M4_two_minus_M2_nats_per_transition",
        "M4_tied_minus_M2_nats_per_transition",
        "M4_two_minus_heteroscedastic_drift_nats_per_transition",
    ):
        values = np.array([row["summary"][name] for row in complete.values()])
        by_animal = {
            animal: float(np.mean([
                row["summary"][name] for row in complete.values() if row["animal"] == animal
            ]))
            for animal in sorted({row["animal"] for row in complete.values()})
        }
        metrics[name] = {
            "mean": float(np.mean(values)),
            "ci_session_bootstrap": bootstrap(values, rng),
            "n_sessions": int(len(values)),
            "by_animal": by_animal,
        }
    direct = metrics["M4_two_minus_M2_nats_per_transition"]
    tied = metrics["M4_tied_minus_M2_nats_per_transition"]
    heteroscedastic = metrics["M4_two_minus_heteroscedastic_drift_nats_per_transition"]
    metrics["switching_decomposition"] = summarize_switching_decompositions([
        fold for session in complete.values() for fold in session["folds"]
    ])
    if (
        tied["mean"] > 0.01
        and tied["ci_session_bootstrap"][0] > 0.0
        and heteroscedastic["mean"] > 0.01
        and heteroscedastic["ci_session_bootstrap"][0] > 0.0
    ):
        adjudication = "dynamics_switching_candidate_pending_fitted_model_recovery"
    elif direct["mean"] > 0.01:
        adjudication = "free_switching_advantage_does_not_exclude_noise_scale_explanation"
    elif direct["ci_session_bootstrap"][1] < -0.01:
        adjudication = "confined_drift_supported_over_switching"
    else:
        adjudication = "drift_switching_difference_does_not_clear_practical_threshold"
    output = {
        "schema_version": "1.0.0",
        "analysis_id": "panichello_2024_drift_switching",
        "dataset": "Panichello et al. 2024 Dryad 10.5061/dryad.kkwh70sct",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "metadata_decision": {
            "release_files": int(len(files)),
            "expected_sessions": 25,
            "animals": {animal: int(sum(row["animal"] == animal for row in sessions.values())) for animal in ("A", "H", "J")},
            "simultaneous_population": True,
            "intervention": False,
            "independent_unit": "session nested in three macaques",
            "go_no_go": "go" if len(files) == 25 else "release_count_mismatch",
        },
        "preprocessing": "correct trials; unsmoothed non-overlapping 50-ms counts; fold-local transform/PCA/content axis",
        "sessions": sessions,
        "group": metrics,
        "adjudication": adjudication,
        "model_status": {
            "M0": "scored",
            "M1": "not scored -- rotation adjudication is prespecified for DANDI 000469, DANDI 000574, and Miller",
            "M2": "scored",
            "M3": "not scored -- M1 is outside this external switching falsification arm",
            "M4": "scored as free- and tied-variance probabilistic two-state Gaussian AR-HMM fits, with a heteroscedastic one-state drift control; fitted-model recovery is reported separately",
        },
        "limitations": [
            "macaque spatial WM is an external falsification test, not a human replication",
            "the source repository exposes cluster correction but not the complete decoder wrapper, so the On/Off check is an operational reproduction",
            "M4 is a probabilistic Gaussian AR-HMM with explicit state transitions, not a Poisson rSLDS with separate observation noise",
            "the direct M2-versus-M4 score conditions both models on the first bin and scores the identical held-out transitions",
        ],
    }
    destination = ROOT / "results" / "panichello_2024_drift_switching.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"n_complete": len(complete), "group": metrics, "adjudication": adjudication}, indent=2))


if __name__ == "__main__":
    main()
