#!/usr/bin/env python3
"""Leakage-free confined-drift analysis for the Kai Miller ECoG N-back dataset.

Task-generality arm of the human drift spine (see scripts/run_human_drift_spine_000469.py,
the structural template this script follows): the same leave-one-out
condition-residual, Gaussian-state-space / OU-moment confined-drift estimator
is applied here to a structurally different task (continuous N-back
house-repetition detection, no explicit Sternberg-style maintenance delay)
and a structurally different recording modality (lateral subdural ECoG
high-gamma power instead of single-unit spike counts).

Condition factor: N-back load level (0/1/2), the closest analogue this task
has to Sternberg's memory-set-size load. The analysis window is the
stimulus-locked epoch (-0.2 to +1.5 s) built by src/preprocessing.py's
Miller helpers, since N-back has no explicit delay period -- items must be
held across the inter-stimulus interval to the next comparison, not across a
cued maintenance window, so this is an honest task-structure difference from
the Sternberg spine, not an oversight.

Only 4 patients exist in this release. Per-patient point estimates and a
within-patient trial-cluster bootstrap are reported; no cross-patient pooled
interval is computed (4 patients is not a population sample).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import (  # noqa: E402
    compare_switching_models,
    fit_gaussian_state_space,
    fit_ou_moments,
    fit_rotation_drift_comparison,
    gaussian_state_space_conditional_log_likelihood,
    gaussian_state_space_log_likelihood,
    leave_one_out_condition_residuals,
    summarize_switching_decompositions,
)
from preprocessing import (  # noqa: E402
    DATA_DIR,
    MILLER_DATA_DIR,
    SUBJECTS,
    epoch_data,
    high_gamma_power,
    load_subject,
    preprocess,
    reject_bad_channels,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402

BIN_MS = 100
PRE_MS = 200.0
POST_MS = 1500.0
N_COMPONENTS = 8
N_SPLITS = 5
AXIS_WINDOW = (0.2, 1.0)
LOCS_DIR = MILLER_DATA_DIR / "memory_nback" / "memory_nback" / "locs"
RULE_PATH = ROOT / "preregistration" / "rotation_drift_decision_rule.json"
RULE_HASH = "c9505c80aed6b6c82494e472991a519c46a60a00bd8bfab7e6375f0706dc0ecd"

PREPROCESSING_PROVENANCE = {
    "author_guidance_checked": (
        "README_memory_nback_dataset_notes.docx was read in full "
        "(word/document.xml extraction); it describes task structure, "
        "variable layout, and electrode localization only -- it gives no "
        "preprocessing, referencing, filtering, or QC guidance of any kind."
    ),
    "substitute_applied": (
        "literature-standard ECoG QC substituted in its absence: MAD-based "
        "bad-channel rejection (median absolute deviation, robust-stats "
        "standard, threshold=3), common-average re-reference (Engel et al. "
        "2005), 60 Hz US line-noise notch + 4 harmonics, and Hilbert-envelope "
        "high-gamma (70-150 Hz) power (Crone et al. 1998; Ray & Maunsell "
        "2011) -- as already implemented in src/preprocessing.py. This is a "
        "literature-standard substitute, not an author instruction."
    ),
    "no_smoothing_for_drift_fit": (
        "high_gamma_power is called with smooth_ms=0 for this analysis; "
        "drift_dynamics.py requires unsmoothed observations binned into "
        "non-overlapping windows, so the module's default Gaussian smoothing "
        "kernel is disabled and replaced by simple block-averaging into "
        f"{BIN_MS} ms bins."
    ),
}

REQUIRED_ETHICS_STATEMENT = (
    "All patients participated in a purely voluntary manner, after providing "
    "informed written consent, under experimental protocols approved by the "
    "Institutional Review Board of the University of Washington (#12193). "
    "All patient data was anonymized according to IRB protocol, in "
    "accordance with HIPAA mandate. It was made available through the "
    "library described in “A Library of Human Electrocorticographic "
    "Data and Analyses” by Kai Miller, freely available at "
    "https://searchworks.stanford.edu/view/zk881ps0522"
)


def discriminant_direction(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Dominant multiclass shrinkage-LDA direction with deterministic sign."""
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(values, labels)
    _, _, right = np.linalg.svd(model.coef_, full_matrices=False)
    direction = right[0]
    class_projection = np.array([
        np.mean(values[labels == label] @ direction) for label in model.classes_
    ])
    if np.corrcoef(np.arange(len(class_projection)), class_projection)[0, 1] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def projected_residuals(
    train_state: np.ndarray,
    test_state: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_projection = train_state @ direction
    test_projection = test_state @ direction
    train_residuals, _ = leave_one_out_condition_residuals(train_projection, train_labels)
    test_residuals = np.full_like(test_projection, np.nan)
    for label in np.unique(test_labels):
        train_rows = train_labels == label
        test_rows = test_labels == label
        if not np.any(train_rows):
            continue
        test_residuals[test_rows] = test_projection[test_rows] - train_projection[train_rows].mean(axis=0)
    return train_residuals, test_residuals


def matched_complement_direction(
    state_window: np.ndarray,
    labels: np.ndarray,
    content_direction: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int = 200,
) -> tuple[np.ndarray, float, float]:
    class_means = np.vstack([
        state_window[labels == label].mean(axis=0) for label in np.unique(labels)
    ])
    target_variance = float(np.var(class_means @ content_direction))
    candidates = rng.normal(size=(n_candidates, state_window.shape[1]))
    candidates -= (candidates @ content_direction)[:, None] * content_direction[None, :]
    norms = np.linalg.norm(candidates, axis=1)
    candidates = candidates[norms > 1e-10] / norms[norms > 1e-10, None]
    variances = np.var(class_means @ candidates.T, axis=0)
    selected = int(np.argmin(np.abs(variances - target_variance)))
    return candidates[selected], target_variance, float(variances[selected])


def iid_log_likelihood(test: np.ndarray, train: np.ndarray) -> float:
    variance = max(float(np.nanvar(train)), 1e-10)
    values = test[np.isfinite(test)]
    return float(np.sum(-0.5 * (np.log(2 * np.pi * variance) + values * values / variance)))


def trial_prediction_advantage(train: np.ndarray, test: np.ndarray) -> dict:
    x_train = train[:, :-1].reshape(-1)
    y_train = train[:, 1:].reshape(-1)
    finite_train = np.isfinite(x_train) & np.isfinite(y_train)
    design = np.column_stack((np.ones(finite_train.sum()), x_train[finite_train]))
    intercept, coefficient = np.linalg.lstsq(design, y_train[finite_train], rcond=None)[0]
    x_test = test[:, :-1].reshape(-1)
    y_test = test[:, 1:].reshape(-1)
    finite_test = np.isfinite(x_test) & np.isfinite(y_test)
    prediction = intercept + coefficient * x_test[finite_test]
    baseline_error = np.sum(y_test[finite_test] ** 2)
    model_error = np.sum((y_test[finite_test] - prediction) ** 2)
    return {
        "coefficient": float(coefficient),
        "held_out_r2_advantage": float(1.0 - model_error / max(baseline_error, 1e-12)),
    }


def bin_time_axis(epochs_ct: np.ndarray, times: np.ndarray, bin_ms: float, srate: float) -> tuple[np.ndarray, np.ndarray]:
    """Block-average the trailing time axis into non-overlapping bins.

    Unlike a smoothing kernel, non-overlapping block averaging does not
    introduce cross-bin autocorrelation, so it stays compatible with
    drift_dynamics.py's unsmoothed-observation requirement.
    """
    bin_samples = max(int(round(bin_ms * srate / 1000.0)), 1)
    n_bins = epochs_ct.shape[-1] // bin_samples
    trimmed = epochs_ct[..., : n_bins * bin_samples]
    binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
    time_trimmed = times[: n_bins * bin_samples]
    bin_times = time_trimmed.reshape(n_bins, bin_samples).mean(axis=-1)
    return binned, bin_times


def fit_condition_drift(
    epochs_ct: np.ndarray,
    labels: np.ndarray,
    axis_bins: np.ndarray,
    dt: float,
    seed: int,
    n_splits: int = N_SPLITS,
    n_components: int = N_COMPONENTS,
) -> dict:
    """Leakage-free CV confined-drift fit, condition = ``labels``.

    ``epochs_ct`` is (trials, channels, time_bins), unnormalized and
    unsmoothed. Standardization, PCA, the content-discriminant direction,
    and the permuted-axis/matched-complement nulls are all fit on the
    outer-training fold only per split. Also returns a descriptive
    in-sample fit (all trials, no held-out split) whose only purpose is a
    percentile bootstrap confidence interval over this unit's own trials --
    it is not held-out evidence and is labelled as such.
    """
    n_trials, n_channels, n_time = epochs_ct.shape
    n_components = min(n_components, n_channels, n_trials - n_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(n_trials), labels)):
        transform = FrozenPSTHTransform().fit(epochs_ct[train_index])
        train_standardized = transform.transform(epochs_ct[train_index]).transpose(0, 2, 1)
        test_standardized = transform.transform(epochs_ct[test_index]).transpose(0, 2, 1)
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(train_standardized.reshape(-1, train_standardized.shape[-1]))
        train_state = pca.transform(train_standardized.reshape(-1, train_standardized.shape[-1])).reshape(
            len(train_index), n_time, -1,
        )
        test_state = pca.transform(test_standardized.reshape(-1, test_standardized.shape[-1])).reshape(
            len(test_index), n_time, -1,
        )
        train_window = train_state[:, axis_bins].mean(axis=1)
        direction = discriminant_direction(train_window, labels[train_index])
        train_residuals, test_residuals = projected_residuals(
            train_state, test_state, labels[train_index], labels[test_index], direction,
        )
        state_estimate = fit_gaussian_state_space(train_residuals, dt)
        moment_estimate = fit_ou_moments(
            train_residuals, dt, n_boot=120,
            rng=np.random.default_rng(seed + fold_index + 1000),
        )
        m2_log_likelihood = gaussian_state_space_log_likelihood(test_residuals, state_estimate, dt)
        m2_transition_log_likelihood = gaussian_state_space_conditional_log_likelihood(
            test_residuals, state_estimate, dt,
        )
        switching_comparison = compare_switching_models(
            train_residuals, test_residuals, dt, n_restarts=4,
            rng=np.random.default_rng(seed + fold_index + 6000),
        )
        m4_transition_score = switching_comparison["free_log_likelihood_per_transition"]
        m0_log_likelihood = iid_log_likelihood(test_residuals, train_residuals)
        rotation_comparison = fit_rotation_drift_comparison(
            train_state, test_state, labels[train_index], labels[test_index], direction,
            train_residuals, test_residuals,
            m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1), dt,
            rng=np.random.default_rng(seed + fold_index + 7000),
        )
        prediction = trial_prediction_advantage(train_residuals, test_residuals)

        permutation_rng = np.random.default_rng(seed + fold_index + 2000)
        permuted_labels = permutation_rng.permutation(labels[train_index])
        permuted_direction = discriminant_direction(train_window, permuted_labels)
        permuted_train, _ = projected_residuals(
            train_state, test_state, permuted_labels, labels[test_index], permuted_direction,
        )
        permuted_estimate = fit_ou_moments(
            permuted_train, dt, n_boot=80,
            rng=np.random.default_rng(seed + fold_index + 3000),
        )
        complement, target_signal_variance, complement_signal_variance = matched_complement_direction(
            train_window, labels[train_index], direction,
            np.random.default_rng(seed + fold_index + 4000),
        )
        complement_train, _ = projected_residuals(
            train_state, test_state, labels[train_index], labels[test_index], complement,
        )
        complement_estimate = fit_ou_moments(
            complement_train, dt, n_boot=80,
            rng=np.random.default_rng(seed + fold_index + 5000),
        )
        fold_rows.append({
            "fold": fold_index,
            "n_train": int(len(train_index)), "n_test": int(len(test_index)),
            "state_space": state_estimate.to_dict(), "moment": moment_estimate.to_dict(),
            "permuted_axis_moment": permuted_estimate.to_dict(),
            "matched_complement_moment": complement_estimate.to_dict(),
            "content_signal_variance": target_signal_variance,
            "complement_signal_variance": complement_signal_variance,
            "m0_log_likelihood_per_observation": m0_log_likelihood / test_residuals.size,
            "m2_log_likelihood_per_observation": m2_log_likelihood / test_residuals.size,
            "m2_minus_m0_nats_per_observation": (m2_log_likelihood - m0_log_likelihood) / test_residuals.size,
            "m2_log_likelihood_per_transition_conditional": (
                m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1)
            ),
            "m4_log_likelihood_per_transition": (
                m4_transition_score
            ),
            "m4_minus_m2_nats_per_transition": (
                m4_transition_score
                - m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1)
            ),
            "switching_two_state": switching_comparison["free"],
            "rotation_comparison": rotation_comparison,
            "m1_minus_m0_nats_per_observation": rotation_comparison[
                "m1_minus_m0_nats_per_observation"
            ],
            "m3_minus_m2_nats_per_transition": rotation_comparison[
                "m3_minus_m2_nats_per_transition"
            ],
            "switching_decomposition": switching_comparison["free_decomposition"],
            "switching_tied_variance": switching_comparison["tied_variance"],
            "m4_tied_log_likelihood_per_transition": switching_comparison[
                "tied_log_likelihood_per_transition"
            ],
            "heteroscedastic_drift": switching_comparison["heteroscedastic_drift"],
            "heteroscedastic_drift_log_likelihood_per_transition": switching_comparison[
                "heteroscedastic_drift_log_likelihood_per_transition"
            ],
            "m4_minus_heteroscedastic_drift_nats_per_transition": (
                m4_transition_score
                - switching_comparison["heteroscedastic_drift_log_likelihood_per_transition"]
            ),
            "trial_prediction": prediction,
        })

    def identified(field: str, estimator: str) -> np.ndarray:
        return np.array([
            row[estimator][field] for row in fold_rows if row[estimator]["status"] == "identifiable"
        ])

    state_lambdas = identified("lambda_rate", "state_space")
    moment_lambdas = identified("lambda_rate", "moment")
    state_diffusion = identified("diffusion", "state_space")
    moment_diffusion = identified("diffusion", "moment")
    content_minus_permuted = np.array([
        row["moment"]["lambda_rate"] - row["permuted_axis_moment"]["lambda_rate"]
        for row in fold_rows
        if row["moment"]["status"] == "identifiable" and row["permuted_axis_moment"]["status"] == "identifiable"
    ])
    content_minus_complement = np.array([
        row["moment"]["lambda_rate"] - row["matched_complement_moment"]["lambda_rate"]
        for row in fold_rows
        if row["moment"]["status"] == "identifiable" and row["matched_complement_moment"]["status"] == "identifiable"
    ])

    # Descriptive in-sample fit (all trials, single non-held-out split) --
    # its only purpose is a within-unit trial-cluster percentile bootstrap
    # confidence interval, never a held-out claim.
    transform_full = FrozenPSTHTransform().fit(epochs_ct)
    standardized_full = transform_full.transform(epochs_ct).transpose(0, 2, 1)
    pca_full = PCA(n_components=n_components, svd_solver="full")
    pca_full.fit(standardized_full.reshape(-1, standardized_full.shape[-1]))
    state_full = pca_full.transform(standardized_full.reshape(-1, standardized_full.shape[-1])).reshape(
        n_trials, n_time, -1,
    )
    window_full = state_full[:, axis_bins].mean(axis=1)
    direction_full = discriminant_direction(window_full, labels)
    projection_full = state_full @ direction_full
    residuals_full, _ = leave_one_out_condition_residuals(projection_full, labels)
    descriptive_estimate = fit_ou_moments(
        residuals_full, dt, n_boot=500, rng=np.random.default_rng(seed + 9999),
    )

    return {
        "n_trials": int(n_trials), "n_channels": int(n_channels), "n_time_bins": int(n_time),
        "n_components": int(n_components), "dt": float(dt),
        "folds": fold_rows,
        "summary": {
            "state_space_lambda_identified_mean": float(np.mean(state_lambdas)) if len(state_lambdas) else None,
            "moment_lambda_identified_mean": float(np.mean(moment_lambdas)) if len(moment_lambdas) else None,
            "state_space_diffusion_identified_mean": float(np.mean(state_diffusion)) if len(state_diffusion) else None,
            "moment_diffusion_identified_mean": float(np.mean(moment_diffusion)) if len(moment_diffusion) else None,
            "state_space_identifiable_folds": int(len(state_lambdas)),
            "moment_identifiable_folds": int(len(moment_lambdas)),
            "lambda_content_minus_permuted_identified_mean": (
                float(np.mean(content_minus_permuted)) if len(content_minus_permuted) else None
            ),
            "lambda_content_minus_permuted_identified_folds": int(len(content_minus_permuted)),
            "lambda_content_minus_complement_identified_mean": (
                float(np.mean(content_minus_complement)) if len(content_minus_complement) else None
            ),
            "lambda_content_minus_complement_identified_folds": int(len(content_minus_complement)),
            "m2_minus_m0_nats_per_observation": float(np.mean([
                row["m2_minus_m0_nats_per_observation"] for row in fold_rows
            ])),
            "m4_minus_m2_nats_per_transition": float(np.mean([
                row["m4_minus_m2_nats_per_transition"] for row in fold_rows
            ])),
            "m1_minus_m0_nats_per_observation": float(np.mean([
                row["m1_minus_m0_nats_per_observation"] for row in fold_rows
            ])),
            "m3_minus_m2_nats_per_transition": float(np.mean([
                row["m3_minus_m2_nats_per_transition"] for row in fold_rows
            ])),
            "counter_rotation_accuracy_recovery": float(np.mean([
                row["rotation_comparison"]["counter_rotation"]["accuracy_recovery"]
                for row in fold_rows
            ])),
            "m4_tied_minus_m2_nats_per_transition": float(np.mean([
                row["m4_tied_log_likelihood_per_transition"]
                - row["m2_log_likelihood_per_transition_conditional"] for row in fold_rows
            ])),
            "m4_minus_heteroscedastic_drift_nats_per_transition": float(np.mean([
                row["m4_minus_heteroscedastic_drift_nats_per_transition"] for row in fold_rows
            ])),
            "switching_complete_folds": int(sum(
                row["switching_two_state"]["status"] == "complete" for row in fold_rows
            )),
            "trial_prediction_r2_advantage": float(np.mean([
                row["trial_prediction"]["held_out_r2_advantage"] for row in fold_rows
            ])),
        },
        "descriptive_in_sample_bootstrap": {
            "caveat": (
                "content direction fit on all trials (not held out); this block "
                "exists only to report a within-patient trial-cluster percentile "
                "bootstrap interval, per the small-N instruction to avoid a "
                "cross-patient pooled CI -- it is NOT a held-out estimate and must "
                "not be read as confirmatory evidence alongside the CV folds above"
            ),
            "estimate": descriptive_estimate.to_dict(),
        },
    }


def response_signal_status(subj: str) -> dict:
    """Verify, against the actual .mat file, whether response/accuracy data exists.

    The dataset notes describe a "response" (time x 1) finger-flexion
    variable, but the actual file stores a continuous 5-channel dataglove
    trace, not a trial-level correct/error or RT label -- and it is entirely
    absent for one of the four patients.
    """
    raw = sio.loadmat(str(DATA_DIR / f"{subj}_nback.mat"))
    if "response" not in raw:
        return {
            "present": False,
            "note": "no response/dataglove field in this patient's .mat file",
        }
    resp = raw["response"]
    return {
        "present": True,
        "shape": list(resp.shape),
        "n_nonzero_samples": int(np.count_nonzero(resp)),
        "note": (
            "raw continuous dataglove finger-flexion sensor stream "
            "(time x n_fingers), NOT a precomputed trial-level accuracy or "
            "response-time label. Deriving hit/miss/false-alarm labels would "
            "require new threshold-detection and target-window-alignment "
            "engineering that has not been built here; using it directly as "
            "an accuracy proxy without that validation would be fabrication. "
            "Behaviour arm is therefore still not directly buildable, but the "
            "underlying signal is present for this patient (correcting the "
            "blanket 'not present' characterization for this subset)."
        ),
    }


def electrode_coverage(subj: str) -> dict:
    """Talairach electrode extent; ECoG grids/strips are subdural by construction.

    Subdural ECoG electrodes sit on the cortical surface and cannot reach
    medial temporal lobe structures (hippocampus, amygdala, entorhinal
    cortex) the way the depth macro/microelectrodes in DANDI 000469/001187
    do. This is a modality fact, not something that needs an anatomical
    atlas lookup to establish.
    """
    raw = sio.loadmat(str(LOCS_DIR / f"{subj}_electrodes.mat"))
    coords = raw["electrodes"]
    return {
        "n_electrodes": int(coords.shape[0]),
        "talairach_x_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
        "talairach_y_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
        "talairach_z_range": [float(coords[:, 2].min()), float(coords[:, 2].max())],
        "note": (
            "subdural ECoG grid/strip electrodes; by recording modality these "
            "cannot sample medial temporal lobe structures, so no "
            "within-dataset lateral-cortex-vs-MTL contrast is possible here "
            "(unlike Boran, which co-locates iEEG and MTL depth electrodes in "
            "the same patients). The comparison this dataset supports is "
            "cross-dataset: this lateral-cortex-only confinement estimate "
            "against the MTL-inclusive DANDI 000469/001187 estimates."
        ),
    }


def analyze_patient(subj: str, seed: int) -> dict:
    d = load_subject(subj)
    good = reject_bad_channels(d["data"], srate=d["srate"], mains_hz=60.0)
    data_clean = preprocess(d["data"][:, good], srate=d["srate"])
    hgp = high_gamma_power(data_clean, srate=d["srate"], smooth_ms=0.0)
    ep = epoch_data(hgp, d["stim"], d["task"], d["target"], pre_ms=PRE_MS, post_ms=POST_MS, srate=d["srate"])
    labels_full = ep["task_id"].astype(int)
    keep = np.isin(labels_full, (0, 1, 2))
    labels = labels_full[keep]
    epochs_ct = ep["epochs"][keep].transpose(0, 2, 1)  # (N, C, T)
    counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
    if len(counts) < 3 or min(counts.values()) < N_SPLITS:
        return {"status": "excluded", "reason": "N-back load counts cannot support five folds", "counts": counts}
    binned, bin_times = bin_time_axis(epochs_ct, ep["times"], BIN_MS, d["srate"])
    axis_bins = (bin_times >= AXIS_WINDOW[0]) & (bin_times <= AXIS_WINDOW[1])
    dt = BIN_MS / 1000.0
    drift = fit_condition_drift(binned, labels, axis_bins, dt, seed=seed)
    return {
        "status": "complete",
        "n_channels_recorded": int(d["data"].shape[1]), "n_channels_kept": int(good.sum()),
        "n_trials": int(len(labels)), "load_counts": counts,
        "bin_ms": BIN_MS, "axis_window_seconds": list(AXIS_WINDOW),
        "smoothed": False, "condition": "n_back_load",
        "response_signal": response_signal_status(subj),
        "electrode_coverage": electrode_coverage(subj),
        "drift": drift,
    }


def main() -> None:
    if sha256_file(RULE_PATH) != RULE_HASH:
        raise SystemExit("frozen adjudication rule hash mismatch; refusing to fit real data")
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Miller data not staged at configured path: {DATA_DIR}")

    patients = {}
    for index, subj in enumerate(SUBJECTS):
        print(f"fitting {subj}", flush=True)
        patients[subj] = analyze_patient(subj, seed=20260731 + index)

    complete = {subj: row for subj, row in patients.items() if row["status"] == "complete"}
    if not complete:
        raise SystemExit("no Miller patient completed the drift fit")

    switching_decomposition = summarize_switching_decompositions([
        fold for patient in complete.values() for fold in patient["drift"]["folds"]
    ])

    threshold = json.loads(RULE_PATH.read_text())["minimum_effect"]["negative_log_likelihood_nats_per_observation"]
    per_patient_table = {}
    for subj, row in complete.items():
        summary = row["drift"]["summary"]
        descriptive = row["drift"]["descriptive_in_sample_bootstrap"]["estimate"]
        per_patient_table[subj] = {
            "moment_lambda_identified_mean": summary["moment_lambda_identified_mean"],
            "state_space_lambda_identified_mean": summary["state_space_lambda_identified_mean"],
            "moment_diffusion_identified_mean": summary["moment_diffusion_identified_mean"],
            "m2_minus_m0_nats_per_observation": summary["m2_minus_m0_nats_per_observation"],
            "m2_passes_practical_threshold_this_patient": bool(
                summary["m2_minus_m0_nats_per_observation"] >= threshold
            ),
            "descriptive_in_sample_lambda": descriptive["lambda_rate"],
            "descriptive_in_sample_lambda_ci_trial_cluster_bootstrap": descriptive["lambda_ci"],
            "descriptive_in_sample_status": descriptive["status"],
        }

    cross_dataset_reference_path = ROOT / "results" / "human_drift_spine_000469.json"
    cross_dataset_comparison: dict = {
        "note": (
            "qualitative comparison only, not a formal joint test; DANDI 000469 "
            "is single-unit spiking with hippocampal/MTL and lateral-temporal "
            "coverage under a repeated-item Sternberg delay, Miller is lateral "
            "subdural ECoG high-gamma power under a delay-free N-back task -- "
            "modality, region, and task all differ simultaneously, so this "
            "cannot isolate which difference (if any) drives a discrepancy"
        ),
    }
    if cross_dataset_reference_path.is_file():
        reference = json.loads(cross_dataset_reference_path.read_text())
        cross_dataset_comparison["dandi_000469_group"] = {
            "moment_lambda_identified_mean": reference["group"]["moment_lambda_identified_mean"],
            "state_space_lambda_identified_mean": reference["group"]["state_space_lambda_identified_mean"],
        }
    else:
        cross_dataset_comparison["dandi_000469_group"] = None
        cross_dataset_comparison["note"] += "; results/human_drift_spine_000469.json not found on disk"

    output = {
        "schema_version": "1.0.0", "analysis_id": "human_drift_spine_miller_nback",
        "dataset": "Kai Miller ECoG N-back (memory_nback library)",
        "canonical_role": "task-generality arm (N-back vs Sternberg); lateral-cortex ECoG vs MTL-inclusive comparison",
        "required_ethics_statement": REQUIRED_ETHICS_STATEMENT,
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "preprocessing_provenance": PREPROCESSING_PROVENANCE,
        "coordinate": "FrozenPSTHTransform and PCA fit within each outer-training fold",
        "small_n_caution": (
            "n=4 patients. Per-patient point estimates and a within-patient "
            "trial-cluster percentile bootstrap are reported below. No "
            "cross-patient pooled interval is computed -- 4 patients is a "
            "descriptive comparison, not a population-level test, and the "
            "frozen preregistration's patient-cluster winner_rule is "
            "explicitly NOT applied at the group level for this dataset."
        ),
        "patients": patients,
        "per_patient_headline": per_patient_table,
        "switching_decomposition": switching_decomposition,
        "cross_dataset_comparison": cross_dataset_comparison,
        "model_status": {
            "M0": "scored", "M1": "scored as a training-fold axis-rotation operator",
            "M2": "scored", "M3": "scored as rotation plus confined residual drift",
            "M4": "scored as a probabilistic two-state Gaussian AR-HMM; not a full Poisson rSLDS",
        },
    }
    destination = ROOT / "results" / "miller_drift_spine.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"n_complete": len(complete), "per_patient": per_patient_table, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
