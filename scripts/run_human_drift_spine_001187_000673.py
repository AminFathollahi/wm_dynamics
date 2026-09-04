#!/usr/bin/env python3
"""Leakage-free confined-drift analysis for the load-1-vs-load-3 manipulation
shared by DANDI 001187 and DANDI 000673.

Structural template: scripts/run_human_drift_spine_000469.py (repeated-item
content drift) and scripts/run_miller_drift_spine.py (the reusable
``fit_condition_drift`` / ``bin_time_axis`` pattern this script partly
reuses). Same estimators (src/drift_dynamics.py), same firing-rate/accuracy
QC and PSTH construction (src/spike_pipeline.py).

Two things make this dataset pair structurally different from 000469 and
Miller, both handled explicitly below rather than forced into the sibling
scripts' shape:

1. Identity. 001187 (46 recordings) and 000673 (44 recordings) are release
   views of the SAME patient-sessions for 37 verified overlaps (31 patients,
   corrected 2026-08-03 from a prior 19/16 undercount caused by grouping on
   the raw identifier string instead of patient+session -- see
   provenance/dataset_overlap_report.json and results/patient_identity_audit.json),
   not independent cohorts (provenance/canonical_recording_registry.json and
   provenance/canonical_primary_records.json, built by
   scripts/audit_dataset_identity.py -- consumed here, not rebuilt). The
   registry's own ``canonical_view_rule`` prefers 001187 as the primary
   spike-based view for a verified overlap and keeps 000673 as a linked
   sensitivity view. Every canonical session that has a same-native-identifier
   000673 twin (44 of the 71 canonical rows) gets that twin's hippocampal LFP
   attached as a SENSITIVITY arm, whether or not 000673 happens to be the
   primary view for that particular row.

2. Content. Pictures are novel per trial in both releases: there is no repeated item identity to fit an LDA content
   axis against, unlike 000469's fixed 5-picture set. The condition factor
   here is memory load (1 vs 3) itself, so the two load subsets are fit
   SEPARATELY (``fit_load_confinement`` below) rather than jointly the way
   ``fit_condition_drift`` fits one pooled model across item classes. Within
   one load there is no further label, so the projection axis is the leading
   within-fold PCA component (dominant mode of trial-to-trial state
   variability), not an LDA direction, and residuals are leave-one-out
   deviations from the single training-fold trial mean.

LFP arm (000673 only -- 001187 has no continuous voltage field): the release
already states its own preprocessing in the NWB acquisition metadata ("LFP
recordings that have spike potentials removed and is downsampled to 400Hz").
No further author-specified referencing/filtering guidance exists beyond
that, so the literature-standard substitute already used identically
elsewhere in this codebase applies (common-average reference, 60 Hz US
line-noise notch -- Cedars-Sinai is a US site -- and Crone et al. 1998 /
Ray & Maunsell 2011 Hilbert-envelope high-gamma power), matching
scripts/run_miller_drift_spine.py's PREPROCESSING_PROVENANCE pattern.
Hippocampal channels are selected from the electrode table's ``location``
field.

Outputs: results/human_drift_spine_001187_000673.json, containing per-session
unit-based primary fits, per-session LFP-linked sensitivity fits (clearly
labeled, paired to the same session as the primary fit, never treated as an
independent replication), a patient-cluster (sessions nested within patient)
group summary of the load3-minus-load1 lambda/diffusion difference, and its
correlation against each session's already-computed participation-ratio
load3-vs-load1 difference (results/dandi001187_summary.json /
dandi000673_summary.json's ``pr_per_load`` -- reused, not rederived).

Run:
    conda run -n wm_dynamics python scripts/run_human_drift_spine_001187_000673.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_dynamics import (  # noqa: E402
    compare_switching_models,
    discriminant_direction,
    fit_gaussian_state_space,
    fit_ou_moments,
    fit_rotation_drift_comparison,
    gaussian_state_space_conditional_log_likelihood,
    gaussian_state_space_log_likelihood,
    leave_one_out_condition_residuals,
    matched_complement_direction,
    projected_residuals,
    summarize_switching_decompositions,
    trial_prediction_advantage,
)
from preprocessing import common_average_reference, high_gamma_power, line_noise_notch  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import (  # noqa: E402
    ANATOMICAL_REGIONS,
    FrozenPSTHTransform,
    MIN_SESSION_ACCURACY,
    MIN_UNITS_PER_REGION,
    REGIONS_WITH_POOLED,
    build_psth,
    filter_units_by_region,
    load_spike_times,
    low_rate_unit_mask,
    resolve_unit_regions,
)
from statistics import (  # noqa: E402
    bootstrap_ci,
    pearson_permutation_test,
    spearman_permutation_test,
    stable_seed,
)
from run_miller_drift_spine import bin_time_axis, iid_log_likelihood  # noqa: E402

BIN_MS = 100
WINDOW_S = 2.3
N_COMPONENTS = 8
N_SPLITS = 5
MIN_UNITS = 15
MIN_TRIALS_PER_LOAD = 20
LFP_SRATE_HZ = 400.0
LFP_LINE_FREQ_HZ = 60.0  # Cedars-Sinai (Rutishauser lab) is a US site
RULE_PATH = ROOT / "preregistration" / "rotation_drift_decision_rule.json"
RULE_HASH = "c9505c80aed6b6c82494e472991a519c46a60a00bd8bfab7e6375f0706dc0ecd"
PROVENANCE_DIR = ROOT / "provenance"
RESULTS = ROOT / "results"

PREPROCESSING_PROVENANCE = {
    "unit_arm": (
        "Firing-rate and session-accuracy QC identical to run_001187_pipeline.py "
        "/ run_000673_pipeline.py (Daume et al. 2024 QC floors), unsmoothed "
        "100 ms PSTH bins (drift_dynamics.py forbids smoothing for drift fits)."
    ),
    "lfp_arm_author_guidance_checked": (
        "The NWB acquisition/LFPs ElectricalSeries description states the "
        "release's own preprocessing: spike potentials removed, downsampled "
        "to 400 Hz. No further author-specified referencing or band "
        "definition is given for this arm."
    ),
    "lfp_arm_substitute_applied": (
        "literature-standard substitute in the absence of further author "
        "instruction: common-average reference (Engel et al. 2005), 60 Hz US "
        "line-noise notch (3 harmonics; Cedars-Sinai is a US site), and "
        "Crone et al. 1998 / Ray & Maunsell 2011 Hilbert-envelope high-gamma "
        "(70-150 Hz) power -- identical to the substitute already used for "
        "Miller (scripts/run_miller_drift_spine.py) and Boran "
        "(scripts/run_multiband_analysis.py) in this codebase."
    ),
    "no_smoothing_for_drift_fit": (
        "high_gamma_power is called with smooth_ms=0.0; non-overlapping "
        f"{BIN_MS} ms block-averaging (bin_time_axis) replaces the module's "
        "default Gaussian smoothing kernel, matching the unit arm's bin width."
    ),
}


def data_root() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(f"WM_DYNAMICS_DATA_ROOT is not a directory: {path}")
    return path


def canonical_sessions(provenance_dir: Path | None = None) -> list[dict]:
    """Deduplicated 001187/000673 patient-sessions.

    Consumes provenance/canonical_primary_records.json and
    provenance/canonical_recording_registry.json (built by
    scripts/audit_dataset_identity.py) -- that identity logic is already
    correct and tested and is not rebuilt here.
    ``canonical_primary_records.json`` already applies the registry's own
    rule (prefer 001187 for a verified 001187/000673 overlap, keep 000673
    elsewhere), so every (patient, session) pair from these two releases
    appears in the returned list exactly once. Each row also carries the
    matching 000673 LFP path when one exists for that native identifier,
    independent of which release is the row's own primary view.
    """
    provenance_dir = provenance_dir or PROVENANCE_DIR
    registry = json.loads((provenance_dir / "canonical_recording_registry.json").read_text())
    primary = json.loads((provenance_dir / "canonical_primary_records.json").read_text())
    by_native: dict[str, list[dict]] = {}
    for row in registry:
        by_native.setdefault(row["native_identifier"], []).append(row)
    sessions = []
    for row in primary:
        if row["release"] not in ("001187", "000673"):
            continue
        twins = by_native.get(row["native_identifier"], [])
        lfp_twin = next((t for t in twins if t["release"] == "000673"), None)
        sessions.append({
            "patient": row["patient"], "session": row["session"],
            "primary_release": row["release"], "primary_path": row["path"],
            "lfp_path": lfp_twin["path"] if lfp_twin is not None else None,
        })
    return sorted(sessions, key=lambda s: (s["patient"], s["session"], s["primary_release"]))


def fit_load_confinement(
    epochs_ct: np.ndarray,
    dt: float,
    seed: int,
    n_splits: int = N_SPLITS,
    n_components: int = N_COMPONENTS,
) -> dict:
    """Leakage-free CV confined-drift fit for ONE load level, one session.

    Unlike ``fit_condition_drift`` (run_miller_drift_spine.py), there is no
    second within-load label to discriminate on here -- 001187/000673
    pictures are novel per trial -- so the projection axis is the leading
    within-fold PCA component rather than an LDA content direction, and
    residuals are leave-one-out deviations from the single across-trial
    training-fold mean rather than from a per-condition centroid.
    """
    n_trials, n_channels, n_time = epochs_ct.shape
    n_components = max(1, min(n_components, n_channels, n_trials - n_splits))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(np.arange(n_trials))):
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
        train_pc1 = train_state[:, :, 0]
        test_pc1 = test_state[:, :, 0]
        train_residuals, _ = leave_one_out_condition_residuals(
            train_pc1, np.zeros(len(train_index), dtype=int),
        )
        test_residuals = test_pc1 - train_pc1.mean(axis=0, keepdims=True)

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

        fold_rows.append({
            "fold": fold_index,
            "n_train": int(len(train_index)), "n_test": int(len(test_index)),
            "state_space": state_estimate.to_dict(), "moment": moment_estimate.to_dict(),
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
        })

    def identified(field: str, estimator: str) -> np.ndarray:
        return np.array([
            row[estimator][field] for row in fold_rows if row[estimator]["status"] == "identifiable"
        ])

    state_lambdas, moment_lambdas = identified("lambda_rate", "state_space"), identified("lambda_rate", "moment")
    state_diffusion, moment_diffusion = identified("diffusion", "state_space"), identified("diffusion", "moment")

    return {
        "n_trials": int(n_trials), "n_channels": int(n_channels), "n_time_bins": int(n_time),
        "n_components": int(n_components), "dt": float(dt),
        "folds": fold_rows,
        "summary": {
            "state_space_lambda_identified_mean": float(np.mean(state_lambdas)) if len(state_lambdas) else None,
            "moment_lambda_identified_mean": float(np.mean(moment_lambdas)) if len(moment_lambdas) else None,
            "state_space_diffusion_identified_mean": (
                float(np.mean(state_diffusion)) if len(state_diffusion) else None
            ),
            "moment_diffusion_identified_mean": float(np.mean(moment_diffusion)) if len(moment_diffusion) else None,
            "state_space_identifiable_folds": int(len(state_lambdas)),
            "moment_identifiable_folds": int(len(moment_lambdas)),
            "m2_minus_m0_nats_per_observation": float(np.mean([
                row["m2_minus_m0_nats_per_observation"] for row in fold_rows
            ])),
            "m4_minus_m2_nats_per_transition": float(np.mean([
                row["m4_minus_m2_nats_per_transition"] for row in fold_rows
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
        },
    }


def _trial_group(handle: h5py.File, release: str):
    return handle["intervals/WM_trials"] if release == "001187" else handle["intervals/trials"]


def _fit_by_load(epochs_ct: np.ndarray, loads: np.ndarray, dt: float, seed: int, n_components: int) -> dict:
    fits = {}
    for load in (1, 3):
        mask = loads == load
        n_trials = int(mask.sum())
        if n_trials < MIN_TRIALS_PER_LOAD:
            fits[str(load)] = {
                "status": "excluded", "n_trials": n_trials,
                "reason": f"only {n_trials} load-{load} trials (<{MIN_TRIALS_PER_LOAD})",
            }
            continue
        fit = fit_load_confinement(
            epochs_ct[mask], dt, seed=seed + load, n_splits=N_SPLITS, n_components=n_components,
        )
        fit["status"] = "complete"
        fits[str(load)] = fit
    return fits


def analyze_unit_session(path: Path, release: str, seed: int, region: str = "pooled") -> dict:
    with h5py.File(path, "r") as handle:
        if "units" not in handle:
            return {"status": "excluded", "reason": "no units table"}
        n_units_raw = int(handle["units/id"].shape[0])
        if region == "pooled" and n_units_raw < MIN_UNITS:
            return {"status": "excluded", "reason": f"only {n_units_raw} raw units"}
        trials = _trial_group(handle, release)
        loads = trials["loads"][:].astype(int)
        onsets = trials["timestamps_Maintenance"][:]
        accuracy = trials["response_accuracy"][:].astype(bool)
        spike_lists = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle)["region"]

    spike_lists = filter_units_by_region(spike_lists, unit_regions, region)
    rate_mask = low_rate_unit_mask(spike_lists, onsets, WINDOW_S)
    spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
    min_units = MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION
    if len(spike_lists) < min_units:
        return {
            "status": "non_identified" if region != "pooled" else "excluded",
            "reason": f"only {len(spike_lists)} units after firing-rate QC (region={region})",
        }
    if float(np.mean(accuracy)) < MIN_SESSION_ACCURACY:
        return {"status": "excluded", "reason": "session accuracy below prospective QC floor"}

    psth = build_psth(spike_lists, onsets, bin_ms=BIN_MS, smooth_ms=0, window_s=WINDOW_S)
    dt = BIN_MS / 1000.0
    fits = _fit_by_load(psth.astype(float), loads, dt, seed, n_components=N_COMPONENTS)
    return {
        "status": "complete", "region": region,
        "n_units": len(spike_lists), "n_trials_total": int(len(loads)),
        "accuracy": float(np.mean(accuracy)), "bin_ms": BIN_MS, "smoothed": False,
        "by_load": fits,
    }


CONTENT_AXIS_METRIC_NAMES = (
    "state_space_lambda_identified_mean", "moment_lambda_identified_mean",
    "lambda_content_minus_permuted_identified_mean",
    "lambda_content_minus_complement_identified_mean",
    "m2_minus_m0_nats_per_observation", "m4_minus_m2_nats_per_transition",
    "m1_minus_m0_nats_per_observation", "m3_minus_m2_nats_per_transition",
    "counter_rotation_accuracy_recovery",
    "m4_tied_minus_m2_nats_per_transition",
    "m4_minus_heteroscedastic_drift_nats_per_transition",
    "trial_prediction_r2_advantage",
)


def analyze_session_content_axis(path: Path, release: str, seed: int, region: str = "pooled") -> dict:
    """The same M2/M4/content/DA battery
    run_human_drift_spine_000469.py fits on repeated-item identity, run here
    on the ONE trial-level label this dataset pair has -- memory load (1 vs
    3) -- fit JOINTLY across both loads rather than separately per load the
    way ``fit_load_confinement``/``_fit_by_load`` do. Gives 001187/000673 the
    same content axis, M1/M3 rotation comparison, counter-rotation accuracy
    recovery and trial-prediction-R2 estimands 000469 and 000574 already
    have, so the deciding hippocampus-vs-amygdala contrast can be re-run on
    the identical statistic across all three primary human datasets.
    """
    with h5py.File(path, "r") as handle:
        if "units" not in handle:
            return {"status": "excluded", "reason": "no units table"}
        spike_lists = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle)["region"]
        trials = _trial_group(handle, release)
        loads = trials["loads"][:].astype(int)
        accuracy = trials["response_accuracy"][:].astype(bool)
        onsets = trials["timestamps_Maintenance"][:]
    n_units_in_region = int(np.sum(unit_regions == region)) if region != "pooled" else len(spike_lists)
    spike_lists = filter_units_by_region(spike_lists, unit_regions, region)
    rate_mask = low_rate_unit_mask(spike_lists, onsets, WINDOW_S)
    spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
    min_units = MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION
    if len(spike_lists) < min_units:
        return {
            "status": "non_identified" if region != "pooled" else "excluded",
            "reason": f"only {len(spike_lists)} units after firing-rate QC (region={region})",
            "region": region, "n_units_in_region_before_rate_qc": n_units_in_region,
        }
    if float(np.mean(accuracy)) < MIN_SESSION_ACCURACY:
        return {"status": "excluded", "reason": "session accuracy below prospective QC floor"}
    counts = {int(load): int(np.sum(loads == load)) for load in np.unique(loads)}
    usable_classes = [load for load, count in counts.items() if count >= N_SPLITS]
    if len(usable_classes) < 2:
        return {"status": "excluded", "reason": "fewer than two load levels support five folds", "counts": counts}
    keep = np.isin(loads, usable_classes)
    labels = loads[keep]
    onsets_kept = onsets[keep]

    psth = build_psth(spike_lists, onsets_kept, bin_ms=BIN_MS, smooth_ms=0, window_s=WINDOW_S)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        transform = FrozenPSTHTransform().fit(psth[train_index])
        train_standardized = transform.transform(psth[train_index]).transpose(0, 2, 1)
        test_standardized = transform.transform(psth[test_index]).transpose(0, 2, 1)
        n_components = min(N_COMPONENTS, len(spike_lists) - 1, len(train_index) - 1)
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(train_standardized.reshape(-1, train_standardized.shape[-1]))
        train_state = pca.transform(train_standardized.reshape(-1, train_standardized.shape[-1])).reshape(
            len(train_index), psth.shape[2], -1,
        )
        test_state = pca.transform(test_standardized.reshape(-1, test_standardized.shape[-1])).reshape(
            len(test_index), psth.shape[2], -1,
        )
        train_window = train_state.mean(axis=1)
        direction = discriminant_direction(train_window, labels[train_index])
        train_residuals, test_residuals = projected_residuals(
            train_state, test_state, labels[train_index], labels[test_index], direction,
        )
        state_estimate = fit_gaussian_state_space(train_residuals, BIN_MS / 1000.0)
        moment_estimate = fit_ou_moments(
            train_residuals, BIN_MS / 1000.0, n_boot=120,
            rng=np.random.default_rng(seed + fold_index + 1000),
        )
        m2_log_likelihood = gaussian_state_space_log_likelihood(
            test_residuals, state_estimate, BIN_MS / 1000.0,
        )
        m2_transition_log_likelihood = gaussian_state_space_conditional_log_likelihood(
            test_residuals, state_estimate, BIN_MS / 1000.0,
        )
        switching_comparison = compare_switching_models(
            train_residuals, test_residuals, BIN_MS / 1000.0, n_restarts=4,
            rng=np.random.default_rng(seed + fold_index + 6000),
        )
        m4_transition_score = switching_comparison["free_log_likelihood_per_transition"]
        m0_log_likelihood = iid_log_likelihood(test_residuals, train_residuals)
        rotation_comparison = fit_rotation_drift_comparison(
            train_state, test_state, labels[train_index], labels[test_index], direction,
            train_residuals, test_residuals,
            m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1),
            BIN_MS / 1000.0,
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
            permuted_train, BIN_MS / 1000.0, n_boot=80,
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
            complement_train, BIN_MS / 1000.0, n_boot=80,
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
            "m4_log_likelihood_per_transition": m4_transition_score,
            "m4_minus_m2_nats_per_transition": (
                m4_transition_score
                - m2_transition_log_likelihood / max(test_residuals[:, 1:].size, 1)
            ),
            "switching_two_state": switching_comparison["free"],
            "rotation_comparison": rotation_comparison,
            "m1_minus_m0_nats_per_observation": rotation_comparison["m1_minus_m0_nats_per_observation"],
            "m3_minus_m2_nats_per_transition": rotation_comparison["m3_minus_m2_nats_per_transition"],
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
    content_minus_permuted = np.array([
        row["moment"]["lambda_rate"] - row["permuted_axis_moment"]["lambda_rate"]
        for row in fold_rows
        if row["moment"]["status"] == "identifiable" and row["permuted_axis_moment"]["status"] == "identifiable"
    ])
    content_minus_complement = np.array([
        row["moment"]["lambda_rate"] - row["matched_complement_moment"]["lambda_rate"]
        for row in fold_rows
        if row["moment"]["status"] == "identifiable"
        and row["matched_complement_moment"]["status"] == "identifiable"
    ])
    return {
        "status": "complete", "region": region,
        "n_trials": int(len(labels)), "n_units": int(len(spike_lists)),
        "n_load_levels": int(len(usable_classes)), "load_counts": counts,
        "bin_ms": BIN_MS, "smoothed": False, "condition": "load",
        "folds": fold_rows,
        "summary": {
            "state_space_lambda_identified_mean": (
                float(np.mean(state_lambdas)) if len(state_lambdas) else None
            ),
            "moment_lambda_identified_mean": float(np.mean(moment_lambdas)) if len(moment_lambdas) else None,
            "state_space_identifiable_folds": int(len(state_lambdas)),
            "moment_identifiable_folds": int(len(moment_lambdas)),
            "lambda_content_minus_permuted_identified_mean": (
                float(np.mean(content_minus_permuted)) if len(content_minus_permuted) else None
            ),
            "lambda_content_minus_complement_identified_mean": (
                float(np.mean(content_minus_complement)) if len(content_minus_complement) else None
            ),
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
                row["rotation_comparison"]["counter_rotation"]["accuracy_recovery"] for row in fold_rows
            ])),
            "m4_tied_minus_m2_nats_per_transition": float(np.mean([
                row["m4_tied_log_likelihood_per_transition"]
                - row["m2_log_likelihood_per_transition_conditional"] for row in fold_rows
            ])),
            "m4_minus_heteroscedastic_drift_nats_per_transition": float(np.mean([
                row["m4_minus_heteroscedastic_drift_nats_per_transition"] for row in fold_rows
            ])),
            "trial_prediction_r2_advantage": float(np.mean([
                row["trial_prediction"]["held_out_r2_advantage"] for row in fold_rows
            ])),
        },
    }


def content_axis_patient_level_means(
    sessions: dict, metric_names: tuple[str, ...] = CONTENT_AXIS_METRIC_NAMES,
) -> dict[str, dict[str, float]]:
    """Average each metric within patient before any cross-patient inference
    (advisor N4/10.4). Unlike run_human_drift_spine_000574.py's version,
    pairs on the ``patient`` field each session row already carries rather
    than splitting the session key string."""
    by_patient: dict[str, list[dict]] = {}
    for row in sessions.values():
        if row["content_axis_fit"]["status"] != "complete":
            continue
        by_patient.setdefault(row["patient"], []).append(row["content_axis_fit"])
    patient_metrics: dict[str, dict[str, float]] = {}
    for patient, rows in by_patient.items():
        patient_metrics[patient] = {}
        for name in metric_names:
            values = [row["summary"][name] for row in rows if row["summary"][name] is not None]
            patient_metrics[patient][name] = float(np.mean(values)) if values else None
    return patient_metrics


def fit_content_axis_region(
    root: Path, session_meta: list[dict], region: str, rng_seed: int = 20260731,
) -> tuple[dict, dict, dict]:
    """One region's worth of the 000469 M2/M4/content/DA
    battery, across every canonical 001187/000673 session. Same sessions and
    seeds regardless of region -- only the unit population changes. Returns
    ``(sessions, patient_metrics, group)``."""
    sessions = {}
    for meta in session_meta:
        key = f"{meta['patient']}_{meta['session']}"
        seed = stable_seed(key)
        print(f"fitting content-axis {key} region={region} (primary={meta['primary_release']})", flush=True)
        fit = analyze_session_content_axis(
            root / meta["primary_path"], meta["primary_release"], seed, region=region,
        )
        sessions[key] = {
            "patient": meta["patient"], "session": meta["session"],
            "primary_release": meta["primary_release"], "content_axis_fit": fit,
        }
    complete = {k: v for k, v in sessions.items() if v["content_axis_fit"]["status"] == "complete"}
    patient_metrics = content_axis_patient_level_means(sessions)
    n_identified = len(complete)
    n_non_identified = sum(
        1 for v in sessions.values() if v["content_axis_fit"]["status"] in ("non_identified", "excluded")
    )
    if not complete:
        return sessions, patient_metrics, {
            "status": "non_identified",
            "reason": f"no session-region cleared the minimum unit count for region={region}",
            "n_session_regions_identified": n_identified, "n_session_regions_non_identified": n_non_identified,
        }
    group = {}
    rng = np.random.default_rng(rng_seed)
    for name in CONTENT_AXIS_METRIC_NAMES:
        values = np.array([
            row[name] for row in patient_metrics.values() if row[name] is not None
        ])
        if len(values) < 2:
            group[name] = {
                "status": "not_estimable", "n_patients": int(len(values)),
                "reason": f"fewer than 2 patients have an identified estimate ({len(values)})",
            }
            continue
        group[name] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "ci": bootstrap_mean_content_axis(values, rng), "n_patients": int(len(values)),
            "values": values.tolist(),
        }
    threshold = json.loads(RULE_PATH.read_text())["minimum_effect"]["negative_log_likelihood_nats_per_observation"]
    m2 = group["m2_minus_m0_nats_per_observation"]
    m2["passes_practical_threshold"] = bool(
        m2.get("n_patients", 0) >= 2 and m2["mean"] >= threshold and m2["ci"][0] > 0
    )
    m4 = group["m4_minus_m2_nats_per_transition"]
    m4["passes_practical_threshold"] = bool(
        m4.get("n_patients", 0) >= 2 and m4["mean"] >= threshold and m4["ci"][0] > 0
        and np.mean(np.asarray(m4["values"]) > 0) >= 0.60
    )
    group["switching_decomposition"] = summarize_switching_decompositions([
        fold for session in complete.values() for fold in session["content_axis_fit"]["folds"]
    ])
    group["n_session_regions_identified"] = n_identified
    group["n_session_regions_non_identified"] = n_non_identified
    group["n_patients"] = len(patient_metrics)
    return sessions, patient_metrics, group


def bootstrap_mean_content_axis(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> list[float]:
    draws = np.mean(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def content_axis_paired_region_contrast(
    regions_group: dict[str, dict], region_a: str, region_b: str, metric: str,
) -> dict:
    """Within-patient region_a-minus-region_b difference on the content-axis
    battery, restricted to patients contributing both regions -- the SAME
    estimand as run_human_drift_spine_000469.py's deciding contrast
    (own-trial R2 advantage), unlike run_region_stratified's existing
    load3-minus-load1 lambda-delta contrast below."""
    metrics_a = regions_group[region_a]["patient_level_metrics"]
    metrics_b = regions_group[region_b]["patient_level_metrics"]
    shared_patients = sorted(set(metrics_a) & set(metrics_b))
    diffs = [
        metrics_a[p][metric] - metrics_b[p][metric]
        for p in shared_patients
        if metrics_a[p].get(metric) is not None and metrics_b[p].get(metric) is not None
    ]
    if len(diffs) < 2:
        return {
            "status": "non_identified",
            "reason": f"fewer than 2 patients contribute both {region_a} and {region_b}",
            "n_patients_both_regions": len(diffs),
        }
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(20260804)
    ci = bootstrap_mean_content_axis(diffs, rng)
    return {
        "status": "estimable",
        "region_a": region_a, "region_b": region_b, "metric": metric,
        "n_patients_both_regions": int(len(diffs)),
        "mean_difference": float(np.mean(diffs)),
        "median_difference": float(np.median(diffs)),
        "patient_bootstrap_ci95": ci,
        "direction_a_greater_than_b": bool(ci[0] > 0),
    }


def run_content_axis_region_stratified() -> Path:
    """Bring 001187/000673 to estimand parity with 000469
    -- the M2/M4/content/DA battery, region-stratified, fit jointly on load
    as the content label, plus the SAME deciding hippocampus-vs-amygdala
    contrast (own-trial-R2 advantage) 000469 and 000574 both use. Writes a
    new top-level ``content_axis_battery`` block into
    results/region_stratified_drift_001187_000673.json, alongside (not
    replacing) the existing load3-minus-load1 lambda-delta block from
    run_region_stratified() above -- both are estimable and answer different
    questions."""
    root = data_root()
    session_meta = canonical_sessions()
    regions_out: dict[str, dict] = {}
    for region in REGIONS_WITH_POOLED:
        sessions, patient_metrics, group = fit_content_axis_region(root, session_meta, region)
        regions_out[region] = {
            "sessions": sessions, "patient_level_metrics": patient_metrics, "group": group,
            "n_patients_identified": len(patient_metrics),
            "min_units_threshold": MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION,
        }

    deciding_contrast = {
        "hippocampus_minus_amygdala": content_axis_paired_region_contrast(
            regions_out, "hippocampus", "amygdala", "trial_prediction_r2_advantage",
        ),
        "predeclared_direction": (
            "hippocampus > amygdala on held-out trial-prediction R2 advantage, the SAME estimand "
            "run_human_drift_spine_000469.py's deciding contrast uses, unlike this "
            "file's other deciding_contrast_hippocampus_minus_amygdala block (load3-minus-load1 "
            "lambda delta) computed by run_region_stratified() above."
        ),
    }
    region_lambda_medians = {}
    for region in ANATOMICAL_REGIONS:
        group = regions_out[region]["group"]
        state_space = group.get("state_space_lambda_identified_mean") if isinstance(group, dict) else None
        # A single-patient point estimate must never rank in this ordering --
        # its bootstrap CI is numerically well-defined at n=1 but not decisive.
        if isinstance(state_space, dict) and state_space.get("median") is not None and state_space.get("n_patients", 0) >= 2:
            region_lambda_medians[region] = state_space["median"]
    if len(region_lambda_medians) >= 2:
        ordered = sorted(region_lambda_medians, key=region_lambda_medians.get, reverse=True)
        lambda_ordering = {
            "status": "estimable",
            "region_lambda_state_space_median": region_lambda_medians,
            "region_order_fastest_to_slowest": ordered,
            "note": "Rank-based ordering, joint-load-LDA content-axis fit. Not a fitted hierarchy model.",
        }
    else:
        lambda_ordering = {
            "status": "non_identified",
            "reason": "fewer than 2 regions had an identifiable state-space lambda median",
        }

    destination = RESULTS / "region_stratified_drift_001187_000673.json"
    artifact = json.loads(destination.read_text()) if destination.exists() else {}
    artifact["content_axis_battery"] = {
        "note": (
            "Estimand-parity battery: joint 2-class LDA on load (1 vs 3) as the "
            "content axis, same M0/M1/M2/M3/M4/trial-prediction estimands as "
            "run_human_drift_spine_000469.py, region-stratified across the same canonical "
            "001187/000673 sessions run_region_stratified() above uses."
        ),
        "min_units_per_region_threshold": MIN_UNITS_PER_REGION,
        "regions": regions_out,
        "deciding_contrast": deciding_contrast,
        "lambda_regional_ordering": lambda_ordering,
    }
    artifact["estimands_present"] = [
        "load3_minus_load1_lambda_delta (run_region_stratified, this file's top-level `regions`)",
        "pr_slope_correlation (this file's top-level `regions`)",
        "state_space_lambda_identified_mean (M2), moment_lambda_identified_mean, "
        "lambda_content_minus_permuted/complement, m2_minus_m0, m4_minus_m2 (switching), "
        "m1_minus_m0 (rotation), m3_minus_m2 (rotation+drift), counter_rotation_accuracy_recovery, "
        "trial_prediction_r2_advantage (content_axis_battery block)",
    ]
    destination.write_text(canonical_json(artifact))
    print(json.dumps({
        "output": str(destination),
        "regions": {r: regions_out[r]["n_patients_identified"] for r in REGIONS_WITH_POOLED},
        "deciding_contrast": deciding_contrast,
        "lambda_regional_ordering": lambda_ordering,
    }, indent=2))
    return destination


def _hippocampal_channel_mask(locations: np.ndarray) -> np.ndarray:
    return np.array([b"hippocamp" in loc.lower() for loc in locations])


def analyze_lfp_session(path: Path, seed: int) -> dict:
    with h5py.File(path, "r") as handle:
        if "acquisition" not in handle or "LFPs" not in handle["acquisition"]:
            return {"status": "excluded", "reason": "no acquisition/LFPs field in this NWB file"}
        lfp = handle["acquisition/LFPs"]
        electrode_rows = lfp["electrodes"][:]
        locations = handle["general/extracellular_ephys/electrodes/location"][:][electrode_rows]
        hippocampal_mask = _hippocampal_channel_mask(locations)
        if not hippocampal_mask.any():
            return {"status": "excluded", "reason": "no hippocampus-labelled LFP channel in this session"}
        data = lfp["data"][:][:, hippocampal_mask]
        starting_time = float(lfp["starting_time"][()])
        rate = float(lfp["starting_time"].attrs["rate"])
        trials = handle["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        onsets = trials["timestamps_Maintenance"][:]

    n_hippocampal = int(hippocampal_mask.sum())
    clean = common_average_reference(data)
    clean = line_noise_notch(clean, srate=rate, fundamental=LFP_LINE_FREQ_HZ)
    power = high_gamma_power(clean, srate=rate, smooth_ms=0.0)

    n_samples_window = int(round(WINDOW_S * rate))
    epochs = np.full((len(onsets), power.shape[1], n_samples_window), np.nan)
    for trial, onset in enumerate(onsets):
        start = int(round((onset - starting_time) * rate))
        if start < 0 or start + n_samples_window > power.shape[0]:
            continue
        epochs[trial] = power[start:start + n_samples_window].T
    valid = ~np.isnan(epochs).any(axis=(1, 2))
    if valid.sum() < 2 * MIN_TRIALS_PER_LOAD:
        return {
            "status": "excluded", "n_hippocampal_channels": n_hippocampal,
            "reason": f"only {int(valid.sum())} trials fit inside the recorded LFP span",
        }
    epochs_ct, _ = bin_time_axis(epochs[valid], np.arange(n_samples_window) / rate, BIN_MS, rate)
    dt = BIN_MS / 1000.0
    fits = _fit_by_load(epochs_ct, loads[valid], dt, seed, n_components=min(N_COMPONENTS, n_hippocampal))
    return {
        "status": "complete", "n_hippocampal_channels": n_hippocampal,
        "n_trials_total": int(valid.sum()), "srate_hz": rate, "bin_ms": BIN_MS, "smoothed": False,
        "band": "high_gamma_70_150hz", "by_load": fits,
    }


def _identified_delta(by_load: dict, estimator: str, field: str) -> float | None:
    load1, load3 = by_load.get("1", {}), by_load.get("3", {})
    if load1.get("status") != "complete" or load3.get("status") != "complete":
        return None
    key = f"{estimator}_{field}_identified_mean"
    v1, v3 = load1["summary"].get(key), load3["summary"].get(key)
    if v1 is None or v3 is None:
        return None
    return float(v3 - v1)


def pr_load_slope(dataset_summary: dict, session_key: str) -> float | None:
    """Reuses the already-computed per-session participation ratio by load
    (results/dandi001187_summary.json / dandi000673_summary.json's
    ``pr_per_load``, from spike_pipeline.pr_by_load) -- not rederived here."""
    row = dataset_summary.get(session_key)
    if row is None:
        return None
    pr = row.get("pr_per_load", {})
    if "1" not in pr or "3" not in pr:
        return None
    return float(pr["3"]["pr_cv"] - pr["1"]["pr_cv"])


def _patient_level_means(patient_ids: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce session-level values to one mean per patient, dropping
    non-finite entries -- sessions are nested within patient, so the patient, not the session, is the row entering any
    across-unit bootstrap or correlation test below."""
    patients, means = [], []
    for patient in np.unique(patient_ids):
        v = values[(patient_ids == patient) & np.isfinite(values)]
        if len(v):
            patients.append(patient)
            means.append(float(np.mean(v)))
    return np.array(patients), np.array(means)


def group_delta_summary(patient_ids: np.ndarray, values: np.ndarray, rng: np.random.Generator) -> dict:
    _, patient_values = _patient_level_means(patient_ids, values)
    n_sessions = int(np.sum(np.isfinite(values)))
    if len(patient_values) < 3:
        return {
            "status": "not_estimable", "n_patients": int(len(patient_values)), "n_sessions": n_sessions,
            "reason": f"only {len(patient_values)} patients have an identified estimate in both loads",
        }
    stat, lo, hi = bootstrap_ci(patient_values, np.mean, n_boot=5000, rng=rng)
    return {
        "status": "estimable", "mean": stat, "ci_95_patient_cluster_bootstrap": [lo, hi],
        "n_patients": int(len(patient_values)), "n_sessions": n_sessions,
    }


def group_correlation_summary(
    patient_ids: np.ndarray, delta: np.ndarray, pr_slope_values: np.ndarray, rng: np.random.Generator,
) -> dict:
    valid = np.isfinite(delta) & np.isfinite(pr_slope_values)
    p_delta, delta_p = _patient_level_means(patient_ids[valid], delta[valid])
    p_pr, pr_p = _patient_level_means(patient_ids[valid], pr_slope_values[valid])
    if len(delta_p) < 4 or not np.array_equal(p_delta, p_pr):
        return {
            "status": "not_estimable", "n_patients": int(len(delta_p)),
            "reason": f"only {len(delta_p)} patients have both an identified delta and a measured PR slope",
        }
    pearson = pearson_permutation_test(delta_p, pr_p, n_perm=5000, rng=rng)
    spearman = spearman_permutation_test(delta_p, pr_p, n_perm=5000, rng=rng)
    _, lo, hi = bootstrap_ci(
        np.column_stack([delta_p, pr_p]),
        lambda d: float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]),
        n_boot=5000, rng=rng,
    )
    return {
        "status": "estimable", "n_patients": int(len(delta_p)),
        "pearson_r": pearson["r"], "pearson_p_value": pearson["p_value"],
        "pearson_r_ci_95_patient_cluster_bootstrap": [lo, hi],
        "spearman_rho": spearman["rho"], "spearman_p_value": spearman["p_value"],
    }


def fit_all_sessions_one_region(
    root: Path, session_meta: list[dict], region: str, pr_summaries: dict, rng_seed: int = 20260731,
) -> tuple[dict, dict]:
    """One region's worth of the region-stratified drift-fitting apparatus, across every
    canonical 001187/000673 session. Same sessions and seeds regardless of
    region -- only the unit population changes."""
    sessions = {}
    for meta in session_meta:
        key = f"{meta['patient']}_{meta['session']}"
        seed = stable_seed(key)
        print(f"fitting {key} region={region} (primary={meta['primary_release']})", flush=True)
        unit_fit = analyze_unit_session(root / meta["primary_path"], meta["primary_release"], seed, region=region)
        pr_slope = pr_load_slope(pr_summaries[meta["primary_release"]], Path(meta["primary_path"]).stem)
        sessions[key] = {
            "patient": meta["patient"], "session": meta["session"],
            "primary_release": meta["primary_release"], "primary_path": meta["primary_path"],
            "unit_based_primary_fit": unit_fit,
            "measured_pr_load3_minus_load1": pr_slope,
        }

    patient_ids = np.array([row["patient"] for row in sessions.values()])
    pr_slopes = np.array([
        row["measured_pr_load3_minus_load1"]
        if row["measured_pr_load3_minus_load1"] is not None else np.nan
        for row in sessions.values()
    ])
    rng = np.random.default_rng(rng_seed)
    unit_deltas = {
        (estimator, field): np.array([
            _identified_delta(row["unit_based_primary_fit"].get("by_load", {}), estimator, field)
            if row["unit_based_primary_fit"].get("status") == "complete" else None
            for row in sessions.values()
        ], dtype=float)
        for estimator in ("state_space", "moment") for field in ("lambda", "diffusion")
    }
    n_identified = sum(row["unit_based_primary_fit"].get("status") == "complete" for row in sessions.values())
    n_non_identified = sum(row["unit_based_primary_fit"].get("status") == "non_identified" for row in sessions.values())
    group = {
        "n_canonical_sessions": len(sessions),
        "n_session_regions_identified": int(n_identified),
        "n_session_regions_non_identified": int(n_non_identified),
        "min_units_threshold": MIN_UNITS if region == "pooled" else MIN_UNITS_PER_REGION,
        "load3_minus_load1": {
            f"{estimator}_{field}": group_delta_summary(patient_ids, values, rng)
            for (estimator, field), values in unit_deltas.items()
        },
        "pr_slope_correlation": {
            f"{estimator}_{field}_delta_vs_measured_pr_slope": group_correlation_summary(
                patient_ids, values, pr_slopes, rng,
            )
            for (estimator, field), values in unit_deltas.items()
        },
    }
    unit_folds = [
        fold
        for session in sessions.values()
        if session["unit_based_primary_fit"].get("status") == "complete"
        for load_fit in session["unit_based_primary_fit"]["by_load"].values()
        if load_fit.get("status") == "complete"
        for fold in load_fit["folds"]
    ]
    group["switching_decomposition"] = summarize_switching_decompositions(unit_folds)
    return sessions, group


def mtl_restricted_001187_sessions(provenance_dir: Path | None = None) -> list[dict]:
    """Every native DANDI 001187 recording, independent of the 000673 cross-
    release canonical view. 001187 is MTL-only
    (amygdala + hippocampus, ~950 units) and is therefore not chimeric across
    lobes the way the pooled 000469 state or the 001187/000673 canonical view
    (which pulls in 000673's dACC/pre-SMA/vmPFC for its own primary sessions)
    can be. This is the full native 001187 cohort, used only as its own
    independent MTL replication -- never combined with the canonical view's
    N.
    """
    provenance_dir = provenance_dir or PROVENANCE_DIR
    registry = json.loads((provenance_dir / "canonical_recording_registry.json").read_text())
    return sorted(
        (
            {"patient": row["patient"], "session": row["session"],
             "primary_release": "001187", "primary_path": row["path"]}
            for row in registry if row["release"] == "001187"
        ),
        key=lambda s: (s["patient"], s["session"]),
    )


def run_mtl_restricted_001187_replication() -> dict:
    """The independent MTL-restricted check on the deciding hippocampus-vs-amygdala contrast,
    reported BESIDE the 000469 hippocampus result, never averaged with it.
    Extends results/region_stratified_drift_001187_000673.json with a
    top-level ``mtl_restricted_001187_only`` block."""
    root = data_root()
    session_meta = mtl_restricted_001187_sessions()
    pr_summaries = {"001187": {}}
    path = RESULTS / "dandi001187_summary.json"
    if path.is_file():
        pr_summaries["001187"] = json.loads(path.read_text())

    regions_out = {}
    for region in ("pooled", "hippocampus", "amygdala"):
        sessions, group = fit_all_sessions_one_region(root, session_meta, region, pr_summaries)
        regions_out[region] = {"sessions": sessions, "group": group}

    destination = RESULTS / "region_stratified_drift_001187_000673.json"
    artifact = json.loads(destination.read_text()) if destination.exists() else {}
    artifact["mtl_restricted_001187_only"] = {
        "n_native_recordings": len(session_meta),
        "note": (
            "Independent check on the deciding hippocampus-vs-amygdala contrast, "
            "using every native DANDI 001187 recording directly (not the cross-release canonical view "
            "above, which substitutes some sessions with their linked 000673 primary view). Report "
            "beside results/region_stratified_drift_000469.json's hippocampus result; if they "
            "disagree, that is the finding, not resolved by averaging."
        ),
        "regions": regions_out,
    }
    destination.write_text(canonical_json(artifact))
    print(json.dumps({
        "output": str(destination), "n_native_recordings": len(session_meta),
        "regions": {r: regions_out[r]["group"]["n_session_regions_identified"] for r in regions_out},
    }, indent=2))
    return artifact


def run_region_stratified() -> Path:
    """Region-stratified drift fitting for the 001187/000673
    canonical view. No content axis exists in this dataset pair (novel
    pictures per trial), so the deciding-contrast statistic here is the
    load3-minus-load1 state-space lambda delta, not an own-trial-R2 --
    the own-trial-R2 statistic applies to 000469 specifically;
    this pair's applicable half is the matched-flexibility-free M2-vs-M0
    delta already computed by fit_load_confinement.
    """
    root = data_root()
    session_meta = canonical_sessions()
    pr_summaries = {}
    for release, name in (("001187", "dandi001187_summary.json"), ("000673", "dandi000673_summary.json")):
        path = RESULTS / name
        pr_summaries[release] = json.loads(path.read_text()) if path.is_file() else {}

    regions_out = {}
    for region in REGIONS_WITH_POOLED:
        sessions, group = fit_all_sessions_one_region(root, session_meta, region, pr_summaries)
        regions_out[region] = {"sessions": sessions, "group": group}

    def lambda_delta_median(region: str) -> float | None:
        entry = regions_out[region]["group"]["load3_minus_load1"].get("state_space_lambda")
        return entry.get("mean") if entry and entry.get("status") == "estimable" else None

    region_lambda = {r: lambda_delta_median(r) for r in ANATOMICAL_REGIONS if lambda_delta_median(r) is not None}
    hippocampus_sessions = {
        k: v for k, v in regions_out["hippocampus"]["sessions"].items()
        if v["unit_based_primary_fit"].get("status") == "complete"
    }
    amygdala_sessions = {
        k: v for k, v in regions_out["amygdala"]["sessions"].items()
        if v["unit_based_primary_fit"].get("status") == "complete"
    }
    shared = sorted(set(hippocampus_sessions) & set(amygdala_sessions))
    hippocampus_minus_amygdala_diffs = [
        _identified_delta(hippocampus_sessions[k]["unit_based_primary_fit"]["by_load"], "state_space", "lambda")
        - _identified_delta(amygdala_sessions[k]["unit_based_primary_fit"]["by_load"], "state_space", "lambda")
        for k in shared
        if _identified_delta(hippocampus_sessions[k]["unit_based_primary_fit"]["by_load"], "state_space", "lambda") is not None
        and _identified_delta(amygdala_sessions[k]["unit_based_primary_fit"]["by_load"], "state_space", "lambda") is not None
    ]
    if len(hippocampus_minus_amygdala_diffs) >= 2:
        rng = np.random.default_rng(20260803)
        _, lo, hi = bootstrap_ci(np.asarray(hippocampus_minus_amygdala_diffs), np.mean, n_boot=5000, rng=rng)
        deciding_contrast = {
            "status": "estimable", "metric": "state_space_lambda load3_minus_load1",
            "n_sessions_both_regions": len(hippocampus_minus_amygdala_diffs),
            "mean_difference": float(np.mean(hippocampus_minus_amygdala_diffs)),
            "ci_95_bootstrap": [float(lo), float(hi)],
        }
    else:
        deciding_contrast = {
            "status": "non_identified",
            "reason": f"only {len(hippocampus_minus_amygdala_diffs)} sessions had both regions identified",
        }

    output = {
        "schema_version": "1.0.0", "analysis_id": "region_stratified_drift_dandi001187_000673",
        "dataset": "DANDI 001187 (canonical unit view) + DANDI 000673 (linked hippocampal-LFP sensitivity view)",
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "min_units_per_region_threshold": MIN_UNITS_PER_REGION,
        "regions": regions_out,
        "region_lambda_load3_minus_load1_mean": region_lambda,
        "deciding_contrast_hippocampus_minus_amygdala": deciding_contrast,
        "pooled_result_superseded_note": (
            "regions.pooled reproduces results/human_drift_spine_001187_000673.json's unit-based "
            "primary fit exactly (region='pooled' is a no-op filter); retained inline, superseded "
            "by the per-structure headline below."
        ),
    }
    destination = RESULTS / "region_stratified_drift_001187_000673.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(destination),
        "regions": {r: regions_out[r]["group"]["n_session_regions_identified"] for r in REGIONS_WITH_POOLED},
    }, indent=2))
    return destination


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-stratified", action="store_true")
    parser.add_argument("--mtl-restricted-001187", action="store_true")
    parser.add_argument("--content-axis-region-stratified", action="store_true",
                         help="Fit the 000469 M2/M4/content/DA battery, region-stratified")
    args = parser.parse_args()
    if args.mtl_restricted_001187:
        run_mtl_restricted_001187_replication()
        return
    if args.content_axis_region_stratified:
        run_content_axis_region_stratified()
        return
    if args.region_stratified:
        run_region_stratified()
        return
    if sha256_file(RULE_PATH) != RULE_HASH:
        raise SystemExit("frozen adjudication rule hash mismatch; refusing to fit real data")
    root = data_root()
    session_meta = canonical_sessions()

    pr_summaries = {}
    for release, name in (("001187", "dandi001187_summary.json"), ("000673", "dandi000673_summary.json")):
        path = RESULTS / name
        pr_summaries[release] = json.loads(path.read_text()) if path.is_file() else {}
        if not path.is_file():
            print(f"WARNING: {path} not found -- PR-vs-load correlation will be missing for {release} sessions")

    sessions = {}
    for meta in session_meta:
        key = f"{meta['patient']}_{meta['session']}"
        seed = stable_seed(key)
        print(f"fitting {key} (primary={meta['primary_release']}, lfp_linked={meta['lfp_path'] is not None})",
              flush=True)
        unit_fit = analyze_unit_session(root / meta["primary_path"], meta["primary_release"], seed)
        lfp_fit = analyze_lfp_session(root / meta["lfp_path"], seed + 500_000) if meta["lfp_path"] else None
        pr_slope = pr_load_slope(pr_summaries[meta["primary_release"]], Path(meta["primary_path"]).stem)
        sessions[key] = {
            "patient": meta["patient"], "session": meta["session"],
            "primary_release": meta["primary_release"], "primary_path": meta["primary_path"],
            "lfp_path": meta["lfp_path"],
            "unit_based_primary_fit": unit_fit,
            "lfp_linked_sensitivity_fit": lfp_fit,
            "lfp_linked_sensitivity_note": (
                "linked sensitivity analysis on the SAME patient-session as the unit-based "
                "primary fit -- not an independent replication"
                if lfp_fit is not None else None
            ),
            "measured_pr_load3_minus_load1": pr_slope,
        }

    patient_ids = np.array([row["patient"] for row in sessions.values()])
    pr_slopes = np.array([
        row["measured_pr_load3_minus_load1"]
        if row["measured_pr_load3_minus_load1"] is not None else np.nan
        for row in sessions.values()
    ])

    rng = np.random.default_rng(20260731)
    unit_deltas = {
        (estimator, field): np.array([
            _identified_delta(row["unit_based_primary_fit"].get("by_load", {}), estimator, field)
            if row["unit_based_primary_fit"].get("status") == "complete" else None
            for row in sessions.values()
        ], dtype=float)
        for estimator in ("state_space", "moment") for field in ("lambda", "diffusion")
    }
    lfp_deltas = {
        (estimator, field): np.array([
            _identified_delta(row["lfp_linked_sensitivity_fit"]["by_load"], estimator, field)
            if row["lfp_linked_sensitivity_fit"] is not None
            and row["lfp_linked_sensitivity_fit"].get("status") == "complete" else None
            for row in sessions.values()
        ], dtype=float)
        for estimator in ("state_space", "moment") for field in ("lambda", "diffusion")
    }

    group = {
        "n_canonical_sessions": len(sessions),
        "n_unit_fits_complete": int(sum(
            row["unit_based_primary_fit"].get("status") == "complete" for row in sessions.values()
        )),
        "n_lfp_linked_sessions_available": int(sum(row["lfp_path"] is not None for row in sessions.values())),
        "n_lfp_fits_complete": int(sum(
            row["lfp_linked_sensitivity_fit"] is not None
            and row["lfp_linked_sensitivity_fit"].get("status") == "complete"
            for row in sessions.values()
        )),
        "unit_based_load3_minus_load1": {
            f"{estimator}_{field}": group_delta_summary(patient_ids, values, rng)
            for (estimator, field), values in unit_deltas.items()
        },
        "lfp_linked_load3_minus_load1": {
            f"{estimator}_{field}": group_delta_summary(patient_ids, values, rng)
            for (estimator, field), values in lfp_deltas.items()
        },
        "pr_slope_correlation": {
            f"{estimator}_{field}_delta_vs_measured_pr_slope": group_correlation_summary(
                patient_ids, values, pr_slopes, rng,
            )
            for (estimator, field), values in unit_deltas.items()
        },
    }
    unit_folds = [
        fold
        for session in sessions.values()
        if session["unit_based_primary_fit"].get("status") == "complete"
        for load_fit in session["unit_based_primary_fit"]["by_load"].values()
        if load_fit.get("status") == "complete"
        for fold in load_fit["folds"]
    ]
    lfp_folds = [
        fold
        for session in sessions.values()
        if session["lfp_linked_sensitivity_fit"] is not None
        and session["lfp_linked_sensitivity_fit"].get("status") == "complete"
        for load_fit in session["lfp_linked_sensitivity_fit"]["by_load"].values()
        if load_fit.get("status") == "complete"
        for fold in load_fit["folds"]
    ]
    group["switching_decomposition"] = {
        "unit_based_primary": summarize_switching_decompositions(unit_folds),
        "lfp_linked_sensitivity": summarize_switching_decompositions(lfp_folds),
    }

    output = {
        "schema_version": "1.0.0", "analysis_id": "human_drift_spine_dandi001187_000673",
        "dataset": "DANDI 001187 (canonical unit view) + DANDI 000673 (linked hippocampal-LFP sensitivity view)",
        "canonical_role": "load-manipulation human intracranial dataset pair (load 1 vs load 3, novel pictures)",
        "canonical_view_rule": (
            "Consumed from provenance/dataset_overlap_report.json (built by "
            "scripts/audit_dataset_identity.py): prefer 001187 for a verified "
            "001187/000673 shared recording; retain 000673 as a linked "
            "sensitivity view. This script's every canonical session enters "
            "the unit-based primary fit exactly once (see "
            "tests/test_human_drift_spine_001187_000673.py); the matching "
            "000673 LFP view, where one exists, enters only the LFP-linked "
            "sensitivity arm on that same session."
        ),
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "preprocessing_provenance": PREPROCESSING_PROVENANCE,
        "coordinate": "FrozenPSTHTransform and PCA fit within each outer-training fold",
        "smoothing": "none",
        "condition_factor": (
            "memory load (1 vs 3), fit separately per load; no repeated item "
            "identity in this dataset pair, so unlike run_human_drift_spine_000469.py "
            "the projection axis is the leading within-fold PCA component, not "
            "an LDA content direction (see fit_load_confinement docstring)"
        ),
        "sessions": sessions,
        "group": group,
        "model_status": {
            "M0": "scored", "M1": "not applicable -- no content axis in this dataset pair",
            "M2": "scored", "M3": "not applicable",
            "M4": "scored as a probabilistic two-state Gaussian AR-HMM; not a full Poisson rSLDS",
        },
    }
    destination = RESULTS / "human_drift_spine_001187_000673.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "n_sessions": len(sessions), "group": group, "output": str(destination),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
