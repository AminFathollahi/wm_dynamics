#!/usr/bin/env python3
"""Within-patient spike-vs-iEEG confinement-rate agreement for DANDI 000574 (Boran).

Boran's NWB release is the one dataset in this project where single units and
macro iEEG/LFP are recorded from the SAME patients on the SAME trials. That
makes it the sharpest internal-validity check available: if the fitted
confinement rate lambda is a property of the underlying circuit rather than
of the recording modality, an estimate from spiking and an estimate from
high-gamma power should agree on identical patient-sessions and identical
trials. This script fits both arms with the shared confined-drift estimator
(`src/drift_dynamics.py`) on identical outer folds and compares them; it
never treats spiking and iEEG as independent replications -- they are two
views of the same underlying circuit activity in the same patient, so
pooling them as if they were separate observations would double-count that
patient's evidence -- every comparison is within-session, and every
synthesis is within-patient before any cross-patient statement.

Session/trial selection reuses the machinery already used elsewhere in this
project rather than re-deriving a second, independently-implemented NWB
parser and session-identity resolver:
  - session list: `provenance/canonical_recording_registry.json` filtered to
    release "000574", so this arm cannot double-count a patient-session that
    canonical dedup already resolved elsewhere.
  - spike arm: the same firing-rate QC, PSTH construction, discriminant
    direction, leave-one-out residualization, and state-space/OU-moment
    estimators as `scripts/run_human_drift_spine_000574.py`.
  - iEEG arm: `src/preprocessing.py`'s `load_boran_nwb`/`compute_boran_hgp`,
    with channel QC turned on (`reject_channels=True`) so this arm does not
    inherit the false-negative gap documented alongside this script (Boran's
    loader previously ran no channel QC at all); line-noise notch and
    shank-bipolar referencing follow the same convention already established
    in `scripts/run_boran_pipeline.py`. Boran is recorded in Zurich (50 Hz
    mains), not a US site -- `mains_hz=50.0` throughout, never 60.

Both arms are binned identically (100 ms, non-overlapping, unsmoothed) over
the identical maintenance-window trial set, split with the identical
`StratifiedKFold` seed per session, so "same trials" holds at the fold level,
not just at the session level.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_dynamics import (  # noqa: E402
    fit_gaussian_state_space,
    fit_ou_moments,
)
from preprocessing import (  # noqa: E402
    bipolar_reference_by_shank,
    common_average_reference,
    compute_boran_hgp,
    line_noise_notch,
    load_boran_nwb,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import (  # noqa: E402
    FrozenPSTHTransform,
    build_psth,
    load_spike_times,
    low_rate_unit_mask,
)
from run_human_drift_spine_000574 import (  # noqa: E402
    RULE_HASH,
    RULE_PATH,
    bootstrap_mean,
    discriminant_direction,
    patient_level_means,
    projected_residuals,
)

BIN_MS = 100
MAINT_ONSET_S = 3.0
MAINT_WIN = 3.0
N_COMPONENTS = 8
N_SPLITS = 5
MIN_UNITS = 8            # matches scripts/run_000574_units_pipeline.py
MIN_BIPOLAR_CHANNELS = 4
MIN_TRIALS = 20
BORAN_MAINS_HZ = 50.0     # Zurich (Sarnthein lab) -- NOT 60 Hz US; see src/preprocessing.py
BAD_CHANNEL_MAD_THRESHOLD = 3.0
DESCRIPTIVE_ONLY_PATIENT_FLOOR = 5  # below this, a handful of patients is a case series, not a population sample -- no population-level CI language

MOMENT_BOOT = 120


def registry_sessions(data_root: Path) -> list[dict]:
    """DANDI 000574 rows from the canonical recording registry, sorted for determinism."""
    registry = json.loads((ROOT / "provenance" / "canonical_recording_registry.json").read_text())
    rows = [row for row in registry if row.get("release") == "000574"]
    return sorted(rows, key=lambda row: row["session"])


def session_seed(session_key: str) -> int:
    subject, _, session = session_key.partition("_ses-")
    subject_num = int(subject.replace("sub-", ""))
    session_num = int(session)
    return 20260801 + subject_num * 100 + session_num


def lfp_maintenance_tensor(
    ieeg: dict, trial_mask: np.ndarray, n_bins: int, lo: float = 70.0, hi: float = 150.0,
) -> np.ndarray:
    """Notch -> bipolar-reference -> band-envelope power -> maintenance-window bin.

    Matches the convention already established in scripts/run_boran_pipeline.py
    (filter/envelope computed on the full pre-probe-to-post-probe epoch, THEN
    the maintenance sub-window is sliced out, to keep zero-phase filter edge
    effects away from the analysis window boundary). ``lo``/``hi`` default to
    this project's standard high-gamma band; passing a different band (e.g.
    8-45 Hz) reuses the identical filter-and-envelope path with only the
    passband changed, which is the intended way to compare bands at fixed
    sensor and referencing.
    """
    epochs = ieeg["epochs"][trial_mask]
    srate = ieeg["srate"]
    labels = ieeg["electrode_labels"]
    n_trials, n_channels, n_samples = epochs.shape
    notched = np.empty_like(epochs)
    for trial in range(n_trials):
        notched[trial] = line_noise_notch(
            epochs[trial].T, srate, fundamental=BORAN_MAINS_HZ, n_harmonics=3,
        ).T
    flat = notched.transpose(0, 2, 1).reshape(-1, n_channels)
    bipolar_flat, _ = bipolar_reference_by_shank(flat, labels)
    bipolar = bipolar_flat.reshape(n_trials, n_samples, -1).transpose(0, 2, 1).astype(np.float32)
    # smooth_ms=0: an estimator that reports a persistence or confinement
    # contrast must never see a smoothed signal -- a temporal kernel
    # manufactures autocorrelation indistinguishable from genuine confinement.
    # compute_boran_hgp's default (50 ms Gaussian, used by the CTG/geometry
    # pipelines) is correct for those purposes but wrong here; this arm must
    # match the spike arm's build_psth(..., smooth_ms=0, ...) convention
    # exactly.
    band_power = compute_boran_hgp(bipolar, srate=srate, lo=lo, hi=hi, smooth_ms=0.0)
    maint_mask = (ieeg["times"] >= -MAINT_WIN) & (ieeg["times"] < 0)
    band_maint = band_power[:, :, maint_mask]
    edges = np.linspace(0, band_maint.shape[2], n_bins + 1).astype(int)
    return np.stack(
        [band_maint[:, :, edges[k]:edges[k + 1]].mean(axis=2) for k in range(n_bins)], axis=2,
    )


MASTOID_LABELS = ("A1", "A2")


def scalp_reference_excluding_mastoids(data: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Common-average reference across scalp channels, excluding the two
    mastoids from both the average and the analysis -- pre-declared before
    any scalp session is fit, not improvised per session. The depth path's
    bipolar-by-shank scheme (shank_bipolar_index_pairs) has no meaning for a
    scalp 10-20 montage, which has no shank structure to difference along, so
    it is not reused here.

    Parameters
    ----------
    data   : (T, C)
    labels : length-C electrode labels (10-20 names plus the two mastoids)

    Returns
    -------
    referenced  : (T, C - n_mastoids_present)
    kept_labels : list[str], mastoid labels removed, order otherwise preserved
    """
    keep = [i for i, lab in enumerate(labels) if lab not in MASTOID_LABELS]
    referenced = common_average_reference(data[:, keep])
    kept_labels = [labels[i] for i in keep]
    return referenced, kept_labels


def scalp_low_band_maintenance_tensor(
    eeg: dict, trial_mask: np.ndarray, n_bins: int, lo: float = 8.0, hi: float = 45.0,
) -> tuple[np.ndarray, list[str]]:
    """Notch -> common-average-reference-excluding-mastoids -> low-band
    envelope power -> maintenance-window bin. Mirrors lfp_maintenance_tensor's
    structure exactly (same mains notch, same maintenance window, same
    binning convention, same unsmoothed envelope); the referencing step
    differs because a scalp montage has no shank structure to bipolar-
    reference along (scalp_reference_excluding_mastoids). The 8-45 Hz default
    is fixed by two physical constraints -- below the 50 Hz mains notch and
    its harmonics, and below the scalp series' own Nyquist frequency with
    margin -- not tuned per call.

    Returns
    -------
    tensor      : (trials, channels, n_bins) float32
    kept_labels : list[str], the channel labels surviving mastoid exclusion
    """
    epochs = eeg["epochs"][trial_mask]
    srate = eeg["srate"]
    labels = eeg["electrode_labels"]
    n_trials, n_channels, n_samples = epochs.shape
    notched = np.empty_like(epochs)
    for trial in range(n_trials):
        notched[trial] = line_noise_notch(
            epochs[trial].T, srate, fundamental=BORAN_MAINS_HZ, n_harmonics=3,
        ).T
    flat = notched.transpose(0, 2, 1).reshape(-1, n_channels)
    referenced_flat, kept_labels = scalp_reference_excluding_mastoids(flat, labels)
    referenced = referenced_flat.reshape(n_trials, n_samples, -1).transpose(0, 2, 1).astype(np.float32)
    band_power = compute_boran_hgp(referenced, srate=srate, lo=lo, hi=hi, smooth_ms=0.0)
    maint_mask = (eeg["times"] >= -MAINT_WIN) & (eeg["times"] < 0)
    band_maint = band_power[:, :, maint_mask]
    edges = np.linspace(0, band_maint.shape[2], n_bins + 1).astype(int)
    tensor = np.stack(
        [band_maint[:, :, edges[k]:edges[k + 1]].mean(axis=2) for k in range(n_bins)], axis=2,
    )
    return tensor, kept_labels


def fit_modality_fold(
    tensor: np.ndarray, train_index: np.ndarray, test_index: np.ndarray,
    labels: np.ndarray, seed: int,
) -> dict:
    """One fold's state-space + moment confinement estimate for one modality tensor."""
    transform = FrozenPSTHTransform().fit(tensor[train_index])
    train_standardized = transform.transform(tensor[train_index]).transpose(0, 2, 1)
    test_standardized = transform.transform(tensor[test_index]).transpose(0, 2, 1)
    n_features = tensor.shape[1]
    n_components = min(N_COMPONENTS, n_features - 1, len(train_index) - 1)
    pca = PCA(n_components=max(n_components, 1), svd_solver="full")
    pca.fit(train_standardized.reshape(-1, train_standardized.shape[-1]))
    train_state = pca.transform(train_standardized.reshape(-1, train_standardized.shape[-1])).reshape(
        len(train_index), tensor.shape[2], -1,
    )
    test_state = pca.transform(test_standardized.reshape(-1, test_standardized.shape[-1])).reshape(
        len(test_index), tensor.shape[2], -1,
    )
    train_window = train_state.mean(axis=1)
    direction = discriminant_direction(train_window, labels[train_index])
    train_residuals, test_residuals = projected_residuals(
        train_state, test_state, labels[train_index], labels[test_index], direction,
    )
    state_estimate = fit_gaussian_state_space(train_residuals, BIN_MS / 1000.0)
    moment_estimate = fit_ou_moments(
        train_residuals, BIN_MS / 1000.0, n_boot=MOMENT_BOOT,
        rng=np.random.default_rng(seed),
    )
    return {
        "n_features": int(n_features), "n_components": int(pca.n_components_),
        "state_space": state_estimate.to_dict(), "moment": moment_estimate.to_dict(),
    }


def _modality_pair(spike_fit: dict, lfp_fit: dict, estimator: str) -> dict | None:
    spike_est, lfp_est = spike_fit[estimator], lfp_fit[estimator]
    if spike_est["status"] != "identifiable" or lfp_est["status"] != "identifiable":
        return None
    spike_lambda, lfp_lambda = spike_est["lambda_rate"], lfp_est["lambda_rate"]
    log_ratio = (
        float(np.log(spike_lambda / lfp_lambda))
        if spike_lambda > 0 and lfp_lambda > 0 else None
    )
    ci_overlap = None
    if spike_est["lambda_ci"] is not None and lfp_est["lambda_ci"] is not None:
        ci_overlap = bool(
            max(spike_est["lambda_ci"][0], lfp_est["lambda_ci"][0])
            <= min(spike_est["lambda_ci"][1], lfp_est["lambda_ci"][1])
        )
    return {
        "spike_lambda": spike_lambda, "lfp_lambda": lfp_lambda,
        "diff_spike_minus_lfp": float(spike_lambda - lfp_lambda),
        "log_ratio_spike_over_lfp": log_ratio, "ci_overlap": ci_overlap,
    }


def _mean_or_none(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def analyze_session(path: Path, seed: int) -> dict:
    with h5py.File(path, "r") as handle:
        if "units" not in handle:
            return {"status": "excluded", "reason": "no units table in this NWB file"}
        spike_lists = load_spike_times(handle)
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
        set_size = trials["set_size"][:].astype(int)
        correct = trials["correct"][:].astype(bool)
        start_time = trials["start_time"][:]

    ieeg = load_boran_nwb(
        str(path), signal="ieeg", reject_channels=True,
        bad_channel_mad_threshold=BAD_CHANNEL_MAD_THRESHOLD, mains_hz=BORAN_MAINS_HZ,
    )
    if ieeg["epochs"].shape[0] != len(artifact):
        return {"status": "excluded", "reason": "spike and iEEG trial counts disagree in this NWB file"}

    keep = (~artifact) & ieeg["valid"]
    if int(keep.sum()) < MIN_TRIALS:
        return {"status": "excluded", "reason": f"only {int(keep.sum())} trials pass artifact+window validity"}

    set_size_kept = set_size[keep]
    counts = {int(size): int(np.sum(set_size_kept == size)) for size in np.unique(set_size_kept)}
    usable_classes = [size for size, count in counts.items() if count >= N_SPLITS]
    if len(usable_classes) < 2:
        return {"status": "excluded", "reason": "fewer than two set-size levels support five folds", "counts": counts}

    keep_final = keep & np.isin(set_size, usable_classes)
    labels = set_size[keep_final]
    accuracy = correct[keep_final]
    n_trials = int(len(labels))
    onsets = start_time[keep_final] + MAINT_ONSET_S

    rate_mask = low_rate_unit_mask(spike_lists, onsets, MAINT_WIN)
    spike_units = [spikes for spikes, keep_unit in zip(spike_lists, rate_mask) if keep_unit]
    n_units_total = len(spike_lists)
    if len(spike_units) < MIN_UNITS:
        return {
            "status": "excluded",
            "reason": f"only {len(spike_units)}/{n_units_total} units after firing-rate QC",
            "n_trials_common": n_trials,
        }
    psth = build_psth(spike_units, onsets, bin_ms=BIN_MS, smooth_ms=0, window_s=MAINT_WIN)

    lfp_tensor = lfp_maintenance_tensor(ieeg, keep_final, n_bins=psth.shape[2])
    n_channels_bipolar = lfp_tensor.shape[1]
    if n_channels_bipolar < MIN_BIPOLAR_CHANNELS:
        return {
            "status": "excluded",
            "reason": f"only {n_channels_bipolar} bipolar iEEG channels after QC",
            "n_trials_common": n_trials,
        }

    region_composition = dict(Counter(ieeg["electrode_locs"]))

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(n_trials), labels)
    ):
        spike_fit = fit_modality_fold(
            psth, train_index, test_index, labels, seed=seed + fold_index + 1000,
        )
        lfp_fit = fit_modality_fold(
            lfp_tensor, train_index, test_index, labels, seed=seed + fold_index + 5000,
        )
        fold_rows.append({
            "fold": fold_index, "n_train": int(len(train_index)), "n_test": int(len(test_index)),
            "spike": spike_fit, "lfp": lfp_fit,
            "moment_pair": _modality_pair(spike_fit, lfp_fit, "moment"),
            "state_space_pair": _modality_pair(spike_fit, lfp_fit, "state_space"),
        })

    moment_pairs = [row["moment_pair"] for row in fold_rows if row["moment_pair"] is not None]
    state_pairs = [row["state_space_pair"] for row in fold_rows if row["state_space_pair"] is not None]
    moment_ratios = [p["log_ratio_spike_over_lfp"] for p in moment_pairs if p["log_ratio_spike_over_lfp"] is not None]
    state_ratios = [p["log_ratio_spike_over_lfp"] for p in state_pairs if p["log_ratio_spike_over_lfp"] is not None]

    # Marginal (single-modality) identifiability, separate from the joint-both
    # requirement above -- distinguishing "the two lambdas disagree" from
    # "one modality is rarely identifiable at all, which is why they are
    # rarely jointly comparable" needs both numbers side by side.
    marginal = {
        "n_folds": len(fold_rows),
        "spike_moment_identifiable": sum(f["spike"]["moment"]["status"] == "identifiable" for f in fold_rows),
        "lfp_moment_identifiable": sum(f["lfp"]["moment"]["status"] == "identifiable" for f in fold_rows),
        "spike_state_space_identifiable": sum(f["spike"]["state_space"]["status"] == "identifiable" for f in fold_rows),
        "lfp_state_space_identifiable": sum(f["lfp"]["state_space"]["status"] == "identifiable" for f in fold_rows),
    }

    errors = int(np.sum(~accuracy))
    return {
        "status": "complete",
        "n_trials": n_trials, "n_units_total": n_units_total, "n_units_qc": len(spike_units),
        "n_channels_ieeg_good": int(ieeg["good_channels"].sum()),
        "n_channels_bipolar": int(n_channels_bipolar),
        "region_composition": region_composition,
        "n_set_size_levels": len(usable_classes), "set_size_counts": counts,
        "n_errors": errors, "accuracy": float(np.mean(accuracy)), "bin_ms": BIN_MS,
        "folds": fold_rows,
        "marginal_identifiability": marginal,
        "summary": {
            "moment_identifiable_both_folds": len(moment_pairs),
            "moment_mean_diff": _mean_or_none([p["diff_spike_minus_lfp"] for p in moment_pairs]),
            "moment_mean_log_ratio": _mean_or_none(moment_ratios),
            "moment_ci_overlap_fraction": (
                _mean_or_none([float(p["ci_overlap"]) for p in moment_pairs if p["ci_overlap"] is not None])
            ),
            "state_identifiable_both_folds": len(state_pairs),
            "state_mean_diff": _mean_or_none([p["diff_spike_minus_lfp"] for p in state_pairs]),
            "state_mean_log_ratio": _mean_or_none(state_ratios),
            "state_ci_overlap_fraction": (
                _mean_or_none([float(p["ci_overlap"]) for p in state_pairs if p["ci_overlap"] is not None])
            ),
        },
    }


def chase_diagnostics(complete: dict) -> list[dict]:
    """Per-session table asking whether spike-versus-LFP disagreement tracks unit
    count, channel count, region, or firing rate rather than being a uniform
    modality gap."""
    rows = []
    for key, row in complete.items():
        rows.append({
            "session": key,
            "n_units_qc": row["n_units_qc"],
            "n_channels_bipolar": row["n_channels_bipolar"],
            "n_trials": row["n_trials"],
            "accuracy": row["accuracy"],
            "region_composition": row["region_composition"],
            "moment_mean_diff": row["summary"]["moment_mean_diff"],
            "moment_mean_log_ratio": row["summary"]["moment_mean_log_ratio"],
            "moment_identifiable_both_folds": row["summary"]["moment_identifiable_both_folds"],
        })
    return sorted(rows, key=lambda r: r["session"])


def main() -> None:
    if sha256_file(RULE_PATH) != RULE_HASH:
        raise SystemExit("frozen adjudication rule hash mismatch; refusing to fit real data")
    data_root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not data_root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")

    rows = registry_sessions(Path(data_root))
    if not rows:
        raise SystemExit("no DANDI 000574 sessions found in provenance/canonical_recording_registry.json")

    sessions: dict[str, dict] = {}
    for row in rows:
        key = row["session"]
        path = Path(data_root) / row["path"]
        if not path.is_file():
            sessions[key] = {"status": "excluded", "reason": f"registry path not found on disk: {path}"}
            continue
        print(f"fitting {key}", flush=True)
        sessions[key] = analyze_session(path, seed=session_seed(key))

    complete = {key: row for key, row in sessions.items() if row["status"] == "complete"}
    if not complete:
        raise SystemExit("no Boran session produced a dual-modality confinement fit")

    metric_names = (
        "moment_mean_diff", "moment_mean_log_ratio", "moment_ci_overlap_fraction",
        "state_mean_diff", "state_mean_log_ratio", "state_ci_overlap_fraction",
    )
    patient_metrics = patient_level_means(complete, metric_names)
    group = {}
    rng = np.random.default_rng(20260801)
    for name in metric_names:
        values = np.array([
            row[name] for row in patient_metrics.values() if row[name] is not None
        ])
        if not len(values):
            group[name] = {"status": "not_estimable", "n_patients": 0}
            continue
        group[name] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "ci": bootstrap_mean(values, rng), "n_patients": int(len(values)),
            "values": values.tolist(),
        }

    identifiability_asymmetry = {
        key: sum(row["marginal_identifiability"][key] for row in complete.values())
        for key in ("n_folds", "spike_moment_identifiable", "lfp_moment_identifiable",
                    "spike_state_space_identifiable", "lfp_state_space_identifiable")
    }
    n_folds_total = identifiability_asymmetry["n_folds"]
    identifiability_asymmetry["spike_moment_rate"] = identifiability_asymmetry["spike_moment_identifiable"] / n_folds_total
    identifiability_asymmetry["lfp_moment_rate"] = identifiability_asymmetry["lfp_moment_identifiable"] / n_folds_total
    identifiability_asymmetry["spike_state_space_rate"] = identifiability_asymmetry["spike_state_space_identifiable"] / n_folds_total
    identifiability_asymmetry["lfp_state_space_rate"] = identifiability_asymmetry["lfp_state_space_identifiable"] / n_folds_total

    primary = group["moment_mean_log_ratio"]
    n_patients_primary = primary.get("n_patients", 0)
    # A joint-both-modalities-identifiable comparison is a strictly harder bar
    # than either modality's own identifiability; report which bottleneck is
    # binding rather than let a near-degenerate n=1/n=2 bootstrap CI (which is
    # mechanically almost guaranteed to "exclude zero" once it collapses to a
    # near-repeated point) masquerade as a disagreement finding.
    asymmetric = (
        identifiability_asymmetry["lfp_moment_rate"] < 0.5 * identifiability_asymmetry["spike_moment_rate"]
        or identifiability_asymmetry["lfp_state_space_rate"] < 0.5 * identifiability_asymmetry["spike_state_space_rate"]
    )
    if n_patients_primary == 0:
        adjudication = "not_estimable_no_patient_has_a_jointly_identifiable_fold"
    elif asymmetric:
        adjudication = (
            f"identifiability_asymmetry_precludes_agreement_test_n_{n_patients_primary}_patients"
        )
    elif n_patients_primary <= 1:
        adjudication = f"single_patient_point_estimate_only_n_{n_patients_primary}_no_ci_is_meaningful"
    elif n_patients_primary < DESCRIPTIVE_ONLY_PATIENT_FLOOR:
        # n=3-4 patients is not a population sample; report the point
        # estimate and bootstrap CI but do not claim population coverage.
        lo, hi = primary["ci"]
        adjudication = (
            f"descriptive_only_n_{n_patients_primary}_patients_"
            f"{'ci_excludes_zero_log_ratio' if not (lo <= 0.0 <= hi) else 'ci_includes_zero_log_ratio'}"
        )
    else:
        lo, hi = primary["ci"]
        adjudication = "agreement_not_excluded" if lo <= 0.0 <= hi else "disagreement_detected"

    output = {
        "schema_version": "1.0.0", "analysis_id": "boran_modality_consistency",
        "dataset": "DANDI 000574 (Boran verbal Sternberg) -- co-located single units and iEEG",
        "canonical_role": (
            "within-patient spike-vs-LFP confinement-rate agreement test. Spiking and iEEG are two "
            "views of the SAME patients and SAME trials, never independent replications."
        ),
        "independent_unit": "patient, with sessions nested within patient",
        "primary_estimator": "moment (autocovariance/variance-function, lag-0 excluded)",
        "secondary_estimator": "state-space (Gaussian LGSSM, separated process/observation noise)",
        "comparison_metric": "log(spike_lambda / lfp_lambda), computed per fold where BOTH modalities are identifiable, then averaged within session, then within patient",
        "adjudication": adjudication,
        "small_n_caution": (
            f"{n_patients_primary} patients with at least one jointly-identifiable fold. Below "
            f"{DESCRIPTIVE_ONLY_PATIENT_FLOOR} patients this is reported as descriptive case "
            "evidence, not a population-level test."
        ),
        "code_commit": git_commit(ROOT), "decision_rule_hash": RULE_HASH,
        "source_hash": sha256_file(Path(__file__)),
        "preprocessing_provenance": {
            "session_selection": "provenance/canonical_recording_registry.json filtered to release=='000574'",
            "spike_arm": "scripts/run_human_drift_spine_000574.py machinery (firing-rate QC, PSTH, discriminant direction, LOO residuals, state-space/OU-moment estimators)",
            "lfp_arm": (
                "src/preprocessing.py load_boran_nwb(reject_channels=True, mains_hz=50.0) for "
                "channel QC, then per-trial line_noise_notch(50 Hz, 3 harmonics) -> "
                "bipolar_reference_by_shank -> compute_boran_hgp (70-150 Hz Hilbert envelope), "
                "matching scripts/run_boran_pipeline.py's established convention, sliced to the "
                "maintenance window and block-averaged into the same 100 ms bins as the spike PSTH"
            ),
            "mains_hz": BORAN_MAINS_HZ,
            "bad_channel_mad_threshold": BAD_CHANNEL_MAD_THRESHOLD,
            "note_on_task_premise": (
                "Boran (Sarnthein lab, University Hospital Zurich) is a 50 Hz-mains site, not a "
                "US 60 Hz site -- mains_hz=50.0 is used throughout, including inside the shared "
                "reject_bad_channels line-noise criterion"
            ),
        },
        "identifiability_asymmetry": identifiability_asymmetry,
        "sessions": sessions,
        "patient_level_metrics": patient_metrics,
        "group": group,
        "spike_lfp_disagreement_diagnostics_by_session": chase_diagnostics(complete),
        "n_sessions_complete": len(complete),
        "n_sessions_total": len(sessions),
    }
    destination = ROOT / "results" / "boran_modality_consistency.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "n_complete_sessions": len(complete), "n_patients": len(patient_metrics),
        "adjudication": adjudication, "group_moment_log_ratio": primary,
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
