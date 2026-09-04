#!/usr/bin/env python3
"""Cross-modality confinement-rate calibration.

Question: does a non-invasive measurement recover the intracranial
confinement rate at all? This is a calibration, not a discovery claim.

Four rungs, same lab (Sarnthein, Zurich), same verbal Sternberg task
(encoding 2s, maintenance 3s, probe), on the 9 patients shared between the
already-staged Boran/DANDI-000574 release and OpenNeuro ds004752
(Dimakopoulos et al., eLife 2022, doi 10.7554/eLife.78677):

  1. MTL units (Boran spikes, hippocampus/amygdala) -- intracranial,
     permanently invasive-only reference (2C.4). Reused unchanged from
     `results/boran_modality_consistency.json`'s spike arm; no new fit.
  2. MTL depth LFP (ds004752 iEEG, Hipp/Amyg-labelled contacts) --
     intracranial, a second, independent MTL measurement of the same
     patients for cross-checking rung 1, also permanently invasive-only.
  3. Cortical depth LFP (ds004752 iEEG, non-MTL labelled contacts along
     the same depth probes: STG/MTG/ITG/FuG/PhG/etc.) -- intracranial
     cortical reference. This is the "intracranial cortical lambda" that
     2C.3's licensing decision is measured against.
  4. Beamformed cortical virtual sensors (ds004752 derivatives, LCMV
     sources at DLPFC/OFC/PPC/AC/V1) -- non-invasive cortical proxy, the
     licensing candidate. Source data is shipped resampled to 200 Hz, so
     this rung uses 30-70 Hz gamma power, not the 70-150 Hz high-gamma
     used for the invasive rungs -- a data-rate limit, not a methods choice.
  5. Scalp EEG (10-20 montage carried in the same derivatives file as
     rung 4) -- non-invasive whole-head, reported descriptively.

2C.3's predeclared decision: a whole-cortex non-invasive tier is licensed
only if rung 4 recovers rung 3's cross-patient lambda ordering with a
patient-bootstrap Spearman-rank-correlation interval excluding zero.

2C.4's honest limit: rungs 1-2 (MTL) are never validated by any
non-invasive rung, pass or fail. Even a passing 2C.3 calibration licenses
the cortical branch only.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import fit_gaussian_state_space, leave_one_out_condition_residuals  # noqa: E402
from preprocessing import high_gamma_power, line_noise_notch, band_power  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402

mne.set_log_level("ERROR")

SEED = 20260803
BIN_S = 0.1
MAINT_ONSET_S = 2.0
MAINT_DUR_S = 3.0
MAINS_HZ = 50.0  # Zurich site, not 60 Hz
SHARED_PATIENTS = [f"sub-{i:02d}" for i in range(1, 10)]


def _data_root() -> Path:
    import os

    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("WM_DYNAMICS_DATA_ROOT is not set")
    return Path(root) / "ds004752"


def _region_of(label: str) -> str | None:
    label = label.strip().lower()
    if label == "no_label_found":
        return None
    tag = label.split(",")[0].strip()
    if tag in ("hipp", "amyg"):
        return "mtl"
    return "cortical"


def _session_paths(data_root: Path, patient: str) -> list[Path]:
    return sorted((data_root / patient).glob("ses-*"))


def _read_edf_tolerant(edf: Path):
    """Some ds004752 EDF headers encode a start-time seconds field of 60
    (writer rounding artifact, e.g. sub-06/ses-04), which mne rejects. Patch
    just that ASCII header field to 59 in a scratch copy and retry; a 1s
    offset in the recording's absolute start time has no effect on this
    analysis, which only uses relative within-session timing."""
    try:
        return mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    except ValueError as exc:
        if "second must be in 0..59" not in str(exc):
            raise
        raw_bytes = bytearray(edf.read_bytes())
        raw_bytes[176:184] = raw_bytes[176:184].replace(b".60", b".59")
        patched = Path(tempfile.mkstemp(suffix=".edf")[1])
        patched.write_bytes(raw_bytes)
        try:
            return mne.io.read_raw_edf(patched, preload=True, verbose="ERROR")
        finally:
            patched.unlink()


def _load_ieeg_session(ieeg_dir: Path, patient: str, session: str) -> tuple[np.ndarray, list[str], float, dict[str, str | None]]:
    edf = ieeg_dir / f"{patient}_{session}_task-verbalWM_run-01_ieeg.edf"
    electrodes_tsv = ieeg_dir / f"{patient}_{session}_task-verbalWM_run-01_electrodes.tsv"
    raw = _read_edf_tolerant(edf)
    data = raw.get_data().T  # (T, C)
    srate = float(raw.info["sfreq"])
    regions = {}
    with open(electrodes_tsv) as f:
        header = f.readline().strip().split("\t")
        loc_idx = header.index("AnatomicalLocation")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            regions[parts[0]] = _region_of(parts[loc_idx])
    return data, raw.ch_names, srate, regions


def _events(events_dir: Path, patient: str, session: str) -> dict[str, np.ndarray]:
    path = events_dir / f"{patient}_{session}_task-verbalWM_run-01_events.tsv"
    with open(path) as f:
        header = f.readline().strip().split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    cols = {name: [] for name in header}
    for row in rows:
        for name, value in zip(header, row):
            cols[name].append(value)
    out = {"onset": np.array([float(v) for v in cols["onset"]])}
    out["set_size"] = np.array([int(v) for v in cols["SetSize"]])
    out["artifact"] = np.array([int(v) for v in cols.get("Artifact", ["0"] * len(rows))])
    return out


def _bin_power_by_trial(power: np.ndarray, srate: float, onsets: np.ndarray) -> np.ndarray:
    """(trials, bins) mean power in the maintenance window, 100 ms bins."""
    n_bins = int(round(MAINT_DUR_S / BIN_S))
    trial_bins = np.full((len(onsets), n_bins), np.nan)
    bin_samples = int(round(BIN_S * srate))
    start_sample_offset = int(round(MAINT_ONSET_S * srate))
    for i, onset in enumerate(onsets):
        start = int(round(onset * srate)) + start_sample_offset
        for b in range(n_bins):
            lo = start + b * bin_samples
            hi = lo + bin_samples
            if hi > power.shape[0]:
                break
            trial_bins[i, b] = power[lo:hi].mean()
    return trial_bins


def _fit_lambda(trial_bins: np.ndarray, set_size: np.ndarray) -> dict[str, Any]:
    complete = np.all(np.isfinite(trial_bins), axis=1)
    trial_bins, set_size = trial_bins[complete], set_size[complete]
    if len(trial_bins) < 6:
        return {"status": "not_identifiable", "reason": "fewer than 6 complete trials", "lambda_rate": None}
    residuals, _ = leave_one_out_condition_residuals(trial_bins, set_size)
    keep = np.all(np.isfinite(residuals), axis=1)
    estimate = fit_gaussian_state_space(residuals[keep], BIN_S)
    return {"status": estimate.status, "lambda_rate": estimate.lambda_rate, "n_trials": int(keep.sum())}


def _fit_ieeg_rung(data_root: Path, patient: str, region_filter: str) -> dict[str, Any]:
    sessions = _session_paths(data_root, patient)
    session_fits = {}
    for session_dir in sessions:
        session = session_dir.name
        ieeg_dir = session_dir / "ieeg"
        if not ieeg_dir.is_dir():
            continue
        try:
            data, ch_names, srate, regions = _load_ieeg_session(ieeg_dir, patient, session)
            ev = _events(ieeg_dir, patient, session)
        except FileNotFoundError:
            continue
        channel_idx = [i for i, ch in enumerate(ch_names) if regions.get(ch) == region_filter]
        if len(channel_idx) == 0:
            session_fits[session] = {"status": "not_identifiable", "reason": "no channels in region", "lambda_rate": None}
            continue
        selected = data[:, channel_idx]
        selected = line_noise_notch(selected, srate, MAINS_HZ)
        power = high_gamma_power(selected, srate).mean(axis=1, keepdims=True)
        keep_trials = ev["artifact"] == 0
        trial_bins = _bin_power_by_trial(power[:, 0], srate, ev["onset"][keep_trials])
        session_fits[session] = _fit_lambda(trial_bins, ev["set_size"][keep_trials])
    return session_fits


def _fit_beamformed_and_scalp(data_root: Path, patient: str) -> dict[str, dict[str, Any]]:
    mat_path = data_root / "derivatives" / patient / "beamforming" / f"{patient}-task-verbalWM-LCMVsources.mat"
    if not mat_path.is_file():
        return {"beamformed_cortical": {"status": "not_identifiable", "reason": "no derivatives file", "lambda_rate": None},
                "scalp": {"status": "not_identifiable", "reason": "no derivatives file", "lambda_rate": None}}
    mat = sio.loadmat(mat_path, simplify_cells=True)
    epoch = mat["LCMV_maintenance"]
    labels = list(epoch["label"])
    fsample = float(epoch["fsample"])
    trials = epoch["trial"]  # list of (n_channels, n_time)
    beamformed_sources = ["DLPFC", "OFC", "PPC", "AC", "V1"]
    beam_idx = [labels.index(s) for s in beamformed_sources if s in labels]
    scalp_idx = [i for i, l in enumerate(labels) if l not in beamformed_sources]

    def _rung(indices: list[int], band: str) -> dict[str, Any]:
        if not indices:
            return {"status": "not_identifiable", "reason": "no channels", "lambda_rate": None}
        epoch_duration_s = float(epoch["time"][0][-1] - epoch["time"][0][0])
        n_bins = int(epoch_duration_s / BIN_S)
        bin_samples = int(round(BIN_S * fsample))
        trial_bins = np.full((len(trials), n_bins), np.nan)
        for i, trial in enumerate(trials):
            sig_2d = np.asarray(trial)[indices].T  # (T, C)
            power = band_power(sig_2d, band, fsample).mean(axis=1)
            for b in range(n_bins):
                lo, hi = b * bin_samples, (b + 1) * bin_samples
                if hi > len(power):
                    break
                trial_bins[i, b] = power[lo:hi].mean()
        return _fit_lambda(trial_bins, np.zeros(len(trials), dtype=int))

    return {
        # LCMV sources are resampled to 200 Hz (Nyquist 100 Hz), so true
        # high-gamma (70-150 Hz) is unavailable; gamma (30-70 Hz) is the
        # highest fully-available band -- a genuine data-rate limit, not a
        # methodological choice to match the invasive rungs' 70-150 Hz.
        "beamformed_cortical": _rung(beam_idx, "gamma"),
        "scalp": _rung(scalp_idx, "alpha"),
    }


def _mtl_units_reference(patient: str) -> dict[str, Any]:
    ref_path = ROOT / "results" / "boran_modality_consistency.json"
    ref = json.loads(ref_path.read_text())
    lambdas = []
    for session_key, session in ref["sessions"].items():
        if not session_key.startswith(patient + "_"):
            continue
        for fold in session.get("folds", []):
            state = fold.get("spike", {}).get("state_space", {})
            if state.get("status") == "identifiable" and state.get("lambda_rate") is not None:
                lambdas.append(float(state["lambda_rate"]))
    if len(lambdas) == 0:
        return {"status": "not_identifiable", "reason": "no identifiable spike-arm fold", "lambda_rate": None}
    return {"status": "identifiable", "lambda_rate": float(np.mean(lambdas)), "n_folds": len(lambdas)}


def _aggregate_sessions(session_fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    identifiable = [
        f["lambda_rate"] for f in session_fits.values()
        if f.get("status") == "identifiable" and f.get("lambda_rate") is not None
    ]
    if not identifiable:
        return {"status": "not_identifiable", "lambda_rate": None, "n_identifiable_sessions": 0, "n_sessions": len(session_fits)}
    return {
        "status": "identifiable",
        "lambda_rate": float(np.mean(identifiable)),
        "n_identifiable_sessions": len(identifiable),
        "n_sessions": len(session_fits),
    }


def _spearman_rank_correlation(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    if len(pairs) < 4:
        return {"status": "non_identified", "reason": "fewer than 4 patients with both rungs identifiable", "n_patients": len(pairs)}
    arr = np.asarray(pairs, dtype=float)
    rng = np.random.default_rng(SEED)
    rho, lower, upper = bootstrap_ci(arr, lambda d: spearmanr(d[:, 0], d[:, 1]).statistic, rng=rng)
    return {
        "status": "estimable",
        "n_patients": len(pairs),
        "spearman_rho": rho,
        "spearman_rho_patient_bootstrap_ci": [lower, upper],
        "excludes_zero": bool(lower > 0 or upper < 0),
    }


def main() -> None:
    data_root = _data_root()
    per_patient: dict[str, Any] = {}
    for patient in SHARED_PATIENTS:
        if not (data_root / patient).is_dir():
            per_patient[patient] = {"status": "not_downloaded"}
            continue
        mtl_units = _mtl_units_reference(patient)
        mtl_depth = _aggregate_sessions(_fit_ieeg_rung(data_root, patient, "mtl"))
        cortical_depth = _aggregate_sessions(_fit_ieeg_rung(data_root, patient, "cortical"))
        beam_scalp = _fit_beamformed_and_scalp(data_root, patient)
        per_patient[patient] = {
            "mtl_units_boran": mtl_units,
            "mtl_depth_lfp_ds004752": mtl_depth,
            "cortical_depth_lfp_ds004752": cortical_depth,
            "beamformed_cortical_ds004752": beam_scalp["beamformed_cortical"],
            "scalp_ds004752": beam_scalp["scalp"],
        }
        print(json.dumps({patient: per_patient[patient]}, indent=2))

    def _pairs(rung_a: str, rung_b: str) -> list[tuple[float, float]]:
        out = []
        for patient, rungs in per_patient.items():
            if rungs.get("status") == "not_downloaded":
                continue
            a, b = rungs[rung_a], rungs[rung_b]
            if a.get("status") == "identifiable" and b.get("status") == "identifiable":
                out.append((a["lambda_rate"], b["lambda_rate"]))
        return out

    licensing_decision = _spearman_rank_correlation(_pairs("cortical_depth_lfp_ds004752", "beamformed_cortical_ds004752"))
    descriptive = {
        "mtl_units_vs_mtl_depth_lfp": _spearman_rank_correlation(_pairs("mtl_units_boran", "mtl_depth_lfp_ds004752")),
        "cortical_depth_lfp_vs_scalp": _spearman_rank_correlation(_pairs("cortical_depth_lfp_ds004752", "scalp_ds004752")),
        "beamformed_cortical_vs_scalp": _spearman_rank_correlation(_pairs("beamformed_cortical_ds004752", "scalp_ds004752")),
    }

    licensing_verdict = (
        "cortex_wide_tier_licensed"
        if licensing_decision.get("status") == "estimable" and licensing_decision.get("excludes_zero") and licensing_decision.get("spearman_rho", 0) > 0
        else "cortex_wide_tier_not_licensed"
    )

    output = {
        "schema_version": "1.0.0",
        "analysis_id": "cross_modality_calibration",
        "trigger": "non-invasive-vs-intracranial cross-modality calibration",
        "code_commit": git_commit(ROOT),
        "identifiers_verified": {
            "ds004752": "Dimakopoulos, Megevand, Stieglitz, Imbach, Sarnthein; eLife 2022; doi 10.7554/eLife.78677 -- confirmed via OpenNeuro GraphQL and Crossref",
            "boran_2020": "Boran, Fedele, Steiner, Hilfiker, Stieglitz, Grunwald, Sarnthein; Scientific Data 2020; doi 10.1038/s41597-020-0364-3 -- confirmed via Crossref; staged in this project as DANDI 000574",
        },
        "patient_overlap": {
            "method": "age/sex/pathology triple fingerprint match between ds004752 participants.tsv and DANDI 000574 NWB subject metadata, both sub-01..sub-09",
            "n_shared_patients": 9,
            "shared_patients": SHARED_PATIENTS,
            "ds004752_only_patients": [f"sub-{i:02d}" for i in range(10, 16)],
            "note": "recorded in config/datasets.json and results/patient_identity_audit.json per 2B.1",
        },
        "rungs": [
            "mtl_units_boran (spikes, hippocampus/amygdala, intracranial, permanently invasive-only)",
            "mtl_depth_lfp_ds004752 (iEEG high-gamma, Hipp/Amyg-labelled contacts, intracranial, permanently invasive-only)",
            "cortical_depth_lfp_ds004752 (iEEG high-gamma, non-MTL-labelled contacts, intracranial cortical reference)",
            "beamformed_cortical_ds004752 (LCMV virtual sensors: DLPFC, OFC, PPC, AC, V1, non-invasive candidate)",
            "scalp_ds004752 (10-20 montage, alpha power, non-invasive whole-head)",
        ],
        "per_patient": per_patient,
        "licensing_decision_2C3": {
            "predeclared_rule": "cortex-wide non-invasive tier licensed only if beamformed_cortical recovers cortical_depth_lfp's cross-patient lambda ordering with a patient-bootstrap Spearman interval excluding zero",
            "result": licensing_decision,
            "verdict": licensing_verdict,
        },
        "descriptive_rank_correlations": descriptive,
        "honest_limit_2C4": (
            "No scalp or beamformed measurement resolves hippocampus or amygdala. mtl_units_boran and "
            "mtl_depth_lfp_ds004752 are never validated by any non-invasive rung in this artifact, "
            "regardless of the 2C.3 verdict above. Even a licensed cortex-wide tier covers cortical "
            "observables only (cortical_depth_lfp vs. beamformed_cortical/scalp); the MTL branch "
            "remains intracranial-only, permanently. The beamformed_cortical rung is also fit at "
            "30-70 Hz gamma, not 70-150 Hz high-gamma, because ds004752 ships its LCMV sources "
            "resampled to 200 Hz -- any lambda comparison against the invasive rungs (both fit at "
            "70-150 Hz) already carries this band mismatch, on top of the modality mismatch."
        ),
    }
    destination = ROOT / "results" / "cross_modality_calibration.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({"output": str(destination), "licensing_decision": licensing_decision, "verdict": licensing_verdict}, indent=2))


if __name__ == "__main__":
    main()
