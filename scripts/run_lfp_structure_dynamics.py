#!/usr/bin/env python3
"""Per-structure confinement rate (lambda) and diffusion (D) for the LFP
corpora the anatomical census marks `labelled`: ds004752
(15 patients), ds005489 (37), ds005557 (16).

Reuses the state-space machinery exactly as
scripts/run_cross_modality_calibration.py's rungs already do (leave-one-out
condition residuals -> fit_gaussian_state_space on binned high-gamma power),
extended from that script's coarse mtl/cortical split to the full
fine-grained structure vocabulary
(spike_pipeline.normalize_region_label). Does NOT port the spike-based
content/M2/M4/DA battery to LFP: lambda and D are the
only estimands fit here.

ds004752: verbal-Sternberg maintenance window (2.0-5.0 s post-encoding,
matching run_cross_modality_calibration.py), condition = set_size, exactly
as that script already fits -- this script only widens the region grouping
from 2 buckets to the full per-channel structure.

ds005489/ds005557 (RAM): these datasets have no working-memory delay period
at all -- free recall with encoding-period stimulation (see the SCOPE
BOUNDARY note in run_ram_openloop_pipeline.py). The nearest analogue to a
retention interval in this task is the DISTRACT_START/DISTRACT_END math
distractor phase between list presentation and recall (a delayed-recall
retention interval under active interference, ~20-30 s per list) -- there is
no trial-level condition label available here (no repeated item, no
set-size manipulation), so residuals are leave-one-out deviations from the
single across-epoch mean, not from a per-condition centroid. This is a
declared, honest substitution of estimand -- not a working-memory
maintenance delay -- stated explicitly in every output row, per this project's own standing
instruction not to silently port an estimand that does not transfer.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import fit_gaussian_state_space, leave_one_out_condition_residuals  # noqa: E402
from preprocessing import high_gamma_power, line_noise_notch  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from spike_pipeline import normalize_region_label  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402

RESULTS = ROOT / "results"
SEED = 20260804
BIN_S_STERNBERG = 0.1
MAINT_ONSET_S_DS004752 = 2.0
MAINT_DUR_S_DS004752 = 3.0
MAINS_HZ_ZURICH = 50.0
MAINS_HZ_US = 60.0
RAM_DISTRACTOR_WINDOW_S = 15.0
RAM_BIN_S = 0.5
RAM_MIN_TRIALS = 6


def _data_root(name: str) -> Path:
    import os

    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("WM_DYNAMICS_DATA_ROOT is not set")
    return Path(root) / name


def _fit_lambda_diffusion(
    trial_bins: np.ndarray, condition: np.ndarray, bin_s: float, min_trials: int,
) -> dict[str, Any]:
    complete = np.all(np.isfinite(trial_bins), axis=1)
    trial_bins, condition = trial_bins[complete], condition[complete]
    if len(trial_bins) < min_trials:
        return {
            "status": "not_identifiable", "reason": f"fewer than {min_trials} complete trials",
            "lambda_rate": None, "diffusion": None, "n_trials": int(len(trial_bins)),
        }
    residuals, _ = leave_one_out_condition_residuals(trial_bins, condition)
    keep = np.all(np.isfinite(residuals), axis=1)
    if int(keep.sum()) < min_trials:
        return {
            "status": "not_identifiable", "reason": "fewer than min_trials survive leave-one-out residualization",
            "lambda_rate": None, "diffusion": None, "n_trials": int(keep.sum()),
        }
    estimate = fit_gaussian_state_space(residuals[keep], bin_s)
    return {
        "status": estimate.status, "lambda_rate": estimate.lambda_rate,
        "diffusion": estimate.diffusion, "n_trials": int(keep.sum()),
    }


def bootstrap_summary_by_patient(per_patient: dict[str, float], rng: np.random.Generator) -> dict[str, Any]:
    return bootstrap_summary(list(per_patient.values()), rng)


def bootstrap_summary(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(finite) < 2:
        return {"status": "non_identified", "n_patients": int(len(finite)),
                "reason": f"fewer than 2 finite patient estimates ({len(finite)})"}
    _, lo, hi = bootstrap_ci(finite, np.mean, n_boot=5000, rng=rng)
    return {
        "status": "estimable", "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "patient_bootstrap_ci95": [float(lo), float(hi)], "n_patients": int(len(finite)),
    }


# ---------------------------------------------------------------------------
# ds004752 -- verbal Sternberg maintenance window, fine-grained structures
# ---------------------------------------------------------------------------

def _read_edf_tolerant_ds004752(edf: Path):
    """See run_cross_modality_calibration.py -- some ds004752 EDF headers
    encode a start-time seconds field of 60 (writer rounding artifact)."""
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


def _ds004752_session_paths(data_root: Path, patient: str) -> list[Path]:
    return sorted((data_root / patient).glob("ses-*"))


def _load_ds004752_session(ieeg_dir: Path, patient: str, session: str):
    edf = ieeg_dir / f"{patient}_{session}_task-verbalWM_run-01_ieeg.edf"
    electrodes_tsv = ieeg_dir / f"{patient}_{session}_task-verbalWM_run-01_electrodes.tsv"
    raw = _read_edf_tolerant_ds004752(edf)
    data = raw.get_data().T
    srate = float(raw.info["sfreq"])
    structures: dict[str, str] = {}
    with open(electrodes_tsv) as f:
        header = f.readline().strip().split("\t")
        loc_idx = header.index("AnatomicalLocation")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            raw_label = parts[loc_idx]
            structure, _ = normalize_region_label(raw_label, "bids_brainnetome_anatomical_location")
            structures[parts[0]] = structure
    return data, raw.ch_names, srate, structures


def _ds004752_events(events_dir: Path, patient: str, session: str) -> dict[str, np.ndarray]:
    path = events_dir / f"{patient}_{session}_task-verbalWM_run-01_events.tsv"
    with open(path) as f:
        header = f.readline().strip().split("\t")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    cols = {name: [] for name in header}
    for row in rows:
        for name, value in zip(header, row):
            cols[name].append(value)
    return {
        "onset": np.array([float(v) for v in cols["onset"]]),
        "set_size": np.array([int(v) for v in cols["SetSize"]]),
        "artifact": np.array([int(v) for v in cols.get("Artifact", ["0"] * len(rows))]),
    }


def _bin_power_by_trial(power: np.ndarray, srate: float, onsets: np.ndarray, onset_s: float, dur_s: float, bin_s: float) -> np.ndarray:
    n_bins = int(round(dur_s / bin_s))
    trial_bins = np.full((len(onsets), n_bins), np.nan)
    bin_samples = int(round(bin_s * srate))
    start_sample_offset = int(round(onset_s * srate))
    for i, onset in enumerate(onsets):
        start = int(round(onset * srate)) + start_sample_offset
        for b in range(n_bins):
            lo = start + b * bin_samples
            hi = lo + bin_samples
            if hi > power.shape[0]:
                break
            trial_bins[i, b] = power[lo:hi].mean()
    return trial_bins


def run_ds004752() -> dict[str, Any]:
    data_root = _data_root("ds004752")
    patients = [f"sub-{i:02d}" for i in range(1, 16)]
    per_structure: dict[str, dict[str, float]] = {}
    per_structure_diffusion: dict[str, dict[str, float]] = {}
    per_patient_debug: dict[str, Any] = {}
    for patient in patients:
        if not (data_root / patient).is_dir():
            continue
        structure_fits: dict[str, list[dict]] = {}
        for session_dir in _ds004752_session_paths(data_root, patient):
            ieeg_dir = session_dir / "ieeg"
            if not ieeg_dir.is_dir():
                continue
            session = session_dir.name
            try:
                data, ch_names, srate, structures = _load_ds004752_session(ieeg_dir, patient, session)
                ev = _ds004752_events(ieeg_dir, patient, session)
            except FileNotFoundError:
                continue
            by_structure: dict[str, list[int]] = {}
            for i, ch in enumerate(ch_names):
                s = structures.get(ch)
                if s is not None:
                    by_structure.setdefault(s, []).append(i)
            keep_trials = ev["artifact"] == 0
            for structure, idx in by_structure.items():
                selected = data[:, idx]
                selected = line_noise_notch(selected, srate, MAINS_HZ_ZURICH)
                power = high_gamma_power(selected, srate).mean(axis=1)
                trial_bins = _bin_power_by_trial(
                    power, srate, ev["onset"][keep_trials],
                    MAINT_ONSET_S_DS004752, MAINT_DUR_S_DS004752, BIN_S_STERNBERG,
                )
                fit = _fit_lambda_diffusion(
                    trial_bins, ev["set_size"][keep_trials], BIN_S_STERNBERG, min_trials=6,
                )
                structure_fits.setdefault(structure, []).append(fit)
        per_patient_debug[patient] = {
            s: [f["status"] for f in fits] for s, fits in structure_fits.items()
        }
        for structure, fits in structure_fits.items():
            identifiable = [f for f in fits if f["status"] == "identifiable"]
            if identifiable:
                per_structure.setdefault(structure, {})[patient] = (
                    float(np.mean([f["lambda_rate"] for f in identifiable]))
                )
                per_structure_diffusion.setdefault(structure, {})[patient] = (
                    float(np.mean([f["diffusion"] for f in identifiable]))
                )

    rng = np.random.default_rng(SEED)
    structures_out = {}
    for structure in sorted(set(per_structure) | set(per_structure_diffusion)):
        structures_out[structure] = {
            "lambda_s_per_s": bootstrap_summary_by_patient(per_structure.get(structure, {}), rng),
            "diffusion_state_units_sq_per_s": bootstrap_summary_by_patient(
                per_structure_diffusion.get(structure, {}), rng,
            ),
            "per_patient_lambda": per_structure.get(structure, {}),
            "per_patient_diffusion": per_structure_diffusion.get(structure, {}),
        }
    return {
        "status": "estimable" if structures_out else "non_identified",
        "condition_used": "set_size (Sternberg maintenance window, matching run_cross_modality_calibration.py)",
        "structures": structures_out,
        "per_patient_debug_status": per_patient_debug,
    }


# ---------------------------------------------------------------------------
# RAM (ds005489, ds005557) -- math-distractor retention interval, DK/hand-annotated structures
# ---------------------------------------------------------------------------

def _ram_electrode_structures(electrodes_tsv: Path) -> dict[str, str]:
    """One resolved structure per CONTACT (not per bipolar channel), preferring
    stein.region > das.region > wb.region > ind.region -- the same column
    priority scripts/build_anatomical_census.py's RAM census already uses,
    to recover MTL structures the DK surface pipeline (ind.region) alone
    cannot represent."""
    with open(electrodes_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        contacts = {}
        for row in reader:
            raw, source_col = None, None
            for col in ("stein.region", "das.region", "wb.region", "ind.region"):
                v = (row.get(col) or "").strip()
                if v and v.lower() != "n/a":
                    raw, source_col = v, col
                    break
            if raw is None:
                contacts[row["name"]] = "unlabelled"
                continue
            structure, _ = normalize_region_label(raw, "bids_desikan_killiany_ind_region")
            contacts[row["name"]] = structure
    return contacts


def _ram_channel_structures(channel_names: list[str], contact_structures: dict[str, str]) -> dict[str, str]:
    """Bipolar channel "A1-A2" -> structure: agreement of both contacts if
    they agree, else the anode (first) contact's structure (documented, not
    silently resolved)."""
    out = {}
    for ch in channel_names:
        parts = ch.split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        sa = contact_structures.get(a)
        sb = contact_structures.get(b)
        if sa is None and sb is None:
            continue
        if sa is not None and sa == sb:
            out[ch] = sa
        else:
            out[ch] = sa if sa is not None else sb
    return out


def _ram_distractor_epochs(events_tsv: Path) -> list[tuple[float, float]]:
    with open(events_tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    starts = [float(r["onset"]) for r in rows if r["trial_type"] == "DISTRACT_START"]
    ends = [float(r["onset"]) for r in rows if r["trial_type"] == "DISTRACT_END"]
    return list(zip(starts, ends))


def _fit_ram_session(ieeg_dir: Path, stem: str, mains_hz: float) -> dict[str, dict]:
    """``stem`` is the events-file stem (e.g. ``sub-X_ses-0_task-FR2``, no
    ``acq-`` component -- events/electrodes are acquisition-independent);
    the bipolar EDF/channels files carry an extra ``_acq-bipolar`` segment
    not present in the events or electrodes filenames."""
    edf = ieeg_dir / f"{stem}_acq-bipolar_ieeg.edf"
    channels_tsv = ieeg_dir / f"{stem}_acq-bipolar_channels.tsv"
    events_tsv = ieeg_dir / f"{stem}_events.tsv"
    electrodes_candidates = sorted(ieeg_dir.glob(f"{stem}*electrodes.tsv"))
    if not (edf.is_file() and channels_tsv.is_file() and events_tsv.is_file() and electrodes_candidates):
        return {}
    contact_structures = _ram_electrode_structures(electrodes_candidates[0])
    with open(channels_tsv) as f:
        channel_names = [row["name"] for row in csv.DictReader(f, delimiter="\t")]
    channel_structures = _ram_channel_structures(channel_names, contact_structures)
    epochs = _ram_distractor_epochs(events_tsv)
    epochs = [(s, e) for s, e in epochs if e - s >= RAM_DISTRACTOR_WINDOW_S]
    if not epochs or not channel_structures:
        return {}

    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
    srate = float(raw.info["sfreq"])
    data = raw.get_data().T  # (T, C)
    ch_index = {name: i for i, name in enumerate(raw.ch_names)}

    by_structure: dict[str, list[int]] = {}
    for ch, structure in channel_structures.items():
        idx = ch_index.get(ch)
        if idx is not None:
            by_structure.setdefault(structure, []).append(idx)

    n_bins = int(round(RAM_DISTRACTOR_WINDOW_S / RAM_BIN_S))
    bin_samples = int(round(RAM_BIN_S * srate))
    fits = {}
    for structure, idx in by_structure.items():
        selected = data[:, idx]
        selected = line_noise_notch(selected, srate, mains_hz)
        power = high_gamma_power(selected, srate).mean(axis=1)
        trial_bins = np.full((len(epochs), n_bins), np.nan)
        for i, (start_s, _end_s) in enumerate(epochs):
            start_sample = int(round(start_s * srate))
            for b in range(n_bins):
                lo = start_sample + b * bin_samples
                hi = lo + bin_samples
                if hi > power.shape[0]:
                    break
                trial_bins[i, b] = power[lo:hi].mean()
        condition = np.zeros(len(epochs), dtype=int)  # no trial-level label available; single group
        fits[structure] = _fit_lambda_diffusion(trial_bins, condition, RAM_BIN_S, RAM_MIN_TRIALS)
    return fits


def run_ram(dataset_name: str, mains_hz: float = MAINS_HZ_US) -> dict[str, Any]:
    data_root = _data_root(dataset_name)
    per_structure: dict[str, dict[str, float]] = {}
    per_structure_diffusion: dict[str, dict[str, float]] = {}
    per_patient_debug: dict[str, Any] = {}
    patient_dirs = sorted(data_root.glob("sub-*"))
    for patient_dir in patient_dirs:
        patient = patient_dir.name
        structure_fits: dict[str, list[dict]] = {}
        for ieeg_dir in sorted(patient_dir.glob("ses-*/ieeg")):
            for events_tsv in sorted(ieeg_dir.glob("*_events.tsv")):
                stem = events_tsv.name.replace("_events.tsv", "")
                try:
                    fits = _fit_ram_session(ieeg_dir, stem, mains_hz)
                except Exception as exc:  # noqa: BLE001 -- report and continue, one bad session must not stop the run
                    fits = {"_error": str(exc)}
                for structure, fit in fits.items():
                    if structure == "_error":
                        continue
                    structure_fits.setdefault(structure, []).append(fit)
        per_patient_debug[patient] = {
            s: [f["status"] for f in fits] for s, fits in structure_fits.items()
        }
        for structure, fits in structure_fits.items():
            identifiable = [f for f in fits if f["status"] == "identifiable"]
            if identifiable:
                per_structure.setdefault(structure, {})[patient] = (
                    float(np.mean([f["lambda_rate"] for f in identifiable]))
                )
                per_structure_diffusion.setdefault(structure, {})[patient] = (
                    float(np.mean([f["diffusion"] for f in identifiable]))
                )

    rng = np.random.default_rng(SEED)
    structures_out = {}
    for structure in sorted(set(per_structure) | set(per_structure_diffusion)):
        structures_out[structure] = {
            "lambda_s_per_s": bootstrap_summary_by_patient(per_structure.get(structure, {}), rng),
            "diffusion_state_units_sq_per_s": bootstrap_summary_by_patient(
                per_structure_diffusion.get(structure, {}), rng,
            ),
            "per_patient_lambda": per_structure.get(structure, {}),
            "per_patient_diffusion": per_structure_diffusion.get(structure, {}),
        }
    return {
        "status": "estimable" if structures_out else "non_identified",
        "condition_used": (
            "none -- no trial-level label exists in this task; leave-one-out residuals are taken "
            "against the single across-epoch mean, not a per-condition centroid"
        ),
        "retention_interval_substitution_note": (
            "This dataset has no working-memory maintenance/delay period (free recall with "
            "encoding-period stimulation). The DISTRACT_START/DISTRACT_END math-distractor phase "
            "(delayed-recall retention interval under active interference, first "
            f"{RAM_DISTRACTOR_WINDOW_S:.0f} s of each list's distractor epoch) is used as the nearest "
            "available retention interval. This is a declared estimand substitution, not a "
            "working-memory delay-period confinement rate -- this project forbids porting an "
            "estimand without saying why it transfers; the honest answer here is that lambda/D from "
            "this window characterize distractor-period LFP dynamics under interference, not WM "
            "maintenance per se."
        ),
        "structures": structures_out,
        "n_patients_attempted": len(patient_dirs),
        "per_patient_debug_status": per_patient_debug,
    }


def main() -> None:
    print("Fitting ds004752 (fine-grained, Sternberg maintenance window)...", flush=True)
    ds004752 = run_ds004752()
    print("Fitting ds005489 (RAM open-loop, distractor retention interval)...", flush=True)
    ds005489 = run_ram("ds005489-download", mains_hz=MAINS_HZ_US)
    print("Fitting ds005557 (RAM closed-loop, distractor retention interval)...", flush=True)
    ds005557 = run_ram("ds005557-download", mains_hz=MAINS_HZ_US)

    output = {
        "schema_version": "1.0.0", "analysis_id": "lfp_structure_dynamics",
        "code_commit": git_commit(ROOT),
        "estimands_present": [
            "lambda_s_per_s (confinement rate, state-space M2 fit on high-gamma power)",
            "diffusion_state_units_sq_per_s",
        ],
        "estimands_not_ported_from_spike_pipeline": (
            "content/M2/M4/DA battery NOT fit here -- lambda and D are the only "
            "transferable quantities to LFP band power at this trial-count regime"
        ),
        "units_note": (
            "lambda in s^-1; D in (state units)^2/s where the state unit is log high-gamma (70-150 Hz) "
            "power, averaged across the channels resolved to a given structure -- NOT PCA-projected, "
            "unlike the spike-based drift spine scripts. Not comparable in absolute units across "
            "structures or datasets without further standardization."
        ),
        "datasets": {
            "ds004752": ds004752,
            "ds005489_openloop": ds005489,
            "ds005557_closedloop": ds005557,
        },
    }
    destination = RESULTS / "lfp_structure_dynamics.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(destination),
        "ds004752_structures": list(ds004752.get("structures", {})),
        "ds005489_structures": list(ds005489.get("structures", {})),
        "ds005557_structures": list(ds005557.get("structures", {})),
    }, indent=2))


if __name__ == "__main__":
    main()
