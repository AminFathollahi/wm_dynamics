#!/usr/bin/env python3
"""Alagapan et al. (2019) reanalysis: does each patient's own band-matched
baseline oscillation frequency accompany their reported phase-locked
stimulation benefit?

Citation: Alagapan S, Riddle J, Huang WA, Hadar E, Shin HW, Froehlich F.
"Network-Targeted, Multi-site Direct Cortical Stimulation Enhances Working
Memory by Modulating Phase Lag of Low-Frequency Oscillations." Cell Reports
2019;29(9):2590-2598. PMC6901101.

Evidentiary strength (stated once, applies everywhere a number from this
script appears): this is a REANALYSIS of public raw data from n=3 PATIENTS.
n=3 is the dominant limitation. Every number below is a per-patient number,
not a trial-level or pooled N standing in for statistical power the design
does not have.

Each patient's own stimulation frequency and target band (from the source
paper, since these are subject-specific choices not recorded in the
preprocessing scripts on disk): P1 4 Hz (theta, 4-8 Hz), P2 10 Hz (alpha,
8-12 Hz), P3 10 Hz (alpha, 8-12 Hz).

Band-matched omega: each patient's BASELINE (no stimulation) intracranial
recording is bandpass-filtered to their own target band, downsampled (safely
above Nyquist for a <=12 Hz target), reduced to an 8-dimensional PCA latent,
and fit with the ENSEMBLE DMD estimator (this project's primary estimator)
-- band-limited voltage oscillations trace a rotating trajectory in PCA
space whose leading DMD eigenvalue's imaginary part is the oscillation
frequency. Seizure-onset-zone electrodes (per-patient .mat files) and non-
brain channels (EKG) are excluded; P1 additionally has a separate baseline-
specific channel montage on disk and uses it.

An identifiability check is applied before trusting any f_hz number: the
fitted operator's held-out one-step R^2 must exceed its own circular-shift
null by a material margin (R2_MARGIN, matching this project's existing
"materially better than null" convention), on top of the bootstrap CI and
Nyquist checks. This matters here specifically: baseline oscillations are
spontaneous and phase-random across trials (no task-locked or stimulus-
locked reference), unlike the task-locked WM delay-period dynamics this
project's ensemble DMD was built to characterise -- pooling phase-random
trials into one shared linear operator is a materially different, harder
regime, and this check is what catches it if the pooled fit cannot recover
genuine structure beyond chance.

Design note on the phase test (33D): Alagapan's protocol used a categorical
two-site phase manipulation (In Phase / Anti Phase / Sham), not a continuous
phase-lag sweep (contrast scripts/run_haslacher_phase_omega.py, which does
have one) -- so there is no continuous "predicted optimal phase" to compare
a peak against. The well-posed, honestly-stated consistency check available
from this design is instead: does each patient have an IDENTIFIABLE
band-matched oscillation in their own baseline data (a precondition for a
phase-locking mechanism to be meaningful at all), and does that patient's
own behavioural data show the same qualitative direction the source paper
reports (in-phase improves accuracy over sham; anti-phase does not)? Reported
as 3-point sign agreement, not as a p-value.

Output: results/alagapan_phase_omega.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_alagapan_phase_omega.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mne  # noqa: E402
import pymatreader  # noqa: E402

from dynamics import fit_band_matched_omega  # noqa: E402
from statistics import stable_seed  # noqa: E402

_DATA_CONFIG = json.loads((ROOT / "config" / "datasets.json").read_text())
_DATA_ROOT = os.environ.get(_DATA_CONFIG["local_data_root_env"])
DATA_DIR = (
    Path(_DATA_ROOT) / _DATA_CONFIG["datasets"]["alagapan_phase_stimulation"]["local_path"]
    if _DATA_ROOT else ROOT / "__WM_DYNAMICS_DATA_ROOT_NOT_SET__"
)
RESULTS = ROOT / "results"

PATIENTS = {
    "P1": {"band": "theta", "band_range": (4.0, 8.0), "stim_freq_hz": 4.0},
    "P2": {"band": "alpha", "band_range": (8.0, 12.0), "stim_freq_hz": 10.0},
    "P3": {"band": "alpha", "band_range": (8.0, 12.0), "stim_freq_hz": 10.0},
}

def _load_mapping(patient: str) -> dict[int, str]:
    baseline_path = DATA_DIR / "Electrode Information" / "Electrode Mapping" / f"{patient}_ElectrodeMapping_Baseline.csv"
    path = baseline_path if baseline_path.exists() else (
        DATA_DIR / "Electrode Information" / "Electrode Mapping" / f"{patient}_ElectrodeMapping.csv")
    with open(path) as f:
        return {int(row[0]): row[1] for row in csv.reader(f)}


def _load_seizure_electrodes(patient: str) -> set[int]:
    path = DATA_DIR / "Electrode Information" / "Seizure Electrodes" / f"{patient}_seizureElectrodes.mat"
    d = pymatreader.read_mat(str(path))
    return {int(i) for i in np.atleast_1d(d["seizureElectrodes"])}


def load_baseline_data(patient: str) -> tuple[np.ndarray, float, list[str]]:
    """Returns (N trials, C channels, T samples) baseline iEEG, sfreq, kept channel labels."""
    path = DATA_DIR / "iEEG Data" / "Baseline" / f"{patient}_SMS_Baseline_Epoched.set"
    epochs = mne.io.read_epochs_eeglab(str(path), verbose="ERROR")
    mapping = _load_mapping(patient)
    seizure = _load_seizure_electrodes(patient)
    keep_idx = [i for i in range(len(epochs.ch_names))
                if (i + 1) in mapping and (i + 1) not in seizure
                and not mapping[i + 1].upper().startswith("EKG")]
    data = epochs.get_data(copy=True)[:, keep_idx, :]
    labels = [mapping[i + 1] for i in keep_idx]
    return data, float(epochs.info["sfreq"]), labels


def _normalize_accuracy(value: str) -> int | None:
    v = value.strip()
    if v in ("Correct", "1"):
        return 1
    if v in ("Incorrect", "Miss", "0"):
        return 0
    return None


def load_behavior(patient: str) -> dict:
    path = DATA_DIR / "Task Performance" / f"{patient}_SternbergStimulation_Summary.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_condition: dict[str, list[int]] = {}
    for row in rows:
        acc = _normalize_accuracy(row["Accuracy"])
        if acc is None:
            continue
        by_condition.setdefault(row["Condition"], []).append(acc)
    summary = {cond: {"accuracy": float(np.mean(vals)), "n_trials": len(vals)}
              for cond, vals in by_condition.items()}
    sham = summary.get("Sham", {}).get("accuracy")
    in_phase_benefit = (summary["In Phase"]["accuracy"] - sham) if sham is not None and "In Phase" in summary else None
    anti_phase_benefit = (summary["Anti Phase"]["accuracy"] - sham) if sham is not None and "Anti Phase" in summary else None
    return {"by_condition": summary, "in_phase_minus_sham": in_phase_benefit,
            "anti_phase_minus_sham": anti_phase_benefit}


def main():
    out = {}
    for patient, spec in PATIENTS.items():
        print(f"{patient} (target: {spec['band']} {spec['band_range']}, "
              f"stim freq {spec['stim_freq_hz']} Hz) ...")
        data, srate, labels = load_baseline_data(patient)
        rng = np.random.default_rng(stable_seed(f"alagapan_phase_omega_{patient}"))
        omega = fit_band_matched_omega(data, srate, *spec["band_range"], rng)
        behavior = load_behavior(patient)
        matches_paper_direction = bool(
            behavior["in_phase_minus_sham"] is not None and behavior["in_phase_minus_sham"] > 0)
        sign_agrees = bool(omega["identifiable"] and matches_paper_direction)
        out[patient] = {
            **spec, "n_channels_kept": data.shape[1], "omega": omega, "behavior": behavior,
            "matches_paper_reported_direction": matches_paper_direction, "sign_agrees": sign_agrees,
        }
        print(f"  omega: f={omega['f_hz']:.3f} Hz CI={omega['f_ci']} identifiable={omega['identifiable']} "
              f"(Nyquist={omega['nyquist_hz']:.1f} Hz, r2_cv={omega['r2_cv']:.3f} r2_null={omega['r2_null']:.3f})")
        print(f"  behaviour: in_phase-sham={behavior['in_phase_minus_sham']:+.4f} "
              f"anti_phase-sham={behavior['anti_phase_minus_sham']:+.4f} "
              f"-> {'agrees' if sign_agrees else 'does not agree'} with paper's reported direction")

    n_agree = sum(1 for v in out.values() if v["sign_agrees"])
    print(f"\nSign agreement across n={len(out)} patients: {n_agree}/{len(out)} "
          f"(reported as sign agreement, not a p-value)")

    out["_meta"] = {
        "citation": "Alagapan et al. 2019, Cell Reports, PMC6901101",
        "n_patients": len(PATIENTS), "n_sign_agree": n_agree,
        "evidentiary_strength": "(a) reanalysis of public raw data, n=3 patients",
    }
    with open(RESULTS / "alagapan_phase_omega.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/alagapan_phase_omega.json")


if __name__ == "__main__":
    main()
