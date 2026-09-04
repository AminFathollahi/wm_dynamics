#!/usr/bin/env python3
"""Haslacher et al. (2024) CLAM-tACS reanalysis: does each participant's own
baseline-derived alpha oscillation accompany a reliable phase-dependent
working-memory benefit?

Citation: Haslacher D, Cavallo A, Reber P, Kattein A, Thiele M, Nasr K,
Hashemi K, Sokoliuk R, Thut G, Soekadar SR. "Working memory enhancement
using real-time phase-tuned transcranial alternating current stimulation."
Brain Stimulation 17 (2024) 850-859. https://doi.org/10.1016/j.brs.2024.07.007

Design, per the dataset's own Data/README.md: BrainVision EEG (`no_stim`
= baseline, stim OFF; `stim` = task block, stim ON), 64 EEG channels +
`envelope`/`stim` aux, 2 kHz, 64x10-20 montage. Six phase-condition trigger
codes 1-6 map to a target phase lag {3:30, 4:90, 5:150, 6:210, 1:270, 2:330}
degrees; codes 10/11 mark correct/incorrect responses. Participants split
into an ACTIVE group (n=21, occipital stimulation, over the recorded alpha
source) and a CONTROL group (n=25, frontal stimulation, away from the
source) -- the paper's reported phase-dependent effect is in the active
group; the control group is this project's specificity check.

CLAM-tACS targets endogenous alpha (8-14 Hz, the dataset's own band
definition). Band-matched omega is fit from each participant's own
`no_stim` (stimulation-OFF, hence artifact-free -- no SASS needed) baseline
recording via bandpass -> downsample -> PCA -> ensemble-DMD
(dynamics.fit_band_matched_omega), with an identifiability check (bootstrap
CI excludes 0, no Nyquist aliasing, held-out R^2 exceeds the circular-shift
null by a material margin).

This dataset has a continuous phase-lag sweep (6 evenly-spaced lags
spanning 360 degrees), but a bare oscillation frequency does not by itself
predict which phase lag is behaviourally optimal -- that requires an
unknown stimulation-to-oscillator coupling delay the literature does not
give in closed form. The well-posed check run here: (i) is a genuine
baseline alpha oscillation identifiable per participant, and (ii) is that
participant's own behavioural accuracy-vs-phase curve non-trivially
modulated (permutation test on modulation depth, following the dataset's
own accuracy-by-phase method)? Reported per participant, with a group
summary for active vs. control.

Behavioural accuracy needs no EEG cleaning and is read directly from event
markers with `preload=False`.

Output: results/haslacher_phase_omega.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_haslacher_phase_omega.py
"""
from __future__ import annotations

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

from dynamics import fit_band_matched_omega  # noqa: E402
from statistics import permutation_pvalue, stable_seed  # noqa: E402

_DATA_CONFIG = json.loads((ROOT / "config" / "datasets.json").read_text())
_DATA_ROOT = os.environ.get(_DATA_CONFIG["local_data_root_env"])
DATA_DIR = (
    Path(_DATA_ROOT) / _DATA_CONFIG["datasets"]["haslacher_clam_tacs"]["local_path"]
    if _DATA_ROOT else ROOT / "__WM_DYNAMICS_DATA_ROOT_NOT_SET__"
)
RESULTS = ROOT / "results"

ACTIVE_SUBJECTS = ([f"PA{i}" for i in range(1, 17)] + ["PA18", "PA19", "PA20", "PA22", "PA23"])
CONTROL_SUBJECTS = (["PA17", "PA21"] + [f"PA{i}" for i in range(24, 47)])
ALPHA_BAND = (8.0, 14.0)  # dataset's own definition (Data/README.md sec 4)
RETENTION_TMIN, RETENTION_TMAX = 0.6, 3.6  # dataset's own retention window (sec 3/7)
PHASE_CONDITIONS = {3: 30, 4: 90, 5: 150, 6: 210, 1: 270, 2: 330}  # dataset's own code->degrees map
N_PERM_MODULATION = 2000


def load_baseline_epochs(subject: str) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "no_stim.vhdr"),
                                       preload=True, verbose="ERROR")
    raw.drop_channels(["envelope", "stim"])
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    epochs = mne.Epochs(raw, events, event_id=list(PHASE_CONDITIONS), tmin=RETENTION_TMIN,
                        tmax=RETENTION_TMAX, baseline=None, preload=True,
                        on_missing="ignore", verbose="ERROR")
    return epochs.get_data(copy=True), float(raw.info["sfreq"])


def trial_outcomes(subject: str) -> list[tuple[int, int]]:
    """(phase code, correct) per trial, from event markers only -- mirrors the
    dataset README's own accuracy_by_condition (sec 8), reused directly."""
    raw = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "stim.vhdr"),
                                       preload=False, verbose="ERROR")
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    outcomes = []
    for i in range(len(events) - 1):
        code = events[i, 2]
        if code not in PHASE_CONDITIONS:
            continue
        rt = (events[i + 1, 0] - events[i, 0]) / sfreq - 3.8
        if rt < 0.75:
            outcomes.append((code, int(events[i + 1, 2] == 10)))
    return outcomes


def _modulation(values_in_phase_order: np.ndarray) -> tuple[float, float]:
    """Single-cycle DFT modulation depth and optimal phase (rad) -- the
    dataset README's own method (sec 10), reused directly with attribution."""
    x = np.asarray(values_in_phase_order, float)
    n = len(x)
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c = (x * np.exp(-1j * phases)).sum() * 2 / n
    wrapped = (np.angle(c) + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(c)), float(wrapped)


def modulation_from_outcomes(outcomes: list[tuple[int, int]], rng: np.random.Generator) -> dict:
    codes_ordered = sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c])
    by_code = {c: [] for c in codes_ordered}
    for code, correct in outcomes:
        by_code[code].append(correct)
    accuracy = np.array([np.mean(by_code[c]) if by_code[c] else np.nan for c in codes_ordered])
    if np.any(np.isnan(accuracy)):
        return {"depth": None, "optimal_phase_deg": None, "p_value": None,
                "n_trials": len(outcomes), "reason": "at least one phase condition has zero trials"}

    depth_obs, phase_obs = _modulation(accuracy)
    codes_arr = np.array([c for c, _ in outcomes])
    correct_arr = np.array([v for _, v in outcomes])
    null = np.empty(N_PERM_MODULATION)
    for p in range(N_PERM_MODULATION):
        shuffled_codes = rng.permutation(codes_arr)
        acc_p = np.array([correct_arr[shuffled_codes == c].mean() if np.any(shuffled_codes == c)
                          else np.nan for c in codes_ordered])
        d_p, _ = _modulation(acc_p) if not np.any(np.isnan(acc_p)) else (np.nan, 0.0)
        null[p] = d_p
    valid = null[~np.isnan(null)]
    p_value = permutation_pvalue(valid >= depth_obs) if len(valid) else float("nan")
    return {"depth": depth_obs, "optimal_phase_deg": float(np.degrees(phase_obs)),
            "p_value": float(p_value), "n_trials": len(outcomes),
            "accuracy_by_phase_deg": {PHASE_CONDITIONS[c]: float(accuracy[i])
                                      for i, c in enumerate(codes_ordered)}}


def run_group(subjects: list[str], group_name: str) -> dict:
    out = {}
    for subject in subjects:
        try:
            data, srate = load_baseline_epochs(subject)
            outcomes = trial_outcomes(subject)
        except (FileNotFoundError, OSError) as e:
            out[subject] = {"status": "error", "reason": str(e)}
            continue
        rng = np.random.default_rng(stable_seed(f"haslacher_phase_omega_{subject}"))
        omega = fit_band_matched_omega(data, srate, *ALPHA_BAND, rng)
        mod = modulation_from_outcomes(outcomes, rng)
        consistent = bool(omega["identifiable"] and mod.get("p_value") is not None
                          and mod["p_value"] < 0.05)
        out[subject] = {"n_baseline_trials": int(data.shape[0]), "omega": omega,
                        "modulation": mod, "consistent": consistent}
        print(f"  {subject} ({group_name}): f={omega['f_hz']:.2f} Hz identifiable={omega['identifiable']} "
              f"| modulation depth={mod.get('depth')} p={mod.get('p_value')} "
              f"opt_phase={mod.get('optimal_phase_deg')} -> "
              f"{'CONSISTENT' if consistent else 'not consistent / not identifiable'}")
    n_identifiable = sum(1 for v in out.values() if v.get("omega", {}).get("identifiable"))
    n_modulated = sum(1 for v in out.values() if v.get("modulation", {}).get("p_value") is not None
                      and v["modulation"]["p_value"] < 0.05)
    n_consistent = sum(1 for v in out.values() if v.get("consistent"))
    return {"per_subject": out, "n_subjects": len(subjects), "n_identifiable_omega": n_identifiable,
            "n_significant_modulation": n_modulated, "n_consistent": n_consistent}


def main():
    print(f"Active group (n={len(ACTIVE_SUBJECTS)}) ...")
    active = run_group(ACTIVE_SUBJECTS, "active")
    print(f"\nControl group (n={len(CONTROL_SUBJECTS)}) ...")
    control = run_group(CONTROL_SUBJECTS, "control")

    print(f"\nActive: {active['n_identifiable_omega']}/{active['n_subjects']} identifiable omega, "
          f"{active['n_significant_modulation']}/{active['n_subjects']} significant modulation, "
          f"{active['n_consistent']}/{active['n_subjects']} both (consistent)")
    print(f"Control: {control['n_identifiable_omega']}/{control['n_subjects']} identifiable omega, "
          f"{control['n_significant_modulation']}/{control['n_subjects']} significant modulation, "
          f"{control['n_consistent']}/{control['n_subjects']} both (consistent)")

    out = {"active": active, "control": control,
          "_meta": {"citation": "Haslacher et al. 2024, Brain Stimulation, "
                    "doi:10.1016/j.brs.2024.07.007",
                    "alpha_band_hz": list(ALPHA_BAND), "retention_window_s": [RETENTION_TMIN, RETENTION_TMAX]}}
    with open(RESULTS / "haslacher_phase_omega.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/haslacher_phase_omega.json")


if __name__ == "__main__":
    main()
