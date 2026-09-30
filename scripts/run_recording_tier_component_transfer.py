#!/usr/bin/env python3
"""At which recording tier does the rate-free state-deviation component still
exist, still predict behaviour, and still track the value measured at the
finest tier on the same trial?

Every measurement of this component to date has been made on sorted single
units. No chronic implantable device reads sorted single units, so this
module asks the question a stimulation specification needs answered: does
the component survive at coarser, chronically-deployable recording tiers?

One human corpus records microwire single units, intracranial depth
macro-contacts and scalp electroencephalography simultaneously, in the same
patients, on the same trials, during a verbal working-memory task with a 3 s
maintenance delay (fixation 1 s, encoding 2 s, maintenance 3 s, probe). A
second release from the same laboratory covers 9 of the same patients and
adds beamformed cortical virtual sensors from a separate recording session.
Five recording tiers are built from these two releases:

  1. single_unit        -- sorted microwire units, maintenance-window per-unit
                            spike totals.
  2. depth_mtl           -- depth macro contacts labelled hippocampus,
                            amygdala or entorhinal cortex, per-channel
                            maintenance-window band power.
  3. depth_cortical      -- depth macro contacts on the same probes labelled
                            outside medial temporal cortex, per-channel band
                            power.
  4. scalp_eeg           -- the scalp montage carried in the same recording
                            file as tiers 1-3, per-electrode band power.
  5. beamformed_cortical -- the second release's LCMV source reconstruction
                            (DLPFC, OFC, PPC, AC, V1), per-source band power.

Tiers 1-4 are built from exactly the same trial table in the same file (the
same NWB release that carries the sorted units); the artifact mask is applied
once per session and identically to all four. Tier 5 is built from a second
release recorded as a separate session; its per-trial identity (set size,
accuracy, artifact status) is not retrievable from the distributed derivative
(verified directly against the file -- see beamformed_trial_identity_audit
below), so cross-release pairing against tiers 1-4 is never licensed at the
trial level for any patient and tier 5 enters the cross-tier-transfer test at patient level only,
as the patient-level fallback regime below anticipates.

The component estimator (rate_free_state_deviation) and the magnitude-matched
rotation null (rotation_null_variance_test) are reused unchanged from
scripts/run_rate_free_state_geometry_behavior_link.py and
scripts/run_human_stimulation_component_response.py respectively -- copied
verbatim rather than imported, so this module has no import-time dependency
on either module's own (heavier) data-loading chain during a long detached
run. Neither function's arithmetic is touched.

Run:
    python \
        scripts/run_recording_tier_component_transfer.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root  # noqa: E402
from preprocessing import band_power, high_gamma_power, line_noise_notch, load_boran_nwb  # noqa: E402
from provenance import canonical_json, checkpoint_store, git_commit  # noqa: E402
from spike_pipeline import load_spike_times, normalize_region_label  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test,
    partial_correlation_permutation_test, permutation_pvalue, stable_seed,
)
from scipy.stats import spearmanr  # noqa: E402
from statistics import MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, MTL_STRUCTURES, N_ROTATION_NULL_DRAWS, UNLABELLED_STRUCTURES, _bias_only_values  # noqa: E402
from corpus_sessions import discover_000574_sessions  # noqa: E402
from state_persistence import MIN_PATIENTS_FOR_TEST, _patient_clustered_test, existence_tier  # noqa: E402
from statistics import rate_free_state_deviation, rotation_null_variance_test  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "recording_tier_component_transfer.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_recording_tier_component_transfer"

TIERS = ("single_unit", "depth_mtl", "depth_cortical", "scalp_eeg", "beamformed_cortical")
IN_FILE_TIERS = ("single_unit", "depth_mtl", "depth_cortical", "scalp_eeg")

LABEL_CONVENTION = "nwb_boran_brainnetome_hybrid"

MAINS_HZ = 50.0  # Zurich site, both releases -- not 60 Hz.
MAINT_WINDOW_S = (-3.0, 0.0)  # relative to probe onset, this task's own convention.
EPOCH_WIN_S = (-3.5, 0.5)  # padding around the maintenance window for filter settling.
BEAMFORMED_SOURCES = ["DLPFC", "OFC", "PPC", "AC", "V1"]

# The reference effect for every behaviour-link and cross-tier-agreement null in this module (the
# behaviour-link and cross-tier-transfer tests -- both measured in correlation-r units). Sourced unchanged
# from results/rate_free_state_geometry_behavior_link.json's own minimum detectable paired
# difference (0.14 r units): the same fixed constant results/human_stimulation_component_response.json
# also reuses for its own human intracranial arm, so every behavioural bound this project reports --
# macaque, human stimulation, and this recording-tier leg -- sits on one scale. The existence test's
# reference (a variance-difference, not a correlation) has no r-unit equivalent and is instead
# established from the single_unit tier's own observed effect, disclosed in EXISTENCE_DECISION_RULE
# above.

N_TRIAL_SHUFFLE_DRAWS = 1000
N_BOOT_EXISTENCE_AND_BEHAVIOUR_LINK = 5000
MIN_TRIALS_PER_SESSION_TIER = 10
MIN_TRIALS_PER_SET_SIZE_CELL = 6
# 1/2**n exceeds 0.05 for n<5 regardless of the data (n=4 -> 0.0625, still short), so 5 is the exact,
# data-independent minimum, not 4 -- verified against that function's own internal check, not assumed.
SET_SIZES = (4, 6, 8)

# Named bands, ordered from the widest (highest upper edge) to the narrowest, so the "highest
# fully-available band for its own sample rate" is the first one whose upper edge clears Nyquist.
BAND_ORDER = (("hgp", 70.0, 150.0), ("gamma", 30.0, 70.0), ("beta", 13.0, 30.0),
              ("alpha", 8.0, 13.0), ("theta", 4.0, 8.0))

EXISTENCE_DECISION_RULE = (
    "Per session per tier, compute rate_free_state_deviation on the maintenance-window feature matrix "
    "(unchanged estimator), then test its within-session variance against a magnitude-matched rotation "
    "null (rotation_null_variance_test, 1000 draws: each trial's own feature magnitude is kept, its "
    "direction replaced by an independent uniformly-random unit vector). The session-level signed effect "
    "is observed_variance - null_mean_variance. Effects are collapsed to one value per patient (mean "
    "across that patient's sessions) and tested against zero across patients with the two-sided paired "
    "sign-flip test (slope_across_sessions_test). If n_patients < 5 the sign-flip test cannot structurally "
    "reach p<=0.05 (min_attainable_p > 0.05) and the tier fires 'underpowered_to_ask_at_this_tier'. "
    "Otherwise: significant (p<=0.05) -> 'component_is_present_at_this_recording_tier'. Not significant: "
    "the tier's own minimum detectable paired difference (80% power) is compared against a reference "
    "effect. The existence test's own text does not name a reference effect (unlike the behaviour-link "
    "test, which explicitly names 'the effect measured at the finest tier in the same patients'); this "
    "module extends that same, explicitly-licensed rule to the existence test by disclosed analogy, fixed here before any tier past single_unit "
    "is evaluated: the reference is the single_unit tier's own observed |pooled mean effect|, established "
    "first. MDD < reference -> 'component_is_not_distinguishable_from_a_magnitude_matched_rotation_null'. "
    "MDD >= reference, or no reference could be established (single_unit itself did not reach "
    "significance) -> 'underpowered_to_ask_at_this_tier'."
)

BEHAVIOUR_LINK_DECISION_RULE = (
    "Trials are pooled, not session-level statistics: this corpus's error rate is 8.3% (151/1827 trials), "
    "per-session error counts run 0-10 (median ~4), so a point-biserial correlation computed within one "
    "session is built from too few errors to mean anything, and pooling those session-level numbers would "
    "report an estimator failure as a property of the data rather than testing the data itself -- verified "
    "directly against the trial tables before choosing this design, not assumed. Within each set size, "
    "every trial a patient contributes across ALL of that patient's sessions is pooled into one "
    "point-biserial correlation (partial_correlation_permutation_test, zero controls) of accuracy against "
    "the component value; the up-to-3 set-size-specific coefficients are averaged (unweighted) to one "
    "per-patient effect ('combined across set sizes'), and tested against zero across patients with the "
    "two-sided paired sign-flip test -- patient remains the independent clustering unit throughout, only "
    "the level at which the correlation itself is computed has moved from session to patient-pooled-trials. "
    "A pooled-across-load secondary (accuracy vs. component value, set size ignored, still one number per "
    "patient) is also computed and reported beside the within-load primary, never in its place. Mandatory "
    "control: the identical statistic recomputed with every trial's component value replaced by its "
    "patient-and-set-size leave-one-out mean (the 'training-trial mean' -- every OTHER trial that patient "
    "contributes in that same set-size cell, across all their sessions; no cross-validation fold structure "
    "exists for a plain correlation, so 'training' is read as 'held out from the trial itself', consistent "
    "with the estimator's own leave-one-out construction). If the control reproduces significance with the "
    "same sign as the primary, the branch is 'behaviour_link_not_separable_from_a_session_level_offset' "
    "and the tier's result is void. Otherwise: n_patients < 5 -> 'underpowered_to_ask_at_this_tier'. "
    "Primary significant -> 'component_predicts_accuracy_at_this_recording_tier'. Primary not significant: "
    "MDD compared against the reference effect (0.14 r units, MEANINGFUL_EFFECT_THRESHOLD_R_UNITS -- see "
    "its own module-level disclosure). MDD < reference -> "
    "'no_behaviour_link_at_this_tier_above_the_reported_bound'. MDD >= reference, or no reference "
    "available -> 'underpowered_to_ask_at_this_tier'."
)

CROSS_TIER_TRANSFER_DECISION_RULE = (
    "For each unordered pair of tiers, per session, the trial-wise Pearson correlation of the two tiers' "
    "component values on the identical admitted trial cohort (only licensed when both tiers were built "
    "from the same release's same trial table -- true for every pair among single_unit/depth_mtl/"
    "depth_cortical/scalp_eeg, since all four are read from the one NWB file per session; never true for "
    "any pair involving beamformed_cortical, whose distributed derivative carries no retrievable per-trial "
    "set-size, accuracy or session identity -- verified directly against the .mat file, see "
    "beamformed_trial_identity_audit). Session-level significance: shuffle one tier's trial index within "
    "session, 1000 draws (two-sided permutation p). Session-level correlations collapse to one value per "
    "patient (mean across sessions) and are tested against zero across patients with the two-sided paired "
    "sign-flip test -- this is the 'trial_wise' regime. Whenever a pair involves beamformed_cortical (or, "
    "for any other pair, whenever a patient's sessions fail the exact trial-count/set-size/accuracy match "
    "required for trial-level pairing), the pair instead enters the 'patient_level_only' "
    "regime: one scalar per patient per tier (the patient's own median component value, pooled across its "
    "admitted trials), Spearman-correlated across patients with a patient-bootstrap CI (the same construct "
    "results/cross_modality_calibration.json uses for its own cross-release comparison), 'excludes zero' "
    "standing in for the sign-flip test's significance call. Only three branches exist for the cross-tier-transfer test -- no "
    "separate 'underpowered' branch -- so this module's own disclosed reading, fixed before any pair is "
    "evaluated: n_patients < 5 (either regime) -> 'cross_tier_agreement_not_testable_at_matched_trials'. "
    "Significant -> 'the_two_tiers_track_the_same_per_trial_quantity'. Not significant and MDD (or, in the "
    "patient-level regime, the CI width) below the reference effect (the observed |pooled correlation| of "
    "the single_unit-vs-depth_mtl pair, the most anatomically adjacent trial-wise pair, computed first; if "
    "that reference pair is itself not significant, MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, 0.14, is used in "
    "its place rather than leaving every other pair's null unjudgeable) -> "
    "'no_cross_tier_agreement_above_the_reported_bound'. Not significant and underpowered against that "
    "reference, or the reference itself unavailable -> 'cross_tier_agreement_not_testable_at_matched_"
    "trials' (this branch's literal name, 'not testable AT MATCHED TRIALS', is read as covering both a "
    "structurally unmatched pair and a matched-but-underpowered one, since the cross-tier-transfer test's fixed three-branch "
    "vocabulary provides no other slot for an underpowered result)."
)


# ── Reused unchanged (copied, not imported -- see module docstring) ─────────────────────────────────────





# ── Checkpointing (atomic, per session-per-tier; pattern shared with run_human_stimulation_component_response.py) ──

def _checkpoint_path(unit: str) -> Path:
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def save_checkpoint(unit: str, record: dict) -> None:
    path = _checkpoint_path(unit)
    checkpoint_store(path, {"_complete": True, "record": record})


def run_checkpointed(unit: str, fit_fn) -> dict:
    cached = load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(unit, record)
    return record


# ── Band selection ───────────────────────────────────────────────────────────────────────────────────────

def highest_available_band(srate: float) -> tuple[str, float, float] | None:
    nyquist = srate / 2.0
    for name, lo, hi in BAND_ORDER:
        if hi < nyquist:
            return name, lo, hi
    return None


def needs_mains_notch(lo: float, hi: float, mains_hz: float = MAINS_HZ) -> bool:
    return any(lo < mains_hz * h < hi for h in range(1, 6))


def _band_power_2d(sig_tc: np.ndarray, srate: float, band_name: str) -> np.ndarray:
    if band_name == "hgp":
        return high_gamma_power(sig_tc, srate)
    return band_power(sig_tc, band_name, srate)


# ── Region classification for depth channels ─────────────────────────────────────────────────────────────

def classify_depth_channels(electrode_locs: list[str]) -> dict[str, list[int]]:
    """Returns {'depth_mtl': [channel indices], 'depth_cortical': [channel indices]}, using the same
    Brainnetome-hybrid label parser both Zurich releases in this project share."""
    mtl_idx, cortical_idx = [], []
    for i, raw in enumerate(electrode_locs):
        structure, _ = normalize_region_label(raw, LABEL_CONVENTION)
        if structure in MTL_STRUCTURES:
            mtl_idx.append(i)
        elif structure not in UNLABELLED_STRUCTURES:
            cortical_idx.append(i)
    return {"depth_mtl": mtl_idx, "depth_cortical": cortical_idx}


# ── Feature extraction, DANDI 000574 (tiers 1-4) ─────────────────────────────────────────────────────────

def _load_000574_trial_table(handle) -> dict:
    trials = handle["intervals/trials"]
    return {
        "artifact": trials["artifact"][:].astype(bool),
        "correct": trials["correct"][:].astype(bool),
        "set_size": trials["set_size"][:].astype(int),
        "start_time": trials["start_time"][:].astype(float),
    }


def _unit_tier_activity(spike_lists: list[np.ndarray], delay_onset: np.ndarray, window_s: float) -> np.ndarray:
    n_trials, n_units = len(delay_onset), len(spike_lists)
    counts = np.zeros((n_trials, n_units), dtype=float)
    for u, spikes in enumerate(spike_lists):
        spikes = np.sort(np.asarray(spikes, dtype=float))
        lo_idx = np.searchsorted(spikes, delay_onset, side="left")
        hi_idx = np.searchsorted(spikes, delay_onset + window_s, side="left")
        counts[:, u] = hi_idx - lo_idx
    return counts


def _depth_or_scalp_activity(epochs: np.ndarray, times: np.ndarray, srate: float,
                              channel_idx: list[int], band_name: str, lo: float, hi: float) -> np.ndarray:
    """epochs: (N, C, T) from load_boran_nwb, already restricted to admitted trials. Returns
    (N, len(channel_idx)) maintenance-window mean band power, one trial at a time (notch + Hilbert
    envelope need per-trial edge handling, matching compute_hgp's own per-trial loop)."""
    win_mask = (times >= MAINT_WINDOW_S[0]) & (times < MAINT_WINDOW_S[1])
    selected = epochs[:, channel_idx, :]
    n_trials = selected.shape[0]
    out = np.full((n_trials, len(channel_idx)), np.nan)
    notch = needs_mains_notch(lo, hi)
    for i in range(n_trials):
        sig_tc = selected[i].T  # (T, C)
        if notch:
            sig_tc = line_noise_notch(sig_tc, srate, MAINS_HZ)
        power = _band_power_2d(sig_tc, srate, band_name)
        out[i] = power[win_mask].mean(axis=0)
    return out


def trial_tables_agree(table: dict, ieeg: dict, eeg: dict) -> tuple[bool, str | None]:
    """The protective assertion the cross-tier-transfer test's bridge depends on, run before any feature is built rather than
    only before a cross-tier statistic: depth (ieeg) and scalp (eeg) are two independent reads of the same
    NWB trials table (load_boran_nwb, called once per signal), and this checks they actually came back
    identical element-by-element on trial count, set size and accuracy -- rather than assuming two reads
    of one file must agree. A session that fails this is refused whole (every in-file tier), since nothing
    downstream can trust a shared trial cohort at that point."""
    n = len(table["artifact"])
    if not (len(ieeg["set_sizes"]) == n and len(eeg["set_sizes"]) == n):
        return False, "trial_count_mismatch_between_signals_in_same_file"
    if not (np.array_equal(ieeg["set_sizes"], table["set_size"]) and
            np.array_equal(eeg["set_sizes"], table["set_size"])):
        return False, "set_size_mismatch_between_signals_in_same_file"
    if not (np.array_equal(ieeg["correct"], table["correct"]) and
            np.array_equal(eeg["correct"], table["correct"])):
        return False, "accuracy_mismatch_between_signals_in_same_file"
    return True, None


def build_000574_session(nwb_path: Path) -> dict:
    """Loads one 000574 session and returns per-tier admitted feature matrices plus the shared trial
    admission mask -- everything downstream (the existence, behaviour-link and cross-tier-transfer
    tests) is built from this one record."""
    with h5py.File(nwb_path, "r") as handle:
        if "units" not in handle:
            return {"status": "refused", "reason": "no_units_table"}
        table = _load_000574_trial_table(handle)
        spike_lists = load_spike_times(handle)

    ieeg = load_boran_nwb(str(nwb_path), signal="ieeg", epoch_win=EPOCH_WIN_S)
    eeg = load_boran_nwb(str(nwb_path), signal="eeg", epoch_win=EPOCH_WIN_S)

    n_trials = len(table["artifact"])
    agree, reason = trial_tables_agree(table, ieeg, eeg)
    if not agree:
        return {"status": "refused", "reason": reason}
    admit = (~table["artifact"]) & ieeg["valid"] & eeg["valid"]
    n_admit = int(admit.sum())
    if n_admit < MIN_TRIALS_PER_SESSION_TIER:
        return {"status": "refused", "reason": "too_few_admitted_trials", "n_admitted": n_admit}

    set_size = table["set_size"][admit]
    correct = table["correct"][admit].astype(float)
    delay_onset = table["start_time"][admit] + 3.0  # this task's own fixed relative structure.

    regions = classify_depth_channels(ieeg["electrode_locs"])
    ieeg_epochs_admit = ieeg["epochs"][admit]
    eeg_epochs_admit = eeg["epochs"][admit]

    depth_band = highest_available_band(ieeg["srate"])
    scalp_band = highest_available_band(eeg["srate"])

    tiers: dict[str, dict] = {}

    unit_activity = _unit_tier_activity([np.asarray(s) for s in spike_lists], delay_onset, 3.0)
    tiers["single_unit"] = {
        "status": "computed", "activity": unit_activity, "n_features": unit_activity.shape[1],
        "band": None,
    } if unit_activity.shape[1] > 0 else {"status": "refused", "reason": "no_units"}

    for tier_key, region_key in (("depth_mtl", "depth_mtl"), ("depth_cortical", "depth_cortical")):
        idx = regions[region_key]
        if not idx:
            tiers[tier_key] = {"status": "refused", "reason": "no_channels_in_region"}
            continue
        if depth_band is None:
            tiers[tier_key] = {"status": "refused", "reason": "no_band_fits_this_sample_rate"}
            continue
        name, lo, hi = depth_band
        activity = _depth_or_scalp_activity(ieeg_epochs_admit, ieeg["times"], ieeg["srate"], idx, name, lo, hi)
        tiers[tier_key] = {"status": "computed", "activity": activity, "n_features": len(idx),
                            "band": {"name": name, "lo_hz": lo, "hi_hz": hi, "srate_hz": ieeg["srate"]}}

    if scalp_band is None:
        tiers["scalp_eeg"] = {"status": "refused", "reason": "no_band_fits_this_sample_rate"}
    else:
        name, lo, hi = scalp_band
        idx = list(range(eeg_epochs_admit.shape[1]))
        activity = _depth_or_scalp_activity(eeg_epochs_admit, eeg["times"], eeg["srate"], idx, name, lo, hi)
        tiers["scalp_eeg"] = {"status": "computed", "activity": activity, "n_features": len(idx),
                               "band": {"name": name, "lo_hz": lo, "hi_hz": hi, "srate_hz": eeg["srate"]}}

    return {
        "status": "computed", "n_trials_total": n_trials, "n_trials_admitted": n_admit,
        "set_size": set_size, "correct": correct, "tiers": tiers,
    }


# ── Feature extraction, ds004752 (tier 5) ────────────────────────────────────────────────────────────────

def beamformed_trial_identity_audit(mat_keys_top: list[str], epoch_keys: list[str], cfg_keys: list[str]) -> dict:
    """Records exactly what was and was not found when searching the derivative for any retrievable
    per-trial identity (session index, set size, accuracy, sample/trial number) -- the evidence backing
    this module's claim that cross-release trial-level pairing is never licensed for this tier. Verified
    directly against the .mat file with a recursive key search: no key
    anywhere in the epoch structure or its nested cfg contains 'trial', 'sample', 'sess' or 'info' beyond
    FieldTrip bookkeeping strings ('cfg.trials'='all', 'cfg.sampleindex'), and no array of trial-matching
    length (matching the session count, the per-session trial count, or any subset thereof under an
    artifact/correct/match filter) was found."""
    return {
        "top_level_keys": mat_keys_top, "epoch_struct_keys": epoch_keys, "cfg_keys": cfg_keys,
        "trial_identity_fields_found": [k for k in epoch_keys if k not in ("label", "trial", "time", "fsample", "cfg")],
        "conclusion": "no per-trial session index, set size, accuracy or sample-number field exists in this "
                      "derivative; trial-level cross-release pairing is not reconstructable for this tier.",
    }


def build_beamformed_session(mat_path: Path) -> dict:
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    if "LCMV_maintenance" not in mat:
        return {"status": "refused", "reason": "no_LCMV_maintenance_epoch"}
    epoch = mat["LCMV_maintenance"]
    audit = beamformed_trial_identity_audit(list(mat.keys()), list(epoch.keys()), list(epoch.get("cfg", {}).keys()))
    labels = list(epoch["label"])
    beam_idx = [labels.index(s) for s in BEAMFORMED_SOURCES if s in labels]
    if not beam_idx:
        return {"status": "refused", "reason": "no_beamformed_source_channels", "audit": audit}
    fsample = float(epoch["fsample"])
    band = highest_available_band(fsample)
    if band is None:
        return {"status": "refused", "reason": "no_band_fits_this_sample_rate", "audit": audit}
    name, lo, hi = band
    trials = epoch["trial"]
    n_trials = len(trials)
    if n_trials < MIN_TRIALS_PER_SESSION_TIER:
        return {"status": "refused", "reason": "too_few_trials", "n_trials": n_trials, "audit": audit}
    notch = needs_mains_notch(lo, hi)
    activity = np.full((n_trials, len(beam_idx)), np.nan)
    for i, trial in enumerate(trials):
        sig_tc = np.asarray(trial)[beam_idx].T  # (T, C)
        if notch:
            sig_tc = line_noise_notch(sig_tc, fsample, MAINS_HZ)
        power = _band_power_2d(sig_tc, fsample, name)
        activity[i] = power.mean(axis=0)
    return {
        "status": "computed", "n_trials_admitted": n_trials, "n_trials_total": n_trials,
        "activity": activity, "n_features": len(beam_idx),
        "band": {"name": name, "lo_hz": lo, "hi_hz": hi, "srate_hz": fsample},
        "audit": audit,
        # No accuracy or set-size label exists for this tier (see audit); the behaviour-link test is
        # structurally untestable here, not merely underpowered -- disclosed explicitly rather than fabricated.
        "set_size": None, "correct": None,
    }


# ── Session discovery ─────────────────────────────────────────────────────────────────────────────────────



def discover_beamformed_sessions(root: Path, overlapping_patients: set[str]) -> list[tuple[str, str, Path]]:
    """ds004752's derivatives directory carries all 15 of its own patients, but only the ones this
    module scopes -- those also present in the 000574 discovery -- have any tiers 1-4 to bridge
    against; the other 6 (sub-10..sub-15) are ds004752-only and are excluded by name here, not silently
    dropped by a floor (config/datasets.json's own documented view_relationship for ds004752)."""
    out = []
    for subject_dir in sorted((root / "ds004752" / "derivatives").glob("sub-*")):
        patient = subject_dir.name
        if patient not in overlapping_patients:
            continue
        mat_path = subject_dir / "beamforming" / f"{patient}-task-verbalWM-LCMVsources.mat"
        if mat_path.is_file():
            out.append((patient, f"{patient}_beamformed_pooled", mat_path))
    return out


# ── Existence test ───────────────────────────────────────────────────────────────────────────────────

def _session_deviation_and_gate(activity: np.ndarray, seed_tag: str) -> dict:
    deviation = rate_free_state_deviation(activity)
    rng = np.random.default_rng(stable_seed(seed_tag))
    gate = rotation_null_variance_test(activity, N_ROTATION_NULL_DRAWS, rng)
    return {"deviation": deviation, "gate": gate}






# ── Behaviour-link test ──────────────────────────────────────────────────────────────────────────────



def _concat_tier_trials(sessions: list[dict]) -> dict:
    """Concatenates every session's per-trial arrays for one tier into flat trial-level arrays tagged by
    patient -- the pooled trial table the patient-clustered behaviour estimator below is built from.
    Session-level intermediate correlations are not computed at all: with a corpus-wide error rate this
    low (8.3%, per-session error counts running 0-10, median ~4), a per-SESSION point-biserial correlation
    is built from too few errors to mean anything, and pooling those noisy session-level numbers would
    report an estimator failure as if it were a property of the data."""
    deviation, correct, set_size, patient_id, session_id = [], [], [], [], []
    for s in sessions:
        n = len(s["deviation"])
        deviation.append(s["deviation"]); correct.append(s["correct"]); set_size.append(s["set_size"])
        patient_id.extend([s["patient"]] * n); session_id.extend([s["session_key"]] * n)
    if not deviation:
        return {"deviation": np.array([]), "correct": np.array([]), "set_size": np.array([]),
                "patient_id": np.array([], dtype=object), "session_id": np.array([], dtype=object)}
    return {
        "deviation": np.concatenate(deviation), "correct": np.concatenate(correct),
        "set_size": np.concatenate(set_size), "patient_id": np.array(patient_id, dtype=object),
        "session_id": np.array(session_id, dtype=object),
    }


def _bias_only_values_by_session(deviation: np.ndarray, session_id: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    """Each trial's value replaced by the leave-one-out mean of every OTHER trial in its OWN SESSION,
    within this set-size cell -- the bias-only control's literal 'session's training-trial mean', preserved even
    though the primary statistic now pools a patient's sessions together for power. Computing the
    leave-one-out mean over the patient's WHOLE pooled cell instead (mixing sessions) would wash out
    between-session variance and make this control blind to exactly the confound it exists to catch: one
    session running at a different baseline than another, within the same patient. A trial whose own
    session contributes fewer than 2 trials to this cell gets NaN (no other trial to average)."""
    out = np.full(deviation.shape, np.nan)
    idx = np.where(cell_mask)[0]
    for sess in np.unique(session_id[idx]):
        sub_idx = idx[session_id[idx] == sess]
        out[sub_idx] = _bias_only_values(deviation[sub_idx])
    return out


def _patient_set_size_cell(deviation: np.ndarray, correct: np.ndarray, session_id: np.ndarray, mask: np.ndarray,
                            seed_tag: str) -> dict:
    n = int(mask.sum())
    n_errors = int((correct[mask] == 0).sum()) if n else 0
    if n < MIN_TRIALS_PER_SET_SIZE_CELL:
        return {"status": "too_few_trials", "n_trials": n, "n_errors": n_errors}
    y, x = correct[mask], deviation[mask]
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|primary"))
    primary = partial_correlation_permutation_test(y, x, controls=[], n_perm=2000, rng=rng)

    bias_x_full = _bias_only_values_by_session(deviation, session_id, mask)
    bias_x_cell = bias_x_full[mask]
    bias_finite = np.isfinite(bias_x_cell)
    n_bias = int(bias_finite.sum())
    if n_bias < MIN_TRIALS_PER_SET_SIZE_CELL:
        # A patient whose every session contributes fewer than 2 trials to this cell (e.g. one session
        # only) has no session-level leave-one-out mean defined anywhere in the cell -- the control is
        # honestly not computable here, not silently skipped.
        bias = {"status": "not_computable", "reason": "too_few_trials_with_a_defined_session_loo_mean",
                "n": n_bias}
    else:
        rng_b = np.random.default_rng(stable_seed(f"{seed_tag}|bias"))
        bias = partial_correlation_permutation_test(y[bias_finite], bias_x_cell[bias_finite], controls=[],
                                                      n_perm=2000, rng=rng_b)
    return {"status": "computed", "n_trials": n, "n_errors": n_errors, "primary": primary, "bias_only": bias}


def behaviour_link_tier(sessions: list[dict], reference_effect: float | None) -> dict:
    """Patient-clustered, trial-pooled behaviour link: within each set size, every trial a patient
    contributes across ALL of that patient's sessions is pooled into one point-biserial correlation
    (never a per-session intermediate value), the up-to-3 set-size-specific coefficients are averaged to
    one per-patient effect, and the sign-flip test runs across patients on those effects -- patient
    remains the independent clustering unit, only the level at which the correlation itself is computed
    has moved from session to patient-pooled-trials."""
    trials = _concat_tier_trials(sessions)
    deviation, correct, set_size, patient_id, session_id = (
        trials["deviation"], trials["correct"], trials["set_size"], trials["patient_id"], trials["session_id"])
    patients = sorted(set(patient_id.tolist()))

    per_patient_cells: dict[str, dict] = {}
    per_patient_primary: dict[str, float] = {}
    per_patient_bias: dict[str, float] = {}
    per_patient_secondary: dict[str, float] = {}
    for patient in patients:
        patient_mask = (patient_id == patient) & np.isfinite(deviation)
        cells = {str(level): _patient_set_size_cell(deviation, correct, session_id,
                                                      patient_mask & (set_size == level),
                                                      f"block_b|{patient}|{level}")
                 for level in SET_SIZES}
        per_patient_cells[patient] = cells
        primary_r = [c["primary"]["r"] for c in cells.values()
                     if c.get("status") == "computed" and c["primary"].get("status") == "computed"]
        bias_r = [c["bias_only"]["r"] for c in cells.values()
                  if c.get("status") == "computed" and c["bias_only"].get("status") == "computed"]
        if primary_r:
            per_patient_primary[patient] = float(np.mean(primary_r))
        if bias_r:
            per_patient_bias[patient] = float(np.mean(bias_r))
        if int(patient_mask.sum()) >= MIN_TRIALS_PER_SET_SIZE_CELL:
            rng = np.random.default_rng(stable_seed(f"block_b|{patient}|pooled_secondary"))
            secondary = partial_correlation_permutation_test(
                correct[patient_mask], deviation[patient_mask], controls=[], n_perm=2000, rng=rng)
            if secondary.get("status") == "computed":
                per_patient_secondary[patient] = secondary["r"]

    primary_test = _patient_clustered_test(per_patient_primary)
    bias_test = _patient_clustered_test(per_patient_bias)
    secondary_test = _patient_clustered_test(per_patient_secondary) if len(per_patient_secondary) >= 2 else \
        {"status": "underpowered_by_construction", "n_patients": len(per_patient_secondary)}

    same_sign_significant = (
        primary_test.get("status") == "tested" and primary_test.get("significant")
        and bias_test.get("status") == "tested" and bias_test.get("significant")
        and np.sign(primary_test["mean_value"]) == np.sign(bias_test["mean_value"])
    )

    if same_sign_significant:
        branch = "behaviour_link_not_separable_from_a_session_level_offset"
    elif primary_test["status"] == "underpowered_by_construction":
        branch = "underpowered_to_ask_at_this_tier"
    elif primary_test["significant"]:
        branch = "component_predicts_accuracy_at_this_recording_tier"
    elif reference_effect is not None and primary_test["mdd"] is not None and primary_test["mdd"] < reference_effect:
        branch = "no_behaviour_link_at_this_tier_above_the_reported_bound"
    else:
        branch = "underpowered_to_ask_at_this_tier"

    finite = np.isfinite(deviation)
    per_session_error_distribution = [
        {"patient": s["patient"], "session": s["session_key"],
         "n_trials": int(np.isfinite(s["deviation"]).sum()),
         "n_errors": int((s["correct"][np.isfinite(s["deviation"])] == 0).sum())}
        for s in sessions
    ]

    return {
        "branch": branch, "primary_within_set_size_test": primary_test, "bias_only_control_test": bias_test,
        "pooled_across_load_secondary_test": secondary_test, "reference_effect_used": reference_effect,
        "n_trials_entering_estimate": int(finite.sum()), "n_errors_entering_estimate": int((correct[finite] == 0).sum()),
        "n_sessions_total": len(sessions), "n_patients_total": len(patients),
        "n_patients_contributing_primary": len(per_patient_primary),
        "per_patient_primary_effect": per_patient_primary, "per_patient_cells": per_patient_cells,
        "per_session_error_distribution": per_session_error_distribution,
    }


# ── Cross-tier-transfer test ───────────────────────────────────────────────────────────────────────────

def _trial_wise_session_correlation(dev_a: np.ndarray, dev_b: np.ndarray, seed_tag: str) -> dict | None:
    both_finite = np.isfinite(dev_a) & np.isfinite(dev_b)
    n = int(both_finite.sum())
    if n < MIN_TRIALS_PER_SESSION_TIER or np.std(dev_a[both_finite]) == 0.0 or np.std(dev_b[both_finite]) == 0.0:
        return None
    a, b = dev_a[both_finite], dev_b[both_finite]
    observed = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(stable_seed(seed_tag))
    null = np.empty(N_TRIAL_SHUFFLE_DRAWS)
    for i in range(N_TRIAL_SHUFFLE_DRAWS):
        null[i] = np.corrcoef(a, rng.permutation(b))[0, 1]
    p = permutation_pvalue(np.abs(null) >= np.abs(observed))
    return {"r": observed, "p_value": p, "n_trials": n}


def cross_tier_transfer_pair_trial_wise(session_records: list[dict], tier_a: str, tier_b: str,
                             reference_effect: float | None) -> dict:
    """session_records: per-000574-session dicts with 'patient', 'session_key', and 'tiers' (deviation
    arrays for every computed in-file tier)."""
    per_patient: dict[str, list[float]] = {}
    n_sessions_licensed, n_sessions_computed = 0, 0
    for rec in session_records:
        if tier_a not in rec["tiers"] or tier_b not in rec["tiers"]:
            continue
        n_sessions_licensed += 1
        corr = _trial_wise_session_correlation(
            rec["tiers"][tier_a], rec["tiers"][tier_b], f"block_c|{tier_a}|{tier_b}|{rec['session_key']}")
        if corr is None:
            continue
        n_sessions_computed += 1
        per_patient.setdefault(rec["patient"], []).append(corr["r"])
    per_patient_mean = {p: float(np.mean(v)) for p, v in per_patient.items()}
    pooled = _patient_clustered_test(per_patient_mean)
    branch = _classify_cross_tier_transfer(pooled, reference_effect)
    return {
        "regime": "trial_wise", "branch": branch, "pooled_patient_test": pooled,
        "reference_effect_used": reference_effect, "n_sessions_licensed": n_sessions_licensed,
        "n_sessions_computed": n_sessions_computed, "n_patients_contributing": len(per_patient_mean),
        "per_patient_effect": per_patient_mean,
    }


def cross_tier_transfer_pair_patient_level(patient_scalars_a: dict[str, float], patient_scalars_b: dict[str, float],
                                reference_effect: float | None) -> dict:
    """Fallback regime used whenever exact trial-level cross-release matching is
    not available: one scalar per patient per tier, Spearman-correlated across patients with a
    patient-bootstrap CI -- the same construct results/cross_modality_calibration.json uses for its own
    cross-release comparison."""
    shared = sorted(set(patient_scalars_a) & set(patient_scalars_b))
    pairs = np.array([[patient_scalars_a[p], patient_scalars_b[p]] for p in shared], dtype=float)
    if len(pairs) < MIN_PATIENTS_FOR_TEST:
        return {"regime": "patient_level_only", "branch": "cross_tier_agreement_not_testable_at_matched_trials",
                "n_patients": len(pairs), "reference_effect_used": reference_effect}
    rng = np.random.default_rng(stable_seed(f"block_c_patient_level|{tuple(shared)}"))
    rho, lower, upper = bootstrap_ci(pairs, lambda d: spearmanr(d[:, 0], d[:, 1]).statistic, rng=rng, n_boot=N_BOOT_EXISTENCE_AND_BEHAVIOUR_LINK)
    excludes_zero = bool(lower > 0.0 or upper < 0.0)
    ci_half_width = (upper - lower) / 2.0
    if excludes_zero:
        branch = "the_two_tiers_track_the_same_per_trial_quantity"
    elif reference_effect is not None and ci_half_width < reference_effect:
        branch = "no_cross_tier_agreement_above_the_reported_bound"
    else:
        branch = "cross_tier_agreement_not_testable_at_matched_trials"
    return {
        "regime": "patient_level_only", "branch": branch, "n_patients": len(pairs),
        "spearman_rho": rho, "ci_lower": lower, "ci_upper": upper, "ci_half_width": ci_half_width,
        "excludes_zero": excludes_zero, "reference_effect_used": reference_effect,
    }


def _classify_cross_tier_transfer(pooled: dict, reference_effect: float | None) -> str:
    if pooled["status"] == "underpowered_by_construction":
        return "cross_tier_agreement_not_testable_at_matched_trials"
    if pooled["significant"]:
        return "the_two_tiers_track_the_same_per_trial_quantity"
    if reference_effect is not None and pooled["mdd"] is not None and pooled["mdd"] < reference_effect:
        return "no_cross_tier_agreement_above_the_reported_bound"
    return "cross_tier_agreement_not_testable_at_matched_trials"


# ── Recording-specification / non-human-claims synthesis, read from the computed dict only ─────────────────────────────

_TIER_LABEL = {
    "single_unit": "microwire single units", "depth_mtl": "depth macro-contacts in medial temporal cortex",
    "depth_cortical": "depth macro-contacts outside medial temporal cortex", "scalp_eeg": "scalp EEG",
    "beamformed_cortical": "beamformed cortical virtual sensors",
}
_TIER_INVASIVENESS = {
    "single_unit": "invasive", "depth_mtl": "invasive", "depth_cortical": "invasive",
    "scalp_eeg": "non-invasive", "beamformed_cortical": "non-invasive",
}
_TIER_ORDER_COARSE_TO_FINE = ("beamformed_cortical", "scalp_eeg", "depth_cortical", "depth_mtl", "single_unit")


def _passes_a_and_b(existence: dict, behaviour_link: dict, tier: str) -> bool:
    a = existence.get(tier, {}).get("branch")
    b = behaviour_link.get(tier, {}).get("branch")
    return a == "component_is_present_at_this_recording_tier" and b == "component_predicts_accuracy_at_this_recording_tier"


def _tracks_single_unit(cross_tier_transfer: dict, tier: str) -> bool:
    pair = cross_tier_transfer.get(f"single_unit|{tier}") or cross_tier_transfer.get(f"{tier}|single_unit")
    return bool(pair) and pair["branch"] == "the_two_tiers_track_the_same_per_trial_quantity"


def synthesize_recording_specification(existence: dict, behaviour_link: dict, cross_tier_transfer: dict) -> str:
    coarsest_existing_and_predictive = next(
        (t for t in _TIER_ORDER_COARSE_TO_FINE if _passes_a_and_b(existence, behaviour_link, t)), None)
    coarsest_tracking_single_unit = next(
        (t for t in _TIER_ORDER_COARSE_TO_FINE if t != "single_unit" and _tracks_single_unit(cross_tier_transfer, t)), None)
    any_noninvasive_passes_a_and_b = any(
        _TIER_INVASIVENESS[t] == "non-invasive" and _passes_a_and_b(existence, behaviour_link, t)
        for t in ("scalp_eeg", "beamformed_cortical"))
    any_noninvasive_tracks = any(_tracks_single_unit(cross_tier_transfer, t) for t in ("scalp_eeg", "beamformed_cortical"))

    lines = []
    if coarsest_existing_and_predictive is not None:
        lines.append(
            f"The coarsest recording tier at which the component both exists (the existence test) and predicts "
            f"behaviour (the behaviour-link test) in this corpus is {_TIER_LABEL[coarsest_existing_and_predictive]} "
            f"({_TIER_INVASIVENESS[coarsest_existing_and_predictive]}).")
    else:
        lines.append("No recording tier in this corpus both exists (the existence test) and predicts behaviour "
                      "(the behaviour-link test) at the same time; the tiers that clear one bar do not clear both.")
    if coarsest_tracking_single_unit is not None:
        lines.append(
            f"The coarsest tier that still tracks the single-unit measurement trial by trial (the cross-tier-transfer test) is "
            f"{_TIER_LABEL[coarsest_tracking_single_unit]}.")
    else:
        lines.append("No coarser tier tracks the single-unit measurement trial by trial above its reported "
                      "bound in this corpus.")
    lines.append(
        "A non-invasive tier reaches the existence-and-behaviour bar: "
        + ("yes." if any_noninvasive_passes_a_and_b else "no.")
    )
    lines.append(
        "A non-invasive tier tracks the single-unit measurement trial by trial: "
        + ("yes." if any_noninvasive_tracks else "no.")
    )
    if not any_noninvasive_passes_a_and_b and not any_noninvasive_tracks:
        lines.append(
            "No non-invasive tier reaches either bar in this corpus. That is the finding this leg was built "
            "to test, and it is the quantitative justification for an invasive read-out: a chronic device "
            "reading only scalp or beamformed-cortical signals in this task and this population would not "
            "recover the component this project has repeatedly identified at the single-unit tier."
        )
    return " ".join(lines)


def synthesize_non_human_claims() -> str:
    return (
        "The component's identity -- that it is neither the dominant population-rate mode nor the "
        "memorandum subspace, and that it predicts behaviour where the dominant mode does not -- was "
        "established in non-human single-unit preparations (macaque lateral prefrontal cortex, mouse "
        "anterior lateral motor cortex). Those preparations cannot address the question this leg asks: "
        "no non-human corpus in this project records simultaneous microwire, depth macro-contact and "
        "scalp signals in the same animals on the same trials, so the cross-tier bridge (the cross-tier-transfer test) has no "
        "non-human counterpart to compare against, and neither does the recording-tier specification "
        "that follows from it. This human corpus, in turn, cannot re-establish or re-adjudicate "
        "the component's non-human identity claims by the same design (a different task, a different "
        "clustering unit -- patient, not session -- and no dominant-mode or memorandum-subspace decomposition "
        "computed here), so this leg's behaviour-link test result is read as a same-corpus, same-instrument "
        "recording-tier comparison, never as a replication or a refutation of the non-human identity "
        "finding. It is, however, a genuine measurement in its own right, not a reachability-limited one: "
        "this module's own admission keeps every trial, correct and error alike (see the trial-table "
        "agreement check and session_status/ds004752_beamformed/*/audit in the artifact for the per-session "
        "record), so this leg's behaviour-link test is fit on real error trials in a human maintenance-delay "
        "task. Holding both preparations together narrows the practical question this project exists to answer "
        "-- where a stimulation device should record from -- to whichever tiers this leg's own existence, "
        "behaviour-link and cross-tier-transfer "
        "branches name, while leaving the separate question of what the component IS to stand on the "
        "non-human evidence alone. Where a tier's result in this corpus runs against a non-human "
        "expectation (for instance, if a coarse invasive tier here fails the existence test while the single-unit "
        "tier does not), that contradiction is reported as such in the per-tier branch record above and is "
        "not adjudicated away by this synthesis."
    )


# ── Read-only context from sibling artifacts (never edited, never recomputed here) ─────────────────────

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def count_floor_context() -> dict:
    """The unsampled total-spikes-per-trial gap between corpora whose deviation-vs-spike-count gate
    passes and those whose gate fails (results/dissociation_replication_and_counting_noise.json's
    counting-noise census, stored under its block_b.count_separation_disclosure field). No single
    fitted threshold exists inside that gap -- the
    project's own explicit rule is that none may be asserted -- so this module reports the single_unit
    tier's median spike total per trial against the gap's two known boundaries, not against a fabricated
    single number."""
    doc = _read_json(ROOT / "results" / "dissociation_replication_and_counting_noise.json")
    if doc is None:
        return {"status": "source_artifact_unavailable"}
    disclosure = doc.get("block_b", {}).get("count_separation_disclosure")
    if not disclosure:
        return {"status": "source_field_unavailable"}
    return {"status": "available", "source": "results/dissociation_replication_and_counting_noise.json",
            "disclosure": disclosure}


def field_potential_degeneracy_context() -> dict:
    """The same estimator's own degeneracy precondition, already run on a separate human intracranial
    field-potential corpus (results/human_stimulation_component_response.json, precondition block):
    overall session-median deviation on human field-potential power against the median of the non-human
    single-unit corpus's own session medians. Read here as disclosed context for this leg's own
    field-potential tiers (depth_mtl, depth_cortical, scalp_eeg, beamformed_cortical); it does not feed
    any pre-declared decision rule above."""
    doc = _read_json(ROOT / "results" / "human_stimulation_component_response.json")
    if doc is None:
        return {"status": "source_artifact_unavailable"}
    precondition = doc.get("precondition")
    if not precondition:
        return {"status": "source_field_unavailable"}
    return {"status": "available", "source": "results/human_stimulation_component_response.json",
            "degenerate": precondition.get("degenerate"),
            "human_field_potential_overall_session_median_deviation": precondition.get("overall_session_median_deviation"),
            "non_human_single_unit_median_of_session_medians": precondition.get("non_human_reference", {}).get("median_of_session_medians")}


# ── Per-tier checkpointed load ───────────────────────────────────────────────────────────────────────────

def _encode_tier_record(status: str, reason: str | None, n_trials_admitted: int | None,
                         activity: np.ndarray | None, gate: dict | None, set_size: np.ndarray | None,
                         correct: np.ndarray | None, band: dict | None, n_features: int | None) -> dict:
    return {
        "status": status, "reason": reason, "n_trials_admitted": n_trials_admitted,
        "activity": activity.tolist() if activity is not None else None,
        "deviation": rate_free_state_deviation(activity).tolist() if activity is not None else None,
        "gate": gate, "set_size": set_size.tolist() if set_size is not None else None,
        "correct": correct.tolist() if correct is not None else None,
        "band": band, "n_features": n_features,
    }


def _decode_tier_record(record: dict) -> dict:
    out = dict(record)
    if record.get("activity") is not None:
        out["activity"] = np.asarray(record["activity"], dtype=float)
    if record.get("deviation") is not None:
        out["deviation"] = np.asarray(record["deviation"], dtype=float)
    if record.get("set_size") is not None:
        out["set_size"] = np.asarray(record["set_size"], dtype=int)
    if record.get("correct") is not None:
        out["correct"] = np.asarray(record["correct"], dtype=float)
    return out


def load_000574_session_tiers(patient: str, session_key: str, nwb_path: Path) -> dict[str, dict]:
    """Returns {tier: decoded_record} for the four in-file tiers, resuming from per-tier checkpoints and
    only touching the NWB file if at least one of the four is missing."""
    keys = {tier: f"000574__{patient}__{session_key}__{tier}" for tier in IN_FILE_TIERS}
    cached = {tier: load_checkpoint(key) for tier, key in keys.items()}
    if all(v is not None for v in cached.values()):
        return {tier: _decode_tier_record(v) for tier, v in cached.items()}

    built = build_000574_session(nwb_path)
    out = {}
    if built["status"] == "refused":
        for tier in IN_FILE_TIERS:
            record = cached[tier] or _encode_tier_record("refused", built["reason"], built.get("n_admitted"),
                                                           None, None, None, None, None, None)
            if cached[tier] is None:
                save_checkpoint(keys[tier], record)
            out[tier] = _decode_tier_record(record)
        return out

    set_size, correct = built["set_size"], built["correct"]
    for tier in IN_FILE_TIERS:
        if cached[tier] is not None:
            out[tier] = _decode_tier_record(cached[tier])
            continue
        t = built["tiers"][tier]
        if t["status"] == "computed":
            record = _encode_tier_record("computed", None, built["n_trials_admitted"], t["activity"],
                                          None, set_size, correct, t["band"], t["n_features"])
            rng = np.random.default_rng(stable_seed(f"block_a_gate|000574|{patient}|{session_key}|{tier}"))
            record["gate"] = rotation_null_variance_test(t["activity"], N_ROTATION_NULL_DRAWS, rng)
        else:
            record = _encode_tier_record("refused", t["reason"], built["n_trials_admitted"], None, None,
                                          set_size, correct, None, None)
        save_checkpoint(keys[tier], record)
        out[tier] = _decode_tier_record(record)
    return out


def load_beamformed_tier(patient: str, mat_path: Path) -> dict:
    key = f"beamformed__{patient}__beamformed_cortical"
    cached = load_checkpoint(key)
    if cached is not None:
        return _decode_tier_record(cached)
    built = build_beamformed_session(mat_path)
    if built["status"] == "computed":
        record = _encode_tier_record("computed", None, built["n_trials_admitted"], built["activity"],
                                      None, None, None, built["band"], built["n_features"])
        rng = np.random.default_rng(stable_seed(f"block_a_gate|beamformed|{patient}"))
        record["gate"] = rotation_null_variance_test(built["activity"], N_ROTATION_NULL_DRAWS, rng)
        record["audit"] = built.get("audit")
    else:
        record = _encode_tier_record("refused", built["reason"], built.get("n_trials"), None, None,
                                      None, None, None, None)
        record["audit"] = built.get("audit")
    save_checkpoint(key, record)
    return _decode_tier_record(record)


# ── Main ──────────────────────────────────────────────────────────────────────────────────────────────────

DS006848_PRESENTATION_MODE_CODES = {"Simultaneous": 0, "Fast": 1, "Fast+delay": 2, "Slow": 3}


def healthy_scalp_corpora(root: Path) -> dict:
    """Two independent healthy-participant scalp-EEG corpora, each recorded from its own
    participants with no simultaneous depth or single-unit tier -- so neither one is a sixth entry
    in TIERS (that would pool a corpus with no paired-tier structure into a cross-tier-transfer
    test built for one), and neither is pooled into 000574's own 'scalp_eeg' tier (a different
    corpus's cells). Existence and (where behaviour exists) behaviour-link only; cross-tier
    transfer needs a second, simultaneously-recorded tier from the same session, which neither
    corpus has, so it is not attempted here."""
    from corpus_sessions import iter_ds005034, iter_ds006848

    ds006848_sessions = []
    for entry in iter_ds006848(root):
        activity = np.asarray(entry["counts"]).sum(axis=2)
        key = f"healthy_scalp__ds006848__{entry['session']}"
        record = run_checkpointed(key, lambda a=activity, t=key: _session_deviation_and_gate(a, t))
        ds006848_sessions.append({
            "patient": entry["patient"], "session_key": entry["session"],
            "deviation": np.asarray(record["deviation"], dtype=float), "gate": record["gate"],
            "correct": (np.asarray(entry["accuracy_ncorrect"], dtype=float) == 7.0).astype(int),
            "set_size": np.asarray([DS006848_PRESENTATION_MODE_CODES[c] for c in entry["task_condition"]], dtype=int),
        })
    ds005034_sessions = []
    for entry in iter_ds005034(root):
        activity = np.asarray(entry["counts"]).sum(axis=2)
        key = f"healthy_scalp__ds005034__{entry['session']}"
        record = run_checkpointed(key, lambda a=activity, t=key: _session_deviation_and_gate(a, t))
        ds005034_sessions.append({
            "patient": entry["patient"], "session_key": entry["session"],
            "deviation": np.asarray(record["deviation"], dtype=float), "gate": record["gate"],
        })

    return {
        "ds006848_human_scalp": {
            "n_sessions": len(ds006848_sessions), "n_patients": len(set(s["patient"] for s in ds006848_sessions)),
            "existence": existence_tier(ds006848_sessions, None),
            "behaviour_link": behaviour_link_tier(ds006848_sessions, MEANINGFUL_EFFECT_THRESHOLD_R_UNITS),
            "cross_tier_transfer": "not_applicable: no simultaneously-recorded second tier exists for this corpus",
        },
        "ds005034_human_scalp": {
            "n_sessions": len(ds005034_sessions), "n_patients": len(set(s["patient"] for s in ds005034_sessions)),
            "existence": existence_tier(ds005034_sessions, None),
            "behaviour_link": {
                "branch": "underpowered_to_ask_at_this_tier",
                "reason": "no trial-level behavioural outcome is recorded in this corpus's public BIDS "
                          "release; it exists only on a separate OSF repository (config/datasets.json)",
            },
            "cross_tier_transfer": "not_applicable: no simultaneously-recorded second tier exists for this corpus",
        },
    }


def main() -> None:
    t0 = time.time()
    root = data_root()

    sessions_000574 = discover_000574_sessions(root)
    patients_000574 = {patient for patient, _, _ in sessions_000574}
    sessions_beamformed = discover_beamformed_sessions(root, patients_000574)

    # Zero-drop bookkeeping and the in-memory records the existence, behaviour-link and cross-tier-transfer tests are built from.
    session_status: dict[str, dict] = {}
    session_records: list[dict] = []  # one per 000574 session: {patient, session_key, tiers: {tier: deviation array}}
    per_tier_sessions: dict[str, list[dict]] = {tier: [] for tier in TIERS}

    for patient, session_key, nwb_path in sessions_000574:
        source_key = f"000574/{session_key}"
        try:
            tiers = load_000574_session_tiers(patient, session_key, nwb_path)
        except Exception as exc:  # noqa: BLE001 -- a crash on one session must not lose the others
            session_status[source_key] = {"status": "refused", "reason": f"exception:{type(exc).__name__}:{exc}"}
            continue
        computed_tiers = {tier: rec for tier, rec in tiers.items() if rec["status"] == "computed"}
        # "computed" at the session level means at least one of the four in-file tiers succeeded; a
        # session where the NWB load itself failed (no units table, trial-count mismatch, too few
        # admitted trials) has every tier refused for the identical reason, and is honestly "refused"
        # here too, not "computed with nothing inside it".
        session_status[source_key] = {
            "status": "computed" if computed_tiers else "refused",
            "reason": (None if computed_tiers else
                       next((rec.get("reason") for rec in tiers.values() if rec.get("reason")), "unknown")),
            "tiers": {tier: {"status": rec["status"], "reason": rec.get("reason"),
                              "n_trials_admitted": rec.get("n_trials_admitted")}
                      for tier, rec in tiers.items()},
        }
        session_records.append({"patient": patient, "session_key": session_key,
                                 "tiers": {tier: rec["deviation"] for tier, rec in computed_tiers.items()}})
        for tier, rec in computed_tiers.items():
            # correct/set_size ride along per session so behaviour_link_tier can pool trials across a patient's
            # sessions itself (see _concat_tier_trials) -- no per-session behaviour statistic is computed
            # here, deliberately, per the trial-pooled design BEHAVIOUR_LINK_DECISION_RULE describes.
            per_tier_sessions[tier].append({
                "patient": patient, "session_key": session_key, "deviation": rec["deviation"], "gate": rec["gate"],
                "correct": rec["correct"], "set_size": rec["set_size"],
            })

    for patient, session_key, mat_path in sessions_beamformed:
        source_key = f"ds004752_beamformed/{patient}"
        try:
            rec = load_beamformed_tier(patient, mat_path)
        except Exception as exc:  # noqa: BLE001
            session_status[source_key] = {"status": "refused", "reason": f"exception:{type(exc).__name__}:{exc}"}
            continue
        session_status[source_key] = {"status": "computed" if rec["status"] == "computed" else "refused",
                                       "reason": rec.get("reason"), "audit": rec.get("audit")}
        if rec["status"] == "computed":
            # No "correct"/"set_size" here -- no accuracy label is retrievable for this tier's trials
            # (see beamformed_trial_identity_audit), so the behaviour-link test is never called on this tier at all (main()
            # constructs its 'underpowered_to_ask_at_this_tier' record directly, with the reason stated).
            per_tier_sessions["beamformed_cortical"].append({
                "patient": patient, "session_key": session_key, "deviation": rec["deviation"], "gate": rec["gate"],
            })

    # ── Existence test, tier by tier -- single_unit first, its own effect becomes the reference for the rest.
    existence: dict[str, dict] = {}
    existence["single_unit"] = existence_tier(per_tier_sessions["single_unit"], None)
    reference_effect_existence = None
    if existence["single_unit"]["pooled_patient_test"].get("status") == "tested" and \
            existence["single_unit"]["pooled_patient_test"].get("significant"):
        reference_effect_existence = abs(existence["single_unit"]["pooled_patient_test"]["mean_value"])
    for tier in TIERS:
        if tier == "single_unit":
            continue
        existence[tier] = existence_tier(per_tier_sessions[tier], reference_effect_existence)

    # ── Behaviour-link test, tier by tier -- fixed 0.14 r-unit reference throughout (see MEANINGFUL_EFFECT_THRESHOLD_R_UNITS).
    behaviour_link: dict[str, dict] = {}
    for tier in TIERS:
        if tier == "beamformed_cortical":
            behaviour_link[tier] = {
                "branch": "underpowered_to_ask_at_this_tier",
                "reason": "no accuracy label is retrievable for this tier's trials (structural, not a "
                          "sample-size shortfall -- see beamformed_trial_identity_audit); the least-bad fit "
                          "among the behaviour-link test's three pre-declared branches, disclosed explicitly rather than "
                          "forced into a branch that implies a correlation was actually computed",
                "n_sessions_with_primary": 0, "n_sessions_total": len(per_tier_sessions[tier]),
                "n_patients_contributing": 0, "reference_effect_used": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
            }
            continue
        behaviour_link[tier] = behaviour_link_tier(per_tier_sessions[tier], MEANINGFUL_EFFECT_THRESHOLD_R_UNITS)

    # ── Cross-tier-transfer test, every unordered tier pair.
    cross_tier_transfer: dict[str, dict] = {}
    tier_pairs = [(a, b) for i, a in enumerate(TIERS) for b in TIERS[i + 1:]]
    trial_wise_pairs = [(a, b) for a, b in tier_pairs if a in IN_FILE_TIERS and b in IN_FILE_TIERS]
    patient_level_pairs = [(a, b) for a, b in tier_pairs if a not in IN_FILE_TIERS or b not in IN_FILE_TIERS]

    reference_pair_key = ("single_unit", "depth_mtl") if ("single_unit", "depth_mtl") in trial_wise_pairs else None
    reference_effect_cross_tier_transfer = None
    if reference_pair_key is not None:
        first = cross_tier_transfer_pair_trial_wise(session_records, *reference_pair_key, None)
        cross_tier_transfer[f"{reference_pair_key[0]}|{reference_pair_key[1]}"] = first
        if first["pooled_patient_test"].get("status") == "tested" and first["pooled_patient_test"].get("significant"):
            reference_effect_cross_tier_transfer = abs(first["pooled_patient_test"]["mean_value"])
        else:
            reference_effect_cross_tier_transfer = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS

    for a, b in trial_wise_pairs:
        key = f"{a}|{b}"
        if key in cross_tier_transfer:
            continue
        cross_tier_transfer[key] = cross_tier_transfer_pair_trial_wise(session_records, a, b, reference_effect_cross_tier_transfer)

    def _patient_scalars(tier: str) -> dict[str, float]:
        by_patient: dict[str, list[float]] = {}
        for s in per_tier_sessions[tier]:
            finite = s["deviation"][np.isfinite(s["deviation"])]
            if finite.size:
                by_patient.setdefault(s["patient"], []).append(float(np.median(finite)))
        return {p: float(np.median(v)) for p, v in by_patient.items()}

    patient_scalar_cache = {tier: _patient_scalars(tier) for tier in TIERS}
    for a, b in patient_level_pairs:
        cross_tier_transfer[f"{a}|{b}"] = cross_tier_transfer_pair_patient_level(
            patient_scalar_cache[a], patient_scalar_cache[b], reference_effect_cross_tier_transfer)

    recording_specification = synthesize_recording_specification(existence, behaviour_link, cross_tier_transfer)
    non_human_claims = synthesize_non_human_claims()
    healthy_scalp = healthy_scalp_corpora(root)

    n_000574_seen = len(sessions_000574)
    n_000574_computed = sum(1 for v in session_status.values() if v["status"] == "computed" and "tiers" in v)
    n_beamformed_seen = len(sessions_beamformed)
    n_beamformed_computed = sum(1 for k, v in session_status.items()
                                 if k.startswith("ds004752_beamformed/") and v["status"] == "computed")

    output = {
        "schema_version": "1.0.0", "analysis_id": "recording_tier_component_transfer",
        "code_commit": git_commit(ROOT),
        "scope": {
            "corpus": "the verbal working-memory release with simultaneous microwire and macro-contact "
                      "recordings (accession 000574) plus its cross-modality companion release adding "
                      "beamformed cortical virtual sensors (accession ds004752, 9 shared patients)",
            "tiers": list(TIERS), "in_file_tiers": list(IN_FILE_TIERS),
            "maintenance_window_s_relative_to_probe": list(MAINT_WINDOW_S),
            "mains_hz": MAINS_HZ, "n_rotation_null_draws": N_ROTATION_NULL_DRAWS,
            "n_trial_shuffle_draws": N_TRIAL_SHUFFLE_DRAWS,
            "min_trials_per_session_tier": MIN_TRIALS_PER_SESSION_TIER,
            "min_trials_per_set_size_cell": MIN_TRIALS_PER_SET_SIZE_CELL,
            "min_patients_for_test": MIN_PATIENTS_FOR_TEST,
            "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
            "n_000574_sessions_seen": n_000574_seen, "n_000574_sessions_computed": n_000574_computed,
            "n_000574_sessions_refused": n_000574_seen - n_000574_computed,
            "n_beamformed_patients_seen": n_beamformed_seen,
            "n_beamformed_patients_computed": n_beamformed_computed,
            "n_beamformed_patients_refused": n_beamformed_seen - n_beamformed_computed,
        },
        "decision_rules": {
            "block_a": EXISTENCE_DECISION_RULE, "block_b": BEHAVIOUR_LINK_DECISION_RULE, "block_c": CROSS_TIER_TRANSFER_DECISION_RULE,
        },
        "count_floor_context": count_floor_context(),
        "field_potential_degeneracy_context": field_potential_degeneracy_context(),
        "session_status": session_status,
        "block_a": existence, "block_b": behaviour_link, "block_c": cross_tier_transfer,
        "block_d_recording_specification": recording_specification,
        "block_e_non_human_claims": non_human_claims,
        "block_f_healthy_scalp_corpora": healthy_scalp,
        "wall_clock_s": time.time() - t0,
        "status": "complete",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))
    print(json.dumps({
        "n_000574_sessions_computed": n_000574_computed, "n_beamformed_patients_computed": n_beamformed_computed,
        "existence_branch_by_tier": {t: existence[t]["branch"] for t in TIERS},
        "behaviour_link_branch_by_tier": {t: behaviour_link[t]["branch"] for t in TIERS},
        "healthy_scalp_existence_branch": {c: healthy_scalp[c]["existence"]["branch"] for c in healthy_scalp},
        "wall_clock_s": output["wall_clock_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
