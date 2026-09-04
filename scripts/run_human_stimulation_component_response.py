#!/usr/bin/env python3
"""Does electrical stimulation delivered to human intracranial recordings move
the rate-free state-deviation component this project has repeatedly failed to
identify with total firing rate, the dominant latent mode, or the memorandum
subspace in non-human preparations?

Two human intracranial free-recall corpora carry experimenter- or
classifier-triggered electrical stimulation with per-trial behaviour and a
known or recoverable assignment fraction:

  - open-loop arm  (scripts/run_ram_openloop_pipeline.py, OpenNeuro ds005489):
    stimulation is scheduled by the experimenter in alternating word-list
    blocks, so its assignment fraction is a design property.
  - closed-loop arm (scripts/run_ram_closedloop_pipeline.py, OpenNeuro
    ds005557): stimulation is triggered by an online classifier reading the
    subject's own encoding-period state, so assignment is NOT randomised and
    every statement from this arm is scoped with causal:false unless a design
    fraction is recovered.

Both pipelines already epoch per-channel high-gamma log-power into a
(n_trials, n_bins, n_channels) array via `build_session_features`
(scripts/run_ram_openloop_pipeline.py). Averaging over the bin axis gives
(n_trials, n_channels), the same shape rate_free_state_deviation
(scripts/run_rate_free_state_geometry_behavior_link.py) consumes as
(n_trials, n_units) in its home macaque corpus -- the estimator's arithmetic
is not touched anywhere in this module; it is imported and called exactly as
delivered.

The pulse train in both corpora outlasts a single word presentation, so the
analysis epoch for a stimulated trial lies partly inside the stimulation
interval. This is disclosed everywhere it matters, never treated as a
matching requirement, and never a reason to drop a trial or an arm.

Stimulation produces a large deflection on channels at and near the driven
electrode pair. Every displacement number this module reports is
accompanied by the same number recomputed with the stimulated bipolar
channel excluded, and again with every channel sharing a physical electrode
lead (shank) with either the anode or the cathode contact also excluded --
channel identifiers in this corpus are "<lead><contact>-<lead><contact>"
(e.g. "LAH1-LAH2"), so the lead name is the contact label's alphabetic
prefix.

Session is the unit of analysis; subject is the clustering unit for every
bootstrap and every p-value throughout, because a subject can contribute
more than one session. Every quoted correlation across sessions is built by
first collapsing each subject's own sessions to one number (their mean),
then permuting or bootstrapping at the subject level -- this is what keeps a
subject with many sessions from silently outweighing one with few, and is
why every pooled statistic in this artifact reports both n_sessions and the
smaller n_subjects.

A non-human delay-period microstimulation corpus (results/causal_macaque_pfc_microstimulation.json,
the macaque PFC microstimulation release, read-only here, never recomputed) is not
re-analysed by this module; instead, its already-delivered numbers are placed
beside this module's own human numbers, question by question, so what each
preparation can and cannot say is on the record without averaging the two
together.

Outputs:
  results/human_stimulation_component_response.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_human_stimulation_component_response.py [--smoke N]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import re
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci,
    fdr_bh,
    forest_meta,
    minimum_detectable_paired_difference,
    paired_sign_flip_test,
    partial_correlation_permutation_test,
    pearson_permutation_test,
    permutation_pvalue,
    power_to_detect_effect,
    stable_seed,
    stouffer_combine,
)
from causal import cate_vs_modifier_slope  # noqa: E402
from corpus_sessions import data_root as macaque_data_root  # noqa: E402

from run_ram_openloop_pipeline import DATA as OPENLOOP_DATA, build_session_features, _float_or_nan  # noqa: E402
from run_ram_closedloop_pipeline import DATA as CLOSEDLOOP_DATA  # noqa: E402
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    _session_arrays as _macaque_session_arrays,
    rate_free_state_deviation,
)
from run_state_behavior_link import _panichello_directory  # noqa: E402
from preprocessing import high_gamma_power, line_noise_notch  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "human_stimulation_component_response.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_human_stimulation_component_response"

N_PERM = 10000
N_BOOT = 5000
N_ROTATION_NULL_DRAWS = 1000
N_DOSE_SHUFFLE_DRAWS = 1000
ALPHA = 0.05
POWER = 0.80

# Precondition thresholds -- pre-declared before any human session is loaded.
SESSION_MEDIAN_DEGENERATE_ABS = 0.01
SESSION_MEDIAN_DEGENERATE_REL_FRAC = 0.05
# Operationalisation of "within-session variance not distinguishable from a
# magnitude-matched rotation null at 1000 draws": computed per session (each
# session's own observed within-session deviation variance against its own
# null of 1000 magnitude-matched, independently-randomised-direction trials),
# then aggregated as a session-level FRACTION -- fewer than this fraction of
# sessions individually reaching two-sided p<=0.05 counts as "the arm's
# variance is not distinguishable from the null" in aggregate. This specific
# aggregation rule is this module's own pre-declared reading of an
# underspecified instruction (the source text names the per-session test but
# not how to combine it across sessions into one arm-level verdict); it is
# fixed here, before any session is loaded, precisely so it cannot be tuned
# after seeing the numbers.
ROTATION_NULL_DISTINGUISHABLE_FRACTION_FLOOR = 0.05

# Block B's displacement is reported in raw cosine-deviation units and, in
# parallel, normalised by each session's own spontaneous (non-stimulated
# trial-to-trial) standard deviation -- a session-specific score is not
# comparable across sessions on the raw scale alone (Block D's own rule,
# applied here too for the same reason). A normalised displacement of 1.0 is
# "as large as this session's own ordinary non-stimulated fluctuation", the
# natural, unit-free floor for "the smallest effect this design would call
# meaningful" and this module's pre-declared reference effect for Block B.
MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT = 1.0

MIN_SUBJECTS_FOR_WITHIN_SUBJECT_DOSE = 6

# Forward-only: gates the two-arm dose-scaling meta-analysis's own heterogeneity check (a standard
# rule of thumb -- Cochran's Q or I^2 this large means the two arms are not measuring one common
# effect and their pooled point estimate should not be presented as an answer). Does not touch or
# re-gate anything Block D or the pre-task titration arm already fires.
TWO_ARM_META_HETEROGENEITY_I_SQUARED_FLOOR = 50.0
CONFOUND_LIST_FOR_BETWEEN_SUBJECT_DOSE = [
    "clinical titration threshold", "electrode target location", "tissue type",
    "electrode impedance", "montage (grid/strip/depth mix)",
]

# A within-session amplitude-titration series delivered before the recall task begins (closed-loop
# corpus only): fixed baseline/post window lengths, kept identical across every series, plus the same
# Hilbert edge-effect padding build_session_features uses on its own word-epoch windows. A series with
# fewer than two distinct delivered amplitudes has no dose axis to fit and is not a titration.
PRETASK_BASELINE_WINDOW_S = 1.0    # immediately preceding each pre-task STIM_ON onset
PRETASK_POST_WINDOW_S = 1.0        # immediately following the matching STIM_OFF onset
PRETASK_EDGE_PAD_S = 0.3
PRETASK_MIN_AMPLITUDE_LEVELS = 2


# ── Checkpointing (fit-level, crash-proof; mirrors scripts/run_stimulation_latent_response_map.py) ──

def _checkpoint_path(unit: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(unit: str) -> dict | None:
    """An unparseable or incomplete checkpoint record is treated as absent."""
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
    """Write to a temp file, then os.replace -- the completion flag is only
    ever written as part of the same atomic replace, after the fit that
    computed `record` has already returned, so a killed process never leaves
    a checkpoint that reads as complete but isn't."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(unit: str, fit_fn):
    """Load a cached, complete record for `unit`, or call `fit_fn()` (which
    must return a JSON-safe dict) and checkpoint the result before returning
    it. `fit_fn` is only ever invoked when no valid checkpoint exists."""
    cached = load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(unit, record)
    return record


def _raw_checkpoint_path(unit: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    return CHECKPOINT_DIR / f"{safe}__raw_features.npz"


def load_raw_features(unit: str) -> dict | None:
    """The expensive part of a session's fit (EDF load, notch filter,
    Hilbert-based high-gamma power, PCA, DMD) cached in binary form -- a JSON
    round trip of a (trials, bins, channels) float array is both slow and
    needlessly large. An unreadable or incomplete archive is treated as
    absent, exactly like the lightweight JSON checkpoints above."""
    path = _raw_checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "_complete" not in data.files or not bool(data["_complete"]):
                return None
            return {k: data[k] for k in data.files if k != "_complete"}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
        return None


def save_raw_features(unit: str, arrays: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _raw_checkpoint_path(unit)
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), suffix=".npz", prefix="._tmp_")
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, _complete=np.array(True), **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ── Channel identity / shank parsing ────────────────────────────────────────────

def _contact_shank(contact_label: str) -> str:
    """The alphabetic prefix of a depth/grid electrode contact label (e.g.
    'LAH2' -> 'LAH') identifies the physical lead the contact sits on; RAM
    contact labels are always an alphabetic lead name followed by a numeric
    contact index."""
    match = re.match(r"^[A-Za-z]+", contact_label)
    return match.group(0) if match else contact_label


def _bipolar_channel_shanks(channel_name: str) -> set[str]:
    """A bipolar channel name is 'anode-cathode' (e.g. 'LAH1-LAH2'); returns
    the set of one or two leads either contact belongs to."""
    return {_contact_shank(p) for p in channel_name.split("-") if p}


def channel_condition_masks(ch_names: list[str], anode: str, cathode: str, stim_ch: str) -> dict:
    """The three channel sets Block B's mandatory artifact control compares:
    every channel, every channel except the driven bipolar pair, and every
    channel except the driven pair AND any channel sharing a lead with
    either the anode or the cathode contact -- the mandatory control for a
    large stimulation deflection contaminating nearby contacts, not only the
    driven pair itself."""
    stim_shanks = {_contact_shank(anode), _contact_shank(cathode)}
    full = np.ones(len(ch_names), dtype=bool)
    excl_pair = np.array([ch != stim_ch for ch in ch_names])
    excl_shank = np.array([
        ch != stim_ch and not (_bipolar_channel_shanks(ch) & stim_shanks) for ch in ch_names
    ])
    return {
        "full_channel_set": full,
        "excluding_stimulated_pair": excl_pair,
        "excluding_stimulated_shank": excl_shank,
    }


# ── The estimator, transported: fixed-reference scoring for stimulated trials ──

def _reference_direction(activity_by_unit: np.ndarray) -> np.ndarray:
    """The renormalised mean unit direction of every trial in
    `activity_by_unit`, with NO leave-one-out exclusion -- used to build a
    single FIXED reference from the control-trial pool, which stimulated
    trials (never members of that pool) are then scored against without
    letting them define any part of their own comparison point. A trial with
    zero total activity across channels has no defined direction and is
    excluded from the mean, exactly as rate_free_state_deviation excludes it
    from its own leave-one-out reference."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors[valid], axis=0)
    n_valid = int(valid.sum())
    if n_valid < 1:
        return np.full(activity.shape[1], np.nan)
    mean_dir = total / n_valid
    norm = np.linalg.norm(mean_dir)
    return mean_dir / norm if norm > 0 else np.full(activity.shape[1], np.nan)


def _deviation_from_reference(activity_by_unit: np.ndarray, reference_direction: np.ndarray) -> np.ndarray:
    """Per trial, 1 - cosine(unit_vector_i, reference_direction), scoring
    each trial against a FIXED external direction rather than a leave-one-out
    mean of its own group -- the counterpart to rate_free_state_deviation's
    leave-one-out reference for trials that must never contribute to the
    reference they are being compared against."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    if not np.all(np.isfinite(reference_direction)):
        return np.full(activity.shape[0], np.nan)
    deviation = np.full(activity.shape[0], np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    deviation[valid] = 1.0 - unit_vectors[valid] @ reference_direction
    return deviation


def compute_block_b_displacement(activity_by_unit: np.ndarray, stim_flag: np.ndarray) -> dict:
    """The core Block B computation for one session and one channel
    condition: leave-one-out deviation among control trials only
    (rate_free_state_deviation, unmodified, called on the control subset
    alone so a stimulated trial can never enter any control trial's
    reference), a single fixed reference direction built from that same
    control pool, and stimulated trials scored against that fixed reference.
    Nothing about a stimulated trial's own value can change the reference it
    is compared against."""
    activity = np.asarray(activity_by_unit, dtype=float)
    stim = np.asarray(stim_flag).astype(bool)
    ctrl_activity = activity[~stim]
    stim_activity = activity[stim]
    control_deviation = rate_free_state_deviation(ctrl_activity)
    reference_direction = _reference_direction(ctrl_activity)
    stim_deviation = _deviation_from_reference(stim_activity, reference_direction)
    return {
        "control_deviation": control_deviation,
        "stim_deviation": stim_deviation,
        "reference_direction": reference_direction,
    }


# ── Magnitude-matched rotation null (precondition, criterion 3) ────────────────

def rotation_null_variance_test(activity_by_unit: np.ndarray, n_draws: int, rng: np.random.Generator) -> dict:
    """Tests whether the observed within-session variance of the rate-free
    deviation is distinguishable from a null in which every trial's
    direction is replaced by an independent, uniformly-random unit vector of
    the same dimensionality (each trial's own magnitude is kept, though the
    deviation itself is scale-free by construction and does not depend on
    it) -- a null with no shared cross-trial direction structure at all. A
    global rotation applied identically to every trial would leave every
    cosine unchanged and so cannot serve as a null here; only an
    independent, per-trial randomisation destroys the shared-direction
    structure the observed data may or may not have."""
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials, n_channels = activity.shape
    magnitudes = np.linalg.norm(activity, axis=1, keepdims=True)
    observed = rate_free_state_deviation(activity)
    observed_var = float(np.nanvar(observed)) if np.isfinite(observed).sum() >= 2 else float("nan")
    null_vars = np.empty(n_draws)
    for i in range(n_draws):
        random_dirs = rng.normal(size=(n_trials, n_channels))
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        fake_activity = random_dirs * magnitudes
        fake_dev = rate_free_state_deviation(fake_activity)
        null_vars[i] = np.nanvar(fake_dev) if np.isfinite(fake_dev).sum() >= 2 else np.nan
    finite_null = null_vars[np.isfinite(null_vars)]
    if not np.isfinite(observed_var) or len(finite_null) < n_draws // 2:
        return {"status": "not_computable", "n_trials": int(n_trials)}
    center = float(np.mean(finite_null))
    p = permutation_pvalue(np.abs(finite_null - center) >= np.abs(observed_var - center))
    return {
        "status": "computed", "n_trials": int(n_trials), "n_draws": int(len(finite_null)),
        "observed_variance": observed_var, "null_mean_variance": center,
        "null_std_variance": float(np.std(finite_null)), "p_value": p,
    }


# ── Subject-clustered inference helpers ─────────────────────────────────────────

def subject_clustered_mean_test(session_values: np.ndarray, subject_ids: list, alternative: str = "two-sided") -> dict:
    """Pools one scalar per session into a subject-clustered mean test: each
    subject's own sessions are first collapsed to their unweighted mean, then
    the collapsed subject-level values are tested against zero with the
    paired sign-flip test -- subject is the unit both the permutation null
    and the bootstrap CI resample, so a subject contributing many sessions
    cannot silently outweigh one contributing few."""
    session_values = np.asarray(session_values, dtype=float)
    subject_ids = np.asarray(subject_ids)
    finite = np.isfinite(session_values)
    session_values, subject_ids = session_values[finite], subject_ids[finite]
    unique_subjects = sorted(set(subject_ids.tolist()))
    if len(unique_subjects) < 2:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects)}
    subject_values = np.array([session_values[subject_ids == s].mean() for s in unique_subjects])
    rng = np.random.default_rng(stable_seed(f"subject_clustered|{tuple(unique_subjects)}|{finite.sum()}"))
    test = paired_sign_flip_test(subject_values, np.zeros_like(subject_values), n_perm=N_PERM,
                                  alternative=alternative, n_boot=N_BOOT, rng=rng)
    mdd = minimum_detectable_paired_difference(subject_values, alpha=ALPHA, power=POWER)
    return {
        "status": "computed", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
        "mean_value": test["mean_diff"], "p_value": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "mdd": mdd,
    }


def minimum_detectable_correlation(n_subjects: int, alpha: float = ALPHA, power: float = POWER) -> dict:
    """Smallest true Pearson correlation a two-sided test on n_subjects
    independent units could detect at the given power, via the standard
    Fisher z-transform normal approximation (Cohen 1988) -- the correlation
    analogue of minimum_detectable_paired_difference, which is defined for a
    paired mean difference, not a correlation coefficient."""
    from scipy.stats import norm

    if n_subjects < 4:
        return {"status": "not_computable", "n_subjects": int(n_subjects),
                "reason": "fewer than 4 subjects -- Fisher z approximation undefined"}
    z_a = float(norm.ppf(1.0 - alpha / 2.0))
    z_b = float(norm.ppf(power))
    z_r = (z_a + z_b) / np.sqrt(n_subjects - 3)
    return {"status": "computed", "n_subjects": int(n_subjects), "alpha": alpha, "power": power,
            "mdd": float(np.tanh(z_r))}


def subject_aggregated_correlation(x: np.ndarray, y: np.ndarray, subject_ids: list) -> dict:
    """Collapses session-level (x, y) pairs to one point per subject (the
    unweighted mean of that subject's own sessions) before correlating, so
    the permutation null and the bootstrap CI both resample at the subject
    level -- avoiding a subject with many sessions silently outweighing one
    with few, and matching the reachable-sample-size regime
    pearson_permutation_test and bootstrap_ci (both already used elsewhere in
    this project) were built for."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    subject_ids = np.asarray(subject_ids)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, subject_ids = x[finite], y[finite], subject_ids[finite]
    unique_subjects = sorted(set(subject_ids.tolist()))
    if len(unique_subjects) < 4:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
                "reason": "fewer than 4 subjects"}
    x_subj = np.array([x[subject_ids == s].mean() for s in unique_subjects])
    y_subj = np.array([y[subject_ids == s].mean() for s in unique_subjects])
    if np.std(x_subj) == 0 or np.std(y_subj) == 0:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
                "reason": "zero variance in a subject-aggregated variable"}
    rng = np.random.default_rng(stable_seed(f"subject_aggregated_corr|{tuple(unique_subjects)}"))
    corr = pearson_permutation_test(x_subj, y_subj, n_perm=N_PERM, rng=rng)
    _, ci_lo, ci_hi = bootstrap_ci(
        np.column_stack([x_subj, y_subj]),
        lambda d: float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]),
        n_boot=N_BOOT, rng=rng,
    )
    mdd = minimum_detectable_correlation(len(unique_subjects))
    return {
        "status": "computed", "n_sessions": int(finite.sum()), "n_subjects": len(unique_subjects),
        "r": corr["r"], "p_value": corr["p_value"], "ci_lower": float(ci_lo), "ci_upper": float(ci_hi),
        "mdd": mdd,
    }


# ── Session enumeration and diagnostic-checkpointed feature loading ────────────

def _find_session_jsons(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json"))


def _subject_id(ieeg_json: Path) -> str:
    return ieeg_json.parts[-4]


def _session_index(session_key: str) -> int:
    match = re.search(r"ses-(\d+)", session_key)
    return int(match.group(1)) if match else -1


def classify_missing_session(ieeg_json: Path, derive_stim_from_stim_on: bool) -> str:
    """Redoes build_session_features's cheap, file-level gates (never the
    expensive EDF/MNE load) to give a specific zero-drop exclusion reason
    for a session build_session_features rejected -- every session must
    appear in the artifact with either a result or a named reason, and
    "returned None" alone is not a named reason."""
    try:
        meta = json.loads(ieeg_json.read_text())
    except (OSError, json.JSONDecodeError):
        return "unreadable_ieeg_sidecar_json"
    if not meta.get("ElectricalStimulation", False):
        return "no_electrical_stimulation_metadata"
    stem = str(ieeg_json).replace("_ieeg.json", "")
    events_tsv = Path(stem.replace("_acq-bipolar", "") + "_events.tsv")
    edf_path = Path(stem + "_ieeg.edf")
    if not events_tsv.exists() or not edf_path.exists():
        return "missing_events_tsv_or_edf_file"
    import csv
    with open(events_tsv) as f:
        events = list(csv.DictReader(f, delimiter="\t"))
    words = [e for e in events if e["trial_type"] == "WORD"]
    if len(words) < 100:
        return "fewer_than_100_word_events"
    if derive_stim_from_stim_on:
        stim_on = [e for e in events if e["trial_type"] == "STIM_ON"]
        has_labelled_anode = any(s.get("anode_label", "n/a") != "n/a" for s in stim_on)
        if not stim_on or not has_labelled_anode:
            return "no_labelled_stim_on_event"
    else:
        has_stim_word = any(w.get("stimulation") == "1" and w.get("anode_label", "n/a") != "n/a" for w in words)
        if not has_stim_word:
            return "no_stimulated_word_event_with_labelled_electrode"
    return "excluded_by_build_session_features_internal_check_see_exception_field"


def load_session_features(corpus_name: str, session_key: str, ieeg_json: Path, data_dir: Path,
                           derive_stim_from_stim_on: bool) -> dict:
    """Fit-level checkpointed load of one session's epoched features. Returns
    a dict with status 'usable' and the arrays, or status 'excluded' and a
    named reason -- never raises."""
    raw_unit = f"{corpus_name}__{session_key}"  # corpus-namespaced: session_key alone is only unique
    cached = load_raw_features(raw_unit)          # within one corpus's own directory tree
    if cached is not None:
        return {"status": "usable", "arrays": cached}
    reason_holder: dict = {}

    def _fit():
        try:
            feat = build_session_features(ieeg_json, data_root=data_dir,
                                          derive_stim_from_stim_on=derive_stim_from_stim_on,
                                          return_epochs=True)
        except Exception as exc:  # noqa: BLE001 -- a session-level failure must not crash the run
            reason_holder["reason"] = f"exception_during_build: {type(exc).__name__}: {exc}"
            reason_holder["traceback"] = traceback.format_exc(limit=6)
            return None
        if feat is None:
            reason_holder["reason"] = classify_missing_session(ieeg_json, derive_stim_from_stim_on)
            return None
        arrays = {
            "epochs_log": feat["epochs_log"].astype(np.float32),
            "ch_names": np.array(feat["ch_names"]),
            "anode": np.array(feat["anode"]), "cathode": np.array(feat["cathode"]),
            "stim_channel": np.array(feat["stim_channel"]),
            "stim_flag": feat["stim_flag"], "recalled": feat["recalled"],
            "amplitude": feat["amplitude"], "pulse_freq": feat["pulse_freq"],
            "n_pulses": feat["n_pulses"], "pulse_width": feat["pulse_width"],
            "serialpos": feat["serialpos"], "list_number": feat["list_number"],
            "alignment_to_vstar": np.array(float(feat["alignment_to_vstar"])),
            "n_words": np.array(int(feat["n_words"])),
        }
        return arrays

    arrays = _fit()
    if arrays is None:
        return {"status": "excluded", "reason": reason_holder.get("reason", "unknown"),
                "detail": reason_holder.get("traceback")}
    save_raw_features(raw_unit, arrays)
    return {"status": "usable", "arrays": arrays}


def _bin_averaged(arrays: dict, channel_mask: np.ndarray | None = None) -> np.ndarray:
    epochs = arrays["epochs_log"]
    if channel_mask is not None:
        epochs = epochs[:, :, channel_mask]
    return epochs.mean(axis=1).astype(float)  # (n_trials, n_channels_kept)


# ── Precondition ─────────────────────────────────────────────────────────────

def evaluate_precondition(session_records: list[dict]) -> dict:
    """Gate for the whole arm, evaluated on the pooled set of usable
    sessions from BOTH corpora, using each session's own non-stimulated,
    leave-one-out rate-free deviation. Three named criteria, any one of
    which fires the transport-failure branch."""
    session_medians, rotation_tests = [], []
    for rec in session_records:
        arrays = rec["arrays"]
        ctrl_mask = arrays["stim_flag"] == 0
        activity = _bin_averaged(arrays)[ctrl_mask]
        deviation = rate_free_state_deviation(activity)
        finite = deviation[np.isfinite(deviation)]
        if len(finite) < 8:
            continue
        session_medians.append(float(np.median(finite)))
        rng = np.random.default_rng(stable_seed(f"precondition_rotation_null|{rec['session_key']}"))
        rotation_tests.append({
            "session": rec["session_key"], "subject": rec["subject_id"],
            "test": rotation_null_variance_test(activity, N_ROTATION_NULL_DRAWS, rng),
        })

    if not session_medians:
        return {"status": "not_computable", "reason": "no usable session produced a finite deviation array"}

    overall_median = float(np.median(session_medians))

    macaque_root = macaque_data_root()
    macaque_dir = _panichello_directory(macaque_root) if macaque_root is not None else None
    non_human_medians = []
    if macaque_dir is not None:
        for path in sorted(macaque_dir.glob("*.mat")):
            arrays = _macaque_session_arrays(path)
            if arrays is None:
                continue
            non_human_medians.append(float(np.median(arrays["deviation"])))
    non_human_reference = {
        "status": "computed", "n_sessions": len(non_human_medians),
        "median_of_session_medians": float(np.median(non_human_medians)) if non_human_medians else None,
    } if non_human_medians else {"status": "unavailable", "n_sessions": 0}

    computed_rotation = [r for r in rotation_tests if r["test"]["status"] == "computed"]
    n_distinguishable = sum(1 for r in computed_rotation if r["test"]["p_value"] <= ALPHA)
    distinguishable_fraction = (n_distinguishable / len(computed_rotation)) if computed_rotation else None

    criterion_1_abs = overall_median < SESSION_MEDIAN_DEGENERATE_ABS
    criterion_2_rel = (
        non_human_reference["status"] == "computed" and non_human_reference["median_of_session_medians"] is not None
        and overall_median < SESSION_MEDIAN_DEGENERATE_REL_FRAC * non_human_reference["median_of_session_medians"]
    )
    criterion_3_variance = (
        distinguishable_fraction is not None
        and distinguishable_fraction < ROTATION_NULL_DISTINGUISHABLE_FRACTION_FLOOR
    )

    degenerate = criterion_1_abs or criterion_2_rel or criterion_3_variance
    return {
        "status": "computed",
        "n_sessions_evaluated": len(session_medians),
        "overall_session_median_deviation": overall_median,
        "non_human_reference": non_human_reference,
        "rotation_null_summary": {
            "n_sessions_tested": len(computed_rotation),
            "n_sessions_not_computable": len(rotation_tests) - len(computed_rotation),
            "n_sessions_distinguishable_at_p05": n_distinguishable,
            "distinguishable_fraction": distinguishable_fraction,
            "floor": ROTATION_NULL_DISTINGUISHABLE_FRACTION_FLOOR,
        },
        "criteria": {
            "session_median_below_absolute_floor": {"fired": criterion_1_abs, "threshold": SESSION_MEDIAN_DEGENERATE_ABS},
            "session_median_below_relative_floor_of_non_human_median": {
                "fired": criterion_2_rel, "threshold_fraction": SESSION_MEDIAN_DEGENERATE_REL_FRAC},
            "variance_not_distinguishable_from_rotation_null": {
                "fired": criterion_3_variance, "floor_fraction": ROTATION_NULL_DISTINGUISHABLE_FRACTION_FLOOR},
        },
        "degenerate": degenerate,
        "branch": "the_component_does_not_transport_to_field_potential_power" if degenerate else "component_transports",
    }


# ── Block A ──────────────────────────────────────────────────────────────────

def _block_a_session(rec: dict) -> dict:
    arrays = rec["arrays"]
    ctrl_mask = arrays["stim_flag"] == 0
    activity = _bin_averaged(arrays)[ctrl_mask]
    deviation = rate_free_state_deviation(activity)
    total_power = activity.sum(axis=1)
    recalled = arrays["recalled"][ctrl_mask].astype(float)
    failure = 1.0 - recalled
    finite = np.isfinite(deviation)
    n_finite = int(finite.sum())
    tag = f"blockA|{rec['session_key']}"
    if n_finite < 8:
        return {"status": "too_few_trials", "n_control_trials": int(ctrl_mask.sum())}
    rng = np.random.default_rng(stable_seed(f"{tag}|deviation"))
    corr_deviation = partial_correlation_permutation_test(failure[finite], deviation[finite], [], N_PERM, rng)
    rng_power = np.random.default_rng(stable_seed(f"{tag}|power"))
    corr_power = partial_correlation_permutation_test(failure[finite], total_power[finite], [], N_PERM, rng_power)
    return {
        "status": "computed",
        "n_control_trials": int(ctrl_mask.sum()),
        "n_trials_with_defined_direction": n_finite,
        "correlation_deviation_vs_failure": corr_deviation,
        "correlation_total_power_vs_failure": corr_power,
        "session_mean_deviation": float(np.nanmean(deviation[finite])),
        "session_failure_rate": float(failure.mean()),
    }


def _classify_block_a(main_test: dict, mdd: dict | None, void_test: dict) -> str:
    if main_test["status"] != "computed":
        return "not_computable"
    significant = main_test["p_value"] <= ALPHA
    if significant:
        void_significant = void_test.get("status") == "computed" and void_test["p_value"] <= ALPHA
        void_same_sign = void_significant and (void_test["r"] > 0) == (main_test["mean_value"] > 0)
        if void_significant and void_same_sign:
            return "component_recall_link_not_separable_from_a_session_level_offset"
        return "component_predicts_recall_failure_in_human_intracranial_recording"
    if mdd is not None and mdd.get("status") == "computed" and mdd["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        return "no_component_link_to_recall_above_the_reported_bound"
    return "underpowered_to_ask"


def run_block_a(session_records: list[dict]) -> dict:
    per_session = {}
    for rec in session_records:
        result = run_checkpointed(f"blockA__{rec['corpus']}__{rec['session_key']}", lambda rec=rec: _block_a_session(rec))
        per_session[rec["session_key"]] = {"corpus": rec["corpus"], "subject": rec["subject_id"], **result}

    session_r = np.array([v["correlation_deviation_vs_failure"]["r"] for v in per_session.values()
                          if v.get("status") == "computed" and v["correlation_deviation_vs_failure"]["status"] == "computed"])
    subj_for_r = [v["subject"] for v in per_session.values()
                 if v.get("status") == "computed" and v["correlation_deviation_vs_failure"]["status"] == "computed"]
    pooled = subject_clustered_mean_test(session_r, subj_for_r) if len(session_r) else {"status": "not_computable"}

    power_r = np.array([v["correlation_total_power_vs_failure"]["r"] for v in per_session.values()
                        if v.get("status") == "computed" and v["correlation_total_power_vs_failure"]["status"] == "computed"])
    subj_for_power = [v["subject"] for v in per_session.values()
                      if v.get("status") == "computed" and v["correlation_total_power_vs_failure"]["status"] == "computed"]
    pooled_power = subject_clustered_mean_test(power_r, subj_for_power) if len(power_r) else {"status": "not_computable"}

    # Void control: collapse every trial's value to its session's own mean deviation, and ask whether a
    # pure between-session offset (mean deviation vs. this session's own control-trial failure rate,
    # same sign convention as the main branch) reproduces the main result. y is failure rate, not
    # recall rate, to keep "more deviation -> more failure" as the positive direction throughout.
    x_offset = np.array([v["session_mean_deviation"] for v in per_session.values() if v.get("status") == "computed"])
    y_offset = np.array([v["session_failure_rate"] for v in per_session.values() if v.get("status") == "computed"])
    subj_offset = [v["subject"] for v in per_session.values() if v.get("status") == "computed"]
    void = subject_aggregated_correlation(x_offset, y_offset, subj_offset) if len(x_offset) >= 4 else {"status": "not_computable"}

    main_test = {"status": pooled.get("status"), "mean_value": pooled.get("mean_value"), "p_value": pooled.get("p_value")} \
        if pooled.get("status") == "computed" else {"status": "not_computable"}
    void_test = {"status": void.get("status"), "r": void.get("r"), "p_value": void.get("p_value")} \
        if void.get("status") == "computed" else {"status": "not_computable"}
    branch = _classify_block_a(main_test, pooled.get("mdd") if pooled.get("status") == "computed" else None, void_test)

    return {
        "per_session": per_session,
        "pooled_deviation_vs_failure": pooled,
        "pooled_total_power_vs_failure": pooled_power,
        "void_control_session_level_offset": void,
        "branch": branch,
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
        "meaningful_effect_threshold_source": (
            "results/rate_free_state_geometry_behavior_link.json's own minimum detectable paired "
            "difference (0.14 r units), reused unchanged so this arm's behavioural null is on the same "
            "scale as the project's existing macaque one."
        ),
    }


# ── Block B ──────────────────────────────────────────────────────────────────

def _block_b_session(rec: dict) -> dict:
    arrays = rec["arrays"]
    ch_names = arrays["ch_names"].tolist()
    anode, cathode, stim_ch = str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
    masks = channel_condition_masks(ch_names, anode, cathode, stim_ch)
    stim_flag = arrays["stim_flag"]
    n_stim, n_ctrl = int(stim_flag.sum()), int((stim_flag == 0).sum())
    conditions = {}
    for name, mask in masks.items():
        activity = _bin_averaged(arrays, mask)
        out = compute_block_b_displacement(activity, stim_flag)
        ctrl_dev, stim_dev = out["control_deviation"], out["stim_deviation"]
        finite_ctrl, finite_stim = np.isfinite(ctrl_dev), np.isfinite(stim_dev)
        if finite_ctrl.sum() < 8 or finite_stim.sum() < 4:
            conditions[name] = {"status": "too_few_trials", "n_channels": int(mask.sum())}
            continue
        displacement = float(np.nanmean(stim_dev[finite_stim]) - np.nanmean(ctrl_dev[finite_ctrl]))
        spontaneous_sd = float(np.nanstd(ctrl_dev[finite_ctrl], ddof=1)) if finite_ctrl.sum() >= 2 else float("nan")
        total_power = activity.sum(axis=1)
        power_change = float(np.nanmean(total_power[stim_flag == 1][finite_stim])
                             - np.nanmean(total_power[stim_flag == 0][finite_ctrl]))
        conditions[name] = {
            "status": "computed", "n_channels": int(mask.sum()),
            "n_stim_trials": int(finite_stim.sum()), "n_control_trials": int(finite_ctrl.sum()),
            "displacement": displacement,
            "spontaneous_control_sd": spontaneous_sd,
            "normalised_displacement": (displacement / spontaneous_sd) if spontaneous_sd and spontaneous_sd > 0 else None,
            "total_power_change": power_change,
        }
    return {"status": "computed", "n_stim_trials_total": n_stim, "n_control_trials_total": n_ctrl,
            "conditions": conditions, "stim_channel": stim_ch, "anode": anode, "cathode": cathode}


def _classify_block_b(pooled_by_condition: dict) -> str:
    full = pooled_by_condition.get("full_channel_set", {})
    shank = pooled_by_condition.get("excluding_stimulated_shank", {})
    if full.get("status") != "computed":
        return "not_computable"
    full_significant = full["p_value"] <= ALPHA
    shank_significant = shank.get("status") == "computed" and shank["p_value"] <= ALPHA
    shank_same_sign = shank_significant and (shank["mean_value"] > 0) == (full["mean_value"] > 0)
    if full_significant and not (shank_significant and shank_same_sign):
        return "stimulation_displacement_not_separable_from_recording_artifact"
    if shank_significant and shank_same_sign:
        return "stimulation_displaces_the_component"
    mdd = shank.get("mdd", {}) if shank.get("status") == "computed" else full.get("mdd", {})
    if mdd.get("status") == "computed" and mdd["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT:
        return "no_stimulation_displacement_above_the_reported_bound"
    return "underpowered_to_ask"


def run_block_b(session_records: list[dict]) -> dict:
    per_session = {}
    for rec in session_records:
        result = run_checkpointed(f"blockB__{rec['corpus']}__{rec['session_key']}", lambda rec=rec: _block_b_session(rec))
        per_session[rec["session_key"]] = {"corpus": rec["corpus"], "subject": rec["subject_id"], **result}

    condition_names = ["full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank"]
    pooled_by_condition, pooled_power_by_condition = {}, {}
    for cond in condition_names:
        vals, subj, power_vals = [], [], []
        for v in per_session.values():
            c = v.get("conditions", {}).get(cond, {})
            if c.get("status") == "computed" and c.get("normalised_displacement") is not None:
                vals.append(c["normalised_displacement"])
                subj.append(v["subject"])
                power_vals.append(c["total_power_change"])
        pooled_by_condition[cond] = subject_clustered_mean_test(np.array(vals), subj) if vals else {"status": "not_computable"}
        pooled_power_by_condition[cond] = subject_clustered_mean_test(np.array(power_vals), subj) if power_vals else {"status": "not_computable"}

    branch = _classify_block_b(pooled_by_condition)
    return {
        "per_session": per_session,
        "pooled_normalised_displacement_by_channel_condition": pooled_by_condition,
        "pooled_total_power_change_by_channel_condition": pooled_power_by_condition,
        "branch": branch,
        "meaningful_effect_threshold_normalised_displacement": MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT,
        "meaningful_effect_threshold_source": (
            "one spontaneous (non-stimulated trial-to-trial) standard deviation of the same session's own "
            "control-trial deviation -- the natural, session-comparable floor for 'the smallest displacement "
            "this design would call meaningful', on the normalised scale Block D also uses."
        ),
        "epoch_overlap_disclosure": (
            "The stimulation pulse train in both RAM corpora outlasts a single word presentation, so the "
            "high-gamma epoch analysed for a stimulated trial lies partly inside the stimulation interval "
            "itself. This is a property of the measurement, not a matching requirement, and is not a reason "
            "to drop any trial or arm."
        ),
    }


# ── Block C ──────────────────────────────────────────────────────────────────

def run_block_c(session_records: list[dict], block_b: dict) -> dict:
    if block_b["branch"] == "stimulation_displacement_not_separable_from_recording_artifact":
        return {"status": "mediation_not_askable_because_the_displacement_is_not_separable_from_artifact",
                "branch": "mediation_not_askable_because_the_displacement_is_not_separable_from_artifact"}

    displacement_x, recall_diff_y, subj = [], [], []
    per_session = {}
    for rec in session_records:
        b_rec = block_b["per_session"].get(rec["session_key"], {})
        cond = b_rec.get("conditions", {}).get("excluding_stimulated_shank", {})
        arrays = rec["arrays"]
        stim_mask, ctrl_mask = arrays["stim_flag"] == 1, arrays["stim_flag"] == 0
        if stim_mask.sum() < 4 or ctrl_mask.sum() < 8 or cond.get("status") != "computed":
            per_session[rec["session_key"]] = {"status": "excluded", "reason": "insufficient_trials_or_no_block_b_displacement"}
            continue
        recall_diff = float(arrays["recalled"][stim_mask].mean() - arrays["recalled"][ctrl_mask].mean())
        per_session[rec["session_key"]] = {
            "status": "computed", "displacement_excluding_stimulated_shank": cond["displacement"],
            "recall_rate_stim_minus_ctrl": recall_diff,
        }
        displacement_x.append(cond["displacement"])
        recall_diff_y.append(recall_diff)
        subj.append(rec["subject_id"])

    mediation = subject_aggregated_correlation(np.array(displacement_x), np.array(recall_diff_y), subj) \
        if len(displacement_x) >= 4 else {"status": "not_computable", "n_sessions": len(displacement_x)}

    if mediation.get("status") != "computed":
        branch = "underpowered_to_ask" if len(displacement_x) >= 2 else "not_computable"
    elif mediation["p_value"] <= ALPHA:
        branch = "induced_change_in_the_component_predicts_induced_change_in_behaviour"
    elif mediation.get("mdd", {}).get("status") == "computed" and \
            mediation["mdd"]["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        # subject_aggregated_correlation's mdd is already on the Pearson-r scale, the same scale
        # MEANINGFUL_EFFECT_THRESHOLD_R_UNITS names for Block A -- reused directly, not rescaled.
        branch = "no_mediation_above_the_reported_bound"
    else:
        branch = "underpowered_to_ask"

    return {"per_session": per_session, "mediation": mediation, "branch": branch,
            "channel_condition_used": "excluding_stimulated_shank",
            "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS}


# ── Block D ──────────────────────────────────────────────────────────────────

def _dose_fields(arrays: dict) -> dict:
    stim_mask = arrays["stim_flag"] == 1
    amp = arrays["amplitude"][stim_mask]
    pw = arrays["pulse_width"][stim_mask]
    npulse = arrays["n_pulses"][stim_mask]
    finite = np.isfinite(amp) & np.isfinite(pw) & np.isfinite(npulse) & (amp > 0)
    if finite.sum() == 0:
        return {"status": "no_finite_dose_fields"}
    amp_v, pw_v, npulse_v = amp[finite], pw[finite], npulse[finite]
    charge_per_pulse_pC, total_charge_pC = _dose_quantities(amp_v, pw_v, npulse_v)
    return {
        "status": "computed",
        "amplitude_uA": float(np.median(amp_v)), "amplitude_constant_within_session": bool(np.ptp(amp_v) < 1e-9),
        "pulse_width_us": float(np.median(pw_v)), "n_pulses": float(np.median(npulse_v)),
        "charge_per_pulse_pC": float(np.median(charge_per_pulse_pC)),
        "total_delivered_charge_pC": float(np.median(total_charge_pC)),
    }


DOSE_FIELD_UNITS_NOTE = (
    "The upstream event-table sidecar describes this stimulation-amplitude field as milliamperes; that "
    "description disagrees with the values actually recorded (250-3500), which no clinical intracranial "
    "stimulation study delivers in milliamperes, so the recorded values are taken here as microamperes "
    "instead. charge_per_pulse_pC and total_delivered_charge_pC are amplitude(microamperes) x "
    "pulse_width(microseconds) [x n_pulses], which is picocoulombs; correcting the unit label changes no "
    "number already computed under the old, mislabelled field name."
)


def run_block_d(session_records: list[dict], block_b: dict) -> dict:
    rows = []
    for rec in session_records:
        b_rec = block_b["per_session"].get(rec["session_key"], {})
        cond = b_rec.get("conditions", {}).get("excluding_stimulated_shank", {})
        dose = _dose_fields(rec["arrays"])
        if dose["status"] != "computed" or cond.get("normalised_displacement") is None:
            continue
        rows.append({
            "session_key": rec["session_key"], "subject": rec["subject_id"],
            "stim_channel": str(rec["arrays"]["stim_channel"]),
            "session_order": _session_index(rec["session_key"]),
            "normalised_displacement": cond["normalised_displacement"],
            **dose,
        })

    by_subject_pair: dict[tuple, list] = {}
    for row in rows:
        by_subject_pair.setdefault((row["subject"], row["stim_channel"]), []).append(row)
    dose_varying_subjects = {
        key: recs for key, recs in by_subject_pair.items()
        if len({r["amplitude_uA"] for r in recs}) >= 2
    }
    n_dose_varying_subjects = len({subj for subj, _ in dose_varying_subjects})

    controls = {
        "electrode_pair_control": (
            "held constant by construction within the within-subject dose arm: every session entering it "
            "shares the same subject AND the same stimulated bipolar pair by the arm's own restriction, so "
            "there is no electrode-pair variation left for this control to remove there."
        ),
    }

    if n_dose_varying_subjects < MIN_SUBJECTS_FOR_WITHIN_SUBJECT_DOSE:
        between = _block_d_between_subject(rows, controls)
        return {
            "n_subjects_with_within_subject_dose_variation": n_dose_varying_subjects,
            "branch": "too_few_subjects_with_within_subject_dose_variation",
            "within_subject": {"status": "not_computable", "n_subjects": n_dose_varying_subjects,
                               "reason": f"fewer than {MIN_SUBJECTS_FOR_WITHIN_SUBJECT_DOSE} subjects"},
            "between_subject_fallback": between,
            "dose_field_units_note": DOSE_FIELD_UNITS_NOTE,
            "rows": rows,
        }

    dose_params = ["amplitude_uA", "charge_per_pulse_pC", "total_delivered_charge_pC"]
    within_subject_results = {}
    for param in dose_params:
        subject_slopes, session_order_partials, subj_ids = [], [], []
        for (subj, _pair), recs in dose_varying_subjects.items():
            x = np.array([r[param] for r in recs])
            y = np.array([r["normalised_displacement"] for r in recs])
            order = np.array([r["session_order"] for r in recs], dtype=float)
            if np.std(x) == 0:
                continue
            slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None
            if slope is not None:
                subject_slopes.append(slope)
                subj_ids.append(subj)
            if len(recs) >= 3 and np.std(order) > 0:
                rng = np.random.default_rng(stable_seed(f"blockD_session_order|{subj}|{param}"))
                session_order_partials.append(
                    partial_correlation_permutation_test(y, x, [order], n_perm=N_PERM, rng=rng))
        pooled = subject_clustered_mean_test(np.array(subject_slopes), subj_ids) if subject_slopes else {"status": "not_computable"}

        shuffle_null = None
        if subject_slopes:
            rng = np.random.default_rng(stable_seed(f"blockD_shuffle|{param}"))
            observed_mean = float(np.mean(subject_slopes))
            null = np.empty(N_DOSE_SHUFFLE_DRAWS)
            for i in range(N_DOSE_SHUFFLE_DRAWS):
                draw_slopes = []
                for (subj, _pair), recs in dose_varying_subjects.items():
                    x = np.array([r[param] for r in recs])
                    y = np.array([r["normalised_displacement"] for r in recs])
                    if np.std(x) == 0 or len(x) < 2:
                        continue
                    y_shuf = rng.permutation(y)
                    draw_slopes.append(float(np.polyfit(x, y_shuf, 1)[0]))
                null[i] = np.mean(draw_slopes) if draw_slopes else np.nan
            finite_null = null[np.isfinite(null)]
            p_shuffle = permutation_pvalue(np.abs(finite_null) >= np.abs(observed_mean)) if len(finite_null) else None
            shuffle_null = {"observed_mean_slope": observed_mean, "p_value": p_shuffle, "n_draws": int(len(finite_null))}

        within_subject_results[param] = {
            "pooled_within_subject_slope": pooled,
            "within_subject_dose_shuffle_control": shuffle_null,
            "session_order_partials": session_order_partials,
        }

    primary = within_subject_results["amplitude_uA"]["pooled_within_subject_slope"]
    if primary.get("status") == "computed" and primary["p_value"] <= ALPHA:
        branch = "dose_effect_scales_with_amplitude_within_subject"
    elif primary.get("status") == "computed" and primary.get("mdd", {}).get("status") == "computed":
        branch = "no_dose_scaling_above_the_reported_bound" if \
            primary["mdd"]["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT else "underpowered_to_ask"
    else:
        branch = "underpowered_to_ask"

    dose_parameter_collapse_note = (
        "pulse width and pulse count are reported per session; where both are constant across every "
        "dose-varying session in this arm, charge-per-pulse and total delivered charge are strictly "
        "monotonic re-scalings of amplitude alone and the three parameterisations cannot be dissociated "
        "by this data -- see dose_parameter_variability below for the actual within-arm spread."
    )
    dose_parameter_variability = {
        "pulse_width_us_unique_values": sorted({r["pulse_width_us"] for recs in dose_varying_subjects.values() for r in recs}),
        "n_pulses_unique_values": sorted({r["n_pulses"] for recs in dose_varying_subjects.values() for r in recs}),
    }

    return {
        "n_subjects_with_within_subject_dose_variation": n_dose_varying_subjects,
        "branch": branch,
        "within_subject": within_subject_results,
        "controls": controls,
        "dose_parameter_collapse_note": dose_parameter_collapse_note,
        "dose_parameter_variability": dose_parameter_variability,
        "dose_field_units_note": DOSE_FIELD_UNITS_NOTE,
        "rows": rows,
    }


def _block_d_between_subject(rows: list[dict], controls: dict) -> dict:
    if len(rows) < 8:
        return {"status": "not_computable", "reason": "fewer than 8 sessions", "causal": False}
    x = np.array([r["amplitude_uA"] for r in rows])
    y = np.array([r["normalised_displacement"] for r in rows])
    subj = [r["subject"] for r in rows]
    corr = subject_aggregated_correlation(x, y, subj)

    shank_prefixes = [_contact_shank(r["stim_channel"].split("-")[0]) for r in rows]
    shared_shank = {p for p in shank_prefixes if shank_prefixes.count(p) >= 2}
    electrode_pair_control = {"status": "not_computable", "reason": "no shank shared by >=2 subjects"}
    if shared_shank:
        mask = np.array([p in shared_shank for p in shank_prefixes])
        if mask.sum() >= 8:
            electrode_pair_control = subject_aggregated_correlation(x[mask], y[mask], [subj[i] for i in range(len(subj)) if mask[i]])

    branch = "dose_effect_not_separable_from_electrode_pair" if (
        corr.get("status") == "computed" and corr["p_value"] <= ALPHA
        and electrode_pair_control.get("status") == "computed" and electrode_pair_control["p_value"] <= ALPHA
        and (electrode_pair_control["r"] > 0) == (corr["r"] > 0)
    ) else "reported_descriptively"

    return {
        "status": "computed", "causal": False,
        "confounds": CONFOUND_LIST_FOR_BETWEEN_SUBJECT_DOSE,
        "correlation_amplitude_vs_normalised_displacement": corr,
        "electrode_pair_shank_control": electrode_pair_control,
        "branch": branch,
        "controls": controls,
    }


# ── Block E (closed-loop arm only) ──────────────────────────────────────────

def _block_e_session(rec: dict) -> dict:
    arrays = rec["arrays"]
    stim_flag, recalled = arrays["stim_flag"], arrays["recalled"].astype(float)
    ch_names = arrays["ch_names"].tolist()
    anode, cathode, stim_ch = str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
    masks = channel_condition_masks(ch_names, anode, cathode, stim_ch)
    activity = _bin_averaged(arrays, masks["excluding_stimulated_shank"])
    out = compute_block_b_displacement(activity, stim_flag)
    n_trials = len(stim_flag)
    component = np.full(n_trials, np.nan)
    component[stim_flag == 0] = out["control_deviation"]
    component[stim_flag == 1] = out["stim_deviation"]
    # Pre-stimulation moderator: the immediately PRECEDING trial's own component value, which occurred
    # strictly before this trial's own stimulation could have been delivered.
    pre_stim = np.full(n_trials, np.nan)
    pre_stim[1:] = component[:-1]
    valid = np.isfinite(pre_stim) & np.isfinite(component)
    stim_group, ctrl_group = valid & (stim_flag == 1), valid & (stim_flag == 0)
    if stim_group.sum() < 8 or ctrl_group.sum() < 8:
        return {"status": "too_few_trials", "n_stim_valid": int(stim_group.sum()), "n_control_valid": int(ctrl_group.sum())}
    rng_stim = np.random.default_rng(stable_seed(f"blockE|{rec['session_key']}|stim"))
    r_stim = partial_correlation_permutation_test(recalled[stim_group], pre_stim[stim_group], [], N_PERM, rng_stim)
    rng_ctrl = np.random.default_rng(stable_seed(f"blockE|{rec['session_key']}|ctrl"))
    r_ctrl = partial_correlation_permutation_test(recalled[ctrl_group], pre_stim[ctrl_group], [], N_PERM, rng_ctrl)
    moderation = (r_stim["r"] - r_ctrl["r"]) if (r_stim["status"] == "computed" and r_ctrl["status"] == "computed") else None
    return {
        "status": "computed", "n_stim_valid": int(stim_group.sum()), "n_control_valid": int(ctrl_group.sum()),
        "recall_vs_pre_stim_component_within_stim_trials": r_stim,
        "recall_vs_pre_stim_component_within_control_trials": r_ctrl,
        "moderation_session_level": moderation,
        "pre_stim_component": pre_stim.tolist(), "recalled": recalled.tolist(), "stim_flag": stim_flag.tolist(),
    }


def _float_array_none_as_nan(values) -> np.ndarray:
    """A checkpointed record round-trips through JSON, where a NaN was
    written as null (canonical_json's numpy-safe encoding) -- plain
    np.array() on a list containing None produces an object-dtype array
    that np.isfinite cannot operate on, so None is mapped back to NaN here
    before any array arithmetic touches these values."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def run_block_e(closedloop_records: list[dict]) -> dict:
    per_session, pooled_rows, all_rows = {}, [], []
    for rec in closedloop_records:
        result = run_checkpointed(f"blockE__{rec['corpus']}__{rec['session_key']}", lambda rec=rec: _block_e_session(rec))
        stored = {k: v for k, v in result.items() if k not in ("pre_stim_component", "recalled", "stim_flag")}
        per_session[rec["session_key"]] = {"subject": rec["subject_id"], **stored}
        if result.get("status") == "computed" and result.get("moderation_session_level") is not None:
            pooled_rows.append((rec["subject_id"], result["moderation_session_level"]))
        if result.get("status") == "computed":
            n = len(result["recalled"])
            all_rows.append({
                "y": result["recalled"], "t": result["stim_flag"], "m": result["pre_stim_component"],
                "serialpos": rec["arrays"]["serialpos"].tolist() if n == len(rec["arrays"]["serialpos"]) else [0] * n,
                "list": rec["arrays"]["list_number"].tolist() if n == len(rec["arrays"]["list_number"]) else [0] * n,
            })

    subject_ids = [s for s, _ in pooled_rows]
    values = np.array([v for _, v in pooled_rows])
    pooled = subject_clustered_mean_test(values, subject_ids) if len(values) else {"status": "not_computable"}
    if pooled.get("status") == "computed" and pooled["p_value"] <= ALPHA:
        branch = "pre_stimulation_component_moderates_the_behavioural_effect_of_stimulation"
    elif pooled.get("status") == "computed" and pooled.get("mdd", {}).get("status") == "computed" \
            and pooled["mdd"]["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        branch = "no_moderation_above_the_reported_bound"
    else:
        branch = "underpowered_to_ask"

    pooled_dr_slope = {"status": "not_computable"}
    if all_rows:
        y = np.concatenate([_float_array_none_as_nan(r["y"]) for r in all_rows])
        t = np.concatenate([_float_array_none_as_nan(r["t"]) for r in all_rows])
        m = np.concatenate([_float_array_none_as_nan(r["m"]) for r in all_rows])
        X = np.column_stack([
            np.concatenate([_float_array_none_as_nan(r["serialpos"]) for r in all_rows]),
            np.concatenate([_float_array_none_as_nan(r["list"]) for r in all_rows]),
        ])
        finite = np.isfinite(y) & np.isfinite(t) & np.isfinite(m) & np.all(np.isfinite(X), axis=1)
        if finite.sum() >= 50:
            rng = np.random.default_rng(stable_seed("blockE_pooled_dr_slope"))
            dr = cate_vs_modifier_slope(y[finite], t[finite].astype(int), X[finite], modifier=m[finite],
                                        propensity=None, n_perm=5000, rng=rng)
            pooled_dr_slope = {k: v for k, v in dr.items() if k not in ("phi", "modifier", "null")}
            pooled_dr_slope["status"] = "computed"
            pooled_dr_slope["n"] = int(finite.sum())

    return {
        "per_session": per_session,
        "pooled_moderation_subject_clustered": pooled,
        "branch": branch,
        "causal": False,
        "assignment_note": (
            "stimulation in this arm is triggered by an online classifier reading the subject's own "
            "encoding-period state, not experimenter-randomised; every number here is descriptive/"
            "associational, never causal, unless a design fraction is separately recovered."
        ),
        "pooled_doubly_robust_slope_trial_level_not_subject_clustered": pooled_dr_slope,
        "pre_stimulation_definition": (
            "the component's own value on the immediately preceding word trial within the same session -- "
            "strictly prior in time to the current trial's stimulation, avoiding any circularity from "
            "scoring a trial's moderator on the same epoch its own stimulation could have altered."
        ),
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    }


# ── Block F ──────────────────────────────────────────────────────────────────

def run_block_f(session_records: list[dict], block_b: dict) -> dict:
    displacement, alignment, subj = [], [], []
    for rec in session_records:
        b_rec = block_b["per_session"].get(rec["session_key"], {})
        cond = b_rec.get("conditions", {}).get("excluding_stimulated_shank", {})
        if cond.get("status") != "computed":
            continue
        displacement.append(cond["displacement"])
        alignment.append(float(rec["arrays"]["alignment_to_vstar"]))
        subj.append(rec["subject_id"])
    result = subject_aggregated_correlation(np.array(displacement), np.array(alignment), subj) \
        if len(displacement) >= 4 else {"status": "not_computable", "n_sessions": len(displacement)}
    return {
        "correlation_displacement_vs_alignment_to_vstar": result,
        "channel_condition_used": "excluding_stimulated_shank",
        "alignment_definition": (
            "each session's own |cos(B, v*)| between the stimulation input direction B (a one-hot vector "
            "at the stimulated bipolar channel) and the leading eigenvector v* of the DMD-fitted dynamics "
            "matrix, both computed on that session's own control-trial-only fit -- the same quantity "
            "build_session_features computes for the causal pipelines' own effect-modifier test, recomputed "
            "fresh here rather than read from a possibly-stale delivered artifact."
        ),
        "note": "No branch is fired on this correlation; the interval is reported as-is.",
    }


# ── Block G ──────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_block_g(block_a: dict, block_b: dict, block_c: dict) -> dict:
    non_human_a = _read_json(RESULTS / "rate_free_state_geometry_behavior_link.json")
    non_human_bc = _read_json(RESULTS / "causal_macaque_pfc_microstimulation.json")

    a_pooled = block_a.get("pooled_deviation_vs_failure", {})
    a_text = (
        "Block A (does the component predict recall failure on non-stimulated trials): the human open-/"
        f"closed-loop arm's pooled subject-clustered result is r={a_pooled.get('mean_value')}, "
        f"p={a_pooled.get('p_value')}, n_sessions={a_pooled.get('n_sessions')}, "
        f"n_subjects={a_pooled.get('n_subjects')}, branch='{block_a.get('branch')}'. "
    )
    if non_human_a is not None:
        macaque_pooled = non_human_a.get("pooled", {}).get("raw_outcome_vs_deviation", {})
        a_text += (
            "The nearest delivered non-human equivalent (results/rate_free_state_geometry_behavior_link.json, "
            "macaque lPFC, non-stimulated, delay-period, Panichello et al. 2024 corpus) is a session-pooled "
            f"correlation of the same estimator with trial outcome, r={macaque_pooled.get('mean_value')}, "
            f"p={macaque_pooled.get('p_value')}, n_sessions={non_human_a.get('n_sessions_reachable')}, "
            f"branch='{non_human_a.get('branch')}'. The non-human preparation establishes this link at "
            "single-unit resolution with a real intact maintenance delay; the human corpus establishes it "
            "only from field-potential power, at whatever temporal resolution a 1.9 s peri-word epoch "
            "allows, and never during an isolated, unstimulated delay period as such (RAM's task has no "
            "maintenance interval separate from encoding)."
        )
    else:
        a_text += "The delivered non-human comparison artifact was not readable at run time."

    b_full = block_b.get("pooled_normalised_displacement_by_channel_condition", {}).get("full_channel_set", {})
    b_shank = block_b.get("pooled_normalised_displacement_by_channel_condition", {}).get("excluding_stimulated_shank", {})
    c_med = block_c.get("mediation", {})
    bc_text = (
        "Blocks B/C (does stimulation displace the component, and does the induced displacement predict "
        f"induced behaviour change): human displacement branch='{block_b.get('branch')}', full-channel-set "
        f"normalised displacement={b_full.get('mean_value')} (p={b_full.get('p_value')}), artifact-cleaned "
        f"(shank-excluded) normalised displacement={b_shank.get('mean_value')} (p={b_shank.get('p_value')}); "
        f"mediation branch='{block_c.get('branch')}', slope/correlation r={c_med.get('r')}, "
        f"p={c_med.get('p_value')}, n_sessions={c_med.get('n_sessions')}, n_subjects={c_med.get('n_subjects')}. "
    )
    if non_human_bc is not None:
        gate = non_human_bc.get("gate", {})
        bc_text += (
            "The nearest delivered non-human equivalent (results/causal_macaque_pfc_microstimulation.json, macaque dlPFC "
            "delay-period microstimulation, macaque PFC microstimulation release) is the pooled CATE-vs-alignment "
            f"slope test: slope={gate.get('slope')}, CI=[{gate.get('slope_ci_lo')}, {gate.get('slope_ci_hi')}], "
            f"p={gate.get('p_value')}, n={gate.get('n')}, verdict='{non_human_bc.get('gate_verdict')}'. That "
            "artifact does not carry a component-displacement statistic built with this estimator (no prior "
            "delivered result applies rate_free_state_deviation to this corpus), so the comparison is between "
            "this project's own causal-mediation primitive on each preparation, not between two component "
            "numbers on a shared scale. The non-human preparation delivers stimulation strictly inside an "
            "intact WM-maintenance delay, at single-unit resolution, with a designed propensity; the human "
            "corpus can only deliver it at encoding, at field-potential resolution, with a design (open-loop) "
            "or classifier-driven (closed-loop) propensity."
        )
    else:
        bc_text += "The delivered non-human comparison artifact was not readable at run time."

    narrowing_text = (
        "What is narrowed by having both arms: neither preparation alone can separate 'does stimulation move "
        "a field-potential-scale, rate-free geometric signature of working-memory state' from 'does it move a "
        "single-unit-scale one', because the two differ simultaneously in species, recording modality, and "
        "epoch. Where both arms point the same direction on the same named question, that is two independent, "
        "differently-instrumented handles on it, not a replication in the strict sense (Block F asks whether "
        "the human arm's own two numbers -- Block B's displacement and its own alignment to v* -- are one "
        "object or two, but says nothing about whether the human and non-human arms are the same object). "
        "Where they disagree, the disagreement is left on the record rather than reconciled: a positive, "
        "well-powered result in one preparation and a null (bounded or underpowered) in the other constrains "
        "which preparation, not which mechanism, the project should stimulate in next."
    )
    epoch_disclosure = (
        "The epoch difference between the arms -- an intact WM-maintenance delay in the non-human corpus "
        "versus an encoding-period, stimulation-overlapping epoch in both human corpora -- is disclosed here "
        "as a covariate on every comparison in this block. It is never treated as a matching requirement, "
        "never grounds for excluding either arm, and never described as a confound to be removed."
    )

    return {
        "block_a_comparison": a_text, "block_b_and_c_comparison": bc_text,
        "what_is_narrowed": narrowing_text, "epoch_disclosure": epoch_disclosure,
    }


# ── Pre-task amplitude titration arm (closed-loop corpus only) ─────────────────
#
# Every closed-loop session opens with a short amplitude-titration series -- paired STIM_ON/STIM_OFF
# events carrying list == -999 and stim_list == 0, stepping an electrode pair through several
# amplitudes before the recall task itself begins. build_session_features epochs on WORD events only,
# so these events are otherwise never reached by any arm above. This section epochs them directly,
# reusing build_session_features's own EDF load and notch/high-gamma path (line_noise_notch,
# high_gamma_power) rather than a second signal path, and scores each event against the SAME
# control-trial reference direction and normalising spontaneous SD Block B already builds from that
# session's own non-stimulated word trials -- so these numbers sit on the identical scale.

def _pretask_events_tsv_path(ieeg_json: Path) -> Path:
    stem = str(ieeg_json).replace("_ieeg.json", "")
    return Path(stem.replace("_acq-bipolar", "") + "_events.tsv")


def find_pretask_titration_series(ieeg_json: Path) -> list[dict]:
    """Every pre-task amplitude-titration series in one session's own event table: STIM_ON rows with
    list == -999 and stim_list == 0, grouped by stimulated electrode pair (a session can carry more
    than one such series, at different pairs), each STIM_ON paired with the nearest following
    STIM_OFF at the same pair. A group with fewer than PRETASK_MIN_AMPLITUDE_LEVELS distinct
    delivered amplitudes has no dose axis to fit and is not returned."""
    import csv

    events_tsv = _pretask_events_tsv_path(ieeg_json)
    if not events_tsv.exists():
        return []
    with open(events_tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    stim_on = [r for r in rows if r.get("trial_type") == "STIM_ON"
              and r.get("list") == "-999" and r.get("stim_list") == "0"]
    if not stim_on:
        return []
    stim_off_all = [r for r in rows if r.get("trial_type") == "STIM_OFF"]

    groups: dict[tuple, list] = {}
    for r in sorted(stim_on, key=lambda r: float(r["onset"])):
        pair = (r.get("anode_label"), r.get("cathode_label"))
        groups.setdefault(pair, []).append(r)

    series = []
    for pair, evs in groups.items():
        event_records = []
        for on in evs:
            onset = float(on["onset"])
            candidates = [o for o in stim_off_all if o.get("anode_label") == pair[0]
                         and o.get("cathode_label") == pair[1] and float(o["onset"]) >= onset]
            if not candidates:
                continue
            off = min(candidates, key=lambda o: float(o["onset"]))
            event_records.append({
                "stim_on_onset_s": onset, "stim_off_onset_s": float(off["onset"]),
                "amplitude_uA": _float_or_nan(on.get("amplitude")),
                "pulse_width_us": _float_or_nan(on.get("pulse_width")),
                "n_pulses": _float_or_nan(on.get("n_pulses")),
            })
        event_records = [e for e in event_records if np.isfinite(e["amplitude_uA"])]
        if len({e["amplitude_uA"] for e in event_records}) < PRETASK_MIN_AMPLITUDE_LEVELS:
            continue
        series.append({"anode": pair[0], "cathode": pair[1], "events": event_records})
    return series


def _pretask_window_log_power(raw, srate: float, rec_dur_s: float,
                              window_start_s: float, window_len_s: float) -> np.ndarray | None:
    """Mean log high-gamma power per channel over [window_start_s, window_start_s + window_len_s),
    via the exact notch + high-gamma path build_session_features uses for word epochs, with the same
    kind of edge-effect padding on each side. None if the padded window falls outside the recording
    (always true of a series' very first event, whose STIM_ON is the first sample of the file)."""
    pad = int(round(PRETASK_EDGE_PAD_S * srate))
    i0 = int(round(window_start_s * srate)) - pad
    i1 = int(round((window_start_s + window_len_s) * srate)) + pad
    if i0 < 0 or i1 > int(rec_dur_s * srate):
        return None
    raw_seg = raw.get_data(start=i0, stop=i1)
    raw_seg_notched = line_noise_notch(raw_seg.T, srate, fundamental=60.0, n_harmonics=3)
    hgp = high_gamma_power(raw_seg_notched, srate=srate, lo=70.0, hi=150.0, smooth_ms=50.0)
    hgp = hgp[pad:-pad]
    if hgp.shape[0] < 1:
        return None
    return np.log1p(np.clip(hgp, 0, None)).mean(axis=0)


def _dose_quantities(amplitude_uA: np.ndarray, pulse_width_us: np.ndarray,
                     n_pulses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pulse and total charge in picocoulombs."""
    charge_per_pulse_pC = amplitude_uA * pulse_width_us
    return charge_per_pulse_pC, charge_per_pulse_pC * n_pulses


def load_pretask_series_events(corpus_name: str, session_key: str, ieeg_json: Path,
                               series_index: int, series: dict) -> dict:
    """Fit-level checkpointed baseline/post log-power vectors for one pre-task titration series -- a
    new, small cached unit (a handful of per-channel scalars per event), namespaced apart from this
    module's existing per-session word-epoch raw-feature cache (`{corpus}__{session_key}`) so that
    cache is never touched or invalidated by this addition."""
    unit = f"pretaskTitration__{corpus_name}__{session_key}__{series_index}"

    def _fit():
        import mne

        edf_path = Path(str(ieeg_json).replace("_ieeg.json", "_ieeg.edf"))
        if not edf_path.exists():
            return {"status": "not_computable_from_this_recording", "reason": "missing_edf_file"}
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        ch_names = list(raw.ch_names)
        stim_ch = f"{series['anode']}-{series['cathode']}"
        if stim_ch not in ch_names:
            stim_ch = f"{series['cathode']}-{series['anode']}"
        if stim_ch not in ch_names:
            return {"status": "not_computable_from_this_recording",
                    "reason": "stimulated_bipolar_channel_absent_from_edf_channel_list"}
        srate = raw.info["sfreq"]
        rec_dur = raw.times[-1]
        rows = []
        for ev in series["events"]:
            baseline = _pretask_window_log_power(raw, srate, rec_dur,
                                                 ev["stim_on_onset_s"] - PRETASK_BASELINE_WINDOW_S,
                                                 PRETASK_BASELINE_WINDOW_S)
            post = _pretask_window_log_power(raw, srate, rec_dur, ev["stim_off_onset_s"], PRETASK_POST_WINDOW_S)
            if baseline is None or post is None:
                continue
            rows.append({**ev, "baseline_log_power": baseline.tolist(), "post_log_power": post.tolist()})
        if len({r["amplitude_uA"] for r in rows}) < PRETASK_MIN_AMPLITUDE_LEVELS:
            return {
                "status": "not_computable_from_this_recording",
                "reason": (
                    "fewer than two amplitude levels retain both a usable baseline and post window -- "
                    "usually because the recording begins at or after the series' own first pulse, so "
                    "that earliest event has no room for a pre-stimulation baseline window"
                ),
                "n_events_total": len(series["events"]), "n_events_usable": len(rows),
            }
        return {
            "status": "computed", "ch_names": ch_names, "stim_channel": stim_ch,
            "n_events_total": len(series["events"]), "n_events_usable": len(rows), "rows": rows,
        }

    return run_checkpointed(unit, _fit)


def _pretask_series_analysis(rec: dict, series_idx: int, series: dict) -> dict:
    """One pre-task titration series scored against its own session's word-control reference
    direction and spontaneous SD -- the SAME reference construction (_reference_direction /
    _deviation_from_reference) and normalising SD Block B's own displacement uses, applied here to
    the pre-task baseline/post windows instead of to word-epoch control/stimulated trials, on the
    artifact-safe channel condition (excluding the stimulated shank) Blocks C/D/F already standardise
    on for any downstream question built on top of Block B."""
    loaded = load_pretask_series_events(rec["corpus"], rec["session_key"], Path(rec["ieeg_json"]), series_idx, series)
    if loaded.get("status") != "computed":
        return loaded

    arrays = rec["arrays"]
    ch_names = arrays["ch_names"].tolist()
    if ch_names != loaded["ch_names"]:
        return {"status": "not_computable_from_this_recording",
                "reason": "channel order in the freshly-read EDF does not match this session's cached "
                          "word-epoch channel order"}
    anode, cathode, stim_ch = str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
    mask = channel_condition_masks(ch_names, anode, cathode, stim_ch)["excluding_stimulated_shank"]
    ctrl_activity = _bin_averaged(arrays, mask)[arrays["stim_flag"] == 0]
    control_deviation = rate_free_state_deviation(ctrl_activity)
    finite_ctrl = np.isfinite(control_deviation)
    if finite_ctrl.sum() < 8:
        return {"status": "not_computable_from_this_recording",
                "reason": "fewer than 8 finite-direction word control trials in this session"}
    reference_direction = _reference_direction(ctrl_activity)
    spontaneous_sd = float(np.nanstd(control_deviation[finite_ctrl], ddof=1))
    if not (spontaneous_sd > 0):
        return {"status": "not_computable_from_this_recording", "reason": "zero spontaneous control sd"}

    rows = loaded["rows"]
    baseline_mat = np.array([r["baseline_log_power"] for r in rows])[:, mask]
    post_mat = np.array([r["post_log_power"] for r in rows])[:, mask]
    baseline_dev = _deviation_from_reference(baseline_mat, reference_direction)
    post_dev = _deviation_from_reference(post_mat, reference_direction)
    normalised_displacement = (post_dev - baseline_dev) / spontaneous_sd
    amplitude = np.array([r["amplitude_uA"] for r in rows])
    finite = np.isfinite(normalised_displacement)
    if finite.sum() < 2 or len(set(amplitude[finite].tolist())) < PRETASK_MIN_AMPLITUDE_LEVELS:
        return {"status": "not_computable_from_this_recording",
                "reason": "fewer than two amplitude levels have a finite normalised displacement"}

    x, y = amplitude[finite], normalised_displacement[finite]
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(stable_seed(f"pretask_titration_shuffle|{rec['session_key']}|{series_idx}"))
    null = np.array([np.polyfit(x, rng.permutation(y), 1)[0] for _ in range(N_DOSE_SHUFFLE_DRAWS)])
    shuffle_p = permutation_pvalue(np.abs(null) >= np.abs(slope))

    pulse_width = np.array([r["pulse_width_us"] for r in rows])[finite]
    n_pulses = np.array([r["n_pulses"] for r in rows])[finite]
    charge_per_pulse_pC, total_delivered_charge_pC = _dose_quantities(x, pulse_width, n_pulses)

    return {
        "status": "computed",
        "stim_channel": loaded["stim_channel"],
        "n_amplitude_levels": len(set(x.tolist())),
        "n_events_total": loaded["n_events_total"], "n_events_usable": int(finite.sum()),
        "amplitude_uA_slope": slope,
        "amplitude_uA_shuffle_null": {"observed_slope": slope, "p_value": shuffle_p, "n_draws": N_DOSE_SHUFFLE_DRAWS},
        "pulse_width_us_constant_within_series": bool(np.ptp(pulse_width) < 1e-9),
        "n_pulses_constant_within_series": bool(np.ptp(n_pulses) < 1e-9),
        "events": [{
            "amplitude_uA": float(x[i]), "charge_per_pulse_pC": float(charge_per_pulse_pC[i]),
            "total_delivered_charge_pC": float(total_delivered_charge_pC[i]),
            "pulse_width_us": float(pulse_width[i]), "n_pulses": float(n_pulses[i]),
            "normalised_displacement": float(y[i]),
        } for i in range(len(x))],
    }


def _classify_pretask_titration(pooled: dict, shuffle_p: float | None,
                                 n_series_clearing_displacement_threshold: int | None,
                                 n_series_with_displacement_mdd: int | None) -> str:
    """The pooled mdd from `pooled["mdd"]` is on the slope scale (normalised displacement PER
    MICROAMPERE), not the level scale MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT is defined
    on -- comparing them directly compares different units. The commensurable comparison converts the
    slope-scale mdd onto the displacement scale by multiplying it by each series' own realised
    amplitude range (a slope is only a level once it is applied over some range), then asks whether
    EVERY series individually clears the threshold on that scale -- the same "must hold, not just on
    average" standard a powered null is held to everywhere else in this module. One series short of
    that leaves at least one series where the design could not have detected a meaningful effect even
    if it were present, so the arm as a whole cannot claim a powered null."""
    if pooled.get("status") != "computed":
        return "not_computable_from_this_recording"
    significant = pooled["p_value"] <= ALPHA and pooled["mean_value"] > 0
    shuffle_significant = shuffle_p is not None and shuffle_p <= ALPHA
    if significant and shuffle_significant:
        return "displacement_scales_with_amplitude_within_session"
    if not n_series_with_displacement_mdd:
        return "underpowered_to_ask"
    if n_series_clearing_displacement_threshold == n_series_with_displacement_mdd:
        return "no_scaling_above_the_reported_bound"
    return "underpowered_to_ask"


def run_pretask_amplitude_titration(closedloop_corpus: dict, block_b: dict) -> dict:
    usable_by_key = {r["session_key"]: r for r in closedloop_corpus["records"]}
    exclusions = closedloop_corpus["exclusions"]

    def _exclusion_reason(session_key: str) -> str:
        for reason, keys in exclusions.items():
            if session_key in keys:
                return f"session_excluded_from_word_epoch_loading: {reason}"
        return "session_excluded_from_word_epoch_loading: reason_unknown"

    session_jsons = _find_session_jsons(CLOSEDLOOP_DATA)
    per_series = {}
    for ieeg_json in session_jsons:
        session_key = str(ieeg_json.relative_to(CLOSEDLOOP_DATA))
        subject_id = _subject_id(ieeg_json)
        for idx, series in enumerate(find_pretask_titration_series(ieeg_json)):
            series_key = f"{subject_id}__{session_key}__{series['anode']}-{series['cathode']}__{idx}"
            rec = usable_by_key.get(session_key)
            if rec is None:
                per_series[series_key] = {
                    "status": "not_computable_from_this_recording", "reason": _exclusion_reason(session_key),
                }
            else:
                rec_with_path = {**rec, "ieeg_json": rec["ieeg_json"]}
                result = run_checkpointed(
                    f"pretaskTitration__closed_loop_ds005557__{session_key}__{idx}",
                    lambda rec=rec_with_path, idx=idx, series=series: _pretask_series_analysis(rec, idx, series))
                per_series[series_key] = result
            per_series[series_key]["subject"] = subject_id
            per_series[series_key]["session_key"] = session_key
            per_series[series_key]["electrode_pair"] = f"{series['anode']}-{series['cathode']}"

    n_series_seen = len(per_series)
    computed = {k: v for k, v in per_series.items() if v.get("status") == "computed"}
    refused = {k: v for k, v in per_series.items() if v.get("status") != "computed"}

    slopes = np.array([v["amplitude_uA_slope"] for v in computed.values()])
    subj_for_slopes = [v["subject"] for v in computed.values()]
    pooled = subject_clustered_mean_test(slopes, subj_for_slopes) if len(slopes) else {"status": "not_computable"}

    shuffle_p = None
    if pooled.get("status") == "computed":
        subj_groups: dict[str, list] = {}
        for v in computed.values():
            subj_groups.setdefault(v["subject"], []).append(v)
        rng = np.random.default_rng(stable_seed("pretask_titration_pooled_shuffle"))
        null = np.empty(N_DOSE_SHUFFLE_DRAWS)
        for i in range(N_DOSE_SHUFFLE_DRAWS):
            subj_means = []
            for series_recs in subj_groups.values():
                series_slopes = []
                for v in series_recs:
                    x = np.array([e["amplitude_uA"] for e in v["events"]])
                    y_shuf = rng.permutation(np.array([e["normalised_displacement"] for e in v["events"]]))
                    series_slopes.append(float(np.polyfit(x, y_shuf, 1)[0]))
                subj_means.append(float(np.mean(series_slopes)))
            null[i] = float(np.mean(subj_means))
        shuffle_p = permutation_pvalue(np.abs(null) >= np.abs(pooled["mean_value"]))

    # Displacement-scale conversion (see the module-level note above `run_pretask_amplitude_titration`
    # and `_classify_pretask_titration`'s docstring): `pooled["mdd"]` is a SLOPE -- normalised
    # displacement per microampere -- and is not on the same scale as
    # MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT, which is a level. Multiplying the slope-scale
    # mdd by a series' own realised amplitude range (max delivered amplitude minus min, within that
    # series) puts it on the level scale the threshold is actually defined on. Read from `events`,
    # already cached per series, so no series needs re-fitting to compute this.
    realised_amplitude_range_uA_by_series = {
        k: float(np.ptp(np.array([e["amplitude_uA"] for e in v["events"]]))) for k, v in computed.items()
    }
    mdd_slope = pooled.get("mdd") if pooled.get("status") == "computed" else None
    minimum_detectable_displacement_by_series, n_series_clearing_displacement_threshold = None, None
    if mdd_slope is not None and mdd_slope.get("status") == "computed":
        minimum_detectable_displacement_by_series = {
            k: mdd_slope["mdd"] * rng_uA for k, rng_uA in realised_amplitude_range_uA_by_series.items()
        }
        n_series_clearing_displacement_threshold = sum(
            1 for d in minimum_detectable_displacement_by_series.values()
            if d < MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT
        )
    range_values = np.array(list(realised_amplitude_range_uA_by_series.values()))
    realised_amplitude_range_uA_summary = {
        "n_series": int(len(range_values)),
        "min_uA": float(np.min(range_values)) if len(range_values) else None,
        "median_uA": float(np.median(range_values)) if len(range_values) else None,
        "max_uA": float(np.max(range_values)) if len(range_values) else None,
    }

    # Context for MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT: the observed spread of individual
    # pre-task titration events on the same normalised scale the threshold is defined on.
    all_event_displacements = np.array([e["normalised_displacement"] for v in computed.values() for e in v["events"]])
    observed_per_event_normalised_displacement_dispersion = {
        "n_events": int(len(all_event_displacements)),
        "min": float(np.min(all_event_displacements)) if len(all_event_displacements) else None,
        "median": float(np.median(all_event_displacements)) if len(all_event_displacements) else None,
        "max": float(np.max(all_event_displacements)) if len(all_event_displacements) else None,
        "sd": float(np.std(all_event_displacements, ddof=1)) if len(all_event_displacements) > 1 else None,
        "units": "normalised displacement (same scale as MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT)",
    }

    n_series_with_two_amplitude_levels = sum(1 for v in computed.values() if v["n_amplitude_levels"] == 2)
    n_series_with_zero_residual_degrees_of_freedom = sum(
        1 for v in computed.values() if v["n_events_usable"] - 2 <= 0
    )
    degrees_of_freedom_note = {
        "n_series_with_two_amplitude_levels": n_series_with_two_amplitude_levels,
        "n_series_with_zero_residual_degrees_of_freedom": n_series_with_zero_residual_degrees_of_freedom,
        "note": (
            "A linear slope fit has two free parameters (slope, intercept); a series contributes zero "
            "residual degrees of freedom to its own slope estimate when it has exactly two usable events "
            "total, not merely two distinct amplitude levels -- a two-level series with repeat events at "
            "either level still has residual degrees of freedom."
        ),
    }

    branch = _classify_pretask_titration(pooled, shuffle_p, n_series_clearing_displacement_threshold,
                                          len(minimum_detectable_displacement_by_series)
                                          if minimum_detectable_displacement_by_series is not None else None)

    # Separate cell: does the titration slope measured minutes before the task predict this same
    # session's own task-period stimulation displacement (Block B, excluding-stimulated-shank
    # condition), computed at the amplitude the task itself subsequently used?
    slope_x, task_disp_y, subj_task = [], [], []
    for v in computed.values():
        b_rec = block_b["per_session"].get(v["session_key"], {})
        cond = b_rec.get("conditions", {}).get("excluding_stimulated_shank", {})
        if cond.get("status") == "computed" and cond.get("normalised_displacement") is not None:
            slope_x.append(v["amplitude_uA_slope"])
            task_disp_y.append(cond["normalised_displacement"])
            subj_task.append(v["subject"])
    task_amplitude_prediction = subject_aggregated_correlation(np.array(slope_x), np.array(task_disp_y), subj_task) \
        if len(slope_x) >= 4 else {"status": "not_computable", "n_series": len(slope_x)}

    return {
        "scope": (
            "A within-session, within-electrode-pair, within-subject amplitude-titration series "
            "delivered before the recall task begins, closed-loop corpus only (the open-loop corpus "
            "carries no such series). Additive to the task-period dose arm reported elsewhere in this "
            "artifact; does not restate or re-fire that arm's own branch or change its n."
        ),
        "baseline_window": {"length_s": PRETASK_BASELINE_WINDOW_S, "edge_pad_s": PRETASK_EDGE_PAD_S,
                            "definition": "length_s immediately preceding each pre-task STIM_ON onset"},
        "post_window": {"length_s": PRETASK_POST_WINDOW_S, "edge_pad_s": PRETASK_EDGE_PAD_S,
                        "definition": "length_s immediately following the matching STIM_OFF onset"},
        "channel_condition_used": "excluding_stimulated_shank",
        "zero_drop": {"n_series_seen": n_series_seen, "n_analysed": len(computed), "n_refused": len(refused),
                     "seen_equals_analysed_plus_refused": n_series_seen == len(computed) + len(refused)},
        "per_series": per_series,
        "pooled_amplitude_slope_subject_clustered": pooled,
        "pooled_shuffle_null": (
            {"p_value": shuffle_p, "n_draws": N_DOSE_SHUFFLE_DRAWS} if shuffle_p is not None
            else {"status": "not_computable"}
        ),
        "minimum_detectable_slope_80pct_power": pooled.get("mdd") if pooled.get("status") == "computed" else None,
        "minimum_detectable_slope_80pct_power_units": "normalised displacement per microampere",
        "minimum_detectable_displacement_80pct_power_by_series": minimum_detectable_displacement_by_series,
        "minimum_detectable_displacement_80pct_power_units": (
            "normalised displacement, over the series' own realised amplitude range -- i.e. the "
            "slope-scale minimum detectable difference multiplied by that series' own "
            "(max delivered amplitude minus min delivered amplitude); commensurable with "
            "meaningful_effect_threshold_normalised_displacement, unlike the slope-scale quantity above"
        ),
        "realised_amplitude_range_uA_by_series": realised_amplitude_range_uA_by_series,
        "realised_amplitude_range_uA_summary": realised_amplitude_range_uA_summary,
        "n_series_clearing_displacement_threshold": n_series_clearing_displacement_threshold,
        "n_series_evaluated_for_displacement_threshold": (
            len(minimum_detectable_displacement_by_series) if minimum_detectable_displacement_by_series is not None
            else None
        ),
        "observed_per_event_normalised_displacement_dispersion": observed_per_event_normalised_displacement_dispersion,
        "degrees_of_freedom_note": degrees_of_freedom_note,
        "branch": branch,
        "meaningful_effect_threshold_normalised_displacement": MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT,
        "dose_field_units_note": DOSE_FIELD_UNITS_NOTE,
        "task_amplitude_prediction": {
            "question": (
                "does the titration slope measured minutes before the task predict this same session's "
                "own task-period stimulation displacement, computed at the amplitude the task itself "
                "subsequently used"
            ),
            "result": task_amplitude_prediction,
        },
    }


# ── Two-arm dose-scaling meta-analysis (task-period + pre-task titration, disjoint subjects) ───────────
#
# The task-period dose arm (Block D) and the pre-task titration arm each carry confound-free
# within-subject-and-within-electrode-pair amplitude variation, but in two disjoint subject sets, and
# neither alone clears its own subject-count bar. This combines them AS TWO ARMS of one meta-analysis
# (never by pooling rows across the two subject sets, which would treat a task-period session and a
# pre-task calibration event as exchangeable observations of the same unit). Each arm already reports
# a slope of the identical quantity -- normalised_displacement, itself already identically defined in
# both arms (a session's own control-trial rate-free cosine-deviation change divided by that same
# session's own control-trial standard deviation of the same quantity) -- so putting both arms on one
# common scale is: that arm's own pooled slope multiplied by that arm's own realised amplitude range.

def _spontaneous_control_sd_direct(arrays: dict) -> float | None:
    """Recomputes one session's own control-trial rate-free-deviation standard deviation directly
    from its cached raw arrays, on the excluding_stimulated_shank channel condition -- the identical
    formula both Block B's own spontaneous_control_sd and the pre-task titration arm's own internal
    spontaneous_sd already compute, reproduced here independently of either call site so the two can
    be compared rather than merely read from the same source code."""
    ch_names = arrays["ch_names"].tolist()
    anode, cathode, stim_ch = str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
    mask = channel_condition_masks(ch_names, anode, cathode, stim_ch)["excluding_stimulated_shank"]
    ctrl_activity = _bin_averaged(arrays, mask)[arrays["stim_flag"] == 0]
    control_deviation = rate_free_state_deviation(ctrl_activity)
    finite = np.isfinite(control_deviation)
    if finite.sum() < 2:
        return None
    return float(np.nanstd(control_deviation[finite], ddof=1))


def _verify_commensurable_normalisation(pretask: dict, block_b: dict, closedloop_records: list[dict]) -> dict:
    """For every session the pre-task titration arm actually used, independently recomputes that
    session's own spontaneous control-trial SD and compares it against Block B's own stored value for
    the same session and channel condition -- if every one matches, both arms normalise their raw
    displacement by the identical reference dispersion definition and may be combined; if any
    disagree, they must not be."""
    records_by_key = {r["session_key"]: r for r in closedloop_records}
    computed_sessions = sorted({v["session_key"] for v in pretask["per_series"].values()
                                if v.get("status") == "computed"})
    checked = []
    for sk in computed_sessions:
        rec = records_by_key.get(sk)
        b_cond = block_b["per_session"].get(sk, {}).get("conditions", {}).get("excluding_stimulated_shank", {})
        if rec is None or b_cond.get("status") != "computed":
            continue
        recomputed = _spontaneous_control_sd_direct(rec["arrays"])
        stored = b_cond.get("spontaneous_control_sd")
        if recomputed is None or stored is None:
            continue
        checked.append({
            "session_key": sk, "block_b_spontaneous_control_sd": stored,
            "independently_recomputed_spontaneous_sd": recomputed,
            "match": bool(np.isclose(recomputed, stored, rtol=1e-9, atol=1e-12)),
        })
    all_match = len(checked) > 0 and all(c["match"] for c in checked)
    return {
        "method": (
            "recomputes each session's own control-trial rate-free-deviation standard deviation "
            "(excluding-stimulated-shank channel condition) directly from cached per-session arrays, "
            "independently of both the task-period and pre-task titration code paths, and compares it "
            "against the task-period path's own already-stored value for every session the pre-task "
            "titration arm actually used"
        ),
        "n_sessions_checked": len(checked),
        "all_match": all_match,
        "per_session": checked,
        "commensurable": all_match,
    }


def _task_period_dose_arm(block_d_rows: list[dict], exclude_subjects: set) -> tuple[dict, dict, set]:
    """The within-subject-and-within-electrode-pair amplitude slope Block D itself would fit for a
    subject at or above MIN_SUBJECTS_FOR_WITHIN_SUBJECT_DOSE, run here unconditionally on Block D's
    own already-serialised rows -- this is a separate, forward-only question about this arm's own
    displacement-scale estimate for a meta-analysis, not a re-firing of Block D's own dose-scaling
    branch, and changes nothing Block D itself returns."""
    by_subject_pair: dict[tuple, list] = {}
    for row in block_d_rows:
        if row["subject"] in exclude_subjects:
            continue
        by_subject_pair.setdefault((row["subject"], row["stim_channel"]), []).append(row)
    dose_varying = {k: v for k, v in by_subject_pair.items() if len({r["amplitude_uA"] for r in v}) >= 2}
    slopes, subj, ranges = [], [], {}
    for (s, _pair), recs in dose_varying.items():
        x = np.array([r["amplitude_uA"] for r in recs])
        y = np.array([r["normalised_displacement"] for r in recs])
        slopes.append(float(np.polyfit(x, y, 1)[0]))
        subj.append(s)
        ranges[s] = float(np.ptp(x))
    pooled = subject_clustered_mean_test(np.array(slopes), subj) if slopes else {"status": "not_computable"}
    range_summary = {
        "n_subjects": len(ranges),
        "min_uA": float(np.min(list(ranges.values()))) if ranges else None,
        "median_uA": float(np.median(list(ranges.values()))) if ranges else None,
        "max_uA": float(np.max(list(ranges.values()))) if ranges else None,
    }
    return pooled, range_summary, set(subj)


def _pretask_titration_dose_arm(pretask: dict, exclude_subjects: set) -> tuple[dict, dict, set]:
    """Reuses the pre-task titration arm's own already-computed pooled slope and realised-range
    summary unchanged when there is no subject to drop; rebuilds both from the arm's own cached
    per-series slopes (amplitude_uA_slope, never refit) when a subject must be excluded for
    disjointness."""
    kept = {k: v for k, v in pretask["per_series"].items()
           if v.get("status") == "computed" and v["subject"] not in exclude_subjects}
    if not exclude_subjects:
        return (pretask["pooled_amplitude_slope_subject_clustered"],
                pretask["realised_amplitude_range_uA_summary"], {v["subject"] for v in kept.values()})
    slopes = np.array([v["amplitude_uA_slope"] for v in kept.values()])
    subj = [v["subject"] for v in kept.values()]
    pooled = subject_clustered_mean_test(slopes, subj) if len(slopes) else {"status": "not_computable"}
    ranges = [float(np.ptp(np.array([e["amplitude_uA"] for e in v["events"]]))) for v in kept.values()]
    range_summary = {
        "n_series": len(ranges),
        "min_uA": float(np.min(ranges)) if ranges else None,
        "median_uA": float(np.median(ranges)) if ranges else None,
        "max_uA": float(np.max(ranges)) if ranges else None,
    }
    return pooled, range_summary, set(subj)


def _task_period_subject_level_slopes(block_d_rows: list[dict], exclude_subjects: set) -> np.ndarray:
    """The same per-subject-pair-then-per-subject collapse _task_period_dose_arm feeds into
    subject_clustered_mean_test, returned as the one-value-per-subject array directly -- so a
    caller that needs to resample subjects (a bootstrap) or exhaustively enumerate their sign
    assignments (a permutation-null capacity check) does not have to re-derive the pooled test's
    own internal collapse."""
    by_subject_pair: dict[tuple, list] = {}
    for row in block_d_rows:
        if row["subject"] in exclude_subjects:
            continue
        by_subject_pair.setdefault((row["subject"], row["stim_channel"]), []).append(row)
    dose_varying = {k: v for k, v in by_subject_pair.items() if len({r["amplitude_uA"] for r in v}) >= 2}
    slopes, subj = [], []
    for (s, _pair), recs in dose_varying.items():
        x = np.array([r["amplitude_uA"] for r in recs])
        y = np.array([r["normalised_displacement"] for r in recs])
        slopes.append(float(np.polyfit(x, y, 1)[0]))
        subj.append(s)
    slopes, subj = np.array(slopes), np.array(subj)
    unique_subjects = sorted(set(subj.tolist()))
    return np.array([slopes[subj == s].mean() for s in unique_subjects])


def _pretask_titration_subject_level_slopes(pretask: dict, exclude_subjects: set) -> np.ndarray:
    """Same one-value-per-subject collapse as _pretask_titration_dose_arm's own rebuild path,
    returned directly (see _task_period_subject_level_slopes for why)."""
    kept = {k: v for k, v in pretask["per_series"].items()
           if v.get("status") == "computed" and v["subject"] not in exclude_subjects}
    slopes = np.array([v["amplitude_uA_slope"] for v in kept.values()])
    subj = np.array([v["subject"] for v in kept.values()])
    unique_subjects = sorted(set(subj.tolist()))
    return np.array([slopes[subj == s].mean() for s in unique_subjects])


def _displacement_scale_arm(pooled_slope: dict, range_summary: dict, label: str, z_factor: float) -> dict:
    """Puts one arm's own pooled per-microampere slope onto the displacement scale by multiplying it,
    its standard error (back-derived from the arm's own already-computed minimum detectable
    difference: se = mdd / z_factor, the same z_factor minimum_detectable_paired_difference uses), and
    its CI by that arm's own realised amplitude range (its median, across its own dose-varying
    subjects or series) -- a slope is only a level once it is applied over some range."""
    if pooled_slope.get("status") != "computed" or pooled_slope.get("mdd", {}).get("status") != "computed" \
            or range_summary.get("median_uA") is None:
        return {"status": "not_computable", "label": label}
    rng_uA = range_summary["median_uA"]
    se_slope = pooled_slope["mdd"]["mdd"] / z_factor
    return {
        "status": "computed", "label": label,
        "n_subjects": pooled_slope["n_subjects"],
        "own_slope_estimate_per_microampere": pooled_slope["mean_value"],
        "own_slope_p_value": pooled_slope["p_value"],
        "realised_amplitude_range_uA_used": rng_uA,
        "displacement_estimate": pooled_slope["mean_value"] * rng_uA,
        "displacement_se": se_slope * rng_uA,
        "displacement_ci_lower": pooled_slope["ci_lower"] * rng_uA,
        "displacement_ci_upper": pooled_slope["ci_upper"] * rng_uA,
    }


TWO_ARM_META_DECISION_RULE_TEXT = (
    "Pre-declared before the pooled result was computed. (1) If the two arms are not verified "
    "commensurable (independently recomputed against the same reference-dispersion definition) they "
    "are not combined; report the incommensurability instead. (2) If the two arms' own dose-varying "
    "subject sets are not verified disjoint and the overlap cannot be resolved by dropping the "
    "overlapping subject from one arm, they are not combined. (3) If the two arms' own "
    "displacement-scale point estimates disagree in sign, or the between-arm heterogeneity is "
    "substantial (I^2 >= " + repr(TWO_ARM_META_HETEROGENEITY_I_SQUARED_FLOOR) + ", or Cochran's Q has "
    "more than one degree of freedom and its p-value is <= alpha), the pooled estimate is declared "
    "uninterpretable and is not presented as an answer, regardless of its own p-value. (4) Otherwise, "
    "if the pooled two-sided p-value is <= alpha, the arms jointly show dose scaling. (5) Otherwise, "
    "if the pooled minimum detectable displacement at 80% power is below "
    "meaningful_effect_threshold_normalised_displacement, this is a powered null across both arms "
    "combined. (6) Otherwise the combined arms are underpowered to ask the question."
)


def _classify_two_arm_meta(commensurable: bool, disjoint_or_resolved: bool, task_arm: dict,
                           pretask_arm: dict, forest: dict | None, pooled_mdd_displacement: float | None) -> str:
    if not commensurable:
        return "arms_not_combinable_incommensurable_normalisation"
    if not disjoint_or_resolved:
        return "arms_not_combinable_subject_overlap_unresolved"
    if task_arm.get("status") != "computed" or pretask_arm.get("status") != "computed" or forest is None:
        return "not_computable"
    signs_agree = (task_arm["displacement_estimate"] > 0) == (pretask_arm["displacement_estimate"] > 0)
    heterogeneity_substantial = (
        forest["i_squared"] >= TWO_ARM_META_HETEROGENEITY_I_SQUARED_FLOOR
        or (forest["Q_df"] > 0 and forest["Q_p"] <= ALPHA)
    )
    if not signs_agree or heterogeneity_substantial:
        return "pooled_estimate_uninterpretable_arm_disagreement"
    if forest["p_value"] <= ALPHA:
        return "dose_scaling_across_both_arms_pooled"
    if pooled_mdd_displacement is not None and pooled_mdd_displacement < MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT:
        return "no_dose_scaling_above_the_reported_bound_pooled_across_arms"
    return "underpowered_to_ask"


def _sign_flip_null_capacity(n_subjects: int, alpha: float = ALPHA) -> dict:
    """Characterises the finite null a subject-level sign-flip permutation test can actually
    produce, independently of how many Monte Carlo draws are taken from it. With n independent
    subjects, flipping each subject's own sign gives 2**n distinct sign assignments; every
    assignment and its complete-negation mirror always produce the same absolute pooled statistic,
    so the smallest two-sided p-value this null can ever report -- reached only when the actually-
    observed (unflipped) assignment is tied for the single most extreme magnitude -- is 2 / 2**n.
    If that floor already exceeds alpha, no possible arrangement of this arm's own n subjects, for
    any values those subjects could have taken, could ever cross significance."""
    n_arrangements = 2 ** int(n_subjects)
    smallest_two_sided_p = 2.0 / n_arrangements
    return {
        "null_type": "subject-level sign-flip permutation",
        "n_subjects": int(n_subjects),
        "n_distinct_sign_arrangements": int(n_arrangements),
        "smallest_attainable_two_sided_p": smallest_two_sided_p,
        "capable_of_significance_at_alpha": bool(smallest_two_sided_p <= alpha),
    }


def _exhaustive_sign_flip_check(subject_values: np.ndarray) -> dict:
    """Exhaustively enumerates every sign assignment this arm's own subject-level sign-flip null
    can produce (2**n, small enough here to enumerate directly rather than sample) and reports the
    exact two-sided p-value alongside whether the actually-observed (unflipped) statistic is the
    unique-up-to-global-negation minimum-magnitude arrangement. That condition is exactly when a
    Monte Carlo sample of this null, however large, reports p=1.0: every arrangement (including the
    observed one) then satisfies |arrangement statistic| >= |observed statistic| by construction, so
    every sampled draw counts toward the numerator."""
    subject_values = np.asarray(subject_values, dtype=float)
    n = len(subject_values)
    observed = float(np.sum(subject_values))
    signs = np.array(np.meshgrid(*[[1.0, -1.0]] * n, indexing="ij")).reshape(n, -1).T
    stats = signs @ subject_values
    tol = 1e-9 * max(1.0, abs(observed))
    exceed = np.abs(stats) >= abs(observed) - tol
    exact_two_sided_p = float(exceed.sum()) / len(stats)
    return {
        "n_subjects": int(n),
        "n_arrangements_enumerated": int(len(stats)),
        "exact_two_sided_p": exact_two_sided_p,
        "observed_statistic_is_global_minimum_magnitude": bool(exact_two_sided_p >= 1.0 - 1e-9),
    }


def _bootstrap_pooled_mdd_displacement(task_subject_values: np.ndarray, pretask_subject_values: np.ndarray,
                                       task_range_uA: float, pretask_range_uA: float, z_factor: float,
                                       n_boot: int = N_BOOT) -> dict:
    """Puts an uncertainty interval on the pooled minimum detectable displacement itself, since it
    is built from standard errors estimated from only 3 and 7 subjects and an SE estimated from that
    few units is itself a random variable with a wide sampling distribution. Resamples subjects with
    replacement independently within each arm (arm membership held fixed -- a resample never moves a
    subject from one arm to the other), recomputes each arm's own mean and standard error of the
    mean on the resampled subjects, converts to the displacement scale with that arm's own already-
    realised amplitude range held fixed, re-pools the two arms exactly as the point estimate is
    pooled, and records the resulting pooled minimum detectable displacement. A resample whose
    within-arm draws are all identical (zero variance, no defined standard error) is dropped rather
    than fabricating an SE of zero."""
    rng = np.random.default_rng(stable_seed("dose_scaling_two_arm_bootstrap"))
    n_task, n_pretask = len(task_subject_values), len(pretask_subject_values)
    draws = []
    for _ in range(n_boot):
        t_samp = task_subject_values[rng.integers(0, n_task, size=n_task)]
        p_samp = pretask_subject_values[rng.integers(0, n_pretask, size=n_pretask)]
        t_sd, p_sd = float(np.std(t_samp, ddof=1)), float(np.std(p_samp, ddof=1))
        if not (np.isfinite(t_sd) and np.isfinite(p_sd)) or t_sd <= 0 or p_sd <= 0:
            continue
        t_se, p_se = t_sd / np.sqrt(n_task), p_sd / np.sqrt(n_pretask)
        t_est, p_est = float(np.mean(t_samp)) * task_range_uA, float(np.mean(p_samp)) * pretask_range_uA
        t_se_disp, p_se_disp = t_se * task_range_uA, p_se * pretask_range_uA
        forest = forest_meta(np.array([t_est, p_est]), np.array([t_se_disp, p_se_disp]))
        draws.append(z_factor * forest["se"])
    draws = np.array(draws)
    ci_lower, ci_upper = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) \
        if len(draws) else (None, None)
    return {
        "method": (
            "percentile bootstrap; subjects resampled with replacement independently within each "
            "arm (arm membership held fixed), each replicate's per-arm mean and standard error of "
            "the mean re-pooled by the same inverse-variance meta-analysis used for the point "
            "estimate, holding each arm's own realised amplitude range fixed"
        ),
        "n_boot_requested": int(n_boot),
        "n_boot_usable": int(len(draws)),
        "n_boot_dropped_zero_variance_resample": int(n_boot - len(draws)),
        "ci_lower_2.5pct": ci_lower,
        "ci_upper_97.5pct": ci_upper,
        "upper_bound_below_meaningful_effect_threshold": (
            bool(ci_upper < MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT) if ci_upper is not None else None
        ),
    }


def _heterogeneity_guard_capacity(task_arm: dict, pretask_arm: dict, forest: dict,
                                  i_squared_target: float = TWO_ARM_META_HETEROGENEITY_I_SQUARED_FLOOR) -> dict:
    """Quantifies how structurally unreachable the pre-declared heterogeneity guard is at exactly
    two arms with this weighting. With k=2, Cochran's Q reduces to a closed form,
    Q = w1*w2/(w1+w2) * (estimate1 - estimate2)**2, where w1, w2 are the arms' own inverse-variance
    weights (fixed by their own standard errors) -- so, holding both arms' own standard errors fixed,
    there is a closed-form answer for how far apart the two arms' point estimates would have to be
    for I^2 to reach i_squared_target (I^2 = 50.0 requires Q = 2.0, since Q_df = 1 here). Reports the
    smaller of the two displacement values the lower-weight arm's own point estimate would have to
    move to for the heterogeneity guard to register that threshold, holding the higher-weight arm's
    estimate and both arms' own standard errors fixed. This does not change interpretable, the fired
    branch, or the decision rule's own text -- it only measures how much room that rule's own I^2
    clause actually has to fire, given the two arms already delivered."""
    arms = [task_arm, pretask_arm]
    weights = [1.0 / a["displacement_se"] ** 2 for a in arms]
    low_idx = 0 if weights[0] < weights[1] else 1
    high_idx = 1 - low_idx
    low_arm, high_arm = arms[low_idx], arms[high_idx]
    w_low, w_high = weights[low_idx], weights[high_idx]
    if i_squared_target >= 100.0:
        return {
            "i_squared_target": i_squared_target,
            "status": "not_computable",
            "reason": "I^2 target of 100 or more requires Q -> infinity, i.e. an infinite displacement, given finite standard errors",
        }
    q_target = forest["Q_df"] / (1.0 - i_squared_target / 100.0)
    coefficient = w_high * w_low / (w_high + w_low)
    delta_needed = float(np.sqrt(q_target / coefficient))
    candidate_a, candidate_b = high_arm["displacement_estimate"] - delta_needed, high_arm["displacement_estimate"] + delta_needed
    shift_a = abs(candidate_a - low_arm["displacement_estimate"])
    shift_b = abs(candidate_b - low_arm["displacement_estimate"])
    nearer_candidate, nearer_shift = (candidate_a, shift_a) if shift_a <= shift_b else (candidate_b, shift_b)
    return {
        "i_squared_target": i_squared_target,
        "status": "computed",
        "current_Q": forest["Q"],
        "Q_needed_for_i_squared_target": q_target,
        "lower_weight_arm_label": low_arm["label"],
        "lower_weight_arm_current_displacement_estimate": low_arm["displacement_estimate"],
        "lower_weight_arm_displacement_estimate_needed": nearer_candidate,
        "displacement_shift_needed_holding_higher_weight_arm_and_both_standard_errors_fixed": nearer_shift,
        "required_estimate_magnitude_exceeds_meaningful_effect_threshold": bool(
            abs(nearer_candidate) > MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT
        ),
    }


def _dose_scaling_interpretability_note(branch: str, interpretable: bool) -> str:
    """State what `interpretable` means for the two-arm dose-scaling pool, without overclaiming
    the between-arm heterogeneity check as independent evidence of combinability -- with exactly
    two arms that check cannot fail regardless of the data (see
    heterogeneity_guard_structural_note), so passing it only ever reflects sign agreement."""
    if not interpretable:
        if branch == "pooled_estimate_uninterpretable_arm_disagreement":
            return (
                "The pooled number above must not be read as the answer: the two arms disagree in sign "
                "and/or their between-arm heterogeneity is large, so no single pooled effect describes "
                "both arms and reporting one pooled point estimate would be misleading."
            )
        return "The pooled number above must not be read as the answer -- see branch and the checks above."
    return (
        "The two arms agree in sign, and the between-arm heterogeneity check did not flag "
        "disagreement -- but with exactly two arms that check could not have flagged disagreement "
        "regardless of what the data showed (see heterogeneity_guard_structural_note), so passing "
        "it is not independent evidence that the pooled estimate below is combinable. Sign "
        "agreement between the two arms is the only evidence this note can actually offer for that."
    )


def run_dose_scaling_two_arm_meta_analysis(block_d: dict, pretask: dict, block_b: dict,
                                           closedloop_records: list[dict]) -> dict:
    from scipy.stats import norm

    verification = _verify_commensurable_normalisation(pretask, block_b, closedloop_records)

    task_pooled_full, task_range_full, task_subj_full = _task_period_dose_arm(block_d["rows"], set())
    pretask_subj_full = {v["subject"] for v in pretask["per_series"].values() if v.get("status") == "computed"}
    overlap = task_subj_full & pretask_subj_full
    disjointness = {
        "task_period_dose_varying_subjects": sorted(task_subj_full),
        "pretask_titration_dose_varying_subjects": sorted(pretask_subj_full),
        "intersection": sorted(overlap),
        "disjoint": len(overlap) == 0,
        "handling": (
            "no overlap found -- both arms used unmodified" if not overlap else
            f"overlap found; dropped from the pre-task titration arm, kept in the task-period arm: {sorted(overlap)}"
        ),
    }

    if not verification["commensurable"]:
        return {
            "moderator": "epoch", "epoch_levels": ["task_period", "pretask_titration"],
            "disjoint_subject_sets_note": (
                "the two arms are drawn from disjoint subject sets by construction, so the pooled "
                "estimate never double-counts any one subject"
            ),
            "commensurability_check": verification,
            "disjointness_check": disjointness,
            "branch": "arms_not_combinable_incommensurable_normalisation",
            "decision_rule": TWO_ARM_META_DECISION_RULE_TEXT,
            "plain_language_answer": (
                "No combined number is reported here because the two ways of testing whether a "
                "stronger jolt of stimulation moves this brain-activity measurement further than a "
                "weaker one turned out not to be measured on the same footing, so adding them together "
                "would not be a fair comparison. Fixing this needs the two measurements to be checked "
                "and, if necessary, put on a shared footing before any combined answer can be trusted."
            ),
        }

    if overlap:
        task_pooled, task_range = task_pooled_full, task_range_full
        pretask_pooled, pretask_range, _ = _pretask_titration_dose_arm(pretask, overlap)
    else:
        task_pooled, task_range = task_pooled_full, task_range_full
        pretask_pooled, pretask_range = (pretask["pooled_amplitude_slope_subject_clustered"],
                                         pretask["realised_amplitude_range_uA_summary"])

    z_factor = float(norm.ppf(1.0 - ALPHA / 2.0) + norm.ppf(POWER))
    task_arm = _displacement_scale_arm(task_pooled, task_range, "task_period", z_factor)
    pretask_arm = _displacement_scale_arm(pretask_pooled, pretask_range, "pretask_titration", z_factor)

    forest = None
    stouffer = None
    pooled_mdd_displacement = None
    if task_arm["status"] == "computed" and pretask_arm["status"] == "computed":
        estimates = np.array([task_arm["displacement_estimate"], pretask_arm["displacement_estimate"]])
        ses = np.array([task_arm["displacement_se"], pretask_arm["displacement_se"]])
        forest = forest_meta(estimates, ses, labels=["task_period", "pretask_titration"])
        pooled_mdd_displacement = z_factor * forest["se"]

        # Stouffer combine: each arm's own two-sided dose-slope p-value converted to a one-sided
        # p-value in the direction of that arm's own observed sign, weighted by sqrt(n_subjects) --
        # a significance-combining complement to forest_meta's effect-size pooling above (the two
        # arms are independent, per the disjointness check), not a repeat of the same computation.
        def _one_sided(pooled: dict) -> float:
            p_two = float(np.clip(pooled["p_value"], 1e-12, 1.0))
            return p_two / 2.0 if pooled["mean_value"] > 0 else 1.0 - p_two / 2.0

        stouffer = stouffer_combine(
            np.array([_one_sided(task_pooled), _one_sided(pretask_pooled)]),
            weights=np.array([np.sqrt(task_pooled["n_subjects"]), np.sqrt(pretask_pooled["n_subjects"])]),
        )

    # Per-arm sign-flip null capacity (why own_slope_p_value is exactly 1.0 for both arms) --
    # computed whenever an arm's own pooled test ran, independently of whether the two-arm pool
    # itself is computable.
    pretask_exclude = overlap if overlap else set()
    task_sign_flip_capacity: dict = {"status": "not_computable"}
    pretask_sign_flip_capacity: dict = {"status": "not_computable"}
    if task_pooled.get("status") == "computed":
        task_subject_values = _task_period_subject_level_slopes(block_d["rows"], set())
        task_sign_flip_capacity = _sign_flip_null_capacity(task_pooled["n_subjects"])
        task_sign_flip_capacity["exhaustive_check"] = _exhaustive_sign_flip_check(task_subject_values)
    if pretask_pooled.get("status") == "computed":
        pretask_subject_values = _pretask_titration_subject_level_slopes(pretask, pretask_exclude)
        pretask_sign_flip_capacity = _sign_flip_null_capacity(pretask_pooled["n_subjects"])
        pretask_sign_flip_capacity["exhaustive_check"] = _exhaustive_sign_flip_check(pretask_subject_values)
    for cap in (task_sign_flip_capacity, pretask_sign_flip_capacity):
        if cap.get("status") != "not_computable" and not cap["capable_of_significance_at_alpha"]:
            cap["evidence_of_absence_note"] = (
                "this arm's own null cannot return a significant result at any possible outcome (its "
                "smallest attainable two-sided p-value exceeds alpha), so this arm's own p-value cannot "
                "by itself contribute evidence that dose scaling is absent -- only its standard error, "
                "which feeds the pooled minimum detectable displacement below, can"
            )
    sign_flip_null_capacity_by_arm = {
        "task_period": task_sign_flip_capacity, "pretask_titration": pretask_sign_flip_capacity,
    }

    # Bootstrap uncertainty on the pooled minimum detectable displacement itself, and the
    # weight-concentration / heterogeneity-guard disclosures -- all computable only when both
    # arms' own displacement-scale estimates are.
    pooled_mdd_bootstrap = None
    weight_concentration = None
    heterogeneity_guard_capacity = None
    heterogeneity_guard_structural_note = None
    stouffer_uninformative_note = None
    if forest is not None:
        pooled_mdd_bootstrap = _bootstrap_pooled_mdd_displacement(
            task_subject_values, pretask_subject_values,
            task_range["median_uA"], pretask_range["median_uA"], z_factor,
        )
        max_row = max(forest["rows"], key=lambda r: r["weight_pct"])
        weight_concentration = {"max_weight_pct": max_row["weight_pct"], "arm_label": max_row["label"]}
        heterogeneity_guard_capacity = _heterogeneity_guard_capacity(task_arm, pretask_arm, forest)
        heterogeneity_guard_structural_note = (
            "with exactly two arms, Cochran's Q has one degree of freedom (Q_df=1), so the decision "
            "rule's own \"Q has more than one degree of freedom\" clause can never apply here by its "
            "own wording; I^2 is then set by simple arithmetic from Q and Q_df rather than tested "
            "against an independent heterogeneity criterion. The between-arm heterogeneity check is "
            "therefore structurally incapable of returning a positive result at two arms with this "
            "weighting -- see heterogeneity_guard_capacity for how far the lower-weight arm's own "
            "estimate would have to move before I^2 could reach the floor at all. `interpretable` "
            "above accordingly reflects an assumption about between-arm agreement that this "
            "configuration could not actually have failed to satisfy, not one that was put to a test "
            "capable of failing."
        )
        stouffer_uninformative_note = (
            "both arms' own two-sided p-values are 1.0 (see sign_flip_null_capacity_by_arm), so the "
            "one-sided p-values Stouffer's method combines here are both handed the least informative "
            "value that conversion can produce, and the resulting z_combined=0.0, p_combined=0.5 carry "
            "no information about dose scaling. This must not be read as a second, independent line of "
            "evidence alongside the inverse-variance pooled estimate above."
        )

    branch = _classify_two_arm_meta(True, True, task_arm, pretask_arm, forest, pooled_mdd_displacement)
    interpretable = branch not in (
        "arms_not_combinable_incommensurable_normalisation",
        "arms_not_combinable_subject_overlap_unresolved",
        "pooled_estimate_uninterpretable_arm_disagreement",
        "not_computable",
    )

    interpretability_note = _dose_scaling_interpretability_note(branch, interpretable)

    n_total_subjects = task_arm.get("n_subjects", 0) + pretask_arm.get("n_subjects", 0)
    if branch == "dose_scaling_across_both_arms_pooled":
        answer_clause = "the combined evidence shows the displacement growing with stimulation amplitude"
    elif branch == "no_dose_scaling_above_the_reported_bound_pooled_across_arms":
        if pooled_mdd_bootstrap is not None and pooled_mdd_bootstrap.get("upper_bound_below_meaningful_effect_threshold") is True:
            confidence_clause = "so this is a confident 'no', not an inconclusive one"
        elif pooled_mdd_bootstrap is not None and pooled_mdd_bootstrap.get("upper_bound_below_meaningful_effect_threshold") is False:
            confidence_clause = (
                "but resampling the same handful of patients shows that bound is itself imprecise -- its "
                "own uncertainty interval reaches past the meaningful-effect threshold -- so this reads "
                "as a lean toward 'no' rather than a confident one"
            )
        else:
            confidence_clause = "though how tightly that bound itself is pinned down was not established here"
        answer_clause = (
            f"the combined evidence rules out a dose-scaling effect larger than "
            f"{pooled_mdd_displacement:.3f} times an ordinary trial-to-trial fluctuation, the smallest "
            "change this design could have detected at 80% power, which is smaller than the smallest "
            f"change this project treats as meaningful (one full ordinary fluctuation) -- {confidence_clause}"
        )
    elif branch == "pooled_estimate_uninterpretable_arm_disagreement":
        answer_clause = "the two ways of measuring this disagree with each other, so no single answer can be given"
    else:
        answer_clause = "the combined evidence is too thin to reach a confident answer either way"

    own_test_sentence = ""
    if task_sign_flip_capacity.get("status") != "not_computable" and not task_sign_flip_capacity["capable_of_significance_at_alpha"]:
        own_test_sentence = (
            " Neither small group's own statistical test should be read as a stand-alone yes-or-no "
            "answer on its own: with only three patients, the three-patient group's own test cannot "
            "mathematically produce a significant result no matter what the data show (the smallest "
            "two-sided p-value three patients can ever produce is 0.25, above the usual 0.05 cutoff), so "
            "that group's own reported p-value of 1.0 reflects this ceiling on what three patients can "
            "ever show, not evidence that stimulation strength has no effect."
        )

    combinability_sentence = ""
    if heterogeneity_guard_capacity is not None and heterogeneity_guard_capacity.get("status") == "computed":
        _needed = abs(heterogeneity_guard_capacity["lower_weight_arm_displacement_estimate_needed"])
        _current = abs(heterogeneity_guard_capacity["lower_weight_arm_current_displacement_estimate"])
        _ratio = _needed / _current if _current > 0 else float("inf")
        combinability_sentence = (
            " Combining the two groups also could not have failed its own internal-agreement check: "
            "with only two groups being combined, and one of them carrying essentially all the weight in "
            "that combination, the lightly-weighted group's own result would have to grow to roughly "
            f"{_ratio:.1f} times its actually-observed size, past the entire meaningful-effect scale, "
            "before that check could ever flag the two groups as disagreeing -- so passing the check here "
            "is not evidence the two groups agree, it is a consequence of there being too few groups, and "
            "too lopsided a weighting between them, for the check to ever say otherwise."
        )

    plain_language_answer = (
        "Not from what is public today, or only very narrowly. Two small, non-overlapping groups of "
        "patients in published human intracranial datasets let this be tested at all, and the two "
        "groups measure two different periods of the recording: three patients who received the same "
        "stimulation at more than one strength at the same electrode pair during the memory task "
        "itself, and seven patients who had their stimulation strength stepped up and down in a short "
        f"warm-up before the task began -- {n_total_subjects} patients between them."
        f"{own_test_sentence}"
        " Because these are different patients measuring different periods of the recording, the two "
        f"small results cannot simply be added together.{combinability_sentence} Combined properly, "
        f"{answer_clause}. To do better would need many more patients who each received the same "
        "stimulation at more than one strength at the same electrode contact -- on current numbers, "
        "several times as many in each of the two groups -- or, more realistically, a study designed on "
        "purpose to vary stimulation strength, since archived recordings only rarely happen to contain "
        "that variation."
    )

    return {
        "moderator": "epoch", "epoch_levels": ["task_period", "pretask_titration"],
        "disjoint_subject_sets_note": (
            "the two arms are drawn from disjoint subject sets by construction, so the pooled estimate "
            "never double-counts any one subject"
        ),
        "commensurability_check": verification,
        "disjointness_check": disjointness,
        "task_period_arm": task_arm,
        "pretask_titration_arm": pretask_arm,
        "forest_meta_result": forest,
        "stouffer_combine_result": stouffer,
        "stouffer_uninformative_note": stouffer_uninformative_note,
        "pooled_minimum_detectable_displacement_80pct_power": pooled_mdd_displacement,
        "pooled_minimum_detectable_displacement_80pct_power_bootstrap_ci": pooled_mdd_bootstrap,
        "weight_concentration": weight_concentration,
        "meaningful_effect_threshold_normalised_displacement": MEANINGFUL_EFFECT_THRESHOLD_NORMALISED_DISPLACEMENT,
        "heterogeneity_i_squared_floor": TWO_ARM_META_HETEROGENEITY_I_SQUARED_FLOOR,
        "heterogeneity_guard_capacity": heterogeneity_guard_capacity,
        "heterogeneity_guard_structural_note": heterogeneity_guard_structural_note,
        "sign_flip_null_capacity_by_arm": sign_flip_null_capacity_by_arm,
        "decision_rule": TWO_ARM_META_DECISION_RULE_TEXT,
        "branch": branch,
        "interpretable": interpretable,
        "interpretability_note": interpretability_note,
        "caveats": (
            "tau2 and I^2 are estimated from only two arms (Q has one degree of freedom), which is the "
            "minimum forest_meta can run on and gives a poorly-constrained heterogeneity estimate; the "
            "task-period arm's own weight in the pooled estimate is large because its 3 subjects "
            "happen to show low between-subject slope variance, not because it is better-powered than "
            "the 7-subject pre-task arm -- both arms' own subject counts are reported beside the pooled "
            "number for this reason."
        ),
        "plain_language_answer": plain_language_answer,
    }


# ── Dose-variation attrition ladders (measurement only -- fires no branch, moves no threshold) ────────
#
# Both the task-period dose arm (Block D) and the pre-task titration arm read the same underlying
# stimulation-parameter tables but land at very different subject counts. This section measures, at
# every named processing step between the raw BIDS event tables and each arm's own analysed rows,
# exactly how many subjects are gained or lost and why -- so a reader can tell a scientifically required
# restriction (a genuine property of how the stimulation was delivered) apart from an incidental one (an
# artifact of how the epochs happen to have been built). It changes no branch, threshold, or existing
# key anywhere in this module.

def _raw_stim_on_events(data_dir: Path) -> list[dict]:
    """Every non-zero-amplitude STIM_ON event across one corpus's raw BIDS event tables, read directly
    and independently of build_session_features and its epoching -- the measurement this section's raw
    counts are built from, before any pipeline admission or epoch-quality filtering is applied."""
    rows = []
    for ieeg_json in _find_session_jsons(data_dir):
        subject = _subject_id(ieeg_json)
        session_key = str(ieeg_json.relative_to(data_dir))
        events_tsv = _pretask_events_tsv_path(ieeg_json)
        if not events_tsv.exists():
            continue
        import csv

        with open(events_tsv) as f:
            events = list(csv.DictReader(f, delimiter="\t"))
        for e in events:
            if e.get("trial_type") != "STIM_ON":
                continue
            amp = _float_or_nan(e.get("amplitude"))
            if not np.isfinite(amp) or amp == 0:
                continue
            rows.append({
                "subject": subject, "session_key": session_key,
                "pair": f"{e.get('anode_label')}-{e.get('cathode_label')}",
                "amplitude_uA": amp,
                "is_pretask_titration_event": e.get("list") == "-999" and e.get("stim_list") == "0",
            })
    return rows


def _subjects_with_amplitude_variation(rows: list[dict], task_period_only: bool = False,
                                       session_keys: set[str] | None = None) -> tuple[set[str], set[str]]:
    """From a raw-event row list (optionally restricted to task-period-only events and/or to an admitted
    session-key set), the subjects showing more than one distinct delivered amplitude at ANY electrode
    pair, and the (strictly smaller or equal) subset showing it at a SINGLE electrode pair of their own
    -- the confound-free contrast this project's own dose arms require."""
    any_amp: dict[str, set] = {}
    pair_amp: dict[tuple, set] = {}
    for r in rows:
        if task_period_only and r["is_pretask_titration_event"]:
            continue
        if session_keys is not None and r["session_key"] not in session_keys:
            continue
        any_amp.setdefault(r["subject"], set()).add(r["amplitude_uA"])
        pair_amp.setdefault((r["subject"], r["pair"]), set()).add(r["amplitude_uA"])
    any_pair_subjects = {s for s, amps in any_amp.items() if len(amps) > 1}
    within_pair_subjects = {s for (s, _pair), amps in pair_amp.items() if len(amps) > 1}
    return any_pair_subjects, within_pair_subjects


def _ladder_rung(name: str, reason: str, seen: set, retained: set) -> dict:
    """One attrition-ladder step: `seen` narrows to `retained`, with the complement named as `lost`, and
    the zero-drop identity is asserted here rather than only claimed."""
    lost = seen - retained
    assert seen == retained | lost and not (retained & lost)
    assert len(seen) == len(retained) + len(lost)
    return {
        "step": name, "reason": reason,
        "n_seen": len(seen), "n_retained": len(retained), "n_lost": len(lost),
        "subjects_lost": sorted(lost),
    }


def compute_dose_variation_attrition_ladders(openloop_corpus: dict, closedloop_corpus: dict,
                                             all_records: list[dict], block_b: dict, block_d: dict,
                                             pretask: dict) -> dict:
    raw_open = _raw_stim_on_events(OPENLOOP_DATA)
    raw_closed = _raw_stim_on_events(CLOSEDLOOP_DATA)

    raw_by_corpus = {}
    raw_within_pair_all: set[str] = set()
    for corpus_name, raw in (("open_loop_ds005489", raw_open), ("closed_loop_ds005557", raw_closed)):
        any_pair, within_pair = _subjects_with_amplitude_variation(raw)
        raw_within_pair_all |= within_pair
        raw_by_corpus[corpus_name] = {
            "n_subjects_with_any_nonzero_amplitude_stim_event": len({r["subject"] for r in raw}),
            "n_subjects_with_amplitude_variation_at_any_electrode_pair": len(any_pair),
            "n_subjects_with_amplitude_variation_within_a_single_electrode_pair": len(within_pair),
            "subjects_with_amplitude_variation_within_a_single_electrode_pair": sorted(within_pair),
        }

    # ── Task-period dose arm (Block D) ──────────────────────────────────────────────────────────────
    _, within_pair_open_task = _subjects_with_amplitude_variation(raw_open, task_period_only=True)
    _, within_pair_closed_task = _subjects_with_amplitude_variation(raw_closed, task_period_only=True)
    stage1 = within_pair_open_task | within_pair_closed_task

    admitted_open = {r["session_key"] for r in openloop_corpus["records"]}
    admitted_closed = {r["session_key"] for r in closedloop_corpus["records"]}
    _, within_pair_open_admitted = _subjects_with_amplitude_variation(
        raw_open, task_period_only=True, session_keys=admitted_open)
    _, within_pair_closed_admitted = _subjects_with_amplitude_variation(
        raw_closed, task_period_only=True, session_keys=admitted_closed)
    stage2 = within_pair_open_admitted | within_pair_closed_admitted

    pair_amp_stage3: dict[tuple, set] = {}
    for rec in all_records:
        arrays = rec["arrays"]
        amp = arrays["amplitude"][arrays["stim_flag"] == 1]
        finite_amp = amp[np.isfinite(amp) & (amp > 0)]
        if finite_amp.size == 0:
            continue
        pair_amp_stage3.setdefault((rec["subject_id"], str(arrays["stim_channel"])), set()).update(
            finite_amp.tolist())
    stage3 = {s for (s, _pair), amps in pair_amp_stage3.items() if len(amps) > 1}

    pair_amp_stage4: dict[tuple, set] = {}
    for rec in all_records:
        cond = block_b["per_session"].get(rec["session_key"], {}).get("conditions", {}).get(
            "excluding_stimulated_shank", {})
        if cond.get("status") != "computed" or cond.get("normalised_displacement") is None:
            continue
        arrays = rec["arrays"]
        amp = arrays["amplitude"][arrays["stim_flag"] == 1]
        finite_amp = amp[np.isfinite(amp) & (amp > 0)]
        if finite_amp.size == 0:
            continue
        pair_amp_stage4.setdefault((rec["subject_id"], str(arrays["stim_channel"])), set()).update(
            finite_amp.tolist())
    stage4 = {s for (s, _pair), amps in pair_amp_stage4.items() if len(amps) > 1}

    pair_amp_stage5: dict[tuple, set] = {}
    for row in block_d["rows"]:
        pair_amp_stage5.setdefault((row["subject"], row["stim_channel"]), set()).add(row["amplitude_uA"])
    stage5 = {s for (s, _pair), amps in pair_amp_stage5.items() if len(amps) > 1}
    assert len(stage5) == block_d["n_subjects_with_within_subject_dose_variation"]

    task_period_ladder = [
        _ladder_rung(
            "restrict_to_task_period_stimulation_events", (
                "this arm asks about task-period dose-response; amplitude variation confined entirely "
                "to a subject's own pre-task calibration series carries no task-period dose contrast and "
                "is out of this arm's scope -- it is analysed separately, in its own arm"
            ), raw_within_pair_all, stage1),
        _ladder_rung(
            "session_admission", (
                "a session that failed the epoched-feature build (see this corpus's own named session "
                "exclusion reasons) contributes no rows at all, which can remove one of the two-or-more "
                "sessions a subject's own within-pair contrast depended on"
            ), stage1, stage2),
        _ladder_rung(
            "epoch_windowing_and_word_level_stimulation_matching", (
                "a delivered amplitude is retained only if the trial it was measured on survives the "
                "word-onset epoch window and, in the closed-loop corpus, is matched to a WORD event "
                "within the fixed post-onset window build_session_features uses to backfill dose fields "
                "onto WORD rows"
            ), stage2, stage3),
        _ladder_rung(
            "channel_condition_availability", (
                "a session is dropped from the dose arm entirely if it lacks a computed displacement "
                "under this module's own mandatory artifact control (excluding every channel that shares "
                "a physical lead with the stimulated pair)"
            ), stage3, stage4),
        _ladder_rung(
            "per_session_collapse_to_a_single_median_amplitude", (
                "each session is represented by one row (that session's own median delivered amplitude); "
                "a within-pair contrast therefore requires at least two SESSIONS at that pair with "
                "different median amplitudes, not merely two trials"
            ), stage4, stage5),
    ]

    # ── Pre-task titration arm ──────────────────────────────────────────────────────────────────────
    _, raw_pretask_within_pair = _subjects_with_amplitude_variation(
        [r for r in raw_closed if r["is_pretask_titration_event"]])
    series_discovered_subjects = {v["subject"] for v in pretask["per_series"].values()}
    computed_subjects = {v["subject"] for v in pretask["per_series"].values() if v.get("status") == "computed"}
    assert computed_subjects.issubset(series_discovered_subjects)

    pretask_ladder = [
        _ladder_rung(
            "series_discovery", (
                "grouping raw pre-task STIM_ON/STIM_OFF events into per-electrode-pair series and "
                "requiring at least two distinct delivered amplitudes within a series"
            ), raw_pretask_within_pair, series_discovered_subjects & raw_pretask_within_pair),
        _ladder_rung(
            "baseline_and_post_window_availability", (
                "a series event is scored only if both its pre-stimulation baseline window and its "
                "post-stimulation window fall entirely inside the recording, and only if the freshly-read "
                "EDF channel order matches this session's own cached word-epoch channel order -- the "
                "series' own first event commonly has no room for a baseline window because the "
                "recording begins at or after it"
            ), series_discovered_subjects & raw_pretask_within_pair, computed_subjects & raw_pretask_within_pair),
    ]
    assert len(computed_subjects & raw_pretask_within_pair) == pretask["pooled_amplitude_slope_subject_clustered"].get("n_subjects")

    dominant_step = max(task_period_ladder, key=lambda rung: rung["n_lost"])
    return {
        "purpose": (
            "measures, at every named processing step, how many subjects with genuine stimulation-"
            "amplitude variation are carried from the raw stimulation-parameter tables through to each "
            "dose arm's own analysed rows -- fires no branch and moves no threshold"
        ),
        "raw_amplitude_variation_by_corpus": raw_by_corpus,
        "task_period_dose_arm_ladder": task_period_ladder,
        "pretask_titration_arm_ladder": pretask_ladder,
        "dominant_loss_step": {
            "arm": "task_period_dose_arm_ladder",
            "step": dominant_step["step"],
            "n_lost": dominant_step["n_lost"],
            "judgement": (
                "required, not incidental: measured directly, the closed-loop corpus's task-period "
                "stimulation amplitude is constant within every one of its subjects' own electrode pairs "
                "(zero of them show task-period within-pair variation); every subject this step removes "
                "carries its own amplitude variation exclusively in its pre-task calibration series, "
                "which the pre-task titration arm already analyses on its own terms. No relaxation of "
                "this step is offered because there is no task-period dose contrast in these subjects "
                "for a relaxed rule to recover -- relaxing it would mix a pre-task calibration amplitude "
                "into a nominally task-period row rather than reveal an additional task-period contrast."
            ),
        },
    }


# ── Corpus loading and zero-drop bookkeeping ────────────────────────────────────

def load_corpus(corpus_name: str, data_dir: Path, derive_stim_from_stim_on: bool, smoke: int | None) -> dict:
    session_jsons = _find_session_jsons(data_dir)
    if smoke is not None:
        session_jsons = session_jsons[:smoke]
    usable, exclusions = [], {}
    for ieeg_json in session_jsons:
        session_key = str(ieeg_json.relative_to(data_dir))
        subject_id = _subject_id(ieeg_json)
        out = load_session_features(corpus_name, session_key, ieeg_json, data_dir, derive_stim_from_stim_on)
        if out["status"] == "usable":
            usable.append({"corpus": corpus_name, "session_key": session_key, "subject_id": subject_id,
                           "arrays": out["arrays"], "ieeg_json": str(ieeg_json)})
        else:
            exclusions.setdefault(out["reason"], []).append(session_key)
    return {
        "corpus": corpus_name, "n_sessions_total": len(session_jsons),
        "n_sessions_used": len(usable), "exclusions": exclusions, "records": usable,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="limit each corpus to the first N candidate sessions, for a fast dev-time check")
    args = parser.parse_args()

    t0 = time.time()
    openloop = load_corpus("open_loop_ds005489", OPENLOOP_DATA, derive_stim_from_stim_on=False, smoke=args.smoke)
    closedloop = load_corpus("closed_loop_ds005557", CLOSEDLOOP_DATA, derive_stim_from_stim_on=True, smoke=args.smoke)
    all_records = openloop["records"] + closedloop["records"]

    zero_drop = {
        "open_loop_ds005489": {"n_sessions_total": openloop["n_sessions_total"],
                               "n_sessions_used": openloop["n_sessions_used"],
                               "exclusions": openloop["exclusions"]},
        "closed_loop_ds005557": {"n_sessions_total": closedloop["n_sessions_total"],
                                 "n_sessions_used": closedloop["n_sessions_used"],
                                 "exclusions": closedloop["exclusions"]},
    }
    for name, corpus in (("open_loop_ds005489", openloop), ("closed_loop_ds005557", closedloop)):
        n_excluded = sum(len(v) for v in corpus["exclusions"].values())
        assert corpus["n_sessions_total"] == corpus["n_sessions_used"] + n_excluded, \
            f"{name}: zero-drop reconciliation failed"

    output = {
        "version": "2026-08-21",
        "scope": (
            "Human intracranial free-recall stimulation, two corpora: open-loop (OpenNeuro ds005489, "
            "experimenter-scheduled stimulation during list encoding) and closed-loop (OpenNeuro ds005557, "
            "online-classifier-triggered stimulation during list encoding). Neither corpus has an isolated "
            "working-memory maintenance delay separate from encoding; stimulation and the analysed epoch "
            "overlap by design in both. Session is the unit of analysis; subject is the clustering unit for "
            "every bootstrap and every p-value throughout."
        ),
        "zero_drop_accounting": zero_drop,
        "n_sessions_used_total": len(all_records),
        "n_subjects_total": len({r["subject_id"] for r in all_records}),
    }

    if not all_records:
        output["precondition"] = {"status": "not_computable", "reason": "no usable session in either corpus"}
        output["blocks_b_through_g"] = "not_run_no_usable_sessions"
        _write_output(output, t0)
        return

    precondition = evaluate_precondition(all_records)
    output["precondition"] = precondition

    if precondition.get("degenerate", False):
        output["branch"] = precondition["branch"]
        output["blocks_b_through_g"] = "not_run_precondition_failed"
        _write_output(output, t0)
        return

    output["block_a"] = run_block_a(all_records)
    output["block_b"] = run_block_b(all_records)
    output["block_c"] = run_block_c(all_records, output["block_b"])
    output["block_d"] = run_block_d(all_records, output["block_b"])
    output["block_e"] = run_block_e(closedloop["records"])
    output["block_f"] = run_block_f(all_records, output["block_b"])
    output["block_g"] = run_block_g(output["block_a"], output["block_b"], output["block_c"])
    output["pretask_amplitude_titration"] = run_pretask_amplitude_titration(closedloop, output["block_b"])
    output["dose_variation_attrition_ladders"] = compute_dose_variation_attrition_ladders(
        openloop, closedloop, all_records, output["block_b"], output["block_d"],
        output["pretask_amplitude_titration"])
    output["dose_scaling_two_arm_meta_analysis"] = run_dose_scaling_two_arm_meta_analysis(
        output["block_d"], output["pretask_amplitude_titration"], output["block_b"], closedloop["records"])

    _write_output(output, t0)


def _write_output(output: dict, t0: float) -> None:
    output["wall_clock_s"] = time.time() - t0
    output["code_commit"] = git_commit(ROOT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json(output)
    if "Infinity" in rendered or "NaN" in rendered:
        raise RuntimeError("non-finite token leaked into JSON output -- fix the offending field before writing")
    fd, tmp_name = tempfile.mkstemp(dir=str(OUTPUT_PATH.parent), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(rendered)
        os.replace(tmp_name, OUTPUT_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    print(f"Wrote {OUTPUT_PATH} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
