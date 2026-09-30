"""corpus_sessions.py -- shared (dataset, structure, session) iteration over the
three human single-unit Sternberg-family corpora with a delay/maintenance
period: DANDI 000469, the canonical 001187/000673 dedup, and DANDI 000574
(DANDI 000574). Each iterator yields region-filtered, firing-rate-QC'd spike lists
plus four epoch onset arrays (baseline, encoding, delay, probe), so that any
downstream analysis needing the same population point cloud -- dimensionality,
displacement scaling, connectivity graphs -- shares one loading path instead
of three independent re-implementations of NWB field names, session-identity
deduplication, and QC floors.

Two non-human corpora with a delay period are loaded here too, and both hand
back already-binned delay-epoch counts of shape (trials, units, bins) rather
than spike lists, because neither release ships spike times in a session
clock: mouse ALM (``iter_alm``) and the multi-object macaque frontal-cortex
corpus (``iter_watters``).

Epoch anchoring:
  - DANDI 000469 and 001187 (001187 is the deduplicated dandi_001187/000673
    primary release; see run_human_drift_spine_001187_000673.py's
    canonical_sessions): timestamps_FixationCross, timestamps_Encoding1,
    timestamps_Maintenance, timestamps_Probe are present in both releases'
    trial tables (verified directly against the NWB files).
  - DANDI 000574: no named per-epoch timestamp fields exist in its
    trial table. run_human_drift_spine_000574.py's docstring documents the
    task's fixed relative structure (fixation [-6,-5] s, encoding [-5,-3] s,
    maintenance [-3,0] s relative to the probe, i.e. maintenance onset =
    trial start_time + 3.0 s) -- reused here to derive all four epoch onsets
    from start_time.
  - DANDI 000004 (Chandravadia new/old recognition, ``iter_dandi_000004``):
    its trial table carries a genuine maintenance interval (delay1_time to
    delay2_time, ~2.2 s median on the admitted recognition trials, after a
    ~1.0 s stim_on/stim_off encoding period). Region labels resolve through
    the existing `nwb_hemisphere_prefixed_structure` parser (its electrode
    `location` field uses a "{Hemisphere} {Structure}" convention, e.g.
    "Right Hippocampus"). Hippocampus and amygdala are yielded as separate
    entries and are never pooled with each other.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
_repo_root = Path(__file__).resolve().parents[1]

from spike_pipeline import (  # noqa: E402
    ANATOMICAL_REGIONS,  # noqa: F401
    BORAN_ANATOMICAL_REGIONS,  # noqa: F401
    BORAN_REGIONS_WITH_POOLED,
    MIN_UNITS_PER_REGION,
    REGIONS_WITH_POOLED,
    filter_units_by_region,
    load_spike_times,
    low_rate_unit_mask,
    resolve_unit_regions,
)
from preprocessing import high_gamma_power  # noqa: E402
from run_watters_source_replication import add_behavior_columns  # noqa: E402
from project_config import data_root as configured_data_root, load_dataset_registry  # noqa: E402
from statistics import CONTENT_LABEL_K_CLASSES
from preprocessing import load_events
from stimulation_events import _bin_averaged, channel_condition_masks, compute_stimulation_displacement
from spike_pipeline import build_psth, BIN_MS
from statistics import unit_direction_vectors, MIN_TRIALS_WITH_DEFINED_DIRECTION
from info_decoding import _axis_stability, _linear_detrend_activity, _residual_rows, _unit_residual_matrix
from subspace_identity import leading_eigenvector
from project_config import dataset_path
from statistics import MIN_ERROR_TRIALS_FOR_REACHABILITY
from stimulation_response_estimator import rate_free_state_deviation
from info_decoding import MAX_SESSIONS_ENV_VAR, _previous_label, _session_core
from spike_pipeline import _counts_from_spikes
from spike_pipeline import delay_counts
from spike_pipeline import FrozenPSTHTransform
from info_decoding import WATTERS_RECOVERABILITY_K_CLASSES
from info_decoding import session_subtractive_test
from info_decoding import HUMAN_BIN_MS
from preprocessing import bandpass_filter, load_boran_nwb
from info_decoding import FIELD_BAND_HI_HZ, FIELD_BAND_LO_HZ, FIELD_MAINTENANCE_WINDOW_S
from state_persistence import trial_amplitude_covariates
from run_watters_source_replication import CORRECT_REPORT_DISTANCE_THRESHOLD
import warnings
from statistics import stable_seed

MIN_TRIALS = 20
MIN_UNITS_POOLED = 15
EPOCH_WINDOWS_S = {"baseline": 0.5, "encoding": 0.5, "delay": 2.3, "probe": 0.5}
DANDI_000574_PROBE_OFFSET_S = 6.0
BORAN_EPOCH_WINDOWS_S = {"baseline": 1.0, "encoding": 2.0, "delay": 3.0, "probe": 0.5}
DANDI_000004_EPOCH_WINDOWS_S = {"delay": 1.0}
DANDI_000004_RESPONSE_RANGE = np.arange(31, 37)

ALM_WINDOW_S = 2.0
ALM_MIN_UNITS = 15
ALM_MIN_TRIALS_PER_ARM = 8
ALM_MIN_UNIT_RATE_HZ = 0.2


def data_root() -> Path:
    """Compatibility wrapper around the shared project configuration."""
    return configured_data_root()


def region_filtered_units(spike_lists_all: list, unit_regions: np.ndarray, region: str, delay_onset: np.ndarray, delay_window: float) -> list | None:
    spike_lists = filter_units_by_region(spike_lists_all, unit_regions, region)
    rate_mask = low_rate_unit_mask(spike_lists, delay_onset, delay_window)
    spike_lists = [spikes for spikes, keep in zip(spike_lists, rate_mask) if keep]
    min_units = MIN_UNITS_POOLED if region == "pooled" else MIN_UNITS_PER_REGION
    if len(spike_lists) < min_units:
        return None
    return spike_lists


def iter_dandi_000469(root: Path):
    """Yields dict(dataset, patient, session, structure, spike_lists, epoch_onsets, epoch_windows)."""
    directory = root / "000469"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*_ses-2_ecephys+image.nwb")):
            with h5py.File(path, "r") as handle:
                if "units" not in handle:
                    continue
                spike_lists_all = load_spike_times(handle)
                unit_regions = resolve_unit_regions(handle)["region"]
                trials = handle["intervals/trials"]
                accuracy = trials["response_accuracy"][:].astype(bool)
                loads = trials["loads"][:].astype(int)
                item_ids = trials["loadsEnc1_PicIDs"][:].astype(int)
                t_fix = trials["timestamps_FixationCross"][:]
                t_enc1 = trials["timestamps_Encoding1"][:]
                t_maint = trials["timestamps_Maintenance"][:]
                t_probe = trials["timestamps_Probe"][:]
            keep = (loads == 1) & accuracy
            if keep.sum() < MIN_TRIALS:
                continue
            epoch_onsets = {"baseline": t_fix[keep], "encoding": t_enc1[keep], "delay": t_maint[keep], "probe": t_probe[keep]}
            for region in REGIONS_WITH_POOLED:
                spike_lists = region_filtered_units(spike_lists_all, unit_regions, region, epoch_onsets["delay"], EPOCH_WINDOWS_S["delay"])
                if spike_lists is None:
                    continue
                yield {
                    "dataset": "dandi_000469", "patient": subject_dir.name, "session": path.stem, "structure": region,
                    "spike_lists": spike_lists, "epoch_onsets": epoch_onsets, "epoch_windows": EPOCH_WINDOWS_S,
                    "item_ids": item_ids[keep], "item_id_field": "loadsEnc1_PicIDs",
                }


def iter_dandi_001187(root: Path):
    from run_human_drift_spine_001187_000673 import canonical_sessions  # deferred: avoids a
    # module-load cycle with run_human_drift_spine_001187_000673's own back-import of _trial_group

    for meta in canonical_sessions():
        if meta["primary_release"] != "001187":
            continue  # only the release with the full named timestamp fields
        path = root / meta["primary_path"]
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            if "units" not in handle:
                continue
            spike_lists_all = load_spike_times(handle)
            unit_regions = resolve_unit_regions(handle)["region"]
            trials = _trial_group(handle, "001187")
            accuracy = trials["response_accuracy"][:].astype(bool)
            item_ids = trials["PicIDs_Encoding1"][:].astype(int)
            t_fix = trials["timestamps_FixationCross"][:]
            t_enc1 = trials["timestamps_Encoding1"][:]
            t_maint = trials["timestamps_Maintenance"][:]
            t_probe = trials["timestamps_Probe"][:]
        keep = accuracy
        if keep.sum() < MIN_TRIALS:
            continue
        epoch_onsets = {"baseline": t_fix[keep], "encoding": t_enc1[keep], "delay": t_maint[keep], "probe": t_probe[keep]}
        for region in REGIONS_WITH_POOLED:
            spike_lists = region_filtered_units(spike_lists_all, unit_regions, region, epoch_onsets["delay"], EPOCH_WINDOWS_S["delay"])
            if spike_lists is None:
                continue
            yield {
                "dataset": "dandi_001187", "patient": meta["patient"], "session": path.stem, "structure": region,
                "spike_lists": spike_lists, "epoch_onsets": epoch_onsets, "epoch_windows": EPOCH_WINDOWS_S,
                "item_ids": item_ids[keep], "item_id_field": "PicIDs_Encoding1",
            }


def iter_dandi_000574(root: Path):
    directory = root / "000574"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(path, "r") as handle:
                if "units" not in handle:
                    continue
                spike_lists_all = load_spike_times(handle)
                unit_regions = resolve_unit_regions(handle, "nwb_boran_brainnetome_hybrid")["region"]
                trials = handle["intervals/trials"]
                artifact = trials["artifact"][:].astype(bool)
                correct = trials["correct"][:].astype(bool)
                start_time = trials["start_time"][:]
            keep = (~artifact) & correct
            if keep.sum() < MIN_TRIALS:
                continue
            start = start_time[keep]
            epoch_onsets = {"baseline": start + 0.0, "encoding": start + 1.0, "delay": start + 3.0, "probe": start + DANDI_000574_PROBE_OFFSET_S}
            for region in BORAN_REGIONS_WITH_POOLED:
                spike_lists = region_filtered_units(spike_lists_all, unit_regions, region, epoch_onsets["delay"], BORAN_EPOCH_WINDOWS_S["delay"])
                if spike_lists is None:
                    continue
                yield {
                    "dataset": "dandi_000574", "patient": subject_dir.name, "session": path.stem, "structure": region,
                    "spike_lists": spike_lists, "epoch_onsets": epoch_onsets, "epoch_windows": BORAN_EPOCH_WINDOWS_S,
                    "item_ids": None, "item_id_field": None,
                    "item_id_unavailable_reason": "set_letters is 'not available' on every trial in the public "
                                                   "NWB release (see run_human_drift_spine_000574.py's own "
                                                   "item_identity_available=False finding); no per-trial item "
                                                   "identity label exists for this corpus.",
                }


def iter_dandi_000004(root: Path):
    from run_dandi_000004_recognition_generalization import recognition_correct

    directory = root / "000004"
    required_columns = (
        "stim_phase", "new_old_labels_recog", "response_value", "response_time",
        "delay1_time", "delay2_time", "stimCategory",
    )
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(path, "r") as handle:
                if "units" not in handle or "intervals/trials" not in handle:
                    continue
                trials = handle["intervals/trials"]
                if not all(column in trials for column in required_columns):
                    continue
                spike_lists_all = load_spike_times(handle)
                unit_regions = resolve_unit_regions(handle, "nwb_hemisphere_prefixed_structure")["region"]
                phase = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in trials["stim_phase"][:]])
                labels = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in trials["new_old_labels_recog"][:]])
                responses = trials["response_value"][:].astype(float)
                response_time = trials["response_time"][:].astype(float)
                delay1 = trials["delay1_time"][:].astype(float)
                delay2 = trials["delay2_time"][:].astype(float)
                stim_category = trials["stimCategory"][:].astype(float)
            candidate = (
                (phase == "recog") & np.isin(labels, ("0", "1")) & np.isin(responses, DANDI_000004_RESPONSE_RANGE)
                & np.isfinite(delay1) & np.isfinite(delay2) & np.isfinite(response_time)
            )
            if candidate.sum() < MIN_TRIALS:
                continue
            accuracy = recognition_correct(labels[candidate], responses[candidate])
            epoch_onsets = {"delay": delay1[candidate]}
            memorandum_content = stim_category[candidate]
            response_time_admitted = response_time[candidate]
            for region in ("hippocampus", "amygdala"):
                spike_lists = region_filtered_units(
                    spike_lists_all, unit_regions, region, epoch_onsets["delay"], DANDI_000004_EPOCH_WINDOWS_S["delay"]
                )
                if spike_lists is None:
                    continue
                yield {
                    "dataset": "dandi_000004", "patient": subject_dir.name, "session": path.stem, "structure": region,
                    "spike_lists": spike_lists, "epoch_onsets": epoch_onsets, "epoch_windows": DANDI_000004_EPOCH_WINDOWS_S,
                    "memorandum_content": memorandum_content, "memorandum_content_field": "stimCategory",
                    "accuracy": accuracy, "accuracy_field": "response_value",
                    "response_time": response_time_admitted,
                }


def alm_data_directory(root: Path) -> Path:
    """Directory of mouse ALM perturbation sessions (mouse ALM, single structure by design)."""
    config = load_dataset_registry()
    local_path = config["datasets"]["inagaki_alm5"]["local_path"]
    return root / local_path / "RandomDelayTask" / "withPerturbation"


def _alm_trial_condition(trial_types: np.ndarray) -> np.ndarray:
    return np.array([0 if str(value).lower().startswith("l") else 1 for value in trial_types], dtype=int)


def _alm_build_counts(units: np.ndarray, delay_start: np.ndarray, trial_indices: np.ndarray,
                       bin_ms: float, window_s: float) -> np.ndarray:
    """Raw (unstandardized) spike counts, delay-onset-aligned: (trials, units, bins)."""
    starts = np.arange(0.0, window_s, bin_ms / 1000.0)
    counts = np.zeros((len(trial_indices), len(units), len(starts)), dtype=float)
    row_for_trial = {int(trial): row for row, trial in enumerate(trial_indices)}
    for unit_index, unit in enumerate(units):
        spike_times = np.asarray(unit.SpikeTimes, dtype=float).reshape(-1)
        spike_trials = np.asarray(unit.Trial_idx_of_spike, dtype=int).reshape(-1) - 1
        for trial in trial_indices:
            row = row_for_trial[int(trial)]
            relative = spike_times[spike_trials == trial] - delay_start[trial]
            counts[row, unit_index], _ = np.histogram(relative, bins=np.append(starts, window_s))
    return counts


def load_alm_raw_session(path: Path, bin_ms: float = 100.0, window_s: float = ALM_WINDOW_S,
                          require_both_arms: bool = True) -> dict | None:
    """Raw (unstandardized) delay-epoch spike counts for one ALM session, split
    into control and photoinhibition-perturbation trial arms.

    Shared by every analysis needing this session's delay-epoch population
    counts (the attractor-recovery gate, the observability census) so the
    NWB/eligibility/QC logic -- trial-range restriction, minimum delay
    duration, minimum-firing-rate unit exclusion -- lives in one place.
    Returns None if the session fails the minimum trial or unit count.

    ``require_both_arms`` gates the perturbation arm's own minimum-trial
    floor. A widened ``window_s`` (e.g. to match a shorter human recording
    epoch) shrinks the pool of trials with a long-enough delay on both arms
    at once; callers that only need the control arm (the matched-power
    comparison, which never touches the perturbation trials) should pass
    ``False`` so a session is not dropped for a perturbation-arm shortfall
    that does not bear on what they are computing.
    """
    units = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["unit"])
    behavior = units[0].Behavior
    trial_info = units[0].Trial_info
    trial_type = np.asarray(behavior.Trial_types_of_response_vector, dtype=int).reshape(-1)
    stimulation = np.asarray(behavior.stim_trial_vector, dtype=int).reshape(-1)
    delay_duration = np.asarray(behavior.delay_dur, dtype=float).reshape(-1)
    delay_duration_id = np.asarray(behavior.delay_dur_id, dtype=int).reshape(-1)
    delay_start = np.asarray(behavior.Delay_start, dtype=float).reshape(-1)
    condition = _alm_trial_condition(np.asarray(trial_info.Trial_types).reshape(-1))
    start, stop = np.asarray(trial_info.Trial_range_to_analyze, dtype=int).reshape(-1) - 1
    eligible = np.arange(start, stop + 1)
    eligible = eligible[(trial_type[eligible] < 5) & (delay_duration[eligible] >= window_s)]
    control_trials = eligible[stimulation[eligible] == 0]
    perturb_trials = eligible[stimulation[eligible] > 1]
    arm_floor = min(len(control_trials), len(perturb_trials)) if require_both_arms else len(control_trials)
    if min(arm_floor, len(units)) < ALM_MIN_TRIALS_PER_ARM:
        return None
    all_trials = np.concatenate((control_trials, perturb_trials))
    counts = _alm_build_counts(units, delay_start, all_trials, bin_ms, window_s)
    rates = counts[: len(control_trials)].sum(axis=(0, 2)) / (len(control_trials) * window_s)
    unit_mask = rates >= ALM_MIN_UNIT_RATE_HZ
    counts = counts[:, unit_mask]
    if np.sum(unit_mask) < ALM_MIN_UNITS:
        return None
    return {
        "mouse": path.stem.split("_")[0],
        "n_units_after_rate_qc": int(np.sum(unit_mask)),
        "n_control_trials": int(len(control_trials)),
        "n_perturb_trials": int(len(perturb_trials)),
        "control_condition": condition[control_trials],
        "perturb_condition": condition[perturb_trials],
        "control_delay_duration_id": delay_duration_id[control_trials],
        "control_counts": counts[: len(control_trials)],
        "perturb_counts": counts[len(control_trials):],
        # Trial_types_of_response_vector: the raw response code (1-4) each trial was scored with, kept
        # per arm alongside the counts it was already computed from -- callers that need trial outcome
        # (not just condition/instructed side) would otherwise have to re-derive control_trials/
        # perturb_trials themselves and risk a different trial set from the one the counts above use.
        "control_response_code": trial_type[control_trials],
        "perturb_response_code": trial_type[perturb_trials],
    }


def iter_alm(root: Path, bin_ms: float = 100.0, window_s: float = ALM_WINDOW_S):
    """Delay-epoch raw population counts for every eligible ALM session,
    unperturbed (control) trials only -- the calibration comparison for
    the observability census. Single structure by design (config/datasets.json),
    so ``structure`` is reported as "pooled".
    """
    directory = alm_data_directory(root)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.mat")):
        session = load_alm_raw_session(path, bin_ms=bin_ms, window_s=window_s)
        if session is None:
            continue
        yield {
            "dataset": "inagaki_alm5", "patient": session["mouse"], "session": path.stem,
            "structure": "pooled", "epoch": "delay", "counts": session["control_counts"],
            "bin_ms": bin_ms, "n_units": session["n_units_after_rate_qc"],
            "condition": session["control_condition"],
            "delay_duration_id": session["control_delay_duration_id"],
            "delay_duration_id_field": "delay_dur_id (categorical delay length, unpredictable within trial)",
            "item_id_field": "Trial_types_of_response_vector (instructed lick direction, left/right)",
        }


WATTERS_RAW_BIN_MS = 10.0
WATTERS_DELAY_WINDOW_S = 1.0
WATTERS_CACHE_ORIGIN_BEFORE_STIMULUS_S = 0.2
WATTERS_TASK_VARIANTS = ("ring", "triangle")
WATTERS_QUALITY_TIERS = ("good", "mua")
WATTERS_MIN_UNITS = 15
WATTERS_MIN_TRIALS = 20


def watters_directories(root: Path) -> tuple[Path, Path]:
    """(per-trial spike cache, behaviour CSV directory) of the multi-object
    macaque corpus, resolved through config/datasets.json rather than a
    hard-coded path."""
    config = load_dataset_registry()
    configured = root / config["datasets"]["watters_2026"]["local_path"]
    corpus = configured.parent if configured.name == "data_for_modeling" else configured
    spikes = corpus / "data_for_modeling" / "data_for_modeling" / "spikes_per_trial"
    behavior = corpus / "data_for_figures" / "data_for_figures" / "behavior_processing"
    return spikes, behavior


def watters_behaviour(root: Path) -> pd.DataFrame:
    """One row per completed trial of both task variants, with the corpus's
    derived graded report deviation, trial correctness and reaction time
    attached by run_watters_source_replication.add_behavior_columns (the
    report is a continuous saccade, so its deviation from the cued position
    is the graded quantity and correctness is a threshold on it; none of the
    three is a raw column), plus the cued object's polar angle -- the
    continuous memorandum -- and the per-trial delay-epoch timestamps.

    Indexed by (subject, session, trial_num), which is also the key the
    per-trial spike cache's trial numbers join on."""
    _, behavior_dir = watters_directories(root)
    frames = []
    for variant in WATTERS_TASK_VARIANTS:
        frame = add_behavior_columns(pd.read_csv(behavior_dir / f"{variant}.csv"), variant)
        thetas = [frame[f"object_{i}_theta"].to_numpy(dtype=float) for i in range(3)]
        cued = np.choose(frame.target_object_index.to_numpy(dtype=int), thetas)
        frame["cued_theta"] = np.mod(cued, 2.0 * np.pi)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    return table.set_index(["subject", "session", "trial_num"], drop=False).sort_index()


def watters_session_dates(root: Path) -> list[tuple[str, str, str]]:
    """(animal, session date, task variant) for every behavioural session
    date in the corpus, including any with no per-trial spike cache -- the
    denominator zero-drop accounting has to reconcile against."""
    _, behavior_dir = watters_directories(root)
    dates: list[tuple[str, str, str]] = []
    for variant in WATTERS_TASK_VARIANTS:
        frame = pd.read_csv(behavior_dir / f"{variant}.csv", usecols=["subject", "session"])
        dates.extend((str(a), str(s), variant) for a, s in
                     sorted(set(zip(frame.subject, frame.session))))
    return sorted(dates)


def _watters_unit_index(session_dir: Path, quality_tiers: tuple[str, ...]) -> list[dict]:
    """Every unit under one session's probe/quality tree, with its trial
    list already read. Pooled across probes: the shared electrode table
    gives every electrode the location `unknown`, so this corpus supports a
    single pooled population and no area-resolved split."""
    units = []
    for probe_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
        for quality in quality_tiers:
            tier_dir = probe_dir / quality
            if not tier_dir.is_dir():
                continue
            for path in sorted(tier_dir.glob("*_trials.pkl")):
                unit_id = path.name[: -len("_trials.pkl")]
                with open(path, "rb") as handle:
                    trials = np.asarray(pickle.load(handle), dtype=int)
                units.append({"probe": probe_dir.name, "quality": quality, "unit": unit_id,
                              "counts_path": tier_dir / f"{unit_id}_spike_counts.pkl", "trials": trials})
    return units


def load_watters_session(root: Path, animal: str, session_date: str, behaviour: pd.DataFrame,
                          bin_ms: float = 100.0, quality_tiers: tuple[str, ...] = WATTERS_QUALITY_TIERS) -> dict:
    """Delay-epoch population counts for one session of the multi-object
    macaque corpus: (trials, units, bins) raw spike counts over the fixed
    1.0 s maintenance period, the same array shape every other corpus in
    this module hands to the persistence and content estimators.

    The per-trial cache stores each unit's whole trial as a vector of 10 ms
    spike counts whose first bin starts
    ``WATTERS_CACHE_ORIGIN_BEFORE_STIMULUS_S`` before stimulus onset, so the
    maintenance window is cut per trial from that trial's own delay-onset
    timestamp rather than at a fixed offset -- the designed delay is 1.0 s
    but a small fraction of trials run long, and a fixed offset would put
    those trials' windows in the wrong place.

    Units are pooled across probes and kept only if their trial list spans
    every analysed trial, so the returned tensor has no imputed entries.
    Both quality tiers are returned together, labelled per unit in
    ``unit_quality``, so a caller can subset to well-isolated single units
    without a second pass over the cache.

    Always returns a dict; ``status`` is "loaded" or the reason the session
    yields no usable tensor, so every session seen appears in the caller's
    accounting."""
    spikes_dir, _ = watters_directories(root)
    session_dir = spikes_dir / animal / session_date
    base = {"animal": animal, "session_date": session_date,
            "session": f"{animal}_{session_date}", "dataset": "watters_2026", "structure": "pooled"}
    if not session_dir.is_dir():
        return {**base, "status": "no_spike_cache_for_this_behavioural_session_date"}
    if (animal, session_date) not in behaviour.index.droplevel(2).unique():
        return {**base, "status": "no_completed_behavioural_trials"}

    units = _watters_unit_index(session_dir, quality_tiers)
    if not units:
        return {**base, "status": "no_units_in_requested_quality_tiers"}

    reference = max(units, key=lambda u: (len(u["trials"]), -len(u["unit"])))["trials"]
    union = sorted(set().union(*[set(u["trials"].tolist()) for u in units]))
    spanning = [u for u in units if set(reference.tolist()).issubset(set(u["trials"].tolist()))]

    rows = behaviour.loc[(animal, session_date)]
    factor = int(round(bin_ms / WATTERS_RAW_BIN_MS))
    n_raw = int(round(WATTERS_DELAY_WINDOW_S * 1000.0 / WATTERS_RAW_BIN_MS))

    keep_trials, start_bins = [], []
    dropped = {"delay_shorter_than_window": 0, "window_past_end_of_cached_trial": 0,
               "no_completed_behaviour_row": 0}
    with open(spanning[0]["counts_path"], "rb") as handle:
        reference_counts = pickle.load(handle)
    reference_rows = {int(t): i for i, t in enumerate(spanning[0]["trials"])}
    for trial in reference.tolist():
        if trial not in rows.index:
            dropped["no_completed_behaviour_row"] += 1
            continue
        row = rows.loc[trial]
        origin = float(row.time_stimulus_onset) - WATTERS_CACHE_ORIGIN_BEFORE_STIMULUS_S
        start = int(round((float(row.time_delay_onset) - origin) * 1000.0 / WATTERS_RAW_BIN_MS))
        # The display quantises event times to the screen refresh, so a trial's
        # measured delay can fall a fraction of a millisecond short of the
        # designed 1.0 s. Half a cache bin of slack keeps those trials; a real
        # short delay is short by a refresh interval, far outside it.
        if float(row.time_cue_onset) - float(row.time_delay_onset) < WATTERS_DELAY_WINDOW_S - WATTERS_RAW_BIN_MS / 2000.0:
            dropped["delay_shorter_than_window"] += 1
            continue
        if start < 0 or start + n_raw > len(reference_counts[reference_rows[int(trial)]]):
            dropped["window_past_end_of_cached_trial"] += 1
            continue
        keep_trials.append(int(trial))
        start_bins.append(start)

    if len(keep_trials) < WATTERS_MIN_TRIALS:
        return {**base, "status": "too_few_trials_with_a_complete_delay_window",
                "n_trials": len(keep_trials), "n_units_seen": len(units)}
    if len(spanning) < WATTERS_MIN_UNITS:
        return {**base, "status": "too_few_units_spanning_every_trial",
                "n_trials": len(keep_trials), "n_units_seen": len(units), "n_units_spanning": len(spanning)}

    start_bins = np.asarray(start_bins, dtype=int)
    counts = np.zeros((len(keep_trials), len(spanning), n_raw // factor), dtype=float)
    for u_index, unit in enumerate(spanning):
        with open(unit["counts_path"], "rb") as handle:
            per_trial = pickle.load(handle)
        row_for_trial = {int(t): i for i, t in enumerate(unit["trials"])}
        for t_index, trial in enumerate(keep_trials):
            vector = per_trial[row_for_trial[trial]]
            window = np.asarray(vector[start_bins[t_index]:start_bins[t_index] + n_raw], dtype=float)
            counts[t_index, u_index] = window.reshape(-1, factor).sum(axis=1)

    trial_rows = rows.loc[keep_trials]
    return {
        **base, "status": "loaded", "epoch": "delay", "counts": counts, "bin_ms": float(bin_ms),
        "window_s": WATTERS_DELAY_WINDOW_S,
        "n_units": len(spanning), "n_units_seen": len(units), "n_units_in_union_but_not_spanning": len(units) - len(spanning),
        "n_trials_in_reference_unit": int(len(reference)), "n_trials_in_union_over_units": len(union),
        "unit_quality": np.array([u["quality"] for u in spanning]),
        "unit_probe": np.array([u["probe"] for u in spanning]),
        "trial_num": np.asarray(keep_trials, dtype=int),
        "delay_onset_raw_bin": start_bins,
        "task_variant": str(trial_rows.task.iloc[0]),
        "num_objects": trial_rows.num_objects.to_numpy(dtype=int),
        "correct": trial_rows.correct.to_numpy(dtype=bool),
        "report_deviation": trial_rows.report_deviation.to_numpy(dtype=float),
        "cued_theta": trial_rows.cued_theta.to_numpy(dtype=float),
        "reaction_time_ms": trial_rows.reaction_time_ms.to_numpy(dtype=float),
        "trials_dropped_by_reason": dropped,
    }


def iter_watters(root: Path, bin_ms: float = 100.0, quality_tiers: tuple[str, ...] = WATTERS_QUALITY_TIERS):
    """Every behavioural session date of the multi-object macaque corpus,
    loaded or refused with a reason -- the caller sees all of them, so
    seen = loaded + refused reconciles without a second enumeration."""
    behaviour = watters_behaviour(root)
    for animal, session_date, variant in watters_session_dates(root):
        session = load_watters_session(root, animal, session_date, behaviour,
                                       bin_ms=bin_ms, quality_tiers=quality_tiers)
        yield {**session, "behavioural_task_variant": variant}


SCALP_EEG_DELAY_BINS = 6
SCALP_EEG_BROADBAND_HZ = (1.0, 40.0)


def _scalp_delay_bin_power(data_tc: np.ndarray, srate: float, onset_samples: np.ndarray,
                            window_s: float, n_bins: int) -> np.ndarray:
    """(trials, channels, bins) broadband Hilbert-envelope power, uniformly binned over a fixed
    post-onset window -- the scalp-EEG analogue of the binned spike-count tensors iter_alm and
    iter_watters hand back, built from preprocessing.high_gamma_power (bandpass then Hilbert
    envelope) with its band widened to 1-40 Hz and its smoothing disabled. Filtered one short,
    padded per-trial buffer at a time (not the whole multi-thousand-second continuous recording at
    once, as an earlier version of this function did) -- the same per-trial-buffer discipline
    scripts/run_ds005034_tacs_aftereffect.py already uses, and the reason this function needed
    only a few MB per trial rather than the several GB a whole-session bandpass+Hilbert holds."""
    bin_samples = int(round(window_s * srate / n_bins))
    window_samples = bin_samples * n_bins
    pad = int(round(0.5 * srate))
    n_channels = data_tc.shape[1]
    out = np.full((len(onset_samples), n_channels, n_bins), np.nan)
    for trial, start in enumerate(onset_samples):
        lo, hi = start - pad, start + window_samples + pad
        if lo < 0 or hi > data_tc.shape[0]:
            continue
        power = high_gamma_power(data_tc[lo:hi], srate, lo=SCALP_EEG_BROADBAND_HZ[0],
                                  hi=SCALP_EEG_BROADBAND_HZ[1], smooth_ms=0.0)
        trimmed = power[pad:pad + window_samples]
        out[trial] = trimmed.reshape(n_bins, bin_samples, n_channels).mean(axis=1).T
    return out


def iter_ds005034(root: Path):
    """Yields per-session dicts for the healthy-participant scalp-EEG theta-tACS working-memory
    release (config/datasets.json 'ds005034'): one entry per participant/session with an actual
    memory-task EEG recording on disk. Delay window and onset offset (0.5-6.5 s post task/load
    marker) reuse scripts/run_ds005034_tacs_aftereffect.py's own convention, and its EVENT_MAP /
    load_events are imported unchanged rather than re-implemented. No per-trial item identity or
    behavioural outcome exists in this corpus's public release (see config/datasets.json)."""
    from run_ds005034_tacs_aftereffect import load_events

    directory = root / "ds005034"
    for subject_dir in sorted(directory.glob("sub-*")):
        for session in ("sham", "verum"):
            eeg_dir = subject_dir / f"ses-{session}" / "eeg"
            set_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_eeg.set"
            if not set_path.is_file():
                continue
            events_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_events.tsv"
            channels_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_channels.tsv"
            events = load_events(events_path)
            if len(events) < MIN_TRIALS:
                continue
            channel_names = pd.read_csv(channels_path, sep="\t")["name"].tolist()
            payload = loadmat(set_path, variable_names=("data", "srate"), squeeze_me=True)
            data_tc = np.asarray(payload["data"], dtype=np.float64).T
            srate = float(payload["srate"])
            onset_samples = np.array([int(round((event["onset"] + 0.5) * srate)) for event in events])
            counts = _scalp_delay_bin_power(data_tc, srate, onset_samples, window_s=6.0, n_bins=SCALP_EEG_DELAY_BINS)
            keep = np.isfinite(counts).all(axis=(1, 2))
            if keep.sum() < MIN_TRIALS:
                continue
            yield {
                "dataset": "ds005034", "patient": subject_dir.name,
                "session": f"{subject_dir.name}_ses-{session}", "structure": "scalp",
                "channel_names": channel_names, "counts": counts[keep],
                "task_condition": np.array([event["task"] for event in events])[keep],
                "load": np.array([event["load"] for event in events], dtype=float)[keep],
                "stimulation_session": session, "memorandum_content": None, "accuracy": None,
                "item_id_unavailable_reason": "no per-trial item identity is recorded in this "
                                               "corpus's public BIDS release",
                "accuracy_unavailable_reason": "no trial-level behavioural outcome is recorded in "
                                                "this corpus's public BIDS release; it exists only "
                                                "on a separate OSF repository (config/datasets.json)",
            }


def iter_ds006848(root: Path):
    """Yields per-session dicts for the healthy-participant scalp-EEG verbal working-memory
    (digit-span) release (config/datasets.json 'ds006848'): one entry per participant, all 30 of
    which carry both an EEG recording and trial-level behaviour. The 6.0 s retention window is
    exact on every trial (Retention_* to Digits_Retrieval, verified against the raw event table,
    no intervening events). memorandum_content is the first digit of the presented sequence."""
    import mne

    directory = root / "ds006848"
    retention_codes = {28: "Simultaneous", 40: "Fast", 60: "Fast+delay", 100: "Slow"}
    for subject_dir in sorted(directory.glob("sub-*")):
        vhdr_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_eeg.vhdr"
        beh_path = subject_dir / "beh" / f"{subject_dir.name}_task-verbalwm_beh.tsv"
        events_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_events.tsv"
        channels_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_channels.tsv"
        if not (vhdr_path.is_file() and beh_path.is_file() and events_path.is_file()):
            continue
        eeg_channel_names = pd.read_csv(channels_path, sep="\t")
        eeg_channel_names = eeg_channel_names.loc[eeg_channel_names["type"] == "EEG", "name"].tolist()
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose="ERROR")
        raw.pick(picks=eeg_channel_names)
        srate = float(raw.info["sfreq"])
        data_tc = raw.get_data().T * 1e6
        events = pd.read_csv(events_path, sep="\t")
        beh = pd.read_csv(beh_path, sep="\t")
        onset_samples, conditions, memorandum, ncorrect, partial, trigger_correct = [], [], [], [], [], []
        next_row = {name: 0 for name in retention_codes.values()}
        for _, event in events[events["value"].isin(retention_codes)].iterrows():
            condition = retention_codes[int(event["value"])]
            rows = beh[beh["condition"] == condition]
            row_index = next_row[condition]
            if row_index >= len(rows):
                continue
            next_row[condition] += 1
            row = rows.iloc[row_index]
            onset_samples.append(int(round(float(event["onset"]) * srate)))
            conditions.append(condition)
            memorandum.append(int(str(row["sequence"])[0]))
            ncorrect.append(int(row["NCorrect"]))
            partial.append(int(row["partialScore"]))
            trigger_correct.append(str(row["triggerCorrect"]))
        onset_samples = np.asarray(onset_samples, dtype=int)
        if len(onset_samples) < MIN_TRIALS:
            continue
        counts = _scalp_delay_bin_power(data_tc, srate, onset_samples, window_s=6.0, n_bins=SCALP_EEG_DELAY_BINS)
        keep = np.isfinite(counts).all(axis=(1, 2))
        if keep.sum() < MIN_TRIALS:
            continue
        yield {
            "dataset": "ds006848", "patient": subject_dir.name, "session": f"{subject_dir.name}_verbalwm",
            "structure": "scalp", "channel_names": eeg_channel_names, "counts": counts[keep],
            "task_condition": np.asarray(conditions)[keep],
            "memorandum_content": np.asarray(memorandum, dtype=float)[keep],
            "accuracy_ncorrect": np.asarray(ncorrect, dtype=float)[keep],
            "accuracy_partial_score": np.asarray(partial, dtype=float)[keep],
            "accuracy_trigger_correct": np.asarray(trigger_correct)[keep],
        }


def iter_all_corpora(root: Path):
    yield from iter_dandi_000469(root)
    yield from iter_dandi_001187(root)
    yield from iter_dandi_000574(root)


PFC4_DELAY_WINDOW_S = 3.0
PFC4_MIN_UNITS = 2
PFC4_MIN_UNIT_RATE_HZ = 0.1


def pfc4_data_directory(root: Path) -> Path:
    """Directory of the CRCNS pfc-4 (somatosensory delay task) release,
    one animal per top-level subfolder, matching the ALM/macaque multi-object working-memory corpus (doi:10.64898/2026.01.27.702062) resolution
    pattern for a non-NWB release."""
    config = load_dataset_registry()
    local_path = config["datasets"]["pfc4"]["local_path"]
    return root / local_path


def load_pfc4_raw_session(path: Path, bin_ms: float = 100.0, window_s: float = PFC4_DELAY_WINDOW_S) -> dict | None:
    """Raw delay-epoch spike counts for one pfc-4 session, aligned to the end
    of the F1 stimulus (`SF1`) through the start of F2 (`SO2`) -- the
    stimulus-free retention interval of this task. Each trial's up-to-7 spike
    cells are already trial-relative (no session-wide clock), unlike the
    NWB-backed iterators above, so counts are built per trial like the ALM
    loader rather than via `build_psth`."""
    result = loadmat(path, squeeze_me=True, struct_as_record=False)["result"]
    header = list(result[0])
    hit_col, f1_col, f2_col = header.index("hit"), header.index("f1"), header.index("f2")
    spikes_col, sf1_col, so2_col = header.index("spikes"), header.index("SF1"), header.index("SO2")
    starts = np.arange(0.0, window_s, bin_ms / 1000.0)
    trials = []
    for r in range(1, result.shape[0]):
        row = result[r]
        sf1, so2 = np.atleast_1d(row[sf1_col]), np.atleast_1d(row[so2_col])
        if sf1.size != 1 or so2.size != 1:
            continue
        delay_start = float(sf1[0]) / 1000.0
        delay_end = float(so2[0]) / 1000.0
        if delay_end - delay_start < window_s:
            continue
        trials.append((row, delay_start))
    if len(trials) < MIN_TRIALS:
        return None
    counts = np.zeros((len(trials), 7, len(starts)), dtype=float)
    for t_index, (row, delay_start) in enumerate(trials):
        spikes = row[spikes_col]
        for unit in range(7):
            relative = np.atleast_1d(spikes[unit]).astype(float) / 1000.0 - delay_start
            counts[t_index, unit], _ = np.histogram(relative, bins=np.append(starts, window_s))
    rates = counts.sum(axis=(0, 2)) / (len(trials) * window_s)
    unit_mask = rates >= PFC4_MIN_UNIT_RATE_HZ
    if np.sum(unit_mask) < PFC4_MIN_UNITS:
        return None
    return {
        "n_units_after_rate_qc": int(np.sum(unit_mask)),
        "counts": counts[:, unit_mask],
        "correct": np.array([int(np.atleast_1d(row[hit_col])[0]) for row, _ in trials], dtype=bool),
        "f1_hz": np.array([float(np.atleast_1d(row[f1_col])[0]) for row, _ in trials]),
        "f2_hz": np.array([float(np.atleast_1d(row[f2_col])[0]) for row, _ in trials]),
    }


def iter_pfc4(root: Path, bin_ms: float = 100.0, window_s: float = PFC4_DELAY_WINDOW_S):
    """Every pfc-4 session, one recorded animal per top-level subfolder
    (rr014, rr015); rr015 additionally splits into left/right PFC
    subfolders, reported as `structure`."""
    directory = pfc4_data_directory(root)
    if not directory.is_dir():
        return
    for animal_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for path in sorted(animal_dir.rglob("*.mat")):
            session = load_pfc4_raw_session(path, bin_ms=bin_ms, window_s=window_s)
            if session is None:
                continue
            hemisphere = path.parent.name if path.parent.name in ("left", "right") else "pooled"
            yield {
                "dataset": "pfc4", "patient": animal_dir.name,
                "session": f"{animal_dir.name}_{path.stem}", "structure": hemisphere,
                "epoch": "delay", "counts": session["counts"], "bin_ms": bin_ms,
                "n_units": session["n_units_after_rate_qc"],
                "correct": session["correct"],
                "f1_hz": session["f1_hz"], "f2_hz": session["f2_hz"],
                "item_id_field": "f1_hz (first vibrotactile comparison frequency, the only pre-delay stimulus content)",
            }


def independent_unit(corpus: str, session: str) -> str:
    """The animal or participant a session belongs to, for pooling sessions within one
    independent unit before pooling across units. Local import of DATE_BLOCK_TO_ANIMAL avoids a
    module-level import cycle (that module imports corpus_sessions.data_root)."""
    if corpus == "panichello_2024_macaque_lPFC":
        from run_dominant_latent_identity_and_behaviour_breadth import DATE_BLOCK_TO_ANIMAL
        return DATE_BLOCK_TO_ANIMAL.get(session[:2], "unresolved_panichello_animal")
    if corpus == "inagaki_alm5_mouse_ALM":
        return session.split("_", 1)[0]
    if corpus.startswith("dandi_"):
        return session.split("__", 1)[0]
    return session.split("_", 1)[0]


def _watters_bundle_with_label(entry: dict, content_label_k_classes: int) -> dict:
    session, arrays, usable = entry["session"], entry["arrays"], entry["usable"]
    counts = session["counts"]
    activity_by_unit = counts.sum(axis=2)[usable]
    theta = np.mod(np.asarray(session["cued_theta"], dtype=float)[usable], 2.0 * np.pi)
    label = (np.floor(theta / (2.0 * np.pi / content_label_k_classes)).astype(int)) % content_label_k_classes
    return {
        "session": session["session"], "activity_by_unit": activity_by_unit, "deviation": arrays["deviation"],
        "outcome_raw": arrays["report_error"], "spike_count": arrays["spike_count"],
        "trial_index": arrays["trial_index"], "memorandum_label": label.astype(float),
        "item_count": arrays["item_count"],
    }


def pool_draws_within_session(draw_values: list) -> float | None:
    """Mean of a session's own repeated-draw correlation coefficients -- the within-session pooling step
    that runs BEFORE any cross-session significance test, so a session is represented by one value per
    rung regardless of how many independent unit subsamples were drawn for it. None/NaN draws (a
    not-computable fit) are dropped rather than propagated; a session with no usable draw at all pools to
    None, which the caller must treat as an exclusion, not a zero."""
    finite = [float(v) for v in draw_values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def recognition_correct(labels: np.ndarray, responses: np.ndarray) -> np.ndarray:
    true_old = np.asarray(labels).astype(str) == "0"
    response_old = np.asarray(responses, dtype=float) >= 34
    return true_old == response_old


def _watters_bundles(watters_arrays_by_session: dict) -> list[dict]:
    bundles = []
    for session_id, entry in watters_arrays_by_session.items():
        bundle = _watters_bundle_with_label(entry, CONTENT_LABEL_K_CLASSES)
        # cued_theta is not part of _watters_bundle_with_label's own return (it only needs the
        # discretised label from it); the axis-alignment analysis's continuous memorandum regression subspace needs the raw
        # angle, restricted to the identical `usable` trial mask that function already applied.
        bundle["cued_theta"] = np.asarray(entry["session"]["cued_theta"], dtype=float)[entry["usable"]]
        bundles.append(bundle)
    return bundles


def _watters_session_bundle(session: dict, arrays: dict, usable: np.ndarray) -> dict:
    counts = session["counts"]
    activity_by_unit = counts.sum(axis=2)[usable]
    theta = np.mod(np.asarray(session["cued_theta"], dtype=float)[usable], 2.0 * np.pi)
    label = (np.floor(theta / (2.0 * np.pi / CONTENT_LABEL_K_CLASSES)).astype(int)) % CONTENT_LABEL_K_CLASSES
    return {
        "session": session["session"], "corpus": "watters_2026_macaque_multi_object",
        "activity_by_unit": activity_by_unit, "deviation": arrays["deviation"],
        "outcome_raw": arrays["report_error"], "spike_count": arrays["spike_count"],
        "trial_index": arrays["trial_index"], "memorandum_label": label.astype(float), "counts": counts[usable],
        "n_trials_total": int(counts.shape[0]), "n_trials_with_defined_direction": int(usable.sum()),
    }


def _load_alm_for_counting_noise_census(root: Path) -> list[dict]:
    directory = alm_data_directory(root)
    out = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.mat")):
        raw = load_alm_raw_session(path, bin_ms=100.0, window_s=1.2, require_both_arms=False)
        if raw is None:
            continue
        out.append({"session": path.stem, "activity_by_unit": raw["control_counts"].sum(axis=2)})
    return out


SESSIONS = ("sham", "verum")


def paired_inventory(dataset_root: Path) -> dict:
    participants = sorted(p.name for p in dataset_root.glob("sub-*") if p.is_dir())
    paths = {}
    for participant in participants:
        paths[participant] = {
            session: dataset_root / participant / f"ses-{session}" / "eeg"
            / f"{participant}_ses-{session}_task-memory_eeg.set"
            for session in SESSIONS
        }
    complete = [p for p in participants if all(paths[p][s].is_file() for s in SESSIONS)]
    unpaired = [p for p in participants if sum(paths[p][s].is_file() for s in SESSIONS) == 1]
    absent = [p for p in participants if not any(paths[p][s].is_file() for s in SESSIONS)]
    return {
        "registered_participants": participants,
        "complete_pairs": complete,
        "unpaired_raw": unpaired,
        "no_raw_memory_recording": absent,
        "paths": paths,
    }


def _load_ds005034_session(directory, subject_dir, session):
    """Loads one (subject, stimulation-arm) recording directly, duplicating the minimal
    admission/window logic iter_ds005034 (src/corpus_sessions.py) applies -- needed so an
    already-checkpointed session can be skipped without paying that corpus iterator's own
    per-session EEG-load cost (a plain generator cannot be fast-forwarded past expensive
    items, and the iterator itself may not be edited in place). Field-for-field identical
    to what iter_ds005034 yields for the fields this script actually reads."""
    eeg_dir = subject_dir / f"ses-{session}" / "eeg"
    set_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_eeg.set"
    if not set_path.is_file():
        return None
    events_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_events.tsv"
    events = load_events(events_path)
    if len(events) < MIN_TRIALS:
        return None
    payload = loadmat(set_path, variable_names=("data", "srate"), squeeze_me=True)
    data_tc = np.asarray(payload["data"], dtype=np.float64).T
    srate = float(payload["srate"])
    onset_samples = np.array([int(round((event["onset"] + 0.5) * srate)) for event in events])
    counts = _scalp_delay_bin_power(data_tc, srate, onset_samples, window_s=6.0, n_bins=SCALP_EEG_DELAY_BINS)
    keep = np.isfinite(counts).all(axis=(1, 2))
    if keep.sum() < MIN_TRIALS:
        return None
    return {
        "patient": subject_dir.name, "session": f"{subject_dir.name}_ses-{session}",
        "stimulation_session": session, "counts": counts[keep],
        "task_condition": np.array([event["task"] for event in events])[keep],
        "load": np.array([event["load"] for event in events], dtype=float)[keep],
    }


MIN_SESSIONS_FOR_PRIMARY_BRANCH = 8


AXIS_WINDOW = (0.3, 1.0)


MIN_UNITS = 15


N_COMPONENTS = 8


N_SPLITS = 5


WINDOW_S = 2.3


MAINT_ONSET_S = 3.0


MAINT_WIN = 3.0


RULE_HASH = "c9505c80aed6b6c82494e472991a519c46a60a00bd8bfab7e6375f0706dc0ecd"


def patient_level_means(sessions: dict, metric_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Average each metric within patient before any cross-patient inference."""
    by_patient: dict[str, list[dict]] = {}
    for key, row in sessions.items():
        if row["status"] != "complete":
            continue
        patient = key.split("_ses-")[0]
        by_patient.setdefault(patient, []).append(row)
    patient_metrics: dict[str, dict[str, float]] = {}
    for patient, rows in by_patient.items():
        patient_metrics[patient] = {}
        for name in metric_names:
            values = [row["summary"][name] for row in rows if row["summary"][name] is not None]
            patient_metrics[patient][name] = float(np.mean(values)) if values else None
    return patient_metrics


def _trial_group(handle: h5py.File, release: str):
    return handle["intervals/WM_trials"] if release == "001187" else handle["intervals/trials"]


def _stimulation_displacement_session(rec: dict) -> dict:
    arrays = rec["arrays"]
    ch_names = arrays["ch_names"].tolist()
    anode, cathode, stim_ch = str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
    masks = channel_condition_masks(ch_names, anode, cathode, stim_ch)
    stim_flag = arrays["stim_flag"]
    n_stim, n_ctrl = int(stim_flag.sum()), int((stim_flag == 0).sum())
    conditions = {}
    for name, mask in masks.items():
        activity = _bin_averaged(arrays, mask)
        out = compute_stimulation_displacement(activity, stim_flag)
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


EPOCH_WINDOWS_BY_DATASET = {
    "dandi_000469": EPOCH_WINDOWS_S,
    "dandi_001187": EPOCH_WINDOWS_S,
    "dandi_000574": BORAN_EPOCH_WINDOWS_S,
}


def raw_counts_from_entry(entry: dict, bin_ms: float) -> np.ndarray:
    """Integer spike counts (trials, units, bins), delay epoch, un-smoothed."""
    window = EPOCH_WINDOWS_BY_DATASET[entry["dataset"]]["delay"]
    rate = build_psth(entry["spike_lists"], entry["epoch_onsets"]["delay"], bin_ms=bin_ms, smooth_ms=0, window_s=window)
    return np.rint(rate * (bin_ms / 1000.0)).astype(int)


ONSET_BIN = 16


BEHAVIOURAL_REFERENCE_R_UNITS = 0.14


CONTINUITY_ALIGNMENT_FLOOR_ABS_COSINE = 0.05


def _alignment_summary(observed: float, draws: np.ndarray) -> dict:
    finite = draws[np.isfinite(draws)]
    return {
        "observed_abs_cosine": observed,
        "observed_squared_fraction": observed ** 2,
        "null_mean_abs_cosine": float(np.mean(finite)) if finite.size else None,
        "null_sd_abs_cosine": float(np.std(finite)) if finite.size else None,
        "n_null_draws": int(finite.size),
    }


def _bias_only_axis(control_activity: np.ndarray) -> np.ndarray | None:
    """The plain normalised mean unit-direction over control trials -- the one direction the residual
    axis is, by construction, orthogonal to. Standing in here for "every trial's value replaced by its
    session's own mean", this project's established bias-only pattern, applied to the axis itself rather
    than to a per-trial scalar since the primary statistic here is a single per-session direction, not a
    per-trial correlation."""
    u = unit_direction_vectors(control_activity)
    valid = ~np.isnan(u).any(axis=1)
    if int(valid.sum()) < 2:
        return None
    mean_dir = u[valid].mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_dir))
    return (mean_dir / mean_norm).astype(float) if mean_norm > 1e-12 else None


def _bias_only_voids(real_pooled: dict, bias_pooled: dict) -> bool:
    """Sign-and-significance-only voiding, never a magnitude comparison: the bias-only control reproduces
    the real result exactly when both are non-significant, or both are significant with the same
    above-/below-null direction."""
    if real_pooled.get("real_pooled", {}).get("status") != "tested" or bias_pooled.get("real_pooled", {}).get("status") != "tested":
        return False
    if bool(real_pooled.get("significant")) != bool(bias_pooled.get("significant")):
        return False
    if real_pooled.get("significant") and (real_pooled.get("below_null") != bias_pooled.get("below_null")):
        return False
    return True


def _classify_arm(pooled: dict, bias_pooled: dict) -> dict:
    mdd_block = pooled.get("minimum_detectable_difference_80pct_power", {})
    mdd = mdd_block.get("mdd") if isinstance(mdd_block, dict) and mdd_block.get("status") == "computed" else None
    effect = pooled.get("real_pooled", {}).get("mean_value")
    if pooled.get("real_pooled", {}).get("status") != "tested" or mdd is None or effect is None:
        return {"branch": "not_computable", "mdd": mdd, "effect": effect}
    voids = _bias_only_voids(pooled, bias_pooled)
    if voids:
        return {"branch": "displacement_direction_not_separable_from_a_unit_level_offset",
                "mdd": mdd, "effect": effect}
    if mdd >= BEHAVIOURAL_REFERENCE_R_UNITS:
        return {"branch": "inconclusive_below_detection_floor", "mdd": mdd, "effect": effect}
    if pooled.get("significant") and pooled.get("below_null") is False:
        return {"branch": "stimulation_pushes_along_the_deviation_axis", "mdd": mdd, "effect": effect}
    return {"branch": "stimulation_pushes_off_the_deviation_axis", "mdd": mdd, "effect": effect}


def displacement_vector(control_activity: np.ndarray, stim_activity: np.ndarray) -> dict | None:
    """Mean unit-normalised direction over admitted stimulated trials minus the same over the matched
    (same-session) control trials -- rate-free on both sides, in the identical feature space the axis is
    estimated in."""
    u_ctrl = unit_direction_vectors(control_activity)
    u_stim = unit_direction_vectors(stim_activity)
    valid_ctrl = ~np.isnan(u_ctrl).any(axis=1)
    valid_stim = ~np.isnan(u_stim).any(axis=1)
    if int(valid_ctrl.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION or int(valid_stim.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    mean_ctrl = u_ctrl[valid_ctrl].mean(axis=0)
    mean_stim = u_stim[valid_stim].mean(axis=0)
    direction = mean_stim - mean_ctrl
    norm = float(np.linalg.norm(direction))
    unit = (direction / norm).astype(float) if norm > 1e-12 else None
    return {
        "direction_unit": unit, "norm": norm,
        "n_control_trials_used": int(valid_ctrl.sum()), "n_stim_trials_used": int(valid_stim.sum()),
    }


def estimate_axis(activity: np.ndarray, trial_index: np.ndarray, source: str, detrend: bool,
                   seed_tag: str) -> dict:
    """The residual-eigenvector axis, fit on `activity` alone. `source` must be the literal string
    "control_only" -- anything else raises immediately, because no stimulated trial may ever reach this
    fit, and a caller passing e.g. "includes_stimulated_trials" is exactly the mistake this guard exists
    to catch before it can silently contaminate an axis estimate."""
    if source != "control_only":
        raise ValueError(
            "estimate_axis refuses any source other than 'control_only' -- a stimulated trial must never "
            f"enter the axis fit; got source={source!r}")
    used_activity = _linear_detrend_activity(activity, trial_index) if detrend else activity
    rows = _residual_rows(used_activity)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_direction", "n_kept": rows["n_kept"]}
    R, idx = _unit_residual_matrix(rows)
    axis = leading_eigenvector(R)
    stability = _axis_stability(R, seed_tag)
    return {"status": "computed", "axis": axis, "n_trials_kept": int(R.shape[0]), "axis_stability": stability}


DATA_DIR = dataset_path("dandi_000469")


ITEM_FIELDS = {"item1": "loadsEnc1_PicIDs", "item2": "loadsEnc2_PicIDs", "item3": "loadsEnc3_PicIDs"}


def monkey_for_session(stem: str) -> str:
    year = int(stem[:2])
    return {21: "A", 22: "H", 24: "J"}.get(year, "unknown")


N_NEURONS_TARGET = 80


DATA = dataset_path("ram_ds005489_openloop")


def discover_000574_sessions(root: Path) -> list[tuple[str, str, Path]]:
    """Returns (patient, session_key, nwb_path) triples, sorted."""
    out = []
    for subject_dir in sorted((root / "000574").glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            out.append((subject_dir.name, path.stem, path))
    return out


def _panichello_directory(root: Path) -> Path | None:
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "datasets.json").read_text())
    entry = config["datasets"]["panichello_2024"]  # raise if the registry key is missing/mistyped, not a silent skip
    path = root / entry["local_path"]
    return path if path.is_dir() else None


def _macaque_session_bundle(path: Path) -> dict | None:
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1).astype(float)
    counts_all = _counts_from_spikes(spikes, time_ms)
    if counts_all.shape[0] < 16:
        return None
    activity_by_unit = counts_all.sum(axis=2)
    deviation = rate_free_state_deviation(activity_by_unit)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    return {
        "session": path.stem, "corpus": "panichello_2024_macaque_lPFC_single_item",
        "activity_by_unit": activity_by_unit[finite], "deviation": deviation[finite],
        "outcome_raw": is_corr[finite].astype(float), "spike_count": activity_by_unit.sum(axis=1)[finite],
        "trial_index": np.arange(counts_all.shape[0], dtype=float)[finite],
        "memorandum_label": cue_idx[finite], "counts": counts_all[finite],
        "n_trials_total": int(counts_all.shape[0]), "n_trials_with_defined_direction": int(finite.sum()),
    }


DATE_BLOCK_TO_ANIMAL = {"21": "monkey_A", "22": "monkey_H", "24": "monkey_J"}


def _load_session(path: Path) -> dict:
    """Every per-session array the three result blocks need, read once.

    The raster is left in its deposited integer dtype and binned directly:
    the largest session's raster is 810 x 1950 x 716, which a float cast
    would expand to nine gigabytes for no gain, while the binned counts it
    reduces to are a few tens of megabytes.
    """
    raw = loadmat(str(path), squeeze_me=True)
    spikes = raw["spks"]
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr_raw = np.asarray(raw["isCorr"]).reshape(-1)
    cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1)
    cue_ang = np.asarray(raw["cueAng"], dtype=float).reshape(-1)
    counts = np.asarray(_counts_from_spikes(spikes, time_ms), dtype=float)
    is_corr = is_corr_raw.astype(bool)
    finite = np.isfinite(np.asarray(is_corr_raw, dtype=float))
    return {
        "session": path.stem,
        "date_block": path.stem[:2],
        "animal": DATE_BLOCK_TO_ANIMAL.get(path.stem[:2], "unassigned"),
        "counts": counts,
        "is_corr": is_corr,
        "cue_idx": cue_idx.astype(int),
        "cue_ang": cue_ang,
        "n_trials": int(counts.shape[0]),
        "n_units": int(counts.shape[1]),
        "n_bins": int(counts.shape[2]),
        "n_correct": int(is_corr.sum()),
        "n_error": int((~is_corr).sum()),
        "outcome_field_dtype": str(is_corr_raw.dtype),
        "outcome_field_distinct_values": sorted(float(v) for v in np.unique(is_corr_raw)),
        "outcome_field_n_non_finite": int((~finite).sum()),
        "outcome_field_length_matches_trials": bool(len(is_corr_raw) == counts.shape[0]),
        "cue_label_distinct_values": sorted(int(v) for v in np.unique(cue_idx)),
    }


def _session_paths(root: Path) -> list[Path]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    return [Path(p) for p in sorted(glob.glob(str(directory / "*.mat")))]


def _session_arrays(path: Path) -> dict | None:
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    counts_all = _counts_from_spikes(spikes, time_ms)  # (trials, units, bins), whole delay epoch
    if counts_all.shape[0] < 16:
        return None
    activity_by_unit = counts_all.sum(axis=2)  # (trials, units) -- per-unit total spike count, delay epoch
    deviation = rate_free_state_deviation(activity_by_unit)
    total_spike_count = activity_by_unit.sum(axis=1)
    trial_index = np.arange(counts_all.shape[0], dtype=float)
    finite = np.isfinite(deviation)
    if finite.sum() < 16:
        return None
    return {
        "is_corr": is_corr[finite].astype(float),
        "deviation": deviation[finite],
        "spike_count": total_spike_count[finite],
        "trial_index": trial_index[finite],
        "n_trials_total": int(counts_all.shape[0]),
        "n_trials_with_defined_direction": int(finite.sum()),
    }


def _reachable_sessions(root: Path, limit: int | None = None) -> list[Path]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    paths = []
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, simplify_cells=True)
        is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
        n_error = int((~is_corr).sum())
        if n_error >= MIN_ERROR_TRIALS_FOR_REACHABILITY:
            paths.append(Path(path))
            if limit is not None and len(paths) >= limit:
                break
    return paths


def _session_limit() -> int | None:
    raw = os.environ.get(MAX_SESSIONS_ENV_VAR)
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _panichello_session_inputs(root: Path):
    """Yields (session_id, reason_or_None, core_or_None, categorical_spec, continuous_spec,
    counts_all_or_None) for every macaque prefrontal spatial working-memory corpus (Dryad doi:10.5061/dryad.kkwh70sct) session this corpus's own data admits -- factored out
    of run_panichello so the rule-3 mutual-orthogonalisation recomputation (only triggered if two or
    more of this corpus's own candidates align, see _orthogonalized_alignment_for_corpus) can rebuild
    the identical per-session inputs without duplicating this loading logic."""
    directory = _panichello_directory(root)
    paths = sorted(glob.glob(str(directory / "*.mat"))) if directory else []
    limit = _session_limit()
    if limit:
        paths = paths[:limit]
    for path in paths:
        session_id = Path(path).stem
        raw = loadmat(path, simplify_cells=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        cue_idx = np.asarray(raw["cueAngIdx"], dtype=float).reshape(-1)
        counts_all = _counts_from_spikes(spikes, time_ms)
        activity_by_unit = counts_all.sum(axis=2)
        core = _session_core(activity_by_unit)
        if core is None:
            yield (session_id, "fewer than the trial floor have a defined leave-one-out direction",
                   None, {}, {}, None)
            continue
        categorical = {"memorandum_content": cue_idx, "previous_trial_content": _previous_label(cue_idx)}
        continuous = {"gain_total_spike_count": core["spike_count"]}
        yield session_id, None, core, categorical, continuous, counts_all


def _load_panichello_for_counting_noise_census(root: Path) -> list[dict]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    out = []
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, simplify_cells=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        counts_all = _counts_from_spikes(spikes, time_ms)
        out.append({"session": Path(path).stem, "activity_by_unit": counts_all.sum(axis=2)})
    return out


def _load_human_for_counting_noise_census(root: Path, dataset: str) -> list[dict]:
    out = []
    for entry in iter_all_corpora(root):
        if entry["dataset"] != dataset or entry.get("structure") != "pooled":
            continue
        counts = delay_counts(entry["spike_lists"], entry["epoch_onsets"]["delay"], entry["epoch_windows"]["delay"])
        out.append({"session": f"{entry['patient']}|{entry['session']}", "activity_by_unit": counts.sum(axis=2)})
    return out


def _session_trial_arrays(entry: dict) -> dict:
    counts = delay_counts(entry["spike_lists"], entry["delay_onset"], entry["delay_window_s"], bin_ms=BIN_MS)
    activity_by_unit = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity_by_unit)
    spike_count = activity_by_unit.sum(axis=1)
    trial_index = np.arange(activity_by_unit.shape[0], dtype=float)
    finite = np.isfinite(deviation)
    n_finite = int(finite.sum())
    if n_finite < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_direction", "n_trials_total": int(activity_by_unit.shape[0]),
                "n_trials_with_defined_direction": n_finite}
    dev = deviation[finite]
    n = dev.shape[0]
    total_dev = float(dev.sum())
    # Session training-trial mean: the leave-one-out mean of the DEVIATION VALUES themselves (not the
    # unit-vectors rate_free_state_deviation's own reference averages) -- a per-trial constant carrying
    # only this session's between-session offset, with no trial-to-trial information at all. The mandatory
    # placebo control below asks whether this constant alone reproduces any significant result.
    control_dev = (total_dev - dev) / (n - 1) if n > 1 else np.full(n, np.nan)
    return {
        "status": "computed", "patient": entry["patient"], "session": entry["session"], "dataset": entry["dataset"],
        "is_correct": entry["is_correct"][finite].astype(float), "deviation": dev, "control_deviation": control_dev,
        "spike_count": spike_count[finite], "trial_index": trial_index[finite], "load_level": entry["load_level"][finite],
        "n_trials_total": int(activity_by_unit.shape[0]), "n_trials_with_defined_direction": n_finite,
    }


WATTERS_MIN_TRIALS_FOR_CORRELATION = 16


def _watters_reachability(counts: np.ndarray, discretized_label: np.ndarray, seed: int) -> dict:
    """Reachability gate for the continuous multi-object memorandum, using the project's own
    classification-based decodability test (session_subtractive_test) -- the same machinery the macaque
    arm's gate uses -- rather than a continuous regression R-squared. A direct linear regression of unit
    direction on the raw [cos, sin] cued-position target was tried first and was far too weak an estimator
    to answer "does this decode at all" (observed R-squared indistinguishable from its own permutation
    null on every session probed), while the classification test clears cleanly on several sessions of the
    same data. The cued position is discretised into classes with the identical binning rule
    results/watters_state_geometry.json's own cardinality ladder uses for this corpus so the gate result is
    directly comparable to that delivered result. The DECOMPOSITION itself still uses the continuous
    2-dimensional regression subspace; only this gate is discrete."""
    if counts.shape[0] < WATTERS_MIN_TRIALS_FOR_CORRELATION:
        return {"status": "too_few_trials", "n_trials": int(counts.shape[0])}
    window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]
    result = session_subtractive_test(window_mean, discretized_label, seed)
    if result.get("status") != "tested":
        return {"status": "content_reachability_not_computable", "subtractive_status": result.get("status")}
    cleared = bool(result.get("a_full_clears_own_null", False))
    return {
        "status": "tested", "n_trials": int(counts.shape[0]),
        "n_classes_discretised": WATTERS_RECOVERABILITY_K_CLASSES,
        "a_full": result["a_full"], "a_full_p_value": result["a_full_p_value"],
        "cleared": cleared, "k_latents": result["k_latents"],
    }


def alm_sessions(root: Path):
    for meta in iter_alm(root, bin_ms=BIN_MS, window_s=ALM_WINDOW_S):
        yield {
            "corpus": "alm", "session": meta["session"], "counts": meta["counts"],
            "label": meta.get("condition"), "label_field": meta.get("item_id_field"),
            "n_splits": 12, "n_null_replicates": 20,
        }


def _human_counts_from_spikes(spike_lists, onset, window_s: float, bin_ms: float = BIN_MS) -> np.ndarray:
    rate = build_psth(spike_lists, onset, bin_ms=bin_ms, smooth_ms=0.0, window_s=window_s)
    return rate * (bin_ms / 1000.0)


def human_sessions(root: Path):
    for meta in iter_all_corpora(root):
        if meta["structure"] != "pooled":
            continue
        onset = meta["epoch_onsets"]["delay"]
        window_s = meta["epoch_windows"]["delay"]
        counts = _human_counts_from_spikes(meta["spike_lists"], onset, window_s)
        yield {
            "corpus": "human_delay", "dataset": meta["dataset"], "session": meta["session"], "counts": counts,
            "label": meta.get("item_ids"), "label_field": meta.get("item_id_field"),
            "n_splits": 12, "n_null_replicates": 20,
        }


PANICHELLO_DELAY_WINDOW_MS = (300.0, 1450.0)


def macaque_sessions(root: Path):
    directory = _panichello_directory(root)
    if directory is None:
        return
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, squeeze_me=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
        cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1)
        spikes, cue_idx = spikes[correct], cue_idx[correct]
        starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], BIN_MS)
        binned = [spikes[:, (time_ms >= s) & (time_ms < s + BIN_MS), :].sum(axis=1) for s in starts]
        counts = np.stack(binned, axis=2)
        yield {
            "corpus": "panichello_lpfc", "session": Path(path).stem, "counts": counts,
            "label": cue_idx.astype(int), "label_field": "cueAngIdx (8-way cued colour/location bin)",
            "n_splits": 10, "n_null_replicates": 10,
        }


HUMAN_DELAY_WINDOW_S = 2.3


def load_human_session_arrays(entry: dict) -> dict | None:
    """From one corpus_sessions.iter_dandi_000469 entry, build the delay-epoch trial-by-feature
    array (trial-mean firing rate per unit, for the deviation score and the two subspace claims) and
    the full delay-epoch PSTH tensor (for cross-temporal generalisation)."""
    spike_lists = entry["spike_lists"]
    onsets = entry["epoch_onsets"]["delay"]
    n_trials = len(onsets)
    if n_trials < 20 or len(spike_lists) < 8:
        return None
    psth = build_psth(spike_lists, onsets, bin_ms=HUMAN_BIN_MS, smooth_ms=0.0,
                       window_s=HUMAN_DELAY_WINDOW_S)  # (n_trials, n_units, n_bins)
    X_flat = psth.mean(axis=2)  # (n_trials, n_units) -- native trial-by-feature array
    item_ids = entry["item_ids"]
    times = np.arange(psth.shape[2]) * (HUMAN_BIN_MS / 1000.0)
    return {"psth": psth, "X_flat": X_flat, "item_ids": item_ids, "times": times}


def _boran_field_potential_session(nwb_path: Path, signal: str) -> dict | None:
    """One dandi_000574 session's maintenance-window band power for one field-potential signal
    ('ieeg' = depth macro-contacts, 'eeg' = scalp montage), trial-admitted the same way
    src/corpus_sessions.py's own dandi_000574 spike iterator admits trials (artifact-flag exclusion
    plus correct-only), applied here independently since this reads a different signal group from the
    same file and is not a modification of that iterator.
    """
    with h5py.File(str(nwb_path), "r") as handle:
        if "intervals/trials" not in handle:
            return None
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
        correct = trials["correct"][:].astype(bool)
    keep = (~artifact) & correct
    if keep.sum() < MIN_TRIALS:
        return None
    loaded = load_boran_nwb(str(nwb_path), signal=signal, epoch_win=(-3.2, 0.3))
    epochs = loaded["epochs"][keep]  # (N, C, T)
    times = loaded["times"]
    srate = loaded["srate"]
    win_mask = (times >= FIELD_MAINTENANCE_WINDOW_S[0]) & (times < FIELD_MAINTENANCE_WINDOW_S[1])
    win_times = times[win_mask]
    n_bins = int(round((FIELD_MAINTENANCE_WINDOW_S[1] - FIELD_MAINTENANCE_WINDOW_S[0]) * 1000.0 / BIN_MS))
    bin_edges = np.linspace(win_times[0], FIELD_MAINTENANCE_WINDOW_S[1], n_bins + 1)
    n_trials, n_ch, _ = epochs.shape
    power = np.zeros((n_trials, n_ch, n_bins), dtype=float)
    for i in range(n_trials):
        try:
            filtered = bandpass_filter(epochs[i].T, FIELD_BAND_LO_HZ, FIELD_BAND_HI_HZ, srate)  # (T, C)
        except Exception:
            return None
        sq = (filtered ** 2)[win_mask]
        for b in range(n_bins):
            bin_mask = (win_times >= bin_edges[b]) & (win_times < bin_edges[b + 1])
            power[i, :, b] = sq[bin_mask].mean(axis=0) if bin_mask.any() else np.nan
    if not np.isfinite(power).all():
        return None
    patient = nwb_path.parent.name
    return {"dataset": f"dandi_000574_{signal}", "patient": patient, "session": nwb_path.stem,
            "X": power, "bin_ms": BIN_MS}


MATCHED_UNIT_COUNT = WATTERS_MIN_UNITS


MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION = 16


PRIMARY_QUALITY_TIER = "single_and_multi_unit"


def _behaviour_observables(counts: np.ndarray, session: dict) -> tuple[dict, dict, np.ndarray]:
    """The per-trial quantities every behavioural correlation in this script
    is built from, restricted to the trials on which the state direction,
    the saccadic report and the reaction time are all defined, together with
    the exclusion count by reason and the mask itself."""
    activity = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    report_error = np.asarray(session["report_deviation"], dtype=float)
    reaction_time = np.asarray(session["reaction_time_ms"], dtype=float)
    usable = np.isfinite(deviation) & np.isfinite(report_error) & np.isfinite(reaction_time)
    excluded = {
        "state_direction_undefined_zero_total_activity": int(np.sum(~np.isfinite(deviation))),
        "no_saccadic_report_recorded": int(np.sum(~np.isfinite(report_error))),
        "no_reaction_time_recorded": int(np.sum(~np.isfinite(reaction_time) & np.isfinite(report_error))),
    }
    observables = {
        "state_deviation": deviation[usable],
        "spike_count": activity.sum(axis=1).astype(float)[usable],
        "trial_index": np.arange(counts.shape[0], dtype=float)[usable],
        "report_error": report_error[usable],
        "reaction_time": reaction_time[usable],
        "item_count": np.asarray(session["num_objects"], dtype=float)[usable],
        "is_correct": np.asarray(session["correct"], dtype=bool)[usable],
    }
    return observables, excluded, usable


def _subsets(rows_in: list[dict]) -> dict[str, list[dict]]:
    subsets = {"pooled": rows_in}
    for animal in sorted({r["animal"] for r in rows_in}):
        subsets[f"animal_{animal}"] = [r for r in rows_in if r["animal"] == animal]
    for variant in sorted({r["task_variant"] for r in rows_in}):
        subsets[f"task_variant_{variant}"] = [r for r in rows_in if r["task_variant"] == variant]
    return subsets


QUALITY_TIERS = (PRIMARY_QUALITY_TIER, "good_single_units_only", "matched_unit_count_arm")


def _object_geometry(behaviour, session: dict) -> dict:
    """Per-trial swap (primary and strict) and imprecision, aligned to session['counts']'s own trial
    order, computed from the corpus's raw per-object and per-response Cartesian coordinates (no
    re-derivation from the polar columns, and no dependence on the corpus loader's own
    report_deviation beyond a sanity check against it)."""
    rows = behaviour.loc[(session["animal"], session["session_date"])]
    trial_rows = rows.loc[session["trial_num"].tolist()]

    object_x = trial_rows[[f"object_{i}_x" for i in range(3)]].to_numpy(dtype=float)
    object_y = trial_rows[[f"object_{i}_y" for i in range(3)]].to_numpy(dtype=float)
    response_x = trial_rows["response_x"].to_numpy(dtype=float)
    response_y = trial_rows["response_y"].to_numpy(dtype=float)
    target = trial_rows["target_object_index"].to_numpy(dtype=int)
    n = len(target)

    distances = np.hypot(object_x - response_x[:, None], object_y - response_y[:, None])
    all_undefined = np.all(np.isnan(distances), axis=1)

    # Identity check: the distance to the CUED object, computed here from raw Cartesian columns,
    # must equal the corpus loader's own report_deviation (computed independently from polar columns).
    target_col = np.clip(target, 0, 2)
    target_distance = distances[np.arange(n), target_col]
    reported_deviation = np.asarray(session["report_deviation"], dtype=float)
    finite_both = np.isfinite(target_distance) & np.isfinite(reported_deviation)
    identity_diff = np.abs(target_distance[finite_both] - reported_deviation[finite_both])

    landed = np.full(n, -1, dtype=int)
    ok = ~all_undefined
    landed[ok] = np.nanargmin(distances[ok], axis=1)
    landed_distance = np.full(n, np.nan)
    landed_distance[ok] = distances[ok, landed[ok]]

    swap_primary = np.zeros(n, dtype=bool)
    swap_primary[ok] = landed[ok] != target[ok]

    is_target_column = np.arange(3)[None, :] == target_col[:, None]
    uncued_distances = np.where(is_target_column, np.nan, distances)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An item-count-1 trial has no uncued object at all, so its row is all-NaN by construction --
        # numpy's own expected warning for that case, not a sign of a missing value elsewhere.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        uncued_min_distance = np.nanmin(uncued_distances, axis=1)
    swap_strict = uncued_min_distance < CORRECT_REPORT_DISTANCE_THRESHOLD  # NaN comparisons are False

    return {
        "swap_primary": swap_primary, "swap_strict": swap_strict, "imprecision": landed_distance,
        "n_trials_all_object_positions_undefined": int(all_undefined.sum()),
        "identity_check_max_abs_diff_target_distance_vs_report_deviation":
            float(identity_diff.max()) if identity_diff.size else None,
        "identity_check_n_compared": int(finite_both.sum()),
    }


def _swap_imprecision_session_arrays(session: dict, behaviour) -> dict | None:
    observables, _excluded, usable = _behaviour_observables(session["counts"], session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return None
    covariates = trial_amplitude_covariates(session["counts"])
    if covariates["status"] != "computed":
        return None
    amplitude_full = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    geometry = _object_geometry(behaviour, session)
    return {
        "item_count": observables["item_count"],
        "deviation": observables["state_deviation"],
        "amplitude": amplitude_full[usable],
        "spike_count": observables["spike_count"],
        "trial_index": observables["trial_index"],
        "swap_primary": geometry["swap_primary"][usable].astype(float),
        "imprecision": geometry["imprecision"][usable],
    }


def _observable_arrays(counts: np.ndarray, session: dict) -> tuple[dict | None, dict, np.ndarray]:
    """Every array the correlation family needs, restricted to trials with a
    defined state direction, report and reaction time (_behaviour_
    observables' own usable mask), with the amplitude covariate computed on
    the FULL session (trial_amplitude_covariates fits its transform on every
    trial passed in) and then subset by the identical mask -- the same
    convention every other corpus's amplitude arm in this project uses."""
    observables, excluded, usable = _behaviour_observables(counts, session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return None, excluded, usable
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return None, excluded, usable
    amplitude_full = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    arrays = {
        "amplitude": amplitude_full[usable], "deviation": observables["state_deviation"],
        "report_error": observables["report_error"], "spike_count": observables["spike_count"],
        "trial_index": observables["trial_index"], "reaction_time": observables["reaction_time"],
        "item_count": observables["item_count"],
    }
    return arrays, excluded, usable


def _matched_unit_subset(counts: np.ndarray, seed_tag: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|matched_units"))
    drawn = np.sort(rng.choice(counts.shape[1], size=MATCHED_UNIT_COUNT, replace=False))
    return counts[:, drawn]


def _synthetic_time_independent_counts(rng: np.random.Generator, n_trials: int, n_units: int, n_bins: int) -> np.ndarray:
    trial_rate = rng.gamma(shape=2.0, scale=1.5, size=(n_trials, n_units))
    return rng.poisson(trial_rate[:, :, None] * np.ones((1, 1, n_bins))).astype(float)


def data_directory() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the external data root.")
    path = Path(root) / "Wolff" / "data"
    if not path.is_dir():
        raise SystemExit(f"impulse-perturbation scalp-EEG data not staged at {path}")
    return path
