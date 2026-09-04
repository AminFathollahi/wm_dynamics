"""corpus_sessions.py -- shared (dataset, structure, session) iteration over the
three human single-unit Sternberg-family corpora with a delay/maintenance
period: DANDI 000469, the canonical 001187/000673 dedup, and DANDI 000574
(Boran). Each iterator yields region-filtered, firing-rate-QC'd spike lists
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
  - DANDI 000574 (Boran): no named per-epoch timestamp fields exist in its
    trial table. run_human_drift_spine_000574.py's docstring documents the
    task's fixed relative structure (fixation [-6,-5] s, encoding [-5,-3] s,
    maintenance [-3,0] s relative to the probe, i.e. maintenance onset =
    trial start_time + 3.0 s) -- reused here to derive all four epoch onsets
    from start_time.
  - DANDI 000004 (Chandravadia new/old recognition) is out of scope for this
    module for now, though NOT for the reason its task design would suggest:
    its trial table does carry a genuine maintenance interval
    (delay1_time to delay2_time, ~2.2 s, after a ~1.0 s stim_on/stim_off
    encoding period), so a delay epoch is meaningful here. What is actually
    missing is region-label support: its electrode `location` field uses a
    "{Hemisphere} {Structure}" convention ("Right Hippocampus") that neither
    of this project's existing parsers (`nwb_structure_hemisphere_suffix`,
    `nwb_boran_brainnetome_hybrid`) recognizes, and it is not yet registered
    in config/datasets.json. Both are a scoped follow-up, not a property of
    the task.
"""

from __future__ import annotations

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
from run_human_drift_spine_001187_000673 import canonical_sessions, _trial_group  # noqa: E402
from run_watters_source_replication import add_behavior_columns  # noqa: E402
from project_config import data_root as configured_data_root, load_dataset_registry  # noqa: E402

MIN_TRIALS = 20
MIN_UNITS_POOLED = 15
EPOCH_WINDOWS_S = {"baseline": 0.5, "encoding": 0.5, "delay": 2.3, "probe": 0.5}
BORAN_EPOCH_WINDOWS_S = {"baseline": 1.0, "encoding": 2.0, "delay": 3.0, "probe": 0.5}

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
            epoch_onsets = {"baseline": start + 0.0, "encoding": start + 1.0, "delay": start + 3.0, "probe": start + 6.0}
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


def alm_data_directory(root: Path) -> Path:
    """Directory of Inagaki ALM5 perturbation sessions (mouse ALM, single structure by design)."""
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


def iter_all_corpora(root: Path):
    yield from iter_dandi_000469(root)
    yield from iter_dandi_001187(root)
    yield from iter_dandi_000574(root)
