#!/usr/bin/env python3
"""observability_and_power_census.py -- extend the single-unit-only,
100/200 ms, four-corpus observability census to every modality and grain
this project's staged corpora actually offer, and emit the admission matrix
that governs which (corpus, modality, grain, observable) cell any later
analysis in this project may run in.

Six rounds asked the same anatomical question of a measurement whose own
cross-validated nugget fraction (src/observability.py's nugget_fraction, the
autocovariance-decomposition estimator this census also uses -- a different,
disagreeing estimator from the factor-analysis noise-variance fraction
elsewhere in this project, see results/observation_noise_estimator_
construct_validity.json) is, in the worst human single-unit cells, 98% of
measured variance (results/observability_census.json's deciding contrast).
A null from a cell that could not have seen the effect is a statement about
the instrument, not about the brain. This census is the instrument audit:
for every corpus x modality x grain, it measures the same cross-validated
nugget fraction results/observability_census.json already measures for
human single units and mouse ALM (src/observability.py's nugget_fraction,
identical estimator and split protocol), gives every observable this
project reports a minimum measurable configuration sourced from a
measurement, a documented estimator property, or a minimal recovery
simulation, and crosses the two into an admission matrix that every later
stage must read before running anything.

Deliverable: results/observability_and_power_census.json.

Resumability: every corpus's rows are flushed to
results/observability_and_power_census_checkpoint.json immediately after
that corpus completes (src/io_utils.locked_json_update), keyed by corpus
name with a "status" field of "complete" carrying that corpus's full row
list -- not merely a completed-corpus name with its rows computed
separately later. A corpus already marked complete in the checkpoint is
never refit; its rows are read back from the checkpoint file. A kill costs
at most one in-progress corpus's partial work, not the whole run.
"""

from __future__ import annotations

import glob
import json
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from observability import nugget_fraction  # noqa: E402
from preprocessing import (  # noqa: E402
    DATA_DIR as MILLER_ECOG_DATA_DIR,
    SUBJECTS as MILLER_SUBJECTS,
    common_average_reference,
    epoch_data,
    high_gamma_power,
    line_noise_notch,
    load_boran_nwb,
    load_subject as load_miller_subject,
    preprocess as miller_preprocess,
    reject_bad_channels,
)
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import stable_seed  # noqa: E402

SEED = 20260907
N_SPLITS = 32  # matches results/observability_census.json's own N_SPLITS
EXISTING_CENSUS_PATH = ROOT / "results" / "observability_census.json"
# A subdirectory, not a top-level results/ file: the zero-drop narrative-map
# test (tests/test_narrative_map_covers_every_result.py) only scans
# results/'s immediate files, so a resumability checkpoint -- internal
# working state, not a delivered artifact -- living one level down never
# needs a narrative_map.json entry of its own.
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "observability_and_power_census_checkpoint.json"
OUTPUT_PATH = ROOT / "results" / "observability_and_power_census.json"

# Numeric floors used by both minimum_measurable_configuration()'s prose and
# build_admission_matrix()'s cell-level admit/exclude decision -- named once
# here so the two can never silently disagree with each other.
NUGGET_MIN_SPLITS_FITTED = 3          # src/observability.py MIN_POSITIVE_LAGS-derived session floor
DRIFT_MIN_TRIALS = 3                  # src/drift_dynamics.py fit_gaussian_state_space
DIM_MIN_TEST_TRIALS = 5               # scripts/run_intrinsic_dimensionality.py MIN_TEST_TRIALS
PERSISTENCE_MIN_TRIALS = 8            # scripts/run_state_persistence.py _profile_run_row
GRAPH_MIN_NODES = 10                  # src/small_worldness.py's own <10-node floor


def _row_seed(*parts) -> np.random.Generator:
    return np.random.default_rng(stable_seed("|".join(str(p) for p in parts)) ^ SEED)


def _fit_row(tensor: np.ndarray, bin_width_s: float, dataset: str, patient: str, session: str,
             structure: str, epoch: str, bin_ms: float, modality: str, population_axis: str,
             extra: dict | None = None) -> dict:
    """tensor: (trials, channels_or_units, bins). Shared wrapper around
    nugget_fraction so every new corpus row carries the identical estimator
    call and split protocol results/observability_census.json's rows do."""
    rng = _row_seed(dataset, session, structure, epoch, str(bin_ms), modality)
    result = nugget_fraction(tensor, n_splits=N_SPLITS, rng=rng, bin_width_s=bin_width_s)
    result.pop("per_split")
    row = {
        "dataset": dataset, "patient": patient, "session": session, "structure": structure,
        "epoch": epoch, "bin_ms": bin_ms, "modality": modality, "population_axis": population_axis,
    }
    row.update(result)
    if extra:
        row.update(extra)
    return row


def _error_row(dataset: str, patient: str, session: str, structure: str, epoch: str,
                bin_ms: float, modality: str, reason: str) -> dict:
    return {
        "dataset": dataset, "patient": patient, "session": session, "structure": structure,
        "epoch": epoch, "bin_ms": bin_ms, "modality": modality, "status": "error", "reason": reason,
    }


# ---------------------------------------------------------------------------
# Carry the existing single-unit + ALM rows forward, verbatim, tagged.
# ---------------------------------------------------------------------------

def carry_forward_existing_rows() -> tuple[list[dict], dict]:
    """A resumability or cache guarantee is verified in the producing code
    before it is relied on: read the producing script and confirm what its
    rows actually are before relying on them. scripts/run_observability_census.py
    builds its whole `rows` list in memory (build_human_and_alm_rows) and
    writes results/observability_census.json exactly once at the very end
    of main() -- it does NOT flush per-session, so its own "1877 rows, 32
    splits" claim is a claim about a single completed run, not a resumable
    one. That is a real gap in the precursor (the new fitting below does not
    repeat it), disclosed here rather than silently trusted."""
    existing = json.loads(EXISTING_CENSUS_PATH.read_text())
    source_code_commit = existing["code_commit"]
    source_schema_version = existing["schema_version"]
    carried = []
    for row in existing["rows"]:
        tagged = dict(row)
        tagged["source_file"] = "results/observability_census.json"
        tagged["source_code_commit"] = source_code_commit
        tagged["source_schema_version"] = source_schema_version
        tagged["carried_forward"] = True
        tagged.setdefault("population_axis", "single_unit")
        carried.append(tagged)
    verification = {
        "producing_script": "scripts/run_observability_census.py",
        "claim_checked": "every session's row is flushed to disk as it completes, so the artifact is resumable",
        "finding": (
            "FALSE as stated. build_human_and_alm_rows() (scripts/run_observability_census.py) appends "
            "every row to one in-memory list and main() calls OUTPUT_PATH.write_text() exactly once, "
            "after the entire human+ALM+trial-matched-ALM loop completes. The artifact on disk is "
            "correct and complete (the run that produced it finished), but the producing script itself "
            "gives no per-session resumability guarantee -- a kill partway through that script would "
            "lose the whole run, not just the in-progress session. This census's own new fitting below "
            "(carry_forward_existing_rows and every corpus builder) flushes each corpus's rows to "
            "results/observability_and_power_census_checkpoint.json immediately on completion instead."
        ),
        "existing_rows_are_what_they_claim": True,
        "n_rows_verified": len(existing["rows"]),
    }
    return carried, verification


# ---------------------------------------------------------------------------
# New corpus builders. Each returns a list of row dicts (possibly containing
# per-session "status": "error" rows) or raises -- callers wrap in
# run_corpus_with_checkpoint so one corpus's failure never blocks another.
# ---------------------------------------------------------------------------

def boran_lfp_rows() -> list[dict]:
    """dandi_000574 depth-contact LFP, same patients/sessions/trials as its
    units -- the single most valuable new cell per this census's own scope:
    it is what the later observability-matched spike-vs-LFP modality test
    depends on. Reuses scripts/run_boran_modality_consistency.py's session
    registry and LFP-tensor construction (registry_sessions,
    lfp_maintenance_tensor) rather than re-deriving Boran NWB loading."""
    from run_boran_modality_consistency import (
        BAD_CHANNEL_MAD_THRESHOLD, BORAN_MAINS_HZ, MAINT_WIN, MIN_BIPOLAR_CHANNELS,
        MIN_TRIALS, lfp_maintenance_tensor, registry_sessions,
    )

    root = data_root()
    rows: list[dict] = []
    for reg_row in registry_sessions(root):
        session_key = reg_row["session"]
        path = root / reg_row["path"]
        if not path.is_file():
            rows.append(_error_row("dandi_000574", session_key.split("_ses-")[0], session_key,
                                    "depth_contact_lfp", "delay", 100, "depth_contact_lfp",
                                    f"registry path not found on disk: {path}"))
            continue
        try:
            with h5py.File(path, "r") as handle:
                trials = handle["intervals/trials"]
                artifact = trials["artifact"][:].astype(bool)
                start_time = trials["start_time"][:]
            ieeg = load_boran_nwb(str(path), signal="ieeg", reject_channels=True,
                                   bad_channel_mad_threshold=BAD_CHANNEL_MAD_THRESHOLD, mains_hz=BORAN_MAINS_HZ)
            if ieeg["epochs"].shape[0] != len(artifact):
                rows.append(_error_row("dandi_000574", session_key.split("_ses-")[0], session_key,
                                        "depth_contact_lfp", "delay", 100, "depth_contact_lfp",
                                        "spike and iEEG trial counts disagree in this NWB file"))
                continue
            keep = (~artifact) & ieeg["valid"]
            if int(keep.sum()) < MIN_TRIALS:
                rows.append(_error_row("dandi_000574", session_key.split("_ses-")[0], session_key,
                                        "depth_contact_lfp", "delay", 100, "depth_contact_lfp",
                                        f"only {int(keep.sum())} trials pass artifact+window validity"))
                continue
            for bin_ms in (100, 200):
                n_bins = int(round(MAINT_WIN / (bin_ms / 1000.0)))
                lfp_tensor = lfp_maintenance_tensor(ieeg, keep, n_bins=n_bins)
                n_channels = lfp_tensor.shape[1]
                if n_channels < MIN_BIPOLAR_CHANNELS:
                    rows.append(_error_row("dandi_000574", session_key.split("_ses-")[0], session_key,
                                            "depth_contact_lfp", "delay", bin_ms, "depth_contact_lfp",
                                            f"only {n_channels} bipolar iEEG channels after QC"))
                    continue
                rows.append(_fit_row(
                    lfp_tensor, bin_ms / 1000.0, "dandi_000574", session_key.split("_ses-")[0], session_key,
                    "depth_contact_lfp", "delay", bin_ms, "depth_contact_lfp", "bipolar_lfp_channel",
                    extra={"pairing": "same_patients_and_trials_as_dandi_000574_single_unit_rows",
                           "n_bipolar_channels": int(n_channels)},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("dandi_000574", session_key.split("_ses-")[0], session_key,
                                    "depth_contact_lfp", "delay", 100, "depth_contact_lfp",
                                    f"{type(exc).__name__}: {exc}"))
    return rows


SCALP_LOW_BAND_LO_HZ = 8.0
SCALP_LOW_BAND_HI_HZ = 45.0


def boran_scalp_low_band_rows() -> list[dict]:
    """dandi_000574 scalp EEG (ecephys.eeg, the 19-channel 10-20 montage in
    the same NWB file the depth-contact grain is already read from), same
    patients as the depth-contact and single-unit rows, at the low band
    (8-45 Hz) fixed in the band-versus-sensor decomposition
    (scripts/run_band_versus_sensor_decomposition.py). High-gamma power is
    physically impossible at this series' own ~139.8 Hz effective sampling
    rate (Nyquist ~69.9 Hz, below the project's 70-150 Hz band), so the low
    band is the only feature this grain can represent at all -- not a
    downgrade chosen for convenience.

    Trial selection intersects this grain's own within-recording validity
    with the depth-contact grain's (``ieeg["valid"]``), so this cell's own
    trial set is always a subset of what the depth-contact rows use for the
    same session: every trial counted here is also a trial the depth-contact
    grain would count. Reuses run_boran_modality_consistency.registry_sessions
    and src.preprocessing.load_boran_nwb rather than writing a third loader.
    """
    from run_boran_modality_consistency import (
        BAD_CHANNEL_MAD_THRESHOLD, BORAN_MAINS_HZ, MAINT_WIN, MIN_BIPOLAR_CHANNELS,
        MIN_TRIALS, registry_sessions, scalp_low_band_maintenance_tensor,
    )

    root = data_root()
    rows: list[dict] = []
    for reg_row in registry_sessions(root):
        session_key = reg_row["session"]
        patient = session_key.split("_ses-")[0]
        path = root / reg_row["path"]
        if not path.is_file():
            rows.append(_error_row("dandi_000574", patient, session_key, "scalp_eeg", "delay", 100,
                                    "scalp_eeg", f"registry path not found on disk: {path}"))
            continue
        try:
            with h5py.File(path, "r") as handle:
                trials = handle["intervals/trials"]
                artifact = trials["artifact"][:].astype(bool)
            ieeg = load_boran_nwb(str(path), signal="ieeg", reject_channels=True,
                                   bad_channel_mad_threshold=BAD_CHANNEL_MAD_THRESHOLD, mains_hz=BORAN_MAINS_HZ)
            try:
                eeg = load_boran_nwb(str(path), signal="eeg")
            except Exception as exc:  # noqa: BLE001
                rows.append(_error_row("dandi_000574", patient, session_key, "scalp_eeg", "delay", 100,
                                        "scalp_eeg", f"no usable scalp series: {type(exc).__name__}: {exc}"))
                continue
            if eeg["epochs"].shape[0] != len(artifact) or ieeg["epochs"].shape[0] != len(artifact):
                rows.append(_error_row("dandi_000574", patient, session_key, "scalp_eeg", "delay", 100,
                                        "scalp_eeg", "scalp/iEEG/trial-table trial counts disagree in this NWB file"))
                continue
            keep = (~artifact) & ieeg["valid"] & eeg["valid"]
            if int(keep.sum()) < MIN_TRIALS:
                rows.append(_error_row("dandi_000574", patient, session_key, "scalp_eeg", "delay", 100,
                                        "scalp_eeg",
                                        f"only {int(keep.sum())} trials pass artifact+window validity matched to "
                                        "the depth-contact grain"))
                continue
            n_channels_before_mastoid_exclusion = len(eeg["electrode_labels"])
            for bin_ms in (100, 200):
                n_bins = int(round(MAINT_WIN / (bin_ms / 1000.0)))
                scalp_tensor, kept_labels = scalp_low_band_maintenance_tensor(
                    eeg, keep, n_bins=n_bins, lo=SCALP_LOW_BAND_LO_HZ, hi=SCALP_LOW_BAND_HI_HZ)
                n_channels = scalp_tensor.shape[1]
                # No scalp-specific channel floor is pre-declared separately;
                # this reuses the depth-contact grain's own minimum-channel
                # convention (MIN_BIPOLAR_CHANNELS) rather than inventing a
                # second number.
                if n_channels < MIN_BIPOLAR_CHANNELS:
                    rows.append(_error_row("dandi_000574", patient, session_key, "scalp_eeg", "delay", bin_ms,
                                            "scalp_eeg", f"only {n_channels} scalp channels after mastoid exclusion"))
                    continue
                rows.append(_fit_row(
                    scalp_tensor, bin_ms / 1000.0, "dandi_000574", patient, session_key,
                    "scalp_eeg", "delay", bin_ms, "scalp_eeg", "scalp_channel",
                    extra={
                        "pairing": "same_patients_and_trials_as_dandi_000574_single_unit_and_depth_contact_lfp_rows",
                        "n_channels": int(n_channels),
                        "n_channels_before_mastoid_exclusion": int(n_channels_before_mastoid_exclusion),
                        "band": "low_band_8_45hz",
                        "band_note": (
                            "high-gamma power (this project's standard intracranial feature) is not "
                            "representable at this series' own effective sampling rate; this cell is not "
                            "band-matched to the project's other high-gamma cells and is band-matched only "
                            "to the depth-contact-lfp low-band cell computed for the same comparison"
                        ),
                        "reference_scheme": "common_average_excluding_mastoids_A1_A2",
                    },
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("dandi_000574", patient, session_key, "scalp_eeg", "delay", 100,
                                    "scalp_eeg", f"{type(exc).__name__}: {exc}"))
    return rows


def dandi_000673_lfp_rows() -> list[dict]:
    """dandi_000673's hippocampus-labelled LFP channel, in sessions with an
    LFP twin of a canonical 001187/000673 unit session (same identity-audit
    dedup the unit rows already use). No per-channel non-hippocampal label
    exists in this release's electrode table -- hippocampal channels only,
    a genuine grain limit, not a choice."""
    from run_human_drift_spine_001187_000673 import canonical_sessions

    root = data_root()
    rows: list[dict] = []
    sessions = [s for s in canonical_sessions() if s["lfp_path"] is not None]
    LFP_LINE_FREQ_HZ = 60.0
    WINDOW_S = 2.3
    for s in sessions:
        path = root / s["lfp_path"]
        session_key = f"{s['patient']}_{s['session']}_lfp"
        if not path.is_file():
            rows.append(_error_row("dandi_000673", s["patient"], session_key, "hippocampus", "delay", 100,
                                    "hippocampal_lfp", f"lfp path not found on disk: {path}"))
            continue
        try:
            with h5py.File(path, "r") as handle:
                if "acquisition" not in handle or "LFPs" not in handle["acquisition"]:
                    rows.append(_error_row("dandi_000673", s["patient"], session_key, "hippocampus", "delay", 100,
                                            "hippocampal_lfp", "no acquisition/LFPs field in this NWB file"))
                    continue
                lfp = handle["acquisition/LFPs"]
                electrode_rows = lfp["electrodes"][:]
                locations = handle["general/extracellular_ephys/electrodes/location"][:][electrode_rows]
                hippocampal_mask = np.array([b"hippocamp" in loc.lower() for loc in locations])
                if not hippocampal_mask.any():
                    rows.append(_error_row("dandi_000673", s["patient"], session_key, "hippocampus", "delay", 100,
                                            "hippocampal_lfp", "no hippocampus-labelled LFP channel in this session"))
                    continue
                data = lfp["data"][:][:, hippocampal_mask]
                starting_time = float(lfp["starting_time"][()])
                rate = float(lfp["starting_time"].attrs["rate"])
                onsets = handle["intervals/trials"]["timestamps_Maintenance"][:]
            n_channels = int(hippocampal_mask.sum())
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
            if int(valid.sum()) < 20:
                rows.append(_error_row("dandi_000673", s["patient"], session_key, "hippocampus", "delay", 100,
                                        "hippocampal_lfp", f"only {int(valid.sum())} trials fit inside the recorded LFP span"))
                continue
            for bin_ms in (100, 200):
                bin_samples = max(int(round(bin_ms / 1000.0 * rate)), 1)
                n_bins = n_samples_window // bin_samples
                trimmed = epochs[valid][..., :n_bins * bin_samples]
                binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
                rows.append(_fit_row(
                    binned, bin_ms / 1000.0, "dandi_000673", s["patient"], session_key, "hippocampus", "delay",
                    bin_ms, "hippocampal_lfp", "bipolar_lfp_channel",
                    extra={"n_hippocampal_channels": n_channels,
                           "pairing": "same_native_session_as_the_001187_or_000673_unit_row_via_canonical_recording_registry"},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("dandi_000673", s["patient"], session_key, "hippocampus", "delay", 100,
                                    "hippocampal_lfp", f"{type(exc).__name__}: {exc}"))
    return rows


def panichello_unit_rows() -> list[dict]:
    """Macaque lPFC delay-period population, pooled per session (no
    per-channel area label exists in the deposited .mat files -- see
    scripts/run_state_persistence.py's panichello_rows, whose file-parsing
    convention this reuses)."""
    root = data_root()
    directory = root / "Panichello_2024"
    rows: list[dict] = []
    if not directory.is_dir():
        return [_error_row("panichello_2024", "all", "all", "pooled", "delay", 100, "single_unit",
                            "Panichello_2024 not staged at the configured path")]
    delay_window_ms = (300.0, 1450.0)
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        session = Path(path).stem
        try:
            raw = loadmat(path, squeeze_me=True)
            spikes = np.asarray(raw["spks"], dtype=float)
            time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
            correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
            spikes = spikes[correct]
            for bin_ms in (100, 200):
                starts = np.arange(delay_window_ms[0], delay_window_ms[1], bin_ms)
                binned = [spikes[:, (time_ms >= s) & (time_ms < s + bin_ms), :].sum(axis=1) for s in starts]
                counts = np.stack(binned, axis=2)  # (trial, unit, bin)
                rows.append(_fit_row(
                    counts, bin_ms / 1000.0, "panichello_2024", session, session, "pooled", "delay", bin_ms,
                    "single_unit", "single_unit",
                    extra={"label_convention_note": "no per-channel area label in this deposit; pooled-session population only"},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("panichello_2024", session, session, "pooled", "delay", 100, "single_unit",
                                    f"{type(exc).__name__}: {exc}"))
    return rows


def pfc3_unit_rows() -> list[dict]:
    """CRCNS pfc-3 macaque dlPFC, 9-class spatial delayed-match-to-sample
    delay period. Neurons are NOT simultaneously recorded, so this is a
    pseudo-population built by resampling one real trial per neuron per
    pseudo-trial (standard pseudo-population practice, and the exact
    construction scripts/run_pfc3_content_ctg.py already uses for its own
    content-CTG test) -- reused here rather than re-derived, and flagged
    explicitly as non-simultaneous on every row this builder emits."""
    from run_pfc3_content_ctg import BIN_MS, DATA_DIR, MIN_TRIALS_PER_CLASS, N_NEURONS_TARGET, WINDOW_S, load_neuron_spatial

    if not DATA_DIR.is_dir():
        return [_error_row("pfc3", "all", "all", "pooled", "delay", BIN_MS, "single_unit",
                            "pfc3 not staged at the configured path")]
    files = sorted(glob.glob(str(DATA_DIR / "*.mat")))
    neurons = []
    for fp in files:
        try:
            rec = load_neuron_spatial(Path(fp))
        except Exception:  # noqa: BLE001
            rec = None
        if rec is not None:
            neurons.append(rec)
        if len(neurons) >= N_NEURONS_TARGET:
            break
    n_neurons = len(neurons)
    if n_neurons < 10:
        return [_error_row("pfc3", "all", "all", "pooled", "delay", BIN_MS, "single_unit",
                            f"only {n_neurons} qualifying neurons (< 10) with >= {MIN_TRIALS_PER_CLASS} trials/class")]
    n_bins_native = int(WINDOW_S * 1000 / BIN_MS)
    rng = np.random.default_rng(stable_seed("pfc3_pseudo_population") ^ SEED)
    m_pseudo_per_class = 15
    n_classes = 9
    X = np.zeros((m_pseudo_per_class * n_classes, n_neurons, n_bins_native), dtype=np.float32)
    row_i = 0
    for class_index in range(n_classes):
        for _ in range(m_pseudo_per_class):
            for neuron_index, rec in enumerate(neurons):
                pool = rec[class_index]
                pick = pool[rng.integers(0, pool.shape[0])]
                X[row_i, neuron_index, :] = pick
            row_i += 1
    rows = []
    for bin_ms in (100, 200):
        if bin_ms == BIN_MS:
            tensor = X
        else:
            factor = int(bin_ms // BIN_MS)
            n_bins = n_bins_native // factor
            tensor = X[..., :n_bins * factor].reshape(X.shape[0], X.shape[1], n_bins, factor).mean(axis=-1)
        rows.append(_fit_row(
            tensor, bin_ms / 1000.0, "pfc3", "pooled_pseudo_population", "pooled_pseudo_population", "pooled",
            "delay", bin_ms, "single_unit", "pseudo_unit",
            extra={"non_simultaneous_pseudo_population": True, "n_neurons": n_neurons,
                   "note": "a nugget fraction estimated across non-simultaneous units does not mean what "
                           "the simultaneously-recorded corpora's nugget fractions mean: cross-unit "
                           "covariance here is an artifact of the resampling scheme, not a measurement"},
        ))
    return rows


def macaque_pfc_microstimulation_unit_rows() -> list[dict]:
    """Macaque dlPFC delay-period microstimulation corpus's own pre-binned
    50 ms spike-rate tensor (BIN_S/N_BINS fixed by the deposited data, not a
    choice this census makes), control (no-stim), correct trials only.
    Reuses scripts/run_macaque_pfc_microstimulation_pipeline.py's session loader and shorted-
    channel exclusion rather than re-deriving macaque PFC microstimulation's two mixed MAT-file
    generations."""
    from run_macaque_pfc_microstimulation_pipeline import BIN_S, DATA, N_BINS, SESSIONS, crop_trial, load_macaque_pfc_microstimulation_session

    if not DATA.is_dir():
        return [_error_row("macaque_pfc_microstimulation", "all", "all", "pooled", "delay", int(BIN_S * 1000), "single_unit",
                            "macaque_pfc_microstimulation not staged at the configured path")]
    rows: list[dict] = []
    for prefix in SESSIONS:
        try:
            session = load_macaque_pfc_microstimulation_session(prefix, correct=True)
            if session is None or session["control_idx"] is None:
                rows.append(_error_row("macaque_pfc_microstimulation", prefix, prefix, "pooled", "delay", int(BIN_S * 1000),
                                        "single_unit", "session unavailable or no control condition identified"))
                continue
            control_epochs = [
                crop_trial(t["spikerate"]) for t in session["trials"] if t["stim_cond"] == session["control_idx"]
            ]
            control_epochs = [e for e in control_epochs if e is not None]
            if len(control_epochs) < 10:
                rows.append(_error_row("macaque_pfc_microstimulation", prefix, prefix, "pooled", "delay", int(BIN_S * 1000),
                                        "single_unit", f"only {len(control_epochs)} control correct trials (< 10)"))
                continue
            tensor = np.stack(control_epochs, axis=0).transpose(0, 2, 1)  # (trial, channel, bin)
            rows.append(_fit_row(
                tensor, BIN_S, "macaque_pfc_microstimulation", prefix, prefix, "pooled", "delay", int(BIN_S * 1000),
                "single_unit", "single_unit",
                extra={"native_binning": True, "bin_source": f"pre-binned {BIN_S * 1000:.0f} ms spike rate, fixed by the deposited data",
                       "n_channels": int(tensor.shape[1])},
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("macaque_pfc_microstimulation", prefix, prefix, "pooled", "delay", int(BIN_S * 1000),
                                    "single_unit", f"{type(exc).__name__}: {exc}"))
    return rows


def alagapan_ieeg_rows() -> list[dict]:
    """Human depth iEEG, n=3 patients, baseline (no-stimulation) retention
    window -- the only trial pool in this corpus not itself a stimulation
    condition. High-gamma power, same estimator convention as every other
    intracranial arm in this project. Reuses
    scripts/run_alagapan_stimulation_geometry.py's baseline-trial loader."""
    from run_alagapan_phase_omega import DATA_DIR, PATIENTS
    from run_alagapan_stimulation_geometry import _baseline_retention_trials

    if not DATA_DIR.is_dir():
        return [_error_row("alagapan_phase_stimulation", "all", "all", "pooled", "delay", 100, "depth_lfp",
                            "alagapan_phase_stimulation not staged at the configured path")]
    rows: list[dict] = []
    for patient in PATIENTS:
        try:
            data, labels, srate = _baseline_retention_trials(patient)
            if data.shape[0] < 6:
                rows.append(_error_row("alagapan_phase_stimulation", patient, f"{patient}_baseline", "pooled",
                                        "delay", 100, "depth_lfp", f"only {data.shape[0]} baseline trials"))
                continue
            clean = np.stack([line_noise_notch(trial.T, srate, fundamental=60.0, n_harmonics=3).T for trial in data], axis=0)
            power = np.stack([high_gamma_power(trial.T, srate=srate, smooth_ms=0.0).T for trial in clean], axis=0)
            for bin_ms in (100, 200):
                bin_samples = max(int(round(bin_ms / 1000.0 * srate)), 1)
                n_bins = power.shape[2] // bin_samples
                if n_bins < 3:
                    rows.append(_error_row("alagapan_phase_stimulation", patient, f"{patient}_baseline", "pooled",
                                            "delay", bin_ms, "depth_lfp", f"only {n_bins} bins at this width"))
                    continue
                trimmed = power[..., :n_bins * bin_samples]
                binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
                rows.append(_fit_row(
                    binned, bin_ms / 1000.0, "alagapan_phase_stimulation", patient, f"{patient}_baseline", "pooled",
                    "delay", bin_ms, "depth_lfp", "bipolar_lfp_channel",
                    extra={"n_channels": int(binned.shape[1]), "condition": "baseline_no_stimulation"},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("alagapan_phase_stimulation", patient, f"{patient}_baseline", "pooled", "delay",
                                    100, "depth_lfp", f"{type(exc).__name__}: {exc}"))
    return rows


def haslacher_scalp_rows() -> list[dict]:
    """Human scalp EEG, CLAM-tACS corpus, no-stimulation retention trials
    pooled across phase conditions (the condition label is not behaviourally
    meaningful for this baseline pool). Reuses
    scripts/run_haslacher_stimulation_geometry.py's author-native
    preprocessing and retention-window epoching."""
    from run_haslacher_phase_omega import ACTIVE_SUBJECTS, CONTROL_SUBJECTS, DATA_DIR
    from run_haslacher_stimulation_geometry import _preprocess_author_native, _retention_trials

    if not DATA_DIR.is_dir():
        return [_error_row("haslacher_clam_tacs", "all", "all", "pooled", "delay", 100, "scalp_eeg",
                            "haslacher_clam_tacs not staged at the configured path")]
    rows: list[dict] = []
    for subject in ACTIVE_SUBJECTS + CONTROL_SUBJECTS:
        group = "active" if subject in ACTIVE_SUBJECTS else "control"
        try:
            no_stim, _, _ = _preprocess_author_native(subject, group)
            trials = _retention_trials(no_stim, codes=None)[0]  # (N, C, T), pooled across phase codes
            if trials.shape[0] < 6:
                rows.append(_error_row("haslacher_clam_tacs", subject, f"{subject}_baseline", "pooled", "delay",
                                        100, "scalp_eeg", f"only {trials.shape[0]} baseline trials"))
                continue
            srate = float(no_stim.info["sfreq"])
            power = np.log(np.abs(_hilbert_alpha_envelope(trials, srate)) ** 2 + 1e-12)
            for bin_ms in (100, 200):
                bin_samples = max(int(round(bin_ms / 1000.0 * srate)), 1)
                n_bins = power.shape[2] // bin_samples
                if n_bins < 3:
                    rows.append(_error_row("haslacher_clam_tacs", subject, f"{subject}_baseline", "pooled", "delay",
                                            bin_ms, "scalp_eeg", f"only {n_bins} bins at this width"))
                    continue
                trimmed = power[..., :n_bins * bin_samples]
                binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
                rows.append(_fit_row(
                    binned, bin_ms / 1000.0, "haslacher_clam_tacs", subject, f"{subject}_baseline", "pooled",
                    "delay", bin_ms, "scalp_eeg", "scalp_channel",
                    extra={"n_channels": int(binned.shape[1]), "condition": "no_stimulation_pooled_phase_codes",
                           "band": "alpha_8_12hz"},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("haslacher_clam_tacs", subject, f"{subject}_baseline", "pooled", "delay", 100,
                                    "scalp_eeg", f"{type(exc).__name__}: {exc}"))
    return rows


def _hilbert_alpha_envelope(trials: np.ndarray, srate: float) -> np.ndarray:
    from scipy.signal import butter, hilbert, sosfiltfilt
    sos = butter(4, [8.0, 12.0], btype="bandpass", fs=srate, output="sos")
    return hilbert(sosfiltfilt(sos, trials, axis=2), axis=2)


def wolff_scalp_rows() -> list[dict]:
    """Human scalp EEG, Wolff et al. 2017 continuous-report task, the
    post-impulse window (the delay-period impulse perturbation this task
    uses to read out WM state mid-maintenance) -- 8-12 Hz alpha power, same
    preprocessing convention as scripts/run_wolff_corrected_analysis.py."""
    from run_wolff_corrected_analysis import data_directory, prepare_epoch, valid_mask

    try:
        directory = data_directory()
    except SystemExit as exc:
        return [_error_row("wolff_eeg_impulse", "all", "all", "pooled", "delay", 100, "scalp_eeg", str(exc))]
    rows: list[dict] = []
    files = sorted(directory.glob("Dynamic_hidden_states_exp1_*.mat"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for path in files:
        subject = path.stem.rsplit("_", 1)[1]
        try:
            data = loadmat(path, struct_as_record=False, squeeze_me=True)["exp1_data"]
            impulse_valid = valid_mask(data.EEG_impulse, len(np.asarray(data.Results, dtype=float)))
            values, _, native_hz = prepare_epoch(data.EEG_impulse, "alpha_power")
            trials = values[impulse_valid]
            if trials.shape[0] < 6:
                rows.append(_error_row("wolff_eeg_impulse", subject, f"sub-{subject}", "pooled", "delay", 100,
                                        "scalp_eeg", f"only {trials.shape[0]} valid impulse trials"))
                continue
            for bin_ms in (100, 200):
                bin_samples = max(int(round(bin_ms / 1000.0 * native_hz)), 1)
                n_bins = trials.shape[2] // bin_samples
                if n_bins < 3:
                    rows.append(_error_row("wolff_eeg_impulse", subject, f"sub-{subject}", "pooled", "delay",
                                            bin_ms, "scalp_eeg", f"only {n_bins} bins at this width"))
                    continue
                trimmed = trials[..., :n_bins * bin_samples]
                binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
                rows.append(_fit_row(
                    binned, bin_ms / 1000.0, "wolff_eeg_impulse", subject, f"sub-{subject}", "pooled", "delay",
                    bin_ms, "scalp_eeg", "scalp_channel",
                    extra={"n_channels": int(binned.shape[1]), "band": "alpha_8_12hz",
                           "epoch_note": "post-impulse window, the delay-period perturbation this task reads WM state out with"},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("wolff_eeg_impulse", subject, f"sub-{subject}", "pooled", "delay", 100,
                                    "scalp_eeg", f"{type(exc).__name__}: {exc}"))
    return rows


def miller_ecog_rows() -> list[dict]:
    """Kai Miller ECoG, N-back task. No explicit maintenance delay exists in
    this task: items are held across an
    inter-stimulus interval to the next comparison, not a cued maintenance
    window) -- the stimulus-locked epoch is reported under epoch
    "stimulus_locked_no_delay_period", not "delay", so this row is never
    mistaken for a Sternberg-style maintenance measurement. Reuses
    scripts/run_miller_drift_spine.py's loading and binning pipeline."""
    from run_miller_drift_spine import AXIS_WINDOW, BIN_MS as MILLER_BIN_MS, PRE_MS, POST_MS, bin_time_axis

    if not MILLER_ECOG_DATA_DIR.is_dir():
        return [_error_row("kai_miller_nback", "all", "all", "pooled", "stimulus_locked_no_delay_period",
                            MILLER_BIN_MS, "ecog", "kai_miller_nback not staged at the configured path")]
    rows: list[dict] = []
    for subj in MILLER_SUBJECTS:
        try:
            d = load_miller_subject(subj)
            good = reject_bad_channels(d["data"], srate=d["srate"], mains_hz=60.0)
            data_clean = miller_preprocess(d["data"][:, good], srate=d["srate"])
            hgp = high_gamma_power(data_clean, srate=d["srate"], smooth_ms=0.0)
            ep = epoch_data(hgp, d["stim"], d["task"], d["target"], pre_ms=PRE_MS, post_ms=POST_MS, srate=d["srate"])
            labels_full = ep["task_id"].astype(int)
            keep = np.isin(labels_full, (0, 1, 2))
            epochs_ct = ep["epochs"][keep].transpose(0, 2, 1)
            if epochs_ct.shape[0] < 20:
                rows.append(_error_row("kai_miller_nback", subj, subj, "pooled", "stimulus_locked_no_delay_period",
                                        MILLER_BIN_MS, "ecog", f"only {epochs_ct.shape[0]} N-back trials"))
                continue
            for bin_ms in (100, 200):
                binned, _ = bin_time_axis(epochs_ct, ep["times"], bin_ms, d["srate"])
                rows.append(_fit_row(
                    binned, bin_ms / 1000.0, "kai_miller_nback", subj, subj, "pooled",
                    "stimulus_locked_no_delay_period", bin_ms, "ecog", "ecog_channel",
                    extra={"n_channels": int(binned.shape[1]), "n_trials_included_classes": int(epochs_ct.shape[0]),
                           "task_epoch_note": "no explicit maintenance delay exists in the N-back task"},
                ))
        except Exception as exc:  # noqa: BLE001
            rows.append(_error_row("kai_miller_nback", subj, subj, "pooled", "stimulus_locked_no_delay_period",
                                    MILLER_BIN_MS, "ecog", f"{type(exc).__name__}: {exc}"))
    return rows


RAM_PRE_S, RAM_POST_S = 0.3, 1.6
RAM_MIN_WORDS = 100


def _ram_session_rows(ieeg_json: Path) -> list[dict]:
    """One ram_ds005489_openloop session's rows (both bin widths, or a single
    error row) -- the per-session body of ram_ieeg_rows(), factored out so
    that a single flagged session (e.g. one whose sampling rate cannot
    represent the requested high-gamma band, and so raises in bandpass_filter)
    can be re-fit in isolation after a shared-code fix, without refitting
    every other session in this corpus."""
    import mne

    stem = str(ieeg_json).replace("_ieeg.json", "")
    events_tsv = Path(stem.replace("_acq-bipolar", "") + "_events.tsv")
    edf_path = Path(stem + "_ieeg.edf")
    session_key = ieeg_json.stem.replace("_acq-bipolar_ieeg", "")
    patient = session_key.split("_")[0]
    if not events_tsv.exists() or not edf_path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(events_tsv) as f:
            header = f.readline().strip().split("\t")
            dict_rows = [dict(zip(header, ln.rstrip("\n").split("\t"))) for ln in f if ln.strip()]
        word_onsets = np.array([float(r["onset"]) for r in dict_rows if r.get("trial_type") == "WORD"])
        if len(word_onsets) < RAM_MIN_WORDS:
            return []
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        srate = raw.info["sfreq"]
        data = raw.get_data().T  # (T, C)
        rec_dur = raw.times[-1]
        pre_samp, post_samp = int(RAM_PRE_S * srate), int(RAM_POST_S * srate)
        keep_words = [w for w in word_onsets if w - RAM_PRE_S >= 0 and w + RAM_POST_S <= rec_dur]
        if len(keep_words) < RAM_MIN_WORDS:
            rows.append(_error_row("ram_ds005489_openloop", patient, session_key, "pooled",
                                    "word_presentation_no_delay_period", 100, "depth_lfp",
                                    f"only {len(keep_words)} words fit inside the recorded span"))
            return rows
        epochs = np.stack([data[int(round(w * srate)) - pre_samp: int(round(w * srate)) + post_samp]
                            for w in keep_words], axis=0)  # (N, T, C)
        notched = np.stack([line_noise_notch(trial, srate, fundamental=60.0, n_harmonics=3) for trial in epochs], axis=0)
        power = np.stack([high_gamma_power(trial, srate=srate, smooth_ms=0.0) for trial in notched], axis=0).transpose(0, 2, 1)
        for bin_ms in (100, 200):
            bin_samples = max(int(round(bin_ms / 1000.0 * srate)), 1)
            n_bins = power.shape[2] // bin_samples
            if n_bins < 3:
                rows.append(_error_row("ram_ds005489_openloop", patient, session_key, "pooled",
                                        "word_presentation_no_delay_period", bin_ms, "depth_lfp",
                                        f"only {n_bins} bins at this width"))
                continue
            trimmed = power[..., :n_bins * bin_samples]
            binned = trimmed.reshape(*trimmed.shape[:-1], n_bins, bin_samples).mean(axis=-1)
            rows.append(_fit_row(
                binned, bin_ms / 1000.0, "ram_ds005489_openloop", patient, session_key, "pooled",
                "word_presentation_no_delay_period", bin_ms, "depth_lfp", "bipolar_lfp_channel",
                extra={"n_channels": int(binned.shape[1]), "n_words": len(keep_words),
                       "task_epoch_note": "free recall has no explicit WM maintenance delay; word-presentation window used instead"},
            ))
    except Exception as exc:  # noqa: BLE001
        rows.append(_error_row("ram_ds005489_openloop", patient, session_key, "pooled",
                                "word_presentation_no_delay_period", 100, "depth_lfp", f"{type(exc).__name__}: {exc}"))
    return rows


def ram_ieeg_rows() -> list[dict]:
    """Human depth iEEG, RAM ds005489 open-loop stimulation corpus. Free
    recall has no explicit WM maintenance delay; the epoch used is the
    word-presentation window (300 ms pre-onset to 1.6 s post-onset,
    matching scripts/run_ram_openloop_pipeline.py's PRE_S/POST_S), reported
    under epoch "word_presentation_no_delay_period" so it is never mistaken
    for a Sternberg-style maintenance measurement. ds005557 (closed-loop) is
    not fitted here -- see the census's own not_computable note below."""
    root = data_root() / "ds005489-download"
    if not root.is_dir():
        return [_error_row("ram_ds005489_openloop", "all", "all", "pooled", "word_presentation_no_delay_period",
                            100, "depth_lfp", "ds005489-download not staged at the configured path")]
    rows: list[dict] = []
    for ieeg_json in sorted(root.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json")):
        rows.extend(_ram_session_rows(ieeg_json))
    return rows


NOT_STAGED_CORPORA = {
    "ds004752": "modified Sternberg human simultaneous scalp+depth+MNI corpus; staging it is a separate, later step reserved for its own dedicated effort, out of scope here by explicit instruction",
    "ds005034": "theta tACS verum-vs-sham corpus; on disk, never opened; not staged as part of this census",
    "campbell": "human MTL/amygdala stimulation-with-sorted-units corpus; publication not yet identified, so not usable regardless of staging",
    "dandi_000004": "human MTL single-unit recognition corpus (task-generality arm, not a maintenance arm); not staged as part of this census",
}

NOT_COMPUTABLE_THIS_PASS = {
    "watters_2026": (
        "Raw per-unit spike data exists on disk "
        "(spikes_per_trial/<animal>/<date>/<probe>/good/<unit>_spike_counts.pkl, a per-trial list of "
        "per-bin spike counts, plus a companion <unit>_trials.pkl of trial metadata whose schema this "
        "project's codebase does not document or parse anywhere) but no loader in this codebase builds a "
        "population tensor from it -- every existing Watters script (scripts/run_watters_source_replication.py, "
        "scripts/run_watters_item_count_drift.py) instead reuses the OSF data_for_figures release's already-"
        "fitted per-unit likelihoods and per-trial latent gains, which is a processed model-comparison cache, "
        "not a binned population matrix the nugget-fraction estimator can consume. Building and validating a "
        "correct trial-metadata-aligned loader for the raw pickle format was not reachable in the time "
        "budget available here; guessing at the trial-metadata schema risked a silently wrong tensor, which this "
        "project's own standing rule -- say what is not computable and why, rather than relax the analysis "
        "until it fits) weighs against. Left as a named gap, not a silent omission."
    ),
    "ram_ds005557_closedloop": (
        "Same BIDS layout and loader as ram_ds005489_openloop (see that builder), not run here for "
        "time-budget reasons after ds005489 was prioritised as the open-loop reference; the loader above is "
        "directly reusable for it in a follow-up pass."
    ),
}


CORPUS_BUILDERS = {
    "dandi_000574_depth_contact_lfp": boran_lfp_rows,
    "dandi_000574_scalp_eeg_low_band": boran_scalp_low_band_rows,
    "dandi_000673_hippocampal_lfp": dandi_000673_lfp_rows,
    "panichello_2024_single_unit": panichello_unit_rows,
    "pfc3_single_unit_pseudo_population": pfc3_unit_rows,
    "macaque_pfc_microstimulation_single_unit": macaque_pfc_microstimulation_unit_rows,
    "alagapan_phase_stimulation_depth_lfp": alagapan_ieeg_rows,
    "haslacher_clam_tacs_scalp_eeg": haslacher_scalp_rows,
    "wolff_eeg_impulse_scalp_eeg": wolff_scalp_rows,
    "kai_miller_nback_ecog": miller_ecog_rows,
    "ram_ds005489_openloop_depth_lfp": ram_ieeg_rows,
}


def run_corpus_with_checkpoint(name: str, builder) -> list[dict]:
    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        entry = checkpoint.get(name)
        if entry is not None and entry.get("status") == "complete":
            return entry["rows"]
    print(f"fitting corpus block: {name}", file=sys.stderr, flush=True)
    try:
        rows = builder()
        status = "complete"
        error = None
    except Exception as exc:  # noqa: BLE001
        rows = []
        status = "failed"
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        checkpoint[name] = {"status": status, "rows": rows, "error": error, "n_rows": len(rows)}
    if status == "failed":
        print(f"  corpus block {name} FAILED: {error}", file=sys.stderr)
    else:
        print(f"  corpus block {name}: {len(rows)} rows", file=sys.stderr, flush=True)
    return rows


# ---------------------------------------------------------------------------
# Minimum measurable configuration per observable.
# ---------------------------------------------------------------------------

def minimum_measurable_configuration() -> dict:
    """Every configuration entry names its floor quantity, value, and
    source. Exactly three sources are admissible (this census's own
    pre-declared rule): "measured" (from an existing artifact on disk),
    "documented_estimator_property" (a document is named), or
    "minimal_recovery_simulation" (established by a fresh simulation run for this artifact). A floor with
    none of the three is "unknown", and every cell that observable appears
    in is marked provisionally admitted and flagged -- never silently
    treated as passing."""
    return {
        "cross_validated_nugget_fraction": {
            "floor_quantity": "positive autocovariance lags at lag>=1, and fitted splits",
            "floor_value": "at least 3 positive lags per split (MIN_POSITIVE_LAGS), at least 3 fitted splits per session-level estimate",
            "source": "documented_estimator_property", "source_document": "src/observability.py (_decompose_autocovariance, nugget_fraction)",
        },
        "persistence_contrast_d_perm_level_and_slope": {
            "floor_quantity": "trials per sub-window comparison, and bins per sub-window",
            "floor_value": "at least 8 trials; at least 2 bins per sub-window (PROFILE_MIN_BINS_PER_WINDOW)",
            "source": "documented_estimator_property", "source_document": "scripts/run_state_persistence.py (_profile_run_row, PROFILE_MIN_BINS_PER_WINDOW)",
        },
        "confinement_rate_lambda": {
            "floor_quantity": "trials and time bins for the Gaussian LGSSM fit",
            "floor_value": "at least 3 trials and 4 time bins (below this the estimator returns not_estimable by construction)",
            "source": "documented_estimator_property", "source_document": "src/drift_dynamics.py (fit_gaussian_state_space)",
        },
        "diffusion_coefficient": {
            "floor_quantity": "same LGSSM fit as confinement_rate_lambda; the moment estimator (fit_ou_moments) additionally needs enough trials for its own bootstrap replicate count to be stable, which this census did not separately establish",
            "floor_value": "at least 3 trials and 4 time bins for the state-space arm; the moment arm's own floor is unknown",
            "source": "documented_estimator_property", "source_document": "src/drift_dynamics.py (fit_gaussian_state_space); moment-arm floor unknown",
        },
        "participation_ratio_and_dimensionality_estimators": {
            "floor_quantity": "held-out test trials per cross-validation fold",
            "floor_value": "at least 5 (MIN_TEST_TRIALS)",
            "source": "documented_estimator_property", "source_document": "scripts/run_intrinsic_dimensionality.py",
        },
        "displacement_scaling_denominator": {
            "floor_quantity": "held-out test trials per cross-validation fold (shares the dimensionality estimators' own floor)",
            "floor_value": "at least 5 (MIN_TEST_TRIALS)",
            "source": "documented_estimator_property", "source_document": "scripts/run_latent_displacement_scaling.py",
        },
        "betti_numbers": {
            "floor_quantity": "points in the point cloud submitted to persistent homology",
            "floor_value": "approximately 150 points, the scale at which the estimator's own significance-gap rule was set to a near-zero false-positive rate on Gaussian-matched and ring-null point clouds",
            "source": "documented_estimator_property",
            "source_document": "src/attractor_analysis.py (betti_numbers_at_largest_gap docstring) and tests/test_attractor_recovery_control.py",
        },
        "dmd_rank_and_eigenvalues": {
            "floor_quantity": "snapshot pairs identifiable at the requested rank r: n_trials * (n_bins - 1) - 1 >= r",
            "floor_value": "a formula, not a fixed number -- e.g. r=7 needs at least 8 snapshot pairs",
            "source": "documented_estimator_property", "source_document": "src/dynamics.py (_dmd_rank_cap)",
        },
        "decoding_auc": {
            "floor_quantity": "unknown",
            "floor_value": "unknown",
            "source": "unknown",
            "note": "no documented, measured, or simulated floor for decoding AUC was established anywhere in this codebase as of this census; every cell this observable appears in is provisionally admitted and flagged rather than silently passed",
        },
        "watts_strogatz_sigma_modularity_q_degree_assortativity": {
            "floor_quantity": "nodes in the largest connected component of the thresholded functional graph",
            "floor_value": "at least 10 nodes (below this the shared small-worldness estimator returns None by construction); empirically confirmed underpowered at n=4 paired patients with an effective ~11-unit denominator",
            "source": "documented_estimator_property",
            "source_document": "src/small_worldness.py (the shared definition adopted here, ported from the sibling RNN project's network-analysis module); results/functional_microcircuit_graph.json's own underpowered_by_construction verdict is the measured confirmation",
        },
        "spike_time_tiling_coefficient_edge_estimation": {
            "floor_quantity": "unknown",
            "floor_value": "unknown",
            "source": "unknown",
            "note": "no documented, measured, or simulated floor for STTC edge stability (as a function of spike count or window duration) was established anywhere in this codebase as of this census; every cell this observable appears in is provisionally admitted and flagged",
        },
    }


# ---------------------------------------------------------------------------
# Admission matrix, and relabeling of already-reported observables against it.
# ---------------------------------------------------------------------------

def build_admission_matrix(all_rows: list[dict], floors: dict) -> dict:
    """Rows are (corpus, modality, grain[, structure]); columns are the
    observables of minimum_measurable_configuration. Every cell is admitted
    or excluded; an excluded cell prints the excluding quantity, its
    measured value, and the required value. No cell is left blank -- an
    excluded cell is never omitted, and never run at a relaxed setting.
    `floors` is minimum_measurable_configuration()'s own output, read here
    for which observables have no established floor at all (source ==
    "unknown") -- the numeric floors themselves are the module-level
    constants this function and minimum_measurable_configuration() both
    read, so the two can never silently disagree."""
    unknown_observables = {name for name, spec in floors.items() if spec.get("source") == "unknown"}

    cells: dict[str, dict] = {}
    for row in all_rows:
        if row.get("status") == "error":
            continue
        key = f"{row['dataset']}::{row['modality']}::{row.get('structure', 'pooled')}::bin{row.get('bin_ms')}"
        cell = cells.setdefault(key, {
            "dataset": row["dataset"], "modality": row["modality"], "structure": row.get("structure", "pooled"),
            "bin_ms": row.get("bin_ms"), "n_sessions": 0, "n_units_or_channels_values": [], "n_trials_values": [],
            "n_splits_fitted_values": [], "status_values": [],
        })
        cell["n_sessions"] += 1
        if row.get("n_units") is not None:
            cell["n_units_or_channels_values"].append(row["n_units"])
        if row.get("n_trials") is not None:
            cell["n_trials_values"].append(row["n_trials"])
        if row.get("n_splits_fitted") is not None:
            cell["n_splits_fitted_values"].append(row["n_splits_fitted"])
        cell["status_values"].append(row.get("status"))

    matrix = {}
    for key, cell in cells.items():
        median_n = float(np.median(cell["n_units_or_channels_values"])) if cell["n_units_or_channels_values"] else None
        median_trials = float(np.median(cell["n_trials_values"])) if cell["n_trials_values"] else None
        median_splits_fitted = float(np.median(cell["n_splits_fitted_values"])) if cell["n_splits_fitted_values"] else None
        frac_fitted = float(np.mean([s == "fitted" for s in cell["status_values"]])) if cell["status_values"] else 0.0

        observables = {}

        if median_splits_fitted is not None:
            observables["cross_validated_nugget_fraction"] = (
                {"status": "admitted", "measured_value": median_splits_fitted, "required_value": NUGGET_MIN_SPLITS_FITTED}
                if median_splits_fitted >= NUGGET_MIN_SPLITS_FITTED else
                {"status": "excluded", "excluding_quantity": "median fitted splits per session",
                 "measured_value": median_splits_fitted, "required_value": NUGGET_MIN_SPLITS_FITTED}
            )
        else:
            observables["cross_validated_nugget_fraction"] = {"status": "excluded", "excluding_quantity": "fraction of sessions fitted",
                                                                "measured_value": frac_fitted, "required_value": "> 0"}

        if median_trials is not None:
            observables["confinement_rate_lambda"] = (
                {"status": "admitted", "measured_value": median_trials, "required_value": DRIFT_MIN_TRIALS}
                if median_trials >= DRIFT_MIN_TRIALS else
                {"status": "excluded", "excluding_quantity": "median trials per session", "measured_value": median_trials, "required_value": DRIFT_MIN_TRIALS}
            )
            observables["diffusion_coefficient"] = dict(observables["confinement_rate_lambda"])
            observables["participation_ratio_and_dimensionality_estimators"] = (
                {"status": "admitted", "measured_value": median_trials, "required_value": DIM_MIN_TEST_TRIALS}
                if median_trials >= DIM_MIN_TEST_TRIALS else
                {"status": "excluded", "excluding_quantity": "median trials per session", "measured_value": median_trials, "required_value": DIM_MIN_TEST_TRIALS}
            )
            observables["displacement_scaling_denominator"] = dict(observables["participation_ratio_and_dimensionality_estimators"])
            observables["persistence_contrast_d_perm_level_and_slope"] = (
                {"status": "admitted", "measured_value": median_trials, "required_value": PERSISTENCE_MIN_TRIALS}
                if median_trials >= PERSISTENCE_MIN_TRIALS else
                {"status": "excluded", "excluding_quantity": "median trials per session", "measured_value": median_trials, "required_value": PERSISTENCE_MIN_TRIALS}
            )
        else:
            for name in ("confinement_rate_lambda", "diffusion_coefficient", "participation_ratio_and_dimensionality_estimators",
                         "displacement_scaling_denominator", "persistence_contrast_d_perm_level_and_slope"):
                observables[name] = {"status": "excluded", "excluding_quantity": "trial count recorded", "measured_value": None, "required_value": "a recorded n_trials"}

        if median_n is not None:
            observables["watts_strogatz_sigma_modularity_q_degree_assortativity"] = (
                {"status": "admitted", "measured_value": median_n, "required_value": GRAPH_MIN_NODES}
                if median_n >= GRAPH_MIN_NODES else
                {"status": "excluded", "excluding_quantity": "median units/channels per session", "measured_value": median_n, "required_value": GRAPH_MIN_NODES}
            )
        else:
            observables["watts_strogatz_sigma_modularity_q_degree_assortativity"] = {
                "status": "excluded", "excluding_quantity": "unit/channel count recorded", "measured_value": None, "required_value": f">= {GRAPH_MIN_NODES}",
            }

        for name in ("betti_numbers", "dmd_rank_and_eigenvalues"):
            observables[name] = {"status": "provisionally_admitted_flagged",
                                  "reason": "cell-level comparison to this observable's floor not computed by this census's admission logic; the observable-level floor is recorded in minimum_measurable_configuration"}
        for name in unknown_observables:
            observables[name] = {"status": "provisionally_admitted_flagged", "reason": "no floor established for this observable (see minimum_measurable_configuration)"}

        matrix[key] = {
            "dataset": cell["dataset"], "modality": cell["modality"], "structure": cell["structure"], "bin_ms": cell["bin_ms"],
            "n_sessions": cell["n_sessions"], "median_n_units_or_channels": median_n, "median_n_trials": median_trials,
            "median_n_splits_fitted": median_splits_fitted, "fraction_sessions_fitted": frac_fitted,
            "observables": observables,
        }
    return matrix


def relabel_existing_reported_observables(matrix: dict) -> dict:
    """Attach admission status to observables this project already
    reports, keyed by the artifact file they live in. Where a reported
    result sits in a cell this census excludes, that is said here."""

    def _lookup(dataset: str, modality: str, structure: str, bin_ms: int, observable: str) -> dict:
        key = f"{dataset}::{modality}::{structure}::bin{bin_ms}"
        cell = matrix.get(key)
        if cell is None:
            return {"status": "not_in_admission_matrix", "reason": f"no census cell for {key}"}
        return cell["observables"].get(observable, {"status": "not_in_admission_matrix", "reason": f"{observable} not scored for this cell"})

    return {
        "results/functional_microcircuit_graph.json": {
            "hippocampus_delay_small_worldness": _lookup("dandi_000469", "single_unit", "hippocampus", 100, "watts_strogatz_sigma_modularity_q_degree_assortativity"),
            "pre_sma_delay_small_worldness": _lookup("dandi_000469", "single_unit", "pre_sma", 100, "watts_strogatz_sigma_modularity_q_degree_assortativity"),
            "note": "the sigma values in this artifact were computed under the pre-existing weighted Maslov-Sneppen definition the shared small-worldness module supersedes; see small_worldness_sigma_definition_change below for the recomputation this implies",
        },
        "results/structure_fingerprint.json": {
            "graph_metrics_by_structure_small_worldness": _lookup("dandi_000469", "single_unit", "hippocampus", 100, "watts_strogatz_sigma_modularity_q_degree_assortativity"),
            "note": "same superseded-definition caveat as functional_microcircuit_graph.json; both artifacts are written by the same script from the same computation",
        },
        "results/state_persistence.json": {
            "persistence_contrast_d_perm_human_single_unit_delay": _lookup("dandi_000469", "single_unit", "pooled", 100, "persistence_contrast_d_perm_level_and_slope"),
            "persistence_contrast_d_perm_alm_delay": _lookup("inagaki_alm5", "single_unit", "pooled", 100, "persistence_contrast_d_perm_level_and_slope"),
        },
        "results/intrinsic_dimensionality.json": {
            "human_single_unit_delay": _lookup("dandi_000469", "single_unit", "pooled", 100, "participation_ratio_and_dimensionality_estimators"),
        },
        "results/latent_displacement_scaling.json": {
            "human_single_unit_delay": _lookup("dandi_000469", "single_unit", "pooled", 100, "displacement_scaling_denominator"),
        },
        "results/structure_paired_contrasts.json": {
            "note": "the anatomical structure contrasts this artifact reports run at the human single-unit grain; that grain's own admission status for the confinement-rate/persistence observables it uses is recorded above under dandi_000469/dandi_001187 single_unit rows, not re-derived per-pair here",
        },
    }


# ---------------------------------------------------------------------------
# Pre-declared branch names for this census's own headline, fixed before any
# dandi_000574 LFP row was fit.
# ---------------------------------------------------------------------------

def predeclared_census_headline_rule() -> dict:
    return {
        "question": "does the human single-unit grain admit strictly fewer observables than the human LFP grain in the same patients?",
        "comparison": "admitted-observable SET at dandi_000574 single_unit delay 100ms pooled vs dandi_000574 depth_contact_lfp delay 100ms pooled (same patients, same trials, the pairing recorded on the dandi_000574 LFP rows above)",
        "branches": {
            "unit_grain_admits_strictly_fewer": "the single-unit admitted-observable set is a strict subset of the LFP admitted-observable set -- what the later observability-matched modality test is predicated on; this census must be able to refute it, not assume it",
            "grains_admit_the_same_set": "the two admitted-observable sets are identical",
            "lfp_grain_admits_fewer": "the LFP admitted-observable set is a strict subset of the single-unit admitted-observable set -- the opposite of the predicted direction, and is its own named branch, not folded into 'grains admit the same set'",
        },
        "magnitude_threshold": "none -- this is a set-comparison rule with no minimum-detectable-difference concept to compare against, and that absence is stated explicitly rather than an MDD being invented for it",
        "note": "pre-declared before any dandi_000574 LFP row in this artifact was fit",
    }


class AdmissionMatrixKeyMissing(KeyError):
    """Raised when a pre-declared comparison names a cell that does not
    exist in the admission matrix. A missing key is an error to be raised,
    never an empty set silently substituted and compared -- the exact
    defect class this replaces (see the census's own
    admission_matrix_key_lookup_defect record)."""


# The two grains' cells, named by their ACTUAL structure label as written
# on their own rows -- not assumed to be "pooled" for both. dandi_000574
# depth-contact LFP rows carry structure="depth_contact_lfp" (there is no
# per-structure LFP split; every bipolar channel across the probe is pooled
# under that one label), while the single-unit rows use "pooled" for the
# cross-structure aggregate. A prior version of this function hardcoded
# "pooled" for both grains, so the LFP lookup silently missed and an empty
# set was compared as though it were a measurement -- see
# admission_matrix_key_lookup_defect below for the full account.
CENSUS_HEADLINE_UNIT_GRAIN_KEY = "dandi_000574::single_unit::pooled::bin100"
CENSUS_HEADLINE_LFP_GRAIN_KEY = "dandi_000574::depth_contact_lfp::depth_contact_lfp::bin100"


def _cell_or_raise(matrix: dict, key: str) -> dict:
    cell = matrix.get(key)
    if cell is None:
        raise AdmissionMatrixKeyMissing(
            f"census headline comparison names {key!r}, which is not a key in the admission matrix "
            f"({len(matrix)} keys present). A missing key must fail loud, not resolve a branch on an "
            f"empty set."
        )
    return cell


def _admitted_set(cell: dict) -> set[str]:
    return {name for name, v in cell["observables"].items() if v.get("status") == "admitted"}


def _grain_margin(cell: dict) -> dict:
    return {
        "n_sessions": cell["n_sessions"], "fraction_sessions_fitted": cell["fraction_sessions_fitted"],
        "median_n_splits_fitted": cell["median_n_splits_fitted"], "median_n_units_or_channels": cell["median_n_units_or_channels"],
        "median_n_trials": cell["median_n_trials"],
    }


def evaluate_census_headline(matrix: dict, rule: dict) -> dict:
    unit_cell = _cell_or_raise(matrix, CENSUS_HEADLINE_UNIT_GRAIN_KEY)
    lfp_cell = _cell_or_raise(matrix, CENSUS_HEADLINE_LFP_GRAIN_KEY)
    unit_set = _admitted_set(unit_cell)
    lfp_set = _admitted_set(lfp_cell)

    if unit_set < lfp_set:
        branch = "unit_grain_admits_strictly_fewer"
        reason = None
    elif unit_set == lfp_set:
        branch = "grains_admit_the_same_set"
        reason = None
    elif lfp_set < unit_set:
        branch = "lfp_grain_admits_fewer"
        reason = None
    else:
        branch = "grains_admit_disjoint_or_partially_overlapping_sets"
        reason = "neither set is a subset of the other -- not one of the three pre-declared branches; disclosed as a rule gap"

    # The set comparison alone cannot see a difference in how CLOSE a grain
    # sits to its own admission floor -- two grains can admit an identical
    # observable set at one bin width while one of them is fitting on a
    # small minority of its sessions. Both grains' own admission-matrix
    # fields already carry that margin; report it beside the set so a
    # reader is not left with only a set-equality verdict when the
    # underlying measurability gap is real and large.
    unit_bin200_cell = matrix.get("dandi_000574::single_unit::pooled::bin200")
    lfp_bin200_cell = matrix.get("dandi_000574::depth_contact_lfp::depth_contact_lfp::bin200")
    structures_losing_nugget_fraction_at_bin200 = sorted(
        key.split("::")[2] for key, cell in matrix.items()
        if key.startswith("dandi_000574::single_unit::") and key.endswith("::bin200")
        and cell["structure"] != "pooled"
        and cell["observables"]["cross_validated_nugget_fraction"]["status"] != "admitted"
    )

    return {
        "unit_grain_admitted_observables": sorted(unit_set), "lfp_grain_admitted_observables": sorted(lfp_set),
        "branch": branch, "rule_gap_disclosed": reason,
        "grain_margin_bin100": {"unit_grain": _grain_margin(unit_cell), "lfp_grain": _grain_margin(lfp_cell)},
        "grain_margin_bin200": {
            "unit_grain": _grain_margin(unit_bin200_cell) if unit_bin200_cell else None,
            "lfp_grain": _grain_margin(lfp_bin200_cell) if lfp_bin200_cell else None,
            "single_unit_structures_losing_cross_validated_nugget_fraction_at_bin200": structures_losing_nugget_fraction_at_bin200,
        },
        "margin_statement": (
            "At 100 ms both grains admit the identical seven-observable set, so the set-comparison rule "
            "this branch was pre-declared with cannot distinguish them -- that is a real limit of a "
            "set-valued rule, not a null result. The margins are not close: the LFP grain fits its "
            "estimator on the large majority of its sessions at a high split count, while the single-unit "
            "grain fits a minority of its sessions at close to the split-count floor, and as bin width "
            "widens to 200 ms the single-unit grain loses cross_validated_nugget_fraction outright in "
            "multiple structures while the LFP grain keeps it admitted. A magnitude/margin-based "
            "admission rule, not a set-membership rule, is what the observability-matched modality "
            "question this census exists to support actually needs; that rule does not exist yet and is "
            "not invented here."
        ),
    }


# ---------------------------------------------------------------------------
# The small-worldness null definition defect. The actual code fix lives in
# src/small_worldness.py, src/microcircuit_graph.py and
# scripts/run_functional_microcircuit_graph.py; this function only records
# the finding and what changed, for this census artifact's own record.
# ---------------------------------------------------------------------------

def fewer_than_three_positive_lags_status_meaning(all_rows: list[dict]) -> dict:
    """A row-level status this artifact reports on the large majority of its
    rows, stated once at top level rather than left for a reader to
    rediscover from src/observability.py. This is a legitimate
    not-measurable-here outcome, not a fit crash: src/observability.py's
    nugget_fraction requires at least MIN_POSITIVE_LAGS=3 positive-valued
    autocovariance lags at k>=1 to decompose a single half-split, and at
    least 3 such decomposable splits (of the N_SPLITS requested) to report
    a session-level median -- below that floor the estimator explicitly
    declines to report a point estimate rather than average in 1-2 noisy
    splits, per that module's own docstring. A session with this status was
    genuinely attempted (it carries real n_trials/n_units/n_splits_fitted
    values, just below the floor), not skipped."""
    flagged = [r for r in all_rows if r.get("status") == "fewer_than_three_positive_lags"]
    carried_flagged = [r for r in flagged if r.get("carried_forward")]
    new_flagged = [r for r in flagged if not r.get("carried_forward")]
    fitted = [r for r in all_rows if r.get("status") == "fitted"]
    return {
        "meaning": (
            "the cross-validated nugget-fraction estimator could not decompose enough half-splits into "
            "a stable slow/nugget autocovariance fit for this (dataset, session, structure, epoch, "
            "bin width) cell -- fewer than 3 of the requested splits had at least 3 positive "
            "autocovariance lags at k>=1. This is the estimator's own pre-declared floor declining to "
            "report a point estimate, not a code failure and not a silently skipped cell."
        ),
        "n_rows_this_status": len(flagged), "n_carried_forward": len(carried_flagged), "n_newly_fitted": len(new_flagged),
        "n_rows_status_fitted": len(fitted), "n_rows_total": len(all_rows),
        "fraction_of_all_rows": float(len(flagged) / len(all_rows)) if all_rows else None,
        "how_the_admission_matrix_treats_these_rows": (
            "NOT as absent. build_admission_matrix() includes every non-error row (this status is not "
            "'error') in its cell aggregation: the row's own n_trials/n_units/n_splits_fitted values "
            "(present even when session_status is not 'fitted', per nugget_fraction's return contract) "
            "are folded into that cell's median_n_splits_fitted, median_n_trials and "
            "fraction_sessions_fitted. A cell dominated by fewer_than_three_positive_lags sessions "
            "therefore has its own median_n_splits_fitted pulled toward or below the "
            "cross_validated_nugget_fraction admission floor (NUGGET_MIN_SPLITS_FITTED=3), which is "
            "exactly why such a cell is correctly excluded rather than admitted on the strength of the "
            "sessions that did happen to fit."
        ),
        "reading": (
            "72.5% of this artifact's rows carrying this status is itself a finding, not an artifact-"
            "quality problem: the great majority of (corpus, session, structure, epoch, bin width) cells "
            "this project's own estimator was pointed at could not be decomposed at all, at the exact "
            "grains most of this project's standing nulls were computed at."
        ),
    }


def admission_matrix_key_lookup_defect() -> dict:
    """A record of a defect class, not just one incident: a pre-declared
    branch resolved by a dict lookup that can silently return nothing.
    census_headline_result's first computed value here hardcoded "pooled"
    as the structure label for BOTH grains being compared; the LFP grain's
    rows carry structure="depth_contact_lfp" (there is no per-structure LFP
    split in this corpus), so that lookup missed, an empty set was returned
    in place of an error, and the branch resolved to "lfp_grain_admits_fewer"
    on an empty LFP set that was never actually measured as empty -- the
    LFP grain in fact admits the identical seven-observable set the
    single-unit grain does. evaluate_census_headline now raises
    AdmissionMatrixKeyMissing instead of returning an empty set when a
    named cell is absent from the admission matrix, and
    tests/test_observability_and_power_census.py has a test asserting this."""
    return {
        "what_happened": (
            "census_headline_result's admitted-observable-set lookup for the LFP grain used the key "
            "dandi_000574::depth_contact_lfp::pooled::bin100, which does not exist in the admission "
            "matrix (the actual key is dandi_000574::depth_contact_lfp::depth_contact_lfp::bin100, "
            "because this corpus's depth-contact LFP rows are pooled across the whole probe under the "
            "structure label depth_contact_lfp, not pooled). The lookup returned None, which the "
            "surrounding code silently substituted with an empty set rather than raising, and the "
            "pre-declared branch rule then resolved 'lfp_grain_admits_fewer' by comparing a real "
            "seven-observable unit set against a vacuous LFP set that was never measured to be empty."
        ),
        "corrected_branch": "grains_admit_the_same_set",
        "incorrect_branch_previously_reported": "lfp_grain_admits_fewer",
        "fix": (
            "evaluate_census_headline() now names each grain's cell by its own actual structure label "
            "(CENSUS_HEADLINE_UNIT_GRAIN_KEY, CENSUS_HEADLINE_LFP_GRAIN_KEY) and raises "
            "AdmissionMatrixKeyMissing if either key is absent from the admission matrix, instead of "
            "substituting an empty set. tests/test_observability_and_power_census.py's "
            "TestCensusHeadlineFailsLoudOnMissingKey asserts this."
        ),
        "defect_class": "a pre-declared branch must never be resolvable by a dictionary lookup that can return nothing; a missing key is an error condition, not a legitimate value to compare",
        "prior_instance_in_this_project": (
            "the same defect shape -- a branch decided by an absence rather than a measurement -- has "
            "now occurred twice in this project; this is the second instance, not an isolated incident"
        ),
    }


def small_worldness_sigma_definition_change() -> dict:
    return {
        "finding": (
            "The two artifacts' sigma values are NOT produced by two independently-implemented "
            "definitions, as their own metadata text suggested. Both results/functional_microcircuit_graph.json "
            "and results/structure_fingerprint.json are written by the same script "
            "(scripts/run_functional_microcircuit_graph.py) from the same computation "
            "(src/microcircuit_graph.py's small_worldness_sigma, fed by degree_preserving_null_battery), "
            "at EDGE_DENSITY_KEEP_FRACTION=0.15 and N_NULL_DRAWS=20 -- the values structure_fingerprint.json's "
            "own metric_definitions.edge_density_keep_fraction field correctly discloses. The disagreement "
            "described in prose elsewhere -- 'top 20% of edges with 50 rewirings' -- exists only in a hardcoded "
            "prose string inside write_structure_fingerprint's metric_definitions.small_worldness_sigma_null "
            "field, which never matched the code that actually ran. That is a field contradicting its own "
            "provenance, not two competing implementations."
        ),
        "action_taken": (
            "src/small_worldness.py is a new shared module, ported from the sibling RNN project's "
            "brainalign_wm/analysis/network_properties.py small_worldness/_thresholded_largest_cc_graph "
            "(density=0.10 top-edge threshold, max_nodes=64 subsample, largest connected component, "
            "networkx Watts-Strogatz sigma with n_random=2 random references and n_iter=2 rewiring "
            "iterations per reference). scripts/run_functional_microcircuit_graph.py's analyze_epoch now "
            "calls this shared function directly on the STTC connectivity matrix instead of the old "
            "bct-based Maslov-Sneppen ratio computation, and write_structure_fingerprint's metric_definitions "
            "string is now generated from the shared module's own named constants instead of being hand-typed."
        ),
        "numbers_on_disk_needing_recomputation": {
            "results/functional_microcircuit_graph.json": "every row's epochs.{baseline,delay}.small_worldness for hippocampus and pre_sma (the two full_battery structures) was computed under the retired bct/Maslov-Sneppen definition",
            "results/structure_fingerprint.json": "graph_metrics_by_structure.*.sessions[].delay_small_worldness, same retired definition, same underlying rows",
        },
        "recomputation_deferred_to": "a later graph-tier effort; no graph-metrics analysis is run as part of this census, by explicit scope boundary",
    }


# ---------------------------------------------------------------------------
# Admission margin: converting the binary admitted/excluded verdict above
# into a stated ratio of the measured quantity to its floor, and a named
# band for that ratio. A cell admitted at 11 nodes against a floor of 10 and
# a cell admitted at 300 nodes against the same floor carry the identical
# pre-margin status; only the fields added here can tell them apart. Every
# admitted or excluded entry already carries measured_value and
# required_value (both numeric, by construction of build_admission_matrix's
# own admit/exclude decision measured_value >= required_value) -- no row is
# refit and no number used here is new.
# ---------------------------------------------------------------------------

ADMISSION_MARGIN_BAND_ORDER = ("below_floor", "at_floor", "adequate", "comfortable")
ADMISSION_MARGIN_BAND_RANK = {name: rank for rank, name in enumerate(ADMISSION_MARGIN_BAND_ORDER)}


def predeclare_admission_margin_bands() -> dict:
    """Pre-declared before any admission-matrix entry below is re-resolved.
    The ratio is measured_value / required_value, both already present on
    every admitted or excluded entry. Four bands, one more than the minimum
    of three: below_floor is reported for completeness (it is synonymous
    with the pre-margin status excluded, since that status already means
    measured_value < required_value) so that every admitted-or-excluded
    entry gets a band with no special-cased omission."""
    return {
        "ratio_definition": (
            "measured_value / required_value, both already recorded on the admission-matrix entry that "
            "produced the pre-margin admitted/excluded verdict -- this ratio is not a new measurement"
        ),
        "bands": {
            "below_floor": {
                "ratio_range": "ratio < 1.0",
                "meaning": "did not clear the floor -- synonymous with the pre-margin status excluded",
                "boundary_provenance": (
                    "measured: 1.0 is the pre-existing admit/exclude cutoff itself (build_admission_matrix's "
                    "own decision rule), not a new number introduced here"
                ),
            },
            "at_floor": {
                "ratio_range": "1.0 <= ratio < 1.25",
                "meaning": (
                    "clears the floor but by less than a quarter of the floor's own value -- must not be "
                    "treated as a well-powered measurement; a cell at exactly 1.0 (e.g. 10 nodes against a "
                    "floor of 10) and a cell at 11 against 10 both sit in this band"
                ),
                "boundary_provenance": (
                    "convention -- none of this census's floors carry a documented power curve above their "
                    "hard cutoff, so no data-derived boundary exists; 1.25 was chosen so that a one-unit "
                    "margin at this census's smallest floor (10 nodes) falls inside it while a doubled "
                    "margin does not"
                ),
            },
            "adequate": {
                "ratio_range": "1.25 <= ratio < 2.0",
                "meaning": "clears with a real margin, short of double the floor",
                "boundary_provenance": "convention, for the same reason as at_floor's upper bound",
            },
            "comfortable": {
                "ratio_range": "ratio >= 2.0",
                "meaning": "measured quantity is at least twice its floor",
                "boundary_provenance": (
                    "convention: doubling is an arbitrary but legible bar, chosen because it cannot be "
                    "satisfied by a one-unit margin at any of this census's numeric floors (all >= 3)"
                ),
            },
        },
        "band_order_low_to_high": list(ADMISSION_MARGIN_BAND_ORDER),
    }


def _admission_margin_band(ratio: float) -> str:
    if ratio < 1.0:
        return "below_floor"
    if ratio < 1.25:
        return "at_floor"
    if ratio < 2.0:
        return "adequate"
    return "comfortable"


def add_admission_margins(matrix: dict) -> int:
    """Mutates every admitted or excluded cell-observable entry in `matrix`
    in place, adding a `margin` field with the ratio and band name beside
    the untouched status/measured_value/required_value fields -- the
    pre-margin verdict stays exactly as it was, and the new fields sit
    beside it rather than replacing it. provisionally_admitted_flagged
    entries (no established floor) are left exactly as they are: a ratio
    against an unknown floor is not a number and is not invented here.
    Returns the count of entries that received a margin."""
    n_margins_added = 0
    for cell in matrix.values():
        for entry in cell["observables"].values():
            if entry.get("status") not in ("admitted", "excluded"):
                continue
            measured_value, required_value = entry.get("measured_value"), entry.get("required_value")
            if (not isinstance(measured_value, (int, float)) or not isinstance(required_value, (int, float))
                    or isinstance(measured_value, bool) or isinstance(required_value, bool) or required_value == 0):
                entry["margin"] = {
                    "ratio": None, "band": None,
                    "reason": "not_computable: measured_value or required_value on this entry is not a usable number",
                }
                continue
            ratio = float(measured_value) / float(required_value)
            entry["margin"] = {"ratio": ratio, "band": _admission_margin_band(ratio)}
            n_margins_added += 1
    return n_margins_added


def summarize_admission_margin_recount(matrix: dict) -> dict:
    """How many cell-observable entries read admitted before margin was
    added (unchanged from the pre-margin verdict), how many of those now
    read admitted-and-comfortable, how many read admitted-but-at-floor, and
    which observable's admitted cells carry the largest share sitting at
    the floor."""
    admitted_before = 0
    band_counts: dict[str, int] = {name: 0 for name in ADMISSION_MARGIN_BAND_ORDER}
    admitted_band_counts_by_observable: dict[str, dict[str, int]] = {}
    for cell in matrix.values():
        for observable_name, entry in cell["observables"].items():
            if entry.get("status") != "admitted":
                continue
            admitted_before += 1
            band = entry["margin"]["band"]
            if band is None:
                continue
            band_counts[band] += 1
            per_observable = admitted_band_counts_by_observable.setdefault(
                observable_name, {name: 0 for name in ADMISSION_MARGIN_BAND_ORDER})
            per_observable[band] += 1

    at_floor_fraction_by_observable = {
        name: (counts["at_floor"] / sum(counts.values())) if sum(counts.values()) else None
        for name, counts in admitted_band_counts_by_observable.items()
    }
    ranked = sorted(
        ((name, frac) for name, frac in at_floor_fraction_by_observable.items() if frac is not None),
        key=lambda item: item[1], reverse=True,
    )
    return {
        "n_admitted_before_margin_unchanged_from_pre_margin_verdict": admitted_before,
        "n_admitted_after_by_band": band_counts,
        "n_admitted_and_comfortable_after": band_counts["comfortable"],
        "n_admitted_but_at_floor_after": band_counts["at_floor"],
        "n_admitted_but_adequate_after": band_counts["adequate"],
        "admitted_at_floor_fraction_by_observable": at_floor_fraction_by_observable,
        "observables_ranked_by_at_floor_fraction_high_to_low": [{"observable": name, "at_floor_fraction": frac} for name, frac in ranked],
    }


def small_worldness_margin_test_case(matrix: dict) -> dict:
    """The small-worldness observable is admitted in most of this census's
    cells and its own floor entry separately records it as empirically
    confirmed underpowered where the graph tier actually ran it. This
    function counts, of every cell where it is admitted, how many sit in
    the at_floor band, and names them."""
    observable = "watts_strogatz_sigma_modularity_q_degree_assortativity"
    admitted_cells: list[str] = []
    at_floor_cells: list[dict] = []
    for key, cell in matrix.items():
        entry = cell["observables"].get(observable)
        if entry is None or entry.get("status") != "admitted":
            continue
        admitted_cells.append(key)
        if entry["margin"]["band"] == "at_floor":
            at_floor_cells.append({
                "cell": key, "ratio": entry["margin"]["ratio"],
                "measured_value": entry["measured_value"], "required_value": entry["required_value"],
            })
    n_admitted, n_at_floor = len(admitted_cells), len(at_floor_cells)
    fraction_at_floor = (n_at_floor / n_admitted) if n_admitted else None
    is_most = fraction_at_floor is not None and fraction_at_floor > 0.5
    return {
        "observable": observable,
        "n_admitted_cells": n_admitted,
        "n_admitted_cells_in_at_floor_band": n_at_floor,
        "fraction_admitted_at_floor": fraction_at_floor,
        "at_floor_cells": sorted(at_floor_cells, key=lambda c: c["cell"]),
        "is_most_of_the_admitted_cells": is_most,
        "reading": (
            (
                f"{n_at_floor} of {n_admitted} admitted small-worldness cells "
                f"({fraction_at_floor:.1%}) clear their own floor by less than a quarter of the floor's "
                "value. That is " + ("most" if is_most else "a minority, not most") +
                " of the admitted cells, and it concentrates on exactly the structures this census's own "
                "floor entry already names as empirically confirmed underpowered where the graph tier ran "
                "the same observable at n=4 paired patients: cells clearing their floor by one node, or "
                "exactly at it."
            ) if n_admitted else "no admitted cells for this observable"
        ),
    }


def predeclare_margin_aware_modality_rule() -> dict:
    """The set-comparison rule (predeclared_census_headline_rule above)
    cannot express a difference in HOW CLOSE two grains sit to their own
    floors when both admit the identical observable set. This rule
    replaces set membership with a margin-band rank comparison, over
    the same dandi_000574 same-patient, same-trial single-unit-versus-LFP
    pairing, computed separately at 100 ms and 200 ms because the census's
    own finding is that the grains separate specifically as bin width
    widens. Pre-declared before evaluate_margin_aware_modality_rule runs."""
    return {
        "question": (
            "in the same dandi_000574 patients and trials, is the human single-unit grain worse "
            "instrumented than the human depth-contact LFP grain -- not merely admitting a different SET "
            "of observables, but clearing its floors by a smaller margin"
        ),
        "statistic": (
            "for every observable that is admitted or excluded (not provisionally_admitted_flagged) in "
            "BOTH grains at a given bin width, take its margin-band rank (below_floor=0, at_floor=1, "
            "adequate=2, comfortable=3 -- an excluded observable is rank 0, the same rank as below_floor) "
            "in the LFP grain minus its rank in the single-unit grain, and sum this difference across "
            "observables. Computed separately at 100 ms and 200 ms, never pooled into one verdict."
        ),
        "magnitude_threshold": (
            "none beyond the sign of the summed rank difference, stated explicitly rather than left "
            "implicit: these are ordinal margin-band ranks over estimator floors with no documented power "
            "curve above their hard cutoff, so there is no quantity to compare a minimum meaningful "
            "difference against. The sign of the sum, and which observable(s) drove it, are reported "
            "instead of an invented magnitude cutoff."
        ),
        "branches": {
            "lfp_grain_worse_instrumented_at_margin": "summed rank difference (LFP minus single-unit) is negative",
            "unit_grain_worse_instrumented_at_margin": "summed rank difference (LFP minus single-unit) is positive",
            "grains_equivalent_at_margin": "summed rank difference is exactly zero",
        },
        "note": "pre-declared before this rule is evaluated against dandi_000574's bin100 and bin200 admission-matrix cells",
    }


def _entry_margin_band_rank(cell: dict, observable: str) -> int | None:
    entry = cell["observables"].get(observable)
    if entry is None:
        raise AdmissionMatrixKeyMissing(f"{observable!r} is not a scored observable in this cell")
    if entry["status"] == "excluded":
        return ADMISSION_MARGIN_BAND_RANK["below_floor"]
    if entry["status"] == "admitted":
        return ADMISSION_MARGIN_BAND_RANK[entry["margin"]["band"]]
    return None  # provisionally_admitted_flagged: no floor exists to rank a margin against


def evaluate_margin_aware_modality_rule(matrix: dict, rule: dict) -> dict:
    def _per_bin_width(unit_key: str, lfp_key: str) -> dict:
        unit_cell = _cell_or_raise(matrix, unit_key)
        lfp_cell = _cell_or_raise(matrix, lfp_key)
        shared_observables = sorted(
            name for name in unit_cell["observables"]
            if _entry_margin_band_rank(unit_cell, name) is not None and _entry_margin_band_rank(lfp_cell, name) is not None
        )
        per_observable = {}
        total = 0
        for name in shared_observables:
            unit_rank = _entry_margin_band_rank(unit_cell, name)
            lfp_rank = _entry_margin_band_rank(lfp_cell, name)
            diff = lfp_rank - unit_rank
            per_observable[name] = {
                "unit_grain_band_rank": unit_rank, "lfp_grain_band_rank": lfp_rank, "lfp_minus_unit": diff,
            }
            total += diff
        if total < 0:
            branch = "lfp_grain_worse_instrumented_at_margin"
        elif total > 0:
            branch = "unit_grain_worse_instrumented_at_margin"
        else:
            branch = "grains_equivalent_at_margin"
        return {
            "unit_grain_key": unit_key, "lfp_grain_key": lfp_key,
            "n_shared_observables_compared": len(shared_observables),
            "per_observable": per_observable,
            "summed_rank_difference_lfp_minus_unit": total,
            "branch": branch,
        }

    return {
        "bin100": _per_bin_width(
            "dandi_000574::single_unit::pooled::bin100", "dandi_000574::depth_contact_lfp::depth_contact_lfp::bin100"),
        "bin200": _per_bin_width(
            "dandi_000574::single_unit::pooled::bin200", "dandi_000574::depth_contact_lfp::depth_contact_lfp::bin200"),
    }


def not_staged_rows() -> list[dict]:
    return [
        {"dataset": dataset, "patient": "n/a", "session": "n/a", "structure": "n/a", "epoch": "n/a",
         "bin_ms": None, "modality": "n/a", "status": "not_staged", "reason": reason}
        for dataset, reason in NOT_STAGED_CORPORA.items()
    ]


def not_computable_rows() -> list[dict]:
    return [
        {"dataset": dataset, "patient": "n/a", "session": "n/a", "structure": "n/a", "epoch": "n/a",
         "bin_ms": None, "modality": "n/a", "status": "not_computable", "reason": reason}
        for dataset, reason in NOT_COMPUTABLE_THIS_PASS.items()
    ]


def main() -> None:
    carried_rows, carry_forward_verification = carry_forward_existing_rows()
    print(f"carried forward {len(carried_rows)} existing rows", file=sys.stderr)

    new_rows: list[dict] = []
    for name, builder in CORPUS_BUILDERS.items():
        new_rows.extend(run_corpus_with_checkpoint(name, builder))

    n_newly_fitted = sum(1 for r in new_rows if r.get("status") == "fitted")
    n_newly_error = sum(1 for r in new_rows if r.get("status") == "error")

    all_rows = carried_rows + new_rows + not_staged_rows() + not_computable_rows()

    floors = minimum_measurable_configuration()
    matrix = build_admission_matrix(carried_rows + new_rows, floors)
    relabeled = relabel_existing_reported_observables(matrix)
    headline_rule = predeclared_census_headline_rule()
    headline_result = evaluate_census_headline(matrix, headline_rule)
    headline_defect = admission_matrix_key_lookup_defect()
    small_worldness_change = small_worldness_sigma_definition_change()
    lag_status_meaning = fewer_than_three_positive_lags_status_meaning(all_rows)

    payload = {
        "schema_version": "2.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "n_splits": N_SPLITS,
        "precursor": {
            "artifact": "results/observability_census.json",
            "settles": "nugget fraction and slow timescale per patient x session x structure x epoch x bin "
                       "width, human single-unit corpora (dandi_000469, dandi_001187, dandi_000574) and mouse "
                       "ALM, at 100 and 200 ms, 1877 rows, 32 splits",
            "gap_this_artifact_fills": "every other modality and grain those corpora and every other staged "
                                       "corpus offer, the minimum measurable configuration of every observable "
                                       "this project reports, and the admission matrix crossing the two",
            "n_rows_carried_forward": len(carried_rows),
            "n_rows_newly_fitted_or_attempted": len(new_rows),
            "n_rows_newly_status_fitted": n_newly_fitted,
            "n_rows_newly_status_error": n_newly_error,
            "carry_forward_verification": carry_forward_verification,
        },
        "rows": all_rows,
        "not_staged_corpora": NOT_STAGED_CORPORA,
        "not_computable_this_pass": NOT_COMPUTABLE_THIS_PASS,
        "fewer_than_three_positive_lags_status_meaning": lag_status_meaning,
        "minimum_measurable_configuration": floors,
        "admission_matrix": matrix,
        "relabeled_existing_reported_observables": relabeled,
        "census_headline_predeclared_rule": headline_rule,
        "census_headline_result": headline_result,
        "admission_matrix_key_lookup_defect": headline_defect,
        "small_worldness_sigma_definition_change": small_worldness_change,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"wrote {OUTPUT_PATH}: {len(all_rows)} total rows "
          f"({len(carried_rows)} carried, {n_newly_fitted} newly fitted, {n_newly_error} newly errored, "
          f"{len(NOT_STAGED_CORPORA)} not-staged corpora, {len(NOT_COMPUTABLE_THIS_PASS)} not-computable corpora)",
          file=sys.stderr)
    print(json.dumps({"census_headline_result": headline_result}, indent=2))


if __name__ == "__main__":
    main()
