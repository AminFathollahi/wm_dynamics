from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from scipy.io import loadmat

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import (  # noqa: E402
    DANDI_000004_RESPONSE_RANGE, DANDI_000574_PROBE_OFFSET_S, alm_data_directory, watters_behaviour,
    watters_session_dates,
)
from project_config import dataset_path  # noqa: E402
from run_human_drift_spine_001187_000673 import canonical_sessions, _trial_group  # noqa: E402


def _finite_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def response_probe_latency_field(trials) -> np.ndarray:
    return trials["timestamps_Response"][:] - trials["timestamps_Probe"][:]


def response_time_field(trials) -> np.ndarray:
    return trials["response_time"][:]


def dandi_000469_latency(root: Path) -> Iterator[dict]:
    directory = root / "000469"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*_ses-2_ecephys+image.nwb")):
            with h5py.File(path, "r") as handle:
                trials = handle.get("intervals/trials")
                if trials is None or "timestamps_Response" not in trials:
                    continue
                latency = response_probe_latency_field(trials)
            latency = _finite_positive(latency)
            if len(latency) == 0:
                continue
            yield {"dataset": "dandi_000469", "patient": subject_dir.name, "session": path.stem,
                   "n_trials": len(latency), "latency_s": latency,
                   "derivation": "timestamps_Response - timestamps_Probe, /intervals/trials"}


def dandi_000673_latency(root: Path) -> Iterator[dict]:
    directory = root / "000673"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(path, "r") as handle:
                trials = handle.get("intervals/trials")
                if trials is None or "timestamps_Response" not in trials:
                    continue
                latency = response_probe_latency_field(trials)
            latency = _finite_positive(latency)
            if len(latency) == 0:
                continue
            yield {"dataset": "dandi_000673", "patient": subject_dir.name, "session": path.stem,
                   "n_trials": len(latency), "latency_s": latency,
                   "derivation": "timestamps_Response - timestamps_Probe, /intervals/trials"}


def dandi_001187_latency(root: Path) -> Iterator[dict]:
    for meta in canonical_sessions():
        if meta["primary_release"] != "001187":
            continue
        path = root / meta["primary_path"]
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            trials = _trial_group(handle, "001187")
            if "response_time" not in trials:
                continue
            latency = response_time_field(trials)
        latency = _finite_positive(latency)
        if len(latency) == 0:
            continue
        yield {"dataset": "dandi_001187", "patient": meta["patient"], "session": path.stem,
               "n_trials": len(latency), "latency_s": latency,
               "derivation": "explicit response_time column, /intervals/WM_trials"}


def dandi_000574_latency(root: Path) -> Iterator[dict]:
    directory = root / "000574"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(path, "r") as handle:
                trials = handle.get("intervals/trials")
                if trials is None or "response_time" not in trials:
                    continue
                response_time = trials["response_time"][:]
                start_time = trials["start_time"][:]
            latency = _finite_positive(response_time - (start_time + DANDI_000574_PROBE_OFFSET_S))
            if len(latency) == 0:
                continue
            yield {"dataset": "dandi_000574", "patient": subject_dir.name, "session": path.stem,
                   "n_trials": len(latency), "latency_s": latency,
                   "derivation": "response_time is an absolute session-clock timestamp, not a "
                                  "raw latency; resolved as response_time - (start_time + "
                                  f"{DANDI_000574_PROBE_OFFSET_S}s), reusing the probe-onset "
                                  "convention corpus_sessions.iter_dandi_000574 already applies "
                                  "to this corpus (corpus_sessions.DANDI_000574_PROBE_OFFSET_S)"}


def dandi_000004_latency(root: Path) -> Iterator[dict]:
    directory = root / "000004"
    required = ("stim_phase", "new_old_labels_recog", "response_value", "response_time", "stim_on_time")
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(path, "r") as handle:
                trials = handle.get("intervals/trials")
                if trials is None or not all(c in trials for c in required):
                    continue
                phase = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in trials["stim_phase"][:]])
                labels = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in trials["new_old_labels_recog"][:]])
                responses = trials["response_value"][:].astype(float)
                response_time = trials["response_time"][:].astype(float)
                stim_on = trials["stim_on_time"][:].astype(float)
            candidate = (phase == "recog") & np.isin(labels, ("0", "1")) & np.isin(responses, DANDI_000004_RESPONSE_RANGE)
            latency = _finite_positive((response_time - stim_on)[candidate])
            if len(latency) == 0:
                continue
            yield {"dataset": "dandi_000004", "patient": subject_dir.name, "session": path.stem,
                   "n_trials": len(latency), "latency_s": latency,
                   "derivation": "response_time and stim_on_time share one absolute session clock "
                                  "(both NWB fields, verified by direct comparison); latency = "
                                  "response_time - stim_on_time on admitted recognition trials"}


def watters_2026_latency(root: Path) -> Iterator[dict]:
    behaviour = watters_behaviour(root)
    for animal, session_date, variant in watters_session_dates(root):
        if (animal, session_date) not in behaviour.index.droplevel(2).unique():
            continue
        rows = behaviour.loc[(animal, session_date)]
        latency = _finite_positive(rows["reaction_time_ms"].to_numpy(dtype=float) / 1000.0)
        if len(latency) == 0:
            continue
        yield {"dataset": "watters_2026", "patient": animal, "session": f"{animal}_{session_date}",
               "n_trials": len(latency), "latency_s": latency,
               "derivation": "reaction_time_ms field (already read by watters_behaviour), converted to seconds"}


def inagaki_alm5_latency(root: Path) -> Iterator[dict]:
    directory = alm_data_directory(root)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.mat")):
        units = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["unit"])
        behavior = units[0].Behavior
        first_lick = np.asarray(behavior.First_lick, dtype=float).reshape(-1)
        delay_start = np.asarray(behavior.Delay_start, dtype=float).reshape(-1)
        delay_dur = np.asarray(behavior.delay_dur, dtype=float).reshape(-1)
        latency = _finite_positive(first_lick - (delay_start + delay_dur))
        if len(latency) == 0:
            continue
        yield {"dataset": "inagaki_alm5", "patient": path.stem.split("_")[0], "session": path.stem,
               "n_trials": len(latency), "latency_s": latency,
               "derivation": "First_lick - (Delay_start + delay_dur), i.e. time from the delay's own "
                              "end (go cue) to the first detected lick"}


def alagapan_phase_stimulation_latency(root: Path) -> Iterator[dict]:
    task_dir = root / "alagapan, cell_reports" / "Task Performance"
    if not task_dir.is_dir():
        return
    for path in sorted(task_dir.glob("*_Summary.csv")):
        with open(path) as handle:
            rows = list(csv.DictReader(handle))
        latency = _finite_positive(np.array([
            float(row["ReactionTime"]) if row["ReactionTime"] not in ("", "NaN", "nan") else np.nan
            for row in rows
        ]))
        if len(latency) == 0:
            continue
        patient = path.stem.split("_")[0]
        yield {"dataset": "alagapan_phase_stimulation", "patient": patient, "session": path.stem,
               "n_trials": len(latency), "latency_s": latency,
               "derivation": "ReactionTime column, per-patient Task Performance summary CSV"}


def pfc4_latency(root: Path) -> Iterator[dict]:
    directory = dataset_path("pfc4")
    if directory is None or not directory.is_dir():
        return
    for animal_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for path in sorted(animal_dir.rglob("*.mat")):
            data = loadmat(path, squeeze_me=True, struct_as_record=False)
            result = data["result"]
            header = list(result[0])
            if "KU" not in header or "SF2" not in header:
                continue
            ku_col, sf2_col = header.index("KU"), header.index("SF2")
            latencies = []
            for r in range(1, result.shape[0]):
                row = result[r]
                ku, sf2 = np.atleast_1d(row[ku_col]), np.atleast_1d(row[sf2_col])
                if ku.size != 1 or sf2.size != 1:
                    continue
                latencies.append((float(ku[0]) - float(sf2[0])) / 1000.0)
            latency = _finite_positive(np.array(latencies))
            latency = latency[latency < 5.0]
            if len(latency) == 0:
                continue
            yield {"dataset": "pfc4", "patient": animal_dir.name, "session": path.stem,
                   "n_trials": len(latency), "latency_s": latency,
                   "derivation": "(KU - SF2) / 1000, key-release time minus second-stimulus offset, "
                                  "both in the file's own millisecond clock"}


def _ram_bids_latency(root: Path, dataset_key: str, local_path: str) -> Iterator[dict]:
    directory = root / local_path
    if not directory.is_dir():
        return
    for events_path in sorted(directory.glob("sub-*/ses-*/ieeg/*_events.tsv")):
        with open(events_path) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        wanted = [r for r in rows if r["trial_type"] in ("REC_WORD", "REC_WORD_VV")]
        latency = _finite_positive(np.array([
            float(r["response_time"]) if r["response_time"] not in ("n/a", "") else np.nan
            for r in wanted
        ]))
        if len(latency) == 0:
            continue
        subject = wanted[0]["subject"] if wanted else events_path.stem
        yield {"dataset": dataset_key, "patient": f"sub-{subject}", "session": events_path.stem,
               "n_trials": len(latency), "latency_s": latency,
               "derivation": "response_time on REC_WORD and REC_WORD_VV events only (recall-period "
                              "word production latency); PROB events excluded -- PROB is the "
                              "arithmetic distractor task, not a memory response, verified directly "
                              "against the released events tables"}


def ram_ds005489_openloop_latency(root: Path) -> Iterator[dict]:
    yield from _ram_bids_latency(root, "ram_ds005489_openloop", "ds005489-download")


def ram_ds005557_closedloop_latency(root: Path) -> Iterator[dict]:
    yield from _ram_bids_latency(root, "ram_ds005557_closedloop", "ds005557-download")


def ds004752_latency(root: Path) -> Iterator[dict]:
    directory = root / "ds004752"
    if not directory.is_dir():
        return
    for events_path in sorted(directory.glob("sub-*/ses-*/ieeg/*_events.tsv")):
        with open(events_path) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        latency = _finite_positive(np.array([
            float(r["ResponseTime"]) if r.get("ResponseTime", "n/a") not in ("n/a", "") else np.nan
            for r in rows
        ]))
        if len(latency) == 0:
            continue
        yield {"dataset": "ds004752", "patient": events_path.parents[2].name, "session": events_path.stem,
               "n_trials": len(latency), "latency_s": latency,
               "derivation": "explicit ResponseTime column, BIDS events.tsv"}


LATENCY_EXTRACTORS = {
    "dandi_000469": dandi_000469_latency,
    "dandi_001187": dandi_001187_latency,
    "dandi_000673": dandi_000673_latency,
    "dandi_000574": dandi_000574_latency,
    "dandi_000004": dandi_000004_latency,
    "watters_2026": watters_2026_latency,
    "inagaki_alm5": inagaki_alm5_latency,
    "alagapan_phase_stimulation": alagapan_phase_stimulation_latency,
    "pfc4": pfc4_latency,
    "ram_ds005489_openloop": ram_ds005489_openloop_latency,
    "ram_ds005557_closedloop": ram_ds005557_closedloop_latency,
    "ds004752": ds004752_latency,
}

LATENCY_UNAVAILABLE = {
    "panichello_2024": "released .mat files carry only cueAng, cueAngIdx, isCorr, spks, tc -- no "
                        "response-time or saccade-timing field",
    "macaque_pfc_microstimulation": "no latency scalar is shipped; derivable from the eyedat traces "
                                     "(see scripts/run_macaque_pfc_microstimulation_recovery_latency.py, "
                                     "arm three of this task, which reports its own admitted/refused counts)",
    "haslacher_clam_tacs": "the staged release under Working-memory-CLAM-tACS/Data holds only EEG "
                            "(.eeg/.vhdr/.vmrk) and a channel-covariance matrix (C_B.mat) per patient; "
                            "no trial-level behavioural file (accuracy or response time) is present",
    "pfc3": "the released trial struct (Cue_onT, Sample_onT, Reward_onT, trialnum, TS, IsMatch, fix, "
            "cuerate, cuedelay, samplerate, sampledelay) carries no response or movement timestamp; "
            "Reward_onT conflates response timing with juice-delivery latency and is not used as a proxy",
    "wolff_eeg_impulse": "Results_header is angle_left, angle_right, cue, probe_rotation, accuracy -- "
                          "no response-time column",
    "kai_miller_nback": "the release ships only a continuous per-finger movement trace (time x n_fingers), "
                         "not a trial-level response or accuracy label; scripts/run_miller_drift_spine.py "
                         "already documents that deriving one without independent validation risks "
                         "fabrication, and that validation is out of scope here",
    "ds005034": "complete event vocabulary is DIN104, DIN106, DIN114, DIN116, DIN124, DIN126 (delay "
                "onsets) and DIN200 (probe presentation); no response marker exists in any file",
    "ds006848": "only retention and retrieval markers are timed events; the companion beh.tsv carries "
                "accuracy (NCorrect, partialScore, triggerCorrect) but no response-time field",
}
