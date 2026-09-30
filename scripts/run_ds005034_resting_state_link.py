#!/usr/bin/env python3
"""ds005034 ships a resting-state EEG recording for every one of its 50 sessions
(25 participants x sham/verum) that no existing analysis reads. The release's own
README states the within-session order explicitly: 20-minute stimulation, then
3-minute-eyes-closed + 1.5-minute-eyes-open resting EEG, then the working-memory
task. So this recording is POST-stimulation and PRE-task -- never described here
as a pre-stimulation baseline.

Two pre-declared families:

resting_state_arm_contrast -- within-participant paired verum-minus-sham contrast
on resting-state band power, eyes-closed and eyes-open reported separately (never
pooled). Primary: frontal-ROI theta (the stimulation was 6 Hz at Fpz-CPz).
Secondary, a separate multiplicity family per segment: frontal alpha/beta and all
three posterior bands.

resting_state_predicts_task_decodability -- does a session's resting-state read
carry information about that session's own delay-period decodability. Eyes-closed
frontal theta (the same primary feature as the first family; eyes-open fails a
minimum-duration floor in this release, see below) against
operation_forward_vs_transform's diagonal AUC from
scripts/run_ds005034_operation_versus_load_decoding.py, recomputed per session by
importing that script's own estimator (its own delivered artifact stores only
participant/arm-resampled aggregates, and no per-session checkpoint for it exists
on this data root). Reported as two separate cells -- within-participant (paired
deltas, immune to stable between-person traits) and between-participant
(participant-level arm-averaged values, confounded by stable traits) -- never
pooled into one number.

A release-level finding, verified directly against every one of the 50 rest
*_eeg.set files on disk: despite the README's "1.5 minutes eyes open", the
released recording is truncated to almost exactly 1.0 s after the eyes-open
("eyeo") event marker in EVERY session (tail duration 1.0020 +/- 0.00001 s
across all 50 files) -- the shipped eyes-open segment carries essentially no
usable data. This is a property of the release, not a decision made here: the
eyes-open cells in both families are still fit and reported (never silently
dropped), and fail structurally with this reason recorded as a real failure,
never serialised as a zero.

Session order (which arm a participant received first) is not in
participants.tsv or any scans.tsv (neither exists in this release). It IS
recoverable from the acquisition timestamp embedded in each .set file's own
EGI/EEGLAB metadata (etc.recordingtime, a MATLAB datenum), cross-checked against
the session index embedded in the same structure's Patient ID string
(etc.subject.fields 'Patient ID', e.g. 'WMt_02_1RS' vs 'WMt_02_2RS'). Carried as
a reported covariate (a descriptive split of the primary contrast by order), never
used to exclude or reweight a participant, since carryover cannot be separated
from the arm effect in this two-arm crossover design.

Output: results/ds005034_resting_state_link.json
Per-session checkpoints: results/.checkpoints/run_ds005034_resting_state_link/

Run:
    conda run -n wm_dynamics python scripts/run_ds005034_resting_state_link.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import norm, pearsonr

warnings.filterwarnings("ignore")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/wm_dynamics_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wm_dynamics_matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from preprocessing import bandpass_filter, common_average_reference  # noqa: E402
from provenance import canonical_json, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from provenance import _json_safe  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from project_config import data_root
from data_integrity import missing_files  # noqa: E402
from subject_independence import resolve_group, count_independent_groups  # noqa: E402
from corpus_sessions import MIN_TRIALS  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, stable_seed, fdr_bh, paired_sign_flip_test, pearson_permutation_test,
    bootstrap_ci, minimum_detectable_paired_difference,
)
from preprocessing import THETA_ROI, TARGET_SFREQ, periodogram_band_power, _mne_session_context, _global_bad_channels, _interpolate
from preprocessing import BANDS, POSTERIOR_ROI, MAX_GLOBAL_BAD_FRACTION
from corpus_sessions import _load_ds005034_session
from preprocessing import session_cells
from statistics import FDR_ALPHA
from info_decoding import MIN_TRIALS_PER_CLASS

RESULTS = ROOT / "results"
ANALYSIS_ID = "ds005034_resting_state_link"
ANALYSIS_VERSION = "2026-09-21"
SESSIONS = ("sham", "verum")
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_ds005034_resting_state_link"
CHECKPOINT_SCHEMA = "ds005034_resting_state_link_v1"

ROI_DEFS = {"frontal": THETA_ROI, "posterior": POSTERIOR_ROI}
SEGMENTS = ("eyes_closed", "eyes_open")
PRIMARY_KEY = "frontal_theta"
SECONDARY_KEYS = ("frontal_alpha", "frontal_beta", "posterior_theta", "posterior_alpha", "posterior_beta")
SEGMENT_MIN_SECONDS = 30.0
SEGMENT_MIN_SECONDS_REASON = (
    "eyes-closed segments in this release run 150-270 s in every session and eyes-open segments run "
    "almost exactly 1.0 s in every session (verified against all 50 rest recordings); 30 s cleanly "
    "separates the two without being tuned to either observed value"
)

RESTING_STATE_ORDERING_NOTE = (
    "release README, verbatim: 'Following either a 20-minute verum or sham stimulation applied to "
    "Fpz-CPz at 1 mA and 6 Hz, the participants performed WM tasks, while EEG was recorded... In "
    "addition, before the working memory task, resting state EEG with eyes closed was recorded for 3 "
    "minutes and with eyes open for 1.5 minutes.' Order within a session: stimulation, then this "
    "resting recording, then the working-memory task. This recording is POST-stimulation and "
    "PRE-task -- never a pre-stimulation baseline."
)
EYES_OPEN_TRUNCATION_NOTE = (
    "contradicts the README's '1.5 minutes eyes open': every one of the 50 released rest *_eeg.set "
    "files ends 1.0020 +/- 0.00001 s after its own eyes-open ('eyeo') event marker, regardless of "
    "when that marker falls in the recording. The eyes-closed segment is fully present (150-270 s, "
    "matching the README's 3 minutes); the eyes-open segment is not -- the release carries only a "
    "roughly 1-second stub of it. Verified directly against every file on disk, not assumed."
)
ORDER_RECOVERY_METHOD_PREFIX = (
    "participants.tsv carries only participant_id/sex/age/hand; no scans.tsv or sessions.tsv exists "
    "in this release. Session order is recovered instead from metadata embedded in each rest .set "
    "file's own EEGLAB/EGI structure: etc.recordingtime (a MATLAB datenum, the true acquisition "
    "timestamp) gives an unambiguous per-participant order; etc.subject.fields carries a 'Patient ID' "
    "string embedding a session index (e.g. 'WMt_02_1RS' vs 'WMt_02_2RS', or 'WMt_04_01' vs "
    "'WMt_04_2') whose last digit run corroborates that order where both sessions' fields carry one."
)


def _checkpoint_path(session_key):
    return CHECKPOINT_DIR / f"{session_key}.json"


def load_session_checkpoint(session_key):
    path = _checkpoint_path(session_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("complete") is not True:
        return None
    return restore_checkpoint(payload["record"])


def save_session_checkpoint(session_key, record):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(session_key)
    payload = {"schema": CHECKPOINT_SCHEMA, "complete": True, "record": checkpoint_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=CHECKPOINT_DIR, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _matlab_datenum_to_iso(datenum: float) -> str:
    return (datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)).isoformat()


def _parse_session_index_from_patient_id(patient_id):
    if not patient_id:
        return None
    runs = re.findall(r"\d+", patient_id)
    if not runs:
        return None
    value = int(runs[-1])
    return value if value in (1, 2) else None


def recording_order_metadata(set_path: Path) -> dict | None:
    payload = loadmat(str(set_path), squeeze_me=True, variable_names=("etc",))
    etc = payload["etc"]
    if etc.dtype.names is None or "recordingtime" not in etc.dtype.names or "subject" not in etc.dtype.names:
        return None
    recordingtime = float(etc["recordingtime"].item())
    fields = etc["subject"].item()["fields"].item()
    field_map = {str(name): (str(value) if isinstance(value, str) else None) for name, value, _dtype in fields}
    return {
        "matlab_datenum": recordingtime,
        "acquisition_datetime_iso": _matlab_datenum_to_iso(recordingtime),
        "patient_id_field": field_map.get("Patient ID"),
    }


def parse_rest_events(events_path: Path) -> dict:
    with events_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    eyec = sorted(float(r["onset"]) for r in rows if r["value"] == "eyec")
    eyeo = sorted(float(r["onset"]) for r in rows if r["value"] == "eyeo")
    if not eyeo:
        raise ValueError("no eyeo event marker in this session's rest events.tsv")
    eyes_open_start = eyeo[0]
    eyes_closed_start = max(eyec[0], 0.0) if eyec else 0.0
    return {"eyes_closed": (eyes_closed_start, eyes_open_start), "eyes_open": (eyes_open_start, None)}


def segment_bounds(events_path: Path, total_duration: float) -> dict:
    raw_bounds = parse_rest_events(events_path)
    return {name: (lo, hi if hi is not None else total_duration) for name, (lo, hi) in raw_bounds.items()}


def segment_band_power(filtered, times, interval, csd_transform, channel_index) -> dict:
    duration = interval[1] - interval[0]
    if duration < SEGMENT_MIN_SECONDS:
        return {
            "status": "failure_nan", "duration_seconds": duration,
            "reason": f"segment covers {duration:.3f} s, below the {SEGMENT_MIN_SECONDS:g} s floor for a "
                      f"periodogram estimate ({SEGMENT_MIN_SECONDS_REASON})",
        }
    power = periodogram_band_power(filtered, TARGET_SFREQ, interval, times, csd_transform)
    log_power = np.log(np.maximum(power, np.finfo(float).tiny))
    band_index = {name: i for i, name in enumerate(BANDS)}
    roi_band = {}
    for roi_name, channels in ROI_DEFS.items():
        idx = [channel_index[c] for c in channels]
        for band_name, b in band_index.items():
            roi_band[f"{roi_name}_{band_name}"] = float(log_power[idx, b].mean())
    return {"status": "computed", "duration_seconds": duration, "log_power": roi_band}


def analyze_rest_session(set_path: Path, events_path: Path, seed: int) -> dict:
    raw, info, csd_transform = _mne_session_context(set_path)
    payload = loadmat(str(set_path), variable_names=("data", "srate"), squeeze_me=True)
    data_uv = np.asarray(payload["data"], dtype=np.float64)
    sfreq = float(payload["srate"])
    if sfreq != 1000.0 or sfreq != float(raw.info["sfreq"]):
        raise ValueError(f"unexpected sampling frequency {sfreq}")
    total_duration = data_uv.shape[1] / sfreq

    global_bads = _global_bad_channels(data_uv, info, seed)
    if len(global_bads) > int(np.floor(MAX_GLOBAL_BAD_FRACTION * data_uv.shape[0])):
        raise ValueError(f"{len(global_bads)} globally bad channels exceeds the {MAX_GLOBAL_BAD_FRACTION:.0%} gate")
    data_uv = _interpolate(data_uv, info, global_bads)

    downsampled = resample_poly(data_uv, int(TARGET_SFREQ), int(sfreq), axis=1)
    filtered = bandpass_filter(downsampled.T, 1.0, 45.0, TARGET_SFREQ).T
    filtered = common_average_reference(filtered.T).T
    times = np.arange(filtered.shape[1]) / TARGET_SFREQ
    channel_index = {name: i for i, name in enumerate(raw.ch_names)}

    bounds = segment_bounds(events_path, total_duration)
    segments = {
        name: segment_band_power(filtered, times, interval, csd_transform, channel_index)
        for name, interval in bounds.items()
    }
    return {
        "status": "computed", "total_duration_seconds": total_duration,
        "global_bad_channels": global_bads, "segments": segments,
        "order_metadata": recording_order_metadata(set_path),
    }


def analyze_decode_session(directory: Path, subject_dir: Path, session: str) -> dict:
    sess = _load_ds005034_session(directory, subject_dir, session)
    if sess is None:
        return {
            "status": "failure_nan",
            "reason": f"missing memory-task recording or fewer than {MIN_TRIALS} admitted delay-window trials",
        }
    cells = session_cells(sess)
    primary = cells.get("operation_forward_vs_transform")
    if primary is None or primary.get("diag_mean_auc") is None:
        return {
            "status": "failure_nan", "cells": cells,
            "reason": f"operation_forward_vs_transform did not clear the {MIN_TRIALS_PER_CLASS}-trial-per-class floor",
        }
    return {"status": "computed", "cells": cells, "primary_diag_auc": primary["diag_mean_auc"]}


def compute_session(directory: Path, participant: str, session: str, rest_set_path: Path) -> dict:
    session_key = f"{participant}_ses-{session}"
    record = {"participant": participant, "stimulation_session": session}

    events_path = rest_set_path.with_name(rest_set_path.name.replace("_eeg.set", "_events.tsv"))
    seed = stable_seed(f"{ANALYSIS_ID}|rest|{session_key}")
    try:
        record["rest"] = analyze_rest_session(rest_set_path, events_path, seed)
    except Exception as exc:
        record["rest"] = {"status": "failure_nan", "reason": f"{type(exc).__name__}: {exc}"}

    try:
        record["decode"] = analyze_decode_session(directory, directory / participant, session)
    except Exception as exc:
        record["decode"] = {"status": "failure_nan", "reason": f"{type(exc).__name__}: {exc}"}
    return record


def synthesize_order(per_session: dict, participants: list[str]) -> dict:
    rows = {}
    for participant in participants:
        sham = per_session[f"{participant}_ses-sham"]["rest"]
        verum = per_session[f"{participant}_ses-verum"]["rest"]
        sham_order = sham.get("order_metadata") if sham.get("status") == "computed" else None
        verum_order = verum.get("order_metadata") if verum.get("status") == "computed" else None
        if not sham_order or not verum_order:
            rows[participant] = {"status": "not_recoverable", "reason": "acquisition metadata missing from one or both .set files"}
            continue
        sham_dt, verum_dt = sham_order["matlab_datenum"], verum_order["matlab_datenum"]
        order_by_time = "sham_first" if sham_dt < verum_dt else "verum_first"
        sham_idx = _parse_session_index_from_patient_id(sham_order["patient_id_field"])
        verum_idx = _parse_session_index_from_patient_id(verum_order["patient_id_field"])
        if sham_idx is not None and verum_idx is not None and sham_idx != verum_idx:
            order_by_id = "sham_first" if sham_idx < verum_idx else "verum_first"
            id_corroborates = order_by_id == order_by_time
        else:
            order_by_id, id_corroborates = "uninformative", None
        rows[participant] = {
            "status": "recovered", "order": order_by_time, "gap_days": abs(verum_dt - sham_dt),
            "order_by_patient_id_field": order_by_id, "id_corroboration": id_corroborates,
            "sham_acquisition_datetime": sham_order["acquisition_datetime_iso"],
            "verum_acquisition_datetime": verum_order["acquisition_datetime_iso"],
        }
    recovered = [r for r in rows.values() if r["status"] == "recovered"]
    informative = [r for r in recovered if r["id_corroboration"] is not None]
    n_id_corroborated = sum(1 for r in informative if r["id_corroboration"] is True)
    method = (
        f"{ORDER_RECOVERY_METHOD_PREFIX} recordingtime recovered order for {len(recovered)} of "
        f"{len(participants)} participants; the patient-id field was informative and corroborated "
        f"that order for {n_id_corroborated} of those {len(recovered)}."
    )
    return {
        "method": method,
        "n_participants": len(participants), "n_recovered": len(recovered),
        "n_sham_first": sum(1 for r in recovered if r["order"] == "sham_first"),
        "n_verum_first": sum(1 for r in recovered if r["order"] == "verum_first"),
        "n_id_corroborated": n_id_corroborated,
        "n_id_informative": len(informative),
        "per_participant": rows,
    }


def _paired_cell(verum_vals, sham_vals, tag) -> dict:
    verum_vals, sham_vals = np.asarray(verum_vals, dtype=float), np.asarray(sham_vals, dtype=float)
    diffs = verum_vals - sham_vals
    result = paired_sign_flip_test(
        verum_vals, sham_vals, n_perm=100_000, n_boot=10_000, alternative="two-sided",
        rng=np.random.default_rng(stable_seed(f"{ANALYSIS_ID}|{tag}")),
    )
    result.pop("null", None)
    result["minimum_detectable_difference_at_80pct_power"] = minimum_detectable_paired_difference(diffs)
    result["n"] = int(len(diffs))
    result["cell_status"] = None
    return result


def _order_covariate_breakdown(per_session: dict, participants: list[str], order_info: dict) -> dict:
    groups = {"sham_first": [], "verum_first": []}
    for participant in participants:
        row = order_info["per_participant"].get(participant, {})
        if row.get("status") != "recovered":
            continue
        sham_seg = per_session[f"{participant}_ses-sham"]["rest"].get("segments", {}).get("eyes_closed")
        verum_seg = per_session[f"{participant}_ses-verum"]["rest"].get("segments", {}).get("eyes_closed")
        if not sham_seg or not verum_seg or sham_seg["status"] != "computed" or verum_seg["status"] != "computed":
            continue
        groups[row["order"]].append(verum_seg["log_power"][PRIMARY_KEY] - sham_seg["log_power"][PRIMARY_KEY])
    return {
        "note": (
            "descriptive only -- not a formal test, and not used to exclude or reweight any participant. "
            "Carryover between the two sessions cannot be separated from the arm effect in this two-arm "
            "crossover design; this splits the primary eyes-closed frontal-theta verum-minus-sham delta "
            "by which arm each participant received first."
        ),
        "primary_eyes_closed_frontal_theta_verum_minus_sham_by_order": {
            group: {
                "n": len(vals), "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            }
            for group, vals in groups.items()
        },
    }


def synthesize_family1(per_session: dict, participants: list[str], order_info: dict) -> dict:
    out = {}
    for segment in SEGMENTS:
        cells = {}
        for key in (PRIMARY_KEY,) + SECONDARY_KEYS:
            verum_vals, sham_vals, n_excluded = [], [], 0
            for participant in participants:
                sham_rest = per_session[f"{participant}_ses-sham"]["rest"]
                verum_rest = per_session[f"{participant}_ses-verum"]["rest"]
                sham_seg = sham_rest.get("segments", {}).get(segment) if sham_rest.get("status") == "computed" else None
                verum_seg = verum_rest.get("segments", {}).get(segment) if verum_rest.get("status") == "computed" else None
                if not sham_seg or not verum_seg or sham_seg["status"] != "computed" or verum_seg["status"] != "computed":
                    n_excluded += 1
                    continue
                sham_vals.append(sham_seg["log_power"][key])
                verum_vals.append(verum_seg["log_power"][key])
            if len(verum_vals) < 2:
                cells[key] = {
                    "cell_status": "failure_nan", "n": len(verum_vals), "n_excluded": n_excluded,
                    "reason": "fewer than 2 participants had a computed value in both arms for this segment"
                              + (f" ({EYES_OPEN_TRUNCATION_NOTE})" if segment == "eyes_open" else ""),
                }
                continue
            cell = _paired_cell(verum_vals, sham_vals, f"family1|{segment}|{key}")
            cell["n_excluded"] = n_excluded
            cells[key] = cell

        secondary_present = [k for k in SECONDARY_KEYS if cells[k].get("p_value") is not None]
        if secondary_present:
            q = fdr_bh(np.array([cells[k]["p_value"] for k in secondary_present]), alpha=FDR_ALPHA)
            for k, qv in zip(secondary_present, q["q_values"]):
                cells[k]["q_value"] = float(qv)
                cells[k]["cell_status"] = "estimated"
        for k in SECONDARY_KEYS:
            cells[k].setdefault("q_value", None)

        if cells[PRIMARY_KEY].get("p_value") is not None:
            cells[PRIMARY_KEY]["cell_status"] = "estimated"
        cells[PRIMARY_KEY]["q_value"] = None

        out[segment] = {
            "primary": {PRIMARY_KEY: cells[PRIMARY_KEY]},
            "secondary": {k: cells[k] for k in SECONDARY_KEYS},
        }
    out["order_covariate_descriptive"] = _order_covariate_breakdown(per_session, participants, order_info)
    return out


def _pearson_r_of_columns(arr: np.ndarray) -> float:
    if np.std(arr[:, 0]) == 0.0 or np.std(arr[:, 1]) == 0.0:
        return float("nan")
    return float(pearsonr(arr[:, 0], arr[:, 1])[0])


def _correlation_mdd(ci_lower: float, ci_upper: float, alpha: float = 0.05, power: float = 0.80) -> dict:
    z_ci = float(norm.ppf(0.975))
    se = (ci_upper - ci_lower) / (2.0 * z_ci)
    z = Z_80_POWER if (alpha, power) == (0.05, 0.80) else float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    return {"status": "computed", "se": se, "alpha": alpha, "power": power, "z_factor": z, "mdd": z * se}


def _correlation_cell(x: np.ndarray, y: np.ndarray, tag: str, definition: str) -> dict:
    n = len(x)
    if n < 3:
        return {"cell_status": "failure_nan", "n": n, "definition": definition,
                "reason": "fewer than 3 participants have both a computed resting-state feature and a computed decodability value in both arms"}
    corr = pearson_permutation_test(x, y, n_perm=100_000, rng=np.random.default_rng(stable_seed(f"{ANALYSIS_ID}|{tag}")))
    _, ci_lower, ci_upper = bootstrap_ci(
        np.column_stack([x, y]), _pearson_r_of_columns, n_boot=10_000,
        rng=np.random.default_rng(stable_seed(f"{ANALYSIS_ID}|{tag}|boot")),
    )
    return {
        "definition": definition, "estimate": corr["r"], "ci_lower": ci_lower, "ci_upper": ci_upper,
        "minimum_detectable_difference_at_80pct_power": _correlation_mdd(ci_lower, ci_upper),
        "p_value": corr["p_value"], "n": n, "cell_status": None,
    }


def synthesize_family2(per_session: dict, participants: list[str]) -> dict:
    theta_by, auc_by = {}, {}
    for participant in participants:
        theta_by[participant], auc_by[participant] = {}, {}
        for session in SESSIONS:
            rec = per_session[f"{participant}_ses-{session}"]
            rest = rec["rest"]
            seg = rest.get("segments", {}).get("eyes_closed") if rest.get("status") == "computed" else None
            decode = rec.get("decode", {})
            theta_by[participant][session] = seg["log_power"][PRIMARY_KEY] if seg and seg.get("status") == "computed" else None
            auc_by[participant][session] = decode.get("primary_diag_auc") if decode.get("status") == "computed" else None

    complete = [p for p in participants if all(theta_by[p][s] is not None and auc_by[p][s] is not None for s in SESSIONS)]
    delta_theta = np.array([theta_by[p]["verum"] - theta_by[p]["sham"] for p in complete])
    delta_auc = np.array([auc_by[p]["verum"] - auc_by[p]["sham"] for p in complete])
    mean_theta = np.array([np.mean([theta_by[p]["sham"], theta_by[p]["verum"]]) for p in complete])
    mean_auc = np.array([np.mean([auc_by[p]["sham"], auc_by[p]["verum"]]) for p in complete])

    cells = {
        "within_participant": _correlation_cell(
            delta_theta, delta_auc, "family2|within_participant",
            "within a participant, does the arm with the higher resting eyes-closed frontal theta also "
            "show the higher delay-period decodability -- Pearson correlation of the two paired "
            "within-participant deltas (verum minus sham); immune to any stable between-person trait "
            "shared by both measures",
        ),
        "between_participant": _correlation_cell(
            mean_theta, mean_auc, "family2|between_participant",
            "across participants, does a higher resting eyes-closed frontal theta (each participant's "
            "two arms averaged, since a participant's two sessions are not independent draws) go with "
            "higher delay-period decodability -- confounded by every stable trait covarying with both "
            "measures",
        ),
    }
    valid = [k for k in cells if cells[k].get("p_value") is not None]
    if valid:
        q = fdr_bh(np.array([cells[k]["p_value"] for k in valid]), alpha=FDR_ALPHA)
        for k, qv in zip(valid, q["q_values"]):
            cells[k]["q_value"] = float(qv)
            cells[k]["cell_status"] = "estimated"
    for k in cells:
        cells[k].setdefault("q_value", None)

    return {
        "feature_definition": (
            "predictor: eyes-closed frontal-theta log CSD power, the same primary feature as "
            "resting_state_arm_contrast (eyes-open is not used here -- see the eyes-open truncation "
            "note). outcome: operation_forward_vs_transform's same-time-bin (diagonal) decode AUC, "
            "recomputed per session by importing run_ds005034_operation_versus_load_decoding's own "
            "estimator (_load_ds005034_session, session_cells) -- that script's own delivered artifact "
            "stores only participant/arm-resampled aggregates and no per-session checkpoint for it "
            "exists on this data root, so a per-session value could not be reused without recomputing it."
        ),
        "n_participants_with_both_measures_in_both_arms": len(complete),
        "cells": cells,
    }


def _write(artifact: dict, output: Path) -> None:
    output.write_text(canonical_json(artifact))


def main() -> None:
    output_path = RESULTS / f"{ANALYSIS_ID}.json"
    root = data_root()
    missing = missing_files(root, ["ds005034"])
    if missing:
        _write({"status": "blocked_incomplete_data",
                "missing_files": [{"corpus": c, "path": p} for c, p in missing]}, output_path)
        print(f"INCOMPLETE DATA -- {len(missing)} missing file(s), stopping without fitting.")
        return

    directory = root / "ds005034"
    participants = sorted(p.name for p in directory.glob("sub-*") if p.is_dir())
    rest_paths = {
        p: {s: directory / p / f"ses-{s}" / "eeg" / f"{p}_ses-{s}_task-rest_eeg.set" for s in SESSIONS}
        for p in participants
    }
    complete_rest_pairs = [p for p in participants if all(rest_paths[p][s].is_file() for s in SESSIONS)]

    artifact = {
        "analysis_id": ANALYSIS_ID, "analysis_version": ANALYSIS_VERSION, "status": "in_progress",
        "started_unix": time.time(), "code_commit": git_commit(ROOT),
        "design": {
            "within_session_ordering": RESTING_STATE_ORDERING_NOTE,
            "eyes_open_truncation": EYES_OPEN_TRUNCATION_NOTE,
            "primary_roi_and_band": "frontal ROI (run_ds005034_tacs_aftereffect.THETA_ROI), theta band -- "
                                     "the stimulation target and frequency",
            "segment_minimum_duration_seconds": SEGMENT_MIN_SECONDS,
            "segment_minimum_duration_reason": SEGMENT_MIN_SECONDS_REASON,
            "resampling_unit": "participant",
            "multiplicity_families": {
                "resting_state_arm_contrast_secondary_eyes_closed": list(SECONDARY_KEYS),
                "resting_state_arm_contrast_secondary_eyes_open": list(SECONDARY_KEYS),
                "resting_state_predicts_task_decodability": ["within_participant", "between_participant"],
            },
            "primary_cells_outside_any_fdr_family": [
                "resting_state_arm_contrast.eyes_closed.primary.frontal_theta",
                "resting_state_arm_contrast.eyes_open.primary.frontal_theta",
            ],
            "declared_before_fitting": True,
        },
        "inventory": {"registered_participants": participants, "complete_rest_pairs": complete_rest_pairs},
        "sessions": {},
    }
    _write(artifact, output_path)

    per_session = {}
    for participant in complete_rest_pairs:
        artifact["sessions"][participant] = {}
        for session in SESSIONS:
            session_key = f"{participant}_ses-{session}"
            cached = load_session_checkpoint(session_key)
            if cached is None:
                cached = compute_session(directory, participant, session, rest_paths[participant][session])
                save_session_checkpoint(session_key, cached)
                print(f"{session_key}: computed", flush=True)
            else:
                print(f"{session_key}: checkpoint", flush=True)
            per_session[session_key] = cached
            artifact["sessions"][participant][session] = cached
            _write(artifact, output_path)

    order_info = synthesize_order(per_session, complete_rest_pairs)
    artifact["order_recovery"] = order_info
    artifact["families"] = {
        "resting_state_arm_contrast": synthesize_family1(per_session, complete_rest_pairs, order_info),
        "resting_state_predicts_task_decodability": synthesize_family2(per_session, complete_rest_pairs),
    }
    artifact["independent_group"] = resolve_group("ds005034")
    artifact["n_independent_groups"] = count_independent_groups(["ds005034"])
    artifact["status"] = "complete"
    artifact["duration_seconds"] = time.time() - artifact["started_unix"]
    _write(artifact, output_path)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats[ANALYSIS_ID] = _json_safe(artifact)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
