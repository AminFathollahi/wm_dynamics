#!/usr/bin/env python3
"""Test the rate-free state observable before recognition decisions in DANDI 000004."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / directory))

from project_config import dataset_path
from provenance import canonical_json, git_commit
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation
from run_state_content_link import delay_counts
from run_within_session_permutation_control import _cheap_partial_r
from spike_pipeline import filter_units_by_region, load_spike_times, low_rate_unit_mask, resolve_unit_regions
from state_persistence import slope_across_sessions_test
from statistics import fdr_bh, minimum_detectable_paired_difference

OUTPUT_PATH = ROOT / "results" / "dandi_000004_recognition_generalization.json"
REGIONS = ("hippocampus", "amygdala")
WINDOW_S = 0.45
MIN_TRIALS = 40
MIN_UNITS = 8
REFERENCE_R = 0.14


def recognition_correct(labels: np.ndarray, responses: np.ndarray) -> np.ndarray:
    true_old = np.asarray(labels).astype(str) == "0"
    response_old = np.asarray(responses, dtype=float) >= 34
    return true_old == response_old


def _session(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        trials = handle["intervals/trials"]
        phase = np.array([value.decode() if isinstance(value, bytes) else str(value)
                          for value in trials["stim_phase"][:]])
        labels = np.array([value.decode() if isinstance(value, bytes) else str(value)
                           for value in trials["new_old_labels_recog"][:]])
        responses = trials["response_value"][:].astype(float)
        stim_off = trials["stim_off_time"][:].astype(float)
        question_onset = trials["delay1_time"][:].astype(float)
        timing = question_onset - stim_off
        candidate = ((phase == "recog") & np.isin(labels, ("0", "1"))
                     & np.isin(responses, np.arange(31, 37))
                     & np.isfinite(stim_off) & np.isfinite(question_onset))
        admitted = candidate & (timing >= WINDOW_S) & (timing <= 0.75)
        participant = handle["general/subject/subject_id"][()]
        participant = participant.decode() if isinstance(participant, bytes) else str(participant)
        base = {
            "session": path.stem, "participant": participant,
            "n_trials_total": int(len(phase)), "n_recognition_trials": int((phase == "recog").sum()),
            "n_recognition_trials_with_valid_label_response_and_timing": int(candidate.sum()),
            "n_trials_admitted": int(admitted.sum()),
            "n_candidate_trials_refused_by_window_gate": int((candidate & ~admitted).sum()),
        }
        correct = recognition_correct(labels[admitted], responses[admitted]).astype(float)
        base["n_errors_admitted"] = int((correct == 0).sum())
        if admitted.sum() < MIN_TRIALS:
            return {**base, "status": "refused", "reason": "fewer_than_40_valid_recognition_trials"}
        if np.unique(correct).size < 2:
            return {**base, "status": "refused", "reason": "no_accuracy_variation"}
        spike_lists = load_spike_times(handle)
        regions = resolve_unit_regions(handle, "nwb_hemisphere_prefixed_structure")["region"]
        region_rows = {}
        for region in REGIONS:
            selected = filter_units_by_region(spike_lists, regions, region)
            keep = low_rate_unit_mask(selected, stim_off[admitted], WINDOW_S)
            selected = [spikes for spikes, retain in zip(selected, keep) if retain]
            if len(selected) < MIN_UNITS:
                region_rows[region] = {
                    "status": "refused", "reason": "fewer_than_8_units_after_rate_qc",
                    "n_units_after_rate_qc": len(selected),
                }
                continue
            activity = delay_counts(selected, stim_off[admitted], WINDOW_S, bin_ms=50.0).sum(axis=2)
            deviation = rate_free_state_deviation(activity)
            finite = np.isfinite(deviation)
            if finite.sum() < MIN_TRIALS or np.unique(correct[finite]).size < 2:
                region_rows[region] = {"status": "refused", "reason": "too_few_defined_trials_or_no_accuracy_variation"}
                continue
            outcome = correct[finite]
            predictor = deviation[finite]
            spike_count = activity[finite].sum(axis=1)
            trial_index = np.flatnonzero(admitted)[finite].astype(float)
            true_old = (labels[admitted][finite] == "0").astype(float)
            region_rows[region] = {
                "status": "computed", "n_units_after_rate_qc": len(selected),
                "n_trials": int(finite.sum()), "n_errors": int((outcome == 0).sum()),
                "raw_r": _cheap_partial_r(outcome, predictor, []),
                "joint_partial_r": _cheap_partial_r(outcome, predictor, [spike_count, trial_index, true_old]),
            }
    return {**base, "status": "computed", "regions": region_rows}


def _participant_rows(sessions: list[dict], region: str) -> dict[str, dict]:
    grouped = defaultdict(list)
    for session in sessions:
        cell = session.get("regions", {}).get(region, {})
        if cell.get("status") == "computed":
            grouped[session["participant"]].append(cell)
    result = {}
    for participant, cells in grouped.items():
        weights = np.array([cell["n_trials"] for cell in cells], dtype=float)
        result[participant] = {
            "n_sessions": len(cells), "n_trials": int(weights.sum()),
            "raw_r": float(np.average([cell["raw_r"] for cell in cells], weights=weights)),
            "joint_partial_r": float(np.average([cell["joint_partial_r"] for cell in cells], weights=weights)),
        }
    return result


def _pool(participants: dict[str, dict], key: str) -> dict:
    values = [row[key] for row in participants.values() if row[key] is not None and np.isfinite(row[key])]
    result = slope_across_sessions_test(values, alternative="two-sided")
    result["minimum_detectable_r_at_80pct_power"] = minimum_detectable_paired_difference(values)
    return result


def build(data_dir: Path) -> dict:
    paths = sorted(data_dir.rglob("*.nwb"))
    sessions = [_session(path) for path in paths]
    regions = {}
    joint_p = []
    joint_keys = []
    for region in REGIONS:
        participants = _participant_rows(sessions, region)
        raw = _pool(participants, "raw_r")
        joint = _pool(participants, "joint_partial_r")
        regions[region] = {"participants": participants, "pooled_raw": raw, "pooled_joint_partial": joint}
        if joint.get("status") == "tested":
            joint_keys.append(region)
            joint_p.append(joint["p_value"])
    correction = fdr_bh(np.array(joint_p), alpha=0.05) if joint_p else None
    for index, region in enumerate(joint_keys):
        cell = regions[region]
        cell["joint_partial_fdr_q"] = float(correction["q_values"][index])
        mdd = cell["pooled_joint_partial"]["minimum_detectable_r_at_80pct_power"].get("mdd")
        if correction["reject"][index]:
            cell["branch"] = "recognition_accuracy_association_detected"
        elif mdd is not None and mdd < REFERENCE_R:
            cell["branch"] = "no_recognition_accuracy_association_above_the_reference_bound"
        else:
            cell["branch"] = "underpowered_to_exclude_the_reference_effect"
    region_accounting = {}
    for region in REGIONS:
        computed = sum(
            session.get("regions", {}).get(region, {}).get("status") == "computed" for session in sessions
        )
        reasons = Counter(
            session.get("regions", {}).get(region, {}).get("reason", "session_level_refusal")
            for session in sessions
            if session.get("regions", {}).get(region, {}).get("status") != "computed"
        )
        region_accounting[region] = {
            "n_sessions_computed": computed, "n_sessions_refused": len(sessions) - computed,
            "refusal_reasons": dict(reasons), "reconciles": len(sessions) == computed + sum(reasons.values()),
        }
    shared = sorted(set(regions["hippocampus"]["participants"]) & set(regions["amygdala"]["participants"]))
    paired = [regions["hippocampus"]["participants"][p]["joint_partial_r"]
              - regions["amygdala"]["participants"][p]["joint_partial_r"] for p in shared]
    return {
        "analysis_id": "dandi_000004_recognition_generalization",
        "schema_version": "1.0.0", "status": "complete", "code_commit": git_commit(ROOT),
        "design": {
            "task": "new_old_recognition", "window": "first 450 ms after stimulus offset, ending before question onset",
            "regions": list(REGIONS), "participant_is_inference_unit": True,
            "response_mapping": "31-33=new; 34-36=old", "ground_truth_mapping": "0=old; 1=new",
            "primary_statistic": "participant-level mean of session joint partial correlations controlling total spike count, trial index, and old/new ground truth",
            "multiple_comparison_family": "two region-specific primary joint-partial tests, BH FDR 0.05",
            "reference_r": REFERENCE_R,
        },
        "zero_drop_accounting": {
            "n_files_seen": len(paths), "n_sessions_computed": sum(row["status"] == "computed" for row in sessions),
            "n_sessions_refused": sum(row["status"] != "computed" for row in sessions),
            "session_refusal_reasons": dict(Counter(row.get("reason") for row in sessions if row["status"] != "computed")),
            "region_level": region_accounting,
            "reconciles": len(paths) == len(sessions),
        },
        "regions": regions,
        "paired_region_difference_hippocampus_minus_amygdala": {
            "n_participants": len(shared), "participants": shared,
            "test": slope_across_sessions_test(paired, alternative="two-sided"),
        },
        "sessions": sessions,
    }


def main() -> None:
    result = build(dataset_path("dandi_000004"))
    OUTPUT_PATH.write_text(canonical_json(result))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
