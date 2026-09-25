#!/usr/bin/env python3
"""ds005034 (theta-tACS scalp EEG working memory) carries a 3x2 factorial at
delay onset: the operation applied to the memorandum (simple retention --
DIN104/DIN106; backward reordering -- DIN114/DIN116; alphabetical reordering
-- DIN124/DIN126; see scripts/run_ds005034_tacs_aftereffect.py's EVENT_MAP)
crossed with list length (4 vs 6, the last digit of the same codes). Every
existing analysis of this corpus reads only the load digit. This script asks
whether the operation is readable from the
delay-period population signal, using the SAME within-window temporal
generalisation estimator, participant-level resampling, and multiplicity
handling as scripts/run_cross_window_code_generalisation.py, and reuses that
script's helper functions directly rather than re-deriving them.

Unlike the cross-window analysis (encoding vs maintenance), this corpus has
no encoding-window content label (see item_id_unavailable_reason in
src/corpus_sessions.py:iter_ds005034) so there is nothing to cross-generalise
FROM here. The estimand is instead a within-delay-window temporal
generalisation matrix over the SAME six delay-window bins iter_ds005034
already produces (SCALP_EEG_DELAY_BINS, one per second of the 0.5-6.5 s
post-trigger window): the off-diagonal quadrant (train at delay time i, test
at delay time j != i) minus chance is the estimand, and the diagonal
(same-time decode) is reported alongside as the reference that gates it,
exactly as scripts/run_cross_window_code_generalisation.py's
train_window_diag_auc does for its off-diagonal cells.

Cells (every one fit separately for sham and verum sessions -- never
averaged across the stimulation manipulation; a pooled descriptive view is
also reported, excluded from the multiplicity families):
  - three pairwise operation contrasts (forward-vs-backward,
    forward-vs-alphabetical, backward-vs-alphabetical)
  - one named additional cell collapsing the two transformation conditions
    into a single class (forward-vs-transform), run ALONGSIDE the pairwise
    cells, never in place of them
  - load (4 vs 6) decoded separately within each of the three operation
    levels, so the corpus's already-delivered pooled-load cross-window
    number can be read against its own operation sub-conditions

iter_ds005034 (src/corpus_sessions.py) already reads the operation digit into
its own 'task_condition' field -- the corpus iterator itself was not touched.
This script's own per-session loader (_load_ds005034_session) duplicates its
minimal admission/window logic rather than calling it directly, so that a
resumed run can check a session's checkpoint before paying that loader's EEG
read cost rather than after (see the checkpoint functions below).

This corpus has no trial-level behavioural outcome on this data root
(config/datasets.json ds005034.behaviour_available=false) and no per-trial
item identity, so neither a behaviour link nor a content (item-identity)
arm can be fit here; both absences are recorded in the artifact as release
properties, not silently omitted cells. The release's eyes-open/eyes-closed
resting-state blocks are recorded as present and unused.

Output: results/ds005034_operation_versus_load_decoding.json

Run:
    conda run -n wm_dynamics python scripts/run_ds005034_operation_versus_load_decoding.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from project_config import data_root  # noqa: E402

from geometry import _fit_pca_fold, _project_fold, _ctg_splits, _ctg_score_fold  # noqa: E402
from statistics import (  # noqa: E402
    stable_seed, fdr_bh, paired_sign_flip_test, minimum_detectable_paired_difference,
)
from subject_independence import resolve_group, count_independent_groups  # noqa: E402
from data_integrity import missing_files  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint, canonical_json  # noqa: E402
from corpus_sessions import SCALP_EEG_DELAY_BINS, MIN_TRIALS, _scalp_delay_bin_power  # noqa: E402
from run_ds005034_tacs_aftereffect import load_events  # noqa: E402
from run_cross_window_code_generalisation import (  # noqa: E402
    _standardize_train_test, cell_status, N_BOOT, N_PERM_SIGN_FLIP, FDR_ALPHA,
    MIN_TRIALS_PER_CLASS, MIN_TRIALS_PER_CLASS_COMPARISON, _choose_n_splits,
    subject_cluster_bootstrap_paired, STIMULATION_ARM_DEFINITION_SUFFIX, CLEARING_RULE_STATUS,
)

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_ds005034_operation_versus_load_decoding"
CHECKPOINT_SCHEMA = "ds005034_operation_versus_load_v1"
N_PC = 8
OPERATION_LEVELS = ("forward", "backward", "alphabetical")
OPERATION_PAIRS = (("forward", "backward"), ("forward", "alphabetical"), ("backward", "alphabetical"))

BEHAVIOUR_LINK_UNAVAILABLE = (
    "config/datasets.json ds005034.behaviour_available is false: this release ships no trial-level "
    "behavioural outcome (accuracy/response time) on this project's data root; the only place a "
    "trial outcome exists for this corpus is a separate OSF repository not present here. No "
    "behaviour-link cell is fit for this reason -- this is a property of the release, not a decision "
    "made in this script."
)
CONTENT_ARM_UNAVAILABLE = (
    "iter_ds005034 (src/corpus_sessions.py) already records item_id_unavailable_reason: no per-trial "
    "item identity is recorded in this corpus's public BIDS release. So there is no memorandum-content "
    "label to fit here, only operation (this file) and load (already fit by "
    "run_cross_window_code_generalisation.py)."
)
def _resting_state_blocks_note() -> str:
    base = (
        "this release ships a pre-task resting-state recording for every one of the 25 participants "
        "in both stimulation arms -- 50 separate *_task-rest_eeg.set files (one per participant per "
        "sham/verum session), each roughly 95 MB, each with its own channels.tsv and events.tsv sidecars "
        "carrying eyes-open (eyeo) and eyes-closed (eyec) event markers. Verified present on disk for "
        "every participant/arm this script admits. Not read or used by this script."
    )
    link_path = RESULTS / "ds005034_resting_state_link.json"
    if not link_path.exists():
        return base
    reason = json.loads(link_path.read_text()).get("design", {}).get("segment_minimum_duration_reason")
    if not reason:
        return base
    return f"{base} Measured block durations (results/ds005034_resting_state_link.json): {reason}"


RESTING_STATE_NOTE = _resting_state_blocks_note()


def _session_checkpoint_path(session_key):
    return CHECKPOINT_DIR / f"{session_key}.json"


def load_session_checkpoint(session_key):
    path = _session_checkpoint_path(session_key)
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
    path = _session_checkpoint_path(session_key)
    payload = {"schema": CHECKPOINT_SCHEMA, "complete": True, "record": checkpoint_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=CHECKPOINT_DIR, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _ds005034_candidates(directory):
    for subject_dir in sorted(directory.glob("sub-*")):
        for session in ("sham", "verum"):
            yield subject_dir, session, f"{subject_dir.name}_ses-{session}"


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


def within_window_ctg(psth, y, n_components, n_splits, rng):
    y = np.asarray(y)
    t_idx = np.arange(psth.shape[2])
    fold_mats = []
    for tr_idx, te_idx in _ctg_splits(y, n_splits, rng, None):
        X_tr, X_te = _standardize_train_test(psth[tr_idx], psth[te_idx], psth.shape[2], joint=True)
        mu, V = _fit_pca_fold(X_tr, n_components)
        Z_tr, Z_te = _project_fold(X_tr, mu, V), _project_fold(X_te, mu, V)
        fold_mats.append(_ctg_score_fold(Z_tr, y[tr_idx], Z_te, y[te_idx], t_idx))
    return np.nanmean(np.stack(fold_mats), axis=0)


def window_stats(auc_mat):
    n = auc_mat.shape[0]
    offdiag_mask = ~np.eye(n, dtype=bool)
    return {
        "offdiag_effect": float(np.nanmean(auc_mat[offdiag_mask]) - 0.5),
        "diag_mean_auc": float(np.nanmean(np.diag(auc_mat))),
    }


def fit_cell(tag, psth, y_raw):
    y = np.asarray(y_raw)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < MIN_TRIALS_PER_CLASS:
        return None
    n_units = psth.shape[1]
    n_comp = max(2, min(N_PC, n_units - 1))
    n_splits = _choose_n_splits(y)
    rng = np.random.default_rng(stable_seed(tag))
    auc_mat = within_window_ctg(psth, y, n_comp, n_splits, rng)
    stats = window_stats(auc_mat)
    stats["min_class_count"] = int(counts.min())
    stats["admitted_at_comparison_floor"] = bool(counts.min() >= MIN_TRIALS_PER_CLASS_COMPARISON)
    return stats


def session_cells(sess):
    tc, load = sess["task_condition"], sess["load"]
    psth = sess["counts"]
    out = {}

    for op_a, op_b in OPERATION_PAIRS:
        name = f"operation_{op_a}_vs_{op_b}"
        mask = np.isin(tc, (op_a, op_b))
        y = (tc[mask] == op_b).astype(float)
        tag = f"ds005034_{sess['session']}_{name}"
        out[name] = fit_cell(tag, psth[mask], y)

    name = "operation_forward_vs_transform"
    y = (tc != "forward").astype(float)
    out[name] = fit_cell(f"ds005034_{sess['session']}_{name}", psth, y)

    for op in OPERATION_LEVELS:
        name = f"load_within_{op}"
        mask = tc == op
        y = (load[mask] == 6).astype(float)
        out[name] = fit_cell(f"ds005034_{sess['session']}_{name}", psth[mask], y)

    return out


CELL_FAMILY = {
    "operation_forward_vs_backward": "operation_decoding",
    "operation_forward_vs_alphabetical": "operation_decoding",
    "operation_backward_vs_alphabetical": "operation_decoding",
    "operation_forward_vs_transform": "operation_decoding",
    "load_within_forward": "load_within_operation_decoding",
    "load_within_backward": "load_within_operation_decoding",
    "load_within_alphabetical": "load_within_operation_decoding",
}
CELL_DEFINITIONS = {
    "operation_forward_vs_backward": "simple retention (DIN104/DIN106) vs backward reordering-and-"
        "retention (DIN114/DIN116); both loads pooled",
    "operation_forward_vs_alphabetical": "simple retention vs alphabetical reordering-and-retention "
        "(both loads pooled)",
    "operation_backward_vs_alphabetical": "backward reordering-and-retention vs alphabetical "
        "reordering-and-retention (both loads pooled)",
    "operation_forward_vs_transform": "named additional cell: simple retention vs a single collapsed "
        "'transform' class (backward reordering + alphabetical reordering trials pooled), run "
        "alongside the three pairwise cells above, never in place of them",
    "load_within_forward": "list length 4 vs 6, simple-retention trials only",
    "load_within_backward": "list length 4 vs 6, backward-reordering trials only",
    "load_within_alphabetical": "list length 4 vs 6, alphabetical-reordering trials only",
}


def aggregate_cell(entries, dataset_tag):
    subj_off, subj_diag = {}, {}
    for subject, stats in entries:
        subj_off.setdefault(subject, []).append(stats["offdiag_effect"])
        subj_diag.setdefault(subject, []).append(stats["diag_mean_auc"])
    if not subj_off:
        return None
    subject_off = np.array([np.mean(v) for v in subj_off.values()])
    subject_diag = np.array([np.mean(v) for v in subj_diag.values()])
    n_sessions = sum(len(v) for v in subj_off.values())
    cell = {"n_subjects": len(subject_off), "n_sessions": n_sessions}
    if not np.all(np.isfinite(subject_off)) or not np.all(np.isfinite(subject_diag)):
        cell.update(cell_status="failure_nan", estimand=None, ci_lower=None, ci_upper=None,
                    train_window_diag_auc=None, train_window_diag_auc_ci_lower=None,
                    train_window_diag_auc_ci_upper=None, transfer_ratio=None,
                    transfer_ratio_ci_lower=None, transfer_ratio_ci_upper=None,
                    p_value=None, q_value=None, minimum_detectable_difference_at_80pct_power=None)
        return cell
    boot = subject_cluster_bootstrap_paired(
        subject_off, subject_diag, n_boot=N_BOOT,
        rng=np.random.default_rng(stable_seed(f"{dataset_tag}_boot")))
    sign = paired_sign_flip_test(
        subject_off, np.zeros_like(subject_off), n_perm=N_PERM_SIGN_FLIP, alternative="two-sided",
        rng=np.random.default_rng(stable_seed(f"{dataset_tag}_signflip")))
    cell.update(
        estimand=boot["offdiag_observed"], ci_lower=boot["offdiag_ci_lower"], ci_upper=boot["offdiag_ci_upper"],
        train_window_diag_auc=boot["diag_observed"],
        train_window_diag_auc_ci_lower=boot["diag_ci_lower"], train_window_diag_auc_ci_upper=boot["diag_ci_upper"],
        transfer_ratio=boot["transfer_ratio"], transfer_ratio_ci_lower=boot["transfer_ratio_ci_lower"],
        transfer_ratio_ci_upper=boot["transfer_ratio_ci_upper"],
        p_value=sign["p_value"], q_value=None, cell_status=None,
        minimum_detectable_difference_at_80pct_power=minimum_detectable_paired_difference(subject_off),
    )
    return cell


def subject_count_discrepancy_check(root):
    from run_ds005034_tacs_aftereffect import paired_inventory
    inv = paired_inventory(root / "ds005034")
    return {
        "config_datasets_json_prose": "25 registered participants, 18 with an actual memory-task EEG "
            "recording on disk (15 with both sham and verum sessions, 2 sham-only, 1 verum-only), 7 "
            "with only sidecar metadata and no recording file",
        "disk_derived_count": {
            "registered_participants": len(inv["registered_participants"]),
            "complete_pairs_sham_and_verum": len(inv["complete_pairs"]),
            "unpaired_single_session": len(inv["unpaired_raw"]),
            "no_recording_file": len(inv["no_raw_memory_recording"]),
        },
        "resolution": "the config/datasets.json prose is stale, not the delivered cross-window "
            "artifact's 50 sessions / 25 subjects: every one of the 25 registered participant "
            "directories on this data root has both a sham and a verum "
            "*_task-memory_eeg.set file, verified by direct existence and non-zero-size check "
            "(paired_inventory in scripts/run_ds005034_tacs_aftereffect.py) against this run's own "
            "WM_DYNAMICS_DATA_ROOT, giving 25 complete pairs = 50 sessions, zero unpaired, zero "
            "missing. This matches the 50 sessions / 25 subjects already recorded for ds005034 in "
            "results/cross_window_code_generalisation.json. The most likely explanation is that the "
            "config/datasets.json prose was written while a download of this corpus was still in "
            "progress (a detached download-retry process for this project's data root was observed "
            "running at the time of this check) and was never refreshed once the download completed.",
    }


def _write_artifact(artifact):
    with open(RESULTS / "ds005034_operation_versus_load_decoding.json", "w") as f:
        json.dump(_json_safe(artifact), f, indent=2)


def main():
    root = data_root()
    missing = missing_files(root, ["ds005034"])
    if missing:
        _write_artifact({"status": "blocked_incomplete_data",
                         "missing_files": [{"corpus": c, "path": p} for c, p in missing]})
        print(f"INCOMPLETE DATA -- {len(missing)} missing file(s), stopping without fitting.")
        return

    predeclared_rules = {
        "estimand": "the mean of (AUC - 0.5) over the off-diagonal of a within-delay-window temporal "
            "generalisation matrix (train a decoder at one of the six delay-window time bins, test it "
            "at a different delay-window time bin), same estimator convention as "
            "scripts/run_cross_window_code_generalisation.py's cross-window off-diagonal quadrant, "
            "applied within a single window here because this corpus has no encoding-window content "
            "label to cross-generalise from. The diagonal (same-bin decode) is reported alongside as "
            "the reference value, never used to gate the off-diagonal estimand.",
        "resampling_unit": "subject/patient; each subject contributes at most one session per "
            "stimulation arm here, and sessions belonging to one subject are averaged before any "
            "resampling -- the trial is never the resampling unit",
        "three_class_operation_handling": "pairwise binary AUC cells (forward-vs-backward, "
            "forward-vs-alphabetical, backward-vs-alphabetical) are the primary operation-decoding "
            "cells; a named additional cell (operation_forward_vs_transform) collapses backward and "
            "alphabetical into one 'transform' class and is reported alongside the pairwise cells, "
            "never as a replacement for them.",
        "multiplicity_families": {
            "operation_decoding": list(k for k, v in CELL_FAMILY.items() if v == "operation_decoding"),
            "load_within_operation_decoding": list(
                k for k, v in CELL_FAMILY.items() if v == "load_within_operation_decoding"),
        },
        "stimulation_arms": "every cell is fit separately for sham and verum sessions; a pooled view "
            "across both is reported as a descriptive addendum only, excluded from both multiplicity "
            "families (cell_status='descriptive_not_in_fdr_family'), following "
            "scripts/run_cross_window_code_generalisation.py's own convention for this same corpus's "
            "load cell.",
        "minimum_detectable_difference_definition": "src/statistics.py:minimum_detectable_paired_"
            "difference at 80% power, computed on the per-subject off-diagonal estimand values",
        "clearing_rule_status": CLEARING_RULE_STATUS,
        "declared_before_fitting": True,
    }

    artifact = {
        "predeclared_rules": predeclared_rules,
        "behaviour_link_unavailable_reason": BEHAVIOUR_LINK_UNAVAILABLE,
        "content_arm_unavailable_reason": CONTENT_ARM_UNAVAILABLE,
        "resting_state_blocks_note": RESTING_STATE_NOTE,
        "subject_count_discrepancy_check": subject_count_discrepancy_check(root),
        "n_delay_window_bins": SCALP_EEG_DELAY_BINS,
        "cell_definitions": CELL_DEFINITIONS,
        "arms": {}, "status": "in_progress",
    }
    _write_artifact(artifact)

    directory = root / "ds005034"
    per_session = {}
    for subject_dir, session, session_key in _ds005034_candidates(directory):
        cached = load_session_checkpoint(session_key)
        if cached is not None:
            if not cached.get("excluded"):
                per_session[session_key] = cached
                print(f"  {session_key}: {cached['n_trials']} trials (checkpoint)", flush=True)
            continue
        sess = _load_ds005034_session(directory, subject_dir, session)
        if sess is None:
            save_session_checkpoint(session_key, {"excluded": True})
            continue
        record = {"subject": sess["patient"], "stimulation_session": sess["stimulation_session"],
                  "n_trials": len(sess["task_condition"]), "cells": session_cells(sess)}
        save_session_checkpoint(session_key, record)
        per_session[session_key] = record
        artifact["n_sessions"] = len(per_session)
        artifact["n_subjects"] = len({v["subject"] for v in per_session.values()})
        _write_artifact(artifact)
        print(f"  {session_key}: {record['n_trials']} trials", flush=True)
    artifact["n_sessions"] = len(per_session)
    artifact["n_subjects"] = len({v["subject"] for v in per_session.values()})
    artifact["independent_group"] = resolve_group("ds005034")
    artifact["n_independent_groups"] = count_independent_groups(["ds005034"])
    _write_artifact(artifact)

    fdr_cells = {"operation_decoding": [], "load_within_operation_decoding": []}
    for tag in ("sham", "verum", "pooled"):
        subset = {k: v for k, v in per_session.items()
                  if tag == "pooled" or v["stimulation_session"] == tag}
        arm = {"n_sessions": len(subset), "n_subjects": len({v["subject"] for v in subset.values()}),
              "definition_suffix": STIMULATION_ARM_DEFINITION_SUFFIX[tag], "cells": {}}
        for cell_name in CELL_FAMILY:
            entries = [(v["subject"], v["cells"][cell_name]) for v in subset.values()
                      if v["cells"].get(cell_name) is not None]
            n_sessions_below_floor = sum(1 for v in subset.values() if v["cells"].get(cell_name) is None)
            cell = aggregate_cell(entries, f"ds005034_{tag}_{cell_name}")
            if cell is None:
                cell = {"n_subjects": 0, "n_sessions": 0, "cell_status": "failure_nan",
                        "estimand": None, "reason": "no session in this stimulation arm had "
                        f"{MIN_TRIALS_PER_CLASS} trials in each class of this cell"}
            cell["n_sessions_below_trial_floor"] = n_sessions_below_floor
            cell["multiplicity_family"] = CELL_FAMILY[cell_name]
            if tag == "pooled":
                if cell.get("cell_status") != "failure_nan":
                    cell["cell_status"] = "descriptive_not_in_fdr_family"
            elif cell.get("estimand") is not None:
                fdr_cells[CELL_FAMILY[cell_name]].append(cell)
            arm["cells"][cell_name] = cell
        artifact["arms"][tag] = arm
    _write_artifact(artifact)

    for family, cells in fdr_cells.items():
        if not cells:
            continue
        q = fdr_bh(np.array([c["p_value"] for c in cells]), alpha=FDR_ALPHA)
        for cell, qv in zip(cells, q["q_values"]):
            cell["q_value"] = float(qv)
            cell["cell_status"] = cell_status(is_nan=False)

    artifact["status"] = "complete"
    _write_artifact(artifact)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["ds005034_operation_versus_load_decoding"] = _json_safe(artifact)
    print("Saved results/ds005034_operation_versus_load_decoding.json")


if __name__ == "__main__":
    main()
