#!/usr/bin/env python3
"""Cross-window code generalisation: does a decoder trained at an encoding
timepoint generalise to a maintenance timepoint, and vice versa?

Every existing cross-temporal generalisation matrix in this project fits a
single model to a single contiguous window (see run_000469_pipeline.py,
where `psth` (maintenance) and `psth_enc` (encoding) feed separate calls to
load_vs_load_ctg). This script instead concatenates the encoding and
maintenance timepoints of one session into a single joint axis and fits one
model across the whole axis, so a decoder trained at an encoding timepoint
is scored at a maintenance timepoint (and vice versa). The two windows are
not adjacent in real time; the real gap between them is recorded per corpus,
never treated as a continuous clock.

Two label arms are fitted per corpus wherever both exist:
  - load: memory set size (or an analogous task-structure variable). Defined
    only once later items have arrived, so it is a legitimate NEGATIVE
    control for the encoding window on single/first-item trials, not a test
    of content coding.
  - content: which stimulus is held (item category, cued frequency,
    instructed direction, digit). This is what a rotation claim is actually
    about, since it is present at both encoding (as a stimulus) and
    maintenance (as a memory).

Outputs:
  results/cross_window_code_generalisation.json (written incrementally, one
    corpus at a time, so a dropped run costs one corpus, not the whole file)
  results/{corpus}_ctg_cross_window[_content]_{session}.npz

Run:
    conda run -n wm_dynamics python scripts/run_cross_window_code_generalisation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root

import h5py  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from spike_pipeline import (  # noqa: E402
    load_spike_times, build_psth, low_rate_unit_mask, MIN_SESSION_ACCURACY,
    resolve_unit_regions, MIN_UNITS_PER_REGION,
)
from geometry import (  # noqa: E402
    _fit_pca_fold, _project_fold, _ctg_splits, _ctg_score_fold, _ctg_score_fold_multiclass,
)
from statistics import Z_80_POWER, stable_seed, fdr_bh, paired_sign_flip_test  # noqa: E402
from subject_independence import resolve_group, count_independent_groups  # noqa: E402
from data_integrity import missing_files, CANONICAL_RECORDING_REGISTRY_PATH  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_human_drift_spine_001187_000673 import canonical_sessions
from corpus_sessions import _trial_group
from corpus_sessions import (  # noqa: E402
    EPOCH_WINDOWS_S, BORAN_EPOCH_WINDOWS_S, region_filtered_units, alm_data_directory,
    _alm_trial_condition, _alm_build_counts, ALM_MIN_UNITS, ALM_MIN_UNIT_RATE_HZ,
    ALM_MIN_TRIALS_PER_ARM, PFC4_MIN_UNITS, PFC4_MIN_UNIT_RATE_HZ, SCALP_EEG_DELAY_BINS,
)
from info_decoding import MIN_TRIALS_PER_CLASS, MIN_TRIALS_PER_CLASS_COMPARISON  # noqa: E402
from statistics import CLEARING_RULE_STATUS, FDR_ALPHA, N_BOOT, N_PERM_SIGN_FLIP  # noqa: E402
from statistics import STIMULATION_ARM_DEFINITION_SUFFIX, _choose_n_splits, _standardize_train_test, cell_status, subject_cluster_bootstrap_paired  # noqa: E402

RESULTS = ROOT / "results"
MIN_DIAG_AUC = 0.55
BIN_MS = 100.0
SMOOTH_MS = 200.0
N_PC = 8
# 5, not an arbitrary stricter value: this is load_vs_load_ctg's own established
# per-class trial floor for this exact 4-vs-8 / 1-vs-3 contrast elsewhere in this
# project (run_000469_pipeline.py, run_000574_units_pipeline.py, ...). A second,
# stricter count (15) is also reported per corpus for comparison, never used to
# gate admission.

TIERS = {
    "dandi_000469": "human_primary", "dandi_001187": "human_primary",
    "dandi_000673": "human_primary", "dandi_000574": "human_primary",
    "dandi_000004": "human_primary",
    "pfc3": "nonhuman_primate_supplementary", "pfc4": "nonhuman_primate_supplementary",
    "panichello_2024": "nonhuman_primate_supplementary",
    "watters_2026": "nonhuman_primate_supplementary",
    "macaque_pfc_microstimulation": "nonhuman_primate_supplementary",
    "inagaki_alm5": "mouse_supplementary_to_supplementary",
    "haslacher_clam_tacs": "human_primary", "alagapan_phase_stimulation": "human_primary",
    "ram_ds005489_openloop": "human_primary", "ram_ds005557_closedloop": "human_primary",
    "ds004752": "human_primary", "wolff_eeg_impulse": "human_primary",
    "kai_miller_nback": "human_primary", "ds005034": "human_primary", "ds006848": "human_primary",
}

ALL_REGISTERED_CORPORA = sorted(TIERS.keys())
FITTED_CORPORA = ("dandi_000469", "dandi_001187", "dandi_000673", "dandi_000574", "pfc4",
                   "dandi_000004", "inagaki_alm5", "ds005034", "ds006848")

# Every corpus below was checked against its raw file(s) for a second, stimulus-period
# onset. (a) = the onset exists and is a short read away (field named, or the same
# continuous recording at a different offset); this project simply never read it.
# (b) = no such onset exists in the file at all. Neither classification is a
# scientific property of the recording when it is (a) -- it is a statement about
# what this project's code currently reads.
EXCLUSIONS = {
    "pfc3": {
        "property_lacking": "non-simultaneous pseudo-population recording",
        "loader_gap_classification": "b",
        "check": "run_pfc3_content_ctg.py's own docstring states neurons were not recorded "
                 "simultaneously and builds pseudo-trials by independently resampling one real trial "
                 "per neuron per class; loadmat on a raw file confirms each of the 8041 released "
                 "files is one neuron/session with its own MatData, no shared trial index across "
                 "units. This is a genuine recording property (no simultaneous population vector "
                 "exists to read a second window from), not a loader gap.",
    },
    "watters_2026": {
        "property_lacking": "unused capability, not a data gap, plus a continuous (not binary) content label",
        "loader_gap_classification": "a",
        "check": "watters_behaviour (src/corpus_sessions.py) already reads time_stimulus_onset into "
                 "the per-trial behaviour dataframe, and load_watters_session's per-trial cache "
                 "origin (WATTERS_CACHE_ORIGIN_BEFORE_STIMULUS_S=0.2s before stimulus onset) already "
                 "spans through the encoding-to-delay interval -- an encoding window is a different "
                 "index into the SAME cached array already opened for the delay window, not a new "
                 "field or file. Not added in this pass because this corpus's content label (cued "
                 "object identity) is a continuous polar angle (cued_theta), the same circular-label "
                 "complication documented for wolff_eeg_impulse below, and it was not part of the "
                 "corrective scope handed to this pass.",
    },
    "macaque_pfc_microstimulation": {
        "property_lacking": "corpus is organised around delay-epoch stimulation delivery, not content decoding",
        "loader_gap_classification": "b",
        "check": "src/data_integrity.py:_check_macaque_pfc_microstimulation and every "
                 "run_macaque_pfc_microstimulation_*.py script key this corpus by correct/error/"
                 "shorted-channel session files for causal stimulation-response estimation; no "
                 "per-trial content field is read by any existing script, and stimulated maintenance "
                 "trials would confound any content decode regardless of window availability.",
    },
    "haslacher_clam_tacs": {
        "property_lacking": "no per-trial content/memorandum label in the released fields this project reads",
        "loader_gap_classification": "b",
        "check": "run_haslacher_phase_omega.py's docstring lists only phase-condition trigger codes "
                 "1-6 and correct/incorrect codes 10/11 as per-trial fields; no item-identity or "
                 "set-size field is read by any existing script for that corpus.",
    },
    "alagapan_phase_stimulation": {
        "property_lacking": "no per-trial content/memorandum label wired into any existing script",
        "loader_gap_classification": "b",
        "check": "run_alagapan_phase_omega.py / run_alagapan_stimulation_geometry.py / "
                 "run_alagapan_retention_diffusion.py all operate on stimulation condition, "
                 "diffusion geometry, or phase-lag benefit -- none reads a per-trial item-identity "
                 "or set-size field, and n=3 patients is this corpus's own stated dominant limitation.",
    },
    "ram_ds005489_openloop": {
        "property_lacking": "no per-trial short maintenance-delay window",
        "loader_gap_classification": "b",
        "check": "run_ram_openloop_pipeline.py's own docstring states its SCOPE BOUNDARY explicitly: "
                 "stimulation is delivered at ENCODING in a delayed free-recall list-learning design, "
                 "not during a WM maintenance/delay period; the task has no discrete per-item "
                 "encode-then-maintain trial structure -- there is no maintenance window to pair an "
                 "encoding window against, independent of what onsets exist.",
    },
    "ram_ds005557_closedloop": {
        "property_lacking": "no per-trial short maintenance-delay window",
        "loader_gap_classification": "b",
        "check": "run_ram_closedloop_pipeline.py shares ds005489's task design and scope boundary "
                 "(delayed free recall, stimulation at encoding-period word presentation only); same "
                 "absence of a discrete per-item maintenance window as ram_ds005489_openloop.",
    },
    "ds004752": {
        "property_lacking": "no intra-trial onset field, and no existing signal reader for this corpus",
        "loader_gap_classification": "b",
        "check": "every session's events.tsv (e.g. sub-02/ses-04/ieeg/..._events.tsv) carries exactly "
                 "ONE row per 8-second trial block (columns: onset, duration, nTrial, begSample, "
                 "endSample, SetSize, ProbeLetter, Match, Correct, ResponseTime, Artifact) -- no "
                 "second onset column marks an encoding/delay boundary inside that block. This "
                 "corpus resolves to the same subject-independence group as dandi_000574 (boran), "
                 "whose own encoding/delay split is likewise not a named timestamp but a fixed offset "
                 "assumed from task design; applying that same fixed-offset assumption here is "
                 "unverified for this release, and no project code reads this corpus's continuous "
                 "EDF/iEEG signal at all yet (only its file-completeness check exists) -- this is "
                 "more than a second onset read away.",
    },
    "wolff_eeg_impulse": {
        "property_lacking": "continuous circular memorandum, not a binary label",
        "loader_gap_classification": "a",
        "check": "run_wolff_impulse_pipeline.py / run_wolff_corrected_analysis.py docstrings confirm "
                 "the memorandum is a continuous oriented-grating angle, decoded throughout this "
                 "project's existing code for that corpus with a circular Mahalanobis/crossnobis estimator, "
                 "never binarised or AUC-scored; this analysis's estimator is AUC-based (binary or "
                 "categorical one-vs-rest), and ad hoc binarisation of a circular variable is not a "
                 "transform any existing script performs. The encoding window itself is present in "
                 "the released epochs (three EEG epochs per run_wolff_impulse_pipeline.py's own "
                 "docstring) -- this is a label-type gap, not a missing-window gap.",
    },
    "kai_miller_nback": {
        "property_lacking": "continuous N-back stream, no discrete per-trial encoding/maintenance windows",
        "loader_gap_classification": "b",
        "check": "the N-back task presented in this corpus is a continuous letter stream with "
                 "responses scored relative to N items back, not a discrete encode-then-blank-"
                 "maintain-then-probe trial; there is no delay-window onset to pair an encoding "
                 "window against. run_behavior_ctg.py additionally already found no correct/error "
                 "field in the raw MAT files for this corpus.",
    },
    "panichello_2024": {
        "property_lacking": "unused capability for the window split, but the encoding/delay boundary is not itself stored",
        "loader_gap_classification": "a",
        "check": "loadmat on a raw session file (e.g. one of the deposited session .mat files) shows spks shaped "
                 "(496 trials, 1950 timepoints, 135 neurons) against a shared per-trial time axis "
                 "`tc` running -499.5 to +1449.5 ms in 1 ms steps -- the full window this project's "
                 "existing pipeline already treats as a single delay period is present in the file "
                 "with no further onset needed. But splitting it into a true encoding sub-window "
                 "(cue on screen) vs. delay sub-window requires knowing the cue's on-screen duration "
                 "from the source task design, which is not itself a field in this file -- that "
                 "duration has not been looked up/verified in this pass, so no split was made.",
    },
}


def _joint_fold_loop(psth_a, psth_b, y, n_components, n_splits, rng, joint_scaler, groups, score_fold, score_extra):
    n_a = psth_a.shape[2]
    X = np.concatenate([psth_a, psth_b], axis=2)
    y = np.asarray(y)
    t_idx = np.arange(X.shape[2])
    fold_mats = []
    for tr_idx, te_idx in _ctg_splits(y, n_splits, rng, groups):
        X_tr, X_te = _standardize_train_test(X[tr_idx], X[te_idx], n_a, joint_scaler)
        mu, V = _fit_pca_fold(X_tr, n_components)
        Z_tr = _project_fold(X_tr, mu, V)
        Z_te = _project_fold(X_te, mu, V)
        mat = score_fold(Z_tr, y[tr_idx], Z_te, y[te_idx], t_idx, *score_extra)
        fold_mats.append(mat)
    auc_mat = np.nanmean(np.stack(fold_mats), axis=0)
    return auc_mat, n_a


def cross_window_ctg(psth_a, psth_b, y, n_components=N_PC, n_splits=5,
                      rng=None, joint_scaler=False, groups=None):
    """Joint cross-window CTG for a BINARY label: fit one model on
    encoding+maintenance timepoints concatenated. Returns (auc_mat, n_a)."""
    if rng is None:
        rng = np.random.default_rng(0)
    return _joint_fold_loop(psth_a, psth_b, y, n_components, n_splits, rng, joint_scaler, groups,
                             _ctg_score_fold, ())


def cross_window_ctg_multiclass(psth_a, psth_b, y, n_components=N_PC, n_splits=5,
                                 rng=None, joint_scaler=False, groups=None):
    """Joint cross-window CTG for a multiclass content label (macro one-vs-rest
    AUC, chance=0.5 regardless of class count -- same convention as
    geometry.ctg_content_permutation_null). Returns (auc_mat, n_a)."""
    if rng is None:
        rng = np.random.default_rng(0)
    all_classes = np.unique(np.asarray(y))
    return _joint_fold_loop(psth_a, psth_b, y, n_components, n_splits, rng, joint_scaler, groups,
                             _ctg_score_fold_multiclass, (all_classes,))


def quadrant_stats(auc_mat, n_a):
    a_to_a = auc_mat[:n_a, :n_a]
    b_to_b = auc_mat[n_a:, n_a:]
    a_to_b = auc_mat[:n_a, n_a:]   # train on window A, test on window B
    b_to_a = auc_mat[n_a:, :n_a]   # train on window B, test on window A
    return {
        "train_a_test_b_offdiag_effect": float(np.nanmean(a_to_b) - 0.5),
        "train_b_test_a_offdiag_effect": float(np.nanmean(b_to_a) - 0.5),
        "diag_a_mean_auc": float(np.nanmean(np.diag(a_to_a))),
        "diag_b_mean_auc": float(np.nanmean(np.diag(b_to_b))),
        "within_a_mean_auc": float(np.nanmean(a_to_a)),
        "within_b_mean_auc": float(np.nanmean(b_to_b)),
    }


def subject_cluster_bootstrap(values, n_boot=N_BOOT, rng=None):
    """Percentile bootstrap over subject-level values (one per subject,
    sessions already averaged within subject). Returns observed mean, 95% CI,
    and the minimum detectable difference at 80% power (statistics.Z_80_POWER
    times the standard deviation of the bootstrap draws)."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    n = len(values)
    draws = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return {
        "observed": float(values.mean()),
        "ci_lower": float(np.percentile(draws, 2.5)),
        "ci_upper": float(np.percentile(draws, 97.5)),
        "mdd": float(Z_80_POWER * draws.std()),
    }


# ── Corpus loaders. Each yields one dict per admitted session:
#    session_key, subject, gap,
#    label_data {"load": {...} | absent, "content": {...} | absent}, where each
#    present entry is {"a": encoding psth, "b": maintenance psth,
#    "baseline": baseline psth or None, "y": label array, "class_counts": dict
#    or None (multiclass content labels only)},
#    min_class_count {"load":int|None, "content":int|None},
#    admitted_at_15 {"load":bool|None, "content":bool|None} ──

LOAD_LABEL_DEFINITIONS = {
    "dandi_000469": "Sternberg memory set size: load 1 (low) vs load 3 (high). Not present until "
                    "later items arrive; used here as the negative-control label for encoding-window "
                    "single/first-item trials.",
    "dandi_001187": "Sternberg memory set size: load 1 (low) vs load 3 (high). Same construction and "
                    "same caveat as dandi_000469.",
    "dandi_000673": "Sternberg memory set size: load 1 (low) vs load 3 (high). Same construction and "
                    "same caveat as dandi_000469.",
    "dandi_000574": "Sternberg-style set size: 4 items (low) vs 8 items (high). Same construction and "
                    "same caveat as dandi_000469.",
    "ds005034": "list length: 4 items (low) vs 6 items (high), from the corpus's own DIN-coded "
                "forward/backward/alphabetical memory task.",
}
CONTENT_LABEL_DEFINITIONS = {
    "dandi_000469": "stimulus content category, field loadsEnc1_PicIDs (already categorical, values "
                     "1-5 -- no division needed; verified directly against the NWB trial table).",
    "dandi_001187": "stimulus content category, field PicIDs_Encoding1 // 100 (5 categories; verified "
                     "directly against the NWB trial table, same construction settled for this corpus "
                     "previously).",
    "dandi_000673": "stimulus content category, field PicIDs_Encoding1 // 100 (5 categories); same "
                     "patient lineage and task family as dandi_001187, same construction.",
    "pfc4": "vibrotactile F1 comparison frequency, median-split low vs high within session. The "
            "delay-period memorandum is a memory of the same F1 value physically present during the "
            "encoding window.",
    "dandi_000004": "stimCategory (5 categories), the stimulus shown during this trial's own stim_on "
                    "to stim_off encoding period and held through delay1-to-delay2 maintenance before "
                    "the recognition response.",
    "ds006848": "first digit of the presented sequence, physically on screen during the first "
                "Encoding_DigitValue_*/Encoding_Set_Simultaneous event of the trial and held through "
                "the retention window.",
}
# Kept out of CONTENT_LABEL_DEFINITIONS on purpose: this corpus's decoded variable is an instructed
# movement direction (which side to lick), not a memorandum of stimulus content -- the evidence-tier
# rule already makes it mouse-tier and never a headline or a replication of a human content claim, and
# its own key/wording must never be presented as the same kind of quantity as the human content cells.
MOVEMENT_LABEL_DEFINITIONS = {
    "inagaki_alm5": "instructed movement direction (which side to lick, left vs right), physically "
                    "cued during the sample epoch (Sample_start) and held through the delay epoch "
                    "(Delay_start). This is a motor plan, not a memorandum of stimulus content.",
}
MOVEMENT_TIER_CAVEAT = {
    "inagaki_alm5": "this corpus's decoded variable is an instructed movement direction, not a "
                    "memorandum of stimulus content; it is kept fitted under this project's fixed "
                    "evidence-tier rule (a result's tier is fixed by the preparation, not by its "
                    "effect size) but is never evidence about what is held in working memory, never a "
                    "headline, and never a replication of a human content-coding claim.",
}
NO_BASELINE_REASON = {
    "pfc4": "the delayed vibrotactile-discrimination task as released has no pre-trial "
            "fixation/baseline epoch wired into any existing project loader (load_pfc4_raw_session "
            "only extracts F1-onset- and delay-onset-aligned windows).",
    "dandi_000004": "no baseline/fixation onset is extracted anywhere in this project's existing "
                    "dandi_000004 handling (iter_dandi_000004 in src/corpus_sessions.py reads only a "
                    "'delay' onset).",
    "inagaki_alm5": "no pre-sample baseline onset is extracted anywhere in this project's existing "
                    "ALM handling (load_alm_raw_session only reads Delay_start-aligned counts).",
    "ds005034": "no pre-trigger baseline window beyond BASELINE_WINDOW's own -3.7 to -2.7 s "
                "immediately preceding the same trigger is separately validated per admitted trial "
                "here; the encoding window itself sits between that baseline and the trigger, so a "
                "third, non-overlapping baseline window was not attempted in this pass.",
    "ds006848": "no pre-encoding baseline onset distinct from the trial's own Baseline_2s marker "
                "(which precedes encoding, not maintenance) is wired into a maintenance-comparable "
                "control in this pass.",
}
GAP_ZERO_CAVEAT = ("this corpus's encoding and maintenance windows abut (median real gap 0.0 s): "
                    "the task design places delay onset immediately at encoding offset, so this cell "
                    "is an adjacent-time comparison, not a test across a real temporal boundary like "
                    "this analysis's other corpora (median gaps of roughly 0.1-1.0 s) -- read it as a "
                    "weaker test of the same claim.")


def _rutishauser_lineage_sessions(dataset_key, files_with_subject, content_field, content_divisor):
    """Shared body for dandi_000469 / dandi_001187 / dandi_000673: build psth
    over the encoding (Encoding1) and maintenance windows, with BOTH a
    set-size (load) label and a stimulus-category (content) label, mirroring
    run_000469_pipeline.py's own window construction (psth/psth_enc) but
    keeping every load so the load label has variance."""
    enc_win, maint_win = 1.0, 2.3
    min_units = 15
    for subject, path, trials_key in files_with_subject:
        with h5py.File(str(path), "r") as f:
            if "units" not in f or trials_key not in f.get("intervals", {}):
                continue
            n_units_raw = int(f["units/id"].shape[0])
            if n_units_raw < min_units:
                continue
            trials = f[f"intervals/{trials_key}"]
            loads = trials["loads"][:].astype(int)
            content_raw = trials[content_field][:].astype(int)
            t_fix = trials["timestamps_FixationCross"][:]
            t_enc1 = trials["timestamps_Encoding1"][:]
            t_maint = trials["timestamps_Maintenance"][:]
            response_acc = trials["response_accuracy"][:].astype(bool)
            spike_lists = load_spike_times(f)

        content = content_raw // content_divisor if content_divisor > 1 else content_raw
        load_ok = ((loads == 1) | (loads == 3)) & response_acc
        content_ok = response_acc
        if response_acc.mean() < MIN_SESSION_ACCURACY:
            continue

        # union of trials needed by either label, so spikes/psth are built only once
        mask = load_ok | content_ok
        if mask.sum() < 2 * MIN_TRIALS_PER_CLASS:
            continue

        rate_mask = low_rate_unit_mask(spike_lists, t_maint[mask], maint_win)
        n_units = int(rate_mask.sum())
        if n_units < min_units:
            continue
        spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]

        t_fix_m, t_enc1_m, t_maint_m = t_fix[mask], t_enc1[mask], t_maint[mask]
        psth_baseline = build_psth(spike_lists, t_fix_m, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                                    window_s=EPOCH_WINDOWS_S["baseline"])
        psth_enc = build_psth(spike_lists, t_enc1_m, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=enc_win)
        psth_maint = build_psth(spike_lists, t_maint_m, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=maint_win)
        gap = float(np.median(t_maint_m - t_enc1_m - enc_win))

        load_sub = load_ok[mask]
        label_data = {}
        min_class_count, admitted_at_15 = {}, {}
        loads_sub_all = loads[mask]
        if min((loads_sub_all[load_sub] == 1).sum(), (loads_sub_all[load_sub] == 3).sum()) >= MIN_TRIALS_PER_CLASS:
            y_load = (loads_sub_all[load_sub] == 3).astype(float)
            min_class = int(min((y_load == 0).sum(), (y_load == 1).sum()))
            label_data["load"] = {"a": psth_enc[load_sub], "b": psth_maint[load_sub],
                                  "baseline": psth_baseline[load_sub], "y": y_load,
                                  "class_counts": None}
            min_class_count["load"], admitted_at_15["load"] = min_class, min_class >= MIN_TRIALS_PER_CLASS_COMPARISON

        content_sub = content_ok[mask]
        content_vals = content[mask][content_sub]
        classes, class_counts = np.unique(content_vals, return_counts=True)
        if len(classes) >= 2 and class_counts.min() >= MIN_TRIALS_PER_CLASS:
            y_content = content_vals.astype(float)
            min_class = int(class_counts.min())
            label_data["content"] = {"a": psth_enc[content_sub], "b": psth_maint[content_sub],
                                     "baseline": psth_baseline[content_sub], "y": y_content,
                                     "class_counts": {int(c): int(n) for c, n in zip(classes, class_counts)}}
            min_class_count["content"], admitted_at_15["content"] = min_class, min_class >= MIN_TRIALS_PER_CLASS_COMPARISON

        if not label_data:
            continue

        yield {
            "session_key": path.stem, "subject": subject, "gap": gap,
            "label_data": label_data,
            "min_class_count": {"load": min_class_count.get("load"), "content": min_class_count.get("content")},
            "admitted_at_15": {"load": admitted_at_15.get("load"), "content": admitted_at_15.get("content")},
        }


def iter_dandi_000469_sessions(root):
    directory = root / "000469"
    files = []
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*_ses-2_ecephys+image.nwb")):
            files.append((subject_dir.name, path, "trials"))
    yield from _rutishauser_lineage_sessions("dandi_000469", files, "loadsEnc1_PicIDs", 1)


def iter_dandi_001187_sessions(root):
    files = []
    for meta in canonical_sessions():
        if meta["primary_release"] != "001187":
            continue
        path = root / meta["primary_path"]
        if path.exists():
            files.append((meta["patient"], path, "WM_trials"))
    yield from _rutishauser_lineage_sessions("dandi_001187", files, "PicIDs_Encoding1", 100)


def iter_dandi_000673_sessions(root):
    """The FULL dandi_000673 release (every session in the release, not only
    the subset that is NOT also released under 001187). 001187 is preferred
    as this project's primary view of the shared patients; this arm is a
    linked sensitivity view of the whole 000673 release, not a disjoint
    residual -- subject_independence.py already resolves both identifiers to
    one group, so nothing here is double counted as independent evidence."""
    registry = json.loads(CANONICAL_RECORDING_REGISTRY_PATH.read_text())
    files = [(row["patient"], root / row["path"], "trials")
             for row in registry if row.get("release") == "000673"]
    yield from _rutishauser_lineage_sessions("dandi_000673", files, "PicIDs_Encoding1", 100)


def iter_dandi_000574_sessions(root):
    directory = root / "000574"
    min_units = 8
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(str(path), "r") as f:
                if "units" not in f or f["units/id"].shape[0] < min_units:
                    continue
                trials = f["intervals/trials"]
                start_time = trials["start_time"][:]
                set_size = trials["set_size"][:].astype(int)
                correct = trials["correct"][:].astype(bool)
                artifact = trials["artifact"][:].astype(bool)
                spike_lists = load_spike_times(f)
            keep = (~artifact) & correct & ((set_size == 4) | (set_size == 8))
            if keep.sum() < 2 * MIN_TRIALS_PER_CLASS:
                continue
            min_class = min((set_size[keep] == 4).sum(), (set_size[keep] == 8).sum())
            if min_class < MIN_TRIALS_PER_CLASS:
                continue

            start = start_time[keep]
            maint_onsets = start + 3.0
            rate_mask = low_rate_unit_mask(spike_lists, maint_onsets, BORAN_EPOCH_WINDOWS_S["delay"])
            n_units = int(rate_mask.sum())
            if n_units < min_units:
                continue
            spike_lists = [spk for spk, keep2 in zip(spike_lists, rate_mask) if keep2]

            baseline_onsets = start + 0.0
            enc_onsets = start + 1.0
            psth_baseline = build_psth(spike_lists, baseline_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                                        window_s=BORAN_EPOCH_WINDOWS_S["baseline"])
            psth_enc = build_psth(spike_lists, enc_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                                   window_s=BORAN_EPOCH_WINDOWS_S["encoding"])
            psth_maint = build_psth(spike_lists, maint_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                                     window_s=BORAN_EPOCH_WINDOWS_S["delay"])
            y = (set_size[keep] == 8).astype(float)
            gap = float(np.median(maint_onsets - (enc_onsets + BORAN_EPOCH_WINDOWS_S["encoding"])))
            yield {
                "session_key": path.stem, "subject": subject_dir.name, "gap": gap,
                "label_data": {"load": {"a": psth_enc, "b": psth_maint, "baseline": psth_baseline,
                                        "y": y, "class_counts": None}},
                "min_class_count": {"load": int(min_class), "content": None},
                "admitted_at_15": {"load": bool(min_class >= MIN_TRIALS_PER_CLASS_COMPARISON), "content": None},
            }


def iter_pfc4_sessions(root):
    from corpus_sessions import pfc4_data_directory
    enc_win, maint_win = 0.5, 3.0
    min_units, min_rate_hz = PFC4_MIN_UNITS, PFC4_MIN_UNIT_RATE_HZ
    directory = pfc4_data_directory(root)
    if not directory.is_dir():
        return
    for animal_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        for path in sorted(animal_dir.rglob("*.mat")):
            result = loadmat(path, struct_as_record=False, squeeze_me=True)["result"]
            header = list(result[0])
            so1_col, sf1_col, so2_col = header.index("SO1"), header.index("SF1"), header.index("SO2")
            f1_col = header.index("f1")
            trial_idx, so1_arr, sf1_arr, so2_arr, f1_arr = [], [], [], [], []
            for r in range(1, result.shape[0]):
                row = result[r]
                so1, sf1, so2 = np.atleast_1d(row[so1_col]), np.atleast_1d(row[sf1_col]), np.atleast_1d(row[so2_col])
                if so1.size != 1 or sf1.size != 1 or so2.size != 1:
                    continue
                if float(sf1[0] - so1[0]) / 1000.0 < enc_win or float(so2[0] - sf1[0]) / 1000.0 < maint_win:
                    continue
                trial_idx.append(r - 1)
                so1_arr.append(float(so1[0]) / 1000.0)
                sf1_arr.append(float(sf1[0]) / 1000.0)
                so2_arr.append(float(so2[0]) / 1000.0)
                f1_arr.append(float(np.atleast_1d(row[f1_col])[0]))
            if len(trial_idx) < 2 * MIN_TRIALS_PER_CLASS:
                continue
            f1_arr = np.array(f1_arr)
            median_f1 = np.median(f1_arr)
            y = (f1_arr > median_f1).astype(float)
            min_class = min((y == 0).sum(), (y == 1).sum())
            if min_class < MIN_TRIALS_PER_CLASS:
                continue

            n_cells = 7
            spikes_col = header.index("spikes")

            counts_enc = np.zeros((len(trial_idx), n_cells, int(round(enc_win * 1000 / BIN_MS))), dtype=float)
            counts_maint = np.zeros((len(trial_idx), n_cells, int(round(maint_win * 1000 / BIN_MS))), dtype=float)
            starts_enc = np.arange(0.0, enc_win, BIN_MS / 1000.0)
            starts_maint = np.arange(0.0, maint_win, BIN_MS / 1000.0)
            for row_i, r in enumerate(trial_idx):
                row = result[r + 1]
                spikes = row[spikes_col]
                for cell in range(n_cells):
                    st = np.atleast_1d(spikes[cell]).astype(float) / 1000.0
                    rel_enc = st - so1_arr[row_i]
                    counts_enc[row_i, cell], _ = np.histogram(rel_enc, bins=np.append(starts_enc, enc_win))
                    rel_maint = st - sf1_arr[row_i]
                    counts_maint[row_i, cell], _ = np.histogram(rel_maint, bins=np.append(starts_maint, maint_win))

            rates = counts_maint.sum(axis=(0, 2)) / (len(trial_idx) * maint_win)
            unit_mask = rates >= min_rate_hz
            if unit_mask.sum() < min_units:
                continue
            counts_enc = counts_enc[:, unit_mask]
            counts_maint = counts_maint[:, unit_mask]
            gap = float(np.median(sf1_arr) - np.median(so1_arr) - enc_win)
            yield {
                "session_key": path.stem, "subject": animal_dir.name, "gap": gap,
                "label_data": {"content": {"a": counts_enc, "b": counts_maint, "baseline": None,
                                           "y": y, "class_counts": None}},
                "min_class_count": {"load": None, "content": int(min_class)},
                "admitted_at_15": {"load": None, "content": bool(min_class >= MIN_TRIALS_PER_CLASS_COMPARISON)},
            }


def iter_dandi_000004_sessions(root):
    """New: adds the encoding-window read the existing iter_dandi_000004
    (src/corpus_sessions.py) never performs. Each admitted recognition trial
    already has its own stim_on/stim_off (encoding) period, verified directly
    against the NWB trial table, entirely before its own delay1/delay2
    maintenance period -- a genuine single-trial encode-then-maintain design,
    not a study/test split across separate trials."""
    from corpus_sessions import recognition_correct

    directory = root / "000004"
    required_columns = ("stim_phase", "new_old_labels_recog", "response_value", "response_time",
                         "delay1_time", "delay2_time", "stimCategory", "stim_on_time", "stim_off_time")
    enc_win, min_units = 1.0, MIN_UNITS_PER_REGION
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(str(path), "r") as f:
                if "units" not in f or "intervals/trials" not in f:
                    continue
                trials = f["intervals/trials"]
                if not all(c in trials for c in required_columns):
                    continue
                spike_lists_all = load_spike_times(f)
                unit_regions = resolve_unit_regions(f, "nwb_hemisphere_prefixed_structure")["region"]
                phase = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in trials["stim_phase"][:]])
                labels = np.array([v.decode() if isinstance(v, bytes) else str(v)
                                    for v in trials["new_old_labels_recog"][:]])
                responses = trials["response_value"][:].astype(float)
                response_time = trials["response_time"][:].astype(float)
                delay1 = trials["delay1_time"][:].astype(float)
                stim_on = trials["stim_on_time"][:].astype(float)
                stim_category = trials["stimCategory"][:].astype(float)
            candidate = ((phase == "recog") & np.isin(labels, ("0", "1"))
                         & np.isin(responses, np.arange(31, 37)) & np.isfinite(delay1)
                         & np.isfinite(stim_on) & np.isfinite(response_time))
            if candidate.sum() < 2 * MIN_TRIALS_PER_CLASS:
                continue
            content = stim_category[candidate]
            classes, class_counts = np.unique(content, return_counts=True)
            if len(classes) < 2 or class_counts.min() < MIN_TRIALS_PER_CLASS:
                continue

            for region in ("hippocampus", "amygdala"):
                spike_lists = region_filtered_units(spike_lists_all, unit_regions, region,
                                                     delay1[candidate], 1.0)
                if spike_lists is None or len(spike_lists) < min_units:
                    continue
                enc_onsets = stim_on[candidate]
                maint_onsets = delay1[candidate]
                psth_enc = build_psth(spike_lists, enc_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=enc_win)
                psth_maint = build_psth(spike_lists, maint_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=1.0)
                gap = float(np.median(maint_onsets - (enc_onsets + enc_win)))
                yield {
                    "session_key": f"{path.stem}_{region}", "subject": subject_dir.name, "gap": gap,
                    "label_data": {"content": {
                        "a": psth_enc, "b": psth_maint, "baseline": None, "y": content.astype(float),
                        "class_counts": {int(c): int(n) for c, n in zip(classes, class_counts)}}},
                    "min_class_count": {"load": None, "content": int(class_counts.min())},
                    "admitted_at_15": {"load": None,
                                       "content": bool(class_counts.min() >= MIN_TRIALS_PER_CLASS_COMPARISON)},
                }


def iter_inagaki_alm5_sessions(root):
    """New: adds the sample-epoch (Sample_start) read the existing loader
    (load_alm_raw_session in src/corpus_sessions.py) never performs, using
    the same eligibility filters and reusing _alm_build_counts /
    _alm_trial_condition directly rather than re-deriving them."""
    sample_win, delay_win = 1.0, 2.0
    min_units, min_rate_hz, min_trials_arm = ALM_MIN_UNITS, ALM_MIN_UNIT_RATE_HZ, ALM_MIN_TRIALS_PER_ARM
    directory = alm_data_directory(root)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.mat")):
        units = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["unit"])
        behavior = units[0].Behavior
        trial_info = units[0].Trial_info
        trial_type = np.asarray(behavior.Trial_types_of_response_vector, dtype=int).reshape(-1)
        stimulation = np.asarray(behavior.stim_trial_vector, dtype=int).reshape(-1)
        sample_start = np.asarray(behavior.Sample_start, dtype=float).reshape(-1)
        delay_start = np.asarray(behavior.Delay_start, dtype=float).reshape(-1)
        delay_duration = np.asarray(behavior.delay_dur, dtype=float).reshape(-1)
        condition = _alm_trial_condition(np.asarray(trial_info.Trial_types).reshape(-1))
        start, stop = np.asarray(trial_info.Trial_range_to_analyze, dtype=int).reshape(-1) - 1
        eligible = np.arange(start, stop + 1)
        eligible = eligible[(trial_type[eligible] < 5) & (delay_duration[eligible] >= delay_win)
                             & (delay_start[eligible] - sample_start[eligible] >= sample_win)
                             & (stimulation[eligible] == 0)]
        y = condition[eligible]
        min_class = min((y == 0).sum(), (y == 1).sum()) if len(y) else 0
        if min_class < MIN_TRIALS_PER_CLASS:
            continue

        counts_enc = _alm_build_counts(units, sample_start, eligible, BIN_MS, sample_win)
        counts_maint = _alm_build_counts(units, delay_start, eligible, BIN_MS, delay_win)
        rates = counts_maint.sum(axis=(0, 2)) / (len(eligible) * delay_win)
        unit_mask = rates >= min_rate_hz
        if unit_mask.sum() < min_units:
            continue
        counts_enc, counts_maint = counts_enc[:, unit_mask], counts_maint[:, unit_mask]
        if len(eligible) < min_trials_arm:
            continue
        gap = float(np.median(delay_start[eligible] - (sample_start[eligible] + sample_win)))
        yield {
            "session_key": path.stem, "subject": path.stem.split("_")[0], "gap": gap,
            # named "instructed_movement_direction", never "content": this is a cued motor plan
            # (which side to lick), not a memorandum of stimulus content -- see MOVEMENT_LABEL_DEFINITIONS.
            "label_data": {"instructed_movement_direction": {
                "a": counts_enc, "b": counts_maint, "baseline": None,
                "y": y.astype(float), "class_counts": None}},
            "min_class_count": {"instructed_movement_direction": int(min_class)},
            "admitted_at_15": {"instructed_movement_direction": bool(min_class >= MIN_TRIALS_PER_CLASS_COMPARISON)},
        }


def iter_ds005034_sessions(root):
    """New: adds an encoding-window read from the SAME continuous EEG this
    project's existing iter_ds005034 already loads, at an earlier offset
    (-2.5 to -0.5 s relative to the same trigger that anchors its own
    BASELINE_WINDOW=(-3.7,-2.7) and DELAY_WINDOW=(0.5,6.5)); no new field or
    file is read. No content label exists for this corpus (see
    item_id_unavailable_reason in iter_ds005034 -- verified again here), so
    only a load (list-length) arm is fit. Each session is tagged sham/verum
    (stimulation_session) so _process_corpus can fit them as separate arms
    rather than averaging a stimulated and an unstimulated brain state
    together before resampling."""
    from preprocessing import load_events
    from corpus_sessions import _scalp_delay_bin_power

    directory = root / "ds005034"
    enc_win, delay_win, n_bins = 2.0, 6.0, SCALP_EEG_DELAY_BINS
    for subject_dir in sorted(directory.glob("sub-*")):
        for session in ("sham", "verum"):
            eeg_dir = subject_dir / f"ses-{session}" / "eeg"
            set_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_eeg.set"
            if not set_path.is_file():
                continue
            events_path = eeg_dir / f"{subject_dir.name}_ses-{session}_task-memory_events.tsv"
            events = load_events(events_path)
            loads = np.array([e["load"] for e in events], dtype=float)
            if len(events) < 2 * MIN_TRIALS_PER_CLASS or min((loads == 4).sum(), (loads == 6).sum()) < MIN_TRIALS_PER_CLASS:
                continue
            payload = loadmat(set_path, variable_names=("data", "srate"), squeeze_me=True)
            data_tc = np.asarray(payload["data"], dtype=np.float64).T
            srate = float(payload["srate"])
            onset_delay = np.array([int(round((e["onset"] + 0.5) * srate)) for e in events])
            onset_enc = np.array([int(round((e["onset"] - 2.5) * srate)) for e in events])
            counts_maint = _scalp_delay_bin_power(data_tc, srate, onset_delay, window_s=delay_win, n_bins=n_bins)
            counts_enc = _scalp_delay_bin_power(data_tc, srate, onset_enc, window_s=enc_win, n_bins=n_bins)
            keep = np.isfinite(counts_maint).all(axis=(1, 2)) & np.isfinite(counts_enc).all(axis=(1, 2))
            loads_k = loads[keep]
            min_class = min((loads_k == 4).sum(), (loads_k == 6).sum()) if keep.sum() else 0
            if min_class < MIN_TRIALS_PER_CLASS:
                continue
            y = (loads_k == 6).astype(float)
            gap = 2.0  # trigger-2.5..-0.5 (encoding) to trigger+0.5 (delay): fixed by construction
            yield {
                "session_key": f"{subject_dir.name}_ses-{session}", "subject": subject_dir.name, "gap": gap,
                "stimulation_session": session,
                "label_data": {"load": {"a": counts_enc[keep], "b": counts_maint[keep], "baseline": None,
                                        "y": y, "class_counts": None}},
                "min_class_count": {"load": int(min_class), "content": None},
                "admitted_at_15": {"load": bool(min_class >= MIN_TRIALS_PER_CLASS_COMPARISON), "content": None},
            }


def _ds006848_encoding_onset(events, idx):
    j = idx - 1
    while j >= 0 and str(events.iloc[j]["trial_type"]).startswith("Encoding_"):
        j -= 1
    first = j + 1
    return float(events.iloc[first]["onset"]) if first < idx else None


def iter_ds006848_sessions(root):
    """New: for each admitted trial, walks backward from its matched
    Retention_* event to the first contiguous Encoding_* event (digit
    presentation or simultaneous-set presentation) -- verified directly
    against a raw events.tsv (Encoding_DigitValue_*/Encoding_Set_Simultaneous
    rows sit immediately before each Retention_* row). Reuses
    corpus_sessions._scalp_delay_bin_power for both windows."""
    import mne
    from corpus_sessions import _scalp_delay_bin_power

    directory = root / "ds006848"
    retention_codes = {28: "Simultaneous", 40: "Fast", 60: "Fast+delay", 100: "Slow"}
    delay_win, enc_win, n_bins = 6.0, 1.0, SCALP_EEG_DELAY_BINS
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

        onset_delay, onset_enc, memorandum = [], [], []
        next_row = {name: 0 for name in retention_codes.values()}
        for idx, event in events[events["value"].isin(retention_codes)].iterrows():
            condition = retention_codes[int(event["value"])]
            rows = beh[beh["condition"] == condition]
            row_index = next_row[condition]
            if row_index >= len(rows):
                continue
            next_row[condition] += 1
            enc_onset = _ds006848_encoding_onset(events, idx)
            if enc_onset is None:
                continue
            row = rows.iloc[row_index]
            onset_delay.append(int(round(float(event["onset"]) * srate)))
            onset_enc.append(int(round(enc_onset * srate)))
            memorandum.append(int(str(row["sequence"])[0]))
        onset_delay, onset_enc = np.asarray(onset_delay, dtype=int), np.asarray(onset_enc, dtype=int)
        memorandum = np.asarray(memorandum, dtype=float)
        if len(onset_delay) < 2 * MIN_TRIALS_PER_CLASS:
            continue
        counts_maint = _scalp_delay_bin_power(data_tc, srate, onset_delay, window_s=delay_win, n_bins=n_bins)
        counts_enc = _scalp_delay_bin_power(data_tc, srate, onset_enc, window_s=enc_win, n_bins=n_bins)
        keep = np.isfinite(counts_maint).all(axis=(1, 2)) & np.isfinite(counts_enc).all(axis=(1, 2))
        content = memorandum[keep]
        classes, class_counts = np.unique(content, return_counts=True)
        if len(classes) < 2 or class_counts.min() < MIN_TRIALS_PER_CLASS:
            continue
        gap = float(np.median((onset_delay[keep] - onset_enc[keep]) / srate - enc_win))
        yield {
            "session_key": f"{subject_dir.name}_verbalwm", "subject": subject_dir.name, "gap": gap,
            "label_data": {"content": {
                "a": counts_enc[keep], "b": counts_maint[keep], "baseline": None, "y": content,
                "class_counts": {int(c): int(n) for c, n in zip(classes, class_counts)}}},
            "min_class_count": {"load": None, "content": int(class_counts.min())},
            "admitted_at_15": {"load": None, "content": bool(class_counts.min() >= MIN_TRIALS_PER_CLASS_COMPARISON)},
        }


SESSION_ITERATORS = {
    "dandi_000469": iter_dandi_000469_sessions,
    "dandi_001187": iter_dandi_001187_sessions,
    "dandi_000673": iter_dandi_000673_sessions,
    "dandi_000574": iter_dandi_000574_sessions,
    "pfc4": iter_pfc4_sessions,
    "dandi_000004": iter_dandi_000004_sessions,
    "inagaki_alm5": iter_inagaki_alm5_sessions,
    "ds005034": iter_ds005034_sessions,
    "ds006848": iter_ds006848_sessions,
}
NPZ_PREFIX = {
    "dandi_000469": "dandi000469_ctg_cross_window",
    "dandi_001187": "dandi001187_ctg_cross_window",
    "dandi_000673": "dandi000673_ctg_cross_window",
    "dandi_000574": "dandi000574_units_ctg_cross_window",
    "pfc4": "pfc4_ctg_cross_window",
    "dandi_000004": "dandi000004_ctg_cross_window",
    "inagaki_alm5": "inagaki_alm5_ctg_cross_window",
    "ds005034": "ds005034_ctg_cross_window",
    "ds006848": "ds006848_ctg_cross_window",
}
LABEL_DEFINITIONS = {"load": LOAD_LABEL_DEFINITIONS, "content": CONTENT_LABEL_DEFINITIONS,
                     "instructed_movement_direction": MOVEMENT_LABEL_DEFINITIONS}
STANDARDISATION_ARMS_NOTE = (
    "the per-window and joint-scaler standardisation arms are reported side by side below for every "
    "direction, and neither is preferred. Where the two arms' estimates separate, the effect may "
    "reflect a difference in overall scale (gain or variance) between the encoding and maintenance "
    "windows rather than a difference in the geometry of the code, which is what a rotation question "
    "is actually about."
)
GAP_ZERO_CORPORA = ("dandi_000574", "pfc4")
SENSITIVITY_VIEW_NOTE = {
    "dandi_000673": "this is the FULL dandi_000673 release, including sessions also released under "
                    "dandi_001187 -- a linked sensitivity view of that shared patient lineage, not an "
                    "independent group (subject_independence.py resolves both to one group).",
}


def _process_session(dataset_key, sess, npz_prefix):
    tag = f"{dataset_key}_{sess['session_key']}"

    out = {}
    for label_name, ld in sess["label_data"].items():
        psth_a, psth_b, psth_baseline, y = ld["a"], ld["b"], ld["baseline"], ld["y"]
        n_units = psth_a.shape[1]
        n_comp = max(2, min(N_PC, n_units - 1))
        multiclass = len(np.unique(y)) > 2
        scorer = cross_window_ctg_multiclass if multiclass else cross_window_ctg
        n_splits = _choose_n_splits(y)
        arms = {}
        for arm in ("per_window", "joint_scaler"):
            rng = np.random.default_rng(stable_seed(f"{tag}_{label_name}_{arm}"))
            auc_mat, n_a = scorer(psth_a, psth_b, y, n_components=n_comp,
                                   n_splits=n_splits, rng=rng, joint_scaler=(arm == "joint_scaler"))
            arms[arm] = quadrant_stats(auc_mat, n_a)
            if arm == "per_window":
                suffix = "" if label_name == "load" else f"_{label_name}"
                np.savez_compressed(RESULTS / f"{npz_prefix}{suffix}_{sess['session_key']}.npz",
                                    auc_mat=auc_mat.astype(np.float32), n_encoding_timepoints=n_a,
                                    gap_seconds=sess["gap"])
        baseline = None
        if psth_baseline is not None:
            rng_b = np.random.default_rng(stable_seed(f"{tag}_{label_name}_baseline"))
            auc_mat_b, n_ab = scorer(psth_baseline, psth_b, y, n_components=n_comp,
                                     n_splits=n_splits, rng=rng_b, joint_scaler=False)
            baseline = quadrant_stats(auc_mat_b, n_ab)
        var_a, var_b = float(np.var(psth_a)), float(np.var(psth_b))
        variance_ratio_b_over_a = var_b / var_a if var_a > 0 else float("nan")
        out[label_name] = {"arms": arms, "baseline": baseline,
                           "variance_ratio_b_over_a": variance_ratio_b_over_a}
    return out


def _aggregate_cell(per_session, dataset_key, label_name, arm, direction, n_boot=N_BOOT):
    offdiag_key = f"{direction}_offdiag_effect"
    diag_key = "diag_a_mean_auc" if direction == "train_a_test_b" else "diag_b_mean_auc"
    subj_vals, subj_diags = {}, {}
    for srec in per_session.values():
        lrec = srec["label_results"].get(label_name)
        if lrec is None:
            continue
        source = lrec["arms"][arm] if arm != "baseline" else lrec["baseline"]
        if source is None:
            continue
        subj = srec["subject"]
        subj_vals.setdefault(subj, []).append(source[offdiag_key])
        subj_diags.setdefault(subj, []).append(source[diag_key])
    if not subj_vals:
        return None
    subject_values = np.array([np.mean(v) for v in subj_vals.values()])
    subject_diags = np.array([np.mean(v) for v in subj_diags.values()])
    n_sessions = sum(len(v) for v in subj_vals.values())
    cell = {"n_subjects": len(subject_values), "n_sessions": n_sessions,
            "train_window_diag_auc": float(np.mean(subject_diags))}
    if not np.all(np.isfinite(subject_values)) or not np.all(np.isfinite(subject_diags)):
        cell.update(cell_status="failure_nan", p_value=None, q_value=None,
                    estimand=None, ci_lower=None, ci_upper=None, mdd=None,
                    train_window_diag_auc_ci_lower=None, train_window_diag_auc_ci_upper=None,
                    transfer_ratio=None, transfer_ratio_ci_lower=None, transfer_ratio_ci_upper=None)
        return cell
    boot = subject_cluster_bootstrap_paired(
        subject_values, subject_diags, n_boot=n_boot,
        rng=np.random.default_rng(stable_seed(f"{dataset_key}_{label_name}_{arm}_{direction}_boot")))
    sign = paired_sign_flip_test(
        subject_values, np.zeros_like(subject_values), n_perm=N_PERM_SIGN_FLIP,
        alternative="two-sided",
        rng=np.random.default_rng(stable_seed(f"{dataset_key}_{label_name}_{arm}_{direction}_signflip")))
    cell.update(
        estimand=boot["offdiag_observed"], ci_lower=boot["offdiag_ci_lower"], ci_upper=boot["offdiag_ci_upper"],
        mdd=boot["offdiag_mdd"], p_value=sign["p_value"], q_value=None, cell_status=None,
        train_window_diag_auc_ci_lower=boot["diag_ci_lower"], train_window_diag_auc_ci_upper=boot["diag_ci_upper"],
        transfer_ratio=boot["transfer_ratio"], transfer_ratio_ci_lower=boot["transfer_ratio_ci_lower"],
        transfer_ratio_ci_upper=boot["transfer_ratio_ci_upper"],
        transfer_ratio_definition="estimand divided by (train-window diagonal AUC - 0.5): how much of "
                                  "the within-window decodable signal survives the crossing, with a "
                                  "95% interval from the same paired subject-cluster bootstrap draws "
                                  "as the estimand itself. A wide interval near a near-chance diagonal "
                                  "is the honest statement that this cell cannot tell us how much "
                                  "structure survived, not a suppressed or truncated result.",
    )
    return cell


def _label_arm_record(per_session, dataset_key, label_name, definition_suffix=""):
    sessions_with_label = [s for s in per_session.values() if label_name in s["label_results"]]
    if not sessions_with_label:
        return None, []
    has_baseline = any(s["label_results"][label_name]["baseline"] is not None for s in sessions_with_label)
    var_ratios = [s["label_results"][label_name]["variance_ratio_b_over_a"] for s in sessions_with_label]
    record = {
        "label_definition": LABEL_DEFINITIONS[label_name].get(dataset_key, "unavailable") + definition_suffix,
        "n_sessions": len(sessions_with_label),
        "n_subjects": len({s["subject"] for s in sessions_with_label}),
        "encoding_maintenance_variance_ratio_median": float(np.median(var_ratios)),
        "encoding_maintenance_variance_ratio_definition": "median across admitted sessions of "
            "var(maintenance-window firing rate) / var(encoding-window firing rate), pooled over "
            "trials/units/bins; a value far from 1 means the two windows are on different overall "
            "scales before any decoding is done.",
        "arms": {}, "baseline_control": None,
        "baseline_unavailable_reason": None if has_baseline else NO_BASELINE_REASON.get(dataset_key),
    }
    class_counts = {s["session_key"]: s["class_counts_by_label"][label_name] for s in sessions_with_label
                    if s["class_counts_by_label"].get(label_name)}
    if class_counts:
        record["class_counts_per_session"] = class_counts
    cells = []
    for arm in ("per_window", "joint_scaler"):
        record["arms"][arm] = {}
        for direction in ("train_a_test_b", "train_b_test_a"):
            cell = _aggregate_cell(per_session, dataset_key, label_name, arm, direction)
            record["arms"][arm][direction] = cell
            if cell is not None:
                cells.append(cell)
    if has_baseline:
        cell = _aggregate_cell(per_session, dataset_key, label_name, "baseline", "train_a_test_b")
        record["baseline_control"] = cell
        if cell is not None:
            cells.append(cell)
    return record, cells


def _process_corpus(dataset_key, root):
    npz_prefix = NPZ_PREFIX[dataset_key]
    per_session = {}
    label_names_present = set()
    for sess in SESSION_ITERATORS[dataset_key](root):
        label_results = _process_session(dataset_key, sess, npz_prefix)
        label_names_present.update(sess["label_data"].keys())
        per_session[sess["session_key"]] = {
            "session_key": sess["session_key"],
            "subject": sess["subject"], "gap_seconds": sess["gap"],
            "stimulation_session": sess.get("stimulation_session"),
            "min_class_count": sess["min_class_count"], "admitted_at_15": sess["admitted_at_15"],
            "class_counts_by_label": {name: ld.get("class_counts")
                                      for name, ld in sess["label_data"].items()},
            "label_results": label_results,
        }
    if not per_session:
        return None, []

    gaps = [srec["gap_seconds"] for srec in per_session.values()]
    subjects = sorted({srec["subject"] for srec in per_session.values()})

    record = {
        "tier": TIERS[dataset_key],
        "independent_group": resolve_group(dataset_key),
        "real_gap_seconds_median": float(np.median(gaps)),
        "real_gap_seconds_definition": "maintenance-window onset minus (encoding-window onset + "
                                        "encoding-window length), per admitted trial, median across "
                                        "all admitted trials of the corpus; this is a real elapsed-time "
                                        "gap between two separately fitted windows, not a continuous "
                                        "sampled axis",
        "gap_effectively_zero_caveat": GAP_ZERO_CAVEAT if dataset_key in GAP_ZERO_CORPORA else None,
        "n_sessions": len(per_session), "n_subjects": len(subjects),
        "sensitivity_view_note": SENSITIVITY_VIEW_NOTE.get(dataset_key),
        "memorandum_type_caveat": MOVEMENT_TIER_CAVEAT.get(dataset_key),
        "admission_floor_used": MIN_TRIALS_PER_CLASS,
        "admission_floor_comparison": MIN_TRIALS_PER_CLASS_COMPARISON,
        "admitted_sessions_at_floor_used": {
            name: sum(1 for s in per_session.values() if s["min_class_count"].get(name) is not None)
            for name in label_names_present},
        "admitted_sessions_at_comparison_floor": {
            name: sum(1 for s in per_session.values() if s["admitted_at_15"].get(name))
            for name in label_names_present},
        "sessions": {k: {"subject": v["subject"], "gap_seconds": v["gap_seconds"],
                         "min_class_count": v["min_class_count"],
                         "stimulation_session": v["stimulation_session"]}
                     for k, v in per_session.items()},
        "label_arms": {},
    }
    all_cells = []
    for label_name in sorted(label_names_present):
        if dataset_key == "ds005034" and label_name == "load":
            for tag in ("sham", "verum", "pooled"):
                subset = per_session if tag == "pooled" else \
                    {k: v for k, v in per_session.items() if v["stimulation_session"] == tag}
                arm_record, cells = _label_arm_record(subset, dataset_key, label_name,
                                                      definition_suffix=STIMULATION_ARM_DEFINITION_SUFFIX[tag])
                if arm_record is not None:
                    record["label_arms"][f"load_{tag}"] = arm_record
                    if tag == "pooled":
                        for cell in cells:
                            cell["cell_status"] = "descriptive_not_in_fdr_family"
                    else:
                        all_cells.extend(cells)
            continue
        arm_record, cells = _label_arm_record(per_session, dataset_key, label_name)
        if arm_record is not None:
            record["label_arms"][label_name] = arm_record
            all_cells.extend(cells)
    return record, all_cells


def _write_artifact(artifact):
    with open(RESULTS / "cross_window_code_generalisation.json", "w") as f:
        json.dump(_json_safe(artifact), f, indent=2)


def main():
    registered = set(ALL_REGISTERED_CORPORA)
    accounted = set(FITTED_CORPORA) | set(EXCLUSIONS.keys())
    if registered != accounted:
        raise RuntimeError(f"corpus accounting mismatch: {registered ^ accounted}")

    root = data_root()
    missing = missing_files(root, list(FITTED_CORPORA))
    if missing:
        out = {"status": "blocked_incomplete_data",
              "missing_files": [{"corpus": c, "path": p} for c, p in missing]}
        _write_artifact(out)
        print(f"INCOMPLETE DATA -- {len(missing)} missing file(s), stopping without fitting.")
        for c, p in missing[:20]:
            print(f"  {c}: {p}")
        return

    excluded_out = {}
    for k, v in EXCLUSIONS.items():
        missing_for_corpus = missing_files(root, [k])
        excluded_out[k] = {**v, "tier": TIERS[k], "download_state_exclusion": False,
                           "download_state_verified_complete": len(missing_for_corpus) == 0,
                           "download_state_check": "data_integrity.missing_files against this "
                                                   "corpus's own registered completeness check"}

    predeclared_rules = {
        "estimand": "the mean of (AUC - 0.5) over the off-diagonal quadrant that trains a decoder on "
                    "encoding-window timepoints and tests it on maintenance-window timepoints; "
                    "separately, the same quantity for the transposed quadrant (train on maintenance, "
                    "test on encoding). Within-window (diagonal-block) decoding is reported as "
                    "context, never as the estimand. Fitted separately for a load (set-size) label "
                    "and a content (stimulus-identity) label wherever a corpus carries both. The mouse "
                    "corpus's own decoded variable is an instructed movement direction, not stimulus "
                    "content, and is kept under its own name (instructed_movement_direction) rather "
                    "than folded into 'content' anywhere in this artifact.",
        "resampling_unit": "subject/patient; sessions belonging to one subject are averaged before "
                           "any resampling, the trial is never the resampling unit",
        "clearing_rule": "a cell is clear only if its Benjamini-Hochberg FDR q (alpha=0.05, computed "
                         "within its own evidence-tier family, load and content cells pooled into the "
                         "same tier family) is significant AND its 95% subject-cluster bootstrap "
                         "interval excludes zero; either alone leaves the cell inconclusive, never a "
                         "powered null, since no reference effect size on the AUC scale has been "
                         "named for this project",
        "clearing_rule_status": CLEARING_RULE_STATUS,
        "minimum_detectable_difference_definition": f"{Z_80_POWER} times the standard deviation "
                                                     "of the subject-cluster bootstrap draws (minimum "
                                                     "detectable difference at 80% power)",
        "declared_before_fitting": True,
    }
    artifact = {
        "predeclared_rules": predeclared_rules,
        "joint_axis_definition": "the encoding-window timepoints followed by the maintenance-window "
                                 "timepoints, concatenated into one array and fitted in one model; "
                                 "the two windows are not adjacent in real time and this axis is "
                                 "never a continuous sampled clock -- the real gap between them is "
                                 "recorded per corpus below",
        "train_window_diagonal_reference_value": f"{MIN_DIAG_AUC} is the within-window decodability "
            f"reference value; the diagonal is reported for every cell with its own paired bootstrap "
            f"interval and never gates the off-diagonal estimand. The diagonal is a far noisier estimate (roughly one "
            f"value per time bin) than the off-diagonal estimand (roughly bins-squared values); see "
            f"baseline_control in each corpus for the measured diagonal and its interval, and "
            f"transfer_ratio (null wherever the diagonal is not distinguishable from 0.5 at its own "
            f"bootstrap standard error) for whether the quadrant estimator reads as biased there -- a "
            f"diagonal below this value is not by itself evidence against an off-diagonal effect.",
        "evidence_tiers": {"human_primary": "primary; carries headline claims",
                           "nonhuman_primate_supplementary": "supplementary; supports a human claim, "
                                                              "never substitutes for or replicates one",
                           "mouse_supplementary_to_supplementary": "supplementary to the supplementary "
                                                                   "tier; never a headline, never a "
                                                                   "replication of a human claim"},
        "corpora": {}, "excluded_corpora": _json_safe(excluded_out),
        "all_registered_corpora": ALL_REGISTERED_CORPORA, "fitted_corpora": list(FITTED_CORPORA),
        "independent_groups": None, "status": "in_progress",
    }
    _write_artifact(artifact)

    fdr_cells = {"human_primary": [], "nonhuman_primate_supplementary": [],
                "mouse_supplementary_to_supplementary": []}
    for dataset_key in FITTED_CORPORA:
        print(f"=== {dataset_key} ===", flush=True)
        record, cells = _process_corpus(dataset_key, root)
        if record is None:
            artifact["corpora"][dataset_key] = {"status": "no_admitted_sessions", "tier": TIERS[dataset_key]}
        else:
            artifact["corpora"][dataset_key] = record
            fdr_cells[TIERS[dataset_key]].extend(cells)
            print(f"  {record['n_sessions']} sessions, {record['n_subjects']} subjects, "
                  f"label arms: {list(record['label_arms'].keys())}", flush=True)
        _write_artifact(artifact)

    for tier, cells in fdr_cells.items():
        testable = [c for c in cells if c.get("p_value") is not None]
        if not testable:
            continue
        q = fdr_bh(np.array([c["p_value"] for c in testable]), alpha=FDR_ALPHA)
        for cell, qv in zip(testable, q["q_values"]):
            cell["q_value"] = float(qv)
            cell["cell_status"] = cell_status(is_nan=False)

    for dataset_key, record in artifact["corpora"].items():
        if "label_arms" not in record:
            continue
        record["standardisation_arms_note"] = STANDARDISATION_ARMS_NOTE
        record["standardisation_arms_comparison"] = {}
        for label_name, arm_record in record["label_arms"].items():
            comparison = {}
            for direction in ("train_a_test_b", "train_b_test_a"):
                a = arm_record["arms"]["per_window"][direction]
                b = arm_record["arms"]["joint_scaler"][direction]
                if a is not None and b is not None:
                    both_estimated = a["estimand"] is not None and b["estimand"] is not None
                    comparison[direction] = {
                        "per_window": {"estimand": a["estimand"], "ci_lower": a["ci_lower"],
                                      "ci_upper": a["ci_upper"], "q_value": a["q_value"]},
                        "joint_scaler": {"estimand": b["estimand"], "ci_lower": b["ci_lower"],
                                        "ci_upper": b["ci_upper"], "q_value": b["q_value"]},
                        "estimand_difference_per_window_minus_joint_scaler":
                            (a["estimand"] - b["estimand"]) if both_estimated else None,
                    }
            record["standardisation_arms_comparison"][label_name] = comparison

    fitted_human = [k for k in FITTED_CORPORA if TIERS[k] == "human_primary"]
    fitted_nonhuman = [k for k in FITTED_CORPORA if TIERS[k] == "nonhuman_primate_supplementary"]
    fitted_mouse = [k for k in FITTED_CORPORA if TIERS[k] == "mouse_supplementary_to_supplementary"]
    artifact["independent_groups"] = {
        "human_primary": {"corpora": fitted_human,
                          "groups": sorted({resolve_group(k) for k in fitted_human}),
                          "n_independent_groups": count_independent_groups(fitted_human) if fitted_human else 0},
        "nonhuman_primate_supplementary": {
            "corpora": fitted_nonhuman,
            "groups": sorted({resolve_group(k) for k in fitted_nonhuman}),
            "n_independent_groups": count_independent_groups(fitted_nonhuman) if fitted_nonhuman else 0},
        "mouse_supplementary_to_supplementary": {
            "corpora": fitted_mouse,
            "groups": sorted({resolve_group(k) for k in fitted_mouse}),
            "n_independent_groups": count_independent_groups(fitted_mouse) if fitted_mouse else 0},
    }
    artifact["status"] = "complete"
    _write_artifact(artifact)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["cross_window_code_generalisation"] = _json_safe(artifact)
    print("Saved results/cross_window_code_generalisation.json")


if __name__ == "__main__":
    main()
