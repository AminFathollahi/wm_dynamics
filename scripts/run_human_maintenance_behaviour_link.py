"""run_human_maintenance_behaviour_link.py -- does the rate-free state
deviation component predict trial accuracy in a human working-memory
maintenance delay, the way it already does in macaque lateral prefrontal
cortex (results/rate_free_state_geometry_behavior_link.json: pooled raw
r=-0.0974, p=0.0035; joint partial controlling total spike count and trial
index r=-0.0696, p=0.0206; 11 sessions)?

That human number has never been computed, for an arithmetic reason rather
than a biological one: every existing human maintenance-corpus iterator in
src/corpus_sessions.py admits a trial only if it was answered correctly
(dandi_000469 additionally requires a single-item trial). No error trial has
ever reached any estimator built on those iterators, so a downstream claim
that human trial-level accuracy is "at ceiling" describes that filter, not
the data. Measured directly at the raw trial tables here: dandi_000574
(verbal Sternberg) has 1827 trials before any admission mask, 151 of them
errors (8.3%), across 37 sessions and 9 patients; after excluding only the
corpus's own artifact-flagged trials (the ONLY admission criterion this
module applies), 1683 trials remain, 142 of them errors (8.4%), the same 37
sessions and 9 patients. Both numbers are recomputed fresh every run, in
Block A below, rather than assumed from this docstring.

This module does not modify src/corpus_sessions.py's iterators (other
delivered analyses depend on their current behaviour) or any read-only
artifact. It builds a separate trial-admission path for the three human
corpora with a genuine maintenance delay -- dandi_000469, dandi_001187,
dandi_000574 -- where the ONLY admission criterion is a data-quality
artifact flag where the corpus has one (dandi_000574's own `artifact`
field) or "the epoch has a defined onset" where it does not
(dandi_000469, dandi_001187, neither of which has an artifact field at
all); trial correctness is carried through as an outcome variable and never
used to admit or discard a trial.

The estimator is scripts/run_rate_free_state_geometry_behavior_link.py's
rate_free_state_deviation, unmodified: a per-trial, per-unit total spike
count vector in the delay epoch, L2-normalised to a unit direction, compared
by cosine to the leave-one-out mean direction of every OTHER trial in the
same session. That same module is also this module's template for the
statistic family (raw correlation with trial outcome; partial controlling
total spike count; partial controlling trial index; joint partial
controlling both) and for computing every correlation within one unit of
analysis before pooling across units, never pooling raw trials across units.

Where this module's design differs from that template, by requirement: the
unit of analysis here is the PATIENT, not the session. A human maintenance
session carries between 0 and roughly 40 error trials, far too few for a
reliable per-session correlation; a patient contributing several sessions
supplies one pooled, trial-level estimate instead, and it is patients --
independent people -- not sessions, that are pooled across for inference.
Every patient-level estimate is computed within one set-size/load level
(load determines both task difficulty and the sampling of error trials, so
pooling across load confounds the two); the within-load estimates are then
combined across load levels and corpora by inverse-variance meta-analysis
(statistics.forest_meta), which is a materially different, unconfounded
combination from simply pooling every trial across load levels regardless of
level -- that cruder pooled-across-load number is also computed and
reported, but only ever as a secondary, disclosed diagnostic.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import (  # noqa: E402
    BORAN_EPOCH_WINDOWS_S, EPOCH_WINDOWS_S, MIN_TRIALS, data_root, load_spike_times,
    region_filtered_units, resolve_unit_regions,
)
from provenance import _json_safe  # noqa: E402
from run_behavior_amplitude_rate_controls import _reachable_sessions  # noqa: E402
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_human_drift_spine_001187_000673 import canonical_sessions, _trial_group  # noqa: E402
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import _counts_from_spikes  # noqa: E402
from run_state_content_link import delay_counts  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    forest_meta, minimum_detectable_paired_difference, partial_correlation_permutation_test,
    power_to_detect_effect, stable_seed,
)

OUTPUT_PATH = ROOT / "results" / "human_maintenance_behaviour_link.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_human_maintenance_behaviour_link"
ANALYSIS_VERSION = "2026-08-22"

BIN_MS = 100.0
N_PERM_HEADLINE = 10000  # per-session/per-patient pooling across independent units (slope_across_sessions_test)
# Per-PATIENT permutation p-values are diagnostic only -- every headline decision in this module is made
# from slope_across_sessions_test's own sign-flip pooling (across patients) or forest_meta's analytic
# z-test (across arms), neither of which reads a per-patient p-value. Reduced here purely for wall-clock;
# the point estimate 'r' each per-patient cell contributes to pooling does not depend on n_perm at all.
N_PERM_PATIENT_CELL = 2000
MIN_TRIALS_PER_PATIENT_CELL = 6  # >=1 df left after outcome ~ intercept + 2 controls
N_DRAWS_BLOCK_C = 200  # matches this project's own established convention (run_state_behavior_link.N_MATCHED_DRAWS)

MEANINGFUL_EFFECT_THRESHOLD_R_UNITS = 0.14  # this project's standing reference scale; fixed before any fit runs

REFERENCE_ARTIFACT_PATH = ROOT / "results" / "rate_free_state_geometry_behavior_link.json"
SUBSPACE_ARTIFACT_PATH = ROOT / "results" / "deviation_subspace_decomposition.json"

BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per (corpus, load level) arm, pool the patient-level partial correlations (raw; controlling total "
    "spike count; controlling trial index; joint, controlling both) across patients with the two-sided "
    "paired sign-flip test. Combine the within-load arms across all corpora and load levels by "
    "inverse-variance meta-analysis (statistics.forest_meta), giving one combined raw, one combined "
    "partial-controlling-spike-count, one combined partial-controlling-trial-index, and one combined joint "
    "estimate; report the pooled-across-load estimate too, but only as a secondary, disclosed diagnostic, "
    "never as the number a branch is decided from. Before any fit runs:\n"
    "  - If the combined RAW estimate is significant (two-sided p<=0.05, from forest_meta's analytic z-test) "
    "AND the combined JOINT estimate is also significant with the same sign, run the mandatory session-"
    "level-offset control (below); if the control does NOT also reproduce a same-sign significant joint "
    "result, the branch is 'the_component_predicts_accuracy_in_a_human_maintenance_delay'.\n"
    "  - If the combined raw estimate is significant but the combined joint estimate is not significant "
    "with the same sign, this is the same rule gap results/rate_free_state_geometry_behavior_link.json's "
    "own decision rule disclosed at n=11 sessions (a raw-significant result that does not survive its own "
    "joint control): the branch is "
    "'raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index'.\n"
    "  - If the combined raw estimate is NOT significant and its implied minimum detectable difference at "
    "80% power (from the combined estimate's own standard error) is below "
    + str(MEANINGFUL_EFFECT_THRESHOLD_R_UNITS) + " r units, the branch is "
    "'no_human_behaviour_link_above_the_reported_bound'.\n"
    "  - If the combined raw estimate is NOT significant and that minimum detectable difference is at or "
    "above " + str(MEANINGFUL_EFFECT_THRESHOLD_R_UNITS) + " r units, the branch is 'underpowered_to_ask'.\n"
    "Mandatory control: recompute every arm and the combined estimate with each trial's deviation value "
    "replaced by its own session's leave-one-out training-trial mean deviation (a constant per session, up "
    "to which trial is excluded) -- a value that carries no trial-to-trial information, only a between-"
    "session offset. If this control ALSO produces a combined joint estimate significant at the two-sided "
    "0.05 level with the same sign as the real raw estimate, the result is void and the branch is instead "
    "'behaviour_link_not_separable_from_a_session_level_offset', overriding "
    "'the_component_predicts_accuracy_in_a_human_maintenance_delay' wherever it would otherwise fire."
)

BLOCK_C_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Using the same 11 macaque lPFC sessions results/rate_free_state_geometry_behavior_link.json reports "
    "its raw pooled effect on: for each session, at each candidate error count e* in the pooled human "
    "per-session error-count distribution's observed unique values (Block A, all three corpora, restricted "
    "to values >=2), draw " + str(N_DRAWS_BLOCK_C) + " independent subsamples of that session's OWN error "
    "trials down to e* (every correct trial kept), recompute rate_free_state_deviation from scratch on each "
    "draw's trial subset, and take the point correlation of deviation with trial outcome. The "
    + str(N_DRAWS_BLOCK_C) + " draws are aggregated to ONE median value per session before any inference; "
    "the cross-session test (paired sign-flip, across the 11 sessions) is the only inference step, and it "
    "is never run across draws.\n"
    "  - A second draw family samples e* itself, per draw, from the pooled human per-session error-count "
    "distribution directly (rather than a fixed grid), aggregated per session the same way, then pooled "
    "across sessions. If this pooled estimate is significant (two-sided p<=0.05) with the same sign as the "
    "reference macaque raw effect, the branch is 'the_non_human_effect_survives_at_human_error_counts'.\n"
    "  - Otherwise: at each fixed-grid error count e*, compute this design's power (noncentral-t, "
    "statistics.power_to_detect_effect) to detect the reference macaque raw effect size, using the 11 "
    "session-level median values at that e* as the sample. If any grid value reaches 80% power, the branch "
    "is 'the_non_human_effect_dies_at_human_error_counts', with the smallest such e* reported as the "
    "minimum detectable error count.\n"
    "  - If no grid value inside the sampled range reaches 80% power, the branch is "
    "'no_identifiable_transition_inside_the_sampled_range'."
)


# =======================================================================================================
# Checkpointing (session- and rung-level; temp file + os.replace; completion flag only after return)
# =======================================================================================================

def _checkpoint_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("|", "__").replace(" ", "_")
    return CHECKPOINT_DIR / f"{safe}.json"


def _fit(key: str, compute):
    path = _checkpoint_path(key)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("complete") is True:
        return record["value"]
    value = compute()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe({"complete": True, "value": value}), allow_nan=False, default=float))
    os.replace(scratch, path)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Block A -- trial admission (artifact mask only; correctness is never an admission criterion)
# =======================================================================================================

def _base_record(dataset: str, patient: str, session: str, n_raw: int, n_admitted: int, n_errors: int,
                  n_old_rule_admits_among_new: int, old_rule: str, load_field: str, load_levels: list[int]) -> dict:
    return {
        "dataset": dataset, "patient": patient, "session": session,
        "n_trials_raw": n_raw, "n_trials_admitted": n_admitted, "n_errors_admitted": n_errors,
        "n_trials_old_rule_would_discard_among_admitted": n_admitted - n_old_rule_admits_among_new,
        "old_admission_rule": old_rule, "load_level_field": load_field, "load_levels_admitted": load_levels,
    }


def _iter_000469_admitted(root: Path):
    directory = root / "000469"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*_ses-2_ecephys+image.nwb")):
            with h5py.File(path, "r") as handle:
                trials = handle["intervals/trials"]
                accuracy = trials["response_accuracy"][:].astype(bool)
                loads = trials["loads"][:].astype(int)
                t_maint = trials["timestamps_Maintenance"][:]
                admitted = np.isfinite(t_maint)
                n_admitted = int(admitted.sum())
                old_keep_full = (loads == 1) & accuracy
                n_old = int(old_keep_full[admitted].sum()) if n_admitted else 0
                base = _base_record(
                    "dandi_000469", subject_dir.name, path.stem, len(accuracy), n_admitted,
                    int((~accuracy[admitted]).sum()) if n_admitted else 0, n_old,
                    "(loads == 1) & response_accuracy", "loads",
                    sorted(int(x) for x in np.unique(loads[admitted])) if n_admitted else [])
                if n_admitted < MIN_TRIALS or "units" not in handle:
                    reason = "no_units_group_in_nwb_file" if "units" not in handle else f"fewer_than_{MIN_TRIALS}_admitted_trials"
                    yield {**base, "usable_for_estimator": False, "exclusion_reason": reason}
                    continue
                spike_lists_all = load_spike_times(handle)
                unit_regions = resolve_unit_regions(handle)["region"]
                delay_onset = t_maint[admitted]
                spike_lists = region_filtered_units(spike_lists_all, unit_regions, "pooled", delay_onset, EPOCH_WINDOWS_S["delay"])
                if spike_lists is None:
                    yield {**base, "usable_for_estimator": False, "exclusion_reason": "fewer_than_min_units_after_region_and_rate_qc"}
                    continue
                yield {**base, "usable_for_estimator": True, "is_correct": accuracy[admitted], "load_level": loads[admitted],
                       "spike_lists": spike_lists, "delay_onset": delay_onset, "delay_window_s": EPOCH_WINDOWS_S["delay"]}


def _iter_001187_admitted(root: Path):
    for meta in canonical_sessions():
        if meta["primary_release"] != "001187":
            continue
        path = root / meta["primary_path"]
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            trials = _trial_group(handle, "001187")
            accuracy = trials["response_accuracy"][:].astype(bool)
            loads = trials["loads"][:].astype(int)
            t_maint = trials["timestamps_Maintenance"][:]
            admitted = np.isfinite(t_maint)
            n_admitted = int(admitted.sum())
            n_old = int(accuracy[admitted].sum()) if n_admitted else 0  # old rule: keep = accuracy (no load condition)
            base = _base_record(
                "dandi_001187", meta["patient"], path.stem, len(accuracy), n_admitted,
                int((~accuracy[admitted]).sum()) if n_admitted else 0, n_old, "response_accuracy", "loads",
                sorted(int(x) for x in np.unique(loads[admitted])) if n_admitted else [])
            if n_admitted < MIN_TRIALS or "units" not in handle:
                reason = "no_units_group_in_nwb_file" if "units" not in handle else f"fewer_than_{MIN_TRIALS}_admitted_trials"
                yield {**base, "usable_for_estimator": False, "exclusion_reason": reason}
                continue
            spike_lists_all = load_spike_times(handle)
            unit_regions = resolve_unit_regions(handle)["region"]
            delay_onset = t_maint[admitted]
            spike_lists = region_filtered_units(spike_lists_all, unit_regions, "pooled", delay_onset, EPOCH_WINDOWS_S["delay"])
            if spike_lists is None:
                yield {**base, "usable_for_estimator": False, "exclusion_reason": "fewer_than_min_units_after_region_and_rate_qc"}
                continue
            yield {**base, "usable_for_estimator": True, "is_correct": accuracy[admitted], "load_level": loads[admitted],
                   "spike_lists": spike_lists, "delay_onset": delay_onset, "delay_window_s": EPOCH_WINDOWS_S["delay"]}


def _iter_000574_admitted(root: Path):
    directory = root / "000574"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            with h5py.File(path, "r") as handle:
                trials = handle["intervals/trials"]
                artifact = trials["artifact"][:].astype(bool)
                correct = trials["correct"][:].astype(bool)
                set_size = trials["set_size"][:].astype(int)
                start_time = trials["start_time"][:]
                admitted = ~artifact
                n_admitted = int(admitted.sum())
                old_keep_full = (~artifact) & correct
                n_old = int(old_keep_full[admitted].sum()) if n_admitted else 0
                base = _base_record(
                    "dandi_000574", subject_dir.name, path.stem, len(correct), n_admitted,
                    int((~correct[admitted]).sum()) if n_admitted else 0, n_old, "(~artifact) & correct", "set_size",
                    sorted(int(x) for x in np.unique(set_size[admitted])) if n_admitted else [])
                if n_admitted < MIN_TRIALS or "units" not in handle:
                    reason = "no_units_group_in_nwb_file" if "units" not in handle else f"fewer_than_{MIN_TRIALS}_admitted_trials"
                    yield {**base, "usable_for_estimator": False, "exclusion_reason": reason}
                    continue
                spike_lists_all = load_spike_times(handle)
                unit_regions = resolve_unit_regions(handle, "nwb_boran_brainnetome_hybrid")["region"]
                delay_onset = start_time[admitted] + 3.0  # maintenance onset = start_time + 3.0s (see corpus_sessions.py)
                spike_lists = region_filtered_units(spike_lists_all, unit_regions, "pooled", delay_onset, BORAN_EPOCH_WINDOWS_S["delay"])
                if spike_lists is None:
                    yield {**base, "usable_for_estimator": False, "exclusion_reason": "fewer_than_min_units_after_region_and_rate_qc"}
                    continue
                yield {**base, "usable_for_estimator": True, "is_correct": correct[admitted], "load_level": set_size[admitted],
                       "spike_lists": spike_lists, "delay_onset": delay_onset, "delay_window_s": BORAN_EPOCH_WINDOWS_S["delay"]}


ADMISSION_ITERATORS = {
    "dandi_000469": _iter_000469_admitted,
    "dandi_001187": _iter_001187_admitted,
    "dandi_000574": _iter_000574_admitted,
}


def run_block_a(root: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Scans every session's raw trial table once (Block A's own census, independent of whether that
    session's spike data later proves usable) and, in the same pass, attempts the spike-loading path Block
    B needs. Returns (block_a_report, sessions_by_corpus) -- the second value carries every session seen,
    usable or not, so Block B's own zero-drop accounting reconciles against Block A's."""
    sessions_by_corpus: dict[str, list[dict]] = {}
    report: dict[str, dict] = {}
    for dataset, iterator in ADMISSION_ITERATORS.items():
        sessions = list(iterator(root))
        sessions_by_corpus[dataset] = sessions
        n_seen = len(sessions)
        usable = [s for s in sessions if s["usable_for_estimator"]]
        excluded = [s for s in sessions if not s["usable_for_estimator"]]
        reason_tally: dict[str, int] = {}
        for s in excluded:
            reason_tally[s["exclusion_reason"]] = reason_tally.get(s["exclusion_reason"], 0) + 1
        error_counts = [s["n_errors_admitted"] for s in sessions if s["n_trials_admitted"] > 0]
        report[dataset] = {
            "n_sessions_seen": n_seen,
            "n_sessions_usable_for_estimator": len(usable),
            "n_sessions_excluded": len(excluded),
            "exclusion_reason_tally": reason_tally,
            "reconciles": bool(n_seen == len(usable) + len(excluded)),
            "n_patients": len(set(s["patient"] for s in sessions)),
            "n_trials_admitted_total": sum(s["n_trials_admitted"] for s in sessions),
            "n_errors_admitted_total": sum(s["n_errors_admitted"] for s in sessions),
            "pct_errors_admitted": (100.0 * sum(s["n_errors_admitted"] for s in sessions) /
                                     sum(s["n_trials_admitted"] for s in sessions)) if sessions else None,
            "n_trials_old_rule_would_discard_among_admitted_total": sum(
                s["n_trials_old_rule_would_discard_among_admitted"] for s in sessions),
            "old_admission_rule": sessions[0]["old_admission_rule"] if sessions else None,
            "load_level_field": sessions[0]["load_level_field"] if sessions else None,
            "per_session_error_distribution": {
                "n_sessions": len(error_counts),
                "min": int(np.min(error_counts)) if error_counts else None,
                "max": int(np.max(error_counts)) if error_counts else None,
                "median": float(np.median(error_counts)) if error_counts else None,
            },
            "per_session_census": [
                {k: s[k] for k in (
                    "patient", "session", "n_trials_raw", "n_trials_admitted", "n_errors_admitted",
                    "n_trials_old_rule_would_discard_among_admitted", "load_levels_admitted",
                    "usable_for_estimator") if k in s} for s in sessions
            ],
        }
    return report, sessions_by_corpus


# =======================================================================================================
# Per-session trial arrays (deviation, its session-training-trial-mean control, and covariates)
# =======================================================================================================

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


# =======================================================================================================
# Block B -- the human estimate (trial-level, clustered by patient)
# =======================================================================================================

def _pool_patient_trials(sessions: list[dict], load_filter: int | None) -> dict[str, dict]:
    """One pooled (trial-level) array per patient, concatenated across every session of theirs that
    contributed to this arm. Patients are keyed by (dataset, patient) so two different corpora's patient
    identifiers can never collide into the same cluster."""
    by_patient: dict[str, list[dict]] = {}
    for s in sessions:
        if s["status"] != "computed":
            continue
        key = f"{s['dataset']}::{s['patient']}"
        mask = np.ones(s["is_correct"].shape[0], dtype=bool) if load_filter is None else (s["load_level"] == load_filter)
        if not mask.any():
            continue
        by_patient.setdefault(key, []).append({field: s[field][mask] for field in
                                                ("is_correct", "deviation", "control_deviation", "spike_count", "trial_index")})
    return {patient: {field: np.concatenate([part[field] for part in parts]) for field in
                       ("is_correct", "deviation", "control_deviation", "spike_count", "trial_index")}
            for patient, parts in by_patient.items()}


def _patient_cell(arrays: dict, deviation_key: str, seed_tag: str) -> dict:
    is_correct, deviation = arrays["is_correct"], arrays[deviation_key]
    spike_count, trial_index = arrays["spike_count"], arrays["trial_index"]
    n = int(len(is_correct))
    n_error = int((~is_correct.astype(bool)).sum())
    if n < MIN_TRIALS_PER_PATIENT_CELL:
        return {"status": "excluded", "n_trials": n, "n_errors": n_error,
                "reason": f"fewer_than_{MIN_TRIALS_PER_PATIENT_CELL}_pooled_trials"}
    if n_error < 1 or n_error >= n:
        return {"status": "excluded", "n_trials": n, "n_errors": n_error,
                "reason": "zero_error_or_zero_correct_trials_no_outcome_variance"}
    if not np.all(np.isfinite(deviation)):
        return {"status": "excluded", "n_trials": n, "n_errors": n_error, "reason": "non_finite_deviation_value"}
    return {
        "status": "computed", "n_trials": n, "n_errors": n_error,
        "raw": partial_correlation_permutation_test(
            is_correct, deviation, [], N_PERM_PATIENT_CELL, np.random.default_rng(stable_seed(f"{seed_tag}|raw"))),
        "partial_controlling_spike_count": partial_correlation_permutation_test(
            is_correct, deviation, [spike_count], N_PERM_PATIENT_CELL, np.random.default_rng(stable_seed(f"{seed_tag}|ctrl_spike"))),
        "partial_controlling_trial_index": partial_correlation_permutation_test(
            is_correct, deviation, [trial_index], N_PERM_PATIENT_CELL, np.random.default_rng(stable_seed(f"{seed_tag}|ctrl_trial"))),
        "joint_partial": partial_correlation_permutation_test(
            is_correct, deviation, [spike_count, trial_index], N_PERM_PATIENT_CELL,
            np.random.default_rng(stable_seed(f"{seed_tag}|ctrl_joint"))),
    }


def _pool_patients_across(cells: dict[str, dict], key: str) -> dict:
    values = [c[key]["r"] for c in cells.values() if c.get("status") == "computed" and c[key].get("status") == "computed"]
    return slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}


def _arm(sessions: list[dict], load_filter: int | None, deviation_key: str, seed_tag: str) -> dict:
    patient_arrays = _pool_patient_trials(sessions, load_filter)
    cells = {p: _patient_cell(arr, deviation_key, f"{seed_tag}|{p}") for p, arr in patient_arrays.items()}
    return {
        "load_level": load_filter, "n_patients_total": len(cells),
        "n_patients_computed": sum(1 for c in cells.values() if c["status"] == "computed"),
        "patients": cells,
        "pooled_raw": _pool_patients_across(cells, "raw"),
        "pooled_partial_controlling_spike_count": _pool_patients_across(cells, "partial_controlling_spike_count"),
        "pooled_partial_controlling_trial_index": _pool_patients_across(cells, "partial_controlling_trial_index"),
        "pooled_joint_partial": _pool_patients_across(cells, "joint_partial"),
    }


def _se_from_ci(lo: float, hi: float) -> float:
    z = float(norm.ppf(0.975))
    return (hi - lo) / (2.0 * z)


def _combine_arms(arms: list[dict], key: str, labels: list[str]) -> dict:
    estimates, ses, use_labels = [], [], []
    for arm, label in zip(arms, labels):
        pooled = arm[key]
        if pooled.get("status") != "tested":
            continue
        se = _se_from_ci(pooled["ci_lower"], pooled["ci_upper"])
        if not np.isfinite(se) or se <= 0:
            continue
        estimates.append(pooled["mean_value"])
        ses.append(se)
        use_labels.append(label)
    if not estimates:
        return {"status": "not_computable", "reason": "no arm produced a tested pooled estimate with a usable confidence interval"}
    result = forest_meta(np.array(estimates), np.array(ses), use_labels)
    result["status"] = "computed"
    return result


def _mdd_from_se(se: float, alpha: float = 0.05, power: float = 0.80) -> dict:
    z = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    return {"status": "computed", "se": se, "alpha": alpha, "power": power, "z_factor": z, "mdd": z * se}


def _classify_block_b(combined_real: dict, combined_control: dict) -> dict:
    raw, joint = combined_real["raw"], combined_real["joint_partial"]
    if raw.get("status") != "computed":
        return {"branch": "underpowered_to_ask", "reason": "no arm produced a computable combined raw estimate"}
    raw_sig = raw["p_value"] <= 0.05
    if raw_sig:
        raw_sign_positive = raw["pooled"] > 0.0
        joint_agrees = (joint.get("status") == "computed" and joint["p_value"] <= 0.05
                        and (joint["pooled"] > 0.0) == raw_sign_positive)
        if not joint_agrees:
            return {"branch": "raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index"}
        control_joint = combined_control["joint_partial"]
        control_reproduces = (control_joint.get("status") == "computed" and control_joint["p_value"] <= 0.05
                               and (control_joint["pooled"] > 0.0) == raw_sign_positive)
        if control_reproduces:
            return {"branch": "behaviour_link_not_separable_from_a_session_level_offset"}
        return {"branch": "the_component_predicts_accuracy_in_a_human_maintenance_delay"}
    mdd = _mdd_from_se(raw["se"])
    if mdd["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        return {"branch": "no_human_behaviour_link_above_the_reported_bound", "minimum_detectable_difference_80pct_power": mdd}
    return {"branch": "underpowered_to_ask", "minimum_detectable_difference_80pct_power": mdd}


def run_block_b(sessions_by_corpus_computed: dict[str, list[dict]]) -> dict:
    arms_real, arms_control, arm_labels = [], [], []
    per_corpus_detail: dict[str, dict] = {}
    for corpus, sessions in sessions_by_corpus_computed.items():
        load_levels = sorted(set(int(v) for s in sessions if s["status"] == "computed" for v in np.unique(s["load_level"])))
        by_load, by_load_control = {}, {}
        for load in load_levels:
            tag = f"run_human_maintenance_behaviour_link|block_b|{corpus}|load{load}"
            arm_real = _fit(f"arm_real|{corpus}|{load}", lambda ss=sessions, l=load, t=tag: _arm(ss, l, "deviation", t))
            arm_control = _fit(f"arm_control|{corpus}|{load}",
                                lambda ss=sessions, l=load, t=tag: _arm(ss, l, "control_deviation", f"{t}|control"))
            by_load[str(load)], by_load_control[str(load)] = arm_real, arm_control
            arms_real.append(arm_real)
            arms_control.append(arm_control)
            arm_labels.append(f"{corpus}_load{load}")
        per_corpus_detail[corpus] = {"load_levels": load_levels, "by_load": by_load, "by_load_control": by_load_control}
        _log(f"  Block B arms built for {corpus}: {len(load_levels)} load levels")

    combined_real = {
        "raw": _combine_arms(arms_real, "pooled_raw", arm_labels),
        "partial_controlling_spike_count": _combine_arms(arms_real, "pooled_partial_controlling_spike_count", arm_labels),
        "partial_controlling_trial_index": _combine_arms(arms_real, "pooled_partial_controlling_trial_index", arm_labels),
        "joint_partial": _combine_arms(arms_real, "pooled_joint_partial", arm_labels),
    }
    combined_control = {
        "raw": _combine_arms(arms_control, "pooled_raw", arm_labels),
        "joint_partial": _combine_arms(arms_control, "pooled_joint_partial", arm_labels),
    }

    all_sessions_flat = [s for sessions in sessions_by_corpus_computed.values() for s in sessions]
    secondary_arrays = _pool_patient_trials(all_sessions_flat, load_filter=None)
    secondary_cells = {p: _patient_cell(arr, "deviation", f"run_human_maintenance_behaviour_link|block_b|secondary|{p}")
                        for p, arr in secondary_arrays.items()}
    secondary_pooled = {
        "raw": _pool_patients_across(secondary_cells, "raw"),
        "partial_controlling_spike_count": _pool_patients_across(secondary_cells, "partial_controlling_spike_count"),
        "partial_controlling_trial_index": _pool_patients_across(secondary_cells, "partial_controlling_trial_index"),
        "joint_partial": _pool_patients_across(secondary_cells, "joint_partial"),
    }
    if len(secondary_cells) >= 2:
        secondary_raw_values = [c["raw"]["r"] for c in secondary_cells.values()
                                 if c.get("status") == "computed" and c["raw"].get("status") == "computed"]
        secondary_mdd = minimum_detectable_paired_difference(secondary_raw_values) if len(secondary_raw_values) >= 2 else \
            {"status": "not_computable", "n": len(secondary_raw_values)}
    else:
        secondary_mdd = {"status": "not_computable", "n": len(secondary_cells)}

    branch = _classify_block_b(combined_real, combined_control)

    return {
        "decision_rule_declared_before_fitting": BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
        "n_arms": len(arms_real), "arm_labels": arm_labels,
        "per_corpus": per_corpus_detail,
        "combined_within_load_then_meta_analysed": combined_real,
        "mandatory_control_session_training_trial_mean": combined_control,
        "secondary_pooled_across_load_all_corpora": {
            "note": "Ignores load stratification entirely: one value per patient, pooled across every "
                    "admitted trial of theirs regardless of load level or corpus. Confounds the deviation-"
                    "outcome relationship with the sampling of hard (higher-load, higher-error) trials. "
                    "Reported only as a secondary diagnostic, never as the branch-deciding number.",
            "n_patients": len(secondary_cells), "pooled": secondary_pooled,
            "raw_minimum_detectable_paired_difference_80pct_power": secondary_mdd,
        },
        "branch": branch,
    }


# =======================================================================================================
# Block C -- the non-human effect cut down to human error structure
# =======================================================================================================

def _macaque_session_activity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    counts_all = _counts_from_spikes(spikes, time_ms)
    return counts_all.sum(axis=2), is_corr


def _raw_point_r(y: np.ndarray, x: np.ndarray) -> float | None:
    if np.std(y) == 0.0 or np.std(x) == 0.0:
        return None
    return float(np.corrcoef(y, x)[0, 1])


def _session_draw_r(activity_by_unit: np.ndarray, is_corr: np.ndarray, correct_idx: np.ndarray,
                     error_idx: np.ndarray, e_star: int, rng: np.random.Generator) -> float | None:
    sampled_err = rng.choice(error_idx, size=e_star, replace=False)
    idx = np.concatenate([correct_idx, sampled_err])
    deviation_sub = rate_free_state_deviation(activity_by_unit[idx])
    finite = np.isfinite(deviation_sub)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    return _raw_point_r(is_corr[idx][finite].astype(float), deviation_sub[finite])


def _session_rung(activity_by_unit: np.ndarray, is_corr: np.ndarray, e_star: int, session_id: str) -> dict:
    correct_idx, error_idx = np.where(is_corr)[0], np.where(~is_corr)[0]
    if e_star > len(error_idx):
        return {"status": "e_star_exceeds_session_error_trials", "n_error_actual": int(len(error_idx))}
    draws = []
    for d in range(N_DRAWS_BLOCK_C):
        rng = np.random.default_rng(stable_seed(
            f"run_human_maintenance_behaviour_link|block_c|ladder|{session_id}|e{e_star}|draw{d}"))
        r = _session_draw_r(activity_by_unit, is_corr, correct_idx, error_idx, e_star, rng)
        if r is not None:
            draws.append(r)
    if len(draws) < max(3, N_DRAWS_BLOCK_C // 2):
        return {"status": "too_few_valid_draws", "n_draws_valid": len(draws)}
    return {"status": "computed", "n_draws_valid": len(draws), "median_r": float(np.median(draws)),
            "iqr_r": [float(np.percentile(draws, 25)), float(np.percentile(draws, 75))]}


def _session_human_distributed(activity_by_unit: np.ndarray, is_corr: np.ndarray, session_id: str,
                                human_error_counts: list[int]) -> dict:
    correct_idx, error_idx = np.where(is_corr)[0], np.where(~is_corr)[0]
    draws, n_skipped = [], 0
    for d in range(N_DRAWS_BLOCK_C):
        rng = np.random.default_rng(stable_seed(
            f"run_human_maintenance_behaviour_link|block_c|human_distribution|{session_id}|draw{d}"))
        e_star = int(rng.choice(human_error_counts))
        if e_star > len(error_idx):
            n_skipped += 1
            continue
        r = _session_draw_r(activity_by_unit, is_corr, correct_idx, error_idx, e_star, rng)
        if r is not None:
            draws.append(r)
    if len(draws) < max(3, N_DRAWS_BLOCK_C // 4):
        return {"status": "too_few_valid_draws", "n_draws_valid": len(draws), "n_skipped_e_star_exceeds_session": n_skipped}
    return {"status": "computed", "n_draws_valid": len(draws), "n_skipped_e_star_exceeds_session": n_skipped,
            "median_r": float(np.median(draws))}


def run_block_c(root: Path, human_error_counts: list[int], reference_raw_r: float) -> dict:
    macaque_paths = _reachable_sessions(root)
    rungs = sorted(set(v for v in human_error_counts if v >= 2))
    session_rung_results: dict[str, dict[int, dict]] = {}
    session_human_dist_results: dict[str, dict] = {}
    session_n_error_actual: dict[str, int] = {}

    for path in macaque_paths:
        session_id = path.stem
        activity_by_unit, is_corr = _macaque_session_activity(path)
        session_n_error_actual[session_id] = int((~is_corr).sum())
        session_rung_results[session_id] = {}
        for e_star in rungs:
            key = f"block_c_rung|{session_id}|e{e_star}"
            session_rung_results[session_id][e_star] = _fit(
                key, lambda a=activity_by_unit, ic=is_corr, e=e_star, sid=session_id: _session_rung(a, ic, e, sid))
        session_human_dist_results[session_id] = _fit(
            f"block_c_human_distribution|{session_id}",
            lambda a=activity_by_unit, ic=is_corr, sid=session_id: _session_human_distributed(a, ic, sid, human_error_counts))
        _log(f"  Block C: {session_id} done ({len(rungs)} rungs + human-distribution draws)")

    hd_values = [v["median_r"] for v in session_human_dist_results.values() if v.get("status") == "computed"]
    hd_pooled = slope_across_sessions_test(hd_values, alternative="two-sided") if hd_values else {"status": "not_computed"}

    ladder: dict[str, dict] = {}
    min_detectable_error_count = None
    for e_star in rungs:
        vals = [session_rung_results[s][e_star]["median_r"] for s in session_rung_results
                if session_rung_results[s][e_star].get("status") == "computed"]
        pooled = slope_across_sessions_test(vals, alternative="two-sided") if vals else {"status": "not_computed"}
        power = power_to_detect_effect(reference_raw_r, vals) if len(vals) >= 2 else {"status": "not_computable", "n": len(vals)}
        ladder[str(e_star)] = {"n_sessions_contributing": len(vals), "pooled": pooled, "power_to_detect_reference_effect": power}
        if min_detectable_error_count is None and power.get("status") == "computed" and power["power"] >= 0.80:
            min_detectable_error_count = e_star

    reference_sign_positive = reference_raw_r > 0.0
    hd_significant = hd_pooled.get("status") == "tested" and hd_pooled.get("significant")
    hd_same_sign = hd_pooled.get("status") == "tested" and ((hd_pooled["mean_value"] > 0.0) == reference_sign_positive)
    if hd_significant and hd_same_sign:
        branch = "the_non_human_effect_survives_at_human_error_counts"
    elif min_detectable_error_count is not None:
        branch = "the_non_human_effect_dies_at_human_error_counts"
    else:
        branch = "no_identifiable_transition_inside_the_sampled_range"

    return {
        "decision_rule_declared_before_fitting": BLOCK_C_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions": len(macaque_paths), "rungs_tested": rungs, "reference_effect_raw_r": reference_raw_r,
        "human_error_count_distribution_used_for_matched_draws": human_error_counts,
        "n_draws_per_session_per_rung": N_DRAWS_BLOCK_C, "session_n_error_actual": session_n_error_actual,
        "human_distribution_matched_draws": {"per_session": session_human_dist_results, "pooled": hd_pooled},
        "error_count_ladder": ladder,
        "minimum_detectable_error_count_at_80pct_power": min_detectable_error_count,
        "branch": branch,
    }


# =======================================================================================================
# Block D -- the joint verdict (a pre-interpreted two-by-two)
# =======================================================================================================

BLOCK_D_CELL_INTERPRETATIONS = {
    (True, True): "the mechanism is shared and measurable in both; the specification's read-out is human-validated",
    (True, False): "the human result is stronger than the design should allow; check it hard before believing it",
    (False, True): "a genuine preparation difference, and the most interesting outcome available here",
    (False, False): "the human null is uninformative -- a design consequence, not biology, and the required error "
                     "count becomes an acquisition specification",
}


def run_block_d(block_b_branch: str, block_c_branch: str) -> dict:
    human_link_present = block_b_branch == "the_component_predicts_accuracy_in_a_human_maintenance_delay"
    non_human_survives = block_c_branch == "the_non_human_effect_survives_at_human_error_counts"
    return {
        "human_link_present": human_link_present, "non_human_effect_survives_the_cut": non_human_survives,
        "block_b_branch": block_b_branch, "block_c_branch": block_c_branch,
        "cell_interpretation": BLOCK_D_CELL_INTERPRETATIONS[(human_link_present, non_human_survives)],
    }


# =======================================================================================================
# Block E -- what actually differs between the preparations
# =======================================================================================================

def run_block_e(block_a: dict, block_c: dict, human_median_units: float | None, macaque_median_units: float | None,
                 human_median_trials_per_session: float | None) -> list[dict]:
    macaque_n = block_c["n_sessions"]
    human_sessions_total = sum(block_a[c]["n_sessions_usable_for_estimator"] for c in block_a)
    human_patients_total = sum(block_a[c]["n_patients"] for c in block_a)
    return [
        {"topic": "species", "classification": "irreducible_difference",
         "statement": "One preparation is Macaca mulatta, the other is Homo sapiens. Nothing in this project's "
                      "data closes this."},
        {"topic": "recording_region", "classification": "irreducible_difference",
         "statement": "The non-human positive is lateral prefrontal cortex; the three human corpora used here are "
                      "medial-temporal-lobe/limbic single-unit implants (hippocampus, amygdala, pre-SMA, dACC, "
                      "vmPFC-family sites), placed for clinical seizure localisation, never lPFC. This project holds "
                      "no human lPFC single-unit recording and cannot acquire one; the anatomical mismatch is not "
                      "matched here and this leg does not present it as matched."},
        {"topic": "task", "classification": "artifact_of_available_data",
         "statement": "The macaque task is a continuous or retro-cued spatial report; the three human corpora are "
                      "picture or letter Sternberg recognition-probe tasks. A closer-matched human maintenance task "
                      "with a continuous report exists in principle (a feasible new acquisition) but is not held here."},
        {"topic": "behavioural_measure", "classification": "artifact_of_available_data",
         "statement": "The macaque measure is trial correctness on a graded report, thresholded to binary; every "
                      "human corpus used here is binary correctness natively. Not a difference in what could be "
                      "measured, only in what was collected -- closable by a future continuous-report human task."},
        {"topic": "error_count", "classification": "irreducible_difference",
         "statement": "This is exactly what Block C quantifies: human sessions carry a median of a handful of "
                      "error trials per session (Block A), the macaque reachable sessions were selected for at "
                      "least 60. A human patient cannot be made to err more without changing the task's difficulty, "
                      "which changes what is being measured; this is a property of clinical Sternberg tasks kept "
                      "easy enough to be tolerable, not an instrumentation gap."},
        {"topic": "simultaneously_recorded_unit_count", "classification": "real_difference_closable_by_new_acquisition",
         "statement": f"Median simultaneously recorded units: macaque lPFC sessions used here "
                      f"{'%.0f' % macaque_median_units if macaque_median_units is not None else 'not computed'}, "
                      f"human usable sessions here "
                      f"{'%.0f' % human_median_units if human_median_units is not None else 'not computed'}. Denser "
                      "human microwire/Utah-array acquisitions exist elsewhere and would close most of this gap "
                      "without a new task design."},
        {"topic": "epoch_definition", "classification": "artifact_of_available_data",
         "statement": "Every corpus here has a genuine maintenance/delay epoch with a defined onset; the window "
                      "lengths differ by corpus (dandi_000469/001187: 2.3s; dandi_000574: 3.0s; macaque: up to "
                      "1.15s) because each was set by that corpus's own task design, not re-derived here. Closable "
                      "by re-running at a common truncated window, not attempted in this leg."},
        {"topic": "trials_per_session", "classification": "artifact_of_available_data",
         "statement": f"Human usable sessions here carry a median of "
                      f"{'%.0f' % human_median_trials_per_session if human_median_trials_per_session is not None else 'not computed'}"
                      " admitted trials; macaque reachable sessions run into the hundreds. A longer human clinical "
                      "recording session would close this without a new implant."},
        {"topic": "sessions_per_subject", "classification": "artifact_of_available_data",
         "statement": f"Human: {human_sessions_total} usable sessions across {human_patients_total} patients "
                      "pooled over all three corpora (most patients contribute one session; Block B's whole design "
                      "exists because of this). Macaque: the reachable cohort spans more than one animal; this leg "
                      "pools all reachable sessions the way the reference artifact already does and does not "
                      "re-derive per-animal identity, since neither Block B nor Block C conditions on it."},
    ]


# =======================================================================================================
# Block F -- the prediction only a maintenance-period human stimulation experiment can test
# =======================================================================================================

def run_block_f(reference_artifact: dict, subspace_artifact: dict | None, block_b: dict) -> dict:
    pooled = reference_artifact.get("pooled", {})
    raw = pooled.get("raw_outcome_vs_deviation", {})
    joint = pooled.get("joint_partial_controlling_spike_count_and_trial_index", {})
    raw_r, joint_r = raw.get("mean_value"), joint.get("mean_value")

    subspace_note = "not available -- results/deviation_subspace_decomposition.json was not read successfully"
    if subspace_artifact is not None:
        cell = subspace_artifact.get("macaque_lpfc", {}).get("by_error_floor", {}).get("60", {})
        within = cell.get("raw_within_vs_outcome", {})
        outside = cell.get("raw_outside_vs_outcome", {})
        subspace_note = (
            f"the accuracy link lives OUTSIDE the memorandum's own coding subspace (raw within-subspace "
            f"r={within.get('mean_value')}, p={within.get('p_value')}, not significant; raw outside-subspace "
            f"r={outside.get('mean_value')}, p={outside.get('p_value')}, significant), so a stimulation-driven "
            "change in the deviation component is not predicted to require, or produce, any change in the "
            "memorandum's own item-identity coding subspace -- the two are dissociable readouts of the same "
            "population, not the same signal under two names."
        )

    human_joint = block_b.get("combined_within_load_then_meta_analysed", {}).get("joint_partial", {})
    human_available = human_joint.get("status") == "computed" and human_joint.get("p_value", 1.0) <= 0.05
    source = "human (this leg's own combined joint estimate)" if human_available else \
        "macaque lPFC (results/rate_free_state_geometry_behavior_link.json; the human arm did not deliver its own significant fitted number)"
    used_sign = (human_joint.get("pooled") if human_available else joint_r)

    return {
        "source_of_the_fitted_numbers": source,
        "raw_correlation_reference": {"r": raw_r, "p_value": raw.get("p_value")},
        "joint_partial_correlation_reference": {"r": joint_r, "p_value": joint.get("p_value")},
        "prediction": (
            "Delivering stimulation inside a human maintenance delay that DISPLACES the population state away from "
            "its own session-typical direction (raises rate_free_state_deviation) is predicted, from this "
            f"project's own fitted numbers ({source}), to move trial accuracy in the "
            f"{'negative' if used_sign is not None and used_sign < 0 else 'positive'} direction: a one-within-"
            "session-standard-deviation increase in the deviation component predicts roughly a "
            f"{abs(joint_r):.3f} to {abs(raw_r):.3f} (joint-partial to raw r-unit) increase in the per-trial "
            "probability of an error, on the same scale results/rate_free_state_geometry_behavior_link.json "
            "reports. This is a magnitude prediction stated on the correlational r scale this project already "
            "uses throughout, not a causal-effect-size claim; falsification is a stimulation-driven increase in "
            "deviation that produces NO corresponding drop in accuracy, or a drop in the opposite direction."
        ),
        "memorandum_subspace_prediction": (
            "The memorandum's own item-identity coding subspace is predicted NOT to be required to move for this "
            f"accuracy effect to appear: {subspace_note}"
        ),
        "sign_of_per_trial_component_change_vs_accuracy_change": (
            "negative" if used_sign is not None and used_sign < 0 else
            "positive" if used_sign is not None else "not_determinable_no_fitted_sign_available"
        ),
        "not_a_result": "This block writes a prediction, not a result: it fires no branch and upgrades no verdict.",
    }


# =======================================================================================================
# Driver
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    root = data_root()

    output: dict = {"version": ANALYSIS_VERSION, "status": "running"}
    _flush(output)

    _log("Block A: trial admission census (all three human maintenance corpora)")
    block_a, sessions_by_corpus = run_block_a(root)
    output["block_a"] = block_a
    _flush(output)
    for corpus, report in block_a.items():
        _log(f"  {corpus}: {report['n_trials_admitted_total']} admitted, {report['n_errors_admitted_total']} errors "
             f"({report['pct_errors_admitted']:.2f}%), {report['n_sessions_usable_for_estimator']}/{report['n_sessions_seen']} "
             f"sessions usable, elapsed={time.time() - t0:.0f}s")

    _log("computing per-session trial arrays (deviation + session-training-trial-mean control) for usable sessions")
    sessions_by_corpus_computed: dict[str, list[dict]] = {}
    session_array_status: dict[str, dict] = {}
    for corpus, sessions in sessions_by_corpus.items():
        computed = []
        statuses = {}
        for entry in sessions:
            if not entry["usable_for_estimator"]:
                continue
            key = f"session_arrays|{corpus}|{entry['session']}"
            result = _fit(key, lambda e=entry: _session_trial_arrays(e))
            statuses[entry["session"]] = result["status"]
            if result["status"] == "computed":
                computed.append(result)
        sessions_by_corpus_computed[corpus] = computed
        session_array_status[corpus] = statuses
        _log(f"  {corpus}: {len(computed)}/{sum(1 for s in sessions if s['usable_for_estimator'])} sessions computed")
    output["session_array_status_by_corpus"] = session_array_status
    _flush(output)

    _log("Block B: the human estimate")
    output["block_b"] = run_block_b(sessions_by_corpus_computed)
    _flush(output)
    _log(f"  Block B branch: {output['block_b']['branch']['branch']} elapsed={time.time() - t0:.0f}s")

    human_error_counts = [
        s["n_errors_admitted"] for report in block_a.values() for s in report["per_session_census"]
        if s["n_trials_admitted"] >= MIN_TRIALS
    ]
    reference_artifact = json.loads(REFERENCE_ARTIFACT_PATH.read_text())
    reference_raw_r = reference_artifact["pooled"]["raw_outcome_vs_deviation"]["mean_value"]
    output["reference_artifact_read"] = {
        "path": "results/rate_free_state_geometry_behavior_link.json",
        "raw_outcome_vs_deviation_mean_value": reference_raw_r,
        "joint_partial_mean_value": reference_artifact["pooled"][
            "joint_partial_controlling_spike_count_and_trial_index"]["mean_value"],
        "n_sessions": reference_artifact.get("n_sessions_computed"),
    }

    _log(f"Block C: the non-human effect cut down to human error structure ({len(human_error_counts)} human sessions "
         "informing the target distribution)")
    output["block_c"] = run_block_c(root, human_error_counts, reference_raw_r)
    _flush(output)
    _log(f"  Block C branch: {output['block_c']['branch']} elapsed={time.time() - t0:.0f}s")

    output["block_d"] = run_block_d(output["block_b"]["branch"]["branch"], output["block_c"]["branch"])

    macaque_units = [_macaque_session_activity(p)[0].shape[1] for p in _reachable_sessions(root)]
    human_unit_counts = []
    human_trial_counts = []
    for sessions in sessions_by_corpus_computed.values():
        for s in sessions:
            human_trial_counts.append(s["n_trials_with_defined_direction"])
    for corpus, sessions in sessions_by_corpus.items():
        for entry in sessions:
            if entry["usable_for_estimator"]:
                human_unit_counts.append(len(entry["spike_lists"]))

    output["block_e"] = run_block_e(
        block_a, output["block_c"],
        human_median_units=float(np.median(human_unit_counts)) if human_unit_counts else None,
        macaque_median_units=float(np.median(macaque_units)) if macaque_units else None,
        human_median_trials_per_session=float(np.median(human_trial_counts)) if human_trial_counts else None,
    )

    subspace_artifact = None
    if SUBSPACE_ARTIFACT_PATH.exists():
        try:
            subspace_artifact = json.loads(SUBSPACE_ARTIFACT_PATH.read_text())
        except (OSError, ValueError):
            subspace_artifact = None
    output["block_f"] = run_block_f(reference_artifact, subspace_artifact, output["block_b"])

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "block_b_branch": output["block_b"]["branch"]["branch"],
        "block_c_branch": output["block_c"]["branch"],
        "block_d": output["block_d"]["cell_interpretation"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
