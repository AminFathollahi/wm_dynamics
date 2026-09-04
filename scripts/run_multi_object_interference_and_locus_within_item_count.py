"""run_multi_object_interference_and_locus_within_item_count.py -- interference
and temporal-locus tests for the multi-object macaque corpus's rate-free
direction-deviation observable, formed WITHIN item-count level rather than
pooled across it.

Why within item-count level. This corpus's own primary estimator for the
deviation-to-behaviour link -- established by results/watters_load_
decomposition.json and reproduced exactly by results/dissociation_
replication_and_counting_noise.json -- forms the deviation-to-report-error
correlation SEPARATELY inside each item-count level and combines the
per-level coefficients by a trial-count-weighted average, because the
pooled-across-item-count correlation mixes a within-level relationship with
a between-level one (item count and the deviation both vary with load).
results/deviation_serial_dependence_and_temporal_locus.json's own two blocks
were run on the POOLED-across-item-count version of this corpus's deviation
and report error, which carries a NEGATIVE, non-significant association
(read live below), the opposite sign of the corpus's own established
POSITIVE, significant within-item-count-level association. Its interference
and temporal-locus verdicts for this corpus were therefore built on an
untested observable, not the one that actually carries this corpus's
finding. This module repeats both questions on the within-item-count-level
observable and reports the pooled-across-item-count figures beside it so
the reversal is visible rather than resolved silently.

BLOCK A asks whether the within-item-count-level deviation-to-behaviour link
is inter-trial interference: a trial's direction pulled toward the trial
immediately before it. The adjacency statistic that answers "is there
lag-1-specific structure in this corpus's trial sequence at all" is a
property of the trial order alone, not of the behavioural variable, so it is
read live from results/deviation_serial_dependence_and_temporal_locus.json
rather than recomputed. What is new here is the decisive partial: the
deviation-to-behaviour correlation and its two partials (controlling the
trial's own detrended lag-1 alignment, and jointly controlling that
alignment with total spike count and trial index), each formed within
item-count level and combined by the same trial-count-weighted average the
corpus's primary estimator already uses, then pooled across sessions.

BLOCK B asks where in the delay epoch the within-item-count-level link
lives, by splitting the delay into halves (primary) and thirds (sensitivity)
and recomputing the deviation, its own orthogonality gate against total
spike count, and its behavioural association separately in each sub-window
AND within item-count level inside each sub-window.

Scope: the multi-object macaque corpus only. No other corpus is touched, and
nothing already delivered under results/ is modified, re-run or re-labelled.

SIGN CONVENTION. This corpus's behavioural variable is continuous report
error, higher is worse. No sign flip is applied anywhere in this module.

No estimator is forked. rate_free_state_deviation, _observable_arrays,
_session_observable_arm, _pool_cell, partial_correlation_permutation_test,
slope_across_sessions_test, minimum_detectable_paired_difference,
stable_seed, iter_watters, data_root, unit_direction_vectors, _detrend,
_cosine_at_lag, _sub_window_bins and the detrending-window / sub-window-split
constants are every one imported unchanged from where this project already
defines them. The only new functions this module introduces are the
within-item-count-level decisive-partial computation Block A needs and the
within-item-count-level sub-window recomputation Block B needs, together
with the trial-count-weighted combination formula _session_observable_arm
already applies to its own stat family, applied here to the decisive-partial
coefficients that estimator does not itself compute.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    DETREND_WINDOWS_TRIALS, MIN_TRIALS_FOR_LAG_PROFILE, PRIMARY_SPLIT, SUB_WINDOW_SPLITS,
    _cosine_at_lag, _detrend, _sub_window_bins, unit_direction_vectors,
)
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    _observable_arrays, _pool_cell, _session_observable_arm,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_watters_state_geometry import MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)
from state_persistence import slope_across_sessions_test  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "multi_object_interference_and_locus_within_item_count.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "multi_object_interference_and_locus_within_item_count_checkpoint.json"
ANALYSIS_VERSION = "2026-08-19"

N_PERM = 10000
REPRODUCTION_TOLERANCE = 1e-6

DISSOCIATION_ARTIFACT_PATH = ROOT / "results" / "dissociation_replication_and_counting_noise.json"
SERIAL_DEPENDENCE_ARTIFACT_PATH = ROOT / "results" / "deviation_serial_dependence_and_temporal_locus.json"

PRIMARY_WINDOW = DETREND_WINDOWS_TRIALS[0]

BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Read live from results/deviation_serial_dependence_and_temporal_locus.json (not recomputed here, "
    "because it is a property of the trial sequence alone): this corpus's pooled detrended lag-1-versus-"
    "background-lag adjacency statistic at the primary detrending window is positive and significant "
    "against a within-session trial-order shuffle null, so adjacency is PRESENT by that already-computed "
    "value. Only two cells are therefore reachable at the primary window:\n"
    "  - the within-item-count-level joint partial (controlling the trial's own detrended lag-1 alignment "
    "together with total spike count and trial index), pooled across sessions, is significant two-sided at "
    "0.05 with the same sign as the pooled within-item-count-level raw correlation -> "
    "'interference_from_the_preceding_trial_is_present_and_separable_from_the_accuracy_predicting_component', "
    "reported with the raw and both partials side by side.\n"
    "  - otherwise, if the pooled within-item-count-level raw correlation is itself significant with the "
    "corpus's own established worse-behaviour-positive sign (read live from the reproduction reference) AND "
    "the joint partial's magnitude is smaller than the raw correlation's magnitude (shrinks toward zero "
    "rather than grows) -> 'accuracy_predicting_component_is_interference_from_the_preceding_trial', "
    "reported with the raw and joint partial coefficients side by side.\n"
    "  - if either of those two conditions fails -> "
    "'within_level_link_not_present_at_full_epoch_so_interference_cannot_be_adjudicated', reported with the "
    "raw, both partials and the minimum detectable paired difference at 80% power beside it. This cell "
    "exists because a prior analysis fired an interference label on exactly this configuration -- a raw "
    "link that was already absent, with partialling that increased rather than shrank the coefficient -- "
    "and the label was false.\n"
    "The verdict is computed independently at both detrending windows; if they disagree, both are reported "
    "and the disagreement is stated, never resolved by picking one."
)

BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "The delay epoch is split into halves (primary) and thirds (sensitivity). In each sub-window, the "
    "deviation, its own orthogonality gate against total spike count, and its raw association with "
    "behaviour are recomputed from scratch, each formed within item-count level and combined by the same "
    "trial-count-weighted average the corpus's primary estimator uses, then pooled across sessions. A "
    "sub-window whose own pooled gate is significant at the two-sided 0.05 level is VOID and excluded from "
    "every comparison; it is never compared against a surviving sub-window.\n"
    "  - every sub-window void -> 'temporal_locus_unreachable_at_this_count_per_window', with the counts "
    "and gates that made it so.\n"
    "  - behaviour link present (pooled within-item-count-level raw association significant) in every "
    "surviving sub-window, and the direct paired difference between the earliest and latest surviving "
    "sub-window is not significant -> 'accuracy_predicting_component_is_present_throughout_the_delay'.\n"
    "  - present in every surviving sub-window, and the direct paired difference is significant and larger "
    "in the latest surviving sub-window -> 'accuracy_predicting_component_grows_across_the_delay'.\n"
    "  - present in every surviving sub-window, and the direct paired difference is significant and larger "
    "in the earliest surviving sub-window -> 'accuracy_predicting_component_is_strongest_at_delay_onset'.\n"
    "  - absent in every surviving sub-window while present over the full epoch (per the reproduction "
    "reference's own raw correlation) -> 'accuracy_predicting_component_requires_the_full_epoch_to_be_"
    "detected', a statement about detection, not about the delay's structure.\n"
    "An ordering claim requires the earliest-vs-latest difference to be tested directly, paired within "
    "session, never inferred from two separate significance verdicts. Any other reachable combination "
    "(present in some but not all surviving sub-windows) is not named by this rule and is reported as "
    "'block_b_outcome_not_covered_by_the_pre_declared_rule' with every sub-window's own numbers, not forced "
    "onto a listed branch. The halves split alone decides the branch; the thirds split is reported beside "
    "it as a sensitivity check and does not override the halves verdict."
)


# =======================================================================================================
# Checkpointing (fit-level; temp file + os.replace; completion flag written only after the fit returns).
# Only genuinely expensive statistical fits are checkpointed here -- small dicts of floats and permutation-
# test results, never a raw spike tensor or session record -- so a rerun's checkpoint stays proportioned
# to the number of fits, not to the size of the corpus.
# =======================================================================================================

_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    try:
        entries = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: {**entry, "value": restore_checkpoint(entry["value"])} for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("complete") is True}


def _fit(key: str, compute) -> dict:
    global _FITS_SERVED_FROM_CHECKPOINT, _FITS_COMPUTED_HERE
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        _FITS_SERVED_FROM_CHECKPOINT += 1
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    _FITS_COMPUTED_HERE += 1
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(checkpoint_safe(_COMPLETED_FITS), allow_nan=False, default=float))
    os.replace(scratch, CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Reference numbers read live from disk -- never a literal copied from anywhere else into this module.
# =======================================================================================================

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _live_reproduction_reference() -> dict:
    """This corpus's already-delivered within-item-count-level deviation cell (single_and_multi_unit
    tier, pooled group), read straight off disk so the reproduction gate below is checked against the
    artifact itself rather than against a number transcribed into this module."""
    delivered = _load_json(DISSOCIATION_ARTIFACT_PATH)
    cell = delivered["block_a"]["results"]["single_and_multi_unit"]["pooled"]["deviation"]["within_item_count_level"]
    return {
        "raw_vs_report_error": {"r": cell["raw_vs_report_error"]["mean_value"], "p": cell["raw_vs_report_error"]["p_value"]},
        "orthogonality_gate_vs_spike_count": {
            "r": cell["orthogonality_gate_vs_spike_count"]["mean_value"], "p": cell["orthogonality_gate_vs_spike_count"]["p_value"]},
        "source_artifact": "results/dissociation_replication_and_counting_noise.json",
        "source_description": "block_a results, single_and_multi_unit tier, pooled group, within-item-count-level deviation cell",
    }


def _live_adjacency_reference() -> dict:
    """This corpus's already-delivered pooled detrended lag-1-versus-background adjacency statistic at
    the primary detrending window, read live rather than recomputed -- it is a property of the trial
    sequence, not of the behavioural variable, so recomputing it would reproduce the same number at the
    cost of a full shuffle-null pass."""
    delivered = _load_json(SERIAL_DEPENDENCE_ARTIFACT_PATH)
    adj = delivered["block_a"]["watters_2026_macaque_multi_object"]["adjacency_by_window"][str(PRIMARY_WINDOW)]
    return {
        "window": PRIMARY_WINDOW,
        "sign_flip_pooled_mean_value": adj["real_pooled"]["mean_value"],
        "sign_flip_pooled_p_value": adj["real_pooled"]["p_value"],
        "shuffle_null_two_sided_empirical_p_value": adj["two_sided_empirical_p_value"],
        "significant": adj["significant"],
        "n_sessions_contributing_real": adj["n_sessions_contributing_real"],
        "source_artifact": "results/deviation_serial_dependence_and_temporal_locus.json",
        "source_description": "block_a, primary detrending window, adjacency_by_window entry for this corpus",
    }


def _live_pooled_across_item_count_reference() -> dict:
    """The pooled-across-item-count deviation-to-report-error correlation an earlier analysis actually
    computed for this corpus (same trial set restricted to a defined detrended lag-1 alignment, same
    primary detrending window, but not split by item-count level), read live so the reversal this module
    exists to test is stated against a number on disk rather than a transcription."""
    delivered = _load_json(SERIAL_DEPENDENCE_ARTIFACT_PATH)
    raw = delivered["block_a"]["watters_2026_macaque_multi_object"]["decisive_partial_by_window"][str(PRIMARY_WINDOW)]["raw"]
    return {
        "mean_value": raw.get("mean_value"), "p_value": raw.get("p_value"), "significant": raw.get("significant"),
        "n_sessions": raw.get("n_sessions"),
        "source_artifact": "results/deviation_serial_dependence_and_temporal_locus.json",
        "source_description": "block_a, primary detrending window, pooled-across-item-count raw deviation-to-report-error correlation",
    }


def _close(observed: float | None, expected: float, tol: float = REPRODUCTION_TOLERANCE) -> bool:
    return observed is not None and abs(observed - expected) <= tol


# =======================================================================================================
# Reproduction gate and session bundles -- one pass, shared by the gate and both blocks
# =======================================================================================================

def _multi_object_session_bundle(session: dict, arrays: dict, usable: np.ndarray) -> dict:
    counts = session["counts"]
    activity_by_unit = counts.sum(axis=2)[usable]
    return {
        "session": session["session"], "counts": counts[usable], "activity_by_unit": activity_by_unit,
        "deviation": arrays["deviation"], "report_error": arrays["report_error"],
        "spike_count": arrays["spike_count"], "trial_index": arrays["trial_index"],
        "reaction_time": arrays["reaction_time"], "item_count": arrays["item_count"],
        "n_trials_total": int(counts.shape[0]), "n_trials_usable": int(usable.sum()),
    }


def _reproduction_rows(loaded: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Builds this corpus's within-item-count deviation arm at the single_and_multi_unit tier, one
    session at a time, from _observable_arrays and _session_observable_arm unchanged. Each session's
    statistical fit is checkpointed individually (a small dict of floats and permutation-test results);
    the session bundles Block A and Block B reuse are kept in memory only, never checkpointed, so a rerun
    never re-serialises a raw spike tensor."""
    rows: list[dict] = []
    bundles: list[dict] = []
    n_arrays_none = 0
    for session in loaded:
        counts = session["counts"]
        arrays, _excluded, usable = _observable_arrays(counts, session)
        if arrays is None:
            n_arrays_none += 1
            rows.append({"by_tier": {}})
            continue
        tag = f"reproduction_arm|{session['session']}"
        arm = _fit(tag, lambda a=arrays, t=tag: _session_observable_arm(a, "deviation", t))
        rows.append({"by_tier": {"single_and_multi_unit": {"status": "computed", "deviation": arm}}})
        bundles.append(_multi_object_session_bundle(session, arrays, usable))
    return rows, bundles, n_arrays_none


def full_reproduction_gate(rows: list[dict]) -> dict:
    reference = _live_reproduction_reference()
    computed = {
        "orthogonality_gate_vs_spike_count": _pool_cell(rows, "single_and_multi_unit", "deviation", "within_load", "orthogonality_gate_vs_spike_count"),
        "raw_vs_report_error": _pool_cell(rows, "single_and_multi_unit", "deviation", "within_load", "raw_vs_report_error"),
    }
    checks = {
        "raw_mean_value": _close(computed["raw_vs_report_error"].get("mean_value"), reference["raw_vs_report_error"]["r"]),
        "raw_p_value": _close(computed["raw_vs_report_error"].get("p_value"), reference["raw_vs_report_error"]["p"]),
        "gate_mean_value": _close(computed["orthogonality_gate_vs_spike_count"].get("mean_value"), reference["orthogonality_gate_vs_spike_count"]["r"]),
        "gate_p_value": _close(computed["orthogonality_gate_vs_spike_count"].get("p_value"), reference["orthogonality_gate_vs_spike_count"]["p"]),
    }
    return {
        "status": "reproduced_exactly" if all(checks.values()) else "not_reproduced",
        "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "computed": computed, "reference_read_live": reference,
    }


# =======================================================================================================
# BLOCK A -- within-item-count-level decisive partial
# =======================================================================================================

def _trial_count_weighted(tested: list[tuple[int, float]]) -> float | None:
    """The identical trial-count-weighted combination _session_observable_arm applies to its own stat
    family (results/watters_load_decomposition.json's within-load estimator), applied here to the
    decisive-partial coefficients that estimator does not itself compute."""
    if not tested:
        return None
    n_values = np.array([n for n, _ in tested], dtype=float)
    r_values = np.array([r for _, r in tested], dtype=float)
    return float(np.sum((n_values / n_values.sum()) * r_values))


def _decisive_partial_within_item_count(bundle: dict, window: int, seed_prefix: str) -> dict:
    vectors = unit_direction_vectors(bundle["activity_by_unit"])
    n = vectors.shape[0]
    if n < MIN_TRIALS_FOR_LAG_PROFILE:
        return {"status": "too_few_trials_for_lag_profile", "n_trials": n}
    detrended = _detrend(vectors, window)
    lag1_align = _cosine_at_lag(detrended, 1)  # index i pairs trial i+1 against trial i
    idx = np.arange(1, n)
    valid = np.isfinite(lag1_align)
    idx, lag1_align = idx[valid], lag1_align[valid]
    if len(idx) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_lag1_alignment", "n_trials": len(idx)}

    report_error = bundle["report_error"][idx]
    deviation = bundle["deviation"][idx]
    spike_count = bundle["spike_count"][idx]
    trial_index = bundle["trial_index"][idx]
    item_count = bundle["item_count"][idx]
    levels = sorted({int(v) for v in item_count.tolist()})

    per_level: dict[str, dict] = {}
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            per_level[str(level)] = {"status": "too_few_trials_at_this_item_count", "n_trials": n_level}
            continue
        tag = f"{seed_prefix}|w{window}|level{level}"
        raw = partial_correlation_permutation_test(
            report_error[mask], deviation[mask], [], N_PERM, np.random.default_rng(stable_seed(f"{tag}|raw")))
        partial_lag1 = partial_correlation_permutation_test(
            report_error[mask], deviation[mask], [lag1_align[mask]], N_PERM,
            np.random.default_rng(stable_seed(f"{tag}|ctrl_lag1")))
        joint = partial_correlation_permutation_test(
            report_error[mask], deviation[mask], [lag1_align[mask], spike_count[mask], trial_index[mask]], N_PERM,
            np.random.default_rng(stable_seed(f"{tag}|ctrl_joint")))
        per_level[str(level)] = {
            "status": "computed", "n_trials": n_level,
            "raw": raw, "partial_controlling_lag1_alignment": partial_lag1,
            "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index": joint,
        }

    n_levels_tested = sum(1 for lv in levels if per_level[str(lv)].get("status") == "computed")
    within: dict[str, float | None] = {}
    for key in ("raw", "partial_controlling_lag1_alignment",
                "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index"):
        tested = [(per_level[str(lv)]["n_trials"], per_level[str(lv)][key]["r"])
                  for lv in levels if per_level[str(lv)].get("status") == "computed"
                  and per_level[str(lv)][key].get("status") == "computed"]
        within[key] = _trial_count_weighted(tested)

    return {
        "status": "computed" if n_levels_tested > 0 else "not_computable_no_item_count_level_reaches_the_floor",
        "n_trials": len(idx), "item_count_levels_present": levels, "n_levels_tested": n_levels_tested,
        "per_level": per_level, "within_item_count_level": within,
    }


def _pool_decisive_partial_within_item_count(per_session: list[dict], key: str) -> dict:
    values = [s["within_item_count_level"][key] for s in per_session
              if s.get("status") == "computed" and s["within_item_count_level"].get(key) is not None]
    pooled = slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}
    if len(values) >= 2:
        pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    return pooled


def _block_a_branch_within_item_count(raw_pooled: dict, joint_pooled: dict, established_positive: bool) -> dict:
    if raw_pooled.get("status") != "tested" or joint_pooled.get("status") != "tested":
        return {"branch": "not_computable", "reason": "raw_or_joint_pooled_result_not_tested"}
    raw_mean, joint_mean = raw_pooled["mean_value"], joint_pooled["mean_value"]
    behaviour_survives = bool(joint_pooled["significant"] and (joint_mean > 0.0) == (raw_mean > 0.0))
    if behaviour_survives:
        return {"branch": "interference_from_the_preceding_trial_is_present_and_separable_from_the_accuracy_predicting_component"}
    raw_significant_with_established_sign = bool(raw_pooled["significant"] and (raw_mean > 0.0) == established_positive)
    coefficient_shrank = bool(abs(joint_mean) < abs(raw_mean))
    if raw_significant_with_established_sign and coefficient_shrank:
        return {"branch": "accuracy_predicting_component_is_interference_from_the_preceding_trial"}
    return {"branch": "within_level_link_not_present_at_full_epoch_so_interference_cannot_be_adjudicated"}


def _aggregate_level_trial_counts(decisive_sessions: list[dict]) -> dict:
    totals: dict[str, int] = {}
    n_sessions_by_level: dict[str, int] = {}
    for d in decisive_sessions:
        if d.get("status") != "computed":
            continue
        for level, entry in d["per_level"].items():
            if entry.get("status") == "computed":
                totals[level] = totals.get(level, 0) + entry["n_trials"]
                n_sessions_by_level[level] = n_sessions_by_level.get(level, 0) + 1
    return {"total_trials_by_item_count_level": totals, "n_sessions_contributing_by_item_count_level": n_sessions_by_level}


def run_block_a(bundles: list[dict], adjacency_reference: dict, reproduction_reference: dict, seed_prefix: str) -> dict:
    decisive: dict[int, list[dict]] = {w: [] for w in DETREND_WINDOWS_TRIALS}
    for bundle in bundles:
        tag = f"{seed_prefix}|{bundle['session']}"
        for window in DETREND_WINDOWS_TRIALS:
            dp = _fit(f"decisive_partial|{bundle['session']}|w{window}",
                       lambda b=bundle, w=window, t=tag: _decisive_partial_within_item_count(b, w, t))
            decisive[window].append(dp)

    established_positive = bool(reproduction_reference["raw_vs_report_error"]["r"] > 0.0)

    decisive_pooled: dict[str, dict] = {}
    branch_by_window: dict[str, str] = {}
    accounting_by_window: dict[str, dict] = {}
    for window in DETREND_WINDOWS_TRIALS:
        raw_pooled = _pool_decisive_partial_within_item_count(decisive[window], "raw")
        partial_lag1_pooled = _pool_decisive_partial_within_item_count(decisive[window], "partial_controlling_lag1_alignment")
        joint_pooled = _pool_decisive_partial_within_item_count(
            decisive[window], "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index")
        decisive_pooled[str(window)] = {
            "raw": raw_pooled, "partial_controlling_lag1_alignment": partial_lag1_pooled,
            "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index": joint_pooled,
            "item_count_level_trial_counts": _aggregate_level_trial_counts(decisive[window]),
        }
        branch_by_window[str(window)] = _block_a_branch_within_item_count(raw_pooled, joint_pooled, established_positive)["branch"]
        accounting_by_window[str(window)] = {
            "n_sessions_total": len(decisive[window]),
            "n_sessions_computed": sum(1 for d in decisive[window] if d.get("status") == "computed"),
            "n_sessions_no_item_count_level_reaches_the_floor": sum(
                1 for d in decisive[window] if d.get("status") == "not_computable_no_item_count_level_reaches_the_floor"),
            "n_sessions_too_few_trials_for_lag_profile": sum(
                1 for d in decisive[window] if d.get("status") == "too_few_trials_for_lag_profile"),
            "n_sessions_too_few_trials_with_defined_lag1_alignment": sum(
                1 for d in decisive[window] if d.get("status") == "too_few_trials_with_defined_lag1_alignment"),
        }

    primary_branch = branch_by_window[str(PRIMARY_WINDOW)]
    windows_agree = len(set(branch_by_window.values())) == 1

    return {
        "decision_rule_declared_before_fitting": BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "adjacency_input_read_live": adjacency_reference,
        "established_sign_source": "reproduction reference's own within-item-count-level raw correlation, positive means worse behaviour",
        "established_sign_is_positive": established_positive,
        "detrend_windows_trials": list(DETREND_WINDOWS_TRIALS), "primary_window": PRIMARY_WINDOW,
        "n_sessions_total": len(bundles),
        "session_accounting_by_window": accounting_by_window,
        "decisive_partial_within_item_count_level_pooled_by_window": decisive_pooled,
        "per_session_decisive_partial_within_item_count_level": {str(w): decisive[w] for w in DETREND_WINDOWS_TRIALS},
        "branch_by_window": branch_by_window,
        "branch_at_primary_window": primary_branch,
        "windows_agree": windows_agree,
    }


# =======================================================================================================
# BLOCK B -- within-item-count-level temporal locus
# =======================================================================================================

def _session_subwindow_within_item_count(bundle: dict, bin_indices: np.ndarray, seed_prefix: str) -> dict:
    counts = bundle["counts"][:, :, bin_indices]
    activity = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    spike_count = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_direction", "n_trials_finite": int(finite.sum())}
    arrays = {
        "deviation": deviation[finite], "report_error": bundle["report_error"][finite],
        "spike_count": spike_count[finite], "trial_index": bundle["trial_index"][finite],
        "reaction_time": bundle["reaction_time"][finite], "item_count": bundle["item_count"][finite],
    }
    arm = _session_observable_arm(arrays, "deviation", seed_prefix)
    return {
        "status": "computed", "n_trials_finite": int(finite.sum()),
        "median_total_spike_count_per_trial": float(np.median(spike_count[finite])),
        "arm": arm,
    }


def _split_result_within_item_count(bundles: list[dict], n_windows: int, corpus_seed_prefix: str) -> dict:
    per_session_per_window: list[dict] = []
    for bundle in bundles:
        bin_groups = _sub_window_bins(bundle["counts"].shape[2], n_windows)
        row = {"session": bundle["session"], "windows": []}
        for w_index, bin_indices in enumerate(bin_groups):
            tag = f"{corpus_seed_prefix}|{bundle['session']}|n{n_windows}|w{w_index}"
            result = _fit(f"subwindow|{bundle['session']}|n{n_windows}|w{w_index}",
                           lambda b=bundle, bi=bin_indices, t=tag: _session_subwindow_within_item_count(b, bi, t))
            row["windows"].append(result)
        per_session_per_window.append(row)

    windows_summary = []
    for w_index in range(n_windows):
        per_session = [row["windows"][w_index] for row in per_session_per_window]
        gate_values = [s["arm"]["within_load_trial_count_weighted"]["orthogonality_gate_vs_spike_count"]
                       for s in per_session if s.get("status") == "computed"
                       and s["arm"]["within_load_trial_count_weighted"]["orthogonality_gate_vs_spike_count"] is not None]
        raw_values = [s["arm"]["within_load_trial_count_weighted"]["raw_vs_report_error"]
                      for s in per_session if s.get("status") == "computed"
                      and s["arm"]["within_load_trial_count_weighted"]["raw_vs_report_error"] is not None]
        gate_pooled = slope_across_sessions_test(gate_values, alternative="two-sided") if gate_values else {"status": "not_computed"}
        raw_pooled = slope_across_sessions_test(raw_values, alternative="two-sided") if raw_values else {"status": "not_computed"}
        if len(raw_values) >= 2:
            raw_pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(raw_values)
        void = bool(gate_pooled.get("status") == "tested" and gate_pooled.get("significant"))
        medians = [s["median_total_spike_count_per_trial"] for s in per_session if s.get("status") == "computed"]
        windows_summary.append({
            "window_index": w_index, "void": void,
            "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
            "median_total_spike_count_per_trial_across_sessions": float(np.median(medians)) if medians else None,
            "gate_pooled_within_item_count_level": gate_pooled,
            "raw_vs_behaviour_pooled_within_item_count_level": raw_pooled,
        })

    surviving = [w for w in windows_summary if not w["void"]]
    ordering_test = None
    if len(surviving) >= 2:
        earliest_idx, latest_idx = surviving[0]["window_index"], surviving[-1]["window_index"]
        paired = []
        for row in per_session_per_window:
            early, late = row["windows"][earliest_idx], row["windows"][latest_idx]
            if early.get("status") == "computed" and late.get("status") == "computed":
                early_r = early["arm"]["within_load_trial_count_weighted"]["raw_vs_report_error"]
                late_r = late["arm"]["within_load_trial_count_weighted"]["raw_vs_report_error"]
                if early_r is not None and late_r is not None:
                    paired.append(late_r - early_r)
        ordering_test = slope_across_sessions_test(paired, alternative="two-sided") if paired else {"status": "not_computed"}
        ordering_test["earliest_window_index"] = earliest_idx
        ordering_test["latest_window_index"] = latest_idx
        ordering_test["n_sessions_paired"] = len(paired)

    return {
        "n_windows": n_windows, "per_session_per_window": per_session_per_window, "windows": windows_summary,
        "surviving_window_indices": [w["window_index"] for w in surviving],
        "void_window_indices": [w["window_index"] for w in windows_summary if w["void"]],
        "ordering_test_latest_minus_earliest_surviving": ordering_test,
    }


def _block_b_branch_within_item_count(split: dict, full_epoch_behaviour_link_present: bool) -> dict:
    surviving = [w for w in split["windows"] if not w["void"]]
    if not surviving:
        return {"branch": "temporal_locus_unreachable_at_this_count_per_window", "sub_label": None}
    present_flags = [bool(w["raw_vs_behaviour_pooled_within_item_count_level"].get("status") == "tested"
                          and w["raw_vs_behaviour_pooled_within_item_count_level"].get("significant"))
                     for w in surviving]
    ordering = split["ordering_test_latest_minus_earliest_surviving"]
    ordering_significant = bool(ordering and ordering.get("status") == "tested" and ordering.get("significant"))

    if all(present_flags):
        if not ordering_significant:
            return {"branch": "accuracy_predicting_component_is_present_throughout_the_delay", "sub_label": None}
        if ordering["mean_value"] > 0.0:
            return {"branch": "accuracy_predicting_component_grows_across_the_delay", "sub_label": None}
        return {"branch": "accuracy_predicting_component_is_strongest_at_delay_onset", "sub_label": None}
    if not any(present_flags):
        if full_epoch_behaviour_link_present:
            return {"branch": "accuracy_predicting_component_requires_the_full_epoch_to_be_detected", "sub_label": None}
        return {"branch": "block_b_outcome_not_covered_by_the_pre_declared_rule",
                "sub_label": "absent_in_every_surviving_sub_window_but_also_absent_over_the_whole_epoch"}
    return {"branch": "block_b_outcome_not_covered_by_the_pre_declared_rule",
            "sub_label": "behaviour_link_present_in_some_but_not_all_surviving_sub_windows"}


def run_block_b(bundles: list[dict], full_epoch_behaviour_link_present: bool, corpus_seed_prefix: str) -> dict:
    splits = {name: _split_result_within_item_count(bundles, n, f"{corpus_seed_prefix}|{name}")
              for name, n in SUB_WINDOW_SPLITS.items()}
    primary = splits[PRIMARY_SPLIT]
    branch = _block_b_branch_within_item_count(primary, full_epoch_behaviour_link_present)
    thirds_branch = _block_b_branch_within_item_count(splits["thirds"], full_epoch_behaviour_link_present)
    return {
        "decision_rule_declared_before_fitting": BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "primary_split": PRIMARY_SPLIT, "full_epoch_behaviour_link_present": full_epoch_behaviour_link_present,
        "splits": splits, "branch": branch, "thirds_sensitivity_branch": thirds_branch,
        "primary_and_sensitivity_agree": branch["branch"] == thirds_branch["branch"],
    }


# =======================================================================================================
# Driver
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": "The multi-object macaque corpus only. No other corpus is touched.",
        "sign_convention": (
            "This corpus's behavioural variable is continuous report error, higher is worse. No sign flip "
            "is applied anywhere in this artifact."
        ),
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus")
    seen, loaded, refused = 0, [], []
    for session in iter_watters(root, bin_ms=100.0):
        seen += 1
        if session["status"] != "loaded":
            refused.append({"session": session["session"], "status": session["status"]})
            continue
        loaded.append(session)
    _log(f"seen={seen} loaded={len(loaded)} refused={len(refused)} elapsed={time.time() - t0:.0f}s")

    rows, bundles, n_arrays_none = _reproduction_rows(loaded)
    _log(f"reproduction rows built: {len(rows)}, bundles usable: {len(bundles)}, arrays_not_computable: "
         f"{n_arrays_none}, elapsed={time.time() - t0:.0f}s")

    gate_result = full_reproduction_gate(rows)
    output["reproduction_gate"] = gate_result
    output["zero_drop_accounting"] = {
        "n_seen": seen, "n_refused_by_the_shared_loader": len(refused),
        "n_loaded": len(loaded), "n_arrays_not_computable": n_arrays_none,
        "n_sessions_analysed": len(bundles),
        "reconciles": bool(seen == len(refused) + n_arrays_none + len(bundles)),
    }
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")

    if gate_result["status"] != "reproduced_exactly":
        output["status"] = "stopped_reproduction_gate_failed"
        output["block_a"] = {"status": "not_run_reproduction_gate_failed"}
        output["block_b"] = {"status": "not_run_reproduction_gate_failed"}
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce exactly; no new number was read")
        print(json.dumps({"reproduction_gate": gate_result["status"]}, indent=2))
        return

    reproduction_reference = gate_result["reference_read_live"]
    adjacency_reference = _live_adjacency_reference()
    pooled_across_item_count_reference = _live_pooled_across_item_count_reference()
    output["adjacency_input_read_live"] = adjacency_reference
    output["pooled_across_item_count_reference_read_live"] = pooled_across_item_count_reference
    _flush(output)

    full_epoch_link_present = bool(reproduction_reference["raw_vs_report_error"]["p"] <= 0.05)

    _log(f"Block A: {len(bundles)} sessions")
    block_a = run_block_a(bundles, adjacency_reference, reproduction_reference,
                           "multi_object_interference_and_locus_within_item_count|block_a")
    output["block_a"] = block_a
    _flush(output)
    _log(f"Block A branch at primary window: {block_a['branch_at_primary_window']} elapsed={time.time() - t0:.0f}s")

    _log(f"Block B: {len(bundles)} sessions")
    block_b = run_block_b(bundles, full_epoch_link_present,
                           "multi_object_interference_and_locus_within_item_count|block_b")
    output["block_b"] = block_b
    _flush(output)
    _log(f"Block B branch: {block_b['branch']['branch']} elapsed={time.time() - t0:.0f}s")

    primary_raw = block_a["decisive_partial_within_item_count_level_pooled_by_window"][str(PRIMARY_WINDOW)]["raw"]
    signs_differ = bool(
        pooled_across_item_count_reference.get("mean_value") is not None
        and primary_raw.get("mean_value") is not None
        and (pooled_across_item_count_reference["mean_value"] > 0.0) != (primary_raw["mean_value"] > 0.0)
    )
    output["interpretation_guard"] = {
        "note": (
            "The comparison a reader needs: the pooled-across-item-count result already on disk, beside "
            "the within-item-count-level result computed in this artifact, in the same table."
        ),
        "pooled_across_item_count": pooled_across_item_count_reference,
        "within_item_count_level_at_primary_window": primary_raw,
        "signs_differ": signs_differ,
    }

    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/multi_object_interference_and_locus_within_item_count_checkpoint.json",
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": gate_result["status"],
        "block_a_primary_window_branch": block_a["branch_at_primary_window"],
        "block_a_windows_agree": block_a["windows_agree"],
        "block_b_branch": block_b["branch"]["branch"],
        "signs_differ_from_pooled_across_item_count": signs_differ,
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
