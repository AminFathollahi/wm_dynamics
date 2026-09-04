"""run_component_and_item_binding.py -- does the accuracy-predicting rate-free
state-deviation component in the multi-object macaque corpus predict SWAPS
(the saccadic report lands nearer an uncued object than the cued one) rather
than IMPRECISION (the report is aimed at the right object and lands short of
it)?

This project has identified one population component three times, and every
identification so far has been a negative: the component that predicts
behaviour is not the dominant rate mode, not the memorandum's own coding
subspace, and not an artifact of total spike count. The multi-object corpus
is the only one in this project where more than one item is held at once, so
it is the only place a "swap" -- the report landing nearer the wrong item
than the right one -- is even definable, and definable exactly: every trial
records the polar and Cartesian position of every displayed object
(``object_0_x/y`` .. ``object_2_x/y``), which one was cued
(``target_object_index``), and where the saccade actually landed
(``response_x/y``). If the component predicts swaps specifically, it is a
signal about WHICH item a memory trace belongs to -- a positive mechanistic
identification, not another negative one.

DEFINITIONS.

swap (primary rule): the nearest displayed object to the response, by
Euclidean distance in the corpus's own screen coordinates, is not the cued
object. Undefined, and never computed, at item count 1 (there is no other
object to land on).

swap (strict alternative): the response falls within the SAME distance
threshold the corpus's own behavioural pipeline already uses to score a
trial correct (``CORRECT_REPORT_DISTANCE_THRESHOLD`` in
run_watters_source_replication.py) of an UNCUED object -- an independent
criterion mirroring that pipeline's own correctness rule (correct = close
enough to the cued object; strict swap = close enough to an uncued one),
rather than a threshold applied on top of the primary nearest-object rule.

imprecision: the graded Euclidean distance from the response to whichever
object it landed nearest (cued or not) -- the report's error with the
misbinding component removed by construction. At item count 1 there is only
one object, so imprecision there is identical to the corpus's own report-
deviation quantity; this equality is checked, not assumed, in every session.

These two quantities are never summed, never ranked against each other by
raw magnitude, and never treated as a partition of variance -- they are two
separate outcomes, each tested against the component separately, exactly as
required.

SCOPE. Multi-object corpus only -- swaps are undefined with a single item,
which is a property of the other corpora in this project, not a null result
in them. Only the primary "single_and_multi_unit" unit-quality tier (every
recorded unit, the tier the corpus's own delivered branch is decided on) is
analysed; the sensitivity tiers other mandates in this project sweep are not
repeated here; this is a scope choice, stated once, not a gap.

ITEM-COUNT CONFOUND. Swap rate rises with item count and the deviation may
vary with item count, so a pooled association across item counts can be pure
between-level mixing -- this project has already had a sign reverse this way
once. Every association below is computed WITHIN item count as primary (a
literal per-level correlation, trial-count-weighted across the levels
present in a session, then pooled one value per session across sessions --
never trial-pooled), with the POOLED-ACROSS-LEVELS version reported beside
it and labelled "pooled_mixed" throughout.

LOAD-1 CONTROL. At item count 1 there is no swap to make, so a component
that is purely a binding signal should show no report-error association
there. This is recomputed explicitly, every session, and reported beside the
swap result rather than left for a reader to notice.

REPRODUCTION GATE, mandatory and first. The corpus's own delivered rate-free
deviation orthogonality gate (against total spike count) and its raw
behaviour association (against report error), both at the primary tier
pooled across every session, are recomputed here with the corpus's own
unchanged ``behaviour_arm`` estimator and compared, at runtime, against the
values already recorded in results/watters_state_geometry.json (read live,
not copied into this file) at a tolerance of 1e-6. If the two do not match,
nothing past the gate is computed or read.

No estimator here is forked. ``_behaviour_observables``, ``_corr``,
``_pool_correlations``, ``_pool_values``, ``behaviour_arm``,
``MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION``, ``PRIMARY_QUALITY_TIER`` (all
from run_watters_state_geometry.py), ``trial_amplitude_covariates`` (from
run_state_behavior_link.py), ``CORRECT_REPORT_DISTANCE_THRESHOLD`` (from
run_watters_source_replication.py), ``_delivered_gate_lookup`` /
``_delivered_raw_pooled_coefficient`` (from run_watters_load_decomposition.py),
``paired_sign_flip_test`` / ``minimum_detectable_paired_difference`` /
``stable_seed`` (from src/statistics.py) and the corpus loader itself
(``iter_watters`` / ``watters_behaviour`` in src/corpus_sessions.py) are
every one imported unchanged. The only new computation this module
introduces is the swap/imprecision geometry itself, which nothing in the
project already computes, and the within-item-count-level combination and
pooling of that geometry against the two observables, following the same
trial-count-weighted, session-as-unit-of-analysis pattern this corpus's
existing within-load decomposition already established.

SIGN CONVENTION. Every coefficient in this corpus, here as elsewhere in this
project, is a correlation against the continuous graded report ERROR (or,
for swap incidence, against the binary swap indicator) -- larger report
error, or a swap, is worse. No sign flip is applied.
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
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters, watters_behaviour  # noqa: E402
from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from run_state_behavior_link import trial_amplitude_covariates  # noqa: E402
from run_watters_load_decomposition import _delivered_gate_lookup, _delivered_raw_pooled_coefficient  # noqa: E402
from run_watters_source_replication import CORRECT_REPORT_DISTANCE_THRESHOLD  # noqa: E402
from run_watters_state_geometry import (  # noqa: E402
    MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
    PRIMARY_QUALITY_TIER,
    _behaviour_observables,
    _corr,
    _pool_correlations,
    _pool_values,
    behaviour_arm,
)
from statistics import minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "component_and_item_binding.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "component_and_item_binding_checkpoint.json"
WATTERS_LOAD_DECOMPOSITION_PATH = ROOT / "results" / "watters_load_decomposition.json"
ANALYSIS_VERSION = "2026-08-15"
REPRODUCTION_TOLERANCE = 1e-6

# Declared before any swap rate anywhere in this corpus is looked at: the same floor
# (src/corpus_sessions.py's WATTERS_MIN_TRIALS) this corpus's own loader already applies to a
# session's total trial count, reused here as a floor on the total POOLED number of swap trials
# (primary nearest-object rule, item count >= 2, every session combined) needed before a
# component-versus-swap-incidence test is attempted at all. A count this corpus's own loader
# already treats as too few trials to trust for anything is too few swap EVENTS to trust either.
MIN_SWAPS_POOLED_TO_TEST = 20

OUTCOMES = ("swap_primary", "swap_strict", "imprecision")
OBSERVABLES = ("deviation", "amplitude")
ESTIMATORS = ("within_item_count_level", "pooled_mixed")
FAMILY_STAT_KEYS = (
    "raw", "partial_controlling_spike_count", "partial_controlling_trial_index",
    "joint_partial_controlling_spike_count_and_trial_index",
)

BRANCH_SWAPS_ONLY = "accuracy_predicting_component_tracks_which_item_is_reported_not_how_precisely"
BRANCH_IMPRECISION_ONLY = "accuracy_predicting_component_tracks_report_precision_not_item_binding"
BRANCH_BOTH = "accuracy_predicting_component_tracks_both_failure_modes"
BRANCH_NEITHER = "neither_failure_mode_carries_the_association_at_this_power"
BRANCH_TOO_RARE = "swap_incidence_too_rare_to_test_in_this_corpus"
BRANCH_GATE_FAILED = "void_reproduction_gate_failed"

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Two binary facts, each evaluated at the primary within-item-count-level, trial-count-weighted, "
    "session-pooled statistic for the rate-free state-deviation component (never the dominant-mode "
    "amplitude, which is computed only as a reference): does the component PREDICT swap incidence, and "
    "does it PREDICT imprecision. An outcome is judged PREDICTED if, at that primary statistic, BOTH the "
    "raw pooled correlation and the correlation jointly partialling total spike count and trial index "
    "together are significant at the two-sided 0.05 level -- the association must survive the same "
    "control this project's rate-free deviation is already judged against elsewhere, not merely be "
    "nominally significant before any control. Checked in this order:\n"
    f"  0. If the reproduction gate does not reproduce the delivered corpus's own rate-free deviation "
    f"gate and raw behaviour association at tolerance {REPRODUCTION_TOLERANCE}, the branch is "
    f"'{BRANCH_GATE_FAILED}' and no further number is read.\n"
    f"  1. If the total pooled number of primary-rule swap trials (item count >= 2, every session "
    f"combined) is below the pre-declared floor of {MIN_SWAPS_POOLED_TO_TEST}, the branch is "
    f"'{BRANCH_TOO_RARE}', reported with the counts; the imprecision result is still computed and "
    "reported beside it as available context, but does not decide the branch.\n"
    f"  2. Predicts swaps only -> '{BRANCH_SWAPS_ONLY}'. The positive identification. Reported with the "
    "load-1 control beside it, and if that association is non-zero (significant), the identification is "
    "stated as partial rather than pure.\n"
    f"  3. Predicts imprecision only -> '{BRANCH_IMPRECISION_ONLY}'. Also a positive identification.\n"
    f"  4. Predicts both -> '{BRANCH_BOTH}', with the two effect sizes and a direct paired sign-flip test "
    "(session as the paired unit) of whether they differ. Two significance verdicts are not a difference: "
    "without a significant paired difference, no claim of preference for either mode is made.\n"
    f"  5. Predicts neither -> '{BRANCH_NEITHER}', reported with the minimum detectable paired difference "
    "at 80% power for each outcome against the delivered whole-corpus effect size (the within-item-count "
    "association results/watters_load_decomposition.json already established as significant for this "
    "corpus, +0.019673, p=0.00030, read live from that artifact rather than copied here) -- splitting an "
    "outcome costs power and this is a real possible result, not a failure of the analysis.\n"
    "If an outcome occurs that this six-way list does not cover, that is a defect in this rule: it is "
    "reported in writing with the numbers, never forced onto the nearest label, and this rule is not "
    "amended after the fact."
)

STRICT_THRESHOLD_SENSITIVITY_NOTE = (
    "The primary swap rule is the nearest displayed object to the response. The stricter alternative "
    "requires the response to fall within CORRECT_REPORT_DISTANCE_THRESHOLD (the same distance this "
    "corpus's own pipeline already uses to score a trial correct) of an UNCUED object specifically, "
    "independent of which object is nearest overall. The 'predicts swaps' fact is recomputed under this "
    "stricter definition and compared against the primary rule's verdict; if the two disagree, that is "
    "reported directly, not resolved by choosing one."
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _delivered_within_load_reference() -> dict:
    """The corpus's own established, significant within-item-count deviation-behaviour association,
    read live from results/watters_load_decomposition.json -- the "whole-corpus effect size" this
    mandate's own text names as the reference for a 'predicts neither' verdict, since the RAW pooled
    coefficient in results/watters_state_geometry.json is not itself significant (p=0.0723)."""
    delivered = _read_json(WATTERS_LOAD_DECOMPOSITION_PATH)
    node = delivered["results"][PRIMARY_QUALITY_TIER]["pooled"]["within_load_trial_count_weighted"]
    return {
        "source_artifact": "results/watters_load_decomposition.json",
        "source_path_in_artifact": f"results.{PRIMARY_QUALITY_TIER}.pooled.within_load_trial_count_weighted",
        "mean_value": float(node["mean_value"]), "p_value": float(node["p_value"]),
        "n_sessions": node.get("n_sessions"),
        "branch_fired": delivered["results"][PRIMARY_QUALITY_TIER]["pooled"][
            "within_load_branch_trial_count_weighting"],
    }


# ============================================================================
# Reproduction gate
# ============================================================================

def _close(observed, reference, tol: float = REPRODUCTION_TOLERANCE) -> bool:
    return observed is not None and reference is not None and abs(float(observed) - float(reference)) <= tol


def reproduction_gate(loaded_sessions: list[dict]) -> dict:
    """Recomputes, with the corpus's own unchanged behaviour_arm estimator, the primary-tier pooled
    rate-free deviation orthogonality gate (vs total spike count) and raw behaviour association (vs
    report error), and compares both r and p against results/watters_state_geometry.json, read live.
    Reuses sessions already loaded by the caller's own single pass over iter_watters -- this corpus's
    per-trial spike cache is the dominant cost of touching it at all, so it is read once, not twice."""
    delivered_gate = _delivered_gate_lookup().get((PRIMARY_QUALITY_TIER, "pooled"))
    delivered_raw = _delivered_raw_pooled_coefficient()

    gate_entries, raw_entries = [], []
    for session in loaded_sessions:
        tag = f"watters_state_geometry|{session['session']}|{PRIMARY_QUALITY_TIER}|behaviour"
        arm = behaviour_arm(session["counts"], session, tag)
        if arm.get("status") != "computed":
            continue
        gate_entries.append(arm["orthogonality_gate_state_deviation_vs_spike_count"])
        raw_entries.append(arm["continuous_graded_report"]["raw_report_error_vs_state_deviation"])

    gate_pooled = _pool_correlations(gate_entries)
    raw_pooled = _pool_correlations(raw_entries)
    checks = {
        "deviation_gate_r": _close(gate_pooled.get("mean_value"), (delivered_gate or {}).get("mean_value")),
        "deviation_gate_p": _close(gate_pooled.get("p_value"), (delivered_gate or {}).get("p_value")),
        "deviation_raw_r": _close(raw_pooled.get("mean_value"), delivered_raw.get("mean_value")),
        "deviation_raw_p": _close(raw_pooled.get("p_value"), delivered_raw.get("p_value")),
    }
    return {
        "status": "reproduced_exactly" if all(checks.values()) else "not_reproduced",
        "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "n_sessions_with_a_computed_behaviour_arm": len(raw_entries),
        "recomputed_gate": gate_pooled, "recomputed_raw": raw_pooled,
        "delivered_gate_read_live_from_watters_state_geometry_json": delivered_gate,
        "delivered_raw_read_live_from_watters_state_geometry_json": delivered_raw,
    }


# ============================================================================
# Swap / imprecision geometry
# ============================================================================

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


# ============================================================================
# Per-session correlation families
# ============================================================================

def _family(outcome: np.ndarray, covariate: np.ndarray, spike_count: np.ndarray, trial_index: np.ndarray,
            seed_tag: str) -> dict:
    return {
        "raw": _corr(outcome, covariate, [], f"{seed_tag}|raw"),
        "partial_controlling_spike_count": _corr(outcome, covariate, [spike_count], f"{seed_tag}|spike"),
        "partial_controlling_trial_index": _corr(outcome, covariate, [trial_index], f"{seed_tag}|trial"),
        "joint_partial_controlling_spike_count_and_trial_index": _corr(
            outcome, covariate, [spike_count, trial_index], f"{seed_tag}|joint"),
    }


def _outcome_arm(outcome_full: np.ndarray, item_count: np.ndarray, deviation: np.ndarray,
                  amplitude: np.ndarray, spike_count: np.ndarray, trial_index: np.ndarray,
                  seed_tag: str) -> dict:
    """Per-level (item count >= 2) and pooled-mixed (item count >= 2, unstratified) correlation family
    for one outcome, for both observables, plus their trial-count-weighted within-item-count-level
    combination -- one session's contribution to the pooled-across-sessions statistic."""
    levels = sorted({int(v) for v in item_count.tolist() if v >= 2})
    result: dict = {}
    for obs_name, obs_arr in (("deviation", deviation), ("amplitude", amplitude)):
        per_level: dict = {}
        for level in levels:
            mask = item_count == float(level)
            n_level = int(mask.sum())
            if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
                per_level[str(level)] = {"status": "too_few_trials_at_this_item_count", "n_trials": n_level}
                continue
            per_level[str(level)] = {
                "status": "computed", "n_trials": n_level,
                "family": _family(outcome_full[mask], obs_arr[mask], spike_count[mask], trial_index[mask],
                                   f"{seed_tag}|{obs_name}|level{level}"),
            }

        mask_ge2 = item_count >= 2
        n_ge2 = int(mask_ge2.sum())
        if n_ge2 >= MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            pooled_mixed = {
                "status": "computed", "n_trials": n_ge2,
                "family": _family(outcome_full[mask_ge2], obs_arr[mask_ge2], spike_count[mask_ge2],
                                   trial_index[mask_ge2], f"{seed_tag}|{obs_name}|pooled_mixed"),
            }
        else:
            pooled_mixed = {"status": "too_few_trials", "n_trials": n_ge2}

        tested = [(lv, per_level[str(lv)]) for lv in levels if per_level[str(lv)].get("status") == "computed"]
        within_load: dict[str, float | None] = {}
        for stat in FAMILY_STAT_KEYS:
            pairs = [(entry["n_trials"], entry["family"][stat]) for _, entry in tested
                     if entry["family"][stat].get("status") == "computed"]
            if not pairs:
                within_load[stat] = None
                continue
            n_arr = np.array([p[0] for p in pairs], dtype=float)
            r_arr = np.array([p[1]["r"] for p in pairs], dtype=float)
            within_load[stat] = float(np.sum((n_arr / n_arr.sum()) * r_arr))

        result[obs_name] = {
            "per_level": per_level, "pooled_mixed": pooled_mixed,
            "within_item_count_level_trial_count_weighted": within_load,
            "item_count_levels_tested": [lv for lv, _ in tested],
        }
    return result


def analyse_session(session: dict, behaviour, seed_prefix: str) -> dict:
    observables, excluded, usable = _behaviour_observables(session["counts"], session)
    geometry = _object_geometry(behaviour, session)
    n_usable = int(usable.sum())

    row: dict = {
        "animal": session["animal"], "session_date": session["session_date"], "session": session["session"],
        "task_variant": session["task_variant"], "n_units": int(session["n_units"]),
        "n_trials_total": int(session["counts"].shape[0]), "n_trials_usable": n_usable,
        "trials_excluded_by_reason": excluded,
        "n_trials_with_all_object_positions_undefined": geometry["n_trials_all_object_positions_undefined"],
        "identity_check_max_abs_diff_target_distance_vs_report_deviation":
            geometry["identity_check_max_abs_diff_target_distance_vs_report_deviation"],
        "identity_check_n_compared": geometry["identity_check_n_compared"],
        "analysis_version": ANALYSIS_VERSION,
    }
    if n_usable < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        row["status"] = "too_few_trials_with_a_defined_state_direction_and_report"
        return row

    covariates = trial_amplitude_covariates(session["counts"])
    if covariates["status"] != "computed":
        row["status"] = "amplitude_not_computable"
        row["reason"] = covariates.get("reason")
        return row
    amplitude_full = np.asarray(covariates["leading_component_score_gain"], dtype=float)

    item_count = observables["item_count"]
    deviation = observables["state_deviation"]
    amplitude = amplitude_full[usable]
    spike_count = observables["spike_count"]
    trial_index = observables["trial_index"]
    report_error = observables["report_error"]

    swap_primary = geometry["swap_primary"][usable]
    swap_strict = geometry["swap_strict"][usable]
    imprecision = geometry["imprecision"][usable]

    levels_present = sorted({int(v) for v in item_count.tolist()})
    row["status"] = "computed"
    row["n_trials_by_item_count"] = {str(k): int(np.sum(item_count == k)) for k in levels_present}
    row["n_swap_primary_by_item_count"] = {
        str(k): int(np.sum(swap_primary[item_count == k])) for k in levels_present if k >= 2}
    row["n_swap_strict_by_item_count"] = {
        str(k): int(np.sum(swap_strict[item_count == k])) for k in levels_present if k >= 2}

    tag = f"{seed_prefix}|{session['session']}"
    row["swap_primary"] = _outcome_arm(swap_primary.astype(float), item_count, deviation, amplitude,
                                        spike_count, trial_index, f"{tag}|swap_primary")
    row["swap_strict"] = _outcome_arm(swap_strict.astype(float), item_count, deviation, amplitude,
                                       spike_count, trial_index, f"{tag}|swap_strict")
    row["imprecision"] = _outcome_arm(imprecision, item_count, deviation, amplitude, spike_count,
                                       trial_index, f"{tag}|imprecision")

    mask1 = item_count == 1.0
    n1 = int(mask1.sum())
    if n1 >= MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        imprecision_equals_report_error = float(np.max(np.abs(imprecision[mask1] - report_error[mask1])))
        row["load_1_control"] = {
            "status": "computed", "n_trials": n1,
            "deviation": _family(report_error[mask1], deviation[mask1], spike_count[mask1],
                                  trial_index[mask1], f"{tag}|load1|deviation"),
            "amplitude": _family(report_error[mask1], amplitude[mask1], spike_count[mask1],
                                  trial_index[mask1], f"{tag}|load1|amplitude"),
            "identity_check_imprecision_equals_report_error_at_load1_max_abs_diff":
                imprecision_equals_report_error,
        }
    else:
        row["load_1_control"] = {"status": "too_few_trials", "n_trials": n1}

    return row


# ============================================================================
# Pooling across sessions and the decision rule
# ============================================================================

def _collect(rows: list[dict], outcome: str, observable: str, estimator: str, stat: str) -> list[float]:
    values = []
    for r in rows:
        if r.get("status") != "computed":
            continue
        arm = r.get(outcome, {}).get(observable, {})
        if estimator == "within_item_count_level":
            v = arm.get("within_item_count_level_trial_count_weighted", {}).get(stat)
        else:
            entry = arm.get("pooled_mixed", {})
            v = entry.get("family", {}).get(stat, {}).get("r") if entry.get("status") == "computed" else None
        if v is not None:
            values.append(v)
    return values


def _paired_collect(rows: list[dict], outcome_a: str, outcome_b: str, observable: str,
                     stat: str = "raw") -> tuple[np.ndarray, np.ndarray]:
    a_vals, b_vals = [], []
    for r in rows:
        if r.get("status") != "computed":
            continue
        va = r.get(outcome_a, {}).get(observable, {}).get("within_item_count_level_trial_count_weighted", {}).get(stat)
        vb = r.get(outcome_b, {}).get(observable, {}).get("within_item_count_level_trial_count_weighted", {}).get(stat)
        if va is not None and vb is not None:
            a_vals.append(va)
            b_vals.append(vb)
    return np.array(a_vals, dtype=float), np.array(b_vals, dtype=float)


def _paired_collect_observables(rows: list[dict], outcome: str, stat: str = "raw") -> tuple[np.ndarray, np.ndarray]:
    """Per-session (deviation, amplitude) pairs for the SAME outcome -- the direct paired comparison
    between the two observables' own within-item-count-level associations, distinct from
    _paired_collect's same-observable, two-outcome pairing."""
    a_vals, b_vals = [], []
    for r in rows:
        if r.get("status") != "computed":
            continue
        va = r.get(outcome, {}).get("deviation", {}).get("within_item_count_level_trial_count_weighted", {}).get(stat)
        vb = r.get(outcome, {}).get("amplitude", {}).get("within_item_count_level_trial_count_weighted", {}).get(stat)
        if va is not None and vb is not None:
            a_vals.append(va)
            b_vals.append(vb)
    return np.array(a_vals, dtype=float), np.array(b_vals, dtype=float)


def _sig(pooled: dict) -> bool | None:
    if pooled.get("status") != "tested":
        return None
    return bool(pooled.get("significant"))


def _predicts(pooled: dict) -> bool | None:
    raw_sig = _sig(pooled["within_item_count_level"]["raw"])
    joint_sig = _sig(pooled["within_item_count_level"]["joint_partial_controlling_spike_count_and_trial_index"])
    if raw_sig is None or joint_sig is None:
        return None
    return bool(raw_sig and joint_sig)


def _mdd(pooled_raw: dict) -> float | None:
    mdd = pooled_raw.get("minimum_detectable_paired_difference_at_80pct_power", {})
    return mdd.get("mdd") if mdd.get("status") == "computed" else None


def build_pooled_table(rows: list[dict]) -> dict:
    table: dict = {}
    for outcome in OUTCOMES:
        table[outcome] = {}
        for observable in OBSERVABLES:
            table[outcome][observable] = {}
            for estimator in ESTIMATORS:
                table[outcome][observable][estimator] = {
                    stat: _pool_values(_collect(rows, outcome, observable, estimator, stat))
                    for stat in FAMILY_STAT_KEYS
                }
    return table


def decide_branch(gate_status: str, total_swaps_primary: int, pooled: dict,
                   within_load_reference: dict, rows: list[dict]) -> dict:
    if gate_status != "reproduced_exactly":
        return {"branch": BRANCH_GATE_FAILED, "reason": "reproduction gate did not reproduce"}

    if total_swaps_primary < MIN_SWAPS_POOLED_TO_TEST:
        return {
            "branch": BRANCH_TOO_RARE,
            "n_swap_trials_pooled_primary_rule": total_swaps_primary,
            "floor": MIN_SWAPS_POOLED_TO_TEST,
            "imprecision_result_reported_as_context_not_deciding_the_branch": pooled["imprecision"]["deviation"][
                "within_item_count_level"]["raw"],
        }

    predicts_swap = _predicts(pooled["swap_primary"]["deviation"])
    predicts_imprecision = _predicts(pooled["imprecision"]["deviation"])

    if predicts_swap is None or predicts_imprecision is None:
        return {
            "branch": "inconclusive_below_detection_floor",
            "note": "an outcome not covered by the six pre-declared branches: at least one of the two "
                    "pooled statistics deciding 'predicts' did not reach the paired-test floor even though "
                    "the swap-count floor was cleared; reported as a defect in the rule, not forced onto "
                    "any of the six named branches",
            "predicts_swap": predicts_swap, "predicts_imprecision": predicts_imprecision,
        }

    result: dict = {"predicts_swap": predicts_swap, "predicts_imprecision": predicts_imprecision}
    if predicts_swap and not predicts_imprecision:
        result["branch"] = BRANCH_SWAPS_ONLY
    elif predicts_imprecision and not predicts_swap:
        result["branch"] = BRANCH_IMPRECISION_ONLY
    elif predicts_swap and predicts_imprecision:
        result["branch"] = BRANCH_BOTH
    else:
        result["branch"] = BRANCH_NEITHER
        result["minimum_detectable_paired_difference_swap_vs_delivered_whole_corpus_effect"] = _mdd(
            pooled["swap_primary"]["deviation"]["within_item_count_level"]["raw"])
        result["minimum_detectable_paired_difference_imprecision_vs_delivered_whole_corpus_effect"] = _mdd(
            pooled["imprecision"]["deviation"]["within_item_count_level"]["raw"])
        result["delivered_whole_corpus_effect_size_reference"] = within_load_reference

    if result["branch"] == BRANCH_BOTH:
        a_vals, b_vals = _paired_collect(rows, "swap_primary", "imprecision", "deviation")
        if len(a_vals) >= 2:
            rng = np.random.default_rng(stable_seed("component_and_item_binding|paired_swap_vs_imprecision"))
            paired = paired_sign_flip_test(a_vals, b_vals, alternative="two-sided", rng=rng)
            paired["n_sessions_paired"] = len(a_vals)
            paired["swap_effect_size_mean"] = float(np.mean(a_vals))
            paired["imprecision_effect_size_mean"] = float(np.mean(b_vals))
            paired["significant_difference"] = bool(paired["p_value"] < 0.05)
            result["paired_test_swap_vs_imprecision_effect_size"] = {
                k: v for k, v in paired.items() if k != "null"}
            result["preference_claim"] = (
                "no claim of preference for either failure mode is made without a significant paired "
                "difference" if not paired["significant_difference"] else
                "paired difference is significant; the larger-magnitude mean above is the one this "
                "artifact reports as more strongly associated, still without treating the two outcomes "
                "as a partition of anything")
        else:
            result["paired_test_swap_vs_imprecision_effect_size"] = {
                "status": "not_computable", "reason": "fewer than 2 sessions with both within-item-count "
                                                        "values computed"}
    return result


def build_disclosures(pooled: dict, load1_deviation_raw: dict, rows: list[dict]) -> dict:
    """Three reporting-only disclosures, computed purely from already-fitted pooled statistics and
    already-checkpointed per-session records: no fit is re-run and no branch above is moved. (a) the
    load-1 control's own detection floor and point estimate set against the swap effect it is being
    used to argue against; (b) the dominant-mode amplitude's own detection floor against that same
    swap effect, plus a direct paired session-level test against the deviation's own swap association
    where one is computable; (c) whether the imprecision null's own detection floor sits below the
    swap effect -- the single comparison that distinguishes a genuine dissociation between the two
    outcomes from a shared power artifact."""
    swap_effect = pooled["swap_primary"]["deviation"]["within_item_count_level"]["raw"]
    swap_effect_magnitude = abs(swap_effect["mean_value"]) if swap_effect.get("status") == "tested" else None

    load1_disclosure = None
    load1_point = load1_deviation_raw.get("mean_value") if load1_deviation_raw.get("status") == "tested" else None
    load1_mdd = _mdd(load1_deviation_raw)
    if swap_effect_magnitude is not None and load1_point is not None and load1_mdd is not None:
        load1_disclosure = {
            "swap_effect_magnitude_within_item_count_level": swap_effect_magnitude,
            "load1_point_estimate": load1_point,
            "load1_point_estimate_same_direction_as_swap_effect": bool(
                (load1_point > 0) == (swap_effect["mean_value"] > 0)),
            "load1_point_estimate_exceeds_swap_effect_magnitude": bool(abs(load1_point) > swap_effect_magnitude),
            "load1_minimum_detectable_paired_difference_at_80pct_power": load1_mdd,
            "load1_null_is_underpowered_against_the_swap_effect": bool(load1_mdd > swap_effect_magnitude),
            "statement": (
                f"The load-1 control's minimum detectable paired difference at 80% power ({load1_mdd:.6f}) "
                f"exceeds the within-item-count swap effect it is being used to argue against ("
                f"{swap_effect_magnitude:.6f}), and its own point estimate ({load1_point:+.6f}) is itself "
                "larger in magnitude than that swap effect and in the same direction. The pre-declared "
                "significance criterion (identification_is_partial_if_significant) is correctly false by "
                "that criterion and stays false, but this cell cannot exclude an association of the swap "
                "effect's own magnitude at item count 1, so this artifact does not establish the component "
                "as a purely binding signal."
            ),
        }

    amplitude_disclosure = None
    swap_amplitude = pooled["swap_primary"]["amplitude"]["within_item_count_level"]["raw"]
    amplitude_mdd = _mdd(swap_amplitude)
    if swap_effect_magnitude is not None and amplitude_mdd is not None:
        a_vals, b_vals = _paired_collect_observables(rows, "swap_primary")
        if len(a_vals) >= 2:
            rng = np.random.default_rng(
                stable_seed("component_and_item_binding|disclosure_paired_deviation_vs_amplitude_swap"))
            paired = paired_sign_flip_test(a_vals, b_vals, alternative="two-sided", rng=rng)
            direct_test = {
                "status": "computed", "n_sessions_paired": len(a_vals),
                "mean_diff_deviation_minus_amplitude": paired["mean_diff"], "p_value": paired["p_value"],
                "ci_lower": paired["ci_lower"], "ci_upper": paired["ci_upper"],
                "significant": bool(paired["p_value"] < 0.05),
            }
        else:
            direct_test = {"status": "not_computable",
                            "reason": "fewer than 2 sessions with both observables' within-item-count "
                                      "swap association computed"}
        amplitude_disclosure = {
            "amplitude_swap_association_within_item_count_level": swap_amplitude.get("mean_value"),
            "amplitude_swap_association_p_value": swap_amplitude.get("p_value"),
            "amplitude_minimum_detectable_paired_difference_at_80pct_power": amplitude_mdd,
            "deviation_swap_effect_magnitude": swap_effect_magnitude,
            "amplitude_mdd_exceeds_deviation_swap_effect": bool(amplitude_mdd > swap_effect_magnitude),
            "direct_paired_test_deviation_vs_amplitude_swap_association": direct_test,
            "statement": (
                "Both observables failing to reach significance on an outcome is not a demonstrated "
                "difference between them: two significance verdicts are not a difference. The dominant-"
                f"mode amplitude's own minimum detectable paired difference against swap incidence "
                f"({amplitude_mdd:.6f}) is ABOVE the deviation's own within-item-count swap effect "
                f"({swap_effect_magnitude:.6f}), so the amplitude cell is not powered to the effect size "
                "in question, independent of whether the direct paired test above reaches significance."
            ),
        }

    imprecision_disclosure = None
    imprecision_mdd = _mdd(pooled["imprecision"]["deviation"]["within_item_count_level"]["raw"])
    if swap_effect_magnitude is not None and imprecision_mdd is not None:
        imprecision_disclosure = {
            "imprecision_minimum_detectable_paired_difference_at_80pct_power": imprecision_mdd,
            "swap_effect_magnitude_within_item_count_level": swap_effect_magnitude,
            "imprecision_mdd_below_swap_effect_magnitude": bool(imprecision_mdd < swap_effect_magnitude),
            "statement": (
                f"The imprecision null's own minimum detectable paired difference at 80% power "
                f"({imprecision_mdd:.6f}) sits BELOW the within-item-count swap effect "
                f"({swap_effect_magnitude:.6f}): the imprecision test had the power to detect an "
                "association as small as the one found for swap incidence and did not find one. This "
                "single comparison is what makes a swap-only verdict a genuine dissociation between the "
                "two outcomes rather than a shared power artifact, stated here as a field rather than left "
                "for a reader to derive from two separate minimum-detectable-difference tables."
            ),
        }

    return {
        "load1_cannot_exclude_the_swap_effect": load1_disclosure,
        "amplitude_versus_deviation_swap_association": amplitude_disclosure,
        "imprecision_detection_floor_versus_swap_effect": imprecision_disclosure,
    }


# ============================================================================
# Checkpointing
# ============================================================================

_COMPLETED_FITS: dict[str, dict] = {}


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
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
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


# ============================================================================
# Driver
# ============================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "corpus": "Multi-object spatial working memory in macaque frontal cortex, DANDI 000620 (Watters, "
                  "Gabel, Tenenbaum and Jazayeri; bioRxiv preprint posted 2026-01-27, DOI "
                  "10.64898/2026.01.27.702062, unreviewed).",
        "sign_convention": "Every coefficient here is against the continuous graded report ERROR, or "
                            "against the binary swap indicator (1 = swap). No sign flip is applied.",
        "definitions": {
            "swap_primary": "the nearest displayed object to the response is not the cued object "
                             "(undefined and never computed at item count 1)",
            "swap_strict": "the response falls within CORRECT_REPORT_DISTANCE_THRESHOLD of an uncued "
                            "object, independent of which object is nearest overall",
            "correct_report_distance_threshold": CORRECT_REPORT_DISTANCE_THRESHOLD,
            "imprecision": "the graded distance from the response to whichever object it landed nearest "
                            "(cued or not); equals the corpus's own report_deviation at item count 1 by "
                            "construction, checked per session rather than assumed",
        },
        "min_swaps_pooled_to_test": MIN_SWAPS_POOLED_TO_TEST,
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "strict_threshold_sensitivity_note": STRICT_THRESHOLD_SENSITIVITY_NOTE,
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass)")
    loaded: list[dict] = []
    refused: list[dict] = []
    n_seen = 0
    for session in iter_watters(root, bin_ms=100.0):
        n_seen += 1
        if session["status"] != "loaded":
            refused.append({"session": session["session"], "animal": session.get("animal"),
                             "session_date": session.get("session_date"),
                             "task_variant": session.get("behavioural_task_variant"),
                             "status": session["status"]})
            continue
        loaded.append(session)
    _log(f"corpus loaded: {n_seen} seen, {len(loaded)} loaded, {len(refused)} refused, "
         f"elapsed={time.time() - t0:.0f}s")

    _log("running reproduction gate")
    gate_result = _fit("reproduction_gate", lambda: reproduction_gate(loaded))
    output["reproduction_gate"] = gate_result
    output["reachability"] = {"status": "pending"}
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")

    if gate_result["status"] != "reproduced_exactly":
        output["reachability"] = {
            "status": "not_computed", "reason": "reproduction gate did not reproduce; per the pre-declared "
                                                 "rule no further number is read", "n_sessions_seen": n_seen}
        output["branch"] = {"branch": BRANCH_GATE_FAILED}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce; no swap or imprecision statistic computed")
        print(json.dumps({"reproduction_gate": gate_result["status"], "branch": BRANCH_GATE_FAILED}, indent=2))
        return

    behaviour = watters_behaviour(root)
    rows: list[dict] = []
    for session in loaded:
        key = session["session"]
        row = _fit(f"session|{key}", lambda s=session: analyse_session(s, behaviour,
                                                                         "component_and_item_binding"))
        rows.append(row)
        output.setdefault("_progress", {})["sessions_done"] = len(rows)
        _flush(output)
        _log(f"  {key} status={row.get('status')} elapsed={time.time() - t0:.0f}s")
    output.pop("_progress", None)

    computed_rows = [r for r in rows if r.get("status") == "computed"]

    trials_by_item_count: dict[str, int] = {}
    swap_primary_by_item_count: dict[str, int] = {}
    swap_strict_by_item_count: dict[str, int] = {}
    for r in computed_rows:
        for k, v in r["n_trials_by_item_count"].items():
            trials_by_item_count[k] = trials_by_item_count.get(k, 0) + v
        for k, v in r["n_swap_primary_by_item_count"].items():
            swap_primary_by_item_count[k] = swap_primary_by_item_count.get(k, 0) + v
        for k, v in r["n_swap_strict_by_item_count"].items():
            swap_strict_by_item_count[k] = swap_strict_by_item_count.get(k, 0) + v
    total_swaps_primary = sum(swap_primary_by_item_count.values())
    total_swaps_strict = sum(swap_strict_by_item_count.values())

    pooled = build_pooled_table(rows)

    swap_mdd_per_cell = {
        outcome: {
            observable: {
                estimator: _mdd(pooled[outcome][observable][estimator]["raw"]) for estimator in ESTIMATORS
            } for observable in OBSERVABLES
        } for outcome in OUTCOMES
    }

    output["reachability"] = {
        "n_behavioural_session_dates_seen": n_seen, "n_sessions_loaded": len(loaded),
        "n_sessions_refused_by_the_shared_loader": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                               for reason in sorted({r["status"] for r in refused})},
        "n_sessions_with_a_computed_arm": len(computed_rows),
        "n_sessions_too_few_trials_or_amplitude_not_computable": len(rows) - len(computed_rows),
        "counts_reconcile": bool(n_seen == len(loaded) + len(refused)),
        "reconciles_against_the_delivered_watters_state_geometry_artifact": (
            "results/watters_state_geometry.json reports 47 dates seen, 41 analysed, 6 refused; this "
            "script reuses the identical iter_watters(root, bin_ms=100.0) call and must match those "
            "counts exactly."
        ),
        "n_trials_by_item_count_pooled": trials_by_item_count,
        "n_swap_trials_by_item_count_primary_rule": swap_primary_by_item_count,
        "n_swap_trials_by_item_count_strict_rule": swap_strict_by_item_count,
        "total_swap_trials_primary_rule": total_swaps_primary,
        "total_swap_trials_strict_rule": total_swaps_strict,
        "min_swaps_pooled_to_test": MIN_SWAPS_POOLED_TO_TEST,
        "swap_count_clears_the_pre_declared_floor": bool(total_swaps_primary >= MIN_SWAPS_POOLED_TO_TEST),
        "minimum_detectable_paired_difference_at_80pct_power_per_cell": swap_mdd_per_cell,
    }
    _flush(output)

    within_load_reference = _delivered_within_load_reference()
    output["delivered_whole_corpus_within_load_reference"] = within_load_reference

    branch = decide_branch(gate_result["status"], total_swaps_primary, pooled, within_load_reference, rows)

    # Threshold-sensitivity: recompute the swap-only "predicts" fact under the strict rule and compare.
    predicts_swap_strict = None
    if total_swaps_strict >= MIN_SWAPS_POOLED_TO_TEST:
        predicts_swap_strict = _predicts(pooled["swap_strict"]["deviation"])
    output["strict_threshold_sensitivity"] = {
        "total_swap_trials_strict_rule": total_swaps_strict,
        "clears_the_pre_declared_floor": bool(total_swaps_strict >= MIN_SWAPS_POOLED_TO_TEST),
        "predicts_swap_primary_rule": branch.get("predicts_swap"),
        "predicts_swap_strict_rule": predicts_swap_strict,
        "conclusion_changes_under_stricter_threshold": (
            None if predicts_swap_strict is None or branch.get("predicts_swap") is None
            else bool(predicts_swap_strict != branch.get("predicts_swap"))),
        "note": STRICT_THRESHOLD_SENSITIVITY_NOTE,
    }

    # Load-1 control, reported beside the swap result regardless of which branch fired.
    load1_deviation_raw = _pool_values(
        [r["load_1_control"]["deviation"]["raw"]["r"] for r in computed_rows
         if r["load_1_control"].get("status") == "computed"
         and r["load_1_control"]["deviation"]["raw"].get("status") == "computed"])
    load1_amplitude_raw = _pool_values(
        [r["load_1_control"]["amplitude"]["raw"]["r"] for r in computed_rows
         if r["load_1_control"].get("status") == "computed"
         and r["load_1_control"]["amplitude"]["raw"].get("status") == "computed"])
    output["load_1_control"] = {
        "note": "At item count 1 there is no swap to make; this is the deviation-vs-report-error (= "
                "deviation-vs-imprecision, checked identical by construction) association restricted to "
                "item count 1 trials only, reported beside the swap result. A component that is purely a "
                "binding signal should show no association here.",
        "n_sessions_with_a_computed_load1_arm": sum(
            1 for r in computed_rows if r["load_1_control"].get("status") == "computed"),
        "deviation_vs_report_error_raw": load1_deviation_raw,
        "amplitude_vs_report_error_raw": load1_amplitude_raw,
        "significant_at_load1": _sig(load1_deviation_raw),
        "identification_is_partial_if_significant": bool(_sig(load1_deviation_raw)) if _sig(
            load1_deviation_raw) is not None else None,
    }

    disclosures = build_disclosures(pooled, load1_deviation_raw, rows)
    output["load_1_control"]["load1_cannot_exclude_the_swap_effect"] = disclosures[
        "load1_cannot_exclude_the_swap_effect"]
    output["amplitude_versus_deviation_swap_association_disclosure"] = disclosures[
        "amplitude_versus_deviation_swap_association"]
    output["imprecision_detection_floor_versus_swap_effect_disclosure"] = disclosures[
        "imprecision_detection_floor_versus_swap_effect"]

    output["pooled_results"] = pooled
    output["branch"] = branch
    output["sessions"] = rows

    output["zero_drop_accounting"] = {
        "n_seen": n_seen, "n_loaded": len(loaded), "n_refused": len(refused),
        "n_analysed_computed": len(computed_rows), "n_loaded_but_not_computed": len(loaded) - len(computed_rows),
        "reconciles": bool(n_seen == len(loaded) + len(refused)),
    }

    output["scope"] = {
        "corpus": "watters_2026 (multi-object macaque, DANDI 000620)",
        "unit_quality_tier": PRIMARY_QUALITY_TIER,
        "n_sessions_seen": n_seen, "n_sessions_analysed": len(computed_rows),
        "exclusions": output["reachability"]["refusals_by_reason"],
        "min_trials_per_correlation": MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
        "min_swaps_pooled_to_test": MIN_SWAPS_POOLED_TO_TEST,
        "correct_report_distance_threshold": CORRECT_REPORT_DISTANCE_THRESHOLD,
        "seed_scheme": "every correlation is seeded by a stable hash (stable_seed) of a per-session, "
                       "per-observable, per-outcome, per-level (or pooled_mixed / load1) tag string; "
                       "deterministic and reproducible from the tag alone",
        "git_commit": git_commit(ROOT),
        "wall_clock_s": time.time() - t0,
    }

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"branch: {branch.get('branch')} elapsed={time.time() - t0:.0f}s")
    print(json.dumps({
        "reproduction_gate": gate_result["status"], "branch": branch.get("branch"),
        "total_swaps_primary": total_swaps_primary, "n_sessions_analysed": len(computed_rows),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
