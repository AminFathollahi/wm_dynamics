"""run_component_and_content_specific_serial_pull.py -- tests the rate-free
direction-deviation observable's behaviour link against the one interference
quantity it was never actually tested against.

results/deviation_serial_dependence_and_temporal_locus.json established two
facts about the same per-trial direction without joining them. First, a
content-specific serial pull is present: for trials whose memorandum class
differs from the previous trial's, the direction is pulled toward the
previous trial's class mean more than toward other classes' means. Second,
the deviation-to-behaviour link was declared separable from trial history --
but the covariate that decision controlled for was the plain, undirected
cosine between consecutive trials' directions, not the directed,
class-mean-referenced content-specific pull the same artifact shows is
present. This module runs the missing test: does the deviation-to-behaviour
link survive controlling for the content-specific pull itself, and, in the
reverse direction, does the content-specific pull's own link to behaviour
survive controlling for the deviation.

THE DEVIATION-SURVIVAL TEST controls the deviation-to-behaviour correlation by the
content-specific pull (alongside the delivered consecutive-trial-alignment
control, recomputed on the same trial subset so the two are commensurable),
then directly tests, paired within session, whether that control removes
more of the link than the delivered one does.

THE CONTENT-PULL-SURVIVAL TEST is the mirror image: does the content-specific pull's own raw link
to behaviour survive controlling for the deviation, with a direct paired
test of the two raw correlations to order them.

Scope is the same precondition the delivered artifact already measured: the
single-item macaque lateral prefrontal cortex corpus and the multi-object
macaque corpus, the only two whose rate-free deviation passes its own
orthogonality gate against total spike count. The multi-object corpus is
analysed WITHIN item-count level throughout and combined by the identical
trial-count-weighted average that corpus's own primary estimator already
uses; a pooled-across-item-count number is never treated as this corpus's
effect size.

SIGN CONVENTION. Every behavioural correlation reported here is expressed
against WORSE behaviour, applied once at the point each result is packaged,
never inside an estimator: the single-item corpus's native outcome is trial
correctness and is negated once; the multi-object corpus's native outcome is
continuous report error and needs no flip.

No estimator is forked. unit_direction_vectors, _detrend, _cosine_at_lag,
DETREND_WINDOWS_TRIALS, MIN_TRIALS_FOR_LAG_PROFILE,
CONTENT_SPECIFIC_SERIAL_PULL_OPERATIONALISATION, SHARP_TEST_MIN_CLASSES,
SHARP_TEST_MIN_PER_CLASS, SIGN_TO_WORSE_BEHAVIOUR, _sign_to_worse_behaviour,
_macaque_session_bundle, full_reproduction_gate, MIN_TRIALS_WITH_DEFINED_
DIRECTION, _reachable_sessions, _panichello_directory, usable_label,
iter_watters, data_root, partial_correlation_permutation_test,
slope_across_sessions_test, minimum_detectable_paired_difference,
permutation_pvalue, stable_seed and _trial_count_weighted are every one
imported unchanged from where this project already defines them. The only
new functions this module introduces are the per-trial (rather than
session-aggregated) content-specific pull, the within-item-count-level
combination of the eight named correlation statistics both blocks need, the
bias-only and trial-order-shuffle controls, and the two blocks' decision-cell
classifiers.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    CONTENT_LABEL_K_CLASSES, CONTENT_SPECIFIC_SERIAL_PULL_OPERATIONALISATION, DETREND_WINDOWS_TRIALS,
    MIN_TRIALS_FOR_LAG_PROFILE, SHARP_TEST_MIN_CLASSES, SHARP_TEST_MIN_PER_CLASS, SIGN_TO_WORSE_BEHAVIOUR,
    _cosine_at_lag, _detrend, _macaque_session_bundle, full_reproduction_gate, unit_direction_vectors,
)
from run_multi_object_interference_and_locus_within_item_count import _trial_count_weighted  # noqa: E402
from run_behavior_amplitude_rate_controls import _reachable_sessions  # noqa: E402
from run_state_behavior_link import _panichello_directory  # noqa: E402
from run_state_content_link import usable_label  # noqa: E402
from run_watters_state_geometry import MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, permutation_pvalue, stable_seed,
)
from state_persistence import slope_across_sessions_test  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "component_and_content_specific_serial_pull.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "component_and_content_specific_serial_pull_checkpoint.json"
SERIAL_DEPENDENCE_ARTIFACT_PATH = ROOT / "results" / "deviation_serial_dependence_and_temporal_locus.json"
ANALYSIS_VERSION = "2026-08-19"

N_PERM = 10000
REPRODUCTION_TOLERANCE = 1e-6
N_SHUFFLES_PER_SESSION = 1000
COUNT_PRECONDITION_MIN_TRIALS = 300
PRIMARY_WINDOW = DETREND_WINDOWS_TRIALS[0]

# -------------------------------------------------------------------------------------------------------
# Decision rules, declared before any fit runs, and the count precondition, declared before any count is
# looked at. Both blocks' named outcome cells are read off these strings, not re-derived at branch time.
# -------------------------------------------------------------------------------------------------------

COUNT_PRECONDITION_RULE = (
    f"content_pull(t) is defined only on trials whose memorandum class differs from their predecessor's. "
    f"Per corpus, the pooled count of trials with a defined content_pull, a defined deviation and a "
    f"defined behavioural outcome is computed BEFORE either block runs. If that count is below "
    f"{COUNT_PRECONDITION_MIN_TRIALS}, the corpus's branch for both blocks is "
    f"'too_few_trials_with_a_defined_content_specific_pull_to_test', reported with the count, and neither "
    f"block reads a further number for that corpus."
)

DEVIATION_SURVIVAL_DECISION_RULE = (
    "Per corpus, at the primary detrending window (the first of DETREND_WINDOWS_TRIALS), on the trials "
    "where deviation, behaviour, the delivered consecutive-trial-alignment covariate and the "
    "content-specific pull are all defined: 'raw' is deviation against behaviour with no controls; "
    "'partial_controlling_consecutive_trial_alignment' is the delivered control, recomputed on this same "
    "trial subset so it is commensurable; 'partial_controlling_content_specific_pull' is the same "
    "statistic controlling content_pull(t) instead; 'joint_partial_controlling_both_with_spike_count_and_"
    "trial_index' controls both together with total spike count and trial index. Each is formed within "
    "item-count level (for the multi-object corpus) or on the whole session (for the single-item corpus) "
    "and pooled across sessions by the shared paired sign-flip test, with a direct paired session-level "
    "test of partial_controlling_content_specific_pull against partial_controlling_consecutive_trial_"
    "alignment on the same sessions. 'Significant, same sign as raw' means the pooled partial's two-sided "
    "p is <= 0.05 and its mean value has the same sign as the pooled raw mean value. Named cells:\n"
    "  - the content-pull partial is significant with the same sign as raw -> "
    "'accuracy_predicting_component_survives_the_content_specific_serial_pull', reported with raw and "
    "partial side by side; strengthens the delivered separability verdict, does not replace it.\n"
    "  - the content-pull partial is not significant (by the above test) while raw is, AND the paired "
    "test of the two partials is significant -> "
    "'accuracy_predicting_component_is_the_content_specific_serial_pull', a positive identification, "
    "reported with raw, both partials and the paired test together.\n"
    "  - the content-pull partial is not significant, raw is, and the paired test is NOT significant -> "
    "'content_specific_pull_control_removes_the_link_but_not_more_than_consecutive_trial_alignment_does', "
    "no ordering stated, reported with the paired test's minimum detectable difference.\n"
    "  - both partials are significant with the same sign, the paired test is NOT significant, and its "
    "minimum detectable difference lies BELOW the raw effect size -> "
    "'powered_null_the_two_controls_are_not_distinguishable'.\n"
    "  - the paired test is not significant and its minimum detectable difference is AT OR ABOVE the raw "
    "effect size -> 'inconclusive_below_detection_floor', never quoted without that comparison stated "
    "numerically.\n"
    "These cells are evaluated in the listed order, first match wins, the same convention every other "
    "decision-cell classifier in this project follows; a combination none of them covers is reported as "
    "'deviation_survival_outcome_not_covered_by_the_pre_declared_rule' with every number, never forced "
    "onto the nearest label. Both detrending windows are computed; if they disagree, both are reported "
    "and the disagreement is stated, not resolved by picking one."
)

CONTENT_PULL_SURVIVAL_DECISION_RULE = (
    "Runs whenever the count precondition passed. The identical machinery with the two observables "
    "exchanged, at the primary detrending window: 'raw_content_pull_to_behaviour' is content_pull(t) "
    "against behaviour with no controls; 'partial_controlling_the_deviation' controls the deviation; "
    "'joint_partial_controlling_the_deviation_spike_count_and_trial_index' adds total spike count and "
    "trial index; plus the across-item-count-level collinearity between the deviation and content_pull "
    "themselves, and a direct paired session-level test of the deviation-survival test's raw (deviation-to-behaviour) against "
    "the content-pull-survival test's raw (content-pull-to-behaviour) on the same sessions. Let A_survives mean the deviation-survival test's "
    "content-pull partial is significant with the same sign as the deviation-survival test's raw, and B_survives mean "
    "this test's own deviation partial is significant with the same sign as the content-pull-survival test's raw. Named cells:\n"
    "  - A_survives, not B_survives, and the paired test of the two raws is significant -> "
    "'the_deviation_carries_the_behavioural_link_and_the_serial_pull_does_not'.\n"
    "  - B_survives, not A_survives, and the paired test is significant -> "
    "'the_serial_pull_carries_the_behavioural_link_and_the_deviation_does_not' (the mirror image).\n"
    "  - both A_survives and B_survives -> 'both_observables_carry_it_independently', reported with the "
    "collinearity and resolved in favour of neither.\n"
    "  - neither A_survives nor B_survives -> 'neither_observable_survives_the_other', reported with both "
    "minimum detectable differences, stated as not separable at this power.\n"
    "  - A_survives xor B_survives, but the paired test of the two raws is NOT significant -> "
    "'ordering_not_established_by_a_paired_test', no ordering stated.\n"
    "Evaluated in the listed order, first match wins; a combination none of them covers is reported as "
    "'content_pull_survival_outcome_not_covered_by_the_pre_declared_rule' with every number."
)

MANDATORY_CONTROLS_NOTE = (
    "Five controls are computed before any branch above is read: (1) a bias-only control that replaces "
    "each trial's content_pull with its own session's mean content_pull and recomputes the content-pull "
    "partial -- if that reproduces the real result's significance and sign, the branch is void by name, "
    "'content_pull_control_not_separable_from_a_session_level_offset'; (2) a within-session trial-order "
    "shuffle null of at least 1000 draws per session, recomputed end to end (detrending and the "
    "content-specific pull both included) at the primary window, since content_pull(t) does not itself "
    "depend on the detrending window; (3) both detrending windows, disagreement stated rather than "
    "resolved; (4) a reproduction gate re-running the delivered estimators on the delivered sessions and "
    "matching the recorded values in results/deviation_serial_dependence_and_temporal_locus.json at "
    f"tolerance {REPRODUCTION_TOLERANCE}, before any new number is read -- failure voids the run; (5) "
    "zero-drop accounting of every session seen, loaded, refused, computed and excluded, reconciling "
    "exactly and against the delivered artifact's own counts."
)


# =======================================================================================================
# Checkpointing (fit-level; temp file + os.replace; completion flag written only after the fit returns)
# =======================================================================================================

_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    # Array restoration on reload -- undoing the plain-list flattening _json_safe applies
    # to every numpy array before a checkpoint entry is written -- now lives once in
    # provenance.restore_checkpoint rather than as this module's own copy, so every
    # analysis script that checkpoints fits shares the identical repair.
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
# Per-trial content-specific serial pull -- the identical formula
# CONTENT_SPECIFIC_SERIAL_PULL_OPERATIONALISATION already describes and the delivered
# _content_specific_serial_pull already computes, kept per-trial here instead of collapsed to a session
# mean, since both blocks need the per-trial value to correlate against deviation and behaviour.
# =======================================================================================================

def _content_specific_pull_per_trial(activity_by_unit: np.ndarray, memorandum_label: np.ndarray) -> np.ndarray:
    vectors = unit_direction_vectors(activity_by_unit)
    label = memorandum_label
    n = vectors.shape[0]
    out = np.full(n, np.nan)
    finite_label = np.isfinite(label)
    if not finite_label.any():
        return out
    ok, _reason, _mask = usable_label(
        label[finite_label], min_classes=SHARP_TEST_MIN_CLASSES, min_per_class=SHARP_TEST_MIN_PER_CLASS)
    if not ok:
        return out
    classes = np.unique(label[finite_label])
    class_members = {c: np.flatnonzero((label == c) & finite_label) for c in classes}
    class_mean = {c: vectors[idx].mean(axis=0) for c, idx in class_members.items() if len(idx) >= SHARP_TEST_MIN_PER_CLASS}
    eligible_classes = set(class_mean)
    if len(eligible_classes) < SHARP_TEST_MIN_CLASSES:
        return out
    for t in range(1, n):
        if not (finite_label[t] and finite_label[t - 1]):
            continue
        own, prev = label[t], label[t - 1]
        if own == prev or prev not in eligible_classes:
            continue
        other_classes = [c for c in eligible_classes if c != own and c != prev]
        if not other_classes:
            continue
        prev_mean = class_mean[prev]
        prev_norm = np.linalg.norm(prev_mean)
        if prev_norm == 0.0:
            continue
        pull_prev = float(np.dot(vectors[t], prev_mean / prev_norm))
        other_pulls = []
        for c in other_classes:
            m = class_mean[c]
            mn = np.linalg.norm(m)
            if mn > 0.0:
                other_pulls.append(float(np.dot(vectors[t], m / mn)))
        if not other_pulls:
            continue
        out[t] = pull_prev - float(np.mean(other_pulls))
    return out


def _bias_only_content_pull(content_pull: np.ndarray) -> np.ndarray:
    """Replaces every finite per-trial content_pull with the session's own mean over its finite trials --
    collapsing all per-trial variation to one session-level offset. If the real result reproduces under
    this replacement, whatever the content-pull control removes is that offset, not genuine per-trial
    content-specific structure."""
    finite = np.isfinite(content_pull)
    if not finite.any():
        return content_pull
    mean_val = float(np.mean(content_pull[finite]))
    return np.where(finite, mean_val, np.nan)


def _partial_r(outcome: np.ndarray, covariate: np.ndarray, controls: list[np.ndarray]) -> float | None:
    """The observed-value half of statistics.partial_correlation_permutation_test's residual-then-
    correlate formula, without that function's own significance permutation -- used only to build the
    1000-draw trial-order shuffle null's null distribution, where re-running the full N_PERM=10000
    significance permutation on every draw would be computationally intractable. Mirrors that function's
    formula exactly, so the null measures the identical statistic the real value is."""
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    control_list = [np.asarray(c, dtype=float) for c in controls]
    n = len(outcome)
    if control_list:
        design = np.column_stack([np.ones(n), *control_list])
        y_resid = outcome - design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
        x_resid = covariate - design @ np.linalg.lstsq(design, covariate, rcond=None)[0]
    else:
        y_resid, x_resid = outcome - outcome.mean(), covariate - covariate.mean()
    if np.std(y_resid) == 0.0 or np.std(x_resid) == 0.0:
        return None
    return float(np.corrcoef(y_resid, x_resid)[0, 1])


# =======================================================================================================
# Session bundles. Macaque reuses _macaque_session_bundle unchanged. Watters combines the two fields no
# single delivered bundle carries together: _observable_arrays' within-item-count fields (item_count,
# deviation, report_error, spike_count, trial_index) from the reproduction gate's own arrays, and the
# memorandum-class discretisation of cued_theta the delivered module's _watters_session_bundle computes
# -- read live off the reproduction gate's own CONTENT_LABEL_K_CLASSES-width bins, not re-derived here.
# =======================================================================================================

def _watters_bundle_with_label(entry: dict, content_label_k_classes: int) -> dict:
    session, arrays, usable = entry["session"], entry["arrays"], entry["usable"]
    counts = session["counts"]
    activity_by_unit = counts.sum(axis=2)[usable]
    theta = np.mod(np.asarray(session["cued_theta"], dtype=float)[usable], 2.0 * np.pi)
    label = (np.floor(theta / (2.0 * np.pi / content_label_k_classes)).astype(int)) % content_label_k_classes
    return {
        "session": session["session"], "activity_by_unit": activity_by_unit, "deviation": arrays["deviation"],
        "outcome_raw": arrays["report_error"], "spike_count": arrays["spike_count"],
        "trial_index": arrays["trial_index"], "memorandum_label": label.astype(float),
        "item_count": arrays["item_count"],
    }


# =======================================================================================================
# Per-session, per-window statistics -- the eight named correlations both blocks read, formed within
# item-count level (multi-object corpus) or on the whole session (single-item corpus, has_levels=False)
# and combined by the identical trial-count-weighted average the multi-object corpus's own primary
# estimator already uses.
# =======================================================================================================

BEHAVIOUR_KEYS = (
    "raw", "partial_controlling_consecutive_trial_alignment", "partial_controlling_content_specific_pull",
    "joint_partial_controlling_both_with_spike_count_and_trial_index",
    "raw_content_pull_to_behaviour", "partial_controlling_the_deviation",
    "joint_partial_controlling_the_deviation_spike_count_and_trial_index",
)
COLLINEARITY_KEY = "collinearity_deviation_vs_content_pull"


def _level_split_stats(outcome: np.ndarray, deviation: np.ndarray, content_pull: np.ndarray,
                        lag1_align: np.ndarray, spike_count: np.ndarray, trial_index: np.ndarray,
                        item_count: np.ndarray | None, sign: float, seed_prefix: str) -> dict:
    if item_count is None:
        level_masks = {"all": np.ones(len(outcome), dtype=bool)}
    else:
        levels = sorted({int(v) for v in item_count.tolist()})
        level_masks = {str(lv): (item_count == float(lv)) for lv in levels}

    per_level: dict[str, dict] = {}
    for level_key, mask in level_masks.items():
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            per_level[level_key] = {"status": "too_few_trials_at_this_item_count_level", "n_trials": n_level}
            continue
        o, d, cp, la = outcome[mask], deviation[mask], content_pull[mask], lag1_align[mask]
        sc, ti = spike_count[mask], trial_index[mask]
        tag = f"{seed_prefix}|level{level_key}"

        def _rng(name: str) -> np.random.Generator:
            return np.random.default_rng(stable_seed(f"{tag}|{name}"))

        per_level[level_key] = {
            "status": "computed", "n_trials": n_level,
            "raw": partial_correlation_permutation_test(o, d, [], N_PERM, _rng("raw")),
            "partial_controlling_consecutive_trial_alignment": partial_correlation_permutation_test(
                o, d, [la], N_PERM, _rng("ctrl_lag1")),
            "partial_controlling_content_specific_pull": partial_correlation_permutation_test(
                o, d, [cp], N_PERM, _rng("ctrl_cp")),
            "joint_partial_controlling_both_with_spike_count_and_trial_index": partial_correlation_permutation_test(
                o, d, [la, cp, sc, ti], N_PERM, _rng("ctrl_joint")),
            "raw_content_pull_to_behaviour": partial_correlation_permutation_test(o, cp, [], N_PERM, _rng("raw_cp")),
            "partial_controlling_the_deviation": partial_correlation_permutation_test(
                o, cp, [d], N_PERM, _rng("ctrl_dev")),
            "joint_partial_controlling_the_deviation_spike_count_and_trial_index": partial_correlation_permutation_test(
                o, cp, [d, sc, ti], N_PERM, _rng("ctrl_dev_joint")),
            COLLINEARITY_KEY: partial_correlation_permutation_test(d, cp, [], N_PERM, _rng("collinearity")),
        }

    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    within: dict[str, float | None] = {}
    for key in BEHAVIOUR_KEYS:
        tested = [(v["n_trials"], sign * v[key]["r"]) for v in per_level.values()
                  if v.get("status") == "computed" and v[key].get("status") == "computed"]
        within[key] = _trial_count_weighted(tested)
    tested_collin = [(v["n_trials"], v[COLLINEARITY_KEY]["r"]) for v in per_level.values()
                      if v.get("status") == "computed" and v[COLLINEARITY_KEY].get("status") == "computed"]
    within[COLLINEARITY_KEY] = _trial_count_weighted(tested_collin)

    return {
        "status": "computed" if n_levels_tested > 0 else "not_computable_no_item_count_level_reaches_the_floor",
        "n_levels_tested": n_levels_tested, "per_level": per_level, "within_item_count_level": within,
    }


def _session_content_pull_stats(bundle: dict, window: int, sign: float, has_levels: bool, seed_prefix: str,
                                 bias_only: bool = False) -> dict:
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

    content_pull_full = _content_specific_pull_per_trial(bundle["activity_by_unit"], bundle["memorandum_label"])
    if bias_only:
        content_pull_full = _bias_only_content_pull(content_pull_full)
    content_pull = content_pull_full[idx]
    cp_valid = np.isfinite(content_pull)
    idx2, lag1_align2, content_pull2 = idx[cp_valid], lag1_align[cp_valid], content_pull[cp_valid]
    if len(idx2) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_content_pull", "n_trials": len(idx2)}

    outcome = bundle["outcome_raw"][idx2]
    deviation = bundle["deviation"][idx2]
    spike_count = bundle["spike_count"][idx2]
    trial_index = bundle["trial_index"][idx2]
    item_count = bundle["item_count"][idx2] if has_levels else None

    tag = f"{seed_prefix}|w{window}"
    result = _level_split_stats(outcome, deviation, content_pull2, lag1_align2, spike_count, trial_index,
                                 item_count, sign, tag)
    result["n_trials"] = len(idx2)
    return result


# =======================================================================================================
# Pooling across sessions -- the shared paired sign-flip test (slope_across_sessions_test) plus minimum
# detectable difference at 80% power, applied to the within-item-count-level per-session scalars.
# =======================================================================================================

def _pool_within_level(per_session: list[dict]) -> dict:
    pooled: dict[str, dict] = {}
    for key in (*BEHAVIOUR_KEYS, COLLINEARITY_KEY):
        values = [s["within_item_count_level"][key] for s in per_session
                  if s.get("status") == "computed" and s["within_item_count_level"].get(key) is not None]
        p = slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}
        if len(values) >= 2:
            p["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
        pooled[key] = p
    return pooled


def _paired_test(per_session: list[dict], key_a: str, key_b: str) -> dict:
    diffs = [s["within_item_count_level"][key_a] - s["within_item_count_level"][key_b] for s in per_session
             if s.get("status") == "computed" and s["within_item_count_level"].get(key_a) is not None
             and s["within_item_count_level"].get(key_b) is not None]
    pooled = slope_across_sessions_test(diffs, alternative="two-sided") if diffs else {"status": "not_computed"}
    if len(diffs) >= 2:
        pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(diffs)
    pooled["n_sessions_paired"] = len(diffs)
    return pooled


def _bias_only_reproduces(real: dict, bias: dict) -> bool:
    if real.get("status") != "tested" or bias.get("status") != "tested":
        return False
    if real["significant"] != bias["significant"]:
        return False
    if real["significant"] and (real["mean_value"] > 0.0) != (bias["mean_value"] > 0.0):
        return False
    return True


def _bias_only_branch(reproduces: bool) -> str:
    """The pre-declared voiding name fires exactly when collapsing every trial's content_pull to its
    session's own mean reproduces the real content-pull partial's significance and sign -- see
    MANDATORY_CONTROLS_NOTE. Kept as its own function (mirrors _sign_state/_survives above) so the exact
    literal branch string is directly testable rather than only reachable through the full driver."""
    return ("content_pull_control_not_separable_from_a_session_level_offset" if reproduces
            else "content_pull_control_is_not_a_session_level_offset")


# =======================================================================================================
# Within-session trial-order shuffle null of the primary partial (partial_controlling_content_specific_
# pull), recomputed end to end -- detrending, lag-1 alignment and the per-trial content-specific pull are
# all order-dependent and so are all recomputed on the shuffled order; class-mean references are a
# property of class MEMBERSHIP, not order, and are therefore identical under any permutation, computed
# implicitly by the same per-trial formula on the shuffled label array. Uses the cheap _partial_r value
# (no inner significance permutation) since the real value's own significance is already established by
# _level_split_stats; running the full N_PERM=10000 permutation test on every one of 1000 draws per
# session would be computationally intractable.
# =======================================================================================================

def _shuffle_draw_partial_cp_value(bundle: dict, window: int, has_levels: bool, sign: float,
                                    rng: np.random.Generator) -> float | None:
    n = bundle["activity_by_unit"].shape[0]
    perm = rng.permutation(n)
    activity = bundle["activity_by_unit"][perm]
    label = bundle["memorandum_label"][perm]
    outcome = bundle["outcome_raw"][perm]
    deviation = bundle["deviation"][perm]
    item_count = bundle["item_count"][perm] if has_levels else None

    vectors = unit_direction_vectors(activity)
    detrended = _detrend(vectors, window)
    lag1_align = _cosine_at_lag(detrended, 1)
    idx = np.arange(1, n)
    valid = np.isfinite(lag1_align)
    idx = idx[valid]
    if len(idx) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None

    content_pull_full = _content_specific_pull_per_trial(activity, label)
    content_pull = content_pull_full[idx]
    cp_valid = np.isfinite(content_pull)
    idx2 = idx[cp_valid]
    if len(idx2) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None

    o, d, cp = outcome[idx2], deviation[idx2], content_pull_full[idx2]
    # sign is applied here, at the same point _level_split_stats applies it to the real (non-shuffled)
    # draw, so every draw in the null distribution is on the identical against-worse-behaviour convention
    # the real pooled value is packaged in -- required for null_mean/null_sd to be read on the same
    # convention as real_value, even though the two-sided |null| >= |real| comparison the empirical
    # p-value uses is itself sign-invariant.
    if item_count is None:
        r = _partial_r(o, d, [cp])
        return sign * r if r is not None else None
    ic = item_count[idx2]
    levels = sorted({int(v) for v in ic.tolist()})
    tested = []
    for level in levels:
        mask = ic == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            continue
        r = _partial_r(o[mask], d[mask], [cp[mask]])
        if r is not None:
            tested.append((n_level, r))
    combined = _trial_count_weighted(tested)
    return sign * combined if combined is not None else None


def _session_shuffle_draws(bundle: dict, window: int, has_levels: bool, sign: float, seed_prefix: str,
                            n_shuffles: int) -> list[float | None]:
    draws = []
    for d in range(n_shuffles):
        rng = np.random.default_rng(stable_seed(f"{seed_prefix}|draw{d}"))
        draws.append(_shuffle_draw_partial_cp_value(bundle, window, has_levels, sign, rng))
    return draws


def _pool_shuffle_null(per_session_draws: list[list[float | None]], real_value: float | None, n_shuffles: int) -> dict:
    fully = [d for d in per_session_draws if len(d) == n_shuffles and all(v is not None for v in d)]
    if real_value is None or not fully:
        return {"status": "not_computable", "n_sessions_contributing": len(fully), "n_sessions_total": len(per_session_draws)}
    pooled_null = np.array([np.mean([d[i] for d in fully]) for i in range(n_shuffles)])
    p_value = float(permutation_pvalue(np.abs(pooled_null) >= abs(real_value)))
    return {
        "status": "computed", "n_sessions_contributing": len(fully), "n_sessions_total": len(per_session_draws),
        "n_draws": n_shuffles, "null_mean": float(np.mean(pooled_null)), "null_sd": float(np.std(pooled_null, ddof=1)),
        "real_value": real_value, "two_sided_empirical_p_value": p_value, "significant": bool(p_value <= 0.05),
    }


# =======================================================================================================
# the deviation-survival test / the content-pull-survival test decision-cell classifiers, evaluated in the order declared in DEVIATION_SURVIVAL_DECISION_RULE /
# CONTENT_PULL_SURVIVAL_DECISION_RULE, first match wins -- the same convention every other decision-cell classifier in
# this project follows.
# =======================================================================================================

def _sign_state(raw: dict, partial: dict) -> str:
    """Three-way read of one partial correlation against its own raw correlation:
    'not_tested' (either side never reached a testable sample), 'not_significant',
    'same_sign_significant' (this project's standing sense of "survives"), or
    'opposite_sign_significant'. Kept distinct from a plain 'does it survive' boolean because the
    pre-declared cells below are worded in terms of a partial that is genuinely NOT significant --
    a partial that IS significant but with the sign flipped relative to raw is a different situation
    neither 'survives' nor 'not significant' should silently absorb."""
    if raw.get("status") != "tested" or partial.get("status") != "tested":
        return "not_tested"
    if not partial["significant"]:
        return "not_significant"
    if (partial["mean_value"] > 0.0) == (raw["mean_value"] > 0.0):
        return "same_sign_significant"
    return "opposite_sign_significant"


def _survives(raw: dict, partial: dict) -> bool:
    return _sign_state(raw, partial) == "same_sign_significant"


def _deviation_survival_branch(pooled: dict, paired: dict) -> dict:
    raw = pooled["raw"]
    partial_cp = pooled["partial_controlling_content_specific_pull"]
    partial_lag1 = pooled["partial_controlling_consecutive_trial_alignment"]
    if raw.get("status") != "tested":
        return {"branch": "not_computable", "reason": "raw_deviation_to_behaviour_result_not_tested"}

    cp_state = _sign_state(raw, partial_cp)
    if cp_state == "same_sign_significant":
        return {"branch": "accuracy_predicting_component_survives_the_content_specific_serial_pull"}

    raw_significant = bool(raw["significant"])
    paired_significant = bool(paired.get("status") == "tested" and paired.get("significant"))
    raw_effect = abs(raw["mean_value"])
    mdd_entry = paired.get("minimum_detectable_paired_difference_at_80pct_power")
    mdd = mdd_entry.get("mdd") if isinstance(mdd_entry, dict) and mdd_entry.get("status") == "computed" else None
    lag1_state = _sign_state(raw, partial_lag1)

    # The content-pull partial being NOT significant (cells below) is a different state from it being
    # significant with the sign flipped relative to raw (only cell 4 -- "both partials significant" --
    # is worded to admit a significant content-pull partial, and only when it is reached is a significant
    # partial not already claimed by the survives check above).
    if raw_significant and cp_state == "not_significant" and paired_significant:
        return {"branch": "accuracy_predicting_component_is_the_content_specific_serial_pull"}
    if raw_significant and cp_state == "not_significant" and not paired_significant:
        return {"branch": "content_specific_pull_control_removes_the_link_but_not_more_than_consecutive_trial_alignment_does"}
    if (raw_significant and cp_state == "opposite_sign_significant" and lag1_state == "opposite_sign_significant"
            and not paired_significant and mdd is not None and mdd < raw_effect):
        return {"branch": "powered_null_the_two_controls_are_not_distinguishable"}
    if raw_significant and not paired_significant and mdd is not None and mdd >= raw_effect:
        return {"branch": "inconclusive_below_detection_floor"}
    return {"branch": "deviation_survival_outcome_not_covered_by_the_pre_declared_rule",
            "diagnostics": {"raw_significant": raw_significant, "content_pull_partial_state": cp_state,
                             "consecutive_trial_alignment_partial_state": lag1_state,
                             "paired_significant": paired_significant,
                             "minimum_detectable_difference": mdd, "raw_effect": raw_effect}}


def _content_pull_survival_branch(pooled: dict, paired_raws: dict) -> dict:
    raw_a = pooled["raw"]
    partial_cp = pooled["partial_controlling_content_specific_pull"]
    raw_b = pooled["raw_content_pull_to_behaviour"]
    partial_dev = pooled["partial_controlling_the_deviation"]
    if raw_a.get("status") != "tested" or raw_b.get("status") != "tested":
        return {"branch": "not_computable"}

    a_survives = _survives(raw_a, partial_cp)
    b_survives = _survives(raw_b, partial_dev)
    paired_significant = bool(paired_raws.get("status") == "tested" and paired_raws.get("significant"))

    if a_survives and not b_survives and paired_significant:
        return {"branch": "the_deviation_carries_the_behavioural_link_and_the_serial_pull_does_not"}
    if b_survives and not a_survives and paired_significant:
        return {"branch": "the_serial_pull_carries_the_behavioural_link_and_the_deviation_does_not"}
    if a_survives and b_survives:
        return {"branch": "both_observables_carry_it_independently"}
    if not a_survives and not b_survives:
        return {"branch": "neither_observable_survives_the_other"}
    if not paired_significant:
        return {"branch": "ordering_not_established_by_a_paired_test"}
    return {"branch": "content_pull_survival_outcome_not_covered_by_the_pre_declared_rule",
            "diagnostics": {"a_survives": a_survives, "b_survives": b_survives, "paired_significant": paired_significant}}


# =======================================================================================================
# Count precondition -- computed before any other number for the corpus is read.
# =======================================================================================================

def _count_precondition(bundles: list[dict], corpus_key: str) -> dict:
    per_session = []
    total = 0
    for b in bundles:
        cp = _content_specific_pull_per_trial(b["activity_by_unit"], b["memorandum_label"])
        defined = np.isfinite(cp) & np.isfinite(b["deviation"]) & np.isfinite(b["outcome_raw"])
        n_defined = int(defined.sum())
        per_session.append({"session": b["session"], "n_trials_with_defined_content_pull_deviation_and_outcome": n_defined})
        total += n_defined
    return {
        "corpus": corpus_key, "n_sessions": len(bundles),
        "total_trials_with_defined_content_pull_deviation_and_outcome": total,
        "per_session": per_session, "meets_precondition": bool(total >= COUNT_PRECONDITION_MIN_TRIALS),
    }


# =======================================================================================================
# the deviation-survival test / the content-pull-survival test drivers
# =======================================================================================================

def _compute_decisive_all_sessions(bundles: list[dict], sign: float, has_levels: bool, corpus_key: str,
                                    seed_prefix: str) -> dict[int, list[dict]]:
    decisive: dict[int, list[dict]] = {w: [] for w in DETREND_WINDOWS_TRIALS}
    for bundle in bundles:
        tag = f"{seed_prefix}|{bundle['session']}"
        for window in DETREND_WINDOWS_TRIALS:
            dp = _fit(f"decisive|{corpus_key}|{bundle['session']}|w{window}",
                       lambda b=bundle, w=window, t=tag: _session_content_pull_stats(b, w, sign, has_levels, t))
            decisive[window].append(dp)
    return decisive


def run_deviation_survival_test(decisive: dict[int, list[dict]], bundles: list[dict], sign: float, has_levels: bool,
                 corpus_key: str, seed_prefix: str) -> dict:
    decisive_pooled: dict[str, dict] = {}
    branch_by_window: dict[str, str] = {}
    for window in DETREND_WINDOWS_TRIALS:
        pooled = _pool_within_level(decisive[window])
        paired = _paired_test(decisive[window], "partial_controlling_content_specific_pull",
                               "partial_controlling_consecutive_trial_alignment")
        decisive_pooled[str(window)] = {"pooled": pooled, "paired_content_pull_vs_consecutive_trial_alignment": paired}
        branch_by_window[str(window)] = _deviation_survival_branch(pooled, paired)["branch"]

    primary_branch = branch_by_window[str(PRIMARY_WINDOW)]
    windows_agree = len(set(branch_by_window.values())) == 1

    bias_only_per_session = []
    for bundle in bundles:
        tag = f"{seed_prefix}|bias_only|{bundle['session']}"
        dp = _fit(f"bias_only|{corpus_key}|{bundle['session']}|w{PRIMARY_WINDOW}",
                   lambda b=bundle, t=tag: _session_content_pull_stats(b, PRIMARY_WINDOW, sign, has_levels, t, bias_only=True))
        bias_only_per_session.append(dp)
    bias_only_pooled = _pool_within_level(bias_only_per_session)
    real_primary_cp = decisive_pooled[str(PRIMARY_WINDOW)]["pooled"]["partial_controlling_content_specific_pull"]
    bias_reproduces = _bias_only_reproduces(real_primary_cp, bias_only_pooled["partial_controlling_content_specific_pull"])
    bias_only_control = {
        "pooled": bias_only_pooled, "reproduces_the_real_result": bias_reproduces,
        "branch": _bias_only_branch(bias_reproduces),
    }
    # The bias-only control is a gate, not a side note: if collapsing every trial's content_pull to its
    # session's own mean reproduces the real content-pull partial's significance and sign, whatever that
    # partial is removing is a between-session offset rather than genuine per-trial structure, and the
    # headline branch this corpus/window would otherwise report is void by that name -- the classifier
    # verdict computed above is kept, unmodified, in branch_by_window for transparency.
    branch_at_primary_window_before_bias_only_control = primary_branch
    if bias_reproduces:
        primary_branch = "content_pull_control_not_separable_from_a_session_level_offset"

    shuffle_draws = []
    for bundle in bundles:
        tag = f"{seed_prefix}|shuffle|{bundle['session']}"
        draws = _fit(f"shuffle|{corpus_key}|{bundle['session']}|w{PRIMARY_WINDOW}",
                     lambda b=bundle, t=tag: _session_shuffle_draws(b, PRIMARY_WINDOW, has_levels, sign, t, N_SHUFFLES_PER_SESSION))
        shuffle_draws.append(draws)
    real_value = real_primary_cp.get("mean_value") if real_primary_cp.get("status") == "tested" else None
    shuffle_null = _pool_shuffle_null(shuffle_draws, real_value, N_SHUFFLES_PER_SESSION)

    return {
        "decision_rule_declared_before_fitting": DEVIATION_SURVIVAL_DECISION_RULE,
        "detrend_windows_trials": list(DETREND_WINDOWS_TRIALS), "primary_window": PRIMARY_WINDOW,
        "n_sessions_total": len(bundles),
        "per_session_by_window": {str(w): decisive[w] for w in DETREND_WINDOWS_TRIALS},
        "decisive_partial_by_window": decisive_pooled,
        "branch_by_window": branch_by_window, "branch_at_primary_window": primary_branch,
        "windows_agree": windows_agree,
        "bias_only_control_at_primary_window": bias_only_control,
        "trial_order_shuffle_null_at_primary_window": {
            "note": (
                "content_pull(t) does not itself depend on the detrending window, so this control is "
                "computed once, at the primary window, and covers both windows' content-pull-side "
                "variation."
            ),
            **shuffle_null,
        },
    }


def run_content_pull_survival_test(decisive: dict[int, list[dict]]) -> dict:
    # Mandatory control: every verdict is computed at both declared detrending windows (the deviation
    # side of this test is window-dependent even though content_pull(t) itself is not); if the two
    # windows' branches disagree, both are reported and the disagreement is stated, not resolved by
    # picking one.
    pooled_by_window: dict[str, dict] = {}
    paired_by_window: dict[str, dict] = {}
    branch_by_window: dict[str, str] = {}
    branch_detail_by_window: dict[str, dict] = {}
    for window in DETREND_WINDOWS_TRIALS:
        per_session = decisive[window]
        pooled = _pool_within_level(per_session)
        paired_raws = _paired_test(per_session, "raw", "raw_content_pull_to_behaviour")
        branch = _content_pull_survival_branch(pooled, paired_raws)
        pooled_by_window[str(window)] = pooled
        paired_by_window[str(window)] = paired_raws
        branch_by_window[str(window)] = branch["branch"]
        branch_detail_by_window[str(window)] = branch

    primary_key = str(PRIMARY_WINDOW)
    windows_agree = len(set(branch_by_window.values())) == 1
    return {
        "decision_rule_declared_before_fitting": CONTENT_PULL_SURVIVAL_DECISION_RULE,
        "detrend_windows_trials": list(DETREND_WINDOWS_TRIALS), "primary_window": PRIMARY_WINDOW,
        "pooled_by_window": pooled_by_window, "paired_test_of_the_two_raws_by_window": paired_by_window,
        "branch_by_window": branch_by_window, "windows_agree": windows_agree,
        "pooled_at_primary_window": pooled_by_window[primary_key],
        "paired_test_of_the_two_raws": paired_by_window[primary_key],
        "collinearity_deviation_vs_content_pull": pooled_by_window[primary_key][COLLINEARITY_KEY],
        "branch": branch_by_window[primary_key], "branch_detail": branch_detail_by_window[primary_key],
    }


def _gate_result_for_run(computed_gate_result: dict, session_limit: int | None, watters_n_loaded: int,
                          watters_n_delivered: int) -> dict:
    """Packages the reproduction gate's own recomputation into the artifact's reproduction_gate field.
    A session limit truncates the multi-object corpus below the session count the delivered reference
    values were computed on, so a recomputation under the limit disagrees with those values by
    construction -- that is a resource limit, not evidence the analysis code drifted. The gate is a
    statement about the DATA the delivered reference was fit on, not about a smoke test's truncated
    slice of it, so it is not evaluated under a limit; it still runs, and still hard-stops the analysis
    blocks on non-reproduction (see main), only when no session limit is set."""
    if session_limit is None:
        return computed_gate_result
    return {
        "status": "not_evaluated_under_a_session_limit",
        "session_limit": session_limit,
        "watters_n_sessions_loaded_under_the_limit": watters_n_loaded,
        "watters_n_sessions_the_delivered_reference_values_were_computed_on": watters_n_delivered,
        "note": (
            "The reproduction gate compares a recomputation against reference values computed on the "
            "full delivered corpus. Under a session limit the recomputation runs on fewer sessions than "
            "that, so a mismatch is expected and is not read as non-reproduction. This is a "
            "resource-limit record, never a statement about the data or the analysis code; the gate runs "
            "for real, and still hard-stops on non-reproduction, only when no session limit is set."
        ),
        "diagnostic_gate_result_computed_under_the_limit_not_a_verdict": computed_gate_result,
    }


def main(session_limit: int | None = None) -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "The two corpora whose rate-free deviation observable passes its own orthogonality gate "
            "against total spike count: the single-item macaque lateral prefrontal cortex corpus and the "
            "multi-object macaque corpus. The multi-object corpus is analysed within item-count level "
            "throughout, combined by trial-count-weighted average; a pooled-across-item-count number is "
            "never this corpus's effect size."
        ),
        "sign_map": SIGN_TO_WORSE_BEHAVIOUR,
        "sign_map_note": (
            "Applied once, at the point each behavioural-correlation result is packaged, never inside the "
            "correlation estimator itself. Every reported coefficient here is against WORSE behaviour."
        ),
        "content_specific_serial_pull_operationalisation": CONTENT_SPECIFIC_SERIAL_PULL_OPERATIONALISATION,
        "count_precondition_rule": COUNT_PRECONDITION_RULE,
        "count_precondition_min_trials": COUNT_PRECONDITION_MIN_TRIALS,
        "deviation_survival_decision_rule_declared_before_fitting": DEVIATION_SURVIVAL_DECISION_RULE,
        "content_pull_survival_decision_rule_declared_before_fitting": CONTENT_PULL_SURVIVAL_DECISION_RULE,
        "mandatory_controls": MANDATORY_CONTROLS_NOTE,
        "session_limit": session_limit,
        "status": "running",
    }
    _flush(output)
    _log("rule text and count precondition written to the artifact before any number is read")
    root = data_root()
    delivered = json.loads(SERIAL_DEPENDENCE_ARTIFACT_PATH.read_text())
    delivered_reach = delivered["reachability"]

    _log("loading the multi-object macaque corpus (one pass, shared by the reproduction gate and both blocks)")
    watters_seen, watters_loaded, watters_refused = 0, [], []
    for session in iter_watters(root, bin_ms=100.0):
        watters_seen += 1
        if session["status"] != "loaded":
            watters_refused.append({"session": session["session"], "status": session["status"]})
            continue
        watters_loaded.append(session)
        # session_limit is a smoke-test resource limit on how many sessions this invocation processes,
        # applied at the point of first admission for each corpus so seen/loaded/refused still reconcile
        # internally; it is never a statement about the data and is absent (None) on the real run.
        if session_limit is not None and len(watters_loaded) >= session_limit:
            break
    _log(f"multi-object macaque corpus: {watters_seen} seen, {len(watters_loaded)} loaded, "
         f"{len(watters_refused)} refused, elapsed={time.time() - t0:.0f}s")

    _log("reproduction gate: re-running the delivered estimators on the delivered sessions")
    # Keyed by how many watters sessions this invocation actually loaded, not a bare "reproduction_gate":
    # the gate's correctness depends on which sessions it ran over, so a checkpoint written under one
    # session count (e.g. a smoke test's truncated corpus) must never be replayed for an invocation that
    # loaded a different count (e.g. the real, untruncated run) -- a stale cache hit there would silently
    # skip recomputation and report a gate result computed on the wrong corpus.
    computed_gate_result, watters_arrays_by_session = _fit(
        f"reproduction_gate|n_watters_loaded{len(watters_loaded)}",
        lambda: full_reproduction_gate(root, watters_loaded))

    gate_result = _gate_result_for_run(
        computed_gate_result, session_limit, len(watters_loaded),
        delivered_reach["watters_2026_macaque_multi_object"]["n_sessions_with_a_computed_bundle"])

    output["reproduction_gate"] = gate_result
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")

    if session_limit is None and gate_result["status"] != "reproduced_exactly":
        output["status"] = "stopped_reproduction_gate_failed"
        output["deviation_survival_test"] = {"status": "not_run_reproduction_gate_failed"}
        output["content_pull_survival_test"] = {"status": "not_run_reproduction_gate_failed"}
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce exactly; no new number was read")
        print(json.dumps({"reproduction_gate": gate_result["status"]}, indent=2))
        return

    macaque_paths = _reachable_sessions(root)
    if session_limit is not None:
        macaque_paths = macaque_paths[:session_limit]
    macaque_bundles = []
    for path in macaque_paths:
        bundle = _fit(f"macaque_bundle|{path.stem}", lambda p=path: _macaque_session_bundle(p))
        if bundle is not None:
            macaque_bundles.append(bundle)
    panichello_dir = _panichello_directory(root)
    n_macaque_on_disk = len(list(panichello_dir.glob("*.mat"))) if panichello_dir else 0

    watters_bundles = []
    n_watters_arrays_none = 0
    for session in watters_loaded:
        entry = watters_arrays_by_session.get(session["session"])
        if entry is None:
            n_watters_arrays_none += 1
            continue
        watters_bundles.append(_watters_bundle_with_label(entry, CONTENT_LABEL_K_CLASSES))

    output["zero_drop_accounting"] = {
        "panichello_2024_macaque_lPFC_single_item": {
            "n_seen": n_macaque_on_disk,
            "n_excluded_below_reachability_floor": n_macaque_on_disk - len(macaque_paths),
            "n_reaching_floor": len(macaque_paths),
            "n_excluded_too_few_trials_with_defined_direction": len(macaque_paths) - len(macaque_bundles),
            "n_analysed": len(macaque_bundles),
            "reconciles": bool(n_macaque_on_disk == (n_macaque_on_disk - len(macaque_paths)) +
                               (len(macaque_paths) - len(macaque_bundles)) + len(macaque_bundles)),
        },
        "watters_2026_macaque_multi_object": {
            "n_seen": watters_seen, "n_refused_by_shared_loader": len(watters_refused),
            "n_loaded": len(watters_loaded), "n_arrays_not_computable": n_watters_arrays_none,
            "n_analysed": len(watters_bundles),
            "reconciles": bool(watters_seen == len(watters_refused) + n_watters_arrays_none + len(watters_bundles)),
        },
    }
    output["zero_drop_reconciliation_against_delivered_artifact"] = {
        "delivered_macaque_n_sessions": delivered_reach["panichello_2024_macaque_lPFC_single_item"]["n_sessions_with_a_computed_bundle"],
        "here_macaque_n_sessions": len(macaque_bundles),
        "macaque_matches_delivered": len(macaque_bundles) == delivered_reach["panichello_2024_macaque_lPFC_single_item"]["n_sessions_with_a_computed_bundle"],
        "delivered_watters_n_sessions": delivered_reach["watters_2026_macaque_multi_object"]["n_sessions_with_a_computed_bundle"],
        "here_watters_n_sessions": len(watters_bundles),
        "watters_matches_delivered": len(watters_bundles) == delivered_reach["watters_2026_macaque_multi_object"]["n_sessions_with_a_computed_bundle"],
    }
    _flush(output)
    _log(f"macaque bundles: {len(macaque_bundles)}/{len(macaque_paths)}; "
         f"watters bundles: {len(watters_bundles)}/{len(watters_loaded)}; elapsed={time.time() - t0:.0f}s")

    corpora = {
        "panichello_2024_macaque_lPFC_single_item": (macaque_bundles, SIGN_TO_WORSE_BEHAVIOUR["panichello_2024_macaque_lPFC_single_item"], False),
        "watters_2026_macaque_multi_object": (watters_bundles, SIGN_TO_WORSE_BEHAVIOUR["watters_2026_macaque_multi_object"], True),
    }

    output["count_precondition"] = {}
    output["deviation_survival_test"] = {}
    output["content_pull_survival_test"] = {}
    for corpus_key, (bundles, sign, has_levels) in corpora.items():
        precond = _count_precondition(bundles, corpus_key)
        output["count_precondition"][corpus_key] = precond
        _flush(output)
        _log(f"count precondition, {corpus_key}: "
             f"{precond['total_trials_with_defined_content_pull_deviation_and_outcome']} trials, "
             f"meets_precondition={precond['meets_precondition']}")

        if not precond["meets_precondition"]:
            void_branch = {
                "branch": "too_few_trials_with_a_defined_content_specific_pull_to_test",
                "count": precond["total_trials_with_defined_content_pull_deviation_and_outcome"],
                "minimum_required": COUNT_PRECONDITION_MIN_TRIALS,
            }
            output["deviation_survival_test"][corpus_key] = void_branch
            output["content_pull_survival_test"][corpus_key] = void_branch
            _flush(output)
            continue

        _log(f"the deviation-survival test: {corpus_key} ({len(bundles)} sessions)")
        decisive = _compute_decisive_all_sessions(
            bundles, sign, has_levels, corpus_key, f"component_and_content_specific_serial_pull|decisive|{corpus_key}")
        deviation_survival_test = run_deviation_survival_test(
            decisive, bundles, sign, has_levels, corpus_key,
            f"component_and_content_specific_serial_pull|deviation_survival|{corpus_key}")
        output["deviation_survival_test"][corpus_key] = deviation_survival_test
        _flush(output)
        _log(f"  the deviation-survival test {corpus_key} branch at primary window: "
             f"{deviation_survival_test['branch_at_primary_window']} elapsed={time.time() - t0:.0f}s")

        _log(f"the content-pull-survival test: {corpus_key}")
        content_pull_survival_test = run_content_pull_survival_test(decisive)
        output["content_pull_survival_test"][corpus_key] = content_pull_survival_test
        _flush(output)
        _log(f"  the content-pull-survival test {corpus_key} branch: "
             f"{content_pull_survival_test['branch']} elapsed={time.time() - t0:.0f}s")

    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/component_and_content_specific_serial_pull_checkpoint.json",
    }
    output["status"] = "complete_under_a_session_limit_numbers_are_not_results" if session_limit is not None else "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": gate_result["status"],
        "deviation_survival_test": {k: v.get("branch_at_primary_window", v.get("branch")) for k, v in output["deviation_survival_test"].items()},
        "content_pull_survival_test": {k: v.get("branch") for k, v in output["content_pull_survival_test"].items()},
    }, indent=2, default=float))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-limit", type=int, default=None,
                         help="Smoke-test resource limit: process at most this many sessions per corpus. "
                              "Absent (None) on the real run.")
    args = parser.parse_args()
    main(session_limit=args.session_limit)
