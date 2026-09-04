"""run_swap_target_and_preceding_trial_item.py -- when the report lands on
the WRONG object (a swap), does the report land specifically on the object
nearest the PRECEDING trial's remembered item?

Two delivered results in this project have never been joined. (1) A swap is
a binding failure: the report lands nearer an uncued object than the cued
one, and the population component that predicts accuracy tracks WHICH item
is reported, not how precisely
(``results/component_and_item_binding.json``). (2) There is a content-
specific serial pull toward the preceding trial's memorandum, over and above
ordinary attraction to the trial's own class
(``results/deviation_serial_dependence_and_temporal_locus.json``). If the
accuracy-predicting component is a mis-set pointer rather than a generic
vulnerability signal, the obvious candidate for what it points AT is the
thing the population is already being pulled toward: the preceding trial's
item. This module tests that directly, behaviourally (does the swap
destination sit nearer the preceding trial's target than the other uncued
object) and neurally (does the trial's own serial pull predict which uncued
object the swap lands on).

The analysed maintenance window ends at or before cue onset on every
admitted trial (the shared loader's own admission criterion), so nothing
measured here can be explained by variability in memory for the cue
feature itself -- the report lands on one of two objects that were both
already on the display before the cue appeared.

SCOPE. Multi-object macaque corpus only, item count 3 (a swap there leaves
exactly two uncued objects, so "nearer to the preceding trial's item" has a
two-alternative answer with an exact 0.5 null). Nothing already delivered is
modified, re-run or re-labelled.

REUSE. ``_object_geometry`` (swap/imprecision geometry, run_component_and_
item_binding.py) supplies the primary swap indicator unchanged. The content-
specific serial pull estimator (``_content_specific_serial_pull``,
``_watters_session_bundle``, ``unit_direction_vectors``,
run_deviation_serial_dependence_and_temporal_locus.py) supplies the neural
observable; that function returns only a session-level mean, so a per-trial
reproduction of its own internal loop is used to recover the per-trial
values it does not expose, and is checked against the function's own
aggregate output every session as a reproduction, not a variant. The
reproduction gate reuses ``_observable_arrays`` / ``_session_observable_arm``
/ ``_pool_cell`` (run_dissociation_replication_and_counting_noise.py)
unchanged. Pooling and inference reuse ``_pool_values``
(run_watters_state_geometry.py), ``paired_sign_flip_test``,
``minimum_detectable_paired_difference``, ``stable_seed``
(src/statistics.py), and ``slope_across_sessions_test``
(src/state_persistence.py) -- ``_pool_values`` is ``slope_across_sessions_
test`` plus its minimum detectable paired difference and median, already the
project's standard bundle for "one scalar per session, pooled by two-sided
paired sign-flip test against zero".

TWO-ALTERNATIVE STATISTICS AGAINST 0.5. ``_pool_values`` tests a list of
per-session scalars against zero. To test a list of per-session PROPORTIONS
against one-half, each value is shifted by -0.5 before pooling and the
location fields (mean, median, CI) are shifted back by +0.5 when reported;
the p-value, significance flags and minimum detectable difference are
shift-invariant and pass through unchanged. This reuses ``_pool_values``
(and therefore ``slope_across_sessions_test`` and
``minimum_detectable_paired_difference``) exactly rather than writing a
second pooling primitive for a 0.5 null.

SESSION-LEVEL SPATIAL BIAS CONTROL. Block A's target-referenced statistic could in principle be
reproduced by nothing more than a generic, non-trial-specific spatial bias in where the two uncued
objects happen to sit relative to a typical target position in that session, rather than by any
relationship to the SPECIFIC trial that preceded the one being scored. This is tested directly: the
identical two-alternative statistic is recomputed with the preceding trial's own target angle replaced
by the session's own mean cued-object angle (a single constant per session, carrying the session's
spatial layout but no trial-specific information), and the two per-session proportions are compared by
a direct paired test. The result is reported alongside block A's own branch, not folded into it.
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

from corpus_sessions import data_root, iter_watters, watters_behaviour  # noqa: E402
from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from run_component_and_item_binding import _object_geometry  # noqa: E402
from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    CONTENT_LABEL_K_CLASSES, SHARP_TEST_MIN_CLASSES, SHARP_TEST_MIN_PER_CLASS,
    SHARP_TEST_MIN_QUALIFYING_TRIALS, _content_specific_serial_pull, _watters_session_bundle,
    unit_direction_vectors,
)
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    _observable_arrays, _pool_cell, _session_observable_arm,
)
from run_state_content_link import usable_label  # noqa: E402
from run_watters_state_geometry import PRIMARY_QUALITY_TIER, _pool_values  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "swap_target_and_preceding_trial_item.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "swap_target_and_preceding_trial_item_checkpoint.json"
ANALYSIS_VERSION = "2026-08-19"
REPRODUCTION_TOLERANCE = 1e-6
CORPUS_LABEL = "watters_2026_macaque_multi_object"
DISSOCIATION_ARTIFACT_PATH = ROOT / "results" / "dissociation_replication_and_counting_noise.json"
SERIAL_DEPENDENCE_ARTIFACT_PATH = ROOT / "results" / "deviation_serial_dependence_and_temporal_locus.json"

# ---------------------------------------------------------------------------
# Pre-declared constants -- every one of these is fixed before any trial is
# looked at and none is revisited after a number is seen.
# ---------------------------------------------------------------------------

MIN_POOLED_ADMISSIBLE_SWAP_TRIALS = 200
BRANCH_TOO_FEW_TRIALS = "too_few_three_item_swap_trials_with_an_admissible_preceding_trial_to_test"
BRANCH_GATE_FAILED = "void_reproduction_gate_did_not_reproduce"
BRANCH_SYMMETRY_PREMISE_FAILED = "block_0_shuffled_null_not_centred_on_one_half_stopped_before_block_a_or_b"

NEAR_SEPARATION_THRESHOLD_DEGREES = 15.0
BLOCK0_N_SHUFFLES = 1000
BLOCK0_CENTERING_Z_THRESHOLD = 3.0  # three Monte-Carlo standard errors, declared before any draw is run

BLOCK_A_MDD_POWERED_NULL_THRESHOLD = 0.05  # proportion units, both statistics; declared before any number is read

BRANCH_A_TARGET = "swaps_land_on_the_object_nearest_the_preceding_trials_remembered_item"
BRANCH_A_RESPONSE = "swaps_repeat_the_preceding_trials_response_rather_than_its_remembered_item"
BRANCH_A_BOTH_INSEPARABLE = "swaps_follow_the_preceding_trial_but_its_remembered_item_and_its_response_cannot_be_separated"
BRANCH_A_POWERED_NULL = "powered_null_swap_destination_is_unrelated_to_the_preceding_trial"
BRANCH_A_INCONCLUSIVE = "inconclusive_below_detection_floor"
BRANCH_A_AVOID = "swaps_avoid_the_preceding_trials_item"
BRANCH_A_NOT_COVERED = "outcome_not_covered_by_the_pre_declared_rule"

# A control on the target-referenced statistic: the identical two-alternative statistic recomputed with
# the preceding trial's own target angle replaced by a session-constant reference (the circular mean of
# the cued object's angle over every trial the shared loader admits in that session), which carries the
# session's overall spatial layout but no information about which trial preceded the one being scored.
BRANCH_SESSION_BIAS_SURVIVES = "preceding_trial_effect_survives_the_session_level_spatial_bias_control"
BRANCH_SESSION_BIAS_NOT_SEPARABLE = "preceding_trial_effect_not_separable_from_the_session_level_spatial_bias"
BRANCH_SESSION_BIAS_EXCEEDS = "session_level_spatial_bias_exceeds_the_preceding_trial_effect"
BRANCH_SESSION_BIAS_NOT_COVERED = "outcome_not_covered_by_the_pre_declared_rule"

BRANCH_B_POSITIVE = "the_trials_own_serial_pull_predicts_that_its_swap_goes_to_the_preceding_trials_item"
BRANCH_B_POWERED_NULL = "powered_null_serial_pull_magnitude_does_not_determine_the_swap_destination"
BRANCH_B_INCONCLUSIVE = "inconclusive_below_detection_floor"
BRANCH_B_OPPOSITE = "serial_pull_is_larger_on_swaps_that_avoid_the_preceding_trials_item"
BRANCH_B_NOT_SEPARABLE = "serial_pull_group_difference_not_separable_from_a_session_level_or_rate_confound"
BRANCH_B_NOT_COVERED = "outcome_not_covered_by_the_pre_declared_rule"

DROP_LOADER = "excluded_by_shared_loader"
DROP_ITEM_COUNT = "item_count_not_three"
DROP_NOT_SWAP = "not_a_swap_by_primary_definition"
DROP_NO_PRECEDING = "no_admissible_preceding_trial"
DROP_SWAP_DEST_UNDEFINED = "swap_destination_undefined"
DROP_SWAP_DEST_TIE = "swap_destination_ambiguous_tie_between_the_two_uncued_objects"
DROP_SESSION_ARRAYS = "session_excluded_arrays_not_computable"
SURVIVING = "surviving_item_count_3_swap_trial_with_admissible_preceding_trial"

BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Two per-session two-alternative proportions (target-referenced, response-referenced), each pooled "
    "across sessions by a two-sided paired sign-flip test against 0.5. 'Significant' means two-sided "
    "p <= 0.05; 'above'/'below' is the sign of (pooled mean - 0.5). Checked in this order, so a below-0.5 "
    "result is never absorbed into an above-0.5 cell:\n"
    "  0. Either statistic significantly BELOW 0.5 -> 'swaps_avoid_the_preceding_trials_item', reported "
    "with its own effect size, and not folded into any cell below.\n"
    "  1. Target-referenced significantly above 0.5 and response-referenced NOT significantly above "
    "(not significant, or below 0.5 -- already excluded by rule 0) -> "
    "'swaps_land_on_the_object_nearest_the_preceding_trials_remembered_item'.\n"
    "  2. Response-referenced significantly above 0.5 and target-referenced NOT significantly above -> "
    "'swaps_repeat_the_preceding_trials_response_rather_than_its_remembered_item' (response "
    "perseveration, not a memory trace).\n"
    "  3. BOTH significantly above 0.5: a direct paired sign-flip test of the per-session (target - "
    "response) proportions decides between them -- target significantly larger -> rule 1's label; "
    "response significantly larger -> rule 2's label; no significant paired difference -> "
    "'swaps_follow_the_preceding_trial_but_its_remembered_item_and_its_response_cannot_be_separated'.\n"
    "  4. Neither significant (and neither below 0.5): if BOTH statistics' own minimum detectable "
    f"paired difference at 80% power is below {BLOCK_A_MDD_POWERED_NULL_THRESHOLD} (proportion units) -> "
    "'powered_null_swap_destination_is_unrelated_to_the_preceding_trial'; otherwise -> "
    "'inconclusive_below_detection_floor', never quoted without its detection floor.\n"
    "Any pattern not covered above is recorded as 'outcome_not_covered_by_the_pre_declared_rule' with "
    "every number, not stretched onto a listed cell."
)

BLOCK_A_SESSION_LEVEL_SPATIAL_BIAS_CONTROL_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "A control on the target-referenced statistic only, reported alongside block A's own branch and never "
    "gating or overriding it. The identical two-alternative statistic, computed the same way (per-trial "
    "indicator against a reference angle, ties excluded and counted, pooled within session then across "
    "sessions by a two-sided paired sign-flip test against 0.5), but with the preceding trial's target "
    "angle replaced by the session's own mean cued-object angle -- the circular mean of the cued object's "
    "angle over every trial the shared loader admits in that session. That reference is constant within a "
    "session and carries the session's overall spatial layout but no information about which specific "
    "trial preceded the one being scored, so it isolates how much of the target-referenced result a "
    "generic, non-trial-specific spatial bias could reproduce on its own. The two per-session proportions "
    "(target-referenced, session-mean-referenced) are compared by a direct two-sided paired sign-flip "
    "test, which also sets this comparison's own minimum detectable paired difference at 80% power. "
    "Checked in this order:\n"
    "  0. Either pooled statistic does not itself reach the paired-sign-flip test floor -> "
    "'outcome_not_covered_by_the_pre_declared_rule', with every number.\n"
    "  1. The direct paired test is significant and the target-referenced fraction is larger -> "
    "'preceding_trial_effect_survives_the_session_level_spatial_bias_control'.\n"
    "  2. The direct paired test is significant and the session-mean-referenced fraction is larger -> "
    "'session_level_spatial_bias_exceeds_the_preceding_trial_effect', reported with its own effect size "
    "and not folded into any cell above.\n"
    "  3. The direct paired test is not significant -> "
    "'preceding_trial_effect_not_separable_from_the_session_level_spatial_bias', reported with the paired "
    "test's own minimum detectable difference at 80% power."
)

BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, the mean per-trial content-specific serial pull difference on trials whose swap "
    "destination is the object nearer the preceding trial's target ('previous-item swaps') minus the "
    "same mean on trials where it is not, pooled across sessions by a two-sided paired sign-flip test "
    "against zero. Two controls, both computed before any branch fires: the rate control (the identical "
    "per-session-difference-then-pooled statistic with total spike count substituted for serial pull) "
    "and the bias-only control (the identical statistic with each trial's own serial pull replaced by "
    "its session's leave-one-out mean over its other eligible trials). Checked in this order:\n"
    "  0. Bias-only control significant, OR the rate control is significant and the spike-count partial "
    "(the same group difference computed on serial pull residualised on spike count within session) does "
    "not survive (not significant with the same sign as the raw group difference) -> "
    "'serial_pull_group_difference_not_separable_from_a_session_level_or_rate_confound'; no cell below "
    "may fire.\n"
    "  1. Raw group difference significant and positive (larger on previous-item swaps) -> "
    "'the_trials_own_serial_pull_predicts_that_its_swap_goes_to_the_preceding_trials_item'.\n"
    "  2. Raw group difference significant and negative -> "
    "'serial_pull_is_larger_on_swaps_that_avoid_the_preceding_trials_item'.\n"
    "  3. Raw group difference not significant: if its own minimum detectable paired difference at 80% "
    "power is below the delivered pooled content-specific serial-pull effect (read live from results/"
    "deviation_serial_dependence_and_temporal_locus.json) -> "
    "'powered_null_serial_pull_magnitude_does_not_determine_the_swap_destination'; otherwise -> "
    "'inconclusive_below_detection_floor'.\n"
    "Any pattern not covered above is recorded as 'outcome_not_covered_by_the_pre_declared_rule' with "
    "every number."
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


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


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _circular_distance_rad(a, b) -> np.ndarray:
    """Geodesic distance between two angles on the circle, in [0, pi]."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))))


def _pool_vs_half(values: list[float]) -> dict:
    """``_pool_values`` (slope_across_sessions_test plus its minimum detectable paired difference and
    median) tests a list of per-session scalars against zero. Shifting each value by -0.5 before pooling
    and shifting the location fields back by +0.5 afterwards reuses it unchanged for a 0.5 null: the
    p-value, significance flags and minimum detectable difference do not depend on where zero sits."""
    pooled = _pool_values([v - 0.5 for v in values])
    if pooled.get("status") != "tested":
        return pooled
    out = dict(pooled)
    out["mean_value"] = pooled["mean_value"] + 0.5
    out["ci_lower"] = pooled["ci_lower"] + 0.5
    out["ci_upper"] = pooled["ci_upper"] + 0.5
    if pooled.get("median_value") is not None:
        out["median_value"] = pooled["median_value"] + 0.5
    return out


def _mdd(pooled: dict) -> float | None:
    mdd = pooled.get("minimum_detectable_paired_difference_at_80pct_power", {})
    return mdd.get("mdd") if mdd.get("status") == "computed" else None


def _direct_paired_test(a: list[float], b: list[float], tag: str) -> dict:
    rng = np.random.default_rng(stable_seed(tag))
    result = paired_sign_flip_test(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                    alternative="two-sided", rng=rng)
    result = {k: v for k, v in result.items() if k != "null"}
    result["significant"] = bool(result["p_value"] <= 0.05)
    return result


# ---------------------------------------------------------------------------
# Reproduction gate
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _close(observed, reference) -> bool:
    return observed is not None and reference is not None and abs(float(observed) - float(reference)) <= REPRODUCTION_TOLERANCE


def _live_dissociation_reference() -> dict:
    node = _read_json(DISSOCIATION_ARTIFACT_PATH)["block_a"]["results"][PRIMARY_QUALITY_TIER]["pooled"]["deviation"][
        "within_item_count_level"]
    return {
        "orthogonality_gate_vs_spike_count": {"mean_value": node["orthogonality_gate_vs_spike_count"]["mean_value"],
                                               "p_value": node["orthogonality_gate_vs_spike_count"]["p_value"]},
        "raw_vs_report_error": {"mean_value": node["raw_vs_report_error"]["mean_value"],
                                 "p_value": node["raw_vs_report_error"]["p_value"]},
        "source_path": f"block_a.results.{PRIMARY_QUALITY_TIER}.pooled.deviation.within_item_count_level",
    }


def _live_serial_pull_reference() -> dict:
    node = _read_json(SERIAL_DEPENDENCE_ARTIFACT_PATH)["block_a"][CORPUS_LABEL]["content_specific_serial_pull_pooled"]
    return {"mean_value": node["mean_value"], "p_value": node["p_value"],
            "source_path": f"block_a.{CORPUS_LABEL}.content_specific_serial_pull_pooled"}


# ---------------------------------------------------------------------------
# Per-session analysis
# ---------------------------------------------------------------------------

def _trial_rows(behaviour, session: dict):
    rows = behaviour.loc[(session["animal"], session["session_date"])]
    return rows.loc[session["trial_num"].tolist()]


def _content_specific_serial_pull_per_trial(bundle: dict) -> dict:
    """Reproduces ``_content_specific_serial_pull``'s internal per-trial ``pull_previous(t) -
    pull_other_classes(t)`` loop exactly (same constants, same eligibility, same formula), returning
    the per-trial values that function computes but does not expose. Verified against that function's
    own aggregate output by the caller, so this is a reproduction, not a variant."""
    vectors = unit_direction_vectors(bundle["activity_by_unit"])
    label = bundle["memorandum_label"]
    n = vectors.shape[0]
    finite_label = np.isfinite(label)
    if not finite_label.any():
        return {"status": "no_finite_memorandum_label", "per_trial": {}}
    ok, reason, _mask = usable_label(label[finite_label], min_classes=SHARP_TEST_MIN_CLASSES,
                                      min_per_class=SHARP_TEST_MIN_PER_CLASS)
    if not ok:
        return {"status": "no_usable_memorandum_label", "reason": reason, "per_trial": {}}
    classes = np.unique(label[finite_label])
    class_members = {c: np.flatnonzero((label == c) & finite_label) for c in classes}
    class_mean = {c: vectors[idx].mean(axis=0) for c, idx in class_members.items()
                  if len(idx) >= SHARP_TEST_MIN_PER_CLASS}
    eligible_classes = set(class_mean)
    if len(eligible_classes) < SHARP_TEST_MIN_CLASSES:
        return {"status": "fewer_than_minimum_eligible_memorandum_classes",
                "n_eligible_classes": len(eligible_classes), "per_trial": {}}

    per_trial: dict[int, float] = {}
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
        per_trial[t] = pull_prev - float(np.mean(other_pulls))

    if len(per_trial) < SHARP_TEST_MIN_QUALIFYING_TRIALS:
        return {"status": "too_few_qualifying_trials", "n_qualifying_trials": len(per_trial), "per_trial": per_trial}
    return {"status": "computed", "n_qualifying_trials": len(per_trial),
            "mean_pull_difference": float(np.mean(list(per_trial.values()))), "per_trial": per_trial}


def _trial_admission_and_geometry(session: dict, behaviour, usable: np.ndarray) -> dict:
    """Per-trial drop-reason cascade and swap-destination geometry, in the session's own full
    (unfiltered-by-usable) trial order. Every trial gets exactly one drop reason, ending in SURVIVING
    for an item-count-3 primary-rule swap trial with an admissible immediately-preceding trial."""
    trial_rows = _trial_rows(behaviour, session)
    target = trial_rows["target_object_index"].to_numpy(dtype=int)
    obj_x = trial_rows[[f"object_{i}_x" for i in range(3)]].to_numpy(dtype=float)
    obj_y = trial_rows[[f"object_{i}_y" for i in range(3)]].to_numpy(dtype=float)
    obj_theta = trial_rows[[f"object_{i}_theta" for i in range(3)]].to_numpy(dtype=float)
    resp_x = trial_rows["response_x"].to_numpy(dtype=float)
    resp_y = trial_rows["response_y"].to_numpy(dtype=float)
    resp_theta = trial_rows["response_theta"].to_numpy(dtype=float)
    cued_theta = np.asarray(session["cued_theta"], dtype=float)
    trial_num = np.asarray(session["trial_num"], dtype=int)
    item_count = np.asarray(session["num_objects"], dtype=float)
    swap_primary = _object_geometry(behaviour, session)["swap_primary"]
    n = len(trial_num)

    # The session-level spatial bias control's reference: the circular mean of the cued object's angle
    # over every trial the shared loader admits in this session -- a single constant, carrying the
    # session's overall spatial layout but no information about which trial preceded the one being scored.
    usable_cued_theta = cued_theta[usable]
    session_mean_cued_theta = float(np.arctan2(np.mean(np.sin(usable_cued_theta)),
                                                 np.mean(np.cos(usable_cued_theta)))) \
        if usable_cued_theta.size else None

    reason = [""] * n
    swap_dest_idx = np.full(n, -1, dtype=int)
    other_idx = np.full(n, -1, dtype=int)

    for t in range(n):
        if not usable[t]:
            reason[t] = DROP_LOADER
            continue
        if item_count[t] != 3.0:
            reason[t] = DROP_ITEM_COUNT
            continue
        if not swap_primary[t]:
            reason[t] = DROP_NOT_SWAP
            continue
        if t == 0 or not usable[t - 1] or trial_num[t] - trial_num[t - 1] != 1 or not np.isfinite(cued_theta[t - 1]):
            reason[t] = DROP_NO_PRECEDING
            continue
        uncued = [i for i in range(3) if i != target[t]]
        d0 = float(np.hypot(obj_x[t, uncued[0]] - resp_x[t], obj_y[t, uncued[0]] - resp_y[t]))
        d1 = float(np.hypot(obj_x[t, uncued[1]] - resp_x[t], obj_y[t, uncued[1]] - resp_y[t]))
        if not (np.isfinite(d0) and np.isfinite(d1)):
            reason[t] = DROP_SWAP_DEST_UNDEFINED
            continue
        if d0 == d1:
            reason[t] = DROP_SWAP_DEST_TIE
            continue
        swap_dest_idx[t], other_idx[t] = (uncued[0], uncued[1]) if d0 < d1 else (uncued[1], uncued[0])
        reason[t] = SURVIVING

    return {
        "reason": reason, "swap_dest_idx": swap_dest_idx, "other_idx": other_idx,
        "obj_theta": obj_theta, "resp_theta": resp_theta, "cued_theta": cued_theta, "trial_num": trial_num,
    }


def analyse_session(session: dict, behaviour, seed_prefix: str) -> dict:
    tag = f"{seed_prefix}|{session['session']}"
    arrays, excluded, usable = _observable_arrays(session["counts"], session)
    if arrays is None:
        return {"status": DROP_SESSION_ARRAYS, "session": session["session"], "n_trials_total": len(session["trial_num"]),
                "trials_excluded_by_reason": excluded}

    deviation_arm = _session_observable_arm(arrays, "deviation", f"{tag}|gate")
    bundle = _watters_session_bundle(session, arrays, usable)
    delivered_pull = _content_specific_serial_pull(bundle)
    reproduced_pull = _content_specific_serial_pull_per_trial(bundle)

    identity_ok = (delivered_pull.get("status") == reproduced_pull.get("status"))
    identity_max_abs_diff = None
    if delivered_pull.get("status") == "computed" and reproduced_pull.get("status") == "computed":
        identity_ok = identity_ok and delivered_pull["n_qualifying_trials"] == reproduced_pull["n_qualifying_trials"]
        identity_max_abs_diff = abs(delivered_pull["mean_pull_difference"] - reproduced_pull["mean_pull_difference"])
        identity_ok = identity_ok and identity_max_abs_diff <= 1e-9

    geometry = _trial_admission_and_geometry(session, behaviour, usable)
    reason = geometry["reason"]
    n = len(reason)
    bundle_index_of_full = np.cumsum(usable.astype(int)) - 1

    drop_counts: dict[str, int] = {}
    for r in reason:
        drop_counts[r] = drop_counts.get(r, 0) + 1

    survivors = [t for t in range(n) if reason[t] == SURVIVING]

    swap_theta = geometry["obj_theta"][np.arange(n), np.clip(geometry["swap_dest_idx"], 0, 2)]
    other_theta = geometry["obj_theta"][np.arange(n), np.clip(geometry["other_idx"], 0, 2)]
    prev_target_theta = np.full(n, np.nan)
    prev_response_theta = np.full(n, np.nan)
    prev_target_theta[1:] = geometry["cued_theta"][:-1]
    prev_response_theta[1:] = geometry["resp_theta"][:-1]

    trial_records = []
    for t in survivors:
        d_swap_target = float(_circular_distance_rad(swap_theta[t], prev_target_theta[t]))
        d_other_target = float(_circular_distance_rad(other_theta[t], prev_target_theta[t]))
        ind_target = None if d_swap_target == d_other_target else int(d_swap_target < d_other_target)

        resp_defined = bool(np.isfinite(prev_response_theta[t]))
        ind_response = None
        if resp_defined:
            d_swap_resp = float(_circular_distance_rad(swap_theta[t], prev_response_theta[t]))
            d_other_resp = float(_circular_distance_rad(other_theta[t], prev_response_theta[t]))
            ind_response = None if d_swap_resp == d_other_resp else int(d_swap_resp < d_other_resp)

        separation_deg = float(np.degrees(_circular_distance_rad(swap_theta[t], other_theta[t])))
        bundle_idx = int(bundle_index_of_full[t])
        pull_value = reproduced_pull["per_trial"].get(bundle_idx) if reproduced_pull.get("status") in (
            "computed", "too_few_qualifying_trials") else None
        spike_count_value = float(arrays["spike_count"][bundle_idx]) if bundle_idx < len(arrays["spike_count"]) else None

        trial_records.append({
            "full_index": t, "bundle_index": bundle_idx,
            "swap_theta": float(swap_theta[t]), "other_theta": float(other_theta[t]),
            "prev_target_theta": float(prev_target_theta[t]),
            "prev_response_theta": float(prev_response_theta[t]) if resp_defined else None,
            "indicator_target": ind_target, "indicator_response": ind_response,
            "response_defined_for_preceding_trial": resp_defined,
            "angular_separation_between_uncued_objects_degrees": separation_deg,
            "content_specific_serial_pull_diff": pull_value, "total_spike_count": spike_count_value,
        })

    return {
        "status": "computed", "session": session["session"], "animal": session["animal"],
        "n_trials_total": n, "drop_counts_by_reason": drop_counts,
        "n_surviving": len(survivors),
        "deviation_gate_arm": deviation_arm,
        "delivered_content_specific_serial_pull": delivered_pull,
        "reproduced_content_specific_serial_pull_identity_check": {
            "statuses_match": identity_ok, "max_abs_diff_mean_pull_difference": identity_max_abs_diff,
        },
        "trial_records": trial_records,
    }


def _block0_shuffle_draws(swap_theta: np.ndarray, other_theta: np.ndarray, prev_target_theta: np.ndarray,
                           n_shuffles: int, seed_tag: str) -> list[float | None]:
    n = len(swap_theta)
    draws: list[float | None] = []
    for d in range(n_shuffles):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|shuffle{d}"))
        shuffled_prev = prev_target_theta[rng.permutation(n)]
        d_swap = _circular_distance_rad(swap_theta, shuffled_prev)
        d_other = _circular_distance_rad(other_theta, shuffled_prev)
        ind = np.where(d_swap < d_other, 1.0, np.where(d_swap > d_other, 0.0, np.nan))
        finite = ind[np.isfinite(ind)]
        draws.append(float(np.mean(finite)) if finite.size else None)
    return draws


# ---------------------------------------------------------------------------
# Block A / Block B pooling
# ---------------------------------------------------------------------------

def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coeffs


def _block_b_session_values(session_rows: list[dict]) -> dict:
    """Per-session raw / rate-control / bias-only-control / spike-count-partial group differences
    (mean on previous-item-swap trials minus mean on the others), restricted to the trials Block A
    admitted and for which the delivered serial-pull estimator's own eligibility gate also qualifies."""
    per_session = {"raw": [], "rate": [], "bias": [], "partial": [], "sessions": []}
    for row in session_rows:
        if row.get("status") != "computed":
            continue
        eligible = [tr for tr in row["trial_records"]
                    if tr["indicator_target"] is not None and tr["content_specific_serial_pull_diff"] is not None
                    and tr["total_spike_count"] is not None]
        group1_pull = [tr["content_specific_serial_pull_diff"] for tr in eligible if tr["indicator_target"] == 1]
        group0_pull = [tr["content_specific_serial_pull_diff"] for tr in eligible if tr["indicator_target"] == 0]
        if len(group1_pull) < 1 or len(group0_pull) < 1:
            continue
        group1_rate = [tr["total_spike_count"] for tr in eligible if tr["indicator_target"] == 1]
        group0_rate = [tr["total_spike_count"] for tr in eligible if tr["indicator_target"] == 0]

        pulls = np.array([tr["content_specific_serial_pull_diff"] for tr in eligible], dtype=float)
        n_elig = len(eligible)
        loo_mean = (pulls.sum() - pulls) / (n_elig - 1) if n_elig >= 2 else None
        bias_diff = None
        if loo_mean is not None:
            g1 = [loo_mean[i] for i, tr in enumerate(eligible) if tr["indicator_target"] == 1]
            g0 = [loo_mean[i] for i, tr in enumerate(eligible) if tr["indicator_target"] == 0]
            if g1 and g0:
                bias_diff = float(np.mean(g1) - np.mean(g0))

        rates = np.array([tr["total_spike_count"] for tr in eligible], dtype=float)
        partial_diff = None
        if n_elig >= 3 and np.std(rates) > 0.0:
            resid = _residualize(pulls, rates)
            r1 = [resid[i] for i, tr in enumerate(eligible) if tr["indicator_target"] == 1]
            r0 = [resid[i] for i, tr in enumerate(eligible) if tr["indicator_target"] == 0]
            if r1 and r0:
                partial_diff = float(np.mean(r1) - np.mean(r0))

        per_session["raw"].append(float(np.mean(group1_pull) - np.mean(group0_pull)))
        per_session["rate"].append(float(np.mean(group1_rate) - np.mean(group0_rate)))
        per_session["bias"].append(bias_diff)
        per_session["partial"].append(partial_diff)
        per_session["sessions"].append({
            "session": row["session"], "n_eligible": n_elig,
            "n_group_previous_item_swap": len(group1_pull), "n_group_other": len(group0_pull),
        })
    return per_session


def _block_a_branch(target_pooled: dict, response_pooled: dict, rows: list[dict]) -> dict:
    target_vals = [tr["indicator_target"] for row in rows if row.get("status") == "computed"
                   for tr in row["trial_records"] if tr["indicator_target"] is not None]
    response_vals = [tr["indicator_response"] for row in rows if row.get("status") == "computed"
                      for tr in row["trial_records"] if tr["indicator_response"] is not None]

    def _dir(pooled: dict) -> str | None:
        if pooled.get("status") != "tested":
            return None
        if not pooled.get("significant"):
            return "not_significant"
        return "above" if pooled["mean_value"] > 0.5 else "below"

    t_dir, r_dir = _dir(target_pooled), _dir(response_pooled)
    result: dict = {"target_direction": t_dir, "response_direction": r_dir}

    if t_dir is None or r_dir is None:
        result["branch"] = BRANCH_A_NOT_COVERED
        result["note"] = "one of the two pooled statistics did not reach the paired-sign-flip test floor"
        return result

    if t_dir == "below" or r_dir == "below":
        result["branch"] = BRANCH_A_AVOID
        result["target_effect_size"] = target_pooled.get("mean_value")
        result["response_effect_size"] = response_pooled.get("mean_value")
        return result

    if t_dir == "above" and r_dir != "above":
        result["branch"] = BRANCH_A_TARGET
        return result
    if r_dir == "above" and t_dir != "above":
        result["branch"] = BRANCH_A_RESPONSE
        return result

    if t_dir == "above" and r_dir == "above":
        # session-level (target-proportion, response-proportion) pairs, matched by session
        by_session_t = {row["session"]: np.mean([tr["indicator_target"] for tr in row["trial_records"]
                                                   if tr["indicator_target"] is not None])
                         for row in rows if row.get("status") == "computed"
                         and any(tr["indicator_target"] is not None for tr in row["trial_records"])}
        by_session_r = {row["session"]: np.mean([tr["indicator_response"] for tr in row["trial_records"]
                                                   if tr["indicator_response"] is not None])
                        for row in rows if row.get("status") == "computed"
                        and any(tr["indicator_response"] is not None for tr in row["trial_records"])}
        common = sorted(set(by_session_t) & set(by_session_r))
        a_vals = [by_session_t[s] for s in common]
        b_vals = [by_session_r[s] for s in common]
        paired = _direct_paired_test(a_vals, b_vals, "swap_target_and_preceding_trial_item|block_a|target_vs_response")
        paired["n_sessions_paired"] = len(common)
        result["direct_paired_test_target_minus_response"] = paired
        if paired["significant"] and paired["mean_diff"] > 0:
            result["branch"] = BRANCH_A_TARGET
        elif paired["significant"] and paired["mean_diff"] < 0:
            result["branch"] = BRANCH_A_RESPONSE
        else:
            result["branch"] = BRANCH_A_BOTH_INSEPARABLE
        return result

    # neither significant, neither below
    target_mdd = _mdd(target_pooled)
    response_mdd = _mdd(response_pooled)
    result["target_minimum_detectable_paired_difference"] = target_mdd
    result["response_minimum_detectable_paired_difference"] = response_mdd
    if target_mdd is not None and response_mdd is not None and target_mdd < BLOCK_A_MDD_POWERED_NULL_THRESHOLD \
            and response_mdd < BLOCK_A_MDD_POWERED_NULL_THRESHOLD:
        result["branch"] = BRANCH_A_POWERED_NULL
    else:
        result["branch"] = BRANCH_A_INCONCLUSIVE
    return result


def _block_b_branch(raw_pooled: dict, rate_pooled: dict, bias_pooled: dict, partial_pooled: dict,
                     reference_effect: float) -> dict:
    result: dict = {}
    bias_sig = bool(bias_pooled.get("status") == "tested" and bias_pooled.get("significant"))
    rate_sig = bool(rate_pooled.get("status") == "tested" and rate_pooled.get("significant"))
    raw_sig = bool(raw_pooled.get("status") == "tested" and raw_pooled.get("significant"))
    raw_positive = bool(raw_pooled.get("status") == "tested" and raw_pooled.get("mean_value", 0.0) > 0.0)

    partial_survives = True
    if rate_sig:
        partial_survives = bool(
            partial_pooled.get("status") == "tested" and partial_pooled.get("significant")
            and (partial_pooled.get("mean_value", 0.0) > 0.0) == raw_positive)
    result["bias_only_control_significant"] = bias_sig
    result["rate_control_significant"] = rate_sig
    result["spike_count_partial_survives"] = partial_survives

    if bias_sig or (rate_sig and not partial_survives):
        result["branch"] = BRANCH_B_NOT_SEPARABLE
        return result

    if raw_pooled.get("status") != "tested":
        result["branch"] = BRANCH_B_NOT_COVERED
        result["note"] = "raw group-difference statistic did not reach the paired-sign-flip test floor"
        return result

    if raw_sig and raw_positive:
        result["branch"] = BRANCH_B_POSITIVE
        return result
    if raw_sig and not raw_positive:
        result["branch"] = BRANCH_B_OPPOSITE
        return result

    mdd = _mdd(raw_pooled)
    result["minimum_detectable_paired_difference"] = mdd
    result["reference_effect_size"] = reference_effect
    if mdd is not None and mdd < reference_effect:
        result["branch"] = BRANCH_B_POWERED_NULL
    else:
        result["branch"] = BRANCH_B_INCONCLUSIVE
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "corpus": "Multi-object spatial working memory in macaque frontal cortex, DANDI 000620.",
        "scope": "Item count 3 only: a swap there leaves exactly two uncued objects, so the swap "
                 "destination has a two-alternative answer with an exact 0.5 null.",
        "min_pooled_admissible_swap_trials": MIN_POOLED_ADMISSIBLE_SWAP_TRIALS,
        "near_separation_threshold_degrees": NEAR_SEPARATION_THRESHOLD_DEGREES,
        "block_a_decision_rule_declared_before_fitting": BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "block_b_decision_rule_declared_before_fitting": BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass)")
    loaded, refused = [], []
    n_seen = 0
    for session in iter_watters(root, bin_ms=100.0):
        n_seen += 1
        if session["status"] != "loaded":
            refused.append({"session": session["session"], "status": session["status"]})
            continue
        loaded.append(session)
    _log(f"corpus loaded: {n_seen} seen, {len(loaded)} loaded, {len(refused)} refused, "
         f"elapsed={time.time() - t0:.0f}s")

    behaviour = watters_behaviour(root)
    rows: list[dict] = []
    for session in loaded:
        key = session["session"]
        row = _fit(f"session|{key}",
                    lambda s=session: analyse_session(s, behaviour, "swap_target_and_preceding_trial_item"))
        rows.append(row)
        output.setdefault("_progress", {})["sessions_done"] = len(rows)
        _flush(output)
        _log(f"  {key} status={row.get('status')} n_surviving={row.get('n_surviving')} "
             f"elapsed={time.time() - t0:.0f}s")
    output.pop("_progress", None)

    computed_rows = [r for r in rows if r.get("status") == "computed"]

    # -----------------------------------------------------------------
    # Reproduction gate -- read live, compared against my own recomputation
    # -----------------------------------------------------------------
    gate_rows = [{"by_tier": {PRIMARY_QUALITY_TIER: {"status": "computed",
                                                       "deviation": r["deviation_gate_arm"]}}}
                 for r in computed_rows]
    recomputed_gate = _pool_cell(gate_rows, PRIMARY_QUALITY_TIER, "deviation", "within_load",
                                  "orthogonality_gate_vs_spike_count")
    recomputed_raw = _pool_cell(gate_rows, PRIMARY_QUALITY_TIER, "deviation", "within_load", "raw_vs_report_error")
    pull_values = [r["delivered_content_specific_serial_pull"]["mean_pull_difference"] for r in computed_rows
                   if r["delivered_content_specific_serial_pull"].get("status") == "computed"]
    recomputed_pull = slope_across_sessions_test(pull_values, alternative="two-sided") if pull_values else \
        {"status": "not_computed"}

    dissociation_reference = _live_dissociation_reference()
    serial_pull_reference = _live_serial_pull_reference()
    identity_checks_ok = all(
        r["reproduced_content_specific_serial_pull_identity_check"]["statuses_match"] for r in computed_rows)

    checks = {
        "deviation_gate_r": _close(recomputed_gate.get("mean_value"),
                                    dissociation_reference["orthogonality_gate_vs_spike_count"]["mean_value"]),
        "deviation_gate_p": _close(recomputed_gate.get("p_value"),
                                    dissociation_reference["orthogonality_gate_vs_spike_count"]["p_value"]),
        "deviation_raw_r": _close(recomputed_raw.get("mean_value"),
                                   dissociation_reference["raw_vs_report_error"]["mean_value"]),
        "deviation_raw_p": _close(recomputed_raw.get("p_value"),
                                   dissociation_reference["raw_vs_report_error"]["p_value"]),
        "content_specific_serial_pull_r": _close(recomputed_pull.get("mean_value"),
                                                  serial_pull_reference["mean_value"]),
        "content_specific_serial_pull_p": _close(recomputed_pull.get("p_value"), serial_pull_reference["p_value"]),
        "per_session_serial_pull_per_trial_reproduction_identity": identity_checks_ok,
    }
    gate_status = "reproduced_exactly" if all(checks.values()) else "not_reproduced"
    gate_result = {
        "status": gate_status, "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "recomputed_deviation_gate": recomputed_gate, "recomputed_deviation_raw": recomputed_raw,
        "recomputed_content_specific_serial_pull_pooled": recomputed_pull,
        "delivered_dissociation_reference_read_live": dissociation_reference,
        "delivered_serial_pull_reference_read_live": serial_pull_reference,
    }
    output["reproduction_gate"] = gate_result
    _flush(output)
    _log(f"reproduction gate: {gate_status} elapsed={time.time() - t0:.0f}s")

    if gate_status != "reproduced_exactly":
        output["branch"] = {"branch": BRANCH_GATE_FAILED}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce; no new number is read")
        print(json.dumps({"reproduction_gate": gate_status, "branch": BRANCH_GATE_FAILED}, indent=2))
        return

    # -----------------------------------------------------------------
    # Zero-drop accounting and the count precondition
    # -----------------------------------------------------------------
    drop_totals: dict[str, int] = {}
    n_arrays_not_computable_sessions = sum(1 for r in rows if r.get("status") == DROP_SESSION_ARRAYS)
    for r in computed_rows:
        for reason, count in r["drop_counts_by_reason"].items():
            drop_totals[reason] = drop_totals.get(reason, 0) + count
    n_trials_in_arrays_not_computable_sessions = sum(
        r.get("n_trials_total", 0) for r in rows if r.get("status") == DROP_SESSION_ARRAYS)
    n_pooled_surviving = drop_totals.get(SURVIVING, 0)
    n_trials_total = sum(drop_totals.values()) + n_trials_in_arrays_not_computable_sessions
    reconciles = bool(n_trials_total == sum(r.get("n_trials_total", 0) for r in rows))

    output["zero_drop_accounting"] = {
        "n_sessions_seen": n_seen, "n_sessions_loaded": len(loaded), "n_sessions_refused_by_shared_loader": len(refused),
        "n_sessions_loaded_but_arrays_not_computable": n_arrays_not_computable_sessions,
        "n_sessions_with_a_computed_arm": len(computed_rows),
        "sessions_reconcile": bool(n_seen == len(loaded) + len(refused)),
        "n_trials_total_across_computed_sessions": sum(drop_totals.values()),
        "n_trials_in_arrays_not_computable_sessions": n_trials_in_arrays_not_computable_sessions,
        "drop_counts_by_reason": drop_totals,
        "n_pooled_surviving_item_count_3_swap_trials_with_admissible_preceding_trial": n_pooled_surviving,
        "trial_counts_reconcile": reconciles,
    }
    _flush(output)
    _log(f"n_pooled_surviving={n_pooled_surviving} elapsed={time.time() - t0:.0f}s")

    if n_pooled_surviving < MIN_POOLED_ADMISSIBLE_SWAP_TRIALS:
        output["branch"] = {
            "branch": BRANCH_TOO_FEW_TRIALS, "n_pooled_surviving": n_pooled_surviving,
            "floor": MIN_POOLED_ADMISSIBLE_SWAP_TRIALS,
        }
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: fewer than the pre-declared floor of pooled admissible swap trials")
        print(json.dumps({"branch": BRANCH_TOO_FEW_TRIALS, "n_pooled_surviving": n_pooled_surviving}, indent=2))
        return

    # -----------------------------------------------------------------
    # Block 0 -- the symmetry premise
    # -----------------------------------------------------------------
    all_separations = [tr["angular_separation_between_uncued_objects_degrees"]
                        for r in computed_rows for tr in r["trial_records"]]
    n_near = sum(1 for v in all_separations if v < NEAR_SEPARATION_THRESHOLD_DEGREES)

    shuffle_by_session: dict[str, list[float | None]] = {}
    for r in computed_rows:
        recs = r["trial_records"]
        if not recs:
            continue
        swap_theta = np.array([tr["swap_theta"] for tr in recs])
        other_theta = np.array([tr["other_theta"] for tr in recs])
        prev_target_theta = np.array([tr["prev_target_theta"] for tr in recs])
        seed_tag = f"swap_target_and_preceding_trial_item|block0|{r['session']}"
        shuffle_by_session[r["session"]] = _block0_shuffle_draws(
            swap_theta, other_theta, prev_target_theta, BLOCK0_N_SHUFFLES, seed_tag)

    pooled_null = []
    for d in range(BLOCK0_N_SHUFFLES):
        draw_vals = [draws[d] for draws in shuffle_by_session.values() if draws[d] is not None]
        if draw_vals:
            pooled_null.append(float(np.mean(draw_vals)))
    null_mean = float(np.mean(pooled_null)) if pooled_null else None
    null_sd = float(np.std(pooled_null, ddof=1)) if len(pooled_null) >= 2 else None
    mc_error = (null_sd / np.sqrt(len(pooled_null))) if null_sd is not None else None
    z_off_center = (abs(null_mean - 0.5) / mc_error) if (null_mean is not None and mc_error) else None
    centred = bool(z_off_center is not None and z_off_center <= BLOCK0_CENTERING_Z_THRESHOLD)

    block0 = {
        "angular_separation_between_the_two_uncued_objects_degrees": {
            "n_trials": len(all_separations),
            "median": float(np.median(all_separations)) if all_separations else None,
            "p5": float(np.percentile(all_separations, 5)) if all_separations else None,
            "p95": float(np.percentile(all_separations, 95)) if all_separations else None,
        },
        "n_trials_within_15_degrees": n_near,
        "near_separation_threshold_degrees": NEAR_SEPARATION_THRESHOLD_DEGREES,
        "shuffle_null": {
            "n_shuffles": BLOCK0_N_SHUFFLES, "n_draws_pooled": len(pooled_null),
            "mean": null_mean, "sd": null_sd, "monte_carlo_error_of_mean": mc_error,
            "z_offset_from_one_half": z_off_center, "centering_z_threshold": BLOCK0_CENTERING_Z_THRESHOLD,
            "centred_on_one_half": centred,
        },
    }
    output["block_0_symmetry_premise"] = block0
    _flush(output)
    _log(f"block 0: centred={centred} null_mean={null_mean} elapsed={time.time() - t0:.0f}s")

    if not centred:
        output["branch"] = {"branch": BRANCH_SYMMETRY_PREMISE_FAILED, "block_0": block0}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: block 0 shuffled null is not centred on 0.5 within its own Monte Carlo error")
        print(json.dumps({"branch": BRANCH_SYMMETRY_PREMISE_FAILED}, indent=2))
        return

    # -----------------------------------------------------------------
    # Block A -- behavioural
    # -----------------------------------------------------------------
    def _block_a_pooled(exclude_near: bool) -> dict:
        by_session_t: dict[str, list[float]] = {}
        by_session_r: dict[str, list[float]] = {}
        n_ties_t = n_ties_r = 0
        for r in computed_rows:
            for tr in r["trial_records"]:
                if exclude_near and tr["angular_separation_between_uncued_objects_degrees"] < NEAR_SEPARATION_THRESHOLD_DEGREES:
                    continue
                if tr["indicator_target"] is None:
                    n_ties_t += 1
                else:
                    by_session_t.setdefault(r["session"], []).append(float(tr["indicator_target"]))
                if tr["indicator_response"] is None:
                    if tr["response_defined_for_preceding_trial"]:
                        n_ties_r += 1
                else:
                    by_session_r.setdefault(r["session"], []).append(float(tr["indicator_response"]))
        target_session_means = [float(np.mean(v)) for v in by_session_t.values()]
        response_session_means = [float(np.mean(v)) for v in by_session_r.values()]
        target_pooled = _pool_vs_half(target_session_means)
        response_pooled = _pool_vs_half(response_session_means)
        return {
            "target_referenced": target_pooled, "response_referenced": response_pooled,
            "n_sessions_contributing_target": len(target_session_means),
            "n_sessions_contributing_response": len(response_session_means),
            "n_ties_target_excluded": n_ties_t, "n_ties_response_excluded": n_ties_r,
        }

    block_a_primary = _block_a_pooled(exclude_near=False)
    block_a_branch = _block_a_branch(block_a_primary["target_referenced"], block_a_primary["response_referenced"],
                                      computed_rows)
    block_a_sensitivity = _block_a_pooled(exclude_near=True)

    output["block_a_behavioural"] = {
        "primary": block_a_primary, "branch": block_a_branch,
        "sensitivity_excluding_near_15_degree_trials": block_a_sensitivity,
    }
    _flush(output)
    _log(f"block A branch: {block_a_branch['branch']} elapsed={time.time() - t0:.0f}s")

    # -----------------------------------------------------------------
    # Block B -- neural
    # -----------------------------------------------------------------
    block_b_vals = _block_b_session_values(computed_rows)
    raw_pooled = _pool_values([v for v in block_b_vals["raw"] if v is not None])
    rate_pooled = _pool_values([v for v in block_b_vals["rate"] if v is not None])
    bias_vals = [v for v in block_b_vals["bias"] if v is not None]
    partial_vals = [v for v in block_b_vals["partial"] if v is not None]
    bias_pooled = _pool_values(bias_vals) if len(bias_vals) >= 2 else {"status": "not_computable", "n_sessions": len(bias_vals)}
    partial_pooled = _pool_values(partial_vals) if len(partial_vals) >= 2 else {"status": "not_computable", "n_sessions": len(partial_vals)}

    reference_effect = abs(serial_pull_reference["mean_value"])
    block_b_branch = _block_b_branch(raw_pooled, rate_pooled, bias_pooled, partial_pooled, reference_effect)

    output["block_b_neural"] = {
        "n_sessions_contributing": len(block_b_vals["sessions"]),
        "per_session_summary": block_b_vals["sessions"],
        "raw_group_difference": raw_pooled,
        "rate_control_group_difference": rate_pooled,
        "bias_only_control_group_difference": bias_pooled,
        "spike_count_partial_group_difference": partial_pooled,
        "reference_effect_size_delivered_pooled_content_specific_serial_pull": reference_effect,
        "branch": block_b_branch,
    }
    _flush(output)
    _log(f"block B branch: {block_b_branch['branch']} elapsed={time.time() - t0:.0f}s")

    output["branch"] = {"block_a": block_a_branch["branch"], "block_b": block_b_branch["branch"]}
    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/swap_target_and_preceding_trial_item_checkpoint.json",
        "git_commit": git_commit(ROOT),
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"complete elapsed={time.time() - t0:.0f}s")
    print(json.dumps({
        "reproduction_gate": gate_status, "n_pooled_surviving": n_pooled_surviving,
        "block_0_centred": centred, "block_a_branch": block_a_branch["branch"],
        "block_b_branch": block_b_branch["branch"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
