"""run_pre_cue_state_and_reported_item_identity.py -- in the multi-object
macaque prefrontal corpus, does the population state maintained BEFORE the
probe cue carry not just that a binding failure is coming, but WHICH of the
uncued objects the eventual report will land on?

The analysed maintenance window in this corpus ends at or before cue onset
on every admitted trial (the loader's own admission rule, verified as a
measurement in this module before anything else runs), so during it the
animal cannot know which object will be probed and cannot be preparing the
eventual saccade. On a swap trial at item count 3, exactly two objects are
uncued; asking which of those two the pre-cue state leans toward is a
two-alternative question with an exact 0.5 null by symmetry, since neither
candidate was cued and no decoder-accuracy calibration is needed.

Two answers are both results, not a success/failure pair. If the pre-cue
decoded position lands closer to the object the animal will actually report
than to the other uncued object, more often than chance, the state carries a
prioritisation among items and the mis-prioritised one is what gets
reported -- a binding failure already present in memory before the probe. If
it does not, above a floor that rules out simple under-power, the state
signals that a failure is coming without carrying which item it will land
on -- the failure is at retrieval, and the accuracy-predicting component is a
vulnerability signal rather than a mis-set pointer.

Reuse, not reinvention. The rate-free per-trial direction (the L2-normalised
across-unit activity vector), the swap/imprecision geometry, the
reproduction gate, and the content-reachability gate all come from modules
already delivered in this project:
  - scripts/run_component_and_item_binding.py: reproduction_gate,
    _object_geometry (swap definition and identity check against the
    corpus's own report-deviation column), CORRECT_REPORT_DISTANCE_THRESHOLD.
  - scripts/run_deviation_subspace_decomposition.py: _leave_one_out_unit_directions
    (exposes the trial direction rate_free_state_deviation forms internally),
    _orthonormal_basis, cv_regression_subspace, _watters_reachability,
    WATTERS_RECOVERABILITY_K_CLASSES.
  - scripts/run_watters_state_geometry.py: _behaviour_observables,
    PRIMARY_QUALITY_TIER, MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION.
  - scripts/run_state_content_link.py: usable_label, MIN_CLASSES,
    MIN_TRIALS_PER_CLASS, BIN_MS.
  - src/statistics.py: paired_sign_flip_test, minimum_detectable_paired_difference,
    fdr_bh, stable_seed.
No estimator here is forked; the only new computation is the ridge decoder
from a trial direction (or a subspace component of it) to the cued object's
[cos, sin] position, the two-alternative report-following statistic built on
its held-out predictions, and the pooling and decision rules over that
statistic.

The decoder input for Block B's two components is built by fitting the cued
object's 2-dimensional regression subspace with the identical leave-one-out
recipe cv_regression_subspace already uses (never a trial's own data in its
own subspace), then projecting the FULL rate-free direction (not a residual)
onto and off that subspace -- proved, at runtime, to give the same
within/outside magnitudes cv_regression_subspace itself returns before any
decoder is fit on the resulting vectors.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import (  # noqa: E402
    WATTERS_DELAY_WINDOW_S, WATTERS_RAW_BIN_MS, data_root, iter_watters, watters_behaviour,
)
from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from run_component_and_item_binding import (  # noqa: E402
    CORRECT_REPORT_DISTANCE_THRESHOLD, _object_geometry, reproduction_gate,
)
from run_deviation_subspace_decomposition import (  # noqa: E402
    WATTERS_RECOVERABILITY_K_CLASSES, WATTERS_REGRESSION_DIM, _leave_one_out_unit_directions,
    _orthonormal_basis, _watters_reachability, cv_regression_subspace,
)
from run_state_content_link import MIN_CLASSES, MIN_TRIALS_PER_CLASS, usable_label  # noqa: E402
from run_watters_state_geometry import (  # noqa: E402
    MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION, PRIMARY_QUALITY_TIER, _behaviour_observables,
)
from statistics import fdr_bh, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "pre_cue_state_and_reported_item_identity.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "pre_cue_state_and_reported_item_identity_checkpoint.json"
ANALYSIS_VERSION = "2026-08-19"
BIN_MS = 100.0

# ----------------------------------------------------------------------------------------------------
# Constants declared before any fit runs.
# ----------------------------------------------------------------------------------------------------

N_FOLDS = 5
RIDGE_ALPHA_GRID = tuple(float(v) for v in np.logspace(-2, 6, 9))
MIN_TRAIN_TRIALS_PER_SESSION = 30   # non-swap, item count >= 2 -- enough rows for a 5-fold ridge fit
MIN_TEST_TRIALS_PER_SESSION = 8     # held-out item-count-3 swap trials -- a session's own fraction
MIN_POOLED_TEST_TRIALS = 200        # across every gate-cleared session, before Block A is read at all
ANGULAR_EQUIDISTANCE_TOLERANCE_RAD = float(np.deg2rad(15.0))
N_SHUFFLE_DRAWS = 1000
POWERED_NULL_MDD_CEILING = 0.05

BRANCH_BLOCK0_STOP = "stopped_pre_cue_window_overlaps_cue_onset"
BRANCH_REPRODUCTION_GATE_FAILED = "void_reproduction_gate_did_not_reproduce"

BRANCH_A_POSITIVE = "pre_cue_population_state_carries_which_item_will_be_reported"
BRANCH_A_BIAS_CONFOUND = "report_following_not_separable_from_a_spatial_decoding_bias"
BRANCH_A_SURPRISE = "pre_cue_state_leans_away_from_the_item_that_will_be_reported"
BRANCH_A_POWERED_NULL = "powered_null_pre_cue_state_does_not_carry_the_item_that_will_be_reported"
BRANCH_A_INCONCLUSIVE = "inconclusive_below_detection_floor"
BRANCH_A_TOO_FEW = "too_few_three_item_swap_trials_to_test"

BRANCH_B_OUTSIDE = "reported_item_identity_is_carried_outside_the_memorandum_subspace"
BRANCH_B_INSIDE = "reported_item_identity_is_carried_inside_the_memorandum_subspace"
BRANCH_B_BOTH = "carried_by_both_components"
BRANCH_B_NEITHER = "neither_component_carries_it"
BRANCH_B_ORDERING_NOT_ESTABLISHED = "component_ordering_not_established_by_a_paired_test"
BRANCH_B_NOT_RUN = "not_run_block_a_did_not_clear_the_swap_trial_floor"

BRANCH_TASK_GEOM_REPRODUCED = "swap_destination_bias_is_reproduced_by_task_geometry_alone"
BRANCH_TASK_GEOM_NEEDS_DECODED = "swap_destination_bias_needs_the_decoded_position"
BRANCH_TASK_GEOM_NOT_SEPARABLE = "swap_destination_bias_not_separable_from_task_geometry_at_this_power"
BRANCH_TASK_GEOM_POWERED_NULL = "powered_null_no_swap_destination_bias_under_either_reference"
BRANCH_TASK_GEOM_NOT_COVERED = "swap_destination_bias_outcome_not_covered_by_the_declared_rule"

DECISION_RULE_BLOCK_A_DECLARED_BEFORE_FITTING = (
    "Let A be the pooled fraction (across gate-cleared sessions, one fraction per session, two-sided "
    "paired sign-flip test against the exact null 0.5) of held-out item-count-3 swap trials on which the "
    "ridge decoder's predicted angle (fit on non-swap item-count>=2 trials, from the trial's full "
    "rate-free direction to the cued object's [cos, sin] position) is angularly closer to the REPORTED "
    "uncued object than to the OTHER uncued object. Let C be the identical statistic computed with each "
    "test trial's decoded position replaced by the session's own mean decoded position over its training "
    "trials (the bias-only control). Both A and C, and every control below, are computed before any "
    "branch is read. In order:\n"
    f"  0. If the reproduction gate does not reproduce at tolerance 1e-6, branch is "
    f"'{BRANCH_REPRODUCTION_GATE_FAILED}' and no further number is read.\n"
    f"  1. If the pooled count of held-out item-count-3 swap trials across every session that clears the "
    f"content-reachability gate is below {MIN_POOLED_TEST_TRIALS}, branch is '{BRANCH_A_TOO_FEW}', with "
    "the count -- declared before any swap count at item count 3 in this corpus is looked at.\n"
    f"  2. A significantly above 0.5 and C not significantly above 0.5 -> '{BRANCH_A_POSITIVE}'.\n"
    f"  3. A significantly above 0.5 and C also significantly above 0.5 -> '{BRANCH_A_BIAS_CONFOUND}', "
    "void, reported with both fractions and a direct paired session-level test of A against C.\n"
    f"  4. A significantly BELOW 0.5 -> '{BRANCH_A_SURPRISE}', reported as a named surprise with its "
    "effect size, never folded into a positive branch.\n"
    f"  5. A not significant and its minimum detectable departure from 0.5 at 80% power is below "
    f"{POWERED_NULL_MDD_CEILING} -> '{BRANCH_A_POWERED_NULL}'. This is a result: the artifact states in "
    "its own words that it places the binding failure at retrieval rather than in the maintained state.\n"
    f"  6. A not significant and its minimum detectable departure is at or above "
    f"{POWERED_NULL_MDD_CEILING} -> '{BRANCH_A_INCONCLUSIVE}', with the minimum detectable departure "
    "reported.\n"
    "If an outcome occurs that this list does not cover, that is reported in writing as a gap in this "
    "rule, with the numbers, never forced onto the nearest label, and this rule is not amended after the "
    "fact."
)

DECISION_RULE_BLOCK_B_DECLARED_BEFORE_FITTING = (
    "Runs only if Block A's branch is not the swap-trial-floor branch. Block A's primary statistic is "
    "repeated twice more, changing only the vector the decoder reads: the component of the trial's full "
    "rate-free direction OUTSIDE the cued object's 2-dimensional regression subspace (component 2, fit by "
    "the identical leave-one-out recipe cv_regression_subspace already uses), and the component INSIDE "
    "that subspace (component 3). Each is pooled the same way A is pooled and corrected across the two "
    "components by Benjamini-Hochberg at alpha 0.05 (on the raw two-sided p-values against 0.5). A direct "
    "paired session-level test (two-sided paired sign-flip test) compares component 2 against component 3 "
    "on the same sessions; two separate significance verdicts are not by themselves a difference.\n"
    f"  - Both components significant after correction -> '{BRANCH_B_BOTH}', reported with the paired test "
    "and their across-session collinearity, resolved in favour of neither.\n"
    f"  - Exactly one component significant after correction AND the paired test is significant -> "
    f"'{BRANCH_B_OUTSIDE}' (component 2) or '{BRANCH_B_INSIDE}' (component 3).\n"
    f"  - Exactly one component significant after correction but the paired test is NOT significant -> "
    f"'{BRANCH_B_ORDERING_NOT_ESTABLISHED}'; no ordering is stated.\n"
    f"  - Neither component significant after correction -> '{BRANCH_B_NEITHER}', with both minimum "
    "detectable departures from 0.5 at 80% power."
)

DECISION_RULE_SWAP_DESTINATION_TASK_GEOMETRY_DECLARED_BEFORE_FITTING = (
    "The bias-only control C (the fraction statistic built from each session's own mean decoded position "
    "over its training trials, evaluated against the exact 0.5 null) is compared against an identical "
    "statistic built from a task-design quantity carrying no neural signal: each session's own mean CUED "
    "object position over the identical training trials, in place of its mean decoded position. Both are "
    "pooled across sessions by the same two-sided paired sign-flip test against 0.5, and a direct paired "
    "session-level test compares the two pooled statistics on the same sessions. In order:\n"
    f"  1. The task-design version is significantly below 0.5 and the paired test against C is NOT "
    f"significant -> '{BRANCH_TASK_GEOM_REPRODUCED}'.\n"
    f"  2. C is significantly below 0.5, the task-design version is NOT significant, and the paired test "
    f"IS significant -> '{BRANCH_TASK_GEOM_NEEDS_DECODED}'.\n"
    f"  3. The paired test is not significant and its minimum detectable paired difference at 80% power "
    f"exceeds C's own departure from 0.5 -> '{BRANCH_TASK_GEOM_NOT_SEPARABLE}'.\n"
    f"  4. Neither statistic is significant and both minimum detectable departures from 0.5 at 80% power "
    f"are below 0.05 -> '{BRANCH_TASK_GEOM_POWERED_NULL}'.\n"
    "If an outcome occurs that this list does not cover, that is reported in writing, with every number, "
    f"under '{BRANCH_TASK_GEOM_NOT_COVERED}' -- never forced onto the nearest label -- and this rule is not "
    "amended after the fact."
)


def circular_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped absolute angular difference in [0, pi] between two angle arrays (radians)."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


# ----------------------------------------------------------------------------------------------------
# Block 0 -- the timing premise, recorded as a measurement before any decoder is fitted.
# ----------------------------------------------------------------------------------------------------

def block0_timing_premise(loaded_sessions: list[dict], behaviour) -> dict:
    """Median, minimum and 5th percentile of time_cue_onset - time_delay_onset over every analysed
    trial, pooled across every loaded session, plus the loader's own count of trials it already refused
    for a too-short delay. The window overlaps cue onset on an analysed trial only if that trial's own
    margin falls below the loader's own admission threshold (the window length minus the loader's own
    quantisation slack) -- the identical inequality src/corpus_sessions.py already enforces at load time,
    reused here as a measurement rather than re-derived."""
    margins: list[np.ndarray] = []
    n_dropped_delay_shorter = 0
    for session in loaded_sessions:
        rows = behaviour.loc[(session["animal"], session["session_date"])]
        trial_rows = rows.loc[session["trial_num"].tolist()]
        margins.append((trial_rows["time_cue_onset"] - trial_rows["time_delay_onset"]).to_numpy(dtype=float))
        n_dropped_delay_shorter += int(session.get("trials_dropped_by_reason", {}).get(
            "delay_shorter_than_window", 0))
    pooled = np.concatenate(margins) if margins else np.array([], dtype=float)
    slack_s = WATTERS_RAW_BIN_MS / 2000.0
    threshold_s = WATTERS_DELAY_WINDOW_S - slack_s
    n_overlapping = int(np.sum(pooled < threshold_s)) if pooled.size else 0
    return {
        "n_trials_pooled": int(pooled.size),
        "median_margin_s": float(np.median(pooled)) if pooled.size else None,
        "min_margin_s": float(np.min(pooled)) if pooled.size else None,
        "p5_margin_s": float(np.percentile(pooled, 5)) if pooled.size else None,
        "analysed_window_length_s": WATTERS_DELAY_WINDOW_S,
        "loader_admission_slack_s": slack_s,
        "loader_admission_threshold_s": threshold_s,
        "n_trials_overlapping_cue_onset_by_the_loaders_own_threshold": n_overlapping,
        "n_trials_the_loader_already_refused_for_a_too_short_delay": n_dropped_delay_shorter,
        "stop_condition_triggered": bool(n_overlapping > 0),
    }


# ----------------------------------------------------------------------------------------------------
# Landed/target object index, reimplemented locally (the same nearest-object arithmetic
# run_component_and_item_binding._object_geometry already uses) and proved to agree with that
# function's own swap_primary output before its landed index is used for anything.
# ----------------------------------------------------------------------------------------------------

def _landed_and_target_index(behaviour, session: dict) -> dict:
    rows = behaviour.loc[(session["animal"], session["session_date"])]
    trial_rows = rows.loc[session["trial_num"].tolist()]
    object_x = trial_rows[[f"object_{i}_x" for i in range(3)]].to_numpy(dtype=float)
    object_y = trial_rows[[f"object_{i}_y" for i in range(3)]].to_numpy(dtype=float)
    object_theta = trial_rows[[f"object_{i}_theta" for i in range(3)]].to_numpy(dtype=float)
    response_x = trial_rows["response_x"].to_numpy(dtype=float)
    response_y = trial_rows["response_y"].to_numpy(dtype=float)
    target = trial_rows["target_object_index"].to_numpy(dtype=int)
    n = len(target)

    distances = np.hypot(object_x - response_x[:, None], object_y - response_y[:, None])
    ok = ~np.all(np.isnan(distances), axis=1)
    landed = np.full(n, -1, dtype=int)
    landed[ok] = np.nanargmin(distances[ok], axis=1)

    return {"landed": landed, "target": target, "object_theta": object_theta, "ok": ok}


def _landed_identity_check(behaviour, session: dict, geometry: dict, landed_info: dict) -> dict:
    recomputed_swap = landed_info["ok"] & (landed_info["landed"] != landed_info["target"])
    identical = bool(np.array_equal(recomputed_swap, geometry["swap_primary"]))
    n = len(landed_info["target"])
    target_theta = landed_info["object_theta"][np.arange(n), np.clip(landed_info["target"], 0, 2)]
    cued_theta = np.asarray(session["cued_theta"], dtype=float)
    finite = np.isfinite(target_theta) & np.isfinite(cued_theta)
    cued_theta_diff = float(np.max(circular_abs_diff(target_theta[finite], cued_theta[finite]))) if finite.any() \
        else None
    return {
        "swap_primary_matches_object_geometry": identical,
        "cued_theta_matches_target_object_theta_max_abs_diff": cued_theta_diff,
    }


# ----------------------------------------------------------------------------------------------------
# Block B's subspace decomposition of the FULL direction (not a residual), with vectors exposed and
# proved to agree with cv_regression_subspace's own within/outside magnitudes.
# ----------------------------------------------------------------------------------------------------

def _loo_subspace_component_vectors(u: np.ndarray, target_2d: np.ndarray,
                                     dim: int = WATTERS_REGRESSION_DIM) -> tuple[np.ndarray, np.ndarray]:
    """For every trial i, fit the dim-dimensional regression subspace (each unit's direction regressed
    on the cued object's [cos, sin]) from every trial OTHER than i, and project trial i's own direction
    u_i onto and off that subspace -- the identical leave-one-out recipe cv_regression_subspace uses,
    exposing the projected VECTORS rather than collapsing them to a norm."""
    n = u.shape[0]
    inside = np.zeros_like(u)
    outside = np.zeros_like(u)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        x_c = target_2d[keep] - target_2d[keep].mean(axis=0)
        u_c = u[keep] - u[keep].mean(axis=0)
        coefficients, *_ = np.linalg.lstsq(x_c, u_c, rcond=None)
        basis = _orthonormal_basis(coefficients, dim)
        proj = basis @ (basis.T @ u[i])
        inside[i] = proj
        outside[i] = u[i] - proj
    return inside, outside


def _subspace_decomposition_identity_check(u: np.ndarray, target_2d: np.ndarray, inside: np.ndarray,
                                            outside: np.ndarray, dim: int) -> dict:
    within_ref, outside_ref = cv_regression_subspace(u, target_2d, u, dim=dim)
    diff_within = float(np.max(np.abs(np.linalg.norm(inside, axis=1) - within_ref)))
    diff_outside = float(np.max(np.abs(np.linalg.norm(outside, axis=1) - outside_ref)))
    return {
        "max_abs_diff_inside_norm_vs_cv_regression_subspace": diff_within,
        "max_abs_diff_outside_norm_vs_cv_regression_subspace": diff_outside,
        "passed": bool(diff_within < 1e-8 and diff_outside < 1e-8),
    }


# ----------------------------------------------------------------------------------------------------
# Ridge decoder
# ----------------------------------------------------------------------------------------------------

def _ridge_decode(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, fold_seed: int) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    if x_train.shape[0] < N_FOLDS or x_test.shape[0] < 1:
        return {"status": "not_computable"}
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
    model = RidgeCV(alphas=RIDGE_ALPHA_GRID, cv=cv)
    model.fit(x_train, y_train)
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    decoded_theta_test = np.arctan2(y_pred_test[:, 1], y_pred_test[:, 0])
    bias_position = y_pred_train.mean(axis=0)
    bias_theta = float(np.arctan2(bias_position[1], bias_position[0]))
    return {
        "status": "computed", "decoded_theta_test": decoded_theta_test, "bias_theta": bias_theta,
        "alpha_selected": float(model.alpha_), "n_train": int(x_train.shape[0]), "n_test": int(x_test.shape[0]),
    }


def _report_following_fraction(decoded_theta: np.ndarray, reported_theta: np.ndarray,
                                other_theta: np.ndarray) -> dict:
    d_reported = circular_abs_diff(decoded_theta, reported_theta)
    d_other = circular_abs_diff(decoded_theta, other_theta)
    closer_to_reported = d_reported < d_other
    n_ties = int(np.sum(d_reported == d_other))
    return {"fraction": float(np.mean(closer_to_reported)), "n_trials": int(len(decoded_theta)),
            "n_ties": n_ties, "closer_to_reported": closer_to_reported}


def _shuffled_report_null(decoded_theta: np.ndarray, reported_theta: np.ndarray, other_theta: np.ndarray,
                           observed_fraction: float, seed: int) -> dict:
    n = len(decoded_theta)
    if n == 0:
        return {"status": "not_computable"}
    rng = np.random.default_rng(seed)
    flip = rng.random((N_SHUFFLE_DRAWS, n)) < 0.5
    reported_draws = np.where(flip, other_theta[None, :], reported_theta[None, :])
    other_draws = np.where(flip, reported_theta[None, :], other_theta[None, :])
    d_rep = circular_abs_diff(decoded_theta[None, :], reported_draws)
    d_oth = circular_abs_diff(decoded_theta[None, :], other_draws)
    null_fractions = (d_rep < d_oth).mean(axis=1)
    percentile = float(np.mean(null_fractions <= observed_fraction))
    two_sided_p = float((np.sum(np.abs(null_fractions - 0.5) >= np.abs(observed_fraction - 0.5)) + 1) /
                         (N_SHUFFLE_DRAWS + 1))
    return {
        "status": "computed", "n_draws": N_SHUFFLE_DRAWS, "null_mean": float(np.mean(null_fractions)),
        "null_std": float(np.std(null_fractions)), "observed_percentile_in_null": percentile,
        "two_sided_p_value": two_sided_p,
    }


# ----------------------------------------------------------------------------------------------------
# Per-session analysis
# ----------------------------------------------------------------------------------------------------

def analyse_session(session: dict, behaviour, seed_prefix: str) -> dict:
    counts = session["counts"]
    activity = counts.sum(axis=2)
    observables, excluded, usable = _behaviour_observables(counts, session)
    n_usable = int(usable.sum())
    base = {
        "animal": session["animal"], "session_date": session["session_date"], "session": session["session"],
        "task_variant": session["task_variant"], "n_units": int(session["n_units"]),
        "n_trials_total": int(counts.shape[0]), "n_trials_usable": n_usable,
        "trials_excluded_by_reason": excluded, "analysis_version": ANALYSIS_VERSION,
    }
    if n_usable < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {**base, "status": "too_few_usable_trials"}

    geometry = _object_geometry(behaviour, session)
    landed_info = _landed_and_target_index(behaviour, session)
    identity = _landed_identity_check(behaviour, session, geometry, landed_info)
    base["landed_index_identity_check"] = identity
    if not identity["swap_primary_matches_object_geometry"]:
        return {**base, "status": "landed_index_identity_check_failed"}

    item_count = np.asarray(session["num_objects"], dtype=float)
    swap_primary = geometry["swap_primary"]
    train_mask = usable & (item_count >= 2) & (~swap_primary)
    test_mask = usable & (item_count == 3) & swap_primary
    analysis_mask = train_mask | test_mask
    base["n_train_candidate_non_swap_ge2"] = int(train_mask.sum())
    base["n_test_candidate_swap_item_count_3"] = int(test_mask.sum())

    cued_theta = np.asarray(session["cued_theta"], dtype=float)
    theta_mod = np.mod(cued_theta, 2.0 * np.pi)
    discretized_label = (np.floor(theta_mod / (2.0 * np.pi / WATTERS_RECOVERABILITY_K_CLASSES)).astype(int)
                          % WATTERS_RECOVERABILITY_K_CLASSES)

    idx_analysis = np.flatnonzero(analysis_mask)
    if idx_analysis.size < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {**base, "status": "too_few_train_or_test_trials"}
    label_ok, label_reason, label_submask = usable_label(
        discretized_label[idx_analysis], min_classes=MIN_CLASSES, min_per_class=MIN_TRIALS_PER_CLASS)
    if not label_ok:
        return {**base, "status": "no_usable_discretised_label_for_reachability_gate", "reason": label_reason}
    final_idx = idx_analysis[label_submask]

    tag = f"{seed_prefix}|{session['session']}"
    gate = _watters_reachability(counts[final_idx], discretized_label[final_idx],
                                  stable_seed(f"{tag}|reachability"))
    base["content_reachability"] = gate
    if gate.get("status") != "tested":
        return {**base, "status": "content_reachability_not_computable"}
    if not gate.get("cleared", False):
        return {**base, "status": "content_not_reachable_void"}

    train_sub = train_mask[final_idx]
    test_sub = test_mask[final_idx]
    n_train, n_test = int(train_sub.sum()), int(test_sub.sum())
    base["n_train_after_gate"] = n_train
    base["n_test_after_gate"] = n_test
    if n_train < MIN_TRAIN_TRIALS_PER_SESSION or n_test < MIN_TEST_TRIALS_PER_SESSION:
        return {**base, "status": "too_few_train_or_test_trials_after_gate"}

    unit_vectors_all = _leave_one_out_unit_directions(activity)["unit_vectors"]
    u = unit_vectors_all[final_idx]
    target_2d = np.stack([np.cos(cued_theta[final_idx]), np.sin(cued_theta[final_idx])], axis=1)
    object_theta = landed_info["object_theta"][final_idx]
    landed = landed_info["landed"][final_idx]
    target = landed_info["target"][final_idx]
    other_idx = 3 - target - landed

    x_train_full = u[train_sub]
    x_test_full = u[test_sub]
    y_train = target_2d[train_sub]
    fold_seed = stable_seed(f"{tag}|ridge_folds")

    decode_full = _ridge_decode(x_train_full, y_train, x_test_full, fold_seed)
    if decode_full.get("status") != "computed":
        return {**base, "status": "ridge_decoder_not_computable"}

    n = np.arange(int(test_sub.sum()))
    reported_theta = object_theta[test_sub][n, landed[test_sub]]
    other_theta = object_theta[test_sub][n, other_idx[test_sub]]

    primary = _report_following_fraction(decode_full["decoded_theta_test"], reported_theta, other_theta)
    bias_theta_arr = np.full(n_test, decode_full["bias_theta"])
    bias_control = _report_following_fraction(bias_theta_arr, reported_theta, other_theta)

    # The task-geometry control for C: the session's own mean CUED object position over the identical
    # training trials, carrying no neural signal at all, in place of the session's mean DECODED position.
    task_geometry_position = y_train.mean(axis=0)
    task_geometry_theta = float(np.arctan2(task_geometry_position[1], task_geometry_position[0]))
    task_geometry_arr = np.full(n_test, task_geometry_theta)
    task_geometry_control = _report_following_fraction(task_geometry_arr, reported_theta, other_theta)

    d_cued_reported = circular_abs_diff(cued_theta[final_idx][test_sub], reported_theta)
    d_cued_other = circular_abs_diff(cued_theta[final_idx][test_sub], other_theta)
    equidistant = np.abs(d_cued_reported - d_cued_other) <= ANGULAR_EQUIDISTANCE_TOLERANCE_RAD
    n_equidistant = int(equidistant.sum())
    equidistance_sensitivity = (
        _report_following_fraction(decode_full["decoded_theta_test"][equidistant], reported_theta[equidistant],
                                    other_theta[equidistant])
        if n_equidistant >= 1 else {"status": "not_computable", "n_trials": 0}
    )

    shuffled_null = _shuffled_report_null(decode_full["decoded_theta_test"], reported_theta, other_theta,
                                           primary["fraction"], stable_seed(f"{tag}|shuffled_report_null"))

    row: dict = {
        **base, "status": "computed",
        "alpha_selected_full_direction": decode_full["alpha_selected"],
        "block_a": {
            "primary_fraction": primary["fraction"], "n_test_trials": primary["n_trials"],
            "n_ties": primary["n_ties"],
            "bias_only_control_fraction": bias_control["fraction"],
            "bias_only_control_bias_theta": decode_full["bias_theta"],
            "task_geometry_control_fraction": task_geometry_control["fraction"],
            "task_geometry_control_theta": task_geometry_theta,
            "equidistance_sensitivity": {
                "tolerance_rad": ANGULAR_EQUIDISTANCE_TOLERANCE_RAD, "n_trials": n_equidistant,
                "fraction": equidistance_sensitivity.get("fraction"),
            },
            "shuffled_report_null": shuffled_null,
        },
    }

    inside, outside = _loo_subspace_component_vectors(u, target_2d, dim=WATTERS_REGRESSION_DIM)
    subspace_identity = _subspace_decomposition_identity_check(u, target_2d, inside, outside,
                                                                 WATTERS_REGRESSION_DIM)
    row["block_b_subspace_decomposition_identity_check"] = subspace_identity

    decode_outside = _ridge_decode(outside[train_sub], y_train, outside[test_sub], fold_seed)
    decode_inside = _ridge_decode(inside[train_sub], y_train, inside[test_sub], fold_seed)
    block_b = {}
    for name, decoded in (("outside_subspace", decode_outside), ("inside_subspace", decode_inside)):
        if decoded.get("status") != "computed":
            block_b[name] = {"status": "not_computable"}
            continue
        frac = _report_following_fraction(decoded["decoded_theta_test"], reported_theta, other_theta)
        component_bias_arr = np.full(n_test, decoded["bias_theta"])
        component_bias = _report_following_fraction(component_bias_arr, reported_theta, other_theta)
        block_b[name] = {"fraction": frac["fraction"], "n_test_trials": frac["n_trials"],
                          "alpha_selected": decoded["alpha_selected"],
                          "bias_only_control_fraction": component_bias["fraction"],
                          "bias_only_control_bias_theta": decoded["bias_theta"]}
    row["block_b"] = block_b
    return row


# ----------------------------------------------------------------------------------------------------
# Pooling and decision rules
# ----------------------------------------------------------------------------------------------------

def _pool_against_half(values: np.ndarray, seed_tag: str) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"status": "too_few_sessions", "n_sessions": int(len(values))}
    rng = np.random.default_rng(stable_seed(seed_tag))
    half = np.full(len(values), 0.5)
    test = paired_sign_flip_test(values, half, alternative="two-sided", rng=rng)
    mdd = minimum_detectable_paired_difference(values)
    mean_diff = test["mean_diff"]
    return {
        "status": "tested", "n_sessions": int(len(values)), "mean_fraction": float(np.mean(values)),
        "mean_diff_from_half": mean_diff, "p_value": test["p_value"],
        "ci_lower": test["ci_lower"] + 0.5, "ci_upper": test["ci_upper"] + 0.5,
        "significant": bool(test["p_value"] < 0.05),
        "significant_above_half": bool(test["p_value"] < 0.05 and mean_diff > 0.0),
        "significant_below_half": bool(test["p_value"] < 0.05 and mean_diff < 0.0),
        "minimum_detectable_departure_from_half_at_80pct_power": mdd,
    }


def _paired_session_test(a: np.ndarray, b: np.ndarray, seed_tag: str) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return {"status": "not_computable", "n_sessions": int(len(a))}
    rng = np.random.default_rng(stable_seed(seed_tag))
    test = paired_sign_flip_test(a, b, alternative="two-sided", rng=rng)
    return {
        "status": "computed", "n_sessions": int(len(a)), "mean_diff": test["mean_diff"],
        "p_value": test["p_value"], "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "significant": bool(test["p_value"] < 0.05),
    }


def _paired_session_test_full(a: np.ndarray, b: np.ndarray, seed_tag: str) -> dict:
    """`_paired_session_test` plus the paired difference's own standard deviation and minimum detectable
    difference at 80% power -- the two extra fields a separation test needs beside its p-value to say
    whether a non-significant result reflects no difference or too little power to see one."""
    base = _paired_session_test(a, b, seed_tag)
    if base.get("status") != "computed":
        return base
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {**base, "sd": float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else None,
            "minimum_detectable_paired_difference_at_80pct_power": minimum_detectable_paired_difference(diffs)}


def decide_block_a(gate_status: str, total_pooled_test_trials: int, pooled_a: dict, pooled_c: dict,
                    a_values: np.ndarray, c_values: np.ndarray) -> dict:
    if gate_status != "reproduced_exactly":
        return {"branch": BRANCH_REPRODUCTION_GATE_FAILED}
    if total_pooled_test_trials < MIN_POOLED_TEST_TRIALS:
        return {"branch": BRANCH_A_TOO_FEW, "n_pooled_test_trials_across_gate_cleared_sessions":
                 total_pooled_test_trials, "floor": MIN_POOLED_TEST_TRIALS,
                 "pooled_a_reported_as_context_not_deciding_the_branch": pooled_a,
                 "pooled_c_reported_as_context_not_deciding_the_branch": pooled_c}

    if pooled_a.get("status") != "tested":
        return {"branch": BRANCH_A_INCONCLUSIVE, "reason": "pooled primary statistic not computable",
                "pooled_a": pooled_a}

    a_above = pooled_a["significant_above_half"]
    a_below = pooled_a["significant_below_half"]
    c_above = pooled_c.get("status") == "tested" and pooled_c["significant_above_half"]

    # The direct paired test of A against C is reported beside every branch, not only the one it
    # historically decided, so a reader can see the separation test's own power before trusting any
    # branch that turns on A and C's relationship.
    paired_a_vs_c = (
        _paired_session_test_full(a_values, c_values,
                                   "pre_cue_state_and_reported_item_identity|block_a|paired_a_vs_c")
        if pooled_c.get("status") == "tested" else {"status": "not_computable"}
    )
    detection_floor_statement = None
    if paired_a_vs_c.get("status") == "computed":
        mdd_block = paired_a_vs_c.get("minimum_detectable_paired_difference_at_80pct_power", {})
        mdd_value = mdd_block.get("mdd") if mdd_block.get("status") == "computed" else None
        a_departure = abs(pooled_a["mean_diff_from_half"])
        if mdd_value is not None:
            if mdd_value > a_departure:
                detection_floor_statement = (
                    f"The direct paired test of A against C has a minimum detectable paired difference at "
                    f"80% power of {mdd_value}, which is ABOVE A's own departure from 0.5 of {a_departure}. "
                    "This separation test is underpowered to rule out a difference the size of A's own "
                    "effect, so a non-significant paired result here cannot be read as evidence that A and "
                    "C are the same."
                )
            else:
                detection_floor_statement = (
                    f"The direct paired test of A against C has a minimum detectable paired difference at "
                    f"80% power of {mdd_value}, which is AT OR BELOW A's own departure from 0.5 of "
                    f"{a_departure}. This separation test is powered to detect a difference at least as "
                    "large as A's own effect, so its result speaks directly to whether A and C differ."
                )

    result: dict = {"pooled_a": pooled_a, "pooled_c": pooled_c, "paired_test_a_vs_c": paired_a_vs_c,
                     "paired_test_a_vs_c_detection_floor_statement": detection_floor_statement}
    if a_above and not c_above:
        result["branch"] = BRANCH_A_POSITIVE
    elif a_above and c_above:
        result["branch"] = BRANCH_A_BIAS_CONFOUND
    elif a_below:
        result["branch"] = BRANCH_A_SURPRISE
    else:
        mdd = pooled_a.get("minimum_detectable_departure_from_half_at_80pct_power", {})
        mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
        if mdd_value is not None and mdd_value < POWERED_NULL_MDD_CEILING:
            result["branch"] = BRANCH_A_POWERED_NULL
            result["retrieval_locus_statement"] = (
                "The pre-cue population state predicts that a binding failure is coming without carrying "
                "which uncued item that failure will land on: at this power, the decoded pre-cue position "
                "is no closer to the object the animal will actually report than to the other uncued "
                "object. The failure this project's accuracy-predicting component signals is therefore not "
                "yet resolved to a specific item during the analysed pre-cue window; whatever selects which "
                "item the report lands on happens after this window, closer to retrieval."
            )
        else:
            result["branch"] = BRANCH_A_INCONCLUSIVE
    return result


def _component_ok(pooled: dict, q_reject: bool | None) -> bool:
    return pooled.get("status") == "tested" and bool(q_reject)


def decide_block_b(pooled_outside: dict, pooled_inside: dict, outside_values: np.ndarray,
                    inside_values: np.ndarray) -> dict:
    p_outside = pooled_outside.get("p_value") if pooled_outside.get("status") == "tested" else None
    p_inside = pooled_inside.get("p_value") if pooled_inside.get("status") == "tested" else None
    if p_outside is None or p_inside is None:
        return {"branch": BRANCH_B_NEITHER, "pooled_outside": pooled_outside, "pooled_inside": pooled_inside,
                "note": "at least one component's pooled statistic was not computable"}

    fdr = fdr_bh(np.array([p_outside, p_inside], dtype=float), alpha=0.05)
    outside_q, inside_q = float(fdr["q_values"][0]), float(fdr["q_values"][1])
    outside_reject, inside_reject = bool(fdr["reject"][0]), bool(fdr["reject"][1])
    outside_ok = _component_ok(pooled_outside, outside_reject)
    inside_ok = _component_ok(pooled_inside, inside_reject)

    paired = _paired_session_test(outside_values, inside_values,
                                   "pre_cue_state_and_reported_item_identity|block_b|paired_outside_vs_inside")

    result: dict = {
        "pooled_outside": {**pooled_outside, "q_value": outside_q, "q_significant": outside_reject},
        "pooled_inside": {**pooled_inside, "q_value": inside_q, "q_significant": inside_reject},
        "paired_test_outside_vs_inside": paired,
    }

    if outside_ok and inside_ok:
        result["branch"] = BRANCH_B_BOTH
        if len(outside_values) >= 2 and len(inside_values) >= 2 and len(outside_values) == len(inside_values):
            r = float(np.corrcoef(outside_values, inside_values)[0, 1]) if np.std(outside_values) > 0 and \
                np.std(inside_values) > 0 else None
            result["collinearity_outside_vs_inside_across_sessions"] = r
    elif outside_ok and not inside_ok:
        result["branch"] = BRANCH_B_OUTSIDE if paired.get("significant") else BRANCH_B_ORDERING_NOT_ESTABLISHED
    elif inside_ok and not outside_ok:
        result["branch"] = BRANCH_B_INSIDE if paired.get("significant") else BRANCH_B_ORDERING_NOT_ESTABLISHED
    else:
        result["branch"] = BRANCH_B_NEITHER
    return result


def decide_swap_destination_task_geometry(pooled_c: dict, pooled_task_geometry: dict,
                                           paired_c_vs_task_geometry: dict) -> dict:
    c_below = pooled_c.get("status") == "tested" and pooled_c["significant_below_half"]
    task_below = pooled_task_geometry.get("status") == "tested" and pooled_task_geometry["significant_below_half"]
    paired_computed = paired_c_vs_task_geometry.get("status") == "computed"
    paired_significant = paired_computed and paired_c_vs_task_geometry["significant"]

    paired_mdd_block = paired_c_vs_task_geometry.get("minimum_detectable_paired_difference_at_80pct_power", {}) \
        if paired_computed else {}
    paired_mdd = paired_mdd_block.get("mdd") if paired_mdd_block.get("status") == "computed" else None
    c_departure = abs(pooled_c["mean_diff_from_half"]) if pooled_c.get("status") == "tested" else None
    c_mdd_block = pooled_c.get("minimum_detectable_departure_from_half_at_80pct_power", {})
    c_mdd = c_mdd_block.get("mdd") if c_mdd_block.get("status") == "computed" else None
    task_mdd_block = pooled_task_geometry.get("minimum_detectable_departure_from_half_at_80pct_power", {})
    task_mdd = task_mdd_block.get("mdd") if task_mdd_block.get("status") == "computed" else None

    result: dict = {
        "pooled_c_bias_only_control": pooled_c, "pooled_task_geometry_control": pooled_task_geometry,
        "paired_test_c_vs_task_geometry": paired_c_vs_task_geometry,
    }
    if task_below and not paired_significant:
        result["branch"] = BRANCH_TASK_GEOM_REPRODUCED
    elif c_below and not task_below and paired_significant:
        result["branch"] = BRANCH_TASK_GEOM_NEEDS_DECODED
    elif (not paired_significant) and paired_mdd is not None and c_departure is not None \
            and paired_mdd > c_departure:
        result["branch"] = BRANCH_TASK_GEOM_NOT_SEPARABLE
    elif (not c_below) and (not task_below) and c_mdd is not None and task_mdd is not None \
            and c_mdd < 0.05 and task_mdd < 0.05:
        result["branch"] = BRANCH_TASK_GEOM_POWERED_NULL
    else:
        result["branch"] = BRANCH_TASK_GEOM_NOT_COVERED
    return result


def _primary_below_half_bias_only_control_gap_statement(pooled_a: dict, pooled_c: dict) -> str:
    """Reports, in the artifact's own words and from numbers read live rather than assumed, whether the
    delivered configuration (A and C both significantly below 0.5, C the larger magnitude) recurs on
    reproduction -- the configuration the ordered decision list's below-0.5 step does not cover with a
    bias-only clause the way its above-0.5 step does."""
    a_tested = pooled_a.get("status") == "tested"
    c_tested = pooled_c.get("status") == "tested"
    a_dep = pooled_a.get("mean_diff_from_half") if a_tested else None
    c_dep = pooled_c.get("mean_diff_from_half") if c_tested else None
    same_direction_below = bool(a_tested and c_tested and pooled_a["significant_below_half"]
                                 and pooled_c["significant_below_half"])
    larger_magnitude = bool(same_direction_below and c_dep is not None and a_dep is not None
                             and abs(c_dep) > abs(a_dep))
    header = (
        f"Primary statistic A: mean fraction {pooled_a.get('mean_fraction')}, departure from 0.5 "
        f"{a_dep}, p={pooled_a.get('p_value')}. Bias-only control C: mean fraction "
        f"{pooled_c.get('mean_fraction')}, departure from 0.5 {c_dep}, p={pooled_c.get('p_value')}. "
    )
    if same_direction_below and larger_magnitude:
        return header + (
            "The bias-only control fires significantly in the same direction as the primary statistic and "
            "at a larger magnitude. The ordered decision list's below-0.5 departure step carries no "
            "bias-only clause, where its above-0.5 step voids the result whenever the bias-only control "
            "fires in the same direction; that asymmetry is recorded here as a gap in the decision list as "
            "declared, and the decision list itself is not amended after the fact. The below-0.5 departure "
            "is therefore not separable from a session-level spatial decoding bias and must never be read "
            "as a per-trial neural result."
        )
    return header + (
        "This run does not reproduce the same-direction, larger-magnitude configuration that motivates the "
        "gap statement above (A and C both significantly below 0.5, C the larger departure); the gap does "
        "not apply as written here, and the two statistics' relationship should be read directly from the "
        "numbers above."
    )


def _block_b_label_validity_gap_statement(pooled_a: dict) -> str:
    """Whether Block B's component-ordering vocabulary ('carried by', 'carried inside', 'carried outside')
    may be read as declared -- that vocabulary presupposes a primary Block A statistic significantly ABOVE
    0.5, checked live against the pooled primary rather than assumed."""
    a_below = pooled_a.get("status") == "tested" and pooled_a.get("significant_below_half")
    if a_below:
        return (
            "This block's decision-rule vocabulary -- 'carried by', 'carried inside', 'carried outside' -- "
            "presupposes a primary Block A statistic significantly ABOVE 0.5, i.e. that the pre-cue state "
            "leans toward the item that will be reported. The delivered primary is instead significantly "
            "BELOW 0.5. With a below-0.5 primary, no component of the pre-cue state may be described as "
            "carrying reported-item identity, regardless of which component's own bias-only control fires "
            "below; the numbers in this block are reported as measurements only, not as evidence that a "
            "component carries reported-item identity."
        )
    return (
        "The primary Block A statistic is not significantly below 0.5 in this run, so the label-validity "
        "gap that applies to a below-0.5 primary does not apply here; the component labels in this block "
        "may be read as declared in the decision rule."
    )


# ----------------------------------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------------------------------

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
    _COMPLETED_FITS[key] = {"complete": True, "value": checkpoint_safe(value)}
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_COMPLETED_FITS, allow_nan=False))
    os.replace(scratch, CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))
    os.replace(scratch, OUTPUT_PATH)


# ----------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "corpus": "Multi-object spatial working-memory corpus, macaque prefrontal cortex, continuous "
                  "saccadic report, three levels of item count, fixed 1.0 s maintenance delay cut per "
                  "trial from its own delay onset and admitted only if it ends at or before cue onset.",
        "question": (
            "Does the pre-cue population state carry not just that a binding failure is coming, but WHICH "
            "of the uncued objects the eventual report will land on -- tested only where both facts named "
            "in this module's own docstring hold: the analysed window is entirely pre-cue, and a swap "
            "trial at item count 3 leaves exactly two uncued candidates with an exact 0.5 null."
        ),
        "decision_rule_block_a_declared_before_fitting": DECISION_RULE_BLOCK_A_DECLARED_BEFORE_FITTING,
        "decision_rule_block_b_declared_before_fitting": DECISION_RULE_BLOCK_B_DECLARED_BEFORE_FITTING,
        "decision_rule_swap_destination_task_geometry_declared_before_fitting":
            DECISION_RULE_SWAP_DESTINATION_TASK_GEOMETRY_DECLARED_BEFORE_FITTING,
        "constants": {
            "n_folds": N_FOLDS, "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "min_train_trials_per_session": MIN_TRAIN_TRIALS_PER_SESSION,
            "min_test_trials_per_session": MIN_TEST_TRIALS_PER_SESSION,
            "min_pooled_test_trials": MIN_POOLED_TEST_TRIALS,
            "angular_equidistance_tolerance_rad": ANGULAR_EQUIDISTANCE_TOLERANCE_RAD,
            "n_shuffle_draws": N_SHUFFLE_DRAWS, "powered_null_mdd_ceiling": POWERED_NULL_MDD_CEILING,
        },
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass)")
    loaded: list[dict] = []
    refused: list[dict] = []
    n_seen = 0
    for session in iter_watters(root, bin_ms=BIN_MS):
        n_seen += 1
        if session["status"] != "loaded":
            refused.append({"session": session["session"], "animal": session.get("animal"),
                             "session_date": session.get("session_date"), "status": session["status"]})
            continue
        loaded.append(session)
    _log(f"corpus loaded: {n_seen} seen, {len(loaded)} loaded, {len(refused)} refused, "
         f"elapsed={time.time() - t0:.0f}s")

    behaviour = watters_behaviour(root)

    _log("computing Block 0 timing premise")
    block0 = block0_timing_premise(loaded, behaviour)
    output["block0_timing_premise"] = block0
    _flush(output)
    if block0["stop_condition_triggered"]:
        output["status"] = "stopped"
        output["branch"] = {"branch": BRANCH_BLOCK0_STOP}
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: an analysed trial's window overlaps its own cue onset")
        print(json.dumps({"status": "stopped", "branch": BRANCH_BLOCK0_STOP}, indent=2))
        return

    _log("running reproduction gate")
    gate_result = _fit("reproduction_gate", lambda: reproduction_gate(loaded))
    output["reproduction_gate"] = gate_result
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")

    if gate_result["status"] != "reproduced_exactly":
        output["block_a"] = {"status": "not_computed", "reason": "reproduction gate did not reproduce"}
        output["block_b"] = {"status": "not_computed", "reason": "reproduction gate did not reproduce"}
        output["branch_block_a"] = {"branch": BRANCH_REPRODUCTION_GATE_FAILED}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce")
        print(json.dumps({"reproduction_gate": gate_result["status"], "branch": BRANCH_REPRODUCTION_GATE_FAILED},
                          indent=2))
        return

    rows: list[dict] = []
    for session in loaded:
        key = session["session"]
        # Cache key carries its own schema tag: analyse_session's return value grew new fields (the
        # task-geometry control and the two Block B components' own bias-only controls), so a completed
        # fit recorded under the earlier "session|" key would silently omit them if reused here.
        row = _fit(f"session_with_task_geometry_and_component_bias|{key}",
                   lambda s=session: analyse_session(s, behaviour,
                                                       "pre_cue_state_and_reported_item_identity"))
        rows.append(row)
        output.setdefault("_progress", {})["sessions_done"] = len(rows)
        _flush(output)
        _log(f"  {key} status={row.get('status')} elapsed={time.time() - t0:.0f}s")
    output.pop("_progress", None)

    computed_rows = [r for r in rows if r.get("status") == "computed"]
    gate_cleared_rows = [r for r in rows if r.get("status") not in (
        "too_few_usable_trials", "landed_index_identity_check_failed", "too_few_train_or_test_trials",
        "no_usable_discretised_label_for_reachability_gate", "content_reachability_not_computable",
        "content_not_reachable_void")]

    total_pooled_test_trials = sum(r["block_a"]["n_test_trials"] for r in computed_rows)

    a_values = np.array([r["block_a"]["primary_fraction"] for r in computed_rows], dtype=float)
    c_values = np.array([r["block_a"]["bias_only_control_fraction"] for r in computed_rows], dtype=float)
    pooled_a = _pool_against_half(a_values, "pre_cue_state_and_reported_item_identity|block_a|pooled_a")
    pooled_c = _pool_against_half(c_values, "pre_cue_state_and_reported_item_identity|block_a|pooled_c")

    block_a_decision = decide_block_a(gate_result["status"], total_pooled_test_trials, pooled_a, pooled_c,
                                        a_values, c_values)

    equidistant_rows = [r for r in computed_rows
                         if r["block_a"]["equidistance_sensitivity"].get("fraction") is not None]
    equidistant_values = np.array([r["block_a"]["equidistance_sensitivity"]["fraction"] for r in equidistant_rows],
                                   dtype=float)
    equidistance_pooled = _pool_against_half(
        equidistant_values, "pre_cue_state_and_reported_item_identity|block_a|equidistance_sensitivity")

    # The task-geometry control: is the bias-only control C's own below-0.5 departure reproduced by a
    # quantity carrying no neural signal (the session's mean CUED object position), or does it need the
    # decoder's own mean decoded position?
    task_geometry_values = np.array([r["block_a"]["task_geometry_control_fraction"] for r in computed_rows],
                                     dtype=float)
    pooled_task_geometry = _pool_against_half(
        task_geometry_values, "pre_cue_state_and_reported_item_identity|block_a|pooled_task_geometry_control")
    paired_c_vs_task_geometry = _paired_session_test_full(
        c_values, task_geometry_values,
        "pre_cue_state_and_reported_item_identity|block_a|paired_c_vs_task_geometry")
    swap_destination_task_geometry_control = decide_swap_destination_task_geometry(
        pooled_c, pooled_task_geometry, paired_c_vs_task_geometry)

    output["reachability"] = {
        "n_behavioural_session_dates_seen": n_seen, "n_sessions_loaded": len(loaded),
        "n_sessions_refused_by_the_shared_loader": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                               for reason in sorted({r["status"] for r in refused})},
        "counts_reconcile": bool(n_seen == len(loaded) + len(refused)),
        "reconciles_against_the_delivered_watters_state_geometry_artifact": (
            "results/watters_state_geometry.json reports 47 dates seen, 41 analysed, 6 refused; this "
            "script reuses the identical iter_watters(root, bin_ms=100.0) call and must match those "
            "counts exactly."
        ),
        "n_sessions_gate_cleared_or_beyond": len(gate_cleared_rows),
        "n_sessions_computed": len(computed_rows),
        "status_counts": {status: sum(1 for r in rows if r.get("status") == status)
                          for status in sorted({r.get("status") for r in rows})},
        "n_loaded_but_not_computed": len(loaded) - len(computed_rows),
        "total_pooled_item_count_3_swap_test_trials": total_pooled_test_trials,
        "min_pooled_test_trials_floor": MIN_POOLED_TEST_TRIALS,
    }

    output["block_a"] = {
        "pooled_a_primary": pooled_a, "pooled_c_bias_only_control": pooled_c,
        "equidistance_sensitivity_pooled": equidistance_pooled,
        "n_sessions_with_equidistance_sensitivity_computed": len(equidistant_rows),
        "per_session_shuffled_report_null": [
            {"session": r["session"],
             "observed_fraction": r["block_a"]["primary_fraction"],
             "shuffled_report_null": r["block_a"]["shuffled_report_null"]}
            for r in computed_rows
        ],
        "swap_destination_task_geometry_control": swap_destination_task_geometry_control,
    }
    output["branch_block_a"] = block_a_decision
    output["primary_below_half_bias_only_control_gap"] = _primary_below_half_bias_only_control_gap_statement(
        pooled_a, pooled_c)

    if block_a_decision["branch"] == BRANCH_A_TOO_FEW:
        output["block_b"] = {"status": "not_run", "reason": BRANCH_B_NOT_RUN}
    else:
        outside_rows = [r for r in computed_rows if r["block_b"].get("outside_subspace", {}).get("fraction")
                         is not None]
        inside_rows = [r for r in computed_rows if r["block_b"].get("inside_subspace", {}).get("fraction")
                        is not None]
        outside_values = np.array([r["block_b"]["outside_subspace"]["fraction"] for r in outside_rows],
                                   dtype=float)
        inside_values = np.array([r["block_b"]["inside_subspace"]["fraction"] for r in inside_rows],
                                  dtype=float)
        pooled_outside = _pool_against_half(
            outside_values, "pre_cue_state_and_reported_item_identity|block_b|pooled_outside")
        pooled_inside = _pool_against_half(
            inside_values, "pre_cue_state_and_reported_item_identity|block_b|pooled_inside")

        paired_sessions = [r["session"] for r in computed_rows
                            if r["block_b"].get("outside_subspace", {}).get("fraction") is not None
                            and r["block_b"].get("inside_subspace", {}).get("fraction") is not None]
        paired_outside = np.array([r["block_b"]["outside_subspace"]["fraction"] for r in computed_rows
                                    if r["session"] in paired_sessions], dtype=float)
        paired_inside = np.array([r["block_b"]["inside_subspace"]["fraction"] for r in computed_rows
                                   if r["session"] in paired_sessions], dtype=float)

        block_b_decision = decide_block_b(pooled_outside, pooled_inside, paired_outside, paired_inside)
        subspace_identity_all_passed = bool(computed_rows) and all(
            r["block_b_subspace_decomposition_identity_check"]["passed"] for r in computed_rows)

        # Each component's own bias-only control, by the identical recipe used for C in Block A -- the
        # decoder's own mean prediction over its training trials, applied to every held-out test trial.
        outside_bias_values = np.array(
            [r["block_b"]["outside_subspace"]["bias_only_control_fraction"] for r in outside_rows], dtype=float)
        inside_bias_values = np.array(
            [r["block_b"]["inside_subspace"]["bias_only_control_fraction"] for r in inside_rows], dtype=float)
        pooled_outside_bias = _pool_against_half(
            outside_bias_values,
            "pre_cue_state_and_reported_item_identity|block_b|pooled_outside_bias_only_control")
        pooled_inside_bias = _pool_against_half(
            inside_bias_values,
            "pre_cue_state_and_reported_item_identity|block_b|pooled_inside_bias_only_control")
        paired_outside_vs_its_bias = _paired_session_test_full(
            outside_values, outside_bias_values,
            "pre_cue_state_and_reported_item_identity|block_b|paired_outside_vs_its_bias")
        paired_inside_vs_its_bias = _paired_session_test_full(
            inside_values, inside_bias_values,
            "pre_cue_state_and_reported_item_identity|block_b|paired_inside_vs_its_bias")

        output["block_b"] = {
            "status": "computed", "decision": block_b_decision,
            "n_sessions_outside_component_computed": len(outside_rows),
            "n_sessions_inside_component_computed": len(inside_rows),
            "n_sessions_paired": len(paired_sessions),
            "subspace_decomposition_identity_all_sessions_passed": subspace_identity_all_passed,
            "outside_subspace_bias_only_control": pooled_outside_bias,
            "inside_subspace_bias_only_control": pooled_inside_bias,
            "paired_test_outside_subspace_vs_its_bias_only_control": paired_outside_vs_its_bias,
            "paired_test_inside_subspace_vs_its_bias_only_control": paired_inside_vs_its_bias,
            "label_validity_gap_statement": _block_b_label_validity_gap_statement(pooled_a),
        }

    output["sessions"] = rows

    output["zero_drop_accounting"] = {
        "n_seen": n_seen, "n_loaded": len(loaded), "n_refused": len(refused),
        "n_analysed_computed": len(computed_rows),
        "n_loaded_but_not_computed": len(loaded) - len(computed_rows),
        "n_loaded_but_not_computed_by_status": {
            status: sum(1 for r in rows if r.get("status") == status)
            for status in sorted({r.get("status") for r in rows if r.get("status") != "computed"})
        },
        "reconciles": bool(n_seen == len(loaded) + len(refused)),
    }

    output["scope"] = {
        "unit_quality_tier": PRIMARY_QUALITY_TIER, "n_sessions_seen": n_seen,
        "n_sessions_analysed": len(computed_rows),
        "min_trials_for_behavioural_correlation": MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
        "correct_report_distance_threshold": CORRECT_REPORT_DISTANCE_THRESHOLD,
        "content_reachability_k_classes": WATTERS_RECOVERABILITY_K_CLASSES,
        "subspace_dimension": WATTERS_REGRESSION_DIM,
        "seed_scheme": "every fold assignment, shuffle draw and pooled test is seeded by a stable hash "
                       "(stable_seed) of a descriptive tag; deterministic and reproducible from the tag alone",
        "git_commit": git_commit(ROOT), "wall_clock_s": time.time() - t0,
    }

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"branch_block_a: {block_a_decision.get('branch')} elapsed={time.time() - t0:.0f}s")
    print(json.dumps({
        "reproduction_gate": gate_result["status"], "branch_block_a": block_a_decision.get("branch"),
        "branch_block_b": output["block_b"].get("decision", {}).get("branch"),
        "branch_swap_destination_task_geometry_control": swap_destination_task_geometry_control.get("branch"),
        "total_pooled_test_trials": total_pooled_test_trials, "n_sessions_analysed": len(computed_rows),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
