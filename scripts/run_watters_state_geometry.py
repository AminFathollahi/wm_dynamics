"""run_watters_state_geometry.py -- the cross-unit state in a second macaque
corpus: a multi-object spatial working-memory task with a fixed 1.0 s
maintenance delay, two animals, and a continuous saccadic report.

Four questions, in the order they are allowed to be asked:

1. Existence. Does the leading-latent persistence estimator return a signal
   above its own per-unit permutation null here at all, at the same window
   width and bin size the other corpora are measured at? Nothing downstream
   runs if it does not.
2. Label cardinality. The memorandum is a screen position, so the number of
   classes a content decoder is asked to separate is a free parameter, not a
   property of the task. On single-object trials the cued angle is re-binned
   into 2, 3, 4, 6 and 8 classes and the leading latent's ablation-cost rank
   among the variance-ordered latents is recorded at each rung.
3. Behaviour, continuously. The animal saccades to a remembered location, so
   every completed trial carries a graded report deviation -- the distance
   between where it looked and where the cued object was. That deviation,
   before any correctness threshold, is the primary behavioural quantity
   here. A binary correct-versus-error arm is run beside it at the same
   0.2-normalized-unit threshold, so the graded and thresholded designs can
   be compared on one estimator in one corpus.
4. Item count. 1, 2 or 3 objects are held per trial, so state amplitude and
   state persistence can be asked to vary with load.

Everything reuses the estimators the other corpora are measured with rather
than reimplementing them: state_persistence.r_lag_profile and
per_unit_permutation_null_r_lag_profile for the state,
run_state_content_link.session_subtractive_test for the content rank,
run_rate_free_state_geometry_behavior_link.rate_free_state_deviation and
statistics.partial_correlation_permutation_test for the behavioural link,
and run_state_behavior_link.trial_amplitude_covariates for state amplitude.

The corpus is a preprint deposit: "Working Memory of Multi-Object Scenes in
Primate Frontal Cortex" (Watters, Gabel, Tenenbaum, Jazayeri; bioRxiv
preprint, posted 2026-01-27, DOI 10.64898/2026.01.27.702062; data DANDI
000620). No peer-reviewed journal version was found, so it is cited as an
unreviewed preprint.

The script is resumable: every session's row is flushed to the output as it
completes, and a rerun skips the sessions already present at the same
analysis version.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import (  # noqa: E402
    WATTERS_DELAY_WINDOW_S,
    WATTERS_MIN_UNITS,
    WATTERS_TASK_VARIANTS,
    data_root,
    iter_watters,
    watters_behaviour,
    watters_directories,
)
from provenance import _json_safe, git_commit  # noqa: E402
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, _classify, rate_free_state_deviation,
)
from run_state_behavior_link import MIN_ERROR_TRIALS_FOR_REACHABILITY, trial_amplitude_covariates  # noqa: E402
from run_state_content_link import (  # noqa: E402
    BIN_MS, DECIDING_WIDTH_BINS, PANICHELLO_DELAY_WINDOW_MS, session_subtractive_test, usable_label,
)
from run_watters_source_replication import CORRECT_REPORT_DISTANCE_THRESHOLD  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from state_persistence import (  # noqa: E402
    _lag_existence_and_rotation, _lag_pairs, _ols_slope, per_unit_permutation_null_r_lag_profile,
    r_lag_profile, slope_across_sessions_test,
)
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)

OUTPUT_PATH = ROOT / "results" / "watters_state_geometry.json"
ANALYSIS_VERSION = "2026-08-14"
N_PERM = 10000
BIN_WIDTH_S = BIN_MS / 1000.0

CARDINALITY_RUNGS = (2, 3, 4, 6, 8)

# Both quality tiers are cut from ONE loaded tensor, so the trial set, the delay
# window and the behavioural join are identical between them and the only thing
# that changes is which units are in the population.
PRIMARY_QUALITY_TIER = "single_and_multi_unit"
QUALITY_TIER_CHOICE = (
    "The primary population pools well-isolated single units and multi-unit clusters "
    "('single_and_multi_unit'). The corpus's shared per-trial cache labels 5228 of its 8085 "
    "unit-session-probe files as multi-unit against 2857 well-isolated, so a single-unit-only "
    "population discards most of what was recorded and, in this corpus, drops sessions below the "
    f"{WATTERS_MIN_UNITS}-unit floor outright. A 'good_single_units_only' arm is run beside it on "
    "the same trials and the same delay window, and both are reported for every headline test; "
    "neither is left implicit."
)

# The matched-unit-count arm exists so a difference BETWEEN animals or BETWEEN task
# variants is not read off populations of different size. Fixed at the same floor every
# included session clears by construction, declared here before any fit, so the matched
# arm drops no session and needs no post-hoc choice of a matching level.
MATCHED_UNIT_COUNT = WATTERS_MIN_UNITS

MIN_TRIALS_PER_LOAD_LEVEL = 60
MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION = 16

EXISTENCE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, fit the leading latent on half the trials, project the held-out half, and take the "
    "across-trial correlation between two non-overlapping window means at every reachable lag, at the "
    f"same {DECIDING_WIDTH_BINS}-bin window width and {BIN_MS:.0f} ms bins the other corpora are measured "
    "at (state_persistence.r_lag_profile). The reference is the per-unit trial-label permutation null "
    "through the identical estimator (per_unit_permutation_null_r_lag_profile), which destroys any "
    "correlation shared across units while leaving each unit's own trial-count distribution and time "
    "course intact. Pool the per-session observed and null values at each lag with the paired sign-flip "
    "test, one-sided greater, and correct across the lag axis by Benjamini-Hochberg. If at least one lag "
    "is significant after correction the branch is 'cross_unit_state_present'. If no lag is significant "
    "and the minimum detectable paired difference on the mean observed-minus-null difference is below "
    f"{MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r units, the branch is 'cross_unit_state_absent'. Otherwise "
    "the branch is 'inconclusive_below_detection_floor'. The effect size reported with the verdict is the "
    "mean observed-minus-null difference over the reachable lags and the observed and null values it is "
    "the difference of."
)

CARDINALITY_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "On single-object trials only, so that the number of classes is never confounded with the number of "
    "objects held, re-bin the cued polar angle into K equal-width classes for K in "
    f"{list(CARDINALITY_RUNGS)} and run the leave-one-latent-out ablation-cost ranking "
    "(run_state_content_link.session_subtractive_test) unchanged: the same k = min(8, n_units - 2, "
    "max(2, n_trials // 8)) latents, the same held-out decoder, the same macro one-versus-rest AUC and the "
    "same label-permutation null. The statistic is the leading latent's fractional rank among the k "
    "latents by ablation cost, on [0, 1], where 0 means the leading latent is the single most costly "
    "latent to ablate and 1 means it is the least. A rung whose K exceeds the number of distinct cued "
    "positions the task puts on the screen is unreachable by construction and is recorded as such rather "
    "than fitted. Per session, take the ordinary-least-squares slope of fractional rank on K across the "
    "rungs that fitted, and pool the per-session slopes across sessions with the two-sided paired "
    "sign-flip test. If the pooled slope is significant the branch is "
    "'content_rank_tracks_label_cardinality'. If it is not significant and its minimum detectable paired "
    "difference is below 0.0417 fractional-rank units per class (a quarter of the fractional rank's own "
    "[0, 1] range spread over the six-class span of the ladder), the branch is "
    "'content_rank_independent_of_label_cardinality'. Otherwise it is "
    "'inconclusive_below_detection_floor'. The per-rung mean fractional rank and the per-rung decoding "
    "AUC are reported beside the verdict either way."
)

# A quarter of the fractional rank's [0, 1] range spread over the ladder's K = 2 to K = 8 span.
CARDINALITY_MEANINGFUL_SLOPE_PER_CLASS = 0.25 / (max(CARDINALITY_RUNGS) - min(CARDINALITY_RUNGS))

BEHAVIOUR_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "The behavioural observable is the trial's rate-free state deviation: each trial's per-unit total "
    "spike count over the delay epoch, L2-normalised to unit length so only the direction across units "
    "survives, compared by cosine against the leave-one-out mean direction of every other trial in the "
    "session (run_rate_free_state_geometry_behavior_link.rate_free_state_deviation, unchanged).\n"
    "PRIMARY, CONTINUOUS. The report is a saccade to a remembered position, so every completed trial "
    "yields a graded report deviation -- the Euclidean distance between the reported and the cued "
    "position in normalized screen units -- and there is no accuracy-driven exclusion and no error floor: "
    "every session with a fitted tensor is eligible. Per session, correlate report deviation against "
    "state deviation over every completed trial (the zero-controls case of "
    "statistics.partial_correlation_permutation_test, an ordinary Pearson correlation with a permutation "
    "p-value), and pool the per-session coefficients across sessions with the two-sided paired sign-flip "
    "test. EXPECTED SIGN, declared before fitting: POSITIVE -- a trial whose state sits further from the "
    "session's own mean state direction should be reported less accurately, if this corpus agrees with "
    "the corpus where the same observable was first measured.\n"
    "ORTHOGONALITY GATE, run before the behavioural correlation is interpreted: the state deviation is "
    "correlated with the trial's total spike count. If that pooled correlation is significant the "
    "construction failed and the branch is "
    "'rate_free_state_observable_is_not_rate_free_and_this_analysis_is_void'. Report deviation may itself "
    "track reaction time, total spike count, time in session and the number of objects held, so each is "
    "partialled out on its own and all four together, and every partial is reported beside the raw "
    "coefficient rather than only after the raw one is seen.\n"
    "COMPARABILITY ARM, BINARY. The identical design is run with the correct-versus-error label obtained "
    f"by thresholding the same report deviation at {CORRECT_REPORT_DISTANCE_THRESHOLD} normalized units, "
    f"under the same pre-declared floor of at least {MIN_ERROR_TRIALS_FOR_REACHABILITY} error trials per "
    "session. How many sessions clear that floor is itself reported: a binary design draws its power from "
    "task failure, and a corpus where the graded design has every session but the thresholded design has "
    "few is evidence about the design rather than about the biology.\n"
    "BRANCHES. If the gate passes and the pooled raw correlation is significant with the jointly "
    "controlled partial significant in the same direction, the branch is "
    "'rate_free_state_geometry_predicts_accuracy'. If the raw correlation is not significant and its "
    f"minimum detectable paired difference is below {MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r units, the "
    "branch is 'state_geometry_does_not_predict_accuracy'. If the raw correlation is significant but does "
    "not survive the joint control, that is its own outcome and is named as such rather than forced onto "
    "either. Otherwise the branch is 'inconclusive_below_detection_floor'.\n"
    "AGREEMENT. The graded and thresholded arms are compared on effect size first. Because the graded arm "
    "correlates state deviation with report ERROR and the thresholded arm correlates it with CORRECTNESS, "
    "the two agree when their pooled coefficients have OPPOSITE signs. If at least one arm is significant "
    "and the signs stand in that relation the branch is "
    "'continuous_and_binary_behavioural_arms_agree'; if at least one is significant and they do not, the "
    "branch is 'continuous_and_binary_behavioural_arms_disagree', which would be a result about the "
    "design rather than about the biology; if neither is significant it is "
    "'inconclusive_below_detection_floor'."
)

LOAD_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "The number of objects held is 1, 2 or 3 and is ordinal, so both load tests are MONOTONE TREND tests "
    "over that ordinal axis, not pairwise contrasts between levels. EXPECTED DIRECTIONS, declared before "
    "fitting: state amplitude INCREASES with the number of objects held (more items engage more of the "
    "population), and state persistence DECREASES with it (more items interfere with each other over the "
    "delay).\n"
    "AMPLITUDE. Per trial, the leading-component score of the centred trial-by-window matrix "
    "(run_state_behavior_link.trial_amplitude_covariates, the same per-trial amplitude this project's "
    "amplitude analyses use). Per session, correlate it against the number of objects, raw and "
    "controlling total spike count, and pool across sessions with the two-sided paired sign-flip test. "
    "The spike-count control is not decoration: a per-trial amplitude that tracks item count only because "
    "more items mean more spikes is a rate effect, and is reported as one.\n"
    "PERSISTENCE. Within a session, split trials by the number of objects, subsample every level to the "
    "smallest level's trial count so the estimator sees the same number of trials at each level, and "
    "recompute the observed-minus-null difference averaged over the reachable lags at each level. Take "
    "the per-session slope of that difference on the number of objects and pool across sessions the same "
    "way. Levels with fewer than "
    f"{MIN_TRIALS_PER_LOAD_LEVEL} trials are not fitted.\n"
    "BRANCHES, on amplitude. Significant pooled trend: 'state_amplitude_tracks_item_count', with the "
    "sign stated and with whether it survives the spike-count control stated in the same object. Not "
    "significant with a minimum detectable paired difference below "
    f"{MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r units: 'state_amplitude_independent_of_item_count'. "
    "Otherwise 'inconclusive_below_detection_floor'.\n"
    "The two task variants differ in their maximum item count (2 and 3) and never share a session date, "
    "so both load tests are reported per variant as well as pooled."
)

MATCHED_INSTRUMENT_RULE_DECLARED_BEFORE_FITTING = (
    "Unit count varies by an order of magnitude across sessions in this corpus, so no difference between "
    "animals or between task variants is read off populations of different size. Before any between-group "
    "comparison, the per-session unit counts of the two groups are themselves compared by a permutation "
    f"test on their difference in means. Every session is additionally re-analysed at exactly "
    f"{MATCHED_UNIT_COUNT} units drawn without replacement from its own population under a stable "
    "per-session seed -- the floor every included session clears by construction, so the matched arm "
    "drops no session -- and any between-group difference is reported from that matched arm. The "
    "native-unit-count numbers remain the headline within each group; the matched arm is what licenses "
    "comparing them."
)


def reachable_lag_range(n_bins: int, width_bins: int) -> dict:
    """The lags, in bins and in seconds, at which the fixed-width persistence
    estimator has at least one window pair wholly inside an epoch of
    ``n_bins`` bins. Both windows must fit and must not overlap, so the range
    is [width, n_bins - width] and an epoch shorter than twice the window
    width reaches no lag at all."""
    lags = [lag for lag in range(width_bins, n_bins + 1) if _lag_pairs(n_bins, width_bins, lag)]
    return {
        "n_bins": int(n_bins), "width_bins": int(width_bins), "bin_width_s": BIN_WIDTH_S,
        "reachable_lags_bins": lags,
        "reachable_lags_s": [round(lag * BIN_WIDTH_S, 6) for lag in lags],
        "n_reachable_lags": len(lags),
        "shortest_reachable_lag_s": round(min(lags) * BIN_WIDTH_S, 6) if lags else None,
        "longest_reachable_lag_s": round(max(lags) * BIN_WIDTH_S, 6) if lags else None,
    }


def reachability_block() -> dict:
    """The lag range this corpus's 1.0 s delay reaches, next to the range the
    longer-delay macaque corpus this arm replicates reaches, computed from
    each corpus's own epoch length through the same formula -- written into
    the artifact BEFORE any test runs, because a lag one corpus cannot reach
    is not a lag the two can be compared at."""
    here = reachable_lag_range(int(round(WATTERS_DELAY_WINDOW_S * 1000.0 / BIN_MS)), DECIDING_WIDTH_BINS)
    other_window_s = (PANICHELLO_DELAY_WINDOW_MS[1] - PANICHELLO_DELAY_WINDOW_MS[0]) / 1000.0
    there = reachable_lag_range(int(len(np.arange(*PANICHELLO_DELAY_WINDOW_MS, BIN_MS))), DECIDING_WIDTH_BINS)
    common = sorted(set(here["reachable_lags_bins"]) & set(there["reachable_lags_bins"]))
    return {
        "this_corpus": {"delay_epoch_s": WATTERS_DELAY_WINDOW_S, **here},
        "longer_delay_macaque_corpus_the_estimator_is_shared_with": {
            "delay_epoch_analysed_s": other_window_s, **there},
        "lags_reachable_in_both_bins": common,
        "lags_reachable_in_both_s": [round(lag * BIN_WIDTH_S, 6) for lag in common],
        "lags_reachable_only_in_the_longer_epoch_bins": sorted(
            set(there["reachable_lags_bins"]) - set(here["reachable_lags_bins"])),
        "existence_is_reachable": len(here["reachable_lags_bins"]) >= 1,
        "shape_of_the_lag_profile_is_reachable": len(here["reachable_lags_bins"]) >= 3,
        "verdict": (
            f"A {WATTERS_DELAY_WINDOW_S:.1f} s delay at {BIN_MS:.0f} ms bins is "
            f"{here['n_bins']} bins, and a {DECIDING_WIDTH_BINS}-bin window pair fits inside it at "
            f"{here['n_reachable_lags']} lags, {here['shortest_reachable_lag_s']:.1f} s to "
            f"{here['longest_reachable_lag_s']:.1f} s. Existence is therefore reachable and is the test "
            "this corpus is used for. Every lag it reaches is also reached by the longer-delay corpus, so "
            "the two are comparable on existence over the whole of this corpus's range; the longer corpus "
            "additionally reaches "
            f"{[round(lag * BIN_WIDTH_S, 1) for lag in sorted(set(there['reachable_lags_bins']) - set(here['reachable_lags_bins']))]} s, "
            "and no statement here is made at those lags. A decay SHAPE fitted over "
            f"{here['n_reachable_lags']} points spanning "
            f"{here['longest_reachable_lag_s'] - here['shortest_reachable_lag_s']:.1f} s is too short a "
            "baseline to separate a decaying profile from a flat one against this estimator's own "
            "lag-dependent pair count, so the shape of the lag profile is NOT claimed here and no "
            "cross-corpus shape comparison is made."
        ),
    }


def cardinality_reachability_block(root: Path) -> dict:
    """The distinct cued positions each task variant's single-object trials
    actually display, established directly from the corpus's own behaviour
    tables BEFORE any rung of the cardinality ladder is fit or interpreted.
    The ladder asks a decoder to separate K = 2..8 classes of cued position;
    a variant whose task design places the object at fewer than K positions
    in the first place cannot reach that rung no matter how the estimator
    performs, and a null reported there would be unreachable by construction
    rather than a finding. Computed corpus-wide, over every completed
    single-object trial of the variant, not only the sessions with a fitted
    spike tensor -- this is a statement about the task design, not about
    which sessions were staged."""
    behaviour = watters_behaviour(root)
    single_object = behaviour.loc[behaviour["num_objects"].to_numpy(int) == 1]
    variants: dict[str, dict] = {}
    for variant in WATTERS_TASK_VARIANTS:
        theta = single_object.loc[single_object["task"] == variant, "cued_theta"].to_numpy(dtype=float)
        n_distinct = int(len(np.unique(np.round(theta, 6))))
        rungs = []
        for k in CARDINALITY_RUNGS:
            labels = (np.floor(theta / (2.0 * np.pi / k)).astype(int)) % k
            populated = int(len(np.unique(labels))) if len(labels) else 0
            rungs.append({"k": k, "n_populated_classes": populated, "reachable": populated >= k})
        variants[variant] = {
            "n_single_object_trials_corpus_wide": int(len(theta)),
            "n_distinct_cued_positions_corpus_wide": n_distinct,
            "rungs": rungs,
            "reachable_rungs": [r["k"] for r in rungs if r["reachable"]],
            "unreachable_rungs": [r["k"] for r in rungs if not r["reachable"]],
        }
    parts = []
    for variant, info in variants.items():
        if info["unreachable_rungs"]:
            parts.append(
                f"'{variant}' displays only {info['n_distinct_cued_positions_corpus_wide']} distinct cued "
                f"position(s) over its {info['n_single_object_trials_corpus_wide']} single-object trials, so "
                f"rung(s) K={info['unreachable_rungs']} are UNREACHABLE by construction and are never fitted "
                f"for this variant; only K={info['reachable_rungs']} are tested.")
        else:
            parts.append(
                f"'{variant}' displays {info['n_distinct_cued_positions_corpus_wide']} distinct cued positions "
                f"over its {info['n_single_object_trials_corpus_wide']} single-object trials, so every rung "
                f"K={list(CARDINALITY_RUNGS)} is reachable.")
    parts.append(
        "Because the two variants do not reach the same rungs, a 'pooled' fractional-rank slope averages "
        "per-session slopes fitted over different K spans; the per-task-variant breakdown, not the pooled "
        "number, is what licenses a claim about the ladder's shape.")
    return {"variants": variants, "verdict": " ".join(parts)}


def _corr(outcome: np.ndarray, covariate: np.ndarray, controls: list[np.ndarray], seed_tag: str) -> dict:
    rng = np.random.default_rng(stable_seed(seed_tag))
    if len(outcome) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {"status": "not_computable", "reason": "too few trials", "n": int(len(outcome))}
    return partial_correlation_permutation_test(outcome, covariate, controls=controls, n_perm=N_PERM, rng=rng)


def state_profile(counts: np.ndarray, seed_tag: str) -> dict:
    """The observed lag profile, its per-unit permutation null and their
    difference, at the shared window width -- one session, one population."""
    profile = r_lag_profile(counts, DECIDING_WIDTH_BINS, rng=np.random.default_rng(stable_seed(seed_tag)))
    if profile["status"] != "fitted":
        return {"status": profile["status"], "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1])}
    null = per_unit_permutation_null_r_lag_profile(
        counts, DECIDING_WIDTH_BINS, rng=np.random.default_rng(stable_seed(seed_tag + "|null")))
    lags = sorted(set(profile["lags"]) & set(null["lags"]))
    if not lags:
        return {"status": "no_lag_reached_by_both_the_profile_and_its_null",
                "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1])}
    observed = {str(lag): float(profile["lags"][lag]["r_median"]) for lag in lags}
    null_values = {str(lag): float(null["lags"][lag]["r_null_median"]) for lag in lags}
    difference = {str(lag): observed[str(lag)] - null_values[str(lag)] for lag in lags}
    return {
        "status": "fitted", "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
        "lags_bins": lags, "r_observed": observed, "r_permutation_null": null_values,
        "r_observed_minus_null": difference,
        "mean_r_observed_minus_null": float(np.mean(list(difference.values()))),
        "mean_r_observed": float(np.mean(list(observed.values()))),
        "mean_r_permutation_null": float(np.mean(list(null_values.values()))),
        "n_null_replicates_fitted": int(null["n_replicates_fitted"]),
    }


def cardinality_ladder(counts: np.ndarray, cued_theta: np.ndarray, seed_tag: str) -> dict:
    """The ablation-cost rank of the leading latent at each label-cardinality
    rung, on the trials handed in (single-object trials only, so the class
    count is never confounded with the number of objects held)."""
    if counts.shape[0] < 8:
        return {"status": "too_few_single_object_trials", "n_trials": int(counts.shape[0])}
    window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]
    n_distinct_positions = int(len(np.unique(np.round(cued_theta, 6))))
    rungs: dict[str, dict] = {}
    for k_classes in CARDINALITY_RUNGS:
        labels = (np.floor(cued_theta / (2.0 * np.pi / k_classes)).astype(int)) % k_classes
        populated = int(len(np.unique(labels)))
        if populated < k_classes:
            rungs[str(k_classes)] = {
                "status": "label_cardinality_exceeds_the_distinct_cued_positions_the_task_displays",
                "n_populated_classes": populated, "n_distinct_cued_positions": n_distinct_positions}
            continue
        ok, reason, mask = usable_label(labels)
        if not ok:
            rungs[str(k_classes)] = {"status": "no_usable_label", "reason": reason,
                                     "n_populated_classes": populated}
            continue
        rungs[str(k_classes)] = session_subtractive_test(
            window_mean[mask], labels[mask], stable_seed(f"{seed_tag}|k{k_classes}"))
        rungs[str(k_classes)]["n_trials_after_class_filter"] = int(mask.sum())
    fitted = [(k, rungs[str(k)]) for k in CARDINALITY_RUNGS if rungs[str(k)].get("status") == "tested"]
    slope = _ols_slope([float(k) for k, _ in fitted],
                       [r["leading_latent_fractional_rank"] for _, r in fitted]) if len(fitted) >= 2 else None
    return {
        "status": "fitted" if fitted else "no_rung_fitted",
        "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
        "n_distinct_cued_positions": n_distinct_positions,
        "rungs": rungs,
        "n_rungs_fitted": len(fitted),
        "fractional_rank_by_rung": {str(k): r["leading_latent_fractional_rank"] for k, r in fitted},
        "decoding_auc_by_rung": {str(k): r["a_full"] for k, r in fitted},
        "decoding_auc_clears_own_null_by_rung": {str(k): bool(r.get("a_full_clears_own_null", False))
                                                  for k, r in fitted},
        "fractional_rank_slope_per_class": slope,
    }


def _behaviour_observables(counts: np.ndarray, session: dict) -> tuple[dict, dict, np.ndarray]:
    """The per-trial quantities every behavioural correlation in this script
    is built from, restricted to the trials on which the state direction,
    the saccadic report and the reaction time are all defined, together with
    the exclusion count by reason and the mask itself."""
    activity = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    report_error = np.asarray(session["report_deviation"], dtype=float)
    reaction_time = np.asarray(session["reaction_time_ms"], dtype=float)
    usable = np.isfinite(deviation) & np.isfinite(report_error) & np.isfinite(reaction_time)
    excluded = {
        "state_direction_undefined_zero_total_activity": int(np.sum(~np.isfinite(deviation))),
        "no_saccadic_report_recorded": int(np.sum(~np.isfinite(report_error))),
        "no_reaction_time_recorded": int(np.sum(~np.isfinite(reaction_time) & np.isfinite(report_error))),
    }
    observables = {
        "state_deviation": deviation[usable],
        "spike_count": activity.sum(axis=1).astype(float)[usable],
        "trial_index": np.arange(counts.shape[0], dtype=float)[usable],
        "report_error": report_error[usable],
        "reaction_time": reaction_time[usable],
        "item_count": np.asarray(session["num_objects"], dtype=float)[usable],
        "is_correct": np.asarray(session["correct"], dtype=bool)[usable],
    }
    return observables, excluded, usable


def item_count_component(counts: np.ndarray, session: dict, seed_tag: str) -> dict:
    """Item count's own correlations with the two quantities the graded
    behavioural coefficient is built from: the trial's state deviation and
    the trial's report error.

    These are what makes a pooled coefficient in a variable-load task
    decomposable. For Pearson correlations the identity

        r(report, state) = r(report, state | load) * sqrt((1 - r_ls^2)(1 - r_lr^2))
                           + r_ls * r_lr

    holds exactly, where r_ls and r_lr are the two correlations computed
    here, so the product term IS the between-load contribution to the pooled
    coefficient and can be measured rather than inferred from a sign
    difference between the pooled and the within-load estimate."""
    observables, _, usable = _behaviour_observables(counts, session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {"status": "too_few_trials_with_a_defined_state_direction_and_report",
                "n_trials_usable": int(usable.sum())}
    return {
        "status": "computed",
        "n_trials_usable": int(usable.sum()),
        "n_item_count_levels": int(len(np.unique(observables["item_count"]))),
        "item_count_vs_state_deviation": _corr(
            observables["state_deviation"], observables["item_count"], [], f"{seed_tag}|component|state"),
        "item_count_vs_report_error": _corr(
            observables["report_error"], observables["item_count"], [], f"{seed_tag}|component|report"),
    }


def behaviour_arm(counts: np.ndarray, session: dict, seed_tag: str) -> dict:
    """The graded-report primary and the thresholded comparability arm, on
    the same trials, the same observable and the same estimator."""
    observables, excluded, usable = _behaviour_observables(counts, session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {"status": "too_few_trials_with_a_defined_state_direction_and_report",
                "n_trials_usable": int(usable.sum()), "trials_excluded_by_reason": excluded}

    deviation, spike_count = observables["state_deviation"], observables["spike_count"]
    trial_index, report_error = observables["trial_index"], observables["report_error"]
    reaction_time, item_count = observables["reaction_time"], observables["item_count"]
    is_correct = observables["is_correct"]
    every_nuisance = [spike_count, trial_index, reaction_time, item_count]

    continuous = {
        "raw_report_error_vs_state_deviation": _corr(report_error, deviation, [], f"{seed_tag}|graded|raw"),
        "partial_controlling_spike_count": _corr(report_error, deviation, [spike_count], f"{seed_tag}|graded|spike"),
        "partial_controlling_trial_index": _corr(report_error, deviation, [trial_index], f"{seed_tag}|graded|trial"),
        "partial_controlling_reaction_time": _corr(report_error, deviation, [reaction_time], f"{seed_tag}|graded|rt"),
        "partial_controlling_item_count": _corr(report_error, deviation, [item_count], f"{seed_tag}|graded|load"),
        "joint_partial_controlling_spike_count_and_trial_index": _corr(
            report_error, deviation, [spike_count, trial_index], f"{seed_tag}|graded|joint"),
        "joint_partial_controlling_every_nuisance": _corr(
            report_error, deviation, every_nuisance, f"{seed_tag}|graded|all"),
    }

    n_error = int(np.sum(~is_correct))
    if n_error >= MIN_ERROR_TRIALS_FOR_REACHABILITY:
        outcome = is_correct.astype(float)
        binary = {
            "status": "computed", "n_error_trials": n_error,
            "raw_outcome_vs_state_deviation": _corr(outcome, deviation, [], f"{seed_tag}|binary|raw"),
            "partial_controlling_spike_count": _corr(outcome, deviation, [spike_count], f"{seed_tag}|binary|spike"),
            "partial_controlling_trial_index": _corr(outcome, deviation, [trial_index], f"{seed_tag}|binary|trial"),
            "joint_partial_controlling_spike_count_and_trial_index": _corr(
                outcome, deviation, [spike_count, trial_index], f"{seed_tag}|binary|joint"),
        }
    else:
        binary = {"status": "below_the_error_trial_floor", "n_error_trials": n_error,
                  "floor": MIN_ERROR_TRIALS_FOR_REACHABILITY}

    return {
        "status": "computed",
        "n_trials_usable": int(usable.sum()), "n_trials_seen": int(counts.shape[0]),
        "trials_excluded_by_reason": excluded,
        "n_error_trials_at_the_correctness_threshold": n_error,
        "median_report_deviation": float(np.median(report_error)),
        "orthogonality_gate_state_deviation_vs_spike_count": _corr(
            deviation, spike_count, [], f"{seed_tag}|gate"),
        "continuous_graded_report": continuous,
        "binary_correct_versus_error": binary,
        "item_count_component": item_count_component(counts, session, seed_tag),
    }


def load_arm(counts: np.ndarray, session: dict, seed_tag: str) -> dict:
    """Does state amplitude, and does state persistence, track the number of
    objects held? Amplitude is a within-session monotone trend over the
    ordinal item count; persistence is refitted per item count at a matched
    trial count so the estimator sees the same number of trials at each
    level."""
    item_count = np.asarray(session["num_objects"], dtype=float)
    levels = sorted(np.unique(item_count).tolist())
    if len(levels) < 2:
        return {"status": "only_one_item_count_in_this_session", "levels": levels}

    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return {"status": covariates["status"], "reason": covariates.get("reason")}
    amplitude = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    spike_count = counts.sum(axis=(1, 2)).astype(float)

    amplitude_arm = {
        "raw_amplitude_vs_item_count": _corr(amplitude, item_count, [], f"{seed_tag}|amp|raw"),
        "partial_controlling_spike_count": _corr(amplitude, item_count, [spike_count], f"{seed_tag}|amp|spike"),
        "spike_count_vs_item_count": _corr(spike_count, item_count, [], f"{seed_tag}|amp|rate"),
    }

    per_level_counts = {str(int(level)): int(np.sum(item_count == level)) for level in levels}
    matched_n = min(per_level_counts.values())
    persistence: dict[str, dict] = {}
    if matched_n >= MIN_TRIALS_PER_LOAD_LEVEL:
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|load_persistence"))
        for level in levels:
            index = np.flatnonzero(item_count == level)
            drawn = rng.choice(index, size=matched_n, replace=False)
            persistence[str(int(level))] = state_profile(
                counts[np.sort(drawn)], f"{seed_tag}|load{int(level)}")
        fitted = [(level, persistence[str(int(level))]) for level in levels
                  if persistence[str(int(level))].get("status") == "fitted"]
        slope = _ols_slope([float(level) for level, _ in fitted],
                           [r["mean_r_observed_minus_null"] for _, r in fitted]) if len(fitted) >= 2 else None
    else:
        slope = None

    return {
        "status": "computed", "item_counts_present": [int(level) for level in levels],
        "n_trials_per_item_count": per_level_counts,
        "matched_trials_per_item_count": int(matched_n),
        "amplitude": amplitude_arm,
        "persistence_by_item_count": persistence or {
            "status": "below_the_matched_trial_floor", "matched_n": int(matched_n),
            "floor": MIN_TRIALS_PER_LOAD_LEVEL},
        "persistence_slope_per_item": slope,
    }


def _matched_unit_subset(counts: np.ndarray, seed_tag: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|matched_units"))
    drawn = np.sort(rng.choice(counts.shape[1], size=MATCHED_UNIT_COUNT, replace=False))
    return counts[:, drawn]


def analyse_session(session: dict) -> dict:
    """Every arm, for one session, at both quality tiers plus the
    matched-unit-count arm the between-group comparisons are read from."""
    tag = f"watters_state_geometry|{session['session']}"
    quality = np.asarray(session["unit_quality"])
    single_object = np.asarray(session["num_objects"], dtype=int) == 1
    tiers = {PRIMARY_QUALITY_TIER: np.ones(len(quality), dtype=bool)}
    good = quality == "good"
    if int(good.sum()) >= WATTERS_MIN_UNITS:
        tiers["good_single_units_only"] = good
    else:
        tiers_note = {"status": "below_the_unit_floor", "n_good_units": int(good.sum()),
                      "floor": WATTERS_MIN_UNITS}

    row: dict = {
        "animal": session["animal"], "session_date": session["session_date"],
        "session": session["session"], "task_variant": session["task_variant"],
        "n_units_total": int(session["n_units"]), "n_units_good": int(good.sum()),
        "n_units_seen_in_cache": int(session["n_units_seen"]),
        "n_units_in_cache_not_spanning_every_trial": int(session["n_units_in_union_but_not_spanning"]),
        "n_trials": int(len(session["trial_num"])),
        "n_single_object_trials": int(single_object.sum()),
        "trials_dropped_by_reason": session["trials_dropped_by_reason"],
        "analysis_version": ANALYSIS_VERSION,
        "by_quality_tier": {},
    }
    if "good_single_units_only" not in tiers:
        row["good_single_units_only_note"] = tiers_note

    for tier, mask in tiers.items():
        counts = session["counts"][:, mask]
        row["by_quality_tier"][tier] = {
            "n_units": int(mask.sum()),
            "existence": state_profile(counts, f"{tag}|{tier}|existence"),
            "behaviour": behaviour_arm(counts, session, f"{tag}|{tier}|behaviour"),
        }
        if tier == PRIMARY_QUALITY_TIER:
            row["by_quality_tier"][tier]["cardinality_ladder"] = cardinality_ladder(
                counts[single_object], np.asarray(session["cued_theta"])[single_object],
                f"{tag}|{tier}|cardinality")
            row["by_quality_tier"][tier]["load"] = load_arm(counts, session, f"{tag}|{tier}|load")

    matched = _matched_unit_subset(session["counts"], tag)
    row["matched_unit_count_arm"] = {
        "n_units": MATCHED_UNIT_COUNT,
        "existence": state_profile(matched, f"{tag}|matched|existence"),
        "behaviour": behaviour_arm(matched, session, f"{tag}|matched|behaviour"),
    }
    return row


def _tier(rows: list[dict], tier: str) -> list[dict]:
    return [r["by_quality_tier"][tier] for r in rows if tier in r.get("by_quality_tier", {})]


def _pool_values(values: list[float]) -> dict:
    if len(values) < 2:
        return {"status": "not_computable", "n_sessions": len(values)}
    pooled = slope_across_sessions_test(values, alternative="two-sided")
    pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    pooled["median_value"] = float(np.median(values))
    return pooled


def _pool_correlations(entries: list[dict]) -> dict:
    return _pool_values([e["r"] for e in entries if e.get("status") == "computed"])


def _between_load_component(arms: list[dict]) -> dict:
    """The pooled graded coefficient split into the part carried within an
    item count and the part carried by item count itself.

    A pooled correlation computed over trials of different load is a mixture
    of a within-load and a between-load relationship. Only the within-load
    half is commensurable with an estimate from a task that holds a single
    item on every trial. The Pearson identity in ``item_count_component``
    makes the split exact, so it is computed and checked here rather than
    inferred from the raw and within-load estimates disagreeing in sign."""
    raw_values, within_values, product_values, residual_values = [], [], [], []
    state_values, report_values = [], []
    for arm in arms:
        component = arm.get("item_count_component", {})
        graded = arm.get("continuous_graded_report", {})
        parts = (component.get("item_count_vs_state_deviation", {}),
                 component.get("item_count_vs_report_error", {}),
                 graded.get("raw_report_error_vs_state_deviation", {}),
                 graded.get("partial_controlling_item_count", {}))
        if not all(part.get("status") == "computed" for part in parts):
            continue
        state, report, raw, within = (part["r"] for part in parts)
        product = state * report
        reconstructed = within * float(np.sqrt((1.0 - state ** 2) * (1.0 - report ** 2))) + product
        state_values.append(state)
        report_values.append(report)
        raw_values.append(raw)
        within_values.append(within)
        product_values.append(product)
        residual_values.append(raw - reconstructed)
    if len(raw_values) < 2:
        return {"status": "not_computable", "n_sessions": len(raw_values)}
    raw_pooled = _pool_values(raw_values)
    within_pooled = _pool_values(within_values)
    product_pooled = _pool_values(product_values)
    reversal = (raw_pooled.get("mean_value", 0.0) > 0.0) != (within_pooled.get("mean_value", 0.0) > 0.0)
    return {
        "status": "computed",
        "n_sessions": len(raw_values),
        "item_count_vs_state_deviation": _pool_values(state_values),
        "item_count_vs_report_error": _pool_values(report_values),
        "raw_pooled_coefficient": raw_pooled,
        "within_item_count_coefficient": within_pooled,
        "between_item_count_product_term": product_pooled,
        "largest_absolute_reconstruction_residual": float(np.max(np.abs(residual_values))),
        "reconstruction_identity_holds": bool(np.max(np.abs(residual_values)) < 1e-9),
        "raw_and_within_item_count_coefficients_differ_in_sign": bool(reversal),
        "interpretation": (
            f"Pooled over {len(raw_values)} sessions, the raw graded coefficient is "
            f"{raw_pooled.get('mean_value', float('nan')):+.4f} and the coefficient computed within item "
            f"count is {within_pooled.get('mean_value', float('nan')):+.4f}. Item count correlates with the "
            f"state deviation at {_pool_values(state_values).get('mean_value', float('nan')):+.4f} and with "
            f"report error at {_pool_values(report_values).get('mean_value', float('nan')):+.4f}, whose "
            f"product -- the between-item-count contribution to the raw coefficient -- pools to "
            f"{product_pooled.get('mean_value', float('nan')):+.4f}. The two coefficients "
            + ("differ" if reversal else "agree")
            + " in sign, and the split is an exact algebraic identity per session rather than an "
              "inference: the largest per-session discrepancy between the raw coefficient and its "
              f"reconstruction from the two parts is {float(np.max(np.abs(residual_values))):.2e}."),
    }


def _existence_verdict(profiles: list[dict]) -> dict:
    fitted = [p for p in profiles if p.get("status") == "fitted"]
    if len(fitted) < 4:
        return {"status": "too_few_fitted_sessions", "n_sessions_fitted": len(fitted),
                "branch": "inconclusive_below_detection_floor"}
    observed = [{int(lag): {"r_median": p["r_observed"][str(lag)]} for lag in p["lags_bins"]} for p in fitted]
    nulls = [{int(lag): {"r_null_median": p["r_permutation_null"][str(lag)]} for lag in p["lags_bins"]}
             for p in fitted]
    contrast, rotation, n_tested, n_clearing, clears_any, clears_majority = \
        _lag_existence_and_rotation(observed, nulls)
    differences = [p["mean_r_observed_minus_null"] for p in fitted]
    mdd = minimum_detectable_paired_difference(differences)
    effect = {
        "mean_r_observed_minus_null_over_sessions": float(np.mean(differences)),
        "median_r_observed_minus_null_over_sessions": float(np.median(differences)),
        "mean_r_observed_over_sessions": float(np.mean([p["mean_r_observed"] for p in fitted])),
        "mean_r_permutation_null_over_sessions": float(np.mean([p["mean_r_permutation_null"] for p in fitted])),
        "minimum_detectable_paired_difference_at_80pct_power": mdd,
    }
    if clears_any:
        branch = "cross_unit_state_present"
    elif mdd.get("status") == "computed" and mdd["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        branch = "cross_unit_state_absent"
    else:
        branch = "inconclusive_below_detection_floor"
    return {
        "status": "tested", "n_sessions_fitted": len(fitted),
        "per_lag_observed_versus_permutation_null": contrast,
        "n_lags_tested": n_tested, "n_lags_clearing_after_correction": n_clearing,
        "rotation_check": rotation, "clears_at_a_majority_of_lags": bool(clears_majority),
        "effect_size": effect, "branch": branch,
    }


def _cardinality_verdict(ladders: list[dict]) -> dict:
    fitted = [l for l in ladders if l.get("status") == "fitted"]
    slopes = [l["fractional_rank_slope_per_class"] for l in fitted
              if l.get("fractional_rank_slope_per_class") is not None]
    by_rung = {str(k): [l["fractional_rank_by_rung"][str(k)] for l in fitted
                        if str(k) in l["fractional_rank_by_rung"]] for k in CARDINALITY_RUNGS}
    auc_by_rung = {str(k): [l["decoding_auc_by_rung"][str(k)] for l in fitted
                            if str(k) in l["decoding_auc_by_rung"]] for k in CARDINALITY_RUNGS}
    summary = {
        "n_sessions_fitted": len(fitted),
        "mean_fractional_rank_by_rung": {k: (float(np.mean(v)) if v else None) for k, v in by_rung.items()},
        "n_sessions_by_rung": {k: len(v) for k, v in by_rung.items()},
        "mean_decoding_auc_by_rung": {k: (float(np.mean(v)) if v else None) for k, v in auc_by_rung.items()},
        "unreachable_rungs_by_reason": {
            str(k): sum(1 for l in fitted
                        if l["rungs"][str(k)].get("status") ==
                        "label_cardinality_exceeds_the_distinct_cued_positions_the_task_displays")
            for k in CARDINALITY_RUNGS},
    }
    if len(slopes) < 4:
        return {**summary, "status": "too_few_sessions_with_two_or_more_fitted_rungs",
                "n_sessions_with_a_slope": len(slopes), "branch": "inconclusive_below_detection_floor"}
    pooled = slope_across_sessions_test(slopes, alternative="two-sided")
    mdd = minimum_detectable_paired_difference(slopes)
    if pooled.get("significant_negative") or pooled.get("significant_positive"):
        branch = "content_rank_tracks_label_cardinality"
    elif mdd.get("status") == "computed" and mdd["mdd"] < CARDINALITY_MEANINGFUL_SLOPE_PER_CLASS:
        branch = "content_rank_independent_of_label_cardinality"
    else:
        branch = "inconclusive_below_detection_floor"
    return {**summary, "status": "tested", "n_sessions_with_a_slope": len(slopes),
            "pooled_fractional_rank_slope_per_class": pooled,
            "minimum_detectable_paired_difference_at_80pct_power": mdd,
            "meaningful_slope_threshold_per_class": CARDINALITY_MEANINGFUL_SLOPE_PER_CLASS,
            "branch": branch}


def _behaviour_verdict(arms: list[dict]) -> dict:
    computed = [a for a in arms if a.get("status") == "computed"]
    if len(computed) < 2:
        return {"status": "too_few_computed_sessions", "n_sessions": len(computed),
                "branch": "inconclusive_below_detection_floor"}
    gate = _pool_correlations([a["orthogonality_gate_state_deviation_vs_spike_count"] for a in computed])
    graded = {key: _pool_correlations([a["continuous_graded_report"][key] for a in computed])
              for key in computed[0]["continuous_graded_report"]}
    binary_sessions = [a for a in computed if a["binary_correct_versus_error"].get("status") == "computed"]
    binary = ({key: _pool_correlations([a["binary_correct_versus_error"][key] for a in binary_sessions])
               for key in ("raw_outcome_vs_state_deviation", "partial_controlling_spike_count",
                           "partial_controlling_trial_index",
                           "joint_partial_controlling_spike_count_and_trial_index")}
              if len(binary_sessions) >= 2 else
              {"status": "too_few_sessions_clear_the_error_trial_floor",
               "n_sessions_clearing": len(binary_sessions)})

    raw = graded["raw_report_error_vs_state_deviation"]
    joint = graded["joint_partial_controlling_spike_count_and_trial_index"]
    mdd = raw.get("minimum_detectable_paired_difference_at_80pct_power", {})
    mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
    named = _classify(gate, raw, joint, mdd_value)
    branch = {
        "no_rate_free_state_geometry_link_to_accuracy_above_the_reported_bound":
            "state_geometry_does_not_predict_accuracy",
        "underpowered_to_ask": "inconclusive_below_detection_floor",
        "not_computable": "inconclusive_below_detection_floor",
    }.get(named, named)

    sign_note = None
    if raw.get("status") == "tested":
        observed_sign = "positive" if raw["mean_value"] > 0.0 else "negative"
        sign_note = (
            f"The graded arm's pooled coefficient is {raw['mean_value']:.4f}, a {observed_sign} correlation "
            "between the trial's state deviation and its report error. The direction declared before "
            "fitting was positive -- a trial whose state sits further from the session's own mean state "
            "direction is reported less accurately -- so this "
            + ("matches" if observed_sign == "positive" else "CONTRADICTS")
            + " the declared expectation.")

    agreement = {"status": "not_computable"}
    binary_raw = binary.get("raw_outcome_vs_state_deviation") if isinstance(binary, dict) else None
    if raw.get("status") == "tested" and isinstance(binary_raw, dict) and binary_raw.get("status") == "tested":
        opposite = (raw["mean_value"] > 0.0) != (binary_raw["mean_value"] > 0.0)
        either_significant = bool(raw.get("significant")) or bool(binary_raw.get("significant"))
        agreement = {
            "status": "tested",
            "graded_report_error_vs_state_deviation": raw["mean_value"],
            "binary_correctness_vs_state_deviation": binary_raw["mean_value"],
            "signs_stand_in_the_agreeing_relation": bool(opposite),
            "n_sessions_graded": raw.get("n_sessions"), "n_sessions_binary": binary_raw.get("n_sessions"),
            "branch": ("continuous_and_binary_behavioural_arms_agree" if opposite and either_significant else
                       "continuous_and_binary_behavioural_arms_disagree" if either_significant else
                       "inconclusive_below_detection_floor"),
        }

    return {
        "status": "tested", "n_sessions": len(computed),
        "n_sessions_clearing_the_error_trial_floor": len(binary_sessions),
        "fraction_of_sessions_clearing_the_error_trial_floor": len(binary_sessions) / len(computed),
        "orthogonality_gate_state_deviation_vs_spike_count": gate,
        "continuous_graded_report": graded,
        "binary_correct_versus_error": binary,
        "graded_versus_binary_agreement": agreement,
        "sign_versus_the_declared_expectation": sign_note,
        "branch": branch, "branch_under_the_shared_estimator_rule": named,
    }


def _load_verdict(arms: list[dict]) -> dict:
    computed = [a for a in arms if a.get("status") == "computed"]
    if len(computed) < 2:
        return {"status": "too_few_computed_sessions", "n_sessions": len(computed),
                "branch": "inconclusive_below_detection_floor"}
    amplitude = {key: _pool_correlations([a["amplitude"][key] for a in computed])
                 for key in computed[0]["amplitude"]}
    slopes = [a["persistence_slope_per_item"] for a in computed if a.get("persistence_slope_per_item") is not None]
    persistence = (slope_across_sessions_test(slopes, alternative="two-sided") if len(slopes) >= 4 else
                   {"status": "too_few_sessions_with_a_matched_trial_count_at_every_item_count",
                    "n_sessions": len(slopes)})
    if len(slopes) >= 2:
        persistence["minimum_detectable_paired_difference_at_80pct_power"] = \
            minimum_detectable_paired_difference(slopes)
        persistence["mean_slope_per_item"] = float(np.mean(slopes))

    raw = amplitude["raw_amplitude_vs_item_count"]
    controlled = amplitude["partial_controlling_spike_count"]
    mdd = raw.get("minimum_detectable_paired_difference_at_80pct_power", {})
    if raw.get("status") == "tested" and (raw.get("significant_negative") or raw.get("significant_positive")):
        branch = "state_amplitude_tracks_item_count"
    elif mdd.get("status") == "computed" and mdd["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        branch = "state_amplitude_independent_of_item_count"
    else:
        branch = "inconclusive_below_detection_floor"
    survives = bool(controlled.get("status") == "tested" and
                    (controlled.get("significant_negative") or controlled.get("significant_positive")) and
                    raw.get("status") == "tested" and
                    (controlled["mean_value"] > 0.0) == (raw["mean_value"] > 0.0))
    return {
        "status": "tested", "n_sessions": len(computed),
        "amplitude": amplitude, "persistence_slope_per_item": persistence,
        "branch": branch,
        "amplitude_trend_survives_the_spike_count_control": survives,
        "amplitude_trend_interpretation": (
            None if raw.get("status") != "tested" else
            f"Pooled raw trend {raw['mean_value']:.4f} r units per additional item; controlling total "
            f"spike count {controlled.get('mean_value', float('nan')):.4f}. "
            + ("The trend survives the rate control, so it is not simply more items producing more spikes."
               if survives else
               "The trend does not survive the rate control, so on this evidence it is a firing-rate "
               "effect rather than a geometric one and is reported as such.")),
    }


def _group_difference(rows: list[dict], grouping: str, extract) -> dict:
    """A between-group difference read from the matched-unit-count arm, with
    the two groups' native unit counts compared first so a reader sees
    whether matching was needed as well as that it was applied."""
    groups: dict[str, list] = {}
    units: dict[str, list] = {}
    for row in rows:
        key = str(row[grouping])
        value = extract(row)
        if value is not None:
            groups.setdefault(key, []).append(value)
        units.setdefault(key, []).append(row["n_units_total"])
    if len(groups) != 2:
        return {"status": "not_a_two_group_comparison", "groups": sorted(groups)}
    (a_name, a_values), (b_name, b_values) = sorted(groups.items())
    if min(len(a_values), len(b_values)) < 2:
        return {"status": "too_few_sessions_in_a_group",
                "n": {a_name: len(a_values), b_name: len(b_values)}}
    rng = np.random.default_rng(stable_seed(f"watters_state_geometry|{grouping}|unit_counts"))
    pooled_units = np.array(units[a_name] + units[b_name], dtype=float)
    observed_unit_gap = float(np.mean(units[a_name]) - np.mean(units[b_name]))
    draws = np.array([
        (lambda p: np.mean(p[:len(units[a_name])]) - np.mean(p[len(units[a_name]):]))(rng.permutation(pooled_units))
        for _ in range(N_PERM)])
    return {
        "status": "tested", "groups": [a_name, b_name],
        "n_sessions": {a_name: len(a_values), b_name: len(b_values)},
        "native_unit_count_mean": {a_name: float(np.mean(units[a_name])), b_name: float(np.mean(units[b_name]))},
        "native_unit_count_difference": observed_unit_gap,
        "native_unit_count_difference_p_value": float(np.mean(np.abs(draws) >= abs(observed_unit_gap))),
        "matched_unit_count": MATCHED_UNIT_COUNT,
        "mean_at_matched_unit_count": {a_name: float(np.mean(a_values)), b_name: float(np.mean(b_values))},
        "difference_at_matched_unit_count": float(np.mean(a_values) - np.mean(b_values)),
    }


def _behaviour_session_counts(root: Path) -> dict:
    """Attempted and completed trial rows per behavioural session date, from
    the behaviour logs directly -- the denominator the per-session trial
    accounting reconciles against, since the loaded table is already
    restricted to completed trials."""
    _, behaviour_dir = watters_directories(root)
    counts: dict[str, dict] = {}
    for variant in ("ring", "triangle"):
        frame = pd.read_csv(behaviour_dir / f"{variant}.csv", usecols=["subject", "session", "completed"])
        for (animal, date), group in frame.groupby(["subject", "session"]):
            counts[f"{animal}_{date}"] = {
                "task_variant": variant, "n_trials_attempted": int(len(group)),
                "n_trials_completed": int(group.completed.astype(bool).sum())}
    return counts


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))


def _resume() -> dict[str, dict]:
    if not OUTPUT_PATH.exists():
        return {}
    try:
        previous = json.loads(OUTPUT_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    return {r["session"]: r for r in previous.get("sessions", [])
            if r.get("analysis_version") == ANALYSIS_VERSION}


def main() -> None:
    t0 = time.time()
    root = data_root()
    reachability = reachability_block()
    cardinality_reachability = cardinality_reachability_block(root)
    attempted = _behaviour_session_counts(root)
    resumed = _resume()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "corpus": (
            "Multi-object spatial working memory in macaque frontal cortex, two animals, fixed 1.0 s "
            "maintenance delay with the objects absent, continuous saccadic report. Source: "
            "'Working Memory of Multi-Object Scenes in Primate Frontal Cortex', Watters, Gabel, "
            "Tenenbaum and Jazayeri, bioRxiv PREPRINT posted 2026-01-27, DOI 10.64898/2026.01.27.702062, "
            "data DANDI 000620. This is an unreviewed preprint: no peer-reviewed journal version was "
            "found, and it is cited on that basis."
        ),
        "structure": (
            "Pooled single structure. Every electrode in the shared metadata carries the location "
            "'unknown', so no area-resolved split is attempted; units are pooled across all probes."
        ),
        "quality_tier_choice": QUALITY_TIER_CHOICE,
        "matched_instrument_rule_declared_before_fitting": MATCHED_INSTRUMENT_RULE_DECLARED_BEFORE_FITTING,
        "reachability_declared_before_any_test": reachability,
        "existence_decision_rule_declared_before_fitting": EXISTENCE_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "cardinality_reachability_declared_before_any_test": cardinality_reachability,
        "cardinality_decision_rule_declared_before_fitting": CARDINALITY_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "behaviour_decision_rule_declared_before_fitting": BEHAVIOUR_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "load_decision_rule_declared_before_fitting": LOAD_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "scope": {
            "bin_ms": BIN_MS, "window_width_bins": DECIDING_WIDTH_BINS,
            "delay_epoch_s": WATTERS_DELAY_WINDOW_S,
            "n_permutations": N_PERM,
            "correct_report_distance_threshold_normalized_units": CORRECT_REPORT_DISTANCE_THRESHOLD,
            "error_trial_floor_for_the_binary_arm": MIN_ERROR_TRIALS_FOR_REACHABILITY,
            "matched_unit_count": MATCHED_UNIT_COUNT,
            "min_units_per_session": WATTERS_MIN_UNITS,
            "min_trials_per_item_count_level": MIN_TRIALS_PER_LOAD_LEVEL,
            "seeds": "every fit is seeded by a stable hash of the session identifier and the arm name",
            "git_commit": git_commit(ROOT),
        },
        "status": "running", "sessions": [], "sessions_refused": [],
    }
    _flush(output)

    refused: list[dict] = []
    rows: list[dict] = []
    seen = 0
    for session in iter_watters(root, bin_ms=BIN_MS):
        seen += 1
        key = session["session"]
        if session["status"] != "loaded":
            refused.append({
                "session": key, "animal": session["animal"], "session_date": session["session_date"],
                "task_variant": session["behavioural_task_variant"], "status": session["status"],
                "n_units_seen_in_cache": session.get("n_units_seen"),
                "n_trials_with_a_complete_delay_window": session.get("n_trials"),
                **attempted.get(key, {})})
        else:
            row = resumed.get(key) or analyse_session(session)
            row.update(attempted.get(key, {}))
            rows.append(row)
        output["sessions"] = rows
        output["sessions_refused"] = refused
        output["n_sessions_seen"] = seen
        _flush(output)
        print(f"progress {seen} {key} {session['status']} elapsed={time.time() - t0:.0f}s", flush=True)

    output["zero_drop_accounting"] = {
        "n_behavioural_session_dates_seen": seen,
        "n_sessions_analysed": len(rows),
        "n_sessions_refused": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                               for reason in sorted({r["status"] for r in refused})},
        "reconciles": bool(seen == len(rows) + len(refused)),
        "n_trials_attempted_over_all_session_dates": sum(v["n_trials_attempted"] for v in attempted.values()),
        "n_trials_completed_over_all_session_dates": sum(v["n_trials_completed"] for v in attempted.values()),
        "n_trials_analysed": sum(r["n_trials"] for r in rows),
        "by_animal": {animal: sum(1 for r in rows if r["animal"] == animal)
                      for animal in sorted({r["animal"] for r in rows})},
        "by_task_variant": {variant: sum(1 for r in rows if r["task_variant"] == variant)
                            for variant in sorted({r["task_variant"] for r in rows})},
    }

    def _subsets(rows_in: list[dict]) -> dict[str, list[dict]]:
        subsets = {"pooled": rows_in}
        for animal in sorted({r["animal"] for r in rows_in}):
            subsets[f"animal_{animal}"] = [r for r in rows_in if r["animal"] == animal]
        for variant in sorted({r["task_variant"] for r in rows_in}):
            subsets[f"task_variant_{variant}"] = [r for r in rows_in if r["task_variant"] == variant]
        return subsets

    results: dict = {}
    for tier in (PRIMARY_QUALITY_TIER, "good_single_units_only"):
        tier_rows = [r for r in rows if tier in r["by_quality_tier"]]
        tier_block: dict = {"n_sessions": len(tier_rows)}
        for name, subset in _subsets(tier_rows).items():
            block = {
                "n_sessions": len(subset),
                "existence": _existence_verdict([r["by_quality_tier"][tier]["existence"] for r in subset]),
                "behaviour": _behaviour_verdict([r["by_quality_tier"][tier]["behaviour"] for r in subset]),
            }
            if tier == PRIMARY_QUALITY_TIER:
                block["cardinality_ladder"] = _cardinality_verdict(
                    [r["by_quality_tier"][tier]["cardinality_ladder"] for r in subset])
                block["load"] = _load_verdict([r["by_quality_tier"][tier]["load"] for r in subset])
            tier_block[name] = block
        results[tier] = tier_block

    matched_rows = [r for r in rows if r["matched_unit_count_arm"]["existence"].get("status") == "fitted"]
    results["matched_unit_count_arm"] = {
        "n_units": MATCHED_UNIT_COUNT, "n_sessions": len(matched_rows),
        **{name: {
            "n_sessions": len(subset),
            "existence": _existence_verdict([r["matched_unit_count_arm"]["existence"] for r in subset]),
            "behaviour": _behaviour_verdict([r["matched_unit_count_arm"]["behaviour"] for r in subset]),
        } for name, subset in _subsets(matched_rows).items()},
        "between_group_differences": {
            "state_by_animal": _group_difference(
                matched_rows, "animal",
                lambda r: r["matched_unit_count_arm"]["existence"].get("mean_r_observed_minus_null")),
            "state_by_task_variant": _group_difference(
                matched_rows, "task_variant",
                lambda r: r["matched_unit_count_arm"]["existence"].get("mean_r_observed_minus_null")),
            "graded_behaviour_by_animal": _group_difference(
                matched_rows, "animal",
                lambda r: r["matched_unit_count_arm"]["behaviour"].get("continuous_graded_report", {})
                .get("raw_report_error_vs_state_deviation", {}).get("r")),
            "graded_behaviour_by_task_variant": _group_difference(
                matched_rows, "task_variant",
                lambda r: r["matched_unit_count_arm"]["behaviour"].get("continuous_graded_report", {})
                .get("raw_report_error_vs_state_deviation", {}).get("r")),
        },
    }

    output["results"] = results
    primary = results[PRIMARY_QUALITY_TIER]["pooled"]
    output["branches"] = {
        "existence": primary["existence"]["branch"],
        "cardinality_ladder": primary["cardinality_ladder"]["branch"],
        "behaviour": primary["behaviour"]["branch"],
        "graded_versus_binary_agreement": primary["behaviour"].get("graded_versus_binary_agreement", {}).get("branch"),
        "load": primary["load"]["branch"],
    }
    output["wall_clock_s"] = time.time() - t0
    output["status"] = "complete"
    _flush(output)
    print(json.dumps(output["branches"], indent=2))


if __name__ == "__main__":
    main()
