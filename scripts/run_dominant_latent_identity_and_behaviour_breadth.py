"""run_dominant_latent_identity_and_behaviour_breadth.py -- what does the
macaque lPFC dominant latent carry, and how far across the corpus does the
behavioural correlate of state geometry reach?

results/state_latent_identity.json establishes that the leading latent of
this population carries a large, cross-trial-stable share of the variance,
that this is not a position or lag confound, and that it is not the current
item's identity. Three candidate identities have already come back null
(total spike count, trial index, item label) and the one positive --
'state_is_a_per_trial_gain' -- is thin: median observed rank-1 share
0.8950377 against a median matched-null reference of 0.8925854, a
0.24-percentage-point margin. That pooled share is also an average of two
very different recording regimes (median 0.6557 in the 2021 sessions, 56-160
units, against 0.9768 in the 2022-and-2024 sessions, 183-779 units), so its
magnitude is a property of population size, not of biology. Both facts are
stated wherever that verdict is stated, here and downstream.

This module tests the one untested candidate identity -- trial history --
and separately measures how far the corpus's behavioural result generalises.

Three questions, three independent result blocks:

1. Is the dominant latent carrying the PREVIOUS trial's item? The leading
   latent's leave-one-latent-out ablation-cost rank and the population's
   macro one-vs-rest decoding accuracy are recomputed with the item shown
   one trial earlier as the label, on identical delay-epoch activity, then
   extended to a history lag profile (1, 2, 3, 5 and 10 trials back) against
   a circular-shift null that preserves each session's label sequence
   exactly. A history effect decays with lag; a decoding artifact does not.
   Gated on reproducing the corpus's already-published current-item pooled
   fractional rank first: a previous-item number means nothing if the
   current-item number does not reproduce.

2. Why does the behavioural result rest on 11 of the corpus's 25 sessions,
   and on none from the third animal? Per-session correct and error counts
   are recomputed from the deposited files for all 25 sessions, with the
   outcome field's own coding recorded per session so that a near-empty
   error class can be told apart from a differently-coded or missing one.
   The behavioural analysis is then repeated at error-trial floors of 60
   (the pre-declared primary, unchanged), 45 and 30, with per-animal session
   counts at every floor, and against per-session unit count.

3. Does the dominant latent's per-trial amplitude predict trial outcome
   directly, rather than the rate-free deviation observable that carries the
   published behavioural result?

Reuses run_state_content_link.session_subtractive_test (the estimator that
produced the number this module must reproduce), run_state_behavior_link's
delay binning and per-trial amplitude covariates,
run_rate_free_state_geometry_behavior_link's deviation observable and
decision-rule classifier, and the project's paired sign-flip pooling
throughout; no estimator is forked.

Animal identity: the deposit's own README states three animals -- 10
sessions from monkey A, 8 from monkey H, 7 from monkey J -- matching the
25 sessions' three date blocks exactly by size. Any project note describing
this corpus as two-animal is wrong; the count is stated by the depositor,
though the block-to-animal assignment itself remains an inference from
matching session counts rather than a depositor-confirmed table.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import data_root  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, _classify, rate_free_state_deviation,
)
from run_state_behavior_link import (  # noqa: E402
    _counts_from_spikes, _panichello_directory, trial_amplitude_covariates,
)
from run_state_content_link import (  # noqa: E402
    _one_sample_sign_flip, _stable_seed, session_subtractive_test, usable_label,
)
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from state_persistence import _ols_slope, slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test,
    partial_correlation_permutation_test, permutation_pvalue, spearman_permutation_test, stable_seed,
)

CORPUS_KEY = "panichello_2024"

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "dominant_latent_identity_and_behaviour_breadth.json"
CHECKPOINT_PATH = (OUTPUT_PATH.parent / ".checkpoints"
                   / "dominant_latent_identity_and_behaviour_breadth_checkpoint.json")
SEED_NAMESPACE = "dominant_latent_identity_and_behaviour_breadth"

# The published current-item value this module must reproduce before any previous-item number is
# interpreted, and the tolerance that decides "reproduced", both fixed before the first fit runs.
# The estimator is fully seeded, so an exact match is expected; the tolerance exists so that a
# platform-level floating-point difference is not misreported as a failure to reproduce. One tenth
# of the statistic's own resolution (1/(k-1) = 1/7 with k = 8 latents) is far below any difference
# that could change a verdict.
EXPECTED_POOLED_CURRENT_ITEM_FRACTIONAL_RANK = 0.6742857142857143
EXPECTED_N_SESSIONS = 25
FRACTIONAL_RANK_RESOLUTION_AT_EIGHT_LATENTS = 1.0 / 7.0
GATE_TOLERANCE = FRACTIONAL_RANK_RESOLUTION_AT_EIGHT_LATENTS / 10.0

HISTORY_LAGS = (1, 2, 3, 5, 10)
PRIMARY_HISTORY_LAG = 1
N_CIRCULAR_SHIFT_REPLICATES = 30
MIN_CIRCULAR_SHIFT_OFFSET = 20

ERROR_FLOORS = (60, 45, 30)
PRIMARY_ERROR_FLOOR = 60
N_UNIT_MATCHED_DRAWS = 5
N_PERM = 10000

DATE_BLOCK_TO_ANIMAL = {"21": "monkey_A", "22": "monkey_H", "24": "monkey_J"}
ANIMAL_RECORDED_AREA = {
    "monkey_A": "area 8",
    "monkey_H": "areas 8 and 9/46",
    "monkey_J": "area 9/46",
}
ANIMAL_ASSIGNMENT_SOURCE = (
    "The deposited README states three animals with 10, 8 and 7 sessions, for monkey A, monkey H and "
    "monkey J in that order, and the dataset's public landing page records the recorded area per "
    "animal: monkey A in area 8, monkey H in areas 8 and 9/46, monkey J in area 9/46. The 25 staged "
    "sessions form three date blocks of exactly 10 (2021), 8 (2022) and 7 (2024) sessions, so the "
    "NUMBER of animals -- three, not two -- is depositor-stated rather than inferred, and every pooled "
    "number computed on this corpus pools three animals and at least two prefrontal areas. The "
    "assignment of a date block to a named animal remains a strong INFERENCE, stated as such: no "
    "deposited file carries a subject field, and the inference rests on the block sizes matching the "
    "README's 10/8/7 in date order plus two further properties -- unit count per session and task "
    "accuracy -- separating cleanly and without overlap along the same block boundaries."
)

HISTORY_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Reproduce the corpus's pooled current-item fractional rank first. The leading latent's "
    "leave-one-latent-out ablation-cost fractional rank runs from 0 (removing the leading latent costs "
    "content decoding more than removing any other latent) to 1 (it costs least), with a null of 0.5, "
    "and is pooled across sessions by the one-sample paired sign-flip test. If the reproduction does "
    "not land within the declared tolerance of the published value at the published session count, no "
    "previous-item number is interpreted and the history block reports the failure instead.\n"
    "Given reproduction, the same statistic is recomputed on identical delay-epoch activity with the "
    "item shown one trial earlier as the label. Before any fit runs:\n"
    "  - If the pooled previous-item fractional rank is below 0.5 with two-sided p <= 0.05 while the "
    "current-item rank stays above 0.5, the branch is "
    "'rank_inverts_previous_item_in_dominant_latent'.\n"
    "  - If the pooled previous-item fractional rank is not significant at the two-sided 0.05 level and "
    "its minimum detectable paired difference at 80% power is below one rank step (1/(k-1), the "
    "statistic's own resolution -- an effect smaller than one rank step is not reachable by this "
    "statistic at any session count), the branch is "
    "'no_inversion_previous_item_not_in_dominant_latent'.\n"
    "  - If it is not significant and the minimum detectable paired difference is at or above one rank "
    "step, the branch is 'inconclusive_below_detection_floor'.\n"
    "  - The rule as written names no branch for a previous-item rank significantly ABOVE 0.5, the "
    "mirror of the inversion case. If that occurs it is named "
    "'previous_item_rank_significantly_above_the_null_off_branch_list' rather than forced onto either "
    "listed branch.\n"
    "The full-model macro one-vs-rest decoding accuracy for the previous-item label, and the number of "
    "sessions where it clears its own label-permutation null, are reported beside the rank whichever "
    "branch fires: a rank among ablation costs that are all at noise level is a rank of noise, and a "
    "reader cannot tell that from the rank alone."
)

HISTORY_LAG_PROFILE_NULL_DECLARED_BEFORE_FITTING = (
    "The lag profile's null is a circular shift of each session's own label sequence. For each of 30 "
    "replicates a per-session offset is drawn uniformly from the integers in "
    f"[{MIN_CIRCULAR_SHIFT_OFFSET}, n_trials - {MIN_CIRCULAR_SHIFT_OFFSET}], the label sequence is "
    "rolled by that offset, and the pooled mean fractional rank and pooled mean decoding accuracy are "
    "recomputed across all sessions. A circular shift preserves each session's label multiset and its "
    "full sequential ordering statistics exactly and only breaks the alignment between labels and "
    "activity, and the offset floor keeps every replicate clear of both the current item (offset 0) and "
    "every tested history lag (at most 10). The same shift null serves every lag, so the null band is "
    "one distribution rather than five. Per-lag significance is the two-sided fraction of replicates "
    "whose pooled value is at least as far from the null distribution's own centre as the observed "
    f"value is, so the smallest attainable p-value is 1/{N_CIRCULAR_SHIFT_REPLICATES + 1}. A history "
    "effect must decay with lag; a decoding artifact must not."
)

BEHAVIOUR_FLOOR_RULE_DECLARED_BEFORE_FITTING = (
    "The 60-error-trial floor is the pre-declared primary and does not change. The 45 and 30 floors are "
    "sensitivity analyses, reported whatever they show, including where they weaken or reverse the "
    "effect, and are never substituted for the primary. Because no floor named here is above 60, the "
    "session set at each lower floor is a strict superset of the primary's, so a disagreement between "
    "floors is a statement about the added sessions and not about a different analysis. Disagreement "
    "between floors is itself the finding and is reported as such rather than resolved in favour of "
    "whichever floor is more agreeable."
)

ERROR_RATE_DOSE_RESPONSE_RULE_DECLARED_BEFORE_FITTING = (
    "An error-trial floor does not sample animals evenly: it admits sessions in proportion to how often "
    "the animal failed, so the pooled correlation is estimated mostly from the worst-performing "
    "subject. Whether that matters is measurable. Regress each included session's own correlation "
    "coefficient on that session's error rate by ordinary least squares, with a permutation p-value "
    "(the pairing between error rate and coefficient is shuffled) and a percentile bootstrap "
    "confidence interval on the slope. The coefficient is a correlation with a correct-equals-one "
    "outcome, so a MORE NEGATIVE coefficient is a STRONGER coupling; a negative slope therefore means "
    "the coupling strengthens as behaviour degrades. Before any fit runs:\n"
    "  - Significant two-sided negative slope: branch "
    "'coupling_strengthens_as_behaviour_degrades', a dose-response consistent with the state mattering "
    "more when performance is worse.\n"
    "  - Significant two-sided positive slope: branch "
    "'coupling_weakens_as_behaviour_degrades_off_branch_list', the mirror case, named rather than "
    "forced onto either intended branch.\n"
    "  - Not significant, and the slope's confidence-interval half-width is below the slope that would "
    "be needed to carry the whole pooled correlation across the observed error-rate range (that is, "
    "|pooled r| divided by the observed error-rate span -- the slope at which the effect would be zero "
    "at the best-performing included session and equal to the pooled value at the worst): branch "
    "'coupling_is_flat_across_performance_regimes', the effect generalising across performance "
    "regimes.\n"
    "  - Not significant with a wider interval than that: branch "
    "'inconclusive_below_detection_floor'.\n"
    "Neither of the two intended branches is a negative result. Reported alongside: the pooled effect "
    "within each animal that has enough sessions to stand alone, and descriptively for those that do "
    "not."
)

AMPLITUDE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, correlate the leading latent's per-trial amplitude (the per-trial score of the "
    "leading component of the centred trial-by-window matrix, the same quantity "
    "results/behavior_amplitude_rate_controls.json calls the leading-component gain) with trial "
    "outcome, and pool the per-session coefficients across sessions with the two-sided paired sign-flip "
    "test, at each of the three error floors. An orthogonality gate against total delay-epoch spike "
    "count is run first, the same gate the rate-free analysis uses, and reported whatever it shows. "
    "Before any fit runs:\n"
    "  - If the gate is NOT significant, the amplitude is separable from rate and the deciding "
    "statistic is the raw correlation with outcome.\n"
    "  - If the gate IS significant, the amplitude is not separable from rate and the raw correlation "
    "cannot be attributed to the latent rather than to firing rate; the deciding statistic is then the "
    "partial correlation controlling total spike count. Which of the two applies is decided by the gate "
    "and not chosen after seeing either correlation, and both are reported either way.\n"
    "  - The branch is 'dominant_latent_amplitude_predicts_outcome' if the deciding statistic is "
    "significant at the two-sided 0.05 level; 'dominant_latent_amplitude_does_not_predict_outcome' if "
    "it is not significant and its minimum detectable paired difference at 80% power is below "
    f"{MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r units, the same smallest-meaningful-effect scale the rest "
    "of this project's behavioural line reports on; 'inconclusive_below_detection_floor' otherwise."
)


# ============================================================================
# Label alignment
# ============================================================================

def history_labels(labels_all: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Relabel every trial by the item presented ``lag`` trials earlier in
    the session's recorded order, returning (labels, defined_mask).

    Trial i takes ``labels_all[i - lag]``. The predecessor is the trial that
    physically preceded it, whether the animal got that trial right or wrong,
    so this must be applied to the session's FULL trial sequence and any
    correct-trial restriction applied afterwards -- restricting first and
    shifting second would silently label each trial with the previous
    CORRECT trial's item, skipping over the errors in between, which is a
    different quantity.

    The first ``lag`` trials of a session have no predecessor inside the
    session and are excluded by the returned mask; sessions are separate
    recording days and no trial pair may straddle two of them. The masked-out
    positions are filled with the first label purely to keep the array
    dtype-clean and are never read.
    """
    labels_all = np.asarray(labels_all)
    n = len(labels_all)
    if lag < 1:
        raise ValueError(f"history lag must be at least 1 trial, got {lag}")
    shifted = np.empty(n, dtype=labels_all.dtype)
    shifted[lag:] = labels_all[: n - lag]
    shifted[:lag] = labels_all[0] if n else 0
    return shifted, np.arange(n) >= lag


def circular_shift_labels(labels_all: np.ndarray, offset: int) -> np.ndarray:
    """Label sequence rolled forward by ``offset`` trials, wrapping at the
    session boundary: position i takes ``labels_all[(i - offset) % n]``.

    Preserves the session's label multiset and every sequential ordering
    statistic of the label sequence exactly, breaking only its alignment with
    the neural activity. At ``offset == lag`` this coincides with
    :func:`history_labels` except at the wrapped positions, which is why the
    null draws offsets far above any tested lag.
    """
    return np.roll(np.asarray(labels_all), offset)


# ============================================================================
# Session loading
# ============================================================================

def _session_paths(root: Path) -> list[Path]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    return [Path(p) for p in sorted(glob.glob(str(directory / "*.mat")))]


def _load_session(path: Path) -> dict:
    """Every per-session array the three result blocks need, read once.

    The raster is left in its deposited integer dtype and binned directly:
    the largest session's raster is 810 x 1950 x 716, which a float cast
    would expand to nine gigabytes for no gain, while the binned counts it
    reduces to are a few tens of megabytes.
    """
    raw = loadmat(str(path), squeeze_me=True)
    spikes = raw["spks"]
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr_raw = np.asarray(raw["isCorr"]).reshape(-1)
    cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1)
    cue_ang = np.asarray(raw["cueAng"], dtype=float).reshape(-1)
    counts = np.asarray(_counts_from_spikes(spikes, time_ms), dtype=float)
    is_corr = is_corr_raw.astype(bool)
    finite = np.isfinite(np.asarray(is_corr_raw, dtype=float))
    return {
        "session": path.stem,
        "date_block": path.stem[:2],
        "animal": DATE_BLOCK_TO_ANIMAL.get(path.stem[:2], "unassigned"),
        "counts": counts,
        "is_corr": is_corr,
        "cue_idx": cue_idx.astype(int),
        "cue_ang": cue_ang,
        "n_trials": int(counts.shape[0]),
        "n_units": int(counts.shape[1]),
        "n_bins": int(counts.shape[2]),
        "n_correct": int(is_corr.sum()),
        "n_error": int((~is_corr).sum()),
        "outcome_field_dtype": str(is_corr_raw.dtype),
        "outcome_field_distinct_values": sorted(float(v) for v in np.unique(is_corr_raw)),
        "outcome_field_n_non_finite": int((~finite).sum()),
        "outcome_field_length_matches_trials": bool(len(is_corr_raw) == counts.shape[0]),
        "cue_label_distinct_values": sorted(int(v) for v in np.unique(cue_idx)),
    }


# ============================================================================
# Ablation-cost rank of the leading latent, by label
# ============================================================================

def _rank_for_labels(counts: np.ndarray, labels: np.ndarray, seed: int,
                     with_permutation_nulls: bool) -> dict:
    """Leading-latent ablation-cost rank for one session under one labelling.

    The label-eligibility mask is applied to the raw counts BEFORE the
    unit-wise transform is fit, not to the finished feature matrix
    afterwards: the transform's per-unit mean and scale are estimated from
    whichever trials enter the fit, so masking second would standardise
    against a different trial set and give a different number from the same
    data. The corpus's published current-item value was produced by masking
    first, and reproducing it is the precondition for interpreting any
    other labelling.
    """
    ok, reason, mask = usable_label(labels)
    if not ok:
        return {"status": "no_usable_label", "reason": reason}
    kept = counts[mask]
    if kept.shape[0] < 8:
        return {"status": "too_few_trials_after_label_filter", "n_trials": int(kept.shape[0])}
    features = FrozenPSTHTransform().fit(kept).transform(kept).mean(axis=2)[:, :, None]
    result = session_subtractive_test(features, np.asarray(labels)[mask], seed,
                                      with_permutation_nulls=with_permutation_nulls)
    result["n_trials_dropped_by_label_filter"] = int((~mask).sum())
    return result


def _pool_ranks(values: list[float]) -> dict:
    return _one_sample_sign_flip(values, 0.5, alternative="less")


def _paired(a: list[float], b: list[float]) -> dict:
    if len(a) < 4 or len(a) != len(b):
        return {"status": "not_computed", "n": len(a)}
    result = paired_sign_flip_test(np.asarray(a), np.asarray(b), alternative="two-sided")
    return {"status": "tested", "n": int(result["n"]), "mean_difference": float(result["mean_diff"]),
            "p_value": float(result["p_value"]), "ci_lower": float(result["ci_lower"]),
            "ci_upper": float(result["ci_upper"])}


# ============================================================================
# Behavioural blocks
# ============================================================================

def _behaviour_session_arrays(session: dict) -> dict | None:
    """The rate-free deviation observable, the leading latent's per-trial
    amplitude, and the two covariates both are gated against, for one
    session. Trials with no defined state direction (zero total activity)
    are dropped from every series together so all of them stay aligned."""
    counts = session["counts"]
    if counts.shape[0] < 16:
        return None
    activity_by_unit = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity_by_unit)
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return None
    finite = np.isfinite(deviation)
    if finite.sum() < 16:
        return None
    # The previous trial's cue angle enters as its sine and cosine so that a circular quantity is
    # controlled for without imposing an arbitrary cut point on the circle. It is shifted on the
    # session's full trial sequence first and masked second, for the reason history_labels documents.
    previous_angle, has_predecessor = history_labels(session["cue_ang"], PRIMARY_HISTORY_LAG)
    return {
        "is_corr": session["is_corr"][finite].astype(float),
        "deviation": deviation[finite],
        "amplitude": np.asarray(covariates["leading_component_score_gain"])[finite],
        "spike_count": activity_by_unit.sum(axis=1)[finite],
        "trial_index": np.arange(counts.shape[0], dtype=float)[finite],
        "previous_item_sin": np.sin(previous_angle)[finite],
        "previous_item_cos": np.cos(previous_angle)[finite],
        "has_predecessor": has_predecessor[finite],
        "n_trials_total": int(counts.shape[0]),
        "n_trials_with_defined_direction": int(finite.sum()),
    }


def _corr(outcome: np.ndarray, covariate: np.ndarray, controls: list[np.ndarray], seed_tag: str,
          n_perm: int = N_PERM) -> dict:
    rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|{seed_tag}"))
    return partial_correlation_permutation_test(outcome, covariate, controls=controls, n_perm=n_perm, rng=rng)


def _correlation_family(arrays: dict, observable_key: str, session_id: str) -> dict:
    outcome, x = arrays["is_corr"], arrays[observable_key]
    spike_count, trial_index = arrays["spike_count"], arrays["trial_index"]
    tag = f"{observable_key}|{session_id}"
    return {
        "orthogonality_gate": _corr(x, spike_count, [], f"{tag}|gate"),
        "raw_outcome_vs_observable": _corr(outcome, x, [], f"{tag}|raw"),
        "partial_controlling_spike_count": _corr(outcome, x, [spike_count], f"{tag}|ctrl_spike"),
        "partial_controlling_trial_index": _corr(outcome, x, [trial_index], f"{tag}|ctrl_trial"),
        "joint_partial_controlling_spike_count_and_trial_index": _corr(
            outcome, x, [spike_count, trial_index], f"{tag}|ctrl_joint"),
    }


def _pool_correlations(per_session: list[dict], key: str) -> dict:
    values = [s[key]["r"] for s in per_session if s[key].get("status") == "computed"]
    return slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}


def _mdd(per_session: list[dict], key: str) -> dict:
    values = [s[key]["r"] for s in per_session if s[key].get("status") == "computed"]
    if len(values) < 2:
        return {"status": "not_computable", "n": len(values), "reason": "fewer than 2 sessions"}
    return minimum_detectable_paired_difference(values)


def _animal_counts(session_ids: list[str], census: dict) -> dict:
    counts = {animal: 0 for animal in sorted(set(DATE_BLOCK_TO_ANIMAL.values()))}
    for session_id in session_ids:
        counts[census[session_id]["animal"]] = counts.get(census[session_id]["animal"], 0) + 1
    return counts


def _error_rate_dose_response(included: list[str], rows: dict[str, dict], census: dict,
                              pooled_raw: dict) -> dict:
    """Does the per-session strength of the outcome correlate change with how
    badly the animal performed that session?

    An error-trial floor admits sessions in proportion to task failure, so
    the pooled coefficient is estimated mostly from the worst performer. The
    regression of each session's own coefficient on its own error rate says
    whether that selection changes the answer.
    """
    error_rate = [census[s]["error_rate"] for s in included]
    coefficient = [rows[s]["raw_outcome_vs_observable"]["r"] for s in included]
    slope = _ols_slope(error_rate, coefficient) if len(included) >= 4 else None
    if slope is None:
        return {"status": "not_computable", "n_sessions": len(included),
                "reason": "fewer than 4 sessions, or no spread in error rate to regress against"}

    rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|error_rate_dose_response|permutation"))
    null = np.array([_ols_slope(error_rate, rng.permutation(coefficient)) for _ in range(N_PERM)], dtype=float)
    p_value = permutation_pvalue(np.abs(null) >= abs(slope))

    def _slope_of(pairs: np.ndarray) -> float:
        value = _ols_slope(pairs[:, 0], pairs[:, 1])
        return float("nan") if value is None else value

    boot_rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|error_rate_dose_response|bootstrap"))
    _observed, ci_lower, ci_upper = bootstrap_ci(
        np.column_stack([error_rate, coefficient]), _slope_of, rng=boot_rng)

    # The reference slope is the one at which the whole pooled coefficient would be explained by the
    # error-rate spread alone -- zero coupling at the best-performing included session, the full pooled
    # value at the worst. A confidence interval narrower than that has ruled out a dose-response large
    # enough to matter; a wider one has not.
    error_rate_span = float(np.ptp(error_rate))
    pooled_r = pooled_raw.get("mean_value")
    reference_slope = (abs(pooled_r) / error_rate_span
                       if pooled_r is not None and error_rate_span > 0 else None)
    ci_half_width = (ci_upper - ci_lower) / 2.0

    significant = bool(p_value <= 0.05)
    if significant and slope < 0:
        branch = "coupling_strengthens_as_behaviour_degrades"
    elif significant and slope > 0:
        branch = "coupling_weakens_as_behaviour_degrades_off_branch_list"
    elif reference_slope is not None and ci_half_width < reference_slope:
        branch = "coupling_is_flat_across_performance_regimes"
    else:
        branch = "inconclusive_below_detection_floor"

    return {
        "status": "tested",
        "branch": branch,
        "n_sessions": len(included),
        "slope_of_session_correlation_on_session_error_rate": slope,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_half_width": ci_half_width,
        "reference_slope_that_would_carry_the_whole_pooled_correlation": reference_slope,
        "pooled_correlation_used_for_the_reference_slope": pooled_r,
        "error_rate_range_across_included_sessions": [float(np.min(error_rate)), float(np.max(error_rate))],
        "error_rate_span": error_rate_span,
        "per_session": {s: {"error_rate": census[s]["error_rate"], "n_error": census[s]["n_error"],
                            "animal": census[s]["animal"], "raw_r": rows[s]["raw_outcome_vs_observable"]["r"]}
                        for s in included},
        "sign_reading": (
            "The per-session coefficient is a correlation with a correct-equals-one outcome, so a more "
            "negative coefficient is a stronger coupling. A negative slope therefore means the coupling "
            "strengthens as behaviour degrades."
        ),
    }


def _per_animal_effect(included: list[str], rows: dict[str, dict], census: dict) -> dict:
    """The pooled correlation within each animal that has enough sessions to
    stand alone, and the raw per-session values for those that do not.

    The pooled sign-flip test needs at least four sessions before its
    smallest attainable p-value falls to 0.05, so an animal below that is
    reported descriptively and is not given a p-value it cannot support.
    """
    by_animal: dict[str, list[str]] = {}
    for s in included:
        by_animal.setdefault(census[s]["animal"], []).append(s)
    out = {}
    for animal, ids in sorted(by_animal.items()):
        values = [rows[s]["raw_outcome_vs_observable"]["r"] for s in ids]
        entry = {
            "n_sessions": len(ids),
            "sessions": ids,
            "per_session_r": dict(zip(ids, values)),
            "median_r": float(np.median(values)),
            "mean_r": float(np.mean(values)),
            "error_rate_range": [float(min(census[s]["error_rate"] for s in ids)),
                                 float(max(census[s]["error_rate"] for s in ids))],
        }
        entry["pooled"] = (slope_across_sessions_test(values, alternative="two-sided") if len(ids) >= 4
                           else {"status": "reported_descriptively_only",
                                 "reason": "fewer than 4 sessions, below the smallest session count at "
                                           "which the pooled sign-flip test can reach p = 0.05"})
        out[animal] = entry
    return out


def _classify_amplitude(pooled: dict, mdd_by_key: dict) -> dict:
    """Applies the amplitude decision rule: the orthogonality gate, not the
    analyst, picks which correlation decides, and the detection floor is
    measured on whichever correlation that is."""
    gate = pooled["orthogonality_gate"]
    if gate.get("status") != "tested":
        return {"branch": "not_computable", "reason": "orthogonality gate not computable"}
    gate_significant = bool(gate.get("significant"))
    deciding_key = "partial_controlling_spike_count" if gate_significant else "raw_outcome_vs_observable"
    deciding = pooled[deciding_key]
    if deciding.get("status") != "tested":
        return {"branch": "not_computable", "reason": f"{deciding_key} not computable"}
    mdd = mdd_by_key.get(deciding_key, {})
    mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
    if deciding.get("significant"):
        branch = "dominant_latent_amplitude_predicts_outcome"
    elif mdd_value is not None and mdd_value < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        branch = "dominant_latent_amplitude_does_not_predict_outcome"
    else:
        branch = "inconclusive_below_detection_floor"
    return {
        "branch": branch,
        "amplitude_is_separable_from_total_spike_count": not gate_significant,
        "orthogonality_gate_r": gate.get("mean_value"),
        "orthogonality_gate_p_value": gate.get("two_sided_p_value", gate.get("p_value")),
        "deciding_statistic": deciding_key,
        "deciding_statistic_r": deciding.get("mean_value"),
        "deciding_statistic_p_value": deciding.get("two_sided_p_value", deciding.get("p_value")),
        "deciding_statistic_ci": [deciding.get("ci_lower"), deciding.get("ci_upper")],
        "null_reference": 0.0,
        "minimum_detectable_paired_difference_at_80pct_power": mdd_value,
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    }


# ============================================================================
# Driver
# ============================================================================

def _flush(output: dict) -> None:
    """Write the artifact as it stands, so a kill costs the run and not the results.

    Values are made strict-JSON safe before encoding rather than being handed
    to the encoder raw: several of the reused estimators return whole
    permutation null distributions as arrays, which no per-object fallback
    can convert to a single number.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))


_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    """Model fits an earlier invocation of this module recorded as complete.

    A checkpoint file that cannot be parsed, or an entry without its
    completion flag set, is treated as absent: recomputing a fit is always
    correct, and carrying forward a half-written record is not.
    """
    try:
        entries = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: {**entry, "value": restore_checkpoint(entry["value"])} for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("complete") is True}


def _fit(key: str, compute) -> dict:
    """Return one model fit, computing it only if no earlier invocation of
    this module already finished it.

    The completion flag is written only after ``compute`` returns, so a fit
    interrupted part-way is absent from the record rather than half-present.
    Every fit is seeded from a descriptive tag rather than from call order,
    so a fit served from the record is identical to one recomputed here.
    """
    global _FITS_SERVED_FROM_CHECKPOINT, _FITS_COMPUTED_HERE
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        _FITS_SERVED_FROM_CHECKPOINT += 1
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    _FITS_COMPUTED_HERE += 1
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Replacing a fully written temporary file is atomic, so a kill during the
    # write cannot truncate the record of the fits that already finished.
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(checkpoint_safe(_COMPLETED_FITS), allow_nan=False, default=float))
    scratch.replace(CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()
    paths = _session_paths(root)

    output: dict = {
        "version": "2026-09-01",
        "scope": {
            "corpus": "macaque lPFC, Panichello et al. 2024 (Nature 636:422-429, doi:10.1038/s41586-024-08139-9)",
            "n_sessions_on_disk": len(paths),
            "delay_epoch_binning": "300-1450 ms after cue onset, 100 ms bins, the corpus's delay window "
                                   "everywhere else in this project",
            "animal_assignment": ANIMAL_ASSIGNMENT_SOURCE,
            "date_block_to_animal": DATE_BLOCK_TO_ANIMAL,
            "history_lags_trials": list(HISTORY_LAGS),
            "n_circular_shift_replicates": N_CIRCULAR_SHIFT_REPLICATES,
            "min_circular_shift_offset_trials": MIN_CIRCULAR_SHIFT_OFFSET,
            "error_trial_floors": list(ERROR_FLOORS),
            "primary_error_trial_floor": PRIMARY_ERROR_FLOOR,
            "n_permutations_per_correlation": N_PERM,
            "n_unit_count_matched_draws": N_UNIT_MATCHED_DRAWS,
            "seed_namespace": SEED_NAMESPACE,
            "seed_scheme": "every generator is seeded from a crc32 of a descriptive tag, so every "
                           "number here is reproducible from the tag alone",
            "candidate_identities_already_excluded_for_the_dominant_latent": {
                "source": "results/state_latent_identity.json",
                "total_spike_count": {"pooled_rho": -0.151, "p_value": 0.358},
                "trial_index": {"pooled_rho": -0.264, "p_value": 0.099},
                "current_item_label": {"n_sessions_significant": 6, "n_sessions": 25,
                                       "median_macro_one_vs_rest_auc": 0.5005},
                "per_trial_gain": {
                    "verdict": "state_is_a_per_trial_gain",
                    "median_observed_rank1_share": 0.8950377,
                    "median_matched_null_reference_share": 0.8925854,
                    "mean_paired_difference": 0.00418,
                    "mean_paired_difference_ci": [0.00071, 0.00766],
                    "caveat": "The margin is 0.24 percentage points and the pooled share averages two "
                              "very different recording regimes -- median 0.6557 where 56 to 160 units "
                              "were recorded, 0.9768 where 183 to 779 were -- so the share's magnitude "
                              "is a property of population size. This verdict is not quotable without "
                              "both numbers beside it.",
                },
            },
        },
        "history_decision_rule_declared_before_fitting": HISTORY_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "history_lag_profile_null_declared_before_fitting": HISTORY_LAG_PROFILE_NULL_DECLARED_BEFORE_FITTING,
        "behaviour_floor_rule_declared_before_fitting": BEHAVIOUR_FLOOR_RULE_DECLARED_BEFORE_FITTING,
        "error_rate_dose_response_rule_declared_before_fitting": ERROR_RATE_DOSE_RESPONSE_RULE_DECLARED_BEFORE_FITTING,
        "amplitude_decision_rule_declared_before_fitting": AMPLITUDE_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    # ---- per-session census -------------------------------------------------
    sessions: dict[str, dict] = {}
    census: dict[str, dict] = {}
    for path in paths:
        session = _load_session(path)
        sessions[session["session"]] = session
        census[session["session"]] = {
            k: v for k, v in session.items() if k not in ("counts", "is_corr", "cue_idx", "cue_ang")}
        census[session["session"]]["error_rate"] = session["n_error"] / session["n_trials"]
        _log(f"loaded {session['session']} trials={session['n_trials']} units={session['n_units']} "
             f"errors={session['n_error']} elapsed={time.time() - t0:.1f}s")

    session_ids = sorted(census)
    output["per_session_outcome_census"] = {
        "n_sessions_seen": len(session_ids),
        "sessions": [census[s] for s in session_ids],
    }
    _flush(output)

    # ---- why the third animal contributes nothing ---------------------------
    by_animal: dict[str, list[str]] = {}
    for s in session_ids:
        by_animal.setdefault(census[s]["animal"], []).append(s)
    coding_uniform = len({tuple(census[s]["outcome_field_distinct_values"]) for s in session_ids}) == 1
    all_lengths_match = all(census[s]["outcome_field_length_matches_trials"] for s in session_ids)
    no_non_finite = all(census[s]["outcome_field_n_non_finite"] == 0 for s in session_ids)
    max_error_off_primary = {
        animal: max(census[s]["n_error"] for s in ids) for animal, ids in sorted(by_animal.items())
    }
    # Pre-declared: the floor exclusion is attributed to task accuracy only if the outcome field itself
    # is sound in every session and the loader dropped none. Otherwise the exclusion is a data or
    # loading problem and must be reported as one, not as a fact about the animal.
    outcome_field_is_sound = bool(
        coding_uniform and all_lengths_match and no_non_finite and len(session_ids) == len(paths))
    output["why_sessions_fall_below_the_error_floor"] = {
        "explanation": ("sessions_fall_below_the_floor_because_the_animal_made_few_errors"
                        if outcome_field_is_sound else
                        "outcome_field_or_loader_problem_rather_than_task_accuracy"),
        "outcome_field_is_present_complete_and_identically_coded_in_every_session": outcome_field_is_sound,
        "outcome_field_coding_identical_across_all_sessions": coding_uniform,
        "outcome_field_distinct_values_per_animal": {
            animal: sorted({tuple(census[s]["outcome_field_distinct_values"]) for s in ids})[0]
            for animal, ids in sorted(by_animal.items())
        },
        "outcome_field_length_matches_trial_count_in_every_session": all_lengths_match,
        "outcome_field_has_no_non_finite_entries_in_any_session": no_non_finite,
        "n_sessions_read_successfully": len(session_ids),
        "n_sessions_dropped_by_the_loader": len(paths) - len(session_ids),
        "per_animal": {
            animal: {
                "n_sessions": len(ids),
                "n_trials_range": [min(census[s]["n_trials"] for s in ids), max(census[s]["n_trials"] for s in ids)],
                "n_units_range": [min(census[s]["n_units"] for s in ids), max(census[s]["n_units"] for s in ids)],
                "n_error_range": [min(census[s]["n_error"] for s in ids), max(census[s]["n_error"] for s in ids)],
                "accuracy_range": [
                    round(min(census[s]["n_correct"] / census[s]["n_trials"] for s in ids), 4),
                    round(max(census[s]["n_correct"] / census[s]["n_trials"] for s in ids), 4),
                ],
                "max_error_count": max_error_off_primary[animal],
                "reaches_lowest_floor_tested": max_error_off_primary[animal] >= min(ERROR_FLOORS),
            }
            for animal, ids in sorted(by_animal.items())
        },
    }
    _flush(output)

    # ---- gate: reproduce the corpus's published current-item rank ------------
    current: dict[str, dict] = {}
    for s in session_ids:
        session = sessions[s]
        correct = session["is_corr"]
        current[s] = _fit(f"current_item|{s}", lambda: _rank_for_labels(
            session["counts"][correct], session["cue_idx"][correct],
            _stable_seed(CORPUS_KEY, s, "pooled"), with_permutation_nulls=True))
        _log(f"current-item {s} rank={current[s].get('leading_latent_fractional_rank')} "
             f"auc={current[s].get('a_full')} elapsed={time.time() - t0:.1f}s")

    current_tested = [s for s in session_ids if current[s].get("status") == "tested"]
    current_ranks = [current[s]["leading_latent_fractional_rank"] for s in current_tested]
    current_pooled = _pool_ranks(current_ranks)
    observed_mean = current_pooled.get("mean_value")
    difference = (abs(observed_mean - EXPECTED_POOLED_CURRENT_ITEM_FRACTIONAL_RANK)
                  if observed_mean is not None else None)
    if difference is None or len(current_tested) != EXPECTED_N_SESSIONS:
        gate_status = "not_reproduced"
    elif difference == 0.0:
        gate_status = "reproduced_exactly"
    elif difference <= GATE_TOLERANCE:
        gate_status = "reproduced_within_tolerance"
    else:
        gate_status = "not_reproduced"

    output["current_item_reproduction_gate"] = {
        "status": gate_status,
        "expected_pooled_fractional_rank": EXPECTED_POOLED_CURRENT_ITEM_FRACTIONAL_RANK,
        "observed_pooled_fractional_rank": observed_mean,
        "absolute_difference": difference,
        "tolerance": GATE_TOLERANCE,
        "expected_n_sessions": EXPECTED_N_SESSIONS,
        "observed_n_sessions": len(current_tested),
        "pooled_test": current_pooled,
        "per_session": {
            s: {k: current[s].get(k) for k in (
                "status", "n_trials", "n_units", "n_classes", "k_latents", "a_full", "a_full_p_value",
                "a_full_clears_own_null", "leading_latent_rank_from_top", "leading_latent_fractional_rank")}
            for s in session_ids
        },
    }
    _flush(output)
    _log(f"gate {gate_status}: observed {observed_mean} vs expected "
         f"{EXPECTED_POOLED_CURRENT_ITEM_FRACTIONAL_RANK} over {len(current_tested)} sessions")

    # ---- is the dominant latent carrying the previous trial? ----------------
    # The behavioural and amplitude blocks below do not depend on this gate and run either way; only
    # the history block is downstream of it.
    if gate_status == "not_reproduced":
        output["previous_item_in_dominant_latent"] = {
            "status": "not_run_because_the_current_item_value_did_not_reproduce",
            "reason": "a previous-item fractional rank is uninterpretable when the current-item "
                      "fractional rank the same estimator produces does not reproduce",
        }
        output["history_lag_profile"] = {"status": "not_run_because_the_current_item_value_did_not_reproduce"}
        _flush(output)
    else:
        previous: dict[str, dict] = {}
        boundary_drops: dict[str, int] = {}
        for s in session_ids:
            session = sessions[s]
            correct = session["is_corr"]
            labels, defined = history_labels(session["cue_idx"], PRIMARY_HISTORY_LAG)
            keep = correct & defined
            boundary_drops[s] = int((correct & ~defined).sum())
            previous[s] = _fit(f"item_at_trial_minus_{PRIMARY_HISTORY_LAG}|{s}", lambda: _rank_for_labels(
                session["counts"][keep], labels[keep],
                _stable_seed(CORPUS_KEY, s, f"item_at_trial_minus_{PRIMARY_HISTORY_LAG}"),
                with_permutation_nulls=True))
            previous[s]["n_trials_dropped_at_session_start"] = boundary_drops[s]
            _log(f"previous-item {s} rank={previous[s].get('leading_latent_fractional_rank')} "
                 f"auc={previous[s].get('a_full')} elapsed={time.time() - t0:.1f}s")

        previous_tested = [s for s in session_ids if previous[s].get("status") == "tested"]
        previous_ranks = [previous[s]["leading_latent_fractional_rank"] for s in previous_tested]
        previous_pooled = _pool_ranks(previous_ranks)
        previous_mdd = (minimum_detectable_paired_difference(previous_ranks)
                        if len(previous_ranks) >= 2 else {"status": "not_computable"})
        median_k = int(np.median([previous[s]["k_latents"] for s in previous_tested])) if previous_tested else 0
        rank_step = 1.0 / (median_k - 1) if median_k > 1 else None

        previous_significant = bool(previous_pooled.get("significant_two_sided"))
        previous_mean = previous_pooled.get("mean_value")
        current_stays_above = observed_mean is not None and observed_mean > 0.5
        mdd_value = previous_mdd.get("mdd") if previous_mdd.get("status") == "computed" else None
        if previous_significant and previous_mean is not None and previous_mean < 0.5 and current_stays_above:
            history_branch = "rank_inverts_previous_item_in_dominant_latent"
        elif previous_significant and previous_mean is not None and previous_mean > 0.5:
            history_branch = "previous_item_rank_significantly_above_the_null_off_branch_list"
        elif not previous_significant and rank_step is not None and mdd_value is not None and mdd_value < rank_step:
            history_branch = "no_inversion_previous_item_not_in_dominant_latent"
        else:
            history_branch = "inconclusive_below_detection_floor"

        paired_shared = [s for s in previous_tested if s in current_tested]
        output["previous_item_in_dominant_latent"] = {
            "status": "tested",
            "branch": history_branch,
            "lag_trials": PRIMARY_HISTORY_LAG,
            "pooled_previous_item_fractional_rank": previous_pooled,
            "pooled_current_item_fractional_rank": current_pooled,
            "paired_previous_minus_current_fractional_rank": _paired(
                [previous[s]["leading_latent_fractional_rank"] for s in paired_shared],
                [current[s]["leading_latent_fractional_rank"] for s in paired_shared]),
            "minimum_detectable_paired_difference_at_80pct_power": previous_mdd,
            "one_rank_step_at_median_k": rank_step,
            "median_k_latents": median_k,
            "previous_item_decodability": {
                "median_macro_one_vs_rest_auc": float(np.median([previous[s]["a_full"] for s in previous_tested]))
                if previous_tested else None,
                "n_sessions_clearing_own_label_permutation_null": sum(
                    1 for s in previous_tested if previous[s].get("a_full_clears_own_null")),
                "n_sessions_tested": len(previous_tested),
                "current_item_median_macro_one_vs_rest_auc": float(
                    np.median([current[s]["a_full"] for s in current_tested])) if current_tested else None,
                "current_item_n_sessions_clearing_own_label_permutation_null": sum(
                    1 for s in current_tested if current[s].get("a_full_clears_own_null")),
            },
            "trial_pairs_excluded_at_session_boundaries": {
                "n_total": int(sum(boundary_drops.values())),
                "per_session": boundary_drops,
                "note": "No block field exists in any deposited file -- the five deposited variables are "
                        "the raster, its time axis, the cue angle, the cue angle index and the outcome "
                        "flag -- so the only boundary that can be excluded is the session boundary, and "
                        "a within-session block boundary, if the experiment had any, cannot be seen. "
                        "The count above is the number of correct trials that lost their predecessor "
                        "because it fell before the start of the recording.",
            },
            "reachability": (
                "The pooled fractional rank reaches two-sided significance at this session count and "
                "this per-session resolution: the current-item arm, on the same 25 sessions with the "
                f"same estimator, lands at {observed_mean} with two-sided p="
                f"{current_pooled.get('two_sided_p_value')}. A previous-item inversion of comparable "
                "size is therefore detectable, and a non-significant previous-item result is a null "
                "rather than an unreachable statistic."
            ),
            "per_session": {
                s: {k: previous[s].get(k) for k in (
                    "status", "n_trials", "n_units", "n_classes", "k_latents", "a_full", "a_full_p_value",
                    "a_full_clears_own_null", "leading_latent_rank_from_top", "leading_latent_fractional_rank",
                    "n_trials_dropped_at_session_start")}
                for s in session_ids
            },
        }
        _flush(output)
        _log(f"history branch: {history_branch} (previous-item pooled {previous_mean})")

        # ---- history lag profile against a circular-shift null --------------
        observed_by_lag: dict[int, dict[str, list[float]]] = {lag: {"rank": [], "auc": []} for lag in HISTORY_LAGS}
        per_session_by_lag: dict[str, dict[int, dict]] = {s: {} for s in session_ids}
        for s in session_ids:
            session = sessions[s]
            correct = session["is_corr"]
            for lag in HISTORY_LAGS:
                if lag == PRIMARY_HISTORY_LAG:
                    row = previous[s]
                else:
                    labels, defined = history_labels(session["cue_idx"], lag)
                    keep = correct & defined
                    row = _fit(f"item_at_trial_minus_{lag}|{s}", lambda: _rank_for_labels(
                        session["counts"][keep], labels[keep],
                        _stable_seed(CORPUS_KEY, s, f"item_at_trial_minus_{lag}"),
                        with_permutation_nulls=False))
                    row["n_trials_dropped_at_session_start"] = int((correct & ~defined).sum())
                per_session_by_lag[s][lag] = {
                    k: row.get(k) for k in ("status", "n_trials", "k_latents", "a_full",
                                            "leading_latent_fractional_rank", "n_trials_dropped_at_session_start")}
                if row.get("status") == "tested":
                    observed_by_lag[lag]["rank"].append(row["leading_latent_fractional_rank"])
                    observed_by_lag[lag]["auc"].append(row["a_full"])
            _log(f"lag profile {s} done elapsed={time.time() - t0:.1f}s")
            output.setdefault("history_lag_profile", {})["per_session_partial"] = per_session_by_lag
            _flush(output)

        null_pooled_rank: list[float] = []
        null_pooled_auc: list[float] = []
        for replicate in range(N_CIRCULAR_SHIFT_REPLICATES):
            ranks, aucs = [], []
            for s in session_ids:
                session = sessions[s]
                correct = session["is_corr"]
                n_all = session["n_trials"]
                rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|shift|{s}|{replicate}"))
                offset = int(rng.integers(MIN_CIRCULAR_SHIFT_OFFSET, n_all - MIN_CIRCULAR_SHIFT_OFFSET + 1))
                labels = circular_shift_labels(session["cue_idx"], offset)
                row = _fit(f"label_circular_shift_{replicate}|{s}", lambda: _rank_for_labels(
                    session["counts"][correct], labels[correct],
                    _stable_seed(CORPUS_KEY, s, f"label_circular_shift_{replicate}"),
                    with_permutation_nulls=False))
                if row.get("status") == "tested":
                    ranks.append(row["leading_latent_fractional_rank"])
                    aucs.append(row["a_full"])
            if ranks:
                null_pooled_rank.append(float(np.mean(ranks)))
                null_pooled_auc.append(float(np.mean(aucs)))
            _log(f"shift null replicate {replicate + 1}/{N_CIRCULAR_SHIFT_REPLICATES} "
                 f"pooled_rank={null_pooled_rank[-1] if null_pooled_rank else None} "
                 f"elapsed={time.time() - t0:.1f}s")
            output["history_lag_profile"] = {
                **output.get("history_lag_profile", {}),
                "circular_shift_null_partial": {
                    "n_replicates_done": len(null_pooled_rank),
                    "pooled_fractional_rank": null_pooled_rank,
                    "pooled_macro_auc": null_pooled_auc,
                },
            }
            _flush(output)

        def _null_two_sided_p(observed: float, null_values: list[float]) -> float | None:
            if not null_values:
                return None
            centre = float(np.median(null_values))
            extreme = sum(1 for v in null_values if abs(v - centre) >= abs(observed - centre))
            return (extreme + 1) / (len(null_values) + 1)

        by_lag = {}
        for lag in HISTORY_LAGS:
            rank_values, auc_values = observed_by_lag[lag]["rank"], observed_by_lag[lag]["auc"]
            pooled_rank = float(np.mean(rank_values)) if rank_values else None
            pooled_auc = float(np.mean(auc_values)) if auc_values else None
            by_lag[str(lag)] = {
                "n_sessions": len(rank_values),
                "pooled_mean_fractional_rank": pooled_rank,
                "pooled_mean_macro_one_vs_rest_auc": pooled_auc,
                "fractional_rank_vs_null_two_sided_p": _null_two_sided_p(pooled_rank, null_pooled_rank)
                if pooled_rank is not None else None,
                "macro_auc_vs_null_two_sided_p": _null_two_sided_p(pooled_auc, null_pooled_auc)
                if pooled_auc is not None else None,
                "fractional_rank_test_against_half": _pool_ranks(rank_values) if rank_values else {"status": "not_computed"},
            }

        auc_by_lag = [by_lag[str(lag)]["pooled_mean_macro_one_vs_rest_auc"] for lag in HISTORY_LAGS]
        decays = all(a is not None for a in auc_by_lag) and auc_by_lag[0] == max(auc_by_lag)
        output["history_lag_profile"] = {
            "status": "tested",
            "lags_trials": list(HISTORY_LAGS),
            "by_lag": by_lag,
            "circular_shift_null": {
                "n_replicates": len(null_pooled_rank),
                "min_attainable_two_sided_p": 1.0 / (len(null_pooled_rank) + 1) if null_pooled_rank else None,
                "pooled_fractional_rank_median": float(np.median(null_pooled_rank)) if null_pooled_rank else None,
                "pooled_fractional_rank_range": [float(np.min(null_pooled_rank)), float(np.max(null_pooled_rank))]
                if null_pooled_rank else None,
                "pooled_macro_auc_median": float(np.median(null_pooled_auc)) if null_pooled_auc else None,
                "pooled_macro_auc_range": [float(np.min(null_pooled_auc)), float(np.max(null_pooled_auc))]
                if null_pooled_auc else None,
                "pooled_fractional_rank_replicates": null_pooled_rank,
                "pooled_macro_auc_replicates": null_pooled_auc,
            },
            "decoding_accuracy_is_highest_at_the_shortest_lag": bool(decays),
            "per_session": per_session_by_lag,
        }
        _flush(output)

    # ---- behavioural breadth across error floors ----------------------------
    lowest_floor = min(ERROR_FLOORS)
    floor_sessions = [s for s in session_ids if census[s]["n_error"] >= lowest_floor]
    behaviour_rows: dict[str, dict] = {}
    amplitude_rows: dict[str, dict] = {}
    behaviour_excluded: dict[str, str] = {}
    for s in floor_sessions:
        arrays = _behaviour_session_arrays(sessions[s])
        if arrays is None:
            behaviour_excluded[s] = "no_observable_computable_for_this_session"
            continue
        behaviour_rows[s] = {
            "n_trials_total": arrays["n_trials_total"],
            "n_trials_with_defined_direction": arrays["n_trials_with_defined_direction"],
            **_correlation_family(arrays, "deviation", s),
        }
        amplitude_rows[s] = {
            "n_trials_total": arrays["n_trials_total"],
            **_correlation_family(arrays, "amplitude", s),
        }
        _log(f"behaviour {s} raw_deviation_r={behaviour_rows[s]['raw_outcome_vs_observable']['r']:.4f} "
             f"raw_amplitude_r={amplitude_rows[s]['raw_outcome_vs_observable']['r']:.4f} "
             f"elapsed={time.time() - t0:.1f}s")
        output["error_floor_sensitivity"] = {"status": "running", "per_session_state_geometry_partial": behaviour_rows}
        _flush(output)

    def _floor_block(floor: int, rows: dict[str, dict], classify_as: str) -> dict:
        included = [s for s in session_ids if census[s]["n_error"] >= floor and s in rows]
        per_session = [rows[s] for s in included]
        pooled = {key: _pool_correlations(per_session, key) for key in (
            "orthogonality_gate", "raw_outcome_vs_observable", "partial_controlling_spike_count",
            "partial_controlling_trial_index", "joint_partial_controlling_spike_count_and_trial_index")}
        mdd_by_key = {key: _mdd(per_session, key)
                      for key in ("raw_outcome_vs_observable", "partial_controlling_spike_count")}
        mdd = mdd_by_key["raw_outcome_vs_observable"]
        pooled["minimum_detectable_paired_difference_at_80pct_power"] = mdd_by_key
        block = {
            "error_trial_floor": floor,
            "role": "pre-declared primary" if floor == PRIMARY_ERROR_FLOOR else "sensitivity analysis",
            "n_sessions_included": len(included),
            "sessions_included": included,
            "sessions_excluded_below_floor": {
                s: census[s]["n_error"] for s in session_ids if census[s]["n_error"] < floor},
            "sessions_excluded_for_other_reasons": {
                s: behaviour_excluded[s] for s in behaviour_excluded if census[s]["n_error"] >= floor},
            "sessions_by_animal": _animal_counts(included, census),
            "sessions_by_date_block": {
                block_key: sum(1 for s in included if census[s]["date_block"] == block_key)
                for block_key in sorted(DATE_BLOCK_TO_ANIMAL)},
            "pooled": pooled,
        }
        if classify_as == "state_geometry":
            block["branch"] = _classify(
                pooled["orthogonality_gate"], pooled["raw_outcome_vs_observable"],
                pooled["joint_partial_controlling_spike_count_and_trial_index"],
                mdd.get("mdd") if mdd.get("status") == "computed" else None)
            raw = pooled["raw_outcome_vs_observable"]
            block["effect_size"] = {
                "pooled_r": raw.get("mean_value"), "p_value": raw.get("two_sided_p_value", raw.get("p_value")),
                "ci": [raw.get("ci_lower"), raw.get("ci_upper")],
                "null_reference": 0.0,
                "correlation_is_with_correct_equals_one": True,
            }
        else:
            block.update(_classify_amplitude(pooled, mdd_by_key))
        return block

    geometry_floors = {str(floor): _floor_block(floor, behaviour_rows, "state_geometry") for floor in ERROR_FLOORS}
    primary_branch = geometry_floors[str(PRIMARY_ERROR_FLOOR)]["branch"]
    sensitivity_branches = {f: geometry_floors[f]["branch"] for f in geometry_floors if f != str(PRIMARY_ERROR_FLOOR)}
    floors_agree = all(b == primary_branch for b in sensitivity_branches.values())
    output["error_floor_sensitivity"] = {
        "status": "tested",
        "observable": "rate-free state deviation (direction of the per-trial population activity vector "
                      "against the session's own leave-one-out mean direction)",
        "n_sessions_seen": len(session_ids),
        "n_sessions_reaching_the_lowest_floor_tested": len(floor_sessions),
        "n_sessions_with_the_observable_computed": len(behaviour_rows),
        "sessions_excluded_by_reason": {
            **{s: f"n_error {census[s]['n_error']} below the lowest floor tested ({lowest_floor})"
               for s in session_ids if census[s]["n_error"] < lowest_floor},
            **behaviour_excluded,
        },
        "by_floor": geometry_floors,
        "floors_agree_on_the_branch": floors_agree,
        "disagreement_between_floors_is_itself_the_finding": not floors_agree,
        "per_session": behaviour_rows,
    }
    _flush(output)

    # ---- does the coupling depend on the performance regime? ----------------
    primary_included = geometry_floors[str(PRIMARY_ERROR_FLOOR)]["sessions_included"]
    max_error_by_animal = {
        animal: max(census[s]["n_error"] for s in ids) for animal, ids in sorted(by_animal.items())}
    animals_unreachable_at_every_floor = sorted(
        animal for animal, max_error in max_error_by_animal.items() if max_error < min(ERROR_FLOORS))
    output["performance_regime_dependence"] = {
        "status": "tested",
        "error_trial_floor": PRIMARY_ERROR_FLOOR,
        "dose_response": _error_rate_dose_response(
            primary_included, behaviour_rows, census,
            geometry_floors[str(PRIMARY_ERROR_FLOOR)]["pooled"]["raw_outcome_vs_observable"]),
        "by_animal": _per_animal_effect(primary_included, behaviour_rows, census),
        "by_animal_at_every_floor": {
            str(floor): geometry_floors[str(floor)]["sessions_by_animal"] for floor in ERROR_FLOORS},
        "max_error_trial_count_per_animal_whole_corpus": max_error_by_animal,
        "animals_contributing_nothing_at_every_floor_tested": animals_unreachable_at_every_floor,
        "no_floor_can_rebalance_the_animals": bool(animals_unreachable_at_every_floor),
        "why_a_lower_floor_would_not_have_fixed_the_imbalance": (
            "An error-trial floor selects sessions by how often the animal failed, so it cannot sample "
            "the three animals evenly at any setting. The listed animals never reach even the lowest "
            "floor tested in any of their sessions, so lowering the floor recruits sessions only from "
            "the animals that were already contributing and shifts the balance further toward them. A "
            "reader must not assume a lower floor would have restored a balanced design."
        ),
        "reachability": (
            "The slope of the per-session coefficient on the session's own error rate is estimated from "
            f"{len(primary_included)} sessions. Whether it could have detected a dose-response large "
            "enough to matter is answered by comparing its confidence-interval half-width against the "
            "slope that would carry the entire pooled correlation across the observed error-rate span; "
            "both numbers are reported in the dose-response block, and the branch that fires depends on "
            "that comparison rather than on the p-value alone."
        ),
        "correct_versus_error_designs_buy_power_from_task_failure": (
            "A correct-versus-error contrast needs enough errors to estimate the error class, so its "
            "power comes from the subject failing. In a subject that performs well it is not merely "
            "underpowered, it is unrunnable: the sessions with the largest populations and the cleanest "
            "recordings in this corpus contribute nothing to it precisely because the animal was "
            "accurate. That is a measurable observability limit on behavioural-neural correlates, not "
            "an incidental exclusion, and it applies to any design that conditions on failure."
        ),
    }
    _flush(output)

    # ---- unit count against the behavioural correlation ---------------------
    unit_counts = [float(census[s]["n_units"]) for s in primary_included]
    session_correlations = [behaviour_rows[s]["raw_outcome_vs_observable"]["r"] for s in primary_included]
    rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|unit_count_vs_correlation"))
    unit_vs_r = spearman_permutation_test(np.asarray(unit_counts), np.asarray(session_correlations),
                                          n_perm=N_PERM, rng=rng)

    block_unit_ranges = {}
    for block_key in sorted(DATE_BLOCK_TO_ANIMAL):
        ids = [s for s in session_ids if census[s]["date_block"] == block_key]
        block_unit_ranges[DATE_BLOCK_TO_ANIMAL[block_key]] = [
            min(census[s]["n_units"] for s in ids), max(census[s]["n_units"] for s in ids)]
    ranges = sorted(block_unit_ranges.values())
    ranges_overlap = any(ranges[i][1] >= ranges[i + 1][0] for i in range(len(ranges) - 1))

    common_unit_count = int(min(census[s]["n_units"] for s in primary_included))
    matched_rows: dict[str, dict] = {}
    for s in primary_included:
        session = sessions[s]
        draw_r = []
        for draw in range(N_UNIT_MATCHED_DRAWS):
            draw_rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|unit_matched|{s}|{draw}"))
            unit_idx = draw_rng.choice(session["n_units"], size=common_unit_count, replace=False)
            activity = session["counts"][:, unit_idx, :].sum(axis=2)
            deviation = rate_free_state_deviation(activity)
            finite = np.isfinite(deviation)
            if finite.sum() < 16:
                continue
            result = _corr(session["is_corr"][finite].astype(float), deviation[finite], [],
                           f"unit_matched|{s}|{draw}")
            if result.get("status") == "computed":
                draw_r.append(result["r"])
        if draw_r:
            matched_rows[s] = {"n_draws": len(draw_r), "median_r": float(np.median(draw_r)),
                               "r_all_draws": draw_r}
        _log(f"unit-count matched {s} median_r={matched_rows.get(s, {}).get('median_r')} "
             f"elapsed={time.time() - t0:.1f}s")

    matched_values = [row["median_r"] for row in matched_rows.values()]
    matched_pooled = slope_across_sessions_test(matched_values, alternative="two-sided") if matched_values else {
        "status": "not_computed"}
    unmatched = geometry_floors[str(PRIMARY_ERROR_FLOOR)]["pooled"]["raw_outcome_vs_observable"]
    output["unit_count_and_the_behavioural_correlation"] = {
        "status": "tested",
        "error_trial_floor": PRIMARY_ERROR_FLOOR,
        "per_session": {s: {"n_units": census[s]["n_units"], "animal": census[s]["animal"],
                            "raw_r": behaviour_rows[s]["raw_outcome_vs_observable"]["r"]}
                        for s in primary_included},
        "unit_count_range_across_included_sessions": [int(min(unit_counts)), int(max(unit_counts))],
        "unit_count_vs_per_session_correlation": unit_vs_r,
        "per_animal_unit_count_range_whole_corpus": block_unit_ranges,
        "per_animal_unit_count_ranges_overlap": ranges_overlap,
        "session_level_matching_is_possible": ranges_overlap,
        "why_matching_is_done_at_the_unit_level": (
            "Unit count and animal are confounded at the session level: the three animals' unit-count "
            "ranges do not overlap, so no subset of sessions can be balanced on unit count while "
            "keeping more than one animal. Matching is therefore done inside each session instead, by "
            "subsampling every included session's units down to the smallest included session's unit "
            "count and recomputing the observable and its correlation with outcome on the subsample. "
            "This equalises the estimation noise that unit count drives without requiring a "
            "session-level balance that cannot exist."
        ),
        "matched_unit_count": common_unit_count,
        "n_draws_per_session": N_UNIT_MATCHED_DRAWS,
        "matched_per_session": matched_rows,
        "matched_pooled_raw_correlation": matched_pooled,
        "unmatched_pooled_raw_correlation": unmatched,
        "matched_minus_unmatched_pooled_r": (
            matched_pooled.get("mean_value") - unmatched.get("mean_value")
            if matched_pooled.get("mean_value") is not None and unmatched.get("mean_value") is not None else None),
    }
    _flush(output)

    # ---- does the dominant latent's own amplitude predict outcome? ----------
    amplitude_floors = {str(floor): _floor_block(floor, amplitude_rows, "amplitude") for floor in ERROR_FLOORS}
    output["dominant_latent_amplitude_and_outcome"] = {
        "status": "tested",
        "observable": "per-trial score of the leading component of the centred trial-by-window matrix, "
                      "the dominant latent's own per-trial amplitude",
        "by_floor": amplitude_floors,
        "branch_at_primary_floor": amplitude_floors[str(PRIMARY_ERROR_FLOOR)]["branch"],
        "floors_agree_on_the_branch": len({amplitude_floors[f]["branch"] for f in amplitude_floors}) == 1,
        "per_session": amplitude_rows,
    }
    _flush(output)

    # ---- are the history and amplitude effects the same effect? -------------
    history_positive = output.get("previous_item_in_dominant_latent", {}).get(
        "branch") == "rank_inverts_previous_item_in_dominant_latent"
    amplitude_positive = amplitude_floors[str(PRIMARY_ERROR_FLOOR)]["branch"] == "dominant_latent_amplitude_predicts_outcome"
    if history_positive and amplitude_positive:
        rows: dict[str, dict] = {}
        for s in primary_included:
            arrays = _behaviour_session_arrays(sessions[s])
            if arrays is None:
                continue
            # Only trials that have a predecessor inside the session can be conditioned on the previous
            # item, so the raw and the controlled correlation are computed on that same trial subset --
            # a partial that dropped trials the raw one kept would confound the control with the subset.
            keep = arrays["has_predecessor"]
            if keep.sum() < 16:
                continue
            controls = [arrays["previous_item_sin"][keep], arrays["previous_item_cos"][keep]]
            rows[s] = {
                "n_trials_with_a_predecessor": int(keep.sum()),
                "raw_outcome_vs_observable": _corr(
                    arrays["is_corr"][keep], arrays["amplitude"][keep], [], f"same_effect|raw|{s}"),
                "amplitude_vs_outcome_controlling_previous_item": _corr(
                    arrays["is_corr"][keep], arrays["amplitude"][keep], controls, f"same_effect|ctrl|{s}"),
            }
        per_session = list(rows.values())
        raw_pooled = _pool_correlations(per_session, "raw_outcome_vs_observable")
        controlled_pooled = _pool_correlations(per_session, "amplitude_vs_outcome_controlling_previous_item")
        output["are_the_history_and_amplitude_effects_the_same"] = {
            "status": "tested",
            "pooled_raw_on_the_same_trial_subset": raw_pooled,
            "pooled_partial_controlling_previous_item": controlled_pooled,
            "shrinkage_when_the_previous_item_is_controlled": (
                controlled_pooled.get("mean_value") - raw_pooled.get("mean_value")
                if controlled_pooled.get("mean_value") is not None
                and raw_pooled.get("mean_value") is not None else None),
            "reading": (
                "If the amplitude's outcome correlation survives conditioning on the previous item's "
                "cue angle at roughly its raw size, the two effects are separable. If it collapses "
                "toward zero, they are one effect: a history signal that also predicts outcome."
            ),
            "per_session": rows,
        }
    else:
        output["are_the_history_and_amplitude_effects_the_same"] = {
            "status": "not_applicable",
            "reason": "the partial correlation that would separate the two effects is only interpretable "
                      "if both fired positive; "
                      f"the previous-item test fired "
                      f"'{output.get('previous_item_in_dominant_latent', {}).get('branch')}' and the "
                      f"amplitude test fired '{amplitude_floors[str(PRIMARY_ERROR_FLOOR)]['branch']}'",
        }

    # ---- zero-drop accounting -----------------------------------------------
    history_block = output.get("previous_item_in_dominant_latent", {})
    history_per_session = history_block.get("per_session", {})
    history_tested = [s for s in session_ids if history_per_session.get(s, {}).get("status") == "tested"]
    output["zero_drop_accounting"] = {
        "n_session_files_on_disk": len(paths),
        "n_sessions_read_successfully": len(session_ids),
        "n_session_files_the_loader_could_not_read": len(paths) - len(session_ids),
        "arms": {
            "current_item_ablation_cost_rank": {
                "seen": len(session_ids), "tested": len(current_tested),
                "excluded_by_reason": {s: current[s].get("status") for s in session_ids if s not in current_tested},
            },
            "previous_item_ablation_cost_rank": {
                "seen": len(session_ids), "tested": len(history_tested),
                "excluded_by_reason": {
                    s: history_per_session.get(s, {}).get("status", history_block.get("status"))
                    for s in session_ids if s not in history_tested},
            },
            "behavioural_correlation_at_the_primary_floor": {
                "seen": len(session_ids), "tested": len(primary_included),
                "excluded_by_reason": {
                    **{s: f"error trial count {census[s]['n_error']} below the floor of {PRIMARY_ERROR_FLOOR}"
                       for s in session_ids if census[s]["n_error"] < PRIMARY_ERROR_FLOOR},
                    **{s: reason for s, reason in behaviour_excluded.items()},
                },
            },
        },
    }
    for arm in output["zero_drop_accounting"]["arms"].values():
        arm["counts_reconcile"] = arm["seen"] == arm["tested"] + len(arm["excluded_by_reason"])

    output["how_this_artifact_was_assembled"] = {
        "assembled_across_more_than_one_invocation": _FITS_SERVED_FROM_CHECKPOINT > 0,
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/dominant_latent_identity_and_behaviour_breadth_checkpoint.json",
        "explanation": (
            "Every ablation-cost fit -- the current-item fit, the previous-item fit, each history lag "
            "and each circular-shift null replicate, one per session -- is recorded as complete as it "
            "finishes, and an invocation that starts with fits already on record reads those back "
            "instead of refitting them. This makes no difference to any number reported here: each fit "
            "is seeded from a descriptive tag rather than from call order, so a fit read back is the "
            "same fit. The completion flag is written only after a fit returns, and an unparseable "
            "record is treated as absent, so an interrupted fit is recomputed rather than carried "
            "forward. Only fits are recorded; every pooled statistic, decision rule and branch on this "
            "page is recomputed from them in whichever invocation finished the run. The record holds "
            "fitted values alone and no encoding or file-format detail of this page can reach one."
        ),
    }

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "current_item_reproduction_gate": output["current_item_reproduction_gate"]["status"],
        "previous_item_branch": output.get("previous_item_in_dominant_latent", {}).get("branch"),
        "state_geometry_branch_by_floor": {f: geometry_floors[f]["branch"] for f in geometry_floors},
        "amplitude_branch_by_floor": {f: amplitude_floors[f]["branch"] for f in amplitude_floors},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
