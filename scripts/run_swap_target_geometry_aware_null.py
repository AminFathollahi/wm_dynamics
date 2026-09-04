"""run_swap_target_geometry_aware_null.py -- when the report lands on the
WRONG object (a swap), does it land on the uncued object nearer the
PRECEDING trial's remembered item -- tested against the task's own realised
permutation null rather than an assumed symmetry constant?

An earlier analysis of this question (``scripts/run_swap_target_and_
preceding_trial_item.py``, ``results/swap_target_and_preceding_trial_item.
json``) stopped at its own gate: its two-alternative statistics assumed a
null of exactly one half, and that premise is false in this task because
the two uncued objects sit a fixed ~120 degrees apart on effectively every
trial, which makes the two reference frames non-exchangeable. The measured
shuffle-null mean was 0.5038460317572688, offset from one half by z = 16.42
against a threshold of 3.0. That artifact stands as delivered.

This module re-runs the identical admission (the same surviving trials),
the identical per-session statistic definitions and the identical bias-only
and rate controls, but replaces the assumed constant with the EMPIRICAL
shuffle null of each statistic itself: the preceding trial's identity is
permuted across trials within session, every statistic is recomputed under
each permutation exactly as observed, and the observed value is judged by
its two-sided percentile position in that realised distribution. Null
quantiles are reported beside every observed value, so no verdict ever
rests on where zero or one half is ASSUMED to sit.

Because the reference distribution is now realised rather than assumed, the
premise that can still fail is different: the realised null itself must be
estimable stably enough for a percentile test to mean anything. Before any
verdict, each statistic's permutation draws are split into two halves and
the halves' means must agree within their own Monte-Carlo error (z <= 3).
If a realised null mis-centres against its own halves beyond that z, this
module reports that and stops -- a percentile test against an unstable null
is a finding about the estimator, not a result about behaviour.

The realised angular separation between the two uncued objects is reported
as a per-trial covariate alongside the headline statistic, so any reader of
this or the earlier artifact can see the geometric fact that broke the
exact-one-half assumption.

SCOPE. Multi-object macaque corpus only, item count 3 swaps with an
admissible immediately-preceding trial, session as the unit of analysis,
zero-drop accounting throughout. Per-session fits (swap geometry, content-
specific serial pull, deviation arms) are reused verbatim from the earlier
analysis's completed-fit checkpoint; nothing already delivered is modified,
re-run or re-labelled.

CONTROLS. Block B's bias-only control (leave-one-out session mean of the
serial pull substituted for each trial's own value) and its rate-versus-
bias distinction (total spike count substitution plus the spike-count
partial) carry over in percentile-test form: each control statistic is
judged against its own permutation null. The earlier analysis additionally
ran a session-mean-reference spatial-bias control beside block A; that
control is not part of this analysis's pre-declared control set and is not
recomputed here. A sensitivity re-run excluding trials whose uncued-object
separation is below 15 degrees is reported beside the primary and decides
nothing.
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
from provenance import _json_safe, git_commit  # noqa: E402
from run_dissociation_replication_and_counting_noise import _pool_cell  # noqa: E402
from run_swap_target_and_preceding_trial_item import (  # noqa: E402
    BRANCH_A_AVOID, BRANCH_A_BOTH_INSEPARABLE, BRANCH_A_INCONCLUSIVE,
    BRANCH_A_NOT_COVERED, BRANCH_A_POWERED_NULL, BRANCH_A_RESPONSE, BRANCH_A_TARGET,
    BRANCH_B_INCONCLUSIVE, BRANCH_B_NOT_COVERED, BRANCH_B_NOT_SEPARABLE,
    BRANCH_B_OPPOSITE, BRANCH_B_POSITIVE, BRANCH_B_POWERED_NULL,
    BLOCK_A_MDD_POWERED_NULL_THRESHOLD, BRANCH_GATE_FAILED, BRANCH_TOO_FEW_TRIALS,
    CHECKPOINT_PATH as SOURCE_CHECKPOINT_PATH, DROP_SESSION_ARRAYS, MIN_POOLED_ADMISSIBLE_SWAP_TRIALS,
    NEAR_SEPARATION_THRESHOLD_DEGREES, SURVIVING, _circular_distance_rad, _residualize,
    analyse_session,
)
from run_watters_state_geometry import PRIMARY_QUALITY_TIER  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import minimum_detectable_paired_difference, stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "swap_target_geometry_aware_null.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "swap_target_geometry_aware_null_checkpoint.json"
DISSOCIATION_ARTIFACT_PATH = ROOT / "results" / "dissociation_replication_and_counting_noise.json"
SERIAL_DEPENDENCE_ARTIFACT_PATH = ROOT / "results" / "deviation_serial_dependence_and_temporal_locus.json"

ANALYSIS_VERSION = "2026-08-26"
SEED_PREFIX = "swap_target_geometry_aware_null"
CORPUS_LABEL = "watters_2026_macaque_multi_object"
REPRODUCTION_TOLERANCE = 1e-6

# ---------------------------------------------------------------------------
# Pre-declared constants -- fixed before any number is read, never revisited.
# ---------------------------------------------------------------------------

N_SHUFFLE_DRAWS = 2000
PERCENTILE_ALPHA = 0.05  # two-sided percentile p strictly below this is significant
CENTRING_Z_THRESHOLD = 3.0  # split-half agreement of each realised null, in Monte-Carlo error units

STAT_TARGET = "target_referenced"
STAT_RESPONSE = "response_referenced"
STAT_PAIRED = "target_minus_response_direct_paired"
STAT_PULL_RAW = "serial_pull_group_difference"
STAT_RATE = "rate_control_group_difference"
STAT_BIAS = "bias_only_control_group_difference"
STAT_PARTIAL = "spike_count_partial_group_difference"
STATISTIC_NAMES = [STAT_TARGET, STAT_RESPONSE, STAT_PAIRED, STAT_PULL_RAW, STAT_RATE, STAT_BIAS, STAT_PARTIAL]

BRANCH_MISCENTRED_STOP = "empirical_null_still_miscentred_beyond_a_z_of_3"
BRANCH_TOP_FOLLOWS = "swap_destination_follows_the_preceding_trials_item"
BRANCH_TOP_DOES_NOT_FOLLOW = "swap_destination_does_not_follow_the_preceding_trials_item"
BRANCH_TOP_INCONCLUSIVE = "inconclusive_below_detection_floor"
BRANCH_TOP_NOT_SEPARABLE = "not_separable_from_a_session_level_offset"
BRANCH_TOP_NOT_COVERED = "outcome_not_covered_by_the_pre_declared_rule"

PERCENTILE_TEST_DECLARATION = (
    "For every statistic the reference distribution is the empirical shuffle null of that statistic "
    f"itself: {N_SHUFFLE_DRAWS} times, the preceding-trial identity is permuted across surviving trials "
    "within each session (one seeded permutation per session per draw, applied jointly to the preceding "
    "trial's target angle and response angle), every statistic is recomputed exactly as observed, and "
    "each draw's pooled value is the unweighted mean over the sessions contributing a defined value. "
    "The two-sided percentile p-value is 2*min((#draws<=observed)+1, (#draws>=observed)+1)/(n_draws+1), "
    f"capped at 1; 'significant' means p < {PERCENTILE_ALPHA}; 'above'/'below' is the sign of "
    "(observed - null mean). Null mean, sd, Monte-Carlo error of the mean and the 2.5/5/25/50/75/95/97.5 "
    "percentiles are reported beside every observed value."
)

CENTRING_CHECK_DECLARATION = (
    "A percentile test presupposes only that the realised null is estimated stably enough to define "
    "percentiles. Before any verdict, each statistic's pooled draws are split into an earlier half and a "
    "later half by draw index; z = |mean_earlier - mean_later| / sqrt(mc_error_earlier^2 + "
    f"mc_error_later^2). If ANY statistic's z exceeds {CENTRING_Z_THRESHOLD}, the leg reports that and "
    "stops before any branch fires: the realised null cannot be trusted at Monte-Carlo precision, which "
    "is a finding about the estimator. The offset of each realised null from one half is also recorded, "
    "descriptively only: under this analysis's design a null need not sit at one half, because the fixed "
    "~120-degree uncued separation is exactly what the realised null absorbs."
)

TOP_BRANCH_RULE_DECLARED_BEFORE_RUNNING = (
    "Checked in this order:\n"
    "  0. Reproduction gate fails -> stop. Fewer than the pre-declared floor of 200 pooled admissible "
    "swap trials -> stop.\n"
    "  1. Any realised null mis-centres against its own halves beyond z = 3 -> "
    "'empirical_null_still_miscentred_beyond_a_z_of_3'; report and stop before any block verdict.\n"
    "  2. Target-referenced statistic significant above its null centre -> "
    "'swap_destination_follows_the_preceding_trials_item'.\n"
    "  3. Target-referenced statistic significant BELOW its null centre -> "
    "'outcome_not_covered_by_the_pre_declared_rule': avoidance was not pre-declared at this level and no "
    "listed cell is stretched onto it (the block-level record carries the numbers).\n"
    "  4. Not significant, minimum detectable paired difference below 0.05 proportion units -> "
    "'swap_destination_does_not_follow_the_preceding_trials_item', naming that floor as its reference.\n"
    "  5. Otherwise 'inconclusive_below_detection_floor'.\n"
    "  6. Override applied only when step 2 did not fire: if the neural association is not separable "
    "from a session-level or rate confound (block B's own rule) -> "
    "'not_separable_from_a_session_level_offset', because then any apparent association is carried by "
    "the confound rather than by the preceding trial's item."
)

BLOCK_A_PERCENTILE_RULE_DECLARED_BEFORE_RUNNING = (
    "Mirrors the delivered two-alternative decision rule with every sign-flip-vs-one-half test replaced "
    "by the statistic's own two-sided percentile test. Checked in this order:\n"
    "  0. Either statistic significantly below its null centre -> 'swaps_avoid_the_preceding_trials_item'.\n"
    "  1. Target-referenced significantly above its null centre and response-referenced not -> "
    "'swaps_land_on_the_object_nearest_the_preceding_trials_remembered_item'.\n"
    "  2. Response-referenced significantly above and target-referenced not -> "
    "'swaps_repeat_the_preceding_trials_response_rather_than_its_remembered_item'.\n"
    "  3. Both significantly above: the direct paired difference (per-session target proportion minus "
    "response proportion, pooled) decides by its own percentile test -- significantly above its null -> "
    "rule 1's label; significantly below -> rule 2's label; otherwise -> "
    "'swaps_follow_the_preceding_trial_but_its_remembered_item_and_its_response_cannot_be_separated'.\n"
    "  4. Neither significant: both minimum detectable paired differences below 0.05 proportion units -> "
    "'powered_null_swap_destination_is_unrelated_to_the_preceding_trial'; otherwise -> "
    "'inconclusive_below_detection_floor'.\n"
    "Any pattern not covered records 'outcome_not_covered_by_the_pre_declared_rule' with every number."
)

BLOCK_B_PERCENTILE_RULE_DECLARED_BEFORE_RUNNING = (
    "Mirrors the delivered decision rule with the same substitution. Per session, the mean per-trial "
    "content-specific serial pull difference on previous-item swaps minus the same mean on other trials, "
    "pooled across sessions; raw, rate-control (total spike count substituted), bias-only-control "
    "(leave-one-out session mean substituted) and spike-count-partial (serial pull residualised on spike "
    "count within session) variants, each judged by its own percentile test. Checked in this order:\n"
    "  0. Bias-only control significant, OR the rate control significant and the spike-count partial not "
    "surviving (not significant with the same direction as raw) -> "
    "'serial_pull_group_difference_not_separable_from_a_session_level_or_rate_confound'; no cell below "
    "may fire.\n"
    "  1. Raw significant above its null centre -> "
    "'the_trials_own_serial_pull_predicts_that_its_swap_goes_to_the_preceding_trials_item'.\n"
    "  2. Raw significant below -> 'serial_pull_is_larger_on_swaps_that_avoid_the_preceding_trials_item'.\n"
    "  3. Raw not significant: minimum detectable paired difference below the delivered pooled "
    "content-specific serial-pull effect (read live from results/"
    "deviation_serial_dependence_and_temporal_locus.json) -> "
    "'powered_null_serial_pull_magnitude_does_not_determine_the_swap_destination'; otherwise -> "
    "'inconclusive_below_detection_floor'.\n"
    "Any pattern not covered records 'outcome_not_covered_by_the_pre_declared_rule' with every number."
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Checkpoint reuse -- the per-session fits of the delivered analysis are
# null-independent, so they are served verbatim; anything missing is computed
# once here through the original routine and recorded in this analysis's own
# checkpoint. The source checkpoint is opened read-only.
# ---------------------------------------------------------------------------

_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_SOURCE_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in (SOURCE_CHECKPOINT_PATH, CHECKPOINT_PATH):
        try:
            entries = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if isinstance(entry, dict) and entry.get("complete") is True:
                merged[key] = entry["value"]
    return merged


def _store_fit(key: str, value: dict) -> None:
    global _FITS_COMPUTED_HERE
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    _FITS_COMPUTED_HERE += 1
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        existing = {}
    existing[key] = {"complete": True, "value": value}
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(existing), allow_nan=False, default=float))
    os.replace(scratch, CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# Per-session arrays and statistics
# ---------------------------------------------------------------------------

def _build_session_arrays(swap_theta, other_theta, prev_target_theta, prev_response_theta,
                          pull=None, spike=None) -> dict:
    swap_theta = np.asarray(swap_theta, dtype=float)
    n = len(swap_theta)
    arrays = {
        "swap_theta": swap_theta,
        "other_theta": np.asarray(other_theta, dtype=float),
        "prev_target_theta": np.asarray(prev_target_theta, dtype=float),
        "prev_response_theta": np.asarray(prev_response_theta, dtype=float),
        "response_defined": ~np.isnan(np.asarray(prev_response_theta, dtype=float)),
        "pull": None if pull is None else np.asarray(pull, dtype=float),
        "spike": None if spike is None else np.asarray(spike, dtype=float),
    }
    assert all(len(arrays[k]) == n for k in
               ("other_theta", "prev_target_theta", "prev_response_theta", "response_defined"))
    return arrays


def _session_arrays_from_row(row: dict) -> dict:
    recs = row["trial_records"]
    def col(name):
        return [tr[name] if tr[name] is not None else np.nan for tr in recs]
    return _build_session_arrays(
        col("swap_theta"), col("other_theta"), col("prev_target_theta"), col("prev_response_theta"),
        pull=col("content_specific_serial_pull_diff") if row["trial_records"] else None,
        spike=col("total_spike_count") if row["trial_records"] else None,
    )


def _indicators(swap_theta: np.ndarray, other_theta: np.ndarray, ref_theta: np.ndarray) -> np.ndarray:
    d_swap = _circular_distance_rad(swap_theta, ref_theta)
    d_other = _circular_distance_rad(other_theta, ref_theta)
    return np.where(d_swap < d_other, 1.0, np.where(d_swap > d_other, 0.0, np.nan))


def _session_statistics(sa: dict, perm: np.ndarray | None = None) -> dict[str, float | None]:
    """All seven statistics for one session under one assignment of preceding-trial identities.
    ``perm=None`` is the observation; otherwise ``perm`` reassigns which trial's preceding angles
    pair with the current displays, holding every measured quantity of the current trial fixed."""
    n = len(sa["swap_theta"])
    perm = np.arange(n) if perm is None else perm
    prev_t = sa["prev_target_theta"][perm]
    prev_r = sa["prev_response_theta"][perm]
    resp_def = sa["response_defined"][perm]

    ind_t = _indicators(sa["swap_theta"], sa["other_theta"], prev_t)
    fin_t = np.isfinite(ind_t) & np.isfinite(prev_t)
    ind_r = _indicators(sa["swap_theta"], sa["other_theta"], prev_r)
    fin_r = np.isfinite(ind_r) & resp_def & np.isfinite(prev_r)

    out: dict[str, float | None] = {}
    out[STAT_TARGET] = float(np.mean(ind_t[fin_t])) if fin_t.any() else None
    out[STAT_RESPONSE] = float(np.mean(ind_r[fin_r])) if fin_r.any() else None
    out[STAT_PAIRED] = (out[STAT_TARGET] - out[STAT_RESPONSE]
                        if out[STAT_TARGET] is not None and out[STAT_RESPONSE] is not None else None)

    for name in (STAT_PULL_RAW, STAT_RATE, STAT_BIAS, STAT_PARTIAL):
        out[name] = None
    if sa["pull"] is not None and sa["spike"] is not None:
        pull, spike = sa["pull"], sa["spike"]
        elig = fin_t & np.isfinite(pull) & np.isfinite(spike)
        g1, g0 = elig & (ind_t == 1.0), elig & (ind_t == 0.0)
        if g1.any() and g0.any():
            out[STAT_PULL_RAW] = float(np.mean(pull[g1]) - np.mean(pull[g0]))
            out[STAT_RATE] = float(np.mean(spike[g1]) - np.mean(spike[g0]))
            n_elig = int(elig.sum())
            if n_elig >= 2:
                loo = (pull[elig].sum() - pull[elig]) / (n_elig - 1)
                b1, b0 = loo[ind_t[elig] == 1.0], loo[ind_t[elig] == 0.0]
                if len(b1) and len(b0):
                    out[STAT_BIAS] = float(np.mean(b1) - np.mean(b0))
            if n_elig >= 3 and np.std(spike[elig]) > 0.0:
                resid = _residualize(pull[elig], spike[elig])
                r1, r0 = resid[ind_t[elig] == 1.0], resid[ind_t[elig] == 0.0]
                if len(r1) and len(r0):
                    out[STAT_PARTIAL] = float(np.mean(r1) - np.mean(r0))
    return out


def _pool_defined(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def _collect_observed_and_nulls(session_arrays: list[dict], session_names: list[str],
                                n_draws: int, seed_tag_prefix: str) -> tuple[dict, dict]:
    """Observed pooled value per statistic, and the pooled value of every statistic under each
    permutation draw. One permutation per session per draw serves all seven statistics."""
    observed_sessions = [_session_statistics(sa) for sa in session_arrays]
    observed = {name: _pool_defined([row[name] for row in observed_sessions]) for name in STATISTIC_NAMES}

    nulls: dict[str, list[float | None]] = {name: [] for name in STATISTIC_NAMES}
    for draw in range(n_draws):
        draw_rows = []
        for sa, name in zip(session_arrays, session_names):
            rng = np.random.default_rng(stable_seed(f"{seed_tag_prefix}|{name}|{draw}"))
            draw_rows.append(_session_statistics(sa, rng.permutation(len(sa["swap_theta"]))))
        for stat_name in STATISTIC_NAMES:
            nulls[stat_name].append(_pool_defined([row[stat_name] for row in draw_rows]))
    return observed, nulls


# ---------------------------------------------------------------------------
# Percentile inference and the realised-null centring check
# ---------------------------------------------------------------------------

NULL_QUANTILE_LEVELS = [2.5, 5.0, 25.0, 50.0, 75.0, 95.0, 97.5]


def _percentile_result(observed: float | None, draws: list[float | None],
                       per_session_values: list[float | None]) -> dict:
    base = {"status": "not_estimable", "n_draws": 0}
    if observed is None:
        base["reason"] = "observed statistic undefined"
        return base
    clean = np.asarray([d for d in draws if d is not None], dtype=float)
    base["n_draws"] = int(len(clean))
    base["n_sessions_contributing_observed"] = int(sum(v is not None for v in per_session_values))
    base["per_session_values"] = [None if v is None else float(v) for v in per_session_values]
    mdd = minimum_detectable_paired_difference([v for v in per_session_values if v is not None])
    base["minimum_detectable_paired_difference_at_80pct_power"] = \
        mdd if mdd.get("status") == "computed" else {"status": mdd.get("status", "not_computable")}
    if len(clean) < 100:
        base["reason"] = "fewer than 100 usable draws"
        return base
    null_mean = float(np.mean(clean))
    null_sd = float(np.std(clean, ddof=1))
    mc_error = null_sd / np.sqrt(len(clean))
    quantiles = {f"q{level:g}": float(np.percentile(clean, level)) for level in NULL_QUANTILE_LEVELS}
    r_le = (float(np.sum(clean <= observed)) + 1.0) / (len(clean) + 1.0)
    r_ge = (float(np.sum(clean >= observed)) + 1.0) / (len(clean) + 1.0)
    p_two_sided = float(min(1.0, 2.0 * min(r_le, r_ge)))
    base.update({
        "status": "estimated",
        "observed": observed,
        "null_mean": null_mean, "null_sd": null_sd, "null_monte_carlo_error_of_mean": mc_error,
        **quantiles,
        "offset_of_null_mean_from_one_half_descriptive": null_mean - 0.5,
        "observed_minus_null_mean": observed - null_mean,
        "r_less_or_equal": r_le, "r_greater_or_equal": r_ge,
        "p_two_sided_percentile": p_two_sided,
        "significant": bool(p_two_sided < PERCENTILE_ALPHA),
        "direction": ("above" if observed > null_mean else "below" if observed < null_mean else "at_centre"),
        "outside_central_mass_q05_q95": bool(not (quantiles["q5"] <= observed <= quantiles["q95"])),
    })
    return base


def _split_half_centring(draws: list[float | None]) -> dict:
    """Self-consistency of ONE realised null: its two draw-index halves must agree within their own
    Monte-Carlo error, otherwise percentiles of the pooled draws do not describe a stable reference."""
    clean = np.asarray([d for d in draws if d is not None], dtype=float)
    half = len(clean) // 2
    if half < 50:
        return {"status": "not_estimable", "n_draws": int(len(clean))}
    first, second = clean[:half], clean[half: 2 * half]
    m1, m2 = float(np.mean(first)), float(np.mean(second))
    e1 = float(np.std(first, ddof=1)) / np.sqrt(half)
    e2 = float(np.std(second, ddof=1)) / np.sqrt(half)
    denom = float(np.sqrt(e1 ** 2 + e2 ** 2))
    z = abs(m1 - m2) / denom if denom > 0.0 else float("inf")
    return {"status": "checked", "n_draws_per_half": int(half),
            "earlier_half_mean": m1, "later_half_mean": m2,
            "monte_carlo_error_earlier": e1, "monte_carlo_error_later": e2,
            "z_half_disagreement": float(z), "centred_within_monte_carlo_error": bool(z <= CENTRING_Z_THRESHOLD)}


# ---------------------------------------------------------------------------
# Decision rules -- mirrors of the delivered blocks with percentile tests
# ---------------------------------------------------------------------------

def _decide_block_a(target_pct: dict, response_pct: dict, paired_pct: dict,
                    target_pooled_for_mdd: float | None, response_pooled_for_mdd: float | None) -> dict:
    def direction_of(pct: dict) -> str | None:
        if pct.get("status") != "estimated":
            return None
        return pct["direction"] if pct["significant"] else "not_significant"

    t_dir, r_dir = direction_of(target_pct), direction_of(response_pct)
    result: dict = {"target_direction": t_dir, "response_direction": r_dir}
    effect = {"target_effect_size_vs_null_centre":
              None if target_pct.get("status") != "estimated" else target_pct["observed_minus_null_mean"],
              "response_effect_size_vs_null_centre":
              None if response_pct.get("status") != "estimated" else response_pct["observed_minus_null_mean"]}
    result.update(effect)

    if t_dir is None or r_dir is None:
        result["branch"] = BRANCH_A_NOT_COVERED
        result["note"] = "one of the two pooled statistics has no estimable realised null"
        return result

    if t_dir == "below" or r_dir == "below":
        result["branch"] = BRANCH_A_AVOID
        return result
    if t_dir == "above" and r_dir != "above":
        result["branch"] = BRANCH_A_TARGET
        return result
    if r_dir == "above" and t_dir != "above":
        result["branch"] = BRANCH_A_RESPONSE
        return result
    if t_dir == "above" and r_dir == "above":
        p_dir = direction_of(paired_pct)
        if p_dir == "above":
            result["branch"] = BRANCH_A_TARGET
        elif p_dir == "below":
            result["branch"] = BRANCH_A_RESPONSE
        else:
            result["branch"] = BRANCH_A_BOTH_INSEPARABLE
        result["direct_paired_test_direction"] = p_dir
        return result

    mdds = {}
    for label, values in (("target", target_pooled_for_mdd), ("response", response_pooled_for_mdd)):
        mdd = minimum_detectable_paired_difference([v for v in values if v is not None]) if values else \
            {"status": "not_computable"}
        mdds[f"{label}_minimum_detectable_paired_difference"] = \
            mdd.get("mdd") if mdd.get("status") == "computed" else None
    result.update(mdds)
    both_below_floor = all(mdds[k] is not None and mdds[k] < BLOCK_A_MDD_POWERED_NULL_THRESHOLD
                           for k in mdds)
    result["branch"] = BRANCH_A_POWERED_NULL if both_below_floor else BRANCH_A_INCONCLUSIVE
    result["mdd_reference_named"] = BLOCK_A_MDD_POWERED_NULL_THRESHOLD
    return result


def _decide_block_b(raw_pct: dict, rate_pct: dict, bias_pct: dict, partial_pct: dict,
                    reference_effect: float) -> dict:
    result: dict = {}
    bias_sig = bool(bias_pct.get("status") == "estimated" and bias_pct["significant"])
    rate_sig = bool(rate_pct.get("status") == "estimated" and rate_pct["significant"])
    raw_ok = raw_pct.get("status") == "estimated"
    raw_sig = bool(raw_ok and raw_pct["significant"])
    raw_positive = bool(raw_ok and raw_pct["direction"] == "above")

    partial_survives = True
    if rate_sig:
        partial_survives = bool(
            partial_pct.get("status") == "estimated" and partial_pct["significant"]
            and (partial_pct["direction"] == "above") == raw_positive)
    result["bias_only_control_significant"] = bias_sig
    result["rate_control_significant"] = rate_sig
    result["spike_count_partial_survives"] = partial_survives
    result["raw_effect_size_vs_null_centre"] = raw_pct["observed_minus_null_mean"] if raw_ok else None
    result["bias_only_control_effect_size_vs_null_centre"] = \
        bias_pct.get("observed_minus_null_mean") if bias_pct.get("status") == "estimated" else None

    if bias_sig or (rate_sig and not partial_survives):
        result["branch"] = BRANCH_B_NOT_SEPARABLE
        return result
    if not raw_ok:
        result["branch"] = BRANCH_B_NOT_COVERED
        result["note"] = "raw group-difference statistic has no estimable realised null"
        return result
    if raw_sig and raw_positive:
        result["branch"] = BRANCH_B_POSITIVE
        return result
    if raw_sig and not raw_positive:
        result["branch"] = BRANCH_B_OPPOSITE
        return result

    mdd = raw_pct.get("minimum_detectable_paired_difference_at_80pct_power", {})
    mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
    result["minimum_detectable_paired_difference"] = mdd_value
    result["reference_effect_size_delivered_pooled_content_specific_serial_pull"] = reference_effect
    result["branch"] = BRANCH_B_POWERED_NULL if (mdd_value is not None and mdd_value < reference_effect) \
        else BRANCH_B_INCONCLUSIVE
    return result


def _decide_top_branch(centring_ok: bool, target_pct: dict, block_a_fine_branch: str,
                       target_mdd: float | None, block_b_fine_branch: str) -> dict:
    """Pre-declared precedence for the headline verdict. Steps 0-1 (gate, count floor) are enforced
    upstream; this function starts at the centring gate."""
    result: dict = {"centring_gate_passed": bool(centring_ok)}
    if not centring_ok:
        result["branch"] = BRANCH_MISCENTRED_STOP
        return result
    if target_pct.get("status") == "estimated" and target_pct["significant"] and target_pct["direction"] == "above":
        result["branch"] = BRANCH_TOP_FOLLOWS
        result["effect_size_observed"] = target_pct["observed"]
        result["effect_size_null_mean"] = target_pct["null_mean"]
        result["block_a_fine_branch"] = block_a_fine_branch
        return result
    if target_pct.get("status") == "estimated" and target_pct["significant"] and target_pct["direction"] == "below":
        result["branch"] = BRANCH_TOP_NOT_COVERED
        result["note"] = ("target-referenced statistic significant below its null centre: avoidance was "
                          "not pre-declared at this level; every number is in the block records")
        result["effect_size_observed"] = target_pct["observed"]
        result["effect_size_null_mean"] = target_pct["null_mean"]
        return result
    if block_b_fine_branch == BRANCH_B_NOT_SEPARABLE:
        result["branch"] = BRANCH_TOP_NOT_SEPARABLE
        result["block_a_fine_branch"] = block_a_fine_branch
        return result
    if target_mdd is not None and target_mdd < BLOCK_A_MDD_POWERED_NULL_THRESHOLD:
        result["branch"] = BRANCH_TOP_DOES_NOT_FOLLOW
        result["minimum_detectable_paired_difference"] = target_mdd
        result["reference_named"] = (
            f"a minimum detectable difference of {BLOCK_A_MDD_POWERED_NULL_THRESHOLD} proportion units, "
            "the powered-null floor declared with the mirrored block A rule")
        return result
    result["branch"] = BRANCH_TOP_INCONCLUSIVE
    result["minimum_detectable_paired_difference"] = target_mdd
    return result


# ---------------------------------------------------------------------------
# Reproduction gate -- identical checks, read live from the delivered artifacts
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _close(observed, reference) -> bool:
    return observed is not None and reference is not None \
        and abs(float(observed) - float(reference)) <= REPRODUCTION_TOLERANCE


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
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"completed per-session fits available for reuse: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "corpus": "Multi-object spatial working memory in macaque frontal cortex, DANDI 000620.",
        "scope": ("Item count 3 swaps with an admissible immediately-preceding trial. Identical "
                  "admission, statistic definitions and controls to results/"
                  "swap_target_and_preceding_trial_item.json, with every comparison judged against the "
                  "empirical shuffle null of the statistic itself instead of an assumed exact-one-half "
                  "constant."),
        "percentile_test_rule_declared_before_running": PERCENTILE_TEST_DECLARATION,
        "realised_null_centring_rule_declared_before_running": CENTRING_CHECK_DECLARATION,
        "top_branch_rule_declared_before_running": TOP_BRANCH_RULE_DECLARED_BEFORE_RUNNING,
        "block_a_rule_declared_before_running": BLOCK_A_PERCENTILE_RULE_DECLARED_BEFORE_RUNNING,
        "block_b_rule_declared_before_running": BLOCK_B_PERCENTILE_RULE_DECLARED_BEFORE_RUNNING,
        "n_shuffle_draws": N_SHUFFLE_DRAWS,
        "percentile_alpha": PERCENTILE_ALPHA,
        "centring_z_threshold": CENTRING_Z_THRESHOLD,
        "seed_prefix": SEED_PREFIX,
        "near_separation_threshold_degrees": NEAR_SEPARATION_THRESHOLD_DEGREES,
        "min_pooled_admissible_swap_trials": MIN_POOLED_ADMISSIBLE_SWAP_TRIALS,
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass, for accounting)")
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

    rows: list[dict] = []
    behaviour = None
    for session in loaded:
        key = f"session|{session['session']}"
        if key in _COMPLETED_FITS:
            global _FITS_SERVED_FROM_SOURCE_CHECKPOINT
            _FITS_SERVED_FROM_SOURCE_CHECKPOINT += 1
            rows.append(_COMPLETED_FITS[key])
            continue
        if behaviour is None:
            behaviour = watters_behaviour(root)
        row = analyse_session(session, behaviour, SEED_PREFIX)
        _store_fit(key, row)
        rows.append(row)
        _log(f"  fitted here: {session['session']} status={row.get('status')} elapsed={time.time() - t0:.0f}s")

    computed_rows = [r for r in rows if r.get("status") == "computed"]

    # -- reproduction gate ------------------------------------------------
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
    output["reproduction_gate"] = {
        "status": gate_status, "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "delivered_dissociation_reference_read_live": dissociation_reference,
        "delivered_serial_pull_reference_read_live": serial_pull_reference,
    }
    _flush(output)
    _log(f"reproduction gate: {gate_status} elapsed={time.time() - t0:.0f}s")

    if gate_status != "reproduced_exactly":
        output["branch"] = {"branch": BRANCH_GATE_FAILED}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        print(json.dumps({"reproduction_gate": gate_status, "branch": BRANCH_GATE_FAILED}, indent=2))
        return

    # -- zero-drop accounting and the count precondition -------------------
    drop_totals: dict[str, int] = {}
    n_arrays_not_computable_sessions = sum(1 for r in rows if r.get("status") == DROP_SESSION_ARRAYS)
    for r in computed_rows:
        for reason, count in r["drop_counts_by_reason"].items():
            drop_totals[reason] = drop_totals.get(reason, 0) + count
    n_trials_in_arrays_not_computable_sessions = sum(
        r.get("n_trials_total", 0) for r in rows if r.get("status") == DROP_SESSION_ARRAYS)
    n_pooled_surviving = drop_totals.get(SURVIVING, 0)
    reconciles = bool(n_seen == len(loaded) + len(refused))

    output["zero_drop_accounting"] = {
        "n_sessions_seen": n_seen, "n_sessions_loaded": len(loaded),
        "n_sessions_refused_by_shared_loader": len(refused),
        "n_sessions_loaded_but_arrays_not_computable": n_arrays_not_computable_sessions,
        "n_sessions_with_a_computed_arm": len(computed_rows),
        "sessions_reconcile": reconciles,
        "n_trials_total_across_computed_sessions": sum(drop_totals.values()),
        "n_trials_in_arrays_not_computable_sessions": n_trials_in_arrays_not_computable_sessions,
        "drop_counts_by_reason": drop_totals,
        "n_pooled_surviving_item_count_3_swap_trials_with_admissible_preceding_trial": n_pooled_surviving,
        "trial_counts_reconcile": bool(
            sum(drop_totals.values()) + n_trials_in_arrays_not_computable_sessions
            == sum(r.get("n_trials_total", 0) for r in rows)),
    }
    _flush(output)
    _log(f"n_pooled_surviving={n_pooled_surviving} elapsed={time.time() - t0:.0f}s")

    if n_pooled_surviving < MIN_POOLED_ADMISSIBLE_SWAP_TRIALS:
        output["branch"] = {"branch": BRANCH_TOO_FEW_TRIALS, "n_pooled_surviving": n_pooled_surviving,
                            "floor": MIN_POOLED_ADMISSIBLE_SWAP_TRIALS}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        print(json.dumps({"branch": BRANCH_TOO_FEW_TRIALS}, indent=2))
        return

    # -- per-session arrays and the angular-separation covariate ------------
    session_arrays = [_session_arrays_from_row(r) for r in computed_rows]
    session_names = [r["session"] for r in computed_rows]
    separations = [tr["angular_separation_between_uncued_objects_degrees"]
                   for r in computed_rows for tr in r["trial_records"]]
    sep_arr = np.asarray(separations, dtype=float)
    output["angular_separation_covariate"] = {
        "n_trials": int(sep_arr.size),
        "median_degrees": float(np.median(sep_arr)),
        "p5_degrees": float(np.percentile(sep_arr, 5)),
        "p95_degrees": float(np.percentile(sep_arr, 95)),
        "min_degrees": float(np.min(sep_arr)),
        "max_degrees": float(np.max(sep_arr)),
        "n_trials_within_15_degrees": int(np.sum(sep_arr < NEAR_SEPARATION_THRESHOLD_DEGREES)),
        "why_reported": ("The failed exact-one-half premise of the earlier two-alternative analysis is a "
                         "geometric fact of this task (a near-constant uncued separation), so the "
                         "realised separation is carried beside the headline statistic rather than "
                         "assumed away."),
        "per_trial_degrees": [float(v) for v in separations],
    }
    _flush(output)

    # -- observed statistics and their realised nulls -----------------------
    _log(f"generating {N_SHUFFLE_DRAWS} permutation draws x {len(session_arrays)} sessions ...")
    observed, nulls = _collect_observed_and_nulls(session_arrays, session_names, N_SHUFFLE_DRAWS,
                                                  f"{SEED_PREFIX}|null")
    per_session_observed = [{name: row[name] for name in STATISTIC_NAMES}
                            for row in (_session_statistics(sa) for sa in session_arrays)]
    percentile_results = {
        name: _percentile_result(observed[name], nulls[name],
                                 [row[name] for row in per_session_observed])
        for name in STATISTIC_NAMES
    }
    output["observed_statistics_and_empirical_nulls"] = percentile_results
    _flush(output)

    centring_checks = {name: _split_half_centring(nulls[name]) for name in STATISTIC_NAMES}
    failing = {name: chk for name, chk in centring_checks.items()
               if chk.get("status") == "checked" and not chk["centred_within_monte_carlo_error"]}
    max_z = max((chk["z_half_disagreement"] for chk in centring_checks.values()
                 if chk.get("status") == "checked"), default=None)
    output["realised_null_centring_check"] = {
        "rule": CENTRING_CHECK_DECLARATION,
        "per_statistic": centring_checks,
        "max_z_half_disagreement": max_z,
        "all_realised_nulls_centred_within_monte_carlo_error": not failing,
        "statistics_failing": sorted(failing),
    }
    _flush(output)
    _log(f"centring check: max_z={max_z} failing={sorted(failing)} elapsed={time.time() - t0:.0f}s")

    if failing:
        output["branch"] = {
            "branch": BRANCH_MISCENTRED_STOP,
            "max_z_half_disagreement": max_z,
            "threshold": CENTRING_Z_THRESHOLD,
            "statistics_failing": sorted(failing),
            "detail": "the realised permutation null itself could not be estimated stably enough for a "
                      "percentile test; no block verdict is emitted",
        }
        output["how_this_artifact_was_assembled"] = _assembly_block()
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        print(json.dumps({"branch": BRANCH_MISCENTRED_STOP, "max_z": max_z}, indent=2))
        return

    # -- block A -------------------------------------------------------------
    target_pct = percentile_results[STAT_TARGET]
    response_pct = percentile_results[STAT_RESPONSE]
    paired_pct = percentile_results[STAT_PAIRED]

    def _mdd_value(pct: dict) -> float | None:
        node = pct.get("minimum_detectable_paired_difference_at_80pct_power", {})
        return node.get("mdd") if isinstance(node, dict) and node.get("status") == "computed" else None

    block_a_branch = _decide_block_a(target_pct, response_pct, paired_pct,
                                     target_pct.get("per_session_values"),
                                     response_pct.get("per_session_values"))
    block_a_branch["target_minimum_detectable_paired_difference"] = _mdd_value(target_pct)
    block_a_branch["response_minimum_detectable_paired_difference"] = _mdd_value(response_pct)

    near_mask = sep_arr < NEAR_SEPARATION_THRESHOLD_DEGREES
    if near_mask.any():
        # rebuild each session's arrays filtered through its own separation vector
        sens_arrays = []
        idx = 0
        for sa in session_arrays:
            n = len(sa["swap_theta"])
            keep = slice(idx, idx + n)
            mask = ~near_mask[keep]
            sens_arrays.append({k: (v[mask] if isinstance(v, np.ndarray) and len(v) == n else v)
                                for k, v in sa.items()})
            idx += n
        sens_observed, sens_nulls = _collect_observed_and_nulls(
            sens_arrays, session_names, N_SHUFFLE_DRAWS, f"{SEED_PREFIX}|null_near_excluded")
        sens_results = {name: _percentile_result(sens_observed[name], sens_nulls[name], [])
                        for name in STATISTIC_NAMES}
    else:
        sens_results = {"status": "no_trials_below_the_near_separation_threshold_sensitivity_is_vacuous"}

    output["block_a_behavioural"] = {
        "primary": {"target_referenced": target_pct, "response_referenced": response_pct,
                    "target_minus_response_direct_paired": paired_pct},
        "branch": block_a_branch,
        "sensitivity_excluding_near_15_degree_trials_decides_nothing": sens_results,
    }
    _flush(output)
    _log(f"block A branch: {block_a_branch['branch']} elapsed={time.time() - t0:.0f}s")

    # -- block B -------------------------------------------------------------
    reference_effect = abs(serial_pull_reference["mean_value"])
    block_b_branch = _decide_block_b(percentile_results[STAT_PULL_RAW], percentile_results[STAT_RATE],
                                     percentile_results[STAT_BIAS], percentile_results[STAT_PARTIAL],
                                     reference_effect)
    output["block_b_neural"] = {
        "raw_group_difference": percentile_results[STAT_PULL_RAW],
        "rate_control_group_difference": percentile_results[STAT_RATE],
        "bias_only_control_group_difference": percentile_results[STAT_BIAS],
        "spike_count_partial_group_difference": percentile_results[STAT_PARTIAL],
        "reference_effect_size_delivered_pooled_content_specific_serial_pull": reference_effect,
        "branch": block_b_branch,
    }
    _flush(output)
    _log(f"block B branch: {block_b_branch['branch']} elapsed={time.time() - t0:.0f}s")

    # -- headline verdict -----------------------------------------------------
    top = _decide_top_branch(centring_ok=True, target_pct=target_pct,
                             block_a_fine_branch=block_a_branch["branch"],
                             target_mdd=_mdd_value(target_pct),
                             block_b_fine_branch=block_b_branch["branch"])
    output["branch"] = {
        "top": top,
        "top_branch": top["branch"],
        "block_a": block_a_branch["branch"],
        "block_b": block_b_branch["branch"],
    }
    output["how_this_artifact_was_assembled"] = _assembly_block()
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"complete elapsed={time.time() - t0:.0f}s")
    print(json.dumps({
        "reproduction_gate": gate_status, "n_pooled_surviving": n_pooled_surviving,
        "centring_max_z": max_z, "top_branch": top["branch"],
        "block_a_branch": block_a_branch["branch"], "block_b_branch": block_b_branch["branch"],
    }, indent=2, default=float))


def _assembly_block() -> dict:
    return {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_SOURCE_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "source_fit_record": str(SOURCE_CHECKPOINT_PATH.relative_to(ROOT)),
        "own_fit_record": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "git_commit": git_commit(ROOT),
    }


if __name__ == "__main__":
    main()
