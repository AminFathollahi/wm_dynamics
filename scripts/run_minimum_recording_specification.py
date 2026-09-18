"""Estimate channel-by-trial recovery of the rate-free state deviation."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
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

from corpus_sessions import alm_data_directory, data_root, iter_watters  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA as AXIS_DEFINED_CORPORA, _residual_rows, _unit_residual_matrix, leading_eigenvector,
)
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    HUMAN_CORPORA_FOR_THE_CENSUS, _human_seen_denominator, _load_alm_for_counting_noise_census, _load_human_for_counting_noise_census,
    _load_panichello_for_counting_noise_census,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import _panichello_directory  # noqa: E402
from statistics import stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "minimum_recording_specification.json"
ANALYSIS_VERSION = "2026-09-07"

# ---------------------------------------------------------------------------------------------------
# Constants -- every one fixed here, before any session is loaded or any fit runs.
# ---------------------------------------------------------------------------------------------------
CHANNEL_RUNGS_UNIVERSAL = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
CALIBRATION_TRIAL_RUNGS = (8, 16, 32, 64, 128, 256, 512)
NATIVE_LABEL = "native"  # the deterministic full-population / full-trial-pool condition, reported separately
                           # from the common numeric channel grid

N_UNIT_DRAWS = 50        # pre-declared floor for a subsampled channel rung, never reduced
N_TRIAL_DRAWS = 20       # pre-declared floor for a subsampled trial rung, never reduced
N_AXIS_UNIT_DRAWS = 50   # supplementary axis diagnostic, same floor as the primary channel ladder

MIN_UNITS_FOR_A_SESSION = 2
MIN_TRIALS_FOR_A_SESSION = MIN_TRIALS_WITH_DEFINED_DIRECTION  # 16, this project's own floor for a session
                                                               # to have a definable direction at all
MIN_SESSIONS_FOR_CORPUS_CURVE = 4

FIDELITY_TARGETS = (0.5, 0.7, 0.9)  # fractions of the pooled debiased NATIVE (full-population) recovery
SMALL_CHANNEL_FRACTION = 0.25       # a target reached at or below this fraction of a corpus's own median
                                     # native unit count is called "a small channel count"

N_BOOT_SESSION_CLUSTER = 2000

BRANCH_RECOVERABLE_SMALL = "the_metric_is_recoverable_from_a_small_channel_count"
BRANCH_REQUIRES_MORE_THAN_SMALL = "highest_fidelity_requires_more_than_a_small_channel_count"
BRANCH_HIGH_FIDELITY_NOT_REACHED = "highest_fidelity_not_reached_on_the_common_channel_grid"
BRANCH_LOW_FIDELITY_NOT_REACHED = "lowest_fidelity_not_reached_on_the_common_channel_grid"
BRANCH_NOT_DETERMINABLE = "not_determinable_for_a_stated_reason"

ANALYSIS_TIMING_DISCLOSURE = (
    "The held-out split, channel and initial total-trial ladders, draw counts, matched null, fidelity targets, and "
    "session bootstrap were fixed before the initial full fit. A PI audit of that preliminary artifact "
    "found that its branch names exceeded what the common grid identified and that its joint table lacked "
    "fixed-cohort threshold inference. The branch names were corrected and the bidirectional joint "
    "threshold summaries were added. The preliminary implementation varied the size of the score set with "
    "the trial pool, which confounded calibration data with evaluation precision; the final trial axis uses "
    "the corresponding half-sized calibration rungs and one fixed held-out score set. The native anchor's "
    "matched null was also increased from one draw to the same 50-draw floor used at numeric channel rungs. "
    "The channel grid, estimand, and target fractions did not change. The corrected labels and joint summaries are post-fit design guidance, not preregistered "
    "confirmatory findings."
)

NULL_FLOOR_DISCLOSURE = (
    "The matched random-axis null uses an axis drawn independently of any data, at the same channel count "
    "-- exactly as declared. It does not, and cannot, net out a separate, purely geometric contribution to "
    "the observed recovery score: the channel-subset reference direction and the full-population reference "
    "direction are both estimated from the SAME reference trials by design (a real device with fewer "
    "channels observes the SAME reference period, not an independent one), so a channel subset partially "
    "reconstructs the full-population reference direction's own restriction to those channels by "
    "concentration of measure alone, even on activity with no reproducible trial-to-trial structure "
    "whatsoever. Tests quantify the residual on zero-mean, shared-direction-free synthetic activity and "
    "verify that planted structure produces greater recovery at the same channel count. The debiased curve therefore "
    "contains an upward geometric contribution at small q and is not a bias-free estimate of biologically "
    "meaningful recovery. The contribution is not proven additive or monotone, so the curve is not labelled "
    "a mathematical upper or lower bound. Every fidelity-target crossing and branch below must be read with "
    "that construct limitation; no claim in this artifact asserts that the curve isolates biological signal."
)

POWER_SCALE_DISCLOSURE = (
    "The recovery score reported throughout this artifact (a Pearson correlation between a channel-"
    "subset-estimated deviation and a held-out full-population reference, minus its own matched random-"
    "axis null at the same channel count) has no pre-existing named reference effect size anywhere else in "
    "this project -- it is a new scale, defined only here. Every comparison in this artifact is therefore "
    "against the metric's OWN matched null and its own whole-session cluster bootstrap uncertainty "
    f"({N_BOOT_SESSION_CLUSTER} whole-session percentile-bootstrap draws), never against an external literature or project "
    "effect size, and no branch here is ever labelled a literature-referenced 'powered null'."
)

ANALYSIS_RULE = (
    "Per session, a seeded 50/50 split of every trial the session has (the odd trial, if any, goes to the "
    "reference half) separates a REFERENCE set (used only to estimate a direction) from a SCORE set (the "
    "only trials any recovery number is ever computed on). The full-population reference direction is the "
    "mean unit-normalised activity vector across REFERENCE trials, renormalised to unit length -- the same "
    "quantity rate_free_state_deviation's own leave-one-out mean reduces to when its 'every other trial' "
    "pool is exactly the REFERENCE set. Its own deviation on the SCORE trials is the fixed target every "
    "channel-count and trial-count rung is scored against. For a channel rung of q units (drawn without "
    f"replacement from the session's full unit set, >= {N_UNIT_DRAWS} independent draws per rung, with the "
    "full unit count retained only as a deterministic NATIVE descriptive condition), the identical construction is repeated "
    "using only those q units for BOTH the reference direction and the SCORE trials' own vectors, and the "
    "resulting subset deviation is Pearson-correlated against the full-population reference's own SCORE-"
    "trial deviation -- the recovery score. Every recovery score carries a MATCHED random-axis null at the "
    "same channel rung, same score trials and same channel draw: the reference direction is replaced by an "
    "independent random unit vector in the same q-dimensional space, unrelated to any data, and the same "
    "correlation is computed against it. The debiased recovery score is the observed value minus its "
    "matched null, computed per draw before any pooling. The identical channel ladder is repeated while "
    f"subsampling only the fixed REFERENCE half over calibration sizes {CALIBRATION_TRIAL_RUNGS}, with "
    f">= {N_TRIAL_DRAWS} draws per numeric calibration rung. The SCORE half and its full-reference target "
    "remain fixed across the ladder. The NATIVE calibration rung uses all reference trials. This gives the "
    "joint channel-by-calibration-trial table; the channel-only curve is its NATIVE calibration row. Numeric channel and "
    "calibration-trial grids contain only universal rungs available to every eligible session; a session's "
    "own full count is never inserted as a numeric rung. Each session contributes one median debiased value "
    "per cell, and only sessions computable over the complete common joint grid and NATIVE anchors form the "
    "joint threshold cohort. For every fidelity target, the artifact reports the minimum numeric calibration-trial count "
    "at each fixed channel rung and the minimum numeric channel count at each fixed trial rung. Crossings "
    "outside a common grid are right-censored; the heterogeneous NATIVE anchors are shown but never converted "
    "into a numeric threshold. The pooled per-draw table remains descriptive. No interval anywhere in "
    "this artifact is a trial-count formula or an intraclass-correlation design effect; every threshold "
    f"interval is a whole-session percentile cluster bootstrap ({N_BOOT_SESSION_CLUSTER} draws) "
    "over those session summaries.\n"
    f"Fidelity targets are {FIDELITY_TARGETS}, each read as that fraction of the corpus's median session-level "
    "NATIVE (full-population) recovery. For each target, the smallest numeric channel rung whose fixed-cohort "
    "median curve reaches and remains above it at every larger common rung is reported with a whole-session cluster-bootstrap confidence interval on that "
    "rung. The NATIVE anchor is recomputed in each bootstrap sample. An observed target not reached on the "
    "common grid has no finite interval. A bootstrap sample without a sustained crossing is represented as infinity; "
    "an interval endpoint is right-censored when the corresponding percentile lies beyond the grid.\n"
    "Per corpus, evaluated in this order: (1) if fewer than "
    f"{MIN_SESSIONS_FOR_CORPUS_CURVE} sessions produce a computable pooled curve, or no numeric channel "
    f"rung or NATIVE anchor is computable at all, the branch is '{BRANCH_NOT_DETERMINABLE}', with the "
    "reason stated. (2) Else, if the pooled debiased channel-only curve does not reach and remain above the LOWEST "
    f"fidelity target ({min(FIDELITY_TARGETS)}) on the sampled numeric channel grid, the branch is "
    f"'{BRANCH_LOW_FIDELITY_NOT_REACHED}'. (3) Else, if the HIGHEST fidelity target "
    f"({max(FIDELITY_TARGETS)}) is not reached and sustained on the common numeric grid, the branch is "
    f"'{BRANCH_HIGH_FIDELITY_NOT_REACHED}'. (4) Else, if the highest target is reached at a numeric "
    f"channel rung at or below {SMALL_CHANNEL_FRACTION} times the corpus's own median NATIVE unit count, "
    f"the branch is '{BRANCH_RECOVERABLE_SMALL}'. (5) Else the branch is "
    f"'{BRANCH_REQUIRES_MORE_THAN_SMALL}'. These grid-censored branches do not locate an unobserved "
    "crossing between the largest common numeric rung and the heterogeneous NATIVE anchors. "
    + NULL_FLOOR_DISCLOSURE + " " + POWER_SCALE_DISCLOSURE
)


# =======================================================================================================
# Pure arithmetic -- reference/score construction, the matched null, rung ladders, pooling, fidelity-
# target search, the whole-session cluster bootstrap and the branch classifier. No I/O; every one covered
# by tests/test_minimum_recording_specification.py before any real corpus is touched.
# =======================================================================================================

def _reference_direction(activity_ref: np.ndarray) -> np.ndarray | None:
    """Mean unit-normalised activity vector across REFERENCE trials (rows), renormalised to unit length.
    None if no reference trial has a defined direction (all-zero activity) or the mean itself vanishes."""
    activity = np.asarray(activity_ref, dtype=float)
    if activity.shape[0] == 0:
        return None
    norms = np.linalg.norm(activity, axis=1)
    valid = norms > 0
    if not np.any(valid):
        return None
    unit_vectors = activity[valid] / norms[valid, None]
    mean_vec = unit_vectors.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    return None if mean_norm == 0.0 else mean_vec / mean_norm


def _deviation_against_direction(activity_score: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """1 - cosine(unit_vector_i, direction) for each SCORE trial -- rate_free_state_deviation's own
    per-trial formula with its leave-one-out mean replaced by a direction fixed from a disjoint REFERENCE
    set (verified bit-for-bit equivalent to invoking rate_free_state_deviation itself, in the test suite,
    via the leave-one-out identity that a single score trial stacked onto the reference block reduces to
    exactly this fixed-direction cosine). NaN for a trial with zero total activity, matching that
    convention."""
    activity = np.asarray(activity_score, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    return 1.0 - (unit_vectors @ direction)


def recovery_score(reference_values: np.ndarray, candidate_values: np.ndarray) -> float | None:
    finite = np.isfinite(reference_values) & np.isfinite(candidate_values)
    if int(finite.sum()) < 4:
        return None
    a, b = reference_values[finite], candidate_values[finite]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def session_rungs(n_available: int, universal: tuple[int, ...]) -> list[int]:
    """Every universal rung reachable by this session (<= its own count), plus its own full count as the
    top numeric rung if not already exactly a universal value. Mirrors this project's existing native/
    subsample ladder convention (run_count_subsampling_ladder.resolve_unit_target) keyed on a literal
    channel or trial count rather than a derived spike-count target."""
    rungs = [r for r in universal if r <= n_available]
    if not rungs or rungs[-1] != n_available:
        rungs.append(n_available)
    return rungs


def common_channel_rungs(native_unit_counts: list[int], universal: tuple[int, ...]) -> list[int]:
    """Universal channel rungs available to every session in a corpus curve.

    A recording threshold must compare the same sessions at every rung.  Full native
    counts remain descriptive because they differ across sessions and cannot form a
    common numeric threshold grid.
    """
    if not native_unit_counts:
        return []
    smallest_native_count = min(native_unit_counts)
    return [q for q in universal if q <= smallest_native_count]


def draw_index_subset(n_full: int, k: int, seed_tag: str) -> np.ndarray:
    if k >= n_full:
        return np.arange(n_full)
    rng = np.random.default_rng(stable_seed(seed_tag))
    return np.sort(rng.choice(n_full, size=k, replace=False))


def reference_score_split(trial_indices: np.ndarray, seed_tag: str) -> tuple[np.ndarray, np.ndarray]:
    """A pre-declared, seeded 50/50 split of a (possibly already trial-count-subsampled) index set into a
    REFERENCE half and a SCORE half -- design safeguard (a): sharing trials between the reference and the
    score is never permitted anywhere in this module. The reference half gets the extra trial when the
    count is odd."""
    rng = np.random.default_rng(stable_seed(seed_tag))
    shuffled = rng.permutation(np.asarray(trial_indices))
    n_ref = int(np.ceil(len(shuffled) / 2))
    return np.sort(shuffled[:n_ref]), np.sort(shuffled[n_ref:])


def reference_pair_score_split(trial_indices: np.ndarray, seed_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed(seed_tag))
    groups = np.array_split(rng.permutation(np.asarray(trial_indices)), 3)
    return tuple(np.sort(group) for group in groups)


def random_axis(q: int, seed_tag: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed_tag))
    v = rng.standard_normal(q)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _channel_rung_draws(activity_ref: np.ndarray, activity_score: np.ndarray, d_full_score: np.ndarray,
                         q_label, n_units_full: int, n_draws: int, seed_tag: str) -> dict:
    """Compute observed and matched-null recovery draws for one cell."""
    is_native = q_label == NATIVE_LABEL
    q = n_units_full if is_native else q_label
    draws = n_draws
    observed = np.full(draws, np.nan)
    null = np.full(draws, np.nan)
    for d in range(draws):
        tag = f"{seed_tag}|q={q_label}|draw={d}"
        cols = np.arange(n_units_full) if is_native else draw_index_subset(n_units_full, q, f"{tag}|units")
        direction = _reference_direction(activity_ref[:, cols])
        if direction is not None:
            r_obs = recovery_score(d_full_score, _deviation_against_direction(activity_score[:, cols], direction))
            if r_obs is not None:
                observed[d] = r_obs
        axis = random_axis(len(cols), f"{tag}|null_axis")
        r_null = recovery_score(d_full_score, _deviation_against_direction(activity_score[:, cols], axis))
        if r_null is not None:
            null[d] = r_null
    return {"observed": observed, "null": null}


def session_joint_cells(activity_full: np.ndarray, channel_rungs: list, trial_rungs: list,
                         n_unit_draws: int, n_trial_draws: int, seed_tag: str) -> dict:
    """Build recovery cells on a fixed reference/score split."""
    n_trials_full, n_units_full = activity_full.shape
    cells = {(t_label, q_label): {"observed": [], "null": []} for t_label in trial_rungs for q_label in channel_rungs}
    ref_full_idx, score_idx = reference_score_split(np.arange(n_trials_full), f"{seed_tag}|fixed_split")
    activity_score = activity_full[score_idx]
    full_direction = _reference_direction(activity_full[ref_full_idx])
    if full_direction is None:
        return {k: {"observed": np.array([]), "null": np.array([])} for k in cells}
    d_full_score = _deviation_against_direction(activity_score, full_direction)
    for t_label in trial_rungs:
        is_native_t = t_label == NATIVE_LABEL
        t_draws = 1 if is_native_t else n_trial_draws
        for td in range(t_draws):
            if is_native_t:
                ref_idx = ref_full_idx
            else:
                local_idx = draw_index_subset(
                    len(ref_full_idx), t_label, f"{seed_tag}|T={t_label}|draw={td}|select")
                ref_idx = ref_full_idx[local_idx]
            activity_ref = activity_full[ref_idx]
            for q_label in channel_rungs:
                cell_tag = f"{seed_tag}|T={t_label}|tdraw={td}"
                result = _channel_rung_draws(activity_ref, activity_score, d_full_score, q_label,
                                              n_units_full, n_unit_draws, cell_tag)
                cells[(t_label, q_label)]["observed"].extend(result["observed"].tolist())
                cells[(t_label, q_label)]["null"].extend(result["null"].tolist())
    return {k: {"observed": np.array(v["observed"]), "null": np.array(v["null"])} for k, v in cells.items()}


def independent_reference_cells(activity_full: np.ndarray, channel_rungs: list, n_draws: int,
                                seed_tag: str) -> dict:
    n_trials, n_units = activity_full.shape
    target_idx, candidate_idx, score_idx = reference_pair_score_split(
        np.arange(n_trials), f"{seed_tag}|independent_split")
    target_direction = _reference_direction(activity_full[target_idx])
    if target_direction is None:
        return {q: {"observed": np.array([]), "null": np.array([])} for q in channel_rungs}
    target = _deviation_against_direction(activity_full[score_idx], target_direction)
    return {
        q: _channel_rung_draws(
            activity_full[candidate_idx], activity_full[score_idx], target, q, n_units, n_draws,
            f"{seed_tag}|independent")
        for q in channel_rungs
    }


def pool_cell(values: list[float]) -> dict:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"status": "not_computable", "n": 0}
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {"status": "computed", "n": int(arr.size), "median": float(q50), "iqr_low": float(q25), "iqr_high": float(q75)}


def smallest_q_reaching_target(sorted_qs: list[int], curve_by_q: dict, target_value: float) -> int | None:
    for index, q in enumerate(sorted_qs):
        later = [curve_by_q.get(rung) for rung in sorted_qs[index:]]
        if all(v is not None and np.isfinite(v) and v >= target_value for v in later):
            return q
    return None


def bootstrap_ci_for_crossing_q(point_estimate_q: int, session_summaries: list[dict], target_fraction: float,
                                 sorted_qs: list[int], n_boot: int, seed_tag: str) -> dict:
    """Cluster-bootstrap a channel threshold by resampling complete session summaries.

    The native anchor is recomputed in each bootstrap sample because the target is a
    fraction of that anchor.  A sample that does not cross is right-censored beyond
    the observed grid.  It is never converted into the largest observed rung, which
    would manufacture a finite interval.
    """
    n = len(session_summaries)
    if n < 2:
        return {"status": "not_computable", "n_sessions": n}
    invalid_sessions = [
        s.get("session", "unknown") for s in session_summaries
        if (not np.isfinite(s.get("native_debiased_median", np.nan))
            or any(q not in s.get("by_q_debiased_median", {})
                   or not np.isfinite(s["by_q_debiased_median"][q]) for q in sorted_qs))
    ]
    if invalid_sessions:
        return {
            "status": "not_computable", "n_sessions": n,
            "reason": "incomplete_or_noncomparable_session_summaries",
            "sessions_failing_complete_grid": invalid_sessions,
        }
    rng = np.random.default_rng(stable_seed(seed_tag))
    bootstrap_crossings = np.full(n_boot, np.inf)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [session_summaries[i] for i in idx]
        native_values = [s["native_debiased_median"] for s in resampled]
        pooled_curve: dict[int, float] = {}
        for q in sorted_qs:
            pooled_curve[q] = float(np.median([s["by_q_debiased_median"][q] for s in resampled]))
        target_value = target_fraction * float(np.median(native_values))
        crossing = smallest_q_reaching_target(sorted_qs, pooled_curve, target_value)
        if crossing is not None:
            bootstrap_crossings[b] = crossing

    def percentile_bound(probability: float) -> float:
        rank = max(0, int(np.ceil(probability * n_boot)) - 1)
        return float(np.sort(bootstrap_crossings)[rank])

    lower = percentile_bound(0.025)
    upper = percentile_bound(0.975)
    n_right_censored = int(np.isinf(bootstrap_crossings).sum())
    upper_censored = not np.isfinite(upper)
    lower_censored = not np.isfinite(lower)
    return {
        "status": "right_censored" if upper_censored or lower_censored else "computed",
        "method": "whole_session_percentile_bootstrap",
        "n_sessions": n, "n_bootstrap_draws": n_boot,
        "n_bootstrap_crossings": n_boot - n_right_censored,
        "n_bootstrap_right_censored": n_right_censored,
        "right_censor_fraction": n_right_censored / n_boot,
        "point_estimate_q": point_estimate_q,
        "ci_lower": None if lower_censored else lower,
        "ci_lower_status": "right_censored" if lower_censored else "computed",
        "ci_upper": None if upper_censored else upper,
        "ci_upper_status": "right_censored" if upper_censored else "computed",
    }


def compute_fidelity_targets(channel_curve_debiased_by_q: dict, native_anchor_value: float,
                              numeric_qs_sorted: list[int], targets: tuple, session_summaries: list[dict],
                              seed_tag: str) -> dict:
    out = {}
    for frac in targets:
        target_value = frac * native_anchor_value
        q_hit = smallest_q_reaching_target(numeric_qs_sorted, channel_curve_debiased_by_q, target_value)
        if q_hit is None:
            max_q = numeric_qs_sorted[-1] if numeric_qs_sorted else None
            out[str(frac)] = {
                "status": "right_censored", "crossing_status": "not_reached_at_maximum_available_channel_count",
                "target_value": target_value,
                "maximum_available_channel_count": max_q,
                "debiased_value_at_maximum_available_channel_count":
                    channel_curve_debiased_by_q.get(max_q) if max_q is not None else None,
                "censoring_note": "The target was not attained on the observed common channel grid; "
                                  "the crossing is beyond the grid and has no finite confidence interval.",
            }
            continue
        ci = bootstrap_ci_for_crossing_q(q_hit, session_summaries, frac, numeric_qs_sorted,
                                          N_BOOT_SESSION_CLUSTER, f"{seed_tag}|target={frac}")
        out[str(frac)] = {"status": "reached", "target_value": target_value, "smallest_q_reaching_target": q_hit,
                           "cluster_bootstrap_ci": ci}
    return out


def _joint_value(summary: dict, trial_label, channel_label) -> float:
    return summary["joint_debiased_median"].get(str(trial_label), {}).get(str(channel_label), np.nan)


def bootstrap_ci_for_joint_crossing(point_estimate: int, session_summaries: list[dict],
                                     target_fraction: float, varying_rungs: list[int], fixed_label,
                                     vary_trials: bool, n_boot: int, seed_tag: str) -> dict:
    n = len(session_summaries)
    if n < 2:
        return {"status": "not_computable", "n_sessions": n}
    invalid = []
    for summary in session_summaries:
        values = [
            _joint_value(summary, rung, fixed_label) if vary_trials else _joint_value(summary, fixed_label, rung)
            for rung in varying_rungs
        ]
        if not np.isfinite(summary.get("native_debiased_median", np.nan)) or not np.all(np.isfinite(values)):
            invalid.append(summary.get("session", "unknown"))
    if invalid:
        return {"status": "not_computable", "n_sessions": n,
                "reason": "incomplete_or_noncomparable_session_summaries",
                "sessions_failing_complete_grid": invalid}
    rng = np.random.default_rng(stable_seed(seed_tag))
    crossings = np.full(n_boot, np.inf)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sampled = [session_summaries[i] for i in idx]
        target = target_fraction * float(np.median([s["native_debiased_median"] for s in sampled]))
        curve = {}
        for rung in varying_rungs:
            values = [
                _joint_value(s, rung, fixed_label) if vary_trials else _joint_value(s, fixed_label, rung)
                for s in sampled
            ]
            curve[rung] = float(np.median(values))
        crossing = smallest_q_reaching_target(varying_rungs, curve, target)
        if crossing is not None:
            crossings[b] = crossing
    ordered = np.sort(crossings)
    lower = float(ordered[max(0, int(np.ceil(0.025 * n_boot)) - 1)])
    upper = float(ordered[max(0, int(np.ceil(0.975 * n_boot)) - 1)])
    n_censored = int(np.isinf(crossings).sum())
    return {
        "status": "right_censored" if not np.isfinite(lower) or not np.isfinite(upper) else "computed",
        "method": "whole_session_percentile_bootstrap",
        "n_sessions": n,
        "n_bootstrap_draws": n_boot,
        "n_bootstrap_crossings": n_boot - n_censored,
        "n_bootstrap_right_censored": n_censored,
        "right_censor_fraction": n_censored / n_boot,
        "point_estimate": point_estimate,
        "ci_lower": None if not np.isfinite(lower) else lower,
        "ci_lower_status": "right_censored" if not np.isfinite(lower) else "computed",
        "ci_upper": None if not np.isfinite(upper) else upper,
        "ci_upper_status": "right_censored" if not np.isfinite(upper) else "computed",
    }


def _joint_threshold(curve: dict[int, float], native_anchor: float, fraction: float,
                     session_summaries: list[dict], varying_rungs: list[int], fixed_label,
                     vary_trials: bool, seed_tag: str) -> dict:
    target = fraction * native_anchor
    crossing = smallest_q_reaching_target(varying_rungs, curve, target)
    axis = "calibration_trial_count" if vary_trials else "channel_count"
    if crossing is None:
        maximum = varying_rungs[-1] if varying_rungs else None
        return {
            "status": "right_censored",
            "crossing_status": f"not_reached_at_maximum_common_{axis}",
            "target_value": target,
            f"maximum_common_{axis}": maximum,
            "debiased_value_at_maximum": curve.get(maximum) if maximum is not None else None,
        }
    return {
        "status": "reached",
        "target_value": target,
        f"smallest_{axis}_reaching_target": crossing,
        "cluster_bootstrap_ci": bootstrap_ci_for_joint_crossing(
            crossing, session_summaries, fraction, varying_rungs, fixed_label, vary_trials,
            N_BOOT_SESSION_CLUSTER, seed_tag),
    }


def compute_joint_fidelity_specification(session_summaries: list[dict], numeric_calibration_trial_rungs: list[int],
                                         numeric_channel_rungs: list[int], targets: tuple,
                                         seed_tag: str) -> dict:
    labels_t = numeric_calibration_trial_rungs + [NATIVE_LABEL]
    labels_q = numeric_channel_rungs + [NATIVE_LABEL]
    if len(session_summaries) < MIN_SESSIONS_FOR_CORPUS_CURVE:
        return {"status": "not_computable", "n_sessions": len(session_summaries),
                "reason": "below_predeclared_session_floor"}
    if not numeric_calibration_trial_rungs or not numeric_channel_rungs:
        return {"status": "not_computable", "n_sessions": len(session_summaries),
                "reason": "empty_common_numeric_grid"}
    surface = {
        str(t): {
            str(q): float(np.median([_joint_value(s, t, q) for s in session_summaries]))
            for q in labels_q
        }
        for t in labels_t
    }
    native_anchor = surface[NATIVE_LABEL][NATIVE_LABEL]
    target_rows = {}
    for fraction in targets:
        by_channel = {}
        for q in numeric_channel_rungs:
            curve = {t: surface[str(t)][str(q)] for t in numeric_calibration_trial_rungs}
            by_channel[str(q)] = _joint_threshold(
                curve, native_anchor, fraction, session_summaries, numeric_calibration_trial_rungs, q, True,
                f"{seed_tag}|fraction={fraction}|fixed_q={q}")
        by_trial = {}
        for t in numeric_calibration_trial_rungs:
            curve = {q: surface[str(t)][str(q)] for q in numeric_channel_rungs}
            by_trial[str(t)] = _joint_threshold(
                curve, native_anchor, fraction, session_summaries, numeric_channel_rungs, t, False,
                f"{seed_tag}|fraction={fraction}|fixed_t={t}")
        target_rows[str(fraction)] = {
            "minimum_calibration_trials_by_channel_count": by_channel,
            "minimum_channels_by_trial_count": by_trial,
        }
    return {
        "status": "computed",
        "n_sessions_fixed_complete_cohort": len(session_summaries),
        "common_numeric_calibration_trial_grid": numeric_calibration_trial_rungs,
        "common_numeric_channel_grid": numeric_channel_rungs,
        "native_anchor_debiased_median": native_anchor,
        "pooled_fixed_cohort_debiased_median": surface,
        "fidelity_targets": target_rows,
        "heterogeneous_native_anchors_are_descriptive_only": True,
        "inference_timing": "post_fit_design_guidance",
    }


def classify_corpus_branch(n_sessions_analysed: int, channel_curve_debiased_by_q: dict,
                            native_anchor_value: float | None, native_full_median_q: float | None,
                            numeric_qs_sorted: list[int], fidelity_targets: tuple, small_channel_fraction: float,
                            min_sessions: int) -> dict:
    if n_sessions_analysed < min_sessions:
        return {"branch": BRANCH_NOT_DETERMINABLE,
                "reason": f"only {n_sessions_analysed} sessions reached a computable curve, below the "
                          f"pre-declared floor of {min_sessions}"}
    if not numeric_qs_sorted or native_anchor_value is None or native_full_median_q is None:
        return {"branch": BRANCH_NOT_DETERMINABLE,
                "reason": "no numeric channel rung or NATIVE anchor produced a computable pooled curve"}
    lowest_target_value = min(fidelity_targets) * native_anchor_value
    highest_target_value = max(fidelity_targets) * native_anchor_value
    if smallest_q_reaching_target(numeric_qs_sorted, channel_curve_debiased_by_q, lowest_target_value) is None:
        return {"branch": BRANCH_LOW_FIDELITY_NOT_REACHED,
                "lowest_target_value": lowest_target_value,
                "maximum_common_channel_count": max(numeric_qs_sorted)}
    q_highest = smallest_q_reaching_target(numeric_qs_sorted, channel_curve_debiased_by_q, highest_target_value)
    if q_highest is None:
        return {"branch": BRANCH_HIGH_FIDELITY_NOT_REACHED,
                "highest_target_value": highest_target_value,
                "maximum_common_channel_count": max(numeric_qs_sorted)}
    small_channel_threshold = small_channel_fraction * native_full_median_q
    if q_highest <= small_channel_threshold:
        return {"branch": BRANCH_RECOVERABLE_SMALL, "q_reaching_highest_target": q_highest,
                "small_channel_threshold": small_channel_threshold}
    return {"branch": BRANCH_REQUIRES_MORE_THAN_SMALL, "q_reaching_highest_target": q_highest,
            "small_channel_threshold": small_channel_threshold}


# =======================================================================================================
# Supplementary diagnostic -- the associated axis (leading_eigenvector of _unit_residual_matrix's output),
# reused unchanged, reported only where this project already establishes it is defined (AXIS_DEFINED_
# CORPORA). Descriptive only: no null, no trial ladder, no fidelity target, no branch -- out of scope for
# the primary decision rule above, stated as such in its own output block.
# =======================================================================================================

def axis_alignment_recovery(activity_full: np.ndarray, channel_rungs_numeric: list[int], n_draws: int,
                             seed_tag: str) -> dict:
    n_trials_full, n_units_full = activity_full.shape
    ref_idx, _score_idx = reference_score_split(np.arange(n_trials_full), f"{seed_tag}|axis_split")
    activity_ref = activity_full[ref_idx]
    full_rows = _residual_rows(activity_ref)
    if full_rows["n_kept"] < 2:
        return {"status": "not_computable", "reason": "fewer than 2 reference trials with a defined residual"}
    R_full, _ = _unit_residual_matrix(full_rows)
    if R_full.shape[0] < 2 or R_full.shape[1] < 2:
        return {"status": "not_computable", "reason": "reference residual matrix too small"}
    axis_full = leading_eigenvector(R_full)
    by_q: dict[int, list[float]] = {}
    for q in channel_rungs_numeric:
        alignments = []
        for d in range(n_draws):
            cols = draw_index_subset(n_units_full, q, f"{seed_tag}|axis_q={q}|draw={d}|units")
            sub_full_axis = axis_full[cols]
            norm = np.linalg.norm(sub_full_axis)
            if norm == 0.0:
                continue
            rows = _residual_rows(activity_ref[:, cols])
            if rows["n_kept"] < 2:
                continue
            R_sub, _ = _unit_residual_matrix(rows)
            if R_sub.shape[0] < 2 or R_sub.shape[1] < 2:
                continue
            axis_sub = leading_eigenvector(R_sub)
            alignments.append(abs(float(np.dot(axis_sub, sub_full_axis / norm))))
        by_q[q] = alignments
    return {"status": "computed", "n_reference_trials": int(len(ref_idx)), "by_q_alignments": by_q}


# =======================================================================================================
# Corpus-level orchestration
# =======================================================================================================

def run_corpus_recovery(sessions: list[dict], corpus_key: str, compute_axis: bool) -> dict:
    pooled: dict = {}
    per_session_rows: list[dict] = []
    n_excluded_too_few_units = 0
    n_excluded_too_few_trials = 0
    native_units_seen: list[int] = []
    native_trials_seen: list[int] = []
    axis_by_q: dict[int, list[float]] = {}
    n_axis_sessions_computed = 0
    independent_by_q: dict = {}

    eligible_entries: list[tuple[dict, np.ndarray]] = []
    for entry in sessions:
        activity = np.asarray(entry["activity_by_unit"], dtype=float)
        n_trials_full, n_units_full = activity.shape
        if n_units_full < MIN_UNITS_FOR_A_SESSION:
            n_excluded_too_few_units += 1
            per_session_rows.append({"session": entry["session"], "status": "excluded_too_few_units",
                                      "n_units_full": n_units_full})
            continue
        if n_trials_full < MIN_TRIALS_FOR_A_SESSION:
            n_excluded_too_few_trials += 1
            per_session_rows.append({"session": entry["session"], "status": "excluded_too_few_trials",
                                      "n_trials_full": n_trials_full})
            continue
        eligible_entries.append((entry, activity))
        native_units_seen.append(n_units_full)
        native_trials_seen.append(n_trials_full)

    numeric_channel_rungs = common_channel_rungs(native_units_seen, CHANNEL_RUNGS_UNIVERSAL)
    native_calibration_trials_seen = [int(np.ceil(n / 2)) for n in native_trials_seen]
    numeric_trial_rungs = common_channel_rungs(native_calibration_trials_seen, CALIBRATION_TRIAL_RUNGS)
    session_summaries_all: list[dict] = []
    for entry, activity in eligible_entries:
        n_trials_full, n_units_full = activity.shape
        channel_rungs = numeric_channel_rungs + [NATIVE_LABEL]
        trial_rungs = numeric_trial_rungs + [NATIVE_LABEL]
        seed_tag = f"minimum_recording_specification|{corpus_key}|{entry['session']}"
        cells = session_joint_cells(activity, channel_rungs, trial_rungs, N_UNIT_DRAWS, N_TRIAL_DRAWS, seed_tag)
        independent = independent_reference_cells(activity, channel_rungs, N_UNIT_DRAWS, seed_tag)
        for q_label, arrs in independent.items():
            obs, null = arrs["observed"], arrs["null"]
            finite = np.isfinite(obs) & np.isfinite(null)
            independent_by_q.setdefault(q_label, []).extend((obs[finite] - null[finite]).tolist())

        by_q_native_trial: dict[int, float] = {}
        joint_debiased_median: dict[str, dict[str, float]] = {}
        native_debiased_median = np.nan
        for (t_label, q_label), arrs in cells.items():
            obs, null = arrs["observed"], arrs["null"]
            finite = np.isfinite(obs) & np.isfinite(null)
            debiased = obs[finite] - null[finite]
            key = (t_label, q_label)
            pooled.setdefault(key, {"observed": [], "null": [], "debiased": []})
            pooled[key]["observed"].extend(obs[np.isfinite(obs)].tolist())
            pooled[key]["null"].extend(null[np.isfinite(null)].tolist())
            pooled[key]["debiased"].extend(debiased.tolist())
            if debiased.size:
                joint_debiased_median.setdefault(str(t_label), {})[str(q_label)] = float(np.median(debiased))
            if t_label == NATIVE_LABEL and q_label != NATIVE_LABEL and debiased.size:
                by_q_native_trial[q_label] = float(np.median(debiased))
            if t_label == NATIVE_LABEL and q_label == NATIVE_LABEL and debiased.size:
                native_debiased_median = float(np.median(debiased))
        missing_numeric_rungs = [q for q in numeric_channel_rungs if q not in by_q_native_trial]
        is_complete = bool(np.isfinite(native_debiased_median) and not missing_numeric_rungs)
        missing_joint_cells = [
            [str(t), str(q)] for t in trial_rungs for q in channel_rungs
            if not np.isfinite(joint_debiased_median.get(str(t), {}).get(str(q), np.nan))
        ]
        joint_complete = bool(np.isfinite(native_debiased_median) and not missing_joint_cells)
        session_summaries_all.append({
            "session": entry["session"], "by_q_debiased_median": by_q_native_trial,
            "native_debiased_median": native_debiased_median, "complete_common_grid": is_complete,
            "joint_debiased_median": joint_debiased_median, "complete_joint_grid": joint_complete,
            "n_units_full": n_units_full, "n_trials_full": n_trials_full,
        })
        per_session_rows.append({
            "session": entry["session"],
            "status": "computed_common_grid" if is_complete else "excluded_noncomputable_common_grid",
            "n_units_full": n_units_full, "n_trials_full": n_trials_full,
            "missing_numeric_channel_rungs": missing_numeric_rungs,
            "native_anchor_computable": bool(np.isfinite(native_debiased_median)),
            "complete_joint_grid": joint_complete,
            "missing_joint_cells": missing_joint_cells,
        })

        if compute_axis:
            axis_result = axis_alignment_recovery(activity, numeric_channel_rungs, N_AXIS_UNIT_DRAWS, seed_tag)
            if axis_result["status"] == "computed":
                n_axis_sessions_computed += 1
                for q, vals in axis_result["by_q_alignments"].items():
                    axis_by_q.setdefault(q, []).extend(vals)

    session_summaries = [s for s in session_summaries_all if s["complete_common_grid"]]
    joint_session_summaries = [s for s in session_summaries_all if s["complete_joint_grid"]]
    n_analysed = len(session_summaries)

    joint_table: dict = {}
    for (t_label, q_label), arrs in pooled.items():
        joint_table.setdefault(str(t_label), {})[str(q_label)] = {
            "observed": pool_cell(arrs["observed"]), "null": pool_cell(arrs["null"]),
            "debiased": pool_cell(arrs["debiased"]),
        }

    channel_only_debiased_by_q = {
        q: float(np.median([s["by_q_debiased_median"][q] for s in session_summaries]))
        for q in numeric_channel_rungs if session_summaries
    }
    native_anchor_value = (float(np.median([s["native_debiased_median"] for s in session_summaries]))
                           if session_summaries else None)
    numeric_qs_sorted = list(numeric_channel_rungs) if session_summaries else []
    native_full_median_q = (float(np.median([s["n_units_full"] for s in session_summaries]))
                            if session_summaries else None)

    channel_rung_accounting = {
        str(q): {
            "n_sessions_denominator": len(eligible_entries),
            "n_sessions_eligible": len(eligible_entries),
            "n_sessions_used": len(session_summaries),
            "n_sessions_not_used": len(eligible_entries) - len(session_summaries),
        }
        for q in numeric_channel_rungs
    }
    channel_rung_accounting[NATIVE_LABEL] = {
        "n_sessions_denominator": len(eligible_entries),
        "n_sessions_eligible": len(eligible_entries),
        "n_sessions_used": len(session_summaries),
        "n_sessions_not_used": len(eligible_entries) - len(session_summaries),
    }

    fidelity = (compute_fidelity_targets(
        {q: channel_only_debiased_by_q[q] for q in numeric_qs_sorted}, native_anchor_value, numeric_qs_sorted,
        FIDELITY_TARGETS, session_summaries, f"minimum_recording_specification|{corpus_key}")
        if (native_anchor_value is not None and numeric_qs_sorted) else {})

    branch = classify_corpus_branch(n_analysed, {q: channel_only_debiased_by_q[q] for q in numeric_qs_sorted},
                                     native_anchor_value, native_full_median_q, numeric_qs_sorted,
                                     FIDELITY_TARGETS, SMALL_CHANNEL_FRACTION, MIN_SESSIONS_FOR_CORPUS_CURVE)

    joint_specification = compute_joint_fidelity_specification(
        joint_session_summaries, numeric_trial_rungs, numeric_channel_rungs, FIDELITY_TARGETS,
        f"minimum_recording_specification|{corpus_key}|joint")

    axis_block = {"status": "not_in_scope_for_this_corpus"}
    if compute_axis:
        axis_block = {
            "status": "computed" if n_axis_sessions_computed else "not_computable",
            "n_sessions_computed": n_axis_sessions_computed,
            "by_q": {str(q): pool_cell(vals) for q, vals in sorted(axis_by_q.items())},
            "scope_note": (
                "Supplementary diagnostic only, out of scope for decision_rule_declared_before_fitting: "
                "cosine alignment (absolute value, since eigenvector sign is arbitrary) between the leading "
                "eigenvector of the unit-residual matrix estimated on the SAME held-out reference trials at "
                "the full channel set versus at a random channel subset, at the NATIVE trial rung only -- no "
                "matched null, no trial-count ladder, no fidelity target, no branch."
            ),
        }

    return {
        "n_sessions_loaded": len(sessions), "n_sessions_excluded_too_few_units": n_excluded_too_few_units,
        "n_sessions_excluded_too_few_trials": n_excluded_too_few_trials,
        "n_sessions_eligible_for_common_grid": len(eligible_entries),
        "n_sessions_analysed": n_analysed,
        "n_sessions_excluded_noncomputable_common_grid": len(eligible_entries) - n_analysed,
        "n_sessions_complete_joint_grid": len(joint_session_summaries),
        "n_sessions_excluded_noncomputable_joint_grid": len(eligible_entries) - len(joint_session_summaries),
        "per_session": per_session_rows,
        "native_full_unit_count": {
            "median": native_full_median_q,
            "min": min((s["n_units_full"] for s in session_summaries), default=None),
            "max": max((s["n_units_full"] for s in session_summaries), default=None),
        },
        "native_full_trial_count": {
            "median": (float(np.median([s["n_trials_full"] for s in session_summaries]))
                       if session_summaries else None),
            "min": min((s["n_trials_full"] for s in session_summaries), default=None),
            "max": max((s["n_trials_full"] for s in session_summaries), default=None),
        },
        "native_calibration_trial_count": {
            "median": (float(np.median([int(np.ceil(s["n_trials_full"] / 2)) for s in session_summaries]))
                       if session_summaries else None),
            "min": min((int(np.ceil(s["n_trials_full"] / 2)) for s in session_summaries), default=None),
            "max": max((int(np.ceil(s["n_trials_full"] / 2)) for s in session_summaries), default=None),
        },
        "common_numeric_channel_grid": numeric_channel_rungs,
        "common_numeric_calibration_trial_grid": numeric_trial_rungs,
        "channel_curve_session_accounting_by_q": channel_rung_accounting,
        "joint_channel_calibration_trial_table": joint_table,
        "channel_only_recovery_curve_debiased_median_by_q": {
            **{str(k): v for k, v in channel_only_debiased_by_q.items()}, NATIVE_LABEL: native_anchor_value,
        },
        "fidelity_targets": fidelity,
        "joint_fidelity_specification": joint_specification,
        "branch": branch,
        "independent_reference_diagnostic": {
            "status": "computed" if independent_by_q else "not_computable",
            "by_q_debiased": {str(q): pool_cell(values) for q, values in independent_by_q.items()},
            "n_target_reference_trials": "one_third_per_session",
            "n_candidate_reference_trials": "one_third_per_session",
            "n_score_trials": "one_third_per_session",
        },
        "supplementary_axis_alignment_diagnostic": axis_block,
    }


# =======================================================================================================
# Loading -- every loader reused unchanged from the shared corpus modules.
# =======================================================================================================

def _load_watters(root) -> tuple[list[dict], int, list[dict]]:
    loaded, refused = [], []
    seen = 0
    for session in iter_watters(root, bin_ms=100.0):
        seen += 1
        if session.get("status") != "loaded":
            refused.append({"session": session.get("session"), "reason": session.get("status")})
            continue
        loaded.append({"session": session["session"], "activity_by_unit": session["counts"].sum(axis=2)})
    return loaded, seen, refused


def _corpus_specs(root) -> list[dict]:
    panichello_dir = _panichello_directory(root)
    panichello_seen = len(list(panichello_dir.glob("*.mat"))) if panichello_dir is not None else 0
    panichello_loaded = _load_panichello_for_counting_noise_census(root)

    alm_dir = alm_data_directory(root)
    alm_seen = len(list(alm_dir.glob("*.mat"))) if alm_dir.is_dir() else 0
    alm_loaded = _load_alm_for_counting_noise_census(root)

    watters_loaded, watters_seen, watters_refused = _load_watters(root)

    specs = [
        {"corpus": "panichello_2024_macaque_lPFC_single_item", "sessions": panichello_loaded,
         "n_seen": panichello_seen, "n_loaded": len(panichello_loaded),
         "refused": [{"reason": "load_failed_or_absent"}] * (panichello_seen - len(panichello_loaded)),
         "compute_axis": "panichello_2024_macaque_lPFC_single_item" in AXIS_DEFINED_CORPORA},
        {"corpus": "inagaki_alm5_mouse_ALM", "sessions": alm_loaded, "n_seen": alm_seen,
         "n_loaded": len(alm_loaded),
         "refused": [{"reason": "load_alm_raw_session_returned_none"}] * (alm_seen - len(alm_loaded)),
         "compute_axis": "inagaki_alm5_mouse_ALM" in AXIS_DEFINED_CORPORA},
        {"corpus": "watters_2026_macaque_multi_object", "sessions": watters_loaded, "n_seen": watters_seen,
         "n_loaded": len(watters_loaded), "refused": watters_refused,
         "compute_axis": "watters_2026_macaque_multi_object" in AXIS_DEFINED_CORPORA},
    ]
    for dataset in HUMAN_CORPORA_FOR_THE_CENSUS:
        loaded = _load_human_for_counting_noise_census(root, dataset)
        seen = _human_seen_denominator(root, dataset)
        specs.append({
            "corpus": f"{dataset}_human", "sessions": loaded, "n_seen": seen, "n_loaded": len(loaded),
            "refused": [{"reason": "not_pooled_structure_or_other_shared_loader_exclusion"}] * (seen - len(loaded)),
            "compute_axis": f"{dataset}_human" in AXIS_DEFINED_CORPORA,
        })
    return specs


def _flush(output: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = output_path.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="JSON destination; defaults to the canonical results artifact.")
    parser.add_argument("--max-sessions-per-corpus", type=int, default=None,
                        help="Run a bounded smoke validation on the first loaded sessions of each corpus.")
    parser.add_argument("--skip-axis", action="store_true",
                        help="Skip the descriptive supplementary axis diagnostic.")
    args = parser.parse_args()
    if args.max_sessions_per_corpus is not None and args.max_sessions_per_corpus < 1:
        parser.error("--max-sessions-per-corpus must be positive")
    t0 = time.time()
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "How few simultaneously recorded channels, and how many prior calibration trials, a per-trial magnitude-"
            "invariant population state-deviation observable needs before it stops being readable above "
            "its own chance floor. Run separately, with no pooling and no ranking across corpora, on every "
            "corpus in this project where the deviation is defined on a per-trial per-unit spike tensor: a "
            "single-item macaque lateral prefrontal cortex recording, a mouse anterior lateral motor cortex "
            "recording, a multi-object-holding macaque recording, and two human single-unit maintenance-"
            "delay recordings. A supplementary, out-of-decision-rule diagnostic additionally reports the "
            "channel-count recoverability of the deviation's own associated low-dimensional axis, restricted "
            "to the two corpora where this project already establishes that axis is defined."
        ),
        "analysis_rule": ANALYSIS_RULE,
        "analysis_timing_disclosure": ANALYSIS_TIMING_DISCLOSURE,
        "null_floor_disclosure": NULL_FLOOR_DISCLOSURE,
        "power_scale_disclosure": POWER_SCALE_DISCLOSURE,
        "status": "running",
        "run_mode": "smoke_validation" if args.max_sessions_per_corpus is not None else "full_analysis",
        "output_path": str(args.output),
    }
    _flush(output, args.output)
    print("loading corpora...", flush=True)
    specs = _corpus_specs(root)

    corpora_out: dict = {}
    zero_drop: dict = {}
    for spec in specs:
        name = spec["corpus"]
        if args.max_sessions_per_corpus is not None:
            spec = {**spec, "sessions": spec["sessions"][:args.max_sessions_per_corpus]}
        print(f"[{name}] {spec['n_loaded']}/{spec['n_seen']} sessions loaded, running recovery ladder...",
              flush=True)
        corpora_out[name] = run_corpus_recovery(spec["sessions"], name, spec["compute_axis"] and not args.skip_axis)
        n_refused = spec["n_seen"] - spec["n_loaded"]
        zero_drop[name] = {
            "n_seen": spec["n_seen"], "n_loaded": spec["n_loaded"], "n_refused": n_refused,
            "refusal_reasons": {r: sum(1 for x in spec["refused"] if x.get("reason") == r)
                                 for r in sorted({x.get("reason") for x in spec["refused"]})} if spec["refused"] else {},
            "n_analysed": corpora_out[name]["n_sessions_analysed"],
            "n_excluded_by_this_modules_own_floors": (
                corpora_out[name]["n_sessions_excluded_too_few_units"]
                + corpora_out[name]["n_sessions_excluded_too_few_trials"]),
            "reconciles": bool(spec["n_seen"] == spec["n_loaded"] + n_refused),
        }
        output["corpora"] = corpora_out
        output["zero_drop_accounting"] = zero_drop
        _flush(output, args.output)
        print(f"[{name}] branch={corpora_out[name]['branch']['branch']} "
              f"n_analysed={corpora_out[name]['n_sessions_analysed']} elapsed={time.time() - t0:.0f}s", flush=True)

    output["git_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output, args.output)
    print(json.dumps({"status": "complete", "branches": {k: v["branch"]["branch"] for k, v in corpora_out.items()},
                       "wall_clock_s": output["wall_clock_s"]}, indent=2))


if __name__ == "__main__":
    main()
