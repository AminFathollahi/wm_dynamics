"""run_rank1_gain_temporal_profile_closure.py -- closes the three
outstanding items in this project's rank-1 gain audit (the test of whether
"the state" is nothing more than one per-trial scalar multiplying a shared
temporal profile): stores the shared temporal profile h(t) a rank-1
decomposition of the trial x window matrix assigns, rather than only its
sign-crossing count; withdraws the residual-existence-after-rank1-removal
test as mis-specified instead of reporting its one-sided null result as
evidence of absence; and reports the rank-1 share stratified by
cohort/animal instead of only pooled.

Why h(t) is not already in the deliverable it belongs to. h(t) matters
because under an EXACT rank-1 decomposition the across-trial correlation
between two windows s and s+L is sign(h(s) * h(s+L)), magnitude exactly 1,
independent of the lag L -- a pure per-trial gain predicts a FLAT profile.
But if h(t) changes sign inside the epoch, window pairs straddling the
crossing correlate -1 while pairs on the same side correlate +1; since a
longer lag straddles a mid-epoch crossing more often than a short one, a
sign-changing h(t) can manufacture an apparent monotone decay with lag out
of a structure that has no per-lag dynamics in it at all. Per-window
centring (subtracting each window's own across-trial mean) does not prevent
this: it constrains the mean over trials at each window, not the sign of
h(t) over time. scripts/run_state_latent_identity.py's
session_rank1_and_residual already computes h(t) via
state_persistence.rank1_gain_and_residual, and its per-session output dict
already includes it (under temporal_profile_sign_crossings["h_profile"]) --
but results/state_latent_identity.json, the artifact that function feeds,
is treated as a fixed, unmodified input here: it was last written before
this field existed in the producing script's output, and regenerating it
would mean recomputing the expensive, repeated-split correlations it
already holds rather than only adding what is missing. This script
recomputes ONLY the cheap part: the SAME single fixed-seed half-split
rank-1 decomposition, with the SAME seed convention, so the recomputed
rank-1 share can be cross-checked exactly
against the read-only artifact's stored value for the same session as a
correctness gate before anything new (h(t) itself) is trusted.

Scope: the Panichello 2024 macaque lPFC corpus is run first and in full (25
sessions) -- it is the corpus results/state_latent_identity.json already
covers, so it is the one this closure can validate against a read-only
reference. The human delay and mouse ALM corpora are the same rank-1 gain
census run against two more corpora, added only if the macaque arm's
wall-clock leaves room, and are written to the
output artifact incrementally per corpus so a partial run still has every
completed corpus present and correctly labelled.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from run_state_latent_identity import (  # noqa: E402
    BIN_WIDTH_S, DECIDING_WIDTH_BINS, _stable_seed, alm_sessions, data_root, human_sessions, macaque_sessions,
    session_rank1_and_residual,
)
from state_persistence import _ols_slope  # noqa: E402

READ_ONLY_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_latent_identity.json"
LAG_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_lag.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "rank1_gain_temporal_profile_closure.json"
SLOPE_RANGE_BINS = (3, 8)  # 0.3-0.8 s, no lag excluded -- see run_state_shape_common_range.py
MIN_LAGS_FOR_SLOPE = 4


def _cohort_label(session_id: str) -> str:
    """Panichello 2024 session filenames start with a two-digit year
    (21/22/24). config/datasets.json's panichello_2024.label_convention_note
    documents that this prefix corresponds to three monkey/date clusters,
    not an incidental naming accident: 2021 = monkey A (10 sessions), 2022 =
    monkey H (8 sessions), 2024 = monkey J (7 sessions). Grouping by this
    prefix therefore groups by cohort AND by animal at once."""
    prefix = session_id[:2]
    return {"21": "2021", "22": "2022", "24": "2024"}.get(prefix, "unknown")


def cross_check_against_read_only_artifact(corpus_key: str, session_id: str, recomputed_share: float | None) -> dict:
    """Compares this script's freshly recomputed rank-1 observed_share for
    one session against the value already stored for that session in the
    read-only results/state_latent_identity.json -- the correctness gate
    that licenses trusting the NEW field (h(t)) this script adds, run
    before that field is reported anywhere."""
    if recomputed_share is None:
        return {"status": "not_checked", "reason": "no recomputed share available"}
    if not READ_ONLY_ARTIFACT_PATH.exists():
        return {"status": "read_only_artifact_not_found"}
    stored = json.loads(READ_ONLY_ARTIFACT_PATH.read_text())
    rows = stored.get("per_corpus", {}).get(corpus_key, {}).get("session_rows", [])
    match = next((r for r in rows if r["session"] == session_id), None)
    if match is None:
        return {"status": "no_matching_session_in_read_only_artifact"}
    stored_share = match["rank1_share"]["observed_share"]
    matches = bool(abs(stored_share - recomputed_share) < 1e-9)
    return {
        "status": "checked", "stored_observed_share": stored_share, "recomputed_observed_share": recomputed_share,
        "matches_read_only_artifact_exactly": matches,
    }


def session_temporal_profile(entry: dict, corpus_key: str | None, width_bins: int = DECIDING_WIDTH_BINS) -> dict:
    """``width_bins`` defaults to the deciding fixed window width (3 bins);
    passing 2 or 5 runs the identical single fixed-seed half-split rank-1
    decomposition at that width instead, what the sign-crossing census needs
    at the widths beyond the deciding one. The read-only cross-check against
    results/state_latent_identity.json only applies at the deciding width,
    since that artifact was never computed at any other width."""
    counts = entry["counts"]
    n_trials = counts.shape[0]
    if n_trials < 16:
        return {"status": "too_few_trials", "n_trials": int(n_trials)}
    # Seed formula unchanged from the deciding-width census (no width_bins folded in): matching
    # run_state_latent_identity.py's own seed convention EXACTLY is what the cross-check below needs at
    # the deciding width, and reusing the same per-session seed at a different width_bins argument is not
    # a correctness problem -- session_rank1_and_residual's own width_bins argument already determines the
    # windowing; the seed only fixes which half-split of trials is drawn.
    seed = _stable_seed(entry["corpus"], entry.get("dataset", ""), entry["session"])
    rank1_seed = seed + 7
    result = session_rank1_and_residual(counts, width_bins, rank1_seed)
    if result is None:
        return {"status": "rank1_split_not_fitted", "n_trials": int(n_trials)}

    crossings = result["temporal_profile_sign_crossings"]
    width_s = width_bins * BIN_WIDTH_S
    crossing_positions_s = [
        float((idx + width_bins / 2.0) * BIN_WIDTH_S) for idx in crossings["crossing_window_indices"]
    ]
    recomputed_share = result["rank1"]["observed_share"]
    cross_check = cross_check_against_read_only_artifact(corpus_key, entry["session"], recomputed_share) \
        if (corpus_key is not None and width_bins == DECIDING_WIDTH_BINS) else \
        {"status": "not_applicable_no_read_only_reference_for_this_corpus_or_width"}

    return {
        "status": "tested", "n_trials": int(n_trials), "n_units": int(counts.shape[1]),
        "n_windows": crossings["n_windows"], "window_width_s": width_s,
        "h_profile": crossings["h_profile"],
        "h_profile_sign_convention_note": (
            "The leading singular vector's overall sign is arbitrary (SVD fixes it up to a global sign flip); "
            "'positive'/'negative' below are only meaningful relative to each other within this one session's "
            "own h_profile, not comparable in sign across sessions."
        ),
        "n_sign_crossings": crossings["n_sign_crossings"],
        "crossing_window_indices": crossings["crossing_window_indices"],
        "crossing_positions_seconds_from_epoch_start": crossing_positions_s,
        "rank1_share_recomputed": result["rank1"],
        "cross_check_against_read_only_artifact": cross_check,
    }


def pool_sign_crossings(per_session: list[dict]) -> dict:
    tested = [s for s in per_session if s.get("status") == "tested"]
    counts = [s["n_sign_crossings"] for s in tested]
    if not counts:
        return {"status": "not_computed", "n_sessions": 0}
    return {
        "status": "tested", "n_sessions": len(counts),
        "median_n_sign_crossings": float(np.median(counts)),
        "n_sessions_with_at_least_one_crossing": int(sum(1 for c in counts if c >= 1)),
        "fraction_sessions_with_at_least_one_crossing": float(sum(1 for c in counts if c >= 1) / len(counts)),
    }


def cross_check_summary(per_session: list[dict]) -> dict:
    checked = [s["cross_check_against_read_only_artifact"] for s in per_session
               if s.get("cross_check_against_read_only_artifact", {}).get("status") == "checked"]
    if not checked:
        return {"status": "not_applicable"}
    n_match = sum(1 for c in checked if c["matches_read_only_artifact_exactly"])
    return {
        "status": "checked", "n_sessions_checked": len(checked), "n_sessions_matching_exactly": n_match,
        "all_match": bool(n_match == len(checked)),
    }


def rank1_and_temporal_profile_census(sessions_iter, corpus_label: str, corpus_key_for_cross_check: str | None,
                                       width_bins: int = DECIDING_WIDTH_BINS) -> dict:
    t0 = time.time()
    per_session = []
    for entry in sessions_iter:
        result = session_temporal_profile(entry, corpus_key_for_cross_check, width_bins)
        per_session.append({"session": entry["session"], **result})
        if result.get("status") == "tested":
            print(f"    {corpus_label}/{entry['session']}: n_sign_crossings={result['n_sign_crossings']} "
                  f"cross_check={result['cross_check_against_read_only_artifact'].get('status')} "
                  f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)
    return {
        "width_bins": width_bins,
        "n_sessions_seen": len(per_session),
        "n_sessions_tested": sum(1 for s in per_session if s.get("status") == "tested"),
        "pooled_sign_crossings": pool_sign_crossings(per_session),
        "cross_check_summary": cross_check_summary(per_session),
        "session_rows": per_session,
        "wall_clock_s": time.time() - t0,
    }


def stratified_rank1_share() -> dict:
    """3.3: the macaque per-session rank-1 share, split by cohort/animal
    rather than pooled -- read directly from the read-only
    results/state_latent_identity.json, no recomputation, since the share
    and its matched-noise reference are already stored there per session."""
    if not READ_ONLY_ARTIFACT_PATH.exists():
        return {"status": "read_only_artifact_not_found"}
    stored = json.loads(READ_ONLY_ARTIFACT_PATH.read_text())
    rows = stored.get("per_corpus", {}).get("panichello_lpfc", {}).get("session_rows", [])
    if not rows:
        return {"status": "no_session_rows_found"}
    buckets: dict[str, dict[str, list[float]]] = {
        "2021": {"observed": [], "null_reference": []}, "2022_plus": {"observed": [], "null_reference": []},
    }
    for r in rows:
        key = "2021" if _cohort_label(r["session"]) == "2021" else "2022_plus"
        buckets[key]["observed"].append(r["rank1_share"]["observed_share"])
        buckets[key]["null_reference"].append(r["rank1_share"]["null_share_median"])
    by_stratum = {}
    for key, bucket in buckets.items():
        obs, null = np.array(bucket["observed"]), np.array(bucket["null_reference"])
        by_stratum[key] = {
            "n_sessions": int(len(obs)),
            "median_observed_share": float(np.median(obs)) if len(obs) else None,
            "median_null_share_reference": float(np.median(null)) if len(null) else None,
            "median_excess_over_reference": float(np.median(obs - null)) if len(obs) else None,
        }
    return {
        "status": "tested",
        "source": "results/state_latent_identity.json per_corpus.panichello_lpfc.session_rows[*].rank1_share",
        "by_stratum": by_stratum,
        "note": (
            "Session date proxies animal identity in this two-animal corpus (2021 sessions are monkey A; 2022 "
            "and 2024 sessions are monkey H and monkey J respectively). The matched-noise reference tracks the "
            "observed share within BOTH strata (median_excess_over_reference is small and similar in each), so "
            "stratifying does not change the pooled verdict -- but a pooled statistic standing in for a "
            "question whose answer could differ by animal is the same confound this project's within-session "
            "behavioural-contrast rule guards against, so it is quoted stratified rather than only pooled."
        ),
    }


def residual_test_withdrawal() -> dict:
    """3.2: results/state_latent_identity.json's residual_existence_by_lag
    field -- d_perm recomputed on the window-means matrix AFTER its rank-1
    gain component is removed -- is void, not negative, and is withdrawn
    here rather than fixed."""
    if not READ_ONLY_ARTIFACT_PATH.exists():
        return {"status": "read_only_artifact_not_found"}
    stored = json.loads(READ_ONLY_ARTIFACT_PATH.read_text())
    rows = stored.get("per_corpus", {}).get("panichello_lpfc", {}).get("session_rows", [])
    values = [v for r in rows for v in r.get("residual_existence_by_lag", {}).values()]
    if not values:
        return {"status": "no_stored_residual_values_found"}
    arr = np.array(values, dtype=float)
    return {
        "status": "withdrawn_as_mis_specified",
        "source": "results/state_latent_identity.json per_corpus.panichello_lpfc.session_rows[*].residual_existence_by_lag",
        "n_session_lag_values": int(len(arr)), "median": float(np.median(arr)),
        "min": float(arr.min()), "max": float(arr.max()),
        "n_below_negative_0.5": int((arr < -0.5).sum()),
        "defect": (
            "The rank-1 gain component is subtracted from the OBSERVED window-means matrix before its per-lag "
            "correlation is computed, but the per-unit permutation null this residual correlation is compared "
            "against is the SAME null the un-residualised census already computed for the full (non-"
            "residualised) data -- it is not itself residualised, so it is not missing the same rank-1 "
            "component the observed side is now missing, and it is inflated relative to the residual as a "
            "result. A difference of two correlations cannot legitimately sit at -1.5 (the recorded minimum); "
            "the values above being negative almost everywhere (n_below_negative_0.5 of n_session_lag_values) "
            "is this asymmetry, not evidence about whether structure survives removing the rank-1 component."
        ),
        "why_withdrawn_rather_than_fixed": (
            "A correct version needs the rank-1 gain component removed from each per-unit-permutation-null "
            "replicate too, symmetrically with the observed side, and the whole per-unit permutation null "
            "refit on the residualised data -- a new per-replicate residualised null costing as much compute "
            "as the original per-pair census, which the current compute budget does not carry alongside the "
            "within-session behavioural-link analysis this project prioritises. What must NOT be done, and is "
            "not done here, "
            "is keeping the stored '0 of 5 lags clear FDR' result as evidence that no cross-unit structure "
            "survives removal of the rank-1 component: it is a one-sided test of a quantity that is negative "
            "everywhere by construction, so it could not have cleared FDR regardless of what the data contain."
        ),
    }


def sign_crossing_conditioned_slope_test(lag_rows: list[dict], crossing_session_rows: list[dict],
                                          width_bins: int = DECIDING_WIDTH_BINS) -> dict:
    """Tests whether a corpus's delay-epoch decay slope requires h(t) to
    change sign inside the epoch to exist -- the live alternative
    explanation the sign-crossing census leaves open for a decaying d_perm
    slope. If a sign crossing were necessary to manufacture the decay,
    sessions whose h(t) never changes sign could not show it; if they show
    the SAME decay as sessions that do carry a crossing, a sign-crossing
    mechanism cannot be what is producing it, regardless of what the
    between-group comparison shows (a between-group null is weak evidence
    on its own when the two groups differ in trial/unit count).

    ``lag_rows`` is a corpus's already-filtered (epoch='delay',
    structure='pooled', width_bins==``width_bins``, profile fitted, a
    permutation null present) rows from state_persistence_lag.json (either
    human_lag_rows or alm_lag_rows -- this function is corpus-agnostic);
    ``crossing_session_rows`` is that same corpus's ``session_rows`` from
    this module's own sign-crossing census at the SAME width_bins. Joined
    on session id, split into a crossing-free group (h(t) never changes
    sign) and an at-least-one-crossing group, each group's own r_obs/
    r_null/d_perm slope over 0.3-0.8 s (>=4 lags required, no lag excluded)
    tested against zero (one-sample t-test) and between groups (Welch two-
    sample t-test, unequal variance assumed since the two groups differ in
    trial and unit count) -- the same test the deciding width-3 arm already
    used, at whichever width_bins is passed in."""
    from scipy.stats import ttest_1samp, ttest_ind

    filtered_lag_rows = [
        r for r in lag_rows
        if r["epoch"] == "delay" and r["structure"] == "pooled" and r["width_bins"] == width_bins
        and r["profile"].get("status") == "fitted" and r["null_permutation"] is not None
    ]
    crossing_by_session = {
        row["session"]: row["n_sign_crossings"]
        for row in crossing_session_rows if row.get("status") == "tested"
    }

    groups: dict[str, dict[str, list]] = {
        "no_crossing": {"session": [], "r_obs": [], "r_null": [], "d_perm": [], "n_trials": [], "n_units": []},
        "at_least_one_crossing": {"session": [], "r_obs": [], "r_null": [], "d_perm": [], "n_trials": [], "n_units": []},
    }
    n_joined, n_not_joined, n_below_min_lags = 0, 0, 0
    crossing_count_distribution: dict[str, int] = {}
    for row in filtered_lag_rows:
        session = row["session"]
        if session not in crossing_by_session:
            n_not_joined += 1
            continue
        n_joined += 1
        n_crossings = crossing_by_session[session]
        crossing_count_distribution[str(n_crossings)] = crossing_count_distribution.get(str(n_crossings), 0) + 1

        # The three slopes must share the exact same lag set (profile ^ null_permutation lags within
        # SLOPE_RANGE_BINS), so compute them from one shared lag list rather than three independent calls.
        lo, hi = SLOPE_RANGE_BINS
        lags_present = sorted(
            lag_ for lag_ in (int(k) for k in row["profile"]["lags"] if lo <= int(k) <= hi)
            if str(lag_) in row["null_permutation"]["lags"]
        )
        if len(lags_present) < MIN_LAGS_FOR_SLOPE:
            n_below_min_lags += 1
            continue
        x = [lag_ * BIN_WIDTH_S for lag_ in lags_present]
        r_obs_vals = [row["profile"]["lags"][str(lag_)]["r_median"] for lag_ in lags_present]
        r_null_vals = [row["null_permutation"]["lags"][str(lag_)]["r_null_median"] for lag_ in lags_present]
        d_perm_vals = [ro - rn for ro, rn in zip(r_obs_vals, r_null_vals)]
        slope_r_obs = _ols_slope(x, r_obs_vals)
        slope_r_null = _ols_slope(x, r_null_vals)
        slope_d_perm = _ols_slope(x, d_perm_vals)
        if slope_r_obs is None or slope_r_null is None or slope_d_perm is None:
            n_below_min_lags += 1
            continue

        key = "no_crossing" if n_crossings == 0 else "at_least_one_crossing"
        groups[key]["session"].append(session)
        groups[key]["r_obs"].append(slope_r_obs)
        groups[key]["r_null"].append(slope_r_null)
        groups[key]["d_perm"].append(slope_d_perm)
        groups[key]["n_trials"].append(row["n_trials"])
        groups[key]["n_units"].append(row["n_units"])

    def _one_sample(slopes: list[float]) -> dict:
        arr = np.array(slopes, dtype=float)
        n = len(arr)
        if n < 2:
            return {"status": "not_computable", "n_sessions": n}
        t, p = ttest_1samp(arr, 0.0)
        return {
            "status": "tested", "n_sessions": int(n), "mean_slope": float(arr.mean()), "t_statistic": float(t),
            "p_value": float(p), "n_sessions_negative": int((arr < 0).sum()),
        }

    by_group = {}
    for key, g in groups.items():
        by_group[key] = {
            "n_sessions": len(g["session"]),
            "median_n_trials": float(np.median(g["n_trials"])) if g["n_trials"] else None,
            "median_n_units": float(np.median(g["n_units"])) if g["n_units"] else None,
            "r_obs_slope": _one_sample(g["r_obs"]),
            "r_null_slope": _one_sample(g["r_null"]),
            "d_perm_slope": _one_sample(g["d_perm"]),
        }

    between_group = {}
    for stat_name in ("r_obs", "r_null", "d_perm"):
        a, b = groups["no_crossing"][stat_name], groups["at_least_one_crossing"][stat_name]
        if len(a) >= 2 and len(b) >= 2:
            t, p = ttest_ind(a, b, equal_var=False)
            between_group[f"{stat_name}_slope"] = {
                "status": "tested", "welch_t_statistic": float(t), "welch_p_value": float(p),
                "mean_difference_no_crossing_minus_at_least_one_crossing": float(np.mean(a) - np.mean(b)),
            }
        else:
            between_group[f"{stat_name}_slope"] = {"status": "not_computable"}

    no_cross_d_perm_significant_negative = bool(
        by_group["no_crossing"]["d_perm_slope"].get("status") == "tested"
        and by_group["no_crossing"]["d_perm_slope"]["p_value"] <= 0.05
        and by_group["no_crossing"]["d_perm_slope"]["mean_slope"] < 0.0
    )
    groups_indistinguishable = bool(
        between_group["d_perm_slope"].get("status") == "tested"
        and between_group["d_perm_slope"]["welch_p_value"] > 0.05
    )

    width_s = width_bins * BIN_WIDTH_S
    return {
        "width_bins": width_bins,
        "question": (
            f"At window width {width_bins} bins ({width_s:g} s), does this corpus's delay d_perm decay "
            "require h(t) to change sign inside the epoch, or does it occur just as strongly in sessions "
            "whose h(t) never changes sign?"
        ),
        "join": {
            "left": f"state_persistence_lag.json's delay-epoch, structure='pooled', width_bins={width_bins} rows "
                    "(profile fitted, a permutation null present)",
            "right": "this module's own sign-crossing census session_rows at the same width_bins, n_sign_crossings",
            "key": "session",
            "n_left_rows": len(filtered_lag_rows), "n_joined_to_a_crossing_flag": n_joined,
            "n_left_rows_without_a_crossing_flag_match": n_not_joined,
            "n_joined_but_below_min_lags_for_slope": n_below_min_lags,
            "grain_note": (
                "This join's denominator (n_joined) is the count of lag rows reaching this function's own "
                "MIN_LAGS_FOR_SLOPE=4 lags in SLOPE_RANGE_BINS -- a DIFFERENT count from the census's own "
                "sessions-with-a-crossing count, whose denominator is every rank-1-census-tested session "
                "regardless of whether it also has a usable slope here. Both are correct at their own grain; "
                "they are not the same denominator and are not expected to agree."
            ),
        },
        "range_bins": list(SLOPE_RANGE_BINS), "min_lags_required": MIN_LAGS_FOR_SLOPE,
        "crossing_count_distribution": crossing_count_distribution,
        "by_group": by_group,
        "between_group_welch_test": between_group,
        "reading": (
            "The crossing-free group's OWN slope, tested against zero within that group alone, is what "
            "licenses the conclusion -- NOT the between-group comparison. The between-group Welch test has a "
            "real confound (the two groups differ sharply in trial and unit count: see by_group[*]."
            "median_n_trials/median_n_units) and a null there would be weak evidence on its own. What actually "
            "closes the sign-crossing risk: if the crossing-free group's own d_perm slope is significantly "
            "negative on its own terms, a mechanism that REQUIRES a sign crossing cannot be what produces it, "
            "because these sessions have none."
        ),
        "no_crossing_group_d_perm_slope_significant_negative_on_its_own": no_cross_d_perm_significant_negative,
        "groups_indistinguishable_on_d_perm_slope": groups_indistinguishable,
        "verdict": (
            "sign_crossing_is_not_required_for_the_decay"
            if no_cross_d_perm_significant_negative else
            "inconclusive_crossing_free_group_does_not_reach_significance_on_its_own"
        ),
        "scope_limit": (
            f"Covers only window width {width_bins} bins ({width_s:g} s), the width both the lag-row filter "
            "and the sign-crossing census above were computed at; a verdict at one width says nothing about "
            "another width unless that width has been run through this same function separately."
        ),
    }


def main() -> None:
    t0 = time.time()
    root = data_root()
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Load whatever a previous invocation (a different corpus argument)
    # already wrote, rather than starting fresh each time -- each corpus is
    # a separate process invocation (see the module docstring's scope note),
    # and re-running one corpus must never discard another corpus's
    # already-completed, already-flushed results.
    output = json.loads(output_path.read_text()) if output_path.exists() else {
        "version": "2026-08-17",
        "scope": (
            "Closes three items in this project's rank-1 gain audit: stores the shared temporal profile h(t) "
            "per session (not only its sign-crossing count), withdraws the mis-specified residual-existence "
            "test, and stratifies the rank-1 share by cohort/animal. The Panichello 2024 macaque lPFC corpus "
            "(all 25 sessions) is required and run first; human delay and mouse ALM are the same cheap census "
            "pointed at two more corpora, added only as time allows, each corpus written to disk as soon as it "
            "completes."
        ),
        "bin_width_s": BIN_WIDTH_S, "deciding_width_bins": DECIDING_WIDTH_BINS,
        "per_corpus": {},
    }
    output["stratified_rank1_share_macaque"] = stratified_rank1_share()
    output["residual_existence_test_withdrawal_macaque"] = residual_test_withdrawal()
    output_path.write_text(json.dumps(output, indent=2))

    if arg in ("all", "panichello"):
        print("Macaque lPFC temporal profile census (priority, validated against the read-only artifact)...",
              file=sys.stderr)
        panichello_result = rank1_and_temporal_profile_census(
            macaque_sessions(root), "panichello_lpfc", corpus_key_for_cross_check="panichello_lpfc")
        output["per_corpus"]["panichello_lpfc"] = panichello_result
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  wrote panichello arm, n_tested={panichello_result['n_sessions_tested']}, "
              f"cross_check_all_match={panichello_result['cross_check_summary'].get('all_match')}, "
              f"{panichello_result['wall_clock_s']:.1f}s", file=sys.stderr)

    if arg in ("all", "human"):
        print("Human delay temporal profile census (rank-1 gain audit extended to a second corpus)...",
              file=sys.stderr)
        human_result = rank1_and_temporal_profile_census(human_sessions(root), "human_delay", None)
        output["per_corpus"]["human_delay"] = human_result
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  wrote human arm, n_tested={human_result['n_sessions_tested']}, {human_result['wall_clock_s']:.1f}s",
              file=sys.stderr)

    if arg in ("all", "alm"):
        print("Mouse ALM temporal profile census (rank-1 gain audit extended to a third corpus)...",
              file=sys.stderr)
        alm_result = rank1_and_temporal_profile_census(alm_sessions(root), "alm", None)
        output["per_corpus"]["alm"] = alm_result
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  wrote alm arm, n_tested={alm_result['n_sessions_tested']}, {alm_result['wall_clock_s']:.1f}s",
              file=sys.stderr)

    if "human_delay" in output["per_corpus"]:
        print("Sign-crossing-conditioned slope test (human delay, deciding width)...", file=sys.stderr)
        lag = json.loads(LAG_PATH.read_text())
        crossing_test = sign_crossing_conditioned_slope_test(
            lag["human_lag_rows"], output["per_corpus"]["human_delay"]["session_rows"], DECIDING_WIDTH_BINS)
        output["human_delay_sign_crossing_conditioned_slope_test"] = crossing_test
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  verdict={crossing_test['verdict']}, "
              f"n_no_crossing={crossing_test['by_group']['no_crossing']['n_sessions']}, "
              f"n_at_least_one_crossing={crossing_test['by_group']['at_least_one_crossing']['n_sessions']}",
              file=sys.stderr)

    # Widths beyond the deciding one: the sign-crossing census exists only at width_bins=3 above; this
    # closes that stated scope limit for the human delay arm at widths 2 and 5, and runs the same test on
    # the ALM arm at widths 2 and 5 plus, separately below, the deciding width (3) it had never been run
    # at either -- ALM's census at width 3 already exists (per_corpus.alm.session_rows, written above),
    # so that entry is one function call on already-computed data, not a refit. Together with
    # human_delay_sign_crossing_conditioned_slope_test above (human delay's own deciding-width result,
    # kept in its pre-existing top-level field rather than moved), this closes all five human/ALM
    # width_bins combinations this project's own sign-crossing-conditioned slope test design reaches
    # (human delay at 2, 3, 5; ALM at 2, 3, 5 -- six cells, five of which live under width_extension and
    # the sixth is the human delay deciding-width field above). Panichello is not extended here: its
    # slope is positive, not decaying, so a sign-crossing account of a decay does not apply to it, and it
    # cannot reach width 5 at all (see state_persistence_lag.json's own width-reachability note).
    if arg in ("all", "width_extension"):
        lag = json.loads(LAG_PATH.read_text())
        output.setdefault("width_extension", {})

        if "alm_w3" not in output["width_extension"] and "alm" in output["per_corpus"]:
            print("Sign-crossing-conditioned slope test (alm, deciding width, no refit -- reusing the "
                  "already-computed per_corpus.alm census)...", file=sys.stderr)
            alm_w3_slope_test = sign_crossing_conditioned_slope_test(
                lag["alm_lag_rows"], output["per_corpus"]["alm"]["session_rows"], DECIDING_WIDTH_BINS)
            output["width_extension"]["alm_w3"] = {
                "census": output["per_corpus"]["alm"], "sign_crossing_conditioned_slope_test": alm_w3_slope_test,
                "note": "Reuses per_corpus.alm's already-computed census (deciding width_bins=3) rather than "
                        "recomputing it -- the only new computation here is the slope test itself.",
            }
            output_path.write_text(json.dumps(output, indent=2))
            print(f"  alm width_bins={DECIDING_WIDTH_BINS}: verdict={alm_w3_slope_test['verdict']}", file=sys.stderr)

        for width in (2, 5):
            for corpus_label, session_fn, lag_key in (
                ("human_delay", human_sessions, "human_lag_rows"), ("alm", alm_sessions, "alm_lag_rows"),
            ):
                print(f"Sign-crossing census and conditioned slope test ({corpus_label}, width_bins={width})...",
                      file=sys.stderr)
                census = rank1_and_temporal_profile_census(session_fn(root), corpus_label, None, width_bins=width)
                slope_test = sign_crossing_conditioned_slope_test(lag[lag_key], census["session_rows"], width)
                output["width_extension"][f"{corpus_label}_w{width}"] = {
                    "census": census, "sign_crossing_conditioned_slope_test": slope_test,
                }
                output_path.write_text(json.dumps(output, indent=2))
                print(f"  {corpus_label} width_bins={width}: n_tested={census['n_sessions_tested']}, "
                      f"verdict={slope_test['verdict']}, {census['wall_clock_s']:.1f}s", file=sys.stderr)

    output["wall_clock_s_this_invocation"] = time.time() - t0
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path} in {time.time() - t0:.1f}s total this invocation", file=sys.stderr)


if __name__ == "__main__":
    main()
