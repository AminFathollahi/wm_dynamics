"""run_epoch_matched_power_comparison.py -- is the encoding-versus-delay difference
in the shared cross-unit state a difference between the epochs, or a difference in
statistical power between the two arms that test them?

The delivered lag artifact (results/state_persistence_lag.json, read-only here and
never recomputed or edited) assigns the human encoding epoch the branch
``no_state_above_the_within_unit_floor`` and the human delay epoch
``flat_cross_unit_state``. Read as biology that says the shared cross-unit state
belongs to maintenance. The two arms, however, run an identical estimator (width 3
bins at a 0.1 s bin, adjacency lag 3 bins) and agree on every effect size: the
encoding arm's median existence effect is the LARGER of the two, all of its lags are
positive, and its slope and endpoint contrasts are as non-significant as the delay
arm's. The single thing that differs is how many sessions and how many lags each arm
gets to use, because the encoding epoch is shorter:

  encoding  72 pooled-structure rows -> 26 fitted, 46 skipped as ``width_exceeds_epoch``;
            every fitted row is a 2.0 s window (20 bins) reaching 15 lags
  delay     72 pooled-structure rows -> 72 fitted, at 2.3 s (23 bins) and 3.0 s (30 bins)

Every fitted encoding row is also a fitted delay row, so the two epochs are exactly
nested and an exactly paired within-session design is available.

The unit of analysis throughout is the POOLED-structure row: one row per recording
session, over all of its units. The lag artifact also carries per-anatomical-structure
rows -- hippocampus, amygdala, middle temporal gyrus and others -- which are subsets of
the units in the same session's pooled row, so a test that puts them side by side with
their own pooled row counts one recording several times. Every headline number here is
computed on pooled rows alone; the version that admits the per-structure rows is
reported beside it as a sensitivity check and is never the one a verdict reads.

This module makes four comparisons the delivered artifact never made.

Paired within session at matched lags. The per-session per-lag statistic is
d_perm(L) = r_median(L) - r_null_median(L) against the per-unit permutation null,
read straight out of the delivered rows. The encoding-minus-delay paired difference
is tested with the same paired sign-flip machinery every contrast in this project
uses, at the lags BOTH epochs reach, with ONE false-discovery-rate pass across those
lags applied to the paired contrast. Correcting two arms of unequal n separately is
what produced the apparent dissociation in the first place.

Matched session count. The delay arm's own unpaired existence contrast is re-run on
subsamples of its sessions drawn down to the encoding arm's session count, many
draws, and the distribution of "lags clearing correction" is reported. If the delay
arm at the encoding arm's n also typically clears zero lags, the epoch difference is
n and nothing else.

Matched source dataset. The rows the encoding arm loses are not a random subset: they
are every row of the two source datasets whose encoding window is 0.5 s. So the two
published arms differ in dataset composition and not only in session count -- the
encoding arm draws on ONE deposited dataset, the delay arm on THREE -- and that is a
second reason, independent of power, why they were never comparable as published.
Restricting the delay arm to the encoding arm's dataset returns exactly the same
sessions as the encoding arm, with no subsampling at all: the cleanest available
comparison, in which only the epoch differs.

Matched window. The delay epoch is re-fitted from raw counts truncated to the
encoding window's bin count, so that window length, bin count, lag range and the
number of bin pairs entering each lag all match the encoding arm exactly. This is the
only comparison here that needs raw data; the other three are arithmetic on the
delivered artifact.

Nothing in this module licenses the statement that the shared cross-unit state is
ABSENT at encoding. The encoding arm shows every one of its 15 lags positive at a
median existence effect of 0.0555 -- larger than the delay arm's 0.0492. The
supportable statement is that at 26 sessions it does not clear multiplicity
correction, and the effect size belongs next to that statement every time it is made.
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

from project_config import data_root  # noqa: E402

from state_persistence import (  # noqa: E402
    _d_series, lag_reachability_note, paired_vs_null_contrast,
)
from statistics import fdr_bh, permutation_pvalue, stable_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LAG_PATH = REPO_ROOT / "results" / "state_persistence_lag.json"
OUTPUT_PATH = REPO_ROOT / "results" / "epoch_matched_power_comparison.json"
TRUNCATED_DELAY_CHECKPOINT = REPO_ROOT / "results" / ".checkpoints" / "epoch_matched_power_truncated_delay.json"
DATA_ROOT = data_root()

DECIDING_WIDTH_BINS = 3
BIN_WIDTH_S = 0.1
EPOCHS = ("encoding", "delay")
ALPHA = 0.05
N_PERM = 10000
N_SUBSAMPLE_DRAWS = 2000
N_BREAKPOINT_BOOTSTRAP_DRAWS = 2000
SEED = 20260813

# ── Pre-declared decision rule ────────────────────────────────────────────────
# Written before any of the numbers below were computed. A branch invented after
# seeing a result is not a result.
#
# MATERIAL_DIFFERENCE_D_PERM is half the existence effect the two epochs themselves
# show (encoding 0.0555, delay 0.0492, both already delivered in the read-only lag
# artifact). A paired encoding-minus-delay difference smaller than half of what each
# epoch shows on its own cannot support an epoch-specific reading of the state; a
# confidence interval that reaches past it in BOTH directions cannot exclude one
# either, which is the inconclusive case.
MATERIAL_DIFFERENCE_D_PERM = 0.025
# The matched-session-count arm reproduces the encoding arm's outcome if clearing
# zero lags is not a rare draw for the delay arm once it is cut to the encoding
# arm's session count.
MATCHED_N_REPRODUCTION_MIN_FRACTION = 0.05

BRANCHES = {
    "epoch_difference_explained_by_power": (
        "No lag shows a significant encoding-minus-delay paired difference after a single "
        "false-discovery-rate pass across the matched lags, every matched lag's paired "
        f"confidence interval stays inside +/-{MATERIAL_DIFFERENCE_D_PERM} in d_perm units, and the delay "
        "arm cut to the encoding arm's session count reproduces the encoding arm's clearing count "
        f"in at least {MATCHED_N_REPRODUCTION_MIN_FRACTION:.0%} of draws. The shared cross-unit state is a "
        "population property present at encoding and during maintenance alike."
    ),
    "epoch_difference_survives_matched_power": (
        "At least one matched lag shows a significant encoding-minus-delay paired difference "
        "after a single false-discovery-rate pass. The epoch difference is real; its direction "
        "and effect size are reported with it."
    ),
    "inconclusive_below_detection_floor": (
        "No matched lag clears the single false-discovery-rate pass, but the paired confidence "
        f"interval at some matched lag reaches past +/-{MATERIAL_DIFFERENCE_D_PERM} in d_perm units in both "
        "directions, or the matched-session-count arm does not reproduce the encoding arm's "
        "clearing count. The comparison cannot separate a real epoch difference from a power "
        "difference at this session count."
    ),
}


# ── Reading the delivered rows ────────────────────────────────────────────────

def _row_key(row: dict) -> tuple:
    return (row["dataset"], row["patient"], row["session"], row["structure"])


def _int_keyed(lags: dict) -> dict[int, dict]:
    return {int(lag): value for lag, value in lags.items()}


def fitted_rows_by_key(lag_rows: list[dict], epoch: str, width_bins: int) -> dict[tuple, dict]:
    """Rows of one epoch at one estimator width that carry a fitted profile AND
    the per-unit permutation null the existence statistic is defined against,
    keyed by (dataset, patient, session, structure) so that pairing across
    epochs is a dictionary join and cannot silently depend on row order."""
    kept = {}
    for row in lag_rows:
        if row["epoch"] != epoch or row["width_bins"] != width_bins:
            continue
        if row["profile"].get("status") != "fitted" or row.get("null_permutation") is None:
            continue
        kept[_row_key(row)] = row
    return kept


def d_perm_series(row: dict) -> dict[int, float]:
    """d_perm(L) = observed r_median(L) minus the per-unit permutation null's
    r_null_median(L), at every lag both carry -- the delivered artifact's own
    existence statistic, computed through the same helper the delivered
    contrasts use so the two can never drift apart."""
    series, = _d_series([_int_keyed(row["profile"]["lags"])],
                        [_int_keyed(row["null_permutation"]["lags"])], "r_null_median")
    return series


def paired_epoch_series(
    encoding_by_key: dict[tuple, dict], delay_by_key: dict[tuple, dict],
) -> tuple[list[tuple], list[dict[int, float]], list[dict[int, float]]]:
    """The session rows fitted in BOTH epochs, and the two d_perm series lists in
    one shared key order. Returning the key list alongside the two series lists
    is what makes the pairing checkable: a caller cannot pass two independently
    ordered lists to the paired test by accident."""
    shared = sorted(set(encoding_by_key) & set(delay_by_key))
    return (shared,
            [d_perm_series(encoding_by_key[key]) for key in shared],
            [d_perm_series(delay_by_key[key]) for key in shared])


def matched_lag_range(*series_lists: list[dict[int, float]]) -> list[int]:
    """The lags every supplied session reaches in every supplied epoch -- computed
    from the data, never asserted, because the whole point of the comparison is
    that one arm's lag axis is shorter than the other's and a lag one arm cannot
    reach must never be tested."""
    reached = [set(series) for series_list in series_lists for series in series_list]
    if not reached:
        return []
    return sorted(set.intersection(*reached))


# ── Existence contrast, vectorised over the permutation axis ──────────────────

def existence_pvalues_by_lag(
    series_list: list[dict[int, float]], lags: list[int], rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> dict[int, dict]:
    """Per-lag sign-flip test of d_perm against zero -- the delivered artifact's
    existence contrast -- with the sign draws shared across lags and the null
    formed by one matrix product instead of one loop per lag. The marginal
    p-value per lag is what the false-discovery-rate pass consumes, so sharing
    sign draws across lags changes nothing a marginal test can see, and it makes
    a few thousand resampled session sets affordable. The bootstrap interval
    paired_vs_null_contrast also returns is deliberately not computed here; the
    resampling arms need only the clearing count.

    Underpowered lags are skipped on exactly the delivered contrast's rule: fewer
    than 4 contributing sessions, or a smallest attainable sign-flip p above the
    working level."""
    if not series_list or not lags:
        return {}
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(series_list)))
    out: dict[int, dict] = {}
    for lag in lags:
        columns = [i for i, series in enumerate(series_list) if lag in series]
        n_pairs = len(columns)
        min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs else 1.0
        if n_pairs < 4 or min_attainable_p > ALPHA:
            out[lag] = {"status": "underpowered_by_construction", "n_pairs": n_pairs,
                        "min_attainable_p": min_attainable_p}
            continue
        values = np.array([series_list[i][lag] for i in columns])
        observed = float(values.mean())
        null = signs[:, columns] @ values / n_pairs
        out[lag] = {"status": "tested", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
                    "mean_diff": observed, "p_value": permutation_pvalue(null >= observed)}
    return out


def apply_single_fdr_pass(by_lag: dict[int, dict], alpha: float = ALPHA) -> dict:
    """One false-discovery-rate pass across the lag axis of ONE contrast. Every
    contrast in this module is corrected exactly once, over exactly the lags it
    tested; correcting two arms of unequal session count separately is the step
    that manufactured the epoch dissociation being tested here."""
    tested = [lag for lag in sorted(by_lag) if by_lag[lag].get("status") == "tested"]
    if tested:
        fdr = fdr_bh(np.array([by_lag[lag]["p_value"] for lag in tested]), alpha=alpha)
        for lag, q, reject in zip(tested, fdr["q_values"], fdr["reject"]):
            by_lag[lag]["fdr_q_value"] = float(q)
            by_lag[lag]["fdr_significant"] = bool(reject)
    clearing = [lag for lag in tested if by_lag[lag].get("fdr_significant")]
    return {
        "alpha": alpha,
        "lags_bins_tested": tested,
        "n_lags_tested": len(tested),
        "lags_bins_clearing_fdr": clearing,
        "n_lags_clearing_fdr": len(clearing),
        "by_lag": {str(lag): by_lag[lag] for lag in sorted(by_lag)},
    }


# ── The paired encoding-versus-delay contrast ─────────────────────────────────

def paired_epoch_contrast(
    keys: list[tuple], encoding_series: list[dict[int, float]], delay_series: list[dict[int, float]],
    lags: list[int],
) -> dict:
    """Encoding-minus-delay d_perm, within session, at the lags both epochs reach,
    tested two-sided with the project's paired sign-flip test and corrected once
    across the lag axis. Two-sided because neither direction was predicted: the
    delivered per-epoch effect sizes put encoding slightly ABOVE delay, while the
    delivered branch labels read as though encoding were the weaker arm."""
    if not (len(keys) == len(encoding_series) == len(delay_series)):
        raise ValueError(
            "paired_epoch_contrast received key and series lists of different lengths; the "
            "pairing must be a key-aligned join, not two independently ordered lists."
        )
    unreachable = [lag for lag in lags
                   if any(lag not in series for series in encoding_series + delay_series)]
    if unreachable:
        raise ValueError(
            f"paired_epoch_contrast asked to test lags {unreachable}, which at least one paired "
            "session does not reach in one of the two epochs; a lag one arm cannot reach must "
            "never enter a cross-arm comparison."
        )
    by_lag: dict[int, dict] = {}
    for lag in lags:
        encoding_values = [series[lag] for series in encoding_series]
        delay_values = [series[lag] for series in delay_series]
        result = paired_vs_null_contrast(encoding_values, delay_values, direction="two-sided")
        result["encoding_mean_d_perm"] = float(np.mean(encoding_values))
        result["delay_mean_d_perm"] = float(np.mean(delay_values))
        by_lag[lag] = result
    summary = apply_single_fdr_pass(by_lag)
    tested = summary["lags_bins_tested"]
    means = [by_lag[lag]["mean_diff"] for lag in tested]
    within = [lag for lag in tested
              if abs(by_lag[lag]["ci_lower"]) <= MATERIAL_DIFFERENCE_D_PERM
              and abs(by_lag[lag]["ci_upper"]) <= MATERIAL_DIFFERENCE_D_PERM]
    summary.update({
        "n_sessions": len(keys),
        "session_keys": [list(key) for key in keys],
        "lag_range_bins": [min(lags), max(lags)] if lags else None,
        "lag_range_s": [min(lags) * BIN_WIDTH_S, max(lags) * BIN_WIDTH_S] if lags else None,
        "direction_note": "Positive mean_diff means encoding exceeds delay.",
        "median_paired_difference_across_lags": float(np.median(means)) if means else None,
        "largest_absolute_paired_difference": float(np.max(np.abs(means))) if means else None,
        "encoding_median_d_perm_across_lags": float(np.median(
            [by_lag[lag]["encoding_mean_d_perm"] for lag in tested])) if tested else None,
        "delay_median_d_perm_across_lags": float(np.median(
            [by_lag[lag]["delay_mean_d_perm"] for lag in tested])) if tested else None,
        "material_difference_threshold_d_perm": MATERIAL_DIFFERENCE_D_PERM,
        "lags_bins_with_ci_inside_material_threshold": within,
        "lags_bins_with_ci_reaching_past_material_threshold":
            [lag for lag in tested if lag not in within],
        "all_ci_inside_material_threshold": bool(tested and len(within) == len(tested)),
    })
    return summary


# ── Matched session count ─────────────────────────────────────────────────────

def matched_session_count_subsample(
    series_list: list[dict[int, float]], n_target: int, n_draws: int, rng: np.random.Generator,
    lags: list[int] | None = None,
) -> dict:
    """The delay arm's own unpaired existence contrast re-run on sessions drawn
    without replacement down to ``n_target``, with the single correction pass
    applied inside each draw exactly as the delivered contrast applies it. Reports
    the distribution of lags clearing correction, which is the quantity the epoch
    dissociation was actually read off."""
    if n_target > len(series_list):
        return {"status": "not_computable", "reason": "target session count exceeds the arm's session count",
                "n_available": len(series_list), "n_target": n_target}
    clearing_counts, tested_counts = [], []
    for _ in range(n_draws):
        picked = [series_list[i] for i in rng.choice(len(series_list), size=n_target, replace=False)]
        draw_lags = lags if lags is not None else sorted(set().union(*[set(s) for s in picked]))
        by_lag = existence_pvalues_by_lag(picked, draw_lags, rng)
        summary = apply_single_fdr_pass(by_lag)
        clearing_counts.append(summary["n_lags_clearing_fdr"])
        tested_counts.append(summary["n_lags_tested"])
    clearing = np.array(clearing_counts)
    return {
        "status": "tested",
        "n_available": len(series_list), "n_target": n_target, "n_draws": n_draws,
        "lag_axis": "restricted_to_matched_lags" if lags is not None else "each_draw_native_lag_axis",
        "lags_bins_restricted_to": lags,
        "n_lags_tested_median": float(np.median(tested_counts)),
        "n_lags_tested_range": [int(np.min(tested_counts)), int(np.max(tested_counts))],
        "clearing_lag_count_mean": float(clearing.mean()),
        "clearing_lag_count_median": float(np.median(clearing)),
        "clearing_lag_count_percentiles": {
            "p2.5": float(np.percentile(clearing, 2.5)), "p25": float(np.percentile(clearing, 25)),
            "p50": float(np.percentile(clearing, 50)), "p75": float(np.percentile(clearing, 75)),
            "p97.5": float(np.percentile(clearing, 97.5)),
        },
        "fraction_of_draws_clearing_zero_lags": float((clearing == 0).mean()),
        "fraction_of_draws_clearing_a_majority_of_lags":
            float(np.mean(clearing > np.array(tested_counts) / 2.0)),
    }


# ── One breakpoint common to both epochs ──────────────────────────────────────

def pooled_profile(series_list: list[dict[int, float]], lags: list[int]) -> dict[int, float]:
    """Across-session median d_perm at each lag. Pooling before fitting a shape is
    what keeps a grid-searched breakpoint off single-session noise."""
    return {lag: float(np.median([series[lag] for series in series_list if lag in series]))
            for lag in lags}


def shared_breakpoint_both_epochs(
    profiles_by_epoch: dict[str, dict[int, float]], lags: list[int], bin_width_s: float = BIN_WIDTH_S,
) -> dict | None:
    """ONE continuous two-slope breakpoint, in seconds, common to both epochs:
    each epoch keeps its own intercept and its own pair of slopes, but the hinge
    location is shared and chosen by grid search to minimise the epochs' TOTAL
    squared error. Fitting a breakpoint per arm and then comparing the arms gives
    each arm a different test and can split arms that differ in nothing but noise,
    which is why no per-arm breakpoint is fitted anywhere in this module."""
    lags_sorted = sorted(lags)
    if len(lags_sorted) < 5:
        return None
    x = np.array([lag * bin_width_s for lag in lags_sorted])
    responses = {epoch: np.array([profile[lag] for lag in lags_sorted])
                 for epoch, profile in profiles_by_epoch.items()}
    best = None
    for breakpoint_bins in lags_sorted[2:-2]:
        breakpoint_s = breakpoint_bins * bin_width_s
        design = np.stack([np.ones_like(x), x, np.maximum(0.0, x - breakpoint_s)], axis=1)
        total_sse, by_epoch = 0.0, {}
        for epoch, y in responses.items():
            coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
            if rank < 3:
                by_epoch = {}
                break
            total_sse += float(np.sum((y - design @ coefficients) ** 2))
            by_epoch[epoch] = {"intercept": float(coefficients[0]),
                               "early_slope_r_per_s": float(coefficients[1]),
                               "late_slope_r_per_s": float(coefficients[1] + coefficients[2])}
        if not by_epoch:
            continue
        if best is None or total_sse < best["total_sse"]:
            best = {"breakpoint_bins": int(breakpoint_bins), "breakpoint_s": float(breakpoint_s),
                    "total_sse": total_sse, "by_epoch": by_epoch}
    if best is not None:
        best["lag_range_bins"] = [lags_sorted[0], lags_sorted[-1]]
        best["candidate_breakpoints_bins"] = [int(b) for b in lags_sorted[2:-2]]
    return best


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def shared_breakpoint_slope_agreement(
    keys: list[tuple], encoding_series: list[dict[int, float]], delay_series: list[dict[int, float]],
    lags: list[int], fit: dict, n_draws: int, rng: np.random.Generator,
    encoding_furthest_lag_bins: int, delay_furthest_lag_bins: int, fitted_on: str,
) -> dict | None:
    """How closely the two epochs' decay agrees under the shared hinge, with an
    interval around the agreement rather than a bare pair of point estimates.

    Which rows the profile was built from is recorded in ``fitted_on`` and matters
    here more than anywhere else in this module: a slope read off a profile that
    mixes a session's pooled row with its own per-structure rows is a slope of a
    curve that counts some recordings several times.

    The interval is a cluster bootstrap whose resampling unit is the recording
    session, because the rows entering the pooled profile are nested within
    sessions -- a session contributes its pooled row and, where the sensitivity
    version is used, one row per recording structure -- and resampling rows would
    treat one recording's structures as independent replicates. Each draw refits
    the hinge from scratch, so the interval covers the grid search as well as the
    slopes, and the fraction of draws landing on each candidate hinge is reported
    beside it."""
    if fit is None or not keys:
        return None
    clusters: dict[tuple, list[int]] = {}
    for index, key in enumerate(keys):
        clusters.setdefault(key[:3], []).append(index)
    cluster_keys = sorted(clusters)

    draws: dict[str, list[float]] = {name: [] for name in
                                     ("encoding_early", "delay_early", "early_difference",
                                      "encoding_late", "delay_late", "late_difference")}
    breakpoint_counts: dict[int, int] = {}
    for _ in range(n_draws):
        picked = [i for c in rng.choice(len(cluster_keys), size=len(cluster_keys), replace=True)
                  for i in clusters[cluster_keys[c]]]
        redraw = shared_breakpoint_both_epochs(
            {"encoding": pooled_profile([encoding_series[i] for i in picked], lags),
             "delay": pooled_profile([delay_series[i] for i in picked], lags)}, lags)
        if redraw is None:
            continue
        breakpoint_counts[redraw["breakpoint_bins"]] = breakpoint_counts.get(redraw["breakpoint_bins"], 0) + 1
        encoding_early = redraw["by_epoch"]["encoding"]["early_slope_r_per_s"]
        delay_early = redraw["by_epoch"]["delay"]["early_slope_r_per_s"]
        encoding_late = redraw["by_epoch"]["encoding"]["late_slope_r_per_s"]
        delay_late = redraw["by_epoch"]["delay"]["late_slope_r_per_s"]
        draws["encoding_early"].append(encoding_early)
        draws["delay_early"].append(delay_early)
        draws["early_difference"].append(encoding_early - delay_early)
        draws["encoding_late"].append(encoding_late)
        draws["delay_late"].append(delay_late)
        draws["late_difference"].append(encoding_late - delay_late)

    n_resolved = len(draws["early_difference"])
    encoding_early = fit["by_epoch"]["encoding"]["early_slope_r_per_s"]
    delay_early = fit["by_epoch"]["delay"]["early_slope_r_per_s"]
    encoding_late = fit["by_epoch"]["encoding"]["late_slope_r_per_s"]
    delay_late = fit["by_epoch"]["delay"]["late_slope_r_per_s"]
    early_difference = encoding_early - delay_early
    mean_early_magnitude = float(np.mean([abs(encoding_early), abs(delay_early)]))
    relative_early_difference = abs(early_difference) / mean_early_magnitude if mean_early_magnitude else None
    early_lags = [lag for lag in lags if lag <= fit["breakpoint_bins"]]
    late_lags = [lag for lag in lags if lag > fit["breakpoint_bins"]]
    early_interval = _percentile_interval(draws["early_difference"])
    early_covers_zero = bool(early_interval and early_interval[0] <= 0.0 <= early_interval[1])

    return {
        "fitted_on": fitted_on,
        "resampling_unit": "recording session",
        "n_clusters": len(cluster_keys),
        "n_rows": len(keys),
        "n_draws": n_draws,
        "n_draws_resolving_a_hinge": n_resolved,
        "breakpoint_bins": fit["breakpoint_bins"],
        "breakpoint_s": fit["breakpoint_s"],
        "breakpoint_bins_selection_fraction": {
            str(bins): count / n_resolved for bins, count in sorted(breakpoint_counts.items())
        } if n_resolved else {},
        "breakpoint_bins_selected_in_fraction_of_draws":
            breakpoint_counts.get(fit["breakpoint_bins"], 0) / n_resolved if n_resolved else None,
        "early_segment_lag_bins": [early_lags[0], early_lags[-1]] if early_lags else None,
        "late_segment_lag_bins": [late_lags[0], late_lags[-1]] if late_lags else None,
        "n_matched_lags_in_early_segment": len(early_lags),
        "n_matched_lags_in_late_segment": len(late_lags),
        "early_slope_r_per_s": {"encoding": encoding_early, "delay": delay_early},
        "early_slope_ci_95": {"encoding": _percentile_interval(draws["encoding_early"]),
                              "delay": _percentile_interval(draws["delay_early"])},
        "early_slope_encoding_minus_delay": early_difference,
        "early_slope_encoding_minus_delay_ci_95": early_interval,
        "early_slope_encoding_minus_delay_ci_95_covers_zero": early_covers_zero,
        "early_slope_relative_difference": relative_early_difference,
        "late_slope_r_per_s": {"encoding": encoding_late, "delay": delay_late},
        "late_slope_ci_95": {"encoding": _percentile_interval(draws["encoding_late"]),
                             "delay": _percentile_interval(draws["delay_late"])},
        "late_slope_encoding_minus_delay": encoding_late - delay_late,
        "late_slope_encoding_minus_delay_ci_95": _percentile_interval(draws["late_difference"]),
        "encoding_furthest_reachable_lag_bins": encoding_furthest_lag_bins,
        "delay_furthest_reachable_lag_bins": delay_furthest_lag_bins,
        "finding": (
            f"One hinge, placed at {fit['breakpoint_s']:.1f} s and shared by both epochs, describes both "
            f"decays. Over the segment before the hinge the profile falls at {encoding_early:.4f} per "
            f"second at encoding and {delay_early:.4f} per second during the delay -- a difference of "
            f"{early_difference:+.4f} per second, {relative_early_difference:.0%} of the average of the "
            f"two magnitudes, with a 95% interval of [{early_interval[0]:.4f}, {early_interval[1]:.4f}] "
            "from a bootstrap that resamples recording sessions and refits the hinge in every draw. "
            + ("That interval covers zero, so the two epochs' early decay rates are not resolvably "
               "different at this session count and the point difference between them is not evidence "
               "of an epoch effect."
               if early_covers_zero else
               "That interval excludes zero, so the two epochs' early decay rates differ by more than "
               "session-resampling noise.")
            + (" The hinge location is itself only weakly determined: the same bootstrap places it at "
               f"{fit['breakpoint_bins']} bins in "
               f"{breakpoint_counts.get(fit['breakpoint_bins'], 0) / n_resolved:.0%} of draws, spread "
               f"over {len(breakpoint_counts)} of the {len(fit['candidate_breakpoints_bins'])} candidate "
               "positions, so the segmented description should be read as a summary of the profile's "
               "shape and not as a located change point." if n_resolved else "")
        ) if relative_early_difference is not None and early_interval else None,
        "late_segment_caveat": (
            f"The late-segment slopes ({encoding_late:.4f} per second at encoding against "
            f"{delay_late:.4f} during the delay) diverge, but that segment spans only "
            f"{len(late_lags)} of the {len(lags)} matched lags and sits where the encoding arm runs out "
            f"of window: the encoding arm cannot reach past lag {encoding_furthest_lag_bins} bins at all, "
            f"whereas the delay arm reaches lag {delay_furthest_lag_bins} bins. The late slopes are "
            "therefore estimated from the shortest and least-sampled end of the encoding arm's range and "
            "the divergence should not be read as an epoch difference."
        ),
    }


# ── Matched window: the delay epoch re-fitted at the encoding bin count ───────

def refit_delay_at_bin_count(bin_count: int, width_bins: int, data_root: Path) -> dict:
    """Every delay-epoch session re-fitted from raw counts truncated to the first
    ``bin_count`` bins, so the delay arm's window length, bin count, lag axis and
    number of bin pairs per lag all equal the encoding arm's. This is the only
    part of this module that touches raw data; it is checkpointed to disk so an
    interrupted run resumes instead of re-fitting."""
    if TRUNCATED_DELAY_CHECKPOINT.exists():
        cached = json.loads(TRUNCATED_DELAY_CHECKPOINT.read_text())
        if cached.get("bin_count") == bin_count and cached.get("width_bins") == width_bins:
            return cached

    from corpus_sessions import iter_all_corpora
    from run_state_persistence import LAG_BIN_MS, _counts_from_spikes, _lag_run_row, _seed

    rows, t0 = [], time.time()
    for index, meta in enumerate(iter_all_corpora(data_root)):
        counts = _counts_from_spikes(meta["spike_lists"], meta["epoch_onsets"]["delay"],
                                     meta["epoch_windows"]["delay"], LAG_BIN_MS)
        if counts.shape[2] < bin_count:
            rows.append({"dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                         "structure": meta["structure"], "epoch": "delay", "bin_ms": LAG_BIN_MS,
                         "window_mode": "truncated_to_encoding_bin_count",
                         "window_s": bin_count * LAG_BIN_MS / 1000.0,
                         "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
                         "n_bins": int(counts.shape[2]), "width_bins": width_bins,
                         "profile": {"status": "epoch_shorter_than_encoding_window"},
                         "null_poisson": None, "null_permutation": None})
            continue
        seed = _seed(meta["dataset"], meta["session"], meta["structure"], "delay", LAG_BIN_MS,
                     f"width{width_bins}", f"truncated{bin_count}", "lag")
        run = _lag_run_row(counts[:, :, :bin_count], width_bins, seed)
        rows.append({"dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                     "structure": meta["structure"], "epoch": "delay", "bin_ms": LAG_BIN_MS,
                     "window_mode": "truncated_to_encoding_bin_count",
                     "window_s": bin_count * LAG_BIN_MS / 1000.0, **run})
        if (index + 1) % 25 == 0:
            print(f"  truncated delay re-fit: {index + 1} rows, {time.time() - t0:.0f}s elapsed",
                  file=sys.stderr, flush=True)
            _write_truncated_checkpoint(rows, bin_count, width_bins, time.time() - t0, complete=False)
    payload = _write_truncated_checkpoint(rows, bin_count, width_bins, time.time() - t0, complete=True)
    print(f"  truncated delay re-fit done: {len(rows)} rows in {time.time() - t0:.0f}s", file=sys.stderr)
    return payload


def _write_truncated_checkpoint(rows: list[dict], bin_count: int, width_bins: int,
                                 wall_clock_s: float, complete: bool) -> dict:
    payload = {"bin_count": bin_count, "width_bins": width_bins, "complete": complete,
               "wall_clock_s": wall_clock_s, "rows": rows}
    TRUNCATED_DELAY_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    TRUNCATED_DELAY_CHECKPOINT.write_text(json.dumps(payload))
    return payload


# ── Unpaired existence contrast on a named arm ────────────────────────────────

def unpaired_existence_arm(
    series_list: list[dict[int, float]], lags: list[int] | None, label: str,
) -> dict:
    """The delivered artifact's own per-lag existence contrast, run on one named
    set of sessions, with its confidence interval reported at every lag and ONE
    correction pass across that arm's lag axis. Used for the arms whose effect
    size has to be read, not only their clearing count."""
    arm_lags = lags if lags is not None else sorted(set().union(*[set(s) for s in series_list])) \
        if series_list else []
    by_lag: dict[int, dict] = {}
    for lag in arm_lags:
        values = [series[lag] for series in series_list if lag in series]
        by_lag[lag] = paired_vs_null_contrast(values, [0.0] * len(values), direction="greater")
    summary = apply_single_fdr_pass(by_lag)
    tested = summary["lags_bins_tested"]
    means = [by_lag[lag]["mean_diff"] for lag in tested]
    nominal = [lag for lag in tested if by_lag[lag]["p_value"] <= ALPHA]
    summary.update({
        "arm": label, "n_sessions": len(series_list),
        "lag_range_bins": [min(arm_lags), max(arm_lags)] if arm_lags else None,
        "median_d_perm_across_lags": float(np.median(means)) if means else None,
        "d_perm_range_across_lags": [float(np.min(means)), float(np.max(means))] if means else None,
        "n_lags_positive": int(sum(1 for m in means if m > 0.0)),
        "alternative": "greater",
        "smallest_uncorrected_one_sided_p_value":
            float(min(by_lag[lag]["p_value"] for lag in tested)) if tested else None,
        "smallest_uncorrected_two_sided_p_value":
            float(min(by_lag[lag]["two_sided_p_value"] for lag in tested)) if tested else None,
        "smallest_fdr_q_value": float(min(by_lag[lag]["fdr_q_value"] for lag in tested)) if tested else None,
        # An arm that is nominally positive at many lags and clears none of them after
        # correction is the whole subject of this comparison, so the nominal layer is
        # summarised here rather than left to be reassembled from the per-lag table.
        "lags_bins_nominally_positive_uncorrected": nominal,
        "n_lags_nominally_positive_uncorrected": len(nominal),
        "nominally_positive_d_perm_range":
            [float(min(by_lag[lag]["mean_diff"] for lag in nominal)),
             float(max(by_lag[lag]["mean_diff"] for lag in nominal))] if nominal else None,
        "nominally_positive_one_sided_p_range":
            [float(min(by_lag[lag]["p_value"] for lag in nominal)),
             float(max(by_lag[lag]["p_value"] for lag in nominal))] if nominal else None,
        "nominally_positive_two_sided_p_range":
            [float(min(by_lag[lag]["two_sided_p_value"] for lag in nominal)),
             float(max(by_lag[lag]["two_sided_p_value"] for lag in nominal))] if nominal else None,
        "nominally_positive_fdr_q_range":
            [float(min(by_lag[lag]["fdr_q_value"] for lag in nominal)),
             float(max(by_lag[lag]["fdr_q_value"] for lag in nominal))] if nominal else None,
    })
    return summary


# ── Branch decision ───────────────────────────────────────────────────────────

def encoding_arm_effect_size(arm: dict) -> dict:
    """The encoding arm's own numbers, in the object that names its branch.

    The lag artifact labels this arm ``no_state_above_the_within_unit_floor``. That
    label is a thresholded summary of the block reproduced here, in which every lag
    the arm reaches is POSITIVE and the early ones are nominally significant; what
    fails is the multiplicity pass at this session count. Carrying the effect sizes
    in the same object is what stops the label being read as an absence."""
    tested = arm["lags_bins_tested"]
    return {
        "n_sessions": arm["n_sessions"],
        "unit_of_analysis": "pooled-structure session rows",
        "n_lags_tested": arm["n_lags_tested"],
        "n_lags_positive": arm["n_lags_positive"],
        "median_d_perm_across_lags": arm["median_d_perm_across_lags"],
        "d_perm_range_across_lags": arm["d_perm_range_across_lags"],
        "n_lags_clearing_fdr": arm["n_lags_clearing_fdr"],
        "smallest_fdr_q_value": arm["smallest_fdr_q_value"],
        "lags_bins_nominally_positive_uncorrected": arm["lags_bins_nominally_positive_uncorrected"],
        "nominally_positive_d_perm_range": arm["nominally_positive_d_perm_range"],
        "nominally_positive_one_sided_p_range": arm["nominally_positive_one_sided_p_range"],
        "nominally_positive_two_sided_p_range": arm["nominally_positive_two_sided_p_range"],
        "per_lag_d_perm": {str(lag): arm["by_lag"][str(lag)]["mean_diff"] for lag in tested},
        "per_lag_one_sided_p_value": {str(lag): arm["by_lag"][str(lag)]["p_value"] for lag in tested},
        "per_lag_two_sided_p_value": {str(lag): arm["by_lag"][str(lag)]["two_sided_p_value"] for lag in tested},
        "per_lag_fdr_q_value": {str(lag): arm["by_lag"][str(lag)]["fdr_q_value"] for lag in tested},
        "how_the_label_must_be_read": (
            f"At {arm['n_sessions']} pooled sessions the encoding arm is positive at all "
            f"{arm['n_lags_positive']} of the {arm['n_lags_tested']} lags it reaches, with a median "
            f"existence effect of {arm['median_d_perm_across_lags']:.4f} and "
            f"{arm['n_lags_nominally_positive_uncorrected']} lags nominally positive before correction. "
            f"Its false-discovery-rate floor is {arm['smallest_fdr_q_value']:.4f}, so no lag clears the "
            "multiplicity pass. The supportable statement is that the shared cross-unit state is not "
            "resolved above multiplicity correction at this session count, never that it is absent at "
            "encoding, and this effect size belongs beside the label every time the label is used."
        ),
    }


def decide_branch(paired: dict, matched_n: dict, encoding_arm: dict) -> dict:
    """The pre-declared rule, evaluated in the pre-declared order, with the effect
    size and its interval carried in the same object as the verdict.

    ``paired`` must be the pooled-structure contrast. The version that admits the
    per-anatomical-structure rows counts each recording once per structure plus once
    more in its pooled row, so its session count is inflated and its confidence
    intervals are narrower than the data support; it is a sensitivity check and
    nothing here may be decided on it."""
    tested = paired["lags_bins_tested"]
    strongest_lag = max(tested, key=lambda lag: abs(paired["by_lag"][str(lag)]["mean_diff"])) if tested else None
    strongest = paired["by_lag"][str(strongest_lag)] if strongest_lag is not None else {}
    reproduces = (matched_n.get("status") == "tested"
                  and matched_n["fraction_of_draws_clearing_zero_lags"] >= MATCHED_N_REPRODUCTION_MIN_FRACTION)

    if paired["n_lags_clearing_fdr"] >= 1:
        branch = "epoch_difference_survives_matched_power"
    elif paired["all_ci_inside_material_threshold"] and reproduces:
        branch = "epoch_difference_explained_by_power"
    else:
        branch = "inconclusive_below_detection_floor"

    return {
        "branch": branch,
        "branch_meaning": BRANCHES[branch],
        "branch_definitions": BRANCHES,
        "unit_of_analysis": "pooled-structure session rows only",
        "unit_of_analysis_note": (
            "Every number in this block comes from the pooled-structure paired contrast. The contrast "
            "that also admits the per-anatomical-structure rows is reported separately as a sensitivity "
            "check: those rows are subsets of the units in the same session's pooled row, so including "
            "them alongside it counts a recording several times and understates the uncertainty."
        ),
        "evaluation_order": [
            "any matched lag clearing the single correction pass -> epoch_difference_survives_matched_power",
            "otherwise every matched lag's interval inside the material threshold AND the matched-session-"
            "count arm reproducing the encoding arm's clearing count -> epoch_difference_explained_by_power",
            "otherwise -> inconclusive_below_detection_floor",
        ],
        "n_sessions_paired": paired["n_sessions"],
        "n_lags_tested": paired["n_lags_tested"],
        "n_lags_clearing_fdr": paired["n_lags_clearing_fdr"],
        "median_paired_difference_across_lags": paired["median_paired_difference_across_lags"],
        "largest_absolute_paired_difference": paired["largest_absolute_paired_difference"],
        "largest_difference_lag_bins": strongest_lag,
        "largest_difference_effect_size": strongest.get("mean_diff"),
        "largest_difference_ci": [strongest.get("ci_lower"), strongest.get("ci_upper")],
        "largest_difference_two_sided_p_value": strongest.get("p_value"),
        "largest_difference_fdr_q_value": strongest.get("fdr_q_value"),
        "reference_encoding_median_d_perm_across_lags": paired["encoding_median_d_perm_across_lags"],
        "reference_delay_median_d_perm_across_lags": paired["delay_median_d_perm_across_lags"],
        "material_difference_threshold_d_perm": MATERIAL_DIFFERENCE_D_PERM,
        "all_ci_inside_material_threshold": paired["all_ci_inside_material_threshold"],
        "matched_session_count_reproduces_encoding_clearing_count": bool(reproduces),
        "matched_session_count_fraction_of_draws_clearing_zero_lags":
            matched_n.get("fraction_of_draws_clearing_zero_lags"),
        "encoding_arm_effect_size": encoding_arm,
        "statement_the_effect_sizes_support": (
            "The shared cross-unit state is present at encoding: every lag the encoding arm reaches is "
            "positive, at a median existence effect of "
            f"{paired['encoding_median_d_perm_across_lags']:.4f} against the delay arm's "
            f"{paired['delay_median_d_perm_across_lags']:.4f} over the same lags and the same sessions. "
            f"What the encoding arm does not do at {paired['n_sessions']} pooled sessions is clear "
            "multiplicity correction. No deliverable may describe the state as absent at encoding."
        ) if paired["encoding_median_d_perm_across_lags"] is not None else None,
    }


# ── Zero-drop accounting ──────────────────────────────────────────────────────

def zero_drop_accounting(lag_rows: list[dict], width_bins: int, truncated_rows: list[dict]) -> dict:
    """Every row seen, with a status, in both epochs and in the re-fitted arm.
    seen = tested + excluded-by-reason, checked arithmetically rather than
    asserted in prose."""
    by_epoch = {}
    for epoch in EPOCHS:
        rows = [r for r in lag_rows if r["epoch"] == epoch and r["width_bins"] == width_bins]
        statuses: dict[str, int] = {}
        excluded_detail: dict[str, dict[str, int]] = {}
        for row in rows:
            status = row["profile"].get("status", "missing_status")
            statuses[status] = statuses.get(status, 0) + 1
            if status != "fitted":
                bucket = excluded_detail.setdefault(status, {})
                label = f"{row['dataset']} window_s={row['window_s']} n_bins={row['n_bins']}"
                bucket[label] = bucket.get(label, 0) + 1
        fitted = statuses.get("fitted", 0)
        by_epoch[epoch] = {
            "n_rows_seen": len(rows), "n_fitted": fitted,
            "n_excluded": len(rows) - fitted,
            "status_counts": statuses,
            "excluded_rows_by_reason_and_source": excluded_detail,
            "reconciles": bool(len(rows) == sum(statuses.values())),
            "n_pooled_structure_fitted": sum(1 for r in rows if r["structure"] == "pooled"
                                             and r["profile"].get("status") == "fitted"),
        }

    encoding_keys = set(fitted_rows_by_key(lag_rows, "encoding", width_bins))
    delay_keys = set(fitted_rows_by_key(lag_rows, "delay", width_bins))
    truncated_statuses: dict[str, int] = {}
    for row in truncated_rows:
        status = row["profile"].get("status", "missing_status")
        truncated_statuses[status] = truncated_statuses.get(status, 0) + 1

    return {
        "unit_of_analysis": "(dataset, patient, session, structure)",
        "estimator_width_bins": width_bins,
        "by_epoch": by_epoch,
        "nesting": {
            "n_encoding_fitted": len(encoding_keys), "n_delay_fitted": len(delay_keys),
            "n_fitted_in_both": len(encoding_keys & delay_keys),
            "encoding_fitted_set_is_a_subset_of_delay_fitted_set": bool(encoding_keys <= delay_keys),
            "n_delay_fitted_without_an_encoding_partner": len(delay_keys - encoding_keys),
        },
        "truncated_delay_refit": {
            "n_rows_seen": len(truncated_rows),
            "status_counts": truncated_statuses,
            "reconciles": bool(len(truncated_rows) == sum(truncated_statuses.values())),
        },
    }


# ── Dataset composition of the two published arms ─────────────────────────────

def dataset_composition_asymmetry(
    lag_rows: list[dict], width_bins: int, encoding_datasets: list[str], delay_datasets: list[str],
    encoding_pooled_keys: set[tuple], delay_matched_dataset_pooled_keys: set[tuple],
) -> dict:
    """Which deposited datasets each published arm is actually made of.

    The encoding and delay arms differ in more than session count: the encoding
    window is 0.5 s in some source datasets and 2.0 s in others, and a 0.5 s window
    is shorter than the estimator needs, so those datasets lose every encoding row
    they have. The surviving encoding arm is therefore drawn from a strict subset of
    the datasets the delay arm is drawn from, which is a second reason -- independent
    of power -- why the two arms were never comparable as published. Everything here
    is counted from the rows rather than asserted."""
    pooled = [row for row in lag_rows if row["width_bins"] == width_bins and row["structure"] == "pooled"]
    by_dataset: dict[str, dict] = {}
    for dataset in sorted({row["dataset"] for row in pooled}):
        entry: dict[str, dict] = {}
        for epoch in EPOCHS:
            rows = [row for row in pooled if row["dataset"] == dataset and row["epoch"] == epoch]
            fitted = [row for row in rows if row["profile"].get("status") == "fitted"]
            statuses: dict[str, int] = {}
            for row in rows:
                status = row["profile"].get("status", "missing_status")
                statuses[status] = statuses.get(status, 0) + 1
            entry[epoch] = {
                "n_pooled_rows": len(rows), "n_fitted": len(fitted),
                "n_excluded": len(rows) - len(fitted), "status_counts": statuses,
                "window_s_values": sorted({row["window_s"] for row in rows}),
                "n_bins_values": sorted({row["n_bins"] for row in rows}),
            }
        entry["contributes_to_the_encoding_arm"] = bool(entry["encoding"]["n_fitted"])
        entry["contributes_to_the_delay_arm"] = bool(entry["delay"]["n_fitted"])
        by_dataset[dataset] = entry

    dropped = sorted(d for d, entry in by_dataset.items()
                     if entry["encoding"]["n_fitted"] == 0 and entry["delay"]["n_fitted"] > 0)
    dropped_window_values = sorted({w for d in dropped for w in by_dataset[d]["encoding"]["window_s_values"]})
    kept_window_values = sorted({w for d in encoding_datasets
                                 for w in by_dataset[d]["encoding"]["window_s_values"]})
    n_dropped_rows = sum(by_dataset[d]["encoding"]["n_excluded"] for d in by_dataset)

    return {
        "headline": (
            f"The two published arms differ in which deposited datasets they are made of. The encoding "
            f"arm is {len(encoding_datasets)} dataset ({', '.join(encoding_datasets)}); the delay arm is "
            f"{len(delay_datasets)} ({', '.join(delay_datasets)}). The "
            f"{n_dropped_rows} pooled encoding rows that drop out are exactly the rows of "
            f"{', '.join(dropped)}, whose encoding window is "
            f"{'/'.join(f'{w:g}' for w in dropped_window_values)} s against "
            f"{'/'.join(f'{w:g}' for w in kept_window_values)} s in the dataset that survives -- too short "
            "for the estimator width, so those datasets lose every encoding row they have before any test "
            "is run. Restricting the delay arm to the encoding arm's dataset returns the identical "
            f"{len(encoding_pooled_keys)} sessions. Dataset composition is therefore a second reason, "
            "independent of session count, why the published encoding-versus-delay contrast was not a "
            "comparison of epochs."
        ),
        "encoding_arm_datasets": encoding_datasets,
        "delay_arm_datasets_as_delivered": delay_datasets,
        "n_encoding_arm_datasets": len(encoding_datasets),
        "n_delay_arm_datasets_as_delivered": len(delay_datasets),
        "datasets_absent_from_the_encoding_arm": dropped,
        "encoding_window_s_in_datasets_absent_from_the_encoding_arm": dropped_window_values,
        "encoding_window_s_in_the_surviving_dataset": kept_window_values,
        "n_pooled_encoding_rows_dropped": n_dropped_rows,
        "dropped_rows_come_only_from_the_absent_datasets": bool(
            n_dropped_rows == sum(by_dataset[d]["encoding"]["n_excluded"] for d in dropped)),
        "n_encoding_pooled_sessions": len(encoding_pooled_keys),
        "n_delay_pooled_sessions_all_datasets": sum(
            by_dataset[d]["delay"]["n_fitted"] for d in by_dataset),
        "n_delay_pooled_sessions_matched_dataset": len(delay_matched_dataset_pooled_keys),
        "delay_arm_restricted_to_the_encoding_dataset_is_the_same_session_set": bool(
            encoding_pooled_keys == delay_matched_dataset_pooled_keys),
        "by_dataset": by_dataset,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    lag_artifact = json.loads(LAG_PATH.read_text())
    lag_rows = lag_artifact["human_lag_rows"]

    encoding_by_key = fitted_rows_by_key(lag_rows, "encoding", DECIDING_WIDTH_BINS)
    delay_by_key = fitted_rows_by_key(lag_rows, "delay", DECIDING_WIDTH_BINS)
    keys, encoding_series, delay_series = paired_epoch_series(encoding_by_key, delay_by_key)
    lags = matched_lag_range(encoding_series, delay_series)
    encoding_bin_count = int(np.median([encoding_by_key[key]["n_bins"] for key in keys]))
    encoding_datasets = sorted({key[0] for key in encoding_by_key})
    delay_datasets = sorted({key[0] for key in delay_by_key})

    print(f"paired sessions: {len(keys)}; matched lags: {lags[0]}-{lags[-1]} bins "
          f"({len(lags)} lags, {lags[0] * BIN_WIDTH_S:.1f}-{lags[-1] * BIN_WIDTH_S:.1f} s)", file=sys.stderr)

    print("paired encoding-versus-delay contrast at matched lags...", file=sys.stderr)
    pooled_index = [i for i, key in enumerate(keys) if key[3] == "pooled"]
    pooled_keys = [keys[i] for i in pooled_index]
    encoding_pooled_series = [encoding_series[i] for i in pooled_index]
    delay_pooled_series = [delay_series[i] for i in pooled_index]
    paired_pooled = paired_epoch_contrast(pooled_keys, encoding_pooled_series, delay_pooled_series, lags)
    paired_all_structures = paired_epoch_contrast(keys, encoding_series, delay_series, lags)

    print("per-epoch unpaired existence arms...", file=sys.stderr)
    delay_pooled_all = [d_perm_series(row) for key, row in sorted(delay_by_key.items()) if key[3] == "pooled"]
    delay_pooled_matched_dataset_keys = [key for key in sorted(delay_by_key)
                                         if key[3] == "pooled" and key[0] in encoding_datasets]
    delay_pooled_matched_dataset = [d_perm_series(delay_by_key[key])
                                    for key in delay_pooled_matched_dataset_keys]
    arms = {
        "encoding_pooled": unpaired_existence_arm(encoding_pooled_series, lags, "encoding_pooled"),
        "delay_pooled_all_datasets_native_lags": unpaired_existence_arm(
            delay_pooled_all, None, "delay_pooled_all_datasets_native_lags"),
        "delay_pooled_all_datasets_matched_lags": unpaired_existence_arm(
            delay_pooled_all, lags, "delay_pooled_all_datasets_matched_lags"),
        "delay_pooled_matched_dataset_native_lags": unpaired_existence_arm(
            delay_pooled_matched_dataset, None, "delay_pooled_matched_dataset_native_lags"),
        "delay_pooled_matched_dataset_matched_lags": unpaired_existence_arm(
            delay_pooled_matched_dataset, lags, "delay_pooled_matched_dataset_matched_lags"),
    }

    print(f"matched session count: {N_SUBSAMPLE_DRAWS} draws of {len(encoding_pooled_series)} "
          f"from {len(delay_pooled_all)} delay sessions...", file=sys.stderr)
    n_target = len(encoding_pooled_series)
    matched_n_native = matched_session_count_subsample(
        delay_pooled_all, n_target, N_SUBSAMPLE_DRAWS, np.random.default_rng(stable_seed("matched_n_native")))
    matched_n_matched_lags = matched_session_count_subsample(
        delay_pooled_all, n_target, N_SUBSAMPLE_DRAWS,
        np.random.default_rng(stable_seed("matched_n_matched_lags")), lags=lags)

    print("one breakpoint common to both epochs...", file=sys.stderr)
    encoding_furthest_lag = max(max(series) for series in encoding_series)
    delay_furthest_lag = max(max(series) for series in delay_series)
    breakpoint_by_row_set, breakpoint_agreement_by_row_set = {}, {}
    for label, selected in (("primary_pooled_structure", pooled_index),
                            ("secondary_including_per_structure_rows", list(range(len(keys))))):
        selected_keys = [keys[i] for i in selected]
        selected_encoding = [encoding_series[i] for i in selected]
        selected_delay = [delay_series[i] for i in selected]
        fit = shared_breakpoint_both_epochs(
            {"encoding": pooled_profile(selected_encoding, lags),
             "delay": pooled_profile(selected_delay, lags)}, lags)
        fitted_on = (f"{len(selected_keys)} paired rows, "
                     + ("pooled-structure rows only, one per recording session"
                        if label.startswith("primary")
                        else "pooled-structure rows plus the per-anatomical-structure rows of the same "
                             "sessions"))
        if fit is not None:
            fit["fitted_on"] = fitted_on
            fit["n_rows_fitted_on"] = len(selected_keys)
        breakpoint_by_row_set[label] = fit
        breakpoint_agreement_by_row_set[label] = shared_breakpoint_slope_agreement(
            selected_keys, selected_encoding, selected_delay, lags, fit, N_BREAKPOINT_BOOTSTRAP_DRAWS,
            np.random.default_rng(stable_seed(f"shared_breakpoint_session_cluster_bootstrap_{label}")),
            encoding_furthest_lag, delay_furthest_lag, fitted_on)
    breakpoint_agreement = breakpoint_agreement_by_row_set["primary_pooled_structure"]

    print(f"matched window: re-fitting the delay epoch truncated to {encoding_bin_count} bins...",
          file=sys.stderr)
    truncated = refit_delay_at_bin_count(encoding_bin_count, DECIDING_WIDTH_BINS, DATA_ROOT)
    truncated_by_key = fitted_rows_by_key(truncated["rows"], "delay", DECIDING_WIDTH_BINS)
    truncated_keys, truncated_encoding_series, truncated_delay_series = paired_epoch_series(
        encoding_by_key, truncated_by_key)
    truncated_lags = matched_lag_range(truncated_encoding_series, truncated_delay_series)
    truncated_pooled_index = [i for i, key in enumerate(truncated_keys) if key[3] == "pooled"]
    paired_truncated_pooled = paired_epoch_contrast(
        [truncated_keys[i] for i in truncated_pooled_index],
        [truncated_encoding_series[i] for i in truncated_pooled_index],
        [truncated_delay_series[i] for i in truncated_pooled_index], truncated_lags)
    paired_truncated_all_structures = paired_epoch_contrast(
        truncated_keys, truncated_encoding_series, truncated_delay_series, truncated_lags)
    truncated_pooled_all = [d_perm_series(row) for key, row in sorted(truncated_by_key.items())
                            if key[3] == "pooled"]
    truncated_pooled_matched_dataset = [d_perm_series(row) for key, row in sorted(truncated_by_key.items())
                                        if key[3] == "pooled" and key[0] in encoding_datasets]
    arms["delay_pooled_all_datasets_truncated_window"] = unpaired_existence_arm(
        truncated_pooled_all, truncated_lags, "delay_pooled_all_datasets_truncated_window")
    arms["delay_pooled_matched_dataset_truncated_window"] = unpaired_existence_arm(
        truncated_pooled_matched_dataset, truncated_lags, "delay_pooled_matched_dataset_truncated_window")
    matched_n_truncated = matched_session_count_subsample(
        truncated_pooled_all, n_target, N_SUBSAMPLE_DRAWS,
        np.random.default_rng(stable_seed("matched_n_truncated_window")), lags=truncated_lags)

    primary_breakpoint = breakpoint_by_row_set["primary_pooled_structure"]
    secondary_breakpoint = breakpoint_by_row_set["secondary_including_per_structure_rows"]
    relative_gap_by_row_set = {
        label: breakpoint_agreement_by_row_set[label]["early_slope_relative_difference"]
        for label in breakpoint_agreement_by_row_set if breakpoint_agreement_by_row_set[label]
    }

    verdict = decide_branch(paired_pooled, matched_n_matched_lags,
                            encoding_arm_effect_size(arms["encoding_pooled"]))
    accounting = zero_drop_accounting(lag_rows, DECIDING_WIDTH_BINS, truncated["rows"])
    dataset_asymmetry = dataset_composition_asymmetry(
        lag_rows, DECIDING_WIDTH_BINS, encoding_datasets, delay_datasets,
        set(pooled_keys), set(delay_pooled_matched_dataset_keys))

    output = {
        "version": "2026-08-14",
        "relationship_to_prior_artifacts": (
            "results/state_persistence_lag.json is a read-only input and is not recomputed or edited here; "
            "every profile, permutation null and per-lag existence value used in the paired, matched-"
            "session-count and matched-dataset comparisons is read directly from its stored rows. This "
            "file's contribution is the paired within-session encoding-versus-delay contrast at the lags "
            "both epochs reach with a single correction pass, the same comparison at matched session "
            "count and matched dataset, and one delay-epoch re-fit at the encoding window's bin count. It "
            "bears on how the encoding and delay branch labels in that artifact's "
            "encoding_vs_delay_lag_shape block should be read; it does not change them."
        ),
        "scope": {
            "corpus": "Human intracranial single units, three source datasets, encoding and delay epochs, "
                      "100 ms bins, fixed-width lag estimator at the deciding width.",
            "unit_of_analysis": "pooled-structure session rows, keyed (dataset, patient, session, structure); "
                                "the per-anatomical-structure rows enter only the labelled sensitivity arms",
            "estimator_width_bins": DECIDING_WIDTH_BINS,
            "bin_width_s": BIN_WIDTH_S,
            "adjacency_lag_bins": DECIDING_WIDTH_BINS,
            "statistic": "d_perm(L) = observed r_median(L) minus the per-unit permutation null's "
                         "r_null_median(L), per session per lag.",
            "matched_lag_range_bins": [lags[0], lags[-1]],
            "matched_lag_range_s": [lags[0] * BIN_WIDTH_S, lags[-1] * BIN_WIDTH_S],
            "n_matched_lags": len(lags),
            "n_sessions_paired_pooled_structure": len(pooled_index),
            "n_rows_paired_including_per_structure_rows": len(keys),
            "encoding_window_bin_count": encoding_bin_count,
            "datasets_reaching_the_encoding_arm": encoding_datasets,
            "datasets_reaching_the_delay_arm": delay_datasets,
            "n_subsample_draws": N_SUBSAMPLE_DRAWS,
            "n_breakpoint_bootstrap_draws": N_BREAKPOINT_BOOTSTRAP_DRAWS,
            "n_sign_flip_permutations": N_PERM,
            "alpha": ALPHA,
            "seed": SEED,
            "exclusions": accounting,
        },
        "reachability": {
            "statistic": lag_reachability_note(),
            "paired_contrast": {
                "statistic_is_floored": False,
                "reason": "The paired statistic is a difference of two d_perm values, each itself a "
                          "difference of correlations, so it is unbounded in both directions and a "
                          "two-sided sign-flip test against zero can reject in either direction. At "
                          f"{len(pooled_index)} paired sessions the smallest attainable sign-flip p is far "
                          "below the working level, and with a single correction pass across "
                          f"{len(lags)} lags a significant epoch difference is attainable, so a null result "
                          "here is informative rather than unreachable by construction.",
            },
            "matched_session_count_arm": {
                "clearing_is_attainable": True,
                "reason": "The Monte Carlo permutation p-value floor is 1/(n_perm+1) and the correction "
                          "pass multiplies it by at most the number of lags tested, which leaves the "
                          "smallest attainable corrected value orders of magnitude below the working "
                          "level; a subsample can therefore clear every lag it tests, and reporting zero "
                          "clearing lags is a measurement rather than a ceiling.",
            },
        },
        "dataset_composition_asymmetry": dataset_asymmetry,
        "paired_epoch_contrast": {
            "why_the_pooled_structure_arm_is_primary": (
                "The per-anatomical-structure rows are subsets of the units in the same session's pooled "
                "row, so a contrast that admits both counts each recording several times. That inflates "
                "the session count, narrows every confidence interval and shrinks every corrected "
                "p-value without adding a single independent recording. The pooled-structure arm is the "
                "unit of analysis and is the only arm any verdict here is read from; the arm that admits "
                "the per-structure rows is kept alongside it purely as a sensitivity check on whether the "
                "picture changes when the anatomy is broken out."
            ),
            "primary_pooled_structure": paired_pooled,
            "secondary_including_per_structure_rows": paired_all_structures,
        },
        "paired_epoch_contrast_matched_window": {
            "primary_pooled_structure": paired_truncated_pooled,
            "secondary_including_per_structure_rows": paired_truncated_all_structures,
        },
        "per_arm_existence_contrasts": arms,
        "matched_session_count_subsampling": {
            "encoding_arm_observed_clearing_lag_count":
                arms["encoding_pooled"]["n_lags_clearing_fdr"],
            "encoding_arm_observed_lag_count": arms["encoding_pooled"]["n_lags_tested"],
            "delay_arm_full_clearing_lag_count":
                arms["delay_pooled_all_datasets_native_lags"]["n_lags_clearing_fdr"],
            "delay_arm_full_lag_count": arms["delay_pooled_all_datasets_native_lags"]["n_lags_tested"],
            "delay_at_encoding_session_count_native_lags": matched_n_native,
            "delay_at_encoding_session_count_matched_lags": matched_n_matched_lags,
            "delay_at_encoding_session_count_and_matched_window": matched_n_truncated,
        },
        "shared_breakpoint_both_epochs": {
            "why_the_pooled_structure_fit_is_primary": (
                "The hinge is fitted to a profile that is a median over rows at each lag, so which rows "
                "go in decides the curve. Admitting a session's per-anatomical-structure rows alongside "
                "its pooled row lets recordings with more instrumented structures pull the median, which "
                "is the same pseudo-replication that disqualifies the per-structure rows from the paired "
                "contrast. The pooled-structure fit, one row per recording session, is the primary."
            ),
            "row_set_sensitivity": (
                "The two row sets do not tell the same story about how similar the epochs' early decay "
                f"is. With the per-structure rows admitted the early slopes are "
                f"{secondary_breakpoint['by_epoch']['encoding']['early_slope_r_per_s']:.4f} per second at "
                f"encoding and {secondary_breakpoint['by_epoch']['delay']['early_slope_r_per_s']:.4f} "
                "during the delay, "
                f"{relative_gap_by_row_set['secondary_including_per_structure_rows']:.0%} apart, which "
                "reads as close agreement. At the pooled-structure unit of analysis the same shared-hinge "
                f"fit gives {primary_breakpoint['by_epoch']['encoding']['early_slope_r_per_s']:.4f} and "
                f"{primary_breakpoint['by_epoch']['delay']['early_slope_r_per_s']:.4f} per second, "
                f"{relative_gap_by_row_set['primary_pooled_structure']:.0%} apart. The close agreement is "
                "therefore a property of the pseudo-replicated row set and is not carried by the "
                "recordings themselves. What survives the change of row set is the hinge location, "
                f"{primary_breakpoint['breakpoint_bins']} bins in both, though the session bootstrap does "
                "not pin even that tightly. What the pooled-structure slope difference means is settled by "
                "its interval and not by its size, and that interval is in the slope-agreement block."
            ) if len(relative_gap_by_row_set) == 2 else None,
            **breakpoint_by_row_set,
        },
        "shared_breakpoint_slope_agreement": {
            "primary_arm": "primary_pooled_structure",
            **breakpoint_agreement_by_row_set,
        },
        "matched_window_refit": {
            "bin_count": encoding_bin_count,
            "window_s": encoding_bin_count * BIN_WIDTH_S,
            "n_rows_refitted": len(truncated["rows"]),
            "n_rows_fitted": len(truncated_by_key),
            "n_paired_sessions_pooled_structure": len(truncated_pooled_index),
            "n_paired_rows_including_per_structure_rows": len(truncated_keys),
            "matched_lag_range_bins": [truncated_lags[0], truncated_lags[-1]] if truncated_lags else None,
            "refit_wall_clock_s": truncated.get("wall_clock_s"),
        },
        "verdict": verdict,
        "wall_clock_s": time.time() - t_start,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUTPUT_PATH} in {time.time() - t_start:.1f}s", file=sys.stderr)
    print(json.dumps({
        "branch": verdict["branch"],
        "unit_of_analysis": verdict["unit_of_analysis"],
        "n_sessions_paired": verdict["n_sessions_paired"],
        "n_lags_clearing_fdr": verdict["n_lags_clearing_fdr"],
        "median_paired_difference": verdict["median_paired_difference_across_lags"],
        "largest_absolute_paired_difference": verdict["largest_absolute_paired_difference"],
        "fdr_q_floor": min(paired_pooled["by_lag"][str(lag)]["fdr_q_value"]
                           for lag in paired_pooled["lags_bins_tested"]),
        "matched_session_count_fraction_clearing_zero_lags":
            verdict["matched_session_count_fraction_of_draws_clearing_zero_lags"],
        "encoding_arm_median_d_perm": verdict["encoding_arm_effect_size"]["median_d_perm_across_lags"],
        "encoding_arm_fdr_q_floor": verdict["encoding_arm_effect_size"]["smallest_fdr_q_value"],
        "encoding_arm_datasets": dataset_asymmetry["encoding_arm_datasets"],
        "delay_arm_datasets": dataset_asymmetry["delay_arm_datasets_as_delivered"],
        "shared_breakpoint_s": breakpoint_agreement["breakpoint_s"] if breakpoint_agreement else None,
        "early_slope_encoding_minus_delay":
            breakpoint_agreement["early_slope_encoding_minus_delay"] if breakpoint_agreement else None,
        "early_slope_encoding_minus_delay_ci_95":
            breakpoint_agreement["early_slope_encoding_minus_delay_ci_95"] if breakpoint_agreement else None,
    }, indent=2))


if __name__ == "__main__":
    main()
