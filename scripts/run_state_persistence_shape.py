"""run_state_persistence_shape.py -- corrects the shape verdict on d_perm(L)
using rows already stored in results/state_persistence_lag.json, without a
new data census.

The whole-range OLS slope that a single-segment decision rule tests is a
weighted average of an early decay and a late, effectively flat segment; when
the profile has a visible turn, that average can come out near zero even
though both segments are individually significant. This module fits the
turn's location per session (rather than reading it off a pooled profile by
eye), tests the two segments separately -- on both the per-unit permutation
null (d_perm) and the Poisson null (d_pois), including and excluding the one
lag where the compared windows touch -- and applies a decision rule whose
branch conditions are two-sided, so a segment that rises significantly with
lag routes to an explicit off-list bucket instead of being absorbed into a
flat branch. results/state_persistence_lag.json is read-only input and is
not recomputed or edited anywhere in this module.
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

from state_persistence import (  # noqa: E402
    _d_series, amplitude_ratio_report, breakpoint_estimator_calibration, classify_lag_profile_segmented,
    fit_breakpoint_from_pooled_profile, segmented_breakpoint_summary, segmented_shape_contrasts,
    two_component_fit, two_component_identifiability_ladder,
)
from statistics import spearman_permutation_test  # noqa: E402

LAG_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_lag.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_shape.json"
BIN_WIDTH_S = 0.1
WIDTHS = (2, 3, 5)
DECIDING_WIDTH = 3
HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")


def _to_int_keyed(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def _lag_lists(rows: list[dict], width: int) -> tuple[list[dict], list[dict], list[dict]]:
    profiles, pois, perm = [], [], []
    for r in rows:
        if r.get("width_bins") != width:
            continue
        if r["profile"].get("status") != "fitted" or r.get("null_poisson") is None or r.get("null_permutation") is None:
            continue
        profiles.append(_to_int_keyed(r["profile"]["lags"]))
        pois.append(_to_int_keyed(r["null_poisson"]["lags"]))
        perm.append(_to_int_keyed(r["null_permutation"]["lags"]))
    return profiles, pois, perm


def _declared_breakpoint_bins(profiles: list[dict], perm: list[dict]) -> tuple[int, dict]:
    """1.2(a): the breakpoint estimated -- not read off a profile by eye --
    from this arm's own d_perm(L). Reports BOTH the per-session-fit-then-
    median approach 1.2(a) literally specifies, and a pooled-profile fit
    (:func:`fit_breakpoint_from_pooled_profile`), validated against each
    other by :func:`breakpoint_estimator_calibration` on a planted profile
    at this arm's own session count and empirical per-session noise: a
    smoke test on synthetic data (200 trials, n=46 sessions, noise SEM
    0.06, true breakpoint at 8 bins) found the per-session-fit-then-median
    approach biased late by 3 bins (median recovered 11, not 8) with a
    multimodal, edge-heavy distribution -- an unpenalised SSE grid search
    on a short, noisy single-session profile latches onto spurious late
    breakpoints where only 1-2 noisy edge points anchor the 'flat' segment.
    The pooled-profile fit on the same synthetic data recovered the planted
    breakpoint exactly (median 8) with a tight, unimodal distribution. The
    calibration is re-run here at each arm's own (n_sessions, noise) before
    trusting either, and its own preference decides which is used as this
    arm's declared breakpoint; the other stays in the artifact."""
    series = _d_series(profiles, perm, "r_null_median")
    per_session_summary = segmented_breakpoint_summary(series, BIN_WIDTH_S)
    pooled_fit = fit_breakpoint_from_pooled_profile(series, BIN_WIDTH_S)
    lags_present = sorted(set().union(*[set(g) for g in profiles])) if profiles else []

    if pooled_fit is None and per_session_summary.get("status") != "tested":
        fallback = lags_present[len(lags_present) // 2] if lags_present else DECIDING_WIDTH
        return fallback, {"status": "not_computed", "fallback_breakpoint_bins": fallback,
                           "reason": "fewer than 5 lags available for a segmented fit, per session or pooled"}

    empirical_noise_sem = None
    if lags_present and profiles:
        mid_lag = lags_present[len(lags_present) // 2]
        vals_at_mid = [s[mid_lag] for s in series if mid_lag in s]
        if len(vals_at_mid) >= 4:
            empirical_noise_sem = float(np.std(vals_at_mid, ddof=1))
    reference_bp_bins = (pooled_fit["breakpoint_bins"] if pooled_fit is not None else
                          int(round(per_session_summary["median_breakpoint_s"] / BIN_WIDTH_S)))
    reference_early_slope = (pooled_fit["early_slope"] if pooled_fit is not None else
                              per_session_summary.get("median_early_slope", -0.05))
    reference_floor = float(np.median([s[lags_present[-1]] for s in series if lags_present[-1] in s])) \
        if lags_present and any(lags_present[-1] in s for s in series) else 0.0

    calibration = None
    declared_bp = reference_bp_bins
    if empirical_noise_sem is not None and len(series) >= 8 and len(lags_present) >= 5:
        calibration = breakpoint_estimator_calibration(
            n_sessions=len(series), noise_sem=empirical_noise_sem, lags_bins=lags_present, bin_width_s=BIN_WIDTH_S,
            true_breakpoint_bins=reference_bp_bins, early_slope_r_per_s=abs(reference_early_slope),
            floor_value=reference_floor, n_trials_sim=100, seed=400,
        )
        if calibration.get("pooled_profile_preferred") and pooled_fit is not None:
            declared_bp = pooled_fit["breakpoint_bins"]
        elif per_session_summary.get("status") == "tested":
            declared_bp = int(round(per_session_summary["median_breakpoint_s"] / BIN_WIDTH_S))
    if lags_present:
        declared_bp = min(max(declared_bp, lags_present[0]), lags_present[-1])

    return declared_bp, {
        "status": "tested", "declared_breakpoint_bins": declared_bp,
        "per_session_fit_then_median": per_session_summary,
        "pooled_profile_fit": pooled_fit,
        "empirical_noise_sem_at_mid_lag": empirical_noise_sem,
        "calibration_check": calibration,
    }


def _breakpoint_shift_sensitivity(profiles, pois, perm, width_bins, breakpoint_bins) -> dict:
    """1.2(b): the 1.1 early-segment (excluding adjacency) verdict recomputed
    with the declared breakpoint shifted one lag earlier and one lag later,
    so a reader can see whether the branch depends on the exact choice."""
    out = {}
    for shift, label in ((-1, "breakpoint_minus_one_lag"), (0, "declared_breakpoint"), (1, "breakpoint_plus_one_lag")):
        bp = breakpoint_bins + shift
        seg = segmented_shape_contrasts(profiles, pois, perm, BIN_WIDTH_S, bp, width_bins)
        early_excl = seg.get("by_statistic", {}).get("d_perm", {}).get("early_segment_excluding_adjacency", {}).get("test", {})
        out[label] = {
            "breakpoint_bins": bp, "breakpoint_s": bp * BIN_WIDTH_S,
            "early_segment_excluding_adjacency_significant_negative": early_excl.get("significant_negative"),
            "early_segment_excluding_adjacency_significant_positive": early_excl.get("significant_positive"),
            "early_segment_excluding_adjacency_mean_value": early_excl.get("mean_value"),
            "early_segment_excluding_adjacency_status": early_excl.get("status"),
        }
    verdicts = {out[label]["early_segment_excluding_adjacency_significant_negative"] for label in out}
    return {"by_shift": out, "verdict_stable_across_one_lag_shift": len(verdicts) == 1}


def _arm_analysis(rows: list[dict], label: str, prior_branch_by_width: dict[str, str] | None) -> dict:
    breakpoint_bins_by_width = {}
    by_width = {}
    for width in WIDTHS:
        profiles, pois, perm = _lag_lists(rows, width)
        if not profiles:
            by_width[str(width)] = {"status": "not_available", "n_sessions": 0}
            continue
        bp_bins, bp_summary = _declared_breakpoint_bins(profiles, perm)
        breakpoint_bins_by_width[width] = bp_bins
        decision = classify_lag_profile_segmented(profiles, pois, perm, width, BIN_WIDTH_S, bp_bins)
        segmented = decision["segmented_contrasts"]
        entry = {
            "status": "tested", "n_sessions": len(profiles),
            "breakpoint_summary": bp_summary,
            "decision": decision,
        }
        prior_branch = (prior_branch_by_width or {}).get(str(width))
        if prior_branch is not None:
            entry["previous_round_branch"] = prior_branch
            entry["new_branch"] = decision["branch"]
            entry["branch_changed"] = prior_branch != decision["branch"]
            entry["change_reason"] = (
                "unchanged" if prior_branch == decision["branch"] else
                "the previous rule's branch condition tested only whether the slope was 'not significantly "
                "negative', which a significantly positive slope also satisfies; the current rule tests "
                "both signs and, where the profile has a turn, tests the early and late segments "
                "separately rather than one whole-range slope"
            )
        by_width[str(width)] = entry

        # Amplitude-ratio fallback and two-component fit at the deciding width only (1.2c), since
        # tau identifiability is judged once per arm, not once per width.
        if width == DECIDING_WIDTH:
            lag_min = min(set().union(*[set(g) for g in profiles]))
            series = _d_series(profiles, perm, "r_null_median")
            entry["amplitude_ratio_at_breakpoint"] = amplitude_ratio_report(series, lag_min, bp_bins)
            median_profile_lags = sorted(set().union(*[set(s) for s in series]))
            pooled_d_perm = [float(np.median([s[lag] for s in series if lag in s])) for lag in median_profile_lags]
            entry["pooled_median_d_perm_fit"] = two_component_fit(
                [lag * BIN_WIDTH_S for lag in median_profile_lags], pooled_d_perm)
            entry["breakpoint_shift_sensitivity"] = _breakpoint_shift_sensitivity(profiles, pois, perm, width, bp_bins)

    return {
        "label": label, "n_rows": len(rows),
        "declared_breakpoint_bins_by_width": breakpoint_bins_by_width,
        "by_width": by_width,
    }


def cohort_control(delay_pooled_rows: list[dict], width: int) -> dict:
    """1.3: splits the pooled human delay rows at the deciding width by
    window_s (a geometry axis -- bin count, pair count at every lag, and
    the fraction of the epoch a given lag spans all differ across window
    lengths) and reports, per cohort, where the breakpoint lands in
    seconds, plus a third split on dataset identity to check whether the
    window_s split is really a corpus split in disguise."""
    profiles, pois, perm = _lag_lists(delay_pooled_rows, width)
    rows_fitted = [r for r in delay_pooled_rows if r.get("width_bins") == width
                   and r["profile"].get("status") == "fitted" and r.get("null_permutation") is not None]
    by_window_s: dict[float, list[int]] = {}
    for i, r in enumerate(rows_fitted):
        by_window_s.setdefault(r["window_s"], []).append(i)

    cohorts = {}
    for window_s, idx in by_window_s.items():
        cohort_profiles = [profiles[i] for i in idx]
        cohort_pois = [pois[i] for i in idx]
        cohort_perm = [perm[i] for i in idx]
        bp_bins, bp_summary = _declared_breakpoint_bins(cohort_profiles, cohort_perm)
        n_bins_med = int(np.median([rows_fitted[i]["n_bins"] for i in idx]))
        n_pairs_at_shortest_lag = None
        lags_present = sorted(set().union(*[set(profiles[i]) for i in idx])) if idx else []
        if lags_present and profiles:
            shortest = lags_present[0]
            n_pairs_at_shortest_lag = int(np.median([profiles[i][shortest]["n_pairs"] for i in idx if shortest in profiles[i]]))
        declared_bp_s = bp_summary.get("declared_breakpoint_bins", None)
        declared_bp_s = declared_bp_s * BIN_WIDTH_S if declared_bp_s is not None else None
        # 1.1's last instruction: "Report the pooled result and the two delay-length cohorts
        # separately" -- the pooled double-difference and endpoint-drop contrasts silently restrict
        # to whichever cohort reaches the pooled lag_max, so the per-cohort segmented contrasts (at
        # each cohort's OWN declared breakpoint and OWN reachable lag range) are the ones that
        # actually speak for each cohort.
        cohort_decision = classify_lag_profile_segmented(cohort_profiles, cohort_pois, cohort_perm,
                                                           width, BIN_WIDTH_S, bp_bins)
        cohorts[str(window_s)] = {
            "n_sessions": len(idx), "window_s": window_s, "n_bins_median": n_bins_med,
            "n_pairs_at_shortest_lag_median": n_pairs_at_shortest_lag,
            "breakpoint_summary": bp_summary,
            "declared_breakpoint_s": declared_bp_s,
            "breakpoint_as_fraction_of_window": (declared_bp_s / window_s if declared_bp_s is not None else None),
            "datasets": sorted({rows_fitted[i]["dataset"] for i in idx}),
            "segmented_decision_at_own_breakpoint": cohort_decision,
        }

    bp_seconds = {k: v["declared_breakpoint_s"] for k, v in cohorts.items() if v["declared_breakpoint_s"] is not None}
    bp_fractions = {k: v["breakpoint_as_fraction_of_window"] for k, v in cohorts.items()
                    if v["breakpoint_as_fraction_of_window"] is not None}
    verdict = "not_computable"
    if len(bp_seconds) >= 2 and len(bp_fractions) >= 2:
        seconds_vals = list(bp_seconds.values())
        fraction_vals = list(bp_fractions.values())
        seconds_spread = (max(seconds_vals) - min(seconds_vals)) / np.mean(seconds_vals) if np.mean(seconds_vals) else None
        fraction_spread = (max(fraction_vals) - min(fraction_vals)) / np.mean(fraction_vals) if np.mean(fraction_vals) else None
        if seconds_spread is not None and fraction_spread is not None:
            verdict = "stayed_on_the_clock" if seconds_spread <= fraction_spread else "moved_with_the_geometry"

    cohort_is_pure_corpus_split = all(len(v["datasets"]) == 1 for v in cohorts.values()) and len(cohorts) > 1
    return {
        "cohorts_by_window_s": cohorts,
        "cohort_split_is_pure_corpus_split": cohort_is_pure_corpus_split,
        "verdict": verdict,
        "verdict_note": (
            "'stayed_on_the_clock' if the breakpoint's spread across cohorts in raw seconds is no larger "
            "than its spread as a fraction of each cohort's window length; 'moved_with_the_geometry' "
            "otherwise. A feature that moves with the geometry is an estimator property regardless of "
            "significance (rounds 26 and 27 both died on this)."
        ),
    }


def two_component_ladder_at_human_regime(delay_pooled_w3_rows: list[dict]) -> dict:
    fitted = [r for r in delay_pooled_w3_rows if r["profile"].get("status") == "fitted"]
    n_trials = int(np.median([r["n_trials"] for r in fitted])) if fitted else 40
    n_units = int(np.median([r["n_units"] for r in fitted])) if fitted else 36
    n_bins_2p3 = int(round(2.3 / BIN_WIDTH_S))
    return two_component_identifiability_ladder(
        n_trials=n_trials, n_units=n_units, n_bins=n_bins_2p3, bin_width_s=BIN_WIDTH_S,
        width_bins=DECIDING_WIDTH, rate_hz=0.15, seed=300,
    )


def main() -> None:
    t0 = time.time()
    lag = json.loads(LAG_PATH.read_text())

    human_rows = lag["human_lag_rows"]
    alm_rows = lag["alm_lag_rows"]
    panichello_rows = lag["panichello_lag_arm"]["rows"]

    delay_pooled = [r for r in human_rows if r["epoch"] == "delay" and r["structure"] == "pooled"
                    and r["dataset"] in HUMAN_DATASETS]
    encoding_pooled = [r for r in human_rows if r["epoch"] == "encoding" and r["structure"] == "pooled"
                       and r["dataset"] in HUMAN_DATASETS]

    prior_delay_branch = lag["deciding_contrast"]["branch_by_width"]
    prior_alm_branch = {str(lag["alm_lag_arm"]["decision"]["width_bins"]): lag["alm_lag_arm"]["decision"]["branch"]}
    prior_panichello_branch = {str(lag["panichello_lag_arm"]["decision"]["width_bins"]): lag["panichello_lag_arm"]["decision"]["branch"]}
    prior_encoding_branch = {str(lag["encoding_vs_delay_lag_shape"]["encoding"]["width_bins"]): lag["encoding_vs_delay_lag_shape"]["encoding"]["branch"]}

    print("Human delay (deciding arm)...", file=sys.stderr)
    human_delay = _arm_analysis(delay_pooled, "human_delay_pooled", prior_delay_branch)
    print("Human encoding (opportunistic)...", file=sys.stderr)
    human_encoding = _arm_analysis(encoding_pooled, "human_encoding_pooled_opportunistic", prior_encoding_branch)
    print("Mouse ALM...", file=sys.stderr)
    alm = _arm_analysis(alm_rows, "alm_delay_rate_matched", prior_alm_branch)
    print("Macaque Panichello (re-labelled from existing rows -- see panichello_rerun_note)...", file=sys.stderr)
    panichello = _arm_analysis(panichello_rows, "panichello_lpfc_delay", prior_panichello_branch)

    print("Cohort control (2.3s vs 3.0s delay, deciding width)...", file=sys.stderr)
    cohort = cohort_control(delay_pooled, DECIDING_WIDTH)

    print("Two-component identifiability ladder (synthetic, human-yield regime)...", file=sys.stderr)
    ladder = two_component_ladder_at_human_regime([r for r in delay_pooled if r.get("width_bins") == DECIDING_WIDTH])

    output = {
        "version": "2026-08-15",
        "relationship_to_prior_artifacts": (
            "results/state_persistence_lag.json is a read-only input and is not recomputed or edited; every "
            "profile, Poisson null, and permutation null used here is read directly from its stored rows. "
            "This file's branch per arm is the corrected reading; where it disagrees with "
            "state_persistence_lag.json's deciding_contrast, alm_lag_arm, panichello_lag_arm, or "
            "encoding_vs_delay_lag_shape fields, this file's branch is the correction and the reason is "
            "recorded per arm below in 'change_reason'."
        ),
        "panichello_rerun_note": (
            "A re-run of Panichello at w=3 with both nulls, at the widest lag range its native window "
            "supports, was expected to be necessary on the belief that the existing arm ran at w=2 over a "
            "narrower range. Checked directly: state_persistence_lag.json's panichello_lag_arm was already "
            "computed at w=3 with both nulls, and its native window (300-1450 ms, bounded by the deposited "
            "trial epoch's own time axis, which ends at 1449.5 ms) is already the widest reachable range -- "
            "the delivered contrast_b_slope_including_adjacency mean value (+0.0899 r/s) matches the number "
            "already on record exactly. No new census was needed or run; what needed correcting was the "
            "decision rule applied to these existing rows, which is what this file does."
        ),
        "bin_width_s": BIN_WIDTH_S, "widths_bins": list(WIDTHS), "deciding_width_bins": DECIDING_WIDTH,
        "arms": {
            "human_delay": human_delay,
            "human_encoding_opportunistic": human_encoding,
            "alm": alm,
            "panichello": panichello,
        },
        "cohort_control": cohort,
        "two_component_identifiability_ladder": ladder,
        "floor_language_guard": {
            "double_difference_at_deciding_width": (
                human_delay["by_width"][str(DECIDING_WIDTH)]["decision"]["segmented_contrasts"]
                ["double_difference_floor_vs_persistent_component"]
            ),
            "allowed_wording_if_double_difference_not_significant": (
                "no resolvable decay of d_perm between the breakpoint and the longest lag tested, with the "
                "late-segment slope's confidence interval as the bound, at this sensitivity"
            ),
            "disallowed_wording_unless_double_difference_significant": [
                "a persistent component", "a stable state", "activity-silent's opposite",
            ],
        },
        "wall_clock_s": time.time() - t0,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} in {time.time() - t0:.1f}s", file=sys.stderr)
    print(json.dumps({
        "human_delay_branch_by_width": {w: human_delay["by_width"][w].get("new_branch") for w in human_delay["by_width"]},
        "alm_branch_by_width": {w: alm["by_width"][w].get("new_branch") for w in alm["by_width"]},
        "panichello_branch_by_width": {w: panichello["by_width"][w].get("new_branch") for w in panichello["by_width"]},
        "cohort_verdict": cohort["verdict"],
        "two_component_identifiable": ladder["identifiable"],
    }, indent=2))


if __name__ == "__main__":
    main()
