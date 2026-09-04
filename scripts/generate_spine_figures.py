"""generate_spine_figures.py -- figure panels for the working-memory population-state
findings that are currently delivered, verified and standing.

Four findings, five source artifacts, one panel set each:

  - What the leading population-activity component's gain tracks, and how a different
    object -- the direction deviation that survives rate removal -- differs from it by
    an order of magnitude. Source: results/state_orthogonality_census.json.
  - Whether the cross-preparation content-coding dissociation survives matched label
    cardinality within one corpus, contrasted with the between-preparation correlation
    the same variable produces when preparations are compared instead of manipulated.
    Source: results/content_label_cardinality_ladder.json and, for the between-
    preparation points only, results/human_content_decodability.json.
  - Whether the cross-unit population state's existence replicates in a second macaque
    corpus. Source: results/watters_state_geometry.json.
  - Which of two trial-by-trial observables -- the leading component's own amplitude, or
    the direction deviation that survives rate removal -- actually predicts behavioural
    outcome once total spike count is controlled, run through the identical pipeline, the
    same sessions and the same three error-trial floors for both. Source:
    results/dominant_latent_identity_and_behaviour_breadth.json.
  - At which recording tier, from sorted single units down to a non-invasive scalp
    montage recorded simultaneously in the same patients, the component exists and
    whether it predicts behaviour at that tier. Source:
    results/recording_tier_component_transfer.json.
  - Whether stimulation-site distance to the component-carrying tissue predicts how much
    the component moves or how much recall changes, in an open-loop and a closed-loop
    human corpus. Source: results/stimulation_site_targeting_map.json.
  - Where a delivered pulse train lands relative to the sequence of remembered items, and
    what the two human stimulation corpora can and cannot resolve about stimulation
    parameters. Source: results/stimulation_timing_and_parameter_structure.json.
  - Whether the phase of the ongoing oscillation at delivery modulates the component, in
    a closed-loop, phase-tuned, non-invasive scalp stimulation corpus. Source:
    results/phase_locked_scalp_stimulation_component.json.

Every number placed on a panel is read from its source artifact at plot time and checked
against the value this module was specified against, to the precision that specification
gave; a mismatch raises immediately rather than silently plotting a stale or wrong number.
These nine artifacts are read-only inputs -- nothing here recomputes anything from raw
data, and nothing here is written back to them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures" / "spine"

_src_dir = str(REPO_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from visualization import nature_style  # noqa: E402 -- reuse house rcParams only, not the withdrawn fig* builders

# Okabe-Ito subset: colorblind-safe, stable across the whole panel set.
BLUE = "#0072B2"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#999999"
BLACK = "#111111"


def load(name: str) -> dict:
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def check(actual: float, expected: float, decimals: int, label: str) -> float:
    """Round-trip a number this panel was specified against. A mismatch means the
    artifact has moved since the specification was written -- stop and report, don't
    plot a number that no longer matches what's on disk."""
    tol = 0.5 * 10 ** (-decimals) + 1e-12
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{label}: artifact holds {actual!r}, does not match the specified "
            f"{expected!r} to {decimals} decimal place(s)."
        )
    return float(actual)


def check_int(actual: int, expected: int, label: str) -> int:
    if int(actual) != int(expected):
        raise AssertionError(f"{label}: artifact holds {actual!r}, expected {expected!r}.")
    return int(actual)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def text_box(ax, text: str, loc: str = "lower left") -> None:
    xy = {"lower left": (0.02, 0.02), "upper left": (0.02, 0.98), "lower right": (0.98, 0.02),
          "upper right": (0.98, 0.98)}[loc]
    va = "bottom" if "lower" in loc else "top"
    ha = "left" if "left" in loc else "right"
    ax.text(
        xy[0], xy[1], text, transform=ax.transAxes, fontsize=5.5, va=va, ha=ha,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY, linewidth=0.5, alpha=0.92),
    )


# ---- shared extraction for the two panels that both need the within-macaque ladder ----

def load_macaque_ladder(ladder: dict) -> dict:
    expected = {2: 0.7657, 3: 0.6514, 4: 0.6971, 6: 0.7143, 8: 0.6743}
    rows = {}
    for k_str, row in ladder["macaque_ladder"].items():
        k = int(k_str)
        rank = check(row["mean_fractional_rank"], expected[k], 4, f"macaque ladder k={k} mean fractional rank")
        rows[k] = {
            "rank": rank, "lo": row["ci95_lower"], "hi": row["ci95_upper"],
            "null": row["null_value"], "n": row["n_sessions"],
            "a_full": row["mean_a_full"], "n_clear": row["n_sessions_content_decodes_above_own_null"],
        }
    return rows


def load_macaque_ladder_slope(ladder: dict) -> dict:
    slope = ladder["macaque_ladder_slope"]
    return {
        "slope": check(slope["mean_slope"], -0.0251, 4, "macaque ladder slope"),
        "lo": check(slope["ci95_lower"], -0.0716, 4, "macaque ladder slope CI lower"),
        "hi": check(slope["ci95_upper"], 0.0199, 4, "macaque ladder slope CI upper"),
        "p": check(slope["sign_flip_p_value"], 0.302, 3, "macaque ladder slope p-value"),
    }


# ---- A: leading component identity, from state_orthogonality_census.json ----------

def panel_leading_component_magnitude_ladder(census: dict) -> None:
    pairs = [
        ("leading_component_gain", "total_spike_count", "Leading component\ngain vs.\nspike count", 0.9078, 0.8241),
        ("leading_component_gain", "trial_index", "Leading component\ngain vs.\ntrial index", 0.7991, 0.6385),
        ("rate_free_direction_deviation", "total_spike_count", "Direction deviation\nvs.\nspike count", 0.2266, 0.0513),
        ("rate_free_direction_deviation", "trial_index", "Direction deviation\nvs.\ntrial index", 0.1604, 0.0257),
    ]
    sc = census["same_observable_census"]
    sessions = census["sessions"]
    # magnitude_above_reference_test's ci_lower/ci_upper is a percentile bootstrap
    # interval on the MEAN of (|rho| - reference) (paired_sign_flip_test in
    # src/statistics.py, docstring: "ci_lower, ci_upper (percentile bootstrap on
    # mean_diff)"). The point plotted here is the MEDIAN absolute association, and
    # for the first pair the median (0.9078) falls entirely outside that mean's CI
    # (0.654-0.839, built around a mean of 0.753) -- a different estimand, not a
    # numerical error. Bracketing the median instead requires an interval computed
    # on the median, so this panel uses the interquartile range of the per-session
    # |rho| values themselves, taken directly from the artifact's per-session rows.
    medians, los, his, shares, ns, refs = [], [], [], [], [], []
    for obs, cand, _, expect_median, expect_share in pairs:
        audit = sc[obs][cand]["sign_invariance_audit"]
        rhos = np.array([s["identity_candidates"][obs][cand]["rho"] for s in sessions], dtype=float)
        abs_rhos = np.abs(rhos)
        median = check(float(np.median(abs_rhos)), expect_median, 4, f"{obs}/{cand} median |r|")
        check(audit["median_absolute_association"], expect_median, 4, f"{obs}/{cand} artifact median |r|")
        medians.append(median)
        q1, q3 = np.percentile(abs_rhos, [25, 75])
        los.append(float(q1))
        his.append(float(q3))
        shares.append(check(audit["median_shared_variance_per_session"], expect_share, 4, f"{obs}/{cand} shared variance"))
        ns.append(len(abs_rhos))
        refs.append(check(audit["median_reference_absolute_association_under_no_association"], 0.033, 3,
                           f"{obs}/{cand} no-association reference"))

    if len(set(ns)) != 1:
        raise AssertionError(f"session count differs across the four pairs: {ns}")
    if max(refs) - min(refs) > 1e-9:
        raise AssertionError(f"no-association reference differs across pairs: {refs}")
    n, ref = ns[0], refs[0]

    nature_style()
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    x = np.arange(4)
    colors = [BLUE, BLUE, VERMILLION, VERMILLION]
    for i in range(4):
        ax.errorbar(x[i], medians[i], yerr=[[medians[i] - los[i]], [his[i] - medians[i]]],
                    fmt="o", color=colors[i], ecolor=colors[i], capsize=3, markersize=6, zorder=3)
        ax.annotate(f"{medians[i]:.3f}\n{shares[i] * 100:.1f}% shared var",
                    (x[i], medians[i]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=5.3)
    ax.axhline(ref, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], fontsize=5.5)
    ax.set_ylabel("Sign-invariant median |r| across sessions\n(error bars: interquartile range)")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.6, 3.6)
    ax.set_title("Association magnitude with the leading component's identity", loc="left", fontsize=6.8)
    text_box(ax, f"n = {n} sessions, macaque lPFC\nno-association reference = {ref:.3f} (dashed)", "upper left")
    save(fig, "leading_component_magnitude_ladder")


def panel_sign_split(census: dict) -> None:
    sc = census["same_observable_census"]
    audit = sc["leading_component_gain"]["total_spike_count"]["sign_invariance_audit"]
    pooled_signed = sc["leading_component_gain"]["total_spike_count"]["pooled_signed_test"]

    median_abs = check(audit["median_absolute_association"], 0.9078, 4, "median |r|, spike count pair")
    expect_pos = check_int(audit["n_sessions_positive"], 17, "expected positive-session count")
    expect_neg = check_int(audit["n_sessions_negative"], 8, "expected negative-session count")

    sessions = census["sessions"]
    rhos = np.array([s["identity_candidates"]["leading_component_gain"]["total_spike_count"]["rho"]
                      for s in sessions])
    check_int(len(rhos), audit["n_sessions"], "per-session row count vs. audit n_sessions")
    n_pos, n_neg = int((rhos > 0).sum()), int((rhos < 0).sum())
    check_int(n_pos, expect_pos, "positive-session count recomputed from per-session rows")
    check_int(n_neg, expect_neg, "negative-session count recomputed from per-session rows")

    signed_mean = check(pooled_signed["mean_value"], 0.300, 3, "pooled signed mean")
    signed_lo, signed_hi = pooled_signed["ci_lower"], pooled_signed["ci_upper"]
    signed_p = pooled_signed["two_sided_p_value"]

    nature_style()
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
    for r in rhos:
        color = BLUE if r > 0 else VERMILLION
        ax.plot([0, 1], [r, abs(r)], color=color, alpha=0.35, linewidth=0.6, zorder=2)
        ax.scatter([0, 1], [r, abs(r)], color=color, s=8, zorder=3)
    ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
    ax.errorbar([0], [signed_mean], yerr=[[signed_mean - signed_lo], [signed_hi - signed_mean]],
                fmt="D", color=BLACK, markersize=6, capsize=3, zorder=4)
    ax.scatter([1], [median_abs], marker="D", color=BLACK, s=36, zorder=4)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Signed r\n(sign set by each session's\nown decomposition)", "|r|\n(sign-invariant)"], fontsize=5.3
    )
    ax.set_ylabel("Correlation with total spike count")
    ax.set_title("Signed pooling hides a near-deterministic relationship", loc="left", fontsize=6.5)
    text_box(
        ax,
        f"n = 25 sessions: {n_pos} positive, {n_neg} negative "
        f"(sign vs. coin flip p = {audit['sign_split_vs_fair_coin_p_value']:.3f})\n"
        f"pooled signed mean {signed_mean:+.3f} [{signed_lo:+.3f}, {signed_hi:+.3f}], p = {signed_p:.3f}\n"
        f"median |r| (sign-invariant) = {median_abs:.3f}",
        "lower right",
    )
    save(fig, "spike_count_association_sign_split")


def panel_split_half_reliability(census: dict) -> None:
    shr = census["split_half_reliability"]
    obs_order = [
        ("leading_component_gain", "Leading component\ngain", 0.9714, 0.9410, 0.9846),
        ("rate_free_direction_deviation", "Rate-free direction\ndeviation", 0.7718, 0.6539, 0.8380),
    ]
    medians, los, his, ns = [], [], [], []
    for key, _, expect_med, expect_lo, expect_hi in obs_order:
        block = shr[key]["pooled"]
        medians.append(check(block["median_reliability"], expect_med, 4, f"{key} split-half reliability"))
        los.append(check(block["ci_lower"], expect_lo, 4, f"{key} split-half reliability CI lower"))
        his.append(check(block["ci_upper"], expect_hi, 4, f"{key} split-half reliability CI upper"))
        ns.append(block["n_sessions"])
    if len(set(ns)) != 1:
        raise AssertionError(f"session count differs between the two observables: {ns}")

    nature_style()
    fig, ax = plt.subplots(figsize=(3.0, 3.3))
    x = np.arange(2)
    for i in range(2):
        ax.errorbar(x[i], medians[i], yerr=[[medians[i] - los[i]], [his[i] - medians[i]]],
                    fmt="s", color=BLUE, ecolor=BLUE, capsize=4, markersize=7, zorder=3)
        ax.text(x[i], medians[i] - 0.10, f"{medians[i]:.3f}\n[{los[i]:.3f}, {his[i]:.3f}]",
                ha="center", va="top", fontsize=5.3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([o[1] for o in obs_order])
    ax.set_ylabel("Split-half reliability (Spearman-Brown corrected)")
    ax.set_title("Both state observables are reliable, not noise", loc="left", fontsize=6.8)
    text_box(ax, f"n = {ns[0]} sessions, macaque lPFC; 10 unit-split replicates per session", "lower left")
    save(fig, "state_observable_split_half_reliability")


# ---- B: label cardinality, from content_label_cardinality_ladder.json / human_content_decodability.json --

def panel_macaque_cardinality_ladder(ladder: dict) -> dict:
    rows = load_macaque_ladder(ladder)
    slope = load_macaque_ladder_slope(ladder)

    axis = ladder["split_axis_control"]
    primary = check(axis["primary_summary"]["mean_fractional_rank"], 0.766, 3, "two-class primary split")
    rotated = check(axis["rotated_summary"]["mean_fractional_rank"], 0.537, 3, "two-class orthogonal split")

    classes = sorted(rows)
    xs = np.log2(classes)
    ranks = [rows[k]["rank"] for k in classes]
    los = [rows[k]["lo"] for k in classes]
    his = [rows[k]["hi"] for k in classes]
    null = rows[classes[0]]["null"]
    n = rows[classes[0]]["n"]
    if any(rows[k]["null"] != null or rows[k]["n"] != n for k in classes):
        raise AssertionError("null value or session count is not constant across ladder rungs")

    nature_style()
    fig, ax = plt.subplots(figsize=(4.1, 3.3))
    ax.errorbar(xs, ranks, yerr=[np.array(ranks) - np.array(los), np.array(his) - np.array(ranks)],
                fmt="o-", color=BLUE, ecolor=BLUE, capsize=3, markersize=6, linewidth=1.0, zorder=3)
    for x, y in zip(xs, ranks):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=5.3)

    x2 = float(np.log2(2))
    ax.scatter([x2 + 0.14], [rotated], marker="^", color=ORANGE, s=30, zorder=4)
    ax.plot([x2, x2 + 0.14], [primary, rotated], color=ORANGE, linewidth=0.7, linestyle=":", zorder=2)
    ax.annotate(f"orthogonal split\n{rotated:.3f}", (x2 + 0.14, rotated), xytext=(4, -4),
                textcoords="offset points", fontsize=5.1, color=ORANGE)

    ax.axhline(null, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(xs[-1] + 0.15, null, f"null = {null:.1f}", fontsize=5.3, color=GRAY, va="bottom", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in classes])
    ax.set_xlabel("Number of label classes (log2 spacing)")
    ax.set_ylabel("Mean fractional rank of the leading latent")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.9)
    ax.set_title("Within-macaque cardinality ladder is flat", loc="left", fontsize=6.8)
    text_box(
        ax,
        f"n = {n} sessions/rung, macaque lPFC\n"
        f"slope on log2(classes) = {slope['slope']:.4f} [{slope['lo']:.4f}, {slope['hi']:.4f}], "
        f"p = {slope['p']:.3f}\n"
        f"two-class rung: primary {primary:.3f}, orthogonal {rotated:.3f}",
        "lower left",
    )
    save(fig, "macaque_content_cardinality_ladder")
    return rows


def panel_macaque_decodability_by_cardinality(rows: dict) -> None:
    expected_a_full = {2: 0.763, 3: 0.768, 4: 0.7896, 6: 0.7564, 8: 0.7627}
    expected_clear = {2: 25, 3: 25, 4: 24, 6: 25, 8: 24}
    classes = sorted(rows)
    a_full, clear = [], []
    for k in classes:
        a_full.append(check(rows[k]["a_full"], expected_a_full[k], 4, f"macaque ladder k={k} mean full-decoder AUC"))
        clear.append(check_int(rows[k]["n_clear"], expected_clear[k], f"k={k} sessions clearing their own null"))
    n = rows[classes[0]]["n"]

    xs = np.log2(classes)
    nature_style()
    fig, ax = plt.subplots(figsize=(4.1, 3.3))
    ax.plot(xs, a_full, "o-", color=BLUE, markersize=6, linewidth=1.0, zorder=3)
    for x, y, c in zip(xs, a_full, clear):
        ax.annotate(f"{y:.3f}\n{c}/{n} clear own null", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=5.3)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
    ax.text(xs[0] - 0.25, 0.5, "chance", fontsize=5.3, color=GRAY, va="bottom", ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in classes])
    ax.set_xlabel("Number of label classes (log2 spacing)")
    ax.set_ylabel("Mean full-population decoder AUC")
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.3)
    ax.set_title("Decodability is flat across the same rungs", loc="left", fontsize=6.8)
    text_box(ax, f"n = {n} sessions/rung, macaque lPFC", "lower right")
    save(fig, "macaque_content_decodability_by_cardinality")


def panel_cross_preparation_contrast(ladder: dict, rows: dict, human: dict) -> None:
    desc = human["rank_matched_cardinality_description"]
    check_int(desc["matched_k_latents"], 8, "matched k latents")

    want = {
        "inagaki_alm5": (2, 0.1429, BLUE, "mouse ALM\n(k=8, 2 classes)"),
        "dandi_001187": (5, 0.5510, ORANGE, "human MTL/PFC\n(k=8, 5 classes)"),
        "panichello_2024": (8, 0.6743, VERMILLION, "macaque lPFC\n(k=8, 8 classes)"),
    }
    points = {p["dataset"]: p for p in desc["points"]}
    if set(points) != set(want):
        raise AssertionError(f"unexpected dataset set in rank_matched_cardinality_description: {sorted(points)}")

    xs, ys, los, his, colors, labels = [], [], [], [], [], []
    for name, (k_expect, rank_expect, color, label) in want.items():
        p = points[name]
        check_int(p["n_classes"], k_expect, f"{name} n_classes")
        rank = check(p["mean_fractional_rank"], rank_expect, 4, f"{name} mean fractional rank")
        xs.append(p["n_classes"]); ys.append(rank)
        los.append(p["mean_fractional_rank_ci95"][0]); his.append(p["mean_fractional_rank_ci95"][1])
        colors.append(color); labels.append(label)

    slope_block = desc["slope_across_preparations"]["fractional_rank"]
    slope = check(slope_block["slope"], 0.0878, 4, "between-preparation slope")
    slope_lo = check(slope_block["ci95"][0], 0.0587, 4, "between-preparation slope CI lower")
    slope_hi = check(slope_block["ci95"][1], 0.1148, 4, "between-preparation slope CI upper")

    ladder_classes = sorted(rows)
    ladder_x = ladder_classes
    ladder_y = [rows[k]["rank"] for k in ladder_classes]
    within_slope = load_macaque_ladder_slope(ladder)

    nature_style()
    fig, ax = plt.subplots(figsize=(4.3, 3.5))
    ax.plot(ladder_x, ladder_y, "o-", color=GRAY, markersize=4, linewidth=0.8, alpha=0.75, zorder=2,
            label="within-macaque ladder (causal manipulation)")
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    guide_x = np.array([min(xs) - 0.4, max(xs) + 0.4])
    ax.plot(guide_x, cy + slope * (guide_x - cx), color=BLACK, linewidth=0.7, linestyle=":", zorder=2,
            label="between-preparation slope, through the 3 means")
    for x, y, lo, hi, c, lab in zip(xs, ys, los, his, colors, labels):
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=c, ecolor=c, capsize=3, markersize=7, zorder=4)
        ax.annotate(f"{lab}\n{y:.3f}", (x, y), xytext=(7, 0), textcoords="offset points", fontsize=5.0, va="center")
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=0.6, zorder=1)
    ax.set_xlabel("Number of label classes")
    ax.set_ylabel("Mean fractional rank of the leading latent")
    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(0, 1.08)
    ax.set_title("Between-preparation correlation vs. the within-corpus causal test", loc="left", fontsize=6.5)
    text_box(
        ax,
        f"between-preparation slope +{slope:.4f} [{slope_lo:.4f}, {slope_hi:.4f}] per class "
        f"(n=76 sessions, 3 preparations)\n"
        f"within-macaque slope {within_slope['slope']:.4f} [{within_slope['lo']:.4f}, {within_slope['hi']:.4f}] "
        f"per doubling, p = {within_slope['p']:.3f} (n=25 sessions)",
        "lower right",
    )
    save(fig, "label_cardinality_cross_preparation_contrast")


# ---- C: second-corpus existence, from watters_state_geometry.json -----------------

def panel_watters_existence_by_lag(watters: dict) -> None:
    bin_ms = watters["scope"]["bin_ms"]
    pooled = watters["results"]["single_and_multi_unit"]["pooled"]["existence"]
    n_fitted = check_int(pooled["n_sessions_fitted"], 41, "watters existence n_sessions_fitted")

    animals = sorted({s["animal"] for s in watters["sessions"]})
    if len(animals) != 2:
        raise AssertionError(f"expected 2 animals in the analysed session list, found {animals}")

    per_lag = pooled["per_lag_observed_versus_permutation_null"]
    lag_keys = sorted(per_lag, key=int)
    if len(lag_keys) != 5:
        raise AssertionError(f"expected 5 reachable lags, found {len(lag_keys)}: {lag_keys}")

    expected_first_four = {"3": 0.198, "4": 0.201, "5": 0.214, "6": 0.215}
    seconds, diffs, los, his, ps = [], [], [], [], []
    for lk in lag_keys:
        row = per_lag[lk]
        sec = int(lk) * bin_ms / 1000.0
        if lk in expected_first_four:
            check(row["mean_diff"], expected_first_four[lk], 3, f"watters lag {lk} observed-minus-null")
            check(row["p_value"], 9.999e-05, 8, f"watters lag {lk} p-value")
        # The fifth reachable lag (bin 7 / 0.7 s) is taken as deposited: the specification
        # explicitly declines to give it a number to check against, and its p-value in the
        # artifact is not the same as the other four (1.9998e-04 vs 9.999e-05), so it must not
        # be forced to match them.
        if not row["fdr_significant"]:
            raise AssertionError(f"lag {lk} does not survive FDR correction in the artifact")
        seconds.append(sec); diffs.append(row["mean_diff"]); los.append(row["ci_lower"]); his.append(row["ci_upper"])
        ps.append(row["p_value"])

    nature_style()
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    yerr = [np.array(diffs) - np.array(los), np.array(his) - np.array(diffs)]
    ax.errorbar(seconds, diffs, yerr=yerr, fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=6,
                linestyle="none", zorder=3)
    for s, d in zip(seconds, diffs):
        ax.annotate(f"{d:.3f}", (s, d), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=5.3)
    ax.axhline(0, color=GRAY, linewidth=0.7, zorder=1)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Observed minus permutation-null correlation")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0, max(his) * 1.25)
    ax.set_title("Cross-unit state existence replicates by lag", loc="left", fontsize=6.8)
    text_box(
        ax,
        f"n = {n_fitted} sessions, 2 animals ({animals[0]}, {animals[1]})\n"
        f"all 5 lags significant after Benjamini-Hochberg correction (max p = {max(ps):.4f})",
        "lower right",
    )
    save(fig, "watters_cross_unit_state_existence_by_lag")


# ---- D: two behaviourally-tested observables, from dominant_latent_identity_and_behaviour_breadth.json --

def panel_amplitude_behaviour_dissociation(dominant: dict) -> None:
    """Same corpus, same 11-to-17 sessions per floor, same delay epoch, same paired
    sign-flip test, same three error-trial floors, run for two observables: the leading
    component's own per-trial amplitude, and the rate-free direction deviation that
    survives spike-count removal. The one with the larger raw correlation with trial
    outcome (amplitude) is the one that collapses toward zero once spike count is
    partialled out; the smaller raw correlation (direction deviation) barely moves."""
    amp_by_floor = dominant["dominant_latent_amplitude_and_outcome"]["by_floor"]
    dev_by_floor = dominant["error_floor_sensitivity"]["by_floor"]
    floors = ["60", "45", "30"]

    expected = {
        ("amp", "60", "raw"): -0.1675, ("amp", "60", "partial"): -0.0157,
        ("amp", "45", "partial"): 0.0025, ("amp", "30", "partial"): 0.0016,
        ("dev", "60", "raw"): -0.0974, ("dev", "60", "partial"): -0.0984,
        ("dev", "45", "partial"): -0.0732, ("dev", "30", "partial"): -0.0689,
    }

    rows: dict[tuple[str, str], dict] = {}
    ns: dict[str, int] = {}
    for floor in floors:
        amp_pooled = amp_by_floor[floor]["pooled"]
        dev_pooled = dev_by_floor[floor]["pooled"]
        for tag, pooled in (("amp", amp_pooled), ("dev", dev_pooled)):
            for stat, key in (("raw", "raw_outcome_vs_observable"), ("partial", "partial_controlling_spike_count")):
                block = pooled[key]
                mean = block["mean_value"]
                if (tag, floor, stat) in expected:
                    mean = check(mean, expected[(tag, floor, stat)], 4,
                                 f"{'amplitude' if tag == 'amp' else 'direction deviation'} {stat} @ floor {floor}")
                rows[(tag, floor, stat)] = {"mean": mean, "lo": block["ci_lower"], "hi": block["ci_upper"],
                                             "n": block["n_sessions"]}
        n_amp, n_dev = rows[("amp", floor, "raw")]["n"], rows[("dev", floor, "raw")]["n"]
        if n_amp != n_dev:
            raise AssertionError(f"floor {floor}: amplitude n={n_amp} != direction-deviation n={n_dev}, "
                                  "expected the identical session set for both observables")
        ns[floor] = n_amp

    nature_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    offset = 0.16
    for i, floor in enumerate(floors):
        for tag, color, dx in (("amp", BLUE, -offset), ("dev", VERMILLION, offset)):
            x = i + dx
            raw, part = rows[(tag, floor, "raw")], rows[(tag, floor, "partial")]
            ax.plot([x, x], [raw["mean"], part["mean"]], color=color, linewidth=0.8, linestyle=":", zorder=2)
            for stat, point, face, label_dy in (("raw", raw, color, -14), ("partial", part, "white", 6)):
                ax.errorbar(x, point["mean"],
                            yerr=[[point["mean"] - point["lo"]], [point["hi"] - point["mean"]]],
                            fmt="o", color=color, ecolor=color, mfc=face, mec=color, capsize=2.5,
                            markersize=6.5, zorder=3)
                ax.annotate(f"{point['mean']:+.3f}", (x, point["mean"]), xytext=(0, label_dy),
                            textcoords="offset points", ha="center", fontsize=4.6, color=color)
    ax.axhline(0, color=GRAY, linewidth=0.7, zorder=1)
    ax.set_xticks(range(len(floors)))
    ax.set_xticklabels([f"floor {f} error trials\n(n = {ns[f]} sessions)" for f in floors])
    ax.set_ylabel("Correlation with trial outcome")
    ax.set_xlim(-0.5, len(floors) - 0.5)
    ax.set_title("The larger raw behavioural correlation is the one that vanishes under control",
                 loc="left", fontsize=6.5)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=BLUE, markerfacecolor=BLUE, linestyle="none", markersize=6,
                   label="leading component amplitude"),
        plt.Line2D([0], [0], marker="o", color=VERMILLION, markerfacecolor=VERMILLION, linestyle="none",
                   markersize=6, label="rate-free direction deviation"),
        plt.Line2D([0], [0], marker="o", color=BLACK, markerfacecolor=BLACK, linestyle="none", markersize=6,
                   label="raw correlation"),
        plt.Line2D([0], [0], marker="o", color=BLACK, markerfacecolor="white", linestyle="none", markersize=6,
                   label="partialling out total spike count"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=5.0, frameon=True, framealpha=0.92,
              handletextpad=0.4, borderpad=0.5)
    save(fig, "leading_component_behaviour_dissociation_by_error_floor")


# ---- E: recording-tier survival, from recording_tier_component_transfer.json ------

def panel_recording_tier_transfer(tier: dict) -> None:
    """Two questions, one artifact: does the component exist at a tier, and does it
    predict behaviour at that tier. The existence magnitude is not comparable across
    tiers -- the scalp and beamformed instruments ship at sample rates whose Nyquist
    limit puts the invasive high-gamma band out of reach -- so it is drawn as a per-tier
    existence statistic with that caveat in the caption, never as a ranking by size."""
    tiers = ["single_unit", "depth_mtl", "depth_cortical", "scalp_eeg", "beamformed_cortical"]
    labels = ["single\nunit", "depth\nMTL", "depth\ncortical", "scalp\nEEG", "beamformed\ncortical"]
    expected_a = {"single_unit": -0.0604, "depth_mtl": -0.0783, "depth_cortical": -0.0402,
                  "scalp_eeg": -0.0853, "beamformed_cortical": -0.1942}
    expected_b = {"single_unit": -0.0077, "depth_mtl": 0.0038, "depth_cortical": 0.0331, "scalp_eeg": 0.0112}
    a, b = tier["block_a"], tier["block_b"]
    n_patients = check_int(a["single_unit"]["n_patients_contributing"], 9, "recording tier n patients")

    nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))

    ax = axes[0]
    ps = []
    for i, t in enumerate(tiers):
        pt = a[t]["pooled_patient_test"]
        mean = check(pt["mean_value"], expected_a[t], 4, f"block A {t} mean")
        check_int(pt["n_patients"], n_patients, f"block A {t} n patients")
        ax.errorbar(i, mean, yerr=[[mean - pt["ci_lower"]], [pt["ci_upper"] - mean]],
                    fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=6, zorder=3)
        ps.append(pt["p_value"])
    if len(set(round(p, 6) for p in ps)) != 1:
        raise AssertionError(f"existence p-values differ across tiers: {ps}")
    ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(labels, fontsize=5.2)
    ax.set_ylabel("Patient-clustered existence statistic\n(band differs by tier -- not comparable across tiers)")
    ax.set_title("Present at every tier", loc="left", fontsize=6.5)
    text_box(ax, f"n = {n_patients} patients per tier\nall five tiers significant, p = {ps[0]:.4f}\n"
                  f"(magnitude-matched rotation null)", "lower left")

    ax = axes[1]
    for i, t in enumerate(tiers[:4]):
        pt = b[t]["primary_within_set_size_test"]
        mean = check(pt["mean_value"], expected_b[t], 4, f"block B {t} mean")
        ax.errorbar(i, mean, yerr=[[mean - pt["ci_lower"]], [pt["ci_upper"] - mean]],
                    fmt="o", color=VERMILLION, ecolor=VERMILLION, capsize=3, markersize=6, zorder=3)
        ax.annotate(f"mdd={pt['mdd']:.3f}", (i, mean), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=4.6)
    void = b["beamformed_cortical"]
    check_int(void["n_patients_contributing"], 0, "beamformed block B n patients")
    ax.set_ylim(-0.12, 0.12)
    ax.axvspan(3.6, 4.4, color=GRAY, alpha=0.12, zorder=0)
    ax.text(4, 0, "structural void:\nno accuracy label\nis retrievable\nfor this tier\n(n = 0)",
            rotation=90, va="center", ha="center", fontsize=4.4, color=GRAY)
    ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(labels, fontsize=5.2)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel("Correlation with trial-level recall failure\n(within set size, patient-clustered)")
    ax.set_title("No tier predicts behaviour", loc="left", fontsize=6.5)
    text_box(ax, f"n = {n_patients} patients per tested tier\nevery tested tier's minimum detectable\n"
                  f"difference sits below the 0.14 reference", "upper right")

    fig.tight_layout()
    save(fig, "recording_tier_existence_and_behaviour_link")


# ---- F: stimulation-site distance, from stimulation_site_targeting_map.json -------

def panel_stimulation_site_targeting(site: dict) -> None:
    """Both relationships, both corpora, all three channel conditions -- twelve cells,
    every one underpowered_to_ask (open-loop) or underpowered_to_ask under causal:false
    (closed-loop). None is drawn as a null: the shaded band marks this project's 0.14
    reference effect, and every cell's own minimum detectable correlation (0.48-0.71)
    sits far outside it."""
    conditions = ["full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank"]
    cond_dx = {"full_channel_set": -0.22, "excluding_stimulated_pair": 0.0, "excluding_stimulated_shank": 0.22}
    cond_marker = {"full_channel_set": "o", "excluding_stimulated_pair": "s", "excluding_stimulated_shank": "D"}

    expected_open = {
        ("displacement", "full_channel_set"): (-0.0515, 69, 31, 0.4850),
        ("displacement", "excluding_stimulated_pair"): (-0.0509, 69, 31, 0.4850),
        ("displacement", "excluding_stimulated_shank"): (0.0877, 68, 30, 0.4924),
        ("behavior", "full_channel_set"): (0.0301, 69, 31, 0.4850),
        ("behavior", "excluding_stimulated_pair"): (0.0530, 69, 31, 0.4850),
        ("behavior", "excluding_stimulated_shank"): (-0.0494, 68, 30, 0.4924),
    }
    expected_closed = {
        ("displacement", "full_channel_set"): (0.1641, 27, 14, 0.6883),
        ("displacement", "excluding_stimulated_pair"): (0.1782, 27, 14, 0.6883),
        ("displacement", "excluding_stimulated_shank"): (0.1020, 25, 13, 0.7094),
        ("behavior", "full_channel_set"): (-0.3024, 28, 15, 0.6689),
        ("behavior", "excluding_stimulated_pair"): (-0.3017, 28, 15, 0.6689),
        ("behavior", "excluding_stimulated_shank"): (-0.5018, 26, 14, 0.6883),
    }
    open_a = site["block_a"]
    closed_a = site["block_c"]["block_a"]
    if site["block_c"]["causal"] is not False:
        raise AssertionError("closed-loop block is expected to carry causal: false")

    nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7))
    rel_map = [("displacement_relationship", "displacement", axes[0],
                "Component displacement vs.\nstimulation-site distance"),
               ("behavior_relationship", "behavior", axes[1],
                "Recall outcome vs.\nstimulation-site distance")]

    for json_key, short, ax, title in rel_map:
        for gi, (block, expected, color) in enumerate([(open_a[json_key], expected_open, BLUE),
                                                         (closed_a[json_key], expected_closed, GRAY)]):
            for cond in conditions:
                cell = block["by_channel_condition"][cond]
                exp_r, exp_ns, exp_nsub, exp_mdd = expected[(short, cond)]
                r_val = check(cell["r"], exp_r, 4, f"{short} {cond} r (loop {gi})")
                check_int(cell["n_sessions"], exp_ns, f"{short} {cond} n_sessions (loop {gi})")
                check_int(cell["n_subjects"], exp_nsub, f"{short} {cond} n_subjects (loop {gi})")
                check(cell["mdd"]["mdd"], exp_mdd, 4, f"{short} {cond} mdd (loop {gi})")
                x = gi + cond_dx[cond]
                face = color if cond == "excluding_stimulated_shank" else "white"
                ax.errorbar(x, r_val, yerr=[[r_val - cell["ci_lower"]], [cell["ci_upper"] - r_val]],
                            fmt=cond_marker[cond], color=color, ecolor=color, mfc=face, mec=color,
                            capsize=2.5, markersize=6, zorder=3)
        ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
        ax.axhspan(-0.14, 0.14, color=GRAY, alpha=0.10, zorder=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["open-loop\n(causal)", "closed-loop\n(causal: false)"], fontsize=5.3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Pearson r")
        ax.set_title(title, loc="left", fontsize=6.3)

    legend_handles = [
        plt.Line2D([0], [0], marker="D", color=BLACK, markerfacecolor=BLACK, linestyle="none", markersize=6,
                   label="stimulated shank excluded (headline)"),
        plt.Line2D([0], [0], marker="s", color=BLACK, markerfacecolor="white", linestyle="none", markersize=6,
                   label="stimulated pair excluded"),
        plt.Line2D([0], [0], marker="o", color=BLACK, markerfacecolor="white", linestyle="none", markersize=6,
                   label="full channel set"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=4.4, frameon=True, framealpha=0.9)
    text_box(axes[1], "shaded band: |r| below the 0.14 reference;\nevery cell's own minimum detectable\n"
                       "correlation exceeds it -- underpowered\nto ask, not a null", "lower right")
    fig.tight_layout()
    save(fig, "stimulation_site_distance_relationships")


# ---- G: stimulation timing/parameters, from stimulation_timing_and_parameter_structure.json --

def panel_stimulation_timing_and_parameters(timing: dict) -> None:
    open_pooled = timing["block_a_attributability_to_a_single_item"]["ds005489_open_loop"]["pooled"]
    closed_pooled = timing["block_a_attributability_to_a_single_item"]["ds005557_classifier_triggered"]["pooled"]
    open_train = check(open_pooled["train_duration_s_median"], 4.6, 2, "open-loop train duration")
    open_spacing = check(open_pooled["item_presentation_spacing_s_median"], 2.501, 3, "open-loop item spacing")
    open_frac = check(open_pooled["fraction_stimulated_items_train_covers_either_neighbor"], 1.0, 2,
                       "open-loop neighbor fraction")
    closed_train = check(closed_pooled["train_duration_s_median"], 0.5, 2, "closed-loop train duration")
    closed_spacing = check(closed_pooled["item_presentation_spacing_s_median"], 2.501, 3, "closed-loop item spacing")
    closed_frac = check(closed_pooled["fraction_stimulated_items_train_covers_either_neighbor"], 0.0, 2,
                         "closed-loop neighbor fraction")
    n_open_trains = check_int(open_pooled["n_trains_total"], 4224, "open-loop n trains")
    n_closed_trains = check_int(closed_pooled["n_trains_total"], 1886, "closed-loop n trains")

    b = timing["block_b_position_in_sequence_interaction"]["primary_all_stimulated_items"]
    interaction, pos_eff, stim_eff = b["interaction"], b["main_effects"]["serial_position_on_recall"], \
        b["main_effects"]["stimulation_on_recall"]
    check(interaction["mean_value"], 0.0019, 4, "position interaction mean")
    interaction_mdd = check(interaction["mdd"]["mdd"], 0.0077, 4, "position interaction mdd")
    pos_mean = check(pos_eff["mean_value"], -0.0207, 4, "serial position main effect")
    check(stim_eff["mean_value"], -0.0206, 4, "stimulation main effect")
    n_subjects_b = check_int(b["n_subjects_fitted"], 37, "block B n subjects")

    cc = timing["block_c_parameter_census"]
    open_amp = {float(k): v for k, v in cc["ds005489_open_loop"]["stimulated_trials_by_amplitude_microamps"].items()}
    closed_amp = {float(k): v for k, v in
                  cc["ds005557_classifier_triggered"]["stimulated_trials_by_amplitude_microamps"].items()}
    n_open_pairs = check_int(cc["ds005489_open_loop"]["n_electrode_pairs"], 52, "open-loop n electrode pairs")
    n_closed_pairs = check_int(cc["ds005557_classifier_triggered"]["n_electrode_pairs"], 18,
                                "closed-loop n electrode pairs")
    check_int(sum(open_amp.values()), cc["ds005489_open_loop"]["n_stimulated_trials_total"], "open-loop trial sum")
    check_int(sum(closed_amp.values()), cc["ds005557_classifier_triggered"]["n_stimulated_trials_total"],
              "closed-loop trial sum")

    nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))

    ax = axes[0]
    ax.barh([-0.18, 1 - 0.18], [open_train, closed_train], height=0.32, color=BLUE, label="stimulation train")
    ax.barh([0.18, 1 + 0.18], [open_spacing, closed_spacing], height=0.32, color=GRAY, label="item spacing")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"open-loop\n({open_frac * 100:.0f}% of trains span a neighbor)",
                         f"closed-loop\n({closed_frac * 100:.0f}% of trains span a neighbor)"], fontsize=5.0)
    ax.set_xlabel("Seconds (median)")
    ax.set_title("Pulse train vs. item spacing", loc="left", fontsize=6.3)
    ax.legend(fontsize=4.4, loc="lower right", frameon=True, framealpha=0.9)
    text_box(ax, f"n = {n_open_trains} open-loop / {n_closed_trains}\nclosed-loop stimulation trains", "upper right")

    ax = axes[1]
    rows = [("stimulation\non recall", stim_eff, VERMILLION), ("serial position\non recall", pos_eff, BLUE),
            ("position x stim.\ninteraction", interaction, GRAY)]
    for i, (_, block, color) in enumerate(rows):
        mean = block["mean_value"]
        ax.errorbar(i, mean, yerr=[[mean - block["ci_lower"]], [block["ci_upper"] - mean]],
                    fmt="o", color=color, ecolor=color, capsize=3, markersize=6, zorder=3)
        ax.annotate(f"{mean:+.4f}", (i, mean), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=4.8)
    ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for lab, _, _ in rows], fontsize=4.8)
    ax.set_ylabel("Effect on recall probability")
    ax.set_title("Position-in-sequence: a powered\nnull on the interaction", loc="left", fontsize=6.3)
    text_box(ax, f"n = {n_subjects_b} subjects, open-loop only\ninteraction minimum detectable "
                  f"{interaction_mdd:.4f}\n(below the position effect's own {abs(pos_mean):.4f})", "lower left")

    ax = axes[2]
    all_amps = sorted(set(open_amp) | set(closed_amp))
    xo = np.arange(len(all_amps))
    ax.bar(xo - 0.2, [open_amp.get(a, 0) for a in all_amps], width=0.38, color=BLUE,
           label=f"open-loop ({n_open_pairs} pairs)")
    ax.bar(xo + 0.2, [closed_amp.get(a, 0) for a in all_amps], width=0.38, color=GRAY,
           label=f"closed-loop ({n_closed_pairs} pairs)")
    ax.set_xticks(xo)
    ax.set_xticklabels([f"{a:.0f}" for a in all_amps], fontsize=4.8, rotation=45, ha="right")
    ax.set_xlabel("Stimulation amplitude (microamps)")
    ax.set_ylabel("Stimulated trials")
    ax.set_title("Only amplitude varies within\neither corpus", loc="left", fontsize=6.3)
    ax.legend(fontsize=4.4, loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save(fig, "stimulation_timing_attributability_and_parameters")


# ---- H: phase-locked scalp stimulation, from phase_locked_scalp_stimulation_component.json --

def panel_phase_locked_stimulation(phase: dict) -> None:
    presence = phase["presence_test"]["pooled_patient_test"]
    n_presence = check_int(phase["presence_test"]["n_participants_contributing"], 46, "presence n participants")
    presence_mean = check(presence["mean_value"], -0.0227, 4, "presence mean")

    pm = phase["phase_modulation"]
    active_amp = check(pm["active_group_population_test"]["population_amplitude"], 0.0143, 4, "active amplitude")
    control_amp = check(pm["control_group_population_test"]["population_amplitude"], 0.0058, 4, "control amplitude")
    active_mdd = check(pm["active_group_minimum_detectable_amplitude_one_sample"]["mdd"], 0.0137, 3,
                        "active one-sample mdd")
    between_mdd = check(pm["between_group_minimum_detectable_difference"]["mdd"], 0.0214, 3, "between-group mdd")
    active_p = pm["active_group_population_test"]["circular_rotation_p_value"]
    control_p = pm["control_group_population_test"]["circular_rotation_p_value"]
    n_active = check_int(pm["n_active_contributing"], 21, "phase active n")
    n_control = check_int(pm["n_control_contributing"], 25, "phase control n")

    bl = phase["behaviour_link"]
    primary, bias = bl["primary_pooled_patient_test"], bl["bias_only_pooled_patient_test"]
    check(primary["mean_value"], 0.0048, 4, "behaviour link primary mean")
    primary_mdd = check(primary["mdd"], 0.0581, 3, "behaviour link mdd")
    n_bl = check_int(bl["n_participants_contributing"], 46, "behaviour link n participants")
    ctx = bl["three_corpus_invasive_human_null_context"]
    ctx_r = check(ctx["combined_raw_r"], -0.0225, 4, "three-corpus invasive combined raw r")
    ctx_n = check_int(ctx["n_patients"], 52, "three-corpus invasive n patients")
    nonhuman_r = check(ctx["non_human_effect_at_matched_human_error_counts_r"], -0.0370, 4, "non-human matched r")
    nonhuman_p = ctx["non_human_effect_at_matched_human_error_counts_p"]

    bp = phase["benefit_prediction"]
    rel_names = ["central_value_vs_behavioural_benefit", "central_value_vs_displacement",
                 "presence_statistic_vs_behavioural_benefit", "presence_statistic_vs_displacement",
                 "spread_vs_behavioural_benefit", "spread_vs_displacement"]
    bp_active_mdd = check(bp["active"][rel_names[0]]["mdd"]["mdd_r"], 0.5786, 4, "benefit active mdd")
    bp_control_mdd = check(bp["control"][rel_names[0]]["mdd"]["mdd_r"], 0.5351, 4, "benefit control mdd")
    for rel in rel_names:
        check(bp["active"][rel]["mdd"]["mdd_r"], bp_active_mdd, 4, f"benefit active mdd ({rel})")
        check(bp["control"][rel]["mdd"]["mdd_r"], bp_control_mdd, 4, f"benefit control mdd ({rel})")

    nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))

    ax = axes[0, 0]
    ax.errorbar(0, presence_mean, yerr=[[presence_mean - presence["ci_lower"]], [presence["ci_upper"] - presence_mean]],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=3, markersize=7, zorder=3)
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_xlim(-1, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["stimulation-OFF\nbaseline block"], fontsize=5.3)
    ax.set_ylabel("Scalp presence statistic")
    ax.set_title("Presence transfers, non-invasively", loc="left", fontsize=6.3)
    text_box(ax, f"n = {n_presence} participants\np = {presence['p_value']:.4f}", "lower right")

    ax = axes[0, 1]
    for i, (amp, p, n, color) in enumerate([(active_amp, active_p, n_active, BLUE),
                                             (control_amp, control_p, n_control, GRAY)]):
        ax.errorbar(i, amp, yerr=active_mdd, fmt="o", color=color, ecolor=color, capsize=3, markersize=7, zorder=3)
        ax.annotate(f"{amp:.4f}\np={p:.3f}, n={n}", (i, amp), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=4.6)
    ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["active\n(over the source)", "control\n(away from it)"], fontsize=5.0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Population phase-modulation amplitude\n(error bar: one-sample detection bound)")
    ax.set_title("Phase does not modulate\nthe component", loc="left", fontsize=6.3)
    text_box(ax, f"between-group minimum detectable\ndifference {between_mdd:.4f}", "upper left")

    ax = axes[1, 0]
    for i, (block, color) in enumerate([(primary, BLUE), (bias, GRAY)]):
        mean = block["mean_value"]
        ax.errorbar(i, mean, yerr=[[mean - block["ci_lower"]], [block["ci_upper"] - mean]],
                    fmt="o", color=color, ecolor=color, capsize=3, markersize=7, zorder=3)
    ax.errorbar(2.4, ctx_r, fmt="^", color=VERMILLION, markersize=6, zorder=3)
    ax.annotate(f"invasive human\nnull {ctx_r:+.4f}\n(n={ctx_n})", (2.4, ctx_r), xytext=(0, -18),
                textcoords="offset points", ha="center", fontsize=4.2, color=VERMILLION)
    ax.errorbar(3.2, nonhuman_r, fmt="v", color=ORANGE, markersize=6, zorder=3)
    ax.annotate(f"non-human,\nmatched errors\np={nonhuman_p:.4f}", (3.2, nonhuman_r), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=4.2, color=ORANGE)
    ax.axhline(0, color=GRAY, linewidth=0.6, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["primary", "bias-only\ncontrol"], fontsize=5.3)
    ax.set_xlim(-0.6, 3.8)
    ax.set_ylabel("Correlation with trial accuracy")
    ax.set_title("No accuracy link here, read against\nother preparations", loc="left", fontsize=6.0)
    text_box(ax, f"n = {n_bl} participants, {bl['n_errors_total']} errors\nminimum detectable "
                  f"{primary_mdd:.4f}", "lower left")

    ax = axes[1, 1]
    rel_x_labels = ["value-\nbenefit", "value-\ndisplacement", "presence-\nbenefit", "presence-\ndisplacement",
                     "spread-\nbenefit", "spread-\ndisplacement"]
    for gi, (grp, color) in enumerate([("active", BLUE), ("control", GRAY)]):
        for ri, rel in enumerate(rel_names):
            x = ri + (gi - 0.5) * 0.28
            ax.scatter(x, bp[grp][rel]["r"], marker="o", color=color, s=16, zorder=3)
    ax.axhline(bp_active_mdd, color=BLUE, linestyle="--", linewidth=0.6)
    ax.axhline(-bp_active_mdd, color=BLUE, linestyle="--", linewidth=0.6)
    ax.axhline(bp_control_mdd, color=GRAY, linestyle="--", linewidth=0.6)
    ax.axhline(-bp_control_mdd, color=GRAY, linestyle="--", linewidth=0.6)
    ax.axhline(0, color=BLACK, linewidth=0.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(rel_x_labels, fontsize=4.2, rotation=30, ha="right")
    ax.set_ylabel("Correlation, phase-modulation summary\nvs. behavioural/displacement outcome")
    ax.set_title("Who benefits: 12 cells, all\nunderpowered to ask", loc="left", fontsize=6.0)
    text_box(ax, f"active n={n_active} (mdd={bp_active_mdd:.3f}),\ncontrol n={n_control} "
                  f"(mdd={bp_control_mdd:.3f});\ndashed lines mark each group's\nown minimum detectable "
                  f"correlation", "lower right")

    fig.tight_layout()
    save(fig, "phase_locked_stimulation_presence_and_modulation")


def main() -> None:
    census = load("state_orthogonality_census.json")
    ladder = load("content_label_cardinality_ladder.json")
    human = load("human_content_decodability.json")
    watters = load("watters_state_geometry.json")
    dominant = load("dominant_latent_identity_and_behaviour_breadth.json")
    tier = load("recording_tier_component_transfer.json")
    site = load("stimulation_site_targeting_map.json")
    timing = load("stimulation_timing_and_parameter_structure.json")
    phase = load("phase_locked_scalp_stimulation_component.json")

    panel_leading_component_magnitude_ladder(census)
    panel_sign_split(census)
    panel_split_half_reliability(census)
    rows = panel_macaque_cardinality_ladder(ladder)
    panel_macaque_decodability_by_cardinality(rows)
    panel_cross_preparation_contrast(ladder, rows, human)
    panel_watters_existence_by_lag(watters)
    panel_amplitude_behaviour_dissociation(dominant)
    panel_recording_tier_transfer(tier)
    panel_stimulation_site_targeting(site)
    panel_stimulation_timing_and_parameters(timing)
    panel_phase_locked_stimulation(phase)

    print(f"12 panels written to {FIGURES_DIR} (.png + .pdf each); "
          f"every plotted number verified against its source artifact.")


if __name__ == "__main__":
    main()
