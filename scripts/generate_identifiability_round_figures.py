#!/usr/bin/env python3
"""Figures for five artifacts: the tau estimator
calibration, the lambda estimator's no-signal/random-walk limits, the
structure identifiability model, the paired structure contrasts, and the
fidelity-controllability map.

A separate script from generate_paper_figures.py rather than an extension of
it: every other figure in that file is built from `load_all_stats()` /
`results/all_statistics.json` and per-subject geometry arrays, a genuinely
different data pipeline from the five self-contained JSON artifacts this
script reads directly. Uses the same Nature-style plotting utilities
(src/visualization.py: nature_style, save_figure, PALETTE) so the output is
visually consistent with the rest of the figure set.

A figure whose underlying number is non_identified shows that status as an
annotated gap, never an empty panel or an interpolated point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visualization import PALETTE, nature_style, save_figure  # noqa: E402

RESULTS = ROOT / "results"
REGION_COLOR = {
    "hippocampus": "#4E79A7", "amygdala": "#59A14F", "pre_sma": "#E15759",
    "dacc": "#F28E2B", "vmpfc": "#76B7B2", "unspecific": "#BAB0AC",
    "entorhinal_parahippocampal": "#B07AA1", "inferior_temporal_gyrus": "#9C755F",
    "middle_temporal_gyrus": "#EDC948", "superior_temporal_gyrus": "#FF9DA7",
    "vtc": "#BAB0AC", "unlabelled": "#BAB0AC",
}


def _region_color(region: str) -> str:
    return REGION_COLOR.get(region, "#888888")


def _annotate_gap(ax, x, y, text="non_identified"):
    ax.text(x, y, text, fontsize=5.5, color="#999999", ha="center", va="center",
             rotation=90, style="italic")


def figure_tau_estimator_calibration() -> None:
    path = RESULTS / "tau_estimator_calibration.json"
    if not path.exists():
        print("  skip tau_estimator_calibration: artifact missing")
        return
    d = json.loads(path.read_text())
    rates = d["mean_rates_hz"]
    taus = d["planted_taus_s"]
    cells = {(c["planted_tau_s"], c["mean_rate_hz"]): c for c in d["cells"]}
    lag0_cells = {(c["planted_tau_s"], c["mean_rate_hz"]): c for c in d["lag0_inclusive_comparison_for_figure"]}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cmap = plt.get_cmap("viridis")
    for panel, (ax, cell_map, title) in enumerate([
        (axes[0], cells, "corrected (lag 0 excluded)"),
        (axes[1], lag0_cells, "old (lag 0 included)"),
    ]):
        ax.axhspan(0.05, 0.35, color="#F0E68C", alpha=0.3, zorder=0, label="Murray et al. 2014 range")
        ax.plot([0, 0.55], [0, 0.55], "k--", linewidth=0.6, label="identity")
        for i, rate in enumerate(rates):
            xs, ys = [], []
            for tau in taus:
                c = cell_map.get((tau, rate))
                if c and c["median_recovered_tau_s"] is not None:
                    xs.append(tau)
                    ys.append(min(c["median_recovered_tau_s"], 0.55))
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=0.8,
                     color=cmap(i / max(len(rates) - 1, 1)), label=f"{rate:g} Hz")
        ax.set_xlabel("planted tau (s)")
        ax.set_ylabel("recovered tau (s)" if panel == 0 else "")
        ax.set_title(title, fontsize=7)
        ax.set_xlim(0, 0.55)
        ax.set_ylim(0, 0.55)
    axes[1].legend(fontsize=4.5, loc="upper left", frameon=False)
    fig.suptitle(
        f"tau estimator calibration -- resolvable in {d['resolvable_fraction_overall']:.0%} of "
        f"{len(d['cells'])} cells at the observed n_trials={d['n_trials_used']}, "
        f"window={d['baseline_window_s']}s (results/tau_estimator_calibration.json, n=25 cells x "
        f"{d['n_reps_per_cell']} reps)",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "tau_estimator_calibration")
    plt.close(fig)
    print("  tau_estimator_calibration.pdf saved.")


def figure_lambda_estimator_limits() -> None:
    path = RESULTS / "lambda_estimator_limits.json"
    if not path.exists():
        print("  skip lambda_estimator_limits: artifact missing")
        return
    d = json.loads(path.read_text())
    fp = d["fingerprint_comparison_random_walk_vs_no_signal"]
    confined_cells = d["true_confinement_present"]
    lo, hi = -10.0, 60.0

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)

    ax = axes[0]
    for lam_str, cell in sorted(confined_cells.items(), key=lambda kv: float(kv[0])):
        if cell["median_lambda_hat"] is None:
            continue
        ax.errorbar(
            [float(lam_str)], [min(cell["median_lambda_hat"], hi)],
            yerr=[[max(0, min(cell["median_lambda_hat"], hi) - max(cell["lambda_hat_ci95"][0], lo))],
                  [max(0, min(cell["lambda_hat_ci95"][1], hi) - min(cell["median_lambda_hat"], hi))]],
            fmt="o", markersize=3, color="#4E79A7", capsize=2, linewidth=0.8,
        )
    ax.plot([0, 3.5], [0, 3.5], "k--", linewidth=0.6)
    ax.set_xlabel("planted lambda (s$^{-1}$)")
    ax.set_ylabel("lambda-hat (s$^{-1}$)")
    ax.set_title(f"true confinement\n(identified fraction printed)", fontsize=6)
    for lam_str, cell in confined_cells.items():
        ax.text(float(lam_str), -6, f"{cell['identified_fraction']:.0%}", fontsize=5, ha="center", color="#555555")

    for ax, key, title, color in [
        (axes[1], "random_walk", "true random walk\n(lambda=0)", "#59A14F"),
        (axes[2], "no_signal", "no latent signal\n(independent noise)", "#E15759"),
    ]:
        block = fp[key]
        finite_note = f"n={block['n_finite_lambda_hat']}"
        median = block["median_lambda_hat"]
        ci = block["lambda_hat_ci95"]
        if median is not None:
            ax.errorbar([0], [median], yerr=[[median - ci[0]], [ci[1] - median]],
                        fmt="s", markersize=4, color=color, capsize=3)
        ax.axhline(0, color="#999999", linewidth=0.5, linestyle=":")
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=6)
        ax.text(0, hi * 0.9, f"identified: {block['identified_fraction']:.0%}\n{finite_note}",
                 fontsize=5.5, ha="center", color="#555555")
    axes[0].set_ylim(lo, hi)

    fig.suptitle(
        "lambda estimator limits (results/lambda_estimator_limits.json) -- "
        f"random-walk vs no-signal fingerprints separable: {fp['fingerprints_separable']}",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, "lambda_estimator_limits")
    plt.close(fig)
    print("  lambda_estimator_limits.pdf saved.")


def figure_structure_identifiability() -> None:
    path = RESULTS / "structure_identifiability_model.json"
    if not path.exists():
        print("  skip structure_identifiability: artifact missing")
        return
    d = json.loads(path.read_text())
    model = d["mixed_effects_logistic_model"]
    eligibility = d["eligibility_table_summary"]

    raw_by_structure: dict[str, tuple[int, int]] = {}
    for key, row in eligibility.items():
        dataset, structure = key.split("::")
        n_id, n_fit = raw_by_structure.get(structure, (0, 0))
        raw_by_structure[structure] = (n_id + row["n_folds_identified"], n_fit + row["n_folds_fit"])

    # A structure with zero attempted folds carries no information -- an annotated gap, not a
    # misleading 0-height bar that would read as "identified 0% of the time".
    plotted_structures = sorted(
        (s for s, (n_id, n_fit) in raw_by_structure.items() if n_fit > 0),
        key=lambda s: -(raw_by_structure[s][0] / raw_by_structure[s][1]),
    )
    zero_fit_structures = sorted(s for s, (n_id, n_fit) in raw_by_structure.items() if n_fit == 0)
    marginal = model.get("marginal_predicted_identification_probability_at_grand_median", {})

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(plotted_structures))
    raw_fracs = [raw_by_structure[s][0] / raw_by_structure[s][1] for s in plotted_structures]
    ax.bar(x - 0.18, raw_fracs, width=0.34, color="#BAB0AC", label="raw fraction identified")
    model_fracs = [marginal.get(s) for s in plotted_structures]
    model_x = [xi + 0.18 for xi, m in zip(x, model_fracs) if m is not None]
    model_y = [m for m in model_fracs if m is not None]
    if model.get("status") == "estimable":
        ax.bar(model_x, model_y, width=0.34, color="#4E79A7", label="covariate-adjusted (grand median)")
    y_max = max(raw_fracs + model_y + [0.05]) * 1.35
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(plotted_structures, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("P(fold identified)")
    for xi, s in zip(x, plotted_structures):
        n_id, n_fit = raw_by_structure[s]
        frac = n_id / n_fit
        ax.text(xi - 0.18, frac + 0.01 * y_max, f"{n_id}/{n_fit}", fontsize=5, ha="center")
        if s in d.get("structures_excluded_for_dataset_collinearity", {}):
            ax.text(xi + 0.18, y_max * 0.5, "collinear\nw/ dataset", fontsize=5, color="#999999",
                     ha="center", va="center", rotation=90, style="italic")
    if zero_fit_structures:
        ax.text(
            0.99, 0.98, "0 attempted folds (not plotted): " + ", ".join(zero_fit_structures),
            transform=ax.transAxes, fontsize=5, ha="right", va="top", color="#999999",
        )
    ax.set_title(
        f"identifiability by structure (verdict: {d['predeclared_decision']['verdict']}) -- "
        "results/structure_identifiability_model.json",
        fontsize=6.5,
    )
    ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    save_figure(fig, "structure_identifiability")
    plt.close(fig)
    print("  structure_identifiability.pdf saved.")


def figure_structure_paired_contrasts() -> None:
    path = RESULTS / "structure_paired_contrasts.json"
    if not path.exists():
        print("  skip structure_paired_contrasts: artifact missing")
        return
    d = json.loads(path.read_text())
    block = d["datasets"]["dandi_000469"]  # producing script hardcodes this key unconditionally; raise if ever absent
    pairs = [p for p in block.get("pair_matrix", []) if p["region_a"] < p["region_b"]]  # unordered, one direction

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y_positions = []
    labels = []
    estimable_means = []
    for i, pair in enumerate(pairs):
        lam = pair["lambda_s_per_s"]
        label = f"{pair['region_a']} - {pair['region_b']}"
        labels.append(label)
        y_positions.append(i)
        if lam["status"] == "estimable":
            estimable_means.append(lam["mean_difference"])
    unpaired_preview = d.get("unpaired_vs_paired_pre_sma_vs_hippocampus", {})
    unpaired_diff = unpaired_preview.get("unpaired_pooled_mean_difference_pre_sma_minus_hippocampus")
    reference_values = estimable_means + ([unpaired_diff] if unpaired_diff is not None else [])
    x_half_range = max([abs(m) for m in reference_values] + [0.3]) * 1.6
    ax.set_xlim(-x_half_range, x_half_range)

    for i, pair in enumerate(pairs):
        lam = pair["lambda_s_per_s"]
        if lam["status"] == "estimable":
            mean = lam["mean_difference"]
            ci = lam["patient_bootstrap_ci95"]
            ax.errorbar([mean], [i], xerr=[[mean - ci[0]], [ci[1] - mean]], fmt="o",
                        color="#4E79A7", markersize=3.5, capsize=2, linewidth=0.9)
            ax.text(mean, i + 0.22, f"n={lam['n_patients_both_structures']}, p={lam.get('sign_flip_p_value', float('nan')):.3f}",
                    fontsize=5, ha="center")
        else:
            ax.text(0, i, f"non_identified (n={lam.get('n_patients_both_structures', 0)}, below threshold of 3)",
                    fontsize=5.5, color="#999999", ha="center", va="center", style="italic")
    unpaired = d.get("unpaired_vs_paired_pre_sma_vs_hippocampus", {})
    if unpaired.get("status") == "estimable":
        diff_pre_sma_minus_hippocampus = unpaired["unpaired_pooled_mean_difference_pre_sma_minus_hippocampus"]
        if "hippocampus - pre_sma" in labels:
            idx, plotted_diff = labels.index("hippocampus - pre_sma"), -diff_pre_sma_minus_hippocampus
        elif "pre_sma - hippocampus" in labels:
            idx, plotted_diff = labels.index("pre_sma - hippocampus"), diff_pre_sma_minus_hippocampus
        else:
            idx = None
        if idx is not None:
            ax.scatter([plotted_diff], [idx], marker="D", color="#E15759", s=18, zorder=5)
            ax.text(plotted_diff, idx - 0.28, "unpaired pooled\ndifference", fontsize=5,
                    ha="center", va="top", color="#E15759")
    ax.axvline(0, color="#999999", linewidth=0.5, linestyle=":")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("paired lambda difference, region_a minus region_b (s$^{-1}$)")
    ax.set_title(
        "within-patient paired structure contrasts, DANDI 000469 (results/structure_paired_contrasts.json)",
        fontsize=6.5,
    )
    fig.tight_layout()
    save_figure(fig, "structure_paired_contrasts")
    plt.close(fig)
    print("  structure_paired_contrasts.pdf saved.")


def figure_fidelity_controllability_map() -> None:
    path = RESULTS / "fidelity_controllability_map.json"
    if not path.exists():
        print("  skip fidelity_controllability_map: artifact missing")
        return
    d = json.loads(path.read_text())
    per_structure = d["per_structure"]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for region, block in per_structure.items():
        if block.get("status") != "estimable":
            continue
        x = block["control_bandwidth_ms_1_over_lambda"]
        y = block["stationary_variance_D_over_lambda"]
        ax.errorbar(
            [x["mean"]], [y["mean"]],
            xerr=[[x["mean"] - x["bootstrap_ci95"][0]], [x["bootstrap_ci95"][1] - x["mean"]]],
            yerr=[[y["mean"] - y["bootstrap_ci95"][0]], [y["bootstrap_ci95"][1] - y["mean"]]],
            fmt="o", markersize=5, color=_region_color(region), capsize=2, linewidth=0.9,
            label=f"{region} (n={block['n_sessions']})",
        )
        ax.annotate(region, (x["mean"], y["mean"]), fontsize=5.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("control bandwidth 1/lambda (ms)")
    ax.set_ylabel("stationary variance D/lambda (state units$^2$)")
    corr = d["rank_correlation_stationary_variance_vs_bandwidth"]
    verdict = d["predeclared_decision"]["verdict"]
    subtitle = ""
    if corr.get("status") == "estimable":
        subtitle = f"Spearman rho={corr['observed_spearman_rho']:.2f}, CI [{corr['bootstrap_ci95'][0]:.2f}, {corr['bootstrap_ci95'][1]:.2f}]"
    ax.set_title(f"fidelity vs. controllability (verdict: {verdict})\n{subtitle}", fontsize=6.5)
    fig.tight_layout()
    save_figure(fig, "fidelity_controllability_map")
    plt.close(fig)
    print("  fidelity_controllability_map.pdf saved.")


def main() -> None:
    nature_style()
    print("Generating identifiability-round figures...")
    figure_tau_estimator_calibration()
    figure_lambda_estimator_limits()
    figure_structure_identifiability()
    figure_structure_paired_contrasts()
    figure_fidelity_controllability_map()
    print("Done.")


if __name__ == "__main__":
    main()
