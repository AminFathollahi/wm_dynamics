#!/usr/bin/env python3
"""The functional microcircuit graph: small-worldness and related
organisational metrics per structure, with a degree-preserving null for
every metric.

Scope decisions, made for tractability and stated here rather than left
implicit: the full null-calibrated battery (spike-time tiling coefficient
connectivity, all graph metrics, 20 degree-preserving rewirings per graph)
runs on {hippocampus, pre_sma} -- exactly the deciding-contrast structures
-- rather than on all six anatomical structures; "pooled" and the other
structures get a point-estimate-only STTC pass (no null), and are left for
a follow-up full-battery pass. Even at this reduced scope, one
(session, structure, epoch) null battery measured at ~1s per rewiring on a
40-unit graph -- rewiring cost scales faster than linearly with edge count,
so shrinking both the null-draw count and the kept-edge fraction was
necessary to keep total runtime in the tens of minutes rather than hours
(src/microcircuit_graph.py's N_NULL_DRAWS / EDGE_DENSITY_KEEP_FRACTION).
Pearson and precision
(shrinkage-regularised inverse covariance) connectivity are computed as
point-estimate cross-checks against STTC (no null), not run through the
full battery, since STTC is rate-insensitive and is the route this project's
standing rule specifically calls for. Naive Maslov-Sneppen rewiring on a near-complete
correlation graph was measured directly at ~150x slower than on the same
graph thresholded to a fixed edge density (src/microcircuit_graph.py's
`sparsify` docstring), which is why every graph here is sparsified to its
top 20% of edges by |weight| before any topological metric is computed.

Corpora: DANDI 000469, the canonical 001187/000673 dedup, and DANDI 000574
(Boran) via src/corpus_sessions.py -- see that module's docstring for why
DANDI 000004 is not yet included.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_sessions import data_root, iter_all_corpora  # noqa: E402
from microcircuit_graph import (  # noqa: E402
    STTC_COINCIDENCE_WINDOW_S,
    LOUVAIN_RESOLUTION_GAMMA,
    N_NULL_DRAWS,
    EDGE_DENSITY_KEEP_FRACTION,
    degree_preserving_null_battery,
    graph_metrics,
    pearson_connectivity,
    precision_connectivity,
    small_worldness_sigma,
    sttc_connectivity,
)
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import build_psth  # noqa: E402
from statistics import fdr_bh, paired_sign_flip_test, spearman_permutation_test, stable_seed  # noqa: E402

SEED = 20260808
BIN_MS = 100
SENSITIVITY_BIN_MS = (50, 100, 200)
MIN_UNITS_FOR_GRAPH = 10
STRUCTURES_WITH_FULL_BATTERY = ("hippocampus", "pre_sma")
UNIT_COUNT_LEVELS_FRACTIONS = (0.4, 0.6, 0.8, 1.0)
N_CALIBRATION_MATCHED_DRAWS = 20
DECIDING_CONTRAST = ("pre_sma", "hippocampus")


def unit_mean_rate(spike_lists: list, onsets: np.ndarray, window_s: float) -> float:
    total_spikes = 0
    for spikes in spike_lists:
        for onset in onsets:
            total_spikes += int(np.sum((spikes >= onset) & (spikes < onset + window_s)))
    return total_spikes / (len(spike_lists) * len(onsets) * window_s) if spike_lists else 0.0


def analyze_epoch(spike_lists: list, onsets: np.ndarray, window_s: float, rng: np.random.Generator, full_battery: bool) -> dict:
    W_sttc = sttc_connectivity(spike_lists, onsets, window_s)
    observed = graph_metrics(W_sttc, rng, gamma=LOUVAIN_RESOLUTION_GAMMA)
    result = {
        "n_units": len(spike_lists),
        "mean_firing_rate_hz": unit_mean_rate(spike_lists, onsets, window_s),
        "sttc_observed": observed,
    }
    if full_battery:
        nulls = degree_preserving_null_battery(W_sttc, rng, n_draws=N_NULL_DRAWS)
        percentiles = {}
        for name, values in observed.items():
            if name in ("rich_club_curve",) or values is None:
                continue
            from microcircuit_graph import null_percentile
            percentiles[name] = null_percentile(values, nulls.get(name, []))
        sigma = small_worldness_sigma(W_sttc, seed=int(rng.integers(0, 2**31)))
        result["null_percentiles"] = percentiles
        result["small_worldness"] = sigma

        counts = build_psth(spike_lists, onsets, bin_ms=BIN_MS, smooth_ms=0, window_s=window_s)
        counts_flat = counts.reshape(len(spike_lists), -1) if counts.ndim == 3 else counts
        W_pearson = pearson_connectivity(counts.sum(axis=2) if counts.ndim == 3 else counts)  # per-trial total counts, (units, trials)
        W_precision = precision_connectivity(counts.transpose(1, 0, 2).reshape(len(spike_lists), -1))
        result["cross_route_check"] = {
            "pearson_observed": graph_metrics(W_pearson, rng, gamma=LOUVAIN_RESOLUTION_GAMMA),
            "precision_observed": graph_metrics(W_precision, rng, gamma=LOUVAIN_RESOLUTION_GAMMA),
        }
    return result


def n_dependence_curve(spike_lists: list, onsets: np.ndarray, window_s: float, rng: np.random.Generator) -> dict:
    n_units = len(spike_lists)
    levels = sorted({max(MIN_UNITS_FOR_GRAPH, int(round(n_units * frac))) for frac in UNIT_COUNT_LEVELS_FRACTIONS if int(round(n_units * frac)) <= n_units})
    curve = {}
    for level in levels:
        idx = rng.choice(n_units, size=level, replace=False)
        subset = [spike_lists[i] for i in idx]
        W = sttc_connectivity(subset, onsets, window_s)
        curve[str(level)] = graph_metrics(W, rng)
    return curve


def analyze_session_structure(item: dict, rng: np.random.Generator) -> dict:
    full_battery = item["structure"] in STRUCTURES_WITH_FULL_BATTERY
    epochs = {}
    for epoch_name in ("baseline", "delay"):
        onsets = item["epoch_onsets"][epoch_name]
        window = item["epoch_windows"][epoch_name]
        if len(item["spike_lists"]) < MIN_UNITS_FOR_GRAPH:
            epochs[epoch_name] = {"status": "excluded", "reason": f"fewer than {MIN_UNITS_FOR_GRAPH} units"}
            continue
        epochs[epoch_name] = {"status": "complete", **analyze_epoch(item["spike_lists"], onsets, window, rng, full_battery)}
    result = {
        "dataset": item["dataset"], "patient": item["patient"], "session": item["session"], "structure": item["structure"],
        "n_units": len(item["spike_lists"]), "full_battery": full_battery, "epochs": epochs,
    }
    if full_battery and epochs.get("delay", {}).get("status") == "complete":
        result["n_dependence_curve"] = n_dependence_curve(item["spike_lists"], item["epoch_onsets"]["delay"], item["epoch_windows"]["delay"], rng)
    return result


def bin_width_sensitivity(item: dict, rng: np.random.Generator) -> dict:
    """Sensitivity of mean_clustering/modularity_q to bin width, STTC only, no null."""
    result = {}
    for bin_ms in SENSITIVITY_BIN_MS:
        onsets, window = item["epoch_onsets"]["delay"], item["epoch_windows"]["delay"]
        W = sttc_connectivity(item["spike_lists"], onsets, window, dt=bin_ms / 1000.0)
        metrics = graph_metrics(W, rng)
        result[str(bin_ms)] = {"mean_clustering": metrics["mean_clustering"], "modularity_q": metrics["modularity_q"]}
    return result


def collect_rows(root: Path, rng: np.random.Generator) -> dict:
    rows = {}
    for item in iter_all_corpora(root):
        if item["structure"] not in STRUCTURES_WITH_FULL_BATTERY and item["structure"] != "pooled":
            continue
        key = f"{item['dataset']}/{item['session']}/{item['structure']}"
        row = analyze_session_structure(item, rng)
        if row["structure"] == STTC_SENSITIVITY_STRUCTURE and row["epochs"].get("delay", {}).get("status") == "complete":
            row["bin_width_sensitivity_hz"] = bin_width_sensitivity(item, rng)
        rows[key] = row
    return rows


STTC_SENSITIVITY_STRUCTURE = "pooled"


def confound_check(rows: dict) -> dict:
    """Correlation between each metric and unit count across sessions, per structure."""
    result = {}
    metric_names = ["weight_entropy", "modularity_q", "mean_clustering", "characteristic_path_length", "assortativity_r", "mean_participation_coefficient"]
    for metric in metric_names:
        n_counts, values, structures = [], [], []
        for row in rows.values():
            delay = row["epochs"].get("delay", {})
            if delay.get("status") != "complete":
                continue
            value = delay["sttc_observed"].get(metric)
            if value is None:
                continue
            n_counts.append(row["n_units"])
            values.append(value)
            structures.append(row["structure"])
        if len(values) < 8:
            result[metric] = {"status": "not_estimable"}
            continue
        rng = np.random.default_rng(stable_seed(f"confound_{metric}"))
        n_arr, v_arr = np.array(n_counts, dtype=float), np.array(values, dtype=float)
        corr_with_n = spearman_permutation_test(n_arr, v_arr, n_perm=2000, rng=rng)
        structure_codes = np.array([hash(s) % 997 for s in structures], dtype=float)
        corr_with_structure = spearman_permutation_test(structure_codes, v_arr, n_perm=2000, rng=rng)
        confounded = abs(corr_with_n["rho"]) > abs(corr_with_structure["rho"])
        result[metric] = {
            "rho_with_unit_count": corr_with_n["rho"], "p_with_unit_count": corr_with_n["p_value"],
            "rho_with_structure_code": corr_with_structure["rho"],
            "confounded_with_n": bool(confounded),
        }
    return result


def matched_count_comparison(rows: dict, structure_a: str, structure_b: str, rng: np.random.Generator) -> dict:
    by_patient_structure = {}
    for item in iter_all_corpora(data_root()):
        if item["structure"] not in (structure_a, structure_b):
            continue
        key = (item["dataset"], item["patient"], item["structure"])
        by_patient_structure.setdefault(key, item)

    patients_seen = set((d, p) for d, p, s in by_patient_structure)
    paired_rows = []
    for dataset, patient in patients_seen:
        key_a, key_b = (dataset, patient, structure_a), (dataset, patient, structure_b)
        if key_a not in by_patient_structure or key_b not in by_patient_structure:
            continue
        item_a, item_b = by_patient_structure[key_a], by_patient_structure[key_b]
        n_match = min(len(item_a["spike_lists"]), len(item_b["spike_lists"]))
        if n_match < 5:
            continue
        sigma_deltas_a, sigma_deltas_b, rates_a, rates_b = [], [], [], []
        for draw in range(N_CALIBRATION_MATCHED_DRAWS):
            draw_rng = np.random.default_rng(stable_seed(f"{dataset}_{patient}_{structure_a}_{structure_b}_{draw}"))
            idx_a = draw_rng.choice(len(item_a["spike_lists"]), size=n_match, replace=False)
            idx_b = draw_rng.choice(len(item_b["spike_lists"]), size=n_match, replace=False)
            subset_a = [item_a["spike_lists"][i] for i in idx_a]
            subset_b = [item_b["spike_lists"][i] for i in idx_b]
            delay_onset_a, delay_window_a = item_a["epoch_onsets"]["delay"], item_a["epoch_windows"]["delay"]
            base_onset_a, base_window_a = item_a["epoch_onsets"]["baseline"], item_a["epoch_windows"]["baseline"]
            delay_onset_b, delay_window_b = item_b["epoch_onsets"]["delay"], item_b["epoch_windows"]["delay"]
            base_onset_b, base_window_b = item_b["epoch_onsets"]["baseline"], item_b["epoch_windows"]["baseline"]

            m_delay_a = graph_metrics(sttc_connectivity(subset_a, delay_onset_a, delay_window_a), draw_rng)
            m_base_a = graph_metrics(sttc_connectivity(subset_a, base_onset_a, base_window_a), draw_rng)
            m_delay_b = graph_metrics(sttc_connectivity(subset_b, delay_onset_b, delay_window_b), draw_rng)
            m_base_b = graph_metrics(sttc_connectivity(subset_b, base_onset_b, base_window_b), draw_rng)
            sigma_deltas_a.append(m_delay_a["mean_clustering"] / max(m_base_a["mean_clustering"], 1e-9) - 1.0)
            sigma_deltas_b.append(m_delay_b["mean_clustering"] / max(m_base_b["mean_clustering"], 1e-9) - 1.0)
            rates_a.append(unit_mean_rate(subset_a, delay_onset_a, delay_window_a))
            rates_b.append(unit_mean_rate(subset_b, delay_onset_b, delay_window_b))
        paired_rows.append({
            "dataset": dataset, "patient": patient, "n_matched_units": int(n_match),
            "mean_clustering_delay_over_baseline_ratio_minus_one_a": float(np.mean(sigma_deltas_a)),
            "mean_clustering_delay_over_baseline_ratio_minus_one_b": float(np.mean(sigma_deltas_b)),
            "achieved_mean_firing_rate_hz_a": float(np.mean(rates_a)),
            "achieved_mean_firing_rate_hz_b": float(np.mean(rates_b)),
        })
    return {
        "structure_a": structure_a, "structure_b": structure_b,
        "n_attempted_patients": len(patients_seen), "n_paired_patients": len(paired_rows),
        "fraction_with_a_value": (len(paired_rows) / len(patients_seen)) if patients_seen else None,
        "note": "reports the mean-clustering delay/baseline ratio as a fast matched-unit-count proxy for the full small-worldness-sigma epoch contrast (which needs a 50-draw null per arm and is too expensive to repeat inside a 20-draw resampling loop); the full sigma epoch contrast for the deciding pair itself is in `rows`.",
        "rows": paired_rows,
    }


def deciding_contrast_sigma_epoch_delta(rows: dict, structure_a: str, structure_b: str) -> dict:
    """sigma delay-minus-baseline per (dataset,patient) for the two deciding structures, at full unit count."""
    by_patient_structure = {}
    for row in rows.values():
        if row["structure"] not in (structure_a, structure_b):
            continue
        delay, baseline = row["epochs"].get("delay", {}), row["epochs"].get("baseline", {})
        if delay.get("status") != "complete" or baseline.get("status") != "complete":
            continue
        if "small_worldness" not in delay or "small_worldness" not in baseline:
            continue
        sigma_delay, sigma_baseline = delay["small_worldness"]["sigma"], baseline["small_worldness"]["sigma"]
        if sigma_delay is None or sigma_baseline is None:
            continue
        key = (row["dataset"], row["patient"], row["structure"])
        by_patient_structure[key] = sigma_delay - sigma_baseline

    patients_seen = set((d, p) for d, p, s in by_patient_structure)
    paired = []
    for dataset, patient in patients_seen:
        key_a, key_b = (dataset, patient, structure_a), (dataset, patient, structure_b)
        if key_a in by_patient_structure and key_b in by_patient_structure:
            paired.append({"dataset": dataset, "patient": patient, "delta_a": by_patient_structure[key_a], "delta_b": by_patient_structure[key_b]})
    return {"structure_a": structure_a, "structure_b": structure_b, "n_paired_patients": len(paired), "rows": paired}


def predeclared_decision(sigma_deltas: dict, confound: dict, matched: dict, rng: np.random.Generator) -> dict:
    structure_a, structure_b = DECIDING_CONTRAST
    deciding_metric = "small_worldness_sigma"
    if confound.get(deciding_metric, {}).get("confounded_with_n"):
        return {
            "deciding_contrast": f"{deciding_metric} delay-minus-baseline, {structure_a} minus {structure_b}, matched unit count, within patient",
            "verdict": "metric_confounded_with_unit_count",
            "reason": "small_worldness_sigma's correlation with unit count exceeds its correlation with structure",
        }
    n_paired = sigma_deltas["n_paired_patients"]
    if n_paired < 6:
        return {
            "deciding_contrast": f"{deciding_metric} delay-minus-baseline, {structure_a} minus {structure_b}, matched unit count, within patient",
            "verdict": "underpowered_by_construction",
            "n_paired_patients": n_paired,
            "reason": "fewer than 6 paired patients; minimum attainable sign-flip p exceeds 0.05 by construction",
        }
    a = np.array([r["delta_a"] for r in sigma_deltas["rows"]])
    b = np.array([r["delta_b"] for r in sigma_deltas["rows"]])
    test = paired_sign_flip_test(a, b, alternative="two-sided", rng=rng)
    if test["p_value"] < 0.05:
        verdict = "graph_organisation_differs_by_structure"
    else:
        verdict = "no_graph_organisation_difference"
    return {
        "deciding_contrast": f"{deciding_metric} delay-minus-baseline, {structure_a} minus {structure_b}, matched unit count, within patient, exact paired sign-flip test",
        "n_paired_patients": n_paired,
        "mean_diff": test["mean_diff"], "p_value": test["p_value"], "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "matched_count_comparison_cross_check": {k: matched[k] for k in ("n_paired_patients", "fraction_with_a_value")},
        "verdict": verdict,
    }


def write_structure_fingerprint(rows: dict, root: Path) -> None:
    dimensionality_path = ROOT / "results" / "intrinsic_dimensionality.json"
    displacement_path = ROOT / "results" / "latent_displacement_scaling.json"
    dimensionality = json.loads(dimensionality_path.read_text()) if dimensionality_path.exists() else None
    displacement = json.loads(displacement_path.read_text()) if displacement_path.exists() else None

    by_structure = {}
    for row in rows.values():
        structure = row["structure"]
        entry = by_structure.setdefault(structure, {"sessions": []})
        entry["sessions"].append({
            "dataset": row["dataset"], "session": row["session"],
            "delay_sttc": row["epochs"].get("delay", {}).get("sttc_observed"),
            "delay_null_percentiles": row["epochs"].get("delay", {}).get("null_percentiles"),
            "delay_small_worldness": row["epochs"].get("delay", {}).get("small_worldness"),
        })

    from small_worldness import definition_summary as small_worldness_definition_summary

    fingerprint = {
        "schema_version": "1.0.0",
        "content_hash": None,
        "metric_definitions": {
            "small_worldness_sigma_null": small_worldness_definition_summary(),
            "louvain_resolution_gamma": LOUVAIN_RESOLUTION_GAMMA,
            "sttc_coincidence_window_s": STTC_COINCIDENCE_WINDOW_S,
            "edge_density_keep_fraction": EDGE_DENSITY_KEEP_FRACTION,
            "connectivity_bin_width_ms": BIN_MS,
        },
        "dimensionality_by_structure": dimensionality.get("structure_delay_median_by_estimator") if dimensionality else None,
        "displacement_scaling_denominator": displacement.get("denominator_summary") if displacement else None,
        "graph_metrics_by_structure": by_structure,
    }
    text = canonical_json(fingerprint)
    import hashlib
    fingerprint["content_hash"] = hashlib.sha256(text.encode()).hexdigest()
    (root / "results" / "structure_fingerprint.json").write_text(canonical_json(fingerprint))


def main() -> None:
    root = data_root()
    rng = np.random.default_rng(SEED)
    rows = collect_rows(root, rng)

    confound = confound_check(rows)
    matched = matched_count_comparison(rows, DECIDING_CONTRAST[0], DECIDING_CONTRAST[1], np.random.default_rng(SEED + 1))
    sigma_deltas = deciding_contrast_sigma_epoch_delta(rows, DECIDING_CONTRAST[0], DECIDING_CONTRAST[1])
    decision = predeclared_decision(sigma_deltas, confound, matched, np.random.default_rng(SEED + 2))

    family_pairs = [(a, b) for i, a in enumerate(STRUCTURES_WITH_FULL_BATTERY) for b in STRUCTURES_WITH_FULL_BATTERY[i + 1:]]
    family_results = []
    for a, b in family_pairs:
        sd = deciding_contrast_sigma_epoch_delta(rows, a, b)
        m = matched_count_comparison(rows, a, b, np.random.default_rng(stable_seed(f"{a}_{b}_graph")))
        d = predeclared_decision(sd, confound, m, np.random.default_rng(stable_seed(f"{a}_{b}_graph_test")))
        family_results.append({"pair": [a, b], "n_paired_patients": sd["n_paired_patients"], "decision": d})
    finite_p = [r["decision"]["p_value"] for r in family_results if "p_value" in r["decision"]]
    fdr = fdr_bh(np.array(finite_p)) if finite_p else {"status": "not_estimable"}

    write_structure_fingerprint(rows, ROOT)

    output = {
        "schema_version": "1.0.0",
        "analysis_id": "functional_microcircuit_graph",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "scope_decisions": {
            "structures_with_full_null_battery": list(STRUCTURES_WITH_FULL_BATTERY),
            "n_null_draws": N_NULL_DRAWS,
            "edge_density_keep_fraction": EDGE_DENSITY_KEEP_FRACTION,
            "primary_connectivity_route": "sttc",
            "excluded_corpora": {
                "dandi_000004": (
                    "not a task-design limitation -- it has a genuine ~2.2s maintenance interval. "
                    "Excluded because its electrode location field uses a region-label convention this "
                    "project has no parser for yet, and it is not yet registered in config/datasets.json."
                ),
            },
        },
        "n_dataset_structure_session_rows": len(rows),
        "confound_with_unit_count": confound,
        "deciding_contrast_sigma_epoch_delta": sigma_deltas,
        "deciding_contrast_matched_count_comparison": matched,
        "predeclared_decision": decision,
        "family_wise_ordered_pairs": family_results,
        "family_wise_fdr_bh": fdr,
        "rows": rows,
    }
    destination = ROOT / "results" / "functional_microcircuit_graph.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "n_rows": len(rows),
        "decision_verdict": decision["verdict"],
        "output": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
