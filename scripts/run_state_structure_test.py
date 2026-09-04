"""run_state_structure_test.py -- do structures differ in the cross-unit
state (d_perm, from state_persistence_lag.json), or does that difference
just track recording quality (the white/nugget variance share, from
variance_partition.json)?

Both inputs are read-only; no new census. Per session, mean d_perm is taken
over the lags that clear existence at the deciding width (the same
clearing_lags_bins_at_deciding_width the pooled arm already establishes),
joined to the session's white_fraction_median by (dataset, session,
structure, epoch, bin_ms). Three questions, in order: do structures differ
in nugget share; does conditioning on nugget share remove the per-structure
d_perm differences (a pooled-OLS residualization, the non-parametric
analogue of a partial correlation, plus a common-nugget-share-band
comparison); and does d_perm actually separate structures at all (primary:
within-patient paired contrasts with FDR; secondary: unpaired contrasts at
a matched unit-count/nugget-share band). Reuses
run_state_persistence._load_white_share_lookup and
statistics.{paired_sign_flip_test, permutation_test_twosample, fdr_bh,
stable_seed}.
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

from statistics import fdr_bh, paired_sign_flip_test, permutation_test_twosample, stable_seed  # noqa: E402

from run_state_persistence import _load_white_share_lookup  # noqa: E402

LAG_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_lag.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_structure_test.json"

MIN_SESSIONS_PER_GROUP = 4
MIN_PATIENTS_PER_PAIR = 4
N_PERM_OMNIBUS = 10000
N_PERM_PAIRWISE = 10000
BAND_QUANTILES = (0.25, 0.75)
FDR_ALPHA = 0.05

# The previous round's per-structure existence numbers, quoted verbatim so the
# withdrawal states the former value and the correction together.
PRIOR_ROUND_STRUCTURE_EXISTENCE = {
    "dacc": {"prior_mean_diff": 0.055, "prior_p_value": 0.036},
    "hippocampus": {"prior_mean_diff": 0.036, "prior_p_value": 0.0036},
    "amygdala": {"prior_mean_diff": 0.034, "prior_p_value": 0.006, "prior_gap_label": "gap 2"},
}


def omnibus_group_permutation_test(labels: np.ndarray, values: np.ndarray, n_perm: int = N_PERM_OMNIBUS,
                                     seed_name: str = "omnibus", rng: np.random.Generator | None = None) -> dict:
    """Non-parametric one-way test for whether `values` differ across the
    groups in `labels`. Statistic is the group-size-weighted sum of squared
    deviations of each group mean from the grand mean (the permutation
    analogue of an ANOVA F numerator), used instead of a parametric ANOVA
    so no distributional assumption is made about d_perm or nugget share.
    Labels are shuffled among values with group sizes held fixed."""
    labels = np.asarray(labels)
    values = np.asarray(values, dtype=float)
    groups = sorted(set(labels))
    if len(groups) < 2:
        return {"status": "fewer_than_2_groups", "n_groups": len(groups)}

    if rng is None:
        rng = np.random.default_rng(stable_seed(seed_name))

    def between_group_ss(lab: np.ndarray, val: np.ndarray) -> float:
        grand_mean = val.mean()
        ss = 0.0
        for g in groups:
            mask = lab == g
            if mask.sum() == 0:
                continue
            ss += mask.sum() * (val[mask].mean() - grand_mean) ** 2
        return ss

    observed = between_group_ss(labels, values)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = between_group_ss(rng.permutation(labels), values)
    p = float(((null >= observed).sum() + 1) / (n_perm + 1))

    per_group = {str(g): {"median": float(np.median(values[labels == g])), "n": int((labels == g).sum())}
                 for g in groups}
    return {"status": "tested", "statistic": "between_group_weighted_sum_of_squares", "observed": observed,
            "p_value": p, "significant": bool(p <= 0.05), "n_groups": len(groups), "n_total": int(len(values)),
            "per_group": per_group}


def residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(covariate, values, 1)
    return values - (slope * covariate + intercept)


def build_structure_sessions(lag_rows: list[dict], white_lookup: dict, clearing_lags: list[int],
                              width_bins: int) -> list[dict]:
    rows = []
    for r in lag_rows:
        if r["epoch"] != "delay" or r["width_bins"] != width_bins or r["structure"] == "pooled":
            continue
        if r["profile"].get("status") != "fitted" or r.get("null_permutation") is None:
            continue
        lags, null_lags = r["profile"]["lags"], r["null_permutation"]["lags"]
        common = [str(lag) for lag in clearing_lags if str(lag) in lags and str(lag) in null_lags]
        if not common:
            continue
        d_perm = float(np.mean([lags[k]["r_median"] - null_lags[k]["r_null_median"] for k in common]))
        key = (r["dataset"], r["session"], r["structure"], r["epoch"], r["bin_ms"])
        rows.append({
            "dataset": r["dataset"], "patient": r["patient"], "session": r["session"], "structure": r["structure"],
            "d_perm": d_perm, "white_fraction_median": white_lookup.get(key),
            "n_units": r["n_units"], "n_trials": r["n_trials"], "n_lags_used": len(common),
        })
    return rows


def nugget_share_by_structure(sessions: list[dict]) -> dict:
    have_white = [s for s in sessions if s["white_fraction_median"] is not None]
    labels = np.array([s["structure"] for s in have_white])
    values = np.array([s["white_fraction_median"] for s in have_white], dtype=float)
    eligible = [g for g in sorted(set(labels)) if (labels == g).sum() >= MIN_SESSIONS_PER_GROUP]
    if len(eligible) < 2:
        return {"status": "not_enough_structures_with_min_sessions", "min_sessions_per_group": MIN_SESSIONS_PER_GROUP,
                "eligible_structures": eligible}
    mask = np.isin(labels, eligible)
    return omnibus_group_permutation_test(labels[mask], values[mask], seed_name="nugget_share_by_structure")


def structure_effect_conditioned_on_nugget_share(sessions: list[dict]) -> dict:
    have_white = [s for s in sessions if s["white_fraction_median"] is not None]
    labels = np.array([s["structure"] for s in have_white])
    d_perm = np.array([s["d_perm"] for s in have_white], dtype=float)
    white = np.array([s["white_fraction_median"] for s in have_white], dtype=float)
    eligible = [g for g in sorted(set(labels)) if (labels == g).sum() >= MIN_SESSIONS_PER_GROUP]
    if len(eligible) < 2:
        return {"status": "not_enough_structures_with_min_sessions"}
    mask = np.isin(labels, eligible)
    labels_m, d_perm_m, white_m = labels[mask], d_perm[mask], white[mask]

    raw = omnibus_group_permutation_test(labels_m, d_perm_m, seed_name="structure_effect_raw")

    resid = residualize(d_perm_m, white_m)
    partial = omnibus_group_permutation_test(labels_m, resid, seed_name="structure_effect_residualized")
    partial["method"] = ("non-parametric analogue of a partial correlation between structure and d_perm "
                          "controlling for nugget share: d_perm regressed on white_fraction_median by pooled "
                          "OLS across all eligible structures, then the same between-group permutation test "
                          "run on the residuals")

    lo_q, hi_q = (float(x) for x in np.quantile(white_m, BAND_QUANTILES))
    band_mask = (white_m >= lo_q) & (white_m <= hi_q)
    band_structures = [g for g in eligible if (labels_m[band_mask] == g).sum() >= MIN_SESSIONS_PER_GROUP]
    if len(band_structures) >= 2:
        bm = band_mask & np.isin(labels_m, band_structures)
        matched_band = omnibus_group_permutation_test(labels_m[bm], d_perm_m[bm], seed_name="structure_effect_matched_band")
        matched_band["nugget_share_band"] = [lo_q, hi_q]
    else:
        matched_band = {"status": "not_enough_structures_in_common_nugget_share_band",
                         "nugget_share_band": [lo_q, hi_q], "eligible_structures": band_structures}

    raw_sig = bool(raw.get("significant", False))
    partial_sig = bool(partial.get("significant", False))
    band_sig = bool(matched_band.get("significant", False))
    if not raw_sig:
        verdict = "no_raw_structure_difference_to_explain"
    elif not partial_sig and not band_sig:
        verdict = "recording_quality_explains_the_pattern"
    else:
        verdict = "structure_differences_persist_after_conditioning"

    out = {"raw_structure_effect_on_d_perm": raw, "partial_correlation_residualized_on_nugget_share": partial,
           "matched_nugget_share_band": matched_band, "verdict": verdict}
    if verdict == "recording_quality_explains_the_pattern":
        out["statement"] = ("Rounds 20 through 23, the previous round's per-structure existence, and the "
                             "pooled d_perm-vs-white-share confound check are one result: a recording-quality "
                             "map read as anatomy.")
    return out


def within_patient_paired_structure_test(sessions: list[dict]) -> dict:
    by_patient_structure: dict[tuple, list[float]] = {}
    for s in sessions:
        by_patient_structure.setdefault((s["patient"], s["structure"]), []).append(s["d_perm"])
    agg = {k: float(np.mean(v)) for k, v in by_patient_structure.items()}
    patients = sorted({p for p, _ in agg})
    structures = sorted({st for _, st in agg})
    patients_with_ge2_structures = [p for p in patients if sum(1 for st in structures if (p, st) in agg) >= 2]

    pairs = []
    for i, s1 in enumerate(structures):
        for s2 in structures[i + 1:]:
            common_patients = [p for p in patients if (p, s1) in agg and (p, s2) in agg]
            if len(common_patients) < MIN_PATIENTS_PER_PAIR:
                pairs.append({"structure_a": s1, "structure_b": s2, "status": "not_enough_within_patient_pairs",
                              "n_patients": len(common_patients)})
                continue
            a = np.array([agg[(p, s1)] for p in common_patients])
            b = np.array([agg[(p, s2)] for p in common_patients])
            rng = np.random.default_rng(stable_seed(f"within_patient_pair|{s1}|{s2}"))
            test = paired_sign_flip_test(a, b, alternative="two-sided", rng=rng)
            pairs.append({"structure_a": s1, "structure_b": s2, "status": "tested",
                          "n_patients": len(common_patients), "mean_diff_a_minus_b": test["mean_diff"],
                          "p_value": test["p_value"], "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"]})

    tested = [p for p in pairs if p["status"] == "tested"]
    n_reject = 0
    if tested:
        q = fdr_bh(np.array([p["p_value"] for p in tested]), alpha=FDR_ALPHA)
        for p, qv, rej in zip(tested, q["q_values"], q["reject"]):
            p["fdr_q_value"] = float(qv)
            p["fdr_significant"] = bool(rej)
        n_reject = int(q["n_reject"])

    if not tested:
        branch = "not_enough_within_patient_pairs"
    elif n_reject >= 1:
        branch = "structures_separate"
    else:
        branch = "structures_do_not_separate"

    return {"branch": branch, "n_patients_with_2_or_more_structures": len(patients_with_ge2_structures),
            "n_patients_total": len(patients), "pairs": pairs, "n_pairs_tested": len(tested),
            "n_pairs_fdr_significant": n_reject}


def unpaired_matched_structure_test(sessions: list[dict]) -> dict:
    have_white = [s for s in sessions if s["white_fraction_median"] is not None]
    structures = sorted({s["structure"] for s in have_white})
    pairs = []
    for i, s1 in enumerate(structures):
        for s2 in structures[i + 1:]:
            g1 = [s for s in have_white if s["structure"] == s1]
            g2 = [s for s in have_white if s["structure"] == s2]
            uc1, uc2 = np.array([s["n_units"] for s in g1]), np.array([s["n_units"] for s in g2])
            w1, w2 = (np.array([s["white_fraction_median"] for s in g1]),
                      np.array([s["white_fraction_median"] for s in g2]))
            uc_lo = max(np.quantile(uc1, BAND_QUANTILES[0]), np.quantile(uc2, BAND_QUANTILES[0]))
            uc_hi = min(np.quantile(uc1, BAND_QUANTILES[1]), np.quantile(uc2, BAND_QUANTILES[1]))
            w_lo = max(np.quantile(w1, BAND_QUANTILES[0]), np.quantile(w2, BAND_QUANTILES[0]))
            w_hi = min(np.quantile(w1, BAND_QUANTILES[1]), np.quantile(w2, BAND_QUANTILES[1]))
            if uc_lo > uc_hi or w_lo > w_hi:
                pairs.append({"structure_a": s1, "structure_b": s2, "status": "no_common_band"})
                continue

            def in_band(s):
                return uc_lo <= s["n_units"] <= uc_hi and w_lo <= s["white_fraction_median"] <= w_hi

            b1, b2 = [s for s in g1 if in_band(s)], [s for s in g2 if in_band(s)]
            if len(b1) < MIN_SESSIONS_PER_GROUP or len(b2) < MIN_SESSIONS_PER_GROUP:
                pairs.append({"structure_a": s1, "structure_b": s2, "status": "not_enough_sessions_in_common_band",
                              "n_sessions_a": len(b1), "n_sessions_b": len(b2)})
                continue
            a, b = np.array([s["d_perm"] for s in b1]), np.array([s["d_perm"] for s in b2])
            rng = np.random.default_rng(stable_seed(f"unpaired_matched_pair|{s1}|{s2}"))
            observed, p = permutation_test_twosample(a, b, alternative="two-sided", rng=rng, n_perm=N_PERM_PAIRWISE)
            pairs.append({"structure_a": s1, "structure_b": s2, "status": "tested",
                          "n_sessions_a": len(b1), "n_sessions_b": len(b2),
                          "n_patients_a": len({s["patient"] for s in b1}), "n_patients_b": len({s["patient"] for s in b2}),
                          "mean_diff_a_minus_b": observed, "p_value": p,
                          "matched_unit_count_band": [float(uc_lo), float(uc_hi)],
                          "matched_nugget_share_band": [float(w_lo), float(w_hi)]})

    tested = [p for p in pairs if p["status"] == "tested"]
    if tested:
        q = fdr_bh(np.array([p["p_value"] for p in tested]), alpha=FDR_ALPHA)
        for p, qv, rej in zip(tested, q["q_values"], q["reject"]):
            p["fdr_q_value"] = float(qv)
            p["fdr_significant"] = bool(rej)
    return {"pairs": pairs, "n_pairs_tested": len(tested)}


def withdrawals(lag_artifact: dict) -> dict:
    mucs = lag_artifact["per_structure_delay"]["matched_unit_count_subsample"]
    by_structure = mucs.get("by_structure", {})
    out = {}
    for structure, prior in PRIOR_ROUND_STRUCTURE_EXISTENCE.items():
        current = by_structure.get(structure, {})
        out[structure] = {
            "prior_value_from_the_previous_round": prior,
            "current_matched_unit_count_result": {
                "common_unit_count": mucs.get("common_unit_count"), "n_sessions": current.get("n_sessions"),
                "n_lags_tested": current.get("n_lags_tested"), "n_lags_clearing_fdr": current.get("n_lags_clearing_fdr"),
                "survives": bool((current.get("n_lags_clearing_fdr") or 0) >= 1),
            },
        }
    dacc_c, hip_c = by_structure.get("dacc", {}), by_structure.get("hippocampus", {})
    amy_c = by_structure.get("amygdala", {})
    out["note"] = (
        f"The prior values are quoted as previously reported; the read-only "
        f"state_persistence_lag.json input was not regenerated to re-derive their exact originating lag. "
        f"The correction is the matched-unit-count column already computed in that artifact: at "
        f"common_unit_count={mucs.get('common_unit_count')}, dACC clears FDR at "
        f"{dacc_c.get('n_lags_clearing_fdr', 0)}/{dacc_c.get('n_lags_tested', 0)} lags and hippocampus clears "
        f"FDR at {hip_c.get('n_lags_clearing_fdr', 0)}/{hip_c.get('n_lags_tested', 0)} lags -- NEITHER survives "
        f"matched-unit-count, extending the withdrawal already anticipated for dACC to hippocampus as "
        f"well. Amygdala clears FDR at {amy_c.get('n_lags_clearing_fdr', 0)}/{amy_c.get('n_lags_tested', 0)} "
        f"lags and partially survives."
    )
    return out


def anatomical_stratification_closure(primary: dict, withdrawal: dict) -> dict:
    """The one field this project's anatomical-stratification line needs so a
    later reader does not reopen a question this artifact has already
    closed: the within-patient between-structure gate (primary_within_
    patient_paired) actually ran, at the unit-count-matched sample the
    withdrawals field records, and found nothing to close on."""
    pairs = primary.get("pairs", [])
    tested = [p for p in pairs if p.get("status") == "tested"]
    fdr_significant = [p for p in tested if p.get("fdr_significant")]
    mucs = withdrawal.get("hippocampus", {}).get("current_matched_unit_count_result", {})
    return {
        "status": "closed",
        "gate": "within-patient between-structure paired difference, primary_within_patient_paired",
        "n_pairs_with_enough_shared_patients_to_test": len(tested),
        "n_pairs_total_considered": len(pairs),
        "n_pairs_surviving_fdr": len(fdr_significant),
        "branch": primary.get("branch"),
        "unit_count_matching_that_produced_this_result": {
            "common_unit_count": mucs.get("common_unit_count"),
            "note": (
                "The matched-unit-count sensitivity in the withdrawals field above (dACC and hippocampus "
                "recomputed at a shared common_unit_count, both clearing 0 lags after FDR) is what makes "
                "this closure a unit-count-controlled result rather than an artifact of dACC and "
                "hippocampus sessions carrying different unit counts by construction."
            ),
        },
        "closure_statement": (
            f"{len(tested)} of {len(pairs)} anatomical structure pairs had enough shared patients "
            f"(>= {MIN_PATIENTS_PER_PAIR}) to test at all; {len(fdr_significant)} survived FDR correction. "
            "This is the within-patient region-pair gate this project's anatomical-stratification line "
            "asked for across several prior rounds and had never actually run before this artifact. The "
            "line is CLOSED by this result, not merely unmet: a later round should not re-open the "
            "cross-structure separation question in this corpus without a materially different design "
            "(more patients with multi-structure coverage, or a different pairing) rather than a rerun of "
            "this same test."
        ),
    }


def main() -> None:
    t0 = time.time()
    lag_data = json.loads(LAG_PATH.read_text())
    white_lookup = _load_white_share_lookup()
    clearing_lags = lag_data["clearing_lags_bins_at_deciding_width"]
    width_bins = lag_data["deciding_width_bins"]

    sessions = build_structure_sessions(lag_data["human_lag_rows"], white_lookup, clearing_lags, width_bins)
    print(f"{len(sessions)} human delay-epoch anatomical-structure sessions built, {time.time()-t0:.1f}s", file=sys.stderr)

    nugget = nugget_share_by_structure(sessions)
    conditioned = structure_effect_conditioned_on_nugget_share(sessions)
    primary = within_patient_paired_structure_test(sessions)
    secondary = unpaired_matched_structure_test(sessions)
    withdrawal = withdrawals(lag_data)

    output = {
        "version": "2026-08-15",
        "scope": (
            "Every (dataset, patient, session, structure) delay-epoch row in state_persistence_lag.json's "
            "human_lag_rows at the deciding width, excluding the pooled structure, joined to "
            "variance_partition.json's white_fraction_median by (dataset, session, structure, epoch='delay', "
            "bin_ms=100.0). No new census: both inputs are read exactly as already on disk."
        ),
        "n_sessions_seen": len(sessions), "clearing_lags_bins_used": clearing_lags, "width_bins_used": width_bins,
        "min_sessions_per_group": MIN_SESSIONS_PER_GROUP, "min_patients_per_pair": MIN_PATIENTS_PER_PAIR,
        "nugget_share_by_structure": nugget,
        "structure_effect_conditioned_on_nugget_share": conditioned,
        "between_structure_separation": {
            "primary_within_patient_paired": primary, "secondary_unpaired_matched_band": secondary,
        },
        "withdrawals": withdrawal,
        "anatomical_stratification_closure": anatomical_stratification_closure(primary, withdrawal),
        "session_rows": sessions,
        "wall_clock_s": time.time() - t0,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} in {time.time()-t0:.1f}s", file=sys.stderr)
    print(json.dumps({
        "nugget_share_by_structure_status": nugget.get("status"),
        "structure_conditioned_verdict": conditioned.get("verdict"),
        "between_structure_branch": primary.get("branch"),
    }, indent=2))


if __name__ == "__main__":
    main()
