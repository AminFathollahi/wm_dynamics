"""
aggregate_forest_syntheses.py — Phase-0 cross-dataset meta-analytic syntheses.

Reads the authoritative per-dataset artifacts in results/all_statistics.json and
pools them with statistics.forest_meta into the forest-plot estimates that each
constraint section of the reframed paper leads with. Datasets are ROWS, one
pooled estimate per constraint. Writes results/forest_syntheses.json.

SE provenance (no fabrication — every SE is derived from stored quantities):
  - subject-spread : estimate = mean over subjects/sessions, se = std/sqrt(n),
                     used where per-subject values exist (CTG off-diagonal effect,
                     flow divergence).
  - z-from-p       : se = |beta| / z, z = Phi^{-1}(1 - p/2), used where only a
                     permutation beta + two-sided p is stored (PR-vs-load slope,
                     correct-vs-error drift). p is floored at 1e-6.
Every synthesis records which key and which method produced each row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from statistics import forest_meta  # noqa: E402

STATS_PATH = ROOT / "results" / "all_statistics.json"
OUT_PATH = ROOT / "results" / "forest_syntheses.json"


def se_from_beta_p(beta: float, p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-9)
    z = norm.isf(p / 2.0)
    return abs(beta) / z if z > 1e-9 else float("inf")


def subject_spread(values: list[float]) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 2:
        return (float(v[0]) if n == 1 else float("nan"), float("nan"), n)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n)), n


def _meta(rows, method="random"):
    """rows: list of (label, estimate, se). Drops non-finite; returns None if <1 usable."""
    labels = [r[0] for r in rows]
    est = np.array([r[1] for r in rows], dtype=float)
    se = np.array([r[2] for r in rows], dtype=float)
    if not (np.isfinite(est) & np.isfinite(se) & (se > 0)).any():
        return None
    res = forest_meta(est, se, labels=labels, method=method)
    return res


def build(d: dict) -> dict:
    out = {}

    # ── Q1: sustained context/load CTG off-diagonal effect (context = stable target)
    ctg_rows = []
    if "ctg" in d:
        vals = [s["mean_offdiag"] - 0.5 for s in d["ctg"].values() if "mean_offdiag" in s]
        m, se, n = subject_spread(vals)
        ctg_rows.append(("Miller ECoG", m, se))
    if "boran_ctg" in d:
        vals = [s["offdiag_effect"] for s in d["boran_ctg"].values()
                if isinstance(s, dict) and "offdiag_effect" in s]
        m, se, n = subject_spread(vals)
        ctg_rows.append(("Boran iEEG", m, se))
    out["q1_context_ctg_offdiag"] = {
        "description": "Context/load CTG off-diagonal effect (AUC-0.5), pooled across datasets",
        "se_method": "subject-spread",
        "source_keys": ["ctg", "boran_ctg"],
        "note": "DANDI CTG artifacts store PR/accuracy, not an off-diagonal effect field; "
                "extend this forest in the full run once that field is materialised.",
        "meta": _meta(ctg_rows),
    }

    # ── Q1: item-identity (content) CTG off-diagonal effect (content = moving target)
    content_rows = []
    content_points = []  # datasets with an effect but no across-subject SE (not poolable)
    if "pfc3_content_ctg" in d and "mean_offdiag_auc_minus_chance" in d["pfc3_content_ctg"]:
        # single pseudo-population -> no across-subject spread. Carry it as a labelled point
        # (NOT via forest_meta, which would silently drop the NaN-SE row and hide the only
        # cross-species content replication we have).
        content_points.append({
            "label": "CRCNS pfc-3 (macaque)",
            "estimate": d["pfc3_content_ctg"]["mean_offdiag_auc_minus_chance"],
            "tau": d["pfc3_content_ctg"].get("tau"),
            "p_value": d["pfc3_content_ctg"].get("p_value"),
            "note": "single 80-neuron pseudo-population; no subject-level SE to weight",
        })
    rcc = d.get("rutishauser_000469_content_context", {})
    if "per_subject" in rcc:
        vals = [s["content_offdiag_effect"] for s in rcc["per_subject"].values()
                if "content_offdiag_effect" in s]
        m, se, n = subject_spread(vals)
        content_rows.append(("DANDI 000469 (human)", m, se))
    out["q1_content_ctg_offdiag"] = {
        "description": "Item-identity (content) CTG off-diagonal effect (AUC-0.5)",
        "se_method": "subject-spread",
        "source_keys": ["pfc3_content_ctg", "rutishauser_000469_content_context"],
        "point_estimates": content_points,
        "meta": _meta(content_rows),
    }

    # ── Q1: within-subject content/context dissociation (carried through, not re-pooled)
    out["q1_within_subject_dissociation_000469"] = {
        "description": "Paired within-subject context-vs-content stability, DANDI 000469",
        "wilcoxon_p": rcc.get("wilcoxon_p"),
        "paired_gap_mean": rcc.get("paired_gap_mean"),
        "paired_gap_ci": rcc.get("paired_gap_ci"),
        "paired_gap_permutation_p": rcc.get("paired_gap_permutation_p"),
        "n_paired_subjects": rcc.get("n_paired_subjects"),
        "axis_rotation_within_subject": d.get("axis_rotation_within_subject_dandi000469"),
    }

    # ── Q1: PR-vs-load slope (readout is low-dimensional and load-invariant)
    pr_rows = []
    for label, key in [("Miller", "miller"), ("Boran iEEG", "boran_ieeg"),
                       ("Boran units", "boran_units"), ("DANDI 000469", "dandi000469"),
                       ("DANDI 001187", "dandi001187"), ("DANDI 000673", "dandi000673")]:
        r = d.get("pr_lme_by_dataset", {}).get(key)
        if r and "beta" in r:
            pr_rows.append((label, r["beta"], se_from_beta_p(r["beta"], r["p_value"])))
    out["q1_pr_vs_load_slope"] = {
        "description": "PR-vs-load LME slope pooled across datasets (expected pooled ~ 0)",
        "se_method": "z-from-p",
        "source_keys": ["pr_lme_by_dataset"],
        "meta": _meta(pr_rows),
    }

    # ── Q3: flow divergence (mean-trajectory and single-trial ensemble)
    # Round-7 STEP K2: pool the three fillable-gap cohorts (Boran units, DANDI
    # 001187, DANDI 000673) into the SAME forest as their siblings — moves this
    # from ~3 to ~6 datasets. All six are fit at the SAME DMD_RANK=8 (module's
    # full-latent-rank convention; see run_divergence_analysis.py STEP K2 note),
    # so they are on comparable footing.
    for field, name in [("div_scalar", "q3_divergence_mean_traj"),
                        ("ensemble_div_scalar", "q3_divergence_ensemble")]:
        div_rows = []
        for label, key in [("Miller ECoG", "miller"), ("Boran iEEG", "boran"),
                           ("Single units (DANDI 000469)", "rutishauser"),
                           ("Boran units", "boran_units"),
                           ("DANDI 001187", "dandi001187"),
                           ("DANDI 000673", "dandi000673")]:
            grp = d.get("divergence", {}).get(key, {})
            vals = [s[field] for s in grp.values() if isinstance(s, dict) and field in s
                    and np.isfinite(s.get(field, np.nan))]
            if vals:
                m, se, n = subject_spread(vals)
                div_rows.append((label, m, se))
        out[name] = {
            "description": f"Flow divergence ({field}) pooled across datasets [1/s]",
            "se_method": "subject-spread",
            "source_keys": ["divergence"],
            "meta": _meta(div_rows),
        }

    # ── Q2: plant identifiability — cross-validated ensemble-DMD R^2 margin over its
    #    circular-shift null (positive = the linear operator predicts held-out transitions
    #    above chance). Pooled across datasets; SE from subject spread of (cv - null).
    q2_rows = []
    for label, key in [("Miller ECoG", "miller"), ("Boran iEEG", "boran"),
                       ("Single units (DANDI 000469)", "rutishauser"),
                       ("Boran units", "boran_units"),
                       ("DANDI 001187", "dandi001187"),
                       ("DANDI 000673", "dandi000673")]:
        grp = d.get("divergence", {}).get(key, {})
        margins = [s["ensemble_r2_cv"] - s["ensemble_r2_null"]
                   for s in grp.values() if isinstance(s, dict)
                   and s.get("ensemble_r2_cv") is not None and s.get("ensemble_r2_null") is not None
                   and np.isfinite(s["ensemble_r2_cv"]) and np.isfinite(s["ensemble_r2_null"])]
        if margins:
            m, se, n = subject_spread(margins)
            q2_rows.append((label, m, se))
    out["q2_plant_identifiability"] = {
        "description": "Ensemble-DMD cross-validated R^2 margin over circular-shift null "
                       "(plant identifiable above chance)",
        "se_method": "subject-spread",
        "source_keys": ["divergence"],
        "note": "iEEG/ECoG margins are near zero because smooth mean dynamics let even the "
                "shift-null fit well; single units carry the genuine identifiability margin.",
        "meta": _meta(q2_rows),
    }

    # ── Q4: site — dynamic-vs-static electrode-selection alignment gain above 1.0 (only the
    #    field-potential datasets carry a spatial LFP to align; single units are excluded by
    #    design, not omitted silently).
    q4_rows = []
    for label, key in [("Miller ECoG", "miller"), ("Boran iEEG", "boran")]:
        grp = d.get("divergence", {}).get(key, {})
        gains = [s["align_gain_x"] - 1.0 for s in grp.values() if isinstance(s, dict)
                 and s.get("align_gain_x") is not None and np.isfinite(s["align_gain_x"])]
        if gains:
            m, se, n = subject_spread(gains)
            q4_rows.append((label, m, se))
    out["q4_dynamic_alignment_gain"] = {
        "description": "Dynamic (v*-aligned) minus static electrode-selection alignment gain, "
                       "excess over 1.0x (positive = dynamics-informed targeting helps)",
        "se_method": "subject-spread",
        "source_keys": ["divergence"],
        "note": "Single-unit datasets have no spatially-extended LFP to project a stimulation "
                "field onto and are excluded from this forest by design.",
        "meta": _meta(q4_rows),
    }

    # ── Q5: actuation — NOT a cross-dataset forest. LQR/TES1 control is a feasibility
    #    demonstration on the two iEEG datasets (Miller+Boran); B-alignment and rescue are
    #    within-subject contrasts, not an effect measured across all six cohorts. Carry the
    #    key numbers through rather than fabricate a pooled effect.
    lqr = d.get("tes1_lqr", {})
    rescue = d.get("manifold_rescue", {})
    out["q5_control_feasibility"] = {
        "description": "Actuation is a control-design demonstration on Miller+Boran, not a "
                       "poolable cross-dataset effect (documented, not forced into a forest).",
        "not_a_forest": True,
        "source_keys": ["tes1_lqr", "manifold_rescue"],
        "b_norm_iqr_ratio": lqr.get("global_B_norm_iqr_ratio"),
        "b_norm_median": lqr.get("global_B_norm_median"),
        "b_norm_mad": lqr.get("global_B_norm_mad"),
        "mean_fold_range": lqr.get("mean_fold_range"),
        "max_fold_range": lqr.get("max_fold_range"),
        "manifold_rescue_summary": {
            "n_subjects": rescue.get("n_subjects"),
            "mean_controlled_reduction_pct": rescue.get("mean_controlled_reduction_pct"),
            "mean_passive_reduction_pct": rescue.get("mean_passive_reduction_pct"),
            "controlled_vs_passive_p": (rescue.get("controlled_greater_than_passive") or {}).get("p_value"),
            "controlled_vs_passive_ci": [
                (rescue.get("controlled_greater_than_passive") or {}).get("ci_lower"),
                (rescue.get("controlled_greater_than_passive") or {}).get("ci_upper"),
            ],
        } if isinstance(rescue, dict) else {},
    }

    # ── Q6: correct-vs-error drift beta across datasets (positive = correct drifts more)
    drift_rows = []
    drift_sources = [
        ("Boran iEEG", "boran_correct_error_drift"),
        ("Boran units", "dandi000574_units_correct_error_drift"),
        ("DANDI 000469", "dandi000469_correct_error_drift"),
        ("DANDI 001187", "dandi001187_correct_error_drift"),
        ("DANDI 000673", "dandi000673_correct_error_drift"),
    ]
    for label, key in drift_sources:
        r = d.get(key)
        if r and "beta" in r:
            drift_rows.append((label, r["beta"], se_from_beta_p(r["beta"], r["p_value"])))
    out["q6_correct_error_drift"] = {
        "description": "Correct-vs-error geometric drift beta pooled across datasets",
        "se_method": "z-from-p",
        "source_keys": [k for _, k in drift_sources],
        "meta": _meta(drift_rows),
    }

    return out


def main() -> None:
    d = json.load(open(STATS_PATH))
    syntheses = build(d)
    json.dump(syntheses, open(OUT_PATH, "w"), indent=2)

    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    for name, s in syntheses.items():
        meta = s.get("meta")
        if meta:
            print(f"  {name:38s} k={meta['k']}  pooled={meta['pooled']:+.4f} "
                  f"[{meta['ci_lo']:+.4f},{meta['ci_hi']:+.4f}]  p={meta['p_value']:.3g}  "
                  f"I2={meta['i_squared']:.0f}%")
        elif "meta" in s:
            print(f"  {name:38s} (no poolable rows)")
        else:
            print(f"  {name:38s} (carried through)")


if __name__ == "__main__":
    main()
