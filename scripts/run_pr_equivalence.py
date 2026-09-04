#!/usr/bin/env python3
"""Equivalence + Bayesian null for the PR-vs-load slope.

A non-significant slope (p=0.87) is absence of evidence, not evidence of
absence. This upgrades the load-null to an accept-the-null by testing that the
pooled PR-vs-load slope is statistically EQUIVALENT to zero within a
pre-registered smallest effect of interest (SESOI), and by a Bayes factor
favouring the null over a load-dependent alternative.

SESOI (pre-stated BEFORE computing, NOT tuned to the answer): 0.5 effective
dimensions per load level — half of one effective dimension. A load-driven
dimensionality expansion smaller than half an effective dimension per step is
below what the literature reporting load-dependent geometry expansion
(Bernardi et al. 2020; Wasmuht et al. 2018) would treat as a meaningful change
in the number of coding dimensions, and is far below the several-effective-
dimension between-subject PR spread these datasets show.

Reuses the EXACT per-dataset PR-slope machinery the abstract number comes from
(pr_lme_by_dataset -> forest_meta, the same pooling as aggregate_forest_syntheses)
— no re-estimation. TOST and the Bayes factor (statistics.tost_equivalence,
bf_null_slope) run on the pooled slope + SE; per-dataset TOST is also reported.

results/pr_equivalence.json — {sesoi, pooled_slope, pooled_se, tost_p_lower,
    tost_p_upper, tost_reject_null_of_effect, bayes_factor_01, per_dataset:[...]}.

Run:
    conda run -n wm_dynamics python scripts/run_pr_equivalence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from statistics import forest_meta, tost_equivalence, bf_null_slope  # noqa: E402
from provenance import _json_safe

RESULTS = ROOT / "results"
SESOI = 0.5   # effective dimensions per load level (pre-stated; see module docstring)

PR_DATASETS = [("Miller", "miller"), ("Boran iEEG", "boran_ieeg"),
               ("Boran units", "boran_units"), ("DANDI 000469", "dandi000469"),
               ("DANDI 001187", "dandi001187"), ("DANDI 000673", "dandi000673")]


def _se_from_beta_p(beta: float, p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-9)
    z = norm.isf(p / 2.0)
    return abs(beta) / z if z > 1e-9 else float("inf")


def main():
    stats = json.load(open(RESULTS / "all_statistics.json"))
    by_dataset = stats.get("pr_lme_by_dataset", {})

    rows, per_dataset = [], []
    for label, key in PR_DATASETS:
        r = by_dataset.get(key)
        if not (r and "beta" in r):
            continue
        beta, se = float(r["beta"]), _se_from_beta_p(r["beta"], r["p_value"])
        rows.append((label, beta, se))
        td = tost_equivalence(beta, se, SESOI)
        per_dataset.append({"dataset": label, "slope": beta, "se": se,
                            "tost_p": td["p"], "tost_reject": td["reject"]})

    labels = [x[0] for x in rows]
    est = np.array([x[1] for x in rows])
    ses = np.array([x[2] for x in rows])
    meta = forest_meta(est, ses, labels=labels)
    pooled, se_pooled = meta["pooled"], meta["se"]

    tost = tost_equivalence(pooled, se_pooled, SESOI)
    # Bayes factor: Cauchy prior scaled to the SESOI (the pre-stated plausible
    # effect magnitude under H1), so the evidential measure and the equivalence
    # bound use the same effect scale.
    bf = bf_null_slope(pooled, se_pooled, r_scale=SESOI)

    out = {
        "sesoi": SESOI,
        "sesoi_units": "effective dimensions per load level",
        "pooled_slope": pooled,
        "pooled_se": se_pooled,
        "pooled_ci": [meta["ci_lo"], meta["ci_hi"]],
        "pooled_p": meta["p_value"],
        "tost_p_lower": tost["p_lower"],
        "tost_p_upper": tost["p_upper"],
        "tost_p": tost["p"],
        "tost_reject_null_of_effect": tost["reject"],
        "bayes_factor_01": bf["bf_01"],
        "bayes_prior_scale": bf["r_scale"],
        "per_dataset": per_dataset,
    }
    json.dump(_json_safe(out), open(RESULTS / "pr_equivalence.json", "w"), indent=2, allow_nan=False)
    stats["pr_equivalence"] = out
    json.dump(_json_safe(stats), open(RESULTS / "all_statistics.json", "w"), indent=2, allow_nan=False)

    print(f"Pooled PR-vs-load slope = {pooled:+.4f} (SE {se_pooled:.4f}, "
          f"95% CI [{meta['ci_lo']:+.3f}, {meta['ci_hi']:+.3f}], p={meta['p_value']:.3f})")
    print(f"SESOI = ±{SESOI} effective dims/load level")
    print(f"TOST: p_lower={tost['p_lower']:.4g}, p_upper={tost['p_upper']:.4g}, "
          f"binding p={tost['p']:.4g} -> "
          f"{'EQUIVALENT to zero (reject H1)' if tost['reject'] else 'NOT equivalent — SOFTEN THE CLAIM (STOP AND REPORT)'}")
    print(f"Bayes factor BF01 = {bf['bf_01']:.2f} "
          f"({'favours the null' if bf['bf_01'] > 1 else 'favours an effect'})")
    print("Per-dataset TOST:")
    for pd in per_dataset:
        print(f"  {pd['dataset']:14s} slope={pd['slope']:+.4f} tost_p={pd['tost_p']:.4g} "
              f"reject={pd['tost_reject']}")
    print("\nwrote results/pr_equivalence.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
