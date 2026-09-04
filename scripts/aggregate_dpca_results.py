#!/usr/bin/env python3
"""Aggregates the per-subject condition-dependent versus condition-independent
cross-temporal generalization results from run_dpca_analysis.py into paired
statistical tests across subjects, without recomputing the underlying
cross-temporal generalization matrices.

Updates: results/all_statistics.json — "dpca_paired_tests" key

Run (after run_dpca_analysis.py):
    conda run -n wm_dynamics python scripts/aggregate_dpca_results.py
"""
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from statistics import paired_sign_flip_test
from provenance import _json_safe

RESULTS = ROOT / "results"


def paired_test(ci_vals: list[float], cd_vals: list[float], rng) -> dict:
    ci_arr, cd_arr = np.array(ci_vals), np.array(cd_vals)
    res = paired_sign_flip_test(cd_arr, ci_arr, n_perm=10000, alternative="greater", rng=rng)
    return {k: v for k, v in res.items() if k != "null"}


def main():
    rng = np.random.default_rng(0)

    with open(RESULTS / "dpca_miller.json") as f:
        miller = json.load(f)
    with open(RESULTS / "dpca_dandi000469.json") as f:
        d469 = json.load(f)
    with open(RESULTS / "dpca_pfc3.json") as f:
        pfc3 = json.load(f)

    out = {}

    ci_vals = [v["ci_ctg"]["offdiag_effect"] for v in miller.values()]
    cd_vals = [v["cd_ctg"]["offdiag_effect"] for v in miller.values()]
    out["miller_context"] = paired_test(ci_vals, cd_vals, rng)
    out["miller_context"]["n_subjects"] = len(ci_vals)
    print(f"Miller context (N={len(ci_vals)}): mean(CD-CI)={out['miller_context']['mean_diff']:.4f} "
          f"[{out['miller_context']['ci_lower']:.4f}, {out['miller_context']['ci_upper']:.4f}], "
          f"p={out['miller_context']['p_value']:.4f}")

    for marginalization in ["context", "content"]:
        ci_vals = [v[marginalization]["ci_ctg"]["offdiag_effect"] for v in d469.values()
                  if marginalization in v]
        cd_vals = [v[marginalization]["cd_ctg"]["offdiag_effect"] for v in d469.values()
                  if marginalization in v]
        key = f"dandi000469_{marginalization}"
        out[key] = paired_test(ci_vals, cd_vals, rng)
        out[key]["n_subjects"] = len(ci_vals)
        print(f"DANDI 000469 {marginalization} (N={len(ci_vals)}): "
              f"mean(CD-CI)={out[key]['mean_diff']:.4f} "
              f"[{out[key]['ci_lower']:.4f}, {out[key]['ci_upper']:.4f}], p={out[key]['p_value']:.4f}")

    out["pfc3_content"] = {
        "ci_offdiag_effect": pfc3["ci_ctg"]["offdiag_effect"],
        "cd_offdiag_effect": pfc3["cd_ctg"]["offdiag_effect"],
        "n_subjects": 1,
    }
    print(f"pfc-3 content (single pseudo-population): "
          f"CI={pfc3['ci_ctg']['offdiag_effect']:.4f}, CD={pfc3['cd_ctg']['offdiag_effect']:.4f}")

    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    stats["dpca_paired_tests"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nUpdated all_statistics.json[dpca_paired_tests]")


if __name__ == "__main__":
    main()
