#!/usr/bin/env python3
"""Correlates per-subject manifold contraction rate (DANDI 000469 single units)
with content-code temporal stability, behavioral accuracy, and correct-versus-
error trial outcome.

Reuses the per-subject load-3 ensemble contraction rate computed by
scripts/run_divergence_analysis.py (all_statistics.json["divergence"]["rutishauser"]),
the content_tau/context_tau per subject from scripts/run_000469_pipeline.py
(all_statistics.json["rutishauser_000469_content_context"]), and per-subject
accuracy (all_statistics.json["dandi000469_ctg"]). Only the correct-versus-error
split of ensemble DMD is fit here, from results/dandi000469_geometry_*.npz.

Outputs: results/contraction_behavior_analysis_000469.json
Updates: results/all_statistics.json — "contraction_behavior_analysis_000469" key

Run (after run_divergence_analysis.py and run_000469_pipeline.py):
    conda run -n wm_dynamics python scripts/run_contraction_behavior_analysis_000469.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dynamics import ensemble_dmd, divergence_rank_sweep, rank_robustness_sign
from statistics import spearman_permutation_test

RESULTS = ROOT / "results"
DMD_RANK = 8
RANK_SWEEP = (5, 6, 7, 8)   # truncation ranks below full rank (d=8); audit item 9
DT = 0.1   # 100 ms bins, matches run_000469_pipeline.py BIN_MS and run_divergence_analysis.py
MIN_TRIALS_PER_GROUP = 8   # matches PR_MIN_TRIALS_PER_GROUP elsewhere in this project


def correct_error_contraction(subj: str, rng: np.random.Generator) -> dict | None:
    path = RESULTS / f"dandi000469_geometry_{subj}.npz"
    if not path.exists():
        return None
    geo = np.load(path, allow_pickle=True)
    Z, resp = geo["Z"], geo["response_accuracy"].astype(bool)
    if resp.sum() < MIN_TRIALS_PER_GROUP or (~resp).sum() < MIN_TRIALS_PER_GROUP:
        return None
    ens_correct = ensemble_dmd(Z[resp], r=DMD_RANK, dt=DT,
                               n_splits=min(5, int(resp.sum())), n_null=30, rng=rng)
    ens_error = ensemble_dmd(Z[~resp], r=DMD_RANK, dt=DT,
                             n_splits=min(5, int((~resp).sum())), n_null=30, rng=rng)

    # Rank robustness: is sign(div_error - div_correct) stable away from
    # full-rank (r=d) DMD, where Σlog|λ|/dt is log-det sensitive to noise?
    sweep_correct = divergence_rank_sweep(Z[resp], DT, ranks=RANK_SWEEP,
                                          n_splits=min(5, int(resp.sum())), n_null=5, rng=rng)
    sweep_error   = divergence_rank_sweep(Z[~resp], DT, ranks=RANK_SWEEP,
                                          n_splits=min(5, int((~resp).sum())), n_null=5, rng=rng)
    contrast = np.array(sweep_error["div_scalar"]) - np.array(sweep_correct["div_scalar"])
    sign_robust = rank_robustness_sign(contrast)

    return {
        "div_correct": ens_correct["div_scalar"], "div_error": ens_error["div_scalar"],
        "r2_cv_correct": ens_correct["r2_cv"], "r2_cv_error": ens_error["r2_cv"],
        "n_correct": int(resp.sum()), "n_error": int((~resp).sum()),
        "rank_sweep_ranks": sweep_correct["ranks"],
        "rank_sweep_div_correct": sweep_correct["div_scalar"],
        "rank_sweep_div_error": sweep_error["div_scalar"],
        "rank_sweep_sign_robust": sign_robust,
    }


def main():
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)

    divergence_rut = stats.get("divergence", {}).get("rutishauser", {})
    ctg = stats.get("dandi000469_ctg", {})
    content_context = stats.get("rutishauser_000469_content_context", {}).get("per_subject", {})

    if not divergence_rut or not ctg:
        print("SKIP — run run_divergence_analysis.py and run_000469_pipeline.py first "
              "(missing 'divergence'/'rutishauser' or 'dandi000469_ctg' in all_statistics.json).")
        return

    rng = np.random.default_rng(0)
    per_subject = {}
    for subj, ctg_v in ctg.items():
        if subj not in divergence_rut:
            continue
        contraction = divergence_rut[subj]["ensemble_div_scalar"]
        content_v = ctg_v.get("content_ctg")
        row = {
            "contraction_rate": contraction,
            "accuracy": ctg_v["accuracy"],
            "content_offdiag_effect": content_v["offdiag_effect"] if content_v else None,
            "content_tau": content_context.get(subj, {}).get("content_tau"),
        }
        ce = correct_error_contraction(subj, rng)
        if ce is not None:
            row.update(ce)
        per_subject[subj] = row

    print(f"  {len(per_subject)} sessions with a contraction rate available")

    def _paired_arrays(field: str) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for v in per_subject.values():
            if v.get(field) is not None and np.isfinite(v.get(field, np.nan)):
                xs.append(v["contraction_rate"]); ys.append(v[field])
        return np.array(xs), np.array(ys)

    tests = {}

    x, y = _paired_arrays("content_tau")
    if len(x) >= 4:
        tests["contraction_vs_content_tau"] = spearman_permutation_test(x, y, n_perm=10000, rng=rng)
        print(f"  contraction vs content_tau (N={len(x)}): "
              f"rho={tests['contraction_vs_content_tau']['rho']:.3f}, "
              f"p={tests['contraction_vs_content_tau']['p_value']:.4f}")
    else:
        tests["contraction_vs_content_tau"] = {"note": f"only {len(x)} sessions with both values"}

    x, y = _paired_arrays("content_offdiag_effect")
    if len(x) >= 4:
        tests["contraction_vs_content_offdiag"] = spearman_permutation_test(x, y, n_perm=10000, rng=rng)
        print(f"  contraction vs content off-diag effect (N={len(x)}): "
              f"rho={tests['contraction_vs_content_offdiag']['rho']:.3f}, "
              f"p={tests['contraction_vs_content_offdiag']['p_value']:.4f}")
    else:
        tests["contraction_vs_content_offdiag"] = {"note": f"only {len(x)} sessions with both values"}

    x, y = _paired_arrays("accuracy")
    tests["contraction_vs_accuracy"] = spearman_permutation_test(x, y, n_perm=10000, rng=rng)
    print(f"  contraction vs accuracy (N={len(x)}): "
          f"rho={tests['contraction_vs_accuracy']['rho']:.3f}, "
          f"p={tests['contraction_vs_accuracy']['p_value']:.4f}")

    ce_pairs = [(v["div_correct"], v["div_error"]) for v in per_subject.values()
                if "div_correct" in v]
    if len(ce_pairs) >= 4:
        from statistics import paired_sign_flip_test
        div_c = np.array([p[0] for p in ce_pairs])
        div_e = np.array([p[1] for p in ce_pairs])
        # H1: contraction is weaker (less negative / more positive) on error trials
        res = paired_sign_flip_test(div_e, div_c, n_perm=10000, alternative="greater", rng=rng)
        tests["correct_vs_error_contraction"] = res
        print(f"  correct-vs-error contraction (N={len(ce_pairs)}): "
              f"mean(div_error - div_correct)={res['mean_diff']:.4f}, p={res['p_value']:.4f} "
              f"(positive = contraction weaker on error trials)")

        robust_flags = [v["rank_sweep_sign_robust"] for v in per_subject.values()
                        if "rank_sweep_sign_robust" in v]
        n_robust = int(np.sum(robust_flags))
        print(f"  rank robustness (r={RANK_SWEEP}) of correct-vs-error contraction: "
              f"{n_robust}/{len(robust_flags)} sessions sign-stable")
        tests["correct_vs_error_contraction_rank_robustness"] = {
            "ranks_swept": list(RANK_SWEEP), "n_robust": n_robust, "n_total": len(robust_flags),
            "all_robust": bool(n_robust == len(robust_flags)),
        }
    else:
        tests["correct_vs_error_contraction"] = {"note": f"only {len(ce_pairs)} sessions with both trial types"}

    tests_serializable = {k: {kk: vv for kk, vv in v.items() if kk != "null"}
                         for k, v in tests.items()}
    out = {"per_subject": per_subject, "tests": tests_serializable}
    with open(RESULTS / "contraction_behavior_analysis_000469.json", "w") as f:
        json.dump(out, f, indent=2)

    stats["contraction_behavior_analysis_000469"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/contraction_behavior_analysis_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
