#!/usr/bin/env python3
"""Per-dataset dimensionality addendum -- extends, never replaces, the primary
k=8 / r=7-or-8 choices already justified elsewhere.

Two distinct questions, kept explicitly separate throughout: k = the LATENT
dimension (how many principal components define the state space) and r = the
OPERATOR rank (how many DMD eigenvalues of the fitted A are retained). A
single script covers all four addendum questions since they share the same
per-cohort session loaders:

  (B) FEASIBILITY CEILING: the mean-trajectory DMD fit requires k <= T-2 (T =
      time samples per epoch). Each dataset's own parallel-analysis/cv-PR
      dimensionality estimate (already computed in latent_dim_selection.json)
      is compared against this ceiling -- a cv-PR estimate that EXCEEDS the
      ceiling is not fittable at all, which is itself a finding: the
      geometry can be higher-dimensional than the identifiable dynamics.
  (C) Extends run_dim_robustness.py's k in {6,8,10,12} headline-robustness
      sweep to include DANDI 000469's own parallel-analysis k (14, comfortably
      under its ceiling), for the content-vs-context axis-rotation headline.
  (D) Per-cohort held-out one-step cross-validated R^2 across a rank sweep
      (ensemble estimator, reusing dynamics.ensemble_dmd and the same
      per-session iterators as run_vstar_eigen_audit.ALL_ITERS), so every
      cohort has the curve macaque PFC microstimulation already has in dmd_rank_selection.json --
      checked for a genuine elbow rather than assumed monotonic.
  (E) The vector-vs-subspace stability-under-rank-sweep comparison already
      computed for macaque PFC microstimulation (dmd_rank_selection.json's vstar_cos_to_r6 vs
      vstar_subspace_stability.json's per-m rank affinity, both after this
      round's B2 canonicalization fix) is cited directly -- no recomputation
      needed. This comparison is inherently macaque PFC microstimulation-specific in this project:
      it requires a causal stimulation direction to score subspace alignment
      against, which only the macaque PFC microstimulation cohort has.

Output: results/dim_addendum.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_dim_addendum.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics import ensemble_dmd  # noqa: E402
from statistics import stable_seed  # noqa: E402

import run_vstar_eigen_audit as vea  # noqa: E402  (ALL_ITERS -- per-session Z_trials/dt/r)
import run_dim_robustness as ddr  # noqa: E402  (Sternberg loaders + headline-1 axis rotation)

RESULTS = ROOT / "results"
RANK_SWEEP = (3, 4, 5, 6, 7, 8)


def feasibility_ceiling() -> dict:
    """k <= T-2 per cohort (mean-trajectory fit), vs. each Sternberg cohort's
    own cv-PR/parallel-analysis estimate (already on disk, not recomputed)."""
    sternberg_T = int(round(ddr.MAINT_WIN * 1000 / ddr.BIN_MS))  # 23 bins at BIN_MS=100, MAINT_WIN=2.3s
    ceiling = sternberg_T - 2
    dim_sel = json.load(open(RESULTS / "latent_dim_selection.json"))
    out = {}
    for dataset, row in dim_sel.items():
        cv_pr = row["cv_PR_mean"]
        k_pa = row["k_parallel_analysis_median"]
        out[dataset] = {
            "T_bins": sternberg_T, "ceiling_k_le_T_minus_2": ceiling,
            "cv_PR_mean": cv_pr, "cv_PR_exceeds_ceiling": bool(cv_pr > ceiling),
            "k_parallel_analysis_median": k_pa, "k_PA_exceeds_ceiling": bool(k_pa > ceiling),
        }
    # macaque PFC microstimulation: N_BINS=30 (50ms bins) -- fixed constant, no per-dataset cv-PR/k_PA
    # table exists for it (Table 5 covers the three Sternberg cohorts only).
    out["macaque_pfc_microstimulation"] = {"T_bins": 30, "ceiling_k_le_T_minus_2": 28,
                      "note": "operator rank already swept at r in {4..8} (results/dmd_rank_selection.json), "
                              "comfortably under the ceiling; no separate cv-PR/k_PA table exists for this cohort."}
    return out


def extended_axis_rotation_headline() -> dict:
    """DANDI 000469's own parallel-analysis k (14, median from Table 5),
    added to the existing k in {6,8,10,12} axis-rotation sweep."""
    sessions_469 = list(ddr.load_sternberg_sessions("dandi000469"))
    k_pa = 14
    h1 = ddr._headline_1_axis_rotation(sessions_469, k_pa)
    return {"k": k_pa, "source": "dataset's own median parallel-analysis k (Table 5)",
            "n_sessions": len(sessions_469), **h1}


def cv_r2_curve_all_cohorts() -> dict:
    """Held-out one-step R^2 (ensemble estimator) across a rank sweep, per
    cohort, pooled over sessions -- the curve macaque PFC microstimulation already has, extended
    to every cohort. Checked for a genuine elbow (a rank whose R^2 is a local
    max, not the largest rank tested), not assumed to have one."""
    per_cohort_sessions: dict[str, list] = {}
    for it in vea.ALL_ITERS:
        for dataset, session, Z_trials, dt, r in it():
            per_cohort_sessions.setdefault(dataset, []).append((session, Z_trials, dt))

    out = {}
    for dataset, sessions in per_cohort_sessions.items():
        curve = {r: [] for r in RANK_SWEEP}
        for session, Z_trials, dt in sessions:
            d = Z_trials.shape[2]
            for r in RANK_SWEEP:
                r_use = min(r, d, Z_trials.shape[0] * (Z_trials.shape[1] - 1) - 1)
                if r_use < 1:
                    continue
                try:
                    ens = ensemble_dmd(Z_trials, r=r_use, dt=dt, n_splits=5, n_null=5,
                                       rng=np.random.default_rng(stable_seed(f"dimaddendum_{dataset}_{session}_{r}")))
                except np.linalg.LinAlgError:
                    continue
                curve[r].append(ens["r2_cv"])
        mean_curve = {str(r): float(np.mean(v)) if v else None for r, v in curve.items()}
        valid = {int(r): v for r, v in mean_curve.items() if v is not None}
        if len(valid) >= 3:
            ranks_sorted = sorted(valid)
            vals = [valid[r] for r in ranks_sorted]
            best_idx = int(np.argmax(vals))
            has_elbow = 0 < best_idx < len(vals) - 1
        else:
            has_elbow = None
        out[dataset] = {"n_sessions": len(sessions), "r2_cv_by_rank": mean_curve,
                        "has_elbow": has_elbow,
                        "conclusion": ("a rank is data-selected by a genuine interior maximum"
                                      if has_elbow else
                                      "monotonic (or too few valid ranks) -- rank is a reported "
                                      "choice, not a data-selected optimum")}
        print(f"  {dataset:14s} (N={len(sessions):3d} sessions): " +
              "  ".join(f"r={r}:{mean_curve[str(r)]:.3f}" if mean_curve[str(r)] is not None else f"r={r}:NA"
                       for r in RANK_SWEEP) +
              f"  -> {'ELBOW' if has_elbow else 'no elbow'}")
    return out


def vector_vs_subspace_rank_sensitivity_macaque_pfc_microstimulation() -> dict:
    """Cited directly from existing artifacts -- no new computation. macaque PFC microstimulation
    is the one cohort with a causal stimulation direction to score subspace
    alignment against."""
    rank_sel = json.load(open(RESULTS / "dmd_rank_selection.json"))
    subspace = json.load(open(RESULTS / "vstar_subspace_stability.json"))
    vector_cos_by_r = {r: rank_sel[r]["vstar_cos_to_r6"] for r in rank_sel if r != "_meta"}
    subspace_summary = subspace["_meta"]["cross_session_summary"]
    return {
        "vector_cos_to_r6_by_rank": vector_cos_by_r,
        "subspace_mean_rank_affinity_by_m": {m: subspace_summary[m]["mean_rank_affinity_min"] for m in ("1", "2", "3")},
        "conclusion": ("the single vector's (m=1) rank-affinity is markedly less stable across the operator-rank "
                      "sweep than the m=2/m=3 subspaces (mean min-rank-affinity 0.674 for m=1 vs. 0.999/1.000 for "
                      "m=2/m=3): raising the accessible rank degrades the vector far more than the subspace. "
                      "Scope: this comparison requires a causal stimulation direction, which only the macaque PFC microstimulation "
                      "cohort has in this project."),
    }


def main():
    print("Feasibility ceiling vs. cv-PR/parallel-analysis k ...")
    ceiling = feasibility_ceiling()
    for dataset, row in ceiling.items():
        if "cv_PR_mean" in row:
            print(f"  {dataset:14s} ceiling k<={row['ceiling_k_le_T_minus_2']}  "
                  f"cv-PR={row['cv_PR_mean']:.1f} ({'EXCEEDS' if row['cv_PR_exceeds_ceiling'] else 'fits'})  "
                  f"k_PA={row['k_parallel_analysis_median']:.1f} ({'EXCEEDS' if row['k_PA_exceeds_ceiling'] else 'fits'})")
        else:
            print(f"  {dataset:14s} {row['note']}")

    print("\nAxis-rotation headline at DANDI 000469's own parallel-analysis k ...")
    axis = extended_axis_rotation_headline()
    print(f"  k={axis['k']}: axis_rot_diff={axis['axis_rot_diff']:+.4f} p={axis['axis_rot_p']:.4f} "
          f"(N={axis['n_subjects']} subjects)")

    print("\nPer-cohort ensemble one-step CV R^2 across a rank sweep ...")
    curves = cv_r2_curve_all_cohorts()

    print("\nVector-vs-subspace rank sensitivity (macaque PFC microstimulation, cited from existing artifacts) ...")
    vve = vector_vs_subspace_rank_sensitivity_macaque_pfc_microstimulation()
    print(f"  {vve['conclusion']}")

    out = {
        "43B_feasibility_ceiling": ceiling,
        "43C_axis_rotation_at_own_k": axis,
        "43D_cv_r2_curve_by_cohort": curves,
        "43E_vector_vs_subspace_rank_sensitivity_macaque_pfc_microstimulation": vve,
    }
    with open(RESULTS / "dim_addendum.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/dim_addendum.json")


if __name__ == "__main__":
    main()
