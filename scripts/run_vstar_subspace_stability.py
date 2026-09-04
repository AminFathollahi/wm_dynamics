#!/usr/bin/env python3
"""Invariant-subspace (S_m) stability vs single-eigenvector (v*) stability,
across operator rank r in {4..8} and across bootstrap draws, for the macaque PFC microstimulation
uStim cohort. If S_m (m=2,3) is stable where v* (m=1) is not, that is
independent evidence that the causally-relevant target is better described
as a low-dimensional subspace than a single direction -- checked here
BEFORE looking at whether a causal alignment-to-subspace modifier actually
predicts stimulation efficacy (see run_macaque_pfc_microstimulation_headline_robustness.py).

Reuses: run_vstar_eigen_audit's macaque PFC microstimulation control-latent loader and its
mean-trajectory DMD fit machinery; src.control.invariant_subspace_basis and
src.control.canonicalize_eigenvector_phase (not duplicated here).

The m=1 arm is the single, phase-canonicalized leading eigenvector (a 1-D
basis), not invariant_subspace_basis(A, 1) -- that function returns a 2-D
real basis whenever the leading mode is a complex-conjugate pair, which
would silently score a 2-D subspace under the "vector" label for most
sessions here. Using the canonicalized single vector keeps m=1 a genuine
1-D comparison against the m=2/3 subspace arms.

Two stability measures, mirroring the existing v* anchor conventions:
  (i)  RANK stability: subspace affinity (largest principal angle's cosine,
       via ||Q_r^T Q_6|| operator norm) of S_m(r) to S_m(r=6), for
       r in {4,5,7,8} -- the direct S_m analogue of dmd_rank_selection.json's
       vstar_cos_to_r6 (0.783/0.784/1.0/0.904/0.826 at r=4/5/6/7/8).
  (ii) BOOTSTRAP stability: mean subspace affinity of each bootstrap-resampled
       S_m(r=7) to the full-sample S_m(r=7), the S_m analogue of the vector
       bootstrap-cosine dispersion already computed in vstar_eigen_audit.json.

Output: results/vstar_subspace_stability.json.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_vstar_subspace_stability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics import dmd_reconstruction_error
from control import invariant_subspace_basis, dominant_eigenmode
from statistics import stable_seed

import run_vstar_eigen_audit as vea

RESULTS = ROOT / "results"
RANK_SWEEP = (4, 5, 6, 7, 8)
B_BOOT = 300


def _operator_at_rank(Z_mean: np.ndarray, r: int, dt: float) -> np.ndarray:
    T, d = Z_mean.shape
    r_use = min(r, d, T - 2)
    return dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)["A"]


def _subspace_affinity(Q_a: np.ndarray, Q_b: np.ndarray) -> float:
    """||Q_a^T Q_b||_2 (largest singular value) -- 1.0 iff the subspaces are
    identical (or one contains the other), 0.0 iff orthogonal. This is the
    natural subspace generalization of |cos(v_a, v_b)| (equal at 1-D)."""
    return float(np.linalg.norm(Q_a.T @ Q_b, ord=2))


def _basis_for(A: np.ndarray, m: int) -> np.ndarray:
    """m=1: the single phase-canonicalized dominant-by-modulus eigenvector
    (src.control.dominant_eigenmode), as a 1-D basis -- the same v* convention
    used everywhere else in this project. m>1: the real invariant subspace of
    the m largest-|lambda| modes (src.control.invariant_subspace_basis)."""
    if m == 1:
        return dominant_eigenmode(A).v_star.reshape(-1, 1)
    return invariant_subspace_basis(A, m).basis


def main():
    print("Subspace stability (rank sweep + bootstrap) vs vector stability -- macaque PFC microstimulation, m in {1,2,3}")
    out = {}
    for dataset, session, Z_trials, dt, r in vea._iter_macaque_pfc_microstimulation():
        Z_mean = Z_trials.mean(0)
        A6 = _operator_at_rank(Z_mean, 6, dt)
        basis6 = {m: _basis_for(A6, m) for m in (1, 2, 3)}

        rank_affinity = {m: [] for m in (1, 2, 3)}
        for rr in RANK_SWEEP:
            if rr == 6:
                continue
            Ar = _operator_at_rank(Z_mean, rr, dt)
            for m in (1, 2, 3):
                Qr = _basis_for(Ar, m)
                rank_affinity[m].append(_subspace_affinity(Qr, basis6[m]))

        # Bootstrap stability at r=7 (the benchmark's operative rank)
        rng = np.random.default_rng(stable_seed(f"vstar_subspace_stability_{session}"))
        N = Z_trials.shape[0]
        A7_full = _operator_at_rank(Z_mean, 7, dt)
        basis7_full = {m: _basis_for(A7_full, m) for m in (1, 2, 3)}
        boot_affinity = {m: [] for m in (1, 2, 3)}
        for _ in range(B_BOOT):
            idx = rng.integers(0, N, size=N)
            try:
                A7_b = _operator_at_rank(Z_trials[idx].mean(0), 7, dt)
            except np.linalg.LinAlgError:
                continue
            for m in (1, 2, 3):
                Qb = _basis_for(A7_b, m)
                boot_affinity[m].append(_subspace_affinity(Qb, basis7_full[m]))

        row = {
            "rank_affinity_to_r6": {str(m): rank_affinity[m] for m in (1, 2, 3)},
            "rank_affinity_min": {str(m): float(min(rank_affinity[m])) for m in (1, 2, 3)},
            "bootstrap_affinity_mean": {str(m): float(np.mean(boot_affinity[m])) for m in (1, 2, 3)},
        }
        out[session] = row
        print(f"  {session}: min rank-affinity m=1/2/3 = "
              f"{row['rank_affinity_min']['1']:.3f}/{row['rank_affinity_min']['2']:.3f}/{row['rank_affinity_min']['3']:.3f}  "
              f"bootstrap-affinity m=1/2/3 = "
              f"{row['bootstrap_affinity_mean']['1']:.3f}/{row['bootstrap_affinity_mean']['2']:.3f}/{row['bootstrap_affinity_mean']['3']:.3f}")

    # Cross-session summary: is S_2/S_3 markedly more stable and flatter in r than v* (m=1)?
    sessions = list(out.keys())
    summary = {}
    for m in (1, 2, 3):
        rank_mins = [out[s]["rank_affinity_min"][str(m)] for s in sessions]
        boot_means = [out[s]["bootstrap_affinity_mean"][str(m)] for s in sessions]
        summary[str(m)] = {
            "mean_rank_affinity_min": float(np.mean(rank_mins)),
            "mean_bootstrap_affinity": float(np.mean(boot_means)),
        }
    print("\nCross-session mean (macaque PFC microstimulation, N=11):")
    for m in (1, 2, 3):
        s = summary[str(m)]
        print(f"  m={m}: mean(min rank-affinity to r6)={s['mean_rank_affinity_min']:.3f}  "
              f"mean(bootstrap affinity)={s['mean_bootstrap_affinity']:.3f}")

    # Reference: the existing vstar_cos_to_r6 (0.783/0.784/1.0/0.904/0.826 at r=4/5/6/7/8)
    # and the vector bootstrap-cosine dispersion already in vstar_eigen_audit.json.
    eigen_audit = json.load(open(RESULTS / "vstar_eigen_audit.json"))["macaque_pfc_microstimulation"]
    vector_boot_cos = {s: eigen_audit[s]["bootstrap_vstar_cos_mean"] for s in sessions}
    print(f"\nVector (m=1) reference bootstrap |cos| mean across sessions: "
          f"{np.mean(list(vector_boot_cos.values())):.3f}")

    m2_more_stable = summary["2"]["mean_bootstrap_affinity"] > np.mean(list(vector_boot_cos.values()))
    m3_more_stable = summary["3"]["mean_bootstrap_affinity"] > np.mean(list(vector_boot_cos.values()))
    prediction = "CONFIRMED" if (m2_more_stable and m3_more_stable) else "NOT CONFIRMED (report as-is)"
    print(f"PREDICTION (subspace affinity markedly higher/flatter than vector cos): {prediction}")

    out["_meta"] = {
        "rank_sweep": list(RANK_SWEEP), "B": B_BOOT,
        "cross_session_summary": summary,
        "vector_bootstrap_cos_reference": vector_boot_cos,
        "vector_bootstrap_cos_mean": float(np.mean(list(vector_boot_cos.values()))),
        "prediction_subspace_more_stable_than_vector": prediction,
    }
    with open(RESULTS / "vstar_subspace_stability.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/vstar_subspace_stability.json")


if __name__ == "__main__":
    main()
