#!/usr/bin/env python3
"""Trajectory-tangling Q(t) FWER-corrected cluster permutation test (target
vs. non-target 2-back trials, Miller ECoG) -- regenerates results/03_cluster_q.npz.

results/03_dynamics.npz (Q_tgt_pool/Q_ntgt_pool) is written by
notebooks/03_tangling_dynamics.ipynb, but that notebook imports
temporal_cluster_permutation without ever calling it or saving a cluster-test
artifact; results/03_cluster_q.npz was a stale, orphaned file (pre-dating a
dt-default fix that changed Q_tgt_pool's shape from 340 to 1700 samples,
detected as a shape mismatch when Figure 4 panel F tried to plot both
against the same time axis). This script closes that gap as a proper,
re-runnable pipeline step instead of a notebook side effect.

Outputs: results/03_cluster_q.npz (t_stat, times, n_sig_clusters)
Updates: results/all_statistics.json -- "tangling_cluster_test" key

Run:
    conda run -n wm_dynamics python scripts/run_tangling_cluster_test.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from statistics import temporal_cluster_permutation, stable_seed

RESULTS = ROOT / "results"
N_PERM = 5000


def main():
    dyn = np.load(RESULTS / "03_dynamics.npz", allow_pickle=True)
    Q_tgt_pool, Q_ntgt_pool = dyn["Q_tgt_pool"], dyn["Q_ntgt_pool"]
    T = Q_tgt_pool.shape[1]

    geo = np.load(RESULTS / "02_geometry_al.npz", allow_pickle=True)
    times = geo["times"][:T]

    rng = np.random.default_rng(stable_seed("tangling_cluster_test"))
    res = temporal_cluster_permutation(Q_tgt_pool, Q_ntgt_pool, times,
                                       n_perm=N_PERM, rng=rng)
    n_sig = len(res["significant"])
    print(f"Trajectory tangling Q(t), target vs. non-target: {len(res['clusters'])} "
          f"cluster(s) found, {n_sig} significant (FWER-corrected, N_perm={N_PERM})")
    for c in res["clusters"]:
        print(f"  {c['start_s']:.2f}-{c['end_s']:.2f} s: stat={c['cluster_stat']:.2f}, "
              f"p={c['p_value']:.4f}")

    np.savez(RESULTS / "03_cluster_q.npz", t_stat=res["t_stat"], times=times,
             n_sig_clusters=n_sig)

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["tangling_cluster_test"] = {
        "n_clusters": len(res["clusters"]), "n_sig_clusters": n_sig,
        "n_perm": N_PERM,
        "clusters": res["clusters"],
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/03_cluster_q.npz, updated all_statistics.json")


if __name__ == "__main__":
    main()
