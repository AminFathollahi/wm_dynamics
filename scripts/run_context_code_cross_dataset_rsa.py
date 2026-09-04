#!/usr/bin/env python3
"""Representational similarity analysis of the load/context code's geometric
format across datasets, brain regions, participants, and recording modalities.

A trained decoder's weights cannot be applied across datasets directly (each
subject/session has its own unique channel or unit space), so cross-dataset
comparison instead uses representational similarity analysis (Kriegeskorte et
al. 2008): within each subject or session, a low/high load by early/late
delay condition-averaged representational dissimilarity matrix (RDM) is
computed from that subject's own latent trajectories, then averaged across
subjects within a dataset. Because an RDM depends only on relative distances
between condition-averaged patterns, not on the specific basis a given
subject's principal components happen to span, RDMs (unlike raw activity
patterns or decoder weights) are directly comparable across datasets with
non-overlapping feature spaces. Extends the existing Miller-versus-Boran-iEEG
comparison (results/09_cross_dataset_rsa.npz) to all six datasets with a
load/set-size manipulation.

Outputs: results/context_code_cross_dataset_rsa.json
Updates: results/all_statistics.json — "context_code_cross_dataset_rsa" key

Run (after the per-dataset geometry pipelines have produced their npz files):
    conda run -n wm_dynamics python scripts/run_context_code_cross_dataset_rsa.py
"""
import sys, json, itertools
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import representational_dissimilarity_matrix
from statistics import mantel_test, fdr_bh
from provenance import _json_safe

RESULTS = ROOT / "results"
CONDITION_LABELS = ["low-early", "low-late", "high-early", "high-late"]

# Seconds after delay/maintenance-window onset (times[0] of the array passed
# in) at which to split "early" vs "late". A sample-index midpoint
# (Z.shape[1]//2) lands at a different absolute post-onset time in every
# dataset (Miller's 1.1s maintenance window vs Boran's 3.0s vs DANDI's
# 2.2s epoch, at different sampling rates), so "early"/"late" would not
# refer to the same delay-relative moment across datasets. This value
# matches Miller's previous ~850ms split (maintenance onset 0.3s + 0.55s)
# and is applied uniformly so the split is delay-relative-time-matched
# rather than sample-count-matched (audit item 1c-ii).
DELAY_SPLIT_REL_S = 0.55


def _split_index(times: np.ndarray) -> int:
    split_t = times[0] + DELAY_SPLIT_REL_S
    idx = int(np.searchsorted(times, split_t))
    return max(1, min(idx, len(times) - 1))


def condition_rdm(Z: np.ndarray, load_bin: np.ndarray, times: np.ndarray) -> np.ndarray | None:
    """(4, 4) RDM over {low, high} x {early, late} condition-averaged patterns.

    Z : (N, T, k); load_bin : (N,) 0/1; times : (T,) — early/late split at a
    matched absolute delay-relative time (DELAY_SPLIT_REL_S), not T//2.
    """
    mid_t = _split_index(times)
    early = Z[:, :mid_t, :].mean(axis=1)   # (N, k)
    late = Z[:, mid_t:, :].mean(axis=1)    # (N, k)
    patterns = []
    for arr in (early, late):
        for load_val in (0, 1):
            m = load_bin == load_val
            if m.sum() < 3:
                return None
            patterns.append(arr[m].mean(axis=0))
    # order: low-early, high-early, low-late, high-late -> reorder to match CONDITION_LABELS
    patterns = [patterns[0], patterns[2], patterns[1], patterns[3]]
    return representational_dissimilarity_matrix(np.stack(patterns), metric="correlation")


def miller_rdms() -> list[np.ndarray]:
    rdms = []
    for subj in ["al", "ca", "cc", "ug"]:
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id, times = d["Z"], d["task_id"], d["times"]
        mask = (task_id == 0) | (task_id == 2)
        maint = (times >= 0.3) & (times <= 1.4)
        Z_m, task_m = Z[mask][:, maint, :], task_id[mask]
        load_bin = (task_m == 2).astype(int)
        rdm = condition_rdm(Z_m, load_bin, times[maint])
        if rdm is not None:
            rdms.append(rdm)
    return rdms


def boran_ieeg_rdms() -> list[np.ndarray]:
    rdms = []
    for path in sorted(RESULTS.glob("boran_geometry_sub-*.npz")):
        d = np.load(path, allow_pickle=True)
        Z, set_sizes, times = d["Z"], d["set_sizes"], d["times"]
        mask = (set_sizes == 4) | (set_sizes == 8)
        Z_m, ss_m = Z[mask], set_sizes[mask]
        load_bin = (ss_m == 8).astype(int)
        rdm = condition_rdm(Z_m, load_bin, times)
        if rdm is not None:
            rdms.append(rdm)
    return rdms


def boran_units_rdms() -> list[np.ndarray]:
    rdms = []
    for path in sorted(RESULTS.glob("dandi000574_units_geometry_*.npz")):
        d = np.load(path, allow_pickle=True)
        if "Z" not in d or "set_size" not in d:
            continue
        Z, set_sizes, times = d["Z"], d["set_size"], d["times"]
        mask = (set_sizes == 4) | (set_sizes == 8)
        Z_m, ss_m = Z[mask], set_sizes[mask]
        if Z_m.shape[0] < 12:
            continue
        load_bin = (ss_m == 8).astype(int)
        rdm = condition_rdm(Z_m, load_bin, times)
        if rdm is not None:
            rdms.append(rdm)
    return rdms


def dandi_rdms(glob_pattern: str) -> list[np.ndarray]:
    rdms = []
    for path in sorted(RESULTS.glob(glob_pattern)):
        d = np.load(path, allow_pickle=True)
        Z, loads, times = d["Z"], d["loads"], d["times"]
        mask = (loads == 1) | (loads == 3)
        Z_m, load_m = Z[mask], loads[mask]
        if Z_m.shape[0] < 12:
            continue
        load_bin = (load_m == 3).astype(int)
        rdm = condition_rdm(Z_m, load_bin, times)
        if rdm is not None:
            rdms.append(rdm)
    return rdms


def main():
    rng = np.random.default_rng(0)

    datasets = {
        "miller_ecog": miller_rdms(),
        "boran_ieeg": boran_ieeg_rdms(),
        "boran_units": boran_units_rdms(),
        "dandi000469": dandi_rdms("dandi000469_geometry_sub-*.npz"),
        "dandi001187": dandi_rdms("dandi001187_geometry_sub-*.npz"),
        "dandi000673": dandi_rdms("dandi000673_geometry_sub-*.npz"),
    }

    group_rdms = {}
    for name, rdms in datasets.items():
        print(f"{name}: {len(rdms)} subject/session RDMs")
        if rdms:
            group_rdms[name] = np.mean(np.stack(rdms), axis=0)

    pairwise = {}
    for a, b in itertools.combinations(group_rdms.keys(), 2):
        res = mantel_test(group_rdms[a], group_rdms[b], n_perm=5000, rng=rng)
        pairwise[f"{a}_vs_{b}"] = res
        print(f"  {a} vs {b}: r={res['r']:.3f}, p={res['p_value']:.4f}")

    pair_names = list(pairwise.keys())
    fdr = fdr_bh(np.array([pairwise[k]["p_value"] for k in pair_names]))
    for k, q, rej in zip(pair_names, fdr["q_values"], fdr["reject"]):
        pairwise[k]["q_value"] = float(q)
        pairwise[k]["significant_fdr"] = bool(rej)

    n_significant = sum(1 for v in pairwise.values() if v["p_value"] < 0.05)
    n_significant_fdr = int(fdr["n_reject"])
    print(f"\n{n_significant}/{len(pairwise)} dataset pairs significant at uncorrected p<0.05; "
          f"{n_significant_fdr}/{len(pairwise)} survive FDR (BH) q<0.05")

    out = {
        "condition_labels": CONDITION_LABELS,
        "group_rdms": {k: v.tolist() for k, v in group_rdms.items()},
        "n_subject_rdms": {k: len(v) for k, v in datasets.items()},
        "pairwise_mantel": pairwise,
        "n_significant_pairs": n_significant, "n_pairs": len(pairwise),
        "n_significant_pairs_fdr": n_significant_fdr,
    }
    with open(RESULTS / "context_code_cross_dataset_rsa.json", "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)

    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    stats["context_code_cross_dataset_rsa"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/context_code_cross_dataset_rsa.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
