#!/usr/bin/env python3
"""Tests whether an item-identity decoder generalizes across working-memory
load levels in DANDI 000469: loadsEnc1_PicIDs labels the first encoded item's
identity on every trial regardless of final load, so a decoder trained on
load-1 (single-item) trials can be evaluated on load-3 (three-item) trials'
maintenance-period activity, and vice versa.

Uses geometry.cross_condition_decoding_test on the already-computed
8-dimensional latent trajectories (results/dandi000469_geometry_sub-*.npz).

Outputs: results/item_identity_load_generalization_000469.json
Updates: results/all_statistics.json — "item_identity_load_generalization_000469" key

Run (after run_000469_pipeline.py):
    conda run -n wm_dynamics python scripts/run_item_identity_load_generalization_000469.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import cross_condition_decoding_test
from statistics import stable_seed, stouffer_combine

RESULTS = ROOT / "results"
N_PERM = 500
MIN_TRIALS_PER_CLASS = 5


def _class_counts_ok(labels: np.ndarray) -> bool:
    counts = np.bincount(labels - labels.min())
    return counts[counts > 0].min() >= MIN_TRIALS_PER_CLASS and len(counts[counts > 0]) >= 2


def main():
    per_subject = {}
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]

        mask1, mask3 = loads == 1, loads == 3
        y1, y3 = pic_id[mask1], pic_id[mask3]
        if not (_class_counts_ok(y1) and _class_counts_ok(y3)):
            continue

        X1 = Z[mask1].transpose(0, 2, 1)
        X3 = Z[mask3].transpose(0, 2, 1)
        t_idx = np.arange(X1.shape[2])
        n_components = min(8, X1.shape[1])

        rng_13 = np.random.default_rng(stable_seed(subj + "_gen_load1_to_load3"))
        gen_1_to_3 = cross_condition_decoding_test(X1, y1, X3, y3, t_idx,
                                                    n_components=n_components, n_perm=N_PERM, rng=rng_13)
        rng_31 = np.random.default_rng(stable_seed(subj + "_gen_load3_to_load1"))
        gen_3_to_1 = cross_condition_decoding_test(X3, y3, X1, y1, t_idx,
                                                    n_components=n_components, n_perm=N_PERM, rng=rng_31)

        per_subject[subj] = {
            "load1_to_load3": {"mean_auc": float(np.nanmean(gen_1_to_3["auc_per_t"])),
                               "mean_p": float(np.nanmean(gen_1_to_3["p_per_t"])),
                               "auc_per_t": gen_1_to_3["auc_per_t"].tolist()},
            "load3_to_load1": {"mean_auc": float(np.nanmean(gen_3_to_1["auc_per_t"])),
                               "mean_p": float(np.nanmean(gen_3_to_1["p_per_t"])),
                               "auc_per_t": gen_3_to_1["auc_per_t"].tolist()},
        }
        print(f"{subj}: load1->load3 mean AUC={per_subject[subj]['load1_to_load3']['mean_auc']:.3f} "
              f"(p={per_subject[subj]['load1_to_load3']['mean_p']:.3f}) | "
              f"load3->load1 mean AUC={per_subject[subj]['load3_to_load1']['mean_auc']:.3f} "
              f"(p={per_subject[subj]['load3_to_load1']['mean_p']:.3f})")

    p_13 = [v["load1_to_load3"]["mean_p"] for v in per_subject.values()]
    p_31 = [v["load3_to_load1"]["mean_p"] for v in per_subject.values()]
    pooled = {
        "load1_to_load3": stouffer_combine(np.array(p_13)),
        "load3_to_load1": stouffer_combine(np.array(p_31)),
        "n_subjects": len(per_subject),
    }
    print(f"\nPooled (N={len(per_subject)} subjects): "
          f"load1->load3 z={pooled['load1_to_load3']['z_combined']:.3f}, "
          f"p={pooled['load1_to_load3']['p_combined']:.4g}; "
          f"load3->load1 z={pooled['load3_to_load1']['z_combined']:.3f}, "
          f"p={pooled['load3_to_load1']['p_combined']:.4g}")

    out = {"per_subject": per_subject, "pooled": pooled}
    with open(RESULTS / "item_identity_load_generalization_000469.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    stats["item_identity_load_generalization_000469"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/item_identity_load_generalization_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
