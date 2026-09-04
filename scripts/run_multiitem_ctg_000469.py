#!/usr/bin/env python3
"""Full cross-temporal generalization matrix (with tau, not just diagonal
decoding) for each of the three sequentially-encoded items in DANDI 000469
load-3 trials, over the maintenance window.

Uses geometry.ctg_content_permutation_null identically to the load-1 item1
analysis in run_000469_pipeline.py, applied to loadsEnc1/2/3_PicIDs within
load-3 trials, so the three items' results are directly comparable to each
other and to the load-1 case via the same pipeline. Reuses the
already-computed maintenance-window latent trajectories
(results/dandi000469_geometry_sub-*.npz) rather than reprocessing spikes;
item2/item3 identity labels are read directly from the NWB trials table
(a fast metadata read, no spike reprocessing).

Outputs: results/multiitem_ctg_000469.json
Updates: results/all_statistics.json — "multiitem_ctg_000469" key

Run (after run_000469_pipeline.py):
    conda run -n wm_dynamics python scripts/run_multiitem_ctg_000469.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py
from geometry import ctg_content_permutation_null, temporal_stability_tau
from statistics import stable_seed, stouffer_combine
from io_utils import locked_json_update

DATA_DIR = dataset_path("dandi_000469")
RESULTS = ROOT / "results"
N_PC = 8
CTG_STEP = 3
N_SPLITS = 3
N_PERM = 200
MIN_TRIALS_PER_CLASS = 6

ITEM_FIELDS = {"item1": "loadsEnc1_PicIDs", "item2": "loadsEnc2_PicIDs", "item3": "loadsEnc3_PicIDs"}


def _class_counts_ok(labels: np.ndarray) -> bool:
    counts = np.bincount(labels - labels.min())
    return counts[counts > 0].min() >= MIN_TRIALS_PER_CLASS and len(counts[counts > 0]) >= 2


def main():
    per_subject = {name: {} for name in ITEM_FIELDS}

    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        nwb_path = DATA_DIR / subj / f"{subj}_ses-2_ecephys+image.nwb"
        if not nwb_path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        Z, loads = d["Z"], d["loads"]
        mask = loads == 3
        if mask.sum() < MIN_TRIALS_PER_CLASS * 2:
            continue

        with h5py.File(str(nwb_path), "r") as f:
            item_labels = {name: f[f"intervals/trials/{field}"][:].astype(int)
                           for name, field in ITEM_FIELDS.items()}

        X = Z[mask].transpose(0, 2, 1)   # (N, C, T)
        t_idx = np.arange(0, X.shape[2], CTG_STEP)
        print(f"{subj}:")
        for name, field in ITEM_FIELDS.items():
            labels = item_labels[name][mask]
            if not _class_counts_ok(labels):
                continue
            rng = np.random.default_rng(stable_seed(f"{subj}_multiitem_ctg_{name}"))
            res = ctg_content_permutation_null(X, labels, t_idx, n_components=N_PC,
                                               n_splits=N_SPLITS, n_perm=N_PERM, rng=rng)
            tau_info = temporal_stability_tau(res["auc_mat"])
            per_subject[name][subj] = {
                "offdiag_effect": res["mean_offdiag_auc_minus_chance"], "p_value": res["p_value"],
                "tau": tau_info["tau"], "interpretable": tau_info["interpretable"],
                "mean_diag_auc": tau_info["mean_diag_auc"], "n_classes": res["n_classes"],
            }
            print(f"  {name}: offdiag_effect={res['mean_offdiag_auc_minus_chance']:.4f}, "
                  f"tau={tau_info['tau']:.4f} (interpretable={tau_info['interpretable']}), "
                  f"p={res['p_value']:.4f}")

    pooled = {}
    for name, subj_results in per_subject.items():
        p_vals = [v["p_value"] for v in subj_results.values()]
        pooled[name] = stouffer_combine(np.array(p_vals)) if p_vals else None
        if pooled[name] is not None:
            print(f"\n{name} pooled (N={len(p_vals)} sessions): "
                  f"z={pooled[name]['z_combined']:.3f}, p={pooled[name]['p_combined']:.4g}")

    out = {"per_subject": per_subject, "pooled": pooled}
    with open(RESULTS / "multiitem_ctg_000469.json", "w") as f:
        json.dump(out, f, indent=2)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["multiitem_ctg_000469"] = out
    print("\nSaved results/multiitem_ctg_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
