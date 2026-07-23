#!/usr/bin/env python3
"""Tests whether the context (load) decoding axis carries item-identity
information, and whether the item-identity decoding axis carries context
information, in DANDI 000469.

Uses geometry.cross_decoding_leakage_test on the already-computed 8-dimensional
latent trajectories (results/dandi000469_geometry_sub-*.npz), at the temporal
midpoint of the maintenance window.

Outputs: results/cross_decoding_orthogonality_000469.json
Updates: results/all_statistics.json — "cross_decoding_orthogonality_000469" key

Run (after run_000469_pipeline.py):
    conda run -n wm_dynamics python scripts/run_cross_decoding_orthogonality_000469.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import cross_decoding_leakage_test
from statistics import stable_seed, stouffer_combine

RESULTS = ROOT / "results"
N_PERM = 2000


def main():
    per_subject = {}
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]

        ctx_mask = (loads == 1) | (loads == 3)
        if ctx_mask.sum() < 15:
            continue
        X_ctx = Z[ctx_mask].transpose(0, 2, 1)   # (N, C, T)
        context_labels = (loads[ctx_mask] == 3).astype(int)
        content_labels_at_ctx = pic_id[ctx_mask]
        t_mid = X_ctx.shape[2] // 2

        rng_a = np.random.default_rng(stable_seed(subj + "_leak_ctx_to_content"))
        context_to_content = cross_decoding_leakage_test(
            X_ctx, context_labels, content_labels_at_ctx, t_idx=t_mid,
            n_components=min(8, X_ctx.shape[1]), n_splits=5, n_perm=N_PERM, rng=rng_a,
        )

        load1_mask = loads == 1
        content_labels = pic_id[load1_mask]
        if load1_mask.sum() < 15 or len(np.unique(content_labels)) < 2:
            continue
        X_content = Z[load1_mask].transpose(0, 2, 1)
        # context is constant (load=1) within this trial set, so leakage of
        # context information is instead tested using the load 1-vs-3 set,
        # decoding item identity and testing for load leakage
        rng_b = np.random.default_rng(stable_seed(subj + "_leak_content_to_ctx"))
        content_to_context = cross_decoding_leakage_test(
            X_ctx, content_labels_at_ctx, context_labels, t_idx=t_mid,
            n_components=min(8, X_ctx.shape[1]), n_splits=5, n_perm=N_PERM, rng=rng_b,
        )

        per_subject[subj] = {
            "context_axis_leaks_content": context_to_content,
            "content_axis_leaks_context": content_to_context,
        }
        print(f"{subj}: context->content F={context_to_content['f_statistic']:.3f} "
              f"p={context_to_content['p_value']:.4f} | "
              f"content->context F={content_to_context['f_statistic']:.3f} "
              f"p={content_to_context['p_value']:.4f}")

    p_ctx_to_content = [v["context_axis_leaks_content"]["p_value"] for v in per_subject.values()]
    p_content_to_ctx = [v["content_axis_leaks_context"]["p_value"] for v in per_subject.values()]
    pooled = {
        "context_axis_leaks_content": stouffer_combine(np.array(p_ctx_to_content)),
        "content_axis_leaks_context": stouffer_combine(np.array(p_content_to_ctx)),
        "n_subjects": len(per_subject),
    }
    print(f"\nPooled (N={len(per_subject)} subjects): "
          f"context->content z={pooled['context_axis_leaks_content']['z_combined']:.3f}, "
          f"p={pooled['context_axis_leaks_content']['p_combined']:.4g}; "
          f"content->context z={pooled['content_axis_leaks_context']['z_combined']:.3f}, "
          f"p={pooled['content_axis_leaks_context']['p_combined']:.4g}")

    out = {"per_subject": per_subject, "pooled": pooled}
    with open(RESULTS / "cross_decoding_orthogonality_000469.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    stats["cross_decoding_orthogonality_000469"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/cross_decoding_orthogonality_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
