#!/usr/bin/env python3
"""Toward anatomical localization of the optimal stimulation site.

Timing (∇·v, sign of the flow divergence) tells us WHEN a contracting vs
expanding regime holds. This script asks the complementary WHERE question:
within a subject's own electrode array, which recording site is best placed
to drive the dominant unstable/growth direction v* of the WM attractor?

Two independent signals are computed per subject and cross-checked:

  1. Intrinsic geometric alignment (model-free, no TES1 needed):
     for each of the subject's own recording electrodes i, the PCA loading
     row V[i, :] maps that electrode's activity onto the latent state. Its
     alignment with v* — score_i = |cos(V[i,:], v*)| — indexes how strongly
     that electrode's local field is coupled to the unstable mode, i.e. how
     much a perturbation delivered there would project onto v*.

  2. TES1-informed per-electrode input strength: for the dynamically-best
     TES1 donor already identified in run_divergence_analysis.py
     (argmax alignment of donor B with v*), we rebuild the per-electrode
     induced-voltage field B_electrode at the subject's own electrode
     positions (build_tes1_input_matrix) and rank electrodes by |B_i|.

Both signals are geometric proxies, not causal evidence — no dataset in
hand delivers real intracranial stimulation with a WM behavioral outcome
at cortical sites during maintenance (TES1 gives scalp-tES-induced
intracranial voltage field magnitudes, not behavior; FR3 gives free-recall
stimulation but only during episodic encoding — see Discussion). They are
reported as a hypothesis-generating
localization signal, cross-validated by (i) agreement between the two
independent scores and (ii) consistency of the anatomical gradient across
subjects within a dataset (Miller PFC ECoG, Boran MTL iEEG).

Saves: results/stim_location_analysis.npz, updates all_statistics.json

Run from project root:
    conda run -n wm_dynamics python scripts/run_stim_location_analysis.py
"""
from __future__ import annotations
import sys, json
import numpy as np
from scipy import stats as sstats
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

from preprocessing import load_tes1_stimulation, build_tes1_input_matrix
from statistics import pearson_permutation_test, spearman_permutation_test
from provenance import _json_safe

RESULTS  = ROOT / "results"
EXT_DATA = data_root()
TES1_ZIP = EXT_DATA / "Tes1" / "HuangLiu2016dataset.zip"

MILLER_SUBJECTS = ["al", "ca", "cc", "ug"]
BORAN_SUBJECTS  = [f"sub-{i:02d}" for i in range(1, 10)]
SIGMA_MM = 20.0


def electrode_scores(V: np.ndarray, v_star: np.ndarray) -> np.ndarray:
    """|cos(V[i,:], v*)| per electrode — intrinsic geometric alignment."""
    V_hat = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    v_hat = v_star / (np.linalg.norm(v_star) + 1e-12)
    return np.abs(V_hat @ v_hat)


def axis_gradient(mni: np.ndarray, score: np.ndarray, rng: np.random.Generator | None = None) -> dict:
    """Correlate per-electrode score with each MNI axis (x=lateral, y=AP, z=SI).

    Reports a permutation p-value (primary -- valid at the small per-subject
    electrode counts here) alongside the analytic pearsonr p-value
    (`p_{name}_analytic`) for comparison.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for ax, name in zip(range(3), ["x_lat", "y_ap", "z_si"]):
        res = pearson_permutation_test(mni[:, ax], score, rng=rng)
        out[f"r_{name}"] = res["r"]
        out[f"p_{name}"] = res["p_value"]
        out[f"p_{name}_analytic"] = res["p_analytic"]
    return out


def process_dataset(label, subjects, out, all_rows, all_tes1):
    print(f"\n  {label}:")
    for subj in subjects:
        div_key = f"{'miller' if label == 'Miller' else 'boran'}_{subj}"
        if f"{div_key}_v_star" not in DIV:
            print(f"    {subj}: no divergence result — skipping")
            continue
        v_star = DIV[f"{div_key}_v_star"]
        dyn_idx = int(DIV[f"{div_key}_dynamic_best_idx"])

        if label == "Miller":
            V = TES1_M[f"{subj}_V"]
            mni = TES1_M[f"{subj}_target_mni"]
        else:
            geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
            if "V" not in geo or "electrode_mni" not in geo:
                print(f"    {subj}: no V/electrode_mni — skipping")
                continue
            V = geo["V"].astype(np.float64)
            mni = geo["electrode_mni"].astype(np.float64)

        n_ch = V.shape[0]
        score_intrinsic = electrode_scores(V, v_star)

        B_el = build_tes1_input_matrix(
            all_tes1[dyn_idx], mni, n_sources=1, sigma_mm=SIGMA_MM
        )[:, 0]
        score_tes1 = np.abs(B_el)

        rho_res = spearman_permutation_test(score_intrinsic, score_tes1)
        rho, p_rho = rho_res["rho"], rho_res["p_value"]
        _, p_rho_analytic = sstats.spearmanr(score_intrinsic, score_tes1)

        top_i = int(np.argmax(score_intrinsic))
        top_mni = mni[top_i]
        centroid_mni = mni.mean(axis=0)
        offset = top_mni - centroid_mni

        grad = axis_gradient(mni, score_intrinsic)

        key = f"{label.lower()}_{subj}"
        out[f"{key}_mni"]              = mni
        out[f"{key}_score_intrinsic"]  = score_intrinsic
        out[f"{key}_score_tes1"]       = score_tes1
        out[f"{key}_top_idx"]          = np.array(top_i)
        out[f"{key}_top_mni"]          = top_mni
        out[f"{key}_centroid_mni"]     = centroid_mni
        out[f"{key}_offset_from_centroid"] = offset
        out[f"{key}_rho_intrinsic_vs_tes1"] = np.array(rho)

        row = {
            "n_channels": n_ch,
            "top_electrode_mni": top_mni.tolist(),
            "array_centroid_mni": centroid_mni.tolist(),
            "offset_from_centroid_mm": offset.tolist(),
            "offset_norm_mm": float(np.linalg.norm(offset)),
            "rho_intrinsic_vs_tes1_informed": float(rho),
            "p_rho": float(p_rho),
            "p_rho_analytic": float(p_rho_analytic),
            **grad,
        }
        all_rows[label.lower()][subj] = row

        print(f"    {subj}: N_ch={n_ch}  top-electrode MNI=({top_mni[0]:.1f},"
              f"{top_mni[1]:.1f},{top_mni[2]:.1f})  offset from centroid="
              f"{np.linalg.norm(offset):.1f} mm  rho(intrinsic,TES1)={rho:.3f} (p={p_rho:.3f})")


def main():
    global DIV, TES1_M

    DIV = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)
    TES1_M = np.load(RESULTS / "tes1_comprehensive.npz", allow_pickle=True)

    print("Loading 17 TES1 CCEP donor subjects...")
    all_tes1 = load_tes1_stimulation(str(TES1_ZIP))
    print(f"  {len(all_tes1)} loaded")

    out = {}
    all_rows = {"miller": {}, "boran": {}}

    process_dataset("Miller", MILLER_SUBJECTS, out, all_rows, all_tes1)
    process_dataset("Boran",  BORAN_SUBJECTS,  out, all_rows, all_tes1)

    np.savez(RESULTS / "stim_location_analysis.npz", **out)
    print("\n  Saved: results/stim_location_analysis.npz")

    # ── Cross-subject summary: consistent anatomical gradient? ────────────────
    for label in ["miller", "boran"]:
        rows = all_rows[label]
        if not rows:
            continue
        rhos = [r["rho_intrinsic_vs_tes1_informed"] for r in rows.values()]
        offs = [r["offset_norm_mm"] for r in rows.values()]
        r_y  = [r["r_y_ap"] for r in rows.values()]
        # sign test: do subjects agree on the sign of the AP (y) gradient?
        n_pos = sum(1 for r in r_y if r > 0)
        n_neg = sum(1 for r in r_y if r < 0)
        print(f"\n  {label.capitalize()} cross-subject summary (N={len(rows)}):")
        print(f"    rho(intrinsic, TES1-informed): {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
        print(f"    |offset from array centroid|: {np.mean(offs):.1f} ± {np.std(offs):.1f} mm")
        print(f"    AP (y) gradient sign: {n_pos} positive / {n_neg} negative "
              f"(of {len(r_y)} subjects)")

    all_stats_path = RESULTS / "all_statistics.json"
    with open(all_stats_path) as f:
        all_stats = json.load(f)
    all_stats["stim_location"] = all_rows
    with open(all_stats_path, "w") as f:
        json.dump(_json_safe(all_stats), f, indent=2, allow_nan=False)
    print("\n  Updated all_statistics.json")


if __name__ == "__main__":
    print("Stimulation location analysis (intrinsic geometry + TES1-informed)...")
    main()
    print("Done.")
