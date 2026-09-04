#!/usr/bin/env python3
"""Comprehensive TES1 (Huang et al. 2017 eLife) utilisation.

For each Miller subject:
  1. Load actual electrode MNI coordinates from .mat files.
  2. Re-run PCA on the maintenance-window epochs → V matrix (n_ch, n_pc).
  3. For each of the 17 TES1 subjects: build B_electrode via Gaussian interpolation
     then project to latent space: B_latent = V.T @ B_electrode.
  4. Run exact DMD on the maintenance-window mean trajectory → A matrix.
  5. Run LQR with mean TES1 B, best-matched B, and worst-matched B.
  6. Compute Controllability Gramian trace per TES1 subject.

For each Boran subject:
  - Extract iEEG MNI coordinates from NWB.
  - Build B_electrode from TES1 (Gaussian interpolation at electrode positions).
  - Report B statistics; note that latent projection requires saving V in the
    Boran pipeline (boran_geometry files do not currently store V).

Saves:
  results/tes1_comprehensive.npz
  results/tes1_boran_B.npz
  results/all_statistics.json  (updated with 'tes1_lqr' key)

Run from project root:
  conda run -n wm_dynamics python scripts/run_tes1_analysis.py
"""

from __future__ import annotations
import sys, json, warnings
import numpy as np
import scipy.io as sio
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

from preprocessing import load_tes1_stimulation, build_tes1_input_matrix
from dynamics import exact_dmd
from control import (
    lqr_design, lqr_simulate, controllability_gramian, is_controllable
)
from statistics import robust_dispersion
from provenance import _json_safe

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS   = ROOT / "results"
EXT_DATA  = data_root()
TES1_ZIP  = EXT_DATA / "Tes1" / "HuangLiu2016dataset.zip"
MILLER_LOCS = EXT_DATA / "kai miller" / "memory_nback" / "memory_nback" / "locs"
BORAN_DIR   = EXT_DATA / "000574"

MILLER_SUBJECTS = ["al", "ca", "cc", "ug"]
BORAN_SUBJECTS  = [f"sub-{i:02d}" for i in range(1, 10)]

N_PC      = 8      # latent dimensions (matches geometry files)
SIGMA_MM  = 20.0   # Gaussian interpolation width (mm)
MAINT_T0  = 0.30   # maintenance window start (s)
MAINT_T1  = 1.40   # maintenance window end (s)
T_CTRL    = 150    # LQR simulation horizon (steps)
Q_STATE   = 10.0   # LQR state cost weight


# ── PCA helper ────────────────────────────────────────────────────────────────

def run_pca_maintenance(epochs: np.ndarray, times: np.ndarray, n_pc: int = N_PC):
    """Fit PCA on maintenance-window data, return V (n_ch, n_pc) and Z."""
    maint = (times >= MAINT_T0) & (times <= MAINT_T1)
    X_maint = epochs[:, :, maint]                         # (N, n_ch, T_maint)
    N, n_ch, T_m = X_maint.shape
    X_flat = X_maint.transpose(0, 2, 1).reshape(-1, n_ch) # (N*T_m, n_ch)
    X_flat -= X_flat.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_flat, full_matrices=False)
    V = Vt[:n_pc].T                                        # (n_ch, n_pc)
    Z = X_flat @ V                                         # (N*T_m, n_pc)
    return V, Z, X_flat


def get_maintenance_centroids(epochs: np.ndarray, task_id: np.ndarray,
                               times: np.ndarray, V: np.ndarray):
    """Return per-load mean latent centroids in maintenance window."""
    maint = (times >= MAINT_T0) & (times <= MAINT_T1)
    centroids = {}
    for load in np.unique(task_id):
        mask = task_id == load
        X_m = epochs[mask][:, :, maint]      # (n_trials, n_ch, T_m)
        N_, n_ch_, Tm_ = X_m.shape
        X_flat = X_m.transpose(0, 2, 1).reshape(-1, n_ch_)
        centroids[int(load)] = (X_flat @ V).mean(axis=0)  # (n_pc,)
    return centroids


def get_A_from_mean_trajectory(Z_geo: np.ndarray, times: np.ndarray,
                                task_id: np.ndarray, load: int = 2,
                                r: int = N_PC):
    """Fit DMD to mean maintenance trajectory of a given load condition."""
    maint = (times >= MAINT_T0) & (times <= MAINT_T1)
    mask  = task_id == load
    Z_sub = Z_geo[mask][:, maint, :]           # (N_trials, T_m, n_pc)
    Z_mean = Z_sub.mean(axis=0).T              # (n_pc, T_m)
    dt = float(times[1] - times[0])
    dmd = exact_dmd(Z_mean, r=r, dt=dt)
    n  = Z_mean.shape[0]
    # Reconstruct A_discrete via Phi lam Phi^{-1}
    Phi = dmd["modes"]
    lam = dmd["eigenvalues"]
    try:
        A_dmd = np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))
    except Exception:
        A_dmd = 0.99 * np.eye(n)
    return A_dmd


# ── Miller subjects ───────────────────────────────────────────────────────────

def run_miller(all_tes1: list[dict]) -> dict:
    """Full TES1 B-matrix analysis + LQR for all 4 Miller subjects."""
    miller_results = {}
    n_tes1 = len(all_tes1)

    for subj in MILLER_SUBJECTS:
        print(f"\n  Miller {subj}")

        # Load epoch data
        ep_file = RESULTS / f"01_epochs_{subj}.npz"
        if not ep_file.exists():
            print(f"    SKIP: {ep_file} not found")
            continue
        ep = np.load(ep_file, allow_pickle=True)
        epochs   = ep["epochs"].transpose(0, 2, 1)  # → (N, n_ch, T)
        times    = ep["times"]
        task_id  = ep["task_id"]
        good_ch  = ep["good_channels"]

        # Load actual electrode MNI coordinates
        loc_file = MILLER_LOCS / f"{subj}_electrodes.mat"
        if not loc_file.exists():
            print(f"    SKIP: electrode file {loc_file} not found")
            continue
        coords_all = sio.loadmat(str(loc_file))["electrodes"]  # (n_all, 3)
        target_mni = coords_all[good_ch]                        # (n_good, 3)
        n_ch = target_mni.shape[0]
        print(f"    {n_ch} good channels, {len(times)} time points")

        # PCA on maintenance window
        V, _, _ = run_pca_maintenance(epochs, times, n_pc=N_PC)
        print(f"    V shape: {V.shape}")

        # Maintenance centroids for x0 (0-back) and xf (2-back)
        geo_file = RESULTS / f"02_geometry_{subj}.npz"
        geo = np.load(geo_file, allow_pickle=True)
        Z_geo = geo["Z"]          # (N, T, n_pc)
        centroids = get_maintenance_centroids(epochs, task_id, times, V)
        x0 = centroids.get(0, centroids[min(centroids)])   # 0-back centroid
        xf = centroids.get(2, centroids[max(centroids)])   # 2-back centroid

        # A matrix from DMD on mean maintenance trajectory
        A = get_A_from_mean_trajectory(Z_geo, geo["times"], task_id,
                                        load=2, r=N_PC)
        sr = float(np.max(np.abs(np.linalg.eigvals(A))))
        print(f"    A spectral radius: {sr:.4f}")

        # Build B_latent for each TES1 subject
        B_latent_per_tes1 = []
        gramian_traces = []
        for t_data in all_tes1:
            B_el = build_tes1_input_matrix(
                t_data, target_mni, n_sources=1, sigma_mm=SIGMA_MM
            )                                             # (n_ch, 1)
            B_lat = V.T @ B_el                            # (n_pc, 1)
            B_latent_per_tes1.append(B_lat)

            Wc = controllability_gramian(A, B_lat, T=20)
            gramian_traces.append(float(np.trace(Wc)))

        B_latent_per_tes1 = np.array(B_latent_per_tes1)  # (17, n_pc, 1)
        gramian_traces    = np.array(gramian_traces)       # (17,)

        # Mean / best / worst TES1 B
        B_mean  = B_latent_per_tes1.mean(axis=0)          # (n_pc, 1)
        best_i  = int(np.argmax(gramian_traces))
        worst_i = int(np.argmin(gramian_traces))
        B_best  = B_latent_per_tes1[best_i]
        B_worst = B_latent_per_tes1[worst_i]

        # LQR with each B
        lqr_results = {}
        for label, B in [("mean", B_mean), ("best", B_best), ("worst", B_worst)]:
            try:
                lqr = lqr_design(A, B, q_state=Q_STATE, r_control=1.0)
                x_traj, u_traj = lqr_simulate(
                    A, B, lqr["K"], x0, T=T_CTRL, x_ref=xf
                )
                dist = np.linalg.norm(x_traj - xf, axis=1)
                reduction_pct = 100.0 * (1.0 - dist[-1] / dist[0])
                lqr_results[label] = {
                    "final_error":    float(dist[-1]),
                    "initial_dist":   float(dist[0]),
                    "reduction_pct":  float(reduction_pct),
                    "energy":         float(np.sum(u_traj**2)),
                    "is_stable":      lqr["is_stable"],
                }
                print(f"    LQR {label}: {reduction_pct:.1f}% reduction "
                      f"(err={dist[-1]:.3f})")
            except Exception as e:
                lqr_results[label] = {"error": str(e)}

        # B norm statistics across TES1 subjects
        B_norms = np.linalg.norm(B_latent_per_tes1.reshape(n_tes1, -1), axis=1)
        tes1_ids = [t["subject"] for t in all_tes1]

        miller_results[subj] = {
            "V":                   V,
            "A_dmd":               A,
            "target_mni":          target_mni,
            "x0_centroid":         x0,
            "xf_centroid":         xf,
            "B_latent_per_tes1":   B_latent_per_tes1,
            "gramian_traces":      gramian_traces,
            "B_norms":             B_norms,
            "B_mean":              B_mean,
            "B_best":              B_best,
            "B_worst":             B_worst,
            "best_tes1_idx":       best_i,
            "worst_tes1_idx":      worst_i,
            "best_tes1_subj":      tes1_ids[best_i],
            "worst_tes1_subj":     tes1_ids[worst_i],
            "lqr_results":         lqr_results,
            "spectral_radius":     float(sr),
            "tes1_subject_ids":    tes1_ids,
        }

    return miller_results


# ── Boran subjects ────────────────────────────────────────────────────────────

def run_boran(all_tes1: list[dict]) -> dict:
    """Full TES1 B-matrix analysis + LQR for all 9 Boran subjects.

    Uses V and electrode_mni saved in boran_geometry_{subj}.npz.
    Requires that run_boran_pipeline.py has been run after the V-saving update.
    Falls back to electrode-space B reporting if V is unavailable.
    """
    boran_results = {}
    n_tes1 = len(all_tes1)

    for subj in BORAN_SUBJECTS:
        geo_file = RESULTS / f"boran_geometry_{subj}.npz"
        if not geo_file.exists():
            print(f"  Boran {subj}: geometry file not found — skipping")
            continue

        geo = np.load(geo_file, allow_pickle=True)
        Z_geo = geo["Z"].astype(np.float64)  # (N, T, 8)
        times  = geo["times"]
        set_sizes = geo["set_sizes"]

        # Check if V and electrode coords were saved (new pipeline output)
        has_V     = "V" in geo
        has_coords = "electrode_mni" in geo

        n_ch_good = int(geo["n_channels_good"])
        print(f"  Boran {subj}: {n_ch_good} good channels, V={'yes' if has_V else 'NO'}, "
              f"coords={'yes' if has_coords else 'NO'}")

        if not (has_V and has_coords):
            print(f"    Run run_boran_pipeline.py first to save V and electrode_mni")
            boran_results[subj] = {"status": "needs_pipeline_rerun"}
            continue

        V          = geo["V"].astype(np.float64)          # (C_good, 8)
        target_mni = geo["electrode_mni"].astype(np.float64)  # (C_good, 3)

        # A matrix from DMD on mean maintenance trajectory (all correct trials)
        # Boran maintenance window spans the full times range
        correct = geo.get("correct", np.ones(Z_geo.shape[0], dtype=bool))
        mask_correct = correct.astype(bool)
        Z_maint = Z_geo[mask_correct].mean(axis=0).T   # (8, T)
        dt = float(times[1] - times[0]) if len(times) > 1 else 1.0 / 1398.0
        try:
            dmd = exact_dmd(Z_maint, r=N_PC, dt=dt)
            Phi = dmd["modes"]
            lam = dmd["eigenvalues"]
            A = np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))
        except Exception as e:
            print(f"    DMD failed: {e} — using near-identity A")
            A = 0.99 * np.eye(N_PC)
        sr = float(np.max(np.abs(np.linalg.eigvals(A))))

        # x0 = centroid of set-4 trials (lowest load → easiest WM)
        # xf = centroid of set-8 trials (highest load → hardest WM)
        mask4 = (set_sizes == 4) & correct.astype(bool)
        mask8 = (set_sizes == 8) & correct.astype(bool)
        if mask4.sum() > 0 and mask8.sum() > 0:
            x0 = Z_geo[mask4].mean(axis=(0, 1))   # (8,)
            xf = Z_geo[mask8].mean(axis=(0, 1))   # (8,)
        else:
            x0 = Z_geo.mean(axis=(0, 1))
            xf = x0 + np.random.default_rng(42).normal(0, 0.1, N_PC)

        # Build B_latent per TES1 subject
        B_latent_per_tes1 = []
        gramian_traces = []
        for t_data in all_tes1:
            B_el = build_tes1_input_matrix(
                t_data, target_mni, n_sources=1, sigma_mm=SIGMA_MM
            )                                         # (C_good, 1)
            B_lat = V.T @ B_el                        # (8, 1)
            B_latent_per_tes1.append(B_lat)
            Wc = controllability_gramian(A, B_lat, T=20)
            gramian_traces.append(float(np.trace(Wc)))

        B_latent_per_tes1 = np.array(B_latent_per_tes1)
        gramian_traces    = np.array(gramian_traces)

        B_mean  = B_latent_per_tes1.mean(axis=0)
        best_i  = int(np.argmax(gramian_traces))
        worst_i = int(np.argmin(gramian_traces))
        B_best  = B_latent_per_tes1[best_i]
        B_worst = B_latent_per_tes1[worst_i]

        tes1_ids = [t["subject"] for t in all_tes1]

        # LQR with mean / best / worst B
        lqr_results = {}
        for label, B in [("mean", B_mean), ("best", B_best), ("worst", B_worst)]:
            try:
                lqr = lqr_design(A, B, q_state=Q_STATE, r_control=1.0)
                x_traj, u_traj = lqr_simulate(
                    A, B, lqr["K"], x0, T=T_CTRL, x_ref=xf
                )
                dist = np.linalg.norm(x_traj - xf, axis=1)
                reduction_pct = 100.0 * (1.0 - dist[-1] / dist[0])
                lqr_results[label] = {
                    "final_error":   float(dist[-1]),
                    "initial_dist":  float(dist[0]),
                    "reduction_pct": float(reduction_pct),
                    "energy":        float(np.sum(u_traj**2)),
                    "is_stable":     lqr["is_stable"],
                }
                print(f"    LQR {label}: {reduction_pct:.1f}% reduction")
            except Exception as e:
                lqr_results[label] = {"error": str(e)}

        B_norms = np.linalg.norm(B_latent_per_tes1.reshape(n_tes1, -1), axis=1)

        boran_results[subj] = {
            "V":                   V,
            "A_dmd":               A,
            "target_mni":          target_mni,
            "x0_centroid":         x0,
            "xf_centroid":         xf,
            "B_latent_per_tes1":   B_latent_per_tes1,
            "gramian_traces":      gramian_traces,
            "B_norms":             B_norms,
            "B_mean":              B_mean,
            "B_best":              B_best,
            "B_worst":             B_worst,
            "best_tes1_idx":       best_i,
            "worst_tes1_idx":      worst_i,
            "best_tes1_subj":      tes1_ids[best_i],
            "worst_tes1_subj":     tes1_ids[worst_i],
            "lqr_results":         lqr_results,
            "spectral_radius":     float(sr),
            "tes1_subject_ids":    tes1_ids,
            "status":              "complete",
        }

    return boran_results


# ── Summaries for stats ────────────────────────────────────────────────────────

def _subj_summary(r: dict) -> dict:
    """Build per-subject stats dict for Miller or Boran result."""
    if r.get("status") in ("needs_pipeline_rerun",):
        return {"status": r["status"]}
    traces = r["gramian_traces"]
    lr     = r["lqr_results"]
    gramian_disp = robust_dispersion(traces)
    b_norm_disp  = robust_dispersion(r["B_norms"])
    return {
        "gramian_trace_mean":      float(traces.mean()),
        "gramian_trace_best":      float(traces.max()),
        "gramian_trace_worst":     float(traces.min()),
        "gramian_fold_range":      float(traces.max() / (traces.min() + 1e-10)),
        "gramian_iqr_ratio":       gramian_disp["iqr_ratio"],
        "B_norm_mean":             float(r["B_norms"].mean()),
        "B_norm_std":              float(r["B_norms"].std()),
        "B_norm_fold_range":       float(r["B_norms"].max() / (r["B_norms"].min() + 1e-10)),
        "B_norm_iqr_ratio":        b_norm_disp["iqr_ratio"],
        "B_norm_mad":              b_norm_disp["mad"],
        "best_tes1_subj":          r["best_tes1_subj"],
        "worst_tes1_subj":         r["worst_tes1_subj"],
        "spectral_radius":         r.get("spectral_radius", np.nan),
        "lqr_mean_reduction_pct":  lr.get("mean", {}).get("reduction_pct", np.nan),
        "lqr_best_reduction_pct":  lr.get("best", {}).get("reduction_pct", np.nan),
        "lqr_worst_reduction_pct": lr.get("worst", {}).get("reduction_pct", np.nan),
        "lqr_mean_energy":         lr.get("mean", {}).get("energy", np.nan),
        "lqr_best_energy":         lr.get("best", {}).get("energy", np.nan),
    }


def build_stats_summary(miller_results: dict, boran_results: dict,
                         all_tes1: list[dict]) -> dict:
    n_tes1 = len(all_tes1)
    miller_summary = {s: _subj_summary(r) for s, r in miller_results.items()}
    boran_summary  = {s: _subj_summary(r) for s, r in boran_results.items()
                      if r.get("status") != "needs_pipeline_rerun"}

    # Global B fold-range across all subjects × all TES1 subjects. Reported
    # alongside a robust (IQR-based) dispersion measure: the naive max/min
    # fold-range is dominated by whichever single donor has a near-zero
    # interpolated field and grows with donor count, so it is not by itself
    # a stable summary of cross-donor variability.
    all_fold_ranges = (
        [v["B_norm_fold_range"] for v in miller_summary.values()
         if "B_norm_fold_range" in v]
        + [v["B_norm_fold_range"] for v in boran_summary.values()
           if "B_norm_fold_range" in v]
    )
    all_B_norms = np.concatenate(
        [r["B_norms"] for r in list(miller_results.values()) + list(boran_results.values())
         if "B_norms" in r]
    )
    global_b_disp = robust_dispersion(all_B_norms)

    # Cross-dataset LQR comparison (mean B, % reduction)
    miller_lqr = [v.get("lqr_mean_reduction_pct", np.nan)
                  for v in miller_summary.values()]
    boran_lqr  = [v.get("lqr_mean_reduction_pct", np.nan)
                  for v in boran_summary.values()]

    return {
        "miller":                   miller_summary,
        "boran":                    boran_summary,
        "n_tes1_subjects":          n_tes1,
        "mean_fold_range":          float(np.nanmean(all_fold_ranges)) if all_fold_ranges else 0.0,
        "max_fold_range":           float(np.nanmax(all_fold_ranges)) if all_fold_ranges else 0.0,
        "global_B_norm_iqr_ratio":  global_b_disp["iqr_ratio"],
        "global_B_norm_mad":        global_b_disp["mad"],
        "global_B_norm_median":     global_b_disp["median"],
        "miller_lqr_mean_pct":      float(np.nanmean(miller_lqr)) if miller_lqr else np.nan,
        "boran_lqr_mean_pct":       float(np.nanmean(boran_lqr)) if boran_lqr else np.nan,
        "n_boran_complete":         sum(1 for r in boran_results.values()
                                        if r.get("status") == "complete"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not TES1_ZIP.exists():
        raise FileNotFoundError(f"TES1 data not found: {TES1_ZIP}")

    print("Loading all 17 TES1 subjects...")
    all_tes1 = load_tes1_stimulation(str(TES1_ZIP))
    print(f"  {len(all_tes1)} TES1 subjects loaded")

    # Print basic TES1 stats
    all_volts = np.concatenate([t["voltage_mV"][~np.isnan(t["voltage_mV"])]
                                 for t in all_tes1])
    print(f"  Voltage: min={all_volts.min():.2f}, max={all_volts.max():.2f}, "
          f"std={all_volts.std():.2f} mV/mA")

    print("\n=== Miller subjects ===")
    miller_results = run_miller(all_tes1)

    print("\n=== Boran subjects ===")
    boran_results = run_boran(all_tes1)

    # ── Save comprehensive NPZ ─────────────────────────────────────────────────
    print("\nSaving results...")
    out_miller = {}
    for subj, r in miller_results.items():
        out_miller[f"{subj}_V"]                 = r["V"]
        out_miller[f"{subj}_A_dmd"]             = r["A_dmd"]
        out_miller[f"{subj}_target_mni"]        = r["target_mni"]
        out_miller[f"{subj}_x0"]                = r["x0_centroid"]
        out_miller[f"{subj}_xf"]                = r["xf_centroid"]
        out_miller[f"{subj}_B_latent_per_tes1"] = r["B_latent_per_tes1"]
        out_miller[f"{subj}_gramian_traces"]     = r["gramian_traces"]
        out_miller[f"{subj}_B_norms"]           = r["B_norms"]
        out_miller[f"{subj}_B_mean"]            = r["B_mean"]
        out_miller[f"{subj}_B_best"]            = r["B_best"]
        out_miller[f"{subj}_B_worst"]           = r["B_worst"]

    np.savez_compressed(
        str(RESULTS / "tes1_comprehensive.npz"),
        tes1_subject_ids=np.array([t["subject"] for t in all_tes1]),
        **out_miller,
    )

    # Save Boran TES1 results (latent space, mirrors Miller format)
    boran_complete = {s: r for s, r in boran_results.items()
                      if r.get("status") == "complete"}
    if boran_complete:
        out_boran = {}
        for subj, r in boran_complete.items():
            out_boran[f"{subj}_A_dmd"]             = r["A_dmd"]
            out_boran[f"{subj}_target_mni"]        = r["target_mni"]
            out_boran[f"{subj}_x0"]                = r["x0_centroid"]
            out_boran[f"{subj}_xf"]                = r["xf_centroid"]
            out_boran[f"{subj}_B_latent_per_tes1"] = r["B_latent_per_tes1"]
            out_boran[f"{subj}_gramian_traces"]     = r["gramian_traces"]
            out_boran[f"{subj}_B_norms"]           = r["B_norms"]
            out_boran[f"{subj}_B_mean"]            = r["B_mean"]
        np.savez_compressed(str(RESULTS / "tes1_boran_B.npz"), **out_boran)
    elif boran_results:
        print("  Boran geometry missing V — re-run run_boran_pipeline.py first")

    # ── Update all_statistics.json ─────────────────────────────────────────────
    stats_path = RESULTS / "all_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}

    stats["tes1_lqr"] = build_stats_summary(miller_results, boran_results, all_tes1)

    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n=== Miller Summary ===")
    for subj, r in miller_results.items():
        lqr = r["lqr_results"]
        print(f"\n  {subj}:")
        print(f"    Gramian trace: mean={r['gramian_traces'].mean():.2e}  "
              f"best={r['gramian_traces'].max():.2e}  worst={r['gramian_traces'].min():.2e}")
        b_disp = robust_dispersion(r["B_norms"])
        print(f"    B norm: naive fold-range={r['B_norms'].max()/(r['B_norms'].min()+1e-10):.1f}×  "
              f"IQR ratio={b_disp['iqr_ratio']:.2f}×  MAD={b_disp['mad']:.3g}  "
              f"best TES1={r['best_tes1_subj']}")
        for label in ["mean", "best", "worst"]:
            if label in lqr and "reduction_pct" in lqr[label]:
                print(f"    LQR ({label} B): {lqr[label]['reduction_pct']:.1f}% reduction  "
                      f"err={lqr[label]['final_error']:.3f}")

    boran_complete = {s: r for s, r in boran_results.items()
                      if r.get("status") == "complete"}
    if boran_complete:
        print("\n=== Boran Summary ===")
        for subj, r in boran_complete.items():
            lqr = r["lqr_results"]
            fold = r["B_norms"].max() / (r["B_norms"].min() + 1e-10)
            print(f"\n  {subj}:")
            print(f"    Gramian: mean={r['gramian_traces'].mean():.2e}, fold={fold:.1f}×")
            for label in ["mean", "best"]:
                if label in lqr and "reduction_pct" in lqr[label]:
                    print(f"    LQR ({label}): {lqr[label]['reduction_pct']:.1f}% reduction")
    elif boran_results:
        print("\n  Boran: V not saved — re-run run_boran_pipeline.py first")

    summary_stats = build_stats_summary(miller_results, boran_results, all_tes1)
    print(f"\n  Global B fold-range (naive, single-donor-sensitive): "
          f"mean={summary_stats['mean_fold_range']:.1f}×  max={summary_stats['max_fold_range']:.1f}×")
    print(f"  Global B norm robust dispersion: IQR ratio={summary_stats['global_B_norm_iqr_ratio']:.2f}×  "
          f"MAD={summary_stats['global_B_norm_mad']:.3g}  median={summary_stats['global_B_norm_median']:.3g}")
    print(f"  Miller LQR (mean B): {summary_stats['miller_lqr_mean_pct']:.1f}% reduction")
    if not np.isnan(summary_stats["boran_lqr_mean_pct"]):
        print(f"  Boran LQR (mean B): {summary_stats['boran_lqr_mean_pct']:.1f}% reduction")
    print(f"  Results saved to results/tes1_comprehensive.npz")
    print("Done.")


if __name__ == "__main__":
    main()
