#!/usr/bin/env python3
"""DANDI 000673 (Daume/Kaminski/Rutishauser lab) single-unit Sternberg WM pipeline.

Third of three Rutishauser-lab Sternberg single-unit datasets (000469,
001187, 000673 — see run_000469_pipeline.py, run_001187_pipeline.py).
"Control of working memory by phase-amplitude coupling of human hippocampal
neurons" — ``intervals/trials`` schema (like 000469) but ``PicIDs_Encoding1``
field naming (like 001187). NOT an independent cohort from 001187: direct NWB
identity checks (native identifier, trial timestamps, accuracy, picture IDs)
confirm 31 shared patients across 37 recording sessions (corrected 2026-08-03 from a prior 16/19 undercount -- see provenance/dataset_overlap_report.json), released twice as
different curated views of the same patient-sessions (see
provenance/dataset_overlap_report.json and
provenance/canonical_primary_records.json, built by
scripts/audit_dataset_identity.py). 001187 is the canonical view for those
sessions, so this pipeline's own pooled statistics below (correct-vs-error
drift, Stouffer item-identity CTG) exclude the linked-duplicate sessions --
they still get a per-session artifact under this dataset's stats key, as a
sensitivity view, just not double-counted in cross-dataset pooling. Shared
numerical pipeline: src/spike_pipeline.py.

Outputs (per session):
  results/dandi000673_geometry_{key}.npz, results/dandi000673_ctg_{key}.npz,
  results/dandi000673_summary.json
Updates:
  results/all_statistics.json — "dandi000673_ctg" and "content_ctg_pooled" keys

Run:
    conda run -n wm_dynamics python scripts/run_000673_pipeline.py
"""
import sys, json, glob, warnings
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py
from spike_pipeline import (load_spike_times, build_psth, fit_pca_psth,
                            load_vs_load_ctg, item_identity_ctg, correct_error_drift,
                            pr_by_load, low_rate_unit_mask, MIN_SESSION_ACCURACY,
                            FrozenPSTHTransform)
from statistics import linear_mixed_effects_test, fdr_bh, stouffer_combine, stable_seed
from provenance import load_overlap_report, linked_duplicate_000673_session_keys, _json_safe

DATA_DIR = dataset_path("dandi_000673")
RESULTS = ROOT / "results"
PROVENANCE = ROOT / "provenance"
N_PC = 8
BIN_MS = 100
SMOOTH_MS = 200
MAINT_WIN = 2.3  # true delay (timestamps_Probe - timestamps_Maintenance) is only
                 # 2.53-2.82s per trial (jittered, verified against the NWB data
                 # directly); 3.0s ran past probe onset for every single trial,
                 # contaminating "maintenance" bins with the probe-locked response
                 # (worsened by the 200ms Gaussian smoothing tail). 2.3s keeps a
                 # >200ms margin below the global per-trial minimum delay.
MIN_UNITS = 15
CTG_STEP = 3
CTG_N_SPLITS = 5
# 50 -> 5000: at 50, the label-shuffle p-floor (c+1)/(n+1) ~= 0.0196 can never
# survive BH-FDR across ~32 sessions regardless of true effect size -- the same
# p-floor/N_PERM interaction audit item 1b flags for run_dpca_analysis.py.
CTG_N_PERM = 5000
N_JOBS = -1


def _process_session(fp: str):
    key = Path(fp).stem
    with threadpool_limits(limits=1):
        with h5py.File(fp, "r") as f:
            if "trials" not in f.get("intervals", {}) or "units" not in f:
                return None
            n_units_raw = int(f["units/id"].shape[0])
            if n_units_raw < MIN_UNITS:
                print(f"  SKIP {key} — only {n_units_raw} units", flush=True)
                return None
            trials = f["intervals/trials"]
            loads = trials["loads"][:].astype(int)
            t_maint = trials["timestamps_Maintenance"][:]
            pic_id_enc1 = trials["PicIDs_Encoding1"][:].astype(int)
            response_acc = trials["response_accuracy"][:].astype(bool)
            spike_lists = load_spike_times(f)

        n_trials = len(loads)
        print(f"\n{'='*55}\n  {key}: {n_units_raw} units, {n_trials} trials\n{'='*55}", flush=True)
        print(f"  Accuracy: {100*response_acc.mean():.1f}%", flush=True)

        if response_acc.mean() < MIN_SESSION_ACCURACY:
            print(f"  SKIP — accuracy {100*response_acc.mean():.1f}% < "
                  f"{100*MIN_SESSION_ACCURACY:.0f}% (Daume et al. 2024 QC floor)", flush=True)
            return None

        rate_mask = low_rate_unit_mask(spike_lists, t_maint, MAINT_WIN)
        n_units = int(rate_mask.sum())
        if n_units < n_units_raw:
            print(f"  Firing-rate QC: dropping {n_units_raw - n_units}/{n_units_raw} units", flush=True)
        if n_units < MIN_UNITS:
            print(f"  SKIP {key} — only {n_units} units survive firing-rate QC", flush=True)
            return None
        spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]

        times = np.arange(BIN_MS / 2, MAINT_WIN * 1000, BIN_MS) / 1000.0
        psth = build_psth(spike_lists, t_maint, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=MAINT_WIN)
        psth_z = FrozenPSTHTransform().fit_transform(psth)
        Z, V, var_ratio = fit_pca_psth(psth_z, n_comp=N_PC)

        pr_per_load = pr_by_load(psth_z, loads, rng=np.random.default_rng(0))

        rng = np.random.default_rng(stable_seed(key))
        ctg_res = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=N_PC, ctg_step=CTG_STEP,
                                   n_splits=CTG_N_SPLITS, n_perm=CTG_N_PERM, rng=rng)
        if ctg_res is not None:
            print(f"  CTG (load1v3): tau={ctg_res['tau_info']['tau']:.4f}, "
                  f"offdiag_effect={ctg_res['mean_offdiag_auc_minus_chance']:.4f}, "
                  f"p={ctg_res['p_value']:.4f}", flush=True)

        content_rng = np.random.default_rng(stable_seed(key + "_2"))
        content_res = item_identity_ctg(psth_z, loads, pic_id_enc1, target_load=1,
                                        n_components=N_PC, ctg_step=CTG_STEP, n_perm=CTG_N_PERM,
                                        rng=content_rng)
        if content_res is not None:
            print(f"  Content-CTG (load1, {content_res['n_classes']}-way): "
                  f"offdiag_effect={content_res['mean_offdiag_auc_minus_chance']:.4f}, "
                  f"p={content_res['p_value']:.4f}", flush=True)

        drift = correct_error_drift(Z, loads, times, MAINT_WIN)

        np.savez_compressed(
            RESULTS / f"dandi000673_geometry_{key}.npz",
            Z=Z.astype(np.float32), loads=loads, response_accuracy=response_acc,
            drift=drift, times=times, var_ratio=var_ratio, n_trials=n_trials, n_units=n_units,
        )
        if ctg_res is not None:
            np.savez_compressed(
                RESULTS / f"dandi000673_ctg_{key}.npz",
                auc_mat=ctg_res["auc_mat"], t_idx=ctg_res["t_idx"], null=ctg_res["null"],
                content_auc_mat=(content_res["auc_mat"] if content_res is not None else np.array([])),
            )

        row = {
            "n_trials": int(n_trials), "n_units": int(n_units),
            "accuracy": float(response_acc.mean()),
            "pr_per_load": pr_per_load,
            "tau": ctg_res["tau_info"]["tau"] if ctg_res else float("nan"),
            "mean_offdiag_auc": ctg_res["tau_info"]["mean_offdiag_auc"] if ctg_res else float("nan"),
            "mean_diag_auc": ctg_res["tau_info"]["mean_diag_auc"] if ctg_res else float("nan"),
            "offdiag_effect": ctg_res["mean_offdiag_auc_minus_chance"] if ctg_res else float("nan"),
            "p_offdiag_vs_chance": ctg_res["p_value"] if ctg_res else float("nan"),
            "content_ctg": (
                {"offdiag_effect": content_res["mean_offdiag_auc_minus_chance"],
                 "p_value": content_res["p_value"], "n_classes": content_res["n_classes"]}
                if content_res is not None else None
            ),
        }
    return key, row, drift.tolist(), response_acc.astype(int).tolist()


def main():
    files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*_ecephys*.nwb")))
    print(f"Found {len(files)} session files")

    results = Parallel(n_jobs=N_JOBS)(delayed(_process_session)(fp) for fp in files)

    overlap_report = load_overlap_report(PROVENANCE)
    linked_keys = (linked_duplicate_000673_session_keys(overlap_report)
                   if overlap_report is not None else set())
    if overlap_report is None:
        print("  WARNING: provenance/dataset_overlap_report.json not found -- "
              "cannot exclude 001187-linked duplicate sessions from pooling", flush=True)

    summary = {}
    pooled_drift, pooled_correct, pooled_key = [], [], []
    for r in results:
        if r is None:
            continue
        key, row, drift, correct = r
        summary[key] = row  # kept even for linked-duplicate sessions: a valid sensitivity view
        if key in linked_keys:
            continue  # already contributed via the canonical 001187 session; don't double-count
        pooled_drift.extend(drift)
        pooled_correct.extend(correct)
        pooled_key.extend([key] * len(drift))

    with open(RESULTS / "dandi000673_summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    p_vals = np.array([v["p_offdiag_vs_chance"] for v in summary.values()
                        if np.isfinite(v["p_offdiag_vs_chance"])])
    fdr = fdr_bh(p_vals, alpha=0.05) if len(p_vals) else {"n_reject": 0}
    print(f"\nFDR (BH), 000673 load1v3 CTG: {fdr['n_reject']}/{len(p_vals)} sessions survive q<0.05")

    drift_arr = np.array(pooled_drift)
    correct_arr = np.array(pooled_correct, dtype=float)
    key_arr = np.array(pooled_key)
    lme_drift = linear_mixed_effects_test(drift_arr, correct_arr, key_arr, n_perm=5000,
                                           rng=np.random.default_rng(0))
    print(f"Correct-vs-error drift (000673, pooled N={len(drift_arr)}, "
          f"{len(linked_keys)} 001187-linked sessions excluded): "
          f"beta={lme_drift['beta']:.4f}, p={lme_drift['p_value']:.4f}")

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["dandi000673_ctg"] = summary
    stats["dandi000673_correct_error_drift"] = {
        "beta": lme_drift["beta"], "p_value": lme_drift["p_value"], "n_trials": int(len(drift_arr)),
    }

    p_pool = []
    for k in ["dandi000469_ctg", "dandi001187_ctg", "dandi000673_ctg"]:
        for sess_key, v in stats.get(k, {}).items():
            if k == "dandi000673_ctg" and sess_key in linked_keys:
                continue  # linked duplicate of a 001187 session already in the pool
            if v.get("content_ctg") is not None:
                p_pool.append(v["content_ctg"]["p_value"])
    if len(p_pool) >= 2:
        pooled = stouffer_combine(np.array(p_pool))
        stats["content_ctg_pooled"] = {
            "n_sessions": len(p_pool), "z_combined": pooled["z_combined"], "p_combined": pooled["p_combined"],
        }
        print(f"\nPooled item-identity CTG (000469 + 001187 + 000673, N={len(p_pool)} sessions, "
              f"{len(linked_keys)} 001187-linked 000673 sessions excluded, Stouffer's method): "
              f"z={pooled['z_combined']:.3f}, p={pooled['p_combined']:.4e}")

    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/dandi000673_summary.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
