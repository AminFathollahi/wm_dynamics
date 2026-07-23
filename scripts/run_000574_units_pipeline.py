#!/usr/bin/env python3
"""Boran (DANDI 000574) single-unit CTG — within-subject cross-modality comparison.

Boran's NWB files carry co-located microwire single units (hippocampus,
amygdala) alongside the macro iEEG contacts already used in
run_boran_pipeline.py, from the SAME subjects and SAME Sternberg trials.
This is the one dataset in the project where band-power (HGP, run via the
iEEG pipeline) and single-unit spiking geometry can be compared within
subject rather than across independent cohorts, directly addressing the
limitation, noted in the Methods, that the Rutishauser-lineage single-unit
datasets are a different task/cohort from Miller/Boran.

Each session (subjects have 1-4 sessions, not necessarily unit-matched across
sessions) is treated as its own analysis unit, exactly as for 000469/001187.
Set-size (4 vs 8) CTG is the same load/context contrast as the iEEG analysis;
content-CTG is not run here (Boran's set_letters/item identity is "not
available" in the public NWB release — see run_boran_pipeline.py).

Outputs: results/dandi000574_units_geometry_{key}.npz,
  results/dandi000574_units_ctg_{key}.npz, results/dandi000574_units_summary.json
Updates: all_statistics.json — "dandi000574_units_ctg" key

Run:
    conda run -n wm_dynamics python scripts/run_000574_units_pipeline.py
"""
import sys, json, glob, warnings
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import h5py
from spike_pipeline import (load_spike_times, build_psth, fit_pca_psth,
                            load_vs_load_ctg, correct_error_drift, pr_by_load)
from statistics import linear_mixed_effects_test, fdr_bh, stable_seed

DATA_DIR = Path("/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/000574")
RESULTS = ROOT / "results"
SUBJECTS = [f"sub-0{i}" for i in range(1, 10)]
N_PC = 8
BIN_MS = 100
SMOOTH_MS = 200
MAINT_ONSET_S = 3.0   # trial start -> maintenance onset (same convention as run_boran_pipeline.py)
MAINT_WIN = 3.0
MIN_UNITS = 8          # microwire yield is lower than Rutishauser's dedicated bundles
CTG_STEP = 3
CTG_N_SPLITS = 4
# 50 -> 5000: at 50, the label-shuffle p-floor (c+1)/(n+1) ~= 0.0196 can never
# survive BH-FDR across ~26 sessions (needs ~(1/26)*0.05 ~= 0.0019 for even the
# top-ranked session) regardless of true effect size -- the same
# p-floor/N_PERM interaction audit item 1b flags for run_dpca_analysis.py.
CTG_N_PERM = 5000
N_JOBS = -1


def _process_session(subj: str, fp: Path):
    key = fp.stem
    with threadpool_limits(limits=1):
        with h5py.File(str(fp), "r") as f:
            if "units" not in f or f["units/id"].shape[0] < MIN_UNITS:
                return None
            n_units = int(f["units/id"].shape[0])
            trials = f["intervals/trials"]
            start_time = trials["start_time"][:]
            set_size = trials["set_size"][:].astype(int)
            correct = trials["correct"][:].astype(bool)
            artifact = trials["artifact"][:].astype(bool)
            spike_lists = load_spike_times(f)

        good = ~artifact
        start_time, set_size, correct = start_time[good], set_size[good], correct[good]
        maint_onsets = start_time + MAINT_ONSET_S
        n_trials = len(maint_onsets)
        if n_trials < 20:
            return None

        print(f"\n{'='*55}\n  {key}: {n_units} units, {n_trials} trials\n{'='*55}", flush=True)
        psth = build_psth(spike_lists, maint_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                          window_s=MAINT_WIN)
        mu = psth.mean(axis=0, keepdims=True)
        sd = psth.std(axis=0, keepdims=True) + 1e-8
        psth_z = (psth - mu) / sd
        # Rutishauser cohorts (000469/001187/000673, MIN_UNITS=15) fit a fixed
        # N_PC=8 from >=15 units (reduction ratio >=1.875x); Boran's lower
        # microwire yield (MIN_UNITS=8) means n_comp is capped below n_units
        # (never true full rank) but can still be a much weaker reduction
        # (e.g. 8 units -> 7 PCs, ratio 1.14x). Both are recorded so downstream
        # aggregation does not silently pool a near-unreduced estimate with a
        # genuinely-reduced one (audit item 1c-iii).
        n_pc_used = min(N_PC, n_units - 1)
        reduction_ratio = n_units / n_pc_used
        low_reduction = bool(reduction_ratio < 1.5)   # < Rutishauser's 15/8 = 1.875x
        Z, V, var_ratio = fit_pca_psth(psth_z, n_comp=n_pc_used)

        pr_per_set = pr_by_load(psth_z, set_size, load_values=(4, 6, 8), min_trials=5,
                                rng=np.random.default_rng(0))

        rng = np.random.default_rng(stable_seed(key))
        ctg_res = load_vs_load_ctg(psth_z, set_size, 4, 8, n_components=n_pc_used,
                                   ctg_step=CTG_STEP, n_splits=CTG_N_SPLITS, n_perm=CTG_N_PERM, rng=rng)
        if ctg_res is not None:
            print(f"  CTG (set4v8, single-unit): tau={ctg_res['tau_info']['tau']:.4f}, "
                  f"offdiag_effect={ctg_res['mean_offdiag_auc_minus_chance']:.4f}, "
                  f"p={ctg_res['p_value']:.4f}", flush=True)

        times = np.arange(BIN_MS / 2, MAINT_WIN * 1000, BIN_MS) / 1000.0
        drift = correct_error_drift(Z, set_size, times, MAINT_WIN)

        np.savez_compressed(
            RESULTS / f"dandi000574_units_geometry_{key}.npz",
            Z=Z.astype(np.float32), set_size=set_size, correct=correct, drift=drift,
            times=times, var_ratio=var_ratio, n_trials=n_trials, n_units=n_units,
            n_pc_used=n_pc_used, reduction_ratio=reduction_ratio, low_reduction=low_reduction,
        )
        if ctg_res is not None:
            np.savez_compressed(
                RESULTS / f"dandi000574_units_ctg_{key}.npz",
                auc_mat=ctg_res["auc_mat"], t_idx=ctg_res["t_idx"], null=ctg_res["null"],
            )

        row = {
            "subject": subj, "n_trials": int(n_trials), "n_units": int(n_units),
            "n_pc_used": int(n_pc_used), "reduction_ratio": float(reduction_ratio),
            "low_reduction": low_reduction,
            "accuracy": float(correct.mean()), "pr_per_set": pr_per_set,
            "tau": ctg_res["tau_info"]["tau"] if ctg_res else float("nan"),
            "mean_offdiag_auc": ctg_res["tau_info"]["mean_offdiag_auc"] if ctg_res else float("nan"),
            "mean_diag_auc": ctg_res["tau_info"]["mean_diag_auc"] if ctg_res else float("nan"),
            "offdiag_effect": ctg_res["mean_offdiag_auc_minus_chance"] if ctg_res else float("nan"),
            "p_offdiag_vs_chance": ctg_res["p_value"] if ctg_res else float("nan"),
        }
    return key, row, drift.tolist(), correct.astype(int).tolist()


def main():
    tasks = [(subj, fp) for subj in SUBJECTS for fp in sorted((DATA_DIR / subj).glob("*.nwb"))]
    results = Parallel(n_jobs=N_JOBS)(delayed(_process_session)(subj, fp) for subj, fp in tasks)

    summary = {}
    pooled_drift, pooled_correct, pooled_key = [], [], []
    for r in results:
        if r is None:
            continue
        key, row, drift, correct = r
        summary[key] = row
        pooled_drift.extend(drift)
        pooled_correct.extend(correct)
        pooled_key.extend([key] * len(drift))

    with open(RESULTS / "dandi000574_units_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    p_vals = np.array([v["p_offdiag_vs_chance"] for v in summary.values()
                        if np.isfinite(v["p_offdiag_vs_chance"])])
    fdr = fdr_bh(p_vals, alpha=0.05) if len(p_vals) else {"n_reject": 0}
    print(f"\nFDR (BH), Boran single-unit set4v8 CTG: {fdr['n_reject']}/{len(p_vals)} sessions survive q<0.05")

    n_low_reduction = sum(1 for v in summary.values() if v.get("low_reduction"))
    print(f"Dimensionality-reduction caveat: {n_low_reduction}/{len(summary)} sessions have "
          f"reduction ratio < 1.5x (vs Rutishauser's 15units/8PC=1.875x) — weakly reduced, "
          "not on the same comparability footing (audit item 1c-iii)")

    if pooled_drift:
        drift_arr = np.array(pooled_drift)
        correct_arr = np.array(pooled_correct, dtype=float)
        key_arr = np.array(pooled_key)
        lme_drift = linear_mixed_effects_test(drift_arr, correct_arr, key_arr, n_perm=5000,
                                               rng=np.random.default_rng(0))
        print(f"Correct-vs-error drift (Boran units, pooled N={len(drift_arr)}): "
              f"beta={lme_drift['beta']:.4f}, p={lme_drift['p_value']:.4f}")
    else:
        lme_drift = {"beta": float("nan"), "p_value": float("nan")}

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["dandi000574_units_ctg"] = summary
    stats["dandi000574_units_correct_error_drift"] = {
        "beta": lme_drift["beta"], "p_value": lme_drift["p_value"], "n_trials": int(len(pooled_drift)),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/dandi000574_units_summary.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
