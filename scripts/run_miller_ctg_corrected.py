#!/usr/bin/env python3
"""Corrected Miller CTG: PCA folded into CV, label-shuffle permutation null,
a dynamic-code negative control, a condition-independent/-dependent variance
decomposition (dPCA-lite), and a trial-count confound check for subject 'al'.

Supersedes the CTG numbers previously computed in notebooks/08_extended_analysis.ipynb,
which fed a fixed global PCA projection into CTG and assessed significance with a
one-sample t-test over non-independent off-diagonal cells.

Outputs: results/miller_ctg_corrected.json, updates results/all_statistics.json
  under "miller_ctg_corrected".

Compute note: the headline N_PERM=10000 label-permutation test is the
project's single most expensive per-subject computation (a prior sequential
run took 1h44m for 4 subjects). Each subject's test is independent (own
PCA/classifiers fit on its own trials, own stable_seed(subj)-derived RNG), so
subjects run in parallel via joblib; each worker pins BLAS to 1 thread
(threadpoolctl) to avoid oversubscribing the machine.

Run:
    conda run -n wm_dynamics python scripts/run_miller_ctg_corrected.py
"""
import sys, json, warnings
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import (ctg_label_permutation_null, ctg_nested_cv, temporal_stability_tau,
                      latent_trajectories, marginalize_condition_time)
from statistics import fdr_bh, stable_seed
from provenance import _json_safe

RESULTS = ROOT / "results"
SUBJECTS = ["al", "ca", "cc", "ug"]
N_JOBS = -1

MAINT_WINDOW = (0.3, 1.4)     # claimed stable/"square" period
TRANSIENT_WINDOW = (0.0, 0.3)  # early post-stimulus evoked transient — dynamic-code control
N_COMPONENTS = 8
CTG_N_POINTS = 15
N_SPLITS = 5
N_PERM = 10000   # headline test (Table 1, Fig 3A-D): resolves p<0.0001
N_SUBSAMPLE_REPEATS = 50   # trial-count confound: repeats of 50-trial subsampling


def window_t_idx(times: np.ndarray, window: tuple[float, float], n_points: int) -> np.ndarray:
    mask_idx = np.where((times >= window[0]) & (times <= window[1]))[0]
    step = max(1, len(mask_idx) // n_points)
    return mask_idx[::step][:n_points]


def run_ctg_on_window(epochs_ct: np.ndarray, y: np.ndarray, times: np.ndarray,
                       window: tuple[float, float], n_perm: int, rng) -> dict:
    t_idx = window_t_idx(times, window, CTG_N_POINTS)
    res = ctg_label_permutation_null(
        epochs_ct, y, t_idx, n_components=N_COMPONENTS,
        n_splits=N_SPLITS, n_perm=n_perm, rng=rng,
    )
    tau_info = temporal_stability_tau(res["auc_mat"])
    return {**res, **tau_info, "t_idx": t_idx}


def _process_subject(subj: str) -> tuple[str, tuple, dict]:
    with threadpool_limits(limits=1):
        print(f"\n{'='*55}\n  Miller CTG (corrected): {subj}\n{'='*55}", flush=True)
        dat = np.load(RESULTS / f"01_epochs_{subj}.npz", allow_pickle=True)
        epochs, times, task_id = dat["epochs"], dat["times"], dat["task_id"]  # (N,T,C)

        mask = (task_id == 0) | (task_id == 2)
        X = epochs[mask].transpose(0, 2, 1)     # (N, C, T) for ctg_* functions
        y = (task_id[mask] == 2).astype(int)
        print(f"  N={mask.sum()} trials ({(y==0).sum()} 0-back, {(y==1).sum()} 2-back), "
              f"{X.shape[1]} channels", flush=True)

        rng = np.random.default_rng(stable_seed(subj))
        maint_res = run_ctg_on_window(X, y, times, MAINT_WINDOW, N_PERM, rng)
        print(f"  Maintenance [{MAINT_WINDOW[0]},{MAINT_WINDOW[1]}]s: "
              f"tau={maint_res['tau']:.4f} (interpretable={maint_res['interpretable']}), "
              f"offdiag_effect={maint_res['mean_offdiag_auc_minus_chance']:.4f}, "
              f"p(label-shuffle)={maint_res['p_value']:.4f}", flush=True)

        rng2 = np.random.default_rng(stable_seed(subj + "_1"))
        trans_res = run_ctg_on_window(X, y, times, TRANSIENT_WINDOW, 50, rng2)
        print(f"  Negative control, transient [{TRANSIENT_WINDOW[0]},{TRANSIENT_WINDOW[1]}]s: "
              f"tau={trans_res['tau']:.4f}, "
              f"offdiag_effect={trans_res['mean_offdiag_auc_minus_chance']:.4f}, "
              f"p={trans_res['p_value']:.4f} (expect weaker/narrower than maintenance)", flush=True)

        Z, _, _ = latent_trajectories(epochs[mask], N_COMPONENTS)
        dpca = marginalize_condition_time(Z, y)
        print(f"  dPCA-lite: condition-independent variance={dpca['frac_condition_independent']:.3f}, "
              f"condition-dependent={dpca['frac_condition_dependent']:.3f} "
              f"(PR_ci={dpca['pr_condition_independent']:.2f}, PR_cd={dpca['pr_condition_dependent']:.2f})",
              flush=True)

        summary_row = {
            "n_trials": int(mask.sum()),
            "maintenance": {
                "tau": maint_res["tau"], "interpretable": maint_res["interpretable"],
                "mean_diag_auc": maint_res["mean_diag_auc"],
                "mean_offdiag_auc": maint_res["mean_offdiag_auc"],
                "offdiag_effect": maint_res["mean_offdiag_auc_minus_chance"],
                "p_value": maint_res["p_value"],
            },
            "negative_control_transient": {
                "tau": trans_res["tau"],
                "offdiag_effect": trans_res["mean_offdiag_auc_minus_chance"],
                "p_value": trans_res["p_value"],
            },
            "dpca_lite": dpca,
        }

        np.savez_compressed(
            RESULTS / f"miller_ctg_corrected_{subj}.npz",
            auc_mat=maint_res["auc_mat"],
            t_idx=maint_res["t_idx"],
            times_ctg=times[maint_res["t_idx"]],
            null=maint_res["null"],
            auc_mat_transient_control=trans_res["auc_mat"],
            t_idx_transient=trans_res["t_idx"],
        )
    return subj, (epochs, times, task_id), summary_row


def main():
    results = Parallel(n_jobs=N_JOBS)(delayed(_process_subject)(s) for s in SUBJECTS)
    # Preserve SUBJECTS order regardless of parallel completion order, so
    # downstream FDR / dict iteration order matches the original sequential
    # run exactly (BH-FDR is order-invariant given the same p-value set, but
    # matching order keeps summary.json diffable against prior runs).
    results_by_subj = {subj: (epochs_tuple, row) for subj, epochs_tuple, row in results}
    per_subject_epochs = {s: results_by_subj[s][0] for s in SUBJECTS}
    summary = {s: results_by_subj[s][1] for s in SUBJECTS}

    p_vals = np.array([v["maintenance"]["p_value"] for v in summary.values()])
    fdr = fdr_bh(p_vals, alpha=0.05)
    for subj, q in zip(summary.keys(), fdr["q_values"]):
        summary[subj]["maintenance"]["q_value_fdr"] = float(q)
    print(f"\n  FDR (BH) across {len(SUBJECTS)} Miller subjects: "
          f"{fdr['n_reject']}/{len(p_vals)} survive q<0.05")

    # Trial-count confound: does subsampling cc/ug/ca to al's N=50/condition
    # alone reproduce al's weaker/less-interpretable maintenance CTG?
    print(f"\n{'='*55}\n  Trial-count confound check ('al' has 50/condition, others 100)\n{'='*55}")
    n_al = summary["al"]["n_trials"] // 2   # per-condition count
    subsample_results = {}
    for subj in ["ca", "cc", "ug"]:
        epochs, times, task_id = per_subject_epochs[subj]
        mask = (task_id == 0) | (task_id == 2)
        X_full = epochs[mask].transpose(0, 2, 1)
        y_full = (task_id[mask] == 2).astype(int)
        t_idx = window_t_idx(times, MAINT_WINDOW, CTG_N_POINTS)

        rng = np.random.default_rng(stable_seed(subj + "_subsample"))
        offdiag_effects, taus = [], []
        idx0 = np.where(y_full == 0)[0]
        idx1 = np.where(y_full == 1)[0]
        for _ in range(N_SUBSAMPLE_REPEATS):
            sub_idx = np.concatenate([
                rng.choice(idx0, size=n_al, replace=False),
                rng.choice(idx1, size=n_al, replace=False),
            ])
            X_sub, y_sub = X_full[sub_idx], y_full[sub_idx]
            auc_mat = ctg_nested_cv(X_sub, y_sub, t_idx, n_components=N_COMPONENTS,
                                    n_splits=3, rng=rng)
            tau_info = temporal_stability_tau(auc_mat)
            offdiag_effects.append(tau_info["mean_offdiag_auc"] - 0.5)
            taus.append(tau_info["tau"])

        subsample_results[subj] = {
            "n_subsample_repeats": N_SUBSAMPLE_REPEATS,
            "offdiag_effect_mean": float(np.nanmean(offdiag_effects)),
            "offdiag_effect_std": float(np.nanstd(offdiag_effects)),
            "tau_mean": float(np.nanmean(taus)),
            "tau_std": float(np.nanstd(taus)),
            "full_n_offdiag_effect": summary[subj]["maintenance"]["offdiag_effect"],
            "full_n_tau": summary[subj]["maintenance"]["tau"],
        }
        print(f"  {subj}: full N offdiag_effect={summary[subj]['maintenance']['offdiag_effect']:.4f} -> "
              f"subsampled to N={n_al}/cond: {np.nanmean(offdiag_effects):.4f}±{np.nanstd(offdiag_effects):.4f} "
              f"(al full-N offdiag_effect={summary['al']['maintenance']['offdiag_effect']:.4f})")

    with open(RESULTS / "miller_ctg_corrected.json", "w") as f:
        json.dump(_json_safe({"per_subject": summary, "trial_count_confound": subsample_results}), f, indent=2, allow_nan=False)

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["miller_ctg_corrected"] = summary
    stats["miller_trial_count_confound"] = subsample_results
    # Overwrite the legacy "ctg" key (same schema consumed throughout
    # generate_paper_figures.py) with the corrected values, so the headline
    # Miller CTG numbers everywhere in the paper come from the nested-CV +
    # label-permutation pipeline rather than the old fixed-PCA + t-test one.
    stats["ctg"] = {
        subj: {
            "mean_diag": v["maintenance"]["mean_diag_auc"],
            "mean_offdiag": v["maintenance"]["mean_offdiag_auc"],
            "diag_minus_offdiag": v["maintenance"]["mean_diag_auc"] - v["maintenance"]["mean_offdiag_auc"],
            "p_offdiag_vs_chance": v["maintenance"]["p_value"],
            "q_value_fdr": v["maintenance"]["q_value_fdr"],
            "temporal_stability": v["maintenance"]["tau"],
            "tau_interpretable": v["maintenance"]["interpretable"],
        }
        for subj, v in summary.items()
    }
    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/miller_ctg_corrected.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
