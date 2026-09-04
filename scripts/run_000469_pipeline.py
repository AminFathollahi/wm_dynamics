#!/usr/bin/env python3
"""DANDI 000469 (Kyzar/Rutishauser lab) single-unit Sternberg WM pipeline.

One of three Rutishauser-lab Sternberg single-unit datasets used in this
project (000469, 001187, 000673 — see run_001187_pipeline.py and
run_000673_pipeline.py); kept as separate stats keys / result files per
dataset since they are the same lab/task lineage, not independent
replications. Shared numerical pipeline lives in src/spike_pipeline.py;
this script only handles 000469's NWB schema (``intervals/trials``,
``loadsEnc1_PicIDs``).

Outputs (per session):
  results/dandi000469_geometry_sub-{n}.npz — Z (N,T,8), loads, times, V, electrode_mni
  results/dandi000469_ctg_sub-{n}.npz      — CTG matrix (load1 vs load3)
  results/dandi000469_summary.json
  results/rutishauser_000469_content_context.json — paired within-subject
    context_tau vs content_tau comparison
Updates:
  results/all_statistics.json — "dandi000469_ctg", "rutishauser_000469_content_context" keys

Run:
    conda run -n wm_dynamics python scripts/run_000469_pipeline.py
"""
import sys, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py
from spike_pipeline import (load_spike_times, build_psth, fit_pca_psth,
                            load_vs_load_ctg, item_identity_ctg, correct_error_drift,
                            pr_by_load, low_rate_unit_mask, MIN_SESSION_ACCURACY,
                            FrozenPSTHTransform)
from geometry import temporal_stability_tau
from statistics import linear_mixed_effects_test, fdr_bh, stable_seed, paired_sign_flip_test
from provenance import _json_safe

DATA_DIR = dataset_path("dandi_000469")
RESULTS = ROOT / "results"
N_PC = 8
BIN_MS = 100
SMOOTH_MS = 200
MAINT_WIN = 2.3  # true delay (timestamps_Probe - timestamps_Maintenance) is only
                 # 2.53-2.82s per trial (jittered, verified against the NWB data
                 # directly); 3.0s ran past probe onset for every single trial,
                 # contaminating "maintenance" bins with the probe-locked response
                 # (worsened by the 200ms Gaussian smoothing tail). 2.3s keeps a
                 # >200ms margin below the global per-trial minimum delay.
ENC_WIN = 1.0
MIN_UNITS = 15
CTG_STEP = 3
CTG_N_SPLITS = 5
CTG_N_PERM = 100
PR_MIN_TRIALS_PER_GROUP = 8


def main():
    summary = {}
    times = np.arange(BIN_MS / 2, MAINT_WIN * 1000, BIN_MS) / 1000.0
    pooled_drift, pooled_correct, pooled_subj, pooled_n_units, pooled_load = [], [], [], [], []

    for sub_n in range(1, 22):
        subj = f"sub-{sub_n}"
        nwb_wm = DATA_DIR / subj / f"{subj}_ses-2_ecephys+image.nwb"
        if not nwb_wm.exists():
            print(f"  SKIP {subj} — no ses-2 file")
            continue

        print(f"\n{'='*55}\n  Processing {subj}\n{'='*55}")

        with h5py.File(str(nwb_wm), "r") as f:
            n_units_raw = int(f["units/id"].shape[0])
            if n_units_raw < MIN_UNITS:
                print(f"  SKIP — only {n_units_raw} units (< {MIN_UNITS})")
                continue

            spike_lists = load_spike_times(f)
            trials = f["intervals/trials"]
            loads = trials["loads"][:].astype(int)
            t_maint = trials["timestamps_Maintenance"][:]
            t_enc1 = trials["timestamps_Encoding1"][:]
            response_acc = trials["response_accuracy"][:].astype(bool)
            pic_id_enc1 = trials["loadsEnc1_PicIDs"][:].astype(int)
            n_trials = len(loads)

            elec = f["general/extracellular_ephys/electrodes"]
            electrode_mni = np.column_stack([elec["x"][:], elec["y"][:], elec["z"][:]]).astype(np.float32)
            unit_mni = electrode_mni[f["units/electrodes"][:]]

        print(f"  {n_units_raw} units, {n_trials} trials  "
              f"(load1={np.sum(loads==1)}, load2={np.sum(loads==2)}, load3={np.sum(loads==3)})")
        print(f"  Accuracy: {100*response_acc.mean():.1f}%")

        if response_acc.mean() < MIN_SESSION_ACCURACY:
            print(f"  SKIP — accuracy {100*response_acc.mean():.1f}% < "
                  f"{100*MIN_SESSION_ACCURACY:.0f}% (Daume et al. 2024 QC floor)")
            continue

        # Firing-rate QC floor (Daume et al. 2024): drop units whose mean rate
        # across the maintenance window falls below MIN_UNIT_FIRING_RATE_HZ,
        # then re-check the session still has enough units to analyze.
        rate_mask = low_rate_unit_mask(spike_lists, t_maint, MAINT_WIN)
        n_units = int(rate_mask.sum())
        if n_units < n_units_raw:
            print(f"  Firing-rate QC: dropping {n_units_raw - n_units}/{n_units_raw} "
                  f"units below {0.1} Hz")
        if n_units < MIN_UNITS:
            print(f"  SKIP — only {n_units} units survive firing-rate QC (< {MIN_UNITS})")
            continue
        spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
        unit_mni = unit_mni[rate_mask]

        psth = build_psth(spike_lists, t_maint, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=MAINT_WIN)
        psth_enc = build_psth(spike_lists, t_enc1, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=ENC_WIN)
        print(f"  PSTH: {psth.shape}, mean FR={psth.mean():.1f} Hz")

        maint_transform = FrozenPSTHTransform().fit(psth)
        psth_z = maint_transform.transform(psth)
        # Reuses the maintenance-period fit so the encoding-period control
        # arm is expressed in the same units, not fit fresh from itself.
        psth_enc_z = maint_transform.transform(psth_enc)

        Z, V, var_ratio = fit_pca_psth(psth_z, n_comp=N_PC)
        print(f"  Z: {Z.shape}, var in {N_PC} PCs: {100*var_ratio:.1f}%")

        pr_per_load = pr_by_load(psth_z, loads, rng=np.random.default_rng(0))
        pr_str = ", ".join(f"{k}: {v.get('pr_cv', float('nan')):.2f}" for k, v in pr_per_load.items())
        print(f"  PR (cv, native {n_units}-unit space): {{{pr_str}}}")

        ctg_rng = np.random.default_rng(stable_seed(subj))
        ctg_res = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=N_PC, ctg_step=CTG_STEP,
                                   n_splits=CTG_N_SPLITS, n_perm=CTG_N_PERM, rng=ctg_rng)
        if ctg_res is None:
            print("  SKIP CTG — too few trials")
            tau_info = {"tau": float("nan"), "interpretable": False,
                        "mean_diag_auc": float("nan"), "mean_offdiag_auc": float("nan")}
            enc_res = {"auc_mat": None, "mean_offdiag_auc_minus_chance": float("nan"), "p_value": float("nan")}
            enc_tau_info = dict(tau_info)
        else:
            tau_info = ctg_res["tau_info"]
            print(f"  τ={tau_info['tau']:.4f} (interpretable={tau_info['interpretable']}), "
                  f"offdiag_effect={ctg_res['mean_offdiag_auc_minus_chance']:.4f}, "
                  f"p(label-shuffle)={ctg_res['p_value']:.4f}")

            enc_rng = np.random.default_rng(stable_seed(subj + "_1"))
            enc_res = load_vs_load_ctg(psth_enc_z, loads, 1, 3, n_components=N_PC,
                                       ctg_step=max(1, CTG_STEP // 3), n_splits=CTG_N_SPLITS,
                                       n_perm=50, rng=enc_rng)
            if enc_res is None:
                enc_res = {"auc_mat": None, "mean_offdiag_auc_minus_chance": float("nan"), "p_value": float("nan")}
                enc_tau_info = dict(tau_info)
                enc_tau_info.update(tau=float("nan"))
            else:
                enc_tau_info = enc_res["tau_info"]
                print(f"  Negative control (encoding): τ={enc_tau_info['tau']:.4f}, "
                      f"offdiag_effect={enc_res['mean_offdiag_auc_minus_chance']:.4f}, "
                      f"p={enc_res['p_value']:.4f}")

        content_rng = np.random.default_rng(stable_seed(subj + "_2"))
        content_res = item_identity_ctg(psth_z, loads, pic_id_enc1, target_load=1,
                                        n_components=N_PC, ctg_step=CTG_STEP, n_perm=CTG_N_PERM,
                                        rng=content_rng)
        if content_res is not None:
            content_tau_info = temporal_stability_tau(content_res["auc_mat"])
            print(f"  Content-CTG (load1, {content_res['n_classes']}-way item identity): "
                  f"tau={content_tau_info['tau']:.4f} (interpretable={content_tau_info['interpretable']}), "
                  f"offdiag_effect={content_res['mean_offdiag_auc_minus_chance']:.4f}, "
                  f"p={content_res['p_value']:.4f}")
        else:
            content_tau_info = None
            print("  Content-CTG skipped — too few load-1 trials")

        drift = correct_error_drift(Z, loads, times, MAINT_WIN)
        pooled_drift.extend(drift.tolist())
        pooled_correct.extend(response_acc.astype(int).tolist())
        pooled_subj.extend([subj] * len(drift))
        pooled_n_units.extend([n_units] * len(drift))
        pooled_load.extend(loads.tolist())

        pr_correct_error = {"n_correct": int(response_acc.sum()), "n_error": int((~response_acc).sum())}
        if response_acc.sum() >= PR_MIN_TRIALS_PER_GROUP and (~response_acc).sum() >= PR_MIN_TRIALS_PER_GROUP:
            from geometry import spatiotemporal_participation_ratio
            pr_correct_error["pr_correct"] = spatiotemporal_participation_ratio(
                psth_z[response_acc], n_splits=2, rng=np.random.default_rng(stable_seed(subj + "_pr_correct")))
            pr_correct_error["pr_error"] = spatiotemporal_participation_ratio(
                psth_z[~response_acc], n_splits=2, rng=np.random.default_rng(stable_seed(subj + "_pr_error")))

        np.savez_compressed(
            RESULTS / f"dandi000469_geometry_{subj}.npz",
            Z=Z.astype(np.float32), V=V.astype(np.float32), electrode_mni=unit_mni,
            loads=loads, response_accuracy=response_acc, drift=drift, times=times,
            var_ratio=var_ratio, pr_per_load=pr_per_load, n_trials=n_trials, n_units=n_units,
            pic_id_enc1=pic_id_enc1,   # item identity labels, reused by the axis-rotation analysis
        )
        if ctg_res is not None:
            np.savez_compressed(
                RESULTS / f"dandi000469_ctg_{subj}.npz",
                auc_mat=ctg_res["auc_mat"], t_idx=ctg_res["t_idx"], times_ctg=times[ctg_res["t_idx"]],
                null=ctg_res["null"],
                auc_mat_encoding_control=(enc_res["auc_mat"] if enc_res["auc_mat"] is not None else np.array([])),
                content_auc_mat=(content_res["auc_mat"] if content_res is not None else np.array([])),
            )

        summary[subj] = {
            "n_trials": int(n_trials), "n_units": int(n_units),
            "accuracy": float(response_acc.mean()), "var_ratio": float(var_ratio),
            "pr_per_load": pr_per_load,
            "tau": tau_info["tau"], "tau_interpretable": tau_info["interpretable"],
            "mean_diag_auc": tau_info["mean_diag_auc"], "mean_offdiag_auc": tau_info["mean_offdiag_auc"],
            "offdiag_effect": ctg_res["mean_offdiag_auc_minus_chance"] if ctg_res else float("nan"),
            "p_offdiag_vs_chance": ctg_res["p_value"] if ctg_res else float("nan"),
            "negative_control": {
                "tau": enc_tau_info["tau"],
                "offdiag_effect": enc_res["mean_offdiag_auc_minus_chance"],
                "p_value": enc_res["p_value"],
            },
            "content_ctg": (
                {"offdiag_effect": content_res["mean_offdiag_auc_minus_chance"],
                 "p_value": content_res["p_value"], "n_classes": content_res["n_classes"],
                 "tau": content_tau_info["tau"], "interpretable": content_tau_info["interpretable"],
                 "mean_diag_auc": content_tau_info["mean_diag_auc"],
                 "mean_offdiag_auc": content_tau_info["mean_offdiag_auc"]}
                if content_res is not None else None
            ),
            "pr_correct_error": pr_correct_error,
        }
        print(f"  Saved dandi000469_geometry_{subj}.npz")

    with open(RESULTS / "dandi000469_summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    p_vals = np.array([v["p_offdiag_vs_chance"] for v in summary.values()
                        if np.isfinite(v["p_offdiag_vs_chance"])])
    fdr = fdr_bh(p_vals, alpha=0.05) if len(p_vals) else {"n_reject": 0}
    print(f"\n  FDR (BH): {fdr['n_reject']}/{len(p_vals)} subjects survive q<0.05")

    drift_arr = np.array(pooled_drift)
    correct_arr = np.array(pooled_correct, dtype=float)
    subj_arr = np.array(pooled_subj)
    lme_drift = linear_mixed_effects_test(drift_arr, correct_arr, subj_arr, n_perm=5000,
                                           rng=np.random.default_rng(0))
    print(f"\n  Correct-vs-error drift (pooled N={len(drift_arr)}): "
          f"beta={lme_drift['beta']:.4f}, p={lme_drift['p_value']:.4f}")

    # Q6 DML upgrade (WP-CAUSAL-4ii): causal-flavored association, drift -> correct,
    # controlling load, unit count, and subject identity (set-size analogue = load;
    # no per-trial RT is loaded for this dataset). Observational, not manipulated --
    # report an E-value alongside, not a causal claim.
    from causal import dml_partial_linear, e_value
    n_units_arr = np.array(pooled_n_units, dtype=float)
    load_arr = np.array(pooled_load, dtype=float)
    _, subj_idx469 = np.unique(subj_arr, return_inverse=True)
    X_conf469 = np.column_stack([load_arr, n_units_arr, subj_idx469.astype(float)])
    dml_res469 = dml_partial_linear(correct_arr, drift_arr, X_conf469, n_folds=5,
                                    rng=np.random.default_rng(0))
    dml_ev469 = e_value(dml_res469["theta"], dml_res469["se"],
                        y_sd=float(correct_arr.std()), d_sd=float(drift_arr.std()))
    print(f"  Q6 DML (drift -> correct, controlling load/unit-count/subject, "
          f"N={dml_res469['n']}): theta={dml_res469['theta']:.4f} "
          f"[{dml_res469['ci_lo']:.4f}, {dml_res469['ci_hi']:.4f}] p={dml_res469['p_value']:.4f}; "
          f"E-value={dml_ev469['e_value']:.2f} (CI-edge {dml_ev469['e_value_ci']:.2f})")

    # Within-subject content-vs-context dissociation, same sessions, same
    # dataset. Primary comparison is on the off-diagonal effect size
    # (AUC-0.5), not the tau ratio: per the paper's stated statistical
    # methodology, tau is an unstable ratio near a near-zero diagonal effect
    # and is only "interpretable" when diag AUC>=0.55 — a compound gate on
    # BOTH context and content tau leaves only 1/18 sessions (content
    # decoding, 5-way with ~9 trials/class after CV, rarely clears 0.55
    # individually), too few for a paired test. Off-diagonal effect size
    # needs no such gate (chance is 0.5 AUC regardless of n_classes for
    # macro-OVR AUC) and is well-defined for all 18 sessions, so it is the
    # primary paired statistic; tau is reported descriptively for the
    # smaller subset where both are individually interpretable.
    from scipy.stats import wilcoxon

    paired = {
        subj: {
            "context_offdiag_effect": v["offdiag_effect"],
            "content_offdiag_effect": v["content_ctg"]["offdiag_effect"],
            "context_tau": v["tau"], "content_tau": v["content_ctg"]["tau"],
            "context_tau_interpretable": v["tau_interpretable"],
            "content_tau_interpretable": v["content_ctg"]["interpretable"],
            "context_diag_auc": v["mean_diag_auc"], "content_diag_auc": v["content_ctg"]["mean_diag_auc"],
            "p_context": v["p_offdiag_vs_chance"], "p_content": v["content_ctg"]["p_value"],
        }
        for subj, v in summary.items()
        if v.get("content_ctg") is not None
    }
    tau_both_interpretable = {s: v for s, v in paired.items()
                              if v["context_tau_interpretable"] and v["content_tau_interpretable"]}

    if len(paired) >= 4:
        context_arr = np.array([v["context_offdiag_effect"] for v in paired.values()])
        content_arr = np.array([v["content_offdiag_effect"] for v in paired.values()])
        wstat, wp = wilcoxon(content_arr, context_arr, alternative="less")
        sign_flip = paired_sign_flip_test(content_arr, context_arr, n_perm=10000,
                                          alternative="less", rng=np.random.default_rng(2))
        content_context = {
            "per_subject": paired,
            "n_paired_subjects": len(paired),
            "primary_statistic": "offdiag_effect",
            "wilcoxon_statistic": float(wstat), "wilcoxon_p": float(wp),
            "paired_gap_mean": sign_flip["mean_diff"], "paired_gap_ci": [sign_flip["ci_lower"], sign_flip["ci_upper"]],
            "paired_gap_permutation_p": sign_flip["p_value"],
            "n_tau_both_interpretable": len(tau_both_interpretable),
            "tau_both_interpretable_sessions": tau_both_interpretable,
        }
        print(f"\n  Within-subject content vs context, off-diag effect (N={len(paired)} sessions): "
              f"mean gap={sign_flip['mean_diff']:.4f} [{sign_flip['ci_lower']:.4f}, {sign_flip['ci_upper']:.4f}], "
              f"Wilcoxon p={wp:.4f}, sign-flip p={sign_flip['p_value']:.4f}")
        print(f"  (tau interpretable in both context and content: {len(tau_both_interpretable)}/{len(paired)} sessions)")
    else:
        content_context = {"per_subject": paired, "n_paired_subjects": len(paired),
                            "note": "fewer than 4 sessions with a content CTG result at all"}
        print(f"\n  Within-subject content vs context: only {len(paired)} qualifying sessions, skipping paired test")

    with open(RESULTS / "rutishauser_000469_content_context.json", "w") as f:
        json.dump(_json_safe(content_context), f, indent=2, allow_nan=False)

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["dandi000469_ctg"] = summary
    stats["dandi000469_correct_error_drift"] = {
        "beta": lme_drift["beta"], "p_value": lme_drift["p_value"], "n_trials": int(len(drift_arr)),
        "dml_causal_flavored": {
            "theta": dml_res469["theta"], "ci_lo": dml_res469["ci_lo"], "ci_hi": dml_res469["ci_hi"],
            "p_value": dml_res469["p_value"], "e_value": dml_ev469["e_value"],
            "e_value_ci": dml_ev469["e_value_ci"], "n": dml_res469["n"],
            "confounders": ["load", "n_units", "subject_id"],
            "label": "causal-flavored association on observational data (drift is "
                     "measured, not manipulated -- no positivity for a genuine effect)",
        },
    }
    stats["rutishauser_000469_content_context"] = content_context
    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)

    print("\n" + "="*55 + "\nDANDI 000469 PIPELINE COMPLETE\n" + "="*55)
    for sub, v in summary.items():
        print(f"  {sub}: N={v['n_trials']}, U={v['n_units']}, τ={v['tau']:.4f}, "
              f"offdiag_effect={v['offdiag_effect']:.4f}, p={v['p_offdiag_vs_chance']:.3e}")


if __name__ == "__main__":
    main()
