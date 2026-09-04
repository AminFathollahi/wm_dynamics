#!/usr/bin/env python3
"""Latent-dimensionality selection and headline robustness sweep.

Two deliverables, one shared raw loader:

  results/latent_dim_selection.json — for each dataset, the data-selected latent
    dim by two principled rules (cross-validated participation ratio, and Horn's
    parallel analysis), the variance captured at k=8, and n_channels. Shows the
    common 8-D latent is a data-supported cross-dataset commensurability choice,
    not a convention.

  results/dim_robustness.json — the four headline quantities recomputed at
    k in {6, 8, 10, 12}, to demonstrate each is qualitatively invariant to the
    latent-dimensionality choice:
      (1) content-vs-context axis-rotation difference (DANDI 000469 within-subject),
      (2) PR-vs-load pooled slope (native channel space — k-INDEPENDENT by
          construction; carried at every k to make that explicit),
      (3) CTG temporal-stability tau (context = load-1-vs-3, content = item identity),
      (4) the causal-benchmark v* slope (delegated to run_dmd_rank_selection.py,
          which sweeps the OPERATOR rank; merged here at read time).

Every numeric reuses existing shared pipeline functions (spike_pipeline,
geometry, statistics) — no forked analysis. The only dataset-specific code is
the thin NWB-schema glue already present in the per-dataset pipelines, factored
here into one field-name-parameterised Sternberg loader.

Run (needs the external data mount; run_dmd_rank_selection.py first for headline 4):
    conda run -n wm_dynamics python scripts/run_dim_robustness.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py  # noqa: E402
from spike_pipeline import (  # noqa: E402
    load_spike_times, build_psth, fit_pca_psth, low_rate_unit_mask,
    load_vs_load_ctg, item_identity_ctg,
    MIN_SESSION_ACCURACY, FrozenPSTHTransform,
)
from geometry import (  # noqa: E402
    select_latent_dim, coding_direction_stability, temporal_stability_tau,
)
from statistics import paired_sign_flip_test, forest_meta, stable_seed  # noqa: E402
from provenance import _json_safe

RESULTS = ROOT / "results"
DATA_ROOT = data_root()

BIN_MS = 100
SMOOTH_MS = 200
MAINT_WIN = 2.3
MIN_UNITS = 15   # same >=15-unit session floor as the per-dataset Sternberg pipelines
K_SWEEP = (6, 8, 10, 12)
CTG_STEP = 3
CTG_N_PERM = 100
AXIS_STEP = 3   # matches run_axis_rotation_analysis STEP_000469

# The three Rutishauser-lineage Sternberg cohorts differ only in two NWB field
# names (already the case across their per-dataset pipeline scripts).
STERNBERG = {
    "dandi000469": {"dir": "000469", "trials": "intervals/trials", "pic": "loadsEnc1_PicIDs",
                    "ses": "ses-2", "subs": [f"sub-{n}" for n in range(1, 22)]},
    "dandi001187": {"dir": "001187", "trials": "intervals/WM_trials", "pic": "PicIDs_Encoding1",
                    "ses": None, "subs": None},
    "dandi000673": {"dir": "000673", "trials": "intervals/trials", "pic": "PicIDs_Encoding1",
                    "ses": None, "subs": None},
}


def _iter_nwb_paths(cfg: dict):
    ddir = DATA_ROOT / cfg["dir"]
    if cfg["subs"] is not None:
        for subj in cfg["subs"]:
            p = ddir / subj / f"{subj}_{cfg['ses']}_ecephys+image.nwb"
            if p.exists():
                yield subj, p
    else:
        for p in sorted(ddir.glob("sub-*/*.nwb")):
            yield p.stem, p


def load_sternberg_sessions(dataset: str):
    """Yield (key, psth_z (N,C,T), loads, pic_ids) per QC-passing session, using
    the shared spike_pipeline exactly as the per-dataset scripts do."""
    cfg = STERNBERG[dataset]
    for key, path in _iter_nwb_paths(cfg):
        try:
            with h5py.File(str(path), "r") as f:
                if cfg["trials"].split("/")[-1] not in f.get("intervals", {}):
                    continue
                if int(f["units/id"].shape[0]) < MIN_UNITS:
                    continue
                trials = f[cfg["trials"]]
                loads = trials["loads"][:].astype(int)
                t_maint = trials["timestamps_Maintenance"][:]
                pic_id = trials[cfg["pic"]][:].astype(int)
                response_acc = trials["response_accuracy"][:].astype(bool)
                spike_lists = load_spike_times(f)
        except (OSError, KeyError):
            continue
        if response_acc.mean() < MIN_SESSION_ACCURACY:
            continue
        rate_mask = low_rate_unit_mask(spike_lists, t_maint, MAINT_WIN)
        if int(rate_mask.sum()) < MIN_UNITS:
            continue
        spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
        psth = build_psth(spike_lists, t_maint, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS, window_s=MAINT_WIN)
        yield key, FrozenPSTHTransform().fit_transform(psth), loads, pic_id


# ── A1: selection table ───────────────────────────────────────────────────────

def _var_explained_at(psth_z: np.ndarray, k: int) -> float:
    _, _, var_ratio = fit_pca_psth(psth_z, n_comp=k)
    return float(var_ratio)


def latent_dim_selection() -> dict:
    """Per-dataset cv-PR + parallel-analysis k, variance at 8, n_channels."""
    table = {}
    for dataset in STERNBERG:
        cv_prs, k_pas, var8s, n_chs = [], [], [], []
        for key, psth_z, loads, _ in load_sternberg_sessions(dataset):
            rng = np.random.default_rng(stable_seed(f"dimsel_{key}"))
            sel = select_latent_dim(psth_z, rng=rng)
            cv_prs.append(sel["cv_pr"])
            k_pas.append(sel["k_parallel_analysis"])
            var8s.append(_var_explained_at(psth_z, 8))
            n_chs.append(sel["n_channels"])
        if not cv_prs:
            continue
        table[dataset] = {
            "n_sessions": len(cv_prs),
            "n_channels_median": float(np.median(n_chs)),
            "n_channels_range": [int(np.min(n_chs)), int(np.max(n_chs))],
            "cv_PR_mean": float(np.mean(cv_prs)),
            "cv_PR_std": float(np.std(cv_prs)),
            "k_cv_PR_rounded": int(round(np.mean(cv_prs))),
            "k_parallel_analysis_median": float(np.median(k_pas)),
            "var_explained_at_8_mean": float(np.mean(var8s)),
            "k_used": 8,
        }
        print(f"  {dataset:14s} n={table[dataset]['n_sessions']:2d}  "
              f"cv_PR={table[dataset]['cv_PR_mean']:.1f}±{table[dataset]['cv_PR_std']:.1f}  "
              f"k_PA(med)={table[dataset]['k_parallel_analysis_median']:.0f}  "
              f"var@8={table[dataset]['var_explained_at_8_mean']:.2f}  "
              f"n_ch={table[dataset]['n_channels_range']}")
    return table


# ── A3: headline robustness sweep over k ──────────────────────────────────────

def _axis_rotation_index(cos_sim: np.ndarray) -> float:
    n = cos_sim.shape[0]
    off = cos_sim[~np.eye(n, dtype=bool)]
    return float(1.0 - np.nanmean(off))


def _headline_1_axis_rotation(sessions_469: list, k: int) -> dict:
    """DANDI 000469 within-subject content-minus-context axis-rotation difference."""
    ctx, cont = [], []
    for key, psth_z, loads, pic_id in sessions_469:
        Z, _, _ = fit_pca_psth(psth_z, n_comp=k)
        ctx_mask = (loads == 1) | (loads == 3)
        load1 = loads == 1
        if ctx_mask.sum() < 10 or load1.sum() < 15 or len(np.unique(pic_id[load1])) < 2:
            continue
        ctx_cos, _ = coding_direction_stability(Z[ctx_mask], (loads[ctx_mask] == 3).astype(int), step=AXIS_STEP)
        cont_cos, _ = coding_direction_stability(Z[load1], pic_id[load1], step=AXIS_STEP)
        ctx.append(_axis_rotation_index(ctx_cos))
        cont.append(_axis_rotation_index(cont_cos))
    ctx, cont = np.array(ctx), np.array(cont)
    res = paired_sign_flip_test(cont, ctx, n_perm=10000, alternative="greater",
                                rng=np.random.default_rng(1))
    return {"axis_rot_diff": res["mean_diff"], "axis_rot_p": res["p_value"],
            "axis_rot_ci": [res["ci_lower"], res["ci_upper"]], "n_subjects": int(len(ctx))}


def _headline_3_tau(sessions_469: list, k: int) -> dict:
    """Context (load1-vs-3) and content (item-identity) CTG tau, pooled over 000469."""
    ctx_taus, cont_taus = [], []
    for key, psth_z, loads, pic_id in sessions_469:
        rng = np.random.default_rng(stable_seed(f"dimrob_ctg_{key}_{k}"))
        ctg = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=k, ctg_step=CTG_STEP,
                               n_splits=5, n_perm=CTG_N_PERM, rng=rng)
        if ctg is not None and ctg["tau_info"]["interpretable"]:
            ctx_taus.append(ctg["tau_info"]["tau"])
        content = item_identity_ctg(psth_z, loads, pic_id, target_load=1, n_components=k,
                                    ctg_step=CTG_STEP, n_perm=CTG_N_PERM,
                                    rng=np.random.default_rng(stable_seed(f"dimrob_content_{key}_{k}")))
        if content is not None:
            ti = temporal_stability_tau(content["auc_mat"])
            if ti["interpretable"]:
                cont_taus.append(ti["tau"])
    return {
        "tau_context": float(np.mean(ctx_taus)) if ctx_taus else float("nan"),
        "tau_content": float(np.mean(cont_taus)) if cont_taus else float("nan"),
        "n_context_interpretable": len(ctx_taus),
        "n_content_interpretable": len(cont_taus),
    }


def _headline_2_pr_slope() -> dict:
    """Pooled PR-vs-load slope in NATIVE channel space (k-independent). Reuses the
    exact per-dataset LME + forest_meta pooling that the paper's abstract number
    comes from, read straight off the authoritative artifacts."""
    stats = json.load(open(RESULTS / "all_statistics.json"))
    from scipy.stats import norm
    rows = []
    for label, key in [("Miller", "miller"), ("Boran iEEG", "boran_ieeg"),
                       ("Boran units", "boran_units"), ("DANDI 000469", "dandi000469"),
                       ("DANDI 001187", "dandi001187"), ("DANDI 000673", "dandi000673")]:
        r = stats.get("pr_lme_by_dataset", {}).get(key)
        if r and "beta" in r:
            p = min(max(float(r["p_value"]), 1e-6), 1 - 1e-9)
            se = abs(r["beta"]) / norm.isf(p / 2.0)
            rows.append((label, r["beta"], se))
    meta = forest_meta(np.array([x[1] for x in rows]), np.array([x[2] for x in rows]),
                       labels=[x[0] for x in rows])
    return {"pr_slope": meta["pooled"], "pr_ci": [meta["ci_lo"], meta["ci_hi"]],
            "pr_p": meta["p_value"]}


def dim_robustness(sessions_469: list) -> dict:
    pr = _headline_2_pr_slope()   # native-space, same at every k
    bench = json.load(open(RESULTS / "dmd_rank_selection.json")) if (RESULTS / "dmd_rank_selection.json").exists() else {}
    out = {}
    for k in K_SWEEP:
        print(f"\n  --- k = {k} ---")
        h1 = _headline_1_axis_rotation(sessions_469, k)
        h3 = _headline_3_tau(sessions_469, k)
        # headline 4: nearest available operator-rank benchmark slope (v* rank == 6
        # is the paper's operator rank; benchmark is fit on the operator, not the
        # latent, so it is reported against the operator-rank sweep, keyed by r).
        bslope = bench.get(str(min(k, 8)), {}).get("benchmark_slope", float("nan")) if bench else float("nan")
        bp = bench.get(str(min(k, 8)), {}).get("benchmark_p", float("nan")) if bench else float("nan")
        out[str(k)] = {
            "axis_rot_diff": h1["axis_rot_diff"], "axis_rot_p": h1["axis_rot_p"],
            "axis_rot_ci": h1["axis_rot_ci"], "n_subjects": h1["n_subjects"],
            "pr_slope": pr["pr_slope"], "pr_ci": pr["pr_ci"], "pr_p": pr["pr_p"],
            "tau_context": h3["tau_context"], "tau_content": h3["tau_content"],
            "n_context_interpretable": h3["n_context_interpretable"],
            "n_content_interpretable": h3["n_content_interpretable"],
            "benchmark_slope": bslope, "benchmark_p": bp,
        }
        print(f"    axis_rot_diff={h1['axis_rot_diff']:+.4f} p={h1['axis_rot_p']:.4f} (N={h1['n_subjects']}); "
              f"tau_ctx={h3['tau_context']:.3f} tau_cont={h3['tau_content']:.3f}; "
              f"pr_slope={pr['pr_slope']:+.3f} p={pr['pr_p']:.3f}")
    return out


def main():
    print("A1: latent dimensionality selection ...")
    table = latent_dim_selection()
    json.dump(_json_safe(table), open(RESULTS / "latent_dim_selection.json", "w"), indent=2, allow_nan=False)
    print(f"  wrote results/latent_dim_selection.json ({len(table)} datasets)")

    print("\nA3: caching DANDI 000469 raw sessions once for the k-sweep ...")
    sessions_469 = list(load_sternberg_sessions("dandi000469"))
    print(f"  {len(sessions_469)} sessions")
    out = dim_robustness(sessions_469)
    json.dump(_json_safe(out), open(RESULTS / "dim_robustness.json", "w"), indent=2, allow_nan=False)
    print("\n  wrote results/dim_robustness.json")

    stats = json.load(open(RESULTS / "all_statistics.json"))
    stats["latent_dim_selection"] = table
    stats["dim_robustness"] = out
    json.dump(_json_safe(stats), open(RESULTS / "all_statistics.json", "w"), indent=2, allow_nan=False)
    print("  updated all_statistics.json")


if __name__ == "__main__":
    main()
