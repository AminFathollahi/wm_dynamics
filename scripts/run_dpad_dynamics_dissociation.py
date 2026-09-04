#!/usr/bin/env python3
"""DPAD dissociation of behaviorally-relevant vs. behaviorally-irrelevant
latent dynamics (Sani, Pesaran & Shanechi 2024, Nat Neurosci) on the DANDI
000574 (Boran) single-unit sessions, compared against this project's existing
unsupervised DMD/eigenmode dynamics characterization (v* = the fitted linear
operator A's dominant-by-modulus eigenvector, src.control.dominant_eigenmode).

Uses the official DPAD package (pip: DPAD, ShanechiLab/DPAD), which pins
tensorflow==2.15.1 and numpy==1.26.4 -- incompatible with this project's main
conda env (numpy 2.2.6 / torch 2.11+cu128). Runs in an isolated venv
(.venv_dpad/, gitignored) instead of touching the shared env's pins.

v* is unsupervised (derived purely from the fitted dynamics operator A, no
behavioral label). DPAD instead learns an nx-dimensional latent state and
explicitly prioritizes an n1-dimensional subspace (here n1=1, matching v*'s
rank) for predicting a behavioral variable -- set-size (memory load) 4-vs-8,
this project's existing primary WM contrast (matches the CTG set4v8 analyses
already run on this cohort). Correct/error was considered first but this
cohort's accuracy is near-ceiling (an established finding elsewhere in this
project), leaving too few error trials per session for a class-balanced
target. This tests whether the direction that best explains behaviorally
(load)-relevant dynamics coincides with the direction of fastest dynamical
growth, or whether they dissociate.

Reuses results/dandi000574_units_geometry_*.npz (already-PCA-reduced
per-trial latent trajectories + correct/error labels, written by
run_000574_units_pipeline.py) rather than re-deriving PCA from raw spikes.

Run:
    ./.venv_dpad/bin/python scripts/run_dpad_dynamics_dissociation.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # DPAD's TF 2.15 predates this GPU's CUDA 12.8/Blackwell support -- CPU only
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

from dynamics import ensemble_dmd
from control import dominant_eigenmode
from statistics import permutation_pvalue, paired_sign_flip_test, fdr_bh, stable_seed
from spike_pipeline import load_precomputed_session_geometry

from DPAD import DPADModel
from provenance import _json_safe

MIN_TRIALS = 25  # counted AFTER filtering to set_size in {4, 8}, so lower than a raw per-session floor
MIN_PER_CLASS = 10
N_NULL = 2000
DPAD_EPOCHS = 200
DPAD_PATIENCE = 12


def _dpad_relevant_direction(Y_trials: np.ndarray, Xp1_trials: np.ndarray) -> np.ndarray:
    """Direction in the shared PCA/observation space whose projection best
    explains DPAD's stage-1 (behaviorally-prioritized) latent trajectory --
    i.e. the least-squares row-space direction of DPAD's Cy readout, without
    needing to reach into DPADModel's internal weight matrices."""
    Y_flat = Y_trials.reshape(-1, Y_trials.shape[-1])
    x1_flat = Xp1_trials.reshape(-1)
    w, *_ = np.linalg.lstsq(Y_flat, x1_flat, rcond=None)
    return w / (np.linalg.norm(w) + 1e-12)


def _decode_score(Zp_trials: np.ndarray, n_tail: int = 5) -> np.ndarray:
    """Mean DPAD z-prediction over the last `n_tail` bins of each trial."""
    return Zp_trials[:, -n_tail:, 0].mean(axis=1)


def process_session(key: str, data: dict, rng: np.random.Generator) -> dict | None:
    # Correct/error is near-ceiling in this cohort (already an established
    # finding elsewhere in this project -- see Round-8 behavior-bounds work),
    # so error trials are too rare per session for a class-balanced DPAD
    # supervision target. Set-size (memory load) 4-vs-8 is this project's
    # existing primary WM contrast (matches the CTG set4v8 analyses already
    # run on this cohort) and is well-balanced within session.
    Z_full = np.asarray(data["Z"], dtype=np.float64)
    set_size = np.asarray(data["set_size"])
    keep = (set_size == 4) | (set_size == 8)
    Z = Z_full[keep]
    high_load = (set_size[keep] == 8)  # binary label: set-size 8 (True) vs set-size 4 (False)
    n_trials, T, d = Z.shape
    if n_trials < MIN_TRIALS:
        return None
    if min(high_load.sum(), (~high_load).sum()) < MIN_PER_CLASS:
        return None

    mu, sd = Z.mean(axis=(0, 1), keepdims=True), Z.std(axis=(0, 1), keepdims=True) + 1e-8
    Z = (Z - mu) / sd

    idx_train, idx_test = train_test_split(
        np.arange(n_trials), test_size=0.3, stratify=high_load, random_state=int(rng.integers(1 << 31))
    )
    Z_train, Z_test = Z[idx_train], Z[idx_test]
    y_train, y_test = high_load[idx_train], high_load[idx_test]

    dmd = ensemble_dmd(Z_train, r=d, rng=rng)
    mode = dominant_eigenmode(dmd["A"])
    v_star, growth_rate = mode.v_star, mode.rho
    X1_te = Z_test[:, :-1, :].reshape(-1, d).T
    X2_te = Z_test[:, 1:, :].reshape(-1, d).T
    pred_te = dmd["A"] @ X1_te
    ss_res = np.sum((X2_te - pred_te) ** 2)
    ss_tot = np.sum((X2_te - X2_te.mean(axis=1, keepdims=True)) ** 2)
    dmd_r2_test = float(1.0 - ss_res / (ss_tot + 1e-10))

    Y_fit = [Z_train[i].T for i in range(len(idx_train))]
    Z_fit = [np.tile(float(y_train[i]), (1, T)) for i in range(len(idx_train))]
    model = DPADModel()
    model.fit(Y_fit, Z_fit, nx=d, n1=1, epochs=DPAD_EPOCHS, verbose=False, save_logs=False,
              early_stopping_patience=DPAD_PATIENCE, create_val_from_training=True)

    Xp_train = np.stack([model.predict(Z_train[i])[2] for i in range(len(idx_train))])
    w_relevant = _dpad_relevant_direction(Z_train, Xp_train[:, :, 0])
    cos_align = float(abs(np.dot(w_relevant, v_star)))

    rand_dirs = rng.normal(size=(N_NULL, d))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_cos = np.abs(rand_dirs @ v_star)
    p_align = permutation_pvalue(rand_cos >= cos_align)

    Zp_test = np.stack([model.predict(Z_test[i])[0] for i in range(len(idx_test))])
    dpad_score = _decode_score(Zp_test)
    auc_dpad = roc_auc_score(y_test, dpad_score)

    vstar_score_train = (Z_train @ v_star)[:, -5:].mean(axis=1, keepdims=True)
    vstar_score_test = (Z_test @ v_star)[:, -5:].mean(axis=1, keepdims=True)
    clf_v = LogisticRegression().fit(vstar_score_train, y_train)
    auc_vstar = roc_auc_score(y_test, clf_v.predict_proba(vstar_score_test)[:, 1])

    pca_feat_train = Z_train[:, -5:, :].mean(axis=1)
    pca_feat_test = Z_test[:, -5:, :].mean(axis=1)
    clf_pca = LogisticRegression(max_iter=1000).fit(pca_feat_train, y_train)
    auc_pca = roc_auc_score(y_test, clf_pca.predict_proba(pca_feat_test)[:, 1])

    return {
        "n_trials": int(n_trials), "n_pc": int(d),
        "dmd_growth_rate": growth_rate, "dmd_r2_test": dmd_r2_test,
        "cos_align_dpad_vstar": cos_align, "p_align_vs_random": p_align,
        "auc_dpad_relevant": float(auc_dpad), "auc_vstar_baseline": float(auc_vstar),
        "auc_full_pca_baseline": float(auc_pca),
    }


def main():
    sessions = load_precomputed_session_geometry(RESULTS, "dandi000574_units_geometry_*.npz")
    print(f"Loaded {len(sessions)} precomputed DANDI 000574 sessions")

    per_session = {}
    for key, data in sessions.items():
        rng = np.random.default_rng(stable_seed(key))
        res = process_session(key, data, rng)
        if res is None:
            print(f"  {key}: SKIP (n_trials/class-balance below floor)")
            continue
        per_session[key] = res
        print(f"  {key}: n={res['n_trials']} DPAD-vstar cos={res['cos_align_dpad_vstar']:.3f} "
              f"(p={res['p_align_vs_random']:.4f}) AUC dpad={res['auc_dpad_relevant']:.3f} "
              f"vstar={res['auc_vstar_baseline']:.3f} pca={res['auc_full_pca_baseline']:.3f} "
              f"dmd_r2_test={res['dmd_r2_test']:.3f}", flush=True)

    if len(per_session) < 3:
        print("Too few qualifying sessions for a population-level comparison.")
        summary = {"per_session": per_session, "n_sessions": len(per_session)}
    else:
        keys = list(per_session.keys())
        cos_vals = np.array([per_session[k]["cos_align_dpad_vstar"] for k in keys])
        auc_dpad = np.array([per_session[k]["auc_dpad_relevant"] for k in keys])
        auc_vstar = np.array([per_session[k]["auc_vstar_baseline"] for k in keys])
        auc_pca = np.array([per_session[k]["auc_full_pca_baseline"] for k in keys])

        dpad_vs_vstar = paired_sign_flip_test(auc_vstar, auc_dpad, alternative="less")
        dpad_vs_pca = paired_sign_flip_test(auc_pca, auc_dpad, alternative="less")
        align_p = np.array([per_session[k]["p_align_vs_random"] for k in keys])
        align_fdr = fdr_bh(align_p)
        for k, q in zip(keys, align_fdr["q_values"]):
            per_session[k]["q_align_vs_random"] = float(q)

        print(f"\nAcross {len(keys)} sessions: mean cos(DPAD-relevant, v*)={cos_vals.mean():.3f} "
              f"(range {cos_vals.min():.3f}-{cos_vals.max():.3f}); "
              f"{align_fdr['n_reject']}/{len(keys)} sessions survive FDR (BH) q<0.05 on cos-vs-random")
        print(f"Paired AUC (DPAD-relevant vs v*-baseline): mean diff={dpad_vs_vstar['mean_diff']:.4f}, "
              f"p={dpad_vs_vstar['p_value']:.4f}")
        print(f"Paired AUC (DPAD-relevant vs full-PCA-baseline): mean diff={dpad_vs_pca['mean_diff']:.4f}, "
              f"p={dpad_vs_pca['p_value']:.4f}")

        summary = {
            "per_session": per_session, "n_sessions": len(keys),
            "mean_cos_align_dpad_vstar": float(cos_vals.mean()),
            "n_sessions_survive_fdr_align": int(align_fdr["n_reject"]),
            "dpad_vs_vstar_baseline": {"mean_diff": dpad_vs_vstar["mean_diff"], "p_value": dpad_vs_vstar["p_value"]},
            "dpad_vs_pca_baseline": {"mean_diff": dpad_vs_pca["mean_diff"], "p_value": dpad_vs_pca["p_value"]},
        }

    with open(RESULTS / "dpad_dynamics_dissociation.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)
    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["dpad_dynamics_dissociation"] = summary
    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/dpad_dynamics_dissociation.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
