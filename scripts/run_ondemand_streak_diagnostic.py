#!/usr/bin/env python3
"""Diagnostic (R5 personalization note): WHY does the on-demand trigger fail
for some cohorts and not others?

The decoder-gated trigger (STEP A2) engages only when the trained decoder
flags drift; between triggers the plant runs open-loop. For a cohort whose
decoder rarely flags large excursions (its decision boundary happens to
extrapolate to "on-target" far off the real-data manifold it was trained
on), the plant can run open-loop for long consecutive stretches, letting
even a mild instability (rho_open only slightly above 1) compound. This
script quantifies that per cohort, generically (no cohort singled out by
name): mean/max consecutive un-triggered steps per trial, alongside the
already-stored decoder_cv_acc and ondemand drift_reduction, so the claim
that reliability is cohort-specific -- not predicted by decoder validity
alone -- traces to a stored artifact.

Outputs: results/ondemand_streak_diagnostic.json
Updates: results/all_statistics.json -- "ondemand_streak_diagnostic" key

Run (after run_closed_loop_analysis.py):
    conda run -n wm_dynamics python scripts/run_ondemand_streak_diagnostic.py
"""
from __future__ import annotations
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from control import lqr_design, unstable_eigenvector
from closed_loop import _b_hat_at_angle

RESULTS = ROOT / "results"
B_HAT_MISMATCH_DEG = 20.0
N_TRIALS_DIAG = 40


def _decoder_and_axis(subj: str, dataset: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if dataset == "miller":
        geo = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id, times = geo["Z"], geo["task_id"], geo["times"]
        maint = (times >= 0.30) & (times <= 1.40)
        mask = (task_id == 0) | (task_id == 2)
        Z_win = Z[mask][:, maint, :]
        labels = np.repeat((task_id[mask] == 2).astype(int), Z_win.shape[1])
    else:
        geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
        Z, set_sizes = geo["Z"], geo["set_sizes"]
        correct = geo.get("correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)
        mask = ((set_sizes == 4) | (set_sizes == 8)) & correct
        Z_win = Z[mask]
        labels = np.repeat((set_sizes[mask] == 8).astype(int), Z_win.shape[1])

    Z_feat = Z_win.reshape(-1, Z_win.shape[-1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    pipe.fit(Z_feat, labels)
    w = pipe.named_steps["clf"].coef_[0] / pipe.named_steps["scaler"].scale_
    w = w / (np.linalg.norm(w) + 1e-12)
    return (lambda X: pipe.predict(X)), w


def _streak_stats(A, B_true, B_hat, x0, target, decoder, horizon, obs_noise, proc_noise,
                  rng) -> dict:
    K = lqr_design(A, B_hat, q_state=1.0, r_control=1.0)["K"]
    n = A.shape[0]
    max_streaks, engaged_fracs = [], []
    for _ in range(N_TRIALS_DIAG):
        trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        x = x0.copy()
        cur_streak, max_streak, n_engaged = 0, 0, 0
        for t in range(horizon):
            x_hat = x + trial_rng.normal(0.0, obs_noise, size=n)
            engaged = decoder(x_hat.reshape(1, -1))[0] != 1
            if engaged:
                n_engaged += 1
                max_streak = max(max_streak, cur_streak)
                cur_streak = 0
                u = -K @ (x_hat - target)
                un = np.linalg.norm(u)
                if un > 1.0:
                    u = u / un
            else:
                cur_streak += 1
                u = np.zeros(B_true.shape[1])
            w_noise = trial_rng.normal(0.0, proc_noise, size=n)
            x = A @ x + B_true @ u + w_noise
            x_norm = np.linalg.norm(x)
            if x_norm > 1e4:
                x = x * (1e4 / x_norm)
        max_streak = max(max_streak, cur_streak)
        max_streaks.append(max_streak)
        engaged_fracs.append(n_engaged / horizon)
    return {"mean_max_uncontrolled_streak": float(np.mean(max_streaks)),
            "max_uncontrolled_streak": int(np.max(max_streaks)),
            "mean_duty_cycle_diag": float(np.mean(engaged_fracs))}


def main():
    cl = json.load(open(RESULTS / "closed_loop.json"))
    tes1_miller = np.load(RESULTS / "tes1_comprehensive.npz", allow_pickle=True)
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
    div = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)

    pooled = cl["pooled"]
    stable = [k for k in cl if k != "pooled" and k not in pooled["excluded_destabilized"]]

    out = {}
    for key in stable:
        dataset, subj = ("miller", key.split("_", 1)[1]) if key.startswith("miller_") \
            else ("boran", key.split("_", 1)[1])
        tes1 = tes1_miller if dataset == "miller" else tes1_boran
        div_key = f"{dataset}_{subj}_dynamic_best_idx"
        dyn_idx = int(div[div_key])
        A = tes1[f"{subj}_A_dmd"]
        x0, target = tes1[f"{subj}_x0"], tes1[f"{subj}_xf"]
        B_true = tes1[f"{subj}_B_latent_per_tes1"][dyn_idx]

        row = cl[key]
        decoder, w_axis = _decoder_and_axis(subj, dataset)
        v_star, _ = unstable_eigenvector(A)
        rng = np.random.default_rng(hash(key) % (2**31 - 1))
        B_hat = _b_hat_at_angle(B_true, B_HAT_MISMATCH_DEG, rng)
        diag = _streak_stats(A, B_true, B_hat, x0, target, decoder, row["horizon"],
                             row["obs_noise"], row["proc_noise"], rng)
        diag["decoder_cv_acc"] = row["decoder_cv_acc"]
        diag["decoder_vstar_cos"] = float(abs(w_axis @ v_star))
        diag["ondemand_drift_reduction"] = row["ondemand"]["drift_reduction"]
        diag["horizon"] = row["horizon"]
        out[key] = diag
        print(f"  {key}: max_streak={diag['max_uncontrolled_streak']:4d}  "
              f"cv_acc={diag['decoder_cv_acc']:.3f}  cos(decoder,v*)={diag['decoder_vstar_cos']:.3f}  "
              f"ondemand_dr={diag['ondemand_drift_reduction']:.2f}")

    worst = min(out, key=lambda k: out[k]["ondemand_drift_reduction"])
    print(f"\nWorst on-demand cohort (auto-identified): {worst}")

    with open(RESULTS / "ondemand_streak_diagnostic.json", "w") as f:
        json.dump({"per_cohort": out, "worst_cohort": worst}, f, indent=2)

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["ondemand_streak_diagnostic"] = {"per_cohort": out, "worst_cohort": worst}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/ondemand_streak_diagnostic.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
