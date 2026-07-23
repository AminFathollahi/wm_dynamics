#!/usr/bin/env python3
"""Minimum-energy manifold-rescue analysis: error-trial versus correct-trial
latent centroids as the LQR target, for all nine Boran subjects.

scripts/run_tes1_analysis.py already validated this LQR framework (src/control.py)
using condition-averaged latent centroids as the state-space target (0-back
versus 2-back for Miller; set-size 4 versus 8 for Boran), with a real,
stimulation-response-derived input matrix B interpolated from the TES1
(Huang et al. 2017 eLife) dataset at each subject's own electrode positions.
This script substitutes a different pair of centroids — the mean latent state
of error trials (x0) versus correct trials (xf), both averaged over the
maintenance window already saved in results/boran_geometry_sub-*.npz — reusing
the identical dynamics matrix (A, refit via exact DMD on correct-trial
trajectories) and mean interpolated TES1 input matrix (B) already computed and
saved in results/tes1_boran_B.npz by run_tes1_analysis.py.

For each subject, closed-loop LQR steering from the error centroid to the
correct centroid is compared against the passive (uncontrolled, autonomous)
trajectory under the same dynamics matrix, to establish whether reaching the
correct-trial manifold requires active intervention rather than occurring
under the fitted dynamics alone.

Requires: results/boran_geometry_sub-*.npz and results/tes1_boran_B.npz,
both produced by run_boran_pipeline.py and run_tes1_analysis.py respectively.

Outputs: results/manifold_rescue.json
Updates: results/all_statistics.json — "manifold_rescue" key

Run:
    conda run -n wm_dynamics python scripts/run_manifold_rescue_analysis.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dynamics import exact_dmd
from control import lqr_design, lqr_simulate, stimulation_energy_to_current
from statistics import paired_sign_flip_test, stable_seed
from io_utils import locked_json_update

RESULTS = ROOT / "results"
BORAN_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 10)]

N_PC = 8
T_CTRL = 150
Q_STATE = 10.0
R_CONTROL = 1.0
MIN_ERROR_TRIALS = 5


def fit_A_from_correct_trials(Z_geo: np.ndarray, correct: np.ndarray, times: np.ndarray) -> np.ndarray:
    Z_mean = Z_geo[correct].mean(axis=0).T  # (n_pc, T)
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
    dmd = exact_dmd(Z_mean, r=N_PC, dt=dt)
    Phi, lam = dmd["modes"], dmd["eigenvalues"]
    try:
        return np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))
    except np.linalg.LinAlgError:
        return 0.99 * np.eye(N_PC)


def analyze_subject(subj: str, tes1_B: dict) -> dict | None:
    geo_path = RESULTS / f"boran_geometry_{subj}.npz"
    if not geo_path.exists():
        return None
    geo = np.load(geo_path, allow_pickle=True)
    Z_geo = geo["Z"].astype(np.float64)
    times = geo["times"]
    correct = geo["correct"].astype(bool)
    error = ~correct
    if error.sum() < MIN_ERROR_TRIALS:
        return None

    x0 = Z_geo[error].mean(axis=(0, 1))
    xf = Z_geo[correct].mean(axis=(0, 1))
    A = fit_A_from_correct_trials(Z_geo, correct, times)
    B = tes1_B[f"{subj}_B_mean"].astype(np.float64)

    initial_dist = float(np.linalg.norm(x0 - xf))

    x_passive = np.zeros((T_CTRL + 1, N_PC))
    x_passive[0] = x0
    for k in range(T_CTRL):
        x_passive[k + 1] = A @ x_passive[k]
    passive_final_error = float(np.linalg.norm(x_passive[-1] - xf))
    passive_reduction_pct = 100.0 * (1.0 - passive_final_error / initial_dist)

    lqr = lqr_design(A, B, q_state=Q_STATE, r_control=R_CONTROL)
    x_traj, u_traj = lqr_simulate(A, B, lqr["K"], x0, T=T_CTRL, x_ref=xf)
    dist = np.linalg.norm(x_traj - xf, axis=1)
    controlled_final_error = float(dist[-1])
    controlled_reduction_pct = 100.0 * (1.0 - controlled_final_error / initial_dist)
    energy = float(np.sum(u_traj**2))
    stim = stimulation_energy_to_current(energy, n_channels=B.shape[1])

    return {
        "n_error_trials": int(error.sum()),
        "n_correct_trials": int(correct.sum()),
        "initial_dist": initial_dist,
        "passive_final_error": passive_final_error,
        "passive_reduction_pct": passive_reduction_pct,
        "controlled_final_error": controlled_final_error,
        "controlled_reduction_pct": controlled_reduction_pct,
        "control_energy": energy,
        "is_stable": lqr["is_stable"],
        "estimated_rms_current_uA": stim["rms_current_uA"],
    }


def main():
    tes1_B_path = RESULTS / "tes1_boran_B.npz"
    if not tes1_B_path.exists():
        print(f"SKIP - {tes1_B_path} not found; run run_tes1_analysis.py first.")
        return
    tes1_B = np.load(tes1_B_path, allow_pickle=True)

    per_subject = {}
    for subj in BORAN_SUBJECTS:
        r = analyze_subject(subj, tes1_B)
        if r is None:
            print(f"  {subj}: SKIP (missing geometry file or too few error trials)")
            continue
        per_subject[subj] = r
        print(f"  {subj}: n_error={r['n_error_trials']}, initial_dist={r['initial_dist']:.3f}, "
              f"passive_reduction={r['passive_reduction_pct']:.1f}%, "
              f"controlled_reduction={r['controlled_reduction_pct']:.1f}%, "
              f"energy={r['control_energy']:.3g}")

    if len(per_subject) < 2:
        print("Too few subjects with usable data — aborting.")
        return

    passive = np.array([r["passive_reduction_pct"] for r in per_subject.values()])
    controlled = np.array([r["controlled_reduction_pct"] for r in per_subject.values()])
    rng = np.random.default_rng(stable_seed("manifold_rescue_controlled_vs_passive"))
    contrast = paired_sign_flip_test(controlled, passive, alternative="greater", rng=rng)

    print(f"\nControlled > passive reduction (paired sign-flip, N={len(per_subject)}): "
          f"mean_diff={contrast['mean_diff']:.2f} pts, p={contrast['p_value']:.4g}")
    print(f"Mean passive reduction: {passive.mean():.1f}%  "
          f"Mean controlled reduction: {controlled.mean():.1f}%")
    print(f"Mean control energy: {np.mean([r['control_energy'] for r in per_subject.values()]):.3g}")

    out = {
        "per_subject": per_subject,
        "n_subjects": int(len(per_subject)),
        "mean_passive_reduction_pct": float(passive.mean()),
        "mean_controlled_reduction_pct": float(controlled.mean()),
        "controlled_greater_than_passive": {
            "mean_diff": contrast["mean_diff"],
            "p_value": contrast["p_value"],
            "ci_lower": contrast["ci_lower"],
            "ci_upper": contrast["ci_upper"],
        },
        "q_state": Q_STATE,
        "r_control": R_CONTROL,
        "t_ctrl": T_CTRL,
    }
    import json
    with open(RESULTS / "manifold_rescue.json", "w") as f:
        json.dump(out, f, indent=2)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["manifold_rescue"] = out
    print("\nSaved results/manifold_rescue.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
