#!/usr/bin/env python3
"""R4/R5: in-silico closed-loop demonstration + robustness sweep.

For every cohort with BOTH a fitted plant (A, from run_tes1_analysis.py's
DMD fit) and a TES1 B matrix (Miller + Boran iEEG — the only cohorts with
DLPFC TES1 coverage), this:
  1. Builds a content/context decoder trained on that cohort's own REAL,
     uncontrolled trial data (02_geometry_*.npz for Miller, boran_geometry_*.npz
     for Boran) — never on anything this script simulates (anti-circularity
     guardrail 2).
  2. Runs src/closed_loop.simulate_closed_loop from the "low" condition
     centroid (x0) toward the "high" condition centroid (target = xf), over
     a horizon matched to that cohort's real maintenance-window sample count
     (the A matrix was fit at native per-sample dt, so a short arbitrary
     horizon under-samples the near-unit-circle dynamics R2 identified and
     manufactures a spuriously small open-loop drift) — CAPPED at
     N_TIME_CONSTANTS e-folding times of A's least-stable eigenvalue when
     that is shorter than the real sample count. Every fitted A here has
     max|eig|>1 (near-unit-circle-to-mildly-unstable, per R2), with e-folding
     time constants ranging ~70-850 steps across cohorts; at the full native
     horizon (1099-4193 steps) the fast-diverging cohorts accumulate tens of
     e-foldings and the open-loop rollout saturates src/closed_loop.py's
     numerical state-norm cap well before the horizon ends, making
     drift_reduction an artifact of that arbitrary cap rather than a real
     control-benefit estimate. Capping at a fixed, principled number of each
     cohort's OWN time constants keeps every cohort in the cap-free regime
     while still showing genuine (non-under-sampled) open-loop divergence.
     The controller is designed on B_hat — the v*-policy-selected donor
     (dynamic_best_idx from run_divergence_analysis.py) rotated by a
     moderate, fixed angle, standing in for realistic B-estimation error —
     but evaluated against the true plant using the UNROTATED
     policy-selected donor as B (guardrail 1).
  3. Runs src/closed_loop.robustness_sweep on the same (A, B_true) pair.

Outputs: results/closed_loop.json, results/closed_loop_robustness.json
Updates: results/all_statistics.json — "closed_loop" / "closed_loop_robustness" keys

Run (after run_tes1_analysis.py and run_divergence_analysis.py):
    conda run -n wm_dynamics python scripts/run_closed_loop_analysis.py
"""
from __future__ import annotations
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from closed_loop import simulate_closed_loop, robustness_sweep, _b_hat_at_angle
from statistics import bootstrap_ci, stable_seed

RESULTS = ROOT / "results"
DECODER_VALID_MARGIN = 0.03   # decoder_cv_acc must beat chance (1/n_cls) by this much

MILLER_SUBJECTS = ["al", "ca", "cc", "ug"]
BORAN_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 10)]

MAINT_T0, MAINT_T1 = 0.30, 1.40   # Miller maintenance window (s)
B_HAT_MISMATCH_DEG = 20.0         # realistic B-estimation-error stand-in (guardrail 1)
N_TIME_CONSTANTS = 3.0            # cap horizon at this many e-folding times of A's
                                   # least-stable eigenvalue (see module docstring) —
                                   # e^3≈20x growth from baseline is a clear, meaningful
                                   # divergence signal without approaching the state cap
N_TRIALS = 100
N_BOOT = 1000
U_BUDGET = 1.0

SWEEP_N_TRIALS = 30
SWEEP_N_BOOT = 300
SWEEP_ANGLES = np.linspace(0.0, 90.0, 8)
SWEEP_NONLINEARITY = np.linspace(0.0, 1.0, 8)


def _fit_decoder(Z_feat: np.ndarray, labels: np.ndarray):
    """Logistic-regression content/context decoder on real, uncontrolled
    per-trial features (guardrail 2: fit here, on real data, before any
    simulation happens)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    pipe.fit(Z_feat, labels)
    return lambda X: pipe.predict(X)


def _decoder_cv_acc(Z_feat: np.ndarray, labels: np.ndarray, n_splits: int = 5) -> float:
    """Cross-validated balanced accuracy of the same decoder architecture on
    the cohort's real trials — a validity check independent of the fit
    above (decoder must beat chance on real data, not just exist)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    n_splits = min(n_splits, int(np.min(np.bincount(labels))))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, Z_feat, labels, cv=cv, scoring="balanced_accuracy")
    return float(scores.mean())


def _miller_decoder_and_horizon(subj: str):
    geo = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
    Z, task_id, times = geo["Z"], geo["task_id"], geo["times"]
    maint = (times >= MAINT_T0) & (times <= MAINT_T1)
    mask = (task_id == 0) | (task_id == 2)
    # Per-timestep (not trial-averaged) samples: simulate_closed_loop scores
    # the decoder on instantaneous simulated states, so the decoder must be
    # trained on instantaneous real states to match that feature distribution.
    Z_win = Z[mask][:, maint, :]           # (N, T_maint, k)
    n_trials_, t_maint_, k_ = Z_win.shape
    Z_feat = Z_win.reshape(-1, k_)
    labels = np.repeat((task_id[mask] == 2).astype(int), t_maint_)
    horizon = t_maint_ - 1
    cv_acc = _decoder_cv_acc(Z_feat, labels)
    return _fit_decoder(Z_feat, labels), horizon, cv_acc


def _boran_decoder_and_horizon(subj: str):
    geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
    Z, set_sizes, times = geo["Z"], geo["set_sizes"], geo["times"]
    correct = geo.get("correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)
    mask = ((set_sizes == 4) | (set_sizes == 8)) & correct
    Z_win = Z[mask]                        # (N, T, k)
    n_trials_, t_, k_ = Z_win.shape
    Z_feat = Z_win.reshape(-1, k_)
    labels = np.repeat((set_sizes[mask] == 8).astype(int), t_)
    horizon = t_ - 1
    cv_acc = _decoder_cv_acc(Z_feat, labels)
    return _fit_decoder(Z_feat, labels), horizon, cv_acc


def _stability_horizon(A: np.ndarray, n_time_constants: float) -> float:
    """n_time_constants e-folding times of A's least-stable eigenvalue, in steps."""
    log_lam_max = np.log(np.max(np.abs(np.linalg.eigvals(A))))
    if abs(log_lam_max) < 1e-12:
        return np.inf
    return n_time_constants / abs(log_lam_max)


def _run_cohort(bundle: dict, dynamic_best_idx: int, decoder, horizon_real: int,
                decoder_cv_acc: float, n_cls: int, rng) -> tuple[dict, dict]:
    A = bundle["A_dmd"]
    x0 = bundle["x0"]
    target = bundle["xf"]
    B_true = bundle["B_latent_per_tes1"][dynamic_best_idx]
    B_hat = _b_hat_at_angle(B_true, B_HAT_MISMATCH_DEG, rng)

    stab_horizon = _stability_horizon(A, N_TIME_CONSTANTS)
    horizon = int(min(horizon_real, np.ceil(stab_horizon))) if np.isfinite(stab_horizon) else horizon_real

    state_scale = float(np.linalg.norm(target - x0)) + 1e-6
    proc_noise = 0.05 * state_scale
    obs_noise = 0.10 * state_scale

    # res_seed is drawn from the shared `rng` (as before A2/A4 — preserves the
    # exact rng-consumption sequence downstream cohorts' B_hat draws depend
    # on, so the stored Round-4 continuous numbers stay bit-for-bit
    # unchanged); the on-demand run's seed is DERIVED, not drawn, so it adds
    # no extra call to the shared stream.
    res_seed = int(rng.integers(0, 2**31 - 1))
    res = simulate_closed_loop(
        A, B_true, x0, target, decoder, label=1,
        A_hat=A, B_hat=B_hat,   # guardrail 1: moderately mismatched B_hat
        obs_noise=obs_noise, proc_noise=proc_noise, u_budget=U_BUDGET,
        horizon=horizon, n_trials=N_TRIALS, n_boot=N_BOOT,
        rng=np.random.default_rng(res_seed),
    )
    ondemand = simulate_closed_loop(
        A, B_true, x0, target, decoder, label=1, trigger="decoder",
        A_hat=A, B_hat=B_hat, obs_noise=obs_noise, proc_noise=proc_noise, u_budget=U_BUDGET,
        horizon=horizon, n_trials=N_TRIALS, n_boot=N_BOOT,
        rng=np.random.default_rng(res_seed + 1),
    )
    sweep = robustness_sweep(
        A, B_true, x0, target, decoder, label=1,
        obs_noise=obs_noise, proc_noise=proc_noise, u_budget=U_BUDGET,
        horizon=horizon, n_trials=SWEEP_N_TRIALS, n_boot=SWEEP_N_BOOT,
        mismatch_angles_deg=SWEEP_ANGLES, nonlinearity_scales=SWEEP_NONLINEARITY,
        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
    )

    destabilized = bool(res["rho_closed"] > res["rho_open"])
    decoder_valid = bool(np.isfinite(decoder_cv_acc) and decoder_cv_acc > (1.0 / n_cls) + DECODER_VALID_MARGIN)

    demo_row = {
        "drift_on": res["drift_on"], "drift_on_ci": res["drift_on_ci"],
        "drift_off": res["drift_off"], "drift_off_ci": res["drift_off_ci"],
        "drift_reduction": res["drift_reduction"], "drift_reduction_ci": res["drift_reduction_ci"],
        "decodability_on": res["decodability_on"], "decodability_on_ci": res["decodability_on_ci"],
        "decodability_off": res["decodability_off"], "decodability_off_ci": res["decodability_off_ci"],
        "decodability_lift": res["decodability_lift"], "decodability_lift_ci": res["decodability_lift_ci"],
        "rho_open": res["rho_open"], "rho_closed": res["rho_closed"], "destabilized": destabilized,
        "decoder_cv_acc": decoder_cv_acc, "decoder_valid": decoder_valid,
        "state_scale": state_scale, "proc_noise": proc_noise, "obs_noise": obs_noise,
        "horizon": horizon, "horizon_real": horizon_real,
        "horizon_stability_cap": float(stab_horizon) if np.isfinite(stab_horizon) else None,
        "dynamic_best_tes1_idx": int(dynamic_best_idx),
        "ondemand": {
            "drift_reduction": ondemand["drift_reduction"], "drift_reduction_ci": ondemand["drift_reduction_ci"],
            "decodability_lift": ondemand["decodability_lift"], "decodability_lift_ci": ondemand["decodability_lift_ci"],
            "rho_closed": ondemand["rho_closed"], "destabilized": bool(ondemand["rho_closed"] > ondemand["rho_open"]),
            "control_energy": ondemand["control_energy"], "duty_cycle": ondemand["duty_cycle"],
            "continuous_control_energy": res["control_energy"],
        },
    }
    sweep_row = {
        "nominal_drift_reduction": sweep["nominal_drift_reduction"],
        "angle_sweep": sweep["angle_sweep"], "noise_sweep": sweep["noise_sweep"],
        "nonlinearity_sweep": sweep["nonlinearity_sweep"],
        "failure_boundary_angle_deg": sweep["failure_boundary_angle_deg"],
        "failure_boundary_obs_noise": sweep["failure_boundary_obs_noise"],
        "failure_boundary_nonlinearity_scale": sweep["failure_boundary_nonlinearity_scale"],
    }
    return demo_row, sweep_row


def main():
    rng = np.random.default_rng(0)
    tes1 = np.load(RESULTS / "tes1_comprehensive.npz", allow_pickle=True)
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
    div = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)

    demo_out, sweep_out = {}, {}

    print("Miller:")
    for subj in MILLER_SUBJECTS:
        if f"{subj}_A_dmd" not in tes1:
            print(f"  SKIP {subj} — no TES1 bundle")
            continue
        div_key = f"miller_{subj}_dynamic_best_idx"
        if div_key not in div:
            print(f"  SKIP {subj} — no divergence_analysis dynamic_best_idx")
            continue
        bundle = {
            "A_dmd": tes1[f"{subj}_A_dmd"], "x0": tes1[f"{subj}_x0"], "xf": tes1[f"{subj}_xf"],
            "B_latent_per_tes1": tes1[f"{subj}_B_latent_per_tes1"],
        }
        decoder, horizon, cv_acc = _miller_decoder_and_horizon(subj)
        demo_row, sweep_row = _run_cohort(bundle, int(div[div_key]), decoder, horizon, cv_acc, 2, rng)
        demo_out[f"miller_{subj}"], sweep_out[f"miller_{subj}"] = demo_row, sweep_row
        print(f"  {subj}: horizon={demo_row['horizon']} (real={demo_row['horizon_real']}) "
              f"drift_reduction={demo_row['drift_reduction']:.4f} "
              f"decodability_lift={demo_row['decodability_lift']} "
              f"destabilized={demo_row['destabilized']} decoder_valid={demo_row['decoder_valid']}")

    print("Boran iEEG:")
    for subj in BORAN_SUBJECTS:
        if f"{subj}_A_dmd" not in tes1_boran:
            print(f"  SKIP {subj} — no TES1 bundle")
            continue
        div_key = f"boran_{subj}_dynamic_best_idx"
        if div_key not in div:
            print(f"  SKIP {subj} — no divergence_analysis dynamic_best_idx")
            continue
        bundle = {
            "A_dmd": tes1_boran[f"{subj}_A_dmd"], "x0": tes1_boran[f"{subj}_x0"],
            "xf": tes1_boran[f"{subj}_xf"], "B_latent_per_tes1": tes1_boran[f"{subj}_B_latent_per_tes1"],
        }
        decoder, horizon, cv_acc = _boran_decoder_and_horizon(subj)
        demo_row, sweep_row = _run_cohort(bundle, int(div[div_key]), decoder, horizon, cv_acc, 2, rng)
        demo_out[f"boran_{subj}"], sweep_out[f"boran_{subj}"] = demo_row, sweep_row
        print(f"  {subj}: horizon={demo_row['horizon']} (real={demo_row['horizon_real']}) "
              f"drift_reduction={demo_row['drift_reduction']:.4f} "
              f"decodability_lift={demo_row['decodability_lift']} "
              f"destabilized={demo_row['destabilized']} decoder_valid={demo_row['decoder_valid']}")

    # Pooled summary — the destabilized set (rho_closed > rho_open, on the TRUE
    # plant and the run's own K) and the decoder-validity set are the a-priori
    # inclusion criteria; the naive mean over all 13 (including destabilized
    # runs whose drift diverges without bound) is not a meaningful "benefit"
    # estimate and is never reported as the paper's number.
    names = list(demo_out.keys())
    stable = [k for k in names if not demo_out[k]["destabilized"]]
    valid = [k for k in stable if demo_out[k]["decoder_valid"]]
    excluded_destabilized = [k for k in names if demo_out[k]["destabilized"]]
    excluded_degenerate_decoder = [k for k in stable if not demo_out[k]["decoder_valid"]]

    def _mode_block(mode: str, seed_tag: str) -> dict:
        def _get(k, field):
            return demo_out[k][field] if mode == "continuous" else demo_out[k]["ondemand"][field]

        dr = np.array([_get(k, "drift_reduction") for k in stable])
        lift = np.array([_get(k, "decodability_lift") for k in valid])
        dr_m, dr_lo, dr_hi = bootstrap_ci(dr, np.mean, n_boot=20000,
                                          rng=np.random.default_rng(stable_seed(f"{seed_tag}_drift")))
        lift_m, lift_lo, lift_hi = bootstrap_ci(lift, np.mean, n_boot=20000,
                                                rng=np.random.default_rng(stable_seed(f"{seed_tag}_lift")))
        block = {
            "n_drift_pool": len(stable), "n_drift_positive": int(np.sum(dr > 0)),
            "drift_reduction_mean_ci": [dr_m, dr_lo, dr_hi],
            "n_decod_pool": len(valid), "n_decod_positive": int(np.sum(lift > 0)),
            "decodability_lift_mean_ci": [lift_m, lift_lo, lift_hi],
        }
        if mode == "ondemand":
            duty = np.array([demo_out[k]["ondemand"]["duty_cycle"] for k in stable])
            ratio = np.array([demo_out[k]["ondemand"]["control_energy"]
                              / max(demo_out[k]["ondemand"]["continuous_control_energy"], 1e-12)
                              for k in stable])
            block["mean_duty_cycle"] = float(duty.mean())
            block["mean_control_energy_ratio_to_continuous"] = float(ratio.mean())

            # Sensitivity to the single worst realized on-demand trial: a real
            # decoder can fail as an early-warning trigger over a long,
            # only-marginally-stabilizable horizon (rho_closed close to
            # rho_open), letting the plant run open-loop long enough between
            # triggers to compound -- report the pooled estimate BOTH with
            # and without whatever cohort that turns out to be (identified
            # generically, by worst realized drift_reduction, not by name).
            worst_idx = int(np.argmin(dr))
            worst_name = stable[worst_idx]
            dr_excl = np.delete(dr, worst_idx)
            stable_excl = [k for i, k in enumerate(stable) if i != worst_idx]
            m2, lo2, hi2 = bootstrap_ci(dr_excl, np.mean, n_boot=20000,
                                        rng=np.random.default_rng(stable_seed(f"{seed_tag}_drift_excl_worst")))
            block["sensitivity_excl_worst"] = {
                "excluded_cohort": worst_name,
                "n_drift_pool": len(dr_excl), "n_drift_positive": int(np.sum(dr_excl > 0)),
                "drift_reduction_mean_ci": [m2, lo2, hi2],
            }
        return block

    pooled = {
        "n_total": len(names),
        "excluded_destabilized": excluded_destabilized,
        "excluded_degenerate_decoder": excluded_degenerate_decoder,
        "continuous": _mode_block("continuous", "cl_pooled_continuous"),
        "ondemand": _mode_block("ondemand", "cl_pooled_ondemand"),
    }
    print(f"\nPooled: n_total={pooled['n_total']} "
          f"destabilized={excluded_destabilized} degenerate_decoder={excluded_degenerate_decoder}")
    print(f"  continuous: {json.dumps(pooled['continuous'])}")
    print(f"  ondemand:   {json.dumps(pooled['ondemand'])}")

    demo_out["pooled"] = pooled
    with open(RESULTS / "closed_loop.json", "w") as f:
        json.dump(demo_out, f, indent=2)
    with open(RESULTS / "closed_loop_robustness.json", "w") as f:
        json.dump(sweep_out, f, indent=2)

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["closed_loop"] = demo_out
    stats["closed_loop_robustness"] = sweep_out
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/closed_loop.json, results/closed_loop_robustness.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
