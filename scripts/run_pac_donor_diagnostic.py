#!/usr/bin/env python3
"""Diagnostic for the 3 macrosignal_pac subjects whose
argmax-alignment TES1 donor destabilizes the plant (sub-02, sub-03, sub-05),
check whether a NEAR-TIE donor (pre-specified tolerance, defined BELOW before
looking at any result: alignment >= 0.90 * max_alignment) exists that does
NOT destabilize. This is the ONE legitimate lever this project's targeting-benchmark rules permit -- if
found, the same tolerance-based "prefer non-destabilizing among near-ties"
rule must be applied uniformly to every arm, not just PAC. If not found for
any of the 3 subjects, PAC's direction is genuinely destabilizing and stays
N/A on the pooling floor (a valid, reportable outcome).

Read-only: does not modify targeting_benchmark_boran.json or causal_benchmark.json.

Run: conda run -n wm_dynamics python scripts/run_pac_donor_diagnostic.py
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
sys.path.insert(0, str(Path(__file__).parent))

from closed_loop import simulate_closed_loop, _b_hat_at_angle
from statistics import stable_seed
from run_targeting_benchmark import (
    _pac_channel_weights, _stability_horizon, B_HAT_MISMATCH_DEG, RESULTS,
)

NEAR_TIE_REL_TOL = 0.90  # pre-specified BEFORE inspecting any donor's destabilization
TARGET_SUBJECTS = ["sub-02", "sub-03", "sub-05"]


def _rollout_destabilizes(A, B_true, x0, xf, Z, set_sizes, rng) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    mask_ctx = (set_sizes == 4) | (set_sizes == 8)
    Z_ctx = Z[mask_ctx]
    labels_ctx = np.repeat((set_sizes[mask_ctx] == 8).astype(int), Z_ctx.shape[1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    pipe.fit(Z_ctx.reshape(-1, Z_ctx.shape[-1]), labels_ctx)
    context_decoder = lambda X: pipe.predict(X)

    state_scale = float(np.linalg.norm(xf - x0)) + 1e-6
    proc_noise, obs_noise = 0.05 * state_scale, 0.10 * state_scale
    horizon_real = min(Z.shape[1] - 1, 300)
    stab_horizon = _stability_horizon(A, n_time_constants=3.0)
    horizon = int(min(horizon_real, np.ceil(stab_horizon))) if np.isfinite(stab_horizon) else horizon_real
    B_hat = _b_hat_at_angle(B_true, B_HAT_MISMATCH_DEG, rng)
    res = simulate_closed_loop(
        A, B_true, x0, xf, context_decoder, label=1, trigger="decoder",
        A_hat=A, B_hat=B_hat, obs_noise=obs_noise, proc_noise=proc_noise, u_budget=1.0,
        horizon=horizon, n_trials=30, n_boot=500,
        rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
    )
    return {"destabilized": bool(res["rho_closed"] > res["rho_open"]),
            "rho_open": res["rho_open"], "rho_closed": res["rho_closed"],
            "drift_reduction": res["drift_reduction"]}


def main():
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
    report = {}
    for subj in TARGET_SUBJECTS:
        geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
        Z, correct = geo["Z"], geo.get("correct", np.ones(geo["Z"].shape[0], dtype=bool)).astype(bool)
        V = geo["V"]
        A = tes1_boran[f"{subj}_A_dmd"]
        B_bank = tes1_boran[f"{subj}_B_latent_per_tes1"]
        x0, xf = tes1_boran[f"{subj}_x0"], tes1_boran[f"{subj}_xf"]
        set_sizes = geo["set_sizes"]

        mi_channels = _pac_channel_weights(subj, geo["good_ch_indices"])
        B_pac_chan = np.nan_to_num(mi_channels)
        B_pac_lat = V.T @ B_pac_chan
        B_pac_unit = B_pac_lat / (np.linalg.norm(B_pac_lat) + 1e-12)
        B_units = B_bank[:, :, 0] / (np.linalg.norm(B_bank[:, :, 0], axis=1, keepdims=True) + 1e-12)
        pac_align_to_donors = np.abs(B_units @ B_pac_unit)

        argmax_idx = int(np.argmax(pac_align_to_donors))
        max_align = float(pac_align_to_donors[argmax_idx])
        near_tie_idx = [i for i in range(len(pac_align_to_donors))
                        if pac_align_to_donors[i] >= NEAR_TIE_REL_TOL * max_align]
        # rank by alignment descending, evaluate destabilization for each near-tie donor
        near_tie_idx.sort(key=lambda i: -pac_align_to_donors[i])

        rng = np.random.default_rng(stable_seed(f"pacdiag_{subj}"))
        donor_results = []
        for idx in near_tie_idx:
            rng_i = np.random.default_rng(stable_seed(f"pacdiag_{subj}_{idx}"))
            r = _rollout_destabilizes(A, B_bank[idx], x0, xf, Z, set_sizes, rng_i)
            r["donor_idx"] = idx
            r["align"] = float(pac_align_to_donors[idx])
            donor_results.append(r)
            print(f"  {subj} donor#{idx} align={r['align']:.4f} "
                  f"destabilized={r['destabilized']} rho_open={r['rho_open']:.3f} rho_closed={r['rho_closed']:.3f}")

        argmax_destabilizes = donor_results[0]["destabilized"]
        rescue_candidates = [r for r in donor_results[1:] if not r["destabilized"]]
        report[subj] = {
            "argmax_donor_idx": argmax_idx, "argmax_align": max_align,
            "argmax_destabilizes": argmax_destabilizes,
            "n_near_tie_donors_tested": len(near_tie_idx),
            "near_tie_tolerance": NEAR_TIE_REL_TOL,
            "non_destabilizing_near_tie_found": len(rescue_candidates) > 0,
            "rescue_candidates": rescue_candidates,
            "all_tested": donor_results,
        }
        verdict = "RESCUABLE" if rescue_candidates else "GENUINELY DESTABILIZING (no near-tie donor rescues it)"
        print(f"{subj}: {verdict}\n")

    any_rescue = any(v["non_destabilizing_near_tie_found"] for v in report.values())
    report["_summary"] = {
        "near_tie_tolerance": NEAR_TIE_REL_TOL,
        "any_subject_rescuable": any_rescue,
        "decision": ("apply near-tie non-destabilizing donor preference uniformly to all arms and re-pool"
                    if any_rescue else
                    "no legitimate rescue found -- PAC N/A stands, destabilization is a genuine property "
                    "of the PAC-weighted direction, not an argmax-donor-pick artifact"),
    }
    with open(RESULTS / "pac_donor_diagnostic.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Decision:", report["_summary"]["decision"])
    print("Saved results/pac_donor_diagnostic.json")


if __name__ == "__main__":
    main()
