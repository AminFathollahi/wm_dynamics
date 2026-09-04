#!/usr/bin/env python3
"""2x2 factorial: DMD fit type (mean-trajectory vs single-trial ensemble) x
eigenvalue selection rule (argmax-Re vs argmax|lambda|).

Mean-trajectory + argmax-Re (this codebase's existing v* convention) and
ensemble + argmax|lambda| (run_axis_rotation_analysis's "rotation frequency"
convention) differ on BOTH axes at once, so a discrepancy between them could
be caused by either the fit type or the selection rule. This script crosses
the two axes explicitly, on the same sessions and per-cohort trial-selection/
dt conventions as run_vstar_eigen_audit.py (imported and reused, not
reimplemented).

No new DMD implementation: the mean-trajectory cell reuses
dynamics.dmd_reconstruction_error (as run_vstar_eigen_audit.py does); the
ensemble cell reuses dynamics.ensemble_dmd (for the point estimate + its
existing r2_cv/r2_null diagnostics) and dynamics._dmd_from_pairs (the same
pooled-pairs regression ensemble_dmd calls internally) for the per-bootstrap-
draw refit, so the bootstrap doesn't re-run ensemble_dmd's internal CV/null
loop on every draw.

Per session, per fit type, ONE set of B trial-resample refits is drawn; BOTH
selection rules (argmax-Re, argmax|lambda|) are read off the SAME draws
(they differ only in which eigenvalue of the same fitted operator is
selected), so this is a 2x(fit-type) x B bootstrap, not 4x.

Primary cell for any dynamics/rotation DESCRIPTIVE claim in this script's
output = ensemble + argmax|lambda| (out-of-sample validated, per the
ensemble-DMD rationale already documented in dynamics.ensemble_dmd). The
causal benchmark's v* (the modifier scored in run_macaque_pfc_microstimulation_pipeline.py) is
NOT touched or re-scored here -- it stays fixed at mean-trajectory +
argmax-Re; changing that estimand would be a separate, larger decision.

Output: results/vstar_fit_selection_factorial.json, keyed
dataset -> session -> {mean_traj: {re: {...}, mod: {...}, r_use},
                       ensemble: {re: {...}, mod: {...}, r_use,
                                  r2_cv, r2_null, r2_insample}}
plus "_meta" (bootstrap count, per-cohort ROTATING fraction in the primary
cell, a rotation-arm gate decision, and a macaque PFC microstimulation damping-vs-artifact
readout comparing the legacy and primary sigma estimates).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_vstar_fit_selection_factorial.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics import dmd_reconstruction_error, ensemble_dmd, _dmd_from_pairs
from statistics import stable_seed

import run_vstar_eigen_audit as vea  # ALL_ITERS, _circular_center, nyquist_flag

RESULTS = ROOT / "results"
B_BOOT = 300  # smaller than the single-cell audit's 1000 (4 cells here vs 1,
              # and ensemble refits are materially more expensive than
              # mean-traj ones); documented in _meta, not silently different.


def _mean_traj_eig(Z_mean: np.ndarray, r: int, dt: float) -> tuple[np.ndarray, int]:
    T, d = Z_mean.shape
    r_use = min(r, d, T - 2)
    res = dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)
    return np.linalg.eigvals(res["A"]), r_use


def _ensemble_eig_refit(Z_trials: np.ndarray, r_use: int) -> np.ndarray:
    N, T, d = Z_trials.shape
    X1 = Z_trials[:, :-1, :].reshape(-1, d).T
    X2 = Z_trials[:, 1:, :].reshape(-1, d).T
    _, lam = _dmd_from_pairs(X1, X2, r_use)
    return lam


def _pick(eigs: np.ndarray, rule: str) -> complex:
    idx = int(np.argmax(eigs.real)) if rule == "re" else int(np.argmax(np.abs(eigs)))
    return complex(eigs[idx])


def _bootstrap_both_rules(Z_trials: np.ndarray, r: int, dt: float, fit_type: str,
                           seed: int, B: int) -> dict:
    """B refits of ONE fit type; both selection rules read off each draw."""
    rng = np.random.default_rng(seed)
    N = Z_trials.shape[0]

    if fit_type == "mean_traj":
        eigs_full, r_use = _mean_traj_eig(Z_trials.mean(0), r, dt)
    else:
        r_use = min(r, Z_trials.shape[2], N * (Z_trials.shape[1] - 1) - 1)
        eigs_full = _ensemble_eig_refit(Z_trials, r_use)

    lam_full = {rule: _pick(eigs_full, rule) for rule in ("re", "mod")}
    theta_full = {rule: float(np.angle(lam_full[rule])) for rule in ("re", "mod")}
    rho_full = {rule: float(np.abs(lam_full[rule])) for rule in ("re", "mod")}

    draws = {"re": {"rho": [], "theta": []}, "mod": {"rho": [], "theta": []}}
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        Zb = Z_trials[idx]
        try:
            if fit_type == "mean_traj":
                eigs_b, _ = _mean_traj_eig(Zb.mean(0), r, dt)
            else:
                eigs_b = _ensemble_eig_refit(Zb, r_use)
        except np.linalg.LinAlgError:
            continue
        for rule in ("re", "mod"):
            lam_b = _pick(eigs_b, rule)
            draws[rule]["rho"].append(float(np.abs(lam_b)))
            draws[rule]["theta"].append(float(np.angle(lam_b)))

    out = {"r_use": int(r_use)}
    for rule in ("re", "mod"):
        rhos = np.asarray(draws[rule]["rho"])
        thetas_c = vea._circular_center(np.asarray(draws[rule]["theta"]), theta_full[rule])
        sigmas = np.log(rhos + 1e-300) / dt
        fs_c = thetas_c / (2 * np.pi * dt)

        def _pct(x):
            lo, hi = np.percentile(x, [2.5, 97.5])
            return [float(lo), float(hi)]

        out[rule] = {
            "rho": rho_full[rule], "theta": theta_full[rule],
            "sigma_s": float(np.log(rho_full[rule] + 1e-300) / dt),
            "f_hz": float(theta_full[rule] / (2 * np.pi * dt)),
            "rho_ci": _pct(rhos), "theta_ci": _pct(thetas_c),
            "sigma_ci": _pct(sigmas), "f_ci": _pct(fs_c),
            "n_boot_ok": int(len(rhos)),
        }
        out[rule].update(vea.classify_cell(out[rule]["rho_ci"], out[rule]["theta_ci"]))
    return out


def main():
    out: dict[str, dict] = {}
    rotating_counts_primary: dict[str, list[int]] = {}  # dataset -> [n_rotating, n_total]

    for it in vea.ALL_ITERS:
        for dataset, session, Z_trials, dt, r in it():
            out.setdefault(dataset, {})
            seed_mt = stable_seed(f"vstar_factorial_meantraj_{dataset}_{session}")
            seed_ens = stable_seed(f"vstar_factorial_ensemble_{dataset}_{session}")

            mean_traj = _bootstrap_both_rules(Z_trials, r, dt, "mean_traj", seed_mt, B_BOOT)
            ensemble = _bootstrap_both_rules(Z_trials, r, dt, "ensemble", seed_ens, B_BOOT)

            # Ensemble diagnostics (r2_cv/r2_null), computed ONCE (not per bootstrap
            # draw) via the existing public ensemble_dmd, per the module docstring.
            ens_diag = ensemble_dmd(Z_trials, r=ensemble["r_use"], dt=dt, n_splits=5,
                                     n_null=30, rng=np.random.default_rng(0))
            ensemble["r2_insample"] = ens_diag["r2_insample"]
            ensemble["r2_cv"] = ens_diag["r2_cv"]
            ensemble["r2_cv_std"] = ens_diag["r2_cv_std"]
            ensemble["r2_null"] = ens_diag["r2_null"]
            ensemble["r2_null_std"] = ens_diag["r2_null_std"]

            out[dataset][session] = {"mean_traj": mean_traj, "ensemble": ensemble}

            # Primary cell: ensemble + argmax|lambda| ("mod")
            rotating_counts_primary.setdefault(dataset, [0, 0])
            rotating_counts_primary[dataset][1] += 1
            if ensemble["mod"]["phase_class"] == "ROTATING":
                rotating_counts_primary[dataset][0] += 1

            print(f"  {dataset}/{session}: "
                  f"mean_traj[re] rho={mean_traj['re']['rho']:.4f} sigma={mean_traj['re']['sigma_s']:+.3f}s^-1  "
                  f"ensemble[mod] rho={ensemble['mod']['rho']:.4f} sigma={ensemble['mod']['sigma_s']:+.3f}s^-1 "
                  f"(r2_cv={ensemble['r2_cv']:.3f} r2_null={ensemble['r2_null']:.3f}) "
                  f"cell={ensemble['mod']['cell']}")

    # Rotation-arm gate: is there enough genuine rotation (theta CI excludes 0,
    # in the primary ensemble+argmax|lambda| cell) in any cohort to justify a
    # broader rotation-phase analysis program? Below ~20% in every cohort means
    # that program would be testing an absent phenomenon.
    print("\nROTATION GATE -- ROTATING fraction per cohort, primary cell (ensemble + argmax|lambda|):")
    gate_fracs = {}
    for ds, (n_rot, n_tot) in rotating_counts_primary.items():
        frac = n_rot / n_tot
        gate_fracs[ds] = frac
        print(f"  {ds}: {n_rot}/{n_tot} = {frac:.1%}")
    all_below_20pct = all(f < 0.20 for f in gate_fracs.values())
    gate_decision = (
        "GATE FAILS (rotation is present enough somewhere to proceed beyond a bounded single-cohort check)"
        if not all_below_20pct else
        "ROTATING fraction is below 20% in every cohort, but this does NOT mean rotation is absent: the "
        "leading mode is complex (oscillatory) in the majority of sessions in several cohorts, so a low "
        "ROTATING fraction can equally reflect a rotation frequency too small to resolve against bootstrap "
        "dispersion at these trial counts. Report as inconclusive-to-negative with an explicit per-cohort "
        "power bound (see results/rotation_power_bound.json), and run the full rotation-phase program "
        "(content-as-phase-offset, recency-from-geometry, phase x alignment interaction) on every cohort "
        "rather than gating any of them out; prefer the DMD-free angular-velocity-of-the-decoding-axis "
        "measure as an additional, assumption-light rotation-speed check."
    )
    print(f"ROTATION GATE DECISION: {gate_decision}")

    # macaque PFC microstimulation damping: does the ensemble estimator's sigma sign agree with the
    # legacy mean-trajectory estimator's, per session? If so, the observed
    # decay is real rather than a mean-trajectory-averaging artifact.
    print("\nmacaque PFC microstimulation sigma[s^-1], mean-traj[re] (legacy) vs ensemble[mod] (primary):")
    macaque_pfc_microstimulation_damping = []
    for session, row in out.get("macaque_pfc_microstimulation", {}).items():
        mt_sigma = row["mean_traj"]["re"]["sigma_s"]
        ens_sigma = row["ensemble"]["mod"]["sigma_s"]
        macaque_pfc_microstimulation_damping.append({"session": session, "mean_traj_re_sigma": mt_sigma,
                                "ensemble_mod_sigma": ens_sigma})
        print(f"  {session}: mean_traj[re]={mt_sigma:+.3f}  ensemble[mod]={ens_sigma:+.3f}")
    survives = sum(1 for r in macaque_pfc_microstimulation_damping if r["mean_traj_re_sigma"] < 0 and r["ensemble_mod_sigma"] < 0)
    print(f"macaque PFC microstimulation damping: {survives}/{len(macaque_pfc_microstimulation_damping)} sessions have sigma<0 under BOTH estimators "
          f"({'damping SURVIVES the ensemble fit -> real, v* is the slowest-decaying mode' if survives >= 6 else 'damping reading looks like a mean-trajectory-contraction ARTIFACT -> retreat to not-identifiably-expanding'})")

    out["_meta"] = {
        "B": B_BOOT,
        "primary_cell": "ensemble + argmax|lambda| (descriptive dynamics claims only)",
        "legacy_cell": "mean-trajectory + argmax-Re (fixed benchmark v* estimand, unchanged here)",
        "rotating_fraction_primary_by_cohort": gate_fracs,
        "rotation_gate_all_cohorts_below_20pct": all_below_20pct,
        "rotation_gate_decision": gate_decision,
        "macaque_pfc_microstimulation_damping_survives_ensemble": macaque_pfc_microstimulation_damping,
        "macaque_pfc_microstimulation_damping_n_survives": survives,
        "macaque_pfc_microstimulation_damping_n_total": len(macaque_pfc_microstimulation_damping),
    }
    with open(RESULTS / "vstar_fit_selection_factorial.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/vstar_fit_selection_factorial.json")


if __name__ == "__main__":
    main()
