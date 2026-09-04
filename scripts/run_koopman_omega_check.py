#!/usr/bin/env python3
"""Koopman/EDMD lifted-fit omega as a robustness check against plain DMD.

This project already validated that a polynomial-plus-delay-embedding
Koopman lifting (dynamics.koopman_edmd) does not materially improve on plain
DMD's one-step reconstruction (Miller subjects: EDMD R^2_orig 0.971-0.989 vs.
plain-DMD R^2 0.978-0.996, results/all_statistics.json's "dmd_koopman_sindy"
entry) -- the 8-D PCA latent is already close to Koopman-invariant, so
lifting recovers nothing extra. This script does not rebuild that R^2
comparison; it extracts the ROTATION FREQUENCY from the same lifted fit as
an additional, independent robustness check: if EDMD's richer eigenbasis
agreed with the R^2 convergence result but disagreed sharply on omega, that
would itself be informative. Extended from Miller (where the R^2 result
already lives) to macaque PFC microstimulation (the cohort this project's rotation claims
actually concern).

Output: results/koopman_omega_check.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_koopman_omega_check.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics import dmd_reconstruction_error, koopman_edmd  # noqa: E402

import run_vstar_eigen_audit as vea  # noqa: E402  (_iter_miller, _iter_macaque_pfc_microstimulation)
from provenance import _json_safe

RESULTS = ROOT / "results"
DMD_RANK = 8


def _omega_from_eigs(lam: np.ndarray, dt: float) -> float:
    dominant = lam[np.argmax(np.abs(lam))]
    return float(np.abs(np.log(dominant + 1e-300).imag) / dt / (2 * np.pi))


def _omega_from_ct(lam_ct: np.ndarray) -> float:
    dominant = lam_ct[np.argmax(np.abs(lam_ct))]
    return float(np.abs(dominant.imag) / (2 * np.pi))


def check_cohort(dataset: str, iterator) -> dict:
    out = {}
    for _dataset, session, Z_trials, dt, r in iterator():
        Z_mean = Z_trials.mean(axis=0)
        d = Z_mean.shape[1]
        r_use = min(DMD_RANK, d, Z_mean.shape[0] - 2)

        plain = dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)
        lam_plain = np.linalg.eigvals(plain["A"])
        omega_plain = _omega_from_eigs(lam_plain, dt)

        edmd = koopman_edmd(Z_mean, r=r_use, dt=dt, poly_degree=2, delay_embeddings=3)
        omega_edmd = _omega_from_ct(edmd["eigenvalues_ct"])

        out[session] = {
            "omega_plain_dmd_hz": omega_plain, "omega_koopman_edmd_hz": omega_edmd,
            "abs_diff_hz": abs(omega_plain - omega_edmd),
            "r2_plain": plain["r_squared"], "r2_edmd_orig": edmd["r_squared_orig"],
        }
        print(f"  {dataset}/{session}: omega_plain={omega_plain:.4f} Hz "
              f"omega_edmd={omega_edmd:.4f} Hz (diff={abs(omega_plain-omega_edmd):.4f}) "
              f"r2_plain={plain['r_squared']:.3f} r2_edmd={edmd['r_squared_orig']:.3f}")
    return out


def main():
    print("Miller (existing EDMD-convergence cohort) ...")
    miller = check_cohort("miller", vea._iter_miller)

    print("\nmacaque PFC microstimulation (this project's rotation-claim cohort) ...")
    macaque_pfc_microstimulation = check_cohort("macaque_pfc_microstimulation", vea._iter_macaque_pfc_microstimulation)

    out = {"miller": miller, "macaque_pfc_microstimulation": macaque_pfc_microstimulation}
    with open(RESULTS / "koopman_omega_check.json", "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)
    print("\nSaved results/koopman_omega_check.json")


if __name__ == "__main__":
    main()
