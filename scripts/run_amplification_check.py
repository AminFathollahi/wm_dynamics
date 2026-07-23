#!/usr/bin/env python3
"""Round-11 PART 17A -- the deferred 16A amplification check.

WHY: PAPER_REPORT.tex's 16A paragraph already ASSERTS (as interpretation)
that v* has causal leverage because cortical dynamics are non-normal and a
perturbation is amplified along the propagator's top RIGHT SINGULAR vector,
not its top eigenvector (Murphy&Miller 2009; Goldman 2009; Hennequin/Vogels/
Gerstner 2014; Bondanelli&Ostojic 2020). This script supplies the one missing
number: cos(v*, w1) per Soldado session, where v* is the EXACT vector the
causal benchmark already uses (src/control.py:unstable_eigenvector, same
argmax(eigs.real)+unit-norm as run_soldado_pipeline.build_session_features)
and w1 is the top right singular vector of the SAME control-epoch-fit A.

Does NOT redefine v* or refit A independently: reuses
run_soldado_pipeline.load_soldado_session + the identical PCA/DMD fit
(src/geometry.pca_decompose, src/dynamics.dmd_reconstruction_error) so A is
bit-identical to the one the benchmark scores.

Run:
    /home/amin/miniconda3/bin/graphify query ...  (done -- see comments.txt)
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_amplification_check.py
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

from geometry import pca_decompose
from dynamics import dmd_reconstruction_error
from control import unstable_eigenvector

sys.path.insert(0, str(ROOT / "scripts"))
from run_soldado_pipeline import (
    load_soldado_session, crop_trial, SESSIONS, N_PC, DMD_RANK, N_BINS, BIN_S,
)

RESULTS = ROOT / "results"


def _self_check() -> None:
    """Planted non-normal 2x2: eigenvector != singular vector, cos matches hand calc."""
    A = np.array([[1.0, 5.0], [0.0, 0.5]])  # strongly non-normal, upper-triangular
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)[::-1]
    v_star = vecs[:, order[0]].real
    v_star /= np.linalg.norm(v_star) + 1e-12

    U, S, Vt = np.linalg.svd(A)
    w1 = Vt[0].real
    w1 /= np.linalg.norm(w1) + 1e-12

    cos_hand = abs(float(np.dot(v_star, w1)))
    # hand-computed expected values for this specific matrix (eigenvector of
    # the dominant eigenvalue 1.0 is [1,0]; top right singular vector is NOT
    # [1,0] because A is non-normal -- assert they differ and cos < 1).
    assert np.allclose(v_star, [1.0, 0.0], atol=1e-8), v_star
    assert not np.allclose(w1, v_star, atol=1e-3), "self-check matrix must be non-normal (w1 != v*)"
    assert 0.0 < cos_hand < 0.999, f"self-check cos out of expected non-normal range: {cos_hand}"
    # recompute via the same unstable_eigenvector() helper used for the real data
    v_star2, _ = unstable_eigenvector(A)
    assert np.allclose(np.abs(v_star2), np.abs(v_star), atol=1e-8)
    print(f"[self-check] planted non-normal 2x2: cos(v*,w1)={cos_hand:.4f} "
          f"(eigenvector != singular vector, as expected) -- PASS")


def _session_amp(prefix: str) -> dict | None:
    corr = load_soldado_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    control_idx = corr["control_idx"]
    channel_ids = corr["channel_ids"]
    C = len(channel_ids)

    ctrl_epochs = [
        crop_trial(tr["spikerate"]) for tr in corr["trials"] if tr["stim_cond"] == control_idx
    ]
    ctrl_epochs = [e for e in ctrl_epochs if e is not None]
    if len(ctrl_epochs) < 10:
        return None

    Z_ctrl = np.stack(ctrl_epochs, axis=0)  # (N, N_BINS, C)
    X_flat = Z_ctrl.reshape(-1, C)
    _, V, _ = pca_decompose(X_flat, N_PC)
    k = V.shape[1]
    Z_ctrl_mean = ((Z_ctrl.reshape(-1, C) - X_flat.mean(0)) @ V).reshape(
        Z_ctrl.shape[0], N_BINS, k).mean(0)
    r_use = min(DMD_RANK, k, N_BINS - 2)
    dmd = dmd_reconstruction_error(Z_ctrl_mean, r=r_use, dt=BIN_S)
    A = dmd["A"]

    v_star, max_re_eig = unstable_eigenvector(A)

    U, S, Vt = np.linalg.svd(A)
    w1 = Vt[0].real
    w1 /= np.linalg.norm(w1) + 1e-12
    sigma1 = float(S[0])

    cos_va = abs(float(np.dot(v_star, w1)))
    amp_factor = sigma1 / (abs(max_re_eig) + 1e-12)

    # multistep note (transient growth): A^k for k = N_BINS (delay length),
    # top singular vector of A^k -- report only as a convergence note, not primary.
    Ak = np.linalg.matrix_power(A, N_BINS)
    Uk, Sk, Vtk = np.linalg.svd(Ak)
    wk = Vtk[0].real
    wk /= np.linalg.norm(wk) + 1e-12
    cos_va_multistep = abs(float(np.dot(v_star, wk)))

    return {
        "cos_vstar_w1": cos_va,
        "amp_factor_sigma1_over_lambda": amp_factor,
        "max_real_eig": float(max_re_eig),
        "sigma1": sigma1,
        "cos_vstar_w1_multistep_Nbins": cos_va_multistep,
        "n_pc": k,
    }


def main() -> None:
    _self_check()

    per_session = {}
    for prefix in SESSIONS:
        print(f"  {prefix} ...", end=" ")
        try:
            res = _session_amp(prefix)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        if res is None:
            print("SKIP (insufficient data)")
            continue
        per_session[prefix] = res
        print(f"cos(v*,w1)={res['cos_vstar_w1']:.4f} amp_factor={res['amp_factor_sigma1_over_lambda']:.3f}")

    if not per_session:
        out = {"status": "infeasible", "reason": "no usable Soldado sessions",
               "per_session": {}, "n_sessions": 0}
        with open(RESULTS / "amplification_check.json", "w") as f:
            json.dump(out, f, indent=2)
        print("No usable sessions -- wrote infeasible status.")
        return

    cos_vals = np.array([v["cos_vstar_w1"] for v in per_session.values()])
    amp_vals = np.array([v["amp_factor_sigma1_over_lambda"] for v in per_session.values()])
    median_cos = float(np.median(cos_vals))
    median_amp = float(np.median(amp_vals))

    out = {
        "per_session": per_session,
        "median_cos_vstar_w1": median_cos,
        "median_amp_factor": median_amp,
        "A_fit_source": "control_epochs",
        "n_sessions": len(per_session),
    }
    with open(RESULTS / "amplification_check.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    print(f"\nPART 17A: median cos(v*, w1) = {median_cos:.4f} over n={len(per_session)} sessions "
          f"(median amp factor sigma1/|lambda_max| = {median_amp:.3f})")
    if median_cos >= 0.9:
        print("median cos >= 0.9 -> v* IS empirically the maximally-amplified mode; "
              "17B SKIPPED (no dissociable difference between eigen-v* and singular-w1).")
    else:
        print("median cos < 0.9 -> directions genuinely differ; PART 17B should run.")


if __name__ == "__main__":
    main()
