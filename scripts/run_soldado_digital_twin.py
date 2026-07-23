#!/usr/bin/env python3
"""Round-9 Part 14 -- digital-twin input-LDS depth move.

WHY: the existing Soldado causal arm (run_soldado_pipeline.py) only
CORRELATES the real per-condition causal effect with a SCALAR modifier
(|cos(B, v*)|). This script fits an explicit input-driven latent
state-space model (src.dynamics.fit_input_lds), delivers the SIMULATED
microstim as an input pulse (src.dynamics.simulate_input_response), and
asks whether the simulated latent DISPLACEMENT MAGNITUDE predicts the real
per-condition causal effect BETTER than the scalar v*-alignment baseline.
Turns a scalar correlation into a mechanistic trajectory prediction, on
data already in hand -- no new dataset, no new causal estimator.

PI GUARDRAILS (mandatory, see comments.txt Part 14B):
  (i)   NO-CIRCULARITY: A (and C) are fit from CONTROL / no-stim epochs
        ONLY -- via fit_input_lds with an all-zero input (so B is
        unidentified from that fit and DISCARDED); B is instead the
        stimulated-electrode one-hot projected into the control-fit latent
        space (C), EXACTLY as run_soldado_pipeline.build_session_features
        already does for its own alignment-to-v* modifier. This is
        recorded as A_fit_source="control_epochs" in the output JSON.
  (ii)  SAME UNITS, APPLES-TO-APPLES: the twin predictor and the scalar
        v*-alignment baseline are scored on the IDENTICAL set of
        per-condition effects (same conditions, same sessions, same
        exclusions). n_conditions is recorded.
  (iii) SMALL-n HONESTY: n is the number of stim conditions pooled across
        sessions (tens). The comparison uses a condition-level bootstrap
        CI (src.statistics.bootstrap_ci, resampling CONDITIONS), not a
        trial-level p-value. If n_conditions < ~10 the result is reported
        as suggestive, not adjudicated.

The REAL per-condition causal effect reuses the EXISTING doubly-robust
pseudo-outcome (src.causal.aipw_pseudo_outcome / crossfit_nuisances, the
same call run_soldado_pipeline.py already makes for the CATE-vs-alignment
gate) -- NOT a new causal estimator. The per-CONDITION effect is simply the
mean of that existing per-TRIAL pseudo-outcome phi over the condition's
stim trials (a direct aggregation of an object already computed elsewhere
in this project, not a new statistical claim).

Self-gate: if the amp/onset structure needed to build U cannot be
resolved for ANY session, or no session yields >=1 usable condition,
write results/digital_twin.json {"status": "infeasible", "reason": ...}
and STOP -- do not fabricate.

Run:
    conda run -n wm_dynamics python scripts/run_soldado_digital_twin.py
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

import run_soldado_pipeline as soldado  # noqa: E402  (reuse SESSIONS, loaders, N_BINS, PRE_S, BIN_S, N_PC, DMD_RANK)
from dynamics import fit_input_lds, simulate_input_response  # noqa: E402
from causal import crossfit_nuisances, aipw_pseudo_outcome  # noqa: E402
from statistics import stable_seed, bootstrap_ci  # noqa: E402
from geometry import pca_decompose  # noqa: E402

RESULTS = ROOT / "results"

# Onset bin: PRE_S=0.8s pre-stim-onset, BIN_S=0.05s bins -> bin 16 is the
# first post-onset bin (matches build_session_features' crop window,
# -0.8..+0.7 s re: stim onset over N_BINS=30 bins).
ONSET_BIN = round(soldado.PRE_S / soldado.BIN_S)
STIM_ON_DURATION_S = 0.3   # boxcar width: conservative sub-window of the
                            # post-onset delay period actually available
                            # (30 - 16 = 14 bins ~= 0.7s); 0.3s = 6 bins
                            # keeps the "pulse" well inside the crop window
                            # for every session.
STIM_ON_BINS = max(1, round(STIM_ON_DURATION_S / soldado.BIN_S))


def _fit_twin_for_session(prefix: str) -> dict | None:
    """Build (A, B, C) for one session honoring guardrail (i), then the
    per-condition simulated displacement magnitude alongside the real
    per-condition causal effect (mean phi) and the scalar v*-alignment
    baseline, on the IDENTICAL condition set (guardrail ii)."""
    corr = soldado.load_soldado_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    err = soldado.load_soldado_session(prefix, correct=False)

    control_idx = corr["control_idx"]
    channel_ids = corr["channel_ids"]
    C_chan = len(channel_ids)

    def _epochs_for(cond_source, cond, label_correct):
        out = []
        for tr in cond_source["trials"]:
            if tr["stim_cond"] != cond:
                continue
            cropped = soldado.crop_trial(tr["spikerate"])
            if cropped is not None:
                out.append((cropped, label_correct, tr["angle_idx"]))
        return out

    ctrl_epochs = [e for e, _, _ in _epochs_for(corr, control_idx, 1)]
    if len(ctrl_epochs) < 10:
        return None
    n_bins = soldado.N_BINS
    if n_bins <= ONSET_BIN:
        # Self-gate: no post-onset bins available to build a stim-on boxcar.
        return None

    Z_ctrl = np.stack(ctrl_epochs, axis=0)          # (N, N_BINS, C_chan)
    X_flat = Z_ctrl.reshape(-1, C_chan)

    # (i) NO-CIRCULARITY: A and C are fit from CONTROL epochs ONLY. PCA
    # (identical call to build_session_features: pca_decompose(X_flat,
    # N_PC)) gives the control-fit latent space; the trial-averaged control
    # trajectory in that space is then fed to fit_input_lds with an
    # ALL-ZERO input (so any B that call returns is unidentified from
    # no-stim data -- discarded below, never used; only A is kept).
    _, V, _ = pca_decompose(X_flat, soldado.N_PC)   # V: (C_chan, k) -- control-fit latent basis
    k_pc = V.shape[1]
    x_mean = X_flat.mean(0)
    Z_ctrl_mean = ((X_flat - x_mean) @ V).reshape(Z_ctrl.shape[0], n_bins, k_pc).mean(0)  # (N_BINS, k_pc)
    U_ctrl = np.zeros((n_bins, 1))
    r_use = min(soldado.DMD_RANK, k_pc, n_bins - 2)
    A, _B_unused, C_local, _z = fit_input_lds(Z_ctrl_mean, U_ctrl, latent_dim=r_use)
    k = A.shape[0]
    # Compose: control-space channel coords -> PCA latent (V) -> fit_input_lds's
    # own (possibly rank-reduced) latent basis (C_local) = the full map used
    # for B's projection below (mirrors build_session_features' V.T @ B_chan,
    # generalised to fit_input_lds's own internal basis).
    Cmat = V @ C_local   # (C_chan, k)

    # B per stim condition: stimulated-electrode one-hot projected into the
    # CONTROL-fit latent space via Cmat (Cmat columns span the same PCA
    # subspace fit_input_lds derived from X_flat) -- identical construction
    # to build_session_features' B_lat, just reusing Cmat instead of a
    # separately-named V.
    n_cond = len(corr["stim_channels"])
    cond_ids, B_by_cond, align_by_cond = [], {}, {}
    for c in range(n_cond):
        if c == control_idx:
            continue
        chan_ids = corr["stim_channels"][c]
        idx = [i for i, cid in enumerate(channel_ids) if cid in chan_ids]
        if not idx:
            continue
        B_chan = np.zeros((C_chan, 1))
        B_chan[idx, 0] = 1.0 / len(idx)
        B_lat = Cmat.T @ B_chan   # (k, 1) -- projection into control latent space
        cond_ids.append(c)
        B_by_cond[c] = B_lat

    if not cond_ids:
        return None

    # Simulated microstim: unit boxcar input over the post-onset bins,
    # zero pre-onset -- reconstructable per-step stim timing (feasibility
    # gate, comments.txt Part 14). Latent displacement magnitude = the
    # simulated latent state's net excursion from a zero baseline over the
    # stim-on window.
    n_post = n_bins - ONSET_BIN
    stim_bins = min(STIM_ON_BINS, n_post)
    U_stim = np.zeros((stim_bins, 1))
    U_stim[:, 0] = 1.0

    twin_displacement = {}
    for c in cond_ids:
        Z_sim = simulate_input_response(A, B_by_cond[c], Cmat, np.zeros(k), U_stim)
        disp = Z_sim[-1] - Z_sim[0]
        twin_displacement[c] = float(np.linalg.norm(disp))
        # scalar v*-alignment baseline: dominant eigenvector of A (same
        # definition as build_session_features' v_star).
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)[::-1]
    v_star = vecs[:, order[0]].real
    v_star = v_star / (np.linalg.norm(v_star) + 1e-12)
    for c in cond_ids:
        b_unit = B_by_cond[c][:, 0] / (np.linalg.norm(B_by_cond[c]) + 1e-12)
        align_by_cond[c] = float(np.abs(b_unit @ v_star))

    # Real per-condition causal effect: mean of the EXISTING per-trial
    # doubly-robust pseudo-outcome phi (same crossfit_nuisances /
    # aipw_pseudo_outcome call run_soldado_pipeline.py already makes for the
    # CATE-vs-alignment gate) over that condition's stim trials -- an
    # aggregation of an existing object, not a new causal estimate.
    ctrl_all = _epochs_for(corr, control_idx, 1) + (
        _epochs_for(err, control_idx, 0) if err is not None else [])
    rows_y, rows_t, rows_angle, rows_cond = [], [], [], []
    for c in cond_ids:
        stim_all = _epochs_for(corr, c, 1) + (_epochs_for(err, c, 0) if err is not None else [])
        if len(stim_all) < 5 or len(ctrl_all) < 5:
            continue
        for _, y, angle_idx in stim_all:
            rows_y.append(y); rows_t.append(1); rows_angle.append(angle_idx); rows_cond.append(c)
        for _, y, angle_idx in ctrl_all:
            rows_y.append(y); rows_t.append(0); rows_angle.append(angle_idx); rows_cond.append(c)

    valid_cond_ids = sorted(set(rows_cond))
    if not valid_cond_ids:
        return None

    y = np.array(rows_y, dtype=float)
    t = np.array(rows_t, dtype=int)
    angle_idx = np.array(rows_angle, dtype=int)
    cond_arr = np.array(rows_cond, dtype=int)
    angle_oh = np.eye(angle_idx.max() + 1)[angle_idx]

    rng = np.random.default_rng(stable_seed(f"digital_twin_{prefix}"))
    nu = crossfit_nuisances(y, t, angle_oh, n_folds=5, propensity=None, rng=rng)
    phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])

    real_effect_by_cond = {}
    for c in valid_cond_ids:
        mask = (cond_arr == c) & (t == 1)
        if mask.sum() == 0:
            continue
        real_effect_by_cond[c] = float(np.nanmean(phi[mask]))

    final_cond_ids = sorted(set(valid_cond_ids) & set(twin_displacement.keys())
                             & set(align_by_cond.keys()) & set(real_effect_by_cond.keys()))
    if not final_cond_ids:
        return None

    return {
        "session": prefix,
        "n_conditions": len(final_cond_ids),
        "twin_displacement": [twin_displacement[c] for c in final_cond_ids],
        "vstar_alignment": [align_by_cond[c] for c in final_cond_ids],
        "real_effect": [real_effect_by_cond[c] for c in final_cond_ids],
        "cond_ids": final_cond_ids,
    }


def _pearson(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _self_check() -> bool:
    """Synthetic runnable check (no data mount needed) for the comparison
    plumbing this script adds on top of fit_input_lds/simulate_input_response
    (which already have their own unit tests in tests/test_dynamics.py):
    _pearson recovers a planted strong correlation and near-zero for
    unrelated noise, and bootstrap_ci's CI brackets the planted value."""
    rng = np.random.default_rng(0)
    n = 40
    x = rng.standard_normal(n)
    y_strong = 0.9 * x + 0.1 * rng.standard_normal(n)
    y_noise = rng.standard_normal(n)

    r_strong = _pearson(np.column_stack([x, y_strong]))
    r_noise = _pearson(np.column_stack([x, y_noise]))
    ok_strong = r_strong > 0.8
    ok_noise = abs(r_noise) < 0.4

    obs, lo, hi = bootstrap_ci(np.column_stack([x, y_strong]), _pearson, n_boot=1000, rng=rng)
    ok_ci = lo < obs < hi and lo > 0.5

    return bool(ok_strong and ok_noise and ok_ci)


def main():
    assert _self_check(), "digital-twin self-check FAILED: _pearson/bootstrap_ci plumbing did not recover a planted correlation"
    per_session = {}
    all_twin, all_vstar, all_real = [], [], []
    for prefix in soldado.SESSIONS:
        print(f"  {prefix} ...", end=" ")
        try:
            res = _fit_twin_for_session(prefix)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        if res is None:
            print("SKIP (no usable amp/onset structure or insufficient data)")
            continue
        print(f"{res['n_conditions']} conditions")
        per_session[prefix] = res
        all_twin.extend(res["twin_displacement"])
        all_vstar.extend(res["vstar_alignment"])
        all_real.extend(res["real_effect"])

    if not per_session or len(all_twin) == 0:
        out = {
            "status": "infeasible",
            "reason": "No Soldado session yielded a usable input U (amp/onset structure "
                      "unresolvable or insufficient control/stim trials for any session).",
        }
        with open(RESULTS / "digital_twin.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nINFEASIBLE -- see results/digital_twin.json")
        return

    n_conditions = len(all_twin)
    twin_arr = np.array(all_twin)
    vstar_arr = np.array(all_vstar)
    real_arr = np.array(all_real)

    r_twin = _pearson(np.column_stack([twin_arr, real_arr]))
    r_vstar = _pearson(np.column_stack([vstar_arr, real_arr]))
    print(f"\nn_conditions = {n_conditions} (pooled across {len(per_session)} sessions)")
    print(f"CORRELATION twin-displacement vs real effect:  r = {r_twin:+.4f}")
    print(f"CORRELATION scalar v*-alignment vs real effect: r = {r_vstar:+.4f}")

    rng = np.random.default_rng(stable_seed("digital_twin_bootstrap"))
    r_twin_obs, r_twin_lo, r_twin_hi = bootstrap_ci(
        np.column_stack([twin_arr, real_arr]), _pearson, n_boot=5000, rng=rng)
    r_vstar_obs, r_vstar_lo, r_vstar_hi = bootstrap_ci(
        np.column_stack([vstar_arr, real_arr]), _pearson, n_boot=5000, rng=rng)

    twin_beats_baseline = abs(r_twin_obs) > abs(r_vstar_obs)
    small_n = n_conditions < 10
    print(f"Bootstrap 95% CI (condition-resampled, B=5000):")
    print(f"  twin:  r={r_twin_obs:+.4f} [{r_twin_lo:+.4f}, {r_twin_hi:+.4f}]")
    print(f"  vstar: r={r_vstar_obs:+.4f} [{r_vstar_lo:+.4f}, {r_vstar_hi:+.4f}]")
    print(f"Twin {'BEATS' if twin_beats_baseline else 'does NOT beat'} the scalar v*-alignment baseline "
          f"(|r_twin|={abs(r_twin_obs):.4f} vs |r_vstar|={abs(r_vstar_obs):.4f}).")
    if small_n:
        print(f"SMALL-n HONESTY: n_conditions={n_conditions} < ~10 -- this comparison is SUGGESTIVE, "
              f"not adjudicated.")

    out = {
        "status": "ok",
        "A_fit_source": "control_epochs",
        "n_conditions": n_conditions,
        "n_sessions_used": len(per_session),
        "onset_bin": ONSET_BIN,
        "stim_on_bins": STIM_ON_BINS,
        "twin_vs_real": {"r": r_twin_obs, "ci_lo": r_twin_lo, "ci_hi": r_twin_hi},
        "vstar_alignment_vs_real": {"r": r_vstar_obs, "ci_lo": r_vstar_lo, "ci_hi": r_vstar_hi},
        "twin_beats_baseline": bool(twin_beats_baseline),
        "small_n_suggestive_only": bool(small_n),
        "per_session": {k: {kk: vv for kk, vv in v.items()} for k, v in per_session.items()},
    }
    with open(RESULTS / "digital_twin.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/digital_twin.json")


if __name__ == "__main__":
    main()
