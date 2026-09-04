#!/usr/bin/env python3
"""Flow divergence ∇·v and personalized stimulation electrode selection — all 3 datasets.

For each subject across Miller (ECoG), Boran (iEEG), and Rutishauser (single-unit):
  1. Extract mean high-load maintenance trajectory Z_mean(t).
  2. Fit DMD on Z_mean → linear operator A  (x(t+1) ≈ A x(t)).
  3. Divergence of the flow field: ∇·v = trace(A − I) / dt  [s⁻¹].
     Negative → contracting attractor; positive → expanding / unstable.
  4. Dominant unstable direction v* = eigvec of A with largest Re(eigenvalue).
     This is the direction perturbations grow fastest along.

For Miller and Boran (TES1 DLPFC coverage exists):
  5. Compute alignment of each TES1 B_i with v*: align_i = |cos(B_i, v*)|.
     Dynamic optimal donor = argmax_i align_i  (subject-specific, state-dependent).
     Compare to static best-by-Gramian (current approach).

For Rutishauser: divergence only (MTL/medial frontal outside TES1 DLPFC coverage).

Saves: results/divergence_analysis.npz, updates all_statistics.json

Run from project root:
    conda run -n wm_dynamics python scripts/run_divergence_analysis.py
"""
from __future__ import annotations
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dynamics import (dmd_reconstruction_error, ensemble_dmd, divergence_rank_sweep,
                      mean_trajectory_divergence_rank_sweep, rank_robustness_sign,
                      trajectory_tangling)
from provenance import _json_safe

RESULTS = ROOT / "results"

MILLER_SUBJECTS   = ["al", "ca", "cc", "ug"]
BORAN_SUBJECTS    = [f"sub-{i:02d}" for i in range(1, 10)]
RUSHI_SUBJECTS    = [f"sub-{n}" for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,21]]

DMD_RANK    = 8
RANK_SWEEP  = (5, 6, 7, 8)   # truncation ranks below full rank (d=8); audit item 1b
MAINT_T0  = 0.30   # Miller: maintenance onset (s)
MAINT_T1  = 1.40   # Miller: maintenance offset (s)


# ── Core computation ──────────────────────────────────────────────────────────

def dmd_divergence(Z_mean: np.ndarray, dt: float, r: int = DMD_RANK) -> dict:
    """Fit DMD on mean trajectory; return divergence and unstable direction.

    Parameters
    ----------
    Z_mean : (T, d)
    dt     : time step in seconds

    Returns
    -------
    dict with: A, div_scalar, max_re_eig, v_star, eigenvalues, r2
    """
    T, d = Z_mean.shape
    r_use = min(r, d, T - 2)
    res  = dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)
    A    = res["A"]

    eigs, vecs = np.linalg.eig(A)
    idx = np.argsort(eigs.real)[::-1]
    eigs = eigs[idx];  vecs = vecs[:, idx]

    v_star = vecs[:, 0].real
    norm   = np.linalg.norm(v_star)
    v_star = v_star / (norm + 1e-12)

    # True continuous-time flow divergence Σ log|λᵢ|/dt, not the first-order
    # trace(A-I)/dt approximation, which is contaminated by rotation (nonzero
    # phase eigenvalues near the unit circle register as spurious "contraction").
    div_scalar = float(np.sum(np.log(np.abs(eigs) + 1e-300))) / dt

    return {
        "A":           A,
        "div_scalar":  div_scalar,
        "max_re_eig":  float(eigs[0].real),
        "v_star":      v_star,
        "eigenvalues": eigs,
        "r2":          float(res["r_squared"]),
    }


def sliding_window_divergence(Z_mean: np.ndarray, times: np.ndarray, dt: float,
                               win_s: float = 0.25, step_s: float = 0.05,
                               r: int = DMD_RANK) -> dict:
    """Time-resolved ∇·v(t): local DMD fit in a sliding window.

    The whole-trajectory divergence (dmd_divergence) gives one scalar for
    the entire maintenance regime. This asks WHEN within that regime the
    local flow is most contracting (safest, self-correcting — a nudge is
    least likely to be amplified) vs most expanding (highest control
    leverage per unit stimulation energy, but also higher risk of runaway
    drift if mistimed). Both are candidate optimal-timing signals for
    closed-loop stimulation and are reported together.

    Parameters
    ----------
    Z_mean : (T, d) mean latent trajectory
    times  : (T,) time vector matching Z_mean
    dt     : time step (s)
    win_s  : window length (s)
    step_s : step between window centers (s)
    r      : DMD rank (must match latent dimensionality; see band_dmd_divergence)

    Returns
    -------
    dict with: t_centers, div_trace, r2_trace, t_min_div, min_div, t_max_div, max_div
    """
    T, d = Z_mean.shape
    win  = max(int(round(win_s / dt)), d + 2)
    step = max(int(round(step_s / dt)), 1)

    t_centers, div_trace, r2_trace = [], [], []
    for start in range(0, T - win, step):
        seg = Z_mean[start:start + win]
        r_use = min(r, d, seg.shape[0] - 2)
        res = dmd_reconstruction_error(seg, r=r_use, dt=dt)
        A = res["A"]
        eigs_seg = np.linalg.eigvals(A)
        div_trace.append(float(np.sum(np.log(np.abs(eigs_seg) + 1e-300))) / dt)
        r2_trace.append(float(res["r_squared"]))
        t_centers.append(float(times[start:start + win].mean()))

    t_centers = np.array(t_centers)
    div_trace = np.array(div_trace)
    r2_trace  = np.array(r2_trace)

    i_min = int(np.argmin(div_trace))
    i_max = int(np.argmax(div_trace))

    return {
        "t_centers": t_centers,
        "div_trace": div_trace,
        "r2_trace":  r2_trace,
        "t_min_div": float(t_centers[i_min]), "min_div": float(div_trace[i_min]),
        "t_max_div": float(t_centers[i_max]), "max_div": float(div_trace[i_max]),
    }


def electrode_alignment(B_per_tes1: np.ndarray, v_star: np.ndarray) -> np.ndarray:
    """Return |cos θ_i| for each TES1 donor B_i against v*.

    Parameters
    ----------
    B_per_tes1 : (17, d, 1)
    v_star     : (d,)

    Returns
    -------
    align : (17,)
    """
    B = B_per_tes1[:, :, 0]                             # (17, d)
    B_hat = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    v_hat = v_star / (np.linalg.norm(v_star) + 1e-12)
    return np.abs(B_hat @ v_hat)


# ── Per-dataset processors ────────────────────────────────────────────────────

def _ensemble_check(out: dict, key: str, Z_trials: np.ndarray, dt: float, dmd: dict) -> dict:
    """Fit ensemble (single-trial) DMD alongside the mean-trajectory fit.

    A trial-averaged mean fit at rank 8 hits near-perfect in-sample R^2 by
    construction and is confounded by trial-averaging contraction (jittered
    single trials can average into an artificially smooth, contracting
    mean). Ensemble DMD pools every trial's own (x_t -> x_t+1) transitions
    and is validated out-of-sample, with a circular-shift null.
    """
    ens = ensemble_dmd(Z_trials, r=DMD_RANK, dt=dt, n_splits=5, n_null=30,
                        rng=np.random.default_rng(0))
    out[f"{key}_ensemble_div_scalar"] = np.array(ens["div_scalar"])
    out[f"{key}_ensemble_r2_insample"] = np.array(ens["r2_insample"])
    out[f"{key}_ensemble_r2_cv"]       = np.array(ens["r2_cv"])
    out[f"{key}_ensemble_r2_null"]     = np.array(ens["r2_null"])
    print(f"      ensemble DMD: div={ens['div_scalar']:.4f} s⁻¹ (mean-traj div={dmd['div_scalar']:.4f} s⁻¹)  "
          f"R²_insample={ens['r2_insample']:.4f}  R²_cv={ens['r2_cv']:.4f}±{ens['r2_cv_std']:.4f}  "
          f"R²_null={ens['r2_null']:.4f}±{ens['r2_null_std']:.4f}  (mean-traj R²={dmd['r2']:.6f})")

    rank_sweep = rank_robustness_check(Z_trials, dt)
    out[f"{key}_rank_sweep_ranks"]       = np.array(rank_sweep["ranks"])
    out[f"{key}_rank_sweep_mean_div"]    = np.array(rank_sweep["mean_div"])
    out[f"{key}_rank_sweep_ensemble_div"] = np.array(rank_sweep["ensemble_div"])
    out[f"{key}_rank_sweep_sign_robust"] = np.array(rank_sweep["sign_robust"])
    print(f"      rank sweep r={rank_sweep['ranks']}: mean_div={['%.3f' % v for v in rank_sweep['mean_div']]}  "
          f"ensemble_div={['%.3f' % v for v in rank_sweep['ensemble_div']]}  "
          f"sign_robust={rank_sweep['sign_robust']}")

    return {
        "ensemble_div_scalar": ens["div_scalar"],
        "ensemble_r2_insample": ens["r2_insample"],
        "ensemble_r2_cv": ens["r2_cv"],
        "ensemble_r2_cv_std": ens["r2_cv_std"],
        "ensemble_r2_null": ens["r2_null"],
        "ensemble_r2_null_std": ens["r2_null_std"],
        "rank_sweep_ranks": rank_sweep["ranks"],
        "rank_sweep_mean_div": rank_sweep["mean_div"],
        "rank_sweep_ensemble_div": rank_sweep["ensemble_div"],
        "rank_sweep_sign_robust": rank_sweep["sign_robust"],
    }


def rank_robustness_check(Z_trials: np.ndarray, dt: float) -> dict:
    """Recompute the single-trial-ensemble-vs-trial-mean divergence contrast
    (the R2 claim) at truncation ranks below full rank, and check the sign of
    the contrast (ensemble_div - mean_div) is stable across ranks — i.e. the
    claim is not an artefact of fitting DMD at r == d (log-det edge case).
    """
    Z_mean = Z_trials.mean(0)
    mean_sweep = mean_trajectory_divergence_rank_sweep(Z_mean, dt, ranks=RANK_SWEEP)
    ens_sweep  = divergence_rank_sweep(Z_trials, dt, ranks=RANK_SWEEP, n_splits=5, n_null=10,
                                        rng=np.random.default_rng(0))
    contrast = np.array(ens_sweep["div_scalar"]) - np.array(mean_sweep["div_scalar"])
    return {
        "ranks": ens_sweep["ranks"],
        "mean_div": mean_sweep["div_scalar"],
        "ensemble_div": ens_sweep["div_scalar"],
        "sign_robust": rank_robustness_sign(contrast),
    }


def process_miller(out: dict, all_rows: dict, tes1):
    print("  Miller ECoG:")
    for subj in MILLER_SUBJECTS:
        geo   = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z     = geo["Z"]; task_id = geo["task_id"]; times = geo["times"]

        maint   = (times >= MAINT_T0) & (times <= MAINT_T1)
        Z_trials_2back = Z[task_id == 2][:, maint, :]
        Z_mean  = Z_trials_2back.mean(0)
        t_maint = times[maint]
        dt      = float(np.median(np.diff(t_maint)))

        dmd = dmd_divergence(Z_mean, dt)
        ens_row = _ensemble_check(out, f"miller_{subj}", Z_trials_2back, dt, dmd)
        v_star = dmd["v_star"]
        sw = sliding_window_divergence(Z_mean, t_maint, dt)

        B_per_tes1   = tes1[f"{subj}_B_latent_per_tes1"]
        gramians     = tes1[f"{subj}_gramian_traces"]
        align        = electrode_alignment(B_per_tes1, v_star)
        static_idx   = int(np.argmax(gramians))
        dynamic_idx  = int(np.argmax(align))
        align_gain   = float(align[dynamic_idx]) / (float(align[static_idx]) + 1e-12)

        _save_subj(out, f"miller_{subj}", dmd, t_maint, align,
                   static_idx, dynamic_idx, align_gain, has_tes1=True)
        _save_timing(out, f"miller_{subj}", sw)
        _print_subj("Miller", subj, dmd, static_idx, dynamic_idx,
                    align, align_gain)
        _print_timing("Miller", subj, sw)
        all_rows["miller"][subj] = _row(dmd, static_idx, dynamic_idx,
                                        align[static_idx], align[dynamic_idx],
                                        align_gain)
        all_rows["miller"][subj].update(_timing_row(sw))
        all_rows["miller"][subj].update(ens_row)


def process_boran(out: dict, all_rows: dict, tes1_boran):
    print("  Boran iEEG:")
    for subj in BORAN_SUBJECTS:
        geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
        Z   = geo["Z"]; ss = geo["set_sizes"]; times = geo["times"]

        # full epoch is the maintenance window for Boran (0–3 s)
        Z_trials_ss8 = Z[ss == 8]
        Z_mean = Z_trials_ss8.mean(0)                 # (T, 8)
        dt     = float(np.median(np.diff(times)))

        dmd    = dmd_divergence(Z_mean, dt)
        ens_row = _ensemble_check(out, f"boran_{subj}", Z_trials_ss8, dt, dmd)
        v_star = dmd["v_star"]
        sw     = sliding_window_divergence(Z_mean, times, dt)

        B_per_tes1 = tes1_boran[f"{subj}_B_latent_per_tes1"]
        gramians   = tes1_boran[f"{subj}_gramian_traces"]
        align      = electrode_alignment(B_per_tes1, v_star)
        static_idx  = int(np.argmax(gramians))
        dynamic_idx = int(np.argmax(align))
        align_gain  = float(align[dynamic_idx]) / (float(align[static_idx]) + 1e-12)

        _save_subj(out, f"boran_{subj}", dmd, times, align,
                   static_idx, dynamic_idx, align_gain, has_tes1=True)
        _save_timing(out, f"boran_{subj}", sw)
        _print_subj("Boran", subj, dmd, static_idx, dynamic_idx,
                    align, align_gain)
        _print_timing("Boran", subj, sw)
        all_rows["boran"][subj] = _row(dmd, static_idx, dynamic_idx,
                                       align[static_idx], align[dynamic_idx],
                                       align_gain)
        all_rows["boran"][subj].update(_timing_row(sw))
        all_rows["boran"][subj].update(ens_row)


def process_rutishauser(out: dict, all_rows: dict):
    print("  Rutishauser single-unit:")
    for subj in RUSHI_SUBJECTS:
        path = RESULTS / f"dandi000469_geometry_{subj}.npz"
        if not path.exists():
            continue
        geo   = np.load(path, allow_pickle=True)
        Z     = geo["Z"]; loads = geo["loads"]; times = geo["times"]

        Z_trials_l3 = Z[loads == 3]
        Z_mean = Z_trials_l3.mean(0)                  # (30, 8)
        dt     = float(np.median(np.diff(times)))

        dmd = dmd_divergence(Z_mean, dt)
        ens_row = _ensemble_check(out, f"dandi000469_{subj}", Z_trials_l3, dt, dmd)

        _save_subj(out, f"dandi000469_{subj}", dmd, times,
                   align=None, static_idx=None, dynamic_idx=None,
                   align_gain=None, has_tes1=False)
        print(f"    {subj}: div={dmd['div_scalar']:.4f} s⁻¹  "
              f"max_Re(λ)={dmd['max_re_eig']:.6f}  R²={dmd['r2']:.4f}")
        all_rows["rutishauser"][subj] = _row(dmd)
        all_rows["rutishauser"][subj].update(ens_row)


# ── Previously-uncovered cohorts (Boran units, DANDI 001187, DANDI 000673) ──
# Growth rate + tangling at DMD_RANK=8 (this module's existing full-latent-rank
# convention -- ALL of Miller/Boran-iEEG/Rutishauser above are fit the same way,
# so keeping these three cohorts at the same rank is what makes them poolable
# into the SAME forest as their siblings via aggregate_forest_syntheses.py; see
# agent_report.md for why this differs from the Soldago-benchmark r=7 in STEP A.
# r=7 is still reported per-cohort via the EXISTING RANK_SWEEP=(5,6,7,8)
# rank-robustness mechanism below (rank_robustness_check), not by truncating
# the primary fit differently from its siblings. No LQR/TES1 (MTL, no TES1
# DLPFC coverage -- DATASET_ANALYSIS_MATRIX.md principled exclusion #2).
MIN_TRIALS_K2 = 10


def _rotation_freq_hz(eigenvalues: np.ndarray, dt: float) -> float:
    """Dominant (largest-magnitude) discrete eigenvalue's imaginary part, in Hz
    -- same convention as run_axis_rotation_analysis.rotation_frequency_hz."""
    dominant = eigenvalues[np.argmax(np.abs(eigenvalues))]
    omega = np.log(dominant + 1e-300) / dt
    return float(np.abs(omega.imag) / (2 * np.pi))


def process_boran_units(out: dict, all_rows: dict):
    print("  Boran units (single-unit):")
    for path in sorted(RESULTS.glob("dandi000574_units_geometry_sub-*.npz")):
        key = path.stem.replace("dandi000574_units_geometry_", "")
        geo = np.load(path, allow_pickle=True)
        Z, set_size, times = geo["Z"], geo["set_size"], geo["times"]

        Z_trials_ss8 = Z[set_size == 8]
        if Z_trials_ss8.shape[0] < MIN_TRIALS_K2:
            print(f"    SKIP {key} -- only {Z_trials_ss8.shape[0]} set-8 trials (<{MIN_TRIALS_K2})")
            continue
        Z_mean = Z_trials_ss8.mean(0)
        dt = float(np.median(np.diff(times)))

        dmd = dmd_divergence(Z_mean, dt)
        ens_row = _ensemble_check(out, f"boran_units_{key}", Z_trials_ss8, dt, dmd)
        rot_hz = _rotation_freq_hz(dmd["eigenvalues"], dt)
        tangling_mean = float(trajectory_tangling(Z_mean, dt=dt).mean())

        _save_subj(out, f"boran_units_{key}", dmd, times,
                   align=None, static_idx=None, dynamic_idx=None,
                   align_gain=None, has_tes1=False)
        print(f"    {key}: div={dmd['div_scalar']:.4f} s⁻¹  max_Re(λ)={dmd['max_re_eig']:.6f}  "
              f"rot={rot_hz:.4f} Hz  tangling={tangling_mean:.4f}  R²={dmd['r2']:.4f}  "
              f"n_trials={Z_trials_ss8.shape[0]}")
        all_rows["boran_units"][key] = _row(dmd)
        all_rows["boran_units"][key].update(ens_row)
        all_rows["boran_units"][key].update(rotation_freq_hz=rot_hz, tangling_mean=tangling_mean,
                                            n_trials=int(Z_trials_ss8.shape[0]))


def _process_load1v3_dynamics(out: dict, all_rows: dict, group: str, geom_prefix: str):
    """Shared loader for DANDI 001187 / 000673: load-1-vs-load-3 (context)
    high-load-trajectory dynamics, mirroring process_rutishauser's load-3-only
    convention for the sibling Rutishauser-lineage cohort (DANDI 000469)."""
    for path in sorted(RESULTS.glob(f"{geom_prefix}_sub-*.npz")):
        key = path.stem.replace(f"{geom_prefix}_", "")
        geo = np.load(path, allow_pickle=True)
        Z, loads, times = geo["Z"], geo["loads"], geo["times"]

        Z_trials_l3 = Z[loads == 3]
        if Z_trials_l3.shape[0] < MIN_TRIALS_K2:
            print(f"    SKIP {key} -- only {Z_trials_l3.shape[0]} load-3 trials (<{MIN_TRIALS_K2})")
            continue
        Z_mean = Z_trials_l3.mean(0)
        dt = float(np.median(np.diff(times)))

        dmd = dmd_divergence(Z_mean, dt)
        ens_row = _ensemble_check(out, f"{group}_{key}", Z_trials_l3, dt, dmd)
        rot_hz = _rotation_freq_hz(dmd["eigenvalues"], dt)
        tangling_mean = float(trajectory_tangling(Z_mean, dt=dt).mean())

        _save_subj(out, f"{group}_{key}", dmd, times,
                   align=None, static_idx=None, dynamic_idx=None,
                   align_gain=None, has_tes1=False)
        print(f"    {key}: div={dmd['div_scalar']:.4f} s⁻¹  max_Re(λ)={dmd['max_re_eig']:.6f}  "
              f"rot={rot_hz:.4f} Hz  tangling={tangling_mean:.4f}  R²={dmd['r2']:.4f}  "
              f"n_trials={Z_trials_l3.shape[0]}")
        all_rows[group][key] = _row(dmd)
        all_rows[group][key].update(ens_row)
        all_rows[group][key].update(rotation_freq_hz=rot_hz, tangling_mean=tangling_mean,
                                    n_trials=int(Z_trials_l3.shape[0]))


def process_dandi001187(out: dict, all_rows: dict):
    print("  DANDI 001187 (single-unit):")
    _process_load1v3_dynamics(out, all_rows, "dandi001187", "dandi001187_geometry")


def process_dandi000673(out: dict, all_rows: dict):
    print("  DANDI 000673 (single-unit + LFP):")
    _process_load1v3_dynamics(out, all_rows, "dandi000673", "dandi000673_geometry")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_subj(out, key, dmd, times, align, static_idx, dynamic_idx,
               align_gain, has_tes1):
    out[f"{key}_t"]           = times
    out[f"{key}_A"]           = dmd["A"]
    out[f"{key}_v_star"]      = dmd["v_star"]
    out[f"{key}_eigenvalues"] = dmd["eigenvalues"]
    out[f"{key}_div_scalar"]  = np.array(dmd["div_scalar"])
    out[f"{key}_max_re_eig"]  = np.array(dmd["max_re_eig"])
    out[f"{key}_r2"]          = np.array(dmd["r2"])
    if has_tes1:
        out[f"{key}_alignment"]        = align
        out[f"{key}_static_best_idx"]  = np.array(static_idx)
        out[f"{key}_dynamic_best_idx"] = np.array(dynamic_idx)
        out[f"{key}_align_gain"]       = np.array(align_gain)


def _save_timing(out, key, sw):
    out[f"{key}_sw_t_centers"] = sw["t_centers"]
    out[f"{key}_sw_div_trace"] = sw["div_trace"]
    out[f"{key}_sw_r2_trace"]  = sw["r2_trace"]
    out[f"{key}_t_min_div"]    = np.array(sw["t_min_div"])
    out[f"{key}_min_div"]      = np.array(sw["min_div"])
    out[f"{key}_t_max_div"]    = np.array(sw["t_max_div"])
    out[f"{key}_max_div"]      = np.array(sw["max_div"])


def _print_timing(dataset, subj, sw):
    print(f"      timing: most-contracting window centered at t={sw['t_min_div']:.3f}s "
          f"(∇·v={sw['min_div']:.2f} s⁻¹); most-expanding at t={sw['t_max_div']:.3f}s "
          f"(∇·v={sw['max_div']:.2f} s⁻¹)")


def _timing_row(sw):
    return {
        "t_min_div": sw["t_min_div"], "min_div": sw["min_div"],
        "t_max_div": sw["t_max_div"], "max_div": sw["max_div"],
    }


def _row(dmd, static_idx=None, dynamic_idx=None, static_align=None,
         dynamic_align=None, align_gain=None):
    r = {
        "div_scalar":  float(dmd["div_scalar"]),
        "max_re_eig":  float(dmd["max_re_eig"]),
        "dmd_r2":      float(dmd["r2"]),
    }
    if static_idx is not None:
        r.update({
            "static_best_donor":  int(static_idx),
            "dynamic_best_donor": int(dynamic_idx),
            "static_align":       float(static_align),
            "dynamic_align":      float(dynamic_align),
            "align_gain_x":       float(align_gain),
        })
    return r


def _print_subj(dataset, subj, dmd, static_idx, dynamic_idx, align, gain):
    print(f"    {dataset}/{subj}: div={dmd['div_scalar']:.4f} s⁻¹  "
          f"max_Re(λ)={dmd['max_re_eig']:.6f}  R²={dmd['r2']:.4f}")
    print(f"      static-best donor={static_idx} "
          f"(align={align[static_idx]:.3f})  "
          f"dynamic-best={dynamic_idx} "
          f"(align={align[dynamic_idx]:.3f})  gain={gain:.2f}×")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tes1       = np.load(RESULTS / "tes1_comprehensive.npz", allow_pickle=True)
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz",       allow_pickle=True)
    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        all_stats = json.load(f)

    out      = {}
    all_rows = {"miller": {}, "boran": {}, "rutishauser": {},
                "boran_units": {}, "dandi001187": {}, "dandi000673": {}}

    process_miller(out, all_rows, tes1)
    process_boran(out, all_rows, tes1_boran)
    process_rutishauser(out, all_rows)
    process_boran_units(out, all_rows)
    process_dandi001187(out, all_rows)
    process_dandi000673(out, all_rows)

    np.savez(RESULTS / "divergence_analysis.npz", **out)
    print("\n  Saved: results/divergence_analysis.npz")

    # Cross-dataset summary
    miller_divs = [all_rows["miller"][s]["div_scalar"]       for s in MILLER_SUBJECTS]
    boran_divs  = [all_rows["boran"][s]["div_scalar"]
                   for s in BORAN_SUBJECTS if s in all_rows["boran"]]
    rushi_divs  = [all_rows["rutishauser"][s]["div_scalar"]
                   for s in RUSHI_SUBJECTS if s in all_rows["rutishauser"]]

    # Previously-uncovered cohorts
    bu_divs   = [v["div_scalar"] for v in all_rows["boran_units"].values()]
    d1187_divs = [v["div_scalar"] for v in all_rows["dandi001187"].values()]
    d0673_divs = [v["div_scalar"] for v in all_rows["dandi000673"].values()]

    print(f"\n  Cross-dataset divergence (mean ± SD) [s⁻¹]:")
    print(f"    Miller: {np.mean(miller_divs):.4f} ± {np.std(miller_divs):.4f}")
    print(f"    Boran:  {np.mean(boran_divs):.4f}  ± {np.std(boran_divs):.4f}")
    print(f"    Rushi.: {np.mean(rushi_divs):.4f}  ± {np.std(rushi_divs):.4f}")
    if bu_divs:
        print(f"    Boran units:   {np.mean(bu_divs):.4f} ± {np.std(bu_divs):.4f} (N={len(bu_divs)})")
    if d1187_divs:
        print(f"    DANDI 001187:  {np.mean(d1187_divs):.4f} ± {np.std(d1187_divs):.4f} (N={len(d1187_divs)})")
    if d0673_divs:
        print(f"    DANDI 000673:  {np.mean(d0673_divs):.4f} ± {np.std(d0673_divs):.4f} (N={len(d0673_divs)})")

    # Boran units (spiking) vs Boran iEEG (LFP) dynamics, within
    # the same subjects/trials -- paired on shared subjects, same design as the
    # axis-rotation comparison in run_axis_rotation_analysis.py.
    from statistics import paired_sign_flip_test
    bu_by_subj: dict[str, list] = {}
    for key, v in all_rows["boran_units"].items():
        subj = key.split("_ses-")[0]
        bu_by_subj.setdefault(subj, []).append(v)
    bu_subj_div = {s: float(np.mean([r["div_scalar"] for r in rs])) for s, rs in bu_by_subj.items()}
    bu_subj_rot = {s: float(np.mean([r["rotation_freq_hz"] for r in rs])) for s, rs in bu_by_subj.items()}
    shared = sorted(set(bu_subj_div) & {s for s in BORAN_SUBJECTS if s in all_rows["boran"]})
    k3_dynamics = None
    if len(shared) >= 4:
        div_bu = np.array([bu_subj_div[s] for s in shared])
        div_ie = np.array([all_rows["boran"][s]["div_scalar"] for s in shared])
        res_div = paired_sign_flip_test(div_bu, div_ie, n_perm=10000, alternative="two-sided",
                                        rng=np.random.default_rng(3))
        k3_dynamics = {
            "n_subjects": len(shared), "subjects": shared,
            "div_scalar": {"units_mean": float(div_bu.mean()), "ieeg_mean": float(div_ie.mean()),
                          "mean_diff": res_div["mean_diff"], "ci_lower": res_div["ci_lower"],
                          "ci_upper": res_div["ci_upper"], "p_value": res_div["p_value"]},
        }
        print(f"\n  Spiking-vs-LFP div_scalar (Boran units vs Boran iEEG, N={len(shared)}): "
              f"units={div_bu.mean():.4f} ieeg={div_ie.mean():.4f} diff={res_div['mean_diff']:+.4f} "
              f"[{res_div['ci_lower']:.4f},{res_div['ci_upper']:.4f}] p={res_div['p_value']:.4f} "
              f"({'AGREE' if res_div['p_value'] >= 0.05 else 'DIVERGE'})")
    else:
        print(f"\n  Spiking-vs-LFP div_scalar: only {len(shared)} shared subjects (<4) -- "
              f"underpowered, STOP-and-report, not computed.")
    all_stats["divergence_spiking_vs_lfp_boran"] = k3_dynamics

    # Cross-dataset electrode alignment gain summary
    miller_gains = [all_rows["miller"][s]["align_gain_x"] for s in MILLER_SUBJECTS]
    boran_gains  = [all_rows["boran"][s]["align_gain_x"]
                    for s in BORAN_SUBJECTS if s in all_rows["boran"]]
    print(f"\n  Dynamic vs static electrode alignment gain (mean ± SD):")
    print(f"    Miller: {np.mean(miller_gains):.2f}× ± {np.std(miller_gains):.2f}×")
    print(f"    Boran:  {np.mean(boran_gains):.2f}×  ± {np.std(boran_gains):.2f}×")

    # Cross-subject timing summary (time-resolved sliding-window divergence)
    print(f"\n  Optimal stimulation timing (sliding-window ∇·v, N.B. Rutishauser "
          f"excluded — only 30 samples/trial, too coarse to window):")
    for label, subs in [("Miller", MILLER_SUBJECTS), ("Boran", BORAN_SUBJECTS)]:
        key = label.lower()
        t_mins = [all_rows[key][s]["t_min_div"] for s in subs if s in all_rows[key]]
        t_maxs = [all_rows[key][s]["t_max_div"] for s in subs if s in all_rows[key]]
        print(f"    {label}: most-contracting window at t={np.mean(t_mins):.3f}"
              f"±{np.std(t_mins):.3f}s; most-expanding at t={np.mean(t_maxs):.3f}"
              f"±{np.std(t_maxs):.3f}s (N={len(t_mins)})")

    # Rank-robustness summary: is the single-trial-ensemble-vs-mean divergence
    # contrast (the R2 claim) stable away from full-rank DMD (audit item 1b)?
    robust_flags = [row["rank_sweep_sign_robust"]
                    for ds in all_rows.values() for row in ds.values()
                    if "rank_sweep_sign_robust" in row]
    n_robust = int(np.sum(robust_flags))
    print(f"\n  Rank robustness (r={RANK_SWEEP}) of ensemble-vs-mean divergence contrast: "
          f"{n_robust}/{len(robust_flags)} subjects sign-stable")
    all_stats["divergence_rank_robustness"] = {
        "ranks_swept": list(RANK_SWEEP),
        "n_robust": n_robust,
        "n_total": len(robust_flags),
        "all_robust": bool(n_robust == len(robust_flags)),
    }

    all_stats["divergence"] = all_rows
    with open(stats_path, "w") as f:
        json.dump(_json_safe(all_stats), f, indent=2, allow_nan=False)
    print("  Updated all_statistics.json")


if __name__ == "__main__":
    print("Computing ∇·v (DMD) — all 3 datasets...")
    main()
    print("Done.")
