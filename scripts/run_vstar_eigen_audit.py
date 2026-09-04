#!/usr/bin/env python3
"""v*-eigenvalue identifiability audit across every cohort/session.

For each recording session, fits the leading (argmax-Re(lambda)) discrete-time
eigenvalue of the mean-trajectory DMD operator, then bootstraps a confidence
interval on its modulus (rho) and phase (theta) by resampling trials and
refitting -- replacing a point-estimate classification with an honest
CI-based one (most sessions turn out UNDETERMINED, which is the correct
answer, not a null result to hide).

Estimator used here: mean-trajectory + argmax-Re(lambda), the paper's
original v* convention -- deliberately NOT src/control.py's
dominant_eigenmode, which selects by modulus (argmax|lambda|) instead, per
the corrected discrete-time-stability convention now used elsewhere in this
codebase (run_divergence_analysis.dmd_divergence,
run_macaque_pfc_microstimulation_pipeline.build_session_features). This script reproduces the
paper's original convention with a bootstrap CI instead of a point estimate; a
sibling script (run_vstar_fit_selection_factorial.py) cross-checks it against
a single-trial-ensemble estimator with both argmax-Re and argmax|lambda|
selection rules.

No new DMD implementation: reuses dynamics.dmd_reconstruction_error (mean-
trajectory exact DMD, per run_divergence_analysis.dmd_divergence) and, for
the macaque PFC microstimulation uStim cohort, geometry.pca_decompose + run_macaque_pfc_microstimulation_pipeline's
own control-epoch loader (load_macaque_pfc_microstimulation_session/crop_trial), imported, not
reimplemented.

Per-dataset trial-selection/dt conventions are imported from
run_divergence_analysis.py (MAINT_T0/T1, task_id==2, set_sizes==8, loads==3,
MIN_TRIALS_K2) rather than restated.

Output: results/vstar_eigen_audit.json, keyed dataset -> session -> {
  r_use, dt, nyquist_hz, n_trials, rho, theta, sigma_s, f_hz,
  rho_ci, theta_ci, sigma_ci, f_ci, bootstrap_vstar_cos_mean,
  gap_mod, gap_re (raw-eigenvalue gap -- degenerate/0 whenever the leading
    mode is a complex-conjugate pair; kept for audit trail only),
  gap_to_next_distinct_mode_mod, gap_to_next_distinct_mode_re (corrected gap:
    conjugate pairs grouped into one mode first -- use these for anything),
  leading_mode_is_conjugate_pair, n_modes,
  aliasing_limited, latent_substrate, modulus_class, phase_class, cell,
}, plus "_meta" (bootstrap count, seed scheme, CI method, per-dataset cell counts).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_vstar_eigen_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dynamics import dmd_reconstruction_error
from geometry import pca_decompose
from statistics import stable_seed
from control import canonicalize_eigenvector_phase

import run_divergence_analysis as rda
import run_macaque_pfc_microstimulation_pipeline as rsp

RESULTS = ROOT / "results"
B_BOOT = 1000

LATENT_SUBSTRATE = {
    "miller": "HGP 70-150Hz Hilbert envelope, 50ms Gaussian smoothing (Methods 5.1-5.2)",
    "boran": "HGP 70-150Hz Hilbert envelope, 50ms Gaussian smoothing (Methods 5.1-5.2)",
    "boran_units": "single-unit spike counts, binned (Boran cohort, Methods 5.1-5.2)",
    "dandi000469": "spike times, 100ms bins, 200ms Gaussian kernel (Methods 5.1-5.2)",
    "dandi001187": "spike times, 100ms bins, 200ms Gaussian kernel (Methods 5.1-5.2)",
    "dandi000673": "spike times, 100ms bins, 200ms Gaussian kernel (Methods 5.1-5.2)",
    "macaque_pfc_microstimulation": "50ms-binned spikerate, channel-wise z-scored log-power, 30 bins (Methods 5.1-5.2)",
}


# ── Core: leading-mode (argmax-Re) mean-trajectory DMD fit ────────────────────

def leading_mode(Z_mean: np.ndarray, r: int, dt: float) -> tuple[complex, int]:
    """Argmax-Re(lambda) discrete-time eigenvalue and its phase-canonicalized
    real direction, fit on the mean-trajectory DMD operator."""
    T, d = Z_mean.shape
    r_use = min(r, d, T - 2)
    res = dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)
    A = res["A"]
    eigs, vecs = np.linalg.eig(A)
    idx = int(np.argmax(eigs.real))
    return complex(eigs[idx]), r_use, canonicalize_eigenvector_phase(vecs[:, idx])


def _grouped_mode_values(eigs: np.ndarray) -> list[tuple[float, float]]:
    """Group raw eigenvalues into physical modes: a real eigenvalue is one
    mode, a complex-conjugate pair is one mode (same pairing convention as
    control.invariant_subspace_basis: nearest-conjugate matching). Returns
    (modulus, real_part) per mode, modulus-sorted descending.

    This grouping is what the raw-eigenvalue gap below is missing: a
    conjugate pair's two eigenvalues share both modulus and real part, so
    ranking raw eigenvalues treats one physical mode as two adjacent
    'eigenvalues' with a gap of exactly 0 -- not a small gap, a
    by-construction one."""
    used: set[int] = set()
    modes = []
    for i in range(len(eigs)):
        if i in used:
            continue
        lam = eigs[i]
        if abs(lam.imag) < 1e-8 * (abs(lam.real) + 1e-12):
            used.add(i)
        else:
            j = int(np.argmin(np.abs(eigs - np.conj(lam))))
            used.add(i)
            used.add(j)
        modes.append((float(abs(lam)), float(lam.real)))
    modes.sort(key=lambda t: -t[0])
    return modes


def spectral_gaps(Z_mean: np.ndarray, r_use: int, dt: float) -> dict:
    """Two gap statistics, reported side by side:

    - `gap_mod`/`gap_re`: the raw-eigenvalue gap = (|lam1|-|lam2|)/|lam1| (and
      the Re-ranked analogue) over the modulus/Re-sorted RAW spectrum. This is
      DEGENERATE by construction whenever the leading mode is a
      complex-conjugate pair: lam2 = conj(lam1) so |lam2| = |lam1| exactly,
      giving gap_mod = 0 identically (gap_re has the same defect, since
      conjugates share a real part). Kept here, unchanged, for audit trail --
      do not use it as a predictor of anything.
    - `gap_to_next_distinct_mode_mod`/`_re`: the corrected statistic. Groups
      a conjugate pair into one mode (`_grouped_mode_values`) before ranking,
      so the gap is measured to the next mode of genuinely distinct
      modulus/real-part, never to a mode's own conjugate partner. This is the
      one to correlate against anything -- see run_vstar_identifiability.py.
    """
    res = dmd_reconstruction_error(Z_mean, r=r_use, dt=dt)
    eigs = np.linalg.eigvals(res["A"])

    mod = np.sort(np.abs(eigs))[::-1]
    gap_mod = float((mod[0] - mod[1]) / (mod[0] + 1e-300)) if len(mod) > 1 else float("nan")
    re = np.sort(eigs.real)[::-1]
    gap_re = float((re[0] - re[1]) / (abs(re[0]) + 1e-300)) if len(re) > 1 else float("nan")

    idx_leading_re = int(np.argmax(eigs.real))
    leading_mode_is_conjugate_pair = bool(
        abs(eigs[idx_leading_re].imag) > 1e-8 * (abs(eigs[idx_leading_re].real) + 1e-12))

    modes = _grouped_mode_values(eigs)
    gap_to_next_distinct_mode_mod = (
        float((modes[0][0] - modes[1][0]) / (modes[0][0] + 1e-300)) if len(modes) > 1 else float("nan"))
    modes_by_re = sorted(modes, key=lambda t: -t[1])
    gap_to_next_distinct_mode_re = (
        float((modes_by_re[0][1] - modes_by_re[1][1]) / (abs(modes_by_re[0][1]) + 1e-300))
        if len(modes_by_re) > 1 else float("nan"))

    return {
        "gap_mod": gap_mod, "gap_re": gap_re,
        "leading_mode_is_conjugate_pair": leading_mode_is_conjugate_pair,
        "gap_to_next_distinct_mode_mod": gap_to_next_distinct_mode_mod,
        "gap_to_next_distinct_mode_re": gap_to_next_distinct_mode_re,
        "n_modes": len(modes),
    }


def _circular_center(theta: np.ndarray, ref: float) -> np.ndarray:
    """Unwrap bootstrap thetas onto the branch nearest `ref` (the full-sample
    point estimate) before taking a percentile CI, so wraparound near +-pi
    doesn't distort the interval. Each draw is transformed individually
    (rather than transforming CI endpoints after the fact), which handles
    the circular case correctly."""
    return ref + (theta - ref + np.pi) % (2 * np.pi) - np.pi


def bootstrap_audit(Z_trials: np.ndarray, r: int, dt: float, seed: int,
                     B: int = B_BOOT) -> dict:
    """B trial-resamples, REFIT the mean-trajectory operator each draw (not
    resample fixed eigenvalues -- fit variability is the point), transform
    rho/theta -> sigma/f per draw, then percentile CI. Also returns the mean
    |cos| of each bootstrap v* to the full-sample v* (a bootstrap-dispersion
    diagnostic, computed here since the resampling loop already exists)."""
    rng = np.random.default_rng(seed)
    N = Z_trials.shape[0]

    lam_full, r_use, v_full = leading_mode(Z_trials.mean(0), r, dt)
    rho_full, theta_full = float(np.abs(lam_full)), float(np.angle(lam_full))
    sigma_full = float(np.log(rho_full + 1e-300) / dt)
    f_full = float(theta_full / (2 * np.pi * dt))

    rhos, thetas, sigmas, fs, coss = [], [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        try:
            lam_b, _, v_b = leading_mode(Z_trials[idx].mean(0), r, dt)
        except np.linalg.LinAlgError:
            continue
        rho_b = float(np.abs(lam_b))
        theta_b = float(np.angle(lam_b))
        rhos.append(rho_b)
        thetas.append(theta_b)
        sigmas.append(np.log(rho_b + 1e-300) / dt)
        fs.append(theta_b / (2 * np.pi * dt))  # per-draw transform, raw branch; re-centered below
        coss.append(abs(float(v_b @ v_full)))

    rhos = np.asarray(rhos)
    thetas_c = _circular_center(np.asarray(thetas), theta_full)
    fs_c = thetas_c / (2 * np.pi * dt)
    sigmas = np.asarray(sigmas)

    def _pct(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "r_use": int(r_use), "dt": float(dt), "n_trials": int(N),
        "rho": rho_full, "theta": theta_full, "sigma_s": sigma_full, "f_hz": f_full,
        "rho_ci": _pct(rhos), "theta_ci": _pct(thetas_c),
        "sigma_ci": _pct(sigmas), "f_ci": _pct(fs_c),
        "bootstrap_vstar_cos_mean": float(np.mean(coss)) if coss else float("nan"),
        "n_boot_ok": int(len(rhos)),
    }


def classify_cell(rho_ci: list[float], theta_ci: list[float]) -> dict:
    """CI-based 3x3 modulus x phase classification (replaces classifying by
    an arbitrary point-estimate tolerance): a session is only called
    EXPANDING/DECAYING or ROTATING if its bootstrap CI excludes the
    boundary; most sessions land in UNDETERMINED, which is the honest
    answer at these trial counts, not something to paper over."""
    if rho_ci[0] > 1.0:
        modulus_class = "EXPANDING"
    elif rho_ci[1] < 1.0:
        modulus_class = "DECAYING"
    else:
        modulus_class = "UNDETERMINED"
    phase_class = "ROTATING" if (theta_ci[0] > 0.0 or theta_ci[1] < 0.0) else "NON-ROTATING/UNDETERMINED"
    return {"modulus_class": modulus_class, "phase_class": phase_class,
            "cell": f"{modulus_class}+{phase_class}"}


def nyquist_flag(f_hz: float, dt: float) -> dict:
    nyq = 1.0 / (2.0 * dt)
    aliasing = bool(abs(f_hz) > 0.8 * nyq)
    return {"nyquist_hz": float(nyq), "aliasing_limited": aliasing}


# ── Self-check: planted 2x2 rotation, assert-based, no framework ────────────

def _self_check() -> None:
    def _make_trials(A, n_trials=40, T=60, seed=0):
        rng = np.random.default_rng(seed)
        d = A.shape[0]
        trials = np.zeros((n_trials, T, d))
        for i in range(n_trials):
            x0 = rng.standard_normal(d)
            traj = [x0]
            for _ in range(T - 1):
                traj.append(A @ traj[-1])
            trials[i] = np.array(traj)
        return trials

    # Case 1: rho=0.95, theta=0.3 (decaying, rotating)
    rho, theta = 0.95, 0.3
    A1 = rho * np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    Z1 = _make_trials(A1, seed=1)
    lam1, r_use1, _ = leading_mode(Z1.mean(0), r=2, dt=1.0)
    assert abs(abs(lam1) - rho) < 1e-6, f"self-check rho mismatch: {abs(lam1)} vs {rho}"
    # angle sign is eigenvector-order-arbitrary (complex-conjugate pair); compare |theta|
    assert abs(abs(np.angle(lam1)) - theta) < 1e-6, f"self-check theta mismatch: {np.angle(lam1)} vs {theta}"
    cls1 = classify_cell([rho - 0.01, rho + 0.01], [theta - 0.01, theta + 0.01])
    assert cls1["cell"] == "DECAYING+ROTATING", cls1

    # Case 2: theta=0 (real, non-rotating), rho=1.05 (expanding)
    rho2 = 1.05
    A2 = rho2 * np.eye(2)
    Z2 = _make_trials(A2, seed=2)
    lam2, r_use2, _ = leading_mode(Z2.mean(0), r=2, dt=1.0)
    assert abs(abs(lam2) - rho2) < 1e-6, f"self-check rho2 mismatch: {abs(lam2)} vs {rho2}"
    assert abs(np.angle(lam2)) < 1e-6, f"self-check theta2 mismatch: {np.angle(lam2)}"
    cls2 = classify_cell([rho2 - 0.01, rho2 + 0.01], [-0.01, 0.01])
    assert cls2["cell"] == "EXPANDING+NON-ROTATING/UNDETERMINED", cls2

    # classify_cell: CI straddling boundaries -> UNDETERMINED
    cls3 = classify_cell([0.9, 1.1], [-0.1, 0.1])
    assert cls3["modulus_class"] == "UNDETERMINED"
    assert cls3["phase_class"] == "NON-ROTATING/UNDETERMINED"

    # Case 4: a 3-D system whose leading mode is a complex-conjugate pair
    # (rho=0.9, theta=0.4) plus one distinct real mode (0.5) -- the raw
    # gap_mod/gap_re must be ~0 (degenerate, conjugate shares modulus/Re with
    # itself), while the corrected gap must recover the true separation
    # (0.9-0.5)/0.9.
    rho4, theta4, lam_real = 0.9, 0.4, 0.5
    R4 = rho4 * np.array([[np.cos(theta4), -np.sin(theta4)], [np.sin(theta4), np.cos(theta4)]])
    A4 = np.zeros((3, 3))
    A4[:2, :2] = R4
    A4[2, 2] = lam_real
    Z4 = _make_trials(A4, seed=4)
    gaps4 = spectral_gaps(Z4.mean(0), r_use=3, dt=1.0)
    assert gaps4["leading_mode_is_conjugate_pair"], gaps4
    assert abs(gaps4["gap_mod"]) < 1e-6, f"raw gap_mod should be ~0 (degenerate): {gaps4['gap_mod']}"
    assert abs(gaps4["gap_re"]) < 1e-6, f"raw gap_re should be ~0 (degenerate): {gaps4['gap_re']}"
    expected_gap_mod = (rho4 - lam_real) / rho4
    expected_gap_re = (rho4 * np.cos(theta4) - lam_real) / (rho4 * np.cos(theta4))
    assert abs(gaps4["gap_to_next_distinct_mode_mod"] - expected_gap_mod) < 1e-4, (
        f"corrected gap_mod {gaps4['gap_to_next_distinct_mode_mod']} vs expected {expected_gap_mod}")
    assert abs(gaps4["gap_to_next_distinct_mode_re"] - expected_gap_re) < 1e-4, (
        f"corrected gap_re {gaps4['gap_to_next_distinct_mode_re']} vs expected {expected_gap_re}")

    print("Self-check: PASS (planted rotation recovered to 1e-6; classifier correct; "
          "degenerate-vs-corrected spectral gap recovered on a planted 3-mode system)")


# ── Per-dataset session iterators: (dataset, session_key, Z_trials, dt, r) ───

def _iter_miller():
    for subj in rda.MILLER_SUBJECTS:
        geo = np.load(rda.RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id, times = geo["Z"], geo["task_id"], geo["times"]
        maint = (times >= rda.MAINT_T0) & (times <= rda.MAINT_T1)
        Z_trials = Z[task_id == 2][:, maint, :]
        t_maint = times[maint]
        dt = float(np.median(np.diff(t_maint)))
        yield "miller", subj, Z_trials, dt, rda.DMD_RANK


def _iter_boran():
    for subj in rda.BORAN_SUBJECTS:
        path = rda.RESULTS / f"boran_geometry_{subj}.npz"
        if not path.exists():
            continue
        geo = np.load(path, allow_pickle=True)
        Z, ss, times = geo["Z"], geo["set_sizes"], geo["times"]
        Z_trials = Z[ss == 8]
        dt = float(np.median(np.diff(times)))
        yield "boran", subj, Z_trials, dt, rda.DMD_RANK


def _iter_rutishauser():
    for subj in rda.RUSHI_SUBJECTS:
        path = rda.RESULTS / f"dandi000469_geometry_{subj}.npz"
        if not path.exists():
            continue
        geo = np.load(path, allow_pickle=True)
        Z, loads, times = geo["Z"], geo["loads"], geo["times"]
        Z_trials = Z[loads == 3]
        dt = float(np.median(np.diff(times)))
        yield "dandi000469", subj, Z_trials, dt, rda.DMD_RANK


def _iter_boran_units():
    for path in sorted(rda.RESULTS.glob("dandi000574_units_geometry_sub-*.npz")):
        key = path.stem.replace("dandi000574_units_geometry_", "")
        geo = np.load(path, allow_pickle=True)
        Z, set_size, times = geo["Z"], geo["set_size"], geo["times"]
        Z_trials = Z[set_size == 8]
        if Z_trials.shape[0] < rda.MIN_TRIALS_K2:
            continue
        dt = float(np.median(np.diff(times)))
        yield "boran_units", key, Z_trials, dt, rda.DMD_RANK


def _iter_load1v3(group: str, geom_prefix: str):
    for path in sorted(rda.RESULTS.glob(f"{geom_prefix}_sub-*.npz")):
        key = path.stem.replace(f"{geom_prefix}_", "")
        geo = np.load(path, allow_pickle=True)
        Z, loads, times = geo["Z"], geo["loads"], geo["times"]
        Z_trials = Z[loads == 3]
        if Z_trials.shape[0] < rda.MIN_TRIALS_K2:
            continue
        dt = float(np.median(np.diff(times)))
        yield group, key, Z_trials, dt, rda.DMD_RANK


def _iter_macaque_pfc_microstimulation():
    for prefix in rsp.SESSIONS:
        corr = rsp.load_macaque_pfc_microstimulation_session(prefix, correct=True)
        if corr is None or corr["control_idx"] is None:
            continue
        control_idx = corr["control_idx"]
        ctrl_epochs = [rsp.crop_trial(tr["spikerate"]) for tr in corr["trials"]
                       if tr["stim_cond"] == control_idx]
        ctrl_epochs = [e for e in ctrl_epochs if e is not None]
        if len(ctrl_epochs) < 10:
            continue
        Z_ctrl = np.stack(ctrl_epochs, axis=0)  # (N, N_BINS, C)
        C = Z_ctrl.shape[2]
        X_flat = Z_ctrl.reshape(-1, C)
        _, V, _ = pca_decompose(X_flat, rsp.N_PC)
        Z_trials = ((Z_ctrl.reshape(-1, C) - X_flat.mean(0)) @ V).reshape(
            Z_ctrl.shape[0], rsp.N_BINS, V.shape[1])
        yield "macaque_pfc_microstimulation", prefix, Z_trials, rsp.BIN_S, rsp.DMD_RANK


ALL_ITERS = [_iter_miller, _iter_boran, _iter_rutishauser, _iter_boran_units,
             lambda: _iter_load1v3("dandi001187", "dandi001187_geometry"),
             lambda: _iter_load1v3("dandi000673", "dandi000673_geometry"),
             _iter_macaque_pfc_microstimulation]


def main():
    _self_check()

    out: dict[str, dict] = {}
    cell_counts: dict[str, dict[str, int]] = {}
    n_total = 0
    for it in ALL_ITERS:
        for dataset, session, Z_trials, dt, r in it():
            out.setdefault(dataset, {})
            seed = stable_seed(f"vstar_eigen_audit_{dataset}_{session}")
            audit = bootstrap_audit(Z_trials, r, dt, seed=seed)
            gaps = spectral_gaps(Z_trials.mean(0), audit["r_use"], dt)
            nyq = nyquist_flag(audit["f_hz"], dt)
            cls = classify_cell(audit["rho_ci"], audit["theta_ci"])
            row = {**audit, **nyq, **cls, **gaps,
                   "latent_substrate": LATENT_SUBSTRATE[dataset]}
            out[dataset][session] = row
            cell_counts.setdefault(dataset, {})
            cell_counts[dataset][cls["cell"]] = cell_counts[dataset].get(cls["cell"], 0) + 1
            n_total += 1
            print(f"  {dataset}/{session}: rho={row['rho']:.4f} CI{row['rho_ci']} "
                  f"theta={row['theta']:.4f} CI{row['theta_ci']} sigma={row['sigma_s']:.4f}s^-1 "
                  f"f={row['f_hz']:.4f}Hz (Nyq={nyq['nyquist_hz']:.2f}) cell={cls['cell']}")

    macaque_pfc_microstimulation_decaying = sum(1 for s, r in out.get("macaque_pfc_microstimulation", {}).items()
                           if r["modulus_class"] == "DECAYING")
    print(f"\nTotal sessions audited: {n_total}")
    print(f"macaque PFC microstimulation DECAYING (rho CI entirely below 1): {macaque_pfc_microstimulation_decaying}/11")
    for ds, counts in cell_counts.items():
        print(f"  {ds}: {counts}")

    out["_meta"] = {
        "B": B_BOOT,
        "seed_scheme": "stable_seed(f'vstar_eigen_audit_{dataset}_{session}')",
        "ci_method": "percentile bootstrap (trial resample + refit mean-trajectory "
                     "argmax-Re DMD each draw); theta/f circularly re-centered on the "
                     "full-sample point estimate before taking percentiles",
        "estimator": "mean-trajectory + argmax-Re(lambda) (existing paper convention; "
                     "see run_vstar_fit_selection_factorial.py for the ensemble-estimator "
                     "cross-check)",
        "n_sessions_total": n_total,
        "per_dataset_cell_counts": cell_counts,
        "macaque_pfc_microstimulation_decaying_count": macaque_pfc_microstimulation_decaying,
    }
    with open(RESULTS / "vstar_eigen_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/vstar_eigen_audit.json")


if __name__ == "__main__":
    main()
