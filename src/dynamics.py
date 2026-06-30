"""
dynamics.py — Dynamical systems analysis: DMD, trajectory tangling.

Implements:
  - Trajectory tangling Q(t) from Russo et al. 2018 (Neuron)
  - Exact DMD following Tu et al. 2014 (J Comput Dyn)
  - Per-trial eigenspectrum analysis for system stability assessment

References
----------
Russo AA et al. (2018) Motor cortex embeds muscle-like commands in an
  untangled population response. Neuron 97(4):953-66.
Tu JH et al. (2014) On dynamic mode decomposition: theory and applications.
  J Comput Dyn 1(2):391-421.
Brunton SL & Kutz JN (2022) Data-Driven Science and Engineering: Machine
  Learning, Dynamical Systems, and Control. Cambridge Univ. Press (Ch 7).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ── Trajectory tangling ─────────────────────────────────────────────────────────

def velocity_field(Z: NDArray, dt: float = 1.0) -> NDArray:
    """Central finite-difference velocity estimate of a trajectory.

    Ż[t] ≈ (Z[t+1] - Z[t-1]) / (2Δt),   for 1 ≤ t ≤ T-2
    Endpoints are set to zero.

    Parameters
    ----------
    Z  : (T, d) — latent trajectory
    dt : time step in seconds (default 1 sample; scale Q values accordingly)

    Returns
    -------
    Zdot : (T, d)
    """
    Zdot = np.zeros_like(Z)
    Zdot[1:-1] = (Z[2:] - Z[:-2]) / (2.0 * dt)
    return Zdot


def trajectory_tangling(
    Z: NDArray, epsilon: float = 1e-3, dt: float = 1.0
) -> NDArray:
    """Trajectory tangling metric Q(t) from Russo et al. 2018.

    Q(t) = max_{t'} ‖Ż(t) - Ż(t')‖² / (‖Z(t) - Z(t')‖² + ε)

    Q is high when two time points have similar states but divergent velocities —
    the hallmark of dynamical instability. Russo et al. show motor cortex
    has low Q throughout a movement cycle (untangled flow field), enabling
    noise-robust execution.

    Parameters
    ----------
    Z       : (T, d)
    epsilon : regularizer; prevents blow-up when states coincide (1e-3 ≈ 0.1% of typical ‖Z‖²)
    dt      : time step (affects Ż magnitude but not Q ordering)

    Returns
    -------
    Q : (T,) — tangling at each time step (0 at endpoints)
    """
    T = Z.shape[0]
    Zdot = velocity_field(Z, dt)
    Q = np.zeros(T)

    # Fully vectorised: build (T, T) pairwise matrices at once.
    # Memory: 2 × T² × d floats — fine for T ≤ 3000, d ≤ 16 (< 1 GB).
    # For very long recordings fall back to the loop (guarded by T > 2000).
    if T <= 2000:
        dstate = Z[:, None, :] - Z[None, :, :]        # (T, T, d)
        dvel   = Zdot[:, None, :] - Zdot[None, :, :]  # (T, T, d)
        num = (dvel**2).sum(axis=2)                    # (T, T)
        den = (dstate**2).sum(axis=2) + epsilon        # (T, T)
        ratio = num / den                               # (T, T)
        Q[1:-1] = ratio[1:-1].max(axis=1)
    else:
        for t in range(1, T - 1):
            dstate = Z - Z[t]
            dvel   = Zdot - Zdot[t]
            num = (dvel**2).sum(axis=1)
            den = (dstate**2).sum(axis=1) + epsilon
            Q[t] = (num / den).max()

    return Q


def trial_tangling(
    Z_trials: NDArray,
    epsilon: float = 1e-3,
    dt: float = 1.0 / 1200,
) -> NDArray:
    """Compute Q(t) for each trial independently.

    Parameters
    ----------
    Z_trials : (N, T, d) — latent trajectories for N trials

    Returns
    -------
    Q_trials : (N, T)
    """
    N, T, d = Z_trials.shape
    Q = np.zeros((N, T))
    for i in range(N):
        Q[i] = trajectory_tangling(Z_trials[i], epsilon=epsilon, dt=dt)
    return Q


def mean_tangling_timecourse(
    Q_trials: NDArray,
    mask: NDArray,
) -> tuple[NDArray, NDArray]:
    """Mean and SEM of Q(t) across a set of trials.

    Parameters
    ----------
    Q_trials : (N, T)
    mask     : (N,) bool

    Returns
    -------
    mean : (T,)
    sem  : (T,)
    """
    Q_sub = Q_trials[mask]
    return Q_sub.mean(axis=0), Q_sub.std(axis=0) / np.sqrt(mask.sum())


# ── Dynamic Mode Decomposition ─────────────────────────────────────────────────

def exact_dmd(
    X: NDArray,
    r: int | None = None,
    dt: float = 1.0,
) -> dict:
    """Exact DMD (Tu et al. 2014) on a state matrix X.

    Fits the linear system: X' ≈ A X  (snapshot pair formulation)

    where X  = X[:, :-1] (first T-1 snapshots)
          X' = X[:, 1:]  (next T-1 snapshots)

    Procedure:
      1. SVD of X  →  U, Σ, Vᵀ; truncate to rank r
      2. Ã = Uᵣᵀ X' Vᵣ Σᵣ⁻¹   (reduced operator)
      3. Eig(Ã)  →  eigenvalues λ̃, eigenvectors W̃
      4. DMD modes: Φ = X' Vᵣ Σᵣ⁻¹ W̃
      5. Continuous eigenvalues: ω = log(λ) / dt

    Parameters
    ----------
    X  : (d, T) — state matrix (d dimensions, T time steps)
    r  : truncation rank; None = no truncation
    dt : time step in seconds

    Returns
    -------
    dict:
      eigenvalues     : (r,) complex — discrete-time eigenvalues
      eigenvalues_ct  : (r,) complex — continuous-time (ω = log(λ)/dt)
      modes           : (d, r) complex — DMD spatial modes
      amplitudes      : (r,) complex — initial modal amplitudes
      rank            : int — truncation rank used
    """
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    if r is not None:
        U, s, Vt = U[:, :r], s[:r], Vt[:r]

    r_actual = len(s)
    S_inv = np.diag(1.0 / s)

    Atilde = U.T @ X2 @ Vt.T @ S_inv
    lam, W = np.linalg.eig(Atilde)

    Phi = X2 @ Vt.T @ S_inv @ W  # DMD modes

    # Initial amplitudes: least-squares fit to X1[:, 0]
    b = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)[0]

    return {
        "eigenvalues": lam,
        "eigenvalues_ct": np.log(lam + 1e-300) / dt,
        "modes": Phi,
        "amplitudes": b,
        "rank": r_actual,
    }


def trial_dmd(
    Z: NDArray,
    r: int = 6,
    dt: float = 1.0 / 1200,
) -> dict:
    """Apply exact DMD to a single trial latent trajectory.

    Parameters
    ----------
    Z  : (T, d) — latent trajectory (time × dimensions)
    r  : truncation rank

    Returns
    -------
    dict with eigenvalues, eigenvalues_ct, modes
    """
    return exact_dmd(Z.T, r=r, dt=dt)


def eigenspectrum_stability(
    eigenvalues: NDArray,
) -> dict:
    """Stability statistics from discrete-time eigenspectrum.

    For a stable, maintenance-capable system, discrete eigenvalues should
    cluster near the unit circle (|λ| ≈ 1): persistent but bounded dynamics.
    |λ| < 1 → decaying (memory fades), |λ| > 1 → growing (unbounded).

    Returns
    -------
    dict:
      unit_circle_dist : mean distance of |λ| from 1.0
      max_growth_rate  : max |λ| - 1 (positive = unstable mode)
      n_stable_modes   : count of modes with |λ| ≤ 1.01
      dominant_freq_hz : frequency of largest-amplitude mode (requires dt)
    """
    lam = np.asarray(eigenvalues)
    mod = np.abs(lam)
    return {
        "unit_circle_dist": np.mean(np.abs(mod - 1.0)),
        "max_growth_rate": float(np.max(mod) - 1.0),
        "n_stable_modes": int(np.sum(mod <= 1.01)),
        "eigenvalue_moduli": mod,
    }


def maintenance_eigenspectra(
    Z_trials: NDArray,
    times: NDArray,
    task_id: NDArray,
    tgt_id: NDArray,
    maint_window: tuple[float, float] = (0.3, 1.4),
    r: int = 6,
    dt: float = 1.0 / 1200,
) -> dict:
    """Compare DMD eigenspectra between trial conditions during maintenance.

    Fits DMD to the maintenance window of each trial, collects eigenvalues,
    and computes stability statistics for each condition.

    Returns
    -------
    dict:
      evals_by_condition : {condition_key: (N, r) complex eigenvalue arrays}
      stability_by_condition : {condition_key: stability_dict}
    """
    maint_mask = (times >= maint_window[0]) & (times <= maint_window[1])

    conditions = {
        "2back_target": (task_id == 2) & (tgt_id == 2),
        "2back_nontarget": (task_id == 2) & (tgt_id == 1),
        "0back": task_id == 0,
    }

    evals_by_cond = {}
    stab_by_cond = {}

    for key, mask in conditions.items():
        if mask.sum() < 3:
            continue
        evals_list = []
        for i in np.where(mask)[0]:
            Z_trial = Z_trials[i][maint_mask]  # (T_maint, d)
            if Z_trial.shape[0] < r + 2:
                continue
            try:
                dmd = trial_dmd(Z_trial, r=r, dt=dt)
                evals_list.append(dmd["eigenvalues"])
            except np.linalg.LinAlgError:
                continue

        if evals_list:
            evals_arr = np.array(evals_list)  # (N, r)
            evals_by_cond[key] = evals_arr
            stab_by_cond[key] = eigenspectrum_stability(evals_arr.ravel())

    return {"evals_by_condition": evals_by_cond, "stability_by_condition": stab_by_cond}


# ── Extended dynamical analyses ────────────────────────────────────────────────

def velocity_autocorrelation(
    Z: NDArray,
    dt: float = 1.0,
    max_lag: int = 50,
) -> NDArray:
    """Temporal autocorrelation of the velocity field across all latent dimensions.

    Computes the normalised autocorrelation of Ż(t) averaged across dimensions.
    Slow decay → persistent dynamics (integrator-like WM).
    Fast decay → noisy / transient dynamics.

    Parameters
    ----------
    Z       : (T, d) — latent trajectory
    dt      : time step (for informational purposes; result is in lag units)
    max_lag : maximum lag in samples

    Returns
    -------
    ac : (max_lag + 1,) — autocorrelation at lags 0, 1, ..., max_lag
    """
    Zdot = velocity_field(Z, dt)
    T, d = Zdot.shape
    ac = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag >= T - 2:
            break
        v0 = Zdot[1 : T - 1 - lag]
        v1 = Zdot[1 + lag : T - 1]
        num = (v0 * v1).sum(axis=1).mean()
        norm = (Zdot[1:-1] ** 2).sum(axis=1).mean() + 1e-15
        ac[lag] = num / norm
    return ac


def ring_attractor_phase(
    Z: NDArray,
    smooth_sigma: float = 3.0,
) -> NDArray:
    """Estimate ring-attractor phase from a latent trajectory.

    Projects Z onto its top-2 PCA dimensions, then computes the instantaneous
    phase angle θ(t) = arctan2(PC2, PC1) ∈ (-π, π].

    A ring attractor manifests as a monotonic or slow-drifting θ(t) during
    WM maintenance. Decoded phase encodes WM content (Compte et al. 2000).

    Parameters
    ----------
    Z            : (T, d) — latent trajectory
    smooth_sigma : Gaussian smoothing σ in samples before phase estimation

    Returns
    -------
    phase : (T,) — instantaneous phase in radians
    """
    Zc = Z - Z.mean(axis=0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    proj = Zc @ Vt[:2].T  # (T, 2)

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        proj = gaussian_filter1d(proj, sigma=smooth_sigma, axis=0)

    return np.arctan2(proj[:, 1], proj[:, 0])


def local_linear_stability(
    Z: NDArray,
    dt: float = 1.0,
    n_neighbors: int = 15,
) -> dict:
    """Local Jacobian eigenspectrum at each time point along a trajectory.

    At each t, finds the n_neighbors nearest time points in state space,
    fits a local linear model dŻ/dZ ≈ J_t (local Jacobian via least squares),
    and returns the eigenvalues of J_t.

    A globally stable trajectory has Re(eig(J_t)) < 0 everywhere.
    Excursions above 0 mark transient instabilities — candidate moments for
    memory failure.

    Parameters
    ----------
    Z           : (T, d)
    dt          : time step (for continuous-time Jacobian scaling)
    n_neighbors : neighbourhood size for local regression

    Returns
    -------
    dict:
      eigenvalues : (T, d) complex — local Jacobian eigenvalues at each t
      max_real    : (T,) — max Re(eig) at each t  (positive = locally unstable)
      mean_real   : (T,) — mean Re(eig) at each t
    """
    T, d = Z.shape
    Zdot = velocity_field(Z, dt)

    evals = np.full((T, d), np.nan, dtype=complex)
    max_re = np.full(T, np.nan)
    mean_re = np.full(T, np.nan)

    diffs = Z[:, None, :] - Z[None, :, :]  # (T, T, d)
    dists = np.sqrt((diffs**2).sum(axis=-1))   # (T, T)

    for t in range(1, T - 1):
        nn = np.argsort(dists[t])[:n_neighbors + 1]
        nn = nn[nn != t][:n_neighbors]
        if len(nn) < d + 1:
            continue
        dX = Z[nn] - Z[t]            # (k, d)
        dV = Zdot[nn] - Zdot[t]      # (k, d)
        J, _, _, _ = np.linalg.lstsq(dX, dV, rcond=None)  # (d, d)
        lam = np.linalg.eigvals(J)
        evals[t] = lam
        max_re[t] = lam.real.max()
        mean_re[t] = lam.real.mean()

    return {"eigenvalues": evals, "max_real": max_re, "mean_real": mean_re}
