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
    dt: float = 1.0,
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
        # np.linalg.eig returns real dtype when the spectrum happens to be all-real
        # (e.g. a rank-deficient trajectory); log() of a negative real then NaNs
        # instead of taking the complex branch, so go through abs/angle explicitly.
        "eigenvalues_ct": np.log(np.abs(lam) + 1e-300) / dt + 1j * np.angle(lam) / dt,
        "modes": Phi,
        "amplitudes": b,
        "rank": r_actual,
    }


def trial_dmd(
    Z: NDArray,
    r: int = 6,
    dt: float = 1.0,
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


def _dmd_from_pairs(X1: NDArray, X2: NDArray, r: int) -> tuple[NDArray, NDArray]:
    """Exact-DMD operator fit from explicit (not necessarily contiguous) snapshot pairs.

    X1, X2 : (d, M) — M paired snapshots x_t (X1) and x_{t+1} (X2); pairs need not
    come from a single trajectory, so this is what lets DMD be fit on pooled
    single-trial transitions rather than only a trial-averaged mean trajectory.
    """
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(r, len(s))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    S_inv = np.diag(1.0 / s)
    Atilde = U.T @ X2 @ Vt.T @ S_inv
    lam, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vt.T @ S_inv @ W
    A = np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))
    return A, lam


def ensemble_dmd(
    Z_trials: NDArray,
    r: int = 6,
    dt: float = 1.0,
    n_splits: int = 5,
    n_null: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """DMD fit on pooled single-trial transition pairs, with CV and a null.

    Fitting DMD to a trial-averaged mean trajectory at rank r hits near-perfect
    R² by construction (an r×r operator fit to one smooth curve) and is
    additionally confounded by trial-averaging contraction: averaging
    trajectories with trial-to-trial timing jitter makes the *mean* contract
    even when individual trials don't. This function instead stacks every
    trial's (x_t → x_{t+1}) pairs into one snapshot-pair regression, fits a
    single operator A on the ensemble, and validates it out-of-sample.

    Cross-validation: A is fit on a subset of trials and scored (one-step R²)
    on held-out trials — unlike the trivial in-sample R² of the mean-trajectory
    fit, this can genuinely fail if the linear approximation doesn't
    generalise across trials.

    Null: each trial is independently circularly time-shifted before pairing,
    destroying true (x_t, x_{t+1}) correspondence while preserving each
    trial's marginal statistics and autocorrelation structure; an operator
    fit to this scrambled ensemble should show near-chance one-step R².

    Parameters
    ----------
    Z_trials : (N, T, d) — single-trial latent trajectories
    r        : DMD truncation rank
    dt       : time step (s)
    n_splits : number of trial-wise CV folds
    n_null   : number of circular-shift null resamples
    rng      : random number generator

    Returns
    -------
    dict:
      A            : (d, d) — operator fit on the full trial ensemble
      eigenvalues  : (r,) complex
      div_scalar   : float — Σ log|λᵢ| / dt (true continuous-time divergence)
      r2_insample  : float — one-step R² on the trials A was fit on
      r2_cv        : float — mean out-of-sample one-step R² across folds
      r2_cv_std    : float
      r2_null      : float — mean one-step R² under the circular-shift null
      r2_null_std  : float
      n_trials     : int — trials contributing snapshot pairs
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N, T, d = Z_trials.shape
    r_use = min(r, d, N * (T - 1) - 1)

    def _pairs(Z_sub: NDArray) -> tuple[NDArray, NDArray]:
        X1 = Z_sub[:, :-1, :].reshape(-1, d).T  # (d, n*(T-1))
        X2 = Z_sub[:, 1:, :].reshape(-1, d).T
        return X1, X2

    def _r2(A: NDArray, X1: NDArray, X2: NDArray) -> float:
        pred = A @ X1
        ss_res = np.sum((X2 - pred) ** 2)
        ss_tot = np.sum((X2 - X2.mean(axis=1, keepdims=True)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-10))

    X1_all, X2_all = _pairs(Z_trials)
    A_all, lam_all = _dmd_from_pairs(X1_all, X2_all, r_use)
    div_scalar = float(np.sum(np.log(np.abs(lam_all) + 1e-300))) / dt
    r2_insample = _r2(A_all, X1_all, X2_all)

    trial_idx = rng.permutation(N)
    folds = np.array_split(trial_idx, min(n_splits, N))
    r2_cv_list = []
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        if len(tr) < 2 or len(te) < 1:
            continue
        X1_tr, X2_tr = _pairs(Z_trials[tr])
        X1_te, X2_te = _pairs(Z_trials[te])
        A_tr, _ = _dmd_from_pairs(X1_tr, X2_tr, r_use)
        r2_cv_list.append(_r2(A_tr, X1_te, X2_te))
    r2_cv = float(np.mean(r2_cv_list)) if r2_cv_list else float("nan")
    r2_cv_std = float(np.std(r2_cv_list)) if r2_cv_list else float("nan")

    r2_null_list = []
    for _ in range(n_null):
        Z_shift = np.empty_like(Z_trials)
        for i in range(N):
            shift = int(rng.integers(1, max(T - 1, 2)))
            Z_shift[i] = np.roll(Z_trials[i], shift, axis=0)
        X1_s, X2_s = _pairs(Z_shift)
        A_s, _ = _dmd_from_pairs(X1_s, X2_s, r_use)
        r2_null_list.append(_r2(A_s, X1_s, X2_s))
    r2_null = float(np.mean(r2_null_list))
    r2_null_std = float(np.std(r2_null_list))

    return {
        "A": A_all,
        "eigenvalues": lam_all,
        "div_scalar": div_scalar,
        "r2_insample": r2_insample,
        "r2_cv": r2_cv,
        "r2_cv_std": r2_cv_std,
        "r2_null": r2_null,
        "r2_null_std": r2_null_std,
        "n_trials": int(N),
    }


def divergence_rank_sweep(
    Z_trials: NDArray,
    dt: float,
    ranks: tuple[int, ...] = (5, 6, 7, 8),
    n_splits: int = 5,
    n_null: int = 10,
    rng: np.random.Generator | None = None,
) -> dict:
    """Ensemble-DMD divergence Σlog|λ|/dt at a sweep of truncation ranks.

    At r == d (full rank), Σlog|λ|/dt is the log-determinant of the empirical
    transition matrix and is numerically sensitive to small eigenvalues; a
    divergence-based contrast (e.g. single-trial vs trial-mean, or condition A
    vs condition B) is only trustworthy if it survives truncation away from
    that edge case (Round-4 audit item 1b/9).

    Parameters
    ----------
    Z_trials : (N, T, d) — single-trial latent trajectories
    dt       : time step (s)
    ranks    : truncation ranks to sweep (each clipped to <= d)

    Returns
    -------
    dict: ranks (list[int], actually used, deduplicated), div_scalar (list[float])
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = Z_trials.shape[2]
    ranks_use = sorted(set(min(r, d) for r in ranks))
    div_scalar = [
        ensemble_dmd(Z_trials, r=r, dt=dt, n_splits=n_splits, n_null=n_null, rng=rng)["div_scalar"]
        for r in ranks_use
    ]
    return {"ranks": ranks_use, "div_scalar": div_scalar}


def mean_trajectory_divergence_rank_sweep(
    Z_mean: NDArray,
    dt: float,
    ranks: tuple[int, ...] = (5, 6, 7, 8),
) -> dict:
    """Trial-averaged-mean-trajectory divergence Σlog|λ|/dt at a sweep of ranks.

    Companion to divergence_rank_sweep — same rank-robustness rationale,
    applied to the mean-trajectory (rather than single-trial ensemble) fit.

    Returns
    -------
    dict: ranks (list[int]), div_scalar (list[float])
    """
    d = Z_mean.shape[1]
    ranks_use = sorted(set(min(r, d, Z_mean.shape[0] - 2) for r in ranks))
    div_scalar = []
    for r in ranks_use:
        dmd = exact_dmd(Z_mean.T, r=r, dt=dt)
        div_scalar.append(float(np.sum(np.log(np.abs(dmd["eigenvalues"]) + 1e-300))) / dt)
    return {"ranks": ranks_use, "div_scalar": div_scalar}


def rank_robustness_sign(contrast: NDArray) -> bool:
    """True if every element of `contrast` (e.g. ensemble-div minus mean-div,
    or condition-B-div minus condition-A-div, across a rank sweep) shares the
    same nonzero sign — i.e. the qualitative result does not flip with rank."""
    c = np.asarray(contrast)
    if np.any(c == 0):
        return False
    return bool(np.all(np.sign(c) == np.sign(c[0])))


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
    conditions: dict[str, NDArray],
    maint_window: tuple[float, float] = (0.3, 1.4),
    r: int = 6,
    dt: float = 1.0,
) -> dict:
    """Compare DMD eigenspectra between trial conditions during maintenance.

    Fits DMD to the maintenance window of each trial, collects eigenvalues,
    and computes stability statistics for each condition.

    Parameters
    ----------
    Z_trials     : (N, T, d) — single-trial latent trajectories
    times        : (T,) — time vector matching Z_trials' T axis
    conditions   : {condition_name: (N,) boolean trial mask} — caller-defined
        (e.g. Miller's 0/2-back x target/non-target, Boran's set-size levels,
        Rutishauser's load levels); this function makes no assumption about
        the task's condition scheme (Round-4 audit item 1d).
    maint_window : (t0, t1) window (in `times` units) each trial is cropped to

    Returns
    -------
    dict:
      evals_by_condition : {condition_key: (N, r) complex eigenvalue arrays}
      stability_by_condition : {condition_key: stability_dict}
    """
    maint_mask = (times >= maint_window[0]) & (times <= maint_window[1])

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


def dmd_reconstruction_error(
    Z: NDArray,
    r: int = 6,
    dt: float = 1.0,
) -> dict:
    """Validate DMD linearity assumption via one-step-ahead reconstruction error.

    Computes ||x(t+1) - A_dmd x(t)||₂ / ||x(t)||₂ for each time step.
    Low relative error → linear approximation is valid for that trajectory.

    Also computes the variance explained (R²) of the linear model.

    Parameters
    ----------
    Z  : (T, d) — latent trajectory
    r  : DMD truncation rank

    Returns
    -------
    dict:
      relative_error : (T-1,) — per-step relative reconstruction error
      mean_rel_error : float
      r_squared      : float — variance explained by linear model
      A              : (d, d) — fitted linear operator
    """
    dmd = exact_dmd(Z.T, r=r, dt=dt)
    Phi = dmd["modes"]     # (d, r)
    lam = dmd["eigenvalues"]
    b   = dmd["amplitudes"]

    # Reconstruct A from DMD: A = Phi @ diag(lam) @ pinv(Phi)
    A = np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))

    T = Z.shape[0]
    X_pred = (A @ Z[:-1].T).T   # (T-1, d)
    X_true = Z[1:]               # (T-1, d)

    err = np.linalg.norm(X_true - X_pred, axis=1)
    nrm = np.linalg.norm(X_true, axis=1) + 1e-10
    rel_err = err / nrm

    ss_res = np.sum((X_true - X_pred) ** 2)
    ss_tot = np.sum((X_true - X_true.mean(0)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    return {
        "relative_error": rel_err,
        "mean_rel_error": float(rel_err.mean()),
        "r_squared": float(r2),
        "A": A,
    }


def koopman_edmd(
    Z: NDArray,
    r: int = 6,
    dt: float = 1.0,
    poly_degree: int = 2,
    delay_embeddings: int = 3,
) -> dict:
    """Extended DMD (EDMD) — finite-dimensional Koopman approximation.

    Koopman operator theory (Mezić 2005; Williams et al. 2015) states that any
    nonlinear dynamical system ẋ = f(x) has an EXACT linear representation in
    the infinite-dimensional space of all observables g(x).  EDMD approximates
    this by lifting x → ψ(x) with a finite dictionary of observables (polynomial
    basis functions + delay embeddings), then fitting DMD on the lifted state.

    This provides:
    1. A principled Koopman justification for DMD on nonlinear neural dynamics
    2. A validity check: if EDMD (richer basis) produces the same eigenspectrum
       as plain DMD, the dynamics are well-approximated linearly in the original
       basis. If they diverge, nonlinear lifting is needed.
    3. Koopman eigenvalues ωₖ govern the frequencies of all observables, not
       just linear ones — so they characterize the attractor geometry globally.

    Parameters
    ----------
    Z              : (T, d) — latent trajectory
    r              : EDMD truncation rank
    poly_degree    : degree of polynomial lifting (1=linear=standard DMD; 2=quadratic)
    delay_embeddings : number of Takens delay embeddings to include

    Returns
    -------
    dict:
      eigenvalues     : (r,) complex Koopman eigenvalues
      eigenvalues_ct  : continuous-time ω = log(λ)/dt
      modes           : (D_lift, r) Koopman modes in lifted space
      r_squared_lift  : variance explained in LIFTED space
      r_squared_orig  : variance explained in ORIGINAL space
      lifting_dim     : D_lift (dimensionality after lifting)
    """
    from itertools import combinations_with_replacement

    T, d = Z.shape

    # Build polynomial lifting: [x, x²_monomials]
    def polynomial_features(X, degree):
        cols = [X]  # degree 1 already included
        if degree >= 2:
            # All degree-2 monomials
            for i, j in combinations_with_replacement(range(d), 2):
                cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(cols)

    # Delay embedding: concatenate Z(t), Z(t-1), ..., Z(t-n_delay). No padding —
    # rows without a full n_delay history are dropped rather than backfilled with
    # Z[0], which would otherwise fake a constant pre-trial history.
    n_delay = delay_embeddings - 1
    if n_delay > 0:
        Z_delay = np.hstack([Z[n_delay - k:T - k] for k in range(n_delay + 1)])
    else:
        Z_delay = Z

    Psi = polynomial_features(Z_delay, poly_degree)   # (T-n_delay, D_lift)
    D_lift = Psi.shape[1]

    # Fit EDMD: Psi' ≈ K Psi
    Psi1 = Psi[:-1]   # (T-n_delay-1, D_lift)
    Psi2 = Psi[1:]

    # DMD on lifted space
    dmd_lift = exact_dmd(Psi1.T, r=min(r, D_lift - 1), dt=dt)
    lam = dmd_lift["eigenvalues"]
    Phi = dmd_lift["modes"]   # (D_lift, r_actual)
    b   = dmd_lift["amplitudes"]

    # R² in lifted space
    K = np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))
    Psi2_pred = (K @ Psi1.T).T
    ss_res_lift = np.sum((Psi2 - Psi2_pred) ** 2)
    ss_tot_lift = np.sum((Psi2 - Psi2.mean(0)) ** 2)
    r2_lift = 1.0 - ss_res_lift / (ss_tot_lift + 1e-10)

    # R² back in original space (first d columns of lifted state)
    Z2_pred = Psi2_pred[:, :d]
    Z2_true = Z[n_delay + 1:]
    ss_res_orig = np.sum((Z2_true - Z2_pred) ** 2)
    ss_tot_orig = np.sum((Z2_true - Z2_true.mean(0)) ** 2)
    r2_orig = 1.0 - ss_res_orig / (ss_tot_orig + 1e-10)

    return {
        "eigenvalues":     lam,
        "eigenvalues_ct":  np.log(np.abs(lam) + 1e-300) / dt + 1j * np.angle(lam) / dt,
        "modes":           Phi,
        "r_squared_lift":  float(r2_lift),
        "r_squared_orig":  float(r2_orig),
        "lifting_dim":     D_lift,
    }


def sindy_neural_dynamics(
    Z: NDArray,
    times: NDArray | None = None,
    poly_degree: int = 2,
    threshold: float = 0.1,
    alpha: float = 0.05,
    maintenance_window: tuple[float, float] = (0.3, 1.4),
) -> dict:
    """SINDy (Sparse Identification of Nonlinear Dynamics) on latent trajectories.

    Brunton, Proctor & Kutz (2016, PNAS): fits ẋ = Θ(x)ξ where Θ is a library
    of candidate terms (polynomial) and ξ is a sparse coefficient matrix found
    via sequential thresholded least squares (STLSQ).

    For neural WM dynamics, SINDy identifies whether:
    - Only linear terms survive (ẋ ≈ Ax) → DMD is exact
    - Quadratic terms survive → attractor is nonlinearly bounded
    - Stuart-Landau-type terms x(x₁²+x₂²) survive → ring attractor in PC1-PC2

    The degree of nonlinearity in the identified model is the empirical answer
    to "how nonlinear is WM dynamics?" — addressing the DMD linearity assumption.

    Parameters
    ----------
    Z          : (N, T, d) or (T, d) — latent trajectories (trials × time × dim
                 or single trajectory). If 3D, uses mean trajectory.
    times      : (T,) time vector; if provided, restricts to maintenance window
    poly_degree: maximum polynomial degree in Θ (1=linear, 2=quadratic)
    threshold  : STLSQ threshold (ξ below this are zeroed)
    alpha      : regularisation parameter

    Returns
    -------
    dict:
      coefficients   : (d, n_terms) — sparse coefficient matrix ξ
      feature_names  : list of str  — names of Θ columns (e.g. "x0", "x0^2", ...)
      r_squared      : (d,) variance explained per dimension
      n_nonzero      : int  — total nonzero terms (sparsity measure)
      dominant_terms : list of str — top terms by magnitude
      linear_fraction: float — fraction of nonzero terms that are linear
    """
    import pysindy as ps

    if Z.ndim == 3:
        Z_use = Z.mean(0)   # use trial-mean trajectory
    else:
        Z_use = Z.copy()

    if times is not None and len(times) == Z_use.shape[0]:
        maint = (times >= maintenance_window[0]) & (times <= maintenance_window[1])
        Z_use = Z_use[maint]
        t_use = times[maint]
    else:
        t_use = (np.arange(Z_use.shape[0]) / 1200.0
                 if times is None else times[:Z_use.shape[0]])

    dt_use = float(np.median(np.diff(t_use)))

    # SINDy with polynomial library
    lib = ps.PolynomialLibrary(degree=poly_degree)
    opt = ps.STLSQ(threshold=threshold, alpha=alpha)
    model = ps.SINDy(feature_library=lib, optimizer=opt)
    model.fit(Z_use, t=dt_use)

    coef = model.coefficients()          # (d, n_terms)
    names = model.get_feature_names()    # list of str

    # R² per dimension: compare predicted vs finite-difference derivative
    Z_dot = np.gradient(Z_use, dt_use, axis=0)         # (T, d)
    Z_dot_pred = model.predict(Z_use)                   # (T, d)
    r2 = np.array([
        1 - np.sum((Z_dot[:, i] - Z_dot_pred[:, i])**2) /
            (np.sum((Z_dot[:, i] - Z_dot[:, i].mean())**2) + 1e-10)
        for i in range(Z_use.shape[1])
    ])

    n_nonzero = int(np.sum(np.abs(coef) > 1e-10))
    magnitudes = np.abs(coef).mean(axis=0)   # mean across dims
    top_idx = np.argsort(magnitudes)[::-1][:5]
    dominant_terms = [names[i] for i in top_idx if magnitudes[i] > 1e-10]

    # Count linear vs nonlinear terms
    linear_names = [n for n in names if n.count("x") == 1 and "^" not in n and " " not in n]
    linear_mask = np.array([n in linear_names for n in names])
    coef_nonzero = np.abs(coef) > 1e-10
    n_lin = int(np.sum(coef_nonzero[:, linear_mask]))
    linear_fraction = n_lin / (n_nonzero + 1e-10)

    return {
        "coefficients":    coef,
        "feature_names":   names,
        "r_squared":       r2,
        "n_nonzero":       n_nonzero,
        "dominant_terms":  dominant_terms,
        "linear_fraction": float(linear_fraction),
        "mean_r2":         float(r2.mean()),
    }


def flow_divergence(
    Z: NDArray,
    dt: float = 1.0,
    method: str = "dmd",
    n_neighbors: int = 20,
    r: int = 6,
) -> dict:
    """Divergence of the neural flow field ∇·v(x, t).

    For a dynamical system ẋ = f(x), the divergence ∇·v = trace(∇f) measures
    whether the phase-space volume is expanding (∇·v > 0, trajectories diverging)
    or contracting (∇·v < 0, converging toward attractor). Zero crossings from
    negative to positive mark the onset of trajectory instability — the optimal
    trigger time for closed-loop stimulation.

    Two estimation methods:

    **dmd**: Fits global discrete-time DMD operator A (x_{t+1} = A x_t), then
        recovers the generator F of the continuous-time flow A = exp(F Δt) via
        its eigenvalues: ∇·v = trace(F) = Σᵢ log|λᵢ(A)| / dt.
        This is the exact continuous-time volume expansion/contraction rate.
        The naive first-order approximation trace(A − I)/dt = Σᵢ(Re λᵢ − 1)/dt
        is only valid when A ≈ I with near-real eigenvalues; for eigenvalues
        near the unit circle with nonzero phase (rotation), Re λ − 1 ≈ −θ²/2 < 0
        registers spurious "contraction" even for a purely rotating, volume-
        preserving flow (|λ| = 1 ⇒ log|λ| = 0). Using log|λ| instead of Re λ − 1
        cleanly separates rotation from genuine expansion/contraction.

    **local**: Estimates local Jacobians J(t) via nearest-neighbour regression
        at each time point; ∇·v(t) = trace(J(t))/dt.
        Time-varying — captures transient divergence events during maintenance.

    Parameters
    ----------
    Z           : (T, d) mean trajectory
    dt          : time step in seconds
    method      : 'dmd' | 'local'
    n_neighbors : neighbourhood size for 'local' method
    r           : DMD rank for 'dmd' method

    Returns
    -------
    dict:
      divergence      : (T,) — ∇·v(t); constant for 'dmd', time-varying for 'local'
      mean_divergence : float — temporal mean
      trigger_time    : int | None — first sample where div crosses 0 from below
                         (None if never positive) — optimal stimulation onset
      A               : (d, d) — DMD operator (only for 'dmd' method)
    """
    T, d = Z.shape
    Zdot = velocity_field(Z, dt)

    if method == "dmd":
        res = dmd_reconstruction_error(Z, r=r, dt=dt)
        A = res["A"]
        eigs_A = np.linalg.eigvals(A)
        div_scalar = float(np.sum(np.log(np.abs(eigs_A) + 1e-300))) / dt
        divergence = np.full(T, div_scalar)
        trigger = 0 if div_scalar > 0 else None
        return {
            "divergence":      divergence,
            "mean_divergence": div_scalar,
            "trigger_time":    trigger,
            "A":               A,
        }

    # local Jacobian method
    divergence = np.full(T, np.nan)
    diffs = Z[:, None, :] - Z[None, :, :]   # (T, T, d)
    dists = np.sqrt((diffs ** 2).sum(-1))    # (T, T)

    for t in range(1, T - 1):
        nn = np.argsort(dists[t])[:n_neighbors + 1]
        nn = nn[nn != t][:n_neighbors]
        if len(nn) < d + 1:
            continue
        dX = Z[nn] - Z[t]        # (k, d)
        dV = Zdot[nn] - Zdot[t]  # (k, d)
        J, _, _, _ = np.linalg.lstsq(dX, dV, rcond=None)
        divergence[t] = float(np.trace(J)) / dt

    # fill endpoints
    divergence[0] = divergence[1]
    divergence[-1] = divergence[-2]

    # zero-crossing: neg → pos
    pos = divergence > 0
    crosses = np.where(~pos[:-1] & pos[1:])[0]
    trigger = int(crosses[0]) if len(crosses) > 0 else None

    return {
        "divergence":      divergence,
        "mean_divergence": float(np.nanmean(divergence)),
        "trigger_time":    trigger,
        "A":               None,
    }


def stimulation_trigger_window(
    Z: NDArray,
    times: NDArray,
    dt: float = 1.0,
    n_neighbors: int = 20,
    threshold: float = 0.0,
    min_duration_s: float = 0.050,
) -> dict:
    """Identify windows where the flow diverges beyond threshold.

    Used to determine when closed-loop stimulation should be triggered:
    stimulate during any window where ∇·v(t) > threshold for at least
    min_duration_s seconds.

    Returns
    -------
    dict:
      trigger_onsets  : list of int — sample indices where div first crosses threshold
      trigger_offsets : list of int — sample indices where div drops below threshold
      divergence      : (T,) — ∇·v(t) trace (local method)
    """
    div_res = flow_divergence(Z, dt=dt, method="local", n_neighbors=n_neighbors)
    div = div_res["divergence"]

    above = div > threshold
    min_dur = int(min_duration_s / dt)

    onsets, offsets = [], []
    in_window = False
    count = 0
    for t, a in enumerate(above):
        if a:
            if not in_window:
                count += 1
                if count >= min_dur:
                    in_window = True
                    onsets.append(t - count + 1)
            else:
                count += 1
        else:
            if in_window:
                offsets.append(t)
            in_window = False
            count = 0
    if in_window:
        offsets.append(len(above) - 1)

    return {
        "trigger_onsets":  onsets,
        "trigger_offsets": offsets,
        "trigger_times":   [times[o] for o in onsets] if len(onsets) > 0 else [],
        "divergence":      div,
    }


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


# ── Input-driven linear state-space (digital twin) ─────────────────────────────

def fit_input_lds(
    X: NDArray,
    U: NDArray,
    latent_dim: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Fit a linear latent state-space model WITH an explicit input channel:

        z[t+1] = A z[t] + B u[t]
        x[t]   = C z[t]

    Two-stage subspace-ID-lite (PCA for the observation map, then one joint
    least-squares solve for [A | B] on the latent transitions) — this is
    exact-DMD's snapshot-pair regression (`exact_dmd`) generalised with an
    exogenous-input column appended to the regressor, not a new estimator
    family.

    Parameters
    ----------
    X          : (T, d) — observed trajectory (time x channels)
    U          : (T, m) — input trajectory, same T as X (u[T-1] is unused,
                 there being no z[T] transition to explain it)
    latent_dim : PCA latent dimensionality k

    Returns
    -------
    A : (k, k) — latent dynamics
    B : (k, m) — input matrix
    C : (d, k) — observation map (x ≈ mean + z @ C.T)
    z : (T, k) — fitted latent trajectory
    """
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    T = X.shape[0]

    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    Uu, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(latent_dim, len(s))
    C = Vt[:k].T                      # (d, k)
    z = Xc @ C                        # (T, k)

    z1, z2, u1 = z[:-1], z[1:], U[:-1]
    reg = np.hstack([z1, u1])         # (T-1, k+m)
    coef, *_ = np.linalg.lstsq(reg, z2, rcond=None)   # (k+m, k)
    A = coef[:k].T                    # (k, k)
    B = coef[k:].T                    # (k, m)
    return A, B, C, z


def simulate_input_response(
    A: NDArray,
    B: NDArray,
    C: NDArray,
    z0: NDArray,
    U: NDArray,
) -> NDArray:
    """Roll out the fitted input-LDS z[t+1] = A z[t] + B u[t] from z0 under a
    given input sequence U (open-loop simulation — no observation feedback).

    Parameters
    ----------
    A, B, C : as returned by fit_input_lds (C unused in the rollout itself,
              accepted for signature symmetry / future observation-space output)
    z0      : (k,) initial latent state
    U       : (T, m) input sequence to inject

    Returns
    -------
    Z_sim : (T+1, k) — simulated latent trajectory, Z_sim[0] == z0
    """
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    k = A.shape[0]
    T = U.shape[0]
    Z_sim = np.zeros((T + 1, k))
    Z_sim[0] = z0
    for t in range(T):
        Z_sim[t + 1] = A @ Z_sim[t] + B @ U[t]
    return Z_sim
