"""
control.py — Optimal control theory for neural trajectory rescue.

Implements:
  - LQR design via discrete algebraic Riccati equation (DARE)
  - Minimum-energy control (exact endpoint controllability)
  - Controllability Gramian analysis
  - Energy-accuracy Pareto curve for stimulation parameter sweep

The LQR framework operationalizes the theoretical BCI application:
given the failing neural state identified by geometric biomarkers,
what is the minimum-energy perturbation that rescues the trajectory?

References
----------
Stengel RF (1994) Optimal Control and Estimation. Dover.
  Chapters 3-4 (LQR derivation) and Ch. 5 (DARE).
Kirk DE (1970) Optimal Control Theory. Dover.
Brunton SL & Kutz JN (2022) Data-Driven Science and Engineering. Ch. 8.
Gu S et al. (2015) Controllability of structural brain networks. Nat Commun.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.linalg as sla
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Controllability ────────────────────────────────────────────────────────────

def controllability_gramian(
    A: NDArray, B: NDArray, T: int
) -> NDArray:
    """Finite-horizon controllability Gramian W_c(T) = Σ_{k=0}^{T-1} Aᵏ B Bᵀ (Aᵀ)ᵏ.

    W_c characterises which directions of state space can be reached from
    the origin in T steps with bounded energy. If W_c is full rank, the system
    is controllable. The eigenvalues of W_c quantify the ease of control in
    each direction: large eigenvalue = easy (low energy), small = hard.

    Parameters
    ----------
    A : (n, n) — system matrix
    B : (n, m) — input matrix
    T : horizon length in steps

    Returns
    -------
    Wc : (n, n) — symmetric positive semi-definite Gramian
    """
    n = A.shape[0]
    Wc = np.zeros((n, n))
    Ak = np.eye(n)
    for _ in range(T):
        Wc += Ak @ B @ B.T @ Ak.T
        Ak = A @ Ak
    return Wc


def unstable_eigenvector(A: NDArray) -> tuple[NDArray, float]:
    """Direction of fastest growth (or slowest decay): eigenvector of A's
    eigenvalue with the largest real part.

    For a discrete-time operator A, this is the direction a perturbation
    grows fastest along — the steering axis v* a control policy read off
    fitted dynamics naturally targets (used by run_divergence_analysis.py
    and src/closed_loop.py).

    Returns
    -------
    v_star     : (n,) real, unit-norm
    max_re_eig : float — Re(eigenvalue) of the selected mode
    """
    eigs, vecs = np.linalg.eig(A)
    idx = int(np.argmax(eigs.real))
    v = vecs[:, idx].real
    v_star = v / (np.linalg.norm(v) + 1e-12)
    return v_star, float(eigs[idx].real)


def _normalize_adjacency(W: NDArray) -> NDArray:
    """Symmetrize and rescale a weighted adjacency matrix to spectral radius
    < 1 (Gu et al. 2015 Nat Commun, Methods): required for the closed-form
    average/modal controllability below, which assumes a stable, symmetric
    (orthogonally diagonalizable) system matrix."""
    Ws = (W + W.T) / 2.0
    s_max = np.linalg.svd(Ws, compute_uv=False)[0]
    return Ws / (1.0 + s_max)


def average_controllability(W: NDArray) -> NDArray:
    """Average controllability of every node (Gu et al. 2015 Nat Commun, Eq
    1): ease of steering the network with average input energy, in closed
    form from the eigendecomposition of the normalized (symmetric) adjacency:

        y_avg(i) = sum_k v_k(i)^2 / (1 - lambda_k^2)

    Large y_avg(i): node i can, on average, reach many states cheaply.

    Returns
    -------
    (n,) per-node average controllability
    """
    A = _normalize_adjacency(W)
    eigvals, eigvecs = np.linalg.eigh(A)
    denom = np.clip(1.0 - eigvals**2, 1e-12, None)
    return (eigvecs**2) @ (1.0 / denom)


def modal_controllability(W: NDArray) -> NDArray:
    """Modal controllability of every node (Gu et al. 2015 Nat Commun, Eq 2):
    ability to steer the network into its hardest-to-reach (weakly-coupled)
    modes, in closed form from the same eigendecomposition:

        phi(i) = sum_k (1 - lambda_k^2) * v_k(i)^2

    Small phi(i): node i loads mostly onto near-unity (slow) eigenmodes and
    is a comparatively weak driver of the network's hard-to-reach states.

    Returns
    -------
    (n,) per-node modal controllability
    """
    A = _normalize_adjacency(W)
    eigvals, eigvecs = np.linalg.eigh(A)
    weight = 1.0 - eigvals**2
    return (eigvecs**2) @ weight


def is_controllable(A: NDArray, B: NDArray) -> bool:
    """Check full-rank controllability matrix [B, AB, A²B, ..., A^{n-1}B]."""
    n = A.shape[0]
    C_mat = B.copy()
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = A @ Ak
        C_mat = np.hstack([C_mat, Ak @ B])
    return np.linalg.matrix_rank(C_mat) == n


# ── LQR design ─────────────────────────────────────────────────────────────────

def dare_solve(
    A: NDArray, B: NDArray, Q: NDArray, R: NDArray
) -> tuple[NDArray, NDArray]:
    """Solve the Discrete Algebraic Riccati Equation (DARE).

    DARE: P = AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q

    The solution P is the optimal cost-to-go matrix. The LQR gain is:
      K = (R + BᵀPB)⁻¹ BᵀPA

    Parameters
    ----------
    A : (n, n) — discrete system matrix
    B : (n, m) — input matrix
    Q : (n, n) — state cost (positive semi-definite)
    R : (m, m) — control cost (positive definite)

    Returns
    -------
    P : (n, n) — DARE solution (optimal cost matrix)
    K : (m, n) — LQR feedback gain  u = -K x
    """
    if _HAS_SCIPY:
        P = sla.solve_discrete_are(A, B, Q, R)
    else:
        # Value iteration (slower but dependency-free)
        P = Q.copy().astype(float)
        n_iter = 10000
        for i in range(n_iter):
            P_new = (
                A.T @ P @ A
                - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
                + Q
            )
            if np.max(np.abs(P_new - P)) < 1e-10:
                P = P_new
                break
            P = P_new
        else:
            import warnings
            warnings.warn(
                f"dare_solve: value iteration did not converge in {n_iter} iterations "
                f"(final delta={np.max(np.abs(P_new - P)):.2e}); P may be inaccurate.",
                RuntimeWarning,
            )

    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def lqr_design(
    A: NDArray,
    B: NDArray,
    q_state: float = 1.0,
    r_control: float = 1.0,
) -> dict:
    """Design LQR controller with identity-scaled cost matrices.

    Q = q_state * I_n,  R = r_control * I_m

    The ratio q_state/r_control sets the trade-off between state accuracy
    and control energy. High ratio → aggressive, high-energy correction.
    Low ratio → gentle, low-energy correction.

    Returns
    -------
    dict with: P (DARE solution), K (gain), closed_loop_A (A - BK), is_stable
    """
    n, m = A.shape[0], B.shape[1]
    Q_mat = q_state * np.eye(n)
    R_mat = r_control * np.eye(m)
    P, K = dare_solve(A, B, Q_mat, R_mat)
    A_cl = A - B @ K
    eigs = np.linalg.eigvals(A_cl)
    return {
        "P": P,
        "K": K,
        "closed_loop_A": A_cl,
        "is_stable": bool(np.all(np.abs(eigs) < 1.0)),
        "closed_loop_eigenvalues": eigs,
        "q_state": q_state,
        "r_control": r_control,
    }


# ── Minimum-energy control ─────────────────────────────────────────────────────

def minimum_energy_trajectory(
    A: NDArray,
    B: NDArray,
    x0: NDArray,
    xf: NDArray,
    T: int,
) -> tuple[NDArray, NDArray, float]:
    """Exact minimum-energy control to steer x0 → xf in T steps.

    The minimum-energy control (with no running cost on states) is:
      u* = Bᵀ (Aᵀ)^{T-1-k} W_c(T)⁻¹ (A^T x0 - xf)

    Total energy: E = u*ᵀ u*  (sum of squared control inputs)

    Parameters
    ----------
    A  : (n, n)
    B  : (n, m)
    x0 : (n,) — initial state (failing neural state)
    xf : (n,) — target state (correct maintenance trajectory)
    T  : control horizon in steps

    Returns
    -------
    x_traj : (T+1, n) — state trajectory under optimal control
    u_traj : (T, m)   — optimal control sequence
    energy : float    — total input energy ‖u*‖²
    """
    n = A.shape[0]
    Wc = controllability_gramian(A, B, T)

    AT = np.linalg.matrix_power(A, T)
    xf_adj = xf - AT @ x0

    try:
        Wc_inv_xf = np.linalg.solve(Wc + 1e-10 * np.eye(n), xf_adj)
    except np.linalg.LinAlgError:
        Wc_inv_xf = np.linalg.lstsq(Wc, xf_adj, rcond=None)[0]

    # Reconstruct control sequence and trajectory
    u_traj = np.zeros((T, B.shape[1]))
    x_traj = np.zeros((T + 1, n))
    x_traj[0] = x0

    Ak = np.eye(n)
    for k in range(T):
        u_traj[k] = B.T @ np.linalg.matrix_power(A, T - 1 - k).T @ Wc_inv_xf
        Ak = A @ Ak

    for k in range(T):
        x_traj[k + 1] = A @ x_traj[k] + B @ u_traj[k]

    energy = float(np.sum(u_traj**2))
    return x_traj, u_traj, energy


def lqr_simulate(
    A: NDArray,
    B: NDArray,
    K: NDArray,
    x0: NDArray,
    T: int,
    x_ref: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Simulate closed-loop LQR system for T steps.

    u[k] = -K (x[k] - x_ref)  (tracking control)

    Parameters
    ----------
    x_ref : reference state to track; None = stabilise to origin

    Returns
    -------
    x_traj : (T+1, n)
    u_traj : (T, m)
    """
    n = A.shape[0]
    if x_ref is None:
        x_ref = np.zeros(n)

    x_traj = np.zeros((T + 1, n))
    u_traj = np.zeros((T, K.shape[0]))
    x_traj[0] = x0

    for k in range(T):
        u_traj[k] = -K @ (x_traj[k] - x_ref)
        x_traj[k + 1] = A @ x_traj[k] + B @ u_traj[k]

    return x_traj, u_traj


# ── Pareto analysis ────────────────────────────────────────────────────────────

def energy_accuracy_pareto(
    A: NDArray,
    B: NDArray,
    x0_list: list[NDArray],
    xf_list: list[NDArray],
    q_values: NDArray,
    T: int = 50,
) -> dict:
    """Sweep the LQR cost ratio to trace the energy–accuracy Pareto frontier.

    For each q (state cost weight, with r=1 fixed), compute:
      - Mean total control energy across all (x0, xf) pairs
      - Final state error ‖x(T) - xf‖ as a proxy for accuracy

    Returns
    -------
    dict:
      q_values   : (nq,) array
      energies   : (nq,) mean total energy
      errors     : (nq,) mean final state error
    """
    energies = []
    errors = []

    for q in q_values:
        lqr = lqr_design(A, B, q_state=q, r_control=1.0)
        K = lqr["K"]

        trial_energies = []
        trial_errors = []
        for x0, xf in zip(x0_list, xf_list):
            x_traj, u_traj = lqr_simulate(A, B, K, x0, T, x_ref=xf)
            trial_energies.append(float(np.sum(u_traj**2)))
            trial_errors.append(float(np.linalg.norm(x_traj[-1] - xf)))

        energies.append(float(np.mean(trial_energies)))
        errors.append(float(np.mean(trial_errors)))

    return {
        "q_values": np.array(q_values),
        "energies": np.array(energies),
        "errors": np.array(errors),
    }


# ── BCI interpretation ─────────────────────────────────────────────────────────

def stimulation_energy_to_current(
    energy: float,
    n_channels: int,
    duration_ms: float = 100.0,
    impedance_kohm: float = 10.0,
) -> dict:
    """Convert dimensionless LQR energy to approximate stimulation parameters.

    This is an ORDER-OF-MAGNITUDE estimate only.

    E = Σ u² (dimensionless latent units)
    Assume each latent unit corresponds to 1 µA at the electrode.
    Power = I² × Z; Energy = Power × time

    Parameters
    ----------
    energy       : LQR control energy (latent units²)
    n_channels   : number of stimulation channels
    duration_ms  : pulse duration in ms
    impedance_kohm: electrode impedance in kΩ

    Returns
    -------
    dict with estimated current (µA), power (µW), charge (µC)
    """
    # Rough scaling: each latent unit ~ 1 µA RMS
    rms_current_uA = np.sqrt(energy / n_channels)
    duration_s = duration_ms / 1000.0
    # P[W] = I²[A²] × R[Ω] = (I_µA × 1e-6)² × (Z_kΩ × 1e3) = I_µA² × Z_kΩ × 1e-9 W
    # → convert to µW: × 1e6  →  I_µA² × Z_kΩ × 1e-3 µW
    power_uW = rms_current_uA**2 * impedance_kohm * 1e-3
    charge_uC = rms_current_uA * duration_s

    return {
        "rms_current_uA": float(rms_current_uA),
        "power_uW": float(power_uW),
        "charge_uC": float(charge_uC),
        "note": "Order-of-magnitude estimate. Latent→current scaling is arbitrary.",
    }
