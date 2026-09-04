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

from dataclasses import dataclass

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


def canonicalize_eigenvector_phase(v: NDArray) -> NDArray:
    """Deterministic real direction from a (possibly complex) eigenvector.

    `numpy.linalg.eig` returns eigenvectors defined only up to an arbitrary
    overall phase e^{i*phi} (BLAS/version-dependent, not physically
    meaningful): as phi varies, Re(e^{i*phi} v) sweeps the ENTIRE real
    invariant plane of a complex-conjugate pair, so |cos| between the
    `.real` parts of two draws of the SAME physical mode can land anywhere
    in [0, 1] with the mode itself unchanged. Every v*-stability number
    computed from a raw `.real` extraction (rank sweep, bootstrap, split-half)
    is contaminated by this until the phase is fixed by a rule that does not
    depend on numpy's arbitrary convention.

    Rule (deterministic, data-dependent only): rotate v by the phase that
    makes its largest-magnitude entry real and positive, then take the real
    part. For a genuinely real eigenvector this only fixes the overall sign.
    """
    idx = int(np.argmax(np.abs(v)))
    phase = np.angle(v[idx])
    v_rot = v * np.exp(-1j * phase)
    real_part = v_rot.real
    return real_part / (np.linalg.norm(real_part) + 1e-12)


@dataclass(frozen=True)
class DominantEigenmode:
    """Dominant discrete-time eigenmode of A, selected by spectral modulus,
    decomposed into modulus (rho) and phase (theta) with a stability/rotation
    classification.

    For a discrete-time linear operator, asymptotic growth or decay is set by
    ``rho = abs(lambda)``, not ``Re(lambda)``. A complex-conjugate pair with
    rho near 1 is a persistent ROTATION at angular frequency
    ``theta / dt``, not an unstable direction, however large its real part.
    Only ``classification == "unstable_real"`` licenses the word "unstable" —
    report rho and theta together and branch on `classification`, never on
    the function name or on Re(lambda) alone.

    Attributes
    ----------
    v_star : (n,) real, unit-norm, phase-canonicalized eigenvector (see
        canonicalize_eigenvector_phase). For a complex-conjugate pair this is
        one arbitrary real direction within their invariant plane, fixed only
        for plotting — use invariant_subspace_basis for inference.
    rho : spectral modulus abs(lambda) of the selected mode.
    theta : phase arg(lambda) in radians; exactly 0 for a real eigenvalue.
    is_complex : whether the selected mode belongs to a complex-conjugate pair.
    classification : one of
        "unstable_real"    — real eigenvalue, rho > 1
        "damped_real"       — real eigenvalue, rho <= 1
        "rotation"           — complex pair, rho within rho_tol of 1
        "damped_rotation"    — complex pair, rho < 1 (outside rho_tol)
        "growing_rotation"   — complex pair, rho > 1 (outside rho_tol)
    """

    v_star: NDArray
    rho: float
    theta: float
    is_complex: bool
    classification: str


def dominant_eigenmode(A: NDArray, rho_tol: float = 0.02) -> DominantEigenmode:
    """Select A's dominant discrete-time mode by spectral modulus and classify it.

    Selecting the maximum-modulus mode and calling it "unstable" (as the
    former name of this function did) is wrong whenever that modulus is below
    1, which is the common case for fitted retention dynamics. Use
    `classification` to decide whether "unstable" applies; the modulus alone
    does not.

    Parameters
    ----------
    A : (n, n) real system matrix
    rho_tol : distance from rho=1 within which a complex-conjugate pair is
        classified as a (non-decaying, non-growing) rotation rather than a
        damped or growing rotation

    Returns
    -------
    DominantEigenmode
    """
    eigs, vecs = np.linalg.eig(A)
    idx = int(np.argmax(np.abs(eigs)))
    lam = eigs[idx]
    v_star = canonicalize_eigenvector_phase(vecs[:, idx])
    rho = float(np.abs(lam))
    theta = float(np.angle(lam))
    is_complex = abs(lam.imag) > 1e-8 * (abs(lam.real) + 1e-12)
    if is_complex:
        if abs(rho - 1.0) <= rho_tol:
            classification = "rotation"
        elif rho < 1.0:
            classification = "damped_rotation"
        else:
            classification = "growing_rotation"
    else:
        classification = "unstable_real" if rho > 1.0 else "damped_real"
    return DominantEigenmode(
        v_star=v_star, rho=rho, theta=theta,
        is_complex=is_complex, classification=classification,
    )


@dataclass(frozen=True)
class InvariantSubspaceBasis:
    """Real orthonormal basis for A's m largest-|lambda| invariant subspace,
    with an explicit conditioning status instead of a silently unreliable
    basis when a mode is near-degenerate.

    Attributes
    ----------
    basis : (n, dim) real orthonormal basis; `dim` can fall short of the
        naive bound (1 per real mode, 2 per complex-conjugate pair) when a
        degenerate mode was encountered and only its reliable direction(s)
        were kept.
    dim : number of columns in `basis`.
    status : "ok" if every included mode was well-conditioned, otherwise the
        name of the first degeneracy encountered:
          "near_real_pair"      — a mode classified as complex had an
                                   imaginary component too small, relative to
                                   its real component, to fix a reliable
                                   second basis direction; only its real
                                   direction was kept.
          "unmatched_conjugate"  — no eigenvalue was found close enough to
                                   conj(lambda) for a complex mode (a
                                   near-degenerate spectrum); the mode was
                                   still included using its own real/imaginary
                                   parts, but its partner could not be
                                   verified, so treat it as unreliable.
    notes : one human-readable string per degeneracy encountered (mode index
        and reason); empty when status == "ok".
    """

    basis: NDArray
    dim: int
    status: str
    notes: list[str]


def invariant_subspace_basis(A: NDArray, m: int, near_real_tol: float = 1e-6) -> InvariantSubspaceBasis:
    """Real orthonormal basis for the invariant subspace spanned by A's m
    largest-|lambda| MODES: a real eigenvalue contributes 1 dimension; a
    complex-conjugate pair contributes the real 2-D invariant subspace
    span([Re(w), Im(w)]) and counts as ONE mode (not two), since it is a
    single oscillatory/decay mode of the real system. Orthonormalized via QR
    (equivalent to Gram-Schmidt on the same columns).

    Strict generalization of dominant_eigenmode: for m=1 on a real leading
    mode, S_1 = span(v*) exactly (same eigenvector, same convention).

    Parameters
    ----------
    A : (n, n) real system matrix
    m : number of modes to include
    near_real_tol : a complex-classified mode whose Im(w) norm is below this
        fraction of its Re(w) norm is treated as near-real (see
        InvariantSubspaceBasis.status)

    Returns
    -------
    InvariantSubspaceBasis
    """
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(np.abs(eigs))[::-1]
    basis: list[NDArray] = []
    notes: list[str] = []
    used: set[int] = set()
    count, i = 0, 0
    while count < m and i < len(order):
        idx = int(order[i])
        if idx in used:
            i += 1
            continue
        lam = eigs[idx]
        relative_imaginary_eigenvalue = abs(lam.imag) / (abs(lam.real) + 1e-12)
        if 0.0 < relative_imaginary_eigenvalue < near_real_tol:
            w = vecs[:, idx]
            basis.append(canonicalize_eigenvector_phase(w))
            used.add(idx)
            notes.append(
                f"mode {idx}: |Im(lambda)|/|Re(lambda)|="
                f"{relative_imaginary_eigenvalue:.2e} below "
                f"near_real_tol={near_real_tol:.1e}; oscillatory plane is "
                "not temporally resolvable, so only one canonical direction was kept"
            )
        elif abs(lam.imag) < 1e-8 * (abs(lam.real) + 1e-12):
            basis.append(vecs[:, idx].real)
            used.add(idx)
        else:
            conj_idx = int(np.argmin(np.abs(eigs - np.conj(lam))))
            partner_gap = np.abs(eigs[conj_idx] - np.conj(lam))
            w = vecs[:, idx]
            re_norm, im_norm = float(np.linalg.norm(w.real)), float(np.linalg.norm(w.imag))
            if im_norm < near_real_tol * (re_norm + 1e-12):
                basis.append(w.real)
                used.add(idx)
                notes.append(
                    f"mode {idx}: |Im(w)|/|Re(w)|={im_norm / (re_norm + 1e-12):.2e} "
                    f"below near_real_tol={near_real_tol:.1e}; kept only the real direction"
                )
            else:
                basis.append(w.real)
                basis.append(w.imag)
                used.add(idx)
                used.add(conj_idx)
                if partner_gap > 1e-6 * (abs(lam) + 1e-12):
                    notes.append(
                        f"mode {idx}: nearest candidate conjugate partner is {partner_gap:.2e} "
                        f"away from conj(lambda) (expected ~0); spectrum may be near-degenerate, "
                        f"treat this mode's basis directions as unreliable"
                    )
        count += 1
        i += 1
    B = np.column_stack(basis)
    Q, _ = np.linalg.qr(B)
    if notes:
        status = (
            "near_real_pair"
            if "only one canonical direction" in notes[0] or "kept only the real direction" in notes[0]
            else "unmatched_conjugate"
        )
    else:
        status = "ok"
    return InvariantSubspaceBasis(basis=Q, dim=Q.shape[1], status=status, notes=notes)


def subspace_alignment(basis: NDArray, b_hat: NDArray) -> float:
    """||P_S b_hat|| in [0, 1] for an orthonormal subspace basis S (columns
    of `basis`, e.g. from invariant_subspace_basis) and a direction b_hat —
    the strict generalization of |cos(b_hat, v*)| (equal at m=1, real mode:
    dominant_eigenmode's span)."""
    b = b_hat / (np.linalg.norm(b_hat) + 1e-12)
    return float(np.linalg.norm(basis.T @ b))


def stimulation_input_alignment(
    A: NDArray,
    components: NDArray,
    channel_weight: NDArray,
    v_star: NDArray,
    v_stable: NDArray,
    rng: np.random.Generator,
    gramian_horizon: int = 20,
    n_random_dirs: int = 20,
) -> dict:
    """Where a dataset's known, fixed stimulation electrode(s) sit relative
    to that participant's own fitted dynamics: does the input direction align
    with the native unstable/slowest-decaying mode (v_star), the most-stable
    mode (v_stable, a content-specificity-style control), or neither more
    than a random latent direction — and how much Gramian controllability
    energy does the fitted plant afford along that input.

    `channel_weight` is a (C,) real-space one-hot/averaged indicator of the
    stimulated channels (not yet projected into the latent space) — dataset-
    specific (which channels were actually stimulated), everything downstream
    of it is shared, reusable scoring logic (dynamics.fit_retention_dynamics
    provides A, components, v_star, v_stable).

    Parameters
    ----------
    A              : (k, k) fitted operator (from fit_retention_dynamics)
    components     : (C, k) PCA loadings (from fit_retention_dynamics)
    channel_weight : (C,) stimulation-electrode indicator in channel space
    v_star, v_stable : (k,) unit eigenvectors (from fit_retention_dynamics)
    rng            : random number generator (fixed-seed null draws)

    Returns
    -------
    dict: alignment_to_vstar, alignment_to_stable_mode,
          random_direction_alignment, gramian_trace
    """
    k = components.shape[1]
    b_lat = components.T @ channel_weight
    b_hat = b_lat / (np.linalg.norm(b_lat) + 1e-12)

    Wc = controllability_gramian(A, b_lat[:, None], T=gramian_horizon)
    random_dirs = rng.standard_normal((n_random_dirs, k))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-12

    return {
        "alignment_to_vstar": float(np.abs(b_hat @ v_star)),
        "alignment_to_stable_mode": float(np.abs(b_hat @ v_stable)),
        "random_direction_alignment": float(np.mean(np.abs(random_dirs @ b_hat))),
        "gramian_trace": float(np.trace(Wc)),
    }


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


def scalar_steady_state_kalman_error(a: float, process_variance: float, observation_variance: float) -> tuple[float, float]:
    """Steady-state Kalman filter error for the scalar system
    ``x(t+1) = a*x(t) + w``, ``y(t) = x(t) + v``, ``w ~ N(0, process_variance)``,
    ``v ~ N(0, observation_variance)``.

    Solved via the control/estimation DARE duality: the filtering Riccati
    recursion for the a priori (prediction) error covariance,
    ``P = a P a - a P (R + P)^-1 P a + Q``, is exactly :func:`dare_solve`'s
    control DARE ``P = AᵀPA - AᵀPB(R+BᵀPB)⁻¹BᵀPA + Q`` under ``A -> a`` (a
    scalar is its own transpose) and ``B -> 1`` (the observation matrix here
    is the identity), reusing the same solver rather than re-deriving the
    scalar quadratic root by hand.

    Returns
    -------
    (p_prior, p_post) : the a priori (one-step-ahead prediction) and a
        posteriori (filtered, after incorporating the current observation)
        steady-state error variance. ``p_post`` is the irreducible
        uncertainty about the current state given every past and current
        observation -- the estimation-error floor no feedback gain can
        shrink further.
    """
    a_mat = np.array([[a]])
    c_mat = np.array([[1.0]])
    q_mat = np.array([[process_variance]])
    r_mat = np.array([[observation_variance]])
    p_prior, _ = dare_solve(a_mat, c_mat, q_mat, r_mat)
    p = float(p_prior[0, 0])
    r = float(observation_variance)
    p_post = p * r / (p + r) if (p + r) > 0 else 0.0
    return p, p_post


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
