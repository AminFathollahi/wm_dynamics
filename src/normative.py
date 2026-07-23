"""
normative.py — STEP C: why the geometry takes the form it does (a stable
context axis + a rotating memorandum), not just that it does.

Principle: a linear maintenance operator scored by the SAME quantity for
both roles — temporal self-overlap <M^t x, x> of a unit-norm state with its
own earlier trajectory — should make opposite choices for the two roles.
Context must stay cross-time decodable at a FIXED readout axis, i.e. high
self-overlap (near a stationary point). A memorandum that must not interfere
with itself (or a second item loaded into the same subspace later in the
trial) should keep LOW self-overlap over the delay, which for a norm-
preserving 2D block is minimized away from the identity, i.e. by rotating
(Libby & Buschman 2021; interference-reducing rotational coding).

Two public functions:
  optimize_maintenance_operator — abstract (theta, lambda_c) optimum for a
    given horizon T.
  embed_maintenance_operator    — places that 2-parameter solution into a
    real k-dim latent space using a real context axis and memorandum seed
    vector, for comparison against a fitted A's v*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def _rotation_block(theta: float) -> NDArray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _loss(params: NDArray, T: int, energy_eps: float) -> float:
    theta, lambda_c = params
    t = np.arange(1, T + 1)
    interference = np.mean(np.cos(theta * t) ** 2)          # content: minimize
    retention = np.mean(np.clip(lambda_c, 0.0, 1.0) ** (2 * t))  # context: maximize
    energy = energy_eps * ((1.0 - lambda_c) ** 2 + theta ** 2)
    return float(interference - retention + energy)


def optimize_maintenance_operator(T: int, energy_eps: float = 0.01, n_starts: int = 12) -> dict:
    """Find (theta, lambda_c) minimizing content self-overlap (interference)
    while maximizing context self-overlap (retention) over a T-step delay,
    under a small energy penalty. Multi-start (theta is periodic and the
    landscape is non-convex) local optimization, keeping the best.

    Returns
    -------
    dict: theta (rad/step), lambda_c, loss
    """
    best = None
    rng = np.random.default_rng(0)
    theta0s = rng.uniform(0.05, np.pi - 0.05, n_starts)
    for theta0 in theta0s:
        res = minimize(_loss, x0=[theta0, 0.9], args=(T, energy_eps),
                       bounds=[(1e-6, np.pi), (0.0, 1.0)], method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    theta, lambda_c = best.x
    return {"theta": float(theta), "lambda_c": float(lambda_c), "loss": float(best.fun)}


def embed_maintenance_operator(
    theta: float, lambda_c: float, c_axis: NDArray, x0_mem: NDArray, k: int,
    lambda_rest: float = 0.3,
) -> NDArray:
    """Embed the abstract (theta, lambda_c) solution into a real k-dim
    (k >= 3) latent space as a sum of three mutually orthogonal invariant
    subspaces: context (1D, span(b1) = c_axis, eigenvalue lambda_c),
    memorandum (2D, span(b2, b3) with b2 = x0_mem's component orthogonal to
    b1, rotated by theta), and the remaining k-3 dims (a small decay
    `lambda_rest` -- the "energy penalty": unclaimed activity does not
    persist for free).

    Returns
    -------
    M : (k, k) real
    """
    if k < 3:
        raise ValueError("embed_maintenance_operator needs k >= 3 (1D context + 2D memorandum)")
    c_axis = np.asarray(c_axis, dtype=float)
    x0_mem = np.asarray(x0_mem, dtype=float)
    b1 = c_axis / (np.linalg.norm(c_axis) + 1e-12)
    resid = x0_mem - (x0_mem @ b1) * b1
    resid_norm = np.linalg.norm(resid)
    if resid_norm < 1e-8:
        rng = np.random.default_rng(0)
        resid = rng.standard_normal(k)
        resid = resid - (resid @ b1) * b1
        resid_norm = np.linalg.norm(resid)
    b2 = resid / resid_norm

    rng = np.random.default_rng(0)
    extra = rng.standard_normal((k, k - 2))
    full, _ = np.linalg.qr(np.column_stack([b1, b2, extra]))
    basis = full[:, :k]
    basis[:, 0], basis[:, 1] = b1, b2   # QR already re-orthonormalized b2 against b1

    P_ctx = np.outer(b1, b1)
    mem_basis = basis[:, 1:3]           # b2 and one arbitrary orthogonal complement
    P_mem = mem_basis @ mem_basis.T
    P_rest = np.eye(k) - P_ctx - P_mem

    mem_contrib = mem_basis @ _rotation_block(theta) @ mem_basis.T
    return lambda_c * P_ctx + mem_contrib + lambda_rest * P_rest
