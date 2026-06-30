"""
neuroai.py — NeuroAI analysis: representational alignment and model-brain comparison.

Implements:
  - Centered Kernel Alignment (CKA; Kornblith et al. 2019)
  - Orthogonal Procrustes alignment between representation spaces
  - Noise ceiling for RSA (Nili et al. 2014)
  - Cross-subject RSA and model-brain comparison
  - Representational geometry metrics for comparing ANNs to neural data

These tools ask whether an artificial neural network (ANN) trained on
N-back or sequential prediction tasks learns representations geometrically
similar to those found in human prefrontal cortex — the NeuroAI hypothesis
(Yamins & DiCarlo 2016; Schrimpf et al. 2021).

References
----------
Kornblith S et al. (2019) Similarity of neural network representations
  revisited. ICML. arXiv:1905.00414.
Kriegeskorte N et al. (2008) Representational similarity analysis.
  Front Syst Neurosci 2:4.
Nili H et al. (2014) A toolbox for representational similarity analysis.
  PLoS Comput Biol 10(4):e1003553.
Yamins DLK & DiCarlo JJ (2016) Using goal-driven deep learning models to
  understand sensory cortex. Nat Neurosci 19(3):356-65.
Procrustes: Schönemann PH (1966) A generalised solution of the orthogonal
  procrustes problem. Psychometrika 31(1):1-10.
"""

from __future__ import annotations

import sys
import os
import numpy as np
from numpy.typing import NDArray

# Allow `from geometry import ...` to work whether src/ is on sys.path or not.
_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


# ── Centered Kernel Alignment ──────────────────────────────────────────────────

def _hsic(K: NDArray, L: NDArray) -> float:
    """Hilbert-Schmidt Independence Criterion (centred, unbiased form).

    HSIC(K, L) = ⟨K_c, L_c⟩_F / (n-1)²
    where K_c = H K H  (H = I - 11ᵀ/n)
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float((Kc * Lc).sum()) / (n - 1) ** 2


def centered_kernel_alignment(
    X: NDArray,
    Y: NDArray,
) -> float:
    """Linear CKA between representations X and Y.

    CKA(X, Y) = HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T, XX^T) * HSIC(YY^T, YY^T))

    Range [0, 1]:  1 → identical (up to orthogonal transform + scaling)
                   0 → completely dissimilar

    Invariant to orthogonal transforms and isotropic scaling, but NOT to
    non-isotropic scaling — appropriate for comparing layer activations
    where scale is arbitrary but geometry matters.

    Parameters
    ----------
    X : (N, d1) — N examples × d1 features (e.g. neural data; N trials)
    Y : (N, d2) — N examples × d2 features (e.g. model activations)

    Returns
    -------
    cka : float in [0, 1]
    """
    K = X @ X.T
    L = Y @ Y.T
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    denom = np.sqrt(hsic_kk * hsic_ll) + 1e-12
    return float(hsic_kl / denom)


def cka_timecourse(
    Z: NDArray,
    Y: NDArray,
) -> NDArray:
    """Linear CKA at each time point along a trial-averaged trajectory.

    Parameters
    ----------
    Z : (N, T, d1) — neural latent trajectories
    Y : (N, T, d2) — model latent trajectories (same N, T)

    Returns
    -------
    cka_t : (T,) — CKA at each time point
    """
    _, T, _ = Z.shape
    return np.array([
        centered_kernel_alignment(Z[:, t, :], Y[:, t, :])
        for t in range(T)
    ])


# ── Orthogonal Procrustes alignment ───────────────────────────────────────────

def procrustes_align(
    X: NDArray,
    Y: NDArray,
) -> dict:
    """Orthogonal Procrustes alignment: find R minimising ‖X - Y R‖_F.

    After aligning, the disparity = ‖X - Y R‖_F² / ‖X‖_F² measures how
    well Y explains X after optimal rotation. Disparity = 0 → perfect
    structural match.

    Algorithm (Schönemann 1966):
      SVD(Y^T X) = U Σ V^T  →  R = U V^T

    Parameters
    ----------
    X : (N, k) — target representation (fixed)
    Y : (N, k) — source representation (to be rotated)

    Returns
    -------
    dict:
      R          : (k, k) — orthogonal rotation matrix
      Y_aligned  : (N, k) — Y @ R
      disparity  : float  — normalised Frobenius distance after alignment
    """
    M = Y.T @ X
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    Y_aligned = Y @ R
    disparity = float(np.linalg.norm(X - Y_aligned, "fro") ** 2 /
                      (np.linalg.norm(X, "fro") ** 2 + 1e-12))
    return {"R": R, "Y_aligned": Y_aligned, "disparity": disparity}


# ── RSA noise ceiling ──────────────────────────────────────────────────────────

def noise_ceiling_rsa(
    rdms: NDArray,
) -> tuple[float, float]:
    """Noise ceiling for RSA: theoretical upper and lower bounds on Spearman r.

    Given RDMs from N subjects, any model's RSA correlation is bounded by
    the inter-subject reliability. The noise ceiling quantifies how much of
    the RSA signal is attributable to the true underlying structure vs noise.

    Upper bound: mean Spearman r of each subject with the mean RDM of ALL subjects.
    Lower bound: mean Spearman r of each subject with the mean RDM of OTHER subjects.

    (Nili et al. 2014, PLoS Comput Biol)

    Parameters
    ----------
    rdms : (N, n_items, n_items) — one RDM per subject

    Returns
    -------
    (lower, upper) noise ceiling bounds
    """
    from geometry import rsa_compare

    N = rdms.shape[0]
    uppers, lowers = [], []

    mean_rdm = rdms.mean(axis=0)
    for i in range(N):
        uppers.append(rsa_compare(rdms[i], mean_rdm))
        loo_mean = (rdms.sum(axis=0) - rdms[i]) / (N - 1)
        lowers.append(rsa_compare(rdms[i], loo_mean))

    return float(np.mean(lowers)), float(np.mean(uppers))


# ── Cross-subject RSA ──────────────────────────────────────────────────────────

def cross_subject_rsa(
    rdms_by_subject: dict[str, NDArray],
    model_rdm: NDArray | None = None,
) -> dict:
    """Compute RSA between subjects and optionally against a model RDM.

    Parameters
    ----------
    rdms_by_subject : dict mapping subject ID → (n_items, n_items) RDM
    model_rdm       : (n_items, n_items) — e.g. from an RNN trained on N-back

    Returns
    -------
    dict:
      pairwise_r : (N, N) — Spearman r between all subject pairs
      model_r    : (N,) or None — each subject's RSA with the model
      mean_model_r : float or None
    """
    from geometry import rsa_compare

    subjects = list(rdms_by_subject.keys())
    N = len(subjects)
    pairwise = np.full((N, N), np.nan)

    for i, si in enumerate(subjects):
        for j, sj in enumerate(subjects):
            if i == j:
                pairwise[i, j] = 1.0
                continue
            pairwise[i, j] = rsa_compare(
                rdms_by_subject[si], rdms_by_subject[sj]
            )

    result = {
        "subjects": subjects,
        "pairwise_r": pairwise,
        "mean_pairwise_r": float(np.nanmean(pairwise[np.tril_indices(N, k=-1)])),
        "model_r": None,
        "mean_model_r": None,
    }

    if model_rdm is not None:
        model_rs = np.array([
            rsa_compare(rdms_by_subject[s], model_rdm) for s in subjects
        ])
        result["model_r"] = model_rs
        result["mean_model_r"] = float(model_rs.mean())

    return result


# ── Representational geometry summary ─────────────────────────────────────────

def geometry_summary(
    Z_neural: NDArray,
    Z_model: NDArray,
    condition_labels: NDArray,
) -> dict:
    """High-level comparison of neural vs model representational geometry.

    Computes CKA, Procrustes disparity, and between-condition RSA for a
    single time-point or maintenance-window-averaged representation.

    Parameters
    ----------
    Z_neural : (N, d_n) — neural latent vectors (N trials)
    Z_model  : (N, d_m) — model latent vectors (same N trials)
    condition_labels : (N,) — condition labels for constructing RDMs

    Returns
    -------
    dict with cka, procrustes_disparity, neural_rdm, model_rdm, rsa_r
    """
    from geometry import representational_dissimilarity_matrix, rsa_compare

    cka = centered_kernel_alignment(Z_neural, Z_model)
    proc = procrustes_align(Z_neural, Z_model) if Z_neural.shape[1] == Z_model.shape[1] else None

    rdm_n = representational_dissimilarity_matrix(Z_neural)
    rdm_m = representational_dissimilarity_matrix(Z_model)
    rsa_r = rsa_compare(rdm_n, rdm_m)

    return {
        "cka": cka,
        "procrustes_disparity": proc["disparity"] if proc else None,
        "rsa_r": rsa_r,
        "neural_rdm": rdm_n,
        "model_rdm": rdm_m,
    }
