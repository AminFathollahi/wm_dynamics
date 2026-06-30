"""
geometry.py — Neural manifold geometry analysis.

Implements:
  - PCA decomposition (SVD-based, publication standard)
  - Participation ratio (intrinsic dimensionality; Abbott et al. 2011)
  - Principal angles between subspaces (Björck & Golub 1973)
  - Representational similarity analysis (RSA / RDM; Kriegeskorte et al. 2008)
  - Cross-temporal generalization decoding matrix
  - Subspace overlap metric
  - Time-resolved geometric biomarkers
  - Electrode capacity analysis for BCI feasibility

References
----------
Cunningham JP & Yu BM (2014) Dimensionality reduction for large-scale
  neural recordings. Nat Neurosci 17(11):1500-9.
Russo AA et al. (2018) Motor cortex embeds muscle-like commands in an
  untangled population response. Neuron 97(4):953-66.
Abbott LF et al. (2011) Interactions between intrinsic and
  stimulus-evoked activity in recurrent neural networks.
Panichello MF & Buschman TJ (2021) Shared mechanisms underlie the
  control of working memory and attention. Nature 592(7855):601-5.
Kriegeskorte N et al. (2008) Representational similarity analysis —
  connecting the branches of systems neuroscience. Front Syst Neurosci 2:4.
King JR & Dehaene S (2014) Characterizing the dynamics of mental
  representations: the temporal generalization method. Trends Cogn Sci 18(4):203-10.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ── PCA ────────────────────────────────────────────────────────────────────────

def pca_decompose(
    X: NDArray, n_components: int = 10
) -> tuple[NDArray, NDArray, NDArray]:
    """PCA via full SVD with centering.

    Parameters
    ----------
    X           : (N, D) — N observations, D features
    n_components: number of principal components to return

    Returns
    -------
    scores     : (N, k) — projections onto top-k PCs
    components : (D, k) — principal directions (columns = PCs)
    var_ratio  : (k,)   — fraction of variance per PC
    """
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, len(s))
    scores = U[:, :k] * s[:k]
    var_ratio = (s**2) / (s**2).sum()
    return scores, Vt[:k].T, var_ratio[:k]


def participation_ratio(eigenvalues: NDArray) -> float:
    """Participation ratio (PR): effective dimensionality of a covariance spectrum.

    PR = (Σλᵢ)² / Σλᵢ²

    PR = 1 when a single eigenvalue dominates (collapsed manifold).
    PR = D when all eigenvalues are equal (maximally uniform; D = n_dims).

    Equivalent to inverse participation ratio (IPR) inverted. Used as a
    dimensionality index in Abbott et al. 2011, Gao et al. 2017.

    Parameters
    ----------
    eigenvalues : (D,) — non-negative eigenvalues (e.g. singular values squared)

    Returns
    -------
    pr : scalar
    """
    lam = np.asarray(eigenvalues, dtype=float)
    lam = lam[lam > 0]
    return (lam.sum() ** 2) / (lam**2).sum()


def pca_participation_ratio(X: NDArray) -> float:
    """PR of the PCA covariance spectrum of X."""
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return participation_ratio(s**2)


# ── Principal angles ───────────────────────────────────────────────────────────

def principal_angles(A: NDArray, B: NDArray) -> NDArray:
    """Principal angles between subspaces spanned by columns of A and B.

    Algorithm (Björck & Golub 1973):
      1. QR-factorise A → Qₐ and B → Qᵦ (orthonormal bases)
      2. SVD of QₐᵀQᵦ → singular values σᵢ = cos(θᵢ)
      3. θᵢ = arccos(σᵢ)  ∈ [0, π/2]

    Interpretation: θ_min ≈ 0 → subspaces nearly parallel (representations
    overlap/interfere); θ_min ≈ π/2 → subspaces orthogonal (representations
    fully separated). See Panichello & Buschman 2021.

    Parameters
    ----------
    A : (N, k) — k basis vectors for subspace A (need not be orthonormal)
    B : (N, m) — m basis vectors for subspace B

    Returns
    -------
    angles : (min(k,m),) in radians
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    _, sv, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv)


# ── Time-resolved geometry ─────────────────────────────────────────────────────

def latent_trajectories(
    epochs: NDArray,
    n_components: int = 8,
) -> tuple[NDArray, NDArray, NDArray]:
    """Project epoch tensor to PCA latent space.

    Fits PCA on all trials pooled, then projects each trial individually.

    Parameters
    ----------
    epochs      : (N, T, C) — N trials, T time points, C channels
    n_components: latent space dimension

    Returns
    -------
    Z          : (N, T, k) — latent trajectories
    components : (C, k)    — PCA components (for reconstruction)
    var_ratio  : (k,)      — variance explained per PC
    """
    N, T, C = epochs.shape
    X_flat = epochs.reshape(-1, C)
    scores, comps, var_ratio = pca_decompose(X_flat, n_components)
    Z = scores.reshape(N, T, n_components)
    return Z, comps, var_ratio


def time_resolved_principal_angles(
    Z: NDArray,
    labels_a: NDArray,
    labels_b: NDArray,
    n_dims: int = 4,
) -> NDArray:
    """Compute principal angles at each time point between two trial groups.

    At each time step t:
      - Collect all latent vectors for group A trials: X_A = Z[labels_a, t, :]   (nA, k)
      - Collect all latent vectors for group B trials: X_B = Z[labels_b, t, :]   (nB, k)
      - Fit n_dims-dim PCA to each, get orthonormal bases Q_A, Q_B
      - Compute principal angles(Q_A, Q_B)

    Parameters
    ----------
    Z        : (N, T, k) — latent trajectories
    labels_a : (nA,) bool or int — trial indices for group A
    labels_b : (nB,) bool or int — trial indices for group B
    n_dims   : subspace dimension to project each group into before computing angles

    Returns
    -------
    theta    : (T, n_dims) — principal angles at each time point (radians)
    """
    _, T, k = Z.shape
    n_out = min(n_dims, k)
    theta = np.full((T, n_out), np.nan)

    for t in range(T):
        Xa = Z[labels_a, t, :]  # (nA, k)
        Xb = Z[labels_b, t, :]  # (nB, k)
        if Xa.shape[0] < n_dims or Xb.shape[0] < n_dims:
            continue
        # Orthonormal bases via PCA
        _, Qa, _ = pca_decompose(Xa, n_dims)  # (k, n_dims)
        _, Qb, _ = pca_decompose(Xb, n_dims)
        theta[t] = principal_angles(Qa, Qb)

    return theta


def time_resolved_pr(
    Z: NDArray,
    labels: NDArray,
) -> NDArray:
    """Compute participation ratio at each time point for a subset of trials.

    Parameters
    ----------
    Z      : (N, T, k)
    labels : (n,) bool or int

    Returns
    -------
    pr : (T,)
    """
    T = Z.shape[1]
    pr = np.zeros(T)
    for t in range(T):
        X = Z[labels, t, :]  # (n, k)
        Xc = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        pr[t] = participation_ratio(s**2)
    return pr


# ── Multi-subject geometric biomarker ─────────────────────────────────────────

def maintenance_window_geometry(
    epochs: NDArray,
    times: NDArray,
    task_id: NDArray,
    tgt_id: NDArray,
    maint_window: tuple[float, float] = (0.3, 1.4),
    n_components: int = 8,
    n_angle_dims: int = 4,
) -> dict:
    """Compute all geometric biomarkers during the maintenance window.

    For the 2-back condition, compares:
      - target trials (tgt_id=2) vs. non-target trials (tgt_id=1)
      - across load levels (0-back vs. 1-back vs. 2-back)

    Parameters
    ----------
    maint_window : (t_start, t_end) in seconds — delay/maintenance period

    Returns
    -------
    dict of biomarker arrays and metadata
    """
    maint_mask = (times >= maint_window[0]) & (times <= maint_window[1])
    Z, comps, var_ratio = latent_trajectories(epochs, n_components)

    results = {
        "var_ratio": var_ratio,
        "n_components": n_components,
        "maint_window": maint_window,
    }

    # Participation ratio per trial (mean over maintenance window)
    pr_per_trial = np.zeros(len(epochs))
    for i in range(len(epochs)):
        X = epochs[i][maint_mask, :]
        Xc = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        pr_per_trial[i] = participation_ratio(s**2)
    results["pr_per_trial"] = pr_per_trial

    # Time-resolved geometry for 2-back: target vs non-target
    twoback_tgt = (task_id == 2) & (tgt_id == 2)
    twoback_ntgt = (task_id == 2) & (tgt_id == 1)

    if twoback_tgt.sum() >= 5 and twoback_ntgt.sum() >= 5:
        theta = time_resolved_principal_angles(
            Z, twoback_tgt, twoback_ntgt, n_dims=n_angle_dims
        )
        results["theta_tgt_vs_ntgt"] = theta
        results["theta_min_tgt_vs_ntgt"] = theta[:, 0]

    # PR by load (mean over maintenance window per condition)
    results["pr_by_load"] = {}
    for load in [0, 1, 2]:
        mask = task_id == load
        if mask.sum() >= 3:
            results["pr_by_load"][load] = pr_per_trial[mask]

    return results


# ── Electrode capacity analysis ────────────────────────────────────────────────

def electrode_capacity_curve(
    epochs: NDArray,
    task_id: NDArray,
    channel_counts: list[int] | None = None,
    n_bootstrap: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """How does geometric biomarker quality scale with number of electrodes?

    At each electrode count n, randomly subsamples n channels, computes PR
    and principal angle separation between load conditions. Repeats n_bootstrap
    times to estimate mean ± SD.

    This analysis informs the minimum electrode count for a closed-loop BCI
    that uses manifold geometry as its state observable.

    Parameters
    ----------
    epochs        : (N, T, C)
    task_id       : (N,)
    channel_counts: list of n_ch values to test; defaults to [2,4,8,16,32,C]
    n_bootstrap   : resamples per n_ch

    Returns
    -------
    dict:
      n_channels   : list of tested counts
      pr_mean      : (n_counts,) mean PR contrast (2-back minus 0-back)
      pr_std        : (n_counts,)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N, T, C = epochs.shape
    if channel_counts is None:
        channel_counts = sorted(set([2, 4, 8, 16, min(32, C), C]))

    pr_means, pr_stds = [], []

    mask_0 = task_id == 0
    mask_2 = task_id == 2

    maint_t = slice(T // 4, 3 * T // 4)  # rough maintenance window

    for n_ch in channel_counts:
        contrasts = []
        for _ in range(n_bootstrap):
            ch_idx = rng.choice(C, size=n_ch, replace=False)
            sub = epochs[:, :, ch_idx]

            def _pr(mask):
                X = sub[mask][:, maint_t, :].reshape(-1, n_ch)
                _, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
                return participation_ratio(s**2)

            if mask_0.sum() >= 3 and mask_2.sum() >= 3:
                contrasts.append(_pr(mask_2) - _pr(mask_0))

        pr_means.append(np.mean(contrasts) if contrasts else np.nan)
        pr_stds.append(np.std(contrasts) if contrasts else np.nan)

    return {
        "n_channels": channel_counts,
        "pr_contrast_mean": np.array(pr_means),
        "pr_contrast_std": np.array(pr_stds),
    }


# ── Representational Similarity Analysis ───────────────────────────────────────

def representational_dissimilarity_matrix(
    X: NDArray,
    metric: str = "correlation",
) -> NDArray:
    """Compute a representational dissimilarity matrix (RDM).

    Each entry (i, j) is the dissimilarity between items i and j in the
    representation space X. Diagonal is zero.

    Metrics
    -------
    'correlation'  : 1 - Pearson r  (Kriegeskorte et al. 2008; most common in RSA)
    'euclidean'    : ‖xᵢ - xⱼ‖₂
    'cosine'       : 1 - cos(θ)

    Parameters
    ----------
    X      : (N, D) — N items × D features (e.g. N stimuli × C neural channels)
    metric : dissimilarity metric

    Returns
    -------
    rdm : (N, N) — symmetric, zero diagonal
    """
    N = X.shape[0]
    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]  # (N, N, D)
        return np.sqrt((diff**2).sum(axis=-1))

    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12

    if metric in ("correlation", "cosine"):
        Xn = (Xc if metric == "correlation" else X) / (
            norms if metric == "correlation" else
            (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        )
        sim = Xn @ Xn.T
        np.fill_diagonal(sim, 1.0)
        return 1.0 - np.clip(sim, -1.0, 1.0)

    raise ValueError(f"Unknown metric '{metric}'.")


def rsa_compare(
    rdm1: NDArray,
    rdm2: NDArray,
) -> float:
    """Compare two RDMs via Spearman rank correlation of their lower triangles.

    This is the standard RSA metric (Kriegeskorte et al. 2008). The lower
    triangle excludes self-comparisons (diagonal = 0 by construction).

    Parameters
    ----------
    rdm1, rdm2 : (N, N) — symmetric dissimilarity matrices

    Returns
    -------
    rho : Spearman r in [-1, 1]
    """
    n = rdm1.shape[0]
    idx = np.tril_indices(n, k=-1)
    v1 = rdm1[idx].ravel()
    v2 = rdm2[idx].ravel()

    def _rank(v: NDArray) -> NDArray:
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v))
        return ranks

    r1, r2 = _rank(v1), _rank(v2)
    r1 -= r1.mean(); r2 -= r2.mean()
    denom = np.sqrt((r1**2).sum() * (r2**2).sum()) + 1e-12
    return float((r1 * r2).sum() / denom)


def cross_temporal_generalization(
    Z: NDArray,
    labels: NDArray,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Cross-temporal generalization (CTG) decoding matrix.

    Train a linear SVM at each time point t₁; test at every time point t₂.
    The (T, T) result is called the "temporal generalization matrix"
    (King & Dehaene 2014, Trends Cogn Sci).

    Diagonal = standard time-resolved decoding accuracy.
    Off-diagonal generalization > chance → temporally stable code.
    Narrow diagonal → rapidly evolving code (dynamic coding).

    Uses k-fold cross-validation; returns mean AUC across folds.

    Parameters
    ----------
    Z       : (N, T, k) — latent trajectories
    labels  : (N,) — binary or integer class labels
    n_splits: k-fold CV splits

    Returns
    -------
    auc_matrix : (T, T) — AUC at each (train_t, test_t) pair
    """
    if rng is None:
        rng = np.random.default_rng(0)

    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    N, T, k = Z.shape
    labels = np.asarray(labels)
    auc_mat = np.full((T, T), np.nan)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=int(rng.integers(0, 1000)))
    folds = list(skf.split(np.zeros(N), labels))

    for t1 in range(T):
        X_train_t1 = Z[:, t1, :]  # (N, k)
        fold_mats = np.zeros((len(folds), T))

        for fi, (tr_idx, te_idx) in enumerate(folds):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train_t1[tr_idx])
            clf = LinearSVC(C=1.0, max_iter=2000)
            clf.fit(X_tr, labels[tr_idx])

            for t2 in range(T):
                X_te = scaler.transform(Z[te_idx, t2, :])
                scores = clf.decision_function(X_te)
                y_te = labels[te_idx]
                if len(np.unique(y_te)) < 2:
                    continue
                try:
                    fold_mats[fi, t2] = roc_auc_score(y_te, scores)
                except Exception:
                    pass

        auc_mat[t1] = fold_mats.mean(axis=0)

    return auc_mat


def geometric_drift(
    Z: NDArray,
    task_id: NDArray,
    times: NDArray,
    maint_window: tuple[float, float] = (0.3, 1.4),
) -> NDArray:
    """Per-trial Euclidean drift from the condition centroid in latent space.

    For each trial, compute the mean distance from the condition's centroid
    trajectory during the maintenance window. High drift → trajectory has
    wandered far from the typical maintenance attractor (predicts longer RT
    or higher error rate; Hypothesis H5d in PAPER_DRAFT).

    Parameters
    ----------
    Z        : (N, T, k) — latent trajectories
    task_id  : (N,) — condition labels (0, 1, 2 for N-back load)
    times    : (T,) — time axis in seconds
    maint_window : (t_start, t_end) — maintenance window in seconds

    Returns
    -------
    drift : (N,) — mean Euclidean distance from condition centroid
    """
    maint = (times >= maint_window[0]) & (times <= maint_window[1])
    Z_m = Z[:, maint, :]   # (N, T_m, k)
    N = len(Z_m)
    drift = np.zeros(N)

    for cond in np.unique(task_id):
        mask = task_id == cond
        if mask.sum() < 2:
            continue
        centroid = Z_m[mask].mean(axis=0)   # (T_m, k)
        diff = Z_m[mask] - centroid          # (n_cond, T_m, k)
        drift[mask] = np.sqrt((diff**2).sum(axis=2)).mean(axis=1)

    return drift


def subspace_overlap(
    A: NDArray,
    B: NDArray,
) -> float:
    """Normalised subspace overlap between two subspaces.

    Computed as the mean of cos²(θᵢ), where θᵢ are the principal angles
    between A and B. Range: [0, 1].

    Interpretation:
      0 → fully orthogonal subspaces (no shared directions)
      1 → identical subspaces

    Parameters
    ----------
    A : (N, k)
    B : (N, m)

    Returns
    -------
    overlap : scalar in [0, 1]
    """
    angles = principal_angles(A, B)
    return float(np.mean(np.cos(angles) ** 2))
