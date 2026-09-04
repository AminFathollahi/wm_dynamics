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

import sys
import os
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
from statistics import permutation_pvalue


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


def fit_frozen_pca(X_train: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
    """Fit a PCA basis on training data only, for later application to held-out data.

    Returns (mean, components) with components (D, k); apply with
    `apply_frozen_pca`. Needed whenever a held-out condition (e.g. a
    perturbation arm) must be scored in the same latent as a train-only fit,
    which plain `pca_decompose` / `latent_trajectories` (fit and project in
    one call) cannot do.
    """
    mean = X_train.mean(axis=0)
    _, components, _ = pca_decompose(X_train, n_components)
    return mean, components


def apply_frozen_pca(X: NDArray, mean: NDArray, components: NDArray) -> NDArray:
    return (X - mean) @ components


def phase_scrambled_null(latent: NDArray, rng: np.random.Generator) -> NDArray:
    """Per-trial, per-channel Fourier phase randomization.

    Replaces each channel's phase spectrum with independent uniform noise
    while keeping its magnitude spectrum (so total power and the coarse
    autocorrelation/roughness scale are preserved); this destroys any
    consistent temporal structure -- confinement, directed drift, periodic
    structure -- while keeping the same marginal power as the real signal,
    making it a stricter null than a plain time shuffle for detecting
    structured (non-stationary-looking) dynamics.

    Parameters
    ----------
    latent : (n_trials, n_bins, k)
    """
    latent = np.asarray(latent, dtype=float)
    n_trials, n_bins, k = latent.shape
    scrambled = np.empty_like(latent)
    freqs = np.fft.rfft(latent, axis=1)
    magnitude = np.abs(freqs)
    n_freqs = freqs.shape[1]
    random_phase = rng.uniform(0, 2 * np.pi, size=(n_trials, n_freqs, k))
    random_phase[:, 0, :] = 0.0  # keep the DC component real
    if n_bins % 2 == 0:
        random_phase[:, -1, :] = 0.0  # keep the Nyquist component real
    scrambled_freqs = magnitude * np.exp(1j * random_phase)
    scrambled = np.fft.irfft(scrambled_freqs, n=n_bins, axis=1)
    return scrambled


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


def twonn_dimension(X: NDArray, discard_fraction: float = 0.1) -> float:
    """TwoNN intrinsic-dimensionality estimator (Facco et al. 2017, Sci Rep 7:12140).

    Uses only the first two nearest-neighbour distances per point, so it is
    far less sensitive to curvature/non-uniform density over a large
    neighbourhood than a full-kNN estimator. Under local uniform density,
    mu_i = r2_i / r1_i follows a Pareto(d) distribution, so a straight line
    through the origin in (log mu, -log(1-F_empirical(mu))) has slope d.
    The largest `discard_fraction` of mu values (least reliable, since the
    empirical CDF there is close to 1) is dropped before the fit, as in the
    original paper.

    Parameters
    ----------
    X : (N, D) point cloud, N >= 3
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    dist = squareform(pdist(X))
    np.fill_diagonal(dist, np.inf)
    nn = np.sort(dist, axis=1)[:, :2]
    mu = nn[:, 1] / np.maximum(nn[:, 0], 1e-12)
    valid = np.isfinite(mu) & (mu > 1.0)
    order = np.argsort(mu[valid])
    mu_sorted = mu[valid][order]
    n_valid = len(mu_sorted)
    # F_empirical(mu_(i)) = i / n_valid for the i-th order statistic (1-indexed);
    # the top discard_fraction is dropped before fitting since F is close to 1
    # there and -log(1-F) blows up on sampling noise (Facco et al. 2017 Sec. 2).
    f_emp = np.arange(1, n_valid + 1) / n_valid
    keep = int(n_valid * (1.0 - discard_fraction))
    mu_fit, f_fit = mu_sorted[:keep], f_emp[:keep]
    if len(mu_fit) < 3:
        return float("nan")
    x = np.log(mu_fit)
    y = -np.log(1.0 - f_fit)
    denom = float(np.sum(x * x))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def levina_bickel_mle_dimension(X: NDArray, k: int = 20) -> float:
    """Levina & Bickel (2004, NeurIPS) maximum-likelihood dimension estimator.

    For each point i, with r_1 <= ... <= r_k its distances to its k nearest
    neighbours, the local MLE is d_hat_i = [(k-1)^-1 * sum_{j=1}^{k-1}
    log(r_k / r_j)]^-1. Following Levina & Bickel's own recommendation, the
    reported estimate averages 1/d_hat_i across points before inverting
    (not the per-point d_hat_i directly, which is upward-biased).
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = min(k, n - 1)
    if k < 2:
        return float("nan")
    dist = squareform(pdist(X))
    np.fill_diagonal(dist, np.inf)
    nn = np.sort(dist, axis=1)[:, :k]
    r_k = nn[:, -1:]
    ratios = np.log(np.maximum(r_k, 1e-12) / np.maximum(nn[:, :-1], 1e-12))
    inv_d_hat = ratios.mean(axis=1)
    inv_d_hat = inv_d_hat[np.isfinite(inv_d_hat) & (inv_d_hat > 0)]
    if len(inv_d_hat) == 0:
        return float("nan")
    return float(1.0 / inv_d_hat.mean())


def correlation_dimension(X: NDArray, n_radii: int = 20) -> float:
    """Grassberger-Procaccia correlation dimension.

    C(r) = fraction of point pairs within distance r; the correlation
    dimension is the OLS slope of log C(r) on log r, fit over a log-spaced
    radius grid spanning the 5th-50th percentile of the pairwise-distance
    distribution (a standard choice of the mid-range "scaling region",
    avoiding both the noisy near-zero tail and the saturating large-r tail).
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    pairwise = pdist(X)
    pairwise = pairwise[pairwise > 0]
    if len(pairwise) < 10:
        return float("nan")
    r_lo, r_hi = np.percentile(pairwise, [5, 50])
    if r_lo <= 0 or r_hi <= r_lo:
        return float("nan")
    radii = np.geomspace(r_lo, r_hi, n_radii)
    counts = np.array([(pairwise < r).sum() for r in radii], dtype=float)
    total_pairs = n * (n - 1) / 2.0
    c_r = counts / total_pairs
    valid = c_r > 0
    if valid.sum() < 3:
        return float("nan")
    x = np.log(radii[valid])
    y = np.log(c_r[valid])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def pca_participation_ratio(X: NDArray) -> float:
    """PR of the PCA covariance spectrum of X."""
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return participation_ratio(s**2)


def parallel_analysis(
    X: NDArray,
    n_surrogate: int = 200,
    percentile: float = 95.0,
    rng: np.random.Generator | None = None,
    n_timepoints_per_trial: int | None = None,
) -> int:
    """Horn's parallel analysis: number of PCs whose eigenvalue exceeds the
    upper percentile of eigenvalues from surrogate data.

    Horn's classic surrogate independently permutes the rows within every
    column, which assumes the rows are exchangeable. That is fine for i.i.d.
    observations, but trial-by-time rows are autocorrelated in time within a
    trial, so a full row permutation destroys that autocorrelation and yields
    a surrogate noise floor that is far too low — the procedure then
    OVER-RETAINS components on real (autocorrelated) data.

    When `n_timepoints_per_trial` is given, rows of X are treated as
    contiguous trial blocks of that length (the layout produced by e.g.
    `epochs.transpose(0, 2, 1).reshape(-1, C)`), and each surrogate instead
    circularly shifts each column WITHIN its own trial block by an
    independent random offset. This still destroys cross-column covariance
    and cross-trial alignment (so it still tests "are these components above
    the chance level of unstructured data"), but preserves each trial's own
    within-trial autocorrelation, giving Horn's exchangeability assumption a
    defensible trial-level analogue. This generally raises the surrogate
    noise floor and retains fewer components than the unblocked row
    permutation, so k selected before this fix may differ from k selected
    after it. When `n_timepoints_per_trial` is None, rows are assumed already
    independent (e.g. one row per trial) and the original full row
    permutation is used.

    Parameters
    ----------
    X                      : (N, D) — N observations, D features
    n_surrogate            : number of surrogates
    percentile             : surrogate-eigenvalue percentile used as the
                              retention threshold
    n_timepoints_per_trial : trial block length in rows, if rows are
                              trial-by-time samples (contiguous per trial);
                              None if rows are already independent
                              observations

    Returns
    -------
    k : number of retained components (>= 0)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Xc = X - X.mean(axis=0)
    N, D = Xc.shape
    obs = np.linalg.svd(Xc, full_matrices=False, compute_uv=False) ** 2

    if n_timepoints_per_trial is not None:
        block = n_timepoints_per_trial
        if N % block != 0:
            raise ValueError("N must be a multiple of n_timepoints_per_trial")
        n_trials = N // block

    surr = np.empty((n_surrogate, len(obs)))
    for i in range(n_surrogate):
        if n_timepoints_per_trial is None:
            Xs = np.column_stack([Xc[rng.permutation(N), j] for j in range(D)])
        else:
            Xs = np.empty_like(Xc)
            for j in range(D):
                trial_blocks = Xc[:, j].reshape(n_trials, block)
                shifts = rng.integers(0, block, size=n_trials)
                Xs[:, j] = np.array(
                    [np.roll(trial_blocks[t], shifts[t]) for t in range(n_trials)]
                ).reshape(-1)
        s = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False, compute_uv=False)
        surr[i, : len(s)] = s**2
    thresh = np.percentile(surr, percentile, axis=0)
    return int(np.sum(obs > thresh))


def select_latent_dim(
    X: NDArray,
    method: str = "cv_pr",
    rng: np.random.Generator | None = None,
) -> dict:
    """Data-driven latent dimensionality, so the latent dim is not a convention.

    Two principled selectors, both reported:
      cv_pr (primary)   round(cross-validated participation ratio) of the pooled
                        maintenance covariance in the native channel/unit space
                        (reuses spatiotemporal_participation_ratio — the same
                        cross-validated PR the paper uses as its intrinsic-
                        dimensionality index, Eq. eq:pr). Tying k to PR makes the
                        latent dimension internally consistent with the null result.
      parallel_analysis Horn's parallel analysis (parallel_analysis above).

    Parameters
    ----------
    X      : (N, C, T) raw per-trial channel/unit responses, matching
             spatiotemporal_participation_ratio's layout.
    method : which selector's k is returned as ``k`` ("cv_pr" or "parallel_analysis").

    Returns
    -------
    dict: k (int, from `method`), k_cv_pr, k_parallel_analysis, cv_pr (float),
          n_channels, method
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N, C, T = X.shape
    pr = spatiotemporal_participation_ratio(X, rng=rng)
    cv_pr = pr["pr_cv"]
    k_cv = int(np.clip(round(cv_pr), 1, C)) if np.isfinite(cv_pr) else C

    # Parallel analysis on the same pooled (observations x channels) matrix that
    # spatiotemporal_participation_ratio SVDs, so both selectors see one space.
    # Rows are contiguous per-trial blocks of length T (trial-major from the
    # transpose+reshape below), so the surrogate null respects trial structure.
    X_pooled = X.transpose(0, 2, 1).reshape(-1, C)
    k_pa = int(np.clip(
        parallel_analysis(X_pooled, rng=rng, n_timepoints_per_trial=T), 1, C
    ))

    k = {"cv_pr": k_cv, "parallel_analysis": k_pa}.get(method)
    if k is None:
        raise ValueError("method must be 'cv_pr' or 'parallel_analysis'.")
    return {
        "k": k,
        "k_cv_pr": k_cv,
        "k_parallel_analysis": k_pa,
        "cv_pr": float(cv_pr),
        "n_channels": int(C),
        "method": method,
    }


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
    Z = scores.reshape(N, T, scores.shape[1])
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
    """Per-trial Euclidean drift from the OUT-OF-FOLD condition centroid in
    latent space.

    For each trial, compute the mean distance from the condition's centroid
    trajectory during the maintenance window, where the centroid for trial i
    is computed from every OTHER trial in its condition (never including
    trial i itself). Including a trial in its own centroid mechanically
    shrinks that trial's distance to the centroid — and shrinks it more in
    conditions with fewer trials — which biases drift comparisons across
    conditions of different sizes; the leave-one-out centroid removes that
    bias. High drift → trajectory has wandered far from the typical
    maintenance attractor (predicts longer RT or higher error rate;
    Hypothesis H5d in PAPER_DRAFT).

    Parameters
    ----------
    Z        : (N, T, k) — latent trajectories
    task_id  : (N,) — condition labels (0, 1, 2 for N-back load)
    times    : (T,) — time axis in seconds
    maint_window : (t_start, t_end) — maintenance window in seconds

    Returns
    -------
    drift : (N,) — mean Euclidean distance from the leave-one-out condition
        centroid. Trials in a condition with fewer than 2 trials have no
        out-of-fold centroid and are reported as NaN, not silently zero.
    """
    maint = (times >= maint_window[0]) & (times <= maint_window[1])
    Z_m = Z[:, maint, :]   # (N, T_m, k)
    N = len(Z_m)
    drift = np.full(N, np.nan)

    for cond in np.unique(task_id):
        mask = task_id == cond
        n_cond = int(mask.sum())
        if n_cond < 2:
            continue
        group_sum = Z_m[mask].sum(axis=0)              # (T_m, k)
        loo_centroid = (group_sum[None] - Z_m[mask]) / (n_cond - 1)  # (n_cond, T_m, k)
        diff = Z_m[mask] - loo_centroid                  # (n_cond, T_m, k)
        drift[mask] = np.sqrt((diff**2).sum(axis=2)).mean(axis=1)

    return drift


def distance_to_attractor(
    train_state: NDArray,
    train_labels: NDArray,
    test_state: NDArray,
    test_labels: NDArray,
) -> NDArray:
    """Normalized distance-to-attractor (Daume et al. 2025, following
    Kaminski et al. 2017).

    For each held-out trial and time point: the Euclidean distance to its
    own-condition centroid divided by the mean distance to every other
    condition's centroid. DA < 1 means the state is nearest its own
    (correct) attractor. Centroids are computed from TRAINING trials only —
    computing them from all trials (including the one being scored) is
    exactly the selection bias Daume et al. control for, and produces
    DA < 1 trivially.

    Parameters
    ----------
    train_state, test_state : (n_trials, n_time, n_dims) latent trajectories
    train_labels, test_labels : (n_trials,) condition labels

    Returns
    -------
    da : (n_test_trials, n_time) — NaN for a test trial whose condition has
        no training centroid, or when fewer than 2 conditions have a
        training centroid (no "other" to normalize against).
    """
    condition_labels = np.unique(train_labels)
    centroids = np.stack([
        train_state[train_labels == label].mean(axis=0) for label in condition_labels
    ])  # (n_conditions, n_time, n_dims)
    n_test, n_time, _ = test_state.shape
    da = np.full((n_test, n_time), np.nan)
    if len(condition_labels) < 2:
        return da
    for i in range(n_test):
        own = np.flatnonzero(condition_labels == test_labels[i])
        if len(own) == 0:
            continue
        own_index = own[0]
        distances = np.linalg.norm(test_state[i][None, :, :] - centroids, axis=2)  # (n_conditions, n_time)
        other_mask = np.ones(len(condition_labels), dtype=bool)
        other_mask[own_index] = False
        other_mean = distances[other_mask].mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            da[i] = np.where(other_mean > 1e-12, distances[own_index] / other_mean, np.nan)
    return da


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


def coding_direction_stability(
    Z: NDArray,
    labels: NDArray,
    step: int = 40,
) -> tuple[NDArray, NDArray]:
    """Coding direction stability: cosine similarity of SVM weight vectors across time.

    Trains a logistic regression at each subsampled timepoint and extracts
    the normalised coefficient vector w(t) ∈ R^k. Returns the absolute cosine
    similarity matrix C[i,j] = |w(tᵢ)·w(tⱼ)|.

    Distinct from CTG AUC: CTG asks "can the classifier generalise?";
    this asks "does it use the same geometric direction?"
    - C ≈ 1 everywhere → same axis is predictive at all times → fixed-point attractor
    - C low off-diagonal → code rotates → ring-like or oscillatory dynamics
    (Murray et al. 2017, PNAS; Stokes et al. 2013, Neuron — extended here to
    human iEEG HGP for the first time)

    For >2 classes (e.g. item identity), a one-vs-rest weight vector w_c(t) is
    fit per class and C[i,j] is the mean over classes of |w_c(tᵢ)·w_c(tⱼ)|,
    the natural multiclass generalisation of the same axis-alignment question
    (used for the content-axis rotation analysis).

    Parameters
    ----------
    Z      : (N, T, k) — latent trajectories
    labels : (N,)      — class labels (binary or multiclass)
    step   : stride for subsampling timepoints

    Returns
    -------
    cos_sim : (n_t, n_t) — absolute cosine similarity matrix
    t_idx   : (n_t,)    — subsampled time indices
    """
    T = Z.shape[1]
    t_idx = np.arange(0, T, step)
    n_t = len(t_idx)
    multiclass = len(np.unique(labels)) > 2
    weights = _fit_axis_weights(Z, labels, t_idx, multiclass)

    if multiclass:
        cos_sim = np.zeros((n_t, n_t))
        for i in range(n_t):
            for j in range(n_t):
                cos_sim[i, j] = np.mean(np.abs(np.sum(weights[i] * weights[j], axis=1)))
    else:
        W = np.concatenate(weights, axis=0)  # (n_t, k)
        cos_sim = np.abs(W @ W.T)

    return cos_sim, t_idx


def _fit_axis_weights(Z: NDArray, labels: NDArray, t_idx: NDArray, multiclass: bool) -> list[NDArray]:
    """Fit a one-vs-rest logistic-regression decoding axis at each of t_idx.

    Neither coding_direction_stability nor axis_angular_velocity holds out a
    separate train/test split of trials — the full trial set given here IS
    the (only) fold — so the scaler is fit ONCE, pooled across every
    requested timepoint in t_idx, and reused unchanged at every timepoint,
    rather than refit per timepoint. Refitting a StandardScaler
    independently at each timepoint changes the standardised feature
    covariance at each t, and because L2 shrinkage is anisotropic under
    correlated features, that alone changes the shrunk weight DIRECTION even
    when the true discriminant axis is perfectly stationary — a spurious
    source of apparent axis rotation. C is chosen per timepoint by
    LogisticRegressionCV's internal cross-validation (using this same frozen
    scaling) instead of being fixed at 1.0.

    Returns a list of unit-normalised weight arrays, (1, k) if binary or
    (n_classes, k) if multiclass, one per timepoint in t_idx.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    k = Z.shape[2]
    scaler = StandardScaler().fit(Z[:, t_idx, :].reshape(-1, k))
    Cs = np.logspace(-2, 2, 5)

    weights = []
    for ti in t_idx:
        X = scaler.transform(Z[:, ti, :])
        clf = LogisticRegressionCV(
            Cs=Cs, cv=3, max_iter=1000 if multiclass else 300,
            solver="lbfgs" if multiclass else "liblinear",
        )
        clf.fit(X, labels)
        w = clf.coef_  # (1, k) binary, (n_classes, k) multiclass
        norms = np.linalg.norm(w, axis=1, keepdims=True)
        weights.append(np.divide(w, norms, out=np.zeros_like(w), where=norms > 1e-12))
    return weights


def axis_angular_velocity(
    Z: NDArray,
    labels: NDArray,
    dt: float,
    step: int = 40,
) -> tuple[NDArray, NDArray]:
    """Angular velocity of the decoding axis, omega_axis(t) [rad/s].

    omega_axis(t) = arccos(|<w(t), w(t+step)>|) / (step * dt) -- a direct,
    DMD-free measure of how fast the coding direction itself rotates,
    reusing the same per-timepoint decoding axis w(t) as
    coding_direction_stability. Requires neither a linear operator nor an
    identifiable eigenvector.

    Parameters
    ----------
    Z      : (N, T, k) — latent trajectories
    labels : (N,)      — class labels (binary or multiclass)
    dt     : sampling interval (s) of Z's time axis
    step   : stride for subsampling timepoints (same convention as
             coding_direction_stability)

    Returns
    -------
    omega : (n_t - 1,) — angular velocity between consecutive subsampled
            timepoints, rad/s
    t_idx : (n_t - 1,) — the earlier timepoint index of each pair
    """
    T = Z.shape[1]
    t_idx = np.arange(0, T, step)
    if len(t_idx) < 2:
        return np.array([]), np.array([], dtype=int)
    multiclass = len(np.unique(labels)) > 2
    weights = _fit_axis_weights(Z, labels, t_idx, multiclass)
    dt_step = float(np.median(np.diff(t_idx))) * dt

    omega = np.empty(len(weights) - 1)
    for i in range(len(weights) - 1):
        if multiclass:
            cos = np.mean(np.abs(np.sum(weights[i] * weights[i + 1], axis=1)))
        else:
            cos = np.abs(float((weights[i] @ weights[i + 1].T).item()))
        omega[i] = np.arccos(np.clip(cos, -1.0, 1.0)) / dt_step
    return omega, t_idx[:-1]


def time_resolved_stability(
    Z: NDArray,
    labels: NDArray,
    step: int = 40,
) -> tuple[NDArray, NDArray]:
    """Time-resolved temporal stability τ(t_train).

    For each training timepoint tᵢ, compute the mean off-diagonal AUC
    (generalisation to all other timepoints) divided by the diagonal AUC
    (decoding at tᵢ itself). This shows WHEN during the trial the code
    becomes time-invariant.

    τ(t) ≈ 1 → code at time t generalises perfectly to all other times.
    τ(t) < 1 → code at time t is partially time-specific.

    Parameters
    ----------
    Z      : (N, T, k) — latent trajectories
    labels : (N,)      — binary labels
    step   : timepoint stride

    Returns
    -------
    tau_t : (n_t,) — stability index per training timepoint
    t_idx : (n_t,) — subsampled time indices
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    T = Z.shape[1]
    t_idx = np.arange(0, T, step)
    n_t = len(t_idx)

    def _auc(scores, y):
        pos = scores[y == 1]
        neg = scores[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        u = float(np.sum(pos[:, None] > neg[None, :]))
        return u / (len(pos) * len(neg))

    tau_t = np.full(n_t, np.nan)

    for i, ti in enumerate(t_idx):
        X_tr = Z[:, ti, :]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
        clf.fit(X_tr_s, labels)

        diag_auc = _auc(clf.decision_function(X_tr_s), labels)
        offdiag_aucs = []
        for j, tj in enumerate(t_idx):
            if i == j:
                continue
            X_te_s = sc.transform(Z[:, tj, :])
            offdiag_aucs.append(_auc(clf.decision_function(X_te_s), labels))

        mean_off = float(np.nanmean(offdiag_aucs))
        tau_t[i] = mean_off / diag_auc if diag_auc > 1e-6 else np.nan

    return tau_t, t_idx


def ctg_temporal_distance_control(
    Z: NDArray,
    labels: NDArray,
    step: int = 40,
) -> tuple[NDArray, NDArray]:
    """AUC as a function of temporal lag — control for autocorrelation confound.

    The core concern: 50 ms Gaussian smoothing creates autocorrelation, so a
    decoder trained at t may generalise to t+Δ simply because Z(t) ≈ Z(t+Δ)
    due to sluggish dynamics, not because of a stable memory attractor.

    This function computes mean AUC at each temporal lag Δ = |t_train − t_test|.
    Under an autocorrelation confound: AUC(Δ) should decay like the autocorrelation
    function — dropping to chance within ~100–200 ms.
    Under a genuine stable attractor: AUC(Δ) remains elevated for all Δ throughout
    the maintenance window.

    Parameters
    ----------
    Z      : (N, T, k) — latent trajectories (0-back and 2-back trials only)
    labels : (N,)      — binary labels (0=0-back, 1=2-back)
    step   : timepoint stride

    Returns
    -------
    lag_samples : (n_lags,) — |t_i − t_j| in samples
    auc_by_lag  : (n_lags,) — mean AUC for that lag
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    T = Z.shape[1]
    t_idx = np.arange(0, T, step)
    n_t = len(t_idx)

    def _auc(scores, y):
        pos, neg = scores[y == 1], scores[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        u = float(np.sum(pos[:, None] > neg[None, :]))
        return u / (len(pos) * len(neg))

    # Full CTG matrix
    auc_mat = np.full((n_t, n_t), np.nan)
    for i, ti in enumerate(t_idx):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(Z[:, ti, :])
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
        clf.fit(X_tr_s, labels)
        for j, tj in enumerate(t_idx):
            X_te_s = sc.transform(Z[:, tj, :])
            auc_mat[i, j] = _auc(clf.decision_function(X_te_s), labels)

    # Bin by lag
    lags = np.abs(t_idx[:, None] - t_idx[None, :])   # (n_t, n_t) in samples
    unique_lags = np.unique(lags)
    auc_by_lag = np.array([np.nanmean(auc_mat[lags == lag]) for lag in unique_lags])
    return unique_lags, auc_by_lag


def spatiotemporal_participation_ratio(
    X: NDArray,
    n_splits: int = 2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Participation ratio in the native (pre-PCA) channel space, cross-validated
    and reported alongside its legacy predecessor.

    Computed identically regardless of dataset: for each trial, treat the
    T time samples during the window as observations of a C-dimensional
    channel/unit vector, pool across trials, and compute PR of the resulting
    C×C covariance spectrum — the same method used for Miller's full-channel
    per-trial spatiotemporal PCA. Using a fixed method (rather than PR of an
    already-PCA-reduced, capped-at-k latent space) makes PR comparable across
    datasets with very different channel/unit counts.

    pr_cv (primary) is a held-out, in-fold-fit / held-out-eval estimator, the
    finite-sample-corrected analogue of cvPCA (Stringer et al. 2019): the
    covariance eigenvectors are fit on a TRAIN split of trials only, and PR is
    computed from the variance those fixed, independently-derived directions
    explain in the disjoint held-out trial split. A single split's own
    eigenvalues are always biased by finite-sample noise (and can be trivially
    rank-deficient whenever a fold has fewer samples than channels), which
    makes data look more low-dimensional than it truly is; scoring a
    train-fit basis on an independent held-out split removes that bias.

    pr_cv_legacy is the original "cross-validated" estimator this replaces as
    primary: PR of the held-out split's OWN covariance, with no train-fit
    basis at all (so it is not actually evaluating a learned estimator
    out-of-fold — it is just PR of a single, half-sized data split). Reported
    only for continuity with prior results.

    Both pr_cv and pr_cv_legacy split at the TRIAL level (a trial's T time
    samples are never divided across the train/test folds), since time
    samples within a trial are autocorrelated, not independent draws.

    A shuffle null independently permutes each channel's trial order —
    destroying cross-channel covariance while preserving each channel's own
    marginal variance and within-trial autocorrelation — to give the PR
    expected from noise alone at the same dimensionality and trial count. (A
    plain channel-identity permutation would NOT do this: PR is basis
    invariant under relabelling channels, so it leaves the covariance
    spectrum, and hence PR, completely unchanged.)

    Parameters
    ----------
    X        : (N, C, T) — raw per-trial channel/unit responses (not yet PCA'd)
    n_splits : train/test folds (results averaged over folds)
    rng      : random number generator

    Returns
    -------
    dict: pr_insample, pr_cv, pr_cv_std, pr_cv_legacy, pr_cv_legacy_std,
          pr_null, pr_null_std, n_channels, n_trials
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N, C, T = X.shape
    flat_all = X.transpose(0, 2, 1).reshape(-1, C)
    flat_all = flat_all - flat_all.mean(0)
    _, s_all, _ = np.linalg.svd(flat_all, full_matrices=False)
    pr_insample = participation_ratio(s_all**2)

    idx = rng.permutation(N)
    folds = np.array_split(idx, min(n_splits, N))
    pr_cv_list, pr_cv_legacy_list = [], []
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        if len(tr) < 2 or len(te) < 2:
            continue
        flat_tr = X[tr].transpose(0, 2, 1).reshape(-1, C)
        mu_tr = flat_tr.mean(0)
        _, _, Vt_tr = np.linalg.svd(flat_tr - mu_tr, full_matrices=False)

        flat_te = X[te].transpose(0, 2, 1).reshape(-1, C) - mu_tr
        lam_heldout = ((flat_te @ Vt_tr.T) ** 2).sum(axis=0)
        pr_cv_list.append(participation_ratio(lam_heldout))

        _, s_te, _ = np.linalg.svd(flat_te, full_matrices=False)
        pr_cv_legacy_list.append(participation_ratio(s_te**2))
    pr_cv = float(np.mean(pr_cv_list)) if pr_cv_list else float("nan")
    pr_cv_std = float(np.std(pr_cv_list)) if pr_cv_list else float("nan")
    pr_cv_legacy = float(np.mean(pr_cv_legacy_list)) if pr_cv_legacy_list else float("nan")
    pr_cv_legacy_std = float(np.std(pr_cv_legacy_list)) if pr_cv_legacy_list else float("nan")

    n_null = 50
    pr_null_list = []
    for _ in range(n_null):
        X_null = np.stack(
            [X[rng.permutation(N), c, :] for c in range(C)], axis=1
        )
        flat_null = X_null.transpose(0, 2, 1).reshape(-1, C)
        flat_null = flat_null - flat_null.mean(0)
        _, s_null, _ = np.linalg.svd(flat_null, full_matrices=False)
        pr_null_list.append(participation_ratio(s_null**2))
    pr_null = float(np.mean(pr_null_list))
    pr_null_std = float(np.std(pr_null_list))

    return {
        "pr_insample": float(pr_insample),
        "pr_cv": pr_cv,
        "pr_cv_std": pr_cv_std,
        "pr_cv_legacy": pr_cv_legacy,
        "pr_cv_legacy_std": pr_cv_legacy_std,
        "pr_null": pr_null,
        "pr_null_std": pr_null_std,
        "n_channels": int(C),
        "n_trials": int(N),
    }


def _fit_pca_fold(X_tr: NDArray, n_components: int) -> tuple[NDArray, NDArray]:
    """PCA basis (mean, loadings) fit on train-fold trials only. X_tr: (n, C, T)."""
    n, C, T = X_tr.shape
    flat = X_tr.transpose(0, 2, 1).reshape(-1, C)
    mu = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    V = Vt[:n_components].T  # (C, n_components)
    return mu, V


def _project_fold(X: NDArray, mu: NDArray, V: NDArray) -> NDArray:
    """Project (n, C, T) raw data into (n, T, n_components) latent space."""
    n, C, T = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, C) - mu
    Z = (flat @ V).reshape(n, T, V.shape[1])
    return Z


def _ctg_score_fold(
    Z_tr: NDArray, y_tr: NDArray, Z_te: NDArray, y_te: NDArray, t_idx: NDArray,
) -> NDArray:
    """AUC matrix for one fold given already-PCA-projected train/test latents."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    n_t = len(t_idx)
    mat = np.full((n_t, n_t), np.nan)
    if len(np.unique(y_tr)) < 2:
        return mat

    for i, ti in enumerate(t_idx):
        sc = StandardScaler()
        X_tr_t = sc.fit_transform(Z_tr[:, ti, :])
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=300)
        clf.fit(X_tr_t, y_tr)
        for j, tj in enumerate(t_idx):
            X_te_t = sc.transform(Z_te[:, tj, :])
            scores = clf.decision_function(X_te_t)
            if len(np.unique(y_te)) != 2:
                continue
            # roc_auc_score implements the Mann--Whitney convention in which
            # tied scores contribute one half, unlike the legacy strict-">"
            # implementation that biased AUC downward whenever scores tied.
            mat[i, j] = float(roc_auc_score(y_te, scores))
    return mat


def _ctg_splits(
    labels: NDArray,
    n_splits: int,
    rng: np.random.Generator,
    groups: NDArray | None,
) -> list[tuple[NDArray, NDArray]]:
    """Build outer folds without allowing a linked recording across folds."""
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    if groups is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        return list(splitter.split(np.zeros(len(labels)), labels))

    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.asarray(groups)
    if len(groups) != len(labels):
        raise ValueError("groups must have one entry per trial")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("linked recording group crossed a CTG outer fold")
    return folds


def _permute_labels(
    labels: NDArray,
    rng: np.random.Generator,
    exchangeability_blocks: NDArray | None,
) -> NDArray:
    """Permute labels at the declared independent randomization unit.

    A block is permitted only when its trials have one treatment label.  This
    avoids the common but invalid shortcut of independently shuffling trials
    within a participant/session when treatment varied at a higher level.
    """
    labels = np.asarray(labels)
    if exchangeability_blocks is None:
        return rng.permutation(labels)
    blocks = np.asarray(exchangeability_blocks)
    if len(blocks) != len(labels):
        raise ValueError("exchangeability_blocks must have one entry per trial")
    unique, inverse = np.unique(blocks, return_inverse=True)
    block_labels = np.empty(len(unique), dtype=labels.dtype)
    for i in range(len(unique)):
        values = np.unique(labels[inverse == i])
        if len(values) != 1:
            raise ValueError(
                "block permutation requires one label per exchangeability block; "
                "use trial-level blocks only when trials are independently randomized"
            )
        block_labels[i] = values[0]
    return rng.permutation(block_labels)[inverse]


def ctg_nested_cv(
    X: NDArray,
    labels: NDArray,
    t_idx: NDArray,
    n_components: int = 8,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
    return_fold_data: bool = False,
    groups: NDArray | None = None,
):
    """Cross-temporal generalization with PCA folded into cross-validation.

    Fitting PCA once on all trials before splitting into CV folds (the
    original pipeline) lets test-trial information leak into the decoding
    basis. Here, for each fold, PCA is fit on train-fold trials only and
    test-fold trials are projected through that train-fitted basis, so no
    information about held-out trials contributes to the representation the
    classifier is trained in.

    Parameters
    ----------
    X                : (N, C, T) — raw per-trial channel/unit responses
    labels           : (N,) binary labels
    t_idx            : timepoints (into T) to include in the CTG matrix
    n_components     : PCA dimensionality fit per fold
    n_splits         : k-fold CV
    return_fold_data : if True, also return per-fold (Z_tr, y_tr, Z_te, y_te,
                        te_idx) so a label-permutation null can reuse the same
                        (label-blind) PCA projections without refitting PCA.

    Returns
    -------
    auc_mat : (n_t, n_t) — mean AUC across folds
    fold_data (only if return_fold_data) : list of tuples
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = X.shape[0]
    labels = np.asarray(labels)
    fold_mats, fold_data = [], []
    for tr_idx, te_idx in _ctg_splits(labels, n_splits, rng, groups):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr = _project_fold(X[tr_idx], mu, V)
        Z_te = _project_fold(X[te_idx], mu, V)
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        mat = _ctg_score_fold(Z_tr, y_tr, Z_te, y_te, t_idx)
        fold_mats.append(mat)
        if return_fold_data:
            fold_data.append((Z_tr, y_tr, Z_te, y_te))

    auc_mat = np.nanmean(np.stack(fold_mats), axis=0)
    if return_fold_data:
        return auc_mat, fold_data
    return auc_mat


def ctg_label_permutation_null(
    X: NDArray,
    labels: NDArray,
    t_idx: NDArray,
    n_components: int = 8,
    n_splits: int = 5,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
    groups: NDArray | None = None,
    exchangeability_blocks: NDArray | None = None,
) -> dict:
    """Valid significance test for CTG: shuffle condition labels, not AUC cells.

    Off-diagonal CTG cells are massively non-independent (same trials,
    adjacent timepoints, overlapping classifier fits), so treating them as
    independent samples for a one-sample t-test (or resampling them directly
    in a bootstrap) manufactures artificially small p-values / narrow CIs.
    This instead permutes the condition LABELS within each fold, recomputes
    the entire CTG matrix from the resulting scrambled decoders, and reduces
    each permutation to the same single summary statistic as the observed
    data — giving a null distribution over the whole-matrix statistic rather
    than over individual (non-independent) cells.

    PCA is unsupervised (never sees labels), so it does not need to be
    refit under permutation — only the per-fold classifiers do, which keeps
    this tractable at n_perm ~ 100-500.

    Parameters
    ----------
    X, labels, t_idx, n_components, n_splits : as in ctg_nested_cv
    n_perm : number of label permutations

    Returns
    -------
    dict:
      auc_mat        : (n_t, n_t) — observed CTG matrix
      mean_offdiag_auc_minus_chance : float — observed effect size (AUC-0.5)
      mean_diag_auc_minus_chance    : float
      tau            : mean_offdiag / mean_diag effect size ratio (see
                        temporal_stability_tau for the gated/interpretable version)
      null           : (n_perm,) — null distribution of the offdiag effect size
      p_value        : float — fraction of null >= observed
    """
    if rng is None:
        rng = np.random.default_rng(0)

    auc_obs = ctg_nested_cv(X, labels, t_idx, n_components, n_splits, rng, groups=groups)

    def _stat(mat: NDArray) -> tuple[float, float]:
        n_t = mat.shape[0]
        off = mat[~np.eye(n_t, dtype=bool)] - 0.5
        dia = np.diag(mat) - 0.5
        return float(np.nanmean(off)), float(np.nanmean(dia))

    off_obs, dia_obs = _stat(auc_obs)

    null = np.zeros(n_perm)
    for p in range(n_perm):
        labels_p = _permute_labels(labels, rng, exchangeability_blocks)
        # Rebuild the complete nested analysis, including split assignment and
        # train-fold PCA/scaling, for every null draw.  Reusing observed-fold
        # projections is only valid for a strictly label-blind split, which
        # stratification is not.
        mat_p = ctg_nested_cv(X, labels_p, t_idx, n_components, n_splits, rng, groups=groups)
        null[p], _ = _stat(mat_p)

    p_value = permutation_pvalue(null >= off_obs)

    return {
        "auc_mat": auc_obs,
        "mean_offdiag_auc_minus_chance": off_obs,
        "mean_diag_auc_minus_chance": dia_obs,
        "tau": float(off_obs / dia_obs) if abs(dia_obs) > 1e-6 else float("nan"),
        "null": null,
        "p_value": p_value,
    }


def _ctg_score_fold_multiclass(
    Z_tr: NDArray, y_tr: NDArray, Z_te: NDArray, y_te: NDArray, t_idx: NDArray,
    all_classes: NDArray,
) -> NDArray:
    """Macro one-vs-rest AUC matrix for one fold (multiclass content decoding)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    n_t = len(t_idx)
    mat = np.full((n_t, n_t), np.nan)
    if len(np.unique(y_tr)) < 2:
        return mat

    for i, ti in enumerate(t_idx):
        sc = StandardScaler()
        X_tr_t = sc.fit_transform(Z_tr[:, ti, :])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(X_tr_t, y_tr)
        for j, tj in enumerate(t_idx):
            X_te_t = sc.transform(Z_te[:, tj, :])
            if len(np.unique(y_te)) < 2:
                continue
            proba_cols = {c: k for k, c in enumerate(clf.classes_)}
            proba = clf.predict_proba(X_te_t)
            present = np.unique(y_te)
            if len(present) < 2:
                continue
            aucs = []
            for c in present:
                if c not in proba_cols:
                    continue
                y_bin = (y_te == c).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                try:
                    aucs.append(roc_auc_score(y_bin, proba[:, proba_cols[c]]))
                except ValueError:
                    continue
            if aucs:
                mat[i, j] = float(np.mean(aucs))
    return mat


def ctg_content_permutation_null(
    X: NDArray,
    labels: NDArray,
    t_idx: NDArray,
    n_components: int = 8,
    n_splits: int = 3,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Multiclass item/content CTG (macro one-vs-rest AUC) with a label-shuffle null.

    Same nested-CV (PCA fit per fold, never sees labels) and label-permutation
    logic as ctg_label_permutation_null, generalised to >2 classes — used to
    decode item identity (not load/condition) within a fixed load, which is
    the analysis that actually speaks to activity-maintained vs silent coding
    of the memorandum itself.

    Returns
    -------
    dict: auc_mat, mean_offdiag_auc_minus_chance, mean_diag_auc_minus_chance,
          tau, null, p_value  (chance for macro OvR AUC is 0.5 regardless of
          n_classes)
    """
    from sklearn.model_selection import StratifiedKFold

    if rng is None:
        rng = np.random.default_rng(0)

    N = X.shape[0]
    labels = np.asarray(labels)
    all_classes = np.unique(labels)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 1_000_000))
    )

    fold_mats, fold_data = [], []
    for tr_idx, te_idx in skf.split(np.zeros(N), labels):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr = _project_fold(X[tr_idx], mu, V)
        Z_te = _project_fold(X[te_idx], mu, V)
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        mat = _ctg_score_fold_multiclass(Z_tr, y_tr, Z_te, y_te, t_idx, all_classes)
        fold_mats.append(mat)
        fold_data.append((Z_tr, y_tr, Z_te, y_te))

    auc_obs = np.nanmean(np.stack(fold_mats), axis=0)

    def _stat(mat: NDArray) -> tuple[float, float]:
        n_t = mat.shape[0]
        off = mat[~np.eye(n_t, dtype=bool)] - 0.5
        dia = np.diag(mat) - 0.5
        return float(np.nanmean(off)), float(np.nanmean(dia))

    off_obs, dia_obs = _stat(auc_obs)

    null = np.zeros(n_perm)
    for p in range(n_perm):
        fold_mats_p = []
        for Z_tr, y_tr, Z_te, y_te in fold_data:
            y_tr_p = rng.permutation(y_tr)
            y_te_p = rng.permutation(y_te)
            fold_mats_p.append(
                _ctg_score_fold_multiclass(Z_tr, y_tr_p, Z_te, y_te_p, t_idx, all_classes)
            )
        mat_p = np.nanmean(np.stack(fold_mats_p), axis=0)
        null[p], _ = _stat(mat_p)

    p_value = permutation_pvalue(null >= off_obs)

    return {
        "auc_mat": auc_obs,
        "mean_offdiag_auc_minus_chance": off_obs,
        "mean_diag_auc_minus_chance": dia_obs,
        "tau": float(off_obs / dia_obs) if abs(dia_obs) > 1e-6 else float("nan"),
        "null": null,
        "p_value": p_value,
        "n_classes": int(len(all_classes)),
    }


def time_resolved_content_decoding(
    X: NDArray,
    labels: NDArray,
    t_idx: NDArray,
    n_components: int = 8,
    n_splits: int = 3,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-timepoint (not cross-temporal) multiclass decoding accuracy, with a
    label-permutation null on each timepoint's macro one-vs-rest AUC.

    Trains and tests at the same timepoint only, at O(T) cost rather than the
    O(T^2) cost of ctg_content_permutation_null's full cross-temporal matrix,
    for use over long time windows with many timepoints (e.g. a full trial
    from baseline through response) where computing off-diagonal
    generalisation is not needed.

    Returns
    -------
    dict: auc_per_t (n_t,), p_per_t (n_t,), t_idx, n_classes
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng(0)

    N = X.shape[0]
    labels = np.asarray(labels)
    all_classes = np.unique(labels)
    n_t = len(t_idx)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 1_000_000))
    )

    fold_data = []
    for tr_idx, te_idx in skf.split(np.zeros(N), labels):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr, Z_te = _project_fold(X[tr_idx], mu, V), _project_fold(X[te_idx], mu, V)
        fold_data.append((Z_tr, labels[tr_idx], Z_te, labels[te_idx]))

    def _auc_at(ti: int, data: list) -> float:
        aucs = []
        for Z_tr, y_tr, Z_te, y_te in data:
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            sc = StandardScaler()
            X_tr_t = sc.fit_transform(Z_tr[:, ti, :])
            X_te_t = sc.transform(Z_te[:, ti, :])
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            clf.fit(X_tr_t, y_tr)
            proba_cols = {c: k for k, c in enumerate(clf.classes_)}
            proba = clf.predict_proba(X_te_t)
            class_aucs = []
            for c in np.unique(y_te):
                if c not in proba_cols:
                    continue
                y_bin = (y_te == c).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                class_aucs.append(roc_auc_score(y_bin, proba[:, proba_cols[c]]))
            if class_aucs:
                aucs.append(float(np.mean(class_aucs)))
        return float(np.mean(aucs)) if aucs else float("nan")

    auc_obs = np.array([_auc_at(i, fold_data) for i in range(n_t)])

    null = np.zeros((n_perm, n_t))
    for p in range(n_perm):
        fold_data_p = [(Z_tr, rng.permutation(y_tr), Z_te, rng.permutation(y_te))
                       for Z_tr, y_tr, Z_te, y_te in fold_data]
        null[p] = np.array([_auc_at(i, fold_data_p) for i in range(n_t)])

    p_per_t = (null >= auc_obs[None, :]).mean(axis=0)

    return {"auc_per_t": auc_obs, "p_per_t": p_per_t, "t_idx": t_idx,
            "n_classes": int(len(all_classes))}


def content_decoding_single_latent(
    X: NDArray,
    labels: NDArray,
    t_idx: NDArray,
    n_components: int,
    latent_index: int,
    n_splits: int = 3,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """The complement of :func:`content_decoding_dropping_latent`: fits the
    same ``n_components``-dimensional per-fold PCA basis but hands the
    classifier ONLY column ``latent_index`` (0-based, ordered by descending
    training-fold variance) rather than all but one. Used to profile which
    individual latent -- not merely whether removing the leading one costs
    anything -- carries a population's item content, at a fixed total k so
    every latent's index means the same thing across the profile.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng(0)

    N = X.shape[0]
    labels = np.asarray(labels)
    all_classes = np.unique(labels)
    n_t = len(t_idx)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))

    fold_data = []
    for tr_idx, te_idx in skf.split(np.zeros(N), labels):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr = _project_fold(X[tr_idx], mu, V)[:, :, [latent_index]]
        Z_te = _project_fold(X[te_idx], mu, V)[:, :, [latent_index]]
        fold_data.append((Z_tr, labels[tr_idx], Z_te, labels[te_idx]))

    def _auc_at(ti: int, data: list) -> float:
        aucs = []
        for Z_tr, y_tr, Z_te, y_te in data:
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            sc = StandardScaler()
            X_tr_t = sc.fit_transform(Z_tr[:, ti, :])
            X_te_t = sc.transform(Z_te[:, ti, :])
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            clf.fit(X_tr_t, y_tr)
            proba_cols = {c: k for k, c in enumerate(clf.classes_)}
            proba = clf.predict_proba(X_te_t)
            class_aucs = []
            for c in np.unique(y_te):
                if c not in proba_cols:
                    continue
                y_bin = (y_te == c).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                class_aucs.append(roc_auc_score(y_bin, proba[:, proba_cols[c]]))
            if class_aucs:
                aucs.append(float(np.mean(class_aucs)))
        return float(np.mean(aucs)) if aucs else float("nan")

    auc_obs = np.array([_auc_at(i, fold_data) for i in range(n_t)])

    null = np.zeros((n_perm, n_t))
    for p in range(n_perm):
        fold_data_p = [(Z_tr, rng.permutation(y_tr), Z_te, rng.permutation(y_te))
                       for Z_tr, y_tr, Z_te, y_te in fold_data]
        null[p] = np.array([_auc_at(i, fold_data_p) for i in range(n_t)])

    p_per_t = (null >= auc_obs[None, :]).mean(axis=0)

    return {"auc_per_t": auc_obs, "p_per_t": p_per_t, "t_idx": t_idx, "n_classes": int(len(all_classes)),
            "n_components": int(n_components), "latent_index": int(latent_index)}


def content_decoding_dropping_latent(
    X: NDArray,
    labels: NDArray,
    t_idx: NDArray,
    n_components: int,
    drop_component_index: int | None = None,
    n_splits: int = 3,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Identical to :func:`time_resolved_content_decoding` -- same per-fold
    PCA fit (:func:`_fit_pca_fold`), the same held-out projection
    (:func:`_project_fold`), the same classifier, cross-validation, and
    label-permutation null -- with one PCA column excluded from the feature
    set handed to the classifier. ``drop_component_index=None`` reproduces
    the full k-latent decoding exactly; an integer index (0-based, ordered
    by descending training-fold variance, so 0 is the leading latent) drops
    that column, leaving k-1 features. This is the one subtraction the
    content-link question needs and no existing function exposes: whether
    removing the state's own leading latent costs content decoding more
    than removing a comparable one.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng(0)

    N = X.shape[0]
    labels = np.asarray(labels)
    all_classes = np.unique(labels)
    n_t = len(t_idx)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 1_000_000)))
    keep_cols = [c for c in range(n_components) if c != drop_component_index]

    fold_data = []
    for tr_idx, te_idx in skf.split(np.zeros(N), labels):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr = _project_fold(X[tr_idx], mu, V)[:, :, keep_cols]
        Z_te = _project_fold(X[te_idx], mu, V)[:, :, keep_cols]
        fold_data.append((Z_tr, labels[tr_idx], Z_te, labels[te_idx]))

    def _auc_at(ti: int, data: list) -> float:
        aucs = []
        for Z_tr, y_tr, Z_te, y_te in data:
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            sc = StandardScaler()
            X_tr_t = sc.fit_transform(Z_tr[:, ti, :])
            X_te_t = sc.transform(Z_te[:, ti, :])
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            clf.fit(X_tr_t, y_tr)
            proba_cols = {c: k for k, c in enumerate(clf.classes_)}
            proba = clf.predict_proba(X_te_t)
            class_aucs = []
            for c in np.unique(y_te):
                if c not in proba_cols:
                    continue
                y_bin = (y_te == c).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                class_aucs.append(roc_auc_score(y_bin, proba[:, proba_cols[c]]))
            if class_aucs:
                aucs.append(float(np.mean(class_aucs)))
        return float(np.mean(aucs)) if aucs else float("nan")

    auc_obs = np.array([_auc_at(i, fold_data) for i in range(n_t)])

    null = np.zeros((n_perm, n_t))
    for p in range(n_perm):
        fold_data_p = [(Z_tr, rng.permutation(y_tr), Z_te, rng.permutation(y_te))
                       for Z_tr, y_tr, Z_te, y_te in fold_data]
        null[p] = np.array([_auc_at(i, fold_data_p) for i in range(n_t)])

    p_per_t = (null >= auc_obs[None, :]).mean(axis=0)

    return {"auc_per_t": auc_obs, "p_per_t": p_per_t, "t_idx": t_idx, "n_classes": int(len(all_classes)),
            "n_components": int(n_components), "drop_component_index": drop_component_index,
            "kept_components": keep_cols}


def out_of_fold_class_confidence(
    X: NDArray,
    labels: NDArray,
    t_idx: int | NDArray,
    n_components: int = 8,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Per-trial decoder confidence in its own true class label, from a
    classifier that never saw that trial during fitting.

    Fits a (binary or multiclass) logistic regression per cross-validation
    fold (PCA folded in, as elsewhere in this module) at each requested
    timepoint and, for each trial, records the predicted probability
    assigned to that trial's own true class when it was in the held-out
    fold. Used to relate single-trial decoding confidence to a per-trial
    outcome variable (e.g. behavioral accuracy) over time, which the
    fold-averaged AUC statistics elsewhere in this module cannot address
    since they discard trial identity.

    Parameters
    ----------
    t_idx : a single timepoint index, or an array of them (confidence is
        computed at each, reusing the same cross-validation folds and
        per-fold PCA fit across all requested timepoints)

    Returns
    -------
    confidence : (N,) if t_idx is a scalar, else (N, n_t) — predicted
        probability of each trial's own true class at each timepoint
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.default_rng(0)
    scalar_input = np.isscalar(t_idx)
    t_idx_arr = np.atleast_1d(np.asarray(t_idx))
    N = X.shape[0]
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=int(rng.integers(0, 1_000_000)))

    confidence = np.full((N, len(t_idx_arr)), np.nan)
    for tr_idx, te_idx in skf.split(np.zeros(N), labels):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr, Z_te = _project_fold(X[tr_idx], mu, V), _project_fold(X[te_idx], mu, V)
        for j, ti in enumerate(t_idx_arr):
            sc = StandardScaler()
            X_tr_t = sc.fit_transform(Z_tr[:, ti, :])
            X_te_t = sc.transform(Z_te[:, ti, :])
            clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            clf.fit(X_tr_t, labels[tr_idx])
            proba = clf.predict_proba(X_te_t)
            proba_cols = {c: k for k, c in enumerate(clf.classes_)}
            for i, te_i in enumerate(te_idx):
                true_class = labels[te_i]
                if true_class in proba_cols:
                    confidence[te_i, j] = proba[i, proba_cols[true_class]]

    return confidence[:, 0] if scalar_input else confidence


def cross_decoding_leakage_test(
    X: NDArray,
    labels_train: NDArray,
    labels_test: NDArray,
    t_idx: int,
    n_components: int = 8,
    n_splits: int = 5,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Fits a (binary or multiclass) decoder for labels_train at one
    timepoint, with PCA folded into cross-validation, and tests via a
    permuted F-statistic whether its held-out decision-function projection
    (reduced to one dimension via its leading principal component when
    labels_train has more than two classes) separates labels_test — i.e.
    whether the labels_train decoding axis carries information about a
    different variable.

    Parameters
    ----------
    X : (N, C, T) — raw or already-reduced per-trial features
    labels_train : (N,) — variable the decoder is trained on
    labels_test : (N,) — different variable tested for leakage into the
        labels_train decoding axis
    t_idx : single timepoint index into the T axis

    Returns
    -------
    dict: f_statistic, p_value, n_trials
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.default_rng(0)
    N = X.shape[0]
    labels_train = np.asarray(labels_train)
    labels_test = np.asarray(labels_test)
    n_classes_train = len(np.unique(labels_train))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=int(rng.integers(0, 1_000_000)))

    proj = np.full((N, n_classes_train if n_classes_train > 2 else 1), np.nan)
    for tr_idx, te_idx in skf.split(np.zeros(N), labels_train):
        mu, V = _fit_pca_fold(X[tr_idx], n_components)
        Z_tr, Z_te = _project_fold(X[tr_idx], mu, V), _project_fold(X[te_idx], mu, V)
        sc = StandardScaler()
        X_tr_t = sc.fit_transform(Z_tr[:, t_idx, :])
        X_te_t = sc.transform(Z_te[:, t_idx, :])
        clf = LogisticRegression(C=1.0, max_iter=1000,
                                 solver="liblinear" if n_classes_train == 2 else "lbfgs")
        clf.fit(X_tr_t, labels_train[tr_idx])
        proj[te_idx] = clf.decision_function(X_te_t).reshape(len(te_idx), -1)

    valid = np.all(np.isfinite(proj), axis=1)
    proj_valid = proj[valid]
    if proj_valid.shape[1] > 1:
        centered = proj_valid - proj_valid.mean(0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_1d = centered @ Vt[0]
    else:
        proj_1d = proj_valid[:, 0]
    labels_test_valid = labels_test[valid]

    def _f_stat(x: NDArray, groups: NDArray) -> float:
        group_vals = [x[groups == g] for g in np.unique(groups)]
        grand_mean = np.mean(x)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_vals)
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in group_vals)
        df_between, df_within = len(group_vals) - 1, len(x) - len(group_vals)
        return (ss_between / df_between) / (ss_within / df_within + 1e-12)

    f_obs = _f_stat(proj_1d, labels_test_valid)
    null = np.array([_f_stat(proj_1d, rng.permutation(labels_test_valid)) for _ in range(n_perm)])
    p_value = permutation_pvalue(null >= f_obs)

    return {"f_statistic": float(f_obs), "p_value": p_value, "n_trials": int(valid.sum())}


def cross_condition_decoding_test(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    t_idx: NDArray,
    n_components: int = 8,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Trains a decoder (PCA fit on X_train only, then logistic regression)
    on one condition or dataset and evaluates macro one-vs-rest AUC on a
    separate condition or dataset's trials, at each timepoint in t_idx
    (train and test at the same timepoint only), with a label-permutation
    null on y_test.

    Distinct from the within-pool cross-validated functions elsewhere in this
    module: X_train/y_train and X_test/y_test are two disjoint trial sets
    (e.g. different load levels or different sessions), so this tests
    generalisation of a decoding axis across conditions rather than
    within-condition reliability.

    Returns
    -------
    dict: auc_per_t (n_t,), p_per_t (n_t,), t_idx
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if rng is None:
        rng = np.random.default_rng(0)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    n_t = len(t_idx)
    mu, V = _fit_pca_fold(X_train, n_components)
    Z_train, Z_test = _project_fold(X_train, mu, V), _project_fold(X_test, mu, V)

    def _auc_at(ti: int, y_te: NDArray) -> float:
        sc = StandardScaler()
        X_tr_t = sc.fit_transform(Z_train[:, ti, :])
        X_te_t = sc.transform(Z_test[:, ti, :])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_tr_t, y_train)
        proba_cols = {c: k for k, c in enumerate(clf.classes_)}
        proba = clf.predict_proba(X_te_t)
        class_aucs = []
        for c in np.unique(y_te):
            if c not in proba_cols:
                continue
            y_bin = (y_te == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            class_aucs.append(roc_auc_score(y_bin, proba[:, proba_cols[c]]))
        return float(np.mean(class_aucs)) if class_aucs else float("nan")

    auc_obs = np.array([_auc_at(i, y_test) for i in range(n_t)])
    null = np.array([[_auc_at(i, rng.permutation(y_test)) for i in range(n_t)]
                     for _ in range(n_perm)])
    p_per_t = (null >= auc_obs[None, :]).mean(axis=0)

    return {"auc_per_t": auc_obs, "p_per_t": p_per_t, "t_idx": t_idx}


def temporal_stability_tau(
    auc_mat: NDArray,
    min_diag_auc: float = 0.55,
) -> dict:
    """Temporal stability index τ, defined on effect size and gated on decodability.

    τ = mean_offdiag(AUC-0.5) / mean_diag(AUC-0.5), not raw-AUC ratio.
    τ<1 is a near-mathematical necessity for any non-stationary signal (the
    diagonal upper-bounds cross-time generalisation), so τ itself is not
    evidence of stability — off-diagonal AUC-0.5 significantly > 0 is.
    τ is only reported as interpretable when the diagonal is meaningfully
    decodable (mean_diag_auc >= min_diag_auc); near-chance decoders (e.g.
    diagonal AUC ≈ 0.53) produce a ratio of two near-zero, noisy numbers that
    is unstable and should not be read as "perfectly invariant."

    Returns
    -------
    dict: tau, mean_diag_auc, mean_offdiag_auc, diag_effect, offdiag_effect,
          interpretable (bool)
    """
    T = auc_mat.shape[0]
    off_mask = ~np.eye(T, dtype=bool)
    mean_diag_auc = float(np.nanmean(np.diag(auc_mat)))
    mean_offdiag_auc = float(np.nanmean(auc_mat[off_mask]))
    diag_effect = mean_diag_auc - 0.5
    offdiag_effect = mean_offdiag_auc - 0.5
    tau = float(offdiag_effect / diag_effect) if abs(diag_effect) > 1e-6 else float("nan")
    return {
        "tau": tau,
        "mean_diag_auc": mean_diag_auc,
        "mean_offdiag_auc": mean_offdiag_auc,
        "diag_effect": diag_effect,
        "offdiag_effect": offdiag_effect,
        "interpretable": bool(mean_diag_auc >= min_diag_auc),
    }


def crossnobis_content_matrix(
    states: NDArray,
    labels: NDArray,
    folds: list[tuple[NDArray, NDArray]],
) -> dict:
    """Estimate an unbiased cross-temporal Mahalanobis content matrix.

    Condition contrasts from the training and held-out partitions are never
    multiplied with themselves.  A shrinkage precision matrix is fitted to
    training residuals only, making the expected distance zero under the null.
    """
    from sklearn.covariance import LedoitWolf

    values = np.asarray(states, dtype=float)
    condition = np.asarray(labels)
    if values.ndim != 3 or len(condition) != len(values):
        raise ValueError("states must have shape (trials, time, features)")
    matrices = []
    for train_index, test_index in folds:
        train_labels = condition[train_index]
        test_labels = condition[test_index]
        classes = np.intersect1d(np.unique(train_labels), np.unique(test_labels))
        if len(classes) < 2:
            continue
        train_means = {
            label: np.mean(values[train_index][train_labels == label], axis=0)
            for label in classes
        }
        test_means = {
            label: np.mean(values[test_index][test_labels == label], axis=0)
            for label in classes
        }
        residuals = np.concatenate([
            values[train_index][train_labels == label] - train_means[label][None, :, :]
            for label in classes
        ], axis=0).reshape(-1, values.shape[2])
        precision = LedoitWolf().fit(residuals).precision_
        pair_matrices = []
        for first_index, first in enumerate(classes):
            for second in classes[first_index + 1:]:
                train_contrast = train_means[first] - train_means[second]
                test_contrast = test_means[first] - test_means[second]
                pair_matrices.append(train_contrast @ precision @ test_contrast.T)
        if pair_matrices:
            matrices.append(np.mean(pair_matrices, axis=0))
    if not matrices:
        return {"status": "not_estimable", "reason": "no fold contains two shared classes"}
    matrix = np.mean(matrices, axis=0)
    matrix = 0.5 * (matrix + matrix.T)
    return {
        "status": "estimable",
        "matrix": matrix,
        "n_folds": len(matrices),
        "mean_diagonal_distance": float(np.mean(np.diag(matrix))),
        "mean_off_diagonal_distance": float(np.mean(matrix[~np.eye(len(matrix), dtype=bool)])),
    }


def crossnobis_decay_timescale(matrix: NDArray, dt: float) -> dict:
    """Fit a ratio-scale exponential timescale to a crossnobis matrix."""
    from scipy.optimize import curve_fit

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(values) < 4:
        return {"status": "not_estimable", "reason": "need a square matrix with four time bins"}
    lag_profile = np.asarray([
        np.mean(np.diag(values, k=lag)) for lag in range(len(values))
    ])
    if not np.all(np.isfinite(lag_profile)) or lag_profile[0] <= 0.0:
        return {
            "status": "not_estimable",
            "reason": "crossnobis diagonal is nonpositive or non-finite",
            "lag_profile": lag_profile.tolist(),
        }

    def decay(lag: NDArray, amplitude: float, tau: float, offset: float) -> NDArray:
        return amplitude * np.exp(-lag / tau) + offset

    lag_seconds = dt * np.arange(len(values))
    lower_tau = dt / 10.0
    upper_tau = dt * len(values) * 20.0
    try:
        parameters, _ = curve_fit(
            decay, lag_seconds, lag_profile,
            p0=(max(lag_profile[0] - lag_profile[-1], 1e-6), max(dt * 3.0, dt), lag_profile[-1]),
            bounds=([0.0, lower_tau, -np.inf], [np.inf, upper_tau, np.inf]),
            maxfev=20000,
        )
    except Exception as exc:
        return {"status": "not_estimable", "reason": f"exponential fit failed: {exc}"}
    fitted = decay(lag_seconds, *parameters)
    denominator = float(np.sum((lag_profile - np.mean(lag_profile)) ** 2))
    r_squared = 1.0 - float(np.sum((lag_profile - fitted) ** 2)) / max(denominator, 1e-12)
    boundary_hit = bool(
        parameters[1] <= lower_tau * (1.0 + 1e-5)
        or parameters[1] >= upper_tau * 0.95
    )
    estimable = bool(r_squared >= 0.0 and not boundary_hit)
    reason = None
    if r_squared < 0.0:
        reason = "exponential fit is worse than a constant"
    elif boundary_hit:
        reason = "timescale reached a declared optimization bound"
    return {
        "status": "estimable" if estimable else "not_estimable",
        "reason": reason,
        "timescale_seconds": float(parameters[1]),
        "timescale_bounds_seconds": [float(lower_tau), float(upper_tau)],
        "timescale_bound_hit": boundary_hit,
        "amplitude": float(parameters[0]),
        "offset": float(parameters[2]),
        "r_squared": r_squared,
        "lag_profile": lag_profile.tolist(),
    }


def _condition_time_marginals(Z: NDArray, labels: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Shared ANOVA-style marginalization used by marginalize_condition_time and
    dpca_condition_subspace_projection: the grand-mean-removed condition-averaged
    PSTH split into a time-only (condition-independent) term and a
    condition-main-effect-plus-interaction (condition-dependent) term.

    Returns
    -------
    time_marginal : (T, k) — condition-independent
    cond_marginal : (C, T, k) — condition-dependent
    conds         : (C,) — unique condition values, in the order used above
    """
    labels = np.asarray(labels)
    conds = np.unique(labels)
    psth = np.stack([Z[labels == c].mean(axis=0) for c in conds], axis=0)  # (C, T, k)
    grand_mean = psth.mean(axis=(0, 1))                                     # (k,)
    time_marginal = psth.mean(axis=0) - grand_mean                          # (T, k)
    cond_marginal = psth - time_marginal[None] - grand_mean                  # (C, T, k)
    return time_marginal, cond_marginal, conds


def marginalize_condition_time(
    Z: NDArray,
    labels: NDArray,
) -> dict:
    """Simplified (marginalization-based) dPCA: condition-independent vs -dependent variance.

    Full dPCA (Kobak et al. 2016, eLife) decomposes population activity into
    variance marginalized over each task parameter. This implements the
    two-factor (time × condition) case directly via ANOVA-style marginalization
    without the external dPCA package: the condition-averaged PSTH is split into
    a purely time-varying, condition-independent component (average over
    conditions at each time) and a condition-dependent component (what's left
    after removing the time-only and grand-mean terms — the condition main
    effect plus the condition×time interaction).

    Used to test whether the temporally-stable ("square") CTG structure lives
    in the condition-independent marginalization (pure time/context, present
    regardless of load) or the condition-dependent one (load/content-specific) —
    if it's almost entirely condition-dependent variance, that marginalization
    is what a classifier decoding "load" is picking up, which is the expected
    and unsurprising result; if condition-independent variance also carries
    separable structure, that would need separate explanation.

    Parameters
    ----------
    Z      : (N, T, k) — latent trajectories
    labels : (N,) — condition labels (any cardinality)

    Returns
    -------
    dict:
      var_total, var_condition_independent, var_condition_dependent : float
      frac_condition_independent, frac_condition_dependent : float
      pr_condition_independent, pr_condition_dependent : float — PR of each
        marginalization's own covariance spectrum (dimensionality of each code)
    """
    time_marginal, cond_marginal, conds = _condition_time_marginals(Z, labels)
    k = Z.shape[2]
    grand_mean_removed_psth = time_marginal[None] + cond_marginal  # psth - grand_mean, reconstructed
    var_total = float(np.sum(grand_mean_removed_psth ** 2))
    var_ci = float(np.sum(time_marginal ** 2) * len(conds))
    var_cd = float(np.sum(cond_marginal ** 2))

    def _pr_of(mat_2d: NDArray) -> float:
        Xc = mat_2d - mat_2d.mean(0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        return float(participation_ratio(s**2))

    return {
        "var_total": var_total,
        "var_condition_independent": var_ci,
        "var_condition_dependent": var_cd,
        "frac_condition_independent": var_ci / (var_total + 1e-12),
        "frac_condition_dependent": var_cd / (var_total + 1e-12),
        "pr_condition_independent": _pr_of(time_marginal),
        "pr_condition_dependent": _pr_of(cond_marginal.reshape(-1, k)),
    }


def dpca_condition_subspace_projection(
    Z: NDArray,
    labels: NDArray,
    n_components: int = 4,
) -> dict:
    """Project every single trial onto its own condition-independent and
    condition-dependent marginalization subspace.

    marginalize_condition_time reports variance FRACTIONS but not per-trial
    projected trajectories, so it cannot answer "does the temporally-stable
    CTG structure ride the condition-dependent or condition-independent
    component" directly — that requires re-running CTG on each component's own
    single-trial projection. This computes the top n_components principal axes
    of each marginalization's own covariance (time_marginal for
    condition-independent; cond_marginal, pooled over conditions, for
    condition-dependent) and projects every trial's full trajectory Z onto
    each subspace, so ctg_label_permutation_null / temporal_stability_tau can
    be run on both and compared.

    Parameters
    ----------
    Z            : (N, T, k) — latent trajectories
    labels       : (N,) — condition labels (any cardinality)
    n_components : subspace dimensionality for each marginalization (<=k)

    Returns
    -------
    dict:
      Z_condition_independent, Z_condition_dependent : (N, T, d) — per-trial
        projections onto each marginalization's own top axes (d = min(n_components, k))
      V_condition_independent, V_condition_dependent : (k, d) — the axes themselves
    """
    time_marginal, cond_marginal, conds = _condition_time_marginals(Z, labels)
    N, T, k = Z.shape
    d = min(n_components, k)

    def _top_axes(mat_2d: NDArray) -> NDArray:
        Xc = mat_2d - mat_2d.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt[:d].T  # (k, d)

    V_ci = _top_axes(time_marginal)
    V_cd = _top_axes(cond_marginal.reshape(-1, k))

    return {
        "Z_condition_independent": Z @ V_ci,
        "Z_condition_dependent": Z @ V_cd,
        "V_condition_independent": V_ci,
        "V_condition_dependent": V_cd,
    }


def phase_scramble_trials(Z: NDArray, rng: np.random.Generator) -> NDArray:
    """Randomise the Fourier phase of each trial, shared across channels.

    Preserves each channel's amplitude spectrum (hence its autocorrelation
    structure) while destroying any condition- or time-aligned structure
    across trials — the standard "could this just be smoothed noise" null.

    Uses ONE random phase perturbation per trial, ADDED to every channel's
    own phase at each frequency (Prichard & Theiler 1994 multivariate
    surrogate construction) rather than replacing each channel's phase
    independently. Because the same perturbation is added to every channel,
    the phase DIFFERENCE between any two channels at a given frequency — and
    therefore the instantaneous cross-channel covariance — is left exactly
    unchanged; only the trial's alignment to task/condition timing is
    destroyed. Randomising phase independently per channel instead destroys
    cross-channel covariance along with temporal structure, which makes the
    resulting null test only "is there any cross-channel structure at all"
    rather than "is the structure time-aligned" — too weak a null.

    Parameters
    ----------
    Z   : (N, T, k) — trajectories (trials x time x channels)
    rng : random number generator

    Returns
    -------
    (N, T, k) phase-scrambled trajectories
    """
    N, T, k = Z.shape
    Z_out = np.empty_like(Z)
    for n in range(N):
        n_freq = T // 2 + 1
        shift = rng.uniform(0, 2 * np.pi, n_freq)  # shared across channels
        rotor = np.exp(1j * shift)
        for d in range(k):
            x = Z[n, :, d]
            fft = np.fft.rfft(x)
            fft_scrambled = fft * rotor
            fft_scrambled[0] = fft[0]  # DC
            if T % 2 == 0:
                fft_scrambled[-1] = np.abs(fft[-1])  # Nyquist: must stay real
            Z_out[n, :, d] = np.fft.irfft(fft_scrambled, n=T)
    return Z_out


def ctg_phase_scramble_null(
    Z: NDArray,
    labels: NDArray,
    step: int = 40,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[float, NDArray]:
    """Bootstrap τ-null via phase scrambling — controls for autocorrelation.

    For each permutation, each trial's latent trajectory is phase-scrambled:
    Fourier phases are randomised (preserving amplitude spectrum, thus preserving
    autocorrelation), but the condition-aligned temporal structure is destroyed.
    CTG is recomputed on scrambled data, yielding τ_null.

    If observed τ > 95th percentile of τ_null, the result cannot be explained
    by temporal autocorrelation alone.

    Parameters
    ----------
    Z              : (N, T, k) — latent trajectories
    labels         : (N,)      — binary labels
    step           : timepoint stride for CTG
    n_permutations : number of null permutations
    rng            : random number generator
    verbose        : if True, print progress every 50 permutations to stderr

    Returns
    -------
    tau_obs  : float — observed τ on unscrambled data
    tau_null : (n_permutations,) — null τ distribution
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.default_rng(42)

    T = Z.shape[1]
    t_idx = np.arange(0, T, step)
    n_t = len(t_idx)

    def _auc(scores, y):
        pos, neg = scores[y == 1], scores[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        u = float(np.sum(pos[:, None] > neg[None, :]))
        return u / (len(pos) * len(neg))

    def _compute_tau(Z_in):
        auc_mat = np.full((n_t, n_t), np.nan)
        for i, ti in enumerate(t_idx):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(Z_in[:, ti, :])
            clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
            clf.fit(X_tr_s, labels)
            for j, tj in enumerate(t_idx):
                auc_mat[i, j] = _auc(clf.decision_function(sc.transform(Z_in[:, tj, :])),
                                      labels)
        off_mask = ~np.eye(n_t, dtype=bool)
        mean_off = float(np.nanmean(auc_mat[off_mask]))
        mean_dia = float(np.nanmean(np.diag(auc_mat)))
        return mean_off / mean_dia if mean_dia > 1e-6 else np.nan

    tau_obs = _compute_tau(Z)

    tau_null = np.zeros(n_permutations)
    for p in range(n_permutations):
        Z_scrambled = phase_scramble_trials(Z, rng)
        tau_null[p] = _compute_tau(Z_scrambled)
        if verbose and (p + 1) % 50 == 0:
            import sys
            print(f"      Phase-scramble null {p+1}/{n_permutations}", file=sys.stderr)

    return tau_obs, tau_null
