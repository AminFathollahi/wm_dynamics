"""attractor_analysis.py — persistent homology, recurrence quantification, and
local-linear fixed-point/Jacobian classification on population latent
trajectories.

Implements the three attractor-identification methods scored against the
mouse ALM ground truth (Inagaki et al. 2019, Nature 566:212-217) before any
of them is licensed to describe an unvalidated human corpus:
  - Vietoris-Rips persistent homology (Betti numbers at the largest
    persistence gap), via ripser (Bauer 2021, J Appl Comput Topol 5:391-423).
  - Recurrence quantification analysis (Marwan et al. 2007, Phys Rep
    438:237-329): recurrence rate, determinism, laminarity, trapping time,
    and connected-component recurrence clusters.
  - Fixed-point discovery and Jacobian classification in the spirit of
    Sussillo & Barak 2013 (Neural Comput 25:626-649), adapted to a
    discrete-time local-linear map (the data are binned, not a continuous
    RNN) found by Newton iteration on a k-nearest-neighbour local-linear
    regression of x[t+1] on x[t].
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

import ripser

# ── Persistent homology ─────────────────────────────────────────────────────


BETTI_GAP_SIGNIFICANCE_RATIO = 6.0
BETTI_GAP_TOP_K = 8


def betti_numbers_at_largest_gap(
    dgms: list[NDArray],
    significance_ratio: float = BETTI_GAP_SIGNIFICANCE_RATIO,
    top_k: int = BETTI_GAP_TOP_K,
) -> dict:
    """Betti number per dimension, read off the largest persistence gap.

    Sort finite persistence lengths (death - birth) descending, restricted
    to the top `top_k` (the only candidates a handful of genuine attractor
    basins could produce; the long tail below is within-basin sampling
    noise and its raw scale is not comparable to the top). The Betti number
    is the count of features above the single largest gap among those
    candidates, plus any always-infinite features (H0's final
    all-connected component) -- but only if that gap is at least
    `significance_ratio` times the median of the surrounding candidate
    gaps. Below that ratio there is no gap distinguishable from sampling
    noise, so the count defaults to 0 finite features (e.g. a unimodal
    Gaussian correctly reads beta_0 = 1, not 2, because its largest gap is
    not appreciably bigger than its neighbours). The ratio was set by
    requiring a near-zero false-positive rate on 2D/3D Gaussian-matched and
    ring-null point clouds at n ~ 150 (see tests/test_attractor_recovery_control.py).
    """
    result = {}
    for dim, dgm in enumerate(dgms):
        dgm = np.asarray(dgm)
        finite_mask = np.isfinite(dgm[:, 1])
        finite = dgm[finite_mask]
        n_infinite = int((~finite_mask).sum())
        lengths = np.sort(finite[:, 1] - finite[:, 0])[::-1]
        if len(lengths) == 0:
            result[dim] = {"betti": n_infinite, "max_persistence": 0.0}
            continue
        if len(lengths) == 1:
            result[dim] = {"betti": int(1 + n_infinite), "max_persistence": float(lengths[0])}
            continue
        k = min(top_k, len(lengths) - 1)
        candidates = lengths[: k + 1]
        gaps = candidates[:-1] - candidates[1:]
        baseline = np.median(lengths[k // 2:]) if len(lengths) > k else np.median(lengths)
        best = int(np.argmax(gaps))
        significance = gaps[best] / (baseline + 1e-12)
        cut = (best + 1) if significance > significance_ratio else 0
        result[dim] = {"betti": int(cut + n_infinite), "max_persistence": float(lengths[0])}
    return result


def persistent_homology(point_cloud: NDArray, maxdim: int = 2) -> dict:
    """Vietoris-Rips persistence diagrams and Betti numbers at the largest gap."""
    computed = ripser.ripser(np.asarray(point_cloud, dtype=float), maxdim=maxdim)
    dgms = computed["dgms"]
    return {
        "betti": betti_numbers_at_largest_gap(dgms),
        "diagrams": [dgm.tolist() for dgm in dgms],
    }


def gaussian_matched_null(point_cloud: NDArray, rng: np.random.Generator) -> NDArray:
    """A Gaussian point cloud matched on the observed mean and covariance.

    beta_0 = 1, beta_1 = 0 by construction (a single unimodal blob); the
    null that makes any topological claim about the real cloud non-trivial.
    """
    point_cloud = np.asarray(point_cloud, dtype=float)
    mean = point_cloud.mean(axis=0)
    cov = np.cov(point_cloud, rowvar=False)
    return rng.multivariate_normal(mean, cov, size=len(point_cloud))


def time_shuffled_null(trial_tensor: NDArray, rng: np.random.Generator) -> NDArray:
    """Shuffle time-bin order within each trial independently.

    trial_tensor : (n_trials, n_bins, n_channels), pre-subsample. Controls
    for autocorrelation while preserving the marginal geometry of the cloud
    (every point keeps its value; only its time index within its trial
    changes).
    """
    shuffled = np.asarray(trial_tensor, dtype=float).copy()
    n_trials, n_bins, _ = shuffled.shape
    for i in range(n_trials):
        shuffled[i] = shuffled[i, rng.permutation(n_bins)]
    return shuffled


def null_percentile(observed: float, null_values: NDArray) -> float:
    """Percentile rank of `observed` within `null_values` (0-100)."""
    null_values = np.asarray(null_values)
    return float(100.0 * (null_values < observed).mean())


# ── Recurrence quantification ───────────────────────────────────────────────


def recurrence_matrix(trajectory: NDArray, threshold: float) -> NDArray:
    dist = squareform(pdist(trajectory))
    mat = (dist <= threshold).astype(int)
    np.fill_diagonal(mat, 0)
    return mat


def threshold_for_target_recurrence_rate(trajectory: NDArray, target_rr: float) -> float:
    """Bisect the pairwise-distance threshold to hit a target recurrence rate."""
    n = len(trajectory)
    dist = pdist(trajectory)
    lo, hi = 0.0, float(dist.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rr = 2.0 * (dist <= mid).sum() / (n * n)
        if rr < target_rr:
            lo = mid
        else:
            hi = mid
    return hi


def _line_lengths(binary_1d: NDArray, min_len: int) -> list[int]:
    lengths = []
    run = 0
    for value in binary_1d:
        if value:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return [length for length in lengths if length >= min_len]


def recurrence_quantification(trajectory: NDArray, threshold: float, min_diag: int = 2, min_vert: int = 2) -> dict:
    """Recurrence rate, determinism, laminarity, and trapping time.

    Definitions follow Marwan et al. 2007 (Phys Rep 438:237-329):
    determinism is the fraction of recurrent points forming diagonal line
    structures of length >= min_diag; laminarity is the fraction forming
    vertical line structures of length >= min_vert; trapping time is the
    mean vertical-line length.
    """
    mat = recurrence_matrix(trajectory, threshold)
    n = mat.shape[0]
    total_recurrent = int(mat.sum())
    rr = total_recurrent / (n * n)

    diag_lengths: list[int] = []
    for offset in range(-(n - 1), n):
        diag_lengths += _line_lengths(np.diagonal(mat, offset=offset), min_diag)
    vert_lengths: list[int] = []
    for col in range(n):
        vert_lengths += _line_lengths(mat[:, col], min_vert)

    det = (sum(diag_lengths) / total_recurrent) if total_recurrent > 0 and diag_lengths else 0.0
    lam = (sum(vert_lengths) / total_recurrent) if total_recurrent > 0 and vert_lengths else 0.0
    tt = float(np.mean(vert_lengths)) if vert_lengths else 0.0
    return {
        "recurrence_rate": float(rr),
        "determinism": float(det),
        "laminarity": float(lam),
        "trapping_time": tt,
        "threshold": float(threshold),
    }


def recurrence_clusters(trajectory: NDArray, threshold: float, min_cluster_frac: float = 0.02) -> dict:
    """Connected components of the recurrence graph, filtered to non-trivial clusters."""
    mat = recurrence_matrix(trajectory, threshold)
    n_components, labels = connected_components(csr_matrix(mat), directed=False)
    counts = np.bincount(labels, minlength=n_components)
    floor = max(2, int(min_cluster_frac * len(trajectory)))
    keep = counts >= floor
    return {
        "n_clusters": int(keep.sum()),
        "labels": labels.tolist(),
        "cluster_sizes": counts.tolist(),
    }


# ── Fixed points and local-linear Jacobians ─────────────────────────────────

NEAR_ONE_TOL = 0.08


def build_transition_pairs(latent: NDArray) -> tuple[NDArray, NDArray]:
    """Within-trial consecutive-bin pairs (x_t, x_t+1); never crosses a trial boundary.

    latent : (n_trials, n_bins, k)
    """
    x_t = latent[:, :-1, :].reshape(-1, latent.shape[2])
    x_tp1 = latent[:, 1:, :].reshape(-1, latent.shape[2])
    return x_t, x_tp1


def local_linear_map(x_t: NDArray, x_tp1: NDArray, query: NDArray, k: int = 60) -> tuple[NDArray, NDArray]:
    """Local-linear regression of x[t+1] on x[t] near `query`.

    Returns (f(query), local Jacobian M) for the discrete map x[t+1] ~= M @
    (x[t]-centre) + f(centre). Eigenvalues of M are the discrete-time
    stability spectrum used by `classify_fixed_point`. `k` trades locality
    for regression variance: with per-step process noise present (any real
    trial-to-trial variability), fitting a d x d Jacobian from only a
    handful of neighbours is dominated by sampling noise and produces
    spurious/inflated eigenvalues; k=60 was the smallest value that gave a
    stable, non-inflated spectrum on the planted two-well/line-attractor
    calibration systems at ALM's real session dimensions (see
    tests/test_attractor_recovery_control.py).
    """
    dists = np.linalg.norm(x_t - query[None, :], axis=1)
    k = min(k, len(x_t))
    nn = np.argsort(dists)[:k]
    centre_x = x_t[nn].mean(axis=0)
    centre_y = x_tp1[nn].mean(axis=0)
    dx = x_t[nn] - centre_x
    dy = x_tp1[nn] - centre_y
    jac, *_ = np.linalg.lstsq(dx, dy, rcond=None)
    f_query = centre_y + (query - centre_x) @ jac
    return f_query, jac


def _newton_fixed_point_step(x_t: NDArray, x_tp1: NDArray, x0: NDArray, k: int, damping: float) -> tuple[NDArray, NDArray]:
    f_x0, jac = local_linear_map(x_t, x_tp1, x0, k=k)
    identity = np.eye(len(x0))
    lhs = (identity - jac).T
    rhs = f_x0 - x0 @ jac
    try:
        target = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        target = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    x_new = x0 + damping * (target - x0)
    return x_new, jac


def find_fixed_point(
    x_t: NDArray, x_tp1: NDArray, x0: NDArray, k: int = 60,
    max_iter: int = 50, tol: float = 1e-4, damping: float = 0.5,
    bounds: tuple[NDArray, NDArray] | None = None,
) -> tuple[NDArray, NDArray, float]:
    """Newton iteration to a fixed point of the local-linear discrete map, from x0."""
    x = np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        x_new, _ = _newton_fixed_point_step(x_t, x_tp1, x, k, damping)
        if bounds is not None:
            x_new = np.clip(x_new, bounds[0], bounds[1])
        step_size = np.linalg.norm(x_new - x)
        x = x_new
        if step_size < tol:
            break
    f_x, jac = local_linear_map(x_t, x_tp1, x, k=k)
    residual = float(np.linalg.norm(f_x - x))
    return x, jac, residual


def classify_fixed_point(eigenvalues: NDArray) -> str:
    """`stable_point` / `line_attractor` / `limit_cycle` / `saddle` from a discrete Jacobian spectrum."""
    eigs = np.asarray(eigenvalues)
    mags = np.abs(eigs)
    near_one = np.abs(mags - 1.0) < NEAR_ONE_TOL
    # "decaying" is exactly "not near one, and below it" -- near_one and decaying
    # partition the whole magnitude axis with "growing" (above 1+tol) so no
    # eigenvalue falls into a classification dead zone.
    decaying = (~near_one) & (mags < 1.0)
    complex_near_one = near_one & (np.abs(eigs.imag) > 1e-6)
    if int(complex_near_one.sum()) >= 2:
        return "limit_cycle"
    if bool(decaying.all()):
        return "stable_point"
    if int(near_one.sum()) == 1 and bool(decaying[~near_one].all()):
        return "line_attractor"
    return "saddle"


MIN_BASIN_SIZE = 3


def find_fixed_points(
    x_t: NDArray, x_tp1: NDArray, n_inits: int, rng: np.random.Generator,
    bounds: tuple[NDArray, NDArray], k: int = 60, max_iter: int = 50,
    tol: float = 1e-4, damping: float = 0.5, residual_tol: float | None = None,
    cluster_tol: float | None = None, min_basin_size: int = MIN_BASIN_SIZE,
) -> list[dict]:
    """Search for fixed points from many initial conditions, cluster, and classify.

    Locates fixed points of the local-linear discrete map by minimising
    ||f(x)-x|| via Newton iteration (Sussillo & Barak 2013 style, adapted to
    a discrete-time k-NN local-linear map since the data are binned).
    Initial conditions are half drawn from the observed data (states the
    system actually visits, including transition/decision regions near
    saddles) and half uniform-random over the observed bounding box (unbiased
    coverage); a saddle between two well-visited basins is rarely visited
    itself, so data-only seeding under-samples it, and bounding-box-only
    seeding under-samples the visited manifold. A discovered fixed point is
    kept only if at least `min_basin_size` independent initial conditions
    converge to it (a cluster of 1-2 is indistinguishable from a local
    numerical artifact of one noisy regression fit).
    """
    dim = x_t.shape[1]
    if residual_tol is None:
        residual_tol = 0.05 * np.linalg.norm(bounds[1] - bounds[0])
    if cluster_tol is None:
        cluster_tol = 0.1 * np.linalg.norm(bounds[1] - bounds[0])

    n_data_inits = n_inits // 2
    data_idx = rng.choice(len(x_t), size=min(n_data_inits, len(x_t)), replace=False)
    inits = np.vstack([
        x_t[data_idx],
        rng.uniform(bounds[0], bounds[1], size=(n_inits - len(data_idx), dim)),
    ])
    candidates = []
    for x0 in inits:
        x_star, jac, residual = find_fixed_point(
            x_t, x_tp1, x0, k=k, max_iter=max_iter, tol=tol, damping=damping, bounds=bounds,
        )
        if residual < residual_tol:
            candidates.append((x_star, jac, residual))

    if not candidates:
        return []

    points = np.array([c[0] for c in candidates])
    if len(points) == 1:
        labels = np.zeros(1, dtype=int)
        n_clusters = 1
    else:
        adjacency = (squareform(pdist(points)) <= cluster_tol).astype(int)
        np.fill_diagonal(adjacency, 0)
        n_clusters, labels = connected_components(csr_matrix(adjacency), directed=False)

    fixed_points = []
    for cluster_id in range(n_clusters):
        members = np.where(labels == cluster_id)[0]
        if len(members) < min_basin_size:
            continue
        representative = members[int(np.argmin([candidates[i][2] for i in members]))]
        x_star, jac, residual = candidates[representative]
        eigs = np.linalg.eigvals(jac)
        fixed_points.append({
            "location": x_star.tolist(),
            "jacobian_eigenvalues_real": eigs.real.tolist(),
            "jacobian_eigenvalues_imag": eigs.imag.tolist(),
            "residual": float(residual),
            "n_converged_inits_in_basin": int(len(members)),
            "classification": classify_fixed_point(eigs),
        })
    return fixed_points


def summarize_fixed_point_pattern(fixed_points: list[dict]) -> str:
    """Collapse a discovered fixed-point set into one of a fixed vocabulary of qualitative patterns."""
    if not fixed_points:
        return "no_fixed_points_found"
    counts = Counter(fp["classification"] for fp in fixed_points)
    n_stable = counts.get("stable_point", 0)
    n_saddle = counts.get("saddle", 0)
    n_line = counts.get("line_attractor", 0)
    n_cycle = counts.get("limit_cycle", 0)
    if n_stable >= 2 and n_saddle >= 1:
        return "two_stable_plus_saddle"
    if n_line >= 1 and n_line >= n_stable:
        return "line_attractor_dominant"
    if n_stable == 1 and n_saddle == 0 and n_line == 0 and n_cycle == 0:
        return "single_stable_point"
    if n_cycle >= 1:
        return "limit_cycle_dominant"
    return "no_consistent_pattern"


# ── Planted-truth simulators (fixed-point-classifier calibration) ──────────


def _embed_and_standardize(latent_true: NDArray, n_units: int, rng: np.random.Generator, loading_scale: float, noise_sd: float) -> NDArray:
    n_trials, n_bins, true_dim = latent_true.shape
    loading = rng.normal(0, loading_scale, size=(true_dim, n_units))
    signal = latent_true @ loading
    observed = signal + rng.normal(0, noise_sd, size=signal.shape)
    mean = observed.mean(axis=(0, 1), keepdims=True)
    std = observed.std(axis=(0, 1), keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (observed - mean) / std


def simulate_two_well(n_trials: int, n_bins: int, n_units: int, dt: float, rng: np.random.Generator,
                       a: float = 0.4, b: float = 1.0, drive_noise_sd: float = 0.15,
                       loading_scale: float = 1.0, observation_noise_sd: float = 0.15) -> NDArray:
    """Classic 2D double well: U(x,y) = a(x^2-1)^2 + b*y^2 -- stable points at (+-1,0), saddle at (0,0).

    The continuous-time rate at the wells is 8a (linearizing -4a*x*(x^2-1)
    at x=+-1); Euler integration is unstable once 8a*dt >~ 2, so `a`'s
    default is set relative to the project's 100ms ALM bin width (dt=0.1) to
    stay well inside the stable regime -- a=3.0 (a natural-looking "steep
    well" choice) discretizes to a genuinely unstable local map at that bin
    width and was caught by tests/test_attractor_recovery_control.py.
    """
    latent = np.zeros((n_trials, n_bins, 2))
    sign = rng.choice([-1.0, 1.0], size=n_trials)
    latent[:, 0, 0] = sign * 0.3 + rng.normal(0, 0.1, n_trials)
    latent[:, 0, 1] = rng.normal(0, 0.1, n_trials)
    for t in range(1, n_bins):
        x, y = latent[:, t - 1, 0], latent[:, t - 1, 1]
        dx = -4 * a * x * (x**2 - 1)
        dy = -2 * b * y
        latent[:, t, 0] = x + dx * dt + rng.normal(0, drive_noise_sd * np.sqrt(dt), n_trials)
        latent[:, t, 1] = y + dy * dt + rng.normal(0, drive_noise_sd * np.sqrt(dt), n_trials)
    return _embed_and_standardize(latent, n_units, rng, loading_scale, observation_noise_sd)


def simulate_line_attractor(n_trials: int, n_bins: int, n_units: int, dt: float, rng: np.random.Generator,
                             k: float = 3.0, drive_noise_sd: float = 0.15,
                             loading_scale: float = 1.0, observation_noise_sd: float = 0.15) -> NDArray:
    """Fast contraction onto the y-axis (x -> 0); free diffusion along y (the line of fixed points)."""
    latent = np.zeros((n_trials, n_bins, 2))
    latent[:, 0] = rng.normal(0, 0.3, (n_trials, 2))
    for t in range(1, n_bins):
        x, y = latent[:, t - 1, 0], latent[:, t - 1, 1]
        latent[:, t, 0] = x - k * x * dt + rng.normal(0, drive_noise_sd * np.sqrt(dt), n_trials)
        latent[:, t, 1] = y + rng.normal(0, drive_noise_sd * np.sqrt(dt), n_trials)
    return _embed_and_standardize(latent, n_units, rng, loading_scale, observation_noise_sd)


def simulate_point_attractor(n_trials: int, n_bins: int, n_units: int, dt: float, rng: np.random.Generator,
                              k: float = 3.0, drive_noise_sd: float = 0.15,
                              loading_scale: float = 1.0, observation_noise_sd: float = 0.15) -> NDArray:
    """Global contraction to the origin in both dimensions -- a single stable point."""
    latent = np.zeros((n_trials, n_bins, 2))
    latent[:, 0] = rng.normal(0, 0.5, (n_trials, 2))
    for t in range(1, n_bins):
        state = latent[:, t - 1]
        latent[:, t] = state - k * state * dt + rng.normal(0, drive_noise_sd * np.sqrt(dt), (n_trials, 2))
    return _embed_and_standardize(latent, n_units, rng, loading_scale, observation_noise_sd)


def simulate_pure_noise(n_trials: int, n_bins: int, n_units: int, dt: float, rng: np.random.Generator,
                         drive_noise_sd: float = 0.4, loading_scale: float = 1.0,
                         observation_noise_sd: float = 0.5) -> NDArray:
    """Undrifted Brownian motion -- no restoring force anywhere, no true fixed point."""
    latent = np.zeros((n_trials, n_bins, 2))
    latent[:, 0] = rng.normal(0, 0.3, (n_trials, 2))
    for t in range(1, n_bins):
        latent[:, t] = latent[:, t - 1] + rng.normal(0, drive_noise_sd * np.sqrt(dt), (n_trials, 2))
    return _embed_and_standardize(latent, n_units, rng, loading_scale, observation_noise_sd)


def simulate_ring(n_points: int, rng: np.random.Generator, radius: float = 1.0, noise_sd: float = 0.05) -> NDArray:
    """A noisy 2D ring point cloud -- for validating beta_1 = 1 recovery on a known loop."""
    angles = rng.uniform(0, 2 * np.pi, n_points)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return ring + rng.normal(0, noise_sd, ring.shape)
