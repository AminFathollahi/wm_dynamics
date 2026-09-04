"""microcircuit_graph.py -- functional connectivity and graph-organisation
metrics on a population of simultaneously recorded units, with a
degree-preserving (Maslov & Sneppen 2002) null for every metric.

Connectivity: Pearson correlation, the spike-time tiling coefficient (Cutts
& Eglen 2014; rate-insensitive, via elephant's reference implementation),
and a shrinkage-regularised precision (inverse covariance) matrix (Ledoit &
Wolf 2004).

Graph metrics use bctpy (Rubinov & Sporns 2010) throughout rather than
reimplementing standard algorithms: weight entropy, Louvain modularity,
small-worldness sigma (Humphries & Gurney 2008), degree assortativity, mean
clustering, characteristic path length, communicability, mean participation
coefficient, and the rich-club coefficient curve.
"""

from __future__ import annotations

import neo
import numpy as np
import quantities as pq
import bct
from numpy.typing import NDArray
from scipy.linalg import expm
from sklearn.covariance import LedoitWolf
from elephant.spike_train_correlation import spike_time_tiling_coefficient

STTC_COINCIDENCE_WINDOW_S = 0.05
LOUVAIN_RESOLUTION_GAMMA = 1.0
N_NULL_DRAWS = 20
N_REWIRE_ITERATIONS_PER_EDGE = 3
EDGE_DENSITY_KEEP_FRACTION = 0.15


def sparsify(W: NDArray, keep_fraction: float = EDGE_DENSITY_KEEP_FRACTION) -> NDArray:
    """Keep only the top `keep_fraction` of edges by |weight|; zero the rest.

    A functional-connectivity matrix from Pearson/precision/STTC is
    virtually complete (every pair has a nonzero weight), and Maslov-Sneppen
    rewiring's runtime blows up on a near-complete graph (few valid
    non-edges to swap into: ~150x slower on a complete 20-node graph than on
    the same graph at 20% density, measured directly). Thresholding to a
    fixed edge density before any topological metric is also standard
    practice in functional-connectivity graph analysis, since weak/spurious
    correlations otherwise dominate the topology.
    """
    Wabs = np.abs(W)
    triu = Wabs[np.triu_indices_from(Wabs, k=1)]
    if len(triu) == 0:
        return W.copy()
    threshold = np.percentile(triu, 100.0 * (1.0 - keep_fraction))
    sparse = np.where(Wabs >= threshold, W, 0.0)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def pearson_connectivity(counts: NDArray) -> NDArray:
    """counts: (n_units, n_samples) binned spike counts -> (n_units, n_units) correlation."""
    with np.errstate(invalid="ignore"):
        W = np.corrcoef(counts)
    W = np.nan_to_num(W, nan=0.0)
    np.fill_diagonal(W, 0.0)
    return W


def precision_connectivity(counts: NDArray) -> NDArray:
    """Shrinkage-regularised (Ledoit-Wolf) precision matrix, off-diagonal only."""
    estimator = LedoitWolf().fit(counts.T)
    precision = estimator.precision_
    W = -precision.copy()  # partial correlation sign convention: positive = excitatory-like coupling
    diag = np.sqrt(np.abs(np.diag(precision)))
    scale = np.outer(diag, diag)
    scale[scale == 0] = 1.0
    W = W / scale
    np.fill_diagonal(W, 0.0)
    return W


def sttc_connectivity(spike_lists: list, onsets: NDArray, window_s: float, dt: float = STTC_COINCIDENCE_WINDOW_S) -> NDArray:
    """Pairwise spike-time tiling coefficient, on each unit's spikes concatenated
    across trials within the epoch window (each trial's window shifted to its
    own zero and offset so trials never overlap in the concatenated train)."""
    n_units = len(spike_lists)
    trains = []
    cursor = 0.0
    concatenated = [[] for _ in range(n_units)]
    for onset in onsets:
        for unit_index, spikes in enumerate(spike_lists):
            in_window = spikes[(spikes >= onset) & (spikes < onset + window_s)] - onset + cursor
            concatenated[unit_index].append(in_window)
        cursor += window_s
    for unit_index in range(n_units):
        times = np.concatenate(concatenated[unit_index]) if concatenated[unit_index] else np.array([])
        trains.append(neo.SpikeTrain(np.sort(times) * pq.s, t_stop=cursor * pq.s))

    W = np.zeros((n_units, n_units))
    for i in range(n_units):
        for j in range(i + 1, n_units):
            value = float(spike_time_tiling_coefficient(trains[i], trains[j], dt=dt * pq.s))
            W[i, j] = W[j, i] = value
    return W


def weight_entropy(W: NDArray) -> float:
    weights = np.abs(W[np.triu_indices_from(W, k=1)])
    total = weights.sum()
    if total <= 0:
        return 0.0
    p = weights[weights > 0] / total
    return float(-np.sum(p * np.log(p)))


def graph_metrics(W: NDArray, rng: np.random.Generator, gamma: float = LOUVAIN_RESOLUTION_GAMMA) -> dict:
    """W is sparsified (see `sparsify`) before any metric is computed."""
    W = sparsify(W)
    n = W.shape[0]
    Wpos = np.abs(W)
    np.fill_diagonal(Wpos, 0.0)
    seed = int(rng.integers(0, 2**31))
    community, modularity_q = bct.community_louvain(Wpos, gamma=gamma, seed=seed)

    clustering = bct.clustering_coef_wu(Wpos)
    mean_clustering = float(np.mean(clustering))

    with np.errstate(divide="ignore"):
        length = np.where(Wpos > 0, 1.0 / Wpos, np.inf)
    np.fill_diagonal(length, 0.0)
    distance, *_ = bct.distance_wei(length)
    char_path_length, _, _, _, _ = bct.charpath(distance, include_diagonal=False, include_infinite=False)

    scale = Wpos.sum() / max(n * (n - 1), 1) + 1e-12
    communicability = expm(Wpos / scale)
    communicability_mean = float(np.mean(communicability[np.triu_indices(n, k=1)]))

    assortativity_r = float(bct.assortativity_wei(Wpos, flag=0))
    participation = bct.participation_coef(Wpos, community)
    mean_participation = float(np.mean(participation))
    rich_club = bct.rich_club_wu(Wpos)
    rich_club = np.asarray(rich_club)
    rich_club = rich_club[np.isfinite(rich_club)]

    return {
        "weight_entropy": weight_entropy(W),
        "modularity_q": float(modularity_q),
        "mean_clustering": mean_clustering,
        "characteristic_path_length": float(char_path_length) if np.isfinite(char_path_length) else None,
        "communicability_mean": communicability_mean,
        "assortativity_r": assortativity_r if np.isfinite(assortativity_r) else None,
        "mean_participation_coefficient": mean_participation,
        "rich_club_mean": float(np.mean(rich_club)) if len(rich_club) else None,
        "rich_club_curve": rich_club.tolist(),
    }


NULL_METRIC_NAMES = (
    "weight_entropy", "modularity_q", "mean_clustering", "characteristic_path_length",
    "communicability_mean", "assortativity_r", "mean_participation_coefficient", "rich_club_mean",
)


def null_percentile(observed: float | None, null_values: list) -> float | None:
    if observed is None:
        return None
    finite_null = [v for v in null_values if v is not None and np.isfinite(v)]
    if len(finite_null) < 10:
        return None
    return float(100.0 * (np.asarray(finite_null) < observed).mean())


def degree_preserving_null_battery(W: NDArray, rng: np.random.Generator, n_draws: int = N_NULL_DRAWS) -> dict:
    """Maslov & Sneppen 2002 degree-preserving rewiring, n_draws independent graphs.

    Rewires the already-sparsified graph (see `sparsify`) -- rewiring the
    dense input is what makes this prohibitively slow.
    """
    W = sparsify(W)
    n_edges = int((np.abs(W) > 0).sum() / 2)
    iterations = max(1, n_edges * N_REWIRE_ITERATIONS_PER_EDGE)
    null_metrics = {name: [] for name in NULL_METRIC_NAMES}
    for _ in range(n_draws):
        seed = int(rng.integers(0, 2**31))
        rewired, _ = bct.randmio_und(W, iterations, seed=seed)
        metrics = graph_metrics(rewired, rng)
        for name in NULL_METRIC_NAMES:
            null_metrics[name].append(metrics[name])
    return null_metrics


def small_worldness_sigma(W: NDArray, seed: int = 0) -> dict:
    """Small-worldness sigma on the RAW (unsparsified) connectivity matrix
    `W` -- unlike this module's other metrics, sigma no longer uses this
    module's own sparsify()/degree_preserving_null_battery() (Maslov-Sneppen,
    weighted). It defers entirely to src/small_worldness.py, this
    repository's single shared definition (top-10%-edge unweighted
    thresholding, networkx random-reference null, adopted from the sibling
    RNN codebase so a graph computed here and one computed there are
    comparable). See scripts/run_observability_and_power_census.py's
    small_worldness_sigma_definition_change for why this replaced the
    previous weighted, Maslov-Sneppen-null implementation that used to live
    here."""
    from small_worldness import definition_summary, watts_strogatz_sigma

    sigma = watts_strogatz_sigma(W, seed=seed)
    return {"sigma": sigma, "definition": definition_summary()}
