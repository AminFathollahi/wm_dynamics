"""small_worldness.py -- the one definition of Watts-Strogatz sigma this
repository uses, adopted from the sibling RNN codebase's own network-analysis
module (brainalign_wm/analysis/network_properties.py) so a functional graph
computed here and one computed there can later be compared rather than
re-derived: same sparsification (top DENSITY fraction of edges by absolute
weight, symmetrized), same null construction (networkx's own random-reference
rewiring inside `nx.algorithms.smallworld.sigma`), same resolution settings
(a coarse, fast estimate -- N_RANDOM/N_ITER are small on purpose, and
MAX_NODES bounds a whole grid's worth of graphs to a tractable runtime).

Two call sites in this repository previously restated this definition with
different settings instead of importing one shared implementation (top 20%
of edges with 50 Maslov-Sneppen rewirings claimed in one artifact's own
metadata string versus the 15%-density, 20-null-draw Maslov-Sneppen battery
the code that produced both artifacts actually ran) -- see
scripts/run_observability_and_power_census.py's
small_worldness_sigma_definition_change for the full account of that defect
and what it turned out to be. Both call sites now import this module.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

DENSITY = 0.10
N_RANDOM = 2
N_ITER = 2
MAX_NODES = 64
MIN_COMPONENT_NODES = 10


def thresholded_largest_component_graph(
    C: NDArray, density: float = DENSITY, max_nodes: int = MAX_NODES, seed: int = 0,
) -> nx.Graph | None:
    """Subsample `C` to `max_nodes` (fixed `seed`, reproducible), symmetrize,
    keep the top `density` fraction of edges by |weight|, and return the
    largest connected component. Returns None if that component has fewer
    than MIN_COMPONENT_NODES nodes -- below that floor, a random-reference
    sigma comparison is not a measurement of anything.
    """
    C = np.asarray(C)
    if C.shape[0] > max_nodes:
        idx = np.random.RandomState(seed).choice(C.shape[0], size=max_nodes, replace=False)
        C = C[np.ix_(idx, idx)]
    C_sym = C + C.T
    n = C_sym.shape[0]
    flat = C_sym[np.triu_indices(n, k=1)]
    if flat.size == 0 or np.count_nonzero(flat) == 0:
        return None
    threshold = np.quantile(flat[flat > 0], 1.0 - density) if np.any(flat > 0) else np.inf
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, j in zip(*np.triu_indices(n, k=1)):
        if C_sym[i, j] >= threshold:
            graph.add_edge(int(i), int(j))
    if graph.number_of_nodes() == 0:
        return None
    largest_cc = max(nx.connected_components(graph), key=len)
    if len(largest_cc) < MIN_COMPONENT_NODES:
        return None
    return graph.subgraph(largest_cc).copy()


def watts_strogatz_sigma(
    C: NDArray, density: float = DENSITY, n_random: int = N_RANDOM, n_iter: int = N_ITER,
    seed: int = 0, max_nodes: int = MAX_NODES,
) -> float | None:
    """Watts-Strogatz small-worldness sigma on the thresholded, subsampled
    largest connected component of `C`. Returns None if that component is
    too small (< MIN_COMPONENT_NODES nodes) or if networkx's own sigma
    computation fails on the resulting graph."""
    graph = thresholded_largest_component_graph(C, density=density, max_nodes=max_nodes, seed=seed)
    if graph is None:
        return None
    try:
        return float(nx.algorithms.smallworld.sigma(graph, niter=n_iter, nrand=n_random, seed=seed))
    except (nx.NetworkXError, ZeroDivisionError):
        return None


def degree_assortativity(
    C: NDArray, density: float = DENSITY, seed: int = 0, max_nodes: int = MAX_NODES,
) -> float | None:
    """Degree assortativity coefficient on the SAME thresholded, subsampled
    graph watts_strogatz_sigma builds, so the two metrics never disagree
    about which graph they were computed on."""
    graph = thresholded_largest_component_graph(C, density=density, max_nodes=max_nodes, seed=seed)
    if graph is None:
        return None
    try:
        r = nx.degree_assortativity_coefficient(graph)
    except (nx.NetworkXError, ZeroDivisionError):
        return None
    return float(r) if r == r else None  # nan-check


def definition_summary() -> dict:
    """Machine-readable settings, for any artifact that needs to disclose
    this definition rather than hand-type a description of it."""
    return {
        "sparsification": f"top {DENSITY:.0%} of edges by |weight|, symmetrized",
        "null_construction": f"networkx smallworld.sigma random-reference rewiring, {N_RANDOM} random references, {N_ITER} rewiring iterations per reference",
        "resolution": f"subsampled to at most {MAX_NODES} nodes, largest connected component only, minimum {MIN_COMPONENT_NODES} nodes to compute",
        "source": "ported from the sibling RNN codebase's brainalign_wm/analysis/network_properties.py",
    }
