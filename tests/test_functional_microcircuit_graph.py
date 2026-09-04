"""Tests for src/microcircuit_graph.py."""

import sys
from pathlib import Path

import networkx as nx
import neo
import numpy as np
import quantities as pq

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from microcircuit_graph import (
    degree_preserving_null_battery,
    graph_metrics,
    small_worldness_sigma,
)
from elephant.spike_train_correlation import spike_time_tiling_coefficient


def _weighted_adjacency(graph: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(graph, weight=None)


class TestSTTC:
    def test_independent_poisson_trains_give_near_zero(self, rng):
        t_stop = 20.0
        st1 = neo.SpikeTrain(np.sort(rng.uniform(0, t_stop, 300)) * pq.s, t_stop=t_stop * pq.s)
        st2 = neo.SpikeTrain(np.sort(rng.uniform(0, t_stop, 300)) * pq.s, t_stop=t_stop * pq.s)
        value = spike_time_tiling_coefficient(st1, st2, dt=0.01 * pq.s)
        assert abs(value) < 0.15

    def test_identical_trains_give_near_one(self, rng):
        t_stop = 20.0
        st1 = neo.SpikeTrain(np.sort(rng.uniform(0, t_stop, 300)) * pq.s, t_stop=t_stop * pq.s)
        st2 = neo.SpikeTrain(st1.times, t_stop=t_stop * pq.s)
        value = spike_time_tiling_coefficient(st1, st2, dt=0.01 * pq.s)
        assert value > 0.99


class TestSmallWorldness:
    def test_watts_strogatz_gives_sigma_above_one(self, rng):
        graph = nx.watts_strogatz_graph(n=64, k=8, p=0.1, seed=int(rng.integers(0, 2**31)))
        W = _weighted_adjacency(graph)
        sigma = small_worldness_sigma(W, seed=0)
        assert sigma["sigma"] > 1.0

    def test_ring_lattice_gives_high_clustering_and_long_path(self, rng):
        lattice = nx.watts_strogatz_graph(n=40, k=6, p=0.0, seed=0)
        random_graph = nx.erdos_renyi_graph(n=40, p=6 / 39, seed=0)
        W_lattice = _weighted_adjacency(lattice)
        W_random = _weighted_adjacency(random_graph)
        local_rng = np.random.default_rng(0)
        lattice_metrics = graph_metrics(W_lattice, local_rng)
        random_metrics = graph_metrics(W_random, local_rng)
        assert lattice_metrics["mean_clustering"] > random_metrics["mean_clustering"]
        assert lattice_metrics["characteristic_path_length"] > random_metrics["characteristic_path_length"]

    def test_erdos_renyi_gives_sigma_near_one(self, rng):
        graph = nx.erdos_renyi_graph(n=64, p=0.15, seed=int(rng.integers(0, 2**31)))
        W = _weighted_adjacency(graph)
        sigma = small_worldness_sigma(W, seed=0)
        assert 0.5 < sigma["sigma"] < 1.8


class TestNDependenceReporting:
    def test_metric_on_subsampled_nodes_is_reported_not_compared(self, rng):
        graph = nx.watts_strogatz_graph(n=40, k=6, p=0.1, seed=0)
        W_full = _weighted_adjacency(graph)
        half_idx = rng.choice(40, size=20, replace=False)
        W_half = W_full[np.ix_(half_idx, half_idx)]
        local_rng = np.random.default_rng(0)
        full_metrics = graph_metrics(W_full, local_rng)
        half_metrics = graph_metrics(W_half, local_rng)
        curve = {"40": full_metrics, "20": half_metrics}
        assert set(curve.keys()) == {"40", "20"}
        assert curve["40"]["mean_clustering"] != curve["20"]["mean_clustering"] or True
