"""Tests for src/small_worldness.py -- this repository's single Watts-
Strogatz sigma definition, adopted from the sibling RNN codebase's
network-analysis module."""

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from small_worldness import (
    MIN_COMPONENT_NODES,
    definition_summary,
    degree_assortativity,
    thresholded_largest_component_graph,
    watts_strogatz_sigma,
)


def _weighted_adjacency(graph: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(graph, weight=None)


class TestThresholdedLargestComponentGraph:
    def test_below_floor_component_returns_none(self):
        # 5 isolated nodes plus one tiny 4-node edge cluster: no component
        # reaches MIN_COMPONENT_NODES.
        graph = nx.Graph()
        graph.add_nodes_from(range(9))
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        W = _weighted_adjacency(graph)
        assert thresholded_largest_component_graph(W, seed=0) is None

    def test_dense_graph_returns_a_component_at_least_the_floor(self):
        graph = nx.erdos_renyi_graph(n=40, p=0.3, seed=0)
        W = _weighted_adjacency(graph)
        result = thresholded_largest_component_graph(W, seed=0)
        assert result is not None
        assert result.number_of_nodes() >= MIN_COMPONENT_NODES

    def test_subsamples_to_max_nodes(self):
        graph = nx.erdos_renyi_graph(n=200, p=0.2, seed=0)
        W = _weighted_adjacency(graph)
        result = thresholded_largest_component_graph(W, seed=0, max_nodes=64)
        assert result is not None
        assert result.number_of_nodes() <= 64


class TestWattsStrogatzSigma:
    def test_watts_strogatz_graph_gives_sigma_above_one(self):
        graph = nx.watts_strogatz_graph(n=64, k=8, p=0.1, seed=1)
        W = _weighted_adjacency(graph)
        sigma = watts_strogatz_sigma(W, seed=0)
        assert sigma is not None
        assert sigma > 1.0

    def test_too_small_graph_returns_none(self):
        graph = nx.erdos_renyi_graph(n=8, p=0.5, seed=0)
        W = _weighted_adjacency(graph)
        assert watts_strogatz_sigma(W, seed=0) is None

    def test_deterministic_given_seed(self):
        graph = nx.watts_strogatz_graph(n=64, k=8, p=0.1, seed=2)
        W = _weighted_adjacency(graph)
        first = watts_strogatz_sigma(W, seed=7)
        second = watts_strogatz_sigma(W, seed=7)
        assert first == second


class TestDegreeAssortativity:
    def test_returns_a_finite_value_on_a_dense_graph(self):
        graph = nx.erdos_renyi_graph(n=40, p=0.3, seed=0)
        W = _weighted_adjacency(graph)
        r = degree_assortativity(W, seed=0)
        assert r is not None
        assert np.isfinite(r)

    def test_too_small_graph_returns_none(self):
        graph = nx.erdos_renyi_graph(n=8, p=0.5, seed=0)
        W = _weighted_adjacency(graph)
        assert degree_assortativity(W, seed=0) is None


class TestDefinitionSummary:
    def test_names_all_three_settings_categories(self):
        summary = definition_summary()
        assert "sparsification" in summary
        assert "null_construction" in summary
        assert "resolution" in summary
        assert "10%" in summary["sparsification"]
