"""Tests for src/attractor_analysis.py: persistent homology on known point-cloud
topologies, and the fixed-point/Jacobian classifier on known dynamical systems."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from attractor_analysis import (
    build_transition_pairs,
    classify_fixed_point,
    find_fixed_points,
    persistent_homology,
    simulate_line_attractor,
    simulate_ring,
    simulate_two_well,
    summarize_fixed_point_pattern,
)
from geometry import latent_trajectories


class TestPersistentHomology:
    def test_two_well_gives_beta0_two(self, rng):
        # Two clearly separated blobs, the textbook case a Betti-0 estimator
        # must recover; the noisy dynamical two-well simulator (used for the
        # fixed-point calibration below) is not used here because its
        # trial-to-trial critical-slowing-down transients bridge the two
        # basins with a thin trickle of points, which defeats single-linkage
        # separation regardless of estimator quality (see
        # src/attractor_analysis.py's betti_numbers_at_largest_gap docstring).
        blob_a = rng.normal(loc=[-4.0, 0.0], scale=0.5, size=(70, 2))
        blob_b = rng.normal(loc=[4.0, 0.0], scale=0.5, size=(70, 2))
        cloud = np.vstack([blob_a, blob_b])
        result = persistent_homology(cloud, maxdim=1)
        assert result["betti"][0]["betti"] == 2

    def test_ring_gives_beta1_one(self, rng):
        cloud = simulate_ring(n_points=200, rng=rng)
        result = persistent_homology(cloud, maxdim=1)
        assert result["betti"][1]["betti"] == 1

    def test_gaussian_cloud_gives_beta0_one_beta1_zero(self, rng):
        cloud = rng.standard_normal((200, 3))
        result = persistent_homology(cloud, maxdim=1)
        assert result["betti"][0]["betti"] == 1
        assert result["betti"][1]["betti"] == 0


class TestFixedPointClassifier:
    def test_line_attractor_labeled_line_attractor(self, rng):
        observed = simulate_line_attractor(n_trials=137, n_bins=20, n_units=41, dt=0.1, rng=rng)
        latent, _, _ = latent_trajectories(observed, n_components=2)
        x_t, x_tp1 = build_transition_pairs(latent)
        bounds = (x_t.min(axis=0), x_t.max(axis=0))
        points = find_fixed_points(x_t, x_tp1, n_inits=80, rng=rng, bounds=bounds)
        assert len(points) > 0
        assert summarize_fixed_point_pattern(points) == "line_attractor_dominant"

    def test_two_well_labeled_two_stable_plus_saddle(self, rng):
        observed = simulate_two_well(n_trials=137, n_bins=20, n_units=41, dt=0.1, rng=rng)
        latent, _, _ = latent_trajectories(observed, n_components=2)
        x_t, x_tp1 = build_transition_pairs(latent)
        bounds = (x_t.min(axis=0), x_t.max(axis=0))
        points = find_fixed_points(x_t, x_tp1, n_inits=80, rng=rng, bounds=bounds)
        labels = [p["classification"] for p in points]
        assert labels.count("stable_point") >= 2
        assert labels.count("saddle") >= 1
        assert summarize_fixed_point_pattern(points) == "two_stable_plus_saddle"

    def test_classify_stable_point(self):
        assert classify_fixed_point(np.array([0.5 + 0j, 0.3 + 0j])) == "stable_point"

    def test_classify_line_attractor(self):
        assert classify_fixed_point(np.array([1.0 + 0j, 0.3 + 0j])) == "line_attractor"

    def test_classify_saddle(self):
        assert classify_fixed_point(np.array([1.5 + 0j, 0.3 + 0j])) == "saddle"

    def test_classify_limit_cycle(self):
        assert classify_fixed_point(np.array([1.0 + 0.05j, 1.0 - 0.05j])) == "limit_cycle"
