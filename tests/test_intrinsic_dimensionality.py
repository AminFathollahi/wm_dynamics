"""Tests for the intrinsic-dimensionality estimators in src/geometry.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from geometry import levina_bickel_mle_dimension, participation_ratio, twonn_dimension

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_intrinsic_dimensionality import compute_estimators, n_dependence_curve


class TestPlantedSubspaceRecovery:
    def test_twonn_and_mle_recover_planted_dimension(self, rng):
        d_true = 3
        loading = rng.normal(size=(d_true, 20))
        latent = rng.normal(size=(1000, d_true))
        cloud = latent @ loading + rng.normal(scale=0.01, size=(1000, 20))
        assert abs(twonn_dimension(cloud) - d_true) < 1.0
        assert abs(levina_bickel_mle_dimension(cloud) - d_true) < 1.0

    def test_isotropic_noise_returns_ambient_dimension(self, rng):
        ambient = 10
        cloud = rng.normal(size=(1000, ambient))
        assert abs(twonn_dimension(cloud) - ambient) < 3.0
        assert abs(levina_bickel_mle_dimension(cloud) - ambient) < 3.0


class TestNDependenceCurve:
    def test_curve_is_monotone_non_decreasing_on_synthetic_data(self, rng):
        n_trials, n_units, n_bins = 60, 40, 10
        psth = rng.normal(size=(n_trials, n_units, n_bins)) + 5.0
        curve = n_dependence_curve(psth, rng)
        levels = sorted(int(k) for k in curve.keys())
        for name in ("twonn", "levina_bickel_mle", "participation_ratio"):
            values = [curve[str(level)][name] for level in levels]
            finite = [v for v in values if np.isfinite(v)]
            assert all(b >= a - 0.5 for a, b in zip(finite, finite[1:])), (name, values)
