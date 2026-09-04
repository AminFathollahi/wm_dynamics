"""Regression tests for scripts/build_structure_control_observables.py.
Pure-arithmetic checks on synthetic (lambda,
diffusion) pairs and synthetic region-stratified artifacts -- no real data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_structure_control_observables as mod  # noqa: E402


class TestBootstrapSummary:
    def test_non_identified_below_two_values(self):
        rng = np.random.default_rng(0)
        assert mod.bootstrap_summary([1.0], rng)["status"] == "non_identified"
        assert mod.bootstrap_summary([], rng)["status"] == "non_identified"

    def test_estimable_reports_mean_and_ci(self):
        rng = np.random.default_rng(0)
        result = mod.bootstrap_summary([1.0, 2.0, 3.0, 4.0, 5.0], rng)
        assert result["status"] == "estimable"
        assert result["mean"] == pytest.approx(3.0)
        assert result["n_sessions"] == 5
        assert result["bootstrap_ci95"][0] <= result["mean"] <= result["bootstrap_ci95"][1]


class TestDisplacementSnrGrid:
    def test_snr_decreases_with_larger_stationary_variance(self):
        small_variance = mod.displacement_snr_grid(1.0, [1.0])["1.0"]
        large_variance = mod.displacement_snr_grid(4.0, [1.0])["1.0"]
        assert small_variance > large_variance
        assert small_variance == pytest.approx(1.0)
        assert large_variance == pytest.approx(0.5)


class TestBuildStructureControlObservables:
    def test_bandwidth_is_inverse_lambda_in_ms_and_stationary_variance_is_d_over_lambda(self):
        rng = np.random.default_rng(0)
        # lambda=2 s^-1, D=4 -> bandwidth 500 ms, stationary variance 2.0
        pairs = [(2.0, 4.0), (2.0, 4.0), (2.0, 4.0)]
        result = mod.build_structure_control_observables(pairs, rng)
        assert result["status"] == "estimable"
        assert result["control_bandwidth_ms"]["mean"] == pytest.approx(500.0)
        assert result["stationary_variance_state_units_sq"]["mean"] == pytest.approx(2.0)
        assert result["displacement_snr_by_delta"]["1.0"] == pytest.approx(1.0 / np.sqrt(2.0))

    def test_non_identified_below_two_sessions(self):
        rng = np.random.default_rng(0)
        result = mod.build_structure_control_observables([(2.0, 4.0)], rng)
        assert result["status"] == "non_identified"


class TestExtractPairsIdentifiabilityFilter:
    def test_000469_only_identifiable_folds_contribute(self):
        artifact = {
            "regions": {
                "hippocampus": {
                    "sessions": {
                        "sub-1": {
                            "status": "complete",
                            "folds": [
                                {"state_space": {"status": "identifiable", "lambda_rate": 1.0, "diffusion": 2.0}},
                                {"state_space": {"status": "not_identifiable", "lambda_rate": None, "diffusion": None}},
                            ],
                        },
                        "sub-2": {"status": "excluded"},
                    }
                }
            }
        }
        pairs = mod.extract_000469_pairs(artifact, "hippocampus")
        assert pairs == [(1.0, 2.0)]

    def test_001187_000673_reads_content_axis_battery_block(self):
        artifact = {
            "content_axis_battery": {
                "regions": {
                    "amygdala": {
                        "sessions": {
                            "p1_s1": {
                                "patient": "p1",
                                "content_axis_fit": {
                                    "status": "complete",
                                    "folds": [
                                        {"state_space": {"status": "identifiable", "lambda_rate": 3.0, "diffusion": 6.0}},
                                    ],
                                },
                            },
                        }
                    }
                }
            }
        }
        pairs = mod.extract_001187_000673_pairs(artifact, "amygdala")
        assert pairs == [(3.0, 6.0)]
