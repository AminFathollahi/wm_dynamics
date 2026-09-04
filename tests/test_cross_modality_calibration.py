import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_cross_modality_calibration.py"
SPEC = importlib.util.spec_from_file_location("run_cross_modality_calibration", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_region_of_buckets_mtl_and_cortical_and_excludes_unlabeled():
    assert MODULE._region_of("Hipp, Left Hippocampus rHipp, rostral hippocampus") == "mtl"
    assert MODULE._region_of("Amyg, Left Amygdala mAmyg, medial amygdala") == "mtl"
    assert MODULE._region_of("STG, Left Superior Temporal Gyrus A22r") == "cortical"
    assert MODULE._region_of("no_label_found") is None


def test_aggregate_sessions_averages_only_identifiable_sessions():
    session_fits = {
        "ses-01": {"status": "identifiable", "lambda_rate": 4.0},
        "ses-02": {"status": "not_identifiable", "lambda_rate": 99.0},
        "ses-03": {"status": "identifiable", "lambda_rate": 6.0},
    }
    result = MODULE._aggregate_sessions(session_fits)
    assert result["status"] == "identifiable"
    assert result["lambda_rate"] == 5.0
    assert result["n_identifiable_sessions"] == 2
    assert result["n_sessions"] == 3


def test_aggregate_sessions_reports_not_identifiable_when_no_session_identifiable():
    session_fits = {"ses-01": {"status": "not_identifiable", "lambda_rate": 1.0}}
    result = MODULE._aggregate_sessions(session_fits)
    assert result["status"] == "not_identifiable"
    assert result["lambda_rate"] is None


def test_fit_lambda_requires_at_least_six_complete_trials():
    trial_bins = np.random.default_rng(0).normal(size=(3, 20))
    result = MODULE._fit_lambda(trial_bins, np.zeros(3, dtype=int))
    assert result["status"] == "not_identifiable"
    assert "fewer than 6" in result["reason"]


def test_spearman_rank_correlation_requires_at_least_four_patients():
    result = MODULE._spearman_rank_correlation([(1.0, 2.0), (2.0, 3.0)])
    assert result["status"] == "non_identified"


def test_spearman_rank_correlation_recovers_perfect_monotonic_agreement():
    pairs = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0), (5.0, 50.0)]
    result = MODULE._spearman_rank_correlation(pairs)
    assert result["status"] == "estimable"
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["excludes_zero"] is True
