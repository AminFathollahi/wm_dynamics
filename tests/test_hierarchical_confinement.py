"""Regression test for direct partial pooling of fold likelihoods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hierarchical_confinement_000469 import _marginal_fit


def test_marginal_fit_recovers_positive_group_rate_without_filtering_folds():
    rng = np.random.default_rng(17)
    groups = []
    for patient_rate in (0.7, 0.9, 1.1, 1.3, 1.5, 0.8, 1.2, 1.0):
        errors = np.linspace(0.18, 1.4, 5)
        values = patient_rate + rng.normal(scale=errors)
        groups.append((values, errors))
    result = _marginal_fit(groups, positive=True)
    assert result["status"] == "complete"
    assert 0.65 < result["group_mean"] < 1.4
    assert result["between_patient_sd"] >= 0.0
