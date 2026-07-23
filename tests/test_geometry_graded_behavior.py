"""Self-check for scripts/run_geometry_graded_behavior.py's LME wrapper.

Not a general statistics.py test (that's tests/test_statistics.py) -- this
checks the specific pattern this script relies on: recovering a planted
drift -> RT slope from multi-subject synthetic data via
statistics.linear_mixed_effects_test plus the bootstrap CI helper.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_geometry_graded_behavior import _lme_report, bootstrap_ci_beta


def _synthetic_drift_rt(true_beta: float, n_subjects: int = 6, n_trials_per_subj: int = 80,
                        rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(0)
    drift, rt, subj = [], [], []
    for s in range(n_subjects):
        subj_intercept = rng.normal(0, 2.0)  # random-effect nuisance the LME must partial out
        d = rng.normal(1.0, 0.5, n_trials_per_subj)
        y = subj_intercept + true_beta * d + rng.normal(0, 0.3, n_trials_per_subj)
        drift.append(d); rt.append(y); subj.append([f"s{s}"] * n_trials_per_subj)
    return np.concatenate(drift), np.concatenate(rt), np.concatenate(subj)


def test_recovers_positive_slope():
    drift, rt, subj = _synthetic_drift_rt(true_beta=1.5, rng=np.random.default_rng(1))
    res = _lme_report(drift, rt, subj, "test_pos")
    assert "beta" in res
    assert res["beta"] == pytest.approx(1.5, abs=0.3)
    assert res["p_value"] < 0.01
    assert res["ci_lo"] < res["beta"] < res["ci_hi"]


def test_recovers_negative_slope():
    drift, rt, subj = _synthetic_drift_rt(true_beta=-1.2, rng=np.random.default_rng(2))
    res = _lme_report(drift, rt, subj, "test_neg")
    assert res["beta"] == pytest.approx(-1.2, abs=0.3)
    assert res["p_value"] < 0.01


def test_null_slope_not_significant():
    drift, rt, subj = _synthetic_drift_rt(true_beta=0.0, rng=np.random.default_rng(3))
    res = _lme_report(drift, rt, subj, "test_null")
    assert abs(res["beta"]) < 0.3
    assert res["p_value"] > 0.05


def test_underpowered_flag():
    drift, rt, subj = _synthetic_drift_rt(true_beta=1.0, n_subjects=1, n_trials_per_subj=5,
                                          rng=np.random.default_rng(4))
    res = _lme_report(drift, rt, subj, "test_underpowered")
    assert res["underpowered"] is True


if __name__ == "__main__":
    test_recovers_positive_slope()
    test_recovers_negative_slope()
    test_null_slope_not_significant()
    test_underpowered_flag()
    print("All geometry_graded_behavior self-checks passed.")
