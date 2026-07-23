"""Self-check for scripts/run_targeting_heterogeneity.py's moderator
regression: recovers a planted covariate -> effect relationship on synthetic
data, and correctly flags an underpowered (n<8) sample.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_targeting_heterogeneity import univariate_moderator_test, MIN_SESSIONS


def test_recovers_planted_positive_slope():
    rng = np.random.default_rng(0)
    n = 40
    covariate = rng.normal(0, 1, n)
    effect = 0.6 + 0.15 * covariate + rng.normal(0, 0.05, n)
    res = univariate_moderator_test(effect, covariate, rng=np.random.default_rng(1))
    assert not res.get("underpowered")
    assert res["beta"] > 0
    assert res["beta"] == pytest.approx(0.15, abs=0.05)
    assert res["p_value"] < 0.01
    assert res["ci_lo"] < res["beta"] < res["ci_hi"]


def test_null_covariate_not_significant():
    rng = np.random.default_rng(2)
    n = 40
    covariate = rng.normal(0, 1, n)
    effect = 0.6 + rng.normal(0, 0.05, n)
    res = univariate_moderator_test(effect, covariate, rng=np.random.default_rng(3))
    assert abs(res["beta"]) < 0.05
    assert res["p_value"] > 0.05


def test_underpowered_below_floor():
    rng = np.random.default_rng(4)
    n = MIN_SESSIONS - 1
    covariate = rng.normal(0, 1, n)
    effect = 0.6 + 0.15 * covariate
    res = univariate_moderator_test(effect, covariate, rng=np.random.default_rng(5))
    assert res["underpowered"] is True


if __name__ == "__main__":
    test_recovers_planted_positive_slope()
    test_null_covariate_not_significant()
    test_underpowered_below_floor()
    print("All targeting_heterogeneity self-checks passed.")
