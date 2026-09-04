"""Tests for scripts/run_state_orthogonality_census.py's bounded_exclusion():
the minimum-detectable-effect-at-80%-power computation itself is covered by
tests/test_statistics.py, so what needs guarding here is the classification
built on top of it -- a null must land in the meaningful/large-only/excludes
bucket its own minimum detectable effect actually falls in, on the correct
side of each pre-declared threshold, and a test whose statistic is not a
correlation must never be classified on the correlation scale at all."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_state_orthogonality_census as census  # noqa: E402


def _alternating(n: int) -> np.ndarray:
    """Deterministic (no RNG) per-session values with a known, reproducible
    sample spread: +-0.5 alternating has sample std (ddof=1) close to 0.5,
    converging as n grows, with no dependence on a random seed."""
    return np.array([0.5, -0.5] * (n // 2))


class TestBoundedExclusionClassifiesOnItsOwnMinimumDetectableEffect:
    def test_large_n_lands_in_meaningful_scale_bucket(self):
        # n=200 -> mdd ~= 2.8016 * 0.5 / sqrt(200) ~= 0.099, well under 0.14
        result = census.bounded_exclusion(
            "name", "a candidate", "an observable", _alternating(200),
            0.01, -0.05, 0.05, 0.9, "source.json")
        assert result["constraining_power"] == census.CONSTRAINS_AT_MEANINGFUL_SCALE
        mdd_r = result["minimum_detectable_effect_correlation_units"]
        assert mdd_r <= census.MEANINGFUL_EFFECT_THRESHOLD_R_UNITS
        assert result["sort_key_minimum_detectable_effect_correlation_units"] == mdd_r
        assert f"{mdd_r:.3f}" in result["bound_in_plain_language"]

    def test_moderate_n_lands_in_large_only_bucket(self):
        # n=50 -> mdd ~= 2.8016 * 0.505 / sqrt(50) ~= 0.200, strictly between the two thresholds
        result = census.bounded_exclusion(
            "name", "a candidate", "an observable", _alternating(50),
            0.01, -0.05, 0.05, 0.9, "source.json")
        assert result["constraining_power"] == census.CONSTRAINS_LARGE_ONLY
        mdd_r = result["minimum_detectable_effect_correlation_units"]
        assert census.MEANINGFUL_EFFECT_THRESHOLD_R_UNITS < mdd_r <= census.LARGE_ASSOCIATION_THRESHOLD_R_UNITS
        assert "leaves the whole range up to" in result["bound_in_plain_language"]

    def test_small_n_lands_in_excludes_nothing_bucket(self):
        # n=10 -> mdd ~= 2.8016 * 0.527 / sqrt(10) ~= 0.467, above the large-association threshold
        result = census.bounded_exclusion(
            "name", "a candidate", "an observable", _alternating(10),
            0.01, -0.05, 0.05, 0.9, "source.json")
        assert result["constraining_power"] == census.EXCLUDES_NOTHING
        assert result["minimum_detectable_effect_correlation_units"] > census.LARGE_ASSOCIATION_THRESHOLD_R_UNITS
        assert "must not be cited as evidence of orthogonality" in result["bound_in_plain_language"]

    def test_fewer_than_two_values_is_not_computable_and_sorts_last(self):
        result = census.bounded_exclusion(
            "name", "a candidate", "an observable", [0.1],
            0.1, None, None, None, "source.json")
        assert result["constraining_power"] == "not_computable"
        assert result["sort_key_minimum_detectable_effect_correlation_units"] is None

    def test_non_correlation_units_are_bounded_only_in_their_own_units(self):
        result = census.bounded_exclusion(
            "name", "a candidate", "an observable", _alternating(50), 0.01, None, None, 0.5,
            "source.json", units="correlation per second of position within the epoch")
        assert result["constraining_power"] == census.NOT_ON_CORRELATION_SCALE
        assert "minimum_detectable_effect_correlation_units" not in result
        assert result["sort_key_minimum_detectable_effect_correlation_units"] is None
        assert "does not convert to a shared-variance fraction" in result["bound_in_plain_language"]

    def test_scale_factor_scales_the_correlation_units_bound_linearly(self):
        values = _alternating(80)
        unscaled = census.bounded_exclusion(
            "name", "a candidate", "an observable", values, 0.01, None, None, 0.5,
            "source.json", scale_factor=1.0)
        doubled = census.bounded_exclusion(
            "name", "a candidate", "an observable", values, 0.01, None, None, 0.5,
            "source.json", scale_factor=2.0)
        assert np.isclose(
            doubled["minimum_detectable_effect_correlation_units"],
            2.0 * unscaled["minimum_detectable_effect_correlation_units"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
