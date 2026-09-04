"""Tests for scripts/run_band_versus_sensor_decomposition.py's branch
resolution: a comparison must never resolve band_restriction_is_expensive /
sensor_is_the_barrier / their favourable counterparts from a missing or
absent value -- only from a measured, paired factor-analysis noise fraction
in both cells. Fewer than 4 paired sessions (or no sessions in common at
all) must return not_computable, never a fabricated branch."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_band_versus_sensor_decomposition as bvs  # noqa: E402


class TestDegeneracyReasonCatchesADeadRecordingBeforeTheExpensiveFit:
    """dandi_000574 sub-08_ses-05's entire recording (both depth and scalp
    series) is exactly zero on every sample -- a dead acquisition channel,
    verified directly against the raw NWB data, not a hypothetical. Fitting
    FactorAnalysis to it is not merely wasted work, it has no graceful
    degeneracy floor the way the census's nugget-fraction estimator does and
    can spend disproportionate wall-clock on it. This must be caught before
    that fit is attempted."""

    def test_all_zero_tensor_is_flagged_degenerate(self):
        tensor = np.zeros((40, 10, 30))
        assert bvs._degeneracy_reason(tensor) is not None

    def test_tensor_with_real_variance_is_not_flagged(self):
        rng = np.random.default_rng(0)
        tensor = rng.standard_normal((40, 10, 30))
        assert bvs._degeneracy_reason(tensor) is None

    def test_a_single_dead_channel_among_healthy_ones_is_not_flagged(self):
        # only a WHOLLY dead tensor should be skipped -- one flat channel
        # among many real ones is still fittable data, not a dead recording.
        rng = np.random.default_rng(1)
        tensor = rng.standard_normal((40, 10, 30))
        tensor[:, 3, :] = 0.0
        assert bvs._degeneracy_reason(tensor) is None


def _session(noise_fraction: float | None, status: str = "fitted") -> dict:
    return {
        "dimensionality": {
            "factor_analysis": {"status": status, "observation_noise_variance_fraction": noise_fraction},
        },
    }


class TestExtractNoiseFractionHandlesMissingValues:
    def test_none_session_returns_none(self):
        assert bvs._extract_noise_fraction(None) is None

    def test_degenerate_factor_model_returns_none_not_a_fabricated_number(self):
        session = _session(None, status="factor_model_did_not_converge_or_degenerate")
        assert bvs._extract_noise_fraction(session) is None

    def test_missing_dimensionality_key_returns_none_rather_than_raising(self):
        assert bvs._extract_noise_fraction({}) is None

    def test_fitted_session_returns_its_own_value(self):
        assert bvs._extract_noise_fraction(_session(0.42)) == 0.42


class TestResolveComparisonNeverResolvesABranchFromAnAbsentValue:
    def test_no_sessions_in_common_is_not_computable_and_carries_no_branch(self):
        result = bvs.resolve_comparison(
            {("sub-01", "ses-01"): _session(0.5)}, {("sub-02", "ses-01"): _session(0.5)},
            np.random.default_rng(0), "costs_little", "is_expensive", "no_resolvable",
        )
        assert result["status"] == "not_computable"
        assert "branch" not in result, "a not_computable result must never carry a branch key"

    def test_fewer_than_four_paired_sessions_is_not_computable(self):
        keys = [("sub-01", "ses-01"), ("sub-02", "ses-01"), ("sub-03", "ses-01")]
        sessions_a = {k: _session(0.5) for k in keys}
        sessions_b = {k: _session(0.3) for k in keys}
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(0),
                                         "costs_little", "is_expensive", "no_resolvable")
        assert result["status"] == "not_computable"
        assert result["n_pairs"] == 3
        assert "branch" not in result

    def test_a_session_present_in_only_one_cell_is_excluded_from_pairing_not_treated_as_zero(self):
        sessions_a = {
            ("sub-01", "ses-01"): _session(0.9), ("sub-02", "ses-01"): _session(0.9),
            ("sub-03", "ses-01"): _session(0.9), ("sub-04", "ses-01"): _session(0.9),
            ("sub-05", "ses-01"): _session(0.9),  # only in A -- must not be paired against a fabricated 0
        }
        sessions_b = {
            ("sub-01", "ses-01"): _session(0.1), ("sub-02", "ses-01"): _session(0.1),
            ("sub-03", "ses-01"): _session(0.1), ("sub-04", "ses-01"): _session(0.1),
        }
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(0),
                                         "costs_little", "is_expensive", "no_resolvable")
        assert result["status"] == "fitted"
        assert result["n_pairs"] == 4, "the unpaired 5th session must not be counted"

    def test_degenerate_factor_model_sessions_are_excluded_from_pairing(self):
        keys = [("sub-0%d" % i, "ses-01") for i in range(1, 6)]
        sessions_a = {k: _session(0.5) for k in keys}
        sessions_b = {k: _session(0.5) for k in keys[:3]}
        sessions_b[keys[3]] = _session(None, status="factor_model_did_not_converge_or_degenerate")
        sessions_b[keys[4]] = _session(None, status="factor_model_did_not_converge_or_degenerate")
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(0),
                                         "costs_little", "is_expensive", "no_resolvable")
        assert result["status"] == "not_computable"
        assert result["n_pairs"] == 3


class TestResolveComparisonBranchesAreDirectionAware:
    def _keys(self, n):
        return [("sub-%02d" % i, "ses-01") for i in range(n)]

    def test_interest_at_or_below_reference_is_costs_little_branch(self):
        keys = self._keys(8)
        sessions_a = {k: _session(0.2) for k in keys}
        sessions_b = {k: _session(0.6) for k in keys}
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(1),
                                         "costs_little", "is_expensive", "no_resolvable")
        assert result["status"] == "fitted"
        assert result["branch"] == "costs_little"
        assert result["median_difference_interest_minus_reference"] <= 0

    def test_interest_materially_above_reference_is_expensive_branch(self):
        keys = self._keys(10)
        rng = np.random.default_rng(2)
        sessions_a = {k: _session(0.85 + 0.01 * rng.standard_normal()) for k in keys}
        sessions_b = {k: _session(0.15 + 0.01 * rng.standard_normal()) for k in keys}
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(3),
                                         "costs_little", "is_expensive", "no_resolvable")
        assert result["status"] == "fitted"
        assert result["branch"] == "is_expensive"
        assert result["p_value"] < 0.05

    def test_every_null_branch_ships_its_minimum_detectable_difference(self):
        # a tiny, noisy positive difference -- not significant, must land in
        # the bounded null branch and carry an MDD, never a bare pass/fail.
        keys = self._keys(5)
        rng = np.random.default_rng(4)
        sessions_a = {k: _session(0.501 + 0.3 * rng.standard_normal()) for k in keys}
        sessions_b = {k: _session(0.500 + 0.3 * rng.standard_normal()) for k in keys}
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(5),
                                         "costs_little", "is_expensive", "no_resolvable")
        if result["branch"] == "no_resolvable":
            assert result["minimum_detectable_paired_difference_80pct_power"]["status"] == "computed"
            assert result["minimum_detectable_paired_difference_80pct_power"]["mdd"] > 0

    def test_fitted_result_reports_both_arm_values_and_their_difference(self):
        # rule: a difference-of-two-quantities result must carry both
        # quantities, not only the difference.
        keys = self._keys(6)
        sessions_a = {k: _session(0.7) for k in keys}
        sessions_b = {k: _session(0.3) for k in keys}
        result = bvs.resolve_comparison(sessions_a, sessions_b, np.random.default_rng(6),
                                         "costs_little", "is_expensive", "no_resolvable")
        assert result["r_obs_median_value_of_interest"] == pytest.approx(0.7)
        assert result["r_obs_median_reference_value"] == pytest.approx(0.3)
        assert result["median_difference_interest_minus_reference"] == pytest.approx(0.4)
