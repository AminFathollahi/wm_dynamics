"""Tests for scripts/run_latent_model_observation_noise_comparison.py: cell
scoping must only include what the census admits at comfortable margin AND
what a loader in this repo can actually build a tensor for, the paired
verdict branch must correctly detect a sign flip, and a degenerate factor
model must fall through as not_computable rather than raising or being
silently dropped."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_latent_model_observation_noise_comparison as mod  # noqa: E402
from observability import _leading_latent_projection_via_factor_analysis  # noqa: E402


def _admitted_comfortable(status="admitted", band="comfortable"):
    return {"status": status, "margin": {"band": band}}


class TestTargetCellScoping:
    def test_cell_needs_both_observables_at_comfortable_margin(self):
        matrix = {
            "both_comfortable": {
                "dataset": "d", "modality": "single_unit", "structure": "s", "bin_ms": 100,
                "observables": {name: _admitted_comfortable() for name in mod.TARGET_OBSERVABLES},
            },
            "only_one_comfortable": {
                "dataset": "d", "modality": "single_unit", "structure": "t", "bin_ms": 100,
                "observables": {
                    mod.TARGET_OBSERVABLES[0]: _admitted_comfortable(),
                    mod.TARGET_OBSERVABLES[1]: _admitted_comfortable(band="adequate"),
                },
            },
            "excluded": {
                "dataset": "d", "modality": "single_unit", "structure": "u", "bin_ms": 100,
                "observables": {name: {"status": "excluded"} for name in mod.TARGET_OBSERVABLES},
            },
        }
        cells = mod.target_cells(matrix)
        assert [c["key"] for c in cells] == ["both_comfortable"]

    def test_lfp_grain_of_dandi_000574_is_attempted(self):
        cell = {"dataset": "dandi_000574", "modality": "depth_contact_lfp", "structure": "depth_contact_lfp", "bin_ms": 100}
        assert mod._cell_is_attempted(cell) is True

    def test_lfp_grain_of_a_dataset_without_an_lfp_builder_is_not_attempted(self):
        cell = {"dataset": "dandi_000469", "modality": "depth_contact_lfp", "structure": "depth_contact_lfp", "bin_ms": 100}
        assert mod._cell_is_attempted(cell) is False

    def test_single_unit_grain_of_a_loader_covered_dataset_is_attempted(self):
        cell = {"dataset": "dandi_000574", "modality": "single_unit", "structure": "hippocampus", "bin_ms": 100}
        assert mod._cell_is_attempted(cell) is True

    def test_alm_only_attempted_at_structure_pooled(self):
        assert mod._cell_is_attempted({"dataset": "inagaki_alm5", "modality": "single_unit", "structure": "pooled", "bin_ms": 100}) is True
        assert mod._cell_is_attempted({"dataset": "inagaki_alm5", "modality": "single_unit", "structure": "left_alm", "bin_ms": 100}) is False

    def test_dataset_without_a_loader_is_not_attempted(self):
        cell = {"dataset": "panichello_2024", "modality": "single_unit", "structure": "pooled", "bin_ms": 100}
        assert mod._cell_is_attempted(cell) is False


class TestVerdictBranchResolution:
    def _summary(self, pr_pca=10.0, pr_fa=4.0, pr_p=0.01, level_pca=0.05, level_fa=0.05, level_p=0.5,
                 slope_pca=0.0, slope_fa=0.0, slope_p=0.5):
        def _fitted(median_pca, median_fa, p):
            return {"status": "fitted", "median_pca": median_pca, "median_factor_analysis": median_fa, "p_value": p}
        return {
            "participation_ratio": _fitted(pr_pca, pr_fa, pr_p),
            "d_perm_level": _fitted(level_pca, level_fa, level_p),
            "d_perm_slope": _fitted(slope_pca, slope_fa, slope_p),
        }

    def test_no_significant_difference_is_conclusions_unchanged(self):
        branch = mod.resolve_cell_branch(self._summary(pr_p=0.9, level_p=0.9, slope_p=0.9), mod.predeclare_verdict_branches())
        assert branch == "conclusions_unchanged"

    def test_significant_but_same_sign_is_magnitude_moves(self):
        branch = mod.resolve_cell_branch(self._summary(pr_pca=10.0, pr_fa=4.0, pr_p=0.001), mod.predeclare_verdict_branches())
        assert branch == "magnitude_moves_but_every_verdict_holds"

    def test_significant_sign_flip_is_verdict_changed(self):
        branch = mod.resolve_cell_branch(
            self._summary(slope_pca=-0.01, slope_fa=0.01, slope_p=0.01), mod.predeclare_verdict_branches())
        assert branch == "verdict_changed"

    def test_all_not_computable_is_factor_model_degenerate(self):
        summary = {
            "participation_ratio": {"status": "not_computable"},
            "d_perm_level": {"status": "not_computable"},
            "d_perm_slope": {"status": "not_computable"},
        }
        assert mod.resolve_cell_branch(summary, mod.predeclare_verdict_branches()) == "factor_model_degenerate_for_this_cell"


class TestDegenerateFactorModelIsNotComputableNotRaised:
    def test_too_few_units_returns_none_not_an_exception(self):
        rng = np.random.default_rng(0)
        psth_fit = rng.normal(size=(10, 1, 8))  # a single unit: no factor model is fittable
        psth_apply = rng.normal(size=(4, 1, 8))
        result = _leading_latent_projection_via_factor_analysis(psth_fit, psth_apply)
        assert result is None

    def test_summarize_cell_with_fewer_than_four_pairs_is_not_computable(self):
        sessions = [
            {
                "dimensionality": {"pca": {"participation_ratio": 5.0}, "factor_analysis": {"status": "fitted", "participation_ratio": 2.0}},
                "persistence_contrast": {
                    "pca": {"status": "fitted", "level": 0.1, "slope_r_per_s": 0.0},
                    "factor_analysis": {"status": "fitted", "level": 0.05, "slope_r_per_s": 0.0},
                },
            }
            for _ in range(2)
        ]
        summary = mod.summarize_cell(sessions)
        assert summary["participation_ratio"]["status"] == "not_computable"
        assert summary["d_perm_level"]["status"] == "not_computable"


class TestDPermLevelAndSlope:
    def test_returns_none_when_null_has_fewer_than_three_replicates(self):
        observed = {"status": "fitted", "lags": {3: {"r_median": 0.1}, 4: {"r_median": 0.05}}}
        null = {"n_replicates_fitted": 2, "lags": {3: {"r_null_median": 0.0}, 4: {"r_null_median": 0.0}}}
        assert mod._d_perm_level_and_slope(observed, null, bin_width_s=0.1) is None

    def test_level_is_read_at_the_shortest_shared_lag(self):
        observed = {"status": "fitted", "lags": {3: {"r_median": 0.2}, 4: {"r_median": 0.1}, 5: {"r_median": 0.0}}}
        null = {"n_replicates_fitted": 5, "lags": {3: {"r_null_median": 0.05}, 4: {"r_null_median": 0.05}, 5: {"r_null_median": 0.05}}}
        result = mod._d_perm_level_and_slope(observed, null, bin_width_s=0.1)
        assert result["level_lag_bins"] == 3
        assert result["level"] == pytest.approx(0.15)


class TestParticipationRatioDimensionalityFinding:
    def _cell(self, pca, fa, p):
        return {"summary": {"participation_ratio": {"status": "fitted", "median_pca": pca, "median_factor_analysis": fa, "p_value": p}}}

    def test_all_drop_and_significant(self):
        cell_results = {f"c{i}": self._cell(10.0, 4.0, 0.001) for i in range(4)}
        result = mod.participation_ratio_dimensionality_finding(cell_results)
        assert result["status"] == "fitted"
        assert result["n_cells_fitted"] == 4
        assert result["n_cells_where_participation_ratio_drops_under_factor_analysis"] == 4
        assert result["n_cells_significant_at_p_less_than_0_05"] == 4
        assert result["median_of_per_cell_pca_over_factor_analysis_ratio"] == pytest.approx(2.5)

    def test_not_computable_when_no_fitted_cells(self):
        result = mod.participation_ratio_dimensionality_finding({"c0": {"summary": {"participation_ratio": {"status": "not_computable"}}}})
        assert result["status"] == "not_computable"

    def test_mixed_drop_and_rise_counted_separately(self):
        cell_results = {"drops": self._cell(10.0, 4.0, 0.001), "rises": self._cell(4.0, 10.0, 0.9)}
        result = mod.participation_ratio_dimensionality_finding(cell_results)
        assert result["n_cells_fitted"] == 2
        assert result["n_cells_where_participation_ratio_drops_under_factor_analysis"] == 1
        assert result["n_cells_significant_at_p_less_than_0_05"] == 1


class TestDPermSlopeSignFlipPowerBound:
    def _cell(self, pca, fa, p):
        return {"summary": {"d_perm_slope": {"status": "fitted", "median_pca": pca, "median_factor_analysis": fa, "p_value": p}}}

    def test_counts_sign_flips_and_splits_by_significance(self):
        cell_results = {
            "flip_ns": self._cell(-0.01, 0.02, 0.5),
            "flip_sig": self._cell(-0.01, 0.02, 0.01),
            "no_flip": self._cell(0.01, 0.02, 0.5),
        }
        result = mod.d_perm_slope_sign_flip_power_bound(cell_results)
        assert result["n_cells_fitted"] == 3
        assert result["n_cells_sign_flipped"] == 2
        assert result["n_sign_flips_significant_at_p_less_than_0_05"] == 1
        assert result["p_value_range_of_non_significant_sign_flips"] == {"min": pytest.approx(0.5), "max": pytest.approx(0.5)}

    def test_not_computable_when_no_fitted_cells(self):
        result = mod.d_perm_slope_sign_flip_power_bound({"c0": {"summary": {"d_perm_slope": {"status": "not_computable"}}}})
        assert result["status"] == "not_computable"


class TestObservationNoiseEstimatorOffset:
    def _cell(self, fa_fraction, census_status="fitted", nugget=None):
        census = {"status": census_status}
        if census_status == "fitted":
            census["median_nugget_fraction"] = nugget
        return {"observation_noise_term_comparison": {
            "factor_analysis_median_noise_variance_fraction": fa_fraction,
            "census_cross_validated_nugget_fraction": census,
        }}

    def test_direction_counts_and_one_directional_flag(self):
        cell_results = {"below1": self._cell(0.1, nugget=0.9), "below2": self._cell(0.2, nugget=0.8)}
        result = mod.observation_noise_estimator_offset(cell_results)
        assert result["n_comparable_pairs"] == 2
        assert result["pooled_all_pairs_including_at_floor"]["n_below"] == 2
        assert result["one_directional_off_floor"] is True

    def test_mixed_direction_is_not_one_directional(self):
        cell_results = {"below": self._cell(0.1, nugget=0.9), "above": self._cell(0.9, nugget=0.1)}
        result = mod.observation_noise_estimator_offset(cell_results)
        assert result["one_directional_off_floor"] is False
        assert result["pooled_all_pairs_including_at_floor"]["n_above"] == 1

    def test_cells_at_the_census_zero_floor_are_excluded_from_one_directional_off_floor(self):
        # Both floored cells necessarily read "above" (a positive factor-analysis fraction minus a
        # census value of exactly 0.0), which alone would make the pooled tally look one-directional;
        # the off-floor partition must not be fooled by that and must reflect only the two genuinely
        # mixed-direction off-floor cells.
        cell_results = {
            "floored1": self._cell(0.3, nugget=0.0),
            "floored2": self._cell(0.5, nugget=0.0),
            "off_floor_below": self._cell(0.1, nugget=0.9),
            "off_floor_above": self._cell(0.9, nugget=0.1),
        }
        result = mod.observation_noise_estimator_offset(cell_results)
        assert result["at_zero_floor"]["n"] == 2
        assert result["at_zero_floor"]["n_above"] == 2
        assert result["off_zero_floor"]["n"] == 2
        assert result["one_directional_off_floor"] is False
        assert result["pooled_all_pairs_including_at_floor"]["one_directional"] is False

    def test_skips_cells_missing_either_estimate(self):
        cell_results = {
            "no_fa": self._cell(None, nugget=0.9),
            "census_not_fitted": self._cell(0.1, census_status="not_computable"),
            "ok": self._cell(0.1, nugget=0.9),
        }
        result = mod.observation_noise_estimator_offset(cell_results)
        assert result["n_comparable_pairs"] == 1

    def test_not_computable_when_no_pairs(self):
        result = mod.observation_noise_estimator_offset({"c0": self._cell(None)})
        assert result["status"] == "not_computable"


class TestRelationshipToPooledCrossCorpusPersistenceSlope:
    def _headline(self, width_bins=3, deciding_width=3, n_sessions=72):
        return {
            "deciding_width_bins": deciding_width,
            "heterogeneity_on_mean_d_perm": {"width_bins": width_bins, "n_sessions": n_sessions},
            "bin_width_s": 0.1, "n_splits": 12, "n_null_replicates": 20,
            "deciding_contrast": {"deciding_branch": "flat_cross_unit_state"},
        }

    def test_reports_settings_differences_and_never_revises(self):
        cell_results = {"c0": {"summary": {"d_perm_slope": {"status": "fitted", "median_pca": -0.02}}}}
        result = mod.relationship_to_pooled_cross_corpus_persistence_slope(cell_results, self._headline())
        assert result["revises_the_pooled_persistence_slope"] is False
        assert result["this_artifact_pca_arm_slope_range_r_per_s"]["n_cells"] == 1
        assert any("null replicates" in s for s in result["settings_differences"])

    def test_raises_when_pooled_session_count_not_at_deciding_width(self):
        with pytest.raises(ValueError):
            mod.relationship_to_pooled_cross_corpus_persistence_slope({}, self._headline(width_bins=5, deciding_width=3))
