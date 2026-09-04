"""Tests for scripts/run_between_session_component_behaviour_state.py: the
planted-signal recovery of the between-session estimator, the removal of a
planted pure unit-count confound by the declared nuisance partial, the
schema separation of the within-session and between-session statistics, and
the refusal of the multi-object corpus's estimators to pool across item
counts. All against synthetic records with an unambiguous ground truth; no
real corpus data is touched. The reused estimators
(partial_correlation_permutation_test, pearson_permutation_test,
forest_meta, slope_across_sessions_test) are already covered by their own
modules' tests and are not re-tested here."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_between_session_component_behaviour_state import (  # noqa: E402
    ALPHA, MixedItemCountsError, REFERENCE_R_UNITS, assemble_corpus,
    between_session_stats, build_unit_record, classify_corpus_branch,
    classify_top_branch, combine_levels, finalize_branches, forbid_mixed_item_count_records,
    minimum_detectable_correlation, within_vs_between_disagreement,
)


N_PERM = 1000  # reduced purely for test wall-clock; every assertion uses margins far wider than the
               # permutation noise at this n_perm


# --------------------------------------------------------------------------------------------------------
# Record factories
# --------------------------------------------------------------------------------------------------------

def _make_records(component: np.ndarray, behaviour: np.ndarray, unit_counts: np.ndarray,
                  item_count: int | None = None) -> list[dict]:
    rng = np.random.default_rng(11)
    return [build_unit_record(
        component[i] + 0.05 * rng.normal(size=20),   # 20 trials per unit, tiny within-unit noise
        behaviour[i] + 0.05 * rng.normal(size=20),
        int(unit_counts[i]), i + 1, float(unit_counts[i]) / 10.0, item_count=item_count)
        for i in range(len(component))]


# --------------------------------------------------------------------------------------------------------
# A planted between-session association is recovered -- raw AND after the nuisance partial
# --------------------------------------------------------------------------------------------------------

def test_planted_between_session_association_is_recovered():
    rng = np.random.default_rng(3)
    n = 40
    latent = rng.normal(size=n)
    # The behaviour mean tracks the SAME latent the component mean carries; the four nuisances are
    # independent of that latent, so the partial has nothing legitimate to remove.
    component = 1.0 * latent + 0.3 * rng.normal(size=n)
    behaviour = 1.0 * latent + 0.3 * rng.normal(size=n)
    unit_counts = rng.integers(20, 200, size=n).astype(float)
    records = _make_records(component, behaviour, unit_counts)

    stats = between_session_stats(records, "test|recovery", n_perm=N_PERM)

    assert stats["status"] == "computed"
    assert stats["raw"]["r"] > 0.7
    assert stats["raw"]["p_value"] <= ALPHA
    part = stats["partialled_after_recording_nuisances"]
    assert part["r"] > 0.5
    assert part["p_value"] <= ALPHA
    assert stats["covariates_partialled"] == [
        "clustering_unit_isolated_unit_count", "clustering_unit_trial_count",
        "clustering_unit_mean_total_spike_count", "order_within_the_recording_series"]


# --------------------------------------------------------------------------------------------------------
# A planted pure unit-count confound is removed by the partial and fires the nuisance branch
# --------------------------------------------------------------------------------------------------------

def test_planted_pure_unit_count_confound_is_removed_and_fires_nuisance_branch():
    rng = np.random.default_rng(7)
    n = 60
    unit_counts = np.linspace(20, 200, n)
    shared = (unit_counts - unit_counts.mean()) / unit_counts.std()
    # BOTH variables load on the unit-count axis and on NOTHING else: the raw association is entirely
    # the confound, and the residualised association must vanish.
    component = shared + 0.6 * rng.normal(size=n)
    behaviour = shared + 0.6 * rng.normal(size=n)
    records = _make_records(component, behaviour, unit_counts)

    stats = between_session_stats(records, "test|confound", n_perm=N_PERM)
    assert stats["status"] == "computed"
    assert stats["raw"]["r"] > 0.6
    assert stats["raw"]["p_value"] <= ALPHA
    assert abs(stats["partialled_after_recording_nuisances"]["r"]) < 0.2

    block = assemble_corpus("synthetic_confound", stats, None, {"status": "not_computable"}, [], n,
                             "non_human")
    _, _top = finalize_branches([block])
    verdict = block["verdict"]
    assert verdict["branch"] == \
        "the_between_session_association_is_explained_by_session_level_recording_nuisances"
    assert verdict["raw_effect_size_r"] == pytest.approx(stats["raw"]["r"])
    assert verdict["partialled_effect_size_r"] == \
        pytest.approx(stats["partialled_after_recording_nuisances"]["r"])
    # effect size beside the verdict, in the same object
    assert verdict["minimum_detectable_correlation_at_80pct_power"] is not None
    assert verdict["standing_reference_r_units"] == REFERENCE_R_UNITS


# --------------------------------------------------------------------------------------------------------
# Within-session and between-session statistics are never pooled
# --------------------------------------------------------------------------------------------------------

def test_within_and_between_statistics_live_in_separate_fields_and_never_merge():
    rng = np.random.default_rng(5)
    n = 24
    latent = rng.normal(size=n)
    records = _make_records(latent, latent, rng.integers(30, 120, size=n).astype(float))
    stats = between_session_stats(records, "test|schema", n_perm=N_PERM)

    within_statistic = {"status": "tested", "mean_value": 0.4, "two_sided_p_value": 0.01,
                         "significant": True}
    block = assemble_corpus("synthetic_schema", stats, None, within_statistic, [], n, "non_human")
    pooled, top = finalize_branches([block])

    # separate sibling fields on the same corpus block -- never a merged statistic
    assert set(block.keys()) >= {"within_session_statistic", "between_session_statistics"}
    assert block["within_session_statistic"] is not block["between_session_statistics"]
    # the between family carries correlation-scale fields only; the sign-flip pooled marker of the
    # within family (mean_value) never appears inside it
    assert "mean_value" not in block["between_session_statistics"]
    assert block["within_session_statistic"]["mean_value"] == 0.4
    assert "declaration" in block["within_session_statistic"]
    assert "never pooled" in block["within_session_statistic"]["declaration"]
    # no key anywhere in the assembled block claims to pool the two families
    for key in block.keys():
        assert "within_and_between" not in key and "pooled_within" not in key

    # sign disagreement fires the required statement naming the within statistic commensurable;
    # aligned signs stay silent
    disagreement = within_vs_between_disagreement({"mean_value": -0.4}, 0.55, 0.60)
    assert disagreement is not None and "commensurable" in disagreement and "never" in disagreement
    assert within_vs_between_disagreement({"mean_value": 0.4}, 0.55, 0.60) is None
    # when the primary partialled estimate is undefined the raw estimate stands in as the comparison
    assert within_vs_between_disagreement({"mean_value": -0.4}, None, 0.60) is not None


# --------------------------------------------------------------------------------------------------------
# The multi-object corpus never falls back to a pooled-across-item-count estimator
# --------------------------------------------------------------------------------------------------------

def _simpson_levels() -> tuple[list[dict], list[dict]]:
    """Two item-count levels, each with a STRONG NEGATIVE within-level association, whose levels sit
    far apart with the higher-item level shifted positively on both variables: the naive
    pooled-across-levels correlation is strongly positive while every within-level one is negative."""
    rng = np.random.default_rng(9)
    low_level, high_level = [], []
    for i in range(14):
        c_low = -1.0 + 0.1 * rng.normal()
        b_low = -c_low + (-3.0) + 0.05 * rng.normal()      # negative within level 1
        c_high = 1.0 + 0.1 * rng.normal()
        b_high = -c_high + 3.0 + 0.05 * rng.normal()       # negative within level 2
        for target, c, b, lv in ((low_level, c_low, b_low, 1), (high_level, c_high, b_high, 2)):
            target.append({
                "component_mean": float(c), "behaviour_mean": float(b), "n_trials": 30,
                "unit_count": 40 + i, "mean_total_spike_count_per_trial": 10.0 + 0.1 * i,
                "series_order": i + 1, "item_count": lv})
    return low_level, high_level


def test_multi_object_estimator_refuses_mixed_item_counts_and_combines_within_levels_only():
    low_level, high_level = _simpson_levels()

    # the guard raises on mixed input rather than pooling
    with pytest.raises(MixedItemCountsError):
        forbid_mixed_item_count_records(low_level + high_level)
    with pytest.raises(MixedItemCountsError):
        between_session_stats(low_level + high_level, "test|mixed", n_perm=N_PERM)
    # homogeneous strata pass the guard
    forbid_mixed_item_count_records(low_level)
    forbid_mixed_item_count_records(high_level)

    # what pooling across levels would have produced (the forbidden estimator): strongly positive
    pooled_x = np.array([r["component_mean"] for r in low_level + high_level])
    pooled_y = np.array([r["behaviour_mean"] for r in low_level + high_level])
    naive_pooled_r = float(np.corrcoef(pooled_x, pooled_y)[0, 1])
    assert naive_pooled_r > 0.8

    # the used estimator: within-level estimates combined by trial-count weighting -- stays negative
    stats_low = between_session_stats(low_level, "test|level1", n_perm=N_PERM)
    stats_high = between_session_stats(high_level, "test|level2", n_perm=N_PERM)
    combined = combine_levels([stats_low, stats_high], [len(low_level) * 30, len(high_level) * 30],
                              ["item_count_1", "item_count_2"])
    assert combined["raw"]["status"] == "computed"
    weighted_avg = combined["raw"]["trial_count_weighted_average_r"]
    assert weighted_avg < -0.8
    assert np.sign(weighted_avg) != np.sign(naive_pooled_r)
    # the combined object aggregates precomputed statistics only; it never saw a trial array
    assert not any(isinstance(v, np.ndarray) for v in combined.values())


# --------------------------------------------------------------------------------------------------------
# Power floors: inconclusive cells carry floor and reference in ONE field
# --------------------------------------------------------------------------------------------------------

def test_below_floor_cells_carry_both_numbers_in_one_field_at_every_level():
    floor_small = minimum_detectable_correlation(6)
    assert floor_small["status"] == "computed"
    assert floor_small["minimum_detectable_correlation"] > REFERENCE_R_UNITS

    cell = {"status": "computed", "raw_effect_size_r": 0.10, "partialled_effect_size_r": 0.05,
            "raw_benjamini_hochberg_q_value": 0.70, "partialled_benjamini_hochberg_q_value": 0.80,
            "minimum_detectable_correlation_at_80pct_power": floor_small}
    verdict = classify_corpus_branch(cell)
    assert verdict["branch"] == "inconclusive_below_detection_floor"
    both = verdict["field_carrying_both_numbers"]
    assert both["this_cell_minimum_detectable_correlation"] > REFERENCE_R_UNITS
    assert both["tested_against_reference_r"] == REFERENCE_R_UNITS

    species_map = {"corpus_a": "non_human", "corpus_b": "human"}

    def below_floor_cell(name):
        base = dict(cell)
        base["minimum_detectable_correlation_at_80pct_power"] = minimum_detectable_correlation(6)
        return base

    cells = {name: below_floor_cell(name) for name in ("corpus_a", "corpus_b")}
    cells["__pooled__"] = {"raw_effect_size_r": 0.02, "partialled_effect_size_r": 0.01,
                            "raw_benjamini_hochberg_q_value": 0.9,
                            "partialled_benjamini_hochberg_q_value": 0.95,
                            "minimum_detectable_correlation_at_80pct_power":
                                minimum_detectable_correlation(12)}
    top = classify_top_branch(cells, species_map)
    assert top["branch"] == "inconclusive_below_detection_floor"
    per_corpus_field = top["field_carrying_both_numbers_per_corpus"]
    for name in ("corpus_a", "corpus_b"):
        assert per_corpus_field[name]["this_corpus_minimum_detectable_correlation"] > REFERENCE_R_UNITS
        assert per_corpus_field[name]["tested_against_reference_r"] == REFERENCE_R_UNITS


def test_too_few_units_for_the_partial_is_inconclusive_not_a_powered_null():
    # five units cannot support even the power formula: the branch must be inconclusive carrying the
    # uncomputability, never 'no_between_session_association_at_either_level'
    cell = {"status": "too_few_clustering_units_for_the_declared_partial",
            "raw_effect_size_r": None, "partialled_effect_size_r": None,
            "raw_benjamini_hochberg_q_value": None, "partialled_benjamini_hochberg_q_value": None,
            "minimum_detectable_correlation_at_80pct_power": minimum_detectable_correlation(4)}
    verdict = classify_corpus_branch(cell)
    assert verdict["branch"] == "inconclusive_below_detection_floor"
    assert verdict["field_carrying_both_numbers"]["this_cell_minimum_detectable_correlation"] is None
