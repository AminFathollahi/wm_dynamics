"""Tests for scripts/run_count_subsampling_ladder.py.

Covers the pieces that could silently break the unit-subsampling ladder
without erroring: (1) per-session target-unit-count resolution from a
declared median-total-spike-count rung, including the unreachable and
native (no-subsampling) cases; (2) within-session pooling of repeated
subsampling draws; (3) the multi-object corpus's within-item-count-level,
trial-count-weighted combination arithmetic; (4) the named branch
classifier for the count ladder's own pre-declared decision rule;
(5) the floor-to-unit-count recording-specification arithmetic; and
(6) an end-to-end planted/null recovery check on synthetic per-trial
per-unit activity, run through the real rate-free deviation estimator and
the real orthogonality-gate primitive rather than a mock of either."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_count_subsampling_ladder import (  # noqa: E402
    classify_block_a_branch,
    combine_within_load_trial_weighted,
    pool_draws_within_session,
    resolve_unit_target,
    run_corpus_ladder,
    translate_floor_to_unit_count,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from statistics import partial_correlation_permutation_test, stable_seed  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# resolve_unit_target
# ---------------------------------------------------------------------------------------------------

def test_resolve_unit_target_native_when_target_equals_full_median():
    result = resolve_unit_target(n_units_full=100, native_median_total=1000.0, target_median_total=1000.0)
    assert result["status"] == "native"
    assert result["n_units"] == 100


def test_resolve_unit_target_unreachable_when_target_exceeds_full_median():
    result = resolve_unit_target(n_units_full=100, native_median_total=500.0, target_median_total=1000.0)
    assert result["status"] == "unreachable"


def test_resolve_unit_target_subsample_scales_with_per_unit_rate():
    # 100 units produce a native median of 1000 total spikes/trial -> ~10/unit. A target of 300 should
    # ask for roughly 30 units, not the full set and not zero.
    result = resolve_unit_target(n_units_full=100, native_median_total=1000.0, target_median_total=300.0)
    assert result["status"] == "subsample"
    assert 25 <= result["n_units"] <= 35


def test_resolve_unit_target_clips_to_at_least_one_unit():
    result = resolve_unit_target(n_units_full=50, native_median_total=1000.0, target_median_total=1.0)
    assert result["status"] == "subsample"
    assert result["n_units"] >= 1


def test_resolve_unit_target_zero_native_median_is_unreachable():
    result = resolve_unit_target(n_units_full=10, native_median_total=0.0, target_median_total=50.0)
    assert result["status"] == "unreachable"


# ---------------------------------------------------------------------------------------------------
# pool_draws_within_session
# ---------------------------------------------------------------------------------------------------

def test_pool_draws_within_session_averages_finite_values():
    assert pool_draws_within_session([0.1, 0.2, 0.3]) == pytest.approx(0.2)


def test_pool_draws_within_session_ignores_none_and_nan():
    assert pool_draws_within_session([0.1, None, float("nan"), 0.3]) == pytest.approx(0.2)


def test_pool_draws_within_session_all_missing_returns_none():
    assert pool_draws_within_session([None, None]) is None
    assert pool_draws_within_session([]) is None


# ---------------------------------------------------------------------------------------------------
# combine_within_load_trial_weighted
# ---------------------------------------------------------------------------------------------------

def test_combine_within_load_trial_weighted_matches_manual_weighted_average():
    # Two levels: 10 trials at r=0.5, 30 trials at r=-0.1 -> weighted mean = (10*0.5 + 30*-0.1) / 40
    levels = {1: {"n_trials": 10, "r": 0.5}, 2: {"n_trials": 30, "r": -0.1}}
    expected = (10 * 0.5 + 30 * -0.1) / 40
    assert combine_within_load_trial_weighted(levels) == pytest.approx(expected)


def test_combine_within_load_trial_weighted_skips_uncomputed_levels():
    levels = {1: {"n_trials": 10, "r": 0.5}, 2: {"n_trials": 30, "r": None}}
    assert combine_within_load_trial_weighted(levels) == pytest.approx(0.5)


def test_combine_within_load_trial_weighted_no_levels_returns_none():
    assert combine_within_load_trial_weighted({}) is None
    assert combine_within_load_trial_weighted({1: {"n_trials": 10, "r": None}}) is None


# ---------------------------------------------------------------------------------------------------
# classify_block_a_branch
# ---------------------------------------------------------------------------------------------------

def _rung(target, significant, mdd=None, n_sessions=10):
    return {"target": target, "significant": significant, "mdd": mdd, "n_sessions": n_sessions}


def test_branch_fails_once_cut_to_failing_counts():
    rungs = [_rung(1847.5, False), _rung(989.5, False), _rung(700.0, False),
             _rung(500.0, False), _rung(353.0, True), _rung(256.25, True), _rung(132.5, True)]
    out = classify_block_a_branch(rungs, highest_failing_target=353.0, failing_reference_effect_abs=0.17)
    assert out["branch"] == "the_orthogonality_gate_fails_once_the_passing_corpora_are_cut_to_the_failing_corpora_counts"


def test_branch_survives_at_failing_counts_when_powered_null():
    rungs = [_rung(1847.5, False, mdd=0.05), _rung(989.5, False, mdd=0.05), _rung(700.0, False, mdd=0.05),
             _rung(500.0, False, mdd=0.05), _rung(353.0, False, mdd=0.05),
             _rung(256.25, False, mdd=0.05), _rung(132.5, False, mdd=0.05)]
    out = classify_block_a_branch(rungs, highest_failing_target=353.0, failing_reference_effect_abs=0.17)
    assert out["branch"] == "the_orthogonality_gate_survives_at_the_failing_corpora_counts"


def test_branch_inconclusive_when_underpowered_at_the_lowest_rung():
    rungs = [_rung(1847.5, False, mdd=0.05), _rung(989.5, False, mdd=0.4), _rung(700.0, False, mdd=0.4),
             _rung(500.0, False, mdd=0.4), _rung(353.0, False, mdd=0.4),
             _rung(256.25, False, mdd=0.4), _rung(132.5, False, mdd=0.4)]
    out = classify_block_a_branch(rungs, highest_failing_target=353.0, failing_reference_effect_abs=0.17)
    assert out["branch"] == "inconclusive_the_subsampled_gate_is_below_its_own_detection_floor"


def test_branch_degrades_with_no_identifiable_transition_when_crossing_sits_above_the_gap():
    # Significant only at the native/full rung (above the sampled gap [353, 989.5]); the rung matching
    # the highest failing corpus's own count (989.5 here) never crosses. This is neither "fails once cut"
    # (that needs the FULL rung non-significant) nor "survives" (not every rung is non-significant), so
    # the pre-declared rule's third named outcome is what must fire.
    rungs = [_rung(1847.5, True), _rung(989.5, False), _rung(700.0, False),
             _rung(500.0, False), _rung(353.0, False), _rung(256.25, False), _rung(132.5, False)]
    out = classify_block_a_branch(rungs, highest_failing_target=989.5, failing_reference_effect_abs=0.17)
    assert out["branch"] == "the_gate_degrades_with_no_identifiable_transition_inside_the_sampled_range"


def test_branch_rule_gap_when_full_and_matching_are_both_already_significant():
    # A pattern the pre-declared rule's first outcome explicitly does not cover (it requires the FULL
    # rung to be non-significant) and that is not "all non-significant" either -- must be reported as a
    # gap in the rule, not forced onto the nearest label.
    rungs = [_rung(1847.5, True), _rung(989.5, True), _rung(700.0, False),
             _rung(500.0, False), _rung(353.0, True), _rung(256.25, False), _rung(132.5, False)]
    out = classify_block_a_branch(rungs, highest_failing_target=353.0, failing_reference_effect_abs=0.17)
    assert out["branch"] == "outcome_pattern_not_covered_by_the_pre_declared_rule"


# ---------------------------------------------------------------------------------------------------
# translate_floor_to_unit_count
# ---------------------------------------------------------------------------------------------------

def test_translate_floor_to_unit_count_basic_division():
    # A floor of 400 total spikes/trial at a preparation firing 4 spikes/unit/trial needs 100 units.
    assert translate_floor_to_unit_count(400.0, 4.0) == pytest.approx(100.0)


def test_translate_floor_to_unit_count_zero_rate_is_not_computable():
    assert translate_floor_to_unit_count(400.0, 0.0) is None


# ---------------------------------------------------------------------------------------------------
# End-to-end planted / null recovery on synthetic per-trial per-unit activity, through the REAL
# rate_free_state_deviation estimator and the real permutation-based orthogonality gate -- not a mock
# of either.
#
# Plain homogeneous-Poisson activity around a shared direction (tried first, not shown here) turns out
# NOT to give a clean "weak at many units, strong at few units" pattern for this cosine-based estimator at
# realistic firing rates -- its relationship to total count is driven mostly by total count itself, only
# weakly by how many units that total is spread across, at the scales this project's own five-corpus
# census actually spans. Demonstrating whether that relationship really is a unit-count-specific artifact
# in REAL data is what the full script exists to answer; the unit test below only needs to exercise the
# LADDER'S OWN MECHANICS (which units get drawn, how draws are pooled within and across sessions, how a
# floor is read off), so the dependence is injected directly and explicitly instead: a per-trial
# multiplicative scalar changes a trial's total without touching its direction at all (cosine similarity
# is exactly scale-invariant), giving a mathematically guaranteed NULL baseline with zero relationship
# between total and deviation at any unit count; a labelled subset of low-total trials additionally gets
# its direction replaced by an independent random one, but ONLY when the unit count is at or below a
# declared threshold -- giving a mathematically guaranteed PLANTED effect that is absent above the
# threshold and present below it.
# ---------------------------------------------------------------------------------------------------

_SYNTHETIC_N_TRIALS = 250
_SYNTHETIC_N_UNITS_FULL = 120
_SYNTHETIC_REFERENCE_TOTAL = 800.0
_SYNTHETIC_CORRUPTED_FRACTION = 0.3
_SYNTHETIC_CORRUPTION_UNIT_THRESHOLD = 15


def _synthetic_activity(session_seed: str, unit_indices, corruption_active: bool) -> np.ndarray:
    direction_full = np.full(_SYNTHETIC_N_UNITS_FULL, 1.0 / _SYNTHETIC_N_UNITS_FULL)
    direction = direction_full if unit_indices is None else direction_full[unit_indices]
    direction = direction / direction.sum()
    k = len(direction)

    label_rng = np.random.default_rng(stable_seed(f"synthetic_labels|{session_seed}"))
    is_low = np.zeros(_SYNTHETIC_N_TRIALS, dtype=bool)
    n_low = int(round(_SYNTHETIC_N_TRIALS * _SYNTHETIC_CORRUPTED_FRACTION))
    is_low[label_rng.choice(_SYNTHETIC_N_TRIALS, size=n_low, replace=False)] = True

    poisson_rng = np.random.default_rng(stable_seed(f"synthetic_poisson|{session_seed}"))
    activity = np.zeros((_SYNTHETIC_N_TRIALS, k))
    for i in range(_SYNTHETIC_N_TRIALS):
        if corruption_active and is_low[i]:
            corrupt_rng = np.random.default_rng(stable_seed(f"synthetic_corrupt|{session_seed}|{i}|{k}"))
            d = np.abs(corrupt_rng.standard_normal(k))
            d = d / d.sum()
        else:
            d = direction
        raw = poisson_rng.poisson(d * _SYNTHETIC_REFERENCE_TOTAL)
        activity[i] = raw * (0.2 if is_low[i] else 1.0)
    return activity


def _synthetic_native_median_total(session_seed: str) -> float:
    activity = _synthetic_activity(session_seed, None, corruption_active=False)
    return float(np.median(activity.sum(axis=1)))


def _draw_stats_from_synthetic(session, unit_indices, n_perm, seed_tag, corruption_threshold):
    corruption_active = unit_indices is not None and len(unit_indices) <= corruption_threshold
    activity = _synthetic_activity(session["session"], unit_indices, corruption_active)
    deviation = rate_free_state_deviation(activity)
    total = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < 8:
        return None
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|gate"))
    gate = partial_correlation_permutation_test(deviation[finite], total[finite], controls=[], n_perm=n_perm, rng=rng)
    return {
        "gate_dev": gate, "raw_dev": {"status": "not_tested"},
        "gate_amp": {"status": "not_tested"}, "raw_amp": {"status": "not_tested"},
        "n_trials_with_defined_direction": int(finite.sum()),
        "achieved_median_total": float(np.median(total[finite])) if finite.any() else None,
        "achieved_median_per_unit": float(np.median(activity[finite])) if finite.any() else None,
    }


def test_ladder_recovers_a_planted_unit_count_dependent_floor():
    sessions = [{"session": f"planted_{s}", "n_units_full": _SYNTHETIC_N_UNITS_FULL,
                 "native_median_total": _synthetic_native_median_total(f"planted_{s}")} for s in range(8)]

    def compute_draw_stats(session, unit_indices, n_perm, seed_tag):
        return _draw_stats_from_synthetic(session, unit_indices, n_perm, seed_tag,
                                           corruption_threshold=_SYNTHETIC_CORRUPTION_UNIT_THRESHOLD)

    rung_targets = [700.0, 90.0]  # ~105 units (above threshold) and ~13 units (at/below threshold)
    result = run_corpus_ladder(
        sessions=sessions, rung_targets=rung_targets, compute_draw_stats=compute_draw_stats,
        n_draws=25, n_perm_draw=500, n_perm_native=2000, seed_namespace="test_planted",
    )
    high = result["by_rung"][700.0]
    low = result["by_rung"][90.0]
    assert high["n_sessions_computed"] >= 5
    assert low["n_sessions_computed"] >= 5
    assert low["achieved_median_unit_count"] <= _SYNTHETIC_CORRUPTION_UNIT_THRESHOLD
    assert high["achieved_median_unit_count"] > _SYNTHETIC_CORRUPTION_UNIT_THRESHOLD
    # The planted floor: significant (and strongly negative, low total -> corrupted direction -> high
    # deviation) once cut down to few units, non-significant with many.
    assert low["pooled_gate_dev"]["significant"] is True
    assert low["pooled_gate_dev"]["mean_value"] < 0.0
    assert high["pooled_gate_dev"].get("significant") is not True

    rungs_for_branch = [
        {"target": target, "significant": rung["pooled_gate_dev"].get("significant"),
         "mdd": rung["pooled_gate_dev"].get("minimum_detectable_paired_difference_at_80pct_power", {}).get("mdd")}
        for target, rung in result["by_rung"].items()
    ]
    branch = classify_block_a_branch(rungs_for_branch, highest_failing_target=90.0, failing_reference_effect_abs=0.1)
    assert branch["branch"] == "the_orthogonality_gate_fails_once_the_passing_corpora_are_cut_to_the_failing_corpora_counts"


def test_ladder_finds_no_floor_when_there_is_no_count_dependence():
    sessions = [{"session": f"null_{s}", "n_units_full": _SYNTHETIC_N_UNITS_FULL,
                 "native_median_total": _synthetic_native_median_total(f"null_{s}")} for s in range(8)]

    def compute_draw_stats(session, unit_indices, n_perm, seed_tag):
        # corruption_threshold=-1: len(unit_indices) is never <= -1, so corruption never activates at any
        # unit count -- the scale-invariance argument in the module docstring above applies everywhere.
        return _draw_stats_from_synthetic(session, unit_indices, n_perm, seed_tag, corruption_threshold=-1)

    rung_targets = [700.0, 90.0]
    result = run_corpus_ladder(
        sessions=sessions, rung_targets=rung_targets, compute_draw_stats=compute_draw_stats,
        n_draws=25, n_perm_draw=500, n_perm_native=2000, seed_namespace="test_null",
    )
    for target in rung_targets:
        pooled = result["by_rung"][target]["pooled_gate_dev"]
        if pooled.get("status") == "tested":
            assert pooled["significant"] is False

    rungs_for_branch = [
        {"target": target, "significant": rung["pooled_gate_dev"].get("significant"),
         "mdd": rung["pooled_gate_dev"].get("minimum_detectable_paired_difference_at_80pct_power", {}).get("mdd")}
        for target, rung in result["by_rung"].items()
    ]
    branch = classify_block_a_branch(rungs_for_branch, highest_failing_target=90.0, failing_reference_effect_abs=0.1)
    assert branch["branch"] != "the_orthogonality_gate_fails_once_the_passing_corpora_are_cut_to_the_failing_corpora_counts"
