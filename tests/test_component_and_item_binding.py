"""Tests for scripts/run_component_and_item_binding.py -- the pieces that
could silently break this analysis without erroring: (1) the swap/
imprecision geometry itself (the only new estimator this module introduces),
on hand-computed trials where the correct answer is known by construction;
(2) the within-item-count-level trial-count-weighted combination arithmetic;
(3) the six-way decision-rule classifier, including the paired swap-vs-
imprecision test; (4) the reproduction-gate tolerance helper; (5) the three
reporting-only disclosures added after the delivered run (load-1 detection
floor, amplitude-vs-deviation detection floor and direct paired test,
imprecision detection floor vs the swap effect) -- these must never move a
branch or recompute a fit, only read already-pooled statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_component_and_item_binding import (  # noqa: E402
    BRANCH_BOTH, BRANCH_GATE_FAILED, BRANCH_IMPRECISION_ONLY, BRANCH_NEITHER, BRANCH_SWAPS_ONLY,
    BRANCH_TOO_RARE, MIN_SWAPS_POOLED_TO_TEST, _close, _family, _object_geometry, _outcome_arm,
    _paired_collect_observables, _predicts, build_disclosures, decide_branch,
)


# ---------------------------------------------------------------------------------------------------
# Swap / imprecision geometry
# ---------------------------------------------------------------------------------------------------

def _synthetic_behaviour_and_session() -> tuple[pd.DataFrame, dict]:
    """Four hand-computable trials: one single-object trial, a two-object swap, a two-object non-swap,
    and a three-object swap that is also within the strict distance threshold."""
    rows = [
        # trial, num_objects, target, obj0(x,y), obj1(x,y), obj2(x,y), response(x,y)
        dict(trial_num=1, num_objects=1, target_object_index=0,
             object_0_x=0.0, object_0_y=0.0, object_1_x=np.nan, object_1_y=np.nan,
             object_2_x=np.nan, object_2_y=np.nan, response_x=0.05, response_y=0.0),
        dict(trial_num=2, num_objects=2, target_object_index=0,
             object_0_x=0.0, object_0_y=0.0, object_1_x=10.0, object_1_y=0.0,
             object_2_x=np.nan, object_2_y=np.nan, response_x=9.9, response_y=0.0),
        dict(trial_num=3, num_objects=2, target_object_index=0,
             object_0_x=0.0, object_0_y=0.0, object_1_x=10.0, object_1_y=0.0,
             object_2_x=np.nan, object_2_y=np.nan, response_x=0.5, response_y=0.0),
        dict(trial_num=4, num_objects=3, target_object_index=0,
             object_0_x=0.0, object_0_y=0.0, object_1_x=1.0, object_1_y=0.0,
             object_2_x=0.0, object_2_y=1.0, response_x=0.9, response_y=0.05),
    ]
    frame = pd.DataFrame(rows)
    frame["subject"] = "TestAnimal"
    frame["session"] = "2026-01-01"
    tx = np.choose(frame.target_object_index.to_numpy(), [frame.object_0_x, frame.object_1_x, frame.object_2_x])
    ty = np.choose(frame.target_object_index.to_numpy(), [frame.object_0_y, frame.object_1_y, frame.object_2_y])
    frame["report_deviation"] = np.hypot(tx - frame.response_x.to_numpy(), ty - frame.response_y.to_numpy())
    behaviour = frame.set_index(["subject", "session", "trial_num"], drop=False).sort_index()

    session = {
        "animal": "TestAnimal", "session_date": "2026-01-01",
        "trial_num": np.array([1, 2, 3, 4], dtype=int),
        "report_deviation": frame["report_deviation"].to_numpy(dtype=float),
    }
    return behaviour, session


def test_object_geometry_matches_hand_computed_swaps_and_imprecision():
    behaviour, session = _synthetic_behaviour_and_session()
    geometry = _object_geometry(behaviour, session)

    # Trial 1 (1 object): always landed on its only object, never a swap.
    assert geometry["swap_primary"][0] == False  # noqa: E712
    assert geometry["swap_strict"][0] == False  # noqa: E712
    assert geometry["imprecision"][0] == pytest.approx(0.05, abs=1e-9)

    # Trial 2 (2 objects): response is 0.1 from the UNCUED object and 9.9 from the cued one -- a swap
    # under both the primary (nearest-object) and strict (within 0.2 of the uncued object) rules.
    assert geometry["swap_primary"][1] == True  # noqa: E712
    assert geometry["swap_strict"][1] == True  # noqa: E712
    assert geometry["imprecision"][1] == pytest.approx(0.1, abs=1e-9)

    # Trial 3 (2 objects): response is close to the CUED object -- not a swap under either rule.
    assert geometry["swap_primary"][2] == False  # noqa: E712
    assert geometry["swap_strict"][2] == False  # noqa: E712
    assert geometry["imprecision"][2] == pytest.approx(0.5, abs=1e-9)

    # Trial 4 (3 objects): nearest object is the uncued object 1, at distance ~0.1118 -- a swap under
    # both rules (0.1118 < the 0.2 strict threshold).
    assert geometry["swap_primary"][3] == True  # noqa: E712
    assert geometry["swap_strict"][3] == True  # noqa: E712
    assert geometry["imprecision"][3] == pytest.approx(np.hypot(0.1, 0.05), abs=1e-9)

    # The identity check: distance-to-target recomputed from raw Cartesian columns must match the
    # loader's own report_deviation (here constructed identically, so this must be ~0).
    assert geometry["identity_check_max_abs_diff_target_distance_vs_report_deviation"] < 1e-9
    assert geometry["identity_check_n_compared"] == 4
    assert geometry["n_trials_all_object_positions_undefined"] == 0


def test_swap_undefined_at_item_count_one_is_never_flagged_a_swap():
    # A single-object trial can never be a swap under either rule, no matter how far off the response
    # lands -- there is no other object to have swapped onto.
    behaviour, session = _synthetic_behaviour_and_session()
    geometry = _object_geometry(behaviour, session)
    assert not geometry["swap_primary"][0]
    assert not geometry["swap_strict"][0]


# ---------------------------------------------------------------------------------------------------
# Within-item-count-level trial-count-weighted combination
# ---------------------------------------------------------------------------------------------------

def test_within_item_count_level_weighting_matches_hand_computed_average():
    rng = np.random.default_rng(0)
    n2, n3 = 20, 30
    item_count = np.concatenate([np.full(n2, 2.0), np.full(n3, 3.0)])

    x2 = rng.normal(size=n2)
    y2 = 0.6 * x2 + rng.normal(scale=0.5, size=n2)  # a real, moderate correlation at level 2
    x3 = rng.normal(size=n3)
    y3 = -0.2 * x3 + rng.normal(scale=0.5, size=n3)  # a different, weaker/opposite correlation at level 3

    deviation = np.concatenate([x2, x3])
    amplitude = deviation.copy()
    outcome = np.concatenate([y2, y3])
    spike_count = rng.normal(size=n2 + n3)
    trial_index = np.arange(n2 + n3, dtype=float)

    arm = _outcome_arm(outcome, item_count, deviation, amplitude, spike_count, trial_index, "unit_test")

    r2 = float(np.corrcoef(x2, y2)[0, 1])
    r3 = float(np.corrcoef(x3, y3)[0, 1])
    expected = (n2 * r2 + n3 * r3) / (n2 + n3)
    observed = arm["deviation"]["within_item_count_level_trial_count_weighted"]["raw"]
    assert observed == pytest.approx(expected, abs=1e-9)

    # The per-level raw r reported alongside must equal the plain Pearson correlation on that level alone.
    assert arm["deviation"]["per_level"]["2"]["family"]["raw"]["r"] == pytest.approx(r2, abs=1e-9)
    assert arm["deviation"]["per_level"]["3"]["family"]["raw"]["r"] == pytest.approx(r3, abs=1e-9)

    # pooled_mixed is the unstratified correlation over every item-count>=2 trial together, and must
    # differ from the within-level combination whenever the two levels have different correlations
    # (mixing them changes the estimate) -- the whole point of reporting both.
    r_pooled_mixed = float(np.corrcoef(deviation, outcome)[0, 1])
    assert arm["deviation"]["pooled_mixed"]["family"]["raw"]["r"] == pytest.approx(r_pooled_mixed, abs=1e-9)


# ---------------------------------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------------------------------

def _stat(sig: bool, mean_value: float = 0.05, mdd: float = 0.02) -> dict:
    return {
        "status": "tested", "significant": sig, "mean_value": mean_value, "p_value": 0.01 if sig else 0.5,
        "minimum_detectable_paired_difference_at_80pct_power": {"status": "computed", "mdd": mdd},
    }


def _pooled_for(swap_raw_sig: bool, swap_joint_sig: bool, imprecision_raw_sig: bool,
                 imprecision_joint_sig: bool) -> dict:
    def _outcome_block(raw_sig: bool, joint_sig: bool) -> dict:
        return {
            "deviation": {
                "within_item_count_level": {
                    "raw": _stat(raw_sig),
                    "partial_controlling_spike_count": _stat(raw_sig),
                    "partial_controlling_trial_index": _stat(raw_sig),
                    "joint_partial_controlling_spike_count_and_trial_index": _stat(joint_sig),
                },
                "pooled_mixed": {k: _stat(raw_sig) for k in (
                    "raw", "partial_controlling_spike_count", "partial_controlling_trial_index",
                    "joint_partial_controlling_spike_count_and_trial_index")},
            },
            "amplitude": {
                "within_item_count_level": {
                    "raw": _stat(False), "partial_controlling_spike_count": _stat(False),
                    "partial_controlling_trial_index": _stat(False),
                    "joint_partial_controlling_spike_count_and_trial_index": _stat(False)},
                "pooled_mixed": {k: _stat(False) for k in (
                    "raw", "partial_controlling_spike_count", "partial_controlling_trial_index",
                    "joint_partial_controlling_spike_count_and_trial_index")},
            },
        }
    return {
        "swap_primary": _outcome_block(swap_raw_sig, swap_joint_sig),
        "swap_strict": _outcome_block(swap_raw_sig, swap_joint_sig),
        "imprecision": _outcome_block(imprecision_raw_sig, imprecision_joint_sig),
    }


_REFERENCE = {"source_artifact": "results/watters_load_decomposition.json", "mean_value": 0.019673, "p_value": 3e-4}


def test_branch_reproduction_gate_failed_short_circuits_everything():
    pooled = _pooled_for(True, True, True, True)
    result = decide_branch("not_reproduced", 100, pooled, _REFERENCE, rows=[])
    assert result["branch"] == BRANCH_GATE_FAILED


def test_branch_swap_too_rare():
    pooled = _pooled_for(True, True, True, True)
    result = decide_branch("reproduced_exactly", MIN_SWAPS_POOLED_TO_TEST - 1, pooled, _REFERENCE, rows=[])
    assert result["branch"] == BRANCH_TOO_RARE
    assert result["n_swap_trials_pooled_primary_rule"] == MIN_SWAPS_POOLED_TO_TEST - 1


def test_branch_swaps_only():
    pooled = _pooled_for(True, True, False, False)
    result = decide_branch("reproduced_exactly", 50, pooled, _REFERENCE, rows=[])
    assert result["branch"] == BRANCH_SWAPS_ONLY
    assert result["predicts_swap"] is True
    assert result["predicts_imprecision"] is False


def test_branch_imprecision_only():
    pooled = _pooled_for(False, False, True, True)
    result = decide_branch("reproduced_exactly", 50, pooled, _REFERENCE, rows=[])
    assert result["branch"] == BRANCH_IMPRECISION_ONLY


def test_branch_neither_reports_mdd_against_the_delivered_reference():
    pooled = _pooled_for(False, False, False, False)
    result = decide_branch("reproduced_exactly", 50, pooled, _REFERENCE, rows=[])
    assert result["branch"] == BRANCH_NEITHER
    assert result["minimum_detectable_paired_difference_swap_vs_delivered_whole_corpus_effect"] == 0.02
    assert result["delivered_whole_corpus_effect_size_reference"] == _REFERENCE


def test_branch_both_runs_the_paired_test_and_does_not_claim_preference_without_significance():
    pooled = _pooled_for(True, True, True, True)
    # Identical per-session swap and imprecision within-load values -- the paired difference must be
    # exactly zero and therefore never significant, so no preference claim is made.
    rows = [
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.05}}},
         "imprecision": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.05}}}},
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.08}}},
         "imprecision": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.08}}}},
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.03}}},
         "imprecision": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.03}}}},
    ]
    result = decide_branch("reproduced_exactly", 50, pooled, _REFERENCE, rows=rows)
    assert result["branch"] == BRANCH_BOTH
    paired = result["paired_test_swap_vs_imprecision_effect_size"]
    assert paired["n_sessions_paired"] == 3
    assert paired["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert paired["significant_difference"] is False
    assert "no claim of preference" in result["preference_claim"]


def test_branch_both_with_too_few_paired_sessions_reports_not_computable():
    pooled = _pooled_for(True, True, True, True)
    result = decide_branch("reproduced_exactly", 50, pooled, _REFERENCE, rows=[])
    assert result["branch"] == BRANCH_BOTH
    assert result["paired_test_swap_vs_imprecision_effect_size"]["status"] == "not_computable"


# ---------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------

def test_predicts_requires_both_raw_and_joint_partial_significant():
    pooled = _pooled_for(True, False, False, False)  # raw significant, joint partial not
    assert _predicts(pooled["swap_primary"]["deviation"]) is False
    pooled = _pooled_for(True, True, False, False)
    assert _predicts(pooled["swap_primary"]["deviation"]) is True


def test_close_respects_tolerance():
    assert _close(1.0, 1.0 + 1e-9)
    assert not _close(1.0, 1.0 + 1e-3)
    assert not _close(None, 1.0)


def test_family_no_controls_matches_plain_pearson_r():
    rng = np.random.default_rng(1)
    n = 40
    x = rng.normal(size=n)
    y = 0.4 * x + rng.normal(scale=0.7, size=n)
    spike_count = rng.normal(size=n)
    trial_index = np.arange(n, dtype=float)
    fam = _family(y, x, spike_count, trial_index, "unit_test_family")
    assert fam["raw"]["r"] == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=1e-9)


# ---------------------------------------------------------------------------------------------------
# Post-delivery disclosures (added after the run: reporting only, no fit re-run, no branch moved)
# ---------------------------------------------------------------------------------------------------

def _leaf(mean_value: float, mdd: float, sig: bool = True) -> dict:
    return {
        "status": "tested", "significant": sig, "mean_value": mean_value, "p_value": 0.01 if sig else 0.5,
        "minimum_detectable_paired_difference_at_80pct_power": {"status": "computed", "mdd": mdd},
    }


def _minimal_pooled(swap_deviation_mean: float, swap_amplitude_mean: float, swap_amplitude_mdd: float,
                     imprecision_deviation_mdd: float) -> dict:
    return {
        "swap_primary": {
            "deviation": {"within_item_count_level": {"raw": _leaf(swap_deviation_mean, 0.01)}},
            "amplitude": {"within_item_count_level": {"raw": _leaf(
                swap_amplitude_mean, swap_amplitude_mdd, sig=False)}},
        },
        "imprecision": {
            "deviation": {"within_item_count_level": {"raw": _leaf(
                0.001, imprecision_deviation_mdd, sig=False)}},
        },
    }


def test_disclosure_load1_flags_an_underpowered_null_that_exceeds_the_swap_effect():
    pooled = _minimal_pooled(swap_deviation_mean=0.02, swap_amplitude_mean=-0.006,
                              swap_amplitude_mdd=0.03, imprecision_deviation_mdd=0.01)
    # Same shape as _pool_values' own output: a mean_value larger than the swap effect, same sign, and
    # an MDD that exceeds it -- exactly the delivered load-1 cell's own situation.
    load1_deviation_raw = _leaf(0.03, 0.04, sig=False)
    disclosures = build_disclosures(pooled, load1_deviation_raw, rows=[])
    d = disclosures["load1_cannot_exclude_the_swap_effect"]
    assert d["load1_point_estimate_exceeds_swap_effect_magnitude"] is True
    assert d["load1_point_estimate_same_direction_as_swap_effect"] is True
    assert d["load1_null_is_underpowered_against_the_swap_effect"] is True
    assert "does not establish the component as a purely binding signal" in d["statement"]


def test_disclosure_load1_does_not_fire_when_it_can_exclude_the_swap_effect():
    # A load-1 cell whose MDD is BELOW the swap effect, and whose point estimate is smaller, does not
    # get the "cannot exclude" flags -- this disclosure must not overclaim in the other direction.
    pooled = _minimal_pooled(swap_deviation_mean=0.02, swap_amplitude_mean=-0.006,
                              swap_amplitude_mdd=0.03, imprecision_deviation_mdd=0.01)
    load1_deviation_raw = _leaf(0.001, 0.005, sig=False)
    disclosures = build_disclosures(pooled, load1_deviation_raw, rows=[])
    d = disclosures["load1_cannot_exclude_the_swap_effect"]
    assert d["load1_point_estimate_exceeds_swap_effect_magnitude"] is False
    assert d["load1_null_is_underpowered_against_the_swap_effect"] is False


def test_disclosure_amplitude_mdd_exceeds_swap_effect_and_runs_the_direct_paired_test():
    pooled = _minimal_pooled(swap_deviation_mean=0.02, swap_amplitude_mean=-0.006,
                              swap_amplitude_mdd=0.03, imprecision_deviation_mdd=0.01)
    load1_deviation_raw = _leaf(0.001, 0.005, sig=False)
    rows = [
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.05}},
                           "amplitude": {"within_item_count_level_trial_count_weighted": {"raw": -0.01}}}},
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.03}},
                           "amplitude": {"within_item_count_level_trial_count_weighted": {"raw": 0.00}}}},
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.07}},
                           "amplitude": {"within_item_count_level_trial_count_weighted": {"raw": -0.02}}}},
    ]
    disclosures = build_disclosures(pooled, load1_deviation_raw, rows)
    d = disclosures["amplitude_versus_deviation_swap_association"]
    assert d["amplitude_mdd_exceeds_deviation_swap_effect"] is True
    assert d["direct_paired_test_deviation_vs_amplitude_swap_association"]["status"] == "computed"
    assert d["direct_paired_test_deviation_vs_amplitude_swap_association"]["n_sessions_paired"] == 3
    assert "not a demonstrated difference" in d["statement"]


def test_disclosure_amplitude_direct_test_not_computable_with_too_few_paired_sessions():
    pooled = _minimal_pooled(swap_deviation_mean=0.02, swap_amplitude_mean=-0.006,
                              swap_amplitude_mdd=0.03, imprecision_deviation_mdd=0.01)
    load1_deviation_raw = _leaf(0.001, 0.005, sig=False)
    disclosures = build_disclosures(pooled, load1_deviation_raw, rows=[])
    direct_test = disclosures["amplitude_versus_deviation_swap_association"][
        "direct_paired_test_deviation_vs_amplitude_swap_association"]
    assert direct_test["status"] == "not_computable"


def test_disclosure_imprecision_mdd_below_swap_effect_is_flagged():
    pooled = _minimal_pooled(swap_deviation_mean=0.02, swap_amplitude_mean=-0.006,
                              swap_amplitude_mdd=0.03, imprecision_deviation_mdd=0.01)
    load1_deviation_raw = _leaf(0.001, 0.005, sig=False)
    disclosures = build_disclosures(pooled, load1_deviation_raw, rows=[])
    d = disclosures["imprecision_detection_floor_versus_swap_effect"]
    assert d["imprecision_mdd_below_swap_effect_magnitude"] is True
    assert d["imprecision_minimum_detectable_paired_difference_at_80pct_power"] == 0.01
    assert d["swap_effect_magnitude_within_item_count_level"] == 0.02


def test_disclosure_imprecision_mdd_above_swap_effect_is_not_flagged():
    pooled = _minimal_pooled(swap_deviation_mean=0.02, swap_amplitude_mean=-0.006,
                              swap_amplitude_mdd=0.03, imprecision_deviation_mdd=0.05)
    load1_deviation_raw = _leaf(0.001, 0.005, sig=False)
    disclosures = build_disclosures(pooled, load1_deviation_raw, rows=[])
    d = disclosures["imprecision_detection_floor_versus_swap_effect"]
    assert d["imprecision_mdd_below_swap_effect_magnitude"] is False


def test_paired_collect_observables_pairs_deviation_and_amplitude_per_session():
    rows = [
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.1}},
                           "amplitude": {"within_item_count_level_trial_count_weighted": {"raw": -0.2}}}},
        {"status": "too_few_trials_with_a_defined_state_direction_and_report"},  # must be skipped
        {"status": "computed",
         "swap_primary": {"deviation": {"within_item_count_level_trial_count_weighted": {"raw": 0.3}},
                           "amplitude": {"within_item_count_level_trial_count_weighted": {"raw": 0.4}}}},
    ]
    a_vals, b_vals = _paired_collect_observables(rows, "swap_primary")
    assert list(a_vals) == [0.1, 0.3]
    assert list(b_vals) == [-0.2, 0.4]
