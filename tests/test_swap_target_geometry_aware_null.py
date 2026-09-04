"""Tests for scripts/run_swap_target_geometry_aware_null.py -- the pieces
that could silently break this analysis without erroring, on synthetic data
whose correct answer is known by construction:

(1) a planted swap-to-preceding-item preference is detected by the
    percentile test against the statistic's own realised shuffle null;
(2) symmetric synthetic data (no preference) passes through the centre of
    its realised null -- not significant, observed inside the central mass;
(3) on that same symmetric data the realised null itself is centred: its
    mean sits at the symmetric expectation within tolerance and its two
    draw-index halves agree within Monte-Carlo error;
(4) the mis-centred-stop branch is reachable: draws whose halves disagree
    beyond Monte-Carlo error fail the centring check, and the headline
    decider returns the stop branch when handed that failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_swap_target_geometry_aware_null import (  # noqa: E402
    BLOCK_A_MDD_POWERED_NULL_THRESHOLD, BRANCH_A_TARGET, BRANCH_B_NOT_SEPARABLE, BRANCH_MISCENTRED_STOP,
    BRANCH_TOP_DOES_NOT_FOLLOW, BRANCH_TOP_FOLLOWS, BRANCH_TOP_INCONCLUSIVE, CENTRING_Z_THRESHOLD,
    PERCENTILE_ALPHA, STAT_RESPONSE, STAT_TARGET, _build_session_arrays, _collect_observed_and_nulls,
    _decide_block_a, _decide_block_b, _decide_top_branch, _indicators, _percentile_result,
    _session_statistics, _split_half_centring,
)

N_SESSIONS = 8
TRIALS_PER_SESSION = 80
TEST_DRAWS = 400


def _synthetic_sessions(mode: str, seed: int = 11):
    """Sessions with the task's own geometry -- two uncued objects a fixed 120 degrees apart --
    under two regimes for the preceding trial's target angle: 'planted' places it next to the swap
    destination (a real preference); 'symmetric' leaves it independent of the display."""
    rng = np.random.default_rng(seed)
    arrays, names = [], []
    for s in range(N_SESSIONS):
        swap_theta = rng.uniform(0.0, 2.0 * np.pi, TRIALS_PER_SESSION)
        other_theta = swap_theta + 2.0 * np.pi / 3.0
        if mode == "planted":
            # wide enough that both indicator groups occur in every session
            prev_target = swap_theta + rng.normal(0.0, np.radians(40.0), TRIALS_PER_SESSION)
        elif mode == "symmetric":
            prev_target = rng.uniform(0.0, 2.0 * np.pi, TRIALS_PER_SESSION)
        else:
            raise ValueError(mode)
        prev_response = prev_target + rng.normal(0.0, np.radians(10.0), TRIALS_PER_SESSION)
        # serial pull carries no group structure except under 'planted_block_b'
        pull = rng.normal(0.0, 0.01, TRIALS_PER_SESSION)
        spike = rng.poisson(12.0, TRIALS_PER_SESSION).astype(float)
        if mode == "planted":
            ind = _indicators(swap_theta, other_theta, prev_target)
            pull = pull + np.where(ind == 1.0, 0.05, 0.0)
        arrays.append(_build_session_arrays(swap_theta, other_theta, prev_target, prev_response,
                                            pull=pull, spike=spike))
        names.append(f"synthetic_{s}")
    return arrays, names


def _run_percentile_pipeline(arrays, names, draws: int = TEST_DRAWS):
    observed, nulls = _collect_observed_and_nulls(arrays, names, draws, "test|pipeline")
    per_session = [{name: row[name] for name in observed} for row in
                   (_session_statistics(sa) for sa in arrays)]
    results = {name: _percentile_result(observed[name], nulls[name],
                                        [row[name] for row in per_session]) for name in observed}
    return results, nulls


# ---------------------------------------------------------------------------------------------------
# (1) planted preference detected
# ---------------------------------------------------------------------------------------------------

def test_planted_preference_is_detected():
    arrays, names = _synthetic_sessions("planted")
    results, _ = _run_percentile_pipeline(arrays, names)
    target = results[STAT_TARGET]
    assert target["status"] == "estimated"
    assert target["observed"] > 0.9
    assert target["significant"] and target["p_two_sided_percentile"] < PERCENTILE_ALPHA
    assert target["direction"] == "above"
    assert target["outside_central_mass_q05_q95"]
    top = _decide_top_branch(True, target, "swaps_land_on_the_object_nearest_the_preceding_trials_remembered_item",
                             target["minimum_detectable_paired_difference_at_80pct_power"].get("mdd"),
                             "irrelevant")
    assert top["branch"] == BRANCH_TOP_FOLLOWS


def test_planted_serial_pull_group_difference_is_detected():
    arrays, names = _synthetic_sessions("planted")
    results, _ = _run_percentile_pipeline(arrays, names)
    raw = results["serial_pull_group_difference"]
    assert raw["status"] == "estimated"
    assert raw["significant"] and raw["direction"] == "above"


# ---------------------------------------------------------------------------------------------------
# (2) no preference passes through the centre
# ---------------------------------------------------------------------------------------------------

def test_symmetric_data_passes_through_the_centre():
    arrays, names = _synthetic_sessions("symmetric")
    results, nulls = _run_percentile_pipeline(arrays, names)
    target = results[STAT_TARGET]
    assert target["status"] == "estimated"
    assert not target["significant"]
    assert target["p_two_sided_percentile"] > 0.3
    assert target["q5"] <= target["observed"] <= target["q95"]
    response = results[STAT_RESPONSE]
    assert not response["significant"]
    top = _decide_top_branch(True, target, "powered_null_swap_destination_is_unrelated_to_the_preceding_trial",
                             target["minimum_detectable_paired_difference_at_80pct_power"]["mdd"],
                             "irrelevant")
    assert top["branch"] in (BRANCH_TOP_DOES_NOT_FOLLOW, BRANCH_TOP_INCONCLUSIVE)


def test_percentile_p_value_is_bounded_and_additive():
    rng = np.random.default_rng(5)
    draws = rng.normal(0.0, 1.0, 500).tolist()
    for observed in (-4.0, 0.0, 4.0):
        result = _percentile_result(observed, draws, [])
        assert result["status"] == "estimated"
        assert 0.0 < result["p_two_sided_percentile"] <= 1.0


# ---------------------------------------------------------------------------------------------------
# (3) realised null centring on symmetric data
# ---------------------------------------------------------------------------------------------------

def test_realised_null_is_centred_on_symmetric_data():
    arrays, names = _synthetic_sessions("symmetric")
    _, nulls = _run_percentile_pipeline(arrays, names)
    for name in (STAT_TARGET, STAT_RESPONSE, "serial_pull_group_difference",
                 "rate_control_group_difference", "bias_only_control_group_difference"):
        clean = np.asarray([d for d in nulls[name] if d is not None])
        if name in (STAT_TARGET, STAT_RESPONSE):
            # two-point Voronoi cells on a circle are equal halves: the null mean sits at one half
            assert abs(clean.mean() - 0.5) < 0.02, name
        check = _split_half_centring(nulls[name])
        assert check["status"] == "checked"
        assert check["z_half_disagreement"] <= CENTRING_Z_THRESHOLD, name
        assert check["centred_within_monte_carlo_error"], name


def test_indicator_geometry_matches_two_point_voronoi_expectation():
    # with objects at 0 and 120 degrees, exactly half the circle is nearer each object
    rng = np.random.default_rng(3)
    swap = np.zeros(20000)
    other = np.full(20000, 2.0 * np.pi / 3.0)
    ref = rng.uniform(0.0, 2.0 * np.pi, 20000)
    ind = _indicators(swap, other, ref)
    assert abs(np.mean(ind) - 0.5) < 0.01


# ---------------------------------------------------------------------------------------------------
# (4) mis-centred-stop branch reachable
# ---------------------------------------------------------------------------------------------------

def test_split_half_check_fails_on_disagreeing_halves():
    draws = [0.40 + 1e-4 * k for k in range(300)] + [0.60 + 1e-4 * k for k in range(300)]
    check = _split_half_centring(draws)
    assert check["status"] == "checked"
    assert not check["centred_within_monte_carlo_error"]
    assert check["z_half_disagreement"] > CENTRING_Z_THRESHOLD


def test_headline_decider_returns_stop_branch_when_centring_fails():
    pct = {"status": "estimated", "significant": True, "direction": "above", "observed": 0.9,
           "null_mean": 0.5}
    top = _decide_top_branch(False, pct, BRANCH_A_TARGET, None, BRANCH_B_NOT_SEPARABLE)
    assert top["branch"] == BRANCH_MISCENTRED_STOP


def test_block_deciders_stay_inside_pre_declared_cells():
    def pct(significant, direction, observed=0.6, null_mean=0.5):
        return {"status": "estimated", "significant": significant, "direction": direction,
                "observed": observed, "null_mean": null_mean,
                "observed_minus_null_mean": observed - null_mean,
                "minimum_detectable_paired_difference_at_80pct_power":
                    {"status": "computed", "mdd": 0.01}}

    # target above and response not -> the remembered-item cell
    fine_a = _decide_block_a(pct(True, "above"), pct(False, "not_significant"), pct(False, "not_significant"),
                             [0.6] * 10, [0.5] * 10)
    assert fine_a["branch"] == BRANCH_A_TARGET

    # either statistic below its null centre -> avoidance
    assert _decide_block_a(pct(True, "below"), pct(False, "not_significant"), pct(False, "not_significant"),
                           [0.4] * 10, [0.5] * 10)["branch"] == "swaps_avoid_the_preceding_trials_item"

    # neither significant with a small minimum detectable difference -> powered null
    powered = _decide_block_a(pct(False, "not_significant"), pct(False, "not_significant"),
                              pct(False, "not_significant"), [0.5] * 10, [0.5] * 10)
    assert powered["branch"] == "powered_null_swap_destination_is_unrelated_to_the_preceding_trial"
    assert powered["target_minimum_detectable_paired_difference"] < BLOCK_A_MDD_POWERED_NULL_THRESHOLD

    # a leave-one-out control that fires makes the association inseparable from a session offset
    bias = pct(True, "above")
    fine_b = _decide_block_b(pct(False, "not_significant"), pct(False, "not_significant"), bias,
                             pct(False, "not_significant"), reference_effect=1e9)
    assert fine_b["branch"] == BRANCH_B_NOT_SEPARABLE
    top = _decide_top_branch(True, pct(False, "not_significant"), powered["branch"], 0.01,
                             fine_b["branch"])
    assert top["branch"] == "not_separable_from_a_session_level_offset"

    # raw significant below its null centre -> the opposite-direction cell
    assert _decide_block_b(pct(True, "below"), pct(False, "not_significant"), pct(False, "not_significant"),
                           pct(False, "not_significant"), reference_effect=1e9)["branch"] == \
        "serial_pull_is_larger_on_swaps_that_avoid_the_preceding_trials_item"
