"""Tests for run_within_session_permutation_control.py's own generic engine and branch-decision logic
-- synthetic data only, so this suite is fast and does not depend on the raw corpora the production
cells load. The production cells themselves are exercised by actually running the script against the
real data (results/within_session_permutation_control.json), not by this suite."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_within_session_permutation_control as m  # noqa: E402
from info_decoding import _cheap_partial_r, _circular_outcome_shift, _circular_residual_shift


def _blocks_true_effect(rng, n_sessions=10, n_trials=60, beta=0.6, controls=False):
    session_blocks = []
    for _ in range(n_sessions):
        x = rng.normal(size=n_trials)
        y = beta * x + rng.normal(size=n_trials)
        block = {"n": n_trials, "predictor": x, "outcome": y,
                 "controls": [rng.normal(size=n_trials)] if controls else []}
        session_blocks.append([block])
    return session_blocks


def _blocks_null(rng, n_sessions=10, n_trials=60):
    session_blocks = []
    for _ in range(n_sessions):
        x = rng.normal(size=n_trials)
        y = rng.normal(size=n_trials)
        session_blocks.append([{"n": n_trials, "predictor": x, "outcome": y, "controls": []}])
    return session_blocks


def test_cheap_partial_r_matches_plain_pearson_with_no_controls():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=200), rng.normal(size=200)
    x = x + 0.3 * y
    got = _cheap_partial_r(y, x, [])
    expected = float(np.corrcoef(x, y)[0, 1])
    assert abs(got - expected) < 1e-9


def test_cheap_partial_r_removes_a_perfectly_confounding_control():
    """If outcome and covariate are both exactly linear functions of a control, the partial correlation
    controlling for it should be ~0."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=300)
    x = 2.0 * z + rng.normal(scale=0.001, size=300)
    y = -3.0 * z + rng.normal(scale=0.001, size=300)
    r = _cheap_partial_r(y, x, [z])
    assert abs(r) < 0.05


def test_within_session_permutation_detects_a_true_effect():
    rng = np.random.default_rng(2)
    blocks = _blocks_true_effect(rng, beta=0.6)
    result = m.within_session_permutation_test(blocks, "test|true_effect", n_perm=3000)
    assert result["status"] == "computed"
    assert result["significant"]
    assert result["direction"] == "positive"


def test_within_session_permutation_does_not_fire_on_pure_noise():
    rng = np.random.default_rng(3)
    blocks = _blocks_null(rng)
    result = m.within_session_permutation_test(blocks, "test|null", n_perm=3000)
    assert result["status"] == "computed"
    assert not result["significant"]


def test_within_session_permutation_preserves_session_means_and_sizes():
    """The whole point of this control is that session identity, session means and session sizes are
    held fixed by construction -- only the trial-to-outcome pairing inside a session moves. Checks this
    directly on one shuffled draw."""
    rng = np.random.default_rng(4)
    blocks = _blocks_true_effect(rng, n_sessions=5, n_trials=40)
    for session_blocks in blocks:
        for block in session_blocks:
            before_outcome_mean = float(np.mean(block["outcome"]))
            before_outcome_sorted = np.sort(block["outcome"])
            before_n = block["n"]
            shuffled = m._default_shuffle_block(rng, block)
            assert shuffled["n"] == before_n
            assert np.array_equal(np.sort(shuffled["outcome"]), before_outcome_sorted)
            assert abs(float(np.mean(shuffled["outcome"])) - before_outcome_mean) < 1e-9
            # predictor (the covariate) is untouched by the default shuffle
            assert np.array_equal(shuffled["predictor"], block["predictor"])


def test_circular_outcome_shift_preserves_ordered_series_up_to_rotation():
    block = {"n": 10, "predictor": np.arange(10.0), "outcome": np.arange(10.0), "controls": []}
    shifted = _circular_outcome_shift(np.random.default_rng(7), block)
    assert np.array_equal(np.sort(shifted["outcome"]), block["outcome"])
    assert not np.array_equal(shifted["outcome"], block["outcome"])


def test_circular_residual_shift_uses_existing_partial_correlation_design():
    rng = np.random.default_rng(8)
    nuisance = rng.normal(size=60)
    outcome = 3.0 * nuisance + rng.normal(size=60)
    block = {"n": 60, "predictor": rng.normal(size=60), "outcome": outcome, "controls": [nuisance]}
    shifted = _circular_residual_shift(rng, block)
    assert shifted["controls"] is block["controls"]
    assert np.array_equal(shifted["predictor"], block["predictor"])
    assert not np.array_equal(shifted["outcome"], block["outcome"])


def test_default_stat_of_session_is_trial_count_weighted_average_across_blocks():
    rng = np.random.default_rng(5)
    block_a = {"n": 100, "predictor": rng.normal(size=100), "outcome": None, "controls": []}
    block_b = {"n": 20, "predictor": rng.normal(size=20), "outcome": None, "controls": []}
    block_a["outcome"] = 0.8 * block_a["predictor"] + rng.normal(scale=0.01, size=100)
    block_b["outcome"] = -0.8 * block_b["predictor"] + rng.normal(scale=0.01, size=20)
    combined = m._default_stat_of_session([block_a, block_b])
    r_a = _cheap_partial_r(block_a["outcome"], block_a["predictor"], [])
    r_b = _cheap_partial_r(block_b["outcome"], block_b["predictor"], [])
    expected = (100 * r_a + 20 * r_b) / 120
    assert abs(combined - expected) < 1e-9


def test_pre_cue_shuffle_block_preserves_decoded_and_swaps_only_reported_and_other():
    rng = np.random.default_rng(6)
    n = 30
    block = {"n": n, "decoded": rng.uniform(-np.pi, np.pi, size=n),
             "reported": rng.uniform(-np.pi, np.pi, size=n), "other": rng.uniform(-np.pi, np.pi, size=n)}
    shuffled = m._pre_cue_shuffle_block(rng, block)
    assert np.array_equal(shuffled["decoded"], block["decoded"])
    # every trial's (reported, other) pair is either unchanged or swapped, never mixed with another trial
    same = np.isclose(shuffled["reported"], block["reported"]) & np.isclose(shuffled["other"], block["other"])
    swapped = np.isclose(shuffled["reported"], block["other"]) & np.isclose(shuffled["other"], block["reported"])
    assert np.all(same | swapped)


def test_pre_cue_stat_of_session_is_departure_from_half():
    decoded = np.array([0.0, 0.0, 0.0, 0.0])
    reported = np.array([0.0, 0.0, 3.0, 3.0])  # decoded lands on 'reported' for the first two trials
    other = np.array([3.0, 3.0, 0.0, 0.0])
    stat = m._pre_cue_stat_of_session([{"n": 4, "decoded": decoded, "reported": reported, "other": other}])
    assert abs(stat - 0.0) < 1e-9  # 2/4 trials closer to reported -> fraction 0.5 -> departure 0.0


def test_assemble_cell_branches_inconclusive_when_real_statistic_not_significant():
    cell = m._assemble_cell(
        cell_name="synthetic", corpus="synthetic", n_sessions=10, delivered_artifact="none",
        reproduction_checks={"raw": True}, gate_ok=True,
        real_by_stat={"raw": {"status": "tested", "mean_value": 0.01, "significant": False,
                               "minimum_detectable_paired_difference_at_80pct_power": {"mdd": 0.05}}},
        between_session_control_by_stat={"raw": {"mean_value": 0.3, "p_value": 0.01, "significant": True}},
        permutation_by_stat={"raw": {"status": "computed", "significant": False, "direction": "positive",
                                      "p_value": 0.4}},
        why_at_risk="synthetic test cell",
    )
    assert cell["branch"] == m.BRANCH_REAL_NOT_SIGNIFICANT


def test_assemble_cell_branches_survives_when_permutation_significant_and_same_sign():
    cell = m._assemble_cell(
        cell_name="synthetic", corpus="synthetic", n_sessions=10, delivered_artifact="none",
        reproduction_checks={"raw": True}, gate_ok=True,
        real_by_stat={"raw": {"status": "tested", "mean_value": 0.05, "significant": True,
                               "minimum_detectable_paired_difference_at_80pct_power": {"mdd": 0.05}}},
        between_session_control_by_stat={"raw": {"mean_value": 0.3, "p_value": 0.01, "significant": True}},
        permutation_by_stat={"raw": {"status": "computed", "significant": True, "direction": "positive",
                                      "p_value": 0.0001}},
        why_at_risk="synthetic test cell",
    )
    assert cell["branch"] == m.BRANCH_SURVIVES


def test_assemble_cell_branches_powered_null_when_permutation_fails_but_mdd_is_small():
    cell = m._assemble_cell(
        cell_name="synthetic", corpus="synthetic", n_sessions=10, delivered_artifact="none",
        reproduction_checks={"raw": True}, gate_ok=True,
        real_by_stat={"raw": {"status": "tested", "mean_value": 0.05, "significant": True,
                               "minimum_detectable_paired_difference_at_80pct_power": {"mdd": 0.05}}},
        between_session_control_by_stat={"raw": {"mean_value": 0.3, "p_value": 0.01, "significant": True}},
        permutation_by_stat={"raw": {"status": "computed", "significant": False, "direction": "positive",
                                      "p_value": 0.4}},
        why_at_risk="synthetic test cell",
    )
    assert cell["branch"] == m.BRANCH_POWERED_NULL


def test_assemble_cell_branches_inconclusive_underpowered_when_mdd_is_large():
    cell = m._assemble_cell(
        cell_name="synthetic", corpus="synthetic", n_sessions=10, delivered_artifact="none",
        reproduction_checks={"raw": True}, gate_ok=True,
        real_by_stat={"raw": {"status": "tested", "mean_value": 0.05, "significant": True,
                               "minimum_detectable_paired_difference_at_80pct_power": {"mdd": 0.9}}},
        between_session_control_by_stat={"raw": {"mean_value": 0.3, "p_value": 0.01, "significant": True}},
        permutation_by_stat={"raw": {"status": "computed", "significant": False, "direction": "positive",
                                      "p_value": 0.4}},
        why_at_risk="synthetic test cell",
    )
    assert cell["branch"] == m.BRANCH_INCONCLUSIVE_UNDERPOWERED


def test_assemble_cell_branches_gate_failed_when_reproduction_does_not_match():
    cell = m._assemble_cell(
        cell_name="synthetic", corpus="synthetic", n_sessions=10, delivered_artifact="none",
        reproduction_checks={"raw": False}, gate_ok=False,
        real_by_stat={}, between_session_control_by_stat={}, permutation_by_stat={},
        why_at_risk="synthetic test cell",
    )
    assert cell["branch"] == m.BRANCH_GATE_FAILED


if __name__ == "__main__":
    test_cheap_partial_r_matches_plain_pearson_with_no_controls()
    test_cheap_partial_r_removes_a_perfectly_confounding_control()
    test_within_session_permutation_detects_a_true_effect()
    test_within_session_permutation_does_not_fire_on_pure_noise()
    test_within_session_permutation_preserves_session_means_and_sizes()
    test_default_stat_of_session_is_trial_count_weighted_average_across_blocks()
    test_pre_cue_shuffle_block_preserves_decoded_and_swaps_only_reported_and_other()
    test_pre_cue_stat_of_session_is_departure_from_half()
    test_assemble_cell_branches_inconclusive_when_real_statistic_not_significant()
    test_assemble_cell_branches_survives_when_permutation_significant_and_same_sign()
    test_assemble_cell_branches_powered_null_when_permutation_fails_but_mdd_is_small()
    test_assemble_cell_branches_inconclusive_underpowered_when_mdd_is_large()
    test_assemble_cell_branches_gate_failed_when_reproduction_does_not_match()
    print("all checks passed")
