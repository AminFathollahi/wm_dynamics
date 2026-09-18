"""Tests for scripts/run_component_binding_bias_only_control.py.

Three things could silently break this analysis without erroring: (1) the
multi-level generalisation of the bias-only mask picking the wrong trials
(too many, too few, or the wrong levels) for a session that qualifies at
only one of the two levels the pooled statistic is combined across; (2) the
bias-only statistic silently collapsing to (or being replaced by) the REAL
per-session, sign-flip-pooled statistic -- the exact failure mode that would
make a voided association look like it survived; (3) the voiding rule
(sign and significance only, never magnitude) misfiring. A fourth group
checks the delivered artifact itself for internal consistency, reading it
live rather than re-deriving its numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_component_binding_bias_only_control import (  # noqa: E402
    BRANCH_BELOW_FLOOR,
    BRANCH_SURVIVES,
    BRANCH_VOIDED,
    LEVELS_2_AND_3,
    MIN_SESSIONS_FOR_BETWEEN_SESSION_CORRELATION,
    OUTPUT_PATH,
    _bias_only_between_session_generalized,
    _direction,
    _qualifying_levels,
    _void_verdict,
)


# ---------------------------------------------------------------------------------------------------
# _qualifying_levels
# ---------------------------------------------------------------------------------------------------

def _row(outcome_key: str, level_status: dict[int, str]) -> dict:
    return {"status": "computed", outcome_key: {"deviation": {"per_level": {
        str(lv): {"status": status} for lv, status in level_status.items()}}}}


def test_qualifying_levels_reads_only_per_level_computed_status():
    row = _row("swap_primary", {2: "computed", 3: "too_few_trials_at_this_item_count"})
    assert _qualifying_levels(row, "swap_primary") == [2]


def test_qualifying_levels_returns_both_when_both_computed():
    row = _row("swap_primary", {2: "computed", 3: "computed"})
    assert _qualifying_levels(row, "swap_primary") == [2, 3]


def test_qualifying_levels_empty_when_neither_computed():
    row = _row("imprecision", {2: "too_few_trials_at_this_item_count", 3: "too_few_trials_at_this_item_count"})
    assert _qualifying_levels(row, "imprecision") == []


# ---------------------------------------------------------------------------------------------------
# _bias_only_between_session_generalized -- level-selection correctness
# ---------------------------------------------------------------------------------------------------

def _arrays(item_counts, deviation, outcome) -> dict:
    n = len(item_counts)
    return {
        "item_count": np.asarray(item_counts, dtype=float),
        "deviation": np.asarray(deviation, dtype=float),
        "swap_primary": np.asarray(outcome, dtype=float),
        "spike_count": np.zeros(n), "trial_index": np.arange(n, dtype=float),
    }


def test_bias_only_generalized_pools_only_qualifying_levels_and_matches_hand_computed_correlation():
    """Twenty sessions (the between-session correlation this module builds is computed by _corr,
    imported unchanged, which itself refuses fewer than MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION=16
    points -- reused here exactly as it is reused on the real corpus, where 41 sessions clear it
    easily), hand-designed so every session's combined-level (or single-level, for one session that
    qualifies at only one of the two levels) mean of deviation and outcome is known exactly. The
    function's returned r must equal a plain numpy correlation over those known means, or the
    level-selection logic (which levels are pooled, and whether a non-qualifying level's trials leak
    in) is wrong."""
    rows, rows_arrays = [], {}
    expected_dev_means, expected_out_means = [], []

    # Session A: qualifies at levels 2 and 3. Level-1 trials are present but must never be touched
    # (the selector never returns level 1 for the swap outcome).
    key = "A"
    rows.append(_row("swap_primary", {2: "computed", 3: "computed"}) | {"session": key})
    rows_arrays[key] = _arrays(
        item_counts=[1, 1, 1] + [2] * 10 + [3] * 10,
        deviation=[999, 999, 999] + [2.0] * 10 + [4.0] * 10,   # combined mean = 3.0
        outcome=[999, 999, 999] + [5.0] * 10 + [9.0] * 10,      # combined mean = 7.0
    )
    expected_dev_means.append(3.0)
    expected_out_means.append(7.0)

    # Session B: qualifies at level 2 ONLY. Level-3 trials are physically present with extreme values
    # that must be EXCLUDED -- if the selector (or the mask it drives) pooled every level present in
    # the arrays instead of only the qualifying ones, these would drag the mean far off target.
    key = "B"
    rows.append(_row("swap_primary", {2: "computed", 3: "too_few_trials_at_this_item_count"}) | {"session": key})
    rows_arrays[key] = _arrays(
        item_counts=[2] * 8 + [3] * 3,
        deviation=[1.0] * 8 + [500.0] * 3,   # level-2-only mean = 1.0
        outcome=[2.0] * 8 + [500.0] * 3,     # level-2-only mean = 2.0
    )
    expected_dev_means.append(1.0)
    expected_out_means.append(2.0)

    # Eighteen more sessions, qualifying at both levels, with arbitrary but exactly known combined
    # means -- bulk needed only to clear _corr's own trial-count floor, reused unchanged here.
    for i in range(18):
        key = f"G{i}"
        dev_mean, out_mean = i * 0.37 - 3.0, i * 0.51 + 1.0
        rows.append(_row("swap_primary", {2: "computed", 3: "computed"}) | {"session": key})
        rows_arrays[key] = _arrays(item_counts=[2] * 6 + [3] * 6,
                                    deviation=[dev_mean] * 12, outcome=[out_mean] * 12)
        expected_dev_means.append(dev_mean)
        expected_out_means.append(out_mean)

    def selector(row):
        return _qualifying_levels(row, "swap_primary")

    result = _bias_only_between_session_generalized(
        rows, rows_arrays, "swap_primary", selector, (), "test|identity")

    assert result["status"] == "computed"
    assert result["n_sessions"] == 20
    expected_r = float(np.corrcoef(np.array(expected_out_means), np.array(expected_dev_means))[0, 1])
    assert result["r"] == pytest.approx(expected_r, abs=1e-9)


def test_bias_only_generalized_below_session_floor_is_not_computable():
    rows, rows_arrays = [], {}
    for i in range(3):  # fewer than MIN_SESSIONS_FOR_BETWEEN_SESSION_CORRELATION
        key = f"S{i}"
        rows.append(_row("swap_primary", {2: "computed"}) | {"session": key})
        rows_arrays[key] = _arrays([2] * 20, np.random.default_rng(i).normal(size=20),
                                    np.random.default_rng(i + 100).normal(size=20))

    def selector(row):
        return _qualifying_levels(row, "swap_primary")

    result = _bias_only_between_session_generalized(
        rows, rows_arrays, "swap_primary", selector, (), "test|floor")
    assert result["status"] == "not_computable"
    assert result["n_sessions"] == 3 < MIN_SESSIONS_FOR_BETWEEN_SESSION_CORRELATION


# ---------------------------------------------------------------------------------------------------
# The critical guard: the bias-only statistic must be computed on the collapsed session means, never
# on (nor silently equal to) the real per-trial, per-session statistic -- this is the failure mode that
# would silently fake a survival.
# ---------------------------------------------------------------------------------------------------

def test_bias_only_generalized_does_not_detect_a_purely_within_session_effect():
    """A strong, real, WITHIN-session relationship (deviation predicts outcome trial-by-trial, slope 5,
    substantial per-trial noise but a strong per-session r all the same) built so every session's own
    MEAN deviation is EXACTLY 0.0 (a symmetric linspace, no per-session drift added to it) -- there is
    no between-session variation in the predictor to correlate with anything, by construction, even
    though the real per-trial relationship is real and strong. If the bias-only control were silently
    computing (or reproducing) the real per-trial statistic here, it would report a strong, significant
    effect (real per-trial r is strong at 40 trials/session, pooled over 20 sessions). The actual
    bias-only substitution collapses each session to its own mean first: since every session's mean
    deviation is identically 0.0, the between-session predictor has EXACTLY zero variance across
    sessions, and the underlying partial-correlation estimator (_corr, imported unchanged) must refuse
    to compute a correlation against a zero-variance residual -- not_computable, deterministically."""
    rows, rows_arrays = [], {}
    for i in range(20):  # >= _corr's own MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION floor of 16
        key = f"S{i}"
        rows.append(_row("swap_primary", {2: "computed"}) | {"session": key})
        rng = np.random.default_rng(i)
        deviation = np.linspace(-1.0, 1.0, 40)  # mean EXACTLY 0.0 for every session, no exceptions
        outcome = 5.0 * deviation + rng.normal(scale=1.0, size=40)  # real within-session slope, real noise
        rows_arrays[key] = _arrays([2] * 40, deviation, outcome)

    def selector(row):
        return _qualifying_levels(row, "swap_primary")

    result = _bias_only_between_session_generalized(
        rows, rows_arrays, "swap_primary", selector, (), "test|within_session_only")
    assert result["status"] == "not_computable"


def test_bias_only_generalized_detects_a_purely_between_session_effect():
    """The mirror case: pure trial-level NOISE within every session (no real within-session
    relationship at all) but each session's own mean deviation and mean outcome are set on a perfect
    line -- exactly the session-level-offset confound the bias-only control exists to catch. This must
    be detected (large |r|, significant), confirming the control is sensitive to between-session
    structure specifically, not to the real within-session effect it is being checked against."""
    rows, rows_arrays = [], {}
    session_means = [(i - 10.0, 2.0 * (i - 10.0) + 1.0) for i in range(20)]  # perfect line, N=20 >= 16
    for i, (dev_mean, out_mean) in enumerate(session_means):
        key = f"S{i}"
        rows.append(_row("swap_primary", {2: "computed"}) | {"session": key})
        rng = np.random.default_rng(i + 1000)
        deviation = dev_mean + rng.normal(scale=0.01, size=40)
        outcome = rng.permutation(out_mean + rng.normal(scale=0.01, size=40))  # shuffled: no real link
        rows_arrays[key] = _arrays([2] * 40, deviation, outcome)

    def selector(row):
        return _qualifying_levels(row, "swap_primary")

    result = _bias_only_between_session_generalized(
        rows, rows_arrays, "swap_primary", selector, (), "test|between_session_only")
    assert result["status"] == "computed"
    assert result["r"] > 0.9
    assert result["significant"] is True


# ---------------------------------------------------------------------------------------------------
# _direction / _void_verdict
# ---------------------------------------------------------------------------------------------------

def test_direction_signs():
    assert _direction(0.5) == "positive"
    assert _direction(-0.5) == "negative"
    assert _direction(0.0) == "zero"
    assert _direction(None) is None


def test_void_verdict_voids_when_both_significant_and_same_sign():
    real = {"status": "tested", "significant": True, "direction": "positive"}
    bias = {"status": "computed", "significant": True, "direction": "positive"}
    v = _void_verdict(real, bias)
    assert v["voided"] is True


def test_void_verdict_does_not_void_when_bias_only_not_significant():
    real = {"status": "tested", "significant": True, "direction": "positive"}
    bias = {"status": "computed", "significant": False, "direction": "positive"}
    v = _void_verdict(real, bias)
    assert v["voided"] is False


def test_void_verdict_does_not_void_when_signs_differ():
    real = {"status": "tested", "significant": True, "direction": "positive"}
    bias = {"status": "computed", "significant": True, "direction": "negative"}
    v = _void_verdict(real, bias)
    assert v["voided"] is False


def test_void_verdict_never_voids_an_already_nonsignificant_real_result():
    real = {"status": "tested", "significant": False, "direction": "positive"}
    bias = {"status": "computed", "significant": True, "direction": "positive"}
    v = _void_verdict(real, bias)
    assert v["voided"] is False


# ---------------------------------------------------------------------------------------------------
# Artifact internal consistency -- read the delivered artifact live, never re-derive its numbers.
# ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not OUTPUT_PATH.exists():
        pytest.skip("results/component_binding_bias_only_control.json has not been produced yet")
    return json.loads(OUTPUT_PATH.read_text())


def test_artifact_reproduction_gate_reproduced_exactly(artifact):
    assert artifact["reproduction_gate"]["status"] == "reproduced_exactly"
    assert all(artifact["reproduction_gate"]["checks"].values())


def test_artifact_zero_drop_reconciles_against_delivered_counterpart(artifact):
    zd = artifact["zero_drop_accounting"]
    assert zd["reconciles"] is True
    assert zd["reconciles_against_delivered_artifact"] is True
    assert zd["n_seen"] == zd["n_loaded"] + zd["n_refused"]


def test_artifact_diagnostic_cross_check_matches_delivered_sibling_control(artifact):
    diagnostic = artifact["diagnostic_cross_check_against_delivered_sibling_single_level_control"]
    for level in ("2", "3"):
        assert diagnostic[level]["matches_at_tolerance"] is True


def test_artifact_bias_only_statistic_is_not_the_real_effect_size(artifact):
    """Guards the exact failure mode named in this module's brief: if the bias-only pathway were
    accidentally wired to the real per-session-pooled statistic instead of the collapsed-session-mean
    one, the two numbers would come out identical. They are different estimators on different scales
    by construction, so they must differ materially wherever both are defined."""
    cell = artifact["swap_primary_deviation_within_item_count_level"]["raw"]
    real_mean = cell["real"]["mean_value"]
    bias_mean = cell["bias_only_between_session"]["mean_value"]
    assert bias_mean is not None and real_mean is not None
    assert abs(bias_mean - real_mean) > 0.05


def test_artifact_branch_is_one_of_the_declared_outcomes(artifact):
    assert artifact["branch"]["branch"] in (BRANCH_SURVIVES, BRANCH_VOIDED, BRANCH_BELOW_FLOOR)


def test_artifact_branch_matches_its_own_recorded_voiding_flags(artifact):
    """Recomputes the branch decision from the artifact's own recorded per-cell voiding verdicts
    (never from a fresh fit) and checks it against the recorded branch -- guards the branch-assembly
    logic in main() drifting from the voiding cells it is supposed to summarise."""
    swap = artifact["swap_primary_deviation_within_item_count_level"]
    raw_real_sig = swap["raw"]["real"]["significant"]
    joint_real_sig = swap["joint_partial_controlling_spike_count_and_trial_index"]["real"]["significant"]
    if not (raw_real_sig and joint_real_sig):
        expected = BRANCH_BELOW_FLOOR
    elif swap["raw"]["voiding"]["voided"] or swap["joint_partial_controlling_spike_count_and_trial_index"][
            "voiding"]["voided"]:
        expected = BRANCH_VOIDED
    else:
        expected = BRANCH_SURVIVES
    assert artifact["branch"]["branch"] == expected


def test_artifact_levels_pooled_are_two_and_three(artifact):
    assert artifact["levels_pooled_for_the_within_item_count_level_composite"] == list(LEVELS_2_AND_3)
