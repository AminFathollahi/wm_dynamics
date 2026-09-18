"""Tests for run_occupied_subspace_membership_effect_size.py.

Covers: (1) a synthetic positive control where a direction is planted exactly inside a known
k-dimensional subspace -- within_fraction must come out at approximately 1 and the analytic chance
reference at k/p; (2) a synthetic negative control where the tested direction is drawn uniformly at
random in the full ambient space -- the pooled within_fraction must come out at approximately k/p,
which is the assertion that proves the analytic chance reference is calibrated; (3) a direct pin of the
degenerate within-subspace null's defect, so it is caught by a test rather than left to prose; plus
smaller unit checks on the whole-session cluster bootstrap and the per-corpus branch rule.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_occupied_subspace_rank_estimation import _within_subspace_null_decomposition  # noqa: E402
from run_deviation_axis_structure import _occupied_space_decomposition  # noqa: E402

import run_occupied_subspace_membership_effect_size as mod  # noqa: E402


def test_planted_axis_inside_known_subspace_recovers_within_fraction_near_one():
    rng = np.random.default_rng(0)
    n, p, k = 300, 60, 5
    U = rng.standard_normal((n, p))
    # force the top of U's own SVD basis to be axis-aligned with the first k coordinate directions by
    # inflating the variance along those coordinates so the leading-k basis is, to high approximation,
    # span(e_1..e_k); the axis is planted exactly inside that span.
    U[:, :k] *= 20.0
    axis = np.zeros(p)
    axis[:k] = rng.standard_normal(k)
    axis /= np.linalg.norm(axis)

    ambient = _occupied_space_decomposition(U, axis, k, 200, "test|planted|ambient")
    assert ambient["status"] == "computed"
    assert ambient["within_fraction"] > 0.98

    chance_reference = k / p
    assert abs(chance_reference - 5 / 60) < 1e-12
    contrast = ambient["within_fraction"] - chance_reference
    assert contrast > 0.9  # a planted-inside axis is far above the chance reference


def test_uniform_random_axis_pooled_within_fraction_matches_chance_reference():
    rng = np.random.default_rng(1)
    n, p, k = 400, 50, 8
    within_fracs = []
    for trial in range(60):
        U = rng.standard_normal((n, p))
        axis = rng.standard_normal(p)
        axis /= np.linalg.norm(axis)
        result = _occupied_space_decomposition(U, axis, k, 50, f"test|uniform|{trial}")
        assert result["status"] == "computed"
        within_fracs.append(result["within_fraction"])
    pooled_within = float(np.mean(within_fracs))
    chance_reference = k / p
    # the axis is independent of U's own data-driven basis here, so pooled within_fraction over enough
    # draws must land near k/p -- this is the assertion that calibrates the analytic reference.
    assert abs(pooled_within - chance_reference) < 0.03


def test_degenerate_within_subspace_null_has_near_zero_off_fraction_by_construction():
    rng = np.random.default_rng(2)
    n, p, k = 200, 40, 6
    U = rng.standard_normal((n, p))
    axis = rng.standard_normal(p)
    axis /= np.linalg.norm(axis)
    result = _within_subspace_null_decomposition(U, axis, k, 300, "test|degenerate")
    assert result["status"] == "computed"
    assert result["null_off_fraction_mean"] < 1e-10
    assert result["null_is_degenerate_by_construction"] is True
    # any strictly positive observed off-fraction -- true for essentially every real, non-degenerate axis
    # -- tests as "above" this null; the defect is that this comparison cannot fail.
    assert result["off_fraction"] > result["null_off_fraction_mean"]


def test_whole_session_cluster_bootstrap_ci_brackets_a_clear_positive_effect():
    values = [0.8, 0.85, 0.9, 0.82, 0.88, 0.79, 0.91]
    out = mod._whole_session_cluster_bootstrap(values, "test|bootstrap|positive")
    assert out["status"] == "computed"
    ci_low, ci_high = out["cluster_bootstrap_ci_95pct"]
    assert ci_low > 0.0
    assert ci_low < out["pooled_mean"] < ci_high
    assert out["minimum_detectable_difference_at_80pct_power"] > 0.0


def test_whole_session_cluster_bootstrap_not_computable_below_minimum_sessions():
    out = mod._whole_session_cluster_bootstrap([0.5, 0.6], "test|bootstrap|too_few")
    assert out["status"] == "not_computable"
    assert out["n_sessions"] == 2


def test_estimator_verdict_three_outcomes():
    above = {"status": "computed", "cluster_bootstrap_ci_95pct": [0.1, 0.3]}
    below = {"status": "computed", "cluster_bootstrap_ci_95pct": [-0.3, -0.1]}
    straddling = {"status": "computed", "cluster_bootstrap_ci_95pct": [-0.05, 0.2]}
    too_few = {"status": "not_computable"}
    assert mod._estimator_verdict(above) == "above_chance"
    assert mod._estimator_verdict(below) == "at_or_below_chance"
    assert mod._estimator_verdict(straddling) == "inconclusive_confidence_interval_includes_zero"
    assert mod._estimator_verdict(too_few) == "inconclusive_too_few_sessions_with_a_detected_subspace_to_pool"


def test_corpus_branch_requires_agreement_across_evaluable_estimators():
    assert mod._corpus_branch({"a": "above_chance", "b": "above_chance"}) == \
        "the_axis_lies_inside_the_occupied_space_beyond_chance_expectation"
    assert mod._corpus_branch({"a": "at_or_below_chance"}) == \
        "the_axis_does_not_exceed_chance_expectation_within_the_selected_subspace"
    assert mod._corpus_branch({"a": "above_chance", "b": "at_or_below_chance"}) == \
        "membership_relative_to_chance_is_inconclusive_given_available_power"
    assert mod._corpus_branch({}) == \
        "membership_relative_to_chance_is_inconclusive_given_available_power"


def test_pool_session_levels_is_trial_count_weighted():
    records = [
        {"n_trials": 100, "within_fraction": 0.9, "chance_reference_k_over_p": 0.2},
        {"n_trials": 300, "within_fraction": 0.5, "chance_reference_k_over_p": 0.1},
    ]
    wf, kp = mod._pool_session_levels(records)
    expected_wf = (100 * 0.9 + 300 * 0.5) / 400
    expected_kp = (100 * 0.2 + 300 * 0.1) / 400
    assert abs(wf - expected_wf) < 1e-12
    assert abs(kp - expected_kp) < 1e-12
