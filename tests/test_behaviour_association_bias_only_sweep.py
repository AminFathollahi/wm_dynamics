"""Tests for run_behaviour_association_bias_only_sweep.py.

Split in two: fast unit tests of the sweep's own pure helper functions
(bootstrap MDD, voiding-rule logic, sign convention, census bookkeeping),
which run with synthetic data and no I/O; and integration tests against the
actually-delivered results/behaviour_association_bias_only_sweep.json (once
that run has completed), which check the artifact's own internal
consistency rather than recomputing anything expensive a second time."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_behaviour_association_bias_only_sweep as sweep  # noqa: E402

ARTIFACT_PATH = ROOT / "results" / "behaviour_association_bias_only_sweep.json"


# ============================================================================
# Pure helper unit tests
# ============================================================================

def test_close_handles_none_and_tolerance():
    assert sweep._close(None, None) is True
    assert sweep._close(None, 1.0) is False
    assert sweep._close(1.0000001, 1.0000002, tol=1e-6) is True
    assert sweep._close(1.0, 1.1, tol=1e-6) is False


def test_direction_signs():
    assert sweep._direction(0.5) == "positive"
    assert sweep._direction(-0.5) == "negative"
    assert sweep._direction(0.0) == "zero"
    assert sweep._direction(None) is None


def test_void_verdict_requires_significance_and_same_sign():
    # both significant, same sign -> voided
    v = sweep._void_verdict(True, "negative", True, "negative")
    assert v["voided"] is True
    # both significant, opposite sign -> not voided
    v = sweep._void_verdict(True, "negative", True, "positive")
    assert v["voided"] is False
    # bias-only not significant -> never voided regardless of sign
    v = sweep._void_verdict(True, "negative", False, "negative")
    assert v["voided"] is False
    # real not significant -> never voided
    v = sweep._void_verdict(False, "negative", True, "negative")
    assert v["voided"] is False


def test_bootstrap_mdd_from_mean_matches_analytic_scale():
    """A whole-session cluster bootstrap of a simple mean should land close
    to the analytic sd/sqrt(n) scale (this project's existing
    minimum_detectable_paired_difference formula) for a reasonably-sized,
    well-behaved sample -- not identical (it is a genuine resampling
    procedure, not the analytic formula), but the same order of magnitude."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.1, scale=0.2, size=20).tolist()
    result = sweep._bootstrap_mdd_from_mean(values, "test|bootstrap_mean", n_boot=1000)
    assert result["status"] == "computed"
    analytic = float(np.std(values, ddof=1)) / np.sqrt(len(values))
    # bootstrap SE should be within a factor of 2 of the analytic SE for n=20
    assert 0.5 * analytic < result["bootstrap_se"] < 2.0 * analytic
    assert result["mdd"] == pytest.approx(sweep.MDD_Z_FACTOR * result["bootstrap_se"])


def test_bootstrap_mdd_from_mean_too_few_sessions():
    result = sweep._bootstrap_mdd_from_mean([0.1], "test|too_few")
    assert result["status"] == "not_computable"


def test_bootstrap_mdd_from_between_session_correlation_recovers_strong_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=15)
    y = 2.0 * x + rng.normal(scale=0.01, size=15)  # near-deterministic relationship
    result = sweep._bootstrap_mdd_from_between_session_correlation(x.tolist(), y.tolist(), "test|bias_mdd")
    assert result["status"] == "computed"
    # a near-perfect linear relationship should bootstrap to a small SE and therefore a small MDD
    assert result["mdd"] < 0.3


def test_bootstrap_mdd_from_between_session_correlation_too_few_sessions():
    result = sweep._bootstrap_mdd_from_between_session_correlation([0.1, 0.2], [0.3, 0.4], "test|too_few2")
    assert result["status"] == "not_computable"


def test_build_at_risk_census_counts_sum_to_total():
    census = sweep.build_at_risk_census()
    counts = census["counts"]
    assert sum(counts.values()) == census["total_files"]
    assert census["sums_to_total"] is True
    for key in ("at_risk", "not_at_risk", "already_controlled", "undetermined_from_artifact_contents"):
        assert key in counts


def test_at_risk_census_classifies_the_headline_claim_as_at_risk():
    census = sweep.build_at_risk_census()
    entry = census["entries"]["rate_free_state_geometry_behavior_link.json"]
    assert entry["classification"] == "at_risk"


def test_at_risk_census_every_entry_has_a_classification_and_reason():
    census = sweep.build_at_risk_census()
    allowed = {"at_risk", "not_at_risk", "already_controlled", "undetermined_from_artifact_contents"}
    for filename, entry in census["entries"].items():
        assert entry["classification"] in allowed, filename
        assert isinstance(entry["reason"], str) and entry["reason"], filename


def test_claim_behavior_geometry_link_finds_no_pooled_significance_test():
    result = sweep.claim_behavior_geometry_link()
    assert result["branch"] == "not_at_risk_no_pooled_significance_test_in_this_artifact"


def test_decision_rule_states_voiding_conditions_before_any_fit():
    rule = sweep.DECISION_RULE_DECLARED_BEFORE_FITTING
    assert "significant" in rule
    assert "same sign" in rule or "SAME SIGN" in rule.upper()
    assert str(sweep.MDD_Z_FACTOR) in rule


# ============================================================================
# Integration tests against the delivered artifact (skipped if not yet run)
# ============================================================================

pytestmark_skip_if_missing = pytest.mark.skipif(
    not ARTIFACT_PATH.exists(), reason="results/behaviour_association_bias_only_sweep.json has not been produced yet")


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip("results/behaviour_association_bias_only_sweep.json has not been produced yet")
    return json.loads(ARTIFACT_PATH.read_text())


def test_artifact_declares_its_decision_rule_before_fitting(artifact):
    assert "decision_rule_declared_before_fitting" in artifact
    assert len(artifact["decision_rule_declared_before_fitting"]) > 200


def test_artifact_status_is_complete(artifact):
    assert artifact["status"] == "complete"


def test_artifact_at_risk_census_reconciles(artifact):
    census = artifact["at_risk_census"]
    assert census["sums_to_total"] is True
    assert sum(census["counts"].values()) == census["total_files"]


def test_headline_claim_has_a_reproduction_gate_and_a_recognised_branch(artifact):
    headline = artifact["claims"]["rate_free_state_geometry_behavior_link"]
    assert "reproduction_gate" in headline
    allowed_branches = {
        sweep.BRANCH_GATE_FAILED, sweep.BRANCH_NOT_SIGNIFICANT, sweep.BRANCH_VOIDED,
        sweep.BRANCH_INCONCLUSIVE, sweep.BRANCH_NOT_VOIDED,
    }
    assert headline["branch"] in allowed_branches


def test_headline_claim_reproduction_gate_reproduced_the_delivered_artifact(artifact):
    """This is the load-bearing check: the whole sweep's verdict on the
    headline is meaningless if the reproduction gate itself failed silently
    without being reported. If the gate legitimately failed, this test
    documents that fact rather than hiding it -- it asserts the STATUS
    FIELD is honestly one of the two allowed values, not that it passed."""
    headline = artifact["claims"]["rate_free_state_geometry_behavior_link"]
    gate = headline["reproduction_gate"]
    assert gate["status"] in ("reproduced_exactly", "not_reproduced")
    if gate["status"] == "not_reproduced":
        assert headline["branch"] == sweep.BRANCH_GATE_FAILED


def test_headline_voided_only_if_real_and_bias_both_significant_same_sign(artifact):
    headline = artifact["claims"]["rate_free_state_geometry_behavior_link"]
    if headline["branch"] != sweep.BRANCH_VOIDED:
        pytest.skip("headline was not voided in this run")
    voiding = headline["voiding"]
    assert voiding["raw"]["voided"] or voiding["joint_partial_controlling_spike_count_and_trial_index"]["voided"]


def test_zero_drop_accounting_present_and_reconciles_for_headline(artifact):
    headline = artifact["claims"]["rate_free_state_geometry_behavior_link"]
    zd = headline["zero_drop_accounting"]
    assert zd["reconciles"] is True
    assert zd["n_seen"] == zd["n_loaded"] + zd["n_refused"]


def test_summary_branch_names_headline_verdict_explicitly(artifact):
    """Coordinator-authorised amendment: headline_dissociation_still_holds (a bare boolean asserting a
    conclusion) and the old headline_verdict_statement were replaced because the boolean asserted
    'still holds' beside a branch that explicitly declined to conclude that (inconclusive_below_
    detection_floor). The replacement fields must state the branch's own content and nothing beyond it,
    and must NOT bring back a bare 'still holds'/'confirmed' boolean."""
    summary = artifact["summary_branch"]
    assert "headline_verdict_statement" in summary
    assert "headline_voided_by_this_sweep" in summary
    assert "headline_dissociation_still_holds" not in summary
    assert isinstance(summary["headline_verdict_statement"], str) and len(summary["headline_verdict_statement"]) > 20
    if artifact["claims"]["rate_free_state_geometry_behavior_link"]["branch"] == sweep.BRANCH_INCONCLUSIVE:
        statement = summary["headline_verdict_statement"].lower()
        assert "untested" in statement
        assert summary["headline_voided_by_this_sweep"] is False


def test_priority_order_completion_is_honestly_reported(artifact):
    summary = artifact["summary_branch"]
    assert summary["n_claims_completed"] + summary["n_claims_not_reached"] == summary["n_claims_in_priority_order"]
    assert summary["n_claims_completed"] == 10
    assert summary["n_claims_not_reached"] == 0


def test_sessions_required_extrapolation_matches_bias_only_mdd(artifact):
    """The new sessions-required field on claim 4's reused headline cell must be a genuine
    extrapolation from that same cell's own already-computed bias-only bootstrap MDD, not an
    independent guess."""
    cell = artifact["claims"]["dominant_latent_identity_and_behaviour_breadth"]["primary_floor_rate_free_deviation"]
    power = cell["power"]
    bias_mdd = power["bias_only_control_minimum_detectable_difference_80pct_power"]
    sessions_required = power["sessions_required_for_the_control_to_reach_the_meaningful_effect_threshold"]
    assert sessions_required["status"] == "extrapolated"
    expected = bias_mdd["n_sessions"] * ((sweep.MDD_Z_FACTOR * bias_mdd["bootstrap_se"]) / 0.14) ** 2
    assert sessions_required["n_sessions_required"] == pytest.approx(expected, rel=1e-9)


def test_claim_8_cites_sibling_and_is_voided_without_a_power_block(artifact):
    claim8 = artifact["claims"]["component_and_item_binding"]
    assert claim8["branch"] == sweep.BRANCH_VOIDED
    assert "power" not in claim8  # voided cells carry no MDD under this sweep's own rule
    assert claim8["real"]["raw"] is not None


def test_at_risk_census_completeness_note_is_honest_about_coverage(artifact):
    note = artifact["at_risk_census_completeness_note"]
    assert note["this_sweep_is_not_a_completed_audit"] is True
    total = (note["priority_at_risk_files_fully_resolved_this_round"]
             + note["priority_at_risk_files_partially_resolved_this_round"]
             + note["priority_at_risk_files_not_independently_tested_this_round"])
    assert total == note["at_risk_files_named_in_this_sweep_priority_order"]
    assert note["at_risk_files_outside_this_sweeps_priority_order_never_examined"] == (
        note["total_at_risk_file_level"] - note["at_risk_files_named_in_this_sweep_priority_order"])
