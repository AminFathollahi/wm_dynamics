"""Tests for scripts/run_macaque_maintenance_behaviour_link.py.

Covers the logic this module adds on top of the reused estimator
(src/stimulation_response_estimator.py, exercised directly in
tests/test_stimulation_response_gate_and_panel.py): the trialsequence
condition-code decode used by the temporal validity probe, the probe's
Bonferroni gate, the branch classifier's honest handling of the two-cluster
p-floor, and the correct/error file condition-alignment guards in
_session_arm_outcome_windows. Real-data sweeps (the trap audit and the full
behaviour link) are gated on WM_DYNAMICS_DATA_ROOT being set, matching the
convention used elsewhere in this suite (see tests/test_watters_state_geometry.py)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_macaque_maintenance_behaviour_link as link  # noqa: E402
from stimulation_response_estimator import window_treatment_effect  # noqa: E402


# ── reuse discipline: this module must not reimplement the estimator ────────

def test_estimator_functions_are_imported_not_reimplemented():
    from stimulation_response_estimator import (
        cluster_bootstrap_pooled_effect, nuisance_treatment_effect, window_treatment_effect as wte,
    )
    assert link.window_treatment_effect is wte
    assert link.nuisance_treatment_effect is nuisance_treatment_effect
    assert link.cluster_bootstrap_pooled_effect is cluster_bootstrap_pooled_effect


# ── trialsequence condition-code decode ──────────────────────────────────────

def test_trial_ranks_by_condition_decodes_c_major_and_preserves_order():
    """1-based code = stim_cond * n_angle + angle_idx + 1 (c-major), the
    encoding verified against every session's own grid-derived counts in the
    real data (see the module docstring). Ranks for one condition must be
    strictly increasing file positions, and every occurrence must be
    accounted for."""
    n_angle = 4
    # condition (stim_cond=0, angle=2) -> code 3; (stim_cond=1, angle=0) -> code 5
    trialsequence = np.array([3, 5, 3, 1, 3, 5])
    ranks = link._trial_ranks_by_condition(trialsequence, n_angle)
    assert ranks[(0, 2)] == [0, 2, 4]
    assert ranks[(1, 0)] == [1, 5]
    assert ranks[(0, 0)] == [3]
    assert sum(len(v) for v in ranks.values()) == len(trialsequence)


# ── validity probe: Bonferroni gate ─────────────────────────────────────────

def _fake_session_result(control_ranks, stim_ranks):
    return {"status": "computed", "correct_ranks": {"control_condition": control_ranks, "stimulated_condition": stim_ranks}}


def test_validity_probe_not_rejected_for_a_genuinely_uniform_spread():
    rng = np.random.default_rng(0)
    session_results = {
        f"s{i}": _fake_session_result(rng.uniform(size=40).tolist(), rng.uniform(size=40).tolist())
        for i in range(11)
    }
    probe = link.run_subsample_temporal_validity_probe(session_results)
    assert probe["validity_assumption_status"] == "not_rejected"
    assert probe["n_cells_bonferroni_significant"] == 0


def test_validity_probe_compromised_when_a_cell_is_clustered_at_session_start():
    rng = np.random.default_rng(1)
    session_results = {
        f"s{i}": _fake_session_result(rng.uniform(size=40).tolist(), rng.uniform(size=40).tolist())
        for i in range(11)
    }
    # one cell's correct trials are all crammed into the first 5% of the session -- a clear violation
    session_results["s0"]["correct_ranks"]["control_condition"] = list(np.linspace(0.0, 0.04, 40))
    probe = link.run_subsample_temporal_validity_probe(session_results)
    assert probe["validity_assumption_status"] == "compromised"
    assert probe["n_cells_bonferroni_significant"] >= 1


def test_validity_probe_flags_too_few_trials_rather_than_computing_a_test():
    session_results = {"s0": _fake_session_result([0.1, 0.2, 0.3], [0.5])}
    probe = link.run_subsample_temporal_validity_probe(session_results)
    assert probe["per_session_arm_cell"]["s0|control_condition"]["status"] == "too_few_trials_for_probe"
    assert probe["per_session_arm_cell"]["s0|stimulated_condition"]["status"] == "too_few_trials_for_probe"
    assert probe["n_cells_tested"] == 0
    assert probe["validity_assumption_status"] == "not_rejected"  # nothing to reject with zero tested cells


# ── branch classifier: the two-cluster p-floor is never mislabelled as power ─

def test_classify_cell_branch_names_never_say_powered_alone():
    """The branch name for a cell whose mdd clears the reference must make
    the p-floor limitation explicit in its own name, not just in prose
    elsewhere -- so a reader who only sees the branch key is not misled."""
    pooled = {"status": "computed", "n_clusters": 2, "mean_value": 0.1,
              "mdd": {"status": "computed", "mdd": 0.5}}
    branch, below_own_mdd = link._classify_cell(pooled)
    assert branch == "mdd_clears_reference_but_p_is_uninformative_at_two_clusters"
    assert "powered" not in branch  # the bare word implies a reachable significance test, which this is not
    assert below_own_mdd is True  # 0.1 < mdd 0.5


def test_classify_cell_inconclusive_when_mdd_exceeds_reference():
    pooled = {"status": "computed", "n_clusters": 2, "mean_value": 0.5,
              "mdd": {"status": "computed", "mdd": 2.0}}
    branch, below_own_mdd = link._classify_cell(pooled)
    assert branch == "inconclusive"
    assert below_own_mdd is True


def test_classify_cell_inconclusive_with_fewer_than_two_clusters():
    pooled = {"status": "computed", "n_clusters": 1, "mean_value": 0.5, "mdd": {"status": "not_computable"}}
    branch, below_own_mdd = link._classify_cell(pooled)
    assert branch == "inconclusive"
    assert below_own_mdd is None


def test_classify_cell_inconclusive_when_pooled_not_computable():
    branch, below_own_mdd = link._classify_cell({"status": "not_computable"})
    assert branch == "inconclusive"
    assert below_own_mdd is None


# ── window_treatment_effect reused with correct trials as the reference pool ─

def test_correct_pool_forms_the_reference_error_pool_is_scored_against_it():
    """The mapping this module uses (correct trials play window_treatment_effect's
    reference-forming 'control' role, error trials are scored against the
    fixed reference built from them) must have the same same-trial-reference-
    leakage guarantee the panel's own tests establish for the treatment
    contrast: changing one error trial's value must never move the reference
    or any correct trial's own deviation."""
    rng = np.random.default_rng(2)
    correct = rng.normal(size=(40, 5)) + np.array([1.0, 0, 0, 0, 0])
    error_a = rng.normal(size=(15, 5)) + np.array([0, 1.0, 0, 0, 0])
    error_b = error_a.copy()
    error_b[0] = [500.0, -500.0, 250.0, -250.0, 100.0]
    result_a = window_treatment_effect(correct, error_a)
    result_b = window_treatment_effect(correct, error_b)
    assert result_a["status"] == "computed"
    assert result_a["spontaneous_control_sd"] == result_b["spontaneous_control_sd"]


# ── zero-drop / condition-alignment guards in _session_arm_outcome_windows ──

def test_session_arm_outcome_windows_excludes_on_control_idx_mismatch(monkeypatch):
    def fake_loader(prefix, correct):
        if correct:
            return {"trials": [], "control_idx": 0, "stim_channels": [[], [1]], "channel_ids": np.array([1, 2])}
        return {"trials": [], "control_idx": 1, "stim_channels": [[], [1]], "channel_ids": np.array([1, 2])}
    monkeypatch.setattr(link, "load_macaque_pfc_microstimulation_session", fake_loader)
    res, reason = link._session_arm_outcome_windows("fake_session", pre_bins=16)
    assert res is None
    assert reason == "correct_and_error_control_idx_mismatch"


def test_session_arm_outcome_windows_excludes_on_condition_ordering_mismatch(monkeypatch):
    def fake_loader(prefix, correct):
        stim_channels = [[], [1]] if correct else [[], [2]]  # different token at the same index
        return {"trials": [], "control_idx": 0, "stim_channels": stim_channels, "channel_ids": np.array([1, 2])}
    monkeypatch.setattr(link, "load_macaque_pfc_microstimulation_session", fake_loader)
    res, reason = link._session_arm_outcome_windows("fake_session", pre_bins=16)
    assert res is None
    assert reason == "correct_and_error_condition_ordering_mismatch"


def test_session_arm_outcome_windows_excludes_on_channel_id_mismatch(monkeypatch):
    def fake_loader(prefix, correct):
        channel_ids = np.array([1, 2]) if correct else np.array([1, 3])
        return {"trials": [], "control_idx": 0, "stim_channels": [[], [1]], "channel_ids": channel_ids}
    monkeypatch.setattr(link, "load_macaque_pfc_microstimulation_session", fake_loader)
    res, reason = link._session_arm_outcome_windows("fake_session", pre_bins=16)
    assert res is None
    assert reason == "correct_and_error_channel_id_mismatch"


def test_session_arm_outcome_windows_excludes_when_no_error_file(monkeypatch):
    call_count = {"n": 0}

    def fake_loader(prefix, correct):
        call_count["n"] += 1
        if correct:
            return {"trials": [], "control_idx": 0, "stim_channels": [[]], "channel_ids": np.array([1])}
        return None
    monkeypatch.setattr(link, "load_macaque_pfc_microstimulation_session", fake_loader)
    res, reason = link._session_arm_outcome_windows("fake_session", pre_bins=16)
    assert res is None
    assert reason == "no_usable_error_trial_file"
    assert call_count["n"] == 2  # both files attempted


# ── real-data-dependent sweeps ───────────────────────────────────────────────

DATA_AVAILABLE = bool(os.environ.get("WM_DYNAMICS_DATA_ROOT")) and Path(
    os.environ.get("WM_DYNAMICS_DATA_ROOT", "")
).exists()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set or not mounted")
def test_trial_admission_trap_audit_reconciles_against_the_shared_loader():
    """The naive error fraction the brief warns against must be computed
    (for disclosure) but the correct-count coefficient of variation must be
    far smaller than the error-count one, in every session -- the trap the
    audit exists to detect and report."""
    audit = link.run_trial_admission_trap_audit()
    assert audit["sessions_computed"] == 11
    for prefix, s in audit["per_session"].items():
        assert s["status"] == "computed"
        assert s["behaviour_record_count_reconciles_with_loader"] is True
        if s["correct_count_by_cell_coefficient_of_variation"] is not None and s["error_count_by_cell_coefficient_of_variation"]:
            assert s["correct_count_by_cell_coefficient_of_variation"] < s["error_count_by_cell_coefficient_of_variation"]
    assert set(audit["bad_trial_handling_values_seen_across_all_sessions"]) == {"reshuffle"}


@pytest.mark.skipif(not DATA_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set or not mounted")
def test_session_arm_outcome_windows_reconciles_trial_counts_for_a_real_session():
    res, reason = link._session_arm_outcome_windows("Wa220804_s552", pre_bins=16)
    assert res is not None, reason
    counts = res["counts"]
    assert counts["correct_seen"] == counts["correct_used"] + counts["correct_excluded_short_crop"]
    assert counts["error_seen"] == counts["error_used"] + counts["error_excluded_short_crop"]
    n_correct_total = sum(
        res["pools"][win][arm]["correct_activity"].shape[0]
        for win in ("pre_stimulation_window",) for arm in ("control_condition", "stimulated_condition")
    )
    assert n_correct_total == counts["correct_used"]


@pytest.mark.skipif(not DATA_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set or not mounted")
def test_delivered_artifact_zero_drop_reconciles():
    import json
    output_path = Path(__file__).resolve().parents[1] / "results" / "macaque_maintenance_behaviour_link.json"
    if not output_path.exists():
        pytest.skip("results/macaque_maintenance_behaviour_link.json has not been generated yet")
    d = json.loads(output_path.read_text())
    zd = d["zero_drop_accounting"]
    assert zd["sessions_used"] + len(zd["sessions_excluded_by_reason"]) == zd["sessions_seen"]
    assert d["status"] == "complete"
    for cell in d["behaviour_link"].values():
        if cell.get("status") != "computed":
            continue
        dev = cell["rate_free_deviation_biomarker"]
        assert dev["branch"] in (
            "inconclusive", "validity_assumption_compromised",
            "mdd_clears_reference_but_p_is_uninformative_at_two_clusters",
        )
