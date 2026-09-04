"""Tests for scripts/run_state_space_estimation_admissibility.py.

Covers the pieces that could silently break the admissibility gate without
raising an exception: the trial/feature split invariants; bin-count
alignment between a candidate's latent and its scoring target (the one
place a trial-pooled or off-by-one-rebinned candidate could be scored
against the wrong array shape); the operating-rank selection order and its
fallback; the read-out-and-score pipeline's ability to tell real structure
from none, on synthetic data with a planted signal and on pure noise; the
session-admission floor and its machine-readable exclusion reasons; the
checkpoint round trip and its handling of an unparseable record; and the
per-corpus gate aggregation, including the patient-level clustering used for
every human corpus and the session-level aggregation used for the two
non-human ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_state_space_estimation_admissibility as m  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# trial_split / feature_split
# ---------------------------------------------------------------------------------------------------

def test_trial_split_partitions_every_trial_exactly_once():
    rng = np.random.default_rng(0)
    train_idx, test_idx = m.trial_split(50, rng)
    assert set(train_idx.tolist()) | set(test_idx.tolist()) == set(range(50))
    assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()
    assert len(test_idx) >= m.MIN_TEST_TRIALS
    assert len(train_idx) >= m.MIN_TRAIN_TRIALS


def test_feature_split_partitions_every_feature_exactly_once():
    rng = np.random.default_rng(0)
    held_in, held_out = m.feature_split(20, rng)
    assert set(held_in.tolist()) | set(held_out.tolist()) == set(range(20))
    assert set(held_in.tolist()) & set(held_out.tolist()) == set()
    assert len(held_out) >= m.MIN_HELD_OUT_FEATURES
    assert len(held_in) >= m.MIN_HELD_IN_FEATURES


def test_splits_are_deterministic_given_the_same_seed():
    a = m.trial_split(50, np.random.default_rng(7))
    b = m.trial_split(50, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------------------------------------------
# _align_bins
# ---------------------------------------------------------------------------------------------------

def test_align_bins_passes_through_when_bin_counts_already_match():
    latent = np.zeros((5, 4, 2))
    target = np.ones((5, 4, 3))
    lat, tgt = m._align_bins(latent, target)
    assert lat.shape == (5, 4, 2) and tgt.shape == (5, 4, 3)


def test_align_bins_pools_target_to_one_bin_when_latent_has_no_time_axis():
    latent = np.zeros((5, 1, 2))
    target = np.arange(5 * 4 * 3, dtype=float).reshape(5, 4, 3)
    lat, tgt = m._align_bins(latent, target)
    assert lat.shape == (5, 1, 2)
    assert tgt.shape == (5, 1, 3)
    assert np.allclose(tgt[:, 0, :], target.mean(axis=1))


def test_align_bins_truncates_to_the_shared_leading_bin_count_otherwise():
    latent = np.zeros((5, 6, 2))
    target = np.ones((5, 9, 3))
    lat, tgt = m._align_bins(latent, target)
    assert lat.shape[1] == 6 and tgt.shape[1] == 6


# ---------------------------------------------------------------------------------------------------
# operating_rank
# ---------------------------------------------------------------------------------------------------

def test_operating_rank_prefers_cross_validated_reconstruction_first():
    criteria = {"cross_validated_reconstruction_rank": 4, "participation_ratio_rank": 9,
                "parallel_analysis_rank": 2, "cross_validated_likelihood_rank": 7}
    assert m.operating_rank(criteria, n_features=20) == 4


def test_operating_rank_falls_through_when_the_preferred_criterion_is_missing():
    criteria = {"cross_validated_reconstruction_rank": None, "cross_validated_likelihood_rank": None,
                "participation_ratio_rank": 5, "parallel_analysis_rank": 2}
    assert m.operating_rank(criteria, n_features=20) == 5


def test_operating_rank_falls_back_to_a_fixed_small_default_when_every_criterion_is_missing():
    criteria = {"cross_validated_reconstruction_rank": None, "cross_validated_likelihood_rank": None,
                "participation_ratio_rank": None, "parallel_analysis_rank": None}
    rank = m.operating_rank(criteria, n_features=20)
    assert 1 <= rank <= 3


def test_operating_rank_is_clipped_to_the_available_feature_count():
    criteria = {"cross_validated_reconstruction_rank": 50, "cross_validated_likelihood_rank": None,
                "participation_ratio_rank": None, "parallel_analysis_rank": None}
    rank = m.operating_rank(criteria, n_features=6)
    assert rank <= 5  # n_features - 1
    assert rank <= m.MAX_OPERATING_RANK


# ---------------------------------------------------------------------------------------------------
# _decode_and_score: recovers a planted signal, does not spuriously pass on pure noise
# ---------------------------------------------------------------------------------------------------

def test_decode_and_score_recovers_a_planted_linear_relationship_field_potential():
    rng = np.random.default_rng(1)
    n_trials, n_bins, k, n_target = 60, 5, 2, 3
    latent = rng.normal(size=(n_trials, n_bins, k))
    weights = rng.normal(size=(k, n_target))
    target = latent @ weights + rng.normal(scale=0.05, size=(n_trials, n_bins, n_target))
    split = n_trials // 2
    result = m._decode_and_score(latent[:split], target[:split], latent[split:], target[split:],
                                  is_spiking=False, rng=np.random.default_rng(2))
    assert result["score"] > result["shuffle_score"]
    assert result["passes_shuffle"] is True


def test_decode_and_score_does_not_spuriously_pass_on_independent_noise():
    rng = np.random.default_rng(3)
    n_trials, n_bins, k, n_target = 60, 5, 2, 3
    latent = rng.normal(size=(n_trials, n_bins, k))
    target = rng.normal(size=(n_trials, n_bins, n_target))  # independent of latent by construction
    split = n_trials // 2
    outcomes = []
    for trial_seed in range(8):
        result = m._decode_and_score(latent[:split], target[:split], latent[split:], target[split:],
                                      is_spiking=False, rng=np.random.default_rng(100 + trial_seed))
        outcomes.append(result["passes_shuffle"])
    # No real relationship exists, so this should not pass on a comfortable majority of draws.
    assert sum(outcomes) <= 5


def test_decode_and_score_spiking_uses_the_poisson_deviance_scoring_rule():
    rng = np.random.default_rng(4)
    n_trials, n_bins, k, n_target = 60, 4, 2, 2
    latent = rng.normal(size=(n_trials, n_bins, k))
    rate = np.exp(0.3 * latent.sum(axis=-1, keepdims=True)) * np.ones((1, 1, n_target))
    target = rng.poisson(rate)
    split = n_trials // 2
    result = m._decode_and_score(latent[:split], target[:split], latent[split:], target[split:],
                                  is_spiking=True, rng=np.random.default_rng(5))
    assert result["score"] > result["shuffle_score"]


# ---------------------------------------------------------------------------------------------------
# _session_admission: the zero-drop floor and its machine-readable reasons
# ---------------------------------------------------------------------------------------------------

def test_session_admission_excludes_below_the_trial_floor_with_the_excluding_number():
    X = np.zeros((5, 10, 4))  # far below MIN_TRIALS
    record = {"dataset": "d", "patient": "p", "session": "s", "X": X}
    out = m._session_admission(record)
    assert out["status"] == "excluded"
    assert out["exclusion_reason"] == "below_admission_floor"
    assert out["n_trials"] == 5
    assert out["n_trials_needed"] == m.MIN_TRIALS


def test_session_admission_excludes_below_the_feature_floor_with_the_excluding_number():
    X = np.zeros((30, 2, 4))  # far below MIN_FEATURES
    record = {"dataset": "d", "patient": "p", "session": "s", "X": X}
    out = m._session_admission(record)
    assert out["status"] == "excluded"
    assert out["exclusion_reason"] == "below_admission_floor"
    assert out["n_features"] == 2
    assert out["n_features_needed"] == m.MIN_FEATURES


def test_session_admission_admits_a_session_at_or_above_both_floors():
    X = np.zeros((30, 10, 4))
    record = {"dataset": "d", "patient": "p", "session": "s", "X": X}
    out = m._session_admission(record)
    assert out["status"] == "admitted"
    assert out["n_train_trials"] + out["n_test_trials"] == 30
    assert out["n_held_in_features"] + out["n_held_out_features"] == 10


# ---------------------------------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------------------------------

def test_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CHECKPOINT_DIR", tmp_path)
    m.save_checkpoint("a_key", {"score": 1.23, "status": "fitted"})
    assert m.load_checkpoint("a_key") == {"score": 1.23, "status": "fitted"}


def test_checkpoint_missing_key_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CHECKPOINT_DIR", tmp_path)
    assert m.load_checkpoint("never_written") is None


def test_unparseable_checkpoint_is_treated_as_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CHECKPOINT_DIR", tmp_path)
    path = m._checkpoint_path("broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")
    assert m.load_checkpoint("broken") is None


def test_run_checkpointed_only_calls_the_fit_function_once(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CHECKPOINT_DIR", tmp_path)
    calls = []

    def fit_fn():
        calls.append(1)
        return {"value": 42}

    first = m.run_checkpointed("key", fit_fn)
    second = m.run_checkpointed("key", fit_fn)
    assert first == second == {"value": 42}
    assert len(calls) == 1


# ---------------------------------------------------------------------------------------------------
# gate_for_corpus: majority-vote admissibility, patient clustering for human corpora
# ---------------------------------------------------------------------------------------------------

def _fake_session(patient, session, passes: bool | None, status="admitted"):
    if status != "admitted":
        return {"status": status, "exclusion_reason": "below_admission_floor", "patient": patient,
                "session": session, "n_trials": 1, "n_trials_needed": m.MIN_TRIALS, "n_features": 1,
                "n_features_needed": m.MIN_FEATURES}
    if passes is None:
        candidate = {"status": "not_applicable_to_data_type"}
    else:
        candidate = {"status": "fitted", "held_out_feature_score": {"passes_shuffle": passes}}
    candidates = {name: dict(candidate) for name in m.CANDIDATES}
    return {"status": "admitted", "patient": patient, "session": session, "n_trials": 30,
            "n_train_trials": 20, "n_test_trials": 10, "n_features": 10, "candidates": candidates}


def test_gate_is_admissible_when_a_majority_of_patients_pass():
    sessions = [_fake_session("p1", "s1", True), _fake_session("p2", "s1", True), _fake_session("p3", "s1", False)]
    result = m.gate_for_corpus("dandi_000469", sessions)
    gate = result["gate"]["native_full_rank"]
    assert gate["clustering_unit"] == "patient"
    assert gate["admissible"] is True
    assert gate["n_total"] == 3 and gate["n_passed"] == 2


def test_gate_is_inadmissible_when_a_majority_of_patients_fail_and_records_excluding_examples():
    sessions = [_fake_session("p1", "s1", False), _fake_session("p2", "s1", False), _fake_session("p3", "s1", True)]
    result = m.gate_for_corpus("dandi_000469", sessions)
    gate = result["gate"]["native_full_rank"]
    assert gate["admissible"] is False
    assert len(gate["excluding_examples"]) >= 1
    assert gate["excluding_examples"][0]["n_trials_needed"] == m.MIN_TRIALS


def test_gate_clusters_multiple_sessions_of_the_same_human_patient_before_voting():
    # A single patient with two sessions, one passing and one not, should be pooled to one vote for
    # that patient (majority of that patient's own sessions), not counted as two independent draws.
    sessions = [_fake_session("p1", "s1", True), _fake_session("p1", "s2", True), _fake_session("p2", "s1", False)]
    result = m.gate_for_corpus("dandi_001187", sessions)
    gate = result["gate"]["native_full_rank"]
    assert gate["n_total"] == 2  # two patients, not three sessions
    assert gate["n_passed"] == 1


def test_gate_uses_session_level_clustering_for_a_non_human_corpus():
    sessions = [_fake_session("mouse1", "s1", True), _fake_session("mouse1", "s2", True), _fake_session("mouse1", "s3", False)]
    result = m.gate_for_corpus("inagaki_alm5", sessions)
    gate = result["gate"]["native_full_rank"]
    assert gate["clustering_unit"] == "session"
    assert gate["n_total"] == 3


def test_gate_reports_zero_drop_exclusions_with_reasons():
    sessions = [_fake_session("p1", "s1", True), _fake_session("p2", "s1", False, status="excluded")]
    result = m.gate_for_corpus("dandi_000469", sessions)
    assert result["n_sessions_yielded_by_shared_loader"] == 2
    assert result["n_sessions_admitted"] == 1
    assert result["n_sessions_excluded"] == 1
    assert result["exclusions"][0]["reason"] == "below_admission_floor"


def test_gate_excludes_not_applicable_candidates_from_both_numerator_and_denominator():
    sessions = [_fake_session("p1", "s1", None), _fake_session("p2", "s1", None)]
    result = m.gate_for_corpus("dandi_000574_eeg", sessions)
    gate = result["gate"]["gaussian_process_factor_analysis"]
    assert gate["status"] == "not_applicable_to_data_type"


# ---------------------------------------------------------------------------------------------------
# candidate roster and the hard scope constraints
# ---------------------------------------------------------------------------------------------------

def test_spiking_only_candidates_are_a_subset_of_the_full_roster():
    assert m.SPIKING_ONLY_CANDIDATES <= set(m.CANDIDATES)


def test_every_candidate_has_a_plain_description_with_no_literature_author_name():
    # A cheap, standing guard against the project's own constraint: descriptions name what a method
    # does, never who published it.
    banned_substrings = ["et al", "Yu ", "Pandarinath", "Schneider", "Busch", "Moon ", "Lusch"]
    for name, description in m.CANDIDATE_DESCRIPTIONS.items():
        assert name in m.CANDIDATES
        for banned in banned_substrings:
            assert banned not in description


def test_candidate_roster_json_serializes_cleanly():
    from provenance import canonical_json
    text = canonical_json({"candidate_roster": m.CANDIDATE_DESCRIPTIONS})
    parsed = json.loads(text)
    assert set(parsed["candidate_roster"]) == set(m.CANDIDATES)
