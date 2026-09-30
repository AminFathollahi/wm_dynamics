"""Tests for scripts/run_component_identity_subspace_atlas.py.

Four things could silently break this atlas without erroring: (1) the Gram-Schmidt mutual-
orthogonalisation rule 3 requires could return the wrong residual subspace, or fail to drop a
candidate whose variance is fully shared with an earlier one; (2) zero-drop accounting could
silently fail to reconcile if a corpus runner ever produced more or fewer records than sessions it
saw; (3) the reaction-time bias-only voiding rule could void (or fail to void) on the wrong
condition -- same-direction significance in BOTH the real association and its between-session
control, never a magnitude comparison; (4) the four-way branch decision (rules 2-4) could pick the
wrong branch, or the null branch could fire without every supported cell actually carrying its own
80%-power detection floor. A fifth group reproduces the frozen reproduction gate against the live
delivered artifact when the external data drive is mounted, and is skipped otherwise.

WM_DYNAMICS_DATA_ROOT must point at the currently-mounted external data drive for the data-
dependent tests in the last group; every other test here is synthetic and needs no data root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_component_identity_subspace_atlas import CORPORA, _aligned_anywhere, _bias_only_between_session_atlas, _decide_branch, _gram_schmidt_bases, _panichello_reproduction_gate, _rank_dependency_gate, _reaction_time_branch, _zero_drop_summary
from info_decoding import CANDIDATE_KEYS, CANDIDATE_SUPPORT_MATRIX

DATA_ROOT_AVAILABLE = bool(os.environ.get("WM_DYNAMICS_DATA_ROOT")) and Path(
    os.environ.get("WM_DYNAMICS_DATA_ROOT", "")
).is_dir()
requires_data_root = pytest.mark.skipif(
    not DATA_ROOT_AVAILABLE,
    reason="WM_DYNAMICS_DATA_ROOT is not set to a currently-mounted directory",
)


# ---------------------------------------------------------------------------------------------------
# _gram_schmidt_bases
# ---------------------------------------------------------------------------------------------------

def test_gram_schmidt_orthogonal_inputs_are_returned_unchanged_in_span():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((10, 4)))
    a, b = q[:, :1], q[:, 1:2]  # already mutually orthogonal, unit columns
    out = _gram_schmidt_bases([("a", a), ("b", b)])
    assert out["a"] is not None and out["b"] is not None
    # each candidate's own basis direction is preserved up to sign after orthogonalisation
    assert abs(float((out["a"].T @ a).item())) > 1.0 - 1e-8
    assert abs(float((out["b"].T @ b).item())) > 1.0 - 1e-8


def test_gram_schmidt_duplicate_basis_is_absorbed_by_the_earlier_candidate():
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((10, 3)))
    a = q[:, :1]
    out = _gram_schmidt_bases([("a", a), ("b", a.copy())])
    assert out["a"] is not None
    assert out["b"] is None  # identical direction has zero residual once "a" is regressed out


def test_gram_schmidt_order_determines_which_candidate_keeps_shared_variance():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((10, 1))
    shared /= np.linalg.norm(shared)
    unique_b = rng.standard_normal((10, 1))
    b = shared + unique_b  # partially overlaps "a"
    b /= np.linalg.norm(b)
    out_a_first = _gram_schmidt_bases([("a", shared), ("b", b)])
    out_b_first = _gram_schmidt_bases([("b", b), ("a", shared)])
    # whichever candidate comes FIRST in the order keeps its own basis direction exactly (up to
    # sign); the later one is orthogonalised against it and so changes direction.
    assert abs(float((out_a_first["a"].T @ shared).item())) > 1.0 - 1e-8
    assert abs(float((out_b_first["b"].T @ b).item())) > 1.0 - 1e-8
    assert abs(float((out_a_first["b"].T @ b).item())) < 1.0 - 1e-6  # "b" changed once "a" is regressed out
    assert abs(float((out_b_first["a"].T @ shared).item())) < 1.0 - 1e-6  # "a" changed once "b" is regressed out


# ---------------------------------------------------------------------------------------------------
# _zero_drop_summary
# ---------------------------------------------------------------------------------------------------

def test_zero_drop_summary_reconciles_when_every_record_is_computed_or_refused():
    records = [
        {"session": "s1", "status": "computed"},
        {"session": "s2", "status": "computed"},
        {"session": "s3", "status": "not_computable", "reason": "too few trials"},
        {"session": "s4", "status": "not_computable", "reason": "too few trials"},
        {"session": "s5", "status": "not_computable", "reason": "no units"},
    ]
    summary = _zero_drop_summary(records)
    assert summary["n_seen"] == 5
    assert summary["n_analysed"] == 2
    assert summary["n_refused"] == 3
    assert summary["refusals_by_reason"] == {"too few trials": 2, "no units": 1}
    assert summary["reconciles"] is True


def test_zero_drop_summary_reconciles_on_empty_corpus():
    summary = _zero_drop_summary([])
    assert summary == {"n_seen": 0, "n_loaded": 0, "n_refused": 0, "refusals_by_reason": {},
                        "n_analysed": 0, "reconciles": True}


def test_rank_dependency_gate_stops_on_unidentified_rank(tmp_path):
    path = tmp_path / "rank.json"
    path.write_text('{"status":"complete","branch":{"a":"not_determinable_because_rank_selection_is_nonidentified_or_disagrees"}}')
    result = _rank_dependency_gate(path)
    assert result["status"] == "failed"


def test_rank_dependency_gate_clears_identified_branches(tmp_path):
    path = tmp_path / "rank.json"
    path.write_text('{"status":"complete","branch":{"a":"identified","b":"identified"}}')
    result = _rank_dependency_gate(path)
    assert result["status"] == "cleared"


# ---------------------------------------------------------------------------------------------------
# _bias_only_between_session_atlas / _reaction_time_branch -- voiding rule
# ---------------------------------------------------------------------------------------------------

def _rt_session_records(rs, means_dev, means_out):
    return [
        {"status": "computed", "r": r, "mean_deviation": md, "mean_outcome": mo}
        for r, md, mo in zip(rs, means_dev, means_out)
    ]


def test_bias_only_between_session_needs_at_least_four_sessions():
    records = _rt_session_records([0.1, 0.2, 0.3], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    result = _bias_only_between_session_atlas(records, "test|bias_only")
    assert result["status"] == "not_computable"
    assert result["n_sessions"] == 3


def test_bias_only_between_session_computes_a_between_session_correlation():
    rng = np.random.default_rng(3)
    means_dev = rng.standard_normal(8)
    means_out = means_dev * 2.0 + rng.standard_normal(8) * 0.01  # strong positive between-session link
    records = _rt_session_records(np.zeros(8), means_dev, means_out)
    result = _bias_only_between_session_atlas(records, "test|bias_only|strong")
    assert result["status"] == "computed"
    assert result["n_sessions"] == 8
    assert result["r"] > 0.9


def test_reaction_time_branch_voids_only_when_control_agrees_in_sign_and_is_significant():
    # Real association: strong, consistent positive per-session r across 8 sessions -> significant.
    # Bias-only control: session means constructed to correlate STRONGLY and POSITIVELY too, in the
    # amount the real, sign-flip-pooled effect can be voided by (same direction, significant).
    rng = np.random.default_rng(4)
    n_sessions = 8
    r_values = 0.5 + rng.standard_normal(n_sessions) * 0.02  # consistently positive, low spread
    means_dev = rng.standard_normal(n_sessions)
    means_out = means_dev * 3.0 + rng.standard_normal(n_sessions) * 0.01
    records = _rt_session_records(r_values, means_dev, means_out)
    branch = _reaction_time_branch(records, "test|reaction_time|voided")
    assert branch["real_association"]["status"] == "tested"
    assert branch["real_association"]["significant"] is True
    assert branch["bias_only_between_session_control"]["status"] == "computed"
    assert branch["bias_only_between_session_control"]["p_value"] < 0.05
    same_sign = (branch["real_association"]["mean_value"] > 0) == (branch["bias_only_between_session_control"]["r"] > 0)
    assert same_sign
    assert branch["voided_by_bias_only_control"] is True


def test_reaction_time_branch_does_not_void_when_control_is_not_significant():
    rng = np.random.default_rng(5)
    n_sessions = 8
    r_values = 0.5 + rng.standard_normal(n_sessions) * 0.02
    # Session means carry no between-session relationship: control should not be significant.
    means_dev = rng.standard_normal(n_sessions)
    means_out = rng.standard_normal(n_sessions)
    records = _rt_session_records(r_values, means_dev, means_out)
    branch = _reaction_time_branch(records, "test|reaction_time|not_voided")
    assert branch["real_association"]["significant"] is True
    assert branch["voided_by_bias_only_control"] is False


def test_reaction_time_branch_does_not_void_on_opposite_sign_control():
    rng = np.random.default_rng(6)
    n_sessions = 8
    r_values = 0.5 + rng.standard_normal(n_sessions) * 0.02  # real association positive
    means_dev = rng.standard_normal(n_sessions)
    means_out = -means_dev * 3.0 + rng.standard_normal(n_sessions) * 0.01  # control strongly NEGATIVE
    records = _rt_session_records(r_values, means_dev, means_out)
    branch = _reaction_time_branch(records, "test|reaction_time|opposite_sign")
    assert branch["bias_only_between_session_control"]["p_value"] < 0.05
    real_positive = branch["real_association"]["mean_value"] > 0
    control_positive = branch["bias_only_between_session_control"]["r"] > 0
    assert real_positive != control_positive
    assert branch["voided_by_bias_only_control"] is False


# ---------------------------------------------------------------------------------------------------
# _decide_branch -- rules 2, 3, 4
# ---------------------------------------------------------------------------------------------------

def _synthetic_alignment_atlas(aligned_map: dict[tuple[str, str], bool], every_floor_computed: bool = True) -> dict:
    atlas = {}
    for corpus_key in CORPORA:
        atlas[corpus_key] = {}
        for key in CANDIDATE_KEYS:
            supported = CANDIDATE_SUPPORT_MATRIX[corpus_key][key][0] == "present"
            if not supported:
                atlas[corpus_key][key] = {"status": "absent_from_this_corpus", "reason": "not supported"}
                continue
            floor_status = "computed" if every_floor_computed else "not_computable"
            atlas[corpus_key][key] = {
                "pooled": {"status": "tested"},
                "detection_floor_80pct_power": {"status": floor_status},
                "aligned": aligned_map.get((corpus_key, key), False),
            }
    return atlas


def test_decide_branch_resolves_to_a_single_subspace_when_exactly_one_candidate_type_aligns():
    atlas = _synthetic_alignment_atlas({("panichello_2024_macaque_lPFC", "memorandum_content"): True})
    aligned_anywhere = _aligned_anywhere(atlas)
    branch = _decide_branch(atlas, aligned_anywhere, root=None)
    assert branch["branch"] == "component_identity_resolved_to_a_single_labelled_subspace"
    assert branch["resolved_candidate"] == "memorandum_content"
    assert branch["corpora_aligned_in"] == ["panichello_2024_macaque_lPFC"]


def test_decide_branch_is_null_when_nothing_aligns_and_every_floor_is_present():
    atlas = _synthetic_alignment_atlas({}, every_floor_computed=True)
    aligned_anywhere = _aligned_anywhere(atlas)
    branch = _decide_branch(atlas, aligned_anywhere, root=None)
    assert branch["branch"] == "component_is_not_aligned_with_any_labelled_subspace_tested"


def test_decide_branch_is_inconclusive_when_a_supported_cell_is_missing_its_detection_floor():
    atlas = _synthetic_alignment_atlas({}, every_floor_computed=False)
    aligned_anywhere = _aligned_anywhere(atlas)
    branch = _decide_branch(atlas, aligned_anywhere, root=None)
    assert branch["branch"] == "inconclusive_below_detection_floor"
    assert len(branch["cells_missing_a_detection_floor"]) > 0


def test_decide_branch_multiply_aligned_only_orthogonalises_corpora_with_two_or_more_own_aligned_candidates(monkeypatch):
    import run_component_identity_subspace_atlas as m

    # macaque prefrontal spatial working-memory corpus (Dryad doi:10.5061/dryad.kkwh70sct): two of its own candidates align (triggers Gram-Schmidt for this corpus only).
    # macaque multi-object working-memory corpus (doi:10.64898/2026.01.27.702062): one candidate aligns, but it is macaque multi-object corpus' only aligned candidate (no orthogonalisation).
    atlas = _synthetic_alignment_atlas({
        ("panichello_2024_macaque_lPFC", "memorandum_content"): True,
        ("panichello_2024_macaque_lPFC", "gain_total_spike_count"): True,
        ("watters_2026_macaque_multi_object", "time_in_trial_within_delay"): True,
    })
    aligned_anywhere = _aligned_anywhere(atlas)

    monkeypatch.setattr(m, "_orthogonalized_alignment_for_corpus",
                         lambda corpus_key, root, aligned_keys: {k: {"stub": True} for k in aligned_keys})
    branch = _decide_branch(atlas, aligned_anywhere, root=None)
    assert branch["branch"] == "component_identity_is_multiply_aligned"
    assert set(branch["aligned_candidate_types"]) == {
        "memorandum_content", "gain_total_spike_count", "time_in_trial_within_delay"}
    assert list(branch["per_corpus_mutual_orthogonalisation"].keys()) == ["panichello_2024_macaque_lPFC"]
    ortho = branch["per_corpus_mutual_orthogonalisation"]["panichello_2024_macaque_lPFC"]
    assert set(ortho["own_aligned_candidates"]) == {"memorandum_content", "gain_total_spike_count"}
    # Gram-Schmidt order must follow the CANDIDATE_KEYS tuple order, not discovery order.
    assert ortho["gram_schmidt_order"] == [
        k for k in CANDIDATE_KEYS if k in ("memorandum_content", "gain_total_spike_count")
    ]


# ---------------------------------------------------------------------------------------------------
# Reproduction gate (rule 0) -- needs the live delivered artifact and the mounted data drive.
# ---------------------------------------------------------------------------------------------------

@requires_data_root
def test_reproduction_gate_reproduces_the_delivered_pooled_mean_value():
    import run_component_identity_subspace_atlas as m

    root = m.data_root()
    gate = _panichello_reproduction_gate(root)
    assert gate["status"] == "reproduced_exactly"
    assert gate["n_sessions_reachable"] == 11
    assert gate["absolute_difference"] < 1e-6
    assert gate["session_ids_match_delivered_artifact"] is True
