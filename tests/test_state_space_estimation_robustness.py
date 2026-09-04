"""Tests for the estimation-choice robustness ladder.

Covers the four behaviours the leg stands on: the shared fitting entry point refuses a non-null
label argument (the circularity guard), the subspace-angle path refuses a nonlinear representation,
the estimator-invariant restatement reduces to the delivered subspace-projection quantity in the
linear case, and the pre-declared escalation trigger fires on planted disagreement between
estimator cells and does not fire on planted agreement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_state_space_estimation_robustness import (  # noqa: E402
    RUNG_THREE_MAX_SESSIONS, control_cell_verdict_key, decide_claim_standing,
    dynamics_vote_resolvability, effect_cell_verdict_key, fit_representation,
    require_linear_representation, restated_claim_cell, rung_three_sample_size,
)
from run_state_space_dimensionality_sweep import (  # noqa: E402
    cross_validated_predictable_fraction, in_sample_linear_fraction,
)


RNG = np.random.default_rng(20260826)


# ── Circularity guard ─────────────────────────────────────────────────────────────

def test_fitting_entry_point_refuses_a_label_argument():
    x = RNG.standard_normal((12, 5, 4))
    labels = np.array([0, 1] * 6)
    with pytest.raises(ValueError, match="refuses a non-null label argument"):
        fit_representation("principal_components", x, 2, np.random.default_rng(0),
                           is_spiking=False, bin_ms=100.0, labels=labels)
    with pytest.raises(ValueError, match="refuses a non-null label argument"):
        fit_representation("native_full_rank", x, 2, np.random.default_rng(0),
                           is_spiking=False, bin_ms=100.0, labels=labels)


def test_time_contrastive_candidate_records_its_positive_pair_definition():
    pytest.importorskip("cebra")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10, 6, 4)).astype(float)
    out = fit_representation("time_contrastive_embedding", x, 2, rng,
                             is_spiking=False, bin_ms=100.0)
    if out.get("status") == "fitted":
        assert out["positive_pair_definition"] == "temporal_adjacency_only_no_label_of_any_kind"


# ── Subspace-angle guard ──────────────────────────────────────────────────────────

def test_angle_path_refuses_nonlinear_representations_and_passes_linear_ones():
    with pytest.raises(ValueError, match="no canonical basis"):
        require_linear_representation("temporal_diffusion_embedding")
    with pytest.raises(ValueError, match="no canonical basis"):
        require_linear_representation("time_contrastive_embedding")
    for linear in ("native_full_rank", "principal_components", "factor_analysis",
                   "gaussian_process_factor_analysis"):
        require_linear_representation(linear)  # must not raise


# ── Estimator-invariant restatement reduces in the linear case ────────────────────

def test_restatement_reduces_to_delivered_subspace_quantity_exactly_in_the_linear_case():
    """With an orthonormal design the in-sample linear restatement equals the delivered
    subspace-projection quantity (||P_S w|| / ||w||)**2 to numerical precision."""
    rng = np.random.default_rng(11)
    n, d, k = 400, 10, 3
    # An exactly orthonormal, exactly mean-free design: the leading left singular vectors of the
    # centring projector span the mean-zero subspace, so no centreing term perturbs the algebra.
    projector = np.eye(n) - np.ones((n, n)) / n
    u, _, _ = np.linalg.svd(projector)
    x = u[:, :d]
    s, _ = np.linalg.qr(rng.standard_normal((d, k)))
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    y = x @ w
    z = x @ s
    naive = in_sample_linear_fraction(y, z)
    delivered = float((np.linalg.norm((s @ s.T) @ w) / np.linalg.norm(w)) ** 2)
    assert naive["status"] == "computed"
    assert abs(naive["linear_fraction"] - delivered) < 1e-8


def test_restatement_is_defined_identically_for_an_embedding_coordinate_block():
    """The same function consumes embedding coordinates without any structural change: swapping
    the coordinate block changes nothing about how the quantity is computed."""
    rng = np.random.default_rng(13)
    n = 60
    y = rng.standard_normal(n)
    embedding_coords = rng.standard_normal((n, 4)) + 3.0 * (rng.random((n, 1)) > 0.5)
    out = cross_validated_predictable_fraction(y, embedding_coords, rng=rng)
    assert out["status"] == "computed"
    assert -1.0 <= out["predictable_fraction"] <= 1.0000001


def test_restated_claim_cell_separates_true_class_structure_from_its_permutation_null():
    """A deviation scalar genuinely driven by the class-mean structure must show a positive
    restatement effect against its own label-permutation null."""
    rng = np.random.default_rng(17)
    n_per_class, k = 25, 5
    labels = np.array([0, 1, 2] * n_per_class)
    centers = rng.standard_normal((3, k)) * 4.0
    latent = np.concatenate([centers[c] + 0.3 * rng.standard_normal((n_per_class, k))
                             for c in range(3)])
    y = latent @ rng.standard_normal(k)  # driven by the same class structure
    coords = np.linalg.svd(latent[labels == 0].mean(axis=0)[None, :], full_matrices=True)[2][:1].T
    from run_state_space_estimation_robustness import class_mean_coordinates
    coords = class_mean_coordinates(latent, labels)
    cell = restated_claim_cell(y, coords, labels, latent, "label_permutation", rng, n_perm=40)
    assert cell["status"] == "computed"
    assert cell["effect_size"] > 0


# ── Pre-declared escalation trigger ───────────────────────────────────────────────

def test_planted_agreement_across_estimators_confirms_without_escalation():
    keys = {"native_full_rank": "majority_significant_positive",
            "principal_components": "majority_significant_positive",
            "factor_analysis": "majority_significant_positive",
            "time_contrastive_embedding": "majority_significant_positive",
            "trial_level_variational_autoencoder": "not_applicable"}
    decision = decide_claim_standing(keys)
    assert decision["branch"] == "verdict_confirmed_across_estimators"
    keys_uniform = {name: "dominant_classification_damped_rotation"
                    for name in ("native_full_rank", "factor_analysis")}
    assert decide_claim_standing(keys_uniform)["branch"] == "verdict_confirmed_across_estimators"


def test_planted_disagreement_fires_the_escalation_branch_and_names_both_sides():
    keys = {"native_full_rank": "majority_significant_positive",
            "principal_components": "no_majority_effect_negative",
            "time_contrastive_embedding": "no_majority_effect_negative"}
    decision = decide_claim_standing(keys)
    assert decision["branch"] == "estimation_dependent_rung_three_escalation"
    sides = decision["estimators_by_side"]
    assert sides["majority_significant_positive"] == ["native_full_rank"]
    assert sorted(sides["no_majority_effect_negative"]) == ["principal_components",
                                                            "time_contrastive_embedding"]


def test_not_applicable_cells_do_not_drive_agreement():
    keys = {"native_full_rank": "targeting_alignment_excess_negative",
            "trial_level_variational_autoencoder": "not_applicable",
            "temporal_diffusion_embedding": "not_computable"}
    decision = decide_claim_standing(keys)
    assert decision["branch"] == "verdict_confirmed_across_estimators"


# ── Verdict-key resolvability (sign of an undetectable mean must not drive escalation) ─────────

def test_opposite_signed_unresolvable_means_produce_the_same_verdict_key_and_do_not_escalate():
    """Two estimators that both fail majority-significance AND both fail to clear their own paired
    minimum detectable difference must collapse to the same 'no_majority_effect' key even though
    their mean effects sit on opposite sides of zero -- reproducing the exact shape of the delivered
    defect (e.g. cross_temporal_generalization's -0.00561/mdd 0.02915 against +0.0146/some larger
    mdd): neither mean is distinguishable from zero, so the sign is not part of either verdict."""
    agg_a = {"status": "computed", "fraction_significant_p_below_0p05": 0.2,
             "mean_effect_size": -0.0056, "minimum_detectable_difference": {"status": "computed",
                                                                            "mdd": 0.0292}}
    agg_b = {"status": "computed", "fraction_significant_p_below_0p05": 0.3,
             "mean_effect_size": 0.0146, "minimum_detectable_difference": {"status": "computed",
                                                                           "mdd": 0.0577}}
    key_a, key_b = effect_cell_verdict_key(agg_a), effect_cell_verdict_key(agg_b)
    assert key_a == key_b == "no_majority_effect"
    decision = decide_claim_standing({"estimator_a": key_a, "estimator_b": key_b})
    assert decision["branch"] == "verdict_confirmed_across_estimators"


def test_effect_that_clears_its_own_detection_floor_keeps_its_sign_without_majority_significance():
    agg = {"status": "computed", "fraction_significant_p_below_0p05": 0.11,
           "mean_effect_size": 0.1019,
           "minimum_detectable_difference": {"status": "computed", "mdd": 0.0787}}
    assert effect_cell_verdict_key(agg) == "no_majority_effect_above_detection_floor_positive"


def test_genuine_majority_significant_disagreement_still_escalates():
    """A cell that clears majority significance on one side, against a cell that genuinely fails to
    clear its own detection floor on the other side, is a real disagreement and must still fire the
    escalation branch: the repair must not suppress escalation altogether."""
    agg_pos = {"status": "computed", "fraction_significant_p_below_0p05": 0.7,
               "mean_effect_size": 0.05, "minimum_detectable_difference": {"status": "computed",
                                                                           "mdd": 0.02}}
    agg_null = {"status": "computed", "fraction_significant_p_below_0p05": 0.1,
                "mean_effect_size": -0.001, "minimum_detectable_difference": {"status": "computed",
                                                                              "mdd": 0.05}}
    key_pos, key_null = effect_cell_verdict_key(agg_pos), effect_cell_verdict_key(agg_null)
    assert key_pos == "majority_significant_positive"
    assert key_null == "no_majority_effect"
    decision = decide_claim_standing({"estimator_pos": key_pos, "estimator_null": key_null})
    assert decision["branch"] == "estimation_dependent_rung_three_escalation"


def test_control_cell_opposite_signed_excesses_below_their_own_floor_do_not_escalate():
    """Reproduces the delivered control-cell defect shape: excesses of differing sign that never
    had any resolvability test applied to them at all. Below each cell's own paired detection floor
    they must collapse to the same unresolved key."""
    agg_a = {"alignment_excess_over_random": -0.11989,
             "excess_minimum_detectable_difference": {"status": "computed", "mdd": 0.12562}}
    agg_b = {"alignment_excess_over_random": 0.09381,
             "excess_minimum_detectable_difference": {"status": "computed", "mdd": 0.22806}}
    key_a, key_b = control_cell_verdict_key(agg_a), control_cell_verdict_key(agg_b)
    assert key_a == key_b == "targeting_alignment_excess_unresolved"
    decision = decide_claim_standing({"a": key_a, "b": key_b})
    assert decision["branch"] == "verdict_confirmed_across_estimators"


def test_control_cell_excess_clearing_its_own_floor_keeps_its_sign():
    agg = {"alignment_excess_over_random": -0.06306,
           "excess_minimum_detectable_difference": {"status": "computed", "mdd": 0.04078}}
    assert control_cell_verdict_key(agg) == "targeting_alignment_excess_above_detection_floor_negative"


# ── Dynamics vote resolvability ───────────────────────────────────────────────────

def test_dynamics_vote_resolvability_flags_a_coin_flip_tie_as_not_decidable():
    out = dynamics_vote_resolvability({"stable": 4, "oscillatory": 4})
    assert out["status"] == "computed"
    assert out["winning_fraction"] == 0.5
    assert out["decidable"] is False


def test_dynamics_vote_resolvability_flags_a_clear_majority_as_decidable():
    out = dynamics_vote_resolvability({"stable": 5, "oscillatory": 2, "marginal": 1})
    assert out["decidable"] is True
    assert out["winning_fraction"] == 5 / 8


# ── Tier-three budget sizing ──────────────────────────────────────────────────────

def test_rung_three_sample_size_scales_with_sd_over_effect_and_respects_the_budget_cap():
    tight = rung_three_sample_size(sd=0.02, effect=0.30)
    loose = rung_three_sample_size(sd=0.20, effect=0.01)
    assert tight["status"] == "computed" and tight["n_required"] < loose["n_required"]
    assert loose["feasible_within_budget"] is False
    assert loose["budget_cap_sessions"] == RUNG_THREE_MAX_SESSIONS
    assert rung_three_sample_size(sd=0.0, effect=0.1)["status"] == "not_computable"
    assert rung_three_sample_size(sd=0.1, effect=float("nan"))["status"] == "not_computable"


def test_mdd_at_declared_cap_matches_the_project_convention():
    sizing = rung_three_sample_size(sd=0.1, effect=10.0)  # huge effect -> cap binds
    z = 2.8016015201700604
    expected = z * 0.1 / np.sqrt(min(sizing["n_required"], RUNG_THREE_MAX_SESSIONS))
    assert abs(sizing["mdd_at_cap"] - expected) < 1e-12
