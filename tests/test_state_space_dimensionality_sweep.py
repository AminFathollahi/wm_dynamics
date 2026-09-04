"""Tests for scripts/run_state_space_dimensionality_sweep.py.

Covers the three properties the estimator-invariant restatement and the rank-selection/withdrawal
machinery depend on: (1) the cross-validated predictable-fraction restatement numerically reduces to
the delivered linear subspace-projection quantity in a noiseless synthetic scenario, and separately
agrees with its own in-sample counterpart there; (2) the four rank-selection criteria recover the
correct rank on a planted low-rank synthetic structure; (3) the withdrawal criterion fires when a
synthetic claim's effect-size sign flips across the swept range, and does not fire when it is stable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_state_space_dimensionality_sweep as m  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# cross_validated_predictable_fraction reduces to the delivered linear subspace quantity
# ---------------------------------------------------------------------------------------------------

def test_predictable_fraction_reduces_to_subspace_projection():
    """Noiseless linear scenario: y = w^T x with X isotropic. The delivered subspace-projection
    quantity for a direction w against a basis S is (||P_S w|| / ||w||)**2 (this is exactly what
    claim_memorandum_subspace / claim_occupied_manifold compute as within_frac, squared). Population
    OLS/ridge-with-vanishing-regularisation of y on Z = X @ S converges to that same number, which is
    the numeric sense in which the cross-validated restatement reduces to the delivered quantity."""
    rng = np.random.default_rng(0)
    n, d, k = 4000, 10, 3
    X = rng.standard_normal((n, d))
    S, _ = np.linalg.qr(rng.standard_normal((d, k)))  # orthonormal basis, ambient dim d, rank k
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    y = X @ w  # noiseless

    Z = X @ S
    restated = m.cross_validated_predictable_fraction(y, Z, alpha=1e-6, rng=np.random.default_rng(1))
    assert restated["status"] == "computed"

    P = S @ S.T
    delivered = (np.linalg.norm(P @ w) / np.linalg.norm(w)) ** 2

    assert abs(restated["predictable_fraction"] - delivered) < 0.02


def test_cross_validated_and_in_sample_coincide_noiseless():
    """The cross-validated and in-sample (naive) fractions coincide in the same noiseless linear
    scenario -- the reduction the module's own docstring claims."""
    rng = np.random.default_rng(2)
    n, d, k = 3000, 8, 4
    X = rng.standard_normal((n, d))
    S, _ = np.linalg.qr(rng.standard_normal((d, k)))
    w = rng.standard_normal(d)
    y = X @ w
    Z = X @ S

    cv = m.cross_validated_predictable_fraction(y, Z, alpha=1e-6, rng=np.random.default_rng(3))
    naive = m.in_sample_linear_fraction(y, Z)
    assert cv["status"] == "computed" and naive["status"] == "computed"
    assert abs(cv["predictable_fraction"] - naive["linear_fraction"]) < 0.01


def test_predictable_fraction_restatement_is_one_function_over_named_representations():
    """predictable_fraction_restatement takes any set of named coordinate bases (the 'fitted
    representation' parameter) and returns a verdict dict per name, including a not_computable entry
    when a representation is unavailable -- this is the single call site every claim in this module
    (and any future differently-fitted representation) routes through."""
    rng = np.random.default_rng(4)
    n, d = 200, 6
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = X @ w
    S, _ = np.linalg.qr(rng.standard_normal((d, 2)))

    out = m.predictable_fraction_restatement(y, X, {"memorandum_representation": S, "state_representation": None},
                                              rng=np.random.default_rng(5))
    assert out["memorandum_representation"]["cross_validated"]["status"] == "computed"
    assert out["state_representation"]["status"] == "not_computable"


def test_fit_linear_representation_refuses_non_null_labels():
    """fit_linear_representation is the single fitting entry point a later nonlinear embedding is
    reached through; passing a label argument must raise rather than silently fit on it, since a
    label- or behaviour-conditioned representation is circular for every behaviour claim that
    consumes it project-wide."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((50, 6))
    fit = m.fit_linear_representation(X, rank=3)  # labels omitted: does not raise
    assert fit["kind"] == "pca"
    assert fit["fitting_objective"] == "unsupervised_pca_reconstruction_no_labels"
    try:
        m.fit_linear_representation(X, rank=3, labels=rng.integers(0, 2, size=50))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_require_linear_representation_guards_nonlinear_kind():
    """A subspace-angle projector has no canonical basis in a nonlinear embedding; the guard must
    refuse anything not explicitly flagged as a linear (PCA) representation."""
    m._require_linear_representation({"kind": "pca"})  # does not raise
    try:
        m._require_linear_representation({"kind": "nonlinear_embedding"})
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------------------------------
# rank selectors recover a planted low-rank structure
# ---------------------------------------------------------------------------------------------------

def test_rank_selectors_recover_planted_low_rank_structure():
    rng = np.random.default_rng(6)
    n, d, true_rank = 200, 30, 3
    loadings, _ = np.linalg.qr(rng.standard_normal((d, true_rank)))
    latent = rng.standard_normal((n, true_rank))
    signal = latent @ loadings.T
    signal *= 5.0 / np.std(signal)  # strong signal-to-noise so the rank is unambiguous
    X = signal + rng.standard_normal((n, d))

    report = m.rank_selection_report(X, m.RANK_GRID, rng=np.random.default_rng(7))

    assert report["cv_reconstruction"]["status"] == "computed"
    assert abs(report["cv_reconstruction"]["selected_rank"] - true_rank) <= 1
    assert report["cv_factor_analysis_likelihood"]["status"] == "computed"
    assert abs(report["cv_factor_analysis_likelihood"]["selected_rank"] - true_rank) <= 3
    assert 1 <= report["participation_ratio_rounded"] <= true_rank + 3
    assert 1 <= report["parallel_analysis_selected_rank"] <= true_rank + 3


# ---------------------------------------------------------------------------------------------------
# withdrawal criterion
# ---------------------------------------------------------------------------------------------------

def _fake_session(claim_key: str, rank_to_effect_and_p: dict[int, tuple[float, float]], full_rank: int) -> dict:
    per_rank = {}
    for rank, (effect, p) in rank_to_effect_and_p.items():
        per_rank[str(rank)] = {
            "is_full_rank": rank == full_rank,
            claim_key: {"status": "computed", "effect_size": effect, "p_value": p, "n_trials": 100},
        }
    return {"status": "tested", "per_rank": per_rank}


def test_withdrawal_criterion_fires_on_sign_flip():
    session = _fake_session("some_claim", {2: (0.10, 0.01), 8: (-0.05, 0.30), 16: (0.08, 0.02)}, full_rank=16)
    summary = m._sign_and_significance_summary([session], ("some_claim",))
    assert summary["sign_changes_anywhere_in_range"] is True
    assert summary["withdrawn"] is True


def test_withdrawal_criterion_stable_claim_not_withdrawn():
    session = _fake_session("some_claim", {2: (0.10, 0.01), 8: (0.09, 0.02), 16: (0.11, 0.01)}, full_rank=16)
    summary = m._sign_and_significance_summary([session], ("some_claim",))
    assert summary["sign_changes_anywhere_in_range"] is False
    assert summary["significance_changes_anywhere_in_range"] is False
    assert summary["withdrawn"] is False


def test_full_rank_anchor_pools_across_sessions_regardless_of_numeric_rank():
    """Each session's own full rank is a different number (native, no reduction); the anchor must
    pool them under one label rather than scattering them into per-session singleton grid buckets."""
    session_a = _fake_session("some_claim", {2: (0.10, 0.01), 47: (0.12, 0.01)}, full_rank=47)
    session_b = _fake_session("some_claim", {2: (0.09, 0.02), 63: (0.11, 0.01)}, full_rank=63)
    summary = m._sign_and_significance_summary([session_a, session_b], ("some_claim",))
    assert summary["full_rank_anchor"]["n_sessions"] == 2
    # the grid rank "2" is shared and pooled together; the two distinct full-rank values (47, 63)
    # would otherwise appear as two separate n_sessions=1 rows rather than one n_sessions=2 anchor.
    assert summary["per_rank"]["2"]["n_sessions"] == 2
    assert "47" not in summary["per_rank"] or summary["per_rank"].get("47", {}).get("n_sessions") == 1
    assert "63" not in summary["per_rank"] or summary["per_rank"].get("63", {}).get("n_sessions") == 1


def test_withdrawal_criterion_fires_when_only_full_rank_disagrees():
    """A conclusion holding at every grid rank but flipping sign at the full-rank anchor is exactly
    a case of 'holds at only one rank' -- it must withdraw too, not just a flip among grid ranks."""
    session = _fake_session("some_claim", {2: (0.10, 0.01), 8: (0.09, 0.02), 40: (-0.07, 0.20)}, full_rank=40)
    summary = m._sign_and_significance_summary([session], ("some_claim",))
    assert summary["withdrawn"] is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
