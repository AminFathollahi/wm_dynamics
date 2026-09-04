"""Tests for scripts/run_reduction_sensitive_claim_robustness.py.

Covers the four behaviours the leg stands on: the pre-declared reduction-sensitivity decision fires
the correct branch on planted agreement, planted one-sided disagreement and planted mixed-sign
disagreement; the cross-temporal generalisation restatement separates true class structure that
generalises across time bins from its own label-permutation null and returns retained null draws
usable for cross-level pooling; the checkpoint scheme is schema-tagged so a stale or foreign
checkpoint reads as a miss rather than a silent hit; and the multi-object level-combination path
this module reuses unchanged from the sibling geometry module still combines a genuine two-level
case correctly when fed this module's own cell schema.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_reduction_sensitive_claim_robustness as mod  # noqa: E402
from run_deviation_geometry_estimation_robustness import (  # noqa: E402
    _combine_restated_cells_across_levels,
)

RNG = np.random.default_rng(20260827)


# ── Reduction-sensitivity decision ──────────────────────────────────────────────────

def test_no_estimator_improving_is_a_property_of_the_data():
    keys = {"native_full_rank": "no_majority_effect",
            "principal_components": "no_majority_effect_above_detection_floor_negative",
            "temporal_diffusion_embedding": "no_majority_effect"}
    decision = mod.decide_reduction_sensitivity(keys)
    assert decision["branch"] == "null_is_a_property_of_the_data"
    assert decision["materially_improved_estimators"] == {}


def test_native_full_rank_alone_improving_is_an_artefact_of_the_projection():
    """The signature case this branch exists for: a signal absent under every reduction but present
    at native full rank means the reduction, not the brain, produced the delivered null."""
    keys = {"native_full_rank": "majority_significant_positive",
            "principal_components": "no_majority_effect",
            "factor_analysis": "no_majority_effect",
            "temporal_diffusion_embedding": "no_majority_effect"}
    decision = mod.decide_reduction_sensitivity(keys)
    assert decision["branch"] == "null_is_an_artefact_of_the_projection"
    assert decision["materially_improved_estimators"] == {"native_full_rank": "majority_significant_positive"}


def test_agreement_across_every_improving_estimator_is_still_an_artefact_of_the_projection():
    keys = {"native_full_rank": "majority_significant_positive",
            "principal_components": "majority_significant_positive",
            "temporal_diffusion_embedding": "majority_significant_positive"}
    decision = mod.decide_reduction_sensitivity(keys)
    assert decision["branch"] == "null_is_an_artefact_of_the_projection"
    assert set(decision["materially_improved_estimators"]) == {
        "native_full_rank", "principal_components", "temporal_diffusion_embedding"}


def test_mixed_sign_material_improvement_escalates_rather_than_resolving():
    keys = {"native_full_rank": "majority_significant_positive",
            "principal_components": "majority_significant_negative",
            "temporal_diffusion_embedding": "no_majority_effect"}
    decision = mod.decide_reduction_sensitivity(keys)
    assert decision["branch"] == "verdicts_disagree_escalation_sized_not_spent"
    assert set(decision["materially_improved_estimators"]) == {
        "native_full_rank", "principal_components"}


def test_all_not_computable_falls_back_without_crashing():
    decision = mod.decide_reduction_sensitivity(
        {"native_full_rank": "not_computable", "principal_components": "not_applicable"})
    assert decision["branch"] == "no_computable_estimator_cells"


# ── Cross-temporal generalisation restatement ───────────────────────────────────────

def test_ctg_cell_separates_true_time_stable_class_structure_from_its_null():
    """A latent whose class means are stable across time bins must decode from a held-out time bin
    trained on another, well above chance, and clear its own label-permutation null."""
    n_per_class, n_bins, k = 24, 5, 4
    labels = np.array([0, 1, 2] * n_per_class)
    centers = RNG.standard_normal((3, k)) * 5.0
    n_trials = len(labels)
    latent = np.stack([
        np.stack([centers[c] + 0.4 * RNG.standard_normal(k) for _ in range(n_bins)])
        for c in labels
    ])  # (trials, bins, k), same class centre at every bin
    assert latent.shape == (n_trials, n_bins, k)
    t_idx = np.arange(n_bins)
    rng = np.random.default_rng(3)
    cell = mod.ctg_cell_with_null_draws(latent, labels, t_idx, n_splits=3, n_perm=30, rng=rng)
    assert cell["status"] == "computed"
    assert cell["effect_size"] > 0
    assert cell["p_value"] < 0.10
    assert len(cell["null_values"]) == 30
    assert cell["tau_interpretable"] in (True, False)  # field present and well-formed


def test_ctg_cell_reports_not_computable_on_a_single_class():
    latent = RNG.standard_normal((10, 3, 2))
    labels = np.zeros(10)
    cell = mod.ctg_cell_with_null_draws(latent, labels, np.arange(3), n_splits=3, n_perm=10,
                                        rng=np.random.default_rng(1))
    assert cell["status"] == "not_computable"


def test_ctg_cell_schema_feeds_the_shared_level_combiner():
    """The sibling geometry module's cross-level combiner is reused unchanged for this claim: it
    only needs predictable_fraction/null_mean/null_values, which the CTG cell now carries."""
    n_per_class, n_bins, k = 15, 4, 3
    labels = np.array([0, 1, 2] * n_per_class)
    centers = RNG.standard_normal((3, k)) * 5.0
    latent = np.stack([
        np.stack([centers[c] + 0.4 * RNG.standard_normal(k) for _ in range(n_bins)])
        for c in labels
    ])
    t_idx = np.arange(n_bins)
    cell_a = mod.ctg_cell_with_null_draws(latent, labels, t_idx, n_splits=3, n_perm=20,
                                          rng=np.random.default_rng(5))
    cell_b = mod.ctg_cell_with_null_draws(latent, labels, t_idx, n_splits=3, n_perm=20,
                                          rng=np.random.default_rng(6))
    assert cell_a["status"] == cell_b["status"] == "computed"
    combined = _combine_restated_cells_across_levels(
        [(len(labels), cell_a), (len(labels), cell_b)], n_perm=20)
    assert combined["status"] == "computed"
    assert combined["n_levels_combined"] == 2
    assert combined["n_trials"] == 2 * len(labels)


# ── Checkpoint schema tagging ────────────────────────────────────────────────────────

def test_stale_schema_checkpoint_reads_as_a_miss_not_a_hit():
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = mod.CHECKPOINT_DIR
        mod.CHECKPOINT_DIR = Path(tmp)
        try:
            path = mod._checkpoint_path("some_key")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"_schema": "some_other_schema_v0", "_complete": True,
                                        "record": {"status": "computed"}}))
            assert mod._load_checkpoint("some_key") is None
        finally:
            mod.CHECKPOINT_DIR = original_dir


def test_incomplete_checkpoint_reads_as_a_miss():
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = mod.CHECKPOINT_DIR
        mod.CHECKPOINT_DIR = Path(tmp)
        try:
            path = mod._checkpoint_path("some_key")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"_schema": mod.CHECKPOINT_SCHEMA, "_complete": False,
                                        "record": {"status": "computed"}}))
            assert mod._load_checkpoint("some_key") is None
        finally:
            mod.CHECKPOINT_DIR = original_dir


def test_checkpoint_round_trips_through_restore_checkpoint_with_array_typed_fields():
    """A record containing a numpy array (as the deviation-cell-style records occasionally may) must
    come back as an array after a save/load cycle, not as a plain list -- the exact defect class the
    engineering requirement warns has silently corrupted resumed artifacts on this project before."""
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = mod.CHECKPOINT_DIR
        mod.CHECKPOINT_DIR = Path(tmp)
        try:
            record = {"status": "computed", "null_values": np.array([0.1, 0.2, 0.3])}
            mod._save_checkpoint("round_trip_key", record)
            restored = mod._load_checkpoint("round_trip_key")
            assert restored is not None
            assert isinstance(restored["null_values"], np.ndarray)
            assert np.allclose(restored["null_values"], [0.1, 0.2, 0.3])
        finally:
            mod.CHECKPOINT_DIR = original_dir
