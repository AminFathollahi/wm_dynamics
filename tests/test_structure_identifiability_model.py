"""Regression tests for run_structure_identifiability_model.py's decision logic:
the collinearity guard and the three-branch verdict rule."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_structure_identifiability_model as mod  # noqa: E402


def test_structure_present_in_only_one_dataset_is_flagged_collinear():
    frame = pd.DataFrame({
        "structure": ["hippocampus", "hippocampus", "vmpfc", "vmpfc"],
        "dataset": ["dandi_000469", "dandi_001187_000673", "dandi_000574", "dandi_000574"],
    })
    excluded = mod.structures_collinear_with_dataset(frame)
    assert "vmpfc" in excluded
    assert "hippocampus" not in excluded


def test_verdict_requires_both_model_and_matched_draw_evidence():
    model_result = {
        "status": "estimable",
        "coefficients": {"C(structure)[T.pre_sma]": {"excludes_zero": True}},
    }
    no_matched = {"status": "not_run"}
    only_model = mod.three_branch_verdict(model_result, no_matched, fingerprints_separable=True)
    assert only_model["verdict"] == "no_structure_dissociation"

    matched_with_pair = {
        "status": "estimable",
        "pairs": {"hippocampus__vs__pre_sma": {"paired_summary": {"status": "estimable", "interval_excludes_zero": True}}},
    }
    both = mod.three_branch_verdict(model_result, matched_with_pair, fingerprints_separable=True)
    assert both["verdict"] == "structure_dissociation_supported"


def test_verdict_is_estimator_non_identified_when_fingerprints_do_not_separate():
    model_result = {"status": "estimable", "coefficients": {}}
    result = mod.three_branch_verdict(model_result, {"status": "not_run"}, fingerprints_separable=False)
    assert result["verdict"] == "estimator_non_identified"
