"""Regression tests for run_fidelity_controllability_map.py's three-branch
verdict logic (tradeoff / dissociation / estimator_non_identified)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_fidelity_controllability_map as mod  # noqa: E402


def test_positive_excluding_zero_is_tradeoff():
    correlation = {
        "status": "estimable", "excludes_zero_positive": True, "excludes_zero_negative": False,
        "spans_near_full_range": False,
    }
    assert mod.three_branch_verdict(correlation)["verdict"] == "tradeoff"


def test_negative_excluding_zero_is_dissociation():
    correlation = {
        "status": "estimable", "excludes_zero_positive": False, "excludes_zero_negative": True,
        "spans_near_full_range": False,
    }
    assert mod.three_branch_verdict(correlation)["verdict"] == "dissociation"


def test_interval_including_zero_is_non_identified_not_forced_into_a_branch():
    correlation = {
        "status": "estimable", "excludes_zero_positive": False, "excludes_zero_negative": False,
        "spans_near_full_range": False,
    }
    assert mod.three_branch_verdict(correlation)["verdict"] == "estimator_non_identified"


def test_too_few_structures_is_non_identified():
    correlation = {"status": "non_identified", "reason": "fewer than 3 structures"}
    assert mod.three_branch_verdict(correlation)["verdict"] == "estimator_non_identified"
