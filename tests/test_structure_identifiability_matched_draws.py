"""Regression tests for run_structure_identifiability_matched_draws.py's
rate-matching and seeding helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_structure_identifiability_matched_draws as mod  # noqa: E402


def test_closest_rate_subset_beats_a_single_uniform_draw_on_average():
    rng = np.random.default_rng(0)
    unit_rates = np.concatenate([np.full(20, 1.0), np.full(20, 10.0)])
    target_rate = 1.0
    target_count = 5

    matched_gaps, uniform_gaps = [], []
    for trial in range(30):
        matched = mod._closest_rate_subset(unit_rates, target_count, target_rate, rng, n_candidates=40)
        matched_gaps.append(abs(np.mean(unit_rates[matched]) - target_rate))
        uniform = rng.choice(len(unit_rates), size=target_count, replace=False)
        uniform_gaps.append(abs(np.mean(unit_rates[uniform]) - target_rate))

    assert np.mean(matched_gaps) < np.mean(uniform_gaps)


def test_stable_seed_is_deterministic_and_distinguishes_arms():
    a1 = mod._stable_seed("hippocampus", "pre_sma", "sub-1", "a")
    a2 = mod._stable_seed("hippocampus", "pre_sma", "sub-1", "a")
    b = mod._stable_seed("hippocampus", "pre_sma", "sub-1", "b")
    assert a1 == a2
    assert a1 != b
