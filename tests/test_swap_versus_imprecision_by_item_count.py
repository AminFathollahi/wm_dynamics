"""Tests for scripts/run_swap_versus_imprecision_by_item_count.py.

Two kinds of coverage: (1) the small pure-function helpers this module adds on top of the reused,
unchanged estimators (the direct paired test, its Benjamini-Hochberg family correction, the worst-
imprecision dichotomisation); (2) two end-to-end synthetic-data tests, built on hand-generated
per-trial arrays run through the real, unchanged _outcome_arm and _family estimators so a full
synthetic session row is assembled exactly the way a real one is, then fed through this module's own
per-level table, direct paired tests and named-outcome decision -- one built so a component predicts
swaps and not imprecision and the per-level ladder must recover exactly that story, and one built so a
positive low-item-count cell and a negative high-item-count cell combine to a null, where the mixing
outcome must fire rather than a dissociation being claimed from the null combination alone."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_component_and_item_binding import _family, _outcome_arm  # noqa: E402
from run_swap_versus_imprecision_by_item_count import (  # noqa: E402
    BRANCH_HOLDS_EVERY_LEVEL,
    FLAG_MIXING,
    _bh_family,
    _direct_paired_test,
    _dichotomize_worst,
    _one_level_raw,
    _pool_series,
    block_a_heterogeneity,
    block_a_level_counts,
    block_a_table,
    block_b_primary,
    decide_named_outcomes,
)


# ---------------------------------------------------------------------------------------------------
# Small pure-function helpers
# ---------------------------------------------------------------------------------------------------

def test_dichotomize_worst_flags_exactly_the_k_largest_values():
    values = np.array([0.1, 0.9, 0.3, 0.7, 0.2])
    flag = _dichotomize_worst(values, 2)
    assert list(flag) == [0.0, 1.0, 0.0, 1.0, 0.0]  # indices 1 and 3 hold the two largest values


def test_dichotomize_worst_zero_k_flags_nothing():
    values = np.array([0.1, 0.9, 0.3])
    flag = _dichotomize_worst(values, 0)
    assert list(flag) == [0.0, 0.0, 0.0]


def test_direct_paired_test_identical_arrays_gives_zero_difference_and_is_not_significant():
    a = np.array([0.1, 0.2, 0.3, -0.1, 0.05, 0.15])
    result = _direct_paired_test(a, a.copy(), "unit_test|identical")
    assert result["status"] == "tested"
    assert result["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert result["significant"] is False


def test_direct_paired_test_consistent_large_gap_is_significant():
    rng = np.random.default_rng(0)
    a = 0.5 + rng.normal(scale=0.02, size=6)
    b = 0.0 + rng.normal(scale=0.02, size=6)
    result = _direct_paired_test(a, b, "unit_test|large_gap")
    assert result["status"] == "tested"
    assert result["significant"] is True
    assert result["mean_diff"] > 0.4


def test_direct_paired_test_fewer_than_two_sessions_is_not_computable():
    result = _direct_paired_test(np.array([0.1]), np.array([0.2]), "unit_test|too_few")
    assert result["status"] == "not_computable"


def test_bh_family_corrects_across_labelled_tests():
    tests = {
        "level2": {"status": "tested", "p_value": 0.01},
        "level3": {"status": "tested", "p_value": 0.04},
    }
    result = _bh_family(tests)
    assert result["status"] == "computed"
    assert set(result["labels"]) == {"level2", "level3"}
    # BH q-values are >= the raw p-values.
    assert result["q_values"]["level2"] >= 0.01
    assert result["q_values"]["level3"] >= 0.04


def test_bh_family_skips_not_computable_entries():
    tests = {"level2": {"status": "tested", "p_value": 0.01},
             "level3": {"status": "not_computable"}}
    result = _bh_family(tests)
    assert result["labels"] == ["level2"]


def test_one_level_raw_reads_load1_for_imprecision_and_per_level_otherwise():
    row = {
        "load_1_control": {"status": "computed", "deviation": {"raw": {"status": "computed", "r": 0.3}}},
        "imprecision": {"deviation": {"per_level": {"2": {"status": "computed",
                                                            "family": {"raw": {"status": "computed", "r": 0.4}}}}}},
        "swap_primary": {"deviation": {"per_level": {"2": {"status": "computed",
                                                             "family": {"raw": {"status": "computed", "r": 0.5}}}}}},
    }
    assert _one_level_raw(row, "imprecision", "deviation", 1) == pytest.approx(0.3)
    assert _one_level_raw(row, "imprecision", "deviation", 2) == pytest.approx(0.4)
    assert _one_level_raw(row, "swap_primary", "deviation", 1) is None  # swap undefined at item count 1
    assert _one_level_raw(row, "swap_primary", "deviation", 2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------------
# Synthetic session rows, built from hand-generated per-trial arrays through the real (unchanged)
# _outcome_arm and _family estimators -- an honest session row, not a mocked one.
# ---------------------------------------------------------------------------------------------------

def _make_synthetic_row(name: str, rng: np.random.Generator, n_per_level: int, swap_coef: float,
                         imprecision_coef_by_level: dict[int, float], seed_tag: str) -> dict:
    def _logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    n1 = n_per_level
    dev1 = rng.normal(size=n1)
    amp1 = rng.normal(size=n1)
    sc1 = rng.normal(size=n1)
    ti1 = np.arange(n1, dtype=float)
    report_error1 = imprecision_coef_by_level.get(1, 0.0) * dev1 + rng.normal(scale=1.0, size=n1)

    item_count, deviation, amplitude, spike_count, trial_index = [], [], [], [], []
    swap_primary, imprecision = [], []
    n_trials_by_level = {"1": n1}
    n_swap_by_level = {}
    for level in (2, 3):
        n = n_per_level
        dev = rng.normal(size=n)
        amp = rng.normal(size=n)
        sc = rng.normal(size=n)
        ti = np.arange(n, dtype=float)
        swap_prob = _logistic(swap_coef * dev)
        swap = (rng.random(n) < swap_prob).astype(float)
        imp = imprecision_coef_by_level.get(level, 0.0) * dev + rng.normal(scale=1.0, size=n)
        item_count.append(np.full(n, float(level)))
        deviation.append(dev)
        amplitude.append(amp)
        spike_count.append(sc)
        trial_index.append(ti)
        swap_primary.append(swap)
        imprecision.append(imp)
        n_trials_by_level[str(level)] = n
        n_swap_by_level[str(level)] = int(swap.sum())

    item_count = np.concatenate(item_count)
    deviation = np.concatenate(deviation)
    amplitude = np.concatenate(amplitude)
    spike_count = np.concatenate(spike_count)
    trial_index = np.concatenate(trial_index)
    swap_primary = np.concatenate(swap_primary)
    imprecision = np.concatenate(imprecision)

    tag = f"{seed_tag}|{name}"
    swap_arm = _outcome_arm(swap_primary, item_count, deviation, amplitude, spike_count, trial_index,
                             f"{tag}|swap_primary")
    imprecision_arm = _outcome_arm(imprecision, item_count, deviation, amplitude, spike_count, trial_index,
                                    f"{tag}|imprecision")
    load1 = {
        "status": "computed", "n_trials": n1,
        "deviation": _family(report_error1, dev1, sc1, ti1, f"{tag}|load1|deviation"),
        "amplitude": _family(report_error1, amp1, sc1, ti1, f"{tag}|load1|amplitude"),
    }
    row = {
        "status": "computed", "session": name,
        "n_trials_by_item_count": n_trials_by_level,
        "n_swap_primary_by_item_count": n_swap_by_level,
        "n_swap_strict_by_item_count": n_swap_by_level,
        "swap_primary": swap_arm, "swap_strict": swap_arm, "imprecision": imprecision_arm,
        "load_1_control": load1,
    }
    return row


N_SESSIONS = 6  # the paired sign-flip test's own minimum-attainable-p floor needs n >= 5 to ever reach p<=0.05


def test_per_level_ladder_recovers_a_swap_only_component_synthetic():
    """Deviation drives the swap indicator strongly and positively at every level >= 2, and is pure
    noise against imprecision at every level including item count 1. The per-level Block A table must
    show the swap association significant at every level and the imprecision association null at every
    level, and the direct paired test between them must be significant after BH correction."""
    rows = [
        _make_synthetic_row(f"synthetic_swap_only_{i}", np.random.default_rng(1000 + i), n_per_level=250,
                             swap_coef=3.0, imprecision_coef_by_level={}, seed_tag="swap_only_test")
        for i in range(N_SESSIONS)
    ]

    block_a = block_a_table(rows)
    block_a_counts = block_a_level_counts(rows)
    heterogeneity = block_a_heterogeneity(rows)
    levels_with_swap = (2, 3)

    for level in levels_with_swap:
        swap_cell = block_a["swap_primary"]["deviation"][str(level)]["raw"]
        imprecision_cell = block_a["imprecision"]["deviation"][str(level)]["raw"]
        assert swap_cell["status"] == "tested", f"level {level} swap cell not tested"
        assert swap_cell["significant"] is True, f"level {level} swap association should be significant"
        assert imprecision_cell["status"] == "tested", f"level {level} imprecision cell not tested"
        assert imprecision_cell["significant"] is False, f"level {level} imprecision should be null"

    imprecision_level1 = block_a["imprecision"]["deviation"]["1"]["raw"]
    assert imprecision_level1["status"] == "tested"
    assert imprecision_level1["significant"] is False

    block_b = block_b_primary(rows, levels_with_swap, outcome="swap_primary")
    block_b_bh = _bh_family(block_b)
    for level in levels_with_swap:
        assert block_b[str(level)]["status"] == "tested"
        assert block_b_bh["bh_significant"][str(level)] is True, \
            f"the direct paired test at level {level} should be significant after BH correction"

    load1_deviation_raw = _pool_series([
        r["load_1_control"]["deviation"]["raw"]["r"] for r in rows
        if r["load_1_control"]["deviation"]["raw"]["status"] == "computed"
    ])
    named = decide_named_outcomes(
        block_a, block_a_counts, block_b, block_b_bh, block_b, block_b_bh, heterogeneity,
        levels_with_swap, load1_deviation_raw,
        reproduced_imprecision_combined={"status": "tested", "significant": False}, bias_only={})
    assert named["primary_branch"] == BRANCH_HOLDS_EVERY_LEVEL


def test_mixing_branch_fires_when_opposite_sign_levels_combine_to_a_null_synthetic():
    """Item count 2's imprecision association is strongly positive and item count 3's is strongly
    negative, at matched trial counts, so their trial-count-weighted combination (the only quantity the
    delivered artifact ever reported) sits near zero and is not significant -- exactly the mixing this
    leg exists to catch. The mixing outcome must fire, and the full per-level dissociation outcome must
    not (imprecision is significant, with opposite signs, at levels 2 and 3)."""
    rows = [
        _make_synthetic_row(f"synthetic_mixing_{i}", np.random.default_rng(2000 + i), n_per_level=250,
                             swap_coef=0.0, imprecision_coef_by_level={2: 2.5, 3: -2.5},
                             seed_tag="mixing_test")
        for i in range(N_SESSIONS)
    ]

    block_a = block_a_table(rows)
    block_a_counts = block_a_level_counts(rows)
    heterogeneity = block_a_heterogeneity(rows)
    levels_with_swap = (2, 3)

    level2 = block_a["imprecision"]["deviation"]["2"]["raw"]
    level3 = block_a["imprecision"]["deviation"]["3"]["raw"]
    assert level2["status"] == "tested" and level3["status"] == "tested"
    assert level2["significant"] is True and level3["significant"] is True
    assert (level2["mean_value"] > 0.0) != (level3["mean_value"] > 0.0), \
        "levels 2 and 3 must disagree in sign for this to be a mixing scenario"

    het_test = heterogeneity["imprecision"]["deviation"]["pairwise_tests"]["level2_vs_level3"]
    assert het_test["status"] == "tested"
    assert het_test["significant"] is True

    from run_component_and_item_binding import build_pooled_table
    pooled = build_pooled_table(rows)
    combined = pooled["imprecision"]["deviation"]["within_item_count_level"]["raw"]
    assert combined["status"] == "tested"
    assert combined["significant"] is False, \
        "the matched, opposite-signed trial counts must combine to a null, or this is not a mixing test"

    block_b = block_b_primary(rows, levels_with_swap, outcome="swap_primary")
    block_b_bh = _bh_family(block_b)
    load1_deviation_raw = _pool_series([
        r["load_1_control"]["deviation"]["raw"]["r"] for r in rows
        if r["load_1_control"]["deviation"]["raw"]["status"] == "computed"
    ])
    named = decide_named_outcomes(
        block_a, block_a_counts, block_b, block_b_bh, block_b, block_b_bh, heterogeneity,
        levels_with_swap, load1_deviation_raw, reproduced_imprecision_combined=combined, bias_only={})

    assert named[FLAG_MIXING]["fires"] is True
    assert named["primary_branch"] != BRANCH_HOLDS_EVERY_LEVEL


def test_a_bias_only_reproduced_swap_cell_cannot_carry_a_branch():
    """A regression test for a real defect: a level whose swap association is both real-significant and
    reproduced by its own bias-only control must not be allowed to fire a positive branch from that
    level, even though the raw significance numbers alone would read as a clean dissociation. Reuses
    the swap-only synthetic rows from the per-level ladder test (which, un-voided, fires
    the_dissociation_holds_at_every_item_count_level) and marks level 2's swap cell as reproduced."""
    rows = [
        _make_synthetic_row(f"synthetic_void_{i}", np.random.default_rng(3000 + i), n_per_level=250,
                             swap_coef=3.0, imprecision_coef_by_level={}, seed_tag="void_test")
        for i in range(N_SESSIONS)
    ]
    block_a = block_a_table(rows)
    block_a_counts = block_a_level_counts(rows)
    heterogeneity = block_a_heterogeneity(rows)
    levels_with_swap = (2, 3)
    block_b = block_b_primary(rows, levels_with_swap, outcome="swap_primary")
    block_b_bh = _bh_family(block_b)
    load1_deviation_raw = _pool_series([
        r["load_1_control"]["deviation"]["raw"]["r"] for r in rows
        if r["load_1_control"]["deviation"]["raw"]["status"] == "computed"
    ])

    # Un-voided: this reproduces the per-level ladder test's own result -- a sanity check that the two
    # tests build the same scenario before the voided version is checked against it.
    named_unvoided = decide_named_outcomes(
        block_a, block_a_counts, block_b, block_b_bh, block_b, block_b_bh, heterogeneity,
        levels_with_swap, load1_deviation_raw,
        reproduced_imprecision_combined={"status": "tested", "significant": False}, bias_only={})
    assert named_unvoided["primary_branch"] == BRANCH_HOLDS_EVERY_LEVEL

    bias_only = {"swap_primary|deviation|level2": {"reproduces_the_real_result": True}}
    named_voided = decide_named_outcomes(
        block_a, block_a_counts, block_b, block_b_bh, block_b, block_b_bh, heterogeneity,
        levels_with_swap, load1_deviation_raw,
        reproduced_imprecision_combined={"status": "tested", "significant": False}, bias_only=bias_only)

    assert named_voided["primary_branch"] != BRANCH_HOLDS_EVERY_LEVEL
    assert named_voided["per_level_flags"]["2"]["swap_association_bias_only_void"] is True
    assert named_voided["per_level_flags"]["2"]["swap_predicts"] is False
    assert named_voided["per_level_flags"]["2"]["block_b_primary_bh_significant"] is False
    assert named_voided["bias_only_voiding"]["superseded"] is True
    assert named_voided["bias_only_voiding"]["branch_before_bias_only_voiding"] == BRANCH_HOLDS_EVERY_LEVEL
    assert named_voided["bias_only_voiding"]["branch_after_bias_only_voiding"] == named_voided["primary_branch"]
