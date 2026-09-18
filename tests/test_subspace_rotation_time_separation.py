"""Tests for scripts/run_subspace_rotation_time_separation.py.

Things that could silently break this module without erroring: (1) chunked_interleaved_folds could
fail to actually match the early/late split on sample size while sharing no trials with it -- the
whole point of this design; (2) the interleaved-chunk and quarter-split references could fail to
discriminate a genuinely rotating relation from a stable one; (3) the branch classifier could
conflate the two references or mis-handle the "between" case; (4) pooling could mis-weight sessions
or independent units. A final group reproduces the delivered diagnostic's early-to-late overlap
against real data when the external data drive is mounted, and is skipped otherwise.

WM_DYNAMICS_DATA_ROOT must point at the currently-mounted external data drive for the data-dependent
tests in the last group; every other test here is synthetic and needs no data root.
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

import run_subspace_rotation_time_separation as sep  # noqa: E402
from subspace_identity import block_folds, regression_basis  # noqa: E402

DATA_ROOT_AVAILABLE = bool(os.environ.get("WM_DYNAMICS_DATA_ROOT")) and Path(
    os.environ.get("WM_DYNAMICS_DATA_ROOT", "")
).is_dir()
requires_data_root = pytest.mark.skipif(
    not DATA_ROOT_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set to a currently-mounted directory",
)


def _rotating_relation(rng, n=200, n_units=8):
    """A label relation that lives along unit 0 in the session's first half and unit 1 in the
    second half -- the across-half fits point in genuinely different directions, the positive
    control for rotation."""
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    first_half = np.arange(n) < n // 2
    activity[first_half, 0] += 3.0 * target[first_half]
    activity[~first_half, 1] += 3.0 * target[~first_half]
    return activity, target


def _stable_relation(rng, n=200, n_units=8):
    """The same label relation holds for every trial -- the negative control: any disagreement
    between disjoint splits here is estimation noise, not rotation."""
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    activity[:, 0] += 5.0 * target
    return activity, target


# ---------------------------------------------------------------------------------------------------
# chunked_interleaved_folds: the design property -- matches early/late on size, shares no trials
# ---------------------------------------------------------------------------------------------------

def test_chunked_interleaved_folds_alternates_and_covers_every_trial():
    n, chunk_size = 40, 4
    folds = sep.chunked_interleaved_folds(n, chunk_size)
    assert set(np.unique(folds)) == {0, 1}
    assert len(folds) == n
    # first chunk is set 0, second is set 1, alternating
    assert np.array_equal(folds[0:4], np.zeros(4))
    assert np.array_equal(folds[4:8], np.ones(4))
    assert np.array_equal(folds[8:12], np.zeros(4))


def test_chunked_interleaved_split_spans_the_full_range():
    folds = sep.chunked_interleaved_folds(100, 4)
    set_a = np.flatnonzero(folds == 0)
    assert set_a.min() < 10 and set_a.max() > 90  # set A appears near both the start and the end


def test_chunked_interleaved_and_early_late_splits_have_equal_trial_counts_and_share_no_trials():
    n = 97  # odd, to exercise uneven halves/chunks
    for chunk_size in sep.CHUNK_SIZES_TRIALS:
        if n < sep.MIN_CHUNKS_PER_SESSION * chunk_size:
            continue
        interleaved = sep.chunked_interleaved_folds(n, chunk_size)
        early_late = block_folds(n, 2)

        # disjointness within each split
        assert np.sum(interleaved == 0) + np.sum(interleaved == 1) == n
        assert np.sum(early_late == 0) + np.sum(early_late == 1) == n

        # matched sample size: interleaved set sizes and early/late half sizes agree within one chunk
        n_early, n_late = int(np.sum(early_late == 0)), int(np.sum(early_late == 1))
        n_a, n_b = int(np.sum(interleaved == 0)), int(np.sum(interleaved == 1))
        assert abs(n_a - n_early) <= chunk_size
        assert abs(n_b - n_late) <= chunk_size

        # the two splits are different partitions -- interleaved set 0 is not simply the early half
        assert not np.array_equal(interleaved == 0, early_late == 0)


def test_chunked_interleaved_folds_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        sep.chunked_interleaved_folds(3, 4)
    with pytest.raises(ValueError):
        sep.chunked_interleaved_folds(10, 0)


# ---------------------------------------------------------------------------------------------------
# _chunk_reference / _quarter_split_reference: matched size, disjoint, discriminate rotation
# ---------------------------------------------------------------------------------------------------

def test_chunk_reference_not_computable_below_minimum_chunks():
    directions = np.random.default_rng(1).normal(size=(10, 4))
    target = np.random.default_rng(1).normal(size=10)
    result = sep._chunk_reference(directions, target, regression_basis, 10, chunk_size=16)
    assert result["status"] == "not_computable"


def test_chunk_reference_high_for_a_strong_stable_relation():
    activity, target = _stable_relation(np.random.default_rng(2), n=200)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    result = sep._chunk_reference(directions, target, regression_basis, len(target), chunk_size=4)
    assert result["status"] == "computed"
    assert result["overlap"] > 0.8
    assert result["n_trials_a"] + result["n_trials_b"] == 200


def test_chunk_reference_low_for_a_rotating_relation():
    activity, target = _rotating_relation(np.random.default_rng(3), n=200)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    result = sep._chunk_reference(directions, target, regression_basis, len(target), chunk_size=4)
    assert result["status"] == "computed"
    # chunks alternate rapidly through both halves, so both sets see both regimes -- overlap stays high
    assert result["overlap"] > 0.5


def test_quarter_split_reference_near_across_half_overlap_for_a_stable_relation():
    activity, target = _stable_relation(np.random.default_rng(4), n=200)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    result = sep._quarter_split_reference(directions, target, regression_basis)
    assert result["status"] == "computed"
    assert result["mean"] > 0.5  # noisier (quarter-sized) fits still agree for a truly stable relation


def test_quarter_split_reference_low_for_a_rotating_relation():
    activity, target = _rotating_relation(np.random.default_rng(5), n=200)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    result = sep._quarter_split_reference(directions, target, regression_basis)
    assert result["status"] == "computed"
    # both quarters of a half share that half's single true direction, so overlap stays high
    # WITHIN a half even though the relation differs ACROSS halves
    assert result["mean"] > 0.5


# ---------------------------------------------------------------------------------------------------
# _cell_session: the contrast discriminates rotation, references sit below across-half for rotation
# ---------------------------------------------------------------------------------------------------

def test_cell_session_rotating_relation_across_half_overlap_well_below_both_references():
    activity, target = _rotating_relation(np.random.default_rng(6))
    cell = sep._cell_session(activity, "continuous", target, "test|rotating")
    assert cell["status"] == "computed"
    chunk = cell["chunk_reference_by_size"][sep.PRIMARY_CHUNK_SIZE_TRIALS]
    quarter = cell["quarter_split_reference"]
    assert chunk["status"] == "computed" and quarter["status"] == "computed"
    assert cell["across_half_overlap"] < chunk["overlap"]
    assert cell["across_half_overlap"] < quarter["mean"]


def test_cell_session_stable_relation_across_half_overlap_near_both_references():
    activity, target = _stable_relation(np.random.default_rng(7))
    cell = sep._cell_session(activity, "continuous", target, "test|stable")
    assert cell["status"] == "computed"
    chunk = cell["chunk_reference_by_size"][sep.PRIMARY_CHUNK_SIZE_TRIALS]
    quarter = cell["quarter_split_reference"]
    assert abs(cell["across_half_overlap"] - chunk["overlap"]) < 0.2
    assert abs(cell["across_half_overlap"] - quarter["mean"]) < 0.2


def test_cell_session_too_few_trials_is_not_computable():
    cell = sep._cell_session(np.ones((3, 4)), "continuous", np.arange(3, dtype=float), "test|short")
    assert cell["status"] == "too_few_trials"


# ---------------------------------------------------------------------------------------------------
# pooling: equal-session-then-equal-unit weighting
# ---------------------------------------------------------------------------------------------------

def _record(unit, across, chunk_overlap, quarter_mean, chunk_status="computed", quarter_status="computed"):
    return {
        "independent_unit": unit, "across_half_overlap": across,
        "chunk_reference_by_size": {
            sep.PRIMARY_CHUNK_SIZE_TRIALS: {"status": chunk_status, "overlap": chunk_overlap},
        },
        "quarter_split_reference": {"status": quarter_status, "mean": quarter_mean},
    }


def test_pool_across_half_weights_independent_units_not_sessions():
    records = [
        _record("u1", 0.2, 0.5, 0.5), _record("u1", 0.4, 0.5, 0.5),  # two sessions, unit u1 mean 0.3
        _record("u2", 0.6, 0.5, 0.5), _record("u3", 0.6, 0.5, 0.5), _record("u4", 0.6, 0.5, 0.5),
    ]
    pooled = sep._pool_across_half(records, "seed-a")
    assert pooled["status"] == "computed"
    assert pooled["mean"] == pytest.approx((0.3 + 0.6 + 0.6 + 0.6) / 4)


def test_pool_contrast_chunk_drops_sessions_without_a_computed_chunk_reference():
    records = [
        _record("u1", 0.2, 0.5, 0.5), _record("u2", 0.2, 0.5, 0.5),
        _record("u3", 0.2, 0.5, 0.5), _record("u4", 0.2, 0.0, 0.5, chunk_status="not_computable"),
    ]
    pooled = sep._pool_contrast_chunk(records, sep.PRIMARY_CHUNK_SIZE_TRIALS, "seed-b")
    assert pooled["status"] == "not_computable"  # only 3 units carry a computed contrast -> below floor


def test_pool_contrast_quarter_matches_manual_difference():
    records = [_record(f"u{i}", 0.4, 0.5, 0.1) for i in range(4)]
    pooled = sep._pool_contrast_quarter(records, "seed-c")
    assert pooled["status"] == "computed"
    assert pooled["mean"] == pytest.approx(0.4 - 0.1)


def test_pool_quarter_reference_matches_manual_mean():
    records = [_record(f"u{i}", 0.4, 0.5, 0.2) for i in range(4)]
    pooled = sep._pool_quarter_reference(records, "seed-d")
    assert pooled["mean"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------------------------------
# _reproduction_check
# ---------------------------------------------------------------------------------------------------

def test_reproduction_check_flags_a_real_mismatch():
    delivered = {"status": "computed", "mean": 0.30}
    same = {"status": "computed", "mean": 0.30}
    different = {"status": "computed", "mean": 0.31}
    assert sep._reproduction_check(same, delivered)["matches"]
    assert not sep._reproduction_check(different, delivered)["matches"]


def test_reproduction_check_requires_both_sides_computed():
    delivered = {"status": "computed", "mean": 0.3}
    reproduced = {"status": "not_computable"}
    assert not sep._reproduction_check(reproduced, delivered)["matches"]


# ---------------------------------------------------------------------------------------------------
# _interleaved_only_branch / _branch: the pre-declared outcomes, including the "between" case
# ---------------------------------------------------------------------------------------------------

def _contrast(lo, hi, mdd):
    return {"status": "computed", "cluster_bootstrap_interval_95pct": [lo, hi], "minimum_detectable_difference_80pct_power": mdd}


def test_branch_rotates_beyond_noise_when_below_both_references():
    interleaved = _contrast(-0.3, -0.1, 0.05)
    quarter = _contrast(-0.2, -0.05, 0.05)
    assert sep._branch(interleaved, quarter, 0.1) == "rotates_beyond_noise"


def test_branch_undetermined_between_references_when_only_interleaved_clears():
    interleaved = _contrast(-0.3, -0.1, 0.05)
    quarter = _contrast(-0.05, 0.1, 0.05)  # quarter contrast covers zero -- does not clear
    assert sep._branch(interleaved, quarter, 0.1) == "undetermined_between_references"


def test_branch_time_separation_makes_no_difference_at_parity_and_well_powered():
    interleaved = _contrast(-0.02, 0.02, 0.03)
    quarter = _contrast(-0.3, -0.1, 0.05)
    assert sep._branch(interleaved, quarter, 0.1) == "time_separation_makes_no_difference"


def test_branch_undetermined_when_underpowered_at_parity():
    interleaved = _contrast(-0.15, 0.15, 0.2)
    quarter = _contrast(-0.3, -0.1, 0.05)
    assert sep._branch(interleaved, quarter, 0.1) == "undetermined_underpowered"


def test_branch_undetermined_for_the_unexpected_positive_direction():
    interleaved = _contrast(0.05, 0.2, 0.05)
    quarter = _contrast(-0.3, -0.1, 0.05)
    assert sep._branch(interleaved, quarter, 0.1) == "undetermined_unexpected_direction"


def test_branch_undetermined_below_the_independent_unit_floor():
    not_computable = {"status": "not_computable"}
    assert sep._branch(not_computable, not_computable, 0.1) == "undetermined_below_four_independent_unit_floor"


# ---------------------------------------------------------------------------------------------------
# data-dependent: exact reproduction of one real cell's delivered early-to-late overlap
# ---------------------------------------------------------------------------------------------------

@requires_data_root
def test_reproduces_one_delivered_cell_across_half_overlap_exactly():
    import json

    from corpus_sessions import data_root
    from run_alignment_below_null_diagnostic import OUTPUT_PATH as DELIVERED_DIAGNOSTIC_OUTPUT_PATH

    delivered = json.loads(DELIVERED_DIAGNOSTIC_OUTPUT_PATH.read_text())
    corpus, candidate = "inagaki_alm5_mouse_ALM", "gain_total_spike_count"
    delivered_cell = delivered["cells"][corpus][candidate]["early_late_subspace_overlap"]
    assert delivered_cell["status"] == "computed"

    root = data_root()
    per_candidate = sep._corpus_cells(root, corpus, [candidate], {"test": "reproduction"})
    records = per_candidate[candidate]["records"]
    reproduced = sep._pool_across_half(records, "test-reproduction-seed")
    check = sep._reproduction_check(reproduced, delivered_cell)
    assert check["matches"], check
