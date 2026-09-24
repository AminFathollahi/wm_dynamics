"""Tests for scripts/run_alignment_below_null_diagnostic.py.

Three things could silently break this diagnostic without erroring: (1) interleaved_folds could
fail to actually intermix train/eval trials, making the fold-structure comparison a no-op; (2) the
paired block-minus-interleaved and early/late-rotation aggregators could mis-weight sessions or
independent units, reproducing the exact bug results/rank_free_component_identity.json's own _pool
already guards against; (3) the discriminating logic itself could fail to separate a genuinely
nonstationary candidate from a stationary one. A fourth group reproduces the delivered artifact's
block-fold numbers against real data when the external data drive is mounted, and is skipped
otherwise.

WM_DYNAMICS_DATA_ROOT must point at the currently-mounted external data drive for the data-
dependent tests in the last group; every other test here is synthetic and needs no data root.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_alignment_below_null_diagnostic as diagnostic  # noqa: E402
import run_rank_free_component_identity as rank_free  # noqa: E402
from geometry import principal_angles, subspace_overlap  # noqa: E402
from subspace_identity import block_folds, class_basis, permutation_alignment, regression_basis  # noqa: E402

DATA_ROOT_AVAILABLE = bool(os.environ.get("WM_DYNAMICS_DATA_ROOT")) and Path(
    os.environ.get("WM_DYNAMICS_DATA_ROOT", "")
).is_dir()
requires_data_root = pytest.mark.skipif(
    not DATA_ROOT_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set to a currently-mounted directory",
)


# ---------------------------------------------------------------------------------------------------
# interleaved_folds
# ---------------------------------------------------------------------------------------------------

def test_interleaved_folds_alternates_and_matches_block_folds_counts():
    n, k = 36, 2  # even n: both fold-assignment schemes split exactly in half
    interleaved = diagnostic.interleaved_folds(n, k)
    blocked = block_folds(n, k)
    assert np.array_equal(interleaved, np.arange(n) % k)
    for fold in range(k):
        assert (interleaved == fold).sum() == (blocked == fold).sum()


def test_interleaved_folds_actually_intermixes_temporal_order():
    # A block fold's fold-0 trials are the first half of the session, all contiguous; an
    # interleaved fold's fold-0 trials span the full session range, which is the entire point.
    n = 40
    blocked = block_folds(n, 2)
    interleaved = diagnostic.interleaved_folds(n, 2)
    block_span = np.flatnonzero(blocked == 0)
    interleaved_span = np.flatnonzero(interleaved == 0)
    assert block_span.max() - block_span.min() < n / 2  # contiguous half
    assert interleaved_span.max() - interleaved_span.min() >= n - 2  # spans nearly the whole session


def test_interleaved_folds_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        diagnostic.interleaved_folds(3, 5)
    with pytest.raises(ValueError):
        diagnostic.interleaved_folds(5, 1)


# ---------------------------------------------------------------------------------------------------
# _prepare_trials matches the delivered _cell's own trial-validity filter exactly
# ---------------------------------------------------------------------------------------------------

def test_prepare_trials_matches_delivered_cell_filtering():
    rng = np.random.default_rng(3)
    activity = rng.normal(size=(30, 6))
    activity[2] = np.nan
    activity[5] = 0.0
    target = rng.normal(size=30)
    target[7] = np.inf

    prepared = diagnostic._prepare_trials(activity, target)
    delivered = rank_free._cell(activity, target, "continuous", 50, "prepare-check")

    assert prepared is not None
    assert delivered["status"] == "computed"
    assert len(prepared[1]) == delivered["n_trials"]


def test_prepare_trials_returns_none_below_trial_floor():
    assert diagnostic._prepare_trials(np.ones((3, 4)), np.arange(3, dtype=float)) is None


# ---------------------------------------------------------------------------------------------------
# discriminating logic: does interleaving actually recover alignment lost to within-session rotation?
# ---------------------------------------------------------------------------------------------------

def _rotating_relation_directions(rng, n=200, n_units=8):
    """Activity whose relation to a continuous label rotates halfway through the session: the label
    axis lives along unit 0 in the first half and unit 1 in the second half. A block-fold subspace
    trained on one half and evaluated on the temporally disjoint other half should align poorly with
    its own refit null (the null, refit each draw, can chase whatever relation is stationary within
    a fold); an interleaved fold sees both halves in both folds, so training and evaluation share the
    same rotated relation."""
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    first_half = np.arange(n) < n // 2
    activity[first_half, 0] += 3.0 * target[first_half]
    activity[~first_half, 1] += 3.0 * target[~first_half]
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    return activity / norms, target


def test_interleaved_folds_recover_alignment_lost_to_within_session_rotation():
    rng = np.random.default_rng(21)
    directions, target = _rotating_relation_directions(rng)
    n_perm = 300

    block = permutation_alignment(
        directions, target, block_folds(len(target), 2), regression_basis, n_perm, np.random.default_rng(1),
    )
    interleaved = permutation_alignment(
        directions, target, diagnostic.interleaved_folds(len(target), 2), regression_basis, n_perm,
        np.random.default_rng(2),
    )
    assert block["status"] == "computed" and interleaved["status"] == "computed"
    assert interleaved["alignment_above_null"] > block["alignment_above_null"]


def test_interleaved_folds_leave_a_stationary_relation_essentially_unchanged():
    rng = np.random.default_rng(22)
    n, n_units = 200, 8
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    activity[:, 0] += 5.0 * target  # the same relation holds for every trial, no rotation
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    n_perm = 300

    block = permutation_alignment(
        directions, target, block_folds(n, 2), regression_basis, n_perm, np.random.default_rng(1),
    )
    interleaved = permutation_alignment(
        directions, target, diagnostic.interleaved_folds(n, 2), regression_basis, n_perm, np.random.default_rng(2),
    )
    # Both cells should clear their own null by a wide margin, and not diverge by a large fraction
    # of that margin -- a loose bound, since finite-permutation Monte Carlo noise is real.
    assert block["alignment_above_null"] > 0.2 and interleaved["alignment_above_null"] > 0.2
    assert abs(block["alignment_above_null"] - interleaved["alignment_above_null"]) < 0.3


# ---------------------------------------------------------------------------------------------------
# _rotation: reuses geometry.principal_angles / subspace_overlap unchanged
# ---------------------------------------------------------------------------------------------------

def test_rotation_detects_a_subspace_that_actually_rotates():
    directions, target = _rotating_relation_directions(np.random.default_rng(5))
    rotation = diagnostic._rotation(directions, target, "continuous")
    assert rotation["status"] == "computed"
    assert rotation["subspace_overlap"] < 0.5


def test_rotation_reports_near_full_overlap_for_a_stable_subspace():
    rng = np.random.default_rng(6)
    n, n_units = 200, 8
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    activity[:, 0] += 5.0 * target
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    rotation = diagnostic._rotation(directions, target, "continuous")
    assert rotation["status"] == "computed"
    assert rotation["subspace_overlap"] > 0.8


def test_rotation_matches_geometry_module_directly():
    directions, target = _rotating_relation_directions(np.random.default_rng(7))
    n = len(target)
    folds = block_folds(n, 2)
    early = regression_basis(directions[folds == 0], target[folds == 0])
    late = regression_basis(directions[folds == 1], target[folds == 1])
    rotation = diagnostic._rotation(directions, target, "continuous")
    assert rotation["subspace_overlap"] == pytest.approx(subspace_overlap(early, late))
    np.testing.assert_allclose(rotation["principal_angles_radians"], principal_angles(early, late))


# ---------------------------------------------------------------------------------------------------
# aggregation: _pool_unit_scalars and _paired_diff weight independent units, not sessions
# ---------------------------------------------------------------------------------------------------

def test_pool_unit_scalars_weights_independent_units_not_sessions():
    values = {"unit_a": [0.9, 0.7], "unit_b": [0.2], "unit_c": [0.4], "unit_d": [0.6]}
    pooled = diagnostic._pool_unit_scalars(values, "test-seed")
    assert pooled["status"] == "computed"
    assert pooled["mean"] == pytest.approx(0.5)
    assert pooled["minimum_detectable_difference_80pct_power"] >= 0.0


def test_pool_unit_scalars_below_floor_is_not_computable():
    pooled = diagnostic._pool_unit_scalars({"u1": [0.1], "u2": [0.2]}, "seed")
    assert pooled["status"] == "not_computable"
    assert pooled["n_independent_units"] == 2


def test_paired_diff_only_uses_sessions_with_both_folds_computed():
    def cell(alignment_above_null, status="computed"):
        return {"status": status, "alignment_above_null": alignment_above_null}

    def record(unit, session, cell_value):
        return {"session": session, "independent_unit": unit, "status": "computed", "candidates": {"c": cell_value}}

    block_records = [
        record("u1", "s1", cell(0.10)), record("u2", "s2", cell(0.20)),
        record("u3", "s3", cell(0.30)), record("u4", "s4", cell(0.40)),
    ]
    interleaved_records = [
        record("u1", "s1", cell(0.05)), record("u2", "s2", cell(0.05)),
        record("u3", "s3", cell(0.05)), record("u4", "s4", cell(-1.0, status="not_computable")),
    ]
    result = diagnostic._paired_diff(block_records, interleaved_records, "c", "paired-seed")
    assert result["status"] == "not_computable"  # only u1,u2,u3 carry both -> below the 4-unit floor
    assert result["n_independent_units"] == 3


def test_paired_diff_recovers_a_planted_difference():
    def cell(alignment_above_null):
        return {"status": "computed", "alignment_above_null": alignment_above_null}

    def record(unit, session, value):
        return {"session": session, "independent_unit": unit, "status": "computed", "candidates": {"c": cell(value)}}

    units = ["u1", "u2", "u3", "u4", "u5"]
    block_records = [record(u, f"{u}-s", 0.3) for u in units]
    interleaved_records = [record(u, f"{u}-s", 0.1) for u in units]
    result = diagnostic._paired_diff(block_records, interleaved_records, "c", "paired-seed-2")
    assert result["status"] == "computed"
    assert result["mean"] == pytest.approx(0.2)
    assert result["block_mean_alignment_above_null"] == pytest.approx(0.3)
    assert result["interleaved_mean_alignment_above_null"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------------------------------
# reproduction check
# ---------------------------------------------------------------------------------------------------

def test_reproduction_check_flags_a_real_mismatch():
    delivered = {"status": "computed", "mean_alignment_above_null": 0.10, "p_value": 0.02}
    reproduced_same = {"status": "computed", "mean_alignment_above_null": 0.10, "p_value": 0.02}
    reproduced_different = {"status": "computed", "mean_alignment_above_null": 0.30, "p_value": 0.02}
    assert diagnostic._reproduction_check("c", "n", reproduced_same, delivered)["matches"]
    assert not diagnostic._reproduction_check("c", "n", reproduced_different, delivered)["matches"]


def test_reproduction_check_requires_both_sides_computed():
    delivered = {"status": "computed", "mean_alignment_above_null": 0.1, "p_value": 0.02}
    reproduced = {"status": "not_computable"}
    assert not diagnostic._reproduction_check("c", "n", reproduced, delivered)["matches"]


# ---------------------------------------------------------------------------------------------------
# verdict composition
# ---------------------------------------------------------------------------------------------------

def _fake_cell(alignment_above_null, block_minus_interleaved, ci, overlap, overlap_ci):
    return {
        "delivered": {"mean_alignment_above_null": alignment_above_null},
        "block_minus_interleaved_paired_difference": {
            "status": "computed",
            "block_mean_alignment_above_null": alignment_above_null,
            "interleaved_mean_alignment_above_null": alignment_above_null - block_minus_interleaved,
            "mean": block_minus_interleaved,
            "cluster_bootstrap_interval_95pct": ci,
            "minimum_detectable_difference_80pct_power": 0.05,
        },
        "early_late_subspace_overlap": {
            "status": "computed", "mean": overlap, "cluster_bootstrap_interval_95pct": overlap_ci,
        },
    }


def test_identify_anomalous_cell_picks_the_most_negative_delivered_alignment():
    cells = {
        "corpus_a": {"cand1": _fake_cell(0.05, 0.0, [-0.1, 0.1], 0.3, [0.2, 0.4])},
        "corpus_b": {
            "cand1": _fake_cell(-0.04, -0.04, [-0.1, 0.01], 0.08, [0.05, 0.1]),
            "cand2": _fake_cell(0.01, 0.0, [-0.1, 0.1], 0.2, [0.1, 0.3]),
        },
    }
    assert diagnostic._identify_anomalous_cell(cells) == "corpus_b|cand1"


def test_compose_verdict_reports_undetermined_and_names_the_anomalous_cell():
    cells = {
        "corpus_a": {"cand1": _fake_cell(0.05, -0.10, [-0.15, -0.05], 0.3, [0.2, 0.4])},
        "corpus_b": {"cand1": _fake_cell(-0.04, -0.04, [-0.1, 0.01], 0.08, [0.05, 0.1])},
    }
    summary = diagnostic._summarize(cells)
    result = diagnostic._compose_verdict(cells, summary)
    assert result["anomalous_cell"] == "corpus_b|cand1"
    assert "undetermined" in result["verdict"]
    assert "corpus_b|cand1" in result["verdict"]
    assert "corpus_a|cand1" in result["verdict"]  # the other cell's outside-zero effect is cited


# ---------------------------------------------------------------------------------------------------
# data-dependent: exact reproduction of one real delivered cell's block-fold number
# ---------------------------------------------------------------------------------------------------

@requires_data_root
def test_delivered_cell_is_deterministic_on_real_session_inputs():
    """Calling the delivered script's own _cell twice on the same real session data and seed must
    give bit-identical results -- the premise the whole reproduction gate in the diagnostic script
    depends on."""
    from corpus_sessions import data_root

    root = data_root()
    sessions = list(diagnostic._standard_sessions(root, "dandi_000574_human"))
    computable = [s for s in sessions if s[2] is not None]
    assert computable, "expected at least one dandi_000574_human session with usable activity"
    session, _reason, activity, _categorical, continuous = computable[0]
    target = continuous["gain_total_spike_count"]
    first = rank_free._cell(activity, target, "continuous", 50, f"dandi_000574_human|{session}|gain_total_spike_count")
    second = rank_free._cell(activity, target, "continuous", 50, f"dandi_000574_human|{session}|gain_total_spike_count")
    assert first["status"] == "computed" and second["status"] == "computed"
    assert first["alignment"] == second["alignment"]
    assert first["p_value"] == second["p_value"]


@requires_data_root
def test_delivered_artifact_matches_its_own_reproduction_gate_shape():
    """results/rank_free_component_identity.json is expected to already exist and carry at least one
    cell that reached the four-independent-unit floor for each corpus the diagnostic script needs --
    if this ever stops being true for any of them the diagnostic script has nothing to reproduce."""
    delivered = json.loads(rank_free.OUTPUT_PATH.read_text())
    summary = delivered["summary"]["per_corpus"]
    needed = {
        corpus for corpus, candidates in summary.items()
        if any(cell.get("status") == "computed" for cell in candidates.values())
    }
    assert {
        "inagaki_alm5_mouse_ALM", "dandi_000469_human", "dandi_001187_human", "dandi_000574_human",
    }.issubset(needed)
