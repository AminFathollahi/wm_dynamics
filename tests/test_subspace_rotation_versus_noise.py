"""Tests for scripts/run_subspace_rotation_versus_noise.py.

Three things could silently break this module without erroring: (1) the within-half bootstrap
reference could fail to actually use the sample size an across-half fit uses, making the "matched
sample size" claim false; (2) the reference could fail to discriminate a genuinely rotating relation
from a stable one, which would make the whole contrast meaningless; (3) the branch classifier or the
pooling could mis-weight sessions or independent units, reproducing the exact bug
results/rank_free_component_identity.json's own pooling already guards against. A fourth group
reproduces the delivered diagnostic's early-to-late overlap against real data when the external data
drive is mounted, and is skipped otherwise.

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

import run_subspace_rotation_versus_noise as rotation_noise  # noqa: E402
from subspace_identity import regression_basis  # noqa: E402

DATA_ROOT_AVAILABLE = bool(os.environ.get("WM_DYNAMICS_DATA_ROOT")) and Path(
    os.environ.get("WM_DYNAMICS_DATA_ROOT", "")
).is_dir()
requires_data_root = pytest.mark.skipif(
    not DATA_ROOT_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set to a currently-mounted directory",
)


def _rotating_relation(rng, n=200, n_units=8):
    """A label relation that lives along unit 0 in the session's first half and unit 1 in the
    second half -- the across-half fits point in genuinely different directions, so this is the
    positive control for rotation."""
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    first_half = np.arange(n) < n // 2
    activity[first_half, 0] += 3.0 * target[first_half]
    activity[~first_half, 1] += 3.0 * target[~first_half]
    return activity, target


def _stable_relation(rng, n=200, n_units=8):
    """The same label relation holds for every trial -- the negative control: any across-half
    disagreement here is estimation noise, not rotation."""
    target = rng.normal(size=n)
    activity = rng.normal(0.0, 1.0, size=(n, n_units))
    activity[:, 0] += 5.0 * target
    return activity, target


# ---------------------------------------------------------------------------------------------------
# _bootstrap_self_overlap: matched sample size, deterministic, high overlap for a stable relation
# ---------------------------------------------------------------------------------------------------

def test_bootstrap_self_overlap_uses_the_input_sample_size():
    rng = np.random.default_rng(3)
    activity, target = _stable_relation(rng, n=80)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    result = rotation_noise._bootstrap_self_overlap(directions, target, regression_basis, "seed-a")
    assert result is not None
    assert result["n_samples"] == 80
    assert result["n_draws_used"] == rotation_noise.WITHIN_HALF_BOOTSTRAP_DRAWS


def test_bootstrap_self_overlap_is_deterministic_for_the_same_seed():
    rng = np.random.default_rng(4)
    activity, target = _stable_relation(rng, n=60)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    first = rotation_noise._bootstrap_self_overlap(directions, target, regression_basis, "seed-b")
    second = rotation_noise._bootstrap_self_overlap(directions, target, regression_basis, "seed-b")
    assert first == second


def test_bootstrap_self_overlap_high_for_a_strong_stable_relation():
    rng = np.random.default_rng(5)
    activity, target = _stable_relation(rng, n=200)
    directions = activity / np.linalg.norm(activity, axis=1, keepdims=True)
    result = rotation_noise._bootstrap_self_overlap(directions, target, regression_basis, "seed-c")
    assert result["mean_overlap"] > 0.8


def test_bootstrap_self_overlap_none_below_minimum_size():
    result = rotation_noise._bootstrap_self_overlap(np.zeros((3, 4)), np.arange(3, dtype=float), regression_basis, "seed-d")
    assert result is None


# ---------------------------------------------------------------------------------------------------
# _cell_session / _within_half_reference: the contrast discriminates rotation from noise
# ---------------------------------------------------------------------------------------------------

def test_within_half_reference_near_across_half_overlap_for_a_stable_relation():
    activity, target = _stable_relation(np.random.default_rng(6))
    cell = rotation_noise._cell_session(activity, "continuous", target, "test|stable")
    assert cell["status"] == "computed"
    within = cell["within_half_reference"]
    assert within["status"] == "computed"
    contrast = cell["across_half_overlap"] - within["mean"]
    assert abs(contrast) < 0.15  # parity: no true rotation, so noise alone should not separate them


def test_within_half_reference_well_above_across_half_overlap_for_a_rotating_relation():
    activity, target = _rotating_relation(np.random.default_rng(7))
    cell = rotation_noise._cell_session(activity, "continuous", target, "test|rotating")
    assert cell["status"] == "computed"
    within = cell["within_half_reference"]
    assert within["status"] == "computed"
    contrast = cell["across_half_overlap"] - within["mean"]
    assert contrast < -0.3  # rotation drives across-half overlap well below the matched noise floor


def test_cell_session_too_few_trials_is_not_computable():
    cell = rotation_noise._cell_session(np.ones((3, 4)), "continuous", np.arange(3, dtype=float), "test|short")
    assert cell["status"] == "too_few_trials"


# ---------------------------------------------------------------------------------------------------
# pooling: equal-session-then-equal-unit weighting, matching the delivered pooling
# ---------------------------------------------------------------------------------------------------

def _record(unit, across, within_mean, within_status="computed"):
    return {
        "independent_unit": unit, "across_half_overlap": across,
        "within_half_reference": {"status": within_status, "mean": within_mean},
    }


def test_pool_contrast_weights_independent_units_not_sessions():
    records = [
        _record("u1", 0.2, 0.5), _record("u1", 0.2, 0.5),  # two sessions, same unit
        _record("u2", 0.2, 0.5), _record("u3", 0.2, 0.5), _record("u4", 0.2, 0.5),
    ]
    pooled = rotation_noise._pool_contrast(records, "seed-e")
    assert pooled["status"] == "computed"
    assert pooled["mean"] == pytest.approx(-0.3)
    assert pooled["n_sessions_paired"] == 5


def test_pool_contrast_drops_sessions_without_a_computed_within_half_reference():
    records = [
        _record("u1", 0.2, 0.5), _record("u2", 0.2, 0.5),
        _record("u3", 0.2, 0.5), _record("u4", 0.2, 0.0, within_status="not_computable"),
    ]
    pooled = rotation_noise._pool_contrast(records, "seed-f")
    assert pooled["n_sessions_paired"] == 3
    assert pooled["status"] == "not_computable"  # only 3 units carry a paired value -> below the floor


def test_pool_from_records_matches_pool_contrast_inputs():
    records = [_record("u1", 0.4, 0.1), _record("u2", 0.6, 0.1), _record("u3", 0.5, 0.1), _record("u4", 0.5, 0.1)]
    pooled = rotation_noise._pool_from_records(records, "across_half_overlap", "seed-g")
    assert pooled["n_independent_units"] == 4
    assert pooled["mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------------
# _reproduction_check
# ---------------------------------------------------------------------------------------------------

def test_reproduction_check_flags_a_real_mismatch():
    delivered = {"status": "computed", "mean": 0.30}
    same = {"status": "computed", "mean": 0.30}
    different = {"status": "computed", "mean": 0.31}
    assert rotation_noise._reproduction_check(same, delivered)["matches"]
    assert not rotation_noise._reproduction_check(different, delivered)["matches"]


def test_reproduction_check_requires_both_sides_computed():
    delivered = {"status": "computed", "mean": 0.3}
    reproduced = {"status": "not_computable"}
    assert not rotation_noise._reproduction_check(reproduced, delivered)["matches"]


# ---------------------------------------------------------------------------------------------------
# _branch: the three pre-declared outcomes
# ---------------------------------------------------------------------------------------------------

def test_branch_rotates_beyond_noise_when_interval_is_entirely_negative():
    contrast = {"status": "computed", "cluster_bootstrap_interval_95pct": [-0.3, -0.1], "minimum_detectable_difference_80pct_power": 0.05}
    assert rotation_noise._branch(contrast, 0.1) == "rotates_beyond_noise"


def test_branch_noise_explains_low_overlap_at_parity_and_well_powered():
    contrast = {"status": "computed", "cluster_bootstrap_interval_95pct": [-0.02, 0.02], "minimum_detectable_difference_80pct_power": 0.03}
    assert rotation_noise._branch(contrast, 0.1) == "noise_explains_low_overlap"


def test_branch_undetermined_when_underpowered_at_parity():
    contrast = {"status": "computed", "cluster_bootstrap_interval_95pct": [-0.15, 0.15], "minimum_detectable_difference_80pct_power": 0.2}
    assert rotation_noise._branch(contrast, 0.1) == "undetermined_underpowered"


def test_branch_undetermined_for_the_unexpected_positive_direction():
    contrast = {"status": "computed", "cluster_bootstrap_interval_95pct": [0.05, 0.2], "minimum_detectable_difference_80pct_power": 0.05}
    assert rotation_noise._branch(contrast, 0.1) == "undetermined_unexpected_direction"


def test_branch_undetermined_below_the_independent_unit_floor():
    assert rotation_noise._branch({"status": "not_computable"}, 0.1) == "undetermined_below_four_independent_unit_floor"


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
    per_candidate = rotation_noise._corpus_cells(root, corpus, [candidate], {"test": "reproduction"})
    records = per_candidate[candidate]["records"]
    reproduced = rotation_noise._pool_from_records(records, "across_half_overlap", "test-reproduction-seed")
    check = rotation_noise._reproduction_check(reproduced, delivered_cell)
    assert check["matches"], check
