"""Tests for scripts/run_readout_transfer_decay.py.

Things that could silently break this module without erroring: (1) chunk assignment could drop or
duplicate trials; (2) the read-out performance measurement could leak chunk j into the chunk i fit;
(3) the circular-shift null could fail to destroy a real gap-performance relation, or could destroy
signal it should not touch; (4) pooling could mis-weight sessions or independent units; (5) the
clearing/bound/inconclusive branch could mis-classify a cell; (6) the reproduction-gate comparator
could pass on a real mismatch. A final group runs the real pipeline end to end on the mounted data
drive with a tiny session cap and is skipped otherwise.

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

import run_readout_transfer_decay as decay  # noqa: E402

DATA_ROOT_AVAILABLE = bool(os.environ.get("WM_DYNAMICS_DATA_ROOT")) and Path(
    os.environ.get("WM_DYNAMICS_DATA_ROOT", "")
).is_dir()
requires_data_root = pytest.mark.skipif(
    not DATA_ROOT_AVAILABLE, reason="WM_DYNAMICS_DATA_ROOT is not set to a currently-mounted directory",
)


# ---------------------------------------------------------------------------------------------------
# _chunk_assign
# ---------------------------------------------------------------------------------------------------

def test_chunk_assign_covers_every_trial_exactly_once():
    chunks, centres = decay._chunk_assign(23, 4)
    all_indices = np.concatenate(chunks)
    assert np.array_equal(np.sort(all_indices), np.arange(23))
    assert len(chunks) == 6  # ceil(23/4)
    assert len(centres) == 6


def test_chunk_assign_centres_increase_monotonically():
    _, centres = decay._chunk_assign(40, 4)
    assert np.all(np.diff(centres) > 0)


# ---------------------------------------------------------------------------------------------------
# _ols_slope_intercept
# ---------------------------------------------------------------------------------------------------

def test_ols_matches_numpy_polyfit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 2.5 * x - 1.0 + rng.normal(scale=0.1, size=50)
    slope, intercept = decay._ols_slope_intercept(x, y)
    ref_slope, ref_intercept = np.polyfit(x, y, 1)
    assert slope == pytest.approx(ref_slope, abs=1e-9)
    assert intercept == pytest.approx(ref_intercept, abs=1e-9)


# ---------------------------------------------------------------------------------------------------
# _session_chunk_cell: chunk j is never seen by the chunk i fit, and a planted decay is recoverable
# ---------------------------------------------------------------------------------------------------

def _no_decay_session(rng, n=120, n_units=8):
    target = rng.gamma(4.0, 2.0, size=n)
    directions = rng.normal(size=(n, n_units))
    directions[:, 0] += 2.0 * (target - target.mean())
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions, target


def _planted_decay_session(rng, n=200, n_units=8):
    """The read-out direction rotates smoothly across the session, so performance measured with a
    training chunk far from the evaluation chunk should be systematically worse than with a nearby
    training chunk -- a real, planted gap-performance slope."""
    target = rng.gamma(4.0, 2.0, size=n)
    phase = np.linspace(0.0, np.pi / 2, n)
    directions = rng.normal(scale=0.3, size=(n, n_units))
    centred = target - target.mean()
    directions[:, 0] += np.cos(phase) * 3.0 * centred
    directions[:, 1] += np.sin(phase) * 3.0 * centred
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions, target


def test_session_chunk_cell_not_computable_below_minimum_chunks():
    rng = np.random.default_rng(1)
    directions, target = _no_decay_session(rng, n=10)
    cell = decay._session_chunk_cell(directions, target, chunk_size=16, rng=np.random.default_rng(2))
    assert cell["status"] == "not_computable"


def test_session_chunk_cell_no_decay_relation_gives_small_slope():
    rng = np.random.default_rng(3)
    directions, target = _no_decay_session(rng)
    cell = decay._session_chunk_cell(directions, target, chunk_size=4, rng=np.random.default_rng(4))
    assert cell["status"] == "computed"
    assert cell["mean_performance"] > 0.3  # the stable relation transfers well regardless of gap
    assert abs(cell["observed_slope"]) < cell["mean_performance"] / 10


def test_session_chunk_cell_planted_decay_gives_negative_slope():
    rng = np.random.default_rng(5)
    directions, target = _planted_decay_session(rng)
    cell = decay._session_chunk_cell(directions, target, chunk_size=4, rng=np.random.default_rng(6))
    assert cell["status"] == "computed"
    assert cell["observed_slope"] < 0.0


def test_session_chunk_cell_never_lets_evaluation_chunk_influence_the_fit(monkeypatch):
    rng = np.random.default_rng(7)
    directions, target = _no_decay_session(rng, n=40)
    calls = []
    original = decay.regression_basis

    def spy(chunk_directions, chunk_target):
        calls.append((chunk_directions.copy(), chunk_target.copy()))
        return original(chunk_directions, chunk_target)

    monkeypatch.setattr(decay, "regression_basis", spy)
    chunks, _ = decay._chunk_assign(40, 4)
    decay._session_chunk_cell(directions, target, chunk_size=4, rng=np.random.default_rng(8))
    assert len(calls) == len(chunks)  # exactly one basis fit per chunk, never per pair
    for fitted_directions, fitted_target, chunk in zip(
        [c[0] for c in calls], [c[1] for c in calls], chunks,
    ):
        assert np.array_equal(fitted_directions, directions[chunk])
        assert np.array_equal(fitted_target, target[chunk])


# ---------------------------------------------------------------------------------------------------
# _circular_shift_null_slopes: destroys a real relation, is symmetric under no relation
# ---------------------------------------------------------------------------------------------------

def test_circular_shift_null_centered_near_zero_for_a_stable_relation():
    rng = np.random.default_rng(9)
    directions, target = _no_decay_session(rng, n=120)
    cell = decay._session_chunk_cell(directions, target, chunk_size=4, rng=np.random.default_rng(10))
    assert cell["status"] == "computed"
    assert abs(np.mean(cell["null_slopes"])) < 0.02


def test_circular_shift_null_does_not_reproduce_a_planted_decay_slope():
    rng = np.random.default_rng(11)
    directions, target = _planted_decay_session(rng)
    cell = decay._session_chunk_cell(directions, target, chunk_size=4, rng=np.random.default_rng(12))
    assert cell["status"] == "computed"
    observed = cell["observed_slope"]
    null = cell["null_slopes"]
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (null.size + 1)
    assert p < 0.05  # the planted decay is far outside its own circular-shift null


def test_circular_shift_changes_gap_values_for_at_least_one_shift():
    centres = np.array([1.5, 5.5, 9.5, 13.5, 17.5])
    pair_i = np.array([0, 1, 2])
    pair_j = np.array([4, 3, 0])
    observed_gaps = np.abs(centres[pair_j] - centres[pair_i])
    shifted_gaps_by_offset = []
    for shift in (1, 2, 3, 4):
        shifted = np.roll(centres, shift)
        shifted_gaps_by_offset.append(np.abs(shifted[pair_j] - shifted[pair_i]))
    assert any(not np.array_equal(observed_gaps, g) for g in shifted_gaps_by_offset)


# ---------------------------------------------------------------------------------------------------
# pooling: equal-session-then-equal-unit weighting, mirroring the rotation module's own tests
# ---------------------------------------------------------------------------------------------------

def _fake_cell(slope, intercept, n_trials=40, null_slopes=None):
    if null_slopes is None:
        null_slopes = np.random.default_rng(0).normal(0.0, 0.01, size=200)
    return {
        "status": "computed", "observed_slope": slope, "observed_intercept": intercept,
        "n_trials": n_trials, "null_slopes": null_slopes,
    }


def test_pool_slope_cell_weights_independent_units_not_sessions():
    sessions = [
        {"session": "s1", "independent_unit": "u1", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.0, 0.1)}}},
        {"session": "s2", "independent_unit": "u1", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.2, 0.1)}}},
        {"session": "s3", "independent_unit": "u2", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.6, 0.1)}}},
        {"session": "s4", "independent_unit": "u3", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.6, 0.1)}}},
        {"session": "s5", "independent_unit": "u4", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.6, 0.1)}}},
    ]
    pooled = decay._pool_slope_cell(sessions, 4, "seed-a")
    assert pooled["status"] == "computed"
    assert pooled["slope"]["mean"] == pytest.approx((0.1 + 0.6 + 0.6 + 0.6) / 4)


def test_pool_slope_cell_not_computable_below_independent_unit_floor():
    sessions = [
        {"session": "s1", "independent_unit": "u1", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.1, 0.0)}}},
        {"session": "s2", "independent_unit": "u2", "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.1, 0.0)}}},
    ]
    pooled = decay._pool_slope_cell(sessions, 4, "seed-b")
    assert pooled["status"] == "not_computable"
    assert pooled["n_independent_units"] == 2


def test_pool_slope_cell_skips_sessions_with_uncomputed_chunk_size():
    sessions = [
        {"session": f"s{i}", "independent_unit": f"u{i}",
         "entry": {"status": "computed", "chunk_cells": {4: _fake_cell(0.3, 0.0)}}}
        for i in range(4)
    ]
    sessions.append({
        "session": "s5", "independent_unit": "u5",
        "entry": {"status": "computed", "chunk_cells": {4: {"status": "not_computable"}}},
    })
    pooled = decay._pool_slope_cell(sessions, 4, "seed-c")
    assert pooled["status"] == "computed"
    assert pooled["n_independent_units"] == 4
    assert pooled["n_sessions_not_computable_this_chunk_size"] == 1


def test_pool_null_draws_equal_weight_per_unit():
    sessions_by_unit = {
        "u1": [np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0, 4.0])],  # mean [1,2,3]
        "u2": [np.array([4.0, 4.0, 4.0])],
    }
    pooled = decay._pool_null_draws(sessions_by_unit)
    assert np.allclose(pooled, [(1 + 4) / 2, (2 + 4) / 2, (3 + 4) / 2])


# ---------------------------------------------------------------------------------------------------
# _cell_verdict: clears / bounded_negative / inconclusive / not_computable
# ---------------------------------------------------------------------------------------------------

def _pooled(mean, lo, hi, mdd, q, mean_trials=100.0):
    return {
        "status": "computed", "fdr_q_value": q, "mean_trials_per_contributing_session": mean_trials,
        "slope": {
            "mean": mean, "cluster_bootstrap_interval_95pct": [lo, hi],
            "minimum_detectable_difference_80pct_power": mdd,
        },
    }


def test_verdict_clears_when_q_significant_and_ci_excludes_zero():
    pooled = _pooled(-0.01, -0.02, -0.005, 0.001, q=0.01)
    verdict = decay._cell_verdict(pooled, {"mean_alignment_above_null": 0.05})
    assert verdict["verdict"] == "clears"


def test_verdict_does_not_clear_when_only_one_half_passes():
    pooled_q_only = _pooled(-0.01, -0.02, 0.001, 0.02, q=0.01)  # CI covers zero
    pooled_ci_only = _pooled(-0.01, -0.02, -0.005, 0.02, q=0.2)  # q not significant
    ref = {"mean_alignment_above_null": 0.05}
    assert decay._cell_verdict(pooled_q_only, ref)["verdict"] != "clears"
    assert decay._cell_verdict(pooled_ci_only, ref)["verdict"] != "clears"


def test_verdict_bounded_negative_when_mdd_below_reference_decay_rate():
    pooled = _pooled(-0.001, -0.02, 0.02, mdd=0.0001, q=0.5, mean_trials=100.0)
    reference = {"mean_alignment_above_null": 0.05}  # reference decay rate = 0.05/100 = 0.0005
    verdict = decay._cell_verdict(pooled, reference)
    assert verdict["verdict"] == "bounded_negative"
    assert verdict["minimum_detectable_slope_80pct_power"] == pytest.approx(0.0001)
    assert verdict["reference_decay_rate_per_trial"] == pytest.approx(0.05 / 100.0)


def test_verdict_inconclusive_when_mdd_above_reference_decay_rate():
    pooled = _pooled(-0.001, -0.02, 0.02, mdd=0.01, q=0.5, mean_trials=100.0)
    reference = {"mean_alignment_above_null": 0.05}  # reference decay rate = 0.0005
    verdict = decay._cell_verdict(pooled, reference)
    assert verdict["verdict"] == "inconclusive_underpowered_relative_to_reference"


def test_verdict_not_computable_passthrough():
    pooled = {"status": "not_computable", "reason": "fewer than 4 independent units"}
    verdict = decay._cell_verdict(pooled, None)
    assert verdict["verdict"] == "not_computable"


def test_verdict_uses_absolute_reference_effect_for_negative_alignment():
    pooled = _pooled(-0.001, -0.02, 0.02, mdd=0.0003, q=0.5, mean_trials=100.0)
    reference = {"mean_alignment_above_null": -0.04}  # negative delivered effect, e.g. below-null cell
    verdict = decay._cell_verdict(pooled, reference)
    assert verdict["reference_decay_rate_per_trial"] == pytest.approx(0.04 / 100.0)
    assert verdict["verdict"] == "bounded_negative"


# ---------------------------------------------------------------------------------------------------
# _compare_cells: the reproduction-gate comparator
# ---------------------------------------------------------------------------------------------------

def test_compare_cells_matches_identical_passes():
    cells = {"corpusA": {"candA": {"primary": _pooled(0.1, 0.0, 0.2, 0.01, q=0.02)}}}
    cells["corpusA"]["candA"]["primary"]["permutation_p_value"] = 0.004
    result = decay._compare_cells(cells, cells)
    assert result["matches"]


def test_compare_cells_flags_a_real_mismatch():
    a = {"corpusA": {"candA": {"primary": _pooled(0.1, 0.0, 0.2, 0.01, q=0.02)}}}
    a["corpusA"]["candA"]["primary"]["permutation_p_value"] = 0.004
    b = {"corpusA": {"candA": {"primary": _pooled(0.2, 0.0, 0.2, 0.01, q=0.02)}}}
    b["corpusA"]["candA"]["primary"]["permutation_p_value"] = 0.004
    result = decay._compare_cells(a, b)
    assert not result["matches"]
    assert result["mismatches"]


# ---------------------------------------------------------------------------------------------------
# null calibration: mechanism runs and returns a plausible rate (full 200-replicate run happens in
# the actual analysis execution, not here -- this just checks the calibration harness is correct)
# ---------------------------------------------------------------------------------------------------

def test_null_calibration_check_runs_and_returns_a_rate_in_bounds():
    result = decay._null_calibration(n_replicates=20, n_perm=100, seed="test-calibration", n_trials=48, n_units=6, n_bins=6)
    assert result["n_replicates_computed"] > 0
    assert 0.0 <= result["false_positive_rate_at_p_0.05"] <= 1.0


# ---------------------------------------------------------------------------------------------------
# _needed_by_corpus: reads the rotation module's own delivered cell list for 000574, not hardcoded
# ---------------------------------------------------------------------------------------------------

@requires_data_root
def test_needed_by_corpus_includes_gain_everywhere_and_field_potential_only_for_000574():
    needed = decay._needed_by_corpus()
    for corpus in ("inagaki_alm5_mouse_ALM", "dandi_000469_human", "dandi_001187_human",
                   "dandi_000574_human", "panichello_2024_macaque_lPFC", "watters_2026_macaque_multi_object"):
        assert "gain_total_spike_count" in needed[corpus]
    assert len(needed["dandi_000574_human"]) > 1
    for corpus in ("inagaki_alm5_mouse_ALM", "dandi_000469_human", "dandi_001187_human"):
        assert needed[corpus] == ["gain_total_spike_count"]


# ---------------------------------------------------------------------------------------------------
# end-to-end smoke test on real data, tiny session cap
# ---------------------------------------------------------------------------------------------------

@requires_data_root
def test_main_runs_end_to_end_on_a_small_slice(tmp_path):
    from corpus_sessions import data_root

    output_path = tmp_path / "readout_transfer_decay.json"
    checkpoint_dir = tmp_path / "checkpoints"
    sys.argv = [
        "run_readout_transfer_decay.py",
        "--max-sessions", "3",
        "--output", str(output_path),
        "--checkpoint-dir", str(checkpoint_dir),
        "--calibration-replicates", "5",
        "--calibration-n-perm", "50",
    ]
    decay.main()
    assert output_path.exists()
    import json
    result = json.loads(output_path.read_text())
    assert result["status"] in ("complete", "reproduction_gate_failed")
    assert result["reproduction_gate_passed"] is True
    assert "cell_table" in result
    assert "null_calibration_check" in result
