import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_within_trial_time_component_identity as analysis  # noqa: E402
import run_rank_free_component_identity as rank_free  # noqa: E402


def _planted_time_counts(rng, n_trials=30, n_units=10, n_bins=8, effect=3.0, base_rate=3.0):
    direction = rng.normal(size=n_units)
    direction /= np.linalg.norm(direction)
    bin_time = np.arange(n_bins, dtype=float)
    counts = np.empty((n_trials, n_units, n_bins))
    for trial in range(n_trials):
        trial_rate = rng.gamma(2.0, 1.0, size=n_units)
        for b in range(n_bins):
            rate = np.clip(trial_rate + effect * bin_time[b] * direction, 0.05, None)
            counts[trial, :, b] = rng.poisson(rate)
    return counts.astype(float)


def test_bin_samples_assigns_whole_trials_to_one_fold_and_drops_zero_norm_bins():
    counts = np.zeros((8, 5, 4))
    counts[0, :, 0] = 1.0  # every other (trial, bin) stays exactly zero-norm
    counts[1:, 0, :] = 1.0
    samples = analysis._bin_samples(counts)
    assert samples is not None
    assert samples["n_bin_samples_total"] == 8 * 4
    assert samples["n_bin_samples_used"] == 1 + 7 * 4
    assert samples["n_bin_samples_dropped_nonfinite_or_zero_norm"] == samples["n_bin_samples_total"] - samples["n_bin_samples_used"]
    from subspace_identity import block_folds
    trial_folds = block_folds(8, analysis.N_FOLDS)
    for trial, positions in samples["trial_positions"].items():
        sample_folds = samples["folds"][positions]
        assert np.all(sample_folds == trial_folds[trial])


def test_bin_samples_returns_none_below_trial_floor():
    assert analysis._bin_samples(np.ones((3, 4, 5))) is None
    assert analysis._bin_samples(np.ones((10, 4, 1))) is None


def test_time_permutation_alignment_recovers_a_planted_time_effect():
    rng = np.random.default_rng(11)
    counts = _planted_time_counts(rng, effect=4.0)
    cell = analysis._session_cell(counts, 300, "planted")
    assert cell["status"] == "computed"
    assert cell["alignment_above_null"] > 0.05
    assert cell["p_value"] <= 0.05


def test_time_permutation_alignment_keeps_axis_fixed_across_permutations(monkeypatch):
    from subspace_identity import block_folds, residual_axis

    rng = np.random.default_rng(4)
    directions = rng.normal(size=(40, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    trial_index = np.repeat(np.arange(10), 4)
    trial_folds = block_folds(10, 2)
    folds = trial_folds[trial_index]
    target = np.tile(np.arange(4, dtype=float), 10)
    trial_positions = {t: np.flatnonzero(trial_index == t) for t in range(10)}

    axis_calls = []
    original = residual_axis

    def spy(reference, evaluation):
        axis_calls.append(1)
        return original(reference, evaluation)

    monkeypatch.setattr(analysis, "residual_axis", spy)
    analysis._time_permutation_alignment(directions, target, folds, trial_positions, 25, np.random.default_rng(1))
    assert len(axis_calls) == len(np.unique(folds))  # fit once per fold, never refit under a permutation draw


def test_null_is_calibrated_on_time_independent_synthetic_activity():
    result = analysis.null_calibration_check(200, 150, "test_calibration")
    assert result["n_replicates_computed"] >= 190
    assert result["false_positive_rate_at_p_0.05"] is not None
    assert abs(result["false_positive_rate_at_p_0.05"] - 0.05) <= 0.04
    assert result["calibrated"]


def test_pool_requires_the_same_independent_unit_floor_as_the_delivered_run():
    assert analysis.MIN_INDEPENDENT_UNITS == rank_free.MIN_INDEPENDENT_UNITS

    def record(session, unit, status="not_computable"):
        return {"session": session, "independent_unit": unit, "status": status}

    records = [record(f"s{i}", f"unit{i}") for i in range(3)]
    pooled = analysis._pool(records, "floor-test")
    assert pooled["status"] == "not_computable"
    assert pooled["n_independent_units"] == 0
    assert pooled["n_sessions_with_session_geometry"] == 0


def test_pool_weights_independent_units_not_sessions():
    draws = np.full(20, 0.1)

    def record(session, unit, alignment):
        return {
            "session": session, "independent_unit": unit, "status": "computed",
            "candidates": {analysis.CANDIDATE: {"status": "computed", "alignment": alignment, "null_draws": draws}},
        }

    records = [
        record("a1", "unit_a", 0.9), record("a2", "unit_a", 0.7),
        record("b1", "unit_b", 0.2), record("c1", "unit_c", 0.4), record("d1", "unit_d", 0.6),
    ]
    pooled = analysis._pool(records, "pool-test")
    assert pooled["status"] == "computed"
    assert pooled["n_sessions"] == 5
    assert pooled["n_independent_units"] == 4
    assert pooled["mean_alignment"] == 0.5
    assert pooled["mean_alignment_above_null"] == 0.4
    assert pooled["minimum_detectable_difference_80pct_power"] >= 0.0


def test_rollup_reports_the_stub_honestly_for_the_delivered_pipeline():
    records = [
        {"session": "s1", "independent_unit": "u1", "status": "computed",
         "candidates": {"time_in_trial_within_delay": {"status": "not_computable", "reason": "deferred"}}},
        {"session": "s2", "independent_unit": "u2", "status": "computed",
         "candidates": {"time_in_trial_within_delay": {"status": "not_computable", "reason": "deferred"}}},
    ]
    pooled = rank_free._pool(records, "time_in_trial_within_delay", "seed")
    assert pooled["n_sessions"] == 0
    assert pooled["n_sessions_with_session_geometry"] == 2
