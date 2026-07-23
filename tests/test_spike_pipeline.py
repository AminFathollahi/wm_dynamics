"""Tests for src/spike_pipeline.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from spike_pipeline import (
    build_psth,
    zscore_psth,
    fit_pca_psth,
    load_vs_load_ctg,
    item_identity_ctg,
    pr_by_load,
    low_rate_unit_mask,
)


def _synthetic_spike_lists(rng, n_units=10, rate_hz=5.0, duration_s=10.0):
    lists = []
    for _ in range(n_units):
        n_spikes = rng.poisson(rate_hz * duration_s)
        lists.append(np.sort(rng.uniform(0, duration_s, n_spikes)))
    return lists


class TestBuildPSTH:
    def test_output_shape(self, rng):
        spikes = _synthetic_spike_lists(rng, n_units=6)
        onsets = np.array([1.0, 3.0, 5.0])
        psth = build_psth(spikes, onsets, bin_ms=100, smooth_ms=0, window_s=2.0)
        assert psth.shape == (3, 6, 20)

    def test_rate_matches_expected_order_of_magnitude(self, rng):
        spikes = _synthetic_spike_lists(rng, n_units=4, rate_hz=10.0, duration_s=20.0)
        onsets = np.arange(0.5, 15.0, 1.0)
        psth = build_psth(spikes, onsets, bin_ms=100, smooth_ms=0, window_s=1.0)
        assert 3.0 < psth.mean() < 25.0

    def test_smoothing_reduces_variance(self, rng):
        spikes = _synthetic_spike_lists(rng, n_units=8, rate_hz=8.0, duration_s=20.0)
        onsets = np.arange(0.5, 15.0, 1.0)
        raw = build_psth(spikes, onsets, bin_ms=50, smooth_ms=0, window_s=2.0)
        smoothed = build_psth(spikes, onsets, bin_ms=50, smooth_ms=200, window_s=2.0)
        assert smoothed.var() < raw.var()


class TestFitPcaPsth:
    def test_output_shapes(self, rng):
        psth = rng.standard_normal((40, 12, 15)).astype(np.float32)
        Z, V, var_ratio = fit_pca_psth(psth, n_comp=5)
        assert Z.shape == (40, 15, 5)
        assert V.shape == (12, 5)
        assert 0.0 <= var_ratio <= 1.0 + 1e-8

    def test_low_rank_signal_captured(self, rng):
        N, U, T = 50, 12, 15
        direction = rng.standard_normal(U)
        amp = rng.standard_normal((N, 1, T))
        psth = (amp * direction[None, :, None]).astype(np.float32)
        psth += 0.01 * rng.standard_normal((N, U, T)).astype(np.float32)
        _, _, var_ratio = fit_pca_psth(psth, n_comp=3)
        assert var_ratio > 0.9


class TestLoadVsLoadCTG:
    def test_returns_none_when_underpowered(self, rng):
        psth_z = rng.standard_normal((10, 8, 12)).astype(np.float32)
        loads = np.array([1] * 3 + [3] * 3 + [2] * 4)
        res = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=4, n_splits=2, n_perm=10, rng=rng)
        assert res is None

    def test_detects_signal(self, rng):
        N, U, T = 60, 10, 16
        loads = np.array([1] * 30 + [3] * 30)
        psth_z = rng.standard_normal((N, U, T)).astype(np.float32)
        direction = rng.standard_normal(U)
        psth_z += 3.0 * (loads == 3)[:, None, None] * direction[None, :, None]
        res = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=4, n_splits=3, n_perm=50, rng=rng)
        assert res is not None
        assert res["p_value"] < 0.05


class TestItemIdentityCTG:
    def test_returns_none_when_too_few_trials(self, rng):
        psth_z = rng.standard_normal((10, 8, 12)).astype(np.float32)
        loads = np.ones(10, dtype=int)
        items = rng.integers(0, 5, 10)
        res = item_identity_ctg(psth_z, loads, items, target_load=1, min_trials=20, rng=rng)
        assert res is None

    def test_returns_none_when_single_trial_per_class(self, rng):
        N, U, T = 25, 8, 10
        psth_z = rng.standard_normal((N, U, T)).astype(np.float32)
        loads = np.ones(N, dtype=int)
        items = np.arange(N) % 25  # every item unique -> min class count 1
        res = item_identity_ctg(psth_z, loads, items, target_load=1, min_trials=20, rng=rng)
        assert res is None

    def test_detects_multiclass_signal(self, rng):
        N, U, T, n_classes = 100, 10, 12, 4
        loads = np.ones(N, dtype=int)
        items = rng.integers(0, n_classes, N)
        psth_z = rng.standard_normal((N, U, T)).astype(np.float32)
        directions = rng.standard_normal((n_classes, U))
        for c in range(n_classes):
            psth_z[items == c] += 3.0 * directions[c][None, :, None]
        res = item_identity_ctg(psth_z, loads, items, target_load=1, min_trials=20,
                                n_components=6, n_perm=50, rng=rng)
        assert res is not None
        assert res["p_value"] < 0.05


class TestPRByLoad:
    def test_only_qualifying_loads_included(self, rng):
        psth_z = rng.standard_normal((20, 6, 10)).astype(np.float32)
        loads = np.array([1] * 15 + [2] * 3 + [3] * 2)
        res = pr_by_load(psth_z, loads, load_values=(1, 2, 3), min_trials=5, rng=rng)
        assert 1 in res
        assert 2 not in res
        assert 3 not in res


class TestLowRateUnitMask:
    def test_keeps_active_unit_drops_silent_unit(self):
        onsets = np.arange(0.0, 100.0, 2.0)   # 50 trials
        window_s = 2.0
        active = np.arange(0.5, 100.0, 2.0)   # 1 spike/trial -> 0.5 Hz, well above floor
        silent = np.array([])                  # 0 Hz
        mask = low_rate_unit_mask([active, silent], onsets, window_s, min_rate_hz=0.1)
        np.testing.assert_array_equal(mask, [True, False])

    def test_threshold_boundary(self):
        # 10 trials x 1s window = 10s total; exactly 1 spike -> 0.1 Hz.
        onsets = np.arange(0.0, 10.0, 1.0)
        window_s = 1.0
        exactly_at_floor = np.array([0.5])   # 1 spike total -> 1/10 = 0.1 Hz
        below_floor = np.array([])
        mask = low_rate_unit_mask([exactly_at_floor, below_floor], onsets, window_s,
                                  min_rate_hz=0.1)
        np.testing.assert_array_equal(mask, [True, False])

    def test_empty_onsets_keeps_nothing(self):
        mask = low_rate_unit_mask([np.array([1.0, 2.0])], np.array([]), window_s=1.0)
        np.testing.assert_array_equal(mask, [False])
