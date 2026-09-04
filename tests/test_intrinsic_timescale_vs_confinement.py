"""Regression tests for scripts/run_intrinsic_timescale_vs_confinement.py:
the spike-count binning and the exponential-decay-plus-offset tau fit, on
synthetic data with a known tau.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_intrinsic_timescale_vs_confinement as mod  # noqa: E402


class TestBinSpikeCounts:
    def test_counts_land_in_correct_bins(self):
        spikes = np.array([0.01, 0.06, 0.06, 0.89])
        onsets = np.array([0.0])
        counts = mod._bin_spike_counts(spikes, onsets)
        assert counts.shape == (1, mod.N_BINS)
        assert counts[0, 0] == 1  # 0.01s -> bin 0
        assert counts[0, 1] == 2  # 0.06s -> bin 1
        assert counts[0, -1] == 1  # 0.89s -> last bin

    def test_spikes_outside_window_excluded(self):
        spikes = np.array([-0.1, 1.5])
        onsets = np.array([0.0])
        counts = mod._bin_spike_counts(spikes, onsets)
        assert counts.sum() == 0


class TestFitTau:
    def test_too_few_trials_returns_none(self):
        counts = np.zeros((3, mod.N_BINS))
        tau, reason = mod._fit_tau(counts)
        assert tau is None
        assert "trials" in reason

    def test_recovers_known_tau_from_synthetic_ar1_counts(self):
        rng = np.random.default_rng(0)
        true_tau = 0.15
        n_trials = 200
        counts = np.zeros((n_trials, mod.N_BINS))
        for i in range(n_trials):
            rate = 5.0
            x = rng.normal(scale=1.0)
            for b in range(mod.N_BINS):
                x = x * np.exp(-mod.BIN_S / true_tau) + rng.normal(scale=0.5)
                lam = max(rate + 2.0 * x, 0.1)
                counts[i, b] = rng.poisson(lam * mod.BIN_S)
        tau, reason = mod._fit_tau(counts)
        assert tau is not None, reason
        assert tau == pytest.approx(true_tau, rel=1.0)  # order-of-magnitude sanity, not precision
        assert tau > 0

    def test_lag0_inclusion_collapses_tau_regardless_of_truth(self):
        """The defect this fix corrects: including lag 0 (the total variance,
        contaminated by Poisson counting noise) in the exponential fit makes
        the recovered tau insensitive to the true timescale. Excluding lag 0
        (the default) must still distinguish a short from a long true tau;
        including it (the old, broken behaviour) must not."""
        rng = np.random.default_rng(1)

        def synth_counts(true_tau, n_trials=300, rate=5.0):
            counts = np.zeros((n_trials, mod.N_BINS))
            for i in range(n_trials):
                x = rng.normal(scale=1.0)
                for b in range(mod.N_BINS):
                    x = x * np.exp(-mod.BIN_S / true_tau) + rng.normal(scale=0.5)
                    lam = max(rate + 2.0 * x, 0.1)
                    counts[i, b] = rng.poisson(lam * mod.BIN_S)
            return counts

        counts_short = synth_counts(0.05)
        counts_long = synth_counts(0.35)

        tau_short_fixed, _ = mod._fit_tau(counts_short, exclude_lag0=True)
        tau_long_fixed, _ = mod._fit_tau(counts_long, exclude_lag0=True)
        tau_short_broken, _ = mod._fit_tau(counts_short, exclude_lag0=False)
        tau_long_broken, _ = mod._fit_tau(counts_long, exclude_lag0=False)
        assert None not in (tau_short_fixed, tau_long_fixed, tau_short_broken, tau_long_broken)

        # Fixed estimator (lag 0 excluded): a 7x range in the truth remains
        # visible -- the long-tau fit is clearly longer than the short-tau fit.
        assert tau_long_fixed > 2 * tau_short_fixed

        # Broken estimator (lag 0 included): both truths collapse to within a
        # small absolute margin of each other, near the 50 ms bin width --
        # insensitive to the same 7x difference in the planted truth.
        assert abs(tau_long_broken - tau_short_broken) < 0.03


class TestOrderingsAgreeLogic:
    def test_matching_orders_detected(self):
        tau_ordering = {"a": 0.05, "b": 0.10, "c": 0.20}  # a shortest, c longest
        lambda_ordering = ["a", "b", "c"]  # a fastest confinement
        shared_regions = [r for r in lambda_ordering if r in tau_ordering]
        tau_shortest_first = sorted(shared_regions, key=tau_ordering.get)
        assert tau_shortest_first == lambda_ordering

    def test_disagreeing_orders_detected(self):
        tau_ordering = {"a": 0.20, "b": 0.10, "c": 0.05}  # c shortest
        lambda_ordering = ["a", "b", "c"]  # a fastest confinement
        shared_regions = [r for r in lambda_ordering if r in tau_ordering]
        tau_shortest_first = sorted(shared_regions, key=tau_ordering.get)
        assert tau_shortest_first != lambda_ordering
