"""Tests for scripts/run_behavior_ctg.py (Round-7 STEP B/I).

The script reuses geometry.ctg_label_permutation_null unchanged (its own
docstring: "does not implement any new numerical routine, only supplies a
different label"), so these tests target the SPECIFIC scenario Step B needs
that the existing geometry tests (tests/test_geometry.py's
TestCtgLabelPermutationNull) do not cover: a LOCALIZED-IN-TIME (not
sustained) signal on an IMBALANCED label (matching real trial-outcome data,
where correct >> error) -- does the recovered diagonal peak land at the
known injected timepoint, and does label-shuffling collapse it to chance?
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from geometry import ctg_label_permutation_null, temporal_stability_tau


def _localized_outcome_signal(rng, N=90, C=10, T=24, peak_idx=15, effect=3.0,
                              error_frac=0.15):
    """(N, C, T) raw features where the outcome label is linearly decodable
    ONLY near t=peak_idx (a narrow Gaussian-in-time signal bump), on an
    imbalanced binary label (error_frac << 0.5, matching real correct>>error
    trial-outcome data)."""
    labels = (rng.random(N) < error_frac).astype(int)   # 1 = "error" (minority)
    X = rng.standard_normal((N, C, T)).astype(np.float32)
    direction = rng.standard_normal(C)
    direction /= np.linalg.norm(direction)
    t = np.arange(T)
    time_weight = np.exp(-0.5 * ((t - peak_idx) / 1.5) ** 2)   # narrow bump at peak_idx
    X += effect * labels[:, None, None] * direction[None, :, None] * time_weight[None, None, :]
    return X, labels


class TestBehaviorCtgRecoversLocalizedPeak:
    def test_diagonal_peak_lands_near_injected_timepoint(self, rng):
        X, labels = _localized_outcome_signal(rng, peak_idx=15)
        t_idx = np.arange(0, 24, 2)   # includes 14, 16 -- brackets peak_idx=15
        res = ctg_label_permutation_null(X, labels, t_idx, n_components=4, n_splits=3,
                                         n_perm=100, rng=rng)
        diag = np.diag(res["auc_mat"])
        peak_t = t_idx[int(np.nanargmax(diag))]
        assert abs(peak_t - 15) <= 4   # within one time_weight sigma-ish of the true bump
        assert np.nanmax(diag) > 0.6   # genuinely above-chance at the peak

    def test_offdiagonal_lower_than_diagonal_for_transient_signal(self, rng):
        # A signal present at only ONE timepoint should generalize POORLY to
        # other timepoints -- this is the CTG-vs-transient distinction Step B's
        # tau statistic is built to detect (temporal_stability_tau).
        X, labels = _localized_outcome_signal(rng, peak_idx=15, effect=4.0)
        t_idx = np.arange(0, 24, 2)
        res = ctg_label_permutation_null(X, labels, t_idx, n_components=4, n_splits=3,
                                         n_perm=100, rng=rng)
        tau_info = temporal_stability_tau(res["auc_mat"])
        assert tau_info["mean_diag_auc"] > tau_info["mean_offdiag_auc"]

    def test_label_shuffled_data_gives_chance_auc(self, rng):
        X, labels = _localized_outcome_signal(rng, peak_idx=15, effect=4.0)
        shuffled_labels = rng.permutation(labels)
        t_idx = np.arange(0, 24, 2)
        res = ctg_label_permutation_null(X, shuffled_labels, t_idx, n_components=4,
                                         n_splits=3, n_perm=100, rng=rng)
        diag = np.diag(res["auc_mat"])
        assert abs(np.nanmean(diag) - 0.5) < 0.15

    def test_imbalanced_null_is_not_significant(self, rng):
        # Pure noise on a realistically-imbalanced label (no injected signal
        # anywhere) should not spuriously reject -- guards against class
        # imbalance itself inflating the permutation p-value's false-positive rate.
        N, C, T = 90, 10, 24
        X = rng.standard_normal((N, C, T)).astype(np.float32)
        labels = (rng.random(N) < 0.15).astype(int)
        t_idx = np.arange(0, T, 4)
        res = ctg_label_permutation_null(X, labels, t_idx, n_components=4, n_splits=3,
                                         n_perm=200, rng=rng)
        assert res["p_value"] > 0.05
