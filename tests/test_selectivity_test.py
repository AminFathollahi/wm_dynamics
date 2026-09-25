from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selectivity_test import two_stage_selectivity_test  # noqa: E402


def test_detects_planted_tuning_single_presentation_per_trial():
    rng = np.random.default_rng(1)
    n_trials = 80
    labels = rng.integers(1, 6, size=n_trials)
    rates = rng.normal(loc=2.0, scale=0.5, size=n_trials)
    rates[labels == 3] += 3.0
    trial_id = np.arange(n_trials)
    out = two_stage_selectivity_test(rates, labels, trial_id, rng)
    assert out["status"] == "computed"
    assert out["preferred_level"] == 3
    assert out["meets_selectivity_criterion"] is True


def test_widened_presentation_set_keeps_trial_count_not_row_count():
    rng = np.random.default_rng(2)
    n_trials = 60
    labels = rng.integers(1, 6, size=n_trials)
    rates = rng.normal(loc=2.0, scale=0.5, size=n_trials)
    rates[labels == 2] += 3.0
    rates_widened = np.concatenate([rates, rates])
    labels_widened = np.concatenate([labels, labels])
    trial_id_widened = np.concatenate([np.arange(n_trials), np.arange(n_trials)])
    out = two_stage_selectivity_test(rates_widened, labels_widened, trial_id_widened, rng)
    assert out["status"] == "computed"
    assert out["n_trials"] == n_trials


def test_underpowered_below_trial_floor():
    rng = np.random.default_rng(3)
    rates = rng.standard_normal(5)
    labels = np.array([1, 1, 2, 2, 3])
    trial_id = np.arange(5)
    out = two_stage_selectivity_test(rates, labels, trial_id, rng)
    assert out["status"] == "underpowered"


def test_preferred_vs_rest_null_recomputes_argmax_per_draw():
    # the preferred level is argmax(means_obs), a selection over 5 candidates -- the null must
    # re-select its own argmax per permutation draw, or permutation_p is anti-conservative: comparing
    # a maximum to a fixed-group null understates how often a maximum that large arises by chance.
    rng = np.random.default_rng(4)
    n_below = 0
    n_draws = 200
    for _ in range(n_draws):
        n_trials = 60
        labels = rng.integers(1, 6, size=n_trials)
        rates = rng.normal(0, 1, size=n_trials)
        trial_id = np.arange(n_trials)
        out = two_stage_selectivity_test(rates, labels, trial_id, rng, n_perm=1000)
        if out["permutation_p"] < 0.05:
            n_below += 1
    assert n_below / n_draws < 0.12


if __name__ == "__main__":
    test_detects_planted_tuning_single_presentation_per_trial()
    test_widened_presentation_set_keeps_trial_count_not_row_count()
    test_underpowered_below_trial_floor()
    test_preferred_vs_rest_null_recomputes_argmax_per_draw()
    print("ok")
