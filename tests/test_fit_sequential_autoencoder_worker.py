import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fit_sequential_autoencoder_worker as worker  # noqa: E402


def test_validation_split_is_seeded_and_noncontiguous():
    fit_a, val_a = worker._split_indices(20, 17)
    fit_b, val_b = worker._split_indices(20, 17)
    _, val_c = worker._split_indices(20, 18)

    np.testing.assert_array_equal(fit_a, fit_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert not np.array_equal(val_a, val_c)
    assert not np.array_equal(val_a, np.arange(val_a.size))
    assert set(fit_a).isdisjoint(val_a)
    np.testing.assert_array_equal(np.sort(np.concatenate((fit_a, val_a))), np.arange(20))


def test_kl_weight_is_positive_and_reaches_configured_scale():
    assert 0 < worker._kl_weight(0) < worker.KL_IC_SCALE
    assert worker._kl_weight(worker.KL_INCREASE_EPOCHS) == worker.KL_IC_SCALE
    assert worker._kl_weight(worker.KL_INCREASE_EPOCHS + 1) == worker.KL_IC_SCALE
