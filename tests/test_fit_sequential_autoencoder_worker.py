import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_sequential_autoencoder_worker as worker  # noqa: E402
from src.project_config import executable  # noqa: E402

LFADS_PYTHON = executable("lfads_python") or ""


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


def test_causal_fill_leaves_prefix_untouched_and_replaces_the_rest():
    X = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    fill = np.full((4, 3), -1.0)
    filled = worker._causal_fill(X, 2, fill)
    np.testing.assert_array_equal(filled[:, :2], X[:, :2])
    for trial in range(filled.shape[0]):
        np.testing.assert_array_equal(filled[trial, 2:], fill[2:])
    # original array is not mutated
    assert not np.array_equal(X, filled)


@pytest.mark.skipif(not LFADS_PYTHON or not Path(LFADS_PYTHON).exists(), reason="lfads_torch_py310 environment not available")
def test_causal_mode_end_to_end_matches_up_to_truncation_only(tmp_path):
    rng = np.random.default_rng(3)
    n_train, n_test, n_bins, n_units = 6, 4, 5, 4
    train_X = rng.poisson(2.0, size=(n_train, n_bins, n_units)).astype(np.float32)
    test_X = rng.poisson(2.0, size=(n_test, n_bins, n_units)).astype(np.float32)
    test_X[1] = test_X[0].copy()
    test_X[1, 2:] = rng.poisson(5.0, size=(n_bins - 2, n_units)).astype(np.float32)
    in_path, out_path = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(in_path, train_X=train_X, test_X=test_X, k=np.int64(2), seed=np.int64(0), causal=np.bool_(True))
    proc = subprocess.run(
        [LFADS_PYTHON, str(ROOT / "scripts" / "fit_sequential_autoencoder_worker.py"), str(in_path), str(out_path)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    out = dict(np.load(out_path, allow_pickle=True))
    assert "reason" not in out
    latent_test = out["latent_test"]
    np.testing.assert_array_equal(latent_test[0, :2], latent_test[1, :2])
    assert not np.allclose(latent_test[0, 2], latent_test[1, 2])
