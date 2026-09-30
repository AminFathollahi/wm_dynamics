"""Real subprocess checks against the ndt environment: small synthetic arrays,
no project data."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.project_config import executable

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "fit_neural_data_transformer_worker.py"
NDT_PYTHON = executable("ndt_python") or ""

pytestmark = pytest.mark.skipif(not NDT_PYTHON or not Path(NDT_PYTHON).exists(), reason="ndt environment not available")


def _run(tmp_path, train_X, test_X, k=2, seed=0, causal=False, timeout=300):
    in_path, out_path = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(in_path, train_X=train_X.astype(np.float32), test_X=test_X.astype(np.float32),
              k=np.int64(k), seed=np.int64(seed), causal=np.bool_(causal))
    proc = subprocess.run([NDT_PYTHON, str(WORKER), str(in_path), str(out_path)],
                            capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(out_path, allow_pickle=True))


def test_too_few_trials_reports_failed_to_train(tmp_path):
    train_X = np.zeros((2, 3, 2), dtype=np.float32)
    test_X = np.zeros((2, 3, 2), dtype=np.float32)
    out = _run(tmp_path, train_X, test_X, timeout=60)
    assert "reason" in out


def test_causal_mode_returns_right_shapes_and_matches_up_to_truncation_only(tmp_path):
    rng = np.random.default_rng(2)
    n_train, n_test, n_bins, n_units = 10, 4, 8, 6
    train_X = rng.poisson(2.0, size=(n_train, n_bins, n_units)).astype(np.float32)
    test_X = rng.poisson(2.0, size=(n_test, n_bins, n_units)).astype(np.float32)
    test_X[1] = test_X[0].copy()
    test_X[1, 4:] = rng.poisson(6.0, size=(n_bins - 4, n_units)).astype(np.float32)
    out = _run(tmp_path, train_X, test_X, k=2, causal=True)
    assert "reason" not in out
    assert out["latent_train"].shape == (n_train, n_bins, 2)
    latent_test = out["latent_test"]
    assert latent_test.shape == (n_test, n_bins, 2)
    np.testing.assert_array_equal(latent_test[0, :4], latent_test[1, :4])
    assert not np.allclose(latent_test[0, 4], latent_test[1, 4])
