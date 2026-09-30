"""Real subprocess checks against the ssm environment: small synthetic arrays,
no project data. Each invocation trains a tiny model, so these are slower
than pure-python unit tests but exercise the genuine bridge contract."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.project_config import executable

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "fit_recurrent_switching_linear_dynamics_worker.py"
SSM_PYTHON = executable("ssm_python") or ""

pytestmark = pytest.mark.skipif(not SSM_PYTHON or not Path(SSM_PYTHON).exists(), reason="ssm environment not available")


def _run(tmp_path, train_X, test_X, k=2, seed=0, causal=False, timeout=180):
    in_path, out_path = tmp_path / "in.npz", tmp_path / "out.npz"
    np.savez(in_path, train_X=train_X.astype(np.float32), test_X=test_X.astype(np.float32),
              k=np.int64(k), seed=np.int64(seed), causal=np.bool_(causal))
    proc = subprocess.run([SSM_PYTHON, str(WORKER), str(in_path), str(out_path)],
                            capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr
    return dict(np.load(out_path, allow_pickle=True))


def test_normal_mode_returns_right_shapes(tmp_path):
    rng = np.random.default_rng(0)
    train_X = rng.poisson(2.0, size=(8, 6, 5)).astype(np.float32)
    test_X = rng.poisson(2.0, size=(4, 6, 5)).astype(np.float32)
    out = _run(tmp_path, train_X, test_X, k=2)
    assert "reason" not in out
    assert out["latent_train"].shape == (8, 6, 2)
    assert out["latent_test"].shape == (4, 6, 2)
    assert int(out["k_used"]) == 2


def test_too_few_trials_reports_failed_to_train(tmp_path):
    train_X = np.zeros((2, 3, 2), dtype=np.float32)
    test_X = np.zeros((2, 3, 2), dtype=np.float32)
    out = _run(tmp_path, train_X, test_X)
    assert "reason" in out
    assert "n_train_trials" in str(out["reason"])


def test_causal_mode_matches_up_to_truncation_only(tmp_path):
    rng = np.random.default_rng(1)
    n_train, n_test, n_bins, n_units = 6, 4, 6, 5
    train_X = rng.poisson(2.0, size=(n_train, n_bins, n_units)).astype(np.float32)
    test_X = rng.poisson(2.0, size=(n_test, n_bins, n_units)).astype(np.float32)
    test_X[1] = test_X[0].copy()
    test_X[1, 3:] = rng.poisson(6.0, size=(n_bins - 3, n_units)).astype(np.float32)
    out = _run(tmp_path, train_X, test_X, k=2, causal=True, timeout=300)
    assert "reason" not in out
    latent_test = out["latent_test"]
    np.testing.assert_array_equal(latent_test[0, :3], latent_test[1, :3])
    assert not np.allclose(latent_test[0, 3], latent_test[1, 3])


def test_causal_mode_every_prefix_length_matches_for_identical_trials(tmp_path):
    """Two fully identical trials must get identical causal latents at every bin,
    for every prefix length 1..n_bins the worker computes by default (bins=None) --
    the latent at bin t depends on bins 1..t of that trial alone, for every t."""
    rng = np.random.default_rng(2)
    n_train, n_bins, n_units = 6, 5, 4
    train_X = rng.poisson(2.0, size=(n_train, n_bins, n_units)).astype(np.float32)
    one_trial = rng.poisson(2.0, size=(1, n_bins, n_units)).astype(np.float32)
    test_X = np.concatenate([one_trial, one_trial], axis=0)
    out = _run(tmp_path, train_X, test_X, k=2, causal=True, timeout=300)
    assert "reason" not in out
    latent_test = out["latent_test"]
    assert latent_test.shape == (2, n_bins, 2)
    np.testing.assert_array_equal(latent_test[0], latent_test[1])


def _load_worker_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rslds_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PrefixModel:
    """Stands in for an ssm model: the latent at a prefix is its length; listed
    prefix lengths diverge on the first `fail_attempts` seeds."""
    D = 1

    def __init__(self, fail_attempts):
        self.fail_attempts = fail_attempts
        self.calls = {}

    def approximate_posterior(self, trials, **kwargs):
        t = len(trials[0])
        self.calls[t] = self.calls.get(t, 0) + 1
        if self.calls[t] <= self.fail_attempts.get(t, 0):
            raise AssertionError
        q = type("Posterior", (), {"mean_continuous_states": [np.full((t, 1), float(t))]})()
        return None, q


def test_causal_divergent_prefixes_are_retried_or_carried():
    worker = _load_worker_module()
    model = _PrefixModel({2: 1, 3: worker.CAUSAL_ATTEMPTS})
    trials = [np.zeros((4, 2))]
    latents, counts = worker._causal_latents(model, trials, 4, 5, 0)
    assert latents[0, :, 0].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert counts == {"retried": 1, "carried": 1}


def test_normal_mode_writes_no_inference_counts(tmp_path):
    rng = np.random.default_rng(0)
    train_X = rng.poisson(2.0, size=(8, 6, 5)).astype(np.float32)
    test_X = rng.poisson(2.0, size=(4, 6, 5)).astype(np.float32)
    out = _run(tmp_path, train_X, test_X, k=2)
    assert "inference_retried_bins" not in out and "inference_carried_bins" not in out
