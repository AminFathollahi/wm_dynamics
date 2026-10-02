import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / _subdir))

from dynamics import pooled_nback_dynamics  # noqa: E402
from geometry import latent_trajectories, nback_geometry_arrays  # noqa: E402
from preprocessing import (  # noqa: E402
    SRATE,
    baseline_normalize,
    epoch_data,
    high_gamma_power,
    nback_epoch_arrays,
    preprocess,
    reject_bad_channels,
)


def _subject(n_trials=6, n_channels=10, spacing=2000, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = spacing * (n_trials + 1)
    data = rng.normal(size=(n_samples, n_channels))
    data[:, 3] *= 50.0
    stim = np.zeros(n_samples, dtype=np.uint8)
    task = np.zeros(n_samples, dtype=np.int8)
    target = np.zeros(n_samples, dtype=np.uint8)
    for k in range(n_trials):
        onset = spacing * (k + 1) - 1000
        stim[onset:onset + 300] = k % 3 + 1
        task[onset:onset + 300] = k % 3
        target[onset:onset + 300] = 1 + k % 2
    return {"data": data, "stim": stim, "task": task, "target": target}


def test_epoch_arrays_have_expected_keys_dtypes_and_drop_the_variance_outlier():
    out = nback_epoch_arrays(_subject())
    assert set(out) == {"epochs", "times", "task_id", "tgt_id", "stim_id", "good_channels"}
    n_epochs, n_times, n_channels = out["epochs"].shape
    assert (n_epochs, n_times) == (6, 1700) and n_channels == len(out["good_channels"])
    assert 3 not in out["good_channels"]
    assert out["epochs"].dtype == np.float32 and out["times"].dtype == np.float64
    assert out["task_id"].dtype == np.int8 and out["tgt_id"].dtype == np.uint8
    assert out["stim_id"].dtype == np.uint8 and out["good_channels"].dtype == np.int64
    baseline = out["epochs"][:, out["times"] <= 0.0, :]
    np.testing.assert_allclose(baseline.mean(axis=(0, 1)), 0.0, atol=1e-4)
    np.testing.assert_allclose(baseline.std(axis=(0, 1)), 1.0, atol=1e-3)


def test_epoch_arrays_equal_the_stepwise_pipeline_with_variance_only_channel_rejection():
    subject = _subject(seed=1)
    out = nback_epoch_arrays(subject)
    good = reject_bad_channels(subject["data"], threshold_mad=3.0, srate=None)
    power = high_gamma_power(preprocess(subject["data"][:, good], srate=SRATE), srate=SRATE)
    ep = epoch_data(power, subject["stim"], subject["task"], subject["target"])
    np.testing.assert_array_equal(out["epochs"], baseline_normalize(ep["epochs"], ep["times"]))
    np.testing.assert_array_equal(out["good_channels"], np.where(good)[0])


def _epochs(n_target=8, n_nontarget=8, n_zero_back=4, n_times=60, n_channels=6, seed=0):
    rng = np.random.default_rng(seed)
    task_id = np.array([2] * (n_target + n_nontarget) + [0] * n_zero_back, dtype=np.int8)
    tgt_id = np.array([2] * n_target + [1] * n_nontarget + [1] * n_zero_back, dtype=np.uint8)
    epochs = rng.normal(size=(len(task_id), n_times, n_channels)).astype(np.float32)
    return epochs, np.linspace(-0.2, 1.5, n_times), task_id, tgt_id


def test_geometry_arrays_match_the_primitives_and_keep_float32_trajectories():
    epochs, times, task_id, tgt_id = _epochs()
    out = nback_geometry_arrays(epochs, times, task_id, tgt_id)
    assert set(out) == {
        "pr_per_trial", "var_ratio", "Z", "task_id", "tgt_id", "times", "theta_tgt_vs_ntgt",
    }
    Z, _, var_ratio = latent_trajectories(epochs, 8)
    assert out["Z"].dtype == np.float32 and out["Z"].shape == (20, 60, 6)
    np.testing.assert_array_equal(out["Z"], Z.astype(np.float32))
    np.testing.assert_array_equal(out["var_ratio"], var_ratio)
    assert out["pr_per_trial"].shape == (20,) and out["theta_tgt_vs_ntgt"].shape == (60, 4)


def test_geometry_arrays_omit_the_angle_timecourse_when_a_two_back_group_is_small():
    epochs, times, task_id, tgt_id = _epochs(n_target=4)
    assert "theta_tgt_vs_ntgt" not in nback_geometry_arrays(epochs, times, task_id, tgt_id)


def _trajectories(n_target, n_nontarget, seed, n_times=50, dim=8):
    rng = np.random.default_rng(seed)
    task_id = np.array([2] * (n_target + n_nontarget) + [0] * 3, dtype=np.int8)
    tgt_id = np.array([2] * n_target + [1] * n_nontarget + [1] * 3, dtype=np.uint8)
    Z = np.cumsum(rng.normal(size=(len(task_id), n_times, dim)), axis=1).astype(np.float32)
    return {"Z": Z, "task_id": task_id, "tgt_id": tgt_id, "times": np.linspace(-0.2, 1.5, n_times)}


def test_pooled_dynamics_stack_trials_across_subjects():
    geometry = {"a": _trajectories(5, 6, 0), "b": _trajectories(4, 7, 1)}
    out = pooled_nback_dynamics(geometry, dt=0.001)
    assert out["Q_tgt_pool"].shape == (9, 50) and out["Q_ntgt_pool"].shape == (13, 50)
    assert out["Q_tgt_pool"].dtype == np.float32
    assert out["evals_tgt"].shape == (9, 6) and out["evals_ntgt"].shape == (13, 6)
    assert np.iscomplexobj(out["evals_tgt"])


def test_pooled_dynamics_skip_subjects_with_too_few_trials_and_omit_empty_outputs():
    out = pooled_nback_dynamics({"a": _trajectories(3, 6, 0), "b": _trajectories(5, 6, 1)}, dt=0.001)
    assert out["Q_tgt_pool"].shape[0] == 5 and out["Q_ntgt_pool"].shape[0] == 12
    only_small = pooled_nback_dynamics({"a": _trajectories(3, 3, 0)}, dt=0.001)
    assert "Q_tgt_pool" not in only_small and "Q_ntgt_pool" not in only_small
    assert pooled_nback_dynamics({"a": _trajectories(2, 2, 0)}, dt=0.001) == {}
