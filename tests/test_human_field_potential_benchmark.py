from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import preprocessing as pp  # noqa: E402
import run_human_field_potential_benchmark as mod  # noqa: E402
from project_config import executable  # noqa: E402


# ---------------------------------------------------------------------------
# Stacked band-power shape and Nyquist-margin band dropping.
# ---------------------------------------------------------------------------

def test_multiband_maintenance_tensor_shape_and_bands_kept():
    rng = np.random.default_rng(0)
    n_trials, n_channels, n_samples, srate = 6, 4, 500, 1000.0
    epochs = rng.normal(size=(n_trials, n_channels, n_samples))
    times = np.arange(n_samples) / srate
    out = pp.multiband_maintenance_tensor(
        epochs, times, ["A1", "A2", "B1", "B2"], srate, (0.0, 0.4), 4, referencing="bipolar",
    )
    n_channels_out = 2  # two shanks of two contacts each -> one bipolar pair per shank
    assert out["kept_bands"] == list(pp.FIELD_BANDS)
    assert out["dropped_bands"] == []
    assert out["tensor"].shape == (n_trials, 4, n_channels_out * len(pp.FIELD_BANDS))
    assert len(out["channel_labels"]) == n_channels_out


def test_multiband_maintenance_tensor_drops_bands_at_low_nyquist():
    rng = np.random.default_rng(1)
    srate = 100.0  # Nyquist 50 Hz; 0.8 x 50 = 40 Hz margin
    epochs = rng.normal(size=(4, 2, 200))
    times = np.arange(200) / srate
    out = pp.multiband_maintenance_tensor(
        epochs, times, ["A1", "A2"], srate, (0.0, 1.0), 4, referencing="as_released",
    )
    assert out["kept_bands"] == ["theta", "alpha", "beta"]
    assert out["dropped_bands"] == ["gamma", "hgp"]
    assert out["tensor"].shape[-1] == 2 * 3


def test_multiband_maintenance_tensor_raises_when_every_band_drops():
    rng = np.random.default_rng(2)
    srate = 8.0  # every band's high edge exceeds 0.8 x Nyquist (3.2 Hz)
    epochs = rng.normal(size=(4, 2, 40))
    times = np.arange(40) / srate
    with pytest.raises(ValueError):
        pp.multiband_maintenance_tensor(epochs, times, ["A1", "A2"], srate, (0.0, 1.0), 2, referencing="as_released")


# ---------------------------------------------------------------------------
# Bipolar region assignment.
# ---------------------------------------------------------------------------

def test_bipolar_region_assignment_requires_both_contacts_to_match():
    labels = ["AHL1", "AHL2", "AHL3", "AHL4", "AL1", "AL2"]
    locations = [
        "Hipp, Left Hippocampus rHipp", "Hipp, Left Hippocampus cHipp",
        "unspecific", "unspecific",
        "Amyg, Left Amygdala mAmyg", "Amyg, Left Amygdala lAmyg",
    ]
    regions = mod.bipolar_region_assignment(labels, locations)
    # pairs (adjacent-contact, sliding): AHL1-AHL2, AHL2-AHL3, AHL3-AHL4, AL1-AL2
    assert regions == ["hippocampus", "other", "other", "amygdala"]


def test_bipolar_region_assignment_mixed_pair_is_other():
    labels = ["X1", "X2"]
    locations = ["Hipp, Left Hippocampus", "Amyg, Left Amygdala"]
    regions = mod.bipolar_region_assignment(labels, locations)
    assert regions == ["other"]


def test_bipolar_region_assignment_orphan_uses_its_own_contact():
    labels = ["X1"]
    locations = ["Amyg, Right Amygdala"]
    regions = mod.bipolar_region_assignment(labels, locations)
    assert regions == ["amygdala"]


# ---------------------------------------------------------------------------
# Held-out-channel groups keep all bands of a channel together.
# ---------------------------------------------------------------------------

def test_channel_holdout_keeps_bands_of_a_channel_together():
    n_channels, n_bands = 9, 5
    result = mod.channel_holdout(n_channels, n_bands, seed=7)
    held_out_features = set(result["held_out_features"].tolist())
    held_in_features = set(result["held_in_features"].tolist())
    assert held_out_features.isdisjoint(held_in_features)
    assert held_out_features | held_in_features == set(range(n_channels * n_bands))
    for channel in result["held_out_channels"]:
        band_features = set(range(int(channel) * n_bands, (int(channel) + 1) * n_bands))
        assert band_features <= held_out_features
    for channel in result["held_in_channels"]:
        band_features = set(range(int(channel) * n_bands, (int(channel) + 1) * n_bands))
        assert band_features <= held_in_features
    # a quarter of the channels, rounded, at least one
    assert len(result["held_out_channels"]) == max(1, round(n_channels / 4))


# ---------------------------------------------------------------------------
# Gaussian bits per observation: 0 for the null model, positive for a planted latent.
# ---------------------------------------------------------------------------

def test_bits_per_observation_zero_for_null_model():
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    variance = float(np.var(y))
    ll_null = mod.gaussian_log_likelihood(y, np.full_like(y, y.mean()), variance)
    assert mod.bits_per_observation(ll_null, ll_null, len(y)) == pytest.approx(0.0, abs=1e-12)


def test_bits_per_observation_positive_for_a_planted_latent():
    rng = np.random.default_rng(4)
    n = 2000
    latent = rng.normal(size=n)
    y = 3.0 * latent + rng.normal(scale=0.2, size=n)  # a genuinely informative predictor
    mean_pred = latent * (np.cov(latent, y)[0, 1] / np.var(latent))
    resid = y - mean_pred
    model_variance = float(np.var(resid))
    ll_model = mod.gaussian_log_likelihood(y, mean_pred, model_variance)
    null_variance = float(np.var(y - y.mean()))
    ll_null = mod.gaussian_log_likelihood(y, np.full_like(y, y.mean()), null_variance)
    assert mod.bits_per_observation(ll_model, ll_null, n) > 0.0


# ---------------------------------------------------------------------------
# One fold seed across candidates, dPCA included.
# ---------------------------------------------------------------------------

def test_content_block_computes_folds_once_shared_by_every_candidate(monkeypatch):
    calls = []
    real_seeded_folds = mod._seeded_folds

    def spy(labels, n_splits, seed):
        calls.append(seed)
        return real_seeded_folds(labels, n_splits, seed)

    monkeypatch.setattr(mod, "_seeded_folds", spy)

    rng = np.random.default_rng(5)
    n_trials, n_bins, n_features = 20, 4, 6
    activity = rng.normal(size=(n_trials, n_bins, n_features))
    labels = np.array([0, 1] * (n_trials // 2))

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        mod.evaluate_content_block(
            "dandi_000574", "pooled", "sub-test_ses-01", "set_size", activity, labels,
            (mod.DEMIXED_CANDIDATE, "native_full_rank"), ("linear",), n_splits=2, n_perm=1,
            time_step=1, seed=0, checkpoint_dir=checkpoint_dir, progress=lambda partial: None,
        )
    assert len(calls) == 1
    assert calls[0] == mod.stable_seed(
        "human_field_potential_benchmark|seed0|dandi_000574|pooled|sub-test_ses-01|set_size|folds"
    )


# ---------------------------------------------------------------------------
# Cross-session calibration never uses B trials beyond the first k to fit anything.
# ---------------------------------------------------------------------------

def _synthetic_entry(rng, n_trials, n_channels, n_bins, n_bands, seed_offset=0.0):
    activity = rng.normal(size=(n_trials, n_bins, n_channels * n_bands)) + seed_offset
    labels = np.array([0, 1] * (n_trials // 2) + [0] * (n_trials % 2))
    return {"activity": activity, "labels": labels, "n_bands": n_bands}


def test_calibrated_k_never_fits_on_b_trials_beyond_k(monkeypatch):
    import info_decoding
    decoder_calls, readout_calls, statistics_sizes = [], [], []
    real_split_auc = info_decoding.split_auc
    real_readout_scores = mod.readout_scores
    real_train_statistics = mod.train_statistics

    def spy_split_auc(train_activity, train_labels, test_activity, test_labels, **kwargs):
        decoder_calls.append((np.asarray(train_labels).copy(), np.asarray(test_labels).copy()))
        return real_split_auc(train_activity, train_labels, test_activity, test_labels, **kwargs)

    def spy_readout_scores(latent_train, target_train, latent_test, target_test):
        readout_calls.append((len(latent_train), len(target_train), len(latent_test), len(target_test)))
        return real_readout_scores(latent_train, target_train, latent_test, target_test)

    def spy_train_statistics(train_x):
        statistics_sizes.append(train_x.shape[0])
        return real_train_statistics(train_x)

    monkeypatch.setattr(info_decoding, "split_auc", spy_split_auc)
    monkeypatch.setattr(mod, "readout_scores", spy_readout_scores)
    monkeypatch.setattr(mod, "train_statistics", spy_train_statistics)

    rng = np.random.default_rng(6)
    n_channels, n_bands, n_bins = 8, 2, 3
    k = 10
    n_trials_b = k + mod.CALIBRATION_MIN_EXTRA_TRIALS + 5  # calibrated_20 is skipped; only calibrated_10 runs
    entry_a = _synthetic_entry(rng, 24, n_channels, n_bins, n_bands)
    entry_a["session"] = "sub-cal_ses-01"
    entry_b = _synthetic_entry(rng, n_trials_b, n_channels, n_bins, n_bands)
    entry_b["session"] = "sub-cal_ses-02"

    with tempfile.TemporaryDirectory() as tmp:
        records = mod.calibration_pair(
            "sub-cal", "pooled", entry_a, entry_b, ("principal_components",), "linear", 1, 0, Path(tmp),
        )

    calibrated_calls = [c for c in decoder_calls if len(c[0]) == k]
    assert len(calibrated_calls) == 1
    np.testing.assert_array_equal(calibrated_calls[0][0], entry_b["labels"][:k])
    np.testing.assert_array_equal(calibrated_calls[0][1], entry_b["labels"][k:])
    assert (k, k, n_trials_b - k, n_trials_b - k) in readout_calls
    assert n_trials_b not in statistics_sizes
    assert k in statistics_sizes
    skipped = [r for r in records if r["status"] == "skipped"]
    assert [r["mode"] for r in skipped] == ["calibrated_20"]
    computed_modes = {(r["mode"], r["metric"]) for r in records if r["status"] == "computed"}
    for mode in ("within_session", "zero_shot", "calibrated_10"):
        assert (mode, "content_decoding") in computed_modes
        assert (mode, "reconstruction") in computed_modes


# ---------------------------------------------------------------------------
# Records carry the inference mode and the label; failures carry full exception detail.
# ---------------------------------------------------------------------------

def test_records_carry_mode_and_label():
    rng = np.random.default_rng(8)
    activity = rng.normal(size=(20, 4, 12))
    labels = np.array([0, 1] * 10)
    with tempfile.TemporaryDirectory() as tmp:
        content = mod.evaluate_content_block(
            "dandi_000574", "pooled", "s", "set_size", activity, labels, ("principal_components",), ("linear",),
            n_splits=2, n_perm=1, time_step=1, seed=0, checkpoint_dir=Path(tmp), progress=lambda r: None,
        )
        reconstruction = mod.evaluate_reconstruction_entry(
            "dandi_000574", "pooled", "s", "set_size", activity, labels, 3, ("principal_components",),
            2, 0, Path(tmp), lambda r: None,
        )
    assert content[0]["mode"] == "whole_trial" and content[0]["label"] == "set_size"
    assert reconstruction[0]["mode"] == "whole_trial" and reconstruction[0]["status"] == "computed"
    assert np.isfinite(reconstruction[0]["bits_per_observation"])


def test_failure_detail_has_type_message_and_traceback_tail():
    try:
        raise AssertionError("shape mismatch")
    except AssertionError as exc:
        detail = mod.failure_detail(exc)
    assert detail.startswith("AssertionError: shape mismatch")
    assert "raise AssertionError" in detail


def test_co_primary_needs_one_interval_above_zero_and_the_other_not_below():
    def cell(lo, hi, **extra):
        return {"corpus": "dandi_000574", "region": "pooled", "level": "set_size", "decoder": "linear",
                "candidate": "c", "reference": "principal_components", "status": "estimable",
                "ci_95_patient_cluster_bootstrap": [lo, hi], **extra}

    content = [cell(0.01, 0.05, metric="same_time")]
    reconstruction = [cell(-0.02, 0.03, metric="bits_per_observation")]
    assert mod.co_primary_summary(content, reconstruction)[0]["co_primary"] is True
    reconstruction = [cell(-0.05, -0.01, metric="bits_per_observation")]
    assert mod.co_primary_summary(content, reconstruction)[0]["co_primary"] is False


# ---------------------------------------------------------------------------
# Poisson path of each edited worker is byte-identical on a fixed input.
# ---------------------------------------------------------------------------

def _run_worker(python, script, payload, timeout=180):
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.npz"
        out_path = Path(tmp) / "output.npz"
        np.savez(in_path, **payload)
        proc = subprocess.run([python, str(script), str(in_path), str(out_path)],
                               capture_output=True, text=True, timeout=timeout)
        assert proc.returncode == 0, proc.stderr
        result = dict(np.load(out_path, allow_pickle=True))
        return result


@pytest.mark.skipif(not executable("ssm_python"), reason="ssm_python executable not configured")
def test_ssm_worker_poisson_path_byte_identical_with_and_without_emission_key():
    script = ROOT / "scripts" / "fit_recurrent_switching_linear_dynamics_worker.py"
    rng = np.random.default_rng(11)
    train_x = rng.poisson(2.0, size=(5, 4, 3)).astype(np.float32)
    test_x = rng.poisson(2.0, size=(3, 4, 3)).astype(np.float32)
    base_payload = dict(train_X=train_x, test_X=test_x, k=np.int64(2), seed=np.int64(0))
    result_default = _run_worker(executable("ssm_python"), script, base_payload)
    result_explicit = _run_worker(executable("ssm_python"), script, {**base_payload, "emission": np.array("poisson")})
    assert result_default.keys() == result_explicit.keys()
    for key in result_default:
        np.testing.assert_array_equal(result_default[key], result_explicit[key])


@pytest.mark.skipif(not executable("lfads_python"), reason="lfads_python executable not configured")
def test_lfads_worker_poisson_path_byte_identical_with_and_without_emission_key():
    script = ROOT / "scripts" / "fit_sequential_autoencoder_worker.py"
    rng = np.random.default_rng(12)
    train_x = rng.poisson(2.0, size=(5, 4, 3)).astype(np.float32)
    test_x = rng.poisson(2.0, size=(3, 4, 3)).astype(np.float32)
    base_payload = dict(train_X=train_x, test_X=test_x, k=np.int64(2), seed=np.int64(0))
    result_default = _run_worker(executable("lfads_python"), script, base_payload, timeout=600)
    result_explicit = _run_worker(executable("lfads_python"), script, {**base_payload, "emission": np.array("poisson")}, timeout=600)
    assert result_default.keys() == result_explicit.keys()
    for key in result_default:
        np.testing.assert_array_equal(result_default[key], result_explicit[key])


@pytest.mark.skipif(not executable("lfads_python"), reason="lfads_python executable not configured")
def test_lfads_worker_poisson_path_matches_the_committed_worker(tmp_path):
    committed = subprocess.run(
        ["git", "show", "HEAD:scripts/fit_sequential_autoencoder_worker.py"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    committed_path = tmp_path / "committed_worker.py"
    committed_path.write_text(committed)
    rng = np.random.default_rng(13)
    payload = dict(train_X=rng.poisson(2.0, size=(5, 4, 3)).astype(np.float32),
                   test_X=rng.poisson(2.0, size=(3, 4, 3)).astype(np.float32), k=np.int64(2), seed=np.int64(0))
    current = _run_worker(executable("lfads_python"), ROOT / "scripts" / "fit_sequential_autoencoder_worker.py",
                          payload, timeout=600)
    original = _run_worker(executable("lfads_python"), committed_path, payload, timeout=600)
    assert current.keys() == original.keys()
    for key in current:
        np.testing.assert_array_equal(current[key], original[key])
