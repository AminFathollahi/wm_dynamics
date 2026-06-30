"""Tests for src/preprocessing.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import (
    bandpass_filter,
    notch_filter,
    common_average_reference,
    reject_bad_channels,
    preprocess,
    high_gamma_power,
    find_stimulus_onsets,
    epoch_data,
    baseline_normalize,
    compute_hgp,
    boran_baseline_normalize,
    load_tes1_stimulation,
    build_tes1_input_matrix,
)

TES1_ZIP = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory"
    "/data/Tes1/HuangLiu2016dataset.zip"
)
BORAN_NWB = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory"
    "/data/000574/sub-01/sub-01_ses-01.nwb"
)


SRATE = 1200


class TestBandpassFilter:
    def test_output_shape_1d(self, rng):
        x = rng.standard_normal(1200)
        y = bandpass_filter(x, 70, 150, SRATE)
        assert y.shape == x.shape

    def test_output_shape_2d(self, rng):
        x = rng.standard_normal((1200, 8))
        y = bandpass_filter(x, 70, 150, SRATE)
        assert y.shape == x.shape

    def test_attenuates_out_of_band(self, rng):
        t = np.arange(SRATE) / SRATE
        # DC signal (0 Hz) — should be attenuated
        dc = np.ones(SRATE) * 5.0
        out = bandpass_filter(dc, 70, 150, SRATE)
        assert np.max(np.abs(out)) < 0.1

    def test_passes_inband(self):
        t = np.arange(SRATE) / SRATE
        # 100 Hz sinusoid — in the 70-150 Hz passband
        sig_in = np.sin(2 * np.pi * 100 * t)
        out = bandpass_filter(sig_in, 70, 150, SRATE)
        # After transient, amplitude should be close to 1
        assert np.max(np.abs(out[SRATE // 4:])) > 0.8

    def test_no_mutation_of_input(self, rng):
        x = rng.standard_normal(1200)
        x_copy = x.copy()
        bandpass_filter(x, 70, 150, SRATE)
        np.testing.assert_array_equal(x, x_copy)


class TestNotchFilter:
    def test_attenuates_60hz(self):
        # Use 5 s to allow filter edge-effects to decay before measuring
        t = np.arange(SRATE * 5) / SRATE
        noise = np.sin(2 * np.pi * 60 * t)
        out = notch_filter(noise, 60.0, SRATE)
        # Measure only in steady-state centre (drop first and last second)
        centre = out[SRATE:-SRATE]
        assert np.max(np.abs(centre)) < 0.05

    def test_preserves_broadband(self, rng):
        x = rng.standard_normal(SRATE * 5)
        out = notch_filter(x, 60.0, SRATE)
        # Power should be mostly preserved
        assert np.var(out) > 0.5 * np.var(x)


class TestCAR:
    def test_zero_mean_across_channels(self, rng):
        x = rng.standard_normal((100, 8))
        out = common_average_reference(x)
        np.testing.assert_allclose(out.mean(axis=1), np.zeros(100), atol=1e-10)

    def test_shape_preserved(self, rng):
        x = rng.standard_normal((500, 16))
        assert common_average_reference(x).shape == x.shape


class TestRejectBadChannels:
    def test_rejects_flat_channel(self, rng):
        data = rng.standard_normal((1000, 8)) * 50.0
        data[:, 3] = 0.0  # flat channel
        good = reject_bad_channels(data, threshold_mad=3.0)
        assert not good[3], "Flat channel should be rejected"
        assert good.sum() >= 7

    def test_rejects_high_variance(self, rng):
        data = rng.standard_normal((1000, 8)) * 50.0
        data[:, 5] *= 100  # very high variance
        good = reject_bad_channels(data, threshold_mad=3.0)
        assert not good[5]

    def test_returns_bool_array(self, rng):
        data = rng.standard_normal((500, 8))
        good = reject_bad_channels(data)
        assert good.dtype == bool
        assert len(good) == 8


class TestHighGammaPower:
    def test_output_shape(self, synthetic_ieeg):
        d = synthetic_ieeg
        hgp = high_gamma_power(d["data"], srate=d["srate"])
        assert hgp.shape == d["data"].shape

    def test_nonnegative(self, synthetic_ieeg):
        d = synthetic_ieeg
        hgp = high_gamma_power(d["data"], srate=d["srate"])
        assert np.all(hgp >= 0)

    def test_scales_with_amplitude(self, rng):
        t = np.arange(SRATE * 3) / SRATE
        # 100 Hz sine with amplitude 10 vs 1
        s1 = np.sin(2 * np.pi * 100 * t)
        s10 = 10 * s1
        hgp1 = high_gamma_power(s1[:, None], SRATE, smooth_ms=0)[:, 0]
        hgp10 = high_gamma_power(s10[:, None], SRATE, smooth_ms=0)[:, 0]
        # Power scales as amplitude²
        ratio = hgp10.mean() / (hgp1.mean() + 1e-10)
        assert 80 < ratio < 120, f"Expected ~100x power ratio, got {ratio:.1f}"


class TestFindStimulusOnsets:
    def test_detects_onsets(self):
        stim = np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 0], dtype=np.uint8)
        onsets = find_stimulus_onsets(stim)
        np.testing.assert_array_equal(onsets, [2, 7])

    def test_no_onset_at_zero(self):
        stim = np.zeros(100, dtype=np.uint8)
        assert len(find_stimulus_onsets(stim)) == 0

    def test_consecutive_stimuli(self):
        # stim changes value without going to 0 — not a new onset by our definition
        stim = np.array([0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        onsets = find_stimulus_onsets(stim)
        assert len(onsets) == 2


class TestEpochData:
    def test_epoch_shape(self, synthetic_ieeg):
        d = synthetic_ieeg
        hgp = np.abs(d["data"])
        ep = epoch_data(hgp, d["stim"], d["task"], d["target"],
                        pre_ms=100, post_ms=500, srate=d["srate"])
        N, T, C = ep["epochs"].shape
        assert T == int((100 + 500) * d["srate"] / 1000)
        assert C == d["data"].shape[1]

    def test_metadata_shapes(self, synthetic_ieeg):
        d = synthetic_ieeg
        hgp = np.abs(d["data"])
        ep = epoch_data(hgp, d["stim"], d["task"], d["target"])
        N = len(ep["epochs"])
        assert len(ep["task_id"]) == N
        assert len(ep["tgt_id"]) == N


class TestBaselineNormalize:
    def test_baseline_mean_zero(self):
        N, T, C = 20, 100, 4
        epochs = np.random.randn(N, T, C) + 5.0
        times = np.linspace(-0.2, 0.8, T)
        epochs_z = baseline_normalize(epochs, times, baseline_window=(-0.2, 0.0))
        bl = (times >= -0.2) & (times <= 0.0)
        np.testing.assert_allclose(
            epochs_z[:, bl, :].mean(axis=(0, 1)), np.zeros(C), atol=1e-5
        )

    def test_shape_preserved(self):
        epochs = np.random.randn(30, 80, 8).astype(np.float32)
        times = np.linspace(-0.2, 0.6, 80)
        out = baseline_normalize(epochs, times)
        assert out.shape == epochs.shape


class TestComputeHGP:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        epochs = rng.standard_normal((10, 8, 1200)).astype(np.float32)
        hgp = compute_hgp(epochs, srate=1200)
        assert hgp.shape == epochs.shape

    def test_output_nonnegative(self):
        rng = np.random.default_rng(1)
        epochs = rng.standard_normal((5, 4, 600)).astype(np.float32)
        hgp = compute_hgp(epochs, srate=1200)
        assert np.all(hgp >= 0)

    def test_output_dtype_float32(self):
        rng = np.random.default_rng(2)
        epochs = rng.standard_normal((4, 4, 600)).astype(np.float32)
        hgp = compute_hgp(epochs, srate=1200)
        assert hgp.dtype == np.float32


class TestBoranBaselineNormalize:
    def test_shape_preserved(self):
        rng = np.random.default_rng(3)
        epochs = rng.standard_normal((20, 8, 200)).astype(np.float32)
        times = np.linspace(-6.5, 2.5, 200)
        out = boran_baseline_normalize(epochs, times)
        assert out.shape == epochs.shape

    def test_baseline_zero_mean(self):
        rng = np.random.default_rng(4)
        N, C, T = 30, 6, 300
        epochs = rng.standard_normal((N, C, T)) + 5.0
        times = np.linspace(-6.5, 2.5, T)
        out = boran_baseline_normalize(epochs, times, baseline_window=(-6.5, -6.0))
        bl_mask = (times >= -6.5) & (times <= -6.0)
        np.testing.assert_allclose(
            out[:, :, bl_mask].mean(axis=(0, 2)), np.zeros(C), atol=1e-4
        )


@pytest.mark.skipif(not TES1_ZIP.exists(), reason="TES1 data not on external drive")
class TestTES1Loader:
    def test_load_all_subjects(self):
        data = load_tes1_stimulation(str(TES1_ZIP))
        assert isinstance(data, list)
        assert len(data) >= 10

    def test_single_subject_keys(self):
        d = load_tes1_stimulation(str(TES1_ZIP), subject="P04")
        assert "names" in d and "mni_coords" in d and "voltage_mV" in d
        assert d["mni_coords"].shape[1] == 3

    def test_electrode_count(self):
        d = load_tes1_stimulation(str(TES1_ZIP), subject="P04")
        assert len(d["names"]) == d["mni_coords"].shape[0] == len(d["voltage_mV"])
        assert len(d["names"]) > 50

    def test_voltage_range_physical(self):
        d = load_tes1_stimulation(str(TES1_ZIP), subject="P04")
        volts = d["voltage_mV"][~np.isnan(d["voltage_mV"])]
        assert volts.max() < 20.0   # mV per mA — physically plausible
        assert volts.min() > -20.0


@pytest.mark.skipif(not TES1_ZIP.exists(), reason="TES1 data not on external drive")
class TestBuildTES1InputMatrix:
    def test_output_shape(self):
        d = load_tes1_stimulation(str(TES1_ZIP), subject="P04")
        target_mni = np.array([[-40, 20, -10], [0, -30, 20], [30, 10, 5]])
        B = build_tes1_input_matrix(d, target_mni, n_sources=1)
        assert B.shape == (3, 1)

    def test_output_finite(self):
        d = load_tes1_stimulation(str(TES1_ZIP), subject="P04")
        target_mni = np.array([[-40, 20, -10]])
        B = build_tes1_input_matrix(d, target_mni)
        assert np.all(np.isfinite(B))


@pytest.mark.skipif(not BORAN_NWB.exists(), reason="DANDI 000574 data not on external drive")
class TestLoadBoranNWB:
    def test_output_keys(self):
        from preprocessing import load_boran_nwb
        result = load_boran_nwb(str(BORAN_NWB), signal="ieeg")
        for key in ("epochs", "times", "set_sizes", "correct", "artifact", "srate"):
            assert key in result

    def test_epochs_shape(self):
        from preprocessing import load_boran_nwb
        result = load_boran_nwb(str(BORAN_NWB), signal="ieeg")
        epochs = result["epochs"]
        assert epochs.ndim == 3             # (N, C, T)
        assert epochs.shape[0] == 50        # 50 trials per session
        assert epochs.shape[1] == 48        # 48 iEEG channels

    def test_set_sizes_valid(self):
        from preprocessing import load_boran_nwb
        result = load_boran_nwb(str(BORAN_NWB))
        assert set(np.unique(result["set_sizes"])).issubset({4, 6, 8})

    def test_srate_approx(self):
        from preprocessing import load_boran_nwb
        result = load_boran_nwb(str(BORAN_NWB))
        assert 1300 < result["srate"] < 1500
