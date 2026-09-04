"""Tests for src/spike_pipeline.py."""

import h5py
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from spike_pipeline import (
    build_psth,
    zscore_psth,
    FrozenPSTHTransform,
    fit_pca_psth,
    load_vs_load_ctg,
    item_identity_ctg,
    pr_by_load,
    low_rate_unit_mask,
    normalize_region_label,
    resolve_unit_regions,
    filter_units_by_region,
    COARSE_REGION_GROUP,
)


def _synthetic_spike_lists(rng, n_units=10, rate_hz=5.0, duration_s=10.0):
    lists = []
    for _ in range(n_units):
        n_spikes = rng.poisson(rate_hz * duration_s)
        lists.append(np.sort(rng.uniform(0, duration_s, n_spikes)))
    return lists


class TestBuildPSTH:
    def test_output_shape(self, rng):
        spikes = _synthetic_spike_lists(rng, n_units=6)
        onsets = np.array([1.0, 3.0, 5.0])
        psth = build_psth(spikes, onsets, bin_ms=100, smooth_ms=0, window_s=2.0)
        assert psth.shape == (3, 6, 20)

    def test_rate_matches_expected_order_of_magnitude(self, rng):
        spikes = _synthetic_spike_lists(rng, n_units=4, rate_hz=10.0, duration_s=20.0)
        onsets = np.arange(0.5, 15.0, 1.0)
        psth = build_psth(spikes, onsets, bin_ms=100, smooth_ms=0, window_s=1.0)
        assert 3.0 < psth.mean() < 25.0

    def test_smoothing_reduces_variance(self, rng):
        spikes = _synthetic_spike_lists(rng, n_units=8, rate_hz=8.0, duration_s=20.0)
        onsets = np.arange(0.5, 15.0, 1.0)
        raw = build_psth(spikes, onsets, bin_ms=50, smooth_ms=0, window_s=2.0)
        smoothed = build_psth(spikes, onsets, bin_ms=50, smooth_ms=200, window_s=2.0)
        assert smoothed.var() < raw.var()


class TestFitPcaPsth:
    def test_output_shapes(self, rng):
        psth = rng.standard_normal((40, 12, 15)).astype(np.float32)
        Z, V, var_ratio = fit_pca_psth(psth, n_comp=5)
        assert Z.shape == (40, 15, 5)
        assert V.shape == (12, 5)
        assert 0.0 <= var_ratio <= 1.0 + 1e-8

    def test_low_rank_signal_captured(self, rng):
        N, U, T = 50, 12, 15
        direction = rng.standard_normal(U)
        amp = rng.standard_normal((N, 1, T))
        psth = (amp * direction[None, :, None]).astype(np.float32)
        psth += 0.01 * rng.standard_normal((N, U, T)).astype(np.float32)
        _, _, var_ratio = fit_pca_psth(psth, n_comp=3)
        assert var_ratio > 0.9


class TestFrozenPsthTransform:
    def test_uses_one_scale_across_time(self):
        train = np.zeros((4, 2, 3), dtype=float)
        train[:, 0, :] = np.array([1.0, 2.0, 3.0])
        train[:, 1, :] = np.array([4.0, 5.0, 6.0])
        transform = FrozenPSTHTransform().fit(train)
        held_out = train + 10.0
        out = transform.transform(held_out)
        # A time-varying z-score would make each time point separately zero;
        # the fixed transform preserves the known temporal displacement.
        assert not np.allclose(out[:, :, 0], out[:, :, 1])
        assert transform.mean_.shape == (1, 2, 1)

    def test_baseline_is_fit_from_train_only(self):
        train = np.ones((3, 1, 4))
        test = np.full((2, 1, 4), 5.0)
        transform = FrozenPSTHTransform(baseline_bins=slice(0, 2)).fit(train)
        np.testing.assert_allclose(transform.transform(test), 4.0)

    def test_diverges_from_legacy_time_varying_zscore(self):
        # A unit whose across-trial mean ramps over time: zscore_psth removes
        # that ramp bin-by-bin (each timepoint independently forced to mean
        # 0), destroying the very temporal structure a dynamics estimate
        # needs; FrozenPSTHTransform fits one mean/scale and leaves it intact.
        n_trials, n_bins = 20, 5
        rng = np.random.default_rng(0)
        ramp = np.linspace(0.0, 10.0, n_bins)
        psth = ramp[None, None, :] + 0.01 * rng.standard_normal((n_trials, 1, n_bins))

        legacy = zscore_psth(psth)
        frozen = FrozenPSTHTransform().fit_transform(psth)

        legacy_bin_means = legacy[:, 0, :].mean(axis=0)
        frozen_bin_means = frozen[:, 0, :].mean(axis=0)
        # Legacy: every timepoint independently zeroed -> flat at ~0.
        np.testing.assert_allclose(legacy_bin_means, 0.0, atol=1e-6)
        # Frozen: the ramp survives, so per-bin means still span the ramp.
        assert frozen_bin_means.max() - frozen_bin_means.min() > 1.0


class TestLoadVsLoadCTG:
    def test_returns_none_when_underpowered(self, rng):
        psth_z = rng.standard_normal((10, 8, 12)).astype(np.float32)
        loads = np.array([1] * 3 + [3] * 3 + [2] * 4)
        res = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=4, n_splits=2, n_perm=10, rng=rng)
        assert res is None

    def test_detects_signal(self, rng):
        N, U, T = 60, 10, 16
        loads = np.array([1] * 30 + [3] * 30)
        psth_z = rng.standard_normal((N, U, T)).astype(np.float32)
        direction = rng.standard_normal(U)
        psth_z += 3.0 * (loads == 3)[:, None, None] * direction[None, :, None]
        res = load_vs_load_ctg(psth_z, loads, 1, 3, n_components=4, n_splits=3, n_perm=50, rng=rng)
        assert res is not None
        assert res["p_value"] < 0.05


class TestItemIdentityCTG:
    def test_returns_none_when_too_few_trials(self, rng):
        psth_z = rng.standard_normal((10, 8, 12)).astype(np.float32)
        loads = np.ones(10, dtype=int)
        items = rng.integers(0, 5, 10)
        res = item_identity_ctg(psth_z, loads, items, target_load=1, min_trials=20, rng=rng)
        assert res is None

    def test_returns_none_when_single_trial_per_class(self, rng):
        N, U, T = 25, 8, 10
        psth_z = rng.standard_normal((N, U, T)).astype(np.float32)
        loads = np.ones(N, dtype=int)
        items = np.arange(N) % 25  # every item unique -> min class count 1
        res = item_identity_ctg(psth_z, loads, items, target_load=1, min_trials=20, rng=rng)
        assert res is None

    def test_detects_multiclass_signal(self, rng):
        N, U, T, n_classes = 100, 10, 12, 4
        loads = np.ones(N, dtype=int)
        items = rng.integers(0, n_classes, N)
        psth_z = rng.standard_normal((N, U, T)).astype(np.float32)
        directions = rng.standard_normal((n_classes, U))
        for c in range(n_classes):
            psth_z[items == c] += 3.0 * directions[c][None, :, None]
        res = item_identity_ctg(psth_z, loads, items, target_load=1, min_trials=20,
                                n_components=6, n_perm=50, rng=rng)
        assert res is not None
        assert res["p_value"] < 0.05


class TestPRByLoad:
    def test_only_qualifying_loads_included(self, rng):
        psth_z = rng.standard_normal((20, 6, 10)).astype(np.float32)
        loads = np.array([1] * 15 + [2] * 3 + [3] * 2)
        res = pr_by_load(psth_z, loads, load_values=(1, 2, 3), min_trials=5, rng=rng)
        assert 1 in res
        assert 2 not in res
        assert 3 not in res


class TestLowRateUnitMask:
    def test_keeps_active_unit_drops_silent_unit(self):
        onsets = np.arange(0.0, 100.0, 2.0)   # 50 trials
        window_s = 2.0
        active = np.arange(0.5, 100.0, 2.0)   # 1 spike/trial -> 0.5 Hz, well above floor
        silent = np.array([])                  # 0 Hz
        mask = low_rate_unit_mask([active, silent], onsets, window_s, min_rate_hz=0.1)
        np.testing.assert_array_equal(mask, [True, False])

    def test_threshold_boundary(self):
        # 10 trials x 1s window = 10s total; exactly 1 spike -> 0.1 Hz.
        onsets = np.arange(0.0, 10.0, 1.0)
        window_s = 1.0
        exactly_at_floor = np.array([0.5])   # 1 spike total -> 1/10 = 0.1 Hz
        below_floor = np.array([])
        mask = low_rate_unit_mask([exactly_at_floor, below_floor], onsets, window_s,
                                  min_rate_hz=0.1)
        np.testing.assert_array_equal(mask, [True, False])

    def test_empty_onsets_keeps_nothing(self):
        mask = low_rate_unit_mask([np.array([1.0, 2.0])], np.array([]), window_s=1.0)
        np.testing.assert_array_equal(mask, [False])


class TestNormalizeRegionLabel:
    def test_known_structure_with_hemisphere(self):
        assert normalize_region_label("hippocampus_left") == ("hippocampus", "left")
        assert normalize_region_label("dorsal_anterior_cingulate_cortex_right") == ("dacc", "right")

    def test_unrecognized_stem_is_other(self):
        assert normalize_region_label("insula_left") == ("other", "left")

    def test_no_hemisphere_suffix(self):
        assert normalize_region_label("hippocampus") == ("hippocampus", None)


class TestNormalizeRegionLabelPerCorpusConventions:
    """One real raw label per corpus, plus one unmappable
    string per convention asserted to reach "other" with the raw text
    retained by the caller (normalize_region_label itself only returns the
    structure/hemisphere tuple; retention is the caller's job)."""

    def test_boran_000574_hippocampus(self):
        assert normalize_region_label(
            "Hipp, Left Hippocampus cHipp, caudal hippocampus",
            "nwb_boran_brainnetome_hybrid",
        ) == ("hippocampus", "left")

    def test_boran_000574_vtc_fusiform(self):
        assert normalize_region_label(
            "FuG, Right Fusiform Gyrus A20rv, rostroventral area 20",
            "nwb_boran_brainnetome_hybrid",
        ) == ("vtc", "right")

    def test_boran_000574_unspecific_is_own_status_not_other(self):
        assert normalize_region_label("unspecific", "nwb_boran_brainnetome_hybrid") == ("unspecific", None)

    def test_boran_000574_unmappable_is_other(self):
        structure, hemisphere = normalize_region_label(
            "Xyz, Left Nonsense Structure", "nwb_boran_brainnetome_hybrid"
        )
        assert structure == "other"

    def test_ds004752_hippocampus_shares_boran_parser(self):
        assert normalize_region_label(
            "Hipp, Left Hippocampus rHipp, rostral hippocampus",
            "bids_brainnetome_anatomical_location",
        ) == ("hippocampus", "left")

    def test_ds004752_no_label_found_is_unlabelled_not_other(self):
        assert normalize_region_label("no_label_found", "bids_brainnetome_anatomical_location") == (
            "unlabelled", None,
        )

    def test_ram_ind_region_temporalpole(self):
        assert normalize_region_label("temporalpole", "bids_desikan_killiany_ind_region") == (
            "temporal_pole", None,
        )

    def test_ram_hand_annotation_hippocampal_subfield(self):
        # das.region-style bare code, no hemisphere token.
        assert normalize_region_label("CA1", "bids_desikan_killiany_ind_region") == ("hippocampus", None)

    def test_ram_wb_region_amygdala_with_hemisphere(self):
        assert normalize_region_label("Left Amygdala", "bids_desikan_killiany_ind_region") == (
            "amygdala", "left",
        )

    def test_ram_n_a_is_unlabelled_not_other(self):
        assert normalize_region_label("n/a", "bids_desikan_killiany_ind_region") == ("unlabelled", None)

    def test_ram_unmappable_is_other(self):
        structure, _ = normalize_region_label("totallyunknownlabel", "bids_desikan_killiany_ind_region")
        assert structure == "other"

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError):
            normalize_region_label("hippocampus_left", "nonexistent_convention")

    def test_every_reachable_structure_has_a_coarse_group(self):
        # COARSE_REGION_GROUP must cover everything the parsers can emit, or
        # a later analysis silently loses the grain a caller asked to choose.
        reachable = set()
        for raw in ("Hipp, Left Hippocampus", "unspecific", "no_label_found"):
            for conv in ("nwb_boran_brainnetome_hybrid", "bids_brainnetome_anatomical_location"):
                reachable.add(normalize_region_label(raw, conv)[0])
        for raw in ("temporalpole", "n/a", "CA1", "Left Amygdala"):
            reachable.add(normalize_region_label(raw, "bids_desikan_killiany_ind_region")[0])
        for raw in ("hippocampus_left", "insula_left"):
            reachable.add(normalize_region_label(raw)[0])
        for structure in reachable:
            assert structure in COARSE_REGION_GROUP, structure


def _write_nwb_stub(path, locations, unit_electrodes, ragged=False):
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "general/extracellular_ephys/electrodes/location",
            data=np.array(locations, dtype="S"),
        )
        f.create_dataset("units/id", data=np.arange(len(unit_electrodes) if not ragged
                                                      else len(unit_electrodes)))
        if ragged:
            flat, index = [], []
            for rows in unit_electrodes:
                flat.extend(rows)
                index.append(len(flat))
            f.create_dataset("units/electrodes", data=np.array(flat, dtype="int32"))
            f.create_dataset("units/electrodes_index", data=np.array(index, dtype="int64"))
        else:
            f.create_dataset("units/electrodes", data=np.array(unit_electrodes, dtype="int32"))


class TestResolveUnitRegions:
    def test_flat_layout(self, tmp_path):
        path = tmp_path / "flat.nwb"
        _write_nwb_stub(
            path,
            locations=["hippocampus_left", "amygdala_right", "pre_supplementary_motor_area_left"],
            unit_electrodes=[0, 0, 1, 2],
        )
        with h5py.File(path, "r") as f:
            out = resolve_unit_regions(f)
        np.testing.assert_array_equal(out["region"], ["hippocampus", "hippocampus", "amygdala", "pre_sma"])
        np.testing.assert_array_equal(out["hemisphere"], ["left", "left", "right", "left"])

    def test_ragged_layout_single_electrode_per_unit(self, tmp_path):
        path = tmp_path / "ragged.nwb"
        _write_nwb_stub(
            path,
            locations=["hippocampus_left", "amygdala_right"],
            unit_electrodes=[[0], [1], [0]],
            ragged=True,
        )
        with h5py.File(path, "r") as f:
            out = resolve_unit_regions(f)
        np.testing.assert_array_equal(out["region"], ["hippocampus", "amygdala", "hippocampus"])

    def test_ragged_layout_disagreeing_regions_is_other(self, tmp_path):
        path = tmp_path / "ragged_mixed.nwb"
        _write_nwb_stub(
            path,
            locations=["hippocampus_left", "amygdala_right"],
            unit_electrodes=[[0, 1]],
            ragged=True,
        )
        with h5py.File(path, "r") as f:
            out = resolve_unit_regions(f)
        np.testing.assert_array_equal(out["region"], ["other"])
        assert out["raw"][0] == "hippocampus_left|amygdala_right"

    def test_pooled_returns_every_unit_unfiltered(self):
        items = ["a", "b", "c"]
        assert filter_units_by_region(items, np.array(["hippocampus", "amygdala", "other"]), "pooled") == items

    def test_region_keeps_only_matching_units(self):
        items = [10, 20, 30, 40]
        regions = np.array(["hippocampus", "amygdala", "hippocampus", "vmpfc"])
        assert filter_units_by_region(items, regions, "hippocampus") == [10, 30]

    def test_missing_electrode_metadata_raises_with_filename(self, tmp_path):
        path = tmp_path / "broken.nwb"
        with h5py.File(path, "w") as f:
            f.create_dataset("units/id", data=np.arange(3))
        with h5py.File(path, "r") as f:
            with pytest.raises(RuntimeError, match=r"broken\.nwb"):
                resolve_unit_regions(f)
