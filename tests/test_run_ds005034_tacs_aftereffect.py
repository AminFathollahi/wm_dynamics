from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ds005034_tacs_aftereffect import MIN_CLEAN_PER_CELL, balanced_condition_summary, condition_summary, exact_vector_sign_flip, robust_high_outliers, spectral_features, transient_bad_channels
from preprocessing import load_events, periodogram_band_power
from corpus_sessions import paired_inventory
from preprocessing import CELLS


def test_paired_inventory_partitions_registered_participants(tmp_path):
    for subject, sessions in {
        "sub-complete": ("sham", "verum"),
        "sub-unpaired": ("sham",),
        "sub-absent": (),
    }.items():
        (tmp_path / subject).mkdir()
        for session in sessions:
            path = tmp_path / subject / f"ses-{session}" / "eeg"
            path.mkdir(parents=True)
            (path / f"{subject}_ses-{session}_task-memory_eeg.set").touch()

    inventory = paired_inventory(tmp_path)

    assert inventory["complete_pairs"] == ["sub-complete"]
    assert inventory["unpaired_raw"] == ["sub-unpaired"]
    assert inventory["no_raw_memory_recording"] == ["sub-absent"]
    assert inventory["paths"]["sub-complete"]["verum"].is_file()


def test_load_events_maps_recognized_codes_and_ignores_unmapped_rows(tmp_path):
    path = tmp_path / "events.tsv"
    path.write_text(
        "onset\tduration\tvalue\n"
        "1.25\t0\tDIN104\n"
        "2.50\t0\tDIN126\n"
        "3.75\t0\tDIN999\n"
    )

    assert load_events(path) == [
        {"onset": 1.25, "task": "forward", "load": 4, "code": "DIN104"},
        {"onset": 2.5, "task": "alphabetical", "load": 6, "code": "DIN126"},
    ]


def test_outlier_and_transient_channel_gates_only_flag_large_excursions():
    assert robust_high_outliers(np.array([1.0, 1.0, 1.0, 2.0])).tolist() == [False, False, False, True]
    epoch = np.zeros((6, 20))
    epoch[:4, -1] = 10.0
    epoch[4, -1] = 150.0
    epoch[5, -1] = 151.0
    assert transient_bad_channels(epoch).tolist() == [5]


def test_spectral_features_with_identity_transform_are_finite_and_channel_by_band():
    sfreq = 1000.0
    times = np.arange(int(12 * sfreq)) / sfreq - 5.0
    epoch = np.stack([
        np.sin(2 * np.pi * 6 * times),
        0.7 * np.sin(2 * np.pi * 10 * times + 0.8),
        0.5 * np.sin(2 * np.pi * 16 * times + 1.6),
    ])

    log_ratio, delay = spectral_features(epoch, sfreq, np.eye(epoch.shape[0]))

    assert log_ratio.shape == delay.shape == (3, 3)
    assert np.isfinite(log_ratio).all()
    assert np.isfinite(delay).all()
    direct = periodogram_band_power(epoch[:, ::4], 250.0, (0.5, 6.5), times[::4], np.eye(3))
    assert direct.shape == (3, 3)
    assert np.isfinite(direct).all()


def _balanced_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9)
    features, tasks, loads = [], [], []
    for task, load in CELLS:
        direction = rng.normal(size=(MIN_CLEAN_PER_CELL, 4))
        scale = rng.uniform(0.2, 4.0, size=(MIN_CLEAN_PER_CELL, 1))
        features.append(direction * scale)
        tasks.extend([task] * MIN_CLEAN_PER_CELL)
        loads.extend([load] * MIN_CLEAN_PER_CELL)
    return np.vstack(features), np.asarray(tasks), np.asarray(loads)


def test_condition_summary_requires_each_cell_and_is_rate_free():
    features, tasks, loads = _balanced_features()
    summary = condition_summary(features, tasks, loads)
    scaled = condition_summary(features * np.linspace(0.5, 2.0, len(features))[:, None], tasks, loads)

    assert set(summary["cell_counts"]) == {f"{task}_load{load}" for task, load in CELLS}
    assert set(summary["cell_counts"].values()) == {MIN_CLEAN_PER_CELL}
    assert np.isfinite(summary["equal_cell_mean_deviation"])
    assert np.isclose(summary["equal_cell_mean_deviation"], scaled["equal_cell_mean_deviation"])
    np.testing.assert_allclose(summary["equal_cell_centroid"], scaled["equal_cell_centroid"])
    with pytest.raises(ValueError, match="forward/load4 has 11 clean trials"):
        condition_summary(features[1:], tasks[1:], loads[1:])


def test_balanced_condition_summary_selects_equal_cells_deterministically():
    features, tasks, loads = _balanced_features()
    extra = np.repeat(features[:1], 3, axis=0)
    features = np.vstack([features, extra])
    tasks = np.append(tasks, ["forward"] * 3)
    loads = np.append(loads, [4] * 3)

    first = balanced_condition_summary(features, tasks, loads, MIN_CLEAN_PER_CELL, seed=71)
    second = balanced_condition_summary(features, tasks, loads, MIN_CLEAN_PER_CELL, seed=71)

    assert set(first["cell_counts"].values()) == {MIN_CLEAN_PER_CELL}
    assert first["cell_counts"] == second["cell_counts"]
    assert first["equal_cell_mean_deviation"] == second["equal_cell_mean_deviation"]
    np.testing.assert_array_equal(first["equal_cell_centroid"], second["equal_cell_centroid"])
    with pytest.raises(ValueError, match="forward/load4 cannot supply 16 trials"):
        balanced_condition_summary(features, tasks, loads, 16, seed=71)


def test_exact_vector_sign_flip_enumerates_the_two_pair_null():
    result = exact_vector_sign_flip(np.array([[1.0, 0.0], [1.0, 0.0]]))

    assert result["n_pairs"] == 2
    assert result["mean_vector_norm"] == 1.0
    assert result["standardized_vector_norm"] == 1.0
    assert result["exact_sign_flip_p_value"] == 0.5
    assert result["null_quantiles"]["q025"] == 0.0
    assert result["null_quantiles"]["q975"] == 1.0


def test_exact_vector_sign_flip_p_value_is_orthogonally_invariant():
    vectors = np.array([[1.0, 2.0, -1.0], [-2.0, 0.5, 1.5], [0.2, -1.2, 3.0]])
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, 1.0], [2.0, 1.0, -1.0], [1.0, -1.0, 2.0]]))

    original = exact_vector_sign_flip(vectors)
    rotated = exact_vector_sign_flip(vectors @ rotation)

    assert rotated["exact_sign_flip_p_value"] == original["exact_sign_flip_p_value"]
