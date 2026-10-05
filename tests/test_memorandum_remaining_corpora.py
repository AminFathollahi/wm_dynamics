import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import memorandum_decoding as md  # noqa: E402
import scalp_memorandum_decoding as scalp  # noqa: E402


def test_recognition_trials_map_to_category_labels_and_blank_windows():
    trials = {
        "stim_phase": np.array(["learn", "recog", "recog", "recog", "recog", "recog", "recog"]),
        "new_old_labels_recog": np.array(["NA", "0", "1", "0", "NA", "0", "1"]),
        "response_value": np.array([1.0, 35, 31, 32, 34, 34, 36]),
        "stim_on_time": np.array([0.0, 10, 20, 30, 40, 50, 60]),
        "stim_off_time": np.array([1.0, 11, 21, 31, 41, 51, 61]),
        "delay1_time": np.array([1.5, 11.55, 21.6, 31.5, 41.5, 51.3, np.nan])}
    chosen = md.select_recognition_trials(trials, blank_s=0.5)
    assert chosen["candidate"].tolist() == [False, True, True, True, False, False, False]
    assert chosen["correct"].tolist() == [False, True, True, False, False, False, False]
    assert chosen["removed_by_step"] == {
        "not_recognition_phase": 1, "invalid_truth_label_or_response": 1, "non_finite_times": 1,
        "post_picture_blank_shorter_than_window": 1, "error_trials": 1}
    assert np.allclose(chosen["blank_s"][[1, 2, 5]], [0.55, 0.6, 0.3])
    window, encoding = md.MAINTENANCE_S, md.ENCODING_SEGMENT_S
    md.configure_window_geometry(0.5, 1.2)
    try:
        spikes = [np.array([10.35, 10.9, 11.05, 11.6, 20.35, 21.45, 21.6])]
        session = {"patient": "p", "release": "000004", "session": "s", "spikes": spikes,
                   "regions": np.array(["hippocampus"]), "loads": np.ones(2, dtype=int),
                   "correct": np.array([True, True]), "category": np.array([3, 5]),
                   "t_picture": np.array([10.0, 20.0]), "t_maintenance": np.array([11.0, 21.0]),
                   "task_rate_hz": np.array([1.0])}
        entry = md.build_entry(session, 1)
        assert entry["labels"].tolist() == [3, 5]
        assert entry["features"]["selection_window"][:, 0, 0].tolist() == [3.0, 1.0]
        assert entry["features"]["encoding_100ms"].shape == (2, 1, 12)
        assert entry["features"]["encoding_100ms"][0, 0, 3] == 1.0
        assert entry["features"]["maintenance_whole"][:, 0, 0].tolist() == [1.0, 1.0]
        assert entry["features"]["maintenance_100ms"].shape == (2, 1, 5)
        assert entry["features"]["maintenance_100ms"][0, 0, 0] == 1.0
        assert entry["features"]["maintenance_250ms"].shape == (2, 1, 3)
    finally:
        md.configure_window_geometry(window, encoding)


def planted_scores(rng, n=180, channels=6):
    y = np.tile(np.arange(1, 4), n // 3)
    x = rng.normal(size=(n, channels, 1))
    x[:, 0, 0] += 1.5 * (y - 2)
    return x, y, md.stratified_fold_ids(y, 5, np.random.default_rng(1))


def test_scalp_nested_cross_validation_never_uses_held_out_trials():
    rng = np.random.default_rng(0)
    x, y, folds = planted_scores(rng)
    transforms = [(np.zeros(6), np.eye(6))] * 5
    valid, classes = np.ones(len(y), dtype=bool), np.arange(1, 4)
    base = scalp.score_readout(x, x, valid, y, folds, transforms, classes, seed=3, keep_scores=True)
    assert base["balanced_accuracy"][0] > 0.5
    held_out = folds == 0
    y_changed = y.copy()
    y_changed[held_out] = y[held_out] % 3 + 1
    relabelled = scalp.score_readout(x, x, valid, y_changed, folds, transforms, classes, seed=3, keep_scores=True)
    assert np.array_equal(base["picks"][0], relabelled["picks"][0])
    assert np.array_equal(base["fold_scores"][0][0], relabelled["fold_scores"][0][0])
    x_changed = x.copy()
    x_changed[held_out] = rng.normal(size=x[held_out].shape) * 5
    moved = scalp.score_readout(x_changed, x, valid, y, folds, transforms, classes, seed=3, keep_scores=True)
    assert np.array_equal(base["picks"][0], moved["picks"][0])
    assert np.array_equal(base["fold_scores"][0][0], moved["fold_scores"][0][0])


def test_scalp_channel_selection_uses_only_training_trials():
    rng = np.random.default_rng(2)
    n, channels = 240, 8
    y = np.tile(np.arange(1, 7), n // 6)
    window = rng.normal(size=(n, channels))
    window[:, 0] += 1.2 * y
    course = rng.normal(size=(n, 5, channels))
    rows = np.flatnonzero(np.arange(n) % 4 != 0)
    kept = scalp.fit_transforms(course, window, y, rows, ("native_selected",), np.random.default_rng(5), 200)
    weights = kept["native_selected"][1]
    assert weights.shape[1] <= 3 and weights[0].sum() == 1
    garbage_window, garbage_y = window.copy(), y.copy()
    garbage_window[np.arange(n) % 4 == 0] = rng.normal(size=(n // 4, channels)) * 10
    garbage_y[np.arange(n) % 4 == 0] = 1
    other = scalp.fit_transforms(course, garbage_window, garbage_y, rows, ("native_selected",),
                                 np.random.default_rng(5), 200)
    assert np.array_equal(weights, other["native_selected"][1])


def test_orientation_marginal_demixed_fit_recovers_the_planted_subspace():
    rng = np.random.default_rng(4)
    n, bins, channels = 360, 10, 12
    y = np.tile(np.arange(6), n // 6)
    ramp = rng.normal(size=channels)
    ramp /= np.linalg.norm(ramp)
    free = rng.normal(size=(channels, 4))
    planted = np.linalg.qr(free - np.outer(ramp, ramp @ free))[0]
    effects = rng.normal(size=(6, 4))
    effects -= effects.mean(axis=0)
    course = (rng.normal(size=(n, bins, channels)) * 0.5
              + np.linspace(0, 1, bins)[None, :, None] * 8.0 * ramp[None, None, :]
              + 2.0 * (effects[y] @ planted.T)[:, None, :])
    window = course.mean(axis=1)
    fits = scalp.fit_transforms(course, window, y, np.arange(n), ("demixed_principal_components", "principal_components"),
                                np.random.default_rng(0), 100)
    demixed_overlap = md.subspace_overlap(fits["demixed_principal_components"][1], planted)
    principal_overlap = md.subspace_overlap(fits["principal_components"][1], planted)
    assert demixed_overlap > 0.8 and demixed_overlap > principal_overlap
