import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / _subdir))

import run_readout_timing_pilot as pilot  # noqa: E402


def _synthetic_sessions():
    rows = []
    for region in ("hippocampus", "amygdala"):
        for i in range(12):
            rows.append({"status": "loaded", "level": pilot.LEVEL, "region": region,
                          "patient": f"P{i}", "session": "ses-1", "release": "001187"})
        # one patient with two sessions in the same region -- must contribute at most one entry
        rows.append({"status": "loaded", "level": pilot.LEVEL, "region": region,
                      "patient": "P0", "session": "ses-2", "release": "001187"})
        # a not-loaded row must be ignored
        rows.append({"status": "not_loaded", "level": pilot.LEVEL, "region": region,
                      "patient": "P99", "session": "ses-1", "release": "001187"})
    return rows


def test_drawn_entries_deterministic_and_bounded():
    rows = _synthetic_sessions()
    first = pilot.drawn_entries(rows)
    second = pilot.drawn_entries(rows)
    assert first == second
    assert len(first) == 2 * pilot.N_PER_REGION
    for region in pilot.REGIONS:
        region_entries = [e for e in first if e["region"] == region]
        assert len(region_entries) == pilot.N_PER_REGION
        patients = [e["patient"] for e in region_entries]
        assert len(patients) == len(set(patients))
        assert "P99" not in patients


def test_chronological_split_keeps_order_and_excludes_short_test_class():
    labels = np.array([0, 1] * 6)
    folds, info = pilot.chronological_split(labels)
    assert folds is not None
    (calib, test), = folds
    np.testing.assert_array_equal(calib, np.arange(0, 8))
    np.testing.assert_array_equal(test, np.arange(8, 12))
    assert set(calib.tolist()).isdisjoint(test.tolist())
    assert info["test_counts"] == {"0": 2, "1": 2}

    # all of one class lands in calibration only -> excluded, with counts
    short_labels = np.array([0] * 10 + [1] * 2)
    folds, info = pilot.chronological_split(short_labels)
    assert folds is None
    assert "test_counts" in info


def test_stratified_folds_equal_run_info_benchmarks_own_folds(monkeypatch, tmp_path):
    """The pilot's stratified split, and the new-model candidates registered through
    run_human_representation_benchmark's rebound REPRESENTATION_FITS, must reproduce the
    exact fold assignment run_info_benchmark.evaluate_block computes internally for the
    seven other native candidates -- not the different fold seed
    run_human_representation_benchmark.evaluate_demixed_block uses for demixed principal
    components alone. Captured directly from evaluate_block's own call to _seeded_folds,
    not re-derived, so this fails if either formula ever drifts."""
    import run_info_benchmark

    captured = {}
    original = run_info_benchmark._seeded_folds

    def _spy(labels, n_splits, seed):
        result = original(labels, n_splits, seed)
        captured["folds"] = result
        return result

    monkeypatch.setattr(run_info_benchmark, "_seeded_folds", _spy)

    rng = np.random.default_rng(0)
    labels = np.array([0, 1] * 15)
    activity = rng.normal(size=(30, 4, 3))

    run_info_benchmark.evaluate_block(
        "hippocampus", "P1_ses-1", pilot.LEVEL, activity, labels, ("native_full_rank",), ("linear",),
        pilot.N_SPLITS, 1, 1, lambda records: None, None, 0, tmp_path, "test_stratified_folds_match",
    )
    benchmark_folds = captured["folds"]
    assert benchmark_folds is not None

    pilot_folds, _ = pilot.stratified_split("hippocampus", "P1_ses-1", labels)
    assert len(pilot_folds) == len(benchmark_folds)
    for (train_a, test_a), (train_b, test_b) in zip(pilot_folds, benchmark_folds):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_chronological_null_draw_preserves_class_counts_within_each_set(monkeypatch):
    """info_decoding.score_null_draw's own null (_stratified_permutation) only fills the
    positions that lie in some fold's test set, leaving the rest as np.empty_like's
    uninitialised memory -- harmless with the benchmark's 5-fold cross-validation (every
    trial is someone's test position), fatal with the chronological split's single fold
    (the whole calibration set would be garbage). pilot._chronological_null_draw instead
    permutes the calibration set and the test set separately, each within itself: every
    drawn null label vector must therefore carry the same class counts in calibration and
    in test as the observed labels, and no value outside the label set."""
    captured = []

    def _spy(latent_train, y_train, latent_test, y_test, decoder=None, seed=None):
        captured.append((np.asarray(y_train).copy(), np.asarray(y_test).copy()))
        return 0.5

    monkeypatch.setattr(pilot, "split_auc", _spy)

    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    calib_idx, test_idx = np.arange(0, 8), np.arange(8, 12)
    fits = [{"latent_train": np.zeros((8, 3, 2)), "latent_test": np.zeros((4, 3, 2))}]

    label_set = set(labels.tolist())
    for draw in range(20):
        pilot._chronological_null_draw(fits, labels, "linear", calib_idx, test_idx,
                                        shuffle_seed=draw, decoder_seed=draw, time_step=1)
    assert len(captured) == 20
    for y_calib, y_test in captured:
        assert set(np.unique(y_calib).tolist()) <= label_set
        assert set(np.unique(y_test).tolist()) <= label_set
        for value in label_set:
            assert (y_calib == value).sum() == (labels[calib_idx] == value).sum()
            assert (y_test == value).sum() == (labels[test_idx] == value).sum()


def test_consistency_check_flags_mismatch_and_labels_sae_as_seed_variation():
    pilot_records = {
        ("hippocampus::P1::ses-1", "hippocampus", "P1_ses-1", "principal_components"):
            {"status": "computed", "same_time": {"auc": 0.6, "auc_above_null": 0.1}},
        ("hippocampus::P1::ses-1", "hippocampus", "P1_ses-1", "sequential_autoencoder"):
            {"status": "computed", "same_time": {"auc": 0.55, "auc_above_null": 0.05}},
    }
    seed0_index = {
        ("hippocampus", "P1_ses-1", pilot.LEVEL, "principal_components", "linear"):
            {"status": "computed", "same_time": {"auc": 0.7, "auc_above_null": 0.2}},
        ("hippocampus", "P1_ses-1", pilot.LEVEL, "sequential_autoencoder", "linear"):
            {"status": "computed", "same_time": {"auc": 0.50, "auc_above_null": 0.0}},
    }
    mismatches = pilot.consistency_check(pilot_records, seed0_index)
    kinds = {(m["candidate"], m["kind"]) for m in mismatches}
    assert ("principal_components", "mismatch") in kinds
    assert ("sequential_autoencoder", "seed_to_seed_variation_not_a_check") in kinds


def test_consistency_check_reaches_whole_trial_keys_and_holds_only_auc():
    pilot_records = {
        ("hippocampus::P1::ses-1", "hippocampus", "P1_ses-1", "gaussian_process_factor_analysis__whole_trial"):
            {"status": "computed", "same_time": {"auc": 0.6, "auc_above_null": 0.12}},
    }
    seed0_index = {
        ("hippocampus", "P1_ses-1", pilot.LEVEL, "gaussian_process_factor_analysis", "linear"):
            {"status": "computed", "same_time": {"auc": 0.6, "auc_above_null": 0.1}},
    }
    kinds = [m["kind"] for m in pilot.consistency_check(pilot_records, seed0_index)]
    assert kinds == ["difference_from_independent_null_draws_not_a_check"]
    seed0_index[next(iter(seed0_index))]["same_time"]["auc"] = 0.7
    kinds = [m["kind"] for m in pilot.consistency_check(pilot_records, seed0_index)]
    assert "mismatch" in kinds


def test_contrast_pairs_within_patient_and_needs_at_least_two_entries():
    scores = {
        ("e1", "stratified_folds", "a"): {"status": "computed", "same_time": {"auc_above_null": 0.2}},
        ("e1", "stratified_folds", "b"): {"status": "computed", "same_time": {"auc_above_null": 0.1}},
        ("e2", "stratified_folds", "a"): {"status": "computed", "same_time": {"auc_above_null": 0.3}},
        ("e2", "stratified_folds", "b"): {"status": "computed", "same_time": {"auc_above_null": 0.1}},
    }
    patient_by_entry = {"e1": "P1", "e2": "P2"}
    pairs = [(("e1", "stratified_folds", "a"), ("e1", "stratified_folds", "b")),
             (("e2", "stratified_folds", "a"), ("e2", "stratified_folds", "b"))]
    result = pilot.contrast(scores, pairs, patient_by_entry, "test")
    assert result["status"] == "computed"
    assert result["n_entries"] == 2
    assert result["n_patients"] == 2
    assert result["mean"] == pytest.approx(0.15)

    single = pilot.contrast(scores, pairs[:1], patient_by_entry, "test-single")
    assert single["status"] == "not_estimable"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("elephant") is None, reason="elephant not available"
)
def test_gpfa_causal_matches_up_to_truncation_only():
    rng = np.random.default_rng(4)
    n_train, n_test, n_bins, n_units = 8, 4, 6, 6
    train_X = rng.poisson(2.0, size=(n_train, n_bins, n_units)).astype(float)
    test_X = rng.poisson(2.0, size=(n_test, n_bins, n_units)).astype(float)
    test_X[1] = test_X[0].copy()
    test_X[1, 3:] = rng.poisson(6.0, size=(n_bins - 3, n_units))
    out = pilot._fit_gpfa(train_X, test_X, 2, rng, 100.0, True)
    assert out["status"] == "fitted"
    latent_test = out["latent_test"]
    np.testing.assert_allclose(latent_test[0, :3], latent_test[1, :3])
    assert not np.allclose(latent_test[0, 3], latent_test[1, 3])
