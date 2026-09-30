from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_info_benchmark as mod  # noqa: E402
import info_decoding  # noqa: E402
from info_decoding import _stratified_permutation, make_folds  # noqa: E402
from info_decoding import CANDIDATES, DECODERS, atomic_write, fold_signature
from info_decoding import _metric, score_null_draw, score_observed
from info_decoding import render_summary


def test_metric_uses_matched_draws():
    observed = np.array([0.8, 0.7])
    null = np.array([[0.5, 0.5], [0.6, 0.5], [0.4, 0.6]])
    result = _metric(observed, null)
    assert result["status"] == "computed"
    assert np.isclose(result["auc"], 0.75)
    assert np.isclose(result["null_auc"], 31 / 60)
    assert result["p_value"] == 0.25


def test_scoring_uses_fold_specific_latents_for_observed_and_null(monkeypatch):
    folds = (
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([2, 3]), np.array([0, 1])),
    )
    seen = []

    def split(train_activity, train_labels, test_activity, test_labels, **kwargs):
        seen.append((train_activity.copy(), test_activity.copy()))
        return np.array([[0.8, 0.7], [0.7, 0.8]])

    monkeypatch.setattr(
        info_decoding,
        "_decoder_api",
        lambda: SimpleNamespace(
            split_auc=split,
            stratified_permutation=lambda labels, folds, rng: labels[::-1],
        ),
    )
    fits = [
        {"latent_train": np.full((2, 2, 3), i), "latent_test": np.full((2, 2, 3), i + 10)}
        for i in range(2)
    ]
    observed = score_observed(fits, np.array([0, 0, 1, 1]), "linear", folds, 3)
    null = score_null_draw(
        fits, np.array([0, 0, 1, 1]), "linear", folds, 4, 3
    )
    assert observed["status"] == "computed"
    assert null["status"] == "computed"
    assert np.array_equal(seen[0][0], fits[0]["latent_train"])
    assert np.array_equal(seen[3][0], fits[1]["latent_train"])


def test_null_draw_assigns_one_label_per_trial_across_folds(monkeypatch):
    labels = np.repeat(np.arange(3), 4)
    folds = make_folds(labels, n_splits=3, seed=2)
    fits = []
    for train, test in folds:
        fits.append({
            "latent_train": train[:, None, None],
            "latent_test": test[:, None, None],
        })
    assignments = {}

    def split(train_activity, train_labels, test_activity, test_labels, **kwargs):
        for trial, label in zip(train_activity[:, 0, 0], train_labels):
            assignments.setdefault(int(trial), set()).add(int(label))
        for trial, label in zip(test_activity[:, 0, 0], test_labels):
            assignments.setdefault(int(trial), set()).add(int(label))
        return np.full((1, 1), 0.5)

    monkeypatch.setattr(
        info_decoding,
        "_decoder_api",
        lambda: SimpleNamespace(
            split_auc=split,
            stratified_permutation=_stratified_permutation,
        ),
    )
    score_null_draw(fits, labels, "linear", folds, 7, 11)
    assert set(assignments) == set(range(labels.size))
    assert all(len(values) == 1 for values in assignments.values())


def test_native_activity_runs_both_decoders_on_shared_folds(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(
        mod,
        "_seeded_folds",
        lambda labels, n_splits, seed: ((np.array([0, 2]), np.array([1, 3])),),
    )
    monkeypatch.setattr(
        mod,
        "fit_candidate",
        lambda train_activity, test_activity, candidate, seed: {
            "status": "fitted",
            "candidate": candidate,
            "k_used": train_activity.shape[-1],
            "latent_train": train_activity,
            "latent_test": test_activity,
        },
    )
    calls = []

    def score(fits, labels, decoder, folds, seed, time_step):
        calls.append((decoder, folds))
        return {"status": "computed", "temporal_auc": np.full((2, 2), 0.7)}

    monkeypatch.setattr(mod, "score_observed", score)
    monkeypatch.setattr(
        mod,
        "score_null_draw",
        lambda fits, labels, decoder, folds, shuffle_seed, decoder_seed, time_step: {
            "status": "computed",
            "temporal_auc": np.full((2, 2), 0.5),
        },
    )
    records = mod.evaluate_block(
        "corpus",
        "session",
        "all",
        np.zeros((4, 2, 3)),
        np.array([0, 1, 0, 1]),
        ("native_full_rank",),
        ("linear", "nonlinear"),
        2,
        3,
        1,
        lambda records: None,
    )
    assert [record["decoder"] for record in records] == ["linear", "nonlinear"]
    assert all(record["status"] == "computed" for record in records)
    assert calls[0][1] is calls[1][1]


def test_completed_records_preserve_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: ((np.array([0, 2]), np.array([1, 3])),))
    monkeypatch.setattr(mod, "fit_candidate", lambda train_activity, test_activity, candidate, seed: {"status": "fitted", "k_used": 1, "latent_train": train_activity, "latent_test": test_activity})
    monkeypatch.setattr(mod, "score_observed", lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(2)})
    monkeypatch.setattr(mod, "score_null_draw", lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(2) / 2})
    records = mod.evaluate_block("c", "s", "all", np.zeros((4, 2, 1)), np.array([0, 1, 0, 1]), ("native_full_rank",), ("linear",), 2, 1, 1, lambda _: None, seed=17)
    assert records[0]["seed"] == 17


def test_representation_receives_exact_outer_fold_arrays(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    train = np.array([0, 3, 4, 7])
    test = np.array([1, 2, 5, 6])
    folds = ((train, test),)
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: folds)
    activity = np.arange(8 * 3 * 2, dtype=float).reshape(8, 3, 2)
    seen = []

    def fit(train_activity, test_activity, candidate, seed):
        seen.append((train_activity.copy(), test_activity.copy()))
        return {
            "status": "fitted",
            "candidate": candidate,
            "k_used": 2,
            "latent_train": train_activity,
            "latent_test": test_activity,
        }

    monkeypatch.setattr(mod, "fit_candidate", fit)
    monkeypatch.setattr(
        mod,
        "score_observed",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(3)},
    )
    monkeypatch.setattr(
        mod,
        "score_null_draw",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(3) / 2},
    )
    mod.evaluate_block(
        "c", "s", "all", activity, np.tile([0, 1], 4),
        ("native_full_rank",), ("linear",), 2, 1, 1, lambda records: None,
    )
    np.testing.assert_array_equal(seen[0][0], activity[train])
    np.testing.assert_array_equal(seen[0][1], activity[test])


def test_null_draws_are_matched_across_candidates_and_decoders(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    folds = ((np.array([0, 2]), np.array([1, 3])),)
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: folds)
    monkeypatch.setattr(mod, "implementation_identity", lambda: {"version": "a"})
    monkeypatch.setattr(
        mod,
        "fit_candidate",
        lambda train_activity, test_activity, candidate, seed: {
            "status": "fitted",
            "candidate": candidate,
            "k_used": 2,
            "latent_train": train_activity,
            "latent_test": test_activity,
        },
    )
    monkeypatch.setattr(
        mod,
        "score_observed",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(2)},
    )
    seeds = []

    def null(fits, labels, decoder, folds, shuffle_seed, decoder_seed, time_step):
        seeds.append(shuffle_seed)
        return {"status": "computed", "temporal_auc": np.eye(2) / 2}

    monkeypatch.setattr(mod, "score_null_draw", null)
    mod.evaluate_block(
        "c", "s", "all", np.zeros((4, 2, 2)), np.array([0, 1, 0, 1]),
        ("native_full_rank", "principal_components"),
        ("linear", "nonlinear"), 2, 2, 1, lambda records: None,
    )
    counts = {seed: seeds.count(seed) for seed in set(seeds)}
    assert sorted(counts.values()) == [4, 4]


def test_checkpoint_and_summary_are_incrementally_recoverable(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    record = {"status": "computed", "values": np.array([1.0, 2.0])}
    mod.save_checkpoint("a/cell", record)
    restored = mod.load_checkpoint("a/cell")
    assert np.array_equal(restored["values"], record["values"])
    path = tmp_path / "summary.md"
    row = {
        "status": "computed",
        "corpus": "c",
        "session": "s",
        "level": "all",
        "candidate": "native_full_rank",
        "decoder": "linear",
        "same_time": {
            "auc": 0.7,
            "null_auc": 0.5,
            "auc_above_null": 0.2,
            "p_value": 0.01,
        },
        "cross_temporal": {
            "auc": 0.6,
            "null_auc": 0.5,
            "auc_above_null": 0.1,
            "p_value": 0.04,
        },
    }
    atomic_write(path, render_summary([row], complete=False))
    text = path.read_text()
    assert "Status: running" in text
    assert "native_full_rank" in text
    assert "Same-time null" in text
    assert "Cross-time null" in text
    assert "0.700 | 0.500 | 0.200 | 0.01" in text
    assert "0.600 | 0.500 | 0.100 | 0.04" in text
    assert "Same-time p" in text
    assert "0.01" in text
    assert "0.04" in text
    assert not list(tmp_path.glob(".summary.md.*"))


@pytest.mark.parametrize(
    ("latent_train", "latent_test", "reason"),
    (
        (np.zeros((4, 2, 2)), np.zeros((3, 2, 3)), "invalid latent shape"),
        (np.full((4, 2, 2), np.nan), np.zeros((3, 2, 2)), "non-finite"),
    ),
)
def test_candidate_rejects_invalid_latents(
    monkeypatch, latent_train, latent_test, reason
):
    monkeypatch.setitem(
        mod.REPRESENTATION_FITS,
        "native_full_rank",
        lambda *args: {
            "status": "fitted",
            "k_used": 2,
            "latent_train": latent_train,
            "latent_test": latent_test,
        },
    )
    result = mod.fit_candidate(
        np.zeros((4, 2, 3)), np.zeros((3, 2, 3)), "native_full_rank", 2
    )
    assert result["status"] == "failed_to_train"
    assert reason in result["reason"]


def test_checkpoint_rejects_foreign_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    path = mod._checkpoint_path("cell")
    path.write_text(json.dumps({"schema": "old", "complete": True, "record": {"status": "computed"}}))
    assert mod.load_checkpoint("cell") is None


def test_transient_failure_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    calls = []

    def run():
        calls.append(None)
        status = "failed_to_train" if len(calls) == 1 else "fitted"
        return {"status": status}

    first, first_hit = mod.run_checkpointed("cell", run, {"data": "a"})
    second, second_hit = mod.run_checkpointed("cell", run, {"data": "a"})
    third, third_hit = mod.run_checkpointed("cell", run, {"data": "a"})
    assert first["status"] == "failed_to_train"
    assert second["status"] == "fitted"
    assert (first_hit, second_hit, third_hit) == (False, False, True)
    assert len(calls) == 2


def test_checkpoint_requires_matching_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    mod.save_checkpoint("cell", {"status": "computed"}, {"data": "a", "code": "1"})
    assert mod.load_checkpoint("cell", {"data": "a", "code": "1"}) is not None
    assert mod.load_checkpoint("cell", {"data": "b", "code": "1"}) is None
    assert mod.load_checkpoint("cell", {"data": "a", "code": "2"}) is None


def test_lfads_identity_normalizes_distribution_name(monkeypatch, tmp_path):
    metadata = tmp_path / "lfads_torch-9.9.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lfads_torch\nVersion: 9.9\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    identity = mod._lfads_identity(sys.executable)
    assert identity["runtime"]["packages"]["lfads-torch"] == "9.9"


def test_fold_signature_changes_with_split_definition():
    folds_a = ((np.array([0, 1]), np.array([2, 3])), (np.array([2, 3]), np.array([0, 1])))
    folds_b = ((np.array([0, 2]), np.array([1, 3])), (np.array([1, 3]), np.array([0, 2])))
    assert fold_signature(folds_a) != fold_signature(folds_b)


def test_changed_fold_definition_does_not_reuse_representation(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    folds = {
        2: ((np.array([0, 1]), np.array([2, 3])),),
        3: ((np.array([0, 2]), np.array([1, 3])),),
    }
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: folds[n_splits])
    calls = []

    def fit(train_activity, test_activity, candidate, seed):
        calls.append(seed)
        return {
            "status": "fitted",
            "candidate": candidate,
            "k_used": train_activity.shape[-1],
            "latent_train": train_activity,
            "latent_test": test_activity,
        }

    monkeypatch.setattr(mod, "fit_candidate", fit)
    monkeypatch.setattr(
        mod,
        "score_observed",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(3)},
    )
    monkeypatch.setattr(
        mod,
        "score_null_draw",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(3) / 2},
    )
    arguments = (
        "c",
        "s",
        "all",
        np.zeros((4, 3, 2)),
        np.array([0, 0, 1, 1]),
        ("native_full_rank",),
        ("linear",),
    )
    mod.evaluate_block(*arguments, 2, 1, 1, lambda records: None)
    mod.evaluate_block(*arguments, 3, 1, 1, lambda records: None)
    assert len(calls) == 2


def test_representations_fit_full_resolution_before_decoder_thinning(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    folds = ((np.array([0, 2]), np.array([1, 3])),)
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: folds)
    shapes = []

    def fit(train_activity, test_activity, candidate, seed):
        shapes.append(train_activity.shape)
        return {
            "status": "fitted",
            "candidate": candidate,
            "k_used": train_activity.shape[-1],
            "latent_train": train_activity,
            "latent_test": test_activity,
        }

    monkeypatch.setattr(mod, "fit_candidate", fit)
    monkeypatch.setattr(
        mod,
        "score_observed",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(3)},
    )
    monkeypatch.setattr(
        mod,
        "score_null_draw",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(3) / 2},
    )
    mod.evaluate_block(
        "c",
        "s",
        "all",
        np.zeros((4, 7, 3)),
        np.array([0, 1, 0, 1]),
        ("native_full_rank",),
        ("linear",),
        2,
        2,
        3,
        lambda records: None,
    )
    assert shapes == [(2, 7, 3)]


def test_decoder_scoring_applies_time_step_to_fold_latents(monkeypatch):
    folds = (
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([2, 3]), np.array([0, 1])),
    )
    fits = [
        {
            "latent_train": np.zeros((2, 7, 3)),
            "latent_test": np.zeros((2, 7, 3)),
        }
        for _ in folds
    ]
    shapes = []

    def split(train_activity, train_labels, test_activity, test_labels, **kwargs):
        shapes.append((train_activity.shape, test_activity.shape))
        return np.zeros((3, 3))

    monkeypatch.setattr(
        info_decoding,
        "_decoder_api",
        lambda: SimpleNamespace(
            split_auc=split,
            stratified_permutation=_stratified_permutation,
        ),
    )
    labels = np.array([0, 1, 0, 1])
    score_observed(fits, labels, "linear", folds, 2, time_step=3)
    score_null_draw(fits, labels, "linear", folds, 3, 2, time_step=3)
    assert shapes == [((2, 3, 3), (2, 3, 3))] * 4


def test_cache_identity_tracks_transitive_sources():
    paths = {path.name for path in mod.IMPLEMENTATION_PATHS}
    assert {
        "statistics.py",
        "project_config.py",
        "run_latent_model_comparison.py",
        "run_state_space_estimation_robustness.py",
        "run_dissociation_cross_preparation_test.py",
        "project.json",
    } <= paths


def test_cebra_padding_uses_training_filler_and_separates_trials():
    train = np.arange(12, dtype=float).reshape(6, 2)
    test = np.arange(100, 108, dtype=float).reshape(4, 2)
    filler = train.mean(axis=0)
    padded, is_real = mod.estimation._trial_gap_pad(
        test,
        n_trials=2,
        n_bins=2,
        gap=mod.estimation.CEBRA_BOUNDARY_GAP,
        filler=filler,
    )
    separators = padded[~is_real]
    np.testing.assert_allclose(
        separators, np.repeat(filler[None], len(separators), axis=0)
    )
    assert mod.estimation.CEBRA_BOUNDARY_GAP >= mod.estimation.CEBRA_TIME_OFFSET
    assert mod.estimation.CEBRA_BOUNDARY_GAP >= mod.estimation.CEBRA_RECEPTIVE_FIELD


def test_tphate_padding_uses_training_filler_and_removes_separators(monkeypatch):
    captured = {}

    class FakeTPHATE:
        def __init__(self, n_components, **kwargs):
            captured["n_components"] = n_components
            captured["smooth_window"] = kwargs["smooth_window"]

        def fit_transform(self, values):
            captured["train"] = values.copy()
            return values[:, : captured["n_components"]]

        def transform(self, values):
            captured["test"] = values.copy()
            return values[:, : captured["n_components"]]

    monkeypatch.setitem(sys.modules, "tphate", SimpleNamespace(TPHATE=FakeTPHATE))
    train = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    test = np.arange(100, 100 + 2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    result = mod.estimation.fit_temporal_diffusion_embedding(
        train, test, 2, np.random.default_rng(4), False, 100.0
    )
    filler = train.reshape(-1, 4).mean(axis=0)
    expected_train, train_real = mod.estimation._trial_gap_pad(
        train.reshape(-1, 4), 2, 4, captured["smooth_window"], filler
    )
    expected_test, test_real = mod.estimation._trial_gap_pad(
        test.reshape(-1, 4), 2, 4, captured["smooth_window"], filler
    )
    assert result["status"] == "fitted"
    assert captured["smooth_window"] == 3
    np.testing.assert_array_equal(captured["train"], expected_train)
    np.testing.assert_array_equal(captured["test"], expected_test)
    np.testing.assert_array_equal(
        result["latent_train"], expected_train[train_real, :2].reshape(2, 4, 2)
    )
    np.testing.assert_array_equal(
        result["latent_test"], expected_test[test_real, :2].reshape(2, 4, 2)
    )


def test_changed_data_and_implementation_refit(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    folds = ((np.array([0, 2]), np.array([1, 3])),)
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: folds)
    version = {"value": "a"}
    monkeypatch.setattr(mod, "implementation_identity", lambda: {"version": version["value"]})
    calls = []

    def fit(train_activity, test_activity, candidate, seed):
        calls.append(train_activity.copy())
        return {
            "status": "fitted",
            "candidate": candidate,
            "k_used": train_activity.shape[-1],
            "latent_train": train_activity,
            "latent_test": test_activity,
        }

    monkeypatch.setattr(mod, "fit_candidate", fit)
    monkeypatch.setattr(
        mod,
        "score_observed",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(2)},
    )
    monkeypatch.setattr(
        mod,
        "score_null_draw",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(2) / 2},
    )
    base = np.zeros((4, 2, 2))
    args = (
        "c", "s", "all", np.array([0, 1, 0, 1]),
        ("native_full_rank",), ("linear",), 2, 1, 1, lambda records: None,
    )
    mod.evaluate_block(args[0], args[1], args[2], base, *args[3:])
    mod.evaluate_block(args[0], args[1], args[2], base, *args[3:])
    changed = base.copy()
    changed[0, 0, 0] = 1
    mod.evaluate_block(args[0], args[1], args[2], changed, *args[3:])
    version["value"] = "b"
    mod.evaluate_block(args[0], args[1], args[2], changed, *args[3:])
    assert len(calls) == 3


def test_null_draws_resume_after_interruption(monkeypatch, tmp_path):
    resumed_dir = tmp_path / "resumed"
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", resumed_dir)
    folds = ((np.array([0, 2]), np.array([1, 3])),)
    monkeypatch.setattr(mod, "_seeded_folds", lambda labels, n_splits, seed: folds)
    monkeypatch.setattr(mod, "implementation_identity", lambda: {"version": "a"})
    monkeypatch.setattr(
        mod,
        "fit_candidate",
        lambda train_activity, test_activity, candidate, seed: {
            "status": "fitted",
            "candidate": candidate,
            "k_used": 2,
            "latent_train": train_activity,
            "latent_test": test_activity,
        },
    )
    monkeypatch.setattr(
        mod,
        "score_observed",
        lambda *args, **kwargs: {"status": "computed", "temporal_auc": np.eye(2)},
    )
    calls = []

    def matrix(seed):
        value = 0.4 + (seed % 100) / 1_000
        return np.array([[value, value + 0.01], [value + 0.02, value + 0.03]])

    def interrupted(*args, **kwargs):
        shuffle_seed = args[-2]
        calls.append(shuffle_seed)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return {"status": "computed", "temporal_auc": matrix(shuffle_seed)}

    monkeypatch.setattr(mod, "score_null_draw", interrupted)
    updates = []
    call = (
        "c", "s", "all", np.zeros((4, 2, 2)), np.array([0, 1, 0, 1]),
        ("native_full_rank",), ("linear",), 2, 3, 1,
        lambda records: updates.append(dict(records[-1])),
    )
    first = mod.evaluate_block(*call)
    second = mod.evaluate_block(*call)
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "clean")
    monkeypatch.setattr(
        mod,
        "score_null_draw",
        lambda *args, **kwargs: {
            "status": "computed",
            "temporal_auc": matrix(args[-2]),
        },
    )
    clean = mod.evaluate_block(*call)
    assert first[0]["status"] == "failed_to_score"
    assert second[0]["status"] == "computed"
    assert second[0]["null_checkpoint_hits"] == 1
    assert len(calls) == 4
    assert any(
        update.get("status") == "running"
        and update.get("completed_permutations") == 0
        for update in updates
    )
    np.testing.assert_array_equal(second[0]["temporal_auc"], clean[0]["temporal_auc"])
    np.testing.assert_array_equal(
        second[0]["temporal_null_mean"], clean[0]["temporal_null_mean"]
    )
    assert second[0]["same_time"] == clean[0]["same_time"]
    assert second[0]["cross_temporal"] == clean[0]["cross_temporal"]


def test_smoke_preset_is_bounded():
    args = SimpleNamespace(
        smoke=True,
        candidates=list(CANDIDATES),
        decoders=list(DECODERS),
        single_item_sessions_limit=None,
        multi_object_sessions_limit=None,
        multi_object_levels_limit=None,
        n_splits=5,
        n_perm=100,
        time_step=2,
    )
    mod.apply_smoke(args)
    assert args.candidates == ["native_full_rank", "principal_components"]
    assert args.decoders == ["linear", "nonlinear"]
    assert args.n_splits == 2
    assert args.n_perm == 2
    assert args.time_step == 10
    assert args.single_item_sessions_limit == 1
    assert args.multi_object_sessions_limit == 1
    assert args.multi_object_levels_limit == 1


def test_limited_loaders_stop_after_requested_sessions(monkeypatch, tmp_path):
    paths = [tmp_path / "a.mat", tmp_path / "b.mat"]
    monkeypatch.setattr(
        mod,
        "_reachable_sessions",
        lambda root, limit: paths[:limit],
    )
    single_calls = []
    monkeypatch.setattr(
        mod,
        "_macaque_session_bundle",
        lambda path: single_calls.append(path) or {"session": path.stem},
    )
    assert [bundle["session"] for bundle in mod.load_single(tmp_path, 1)] == ["a"]
    assert single_calls == paths[:1]

    yielded = []
    monkeypatch.setattr(
        mod,
        "_observable_arrays",
        lambda counts, session: ({}, {}, np.array([True])),
    )
    monkeypatch.setattr(
        mod,
        "iter_watters",
        lambda root, bin_ms: (
            yielded.append(name) or {
                "status": "loaded",
                "session": name,
                "counts": np.zeros((1, 1, 1)),
            }
            for name in ("a", "b", "c")
        ),
    )
    monkeypatch.setattr(mod, "_watters_bundles", lambda arrays: list(arrays.values()))
    multi, arrays = mod.load_multi(tmp_path, 1)
    assert len(multi) == 1
    assert list(arrays) == ["a"]
    assert yielded == ["a"]


def test_nonpositive_limits_are_rejected(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_info_benchmark.py", "--single-item-sessions-limit", "-1"],
    )
    with pytest.raises(SystemExit, match="limits must be positive"):
        mod.main()


def test_native_candidate_runs_end_to_end_on_synthetic_data(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path)
    rng = np.random.default_rng(21)
    labels = np.repeat(np.arange(3), 8)
    signal = np.eye(3)[labels]
    activity = np.stack((signal, signal), axis=1)
    activity += rng.normal(scale=0.2, size=activity.shape)
    records = mod.evaluate_block(
        "c", "s", "all", activity, labels,
        ("native_full_rank",), ("linear", "nonlinear"),
        2, 2, 1, lambda records: None,
    )
    assert [record["status"] for record in records] == ["computed", "computed"]
    assert all(record["same_time"]["auc"] > 0.9 for record in records)


def _main_paths(monkeypatch, tmp_path, corpus):
    output = tmp_path / "result.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(mod, "data_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_info_benchmark.py",
            "--corpora",
            corpus,
            "--output",
            str(output),
            "--summary",
            str(summary),
        ],
    )
    return output, summary


def test_empty_single_source_fails_closed(monkeypatch, tmp_path):
    output, summary = _main_paths(monkeypatch, tmp_path, "single_item")
    monkeypatch.setattr(mod, "_macaque_bundles", lambda root: ([], []))
    with pytest.raises(SystemExit, match="no single-item"):
        mod.main()
    artifact = json.loads(output.read_text())
    assert artifact["status"] == "failed"
    assert artifact["records"][0]["status"] == "source_empty"
    assert "Status: failed" in summary.read_text()


def test_empty_multi_source_fails_closed(monkeypatch, tmp_path):
    output, summary = _main_paths(monkeypatch, tmp_path, "multi_object")
    monkeypatch.setattr(mod, "iter_watters", lambda root, bin_ms: iter(()))
    with pytest.raises(SystemExit, match="no multi-object"):
        mod.main()
    artifact = json.loads(output.read_text())
    assert artifact["status"] == "failed"
    assert artifact["records"][0]["status"] == "source_empty"
    assert "Status: failed" in summary.read_text()


def test_loader_exception_is_written_before_exit(monkeypatch, tmp_path):
    output, summary = _main_paths(monkeypatch, tmp_path, "single_item")

    def fail(root):
        raise FileNotFoundError("missing source file")

    monkeypatch.setattr(mod, "_macaque_bundles", fail)
    with pytest.raises(SystemExit, match="missing source file"):
        mod.main()
    artifact = json.loads(output.read_text())
    assert artifact["status"] == "failed"
    assert artifact["records"][0]["status"] == "source_failed"
    assert "Status: failed" in summary.read_text()
