import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_rank_free_component_identity as analysis
import subspace_identity


def _directions(seed: int, labels: np.ndarray | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 120 if labels is None else len(labels)
    values = rng.normal(0.0, 0.15, (n, 8))
    values[:, 0] += 3.0
    if labels is not None:
        values[:, 1] += 0.5 * labels
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_crossfit_uses_training_trials_for_reference_and_label_basis(monkeypatch):
    directions = np.arange(48, dtype=float).reshape(12, 4) + 1.0
    target = np.arange(12, dtype=float)
    folds = subspace_identity.block_folds(12, 2)
    axis_calls = []
    basis_calls = []

    def axis(reference, evaluation):
        axis_calls.append((reference.copy(), evaluation.copy()))
        return np.array([1.0, 0.0, 0.0, 0.0])

    def basis(train, labels):
        basis_calls.append((train.copy(), labels.copy()))
        return np.array([[1.0], [0.0], [0.0], [0.0]])

    monkeypatch.setattr(subspace_identity, "residual_axis", axis)
    result = subspace_identity.crossfit_alignment(directions, target, folds, basis)

    assert result["status"] == "computed"
    for fold, ((reference, evaluation), (train, labels)) in enumerate(zip(axis_calls, basis_calls)):
        test = folds == fold
        np.testing.assert_array_equal(reference, directions[~test])
        np.testing.assert_array_equal(evaluation, directions[test])
        np.testing.assert_array_equal(train, directions[~test])
        np.testing.assert_array_equal(labels, target[~test])


def test_every_permutation_refits_labels_and_preserves_fold_label_counts(monkeypatch):
    labels = np.tile(np.array([-1.0, 1.0]), 60)
    directions = _directions(2, labels)
    folds = subspace_identity.block_folds(len(labels), 2)
    fitted_labels = []
    axis_calls = []
    original_axis = subspace_identity.residual_axis

    def axis(reference, evaluation):
        axis_calls.append((len(reference), len(evaluation)))
        return original_axis(reference, evaluation)

    def basis(train, target):
        fitted_labels.append(np.sort(target.copy()))
        return subspace_identity.regression_basis(train, target)

    monkeypatch.setattr(subspace_identity, "residual_axis", axis)
    result = subspace_identity.permutation_alignment(
        directions, labels, folds, basis, 7, np.random.default_rng(3),
    )

    assert result["status"] == "computed"
    assert axis_calls == [(60, 60), (60, 60)]
    assert len(fitted_labels) == 2 * (7 + 1)
    expected = np.sort(labels[folds == 0])
    for fitted in fitted_labels:
        np.testing.assert_array_equal(fitted, expected)


def test_planted_label_subspace_is_recovered():
    labels = np.tile(np.array([-1.0, 1.0]), 60)
    result = subspace_identity.permutation_alignment(
        _directions(4, labels),
        labels,
        subspace_identity.block_folds(len(labels), 2),
        subspace_identity.regression_basis,
        199,
        np.random.default_rng(8),
    )

    assert result["alignment"] > 0.9
    assert result["alignment_above_null"] > 0.2
    assert result["p_value"] <= 0.01


def test_independent_labels_do_not_clear_the_refit_null():
    rng = np.random.default_rng(19)
    directions = _directions(9)
    labels = rng.normal(size=len(directions))
    result = subspace_identity.permutation_alignment(
        directions,
        labels,
        subspace_identity.block_folds(len(labels), 2),
        subspace_identity.regression_basis,
        499,
        np.random.default_rng(10),
    )

    assert abs(result["alignment_above_null"]) < 0.1
    assert result["p_value"] > 0.05


def test_pooling_weights_independent_units_not_sessions(monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 50)

    def record(session, unit, alignment):
        draws = np.full(20, 0.1)
        return {
            "session": session,
            "independent_unit": unit,
            "candidates": {
                "memorandum_content": {
                    "status": "computed",
                    "alignment": alignment,
                    "alignment_above_null": alignment - draws.mean(),
                    "null_draws": draws,
                }
            },
        }

    pooled = analysis._pool(
        [
            record("a1", "animal_a", 0.9),
            record("a2", "animal_a", 0.7),
            record("b1", "animal_b", 0.2),
            record("c1", "animal_c", 0.4),
            record("d1", "animal_d", 0.6),
        ],
        "memorandum_content",
        "pool-test",
    )

    assert pooled["n_sessions"] == 5
    assert pooled["n_independent_units"] == 4
    assert pooled["mean_alignment"] == 0.5
    assert pooled["mean_alignment_above_null"] == 0.4


def test_watters_checkpoint_prevents_loader_recomputation(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(analysis, "watters_session_dates", lambda root: [("A", "2020-01-01", "task")])
    calls = {"behaviour": 0, "fit": 0}

    def behaviour(root):
        calls["behaviour"] += 1
        return object()

    def fit(root, loaded_behaviour, animal, date, variant, n_perm):
        calls["fit"] += 1
        return {"session": "A_2020-01-01_task", "independent_unit": "A", "status": "computed", "candidates": {}}

    monkeypatch.setattr(analysis, "watters_behaviour", behaviour)
    monkeypatch.setattr(analysis, "_watters_session_record", fit)
    identity = {"version": "test", "data_root": str(tmp_path), "n_perm": 2}

    first = analysis._collect_watters(tmp_path, 2, identity)
    second = analysis._collect_watters(tmp_path, 2, identity)

    assert first == second
    assert calls == {"behaviour": 1, "fit": 1}


def test_level_combination_handles_unequal_valid_null_counts():
    cell_a = {
        "status": "computed", "alignment": 0.8, "null_draws": np.array([0.1, 0.2, 0.3]),
        "target_kind": "continuous", "seed_id": "a",
    }
    cell_b = {
        "status": "computed", "alignment": 0.4, "null_draws": np.array([0.2, 0.4]),
        "target_kind": "continuous", "seed_id": "b",
    }

    combined = analysis._combine_levels([(10, cell_a), (30, cell_b)])

    assert combined["n_permutations"] == 2
    assert combined["n_trials"] == 40
    assert combined["seed_ids"] == ["a", "b"]


def test_summary_uses_global_fdr_and_distinguishes_no_computable_tests(monkeypatch):
    monkeypatch.setattr(analysis, "CANDIDATE_KEYS", ("candidate",))
    monkeypatch.setattr(analysis, "CANDIDATE_SUPPORT_MATRIX", {
        "corpus_a": {"candidate": ("present", "")},
        "corpus_b": {"candidate": ("present", "")},
    })

    def pool(records, candidate, seed):
        if not records:
            return {"status": "not_computable"}
        return {
            "status": "computed",
            "p_value": records[0]["p_value"],
            "mean_alignment_above_null": 0.2,
        }

    monkeypatch.setattr(analysis, "_pool", pool)
    empty = analysis._summarize({"corpus_a": [], "corpus_b": []})
    summary = analysis._summarize({"corpus_a": [{"p_value": 0.03}], "corpus_b": [{"p_value": 0.2}]})

    assert empty["branch"] == "no_candidate_corpus_cell_was_computable"
    assert summary["per_corpus"]["corpus_a"]["candidate"]["fdr_q_value"] == 0.06
    assert summary["aligned_anywhere"] == {}
    assert summary["fdr_scope"] == "all_computed_candidate_by_corpus_cells"
