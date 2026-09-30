import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import data_integrity
import run_rank_free_component_identity as analysis
import subspace_identity
from info_decoding import _combine_levels


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

    combined = _combine_levels([(10, cell_a), (30, cell_b)])

    assert combined["n_permutations"] == 2
    assert combined["n_trials"] == 40
    assert combined["seed_ids"] == ["a", "b"]


def test_completeness_gate_catches_missing_companion_binaries_and_partial_downloads(tmp_path, monkeypatch):
    spikes_dir = tmp_path / "spikes"
    behavior_dir = tmp_path / "behavior"
    monkeypatch.setattr(data_integrity, "watters_directories", lambda root: (spikes_dir, behavior_dir))
    monkeypatch.setattr(data_integrity, "watters_session_dates", lambda root: [("A", "2020-01-01", "ring")])
    tier_dir = spikes_dir / "A" / "2020-01-01" / "probe0" / "good"
    tier_dir.mkdir(parents=True)
    (tier_dir / "unit1_trials.pkl").write_bytes(b"x")
    (tier_dir / "unit1_spike_counts.pkl").write_bytes(b"x")
    (tier_dir / "unit2_trials.pkl").write_bytes(b"x")

    vhdr_dir = tmp_path / "ds006848" / "sub-01" / "eeg"
    vhdr_dir.mkdir(parents=True)
    (vhdr_dir / "sub-01_task-verbalwm_eeg.vhdr").write_bytes(b"x")
    (vhdr_dir / "sub-01_task-verbalwm_eeg.vmrk").write_bytes(b"x")

    staging = tmp_path / "000004" / "sub-p1" / "sub-p1_ses-a.nwb.dandidownload"
    staging.mkdir(parents=True)

    missing = data_integrity._completeness_gate(tmp_path)

    missing_paths = {path for _corpus, path in missing}
    assert any(path.endswith("unit2_spike_counts.pkl") for path in missing_paths)
    assert not any(path.endswith("unit1_spike_counts.pkl") for path in missing_paths)
    assert any("sub-01_task-verbalwm_eeg.eeg" in path for path in missing_paths)
    assert not any("sub-01_task-verbalwm_eeg.vmrk" in path for path in missing_paths)
    assert any("dandidownload" in path for path in missing_paths)
    assert {corpus for corpus, _path in missing} == {
        "watters_2026_macaque_multi_object", "ds006848_human_scalp", "dandi_000004_human",
    }


def test_dataset_keys_are_all_registered_with_missing_files():
    for key in analysis.DATASET_KEYS:
        assert key in data_integrity.CORPUS_CHECKS


def test_dandi_000004_region_sessions_filters_by_structure_and_relabels_pooled(monkeypatch):
    fake_entries = [
        {"structure": "hippocampus", "patient": "sub-a", "session": "s1", "memorandum_content": np.array([1.0])},
        {"structure": "amygdala", "patient": "sub-a", "session": "s1", "memorandum_content": np.array([2.0])},
    ]
    monkeypatch.setattr(analysis, "iter_dandi_000004", lambda root: iter(fake_entries))

    hippocampus = list(analysis._dandi_000004_region_sessions(None, "hippocampus"))
    amygdala = list(analysis._dandi_000004_region_sessions(None, "amygdala"))

    assert len(hippocampus) == 1 and hippocampus[0]["structure"] == "pooled"
    assert hippocampus[0]["item_ids"] == fake_entries[0]["memorandum_content"]
    assert len(amygdala) == 1 and amygdala[0]["structure"] == "pooled"
    assert amygdala[0]["item_ids"] == fake_entries[1]["memorandum_content"]


def test_dandi_000004_corpora_are_registered_with_full_candidate_support():
    from info_decoding import CANDIDATE_KEYS, CANDIDATE_SUPPORT_MATRIX

    for corpus in ("dandi_000004_human_hippocampus", "dandi_000004_human_amygdala"):
        assert corpus in analysis.GAIN_QUANTITY_BY_CORPUS
        assert analysis.GAIN_QUANTITY_BY_CORPUS[corpus] == "spike counts summed over the delay window"
        assert set(CANDIDATE_SUPPORT_MATRIX[corpus]) == set(CANDIDATE_KEYS)
        assert CANDIDATE_SUPPORT_MATRIX[corpus]["memorandum_content"][0] == "present"
        assert CANDIDATE_SUPPORT_MATRIX[corpus]["gain_total_spike_count"][0] == "present"


def test_pool_reports_minimum_detectable_effect_at_80pct_power(monkeypatch):
    monkeypatch.setattr(analysis, "N_BOOT", 200)

    def record(session, unit, alignment):
        draws = np.full(20, 0.1)
        return {
            "session": session, "independent_unit": unit,
            "candidates": {"memorandum_content": {
                "status": "computed", "alignment": alignment,
                "alignment_above_null": alignment - draws.mean(), "null_draws": draws,
            }},
        }

    pooled = analysis._pool(
        [record("a1", "unit_a", 0.9), record("b1", "unit_b", 0.3),
         record("c1", "unit_c", 0.5), record("d1", "unit_d", 0.4)],
        "memorandum_content", "mdd-test",
    )

    assert "minimum_detectable_alignment_above_null_at_80pct_power" in pooled
    mdd = pooled["minimum_detectable_alignment_above_null_at_80pct_power"]
    assert np.isfinite(mdd) and mdd > 0


def test_pool_carries_unanimous_per_session_reason_to_the_summary(monkeypatch):
    def record(session, unit):
        return {
            "session": session, "independent_unit": unit,
            "candidates": {"time_in_trial_within_delay": {
                "status": "not_computable",
                "reason": "the candidate varies across bins rather than trials and requires a separate blocked time-label null",
            }},
        }

    pooled = analysis._pool(
        [record("a1", "unit_a"), record("b1", "unit_b")],
        "time_in_trial_within_delay", "reason-test",
    )
    assert pooled["status"] == "not_computable"
    assert pooled["reason"] == (
        "the candidate varies across bins rather than trials and requires a separate blocked time-label null"
    )


def test_pool_falls_back_to_the_generic_reason_when_sessions_disagree(monkeypatch):
    def record(session, unit, reason):
        return {
            "session": session, "independent_unit": unit,
            "candidates": {"some_candidate": {"status": "not_computable", "reason": reason}},
        }

    pooled = analysis._pool(
        [record("a1", "unit_a", "reason one"), record("b1", "unit_b", "reason two")],
        "some_candidate", "reason-test-2",
    )
    assert pooled["status"] == "not_computable"
    assert "fewer than" in pooled["reason"]


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
