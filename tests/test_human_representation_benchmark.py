from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_human_representation_benchmark as mod  # noqa: E402


def test_demixed_candidate_recovers_separable_categories():
    rng = np.random.default_rng(0)
    n_per_cat, n_cat, units, bins = 12, 5, 10, 4
    category = np.repeat(np.arange(1, n_cat + 1), n_per_cat)
    basis = rng.standard_normal((units, 3))
    activity = rng.poisson(2.0, size=(len(category), bins, units)).astype(float)
    for i, c in enumerate(category):
        activity[i] += (c * 2.0) * basis[:, 0]
    train_idx = np.arange(0, len(category), 2)
    test_idx = np.arange(1, len(category), 2)
    fit = mod.fit_demixed_principal_components(
        activity[train_idx], category[train_idx], activity[test_idx], rank=3, rng=np.random.default_rng(1)
    )
    assert fit["status"] == "fitted"
    assert fit["k_used"] >= 1
    assert fit["latent_train"].shape == (len(train_idx), bins, fit["k_used"])
    assert fit["latent_test"].shape == (len(test_idx), bins, fit["k_used"])


def test_demixed_candidate_rejects_single_class_fold():
    activity = np.zeros((4, 2, 3))
    labels = np.array([1, 1, 1, 1])
    fit = mod.fit_demixed_principal_components(activity, labels, activity, rank=2, rng=np.random.default_rng(0))
    assert fit["status"] == "failed_to_train"


def _computed_record(corpus, session, level, decoder, candidate, above_null):
    return {
        "status": "computed", "corpus": corpus, "session": session, "level": level,
        "decoder": decoder, "candidate": candidate,
        "same_time": {"status": "computed", "auc_above_null": above_null},
        "cross_temporal": {"status": "computed", "auc_above_null": above_null},
    }


def test_improvement_summary_flags_positive_lower_bound():
    records, patients = [], {}
    for i, (session, patient, diff) in enumerate(
        [("s1", "p1", 0.30), ("s2", "p1", 0.28), ("s3", "p2", 0.32), ("s4", "p3", 0.31)]
    ):
        records.append(_computed_record("hippocampus", session, "load1_maintenance", "linear", "principal_components", 0.10))
        records.append(_computed_record("hippocampus", session, "load1_maintenance", "linear", "native_full_rank", 0.10 + diff))
        patients[session] = patient
    summary = mod.improvement_summary(records, patients, np.random.default_rng(0))
    row = next(
        r for r in summary
        if r["candidate"] == "native_full_rank" and r["reference"] == "principal_components" and r["metric"] == "same_time"
    )
    assert row["status"] == "estimable"
    assert row["n_patients"] == 3
    assert row["improves_on_reference"] == 1
    assert row["ci_95_patient_cluster_bootstrap"][0] > 0.0


def test_improvement_summary_marks_sparse_cells_not_estimable():
    records = [
        _computed_record("hippocampus", "s1", "load1_maintenance", "linear", "principal_components", 0.1),
        _computed_record("hippocampus", "s1", "load1_maintenance", "linear", "native_full_rank", 0.2),
    ]
    summary = mod.improvement_summary(records, {"s1": "p1"}, np.random.default_rng(0))
    row = next(
        r for r in summary
        if r["candidate"] == "native_full_rank" and r["reference"] == "principal_components" and r["metric"] == "same_time"
    )
    assert row["status"] == "not_estimable"


def test_build_manifest_excludes_000469_patients_already_canonical(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mod, "canonical_sessions",
        lambda: [{"primary_release": "001187", "patient": "P1", "session": "ses-1", "primary_path": "001187/x.nwb"}],
    )
    monkeypatch.setattr(mod, "data_root", lambda: tmp_path)
    p1 = tmp_path / "p1.nwb"
    p2 = tmp_path / "p2.nwb"
    monkeypatch.setattr(
        mod, "REGION_CORPUS_SPECS", {"dandi_000469": {"glob": lambda: [p1, p2]}},
    )
    monkeypatch.setattr(mod, "_canonical_patient", lambda release, path: {"p1.nwb": "P1", "p2.nwb": "P2"}[path.name])
    primary, supplement = mod.build_manifest()
    assert len(primary) == 1
    assert [row["patient"] for row in supplement] == ["P2"]
