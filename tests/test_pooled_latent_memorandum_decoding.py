import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memorandum_decoding as md  # noqa: E402

CLASSES = np.arange(1, 6)
KIND = "maintenance_whole"
N_BINS, N_FOLDS = 4, 5


def identity_fit(train, test, labels, seed):
    return {"status": "fitted", "latent_train": train, "latent_test": test}


def projection_fit(train, test, labels, seed):
    matrix = np.random.default_rng(0).normal(size=(train.shape[2], 3))
    return {"status": "fitted", "latent_train": train @ matrix, "latent_test": test @ matrix}


def principal_component_fit(train, test, labels, seed):
    flat = train.reshape(-1, train.shape[2])
    mean = flat.mean(axis=0)
    axes = np.linalg.svd(flat - mean, full_matrices=False)[2][:3].T
    return {"status": "fitted", "latent_train": (train - mean) @ axes, "latent_test": (test - mean) @ axes}


def make_session(rng, key, patient, per_class=15, n_units=10, strength=1.5):
    labels = np.repeat(CLASSES, per_class)
    class_axes = rng.normal(size=(len(CLASSES), 2))
    loading = rng.normal(size=(2, n_units))
    rate = np.exp(0.2 + strength * class_axes[labels - 1] @ loading / 2)
    counts = rng.poisson(rate[:, None, :], size=(len(labels), N_BINS, n_units)).astype(float)
    fold_ids = md.stratified_fold_ids(labels, N_FOLDS, rng)
    return {"key": key, "patient": patient, "labels": labels, "fold_ids": fold_ids, "counts": counts}


def blocks_for(sessions, representation, fits, keys=None):
    blocks = []
    for s in sessions:
        if keys is not None and s["key"] not in keys:
            continue
        series = [md.fold_latent_series(s["counts"], s["labels"], s["fold_ids"], f, representation, fits, 0)["series"]
                  for f in range(N_FOLDS)]
        blocks.append({"key": s["key"], "copy": 0, "patient": s["patient"], "labels": s["labels"],
                       "fold_ids": s["fold_ids"], "series": series})
    return blocks


def test_session_blocked_assembly_keeps_test_trials_out_of_training_and_one_trial_per_session():
    rng = np.random.default_rng(1)
    sessions = [make_session(rng, k, f"p{k}", per_class=n) for k, n in enumerate((12, 14, 16))]
    for s in sessions:
        marker = (1000 * (s["key"] + 1) + np.arange(len(s["labels"]))).astype(float)
        s["counts"] = np.repeat(marker[:, None, None], 4, axis=2).repeat(N_BINS, axis=1)
    blocks = blocks_for(sessions, "marker", {"marker": identity_fit})
    for f, fold in enumerate(md.session_blocked_folds(blocks, KIND, CLASSES, N_FOLDS, 0, 0)):
        for part, y in (("train", fold["y_train"]), ("test", fold["y_test"])):
            values = fold[part][KIND][:, :, 0]
            for s in sessions:
                own = values[:, 4 * s["key"]:4 * s["key"] + 4]
                assert (own == own[:, :1]).all()
                rows = (own[:, 0] - 1000 * (s["key"] + 1)).astype(int)
                assert (s["labels"][rows] == y).all()
                in_fold = s["fold_ids"][rows] == f
                assert in_fold.all() if part == "test" else not in_fold.any()


def test_planted_signal_is_recovered_from_pooled_principal_components_and_native_counts():
    rng = np.random.default_rng(2)
    sessions = [make_session(rng, k, f"p{k}") for k in range(4)]
    fits = {"principal_components": principal_component_fit}
    for representation in ("native", "principal_components"):
        blocks = blocks_for(sessions, representation, fits)
        out = md.pooled_observed(blocks, KIND, ("linear_svm",), CLASSES, N_FOLDS, 2, 0, 3)
        assert out["scores"]["linear_svm"][KIND]["balanced_accuracy"].mean() > 0.4, representation
        assert out["scores"]["linear_svm"][KIND]["auc"].mean() > 0.7, representation


def test_shuffled_labels_give_chance():
    rng = np.random.default_rng(3)
    sessions = [make_session(rng, k, f"p{k}") for k in range(4)]
    blocks = blocks_for(sessions, "projection", {"projection": projection_fit})
    out = md.pooled_observed(blocks, KIND, ("linear_svm",), CLASSES, N_FOLDS, 1, 60, 4)
    null = out["null"]["linear_svm"][KIND]
    assert abs(null["balanced_accuracy"].mean() - 0.2) < 0.04
    assert abs(null["auc"].mean() - 0.5) < 0.04
    assert out["scores"]["linear_svm"][KIND]["balanced_accuracy"].mean() > 0.4


def test_patient_bootstrap_draws_whole_patients_with_all_their_sessions():
    sessions = [{"key": k, "patient": p, "copy": 0} for k, p in enumerate("AABCCC")]
    sizes = {"A": 2, "B": 1, "C": 3}
    repeated = False
    for seed in range(20):
        drawn = md.patient_bootstrap_blocks(sessions, np.random.default_rng(seed))
        for patient, size in sizes.items():
            mine = [b for b in drawn if b["patient"] == patient]
            copies = sorted({b["copy"] for b in mine})
            assert len(mine) == size * len(copies) and copies == list(range(len(copies)))
            assert all(sorted(b["key"] for b in mine if b["copy"] == c) == [s["key"] for s in sessions
                                                                          if s["patient"] == patient] for c in copies)
            repeated |= len(copies) > 1
    assert repeated


def test_fit_failure_in_one_fold_removes_the_session_from_the_representation_and_its_native_row():
    rng = np.random.default_rng(5)
    sessions = [make_session(rng, k, f"p{k}") for k in range(4)]
    counter = {"calls": 0}

    def fails_once(train, test, labels, seed):
        counter["calls"] += 1
        if counter["calls"] == 13:
            return {"status": "failed_to_train", "reason": "did not converge"}
        return projection_fit(train, test, labels, seed)

    records = {s["key"]: [{k: v for k, v in md.fold_latent_series(
        s["counts"], s["labels"], s["fold_ids"], f, "projection", {"projection": fails_once}, 0).items()
        if k != "series"} for f in range(N_FOLDS)] for s in sessions}
    kept, excluded = md.representation_session_set(records)
    assert kept == [0, 1, 3] and excluded == {2: "did not converge"}
    latent = md.session_blocked_folds(blocks_for(sessions, "projection", {"projection": projection_fit}, kept),
                                      KIND, CLASSES, N_FOLDS, 0, 0)
    native = md.session_blocked_folds(blocks_for(sessions, "native", {}, kept), KIND, CLASSES, N_FOLDS, 0, 0)
    assert all(f["n_sessions"] == 3 for f in latent + native)
    assert all(f["n_units"] == 3 * 3 for f in latent) and all(f["n_units"] == 3 * 10 for f in native)
