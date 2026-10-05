import sys
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memorandum_decoding as md  # noqa: E402

CLASSES = np.arange(1, 6)


def make_entry(rng, n_per_class=20, n_units=15, tuned=0.9, baseline=0.3, per_class_units=3):
    labels = np.repeat(CLASSES, n_per_class)
    rates = np.full((len(CLASSES), n_units), baseline)
    for c in range(len(CLASSES)):
        rates[c, c * per_class_units:(c + 1) * per_class_units] = tuned
    def counts(width):
        return rng.poisson(rates[labels - 1] * width)[:, :, None].astype(float)
    return {"labels": labels, "unit_region": np.array(["hippocampus"] * n_units),
            "features": {"selection_window": counts(1.0), "maintenance_whole": counts(md.MAINTENANCE_S)}}


def test_planted_category_signal_is_recovered_by_every_decoder():
    entry = make_entry(np.random.default_rng(0))
    folds, _ = md.session_folds(entry, "pooled", "all", ("maintenance_whole",), 5, np.random.default_rng(1))
    scores, _ = md.evaluate_folds(folds, ("maintenance_whole",), md.DECODERS, CLASSES)
    for decoder in md.DECODERS:
        result = scores[decoder]["maintenance_whole"]
        assert result["balanced_accuracy"][0] > 0.4, decoder
        assert result["auc"][0] > 0.7, decoder


def test_selection_inside_folds_gives_chance_on_pure_noise_units(monkeypatch):
    monkeypatch.setattr(md, "SELECTION_PERMUTATIONS", 200)
    rng = np.random.default_rng(2)
    labels = np.repeat(CLASSES, 20)
    n_units = 300
    noise = rng.poisson(1.0, (100, n_units, 1)).astype(float)
    entry = {"labels": labels, "unit_region": np.array(["hippocampus"] * n_units),
             "features": {"selection_window": noise, "maintenance_whole": noise.copy()}}
    accuracies = []
    for seed in range(3):
        folds, _ = md.session_folds(entry, "pooled", "selective", ("maintenance_whole",), 5, np.random.default_rng(seed))
        scores, _ = md.evaluate_folds(folds, ("maintenance_whole",), ("linear_svm",), CLASSES)
        accuracies.append(scores["linear_svm"]["maintenance_whole"]["balanced_accuracy"][0])
    assert abs(np.mean(accuracies) - 0.2) < 0.06

    window = entry["features"]["selection_window"][:, :, 0]
    f_values = f_oneway(*[window[labels == c] for c in CLASSES]).statistic
    chosen = np.argsort(f_values)[-30:]
    leaked = {**entry, "features": {k: v[:, chosen, :] for k, v in entry["features"].items()},
              "unit_region": entry["unit_region"][chosen]}
    folds, _ = md.session_folds(leaked, "pooled", "all", ("maintenance_whole",), 5, np.random.default_rng(0))
    scores, _ = md.evaluate_folds(folds, ("maintenance_whole",), ("linear_svm",), CLASSES)
    assert scores["linear_svm"]["maintenance_whole"]["balanced_accuracy"][0] > np.mean(accuracies) + 0.1


def test_pseudo_trial_assembly_never_puts_a_test_trial_in_training():
    rng = np.random.default_rng(4)
    entries, label_of = [], {}
    for e, per_class in enumerate((12, 14, 16, 4)):
        labels = np.repeat(CLASSES, per_class)
        marker = (1000 * (e + 1) + np.arange(len(labels))).astype(float)
        label_of.update(zip(marker, labels))
        values = np.repeat(marker[:, None, None], 4, axis=1)
        entries.append({"labels": labels, "unit_region": np.array(["amygdala"] * 4),
                        "features": {"selection_window": values, "maintenance_whole": values}})
    folds = md.pseudo_population_folds(entries, "pooled", "all", ("maintenance_whole",), 5, rng)
    for fold in folds:
        train, test = fold["train"]["maintenance_whole"], fold["test"]["maintenance_whole"]
        assert train.shape[0] == 5 * md.PSEUDO_TRAIN_PER_CATEGORY and test.shape[0] == 5 * md.PSEUDO_TEST_PER_CATEGORY
        assert fold["n_units"] == 4 * (4 - (fold["dropped_units"]["too_few_test_trials"] > 0))
        for unit in range(train.shape[1]):
            assert not set(train[:, unit, 0]) & set(test[:, unit, 0])
        for part, y in (("train", fold["y_train"]), ("test", fold["y_test"])):
            values = fold[part]["maintenance_whole"][:, :, 0]
            assert all(label_of[v] == label for row, label in zip(values, y) for v in row)
    assert sum(fold["dropped_units"]["too_few_test_trials"] for fold in folds) > 0


def test_shuffled_labels_give_chance():
    entry = make_entry(np.random.default_rng(5))
    classes = np.unique(entry["labels"])
    decoders = ("linear_svm", "poisson_naive_bayes")
    kinds = ("maintenance_whole",)
    observed = md.observed_runs(md.session_make_folds(entry, "pooled", "all", kinds, 5, False), kinds, decoders,
                                classes, 1, 0)
    shuffled = md.shuffled_runs(md.session_make_folds(entry, "pooled", "all", kinds, 5, True), kinds, decoders,
                                classes, observed["tuned"], range(30), 0)
    for decoder in decoders:
        assert abs(shuffled[decoder]["maintenance_whole"]["balanced_accuracy"].mean() - 0.2) < 0.04
        assert abs(shuffled[decoder]["maintenance_whole"]["auc"].mean() - 0.5) < 0.04
        assert observed["scores"][decoder]["maintenance_whole"]["balanced_accuracy"].mean() > 0.4


def test_hyperparameter_choice_never_reads_outer_test_trials(monkeypatch):
    rng = np.random.default_rng(6)
    entry = make_entry(rng)
    n = len(entry["labels"])
    for key in entry["features"]:
        entry["features"][key][:, 0, 0] = 10_000 + np.arange(n)
    folds, _ = md.session_folds(entry, "pooled", "all", ("maintenance_whole",), 5, np.random.default_rng(1))
    seen = []
    original = md.choose_hyperparameter

    def recording(decoder, train, labels, exposure, classes, seed):
        seen.append((train[:, 0].copy(), labels.copy()))
        return original(decoder, train, labels, exposure, classes, seed)

    monkeypatch.setattr(md, "choose_hyperparameter", recording)
    md.evaluate_folds(folds, ("maintenance_whole",), ("linear_svm", "radial_svm"), CLASSES)
    assert len(seen) == 2 * len(folds)
    for call, fold in zip(seen, folds + folds):
        outer_test = set(fold["test"]["maintenance_whole"][:, 0, 0])
        assert not outer_test & set(call[0])
        assert len(call[0]) == len(fold["y_train"])


def test_window_counts_and_planted_unit_selection():
    spikes = [np.array([0.0, 0.125, 0.25, 0.375])]
    counts = md.window_counts(spikes, np.array([0.0]), np.array([0.0, 0.125]), 0.25)
    assert counts[0, 0].tolist() == [2, 2]

    rng = np.random.default_rng(7)
    labels = np.repeat(CLASSES, 20)
    rates = rng.poisson(1.0, (100, 30)).astype(float)
    rates[:, 0] = rng.poisson(np.where(labels == 3, 6.0, 1.0))
    selected = md.category_selective_mask(rates, labels, np.random.default_rng(8), n_perm=500)
    assert selected[0] and selected[1:].sum() <= 4
