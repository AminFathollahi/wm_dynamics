import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from effect_size import (
    minimum_detectable_correlation,
    poolable,
    random_effects_meta_analysis,
    within_cluster_partial_correlation,
)
from statistics import Z_80_POWER, bootstrap_ci


def simulate(rng, n_clusters=12, sessions_per_cluster=2, trials=25, within_slope=0.4, confound=3.0):
    rows = {key: [] for key in ("x", "y", "session", "cluster")}
    for c in range(n_clusters):
        for s in range(sessions_per_cluster):
            offset = rng.normal(0, 1)
            x = rng.normal(0, 1, trials)
            y = within_slope * x + np.sqrt(1 - within_slope**2) * rng.normal(0, 1, trials)
            rows["x"].append(x + confound * offset)
            rows["y"].append(y + confound * offset)
            rows["session"].append(np.full(trials, f"{c}-{s}"))
            rows["cluster"].append(np.full(trials, c))
    return {key: np.concatenate(value) for key, value in rows.items()}


def test_planted_within_session_correlation_is_recovered_and_between_session_confound_removed(rng):
    data = simulate(rng)
    naive = np.corrcoef(data["x"], data["y"])[0, 1]
    result = within_cluster_partial_correlation(
        data["x"], data["y"], data["session"], data["cluster"], n_boot=200, seed=1
    )
    assert naive > 0.8
    assert result["status"] == "estimable"
    assert abs(result["r"] - 0.4) < 0.15
    low, high = result["ci_95_r"]
    assert low < 0.4 < high
    assert result["n_clusters"] == 12 and result["n_sessions"] == 24 and result["n_trials"] == 600


def test_covariate_removal(rng):
    n = 600
    session = np.repeat(np.arange(12), 50)
    cluster = np.repeat(np.arange(6), 100)
    covariate = rng.normal(size=n)
    x = covariate + 0.3 * rng.normal(size=n)
    y = covariate + 0.3 * rng.normal(size=n)
    raw = within_cluster_partial_correlation(x, y, session, cluster, n_boot=100, seed=1)
    adjusted = within_cluster_partial_correlation(x, y, session, cluster, covariates=covariate, n_boot=100, seed=1)
    assert raw["r"] > 0.8
    assert abs(adjusted["r"]) < 0.1


def test_cluster_bootstrap_is_wider_than_trial_bootstrap_when_clusters_differ(rng):
    n_clusters, trials = 10, 30
    x, y, session, cluster = [], [], [], []
    for c in range(n_clusters):
        slope = 0.9 if c % 2 == 0 else -0.5
        xc = rng.normal(size=trials)
        x.append(xc)
        y.append(slope * xc + rng.normal(size=trials))
        session.append(np.full(trials, c))
        cluster.append(np.full(trials, c))
    x, y, session, cluster = (np.concatenate(v) for v in (x, y, session, cluster))
    clustered = within_cluster_partial_correlation(x, y, session, cluster, n_boot=200, seed=2)
    demeaned_x = x - np.repeat([x[session == s].mean() for s in range(n_clusters)], trials)
    demeaned_y = y - np.repeat([y[session == s].mean() for s in range(n_clusters)], trials)
    stacked = np.column_stack([demeaned_x, demeaned_y])
    _, trial_low, trial_high = bootstrap_ci(
        stacked, lambda d: np.corrcoef(d[:, 0], d[:, 1])[0, 1], n_boot=200, rng=np.random.default_rng(2)
    )
    assert clustered["ci_95_r"][1] - clustered["ci_95_r"][0] > 2 * (trial_high - trial_low)


def test_hartung_knapp_interval_matches_hand_computation():
    z, se = np.array([0.1, 0.3, 0.5]), np.array([0.1, 0.1, 0.1])
    result = random_effects_meta_analysis(z, se, ["a", "b", "c"])
    tau2 = (8 - 2) / 200
    variance = 25 * 0.08 / (2 * 75)
    half = student_t.ppf(0.975, 2) * np.sqrt(variance)
    assert result["tau2"] == pytest.approx(tau2)
    assert result["pooled_z"] == pytest.approx(0.3)
    assert result["se_z"] == pytest.approx(np.sqrt(variance))
    assert result["ci_95_z"] == pytest.approx([0.3 - half, 0.3 + half])
    assert result["pooled_r"] == pytest.approx(np.tanh(0.3))
    assert result["ci_95_r"] == pytest.approx([np.tanh(0.3 - half), np.tanh(0.3 + half)])
    assert result["i_squared"] == pytest.approx(75.0)
    assert result["degrees_of_freedom"] == 2


def test_prediction_interval_uses_t_with_k_minus_two_degrees_of_freedom():
    z, se = np.array([0.1, 0.3, 0.5]), np.array([0.1, 0.1, 0.1])
    result = random_effects_meta_analysis(z, se, ["a", "b", "c"])
    half = student_t.ppf(0.975, 1) * np.sqrt(0.03 + 25 * 0.08 / 150)
    assert result["prediction_interval_z"] == pytest.approx([0.3 - half, 0.3 + half])
    assert result["prediction_interval_r"] == pytest.approx([np.tanh(0.3 - half), np.tanh(0.3 + half)])


def test_stimulation_links_are_never_poolable():
    base = {"species": "human", "recording_type": "single_units", "task_class": "sternberg"}
    for link in (1, 2, 3, 8, 9, 11, 12):
        assert not poolable([{**base, "link": link}, {**base, "link": link}, {**base, "link": link}])
    for link in (4, 5, 6, 7, 10):
        assert poolable([{**base, "link": link}, {**base, "link": link}])


def test_cells_that_differ_in_species_recording_type_task_class_or_link_are_not_poolable():
    base = {"species": "human", "recording_type": "single_units", "task_class": "sternberg", "link": 6}
    for key, other in (("species", "macaque"), ("recording_type", "scalp_eeg"), ("task_class", "recognition"),
                       ("link", 7)):
        assert not poolable([base, {**base, key: other}])
    assert not poolable([base])


def test_minimum_detectable_correlation_bracket_and_small_n_branches():
    bracket = minimum_detectable_correlation(10, 1000)
    assert bracket["cluster_bound"] == pytest.approx(np.tanh(Z_80_POWER / np.sqrt(7)))
    assert bracket["trial_bound"] == pytest.approx(np.tanh(Z_80_POWER / np.sqrt(997)))
    assert bracket["trial_bound"] < bracket["cluster_bound"]
    assert minimum_detectable_correlation(3)["status"] == "not_estimable"
    assert minimum_detectable_correlation(4)["trial_bound"] is None

    few = within_cluster_partial_correlation(
        np.arange(6.0), np.arange(6.0)[::-1], np.array([0, 0, 1, 1, 1, 0]), np.array([0, 0, 1, 1, 1, 0]), n_boot=10
    )
    assert few["status"] == "not_estimable" and few["n_clusters"] == 2

    two = random_effects_meta_analysis([0.1, 0.2], [0.1, 0.1], ["a", "b"])
    assert two["status"] == "not_estimable" and two["k"] == 2
    assert random_effects_meta_analysis([0.1, 0.2, np.nan], [0.1, 0.1, 0.1], list("abc"))["status"] == "not_estimable"
