"""Scientific regression tests for confined-drift estimators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_dynamics import (
    compare_temporal_dependence_models,
    confinement_identifiability,
    fit_gaussian_state_space,
    fit_heteroscedastic_gaussian_state_space,
    fit_ou_moments,
    fit_rotation_drift_comparison,
    fit_switching_ar_hmm,
    gaussian_state_space_conditional_log_likelihood,
    gaussian_state_space_log_likelihood,
    heteroscedastic_gaussian_state_space_conditional_log_likelihood,
    leave_one_out_condition_residuals,
    neighbouring_trial_prediction_advantage,
    preceding_same_condition_trial_residuals,
    planted_rotation_recovery,
    simulate_confined_diffusion,
    simulate_switching_ar_hmm,
    switching_fit_decomposition,
    switching_ar_hmm_log_likelihood,
)


def test_leave_one_out_centroid_never_contains_measured_trial():
    y = np.array([[0.0, 1.0], [2.0, 3.0], [10.0, 11.0]])
    residuals, centroids = leave_one_out_condition_residuals(y, np.array([0, 0, 1]))
    np.testing.assert_allclose(centroids[0], y[1])
    np.testing.assert_allclose(centroids[1], y[0])
    np.testing.assert_allclose(residuals[0], y[0] - y[1])
    assert np.isnan(residuals[2]).all()


def test_identifiability_reports_short_delay_and_unconfined_cases():
    assert confinement_identifiability(1.0, 0.05, 4.0).status == "identifiable"
    assert confinement_identifiability(1.0, 0.05, 0.5).status == "not_identifiable"
    assert confinement_identifiability(-0.1, 0.05, 4.0).status == "unconfined"


def test_gaussian_state_space_separates_process_and_observation_noise():
    _, observations = simulate_confined_diffusion(
        180, 80, 0.05, lambda_rate=1.2, diffusion=0.35,
        equilibrium=0.25, observation_sd=0.3, initial_sd=0.1,
        rng=np.random.default_rng(12),
    )
    estimate = fit_gaussian_state_space(observations, dt=0.05)
    assert estimate.converged
    assert estimate.lambda_rate == pytest.approx(1.2, rel=0.35)
    assert estimate.diffusion == pytest.approx(0.35, rel=0.4)
    assert estimate.equilibrium == pytest.approx(0.25, abs=0.15)
    assert estimate.observation_variance == pytest.approx(0.3**2, rel=0.5)
    assert estimate.process_variance is not None and estimate.process_variance > 0
    assert np.isfinite(gaussian_state_space_log_likelihood(observations[:30], estimate, 0.05))
    conditional = gaussian_state_space_conditional_log_likelihood(observations[:30], estimate, 0.05)
    assert np.isfinite(conditional)


def test_conditional_state_space_score_omits_first_observation_density():
    _, observations = simulate_confined_diffusion(
        100, 30, 0.05, lambda_rate=1.0, diffusion=0.2,
        observation_sd=0.25, rng=np.random.default_rng(14),
    )
    estimate = fit_gaussian_state_space(observations, dt=0.05)
    changed = observations.copy()
    changed[:, 0] += 0.2
    original_score = gaussian_state_space_conditional_log_likelihood(observations, estimate, 0.05)
    changed_score = gaussian_state_space_conditional_log_likelihood(changed, estimate, 0.05)
    assert np.isfinite(original_score)
    assert np.isfinite(changed_score)
    assert original_score != changed_score


def test_moment_fit_excludes_lag_zero_and_recovers_confinement():
    _, observations = simulate_confined_diffusion(
        300, 90, 0.05, lambda_rate=1.0, diffusion=0.25,
        observation_sd=0.25, initial_sd=0.05,
        rng=np.random.default_rng(21),
    )
    estimate = fit_ou_moments(observations, dt=0.05, max_lag=15)
    assert estimate.converged
    assert estimate.diagnostics["lag_zero_excluded"] is True
    assert estimate.lambda_rate == pytest.approx(1.0, rel=0.35)
    assert estimate.diffusion == pytest.approx(0.25, rel=0.4)
    assert estimate.observation_variance == pytest.approx(0.25**2, rel=0.6)


def test_constant_observations_are_explicitly_not_estimable():
    estimate = fit_gaussian_state_space(np.ones((10, 20)), dt=0.05)
    assert estimate.status == "not_estimable"
    assert estimate.reason
    assert estimate.lambda_rate is None


def test_switching_ar_hmm_recovers_persistent_regimes_and_beats_one_state():
    rng = np.random.default_rng(44)
    observations = np.zeros((120, 30))
    for trial in observations:
        state = 0
        for time_index in range(len(trial) - 1):
            if rng.random() < 0.08:
                state = 1 - state
            trial[time_index + 1] = (
                (0.92 if state == 0 else 0.15) * trial[time_index]
                + (0.5 if state == 0 else -0.5)
                + rng.normal(scale=0.18)
            )
    one = fit_switching_ar_hmm(
        observations[:80], n_states=1, n_restarts=2, rng=np.random.default_rng(1),
    )
    two = fit_switching_ar_hmm(
        observations[:80], n_states=2, n_restarts=3, rng=np.random.default_rng(2),
    )
    assert two["status"] == "complete"
    assert min(np.diag(two["transition_probability"])) > 0.8
    assert switching_ar_hmm_log_likelihood(observations[80:], two) > (
        switching_ar_hmm_log_likelihood(observations[80:], one) + 100.0
    )


def test_switching_ar_hmm_reports_small_samples_as_not_estimable():
    estimate = fit_switching_ar_hmm(np.ones((2, 3)), n_states=2)
    assert estimate["status"] == "not_estimable"
    assert estimate["reason"]


def test_tied_switching_m_step_returns_one_variance_on_homoscedastic_data():
    rng = np.random.default_rng(71)
    observations = np.zeros((100, 24))
    for trial in observations:
        state = int(rng.integers(0, 2))
        for time_index in range(1, len(trial)):
            if rng.random() < 0.1:
                state = 1 - state
            trial[time_index] = (
                (0.85 if state == 0 else 0.2) * trial[time_index - 1]
                + rng.normal(scale=0.3)
            )
    estimate = fit_switching_ar_hmm(
        observations, tie_variances=True, n_restarts=2,
        rng=np.random.default_rng(72),
    )
    assert estimate["status"] in {"complete", "nonconverged"}
    assert estimate["tie_variances"] is True
    np.testing.assert_allclose(estimate["variances"], estimate["variances"][0], rtol=0, atol=1e-12)
    decomposition = switching_fit_decomposition(estimate)
    assert decomposition["absolute_log_variance_separation"] == pytest.approx(0.0, abs=1e-12)


def test_heteroscedastic_drift_scores_heavy_tailed_observation_noise():
    rng = np.random.default_rng(81)
    _, observations = simulate_confined_diffusion(
        90, 28, 0.05, lambda_rate=1.0, diffusion=0.25,
        observation_sd=0.12, rng=rng,
    )
    large_noise = rng.random(observations.shape) < 0.12
    observations[large_noise] += rng.normal(scale=0.8, size=int(large_noise.sum()))
    train, test = observations[:65], observations[65:]
    estimate = fit_heteroscedastic_gaussian_state_space(train, 0.05, n_restarts=2)
    assert estimate["status"] == "complete"
    assert estimate["observation_variances"][1] > estimate["observation_variances"][0]
    assert np.isfinite(heteroscedastic_gaussian_state_space_conditional_log_likelihood(test, estimate))


def test_parametric_null_has_near_zero_median_switching_advantage():
    rng = np.random.default_rng(91)
    advantages = []
    for repetition in range(8):
        _, observations = simulate_confined_diffusion(
            52, 18, 0.08, lambda_rate=1.1, diffusion=0.3,
            observation_sd=0.25, rng=rng,
        )
        train, test = observations[:40], observations[40:]
        drift = fit_gaussian_state_space(train, 0.08)
        switching = fit_switching_ar_hmm(
            train, n_states=2, n_iter=45, n_restarts=1,
            rng=np.random.default_rng(100 + repetition),
        )
        denominator = test[:, 1:].size
        advantages.append((
            switching_ar_hmm_log_likelihood(test, switching)
            - gaussian_state_space_conditional_log_likelihood(test, drift, 0.08)
        ) / denominator)
    assert abs(float(np.nanmedian(advantages))) < 0.12


def test_switching_simulator_preserves_requested_shape():
    planted = {
        "status": "complete",
        "initial_probability": [0.5, 0.5],
        "transition_probability": [[0.9, 0.1], [0.2, 0.8]],
        "coefficients": [0.8, 0.2],
        "intercepts": [0.1, -0.1],
        "variances": [0.2, 0.8],
    }
    observations = simulate_switching_ar_hmm(17, 13, planted, np.random.default_rng(92))
    assert observations.shape == (17, 13)
    assert np.all(np.isfinite(observations))


def test_rotation_comparison_recovers_planted_axis_motion():
    rng = np.random.default_rng(42)
    n_train, n_test, n_time = 90, 45, 12
    train_labels = np.repeat(np.arange(3), n_train // 3)
    test_labels = np.repeat(np.arange(3), n_test // 3)
    angles = np.linspace(0.0, 1.0, n_time)
    axes = np.column_stack([np.cos(angles), np.sin(angles)])

    def sample(labels):
        signal = (labels - 1.0)[:, None, None] * axes[None, :, :]
        return signal + rng.normal(scale=0.28, size=signal.shape)

    train = sample(train_labels)
    test = sample(test_labels)
    direction = axes.mean(axis=0)
    direction /= np.linalg.norm(direction)
    train_projection = train @ direction
    test_projection = test @ direction
    train_residual, _ = leave_one_out_condition_residuals(
        train_projection, train_labels
    )
    test_residual = np.empty_like(test_projection)
    for label in np.unique(test_labels):
        test_residual[test_labels == label] = (
            test_projection[test_labels == label]
            - np.mean(train_projection[train_labels == label], axis=0)
        )
    result = fit_rotation_drift_comparison(
        train, test, train_labels, test_labels, direction,
        train_residual, test_residual, -2.0, 0.05, rng=rng,
    )
    assert result["status"] == "complete"
    assert result["counter_rotation"]["accuracy_recovery"] > 0.0
    assert result["mean_axis_rotation_rate_radians_per_second"] > 0.5


def test_matched_flexibility_controls_do_not_create_dynamics_under_heteroscedastic_noise():
    rng = np.random.default_rng(20260802)
    observations = rng.normal(size=(120, 20))
    high_variance = rng.random(observations.shape) < 0.18
    observations[high_variance] *= 4.0
    comparison = compare_temporal_dependence_models(observations[:90], observations[90:], 0.1)
    assert comparison["m2_minus_heteroscedastic_m0_nats_per_transition"] < 0.08
    assert comparison["m2_minus_free_variance_ar1_m0_nats_per_transition"] < 0.08


def test_neighbour_predictor_separates_session_trend_from_within_trial_diffusion():
    rng = np.random.default_rng(20260803)
    n_trials, n_time = 120, 18
    labels = np.arange(n_trials) % 4
    train = np.arange(n_trials) % 5 != 0
    test = ~train
    session_level = np.linspace(-2.0, 2.0, n_trials)[:, None]
    trend_data = session_level + rng.normal(scale=0.08, size=(n_trials, n_time))
    trend_result = neighbouring_trial_prediction_advantage(
        trend_data, labels, np.flatnonzero(train), np.flatnonzero(test)
    )
    assert abs(trend_result["own_minus_neighbour_r2_advantage"]) < 0.08

    diffusion = np.zeros((n_trials, n_time))
    for time_index in range(1, n_time):
        diffusion[:, time_index] = (
            0.88 * diffusion[:, time_index - 1]
            + rng.normal(scale=0.35, size=n_trials)
        )
    diffusion_result = neighbouring_trial_prediction_advantage(
        diffusion, labels, np.flatnonzero(train), np.flatnonzero(test)
    )
    assert diffusion_result["own_minus_neighbour_r2_advantage"] > 0.25


def test_neighbour_predictor_does_not_feed_held_out_residuals_into_training():
    residuals = np.repeat(np.arange(5, dtype=float)[:, None], 3, axis=1)
    train = np.array([0, 2, 3])
    test = np.array([1, 4])

    neighbours = preceding_same_condition_trial_residuals(
        residuals, np.zeros(5, dtype=int), train, test
    )

    np.testing.assert_allclose(neighbours[2], residuals[0])
    np.testing.assert_allclose(neighbours[3], residuals[2])
    np.testing.assert_allclose(neighbours[1], residuals[0])
    np.testing.assert_allclose(neighbours[4], residuals[3])


def test_switching_variance_floor_excludes_near_duplicate_collapse():
    sanity_bound_nats_per_transition = 100.0
    rng = np.random.default_rng(20260805)
    observations = rng.normal(size=(30, 15))
    observations[0] = np.linspace(0.0, 1.0, 15)
    observations[1] = observations[0] + 1e-10
    estimate = fit_switching_ar_hmm(observations, n_restarts=5, rng=rng)
    assert estimate["status"] == "non_identified"
    assert estimate["variance_floor_hit"] is True
    held_out_score = (
        switching_ar_hmm_log_likelihood(observations, estimate)
        / observations[:, 1:].size
    )
    assert np.isnan(held_out_score)
    assert not (np.isfinite(held_out_score) and held_out_score > sanity_bound_nats_per_transition)


def test_planted_rotation_positive_control_detects_large_rotation():
    result = planted_rotation_recovery(
        120, 60, 24, 8, 3, 0.1, residual_snr=0.8, rotation_rate=2.0,
        rng=np.random.default_rng(20260806),
    )
    assert result["counter_rotation_accuracy_recovery"] > 0.05
