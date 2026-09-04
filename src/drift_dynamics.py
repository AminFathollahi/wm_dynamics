"""Leakage-safe estimation of confined single-trial working-memory drift.

The estimand is the trial-specific deviation from an out-of-fold
condition-by-time mean.  The primary estimator is a scalar Gaussian state
space model with distinct process and observation variances.  A transparent
autocovariance/variance-function fit estimates the same confinement and
diffusion parameters independently.

This module never smooths observations.  Callers must supply unsmoothed,
non-overlapping bins transformed by a coordinate system fit only on outer
training trials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit, minimize
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass(frozen=True)
class Identifiability:
    """Whether the sampling interval and delay span resolve confinement."""

    status: str
    reason: str | None
    lambda_dt: float
    delay_time_constants: float


@dataclass(frozen=True)
class DriftEstimate:
    """A confined-diffusion estimate with explicit diagnostic status."""

    estimator: str
    status: str
    reason: str | None
    lambda_rate: float | None
    diffusion: float | None
    equilibrium: float | None
    stationary_variance: float | None
    observation_variance: float | None
    process_variance: float | None
    lambda_ci: tuple[float, float] | None
    diffusion_ci: tuple[float, float] | None
    equilibrium_ci: tuple[float, float] | None
    log_likelihood: float | None
    converged: bool
    identifiability: Identifiability | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def leave_one_out_condition_residuals(
    projected_state: NDArray,
    condition: NDArray,
) -> tuple[NDArray, NDArray]:
    """Subtract a condition-by-time mean that excludes the measured trial.

    Parameters
    ----------
    projected_state
        Array shaped ``(trials, time)`` or ``(trials, time, dimensions)``.
    condition
        Condition label for every trial.

    Returns
    -------
    residuals, centroids
        Arrays matching ``projected_state``.  Conditions represented by only
        one trial are returned as NaN because no out-of-fold centroid exists.
    """
    y = np.asarray(projected_state, dtype=float)
    labels = np.asarray(condition)
    if y.ndim not in (2, 3):
        raise ValueError("projected_state must have shape (trials, time[, dimensions])")
    if len(labels) != len(y):
        raise ValueError("condition length must equal the number of trials")
    residuals = np.full_like(y, np.nan, dtype=float)
    centroids = np.full_like(y, np.nan, dtype=float)
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        if len(index) < 2:
            continue
        total = np.sum(y[index], axis=0)
        loo = (total[None, ...] - y[index]) / (len(index) - 1)
        centroids[index] = loo
        residuals[index] = y[index] - loo
    return residuals, centroids


def discriminant_direction(values: NDArray, labels: NDArray) -> NDArray:
    """Dominant multiclass shrinkage-LDA direction with deterministic sign.

    Shared by every per-session confined-drift script that fits a leakage-free
    content/condition axis on outer-training trials only (000469, 000574,
    001187/000673) -- factored here once these had three call sites rather
    than kept as three near-identical copies.
    """
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(values, labels)
    _, _, right = np.linalg.svd(model.coef_, full_matrices=False)
    direction = right[0]
    class_projection = np.array([
        np.mean(values[labels == label] @ direction) for label in model.classes_
    ])
    if np.corrcoef(np.arange(len(class_projection)), class_projection)[0, 1] < 0:
        direction = -direction
    return direction / np.linalg.norm(direction)


def projected_residuals(
    train_state: NDArray,
    test_state: NDArray,
    train_labels: NDArray,
    test_labels: NDArray,
    direction: NDArray,
) -> tuple[NDArray, NDArray]:
    """Leave-one-out train residuals and held-out test residuals along one axis."""
    train_projection = train_state @ direction
    test_projection = test_state @ direction
    train_residuals, _ = leave_one_out_condition_residuals(train_projection, train_labels)
    test_residuals = np.full_like(test_projection, np.nan)
    for label in np.unique(test_labels):
        train_rows = train_labels == label
        test_rows = test_labels == label
        if not np.any(train_rows):
            continue
        test_residuals[test_rows] = test_projection[test_rows] - train_projection[train_rows].mean(axis=0)
    return train_residuals, test_residuals


def matched_complement_direction(
    state_window: NDArray,
    labels: NDArray,
    content_direction: NDArray,
    rng: np.random.Generator,
    n_candidates: int = 200,
) -> tuple[NDArray, float, float]:
    """A random direction orthogonal to ``content_direction``, signal-matched
    to it by between-class variance -- a control for "any direction with this
    much class separation looks confined"."""
    class_means = np.vstack([
        state_window[labels == label].mean(axis=0) for label in np.unique(labels)
    ])
    target_variance = float(np.var(class_means @ content_direction))
    candidates = rng.normal(size=(n_candidates, state_window.shape[1]))
    candidates -= (candidates @ content_direction)[:, None] * content_direction[None, :]
    norms = np.linalg.norm(candidates, axis=1)
    candidates = candidates[norms > 1e-10] / norms[norms > 1e-10, None]
    variances = np.var(class_means @ candidates.T, axis=0)
    selected = int(np.argmin(np.abs(variances - target_variance)))
    return candidates[selected], target_variance, float(variances[selected])


def iid_log_likelihood(test: NDArray, train: NDArray) -> float:
    """M0 baseline: i.i.d. Gaussian with variance fit on training residuals only."""
    variance = max(float(np.nanvar(train)), 1e-10)
    values = test[np.isfinite(test)]
    return float(np.sum(-0.5 * (np.log(2 * np.pi * variance) + values * values / variance)))


def trial_prediction_advantage(train: NDArray, test: NDArray) -> dict:
    """Held-out one-step-ahead linear prediction R2 advantage over the
    zero-drift (no-change) baseline, along the residual trajectory."""
    x_train = train[:, :-1].reshape(-1)
    y_train = train[:, 1:].reshape(-1)
    finite_train = np.isfinite(x_train) & np.isfinite(y_train)
    design = np.column_stack((np.ones(finite_train.sum()), x_train[finite_train]))
    intercept, coefficient = np.linalg.lstsq(design, y_train[finite_train], rcond=None)[0]
    x_test = test[:, :-1].reshape(-1)
    y_test = test[:, 1:].reshape(-1)
    finite_test = np.isfinite(x_test) & np.isfinite(y_test)
    prediction = intercept + coefficient * x_test[finite_test]
    baseline_error = np.sum(y_test[finite_test] ** 2)
    model_error = np.sum((y_test[finite_test] - prediction) ** 2)
    return {
        "coefficient": float(coefficient),
        "held_out_r2_advantage": float(1.0 - model_error / max(baseline_error, 1e-12)),
    }


def confinement_identifiability(
    lambda_rate: float,
    dt: float,
    duration: float,
    max_lambda_dt: float = 0.25,
    min_time_constants: float = 2.0,
) -> Identifiability:
    """Check whether binning and delay length can resolve a leak rate.

    ``lambda * dt`` must be small enough to resolve decay within a bin and
    ``lambda * duration`` must span at least two time constants.  Nonpositive
    rates are reported as unconfined rather than coerced to zero.
    """
    if not np.isfinite(lambda_rate):
        return Identifiability("not_identifiable", "non-finite confinement estimate", np.nan, np.nan)
    lambda_dt = float(lambda_rate * dt)
    time_constants = float(lambda_rate * duration)
    if lambda_rate <= 0:
        return Identifiability(
            "unconfined", "estimated confinement is nonpositive",
            lambda_dt, time_constants,
        )
    if lambda_dt > max_lambda_dt:
        return Identifiability(
            "not_identifiable", "decay is too fast relative to the bin width",
            lambda_dt, time_constants,
        )
    if time_constants < min_time_constants:
        return Identifiability(
            "not_identifiable", "delay spans fewer than two estimated time constants",
            lambda_dt, time_constants,
        )
    return Identifiability("identifiable", None, lambda_dt, time_constants)


def _kalman_negative_log_likelihood(
    unconstrained: NDArray,
    observations: NDArray,
    dt: float,
) -> float:
    log_a, intercept, log_q, log_r, initial_mean, log_initial_variance = unconstrained
    a = float(np.exp(log_a))
    q = float(np.exp(log_q))
    r = float(np.exp(log_r))
    p0 = float(np.exp(log_initial_variance))
    if not (0.005 <= a <= 2.0):
        return 1e30
    nll = 0.0
    for trial in observations:
        mean = initial_mean
        variance = p0
        for value in trial:
            innovation_variance = variance + r
            innovation = value - mean
            nll += 0.5 * (
                np.log(2.0 * np.pi * innovation_variance)
                + innovation * innovation / innovation_variance
            )
            gain = variance / innovation_variance
            mean = mean + gain * innovation
            variance = max((1.0 - gain) * variance, 1e-12)
            mean = a * mean + intercept
            variance = a * a * variance + q
    return float(nll)


def _state_parameters(unconstrained: NDArray, dt: float) -> NDArray:
    log_a, intercept, log_q, log_r, _initial_mean, _log_initial_variance = unconstrained
    a = float(np.exp(log_a))
    q = float(np.exp(log_q))
    r = float(np.exp(log_r))
    lambda_rate = float(-np.log(a) / dt)
    if abs(1.0 - a * a) > 1e-8 and lambda_rate > 0:
        stationary_variance = q / (1.0 - a * a)
        diffusion = lambda_rate * stationary_variance
    else:
        stationary_variance = np.nan
        diffusion = q / (2.0 * dt)
    equilibrium = intercept / (1.0 - a) if abs(1.0 - a) > 1e-6 else np.nan
    return np.array([lambda_rate, diffusion, equilibrium, stationary_variance, r, q])


def _delta_method_intervals(
    optimum: NDArray,
    covariance: NDArray,
    dt: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, tuple[float, float] | None]:
    transformed = _state_parameters(optimum, dt)
    jacobian = np.empty((3, len(optimum)))
    for column in range(len(optimum)):
        step = 1e-5 * max(1.0, abs(float(optimum[column])))
        upper = optimum.copy()
        lower = optimum.copy()
        upper[column] += step
        lower[column] -= step
        jacobian[:, column] = (
            _state_parameters(upper, dt)[:3] - _state_parameters(lower, dt)[:3]
        ) / (2.0 * step)
    transformed_covariance = jacobian @ covariance @ jacobian.T
    intervals: list[tuple[float, float] | None] = []
    for index in range(3):
        variance = transformed_covariance[index, index]
        if not np.isfinite(transformed[index]) or not np.isfinite(variance) or variance < 0:
            intervals.append(None)
            continue
        half_width = 1.96 * np.sqrt(variance)
        intervals.append((float(transformed[index] - half_width), float(transformed[index] + half_width)))
    return intervals[0], intervals[1], intervals[2]


def fit_gaussian_state_space(
    residuals: NDArray,
    dt: float,
) -> DriftEstimate:
    """Maximum-likelihood scalar LGSSM with separate process and observation noise.

    The discrete model is ``x[t+1] = a*x[t] + b + w`` and
    ``y[t] = x[t] + v``.  Confinement is ``-log(a)/dt``; the equilibrium is
    ``b/(1-a)``; process diffusion is obtained from the exact stationary OU
    mapping when confinement is positive.  Likelihood curvature supplies
    approximate 95% intervals and is labelled as such in diagnostics.
    """
    y = np.asarray(residuals, dtype=float)
    if y.ndim != 2 or y.shape[0] < 3 or y.shape[1] < 4:
        return DriftEstimate(
            "gaussian_lgssm", "not_estimable", "need at least 3 trials and 4 time bins",
            None, None, None, None, None, None, None, None, None, None, False, None,
            {"n_trials": int(y.shape[0]) if y.ndim else 0},
        )
    complete = np.all(np.isfinite(y), axis=1)
    y = y[complete]
    if len(y) < 3 or np.var(y) < 1e-12:
        return DriftEstimate(
            "gaussian_lgssm", "not_estimable", "insufficient finite nonconstant trials",
            None, None, None, None, None, None, None, None, None, None, False, None,
            {"n_trials": int(len(y))},
        )
    variance = float(np.var(y))
    lag_product = np.mean((y[:, :-1] - y[:, :-1].mean()) * (y[:, 1:] - y[:, 1:].mean()))
    a_initial = float(np.clip(lag_product / variance, 0.05, 1.2))
    difference_variance = float(np.var(np.diff(y, axis=1)))
    starting_points = []
    for observation_fraction in (0.1, 0.3, 0.6):
        r0 = max(variance * observation_fraction, 1e-8)
        q0 = max(difference_variance - 2.0 * r0, variance * 0.02, 1e-8)
        starting_points.append(np.array([
            np.log(a_initial), 0.0, np.log(q0), np.log(r0),
            float(np.mean(y[:, 0])), np.log(max(np.var(y[:, 0]), 1e-8)),
        ]))
    bounds = [
        (np.log(0.005), np.log(2.0)),
        (-10.0 * np.sqrt(variance), 10.0 * np.sqrt(variance)),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
        (-10.0 * np.sqrt(variance), 10.0 * np.sqrt(variance)),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
    ]
    candidates = [
        minimize(
            _kalman_negative_log_likelihood, start, args=(y, dt),
            method="L-BFGS-B", bounds=bounds,
        )
        for start in starting_points
    ]
    successful = [candidate for candidate in candidates if candidate.success and np.isfinite(candidate.fun)]
    if not successful:
        messages = [str(candidate.message) for candidate in candidates]
        return DriftEstimate(
            "gaussian_lgssm", "nonconverged", "; ".join(messages),
            None, None, None, None, None, None, None, None, None, None, False, None,
            {"n_trials": int(len(y)), "n_time": int(y.shape[1])},
        )
    result = min(successful, key=lambda candidate: candidate.fun)
    parameters = _state_parameters(result.x, dt)
    lambda_rate, diffusion, equilibrium, stationary_variance, observation_variance, process_variance = parameters
    try:
        covariance = np.asarray(result.hess_inv.todense(), dtype=float)
        lambda_ci, diffusion_ci, equilibrium_ci = _delta_method_intervals(result.x, covariance, dt)
    except Exception:
        lambda_ci = diffusion_ci = equilibrium_ci = None
    identifiability = confinement_identifiability(
        float(lambda_rate), dt, dt * (y.shape[1] - 1),
    )
    status = identifiability.status
    reason = identifiability.reason
    return DriftEstimate(
        "gaussian_lgssm", status, reason,
        float(lambda_rate), float(diffusion),
        None if not np.isfinite(equilibrium) else float(equilibrium),
        None if not np.isfinite(stationary_variance) else float(stationary_variance),
        float(observation_variance), float(process_variance),
        lambda_ci, diffusion_ci, equilibrium_ci,
        float(-result.fun), True, identifiability,
        {
            "n_trials": int(len(y)), "n_time": int(y.shape[1]),
            "optimizer": "L-BFGS-B", "n_successful_starts": len(successful),
            "interval_method": "Wald delta method from inverse likelihood Hessian",
            "discrete_a": float(np.exp(result.x[0])),
            "discrete_intercept": float(result.x[1]),
        },
    )


def gaussian_state_space_log_likelihood(
    observations: NDArray,
    estimate: DriftEstimate,
    dt: float,
) -> float:
    """Score held-out trials under a fitted Gaussian state-space estimate."""
    if not estimate.converged:
        return float("nan")
    required = (
        estimate.lambda_rate, estimate.process_variance,
        estimate.observation_variance,
    )
    if any(value is None or not np.isfinite(value) for value in required):
        return float("nan")
    a = float(np.exp(-float(estimate.lambda_rate) * dt))
    equilibrium = 0.0 if estimate.equilibrium is None else float(estimate.equilibrium)
    intercept = equilibrium * (1.0 - a)
    initial_variance = (
        float(estimate.stationary_variance)
        if estimate.stationary_variance is not None and estimate.stationary_variance > 0
        else float(np.var(observations))
    )
    parameters = np.array([
        np.log(max(a, 0.005)), intercept,
        np.log(max(float(estimate.process_variance), 1e-12)),
        np.log(max(float(estimate.observation_variance), 1e-12)),
        equilibrium, np.log(max(initial_variance, 1e-12)),
    ])
    values = np.asarray(observations, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        return float("nan")
    return float(-_kalman_negative_log_likelihood(parameters, values, dt))


def gaussian_state_space_conditional_log_likelihood(
    observations: NDArray,
    estimate: DriftEstimate,
    dt: float,
) -> float:
    """Score held-out transitions ``y[1:]`` conditional on ``y[0]``.

    Observation-level AR competitors use the first bin as a covariate rather
    than a scored target.  This conditional Kalman score gives the LGSSM the
    identical held-out targets while retaining distinct fitted process and
    observation variances.  The density of the conditioning bin is omitted.
    """
    if not estimate.converged:
        return float("nan")
    required = (
        estimate.lambda_rate,
        estimate.process_variance,
        estimate.observation_variance,
    )
    if any(value is None or not np.isfinite(value) for value in required):
        return float("nan")
    values = np.asarray(observations, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values) or values.shape[1] < 2:
        return float("nan")
    a = float(np.exp(-float(estimate.lambda_rate) * dt))
    equilibrium = 0.0 if estimate.equilibrium is None else float(estimate.equilibrium)
    intercept = equilibrium * (1.0 - a)
    process_variance = max(float(estimate.process_variance), 1e-12)
    observation_variance = max(float(estimate.observation_variance), 1e-12)
    initial_variance = (
        float(estimate.stationary_variance)
        if estimate.stationary_variance is not None and estimate.stationary_variance > 0
        else float(np.var(values))
    )
    total = 0.0
    for trial in values:
        mean = equilibrium
        variance = max(initial_variance, 1e-12)
        innovation_variance = variance + observation_variance
        gain = variance / innovation_variance
        mean += gain * (trial[0] - mean)
        variance = max((1.0 - gain) * variance, 1e-12)
        for value in trial[1:]:
            mean = a * mean + intercept
            variance = a * a * variance + process_variance
            innovation_variance = variance + observation_variance
            innovation = value - mean
            total += -0.5 * (
                np.log(2.0 * np.pi * innovation_variance)
                + innovation * innovation / innovation_variance
            )
            gain = variance / innovation_variance
            mean += gain * innovation
            variance = max((1.0 - gain) * variance, 1e-12)
    return float(total)


def fit_gaussian_scale_mixture(
    observations: NDArray,
    n_iter: int = 200,
) -> dict[str, Any]:
    """Fit a zero-mean two-variance Gaussian mixture without temporal dependence.

    This comparator has flexible observation noise but no trial-specific state,
    transition, equilibrium, or process/observation variance decomposition.
    """
    values = np.asarray(observations, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 8 or np.var(values) < 1e-12:
        return {"status": "not_estimable", "reason": "insufficient finite nonconstant observations"}
    pooled_variance = float(np.mean(values * values))
    variances = np.array([0.5 * pooled_variance, 2.0 * pooled_variance])
    weight = 0.5
    previous = -np.inf
    converged = False
    for iteration in range(n_iter):
        log_components = np.column_stack([
            np.log(max(weight, 1e-12)) - 0.5 * (
                np.log(2.0 * np.pi * variances[0]) + values * values / variances[0]
            ),
            np.log(max(1.0 - weight, 1e-12)) - 0.5 * (
                np.log(2.0 * np.pi * variances[1]) + values * values / variances[1]
            ),
        ])
        normalizer = logsumexp(log_components, axis=1)
        responsibilities = np.exp(log_components - normalizer[:, None])
        effective = responsibilities.sum(axis=0)
        weight = float(np.clip(effective[0] / values.size, 1e-4, 1.0 - 1e-4))
        variances = np.maximum(
            (responsibilities * values[:, None] ** 2).sum(axis=0) / np.maximum(effective, 1e-12),
            pooled_variance * 1e-6,
        )
        order = np.argsort(variances)
        variances = variances[order]
        if order[0] == 1:
            weight = 1.0 - weight
        likelihood = float(np.sum(normalizer))
        if iteration > 1 and abs(likelihood - previous) <= 1e-9 * (1.0 + abs(previous)):
            converged = True
            break
        previous = likelihood
    return {
        "status": "complete" if converged else "nonconverged",
        "reason": None if converged else "maximum EM iterations reached",
        "variances": variances.tolist(),
        "low_variance_weight": weight,
        "pooled_residual_variance": pooled_variance,
        "iterations": iteration + 1,
        "log_likelihood": likelihood,
    }


def gaussian_scale_mixture_log_likelihood(
    observations: NDArray,
    estimate: dict[str, Any],
) -> float:
    """Score observations under a fitted zero-mean Gaussian scale mixture."""
    if estimate.get("status") not in {"complete", "nonconverged"}:
        return float("nan")
    values = np.asarray(observations, dtype=float)
    values = values[np.isfinite(values)]
    variances = np.asarray(estimate["variances"], dtype=float)
    weight = float(estimate["low_variance_weight"])
    log_components = np.column_stack([
        np.log(max(weight, 1e-12)) - 0.5 * (
            np.log(2.0 * np.pi * variances[0]) + values * values / variances[0]
        ),
        np.log(max(1.0 - weight, 1e-12)) - 0.5 * (
            np.log(2.0 * np.pi * variances[1]) + values * values / variances[1]
        ),
    ])
    return float(np.sum(logsumexp(log_components, axis=1)))


def fit_free_variance_ar1(observations: NDArray) -> dict[str, Any]:
    """Fit one observation-level AR(1) with one free residual variance.

    The model has no latent state, confinement interpretation, equilibrium
    parameterization, or separately identified process and observation noise.
    """
    y = np.asarray(observations, dtype=float)
    y = y[np.all(np.isfinite(y), axis=1)]
    if len(y) < 3 or y.shape[1] < 2:
        return {"status": "not_estimable", "reason": "need at least three complete trials"}
    predictor = y[:, :-1].reshape(-1)
    target = y[:, 1:].reshape(-1)
    design = np.column_stack([np.ones_like(predictor), predictor])
    intercept, coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ np.array([intercept, coefficient])
    variance = max(float(np.mean(residual * residual)), float(np.var(target)) * 1e-8, 1e-12)
    return {
        "status": "complete",
        "reason": None,
        "intercept": float(intercept),
        "coefficient": float(coefficient),
        "residual_variance": variance,
    }


def free_variance_ar1_log_likelihood(
    observations: NDArray,
    estimate: dict[str, Any],
) -> float:
    """Score held-out transitions under a fitted observation-level AR(1)."""
    if estimate.get("status") != "complete":
        return float("nan")
    y = np.asarray(observations, dtype=float)
    y = y[np.all(np.isfinite(y), axis=1)]
    prediction = float(estimate["intercept"]) + float(estimate["coefficient"]) * y[:, :-1]
    residual = y[:, 1:] - prediction
    variance = float(estimate["residual_variance"])
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + residual * residual / variance)))


def compare_temporal_dependence_models(
    train_observations: NDArray,
    test_observations: NDArray,
    dt: float,
) -> dict[str, Any]:
    """Score M2 and two matched-flexibility no-latent-dynamics comparators."""
    state_space = fit_gaussian_state_space(train_observations, dt)
    mixture = fit_gaussian_scale_mixture(np.asarray(train_observations)[:, 1:])
    ar1 = fit_free_variance_ar1(train_observations)
    denominator = max(np.asarray(test_observations)[:, 1:].size, 1)
    m2_score = gaussian_state_space_conditional_log_likelihood(
        test_observations, state_space, dt
    ) / denominator
    mixture_score = gaussian_scale_mixture_log_likelihood(
        np.asarray(test_observations)[:, 1:], mixture
    ) / denominator
    ar1_score = free_variance_ar1_log_likelihood(test_observations, ar1) / denominator
    return {
        "state_space": state_space.to_dict(),
        "heteroscedastic_m0": mixture,
        "free_variance_ar1_m0": ar1,
        "m2_log_likelihood_per_transition": m2_score,
        "heteroscedastic_m0_log_likelihood_per_transition": mixture_score,
        "free_variance_ar1_m0_log_likelihood_per_transition": ar1_score,
        "m2_minus_heteroscedastic_m0_nats_per_transition": m2_score - mixture_score,
        "m2_minus_free_variance_ar1_m0_nats_per_transition": m2_score - ar1_score,
    }


def held_out_linear_prediction(
    train_predictor: NDArray,
    train_target: NDArray,
    test_predictor: NDArray,
    test_target: NDArray,
) -> dict[str, Any]:
    """Fit a scalar linear predictor and report held-out R-squared over zero."""
    x_train = np.asarray(train_predictor, dtype=float).reshape(-1)
    y_train = np.asarray(train_target, dtype=float).reshape(-1)
    x_test = np.asarray(test_predictor, dtype=float).reshape(-1)
    y_test = np.asarray(test_target, dtype=float).reshape(-1)
    finite_train = np.isfinite(x_train) & np.isfinite(y_train)
    finite_test = np.isfinite(x_test) & np.isfinite(y_test)
    if finite_train.sum() < 4 or finite_test.sum() < 1:
        return {"status": "not_estimable", "reason": "insufficient finite predictor-target pairs"}
    design = np.column_stack([np.ones(finite_train.sum()), x_train[finite_train]])
    intercept, coefficient = np.linalg.lstsq(design, y_train[finite_train], rcond=None)[0]
    prediction = intercept + coefficient * x_test[finite_test]
    baseline_error = float(np.sum(y_test[finite_test] ** 2))
    model_error = float(np.sum((y_test[finite_test] - prediction) ** 2))
    return {
        "status": "estimable",
        "reason": None,
        "intercept": float(intercept),
        "coefficient": float(coefficient),
        "held_out_r2_advantage": float(1.0 - model_error / max(baseline_error, 1e-12)),
        "n_train_pairs": int(finite_train.sum()),
        "n_test_pairs": int(finite_test.sum()),
    }


def preceding_same_condition_trial_residuals(
    residuals: NDArray,
    labels: NDArray,
    train_indices: NDArray,
    test_indices: NDArray,
) -> NDArray:
    """Construct leakage-safe preceding-trial predictors for one fold.

    Trial indices must preserve acquisition order.  Each neighbouring
    predictor uses only an earlier trial with the same condition, so no future
    session activity enters a prediction. Training targets use earlier training
    trials only; held-out targets use the closest earlier observed trial. Thus
    held-out residuals cannot enter the fitted neighbor coefficient.
    """
    y = np.asarray(residuals, dtype=float)
    condition = np.asarray(labels)
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    neighbour = np.full_like(y, np.nan)
    train_set = set(train_indices.tolist())
    test_set = set(test_indices.tolist())
    previous_any: dict[Any, int] = {}
    previous_train: dict[Any, int] = {}
    for index in range(len(y)):
        label = condition[index].item() if hasattr(condition[index], "item") else condition[index]
        if index in train_set and label in previous_train:
            neighbour[index] = y[previous_train[label]]
        elif index in test_set and label in previous_any:
            neighbour[index] = y[previous_any[label]]
        previous_any[label] = index
        if index in train_set:
            previous_train[label] = index
    return neighbour


def neighbouring_trial_prediction_advantage(
    residuals: NDArray,
    labels: NDArray,
    train_indices: NDArray,
    test_indices: NDArray,
) -> dict[str, Any]:
    """Compare own-lag and preceding same-condition-trial prediction."""
    y = np.asarray(residuals, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    neighbour = preceding_same_condition_trial_residuals(
        y, labels, train_indices, test_indices
    )
    own = held_out_linear_prediction(
        y[train_indices, :-1], y[train_indices, 1:],
        y[test_indices, :-1], y[test_indices, 1:],
    )
    adjacent = held_out_linear_prediction(
        neighbour[train_indices, :-1], y[train_indices, 1:],
        neighbour[test_indices, :-1], y[test_indices, 1:],
    )
    difference = (
        float(own["held_out_r2_advantage"] - adjacent["held_out_r2_advantage"])
        if own.get("status") == "estimable" and adjacent.get("status") == "estimable"
        else None
    )
    return {
        "own_trial": own,
        "preceding_same_condition_trial": adjacent,
        "own_minus_neighbour_r2_advantage": difference,
    }


def _autocovariances(y: NDArray, max_lag: int) -> NDArray:
    centered = y - np.mean(y, axis=0, keepdims=True)
    values = np.empty(max_lag + 1)
    values[0] = np.mean(centered * centered)
    for lag in range(1, max_lag + 1):
        values[lag] = np.mean(centered[:, :-lag] * centered[:, lag:])
    return values


def fit_ou_moments(
    residuals: NDArray,
    dt: float,
    max_lag: int | None = None,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> DriftEstimate:
    """Fit OU autocovariance and variance functions without using lag zero.

    Lag zero is excluded from the exponential fit so white observation noise
    is not mistaken for diffusion.  The variance-function estimate is fit
    independently and reported in diagnostics; non-overlapping confidence
    intervals are left for the caller's crack register.
    """
    y = np.asarray(residuals, dtype=float)
    if y.ndim != 2:
        raise ValueError("residuals must have shape (trials, time)")
    y = y[np.all(np.isfinite(y), axis=1)]
    n_trials, n_time = y.shape if y.ndim == 2 else (0, 0)
    if n_trials < 4 or n_time < 5 or np.var(y) < 1e-12:
        return DriftEstimate(
            "ou_moments", "not_estimable", "need at least 4 nonconstant trials and 5 time bins",
            None, None, None, None, None, None, None, None, None, None, False, None,
            {"n_trials": int(n_trials), "n_time": int(n_time)},
        )
    max_lag = min(max_lag or max(2, n_time // 3), n_time - 2)
    covariance = _autocovariances(y, max_lag)
    lag_times = dt * np.arange(1, max_lag + 1)

    def autocovariance_model(tau: NDArray, stationary_variance: float, lambda_rate: float) -> NDArray:
        return stationary_variance * np.exp(-lambda_rate * tau)

    positive_scale = max(float(covariance[1]), float(np.var(y)) * 0.05, 1e-10)
    try:
        auto_parameters, auto_covariance = curve_fit(
            autocovariance_model, lag_times, covariance[1:],
            p0=(positive_scale, 1.0 / max(dt * n_time / 3.0, dt)),
            bounds=((1e-12, 1e-8), (np.inf, 20.0 / dt)),
            maxfev=20000,
        )
    except Exception as exc:
        return DriftEstimate(
            "ou_moments", "nonconverged", f"autocovariance fit failed: {exc}",
            None, None, None, None, None, None, None, None, None, None, False, None,
            {"n_trials": int(n_trials), "n_time": int(n_time)},
        )
    stationary_variance, lambda_rate = map(float, auto_parameters)
    observation_variance = max(float(covariance[0] - stationary_variance), 0.0)
    diffusion = float(lambda_rate * stationary_variance)
    rng = np.random.default_rng(0) if rng is None else rng
    bootstrap_parameters = []
    for _ in range(n_boot):
        sample = y[rng.integers(0, n_trials, size=n_trials)]
        sample_covariance = _autocovariances(sample, max_lag)
        try:
            parameters, _ = curve_fit(
                autocovariance_model, lag_times, sample_covariance[1:],
                p0=(stationary_variance, lambda_rate),
                bounds=((1e-12, 1e-8), (np.inf, 20.0 / dt)),
                maxfev=10000,
            )
        except Exception:
            continue
        if np.all(np.isfinite(parameters)):
            bootstrap_parameters.append(parameters)
    bootstrap_parameters = np.asarray(bootstrap_parameters)
    if len(bootstrap_parameters) >= max(50, int(0.8 * n_boot)):
        bootstrap_lambda = bootstrap_parameters[:, 1]
        bootstrap_diffusion = bootstrap_parameters[:, 0] * bootstrap_parameters[:, 1]
        lambda_ci = tuple(map(float, np.percentile(bootstrap_lambda, [2.5, 97.5])))
        diffusion_ci = tuple(map(float, np.percentile(bootstrap_diffusion, [2.5, 97.5])))
        interval_method = "trial-cluster percentile bootstrap"
    else:
        lambda_ci = None
        diffusion_ci = None
        interval_method = "not_estimable"

    times = dt * np.arange(n_time)
    variances = np.var(y, axis=0, ddof=1)

    def variance_model(time: NDArray, stationary: float, initial: float, rate: float) -> NDArray:
        return stationary + (initial - stationary) * np.exp(-2.0 * rate * time)

    variance_fit: dict[str, Any]
    try:
        variance_parameters, variance_covariance = curve_fit(
            variance_model, times, variances,
            p0=(stationary_variance + observation_variance, float(variances[0]), lambda_rate),
            bounds=((0.0, 0.0, 1e-8), (np.inf, np.inf, 20.0 / dt)),
            maxfev=20000,
        )
        variance_lambda = float(variance_parameters[2])
        variance_lambda_se = float(np.sqrt(max(variance_covariance[2, 2], 0.0)))
        variance_fit = {
            "status": "ok", "lambda_rate": variance_lambda,
            "lambda_ci": [variance_lambda - 1.96 * variance_lambda_se,
                          variance_lambda + 1.96 * variance_lambda_se],
        }
    except Exception as exc:
        variance_fit = {"status": "nonconverged", "reason": str(exc)}
    identifiability = confinement_identifiability(lambda_rate, dt, dt * (n_time - 1))
    return DriftEstimate(
        "ou_moments", identifiability.status, identifiability.reason,
        lambda_rate, diffusion, 0.0, stationary_variance, observation_variance,
        None, lambda_ci, diffusion_ci, None, None, True, identifiability,
        {
            "n_trials": int(n_trials), "n_time": int(n_time), "max_lag": int(max_lag),
            "lag_zero_excluded": True, "autocovariance": covariance.tolist(),
            "variance_function_fit": variance_fit,
            "interval_method": interval_method,
            "n_boot_requested": int(n_boot),
            "n_boot_converged": int(len(bootstrap_parameters)),
        },
    )


def simulate_confined_diffusion(
    n_trials: int,
    n_time: int,
    dt: float,
    lambda_rate: float,
    diffusion: float,
    equilibrium: float = 0.0,
    observation_sd: float = 0.0,
    initial_sd: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Simulate exact-discretized scalar OU states and noisy observations."""
    rng = np.random.default_rng(0) if rng is None else rng
    if lambda_rate > 1e-10:
        a = float(np.exp(-lambda_rate * dt))
        stationary_variance = diffusion / lambda_rate
        process_variance = stationary_variance * (1.0 - a * a)
        default_initial_sd = np.sqrt(stationary_variance)
    else:
        a = 1.0
        process_variance = 2.0 * diffusion * dt
        default_initial_sd = 0.0
    states = np.empty((n_trials, n_time), dtype=float)
    states[:, 0] = equilibrium + rng.normal(
        scale=default_initial_sd if initial_sd is None else initial_sd,
        size=n_trials,
    )
    for time_index in range(1, n_time):
        states[:, time_index] = (
            equilibrium + a * (states[:, time_index - 1] - equilibrium)
            + rng.normal(scale=np.sqrt(process_variance), size=n_trials)
        )
    observations = states + rng.normal(scale=observation_sd, size=states.shape)
    return states, observations


def _ar_hmm_expectation(
    observations: NDArray,
    initial_probability: NDArray,
    transition_probability: NDArray,
    coefficients: NDArray,
    intercepts: NDArray,
    variances: NDArray,
) -> tuple[NDArray, NDArray, float]:
    """Forward-backward expectations for a Gaussian switching AR model."""
    log_initial = np.log(np.clip(initial_probability, 1e-12, 1.0))
    log_transition = np.log(np.clip(transition_probability, 1e-12, 1.0))
    x = observations[:, :-1]
    target = observations[:, 1:]
    residual = target[:, :, None] - (
        x[:, :, None] * coefficients[None, None, :]
        + intercepts[None, None, :]
    )
    log_emission = -0.5 * (
        np.log(2.0 * np.pi * variances)[None, None, :]
        + residual * residual / variances[None, None, :]
    )
    n_trials, n_steps, n_states = log_emission.shape
    alpha = np.empty((n_trials, n_steps, n_states))
    alpha[:, 0] = log_initial[None, :] + log_emission[:, 0]
    for time_index in range(1, n_steps):
        alpha[:, time_index] = log_emission[:, time_index] + logsumexp(
            alpha[:, time_index - 1, :, None] + log_transition[None, :, :], axis=1,
        )
    sequence_log_likelihood = logsumexp(alpha[:, -1], axis=1)
    beta = np.zeros((n_trials, n_steps, n_states))
    for time_index in range(n_steps - 2, -1, -1):
        beta[:, time_index] = logsumexp(
            log_transition[None, :, :]
            + log_emission[:, time_index + 1, None, :]
            + beta[:, time_index + 1, None, :],
            axis=2,
        )
    gamma = np.exp(alpha + beta - sequence_log_likelihood[:, None, None])
    xi = np.empty((n_trials, max(n_steps - 1, 0), n_states, n_states))
    for time_index in range(n_steps - 1):
        xi[:, time_index] = np.exp(
            alpha[:, time_index, :, None]
            + log_transition[None, :, :]
            + log_emission[:, time_index + 1, None, :]
            + beta[:, time_index + 1, None, :]
            - sequence_log_likelihood[:, None, None]
        )
    return gamma, xi, float(np.sum(sequence_log_likelihood))


def fit_switching_ar_hmm(
    observations: NDArray,
    n_states: int = 2,
    n_iter: int = 100,
    n_restarts: int = 5,
    ridge: float = 1e-5,
    tie_variances: bool = False,
    variance_floor_fraction: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Fit a probabilistic switching scalar AR model with Markov persistence.

    This is an observation-level switching linear dynamical competitor:
    ``y[t+1] = a[z[t]]*y[t] + b[z[t]] + noise`` and ``z`` follows a learned
    Markov transition matrix.  Unlike the legacy hard-assignment switching AR,
    it has an explicit state-transition model and marginalizes state sequences
    by forward-backward inference.  It is still not a Poisson rSLDS with a
    separately inferred continuous latent observation model; callers must keep
    that limitation in the artifact.  Every state variance is constrained to
    at least ``1e-3`` of the fold's pooled target variance by default.  A fit
    that lands on this numerical boundary is non-identified and must not be
    scored or included in summaries.
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or y.shape[0] < 4 or y.shape[1] < 5:
        return {
            "status": "not_estimable",
            "reason": "need at least 4 complete trials and 5 time bins",
            "n_states": int(n_states),
        }
    y = y[np.all(np.isfinite(y), axis=1)]
    if len(y) < 4 or np.var(y) < 1e-12:
        return {
            "status": "not_estimable",
            "reason": "insufficient finite nonconstant trials",
            "n_states": int(n_states),
        }
    if n_states < 1:
        raise ValueError("n_states must be positive")
    rng = np.random.default_rng(0) if rng is None else rng
    x_all = y[:, :-1].reshape(-1)
    target_all = y[:, 1:].reshape(-1)
    if variance_floor_fraction <= 0.0:
        raise ValueError("variance_floor_fraction must be positive")
    pooled_target_variance = float(np.var(target_all))
    variance_floor = max(pooled_target_variance * variance_floor_fraction, 1e-10)
    best: dict[str, Any] | None = None
    for restart in range(n_restarts):
        if n_states == 1:
            coefficients = np.array([np.cov(x_all, target_all)[0, 1] / max(np.var(x_all), 1e-12)])
        else:
            coefficients = np.linspace(0.15, 0.9, n_states)
            coefficients += rng.normal(scale=0.04, size=n_states)
        intercepts = np.full(n_states, float(np.mean(target_all) * (1.0 - np.mean(coefficients))))
        variances = np.full(n_states, max(float(np.var(target_all - np.mean(coefficients) * x_all)), variance_floor))
        transition = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
        np.fill_diagonal(transition, 0.9 if n_states > 1 else 1.0)
        transition /= transition.sum(axis=1, keepdims=True)
        initial = np.full(n_states, 1.0 / n_states)
        previous = -np.inf
        converged = False
        trace: list[float] = []
        for iteration in range(n_iter):
            gammas, xis, log_likelihood = _ar_hmm_expectation(
                y, initial, transition, coefficients, intercepts, variances,
            )
            trace.append(float(log_likelihood))
            initial = np.mean(gammas[:, 0, :], axis=0)
            initial /= initial.sum()
            transition_counts = np.full((n_states, n_states), 1e-3)
            if xis.shape[1]:
                transition_counts += xis.sum(axis=(0, 1))
            transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
            design = np.column_stack((x_all, np.ones_like(x_all)))
            weights = gammas.reshape(-1, n_states)
            residual_sums = np.empty(n_states)
            effective_counts = np.empty(n_states)
            for state in range(n_states):
                weight = weights[:, state]
                weighted_design = design * weight[:, None]
                penalty = ridge * np.eye(2)
                beta = np.linalg.solve(design.T @ weighted_design + penalty, weighted_design.T @ target_all)
                coefficients[state], intercepts[state] = beta
                residual = target_all - design @ beta
                residual_sums[state] = float(np.sum(weight * residual * residual))
                effective_counts[state] = float(np.sum(weight))
                variances[state] = max(residual_sums[state] / max(effective_counts[state], 1e-12), variance_floor)
            if tie_variances:
                pooled_variance = max(
                    float(np.sum(residual_sums) / max(np.sum(effective_counts), 1e-12)),
                    variance_floor,
                )
                variances.fill(pooled_variance)
            if iteration > 1 and abs(log_likelihood - previous) <= 1e-7 * (1.0 + abs(previous)):
                converged = True
                break
            previous = log_likelihood
        gammas, _xis, log_likelihood = _ar_hmm_expectation(
            y, initial, transition, coefficients, intercepts, variances,
        )
        occupancy = np.sum(gammas, axis=(0, 1))
        occupancy /= occupancy.sum()
        variance_parameters = 1 if tie_variances else n_states
        n_parameters = n_states * 2 + variance_parameters + n_states * (n_states - 1) + (n_states - 1)
        candidate = {
            "status": "complete" if converged else "nonconverged",
            "reason": None if converged else "maximum EM iterations reached",
            "n_states": int(n_states),
            "coefficients": coefficients.tolist(),
            "intercepts": intercepts.tolist(),
            "variances": variances.tolist(),
            "pooled_target_variance": pooled_target_variance,
            "variance_floor": variance_floor,
            "variance_floor_fraction": float(variance_floor_fraction),
            "variance_floor_hit": bool(np.any(variances <= variance_floor * (1.0 + 1e-7))),
            "tie_variances": bool(tie_variances),
            "initial_probability": initial.tolist(),
            "transition_probability": transition.tolist(),
            "state_occupancy": occupancy.tolist(),
            "log_likelihood": float(log_likelihood),
            "n_parameters": int(n_parameters),
            "bic": float(-2.0 * log_likelihood + n_parameters * np.log(y[:, 1:].size)),
            "iterations": int(len(trace)),
            "log_likelihood_trace": trace,
            "limitation": "Gaussian observation-level AR-HMM; not a Poisson rSLDS with separate continuous latent observation noise",
        }
        if best is None or candidate["log_likelihood"] > best["log_likelihood"]:
            best = candidate
    assert best is not None
    if best["variance_floor_hit"]:
        best["status"] = "non_identified"
        best["reason"] = "at least one fitted state variance reached the declared variance floor"
    return best


def switching_fit_decomposition(estimate: dict[str, Any]) -> dict[str, Any]:
    """Summarize the separation learned by a fitted two-state AR-HMM."""
    if estimate.get("status") not in {"complete", "nonconverged"}:
        return {
            "status": "not_estimable",
            "reason": estimate.get("reason", "switching fit unavailable"),
        }
    coefficients = np.asarray(estimate.get("coefficients", []), dtype=float)
    variances = np.asarray(estimate.get("variances", []), dtype=float)
    transition = np.asarray(estimate.get("transition_probability", []), dtype=float)
    occupancy = np.asarray(estimate.get("state_occupancy", []), dtype=float)
    if coefficients.size != 2 or variances.size != 2 or transition.shape != (2, 2):
        return {
            "status": "not_estimable",
            "reason": "decomposition requires a fitted two-state model",
        }
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        return {
            "status": "not_estimable",
            "reason": "state variances must be finite and positive",
        }
    return {
        "status": "estimable",
        "absolute_coefficient_separation": float(abs(coefficients[0] - coefficients[1])),
        "absolute_log_variance_separation": float(abs(np.log(variances[0] / variances[1]))),
        "mean_self_transition_probability": float(np.mean(np.diag(transition))),
        "state_occupancy": occupancy.tolist(),
        "minimum_state_occupancy": float(np.min(occupancy)) if occupancy.size else None,
    }


def summarize_switching_decompositions(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fitted two-state separations across a collection of folds."""
    records = [
        row.get("switching_decomposition")
        or switching_fit_decomposition(
            row.get("switching_two_state", row.get("switching_two", {}))
        )
        for row in fold_rows
    ]
    records = [record for record in records if record.get("status") == "estimable"]
    if not records:
        return {
            "status": "not_estimable",
            "reason": "no fold has a complete two-state decomposition",
            "n_folds": 0,
        }
    fields = (
        "absolute_coefficient_separation",
        "absolute_log_variance_separation",
        "mean_self_transition_probability",
        "minimum_state_occupancy",
    )
    summary: dict[str, Any] = {"status": "estimable", "n_folds": len(records)}
    for field in fields:
        values = np.asarray(
            [record[field] for record in records if record.get(field) is not None],
            dtype=float,
        )
        summary[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "interquartile_range": list(map(float, np.percentile(values, [25.0, 75.0]))),
        }
    return summary


def _heteroscedastic_kalman_log_likelihood(
    parameters: NDArray,
    observations: NDArray,
    score_first_observation: bool,
) -> float:
    """Collapsed Gaussian-mixture observation update for a scalar LGSSM."""
    log_a, intercept, log_q, log_r_low, log_r_gap, logit_weight, initial_mean, log_initial_variance = parameters
    a = float(np.exp(log_a))
    if not 0.005 <= a <= 2.0:
        return -1e30
    q = float(np.exp(log_q))
    r_low = float(np.exp(log_r_low))
    r_high = r_low + float(np.exp(log_r_gap))
    weight_low = float(1.0 / (1.0 + np.exp(-np.clip(logit_weight, -30.0, 30.0))))
    mixture_weights = np.array([weight_low, 1.0 - weight_low])
    observation_variances = np.array([r_low, r_high])
    n_trials = observations.shape[0]
    means = np.full(n_trials, float(initial_mean))
    variances = np.full(n_trials, float(np.exp(log_initial_variance)))
    total = 0.0

    def update(values: NDArray, include_score: bool) -> None:
        nonlocal means, variances, total
        innovations = values - means
        component_variances = variances[:, None] + observation_variances[None, :]
        log_components = np.log(np.clip(mixture_weights, 1e-12, 1.0))[None, :] - 0.5 * (
            np.log(2.0 * np.pi * component_variances)
            + innovations[:, None] ** 2 / component_variances
        )
        log_densities = logsumexp(log_components, axis=1)
        if include_score:
            total += float(np.sum(log_densities))
        responsibilities = np.exp(log_components - log_densities[:, None])
        gains = variances[:, None] / component_variances
        component_means = means[:, None] + gains * innovations[:, None]
        component_post_variances = np.maximum(
            (1.0 - gains) * variances[:, None], 1e-12
        )
        means = np.sum(responsibilities * component_means, axis=1)
        variances = np.sum(
            responsibilities
            * (component_post_variances + (component_means - means[:, None]) ** 2),
            axis=1,
        )

    start = 0
    if not score_first_observation:
        update(observations[:, 0], False)
        start = 1
    for time_index in range(start, observations.shape[1]):
        if time_index > 0:
            means = a * means + intercept
            variances = a * a * variances + q
        update(observations[:, time_index], True)
    return float(total)


def fit_heteroscedastic_gaussian_state_space(
    observations: NDArray,
    dt: float,
    n_restarts: int = 3,
) -> dict[str, Any]:
    """Fit confined drift with a two-scale Gaussian observation-noise mixture.

    The latent dynamics remain a single scalar LGSSM.  Only the observation
    variance is mixed, and the Gaussian posterior is moment-matched after
    each mixture update so the recursion stays linear in sequence length.
    """
    y = np.asarray(observations, dtype=float)
    if y.ndim != 2 or y.shape[0] < 4 or y.shape[1] < 5:
        return {"status": "not_estimable", "reason": "need at least 4 complete trials and 5 time bins"}
    y = y[np.all(np.isfinite(y), axis=1)]
    if len(y) < 4 or np.var(y) < 1e-12:
        return {"status": "not_estimable", "reason": "insufficient finite nonconstant trials"}
    baseline = fit_gaussian_state_space(y, dt)
    if not baseline.converged:
        return {"status": "nonconverged", "reason": "single-variance initialization failed"}
    variance = float(np.var(y))
    a0 = float(np.exp(-float(baseline.lambda_rate) * dt))
    q0 = max(float(baseline.process_variance), variance * 1e-6)
    r0 = max(float(baseline.observation_variance), variance * 1e-6)
    equilibrium = 0.0 if baseline.equilibrium is None else float(baseline.equilibrium)
    intercept0 = equilibrium * (1.0 - a0)
    initial_variance = max(
        float(baseline.stationary_variance or np.var(y[:, 0])), variance * 1e-6,
    )
    starts = []
    for ratio in np.geomspace(2.0, 8.0, max(n_restarts, 1)):
        starts.append(np.array([
            np.log(np.clip(a0, 0.005, 2.0)), intercept0, np.log(q0),
            np.log(max(r0 / ratio, variance * 1e-8)),
            np.log(max(r0 * ratio - r0 / ratio, variance * 1e-8)),
            0.0, equilibrium, np.log(initial_variance),
        ]))
    scale = max(np.sqrt(variance), 1e-4)
    bounds = [
        (np.log(0.005), np.log(2.0)), (-10.0 * scale, 10.0 * scale),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
        (-8.0, 8.0), (-10.0 * scale, 10.0 * scale),
        (np.log(max(variance * 1e-9, 1e-12)), np.log(variance * 100.0)),
    ]
    candidates = [
        minimize(
            lambda value: -_heteroscedastic_kalman_log_likelihood(value, y, True),
            start, method="L-BFGS-B", bounds=bounds,
        )
        for start in starts
    ]
    successful = [candidate for candidate in candidates if candidate.success and np.isfinite(candidate.fun)]
    if not successful:
        return {
            "status": "nonconverged",
            "reason": "; ".join(str(candidate.message) for candidate in candidates),
        }
    result = min(successful, key=lambda candidate: candidate.fun)
    log_a, intercept, log_q, log_r_low, log_r_gap, logit_weight, initial_mean, log_initial_variance = result.x
    a = float(np.exp(log_a))
    lambda_rate = float(-np.log(a) / dt)
    q = float(np.exp(log_q))
    r_low = float(np.exp(log_r_low))
    r_high = r_low + float(np.exp(log_r_gap))
    weight_low = float(1.0 / (1.0 + np.exp(-np.clip(logit_weight, -30.0, 30.0))))
    equilibrium_fit = float(intercept / (1.0 - a)) if abs(1.0 - a) > 1e-6 else None
    return {
        "status": "complete",
        "reason": None,
        "lambda_rate": lambda_rate,
        "discrete_a": a,
        "intercept": float(intercept),
        "equilibrium": equilibrium_fit,
        "process_variance": q,
        "observation_variances": [r_low, r_high],
        "low_variance_weight": weight_low,
        "initial_mean": float(initial_mean),
        "initial_variance": float(np.exp(log_initial_variance)),
        "log_likelihood": float(-result.fun),
        "n_parameters": 8,
        "n_successful_starts": int(len(successful)),
        "filter_approximation": "moment-matched Gaussian posterior after each observation-mixture update",
    }


def heteroscedastic_gaussian_state_space_conditional_log_likelihood(
    observations: NDArray,
    estimate: dict[str, Any],
) -> float:
    """Score transitions under a fitted heteroscedastic confined-drift model."""
    if estimate.get("status") != "complete":
        return float("nan")
    y = np.asarray(observations, dtype=float)
    y = y[np.all(np.isfinite(y), axis=1)]
    if not len(y) or y.shape[1] < 2:
        return float("nan")
    r_low, r_high = map(float, estimate["observation_variances"])
    gap = max(r_high - r_low, 1e-12)
    weight = float(np.clip(estimate["low_variance_weight"], 1e-8, 1.0 - 1e-8))
    parameters = np.array([
        np.log(max(float(estimate["discrete_a"]), 0.005)),
        float(estimate["intercept"]),
        np.log(max(float(estimate["process_variance"]), 1e-12)),
        np.log(max(r_low, 1e-12)), np.log(gap), np.log(weight / (1.0 - weight)),
        float(estimate["initial_mean"]),
        np.log(max(float(estimate["initial_variance"]), 1e-12)),
    ])
    return _heteroscedastic_kalman_log_likelihood(parameters, y, False)


def simulate_heteroscedastic_confined_diffusion(
    n_trials: int,
    n_time: int,
    estimate: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Simulate one confined process with fitted two-scale observation noise."""
    if estimate.get("status") != "complete":
        raise ValueError("a complete heteroscedastic drift fit is required")
    rng = np.random.default_rng(0) if rng is None else rng
    a = float(estimate["discrete_a"])
    intercept = float(estimate["intercept"])
    process_variance = float(estimate["process_variance"])
    initial_mean = float(estimate["initial_mean"])
    initial_variance = float(estimate["initial_variance"])
    observation_variances = np.asarray(estimate["observation_variances"], dtype=float)
    weight = float(estimate["low_variance_weight"])
    latent = np.empty((n_trials, n_time), dtype=float)
    latent[:, 0] = rng.normal(initial_mean, np.sqrt(initial_variance), size=n_trials)
    for time_index in range(1, n_time):
        latent[:, time_index] = (
            a * latent[:, time_index - 1] + intercept
            + rng.normal(scale=np.sqrt(process_variance), size=n_trials)
        )
    component = rng.random((n_trials, n_time)) >= weight
    noise_sd = np.sqrt(observation_variances[component.astype(int)])
    return latent + rng.normal(size=latent.shape) * noise_sd


def simulate_switching_ar_hmm(
    n_trials: int,
    n_time: int,
    estimate: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Simulate observations from a fitted switching scalar AR-HMM."""
    if estimate.get("status") not in {"complete", "nonconverged"}:
        raise ValueError("a fitted switching model is required")
    rng = np.random.default_rng(0) if rng is None else rng
    initial = np.asarray(estimate["initial_probability"], dtype=float)
    transition = np.asarray(estimate["transition_probability"], dtype=float)
    coefficients = np.asarray(estimate["coefficients"], dtype=float)
    intercepts = np.asarray(estimate["intercepts"], dtype=float)
    variances = np.asarray(estimate["variances"], dtype=float)
    observations = np.zeros((n_trials, n_time), dtype=float)
    stationary_scale = np.sqrt(max(float(np.mean(variances)), 1e-12))
    observations[:, 0] = rng.normal(scale=stationary_scale, size=n_trials)
    states = np.array([rng.choice(len(initial), p=initial) for _ in range(n_trials)], dtype=int)
    for time_index in range(1, n_time):
        observations[:, time_index] = (
            coefficients[states] * observations[:, time_index - 1]
            + intercepts[states]
            + rng.normal(scale=np.sqrt(variances[states]), size=n_trials)
        )
        states = np.array([
            rng.choice(len(initial), p=transition[state]) for state in states
        ], dtype=int)
    return observations


def compare_switching_models(
    train_observations: np.ndarray,
    test_observations: np.ndarray,
    dt: float,
    *,
    n_restarts: int = 4,
    rng: np.random.Generator | None = None,
) -> dict:
    """Fit free, tied-variance, and heteroscedastic-drift competitors.

    All three models are trained on the same observations and scored on the
    same held-out transitions.  This keeps the switching decomposition and
    its two most important controls identical across dataset pipelines.
    """
    train = np.asarray(train_observations, dtype=float)
    test = np.asarray(test_observations, dtype=float)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("train and test observations must be two-dimensional")
    train = train[np.all(np.isfinite(train), axis=1)]
    test = test[np.all(np.isfinite(test), axis=1)]
    if len(train) < 4 or len(test) < 1 or train.shape[1] < 5 or test.shape[1] < 2:
        raise ValueError("insufficient complete trials or time bins for held-out comparison")
    generator = np.random.default_rng(0) if rng is None else rng
    seeds = generator.integers(0, np.iinfo(np.int32).max, size=2)
    free = fit_switching_ar_hmm(
        train,
        n_states=2,
        n_restarts=n_restarts,
        rng=np.random.default_rng(int(seeds[0])),
    )
    tied = fit_switching_ar_hmm(
        train,
        n_states=2,
        n_restarts=n_restarts,
        tie_variances=True,
        rng=np.random.default_rng(int(seeds[1])),
    )
    heteroscedastic = fit_heteroscedastic_gaussian_state_space(
        train,
        dt,
        n_restarts=max(2, n_restarts // 2),
    )
    denominator = max(test[:, 1:].size, 1)
    return {
        "free": free,
        "free_decomposition": switching_fit_decomposition(free),
        "tied_variance": tied,
        "heteroscedastic_drift": heteroscedastic,
        "free_log_likelihood_per_transition": (
            switching_ar_hmm_log_likelihood(test, free) / denominator
        ),
        "tied_log_likelihood_per_transition": (
            switching_ar_hmm_log_likelihood(test, tied) / denominator
        ),
        "heteroscedastic_drift_log_likelihood_per_transition": (
            heteroscedastic_gaussian_state_space_conditional_log_likelihood(
                test, heteroscedastic
            )
            / denominator
        ),
    }


def _time_resolved_coding_axes(states: NDArray, labels: NDArray) -> NDArray:
    classes = np.unique(labels)
    axes = []
    previous: NDArray | None = None
    for time_index in range(states.shape[1]):
        centroids = np.vstack([
            np.mean(states[labels == label, time_index], axis=0) for label in classes
        ])
        centered = centroids - np.mean(centroids, axis=0, keepdims=True)
        _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
        axis = right[0]
        if previous is not None and np.dot(axis, previous) < 0.0:
            axis = -axis
        axes.append(axis / max(np.linalg.norm(axis), 1e-12))
        previous = axes[-1]
    return np.asarray(axes)


def _fit_axis_rotation(axes: NDArray, ridge: float = 1e-3) -> tuple[NDArray, NDArray]:
    predictors = axes[:-1]
    targets = axes[1:]
    gram = predictors.T @ predictors + ridge * np.eye(predictors.shape[1])
    operator = np.linalg.solve(gram, predictors.T @ targets).T
    predicted = np.empty_like(axes)
    predicted[0] = axes[0]
    for time_index in range(1, len(axes)):
        predicted[time_index] = operator @ predicted[time_index - 1]
        predicted[time_index] /= max(np.linalg.norm(predicted[time_index]), 1e-12)
    return operator, predicted


def _rotation_residuals(
    train_states: NDArray,
    test_states: NDArray,
    train_labels: NDArray,
    test_labels: NDArray,
    predicted_axes: NDArray,
    target_direction: NDArray,
) -> tuple[NDArray, NDArray]:
    train_projection = train_states @ target_direction
    test_projection = test_states @ target_direction
    train_residuals = np.full_like(train_projection, np.nan)
    test_residuals = np.full_like(test_projection, np.nan)
    global_sum = np.sum(train_states, axis=0)
    for label in np.unique(train_labels):
        train_mask = train_labels == label
        test_mask = test_labels == label
        count = int(np.sum(train_mask))
        class_sum = np.sum(train_states[train_mask], axis=0)
        if count > 1:
            class_loo = (class_sum[None, :, :] - train_states[train_mask]) / (count - 1)
            global_loo = (global_sum[None, :, :] - train_states[train_mask]) / (
                len(train_states) - 1
            )
            centered = class_loo - global_loo
            amplitudes = np.sum(centered * predicted_axes[None, :, :], axis=2)
            predicted_vectors = global_loo + amplitudes[:, :, None] * predicted_axes[None, :, :]
            predicted_scalar = predicted_vectors @ target_direction
            train_residuals[train_mask] = train_projection[train_mask] - predicted_scalar
        if np.any(test_mask):
            class_mean = class_sum / count
            global_mean = global_sum / len(train_states)
            centered = class_mean - global_mean
            amplitudes = np.sum(centered * predicted_axes, axis=1)
            predicted_vectors = global_mean + amplitudes[:, None] * predicted_axes
            test_residuals[test_mask] = (
                test_projection[test_mask] - (predicted_vectors @ target_direction)[None, :]
            )
    return train_residuals, test_residuals


def _nearest_centroid_accuracy(
    train_states: NDArray,
    test_states: NDArray,
    train_labels: NDArray,
    test_labels: NDArray,
    axes: NDArray,
) -> float:
    classes = np.unique(train_labels)
    correct = 0
    total = 0
    for time_index, axis in enumerate(axes):
        train_projection = train_states[:, time_index] @ axis
        test_projection = test_states[:, time_index] @ axis
        centroids = np.asarray([
            np.mean(train_projection[train_labels == label]) for label in classes
        ])
        predictions = classes[np.argmin(
            np.abs(test_projection[:, None] - centroids[None, :]), axis=1
        )]
        correct += int(np.sum(predictions == test_labels))
        total += len(test_labels)
    return float(correct / max(total, 1))


def planted_rotation_recovery(
    n_train: int,
    n_test: int,
    n_time: int,
    n_dimensions: int,
    n_classes: int,
    dt: float,
    residual_snr: float,
    rotation_rate: float,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Run the counter-rotation test on a known rotation at matched design and SNR."""
    if n_dimensions < 2 or n_classes < 2:
        raise ValueError("planted rotation requires at least two dimensions and classes")
    generator = np.random.default_rng(0) if rng is None else rng
    train_labels = np.resize(np.arange(n_classes), n_train)
    test_labels = np.resize(np.arange(n_classes), n_test)
    generator.shuffle(train_labels)
    generator.shuffle(test_labels)
    class_amplitudes = np.linspace(-1.0, 1.0, n_classes)
    class_amplitudes /= max(float(np.std(class_amplitudes)), 1e-12)
    noise_sd = 1.0
    signal_scale = np.sqrt(max(float(residual_snr), 1e-8)) * noise_sd
    angles = rotation_rate * dt * np.arange(n_time)
    axes = np.zeros((n_time, n_dimensions), dtype=float)
    axes[:, 0] = np.cos(angles)
    axes[:, 1] = np.sin(angles)

    def sample(labels: NDArray) -> NDArray:
        signal = signal_scale * class_amplitudes[labels, None, None] * axes[None, :, :]
        return signal + generator.normal(scale=noise_sd, size=signal.shape)

    train_states = sample(train_labels)
    test_states = sample(test_labels)
    fitted_axes = _time_resolved_coding_axes(train_states, train_labels)
    _operator, predicted_axes = _fit_axis_rotation(fitted_axes)
    fixed_axes = np.repeat(fitted_axes[:1], n_time, axis=0)
    fixed_accuracy = _nearest_centroid_accuracy(
        train_states, test_states, train_labels, test_labels, fixed_axes
    )
    counter_accuracy = _nearest_centroid_accuracy(
        train_states, test_states, train_labels, test_labels, predicted_axes
    )
    fitted_angles = np.arccos(np.clip(
        np.sum(fitted_axes[:-1] * fitted_axes[1:], axis=1), -1.0, 1.0
    ))
    return {
        "planted_rotation_rate_radians_per_second": float(rotation_rate),
        "recovered_apparent_rotation_rate_radians_per_second": float(np.mean(fitted_angles) / dt),
        "fixed_axis_accuracy": fixed_accuracy,
        "counter_rotated_axis_accuracy": counter_accuracy,
        "counter_rotation_accuracy_recovery": float(counter_accuracy - fixed_accuracy),
    }


def fit_rotation_drift_comparison(
    train_states: NDArray,
    test_states: NDArray,
    train_labels: NDArray,
    test_labels: NDArray,
    target_direction: NDArray,
    m0_train_residuals: NDArray,
    m0_test_residuals: NDArray,
    m2_log_likelihood_per_transition: float,
    dt: float,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Score deterministic axis rotation and rotation-plus-drift on held-out data."""
    train_states = np.asarray(train_states, dtype=float)
    test_states = np.asarray(test_states, dtype=float)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    direction = np.asarray(target_direction, dtype=float)
    generator = np.random.default_rng(0) if rng is None else rng
    axes = _time_resolved_coding_axes(train_states, train_labels)
    operator, predicted_axes = _fit_axis_rotation(axes)
    rotation_train, rotation_test = _rotation_residuals(
        train_states, test_states, train_labels, test_labels, predicted_axes, direction
    )
    m0_variance = max(float(np.nanvar(m0_train_residuals)), 1e-10)
    m1_variance = max(float(np.nanvar(rotation_train)), 1e-10)
    m0_values = m0_test_residuals[np.isfinite(m0_test_residuals)]
    m1_values = rotation_test[np.isfinite(rotation_test)]
    m0_score = float(np.sum(-0.5 * (
        np.log(2.0 * np.pi * m0_variance) + m0_values ** 2 / m0_variance
    )))
    m1_score = float(np.sum(-0.5 * (
        np.log(2.0 * np.pi * m1_variance) + m1_values ** 2 / m1_variance
    )))
    m3 = fit_gaussian_state_space(rotation_train, dt)
    m3_score = gaussian_state_space_conditional_log_likelihood(rotation_test, m3, dt)
    transition_denominator = max(rotation_test[:, 1:].size, 1)
    fixed_axes = np.repeat(direction[None, :], train_states.shape[1], axis=0)
    fixed_accuracy = _nearest_centroid_accuracy(
        train_states, test_states, train_labels, test_labels, fixed_axes
    )
    counter_rotated_accuracy = _nearest_centroid_accuracy(
        train_states, test_states, train_labels, test_labels, predicted_axes
    )

    class_means = {
        label: np.mean(train_states[train_labels == label], axis=(0, 1))
        for label in np.unique(train_labels)
    }
    empirical_means = np.empty_like(train_states)
    for index, label in enumerate(train_labels):
        empirical_means[index] = class_means[label]
    residual_pool = train_states - empirical_means
    noise_scale = np.std(residual_pool, axis=(0, 1), ddof=1)
    stationary_train = np.empty_like(train_states)
    for index, label in enumerate(train_labels):
        stationary_train[index] = class_means[label] + generator.normal(
            scale=noise_scale, size=train_states.shape[1:]
        )
    stationary_axes = _time_resolved_coding_axes(stationary_train, train_labels)
    stationary_test = np.empty_like(test_states)
    for index, label in enumerate(test_labels):
        stationary_test[index] = class_means[label] + generator.normal(
            scale=noise_scale, size=test_states.shape[1:]
        )
    real_time_axis_accuracy = _nearest_centroid_accuracy(
        train_states, test_states, train_labels, test_labels, axes
    )
    stationary_accuracy = _nearest_centroid_accuracy(
        stationary_train, stationary_test, train_labels, test_labels, stationary_axes
    )
    class_mean_array = np.stack([class_means[label] for label in sorted(class_means)], axis=0)
    real_signal_variance = float(np.mean(np.var(class_mean_array, axis=0)))
    real_residual_variance = float(np.mean(np.var(residual_pool, axis=(0, 1), ddof=1)))
    stationary_class_means = np.stack([
        np.mean(stationary_train[train_labels == label], axis=(0, 1))
        for label in sorted(class_means)
    ])
    stationary_residuals = np.empty_like(stationary_train)
    for index, label in enumerate(train_labels):
        stationary_residuals[index] = stationary_train[index] - stationary_class_means[
            sorted(class_means).index(label)
        ]
    stationary_signal_variance = float(np.mean(np.var(stationary_class_means, axis=0)))
    stationary_residual_variance = float(
        np.mean(np.var(stationary_residuals, axis=(0, 1), ddof=1))
    )
    stationary_angles = np.arccos(np.clip(
        np.sum(stationary_axes[:-1] * stationary_axes[1:], axis=1), -1.0, 1.0
    ))
    observed_angles = np.arccos(np.clip(
        np.sum(axes[:-1] * axes[1:], axis=1), -1.0, 1.0
    ))
    return {
        "status": "complete",
        "operator": operator.tolist(),
        "mean_axis_rotation_rate_radians_per_second": float(np.mean(observed_angles) / dt),
        "m1_minus_m0_nats_per_observation": float(
            (m1_score - m0_score) / max(len(m1_values), 1)
        ),
        "m3_minus_m2_nats_per_transition": float(
            m3_score / transition_denominator - m2_log_likelihood_per_transition
        ),
        "m3_state_space": m3.to_dict(),
        "counter_rotation": {
            "fixed_axis_accuracy": fixed_accuracy,
            "counter_rotated_axis_accuracy": counter_rotated_accuracy,
            "accuracy_recovery": float(counter_rotated_accuracy - fixed_accuracy),
            "scoring": "held-out nearest-centroid accuracy with time-specific training centroids",
        },
        "stationary_code_floor": {
            "matched_train_trials": int(len(train_states)),
            "matched_test_trials": int(len(test_states)),
            "matched_time_bins": int(train_states.shape[1]),
            "matched_dimensions": int(train_states.shape[2]),
            "matched_classes": int(len(np.unique(train_labels))),
            "real_time_resolved_axis_accuracy": real_time_axis_accuracy,
            "stationary_time_resolved_axis_accuracy": stationary_accuracy,
            "real_residual_snr": real_signal_variance / max(real_residual_variance, 1e-12),
            "stationary_residual_snr": (
                stationary_signal_variance / max(stationary_residual_variance, 1e-12)
            ),
            "mean_apparent_rotation_rate_radians_per_second": float(
                np.mean(stationary_angles) / dt
            ),
        },
    }


def switching_ar_hmm_log_likelihood(observations: NDArray, estimate: dict[str, Any]) -> float:
    """Marginal held-out log likelihood for :func:`fit_switching_ar_hmm`."""
    if estimate.get("status") not in {"complete", "nonconverged"}:
        return float("nan")
    y = np.asarray(observations, dtype=float)
    y = y[np.all(np.isfinite(y), axis=1)]
    if not len(y):
        return float("nan")
    _gamma, _xi, log_likelihood = _ar_hmm_expectation(
        y,
        np.asarray(estimate["initial_probability"], dtype=float),
        np.asarray(estimate["transition_probability"], dtype=float),
        np.asarray(estimate["coefficients"], dtype=float),
        np.asarray(estimate["intercepts"], dtype=float),
        np.asarray(estimate["variances"], dtype=float),
    )
    return float(log_likelihood)
