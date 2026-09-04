#!/usr/bin/env python3
"""Validate drift, rotation, and switching estimators on planted mechanisms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_dynamics import (  # noqa: E402
    fit_gaussian_state_space,
    fit_ou_moments,
    simulate_confined_diffusion,
)
from dynamics import switching_ar_em, switching_ar_score  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402


def relative_error(estimate: float | None, truth: float) -> float:
    if estimate is None or not np.isfinite(estimate):
        return float("inf")
    return float(abs(estimate - truth) / max(abs(truth), 1e-12))


def rotation_recovery(rng: np.random.Generator) -> dict:
    n_trials, n_time = 240, 50
    labels = rng.choice([-1.0, 1.0], size=n_trials)
    angle = np.linspace(0.0, 1.4, n_time)
    axes = np.column_stack((np.cos(angle), np.sin(angle)))
    observations = labels[:, None, None] * axes[None, :, :] + rng.normal(
        scale=0.35, size=(n_trials, n_time, 2),
    )
    fitted_axes = np.empty_like(axes)
    for time_index in range(n_time):
        fitted_axes[time_index] = (
            observations[labels > 0, time_index].mean(axis=0)
            - observations[labels < 0, time_index].mean(axis=0)
        )
        fitted_axes[time_index] /= np.linalg.norm(fitted_axes[time_index])
    operator = np.linalg.lstsq(fitted_axes[:-1], fitted_axes[1:], rcond=None)[0].T
    counter_rotated = []
    inverse = np.linalg.inv(operator)
    for time_index, axis in enumerate(fitted_axes):
        corrected = np.linalg.matrix_power(inverse, time_index) @ axis
        corrected /= np.linalg.norm(corrected)
        counter_rotated.append(abs(float(corrected @ fitted_axes[0])))
    raw = np.abs(fitted_axes @ fitted_axes[0])
    recovered = np.asarray(counter_rotated)
    return {
        "raw_late_axis_cosine": float(np.mean(raw[n_time // 2 :])),
        "counter_rotated_late_axis_cosine": float(np.mean(recovered[n_time // 2 :])),
        "pass": bool(np.mean(recovered[n_time // 2 :]) > np.mean(raw[n_time // 2 :]) + 0.25),
    }


def switching_comparison(rng: np.random.Generator) -> dict:
    n_pairs = 5000
    x1 = rng.normal(size=(n_pairs, 1))
    regimes = rng.integers(0, 2, size=n_pairs)
    slopes = np.where(regimes == 0, 0.2, 1.25)
    x2 = slopes[:, None] * x1 + np.where(regimes[:, None] == 0, -0.8, 0.8)
    x2 += rng.normal(scale=0.12, size=x2.shape)
    order = rng.permutation(n_pairs)
    train, test = order[:3500], order[3500:]
    one = switching_ar_em(x1[train], x2[train], n_states=1, rng=np.random.default_rng(11))
    two = switching_ar_em(x1[train], x2[train], n_states=2, rng=np.random.default_rng(12))
    ll_one = switching_ar_score(one, x1[test], x2[test])
    ll_two = switching_ar_score(two, x1[test], x2[test])
    bic_one = -2 * ll_one + one["n_params"] * np.log(len(test))
    bic_two = -2 * ll_two + two["n_params"] * np.log(len(test))
    return {
        "held_out_log_likelihood_one_state": float(ll_one),
        "held_out_log_likelihood_two_state": float(ll_two),
        "bic_one_state": float(bic_one),
        "bic_two_state": float(bic_two),
        "pass": bool(bic_two < bic_one),
        "limitation": "dependency-free hard-assignment switching AR; a full rSLDS remains required",
    }


def coverage_check(rng: np.random.Generator, n_repetitions: int = 30) -> dict:
    truth = 1.0
    covered = 0
    estimable = 0
    for _ in range(n_repetitions):
        _, observations = simulate_confined_diffusion(
            120, 80, 0.05, truth, 0.25, observation_sd=0.2, initial_sd=0.08, rng=rng,
        )
        estimate = fit_ou_moments(
            observations, 0.05, max_lag=14, n_boot=120,
            rng=np.random.default_rng(int(rng.integers(0, 2**32 - 1))),
        )
        if estimate.lambda_ci is None:
            continue
        estimable += 1
        covered += int(estimate.lambda_ci[0] <= truth <= estimate.lambda_ci[1])
    coverage = covered / estimable if estimable else np.nan
    return {
        "n_repetitions": n_repetitions,
        "n_estimable": estimable,
        "lambda_interval_coverage": float(coverage),
        "acceptance_interval": [0.75, 1.0],
        "pass": bool(estimable == n_repetitions and 0.75 <= coverage <= 1.0),
    }


def main() -> None:
    rng = np.random.default_rng(20260731)
    scenarios: dict[str, dict] = {}

    stationary = rng.normal(scale=0.35, size=(180, 80))
    stationary_fit = fit_gaussian_state_space(stationary, 0.05)
    scenarios["stationary_code"] = {
        "estimate": stationary_fit.to_dict(),
        "pass": bool(
            stationary_fit.status == "not_identifiable"
            and stationary_fit.identifiability is not None
            and stationary_fit.identifiability.lambda_dt > 0.25
        ),
        "interpretation": "white observations cannot identify a separate fast latent process; no finite drift claim is licensed",
    }

    scenarios["deterministic_rotation"] = rotation_recovery(rng)

    _, drift_observations = simulate_confined_diffusion(
        220, 90, 0.05, 1.1, 0.3, observation_sd=0.25, initial_sd=0.08, rng=rng,
    )
    drift_state = fit_gaussian_state_space(drift_observations, 0.05)
    drift_moment = fit_ou_moments(
        drift_observations, 0.05, max_lag=15, n_boot=300,
        rng=np.random.default_rng(310),
    )
    scenarios["confined_diffusion"] = {
        "truth": {"lambda_rate": 1.1, "diffusion": 0.3},
        "state_space": drift_state.to_dict(),
        "moment": drift_moment.to_dict(),
        "pass": bool(
            relative_error(drift_state.lambda_rate, 1.1) < 0.4
            and relative_error(drift_state.diffusion, 0.3) < 0.45
            and relative_error(drift_moment.lambda_rate, 1.1) < 0.4
            and relative_error(drift_moment.diffusion, 0.3) < 0.45
        ),
    }

    _, combined_residuals = simulate_confined_diffusion(
        220, 90, 0.05, 0.9, 0.22, observation_sd=0.2, initial_sd=0.08, rng=rng,
    )
    combined_state = fit_gaussian_state_space(combined_residuals, 0.05)
    combined_rotation = rotation_recovery(rng)
    scenarios["rotation_and_diffusion"] = {
        "state_space": combined_state.to_dict(),
        "rotation": combined_rotation,
        "pass": bool(
            combined_rotation["pass"]
            and relative_error(combined_state.lambda_rate, 0.9) < 0.4
            and relative_error(combined_state.diffusion, 0.22) < 0.45
        ),
    }

    scenarios["switching"] = switching_comparison(rng)

    _, biased = simulate_confined_diffusion(
        220, 90, 0.05, 1.0, 0.25, equilibrium=0.6,
        observation_sd=0.2, initial_sd=0.08, rng=rng,
    )
    biased_fit = fit_gaussian_state_space(biased, 0.05)
    scenarios["biased_drift"] = {
        "truth": {"equilibrium": 0.6}, "estimate": biased_fit.to_dict(),
        "pass": bool(
            biased_fit.equilibrium is not None
            and abs(biased_fit.equilibrium - 0.6) < 0.18
        ),
    }

    measurement_only = rng.normal(scale=0.4, size=(220, 90))
    measurement_fit = fit_gaussian_state_space(measurement_only, 0.05)
    scenarios["measurement_noise_only"] = {
        "estimate": measurement_fit.to_dict(),
        "pass": bool(
            measurement_fit.status == "not_identifiable"
            and measurement_fit.lambda_ci is not None
            and measurement_fit.lambda_ci[0] < 0 < measurement_fit.lambda_ci[1]
        ),
        "interpretation": "process and observation noise are not separately identified when the fitted latent timescale is below one bin",
    }

    smoothed = gaussian_filter1d(drift_observations, sigma=1.5, axis=1, mode="nearest")
    smoothed_fit = fit_ou_moments(
        smoothed, 0.05, max_lag=15, n_boot=300,
        rng=np.random.default_rng(311),
    )
    scenarios["smoothing_failure"] = {
        "unsmoothed_lambda": drift_moment.lambda_rate,
        "smoothed_lambda": smoothed_fit.lambda_rate,
        "unsmoothed_observation_variance": drift_moment.observation_variance,
        "smoothed_observation_variance": smoothed_fit.observation_variance,
        "unsmoothed_diffusion": drift_moment.diffusion,
        "smoothed_diffusion": smoothed_fit.diffusion,
        "lag_zero_excluded": drift_moment.diagnostics.get("lag_zero_excluded"),
        "separation_destroyed": bool(
            drift_moment.observation_variance is not None
            and smoothed_fit.observation_variance is not None
            and drift_moment.observation_variance > 0
            and smoothed_fit.observation_variance < 0.1 * drift_moment.observation_variance
        ),
        "pass": bool(
            smoothed_fit.lambda_rate is not None
            and drift_moment.lambda_rate is not None
            and smoothed_fit.observation_variance < 0.1 * drift_moment.observation_variance
            and drift_moment.observation_variance > 0
        ),
    }

    coverage = coverage_check(rng)
    all_pass = bool(all(scenario["pass"] for scenario in scenarios.values()) and coverage["pass"])
    output = {
        "schema_version": "1.0.0",
        "seed": 20260731,
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "scenarios": scenarios,
        "coverage": coverage,
        "all_pass": all_pass,
        "real_data_authorized": all_pass,
    }
    destination = ROOT / "results" / "drift_simulation_gate.json"
    destination.write_text(canonical_json(output))
    print(json.dumps({
        "all_pass": all_pass,
        "scenario_passes": {name: value["pass"] for name, value in scenarios.items()},
        "coverage": coverage,
        "output": str(destination.relative_to(ROOT)),
    }, indent=2))
    if not all_pass:
        raise SystemExit("drift estimator simulation gate failed; real-data fitting remains blocked")


if __name__ == "__main__":
    main()
