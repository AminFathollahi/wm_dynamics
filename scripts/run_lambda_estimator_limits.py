#!/usr/bin/env python3
"""Characterises what `fit_gaussian_state_space` and `confinement_identifiability`
(src/drift_dynamics.py) do under three latent-signal regimes, at the real
DANDI 000469 delay-period dimensions: true confinement, a true random walk
(lambda = 0 exactly), and no latent signal at all (independent noise, no
shared trajectory).

The two failure regimes -- no confinement (a random walk) and no signal --
are mechanistically different but both leave the confinement estimate
unsupported. Whether they leave a different FINGERPRINT in lambda-hat is
what would let a real non-identified fold be read as "this circuit's delay
state is close to a random walk" versus "this circuit's units are too noisy
to tell". That fingerprint is established here, by simulation, before any
such reading is applied to real data.

No estimator logic is reimplemented: `fit_gaussian_state_space` and
`confinement_identifiability` are imported and called directly on simulated
observation trajectories.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import bootstrap_ci, permutation_test_twosample  # noqa: E402
from drift_dynamics import confinement_identifiability, fit_gaussian_state_space  # noqa: E402
import run_human_drift_spine_000469 as spine469  # noqa: E402
from spike_pipeline import (  # noqa: E402
    ANATOMICAL_REGIONS, load_spike_times, low_rate_unit_mask, filter_units_by_region,
    resolve_unit_regions,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "lambda_estimator_limits.json"
SEED = 20260807

# Real DANDI 000469 delay-period dimensions (run_human_drift_spine_000469.py: BIN_MS=100,
# WINDOW_S=2.3 -> 23 bins of 0.1s; fit_gaussian_state_space is called on each fold's
# ~4/5-of-45-trial training split). Representative point = observed medians across the five
# co-recorded structures (results/region_stratified_drift_000469.json#regions.*.sessions).
DT_S = 0.1
N_TIME_BINS = 23  # 2.3 s window
REPRESENTATIVE_N_UNITS = 16
REPRESENTATIVE_N_TRIALS = 36
REPRESENTATIVE_RATE_HZ = 1.5
TRUE_LAMBDAS = [0.5, 1.0, 1.65, 2.1, 3.0]
# Stationary variance D/lambda ~ 0.11 at the representative lambda, matching the
# 0.105-0.119 range already observed across structures in structure_control_observables.json.
DIFFUSION_D = 0.18
# Declared, not fit to data: observation-noise variance r = R0 / (n_units * rate * dt), i.e.
# inversely proportional to the population's expected per-bin spike count -- more units or a
# higher rate means a less noisy population-average estimate of the latent trajectory. R0 is
# chosen so the representative point above sits at a moderate (not floor/ceiling) identified
# fraction, giving the one-factor-at-a-time nuisance sweep room to move in both directions.
OBSERVATION_NOISE_R0 = 1.5
N_REPS = 200
N_WORKERS = 28


def _observation_noise_variance(n_units: int, rate_hz: float, dt: float) -> float:
    return OBSERVATION_NOISE_R0 / (n_units * rate_hz * dt)


def _simulate(regime: str, lam: float, diffusion: float, obs_noise_var: float,
              n_trials: int, n_time: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((n_trials, n_time))
    if regime == "confined":
        phi = np.exp(-lam * dt)
        stationary_variance = diffusion / (2.0 * lam)
        sigma = np.sqrt(diffusion * (1.0 - phi ** 2) / (2.0 * lam))
        cur = rng.normal(scale=np.sqrt(stationary_variance), size=n_trials)
        for t in range(n_time):
            cur = phi * cur + sigma * rng.normal(size=n_trials)
            x[:, t] = cur
    elif regime == "random_walk":
        sigma = np.sqrt(diffusion * dt)
        cur = np.zeros(n_trials)
        for t in range(n_time):
            cur = cur + sigma * rng.normal(size=n_trials)
            x[:, t] = cur
    elif regime == "no_signal":
        pass  # x stays identically zero: no latent trajectory of any kind
    else:
        raise ValueError(regime)
    return x + rng.normal(scale=np.sqrt(obs_noise_var), size=(n_trials, n_time))


def _fit_one(args: tuple) -> dict:
    regime, lam, diffusion, obs_noise_var, n_trials, n_time, dt, seed = args
    rng = np.random.default_rng(seed)
    y = _simulate(regime, lam, diffusion, obs_noise_var, n_trials, n_time, dt, rng)
    estimate = fit_gaussian_state_space(y, dt)
    return {"status": estimate.status, "lambda_rate": estimate.lambda_rate}


def _run_cell(regime: str, lam: float, n_units: int, rate_hz: float, n_trials: int, n_time: int,
              dt: float, diffusion: float, n_reps: int, seed_offset: int, pool: Pool) -> list[dict]:
    obs_noise_var = _observation_noise_variance(n_units, rate_hz, dt)
    args = [
        (regime, lam, diffusion, obs_noise_var, n_trials, n_time, dt, SEED + seed_offset + i)
        for i in range(n_reps)
    ]
    return pool.map(_fit_one, args)


def _lambda_bounds(dt: float) -> tuple[float, float]:
    """The hard optimizer bounds on lambda_rate, translated from the log_a bounds in
    src/drift_dynamics.py's `fit_gaussian_state_space` (log_a in [log(0.005), log(2.0)])."""
    return float(-np.log(2.0) / dt), float(-np.log(0.005) / dt)


def _fingerprint(results: list[dict], dt: float) -> dict:
    lower_bound, upper_bound = _lambda_bounds(dt)
    tol = 0.02 * (upper_bound - lower_bound)
    finite = [r["lambda_rate"] for r in results if r["lambda_rate"] is not None and np.isfinite(r["lambda_rate"])]
    status_counts = dict(Counter(r["status"] for r in results))
    out = {
        "n_reps": len(results), "status_counts": status_counts,
        "identified_fraction": status_counts.get("identifiable", 0) / len(results),
        "n_finite_lambda_hat": len(finite),
    }
    if finite:
        arr = np.asarray(finite)
        out.update({
            "median_lambda_hat": float(np.median(arr)),
            "lambda_hat_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "fraction_near_zero": float(np.mean(np.abs(arr) < 0.1)),
            "fraction_at_lower_bound": float(np.mean(arr <= lower_bound + tol)),
            "fraction_at_upper_bound": float(np.mean(arr >= upper_bound - tol)),
            "fraction_interior": float(np.mean((arr > lower_bound + tol) & (arr < upper_bound - tol) & (np.abs(arr) >= 0.1))),
        })
    else:
        out.update({
            "median_lambda_hat": None, "lambda_hat_ci95": [None, None],
            "fraction_near_zero": None, "fraction_at_lower_bound": None,
            "fraction_at_upper_bound": None, "fraction_interior": None,
        })
    return out


def main() -> None:
    identifiability_criterion = {
        "function": "confinement_identifiability",
        "file_line": "src/drift_dynamics.py:191-221",
        "called_from": (
            "fit_gaussian_state_space, src/drift_dynamics.py (~line 368): "
            "confinement_identifiability(float(lambda_rate), dt, dt * (y.shape[1] - 1))"
        ),
        "rule_verbatim": (
            "if not finite(lambda_rate): 'not_identifiable' (non-finite confinement estimate). "
            "elif lambda_rate <= 0: 'unconfined' (estimated confinement is nonpositive). "
            "elif lambda_rate * dt > max_lambda_dt (0.25): 'not_identifiable' (decay too fast "
            "relative to bin width). "
            "elif lambda_rate * duration < min_time_constants (2.0): 'not_identifiable' (delay "
            "spans fewer than two estimated time constants). "
            "else: 'identifiable'."
        ),
        "thresholds": {"max_lambda_dt": 0.25, "min_time_constants": 2.0},
        "thresholds_hard_coded": True,
        "thresholds_provenance": (
            "None declared in source: both are unlabelled default arguments with no citation, "
            "derivation, or calibration reference in the function's docstring or callers. This is "
            "a defect; see the crack register entry "
            "'confinement_identifiability_threshold_no_provenance'."
        ),
        "consequence_for_this_round": (
            f"at DT_S={DT_S}, N_TIME_BINS={N_TIME_BINS} (duration={DT_S * (N_TIME_BINS - 1)}s), "
            f"the resolvable window is lambda in "
            f"[{2.0 / (DT_S * (N_TIME_BINS - 1)):.3f}, {0.25 / DT_S:.3f}] s^-1 by construction, "
            "independent of noise -- true lambda outside that range cannot register as "
            "'identifiable' even with a perfect fit. TRUE_LAMBDAS below include values inside "
            "and outside that window on purpose, to show the ceiling effect is real."
        ),
    }

    with Pool(N_WORKERS) as pool:
        # 2.2: the three limits, >= 200 reps per cell, at the representative real dimensions.
        limit_confined = {}
        for i, lam in enumerate(TRUE_LAMBDAS):
            results = _run_cell(
                "confined", lam, REPRESENTATIVE_N_UNITS, REPRESENTATIVE_RATE_HZ,
                REPRESENTATIVE_N_TRIALS, N_TIME_BINS, DT_S, DIFFUSION_D, N_REPS,
                seed_offset=1000 * (i + 1), pool=pool,
            )
            limit_confined[str(lam)] = {"planted_lambda": lam, **_fingerprint(results, DT_S), "raw_results": results}

        limit_random_walk_results = _run_cell(
            "random_walk", 0.0, REPRESENTATIVE_N_UNITS, REPRESENTATIVE_RATE_HZ,
            REPRESENTATIVE_N_TRIALS, N_TIME_BINS, DT_S, DIFFUSION_D, N_REPS,
            seed_offset=90000, pool=pool,
        )
        limit_random_walk = _fingerprint(limit_random_walk_results, DT_S)

        limit_no_signal_results = _run_cell(
            "no_signal", 0.0, REPRESENTATIVE_N_UNITS, REPRESENTATIVE_RATE_HZ,
            REPRESENTATIVE_N_TRIALS, N_TIME_BINS, DT_S, DIFFUSION_D, N_REPS,
            seed_offset=91000, pool=pool,
        )
        limit_no_signal = _fingerprint(limit_no_signal_results, DT_S)

        # 2.3: do the two failure fingerprints differ?
        rw_finite = np.array([r["lambda_rate"] for r in limit_random_walk_results if r["lambda_rate"] is not None and np.isfinite(r["lambda_rate"])])
        ns_finite = np.array([r["lambda_rate"] for r in limit_no_signal_results if r["lambda_rate"] is not None and np.isfinite(r["lambda_rate"])])
        if len(rw_finite) >= 5 and len(ns_finite) >= 5:
            observed_diff, p_value = permutation_test_twosample(
                rw_finite, ns_finite, stat_fn=lambda a, b: np.median(a) - np.median(b),
                n_perm=5000, rng=np.random.default_rng(SEED + 1),
            )
            median_test = {
                "status": "estimable",
                "observed_median_lambda_hat_diff_random_walk_minus_no_signal": observed_diff,
                "p_value": p_value,
            }
            fingerprints_separable = bool(p_value < 0.05)
        else:
            median_test = {"status": "non_identified", "reason": "fewer than 5 finite lambda_hat in one or both regimes"}
            fingerprints_separable = None

        fingerprint_comparison = {
            "random_walk": limit_random_walk, "no_signal": limit_no_signal,
            "median_permutation_test_lambda_hat_random_walk_vs_no_signal": median_test,
            "fingerprints_separable": fingerprints_separable,
            "interpretation": (
                "fingerprints differ (separable): a non-identified real fold can be classified "
                "toward one mechanism or the other by comparing its lambda_hat to these two "
                "reference distributions."
                if fingerprints_separable else
                "fingerprints do NOT separate (or cannot be tested): the identifiability "
                "dissociation is not separable into a mechanism by this route with the current "
                "estimator and window; it is reported without a mechanism, and a crack is filed "
                "naming what would separate them (e.g. a non-Gaussian/switching-aware estimator, "
                "or a longer delay window)."
                if fingerprints_separable is False else
                "insufficient finite lambda_hat draws in one or both regimes to test separability."
            ),
        }

        # 2.4: selection bias among identified fits in the true-confinement limit.
        bias_among_identified = {}
        for lam_str, cell in limit_confined.items():
            identified_lambdas = [
                r["lambda_rate"] for r in cell["raw_results"]
                if r["status"] == "identifiable" and r["lambda_rate"] is not None
            ]
            lam = cell["planted_lambda"]
            if len(identified_lambdas) < 5:
                bias_among_identified[lam_str] = {
                    "status": "non_identified", "n_identified": len(identified_lambdas),
                    "reason": "fewer than 5 identified reps to assess bias",
                }
                continue
            arr = np.asarray(identified_lambdas)
            _, lo, hi = bootstrap_ci(arr, np.median, n_boot=5000, rng=np.random.default_rng(SEED + 2))
            bias_among_identified[lam_str] = {
                "status": "estimable", "n_identified": len(arr), "planted_lambda": lam,
                "median_lambda_hat_given_identified": float(np.median(arr)),
                "bias_vs_truth": float(np.median(arr) - lam),
                "bootstrap_ci95_median": [float(lo), float(hi)],
                "bias_ci95": [float(lo - lam), float(hi - lam)],
                "biased": bool(lo > lam or hi < lam),
            }

        # 2.5: one-factor-at-a-time nuisance sweep at fixed true lambda = 1.65.
        NUISANCE_LAMBDA = 1.65
        n_units_grid = [4, 8, 16, 32, 64]
        rate_grid = [1.0, 2.0, 5.0, 10.0, 20.0]
        n_trials_grid = [20, 40, 80, 160]
        # Delay lengths actually present across the corpora this project fits (000469 /
        # 001187+000673 = 2.3s; 000574 Boran = 3.0s), both at the shared 0.1s bin width.
        delay_length_grid_s = [2.3, 3.0]

        def _sweep(factor_name: str, grid: list, seed_base: int) -> dict:
            out = {}
            for i, level in enumerate(grid):
                n_units = level if factor_name == "n_units" else REPRESENTATIVE_N_UNITS
                rate = level if factor_name == "rate_hz" else REPRESENTATIVE_RATE_HZ
                n_trials = level if factor_name == "n_trials" else REPRESENTATIVE_N_TRIALS
                n_time = int(round(level / DT_S)) if factor_name == "delay_length_s" else N_TIME_BINS
                results = _run_cell(
                    "confined", NUISANCE_LAMBDA, n_units, rate, n_trials, n_time, DT_S,
                    DIFFUSION_D, N_REPS, seed_offset=seed_base + 1000 * i, pool=pool,
                )
                out[str(level)] = _fingerprint(results, DT_S)
                out[str(level)].pop("raw_results", None)
            return out

        nuisance_sweep = {
            "fixed_true_lambda": NUISANCE_LAMBDA,
            "fixed_other_dimensions": {
                "n_units": REPRESENTATIVE_N_UNITS, "rate_hz": REPRESENTATIVE_RATE_HZ,
                "n_trials": REPRESENTATIVE_N_TRIALS, "delay_length_s": DT_S * (N_TIME_BINS - 1),
                "bin_width_s": DT_S,
            },
            "n_units": _sweep("n_units", n_units_grid, 200000),
            "rate_hz": _sweep("rate_hz", rate_grid, 210000),
            "n_trials": _sweep("n_trials", n_trials_grid, 220000),
            "delay_length_s": _sweep("delay_length_s", delay_length_grid_s, 230000),
        }

    for cell in limit_confined.values():
        cell.pop("raw_results", None)

    output = {
        "schema_version": "1.0.0", "analysis_id": "lambda_estimator_limits",
        "trigger": "a ~30x identifiability gap between hippocampus and pre-SMA at matched unit count -- is it biological or an SNR/estimator artifact?",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "scope": (
            "Simulation only, no real-data analysis this artifact. Dimensions (bin width, delay "
            "length, representative n_units/n_trials/rate) are read from DANDI 000469's delay-"
            "period drift fit (run_human_drift_spine_000469.py) and "
            "results/region_stratified_drift_000469.json; the observation-noise scale constant "
            "is a declared modelling choice, not fit to any real corpus."
        ),
        "identifiability_criterion": identifiability_criterion,
        "simulation_dimensions": {
            "dt_s": DT_S, "n_time_bins": N_TIME_BINS, "delay_length_s": DT_S * (N_TIME_BINS - 1),
            "representative_n_units": REPRESENTATIVE_N_UNITS,
            "representative_n_trials": REPRESENTATIVE_N_TRIALS,
            "representative_rate_hz": REPRESENTATIVE_RATE_HZ,
            "diffusion_d": DIFFUSION_D,
            "observation_noise_r0": OBSERVATION_NOISE_R0,
            "observation_noise_model": "r = R0 / (n_units * rate_hz * dt); declared, not fit to data",
            "n_reps_per_cell": N_REPS,
        },
        "true_confinement_present": limit_confined,
        "fingerprint_comparison_random_walk_vs_no_signal": fingerprint_comparison,
        "bias_among_identified_fits": bias_among_identified,
        "nuisance_sweep_one_factor_at_a_time": nuisance_sweep,
    }
    OUTPUT_PATH.write_text(canonical_json(output))

    crack_path = RESULTS / "crack_register.json"
    cracks = json.loads(crack_path.read_text())
    if not any(entry.get("crack_id") == "confinement_identifiability_threshold_no_provenance" for entry in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "confinement_identifiability_threshold_no_provenance",
            "trigger": (
                "confinement_identifiability (src/drift_dynamics.py:191-221) hard-codes "
                "max_lambda_dt=0.25 and min_time_constants=2.0 as default arguments with no "
                "citation, derivation, or calibration reference anywhere in the function, its "
                "docstring, or its only caller (fit_gaussian_state_space)."
            ),
            "chase": (
                "Documented the thresholds verbatim with file:line in "
                "results/lambda_estimator_limits.json#identifiability_criterion and reported "
                "their consequence at the real 000469 delay dimensions (dt=0.1s, 23 bins): "
                "lambda outside roughly [0.91, 2.5] s^-1 cannot register as identifiable "
                "regardless of noise, a deterministic ceiling effect distinct from a noise "
                "failure."
            ),
            "resolution": (
                "Not resolved here -- recorded, not fixed. Choosing or deriving a "
                "principled threshold (e.g. from a formal Fisher-information / CRLB argument, "
                "or from the calibration-style planted-effect recovery check used elsewhere in "
                "this project) is future work."
            ),
            "status": "open",
            "artifact": "src/drift_dynamics.py, results/lambda_estimator_limits.json",
        })
    crack_path.write_text(canonical_json(cracks))

    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "identified_fraction_by_lambda": {k: v["identified_fraction"] for k, v in limit_confined.items()},
        "random_walk_identified_fraction": limit_random_walk["identified_fraction"],
        "no_signal_identified_fraction": limit_no_signal["identified_fraction"],
        "fingerprints_separable": fingerprints_separable,
    }, indent=2))


if __name__ == "__main__":
    main()
