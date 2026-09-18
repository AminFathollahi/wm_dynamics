#!/usr/bin/env python3
"""Calibrate trial-to-outcome permutation nulls on data with known truth.

This is a simulation audit of the generic engine in
``run_within_session_permutation_control.py``.  It leaves that engine and every result artifact
untouched.  The audit measures rejection rates for its unrestricted outcome shuffle under iid and
serially correlated nulls, a coupled positive control, and a nuisance-only partial-correlation
null.  It also evaluates two candidate alternatives in the regimes their assumptions address:
circular outcome shifts for stationary AR(1) series, and outcome-residual permutation conditional
on measured nuisance covariates with iid residuals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _path = str(ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from provenance import _json_safe, git_commit  # noqa: E402
from statistics import permutation_pvalue, stable_seed  # noqa: E402
import run_within_session_permutation_control as current_engine  # noqa: E402


OUTPUT_PATH = ROOT / "results" / "trial_exchangeability_calibration.json"
N_REPLICATIONS = 200
N_PERMUTATIONS = 199
N_SESSIONS = 11
N_TRIALS = 100
ALPHA = 0.05
AR1_PHI = 0.9
COUPLING = 0.45


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray] | None = None) -> dict:
    return {
        "n": len(x), "predictor": np.asarray(x, dtype=float), "outcome": np.asarray(y, dtype=float),
        "controls": [] if controls is None else [np.asarray(value, dtype=float) for value in controls],
    }


def _stationary_ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    values = np.empty(n, dtype=float)
    values[0] = rng.normal()
    innovation_scale = math.sqrt(1.0 - phi ** 2)
    for index in range(1, n):
        values[index] = phi * values[index - 1] + innovation_scale * rng.normal()
    return values


def _simulate_iid_null(rng: np.random.Generator) -> list[list[dict]]:
    return [[_block(rng.normal(size=N_TRIALS), rng.normal(size=N_TRIALS))]
            for _ in range(N_SESSIONS)]


def _simulate_ar1_null(rng: np.random.Generator) -> list[list[dict]]:
    return [[_block(_stationary_ar1(rng, N_TRIALS, AR1_PHI), _stationary_ar1(rng, N_TRIALS, AR1_PHI))]
            for _ in range(N_SESSIONS)]


def _simulate_coupling(rng: np.random.Generator) -> list[list[dict]]:
    blocks = []
    for _ in range(N_SESSIONS):
        x = rng.normal(size=N_TRIALS)
        y = COUPLING * x + rng.normal(scale=math.sqrt(1.0 - COUPLING ** 2), size=N_TRIALS)
        blocks.append([_block(x, y)])
    return blocks


def _simulate_nuisance_only(rng: np.random.Generator) -> list[list[dict]]:
    blocks = []
    for _ in range(N_SESSIONS):
        z = rng.normal(size=N_TRIALS)
        x = z + rng.normal(size=N_TRIALS)
        y = z + rng.normal(size=N_TRIALS)
        blocks.append([_block(x, y, [z])])
    return blocks


def _simulate_binary_ar1_null(rng: np.random.Generator) -> list[list[dict]]:
    blocks = []
    for _ in range(N_SESSIONS):
        x = _stationary_ar1(rng, N_TRIALS, AR1_PHI)
        latent_outcome = _stationary_ar1(rng, N_TRIALS, AR1_PHI)
        y = (latent_outcome > np.quantile(latent_outcome, 0.8)).astype(float)
        blocks.append([_block(x, y)])
    return blocks


def _simulate_serial_nuisance_null(rng: np.random.Generator) -> list[list[dict]]:
    blocks = []
    for _ in range(N_SESSIONS):
        z = _stationary_ar1(rng, N_TRIALS, 0.8)
        x = 0.8 * z + _stationary_ar1(rng, N_TRIALS, AR1_PHI)
        y = -0.6 * z + _stationary_ar1(rng, N_TRIALS, AR1_PHI)
        blocks.append([_block(x, y, [z])])
    return blocks


def _unrestricted_outcome_shuffle(rng: np.random.Generator, block: dict) -> dict:
    return {**block, "outcome": block["outcome"][rng.permutation(block["n"])]}


def _residual_outcome_shuffle(rng: np.random.Generator, block: dict) -> dict:
    """Shuffle iid outcome residuals after conditioning on the supplied nuisance covariates."""
    controls = block["controls"]
    if not controls:
        return _unrestricted_outcome_shuffle(rng, block)
    design = np.column_stack([np.ones(block["n"]), *controls])
    fitted = design @ np.linalg.lstsq(design, block["outcome"], rcond=None)[0]
    residual = block["outcome"] - fitted
    return {**block, "outcome": fitted + residual[rng.permutation(block["n"])]}


def _replica_test(session_blocks: list[list[dict]], seed_tag: str, n_perm: int,
                  shuffle_block=_unrestricted_outcome_shuffle) -> dict:
    """Small transparent replica of the current engine, used only to establish equivalence."""
    rng = np.random.default_rng(stable_seed(seed_tag))
    observed_by_session = [current_engine._default_stat_of_session(blocks) for blocks in session_blocks]
    observed_by_session = [value for value in observed_by_session if value is not None]
    if len(observed_by_session) < 2:
        return {"status": "not_computable", "n_sessions_contributing": len(observed_by_session)}
    observed = float(np.mean(observed_by_session))
    null = np.full(n_perm, np.nan)
    for draw in range(n_perm):
        values = []
        for blocks in session_blocks:
            shuffled = [shuffle_block(rng, block) for block in blocks]
            value = current_engine._default_stat_of_session(shuffled)
            if value is not None:
                values.append(value)
        if values:
            null[draw] = float(np.mean(values))
    valid = np.isfinite(null)
    p_value = permutation_pvalue(np.abs(null[valid]) >= abs(observed))
    return {
        "status": "computed", "n_sessions_contributing": len(observed_by_session),
        "observed_pooled_statistic": observed, "n_permutations_requested": n_perm,
        "n_permutations_valid": int(valid.sum()), "p_value": p_value,
        "significant": bool(p_value < ALPHA),
    }


def _equivalence_check() -> dict:
    rng = np.random.default_rng(90210)
    blocks = _simulate_nuisance_only(rng)
    seed_tag = "trial_exchangeability_calibration|equivalence"
    original = current_engine.within_session_permutation_test(
        blocks, seed_tag, n_perm=59, shuffle_block=_unrestricted_outcome_shuffle)
    replica = _replica_test(blocks, seed_tag, n_perm=59)
    compared = ("n_sessions_contributing", "observed_pooled_statistic", "n_permutations_requested",
                "n_permutations_valid", "p_value", "significant")
    exact = all(original[key] == replica[key] for key in compared)
    if not exact:
        raise AssertionError(f"replica differs from current engine: original={original}, replica={replica}")
    return {"status": "passed", "n_sessions": N_SESSIONS, "n_trials_per_session": N_TRIALS,
            "n_permutations": 59, "exact_fields": list(compared), "original": original}


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    phat = successes / n
    denominator = 1.0 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denominator
    half_width = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denominator
    return [center - half_width, center + half_width]


def _calibrate(name: str, simulator, shuffle_block, n_replications: int, n_permutations: int) -> dict:
    rejections = 0
    statistics = []
    p_values = []
    started = time.time()
    for replicate in range(n_replications):
        generator = np.random.default_rng(stable_seed(f"trial_exchangeability_calibration|{name}|replicate={replicate}"))
        result = current_engine.within_session_permutation_test(
            simulator(generator), f"trial_exchangeability_calibration|{name}|permutation={replicate}",
            n_perm=n_permutations, shuffle_block=shuffle_block)
        if result["status"] != "computed":
            raise AssertionError(f"synthetic regime did not compute: {name}, {result}")
        rejections += int(result["significant"])
        statistics.append(result["observed_pooled_statistic"])
        p_values.append(result["p_value"])
    return {
        "status": "complete", "regime": name, "n_replications": n_replications,
        "n_permutations_per_replication": n_permutations, "alpha": ALPHA,
        "rejections": rejections, "rejection_rate": rejections / n_replications,
        "rejection_rate_wilson_95": _wilson_interval(rejections, n_replications),
        "mean_observed_pooled_statistic": float(np.mean(statistics)),
        "mean_permutation_p_value": float(np.mean(p_values)),
        "minimum_attainable_p_value": 1.0 / (n_permutations + 1),
        "wall_clock_s": round(time.time() - started, 3),
    }


def run_audit(n_replications: int = N_REPLICATIONS, n_permutations: int = N_PERMUTATIONS) -> dict:
    """Run fixed, data-free calibration regimes and return a serializable artifact."""
    started = time.time()
    equivalence = _equivalence_check()
    results = {
        "unrestricted_outcome_shuffle": {
            "iid_independent_null": _calibrate("iid_independent_null", _simulate_iid_null,
                                                 _unrestricted_outcome_shuffle, n_replications, n_permutations),
            "stationary_ar1_independent_null": _calibrate("stationary_ar1_independent_null", _simulate_ar1_null,
                                                            _unrestricted_outcome_shuffle, n_replications,
                                                            n_permutations),
            "binary_stationary_ar1_independent_null": _calibrate(
                "binary_stationary_ar1_independent_null", _simulate_binary_ar1_null,
                _unrestricted_outcome_shuffle, n_replications, n_permutations),
            "true_coupling_positive_control": _calibrate("true_coupling_positive_control", _simulate_coupling,
                                                           _unrestricted_outcome_shuffle, n_replications,
                                                           n_permutations),
            "nuisance_only_partial_correlation_null": _calibrate("nuisance_only_partial_correlation_null",
                                                                    _simulate_nuisance_only,
                                                                    _unrestricted_outcome_shuffle,
                                                                    n_replications, n_permutations),
        },
        "candidate_nulls": {
            "circular_outcome_shift_on_stationary_ar1": _calibrate("circular_outcome_shift_on_stationary_ar1",
                                                                      _simulate_ar1_null, current_engine._circular_outcome_shift,
                                                                      n_replications, n_permutations),
            "circular_outcome_shift_on_binary_stationary_ar1": _calibrate(
                "circular_outcome_shift_on_binary_stationary_ar1", _simulate_binary_ar1_null,
                current_engine._circular_outcome_shift, n_replications, n_permutations),
            "residual_outcome_shuffle_on_nuisance_only_partial_null": _calibrate(
                "residual_outcome_shuffle_on_nuisance_only_partial_null", _simulate_nuisance_only,
                _residual_outcome_shuffle, n_replications, n_permutations),
            "circular_residual_shift_on_serial_nuisance_partial_null": _calibrate(
                "circular_residual_shift_on_serial_nuisance_partial_null", _simulate_serial_nuisance_null,
                current_engine._circular_residual_shift, n_replications, n_permutations),
        },
    }
    return {
        "analysis_id": "trial_exchangeability_calibration",
        "schema_version": "1.0.0",
        "status": "complete",
        "purpose": "simulation calibration of a null mechanism; it does not diagnose or alter any empirical result",
        "current_engine": {
            "path": "scripts/run_within_session_permutation_control.py",
            "sha256": _sha256(ROOT / "scripts" / "run_within_session_permutation_control.py"),
            "function": "within_session_permutation_test",
            "statistic": "unweighted mean of within-session partial correlations; unrestricted shuffle permutes outcome trial labels independently within session",
        },
        "producer": {"path": "scripts/run_trial_exchangeability_calibration.py",
                     "sha256": _sha256(Path(__file__)), "code_commit": git_commit(ROOT)},
        "fixed_regimes": {
            "n_sessions": N_SESSIONS, "n_trials_per_session": N_TRIALS, "n_replications": n_replications,
            "n_permutations_per_replication": n_permutations, "alpha": ALPHA, "ar1_phi": AR1_PHI,
            "coupling": COUPLING, "seeds": "stable_seed with regime and replicate labels",
        },
        "engine_equivalence": equivalence,
        "results": results,
        "candidate_assumptions": {
            "circular_outcome_shift": (
                "Evaluated only for stationary AR(1) series. A circular shift preserves each series' serial "
                "structure but is not universally exact for a finite noncircular AR(1) realization because the "
                "wraparound is an added circular-stationarity assumption."
            ),
            "residual_outcome_shuffle": (
                "Evaluated only for the nuisance-only partial-correlation regime with iid residual errors and "
                "a measured linear nuisance covariate. It retains the reduced model's fitted nuisance component "
                "before adding permuted residuals; its validity is not established here for serially dependent "
                "or misspecified residuals."
            ),
            "circular_residual_shift": (
                "Evaluated for a linear nuisance model with stationary autocorrelated residuals. It retains the "
                "reduced model's fitted nuisance component and circularly shifts its residual series. This adds "
                "the same finite-series circular-stationarity assumption as the outcome-shift test."
            ),
        },
        "interpretation_scope": (
            "Rejection rates reveal potential failure modes of a null under the listed generative assumptions. "
            "They do not automatically withdraw, confirm, or diagnose any data-specific empirical effect."
        ),
        "wall_clock_s": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=N_REPLICATIONS)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.replications < 200 or args.permutations < 199:
        raise ValueError("the calibration requires at least 200 replications and 199 permutations")
    artifact = run_audit(args.replications, args.permutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(artifact), indent=2, allow_nan=False, default=float))
    print(f"wrote {args.output} in {artifact['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()
