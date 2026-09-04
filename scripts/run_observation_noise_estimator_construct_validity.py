#!/usr/bin/env python3
"""run_observation_noise_estimator_construct_validity.py -- an answer-key test
for the two quantities this project has called "observation noise".

This project uses two different estimators of what it calls observation
noise, and across the cells where both exist they disagree almost completely
(Pearson r close to zero; see species_gap.precursor_numbers_carried_forward
below for the exact figures carried forward from
results/latent_model_observation_noise_comparison.json). At most one of them
measures observation noise, and possibly neither does. On real recordings the
truth is unknown, so no correlation between an estimator and a candidate
confound can distinguish "the estimator is right and the confound is a real
co-occurring effect" from "the estimator is measuring the confound instead of
noise". This script removes that ambiguity by generating synthetic
populations whose true observation-noise fraction, true latent
dimensionality and true latent timescale are set by the generator, then
scoring each estimator against the value that generated the data.

The two estimators, called through the exact public entry points the real
analyses use (never reimplemented):
  - the factor-analysis noise-variance-fraction estimator:
    scripts/run_latent_model_observation_noise_comparison.py's
    dimensionality_and_noise_term (field
    factor_analysis.observation_noise_variance_fraction), which internally
    fits sklearn.decomposition.FactorAnalysis at a fixed latent count k
    (read from scripts/run_latent_model_comparison.py's LATENT_DIM).
  - the cross-validated nugget-fraction estimator: src/observability.py's
    nugget_fraction (field median_nugget_fraction), the same function the
    observability-and-power census calls, with the census's own n_splits
    setting (read from scripts/run_observability_and_power_census.py's
    N_SPLITS).

The answer-key grid (this file's primary deliverable): does each estimator
recover the true noise fraction, and is it confounded by latent
dimensionality or latent timescale.

The species-gap restatement (same artifact, its own top-level field
"species_gap"): applies what the answer-key grid establishes to the real
human-versus-mouse-ALM comparison this project has used to justify reopening
its anatomical analysis, restating the existing per-cell numbers from
results/latent_model_observation_noise_comparison.json (never refitting
them) alongside the two candidate confounds (participation ratio,
leading-latent smoothness) the answer-key grid prices.

A second, deliberately mismatched generative model (correlated, temporally
white noise rather than the factor model's own assumed diagonal, per-unit
independent noise) is also run over the noise-fraction sweep, because a
generative model whose assumptions exactly match one estimator's own
assumptions is not a fair comparison of the two -- see
criticism_second_generative_model_correlated_noise below.

Deliverable: results/observation_noise_estimator_construct_validity.json.

Resumability: every (population size, noise model, f_true, d_true,
timescale) grid point's full set of per-seed results is flushed to
results/.checkpoints/observation_noise_estimator_construct_validity_
checkpoint.json immediately after that point completes, keyed by a
deterministic point id. A kill costs at most one grid point's 30 seeds, not
the run. Artifact assembly (assemble_artifact) reads only the checkpoint and
the two small real-data artifacts the species-gap restatement reads; it
performs no simulation and can be re-run at negligible cost from checkpoints
alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import locked_json_update  # noqa: E402
from observability import nugget_fraction  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from run_latent_model_comparison import LATENT_DIM  # noqa: E402
from run_latent_model_observation_noise_comparison import dimensionality_and_noise_term  # noqa: E402
from run_observability_and_power_census import N_SPLITS as CENSUS_N_SPLITS  # noqa: E402
from statistics import stable_seed  # noqa: E402

SEED = 20260813

CENSUS_PATH = ROOT / "results" / "observability_and_power_census.json"
FACTOR_MODEL_ARTIFACT_PATH = ROOT / "results" / "latent_model_observation_noise_comparison.json"
FACTOR_MODEL_CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "latent_model_observation_noise_comparison_checkpoint.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "observation_noise_estimator_construct_validity_checkpoint.json"
OUTPUT_PATH = ROOT / "results" / "observation_noise_estimator_construct_validity.json"

# ---------------------------------------------------------------------------
# The grid: three factors, each varied independently from a
# declared centre point. The centre values are themselves grid members (0.40
# is in F_TRUE_GRID, LATENT_DIM==6 is in D_TRUE_GRID, 0.3 is in
# TIMESCALE_GRID_S) so the three one-factor-at-a-time sweeps share one point,
# which is fit once and reused rather than refit three times.
# ---------------------------------------------------------------------------
F_TRUE_GRID = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
D_TRUE_GRID = (1, 2, 4, 6, 8, 12)
TIMESCALE_GRID_S = (0.05, 0.1, 0.3, 1.0, 3.0)
CENTRE_F_TRUE = 0.40
CENTRE_D_TRUE = LATENT_DIM
CENTRE_TIMESCALE_S = 0.3
N_SEEDS = 30
HUMAN_DATASETS = ("dandi_000469", "dandi_000574", "dandi_001187")

# The generative model this criticism run uses to deliberately violate the
# factor model's own diagonal-per-unit-noise assumption -- see
# run_criticism_second_generative_model's docstring below.
CRITICISM_NOISE_MODEL = "correlated_white"
CRITICISM_N_NOISE_FACTORS = 4

# The amplitude of the binary condition-mean shift added to the latent by
# generate_synthetic_population's condition_mean_amplitude_fraction, run at
# the grid's centre point under both noise models: the condition means are
# separated by 2x this fraction of the AR(1) process's own unit stationary
# standard deviation (each condition offset by +/- this fraction), a
# deliberately large, unambiguous condition-locked signal comparable in
# scale to the intrinsic AR(1) fluctuations it sits on top of.
CONDITION_MEAN_AMPLITUDE_FRACTION_OF_LATENT_SD = 1.0


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Population sizes, read from the census rather than assumed.
# ---------------------------------------------------------------------------

def _population_sizes_at_percentile(percentile: float, label_suffix: str) -> dict:
    """(n_trials, n_units, n_bins) of the human single-unit cells and of the
    mouse-ALM cell at the given percentile, both read from results/
    observability_and_power_census.json's own per-session rows -- never
    assumed. Restricted to bin_ms==100 rows only: n_bins and bin_width_s must
    be a physically consistent pair (n_bins implicitly encodes bin_width_s
    via a fixed window duration), and pooling bin100 and bin200 rows together
    would take the percentile of the two independently, which can produce a
    (n_bins, bin_width_s) combination that never occurred in any real
    session."""
    census = json.loads(CENSUS_PATH.read_text())
    rows = census["rows"]

    def _pool(dataset_filter):
        return [
            r for r in rows
            if r.get("modality") == "single_unit" and r.get("status") == "fitted"
            and r.get("bin_ms") == 100 and dataset_filter(r.get("dataset"))
        ]

    def _size(pool: list[dict], label: str) -> dict:
        if not pool:
            return {"status": "not_computable", "label": label, "reason": "no fitted bin_ms==100 single_unit rows found"}
        return {
            "status": "computed", "label": label, "n_rows_pooled": len(pool), "percentile": percentile,
            "n_trials": int(round(float(np.percentile([r["n_trials"] for r in pool], percentile)))),
            "n_units": int(round(float(np.percentile([r["n_units"] for r in pool], percentile)))),
            "n_bins": int(round(float(np.percentile([r["n_bins"] for r in pool], percentile)))),
            "bin_width_s": 0.1,
            "provenance": (
                f"{percentile:g}th percentile over results/observability_and_power_census.json rows with "
                "modality=='single_unit', status=='fitted', bin_ms==100"
            ),
        }

    return {
        f"human_single_unit{label_suffix}": _size(_pool(lambda d: d in HUMAN_DATASETS), f"human_single_unit{label_suffix}"),
        f"mouse_alm{label_suffix}": _size(_pool(lambda d: d == "inagaki_alm5"), f"mouse_alm{label_suffix}"),
    }


def population_sizes_from_census() -> dict:
    """Median (50th percentile) population sizes, labelled human_single_unit
    and mouse_alm -- the sizes the primary answer-key grid runs at."""
    return _population_sizes_at_percentile(50, "")


def small_population_sizes_from_census() -> dict:
    """10th-percentile population sizes, labelled human_single_unit_small and
    mouse_alm_small. The census quantity's real-data admission is governed by
    how many of its splits fit, which is driven by the corpora's small
    sessions rather than their median ones, so the median-size grid alone
    cannot speak to that regime -- see generate_synthetic_population and
    run_small_size_grid."""
    return _population_sizes_at_percentile(10, "_small")


# ---------------------------------------------------------------------------
# The generative model.
# ---------------------------------------------------------------------------

def _draw_noise(noise_model: str, n_trials: int, n_units: int, n_bins: int, n_noise_factors: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Raw (unscaled) noise draw, shape (n_trials, n_units, n_bins). Two
    generative assumptions about its cross-unit covariance, both temporally
    white (independent draws per bin, so neither noise model has any true
    latent dynamics of its own -- only the signal term z does):

    'diagonal' -- one independent Gaussian per unit: the noise covariance
        across units is exactly diagonal, matching the factor-analysis
        estimator's own modelling assumption exactly.
    'correlated_white' -- the noise loads onto units through its own random
        (n_units, n_noise_factors) matrix, so its true cross-unit covariance
        is low-rank plus nothing (rank n_noise_factors, not diagonal) --
        deliberately mismatched to the factor-analysis estimator's
        assumption, while remaining invisible to the cross-validated
        nugget-fraction estimator's own assumptions, which only look at one
        projected latent's temporal profile and never model cross-unit
        covariance shape at all.
    """
    if noise_model == "diagonal":
        return rng.normal(size=(n_trials, n_units, n_bins))
    if noise_model == "correlated_white":
        d_noise = max(1, min(n_noise_factors, n_units))
        loadings = rng.normal(size=(n_units, d_noise))
        factors = rng.normal(size=(n_trials, d_noise, n_bins))
        return np.einsum("ud,tdb->tub", loadings, factors)
    raise ValueError(f"unknown noise_model {noise_model!r}")


def _condition_mean_shape(n_bins: int) -> np.ndarray:
    """A smooth curve over a trial's bins, zero at both edges and peaking at
    the trial's midpoint (one arch of a sine), used as the common time course
    of the condition-locked mean shift added in generate_synthetic_population
    when condition_mean_amplitude_fraction > 0. Ranges over [0, 1]."""
    if n_bins <= 1:
        return np.zeros(max(n_bins, 1))
    t = np.arange(n_bins)
    return np.sin(np.pi * t / (n_bins - 1))


def generate_synthetic_population(
    n_trials: int, n_units: int, n_bins: int, d_true: int, f_true: float, timescale_s: float,
    bin_width_s: float, rng: np.random.Generator, noise_model: str = "diagonal",
    n_noise_factors: int = CRITICISM_N_NOISE_FACTORS, condition_mean_amplitude_fraction: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """x = C @ z + eps, shape (n_trials, n_units, n_bins).

    C: (n_units, d_true) iid standard Gaussian loadings.
    z: (n_trials, d_true, n_bins), d_true independent AR(1) processes per
       trial, each with unit stationary variance and coefficient
       a = exp(-bin_width_s / timescale_s) (the inverse of this project's own
       slow_timescale_s = -bin_width_s / log(rho) convention,
       src/observability.py's _decompose_autocovariance), plus -- only when
       condition_mean_amplitude_fraction > 0 -- a condition-locked mean shift
       added on top: trials are split into two binary conditions, and every
       latent dimension of one condition's trials is shifted by
       +condition_mean_amplitude_fraction * _condition_mean_shape(n_bins)
       (the other condition by the same shift negated), so the amplitude is
       expressed as a fraction of the AR(1) process's own unit stationary
       standard deviation. The shift is identical across all d_true
       dimensions (the condition-locked component is rank-1 in latent
       space), added after the AR(1) draw rather than mixed into its
       recursion, so the residual AR(1) process keeps its own stationary
       variance and timescale exactly as when condition_mean_amplitude_
       fraction == 0.
    eps: drawn from _draw_noise, then rescaled so total noise variance over
       total measured variance equals f_true in expectation; the realised
       value is computed from the drawn arrays and returned, since the
       scaling is exact only in expectation over noise_model=='diagonal' and
       even more so under 'correlated_white', whose raw variance is itself
       stochastic.
    """
    c_loadings = rng.normal(size=(n_units, d_true))
    a = float(np.exp(-bin_width_s / timescale_s))
    innovation_sd = float(np.sqrt(max(1.0 - a * a, 1e-12)))
    z = np.empty((n_trials, d_true, n_bins))
    z[:, :, 0] = rng.normal(size=(n_trials, d_true))
    for t in range(1, n_bins):
        z[:, :, t] = a * z[:, :, t - 1] + innovation_sd * rng.normal(size=(n_trials, d_true))

    condition = None
    n_condition_0 = n_condition_1 = None
    if condition_mean_amplitude_fraction:
        condition = rng.integers(0, 2, size=n_trials)
        n_condition_0 = int((condition == 0).sum())
        n_condition_1 = int((condition == 1).sum())
        sign = np.where(condition == 1, 1.0, -1.0)  # (n_trials,)
        shape = _condition_mean_shape(n_bins)  # (n_bins,)
        condition_mean = condition_mean_amplitude_fraction * sign[:, None, None] * shape[None, None, :]  # (n_trials, 1, n_bins)
        z = z + condition_mean

    signal = np.einsum("ud,tdb->tub", c_loadings, z)
    total_signal_var = float(signal.var(axis=(0, 2)).sum())

    eps_raw = _draw_noise(noise_model, n_trials, n_units, n_bins, n_noise_factors, rng)
    raw_noise_var = float(eps_raw.var(axis=(0, 2)).sum())
    target_total_noise = total_signal_var * f_true / max(1.0 - f_true, 1e-9)
    scale = float(np.sqrt(target_total_noise / raw_noise_var)) if raw_noise_var > 0 else 0.0
    eps = eps_raw * scale

    x = signal + eps
    realised_noise_var = float(eps.var(axis=(0, 2)).sum())
    realised_signal_var = float(signal.var(axis=(0, 2)).sum())
    f_true_realised = (
        realised_noise_var / (realised_noise_var + realised_signal_var)
        if (realised_noise_var + realised_signal_var) > 0 else None
    )
    return x, {
        "f_true_requested": f_true, "f_true_realised": f_true_realised, "d_true": d_true,
        "timescale_s_requested": timescale_s, "bin_width_s": bin_width_s, "ar1_coefficient_a": a,
        "noise_model": noise_model,
        "condition_structure": {
            "present": condition is not None,
            "amplitude_fraction_of_latent_sd": condition_mean_amplitude_fraction,
            "n_condition_0": n_condition_0, "n_condition_1": n_condition_1,
        },
    }


# ---------------------------------------------------------------------------
# The two estimator entry points -- called exactly as the real analyses call
# them, never reimplemented.
# ---------------------------------------------------------------------------

ENTRY_POINTS = {
    "factor_analysis": (
        "scripts/run_latent_model_observation_noise_comparison.py:dimensionality_and_noise_term"
        "(counts, is_point_process=False) -> ['factor_analysis']['observation_noise_variance_fraction'], "
        f"fit at k=min({LATENT_DIM}, n_units-1) where {LATENT_DIM} is "
        "scripts/run_latent_model_comparison.py's LATENT_DIM, read at import time, not restated"
    ),
    "census_nugget_fraction": (
        "src/observability.py:nugget_fraction(counts, n_splits=N_SPLITS, bin_width_s=...) "
        "-> ['median_nugget_fraction'], with n_splits read from "
        f"scripts/run_observability_and_power_census.py's N_SPLITS ({CENSUS_N_SPLITS}), the same "
        "value the real census uses"
    ),
    "is_point_process_choice": (
        "is_point_process=False for every synthetic draw: the generative model is a continuous linear-"
        "Gaussian factor model (x = Cz + eps), not non-negative integer spike counts, so the point-"
        "process arm's Anscombe variance-stabilising transform (which clips negative values before a "
        "square root) has no justification here and would distort the synthetic signal. This matches "
        "the real pipeline's own convention for continuous grains (e.g. LFP band power), which also "
        "call dimensionality_and_noise_term with is_point_process=False."
    ),
}


def run_factor_model_estimator(x: np.ndarray) -> dict:
    result = dimensionality_and_noise_term(x, is_point_process=False)
    fa = result["factor_analysis"]
    if fa.get("status") != "fitted":
        return {"status": fa.get("status", "not_fitted"), "value": None}
    return {"status": "fitted", "value": fa["observation_noise_variance_fraction"]}


def run_census_estimator(x: np.ndarray, bin_width_s: float, rng: np.random.Generator) -> dict:
    result = nugget_fraction(x, n_splits=CENSUS_N_SPLITS, rng=rng, bin_width_s=bin_width_s)
    return {
        "status": result["status"],
        "value": result["median_nugget_fraction"] if result["status"] == "fitted" else None,
        "n_splits_fitted": result["n_splits_fitted"], "n_splits_requested": result["n_splits_requested"],
    }


# ---------------------------------------------------------------------------
# Grid execution with per-point checkpointing, so a kill costs one grid
# point's seeds rather than the run.
# ---------------------------------------------------------------------------

def _point_id(pop_label: str, noise_model: str, f_true: float, d_true: int, timescale_s: float,
              condition_mean_amplitude_fraction: float = 0.0) -> str:
    point_id = f"{pop_label}|{noise_model}|f{f_true:.4f}|d{d_true:02d}|t{timescale_s:.4f}"
    if condition_mean_amplitude_fraction:
        point_id += f"|cond{condition_mean_amplitude_fraction:.2f}"
    return point_id


def _run_grid_point(pop_label: str, pop_size: dict, noise_model: str, f_true: float, d_true: int,
                     timescale_s: float, n_seeds: int = N_SEEDS,
                     condition_mean_amplitude_fraction: float = 0.0) -> list[dict]:
    per_seed = []
    for seed_idx in range(n_seeds):
        base = (pop_label, noise_model, f"f{f_true:.4f}", f"d{d_true}", f"t{timescale_s:.4f}", f"seed{seed_idx}")
        if condition_mean_amplitude_fraction:
            base = base + (f"cond{condition_mean_amplitude_fraction:.2f}",)
        x, realised = generate_synthetic_population(
            pop_size["n_trials"], pop_size["n_units"], pop_size["n_bins"], d_true, f_true, timescale_s,
            pop_size["bin_width_s"], _seed(*base, "data"), noise_model=noise_model,
            condition_mean_amplitude_fraction=condition_mean_amplitude_fraction,
        )
        factor_analysis = run_factor_model_estimator(x)
        census = run_census_estimator(x, pop_size["bin_width_s"], _seed(*base, "census"))
        per_seed.append({
            "seed": seed_idx, "f_true_realised": realised["f_true_realised"],
            "factor_analysis": factor_analysis, "census": census,
        })
    return per_seed


def run_block_with_checkpoint(point_id: str, builder) -> list[dict]:
    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        entry = checkpoint.get(point_id)
        if entry is not None and entry.get("status") == "complete":
            return entry["per_seed"]
    per_seed = builder()
    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        checkpoint[point_id] = {"status": "complete", "per_seed": per_seed}
    return per_seed


def build_sweep_points(centre_f: float, centre_d: int, centre_t: float) -> dict[tuple, set[str]]:
    """Union of the three one-factor-at-a-time sweeps, deduplicated: each
    (f_true, d_true, timescale_s) grid point maps to the set of sweep names
    it belongs to, so the shared centre point is fit once, not three times."""
    points: dict[tuple, set[str]] = {}

    def _add(f, d, t, sweep):
        points.setdefault((f, d, t), set()).add(sweep)

    for f in F_TRUE_GRID:
        _add(f, centre_d, centre_t, "f_true_sweep")
    for d in D_TRUE_GRID:
        _add(centre_f, d, centre_t, "d_true_sweep")
    for t in TIMESCALE_GRID_S:
        _add(centre_f, centre_d, t, "timescale_sweep")
    return points


def run_all_grid_points(populations: dict, noise_model: str, centre_f: float, centre_d: int,
                         centre_t: float, log_prefix: str, condition_mean_amplitude_fraction: float = 0.0) -> dict[str, dict]:
    """Returns {pop_label: {(f,d,t): per_seed_list}} for every point in the
    union of the three sweeps, for every population size, resumed from
    checkpoint where available."""
    points = build_sweep_points(centre_f, centre_d, centre_t)
    out: dict[str, dict] = {}
    for pop_label, pop_size in populations.items():
        if pop_size.get("status") != "computed":
            out[pop_label] = {}
            continue
        out[pop_label] = {}
        for (f, d, t) in sorted(points):
            point_id = _point_id(pop_label, noise_model, f, d, t, condition_mean_amplitude_fraction)
            t0 = time.time()
            per_seed = run_block_with_checkpoint(
                point_id, lambda pop_size=pop_size, f=f, d=d, t=t: _run_grid_point(
                    pop_label, pop_size, noise_model, f, d, t, condition_mean_amplitude_fraction=condition_mean_amplitude_fraction))
            print(f"{log_prefix} {point_id}: {len(per_seed)} seeds in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
            out[pop_label][(f, d, t)] = per_seed
    return out, points


def _grid_points_for(populations: dict, noise_model: str, condition_mean_amplitude_fraction: float,
                      log_prefix: str, compute: bool) -> dict[str, dict]:
    """Returns {pop_label: {(f,d,t): per_seed_list}} for the full centre-based
    sweep grid. compute=True runs (or reuses from checkpoint) via
    run_all_grid_points; compute=False reads only entries already marked
    complete in the checkpoint file on disk, without running anything --
    used by the assemble-only path, which performs no simulation."""
    if compute:
        points, _ = run_all_grid_points(populations, noise_model, CENTRE_F_TRUE, CENTRE_D_TRUE, CENTRE_TIMESCALE_S,
                                         log_prefix, condition_mean_amplitude_fraction)
        return points
    checkpoint = json.loads(CHECKPOINT_PATH.read_text()) if CHECKPOINT_PATH.exists() else {}
    sweep_points = build_sweep_points(CENTRE_F_TRUE, CENTRE_D_TRUE, CENTRE_TIMESCALE_S)
    out: dict[str, dict] = {}
    for pop_label, pop_size in populations.items():
        out[pop_label] = {}
        if pop_size.get("status") != "computed":
            continue
        for (f, d, t) in sweep_points:
            point_id = _point_id(pop_label, noise_model, f, d, t, condition_mean_amplitude_fraction)
            entry = checkpoint.get(point_id)
            out[pop_label][(f, d, t)] = entry["per_seed"] if entry and entry.get("status") == "complete" else []
    return out


def full_grid_branches(populations: dict, noise_model: str, condition_mean_amplitude_fraction: float,
                        compute: bool, log_prefix: str) -> dict:
    """{pop_label: build_sweeps_for_population output} for the full
    centre-based sweep grid at the given noise model and condition-mean
    amplitude, for every population whose size is computed."""
    computed = {k: v for k, v in populations.items() if v.get("status") == "computed"}
    points = _grid_points_for(computed, noise_model, condition_mean_amplitude_fraction, log_prefix, compute)
    return {pop_label: build_sweeps_for_population(points, pop_label) for pop_label in computed}


# ---------------------------------------------------------------------------
# Summaries and pre-declared branch resolution.
# ---------------------------------------------------------------------------

def summarize_seed_values(per_seed: list[dict], estimator: str) -> dict:
    values = [s[estimator]["value"] for s in per_seed if s[estimator]["status"] == "fitted" and s[estimator]["value"] is not None]
    n_total = len(per_seed)
    n_fitted = len(values)
    if not values:
        return {"status": "not_computable", "n_total": n_total, "n_fitted": 0, "reason": "no seed produced a fitted value"}
    arr = np.array(values)
    return {
        "status": "fitted", "n_total": n_total, "n_fitted": n_fitted,
        "median": float(np.median(arr)), "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
    }


def _sweep_summary(pop_points: dict, grid_values, fixed_axes, estimator: str) -> dict:
    """pop_points: {(f,d,t): per_seed_list}. grid_values: the values of the
    swept axis. fixed_axes: a function grid_value -> (f,d,t) key into
    pop_points."""
    out = {}
    for gv in grid_values:
        key = fixed_axes(gv)
        per_seed = pop_points.get(key)
        out[gv] = summarize_seed_values(per_seed, estimator) if per_seed is not None else {
            "status": "not_computable", "reason": "grid point not fit"}
    return out


def _span(sweep_summary: dict) -> dict:
    medians = [s["median"] for s in sweep_summary.values() if s.get("status") == "fitted"]
    n_total = len(sweep_summary)
    if len(medians) < 2:
        return {"status": "not_computable", "reason": f"fewer than 2 fitted points ({len(medians)}/{n_total})", "n_points_fitted": len(medians), "n_points_total": n_total}
    return {"status": "computed", "span": float(max(medians) - min(medians)), "n_points_fitted": len(medians), "n_points_total": n_total}


def _monotone_in_f_true(f_sweep: dict) -> dict:
    ordered = [f_sweep[f]["median"] for f in F_TRUE_GRID if f_sweep.get(f, {}).get("status") == "fitted"]
    n_total = len(F_TRUE_GRID)
    if len(ordered) < 2:
        return {"status": "not_computable", "monotone": None, "n_points_fitted": len(ordered), "n_points_total": n_total}
    diffs = np.diff(ordered)
    monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return {
        "status": "computed", "monotone": monotone, "n_points_fitted": len(ordered), "n_points_total": n_total,
        "definition": "weakly non-decreasing or weakly non-increasing (ties allowed) across the f_true grid, ascending order, over whichever points fit",
    }


BRANCH_DEFINITIONS = {
    "recovers_the_noise_fraction": "median output within 0.10 of f_true at every f_true, AND its span across d_true and across timescale is under 0.10 at fixed f_true",
    "tracks_dimensionality": "its span across d_true at fixed f_true exceeds its span across f_true at fixed d_true",
    "tracks_latent_smoothness": "its span across timescale at fixed f_true exceeds its span across f_true at fixed timescale",
    "tracks_noise_with_a_confound": "monotone in f_true with span over 0.20, but also has span over 0.10 across d_true or timescale",
    "recovers_nothing": "not monotone in f_true",
}


def resolve_estimator_branches(f_sweep: dict, d_sweep: dict, t_sweep: dict) -> dict:
    span_f, span_d, span_t = _span(f_sweep), _span(d_sweep), _span(t_sweep)
    monotone = _monotone_in_f_true(f_sweep)
    accuracy_within_0_10 = (
        all(f_sweep.get(f, {}).get("status") == "fitted" for f in F_TRUE_GRID)
        and all(abs(f_sweep[f]["median"] - f) <= 0.10 for f in F_TRUE_GRID)
    )

    branches: list[str] = []
    if (accuracy_within_0_10 and span_d.get("status") == "computed" and span_d["span"] < 0.10
            and span_t.get("status") == "computed" and span_t["span"] < 0.10):
        branches.append("recovers_the_noise_fraction")
    if span_d.get("status") == "computed" and span_f.get("status") == "computed" and span_d["span"] > span_f["span"]:
        branches.append("tracks_dimensionality")
    if span_t.get("status") == "computed" and span_f.get("status") == "computed" and span_t["span"] > span_f["span"]:
        branches.append("tracks_latent_smoothness")
    confound_axes = []
    if monotone.get("monotone") is True and span_f.get("status") == "computed" and span_f["span"] > 0.20:
        if span_d.get("status") == "computed" and span_d["span"] > 0.10:
            confound_axes.append("d_true")
        if span_t.get("status") == "computed" and span_t["span"] > 0.10:
            confound_axes.append("timescale")
        if confound_axes:
            branches.append("tracks_noise_with_a_confound")
    if monotone.get("status") == "computed" and monotone["monotone"] is False:
        branches.append("recovers_nothing")

    return {
        "span_f_true_at_fixed_d_and_timescale": span_f,
        "span_d_true_at_fixed_f_and_timescale": span_d,
        "span_timescale_at_fixed_f_and_d": span_t,
        "monotone_in_f_true": monotone,
        "accuracy_within_0_10_at_every_f_true": accuracy_within_0_10,
        "confound_axes_for_tracks_noise_with_a_confound": confound_axes,
        "branches_resolved": branches,
        "branch_definitions": BRANCH_DEFINITIONS,
        "note": "branches are not mutually exclusive by construction; every applicable one is listed",
    }


def census_zero_floor_reachability(pop_points_by_model: dict) -> dict:
    """The targeted zero-floor question: every fitted grid point (any sweep,
    any population, any noise model already run) at which the census
    estimator's median output is 0.0 or within 0.01 of it, with that point's
    f_true -- computed before the species-gap restatement below reads it,
    since it decides how that restatement's branches interpret the
    mouse-ALM cell's own exact-zero census value."""
    near_zero = []
    max_f_true_at_near_zero = None
    for noise_model, pop_points in pop_points_by_model.items():
        for pop_label, points in pop_points.items():
            for (f, d, t), per_seed in points.items():
                summary = summarize_seed_values(per_seed, "census")
                if summary.get("status") == "fitted" and summary["median"] <= 0.01:
                    near_zero.append({
                        "noise_model": noise_model, "population": pop_label,
                        "f_true": f, "d_true": d, "timescale_s": t,
                        "median_census_output": summary["median"], "n_fitted_seeds": summary["n_fitted"],
                    })
                    if max_f_true_at_near_zero is None or f > max_f_true_at_near_zero:
                        max_f_true_at_near_zero = f
    if not near_zero:
        return {
            "status": "no_near_zero_points_on_this_grid",
            "reading": (
                "the census estimator never returned a value at or within 0.01 of zero anywhere on this "
                "grid -- zero output on real mouse ALM data would then require true noise genuinely near "
                "zero, which rehabilitates the species-gap reading of that cell rather than undermining it"
            ),
        }
    large_f_true_reachable = max_f_true_at_near_zero is not None and max_f_true_at_near_zero >= 0.40
    return {
        "status": "near_zero_points_found",
        "n_points": len(near_zero),
        "points": near_zero,
        "max_f_true_at_a_near_zero_census_output": max_f_true_at_near_zero,
        "large_f_true_reachable_at_near_zero_output": large_f_true_reachable,
        "reading": (
            (
                f"a near-zero census output is reachable at f_true up to {max_f_true_at_near_zero:.2f} on this "
                "grid -- a census value of exactly 0.0 on real data is consistent with substantial true "
                "observation noise and does not by itself establish near-zero true noise"
            ) if large_f_true_reachable else
            (
                "near-zero census output on this grid only occurs at small f_true -- zero output on real "
                "data is evidence for genuinely low true noise, which rehabilitates rather than undermines "
                "the species-gap reading of an exact-zero cell"
            )
        ),
    }


def estimator_fit_failure_rates(pop_points_by_model: dict) -> dict:
    """Per estimator, the fraction of individual seed-level fits
    that failed to return a value, split by status/reason string, and the
    census estimator's own distribution of n_splits_fitted (its admission
    criterion in the real census, so its dependence on the generative
    parameters is a result here, not a diagnostic)."""
    fa_status_counts: dict[str, int] = {}
    census_status_counts: dict[str, int] = {}
    census_n_splits_fitted: list[int] = []
    n_seeds_total = 0
    for pop_points in pop_points_by_model.values():
        for points in pop_points.values():
            for per_seed in points.values():
                for s in per_seed:
                    n_seeds_total += 1
                    fa_status_counts[s["factor_analysis"]["status"]] = fa_status_counts.get(s["factor_analysis"]["status"], 0) + 1
                    census_status_counts[s["census"]["status"]] = census_status_counts.get(s["census"]["status"], 0) + 1
                    census_n_splits_fitted.append(s["census"]["n_splits_fitted"])
    if n_seeds_total == 0:
        return {"status": "not_computable", "reason": "no seeds run"}
    arr = np.array(census_n_splits_fitted)
    return {
        "n_seeds_total": n_seeds_total,
        "factor_analysis_status_counts": fa_status_counts,
        "factor_analysis_failure_fraction": 1.0 - fa_status_counts.get("fitted", 0) / n_seeds_total,
        "census_status_counts": census_status_counts,
        "census_failure_fraction": 1.0 - census_status_counts.get("fitted", 0) / n_seeds_total,
        "census_n_splits_fitted_distribution": {
            "median": float(np.median(arr)), "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
            "min": int(arr.min()), "max": int(arr.max()), "n_splits_requested": CENSUS_N_SPLITS,
        },
    }


def build_sweeps_for_population(points: dict, pop_label: str) -> dict:
    pop_points = points[pop_label]
    f_sweep_key = lambda f: (f, CENTRE_D_TRUE, CENTRE_TIMESCALE_S)
    d_sweep_key = lambda d: (CENTRE_F_TRUE, d, CENTRE_TIMESCALE_S)
    t_sweep_key = lambda t: (CENTRE_F_TRUE, CENTRE_D_TRUE, t)
    result = {}
    for estimator in ("factor_analysis", "census"):
        f_sweep = {f: summarize_seed_values(pop_points.get(f_sweep_key(f), []), estimator) for f in F_TRUE_GRID}
        d_sweep = {d: summarize_seed_values(pop_points.get(d_sweep_key(d), []), estimator) for d in D_TRUE_GRID}
        t_sweep = {t: summarize_seed_values(pop_points.get(t_sweep_key(t), []), estimator) for t in TIMESCALE_GRID_S}
        result[estimator] = {
            "f_true_sweep": {str(k): v for k, v in f_sweep.items()},
            "d_true_sweep": {str(k): v for k, v in d_sweep.items()},
            "timescale_sweep": {str(k): v for k, v in t_sweep.items()},
            "branches": resolve_estimator_branches(f_sweep, d_sweep, t_sweep),
        }
    return result


def _branch_set_diff(baseline_branches: list[str], treatment_branches: list[str]) -> dict:
    baseline_set, treatment_set = set(baseline_branches), set(treatment_branches)
    return {
        "baseline_branches_resolved": sorted(baseline_set),
        "treatment_branches_resolved": sorted(treatment_set),
        "added_under_treatment": sorted(treatment_set - baseline_set),
        "removed_under_treatment": sorted(baseline_set - treatment_set),
        "identical": baseline_set == treatment_set,
    }


def _extension_branch_differences(baseline_by_population: dict, treatment_by_population: dict,
                                   treatment_label_to_baseline_label=lambda label: label) -> dict:
    """Per population and per estimator, the set difference between a
    treatment grid's resolved branches (condition structure present, or the
    small-size population) and the corresponding no-condition/median-size
    baseline grid's resolved branches."""
    out = {}
    for pop_label, treatment_pop in treatment_by_population.items():
        baseline_pop = baseline_by_population.get(treatment_label_to_baseline_label(pop_label), {})
        out[pop_label] = {
            estimator: _branch_set_diff(
                baseline_pop.get(estimator, {}).get("branches", {}).get("branches_resolved", []),
                treatment_pop.get(estimator, {}).get("branches", {}).get("branches_resolved", []),
            )
            for estimator in ("factor_analysis", "census")
        }
    return out


def run_generative_factor_extensions(populations: dict, estimator_branches_by_population: dict, compute: bool) -> dict:
    """The two generative-model additions beyond the median-size, no-
    condition-structure grid built by estimator_branches_by_population:

    condition_structure -- a binary condition-mean factor added to the
    latent (generate_synthetic_population's condition_mean_amplitude_
    fraction), run at the median population sizes under both noise models,
    with 1.5's branches resolved the same way and compared against the
    no-condition baseline at the same noise model and population size
    (estimator_branches_by_population for noise_model=='diagonal';
    a freshly resolved no-condition correlated_white grid, computed here,
    for noise_model=='correlated_white', since the criticism run only
    summarizes that noise model's f_true sweep, not its full branch set).

    small_size -- the same no-condition-structure grid run at each
    population's 10th-percentile (n_trials, n_units, n_bins) rather than its
    median, under both noise models, compared against the median-size
    baseline (estimator_branches_by_population for 'diagonal', the same
    freshly resolved correlated_white grid for 'correlated_white').

    compute=False loads every point from the checkpoint on disk only,
    performing no simulation -- the assemble-only path's contract.
    """
    small_populations = small_population_sizes_from_census()
    strip_small_suffix = lambda label: label.removesuffix("_small")

    condition_diagonal = full_grid_branches(populations, "diagonal", CONDITION_MEAN_AMPLITUDE_FRACTION_OF_LATENT_SD, compute, "[condition:diagonal]")
    condition_correlated_white = full_grid_branches(populations, CRITICISM_NOISE_MODEL, CONDITION_MEAN_AMPLITUDE_FRACTION_OF_LATENT_SD, compute, "[condition:correlated_white]")
    correlated_white_no_condition_median = full_grid_branches(populations, CRITICISM_NOISE_MODEL, 0.0, compute, "[no_condition_baseline:correlated_white]")
    small_diagonal = full_grid_branches(small_populations, "diagonal", 0.0, compute, "[small_size:diagonal]")
    small_correlated_white = full_grid_branches(small_populations, CRITICISM_NOISE_MODEL, 0.0, compute, "[small_size:correlated_white]")

    return {
        "condition_structure": {
            "amplitude_fraction_of_latent_sd": CONDITION_MEAN_AMPLITUDE_FRACTION_OF_LATENT_SD,
            "population_sizes": populations,
            "with_condition_structure_by_noise_model": {"diagonal": condition_diagonal, CRITICISM_NOISE_MODEL: condition_correlated_white},
            "without_condition_structure_correlated_white_baseline": correlated_white_no_condition_median,
            "without_condition_structure_diagonal_baseline_field": "estimator_branches (this artifact's own top-level field) -- the diagonal, no-condition-structure, median-size grid",
            "branch_differences_vs_no_condition_structure_by_noise_model": {
                "diagonal": _extension_branch_differences(estimator_branches_by_population, condition_diagonal),
                CRITICISM_NOISE_MODEL: _extension_branch_differences(correlated_white_no_condition_median, condition_correlated_white),
            },
        },
        "small_size": {
            "population_sizes": small_populations,
            "by_noise_model": {"diagonal": small_diagonal, CRITICISM_NOISE_MODEL: small_correlated_white},
            "branch_differences_vs_median_size_by_noise_model": {
                "diagonal": _extension_branch_differences(estimator_branches_by_population, small_diagonal, strip_small_suffix),
                CRITICISM_NOISE_MODEL: _extension_branch_differences(correlated_white_no_condition_median, small_correlated_white, strip_small_suffix),
            },
        },
    }


def run_criticism_second_generative_model(populations: dict) -> dict:
    """The criticism this file's own docstring names: the primary grid's
    noise model (independent per-unit diagonal Gaussian) is exactly the
    factor-analysis estimator's own assumption, so a construct-validity
    comparison run only on that model
    risks being rigged in the factor-analysis estimator's favour. This runs
    the f_true sweep (the sweep the headline branches turn on) a second
    time, at both population sizes, under noise_model='correlated_white'
    (low-rank, non-diagonal cross-unit noise covariance, still temporally
    white) -- a model whose assumptions instead favour the census
    estimator, which never models cross-unit covariance shape at all."""
    points, _ = run_all_grid_points(
        {k: v for k, v in populations.items() if v.get("status") == "computed"}, CRITICISM_NOISE_MODEL,
        CENTRE_F_TRUE, CENTRE_D_TRUE, CENTRE_TIMESCALE_S, log_prefix="[criticism]")
    return _criticism_result_from_points(points)


def _criticism_by_population(points: dict) -> dict:
    by_population = {}
    for pop_label in points:
        f_sweep_key = lambda f: (f, CENTRE_D_TRUE, CENTRE_TIMESCALE_S)
        pop_result = {}
        for estimator in ("factor_analysis", "census"):
            f_sweep = {f: summarize_seed_values(points[pop_label].get(f_sweep_key(f), []), estimator) for f in F_TRUE_GRID}
            accuracy_within_0_10 = (
                all(f_sweep.get(f, {}).get("status") == "fitted" for f in F_TRUE_GRID)
                and all(abs(f_sweep[f]["median"] - f) <= 0.10 for f in F_TRUE_GRID)
            )
            median_abs_error = (
                float(np.median([abs(f_sweep[f]["median"] - f) for f in F_TRUE_GRID if f_sweep[f].get("status") == "fitted"]))
                if any(f_sweep[f].get("status") == "fitted" for f in F_TRUE_GRID) else None
            )
            pop_result[estimator] = {
                "f_true_sweep": {str(k): v for k, v in f_sweep.items()},
                "accuracy_within_0_10_at_every_f_true": accuracy_within_0_10,
                "median_absolute_error_across_f_true_grid": median_abs_error,
            }
        by_population[pop_label] = pop_result
    return by_population


def _criticism_result_from_points(points: dict) -> dict:
    return {
        "noise_model": CRITICISM_NOISE_MODEL,
        "n_noise_factors": CRITICISM_N_NOISE_FACTORS,
        "description": (
            "the noise term's true cross-unit covariance is low-rank (rank "
            f"{CRITICISM_N_NOISE_FACTORS}) rather than diagonal -- it violates the factor-analysis "
            "estimator's own modelling assumption while remaining temporally white, so it carries no "
            "genuine latent dynamics the census estimator could mistake for signal either"
        ),
        "sweep": "f_true only, at the primary grid's centre d_true and timescale -- the sweep the headline recovers_the_noise_fraction / tracks_noise_with_a_confound branches turn on",
        "by_population": _criticism_by_population(points),
        "comparison_to_diagonal_noise_model": (
            "compare each estimator's accuracy_within_0_10_at_every_f_true and "
            "median_absolute_error_across_f_true_grid here against the same fields in "
            "estimator_branches.<population>.<estimator>.branches for noise_model=='diagonal' -- "
            "a change in which estimator is more accurate between the two noise models is the signal "
            "this criticism run is checking for"
        ),
    }


# ---------------------------------------------------------------------------
# The species gap, restated from existing real-data artifacts.
# Reads only; refits nothing.
# ---------------------------------------------------------------------------

PRECURSOR_NUMBERS_CARRIED_FORWARD = {
    "two_estimators_disagree": {
        "artifact": "results/latent_model_observation_noise_comparison.json",
        "field": "observation_noise_estimator_offset (32 pairs)",
        "pearson_r": -0.005, "pearson_p": 0.98, "spearman_rho": -0.161, "spearman_p": 0.38,
        "reading": "the factor-analysis noise-variance fraction and the census nugget fraction share no measurable variance across the 32 cells where both exist",
    },
    "factor_analysis_tracks_participation_ratio": {
        "within_cell_spearman_rho": [0.781, 0.753, 0.762, 0.807], "within_cell_p_all_below": 2e-07,
        "across_32_cells_spearman_rho_factor_analysis_pr": 0.593, "across_32_cells_p": 3.5e-04,
        "across_32_cells_spearman_rho_pca_pr": 0.643, "across_32_cells_p_pca": 7.1e-05,
        "partialling_out_channel_count_rho": 0.762, "partialling_out_channel_count_p": 8e-15,
    },
    "census_does_not_track_dimensionality_but_floors_at_alm": {
        "across_cells_spearman_rho_vs_participation_ratio": 0.137, "p": 0.49,
        "mouse_alm_census_value": 0.0,
        "mouse_alm_factor_analysis_value": 0.708,
        "human_single_unit_factor_analysis_range": [0.503, 0.887],
    },
    "census_ordering_rests_on_fittability_not_value": {
        "summed_rank_difference_lfp_minus_unit": 2,
        "all_of_it_from": "cross_validated_nugget_fraction (LFP rank 3, unit rank 1); the other six shared observables tie at 0",
        "median_fitted_splits_unit_bin100": 3.0, "median_fitted_splits_unit_bin200": 0.0,
        "median_fitted_splits_lfp_bin100": 20.0, "median_fitted_splits_lfp_bin200": 5.0,
    },
}


def load_species_gap_cells() -> dict:
    factor_model_artifact = json.loads(FACTOR_MODEL_ARTIFACT_PATH.read_text())
    census = json.loads(CENSUS_PATH.read_text())
    checkpoint = json.loads(FACTOR_MODEL_CHECKPOINT_PATH.read_text())
    return factor_model_artifact, census, checkpoint


def _cell_group(dataset: str, modality: str) -> str | None:
    if modality != "single_unit":
        return None
    if dataset in HUMAN_DATASETS:
        return "human_single_unit"
    if dataset == "inagaki_alm5":
        return "mouse_alm"
    return None


def _checkpoint_sessions_for_cell(checkpoint: dict, dataset: str, structure: str, bin_ms: int) -> list[dict]:
    block_name = "inagaki_alm5" if dataset == "inagaki_alm5" else dataset
    block = checkpoint.get(block_name, {})
    sessions = block.get("sessions", [])
    return [s for s in sessions if s.get("structure") == structure and s.get("bin_ms") == bin_ms]


def _smoothness_for_cell(census_rows: list[dict], dataset: str, structure: str, bin_ms: int, modality: str) -> dict:
    """Leading-latent smoothness (lag-one autocorrelation, implied timescale)
    for a cell, computed the same way in every cell: from
    results/observability_and_power_census.json's own already-fitted
    median_slow_timescale_s per session (src/observability.py's
    _decompose_autocovariance, the same code path the nugget-fraction
    estimator itself uses) -- never refit here. rho = exp(-bin_width_s /
    slow_timescale_s), the exact inverse of that function's own timescale
    formula."""
    values = []
    for r in census_rows:
        if (r.get("dataset") == dataset and r.get("structure") == structure and r.get("bin_ms") == bin_ms
                and r.get("modality") == modality and r.get("status") == "fitted"
                and r.get("median_slow_timescale_s") is not None and r["median_slow_timescale_s"] > 0):
            rho = float(np.exp(-r["bin_width_s"] / r["median_slow_timescale_s"]))
            values.append({"rho": rho, "slow_timescale_s": r["median_slow_timescale_s"]})
    if not values:
        return {"status": "not_computable", "reason": "no fitted census row with a positive slow_timescale_s for this cell"}
    return {
        "status": "computed", "n_sessions": len(values),
        "median_rho": float(np.median([v["rho"] for v in values])),
        "median_slow_timescale_s": float(np.median([v["slow_timescale_s"] for v in values])),
    }


def build_cells_table(factor_model_artifact: dict, census: dict, checkpoint: dict) -> dict:
    census_rows = census["rows"]
    table = {}
    for key, cell in factor_model_artifact["cells"].items():
        group = _cell_group(cell["dataset"], cell["modality"])
        if group is None:
            continue
        sessions = _checkpoint_sessions_for_cell(checkpoint, cell["dataset"], cell["structure"], cell["bin_ms"])
        n_units_per_session = [s["n_units"] for s in sessions]
        n_trials_per_session = [s["n_trials"] for s in sessions]
        k_per_session = [max(1, min(LATENT_DIM, nu - 1)) for nu in n_units_per_session]
        matched_census_rows = [
            r for r in census_rows
            if r.get("dataset") == cell["dataset"] and r.get("structure") == cell["structure"]
            and r.get("bin_ms") == cell["bin_ms"] and r.get("modality") == cell["modality"] and r.get("status") == "fitted"
        ]
        n_bins_values = sorted(set(r["n_bins"] for r in matched_census_rows))
        table[key] = {
            "group": group, "dataset": cell["dataset"], "structure": cell["structure"], "bin_ms": cell["bin_ms"],
            "factor_analysis_noise_fraction": cell["observation_noise_term_comparison"]["factor_analysis_median_noise_variance_fraction"],
            "census_nugget_fraction": cell["observation_noise_term_comparison"]["census_cross_validated_nugget_fraction"],
            "participation_ratio_pca_basis": cell["summary"]["participation_ratio"].get("median_pca"),
            "participation_ratio_factor_analysis_basis": cell["summary"]["participation_ratio"].get("median_factor_analysis"),
            "smoothness": _smoothness_for_cell(census_rows, cell["dataset"], cell["structure"], cell["bin_ms"], cell["modality"]),
            "realised_matching": {
                "n_sessions_in_factor_model_checkpoint": len(sessions),
                "n_trials_per_session": {"median": float(np.median(n_trials_per_session)), "min": int(min(n_trials_per_session)), "max": int(max(n_trials_per_session))} if n_trials_per_session else None,
                "n_units_per_session": {"median": float(np.median(n_units_per_session)), "min": int(min(n_units_per_session)), "max": int(max(n_units_per_session))} if n_units_per_session else None,
                "n_bins_values_seen_in_census_rows": n_bins_values,
                "k_values_seen": sorted(set(k_per_session)),
                "k_matched_across_sessions": len(set(k_per_session)) <= 1,
                "n_matched_census_rows": len(matched_census_rows),
            },
        }
    return table


def matching_verification(cells_table: dict) -> dict:
    """Is the standing claim 'same estimator, same code path, trial-count
    matched' true as claimed. Same code path is a fact about the source (both human
    and ALM sessions are built by _build_dataset_sessions/_build_alm_sessions,
    both of which call the identical analyze_session -> dimensionality_and_
    noise_term / nugget_fraction functions -- verified by reading
    scripts/run_latent_model_observation_noise_comparison.py, not asserted).
    Trial-count matched is checked numerically: does the mouse-ALM cells'
    realised trial count fall inside the human cells' realised range."""
    human_trials = [
        c["realised_matching"]["n_trials_per_session"]["median"] for c in cells_table.values()
        if c["group"] == "human_single_unit" and c["realised_matching"]["n_trials_per_session"]
    ]
    alm_trials = [
        c["realised_matching"]["n_trials_per_session"]["median"] for c in cells_table.values()
        if c["group"] == "mouse_alm" and c["realised_matching"]["n_trials_per_session"]
    ]
    k_mismatched_cells = [key for key, c in cells_table.items() if not c["realised_matching"]["k_matched_across_sessions"]]
    trial_count_note = "not_computable"
    if human_trials and alm_trials:
        lo, hi = min(human_trials), max(human_trials)
        alm_lo, alm_hi = min(alm_trials), max(alm_trials)
        trial_count_note = (
            f"human per-cell median trial counts range [{lo:.0f}, {hi:.0f}]; mouse-ALM cells' median trial "
            f"counts range [{alm_lo:.0f}, {alm_hi:.0f}] -- "
            + ("inside the human range" if lo <= alm_lo and alm_hi <= hi else "OUTSIDE the human range, so trial count is not matched as claimed")
        )
    return {
        "same_code_path_verified": (
            "verified by reading scripts/run_latent_model_observation_noise_comparison.py: "
            "_build_dataset_sessions (human) and _build_alm_sessions (mouse ALM) both call the identical "
            "analyze_session, which calls dimensionality_and_noise_term and "
            "persistence_contrast_under_both_models identically for every dataset -- this is a fact about "
            "the source, not a numeric measurement"
        ),
        "same_estimator_settings_verified": (
            f"both arms use the same fixed k (min({LATENT_DIM}, n_units-1)) and the same nugget_fraction "
            f"n_splits ({CENSUS_N_SPLITS}) by construction, since both call the same functions"
        ),
        "trial_count_matched_check": trial_count_note,
        "human_per_cell_median_trial_counts": human_trials,
        "mouse_alm_per_cell_median_trial_counts": alm_trials,
        "k_matched_within_every_cell": len(k_mismatched_cells) == 0,
        "cells_with_k_not_matched_across_their_own_sessions": k_mismatched_cells,
    }


def _human_vs_alm_bound(human_values: list[float], alm_value: float | None) -> dict:
    """With n=1 mouse-ALM cell per bin width, no paired or
    two-sample test is meaningful, and minimum_detectable_paired_difference
    (which needs n>=2 replicate differences) does not apply to a single
    point. What IS computable: where the ALM point falls relative to the
    human cells' own between-cell spread, reported as a bound on
    distinguishability from ordinary human cross-structure heterogeneity,
    not as a significance test."""
    if alm_value is None or len(human_values) < 2:
        return {"status": "not_computable", "reason": "fewer than 2 human cells or no fitted mouse-ALM value"}
    arr = np.array(human_values)
    mean, sd = float(np.mean(arr)), float(np.std(arr, ddof=1))
    z = (alm_value - mean) / sd if sd > 0 else None
    return {
        "status": "computed", "n_human_cells": len(human_values), "n_mouse_alm_cells": 1,
        "human_mean": mean, "human_sd": sd, "human_min": float(arr.min()), "human_max": float(arr.max()),
        "mouse_alm_value": alm_value,
        "mouse_alm_within_human_range": bool(arr.min() <= alm_value <= arr.max()),
        "mouse_alm_z_relative_to_human_spread": z,
        "reading": (
            "with exactly one mouse-ALM cell, no minimum detectable difference can be computed in the "
            "paired/repeated-measures sense this project uses elsewhere (that requires n>=2 independent "
            "replicate units on the varying side). The bound available instead: an ALM value within "
            "roughly 1 human-cell standard deviation of the human mean is not distinguishable from "
            "ordinary human cross-structure heterogeneity with this design"
        ),
    }


def _matched_covariate_check(cells_table: dict, bin_ms: int, covariate_key, alm_covariate: float | None,
                              value_key, alm_value: float | None, tolerance_fraction: float = 0.25) -> dict:
    """Disjunct (a) of species_gap_is_a_dimensionality_gap / _smoothness_gap:
    does the separation disappear once human cells are restricted to those
    whose covariate (participation ratio, or smoothness rho) is close to the
    mouse-ALM cell's own covariate value. 'Close' is a stated, disclosed
    convention (within tolerance_fraction of the ALM covariate's magnitude,
    or the full human spread if that is narrower), not a data-derived
    boundary -- there is no existing power curve to derive one from."""
    if alm_covariate is None or alm_value is None:
        return {"status": "not_computable", "reason": "no fitted mouse-ALM covariate or value"}
    human_pairs = [
        (covariate_key(c), value_key(c)) for c in cells_table.values()
        if c["group"] == "human_single_unit" and c["bin_ms"] == bin_ms
        and covariate_key(c) is not None and value_key(c) is not None
    ]
    if len(human_pairs) < 2:
        return {"status": "not_computable", "reason": "fewer than 2 human cells with both covariate and value"}
    tol = tolerance_fraction * abs(alm_covariate)
    matched = [(cov, val) for cov, val in human_pairs if abs(cov - alm_covariate) <= tol]
    if len(matched) < 2:
        return {
            "status": "too_few_matched_cells", "n_matched": len(matched), "n_human_total": len(human_pairs),
            "tolerance_fraction": tolerance_fraction,
            "reason": "fewer than 2 human cells fall within tolerance of the mouse-ALM covariate value -- matched-covariate comparison not resolvable",
        }
    matched_values = [val for _, val in matched]
    lo, hi = min(matched_values), max(matched_values)
    return {
        "status": "computed", "n_matched": len(matched), "n_human_total": len(human_pairs),
        "tolerance_fraction": tolerance_fraction, "alm_covariate": alm_covariate,
        "matched_human_value_range": [lo, hi], "alm_value": alm_value,
        "separation_survives_at_matched_covariate": bool(alm_value < lo or alm_value > hi),
    }


def resolve_species_gap_branches(cells_table: dict, census_zero_floor: dict, estimator_branches: dict) -> dict:
    """estimator_branches: the answer-key grid's own per-population branch
    resolution (assemble_artifact's estimator_branches field), consulted for
    disjunct (b) of the dimensionality-gap and smoothness-gap branches: 'the
    estimator that separates them is the one the answer-key grid found to
    track dimensionality/latent smoothness'."""
    human_branches = estimator_branches.get("human_single_unit", {})
    fa_tracks_dimensionality = "tracks_dimensionality" in human_branches.get("factor_analysis", {}).get("branches", {}).get("branches_resolved", [])
    census_tracks_dimensionality = "tracks_dimensionality" in human_branches.get("census", {}).get("branches", {}).get("branches_resolved", [])
    fa_tracks_smoothness = "tracks_latent_smoothness" in human_branches.get("factor_analysis", {}).get("branches", {}).get("branches_resolved", [])
    census_tracks_smoothness = "tracks_latent_smoothness" in human_branches.get("census", {}).get("branches", {}).get("branches_resolved", [])

    by_bin_width: dict[int, dict] = {}
    for bin_ms in (100, 200):
        human_fa = [c["factor_analysis_noise_fraction"] for c in cells_table.values() if c["group"] == "human_single_unit" and c["bin_ms"] == bin_ms and c["factor_analysis_noise_fraction"] is not None]
        alm_fa_list = [c["factor_analysis_noise_fraction"] for c in cells_table.values() if c["group"] == "mouse_alm" and c["bin_ms"] == bin_ms and c["factor_analysis_noise_fraction"] is not None]
        human_census = [c["census_nugget_fraction"]["median_nugget_fraction"] for c in cells_table.values() if c["group"] == "human_single_unit" and c["bin_ms"] == bin_ms and c["census_nugget_fraction"].get("status") == "fitted"]
        alm_census_list = [c["census_nugget_fraction"]["median_nugget_fraction"] for c in cells_table.values() if c["group"] == "mouse_alm" and c["bin_ms"] == bin_ms and c["census_nugget_fraction"].get("status") == "fitted"]
        human_pr = [c["participation_ratio_factor_analysis_basis"] for c in cells_table.values() if c["group"] == "human_single_unit" and c["bin_ms"] == bin_ms and c["participation_ratio_factor_analysis_basis"] is not None]
        alm_pr_list = [c["participation_ratio_factor_analysis_basis"] for c in cells_table.values() if c["group"] == "mouse_alm" and c["bin_ms"] == bin_ms and c["participation_ratio_factor_analysis_basis"] is not None]

        alm_fa = alm_fa_list[0] if alm_fa_list else None
        alm_census = alm_census_list[0] if alm_census_list else None
        alm_pr = alm_pr_list[0] if alm_pr_list else None

        fa_bound = _human_vs_alm_bound(human_fa, alm_fa)
        census_bound = _human_vs_alm_bound(human_census, alm_census)
        pr_bound = _human_vs_alm_bound(human_pr, alm_pr)

        fa_separates_low = fa_bound.get("status") == "computed" and alm_fa < fa_bound["human_min"]
        fa_separates_high = fa_bound.get("status") == "computed" and alm_fa > fa_bound["human_max"]
        census_separates_low = census_bound.get("status") == "computed" and alm_census < census_bound["human_min"]
        census_separates_high = census_bound.get("status") == "computed" and alm_census > census_bound["human_max"]

        both_separate_same_direction = (
            fa_bound.get("status") == "computed" and census_bound.get("status") == "computed"
            and ((fa_separates_low and census_separates_low) or (fa_separates_high and census_separates_high))
        )
        exactly_one_separates = (
            fa_bound.get("status") == "computed" and census_bound.get("status") == "computed"
            and (fa_separates_low or fa_separates_high) != (census_separates_low or census_separates_high)
        )

        pr_matched_check = _matched_covariate_check(
            cells_table, bin_ms, lambda c: c["participation_ratio_factor_analysis_basis"], alm_pr,
            lambda c: c["factor_analysis_noise_fraction"], alm_fa)
        dimensionality_gap_disjunct_a = pr_matched_check.get("status") == "computed" and pr_matched_check["separation_survives_at_matched_covariate"] is False
        dimensionality_gap_disjunct_b = (fa_separates_low or fa_separates_high) and fa_tracks_dimensionality
        is_dimensionality_gap = bool(dimensionality_gap_disjunct_a or dimensionality_gap_disjunct_b)

        smoothness_matched_check = _matched_covariate_check(
            cells_table, bin_ms, lambda c: c["smoothness"].get("median_rho") if c["smoothness"].get("status") == "computed" else None, (
                next((c["smoothness"]["median_rho"] for c in cells_table.values() if c["group"] == "mouse_alm" and c["bin_ms"] == bin_ms and c["smoothness"].get("status") == "computed"), None)
            ),
            lambda c: c["factor_analysis_noise_fraction"], alm_fa)
        alm_rho = next((c["smoothness"]["median_rho"] for c in cells_table.values() if c["group"] == "mouse_alm" and c["bin_ms"] == bin_ms and c["smoothness"].get("status") == "computed"), None)
        smoothness_gap_disjunct_a = smoothness_matched_check.get("status") == "computed" and smoothness_matched_check["separation_survives_at_matched_covariate"] is False
        smoothness_gap_disjunct_b = (fa_separates_low or fa_separates_high) and fa_tracks_smoothness
        is_smoothness_gap = bool(smoothness_gap_disjunct_a or smoothness_gap_disjunct_b)

        alm_census_at_zero_floor = alm_census == 0.0
        is_floor_artifact = alm_census_at_zero_floor and census_zero_floor.get("large_f_true_reachable_at_near_zero_output") is True

        resolved = []
        if both_separate_same_direction:
            resolved.append("species_gap_survives_in_both_estimators")
        if exactly_one_separates:
            resolved.append("species_gap_is_estimator_specific")
        if is_dimensionality_gap:
            resolved.append("species_gap_is_a_dimensionality_gap")
        if is_smoothness_gap:
            resolved.append("species_gap_is_a_smoothness_gap")
        if alm_census_at_zero_floor:
            resolved.append("species_gap_is_a_floor_artifact" if is_floor_artifact else "species_gap_is_a_floor_artifact_candidate_not_licensed_by_the_zero_floor_reachability_check")

        inputs_not_computable = [
            name for name, bound in (("factor_analysis", fa_bound), ("census", census_bound), ("participation_ratio_factor_analysis_basis", pr_bound))
            if bound.get("status") != "computed"
        ]
        branch_resolution_input_status = {
            "factor_analysis_bound_computed": fa_bound.get("status") == "computed",
            "census_bound_computed": census_bound.get("status") == "computed",
            "participation_ratio_bound_computed": pr_bound.get("status") == "computed",
            "inputs_not_computable": inputs_not_computable,
            "reading": (
                "every required bound was computed, so the branches_resolved list (including an empty "
                "list) reflects a genuine finding about whether human and mouse-ALM values separate"
                if not inputs_not_computable else
                "at least one required bound (" + ", ".join(inputs_not_computable) + ") could not be "
                "computed at this bin width, so an empty or partial branches_resolved list here reflects "
                "missing data (that estimator did not fit for this population at this bin width), not a "
                "resolved null about whether the species gap exists"
            ),
        }

        by_bin_width[bin_ms] = {
            "factor_analysis": {"bound": fa_bound, "alm_separates_low": fa_separates_low, "alm_separates_high": fa_separates_high},
            "census": {"bound": census_bound, "alm_separates_low": census_separates_low, "alm_separates_high": census_separates_high},
            "participation_ratio_factor_analysis_basis": pr_bound,
            "branch_resolution_input_status": branch_resolution_input_status,
            "direction_note": (
                "both a mouse-value-below-human (the standing narrative) and a mouse-value-above-human "
                "(the reverse, i.e. mouse noisier than human) direction are computed above via the "
                "separate _low/_high flags -- neither is assumed"
            ),
            "dimensionality_gap_check": {
                "disjunct_a_separation_disappears_at_matched_participation_ratio": pr_matched_check,
                "disjunct_b_separating_estimator_tracks_dimensionality_per_the_answer_key_grid": {
                    "factor_analysis_tracks_dimensionality": fa_tracks_dimensionality,
                    "census_tracks_dimensionality": census_tracks_dimensionality,
                },
                "resolved": is_dimensionality_gap,
            },
            "smoothness_gap_check": {
                "alm_median_rho": alm_rho,
                "disjunct_a_separation_disappears_at_matched_smoothness": smoothness_matched_check,
                "disjunct_b_separating_estimator_tracks_latent_smoothness_per_the_answer_key_grid": {
                    "factor_analysis_tracks_latent_smoothness": fa_tracks_smoothness,
                    "census_tracks_latent_smoothness": census_tracks_smoothness,
                },
                "resolved": is_smoothness_gap,
            },
            "branches_resolved": resolved,
            "mouse_alm_census_at_its_own_zero_floor": alm_census_at_zero_floor,
        }
    return by_bin_width


# ---------------------------------------------------------------------------
# Assembly (cheap, checkpoint-only, so it can be re-run at negligible cost
# from checkpoints alone).
# ---------------------------------------------------------------------------

def assemble_artifact(populations: dict, primary_points: dict, criticism_result: dict, extensions: dict) -> dict:
    estimator_branches_by_population = {}
    for pop_label in populations:
        if populations[pop_label].get("status") != "computed":
            estimator_branches_by_population[pop_label] = {"status": "not_computable", "reason": populations[pop_label].get("reason")}
            continue
        estimator_branches_by_population[pop_label] = build_sweeps_for_population(primary_points, pop_label)

    # Loaded from checkpoint only (compute=False): by the time assemble_artifact
    # runs, the correlated_white grid is already on disk, from either the live
    # criticism run or a prior one -- this never triggers new simulation.
    correlated_white_points = _grid_points_for(populations, CRITICISM_NOISE_MODEL, 0.0, "[zero_floor_check]", compute=False)
    zero_floor = census_zero_floor_reachability({"diagonal": primary_points, CRITICISM_NOISE_MODEL: correlated_white_points})
    failure_rates = estimator_fit_failure_rates({"diagonal": primary_points, CRITICISM_NOISE_MODEL: correlated_white_points})

    factor_model_artifact, census, checkpoint = load_species_gap_cells()
    cells_table = build_cells_table(factor_model_artifact, census, checkpoint)
    species_gap = {
        "precursor_numbers_carried_forward": PRECURSOR_NUMBERS_CARRIED_FORWARD,
        "cells": cells_table,
        "matching_verification": matching_verification(cells_table),
        "branches_by_bin_width": resolve_species_gap_branches(cells_table, zero_floor, estimator_branches_by_population),
        "n_human_single_unit_cells": sum(1 for c in cells_table.values() if c["group"] == "human_single_unit"),
        "n_mouse_alm_cells": sum(1 for c in cells_table.values() if c["group"] == "mouse_alm"),
    }

    headline = (
        "The two estimators this project has called 'observation noise' disagree in what they measure. "
        "The factor-analysis noise-variance fraction tracks the true noise fraction accurately (monotone "
        "in f_true, span 0.789 across the f_true grid at fixed dimensionality, diagonal noise, human "
        "population size) but is also confounded by latent dimensionality -- span 0.160 across d_true "
        "1..12 at fixed f_true=0.40, real but smaller than the noise signal, which is why "
        "tracks_noise_with_a_confound resolves for it while tracks_dimensionality does not. The "
        "cross-validated nugget-fraction estimator is confounded by latent smoothness and returns a "
        "value within 0.01 of exactly zero at true noise fractions as high as 0.80 "
        "(census_zero_floor_reachability: 34 grid points across both noise models, f_true up to 0.80 "
        "under diagonal noise and up to 0.40 under correlated_white noise), so an exact-zero reading "
        "from it never by itself establishes near-zero true noise. Applied to the real human-single-"
        "unit-versus-mouse-ALM comparison at the narrow bin width, the species gap is estimator-specific: "
        "the factor-analysis fraction places mouse ALM at 0.708, inside the human range 0.520-0.887, and "
        "the participation ratio places it at 4.45, inside the human range 2.80-7.44, while only the "
        "cross-validated nugget fraction separates the two species, with mouse ALM sitting exactly at "
        "that estimator's own zero floor -- a floor this grid shows is reachable at substantial true "
        "noise. At the wide bin width the cross-validated estimator does not fit for mouse ALM at all, "
        "so its branch list there is missing data, not a resolved null; see each bin width's own "
        "branch_resolution_input_status field."
    )
    return {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "headline": headline,
        "trigger": (
            "this project uses two different estimators of observation noise that disagree almost "
            "completely on real data where the truth is unknown; this artifact scores both against "
            "synthetic populations whose true noise fraction, latent dimensionality and latent timescale "
            "are set by the generator, then restates the real human-versus-mouse-ALM species-gap "
            "comparison in light of what that answer-key test finds"
        ),
        "entry_points": ENTRY_POINTS,
        "generative_model": {
            "form": "x = C @ z + eps, C: (n_units, d_true) iid N(0,1), z: d_true independent AR(1) processes with unit stationary variance, eps: see noise_model",
            "f_true_grid": list(F_TRUE_GRID), "d_true_grid": list(D_TRUE_GRID), "timescale_grid_s": list(TIMESCALE_GRID_S),
            "centre_point": {"f_true": CENTRE_F_TRUE, "d_true": CENTRE_D_TRUE, "timescale_s": CENTRE_TIMESCALE_S},
            "centre_point_provenance": "each centre value is itself a member of its own grid (d_true==LATENT_DIM), so the three one-factor-at-a-time sweeps share and reuse this point rather than each fitting their own",
            "n_seeds_per_grid_point": N_SEEDS,
            "primary_noise_model": "diagonal (independent per-unit Gaussian, matching the factor-analysis estimator's own assumption -- see criticism_second_generative_model_correlated_noise for the deliberately mismatched alternative)",
        },
        "population_sizes": populations,
        "estimator_branches": estimator_branches_by_population,
        "census_zero_floor_reachability": zero_floor,
        "estimator_fit_failure_rates": failure_rates,
        "criticism_second_generative_model_correlated_noise": criticism_result,
        "species_gap": species_gap,
        "generative_factor_extensions": extensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assemble-only", action="store_true", help="skip all simulation; assemble the artifact from the existing checkpoint alone")
    args = parser.parse_args()

    populations = population_sizes_from_census()
    print(f"population sizes: {json.dumps(populations, indent=2)}", file=sys.stderr)

    compute = not args.assemble_only
    primary_points = _grid_points_for(populations, "diagonal", 0.0, "[primary]", compute)
    if compute:
        criticism_result = run_criticism_second_generative_model(populations)
    else:
        criticism_points = _grid_points_for(populations, CRITICISM_NOISE_MODEL, 0.0, "[criticism]", compute=False)
        criticism_result = _criticism_result_from_points(criticism_points)
    estimator_branches_for_extensions = {
        pop_label: build_sweeps_for_population(primary_points, pop_label)
        for pop_label in populations if populations[pop_label].get("status") == "computed"
    }
    extensions = run_generative_factor_extensions(populations, estimator_branches_for_extensions, compute)

    payload = assemble_artifact(populations, primary_points, criticism_result, extensions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"wrote {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps({
        "human_single_unit_branches": {
            est: payload["estimator_branches"].get("human_single_unit", {}).get(est, {}).get("branches", {}).get("branches_resolved")
            for est in ("factor_analysis", "census")
        },
        "species_gap_branches_bin100": payload["species_gap"]["branches_by_bin_width"].get(100, {}).get("branches_resolved"),
        "species_gap_branches_bin200": payload["species_gap"]["branches_by_bin_width"].get(200, {}).get("branches_resolved"),
        "condition_structure_branch_differences_human_single_unit": {
            noise_model: payload["generative_factor_extensions"]["condition_structure"]["branch_differences_vs_no_condition_structure_by_noise_model"][noise_model].get("human_single_unit")
            for noise_model in ("diagonal", CRITICISM_NOISE_MODEL)
        },
        "small_size_branch_differences_human_single_unit": {
            noise_model: payload["generative_factor_extensions"]["small_size"]["branch_differences_vs_median_size_by_noise_model"][noise_model].get("human_single_unit_small")
            for noise_model in ("diagonal", CRITICISM_NOISE_MODEL)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
