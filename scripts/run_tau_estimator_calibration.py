#!/usr/bin/env python3
"""Positive control for the lag-0-excluded intrinsic timescale estimator
(`run_intrinsic_timescale_vs_confinement._fit_tau`).

Plants a known tau in an OU latent driving a Poisson spike-count emission, at
the actual trial count, bin width and baseline-window length that
`run_intrinsic_timescale_vs_confinement.py` uses on DANDI 000469. This table
is the estimator's power curve: it decides whether the tau-vs-1/lambda
comparison is even measurable in this corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
import run_human_drift_spine_000469 as spine469  # noqa: E402
import run_intrinsic_timescale_vs_confinement as tau_mod  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "tau_estimator_calibration.json"
SEED = 20260807

PLANTED_TAUS_S = [0.05, 0.1, 0.2, 0.35, 0.5]
MEAN_RATES_HZ = [1.0, 2.0, 5.0, 10.0, 20.0]
N_REPS = 200
MODULATION_DEPTH = 0.5  # latent x has stationary sd 1; rate = mean*(1 + depth*x)

PREDECLARED_DECISION = {
    "supported": (
        "corrected tau is resolvable (this table) at the observed rates and window, AND the "
        "per-structure tau ordering has non-overlapping bootstrap intervals for at least one "
        "structure pair."
    ),
    "refuted": (
        "corrected tau is resolvable and the orderings of tau and 1/lambda agree for the "
        "pre-SMA-vs-hippocampus pair; task-driven confinement is then not dissociable from "
        "intrinsic autocorrelation in this corpus, and the Murray comparison collapses to a "
        "replication."
    ),
    "estimator_non_identified": (
        "this table shows the corrected estimator is NOT resolvable at the observed rates and "
        "window length. Then tau is NOT MEASURABLE in this corpus, the Murray comparison is "
        "WITHDRAWN rather than reported, and the artifact records what window length would be "
        "required, computed from this table."
    ),
    "note": (
        "a 0.9 s window is expected on prior grounds to leave the corrected estimator "
        "unresolvable; this is a prediction, not a preference. Neither of the first two "
        "branches is preferred."
    ),
}


def _observed_trial_counts() -> dict[str, int]:
    """Actual per-patient trial counts clearing the baseline-window gap check,
    read from the DANDI 000469 NWB files (not assumed)."""
    directory = spine469.data_directory()
    counts: dict[str, int] = {}
    for patient_dir in sorted(directory.iterdir()):
        if not patient_dir.is_dir():
            continue
        path = patient_dir / f"{patient_dir.name}_ses-2_ecephys+image.nwb"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            trials = handle["intervals/trials"]
            fixation = trials["timestamps_FixationCross"][:]
            encoding = trials["timestamps_Encoding1"][:]
        gap = encoding - fixation
        counts[patient_dir.name] = int(np.sum(gap >= tau_mod.BASELINE_WINDOW_S))
    return counts


def _simulate_counts(true_tau: float, rate: float, n_trials: int, rng: np.random.Generator,
                      n_bins: int = tau_mod.N_BINS) -> np.ndarray:
    phi = np.exp(-tau_mod.BIN_S / true_tau)
    sigma = np.sqrt(max(1.0 - phi ** 2, 0.0))
    x = rng.normal(size=n_trials)
    counts = np.zeros((n_trials, n_bins))
    for b in range(n_bins):
        x = phi * x + sigma * rng.normal(size=n_trials)
        lam = np.maximum(rate * (1.0 + MODULATION_DEPTH * x), 0.05 * rate)
        counts[:, b] = rng.poisson(lam * tau_mod.BIN_S)
    return counts


def _cell(true_tau: float, rate: float, n_trials: int, rng: np.random.Generator,
          n_reps: int = N_REPS, n_bins: int = tau_mod.N_BINS, exclude_lag0: bool = True) -> dict:
    recovered = []
    for _ in range(n_reps):
        counts = _simulate_counts(true_tau, rate, n_trials, rng, n_bins=n_bins)
        tau_hat, _ = tau_mod._fit_tau(counts, exclude_lag0=exclude_lag0)
        if tau_hat is not None:
            recovered.append(tau_hat)
    converged_fraction = len(recovered) / n_reps
    if not recovered:
        return {
            "planted_tau_s": true_tau, "mean_rate_hz": rate, "n_reps": n_reps,
            "converged_fraction": 0.0, "median_recovered_tau_s": None,
            "recovered_tau_ci95": [None, None], "resolvable": False,
        }
    arr = np.asarray(recovered)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "planted_tau_s": true_tau, "mean_rate_hz": rate, "n_reps": n_reps,
        "converged_fraction": converged_fraction,
        "median_recovered_tau_s": float(np.median(arr)),
        "recovered_tau_ci95": [float(lo), float(hi)],
        "resolvable": None,  # filled in below, needs the full grid for neighbours
    }


def _mark_resolvable(cells: list[dict], group_key: str = "mean_rate_hz") -> None:
    """resolvable: the recovered-tau interval excludes both neighbouring
    planted tau values in the grid."""
    groups = sorted({c[group_key] for c in cells})
    for group_value in groups:
        row = sorted([c for c in cells if c[group_key] == group_value], key=lambda c: c["planted_tau_s"])
        for i, cell in enumerate(row):
            if cell["median_recovered_tau_s"] is None:
                cell["resolvable"] = False
                continue
            lo, hi = cell["recovered_tau_ci95"]
            neighbours = []
            if i > 0:
                neighbours.append(row[i - 1]["planted_tau_s"])
            if i < len(row) - 1:
                neighbours.append(row[i + 1]["planted_tau_s"])
            cell["resolvable"] = bool(all(v < lo or v > hi for v in neighbours))


WINDOW_SWEEP_RATE_HZ = 5.0  # a representative mid-range baseline firing rate
WINDOW_SWEEP_S = [0.9, 1.8, 3.6, 7.2, 14.4, 28.8, 57.6]
WINDOW_SWEEP_REPS = 100


def _window_length_sweep(n_trials: int, rng: np.random.Generator) -> dict:
    """If the observed 0.9 s window is not resolvable, compute what window
    length would be, by extending the same simulation (fixed representative
    rate, all 5 planted
    taus) to longer windows until the estimator resolves the grid."""
    cells = []
    for window_s in WINDOW_SWEEP_S:
        n_bins = int(round(window_s / tau_mod.BIN_S))
        for tau in PLANTED_TAUS_S:
            cell = _cell(tau, WINDOW_SWEEP_RATE_HZ, n_trials, rng, n_reps=WINDOW_SWEEP_REPS, n_bins=n_bins)
            cell["window_s"] = window_s
            cells.append(cell)
    _mark_resolvable(cells, group_key="window_s")
    resolvable_by_window = {
        str(w): sum(1 for c in cells if c["window_s"] == w and c["resolvable"]) / len(PLANTED_TAUS_S)
        for w in WINDOW_SWEEP_S
    }
    first_fully_resolvable = next(
        (w for w in WINDOW_SWEEP_S if resolvable_by_window[str(w)] == 1.0), None
    )
    return {
        "rate_hz": WINDOW_SWEEP_RATE_HZ, "n_reps_per_cell": WINDOW_SWEEP_REPS,
        "windows_s": WINDOW_SWEEP_S, "cells": cells,
        "resolvable_fraction_by_window": resolvable_by_window,
        "first_window_s_with_all_taus_resolvable": first_fully_resolvable,
        "note": (
            "extends the calibration to longer task-free windows at a fixed representative rate "
            "to estimate the window length the corrected estimator would need; not itself a claim "
            "about any real corpus's window length."
        ),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    observed_trial_counts = _observed_trial_counts()
    trial_count_values = sorted(set(observed_trial_counts.values()))
    n_trials = max(set(observed_trial_counts.values()), key=list(observed_trial_counts.values()).count)

    cells = []
    for rate in MEAN_RATES_HZ:
        for tau in PLANTED_TAUS_S:
            cells.append(_cell(tau, rate, n_trials, rng))
    _mark_resolvable(cells)

    n_resolvable = sum(1 for c in cells if c["resolvable"])
    resolvable_fraction = n_resolvable / len(cells)
    window_length_sweep = None if resolvable_fraction > 0 else _window_length_sweep(n_trials, rng)

    # Figure-only comparison (6.1): the same grid fit with lag 0 included (the pre-fix
    # behaviour), at a reduced rep count since this is illustrative, not decision-relevant --
    # the decision rule above uses only the corrected (exclude_lag0=True) cells.
    lag0_inclusive_comparison = [
        _cell(tau, rate, n_trials, rng, n_reps=50, exclude_lag0=False)
        for rate in MEAN_RATES_HZ for tau in PLANTED_TAUS_S
    ]

    output = {
        "schema_version": "1.0.0", "analysis_id": "tau_estimator_calibration",
        "trigger": "positive control for the lag-0-exclusion fix to the intrinsic-timescale estimator",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "scope": (
            "DANDI 000469 baseline-window (pre-stimulus fixation) tau estimator only; the "
            "calibration parameters (n_trials, bin width, window length) match this corpus and "
            "are not assumed to generalise to 000574/001187/000673, which are out of scope this "
            "round (see run_intrinsic_timescale_vs_confinement.py scope note)."
        ),
        "simulation_model": (
            "OU latent x (stationary sd 1, autocorrelation time = planted tau) drives a Poisson "
            "spike-count emission per 50 ms bin: rate_b = max(mean_rate*(1+0.5*x_b), 0.05*mean_rate), "
            "count_b ~ Poisson(rate_b * bin_width). Fit with _fit_tau(exclude_lag0=True), the "
            "corrected estimator."
        ),
        "n_trials_used": n_trials,
        "n_trials_provenance": (
            f"modal per-patient trial count clearing the baseline-window gap check, read directly "
            f"from the DANDI 000469 NWB files (observed values: {trial_count_values}, "
            f"per-patient: {observed_trial_counts})."
        ),
        "bin_width_s": tau_mod.BIN_S,
        "baseline_window_s": tau_mod.BASELINE_WINDOW_S,
        "planted_taus_s": PLANTED_TAUS_S,
        "mean_rates_hz": MEAN_RATES_HZ,
        "n_reps_per_cell": N_REPS,
        "cells": cells,
        "resolvable_fraction_overall": resolvable_fraction,
        "resolvable_by_rate": {
            str(rate): sum(1 for c in cells if c["mean_rate_hz"] == rate and c["resolvable"]) / len(PLANTED_TAUS_S)
            for rate in MEAN_RATES_HZ
        },
        "window_length_sweep": window_length_sweep,
        "lag0_inclusive_comparison_for_figure": lag0_inclusive_comparison,
        "predeclared_decision": PREDECLARED_DECISION,
    }
    OUTPUT_PATH.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(OUTPUT_PATH), "n_cells": len(cells),
        "resolvable_fraction_overall": resolvable_fraction,
        "first_window_s_with_all_taus_resolvable": (
            window_length_sweep["first_window_s_with_all_taus_resolvable"] if window_length_sweep else None
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
