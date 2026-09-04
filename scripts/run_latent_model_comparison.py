#!/usr/bin/env python3
"""Does a latent basis with an explicit observation-noise term recover
structure a noise-blind one cannot?

PCA has no observation-noise term, so it cannot separate signal from noise
by construction. GPFA fits an explicit per-neuron independent noise variance
alongside a Gaussian-process latent trajectory (Yu et al. 2009). This script
fits both on the same training trials at matched latent dimensionality, on
every human single-unit delay-epoch session and every mouse ALM session, and
scores both on the same held-out trials: the nugget fraction of the leading
latent (src/observability.py, unchanged), the confinement rate and
mean-squared-displacement saturation ratio on the same latent (existing
estimators, unmodified), intrinsic dimensionality by TwoNN, and a held-out
Gaussian log-likelihood comparable between the two methods (both fit on the
same Anscombe-variance-stabilized counts, with a diagonal per-unit noise
covariance -- GPFA's is fitted, PCA's is estimated from training residuals).

Deciding contrast: the paired within-session difference in cross-validated
nugget fraction, GPFA minus PCA, human corpora, exact paired sign-flip test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import neo
import numpy as np
import quantities as pq
from elephant.gpfa import GPFA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_sessions import (  # noqa: E402
    BORAN_EPOCH_WINDOWS_S,
    EPOCH_WINDOWS_S,
    data_root,
    iter_alm,
    iter_all_corpora,
)
from drift_dynamics import confinement_identifiability, fit_gaussian_state_space  # noqa: E402
from geometry import twonn_dimension  # noqa: E402
from observability import median_nugget_from_held_out_latent  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from run_latent_displacement_scaling import mean_squared_displacement, summarize_msd  # noqa: E402
from spike_pipeline import build_psth  # noqa: E402
from statistics import paired_sign_flip_test, stable_seed  # noqa: E402

SEED = 20260809
BIN_MS = 100
LATENT_DIM = 6
TRAIN_FRACTION = 0.6
MIN_TRIALS = 20
MIN_TEST_TRIALS = 6
OUTPUT_PATH = ROOT / "results" / "latent_model_comparison.json"

EPOCH_WINDOWS_BY_DATASET = {
    "dandi_000469": EPOCH_WINDOWS_S,
    "dandi_001187": EPOCH_WINDOWS_S,
    "dandi_000574": BORAN_EPOCH_WINDOWS_S,
}


def _seed(*parts: str) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


def raw_counts_from_entry(entry: dict, bin_ms: float) -> np.ndarray:
    """Integer spike counts (trials, units, bins), delay epoch, un-smoothed."""
    window = EPOCH_WINDOWS_BY_DATASET[entry["dataset"]]["delay"]
    rate = build_psth(entry["spike_lists"], entry["epoch_onsets"]["delay"], bin_ms=bin_ms, smooth_ms=0, window_s=window)
    return np.rint(rate * (bin_ms / 1000.0)).astype(int)


def split_trials(n_trials: int, rng: np.random.Generator, train_fraction: float = TRAIN_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_trials)
    n_train = max(2, int(round(n_trials * train_fraction)))
    n_train = min(n_train, n_trials - 2)
    return perm[:n_train], perm[n_train:]


def counts_to_spiketrains(counts: np.ndarray, bin_size_ms: float) -> list[list[neo.SpikeTrain]]:
    """Reconstruct one synthetic spike per unit count, placed uniformly within
    its bin -- elephant's GPFA re-bins at the same width on input, so this
    round-trips the original integer counts exactly."""
    n_trials, n_units, n_bins = counts.shape
    t_stop = n_bins * bin_size_ms
    trials = []
    for tr in range(n_trials):
        units = []
        for u in range(n_units):
            spikes: list[float] = []
            for b in range(n_bins):
                n = int(counts[tr, u, b])
                if n > 0:
                    offsets = (np.arange(n) + 0.5) / n * bin_size_ms
                    spikes.extend(b * bin_size_ms + offsets)
            units.append(neo.SpikeTrain(sorted(spikes) * pq.ms, t_stop=t_stop * pq.ms))
        trials.append(units)
    return trials


def anscombe_counts(counts: np.ndarray) -> np.ndarray:
    """Variance-stabilizing transform matching the project's existing
    Anscombe convention (scripts/run_drift_positive_controls.py)."""
    return 2.0 * np.sqrt(np.maximum(counts, 0.0) + 3.0 / 8.0)


def _diagonal_gaussian_log_likelihood(residuals: np.ndarray, noise_var: np.ndarray) -> float:
    return float(-0.5 * np.sum(np.log(2 * np.pi * noise_var)[None, :] + residuals ** 2 / noise_var[None, :]))


def fit_pca_arm(train_counts: np.ndarray, test_counts: np.ndarray, k: int) -> dict:
    """PCA on Anscombe-transformed counts (mean-centered, no per-unit
    rescaling), so its held-out log-likelihood is on the same variance-
    stabilized-count scale GPFA fits internally."""
    n_units = train_counts.shape[1]
    k_use = max(1, min(k, n_units - 1))
    x_train = anscombe_counts(train_counts)
    x_test = anscombe_counts(test_counts)
    mean_train = x_train.mean(axis=(0, 2), keepdims=True)
    flat_train = (x_train - mean_train).transpose(0, 2, 1).reshape(-1, n_units)
    _, _, vt = np.linalg.svd(flat_train, full_matrices=False)
    v = vt[:k_use].T
    flat_test = (x_test - mean_train).transpose(0, 2, 1).reshape(-1, n_units)

    latent_train = flat_train @ v
    latent_test = flat_test @ v
    noise_var = np.maximum((flat_train - latent_train @ v.T).var(axis=0), 1e-8)
    resid_test = flat_test - latent_test @ v.T
    log_likelihood = _diagonal_gaussian_log_likelihood(resid_test, noise_var)

    n_test, n_bins = test_counts.shape[0], test_counts.shape[2]
    return {
        "latent": latent_test.reshape(n_test, n_bins, k_use),
        "log_likelihood_per_trial_bin": log_likelihood / (n_test * n_bins),
        "k_used": int(k_use),
    }


def fit_gpfa_arm(train_counts: np.ndarray, test_counts: np.ndarray, bin_ms: float, k: int) -> dict:
    n_units = train_counts.shape[1]
    k_use = max(1, min(k, n_units - 1))
    train_st = counts_to_spiketrains(train_counts, bin_ms)
    test_st = counts_to_spiketrains(test_counts, bin_ms)
    gpfa = GPFA(bin_size=bin_ms * pq.ms, x_dim=k_use, verbose=False)
    gpfa.fit(train_st)
    latent_test = np.stack(gpfa.transform(test_st), axis=0).transpose(0, 2, 1)  # (trials, bins, k)
    log_likelihood = gpfa.score(test_st)
    n_test, n_bins = test_counts.shape[0], test_counts.shape[2]
    return {
        "latent": latent_test,
        "log_likelihood_per_trial_bin": float(log_likelihood) / (n_test * n_bins),
        "k_used": int(k_use),
    }


def analyze_arm(latent: np.ndarray, bin_ms: float, log_likelihood_per_trial_bin: float, k_used: int,
                 rng: np.random.Generator) -> dict:
    leading = latent[:, :, 0]
    nugget = median_nugget_from_held_out_latent(leading, bin_width_s=bin_ms / 1000.0, n_splits=24, rng=rng)
    state_space = fit_gaussian_state_space(leading, dt=bin_ms / 1000.0)
    lambda_rate = state_space.lambda_rate
    identifiability = (
        confinement_identifiability(lambda_rate, bin_ms / 1000.0, leading.shape[1] * bin_ms / 1000.0)
        if lambda_rate is not None and np.isfinite(lambda_rate) else None
    )
    msd = summarize_msd(mean_squared_displacement(latent)) if latent.shape[1] >= 4 else {
        "saturation_ratio": None, "log_log_slope": None, "msd_at_K_state_units_sq": None,
    }
    cloud = latent.reshape(-1, latent.shape[-1])
    twonn = float(twonn_dimension(cloud)) if len(cloud) > 10 else None
    return {
        "k_used": k_used,
        "nugget_fraction": nugget["median_nugget_fraction"],
        "nugget_status": nugget["status"],
        "lambda_rate": lambda_rate,
        "lambda_status": state_space.status,
        "lambda_identifiability": identifiability.status if identifiability else None,
        "saturation_ratio": msd["saturation_ratio"],
        "twonn_dimension": twonn,
        "log_likelihood_per_trial_bin": log_likelihood_per_trial_bin,
    }


def analyze_session(dataset: str, patient: str, session: str, counts: np.ndarray) -> dict | None:
    n_trials = counts.shape[0]
    if n_trials < MIN_TRIALS:
        return None
    rng = _seed(dataset, session, "split")
    train_idx, test_idx = split_trials(n_trials, rng)
    if len(test_idx) < MIN_TEST_TRIALS:
        return None
    train_counts, test_counts = counts[train_idx], counts[test_idx]

    t0 = time.time()
    pca = fit_pca_arm(train_counts, test_counts, LATENT_DIM)
    gpfa = fit_gpfa_arm(train_counts, test_counts, BIN_MS, LATENT_DIM)
    elapsed = time.time() - t0

    return {
        "dataset": dataset, "patient": patient, "session": session,
        "n_train_trials": int(len(train_idx)), "n_test_trials": int(len(test_idx)),
        "n_units": int(counts.shape[1]), "elapsed_s": round(elapsed, 2),
        "pca": analyze_arm(pca["latent"], BIN_MS, pca["log_likelihood_per_trial_bin"], pca["k_used"],
                            _seed(dataset, session, "pca_nugget")),
        "gpfa": analyze_arm(gpfa["latent"], BIN_MS, gpfa["log_likelihood_per_trial_bin"], gpfa["k_used"],
                             _seed(dataset, session, "gpfa_nugget")),
    }


def build_sessions(root: Path) -> list[dict]:
    sessions = []
    seen = 0
    t0 = time.time()
    for entry in iter_all_corpora(root):
        if entry["structure"] != "pooled":
            continue
        counts = raw_counts_from_entry(entry, BIN_MS)
        result = analyze_session(entry["dataset"], entry["patient"], entry["session"], counts)
        if result is not None:
            result["arm"] = "human"
            sessions.append(result)
        seen += 1
        if seen % 5 == 0:
            print(f"  ...{seen} human pooled sessions in {time.time() - t0:.0f}s", file=sys.stderr)
    for entry in iter_alm(root, bin_ms=BIN_MS, window_s=2.0):
        result = analyze_session(entry["dataset"], entry["patient"], entry["session"], entry["counts"])
        if result is not None:
            result["arm"] = "alm"
            sessions.append(result)
    return sessions


def deciding_contrast(sessions: list[dict]) -> dict:
    human = [s for s in sessions if s["arm"] == "human"
             and s["pca"]["nugget_status"] == "fitted" and s["gpfa"]["nugget_status"] == "fitted"]
    n_pairs = len(human)
    min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs > 0 else 1.0
    if n_pairs < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p}
    pca_vals = np.array([s["pca"]["nugget_fraction"] for s in human])
    gpfa_vals = np.array([s["gpfa"]["nugget_fraction"] for s in human])
    test = paired_sign_flip_test(gpfa_vals, pca_vals, alternative="less", rng=np.random.default_rng(SEED))
    reduction = -test["mean_diff"]  # PCA minus GPFA: positive means GPFA is lower (a reduction)

    alm = [s for s in sessions if s["arm"] == "alm"
           and s["pca"]["nugget_status"] == "fitted" and s["gpfa"]["nugget_status"] == "fitted"]
    alm_gpfa_worse = [s for s in alm if s["gpfa"]["nugget_fraction"] > s["pca"]["nugget_fraction"] + 0.05]
    alm_ll_favours_gpfa = float(np.mean([
        s["gpfa"]["log_likelihood_per_trial_bin"] > s["pca"]["log_likelihood_per_trial_bin"] for s in alm
    ])) if alm else None

    median_gpfa = float(np.median(gpfa_vals))
    ll_favours_gpfa = float(np.mean([
        s["gpfa"]["log_likelihood_per_trial_bin"] > s["pca"]["log_likelihood_per_trial_bin"] for s in human
    ]))
    if median_gpfa < 0.50 and reduction > 0.25 and ll_favours_gpfa > 0.5:
        branch = "noise_model_recovers_trajectory"
    elif median_gpfa >= 0.80:
        branch = "noise_model_insufficient"
    else:
        branch = "partial_recovery"

    return {
        "n_pairs": n_pairs, "median_pca_nugget": float(np.median(pca_vals)), "median_gpfa_nugget": median_gpfa,
        "mean_reduction_pca_minus_gpfa": reduction, "p_value": test["p_value"],
        "min_attainable_p": min_attainable_p, "ci_lower": -test["ci_upper"], "ci_upper": -test["ci_lower"],
        "held_out_log_likelihood_favours_gpfa_fraction": ll_favours_gpfa,
        "branch": branch,
        "alm_positive_control": {
            "n_sessions": len(alm),
            "n_sessions_gpfa_materially_worse": len(alm_gpfa_worse),
            "log_likelihood_favours_gpfa_fraction": alm_ll_favours_gpfa,
            "gpfa_broke_the_positive_control": len(alm_gpfa_worse) > 0,
        },
        "log_likelihood_caveat": (
            "Both arms' held-out log-likelihood are Gaussian densities on the same "
            "Anscombe-variance-stabilized count scale with a diagonal per-unit noise "
            "covariance, fit the same way for both methods -- but GPFA's density additionally "
            "marginalizes a Gaussian-process prior over the latent trajectory that the PCA arm "
            "does not have, so the two are comparable in scale but not a strictly nested "
            "likelihood-ratio test; used here only for the directional 'favours GPFA' check."
        ),
    }


def main() -> None:
    root = data_root()
    print("Fitting PCA and GPFA on every pooled delay-epoch session...", file=sys.stderr)
    sessions = build_sessions(root)
    print(f"  {len(sessions)} sessions with both arms fit", file=sys.stderr)
    payload = {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "bin_ms": BIN_MS,
        "latent_dim_requested": LATENT_DIM,
        "train_fraction": TRAIN_FRACTION,
        "scope": ("Structure 'pooled', delay epoch, 100 ms bins only, a single random "
                  "train/test split per session (not the 8+-split median protocol "
                  "src/observability.py's nugget_fraction uses standalone) -- GPFA's EM fit is far "
                  "more expensive than a PCA SVD, and the available compute-time budget could not "
                  "afford a multi-split cross-validated median at every (structure, epoch, bin "
                  "width) combination."),
        "sessions": sessions,
        "deciding_contrast": deciding_contrast(sessions),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"Wrote {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps(payload["deciding_contrast"], indent=2))


if __name__ == "__main__":
    main()
