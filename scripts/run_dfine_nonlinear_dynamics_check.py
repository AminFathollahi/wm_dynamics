#!/usr/bin/env python3
"""Does a nonlinear one-step transition model beat this project's linear DMD
operator on the same working-memory latent trajectories?

DFINE (Dynamical Flexible Inference for Nonlinear Estimation; Sani, Abbaspourazad,
Wong, Shanechi 2024) motivates nonlinear latent dynamics for neural population
data. The earlier claim that its GitHub-only official code was unreachable was
retested and corrected on 2026-08-01: ShanechiLab/torchDFINE is reachable from
this environment. This retained script isolates DFINE's
core motivating claim -- nonlinear vs. linear one-step latent dynamics -- with
a minimal, honestly-labeled comparison: a small feedforward network trained to
predict x_{t+1} from x_t, evaluated against src.dynamics.ensemble_dmd's linear
operator A on IDENTICAL train/test trial splits and the SAME PCA latent space
(no separate nonlinear encoder, so any difference is attributable to the
transition function, not to a different embedding). This is a scoped stand-in
for DFINE's architecture, not a reimplementation of it.

Runs in .venv_dpad (Keras/TF already installed there for run_dpad_dynamics_
dissociation.py; avoids adding a second isolated env for one more small model).

Run:
    ./.venv_dpad/bin/python scripts/run_dfine_nonlinear_dynamics_check.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

from dynamics import _dmd_from_pairs
from statistics import paired_sign_flip_test, stable_seed
from spike_pipeline import load_precomputed_session_geometry
from provenance import _json_safe

MIN_TRIALS = 20
HIDDEN_UNITS_MULT = 4
EPOCHS = 300
PATIENCE = 15


def _pairs(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return Z[:, :-1, :].reshape(-1, Z.shape[-1]), Z[:, 1:, :].reshape(-1, Z.shape[-1])


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def process_session(key: str, data: dict, rng: np.random.Generator) -> dict | None:
    Z = np.asarray(data["Z"], dtype=np.float64)
    n_trials, T, d = Z.shape
    if n_trials < MIN_TRIALS:
        return None
    mu, sd = Z.mean(axis=(0, 1), keepdims=True), Z.std(axis=(0, 1), keepdims=True) + 1e-8
    Z = (Z - mu) / sd

    idx_train, idx_test = train_test_split(
        np.arange(n_trials), test_size=0.3, random_state=int(rng.integers(1 << 31))
    )
    X1_train, X2_train = _pairs(Z[idx_train])
    X1_test, X2_test = _pairs(Z[idx_test])

    A, _ = _dmd_from_pairs(X1_train.T, X2_train.T, r=d)
    linear_pred_test = X1_test @ A.T
    r2_linear = _r2(X2_test, linear_pred_test)

    tf.keras.backend.clear_session()
    tf.random.set_seed(int(rng.integers(1 << 31)))
    model = keras.Sequential([
        keras.layers.Input(shape=(d,)),
        keras.layers.Dense(HIDDEN_UNITS_MULT * d, activation="tanh"),
        keras.layers.Dense(d),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X1_train, X2_train, epochs=EPOCHS, batch_size=64, verbose=0,
              validation_split=0.15,
              callbacks=[keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    nonlinear_pred_test = model.predict(X1_test, verbose=0)
    r2_nonlinear = _r2(X2_test, nonlinear_pred_test)

    return {"n_trials": int(n_trials), "n_pc": int(d), "r2_linear_dmd": r2_linear,
            "r2_nonlinear_mlp": r2_nonlinear, "nonlinear_minus_linear": r2_nonlinear - r2_linear}


def main():
    sessions = load_precomputed_session_geometry(RESULTS, "dandi000574_units_geometry_*.npz")
    print(f"Loaded {len(sessions)} precomputed DANDI 000574 sessions")

    per_session = {}
    for key, data in sessions.items():
        rng = np.random.default_rng(stable_seed(key))
        res = process_session(key, data, rng)
        if res is None:
            print(f"  {key}: SKIP (n_trials below floor)")
            continue
        per_session[key] = res
        print(f"  {key}: n={res['n_trials']} R2 linear={res['r2_linear_dmd']:.3f} "
              f"nonlinear={res['r2_nonlinear_mlp']:.3f} diff={res['nonlinear_minus_linear']:+.3f}", flush=True)

    keys = list(per_session.keys())
    r2_lin = np.array([per_session[k]["r2_linear_dmd"] for k in keys])
    r2_non = np.array([per_session[k]["r2_nonlinear_mlp"] for k in keys])
    comparison = paired_sign_flip_test(r2_lin, r2_non, alternative="less")  # H0: linear >= nonlinear

    print(f"\nAcross {len(keys)} sessions: mean R2 linear={r2_lin.mean():.4f}, "
          f"mean R2 nonlinear={r2_non.mean():.4f}, mean diff={comparison['mean_diff']:.4f}, "
          f"p={comparison['p_value']:.4f} (H0: nonlinear does not exceed linear)")

    summary = {
        "per_session": per_session, "n_sessions": len(keys),
        "mean_r2_linear_dmd": float(r2_lin.mean()), "mean_r2_nonlinear_mlp": float(r2_non.mean()),
        "nonlinear_vs_linear": {"mean_diff": comparison["mean_diff"], "p_value": comparison["p_value"]},
        "note": ("Scoped historical stand-in for DFINE's motivating claim (see module "
                "docstring); the official GitHub repository is reachable as rechecked "
                "on 2026-08-01, so network access is not a current limitation."),
    }
    with open(RESULTS / "dfine_nonlinear_dynamics_check.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)
    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["dfine_nonlinear_dynamics_check"] = summary
    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/dfine_nonlinear_dynamics_check.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
