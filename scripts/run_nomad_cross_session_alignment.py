#!/usr/bin/env python3
"""NoMAD-style cross-session dynamics alignment (Karpowicz et al. 2025, Nat
Commun, "Stabilizing brain-computer interfaces through alignment of latent
dynamics", DOI 10.1038/s41467-025-59652-y) on DANDI 000574 (Boran) subjects
with multiple recorded sessions, testing whether aligning each session's
dynamics to a shared reference lets a SINGLE fixed behavioral decoder
(trained once, on the reference session only) transfer to later sessions
without retraining -- vs. applying that same fixed decoder with no alignment
at all.

Official NoMAD repo (github.com/snel-repo/nomad) depends on lfads_tf2, pinned
to Python 3.7.7 + TensorFlow 2 + CUDA 10.0/cuDNN 7.6 -- incompatible with this
machine's GPU (RTX 5070 Ti, Blackwell architecture, needs CUDA 12.8+) and this
project's Python 3.11 env. The suggested modern PyTorch LFADS backbone
(arsedler9/lfads-torch) is GitHub-only with no PyPI release. Both repositories
were confirmed reachable on 2026-08-01; network access is not the blocker.
This script instead implements NoMAD's alignment MECHANISM
directly (the actual contribution being tested): fit a small GRU generator
+ readout on a REFERENCE session, freeze the generator, then for every other
session train only a new read-in (per-session, since these sessions are not
even unit-matched) and readout that minimizes reconstruction loss plus a
KL-divergence term matching that session's generator-state distribution
(Gaussian moment-matched) to the reference session's -- exactly NoMAD's
Section "alignment" procedure, on a from-scratch GRU generator (LFADS'S own
backbone, an RNN sequential encoder/generator) rather than a packaged one.

Observations here are the already-PCA-reduced continuous trial latents
(results/dandi000574_units_geometry_*.npz), not raw spike counts, so
reconstruction uses Gaussian (MSE) loss rather than LFADS's Poisson
likelihood -- a deliberate simplification given this project's existing
preprocessed substrate, not a limitation of the alignment mechanism itself.

Run:
    ./.venv_dpad/bin/python scripts/run_nomad_cross_session_alignment.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"

from statistics import paired_sign_flip_test, stable_seed
from spike_pipeline import load_precomputed_session_geometry
from provenance import _json_safe

MIN_TRIALS = 25
MIN_SESSIONS_PER_SUBJECT = 3
NX_GENERATOR = 6
REF_EPOCHS = 400
ALIGN_EPOCHS = 300
KL_WEIGHT = 0.1
LR = 5e-3


def _load_subject_sessions() -> dict[str, list[str]]:
    sessions = load_precomputed_session_geometry(RESULTS, "dandi000574_units_geometry_*.npz")
    by_subject = defaultdict(list)
    for key, data in sessions.items():
        if data["Z"].shape[0] < MIN_TRIALS:
            continue
        subj = re.match(r"(sub-\d+)_ses-\d+", key).group(1)
        by_subject[subj].append(key)
    return {s: sorted(keys) for s, keys in by_subject.items() if len(keys) >= MIN_SESSIONS_PER_SUBJECT}, sessions


def _build_generator(nx: int, seed: int) -> keras.Model:
    init = keras.initializers.GlorotUniform(seed=seed)
    return keras.layers.GRU(nx, return_sequences=True, return_state=False,
                            kernel_initializer=init, recurrent_initializer=init, name="generator")


def _zscore(Z: np.ndarray) -> np.ndarray:
    mu, sd = Z.mean(axis=(0, 1), keepdims=True), Z.std(axis=(0, 1), keepdims=True) + 1e-8
    return (Z - mu) / sd


def _fit_reference(Z_ref: np.ndarray, nx: int, seed: int):
    """Train encoder + generator + readout on the reference session jointly.
    Returns (generator_layer, encoder, readout, generator_states)."""
    n_trials, T, d = Z_ref.shape
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    Z_in = keras.Input(shape=(T, d))
    z0 = keras.layers.GRU(nx, name="encoder")(Z_in)  # read-in: session -> initial condition
    generator = _build_generator(nx, seed)
    dummy_in = keras.layers.RepeatVector(T)(keras.layers.Lambda(lambda x: tf.zeros_like(x[:, :1]))(z0))
    states = generator(dummy_in, initial_state=z0)
    readout = keras.layers.Dense(d, name="readout")
    recon = readout(states)
    model = keras.Model(Z_in, recon)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    model.fit(Z_ref, Z_ref, epochs=REF_EPOCHS, batch_size=n_trials, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)])
    state_model = keras.Model(Z_in, states)
    ref_states = state_model.predict(Z_ref, verbose=0)
    return generator, model, state_model, ref_states


def _gaussian_kl(mu_p, cov_p, mu_q, cov_q, eps=1e-4):
    """KL(N(mu_p, cov_p) || N(mu_q, cov_q)), TF tensors, cov regularized by eps*I."""
    k = tf.cast(tf.shape(mu_p)[0], tf.float32)
    cov_q_r = cov_q + eps * tf.eye(tf.shape(cov_q)[0])
    cov_p_r = cov_p + eps * tf.eye(tf.shape(cov_p)[0])
    cov_q_inv = tf.linalg.inv(cov_q_r)
    diff = mu_q - mu_p
    term_trace = tf.linalg.trace(cov_q_inv @ cov_p_r)
    term_quad = tf.reduce_sum(diff * tf.linalg.matvec(cov_q_inv, diff))
    _, logdet_q = tf.linalg.slogdet(cov_q_r)
    _, logdet_p = tf.linalg.slogdet(cov_p_r)
    return 0.5 * (term_trace + term_quad - k + logdet_q - logdet_p)


def _fit_aligned_session(Z_sess: np.ndarray, generator, ref_mu, ref_cov, nx: int, seed: int):
    """Freeze `generator`; train a fresh encoder + readout for this session,
    minimizing reconstruction MSE + KL(this session's generator states || reference)."""
    n_trials, T, d = Z_sess.shape
    tf.random.set_seed(seed)
    init = keras.initializers.GlorotUniform(seed=seed)
    encoder = keras.layers.GRU(nx, kernel_initializer=init, recurrent_initializer=init)
    readout = keras.layers.Dense(d, kernel_initializer=init)
    generator.trainable = False

    Z_tf = tf.constant(Z_sess, dtype=tf.float32)
    ref_mu_tf = tf.constant(ref_mu, dtype=tf.float32)
    ref_cov_tf = tf.constant(ref_cov, dtype=tf.float32)
    optimizer = keras.optimizers.Adam(LR)
    best_loss, patience, bad_epochs = np.inf, 20, 0

    for epoch in range(ALIGN_EPOCHS):
        with tf.GradientTape() as tape:
            z0 = encoder(Z_tf)
            dummy_in = tf.zeros((n_trials, T, 1))
            states = generator(dummy_in, initial_state=z0)
            recon = readout(states)
            mse = tf.reduce_mean(tf.square(recon - Z_tf))
            flat_states = tf.reshape(states, (-1, nx))
            mu = tf.reduce_mean(flat_states, axis=0)
            centered = flat_states - mu
            cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(flat_states)[0] - 1, tf.float32)
            kl = _gaussian_kl(mu, cov, ref_mu_tf, ref_cov_tf)
            loss = mse + KL_WEIGHT * kl
        trainable_vars = encoder.trainable_variables + readout.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        loss_val = float(loss.numpy())
        if loss_val < best_loss - 1e-5:
            best_loss, bad_epochs = loss_val, 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                break

    z0 = encoder(Z_tf)
    dummy_in = tf.zeros((n_trials, T, 1))
    states = generator(dummy_in, initial_state=z0).numpy()
    generator.trainable = True
    return states, float(mse.numpy()), float(kl.numpy())


def process_subject(subj: str, session_keys: list[str], all_sessions: dict, rng: np.random.Generator) -> dict:
    ref_key = session_keys[0]
    Z_ref = _zscore(np.asarray(all_sessions[ref_key]["Z"], dtype=np.float32))
    label_ref = np.asarray(all_sessions[ref_key]["set_size"]) == 8
    seed = int(rng.integers(1 << 31))
    generator, ref_model, state_model, ref_states = _fit_reference(Z_ref, NX_GENERATOR, seed)

    flat_ref = ref_states.reshape(-1, NX_GENERATOR)
    ref_mu = flat_ref.mean(axis=0)
    ref_cov = np.cov(flat_ref, rowvar=False) + 1e-4 * np.eye(NX_GENERATOR)

    ref_features = ref_states[:, -5:, :].mean(axis=1)
    decoder = LogisticRegression(max_iter=1000).fit(ref_features, label_ref)

    # Naive-transfer decoder: trained on the reference session's own RAW (unaligned)
    # PCA features, in the SAME space every session's raw Z already lives in --
    # this is the "pretend sessions are directly comparable, no dynamics-based
    # recalibration" baseline. Only valid against another session if that
    # session happens to share the reference's own PCA dimensionality.
    ref_raw_features = Z_ref[:, -5:, :].mean(axis=1)
    naive_decoder = LogisticRegression(max_iter=1000).fit(ref_raw_features, label_ref)
    ref_raw_dim = ref_raw_features.shape[1]

    per_session = {}
    for key in session_keys[1:]:
        Z_sess = _zscore(np.asarray(all_sessions[key]["Z"], dtype=np.float32))
        label_sess = np.asarray(all_sessions[key]["set_size"]) == 8
        if min(label_sess.sum(), (~label_sess).sum()) < 5:
            continue
        aligned_states, mse, kl = _fit_aligned_session(
            Z_sess, generator, ref_mu, ref_cov, NX_GENERATOR, int(rng.integers(1 << 31)))
        aligned_features = aligned_states[:, -5:, :].mean(axis=1)
        auc_aligned = roc_auc_score(label_sess, decoder.predict_proba(aligned_features)[:, 1])

        naive_features = Z_sess[:, -5:, :].mean(axis=1)  # session's own raw PCA scores, no alignment
        auc_naive = roc_auc_score(label_sess, naive_decoder.predict_proba(naive_features)[:, 1]) \
            if naive_features.shape[1] == ref_raw_dim else float("nan")

        per_session[key] = {
            "n_trials": int(Z_sess.shape[0]), "reconstruction_mse": mse, "kl_to_reference": kl,
            "auc_aligned_transfer": float(auc_aligned), "auc_naive_transfer": float(auc_naive),
        }
        print(f"    {key}: n={Z_sess.shape[0]} recon_mse={mse:.3f} kl={kl:.3f} "
              f"AUC aligned={auc_aligned:.3f} naive={auc_naive:.3f}", flush=True)

    return {"reference_session": ref_key, "n_sessions": len(session_keys), "per_session": per_session}


def main():
    by_subject, all_sessions = _load_subject_sessions()
    print(f"Subjects with >= {MIN_SESSIONS_PER_SUBJECT} qualifying sessions: {list(by_subject.keys())}")

    per_subject = {}
    for subj, keys in by_subject.items():
        print(f"  {subj}: {len(keys)} sessions, reference={keys[0]}")
        rng = np.random.default_rng(stable_seed(subj))
        per_subject[subj] = process_subject(subj, keys, all_sessions, rng)

    aligned_aucs, naive_aucs = [], []
    for subj_res in per_subject.values():
        for sess_res in subj_res["per_session"].values():
            if np.isfinite(sess_res["auc_naive_transfer"]):
                aligned_aucs.append(sess_res["auc_aligned_transfer"])
                naive_aucs.append(sess_res["auc_naive_transfer"])

    if len(aligned_aucs) >= 4:
        comparison = paired_sign_flip_test(np.array(naive_aucs), np.array(aligned_aucs), alternative="less")
        print(f"\nAcross {len(aligned_aucs)} non-reference sessions: mean AUC aligned="
              f"{np.mean(aligned_aucs):.3f}, mean AUC naive={np.mean(naive_aucs):.3f}, "
              f"mean diff={comparison['mean_diff']:.4f}, p={comparison['p_value']:.4f} "
              f"(H0: naive transfer is not worse than aligned)")
        aggregate = {"n_pairs": len(aligned_aucs), "mean_auc_aligned": float(np.mean(aligned_aucs)),
                    "mean_auc_naive": float(np.mean(naive_aucs)),
                    "aligned_vs_naive": {"mean_diff": comparison["mean_diff"], "p_value": comparison["p_value"]}}
    else:
        print("\nToo few non-reference session pairs for a population-level comparison.")
        aggregate = {"n_pairs": len(aligned_aucs)}

    summary = {"per_subject": per_subject, "aggregate": aggregate}
    with open(RESULTS / "nomad_cross_session_alignment.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)
    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["nomad_cross_session_alignment"] = summary
    with open(stats_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/nomad_cross_session_alignment.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
