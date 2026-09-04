"""Feasibility probe for the genuine `cebra` package at the session sizes this
project's recordings actually have.

This script does not run any analysis and does not write anything outside of
its own stdout. It only:

1. Confirms the exact constructor call for a purely self-supervised,
   time-contrastive embedding (no outcome/behaviour/condition label of any
   kind enters the model), and empirically shows that call differs from a
   label-conditioned one.
2. Confirms the model can embed data it was not fitted on, and that a
   read-out mapping the embedding back to the original feature space can be
   fitted after training, for both held-out trials and held-out units.
3. Sweeps session size down to find where training starts to fail, and
   records the literal error text.
4. Times CPU vs GPU at a representative session size.
5. Demonstrates the trial-boundary handling needed so that concatenating
   trials within a session never creates a false temporal adjacency at a
   trial seam.

Everything is synthetic (Poisson spike counts generated from a smooth
low-dimensional latent), fixed-seed, and self-contained.
"""

import os

# Keep this probe's own compute footprint to a single thread so it never
# competes with other CPU work running on the machine.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import sys
import time

import numpy as np
import torch
import cebra
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SEED = 20260822


# ---------------------------------------------------------------------------
# Synthetic data: smooth low-dimensional latent -> Poisson spike counts.
# ---------------------------------------------------------------------------

def make_session(n_trials, n_bins, n_features, n_latent=3, rng=None):
    """Generate (n_trials, n_bins, n_features) Poisson counts from a smooth
    per-trial latent trajectory, plus the ground-truth latent for reference.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    t = np.linspace(0.0, 2.0 * np.pi, n_bins)
    freqs = rng.uniform(0.5, 2.0, size=(n_trials, n_latent))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_trials, n_latent))
    # (n_trials, n_bins, n_latent), smooth within a trial by construction.
    latent = np.sin(freqs[:, None, :] * t[None, :, None] + phases[:, None, :])

    loading = rng.normal(scale=0.6, size=(n_latent, n_features))
    baseline = rng.normal(loc=1.5, scale=0.2, size=(n_features,))
    rate = np.logaddexp(0.0, latent @ loading + baseline)  # softplus, > 0
    counts = rng.poisson(rate).astype("float32")
    return counts, latent.astype("float32")


def flatten(X):
    """(n_trials, n_bins, n_features) -> (n_trials * n_bins, n_features)."""
    n_trials, n_bins, n_features = X.shape
    return X.reshape(n_trials * n_bins, n_features)


# ---------------------------------------------------------------------------
# 1. Time-only self-supervised call vs. a label-conditioned one.
# ---------------------------------------------------------------------------

def check_time_only_mode(X_flat, y_continuous):
    torch.manual_seed(SEED)
    time_only = cebra.CEBRA(
        model_architecture="offset1-model",
        conditional=None,
        batch_size=None,
        output_dimension=3,
        max_iterations=800,
        time_offsets=1,
        device="cuda_if_available",
        verbose=False,
    )
    time_only.fit(X_flat)  # no y: falls back to the pure time-contrastive sampler
    assert time_only.label_types_ == [], (
        f"expected no labels recorded for the time-only call, got {time_only.label_types_}")
    embedding_time_only = time_only.transform(X_flat)

    torch.manual_seed(SEED)
    label_conditioned = cebra.CEBRA(
        model_architecture="offset1-model",
        conditional=None,
        batch_size=None,
        output_dimension=3,
        max_iterations=800,
        time_offsets=1,
        device="cuda_if_available",
        verbose=False,
    )
    label_conditioned.fit(X_flat, y_continuous)  # passing y switches the sampler
    assert len(label_conditioned.label_types_) == 1, (
        f"expected one label recorded, got {label_conditioned.label_types_}")
    embedding_label_conditioned = label_conditioned.transform(X_flat)

    # Compare the pairwise-distance structure of the two embeddings: a
    # label-conditioned fit organizes the space around the label and should
    # not reproduce the same geometry as the label-free fit.
    def pdist_flat(E):
        d = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
        iu = np.triu_indices(len(E), k=1)
        return d[iu]

    corr = np.corrcoef(pdist_flat(embedding_time_only),
                        pdist_flat(embedding_label_conditioned))[0, 1]
    return time_only, embedding_time_only, corr


# ---------------------------------------------------------------------------
# 2. Transform on held-out data + a read-out fitted after training.
# ---------------------------------------------------------------------------

def r2_vs_shuffle(embedding_train, target_train, embedding_test, target_test, rng):
    readout = Ridge(alpha=1.0).fit(embedding_train, target_train)
    pred = readout.predict(embedding_test)
    real_r2 = r2_score(target_test, pred, multioutput="uniform_average")

    shuffled_target_train = target_train[rng.permutation(len(target_train))]
    null_readout = Ridge(alpha=1.0).fit(embedding_train, shuffled_target_train)
    null_pred = null_readout.predict(embedding_test)
    null_r2 = r2_score(target_test, null_pred, multioutput="uniform_average")
    return real_r2, null_r2


def check_holdout_trials(X, rng):
    n_trials = X.shape[0]
    perm = rng.permutation(n_trials)
    n_train = int(0.7 * n_trials)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train = flatten(X[train_idx])
    X_test = flatten(X[test_idx])

    torch.manual_seed(SEED)
    model = cebra.CEBRA(model_architecture="offset1-model", conditional=None,
                        batch_size=None, output_dimension=6, max_iterations=1200,
                        time_offsets=1, device="cuda_if_available", verbose=False)
    model.fit(X_train)

    embedding_train = model.transform(X_train)
    embedding_test = model.transform(X_test)  # held out of fit entirely
    return r2_vs_shuffle(embedding_train, X_train, embedding_test, X_test, rng)


def check_holdout_units(X, rng):
    n_trials, n_bins, n_features = X.shape
    n_held_in_units = int(0.8 * n_features)
    unit_perm = rng.permutation(n_features)
    held_in_units, held_out_units = unit_perm[:n_held_in_units], unit_perm[n_held_in_units:]

    trial_perm = rng.permutation(n_trials)
    n_train = int(0.7 * n_trials)
    train_idx, test_idx = trial_perm[:n_train], trial_perm[n_train:]

    X_train_in = flatten(X[train_idx][:, :, held_in_units])
    X_test_in = flatten(X[test_idx][:, :, held_in_units])
    X_train_out = flatten(X[train_idx][:, :, held_out_units])
    X_test_out = flatten(X[test_idx][:, :, held_out_units])

    torch.manual_seed(SEED)
    model = cebra.CEBRA(model_architecture="offset1-model", conditional=None,
                        batch_size=None, output_dimension=6, max_iterations=1200,
                        time_offsets=1, device="cuda_if_available", verbose=False)
    model.fit(X_train_in)  # trained on held-in units only

    embedding_train = model.transform(X_train_in)
    embedding_test = model.transform(X_test_in)
    return r2_vs_shuffle(embedding_train, X_train_out, embedding_test, X_test_out, rng)


# ---------------------------------------------------------------------------
# 3. Minimum session size.
# ---------------------------------------------------------------------------

def try_fit(n_trials, n_bins, n_features, rng):
    X, _ = make_session(n_trials, n_bins, n_features, rng=rng)
    X_flat = flatten(X)
    try:
        model = cebra.CEBRA(model_architecture="offset1-model", conditional=None,
                            batch_size=None, output_dimension=3, max_iterations=5,
                            time_offsets=1, device="cpu", verbose=False)
        model.fit(X_flat)
        return True, "OK"
    except Exception as exc:  # noqa: BLE001 - we want the literal message
        return False, f"{type(exc).__name__}: {exc}"


def sweep_minimum_size():
    rng = np.random.default_rng(SEED)
    sizes = [
        (150, 15, 35),  # typical human
        (60, 10, 20),   # smallest human
        (30, 10, 20),
        (10, 10, 20),
        (5, 10, 20),
        (2, 10, 20),
        (1, 10, 20),
        (1, 5, 20),
        (1, 3, 20),
        (1, 2, 20),
        (1, 1, 20),
    ]
    results = []
    for n_trials, n_bins, n_features in sizes:
        ok, message = try_fit(n_trials, n_bins, n_features, rng)
        results.append((n_trials, n_bins, n_features, ok, message))
    return results


# ---------------------------------------------------------------------------
# 4. CPU vs GPU timing at the typical human session size.
# ---------------------------------------------------------------------------

def time_device(device, X_flat):
    torch.manual_seed(SEED)
    model = cebra.CEBRA(model_architecture="offset1-model", conditional=None,
                        batch_size=None, output_dimension=8, max_iterations=1000,
                        time_offsets=1, device=device, verbose=False)
    start = time.perf_counter()
    model.fit(X_flat)
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# 5. Trial-boundary handling: no false adjacency at trial seams.
# ---------------------------------------------------------------------------

def trial_gap_layout(n_trials, n_bins, n_features, gap, rng):
    """Concatenate trials with `gap` filler rows between them, so that a
    reference sample drawn from the tail of one trial can never be paired
    (at the configured time_offset) with the head of the next trial.

    Returns the padded 2D array, a boolean mask of which rows are real data,
    and the row index where each trial starts.
    """
    X, _ = make_session(n_trials, n_bins, n_features, rng=rng)
    filler = X.reshape(-1, n_features).mean(axis=0, keepdims=True)

    rows = []
    is_real = []
    trial_starts = []
    for trial in X:
        trial_starts.append(sum(len(r) for r in rows))
        rows.append(trial)
        is_real.append(np.ones(n_bins, dtype=bool))
        rows.append(np.repeat(filler, gap, axis=0))
        is_real.append(np.zeros(gap, dtype=bool))
    padded = np.concatenate(rows, axis=0).astype("float32")
    is_real = np.concatenate(is_real, axis=0)
    return padded, is_real, np.array(trial_starts)


def assert_no_cross_trial_adjacency(is_real, trial_starts, n_bins, gap, time_offset):
    """Prove, by index arithmetic, that no real row's time_offset-positive
    partner lands in a different trial's real data.
    """
    trial_of_row = np.full(len(is_real), -1, dtype=int)
    for trial_id, start in enumerate(trial_starts):
        trial_of_row[start:start + n_bins] = trial_id

    real_rows = np.flatnonzero(is_real)
    for row in real_rows:
        partner = row + time_offset
        if partner >= len(is_real):
            continue
        if is_real[partner] and trial_of_row[partner] != trial_of_row[row]:
            raise AssertionError(
                f"row {row} (trial {trial_of_row[row]}) has a real positive "
                f"partner at row {partner} in trial {trial_of_row[partner]}")
    assert gap >= time_offset, "gap must be at least the sampler's time_offset"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    failures = []

    print(f"cebra {cebra.__version__}, torch {torch.__version__}, "
          f"cuda available: {torch.cuda.is_available()}")

    # --- 1. time-only vs label-conditioned -------------------------------
    rng = np.random.default_rng(SEED)
    X_typical, latent_typical = make_session(150, 15, 35, rng=rng)
    X_typical_flat = flatten(X_typical)
    y_continuous = flatten(latent_typical)[:, 0]  # a stand-in behaviour label

    _, _, geometry_corr = check_time_only_mode(X_typical_flat, y_continuous)
    print(f"[1] time-only vs label-conditioned embedding distance-matrix "
          f"correlation: {geometry_corr:.3f} (expect well below 1.0)")
    if not (geometry_corr < 0.9):
        failures.append("time-only and label-conditioned embeddings were not distinguishable")

    # --- 2. held-out transform + post-hoc read-out ------------------------
    rng = np.random.default_rng(SEED + 1)
    trial_r2, trial_null_r2 = check_holdout_trials(X_typical, rng)
    print(f"[2a] held-out-trial reconstruction R2={trial_r2:.4f} "
          f"vs shuffle baseline R2={trial_null_r2:.4f}")
    if not (trial_r2 > trial_null_r2):
        failures.append("held-out-trial reconstruction did not beat the shuffle baseline")

    rng = np.random.default_rng(SEED + 2)
    unit_r2, unit_null_r2 = check_holdout_units(X_typical, rng)
    print(f"[2b] held-out-unit reconstruction R2={unit_r2:.4f} "
          f"vs shuffle baseline R2={unit_null_r2:.4f}")
    if not (unit_r2 > unit_null_r2):
        failures.append("held-out-unit reconstruction did not beat the shuffle baseline")

    # --- 3. minimum session size ------------------------------------------
    print("[3] session-size sweep (n_trials, n_bins, n_features) -> result")
    sweep = sweep_minimum_size()
    for n_trials, n_bins, n_features, ok, message in sweep:
        print(f"    ({n_trials:>4}, {n_bins:>3}, {n_features:>3}) "
              f"n={n_trials * n_bins:>5} -> {'OK' if ok else message}")
    last_ok = [s for s in sweep if s[3]]
    first_fail = [s for s in sweep if not s[3]]
    if not last_ok:
        failures.append("every swept session size failed to fit")
    if first_fail:
        smallest_ok = last_ok[-1]
        print(f"[3] smallest tested session that fit successfully: "
              f"{smallest_ok[0]}x{smallest_ok[1]}x{smallest_ok[2]} "
              f"(n={smallest_ok[0] * smallest_ok[1]})")
        print(f"[3] first failure: {first_fail[0][0]}x{first_fail[0][1]}x{first_fail[0][2]} "
              f"-> {first_fail[0][4]}")
    else:
        print("[3] no failure found down to the smallest size tested")

    # --- 4. CPU vs GPU timing ----------------------------------------------
    cpu_time = time_device("cpu", X_typical_flat)
    print(f"[4] CPU fit time (typical human size, dim=8, 1000 iters): {cpu_time:.2f} s")
    if torch.cuda.is_available():
        gpu_time = time_device("cuda", X_typical_flat)
        print(f"[4] GPU fit time (typical human size, dim=8, 1000 iters): {gpu_time:.2f} s")
    else:
        print("[4] GPU not available, skipping GPU timing")

    # --- 5. trial-boundary handling -----------------------------------------
    rng = np.random.default_rng(SEED + 3)
    time_offset = 1
    gap = time_offset
    padded, is_real, trial_starts = trial_gap_layout(
        n_trials=20, n_bins=15, n_features=20, gap=gap, rng=rng)
    try:
        assert_no_cross_trial_adjacency(is_real, trial_starts, n_bins=15,
                                        gap=gap, time_offset=time_offset)
        print(f"[5] trial-gap layout (gap={gap} filler rows between trials) verified: "
              f"no real row's time_offset={time_offset} partner crosses a trial boundary")
    except AssertionError as exc:
        failures.append(f"trial-boundary layout is unsafe: {exc}")

    # naive concatenation (gap=0) is expected to be unsafe, confirming the risk
    naive_padded, naive_is_real, naive_starts = trial_gap_layout(
        n_trials=20, n_bins=15, n_features=20, gap=0, rng=rng)
    naive_unsafe = False
    try:
        assert_no_cross_trial_adjacency(naive_is_real, naive_starts, n_bins=15,
                                        gap=0, time_offset=time_offset)
    except AssertionError:
        naive_unsafe = True
    print(f"[5] naive concatenation (no gap) creates false adjacency at seams: {naive_unsafe}")
    if not naive_unsafe:
        failures.append("naive concatenation was expected to be unsafe but was not")

    if failures:
        print(f"FAIL: {'; '.join(failures)}")
        return 1
    print("PASS: time-only mode confirmed, held-out transform + post-hoc read-out "
          "beat shuffle for both trials and units, minimum session size mapped, "
          "device timing measured, trial-boundary handling verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
