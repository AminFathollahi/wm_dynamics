"""Subprocess worker: fits lindermanlab/ssm's recurrent switching linear dynamical
system (Laplace-EM) on one train/test split, Poisson or Gaussian emissions.

Run with the interpreter of the ssm environment (WM_DYNAMICS_SSM_PYTHON). Input npz: train_X, test_X (trials, bins,
units), k (int64), seed (int64), optional causal (bool), optional emission (str,
"poisson" (default, raw counts) or "gaussian" (continuous input)). Output npz:
latent_train, latent_test, k_used (causal mode also inference_retried_bins,
inference_carried_bins), or reason (failure).
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import sys
import traceback

import numpy as np

K_STATES = 2
NUM_ITERS = 100
TEST_ITERS = 50
MIN_TRAIN_TRIALS = 3
CAUSAL_ATTEMPTS = 3
RETRY_SEED_STRIDE = 1000003


def _fail(out_path: str, reason: str) -> None:
    np.savez(out_path, reason=str(reason))


def _exception_detail(exc: Exception) -> str:
    """Full exception type, message and traceback tail, so a failure reason is never empty
    (an AssertionError with no message would otherwise report as "AssertionError: ")."""
    tail = "\n".join(traceback.format_exc().strip().splitlines()[-8:])
    return f"{type(exc).__name__}: {exc}\n{tail}"


def _causal_latents(model, trials: list[np.ndarray], n_bins: int, num_iters: int, base_seed: int,
                     bins=None) -> np.ndarray:
    """Per-trial (n_bins, D) filtering latents: approximate_posterior on the
    trial truncated to bins 1..t for every t, keeping only that pass's final
    timestep -- so the latent at bin t depends on bins 1..t of the trial alone.
    bins (0-indexed positions, default every bin) restricts the truncation
    lengths actually computed to the ones scoring will subsample; other
    positions stay zero and are never read downstream.

    Each trial is fit in its own call, reseeded identically per t (not per
    trial), because the structured-meanfield posterior's continuous-state
    initialization is a random draw off the shared numpy global state
    (ssm.variational._initialize_continuous_state_params): fitting several
    trials together in one call would give equal-length identical-prefix
    trials different initializations purely from their order among the
    trials passed to that call, which breaks the causal identity this mode
    exists to guarantee.

    Laplace-EM can diverge on a short prefix (non-finite expected log joint); such a
    prefix is retried with other seeds, then carried forward. Returns (latents, counts)."""
    n_trials = len(trials)
    d = model.D
    out = np.zeros((n_trials, n_bins, d), dtype=np.float64)
    positions = range(n_bins) if bins is None else bins
    counts = {"retried": 0, "carried": 0}
    previous = None
    for bin_idx in positions:
        t = bin_idx + 1
        for i, trial in enumerate(trials):
            latent = None
            for attempt in range(CAUSAL_ATTEMPTS):
                np.random.seed((base_seed + t + attempt * RETRY_SEED_STRIDE) & 0xFFFFFFFF)
                try:
                    _, q = model.approximate_posterior(
                        [trial[:t]], method="laplace_em", variational_posterior="structured_meanfield",
                        num_iters=num_iters, alpha=0.0,
                    )
                except (AssertionError, FloatingPointError, np.linalg.LinAlgError):
                    continue
                candidate = q.mean_continuous_states[0][-1]
                if np.all(np.isfinite(candidate)):
                    latent = candidate
                    break
            if latent is None:
                # Carry the latent from the previous computed prefix (shorter, so still causal).
                latent = out[i, previous] if previous is not None else np.zeros(d)
                counts["carried"] += 1
            elif attempt > 0:
                counts["retried"] += 1
            out[i, bin_idx, :] = latent
        previous = bin_idx
    return out, counts


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    train_X = data["train_X"]
    test_X = data["test_X"]
    k = int(data["k"])
    seed = int(data["seed"])
    causal = bool(data["causal"]) if "causal" in data.files else False
    causal_bins = data["bins"].tolist() if "bins" in data.files else None
    emission = str(data["emission"]) if "emission" in data.files else "poisson"

    try:
        import ssm
    except Exception as exc:
        _fail(out_path, f"import failed -- {type(exc).__name__}: {exc}")
        return

    n_train, n_bins, n_units = train_X.shape
    n_test = test_X.shape[0]
    if n_train < MIN_TRAIN_TRIALS:
        _fail(out_path, f"n_train_trials={n_train} < {MIN_TRAIN_TRIALS}")
        return
    if n_units < 2:
        _fail(out_path, f"n_units={n_units} too small for any latent dimension")
        return
    d_use = int(np.clip(k, 1, n_units - 1))

    np.random.seed(seed)
    if emission == "poisson":
        train_obs = [np.rint(train_X[i]).astype(int) for i in range(n_train)]
        test_obs = [np.rint(test_X[i]).astype(int) for i in range(n_test)]
        emissions_name, emission_kwargs = "poisson_orthog", dict(link="softplus")
    elif emission == "gaussian":
        train_obs = [train_X[i].astype(float) for i in range(n_train)]
        test_obs = [test_X[i].astype(float) for i in range(n_test)]
        emissions_name, emission_kwargs = "gaussian_orthog", None
    else:
        _fail(out_path, f"unknown emission: {emission!r}")
        return

    try:
        model = ssm.SLDS(
            n_units, K_STATES, d_use, transitions="recurrent", dynamics="gaussian",
            emissions=emissions_name, emission_kwargs=emission_kwargs, single_subspace=True,
        )
        model.initialize(train_obs)
        _, q_train_full = model.fit(
            train_obs, method="laplace_em", variational_posterior="structured_meanfield",
            initialize=False, num_iters=NUM_ITERS, alpha=0.0,
        )
    except Exception as exc:
        _fail(out_path, f"training raised -- {_exception_detail(exc)}")
        return

    inference_counts = {}
    try:
        if causal:
            latent_train, counts_train = _causal_latents(model, train_obs, n_bins, TEST_ITERS, seed, causal_bins)
            latent_test, counts_test = _causal_latents(model, test_obs, n_bins, TEST_ITERS, seed, causal_bins)
            inference_counts = {f"inference_{name}_bins": np.int64(counts_train[name] + counts_test[name])
                                for name in ("retried", "carried")}
        else:
            latent_train = np.stack([q_train_full.mean_continuous_states[i] for i in range(n_train)])
            _, q_test = model.approximate_posterior(
                test_obs, method="laplace_em", variational_posterior="structured_meanfield",
                num_iters=TEST_ITERS, alpha=0.0,
            )
            latent_test = np.stack([q_test.mean_continuous_states[i] for i in range(n_test)])
    except Exception as exc:
        _fail(out_path, f"inference raised -- {_exception_detail(exc)}")
        return

    if not (np.all(np.isfinite(latent_train)) and np.all(np.isfinite(latent_test))):
        _fail(out_path, "latents contained non-finite values")
        return
    np.savez(out_path, latent_train=latent_train, latent_test=latent_test, k_used=np.int64(d_use),
             **inference_counts)


if __name__ == "__main__":
    main()
