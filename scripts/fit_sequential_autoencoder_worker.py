import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import sys

import numpy as np

MAX_EPOCHS = 800
EARLY_STOP_PATIENCE = 50
KL_INCREASE_EPOCHS = 80
KL_IC_SCALE = 1e-3
VALIDATION_FRACTION = 0.2
MIN_VALIDATION_TRIALS = 2
MIN_FIT_TRIALS = 3


def _fail(out_path: str, reason: str) -> None:
    np.savez(out_path, reason=str(reason))


def _split_indices(n_trials: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(MIN_VALIDATION_TRIALS, int(round(n_trials * VALIDATION_FRACTION)))
    order = np.random.default_rng(seed).permutation(n_trials)
    return order[n_val:], order[:n_val]


def _kl_weight(epoch: int) -> float:
    return KL_IC_SCALE * min((epoch + 1) / (KL_INCREASE_EPOCHS + 1), 1.0)


def _causal_fill(X: np.ndarray, t: int, fill: np.ndarray) -> np.ndarray:
    """X with bins t..end (0-indexed) replaced by fill -- bins before t are untouched, so
    a forward pass on this array's factor at bin index t-1 depends only on X's own bins
    0..t-1 (the model has no other route to bins at or after t)."""
    filled = X.copy()
    filled[:, t:, :] = fill[None, t:, :]
    return filled


def _causal_factors(model, X: np.ndarray, fill: np.ndarray, as_session_input, bins=None) -> np.ndarray:
    """bins: 0-indexed bin positions to actually compute (default: every bin). Scoring
    only ever reads a subsampled subset of bins, so the T prefix passes this mode needs are
    only ever needed at those positions -- the rest of the output stays zero and is never
    read downstream."""
    import torch

    n_bins = X.shape[1]
    positions = range(n_bins) if bins is None else bins
    columns = {}
    with torch.no_grad():
        for bin_idx in positions:
            t = bin_idx + 1
            factors = model(as_session_input(_causal_fill(X, t, fill)))[0].factors.cpu().numpy()
            columns[bin_idx] = factors[:, t - 1, :]
    fac_dim = next(iter(columns.values())).shape[-1]
    out = np.zeros((X.shape[0], n_bins, fac_dim), dtype=np.float32)
    for bin_idx, column in columns.items():
        out[:, bin_idx, :] = column
    return out


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
        import torch
        from torch import nn
        from lfads_torch.model import LFADS
        from lfads_torch.modules import augmentations
        from lfads_torch.modules.priors import MultivariateNormal, Null
        from lfads_torch.modules.recons import Gaussian, Poisson
        from lfads_torch.tuples import SessionBatch
    except Exception as exc:
        _fail(out_path, f"import failed -- {type(exc).__name__}: {exc}")
        return

    if emission not in ("poisson", "gaussian"):
        _fail(out_path, f"unknown emission: {emission!r}")
        return

    n_train, n_bins, n_units = train_X.shape
    if n_train < MIN_VALIDATION_TRIALS + MIN_FIT_TRIALS:
        _fail(out_path, f"n_train_trials={n_train} too small for an internal validation split")
        return
    fit_idx, val_idx = _split_indices(n_train, seed)
    if len(fit_idx) < MIN_FIT_TRIALS:
        _fail(out_path, f"n_fit_trials={len(fit_idx)} < {MIN_FIT_TRIALS} after carving out the validation split")
        return

    torch.manual_seed(seed)
    device = torch.device("cpu")
    fac_dim = int(np.clip(k, 1, n_units - 1))

    if emission == "poisson":
        reconstruction_module, readout_dim = Poisson(), n_units
    else:
        reconstruction_module, readout_dim = Gaussian(), n_units * 2

    def build_model():
        return LFADS(
            encod_data_dim=n_units, encod_seq_len=n_bins, recon_seq_len=n_bins,
            ext_input_dim=0, ic_enc_seq_len=0, ic_enc_dim=16,
            ci_enc_dim=0, ci_lag=1, con_dim=0, co_dim=0,
            ic_dim=fac_dim, gen_dim=32, fac_dim=fac_dim, dropout_rate=0.0,
            reconstruction=nn.ModuleList([reconstruction_module]),
            readin=nn.ModuleList([nn.Identity()]),
            readout=nn.ModuleList([nn.Linear(fac_dim, readout_dim)]),
            lr_scheduler=False, variational=True,
            co_prior=Null(),
            ic_prior=MultivariateNormal(mean=0.0, variance=0.1, shape=fac_dim),
            ic_post_var_min=1e-4, cell_clip=5.0, loss_scale=1.0, recon_reduce_mean=True,
            lr_init=4e-3, lr_stop=1e-5, lr_decay=0.95, lr_patience=6,
            lr_adam_beta1=0.9, lr_adam_beta2=0.999, lr_adam_epsilon=1e-8,
            weight_decay=0.0, l2_start_epoch=0, l2_increase_epoch=0,
            l2_ic_enc_scale=0.0, l2_ci_enc_scale=0.0, l2_gen_scale=0.0, l2_con_scale=0.0,
            kl_start_epoch=0, kl_increase_epoch=KL_INCREASE_EPOCHS,
            kl_ic_scale=KL_IC_SCALE, kl_co_scale=0.0,
            train_aug_stack=augmentations.AugmentationStack([], []),
            infer_aug_stack=augmentations.AugmentationStack([], []),
        ).to(device)

    def as_session_input(array: np.ndarray):
        data_t = torch.from_numpy(array.astype(np.float32)).to(device)
        empty = torch.zeros(data_t.shape[0], data_t.shape[1], 0, device=device)
        return {0: SessionBatch(data_t, data_t, empty, empty, empty)}

    def reconstruction_loss(model, chunk, sample_posteriors):
        output = model(chunk, sample_posteriors=sample_posteriors, output_means=False)
        loss = model.recon[0].compute_loss(chunk[0].recon_data, output[0].output_params)
        return loss.mean(), output

    fit_input = as_session_input(train_X[fit_idx])
    val_input = as_session_input(train_X[val_idx])

    model = build_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=4e-3)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    try:
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimiser.zero_grad()
            recon, output = reconstruction_loss(model, fit_input, True)
            ic_kl = model.ic_prior(output[0].ic_mean, output[0].ic_std)
            loss = recon + _kl_weight(epoch) * ic_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            model.eval()
            with torch.no_grad():
                val_recon, val_output = reconstruction_loss(model, val_input, False)
                val_kl = model.ic_prior(val_output[0].ic_mean, val_output[0].ic_std)
                val_loss = float((val_recon + KL_IC_SCALE * val_kl).item())
            if epoch < KL_INCREASE_EPOCHS:
                continue
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    break
    except Exception as exc:
        _fail(out_path, f"training raised -- {type(exc).__name__}: {exc}")
        return

    if best_state is None:
        _fail(out_path, "no validation-improving epoch was ever reached")
        return
    model.load_state_dict(best_state)
    model.eval()

    if causal:
        try:
            # Inference-only causal mode, same trained weights: for every bin t, the
            # initial condition is inferred from bins 1..t of that trial with bins t+1..T
            # replaced by the per-neuron, per-bin mean over training trials (train_X, the
            # full outer training set), then the generated factor at t is kept -- one
            # training, T prefix passes, no parameter update.
            fill = train_X.mean(axis=0)  # (n_bins, n_units)
            causal_train = _causal_factors(model, train_X, fill, as_session_input, causal_bins)
            causal_test = _causal_factors(model, test_X, fill, as_session_input, causal_bins)
        except Exception as exc:
            _fail(out_path, f"causal factor extraction raised -- {type(exc).__name__}: {exc}")
            return
        if not (np.all(np.isfinite(causal_train)) and np.all(np.isfinite(causal_test))):
            _fail(out_path, "causal factors contained non-finite values")
            return
        np.savez(out_path, latent_train=causal_train, latent_test=causal_test, k_used=np.int64(fac_dim),
                  causal_fill=np.array("per_neuron_per_bin_training_mean"))
        return

    try:
        with torch.no_grad():
            train_factors = model(as_session_input(train_X))[0].factors.cpu().numpy()
            test_factors = model(as_session_input(test_X))[0].factors.cpu().numpy()
    except Exception as exc:
        _fail(out_path, f"factor extraction raised -- {type(exc).__name__}: {exc}")
        return

    if not (np.all(np.isfinite(train_factors)) and np.all(np.isfinite(test_factors))):
        _fail(out_path, "factors contained non-finite values")
        return

    np.savez(out_path, latent_train=train_factors, latent_test=test_factors, k_used=np.int64(fac_dim))


if __name__ == "__main__":
    main()
