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


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    train_X = data["train_X"]
    test_X = data["test_X"]
    k = int(data["k"])
    seed = int(data["seed"])

    try:
        import torch
        from torch import nn
        from lfads_torch.model import LFADS
        from lfads_torch.modules import augmentations
        from lfads_torch.modules.priors import MultivariateNormal, Null
        from lfads_torch.modules.recons import Poisson
        from lfads_torch.tuples import SessionBatch
    except Exception as exc:
        _fail(out_path, f"import failed -- {type(exc).__name__}: {exc}")
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

    def build_model():
        return LFADS(
            encod_data_dim=n_units, encod_seq_len=n_bins, recon_seq_len=n_bins,
            ext_input_dim=0, ic_enc_seq_len=0, ic_enc_dim=16,
            ci_enc_dim=0, ci_lag=1, con_dim=0, co_dim=0,
            ic_dim=fac_dim, gen_dim=32, fac_dim=fac_dim, dropout_rate=0.0,
            reconstruction=nn.ModuleList([Poisson()]),
            readin=nn.ModuleList([nn.Identity()]),
            readout=nn.ModuleList([nn.Linear(fac_dim, n_units)]),
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

    def as_batch(array: np.ndarray):
        data_t = torch.from_numpy(array.astype(np.float32)).to(device)
        empty = torch.zeros(data_t.shape[0], data_t.shape[1], 0, device=device)
        return {0: SessionBatch(data_t, data_t, empty, empty, empty)}

    def reconstruction_loss(model, batch, sample_posteriors):
        output = model(batch, sample_posteriors=sample_posteriors, output_means=False)
        loss = model.recon[0].compute_loss(batch[0].recon_data, output[0].output_params)
        return loss.mean(), output

    fit_batch = as_batch(train_X[fit_idx])
    val_batch = as_batch(train_X[val_idx])

    model = build_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=4e-3)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    try:
        for epoch in range(MAX_EPOCHS):
            model.train()
            optimiser.zero_grad()
            recon, output = reconstruction_loss(model, fit_batch, True)
            ic_kl = model.ic_prior(output[0].ic_mean, output[0].ic_std)
            loss = recon + _kl_weight(epoch) * ic_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            model.eval()
            with torch.no_grad():
                val_recon, val_output = reconstruction_loss(model, val_batch, False)
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

    try:
        with torch.no_grad():
            train_factors = model(as_batch(train_X))[0].factors.cpu().numpy()
            test_factors = model(as_batch(test_X))[0].factors.cpu().numpy()
    except Exception as exc:
        _fail(out_path, f"factor extraction raised -- {type(exc).__name__}: {exc}")
        return

    if not (np.all(np.isfinite(train_factors)) and np.all(np.isfinite(test_factors))):
        _fail(out_path, "factors contained non-finite values")
        return

    np.savez(out_path, latent_train=train_factors, latent_test=test_factors, k_used=np.int64(fac_dim))


if __name__ == "__main__":
    main()
