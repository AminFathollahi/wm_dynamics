"""Smoke check for the isolated sequential-autoencoder environment.

Runs INSIDE that environment, not the analysis environment, because the two pin incompatible
versions of the tensor library. Trains the smallest workable model for a handful of epochs on
synthetic Poisson counts drawn from a smooth low-dimensional latent, and asserts that the fitted
model beats a constant-rate baseline on held-out trials. Importing the package is not enough: the
question is whether it trains here, on data the size of ours.

Prints one line beginning PASS or FAIL and exits non-zero on failure.
"""

import sys
import time

import numpy as np

SEED = 0
N_TRIALS = 60
N_BINS = 20
N_UNITS = 24
N_LATENT = 3
N_EPOCHS = 400
TEST_FRACTION = 0.25


def synthetic_counts(rng):
    """Poisson counts whose rates follow a smooth low-dimensional latent."""
    time = np.linspace(0.0, 2.0 * np.pi, N_BINS)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(N_TRIALS, N_LATENT))
    latent = np.sin(time[None, :, None] + phase[:, None, :])
    loading = rng.normal(0.0, 0.6, size=(N_LATENT, N_UNITS))
    log_rate = np.einsum("tbl,lu->tbu", latent, loading) + np.log(4.0)
    return rng.poisson(np.exp(log_rate)).astype("float32")


def poisson_negative_log_likelihood(rate, counts):
    rate = np.clip(rate, 1e-6, None)
    from scipy.special import gammaln

    return float(np.mean(rate - counts * np.log(rate) + gammaln(counts + 1.0)))


def main():
    rng = np.random.default_rng(SEED)
    counts = synthetic_counts(rng)
    n_test = int(round(N_TRIALS * TEST_FRACTION))
    order = rng.permutation(N_TRIALS)
    test, train = counts[order[:n_test]], counts[order[n_test:]]

    try:
        import torch
        from torch import nn
        from lfads_torch.model import LFADS
        from lfads_torch.modules import augmentations
        from lfads_torch.modules.recons import Poisson
        from lfads_torch.modules.priors import AutoregressiveMultivariateNormal, MultivariateNormal
        from lfads_torch.tuples import SessionBatch
    except Exception as exc:  # the point of the check is to report this, not to raise it
        print("FAIL: import failed -- %s: %s" % (type(exc).__name__, exc))
        return 1

    torch.manual_seed(SEED)
    try:
        model = LFADS(
            encod_data_dim=N_UNITS,
            encod_seq_len=N_BINS,
            recon_seq_len=N_BINS,
            ext_input_dim=0,
            ic_enc_seq_len=0,
            ic_enc_dim=16,
            ci_enc_dim=0,
            ci_lag=1,
            con_dim=0,
            co_dim=0,
            ic_dim=N_LATENT,
            gen_dim=32,
            fac_dim=N_LATENT,
            dropout_rate=0.0,
            reconstruction=nn.ModuleList([Poisson()]),
            readin=nn.ModuleList([nn.Identity()]),
            readout=nn.ModuleList([nn.Linear(N_LATENT, N_UNITS)]),
            lr_scheduler=False,
            variational=True,
            co_prior=AutoregressiveMultivariateNormal(tau=10.0, nvar=0.1, shape=1),
            ic_prior=MultivariateNormal(mean=0.0, variance=0.1, shape=N_LATENT),
            ic_post_var_min=1e-4,
            cell_clip=5.0,
            loss_scale=1.0,
            recon_reduce_mean=True,
            lr_init=4e-3,
            lr_stop=1e-5,
            lr_decay=0.95,
            lr_patience=6,
            lr_adam_beta1=0.9,
            lr_adam_beta2=0.999,
            lr_adam_epsilon=1e-8,
            weight_decay=0.0,
            l2_start_epoch=0,
            l2_increase_epoch=0,
            l2_ic_enc_scale=0.0,
            l2_ci_enc_scale=0.0,
            l2_gen_scale=0.0,
            l2_con_scale=0.0,
            kl_start_epoch=0,
            kl_increase_epoch=0,
            kl_ic_scale=0.0,
            kl_co_scale=0.0,
            train_aug_stack=augmentations.AugmentationStack([], []),
            infer_aug_stack=augmentations.AugmentationStack([], []),
        )
    except Exception as exc:
        print("FAIL: model construction failed -- %s: %s" % (type(exc).__name__, exc))
        return 1

    def as_batch(array):
        """Wrap counts as the single-session batch the model expects; unused fields stay empty."""
        data = torch.from_numpy(array).to(device)
        empty = torch.zeros(data.shape[0], data.shape[1], 0, device=device)
        return {0: SessionBatch(data, data, empty, empty, empty)}

    # Some builds of this package's pinned tensor library allocate on an unsupported GPU without
    # complaint and then hang on the first kernel, so the device is reported, never assumed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=4e-3)
    train_tensor = as_batch(train)
    test_tensor = as_batch(test)

    t0 = time.time()
    try:
        model.train()
        for _ in range(N_EPOCHS):
            optimiser.zero_grad()
            output = model(train_tensor)
            rate = output[0].output_params
            loss = torch.nn.functional.poisson_nll_loss(
                rate, train_tensor[0].encod_data, log_input=False, full=True, reduction="mean"
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            output = model(test_tensor)
            fitted = output[0].output_params.cpu().numpy()
        elapsed = time.time() - t0
    except Exception as exc:
        print("FAIL: training failed -- %s: %s" % (type(exc).__name__, exc))
        return 1

    if fitted.shape != test.shape:
        print("FAIL: inferred rates have shape %s, expected %s" % (fitted.shape, test.shape))
        return 1
    if not np.all(np.isfinite(fitted)):
        print("FAIL: inferred rates contain non-finite values")
        return 1

    baseline = np.broadcast_to(train.mean(axis=(0, 1)), test.shape)
    fitted_nll = poisson_negative_log_likelihood(fitted, test)
    baseline_nll = poisson_negative_log_likelihood(baseline, test)
    if not fitted_nll < baseline_nll:
        print(
            "FAIL: held-out likelihood no better than a constant rate -- fitted %.4f, baseline %.4f"
            % (fitted_nll, baseline_nll)
        )
        return 1

    print(
        "PASS: %s, %.1f s, trained %d epochs on %d held-in trials, held-out negative log likelihood "
        "%.4f against a constant-rate baseline of %.4f"
        % (device.type, elapsed, N_EPOCHS, train.shape[0], fitted_nll, baseline_nll)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
