"""Subprocess worker: fits snel-repo/neural-data-transformers on one train/test
split, upstream config/Runner/get_rates path, unmodified.

Run with the interpreter of the ndt environment (WM_DYNAMICS_NDT_PYTHON); the upstream
source checkout is the configured ndt_repo (WM_DYNAMICS_NDT_REPO). Input npz: train_X, test_X (trials,
bins, units) raw counts, k (int64), seed (int64), optional causal (bool) --
False trains with full context (MODEL.FULL_CONTEXT=True), True trains with
causal attention (MODEL.FULL_CONTEXT=False, CONTEXT_FORWARD=0,
CONTEXT_BACKWARD=-1). Output npz: latent_train, latent_test, k_used, or
reason (failure). Latents are the final transformer-encoder-layer output
(LAYER_USED, index -1) reduced to k dimensions by a PCA projection fit on
training trials.
"""
import functools
import importlib.util
import os
import shutil
import sys
import tempfile

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import h5py
import numpy as np

_config_spec = importlib.util.spec_from_file_location(
    "project_config", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "project_config.py"))
_project_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_project_config)
NDT_REPO = _project_config.executable("ndt_repo")
if NDT_REPO:
    sys.path.insert(0, NDT_REPO)

LAYER_USED = -1  # final transformer-encoder-layer output
MIN_TRAIN_TRIALS = 4


def _fail(out_path: str, reason: str) -> None:
    np.savez(out_path, reason=str(reason))


def _write_h5(path: str, train_X: np.ndarray, test_X: np.ndarray) -> None:
    n_train, n_bins, n_units = train_X.shape
    n_test = test_X.shape[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("train_spikes_heldin", data=train_X.astype(np.float32))
        f.create_dataset("train_spikes_heldout", data=np.zeros((n_train, n_bins, 0), dtype=np.float32))
        f.create_dataset("train_spikes_heldin_forward", data=np.zeros((n_train, 0, n_units), dtype=np.float32))
        f.create_dataset("train_spikes_heldout_forward", data=np.zeros((n_train, 0, 0), dtype=np.float32))
        f.create_dataset("eval_spikes_heldin", data=test_X.astype(np.float32))
        f.create_dataset("eval_spikes_heldout", data=np.zeros((n_test, n_bins, 0), dtype=np.float32))
        f.create_dataset("eval_spikes_heldin_forward", data=np.zeros((n_test, 0, n_units), dtype=np.float32))
        f.create_dataset("eval_spikes_heldout_forward", data=np.zeros((n_test, 0, 0), dtype=np.float32))


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    data = np.load(in_path)
    train_X = data["train_X"]
    test_X = data["test_X"]
    k = int(data["k"])
    seed = int(data["seed"])
    causal = bool(data["causal"]) if "causal" in data.files else False

    try:
        import torch
        torch.load = functools.partial(torch.load, weights_only=False)
        from sklearn.decomposition import PCA
        from src.config.default import get_config
        from src.dataset import DATASET_MODES
        from src.runner import Runner
    except Exception as exc:
        _fail(out_path, f"import failed -- {type(exc).__name__}: {exc}")
        return

    n_train, n_bins, n_units = train_X.shape
    n_test = test_X.shape[0]
    if n_train < MIN_TRAIN_TRIALS or n_units < 1:
        _fail(out_path, f"n_train_trials={n_train}, n_units={n_units} too small")
        return

    tmp_dir = tempfile.mkdtemp(prefix="ndt_bridge_")
    try:
        h5_path = os.path.join(tmp_dir, "data.h5")
        _write_h5(h5_path, train_X, test_X)

        context_opts = (
            ["MODEL.FULL_CONTEXT", False, "MODEL.CONTEXT_FORWARD", 0, "MODEL.CONTEXT_BACKWARD", -1]
            if causal else ["MODEL.FULL_CONTEXT", True]
        )
        opts = [
            "SEED", seed,
            "DATA.IGNORE_FORWARD", True,
            "TRAIN.BATCH_SIZE", max(1, min(8, n_train)),
            "TRAIN.NUM_UPDATES", 5000,
            "TRAIN.LR.WARMUP", 50,
            "TRAIN.LOG_INTERVAL", 500,
            "TRAIN.VAL_INTERVAL", 10,
            "TRAIN.CHECKPOINT_INTERVAL", 10000,
            "USE_TENSORBOARD", False,
            "DATA.DATAPATH", tmp_dir + os.sep,
            "DATA.TRAIN_FILENAME", "data.h5",
            "DATA.VAL_FILENAME", "data.h5",
            "VARIANT", "bridge",
            "CHECKPOINT_DIR", os.path.join(tmp_dir, "ckpts"),
            "TENSORBOARD_DIR", os.path.join(tmp_dir, "tb"),
            "LOG_DIR", os.path.join(tmp_dir, "logs"),
            *context_opts,
        ]
        config = get_config(config_paths=[os.path.join(NDT_REPO, "configs", "area2_bump.yaml")], opts=opts)
        torch.manual_seed(seed)
        np.random.seed(seed)

        runner = Runner(config=config)
        runner.train()
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "bridge.lve.pth")
        if not os.path.exists(ckpt_path):
            _fail(out_path, "no validation-improving checkpoint was ever written")
            return

        eval_runner = Runner(config=config)
        _, train_layers = eval_runner.get_rates(checkpoint_path=ckpt_path, mode=DATASET_MODES.train)
        _, test_layers = eval_runner.get_rates(checkpoint_path=ckpt_path, mode=DATASET_MODES.val)
        last_train = train_layers[LAYER_USED].detach().cpu().numpy()
        last_test = test_layers[LAYER_USED].detach().cpu().numpy()
    except Exception as exc:
        _fail(out_path, f"{type(exc).__name__}: {exc}")
        return
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    d_feat = last_train.shape[-1]
    n_rows = last_train.shape[0] * last_train.shape[1]
    k_use = int(np.clip(k, 1, min(d_feat, n_rows - 1)))
    try:
        pca = PCA(n_components=k_use, random_state=seed)
        flat_train = last_train.reshape(-1, d_feat)
        pca.fit(flat_train)
        latent_train = pca.transform(flat_train).reshape(n_train, n_bins, k_use)
        latent_test = pca.transform(last_test.reshape(-1, d_feat)).reshape(n_test, n_bins, k_use)
    except Exception as exc:
        _fail(out_path, f"pca projection raised -- {type(exc).__name__}: {exc}")
        return

    if not (np.all(np.isfinite(latent_train)) and np.all(np.isfinite(latent_test))):
        _fail(out_path, "latents contained non-finite values")
        return
    np.savez(out_path, latent_train=latent_train, latent_test=latent_test, k_used=np.int64(k_use),
              layer_used=np.array(f"transformer_encoder_layer[{LAYER_USED}]"))


if __name__ == "__main__":
    main()
