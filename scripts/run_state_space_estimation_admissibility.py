"""Which latent/manifold estimators can this project's own recordings actually support?

This is a gate, not a scientific result: it never concludes anything about behaviour, geometry or
dynamics, and no sentence anywhere in this module or in the artifact it writes may say that one
candidate estimator performed better than another as a matter of neuroscience. The only comparison
made is whether a given session's data are enough to fit a given candidate at all, judged the same
way the field's own held-out-reconstruction benchmark judges it: hold out a random subset of
features (units or channels), fit the candidate on the held-in features only, and ask whether a
read-out fitted AFTER training can predict the held-out features on trials the candidate never saw
during fitting, against a same-session shuffle baseline. A candidate that cannot clear that bar on a
majority of a corpus's sessions is recorded as inadmissible, with the trial and feature counts that
excluded it, and is never fitted anyway and quietly reported as if it had worked.

Two held-out scores are computed and reported separately per session per candidate, because a method
can reconstruct an unseen feature while failing on an unseen trial, or the reverse:

  - a held-out-FEATURE score: units/channels are split into a held-in set (given to the candidate)
    and a held-out set (never given to it, at fit time or at inference time); the candidate is fit on
    training trials' held-in features, a linear or Poisson read-out is fit afterwards from the
    resulting latent state to the held-out features on those same training trials, and both are then
    scored on test trials the candidate has never seen -- so this number is jointly a feature- and a
    trial-generalisation score, the most demanding one computed here;
  - a held-out-TRIAL score: every feature is given to the candidate, which is fit on training trials
    only; a read-out is fit from the training-trial latent state back to the training trials' own full
    feature set (self-reconstruction), and scored on test trials -- this isolates trial-level
    generalisation alone, holding the feature set fixed, because the project's single-trial claims
    depend specifically on that form of generalisation.

Every candidate is scored through a read-out fitted strictly after the latent estimator has finished
fitting, and strictly on the training trials, because several candidates here recover a latent state
only up to an arbitrary rotation and rescaling; comparing raw latent coordinates across held-in and
held-out feature sets would penalise that indeterminacy rather than a genuine failure to reconstruct.

Two data types are scored with the score appropriate to each: spike counts (trials, units, bins) with
a likelihood-based scoring rule for counts (negative Poisson deviance, so higher is better and a
higher held-out score than the shuffle baseline is a pass); field-potential band power (trials,
channels, bins) with an explained-variance score.

Every candidate is fit inside its training fold only -- it never sees a test trial, at fit time or at
inference time (out-of-sample projection for the nonlinear candidates uses only the fitted geometry
from training trials). The unit of inference for the gate is the session, or, in every human corpus,
the patient (a patient passes a candidate only if a majority of that patient's own sessions do): never
the held-out-feature draw and never the fold. Every session or patient this module excludes from a
gate is recorded with a machine-readable reason and, where the reason is a count, the number that
excluded it.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_state_space_estimation_admissibility.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import (  # noqa: E402
    BORAN_EPOCH_WINDOWS_S, EPOCH_WINDOWS_S, data_root, iter_alm, iter_dandi_000469,
    iter_dandi_000574, iter_dandi_001187, iter_watters,
)
from geometry import parallel_analysis, participation_ratio, spatiotemporal_participation_ratio  # noqa: E402
from preprocessing import bandpass_filter, load_boran_nwb  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from run_latent_model_comparison import anscombe_counts, counts_to_spiketrains, raw_counts_from_entry  # noqa: E402
from statistics import stable_seed  # noqa: E402
from project_config import executable  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import FactorAnalysis  # noqa: E402
from sklearn.linear_model import PoissonRegressor, Ridge  # noqa: E402
from sklearn.metrics import explained_variance_score, mean_poisson_deviance  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT_PATH = ROOT / "results" / "state_space_estimation_admissibility.json"
SHARD_VARIABLE = "WM_DYNAMICS_SESSION_SHARD"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_state_space_estimation_admissibility"

SEED = 20260822
BIN_MS = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_TRIALS = 20
MIN_FEATURES = 6
TEST_TRIAL_FRACTION = 0.30
MIN_TEST_TRIALS = 6
MIN_TRAIN_TRIALS = 10
HELD_OUT_FEATURE_FRACTION = 0.25
MIN_HELD_IN_FEATURES = 3
MIN_HELD_OUT_FEATURES = 2
MAX_OPERATING_RANK = 10
PCA_RANK_SWEEP = (2, 3, 4, 6, 8, 10)
CV_LIKELIHOOD_RANK_CANDIDATES = (1, 2, 3, 4, 6, 8, 10)
GATE_MAJORITY_THRESHOLD = 0.5

# Field-potential feature band: safely under the Nyquist of both signal types this project holds at
# this recording site (depth ~1398 Hz, scalp ~140 Hz), so one band definition applies to both without
# a per-signal special case.
FIELD_BAND_LO_HZ = 1.0
FIELD_BAND_HI_HZ = 40.0
FIELD_MAINTENANCE_WINDOW_S = (-3.0, 0.0)  # relative to probe onset, this task's own convention


def _seed(*parts: str) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


# ── Corpus assembly: uniform (trials, features, bins) tensors ───────────────────────────────────────

def spiking_sessions(root: Path):
    """Yields dict(dataset, patient, session, X, bin_ms) for every pooled-structure delay-epoch
    session across the human single-unit corpora, mouse ALM and the multi-object macaque corpus.
    X is (n_trials, n_units, n_bins) raw, un-transformed spike counts."""
    epoch_windows_by_dataset = {
        "dandi_000469": EPOCH_WINDOWS_S, "dandi_001187": EPOCH_WINDOWS_S, "dandi_000574": BORAN_EPOCH_WINDOWS_S,
    }
    for iterator in (iter_dandi_000469, iter_dandi_001187, iter_dandi_000574):
        for entry in iterator(root):
            if entry["structure"] != "pooled":
                continue
            window = epoch_windows_by_dataset[entry["dataset"]]["delay"]
            from spike_pipeline import build_psth
            rate = build_psth(entry["spike_lists"], entry["epoch_onsets"]["delay"], bin_ms=BIN_MS, smooth_ms=0, window_s=window)
            X = np.rint(rate * (BIN_MS / 1000.0)).astype(int)
            yield {"dataset": entry["dataset"], "patient": entry["patient"], "session": entry["session"], "X": X, "bin_ms": BIN_MS}
    for entry in iter_alm(root, bin_ms=BIN_MS):
        yield {"dataset": "inagaki_alm5", "patient": entry["patient"], "session": entry["session"],
               "X": entry["counts"].astype(int), "bin_ms": BIN_MS}
    for entry in iter_watters(root, bin_ms=BIN_MS):
        if entry.get("status") != "loaded":
            continue
        yield {"dataset": "watters_2026", "patient": entry["animal"], "session": entry["session"],
               "X": entry["counts"].astype(int), "bin_ms": BIN_MS}


def _boran_field_potential_session(nwb_path: Path, signal: str) -> dict | None:
    """One dandi_000574 session's maintenance-window band power for one field-potential signal
    ('ieeg' = depth macro-contacts, 'eeg' = scalp montage), trial-admitted the same way
    src/corpus_sessions.py's own dandi_000574 spike iterator admits trials (artifact-flag exclusion
    plus correct-only), applied here independently since this reads a different signal group from the
    same file and is not a modification of that iterator.
    """
    with h5py.File(str(nwb_path), "r") as handle:
        if "intervals/trials" not in handle:
            return None
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
        correct = trials["correct"][:].astype(bool)
    keep = (~artifact) & correct
    if keep.sum() < MIN_TRIALS:
        return None
    loaded = load_boran_nwb(str(nwb_path), signal=signal, epoch_win=(-3.2, 0.3))
    epochs = loaded["epochs"][keep]  # (N, C, T)
    times = loaded["times"]
    srate = loaded["srate"]
    win_mask = (times >= FIELD_MAINTENANCE_WINDOW_S[0]) & (times < FIELD_MAINTENANCE_WINDOW_S[1])
    win_times = times[win_mask]
    n_bins = int(round((FIELD_MAINTENANCE_WINDOW_S[1] - FIELD_MAINTENANCE_WINDOW_S[0]) * 1000.0 / BIN_MS))
    bin_edges = np.linspace(win_times[0], FIELD_MAINTENANCE_WINDOW_S[1], n_bins + 1)
    n_trials, n_ch, _ = epochs.shape
    power = np.zeros((n_trials, n_ch, n_bins), dtype=float)
    for i in range(n_trials):
        try:
            filtered = bandpass_filter(epochs[i].T, FIELD_BAND_LO_HZ, FIELD_BAND_HI_HZ, srate)  # (T, C)
        except Exception:
            return None
        sq = (filtered ** 2)[win_mask]
        for b in range(n_bins):
            bin_mask = (win_times >= bin_edges[b]) & (win_times < bin_edges[b + 1])
            power[i, :, b] = sq[bin_mask].mean(axis=0) if bin_mask.any() else np.nan
    if not np.isfinite(power).all():
        return None
    patient = nwb_path.parent.name
    return {"dataset": f"dandi_000574_{signal}", "patient": patient, "session": nwb_path.stem,
            "X": power, "bin_ms": BIN_MS}


def field_potential_sessions(root: Path):
    """Yields dict(dataset, patient, session, X, bin_ms) for the two field-potential signal groups
    this project holds (depth macro-contact and scalp), both from dandi_000574. X is
    (n_trials, n_channels, n_bins) band power, nonnegative."""
    directory = root / "000574"
    for subject_dir in sorted(directory.glob("sub-*")):
        for path in sorted(subject_dir.glob("*.nwb")):
            for signal in ("ieeg", "eeg"):
                record = _boran_field_potential_session(path, signal)
                if record is not None:
                    yield record


# ── Splits ────────────────────────────────────────────────────────────────────────────────────────

def trial_split(n_trials: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_trials)
    n_test = max(MIN_TEST_TRIALS, int(round(n_trials * TEST_TRIAL_FRACTION)))
    n_test = min(n_test, n_trials - MIN_TRAIN_TRIALS)
    return perm[n_test:], perm[:n_test]  # train, test


def feature_split(n_features: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_features)
    n_held_out = max(MIN_HELD_OUT_FEATURES, int(round(n_features * HELD_OUT_FEATURE_FRACTION)))
    n_held_out = min(n_held_out, n_features - MIN_HELD_IN_FEATURES)
    return perm[n_held_out:], perm[:n_held_out]  # held_in, held_out


# ── Rank-selection criteria (computed once per session, on training-trial held-in features only) ───

def cv_likelihood_rank(rows: np.ndarray, rng: np.random.Generator) -> tuple[int | None, dict]:
    n_rows, n_features = rows.shape
    n_splits = min(3, max(2, n_rows // 6))
    if n_rows < 2 * n_splits:
        return None, {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    scores = {}
    for k in CV_LIKELIHOOD_RANK_CANDIDATES:
        if k < 1 or k >= n_features:
            continue
        fold_scores = []
        for tr, te in kf.split(rows):
            if len(tr) < k + 2 or len(te) < 2:
                continue
            try:
                fa = FactorAnalysis(n_components=k, random_state=0).fit(rows[tr])
                fold_scores.append(fa.score(rows[te]))
            except Exception:
                continue
        if fold_scores:
            scores[k] = float(np.mean(fold_scores))
    if not scores:
        return None, {}
    best_k = max(scores, key=scores.get)
    return best_k, scores


def rank_selection_criteria(X_held_in_train: np.ndarray, rng: np.random.Generator) -> dict:
    """X_held_in_train: (n_train_trials, n_held_in_features, n_bins) raw values on training trials'
    held-in features only. Returns the rank each of four principled selectors would choose, computed
    independently of which candidate estimator is eventually fit."""
    n_trials, n_features, n_bins = X_held_in_train.shape
    X_ntc = X_held_in_train.transpose(0, 2, 1)  # (n_trials, n_bins, n_features), matches
    # spatiotemporal_participation_ratio's (N, C, T) convention when read as (trials, channels, time)
    pr_result = spatiotemporal_participation_ratio(X_held_in_train, rng=rng)
    flat_rows = X_ntc.reshape(-1, n_features)
    pa_rank = int(parallel_analysis(flat_rows, rng=rng, n_timepoints_per_trial=n_bins))
    cvl_rank, cvl_scores = cv_likelihood_rank(flat_rows, rng)
    pr_insample_rank = int(np.clip(round(pr_result["pr_insample"]), 1, n_features - 1)) if np.isfinite(pr_result["pr_insample"]) else None
    cv_reconstruction_rank = int(np.clip(round(pr_result["pr_cv"]), 1, n_features - 1)) if np.isfinite(pr_result["pr_cv"]) else None
    return {
        "participation_ratio_rank": pr_insample_rank,
        "cross_validated_reconstruction_rank": cv_reconstruction_rank,
        "parallel_analysis_rank": pa_rank if pa_rank > 0 else None,
        "cross_validated_likelihood_rank": cvl_rank,
        "cross_validated_likelihood_scores_by_rank": cvl_scores,
    }


OPERATING_RANK_PRIORITY = (
    "cross_validated_reconstruction_rank", "cross_validated_likelihood_rank",
    "participation_ratio_rank", "parallel_analysis_rank",
)


def operating_rank(criteria: dict, n_features: int) -> int:
    """The single rank every rank-parameterised candidate is fit at for this gate (principal
    components alone is additionally swept at a declared range of ranks -- see PCA_RANK_SWEEP).
    Preference order among the four selectors: cross-validated reconstruction first, since it is the
    one most directly tied to genuine out-of-sample generalisation rather than an in-sample spectrum
    or a null comparison; falls through the remaining three, then a fixed small default only if every
    selector failed to return a rank on this session's data."""
    chosen = next((criteria.get(name) for name in OPERATING_RANK_PRIORITY if criteria.get(name)), None)
    if chosen is None:
        chosen = min(3, n_features - 1)
    return int(np.clip(chosen, 1, min(MAX_OPERATING_RANK, n_features - 1)))


# ── Torch latent models: a GRU sequential variational autoencoder and a plain (no-time-axis) one ───

class _GRUVAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, k: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.to_mu = nn.Linear(hidden, k)
        self.to_logvar = nn.Linear(hidden, k)
        self.decode = nn.Linear(k, in_dim)

    def forward(self, x: torch.Tensor):
        h, _ = self.gru(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        out = self.decode(z)
        return z, mu, logvar, out


class _MLPVAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, k: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.to_mu = nn.Linear(hidden, k)
        self.to_logvar = nn.Linear(hidden, k)
        self.decode = nn.Linear(k, in_dim)

    def forward(self, x: torch.Tensor):
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        out = self.decode(z)
        return z, mu, logvar, out


def _vae_loss(out: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
              is_spiking: bool, beta: float = 0.1) -> torch.Tensor:
    if is_spiking:
        recon = F.poisson_nll_loss(out, target, log_input=True, full=False, reduction="mean")
    else:
        recon = F.mse_loss(out, target, reduction="mean")
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


def _fit_one_vae(model_cls, rows_shape_is_3d: bool, x_train: torch.Tensor, x_val: torch.Tensor | None,
                  in_dim: int, hidden: int, k: int, is_spiking: bool, epochs: int) -> tuple[nn.Module, float]:
    model = model_cls(in_dim, hidden, k).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, mu, logvar, out = model(x_train)
        loss = _vae_loss(out, x_train, mu, logvar, is_spiking)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        if x_val is not None and x_val.numel() > 0:
            _, mu, logvar, out = model(x_val)
            val_loss = float(_vae_loss(out, x_val, mu, logvar, is_spiking).item())
        else:
            _, mu, logvar, out = model(x_train)
            val_loss = float(_vae_loss(out, x_train, mu, logvar, is_spiking).item())
    return model, val_loss


def _tune_and_fit_vae(model_cls, rows_3d: np.ndarray, k_center: int, is_spiking: bool, epochs: int = 120) -> dict:
    """rows_3d: (n_train_trials, n_bins, in_dim) training-fold data only. Internally splits an 80/20
    trial validation slice from these same training trials (never touching test trials) to pick a
    latent dimensionality automatically from a small candidate set around the session's operating
    rank, and a hidden width from a small fixed set -- an automatically tuned form rather than one
    hand-picked pair of hyperparameters."""
    n_trials, n_bins, in_dim = rows_3d.shape
    if n_trials < 6:
        return {"status": "failed_to_train", "reason": f"n_train_trials={n_trials} < 6 needed for an internal validation split"}
    n_val = max(2, int(round(n_trials * 0.2)))
    idx = np.arange(n_trials)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    x_fit = torch.tensor(rows_3d[fit_idx], dtype=torch.float32, device=DEVICE)
    x_val = torch.tensor(rows_3d[val_idx], dtype=torch.float32, device=DEVICE)
    k_candidates = sorted({max(1, k_center - 2), k_center, min(in_dim - 1, k_center + 2)} & set(range(1, in_dim)))
    if not k_candidates:
        return {"status": "failed_to_train", "reason": f"no valid latent dim below in_dim={in_dim}"}
    best = None
    for k in k_candidates:
        for hidden in (32, 64):
            try:
                _, val_loss = _fit_one_vae(model_cls, True, x_fit, x_val, in_dim, hidden, k, is_spiking, epochs)
            except Exception as exc:
                continue
            if best is None or val_loss < best[0]:
                best = (val_loss, k, hidden)
    if best is None:
        return {"status": "failed_to_train", "reason": "every (latent_dim, hidden) configuration raised during tuning"}
    _, k_used, hidden_used = best
    x_all = torch.tensor(rows_3d, dtype=torch.float32, device=DEVICE)
    model, _ = _fit_one_vae(model_cls, True, x_all, None, in_dim, hidden_used, k_used, is_spiking, epochs)
    return {"status": "fitted", "model": model, "k_used": int(k_used), "hidden_used": int(hidden_used)}


def _vae_transform(model: nn.Module, X_3d: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_3d, dtype=torch.float32, device=DEVICE)
        _, mu, _, _ = model(x)
    return mu.cpu().numpy()


# ── Candidate fit functions: uniform (n_trials, n_bins, n_features) in, (n_trials, n_bins', k) latent out ──

def fit_native_full_rank(train_X, test_X, k, rng, is_spiking, bin_ms):
    return {"status": "fitted", "k_used": int(train_X.shape[-1]),
            "latent_train": train_X.astype(float), "latent_test": test_X.astype(float)}


def _flatten(X):
    n, b, f = X.shape
    return X.reshape(n * b, f), n, b


def fit_principal_components(train_X, test_X, k, rng, is_spiking, bin_ms):
    flat_train, n_tr, n_b = _flatten(anscombe_counts(train_X) if is_spiking else train_X)
    flat_test, n_te, _ = _flatten(anscombe_counts(test_X) if is_spiking else test_X)
    n_features = flat_train.shape[1]
    k_use = int(np.clip(k, 1, min(n_features - 1, flat_train.shape[0] - 1)))
    if k_use < 1:
        return {"status": "failed_to_train", "reason": f"n_features={n_features}, n_train_rows={flat_train.shape[0]} too small for any rank"}
    mean = flat_train.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat_train - mean, full_matrices=False)
    v = vt[:k_use].T
    latent_train = (flat_train - mean) @ v
    latent_test = (flat_test - mean) @ v
    return {"status": "fitted", "k_used": k_use,
            "latent_train": latent_train.reshape(n_tr, n_b, k_use), "latent_test": latent_test.reshape(n_te, n_b, k_use)}


def fit_factor_analysis(train_X, test_X, k, rng, is_spiking, bin_ms):
    flat_train, n_tr, n_b = _flatten(anscombe_counts(train_X) if is_spiking else train_X)
    flat_test, n_te, _ = _flatten(anscombe_counts(test_X) if is_spiking else test_X)
    n_features = flat_train.shape[1]
    k_use = int(np.clip(k, 1, n_features - 1))
    try:
        fa = FactorAnalysis(n_components=k_use, random_state=int(rng.integers(0, 2**31 - 1))).fit(flat_train)
        latent_train = fa.transform(flat_train)
        latent_test = fa.transform(flat_test)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"factor_analysis raised: {exc}"}
    return {"status": "fitted", "k_used": k_use,
            "latent_train": latent_train.reshape(n_tr, n_b, k_use), "latent_test": latent_test.reshape(n_te, n_b, k_use)}


def fit_gaussian_process_factor_analysis(train_X, test_X, k, rng, is_spiking, bin_ms):
    if not is_spiking:
        return {"status": "not_applicable_to_data_type"}
    from elephant.gpfa import GPFA
    import quantities as pq
    n_features = train_X.shape[-1]
    k_use = int(np.clip(k, 1, n_features - 1))
    try:
        # counts_to_spiketrains expects (n_trials, n_units, n_bins); train_X/test_X here are
        # (n_trials, n_bins, n_features) per this module's uniform candidate interface.
        train_st = counts_to_spiketrains(train_X.transpose(0, 2, 1).astype(int), bin_ms)
        test_st = counts_to_spiketrains(test_X.transpose(0, 2, 1).astype(int), bin_ms)
        gpfa = GPFA(bin_size=bin_ms * pq.ms, x_dim=k_use, em_max_iters=30, verbose=False)
        gpfa.fit(train_st)
        latent_train = np.stack(gpfa.transform(train_st), axis=0).transpose(0, 2, 1)
        latent_test = np.stack(gpfa.transform(test_st), axis=0).transpose(0, 2, 1)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"gpfa raised: {exc}"}
    return {"status": "fitted", "k_used": k_use, "latent_train": latent_train, "latent_test": latent_test}


LFADS_WORKER_PYTHON = executable("lfads_python")
LFADS_WORKER_SCRIPT = ROOT / "scripts" / "fit_sequential_autoencoder_worker.py"
LFADS_SUBPROCESS_TIMEOUT_S = 900  # generous relative to the small (tens-of-trials, tens-of-bins)
# sessions this gate ever fits; a run that has not converged by then is treated as a failure rather
# than left to hang the shard indefinitely.


def fit_sequential_autoencoder(train_X, test_X, k, rng, is_spiking, bin_ms):
    """Fits the genuine `lfads_torch` sequential autoencoder via a subprocess bridge into the
    isolated `lfads_torch_py310` environment, which pins a tensor library incompatible with this
    project's own analysis environment and so cannot be imported here directly. Driven CPU-only
    (CUDA_VISIBLE_DEVICES="") because this machine's GPU allocates without error under that
    environment's pinned build and then hangs forever on the first kernel. A timeout or a nonzero
    exit from the worker is reported as an ordinary failed_to_train, never raised, so one session's
    subprocess trouble cannot crash the sweep."""
    if not is_spiking:
        return {"status": "not_applicable_to_data_type"}
    if not LFADS_WORKER_PYTHON:
        return {
            "status": "failed_to_train",
            "reason": "configure executables.lfads_python or WM_DYNAMICS_LFADS_PYTHON",
        }
    seed = int(rng.integers(0, 2**31 - 1))
    with tempfile.TemporaryDirectory(prefix="lfads_bridge_") as tmp_dir:
        in_path = Path(tmp_dir) / "input.npz"
        out_path = Path(tmp_dir) / "output.npz"
        np.savez(in_path, train_X=train_X.astype(np.float32), test_X=test_X.astype(np.float32),
                  k=np.int64(k), seed=np.int64(seed))
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            proc = subprocess.run(
                [LFADS_WORKER_PYTHON, str(LFADS_WORKER_SCRIPT), str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=LFADS_SUBPROCESS_TIMEOUT_S, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed_to_train",
                    "reason": f"lfads_torch worker timed out after {LFADS_SUBPROCESS_TIMEOUT_S}s"}
        if proc.returncode != 0 or not out_path.exists():
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            return {"status": "failed_to_train",
                    "reason": f"lfads_torch worker exited {proc.returncode}: {stderr_tail}"}
        result = np.load(out_path, allow_pickle=True)
        if "reason" in result.files:
            return {"status": "failed_to_train", "reason": str(np.asarray(result["reason"]).item())}
        return {"status": "fitted", "k_used": int(result["k_used"]),
                "latent_train": result["latent_train"], "latent_test": result["latent_test"]}


CEBRA_TIME_OFFSET = 1
CEBRA_MAX_ITERATIONS = 1200  # fixed across every session, never tuned against a session's own data:
# this is the exact iteration count the feasibility probe (verify_contrastive_embedding_environment.py)
# used for its held-out-reconstruction checks -- the check that most resembles this gate's own
# held-out-feature and held-out-trial scoring -- where it reliably beat a shuffle baseline at this
# project's typical human session size.


def _is_out_of_memory(exc: Exception) -> bool:
    """Whether a raised exception is the graphics device running out of memory, including the case
    where it surfaces indirectly through a compiled-module traceback rather than as its own error."""
    return "out of memory" in str(exc).lower()


def _trial_gap_pad(flat: np.ndarray, n_trials: int, n_bins: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Insert `gap` filler rows (the column mean of `flat`'s own real rows) between every trial of a
    trial-major flattened (n_trials * n_bins, n_features) array, so that a purely time-contrastive fit
    over the concatenated session never treats one trial's last bin and the next trial's first bin as
    temporal neighbours. Returns the padded array and a boolean mask marking which of its rows are
    real (non-filler) data, so the filler can be dropped again after the model has been applied."""
    n_features = flat.shape[1]
    trials = flat.reshape(n_trials, n_bins, n_features)
    filler = flat.mean(axis=0, keepdims=True)
    rows, is_real = [], []
    for trial in trials:
        rows.append(trial)
        is_real.append(np.ones(n_bins, dtype=bool))
        rows.append(np.repeat(filler, gap, axis=0))
        is_real.append(np.zeros(gap, dtype=bool))
    padded = np.concatenate(rows, axis=0).astype(np.float32)
    return padded, np.concatenate(is_real, axis=0)


def fit_time_contrastive_embedding(train_X, test_X, k, rng, is_spiking, bin_ms):
    """A purely self-supervised time-contrastive embedding fit with the genuine `cebra` package: the
    encoder is trained only to place nearby timepoints of the same trial close together, from no
    outcome, behaviour or condition label of any kind -- verified by asserting the fitted model
    recorded no label types at all."""
    import cebra
    flat_train, n_tr, n_b = _flatten(anscombe_counts(train_X) if is_spiking else train_X)
    flat_test, n_te, _ = _flatten(anscombe_counts(test_X) if is_spiking else test_X)
    n_features = flat_train.shape[1]
    k_use = int(np.clip(k, 1, n_features - 1))
    if n_b < 2:
        return {"status": "failed_to_train", "reason": f"n_bins={n_b} < 2, no temporal positive pair exists"}
    mean = flat_train.mean(axis=0, keepdims=True)
    std = flat_train.std(axis=0, keepdims=True) + 1e-6
    z_train = (flat_train - mean) / std
    z_test = (flat_test - mean) / std

    padded_train, real_train = _trial_gap_pad(z_train, n_tr, n_b, CEBRA_TIME_OFFSET)
    padded_test, real_test = _trial_gap_pad(z_test, n_te, n_b, CEBRA_TIME_OFFSET)
    if padded_train.shape[0] < 3:
        return {"status": "failed_to_train",
                "reason": f"n_padded_train_rows={padded_train.shape[0]} < 3, cebra needs at least 3 timepoints to fit"}

    seed = int(rng.integers(0, 2**31 - 1))

    def fit_on(device):
        torch.manual_seed(seed)
        fitted = cebra.CEBRA(
            model_architecture="offset1-model", conditional=None, batch_size=None,
            output_dimension=k_use, max_iterations=CEBRA_MAX_ITERATIONS,
            time_offsets=CEBRA_TIME_OFFSET, device=device, verbose=False,
        )
        fitted.fit(padded_train)
        return fitted

    try:
        model = fit_on("cuda_if_available")
    except Exception as exc:
        # Exhausting the graphics device says nothing about whether this session can support the
        # estimator -- with several sessions fitting concurrently it reports on the machine, not the
        # data -- so the session is refitted on the processor at the same seed and settings rather
        # than being recorded as a session the estimator could not be trained on.
        if not _is_out_of_memory(exc):
            return {"status": "failed_to_train", "reason": f"cebra fit raised: {exc}"}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            model = fit_on("cpu")
        except Exception as cpu_exc:
            return {"status": "failed_to_train", "reason": f"cebra fit raised: {cpu_exc}"}
    if model.label_types_ != []:
        return {"status": "failed_to_train",
                "reason": f"cebra model recorded label_types_={model.label_types_}, expected [] for a purely self-supervised fit"}
    try:
        emb_train = model.transform(padded_train)
        emb_test = model.transform(padded_test)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"cebra transform raised: {exc}"}
    latent_train = emb_train[real_train].reshape(n_tr, n_b, k_use)
    latent_test = emb_test[real_test].reshape(n_te, n_b, k_use)
    return {"status": "fitted", "k_used": k_use, "latent_train": latent_train, "latent_test": latent_test}


def fit_temporal_diffusion_embedding(train_X, test_X, k, rng, is_spiking, bin_ms):
    """A diffusion-manifold embedding whose graph construction is smoothed along the row order before
    the embedding is built, so that rows nearby in time end up nearby in the embedding even when they
    would not be nearest neighbours by instantaneous population state alone -- the temporal variant of
    the plain diffusion embedding used for the trajectory and manifold claims in this project. Rows are
    trial-major (every trial's own bins are contiguous), so the smoothing window is capped at the
    session's own bin count to keep it from blending across a trial boundary more than unavoidably at
    the single row where one trial's last bin sits next to the next trial's first."""
    import tphate
    flat_train, n_tr, n_b = _flatten(anscombe_counts(train_X) if is_spiking else train_X)
    flat_test, n_te, _ = _flatten(anscombe_counts(test_X) if is_spiking else test_X)
    n_features = flat_train.shape[1]
    k_use = int(np.clip(k, 1, min(n_features - 1, 10)))
    try:
        knn = max(2, min(5, flat_train.shape[0] // 4))
        smooth_window = max(1, min(3, n_b - 1))
        op = tphate.TPHATE(n_components=k_use, knn=knn, smooth_window=smooth_window, verbose=0,
                            n_jobs=1, random_state=int(rng.integers(0, 2**31 - 1)))
        latent_train = op.fit_transform(flat_train)
        latent_test = op.transform(flat_test)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"temporal_diffusion_embedding raised: {exc}"}
    return {"status": "fitted", "k_used": k_use,
            "latent_train": latent_train.reshape(n_tr, n_b, k_use), "latent_test": latent_test.reshape(n_te, n_b, k_use)}


def fit_trial_level_variational_autoencoder(train_X, test_X, k, rng, is_spiking, bin_ms):
    # train_X, test_X: (n_trials, n_bins, n_features) -- average over the BIN axis (1), not features,
    # to get one feature vector per trial.
    pooled_train = (anscombe_counts(train_X) if is_spiking else train_X).mean(axis=1)  # (n_trials, n_features)
    pooled_test = (anscombe_counts(test_X) if is_spiking else test_X).mean(axis=1)
    rows_3d = pooled_train[:, None, :]  # (n_trials, 1, n_features), no time axis
    result = _tune_and_fit_vae(_MLPVAE, rows_3d, k, is_spiking=False)  # pooled input is continuous regardless of data type
    if result["status"] != "fitted":
        return result
    model = result["model"]
    latent_train = _vae_transform(model, rows_3d)
    latent_test = _vae_transform(model, pooled_test[:, None, :])
    return {"status": "fitted", "k_used": result["k_used"], "latent_train": latent_train, "latent_test": latent_test}


CANDIDATES = {
    "native_full_rank": fit_native_full_rank,
    "principal_components": fit_principal_components,
    "factor_analysis": fit_factor_analysis,
    "gaussian_process_factor_analysis": fit_gaussian_process_factor_analysis,
    "sequential_autoencoder": fit_sequential_autoencoder,
    "time_contrastive_embedding": fit_time_contrastive_embedding,
    "temporal_diffusion_embedding": fit_temporal_diffusion_embedding,
    "trial_level_variational_autoencoder": fit_trial_level_variational_autoencoder,
}
SPIKING_ONLY_CANDIDATES = {"gaussian_process_factor_analysis", "sequential_autoencoder"}


# ── Decoder-after-training and scoring ───────────────────────────────────────────────────────────────

def _decode(latent_train_flat, target_train_flat, latent_test_flat, is_spiking):
    if is_spiking:
        preds = np.zeros((latent_test_flat.shape[0], target_train_flat.shape[1]))
        for j in range(target_train_flat.shape[1]):
            y = target_train_flat[:, j]
            try:
                reg = PoissonRegressor(alpha=1.0, max_iter=300).fit(latent_train_flat, y)
                preds[:, j] = reg.predict(latent_test_flat)
            except Exception:
                preds[:, j] = max(y.mean(), 1e-6)
        return preds
    reg = Ridge(alpha=1.0).fit(latent_train_flat, target_train_flat)
    return reg.predict(latent_test_flat)


def _score(y_true_flat, y_pred_flat, is_spiking):
    if is_spiking:
        y_pred_flat = np.clip(y_pred_flat, 1e-6, None)
        return float(-mean_poisson_deviance(y_true_flat.ravel(), y_pred_flat.ravel()))
    return float(explained_variance_score(y_true_flat.ravel(), y_pred_flat.ravel()))


def _align_bins(latent: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Some candidates (a trial-pooled, no-time-axis latent; an occasional off-by-one internal
    rebinning in the point-process fit) do not return the same bin count they were given. If the
    latent has a single bin per trial, the target is pooled (meaned) to one bin per trial to match --
    the fair comparison for a candidate that discarded the time axis on purpose. Otherwise both are
    truncated to their shared leading bin count."""
    n_bins_latent, n_bins_target = latent.shape[1], target.shape[1]
    if n_bins_latent == n_bins_target:
        return latent, target
    if n_bins_latent == 1:
        return latent, target.mean(axis=1, keepdims=True)
    if n_bins_target == 1:
        return latent.mean(axis=1, keepdims=True), target
    n = min(n_bins_latent, n_bins_target)
    return latent[:, :n, :], target[:, :n, :]


def _decode_and_score(latent_train, target_train, latent_test, target_test, is_spiking, rng) -> dict:
    latent_train, target_train = _align_bins(latent_train, target_train)
    latent_test, target_test = _align_bins(latent_test, target_test)
    lt_flat = latent_train.reshape(-1, latent_train.shape[-1])
    tt_flat = target_train.reshape(-1, target_train.shape[-1])
    le_flat = latent_test.reshape(-1, latent_test.shape[-1])
    ye_flat = target_test.reshape(-1, target_test.shape[-1])
    preds = _decode(lt_flat, tt_flat, le_flat, is_spiking)
    score = _score(ye_flat, preds, is_spiking)

    # Shuffle baseline: same decoder recipe, fit against training targets whose TRIAL correspondence
    # to the latent state has been permuted, breaking the learnable trial-to-trial mapping while
    # preserving the held-out target's own marginal statistics (rate/power distribution, bin structure).
    n_trials = target_train.shape[0]
    perm = rng.permutation(n_trials)
    tt_shuffled = target_train[perm].reshape(-1, target_train.shape[-1])
    preds_shuffled = _decode(lt_flat, tt_shuffled, le_flat, is_spiking)
    shuffle_score = _score(ye_flat, preds_shuffled, is_spiking)
    return {"score": score, "shuffle_score": shuffle_score, "passes_shuffle": bool(score > shuffle_score)}


# ── Per-session, per-candidate evaluation ────────────────────────────────────────────────────────────

def _session_admission(record: dict) -> dict:
    """The cheap, deterministic part of a session's evaluation: the admission floor check and, for an
    admitted session, its trial/feature split. Recomputed fresh every run (the split is a pure
    function of a seed derived from the dataset and session name) rather than checkpointed, since
    nothing here costs more than a permutation."""
    X = record["X"]
    n_trials, n_features, n_bins = X.shape
    base = {"dataset": record["dataset"], "patient": record["patient"], "session": record["session"],
            "n_trials": int(n_trials), "n_features": int(n_features), "n_bins": int(n_bins)}
    if n_trials < MIN_TRIALS or n_features < MIN_FEATURES:
        return {**base, "status": "excluded", "exclusion_reason": "below_admission_floor",
                "n_trials_needed": MIN_TRIALS, "n_features_needed": MIN_FEATURES}
    rng = _seed(record["dataset"], record["session"], "split")
    train_idx, test_idx = trial_split(n_trials, rng)
    if len(train_idx) < MIN_TRAIN_TRIALS or len(test_idx) < MIN_TEST_TRIALS:
        return {**base, "status": "excluded", "exclusion_reason": "trial_split_below_admission_floor",
                "n_train_trials": int(len(train_idx)), "n_test_trials": int(len(test_idx))}
    held_in_idx, held_out_idx = feature_split(n_features, rng)
    if len(held_in_idx) < MIN_HELD_IN_FEATURES or len(held_out_idx) < MIN_HELD_OUT_FEATURES:
        return {**base, "status": "excluded", "exclusion_reason": "feature_split_below_admission_floor",
                "n_held_in_features": int(len(held_in_idx)), "n_held_out_features": int(len(held_out_idx))}
    return {**base, "status": "admitted", "n_train_trials": int(len(train_idx)), "n_test_trials": int(len(test_idx)),
            "n_held_in_features": int(len(held_in_idx)), "n_held_out_features": int(len(held_out_idx)),
            "train_idx": train_idx, "test_idx": test_idx, "held_in_idx": held_in_idx, "held_out_idx": held_out_idx}


def evaluate_session(key_prefix: str, record: dict, is_spiking: bool) -> dict:
    """Every expensive piece of one session's evaluation -- the rank-selection criteria and each
    candidate's fit -- is checkpointed under its own key, so a run interrupted mid-session resumes
    from the last completed candidate rather than the last completed session."""
    admission = _session_admission(record)
    if admission["status"] == "excluded":
        return admission

    X = record["X"]
    train_idx, test_idx = admission.pop("train_idx"), admission.pop("test_idx")
    held_in_idx, held_out_idx = admission.pop("held_in_idx"), admission.pop("held_out_idx")
    X_train, X_test = X[train_idx], X[test_idx]

    def _rank_selection():
        return rank_selection_criteria(X_train[:, held_in_idx, :], _seed(record["dataset"], record["session"], "rank"))
    criteria = run_checkpointed(f"{key_prefix}__rank_selection", _rank_selection)
    admission["rank_selection_criteria"] = criteria
    k_op = operating_rank(criteria, len(held_in_idx))
    admission["operating_rank"] = k_op

    train_held_in = X_train[:, held_in_idx, :].transpose(0, 2, 1)   # (n_train, n_bins, n_held_in)
    test_held_in = X_test[:, held_in_idx, :].transpose(0, 2, 1)
    train_held_out = X_train[:, held_out_idx, :].transpose(0, 2, 1)
    test_held_out = X_test[:, held_out_idx, :].transpose(0, 2, 1)
    train_all = X_train.transpose(0, 2, 1)
    test_all = X_test.transpose(0, 2, 1)

    candidates_out = {}
    for name, fit_fn in CANDIDATES.items():
        if not is_spiking and name in SPIKING_ONLY_CANDIDATES:
            candidates_out[name] = {"status": "not_applicable_to_data_type"}
            continue

        def _fit(name=name, fit_fn=fit_fn):
            cand_rng = _seed(record["dataset"], record["session"], name)
            out = _evaluate_one_candidate(name, fit_fn, k_op, train_held_in, test_held_in,
                                           train_held_out, test_held_out, train_all, test_all, is_spiking, cand_rng)
            if name == "principal_components":
                out["rank_sweep"] = _pca_rank_sweep(train_all, test_all, is_spiking, cand_rng)
            return out

        candidates_out[name] = run_checkpointed(f"{key_prefix}__{name}", _fit)
    admission["candidates"] = candidates_out
    return admission


def _evaluate_one_candidate(name, fit_fn, k_op, train_held_in, test_held_in, train_held_out, test_held_out,
                             train_all, test_all, is_spiking, rng) -> dict:
    try:
        unit_fit = fit_fn(train_held_in, test_held_in, k_op, rng, is_spiking, BIN_MS)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"unexpected exception during feature-holdout fit: {exc}"}
    if unit_fit["status"] == "not_applicable_to_data_type":
        return unit_fit
    if unit_fit["status"] != "fitted":
        return unit_fit
    try:
        unit_holdout = _decode_and_score(unit_fit["latent_train"], train_held_out, unit_fit["latent_test"],
                                          test_held_out, is_spiking, rng)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"read-out fit raised on the feature-holdout score: {exc}"}

    try:
        trial_fit = fit_fn(train_all, test_all, k_op, rng, is_spiking, BIN_MS)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"unexpected exception during trial-holdout fit: {exc}"}
    if trial_fit["status"] != "fitted":
        return {"status": "fitted_feature_holdout_only", "k_used": unit_fit["k_used"],
                "held_out_feature_score": unit_holdout, "trial_holdout_failure_reason": trial_fit.get("reason")}
    try:
        trial_holdout = _decode_and_score(trial_fit["latent_train"], train_all, trial_fit["latent_test"],
                                           test_all, is_spiking, rng)
    except Exception as exc:
        return {"status": "fitted_feature_holdout_only", "k_used": unit_fit["k_used"],
                "held_out_feature_score": unit_holdout, "trial_holdout_failure_reason": str(exc)}
    return {"status": "fitted", "k_used": unit_fit["k_used"],
            "held_out_feature_score": unit_holdout, "held_out_trial_score": trial_holdout}


def _pca_rank_sweep(train_all, test_all, is_spiking, rng) -> dict:
    """Additional held-out-trial (self-reconstruction, full feature set) scores at every rank in the
    declared sweep -- cheap for principal components since each rank is a truncation of one SVD -- so
    the delivered rank's position in the sweep is visible without re-running the whole candidate
    roster at every rank, which the other candidates' fitting cost does not afford within this gate."""
    sweep = {}
    n_features = train_all.shape[-1]
    for k in PCA_RANK_SWEEP:
        if k >= n_features:
            continue
        try:
            fit = fit_principal_components(train_all, test_all, k, rng, is_spiking, BIN_MS)
            if fit["status"] != "fitted":
                sweep[str(k)] = {"status": fit["status"], "reason": fit.get("reason")}
                continue
            scored = _decode_and_score(fit["latent_train"], train_all, fit["latent_test"], test_all, is_spiking, rng)
            sweep[str(k)] = {"status": "fitted", "k_used": fit["k_used"], "held_out_trial_score": scored}
        except Exception as exc:
            sweep[str(k)] = {"status": "failed_to_train", "reason": str(exc)}
    return sweep


# ── Checkpointing (atomic, per session-per-candidate) ────────────────────────────────────────────────

def _checkpoint_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(key: str) -> dict | None:
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def save_checkpoint(key: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(key)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(key: str, fit_fn) -> dict:
    cached = load_checkpoint(key)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(key, record)
    return record


# ── Aggregation: session/patient-level gate per corpus per candidate ────────────────────────────────

HUMAN_PATIENT_CLUSTERED_CORPORA = {
    "dandi_000469", "dandi_001187", "dandi_000574", "dandi_000574_ieeg", "dandi_000574_eeg",
}


def _candidate_pass(session_record: dict, candidate_name: str) -> bool | None:
    """True/False if the candidate's status is resolved for this admitted session, None if the
    candidate is not applicable to this session's data type (excluded from both numerator and
    denominator of the gate)."""
    cand = session_record.get("candidates", {}).get(candidate_name)
    if cand is None or cand.get("status") == "not_applicable_to_data_type":
        return None
    if cand.get("status") != "fitted":
        return False
    return bool(cand["held_out_feature_score"]["passes_shuffle"])


def gate_for_corpus(dataset: str, sessions: list[dict]) -> dict:
    admitted = [s for s in sessions if s["status"] == "admitted"]
    excluded = [s for s in sessions if s["status"] == "excluded"]
    cluster_by_patient = dataset in HUMAN_PATIENT_CLUSTERED_CORPORA
    gate = {}
    for candidate_name in CANDIDATES:
        per_session = [(s["patient"], _candidate_pass(s, candidate_name)) for s in admitted]
        per_session = [(p, v) for p, v in per_session if v is not None]
        if not per_session:
            gate[candidate_name] = {"status": "not_applicable_to_data_type"}
            continue
        if cluster_by_patient:
            by_patient: dict[str, list[bool]] = {}
            for patient, passed in per_session:
                by_patient.setdefault(patient, []).append(passed)
            unit_outcomes = {p: (sum(v) / len(v)) > 0.5 for p, v in by_patient.items()}
            clustering_unit = "patient"
        else:
            unit_outcomes = {f"{p}::{i}": v for i, (p, v) in enumerate(per_session)}
            clustering_unit = "session"
        n_total = len(unit_outcomes)
        n_passed = sum(unit_outcomes.values())
        fraction = n_passed / n_total if n_total else 0.0
        admissible = fraction > GATE_MAJORITY_THRESHOLD
        gate[candidate_name] = {
            "clustering_unit": clustering_unit, "n_total": n_total, "n_passed": n_passed,
            "fraction_passed": fraction, "admissible": admissible,
        }
        if not admissible:
            failing_examples = [s for s in admitted if _candidate_pass(s, candidate_name) is False][:5]
            gate[candidate_name]["excluding_examples"] = [
                {"session": s["session"], "patient": s["patient"], "n_trials": s["n_trials"],
                 "n_trials_needed": MIN_TRIALS, "n_train_trials": s.get("n_train_trials"),
                 "n_test_trials": s.get("n_test_trials"), "n_features": s["n_features"],
                 "n_features_needed": MIN_FEATURES,
                 "candidate_status": s["candidates"][candidate_name].get("status"),
                 "candidate_reason": s["candidates"][candidate_name].get("reason")}
                for s in failing_examples
            ]
    return {
        "n_sessions_yielded_by_shared_loader": len(sessions), "n_sessions_admitted": len(admitted),
        "n_sessions_excluded": len(excluded),
        "exclusions": [{"session": s["session"], "patient": s["patient"], "reason": s["exclusion_reason"],
                         "n_trials": s["n_trials"], "n_trials_needed": s.get("n_trials_needed"),
                         "n_features": s["n_features"], "n_features_needed": s.get("n_features_needed")}
                        for s in excluded],
        "gate": gate,
    }


CANDIDATE_DESCRIPTIONS = {
    "native_full_rank": "held-in features used directly as the state, with no reduction at all -- the reference every other candidate is measured against.",
    "principal_components": "a linear subspace fit by singular value decomposition, at a declared sweep of ranks.",
    "factor_analysis": "a linear latent model with an explicit per-feature noise term, rank chosen by cross-validated likelihood.",
    "gaussian_process_factor_analysis": "a single-trial latent trajectory with a Gaussian-process temporal prior and an explicit per-unit noise term, spiking populations only.",
    "sequential_autoencoder": "a recurrent variational autoencoder producing a per-timestep latent trajectory for spiking populations, its latent width and hidden width chosen automatically from a small internal validation split rather than fixed by hand.",
    "time_contrastive_embedding": "a self-supervised embedding trained only to place nearby timepoints of the same trial close together; built from no outcome label of any kind.",
    "temporal_diffusion_embedding": "a diffusion-based manifold embedding fit on a short time-lag-augmented view of each timepoint, exploiting within-trial temporal structure.",
    "trial_level_variational_autoencoder": "a plain variational autoencoder over one pooled feature vector per trial, carrying no time axis.",
}


def session_shard() -> tuple[int, int]:
    """Which slice of the session list this process is responsible for, as (index, count).

    Workers share one checkpoint directory and every record is written by atomic rename, so
    concurrent workers on disjoint slices cannot collide. A sharded process deliberately writes
    no pooled artifact: its view of the corpora is partial, and the corpus-level gate is only
    meaningful over the whole session list. Run once unsharded afterwards to aggregate -- every
    record is cached by then, so that pass only reloads sessions and writes the artifact.
    """
    index, count = (int(part) for part in os.environ.get(SHARD_VARIABLE, "0/1").split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"{SHARD_VARIABLE} must be i/n with 0 <= i < n; got {index}/{count}")
    return index, count


def main() -> None:
    t0 = time.time()
    shard_index, shard_count = session_shard()
    root = data_root()
    corpora: dict[str, list[dict]] = {}

    print("Building spiking sessions...", file=sys.stderr)
    for i, entry in enumerate(spiking_sessions(root)):
        if i % shard_count != shard_index:
            continue
        key = f"spiking__{entry['dataset']}__{entry['session']}"
        out = evaluate_session(key, entry, is_spiking=True)
        corpora.setdefault(entry["dataset"], []).append(out)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1} spiking sessions, {time.time() - t0:.0f}s", file=sys.stderr)

    print("Building field-potential sessions...", file=sys.stderr)
    for i, entry in enumerate(field_potential_sessions(root)):
        if i % shard_count != shard_index:
            continue
        key = f"field_potential__{entry['dataset']}__{entry['session']}"
        out = evaluate_session(key, entry, is_spiking=False)
        corpora.setdefault(entry["dataset"], []).append(out)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1} field-potential sessions, {time.time() - t0:.0f}s", file=sys.stderr)

    if shard_count > 1:
        print(f"Shard {shard_index} of {shard_count} finished its sessions in "
              f"{time.time() - t0:.0f}s; checkpoints written, no artifact aggregated.", file=sys.stderr)
        return

    data_type_by_dataset = {
        "dandi_000469": "spiking", "dandi_001187": "spiking", "dandi_000574": "spiking",
        "inagaki_alm5": "spiking", "watters_2026": "spiking",
        "dandi_000574_ieeg": "field_potential", "dandi_000574_eeg": "field_potential",
    }
    output = {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "bin_ms": BIN_MS,
        "test_trial_fraction": TEST_TRIAL_FRACTION,
        "held_out_feature_fraction": HELD_OUT_FEATURE_FRACTION,
        "gate_majority_threshold": GATE_MAJORITY_THRESHOLD,
        "pca_rank_sweep": list(PCA_RANK_SWEEP),
        "candidate_roster": CANDIDATE_DESCRIPTIONS,
        "corpora": {
            dataset: {"data_type": data_type_by_dataset.get(dataset, "unknown"), **gate_for_corpus(dataset, sessions),
                      "sessions": sessions}
            for dataset, sessions in corpora.items()
        },
        "wall_clock_s": None,
    }
    output["wall_clock_s"] = round(time.time() - t0, 1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))
    print(f"Wrote {OUTPUT_PATH} in {output['wall_clock_s']}s", file=sys.stderr)
    for dataset, corpus in output["corpora"].items():
        print(f"{dataset}: {corpus['n_sessions_admitted']}/{corpus['n_sessions_yielded_by_shared_loader']} admitted")
        for cand, g in corpus["gate"].items():
            if g.get("status") == "not_applicable_to_data_type":
                continue
            print(f"  {cand}: admissible={g['admissible']} ({g['n_passed']}/{g['n_total']} {g['clustering_unit']}s)")


if __name__ == "__main__":
    main()
