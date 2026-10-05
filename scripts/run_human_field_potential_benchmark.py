"""Human field-potential representation benchmark (Stage 3): the human single-unit
benchmark's own candidates, decoder, folding, scoring and checkpoint machinery, run on
continuous band-power state instead of spike counts, on two corpora -- DANDI 000574 depth
macro-contacts (set-size decoding, multi-region, cross-session calibration) and DANDI
000673 hippocampal microwire local field potential (category decoding at loads 1 and 3,
the same sessions and labels as the single-unit benchmark).

Run:
    python scripts/run_human_field_potential_benchmark.py --corpora dandi_000574 dandi_000673
"""
from __future__ import annotations

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    import os as _os
    _os.environ[_name] = "1"

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    _path = str(ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_info_benchmark as benchmark_core  # noqa: E402
from run_info_benchmark import block_identity, implementation_identity, run_checkpointed  # noqa: E402
from run_human_representation_benchmark import DEMIXED_CANDIDATE, load_session_for_load  # noqa: E402
import run_state_space_estimation_admissibility as estimation  # noqa: E402
from run_state_space_estimation_admissibility import _align_bins  # noqa: E402
from run_boran_modality_consistency import registry_sessions  # noqa: E402
from run_human_drift_spine_001187_000673 import (  # noqa: E402
    LFP_LINE_FREQ_HZ, _hippocampal_channel_mask, canonical_sessions,
)
from memorandum_decoding import fit_demixed_axes  # noqa: E402
from info_decoding import (  # noqa: E402
    CTG_STEP, DECODERS, FIELD_MAINTENANCE_WINDOW_S,
    OPERATING_RANK, _decoder_api, _seeded_folds, atomic_write, score_null_draw, score_observed, summarize_scores,
)
from preprocessing import (  # noqa: E402
    BAND_DEFINITIONS, BORAN_MAINS_HZ, FIELD_BANDS, MIN_BIPOLAR_CHANNELS, load_boran_nwb,
    multiband_maintenance_tensor, shank_bipolar_index_pairs,
)
from corpus_sessions import EPOCH_WINDOWS_S, MIN_TRIALS, data_root  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed,
)
from project_config import executable  # noqa: E402
from run_human_cosmoothing_benchmark import LN2, N_PENALTY_FOLDS, PENALTY_GRID, neuron_holdout  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "human_field_potential_benchmark.json"
UNIT_BENCHMARK_RESULTS = {
    "load1_maintenance": RESULTS / "human_representation_benchmark_seeds" / "seed_0.json",
    "load3_first_item_maintenance": RESULTS / "human_representation_benchmark_seeds" / "seed_0_load3.json",
}
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_human_field_potential_benchmark"
CHECKPOINT_SCHEMA = "human_field_potential_benchmark_v1"

GPFA_CANDIDATE = "gaussian_process_factor_analysis"
LFADS_CANDIDATE = "sequential_autoencoder"
SSM_CANDIDATE = "recurrent_switching_linear_dynamics"
NDT_CANDIDATE = "neural_data_transformer"
NATIVE_CONTINUOUS_CANDIDATES = (
    "native_full_rank", "principal_components", "factor_analysis",
    "time_contrastive_embedding", "temporal_diffusion_embedding",
)
CANDIDATES = NATIVE_CONTINUOUS_CANDIDATES + (
    DEMIXED_CANDIDATE, GPFA_CANDIDATE, LFADS_CANDIDATE, SSM_CANDIDATE, NDT_CANDIDATE,
)
REFERENCE_CANDIDATES = ("principal_components", DEMIXED_CANDIDATE)

SSM_WORKER_PYTHON = executable("ssm_python")
SSM_WORKER_SCRIPT = ROOT / "scripts" / "fit_recurrent_switching_linear_dynamics_worker.py"
SSM_SUBPROCESS_TIMEOUT_S = 1800
LFADS_WORKER_PYTHON = executable("lfads_python")
LFADS_WORKER_SCRIPT = ROOT / "scripts" / "fit_sequential_autoencoder_worker.py"
LFADS_SUBPROCESS_TIMEOUT_S = 900
FIELD_SOURCE_PATHS = (
    Path(__file__), ROOT / "src" / "preprocessing.py", SSM_WORKER_SCRIPT, LFADS_WORKER_SCRIPT,
)

CORPORA = ("dandi_000574", "dandi_000673")
FIELD_574_LEVEL = "set_size"
FIELD_574_LEVELS = (FIELD_574_LEVEL,)
FIELD_673_LEVELS = (("load1_maintenance", 1), ("load3_first_item_maintenance", 3))
BIN_MS = 100
FIELD_574_N_BINS = int(round((FIELD_MAINTENANCE_WINDOW_S[1] - FIELD_MAINTENANCE_WINDOW_S[0]) * 1000.0 / BIN_MS))
FIELD_673_MAINT_WIN = EPOCH_WINDOWS_S["delay"]  # 2.3 s, the single-unit benchmark's own window
FIELD_673_N_BINS = int(round(FIELD_673_MAINT_WIN * 1000.0 / BIN_MS))
FIELD_673_PAD_S = 0.5  # context on each side of the maintenance window, filtered then sliced away,
# so band-pass edge effects sit outside the scored window (matches lfp_maintenance_tensor's own
# full-epoch-then-slice convention for the depth-contact corpus)
N_SPLITS = 5
MEDIAN_BOOTSTRAP_SEED = 0
CALIBRATION_KS = (10, 20)
CALIBRATION_MIN_EXTRA_TRIALS = 15
CALIBRATION_MODES = ("within_session", "zero_shot") + tuple(f"calibrated_{k}" for k in CALIBRATION_KS)
INFERENCE_MODE = "whole_trial"


# ---------------------------------------------------------------------------
# Continuous-input candidate wiring. Every native candidate already branches on
# is_spiking internally (estimation.CANDIDATES); the wrapper below forces that
# branch to the continuous path. Inputs are z-scored by the caller with
# training-trial statistics, then run_info_benchmark.fit_candidate is reused
# unmodified for the shape and finiteness validation every other benchmark shares.
# ---------------------------------------------------------------------------

def failure_detail(exc: BaseException) -> str:
    tail = "\n".join("".join(traceback.format_exception(exc)).strip().splitlines()[-8:])
    return f"{type(exc).__name__}: {exc}\n{tail}"


def field_implementation_identity() -> dict:
    return {**implementation_identity(),
            "field_sources": {str(path.resolve()): sha256_file(path) for path in FIELD_SOURCE_PATHS}}


def train_statistics(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=(0, 1), keepdims=True)
    sd = train_x.std(axis=(0, 1), keepdims=True)
    return mu, np.where(sd < 1e-8, 1.0, sd)


def _zscore_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train_statistics(train_x)
    return (train_x - mu) / sd, (test_x - mu) / sd


def _continuous_candidate(fit_fn):
    def wrapped(train_x, test_x, k, rng, is_spiking, bin_ms):
        return fit_fn(train_x, test_x, k, rng, False, bin_ms)
    return wrapped


def fit_gaussian_process_factor_analysis_continuous(train_x, test_x, k, rng, is_spiking, bin_ms):
    """Elephant's own GPFA EM (gpfa_core.fit) and exact inference (exact_inference_with_ll,
    orthonormalize), called directly on already-binned continuous state -- the observation model
    these routines fit is linear-Gaussian regardless of input (the square-root transform for spike
    counts happens one layer up, in gpfa_util.get_seqs, and is simply skipped here). No upstream
    code is modified; get_seqs's binning step (which needs spike trains) is bypassed because the
    state is already binned."""
    from elephant.gpfa import gpfa_core
    n_features = train_x.shape[-1]
    k_use = int(np.clip(k, 1, n_features - 1))
    if k_use < 1:
        return {"status": "failed_to_train", "reason": f"n_features={n_features} too small for any rank"}

    def _seqs(x):
        return np.array(
            [(trial.shape[0], trial.T.astype(float)) for trial in x], dtype=[("T", int), ("y", object)],
        )

    try:
        seqs_train = _seqs(train_x)
        seqs_test = _seqs(test_x)
        if np.linalg.matrix_rank(np.cov(np.hstack(seqs_train["y"]))) < n_features:
            return {"status": "failed_to_train", "reason": "training observation covariance is rank deficient"}
        params, _ = gpfa_core.fit(seqs_train, x_dim=k_use, bin_width=float(bin_ms), em_max_iters=30, verbose=False)
        seqs_train_latent, _ = gpfa_core.exact_inference_with_ll(seqs_train, params, get_ll=False)
        seqs_test_latent, _ = gpfa_core.exact_inference_with_ll(seqs_test, params, get_ll=False)
        _, seqs_train_latent = gpfa_core.orthonormalize(params, seqs_train_latent)
        _, seqs_test_latent = gpfa_core.orthonormalize(params, seqs_test_latent)
        latent_train = np.stack([row.T for row in seqs_train_latent["latent_variable_orth"]], axis=0)
        latent_test = np.stack([row.T for row in seqs_test_latent["latent_variable_orth"]], axis=0)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"gpfa raised: {failure_detail(exc)}"}
    if not (np.isfinite(latent_train).all() and np.isfinite(latent_test).all()):
        return {"status": "failed_to_train", "reason": "non-finite gpfa latent values"}
    return {"status": "fitted", "k_used": k_use, "latent_train": latent_train, "latent_test": latent_test}


def _run_continuous_worker_bridge(python, script, train_x, test_x, k, seed, timeout_s, emission):
    if not python:
        return {"status": "failed_to_train", "reason": f"configure an executable for {script.name}"}
    with tempfile.TemporaryDirectory(prefix=f"{script.stem}_field_bridge_") as tmp_dir:
        in_path = Path(tmp_dir) / "input.npz"
        out_path = Path(tmp_dir) / "output.npz"
        np.savez(
            in_path, train_X=train_x.astype(np.float32), test_X=test_x.astype(np.float32),
            k=np.int64(k), seed=np.int64(seed), emission=np.array(emission),
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            proc = subprocess.run(
                [python, str(script), str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=timeout_s, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed_to_train", "reason": f"{script.name} worker timed out after {timeout_s}s"}
        if proc.returncode != 0 or not out_path.exists():
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            return {"status": "failed_to_train", "reason": f"{script.name} worker exited {proc.returncode}: {stderr_tail}"}
        result = np.load(out_path, allow_pickle=True)
        if "reason" in result.files:
            return {"status": "failed_to_train", "reason": str(np.asarray(result["reason"]).item())}
        return {"status": "fitted", "k_used": int(result["k_used"]),
                "latent_train": result["latent_train"], "latent_test": result["latent_test"]}


def fit_sequential_autoencoder_gaussian(train_x, test_x, k, rng, is_spiking, bin_ms):
    seed = int(rng.integers(0, 2**31 - 1))
    return _run_continuous_worker_bridge(
        LFADS_WORKER_PYTHON, LFADS_WORKER_SCRIPT, train_x, test_x, k, seed,
        LFADS_SUBPROCESS_TIMEOUT_S, emission="gaussian",
    )


def fit_recurrent_switching_linear_dynamics_gaussian(train_x, test_x, k, rng, is_spiking, bin_ms):
    seed = int(rng.integers(0, 2**31 - 1))
    return _run_continuous_worker_bridge(
        SSM_WORKER_PYTHON, SSM_WORKER_SCRIPT, train_x, test_x, k, seed,
        SSM_SUBPROCESS_TIMEOUT_S, emission="gaussian",
    )


def fit_neural_data_transformer_not_applicable(train_x, test_x, k, rng, is_spiking, bin_ms):
    return {
        "status": "not_applicable_to_data_type",
        "reason": "the upstream neural data transformer loss is Poisson on spike counts; no genuine "
                  "continuous-emission path exists in the upstream package",
    }


_FIELD_FITS = {name: _continuous_candidate(estimation.CANDIDATES[name]) for name in NATIVE_CONTINUOUS_CANDIDATES}
_FIELD_FITS[GPFA_CANDIDATE] = _continuous_candidate(fit_gaussian_process_factor_analysis_continuous)
_FIELD_FITS[LFADS_CANDIDATE] = _continuous_candidate(fit_sequential_autoencoder_gaussian)
_FIELD_FITS[SSM_CANDIDATE] = _continuous_candidate(fit_recurrent_switching_linear_dynamics_gaussian)
_FIELD_FITS[NDT_CANDIDATE] = fit_neural_data_transformer_not_applicable


def use_field_representation_fits() -> None:
    """Rebinds the shared dict (a new dict, never mutated in place) so run_info_benchmark.fit_candidate
    dispatches every continuous candidate name to the wiring above. Called from main rather than at
    import so that importing this module leaves the single-unit benchmark's dispatch unchanged."""
    benchmark_core.REPRESENTATION_FITS = {**benchmark_core.REPRESENTATION_FITS, **_FIELD_FITS}


def fit_demixed_principal_components_field(train_activity, train_labels, test_activity, rank, rng) -> dict:
    """Ridge dPCA on already z-scored input, with no Anscombe transform (the unit benchmark's
    fit_demixed_principal_components applies one because it fits counts)."""
    try:
        if len(np.unique(train_labels)) < 2:
            return {"status": "failed_to_train", "reason": "fewer than two classes in training fold"}
        fit = fit_demixed_axes(train_activity, train_labels, rng, rank)
        axes = fit["decoder"]
        k_used = int(axes.shape[1])
        if k_used < 1:
            return {"status": "failed_to_train", "reason": "no demixed axes recovered"}
        mu = train_activity.reshape(-1, train_activity.shape[-1]).mean(axis=0)
        latent_train = (train_activity - mu) @ axes
        latent_test = (test_activity - mu) @ axes
    except Exception as exc:
        return {"status": "failed_to_train", "reason": failure_detail(exc)}
    if not (np.isfinite(latent_train).all() and np.isfinite(latent_test).all()):
        return {"status": "failed_to_train", "reason": "non-finite latent values"}
    return {"status": "fitted", "candidate": DEMIXED_CANDIDATE, "k_used": k_used,
            "latent_train": latent_train, "latent_test": latent_test,
            "decoder_axes": axes, "encoder_axes": fit["encoder"]}


def fit_representation(candidate, z_train, train_labels, z_test, seed) -> dict:
    if candidate == DEMIXED_CANDIDATE:
        return fit_demixed_principal_components_field(
            z_train, train_labels, z_test, OPERATING_RANK, np.random.default_rng(seed),
        )
    return benchmark_core.fit_candidate(z_train, z_test, candidate, seed)


def fit_field_candidate(activity, labels, train_idx, test_idx, candidate, seed) -> dict:
    z_train, z_test = _zscore_train_test(activity[train_idx], activity[test_idx])
    return fit_representation(candidate, z_train, labels[train_idx], z_test, seed)


# ---------------------------------------------------------------------------
# Region assignment (000574 bipolar pairs) and channel holdout (both corpora).
# ---------------------------------------------------------------------------

def _location_region(location: str) -> str:
    prefix = location.split(",", 1)[0].strip()
    if prefix == "Hipp":
        return "hippocampus"
    if prefix == "Amyg":
        return "amygdala"
    return "other"


def bipolar_region_assignment(labels: list[str], locations: list[str]) -> list[str]:
    """Region per bipolar-referenced output channel, in bipolar_reference_by_shank's own output
    order (pairs, then orphans). A pair is assigned a region only when both its contacts carry
    that region's location prefix; an orphan (single-contact shank, CAR-referenced) uses its own
    contact's location."""
    pairs, orphans = shank_bipolar_index_pairs(labels)
    regions = []
    for i0, i1, _ in pairs:
        r0, r1 = _location_region(locations[i0]), _location_region(locations[i1])
        regions.append(r0 if r0 == r1 and r0 in ("hippocampus", "amygdala") else "other")
    for i in orphans:
        r = _location_region(locations[i])
        regions.append(r if r in ("hippocampus", "amygdala") else "other")
    return regions


def channel_holdout(n_channels: int, n_bands: int, seed: int) -> dict:
    """A quarter of the channels (rounded, at least 1) held out, with all of a held-out
    channel's bands held out together."""
    held_in_channels, held_out_channels = neuron_holdout(n_channels, seed)

    def _features(channels):
        if len(channels) == 0:
            return np.array([], dtype=int)
        return np.concatenate([np.arange(c * n_bands, (c + 1) * n_bands) for c in channels])

    return {
        "held_in_channels": held_in_channels, "held_out_channels": held_out_channels,
        "held_in_features": entry_feature_indices(held_in_channels, n_bands),
        "held_out_features": entry_feature_indices(held_out_channels, n_bands),
    }


def entry_feature_indices(channel_indices: np.ndarray, n_bands: int) -> np.ndarray:
    if len(channel_indices) == 0:
        return np.array([], dtype=int)
    return np.concatenate([np.arange(c * n_bands, (c + 1) * n_bands) for c in channel_indices])


# ---------------------------------------------------------------------------
# Entry construction: 000574 depth and 000673 hippocampal LFP.
# ---------------------------------------------------------------------------

def load_000574_entries(row: dict) -> dict:
    """One session's manifest record plus, per region, an entry (activity, labels, channel
    labels, correct and error counts, kept and dropped bands) or an excluded reason."""
    path = data_root() / row["path"]
    manifest = {"patient": row["patient"], "session": row["session"], "release": "000574"}
    if not path.is_file():
        return {**manifest, "status": "not_loaded", "reason": "path not found on disk", "entries": {}}
    try:
        ieeg = load_boran_nwb(str(path), signal="ieeg", reject_channels=True, mains_hz=BORAN_MAINS_HZ)
    except Exception as exc:
        return {**manifest, "status": "not_loaded", "reason": f"load_boran_nwb raised: {failure_detail(exc)}", "entries": {}}
    keep = ieeg["valid"] & ~np.asarray(ieeg["artifact"], dtype=bool)
    if int(keep.sum()) < MIN_TRIALS:
        return {**manifest, "status": "not_loaded",
                "reason": f"n_trials={int(keep.sum())} < MIN_TRIALS={MIN_TRIALS}", "entries": {}}
    labels_all = np.asarray(ieeg["set_sizes"])[keep].astype(int)
    correct_all = np.asarray(ieeg["correct"], dtype=bool)[keep]
    tensor_out = multiband_maintenance_tensor(
        epochs=ieeg["epochs"][keep], times=ieeg["times"], labels=ieeg["electrode_labels"], srate=ieeg["srate"],
        window_s=FIELD_MAINTENANCE_WINDOW_S, n_bins=FIELD_574_N_BINS, referencing="bipolar", mains_hz=BORAN_MAINS_HZ,
    )
    n_bands = len(tensor_out["kept_bands"])
    regions = bipolar_region_assignment(ieeg["electrode_labels"], ieeg["electrode_locs"])
    n_channels = len(regions)
    channel_idx_by_region = {
        "hippocampus": [c for c in range(n_channels) if regions[c] == "hippocampus"],
        "amygdala": [c for c in range(n_channels) if regions[c] == "amygdala"],
        "pooled": list(range(n_channels)),
    }
    entries = {}
    for region, channel_idx in channel_idx_by_region.items():
        if len(channel_idx) < MIN_BIPOLAR_CHANNELS:
            entries[region] = {"status": "excluded", "n_channels": len(channel_idx),
                                "reason": f"n_channels={len(channel_idx)} < MIN_BIPOLAR_CHANNELS={MIN_BIPOLAR_CHANNELS}"}
            continue
        feature_idx = entry_feature_indices(np.asarray(channel_idx, dtype=int), n_bands)
        entries[region] = {
            "status": "loaded", "activity": tensor_out["tensor"][:, :, feature_idx], "labels": labels_all,
            "channel_labels": [tensor_out["channel_labels"][c] for c in channel_idx], "n_channels": len(channel_idx), "n_bands": n_bands, "kept_bands": tensor_out["kept_bands"],
            "dropped_bands": tensor_out["dropped_bands"], "n_trials": int(keep.sum()),
            "n_correct": int(correct_all.sum()), "n_error": int((~correct_all).sum()),
        }
    return {**manifest, "status": "loaded", "reason": None, "entries": entries}


def _000673_lfp_path(row: dict) -> Path | None:
    if row["primary_release"] == "001187":
        return (data_root() / row["lfp_path"]) if row["lfp_path"] else None
    if row["primary_release"] == "000673":
        return data_root() / row["primary_path"]
    return None


def load_000673_entry(row: dict, level_name: str, target_load: int) -> dict:
    manifest = {"patient": row["patient"], "session": row["session"], "release": "000673", "level": level_name}
    lfp_path = _000673_lfp_path(row)
    if lfp_path is None or not lfp_path.is_file():
        return {**manifest, "status": "not_loaded", "reason": "no 000673 LFP twin for this canonical session", "entries": {}}
    primary_path = data_root() / row["primary_path"]
    if not primary_path.is_file():
        return {**manifest, "status": "not_loaded", "reason": "primary_path not found on disk", "entries": {}}
    session_data = load_session_for_load(primary_path, row["primary_release"], target_load)
    if session_data is None:
        return {**manifest, "status": "not_loaded", "reason": "load_session_for_load admitted no trials", "entries": {}}
    try:
        with h5py.File(str(primary_path), "r") as handle:
            trials = handle["intervals/trials"]
            loads_all = trials["loads"][:].astype(int)
            accuracy_all = trials["response_accuracy"][:].astype(bool)
        keep = loads_all == target_load
        correct_all = accuracy_all[keep]
    except Exception:
        correct_all = None
    try:
        with h5py.File(str(lfp_path), "r") as handle:
            if "acquisition" not in handle or "LFPs" not in handle["acquisition"]:
                return {**manifest, "status": "not_loaded", "reason": "no acquisition/LFPs in the 000673 twin", "entries": {}}
            lfp = handle["acquisition/LFPs"]
            electrode_rows = lfp["electrodes"][:]
            locations = handle["general/extracellular_ephys/electrodes/location"][:][electrode_rows]
            hippocampal_mask = _hippocampal_channel_mask(locations)
            n_channels = int(hippocampal_mask.sum())
            if n_channels < MIN_BIPOLAR_CHANNELS:
                return {**manifest, "status": "excluded", "n_channels": n_channels,
                        "reason": f"n_channels={n_channels} < MIN_BIPOLAR_CHANNELS={MIN_BIPOLAR_CHANNELS}", "entries": {}}
            data = lfp["data"][:][:, hippocampal_mask]
            starting_time = float(lfp["starting_time"][()])
            rate = float(lfp["starting_time"].attrs["rate"])
    except Exception as exc:
        return {**manifest, "status": "not_loaded", "reason": f"LFP twin read raised: {failure_detail(exc)}", "entries": {}}

    maint_win = session_data["maint_win"]
    n_samples_window = int(round(maint_win * rate))
    pad_samples = int(round(FIELD_673_PAD_S * rate))
    n_samples_padded = n_samples_window + 2 * pad_samples
    t_maint = session_data["t_maint"]
    category = np.asarray(session_data["category"])
    epochs = np.full((len(t_maint), n_channels, n_samples_padded), np.nan)
    for trial, onset in enumerate(t_maint):
        start = int(round((onset - starting_time) * rate)) - pad_samples
        if start < 0 or start + n_samples_padded > data.shape[0]:
            continue
        epochs[trial] = data[start:start + n_samples_padded].T
    valid = ~np.isnan(epochs).any(axis=(1, 2))
    if int(valid.sum()) < MIN_TRIALS:
        return {**manifest, "status": "not_loaded",
                "reason": f"only {int(valid.sum())} trials fit inside the recorded LFP span", "entries": {}}
    labels_all = category[valid]
    if len(np.unique(labels_all)) < 2:
        return {**manifest, "status": "excluded", "reason": "fewer than two classes admitted", "entries": {}}
    # filter/envelope computed on the padded epoch, THEN the maintenance sub-window sliced out --
    # matches lfp_maintenance_tensor's own edge-effect avoidance for the depth-contact corpus.
    times = np.arange(n_samples_padded) / rate
    window_s = (FIELD_673_PAD_S, FIELD_673_PAD_S + maint_win)
    channel_labels = [f"lfp_ch{i}" for i in range(n_channels)]
    tensor_out = multiband_maintenance_tensor(
        epochs=epochs[valid], times=times, labels=channel_labels, srate=rate,
        window_s=window_s, n_bins=FIELD_673_N_BINS, referencing="as_released", mains_hz=LFP_LINE_FREQ_HZ,
    )
    n_correct = int(correct_all[valid].sum()) if correct_all is not None and len(correct_all) == len(valid) else None
    n_error = (int((~correct_all[valid]).sum())
               if correct_all is not None and len(correct_all) == len(valid) else None)
    entries = {"hippocampus": {
        "status": "loaded", "activity": tensor_out["tensor"], "labels": labels_all,
        "n_channels": n_channels, "n_bands": len(tensor_out["kept_bands"]), "kept_bands": tensor_out["kept_bands"],
        "dropped_bands": tensor_out["dropped_bands"], "n_trials": int(valid.sum()),
        "n_correct": n_correct, "n_error": n_error,
    }}
    return {**manifest, "status": "loaded", "reason": None, "entries": entries}


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------

_IMPLEMENTATION: dict | None = None


def implementation() -> dict:
    global _IMPLEMENTATION
    if _IMPLEMENTATION is None:
        _IMPLEMENTATION = field_implementation_identity()
    return _IMPLEMENTATION


def label_name(corpus: str) -> str:
    return "set_size" if corpus == "dandi_000574" else "category"


def same_time_score(temporal_auc) -> float:
    return float(np.nanmean(np.diag(np.atleast_2d(np.asarray(temporal_auc, dtype=float)))))


def folds_for(corpus, region, session, level, labels, n_splits, seed):
    fold_seed = stable_seed(f"human_field_potential_benchmark|seed{seed}|{corpus}|{region}|{session}|{level}|folds")
    return _seeded_folds(labels, n_splits, fold_seed)


def excluded_records(candidates, corpus, region, session, level, reason, **extra) -> list[dict]:
    return [{"status": "excluded", "corpus": corpus, "region": region, "session": session, "level": level,
              "candidate": c, "reason": reason, "mode": INFERENCE_MODE, **extra} for c in candidates]


def identity_key(identity: dict) -> str:
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Content-decoding score: same-time AUC above the shuffled-label null.
# ---------------------------------------------------------------------------

def content_base(seed, corpus, region, session, level, candidate, identity) -> str:
    return (f"seed{seed}|{corpus}|{region}|{session}|{level}|{candidate}|rank{OPERATING_RANK}"
            f"|identity{identity_key(identity)}")


def content_fold_fits(activity, labels, folds, candidate, base, identity, checkpoint_dir):
    fits, hits = [], []
    for fold_id, (train, test) in enumerate(folds):
        fit, hit = run_checkpointed(
            f"{base}|fold{fold_id}|representation",
            lambda tr=train, te=test, i=fold_id: fit_field_candidate(
                activity, labels, tr, te, candidate,
                stable_seed(f"human_field_potential_benchmark|{base}|fold{i}|representation"),
            ),
            identity, checkpoint_dir, CHECKPOINT_SCHEMA,
        )
        fits.append(fit)
        hits.append(hit)
        if fit.get("status") != "fitted":
            break
    return fits, hits


def content_observed(fits, labels, decoder, folds, base, time_step, identity, checkpoint_dir):
    score_base = f"{base}|step{time_step}|{decoder}"

    def compute():
        try:
            return score_observed(fits, labels, decoder, folds,
                                   stable_seed(f"human_field_potential_benchmark|{score_base}|observed"), time_step)
        except Exception as exc:
            return {"status": "failed_to_score", "reason": failure_detail(exc)}

    observed, hit = run_checkpointed(f"{score_base}|observed", compute, identity, checkpoint_dir, CHECKPOINT_SCHEMA)
    return observed, hit, score_base


def evaluate_content_block(corpus, region, session, level, activity, labels, candidates, decoders,
                            n_splits, n_perm, time_step, seed, checkpoint_dir, progress) -> list[dict]:
    seed_tag = f"seed{seed}"
    folds = folds_for(corpus, region, session, level, labels, n_splits, seed)
    common = {"corpus": corpus, "region": region, "session": session, "level": level, "seed": seed,
              "label": label_name(corpus), "mode": INFERENCE_MODE, "metric": "content_decoding"}
    if folds is None:
        return excluded_records(candidates, corpus, region, session, level,
                                 "fewer than two trials in one or more classes", metric="content_decoding")
    identity = block_identity(activity, labels, folds, implementation())
    records = []
    for candidate in candidates:
        started = time.perf_counter()
        base = content_base(seed, corpus, region, session, level, candidate, identity)
        fits, fit_hits = content_fold_fits(activity, labels, folds, candidate, base, identity, checkpoint_dir)
        failed = next((f for f in fits if f.get("status") != "fitted"), None)
        if failed is not None:
            record = {**common, "status": failed.get("status", "failed_to_train"), "candidate": candidate,
                       "reason": failed.get("reason"), "fit_checkpoint_hits": int(sum(fit_hits)),
                       "wall_seconds": time.perf_counter() - started}
            records.append(record)
            progress(record)
            continue

        for decoder in decoders:
            observed, observed_hit, score_base = content_observed(
                fits, labels, decoder, folds, base, time_step, identity, checkpoint_dir,
            )
            if observed.get("status") != "computed":
                record = {**common, **observed, "candidate": candidate, "decoder": decoder,
                           "wall_seconds": time.perf_counter() - started}
                records.append(record)
                progress(record)
                continue

            null_draws, null_hits, fail = [], 0, None
            for draw in range(n_perm):
                shuffle_seed = stable_seed(
                    f"human_field_potential_benchmark|{seed_tag}|{corpus}|{region}|{session}|{level}|null|{draw}")
                decoder_seed = stable_seed(f"human_field_potential_benchmark|{seed_tag}|{decoder}|null_decoder|{draw}")

                def compute_null(d=decoder, s=shuffle_seed, m=decoder_seed):
                    try:
                        return score_null_draw(fits, labels, d, folds, s, m, time_step)
                    except Exception as exc:
                        return {"status": "failed_to_score", "reason": failure_detail(exc)}

                null, hit = run_checkpointed(f"{score_base}|null|{draw}", compute_null, identity,
                                              checkpoint_dir, CHECKPOINT_SCHEMA)
                if null.get("status") != "computed":
                    fail = null
                    break
                null_draws.append(np.asarray(null["temporal_auc"], dtype=float))
                null_hits += int(hit)

            if fail is not None:
                record = {**common, **fail, "candidate": candidate, "decoder": decoder,
                           "completed_permutations": len(null_draws), "wall_seconds": time.perf_counter() - started}
            else:
                record = {
                    **common,
                    **summarize_scores(np.asarray(observed["temporal_auc"], dtype=float), np.stack(null_draws), decoder),
                    "candidate": candidate, "n_trials": int(len(labels)), "k_used": int(fits[0]["k_used"]),
                    "n_splits": int(len(folds)), "fit_checkpoint_hits": int(sum(fit_hits)),
                    "observed_checkpoint_hit": observed_hit, "null_checkpoint_hits": null_hits,
                    "wall_seconds": time.perf_counter() - started,
                }
            records.append(record)
            progress(record)
    return records


# ---------------------------------------------------------------------------
# Held-out-channel reconstruction (co-smoothing analogue): Gaussian bits per observation.
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(y: np.ndarray, mean: np.ndarray, variance: float) -> float:
    variance = max(float(variance), 1e-12)
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + (y - mean) ** 2 / variance)))


def bits_per_observation(ll_model: float, ll_null: float, n_obs: float) -> float:
    return (ll_model - ll_null) / (n_obs * LN2)


def select_ridge_penalty(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray) -> float:
    totals = []
    for alpha in PENALTY_GRID:
        total = 0.0
        for cv_train, cv_test in GroupKFold(n_splits=N_PENALTY_FOLDS).split(x_train, y_train, groups):
            model = Ridge(alpha=alpha).fit(x_train[cv_train], y_train[cv_train])
            variance = max(float(np.var(y_train[cv_train] - model.predict(x_train[cv_train]))), 1e-12)
            total += gaussian_log_likelihood(y_train[cv_test], model.predict(x_train[cv_test]), variance)
        totals.append(total)
    return PENALTY_GRID[int(np.argmax(totals))]


def readout_scores(latent_train, target_train, latent_test, target_test) -> dict:
    """Per held-out feature: ridge read-out from latent to feature fitted on the training rows
    (penalty by trial-grouped cross-validation on those rows), Gaussian log-likelihood on the
    test rows with the training residual variance, against a constant training mean with its
    training variance."""
    latent_train, target_train = _align_bins(np.asarray(latent_train, dtype=float), np.asarray(target_train, dtype=float))
    latent_test, target_test = _align_bins(np.asarray(latent_test, dtype=float), np.asarray(target_test, dtype=float))
    if latent_train.shape[1] != latent_test.shape[1]:
        return {"status": "failed_to_score", "reason": "train and test latents aligned to different bin counts"}
    k_used = latent_train.shape[-1]
    scaler = StandardScaler().fit(latent_train.reshape(-1, k_used))
    x_train = scaler.transform(latent_train.reshape(-1, k_used))
    x_test = scaler.transform(latent_test.reshape(-1, k_used))
    n_bins = latent_train.shape[1]
    groups = np.repeat(np.arange(latent_train.shape[0]), n_bins)
    n_features = target_train.shape[-1]
    predictions = np.zeros(target_test.shape)
    ll_model, ll_null, penalties = 0.0, 0.0, []
    try:
        for j in range(n_features):
            y_train = target_train[:, :, j].reshape(-1)
            y_test = target_test[:, :, j].reshape(-1)
            alpha = select_ridge_penalty(x_train, y_train, groups)
            model = Ridge(alpha=alpha).fit(x_train, y_train)
            model_variance = float(np.var(y_train - model.predict(x_train)))
            train_mean = float(y_train.mean())
            null_variance = float(np.var(y_train - train_mean))
            prediction = model.predict(x_test)
            predictions[:, :, j] = prediction.reshape(target_test.shape[:2])
            ll_model += gaussian_log_likelihood(y_test, prediction, model_variance)
            ll_null += gaussian_log_likelihood(y_test, np.full_like(y_test, train_mean), null_variance)
            penalties.append(float(alpha))
    except Exception as exc:
        return {"status": "failed_to_score", "reason": failure_detail(exc)}
    return {
        "status": "computed", "k_used": int(k_used), "n_held_out_features": int(n_features),
        "chosen_penalties": penalties, "n_bins": int(n_bins), "ll_model": ll_model, "ll_null": ll_null,
        "n_obs": int(target_test.size), "sum_sq_residual": float(np.sum((target_test - predictions) ** 2)),
        "sum_sq_total": float(np.sum((target_test - target_test.mean(axis=(0, 1), keepdims=True)) ** 2)),
    }


def aggregate_readout(parts: list[dict]) -> dict:
    ll_model = sum(r["ll_model"] for r in parts)
    ll_null = sum(r["ll_null"] for r in parts)
    n_obs = sum(r["n_obs"] for r in parts)
    ss_res = sum(r["sum_sq_residual"] for r in parts)
    ss_tot = sum(r["sum_sq_total"] for r in parts)
    return {
        "bits_per_observation": bits_per_observation(ll_model, ll_null, n_obs),
        "held_out_r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "n_obs": int(n_obs), "n_folds": len(parts), "k_used": [r["k_used"] for r in parts],
    }


def reconstruction_fold(candidate, activity, labels, train_idx, test_idx, held_in_features, held_out_features,
                         fold_id, seed_prefix) -> dict:
    z_train, z_test = _zscore_train_test(activity[train_idx][:, :, held_in_features],
                                          activity[test_idx][:, :, held_in_features])
    fit = fit_representation(candidate, z_train, labels[train_idx], z_test,
                              stable_seed(f"{seed_prefix}|fold{fold_id}|representation"))
    if fit.get("status") != "fitted":
        return {"status": fit.get("status", "failed_to_train"), "reason": fit.get("reason")}
    return readout_scores(fit["latent_train"], activity[train_idx][:, :, held_out_features],
                          fit["latent_test"], activity[test_idx][:, :, held_out_features])


def reconstruction_folds(candidate, activity, labels, folds, fold_channels, base, identity, checkpoint_dir):
    parts, hits = [], []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        fc = fold_channels[fold_id]
        part, hit = run_checkpointed(
            f"{base}|fold{fold_id}",
            lambda tr=train_idx, te=test_idx, hi=fc["held_in_features"], ho=fc["held_out_features"], f=fold_id:
                reconstruction_fold(candidate, activity, labels, tr, te, hi, ho, f, base),
            identity, checkpoint_dir, CHECKPOINT_SCHEMA,
        )
        parts.append(part)
        hits.append(hit)
        if part.get("status") != "computed":
            break
    return parts, hits


def evaluate_reconstruction_entry(corpus, region, session, level, activity, labels, n_bands, candidates,
                                   n_splits, seed, checkpoint_dir, progress) -> list[dict]:
    seed_tag = f"seed{seed}"
    folds = folds_for(corpus, region, session, level, labels, n_splits, seed)
    if folds is None:
        return excluded_records(candidates, corpus, region, session, level,
                                 "fewer than two trials in one or more classes", metric="reconstruction")
    n_channels = activity.shape[2] // n_bands
    fold_channels = [
        channel_holdout(n_channels, n_bands, stable_seed(
            f"human_field_potential_benchmark|{seed_tag}|{corpus}|{region}|{session}|{level}|fold{fold_id}|channel_holdout"))
        for fold_id in range(len(folds))
    ]
    if any(len(fc["held_in_channels"]) < 2 for fc in fold_channels):
        return excluded_records(candidates, corpus, region, session, level,
                                 "fewer than 2 held-in channels in a fold", metric="reconstruction")
    identity = {**block_identity(activity, labels, folds, implementation()),
                "held_out_channels": [fc["held_out_channels"].tolist() for fc in fold_channels]}
    common = {"corpus": corpus, "region": region, "session": session, "level": level, "seed": seed,
              "label": label_name(corpus), "mode": INFERENCE_MODE, "metric": "reconstruction",
              "n_channels": int(n_channels), "n_bands": int(n_bands)}
    records = []
    for candidate in candidates:
        started = time.perf_counter()
        base = f"human_field_potential_benchmark|reconstruction|{seed_tag}|{corpus}|{region}|{session}|{level}|{candidate}"
        parts, hits = reconstruction_folds(candidate, activity, labels, folds, fold_channels, base, identity, checkpoint_dir)
        failed = next((p for p in parts if p.get("status") != "computed"), None)
        if failed is not None:
            record = {**common, "status": failed.get("status", "failed_to_train"), "candidate": candidate,
                       "reason": failed.get("reason"), "checkpoint_hits": int(sum(hits))}
        else:
            record = {**common, "status": "computed", "candidate": candidate, **aggregate_readout(parts),
                       "checkpoint_hits": int(sum(hits))}
        record["wall_seconds"] = time.perf_counter() - started
        records.append(record)
        progress(record)
    return records


# ---------------------------------------------------------------------------
# Per-entry orchestration (content decoding and reconstruction, one activity load).
# ---------------------------------------------------------------------------

def emit_progress(record: dict) -> None:
    keys = ("corpus", "region", "session", "level", "candidate", "decoder", "metric", "mode", "status")
    print(" ".join(f"{k}={record[k]}" for k in keys if record.get(k) is not None), flush=True)


def session_record(row: dict, release: str, level: str, region: str, entry: dict) -> dict:
    base = {"patient": row["patient"], "session": row["session"], "release": release, "level": level, "region": region}
    if entry["status"] != "loaded":
        return {**base, **{k: v for k, v in entry.items() if k not in ("activity", "labels")}}
    return {**base, "status": "loaded", "n_trials": entry["n_trials"], "n_channels": entry["n_channels"],
            "n_bands": entry["n_bands"], "kept_bands": entry["kept_bands"], "dropped_bands": entry["dropped_bands"],
            "n_correct": entry["n_correct"], "n_error": entry["n_error"]}


def process_entries(corpus, session_key, row, release, level, entries, candidates, decoders, n_splits, n_perm,
                     time_step, seed, checkpoint_dir) -> dict:
    session_records, content_records, reconstruction_records = [], [], []
    for region, entry in entries.items():
        session_records.append(session_record(row, release, level, region, entry))
        if entry["status"] != "loaded":
            continue
        content_records.extend(evaluate_content_block(
            corpus, region, session_key, level, entry["activity"], entry["labels"], candidates, decoders,
            n_splits, n_perm, time_step, seed, checkpoint_dir, emit_progress,
        ))
        reconstruction_records.extend(evaluate_reconstruction_entry(
            corpus, region, session_key, level, entry["activity"], entry["labels"], entry["n_bands"], candidates,
            n_splits, seed, checkpoint_dir, emit_progress,
        ))
    return {"kind": "entry", "corpus": corpus, "session_key": session_key, "patient": row["patient"],
            "session_records": session_records, "content_records": content_records,
            "reconstruction_records": reconstruction_records}


def process_000574_entry(row, candidates, decoders, n_splits, n_perm, time_step, seed, checkpoint_dir) -> dict:
    loaded = load_000574_entries(row)
    if loaded["status"] != "loaded":
        return {"kind": "entry", "corpus": "dandi_000574", "session_key": row["session"], "patient": row["patient"],
                "session_records": [{**loaded, "level": FIELD_574_LEVEL}], "content_records": [],
                "reconstruction_records": []}
    return process_entries("dandi_000574", row["session"], row, "000574", FIELD_574_LEVEL, loaded["entries"],
                            candidates, decoders, n_splits, n_perm, time_step, seed, checkpoint_dir)


def process_000673_entry(row, level_name, target_load, candidates, decoders, n_splits, n_perm, time_step, seed,
                          checkpoint_dir) -> dict:
    loaded = load_000673_entry(row, level_name, target_load)
    session_key = f"{row['patient']}_{row['session']}"
    if loaded["status"] != "loaded":
        return {"kind": "entry", "corpus": "dandi_000673", "session_key": session_key, "patient": row["patient"],
                "session_records": [loaded], "content_records": [], "reconstruction_records": []}
    return process_entries("dandi_000673", session_key, row, "000673", level_name, loaded["entries"], candidates,
                            decoders, n_splits, n_perm, time_step, seed, checkpoint_dir)


# ---------------------------------------------------------------------------
# Cross-session calibration (000574 only; fixed contacts; all depth contacts pooled).
# ---------------------------------------------------------------------------

def _session_order_key(session: str) -> int:
    if "_ses-" in session:
        try:
            return int(session.rsplit("_ses-", 1)[1])
        except ValueError:
            return 0
    return 0


def frozen_transfer(candidate, activity_a, labels_a, activity_b, features, ks, seed, identity, key, checkpoint_dir):
    """One representation fitted on session A alone (z-scored with A's statistics) over ``features``, and
    the latent it gives every trial of session B under each input scaling: zero_shot uses A's statistics,
    calibrated_k uses statistics from B's first k trials only. B enters as inference input, never as
    fitting data."""
    a, b = activity_a[:, :, features], activity_b[:, :, features]
    mu_a, sd_a = train_statistics(a)
    variants = {"zero_shot": (b - mu_a) / sd_a}
    for k in ks:
        mu_k, sd_k = train_statistics(b[:k])
        variants[f"calibrated_{k}"] = (b - mu_k) / sd_k
    stacked = np.concatenate(list(variants.values()), axis=0)
    fit, _ = run_checkpointed(
        key, lambda: fit_representation(candidate, (a - mu_a) / sd_a, labels_a, stacked, seed),
        identity, checkpoint_dir, CHECKPOINT_SCHEMA,
    )
    if fit.get("status") != "fitted":
        return fit, None
    latent_b = np.asarray(fit["latent_test"], dtype=float)
    n_b = b.shape[0]
    return fit, {mode: latent_b[i * n_b:(i + 1) * n_b] for i, mode in enumerate(variants)}


def calibration_pair(patient, region, entry_a, entry_b, candidates, decoder, time_step, seed, checkpoint_dir,
                     pair_channels: dict | None = None) -> list[dict]:
    """Records for one ordered pair of sessions (A earlier, B later), per candidate and mode
    (within_session, zero_shot, calibrated_k), for both scores."""
    activity_a, labels_a = entry_a["activity"], entry_a["labels"]
    activity_b, labels_b = entry_b["activity"], entry_b["labels"]
    session_a, session_b = entry_a["session"], entry_b["session"]
    n_bands = entry_a["n_bands"]
    n_channels = activity_a.shape[2] // n_bands
    n_b = len(labels_b)
    ks = [k for k in CALIBRATION_KS if n_b >= k + CALIBRATION_MIN_EXTRA_TRIALS]
    pair_key = f"human_field_potential_benchmark|calibration|seed{seed}|{patient}|{region}|{session_a}|{session_b}"
    common = {"patient": patient, "region": region, "session_a": session_a, "session_b": session_b,
              "decoder": decoder, "label": FIELD_574_LEVEL, "inference_mode": INFERENCE_MODE, **(pair_channels or {})}
    records = [
        {**common, "status": "skipped", "mode": f"calibrated_{k}", "reason": f"n_trials_b={n_b} < k + {CALIBRATION_MIN_EXTRA_TRIALS}"}
        for k in CALIBRATION_KS if k not in ks
    ]
    holdout = channel_holdout(n_channels, n_bands, stable_seed(f"{pair_key}|channel_holdout"))
    if len(holdout["held_in_channels"]) < 2:
        return records + [{**common, "status": "excluded", "reason": "fewer than 2 held-in channels"}]
    within_folds = folds_for("dandi_000574", region, session_b, FIELD_574_LEVEL, labels_b, N_SPLITS, seed)
    impl = implementation()
    pair_identity = {"activity_a": benchmark_core._array_hash(activity_a), "labels_a": benchmark_core._array_hash(labels_a),
                     "activity_b": benchmark_core._array_hash(activity_b), "labels_b": benchmark_core._array_hash(labels_b),
                     "held_out_channels": holdout["held_out_channels"].tolist(), "implementation": impl}
    within_identity = (block_identity(activity_b, labels_b, within_folds, impl) if within_folds is not None else None)
    all_features = np.arange(activity_a.shape[2])
    held_in, held_out = holdout["held_in_features"], holdout["held_out_features"]
    api = _decoder_api()

    def add(candidate, mode, metric, result, started, **extra):
        records.append({**common, "candidate": candidate, "mode": mode, "metric": metric, **result, **extra,
                         "elapsed_seconds_in_candidate": time.perf_counter() - started})

    for candidate in candidates:
        base = f"{pair_key}|{candidate}"
        started = time.perf_counter()

        # within_session: 5-fold on B alone. Content reuses the entry-level fits and scores (same keys).
        if within_folds is None:
            add(candidate, "within_session", "content_decoding",
                {"status": "excluded", "reason": "fewer than two trials in one or more classes"}, started)
        else:
            entry_base = content_base(seed, "dandi_000574", region, session_b, FIELD_574_LEVEL, candidate, within_identity)
            fits, _ = content_fold_fits(activity_b, labels_b, within_folds, candidate, entry_base, within_identity, checkpoint_dir)
            failed = next((f for f in fits if f.get("status") != "fitted"), None)
            if failed is not None:
                add(candidate, "within_session", "content_decoding",
                    {"status": failed.get("status", "failed_to_train"), "reason": failed.get("reason")}, started)
            else:
                observed, _hit, _ = content_observed(fits, labels_b, decoder, within_folds, entry_base, time_step,
                                                      within_identity, checkpoint_dir)
                if observed.get("status") == "computed":
                    add(candidate, "within_session", "content_decoding",
                        {"status": "computed", "score": same_time_score(observed["temporal_auc"]), "n_test_trials": n_b}, started)
                else:
                    add(candidate, "within_session", "content_decoding", observed, started)
            fold_channels = [holdout] * len(within_folds)
            parts, _ = reconstruction_folds(candidate, activity_b, labels_b, within_folds, fold_channels,
                                             f"{base}|within_session|reconstruction", pair_identity, checkpoint_dir)
            failed = next((p for p in parts if p.get("status") != "computed"), None)
            if failed is not None:
                add(candidate, "within_session", "reconstruction",
                    {"status": failed.get("status", "failed_to_train"), "reason": failed.get("reason")}, started)
            else:
                summary = aggregate_readout(parts)
                add(candidate, "within_session", "reconstruction",
                    {"status": "computed", "score": summary["bits_per_observation"], "held_out_r2": summary["held_out_r2"],
                     "n_obs": summary["n_obs"], "n_test_trials": n_b}, started)

        # zero_shot and calibrated_k: representation fitted on all of A and frozen.
        rep_seed = stable_seed(f"{base}|frozen_representation")
        fit_full, latent_full = frozen_transfer(candidate, activity_a, labels_a, activity_b, all_features, ks, rep_seed,
                                                pair_identity, f"{base}|frozen_all_channels", checkpoint_dir)
        fit_in, latent_in = frozen_transfer(candidate, activity_a, labels_a, activity_b, held_in, ks, rep_seed,
                                            pair_identity, f"{base}|frozen_held_in_channels", checkpoint_dir)
        for mode in ("zero_shot", *[f"calibrated_{k}" for k in ks]):
            n_fit = None if mode == "zero_shot" else int(mode.rsplit("_", 1)[1])
            decoder_seed = stable_seed(f"{base}|{mode}|decoder")
            if latent_full is None:
                add(candidate, mode, "content_decoding",
                    {"status": fit_full.get("status", "failed_to_train"), "reason": fit_full.get("reason")}, started)
            else:
                def compute_content():
                    try:
                        if n_fit is None:
                            train_x, train_y = np.asarray(fit_full["latent_train"], dtype=float)[:, ::time_step], labels_a
                            test_x, test_y = latent_full[mode][:, ::time_step], labels_b
                        else:
                            train_x, train_y = latent_full[mode][:n_fit, ::time_step], labels_b[:n_fit]
                            test_x, test_y = latent_full[mode][n_fit:, ::time_step], labels_b[n_fit:]
                        auc = api.split_auc(train_x, train_y, test_x, test_y, decoder=decoder, seed=decoder_seed)
                        return {"status": "computed", "score": same_time_score(auc), "n_test_trials": int(len(test_y))}
                    except Exception as exc:
                        return {"status": "failed_to_score", "reason": failure_detail(exc)}

                result, _ = run_checkpointed(f"{base}|{mode}|content", compute_content, pair_identity,
                                              checkpoint_dir, CHECKPOINT_SCHEMA)
                add(candidate, mode, "content_decoding", result, started)
            if latent_in is None:
                add(candidate, mode, "reconstruction",
                    {"status": fit_in.get("status", "failed_to_train"), "reason": fit_in.get("reason")}, started)
            else:
                def compute_reconstruction():
                    if n_fit is None:
                        args = (np.asarray(fit_in["latent_train"], dtype=float), activity_a[:, :, held_out],
                                latent_in[mode], activity_b[:, :, held_out])
                    else:
                        args = (latent_in[mode][:n_fit], activity_b[:n_fit][:, :, held_out],
                                latent_in[mode][n_fit:], activity_b[n_fit:][:, :, held_out])
                    part = readout_scores(*args)
                    if part.get("status") != "computed":
                        return part
                    summary = aggregate_readout([part])
                    return {"status": "computed", "score": summary["bits_per_observation"],
                            "held_out_r2": summary["held_out_r2"], "n_obs": summary["n_obs"],
                            "n_test_trials": int(args[3].shape[0])}

                result, _ = run_checkpointed(f"{base}|{mode}|reconstruction", compute_reconstruction, pair_identity,
                                              checkpoint_dir, CHECKPOINT_SCHEMA)
                add(candidate, mode, "reconstruction", result, started)
    return records


def shared_channel_entries(entry_a: dict, entry_b: dict) -> tuple[dict, dict]:
    """Both entries restricted to the contacts they share, in the first session's order."""
    position_b = {label: i for i, label in enumerate(entry_b["channel_labels"])}
    keep_a = [i for i, label in enumerate(entry_a["channel_labels"]) if label in position_b]
    keep_b = [position_b[entry_a["channel_labels"][i]] for i in keep_a]

    def restrict(entry, keep):
        features = entry_feature_indices(np.asarray(keep, dtype=int), entry["n_bands"])
        return {**entry, "activity": entry["activity"][:, :, features],
                "channel_labels": [entry["channel_labels"][i] for i in keep], "n_channels": len(keep)}

    return restrict(entry_a, keep_a), restrict(entry_b, keep_b)


def process_000574_calibration(patient, rows, candidates, decoder, time_step, seed, checkpoint_dir) -> dict:
    pooled = []
    records = []
    for row in sorted(rows, key=lambda r: _session_order_key(r["session"])):
        loaded = load_000574_entries(row)
        entry = loaded["entries"].get("pooled") if loaded["status"] == "loaded" else None
        if entry is None or entry["status"] != "loaded":
            records.append({"patient": patient, "session_b": row["session"], "status": "not_loaded",
                             "reason": loaded.get("reason") or "pooled entry not loaded"})
            continue
        pooled.append({**entry, "session": row["session"]})
    for entry_a, entry_b in zip(pooled, pooled[1:]):
        identical = entry_a["channel_labels"] == entry_b["channel_labels"]
        shared_a, shared_b = (entry_a, entry_b) if identical else shared_channel_entries(entry_a, entry_b)
        pair_channels = {"n_shared_channels": int(shared_a["n_channels"]), "contact_sets_identical": identical}
        if shared_a["n_channels"] < MIN_BIPOLAR_CHANNELS:
            records.append({"patient": patient, "session_a": entry_a["session"], "session_b": entry_b["session"],
                             "status": "excluded", **pair_channels,
                             "reason": f"n_shared_channels={shared_a['n_channels']} < MIN_BIPOLAR_CHANNELS={MIN_BIPOLAR_CHANNELS}"})
            continue
        records.extend(calibration_pair(patient, "pooled", shared_a, shared_b, candidates, decoder, time_step, seed,
                                         checkpoint_dir, pair_channels))
    for record in records:
        if record.get("candidate") is not None and record.get("mode") is not None:
            emit_progress({**record, "corpus": "dandi_000574", "session": record.get("session_b")})
    return {"kind": "calibration", "patient": patient, "calibration_records": records}


# ---------------------------------------------------------------------------
# Summaries: candidate minus reference per entry, averaged within patient, patient-resampled interval.
# ---------------------------------------------------------------------------

def patient_estimate(diffs: list[float], patients: list[str], rng: np.random.Generator) -> dict:
    diffs_arr, patients_arr = np.asarray(diffs, dtype=float), np.asarray(patients)
    uniq = np.unique(patients_arr)
    if len(uniq) < 3:
        return {"status": "not_estimable", "n_sessions": len(diffs), "n_patients": int(len(uniq))}
    patient_means = np.array([diffs_arr[patients_arr == p].mean() for p in uniq])
    mean_stat, lo, hi = bootstrap_ci(patient_means, np.mean, n_boot=5000, rng=rng)
    sign_flip = paired_sign_flip_test(patient_means, np.zeros_like(patient_means), alternative="two-sided", rng=rng)
    median_stat, median_lo, median_hi = bootstrap_ci(
        patient_means, np.median, n_boot=5000, rng=np.random.default_rng(MEDIAN_BOOTSTRAP_SEED))
    return {
        "status": "estimable", "n_sessions": int(len(diffs)), "n_patients": int(len(uniq)), "mean_improvement": mean_stat,
        "median_improvement": median_stat, "median_ci_95_patient_cluster_bootstrap": [median_lo, median_hi],
        "ci_95_patient_cluster_bootstrap": [lo, hi], "sign_flip_p": sign_flip["p_value"],
        "mdd": minimum_detectable_paired_difference(patient_means).get("mdd"), "improves_on_reference": int(lo > 0.0),
    }


def summarize_against_references(values: dict, rng: np.random.Generator) -> list[dict]:
    """values: {(cell, entry, candidate): (patient, value)}; one summary per cell, reference and candidate."""
    cells = sorted({cell for (cell, _e, _c) in values}, key=str)
    candidates = sorted({c for (_cell, _e, c) in values})
    out = []
    for cell in cells:
        for reference in REFERENCE_CANDIDATES:
            for candidate in candidates:
                if candidate == reference:
                    continue
                diffs, patients = [], []
                for (c, entry, cd), (patient, value) in values.items():
                    if c != cell or cd != candidate:
                        continue
                    ref = values.get((cell, entry, reference))
                    if ref is None:
                        continue
                    diffs.append(value - ref[1])
                    patients.append(patient)
                out.append({**dict(cell), "candidate": candidate, "reference": reference,
                            **patient_estimate(diffs, patients, rng)})
    return out


def _cell(**fields) -> tuple:
    return tuple(fields.items())


def region_variants(region: str) -> tuple[str, ...]:
    return (region, "regions_pooled_by_patient") if region in ("hippocampus", "amygdala") else (region,)


def content_summary(records: list[dict], patient_by_session: dict, rng: np.random.Generator) -> list[dict]:
    values = {}
    for r in records:
        if r.get("status") != "computed":
            continue
        for metric in ("same_time", "cross_temporal"):
            value = r.get(metric, {}).get("auc_above_null")
            if value is None:
                continue
            for region in region_variants(r["region"]):
                entry = r["session"] if region == r["region"] else f"{r['session']}|{r['region']}"
                cell = _cell(corpus=r["corpus"], region=region, level=r["level"], decoder=r["decoder"],
                             metric=metric, label=r["label"], mode=r["mode"])
                values[(cell, entry, r["candidate"])] = (patient_by_session.get(r["session"], r["session"]), value)
    return summarize_against_references(values, rng)


def reconstruction_summary(records: list[dict], patient_by_session: dict, rng: np.random.Generator) -> list[dict]:
    values = {}
    for r in records:
        if r.get("status") != "computed":
            continue
        for metric, field in (("bits_per_observation", "bits_per_observation"), ("held_out_r2", "held_out_r2")):
            if r.get(field) is None:
                continue
            for region in region_variants(r["region"]):
                entry = r["session"] if region == r["region"] else f"{r['session']}|{r['region']}"
                cell = _cell(corpus=r["corpus"], region=region, level=r["level"], metric=metric,
                             label=r["label"], mode=r["mode"])
                values[(cell, entry, r["candidate"])] = (patient_by_session.get(r["session"], r["session"]), r[field])
    return summarize_against_references(values, rng)


def co_primary_summary(content: list[dict], reconstruction: list[dict]) -> list[dict]:
    """Co-primary when, at every level of the corpus, one score's interval lies above 0 and the other
    score's interval has its upper end at or above 0. dandi_000574 has a single level (set_size), so the
    rule is applied at that one level; the record states this."""
    content_index = {(r["corpus"], r["region"], r["level"], r["decoder"], r["candidate"], r["reference"]): r
                      for r in content if r["status"] == "estimable" and r["metric"] == "same_time"}
    recon_index = {(r["corpus"], r["region"], r["level"], r["candidate"], r["reference"]): r
                   for r in reconstruction if r["status"] == "estimable" and r["metric"] == "bits_per_observation"}
    combos = {(c, region, cand, ref, dec) for (c, region, _l, dec, cand, ref) in content_index}
    out = []
    for corpus, region, candidate, reference, decoder in sorted(combos):
        levels = FIELD_574_LEVELS if corpus == "dandi_000574" else tuple(n for n, _ in FIELD_673_LEVELS)
        per_level = []
        for level in levels:
            content_cell = content_index.get((corpus, region, level, decoder, candidate, reference))
            recon_cell = recon_index.get((corpus, region, level, candidate, reference))
            c_lo, c_hi = content_cell["ci_95_patient_cluster_bootstrap"] if content_cell else (None, None)
            r_lo, r_hi = recon_cell["ci_95_patient_cluster_bootstrap"] if recon_cell else (None, None)
            if content_cell is None or recon_cell is None:
                qualifies = None
            else:
                qualifies = bool((c_lo > 0.0 and r_hi >= 0.0) or (r_lo > 0.0 and c_hi >= 0.0))
            per_level.append({"level": level, "content_ci": [c_lo, c_hi], "reconstruction_ci": [r_lo, r_hi],
                               "qualifies": qualifies})
        out.append({
            "corpus": corpus, "region": region, "candidate": candidate, "reference": reference, "decoder": decoder,
            "per_level": per_level,
            "co_primary": bool(all(p["qualifies"] for p in per_level)) if all(p["qualifies"] is not None for p in per_level) else False,
            "levels_evaluated": [p["level"] for p in per_level],
            "note": ("applied at the single set_size level, the only level dandi_000574 has" if corpus == "dandi_000574"
                     else "applied at load1_maintenance and load3_first_item_maintenance, as for units"),
        })
    return out


def calibration_summary(records: list[dict], rng: np.random.Generator) -> list[dict]:
    """Per candidate, metric and mode: mode score minus within_session score for the same pair, averaged
    within patient, patient-resampled interval."""
    scores = {}
    for r in records:
        if r.get("status") == "computed" and r.get("mode") in CALIBRATION_MODES and r.get("candidate"):
            scores[(r["candidate"], r["metric"], r["patient"], r["session_a"], r["session_b"], r["mode"])] = r["score"]
    out = []
    groups = sorted({(c, m, mode) for (c, m, _p, _a, _b, mode) in scores if mode != "within_session"})
    for candidate, metric, mode in groups:
        diffs, patients = [], []
        for (c, m, patient, a, b, md), value in scores.items():
            if (c, m, md) != (candidate, metric, mode):
                continue
            reference = scores.get((c, m, patient, a, b, "within_session"))
            if reference is None:
                continue
            diffs.append(value - reference)
            patients.append(patient)
        out.append({"candidate": candidate, "metric": metric, "mode": mode, "reference_mode": "within_session",
                    **patient_estimate(diffs, patients, rng)})
    return out


def unit_benchmark_comparison(content_records: list[dict]) -> list[dict]:
    """Beside each matched dandi_000673 hippocampal session, the single-unit benchmark's own same-time
    area under the curve above the shuffled-label null. The single-unit results are read, never recomputed."""
    out = []
    for level, path in UNIT_BENCHMARK_RESULTS.items():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            out.append({"level": level, "status": "unreadable", "reason": failure_detail(exc)})
            continue
        unit = {(r["session"], r["candidate"], r["decoder"]): r["same_time"].get("auc_above_null")
                for r in payload.get("records", [])
                if r.get("status") == "computed" and r["corpus"] == "hippocampus" and r["level"] == level}
        for r in content_records:
            if r.get("status") != "computed" or r["corpus"] != "dandi_000673" or r["level"] != level:
                continue
            match = unit.get((r["session"], r["candidate"], r["decoder"]))
            if match is None:
                continue
            out.append({"level": level, "session": r["session"], "candidate": r["candidate"], "decoder": r["decoder"],
                        "field_auc_above_null": r["same_time"].get("auc_above_null"), "unit_auc_above_null": match,
                        "unit_source": path.name, "unit_source_status": payload.get("status")})
    return out


def render_summary(session_records, content_records, reconstruction_records, calibration_records, failures,
                    complete: bool, content_summ=None, recon_summ=None) -> str:
    lines = ["# Human field-potential benchmark", "", f"status: {'complete' if complete else 'running'}", ""]
    by_status: dict = {}
    for r in session_records:
        by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
    lines.append("## entries by status")
    lines += [f"- {s}: {n}" for s, n in sorted(by_status.items(), key=lambda kv: str(kv[0]))]
    lines += ["", "## records by corpus, score, candidate and status (count, mean seconds)"]
    groups: dict = {}
    for score, records in (("content", content_records), ("reconstruction", reconstruction_records),
                           ("calibration", calibration_records)):
        for r in records:
            if r.get("candidate") is None:
                continue
            key = (r.get("corpus", "dandi_000574"), score, r["candidate"], r.get("status"))
            groups.setdefault(key, []).append(r.get("wall_seconds", r.get("elapsed_seconds_in_candidate")))
    for (corpus, score, candidate, status), times in sorted(groups.items(), key=str):
        seconds = [t for t in times if t is not None]
        mean = f"{np.mean(seconds):.1f}" if seconds else "n/a"
        lines.append(f"- {corpus} {score} {candidate} {status}: {len(times)}, {mean}")
    lines += ["", f"## task failures: {len(failures)}"]
    lines += [f"- {f['task']}: {f['reason'].splitlines()[0]}" for f in failures]
    for title, rows in (("content", content_summ), ("reconstruction", recon_summ)):
        if not rows:
            continue
        lines += ["", f"## {title}: candidate minus reference (estimable cells)"]
        for r in rows:
            if r["status"] != "estimable":
                continue
            lo, hi = r["ci_95_patient_cluster_bootstrap"]
            cell = " ".join(f"{k}={r[k]}" for k in ("corpus", "region", "level", "decoder", "metric") if k in r)
            lines.append(f"- {cell} {r['candidate']} - {r['reference']}: {r['mean_improvement']:.4f} "
                         f"[{lo:.4f}, {hi:.4f}] n_patients={r['n_patients']} p={r['sign_flip_p']:.4f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Human field-potential representation benchmark.")
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--entries-limit", type=int, help="first N sessions of dandi_000574 (and of dandi_000673 unless overridden)")
    parser.add_argument("--entries-limit-000673", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--time-step", type=int, default=CTG_STEP)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--decoders", nargs="+", choices=DECODERS, default=list(DECODERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-only", action="store_true",
                        help="run only the cross-session calibration of dandi_000574 and write only its records and summary")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="score the entries only; the cross-session calibration is left to --calibration-only")
    parser.add_argument("--summaries-only", type=Path, metavar="EXISTING",
                        help="with --calibration-only: recompute the calibration summary from the records of an "
                             "existing output file and write it to --output, without refitting")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, help="Markdown summary path (default: beside --output)")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    return parser


def recompute_calibration_summary(existing: Path, output: Path) -> None:
    """Rewrites the calibration summary of a finished calibration-only file from its stored records."""
    if existing.resolve() == output.resolve():
        raise SystemExit("--output must differ from the file named by --summaries-only")
    artifact = json.loads(existing.read_text())
    artifact["calibration_summary"] = calibration_summary(artifact["cross_session_calibration"],
                                                          np.random.default_rng(1))
    artifact["summaries_recomputed"] = {"source_file": existing.name, "source_sha256": sha256_file(existing),
                                        "code_commit": git_commit(ROOT)}
    atomic_write(output, canonical_json(artifact))


def guarded(task: str, fn, *args) -> dict:
    try:
        return fn(*args)
    except Exception as exc:
        return {"kind": "failure", "task": task, "reason": failure_detail(exc)}


def main() -> None:
    args = _parser().parse_args()
    use_field_representation_fits()
    if args.n_splits < 2 or args.n_perm < 1 or args.time_step < 1:
        raise SystemExit("--n-splits must be at least 2; --n-perm and --time-step must be positive")
    if args.summaries_only is not None:
        if not args.calibration_only:
            raise SystemExit("--summaries-only applies to --calibration-only files")
        recompute_calibration_summary(args.summaries_only, args.output)
        print(f"wrote {args.output}", flush=True)
        return
    summary_path = args.summary or args.output.with_suffix(".md")
    started = time.time()
    session_records: list[dict] = []
    content_records: list[dict] = []
    reconstruction_records: list[dict] = []
    calibration_records: list[dict] = []
    failures: list[dict] = []
    patient_by_session: dict[str, str] = {}
    candidates, decoders = tuple(args.candidates), tuple(args.decoders)

    def flush(complete: bool) -> None:
        rng = lambda: np.random.default_rng(1)
        content_summ = content_summary(content_records, patient_by_session, rng()) if complete else []
        recon_summ = reconstruction_summary(reconstruction_records, patient_by_session, rng()) if complete else []
        calibration_summ = calibration_summary(calibration_records, rng()) if complete else []
        if args.calibration_only:
            artifact = {
                "version": CHECKPOINT_SCHEMA, "code_commit": git_commit(ROOT), "implementation": implementation(),
                "status": "complete" if complete else "running", "inference_mode": INFERENCE_MODE,
                "scope": {"corpora": ["dandi_000574"], "calibration_only": True, "candidates": list(candidates),
                          "decoder": "linear", "time_step": args.time_step, "seed": args.seed,
                          "calibration_ks": list(CALIBRATION_KS), "min_bipolar_channels": MIN_BIPOLAR_CHANNELS},
                "cross_session_calibration": calibration_records, "task_failures": failures,
                "calibration_summary": calibration_summ, "wall_clock_s": float(time.time() - started),
            }
            atomic_write(args.output, canonical_json(artifact))
            return
        artifact = {
            "version": CHECKPOINT_SCHEMA, "code_commit": git_commit(ROOT), "implementation": implementation(),
            "status": "complete" if complete else "running", "inference_mode": INFERENCE_MODE,
            "scope": {"corpora": args.corpora, "candidates": list(candidates), "decoders": list(decoders),
                      "n_splits": args.n_splits, "n_permutations": args.n_perm, "time_step": args.time_step,
                      "seed": args.seed, "entries_limit": args.entries_limit,
                      "entries_limit_000673": args.entries_limit_000673, "bands": list(FIELD_BANDS),
                      "band_definitions_hz": {b: list(BAND_DEFINITIONS[b]) for b in FIELD_BANDS}, "bin_ms": BIN_MS,
                      "operating_rank": OPERATING_RANK, "calibration_ks": list(CALIBRATION_KS)},
            "sessions": session_records, "content_records": content_records,
            "reconstruction_records": reconstruction_records, "cross_session_calibration": calibration_records,
            "task_failures": failures,
            "content_summary": content_summ, "reconstruction_summary": recon_summ,
            "co_primary_summary": co_primary_summary(content_summ, recon_summ) if complete else [],
            "calibration_summary": calibration_summ,
            "unit_benchmark_comparison": unit_benchmark_comparison(content_records) if complete else [],
            "wall_clock_s": float(time.time() - started),
        }
        atomic_write(args.output, canonical_json(artifact))
        atomic_write(summary_path, render_summary(session_records, content_records, reconstruction_records,
                                                    calibration_records, failures, complete, content_summ, recon_summ))

    tasks = []
    if "dandi_000574" in args.corpora:
        rows = registry_sessions(data_root())
        if args.entries_limit is not None:
            rows = rows[: args.entries_limit]
        for row in ([] if args.calibration_only else rows):
            tasks.append((f"entry {row['session']}", process_000574_entry,
                          (row, candidates, decoders, args.n_splits, args.n_perm, args.time_step, args.seed,
                           args.checkpoint_dir)))
        by_patient: dict[str, list[dict]] = {}
        for row in rows:
            by_patient.setdefault(row["patient"], []).append(row)
        for patient, patient_rows in by_patient.items():
            if len(patient_rows) > 1 and not args.skip_calibration:
                tasks.append((f"calibration {patient}", process_000574_calibration,
                              (patient, patient_rows, candidates, "linear", args.time_step, args.seed,
                               args.checkpoint_dir)))
    if "dandi_000673" in args.corpora and not args.calibration_only:
        rows = [r for r in canonical_sessions() if _000673_lfp_path(r) is not None]
        limit = args.entries_limit_000673 if args.entries_limit_000673 is not None else args.entries_limit
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            for level_name, target_load in FIELD_673_LEVELS:
                tasks.append((f"entry {row['patient']}_{row['session']} {level_name}", process_000673_entry,
                              (row, level_name, target_load, candidates, decoders, args.n_splits, args.n_perm,
                               args.time_step, args.seed, args.checkpoint_dir)))

    flush(False)
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(tasks) or 1)),
                              mp_context=multiprocessing.get_context("fork")) as executor:
        futures = {executor.submit(guarded, name, fn, *fn_args): name for name, fn, fn_args in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result["kind"] == "failure":
                failures.append(result)
            elif result["kind"] == "calibration":
                calibration_records.extend(result["calibration_records"])
            else:
                patient_by_session[result["session_key"]] = result["patient"]
                session_records.extend(result["session_records"])
                content_records.extend(result["content_records"])
                reconstruction_records.extend(result["reconstruction_records"])
            flush(False)
    flush(True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
