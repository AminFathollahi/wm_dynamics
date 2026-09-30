"""Read-out timing pilot: for the four benchmark candidates whose normal form
sees future time bins at inference (Gaussian-process factor analysis, the
sequential autoencoder, the neural data transformer, the recurrent switching
linear dynamical system), does memorandum content stay decodable when each is
made to infer online (causal, bins 1..t only) instead of full-context, and
under a chronological calibrate/test split instead of the benchmark's own
cross-validated folds? Principal components and demixed principal components
are carried on every split as bin-local references. Load 1 only, 16 drawn
region-session entries. This pilot does not decide which variant, if any,
enters the benchmark.

Run:
    python scripts/run_readout_timing_pilot.py
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    _path = str(ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_human_representation_benchmark as benchmark  # noqa: E402
import run_state_space_estimation_admissibility as estimation  # noqa: E402
from run_info_benchmark import block_identity, implementation_identity, run_checkpointed  # noqa: E402
from info_decoding import CTG_STEP, OPERATING_RANK, counts_to_spiketrains  # noqa: E402
from info_decoding import atomic_write, score_null_draw, score_observed, summarize_scores  # noqa: E402
from info_decoding import _seeded_folds, fold_signature, split_auc  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from spike_pipeline import BIN_MS  # noqa: E402
from statistics import bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "readout_timing_pilot.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_readout_timing_pilot"
CHECKPOINT_SCHEMA = "readout_timing_pilot_v1"
SEED0_DEFAULT = RESULTS / "human_representation_benchmark_seeds" / "seed_0.json"

DECODER = "linear"
N_SPLITS = 5
N_PERM_DEFAULT = 100
BENCHMARK_SEED = 0
N_PER_REGION = 8
REGIONS = ("hippocampus", "amygdala")
LEVEL = "load1_maintenance"

REFERENCE_MODELS = ("principal_components", "demixed_principal_components")
FUTURE_MODELS = (
    "gaussian_process_factor_analysis", "sequential_autoencoder",
    "recurrent_switching_linear_dynamics", "neural_data_transformer",
)
MODES = ("whole_trial", "causal")
SPLITS = ("stratified_folds", "chronological")
CANDIDATE_KEYS = REFERENCE_MODELS + tuple(f"{m}__{mode}" for m in FUTURE_MODELS for mode in MODES)


# ── Entry draw ────────────────────────────────────────────────────────────────

def drawn_entries(seed0_sessions: list[dict], n_per_region: int = N_PER_REGION) -> list[dict]:
    """Deterministic draw of at most one entry per patient per region, from the
    sessions seed_0.json recorded as loaded at load 1."""
    by_region: dict[str, dict[str, list[dict]]] = {}
    for row in seed0_sessions:
        if row.get("status") != "loaded" or row.get("level") != LEVEL:
            continue
        by_region.setdefault(row["region"], {}).setdefault(row["patient"], []).append(row)
    rng = np.random.default_rng(stable_seed("readout_timing_pilot"))
    drawn = []
    for region in sorted(by_region):
        patients = sorted(by_region[region])
        idx = sorted(rng.choice(len(patients), size=n_per_region, replace=False).tolist())
        for i in idx:
            patient = patients[i]
            chosen = sorted(by_region[region][patient], key=lambda r: r["session"])[0]
            drawn.append({"region": region, "patient": patient, "session": chosen["session"],
                          "release": chosen["release"]})
    return drawn


def load_entry(entry: dict, manifest_lookup: dict) -> tuple[np.ndarray, np.ndarray] | None:
    row = manifest_lookup.get((entry["release"], entry["patient"], entry["session"]))
    if row is None:
        return None
    session_data = benchmark.load_session_for_load(row["path"], entry["release"], 1)
    if session_data is None:
        return None
    return benchmark.build_cell(session_data, entry["region"])


# ── Splits ───────────────────────────────────────────────────────────────────

def stratified_fold_seed(region: str, session: str) -> int:
    """Matches run_info_benchmark.evaluate_block's own fold_seed formula exactly (the
    "info_benchmark" prefix, not this repository's "human_representation_benchmark" one --
    that second string is what run_human_representation_benchmark.evaluate_demixed_block
    uses for demixed principal components alone, a different split from the other seven
    native candidates within the benchmark itself)."""
    return stable_seed(f"info_benchmark|seed{BENCHMARK_SEED}|{region}|{session}|{LEVEL}|folds")


def stratified_split(region: str, session: str, labels: np.ndarray):
    folds = _seeded_folds(labels, N_SPLITS, stratified_fold_seed(region, session))
    if folds is None:
        return None, {"reason": "fewer than two trials in one or more classes"}
    return folds, {}


def chronological_split(labels: np.ndarray):
    """First two thirds (recorded order) calibrate, last third tests."""
    n = len(labels)
    n_calib = int(round(n * 2.0 / 3.0))
    calib_idx, test_idx = np.arange(0, n_calib), np.arange(n_calib, n)
    all_classes = np.unique(labels)
    test_classes, test_counts = np.unique(labels[test_idx], return_counts=True)
    calib_classes = np.unique(labels[calib_idx])
    counts = {str(c): int(n_) for c, n_ in zip(test_classes.tolist(), test_counts.tolist())}
    if not np.array_equal(calib_classes, all_classes) or not np.array_equal(test_classes, all_classes):
        return None, {"reason": "calibration and test sets do not contain the same classes", "test_counts": counts}
    if test_counts.min() < 2:
        return None, {"reason": "fewer than 2 test trials in a class", "test_counts": counts}
    return ((calib_idx, test_idx),), {"test_counts": counts}


# ── Representation fits: references (bin-local) and future-context models ──────

def _fit_reference(candidate: str, train_activity, train_labels, test_activity, rng):
    if candidate == "principal_components":
        return estimation.fit_principal_components(train_activity, test_activity, OPERATING_RANK, rng, True, BIN_MS)
    return benchmark.fit_demixed_principal_components(train_activity, train_labels, test_activity, OPERATING_RANK, rng)


def _fit_gpfa(train_X, test_X, k, rng, bin_ms, causal, bins=None, gpu_lock=None):
    from elephant.gpfa import GPFA
    import quantities as pq

    n_features = train_X.shape[-1]
    k_use = int(np.clip(k, 1, n_features - 1))
    try:
        train_st = counts_to_spiketrains(train_X.transpose(0, 2, 1).astype(int), bin_ms)
        gpfa = GPFA(bin_size=bin_ms * pq.ms, x_dim=k_use, em_max_iters=30, verbose=False)
        gpfa.fit(train_st)
        if not causal:
            test_st = counts_to_spiketrains(test_X.transpose(0, 2, 1).astype(int), bin_ms)
            latent_train = np.stack(gpfa.transform(train_st), axis=0).transpose(0, 2, 1)
            latent_test = np.stack(gpfa.transform(test_st), axis=0).transpose(0, 2, 1)
        else:
            latent_train = _gpfa_causal_transform(gpfa, train_X, bin_ms, k_use, bins)
            latent_test = _gpfa_causal_transform(gpfa, test_X, bin_ms, k_use, bins)
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"gpfa raised: {exc}"}
    return {"status": "fitted", "k_used": k_use, "latent_train": latent_train, "latent_test": latent_test}


def _gpfa_causal_transform(gpfa, X: np.ndarray, bin_ms: float, k_use: int, bins=None) -> np.ndarray:
    """Posterior mean at bin t from the trial truncated to bins 1..t: elephant's
    GPFA.transform builds the kernel covariance for whatever length it is given,
    so no fixed-length filler is needed (unlike the sequential autoencoder). bins
    (0-indexed, default every bin) restricts the truncation lengths actually
    computed to the ones scoring subsamples; other positions stay zero."""
    n_trials, n_bins, _ = X.shape
    out = np.zeros((n_trials, n_bins, k_use))
    positions = range(n_bins) if bins is None else bins
    for bin_idx in positions:
        t = bin_idx + 1
        truncated_st = counts_to_spiketrains(X[:, :t, :].transpose(0, 2, 1).astype(int), bin_ms)
        transformed = np.stack(gpfa.transform(truncated_st), axis=0).transpose(0, 2, 1)  # (n_trials, t, k)
        out[:, bin_idx, :] = transformed[:, -1, :]
    return out


def _fit_sae(train_X, test_X, k, rng, bin_ms, causal, bins=None, gpu_lock=None):
    if not causal:
        return estimation.fit_sequential_autoencoder(train_X, test_X, k, rng, True, bin_ms)
    seed = int(rng.integers(0, 2**31 - 1))
    if not estimation.LFADS_WORKER_PYTHON:
        return {"status": "failed_to_train", "reason": "configure executables.lfads_python"}
    with tempfile.TemporaryDirectory(prefix="lfads_causal_bridge_") as tmp_dir:
        in_path, out_path = Path(tmp_dir) / "input.npz", Path(tmp_dir) / "output.npz"
        payload = dict(train_X=train_X.astype(np.float32), test_X=test_X.astype(np.float32),
                        k=np.int64(k), seed=np.int64(seed), causal=np.bool_(True))
        if bins is not None:
            payload["bins"] = np.asarray(bins, dtype=np.int64)
        np.savez(in_path, **payload)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            proc = subprocess.run(
                [estimation.LFADS_WORKER_PYTHON, str(estimation.LFADS_WORKER_SCRIPT), str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=estimation.LFADS_SUBPROCESS_TIMEOUT_S, env=env,
            )
        except subprocess.TimeoutExpired:
            return {"status": "failed_to_train",
                    "reason": f"lfads_torch worker timed out after {estimation.LFADS_SUBPROCESS_TIMEOUT_S}s"}
        if proc.returncode != 0 or not out_path.exists():
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            return {"status": "failed_to_train", "reason": f"lfads_torch worker exited {proc.returncode}: {stderr_tail}"}
        result = np.load(out_path, allow_pickle=True)
        if "reason" in result.files:
            return {"status": "failed_to_train", "reason": str(np.asarray(result["reason"]).item())}
        return {"status": "fitted", "k_used": int(result["k_used"]),
                "latent_train": result["latent_train"], "latent_test": result["latent_test"]}


def _fit_slds(tr, te, k, rng, bin_ms, causal, bins=None, gpu_lock=None):
    return benchmark.fit_recurrent_switching_linear_dynamics(tr, te, k, rng, True, bin_ms, causal=causal, causal_bins=bins)


def _fit_ndt(tr, te, k, rng, bin_ms, causal, bins=None, gpu_lock=None):
    return benchmark.fit_neural_data_transformer(tr, te, k, rng, True, bin_ms, causal=causal, gpu_lock=gpu_lock)


FUTURE_MODEL_FITS = {
    "gaussian_process_factor_analysis": _fit_gpfa,
    "sequential_autoencoder": _fit_sae,
    "recurrent_switching_linear_dynamics": _fit_slds,
    "neural_data_transformer": _fit_ndt,
}


def _fit_one(candidate_key: str, train_activity, train_labels, test_activity, rng, bins=None, gpu_lock=None):
    if candidate_key in REFERENCE_MODELS:
        return _fit_reference(candidate_key, train_activity, train_labels, test_activity, rng)
    model, mode = candidate_key.rsplit("__", 1)
    causal = mode == "causal"
    return FUTURE_MODEL_FITS[model](train_activity, test_activity, OPERATING_RANK, rng, BIN_MS, causal,
                                     bins if causal else None, gpu_lock)


# ── Fit + score one (entry, split, candidate) cell, checkpointed per fold ──────

def _chronological_null_draw(fits: list[dict], labels: np.ndarray, decoder: str, calib_idx: np.ndarray,
                              test_idx: np.ndarray, shuffle_seed: int, decoder_seed: int, time_step: int) -> dict:
    """score_null_draw's own null (info_decoding._stratified_permutation) only assigns each
    fold's own test positions into its shuffled-label array via `shuffled[test] = ...`,
    leaving every other position as np.empty_like's uninitialised memory -- harmless with the
    benchmark's multi-fold cross-validation (every trial is someone's test position), but the
    chronological split has exactly one fold, so the whole calibration set is left as garbage.
    This shuffles the calibration set and the test set separately, each within itself, keeping
    each set's own label multiset, then scores with the same split_auc path score_null_draw
    itself uses."""
    rng = np.random.default_rng(shuffle_seed)
    shuffled = np.empty_like(labels)
    shuffled[calib_idx] = rng.permutation(labels[calib_idx])
    shuffled[test_idx] = rng.permutation(labels[test_idx])
    fit = fits[0]
    latent_train = np.asarray(fit["latent_train"], dtype=float)[:, ::time_step]
    latent_test = np.asarray(fit["latent_test"], dtype=float)[:, ::time_step]
    temporal_auc = split_auc(latent_train, shuffled[calib_idx], latent_test, shuffled[test_idx],
                              decoder=decoder, seed=decoder_seed)
    return {"status": "computed", "temporal_auc": temporal_auc}


def evaluate_cell(region: str, session: str, level: str, activity, labels, folds, split_name: str,
                   candidate_key: str, n_perm: int, checkpoint_dir: Path, identity: dict,
                   gpu_lock=None) -> dict:
    key_base = (
        f"readout_timing_pilot|{region}|{session}|{level}|{split_name}|{fold_signature(folds)}|{candidate_key}"
    )
    scored_bins = list(range(0, activity.shape[1], CTG_STEP))
    fits, fit_hits = [], []
    for fold_id, (train, test) in enumerate(folds):
        seed = stable_seed(f"{key_base}|fold{fold_id}|representation")

        def _fit(tr=train, te=test, s=seed):
            return _fit_one(candidate_key, activity[tr], labels[tr], activity[te],
                             np.random.default_rng(s), scored_bins, gpu_lock)

        rec, hit = run_checkpointed(f"{key_base}|fold{fold_id}|representation", _fit, identity, checkpoint_dir, CHECKPOINT_SCHEMA)
        fits.append(rec)
        fit_hits.append(hit)
        if rec.get("status") != "fitted":
            break
    failed = next((f for f in fits if f.get("status") != "fitted"), None)
    if failed is not None:
        return {"status": failed.get("status", "failed_to_train"), "reason": failed.get("reason"),
                "fit_checkpoint_hits": int(sum(fit_hits))}

    def _observed():
        try:
            return score_observed(fits, labels, DECODER, folds, stable_seed(f"{key_base}|observed"), CTG_STEP)
        except Exception as exc:
            return {"status": "failed_to_score", "reason": str(exc)}

    observed, observed_hit = run_checkpointed(f"{key_base}|observed", _observed, identity, checkpoint_dir, CHECKPOINT_SCHEMA)
    if observed.get("status") != "computed":
        return {**observed, "fit_checkpoint_hits": int(sum(fit_hits))}

    null_draws, null_hits = [], 0
    for draw in range(n_perm):
        shuffle_seed = stable_seed(f"readout_timing_pilot|{region}|{session}|{level}|{split_name}|null|{draw}")
        decoder_seed = stable_seed(f"readout_timing_pilot|{region}|{session}|{level}|{split_name}|{DECODER}|null_decoder|{draw}")

        def _null(s=shuffle_seed, d=decoder_seed):
            try:
                if split_name == "chronological":
                    calib_idx, test_idx = folds[0]
                    return _chronological_null_draw(fits, labels, DECODER, calib_idx, test_idx, s, d, CTG_STEP)
                return score_null_draw(fits, labels, DECODER, folds, s, d, CTG_STEP)
            except Exception as exc:
                return {"status": "failed_to_score", "reason": str(exc)}

        null, hit = run_checkpointed(f"{key_base}|null|{draw}", _null, identity, checkpoint_dir, CHECKPOINT_SCHEMA)
        if null.get("status") != "computed":
            return {**null, "fit_checkpoint_hits": int(sum(fit_hits)), "null_checkpoint_hits": null_hits}
        null_draws.append(np.asarray(null["temporal_auc"], dtype=float))
        null_hits += int(hit)

    record = summarize_scores(np.asarray(observed["temporal_auc"], dtype=float), np.stack(null_draws), DECODER)
    record["k_used"] = int(fits[0]["k_used"])
    for name in benchmark.INFERENCE_COUNT_KEYS:
        if any(name in f for f in fits):
            record[name] = int(sum(f.get(name, 0) for f in fits))
    record["n_splits"] = int(len(folds))
    record["fit_checkpoint_hits"] = int(sum(fit_hits))
    record["null_checkpoint_hits"] = null_hits
    return record


def _strip_arrays(record: dict) -> dict:
    return {k: v for k, v in record.items() if not isinstance(v, np.ndarray)}


# ── Consistency check ───────────────────────────────────────────────────────
# principal_components and gaussian_process_factor_analysis are scored by the benchmark
# through evaluate_block, on the same info_benchmark-seeded folds this pilot now uses for
# every candidate: their observed auc must match seed_0.json to within 1e-9. demixed_principal_components
# is scored by the benchmark's own evaluate_demixed_block, on a *different* fold seed
# (the "human_representation_benchmark" string), so within-pilot it uses the same folds as
# everything else here but seed_0.json's record does not -- its difference is reported, not
# held to a tolerance. sequential_autoencoder's fit seed hashes the code identity, so a new
# snapshot gives it a new seed regardless of folds: reported as seed-to-seed variation.
STRICT_CONSISTENCY_CANDIDATES = ("principal_components", "gaussian_process_factor_analysis")
DIFFERENCE_ONLY_CANDIDATES = ("demixed_principal_components",)
SEED_VARIATION_CANDIDATES = ("sequential_autoencoder",)
CONSISTENCY_CANDIDATES = STRICT_CONSISTENCY_CANDIDATES + DIFFERENCE_ONLY_CANDIDATES + SEED_VARIATION_CANDIDATES


def consistency_check(pilot_records: dict, seed0_index: dict) -> list[dict]:
    mismatches = []
    for (entry_key, region, session, candidate) in pilot_records:
        ours = pilot_records[(entry_key, region, session, candidate)]
        base_candidate = candidate.removesuffix("__whole_trial")
        if candidate.endswith("__causal"):
            continue
        theirs = seed0_index.get((region, session, LEVEL, base_candidate, DECODER))
        if theirs is None or ours.get("status") != "computed" or theirs.get("status") != "computed":
            continue
        for field in ("auc", "auc_above_null"):
            a, b = ours["same_time"].get(field), theirs["same_time"].get(field)
            difference = None if a is None or b is None else float(a - b)
            entry = {"session": session, "region": region, "candidate": candidate, "field": field,
                      "pilot": a, "benchmark": b, "difference": difference}
            if base_candidate in SEED_VARIATION_CANDIDATES:
                entry["kind"] = "seed_to_seed_variation_not_a_check"
                mismatches.append(entry)
            elif base_candidate in DIFFERENCE_ONLY_CANDIDATES:
                entry["kind"] = "difference_scored_on_its_own_folds_in_the_benchmark_not_a_check"
                mismatches.append(entry)
            elif field == "auc_above_null":
                # The shuffled-label null uses this pilot's own draws and count, so only auc is held to 1e-9.
                entry["kind"] = "difference_from_independent_null_draws_not_a_check"
                mismatches.append(entry)
            elif a is None or b is None or abs(a - b) > 1e-9:
                entry["kind"] = "mismatch"
                mismatches.append(entry)
    return mismatches


# ── Paired-difference statistics ────────────────────────────────────────────

def contrast(scores: dict, pairs: list[tuple], patient_by_entry: dict, label: str, n_boot: int = 2000) -> dict:
    diffs, patients = [], []
    for key_a, key_b in pairs:
        ra, rb = scores.get(key_a), scores.get(key_b)
        if ra is None or rb is None or ra.get("status") != "computed" or rb.get("status") != "computed":
            continue
        diffs.append(ra["same_time"]["auc_above_null"] - rb["same_time"]["auc_above_null"])
        patients.append(patient_by_entry[key_a[0]])
    if len(diffs) < 2:
        return {"status": "not_estimable", "label": label, "n_entries": len(diffs)}
    diffs_arr, patients_arr = np.asarray(diffs), np.asarray(patients)
    uniq = np.unique(patients_arr)
    patient_means = np.array([diffs_arr[patients_arr == p].mean() for p in uniq])
    rng_ci = np.random.default_rng(stable_seed(f"readout_timing_pilot|contrast|{label}|ci"))
    mean_stat, lo, hi = bootstrap_ci(patient_means, np.mean, n_boot=n_boot, rng=rng_ci)
    rng_sf = np.random.default_rng(stable_seed(f"readout_timing_pilot|contrast|{label}|signflip"))
    sign_flip = paired_sign_flip_test(patient_means, np.zeros_like(patient_means), alternative="two-sided", rng=rng_sf)
    mdd = minimum_detectable_paired_difference(patient_means)
    return {"status": "computed", "label": label, "mean": mean_stat, "ci_95_patient_cluster_bootstrap": [lo, hi],
            "sign_flip_p": sign_flip["p_value"], "mdd": mdd.get("mdd"),
            "n_entries": int(len(diffs)), "n_patients": int(len(uniq))}


def build_contrasts(scores: dict, entry_keys: list[str], patient_by_entry: dict) -> dict:
    out = {}
    for model in FUTURE_MODELS:
        for split in SPLITS:
            pairs = [((e, split, f"{model}__causal"), (e, split, f"{model}__whole_trial")) for e in entry_keys]
            out[f"causal_minus_whole_trial|{model}|{split}"] = contrast(scores, pairs, patient_by_entry, f"causal_minus_whole_trial|{model}|{split}")
    for candidate_key in CANDIDATE_KEYS:
        pairs = [((e, "chronological", candidate_key), (e, "stratified_folds", candidate_key)) for e in entry_keys]
        out[f"chronological_minus_stratified|{candidate_key}"] = contrast(scores, pairs, patient_by_entry, f"chronological_minus_stratified|{candidate_key}")
    for model in FUTURE_MODELS:
        for mode in MODES:
            for split in SPLITS:
                variant = f"{model}__{mode}"
                for reference in REFERENCE_MODELS:
                    pairs = [((e, split, variant), (e, split, reference)) for e in entry_keys]
                    out[f"{variant}_minus_{reference}|{split}"] = contrast(scores, pairs, patient_by_entry, f"{variant}_minus_{reference}|{split}")
    return out


# ── Per-bin causal-minus-whole-trial curve ─────────────────────────────────

def per_bin_curves(raw_records: dict, entry_keys: list[str]) -> dict:
    curves = {}
    for model in FUTURE_MODELS:
        for split in SPLITS:
            diffs = []
            for e in entry_keys:
                whole = raw_records.get((e, split, f"{model}__whole_trial"))
                causal = raw_records.get((e, split, f"{model}__causal"))
                if whole is None or causal is None or whole.get("status") != "computed" or causal.get("status") != "computed":
                    continue
                w = np.asarray(whole["same_time_auc"]) - np.asarray(whole["same_time_null_mean"])
                c = np.asarray(causal["same_time_auc"]) - np.asarray(causal["same_time_null_mean"])
                if w.shape != c.shape:
                    continue
                diffs.append(c - w)
            if diffs:
                stacked = np.stack(diffs)
                curves[f"{model}|{split}"] = {"mean_diff_by_bin": stacked.mean(axis=0).tolist(), "n_entries": len(diffs)}
            else:
                curves[f"{model}|{split}"] = {"mean_diff_by_bin": None, "n_entries": 0}
    return curves


# ── Parallel entry processing (process pool, one entry per worker) ─────────────
# Each worker is single-threaded (the OMP/OPENBLAS/MKL/... env vars pinned at the top of this
# file are set before any child is forked, so every child inherits them). The neural data
# transformer is the only GPU user; its subprocess calls are serialised across workers through
# _GPU_LOCK rather than measured to fit concurrently, which is always safe regardless of actual
# per-process GPU memory.

_GPU_LOCK = None


def _init_worker(lock) -> None:
    global _GPU_LOCK
    _GPU_LOCK = lock


def process_entry(entry: dict, manifest_lookup: dict, checkpoint_dir: Path, n_perm: int, runtime: dict) -> dict:
    entry_key = f"{entry['region']}::{entry['patient']}::{entry['session']}"
    cell = load_entry(entry, manifest_lookup)
    if cell is None:
        return {"entry_key": entry_key, "patient": entry["patient"],
                "entry_record": {"entry": entry, "status": "not_loaded"}, "raw_records": {}}
    activity, labels = cell
    entry_record = {"entry": entry, "status": "loaded", "n_trials": int(activity.shape[0]), "splits": {}}
    raw: dict = {}
    for split_name in SPLITS:
        if split_name == "stratified_folds":
            folds, split_info = stratified_split(entry["region"], f"{entry['patient']}_{entry['session']}", labels)
        else:
            folds, split_info = chronological_split(labels)
        if folds is None:
            entry_record["splits"][split_name] = {"status": "excluded", **split_info}
            continue
        identity = block_identity(activity, labels, folds, runtime)
        split_out = {"status": "computed", "n_folds": len(folds), "candidates": {}, **split_info}
        for candidate_key in CANDIDATE_KEYS:
            record = evaluate_cell(
                entry["region"], f"{entry['patient']}_{entry['session']}", LEVEL, activity, labels,
                folds, split_name, candidate_key, n_perm, checkpoint_dir, identity, _GPU_LOCK,
            )
            raw[(split_name, candidate_key)] = record
            split_out["candidates"][candidate_key] = _strip_arrays(record)
            reason = f" reason={record['reason']}" if record.get("reason") else ""
            print(f"readout_timing_pilot: {entry_key} {split_name} {candidate_key} "
                  f"status={record.get('status')}{reason}", flush=True)
        entry_record["splits"][split_name] = split_out
    return {"entry_key": entry_key, "patient": entry["patient"], "entry_record": entry_record, "raw_records": raw}


# ── CLI / main ───────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-out timing pilot for future-context representation candidates.")
    p.add_argument("--seed0-path", type=Path, default=SEED0_DEFAULT)
    p.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    p.add_argument("--entries-limit", type=int, help="restrict to the first N drawn entries (smoke runs)")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--workers", type=int, default=8)
    return p


def main() -> None:
    args = _parser().parse_args()
    if not args.seed0_path.exists():
        raise SystemExit(f"missing benchmark reference: {args.seed0_path}")
    seed0 = json.loads(args.seed0_path.read_text())
    entries = drawn_entries(seed0["sessions"])
    if args.entries_limit is not None:
        entries = entries[: args.entries_limit]

    seed0_index = {}
    for r in seed0.get("records", []):
        if r.get("status") == "computed":
            seed0_index[(r["corpus"], r["session"], r["level"], r["candidate"], r["decoder"])] = r

    primary, supplement = benchmark.build_manifest()
    manifest_lookup = {(row["release"], row["patient"], row["session"]): row for row in primary + supplement}

    runtime = implementation_identity()
    scores: dict = {}
    raw_records: dict = {}
    patient_by_entry: dict = {}
    entry_records: list[dict] = []
    started = time.time()

    def flush(complete: bool) -> None:
        if complete:
            consistency_input = {}
            for (entry_key, split_name, candidate_key), record in raw_records.items():
                if split_name != "stratified_folds" or candidate_key.removesuffix("__whole_trial") not in CONSISTENCY_CANDIDATES:
                    continue
                region, patient, session = entry_key.split("::")
                consistency_input[(entry_key, region, f"{patient}_{session}", candidate_key)] = record
            mismatches = consistency_check(consistency_input, seed0_index)
            entry_keys = list(patient_by_entry)
            contrasts = build_contrasts(scores, entry_keys, patient_by_entry)
            curves = per_bin_curves(raw_records, entry_keys)
        else:
            mismatches, contrasts, curves = [], {}, {}
        output = {
            "schema_version": "1.0.0",
            "code_commit": git_commit(ROOT),
            "status": "complete" if complete else "running",
            "drawn_entries": entries,
            "decoder": DECODER,
            "n_permutations": args.n_perm,
            "time_step": CTG_STEP,
            "operating_rank": OPERATING_RANK,
            "entries": entry_records,
            "consistency_check_against_seed_0": mismatches,
            "contrasts": contrasts,
            "per_bin_causal_minus_whole_trial_curves": curves,
            "wall_clock_s": float(time.time() - started),
        }
        atomic_write(args.output, canonical_json(output))

    max_workers = max(1, min(args.workers, len(entries)))
    mp_context = multiprocessing.get_context("fork")
    gpu_lock = mp_context.Lock()
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp_context, initializer=_init_worker, initargs=(gpu_lock,),
    ) as executor:
        futures = [
            executor.submit(process_entry, entry, manifest_lookup, args.checkpoint_dir, args.n_perm, runtime)
            for entry in entries
        ]
        for future in as_completed(futures):
            result = future.result()
            entry_key = result["entry_key"]
            patient_by_entry[entry_key] = result["patient"]
            entry_records.append(result["entry_record"])
            for (split_name, candidate_key), record in result["raw_records"].items():
                raw_records[(entry_key, split_name, candidate_key)] = record
                if record.get("status") == "computed":
                    scores[(entry_key, split_name, candidate_key)] = record
            flush(complete=False)

    flush(complete=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
