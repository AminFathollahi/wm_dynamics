"""Held-out-neuron co-smoothing score for the human representation benchmark: for every
candidate the content score evaluates (the delivered eight, the neural data transformer, the
recurrent switching linear dynamical system), on the same region-session entries and levels,
fit on a quarter of an entry's neurons held out per trial fold and score how well the resulting
latents predict the spike counts of the neurons never seen during fitting.

Run:
    python scripts/run_human_cosmoothing_benchmark.py
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import functools
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    _path = str(ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_human_representation_benchmark as benchmark  # noqa: E402
import run_info_benchmark as benchmark_core  # noqa: E402
from run_info_benchmark import block_identity, implementation_identity, run_checkpointed  # noqa: E402
from run_state_space_estimation_admissibility import _align_bins  # noqa: E402
from info_decoding import OPERATING_RANK, _seeded_folds, atomic_write  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "human_cosmoothing_benchmark.json"
SUMMARY_PATH = ROOT / "HUMAN_COSMOOTHING_BENCHMARK.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_human_cosmoothing_benchmark"
CHECKPOINT_SCHEMA = "human_cosmoothing_benchmark_v1"

CANDIDATES = benchmark.CANDIDATES
REFERENCE_CANDIDATES = benchmark.REFERENCE_CANDIDATES
LEVELS = benchmark.LEVELS
REGIONS = benchmark.REGIONS

N_SPLITS = 5
PENALTY_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_PENALTY_FOLDS = 3
RATE_FLOOR = 1e-9
LN2 = float(np.log(2.0))


def neuron_holdout(n_neurons: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A quarter of the neurons (rounded to the nearest integer, at least 1), held in/out."""
    n_held_out = max(1, round(n_neurons / 4))
    rng = np.random.default_rng(seed)
    held_out = np.sort(rng.choice(n_neurons, size=n_held_out, replace=False))
    held_in = np.setdiff1d(np.arange(n_neurons), held_out)
    return held_in, held_out


def poisson_log_likelihood(y: np.ndarray, rate: np.ndarray) -> float:
    rate = np.clip(rate, RATE_FLOOR, None)
    return float(np.sum(y * np.log(rate) - rate - gammaln(y + 1.0)))


def bits_per_spike(ll_model: float, ll_null: float, n_spikes: float) -> float:
    return (ll_model - ll_null) / (n_spikes * LN2)


def select_penalty(x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray) -> float:
    """3-fold trial-grouped cross-validation over the training rows only, by summed
    held-out Poisson log-likelihood."""
    totals = []
    for alpha in PENALTY_GRID:
        total = 0.0
        for cv_train, cv_test in GroupKFold(n_splits=N_PENALTY_FOLDS).split(x_train, y_train, groups):
            model = PoissonRegressor(alpha=alpha, max_iter=300).fit(x_train[cv_train], y_train[cv_train])
            total += poisson_log_likelihood(y_train[cv_test], model.predict(x_train[cv_test]))
        totals.append(total)
    return PENALTY_GRID[int(np.argmax(totals))]


def fit_readout(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, float]:
    alpha = select_penalty(x_train, y_train, groups)
    model = PoissonRegressor(alpha=alpha, max_iter=300).fit(x_train, y_train)
    return model.predict(x_test), alpha


def _evaluate_fold(candidate: str, activity: np.ndarray, labels: np.ndarray, train_idx: np.ndarray,
                    test_idx: np.ndarray, held_in: np.ndarray, held_out: np.ndarray, region: str,
                    session: str, level: str, fold_id: int) -> dict:
    train_x = activity[train_idx][:, :, held_in]
    test_x = activity[test_idx][:, :, held_in]
    seed = stable_seed(f"human_cosmoothing_benchmark|{region}|{session}|{level}|{candidate}|fold{fold_id}|representation")
    if candidate == benchmark.DEMIXED_CANDIDATE:
        fit = benchmark.fit_demixed_principal_components(
            train_x, labels[train_idx], test_x, OPERATING_RANK, np.random.default_rng(seed)
        )
    else:
        fit = benchmark_core.fit_candidate(train_x, test_x, candidate, seed)
    if fit.get("status") != "fitted":
        return {"status": fit.get("status", "failed_to_train"), "reason": fit.get("reason")}

    target_train = activity[train_idx][:, :, held_out]
    target_test = activity[test_idx][:, :, held_out]
    latent_train, target_train = _align_bins(np.asarray(fit["latent_train"], dtype=float), target_train)
    latent_test, target_test = _align_bins(np.asarray(fit["latent_test"], dtype=float), target_test)
    if latent_train.shape[1] != latent_test.shape[1]:
        return {"status": "failed_to_score", "reason": "train and test latents aligned to different bin counts"}

    k_used = latent_train.shape[-1]
    scaler = StandardScaler().fit(latent_train.reshape(-1, k_used))
    x_train = scaler.transform(latent_train.reshape(-1, k_used))
    x_test = scaler.transform(latent_test.reshape(-1, k_used))
    n_bins = latent_train.shape[1]
    groups = np.repeat(np.arange(len(train_idx)), n_bins)
    n_held_out = len(held_out)

    pred_test = np.zeros((len(test_idx), n_bins, n_held_out))
    chosen_penalties = []
    try:
        for j in range(n_held_out):
            rate, alpha = fit_readout(x_train, target_train[:, :, j].reshape(-1), x_test, groups)
            pred_test[:, :, j] = rate.reshape(len(test_idx), n_bins)
            chosen_penalties.append(float(alpha))
    except Exception as exc:
        return {"status": "failed_to_score", "reason": str(exc)}

    null_test = np.broadcast_to(target_test.mean(axis=(0, 1), keepdims=True), target_test.shape)
    null_train = np.broadcast_to(target_train.mean(axis=(0, 1), keepdims=True), target_test.shape)
    return {
        "status": "computed", "k_used": int(k_used), "n_held_out_neurons": int(n_held_out),
        "n_held_in_neurons": int(len(held_in)), "held_out_neuron_indices": held_out.tolist(),
        "chosen_penalties": chosen_penalties, "n_bins": int(n_bins),
        "ll_model": poisson_log_likelihood(target_test, pred_test),
        "ll_null_test": poisson_log_likelihood(target_test, null_test),
        "ll_null_train": poisson_log_likelihood(target_test, null_train),
        "n_spikes": float(target_test.sum()),
    }


def evaluate_candidate(region: str, session: str, level: str, activity: np.ndarray, labels: np.ndarray,
                        folds, fold_neurons: list[tuple[np.ndarray, np.ndarray]], candidate: str,
                        identity: dict, checkpoint_dir: Path) -> dict:
    base = f"human_cosmoothing_benchmark|{region}|{session}|{level}|{candidate}"
    fold_results, checkpoint_hits = [], []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        held_in, held_out = fold_neurons[fold_id]

        def _fold(tr=train_idx, te=test_idx, hi=held_in, ho=held_out, f=fold_id):
            return _evaluate_fold(candidate, activity, labels, tr, te, hi, ho, region, session, level, f)

        record, hit = run_checkpointed(f"{base}|fold{fold_id}", _fold, identity, checkpoint_dir, CHECKPOINT_SCHEMA)
        fold_results.append(record)
        checkpoint_hits.append(hit)
        if record.get("status") != "computed":
            break

    failed = next((r for r in fold_results if r.get("status") != "computed"), None)
    if failed is not None:
        return {"status": failed.get("status", "failed_to_train"), "region": region, "session": session,
                "level": level, "candidate": candidate, "reason": failed.get("reason"),
                "checkpoint_hits": int(sum(checkpoint_hits))}

    n_spikes = sum(r["n_spikes"] for r in fold_results)
    if n_spikes == 0:
        return {"status": "excluded", "region": region, "session": session, "level": level,
                "candidate": candidate, "reason": "zero held-out spikes", "n_spikes": 0,
                "checkpoint_hits": int(sum(checkpoint_hits))}

    ll_model = sum(r["ll_model"] for r in fold_results)
    ll_null_test = sum(r["ll_null_test"] for r in fold_results)
    ll_null_train = sum(r["ll_null_train"] for r in fold_results)
    return {
        "status": "computed", "region": region, "session": session, "level": level, "candidate": candidate,
        "bits_per_spike": bits_per_spike(ll_model, ll_null_test, n_spikes),
        "bits_per_spike_training_null": bits_per_spike(ll_model, ll_null_train, n_spikes),
        "n_spikes": int(n_spikes), "n_folds": len(fold_results),
        "n_held_out_neurons": [r["n_held_out_neurons"] for r in fold_results],
        "n_held_in_neurons": [r["n_held_in_neurons"] for r in fold_results],
        "held_out_neuron_indices": [r["held_out_neuron_indices"] for r in fold_results],
        "k_used": [r["k_used"] for r in fold_results],
        "chosen_penalties": [r["chosen_penalties"] for r in fold_results],
        "checkpoint_hits": int(sum(checkpoint_hits)),
    }


def evaluate_region_level(region: str, session: str, level: str, activity: np.ndarray, labels: np.ndarray,
                           candidates: tuple[str, ...], checkpoint_dir: Path, runtime: dict) -> list[dict]:
    fold_seed = stable_seed(f"info_benchmark|seed0|{region}|{session}|{level}|folds")
    folds = _seeded_folds(labels, N_SPLITS, fold_seed)
    if folds is None:
        return [{"status": "excluded", "region": region, "session": session, "level": level,
                  "candidate": c, "reason": "fewer than two trials in one or more classes"} for c in candidates]

    n_neurons = activity.shape[2]
    fold_neurons = []
    for fold_id in range(len(folds)):
        neuron_seed = stable_seed(f"cosmoothing|{region}|{session}|{level}|fold{fold_id}|neurons")
        fold_neurons.append(neuron_holdout(n_neurons, neuron_seed))
    neuron_counts = [{"n_neurons": n_neurons, "n_held_in": len(hi), "n_held_out": len(ho)} for hi, ho in fold_neurons]
    if any(c["n_held_in"] < 2 for c in neuron_counts):
        return [{"status": "excluded", "region": region, "session": session, "level": level, "candidate": c,
                  "reason": "fewer than 2 held-in neurons in a fold", "neuron_counts": neuron_counts}
                for c in candidates]

    identity = block_identity(activity, labels, folds, runtime)
    return [evaluate_candidate(region, session, level, activity, labels, folds, fold_neurons, c, identity, checkpoint_dir)
            for c in candidates]


def process_entry(row: dict, level_name: str, target_load: int, candidates: tuple[str, ...],
                   regions: tuple[str, ...], checkpoint_dir: Path, runtime: dict) -> dict:
    session_key = f"{row['patient']}_{row['session']}"
    session_data = benchmark.load_session_for_load(row["path"], row["release"], target_load)
    if session_data is None:
        return {"session_key": session_key, "patient": row["patient"], "records": [], "session_records": [
            {"status": "not_loaded", "patient": row["patient"], "session": row["session"],
             "release": row["release"], "level": level_name},
        ]}
    session_records, records = [], []
    for region in regions:
        cell = benchmark.build_cell(session_data, region)
        if cell is None:
            session_records.append({"status": "insufficient_units", "patient": row["patient"], "session": row["session"],
                                      "release": row["release"], "level": level_name, "region": region})
            continue
        activity, labels = cell
        session_records.append({"status": "loaded", "patient": row["patient"], "session": row["session"],
                                  "release": row["release"], "level": level_name, "region": region,
                                  "n_trials": int(activity.shape[0]), "n_units": int(activity.shape[2])})
        records.extend(evaluate_region_level(
            region, session_key, level_name, activity, labels, candidates, checkpoint_dir, runtime
        ))
        print(f"human_cosmoothing_benchmark: {session_key} {region} {level_name} "
              f"{len(records)} candidate records", flush=True)
    return {"session_key": session_key, "patient": row["patient"], "records": records, "session_records": session_records}


def paired_summary(records: list[dict], patient_by_session: dict[str, str], rng: np.random.Generator) -> list[dict]:
    """Mirrors run_human_representation_benchmark.improvement_summary: same aggregation
    (per-session paired difference against a reference candidate, averaged within patient,
    patient-resampled bootstrap CI, sign-flip p, minimum detectable difference), but over
    (region, level) cells and the single bits_per_spike metric instead of (region, level,
    decoder) cells and the two temporal auc metrics."""
    index: dict[tuple, dict] = {}
    for r in records:
        if r.get("status") != "computed":
            continue
        index[(r["region"], r["session"], r["level"], r["candidate"])] = r

    cells = sorted({(region, level) for (region, _session, level, _candidate) in index})
    candidates = sorted({candidate for (*_x, candidate) in index})
    summaries = []
    for region, level in cells:
        for reference in REFERENCE_CANDIDATES:
            for candidate in candidates:
                if candidate == reference:
                    continue
                diffs, patients = [], []
                for (r, session, l, c), record in index.items():
                    if (r, l, c) != (region, level, candidate):
                        continue
                    ref_record = index.get((region, session, level, reference))
                    if ref_record is None:
                        continue
                    diffs.append(record["bits_per_spike"] - ref_record["bits_per_spike"])
                    patients.append(patient_by_session.get(session, session))
                if len(diffs) < 3:
                    summaries.append({
                        "corpus": region, "level": level, "metric": "bits_per_spike", "candidate": candidate,
                        "reference": reference, "status": "not_estimable", "n_sessions": len(diffs),
                    })
                    continue
                diffs_arr, patients_arr = np.asarray(diffs, dtype=float), np.asarray(patients)
                uniq = np.unique(patients_arr)
                patient_means = np.array([diffs_arr[patients_arr == p].mean() for p in uniq])
                mean_stat, lo, hi = bootstrap_ci(patient_means, np.mean, n_boot=5000, rng=rng)
                sign_flip = paired_sign_flip_test(patient_means, np.zeros_like(patient_means), alternative="two-sided", rng=rng)
                mdd = minimum_detectable_paired_difference(patient_means)
                summaries.append({
                    "corpus": region, "level": level, "metric": "bits_per_spike", "candidate": candidate,
                    "reference": reference, "status": "estimable", "n_sessions": int(len(diffs)),
                    "n_patients": int(len(uniq)), "mean_improvement": mean_stat,
                    "ci_95_patient_cluster_bootstrap": [lo, hi], "sign_flip_p": sign_flip["p_value"],
                    "mdd": mdd.get("mdd"), "improves_on_reference": int(lo > 0.0),
                })
    return summaries


def render_summary(records: list[dict], complete: bool) -> str:
    lines = ["# Human co-smoothing benchmark", "", f"status: {'complete' if complete else 'running'}", ""]
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.get("status")] = by_status.get(r.get("status"), 0) + 1
    lines.append("## record counts by status")
    for status, count in sorted(by_status.items()):
        lines.append(f"- {status}: {count}")
    computed = [r for r in records if r["status"] == "computed"]
    lines += ["", "## bits per spike by region, level, candidate"]
    for region, level, candidate in sorted({(r["region"], r["level"], r["candidate"]) for r in computed}):
        values = [r["bits_per_spike"] for r in computed
                  if (r["region"], r["level"], r["candidate"]) == (region, level, candidate)]
        lines.append(f"- {region}/{level}/{candidate}: n={len(values)} mean={np.mean(values):.4f}")
    return "\n".join(lines) + "\n"


def _init_worker(gpu_lock) -> None:
    """Binds the shared lock into the neural data transformer's fit so run_info_benchmark's
    own fit_candidate serialises its GPU calls across pool workers without changes to
    fit_candidate itself."""
    benchmark_core.REPRESENTATION_FITS = {
        **benchmark_core.REPRESENTATION_FITS,
        benchmark.NEURAL_DATA_TRANSFORMER_CANDIDATE: functools.partial(
            benchmark.fit_neural_data_transformer, gpu_lock=gpu_lock
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Held-out-neuron co-smoothing score for the human representation benchmark.")
    parser.add_argument("--levels", nargs="+", choices=[name for name, _ in LEVELS], default=[name for name, _ in LEVELS])
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser


def apply_smoke(args) -> None:
    if not args.smoke:
        return
    args.candidates = ["principal_components", benchmark.DEMIXED_CANDIDATE]
    args.workers = 1


def main() -> None:
    args = _parser().parse_args()
    apply_smoke(args)

    primary, supplement = benchmark.build_manifest()
    manifest = primary + supplement
    levels = args.levels[:1] if args.smoke else args.levels
    regions = REGIONS[:1] if args.smoke else REGIONS
    if args.smoke:
        manifest = manifest[:1]

    level_lookup = dict(LEVELS)
    runtime = implementation_identity()
    records: list[dict] = []
    session_manifest: list[dict] = []
    patient_by_session: dict[str, str] = {}
    started = time.time()

    def flush(complete: bool) -> None:
        artifact = {
            "version": CHECKPOINT_SCHEMA,
            "code_commit": git_commit(ROOT),
            "implementation": runtime,
            "status": "complete" if complete else "running",
            "scope": {"levels": args.levels, "candidates": args.candidates, "smoke": args.smoke},
            "sessions": session_manifest,
            "records": records,
            "paired_summary": paired_summary(records, patient_by_session, np.random.default_rng(1)) if complete else [],
            "wall_clock_s": float(time.time() - started),
        }
        atomic_write(args.output, canonical_json(artifact))
        atomic_write(args.summary, render_summary(records, complete))

    tasks = [(row, level_name, level_lookup[level_name]) for row in manifest for level_name in levels]
    max_workers = max(1, min(args.workers, len(tasks) or 1))
    mp_context = multiprocessing.get_context("fork")
    gpu_lock = mp_context.Lock()
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp_context, initializer=_init_worker, initargs=(gpu_lock,),
    ) as executor:
        futures = [
            executor.submit(process_entry, row, level_name, target_load, tuple(args.candidates), regions,
                             args.checkpoint_dir, runtime)
            for row, level_name, target_load in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            patient_by_session[result["session_key"]] = result["patient"]
            session_manifest.extend(result["session_records"])
            records.extend(result["records"])
            flush(complete=False)

    flush(complete=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
