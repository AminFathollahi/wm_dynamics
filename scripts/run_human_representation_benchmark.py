"""Matched information benchmark for human hippocampal/amygdala single units.

Reuses the macaque benchmark's decoder, folding, scoring and checkpoint
machinery (run_info_benchmark.py) unchanged, adds one supervised candidate
(ridge demixed principal components) that cannot go through that path because
it needs the training labels, and loads sessions through the same corpus
identity, label and window primitives the selective-persistence producer
already builds.

Run:
    conda run -n wm_dynamics python scripts/run_human_representation_benchmark.py
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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    _path = str(ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_info_benchmark as benchmark_core  # noqa: E402
from run_info_benchmark import block_identity, evaluate_block, implementation_identity, run_checkpointed
from info_decoding import render_summary
from info_decoding import score_null_draw, score_observed, summarize_scores
from info_decoding import CANDIDATES as NATIVE_CANDIDATES, DECODERS, atomic_write, fold_signature, _seeded_folds
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed,
)
from spike_pipeline import MIN_SESSION_ACCURACY, load_spike_times, resolve_unit_regions  # noqa: E402
from corpus_sessions import EPOCH_WINDOWS_S, MIN_TRIALS, region_filtered_units  # noqa: E402
from spike_pipeline import build_psth  # noqa: E402
from project_config import data_root, executable
from run_human_drift_spine_001187_000673 import canonical_sessions
from corpus_sessions import _trial_group
from run_region_resolved_rate_stability_behaviour import CORPUS_SPECS as REGION_CORPUS_SPECS  # noqa: E402
from run_selective_persistence_and_attractor_geometry import _canonical_patient, _dpca_ridge_axes, _fit_axes_arm_a
from info_decoding import CATEGORIES_100_DIVIDED, CATEGORIES_469, DPCA_RIDGE_LAMBDA_GRID, N_CV_FOLDS_LAMBDA, _category_469, _category_divided
from info_decoding import anscombe_counts
from info_decoding import OPERATING_RANK
from spike_pipeline import BIN_MS
from info_decoding import CTG_STEP

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "human_representation_benchmark.json"
SUMMARY_PATH = ROOT / "HUMAN_REPRESENTATION_BENCHMARK.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_human_representation_benchmark"
CHECKPOINT_SCHEMA = "human_representation_benchmark_v1"

REGIONS = ("hippocampus", "amygdala")
DEMIXED_CANDIDATE = "demixed_principal_components"
RECURRENT_SWITCHING_CANDIDATE = "recurrent_switching_linear_dynamics"
NEURAL_DATA_TRANSFORMER_CANDIDATE = "neural_data_transformer"
NEW_MODEL_CANDIDATES = (RECURRENT_SWITCHING_CANDIDATE, NEURAL_DATA_TRANSFORMER_CANDIDATE)
CANDIDATES = tuple(NATIVE_CANDIDATES) + (DEMIXED_CANDIDATE,) + NEW_MODEL_CANDIDATES
REFERENCE_CANDIDATES = ("principal_components", DEMIXED_CANDIDATE)
LEVELS = (("load1_maintenance", 1), ("load3_first_item_maintenance", 3))

SSM_WORKER_PYTHON = executable("ssm_python")
SSM_WORKER_SCRIPT = ROOT / "scripts" / "fit_recurrent_switching_linear_dynamics_worker.py"
SSM_SUBPROCESS_TIMEOUT_S = 1800
NDT_WORKER_PYTHON = executable("ndt_python")
NDT_WORKER_SCRIPT = ROOT / "scripts" / "fit_neural_data_transformer_worker.py"
NDT_SUBPROCESS_TIMEOUT_S = 1800


INFERENCE_COUNT_KEYS = ("inference_retried_bins", "inference_carried_bins")


def _run_worker_bridge(python: str | None, script: Path, train_X: np.ndarray, test_X: np.ndarray,
                        k: int, seed: int, timeout_s: float, causal: bool = False,
                        gpu: bool = False, causal_bins=None, gpu_lock=None) -> dict:
    """Shared subprocess bridge: temporary npz in/out, timeout, nonzero exit or timeout
    reported as failed_to_train, never raised -- matches fit_sequential_autoencoder's bridge.
    causal_bins (0-indexed, optional) restricts causal-mode truncation lengths to the bins
    that will actually be scored. gpu_lock, if given, is held for the whole subprocess call
    when gpu=True -- used by the pilot's process pool to serialise transformer fits."""
    if not python:
        return {"status": "failed_to_train", "reason": f"configure an executable for {script.name}"}
    with tempfile.TemporaryDirectory(prefix=f"{script.stem}_bridge_") as tmp_dir:
        in_path = Path(tmp_dir) / "input.npz"
        out_path = Path(tmp_dir) / "output.npz"
        payload = dict(train_X=train_X.astype(np.float32), test_X=test_X.astype(np.float32),
                        k=np.int64(k), seed=np.int64(seed), causal=np.bool_(causal))
        if causal_bins is not None:
            payload["bins"] = np.asarray(causal_bins, dtype=np.int64)
        np.savez(in_path, **payload)
        import os
        env = dict(os.environ)
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""

        def _call():
            return subprocess.run(
                [python, str(script), str(in_path), str(out_path)],
                capture_output=True, text=True, timeout=timeout_s, env=env,
            )

        try:
            proc = _call() if not (gpu and gpu_lock is not None) else _locked_call(gpu_lock, _call)
        except subprocess.TimeoutExpired:
            return {"status": "failed_to_train", "reason": f"{script.name} worker timed out after {timeout_s}s"}
        if proc.returncode != 0 or not out_path.exists():
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            return {"status": "failed_to_train", "reason": f"{script.name} worker exited {proc.returncode}: {stderr_tail}"}
        result = np.load(out_path, allow_pickle=True)
        if "reason" in result.files:
            return {"status": "failed_to_train", "reason": str(np.asarray(result["reason"]).item())}
        counts = {name: int(result[name]) for name in INFERENCE_COUNT_KEYS if name in result.files}
        return {"status": "fitted", "k_used": int(result["k_used"]),
                "latent_train": result["latent_train"], "latent_test": result["latent_test"], **counts}


def _locked_call(gpu_lock, call):
    with gpu_lock:
        return call()


def fit_recurrent_switching_linear_dynamics(train_X, test_X, k, rng, is_spiking, bin_ms, causal=False,
                                             causal_bins=None, gpu_lock=None):
    if not is_spiking:
        return {"status": "not_applicable_to_data_type"}
    seed = int(rng.integers(0, 2**31 - 1))
    return _run_worker_bridge(SSM_WORKER_PYTHON, SSM_WORKER_SCRIPT, train_X, test_X, k, seed,
                               SSM_SUBPROCESS_TIMEOUT_S, causal=causal, gpu=False, causal_bins=causal_bins)


def fit_neural_data_transformer(train_X, test_X, k, rng, is_spiking, bin_ms, causal=False,
                                 causal_bins=None, gpu_lock=None):
    if not is_spiking:
        return {"status": "not_applicable_to_data_type"}
    seed = int(rng.integers(0, 2**31 - 1))
    return _run_worker_bridge(NDT_WORKER_PYTHON, NDT_WORKER_SCRIPT, train_X, test_X, k, seed,
                               NDT_SUBPROCESS_TIMEOUT_S, causal=causal, gpu=True, gpu_lock=gpu_lock)


NEW_MODEL_FITS = {
    RECURRENT_SWITCHING_CANDIDATE: fit_recurrent_switching_linear_dynamics,
    NEURAL_DATA_TRANSFORMER_CANDIDATE: fit_neural_data_transformer,
}

# Route the two new candidates through run_info_benchmark's own evaluate_block/fit_candidate --
# the same path every other native candidate takes -- so they get identical folds, seeds and
# checkpointed records. Rebinding the module-level dict (never mutating estimation.CANDIDATES in
# place) keeps run_state_space_estimation_admissibility.py's own candidate roster untouched.
benchmark_core.REPRESENTATION_FITS = {**benchmark_core.REPRESENTATION_FITS, **NEW_MODEL_FITS}

_PIC_FIELD = {"000469": "loadsEnc1_PicIDs", "000673": "PicIDs_Encoding1", "001187": "PicIDs_Encoding1"}
_CATEGORY_FN = {"000469": _category_469, "000673": _category_divided, "001187": _category_divided}
_CATEGORY_BASIS = {"000469": CATEGORIES_469, "000673": CATEGORIES_100_DIVIDED, "001187": CATEGORIES_100_DIVIDED}


def build_manifest() -> tuple[list[dict], list[dict]]:
    canonical = canonical_sessions()
    primary = [
        {"release": row["primary_release"], "patient": row["patient"], "session": row["session"],
         "path": data_root() / row["primary_path"]}
        for row in canonical
    ]
    used_patients = {row["patient"] for row in canonical}
    supplement = []
    for path in REGION_CORPUS_SPECS["dandi_000469"]["glob"]():
        patient = _canonical_patient("000469", path)
        if patient is None or patient in used_patients:
            continue
        supplement.append({"release": "000469", "patient": patient, "session": path.stem, "path": path})
        used_patients.add(patient)
    return primary, supplement


def load_session_for_load(path: Path, release: str, target_load: int) -> dict | None:
    patient = _canonical_patient(release, path)
    if patient is None:
        return None
    trial_key = "WM_trials" if release == "001187" else "trials"
    with h5py.File(str(path), "r") as f:
        if "units" not in f or "intervals" not in f or trial_key not in f["intervals"]:
            return None
        trials = _trial_group(f, release)
        loads = trials["loads"][:].astype(int)
        accuracy_all = trials["response_accuracy"][:].astype(bool)
        if len(loads) < MIN_TRIALS or accuracy_all.mean() < MIN_SESSION_ACCURACY:
            return None
        keep = loads == target_load
        if keep.sum() < MIN_TRIALS:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        pic_ids = trials[_PIC_FIELD[release]][:][keep]
        t_maint = trials["timestamps_Maintenance"][:][keep]
    return dict(
        patient=patient, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
        category=_CATEGORY_FN[release](pic_ids), category_basis=_CATEGORY_BASIS[release],
        t_maint=t_maint, maint_win=EPOCH_WINDOWS_S["delay"],
    )


def build_cell(session: dict, region: str) -> tuple[np.ndarray, np.ndarray] | None:
    spike_lists = region_filtered_units(
        session["spike_lists_all"], session["unit_regions"], region, session["t_maint"], session["maint_win"]
    )
    if spike_lists is None:
        return None
    counts = build_psth(
        spike_lists, session["t_maint"], bin_ms=BIN_MS, smooth_ms=0.0, window_s=session["maint_win"]
    ) * (BIN_MS / 1000.0)
    activity = np.asarray(counts, dtype=float).transpose(0, 2, 1)
    return activity, np.asarray(session["category"])


def fit_demixed_principal_components(
    train_activity: np.ndarray, train_labels: np.ndarray, test_activity: np.ndarray,
    rank: int, rng: np.random.Generator,
) -> dict:
    try:
        if len(np.unique(train_labels)) < 2:
            return {"status": "failed_to_train", "reason": "fewer than two classes in training fold"}
        train_t = anscombe_counts(train_activity)
        test_t = anscombe_counts(test_activity)
        fit = _fit_axes_arm_a(
            train_t, train_labels, rng, rank, grid=DPCA_RIDGE_LAMBDA_GRID, fit_fn=_dpca_ridge_axes
        )
        axes = fit["V"]
        k_used = int(axes.shape[1])
        if k_used < 1:
            return {"status": "failed_to_train", "reason": "no ridge dpca axes recovered"}
        mu = train_t.reshape(-1, train_t.shape[-1]).mean(axis=0)
        latent_train = (train_t - mu) @ axes
        latent_test = (test_t - mu) @ axes
    except Exception as exc:
        return {"status": "failed_to_train", "reason": str(exc)}
    if not np.isfinite(latent_train).all() or not np.isfinite(latent_test).all():
        return {"status": "failed_to_train", "reason": "non-finite latent values"}
    return {
        "status": "fitted", "candidate": DEMIXED_CANDIDATE, "k_used": k_used,
        "latent_train": latent_train, "latent_test": latent_test,
    }


def evaluate_demixed_block(
    corpus: str, session: str, level: str, activity: np.ndarray, labels: np.ndarray,
    decoders: tuple[str, ...], n_splits: int, n_perm: int, time_step: int, progress,
    implementation: dict, seed: int,
) -> list[dict]:
    seed_tag = f"seed{seed}"
    fold_seed = stable_seed(f"human_representation_benchmark|{seed_tag}|{corpus}|{session}|{level}|folds")
    folds = _seeded_folds(labels, n_splits, fold_seed)
    if folds is None:
        return [{
            "status": "excluded", "seed": seed, "corpus": corpus, "session": session, "level": level,
            "candidate": DEMIXED_CANDIDATE, "reason": "fewer than two trials in one or more classes",
        }]

    identity = block_identity(activity, labels, folds, implementation)
    identity_key = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]
    base = (
        f"{seed_tag}|{corpus}|{session}|{level}|{DEMIXED_CANDIDATE}|rank{OPERATING_RANK}"
        f"|identity{identity_key}"
    )

    fits, fit_hits = [], []
    for fold_id, (train, test) in enumerate(folds):
        rep, hit = run_checkpointed(
            f"{base}|fold{fold_id}|representation",
            lambda tr=train, te=test, i=fold_id: fit_demixed_principal_components(
                activity[tr], labels[tr], activity[te], OPERATING_RANK,
                np.random.default_rng(
                    stable_seed(f"human_representation_benchmark|{base}|fold{i}|representation")
                ),
            ),
            identity, CHECKPOINT_DIR, CHECKPOINT_SCHEMA,
        )
        fits.append(rep)
        fit_hits.append(hit)
        progress([{
            "status": rep.get("status", "failed_to_train"), "corpus": corpus, "session": session,
            "level": level, "candidate": DEMIXED_CANDIDATE, "fold": fold_id,
        }])
        if rep.get("status") != "fitted":
            break

    failed = next((f for f in fits if f.get("status") != "fitted"), None)
    if failed is not None:
        record = {
            "status": failed.get("status", "failed_to_train"), "corpus": corpus, "session": session,
            "level": level, "candidate": DEMIXED_CANDIDATE, "reason": failed.get("reason"),
            "fit_checkpoint_hits": int(sum(fit_hits)), "seed": seed,
        }
        progress([record])
        return [record]

    records = []
    for decoder in decoders:
        score_base = f"{base}|step{time_step}|{decoder}"

        def _observed(d=decoder):
            try:
                return score_observed(
                    fits, labels, d, folds,
                    stable_seed(f"human_representation_benchmark|{score_base}|observed"), time_step,
                )
            except Exception as exc:
                return {"status": "failed_to_score", "reason": str(exc)}

        observed, observed_hit = run_checkpointed(
            f"{score_base}|observed", _observed, identity, CHECKPOINT_DIR, CHECKPOINT_SCHEMA,
        )
        if observed.get("status") != "computed":
            record = {
                **observed, "corpus": corpus, "session": session, "level": level,
                "candidate": DEMIXED_CANDIDATE, "decoder": decoder, "seed": seed,
            }
            records.append(record)
            progress([record])
            continue

        null_draws, null_hits, fail = [], 0, None
        for draw in range(n_perm):
            shuffle_seed = stable_seed(
                f"human_representation_benchmark|{seed_tag}|{corpus}|{session}|{level}|null|{draw}"
            )
            decoder_seed = stable_seed(
                f"human_representation_benchmark|{seed_tag}|{decoder}|null_decoder|{draw}"
            )

            def _null(d=decoder, s=shuffle_seed, m=decoder_seed):
                try:
                    return score_null_draw(fits, labels, d, folds, s, m, time_step)
                except Exception as exc:
                    return {"status": "failed_to_score", "reason": str(exc)}

            null, hit = run_checkpointed(
                f"{score_base}|null|{draw}", _null, identity, CHECKPOINT_DIR, CHECKPOINT_SCHEMA,
            )
            if null.get("status") != "computed":
                fail = null
                break
            null_draws.append(np.asarray(null["temporal_auc"], dtype=float))
            null_hits += int(hit)
            progress([{
                "status": "running", "corpus": corpus, "session": session, "level": level,
                "candidate": DEMIXED_CANDIDATE, "decoder": decoder,
                "completed_permutations": draw + 1, "n_permutations": n_perm,
            }])

        if fail is not None:
            record = {
                **fail, "corpus": corpus, "session": session, "level": level,
                "candidate": DEMIXED_CANDIDATE, "decoder": decoder,
                "completed_permutations": len(null_draws), "seed": seed,
            }
            records.append(record)
            progress([record])
            continue

        record = {
            **summarize_scores(
                np.asarray(observed["temporal_auc"], dtype=float), np.stack(null_draws), decoder
            ),
            "corpus": corpus, "session": session, "level": level, "candidate": DEMIXED_CANDIDATE,
            "n_trials": int(len(labels)),
            "n_time_bins": int(np.asarray(fits[0]["latent_train"])[:, ::time_step].shape[1]),
            "n_features": int(np.asarray(fits[0]["latent_train"]).shape[2]), "k_used": int(fits[0]["k_used"]),
            "n_splits": int(len(folds)), "fit_checkpoint_hits": int(sum(fit_hits)),
            "observed_checkpoint_hit": observed_hit, "null_checkpoint_hits": null_hits, "seed": seed,
        }
        records.append(record)
        progress([record])
    return records


def improvement_summary(records: list[dict], patient_by_session: dict[str, str], rng: np.random.Generator) -> list[dict]:
    index: dict[tuple, dict] = {}
    for r in records:
        if r.get("status") != "computed":
            continue
        index[(r["corpus"], r["session"], r["level"], r["decoder"], r["candidate"])] = r

    cells = sorted({(c, l, d) for (c, _, l, d, _cd) in index})
    candidates = sorted({cd for (*_x, cd) in index})
    summaries = []
    for corpus, level, decoder in cells:
        for reference in REFERENCE_CANDIDATES:
            for candidate in candidates:
                if candidate == reference:
                    continue
                for metric in ("same_time", "cross_temporal"):
                    diffs, patients = [], []
                    for (c, session, l, d, cd), record in index.items():
                        if (c, l, d, cd) != (corpus, level, decoder, candidate):
                            continue
                        ref_record = index.get((corpus, session, level, decoder, reference))
                        if ref_record is None:
                            continue
                        a = record.get(metric, {}).get("auc_above_null")
                        b = ref_record.get(metric, {}).get("auc_above_null")
                        if a is None or b is None:
                            continue
                        diffs.append(a - b)
                        patients.append(patient_by_session.get(session, session))
                    if len(diffs) < 3:
                        summaries.append({
                            "corpus": corpus, "level": level, "decoder": decoder, "metric": metric,
                            "candidate": candidate, "reference": reference,
                            "status": "not_estimable", "n_sessions": len(diffs),
                        })
                        continue
                    diffs_arr = np.asarray(diffs, dtype=float)
                    patients_arr = np.asarray(patients)
                    uniq = np.unique(patients_arr)
                    patient_means = np.array([diffs_arr[patients_arr == p].mean() for p in uniq])
                    mean_stat, lo, hi = bootstrap_ci(patient_means, np.mean, n_boot=5000, rng=rng)
                    sign_flip = paired_sign_flip_test(
                        patient_means, np.zeros_like(patient_means), alternative="two-sided", rng=rng
                    )
                    mdd = minimum_detectable_paired_difference(patient_means)
                    summaries.append({
                        "corpus": corpus, "level": level, "decoder": decoder, "metric": metric,
                        "candidate": candidate, "reference": reference, "status": "estimable",
                        "n_sessions": int(len(diffs)), "n_patients": int(len(uniq)),
                        "mean_improvement": mean_stat,
                        "ci_95_patient_cluster_bootstrap": [lo, hi],
                        "sign_flip_p": sign_flip["p_value"],
                        "mdd": mdd.get("mdd"),
                        "improves_on_reference": int(lo > 0.0),
                    })
    return summaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched information benchmark for human single units.")
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--decoders", nargs="+", choices=DECODERS, default=list(DECODERS))
    parser.add_argument("--levels", nargs="+", choices=[l for l, _ in LEVELS], default=[l for l, _ in LEVELS])
    parser.add_argument("--include-000469", dest="include_000469", action="store_true", default=True)
    parser.add_argument("--no-include-000469", dest="include_000469", action="store_false")
    parser.add_argument("--sessions-limit", type=int)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--time-step", type=int, default=CTG_STEP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--cost-probe", action="store_true")
    parser.add_argument("--cost-probe-sessions", type=int, default=2)
    parser.add_argument("--cost-probe-output", type=Path, default=RESULTS / "human_representation_benchmark_cost_probe.json")
    return parser


def apply_smoke(args) -> None:
    if not args.smoke:
        return
    args.candidates = ["native_full_rank", "principal_components", DEMIXED_CANDIDATE]
    args.decoders = ["linear"]
    args.sessions_limit = 1
    args.n_splits = 2
    args.n_perm = 2
    args.include_000469 = False


def run_cost_probe(manifest: list[dict], args) -> dict:
    rows = manifest[: args.cost_probe_sessions]
    timings = []
    for row in rows:
        session_data = load_session_for_load(row["path"], row["release"], 1)
        if session_data is None:
            continue
        for region in args.regions:
            cell = build_cell(session_data, region)
            if cell is None:
                continue
            activity, labels = cell
            if len(np.unique(labels)) < 2:
                continue
            rng_split = np.random.default_rng(0)
            n = len(labels)
            test_size = max(2, int(round(n * 0.3)))
            test_idx = rng_split.choice(n, size=test_size, replace=False)
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            for candidate in args.candidates:
                started = time.time()
                if candidate == DEMIXED_CANDIDATE:
                    fit = fit_demixed_principal_components(
                        activity[train_idx], labels[train_idx], activity[test_idx],
                        OPERATING_RANK, np.random.default_rng(0),
                    )
                else:
                    fit = benchmark_core.fit_candidate(
                        activity[train_idx], activity[test_idx], candidate, seed=0,
                    )
                elapsed = time.time() - started
                timings.append({
                    "session": f"{row['patient']}_{row['session']}", "region": region,
                    "candidate": candidate, "status": fit.get("status"), "seconds": elapsed,
                    "n_trials": n, "n_units": int(activity.shape[2]), "n_time_bins": int(activity.shape[1]),
                })
                print(f"cost probe: {timings[-1]}", flush=True)
    return {"timings": timings}


def main() -> None:
    args = _parser().parse_args()
    apply_smoke(args)
    if args.n_splits < 2 or args.n_perm < 1 or args.time_step < 1:
        raise SystemExit("--n-splits must be at least 2; --n-perm and --time-step must be positive")

    primary, supplement = build_manifest()
    manifest = primary + (supplement if args.include_000469 else [])
    if args.sessions_limit is not None:
        manifest = manifest[: args.sessions_limit]

    if args.cost_probe:
        probe = run_cost_probe(manifest, args)
        atomic_write(args.cost_probe_output, canonical_json(probe))
        return

    global CHECKPOINT_DIR
    CHECKPOINT_DIR = args.checkpoint_dir
    native_candidates = tuple(c for c in args.candidates if c != DEMIXED_CANDIDATE)
    run_demixed = DEMIXED_CANDIDATE in args.candidates

    started = time.time()
    runtime = implementation_identity()
    records: list[dict] = []
    patient_by_session: dict[str, str] = {}
    session_manifest: list[dict] = []

    def flush(current: list[dict], complete: bool = False) -> None:
        combined = records + current
        artifact = {
            "version": CHECKPOINT_SCHEMA,
            "code_commit": git_commit(ROOT),
            "implementation": runtime,
            "status": "complete" if complete else "running",
            "scope": {
                "regions": args.regions, "levels": args.levels, "candidates": args.candidates,
                "decoders": args.decoders, "n_splits": args.n_splits, "n_permutations": args.n_perm,
                "time_step": args.time_step, "seed": args.seed, "include_000469": args.include_000469,
                "sessions_limit": args.sessions_limit,
            },
            "sessions": session_manifest,
            "records": combined,
            "improvement_summary": improvement_summary(combined, patient_by_session, np.random.default_rng(1))
            if complete else [],
            "wall_clock_s": float(time.time() - started),
        }
        atomic_write(args.output, canonical_json(artifact))
        atomic_write(args.summary, render_summary(combined, complete, args.time_step))

    level_lookup = dict(LEVELS)
    for row in manifest:
        for level_name in args.levels:
            target_load = level_lookup[level_name]
            session_data = load_session_for_load(row["path"], row["release"], target_load)
            session_key = f"{row['patient']}_{row['session']}"
            patient_by_session[session_key] = row["patient"]
            if session_data is None:
                session_manifest.append({
                    "patient": row["patient"], "session": row["session"], "release": row["release"],
                    "level": level_name, "status": "not_loaded",
                })
                continue
            for region in args.regions:
                cell = build_cell(session_data, region)
                if cell is None:
                    session_manifest.append({
                        "patient": row["patient"], "session": row["session"], "release": row["release"],
                        "level": level_name, "region": region, "status": "insufficient_units",
                    })
                    continue
                activity, labels = cell
                session_manifest.append({
                    "patient": row["patient"], "session": row["session"], "release": row["release"],
                    "level": level_name, "region": region, "status": "loaded",
                    "n_trials": int(activity.shape[0]), "n_units": int(activity.shape[2]),
                })
                if native_candidates:
                    current = evaluate_block(
                        region, session_key, level_name, activity, labels, native_candidates,
                        tuple(args.decoders), args.n_splits, args.n_perm, args.time_step,
                        lambda partial: flush(partial), runtime, args.seed,
                        CHECKPOINT_DIR, CHECKPOINT_SCHEMA,
                    )
                    records.extend(current)
                    flush([])
                if run_demixed:
                    current = evaluate_demixed_block(
                        region, session_key, level_name, activity, labels, tuple(args.decoders),
                        args.n_splits, args.n_perm, args.time_step, lambda partial: flush(partial),
                        runtime, args.seed,
                    )
                    records.extend(current)
                    flush([])
    flush([], complete=True)


if __name__ == "__main__":
    main()
