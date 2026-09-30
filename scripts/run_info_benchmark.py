from __future__ import annotations

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    import os as _os

    _os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    _path = str(ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import (  # noqa: E402
    canonical_json,
    checkpoint_load,
    checkpoint_safe,
    checkpoint_store,
    git_commit,
    restore_checkpoint,
    sha256_file,
)
from statistics import stable_seed  # noqa: E402
from run_deviation_axis_structure import _macaque_bundles
from corpus_sessions import _watters_bundles
from info_decoding import CORPORA
from corpus_sessions import _reachable_sessions
from corpus_sessions import _macaque_session_bundle
from corpus_sessions import _observable_arrays
from spike_pipeline import BIN_MS
import run_state_space_estimation_admissibility as estimation  # noqa: E402
from info_decoding import CTG_STEP
from info_decoding import OPERATING_RANK
from info_decoding import _decoder_api, _seeded_folds, CANDIDATES, DECODERS, atomic_write, fold_signature  # noqa: E402
from info_decoding import score_null_draw, score_observed, summarize_scores  # noqa: E402
from info_decoding import render_summary  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "info_benchmark.json"
SUMMARY_PATH = ROOT / "INFO_BENCHMARK.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_info_benchmark"
CHECKPOINT_SCHEMA = "info_benchmark_v2"

IMPLEMENTATION_PATHS = (
    Path(__file__),
    ROOT / "src" / "info_decoding.py",
    ROOT / "src" / "project_config.py",
    ROOT / "src" / "statistics.py",
    ROOT / "scripts" / "run_state_space_estimation_admissibility.py",
    ROOT / "scripts" / "run_latent_model_comparison.py",
    ROOT / "scripts" / "run_state_space_estimation_robustness.py",
    ROOT / "scripts" / "run_dissociation_cross_preparation_test.py",
    ROOT / "scripts" / "fit_sequential_autoencoder_worker.py",
    ROOT / "config" / "project.json",
)

SINGLE_ITEM_CORPUS, MULTI_OBJECT_CORPUS = CORPORA
REPRESENTATION_FITS = estimation.CANDIDATES




def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _checkpoint_path(key: str, checkpoint_dir: Path | None = None) -> Path:
    checkpoint_dir = CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    stem = _safe_name(key)[:80]
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    return checkpoint_dir / f"{stem}_{digest}.json"


def load_checkpoint(key: str, identity: dict | None = None,
                     checkpoint_dir: Path | None = None, schema: str | None = None) -> dict | None:
    schema = CHECKPOINT_SCHEMA if schema is None else schema
    path = _checkpoint_path(key, checkpoint_dir)
    payload = checkpoint_load(path)
    if payload is None:
        return None
    if payload.get("schema") != schema or payload.get("complete") is not True:
        return None
    if payload.get("identity") != identity:
        return None
    if "record" not in payload:
        return None
    return restore_checkpoint(payload["record"])


def save_checkpoint(key: str, record: dict, identity: dict | None = None,
                     checkpoint_dir: Path | None = None, schema: str | None = None) -> None:
    checkpoint_dir = CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir
    schema = CHECKPOINT_SCHEMA if schema is None else schema
    path = _checkpoint_path(key, checkpoint_dir)
    payload = {
        "schema": schema,
        "complete": True,
        "identity": identity,
        "record": checkpoint_safe(record),
    }
    checkpoint_store(path, payload)


def run_checkpointed(key: str, fn, identity: dict | None = None,
                      checkpoint_dir: Path | None = None, schema: str | None = None) -> tuple[dict, bool]:
    cached = load_checkpoint(key, identity, checkpoint_dir, schema)
    if cached is not None:
        return cached, True
    record = fn()
    if record.get("status") in {"fitted", "computed"}:
        save_checkpoint(key, record, identity, checkpoint_dir, schema)
    return record, False


def _array_hash(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical_json(list(array.shape)).encode())
    if array.dtype.hasobject or array.dtype.kind in "US":
        digest.update(canonical_json(array.tolist()).encode())
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _package_identity(distribution: str, module: str) -> dict:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        spec = importlib.util.find_spec(module)
        origin = spec.origin if spec is not None else None
    except (ImportError, ModuleNotFoundError, ValueError):
        origin = None
    return {"version": version, "path": origin}


def _lfads_identity(executable: str | None) -> dict | None:
    if not executable:
        return None
    path = Path(executable).resolve()
    identity = {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    code = (
        "import importlib.metadata as m,json,platform,sys;"
        "names=('numpy','torch','lfads-torch');"
        "a={str(d.metadata.get('Name','')).lower().replace('_','-'):d.version for d in m.distributions()};"
        "v={n:a.get(n) for n in names};"
        "print(json.dumps({'python':platform.python_version(),'python_path':sys.executable,'packages':v}))"
    )
    try:
        result = subprocess.run(
            [str(path), "-c", code], capture_output=True, text=True, timeout=15, check=True
        )
        identity["runtime"] = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        identity["runtime_error"] = f"{type(exc).__name__}: {exc}"
    return identity


def _accelerator_identity() -> dict:
    torch = estimation.torch
    available = bool(torch.cuda.is_available())
    return {
        "device": str(estimation.DEVICE),
        "cuda": torch.version.cuda,
        "available": available,
        "name": torch.cuda.get_device_name(0) if available else None,
    }


def implementation_identity() -> dict:
    return {
        "python": platform.python_version(),
        "python_path": sys.executable,
        "packages": {
            distribution: _package_identity(distribution, module)
            for distribution, module in (
                ("numpy", "numpy"),
                ("scikit-learn", "sklearn"),
                ("elephant", "elephant"),
                ("neo", "neo"),
                ("quantities", "quantities"),
                ("cebra", "cebra"),
                ("tphate", "tphate"),
                ("torch", "torch"),
                ("lfads-torch", "lfads_torch"),
            )
        },
        "sources": {
            str(path.resolve()): sha256_file(path)
            for path in IMPLEMENTATION_PATHS
            if path.exists()
        },
        "accelerator": _accelerator_identity(),
        "lfads": _lfads_identity(estimation.LFADS_WORKER_PYTHON),
    }


def block_identity(
    activity: np.ndarray,
    labels: np.ndarray,
    folds,
    implementation: dict | None = None,
) -> dict:
    return {
        "activity": _array_hash(activity),
        "labels": _array_hash(labels),
        "folds": fold_signature(folds),
        "implementation": implementation or implementation_identity(),
    }




def load_single(root: Path, limit: int | None) -> list[dict]:
    if limit is None:
        return _macaque_bundles(root)[0]
    bundles = []
    for path in _reachable_sessions(root, limit):
        bundle = _macaque_session_bundle(path)
        if bundle is not None:
            bundles.append(bundle)
    return bundles


def load_multi(root: Path, limit: int | None) -> tuple[list[dict], dict]:
    loaded = []
    for session in iter_watters(root, bin_ms=BIN_MS):
        if session["status"] == "loaded":
            loaded.append(session)
            if limit is not None and len(loaded) >= limit:
                break
    if not loaded:
        return [], {}
    arrays = {}
    for session in loaded:
        values, _, usable = _observable_arrays(session["counts"], session)
        if values is not None:
            arrays[session["session"]] = {
                "arrays": values,
                "usable": usable,
                "session": session,
            }
    return _watters_bundles(arrays), arrays












def fit_candidate(
    train_activity: np.ndarray,
    test_activity: np.ndarray,
    candidate: str,
    seed: int,
) -> dict:
    try:
        fit = REPRESENTATION_FITS[candidate](
            train_activity,
            test_activity,
            OPERATING_RANK,
            np.random.default_rng(seed),
            True,
            BIN_MS,
        )
    except Exception as exc:
        return {"status": "failed_to_train", "reason": str(exc)}
    if fit.get("status") != "fitted":
        return {"status": fit.get("status", "failed_to_train"), "reason": fit.get("reason")}
    latent_train = np.asarray(fit["latent_train"], dtype=float)
    latent_test = np.asarray(fit["latent_test"], dtype=float)
    k_used = int(fit["k_used"])
    valid_shape = (
        latent_train.ndim == 3
        and latent_test.ndim == 3
        and latent_train.shape[0] == train_activity.shape[0]
        and latent_test.shape[0] == test_activity.shape[0]
        and latent_train.shape[1:] == latent_test.shape[1:]
        and latent_train.shape[1] > 0
        and latent_train.shape[2] == k_used
    )
    if not valid_shape:
        return {"status": "failed_to_train", "reason": "invalid latent shape"}
    if not np.isfinite(latent_train).all() or not np.isfinite(latent_test).all():
        return {"status": "failed_to_train", "reason": "non-finite latent values"}
    return {
        "status": "fitted",
        "candidate": candidate,
        "k_used": k_used,
        "latent_train": latent_train,
        "latent_test": latent_test,
    }






def evaluate_block(
    corpus: str,
    session: str,
    level: str,
    activity: np.ndarray,
    labels: np.ndarray,
    candidates: tuple[str, ...],
    decoders: tuple[str, ...],
    n_splits: int,
    n_perm: int,
    time_step: int,
    progress,
    implementation: dict | None = None,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
    schema: str | None = None,
) -> list[dict]:
    seed_tag = f"seed{seed}"
    fold_seed = stable_seed(f"info_benchmark|{seed_tag}|{corpus}|{session}|{level}|folds")
    folds = _seeded_folds(labels, n_splits, fold_seed)
    if folds is None:
        return [{
            "status": "excluded",
            "seed": seed,
            "corpus": corpus,
            "session": session,
            "level": level,
            "reason": "fewer than two trials in one or more classes",
        }]

    records = []
    identity = block_identity(activity, labels, folds, implementation)
    identity_key = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:16]
    print(
        "Workload: "
        f"{corpus}/{session}/{level}; {len(candidates)} representations; "
        f"{len(decoders)} decoders; {len(folds)} folds; {n_perm} null draws; "
        f"{activity.shape[1]} time bins",
        flush=True,
    )
    for candidate in candidates:
        base = (
            f"{seed_tag}|{corpus}|{session}|{level}|{candidate}|rank{OPERATING_RANK}"
            f"|identity{identity_key}"
        )
        fits = []
        fit_hits = []
        for fold_id, (train, test) in enumerate(folds):
            rep, hit = run_checkpointed(
                f"{base}|fold{fold_id}|representation",
                lambda c=candidate, i=fold_id, tr=train, te=test: fit_candidate(
                    activity[tr],
                    activity[te],
                    c,
                    stable_seed(f"info_benchmark|{base}|fold{i}|representation"),
                ),
                identity, checkpoint_dir, schema,
            )
            fits.append(rep)
            fit_hits.append(hit)
            progress(records + [{
                "status": rep.get("status", "failed_to_train"),
                "corpus": corpus,
                "session": session,
                "level": level,
                "candidate": candidate,
                "fold": fold_id,
            }])
            if rep.get("status") != "fitted":
                break
        failed = next((fit for fit in fits if fit.get("status") != "fitted"), None)
        if failed is not None:
            record = {
                "status": failed.get("status", "failed_to_train"),
                "corpus": corpus,
                "session": session,
                "level": level,
                "candidate": candidate,
                "reason": failed.get("reason"),
                "fit_checkpoint_hits": int(sum(fit_hits)),
            }
            records.append(record)
            progress(records)
            continue
        for decoder in decoders:
            score_base = f"{base}|step{time_step}|{decoder}"

            def _observed(d=decoder):
                try:
                    return score_observed(
                        fits,
                        labels,
                        d,
                        folds,
                        stable_seed(f"info_benchmark|{score_base}|observed"),
                        time_step,
                    )
                except Exception as exc:
                    return {"status": "failed_to_score", "reason": str(exc)}

            observed, observed_hit = run_checkpointed(
                f"{score_base}|observed", _observed, identity, checkpoint_dir, schema
            )
            if observed.get("status") != "computed":
                record = {
                    **observed,
                    "corpus": corpus,
                    "session": session,
                    "level": level,
                    "candidate": candidate,
                    "decoder": decoder,
                }
                records.append(record)
                progress(records)
                continue

            progress(records + [{
                "status": "running",
                "corpus": corpus,
                "session": session,
                "level": level,
                "candidate": candidate,
                "decoder": decoder,
                "completed_permutations": 0,
                "n_permutations": n_perm,
            }])

            null_draws = []
            null_hits = 0
            failed = None
            for draw in range(n_perm):
                shuffle_seed = stable_seed(
                    f"info_benchmark|{seed_tag}|{corpus}|{session}|{level}|null|{draw}"
                )
                decoder_seed = stable_seed(f"info_benchmark|{seed_tag}|{decoder}|null_decoder|{draw}")

                def _null(d=decoder, s=shuffle_seed, m=decoder_seed):
                    try:
                        return score_null_draw(fits, labels, d, folds, s, m, time_step)
                    except Exception as exc:
                        return {"status": "failed_to_score", "reason": str(exc)}

                null, hit = run_checkpointed(
                    f"{score_base}|null|{draw}", _null, identity, checkpoint_dir, schema
                )
                if null.get("status") != "computed":
                    failed = null
                    break
                null_draws.append(np.asarray(null["temporal_auc"], dtype=float))
                null_hits += int(hit)
                progress(records + [{
                    "status": "running",
                    "corpus": corpus,
                    "session": session,
                    "level": level,
                    "candidate": candidate,
                    "decoder": decoder,
                    "completed_permutations": draw + 1,
                    "n_permutations": n_perm,
                }])
            if failed is not None:
                record = {
                    **failed,
                    "corpus": corpus,
                    "session": session,
                    "level": level,
                    "candidate": candidate,
                    "decoder": decoder,
                    "completed_permutations": len(null_draws),
                }
                records.append(record)
                progress(records)
                continue

            record = {
                **summarize_scores(
                    np.asarray(observed["temporal_auc"], dtype=float),
                    np.stack(null_draws),
                    decoder,
                ),
                "corpus": corpus,
                "session": session,
                "level": level,
                "candidate": candidate,
                "n_trials": int(len(labels)),
                "n_time_bins": int(np.asarray(fits[0]["latent_train"])[:, ::time_step].shape[1]),
                "n_features": int(np.asarray(fits[0]["latent_train"]).shape[2]),
                "k_used": int(fits[0]["k_used"]),
                "n_splits": int(len(folds)),
                "fit_checkpoint_hits": int(sum(fit_hits)),
                "observed_checkpoint_hit": observed_hit,
                "null_checkpoint_hits": null_hits,
            }
            records.append(record)
            progress(records)
    for record in records:
        record["seed"] = seed
    return records




def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run matched information decoding across macaque corpora.")
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=("single_item", "multi_object"),
        default=["single_item", "multi_object"],
    )
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--decoders", nargs="+", choices=DECODERS, default=list(DECODERS))
    parser.add_argument("--single-item-sessions-limit", type=int)
    parser.add_argument("--multi-object-sessions-limit", type=int)
    parser.add_argument("--multi-object-levels-limit", type=int)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--time-step", type=int, default=CTG_STEP)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    return parser


def apply_smoke(args) -> None:
    if not args.smoke:
        return
    args.candidates = ["native_full_rank", "principal_components"]
    args.decoders = ["linear", "nonlinear"]
    args.single_item_sessions_limit = 1
    args.multi_object_sessions_limit = 1
    args.multi_object_levels_limit = 1
    args.n_splits = 2
    args.n_perm = 2
    args.time_step = max(args.time_step, 10)


def main() -> None:
    args = _parser().parse_args()
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = args.checkpoint_dir
    apply_smoke(args)
    if args.n_splits < 2 or args.n_perm < 1 or args.time_step < 1:
        raise SystemExit("--n-splits must be at least 2; --n-perm and --time-step must be positive")
    limits = (
        args.single_item_sessions_limit,
        args.multi_object_sessions_limit,
        args.multi_object_levels_limit,
    )
    if any(limit is not None and limit < 1 for limit in limits):
        raise SystemExit("session and level limits must be positive")
    started = time.time()
    root = data_root()
    runtime = implementation_identity()
    records: list[dict] = []

    def flush(current: list[dict], complete: bool = False, failed: bool = False) -> None:
        combined = records + current
        artifact = {
            "version": CHECKPOINT_SCHEMA,
            "code_commit": git_commit(ROOT),
            "implementation": runtime,
            "status": "failed" if failed else "complete" if complete else "running",
            "scope": {
                "corpora": args.corpora,
                "candidates": args.candidates,
                "decoders": args.decoders,
                "n_splits": args.n_splits,
                "n_permutations": args.n_perm,
                "time_step": args.time_step,
                "seed": args.seed,
                "single_item_sessions_limit": args.single_item_sessions_limit,
                "multi_object_sessions_limit": args.multi_object_sessions_limit,
                "multi_object_levels_limit": args.multi_object_levels_limit,
            },
            "records": combined,
            "wall_clock_s": float(time.time() - started),
        }
        atomic_write(args.output, canonical_json(artifact))
        atomic_write(args.summary, render_summary(combined, complete, args.time_step, failed))

    def fail_source(corpus: str, status: str, reason: str) -> None:
        records.append({
            "status": status,
            "corpus": corpus,
            "session": "",
            "level": "",
            "reason": reason,
        })
        flush([], failed=True)
        raise SystemExit(reason)

    if "single_item" in args.corpora:
        try:
            single = load_single(root, args.single_item_sessions_limit)
        except Exception as exc:
            fail_source(SINGLE_ITEM_CORPUS, "source_failed", str(exc))
        if not single:
            fail_source(
                SINGLE_ITEM_CORPUS,
                "source_empty",
                "no single-item session bundles were loaded",
            )
        for bundle in single:
            current = evaluate_block(
                SINGLE_ITEM_CORPUS,
                bundle["session"],
                "all",
                np.asarray(bundle["counts"], dtype=float).transpose(0, 2, 1),
                np.asarray(bundle["memorandum_label"]),
                tuple(args.candidates),
                tuple(args.decoders),
                args.n_splits,
                args.n_perm,
                args.time_step,
                lambda partial: flush(partial),
                runtime,
                args.seed,
            )
            records.extend(current)
            flush([])

    if "multi_object" in args.corpora:
        try:
            multi, arrays_by_session = load_multi(root, args.multi_object_sessions_limit)
        except SystemExit:
            raise
        except Exception as exc:
            fail_source(MULTI_OBJECT_CORPUS, "source_failed", str(exc))
        if not multi:
            fail_source(
                MULTI_OBJECT_CORPUS,
                "source_empty",
                "no multi-object sessions passed loading and preparation",
            )
        for bundle in multi:
            entry = arrays_by_session[bundle["session"]]
            usable = np.asarray(entry["usable"], dtype=bool)
            counts = np.asarray(entry["session"]["counts"], dtype=float)[usable]
            item_count = np.asarray(bundle["item_count"])
            levels = sorted(np.unique(item_count).tolist())
            if args.multi_object_levels_limit is not None:
                levels = levels[: args.multi_object_levels_limit]
            for level in levels:
                mask = item_count == level
                current = evaluate_block(
                    MULTI_OBJECT_CORPUS,
                    bundle["session"],
                    str(int(level)),
                    counts[mask].transpose(0, 2, 1),
                    np.asarray(bundle["memorandum_label"])[mask],
                    tuple(args.candidates),
                    tuple(args.decoders),
                    args.n_splits,
                    args.n_perm,
                    args.time_step,
                    lambda partial: flush(partial),
                    runtime,
                    args.seed,
                )
                records.extend(current)
                flush([])
    flush([], complete=True)


if __name__ == "__main__":
    main()
