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
    checkpoint_safe,
    git_commit,
    restore_checkpoint,
    sha256_file,
)
from statistics import stable_seed  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA,
    _macaque_bundles,
    _watters_bundles,
)
from run_behavior_amplitude_rate_controls import _reachable_sessions  # noqa: E402
from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    _macaque_session_bundle,
)
from run_dissociation_replication_and_counting_noise import _observable_arrays  # noqa: E402
from run_dissociation_cross_preparation_test import BIN_MS  # noqa: E402
import run_state_space_estimation_admissibility as estimation  # noqa: E402
from run_state_space_dimensionality_sweep import CTG_STEP  # noqa: E402
from run_state_space_estimation_robustness import OPERATING_RANK  # noqa: E402

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

CANDIDATES = (
    "native_full_rank",
    "principal_components",
    "factor_analysis",
    "gaussian_process_factor_analysis",
    "time_contrastive_embedding",
    "temporal_diffusion_embedding",
    "sequential_autoencoder",
)
DECODERS = ("linear", "nonlinear")
SINGLE_ITEM_CORPUS, MULTI_OBJECT_CORPUS = CORPORA
REPRESENTATION_FITS = estimation.CANDIDATES


def _decoder_api() -> SimpleNamespace:
    from info_decoding import _stratified_permutation, make_folds, split_auc

    return SimpleNamespace(
        make_folds=make_folds,
        split_auc=split_auc,
        stratified_permutation=_stratified_permutation,
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _checkpoint_path(key: str) -> Path:
    stem = _safe_name(key)[:80]
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    return CHECKPOINT_DIR / f"{stem}_{digest}.json"


def load_checkpoint(key: str, identity: dict | None = None) -> dict | None:
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("complete") is not True:
        return None
    if payload.get("identity") != identity:
        return None
    if "record" not in payload:
        return None
    return restore_checkpoint(payload["record"])


def save_checkpoint(key: str, record: dict, identity: dict | None = None) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(key)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "complete": True,
        "identity": identity,
        "record": checkpoint_safe(record),
    }
    fd, tmp_name = tempfile.mkstemp(dir=CHECKPOINT_DIR, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(key: str, fn, identity: dict | None = None) -> tuple[dict, bool]:
    cached = load_checkpoint(key, identity)
    if cached is not None:
        return cached, True
    record = fn()
    if record.get("status") in {"fitted", "computed"}:
        save_checkpoint(key, record, identity)
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


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


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


def _mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _metric(observed: np.ndarray, null: np.ndarray, mask: np.ndarray | None = None) -> dict:
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    if mask is not None:
        observed = observed[mask]
        null = null[:, mask]
    score = _mean(observed)
    null_scores = np.asarray([_mean(draw) for draw in null], dtype=object)
    null_scores = np.asarray([value for value in null_scores if value is not None], dtype=float)
    if score is None or not null_scores.size:
        return {"status": "not_computable"}
    null_mean = float(null_scores.mean())
    p_value = float((1 + np.sum(null_scores >= score)) / (null_scores.size + 1))
    return {
        "status": "computed",
        "auc": score,
        "null_auc": null_mean,
        "auc_above_null": float(score - null_mean),
        "p_value": p_value,
        "n_permutations": int(null_scores.size),
    }


def score_observed(
    fits: list[dict],
    labels: np.ndarray,
    decoder: str,
    folds,
    seed: int,
    time_step: int = 1,
) -> dict:
    api = _decoder_api()
    temporal_folds = []
    for fold_id, (train, test) in enumerate(folds):
        fit = fits[fold_id]
        latent_train = np.asarray(fit["latent_train"], dtype=float)[:, ::time_step]
        latent_test = np.asarray(fit["latent_test"], dtype=float)[:, ::time_step]
        temporal_folds.append(
            api.split_auc(
                latent_train,
                labels[train],
                latent_test,
                labels[test],
                decoder=decoder,
                seed=seed + fold_id,
            )
        )
    return {"status": "computed", "temporal_auc": np.nanmean(temporal_folds, axis=0)}


def score_null_draw(
    fits: list[dict],
    labels: np.ndarray,
    decoder: str,
    folds,
    shuffle_seed: int,
    decoder_seed: int,
    time_step: int = 1,
) -> dict:
    api = _decoder_api()
    shuffled = api.stratified_permutation(
        labels, folds, np.random.default_rng(shuffle_seed)
    )
    fold_scores = []
    for fold_id, (train, test) in enumerate(folds):
        fit = fits[fold_id]
        fold_scores.append(
            api.split_auc(
                np.asarray(fit["latent_train"], dtype=float)[:, ::time_step],
                shuffled[train],
                np.asarray(fit["latent_test"], dtype=float)[:, ::time_step],
                shuffled[test],
                decoder=decoder,
                seed=decoder_seed + fold_id,
            )
        )
    return {"status": "computed", "temporal_auc": np.nanmean(fold_scores, axis=0)}


def summarize_scores(temporal: np.ndarray, temporal_null: np.ndarray, decoder: str) -> dict:
    temporal = np.asarray(temporal, dtype=float)
    temporal_null = np.asarray(temporal_null, dtype=float)
    same_time = np.diag(temporal)
    same_time_null = np.diagonal(temporal_null, axis1=1, axis2=2)
    off_diag = ~np.eye(temporal.shape[0], dtype=bool)
    return {
        "status": "computed",
        "decoder": decoder,
        "same_time": _metric(same_time, same_time_null),
        "cross_temporal": _metric(temporal, temporal_null, off_diag),
        "same_time_auc": np.asarray(same_time),
        "temporal_auc": np.asarray(temporal),
        "same_time_null_mean": np.nanmean(same_time_null, axis=0),
        "temporal_null_mean": np.nanmean(temporal_null, axis=0),
    }


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


def _folds(labels: np.ndarray, n_splits: int, seed: int):
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    usable = min(n_splits, int(counts.min())) if counts.size else 0
    if usable < 2:
        return None
    return _decoder_api().make_folds(labels, n_splits=usable, seed=seed)


def fold_signature(folds) -> str:
    values = [
        [np.asarray(train, dtype=int).tolist(), np.asarray(test, dtype=int).tolist()]
        for train, test in folds
    ]
    return hashlib.sha256(canonical_json(values).encode()).hexdigest()[:12]


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
) -> list[dict]:
    seed_tag = f"seed{seed}"
    fold_seed = stable_seed(f"info_benchmark|{seed_tag}|{corpus}|{session}|{level}|folds")
    folds = _folds(labels, n_splits, fold_seed)
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
                identity,
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
                f"{score_base}|observed", _observed, identity
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
                    f"{score_base}|null|{draw}", _null, identity
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


def render_summary(
    records: list[dict],
    complete: bool,
    time_step: int = CTG_STEP,
    failed: bool = False,
) -> str:
    status = "failed" if failed else "complete" if complete else "running"
    lines = [
        "# Matched information benchmark",
        "",
        f"Status: {status}",
        "",
        "All decoder cells use the same stratified folds within each session and item-count level. "
        "Representations are fit on training trials at full temporal resolution. Each score is "
        "compared with label permutations evaluated on the same held-out folds.",
        f"Decoder temporal grids use every {time_step} time bin(s).",
        "",
        "| Corpus | Session | Level | Representation | Decoder | Status | Same-time AUC | Same-time null | Same-time ΔAUC | Same-time p | Cross-time AUC | Cross-time null | Cross-time ΔAUC | Cross-time p |",
        "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        same = record.get("same_time", {}).get("auc_above_null")
        cross = record.get("cross_temporal", {}).get("auc_above_null")
        same_auc = record.get("same_time", {}).get("auc")
        cross_auc = record.get("cross_temporal", {}).get("auc")
        same_null = record.get("same_time", {}).get("null_auc")
        cross_null = record.get("cross_temporal", {}).get("null_auc")
        same_p = record.get("same_time", {}).get("p_value")
        cross_p = record.get("cross_temporal", {}).get("p_value")
        lines.append(
            "| {corpus} | {session} | {level} | {candidate} | {decoder} | {status} | {same_auc} | {same_null} | {same} | {same_p} | {cross_auc} | {cross_null} | {cross} | {cross_p} |".format(
                corpus=record.get("corpus", ""),
                session=record.get("session", ""),
                level=record.get("level", ""),
                candidate=record.get("candidate", ""),
                decoder=record.get("decoder", ""),
                status=record.get("status", ""),
                same_auc="" if same_auc is None else f"{same_auc:.3f}",
                same_null="" if same_null is None else f"{same_null:.3f}",
                same="" if same is None else f"{same:.3f}",
                same_p="" if same_p is None else f"{same_p:.3g}",
                cross_auc="" if cross_auc is None else f"{cross_auc:.3f}",
                cross_null="" if cross_null is None else f"{cross_null:.3f}",
                cross="" if cross is None else f"{cross:.3f}",
                cross_p="" if cross_p is None else f"{cross_p:.3g}",
            )
        )
    return "\n".join(lines) + "\n"


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
