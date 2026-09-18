from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "scripts"):
    path = str(ROOT / part)
    if path not in sys.path:
        sys.path.insert(0, path)

from corpus_sessions import (  # noqa: E402
    data_root, independent_unit, iter_dandi_000469, iter_dandi_000574, iter_dandi_001187,
    load_watters_session, watters_behaviour, watters_session_dates,
)
from provenance import _json_safe, git_commit  # noqa: E402
from run_component_identity_subspace_atlas import (  # noqa: E402
    CANDIDATE_KEYS, CANDIDATE_SUPPORT_MATRIX, MAX_SESSIONS_ENV_VAR, _alm_session_inputs,
    _human_session_covariate_inputs, _panichello_reproduction_gate, _panichello_session_inputs,
    _previous_label, _session_core,
)
from statistics import fdr_bh, stable_seed  # noqa: E402
from subspace_identity import (  # noqa: E402
    block_folds, class_basis, permutation_alignment, regression_basis,
)

OUTPUT_PATH = ROOT / "results" / "rank_free_component_identity.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_rank_free_component_identity"
SCHEMA_VERSION = "1.0"
VERSION = "2026-09-18"
SEED_NAMESPACE = f"rank_free_component_identity|{VERSION}"
N_FOLDS = 2
N_PERM = 1000
N_BOOT = 2000
MIN_INDEPENDENT_UNITS = 4
FDR_ALPHA = 0.05


def _hash() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "src" / "subspace_identity.py", ROOT / "src" / "corpus_sessions.py",
        ROOT / "src" / "statistics.py", ROOT / "scripts" / "run_component_identity_subspace_atlas.py",
        ROOT / "scripts" / "run_dominant_latent_identity_and_behaviour_breadth.py",
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed(seed_id: str) -> int:
    return stable_seed(f"{SEED_NAMESPACE}|{seed_id}")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _checkpoint(key: str, identity: dict, fit) -> dict:
    readable = "".join(character if character.isalnum() or character in "-_" else "_" for character in key)[:80]
    suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = CHECKPOINT_DIR / f"{readable}.{suffix}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            record = cached.get("record")
            required = {"session", "independent_unit", "status"}
            if (
                cached.get("identity") == identity
                and cached.get("complete") is True
                and isinstance(record, dict)
                and required.issubset(record)
                and (record.get("status") != "computed" or isinstance(record.get("candidates"), dict))
            ):
                return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    record = fit()
    _write(path, {"complete": True, "identity": identity, "record": record})
    return record


def _finite(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    if target.ndim == 1:
        return np.isfinite(target)
    return np.isfinite(target).all(axis=tuple(range(1, target.ndim)))


def _cell(activity: np.ndarray, target: np.ndarray, kind: str, n_perm: int, seed: str) -> dict:
    activity = np.asarray(activity, dtype=float)
    norms = np.linalg.norm(activity, axis=1)
    valid = np.isfinite(activity).all(axis=1) & (norms > 0) & _finite(target)
    directions = activity[valid] / norms[valid, None]
    target = np.asarray(target)[valid]
    if len(target) < 2 * max(3, N_FOLDS):
        return {"status": "not_computable", "reason": "too few finite trials"}
    fit_basis = class_basis if kind == "categorical" else regression_basis
    result = permutation_alignment(
        directions, target, block_folds(len(target), N_FOLDS), fit_basis, n_perm,
        np.random.default_rng(_seed(seed)),
    )
    result["n_trials"] = len(target)
    result["target_kind"] = kind
    result["seed_id"] = seed
    return result


def _session(
    session: str,
    independent_unit: str,
    activity: np.ndarray,
    categorical: dict,
    continuous: dict,
    n_perm: int,
    seed: str,
) -> dict:
    cells = {}
    for name, target in categorical.items():
        cells[name] = _cell(activity, np.asarray(target), "categorical", n_perm, f"{seed}|{name}")
    for name, target in continuous.items():
        cells[name] = _cell(activity, np.asarray(target), "continuous", n_perm, f"{seed}|{name}")
    cells["time_in_trial_within_delay"] = {
        "status": "not_computable",
        "reason": "the candidate varies across bins rather than trials and requires a separate blocked time-label null",
    }
    return {
        "session": session,
        "independent_unit": independent_unit,
        "status": "computed",
        "n_trials": int(np.sum(np.isfinite(activity).all(axis=1) & (np.linalg.norm(activity, axis=1) > 0))),
        "candidates": cells,
    }


def _strip(record: dict) -> dict:
    clean = dict(record)
    clean["candidates"] = {
        name: {key: value for key, value in cell.items() if key != "null_draws"}
        for name, cell in record.get("candidates", {}).items()
    }
    return clean


def _standard_sessions(root: Path, corpus: str):
    if corpus == "panichello_2024_macaque_lPFC":
        for session, reason, core, categorical, continuous, counts in _panichello_session_inputs(root):
            activity = None if core is None or counts is None else np.asarray(counts).sum(axis=2)
            yield session, reason, activity, categorical, continuous
        return
    if corpus == "inagaki_alm5_mouse_ALM":
        for session, reason, core, categorical, continuous, counts in _alm_session_inputs(root):
            activity = None if core is None or counts is None else np.asarray(counts).sum(axis=2)
            yield session, reason, activity, categorical, continuous
        return
    iterator = {
        "dandi_000469_human": iter_dandi_000469,
        "dandi_001187_human": iter_dandi_001187,
        "dandi_000574_human": iter_dandi_000574,
    }[corpus]
    for session, core, categorical, continuous, counts in _human_session_covariate_inputs(root, iterator, corpus):
        activity = None if core is None or counts is None else np.asarray(counts).sum(axis=2)
        yield session, None if activity is not None else "session geometry unavailable", activity, categorical, continuous


def _combine_levels(levels: list[tuple[int, dict]]) -> dict:
    usable = [(weight, cell) for weight, cell in levels if cell.get("status") == "computed"]
    if not usable:
        return {"status": "not_computable", "reason": "no item-count level was computable"}
    weights = np.asarray([weight for weight, _ in usable], dtype=float)
    observed = np.average([cell["alignment"] for _, cell in usable], weights=weights)
    n_draws = min(len(cell["null_draws"]) for _, cell in usable)
    draws = np.average(
        np.stack([np.asarray(cell["null_draws"])[:n_draws] for _, cell in usable]), axis=0, weights=weights,
    )
    return {
        "status": "computed",
        "alignment": float(observed),
        "null_mean": float(draws.mean()),
        "alignment_above_null": float(observed - draws.mean()),
        "p_value": float((1 + np.sum(draws >= observed)) / (len(draws) + 1)),
        "n_permutations": len(draws),
        "null_interval_95pct": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "null_draws": draws,
        "n_levels": len(usable),
        "n_trials": int(weights.sum()),
        "target_kind": usable[0][1].get("target_kind"),
        "seed_ids": [cell.get("seed_id") for _, cell in usable],
    }


def _watters_session_record(
    root: Path,
    behaviour,
    animal: str,
    date: str,
    variant: str,
    n_perm: int,
) -> dict:
    session_id = f"{animal}_{date}_{variant}"
    session = load_watters_session(root, animal, date, behaviour)
    if session.get("status") != "loaded":
        return {
            "session": session_id,
            "independent_unit": animal,
            "status": "not_computable",
            "reason": session.get("status"),
        }
    counts = session["counts"]
    levels = session["num_objects"]
    cued = np.column_stack([np.cos(session["cued_theta"]), np.sin(session["cued_theta"])])
    previous = np.column_stack([_previous_label(cued[:, 0]), _previous_label(cued[:, 1])])
    cells: dict[str, list[tuple[int, dict]]] = {}
    level_trial_counts = []
    for level in np.unique(levels):
        mask = levels == level
        core = _session_core(counts[mask].sum(axis=2))
        if core is None:
            continue
        activity = counts[mask].sum(axis=2)
        level_trial_counts.append(int(core["n_kept"]))
        continuous = {
            "memorandum_content": cued[mask],
            "previous_trial_content": previous[mask],
            "gain_total_spike_count": core["spike_count"],
        }
        record = _session(
            session_id,
            animal,
            activity,
            {},
            continuous,
            n_perm,
            f"watters|{session_id}|level={level}",
        )
        for name, cell in record["candidates"].items():
            cells.setdefault(name, []).append((int(cell.get("n_trials", core["n_kept"])), cell))
    return {
        "session": session_id,
        "independent_unit": animal,
        "status": "computed" if cells else "not_computable",
        "n_trials": sum(level_trial_counts),
        "n_levels": len(level_trial_counts),
        "candidates": {name: _combine_levels(entries) for name, entries in cells.items()},
    }


def _pool(records: list[dict], candidate: str, seed: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        cell = record.get("candidates", {}).get(candidate, {})
        if cell.get("status") == "computed":
            grouped.setdefault(record["independent_unit"], []).append(cell)
    n_sessions = sum(len(cells) for cells in grouped.values())
    if len(grouped) < MIN_INDEPENDENT_UNITS:
        n_sessions_with_session_geometry = sum(1 for record in records if record.get("status") == "computed")
        return {
            "status": "not_computable",
            "n_sessions": n_sessions,
            "n_sessions_with_session_geometry": n_sessions_with_session_geometry,
            "n_independent_units": len(grouped),
            "reason": f"fewer than {MIN_INDEPENDENT_UNITS} independent units carry a computed cell",
        }
    n_draws = min(len(cell["null_draws"]) for cells in grouped.values() for cell in cells)
    unit_observed = []
    unit_draws = []
    for cells in grouped.values():
        unit_observed.append(float(np.mean([cell["alignment"] for cell in cells])))
        unit_draws.append(
            np.mean(np.stack([np.asarray(cell["null_draws"])[:n_draws] for cell in cells]), axis=0)
        )
    observed = float(np.mean(unit_observed))
    draws = np.mean(np.stack(unit_draws), axis=0)
    effects = np.asarray([value - null.mean() for value, null in zip(unit_observed, unit_draws)])
    rng = np.random.default_rng(_seed(seed))
    boot = np.asarray([effects[rng.integers(0, len(effects), len(effects))].mean() for _ in range(N_BOOT)])
    return {
        "status": "computed",
        "n_sessions": n_sessions,
        "n_independent_units": len(grouped),
        "independent_unit_ids": sorted(grouped),
        "session_weighting_within_unit": "equal",
        "independent_unit_weighting": "equal",
        "mean_alignment": observed,
        "mean_null_alignment": float(draws.mean()),
        "mean_alignment_above_null": float(effects.mean()),
        "cluster_bootstrap_interval_95pct": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "p_value": float((1 + np.sum(draws >= observed)) / (len(draws) + 1)),
        "n_permutations": len(draws),
        "bootstrap_seed_id": seed,
    }


def _summarize(corpus_records: dict[str, list[dict]]) -> dict:
    summary = {}
    supported_cells = 0
    for corpus, records in corpus_records.items():
        present = [name for name in CANDIDATE_KEYS if CANDIDATE_SUPPORT_MATRIX[corpus][name][0] == "present"]
        supported_cells += len(present)
        pooled = {name: _pool(records, name, f"{corpus}|{name}|bootstrap") for name in present}
        summary[corpus] = pooled
    tested = [
        (corpus, name, cell)
        for corpus, candidates in summary.items()
        for name, cell in candidates.items()
        if cell["status"] == "computed"
    ]
    if tested:
        correction = fdr_bh(np.asarray([cell["p_value"] for _, _, cell in tested]), alpha=FDR_ALPHA)
        for (_, _, cell), reject, q_value in zip(tested, correction["reject"], correction["q_values"]):
            cell["fdr_q_value"] = float(q_value)
            cell["aligned"] = bool(reject and cell["mean_alignment_above_null"] > 0)
    aligned = {
        name: [corpus for corpus, candidates in summary.items() if candidates.get(name, {}).get("aligned")]
        for name in CANDIDATE_KEYS
    }
    aligned = {name: corpora for name, corpora in aligned.items() if corpora}
    branch = "no_candidate_corpus_cell_was_computable"
    if tested and not aligned:
        branch = "no_labelled_subspace_clears_the_refit_null"
    elif len(aligned) == 1:
        branch = "one_labelled_subspace_clears_the_refit_null"
    elif len(aligned) > 1:
        branch = "multiple_labelled_subspaces_clear_the_refit_null"
    return {
        "per_corpus": summary,
        "aligned_anywhere": aligned,
        "branch": branch,
        "fdr_scope": "all_computed_candidate_by_corpus_cells",
        "n_supported_candidate_corpus_cells": supported_cells,
        "n_computed_candidate_corpus_tests": len(tested),
        "all_supported_cells_tested": len(tested) == supported_cells,
    }


def _collect_standard(root: Path, corpus: str, n_perm: int, identity: dict) -> list[dict]:
    records = []
    for session, reason, activity, categorical, continuous in _standard_sessions(root, corpus):
        key = f"{corpus}|{session}"
        unit_id = independent_unit(corpus, session)

        def fit(
            session=session,
            reason=reason,
            activity=activity,
            categorical=categorical,
            continuous=continuous,
            unit_id=unit_id,
            key=key,
        ):
            if activity is None:
                return {
                    "session": session,
                    "independent_unit": unit_id,
                    "status": "not_computable",
                    "reason": reason,
                }
            return _session(session, unit_id, activity, categorical, continuous, n_perm, key)

        records.append(_checkpoint(key, identity, fit))
    return records


def _collect_watters(root: Path, n_perm: int, identity: dict) -> list[dict]:
    dates = watters_session_dates(root)
    limit = os.environ.get(MAX_SESSIONS_ENV_VAR)
    if limit:
        dates = dates[:int(limit)]
    behaviour_cache = []

    def behaviour():
        if not behaviour_cache:
            behaviour_cache.append(watters_behaviour(root))
        return behaviour_cache[0]

    records = []
    for animal, date, variant in dates:
        session_id = f"{animal}_{date}_{variant}"
        key = f"watters_2026_macaque_multi_object|{session_id}"
        records.append(_checkpoint(
            key,
            identity,
            lambda animal=animal, date=date, variant=variant: _watters_session_record(
                root, behaviour(), animal, date, variant, n_perm,
            ),
        ))
    return records


def main() -> None:
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    CHECKPOINT_DIR = args.checkpoint_dir
    if args.n_perm < 1:
        parser.error("--n-perm must be positive")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    if args.max_sessions is not None:
        os.environ[MAX_SESSIONS_ENV_VAR] = str(args.max_sessions)
    started = time.time()
    root = data_root()
    identity = {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "code_hash": _hash(),
        "data_root": str(root.resolve()),
        "n_perm": args.n_perm,
        "n_folds": N_FOLDS,
        "seed_namespace": SEED_NAMESPACE,
        "seed_algorithm": "statistics.stable_seed_crc32",
    }
    corpora = (
        "panichello_2024_macaque_lPFC", "watters_2026_macaque_multi_object", "inagaki_alm5_mouse_ALM",
        "dandi_000469_human", "dandi_001187_human", "dandi_000574_human",
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "method": "cross_fitted_squared_subspace_alignment",
        "estimand": (
            "equal-independent-unit mean held-out squared projection of an evaluation-fold residual axis "
            "onto a training-fold label subspace"
        ),
        "cross_fitting": (
            "each fold's reference direction and labelled subspace are fit only on training trials; "
            "the residual axis is estimated only from evaluation trials relative to that training reference"
        ),
        "null": (
            "labels are permuted within contiguous folds and every training-fold label subspace is refit "
            "for every permutation"
        ),
        "pooling": (
            "sessions are averaged within animal or participant, then independent units are weighted equally; "
            "label-null draws use independent deterministic session seeds and uncertainty resamples independent units"
        ),
        "identity": identity,
        "data_root": str(root),
        "reproduction_gate": _panichello_reproduction_gate(root),
        "corpora": {},
    }
    _write(args.output, output)
    raw_records = {}
    for corpus in corpora:
        if corpus == "watters_2026_macaque_multi_object":
            records = _collect_watters(root, args.n_perm, identity)
        else:
            records = _collect_standard(root, corpus, args.n_perm, identity)
        raw_records[corpus] = records
        output["corpora"][corpus] = {"records": [_strip(record) for record in records]}
        _write(args.output, output)
    output["summary"] = _summarize(raw_records)
    output["status"] = "complete"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(args.output, output)


if __name__ == "__main__":
    main()
