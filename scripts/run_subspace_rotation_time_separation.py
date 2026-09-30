"""results/subspace_rotation_versus_noise.json contrasts each session's early-to-late candidate-
subspace overlap against a within-half reference built from two bootstrap resamples of that SAME
half. Two bootstrap draws from one half share most of their trials, so that reference is biased
toward high overlap, which biases the across-minus-within contrast toward finding rotation --
sample size was matched to the across-half comparison, but independence was not, and independence
is what governs how much two fits of a stationary relation agree.

This module replaces that reference with two that match the across-half comparison on BOTH sample
size and disjointness, leaving temporal separation as the only remaining difference:

  interleaved-chunk reference: a session's kept trials are cut into many short contiguous chunks,
  assigned alternately to set A and set B (A = even chunks, B = odd chunks). A and B are disjoint,
  each holds about half the trials, and both span the session's full time range -- so this is the
  primary, sample-size- and disjointness-matched reference. Its own free parameter, chunk length,
  is swept rather than fixed at one value: too long and it starts to resemble the early/late split
  it is meant to be compared against; too short and consecutive chunks stop being independent draws
  of the underlying relation. The sweep is declared below, before any cell is fit.

  quarter-split reference: each half (early, late) is itself split into two disjoint, contiguous
  quarters, a subspace is fit in each quarter, and their overlap is measured; the result is averaged
  over the early and late half. This also matches disjointness, but each fit now sees only a quarter
  of the session's trials rather than a half, so it is deliberately noisier and biased toward LOWER
  overlap -- a conservative reference that works against concluding rotation, the mirror image of the
  reference being replaced.

The across-half overlap itself is reproduced unchanged (results/alignment_below_null_diagnostic.py's
own _rotation, called on identical per-session inputs) and held to a reproduction gate. Every basis-
fitting primitive (block_folds, class_basis, regression_basis) and subspace_overlap are imported
unchanged from src/subspace_identity.py and src/geometry.py; the equal-session-then-equal-unit
pooling with its cluster bootstrap (_pool_unit_scalars) is reused unchanged from
results/alignment_below_null_diagnostic.py. The two new pieces of math are the interleaved-chunk fold
assignment and the quarter-split reference below -- everything else is reused estimator code.
"""

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

from corpus_sessions import data_root, independent_unit  # noqa: E402
from geometry import subspace_overlap  # noqa: E402
from provenance import _json_safe, git_commit, restore_checkpoint  # noqa: E402
from run_alignment_below_null_diagnostic import OUTPUT_PATH as DELIVERED_DIAGNOSTIC_OUTPUT_PATH, _pool_unit_scalars
from info_decoding import _prepare_trials
from info_decoding import _rotation
from info_decoding import REPRODUCTION_TOLERANCE
from info_decoding import MAX_SESSIONS_ENV_VAR
from run_rank_free_component_identity import _standard_sessions
from info_decoding import N_FOLDS
from statistics import stable_seed  # noqa: E402
from subspace_identity import block_folds, class_basis, regression_basis  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "subspace_rotation_time_separation.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_subspace_rotation_time_separation"
SCHEMA_VERSION = "1.1"
VERSION = "2026-09-18"
SEED_NAMESPACE = f"subspace_rotation_time_separation|{VERSION}"

CHUNK_SIZES_TRIALS = (2, 4, 8, 16)
PRIMARY_CHUNK_SIZE_TRIALS = 4
MIN_CHUNKS_PER_SESSION = 4  # at least two chunks per side, so alternation is meaningful


def _hash() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "src" / "subspace_identity.py", ROOT / "src" / "corpus_sessions.py",
        ROOT / "src" / "statistics.py", ROOT / "src" / "geometry.py", ROOT / "src" / "provenance.py", ROOT / "src" / "info_decoding.py",
        ROOT / "scripts" / "run_component_identity_subspace_atlas.py",
        ROOT / "scripts" / "run_rank_free_component_identity.py",
        ROOT / "scripts" / "run_alignment_below_null_diagnostic.py",
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


def _checkpoint(key: str, identity: dict, fit):
    readable = "".join(character if character.isalnum() or character in "-_" else "_" for character in key)[:80]
    suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = CHECKPOINT_DIR / f"{readable}.{suffix}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            record = restore_checkpoint(cached.get("record"))
            if cached.get("identity") == identity and cached.get("complete") is True and isinstance(record, dict):
                return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    record = fit()
    _write(path, {"complete": True, "identity": identity, "record": record})
    return record


# ================================================================================================
# the two new estimand primitives. Everything downstream (class_basis/regression_basis,
# subspace_overlap) is reused unchanged.
# ================================================================================================

def chunked_interleaved_folds(n: int, chunk_size: int) -> np.ndarray:
    if chunk_size < 1 or n < MIN_CHUNKS_PER_SESSION * chunk_size:
        raise ValueError("chunk_size must be positive and n must hold enough chunks")
    n_chunks = int(np.ceil(n / chunk_size))
    folds = np.empty(n, dtype=int)
    for chunk in range(n_chunks):
        start, stop = chunk * chunk_size, min((chunk + 1) * chunk_size, n)
        folds[start:stop] = chunk % 2
    return folds


def _chunk_reference(directions: np.ndarray, target: np.ndarray, fit_basis, n: int, chunk_size: int) -> dict:
    if n < MIN_CHUNKS_PER_SESSION * chunk_size:
        return {"status": "not_computable", "reason": "fewer than four chunks of this size fit in the session"}
    folds = chunked_interleaved_folds(n, chunk_size)
    basis_a = fit_basis(directions[folds == 0], target[folds == 0])
    basis_b = fit_basis(directions[folds == 1], target[folds == 1])
    if basis_a is None or basis_b is None:
        return {"status": "not_computable", "reason": "label subspace could not be fit in one or both chunk sets"}
    return {
        "status": "computed",
        "overlap": float(subspace_overlap(basis_a, basis_b)),
        "n_trials_a": int((folds == 0).sum()),
        "n_trials_b": int((folds == 1).sum()),
        "n_chunks": int(np.ceil(n / chunk_size)),
    }


def _quarter_split_reference(directions: np.ndarray, target: np.ndarray, fit_basis) -> dict:
    n = len(target)
    halves = block_folds(n, N_FOLDS)
    per_half = {}
    overlaps = []
    for half_label, half_name in ((0, "early"), (1, "late")):
        mask = halves == half_label
        half_directions, half_target = directions[mask], target[mask]
        half_n = len(half_target)
        if half_n < 4:
            per_half[half_name] = {"status": "not_computable", "reason": "half has fewer than four trials to quarter"}
            continue
        quarters = block_folds(half_n, 2)
        basis_a = fit_basis(half_directions[quarters == 0], half_target[quarters == 0])
        basis_b = fit_basis(half_directions[quarters == 1], half_target[quarters == 1])
        if basis_a is None or basis_b is None:
            per_half[half_name] = {"status": "not_computable", "reason": "label subspace could not be fit in one or both quarters"}
            continue
        overlap = float(subspace_overlap(basis_a, basis_b))
        per_half[half_name] = {
            "status": "computed", "overlap": overlap,
            "n_trials_quarter_a": int((quarters == 0).sum()), "n_trials_quarter_b": int((quarters == 1).sum()),
        }
        overlaps.append(overlap)
    if not overlaps:
        return {
            "status": "not_computable", "reason": "quarter subspace fit failed in both halves",
            "early": per_half.get("early"), "late": per_half.get("late"),
        }
    return {"status": "computed", "mean": float(np.mean(overlaps)), "early": per_half.get("early"), "late": per_half.get("late")}


def _cell_session(activity: np.ndarray, kind: str, target: np.ndarray, key: str) -> dict:
    prepared = _prepare_trials(activity, target)
    if prepared is None:
        return {"status": "too_few_trials"}
    directions, filtered_target = prepared
    rotation = _rotation(directions, filtered_target, kind)
    if rotation.get("status") != "computed":
        return {"status": "across_half_not_computable"}
    n = len(filtered_target)
    fit_basis = class_basis if kind == "categorical" else regression_basis
    chunk_by_size = {
        chunk_size: _chunk_reference(directions, filtered_target, fit_basis, n, chunk_size)
        for chunk_size in CHUNK_SIZES_TRIALS
    }
    quarter = _quarter_split_reference(directions, filtered_target, fit_basis)
    return {
        "status": "computed",
        "across_half_overlap": rotation["subspace_overlap"],
        "n_trials_early": rotation["n_trials_early"],
        "n_trials_late": rotation["n_trials_late"],
        "n_trials_total": n,
        "chunk_reference_by_size": chunk_by_size,
        "quarter_split_reference": quarter,
    }


def _corpus_cells(root: Path, corpus: str, names: list[str], identity: dict) -> dict:
    per_candidate = {
        name: {
            "records": [], "n_sessions_seen": 0, "n_sessions_activity_unavailable": 0,
            "n_sessions_candidate_absent": 0, "n_sessions_too_few_trials": 0,
            "n_sessions_across_half_not_computable": 0, "n_sessions_across_half_computed": 0,
            "n_sessions_quarter_not_computable": 0, "n_sessions_quarter_computed": 0,
            **{f"n_sessions_chunk_{cs}_not_computable": 0 for cs in CHUNK_SIZES_TRIALS},
            **{f"n_sessions_chunk_{cs}_computed": 0 for cs in CHUNK_SIZES_TRIALS},
        }
        for name in names
    }
    for session, reason, activity, categorical, continuous in _standard_sessions(root, corpus):
        key = f"{corpus}|{session}"
        unit_id = independent_unit(corpus, session)

        def fit(reason=reason, activity=activity, categorical=categorical, continuous=continuous, key=key):
            if activity is None:
                return {"status": "not_computable", "reason": reason, "candidates": {}}
            per_name = {}
            for name in names:
                if name in categorical:
                    kind, target = "categorical", categorical[name]
                elif name in continuous:
                    kind, target = "continuous", continuous[name]
                else:
                    per_name[name] = {"status": "candidate_absent"}
                    continue
                per_name[name] = _cell_session(activity, kind, np.asarray(target), f"{key}|{name}")
            return {"status": "computed", "candidates": per_name}

        record = _checkpoint(key, identity, fit)
        for name in names:
            bucket = per_candidate[name]
            bucket["n_sessions_seen"] += 1
            if record["status"] != "computed":
                bucket["n_sessions_activity_unavailable"] += 1
                continue
            entry = record["candidates"].get(name, {"status": "candidate_absent"})
            status = entry["status"]
            if status == "candidate_absent":
                bucket["n_sessions_candidate_absent"] += 1
                continue
            if status == "too_few_trials":
                bucket["n_sessions_too_few_trials"] += 1
                continue
            if status == "across_half_not_computable":
                bucket["n_sessions_across_half_not_computable"] += 1
                continue
            bucket["n_sessions_across_half_computed"] += 1
            quarter = entry["quarter_split_reference"]
            if quarter["status"] == "computed":
                bucket["n_sessions_quarter_computed"] += 1
            else:
                bucket["n_sessions_quarter_not_computable"] += 1
            for chunk_size in CHUNK_SIZES_TRIALS:
                chunk = entry["chunk_reference_by_size"][chunk_size]
                suffix = "computed" if chunk["status"] == "computed" else "not_computable"
                bucket[f"n_sessions_chunk_{chunk_size}_{suffix}"] += 1
            bucket["records"].append({
                "session": session, "independent_unit": unit_id,
                "across_half_overlap": entry["across_half_overlap"],
                "n_trials_early": entry["n_trials_early"], "n_trials_late": entry["n_trials_late"],
                "n_trials_total": entry["n_trials_total"],
                "chunk_reference_by_size": entry["chunk_reference_by_size"],
                "quarter_split_reference": quarter,
            })
    return per_candidate


def _pool_field(records: list[dict], extractor, seed: str) -> dict:
    unit_values: dict[str, list[float]] = {}
    for record in records:
        value = extractor(record)
        if value is not None:
            unit_values.setdefault(record["independent_unit"], []).append(value)
    return _pool_unit_scalars(unit_values, seed)


def _pool_across_half(records: list[dict], seed: str) -> dict:
    return _pool_field(records, lambda r: r["across_half_overlap"], seed)


def _pool_chunk_reference(records: list[dict], chunk_size: int, seed: str) -> dict:
    def extractor(r):
        cell = r["chunk_reference_by_size"][chunk_size]
        return cell["overlap"] if cell["status"] == "computed" else None
    return _pool_field(records, extractor, seed)


def _pool_contrast_chunk(records: list[dict], chunk_size: int, seed: str) -> dict:
    def extractor(r):
        cell = r["chunk_reference_by_size"][chunk_size]
        return (r["across_half_overlap"] - cell["overlap"]) if cell["status"] == "computed" else None
    return _pool_field(records, extractor, seed)


def _pool_quarter_reference(records: list[dict], seed: str) -> dict:
    def extractor(r):
        cell = r["quarter_split_reference"]
        return cell["mean"] if cell["status"] == "computed" else None
    return _pool_field(records, extractor, seed)


def _pool_contrast_quarter(records: list[dict], seed: str) -> dict:
    def extractor(r):
        cell = r["quarter_split_reference"]
        return (r["across_half_overlap"] - cell["mean"]) if cell["status"] == "computed" else None
    return _pool_field(records, extractor, seed)


def _reproduction_check(reproduced: dict, delivered: dict) -> dict:
    if reproduced.get("status") != "computed" or delivered.get("status") != "computed":
        return {"matches": False, "reason": "delivered or reproduced pooled cell is not_computable"}
    diff = abs(float(reproduced["mean"]) - float(delivered["mean"]))
    return {"matches": diff <= REPRODUCTION_TOLERANCE, "absolute_difference": diff}


def _interleaved_only_branch(contrast: dict, meaningful_difference: float) -> str:
    if contrast.get("status") != "computed":
        return "undetermined_below_four_independent_unit_floor"
    lo, hi = contrast["cluster_bootstrap_interval_95pct"]
    mdd = contrast["minimum_detectable_difference_80pct_power"]
    if hi < 0.0:
        return "below_interleaved_chunk_reference"
    if lo > 0.0:
        return "above_interleaved_chunk_reference_unexpected_direction"
    if mdd < meaningful_difference:
        return "at_parity_with_interleaved_chunk_reference"
    return "undetermined_underpowered"


def _branch(contrast_interleaved: dict, contrast_quarter: dict, meaningful_difference: float) -> str:
    primary = _interleaved_only_branch(contrast_interleaved, meaningful_difference)
    if primary == "below_interleaved_chunk_reference":
        if contrast_quarter.get("status") != "computed":
            return "undetermined_quarter_reference_not_computable"
        lo_q, hi_q = contrast_quarter["cluster_bootstrap_interval_95pct"]
        if hi_q < 0.0:
            return "rotates_beyond_noise"
        return "undetermined_between_references"
    if primary == "at_parity_with_interleaved_chunk_reference":
        return "time_separation_makes_no_difference"
    if primary == "above_interleaved_chunk_reference_unexpected_direction":
        return "undetermined_unexpected_direction"
    return primary  # undetermined_underpowered or undetermined_below_four_independent_unit_floor


def _run(root: Path, needed_by_corpus: dict, identity: dict, delivered_cells: dict,
          meaningful_difference: float, output: dict, output_path: Path) -> bool:
    reproduction_ok = True
    for corpus, names in needed_by_corpus.items():
        print(f"starting corpus {corpus} ({len(names)} candidate(s))", file=sys.stderr, flush=True)
        per_candidate = _corpus_cells(root, corpus, names, identity)
        corpus_cells = {}
        for name in names:
            bucket = per_candidate[name]
            records = bucket["records"]
            reproduced = _pool_across_half(records, f"{corpus}|{name}|reproduced_across_half_bootstrap")
            delivered = delivered_cells[corpus][name]["early_late_subspace_overlap"]
            reproduction = _reproduction_check(reproduced, delivered)

            chunk_sweep = {}
            for chunk_size in CHUNK_SIZES_TRIALS:
                reference = _pool_chunk_reference(records, chunk_size, f"{corpus}|{name}|chunk_{chunk_size}_reference_bootstrap")
                contrast = _pool_contrast_chunk(records, chunk_size, f"{corpus}|{name}|chunk_{chunk_size}_contrast_bootstrap")
                chunk_sweep[chunk_size] = {
                    "reference": reference, "contrast": contrast,
                    "branch": _interleaved_only_branch(contrast, meaningful_difference),
                }
            quarter_reference = _pool_quarter_reference(records, f"{corpus}|{name}|quarter_reference_bootstrap")
            quarter_contrast = _pool_contrast_quarter(records, f"{corpus}|{name}|quarter_contrast_bootstrap")

            primary = chunk_sweep[PRIMARY_CHUNK_SIZE_TRIALS]
            branch = _branch(primary["contrast"], quarter_contrast, meaningful_difference)
            computed_branches = {
                v["branch"] for v in chunk_sweep.values()
                if v["branch"] not in ("undetermined_underpowered", "undetermined_below_four_independent_unit_floor")
            }
            stable = len(computed_branches) <= 1

            corpus_cells[name] = {
                "delivered_across_half_overlap": delivered,
                "reproduced_across_half_overlap": reproduced,
                "reproduction": reproduction,
                "quarter_split_reference": quarter_reference,
                "quarter_split_contrast": quarter_contrast,
                "chunk_size_sweep": {str(cs): v for cs, v in chunk_sweep.items()},
                "primary_chunk_size_trials": PRIMARY_CHUNK_SIZE_TRIALS,
                "primary_interleaved_chunk_reference": primary["reference"],
                "primary_interleaved_chunk_contrast": primary["contrast"],
                "chunk_size_sweep_stable": stable,
                "branch": branch,
                "zero_drop": {k: v for k, v in bucket.items() if k != "records"},
            }
            if not reproduction["matches"]:
                reproduction_ok = False
        output["cells"][corpus] = corpus_cells
        _write(output_path, output)
        print(f"finished corpus {corpus}", file=sys.stderr, flush=True)
    return reproduction_ok


def _summarize(cells: dict) -> dict:
    table = []
    for corpus, candidates in cells.items():
        for name, result in candidates.items():
            table.append({
                "corpus": corpus, "candidate": name,
                "across_half_overlap": result["reproduced_across_half_overlap"].get("mean"),
                "interleaved_chunk_reference": result["primary_interleaved_chunk_reference"].get("mean"),
                "interleaved_chunk_contrast": result["primary_interleaved_chunk_contrast"].get("mean"),
                "interleaved_chunk_contrast_interval_95pct": result["primary_interleaved_chunk_contrast"].get("cluster_bootstrap_interval_95pct"),
                "interleaved_chunk_contrast_mdd_80pct_power": result["primary_interleaved_chunk_contrast"].get("minimum_detectable_difference_80pct_power"),
                "quarter_split_reference": result["quarter_split_reference"].get("mean"),
                "quarter_split_contrast": result["quarter_split_contrast"].get("mean"),
                "quarter_split_contrast_interval_95pct": result["quarter_split_contrast"].get("cluster_bootstrap_interval_95pct"),
                "chunk_size_sweep_stable": result["chunk_size_sweep_stable"],
                "branch": result["branch"],
            })
    counts: dict[str, int] = {}
    for row in table:
        counts[row["branch"]] = counts.get(row["branch"], 0) + 1
    n_stable = sum(1 for row in table if row["chunk_size_sweep_stable"])
    established = counts.get("rotates_beyond_noise", 0)
    withdrawn = counts.get("time_separation_makes_no_difference", 0)
    undetermined = len(table) - established - withdrawn
    if established and not withdrawn and not undetermined:
        overall = (
            "the subspace rotates within a session beyond what estimation noise explains, in every cell "
            "this reaches, under both a sample-size-and-disjointness-matched reference and a deliberately "
            "conservative one"
        )
    elif withdrawn and not established and not undetermined:
        overall = (
            "time separation makes no measurable difference to the fit in every cell this reaches -- the "
            "rotation reading is withdrawn everywhere it can be tested"
        )
    else:
        overall = (
            f"of {len(table)} cells, {established} show early-to-late overlap reliably below both the "
            "interleaved-chunk and quarter-split references (rotation established under both directions "
            f"of bias), {withdrawn} sit at parity with the interleaved-chunk reference (the rotation "
            f"reading is withdrawn there), and the remaining {undetermined} are undetermined -- the data "
            "do not support one general answer across all twelve cells"
        )
    return {
        "cell_table": table,
        "n_cells_tested": len(table),
        "branch_counts": counts,
        "n_cells_chunk_size_sweep_stable": n_stable,
        "overall_statement": overall,
    }


def main() -> None:
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    CHECKPOINT_DIR = args.checkpoint_dir
    if args.max_sessions is not None:
        if args.max_sessions < 1:
            parser.error("--max-sessions must be positive")
        os.environ[MAX_SESSIONS_ENV_VAR] = str(args.max_sessions)
    started = time.time()
    root = data_root()

    delivered = json.loads(DELIVERED_DIAGNOSTIC_OUTPUT_PATH.read_text())
    delivered_cells = delivered["cells"]
    needed_by_corpus: dict[str, list[str]] = {}
    for entry in delivered["cells_under_test"]:
        corpus, name = entry["corpus"], entry["candidate"]
        if delivered_cells[corpus][name]["early_late_subspace_overlap"].get("status") == "computed":
            needed_by_corpus.setdefault(corpus, []).append(name)

    mdd_values = [
        delivered_cells[corpus][name]["early_late_subspace_overlap"]["minimum_detectable_difference_80pct_power"]
        for corpus, names in needed_by_corpus.items() for name in names
    ]
    meaningful_difference = float(np.median(mdd_values))

    identity = {
        "version": VERSION, "schema_version": SCHEMA_VERSION, "code_hash": _hash(),
        "data_root": str(root.resolve()), "n_folds": N_FOLDS,
        "chunk_sizes_trials": list(CHUNK_SIZES_TRIALS), "primary_chunk_size_trials": PRIMARY_CHUNK_SIZE_TRIALS,
        "min_chunks_per_session": MIN_CHUNKS_PER_SESSION,
        "seed_namespace": SEED_NAMESPACE, "seed_algorithm": "statistics.stable_seed_crc32",
        "delivered_diagnostic_artifact_path": str(DELIVERED_DIAGNOSTIC_OUTPUT_PATH),
    }

    decision_rules = {
        "cells_under_test": (
            "every (corpus, candidate) cell results/alignment_below_null_diagnostic.json computed an "
            "early-to-late subspace overlap for -- the same twelve cells, not a subset picked after "
            "seeing which look most or least rotated"
        ),
        "across_half_overlap_definition": (
            "unchanged from the delivered diagnostic: a session's kept trials are split into contiguous "
            "early and late halves (block_folds(n, 2)), the candidate's label subspace is fit separately "
            "in each half with class_basis or regression_basis, and subspace_overlap between the two "
            "fitted subspaces is the across-half overlap for that session"
        ),
        "across_half_overlap_reproduction_gate": (
            "the across-half overlap is recomputed here by calling the delivered diagnostic's own "
            "_rotation function, unmodified, on the same per-session activity and target arrays, then "
            "pooled with the same equal-session-then-equal-unit weighting and cluster bootstrap the "
            "delivered diagnostic uses; the pooled mean must match the delivered mean within numerical "
            "tolerance -- if any cell fails this, that cell's references and contrasts are still "
            "computed and reported, but the run-level status is reported as a failed reproduction gate "
            "and no rotation-versus-time-separation verdict is drawn from any cell"
        ),
        "interleaved_chunk_reference_definition": (
            "a session's kept trials are cut into contiguous chunks of a fixed trial count and assigned "
            "alternately to set A (even-indexed chunks) and set B (odd-indexed chunks); A and B are "
            "disjoint, each holds about half the session's trials, and both span the session's full time "
            "range. The candidate subspace is fit in A and in B with the same fit_basis function the "
            "across-half comparison uses, and their overlap is the interleaved-chunk reference. This "
            "matches the across-half comparison on both sample size and disjointness, so temporal "
            "separation is the only remaining difference between the two"
        ),
        "chunk_size_sweep_declaration": (
            f"chunk length is a free parameter of the interleaved-chunk reference and is swept over "
            f"{list(CHUNK_SIZES_TRIALS)} trials, declared here before any cell is fit. Too long a chunk "
            "starts to resemble the early/late split it is meant to be compared against (each chunk "
            "spans a large, temporally localized stretch); too short a chunk risks consecutive same-set "
            f"chunks not being independent draws of the underlying relation. {PRIMARY_CHUNK_SIZE_TRIALS} "
            "trials is used as the primary reference for each cell's branch decision; every swept value "
            "is reported, and whether the branch is stable across the sweep is reported per cell. A "
            f"session needs at least {MIN_CHUNKS_PER_SESSION} chunks of a given size (so both A and B "
            "contain multiple chunks) for that chunk size to be computable in that session"
        ),
        "quarter_split_reference_definition": (
            "a secondary, deliberately conservative reference: each half (early, late) is itself split "
            "into two disjoint, contiguous quarters with the same block_folds primitive the early/late "
            "split uses, the candidate subspace is fit in each quarter, and their overlap is measured; "
            "the quarter-split reference for a session is the average of this quantity for the early "
            "half and the late half. This also matches disjointness, but each fit now sees only a "
            "quarter of the session's trials rather than a half, so it is a noisier estimate biased "
            "toward LOWER overlap than the matched-sample-size interleaved-chunk reference -- a "
            "reference biased against concluding rotation, the mirror image of the within-half bootstrap "
            "reference this run replaces"
        ),
        "meaningful_magnitude": (
            f"a contrast is treated as too small to matter if its minimum detectable difference at 80% "
            f"power is below {meaningful_difference:.4f} overlap units -- the median minimum detectable "
            "difference already reported for these same twelve cells' own early-to-late overlap "
            "estimates in the delivered diagnostic. A difference smaller than that would not have been "
            "distinguishable from noise even in the estimate this run exists to check in the first "
            "place, so it is not a magnitude this run can meaningfully rule in or out on its own terms"
        ),
        "rotates_beyond_noise_branch": (
            "at the primary chunk size, the across-minus-interleaved-chunk contrast's cluster bootstrap "
            "interval lies entirely below zero AND the across-minus-quarter-split contrast's interval "
            "also lies entirely below zero -- the early/late overlap sits reliably below both references, "
            "so rotation is established under both the matched reference and the deliberately "
            "conservative one, and the reading is safe"
        ),
        "time_separation_makes_no_difference_branch": (
            "at the primary chunk size, the across-minus-interleaved-chunk contrast's interval covers "
            "zero and its minimum detectable difference is below the pre-declared meaningful magnitude "
            "-- early/late overlap is at parity with the matched, sample-size-and-disjointness reference, "
            "so time separation does not degrade the fit and the delivered diagnostic's rotation reading "
            "is withdrawn for that cell"
        ),
        "undetermined_branch": (
            "any other outcome is reported as undetermined rather than forced into either of the other "
            "two branches, and the specific reason is named: the early/late overlap sits below the "
            "interleaved-chunk reference but not below the quarter-split reference (between the two "
            "references, ambiguous); the interleaved-chunk contrast's interval covers zero but its "
            "minimum detectable difference is at or above the meaningful magnitude (underpowered); the "
            "interleaved-chunk contrast's interval lies entirely above zero (a direction neither account "
            "predicts); or fewer than four independent units carry a computed value for the quantity in "
            "question"
        ),
        "chunk_size_sweep_stability": (
            "for each cell, the primary-chunk-size branch decision on the interleaved-chunk contrast "
            "alone (ignoring the quarter-split reference) is recomputed at every swept chunk size; the "
            "cell is reported as sweep-stable if every chunk size that reaches a definite (non-"
            "underpowered, non-below-floor) reading agrees on the same reading. An unstable cell means "
            "the conclusion depends on the arbitrary choice of chunk length and is reported as such, not "
            "resolved by picking a preferred chunk size after the fact"
        ),
        "power_reporting": (
            "a contrast that does not clear zero, or clears it but not the meaningful-magnitude bar, is "
            "reported with its cluster bootstrap interval and its minimum detectable difference at 80% "
            "power, never called a null result outright"
        ),
        "pooling": (
            "sessions are averaged with equal weight within their own independent unit, then independent "
            "units are averaged with equal weight, identical to results/rank_free_component_identity.json "
            "and results/alignment_below_null_diagnostic.json's own pooling"
        ),
        "no_ranking": (
            "no corpus, candidate, or estimator is described as outperforming another; only each cell's "
            "own across-half-versus-reference contrasts are reported"
        ),
    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "question": (
            "does the low early-to-late subspace overlap results/alignment_below_null_diagnostic.json "
            "reports reflect the candidate subspace rotating within a session, or does it disappear once "
            "the reference is matched to the across-half comparison on both sample size and disjointness "
            "and only temporal separation differs"
        ),
        "decision_rules_declared_before_fitting": decision_rules,
        "identity": identity,
        "data_root": str(root),
        "delivered_diagnostic_artifact": str(DELIVERED_DIAGNOSTIC_OUTPUT_PATH),
        "delivered_diagnostic_git_commit": delivered.get("git_commit"),
        "meaningful_overlap_difference": meaningful_difference,
        "chunk_sizes_trials": list(CHUNK_SIZES_TRIALS),
        "primary_chunk_size_trials": PRIMARY_CHUNK_SIZE_TRIALS,
        "cells_under_test": [
            {"corpus": corpus, "candidate": name} for corpus, names in needed_by_corpus.items() for name in names
        ],
        "cells": {},
    }
    _write(args.output, output)
    print(f"cells under test: {output['cells_under_test']}", file=sys.stderr, flush=True)

    reproduction_ok = _run(root, needed_by_corpus, identity, delivered_cells, meaningful_difference, output, args.output)

    output["reproduction_gate_passed"] = reproduction_ok
    if reproduction_ok:
        output["summary"] = _summarize(output["cells"])
        output["status"] = "complete"
    else:
        failed = [
            f"{corpus}|{name}" for corpus, candidates in output["cells"].items()
            for name, result in candidates.items() if not result["reproduction"]["matches"]
        ]
        output["summary"] = {
            "overall_statement": (
                "undetermined: reproduction of results/alignment_below_null_diagnostic.json's early-to-late "
                f"overlap did not match within tolerance for {failed}; nothing downstream is interpretable "
                "until that is resolved, so no rotation-versus-time-separation verdict is reported"
            ),
            "cells_failing_reproduction": failed,
        }
        output["status"] = "reproduction_gate_failed"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(args.output, output)
    print(f"run_subspace_rotation_time_separation finished: status={output['status']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
