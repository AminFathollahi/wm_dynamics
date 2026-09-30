"""The alignment-below-null diagnostic reports, per (corpus, candidate) cell, the overlap
between a candidate subspace fit on a session's early half and the same subspace fit on its late
half. Those overlaps are low everywhere (roughly 0.07-0.60 across the twelve cells it computed).
Low across-half overlap is consistent with the candidate subspace rotating within a session, but it
is equally consistent with pure estimation noise: each half carries only half the session's trials,
so two independent fits of a perfectly stationary subspace would also disagree. The delivered
diagnostic has no reference that separates these two accounts.

This module builds that reference. For each cell and session it reproduces the across-half overlap
(the delivered quantity, held to a reproduction gate) and contrasts it against a within-half
reference: each half is fit against an independent same-size bootstrap resample of itself, so the
reference measures how much overlap two independently fit subspaces show at the exact sample size an
across-half fit uses, with no possibility of a true rotation between them (both resamples are drawn
from the same, single, contiguous half). Averaging many resample draws keeps that reference from
being one lucky split.

Every basis-fitting primitive (block_folds, class_basis, regression_basis) and the across-half
overlap computation itself (_rotation) are imported unchanged from src/subspace_identity.py and
scripts/run_alignment_below_null_diagnostic.py; so is the equal-session-then-equal-unit pooling with
its cluster bootstrap (_pool_unit_scalars). The one new piece of math here is the within-half,
matched-sample-size bootstrap reference -- everything else is reused estimator code.
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

OUTPUT_PATH = ROOT / "results" / "subspace_rotation_versus_noise.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_subspace_rotation_versus_noise"
SCHEMA_VERSION = "1.1"
VERSION = "2026-09-18"
SEED_NAMESPACE = f"subspace_rotation_versus_noise|{VERSION}"
WITHIN_HALF_BOOTSTRAP_DRAWS = 200


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
# within-half, matched-sample-size bootstrap reference -- the one new estimand here. Two
# independent bootstrap resamples of a single half, each the same size the across-half comparison
# uses for that half, fit the same fit_basis and measured with the same subspace_overlap used
# everywhere else in this project. No rotation is possible between two resamples of one half, so
# their overlap is a pure estimation-noise floor at a matched sample size.
# ================================================================================================

def _bootstrap_self_overlap(directions: np.ndarray, target: np.ndarray, fit_basis, seed_id: str) -> dict | None:
    n = len(target)
    if n < 4:
        return None
    rng = np.random.default_rng(_seed(seed_id))
    overlaps = []
    for _ in range(WITHIN_HALF_BOOTSTRAP_DRAWS):
        draw_a = rng.integers(0, n, n)
        draw_b = rng.integers(0, n, n)
        basis_a = fit_basis(directions[draw_a], target[draw_a])
        basis_b = fit_basis(directions[draw_b], target[draw_b])
        if basis_a is None or basis_b is None:
            continue
        overlaps.append(subspace_overlap(basis_a, basis_b))
    if not overlaps:
        return None
    return {"mean_overlap": float(np.mean(overlaps)), "n_draws_used": len(overlaps), "n_samples": n}


def _within_half_reference(directions: np.ndarray, target: np.ndarray, kind: str, folds: np.ndarray, seed_prefix: str) -> dict:
    fit_basis = class_basis if kind == "categorical" else regression_basis
    early = _bootstrap_self_overlap(directions[folds == 0], target[folds == 0], fit_basis, f"{seed_prefix}|within_early")
    late = _bootstrap_self_overlap(directions[folds == 1], target[folds == 1], fit_basis, f"{seed_prefix}|within_late")
    halves = [h for h in (early, late) if h is not None]
    if not halves:
        return {"status": "not_computable", "reason": "bootstrap subspace fit failed in both halves", "early": early, "late": late}
    return {
        "status": "computed",
        "mean": float(np.mean([h["mean_overlap"] for h in halves])),
        "early": early,
        "late": late,
    }


def _cell_session(activity: np.ndarray, kind: str, target: np.ndarray, key: str) -> dict:
    prepared = _prepare_trials(activity, target)
    if prepared is None:
        return {"status": "too_few_trials"}
    directions, filtered_target = prepared
    rotation = _rotation(directions, filtered_target, kind)
    if rotation.get("status") != "computed":
        return {"status": "across_half_not_computable"}
    folds = block_folds(len(filtered_target), N_FOLDS)
    within = _within_half_reference(directions, filtered_target, kind, folds, key)
    return {
        "status": "computed",
        "across_half_overlap": rotation["subspace_overlap"],
        "n_trials_early": rotation["n_trials_early"],
        "n_trials_late": rotation["n_trials_late"],
        "within_half_reference": within,
    }


def _corpus_cells(root: Path, corpus: str, names: list[str], identity: dict) -> dict:
    per_candidate = {
        name: {
            "records": [], "n_sessions_seen": 0, "n_sessions_activity_unavailable": 0,
            "n_sessions_candidate_absent": 0, "n_sessions_too_few_trials": 0,
            "n_sessions_across_half_not_computable": 0, "n_sessions_across_half_computed": 0,
            "n_sessions_within_half_not_computable": 0, "n_sessions_within_half_computed": 0,
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
            within = entry["within_half_reference"]
            if within["status"] == "computed":
                bucket["n_sessions_within_half_computed"] += 1
            else:
                bucket["n_sessions_within_half_not_computable"] += 1
            bucket["records"].append({
                "session": session, "independent_unit": unit_id,
                "across_half_overlap": entry["across_half_overlap"],
                "n_trials_early": entry["n_trials_early"], "n_trials_late": entry["n_trials_late"],
                "within_half_reference": within,
            })
    return per_candidate


def _pool_from_records(records: list[dict], field: str, seed: str) -> dict:
    unit_values: dict[str, list[float]] = {}
    for record in records:
        unit_values.setdefault(record["independent_unit"], []).append(record[field])
    return _pool_unit_scalars(unit_values, seed)


def _pool_within_half_reference(records: list[dict], seed: str) -> dict:
    unit_values: dict[str, list[float]] = {}
    for record in records:
        within = record["within_half_reference"]
        if within["status"] == "computed":
            unit_values.setdefault(record["independent_unit"], []).append(within["mean"])
    return _pool_unit_scalars(unit_values, seed)


def _pool_contrast(records: list[dict], seed: str) -> dict:
    unit_values: dict[str, list[float]] = {}
    n_paired = 0
    for record in records:
        within = record["within_half_reference"]
        if within["status"] != "computed":
            continue
        diff = record["across_half_overlap"] - within["mean"]
        unit_values.setdefault(record["independent_unit"], []).append(diff)
        n_paired += 1
    pooled = _pool_unit_scalars(unit_values, seed)
    pooled["n_sessions_paired"] = n_paired
    return pooled


def _reproduction_check(reproduced: dict, delivered: dict) -> dict:
    if reproduced.get("status") != "computed" or delivered.get("status") != "computed":
        return {"matches": False, "reason": "delivered or reproduced pooled cell is not_computable"}
    diff = abs(float(reproduced["mean"]) - float(delivered["mean"]))
    return {"matches": diff <= REPRODUCTION_TOLERANCE, "absolute_difference": diff}


def _branch(contrast: dict, meaningful_difference: float) -> str:
    if contrast.get("status") != "computed":
        return "undetermined_below_four_independent_unit_floor"
    lo, hi = contrast["cluster_bootstrap_interval_95pct"]
    mdd = contrast["minimum_detectable_difference_80pct_power"]
    if hi < 0.0:
        return "rotates_beyond_noise"
    if lo > 0.0:
        return "undetermined_unexpected_direction"
    if mdd < meaningful_difference:
        return "noise_explains_low_overlap"
    return "undetermined_underpowered"


def _run(root: Path, needed_by_corpus: dict, identity: dict, delivered_cells: dict, output: dict, output_path: Path) -> bool:
    reproduction_ok = True
    for corpus, names in needed_by_corpus.items():
        print(f"starting corpus {corpus} ({len(names)} candidate(s))", file=sys.stderr, flush=True)
        per_candidate = _corpus_cells(root, corpus, names, identity)
        corpus_cells = {}
        for name in names:
            bucket = per_candidate[name]
            records = bucket["records"]
            reproduced = _pool_from_records(records, "across_half_overlap", f"{corpus}|{name}|reproduced_across_half_bootstrap")
            delivered = delivered_cells[corpus][name]["early_late_subspace_overlap"]
            reproduction = _reproduction_check(reproduced, delivered)
            within = _pool_within_half_reference(records, f"{corpus}|{name}|within_half_bootstrap")
            contrast = _pool_contrast(records, f"{corpus}|{name}|contrast_bootstrap")
            corpus_cells[name] = {
                "delivered_across_half_overlap": delivered,
                "reproduced_across_half_overlap": reproduced,
                "reproduction": reproduction,
                "within_half_reference": within,
                "rotation_versus_noise_contrast": contrast,
                "zero_drop": {k: v for k, v in bucket.items() if k != "records"},
            }
            if not reproduction["matches"]:
                reproduction_ok = False
        output["cells"][corpus] = corpus_cells
        _write(output_path, output)
        print(f"finished corpus {corpus}", file=sys.stderr, flush=True)
    return reproduction_ok


def _summarize(cells: dict, meaningful_difference: float) -> dict:
    table = []
    for corpus, candidates in cells.items():
        for name, result in candidates.items():
            contrast = result["rotation_versus_noise_contrast"]
            branch = _branch(contrast, meaningful_difference)
            table.append({
                "corpus": corpus, "candidate": name,
                "across_half_overlap": result["reproduced_across_half_overlap"].get("mean"),
                "within_half_reference": result["within_half_reference"].get("mean"),
                "contrast_across_minus_within": contrast.get("mean"),
                "cluster_bootstrap_interval_95pct": contrast.get("cluster_bootstrap_interval_95pct"),
                "minimum_detectable_difference_80pct_power": contrast.get("minimum_detectable_difference_80pct_power"),
                "status": contrast.get("status"),
                "branch": branch,
            })
    counts = {}
    for row in table:
        counts[row["branch"]] = counts.get(row["branch"], 0) + 1
    overall = (
        f"per-cell contrast of across-half overlap against a within-half bootstrap reference, {len(table)} cells; "
        "the reference draws two resamples from a single half and shares most trials between them, so it is not "
        "matched to the across-half comparison on sample independence -- this artifact draws no across-cell rotation "
        "conclusion; the sample-size-matched rotation test holds the sample-size- and disjointness-matched "
        "references and their own overall statement"
    )
    return {
        "cell_table": table,
        "n_cells_tested": len(table),
        "branch_counts": counts,
        "reference_independence_caveat": (
            "the within-half bootstrap reference is not matched to the across-half comparison on sample "
            "independence: its two draws are resampled from the same half's trials and mostly overlap, unlike "
            "the disjoint references in the sample-size-matched rotation test"
        ),
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
        "within_half_bootstrap_draws": WITHIN_HALF_BOOTSTRAP_DRAWS,
        "seed_namespace": SEED_NAMESPACE, "seed_algorithm": "statistics.stable_seed_crc32",
        "delivered_diagnostic_artifact_path": str(DELIVERED_DIAGNOSTIC_OUTPUT_PATH),
    }

    decision_rules = {
        "cells_under_test": (
            "every (corpus, candidate) cell the alignment-below-null diagnostic computed an "
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
            "tolerance -- if any cell fails this, that cell's within-half reference and contrast are "
            "still computed and reported, but the run-level status is reported as a failed reproduction "
            "gate and no rotation-versus-noise verdict is drawn from any cell"
        ),
        "within_half_reference_definition": (
            "for each half (early, late) separately: two independent bootstrap resamples are drawn with "
            "replacement from that half's own trials, each resample the same size as the half itself -- "
            "the same sample size an across-half fit uses for that half -- the candidate subspace is fit "
            "on each resample with the same fit_basis function the across-half comparison uses, and their "
            "overlap is measured; this is repeated over many independent resample draws and averaged, so "
            "the reference is not one lucky split. Because both resamples are drawn from a single, "
            "already-contiguous half, no true rotation can exist between them -- their overlap is a pure "
            "estimation-noise floor at the matched sample size. The within-half reference for a session "
            "is the average of this quantity for its early half and its late half"
        ),
        "matched_sample_size_definition": (
            f"each within-half bootstrap resample uses exactly as many trials as the half it is drawn "
            f"from, which is exactly the sample size the corresponding across-half fit uses for that half "
            f"-- reported per session as n_trials_early and n_trials_late, identical to the delivered "
            f"diagnostic's own fields of the same name. {WITHIN_HALF_BOOTSTRAP_DRAWS} independent resample "
            "draws are averaged per half"
        ),
        "contrast_definition": (
            "per session, across-half overlap minus the within-half reference, only for sessions where "
            "both are computed; sessions are averaged with equal weight within their own independent "
            "unit, then independent units are averaged with equal weight, and a cluster bootstrap over "
            "independent units gives the interval and the minimum detectable difference at 80% power"
        ),
        "meaningful_magnitude": (
            f"a contrast is treated as too small to matter if its minimum detectable difference at 80% "
            f"power is below {meaningful_difference:.4f} overlap units -- the median minimum detectable "
            "difference already reported for these same twelve cells' own early-to-late overlap estimates "
            "in the delivered diagnostic. A difference smaller than that would not have been distinguishable "
            "from noise even in the estimate this run exists to check in the first place, so it is not a "
            "magnitude this run can meaningfully rule in or out on its own terms"
        ),
        "rotates_beyond_noise_branch": (
            "the contrast's cluster bootstrap interval lies entirely below zero (across-half overlap "
            "reliably below the within-half reference) -- the subspace rotates within a session beyond "
            "what the matched-sample-size noise floor explains"
        ),
        "noise_explains_low_overlap_branch": (
            "the contrast's cluster bootstrap interval covers zero and its minimum detectable difference "
            "is below the pre-declared meaningful magnitude -- estimation noise at the matched sample size "
            "explains the low across-half overlap, and the delivered diagnostic's rotation reading is "
            "withdrawn for that cell"
        ),
        "undetermined_branch": (
            "the contrast's interval covers zero but its minimum detectable difference is at or above the "
            "meaningful magnitude (underpowered to rule out a difference that size), or the interval lies "
            "entirely above zero (across-half overlap reliably above the within-half reference, a direction "
            "neither account predicts), or fewer than four independent units carry both a computed "
            "across-half and within-half value -- reported as undetermined rather than forced into either "
            "of the other two branches"
        ),
        "power_reporting": (
            "a contrast that does not clear zero, or clears it but not the meaningful-magnitude bar, is "
            "reported with its cluster bootstrap interval and its minimum detectable difference at 80% "
            "power, never called a null result outright"
        ),
        "pooling": (
            "sessions are averaged with equal weight within their own independent unit, then independent "
            "units are averaged with equal weight, identical to the cross-animal component-identity "
            "estimate and the alignment-below-null diagnostic's own pooling"
        ),
        "no_ranking": (
            "no corpus, candidate, or estimator is described as outperforming another; only each cell's "
            "own across-half-versus-within-half contrast is reported"
        ),
    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "question": (
            "does the low early-to-late subspace overlap the alignment-below-null diagnostic "
            "reports reflect the candidate subspace rotating within a session, or is it explained by "
            "estimation noise at the sample size each half's fit uses"
        ),
        "decision_rules_declared_before_fitting": decision_rules,
        "identity": identity,
        "data_root": str(root),
        "delivered_diagnostic_artifact": str(DELIVERED_DIAGNOSTIC_OUTPUT_PATH),
        "delivered_diagnostic_git_commit": delivered.get("git_commit"),
        "meaningful_overlap_difference": meaningful_difference,
        "within_half_bootstrap_draws": WITHIN_HALF_BOOTSTRAP_DRAWS,
        "cells_under_test": [
            {"corpus": corpus, "candidate": name} for corpus, names in needed_by_corpus.items() for name in names
        ],
        "cells": {},
    }
    _write(args.output, output)
    print(f"cells under test: {output['cells_under_test']}", file=sys.stderr, flush=True)

    reproduction_ok = _run(root, needed_by_corpus, identity, delivered_cells, output, args.output)

    output["reproduction_gate_passed"] = reproduction_ok
    if reproduction_ok:
        output["summary"] = _summarize(output["cells"], meaningful_difference)
        output["status"] = "complete"
    else:
        failed = [
            f"{corpus}|{name}" for corpus, candidates in output["cells"].items()
            for name, result in candidates.items() if not result["reproduction"]["matches"]
        ]
        output["summary"] = {
            "overall_statement": (
                "undetermined: reproduction of the alignment-below-null diagnostic's early-to-late "
                f"overlap did not match within tolerance for {failed}; nothing downstream is interpretable "
                "until that is resolved, so no rotation-versus-noise verdict is reported"
            ),
            "cells_failing_reproduction": failed,
        }
        output["status"] = "reproduction_gate_failed"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(args.output, output)
    print(f"run_subspace_rotation_versus_noise finished: status={output['status']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
