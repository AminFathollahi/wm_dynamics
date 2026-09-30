"""Tests the one candidate results/rank_free_component_identity.json deferred: does the residual
axis identified there live in the within-trial elapsed-time subspace, rather than in a per-trial
label subspace. Every other candidate in that delivered run is a per-trial label applied to
trial-summed activity (counts.sum(axis=2)); elapsed time instead varies across the bin axis that
sum collapses away, so it needs its own unit of observation (a (trial, bin) sample, not a trial)
and its own permutation null (one that cannot destroy the within-trial temporal structure a real
time label has, unlike a null that permutes freely across trials).

Every basis-fitting and cross-fitting primitive below (block_folds, residual_axis, regression_
basis, crossfit_alignment) is imported unchanged from src/subspace_identity.py, the same module
results/rank_free_component_identity.json's own estimator uses. The one new piece of math here is
the null in _time_permutation_alignment: independent circular shifts of the time label within each
trial's own valid bins, not a free permutation across all (trial, bin) samples. A free permutation
reassigns each trial's bins a near-arbitrary subset of the pooled label pool, discarding the one
thing every real elapsed-time label has that a scrambled one does not: a smooth, monotonic,
one-value-per-bin ordering within each trial. If the underlying activity is itself autocorrelated
within a trial for reasons that have nothing to do with elapsed time (state persistence, a shared
trial-level offset), a null built from scrambled labels is not drawing from the same family of
labels a genuine but non-causal elapsed-time label would produce, so it is not the calibrated
comparison. A circular shift keeps each trial's own real time-value multiset and its smooth,
one-jump-per-shift ordering intact while breaking its correspondence to the real bin order --
exactly the substitution a genuinely non-causal time label could have realised (a different, but
still smooth and complete, phase alignment to the same trial) -- which is the calibrated null for
this candidate; see the synthetic check in null_calibration_check and NULL_CONSTRUCTION_RULE below.

The alternative this module rejects -- permuting by swapping whole per-trial time-course vectors
between trials -- is degenerate here: every corpus below cuts a FIXED-length, FIXED-bin-width delay
window (verified against each loader: run_state_behavior_link.BIN_MS/PANICHELLO_DELAY_WINDOW_MS,
run_state_content_link.delay_counts's bin_ms=100.0 default, the mouse ALM loader's bin_ms=100.0
literal, and load_watters_session's bin_ms=100.0 default), so every trial's own canonical time
course is the identical vector [0, 0.1, 0.2, ...] seconds. Swapping that vector between trials
changes nothing -- a no-op with zero null variance -- so the within-trial circular shift is the
only nondegenerate choice.
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

from corpus_sessions import (  # noqa: E402
    data_root, independent_unit, iter_dandi_000469, iter_dandi_000574, iter_dandi_001187,
    load_watters_session, watters_behaviour, watters_session_dates,
)
from provenance import _json_safe, git_commit  # noqa: E402
from run_component_identity_subspace_atlas import _alm_session_inputs, _human_session_covariate_inputs
from corpus_sessions import _panichello_session_inputs
from info_decoding import MAX_SESSIONS_ENV_VAR
from statistics import Z_80_POWER
from run_rank_free_component_identity import N_PERM
from info_decoding import FDR_ALPHA, MIN_INDEPENDENT_UNITS, N_BOOT, N_FOLDS
from statistics import fdr_bh, stable_seed  # noqa: E402
from subspace_identity import block_folds, crossfit_alignment, regression_basis, residual_axis  # noqa: E402
from corpus_sessions import _synthetic_time_independent_counts  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "within_trial_time_component_identity.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_within_trial_time_component_identity"
SCHEMA_VERSION = "1.0"
VERSION = "2026-09-18"
SEED_NAMESPACE = f"within_trial_time_component_identity|{VERSION}"
CANDIDATE = "time_in_trial_within_delay"
BIN_WIDTH_S = 0.1
CORPORA = (
    "panichello_2024_macaque_lPFC", "watters_2026_macaque_multi_object", "inagaki_alm5_mouse_ALM",
    "dandi_000469_human", "dandi_001187_human", "dandi_000574_human",
)

DECISION_RULES = {
    "label": (
        "elapsed time within the maintenance window, restricted to each corpus's own delay epoch, "
        "as the bin index of that corpus's own fixed-width delay-epoch spike-count tensor times "
        f"{BIN_WIDTH_S} s (every corpus below bins its delay epoch at 100 ms, verified against each "
        "loader); treated as continuous"
    ),
    "unit_of_observation": (
        "a (trial, bin) sample, not a trial -- the residual axis and label basis are fit on "
        "unit-normalised per-bin activity vectors, exactly as the delivered estimator's own _cell "
        "fits them on unit-normalised per-trial activity vectors; a (trial, bin) sample is dropped "
        "if its activity vector is non-finite or has zero norm across units, matching that same rule"
    ),
    "fold_assignment": (
        "folds are assigned at the TRIAL level (block_folds over trial order) and every one of a "
        "trial's own bins inherits its trial's fold, so no trial ever has some bins in the training "
        "fold and others in the evaluation fold -- required for the cross-fitting to hold at the "
        "trial level, the level at which a single population trajectory is drawn"
    ),
    "null_construction": (
        "for each permutation draw, independently circular-shift each trial's own sequence of "
        "(kept-bin) time values by a uniform random amount in [0, n_kept_bins_this_trial); the "
        "residual axis is fixed per fold (from the real trial partition, never refit under the "
        "null) and the label basis (regression_basis) is refit on the shifted labels every draw, "
        "identical in structure to the delivered estimator's own permutation_alignment except for "
        "this one substitution -- see module docstring for why a free permutation over all samples "
        "and a whole-trial time-course swap were both rejected"
    ),
    "independent_unit_floor": (
        f"a corpus's cell is not_computable unless at least {MIN_INDEPENDENT_UNITS} independent "
        "units each carry a computed session cell, matching results/rank_free_component_identity."
        "json's own floor so the two artifacts are comparable; the floor is never lowered, and a "
        "corpus with fewer independent units is reported not_computable with that reason, never "
        "pooled with another corpus"
    ),
    "pooling": (
        "sessions are averaged with equal weight within their own independent unit, then "
        "independent units are averaged with equal weight -- identical to results/rank_free_"
        "component_identity.json's own _pool"
    ),
    "interval_and_power": (
        "a cluster bootstrap (resampling independent units with replacement, "
        f"{N_BOOT} draws) gives both the 95% interval on the pooled alignment-above-null effect and "
        f"its standard error; the minimum detectable difference at 80% power is {Z_80_POWER} times "
        "that bootstrap standard error (the z multiplier this project's own component-identity atlas "
        "already fixed for this exact purpose, imported unchanged, never re-derived here); a cell "
        "that does not clear its own refit null is reported with this interval and detectable "
        "difference, never called a null result"
    ),
}


def _hash() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "src" / "subspace_identity.py", ROOT / "src" / "corpus_sessions.py",
        ROOT / "src" / "statistics.py", ROOT / "src" / "info_decoding.py", ROOT / "src" / "provenance.py",
        ROOT / "scripts" / "run_component_identity_subspace_atlas.py",
        ROOT / "scripts" / "run_rank_free_component_identity.py",
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
            if cached.get("identity") == identity and cached.get("complete") is True and isinstance(record, dict) and required.issubset(record):
                return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    record = fit()
    _write(path, {"complete": True, "identity": identity, "record": record})
    return record


# ================================================================================================
# Bin-level sample assembly: the unit of observation is a (trial, bin) sample.
# ================================================================================================

def _bin_samples(counts: np.ndarray) -> dict | None:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3:
        return None
    n_trials, n_units, n_bins = counts.shape
    if n_trials < 2 * max(3, N_FOLDS) or n_bins < 2:
        return None
    trial_folds = block_folds(n_trials, N_FOLDS)
    flat_activity = counts.transpose(0, 2, 1).reshape(n_trials * n_bins, n_units)
    flat_trial = np.repeat(np.arange(n_trials), n_bins)
    flat_bin = np.tile(np.arange(n_bins), n_trials)
    flat_time = flat_bin.astype(float) * BIN_WIDTH_S
    flat_fold = trial_folds[flat_trial]
    norms = np.linalg.norm(flat_activity, axis=1)
    valid = np.isfinite(flat_activity).all(axis=1) & (norms > 0)
    n_total = int(valid.size)
    n_dropped = int((~valid).sum())
    if int(valid.sum()) < 2 * max(3, N_FOLDS):
        return None
    directions = flat_activity[valid] / norms[valid, None]
    target = flat_time[valid]
    trial_index = flat_trial[valid]
    folds = flat_fold[valid]
    trial_positions = {int(trial): np.flatnonzero(trial_index == trial) for trial in np.unique(trial_index)}
    if np.unique(folds).size < N_FOLDS:
        return None
    return {
        "directions": directions, "target": target, "folds": folds, "trial_positions": trial_positions,
        "n_trials": n_trials, "n_bins": n_bins,
        "n_bin_samples_total": n_total, "n_bin_samples_used": int(valid.sum()),
        "n_bin_samples_dropped_nonfinite_or_zero_norm": n_dropped,
    }


def _circular_shift_draw_source(target: np.ndarray, trial_positions: dict, rng: np.random.Generator):
    """A callable that returns one circular-shift permutation of target per call. Every trial with
    the SAME kept-bin count (the common case: corpora drop very few (trial, bin) samples) is handled
    by one vectorised batched roll over a (trials, bins) grid instead of a Python loop per trial per
    draw -- the dominant cost for large-trial sessions (e.g. the multi-object corpus, 1000+ trials).
    Any trial with a different kept-bin count (rare) falls back to the exact per-trial roll."""
    trial_ids = list(trial_positions)
    sizes = np.asarray([len(trial_positions[t]) for t in trial_ids])
    regular_mask = sizes == np.bincount(sizes).argmax()
    regular_ids = [t for t, keep in zip(trial_ids, regular_mask) if keep]
    irregular_positions = [trial_positions[t] for t, keep in zip(trial_ids, regular_mask) if not keep]
    if regular_ids:
        n_kept = len(trial_positions[regular_ids[0]])
        order = np.concatenate([trial_positions[t] for t in regular_ids])
        grid = target[order].reshape(len(regular_ids), n_kept)
        col = np.arange(n_kept)

    def draw_once() -> np.ndarray:
        permuted = target.copy()
        if regular_ids and n_kept > 1:
            shifts = rng.integers(0, n_kept, size=grid.shape[0])
            idx = (col[None, :] - shifts[:, None]) % n_kept
            permuted[order] = np.take_along_axis(grid, idx, axis=1).reshape(-1)
        for positions in irregular_positions:
            n = len(positions)
            if n < 2:
                continue
            shift = int(rng.integers(0, n))
            permuted[positions] = target[positions][(np.arange(n) - shift) % n]
        return permuted

    return draw_once


def _time_permutation_alignment(
    directions: np.ndarray, target: np.ndarray, folds: np.ndarray, trial_positions: dict,
    n_perm: int, rng: np.random.Generator,
) -> dict:
    axes = {}
    for fold in np.unique(folds):
        test = folds == fold
        axes[fold] = residual_axis(directions[~test], directions[test])
    observed = crossfit_alignment(directions, target, folds, regression_basis, axes)
    if observed["status"] != "computed":
        return observed
    draw_source = _circular_shift_draw_source(target, trial_positions, rng)
    draws = np.empty(n_perm, dtype=float)
    valid = np.ones(n_perm, dtype=bool)
    for draw in range(n_perm):
        permuted = draw_source()
        result = crossfit_alignment(directions, permuted, folds, regression_basis, axes)
        if result["status"] == "computed":
            draws[draw] = result["alignment"]
        else:
            valid[draw] = False
    draws = draws[valid]
    if draws.size == 0:
        return {"status": "not_computable", "n_folds": observed["n_folds"], "reason": "no permutation draw was computable"}
    score = observed["alignment"]
    return {
        **observed,
        "null_mean": float(draws.mean()),
        "alignment_above_null": float(score - draws.mean()),
        "p_value": float((1 + np.sum(draws >= score)) / (draws.size + 1)),
        "n_permutations": int(draws.size),
        "null_interval_95pct": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
        "null_draws": draws,
    }


def _session_cell(counts: np.ndarray, n_perm: int, seed: str) -> dict:
    samples = _bin_samples(counts)
    if samples is None:
        return {"status": "not_computable", "reason": "too few finite, nonzero-norm (trial, bin) samples, or too few trials/bins"}
    rng = np.random.default_rng(_seed(seed))
    result = _time_permutation_alignment(
        samples["directions"], samples["target"], samples["folds"], samples["trial_positions"], n_perm, rng,
    )
    for key in ("n_trials", "n_bins", "n_bin_samples_total", "n_bin_samples_used", "n_bin_samples_dropped_nonfinite_or_zero_norm"):
        result[key] = samples[key]
    result["target_kind"] = "continuous"
    result["seed_id"] = seed
    return result


# ================================================================================================
# Synthetic null-calibration check (design rule 3): activity generated with real within-trial
# autocorrelation (a per-trial, per-unit Poisson rate shared across that trial's own bins) but with
# NO time dependence, so the measured false-positive rate at p<=0.05 is the calibration number.
# ================================================================================================



def null_calibration_check(n_replicates: int, n_perm: int, seed: str,
                            n_trials: int = 24, n_units: int = 10, n_bins: int = 8) -> dict:
    rng = np.random.default_rng(_seed(seed))
    n_computed = 0
    n_significant = 0
    for replicate in range(n_replicates):
        counts = _synthetic_time_independent_counts(rng, n_trials, n_units, n_bins)
        cell = _session_cell(counts, n_perm, f"{seed}|replicate={replicate}")
        if cell.get("status") != "computed":
            continue
        n_computed += 1
        if cell["p_value"] <= 0.05:
            n_significant += 1
    rate = (n_significant / n_computed) if n_computed else None
    return {
        "n_replicates_requested": n_replicates,
        "n_replicates_computed": n_computed,
        "n_perm_per_replicate": n_perm,
        "false_positive_rate_at_p_0.05": rate,
        "calibrated": bool(rate is not None and abs(rate - 0.05) <= 0.03),
        "generative_model": (
            "per (trial, unit) a Poisson rate drawn once per trial (shared across every bin of that "
            "trial, giving real within-trial autocorrelation unrelated to elapsed time), spike counts "
            "drawn i.i.d. Poisson per bin from that trial's own rate -- activity is, by construction, "
            "independent of the bin/time index"
        ),
    }


# ================================================================================================
# Per-corpus session sources -- reuses the delivered estimator's own per-session generators
# (raw, bin-resolved counts, not yet summed over the bin axis) so trial admission is identical to
# results/rank_free_component_identity.json's own.
# ================================================================================================

def _standard_time_sessions(root: Path, corpus: str):
    if corpus == "panichello_2024_macaque_lPFC":
        generator = _panichello_session_inputs(root)
    elif corpus == "inagaki_alm5_mouse_ALM":
        generator = _alm_session_inputs(root)
    else:
        generator = None
    if generator is not None:
        for session, reason, core, _categorical, _continuous, counts in generator:
            valid = core is not None and counts is not None
            yield session, (None if valid else reason), (np.asarray(counts) if valid else None)
        return
    iterator = {
        "dandi_000469_human": iter_dandi_000469, "dandi_001187_human": iter_dandi_001187,
        "dandi_000574_human": iter_dandi_000574,
    }[corpus]
    for session, core, _categorical, _continuous, counts in _human_session_covariate_inputs(root, iterator, corpus):
        valid = core is not None and counts is not None
        yield session, (None if valid else "session geometry unavailable"), (np.asarray(counts) if valid else None)


def _watters_time_sessions(root: Path):
    behaviour = watters_behaviour(root)
    dates = watters_session_dates(root)
    limit = os.environ.get(MAX_SESSIONS_ENV_VAR)
    if limit:
        dates = dates[:int(limit)]
    for animal, date, variant in dates:
        session_id = f"{animal}_{date}_{variant}"
        session = load_watters_session(root, animal, date, behaviour)
        if session.get("status") != "loaded":
            yield session_id, animal, session.get("status"), None
        else:
            yield session_id, animal, None, np.asarray(session["counts"])


def _collect_standard(root: Path, corpus: str, n_perm: int, identity: dict) -> list[dict]:
    records = []
    for session, reason, counts in _standard_time_sessions(root, corpus):
        key = f"{corpus}|{session}"
        unit_id = independent_unit(corpus, session)

        def fit(session=session, reason=reason, counts=counts, unit_id=unit_id, key=key):
            if counts is None:
                return {"session": session, "independent_unit": unit_id, "status": "not_computable", "reason": reason}
            cell = _session_cell(counts, n_perm, key)
            return {
                "session": session, "independent_unit": unit_id, "status": "computed",
                "candidates": {CANDIDATE: cell},
            }

        records.append(_checkpoint(key, identity, fit))
    return records


def _collect_watters(root: Path, n_perm: int, identity: dict) -> list[dict]:
    records = []
    for session_id, animal, reason, counts in _watters_time_sessions(root):
        key = f"watters_2026_macaque_multi_object|{session_id}"

        def fit(session_id=session_id, animal=animal, reason=reason, counts=counts, key=key):
            if counts is None:
                return {"session": session_id, "independent_unit": animal, "status": "not_computable", "reason": reason}
            cell = _session_cell(counts, n_perm, key)
            return {
                "session": session_id, "independent_unit": animal, "status": "computed",
                "candidates": {CANDIDATE: cell},
            }

        records.append(_checkpoint(key, identity, fit))
    return records


def _strip(record: dict) -> dict:
    clean = dict(record)
    if "candidates" in clean:
        clean["candidates"] = {
            name: {key: value for key, value in cell.items() if key != "null_draws"}
            for name, cell in record["candidates"].items()
        }
    return clean


def _pool(records: list[dict], seed: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        cell = record.get("candidates", {}).get(CANDIDATE, {})
        if cell.get("status") == "computed":
            grouped.setdefault(record["independent_unit"], []).append(cell)
    n_sessions = sum(len(cells) for cells in grouped.values())
    n_sessions_with_session_geometry = sum(1 for record in records if record.get("status") == "computed")
    if len(grouped) < MIN_INDEPENDENT_UNITS:
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
        unit_draws.append(np.mean(np.stack([np.asarray(cell["null_draws"])[:n_draws] for cell in cells]), axis=0))
    observed = float(np.mean(unit_observed))
    draws = np.mean(np.stack(unit_draws), axis=0)
    effects = np.asarray([value - null.mean() for value, null in zip(unit_observed, unit_draws)])
    rng = np.random.default_rng(_seed(seed))
    boot = np.asarray([effects[rng.integers(0, len(effects), len(effects))].mean() for _ in range(N_BOOT)])
    bootstrap_se = float(np.std(boot, ddof=1))
    return {
        "status": "computed",
        "n_sessions": n_sessions,
        "n_sessions_with_session_geometry": n_sessions_with_session_geometry,
        "n_independent_units": len(grouped),
        "independent_unit_ids": sorted(grouped),
        "session_weighting_within_unit": "equal",
        "independent_unit_weighting": "equal",
        "mean_alignment": observed,
        "mean_null_alignment": float(draws.mean()),
        "mean_alignment_above_null": float(effects.mean()),
        "cluster_bootstrap_interval_95pct": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
        "cluster_bootstrap_standard_error": bootstrap_se,
        "minimum_detectable_difference_80pct_power": float(Z_80_POWER * bootstrap_se),
        "p_value": float((1 + np.sum(draws >= observed)) / (len(draws) + 1)),
        "n_permutations": len(draws),
        "bootstrap_seed_id": seed,
    }


def _summarize(corpus_records: dict[str, list[dict]]) -> dict:
    per_corpus = {corpus: {CANDIDATE: _pool(records, f"{corpus}|{CANDIDATE}|bootstrap")} for corpus, records in corpus_records.items()}
    tested = [(corpus, cell) for corpus, candidates in per_corpus.items() for cell in candidates.values() if cell["status"] == "computed"]
    if tested:
        correction = fdr_bh(np.asarray([cell["p_value"] for _, cell in tested]), alpha=FDR_ALPHA)
        for (_, cell), reject, q_value in zip(tested, correction["reject"], correction["q_values"]):
            cell["fdr_q_value"] = float(q_value)
            cell["clears_own_refit_null"] = bool(reject and cell["mean_alignment_above_null"] > 0)
    return {
        "per_corpus": per_corpus,
        "fdr_scope": "every computed corpus cell for this one candidate",
        "n_corpora_attempted": len(corpus_records),
        "n_corpora_computed": len(tested),
        "corpora_clearing_own_refit_null": sorted(corpus for corpus, cell in tested if cell.get("clears_own_refit_null")),
    }


def main() -> None:
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--calibration-replicates", type=int, default=200)
    parser.add_argument("--calibration-n-perm", type=int, default=200)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    CHECKPOINT_DIR = args.checkpoint_dir
    if args.n_perm < 1:
        parser.error("--n-perm must be positive")
    if args.max_sessions is not None:
        if args.max_sessions < 1:
            parser.error("--max-sessions must be positive")
        os.environ[MAX_SESSIONS_ENV_VAR] = str(args.max_sessions)
    started = time.time()
    root = data_root()
    identity = {
        "version": VERSION, "schema_version": SCHEMA_VERSION, "code_hash": _hash(),
        "data_root": str(root.resolve()), "n_perm": args.n_perm, "n_folds": N_FOLDS,
        "seed_namespace": SEED_NAMESPACE, "seed_algorithm": "statistics.stable_seed_crc32",
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "candidate": CANDIDATE,
        "estimand": (
            "the same cross-fitted, squared, held-out subspace-alignment estimand results/rank_free_"
            "component_identity.json uses for its per-trial candidates, applied with a (trial, bin) "
            "sample as the unit of observation instead of a trial"
        ),
        "decision_rules_declared_before_fitting": DECISION_RULES,
        "identity": identity,
        "data_root": str(root),
        "corpora": {},
    }
    _write(args.output, output)
    print(f"running synthetic null calibration check ({args.calibration_replicates} replicates)", file=sys.stderr, flush=True)
    output["null_calibration_check"] = null_calibration_check(
        args.calibration_replicates, args.calibration_n_perm, "null_calibration",
    )
    _write(args.output, output)
    print(f"measured false-positive rate: {output['null_calibration_check']['false_positive_rate_at_p_0.05']}", file=sys.stderr, flush=True)

    raw_records = {}
    for corpus in CORPORA:
        print(f"starting corpus {corpus}", file=sys.stderr, flush=True)
        if corpus == "watters_2026_macaque_multi_object":
            records = _collect_watters(root, args.n_perm, identity)
        else:
            records = _collect_standard(root, corpus, args.n_perm, identity)
        raw_records[corpus] = records
        output["corpora"][corpus] = {"records": [_strip(record) for record in records]}
        _write(args.output, output)
        print(f"finished corpus {corpus}: {len(records)} sessions seen", file=sys.stderr, flush=True)

    output["summary"] = _summarize(raw_records)
    output["status"] = "complete"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(args.output, output)
    print("run_within_trial_time_component_identity finished", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
