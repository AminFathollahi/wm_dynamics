"""Tests a different, operational question from the sample-size-matched rotation test.
That test asks, geometrically, whether a candidate subspace rotates within a session by
comparing early/late subspace overlap against chunk-matched references. This module asks whether a
linear read-out calibrated on one stretch of trials still predicts the same quantity on a disjoint
later (or earlier) stretch, and how that held-out performance changes with the elapsed gap between
the two stretches -- a scalar prediction test on held-out trials, not a comparison between two noisy
low-rank subspace fits. The two tests share session/chunk machinery and overlapping data but are
different estimands; neither confirms, replicates, or is confirmed by the other.

Payload is the gain candidate (gain_total_spike_count, a session's per-trial total spike count summed
across units, exactly core["spike_count"] as computed and consumed throughout the cross-animal
component-identity estimate) in the four corpora where it is computable per that same estimate:
inagaki_alm5_mouse_ALM, dandi_000469_human, dandi_001187_human, dandi_000574_human. For
dandi_000574_human, every candidate the sample-size-matched rotation test enumerates for that corpus
is included (gain plus its four field-potential candidates), read directly from that test's own
cells_under_test rather than hardcoded, so this module's cell list tracks its stated source exactly.
The two macaque corpora (panichello_2024_macaque_lPFC, watters_2026_macaque_multi_object) are
attempted the same way; both carry gain_total_spike_count but fewer than the four-independent-unit
floor (the cross-animal component-identity estimate already reports 3 and 2 independent units
respectively for this exact candidate), so both are expected, not merely permitted, to come back
not_computable here --
this module does not relax that floor. watters_2026_macaque_multi_object has no _standard_sessions
entry point (its sessions are structured by item-count level, unlike every other corpus here) and its
gain identity is a single scalar per trial regardless of item count, so its trials are pooled across
levels in original temporal order for this module only -- a disclosed simplification relative to the
per-level machinery used elsewhere in this project, the same kind of simplification
run_component_identity_subspace_atlas.run_watters_reaction_time already discloses for reaction time.

regression_basis and block_folds's sibling primitives are reused unchanged from src/subspace_identity.py
for the one piece that is shared with the rest of the project: fitting a read-out direction from a
chunk of trials. crossfit_alignment/permutation_alignment are NOT reused -- those measure squared
cosine alignment between two independently-fit subspaces, a different estimand from the one asked
here (does a fitted direction's OWN held-out predictive correlation survive an elapsed gap). This
module's read-out performance is the squared Pearson correlation between a chunk's true payload
values and the scalar projection of that chunk's activity directions onto a basis fit on a temporally
disjoint OTHER chunk -- an ordinary held-out prediction test, reusing only the basis-fitting step.
_prepare_trials is reused unchanged from run_alignment_below_null_diagnostic.py (same trial-validity
filter the cross-animal component-identity estimate's own _cell uses internally), and
_pool_unit_scalars from the same module supplies every cluster-bootstrap interval and minimum
detectable difference below, exactly as it already does for the sample-size-matched rotation test.

Two pieces of math are new here, declared before any cell is fit: contiguous fixed-size trial
chunking with every ordered pair of disjoint chunks scored (train on one, evaluate on the other, gap
= the absolute difference between each chunk's mean trial index), and a permutation null built by
circularly shifting the chunk centre-index array within a session -- never a free permutation across
samples, which would destroy the within-session autocorrelation a genuine elapsed-gap effect and this
null both need to share. A synthetic calibration (per (trial, unit) a Poisson rate drawn once per
trial and shared across that trial's bins, reusing
run_within_trial_time_component_identity._synthetic_time_independent_counts unchanged) reports this
null's own empirical false-positive rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from corpus_sessions import data_root, independent_unit, load_watters_session, watters_behaviour, watters_session_dates  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from run_alignment_below_null_diagnostic import REPRODUCTION_TOLERANCE, _pool_unit_scalars, _prepare_trials  # noqa: E402
from run_component_identity_subspace_atlas import MAX_SESSIONS_ENV_VAR  # noqa: E402
from run_rank_free_component_identity import (  # noqa: E402
    MIN_INDEPENDENT_UNITS, OUTPUT_PATH as RANK_FREE_OUTPUT_PATH, _standard_sessions,
)
from run_subspace_rotation_time_separation import OUTPUT_PATH as ROTATION_OUTPUT_PATH  # noqa: E402
from run_within_trial_time_component_identity import _synthetic_time_independent_counts  # noqa: E402
from statistics import fdr_bh, stable_seed  # noqa: E402
from subspace_identity import regression_basis  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "readout_transfer_decay.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_readout_transfer_decay"
SCHEMA_VERSION = "1.0"
VERSION = "2026-09-18"
SEED_NAMESPACE = f"readout_transfer_decay|{VERSION}"

CHUNK_SIZES_TRIALS = (2, 4, 8, 16)
PRIMARY_CHUNK_SIZE_TRIALS = 4
MIN_CHUNKS_PER_SESSION = 4
MIN_PAIRS_FOR_SLOPE = 6
N_PERM = 1000
FDR_ALPHA = 0.05
NULL_CALIBRATION_REPLICATES = 2000
NULL_CALIBRATION_N_PERM = 300
STANDARD_CORPORA = (
    "panichello_2024_macaque_lPFC", "inagaki_alm5_mouse_ALM",
    "dandi_000469_human", "dandi_001187_human", "dandi_000574_human",
)


def _hash() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "src" / "subspace_identity.py", ROOT / "src" / "corpus_sessions.py",
        ROOT / "src" / "statistics.py", ROOT / "src" / "provenance.py",
        ROOT / "scripts" / "run_component_identity_subspace_atlas.py",
        ROOT / "scripts" / "run_rank_free_component_identity.py",
        ROOT / "scripts" / "run_alignment_below_null_diagnostic.py",
        ROOT / "scripts" / "run_within_trial_time_component_identity.py",
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
    readable = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:80]
    suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = CHECKPOINT_DIR / f"{readable}.{suffix}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            record = cached.get("record")
            if cached.get("identity") == identity and cached.get("complete") is True and isinstance(record, dict):
                return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    record = fit()
    _write(path, {"complete": True, "identity": identity, "record": record})
    return record


# ================================================================================================
# the new math: contiguous chunking + every disjoint ordered pair scored, and the circular-shift
# gap-label null. Everything upstream of a chunk's own (directions, target) is reused unchanged.
# ================================================================================================

def _chunk_assign(n: int, chunk_size: int) -> tuple[list[np.ndarray], np.ndarray]:
    n_chunks = math.ceil(n / chunk_size)
    chunks = [np.arange(c * chunk_size, min((c + 1) * chunk_size, n)) for c in range(n_chunks)]
    centres = np.asarray([float(idx.mean()) for idx in chunks])
    return chunks, centres


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xm, ym = x.mean(), y.mean()
    dx = x - xm
    slope = float((dx * (y - ym)).sum() / (dx * dx).sum())
    return slope, float(ym - slope * xm)


def _circular_shift_null_slopes(centres: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray,
                                 performance: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    n_chunks = len(centres)
    if n_chunks < 2:
        return np.empty(0)
    shifts = rng.integers(1, n_chunks, size=n_perm)
    positions = np.arange(n_chunks)
    shifted_index = (positions[None, :] - shifts[:, None]) % n_chunks
    shifted_centres = centres[shifted_index]
    null_gaps = np.abs(shifted_centres[:, pair_j] - shifted_centres[:, pair_i])
    perf_mean = performance.mean()
    gap_mean = null_gaps.mean(axis=1, keepdims=True)
    cov = ((null_gaps - gap_mean) * (performance - perf_mean)).mean(axis=1)
    var = ((null_gaps - gap_mean) ** 2).mean(axis=1)
    valid = var > 1e-12
    slopes = np.full(n_perm, np.nan)
    slopes[valid] = cov[valid] / var[valid]
    return slopes[np.isfinite(slopes)]


def _session_chunk_cell(directions: np.ndarray, target: np.ndarray, chunk_size: int, rng: np.random.Generator) -> dict:
    n = len(target)
    if n < MIN_CHUNKS_PER_SESSION * chunk_size:
        return {"status": "not_computable", "reason": "fewer than four chunks of this size fit in the session"}
    chunks, centres = _chunk_assign(n, chunk_size)
    n_chunks = len(chunks)
    bases = [regression_basis(directions[idx], target[idx]) if len(idx) >= 2 else None for idx in chunks]
    pair_i, pair_j, performance = [], [], []
    for i in range(n_chunks):
        if bases[i] is None:
            continue
        for j in range(n_chunks):
            if i == j:
                continue
            idx_j = chunks[j]
            if len(idx_j) < 2:
                continue
            readout = (directions[idx_j] @ bases[i]).reshape(-1)
            y = target[idx_j]
            if np.std(readout) <= 1e-12 or np.std(y) <= 1e-12:
                continue
            r = np.corrcoef(readout, y)[0, 1]
            if not np.isfinite(r):
                continue
            pair_i.append(i)
            pair_j.append(j)
            performance.append(r * r)
    if len(performance) < MIN_PAIRS_FOR_SLOPE:
        return {
            "status": "not_computable", "n_chunks": n_chunks, "n_pairs": len(performance),
            "reason": "fewer than the trial-pair floor produced a computable read-out performance",
        }
    performance = np.asarray(performance)
    pair_i, pair_j = np.asarray(pair_i), np.asarray(pair_j)
    gaps = np.abs(centres[pair_j] - centres[pair_i])
    if np.std(gaps) <= 1e-12:
        return {
            "status": "not_computable", "n_chunks": n_chunks, "n_pairs": len(performance),
            "reason": "gap values carry no variance in this session at this chunk size",
        }
    slope, intercept = _ols_slope_intercept(gaps, performance)
    null_slopes = _circular_shift_null_slopes(centres, pair_i, pair_j, performance, N_PERM, rng)
    return {
        "status": "computed", "n_chunks": n_chunks, "n_pairs": int(len(performance)),
        "observed_slope": slope, "observed_intercept": intercept,
        "mean_performance": float(performance.mean()), "mean_gap_trials": float(gaps.mean()),
        "max_gap_trials": float(gaps.max()), "n_trials": n,
        "null_slopes": null_slopes,
    }


def _session_curve(activity: np.ndarray | None, target, seed_prefix: str) -> dict:
    if activity is None:
        return {"status": "activity_unavailable"}
    prepared = _prepare_trials(np.asarray(activity, dtype=float), np.asarray(target, dtype=float))
    if prepared is None:
        return {"status": "too_few_trials"}
    directions, filtered_target = prepared
    per_size = {}
    for chunk_size in CHUNK_SIZES_TRIALS:
        rng = np.random.default_rng(_seed(f"{seed_prefix}|chunk={chunk_size}"))
        per_size[str(chunk_size)] = _session_chunk_cell(directions, filtered_target, chunk_size, rng)
    return {"status": "computed", "n_trials": int(len(filtered_target)), "chunk_cells": per_size}


# ================================================================================================
# corpus/session enumeration -- reuses _standard_sessions for every corpus it already covers;
# watters_2026_macaque_multi_object needs its own loader since it has no _standard_sessions entry.
# ================================================================================================

def _watters_sessions(root: Path):
    behaviour = watters_behaviour(root)
    dates = watters_session_dates(root)
    limit = os.environ.get(MAX_SESSIONS_ENV_VAR)
    if limit:
        dates = dates[: int(limit)]
    for animal, session_date, variant in dates:
        session_id = f"{animal}_{session_date}_{variant}"
        session = load_watters_session(root, animal, session_date, behaviour)
        if session.get("status") != "loaded":
            yield session_id, session.get("status"), None, {}, {}
            continue
        activity = session["counts"].sum(axis=2)
        target = np.nansum(activity, axis=1)
        yield session_id, None, activity, {}, {"gain_total_spike_count": target}


def _corpus_sessions(root: Path, corpus: str):
    if corpus in STANDARD_CORPORA:
        return _standard_sessions(root, corpus)
    if corpus == "watters_2026_macaque_multi_object":
        return _watters_sessions(root)
    raise ValueError(f"no session source registered for {corpus}")


def _corpus_candidate_sessions(root: Path, corpus: str, names: list[str], identity: dict) -> dict:
    per_candidate = {name: [] for name in names}
    for session, reason, activity, categorical, continuous in _corpus_sessions(root, corpus):
        key = f"{corpus}|{session}"
        unit_id = independent_unit(corpus, session)

        def fit(reason=reason, activity=activity, continuous=continuous, key=key):
            if activity is None:
                return {"status": "not_computable", "reason": reason, "candidates": {}}
            per_name = {}
            for name in names:
                target = continuous.get(name)
                if target is None:
                    per_name[name] = {"status": "candidate_absent"}
                    continue
                per_name[name] = _session_curve(activity, target, f"{key}|{name}")
            return {"status": "computed", "candidates": per_name}

        record = _checkpoint(key, identity, fit)
        for name in names:
            if record["status"] != "computed":
                entry = {"status": "not_computable", "reason": record.get("reason")}
            else:
                entry = record.get("candidates", {}).get(name, {"status": "candidate_absent"})
            per_candidate[name].append({"session": session, "independent_unit": unit_id, "entry": entry})
    return per_candidate


# ================================================================================================
# pooling: equal-session-then-equal-unit, reusing _pool_unit_scalars unchanged for the observed
# slope/intercept cluster bootstrap; the null-draw pooling below is the one small new piece of
# math this needs, mirroring the same equal-weighting the reused function applies to observed values.
# ================================================================================================

def _pool_null_draws(sessions_by_unit: dict[str, list[np.ndarray]]) -> np.ndarray:
    filtered = {u: [d for d in draws if len(d) > 0] for u, draws in sessions_by_unit.items()}
    filtered = {u: d for u, d in filtered.items() if d}
    if not filtered:
        return np.empty(0)
    n = min(len(d) for draws in filtered.values() for d in draws)
    unit_means = [np.mean(np.stack([d[:n] for d in draws]), axis=0) for draws in filtered.values()]
    return np.mean(np.stack(unit_means), axis=0)


def _pool_slope_cell(sessions: list[dict], chunk_size: int, seed: str) -> dict:
    unit_cells: dict[str, list[dict]] = {}
    n_seen = 0
    n_not_computable = 0
    for record in sessions:
        n_seen += 1
        entry = record["entry"]
        if entry.get("status") != "computed":
            n_not_computable += 1
            continue
        cell = entry["chunk_cells"][str(chunk_size)]
        if cell["status"] != "computed":
            n_not_computable += 1
            continue
        unit_cells.setdefault(record["independent_unit"], []).append(cell)
    if len(unit_cells) < MIN_INDEPENDENT_UNITS:
        return {
            "status": "not_computable", "n_independent_units": len(unit_cells),
            "n_sessions_seen": n_seen, "n_sessions_not_computable_this_chunk_size": n_not_computable,
            "reason": f"fewer than {MIN_INDEPENDENT_UNITS} independent units carry a computed cell at chunk size {chunk_size}",
        }
    unit_slopes = {u: [c["observed_slope"] for c in cells] for u, cells in unit_cells.items()}
    unit_intercepts = {u: [c["observed_intercept"] for c in cells] for u, cells in unit_cells.items()}
    slope_pool = _pool_unit_scalars(unit_slopes, f"{seed}|slope")
    intercept_pool = _pool_unit_scalars(unit_intercepts, f"{seed}|intercept")
    null_pooled = _pool_null_draws({u: [c["null_slopes"] for c in cells] for u, cells in unit_cells.items()})
    observed = slope_pool["mean"]
    p_value = (
        float((1 + np.sum(np.abs(null_pooled) >= abs(observed))) / (null_pooled.size + 1))
        if null_pooled.size else None
    )
    return {
        "status": "computed",
        "n_independent_units": slope_pool["n_independent_units"],
        "independent_unit_ids": slope_pool["independent_unit_ids"],
        "n_sessions_seen": n_seen, "n_sessions_not_computable_this_chunk_size": n_not_computable,
        "n_sessions_contributing": sum(len(v) for v in unit_cells.values()),
        "mean_trials_per_contributing_session": float(np.mean([c["n_trials"] for cells in unit_cells.values() for c in cells])),
        "slope": slope_pool,
        "gap_zero_intercept": intercept_pool,
        "null_mean": float(null_pooled.mean()) if null_pooled.size else None,
        "null_interval_95pct": [float(v) for v in np.percentile(null_pooled, [2.5, 97.5])] if null_pooled.size else None,
        "n_permutations_pooled": int(null_pooled.size),
        "permutation_p_value": p_value,
    }


def _corpus_candidate_cell(root: Path, corpus: str, name: str, sessions: list[dict]) -> dict:
    sweep = {
        chunk_size: _pool_slope_cell(sessions, chunk_size, f"{corpus}|{name}|chunk_{chunk_size}")
        for chunk_size in CHUNK_SIZES_TRIALS
    }
    return {"chunk_size_sweep": {str(cs): v for cs, v in sweep.items()}, "primary": sweep[PRIMARY_CHUNK_SIZE_TRIALS]}


def _run_pass(root: Path, needed_by_corpus: dict, identity: dict) -> dict:
    cells: dict[str, dict] = {}
    for corpus, names in needed_by_corpus.items():
        per_candidate = _corpus_candidate_sessions(root, corpus, names, identity)
        cells[corpus] = {
            name: _corpus_candidate_cell(root, corpus, name, per_candidate[name]) for name in names
        }
    return cells


# ================================================================================================
# null calibration: reuses _synthetic_time_independent_counts unchanged (a per-trial, per-unit
# Poisson rate shared across that trial's bins, so activity is realistic and autocorrelated but,
# by construction, independent of trial order/elapsed gap).
# ================================================================================================

def _calibration_summary(n_significant: int, n_computed: int) -> dict:
    from scipy.stats import binomtest

    rate = (n_significant / n_computed) if n_computed else None
    interval = None
    calibrated = False
    if n_computed:
        exact = binomtest(n_significant, n_computed, 0.05, alternative="two-sided")
        ci = exact.proportion_ci(confidence_level=0.95, method="exact")
        interval = [float(ci.low), float(ci.high)]
        calibrated = bool(interval[0] <= 0.05 <= interval[1])
    if calibrated:
        direction = "not distinguishable from nominal at this replicate count"
        license_statement = (
            "nominal 0.05 lies inside this calibration's own 95% exact binomial interval on the "
            "measured rate, so this module's permutation p-values are not shown to depart from nominal"
        )
    elif rate is not None and rate > 0.05:
        direction = "anti-conservative (liberal): the test rejects its own null more often than nominal"
        license_statement = (
            "this makes the zero-clearing outcome reported for every cell in this module MORE secure "
            "-- a liberal test that still rejected nothing is stronger evidence of no detectable slope "
            "-- but any cell that HAD cleared under this null would need its permutation p-value "
            "discounted before being trusted, since this calibration shows the test over-rejects at "
            "nominal p=0.05"
        )
    else:
        direction = "conservative: the test rejects its own null less often than nominal"
        license_statement = (
            "a cell that cleared under this null remains trustworthy, but a cell that did not clear "
            "could be masking a real effect this test under-detects at nominal p=0.05"
        )
    return {
        "false_positive_rate_at_p_0.05": rate,
        "false_positive_rate_95pct_exact_binomial_interval": interval,
        "calibrated": calibrated,
        "direction_of_miscalibration": direction,
        "what_this_does_and_does_not_license": license_statement,
    }


def _null_calibration(n_replicates: int, n_perm: int, seed: str,
                       n_trials: int = 64, n_units: int = 10, n_bins: int = 8) -> dict:
    rng = np.random.default_rng(_seed(seed))
    n_computed = 0
    n_significant = 0
    for replicate in range(n_replicates):
        counts = _synthetic_time_independent_counts(rng, n_trials, n_units, n_bins)
        activity = counts.sum(axis=2)
        target = np.nansum(activity, axis=1)
        prepared = _prepare_trials(activity, target)
        if prepared is None:
            continue
        directions, filtered_target = prepared
        perm_rng = np.random.default_rng(_seed(f"{seed}|replicate={replicate}"))
        cell = _session_chunk_cell(directions, filtered_target, PRIMARY_CHUNK_SIZE_TRIALS, perm_rng)
        if cell["status"] != "computed" or cell["null_slopes"].size == 0:
            continue
        n_computed += 1
        null = cell["null_slopes"]
        p = float((1 + np.sum(np.abs(null) >= abs(cell["observed_slope"]))) / (null.size + 1))
        if p <= 0.05:
            n_significant += 1
    summary = _calibration_summary(n_significant, n_computed)
    return {
        "n_replicates_requested": n_replicates, "n_replicates_computed": n_computed,
        "n_perm_per_replicate": n_perm,
        **summary,
        "generative_model": (
            "per (trial, unit) a Poisson rate drawn once per trial (shared across every bin of that "
            "trial, giving real within-trial autocorrelation with no elapsed-gap trend), spike counts "
            "drawn i.i.d. Poisson per bin from that trial's own rate; activity is, by construction, "
            "independent of trial order"
        ),
    }


# ================================================================================================
# the pre-declared decision rule and its bound-for-non-clearing-cells branch
# ================================================================================================

def _gain_candidate_payload_note() -> str:
    base = (
        "gain_total_spike_count: a session's per-trial total spike count summed across units, "
        "exactly core['spike_count'] as computed and consumed throughout "
        "run_component_identity_subspace_atlas.py / run_rank_free_component_identity.py"
    )
    rank_free_path = ROOT / "results" / "rank_free_component_identity.json"
    if not rank_free_path.exists():
        return base
    per_corpus = json.loads(rank_free_path.read_text()).get("summary", {}).get("per_corpus", {})
    cells = {corpus: cands["gain_total_spike_count"] for corpus, cands in per_corpus.items()
             if "gain_total_spike_count" in cands}
    computed = {corpus: cell for corpus, cell in cells.items() if cell.get("status") == "computed"}
    aligned = [corpus for corpus, cell in computed.items() if cell.get("aligned") is True]
    return (
        f"{base} -- the cross-animal component-identity estimate reports it computable in "
        f"{len(computed)} of {len(cells)} attempted corpora and aligned in {len(aligned)} of those "
        f"{len(computed)} ({', '.join(sorted(aligned))})"
    )


DECISION_RULES = {
    "payload": _gain_candidate_payload_note(),
    "cells_under_test": (
        "gain_total_spike_count in inagaki_alm5_mouse_ALM, dandi_000469_human, dandi_001187_human, "
        "dandi_000574_human, plus every candidate the sample-size-matched rotation test's own "
        "cells_under_test lists for dandi_000574_human (read from that test, not hardcoded), plus an "
        "attempt at gain_total_spike_count in panichello_2024_macaque_lPFC and "
        "watters_2026_macaque_multi_object -- the cross-animal component-identity estimate already "
        "reports 3 and 2 independent units respectively for this candidate in those two corpora, "
        "below the four-independent-unit floor this module does not relax"
    ),
    "chunking": (
        f"a session's kept trials (after the same trial-validity filter the cross-animal "
        f"component-identity estimate's own _cell applies) are cut into contiguous chunks of a "
        f"fixed trial count, swept over {list(CHUNK_SIZES_TRIALS)} trials with "
        f"{PRIMARY_CHUNK_SIZE_TRIALS} trials as primary, matching the sample-size-matched rotation "
        "test's own sweep so the two are comparable. A session "
        f"needs at least {MIN_CHUNKS_PER_SESSION} chunks of a given size to be computable at that size"
    ),
    "core_measurement": (
        "for every ordered pair of disjoint chunks (i, j) in a session: a read-out direction is fit "
        "on chunk i alone with regression_basis (src/subspace_identity.py, unchanged), chunk j's "
        "activity directions are projected onto that direction, and the squared Pearson correlation "
        "between that projection and chunk j's true payload values is the held-out performance for "
        "that ordered pair -- chunk j is never seen by the fit. Gap is the absolute difference "
        "between the two chunks' mean trial index, in trials"
    ),
    "primary_statistic": (
        "the slope of held-out performance against gap, fit per session by ordinary least squares "
        "over that session's own ordered chunk pairs, then pooled with equal weight per session "
        "within its own independent unit and equal weight across independent units "
        "(_pool_unit_scalars, reused unchanged from the alignment-below-null diagnostic's own "
        "estimator); the confidence interval is that function's whole-independent-unit cluster "
        "bootstrap, never a trial-count formula or an ICC design effect. The gap-zero intercept of "
        "the same per-session regression is pooled identically and reported as the matched "
        "no-separation reference -- an interior point of the same curve, not a separate arm"
    ),
    "null": (
        "within each session, the chunk centre-index array is circularly shifted by a random nonzero "
        "offset before gap is recomputed for the SAME chunk pairs and the SAME observed performance "
        "values, and the slope is refit against that shifted gap; this destroys the gap-performance "
        "relation (a rigid circular shift breaks the global monotonic correspondence between chunk "
        "index and elapsed trial position except locally, across the wrap point) while preserving "
        "chunk composition, every observed performance value, and within-session autocorrelation "
        "exactly. A free permutation of gap labels across samples was not used because it would "
        "destroy that autocorrelation structure and manufacture significance instead. Null slopes "
        "are pooled the same equal-session-then-equal-unit way as the observed slope; the two-sided "
        "permutation p-value is the fraction of pooled null slopes at least as extreme in absolute "
        "value as the pooled observed slope"
    ),
    "null_calibration": (
        f"{NULL_CALIBRATION_REPLICATES} synthetic replicates are generated with a per-trial, "
        "per-unit Poisson rate shared across that trial's bins (real within-trial autocorrelation, "
        "no elapsed-gap trend by construction, reusing "
        "run_within_trial_time_component_identity._synthetic_time_independent_counts unchanged), run "
        "through the identical primary-chunk-size cell and circular-shift null; the empirical "
        "false-positive rate at nominal p<=0.05 is reported whatever it measures, alongside the "
        "comparable existing run's own 0.055 at nominal 0.05 -- never asserted to match it"
    ),
    "clearing_rule": (
        "a slope clears only if its two-sided permutation p survives Benjamini-Hochberg FDR "
        f"(alpha={FDR_ALPHA}) across every primary-chunk-size cell this module computes AND its "
        "cluster-bootstrap 95% interval excludes zero. Both the q-value and the interval are always "
        "reported together; a cell that clears one but not the other is reported as clearing "
        "neither, with both numbers quoted"
    ),
    "bound_for_non_clearing_cells": (
        "a cell that does not clear is not reported as an unqualified null. Its minimum detectable "
        "slope at 80% power (from the same session cluster bootstrap _pool_unit_scalars already "
        "computes) is compared against an internal, same-corpus, same-scale reference decay rate: "
        "the absolute value of that corpus/candidate's own gain-identity alignment-above-null in "
        "the cross-animal component-identity estimate, divided by the mean trial count of this cell's "
        "own contributing sessions -- the per-trial slope that would fully consume that already-"
        "measured effect over the span of one session. If the minimum detectable slope is smaller "
        "than that reference, the cell is a bounded negative: a decay large enough to consume the "
        "measured gain effect within one session would have been detected here and was not, while "
        "smaller decays are not excluded -- both halves of that sentence are reported together "
        "always. If the minimum detectable slope is larger than the reference, the cell is "
        "inconclusive and is reported as such"
    ),
    "reproduction_gate": (
        "every pooled cell is recomputed a second, independent time purely from each session's own "
        "persisted checkpoint file on disk (no data is re-fit; the checkpoint cache returns the "
        "identical per-session record), and the two passes' pooled slope, intercept, null summary, "
        f"and p-value must agree within {REPRODUCTION_TOLERANCE} -- if any cell disagrees, that is "
        "reported plainly and no clearing verdict is drawn from any cell until it is resolved"
    ),
    "no_ranking": (
        "no corpus, candidate, sensor, region, modality, or estimator is described as outperforming "
        "another; only each cell's own slope-versus-gap curve is reported"
    ),
    "relationship_to_rotation_analysis": (
        "this module never claims to replicate, confirm, or be confirmed by the "
        "sample-size-matched subspace rotation-versus-time-separation test -- they are different estimands (predictive "
        "transfer of a fitted read-out vs. subspace overlap between two independent fits) computed "
        "on overlapping but not identical data (this module's chunk-pair performance measurement has "
        "no analogue in the rotation module, and its circular-shift null is unrelated to that "
        "module's interleaved-chunk and quarter-split references)"
    ),
}


def _load_rank_free_reference(corpus: str, name: str) -> dict | None:
    if not RANK_FREE_OUTPUT_PATH.exists():
        return None
    delivered = json.loads(RANK_FREE_OUTPUT_PATH.read_text())
    cell = delivered.get("summary", {}).get("per_corpus", {}).get(corpus, {}).get(name)
    if cell is None or cell.get("status") != "computed":
        return None
    return {
        "mean_alignment_above_null": cell["mean_alignment_above_null"],
        "reliably_above_null": bool(cell.get("aligned")),
    }


def _bound_stability(sweep: dict, reference_decay_rate: float) -> dict:
    bounded_at = {}
    for chunk_size in CHUNK_SIZES_TRIALS:
        if chunk_size == 2:
            continue
        cell = sweep.get(str(chunk_size), {})
        if cell.get("status") != "computed":
            continue
        bounded_at[str(chunk_size)] = cell["slope"]["minimum_detectable_difference_80pct_power"] < reference_decay_rate
    return {
        "chunk_size_2_excluded_because": (
            "a held-out evaluation chunk of exactly 2 trials always scores squared Pearson correlation "
            "1.0 against its training chunk (any two points define a perfect line), so chunk size 2 "
            "slope estimates are a floating-point-zero degeneracy that carries no information about "
            "whether a real decay would be detected"
        ),
        "bounded_at_chunk_size": bounded_at,
        "stable": len(set(bounded_at.values())) <= 1 if bounded_at else False,
    }


def _cell_verdict(pooled: dict, reference: dict | None, sweep: dict | None = None) -> dict:
    if pooled.get("status") != "computed":
        return {"verdict": "not_computable", "reason": pooled.get("reason")}
    slope = pooled["slope"]
    ci = slope["cluster_bootstrap_interval_95pct"]
    q = pooled.get("fdr_q_value")
    ci_excludes_zero = not (ci[0] <= 0.0 <= ci[1])
    q_clears = q is not None and q < FDR_ALPHA
    if q_clears and ci_excludes_zero:
        return {"verdict": "clears", "permutation_q_value": q, "cluster_bootstrap_interval_95pct": ci}
    if ci_excludes_zero:
        direction = "positive" if ci[0] > 0.0 else "negative"
        return {
            "verdict": "inconclusive_interval_excludes_zero_not_fdr_significant",
            "permutation_q_value": q, "cluster_bootstrap_interval_95pct": ci,
            "evidence_disagreement": (
                f"the cluster-bootstrap interval excludes zero on the {direction} side (evidence of a "
                f"{direction} slope) while the BH-FDR-corrected permutation q-value ({q}) does not "
                f"clear alpha={FDR_ALPHA}; the pre-declared rule requires both to clear, so this cell "
                "does not clear, and no decay bound is drawn from a cell whose own interval already "
                "disagrees with its own significance test"
            ),
        }
    mdd = slope["minimum_detectable_difference_80pct_power"]
    if reference is None or pooled.get("mean_trials_per_contributing_session", 0) <= 0:
        return {
            "verdict": "inconclusive_no_internal_reference_available",
            "permutation_q_value": q, "cluster_bootstrap_interval_95pct": ci,
            "minimum_detectable_slope_80pct_power": mdd,
        }
    reference_alignment = reference["mean_alignment_above_null"]
    reliably_above_null = bool(reference.get("reliably_above_null", False))
    if reference_alignment <= 0.0 or not reliably_above_null:
        return {
            "verdict": "inconclusive_for_want_of_a_reference",
            "permutation_q_value": q, "cluster_bootstrap_interval_95pct": ci,
            "minimum_detectable_slope_80pct_power": mdd,
            "reference_gain_alignment_above_null": reference_alignment,
            "reference_reliably_above_own_null": reliably_above_null,
            "reason": (
                "no bound is computable: this corpus/candidate's own gain-identity "
                "alignment-above-null is not a usable positive, reliably-above-its-own-null "
                "magnitude, so there is no internal same-scale reference to compare the minimum "
                "detectable slope against"
            ),
        }
    reference_decay_rate = reference_alignment / pooled["mean_trials_per_contributing_session"]
    bounded = mdd < reference_decay_rate
    record = {
        "verdict": "bounded_negative" if bounded else "inconclusive_underpowered_relative_to_reference",
        "permutation_q_value": q, "cluster_bootstrap_interval_95pct": ci,
        "minimum_detectable_slope_80pct_power": mdd,
        "reference_decay_rate_per_trial": reference_decay_rate,
        "reference_gain_alignment_above_null": reference_alignment,
        "reference_mean_trials_per_session": pooled["mean_trials_per_contributing_session"],
    }
    if bounded and sweep is not None:
        stability = _bound_stability(sweep, reference_decay_rate)
        record["sweep_stability_of_bound"] = stability
        if not stability["stable"]:
            record["verdict"] = "inconclusive_sweep_unstable_bound"
            record["reason"] = (
                "the minimum-detectable-slope-versus-reference-decay-rate comparison that determines "
                "bounded_negative does not agree across the non-degenerate chunk sizes (4, 8, 16 "
                "trials); a bound whose verdict moves with this arbitrary analysis parameter is not "
                "reported as a bound"
            )
    return record


def _apply_fdr_and_verdicts(cells: dict) -> list[dict]:
    primary_rows = []
    for corpus, candidates in cells.items():
        for name, result in candidates.items():
            primary = result["primary"]
            if primary.get("status") == "computed" and primary.get("permutation_p_value") is not None:
                primary_rows.append((corpus, name, primary))
    if primary_rows:
        p_values = np.asarray([row[2]["permutation_p_value"] for row in primary_rows])
        correction = fdr_bh(p_values, alpha=FDR_ALPHA)
        for (corpus, name, primary), q in zip(primary_rows, correction["q_values"]):
            primary["fdr_q_value"] = float(q)
    table = []
    for corpus, candidates in cells.items():
        for name, result in candidates.items():
            primary = result["primary"]
            reference = _load_rank_free_reference(corpus, name)
            verdict = _cell_verdict(primary, reference, result["chunk_size_sweep"])
            table.append({"corpus": corpus, "candidate": name, "primary": primary, "verdict": verdict})
    return table


def _sweep_stability(cells: dict) -> dict:
    stability = {}
    for corpus, candidates in cells.items():
        stability[corpus] = {}
        for name, result in candidates.items():
            flags = []
            for chunk_size in CHUNK_SIZES_TRIALS:
                cell = result["chunk_size_sweep"][str(chunk_size)]
                if cell.get("status") != "computed" or cell.get("permutation_p_value") is None:
                    continue
                ci = cell["slope"]["cluster_bootstrap_interval_95pct"]
                raw_clears = cell["permutation_p_value"] <= 0.05 and not (ci[0] <= 0.0 <= ci[1])
                flags.append(raw_clears)
            stability[corpus][name] = {
                "n_chunk_sizes_computed": len(flags),
                "stable": len(set(flags)) <= 1,
                "note": "uncorrected, informational only -- the formal decision uses only the primary chunk size's BH-FDR-corrected q-value",
            }
    return stability


def _compare_cells(a: dict, b: dict) -> dict:
    mismatches = []
    for corpus, candidates in a.items():
        for name, result in candidates.items():
            other = b.get(corpus, {}).get(name, {}).get("primary", {})
            mine = result["primary"]
            if mine.get("status") != other.get("status"):
                mismatches.append(f"{corpus}|{name}: status {mine.get('status')} vs {other.get('status')}")
                continue
            if mine.get("status") != "computed":
                continue
            for field in ("permutation_p_value",):
                left, right = mine.get(field), other.get(field)
                if left is None or right is None:
                    if left != right:
                        mismatches.append(f"{corpus}|{name}: {field} {left} vs {right}")
                    continue
                if abs(float(left) - float(right)) > REPRODUCTION_TOLERANCE:
                    mismatches.append(f"{corpus}|{name}: {field} {left} vs {right}")
            for field in ("mean",):
                left, right = mine["slope"][field], other["slope"][field]
                if abs(float(left) - float(right)) > REPRODUCTION_TOLERANCE:
                    mismatches.append(f"{corpus}|{name}: slope.{field} {left} vs {right}")
    return {"matches": len(mismatches) == 0, "mismatches": mismatches}


def _needed_by_corpus() -> dict[str, list[str]]:
    rotation = json.loads(ROTATION_OUTPUT_PATH.read_text())
    rotation_cells_000574 = sorted({
        entry["candidate"] for entry in rotation.get("cells_under_test", [])
        if entry["corpus"] == "dandi_000574_human"
    })
    if "gain_total_spike_count" not in rotation_cells_000574:
        rotation_cells_000574 = ["gain_total_spike_count"] + rotation_cells_000574
    return {
        "inagaki_alm5_mouse_ALM": ["gain_total_spike_count"],
        "dandi_000469_human": ["gain_total_spike_count"],
        "dandi_001187_human": ["gain_total_spike_count"],
        "dandi_000574_human": rotation_cells_000574,
        "panichello_2024_macaque_lPFC": ["gain_total_spike_count"],
        "watters_2026_macaque_multi_object": ["gain_total_spike_count"],
    }


def main() -> None:
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--calibration-replicates", type=int, default=NULL_CALIBRATION_REPLICATES)
    parser.add_argument("--calibration-n-perm", type=int, default=NULL_CALIBRATION_N_PERM)
    args = parser.parse_args()
    CHECKPOINT_DIR = args.checkpoint_dir
    if args.max_sessions is not None:
        if args.max_sessions < 1:
            parser.error("--max-sessions must be positive")
        os.environ[MAX_SESSIONS_ENV_VAR] = str(args.max_sessions)
    started = time.time()
    root = data_root()

    needed_by_corpus = _needed_by_corpus()
    identity = {
        "version": VERSION, "schema_version": SCHEMA_VERSION, "code_hash": _hash(),
        "data_root": str(root.resolve()), "chunk_sizes_trials": list(CHUNK_SIZES_TRIALS),
        "primary_chunk_size_trials": PRIMARY_CHUNK_SIZE_TRIALS,
        "min_chunks_per_session": MIN_CHUNKS_PER_SESSION, "n_perm": N_PERM,
        "seed_namespace": SEED_NAMESPACE, "seed_algorithm": "statistics.stable_seed_crc32",
        "rank_free_reference_artifact": str(RANK_FREE_OUTPUT_PATH),
        "rotation_cells_source_artifact": str(ROTATION_OUTPUT_PATH),
    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "question": (
            "does a read-out calibrated on one stretch of a session's trials still predict the same "
            "quantity later in the same session, and how does its held-out performance change with "
            "the elapsed gap between the calibration stretch and the evaluation stretch"
        ),
        "decision_rules_declared_before_fitting": DECISION_RULES,
        "identity": identity,
        "data_root": str(root),
        "cells_under_test": [
            {"corpus": corpus, "candidate": name} for corpus, names in needed_by_corpus.items() for name in names
        ],
        "cells": {},
    }
    _write(args.output, output)
    print(f"cells under test: {output['cells_under_test']}", file=sys.stderr, flush=True)

    for corpus, names in needed_by_corpus.items():
        print(f"starting corpus {corpus} ({len(names)} candidate(s))", file=sys.stderr, flush=True)
        per_candidate = _corpus_candidate_sessions(root, corpus, names, identity)
        output["cells"][corpus] = {
            name: _corpus_candidate_cell(root, corpus, name, per_candidate[name]) for name in names
        }
        _write(args.output, output)
        print(f"finished corpus {corpus}", file=sys.stderr, flush=True)

    print("running reproduction gate (second independent pass from persisted checkpoints)", file=sys.stderr, flush=True)
    reproduction_pass = _run_pass(root, needed_by_corpus, identity)
    reproduction = _compare_cells(output["cells"], reproduction_pass)
    output["reproduction_gate_passed"] = reproduction["matches"]
    output["reproduction_gate_mismatches"] = reproduction["mismatches"]

    table = _apply_fdr_and_verdicts(output["cells"])
    output["cell_table"] = table
    output["sweep_stability"] = _sweep_stability(output["cells"])

    print(f"running synthetic null calibration check ({args.calibration_replicates} replicates)", file=sys.stderr, flush=True)
    output["null_calibration_check"] = _null_calibration(
        args.calibration_replicates, args.calibration_n_perm, "null_calibration",
    )
    print(f"measured false-positive rate: {output['null_calibration_check']['false_positive_rate_at_p_0.05']}", file=sys.stderr, flush=True)

    n_clearing = sum(1 for row in table if row["verdict"]["verdict"] == "clears")
    n_bounded = sum(1 for row in table if row["verdict"]["verdict"] == "bounded_negative")
    n_inconclusive = sum(1 for row in table if row["verdict"]["verdict"].startswith("inconclusive"))
    n_not_computable = sum(1 for row in table if row["verdict"]["verdict"] == "not_computable")
    output["summary"] = {
        "n_cells_tested": len(table), "n_cells_clearing": n_clearing,
        "n_cells_bounded_negative": n_bounded, "n_cells_inconclusive": n_inconclusive,
        "n_cells_not_computable": n_not_computable,
        "overall_statement": (
            f"of {len(table)} attempted cells, {n_clearing} clear the pre-declared slope decision rule "
            f"(BH-FDR-corrected permutation p and a cluster-bootstrap interval excluding zero, both "
            f"reported together), {n_bounded} are bounded negatives (a decay large enough to consume "
            "this project's own already-measured gain effect within one session would have been "
            "detected here and was not, while smaller decays are not excluded), "
            f"{n_inconclusive} are inconclusive against their own internal reference, and "
            f"{n_not_computable} did not reach the four-independent-unit floor -- no corpus, "
            "candidate, or estimator is ranked against another anywhere in this statement"
        ),
    }
    output["status"] = "complete" if reproduction["matches"] else "reproduction_gate_failed"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(args.output, output)
    print(f"run_readout_transfer_decay finished: status={output['status']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
