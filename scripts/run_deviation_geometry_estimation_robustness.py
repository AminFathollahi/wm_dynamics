"""Estimation-choice robustness for the two deviation-geometry claims in the corpora that actually
carry the project's load-bearing positive results.

scripts/run_state_space_estimation_robustness.py audits five claims that were already withdrawn, in
a human corpus and a causal-microstimulation corpus where those specific claims were never the ones
supporting a positive finding. This module runs the identical estimator-invariant restatement --
the cross-validated predictable fraction of the trial-wise state deviation's variance, defined
identically for a linear projection and a nonlinear embedding -- on the two macaque corpora where the
rate-free state deviation actually passes its own orthogonality gate against total spike count and
where the accuracy link, the outside-the-memorandum-subspace result and the temporal-locus results
were established: the single-item lateral prefrontal cortex corpus (a spatial delayed-response task) and the
multi-object corpus (DANDI 000620).

Two claims, per corpus:

  1. memorandum coding subspace vs. the trial-wise deviation (does the class-mean coordinate
     structure of the fitted representation predict the deviation score);
  2. occupied state space vs. the trial-wise deviation (does the full fitted representation predict
     the deviation score, against a trial-shuffle null).

Nothing here forks the session loader. Every bundle comes from scripts/run_deviation_axis_structure.py's
own reachability and bundle-building path: ``_reachable_sessions`` and ``_macaque_bundles`` for the
single-item corpus, ``iter_watters`` + ``full_reproduction_gate`` + ``_watters_bundles`` for the
multi-object corpus. The trial-wise deviation score itself is never recomputed here -- every bundle
already carries the one this project's own reproduction gate verified exactly, read directly off it.

The multi-object corpus is analysed WITHIN item-count level throughout (a different memorandum
cardinality is a different task condition) and combined across levels by the trial-count-weighted
average this corpus's own primary behavioural estimator already uses; a pooled-across-item-count
number is never reported as this corpus's effect size. Its own null distribution is combined the
identical trial-count-weighted, draw-index-by-draw-index way scripts/run_deviation_axis_structure.py's
rotation null already combines across levels, so the combined p-value comes from a null built the same
way as the combined effect.

Two standing guards, enforced in code, not only in prose, and imported unchanged from the sibling
module that already implements them:

  - the shared fitting entry point (``fit_representation``) refuses a non-null label argument.
    Positive pairs for any embedding trained here are defined by temporal adjacency and nothing else;
  - the subspace-angle path is never reached: every claim here is expressed only through the
    estimator-invariant restatement, so no subspace angle is ever taken inside a nonlinear embedding.

Candidates run here are the linear tier (native full rank, principal components, factor analysis,
Gaussian-process factor analysis -- all near-instant per fit) plus the two nonlinear embeddings
(temporal diffusion, seconds per fit; time-contrastive, tens of seconds per fit). The sequential
autoencoder is never run by this module: its hours-per-session cost is reserved for a claim whose
verdict genuinely disagrees across the estimators above, and even then only after its pre-declared
sample size has been sized and reported, never spent automatically.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_deviation_geometry_estimation_robustness.py \
        [--candidates ...] [--single-item-sessions-limit N] [--multi-object-sessions-limit N] \
        [--n-perm-restatement N]
"""
from __future__ import annotations

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    import os as _os
    _os.environ[_var] = "1"

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import minimum_detectable_paired_difference, permutation_pvalue, stable_seed  # noqa: E402
from run_component_and_content_specific_serial_pull import _trial_count_weighted  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA, _macaque_bundles, _reachable_sessions, _watters_bundles, _weighted_combine_draws,
    full_reproduction_gate,
)
from run_dissociation_cross_preparation_test import BIN_MS, MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_state_space_estimation_robustness import (  # noqa: E402
    MAJORITY_SIGNIFICANCE_THRESHOLD, MICROSTIM_CANDIDATES, OPERATING_RANK, STATUS_VOCABULARY,
    aggregate_claim, class_mean_coordinates, decide_claim_standing, fit_representation,
    restated_claim_cell, rung_three_sample_size,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "deviation_geometry_estimation_robustness.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_deviation_geometry_estimation_robustness"

# Every admissible candidate this leg runs: the near-instant linear tier plus the two nonlinear
# embeddings, identical to the causal-microstimulation candidate roster in the sibling module -- no
# fork, the same admissible set applied to a different corpus.
CANDIDATES_RUN = MICROSTIM_CANDIDATES

SINGLE_ITEM_CORPUS, MULTI_OBJECT_CORPUS = CORPORA
CLAIM_CELL_KEYS = ("memorandum_coding_subspace_vs_deviation", "occupied_manifold_vs_deviation")
CLAIM_METRIC = {
    "memorandum_coding_subspace_vs_deviation": (
        "cross-validated predictable fraction of the deviation component's variance from the "
        "representation's class-mean coordinate structure, minus its label-permutation null"),
    "occupied_manifold_vs_deviation": (
        "cross-validated predictable fraction of the deviation component's variance from the full "
        "representation, minus its trial-shuffle null"),
}

N_PERM_RESTATEMENT_DEFAULT = 100

SHARD_VARIABLE = "WM_DYNAMICS_SESSION_SHARD"


def session_shard() -> tuple[int, int]:
    """Which slice of this run's session lists this process is responsible for, as (index, count).

    Identical convention to scripts/run_state_space_dimensionality_sweep.py's own sharding: workers
    share one checkpoint directory and every record is written by atomic rename, so concurrent
    workers on disjoint slices cannot collide. A sharded process deliberately writes no artifact --
    its view of both corpora is partial, and the claim aggregates are only meaningful over the whole
    session list. Run once more with the default (\"0/1\", unsharded) to assemble -- every cell is
    cached by then, so that pass only reloads bundles and writes the artifact."""
    index, count = (int(part) for part in os.environ.get(SHARD_VARIABLE, "0/1").split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"{SHARD_VARIABLE} must be i/n with 0 <= i < n; got {index}/{count}")
    return index, count


# ── Checkpointing (identical atomic idiom to every sibling script) ────────────────

def _checkpoint_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return CHECKPOINT_DIR / f"{safe}.json"


def _load_checkpoint(key: str) -> dict | None:
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def _save_checkpoint(key: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(key)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _run_checkpointed(key: str, fit_fn) -> tuple[dict, bool]:
    cached = _load_checkpoint(key)
    if cached is not None:
        return cached, True
    record = fit_fn()
    _save_checkpoint(key, record)
    return record, False


# ── Single-item corpus: one fit per session, no item-count structure ──────────────

def single_item_candidate_cells(bundle: dict, candidate: str, args) -> dict:
    key = bundle["session"]
    common = {"session_key": key, "candidate": candidate}
    # Checkpoint-served bundles round-trip through JSON (scripts/run_deviation_axis_structure.py's own
    # cache); coerce explicitly rather than trust the object type a cache hit hands back.
    counts_raw = np.asarray(bundle["counts"], dtype=float)
    counts_3d = np.transpose(counts_raw, (0, 2, 1))  # (trials, bins, units)
    labels = np.asarray(bundle["memorandum_label"], dtype=float)
    y_dev = np.asarray(bundle["deviation"], dtype=float)
    rng = np.random.default_rng(stable_seed(
        f"deviation_geometry_robustness_single_item_{key}_{candidate}"))
    fit = fit_representation(candidate, counts_3d, OPERATING_RANK, rng, is_spiking=True, bin_ms=BIN_MS)
    if fit.get("status") != "fitted":
        return {**common, "status": "fit_failed", "reason": fit.get("reason", fit.get("status"))}
    latent = np.asarray(fit["latent_train"], dtype=float)
    latent_trial = latent.mean(axis=1)
    cells = {**common, "status": "computed", "k_used": int(fit["k_used"]),
             "positive_pair_definition": fit["positive_pair_definition"],
             "n_trials": int(counts_3d.shape[0]), "n_units": int(counts_3d.shape[2])}
    for cell_key, null_kind, coords in (
        ("memorandum_coding_subspace_vs_deviation", "label_permutation",
         class_mean_coordinates(latent_trial, labels)),
        ("occupied_manifold_vs_deviation", "y_shuffle", np.eye(latent_trial.shape[1])),
    ):
        cells[cell_key] = restated_claim_cell(y_dev, coords, labels, latent_trial, null_kind,
                                              rng, args.n_perm_restatement)
    return cells


# ── Multi-object corpus: one fit per (session, item-count level), combined per session ────────────

def _combine_restated_cells_across_levels(level_cells: list[tuple[int, dict]], n_perm: int) -> dict:
    """Trial-count-weighted combination of per-item-count-level restatement cells into one
    session-level cell. The predictable fraction and its null mean are weighted-averaged the same way
    this corpus's own primary behavioural estimator combines any within-level statistic across levels;
    the null distribution itself is combined the identical draw-index-weighted way the delivered
    rotation-null machinery already uses, so the combined p-value is judged against a null built the
    same way as the combined effect -- never a p-value picked from whichever level happened to have
    the smallest one."""
    computed = [(n, c) for n, c in level_cells if c.get("status") == "computed"]
    if not computed:
        return {"status": "not_computable", "reason": "no item-count level reached the trial floor"}
    frac = _trial_count_weighted([(n, c["predictable_fraction"]) for n, c in computed])
    null_mean = _trial_count_weighted([(n, c["null_mean"]) for n, c in computed])
    padded = []
    for n, c in computed:
        vals = list(c.get("null_values", []))
        vals = vals + [float("nan")] * (n_perm - len(vals))
        padded.append((n, np.asarray(vals[:n_perm], dtype=float)))
    pooled_null = _weighted_combine_draws(padded)
    pooled_null = pooled_null[np.isfinite(pooled_null)] if pooled_null is not None else np.array([])
    p_value = permutation_pvalue(pooled_null >= frac) if len(pooled_null) else None
    return {"status": "computed", "predictable_fraction": float(frac), "null_mean": float(null_mean),
            "effect_size": float(frac - null_mean), "p_value": p_value,
            "n_pooled_null_draws": int(len(pooled_null)), "n_levels_combined": len(computed),
            "n_trials": int(sum(n for n, _ in computed))}


def multi_object_candidate_cells(bundle: dict, counts_raw: np.ndarray, candidate: str, args) -> dict:
    key = bundle["session"]
    common = {"session_key": key, "candidate": candidate}
    item_count = np.asarray(bundle["item_count"], dtype=float)
    labels_all = np.asarray(bundle["memorandum_label"], dtype=float)
    y_dev_all = np.asarray(bundle["deviation"], dtype=float)
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level_status: dict = {}
    memo_levels, occ_levels = [], []
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level_status[str(level)] = {"status": "too_few_trials_at_this_item_count_level",
                                            "n_trials": n_level}
            continue
        counts_level = np.transpose(counts_raw[mask], (0, 2, 1))
        rng = np.random.default_rng(stable_seed(
            f"deviation_geometry_robustness_multi_object_{key}_level{level}_{candidate}"))
        fit = fit_representation(candidate, counts_level, OPERATING_RANK, rng, is_spiking=True,
                                 bin_ms=BIN_MS)
        if fit.get("status") != "fitted":
            per_level_status[str(level)] = {"status": "fit_failed", "n_trials": n_level,
                                            "reason": fit.get("reason", fit.get("status"))}
            continue
        latent = np.asarray(fit["latent_train"], dtype=float)
        latent_trial = latent.mean(axis=1)
        labels_level, y_dev_level = labels_all[mask], y_dev_all[mask]
        memo = restated_claim_cell(y_dev_level, class_mean_coordinates(latent_trial, labels_level),
                                   labels_level, latent_trial, "label_permutation", rng,
                                   args.n_perm_restatement)
        occ = restated_claim_cell(y_dev_level, np.eye(latent_trial.shape[1]), labels_level,
                                  latent_trial, "y_shuffle", rng, args.n_perm_restatement)
        per_level_status[str(level)] = {"status": "computed", "n_trials": n_level,
                                        "k_used": int(fit["k_used"])}
        if memo.get("status") == "computed":
            memo_levels.append((n_level, memo))
        if occ.get("status") == "computed":
            occ_levels.append((n_level, occ))
    if not memo_levels and not occ_levels:
        return {**common, "status": "excluded", "per_level_status": per_level_status,
                "reason": "no item-count level reached the trial floor or fit successfully"}
    n_levels_tested = sum(1 for v in per_level_status.values() if v.get("status") == "computed")
    return {**common, "status": "computed", "per_level_status": per_level_status,
            "n_levels_tested": n_levels_tested,
            "n_trials": int(sum(n for n, _ in memo_levels)) if memo_levels else 0,
            "memorandum_coding_subspace_vs_deviation": _combine_restated_cells_across_levels(
                memo_levels, args.n_perm_restatement),
            "occupied_manifold_vs_deviation": _combine_restated_cells_across_levels(
                occ_levels, args.n_perm_restatement)}


# ── Aggregation and zero-drop accounting ───────────────────────────────────────────

def build_corpus_claims(records_by_candidate: dict[str, list[dict]]) -> dict:
    claims = {}
    for cell_key in CLAIM_CELL_KEYS:
        estimators, verdict_keys = {}, {}
        for candidate, records in sorted(records_by_candidate.items()):
            agg, vkey = aggregate_claim(records, cell_key)
            agg["verdict_key"] = vkey
            estimators[candidate] = agg
            verdict_keys[candidate] = vkey
        standing = decide_claim_standing(verdict_keys)
        block = {"metric": CLAIM_METRIC[cell_key], "estimator_cells": estimators, "agreement": standing}
        branch = standing["branch"]
        if branch == "verdict_confirmed_across_estimators":
            block["status"] = "settled_robust"
        elif branch == "estimation_dependent_rung_three_escalation":
            block["status"] = "requires_refit_budget_decision"
            block["status_note"] = ("disagreement across estimators triggers this claim alone; see "
                                    "rung_three_sizing_preview for the pre-declared sizing and the "
                                    "budget decision it asks for -- no tier-three fit is run by this "
                                    "module")
        else:
            block["status"] = "inconclusive_below_detection_floor"
        native = block["estimator_cells"].get("native_full_rank", {})
        if branch == "estimation_dependent_rung_three_escalation" and native.get("status") == "computed":
            sizing = rung_three_sample_size(native.get("std_effect_size") or float("nan"),
                                            native.get("mean_effect_size") or float("nan"))
            block["rung_three_sizing_preview"] = {
                "sample_sizing": sizing,
                "n_applicable_sessions": len(native.get("per_session", [])),
                "would_require_a_budget_decision_if_run": not sizing.get("feasible_within_budget", False)}
        claims[cell_key] = block
    return claims


def zero_drop(records_by_candidate: dict[str, list[dict]]) -> dict:
    per_candidate = {}
    for candidate, recs in sorted(records_by_candidate.items()):
        statuses: dict[str, int] = {}
        exclusions = []
        for rec in recs:
            st = rec.get("status", "missing")
            statuses[st] = statuses.get(st, 0) + 1
            if st in ("excluded", "fit_failed"):
                exclusions.append({"session_key": rec.get("session_key"), "reason": rec.get("reason")})
        per_candidate[candidate] = {
            "n_seen": len(recs), "statuses": statuses,
            "seen_equals_tested_plus_excluded": len(recs) == sum(statuses.values()),
            "exclusions_with_reasons": exclusions}
    return per_candidate


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="restrict to these candidates (smoke tests)")
    parser.add_argument("--single-item-sessions-limit", type=int, default=None)
    parser.add_argument("--multi-object-sessions-limit", type=int, default=None)
    parser.add_argument("--n-perm-restatement", type=int, default=N_PERM_RESTATEMENT_DEFAULT)
    args = parser.parse_args()

    t0 = time.time()
    root = data_root()
    candidates = tuple(args.candidates) if args.candidates else CANDIDATES_RUN
    shard_index, shard_count = session_shard()

    # ---- Single-item corpus -----------------------------------------------------
    single_paths = _reachable_sessions(root)
    single_bundles, _ = _macaque_bundles(root)
    if args.single_item_sessions_limit is not None:
        single_bundles = single_bundles[: args.single_item_sessions_limit]
    # Sharding slices the already-built, deterministically ordered bundle list -- identical modulo
    # convention to scripts/run_state_space_dimensionality_sweep.py's session_shard, applied here to
    # a list index instead of an enumerate() over the loader because both corpora's bundles are
    # already materialised in one stable-order list before any candidate is fitted.
    single_bundles_sharded = [b for i, b in enumerate(single_bundles) if i % shard_count == shard_index]

    single_records: dict[str, list[dict]] = {}
    cache_hits, refits = 0, 0
    for bundle in single_bundles_sharded:
        for candidate in candidates:
            key = f"single_item__{bundle['session']}__{candidate}"
            rec, hit = _run_checkpointed(
                key, lambda b=bundle, c=candidate: single_item_candidate_cells(b, c, args))
            cache_hits += int(hit)
            refits += int(not hit)
            single_records.setdefault(candidate, []).append(rec)
            print(f"[single_item] {bundle['session']} {candidate}: {rec['status']}", file=sys.stderr)

    # ---- Multi-object corpus ----------------------------------------------------
    watters_seen, watters_loaded, watters_refused = 0, [], []
    for session in iter_watters(root, bin_ms=BIN_MS):
        watters_seen += 1
        if session["status"] != "loaded":
            watters_refused.append({"session": session["session"], "status": session["status"]})
            continue
        watters_loaded.append(session)

    gate_result, watters_arrays_by_session = full_reproduction_gate(root, watters_loaded)
    multi_bundles = _watters_bundles(watters_arrays_by_session)
    if args.multi_object_sessions_limit is not None:
        multi_bundles = multi_bundles[: args.multi_object_sessions_limit]
    multi_bundles_sharded = [b for i, b in enumerate(multi_bundles) if i % shard_count == shard_index]

    multi_records: dict[str, list[dict]] = {}
    for bundle in multi_bundles_sharded:
        entry = watters_arrays_by_session[bundle["session"]]
        # Same explicit coercion as the single-item corpus: this array is read straight off the
        # in-memory loader chain, never a checkpoint, but coerced anyway on the same principle -- a
        # value handed across a module boundary is verified to be an array, not assumed to be one.
        counts_raw = np.asarray(entry["session"]["counts"], dtype=float)[np.asarray(entry["usable"])]
        for candidate in candidates:
            key = f"multi_object__{bundle['session']}__{candidate}"
            rec, hit = _run_checkpointed(
                key, lambda b=bundle, cr=counts_raw, c=candidate: multi_object_candidate_cells(
                    b, cr, c, args))
            cache_hits += int(hit)
            refits += int(not hit)
            multi_records.setdefault(candidate, []).append(rec)
            print(f"[multi_object] {bundle['session']} {candidate}: {rec['status']}", file=sys.stderr)

    if shard_count > 1:
        print(f"Shard {shard_index} of {shard_count} finished its sessions in "
              f"{time.time() - t0:.0f}s; checkpoints written, no artifact aggregated. Run once more "
              f"with {SHARD_VARIABLE} unset (or \"0/1\") to assemble every shard's checkpoints into "
              "the artifact.", file=sys.stderr)
        return

    claims_by_corpus = {
        SINGLE_ITEM_CORPUS: build_corpus_claims(single_records) if single_records else {},
        MULTI_OBJECT_CORPUS: build_corpus_claims(multi_records) if multi_records else {},
    }

    output = {
        "version": "deviation_geometry_estimation_robustness_v1",
        "code_commit": git_commit(ROOT),
        "question": ("whether the deviation's relationship to the memorandum coding subspace and to "
                     "the occupied state space, in the two corpora where the deviation actually "
                     "carries the project's positive results, depends on how the state space was "
                     "estimated"),
        "audit_character": ("a robustness audit, never a comparison between estimators: the only "
                            "question is whether the project's conclusions depend on how the state "
                            "space was estimated"),
        "pre_declared_rules": {
            "operating_rank": OPERATING_RANK,
            "verdict_key_rule": ("identical to scripts/run_state_space_estimation_robustness.py's "
                                 "repaired rule, imported unchanged: a sign enters an effect cell's "
                                 "verdict key only where it is resolvable -- majority significance at "
                                 f"p<0.05 across sessions (threshold {MAJORITY_SIGNIFICANCE_THRESHOLD}"
                                 "), or the mean effect clearing this same aggregate's own paired "
                                 "minimum detectable difference; otherwise the key reads "
                                 "'no_majority_effect' with no sign encoded"),
            "escalation_semantics": ("a claim whose verdict holds across every estimator is settled "
                                     "and is not escalated; a claim whose verdict differs escalates, "
                                     "that claim alone -- this module never spends the tier-three "
                                     "budget itself, it only sizes and reports what an escalation "
                                     "would require"),
            "circularity_guard": ("the shared fitting entry point refuses a non-null label argument; "
                                  "embeddings on this leg are time-contrastive only, positive pairs "
                                  "temporal adjacency and nothing else"),
            "subspace_angle_guard": ("no subspace angle is ever taken here -- every claim is expressed "
                                     "only through the estimator-invariant restatement, defined "
                                     "identically for a linear projection and a nonlinear embedding"),
            "multi_object_level_combination": ("the multi-object corpus is analysed within item-count "
                                               "level, then combined into one session-level cell by "
                                               "trial-count-weighted averaging of the predictable "
                                               "fraction and its null mean, and by trial-count-weighted, "
                                               "draw-index-by-draw-index combination of the underlying "
                                               "null distribution itself, so the combined p-value is "
                                               "judged against a null built the same way as the "
                                               "combined effect -- never a pooled-across-level number "
                                               "reported as this corpus's effect size"),
            "status_vocabulary": STATUS_VOCABULARY,
        },
        "scope": {
            "single_item_corpus": (f"{SINGLE_ITEM_CORPUS}, delay epoch only, "
                                   f"{BIN_MS:.0f} ms bins, reachability and bundle-building reused "
                                   "unchanged from scripts/run_deviation_axis_structure.py"),
            "multi_object_corpus": (f"{MULTI_OBJECT_CORPUS}, delay epoch only, {BIN_MS:.0f} ms bins, "
                                    "analysed within item-count level, reachability and "
                                    "bundle-building reused unchanged from "
                                    "scripts/run_deviation_axis_structure.py"),
            "n_single_item_sessions_seen": len(single_paths),
            "n_single_item_sessions_with_a_bundle": len(single_bundles),
            "n_multi_object_sessions_seen": watters_seen,
            "n_multi_object_sessions_loaded": len(watters_loaded),
            "n_multi_object_sessions_with_a_bundle": len(multi_bundles),
            "candidates": list(candidates),
            "n_perm_restatement": args.n_perm_restatement,
            "seed_scheme": ("stable_seed('deviation_geometry_robustness_<corpus>_<session>"
                            "[_level<k>]_<candidate>')"),
            "cache_hits_vs_refits": {"checkpoint_hits": int(cache_hits), "fresh_fits": int(refits)},
            "sequential_autoencoder": ("never run by this module; reserved for an escalated claim "
                                       "only, under the pre-declared sample-size rule, and only after "
                                       "that sizing is reported for a human budget decision"),
            "wall_clock_s": None,
        },
        "zero_drop_accounting": {
            SINGLE_ITEM_CORPUS: zero_drop(single_records),
            MULTI_OBJECT_CORPUS: zero_drop(multi_records),
            "multi_object_sessions_seen_equals_loaded_plus_refused": bool(
                watters_seen == len(watters_loaded) + len(watters_refused)),
            "multi_object_refusal_reasons": watters_refused,
        },
        "claims": claims_by_corpus,
        "resume_state": {
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "resume_instruction": ("rerun the same command; every (corpus, session[, level], "
                                   "candidate) cell is checkpointed atomically and completed cells "
                                   "are skipped, so a partial pass resumes where it stopped"),
        },
        "wall_clock_s": None,
    }
    output["scope"]["wall_clock_s"] = time.time() - t0
    output["wall_clock_s"] = time.time() - t0
    OUTPUT_PATH.write_text(canonical_json(output))
    print(f"Wrote {OUTPUT_PATH} in {output['wall_clock_s']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
