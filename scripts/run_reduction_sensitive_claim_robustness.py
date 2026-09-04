"""Estimation-choice robustness for the two nulls where a dimensionality reduction is the most
likely explanation of the result as it currently stands: cross-temporal generalisation, and the
memorandum decoding that defines the coding subspace against the trial-wise state deviation.

Both claims were audited across estimators once already, in the human intracranial corpus and the
causal-microstimulation macaque corpus, by scripts/run_state_space_estimation_robustness.py. This
module extends the identical estimator-invariant machinery -- the same fitting entry point, the same
cross-temporal decoding scorer, the same class-mean-coordinate restatement, the same verdict-key rule
-- into the two macaque corpora where the trial-wise deviation actually carries the project's
positive results and where a sibling module (run_deviation_geometry_estimation_robustness.py) already
ran this same kind of audit for the two related subspace-geometry claims, finding them settled_robust.
That sibling module's structure -- the candidate roster, the sharding and checkpoint scheme, the
zero-drop block, and the multi-object corpus's within-item-count-level combination -- is reused here
unchanged rather than reinvented; only the two claim cells and the reduction-sensitivity decision
differ.

Two claims, per corpus:

  1. cross-temporal generalisation: does the memorandum's class structure, decoded in one time bin,
     generalise to a held-out bin, against a label-permutation null -- computed directly on
     precomputed per-timepoint coordinates from any estimator, linear or nonlinear;
  2. memorandum coding subspace vs. the trial-wise deviation: does the class-mean coordinate
     structure of the fitted representation predict the deviation score, against a label-permutation
     null -- the identical restatement the sibling module already delivered for these two corpora,
     recomputed here from the same fresh fit that claim 1 needs anyway, so the dominant cost (fitting
     the representation itself) is paid once per (session[, item-count level], candidate) cell rather
     than twice.

Every bundle comes from scripts/run_deviation_axis_structure.py's own reachability and
bundle-building path, unchanged: ``_reachable_sessions`` and ``_macaque_bundles`` for the single-item
corpus, ``iter_watters`` + ``full_reproduction_gate`` + ``_watters_bundles`` for the multi-object
corpus. Sessions admitted are exactly the sessions the sibling module admitted for the same two
corpora -- neither widened nor narrowed here.

The pre-declared decision rule for whether a null is a property of the data or an artefact of the
projection, written before any number in this module exists:

  A claim improves materially under an estimator when its cross-validated predictable fraction (or,
  for cross-temporal generalisation, its off-diagonal AUC-over-chance) exceeds its own matched null
  with majority significance across sessions (threshold 0.5 at p<0.05), AND the mean effect clears
  that same aggregate's own paired minimum detectable difference at 80% power. This is exactly the
  sibling estimation-robustness module's own ``effect_cell_verdict_key`` rule -- a sign enters the key
  only where it is resolvable by that rule -- imported unchanged and reused as the operational test
  for "materially improves," rather than restated from scratch.

  - ``null_is_a_property_of_the_data`` -- no estimator, native full rank included, produces a
    material improvement. The delivered null hardens.
  - ``null_is_an_artefact_of_the_projection`` -- at least one admissible estimator produces a
    material improvement, and every estimator that does so agrees on its sign. The delivered null is
    about the projection and every sentence resting on it must be flagged for restatement.
  - ``verdicts_disagree_escalation_sized_not_spent`` -- estimators materially improve on both signs,
    or the pattern does not otherwise resolve. The tier-three escalation is sized and its cost
    reported; no tier-three fit is spent.

Two standing guards, enforced in code and imported unchanged from the sibling estimation-robustness
module: the shared fitting entry point (``fit_representation``) refuses a non-null label argument, so
every embedding fitted here is time-contrastive only, positive pairs defined by temporal adjacency
alone; and no subspace angle is ever taken inside a nonlinear embedding -- both claims here are
expressed only through cross-validated, estimator-invariant scoring on precomputed coordinates.

Candidates: the identical six-candidate roster the sibling geometry module uses -- native full rank,
principal components, factor analysis, Gaussian-process factor analysis, temporal diffusion
embedding, time-contrastive embedding. The sequential autoencoder is never run by this module.

Cost warning, from the delivered admissibility sweep: the temporal diffusion embedding is 94% of the
multi-object macaque corpus's per-session cost, about 57 minutes a session against 78 seconds for the
next most expensive candidate. Shard by session, checkpoint every (corpus, session[, item-count
level], candidate) cell atomically, cap BLAS threads to 1 before importing numpy.

This is a robustness audit and never a comparison between estimators. No sentence in the script, the
artifact or the report may say one method outperformed another.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_reduction_sensitive_claim_robustness.py \
        [--candidates ...] [--single-item-sessions-limit N] [--multi-object-sessions-limit N] \
        [--n-perm-ctg N] [--n-perm-restatement N]
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
from geometry import _ctg_score_fold_multiclass, temporal_stability_tau  # noqa: E402
from provenance import canonical_json, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from statistics import permutation_pvalue, stable_seed  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA, _macaque_bundles, _reachable_sessions, _watters_bundles, full_reproduction_gate,
)
from run_deviation_geometry_estimation_robustness import (  # noqa: E402
    _combine_restated_cells_across_levels, zero_drop,
)
from run_dissociation_cross_preparation_test import BIN_MS, MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_state_space_dimensionality_sweep import CTG_N_SPLITS, CTG_STEP  # noqa: E402
from run_state_space_estimation_robustness import (  # noqa: E402
    MAJORITY_SIGNIFICANCE_THRESHOLD, MICROSTIM_CANDIDATES, OPERATING_RANK, STATUS_VOCABULARY,
    aggregate_claim, class_mean_coordinates, fit_representation, restated_claim_cell,
    rung_three_sample_size,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "reduction_sensitive_claim_robustness.json"
# A run that does not cover the full pre-declared scope (a restricted candidate list, a
# permutation count below the pre-declared default, or an artificial session cap) never writes
# to OUTPUT_PATH -- it writes here instead, so the production path can only ever hold a run that
# covered everything it was pre-declared to cover.
SHAKEDOWN_OUTPUT_PATH = RESULTS / "reduction_sensitive_claim_robustness.shakedown.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_reduction_sensitive_claim_robustness"
CHECKPOINT_SCHEMA = "reduction_sensitive_claim_robustness_v1"

# The identical admissible six-candidate roster the sibling geometry module runs -- no fork, the
# same set applied to the same two corpora for a different pair of claims.
CANDIDATES_RUN = MICROSTIM_CANDIDATES

SINGLE_ITEM_CORPUS, MULTI_OBJECT_CORPUS = CORPORA
CLAIM_CELL_KEYS = ("cross_temporal_generalization", "memorandum_coding_subspace_vs_deviation")
CLAIM_METRIC = {
    "cross_temporal_generalization": (
        "cross-validated multiclass memorandum decoding: mean off-diagonal AUC across held-out "
        "time-bin pairs minus chance (0.5), minus the same statistic's own label-permutation null"),
    "memorandum_coding_subspace_vs_deviation": (
        "cross-validated predictable fraction of the deviation component's variance from the "
        "representation's class-mean coordinate structure, minus its label-permutation null"),
}

N_PERM_CTG_DEFAULT = 100
N_PERM_RESTATEMENT_DEFAULT = 100

SHARD_VARIABLE = "WM_DYNAMICS_SESSION_SHARD"


def session_shard() -> tuple[int, int]:
    """Identical sharding convention to the sibling geometry module: workers share one checkpoint
    directory and every record is written by atomic rename, so concurrent workers on disjoint slices
    cannot collide. A sharded process writes no artifact -- its view of both corpora is partial and
    the claim aggregates only mean something over the whole session list. Run once more unsharded to
    assemble; every cell is cached by then, so that pass only reloads cells and writes the artifact."""
    index, count = (int(part) for part in os.environ.get(SHARD_VARIABLE, "0/1").split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"{SHARD_VARIABLE} must be i/n with 0 <= i < n; got {index}/{count}")
    return index, count


# ── Checkpointing (atomic rename, schema-tagged, read through restore_checkpoint) ─────────────────

def _checkpoint_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return CHECKPOINT_DIR / f"{safe}.json"


def _load_checkpoint(key: str) -> dict | None:
    """A checkpoint written under a different schema tag (an older run of this script, or a
    collision with an unrelated one) is treated as absent rather than as a hit -- the schema tag is
    checked before ``_complete``, so a mismatched entry can never silently satisfy a lookup it was
    never written to answer."""
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_schema") != CHECKPOINT_SCHEMA or data.get("_complete") is not True:
        return None
    # Every checkpoint is read through restore_checkpoint rather than trusted as already-typed JSON:
    # a resumed run has silently handed a list back where a fresh run produced an array before, and
    # this is the guard against that recurring.
    return restore_checkpoint(data["record"])


def _save_checkpoint(key: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(key)
    payload = {"_schema": CHECKPOINT_SCHEMA, "_complete": True, "record": checkpoint_safe(record)}
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


# ── Cross-temporal generalisation, restated to retain its own null draws ──────────────────────────

def ctg_cell_with_null_draws(latent: np.ndarray, labels: np.ndarray, t_idx: np.ndarray,
                             n_splits: int, n_perm: int, rng: np.random.Generator) -> dict:
    """Cross-temporal multiclass memorandum decoding on precomputed per-timepoint coordinates from
    any estimator, linear or nonlinear -- identical fold structure, scorer and label-permutation null
    to the delivered CTG restatement in the sibling estimation-robustness module
    (``ctg_content_permutation_null_on_latents``), extended only to retain each permutation's null
    draw. That retention is the one thing this module needs that the delivered function does not
    provide: the multi-object corpus is analysed within item-count level, and combining a null
    distribution across levels (rather than picking whichever level's p-value is smallest) requires
    the raw draws, not just their summary mean. Field names mirror ``restated_claim_cell``'s output
    (``predictable_fraction``, ``null_mean``, ``effect_size``, ``null_values``) so both claims this
    module computes share one combination and aggregation path."""
    from sklearn.model_selection import StratifiedKFold

    n_trials = latent.shape[0]
    labels = np.asarray(labels)
    all_classes = np.unique(labels)
    if len(all_classes) < 2 or n_trials < n_splits or len(t_idx) < 1:
        return {"status": "not_computable",
                "reason": "fewer than 2 classes, fewer trials than folds, or no time bins"}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=int(rng.integers(0, 1_000_000)))
    fold_data = [(latent[tr], labels[tr], latent[te], labels[te])
                 for tr, te in skf.split(np.zeros(n_trials), labels)]
    fold_mats = [_ctg_score_fold_multiclass(z_tr, y_tr, z_te, y_te, t_idx, all_classes)
                 for z_tr, y_tr, z_te, y_te in fold_data]
    auc_obs = np.nanmean(np.stack(fold_mats), axis=0)

    def _offdiag_stat(mat: np.ndarray) -> float:
        n_t = mat.shape[0]
        return float(np.nanmean(mat[~np.eye(n_t, dtype=bool)] - 0.5))

    off_obs = _offdiag_stat(auc_obs)
    null = np.full(n_perm, np.nan)
    for p in range(n_perm):
        mats_p = [_ctg_score_fold_multiclass(z_tr, rng.permutation(y_tr), z_te,
                                             rng.permutation(y_te), t_idx, all_classes)
                  for z_tr, y_tr, z_te, y_te in fold_data]
        null[p] = _offdiag_stat(np.nanmean(np.stack(mats_p), axis=0))
    finite_null = null[np.isfinite(null)]
    if len(finite_null) == 0:
        return {"status": "not_computable", "reason": "every permutation draw was undefined"}
    diag = np.diag(auc_obs)
    tau_info = temporal_stability_tau(auc_obs)
    null_mean = float(np.mean(finite_null))
    return {"status": "computed", "predictable_fraction": off_obs, "null_mean": null_mean,
            "effect_size": float(off_obs - null_mean),
            "p_value": permutation_pvalue(finite_null >= off_obs),
            "null_kind": "label_permutation", "n_null": int(len(finite_null)),
            "null_values": [float(v) for v in null],  # raw draws, NaN kept, for cross-level pooling
            "diag_auc_peak_minus_chance": float(np.nanmax(diag)) if np.isfinite(diag).any() else None,
            "tau": tau_info["tau"], "tau_interpretable": tau_info["interpretable"],
            "n_dims": int(latent.shape[2]), "n_trials": int(n_trials), "n_time_bins": int(len(t_idx))}


# ── Single-item corpus: one fit per session, no item-count structure ──────────────────────────────

def single_item_candidate_cells(bundle: dict, candidate: str, args) -> dict:
    key = bundle["session"]
    common = {"session_key": key, "candidate": candidate}
    # Checkpoint-served bundles round-trip through JSON (the axis-structure module's own cache);
    # coerce explicitly rather than trust the object type a cache hit hands back.
    counts_raw = np.asarray(bundle["counts"], dtype=float)
    counts_3d = np.transpose(counts_raw, (0, 2, 1))  # (trials, bins, units)
    labels = np.asarray(bundle["memorandum_label"], dtype=float)
    y_dev = np.asarray(bundle["deviation"], dtype=float)
    rng = np.random.default_rng(stable_seed(f"reduction_sensitive_single_item_{key}_{candidate}"))
    fit = fit_representation(candidate, counts_3d, OPERATING_RANK, rng, is_spiking=True, bin_ms=BIN_MS)
    if fit.get("status") != "fitted":
        return {**common, "status": "fit_failed", "reason": fit.get("reason", fit.get("status"))}
    latent = np.asarray(fit["latent_train"], dtype=float)
    latent_trial = latent.mean(axis=1)
    cells = {**common, "status": "computed", "k_used": int(fit["k_used"]),
             "positive_pair_definition": fit["positive_pair_definition"],
             "n_trials": int(counts_3d.shape[0]), "n_units": int(counts_3d.shape[2])}

    t_idx = np.arange(0, latent.shape[1], CTG_STEP)
    ctg_rng = np.random.default_rng(
        stable_seed(f"reduction_sensitive_single_item_{key}_{candidate}_ctg"))
    cells["cross_temporal_generalization"] = ctg_cell_with_null_draws(
        latent, labels, t_idx, CTG_N_SPLITS, args.n_perm_ctg, ctg_rng)

    subspace_rng = np.random.default_rng(
        stable_seed(f"reduction_sensitive_single_item_{key}_{candidate}_subspace"))
    cells["memorandum_coding_subspace_vs_deviation"] = restated_claim_cell(
        y_dev, class_mean_coordinates(latent_trial, labels), labels, latent_trial,
        "label_permutation", subspace_rng, args.n_perm_restatement)
    return cells


# ── Multi-object corpus: one fit per (session, item-count level), combined per session ────────────

def multi_object_candidate_cells(bundle: dict, counts_raw: np.ndarray, candidate: str, args) -> dict:
    key = bundle["session"]
    common = {"session_key": key, "candidate": candidate}
    item_count = np.asarray(bundle["item_count"], dtype=float)
    labels_all = np.asarray(bundle["memorandum_label"], dtype=float)
    y_dev_all = np.asarray(bundle["deviation"], dtype=float)
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level_status: dict = {}
    ctg_levels, subspace_levels = [], []
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level_status[str(level)] = {"status": "too_few_trials_at_this_item_count_level",
                                            "n_trials": n_level}
            continue
        counts_level = np.transpose(counts_raw[mask], (0, 2, 1))
        fit_rng = np.random.default_rng(stable_seed(
            f"reduction_sensitive_multi_object_{key}_level{level}_{candidate}"))
        fit = fit_representation(candidate, counts_level, OPERATING_RANK, fit_rng, is_spiking=True,
                                 bin_ms=BIN_MS)
        if fit.get("status") != "fitted":
            per_level_status[str(level)] = {"status": "fit_failed", "n_trials": n_level,
                                            "reason": fit.get("reason", fit.get("status"))}
            continue
        latent = np.asarray(fit["latent_train"], dtype=float)
        latent_trial = latent.mean(axis=1)
        labels_level, y_dev_level = labels_all[mask], y_dev_all[mask]

        t_idx = np.arange(0, latent.shape[1], CTG_STEP)
        ctg_rng = np.random.default_rng(stable_seed(
            f"reduction_sensitive_multi_object_{key}_level{level}_{candidate}_ctg"))
        ctg = ctg_cell_with_null_draws(latent, labels_level, t_idx, CTG_N_SPLITS,
                                       args.n_perm_ctg, ctg_rng)

        subspace_rng = np.random.default_rng(stable_seed(
            f"reduction_sensitive_multi_object_{key}_level{level}_{candidate}_subspace"))
        subspace = restated_claim_cell(y_dev_level, class_mean_coordinates(latent_trial, labels_level),
                                       labels_level, latent_trial, "label_permutation", subspace_rng,
                                       args.n_perm_restatement)

        per_level_status[str(level)] = {"status": "computed", "n_trials": n_level,
                                        "k_used": int(fit["k_used"])}
        if ctg.get("status") == "computed":
            ctg_levels.append((n_level, ctg))
        if subspace.get("status") == "computed":
            subspace_levels.append((n_level, subspace))
    if not ctg_levels and not subspace_levels:
        return {**common, "status": "excluded", "per_level_status": per_level_status,
                "reason": "no item-count level reached the trial floor or fit successfully"}
    n_levels_tested = sum(1 for v in per_level_status.values() if v.get("status") == "computed")
    return {**common, "status": "computed", "per_level_status": per_level_status,
            "n_levels_tested": n_levels_tested,
            "n_trials": int(sum(n for n, _ in subspace_levels)) if subspace_levels else 0,
            "cross_temporal_generalization": _combine_restated_cells_across_levels(
                ctg_levels, args.n_perm_ctg),
            "memorandum_coding_subspace_vs_deviation": _combine_restated_cells_across_levels(
                subspace_levels, args.n_perm_restatement)}


# ── The reduction-sensitivity decision, pre-declared before any number is seen ─────────────────────

def decide_reduction_sensitivity(verdict_keys: dict[str, str]) -> dict:
    """Whether a claim's delivered null is a property of the data or an artefact of the projection.

    An estimator is judged to have produced a material improvement over the delivered null exactly
    when its own verdict key -- imported unchanged from the sibling estimation-robustness module,
    which already encodes the pre-declared rule (majority significance at p<0.05 across sessions,
    with a sign written into the key only when the mean effect ALSO clears that same aggregate's own
    paired minimum detectable difference) -- reads ``majority_significant_positive`` or
    ``majority_significant_negative``. That is the sibling module's own operationalisation of
    "materially improves," imported and reused rather than restated here.

    Native full rank is exactly as eligible to fire this as any reduced estimator: an improvement
    found ONLY at native full rank, with every reduced candidate finding nothing, is precisely the
    signature of a reduction discarding a signal that is genuinely present in the unreduced data --
    the null_is_an_artefact_of_the_projection branch does not special-case it out."""
    computable = {name: key for name, key in verdict_keys.items()
                  if key not in (None, "not_applicable", "not_computable")}
    if not computable:
        return {"branch": "no_computable_estimator_cells", "materially_improved_estimators": {},
                "non_improving_estimators": {}}
    improving = {name: key for name, key in computable.items()
                if key.startswith("majority_significant_")}
    non_improving = {name: key for name, key in computable.items() if name not in improving}
    if not improving:
        return {"branch": "null_is_a_property_of_the_data",
                "materially_improved_estimators": {}, "non_improving_estimators": non_improving}
    sides = {key.rsplit("_", 1)[-1] for key in improving.values()}
    if len(sides) == 1:
        return {"branch": "null_is_an_artefact_of_the_projection",
                "materially_improved_estimators": improving, "non_improving_estimators": non_improving}
    return {"branch": "verdicts_disagree_escalation_sized_not_spent",
            "materially_improved_estimators": improving, "non_improving_estimators": non_improving}


# ── Aggregation and zero-drop accounting ───────────────────────────────────────────────────────────

def build_corpus_claims(records_by_candidate: dict[str, list[dict]]) -> dict:
    claims = {}
    for cell_key in CLAIM_CELL_KEYS:
        estimators, verdict_keys = {}, {}
        for candidate, records in sorted(records_by_candidate.items()):
            agg, vkey = aggregate_claim(records, cell_key)
            agg["verdict_key"] = vkey
            estimators[candidate] = agg
            verdict_keys[candidate] = vkey
        decision = decide_reduction_sensitivity(verdict_keys)
        block = {"metric": CLAIM_METRIC[cell_key], "estimator_cells": estimators,
                "reduction_sensitivity": decision, "status": decision["branch"]}
        native = estimators.get("native_full_rank", {})
        if decision["branch"] == "verdicts_disagree_escalation_sized_not_spent" and \
                native.get("status") == "computed":
            sizing = rung_three_sample_size(native.get("std_effect_size") or float("nan"),
                                            native.get("mean_effect_size") or float("nan"))
            block["rung_three_sizing_preview"] = {
                "sample_sizing": sizing, "n_applicable_sessions": len(native.get("per_session", [])),
                "would_require_a_budget_decision_if_run": not sizing.get("feasible_within_budget", False)}
        claims[cell_key] = block
    return claims


# ── Main ─────────────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="restrict to these candidates (smoke tests)")
    parser.add_argument("--single-item-sessions-limit", type=int, default=None)
    parser.add_argument("--multi-object-sessions-limit", type=int, default=None)
    parser.add_argument("--n-perm-ctg", type=int, default=N_PERM_CTG_DEFAULT)
    parser.add_argument("--n-perm-restatement", type=int, default=N_PERM_RESTATEMENT_DEFAULT)
    args = parser.parse_args()

    t0 = time.time()
    root = data_root()
    candidates = tuple(args.candidates) if args.candidates else CANDIDATES_RUN
    shard_index, shard_count = session_shard()

    # Pre-declared full scope: every one of the six admissible candidates, at least the
    # pre-declared permutation counts, and no artificial cap on how many sessions are admitted.
    # Anything short of that is a shakedown run and must say so in the artifact rather than let a
    # reader mistake a partial pass for a finished one.
    covers_full_scope = (
        set(candidates) == set(CANDIDATES_RUN)
        and args.n_perm_ctg >= N_PERM_CTG_DEFAULT
        and args.n_perm_restatement >= N_PERM_RESTATEMENT_DEFAULT
        and args.single_item_sessions_limit is None
        and args.multi_object_sessions_limit is None
    )

    # ---- Single-item corpus -----------------------------------------------------
    single_paths = _reachable_sessions(root)
    single_bundles, _ = _macaque_bundles(root)
    if args.single_item_sessions_limit is not None:
        single_bundles = single_bundles[: args.single_item_sessions_limit]
    single_bundles_sharded = [b for i, b in enumerate(single_bundles) if i % shard_count == shard_index]

    single_records: dict[str, list[dict]] = {}
    cache_hits, refits = 0, 0
    for bundle in single_bundles_sharded:
        for candidate in candidates:
            # The permutation counts are part of the key, not just the candidate/session: a cell
            # cached from a lower-n_perm smoke pass must be a miss (recomputed at the requested
            # n_perm), never silently served as a hit for a higher-n_perm production run.
            key = (f"single_item__{bundle['session']}__{candidate}"
                  f"__ctg{args.n_perm_ctg}_sub{args.n_perm_restatement}")
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
        counts_raw = np.asarray(entry["session"]["counts"], dtype=float)[np.asarray(entry["usable"])]
        for candidate in candidates:
            key = (f"multi_object__{bundle['session']}__{candidate}"
                  f"__ctg{args.n_perm_ctg}_sub{args.n_perm_restatement}")
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
        "version": "reduction_sensitive_claim_robustness_v1",
        "code_commit": git_commit(ROOT),
        "question": ("whether cross-temporal generalisation and the memorandum decoding that "
                     "defines the coding subspace, in the two corpora where the trial-wise state "
                     "deviation actually carries the project's positive results, depend on how the "
                     "state space was estimated -- i.e. whether their delivered nulls are properties "
                     "of the data or artefacts of the projection that discarded the signal"),
        "audit_character": ("a robustness audit, never a comparison between estimators: the only "
                            "question is whether the project's conclusions depend on how the state "
                            "space was estimated"),
        "pre_declared_rules": {
            "operating_rank": OPERATING_RANK,
            "materially_improves_rule": (
                "identical to scripts/run_state_space_estimation_robustness.py's repaired "
                "effect_cell_verdict_key rule, imported unchanged: a claim materially improves "
                "under an estimator exactly when that estimator's verdict key reads "
                "majority_significant_positive or majority_significant_negative -- majority "
                f"significance at p<0.05 across sessions (threshold "
                f"{MAJORITY_SIGNIFICANCE_THRESHOLD}) with a sign written into the key only when the "
                "mean effect also clears that same aggregate's own paired minimum detectable "
                "difference; otherwise the key reads 'no_majority_effect' with no sign encoded and "
                "counts as no material improvement"),
            "reduction_sensitivity_branches": {
                "null_is_a_property_of_the_data": "no estimator, native full rank included, "
                    "produces a material improvement; the delivered null hardens",
                "null_is_an_artefact_of_the_projection": "at least one admissible estimator "
                    "produces a material improvement and every improving estimator agrees on sign; "
                    "the delivered null is about the projection and every sentence resting on it "
                    "must be flagged for restatement",
                "verdicts_disagree_escalation_sized_not_spent": "estimators materially improve on "
                    "both signs, or the pattern otherwise does not resolve; the tier-three "
                    "escalation is sized and its cost reported, never spent by this module",
            },
            "circularity_guard": ("the shared fitting entry point refuses a non-null label "
                                  "argument; embeddings on this leg are time-contrastive only, "
                                  "positive pairs temporal adjacency and nothing else"),
            "subspace_angle_guard": ("no subspace angle is ever taken here -- both claims are "
                                     "expressed only through cross-validated, estimator-invariant "
                                     "scoring on precomputed coordinates"),
            "multi_object_level_combination": ("the multi-object corpus is analysed within "
                                               "item-count level, then combined into one "
                                               "session-level cell by trial-count-weighted "
                                               "averaging of the effect and its null mean, and by "
                                               "trial-count-weighted, draw-index-by-draw-index "
                                               "combination of the underlying null distribution "
                                               "itself, identical to the sibling geometry module's "
                                               "combination path -- never a pooled-across-level "
                                               "number reported as this corpus's effect size"),
            "status_vocabulary": STATUS_VOCABULARY,
        },
        "scope": {
            "single_item_corpus": (f"{SINGLE_ITEM_CORPUS}, delay epoch only, "
                                   f"{BIN_MS:.0f} ms bins, reachability and bundle-building reused "
                                   "unchanged from scripts/run_deviation_axis_structure.py, "
                                   "identical session admission to the sibling geometry module"),
            "multi_object_corpus": (f"{MULTI_OBJECT_CORPUS}, delay epoch only, {BIN_MS:.0f} ms bins, "
                                    "analysed within item-count level, reachability and "
                                    "bundle-building reused unchanged from "
                                    "scripts/run_deviation_axis_structure.py, identical session "
                                    "admission to the sibling geometry module"),
            "n_single_item_sessions_seen": len(single_paths),
            "n_single_item_sessions_with_a_bundle": len(single_bundles),
            "n_multi_object_sessions_seen": watters_seen,
            "n_multi_object_sessions_loaded": len(watters_loaded),
            "n_multi_object_sessions_with_a_bundle": len(multi_bundles),
            "candidates": list(candidates),
            "n_perm_ctg": args.n_perm_ctg,
            "n_perm_restatement": args.n_perm_restatement,
            "seed_scheme": ("stable_seed('reduction_sensitive_<corpus>_<session>"
                            "[_level<k>]_<candidate>[_ctg|_subspace]')"),
            "cache_hits_vs_refits": {"checkpoint_hits": int(cache_hits), "fresh_fits": int(refits)},
            "sequential_autoencoder": "never run by this module",
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
        # Carried at the top level, unconditionally, so a reader never has to infer completeness
        # from the scope block's candidate/permutation counts by hand: covers_full_pre_declared_scope
        # is the single boolean that answers "is this a result or a shakedown."
        "run_completeness": {
            "n_candidates_run": len(candidates),
            "n_candidates_pre_declared": len(CANDIDATES_RUN),
            "candidates_run": list(candidates),
            "candidates_pre_declared": list(CANDIDATES_RUN),
            "n_perm_ctg": args.n_perm_ctg,
            "n_perm_restatement": args.n_perm_restatement,
            "n_perm_ctg_pre_declared": N_PERM_CTG_DEFAULT,
            "n_perm_restatement_pre_declared": N_PERM_RESTATEMENT_DEFAULT,
            "n_single_item_sessions_with_a_bundle": len(single_bundles),
            "n_multi_object_sessions_with_a_bundle": len(multi_bundles),
            "single_item_sessions_limit_applied": args.single_item_sessions_limit,
            "multi_object_sessions_limit_applied": args.multi_object_sessions_limit,
            "covers_full_pre_declared_scope": covers_full_scope,
        },
        "wall_clock_s": None,
    }
    if covers_full_scope:
        output["status"] = "complete_full_scope"
        output_path = OUTPUT_PATH
    else:
        output["status"] = (
            "This is a reduced-scope shakedown run, not a scientific result. It covered only a "
            "partial candidate list, permutation counts below the pre-declared full run, and/or an "
            "artificial cap on the number of admitted sessions. No claim, verdict, or number in this "
            "file should be quoted or treated as a conclusion. Re-run with the complete candidate "
            "roster, full permutation counts, and no session caps to produce a result."
        )
        output_path = SHAKEDOWN_OUTPUT_PATH
    output["scope"]["wall_clock_s"] = time.time() - t0
    output["wall_clock_s"] = time.time() - t0
    output_path.write_text(canonical_json(output))
    print(f"Wrote {output_path} in {output['wall_clock_s']:.0f}s "
          f"(covers_full_pre_declared_scope={covers_full_scope})", file=sys.stderr)


if __name__ == "__main__":
    main()
