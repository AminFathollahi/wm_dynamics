"""run_occupied_subspace_membership_split_half_circularity_check.py -- tests whether the occupied-space
membership reversal in results/occupied_subspace_membership_effect_size.json survives when the axis and
the occupied basis are estimated from DISJOINT halves of a session's trials, rather than from the same
trials both objects are built from in that artifact (and in the delivered pipeline it corrects).

THE CIRCULARITY CONCERN THIS SETTLES. In the reused, unchanged construction (`_collect_axis_entries`), the
axis is the leading eigenvector of R, the stacked per-trial residual directions (each trial's own unit
direction with its own leave-one-out mean direction projected out) for a session's kept trials. The
occupied basis is the leading singular subspace of U, the per-trial unit-direction matrix for those SAME
kept trials. Both objects are therefore fit on the identical trial set, and R is a near-orthogonal-
complement transform of U's own rows (u_i with a projection of the SAME session's own mean direction
removed) -- a within_fraction of 0.998 against a selected rank of a few dozen out of a few hundred ambient
units is exactly what in-sample joint estimation would produce whether or not the axis is a genuine,
independent geometric feature of the population. The in-sample number cannot, on its own, separate those
two possibilities.

THE SPLIT-HALF DESIGN. For each cell (session, or session x item-count level) that produced a computed,
nonzero rank under entry_holdout_bicross_validation in-sample -- the only estimator evaluable in either
corpus in the effect-size artifact this module extends -- that cell's kept trials are split into two
disjoint random halves, independently 20 times (`N_SPLITS_PER_CELL`). On each split: the occupied basis,
INCLUDING its rank selection, is fit on half A only (entry_holdout_bicross_validation applied to half A's
own unit-direction matrix, at a reduced computational grid disclosed below); the axis is fit on half B only
(the identical residual construction -- leave-one-out mean, cosine, residual, unit-normalisation, leading
eigenvector -- recomputed from half B's own raw trial activity, so half B's own leave-one-out means never
see half A's trials or vice versa). The two objects never share a trial. A calibration control axis is also
fit on half B, by the identical residual-construction pipeline applied to a column-wise-permuted copy of
half B's own activity (each unit's own trial-marginal distribution preserved, cross-unit and cross-trial
structure destroyed) -- this axis inherits every step of the real estimation pipeline but carries no genuine
structure, so its split-half within-fraction shows how much of the in-sample number the pipeline alone can
manufacture between two objects that share nothing but noise.

A DISCLOSED COMPUTATIONAL SIMPLIFICATION, applied identically to every split-half basis fit and to the
control: entry_holdout_bicross_validation's row/column fold count is reduced from the in-sample 5x5 to 3x3,
and its candidate-rank ceiling from 25 to 15 (`SPLIT_HALF_N_ROW_FOLDS`, `SPLIT_HALF_N_COL_FOLDS`,
`SPLIT_HALF_MAX_CANDIDATE_K`). This is a computational simplification for tractability across 117 cells x
20 splits, not a claim that the coarser grid is otherwise preferable; it is the SAME grid throughout this
module's own comparisons, so the split-half-vs-control contrast it produces is internally fair even though
it is not bit-identical to the in-sample estimator it is being checked against.

THE PRE-DECLARED VERDICT, decided in this order from the pooled split-half contrast's whole-session
cluster-bootstrap 95% interval and, where computable, the pooled control contrast's own interval:
  1. too few sessions produce a valid split (see MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP) ->
     'not_determinable_insufficient_sessions_with_a_valid_split'.
  2. the interval's upper bound is below 0 -> 'split_half_contrast_is_below_chance' (kept for completeness;
     not the outcome either the in-sample reading or the circularity concern predicts).
  3. the interval includes 0 (its lower bound is at or below 0) ->
     'the_reversal_is_withdrawn_as_unmeasured_under_split_half_estimation' -- the split-half contrast has
     collapsed toward the chance reference. This is NOT a re-affirmation of the delivered pipeline's
     'outside' branch (that branch is independently voided by the degenerate-null defect documented in
     results/occupied_subspace_membership_effect_size.json); it means this corpus's data cannot separate
     genuine axis membership from an artifact of estimating both objects on the same trials, and the
     reversal this module's own in-sample block reported is withdrawn as UNMEASURED, not replaced.
  4. the interval's lower bound is positive but the pooled control contrast is computable and its own upper
     bound reaches or exceeds the split-half interval's lower bound (the two are not clearly separated) ->
     'inconclusive_split_half_effect_not_clearly_separated_from_the_no_structure_control'.
  5. the interval's lower bound is positive AND (the control is not computable, OR clearly separated below
     it) -> 'the_reversal_survives_the_split_half_circularity_check' -- a genuine, held-out geometric
     concentration of the axis inside a basis fit from disjoint trials, stronger evidence than the in-sample
     number alone because the circularity that number could not rule out is ruled out here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import data_root  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA, _residual_rows, _trial_count_weighted, _unit_residual_matrix, leading_eigenvector,
)
from run_deviation_serial_dependence_and_temporal_locus import unit_direction_vectors  # noqa: E402
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_occupied_subspace_membership_effect_size import _whole_session_cluster_bootstrap  # noqa: E402
from run_occupied_subspace_rank_estimation import entry_holdout_rank  # noqa: E402
from run_occupied_subspace_rank_selection_repair import _load_corpora_and_accounting  # noqa: E402
from statistics import stable_seed  # noqa: E402

ARTIFACT_PATH = ROOT / "results" / "occupied_subspace_membership_effect_size.json"

N_SPLITS_PER_CELL = 20
SPLIT_HALF_N_ROW_FOLDS = 3
SPLIT_HALF_N_COL_FOLDS = 3
SPLIT_HALF_MAX_CANDIDATE_K = 15
ESTIMATOR_NAME = "entry_holdout_bicross_validation"  # the only estimator evaluable in-sample in either corpus

DECISION_RULE_DECLARED_BEFORE_FITTING = __doc__.split("THE PRE-DECLARED VERDICT")[1]
DECISION_RULE_DECLARED_BEFORE_FITTING = "THE PRE-DECLARED VERDICT" + DECISION_RULE_DECLARED_BEFORE_FITTING


def _basis_from_fitted_rank(U_fit: np.ndarray, k: int) -> np.ndarray:
    mean_u = U_fit.mean(axis=0)
    _u_svd, _s_svd, vt = np.linalg.svd(U_fit - mean_u, full_matrices=False)
    k_eff = min(k, vt.shape[0])
    return vt[:k_eff].T, k_eff


def _column_shuffled(activity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permutes each unit's own trial values -- preserves every unit's own marginal
    distribution, destroys any cross-unit or cross-trial structure. Fed through the identical residual/
    axis-estimation pipeline as the real half, this yields a calibration axis with no genuine structure."""
    n, p = activity.shape
    shuffled = np.empty_like(activity)
    for j in range(p):
        shuffled[:, j] = activity[rng.permutation(n), j]
    return shuffled


def _axis_from_half(activity_half: np.ndarray) -> np.ndarray | None:
    rows = _residual_rows(activity_half)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, _idx = _unit_residual_matrix(rows)
    if R.shape[0] < 2:
        return None
    return leading_eigenvector(R)


def _split_half_cell(activity_full: np.ndarray, kept_idx: np.ndarray, tag_base: str) -> dict:
    p = activity_full.shape[1]
    splits = []
    for split_i in range(N_SPLITS_PER_CELL):
        rng = np.random.default_rng(stable_seed(f"{tag_base}|split{split_i}"))
        perm = rng.permutation(kept_idx)
        half = perm.size // 2
        idx_a, idx_b = perm[:half], perm[half:]
        if idx_a.size < MIN_TRIALS_WITH_DEFINED_DIRECTION or idx_b.size < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            splits.append({"status": "not_estimable_half_too_small"})
            continue

        activity_a, activity_b = activity_full[idx_a], activity_full[idx_b]
        U_a = unit_direction_vectors(activity_a)
        valid_a = ~np.isnan(U_a).any(axis=1)
        U_a = U_a[valid_a]
        if U_a.shape[0] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            splits.append({"status": "not_estimable_fitting_half_too_small_after_direction_filter"})
            continue

        rank_result = entry_holdout_rank(
            U_a, f"{tag_base}|split{split_i}|basis",
            max_candidate_k=SPLIT_HALF_MAX_CANDIDATE_K, n_row_folds=SPLIT_HALF_N_ROW_FOLDS,
            n_col_folds=SPLIT_HALF_N_COL_FOLDS)
        if rank_result.get("status") != "computed" or not rank_result.get("best_k"):
            splits.append({"status": "basis_not_estimable_or_zero_rank"})
            continue
        basis, k_eff = _basis_from_fitted_rank(U_a, rank_result["best_k"])

        axis = _axis_from_half(activity_b)
        if axis is None:
            splits.append({"status": "axis_not_estimable_on_held_out_half"})
            continue

        chance = k_eff / p
        within_proj = basis @ (basis.T @ axis)
        within_frac = float(np.sum(within_proj ** 2))
        contrast = within_frac - chance

        control_rng = np.random.default_rng(stable_seed(f"{tag_base}|split{split_i}|control_shuffle"))
        control_axis = _axis_from_half(_column_shuffled(activity_b, control_rng))
        control_contrast = None
        control_within_frac = None
        if control_axis is not None:
            control_proj = basis @ (basis.T @ control_axis)
            control_within_frac = float(np.sum(control_proj ** 2))
            control_contrast = control_within_frac - chance

        splits.append({
            "status": "computed", "selected_rank_k": k_eff, "ambient_unit_count_p": p,
            "within_fraction": within_frac, "chance_reference_k_over_p": chance, "contrast": contrast,
            "control_within_fraction": control_within_frac, "control_contrast": control_contrast,
        })

    computed = [s for s in splits if s.get("status") == "computed"]
    return {
        "n_splits_requested": N_SPLITS_PER_CELL, "n_splits_valid": len(computed),
        "splits": splits,
        "mean_within_fraction": (float(np.mean([s["within_fraction"] for s in computed])) if computed else None),
        "mean_chance_reference": (float(np.mean([s["chance_reference_k_over_p"] for s in computed])) if computed else None),
        "mean_contrast": (float(np.mean([s["contrast"] for s in computed])) if computed else None),
        "mean_control_contrast": (
            float(np.mean([s["control_contrast"] for s in computed if s["control_contrast"] is not None]))
            if any(s["control_contrast"] is not None for s in computed) else None),
    }


def _pool_corpus(bundles: list[dict], corpus_key: str, entry_axis_entries: dict) -> dict:
    per_session = []
    for bundle in bundles:
        session = bundle["session"]
        entries = entry_axis_entries.get(session, [])
        if not entries:
            continue
        cell_pools = []
        for entry in entries:
            level = entry["level"]
            if corpus_key == CORPORA[0]:
                activity_full = bundle["activity_by_unit"]
            else:
                mask = bundle["item_count"] == float(level)
                activity_full = bundle["activity_by_unit"][mask]
            rows_full = _residual_rows(activity_full)
            kept_idx = np.flatnonzero(rows_full["keep"])
            if kept_idx.size < 2 * MIN_TRIALS_WITH_DEFINED_DIRECTION:
                continue
            tag_base = f"occupied_subspace_membership_split_half_circularity_check|{corpus_key}|{session}|{level}"
            pooled_cell = _split_half_cell(activity_full, kept_idx, tag_base)
            pooled_cell.update({"level": level, "n_trials_kept_full": int(kept_idx.size)})
            cell_pools.append(pooled_cell)
        if not cell_pools:
            continue
        usable = [c for c in cell_pools if c["mean_contrast"] is not None]
        if not usable:
            per_session.append({"session": session, "status": "not_computable", "levels": cell_pools})
            continue
        session_wf = _trial_count_weighted([(c["n_trials_kept_full"], c["mean_within_fraction"]) for c in usable])
        session_kp = _trial_count_weighted([(c["n_trials_kept_full"], c["mean_chance_reference"]) for c in usable])
        control_usable = [c for c in usable if c["mean_control_contrast"] is not None]
        session_control_contrast = (
            _trial_count_weighted([(c["n_trials_kept_full"], c["mean_control_contrast"]) for c in control_usable])
            if control_usable else None)
        per_session.append({
            "session": session, "status": "computed", "levels": cell_pools,
            "session_within_fraction": session_wf, "session_chance_reference": session_kp,
            "session_contrast": session_wf - session_kp, "session_control_contrast": session_control_contrast,
        })

    computed = [s for s in per_session if s.get("status") == "computed"]
    tag = f"occupied_subspace_membership_split_half_circularity_check|{corpus_key}"
    within_pooled = _whole_session_cluster_bootstrap(
        [s["session_within_fraction"] for s in computed], f"{tag}|within_fraction")
    chance_pooled = _whole_session_cluster_bootstrap(
        [s["session_chance_reference"] for s in computed], f"{tag}|chance_reference")
    contrast_pooled = _whole_session_cluster_bootstrap(
        [s["session_contrast"] for s in computed], f"{tag}|contrast")
    control_values = [s["session_control_contrast"] for s in computed if s["session_control_contrast"] is not None]
    control_contrast_pooled = _whole_session_cluster_bootstrap(control_values, f"{tag}|control_contrast")

    verdict = _verdict(contrast_pooled, control_contrast_pooled)

    return {
        "n_sessions_with_a_computed_split_half_value": len(computed),
        "per_session": per_session,
        "split_half_within_fraction_pooled": within_pooled,
        "split_half_chance_reference_pooled": chance_pooled,
        "split_half_contrast_pooled": contrast_pooled,
        "no_structure_control_contrast_pooled": control_contrast_pooled,
        "verdict": verdict,
    }


def _verdict(contrast_pooled: dict, control_pooled: dict) -> str:
    if contrast_pooled.get("status") != "computed":
        return "not_determinable_insufficient_sessions_with_a_valid_split"
    ci_low, ci_high = contrast_pooled["cluster_bootstrap_ci_95pct"]
    if ci_high < 0.0:
        return "split_half_contrast_is_below_chance"
    if ci_low <= 0.0:
        return "the_reversal_is_withdrawn_as_unmeasured_under_split_half_estimation"
    if control_pooled.get("status") == "computed" and control_pooled["cluster_bootstrap_ci_95pct"][1] >= ci_low:
        return "inconclusive_split_half_effect_not_clearly_separated_from_the_no_structure_control"
    return "the_reversal_survives_the_split_half_circularity_check"


def main() -> None:
    root = data_root()
    print("loading both corpora for the split-half circularity check", file=sys.stderr)
    loaded, _zero_drop = _load_corpora_and_accounting(root)
    bundles = loaded["bundles"]

    from run_deviation_axis_structure import _collect_axis_entries  # local import, unchanged function
    axis_entries_by_corpus = {c: _collect_axis_entries(bundles[c], c) for c in CORPORA}

    in_sample = json.loads(ARTIFACT_PATH.read_text())["occupied_state_space_membership_by_corpus"]

    block = {
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_splits_per_cell": N_SPLITS_PER_CELL,
        "reduced_grid_for_tractability": {
            "n_row_folds": SPLIT_HALF_N_ROW_FOLDS, "n_col_folds": SPLIT_HALF_N_COL_FOLDS,
            "max_candidate_k": SPLIT_HALF_MAX_CANDIDATE_K,
        },
        "by_corpus": {},
    }
    for corpus_key in CORPORA:
        print(f"split-half circularity check: {corpus_key} ({len(bundles[corpus_key])} sessions)", file=sys.stderr)
        pooled = _pool_corpus(bundles[corpus_key], corpus_key, axis_entries_by_corpus[corpus_key])
        in_sample_contrast = (
            in_sample.get(corpus_key, {}).get("by_estimator", {}).get(ESTIMATOR_NAME, {}).get("contrast_pooled"))
        block["by_corpus"][corpus_key] = {
            "in_sample_contrast_pooled_for_comparison_only_not_recomputed": in_sample_contrast,
            **pooled,
        }
        print(f"  {corpus_key}: verdict={pooled['verdict']} "
              f"split_half_contrast={pooled['split_half_contrast_pooled'].get('pooled_mean')} "
              f"n_sessions={pooled['n_sessions_with_a_computed_split_half_value']}", file=sys.stderr)

    existing = json.loads(ARTIFACT_PATH.read_text())
    existing["split_half_circularity_check"] = _json_safe(block)
    ARTIFACT_PATH.write_text(json.dumps(existing, indent=2, allow_nan=False, default=float))
    print(f"appended split_half_circularity_check to {ARTIFACT_PATH}", file=sys.stderr)
    print(json.dumps({c: block["by_corpus"][c]["verdict"] for c in CORPORA}, indent=2))


if __name__ == "__main__":
    main()
