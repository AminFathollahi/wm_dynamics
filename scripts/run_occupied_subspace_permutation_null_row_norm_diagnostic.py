"""run_occupied_subspace_permutation_null_row_norm_diagnostic.py -- tests whether the permutation-
eigenvalue rank estimator's null is mis-specified for the matrix it is applied to, and appends one new,
forward-only block to results/occupied_subspace_membership_effect_size.json with the answer. No existing
block of that artifact is read for its numbers or modified; only a new top-level key is added.

THE QUESTION. `permutation_eigenvalue_rank` (imported unchanged from run_occupied_subspace_rank_estimation)
is applied to U, the per-trial L2-normalised unit-direction matrix `unit_direction_vectors` builds -- every
row of U has norm exactly 1 by construction. Its null permutes each column of U independently across
trials (`permuted[:, j] = U[rng.permutation(n), j]`) and recomputes the eigenvalue spectrum of
permuted.T @ permuted. An independent per-column permutation preserves each COLUMN's own marginal
distribution and the matrix's total sum of squares (trace(permuted.T @ permuted) == trace(U.T @ U) == n,
since permuting entries within a column cannot change that column's own sum of squares), but it does NOT
preserve the per-ROW unit-norm constraint the observed matrix satisfies: a permuted row is assembled from
independently-shuffled entries potentially drawn from different original trials, so its norm is generally
not 1. The permuted matrix is therefore not a null for "the same matrix with between-unit covariance
destroyed" -- it is a null for a different, unconstrained object, and comparing the observed (constrained)
spectrum to it risks a mis-specified null in either direction.

THE TEST. On the identical real cells and the identical permutation draws (same seed, same column
permutations), a SECOND null is computed that renormalises each permuted row back to unit norm before
its eigenvalue spectrum is taken -- the only change from the delivered estimator is this renormalisation,
so any difference in detected rank between the two isolates the effect of the row-norm constraint alone.
Both nulls are reported on the same cells; neither is described as more sensitive or preferable, since that
judgement is exactly what this diagnostic exists to settle from the constraint argument, not from which one
"finds more."
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
from run_deviation_axis_structure import CORPORA, _collect_axis_entries, leading_eigenvector  # noqa: E402
from run_occupied_subspace_rank_estimation import permutation_eigenvalue_rank  # noqa: E402
from run_occupied_subspace_rank_selection_repair import _load_corpora_and_accounting  # noqa: E402
from statistics import stable_seed  # noqa: E402

ARTIFACT_PATH = ROOT / "results" / "occupied_subspace_membership_effect_size.json"
N_PERMUTATIONS = 500  # identical to the estimator being tested, for a fair comparison

# A fixed, deterministic sample: cells the delivered checkpoint already recorded as rank 0 under the
# current null, spanning both corpora and a range of small-to-moderate ambient unit counts.
DIAGNOSTIC_CELLS = [
    (CORPORA[0], "211006", "all"),
    (CORPORA[0], "211001", "all"),
    (CORPORA[0], "210929", "all"),
    (CORPORA[1], "Perle_2022-06-09", "1"),
    (CORPORA[1], "Perle_2022-05-27", "1"),
    (CORPORA[1], "Perle_2022-06-05", "1"),
]


def permutation_eigenvalue_rank_row_renormalized(U: np.ndarray, seed_tag: str, n_perm: int = N_PERMUTATIONS) -> dict:
    """Identical to permutation_eigenvalue_rank except that every permuted row is renormalised to unit
    norm before its spectrum is taken -- the null then satisfies the same per-row unit-norm constraint the
    observed matrix U does, isolating the effect of that constraint on the detected rank."""
    n, p = U.shape
    if n < 2 or p < 2:
        return {"status": "not_estimable"}
    obs = np.sort(np.linalg.eigvalsh(U.T @ U))[::-1]
    rng = np.random.default_rng(stable_seed(seed_tag))
    perm_eigs = np.empty((n_perm, p))
    for it in range(n_perm):
        permuted = np.empty_like(U)
        for j in range(p):
            permuted[:, j] = U[rng.permutation(n), j]
        norms = np.linalg.norm(permuted, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            permuted = np.where(norms > 0, permuted / np.where(norms > 0, norms, 1.0), 0.0)
        perm_eigs[it] = np.sort(np.linalg.eigvalsh(permuted.T @ permuted))[::-1]
    threshold_95 = np.percentile(perm_eigs, 95, axis=0)
    exceeds = obs > threshold_95
    rank = 0
    for e in exceeds:
        if not e:
            break
        rank += 1
    keep = min(p, 30)
    return {
        "status": "computed", "n_permutations": n_perm, "best_k": rank,
        "observed_eigenvalues_leading": obs[:keep].tolist(),
        "null_95th_percentile_leading": threshold_95[:keep].tolist(),
    }


def main() -> None:
    root = data_root()
    print("loading both corpora to reach the diagnostic cells' U matrices", file=sys.stderr)
    loaded, _zero_drop = _load_corpora_and_accounting(root)
    bundles = loaded["bundles"]

    axis_entries_by_corpus = {c: _collect_axis_entries(bundles[c], c) for c in CORPORA}

    rows = []
    for corpus_key, session, level in DIAGNOSTIC_CELLS:
        entries = axis_entries_by_corpus[corpus_key].get(session, [])
        entry = next((e for e in entries if e["level"] == level), None)
        if entry is None:
            rows.append({"corpus": corpus_key, "session": session, "level": level,
                         "status": "cell_not_found_in_this_run's_loaded_bundles"})
            continue
        U = entry["U"]
        n, p = U.shape
        tag_base = f"occupied_subspace_permutation_null_row_norm_diagnostic|{corpus_key}|{session}|{level}"
        current_null = permutation_eigenvalue_rank(U, f"{tag_base}|current_null")
        row_renormalized_null = permutation_eigenvalue_rank_row_renormalized(U, f"{tag_base}|row_renormalized_null")

        def _top_position_summary(result: dict) -> dict:
            # A detected rank of 0 is uninterpretable on its own: it could mean the leading eigenvalue is
            # far inside the permutation null (isotropic once row magnitude is removed -- no low-dimensional
            # structure to detect) or just under its 95th-percentile threshold (a near miss, i.e. the
            # estimator is under-powered here rather than the object being structureless). The ratio of the
            # observed top eigenvalue to its own null threshold separates the two readings.
            obs_top = result.get("observed_eigenvalues_leading", [None])[0]
            null_top = result.get("null_95th_percentile_leading", [None])[0]
            ratio = (obs_top / null_top) if (obs_top is not None and null_top not in (None, 0.0)) else None
            return {"observed_leading_eigenvalue": obs_top, "null_95th_percentile_leading_eigenvalue": null_top,
                    "ratio_observed_to_null_threshold": ratio}

        current_top = _top_position_summary(current_null)
        row_renorm_top = _top_position_summary(row_renormalized_null)
        rows.append({
            "corpus": corpus_key, "session": session, "level": level,
            "n_trials": int(n), "n_units": int(p),
            "current_within_column_permutation_null": {
                "best_k": current_null.get("best_k"), "status": current_null.get("status"), **current_top},
            "row_renormalized_permutation_null": {
                "best_k": row_renormalized_null.get("best_k"), "status": row_renormalized_null.get("status"),
                **row_renorm_top},
        })
        print(f"  {corpus_key}|{session}|{level}: n={n} p={p} "
              f"current_best_k={current_null.get('best_k')} ratio={current_top['ratio_observed_to_null_threshold']} "
              f"row_renormalized_best_k={row_renormalized_null.get('best_k')} "
              f"ratio={row_renorm_top['ratio_observed_to_null_threshold']}", file=sys.stderr)

    computed_rows = [r for r in rows if r.get("current_within_column_permutation_null", {}).get("status") == "computed"]
    n_current_nonzero = sum(1 for r in computed_rows if r["current_within_column_permutation_null"]["best_k"] > 0)
    n_row_renorm_nonzero = sum(1 for r in computed_rows if r["row_renormalized_permutation_null"]["best_k"] > 0)

    # NEAR-MISS VS ISOTROPIC. A detected rank of 0 alone is uninterpretable: it could mean the leading
    # eigenvalue sits just below its permutation threshold (a near miss -- the estimator is under-powered on
    # this object, real structure may still be there) or that the observed spectrum sits far inside the null
    # distribution (the unit-direction matrix really is close to isotropic once each trial's own magnitude is
    # removed, in which case "the occupied subspace" has no referent in these matrices at all). Declared
    # thresholds on the ratio of the observed leading eigenvalue to its own 95th-percentile null threshold,
    # applied identically to every cell: >= 0.9 is called a near miss, <= 0.5 is called far inside the null
    # (isotropic-leading-direction reading), anything between is intermediate and adjudicated as neither.
    current_ratios = [r["current_within_column_permutation_null"]["ratio_observed_to_null_threshold"]
                       for r in computed_rows
                       if r["current_within_column_permutation_null"]["ratio_observed_to_null_threshold"] is not None]
    n_near_miss = sum(1 for x in current_ratios if x >= 0.9)
    n_far_inside_null = sum(1 for x in current_ratios if x <= 0.5)
    n_intermediate = len(current_ratios) - n_near_miss - n_far_inside_null
    near_miss_vs_isotropic = {
        "near_miss_ratio_threshold": 0.9, "far_inside_null_ratio_threshold": 0.5,
        "current_null_top_position_ratios_observed_to_threshold": current_ratios,
        "n_cells_near_miss": n_near_miss, "n_cells_far_inside_null_isotropic_reading": n_far_inside_null,
        "n_cells_intermediate": n_intermediate,
        "candidate_reading_near_miss": (
            "if the observed leading eigenvalue sits just below its threshold, permutation_eigenvalue_"
            "threshold is under-powered on this object at this permutation count/session size, and real "
            "low-dimensional structure may still be present despite the rank-0 call."
        ),
        "candidate_reading_isotropic": (
            "if the observed leading eigenvalue sits far inside the null distribution, the unit-direction "
            "matrix's own leading-eigenvalue structure is close to what independent-column noise of the same "
            "marginals would produce once each trial's own magnitude is removed by unit-normalisation; in "
            "that reading, the phrase 'the occupied subspace' has no referent in these matrices at all, and "
            "the honest verdict for the whole occupied-space-membership question in this corpus is that it "
            "cannot establish there IS a low-dimensional occupied space for the axis to be inside of -- both "
            "the delivered 'outside' reading and this run's 'inside, beyond chance' reading would then be "
            "withdrawn as unmeasured, not adjudicated in either direction by this diagnostic alone."
        ),
        "caution_against_over_reading_isotropic": (
            "other analyses already delivered elsewhere in this project decode trial content from these same "
            "per-trial direction matrices, which is difficult to reconcile with a genuinely isotropic leading-"
            "eigenvalue structure; this diagnostic reports the eigenvalue numbers and both candidate readings "
            "without adjudicating beyond what six cells and one permutation-based rank estimator can support."
        ),
    }

    if n_row_renorm_nonzero > n_current_nonzero:
        conclusion = (
            "the_constraint_respecting_null_detects_subspaces_the_current_null_misses: the current within-"
            "column permutation null does not preserve the per-row unit-norm constraint the observed matrix "
            "satisfies, so its near-universal zero-rank result on real data is an artifact of that mis-"
            "specification rather than a genuine absence of detectable structure. The two rank estimators do "
            "not, in fact, disagree about these cells: entry_holdout_bicross_validation's positive detections "
            "stand without a contradicting second reading from a correctly-specified permutation null, "
            "though the delivered permutation_eigenvalue_threshold artifact itself is not amended by this "
            "diagnostic and its own zero-rank numbers remain as originally computed."
        )
    else:
        conclusion = (
            "the_constraint_respecting_null_also_detects_nothing: renormalising permuted rows to unit norm "
            "does not change the near-universal zero-rank outcome on these cells, so the disagreement between "
            "the two estimators is real and not an artifact of the row-norm mis-specification. The occupied-"
            "space-membership branch in results/occupied_subspace_membership_effect_size.json rests on "
            "entry_holdout_bicross_validation alone; permutation_eigenvalue_threshold, a second synthetically-"
            "gate-validated estimator, finds no subspace to test membership against at all on nearly every "
            "cell in both corpora. This caveat belongs beside the reported effect size everywhere it is quoted."
        )

    block = {
        "purpose": (
            "Tests whether permutation_eigenvalue_threshold's within-column permutation null is mis-"
            "specified for the per-trial unit-normalised matrix it is applied to (every row of the observed "
            "matrix has norm exactly 1; an independent per-column permutation does not preserve that "
            "constraint), by comparing the estimator's own null against an otherwise-identical null whose "
            "permuted rows are renormalised to unit norm before their spectrum is computed."
        ),
        "n_permutations": N_PERMUTATIONS,
        "cells_tested": rows,
        "n_cells_computed": len(computed_rows),
        "n_cells_current_null_detects_a_nonzero_rank": n_current_nonzero,
        "n_cells_row_renormalized_null_detects_a_nonzero_rank": n_row_renorm_nonzero,
        "conclusion": conclusion,
        "near_miss_vs_isotropic_diagnostic": near_miss_vs_isotropic,
        "neither_null_described_as_more_sensitive_or_preferable": (
            "This block reports what each null detects on the same cells under the same permutation draws; "
            "the constraint argument in the module docstring, not a comparison of detection counts alone, "
            "carries the reasoning for which null is correctly specified."
        ),
    }

    existing = json.loads(ARTIFACT_PATH.read_text())
    existing["permutation_null_row_norm_constraint_diagnostic"] = _json_safe(block)
    ARTIFACT_PATH.write_text(json.dumps(existing, indent=2, allow_nan=False, default=float))
    print(f"appended permutation_null_row_norm_constraint_diagnostic to {ARTIFACT_PATH}", file=sys.stderr)
    print(json.dumps({"n_cells_computed": len(computed_rows),
                       "n_current_nonzero": n_current_nonzero,
                       "n_row_renormalized_nonzero": n_row_renorm_nonzero,
                       "conclusion": conclusion.split(":")[0]}, indent=2))


if __name__ == "__main__":
    main()
