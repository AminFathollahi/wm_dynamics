"""run_occupied_subspace_rank_estimation.py -- builds a rank estimator for the occupied-state-space
decomposition that can actually return an interior rank, and re-answers whether the residual-direction
axis lies inside the occupied state space for the two macaque corpora the delivered decomposition covers.

THE ERROR THIS REPLACES, from first principles. The delivered rank selector picks an occupied-subspace
rank k by 5-fold cross-validated PCA reconstruction error: it fits a subspace on training trials and
reconstructs each held-out TRIAL using that trial's OWN coordinates in every candidate direction. Because
the candidate subspaces of increasing rank are nested, held-out squared error at rank k equals
||x||^2 minus the sum of the first k squared projections of x onto that fold's basis -- a quantity that is
non-increasing in k by construction, for every fold, every dataset and every true rank. The selector is
therefore mathematically incapable of returning anything but the top of its candidate range; "the axis
lies inside the occupied space" measured against a selector that always returns full ambient rank is a
tautology, not a measurement.

TWO ESTIMATORS THAT BREAK THE DEGENERACY, both from first principles in numpy.

(a) Held-out-ENTRY cross-validation (bi-cross-validation of a low-rank factorisation). Rows (trials) and
columns (units) are independently partitioned into fixed, seeded folds. For a test row-block and a test
column-block, the fully-held-in block (train rows x train columns, A11) supplies a rank-k truncated
pseudo-inverse; the held-out row-block's TRAIN columns (A21) are mapped through that pseudo-inverse and
back out through the fully-held-in block's TEST columns (A12) to predict the held-out row-block's TEST
columns (A22 = A21 @ pinv_k(A11) @ A12). The held-out entries A22 never enter their own prediction in any
direction. Unlike the nested-subspace degeneracy above, this has a genuine interior minimum: past the true
rank, inverting increasingly small singular values of A11 amplifies noise, so held-out error falls and then
rises again.

(b) Permutation eigenvalue threshold (parallel analysis). Each column of the unit-direction matrix is
independently permuted across trials, destroying between-unit covariance while preserving each unit's own
marginal distribution; the eigenvalue spectrum of U^T U is recomputed on the permuted matrix, repeated many
times, and the retained rank is the number of leading observed eigenvalues (counted contiguously from the
top) exceeding the upper 95th percentile of the correspondingly-ranked permuted eigenvalue.

SYNTHETIC VALIDATION IS A GATE, not a formality: both estimators (and, for contrast, the original selector,
imported unchanged) are run on low-rank-plus-noise matrices of known true rank before either is applied to
real data. An estimator that cannot recover a known rank within a pre-declared tolerance under favourable
(low-noise) conditions at a given ambient size is reported as failing the gate at that size, and any real
cell whose ambient unit count falls below that estimator's smallest validated size is flagged, never quietly
treated as measured.

THE MATCHED NULL IS ALSO ADDRESSED. The delivered comparison draws a random unit vector from the whole
AMBIENT space and asks whether the axis is closer to the occupied subspace than that is -- a near-guaranteed
win in high dimension, since a random ambient direction's expected component inside a much lower-dimensional
subspace is small regardless of whether the axis itself is special. A second, harder null is added here: a
random direction drawn UNIFORMLY WITHIN the estimated occupied subspace itself (matched on dimension, not on
the ambient space). That null's off-fraction is zero by construction up to floating-point round-off -- it is
not a classical hypothesis-test null but a floor, bracketing the ambient null's near-1 ceiling with a near-0
floor so the real axis's off-fraction can be read against both ends of the achievable range. Both nulls are
reported side by side, never averaged.

No estimator this module needs is reimplemented if it already exists unchanged elsewhere. The loaders
(`data_root`, `_macaque_bundles`, `_watters_bundles`, `_panichello_directory`, `_load_corpora_and_accounting`),
the residual-matrix and axis-entry builder (`_collect_axis_entries`), the leading-eigenvector routine
(`leading_eigenvector`), the ambient-space decomposition (`_occupied_space_decomposition`), the pooled
observed-versus-matched-null comparison (`pooled_off_fraction_against_matched_null`), the trial-count
weighting (`_trial_count_weighted`, `_weighted_combine_draws`), the branch classifier
(`classify_occupied_space_branch`) and the original degenerate selector (`_cv_pca_rank`, used here only as
the synthetic-data contrast column) are all imported unchanged. The only new code in this module is the two
rank estimators, the synthetic recovery gate, the within-subspace harder null, the whole-session cluster-
bootstrap detection floor, and the orchestration that combines all of this into a per-corpus verdict.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
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

from corpus_sessions import data_root  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from info_decoding import N_BOOT_SESSION_CLUSTER
from statistics import Z_80_POWER
from run_deviation_axis_structure import _macaque_bundles
from info_decoding import _collect_axis_entries
from info_decoding import _cv_pca_rank, _occupied_space_decomposition, _weighted_combine_draws, classify_occupied_space_branch, pooled_off_fraction_against_matched_null
from corpus_sessions import _watters_bundles
from subspace_identity import leading_eigenvector
from info_decoding import CORPORA, N_RANDOM_AXIS_DRAWS
from corpus_sessions import _panichello_directory
from run_multi_object_interference_and_locus_within_item_count import _trial_count_weighted
from subspace_identity import _contiguous_folds
from run_occupied_subspace_rank_selection_repair import _load_corpora_and_accounting  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import permutation_pvalue, stable_seed  # noqa: E402
from info_decoding import ESTIMATOR_NAMES, _gate_status_for_corpus  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "occupied_subspace_rank_estimation.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_occupied_subspace_rank_estimation"
ANALYSIS_VERSION = "2026-09-07"

N_ROW_FOLDS = 5
N_COL_FOLDS = 5
ENTRY_HOLDOUT_MAX_CANDIDATE_K = 25
N_PERMUTATIONS_EIGENVALUE_NULL = 500
SYNTHETIC_TRUE_RANKS = (0, 3, 8, 15)
SYNTHETIC_NOISE_LEVELS = {"low": 0.1, "medium": 1.0, "high": 3.0}  # noise sd as a multiple of signal sd
SYNTHETIC_NOISE_MODELS = ("isotropic", "diagonal_anisotropic")
SYNTHETIC_SIZES = (("small", 150, 30), ("medium", 500, 100), ("large", 900, 600))  # (label, n_trials, n_units)
SYNTHETIC_SEARCH_CEILING_CASES = (("entry_holdout_cap_1", 150, 30, 3, 1),)
SYNTHETIC_RANK_TOLERANCE = 3


DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "TWO ESTIMATORS, both new, both from first principles in numpy, applied to the identical (R, U, axis) "
    "objects the delivered occupied-state-space decomposition already builds.\n"
    "(a) entry_holdout_bicross_validation: rows are split into 5 fixed contiguous chronological folds "
    "(matching this project's other cross-validated estimators) and columns into 5 fixed, seeded random "
    "folds (units have no chronological order). For every row-fold x column-fold combination (25 total), a "
    "rank-k truncated pseudo-inverse of the fully-held-in block (train rows x train columns) predicts the "
    "held-out row-block's held-out columns from that same row-block's train columns, mapped back out "
    "through the fully-held-in block's held-out columns: predicted = A21 @ pinv_k(A11) @ A12. The held-out "
    "entries A22 never enter their own prediction. Candidate ranks run 1..min(25, smallest training-block "
    "dimension across all 25 combinations - 1); the cap of 25 is a declared computational simplification, "
    "validated against the synthetic recovery gate below (which tests true ranks up to 15 under this exact "
    "cap and recovers them), not a claim that no real cell could have a higher occupied rank. The selected "
    "rank is the candidate minimising total held-out squared error summed over all 25 combinations.\n"
    "(b) permutation_eigenvalue_threshold: each column of U is independently permuted across trials (500 "
    "permutations), destroying between-unit covariance while preserving each unit's own marginal "
    "distribution; the eigenvalue spectrum of U^T U is recomputed on each permuted matrix. The selected "
    "rank is the number of leading observed eigenvalues of U^T U (descending order, counted contiguously "
    "from the top) exceeding the upper 95th percentile of the correspondingly-ranked permuted eigenvalue.\n"
    "SYNTHETIC VALIDATION GATE, run before any real cell is touched: low-rank-plus-noise matrices at true "
    "ranks 3, 8 and 15, three noise levels (noise sd = 0.1, 1.0 and 3.0 times the signal's own sd) and "
    "three sizes matching the real corpora's observed (n_trials, n_units) range (150x30, 500x100, 900x600). "
    "An estimator PASSES the gate at a given size if, at the LOW noise level, its recovered rank is within "
    "3 (SYNTHETIC_RANK_TOLERANCE) of the true rank for all three true ranks tested at that size; medium and "
    "high noise recoveries are reported for information only and do not affect the pass/fail verdict, since "
    "the gate asks whether the estimator is CAPABLE of recovering an interior rank under favourable "
    "conditions, not how gracefully it degrades under adversarial noise. The original degenerate selector "
    "(_cv_pca_rank, imported unchanged) is run on the identical synthetic matrices and reported in the same "
    "table for contrast; it is expected, and reported if true, to always return the ambient ceiling "
    "regardless of true rank. Validation is SCALE-CONDITIONAL: a real cell is treated as validated for an "
    "estimator if the corpus's median ambient unit count across cells is at or above the smallest tested "
    "size at which that estimator passed the low-noise gate for every true rank tested at that size -- an "
    "estimator that only works above some ambient scale is still usable on cells above that scale, so a "
    "single global pass/fail across all tested sizes is not required in addition. Extrapolation beyond the "
    "largest tested size (600 ambient units) is not attempted. Otherwise (the corpus's median ambient unit "
    "count falls below every tested size at which an estimator passed), that estimator's rank on that "
    "corpus is still reported, but the corpus's final branch is "
    "'not_determinable_because_an_estimator_failed_its_synthetic_gate' rather than a measurement.\n"
    "TWO NULLS PER ESTIMATOR, reported side by side, never averaged. The ambient null (identical, unchanged, "
    "to the delivered pipeline's own _occupied_space_decomposition) draws >= 200 random unit vectors from "
    "the whole ambient space and computes their off-fraction against the estimator's selected-rank subspace. "
    "The harder, second null draws >= 200 random unit vectors UNIFORMLY WITHIN that same selected-rank "
    "subspace (matched on dimension, not on the ambient space) and computes THEIR off-fraction against the "
    "identical subspace -- zero by construction up to floating-point round-off, a floor rather than a "
    "hypothesis-test null, bracketing the ambient null's near-1 ceiling with a near-0 floor. off_fraction is "
    "a squared-norm ratio bounded below by zero, so any two-sided empirical test of it against a strictly "
    "positive null is effectively one-sided (a near-zero observed value has nowhere to go but up); this is "
    "disclosed for both nulls rather than adopted as a different statistic.\n"
    "BRANCHES, exactly one fired per corpus, decided in this order:\n"
    "  1. if either estimator is not validated for this corpus by the synthetic gate (see above) -> "
    "'not_determinable_because_an_estimator_failed_its_synthetic_gate'.\n"
    "  2. otherwise, per estimator, the axis is called 'inside' for that estimator if its pooled off-"
    "fraction is NOT significantly above EITHER null (ambient and harder, both at two-sided p <= 0.05 "
    "against that null's own centre). If both estimators call the axis inside -> "
    "'the_axis_lies_inside_the_occupied_space_under_both_estimators_and_beats_both_nulls'.\n"
    "  3. if exactly one estimator calls it inside -> "
    "'the_axis_lies_inside_the_occupied_space_under_one_estimator_only'.\n"
    "  4. if neither estimator calls it inside -> 'the_axis_lies_outside_the_occupied_space'.\n"
    "POWER: every pooled comparison carries a minimum detectable off-fraction difference at 80% power from "
    f"a whole-SESSION cluster bootstrap ({N_BOOT_SESSION_CLUSTER} draws, resample sessions with replacement, "
    f"recompute the pooled mean off-fraction each draw; mdd = {Z_80_POWER} * that bootstrap standard error), "
    "never a trial-count formula. off_fraction has no established reference effect size on the literature -- "
    "unlike this project's behavioural correlation estimators, there is no named threshold on this scale, so "
    "no comparison here is called a POWERED null on the strength of its mdd alone; the mdd is reported for "
    "transparency only."
)


# =======================================================================================================
# Checkpointing (mirrors this project's other long-running artifacts: one file per unit of work, temp
# file + os.replace, completion flag written only after the fit returns).
# =======================================================================================================

def _checkpoint_path(unit: str) -> Path:
    return CHECKPOINT_DIR / f"{unit.replace('/', '_')}.json"


def _load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if (not isinstance(data, dict) or data.get("_complete") is not True
            or data.get("analysis_version") != ANALYSIS_VERSION):
        return None
    return data["record"]


def _save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "analysis_version": ANALYSIS_VERSION, "record": _json_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


def _strip_null_draws(obj):
    """Recursively removes the (large, JSON-bulky) raw null-draw arrays from a nested record before it is
    written to the delivered artifact; checkpoints keep the full arrays so pooling can reuse them."""
    if isinstance(obj, dict):
        return {k: _strip_null_draws(v) for k, v in obj.items() if k != "null_off_fraction_draws"}
    if isinstance(obj, list):
        return [_strip_null_draws(v) for v in obj]
    return obj


# =======================================================================================================
# ESTIMATOR (a): held-out-entry bi-cross-validation
# =======================================================================================================

def _column_folds(p: int, n_col_folds: int, seed_tag: str) -> np.ndarray:
    """Fixed, seeded partition of columns (units) into n_col_folds groups. Units carry no chronological
    order (unlike trials), so column folds are a seeded random permutation split into contiguous chunks --
    fixed once per cell by seed_tag, never re-randomised across candidate ranks or draws."""
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|column_folds"))
    perm = rng.permutation(p)
    fold_id = np.empty(p, dtype=int)
    for j, idx in enumerate(np.array_split(perm, n_col_folds)):
        fold_id[idx] = j
    return fold_id


def entry_holdout_rank(U: np.ndarray, seed_tag: str, max_candidate_k: int = ENTRY_HOLDOUT_MAX_CANDIDATE_K,
                        n_row_folds: int = N_ROW_FOLDS, n_col_folds: int = N_COL_FOLDS) -> dict:
    n, p = U.shape
    row_folds = _contiguous_folds(n, n_row_folds)
    col_folds = _column_folds(p, n_col_folds, seed_tag)
    min_block = min(
        min(int(np.sum(row_folds != i)) for i in range(n_row_folds)),
        min(int(np.sum(col_folds != j)) for j in range(n_col_folds)),
    )
    max_k = min(max_candidate_k, min_block - 1)
    if max_k < 1:
        return {"status": "not_estimable", "reason": "every_training_block_too_small_for_any_candidate_rank"}
    candidate_ks = list(range(max_k + 1))
    errors = np.zeros(len(candidate_ks))
    n_blocks_used = 0
    for i in range(n_row_folds):
        test_rows, train_rows = row_folds == i, row_folds != i
        for j in range(n_col_folds):
            test_cols, train_cols = col_folds == j, col_folds != j
            A11 = U[np.ix_(train_rows, train_cols)]
            if min(A11.shape) < 2:
                continue
            A12 = U[np.ix_(train_rows, test_cols)]
            A21 = U[np.ix_(test_rows, train_cols)]
            A22 = U[np.ix_(test_rows, test_cols)]
            u_svd, s_svd, vt_svd = np.linalg.svd(A11, full_matrices=False)
            n_blocks_used += 1
            for idx, k in enumerate(candidate_ks):
                k_eff = min(k, s_svd.shape[0])
                sv = s_svd[:k_eff]
                keep = sv > (sv[0] * 1e-10 if sv.size else 0.0)
                inv_sv = np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)
                pinv_k = (vt_svd[:k_eff].T * inv_sv) @ u_svd[:, :k_eff].T
                pred = A21 @ pinv_k @ A12
                errors[idx] += float(np.sum((A22 - pred) ** 2))
    if n_blocks_used == 0:
        return {"status": "not_estimable", "reason": "no_usable_row_by_column_block"}
    best_idx = int(np.argmin(errors))
    if best_idx == len(candidate_ks) - 1:
        return {
            "status": "nonidentified_at_search_ceiling", "n_row_folds": n_row_folds,
            "n_col_folds": n_col_folds, "n_blocks_used": n_blocks_used,
            "candidate_ks": candidate_ks, "held_out_entry_reconstruction_error": errors.tolist(),
            "selected_k_at_search_ceiling": candidate_ks[best_idx], "best_k": None,
        }
    return {
        "status": "computed", "n_row_folds": n_row_folds, "n_col_folds": n_col_folds,
        "n_blocks_used": n_blocks_used, "candidate_ks": candidate_ks,
        "held_out_entry_reconstruction_error": errors.tolist(), "best_k": candidate_ks[best_idx],
        "best_k_is_interior": bool(0 < best_idx < len(candidate_ks) - 1),
    }


# =======================================================================================================
# ESTIMATOR (b): permutation eigenvalue threshold
# =======================================================================================================

def permutation_eigenvalue_rank(U: np.ndarray, seed_tag: str, n_perm: int = N_PERMUTATIONS_EIGENVALUE_NULL) -> dict:
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


# =======================================================================================================
# THE HARDER NULL -- a random direction confined to the estimated occupied subspace itself.
# =======================================================================================================

def _within_subspace_null_decomposition(U: np.ndarray, axis: np.ndarray, k: int, n_draws: int, seed_tag: str) -> dict:
    n, p = U.shape
    k_eff = min(k, min(n, p))
    if k_eff < 1 or np.sum(axis ** 2) <= 0.0:
        return {"status": "not_computable"}
    mean_u = U.mean(axis=0)
    _u_svd, _s_svd, vt = np.linalg.svd(U - mean_u, full_matrices=False)
    k_eff = min(k_eff, vt.shape[0])
    basis = vt[:k_eff].T
    within = basis @ (basis.T @ axis)
    within_frac = float(np.sum(within ** 2) / np.sum(axis ** 2))
    off_frac = 1.0 - within_frac
    rng = np.random.default_rng(stable_seed(seed_tag))
    coeffs = rng.standard_normal((n_draws, k_eff))
    coeffs /= np.linalg.norm(coeffs, axis=1, keepdims=True)
    null_vectors = coeffs @ basis.T  # n_draws x p, each exactly within span(basis) by construction
    null_off = 1.0 - np.sum((null_vectors @ basis) ** 2, axis=1)
    null_mean = float(np.mean(null_off))
    p_value = float(permutation_pvalue(np.abs(null_off - null_mean) >= abs(off_frac - null_mean)))
    return {
        "status": "computed", "k": k_eff, "within_fraction": within_frac, "off_fraction": off_frac,
        "null_off_fraction_mean": null_mean, "null_off_fraction_sd": float(np.std(null_off)),
        "two_sided_p_value": p_value, "off_fraction_above_null": bool(p_value <= 0.05 and off_frac > null_mean),
        "null_is_degenerate_by_construction": True,
        "null_off_fraction_draws": null_off,
    }


# =======================================================================================================
# SYNTHETIC RECOVERY GATE -- mandatory, run before any real cell.
# =======================================================================================================

def _generate_synthetic_matrix(n: int, p: int, true_rank: int, noise_level: float, seed_tag: str,
                               noise_model: str = "isotropic") -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed_tag))
    latent = rng.standard_normal((n, true_rank))
    loadings = rng.standard_normal((true_rank, p))
    signal = latent @ loadings
    signal_sd = float(np.std(signal)) if true_rank else 1.0
    noise_sd = noise_level * signal_sd
    if noise_model == "isotropic":
        scales = np.ones(p)
    elif noise_model == "diagonal_anisotropic":
        scales = np.geomspace(0.25, 4.0, p)
        scales /= np.sqrt(np.mean(scales ** 2))
    else:
        raise ValueError(f"unknown noise model: {noise_model}")
    return signal + rng.standard_normal((n, p)) * (noise_sd * scales)


def run_synthetic_recovery_gate() -> dict:
    rows = []
    for size_label, n, p in SYNTHETIC_SIZES:
        for true_rank in SYNTHETIC_TRUE_RANKS:
            if true_rank >= p:
                rows.append({"size": size_label, "n_trials": n, "n_units": p, "true_rank": true_rank,
                             "status": "skipped_true_rank_not_below_ambient_dimension"})
                continue
            for noise_label, noise_level in SYNTHETIC_NOISE_LEVELS.items():
                for noise_model in SYNTHETIC_NOISE_MODELS:
                    tag = (f"occupied_subspace_rank_estimation|synthetic|{size_label}|rank{true_rank}|"
                           f"{noise_label}|{noise_model}")
                    U = _generate_synthetic_matrix(n, p, true_rank, noise_level, f"{tag}|data", noise_model)
                    max_k_old = min(p, n - 1)
                    old = _cv_pca_rank(U, max_k_old, f"{tag}|old")
                    eh = entry_holdout_rank(U, f"{tag}|entry_holdout")
                    pe = permutation_eigenvalue_rank(U, f"{tag}|perm_eig")
                    rows.append({
                        "case_type": "recovery", "size": size_label, "n_trials": n, "n_units": p,
                        "trial_to_unit_ratio": n / p, "true_rank": true_rank,
                        "noise_level_label": noise_label, "noise_sd_relative_to_signal_sd": noise_level,
                        "noise_model": noise_model,
                        "old_degenerate_selector_recovered_rank": old.get("best_k"),
                        "old_degenerate_selector_status": old.get("status"),
                        "old_degenerate_selector_equals_ambient_ceiling": (
                            old.get("status") == "computed" and old.get("best_k") == max_k_old),
                        "entry_holdout_recovered_rank": eh.get("best_k"),
                        "entry_holdout_status": eh.get("status"),
                        "entry_holdout_selected_k_at_search_ceiling": eh.get("selected_k_at_search_ceiling"),
                        "entry_holdout_within_tolerance": (
                            eh.get("status") == "computed" and (
                                eh["best_k"] == 0 if true_rank == 0
                                else abs(eh["best_k"] - true_rank) <= SYNTHETIC_RANK_TOLERANCE
                            )),
                        "permutation_eigenvalue_recovered_rank": pe.get("best_k"),
                        "permutation_eigenvalue_status": pe.get("status"),
                        "permutation_eigenvalue_within_tolerance": (
                            pe.get("status") == "computed" and (
                                pe["best_k"] == 0 if true_rank == 0
                                else abs(pe["best_k"] - true_rank) <= SYNTHETIC_RANK_TOLERANCE
                            )),
                    })

    for case_label, n, p, true_rank, max_candidate_k in SYNTHETIC_SEARCH_CEILING_CASES:
        tag = f"occupied_subspace_rank_estimation|synthetic|{case_label}"
        U = _generate_synthetic_matrix(n, p, true_rank, 0.0, f"{tag}|data")
        eh = entry_holdout_rank(U, f"{tag}|entry_holdout", max_candidate_k=max_candidate_k)
        rows.append({
            "case_type": "search_ceiling", "size": case_label, "n_trials": n, "n_units": p,
            "trial_to_unit_ratio": n / p, "true_rank": true_rank, "max_candidate_k": max_candidate_k,
            "entry_holdout_recovered_rank": eh.get("best_k"), "entry_holdout_status": eh.get("status"),
            "entry_holdout_selected_k_at_search_ceiling": eh.get("selected_k_at_search_ceiling"),
        })

    def _passes_at_low_noise(column: str, size_label: str) -> bool:
        low_rows = [r for r in rows if r.get("case_type") == "recovery" and r.get("size") == size_label
                    and r.get("noise_level_label") == "low"]
        if not low_rows:
            return False
        return all(r.get(column) for r in low_rows)

    gate_by_estimator = {}
    for column, name in (("entry_holdout_within_tolerance", "entry_holdout_bicross_validation"),
                          ("permutation_eigenvalue_within_tolerance", "permutation_eigenvalue_threshold")):
        per_size_pass = {size_label: _passes_at_low_noise(column, size_label) for size_label, _, _ in SYNTHETIC_SIZES}
        validated_sizes = [p for (size_label, _, p) in SYNTHETIC_SIZES if per_size_pass[size_label]]
        gate_by_estimator[name] = {
            "pass_by_size_at_low_noise": per_size_pass,
            "passes_gate_overall": all(per_size_pass.values()),
            "minimum_validated_ambient_unit_count": (min(validated_sizes) if validated_sizes else None),
        }

    zero_rows = [r for r in rows if r.get("case_type") == "recovery" and r.get("true_rank") == 0]
    ceiling_rows = [r for r in rows if r.get("case_type") == "search_ceiling"]
    validation_summary = {
        "entry_holdout_bicross_validation": {
            "n_rank_zero_cases": len(zero_rows),
            "n_zero_rank_selected": sum(r.get("entry_holdout_status") == "computed"
                                        and r.get("entry_holdout_recovered_rank") == 0 for r in zero_rows),
            "n_positive_rank_selected": sum(r.get("entry_holdout_status") == "computed"
                                               and (r.get("entry_holdout_recovered_rank") or 0) > 0 for r in zero_rows),
            "n_search_ceiling_cases": len(ceiling_rows),
            "n_nonidentified_at_search_ceiling": sum(
                r.get("entry_holdout_status") == "nonidentified_at_search_ceiling" for r in ceiling_rows),
            "n_rank_forced_at_search_ceiling": sum(
                r.get("entry_holdout_recovered_rank") is not None for r in ceiling_rows),
        },
        "permutation_eigenvalue_threshold": {
            "n_rank_zero_cases": len(zero_rows),
            "n_zero_rank_selected": sum(r.get("permutation_eigenvalue_status") == "computed"
                                        and r.get("permutation_eigenvalue_recovered_rank") == 0 for r in zero_rows),
            "n_positive_rank_selected": sum(r.get("permutation_eigenvalue_status") == "computed"
                                               and (r.get("permutation_eigenvalue_recovered_rank") or 0) > 0 for r in zero_rows),
        },
    }

    return {
        "tolerance_absolute_rank_units": SYNTHETIC_RANK_TOLERANCE,
        "true_ranks_tested": list(SYNTHETIC_TRUE_RANKS),
        "noise_levels_tested": SYNTHETIC_NOISE_LEVELS,
        "noise_models_tested": list(SYNTHETIC_NOISE_MODELS),
        "sizes_tested": [{"label": s, "n_trials": n, "n_units": p, "trial_to_unit_ratio": n / p}
                         for s, n, p in SYNTHETIC_SIZES],
        "search_ceiling_cases": [{"label": label, "n_trials": n, "n_units": p,
                                  "true_rank": rank, "max_candidate_k": cap}
                                 for label, n, p, rank, cap in SYNTHETIC_SEARCH_CEILING_CASES],
        "n_permutations_eigenvalue_null": N_PERMUTATIONS_EIGENVALUE_NULL,
        "recovery_table": rows,
        "gate_by_estimator": gate_by_estimator,
        "validation_summary_by_estimator": validation_summary,
    }


# =======================================================================================================
# WHOLE-SESSION CLUSTER BOOTSTRAP MDD -- never a trial-count formula.
# =======================================================================================================

def _whole_session_cluster_bootstrap_mdd(values: list[float], seed_tag: str) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = int(arr.size)
    if n < 4:
        return {"status": "not_computable", "n_sessions": n}
    rng = np.random.default_rng(stable_seed(seed_tag))
    boot_means = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(N_BOOT_SESSION_CLUSTER)])
    se = float(np.std(boot_means, ddof=1))
    return {"status": "computed", "n_sessions": n, "n_bootstrap_draws": N_BOOT_SESSION_CLUSTER,
            "cluster_bootstrap_se": se, "z_multiplier": Z_80_POWER,
            "minimum_detectable_off_fraction_difference_at_80pct_power": Z_80_POWER * se}


# =======================================================================================================
# REAL-DATA APPLICATION -- reuses _collect_axis_entries and leading_eigenvector unchanged.
# =======================================================================================================

def _cell_under_estimator(U: np.ndarray, axis: np.ndarray, estimator_name: str, tag_base: str) -> dict:
    if estimator_name == "entry_holdout_bicross_validation":
        rank_result = entry_holdout_rank(U, f"{tag_base}|rank")
    else:
        rank_result = permutation_eigenvalue_rank(U, f"{tag_base}|rank")
    if rank_result.get("status") != "computed":
        return {"status": rank_result.get("status", "not_estimable"), "rank_selection": rank_result}
    if not rank_result.get("best_k"):
        # a genuine, informative outcome (rank 0: no eigenvalue/held-out gain beat its null) -- not an
        # estimator failure, but there is no subspace of rank >= 1 to decompose the axis against.
        return {"status": "zero_rank_no_occupied_subspace_to_test", "best_k": 0}
    k = rank_result["best_k"]
    ambient_null = _occupied_space_decomposition(U, axis, k, N_RANDOM_AXIS_DRAWS, f"{tag_base}|ambient_null")
    harder_null = _within_subspace_null_decomposition(U, axis, k, N_RANDOM_AXIS_DRAWS, f"{tag_base}|harder_null")
    return {
        "status": "computed", "best_k": k,
        "n_blocks_used": rank_result.get("n_blocks_used"), "n_permutations": rank_result.get("n_permutations"),
        "best_k_is_interior": rank_result.get("best_k_is_interior"),
        "ambient_null": ambient_null if ambient_null.get("status") == "computed"
        else {"status": ambient_null.get("status", "not_computable")},
        "harder_null": harder_null if harder_null.get("status") == "computed"
        else {"status": harder_null.get("status", "not_computable")},
    }


def _run_corpus(bundles: list[dict], corpus_key: str) -> dict:
    axis_entries = _collect_axis_entries(bundles, corpus_key)
    per_session = []
    for bundle in bundles:
        session = bundle["session"]
        unit = f"cells|{corpus_key}|{session}"
        cached = _load_checkpoint(unit)
        if cached is not None:
            per_session.append(cached)
            continue

        entries = axis_entries.get(session, [])
        if not entries:
            record = {"session": session, "status": "not_computable", "levels": []}
            _save_checkpoint(unit, record)
            per_session.append(record)
            continue

        level_records = []
        for entry in entries:
            R, U, n_trials, level = entry["R"], entry["U"], entry["n_trials"], entry["level"]
            axis = leading_eigenvector(R)
            tag_base = f"occupied_subspace_rank_estimation|{corpus_key}|{session}|{level}"
            record = {"n_trials": n_trials, "level": level, "n_units": int(U.shape[1])}
            for est in ESTIMATOR_NAMES:
                record[est] = _cell_under_estimator(U, axis, est, f"{tag_base}|{est}")
            level_records.append(record)

        record = {"session": session, "status": "computed", "levels": level_records}
        per_session.append(record)
        _save_checkpoint(unit, record)

    combined = {est: {"ambient_null": [], "harder_null": []} for est in ESTIMATOR_NAMES}
    for sess in per_session:
        if sess.get("status") != "computed":
            continue
        for est in ESTIMATOR_NAMES:
            for null_name in ("ambient_null", "harder_null"):
                computed_levels = [lvl for lvl in sess["levels"]
                                    if lvl.get(est, {}).get(null_name, {}).get("status") == "computed"]
                if not computed_levels:
                    continue
                off = _trial_count_weighted([(lvl["n_trials"], lvl[est][null_name]["off_fraction"])
                                              for lvl in computed_levels])
                draws = _weighted_combine_draws([(lvl["n_trials"], lvl[est][null_name]["null_off_fraction_draws"])
                                                  for lvl in computed_levels])
                combined[est][null_name].append({"session": sess["session"], "off_fraction": off, "null_draws": draws})

    pooled = {}
    for est in ESTIMATOR_NAMES:
        pooled[est] = {}
        for null_name in ("ambient_null", "harder_null"):
            sessions = combined[est][null_name]
            off_fracs = [s["off_fraction"] for s in sessions if s.get("off_fraction") is not None]
            pooled_test = slope_across_sessions_test(off_fracs, alternative="two-sided") if off_fracs else {"status": "not_computed"}
            mdd = _whole_session_cluster_bootstrap_mdd(
                off_fracs, f"occupied_subspace_rank_estimation|mdd|{corpus_key}|{est}|{null_name}")
            comparison = pooled_off_fraction_against_matched_null(
                [s for s in sessions if s.get("null_draws") is not None])
            branch = classify_occupied_space_branch(comparison)
            pooled[est][null_name] = {
                "n_sessions_pooled": len(off_fracs),
                "pooled_off_fraction_across_sessions": pooled_test,
                "minimum_detectable_off_fraction_difference_at_80pct_power_whole_session_cluster_bootstrap": mdd,
                "pooled_observed_vs_matched_null": comparison,
                "branch": branch,
            }

    n_units_all = [lvl["n_units"] for sess in per_session if sess.get("status") == "computed" for lvl in sess["levels"]]
    n_cells = len(n_units_all)

    return {
        "n_sessions_total": len(bundles),
        "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "n_cells": n_cells,
        "median_ambient_unit_count_across_cells": (float(np.median(n_units_all)) if n_units_all else None),
        "per_session": per_session,
        "pooled_by_estimator_and_null": pooled,
    }




def _rank_identification_summary(corpus_result: dict) -> dict:
    summary = {}
    for estimator in ESTIMATOR_NAMES:
        counts = {"identified_positive_rank": 0, "zero_rank": 0, "search_ceiling": 0, "other_unresolved": 0}
        for session in corpus_result["per_session"]:
            for level in session.get("levels", []):
                result = level.get(estimator, {})
                status = result.get("status")
                at_ceiling = estimator == "entry_holdout_bicross_validation" and (
                    status == "nonidentified_at_search_ceiling"
                    or (status == "computed" and result.get("best_k_is_interior") is False)
                )
                if at_ceiling:
                    counts["search_ceiling"] += 1
                elif status == "zero_rank_no_occupied_subspace_to_test":
                    counts["zero_rank"] += 1
                elif status == "computed" and result.get("best_k", 0) > 0:
                    counts["identified_positive_rank"] += 1
                else:
                    counts["other_unresolved"] += 1
        total = sum(counts.values())
        summary[estimator] = {
            **counts,
            "n_cells": total,
            "identified_positive_rank_fraction": counts["identified_positive_rank"] / total if total else None,
        }
    return summary


def classify_final_branch(corpus_result: dict, gate_status: dict) -> str:
    if not all(v["validated_for_this_corpus"] for v in gate_status.values()):
        return "not_determinable_because_an_estimator_failed_its_synthetic_gate"
    rank_summary = corpus_result.get("rank_identification_summary") or _rank_identification_summary(corpus_result)
    if not all(
        value["identified_positive_rank_fraction"] is not None
        and value["identified_positive_rank_fraction"] > 0.5
        for value in rank_summary.values()
    ):
        return "not_determinable_because_rank_selection_is_nonidentified_or_disagrees"
    closer_than_ambient = {}
    for est_name in ESTIMATOR_NAMES:
        pooled = corpus_result["pooled_by_estimator_and_null"][est_name]
        amb = pooled["ambient_null"]["pooled_observed_vs_matched_null"]
        closer_than_ambient[est_name] = (
            amb.get("status") == "computed" and not amb.get("significant_above_null")
        )
    n_inside = sum(closer_than_ambient.values())
    if n_inside == len(ESTIMATOR_NAMES):
        return "the_axis_is_closer_to_the_occupied_space_than_ambient_directions_under_both_estimators"
    if n_inside == 1:
        return "the_axis_is_closer_to_the_occupied_space_than_ambient_directions_under_one_estimator_only"
    return "the_axis_is_not_closer_to_the_occupied_space_than_ambient_directions_under_either_estimator"


# =======================================================================================================
# ORCHESTRATION
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Two new rank estimators (a held-out-entry bi-cross-validation and a permutation eigenvalue "
            "threshold), each first validated on synthetic low-rank-plus-noise data of known rank, are "
            "applied to the same session-by-item-count-level cells the delivered occupied-state-space "
            "decomposition covers in the single-item macaque lateral prefrontal cortex corpus (11 sessions) "
            "and the multi-object macaque corpus (41 sessions, multiple item-count levels per session). Each "
            "estimator's selected rank is used to redo the within-occupied-space decomposition of the same "
            "residual-direction axis the delivered decomposition analyses, against two nulls: a random "
            "direction in the whole ambient space (identical to the delivered comparison) and a random "
            "direction confined to the estimated occupied subspace itself. The two estimators and two nulls "
            "are reported separately throughout, never averaged, and a synthetic-gate failure at either "
            "estimator's real-data scale is reported as its own branch rather than silently ignored."
        ),
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    _log("running the mandatory synthetic recovery gate before touching any real cell")
    gate = run_synthetic_recovery_gate()
    output["synthetic_recovery_gate"] = gate
    _flush(output)
    for est_name, g in gate["gate_by_estimator"].items():
        _log(f"  gate: {est_name} passes_overall={g['passes_gate_overall']} "
             f"min_validated_p={g['minimum_validated_ambient_unit_count']} elapsed={time.time() - t0:.0f}s")

    _log("loading both corpora (identical loaders to run_deviation_axis_structure.py)")
    loaded, zero_drop = _load_corpora_and_accounting(root)
    gate_result = loaded["gate_result"]
    bundles = loaded["bundles"]
    output["reproduction_gate"] = {"status": gate_result["status"]}
    output["zero_drop_accounting"] = zero_drop
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}; "
         f"macaque bundles={len(bundles[CORPORA[0]])} multi-object bundles={len(bundles[CORPORA[1]])} "
         f"elapsed={time.time() - t0:.0f}s")

    if gate_result["status"] != "reproduced_exactly":
        output["status"] = "void_reproduction_gate_did_not_reproduce"
        output["git_commit"] = git_commit(ROOT)
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce; no new number was read")
        return

    output["occupied_state_space_block_reestimated"] = {}
    branches = {}
    for corpus_key in CORPORA:
        _log(f"occupied-state-space rank re-estimation: {corpus_key} ({len(bundles[corpus_key])} sessions)")
        corpus_result = _run_corpus(bundles[corpus_key], corpus_key)
        gate_status = _gate_status_for_corpus(corpus_result, gate)
        corpus_result["synthetic_gate_status_for_this_corpus"] = gate_status
        corpus_result["rank_identification_summary"] = _rank_identification_summary(corpus_result)
        corpus_result["branch"] = classify_final_branch(corpus_result, gate_status)
        corpus_result = _strip_null_draws(corpus_result)
        output["occupied_state_space_block_reestimated"][corpus_key] = corpus_result
        branches[corpus_key] = corpus_result["branch"]
        _flush(output)
        _log(f"  {corpus_key}: branch={corpus_result['branch']} n_cells={corpus_result['n_cells']} "
             f"elapsed={time.time() - t0:.0f}s")

    output["branch"] = branches
    output["git_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({"reproduction_gate": gate_result["status"], "branch": branches,
                       "wall_clock_s": output["wall_clock_s"]}, indent=2, default=float))
    _log("FINISHED run_occupied_subspace_rank_estimation")


if __name__ == "__main__":
    main()
