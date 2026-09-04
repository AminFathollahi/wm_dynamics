"""run_deviation_axis_structure.py -- the accuracy-predicting state deviation is measured only as a
magnitude (1 - cosine to a leave-one-out reference direction). This module asks whether the part of a
trial's direction that produces that magnitude -- its residual after removing the reference direction --
points somewhere in particular, or scatters with no preferred orientation.

Construction. For a trial with unit-normalised activity direction u_i and leave-one-out reference m_i
(the same two objects rate_free_state_deviation reduces to a single scalar), the residual is
r_i = u_i - (u_i . m_i) m_i, the component of u_i orthogonal to m_i. Trials whose residual norm falls
below a numerical floor (the leave-one-out mean and the trial's own direction coincide to within floating-
point precision, so no direction is defined) are excluded by name and counted, never silently dropped;
the floor is fixed at 1e-8, the same tolerance the delivered decomposition-identity check already uses to
call two floating-point quantities equal, since no smaller difference is distinguishable from rounding
error in this construction.

Per session, the rows r_i / ||r_i|| are stacked into a trials-by-units matrix R and eigen-decomposed via
R^T R. The participation ratio (sum of eigenvalues)^2 / (sum of squared eigenvalues) and the leading
eigenvalue fraction are reported against a rotation null that redraws each trial's residual direction as a
uniformly random unit vector in the subspace orthogonal to that SAME trial's own reference m_i -- holding
trial count, unit count and the orthogonality constraint fixed -- rather than against an arbitrary
reference such as zero, because a small trial-to-unit ratio biases the participation ratio downward on its
own and only a null built from the identical construction shares that bias.

Scope is set by an existing, already-measured precondition, not chosen here: only the single-item macaque
lateral prefrontal cortex corpus (Panichello et al. 2024) and the multi-object macaque corpus (Watters,
Gabel, Tenenbaum and Jazayeri; DANDI 000620) have a rate-free deviation that passes its own orthogonality
gate against total spike count. The mouse and both human maintenance-delay corpora fail that gate and are
excluded on that measured precondition. The multi-object corpus is analysed WITHIN item-count level
throughout (a different memorandum cardinality is a different task condition) and combined across levels
by the trial-count-weighted average this corpus's own primary behavioural estimator already uses; a
pooled-across-item-count number is never reported as this corpus's effect size.

SIGN CONVENTION. Every behavioural coefficient reported here is expressed against WORSE behaviour,
applied once at the point a result is packaged, never inside an estimator: the single-item corpus's
native outcome is trial correctness and is negated once; the multi-object corpus's native outcome is
continuous report error and needs no flip.

No estimator is forked. rate_free_state_deviation, full_reproduction_gate, residual_decomposition_and_
identity_check, _leave_one_out_unit_directions, _orthonormal_basis, _random_orthonormal_basis,
_macaque_session_bundle, _watters_session_bundle, _watters_bundle_with_label, unit_direction_vectors,
SIGN_TO_WORSE_BEHAVIOUR, CONTENT_LABEL_K_CLASSES, SHARP_TEST_MIN_CLASSES, SHARP_TEST_MIN_PER_CLASS,
_reachable_sessions, _contiguous_folds, _trial_count_weighted, _bias_only_reproduces,
partial_correlation_permutation_test, slope_across_sessions_test, minimum_detectable_paired_difference,
permutation_pvalue, stable_seed and fdr_bh are every one imported unchanged from where this project
already defines them. The only new functions this module introduces are the residual-row construction
with its numerical floor, the rotation-null machinery for a participation ratio and for a directional
alignment, the four alignment-reference constructions the axis-alignment analysis needs, the fold-wise
axis-stability check, the cross-validated signed/unsigned projection comparison the signed-displacement
test needs, and the occupied-state-space
decomposition the occupied-state-space analysis below needs.
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

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_behavior_amplitude_rate_controls import _reachable_sessions  # noqa: E402
from run_component_and_content_specific_serial_pull import (  # noqa: E402
    _bias_only_reproduces, _trial_count_weighted, _watters_bundle_with_label,
)
from run_component_effect_size_and_anatomy import N_CV_FOLDS, _contiguous_folds  # noqa: E402
from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    CONTENT_LABEL_K_CLASSES, SHARP_TEST_MIN_CLASSES, SHARP_TEST_MIN_PER_CLASS, SIGN_TO_WORSE_BEHAVIOUR,
    _macaque_session_bundle, _panichello_directory, _watters_session_bundle, full_reproduction_gate,
    unit_direction_vectors,
)
from run_deviation_subspace_decomposition import (  # noqa: E402
    IDENTITY_TOLERANCE, WATTERS_REGRESSION_DIM, _leave_one_out_unit_directions, _orthonormal_basis,
    residual_decomposition_and_identity_check,
)
from run_deviation_serial_dependence_and_temporal_locus import (  # noqa: E402
    MACAQUE_DEVIATION_RAW_R as _MACAQUE_DELIVERED_RAW_R, WATTERS_DEVIATION_RAW_R as _WATTERS_DELIVERED_RAW_R,
)
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, rate_free_state_deviation,
)
from run_watters_state_geometry import MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    fdr_bh, minimum_detectable_paired_difference, partial_correlation_permutation_test, permutation_pvalue,
    stable_seed,
)

OUTPUT_PATH = ROOT / "results" / "deviation_axis_structure.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_deviation_axis_structure"
ANALYSIS_VERSION = "2026-08-24"

N_PERM = 10000
N_ROTATION_DRAWS = 1000
N_RANDOM_AXIS_DRAWS = 200
REPRODUCTION_TOLERANCE = 1e-6
RESIDUAL_NORM_FLOOR = 1e-8  # matches IDENTITY_TOLERANCE's floating-point-equality tolerance
MIN_FOLD_TRIALS = 8

CORPORA = ("panichello_2024_macaque_lPFC_single_item", "watters_2026_macaque_multi_object")

ANISOTROPY_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per corpus: the pooled participation ratio (macaque: per session; multi-object corpus: per session, "
    "combined across item-count levels by the trial-count-weighted average) against its rotation null, "
    "two-sided empirical p-value. Reading 'below the null' as the participation ratio's pooled mean sitting "
    "below the pooled null distribution's mean:\n"
    "  - participation ratio significantly BELOW the null AND the leading eigenvalue fraction is ALSO "
    "significantly ABOVE its own null -> 'residual_directions_concentrate_on_an_axis'.\n"
    "  - participation ratio significantly BELOW the null AND the leading eigenvalue fraction is NOT "
    "significantly above its own null -> 'residual_directions_are_anisotropic_but_not_on_a_single_axis', "
    "asserting no axis count.\n"
    "  - participation ratio NOT significantly below the null, AND the minimum detectable paired difference "
    "of the participation ratio lies below the reduction in participation ratio a single dominant axis with "
    "the corpus's own observed leading eigenvalue fraction would produce (computed in closed form: a single "
    "axis carrying fraction f of the total variance with the remaining D-1 dimensions equally weighted gives "
    "participation ratio 1 / (f^2 + (1-f)^2/(D-1)), D fixed at the corpus's own median unit count minus one) "
    "-> 'residual_directions_are_isotropic', with that comparison stated numerically.\n"
    "  - participation ratio NOT significantly below the null and the minimum detectable difference is at or "
    "above that single-axis reference -> 'inconclusive_below_detection_floor', never quoted without both "
    "numbers.\n"
    "Any outcome this classifier cannot place (e.g. the pooled test itself not computable) is reported as "
    "'not_computable' with the reason, never forced onto a named branch."
)

AXIS_ALIGNMENT_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Runs only when the anisotropy test fires 'residual_directions_concentrate_on_an_axis' or 'residual_directions_"
    "are_anisotropic_but_not_on_a_single_axis' for that corpus -- both are cases where the participation "
    "ratio is significantly below its null and a single leading eigenvector is a well-defined object to "
    "align, whether or not it alone explains the anisotropy. Per session (multi-object corpus: per item-"
    "count level, combined by trial-count weighting), the leading eigenvector of R^T R is aligned, by "
    "absolute cosine (vector references) or principal angle (the memorandum subspace), against each of the "
    "total-spike-count direction, the memorandum-coding subspace, the slow-drift direction, and the "
    "preceding trial's class-mean direction, each against its own rotation null of >= 1000 draws, two-"
    "sided empirical p-value, no ranking asserted across the four without a direct paired session-level "
    "test of that specific pair. Axis stability is additionally assessed by splitting each session's "
    "trials into 5 contiguous chronological folds, fitting the axis independently within each fold, and "
    "testing the pooled absolute cosine between all fold-pairs against a rotation null (one-sided: only "
    "stability ABOVE chance is a meaningful claim here). If axis stability is not significant, the signed-"
    "displacement test is not run for that corpus and this module says so rather than reporting its numbers as if the axis "
    "were a stable session-level object."
)

SIGNED_DISPLACEMENT_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Runs only when the anisotropy test fires a concentration branch for a corpus AND that corpus's own axis-"
    "stability is significant. Five contiguous chronological folds; per fold, the leading eigenvector is "
    "fit on the other four folds' own residual rows, its sign fixed on those same training trials by "
    "requiring the mean signed projection of training trials at or above their own fold's median worse-"
    "behaviour value to be positive, then applied to the held-out fold's own residual rows (already "
    "computed by the session's whole-epoch leave-one-out reference, never refit per fold) to give the "
    "signed projection s_i. |s_i| and the delivered deviation d_i are the two comparison observables. "
    "Each observable's association with worse behaviour is reported two ways: within-fold (Pearson r per "
    "fold, combined by trial-count weighting) and pooled-across-fold (one correlation on every held-out "
    "trial's value, from partial_correlation_permutation_test), neither called the corrected one. A bias-"
    "only control (each held-out trial's signed projection replaced by its own fold's training-trial mean "
    "signed projection) is compared against the real signed-projection result by the same test the "
    "delivered content-specific-pull module uses to decide whether a result reproduces under a session-"
    "level offset; if it reproduces, the branch is 'axis_projection_not_separable_from_a_session_level_"
    "offset' regardless of what the paired tests below say. Otherwise, direct paired session-level tests "
    "(signed vs unsigned, signed vs delivered, unsigned vs delivered), in both estimator versions, "
    "Benjamini-Hochberg corrected across the resulting family:\n"
    "  - signed significantly outpredicts BOTH unsigned and delivered, both estimator versions -> "
    "'the_component_is_a_signed_displacement_along_an_axis'.\n"
    "  - signed does not outpredict delivered and the paired test's minimum detectable difference is below "
    "the delivered deviation's own pooled behavioural effect size -> "
    "'the_component_is_a_magnitude_not_a_signed_displacement'.\n"
    "  - no paired test significant and every minimum detectable difference at or above the delivered "
    "deviation's own effect size -> 'signed_and_unsigned_are_not_distinguishable_at_this_power'.\n"
    "  - the within-fold and pooled-across-fold versions disagree on which named branch above applies -> "
    "'ordering_holds_under_one_estimator_only', both versions reported, disagreement stated, never resolved."
)

OCCUPIED_SPACE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per corpus (multi-object corpus: per item-count level, combined by trial-count weighting), the "
    "occupied subspace is estimated from the session's own unit-direction vectors by 5-fold cross-"
    "validated PCA reconstruction error (contiguous chronological folds, matching this project's other "
    "cross-validated estimators, in place of leave-one-trial-out for compute tractability -- a disclosed "
    "simplification): the rank k in 1..min(n_units, n_trials-1) minimizing mean held-out reconstruction "
    "error is the cross-validated occupied dimensionality. The anisotropy test's leading eigenvector is decomposed "
    "into its component inside the top-k PCA subspace (fit on every kept trial, not held out, since the "
    "axis itself is already a fixed session-level object) and its component outside, reported as a "
    "fraction of squared norm.\n"
    "THE COMPARISON IS AGAINST A MATCHED-DIMENSION RANDOM-AXIS NULL, NEVER AGAINST ZERO. For every level "
    "of every session, >= 200 random unit directions drawn uniformly in the same unit space are pushed "
    "through the identical decomposition at the same rank and with the same basis, giving that level its "
    "own null distribution of off-occupied fractions; per session these are combined across item-count "
    "levels by trial-count weighting, draw-index aligned. Pooled across sessions: the pooled null "
    "distribution is the mean across sessions of the per-session null draws, draw-index aligned (the same "
    "pooling convention the anisotropy block uses for its rotation nulls); the pooled observed off-fraction "
    "is the mean across sessions of the per-session observed off-fractions; the two are compared by a "
    "two-sided empirical percentile test of the pooled observed value against the pooled null's central "
    "mass. The branch verdict carries the pooled observed off-fraction, the pooled null mean and sd, and "
    "the p-value in the same object. The same decomposition is repeated at full rank (k = min(n_units, "
    "n_trials-1), no truncation) as a robustness furniture: at full rank the occupied subspace is the entire "
    "span of the observed trial-to-trial variability, so an outside fraction distinguishable from zero there "
    "would mean the axis leaves even that unrestricted span, a stronger and different claim from leaving the "
    "cross-validated k. Branches:\n"
    "  - pooled comparison not computable (too few sessions or too few finite null draws) -> "
    "'not_separable_at_the_available_dimensionality'.\n"
    "  - the pooled observed off-fraction lies significantly ABOVE the pooled null's central mass (two-sided "
    "empirical p <= 0.05 and observed > null centre) -> 'the_axis_lies_outside_the_occupied_state_space'.\n"
    "  - otherwise -- the pooled observed off-fraction sits at or below the matched null's central mass -- "
    "'the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace' (the memorandum "
    "subspace is already known, from the axis-alignment analysis or the delivered subspace decomposition, to "
    "be lower-dimensional than the occupied space). An observed off-fraction far BELOW the null is the "
    "strongest form of this branch: it means the axis leaves the top-k occupied subspace less than even a "
    "uniformly random direction would."
)


# =======================================================================================================
# Checkpointing: one file per unit of work, temp file + os.replace, completion flag written only after
# the fit returns.
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
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def _save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "record": _json_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _run_checkpointed(unit: str, fit_fn):
    cached = _load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    _save_checkpoint(unit, record)
    return record


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Residual-row construction: the numerical floor, and the trials-by-units unit-residual matrix.
# =======================================================================================================

def _residual_rows(activity_by_unit: np.ndarray, floor: float = RESIDUAL_NORM_FLOOR) -> dict:
    identity = residual_decomposition_and_identity_check(activity_by_unit)
    loo_mean = _leave_one_out_unit_directions(activity_by_unit)["loo_mean_normalized"]
    finite = identity["finite"]
    residual = identity["residual"]
    residual_norm = np.linalg.norm(residual, axis=1)
    with np.errstate(invalid="ignore"):
        above_floor = finite & (residual_norm >= floor)
    return {
        "identity": identity, "loo_mean": loo_mean, "residual": residual, "residual_norm": residual_norm,
        "keep": above_floor, "n_trials_with_defined_direction": int(finite.sum()),
        "n_trials_excluded_by_residual_floor": int((finite & ~above_floor).sum()),
        "n_kept": int(above_floor.sum()),
    }


def _unit_residual_matrix(rows: dict) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(rows["keep"])
    R = rows["residual"][idx] / rows["residual_norm"][idx, None]
    return R, idx


def participation_ratio_and_leading_fraction(R: np.ndarray) -> dict | None:
    if R.shape[0] < 2 or R.shape[1] < 2:
        return None
    eigvals = np.clip(np.linalg.eigvalsh(R.T @ R), 0.0, None)
    total = float(eigvals.sum())
    if total <= 0.0:
        return None
    return {
        "participation_ratio": float((total ** 2) / float(np.sum(eigvals ** 2))),
        "leading_eigenvalue_fraction": float(eigvals.max() / total),
        "n_units": int(R.shape[1]), "n_trials": int(R.shape[0]),
    }


def leading_eigenvector(R: np.ndarray) -> np.ndarray:
    _w, v = np.linalg.eigh(R.T @ R)
    return v[:, -1]


def rotation_null_draws(loo_mean_kept: np.ndarray, n_draws: int, seed_tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over trials within a draw: for every kept trial simultaneously, a random Gaussian in
    unit space, projected off that SAME trial's own reference direction and renormalised -- a uniformly
    random unit vector in the subspace orthogonal to m_i, holding trial count, unit count and the
    orthogonality constraint fixed."""
    n, n_units = loo_mean_kept.shape
    rng = np.random.default_rng(stable_seed(seed_tag))
    pr_draws = np.full(n_draws, np.nan)
    lead_draws = np.full(n_draws, np.nan)
    for d in range(n_draws):
        g = rng.standard_normal((n, n_units))
        g = g - np.sum(g * loo_mean_kept, axis=1, keepdims=True) * loo_mean_kept
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        if (norms <= 0.0).any():
            continue
        stat = participation_ratio_and_leading_fraction(g / norms)
        if stat is not None:
            pr_draws[d] = stat["participation_ratio"]
            lead_draws[d] = stat["leading_eigenvalue_fraction"]
    return pr_draws, lead_draws


def _weighted_combine_draws(entries: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    """Trial-count-weighted combination of per-level null-draw arrays into one session-level array, the
    same weighting _trial_count_weighted applies to scalars, applied here draw-index by draw-index."""
    if not entries:
        return None
    n_arr = np.array([n for n, _ in entries], dtype=float)
    stack = np.array([d for _, d in entries], dtype=float)
    valid = np.isfinite(stack)
    weights = n_arr[:, None] * valid
    denom = weights.sum(axis=0)
    numer = np.nansum(np.where(valid, stack, 0.0) * n_arr[:, None], axis=0)
    return np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)


def _pool_rotation_statistic(session_records: list[dict]) -> dict:
    """Pools a real per-session scalar via the project's paired sign-flip test and separately pools each
    session's own rotation-null draws (mean across sessions per draw index, the same construction
    _pool_adjacency in run_deviation_serial_dependence_and_temporal_locus.py already uses), then reports a
    two-sided empirical p-value of the real pooled mean against that pooled null distribution."""
    observed = [r["observed"] for r in session_records if r.get("observed") is not None]
    real_pooled = slope_across_sessions_test(observed, alternative="two-sided") if observed else {"status": "not_computed"}
    mdd = minimum_detectable_paired_difference(observed) if len(observed) >= 2 else {"status": "not_computable", "n": len(observed)}
    out = {
        "n_sessions": len(observed), "real_pooled": real_pooled,
        "minimum_detectable_difference_80pct_power": mdd,
        "pooled_null_mean": None, "pooled_null_sd": None, "two_sided_empirical_p_value": None,
        "significant": False, "below_null": None,
    }
    draws_list = [np.asarray(r["null_draws"], dtype=float) for r in session_records if r.get("null_draws") is not None]
    if not draws_list or real_pooled.get("status") != "tested":
        return out
    pooled_null = np.nanmean(np.stack(draws_list), axis=0)
    finite_null = pooled_null[np.isfinite(pooled_null)]
    if finite_null.size < 10:
        return out
    null_center = float(np.mean(finite_null))
    real_mean = real_pooled["mean_value"]
    p = float(permutation_pvalue(np.abs(finite_null - null_center) >= abs(real_mean - null_center)))
    out.update({
        "pooled_null_mean": null_center, "pooled_null_sd": float(np.std(finite_null)),
        "two_sided_empirical_p_value": p, "significant": bool(p <= 0.05), "below_null": bool(real_mean < null_center),
    })
    return out


def _single_axis_reference_participation_ratio(leading_fraction: float, dim: float) -> float | None:
    """Closed-form participation ratio of a synthetic residual set concentrated on one axis with the
    given leading eigenvalue fraction, the remaining (dim - 1) dimensions weighted equally -- the
    substitution ANISOTROPY_DECISION_RULE_DECLARED_BEFORE_FITTING names for the isotropic branch's power
    comparison."""
    if dim <= 1 or not (0.0 <= leading_fraction <= 1.0):
        return None
    denom = leading_fraction ** 2 + ((1.0 - leading_fraction) ** 2) / (dim - 1.0)
    return (1.0 / denom) if denom > 0 else None


def classify_residual_direction_structure(pooled_pr: dict, pooled_lead: dict, dim_reference: float) -> str:
    if pooled_pr.get("real_pooled", {}).get("status") != "tested":
        return "not_computable"
    pr_below_null = bool(pooled_pr["significant"] and pooled_pr["below_null"])
    if pr_below_null:
        lead_above_null = bool(
            pooled_lead.get("significant") and pooled_lead.get("below_null") is False
            and pooled_lead.get("real_pooled", {}).get("status") == "tested"
            and pooled_lead["pooled_null_mean"] is not None
            and pooled_lead["real_pooled"]["mean_value"] > pooled_lead["pooled_null_mean"]
        )
        return "residual_directions_concentrate_on_an_axis" if lead_above_null else \
            "residual_directions_are_anisotropic_but_not_on_a_single_axis"
    mdd = pooled_pr["minimum_detectable_difference_80pct_power"]
    mdd_value = mdd.get("mdd") if isinstance(mdd, dict) and mdd.get("status") == "computed" else None
    null_mean = pooled_pr.get("pooled_null_mean")
    observed_lead = (pooled_lead.get("real_pooled", {}).get("mean_value")
                      if pooled_lead.get("real_pooled", {}).get("status") == "tested" else None)
    single_axis_pr = (_single_axis_reference_participation_ratio(observed_lead, dim_reference)
                       if observed_lead is not None else None)
    if mdd_value is not None and null_mean is not None and single_axis_pr is not None:
        single_axis_gap = null_mean - single_axis_pr
        if mdd_value < single_axis_gap:
            return "residual_directions_are_isotropic"
    return "inconclusive_below_detection_floor"


# =======================================================================================================
# Session bundle construction, reusing every delivered loader unchanged.
# =======================================================================================================

def _macaque_bundles(root: Path) -> tuple[list[dict], list[Path]]:
    paths = _reachable_sessions(root)
    bundles = []
    for path in paths:
        bundle = _run_checkpointed(f"macaque_bundle|{path.stem}", lambda p=path: _json_safe(_macaque_session_bundle(p)))
        if bundle is not None:
            bundle["activity_by_unit"] = np.asarray(bundle["activity_by_unit"])
            bundle["deviation"] = np.asarray(bundle["deviation"])
            bundle["outcome_raw"] = np.asarray(bundle["outcome_raw"])
            bundle["spike_count"] = np.asarray(bundle["spike_count"])
            bundle["trial_index"] = np.asarray(bundle["trial_index"])
            bundle["memorandum_label"] = np.asarray(bundle["memorandum_label"])
            bundles.append(bundle)
    return bundles, paths


def _watters_bundles(watters_arrays_by_session: dict) -> list[dict]:
    bundles = []
    for session_id, entry in watters_arrays_by_session.items():
        bundle = _watters_bundle_with_label(entry, CONTENT_LABEL_K_CLASSES)
        # cued_theta is not part of _watters_bundle_with_label's own return (it only needs the
        # discretised label from it); the axis-alignment analysis's continuous memorandum regression subspace needs the raw
        # angle, restricted to the identical `usable` trial mask that function already applied.
        bundle["cued_theta"] = np.asarray(entry["session"]["cued_theta"], dtype=float)[entry["usable"]]
        bundles.append(bundle)
    return bundles


def _worse_behaviour(bundle: dict, sign: float) -> np.ndarray:
    """A literal worse-coded outcome array (higher = worse), for the signed-displacement test's median-split sign rule --
    distinct from the correlation-coefficient sign flip SIGN_TO_WORSE_BEHAVIOUR applies elsewhere."""
    return bundle["outcome_raw"] if sign == 1.0 else (1.0 - bundle["outcome_raw"])


# =======================================================================================================
# RESIDUAL-DIRECTION ANISOTROPY TEST
# =======================================================================================================

def _anisotropy_macaque_session(bundle: dict) -> dict:
    rows = _residual_rows(bundle["activity_by_unit"])
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_after_residual_floor", **{
            k: rows[k] for k in ("n_trials_with_defined_direction", "n_trials_excluded_by_residual_floor", "n_kept")}}
    R, idx = _unit_residual_matrix(rows)
    stat = participation_ratio_and_leading_fraction(R)
    if stat is None:
        return {"status": "not_computable"}
    tag = f"deviation_axis_structure|anisotropy|panichello|{bundle['session']}"
    pr_draws, lead_draws = rotation_null_draws(rows["loo_mean"][idx], N_ROTATION_DRAWS, tag)
    return {
        "status": "computed", **stat,
        "n_trials_with_defined_direction": rows["n_trials_with_defined_direction"],
        "n_trials_excluded_by_residual_floor": rows["n_trials_excluded_by_residual_floor"],
        "pr_null_draws": pr_draws, "lead_null_draws": lead_draws,
    }


def _anisotropy_watters_session(bundle: dict) -> dict:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level, pr_entries, lead_entries = {}, [], []
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level)] = {"status": "too_few_trials_at_this_item_count_level", "n_trials": n_level}
            continue
        rows = _residual_rows(bundle["activity_by_unit"][mask])
        if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level)] = {"status": "too_few_trials_after_residual_floor", "n_trials": n_level,
                                      "n_kept": rows["n_kept"]}
            continue
        R, idx = _unit_residual_matrix(rows)
        stat = participation_ratio_and_leading_fraction(R)
        if stat is None:
            per_level[str(level)] = {"status": "not_computable", "n_trials": n_level}
            continue
        tag = f"deviation_axis_structure|anisotropy|watters|{bundle['session']}|level{level}"
        pr_draws, lead_draws = rotation_null_draws(rows["loo_mean"][idx], N_ROTATION_DRAWS, tag)
        per_level[str(level)] = {
            "status": "computed", **stat,
            "n_trials": n_level, "n_trials_excluded_by_residual_floor": rows["n_trials_excluded_by_residual_floor"],
        }
        pr_entries.append((stat["n_trials"], stat["participation_ratio"], pr_draws))
        lead_entries.append((stat["n_trials"], stat["leading_eigenvalue_fraction"], lead_draws))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level, "n_levels_tested": 0}
    pr_observed = _trial_count_weighted([(n, v) for n, v, _ in pr_entries])
    lead_observed = _trial_count_weighted([(n, v) for n, v, _ in lead_entries])
    pr_null = _weighted_combine_draws([(n, d) for n, _, d in pr_entries])
    lead_null = _weighted_combine_draws([(n, d) for n, _, d in lead_entries])
    return {
        "status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level,
        "participation_ratio": pr_observed, "leading_eigenvalue_fraction": lead_observed,
        "pr_null_draws": pr_null, "lead_null_draws": lead_null,
        "n_units": bundle["activity_by_unit"].shape[1],
    }


def run_anisotropy_test(bundles: list[dict], corpus_key: str) -> dict:
    per_session = []
    fitter = _anisotropy_macaque_session if corpus_key == CORPORA[0] else _anisotropy_watters_session
    for bundle in bundles:
        unit = f"anisotropy|{corpus_key}|{bundle['session']}"
        cached = _load_checkpoint(unit)
        if cached is not None:
            record = cached
        else:
            raw = fitter(bundle)
            record = _json_safe(raw)
            _save_checkpoint(unit, record)
        per_session.append({"session": bundle["session"], **record})

    pr_records = [{"observed": s.get("participation_ratio"), "null_draws": s.get("pr_null_draws")}
                  for s in per_session if s.get("status") == "computed"]
    lead_records = [{"observed": s.get("leading_eigenvalue_fraction"), "null_draws": s.get("lead_null_draws")}
                     for s in per_session if s.get("status") == "computed"]
    pooled_pr = _pool_rotation_statistic(pr_records)
    pooled_lead = _pool_rotation_statistic(lead_records)

    n_units_values = [s.get("n_units") for s in per_session if s.get("status") == "computed" and s.get("n_units")]
    dim_reference = (float(np.median(n_units_values)) - 1.0) if n_units_values else None
    branch = classify_residual_direction_structure(pooled_pr, pooled_lead, dim_reference) if dim_reference else "not_computable"

    return {
        "decision_rule_declared_before_fitting": ANISOTROPY_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions_total": len(bundles), "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "per_session": [{k: v for k, v in s.items() if k not in ("pr_null_draws", "lead_null_draws")} for s in per_session],
        "pooled_participation_ratio": {k: v for k, v in pooled_pr.items()},
        "pooled_leading_eigenvalue_fraction": {k: v for k, v in pooled_lead.items()},
        "dim_reference_for_single_axis_comparison": dim_reference,
        "single_axis_reference_participation_ratio": (
            _single_axis_reference_participation_ratio(pooled_lead["real_pooled"]["mean_value"], dim_reference)
            if dim_reference and pooled_lead.get("real_pooled", {}).get("status") == "tested" else None
        ),
        "branch": branch,
    }


# =======================================================================================================
# AXIS ALIGNMENT AGAINST REFERENCE DIRECTIONS
# =======================================================================================================

def _regression_direction(U: np.ndarray, covariate: np.ndarray) -> np.ndarray | None:
    n = U.shape[0]
    design = np.column_stack([np.ones(n), covariate])
    coef, *_ = np.linalg.lstsq(design, U, rcond=None)
    beta = coef[1]
    norm = np.linalg.norm(beta)
    return (beta / norm) if norm > 0 else None


def _class_mean_subspace_basis(U: np.ndarray, labels: np.ndarray, dim: int) -> np.ndarray | None:
    classes = np.unique(labels)
    if len(classes) < dim + 1:
        return None
    centred = U - U.mean(axis=0)
    class_means = np.stack([centred[labels == c].mean(axis=0) for c in classes])
    return _orthonormal_basis(class_means, dim)


def _regression_subspace_basis(U: np.ndarray, target_2d: np.ndarray, dim: int) -> np.ndarray | None:
    x_c = target_2d - target_2d.mean(axis=0)
    u_c = U - U.mean(axis=0)
    coef, *_ = np.linalg.lstsq(x_c, u_c, rcond=None)
    return _orthonormal_basis(coef, dim)


def _class_mean_dict(U: np.ndarray, labels: np.ndarray) -> dict | None:
    finite_label = np.isfinite(labels)
    if not finite_label.any():
        return None
    classes = np.unique(labels[finite_label])
    class_members = {c: np.flatnonzero((labels == c) & finite_label) for c in classes}
    class_mean = {c: U[idx].mean(axis=0) for c, idx in class_members.items() if len(idx) >= SHARP_TEST_MIN_PER_CLASS}
    return class_mean if len(class_mean) >= SHARP_TEST_MIN_CLASSES else None


def _alignment_null_vector(a: np.ndarray, target: np.ndarray, n_draws: int, seed_tag: str, n_units: int) -> dict:
    observed = abs(float(np.dot(a, target)))
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.abs(g @ target)
    return _empirical_two_sided(observed, draws)


def _alignment_null_subspace(a: np.ndarray, basis: np.ndarray, n_draws: int, seed_tag: str, n_units: int) -> dict:
    observed = min(float(np.linalg.norm(basis.T @ a)), 1.0)
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.minimum(np.linalg.norm(g @ basis, axis=1), 1.0)
    result = _empirical_two_sided(observed, draws)
    result["principal_angle_deg"] = float(np.degrees(np.arccos(min(observed, 1.0))))
    return result


def _empirical_two_sided(observed: float, draws: np.ndarray) -> dict:
    finite = draws[np.isfinite(draws)]
    if finite.size < 10:
        return {"status": "not_computable"}
    null_mean = float(np.mean(finite))
    p = float(permutation_pvalue(np.abs(finite - null_mean) >= abs(observed - null_mean)))
    return {
        "status": "computed", "observed": observed, "null_mean": null_mean, "null_sd": float(np.std(finite)),
        "n_draws": int(finite.size), "two_sided_p_value": p, "significant": bool(p <= 0.05 and observed > null_mean),
    }


def _preceding_class_mean_alignment(a: np.ndarray, U: np.ndarray, labels: np.ndarray, class_mean: dict,
                                     n_draws: int, seed_tag: str, n_units: int) -> dict:
    n = U.shape[0]
    cosines, refs = [], []
    for t in range(1, n):
        if not (np.isfinite(labels[t]) and np.isfinite(labels[t - 1])):
            continue
        own, prev = labels[t], labels[t - 1]
        if own == prev or prev not in class_mean:
            continue
        m = class_mean[prev]
        mn = np.linalg.norm(m)
        if mn <= 0.0:
            continue
        ref = m / mn
        refs.append(ref)
        cosines.append(abs(float(np.dot(a, ref))))
    if len(cosines) < SHARP_TEST_MIN_PER_CLASS:
        return {"status": "too_few_qualifying_trials", "n_qualifying_trials": len(cosines)}
    observed = float(np.mean(cosines))
    refs = np.stack(refs)
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.mean(np.abs(refs @ g.T), axis=0)
    result = _empirical_two_sided(observed, draws)
    result["n_qualifying_trials"] = len(cosines)
    return result


def _axis_stability(R: np.ndarray, seed_tag: str) -> dict:
    n, n_units = R.shape
    if n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "too_few_trials", "n_trials": n}
    folds = _contiguous_folds(n, N_CV_FOLDS)
    fold_axes = [leading_eigenvector(R[folds == f]) if int((folds == f).sum()) >= MIN_FOLD_TRIALS else None
                 for f in range(N_CV_FOLDS)]
    pairs = [(i, j) for i in range(N_CV_FOLDS) for j in range(i + 1, N_CV_FOLDS)
             if fold_axes[i] is not None and fold_axes[j] is not None]
    if len(pairs) < 3:
        return {"status": "too_few_fold_pairs", "n_fold_pairs": len(pairs)}
    observed = float(np.mean([abs(float(np.dot(fold_axes[i], fold_axes[j]))) for i, j in pairs]))
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.empty(N_ROTATION_DRAWS)
    for d in range(N_ROTATION_DRAWS):
        vecs = rng.standard_normal((2 * len(pairs), n_units))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        draws[d] = float(np.mean(np.abs(np.sum(vecs[0::2] * vecs[1::2], axis=1))))
    null_mean = float(np.mean(draws))
    p = float(permutation_pvalue(draws >= observed))
    return {
        "status": "computed", "observed_mean_abs_cosine": observed, "n_fold_pairs": len(pairs),
        "null_mean": null_mean, "null_sd": float(np.std(draws)),
        "one_sided_p_value_stability_above_chance": p, "stable": bool(p <= 0.05),
    }


def _axis_alignment_macaque_session(bundle: dict) -> dict | None:
    rows = _residual_rows(bundle["activity_by_unit"])
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, idx = _unit_residual_matrix(rows)
    U = unit_direction_vectors(bundle["activity_by_unit"])[idx]
    n_units = R.shape[1]
    a = leading_eigenvector(R)
    spike_count = bundle["spike_count"][idx]
    trial_index = bundle["trial_index"][idx]
    labels = bundle["memorandum_label"][idx]
    tag = f"deviation_axis_structure|axis_alignment|panichello|{bundle['session']}"
    alignments = {}
    ref = _regression_direction(U, spike_count)
    alignments["total_spike_count_direction"] = (
        _alignment_null_vector(a, ref, N_ROTATION_DRAWS, f"{tag}|spike", n_units) if ref is not None
        else {"status": "not_computable"})
    dim = len(np.unique(labels)) - 1
    basis = _class_mean_subspace_basis(U, labels, dim) if dim >= 1 else None
    alignments["memorandum_coding_subspace"] = (
        _alignment_null_subspace(a, basis, N_ROTATION_DRAWS, f"{tag}|memorandum", n_units) if basis is not None
        else {"status": "not_computable"})
    ref = _regression_direction(U, trial_index)
    alignments["slow_drift_direction"] = (
        _alignment_null_vector(a, ref, N_ROTATION_DRAWS, f"{tag}|drift", n_units) if ref is not None
        else {"status": "not_computable"})
    class_mean = _class_mean_dict(U, labels)
    alignments["preceding_trial_class_mean_direction"] = (
        _preceding_class_mean_alignment(a, U, labels, class_mean, N_ROTATION_DRAWS, f"{tag}|preceding", n_units)
        if class_mean is not None else {"status": "no_usable_memorandum_label"})
    stability = _axis_stability(R, f"{tag}|stability")
    return {"status": "computed", "alignments": alignments, "axis_stability": stability, "n_trials_kept": R.shape[0]}


def _axis_alignment_watters_session(bundle: dict) -> dict | None:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level = {}
    entries_by_ref: dict[str, list[tuple[int, dict]]] = {
        "total_spike_count_direction": [], "memorandum_coding_subspace": [],
        "slow_drift_direction": [], "preceding_trial_class_mean_direction": [],
    }
    stability_entries = []
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        rows = _residual_rows(bundle["activity_by_unit"][mask])
        if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level)] = {"status": "too_few_trials_after_residual_floor", "n_trials": n_level}
            continue
        R, idx = _unit_residual_matrix(rows)
        U = unit_direction_vectors(bundle["activity_by_unit"][mask])[idx]
        n_units = R.shape[1]
        a = leading_eigenvector(R)
        spike_count = bundle["spike_count"][mask][idx]
        trial_index = bundle["trial_index"][mask][idx]
        labels = bundle["memorandum_label"][mask][idx]
        cued_theta = np.mod(np.asarray(bundle["cued_theta"], dtype=float)[mask][idx], 2.0 * np.pi)
        target_2d = np.stack([np.cos(cued_theta), np.sin(cued_theta)], axis=1)
        tag = f"deviation_axis_structure|axis_alignment|watters|{bundle['session']}|level{level}"

        alignments = {}
        ref = _regression_direction(U, spike_count)
        alignments["total_spike_count_direction"] = (
            _alignment_null_vector(a, ref, N_ROTATION_DRAWS, f"{tag}|spike", n_units) if ref is not None
            else {"status": "not_computable"})
        basis = _regression_subspace_basis(U, target_2d, WATTERS_REGRESSION_DIM)
        alignments["memorandum_coding_subspace"] = (
            _alignment_null_subspace(a, basis, N_ROTATION_DRAWS, f"{tag}|memorandum", n_units) if basis is not None
            else {"status": "not_computable"})
        ref = _regression_direction(U, trial_index)
        alignments["slow_drift_direction"] = (
            _alignment_null_vector(a, ref, N_ROTATION_DRAWS, f"{tag}|drift", n_units) if ref is not None
            else {"status": "not_computable"})
        class_mean = _class_mean_dict(U, labels)
        alignments["preceding_trial_class_mean_direction"] = (
            _preceding_class_mean_alignment(a, U, labels, class_mean, N_ROTATION_DRAWS, f"{tag}|preceding", n_units)
            if class_mean is not None else {"status": "no_usable_memorandum_label"})
        stability = _axis_stability(R, f"{tag}|stability")

        per_level[str(level)] = {"status": "computed", "alignments": alignments, "axis_stability": stability,
                                  "n_trials": n_level, "n_trials_kept": R.shape[0]}
        for ref_name in entries_by_ref:
            if alignments[ref_name].get("status") == "computed":
                entries_by_ref[ref_name].append((n_level, alignments[ref_name]))
        if stability.get("status") == "computed":
            stability_entries.append((n_level, stability))

    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}

    combined_alignments = {}
    for ref_name, entries in entries_by_ref.items():
        if not entries:
            combined_alignments[ref_name] = {"status": "not_computable"}
            continue
        observed = _trial_count_weighted([(n, e["observed"]) for n, e in entries])
        null_mean = _trial_count_weighted([(n, e["null_mean"]) for n, e in entries])
        combined_alignments[ref_name] = {
            "status": "computed", "observed": observed, "null_mean": null_mean,
            "significant": bool(observed is not None and null_mean is not None and observed > null_mean
                                 and any(e.get("significant") for _, e in entries)),
        }
    combined_stability = None
    if stability_entries:
        observed = _trial_count_weighted([(n, e["observed_mean_abs_cosine"]) for n, e in stability_entries])
        null_mean = _trial_count_weighted([(n, e["null_mean"]) for n, e in stability_entries])
        combined_stability = {
            "status": "computed", "observed_mean_abs_cosine": observed, "null_mean": null_mean,
            "stable": bool(observed is not None and null_mean is not None and observed > null_mean
                           and any(e.get("stable") for _, e in stability_entries)),
        }

    return {
        "status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level,
        "alignments": combined_alignments,
        "axis_stability": combined_stability or {"status": "not_computable"},
    }


def run_axis_alignment(bundles: list[dict], corpus_key: str) -> dict:
    per_session = []
    fitter = _axis_alignment_macaque_session if corpus_key == CORPORA[0] else _axis_alignment_watters_session
    for bundle in bundles:
        unit = f"axis_alignment|{corpus_key}|{bundle['session']}"
        cached = _load_checkpoint(unit)
        if cached is not None:
            record = cached
        else:
            raw = fitter(bundle)
            record = _json_safe(raw) if raw is not None else {"status": "not_computable"}
            _save_checkpoint(unit, record)
        per_session.append({"session": bundle["session"], **record})

    computed = [s for s in per_session if s.get("status") == "computed"]
    stable_flags = [s["axis_stability"].get("stable") for s in computed if s.get("axis_stability", {}).get("status") == "computed"]
    n_stable = sum(1 for f in stable_flags if f)
    stability_summary = {
        "n_sessions_with_a_stability_verdict": len(stable_flags), "n_stable": n_stable,
        "fraction_stable": (n_stable / len(stable_flags)) if stable_flags else None,
        "stable_by_majority": bool(stable_flags) and n_stable > len(stable_flags) / 2.0,
    }
    return {
        "decision_rule_declared_before_fitting": AXIS_ALIGNMENT_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions_total": len(bundles), "n_sessions_computed": len(computed),
        "per_session": per_session, "axis_stability_summary": stability_summary,
    }


# =======================================================================================================
# SIGNED VS UNSIGNED DISPLACEMENT TEST
# =======================================================================================================

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 4 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def _fold_combined_and_pooled(feature: np.ndarray, outcome: np.ndarray, folds: np.ndarray, seed_tag: str) -> dict:
    valid = np.isfinite(feature) & np.isfinite(outcome)
    if int(valid.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials"}
    pooled = partial_correlation_permutation_test(
        outcome[valid], feature[valid], [], N_PERM, np.random.default_rng(stable_seed(f"{seed_tag}|pooled")))
    per_fold = []
    for f in range(N_CV_FOLDS):
        test = (folds == f) & valid
        if int(test.sum()) < 4:
            continue
        r = _pearson_r(feature[test], outcome[test])
        if r is not None:
            per_fold.append((int(test.sum()), r))
    within_fold = _trial_count_weighted(per_fold) if per_fold else None
    return {
        "status": "computed", "pooled_across_fold": pooled,
        "within_fold_trial_count_weighted_r": within_fold, "n_folds_contributing": len(per_fold),
    }


def _signed_displacement_core(R: np.ndarray, idx: np.ndarray, deviation: np.ndarray, worse: np.ndarray, seed_tag: str) -> dict | None:
    n = R.shape[0]
    if n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return None
    folds = _contiguous_folds(n, N_CV_FOLDS)
    outcome = worse[idx]
    dev = deviation[idx]

    s_signed = np.full(n, np.nan)
    bias_only = np.full(n, np.nan)
    for f in range(N_CV_FOLDS):
        train, test = folds != f, folds == f
        if int(train.sum()) < MIN_FOLD_TRIALS or not test.any():
            continue
        a_train = leading_eigenvector(R[train])
        median_out = float(np.median(outcome[train]))
        worse_mask = outcome[train] >= median_out
        proj_train = R[train] @ a_train
        mean_signed_worse = float(np.mean(proj_train[worse_mask])) if worse_mask.any() else 0.0
        if mean_signed_worse < 0.0:
            a_train = -a_train
            proj_train = -proj_train
        s_signed[test] = R[test] @ a_train
        bias_only[test] = float(np.mean(proj_train))
    unsigned = np.abs(s_signed)

    return {
        "status": "computed", "n_trials": n,
        "signed_projection": _fold_combined_and_pooled(s_signed, outcome, folds, f"{seed_tag}|signed"),
        "unsigned_projection": _fold_combined_and_pooled(unsigned, outcome, folds, f"{seed_tag}|unsigned"),
        "delivered_deviation": _fold_combined_and_pooled(dev, outcome, folds, f"{seed_tag}|delivered"),
        "bias_only_signed_projection": _fold_combined_and_pooled(bias_only, outcome, folds, f"{seed_tag}|bias"),
    }


def _normalize_signed_displacement_result(result: dict) -> dict:
    combined = {}
    for key in ("signed_projection", "unsigned_projection", "delivered_deviation", "bias_only_signed_projection"):
        r = result.get(key, {}) if result else {}
        if r.get("status") != "computed":
            combined[key] = {"status": "not_computable"}
            continue
        pooled_r = r["pooled_across_fold"]["r"] if r["pooled_across_fold"].get("status") == "computed" else None
        combined[key] = {
            "status": "computed" if pooled_r is not None else "not_computable",
            "pooled_across_fold_r": pooled_r, "within_fold_r": r.get("within_fold_trial_count_weighted_r"),
        }
    return combined


def _signed_displacement_macaque_session(bundle: dict, sign: float) -> dict:
    rows = _residual_rows(bundle["activity_by_unit"])
    if rows["n_kept"] < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "too_few_trials", "n_kept": rows["n_kept"]}
    R, idx = _unit_residual_matrix(rows)
    worse = _worse_behaviour(bundle, sign)
    tag = f"deviation_axis_structure|signed_displacement|panichello|{bundle['session']}"
    raw = _signed_displacement_core(R, idx, bundle["deviation"], worse, tag)
    if raw is None:
        return {"status": "too_few_trials"}
    return {"status": "computed", "combined": _normalize_signed_displacement_result(raw), "raw": raw}


def _signed_displacement_watters_session(bundle: dict, sign: float) -> dict:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    worse = _worse_behaviour(bundle, sign)
    per_level = {}
    entries = {k: [] for k in ("signed_projection", "unsigned_projection", "delivered_deviation", "bias_only_signed_projection")}
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < N_CV_FOLDS * MIN_FOLD_TRIALS:
            per_level[str(level)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        rows = _residual_rows(bundle["activity_by_unit"][mask])
        if rows["n_kept"] < N_CV_FOLDS * MIN_FOLD_TRIALS:
            per_level[str(level)] = {"status": "too_few_trials_after_residual_floor", "n_trials": n_level}
            continue
        R, idx = _unit_residual_matrix(rows)
        tag = f"deviation_axis_structure|signed_displacement|watters|{bundle['session']}|level{level}"
        raw = _signed_displacement_core(R, idx, bundle["deviation"][mask], worse[mask], tag)
        if raw is None:
            per_level[str(level)] = {"status": "not_computable", "n_trials": n_level}
            continue
        combined = _normalize_signed_displacement_result(raw)
        per_level[str(level)] = {"status": "computed", "n_trials": n_level, "combined": combined}
        for key in entries:
            if combined[key].get("status") == "computed":
                entries[key].append((n_level, combined[key]))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}
    combined = {}
    for key, es in entries.items():
        if not es:
            combined[key] = {"status": "not_computable"}
            continue
        pooled_r = _trial_count_weighted([(n, e["pooled_across_fold_r"]) for n, e in es if e.get("pooled_across_fold_r") is not None])
        within_r = _trial_count_weighted([(n, e["within_fold_r"]) for n, e in es if e.get("within_fold_r") is not None])
        combined[key] = {"status": "computed", "pooled_across_fold_r": pooled_r, "within_fold_r": within_r}
    return {"status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level, "combined": combined}


def _pool_signed_displacement_key(per_session: list[dict], key: str, estimator: str) -> dict:
    values = [s["combined"][key][estimator] for s in per_session
              if s.get("combined", {}).get(key, {}).get(estimator) is not None]
    pooled = slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}
    if len(values) >= 2:
        pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    return pooled


def run_signed_displacement_test(bundles: list[dict], corpus_key: str, sign: float, delivered_reference_effect: float) -> dict:
    per_session = []
    for bundle in bundles:
        unit = f"signed_displacement|{corpus_key}|{bundle['session']}"
        cached = _load_checkpoint(unit)
        if cached is not None:
            record = cached
        else:
            raw = (_signed_displacement_macaque_session(bundle, sign) if corpus_key == CORPORA[0]
                   else _signed_displacement_watters_session(bundle, sign))
            record = _json_safe(raw)
            _save_checkpoint(unit, record)
        per_session.append({"session": bundle["session"], **record})

    computed = [s for s in per_session if s.get("status") == "computed"]
    estimators = ("pooled_across_fold_r", "within_fold_r")
    keys = ("signed_projection", "unsigned_projection", "delivered_deviation", "bias_only_signed_projection")
    pooled = {key: {est: _pool_signed_displacement_key(computed, key, est) for est in estimators} for key in keys}

    bias_reproduces = {
        est: _bias_only_reproduces(pooled["signed_projection"][est], pooled["bias_only_signed_projection"][est])
        for est in estimators
    }
    bias_voids = any(bias_reproduces.values())

    pair_defs = [("signed_projection", "unsigned_projection", "signed_vs_unsigned"),
                 ("signed_projection", "delivered_deviation", "signed_vs_delivered"),
                 ("unsigned_projection", "delivered_deviation", "unsigned_vs_delivered")]
    paired: dict[str, dict] = {est: {} for est in estimators}
    all_p, p_index = [], []
    for est in estimators:
        for a_key, b_key, name in pair_defs:
            diffs = []
            for s in computed:
                va = s.get("combined", {}).get(a_key, {}).get(est)
                vb = s.get("combined", {}).get(b_key, {}).get(est)
                if va is not None and vb is not None:
                    diffs.append(abs(va) - abs(vb))
            test = slope_across_sessions_test(diffs, alternative="two-sided") if diffs else {"status": "not_computed"}
            if len(diffs) >= 2:
                test["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(diffs)
            paired[est][name] = test
            if test.get("status") == "tested":
                all_p.append(test["two_sided_p_value"])
                p_index.append((est, name))
    if all_p:
        fdr = fdr_bh(np.asarray(all_p, dtype=float))
        for (est, name), q, rej in zip(p_index, fdr["q_values"], fdr["reject"]):
            paired[est][name]["q_value"] = float(q)
            paired[est][name]["q_significant"] = bool(rej)

    def _signed_wins(est: str) -> bool:
        a, b = paired[est]["signed_vs_unsigned"], paired[est]["signed_vs_delivered"]
        return bool(a.get("q_significant") and a.get("mean_value", 0.0) > 0.0
                    and b.get("q_significant") and b.get("mean_value", 0.0) > 0.0)

    if bias_voids:
        branch = "axis_projection_not_separable_from_a_session_level_offset"
    else:
        wins = {est: _signed_wins(est) for est in estimators}
        if all(wins.values()):
            branch = "the_component_is_a_signed_displacement_along_an_axis"
        elif wins["pooled_across_fold_r"] != wins["within_fold_r"]:
            branch = "ordering_holds_under_one_estimator_only"
        else:
            sd_delivered = paired["pooled_across_fold_r"]["signed_vs_delivered"]
            mdd_block = sd_delivered.get("minimum_detectable_paired_difference_at_80pct_power", {})
            mdd_value = mdd_block.get("mdd") if isinstance(mdd_block, dict) and mdd_block.get("status") == "computed" else None
            if (not sd_delivered.get("significant")) and mdd_value is not None and mdd_value < delivered_reference_effect:
                branch = "the_component_is_a_magnitude_not_a_signed_displacement"
            else:
                branch = "signed_and_unsigned_are_not_distinguishable_at_this_power"

    return {
        "decision_rule_declared_before_fitting": SIGNED_DISPLACEMENT_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions_total": len(bundles), "n_sessions_computed": len(computed),
        "per_session": per_session, "pooled": pooled,
        "bias_only_reproduces_by_estimator": bias_reproduces,
        "paired_tests": paired, "branch": branch,
        "delivered_deviation_reference_effect_size_r_units": delivered_reference_effect,
    }


# =======================================================================================================
# OCCUPIED-STATE-SPACE DECOMPOSITION -- is the axis outside the occupied state space, or only outside the
# coding subspace?
# =======================================================================================================

def _cv_pca_rank(U: np.ndarray, max_k: int, seed_tag: str) -> dict:
    n = U.shape[0]
    if max_k < 1 or n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "not_estimable"}
    folds = _contiguous_folds(n, N_CV_FOLDS)
    candidate_ks = list(range(1, max_k + 1))
    errors = np.zeros(len(candidate_ks))
    n_folds_used = 0
    for f in range(N_CV_FOLDS):
        train, test = folds != f, folds == f
        if int(train.sum()) < 4 or not test.any():
            continue
        mean_train = U[train].mean(axis=0)
        _u_svd, _s_svd, vt_svd = np.linalg.svd(U[train] - mean_train, full_matrices=False)
        test_c = U[test] - mean_train
        n_folds_used += 1
        for i, k in enumerate(candidate_ks):
            k_eff = min(k, vt_svd.shape[0])
            basis = vt_svd[:k_eff].T
            proj = test_c @ basis @ basis.T
            errors[i] += float(np.sum((test_c - proj) ** 2))
    if n_folds_used == 0:
        return {"status": "not_estimable"}
    best_idx = int(np.argmin(errors))
    return {"status": "computed", "n_folds_used": n_folds_used, "candidate_ks": candidate_ks,
            "cv_reconstruction_error": errors.tolist(), "best_k": candidate_ks[best_idx]}


def _occupied_space_decomposition(U: np.ndarray, axis: np.ndarray, k: int, n_draws: int, seed_tag: str) -> dict:
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
    g = rng.standard_normal((n_draws, p))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    w = g @ basis
    null_off = 1.0 - np.sum(w ** 2, axis=1)
    null_mean = float(np.mean(null_off))
    p_value = float(permutation_pvalue(np.abs(null_off - null_mean) >= abs(off_frac - null_mean)))
    return {
        "status": "computed", "k": k_eff, "within_fraction": within_frac, "off_fraction": off_frac,
        "null_off_fraction_mean": null_mean, "null_off_fraction_sd": float(np.std(null_off)),
        "two_sided_p_value": p_value, "off_fraction_above_null": bool(p_value <= 0.05 and off_frac > null_mean),
        # kept for pooling across sessions; stripped from the stored per-level furniture
        "null_off_fraction_draws": null_off,
    }


def _collect_axis_entries(bundles: list[dict], corpus_key: str) -> dict:
    """Rebuilds the identical (R, U) pair the anisotropy test used, per session (per item-count level for
    the multi-object corpus), for the occupied-state-space decomposition -- recomputed rather than threaded
    through the checkpoint, since these arrays are not JSON-safe and the anisotropy test's own checkpoint
    intentionally keeps only its scalar summaries."""
    out: dict[str, list[dict]] = {}
    for bundle in bundles:
        session = bundle["session"]
        entries = []
        if corpus_key == CORPORA[0]:
            rows = _residual_rows(bundle["activity_by_unit"])
            if rows["n_kept"] >= MIN_TRIALS_WITH_DEFINED_DIRECTION:
                R, idx = _unit_residual_matrix(rows)
                U = unit_direction_vectors(bundle["activity_by_unit"])[idx]
                entries.append({"n_trials": int(R.shape[0]), "R": R, "U": U, "level": "all"})
        else:
            item_count = bundle["item_count"]
            for level in sorted({int(v) for v in item_count.tolist()}):
                mask = item_count == float(level)
                if int(mask.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
                    continue
                rows = _residual_rows(bundle["activity_by_unit"][mask])
                if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
                    continue
                R, idx = _unit_residual_matrix(rows)
                U = unit_direction_vectors(bundle["activity_by_unit"][mask])[idx]
                entries.append({"n_trials": int(R.shape[0]), "R": R, "U": U, "level": str(level)})
        out[session] = entries
    return out


def pooled_off_fraction_against_matched_null(computed_sessions: list[dict]) -> dict:
    """Pools each session's own matched-random-axis null draws -- per session first combined across
    item-count levels by trial-count weighting, draw-index aligned; across sessions by their mean,
    draw-index aligned (the same pooling convention the anisotropy block applies to its rotation nulls)
    -- and compares the pooled observed off-fraction (mean across sessions) against that pooled null's
    central mass by a two-sided empirical percentile test. The observed off-fraction is never tested
    against zero: a random axis of the same dimension has a nonzero off-occupied fraction by
    construction, so only the matched null separates 'leaves the occupied space' from 'does not'."""
    off_fracs = [s["off_fraction"] for s in computed_sessions if s.get("off_fraction") is not None]
    draws_list = [np.asarray(s["null_draws"], dtype=float) for s in computed_sessions
                  if s.get("null_draws") is not None]
    out = {"status": "not_computable", "n_sessions": len(off_fracs),
           "n_sessions_with_a_null_distribution": len(draws_list),
           "pooled_observed_off_fraction": None, "pooled_null_mean": None, "pooled_null_sd": None,
           "two_sided_empirical_p_value": None, "above_null": None,
           "significant_above_null": False, "significant_below_null": False}
    if not draws_list or not off_fracs or len(off_fracs) != len(draws_list):
        return out
    pooled_null = np.nanmean(np.stack(draws_list), axis=0)
    finite = pooled_null[np.isfinite(pooled_null)]
    if finite.size < 10:
        return out
    null_centre = float(np.mean(finite))
    observed_pooled = float(np.mean(off_fracs))
    p_value = float(permutation_pvalue(np.abs(finite - null_centre) >= abs(observed_pooled - null_centre)))
    above = observed_pooled > null_centre
    out.update({
        "status": "computed", "pooled_observed_off_fraction": observed_pooled,
        "pooled_null_mean": null_centre, "pooled_null_sd": float(np.std(finite)),
        "two_sided_empirical_p_value": p_value, "above_null": bool(above),
        "significant_above_null": bool(p_value <= 0.05 and above),
        "significant_below_null": bool(p_value <= 0.05 and not above),
    })
    return out


def classify_occupied_space_branch(comparison: dict) -> str:
    """Pre-declared mapping from the pooled observed-versus-matched-null comparison to one of the three
    named branches; see OCCUPIED_SPACE_DECISION_RULE_DECLARED_BEFORE_FITTING."""
    if comparison.get("status") != "computed":
        return "not_separable_at_the_available_dimensionality"
    if comparison["significant_above_null"]:
        return "the_axis_lies_outside_the_occupied_state_space"
    return "the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace"


def run_occupied_space_block(bundles: list[dict], corpus_key: str) -> dict:
    axis_entries = _collect_axis_entries(bundles, corpus_key)
    per_session = []
    for bundle in bundles:
        session = bundle["session"]
        entries = axis_entries.get(session, [])
        if not entries:
            per_session.append({"session": session, "status": "not_computable"})
            continue
        level_results = []
        for entry in entries:
            R, U, n_trials, level = entry["R"], entry["U"], entry["n_trials"], entry["level"]
            n, p = U.shape
            axis = leading_eigenvector(R)
            max_k = min(p, n - 1)
            tag_base = f"deviation_axis_structure|occupied_space|{corpus_key}|{session}|{level}"
            cv = _cv_pca_rank(U, max_k, f"{tag_base}|cv")
            if cv.get("status") != "computed":
                level_results.append({"status": "not_separable_at_the_available_dimensionality", "n_trials": n_trials, "level": level})
                continue
            primary = _occupied_space_decomposition(U, axis, cv["best_k"], N_RANDOM_AXIS_DRAWS, f"{tag_base}|primary")
            full_rank = _occupied_space_decomposition(U, axis, max_k, N_RANDOM_AXIS_DRAWS, f"{tag_base}|full")
            level_results.append({"status": "computed", "n_trials": n_trials, "level": level, "cv_rank": cv,
                                   "primary_at_cv_rank": primary, "full_rank_sensitivity": full_rank})
        computed_levels = [r for r in level_results if r.get("status") == "computed"
                            and r["primary_at_cv_rank"].get("status") == "computed"]
        if not computed_levels:
            per_session.append({"session": session, "status": "not_computable", "levels": [
                {k: ({kk: vv for kk, vv in v.items() if kk != "null_off_fraction_draws"}
                     if isinstance(v, dict) else v)
                 for k, v in r.items()} for r in level_results]})
            continue
        off_fraction = _trial_count_weighted([(r["n_trials"], r["primary_at_cv_rank"]["off_fraction"]) for r in computed_levels])
        null_mean = _trial_count_weighted([(r["n_trials"], r["primary_at_cv_rank"]["null_off_fraction_mean"]) for r in computed_levels])
        any_above_null = any(r["primary_at_cv_rank"]["off_fraction_above_null"] for r in computed_levels)
        session_null_draws = _weighted_combine_draws(
            [(r["n_trials"], r["primary_at_cv_rank"]["null_off_fraction_draws"])
             for r in computed_levels if r["primary_at_cv_rank"].get("null_off_fraction_draws") is not None])
        per_session.append({
            "session": session, "status": "computed",
            "levels": [{k: ({kk: vv for kk, vv in v.items() if kk != "null_off_fraction_draws"}
                            if isinstance(v, dict) else v)
                        for k, v in r.items()} for r in level_results],
            "off_fraction": off_fraction, "null_off_fraction_mean": null_mean,
            "off_fraction_above_null": bool(any_above_null),
        })
        # the draw arrays ride outside the stored record so the pooled matched-null test can use them
        # without inflating the artifact; stripped again below before writing.
        per_session[-1]["null_draws"] = session_null_draws

    computed = [s for s in per_session if s.get("status") == "computed"]
    off_fracs = [s["off_fraction"] for s in computed if s.get("off_fraction") is not None]
    above_null_flags = [s["off_fraction_above_null"] for s in computed]
    pooled = slope_across_sessions_test(off_fracs, alternative="two-sided") if off_fracs else {"status": "not_computed"}
    if len(off_fracs) >= 2:
        pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(off_fracs)

    comparison = pooled_off_fraction_against_matched_null(computed)
    branch = classify_occupied_space_branch(comparison)

    return {
        "decision_rule_declared_before_fitting": OCCUPIED_SPACE_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions_total": len(bundles), "n_sessions_computed": len(computed),
        "n_sessions_off_fraction_above_its_own_null": sum(1 for f in above_null_flags if f),
        "per_session": [{k: v for k, v in s.items() if k != "null_draws"} for s in per_session],
        "pooled_off_fraction": pooled,
        "pooled_observed_vs_matched_null": comparison, "branch": branch,
    }


# =======================================================================================================
# WITHIN-CORPUS STIMULATION-DIRECTION ALIGNMENT -- is the axis the same object as the effective
# stimulation direction recorded from the SAME macaque corpora?
# =======================================================================================================

def stimulation_direction_alignment_block() -> dict:
    """The delivered microstimulation corpus (config key macaque_pfc_microstimulation, results/
    stimulation_latent_response_map.json) is a THIRD macaque corpus, recorded from different animals in
    different sessions than either the single-item lateral prefrontal cortex corpus (Panichello et al.
    2024) or the multi-object corpus (Watters, Gabel, Tenenbaum and Jazayeri) this module's residual axis is
    fit on. A cosine between two directions requires both to live in the same coordinate basis -- the same
    recorded units, in the same session -- which does not exist across three corpora recorded from
    different neurons in different animals. This is checked directly below, not assumed, and reported as
    a disclosed gap rather than forced onto one of the three named branches, none of which anticipates a
    disjoint-corpus precondition failure."""
    root = data_root()
    macaque_sessions = sorted(p.stem for p in (_panichello_directory(root).glob("*.mat")
                                                 if _panichello_directory(root) else []))
    watters_sessions = sorted({f"{a}_{d}" for a, d, _v in _watters_session_dates_safe(root)})
    macaque_pfc_microstimulation_path = ROOT / "results" / "stimulation_latent_response_map.json"
    macaque_pfc_microstimulation_sessions: list[str] = []
    # Distinguish three reasons macaque_pfc_microstimulation_sessions can end up empty -- an upstream
    # arm voided by its own reproduction gate must never be silently coerced into
    # the same empty list a genuinely-computed-but-empty ledger would produce
    # (see this project's own named defect: a null read as a measured zero).
    if not macaque_pfc_microstimulation_path.exists():
        macaque_pfc_microstimulation_source_status, macaque_pfc_microstimulation_source_reason = "source_artifact_not_found", None
    else:
        macaque_pfc_microstimulation_json = json.loads(macaque_pfc_microstimulation_path.read_text())
        arm = macaque_pfc_microstimulation_json["arms"]["macaque_pfc_microstimulation"]  # schema-guaranteed key in every delivered version; raise if absent
        if "ledger" not in arm:
            # the void placeholder this arm's producing script writes when its own
            # reproduction gate fails: {"status": "void_reproduction_gate_did_not_reproduce", "reason": ...}
            macaque_pfc_microstimulation_source_status = "not_computed_upstream_reproduction_gate_void"
            macaque_pfc_microstimulation_source_reason = arm.get("reason")
        else:
            macaque_pfc_microstimulation_sessions = sorted(arm["ledger"]["seen"])
            macaque_pfc_microstimulation_source_status, macaque_pfc_microstimulation_source_reason = "computed", None
    overlap = (set(macaque_sessions) | set(watters_sessions)) & set(macaque_pfc_microstimulation_sessions)
    return {
        "checked": True,
        "panichello_2024_macaque_lPFC_single_item_sessions_n": len(macaque_sessions),
        "watters_2026_macaque_multi_object_sessions_n": len(watters_sessions),
        "macaque_pfc_microstimulation_stimulation_sessions_n": len(macaque_pfc_microstimulation_sessions),
        "macaque_pfc_microstimulation_sessions": macaque_pfc_microstimulation_sessions,
        "macaque_pfc_microstimulation_source_status": macaque_pfc_microstimulation_source_status,
        "macaque_pfc_microstimulation_source_void_reason": macaque_pfc_microstimulation_source_reason,
        "session_id_overlap_with_either_deviation_corpus": sorted(overlap),
        "n_overlapping_sessions": len(overlap),
        "additional_reason_the_raw_direction_vector_is_unavailable_even_where_sessions_overlapped": (
            "results/stimulation_latent_response_map.json serialises only displacement_direction_norm and a "
            "set of already-computed alignment scalars (to v_star, to the content subspace, to the dynamic "
            "subspace) for the macaque_pfc_microstimulation arm -- the raw displacement direction vector itself, in either "
            "raw-unit space or the session-specific PCA frame it was fit in, is never written to that "
            "artifact, only used internally by scripts/run_stimulation_latent_response_map.py's own "
            "macaque_pfc_microstimulation_arm() and discarded."
        ),
        "branch": (
            "not_computable_disjoint_corpus"
            if len(overlap) == 0 else
            "underpowered_to_ask"  # would require re-deriving the raw vector; reported honestly if this ever fires
        ),
        "gap_disclosure": (
            "This block's three named branches (same object / unrelated / underpowered_to_ask) all "
            "presuppose a residual axis and an effective stimulation direction computed in a shared unit "
            "basis. That precondition fails here: the residual axis is fit on the single-item macaque "
            f"lateral prefrontal cortex corpus ({len(macaque_sessions)} sessions on disk) and the multi-"
            f"object macaque corpus ({len(watters_sessions)} sessions), neither of which carries any "
            f"stimulation; the only delivered effective stimulation direction in this project's macaque data "
            f"lives in a disjoint corpus (macaque_pfc_microstimulation, {len(macaque_pfc_microstimulation_sessions)} sessions, 2 animals), "
            "sharing zero session identifiers with either. A cosine between the two "
            "directions is not merely underpowered, it is mathematically undefined: the two vectors would "
            "not share a coordinate basis (different neurons entirely). This is reported as a gap in the "
            "pre-declared branch list, with the numbers, rather than forced onto 'underpowered_to_ask', "
            "which implies a power problem this is not."
        ),
    }


def _watters_session_dates_safe(root) -> list:
    try:
        from corpus_sessions import watters_session_dates
        return list(watters_session_dates(root))
    except Exception:
        return []


# =======================================================================================================
# HUMAN FOOTPRINT -- is this construction computable on human recordings, and does the stimulation-evoked
# displacement seen in the human RAM corpus lie off the occupied state space?
# =======================================================================================================

def human_footprint_block() -> dict:
    """States whether this module's decomposition (participation ratio, rotation null, leading eigenvector)
    can be computed at all on human recordings, and precisely what is missing where it has not been run,
    rather than reporting a null. Two separate human facts are already on disk and are read live, not
    recomputed: (1) results/recording_tier_component_transfer.json already runs rate_free_state_deviation
    -- the identical scalar this module's residual construction is built from -- on human field-potential
    feature matrices at five recording tiers of the same verbal working-memory corpus (accession 000574
    plus its beamformed companion release ds004752), proving the (trials, features) construction this module
    needs is well-defined on human data; (2) results/human_stimulation_component_response.json measures a
    stimulation-evoked displacement of the same rate-free construction on a DIFFERENT human corpus (RAM
    intracranial free-recall stimulation, ds005489/ds005557), an encoding task with no isolated maintenance
    delay, which is the only human corpus in this project with both a stimulation displacement and channel-
    level control-trial activity from which an occupied-state-space could in principle be estimated."""
    tier_path = ROOT / "results" / "recording_tier_component_transfer.json"
    stim_path = ROOT / "results" / "human_stimulation_component_response.json"
    tier_summary, stim_summary = None, None
    if tier_path.exists():
        tier_json = json.loads(tier_path.read_text())
        tier_anisotropy_by_tier = tier_json.get("block_a", {})  # external artifact's own key, not renamed here
        tier_summary = {
            tier: {
                "branch": v.get("branch"), "n_sessions_computed": v.get("n_sessions_computed"),
                "n_trials_pooled": v.get("n_trials_pooled"), "median_per_trial_value": v.get("median_per_trial_value"),
            }
            for tier, v in tier_anisotropy_by_tier.items()
        }
    if stim_path.exists():
        stim_json = json.loads(stim_path.read_text())
        stim_displacement_result = stim_json.get("block_b", {})  # external artifact's own key, not renamed here
        stim_summary = {
            "branch": stim_displacement_result.get("branch"),
            "epoch_overlap_disclosure": stim_displacement_result.get("epoch_overlap_disclosure"),
            "n_sessions": len(stim_displacement_result.get("per_session", {})),
        }
    return {
        "part_1_is_the_construction_computable_on_human_recordings": {
            "answer": "yes, in principle, at every one of five recording tiers of the verbal working-"
                      "memory corpus (accession 000574 / ds004752)",
            "evidence": (
                "results/recording_tier_component_transfer.json's block_a already computes the identical "
                "rate_free_state_deviation scalar this module's residual construction is built from, on "
                "(trials, features) feature matrices at the microwire single-unit, medial-temporal depth, "
                "cortical depth, scalp EEG and beamformed-cortical-source tiers, on the same admitted "
                "trials at every tier -- the construction this module's residual rows and their eigen-"
                "decomposition need (a unit-normalised per-trial direction and a leave-one-out reference in "
                "the same feature space) is exactly what that estimator already forms internally."
            ),
            "per_tier_summary_read_live_from_the_delivered_artifact": tier_summary,
            "what_this_leg_did_not_run": (
                "The residual-axis-structure step this module adds beyond the scalar deviation -- stacking "
                "residual unit rows into R, its eigen-decomposition, the rotation null, and the occupied-"
                "space decomposition -- has NOT been executed on any human tier within this module's compute "
                "budget. What is missing to run it is not a new estimator or a data-availability gap: it is "
                "applying this module's own _residual_rows / _unit_residual_matrix / participation_ratio_"
                "and_leading_fraction / rotation_null_draws functions to the same (trials, features) arrays "
                "results/recording_tier_component_transfer.json already builds per tier, per session, "
                "restricted first to whichever tiers pass their own orthogonality gate against total power "
                "-- a gate this module's own scope restriction (macaque lPFC and the multi-object corpus being "
                "the only two passing corpora among the ones already tested) was never checked for any of "
                "these five human tiers specifically and would need to be run first."
            ),
        },
        "part_2_does_the_human_stimulation_displacement_lie_off_the_occupied_subspace": {
            "answer": "not computed within this module",
            "which_corpus_this_question_is_askable_in": (
                "results/human_stimulation_component_response.json (RAM intracranial free-recall "
                "stimulation, ds005489/ds005557) is the only human corpus in this project carrying BOTH a "
                "stimulation-evoked displacement of the rate-free construction and per-trial control "
                "channel activity in the same session, from which an occupied subspace could be fit. It is "
                "a free-recall ENCODING task with no isolated working-memory maintenance delay -- the "
                "delivered artifact's own epoch_overlap_disclosure states the stimulation interval overlaps "
                "the analysed epoch by design -- so any answer here would carry that caveat and would not "
                "be a maintenance-delay result."
            ),
            "stimulation_summary_read_live_from_the_delivered_artifact": stim_summary,
            "precisely_what_is_missing": (
                "The ingredients exist and are reusable unchanged: run_human_stimulation_component_response."
                "compute_block_b_displacement already returns, per session and per channel condition, the "
                "control-trial activity and the fixed reference direction the stimulated trials are scored "
                "against, in the session's own bipolar-channel coordinate basis. What is missing is this "
                "leg's own occupied-space machinery (_cv_pca_rank / _occupied_space_decomposition) applied "
                "to that control-trial activity, with the displacement direction (the stimulated-trial mean "
                "unit vector minus the reference direction, renormalised) in place of this analysis's "
                "leading eigenvector. This was not run here: it requires reloading every RAM session's raw iEEG feature "
                "cache through that module's own loader, which this module's compute budget did not include. "
                "Stating this precisely, rather than reporting an absent number as a null, is the result "
                "this added block asks for where the decomposition is not run."
            ),
        },
    }


# =======================================================================================================
# Driver
# =======================================================================================================

def _residual_identity_clause(bundles_by_corpus: dict[str, list[dict]]) -> dict:
    """Second clause of the pre-fit gate: on the very trials the blocks analyse, the delivered deviation
    d_i and the residual r_i must stand in the exact algebraic relation that licenses reading ||r_i||'s
    direction -- d_i = 1 - (u_i . m_i) with ||r_i||^2 = 1 - (u_i . m_i)^2 -- to the same tolerance the
    estimator-reproduction clause uses. Verified per session, per item-count level for the multi-object
    corpus, every trial counted."""
    detail: dict[str, list] = {}
    worst_diff = 0.0
    all_masks_match = True
    for corpus_key, bundles in bundles_by_corpus.items():
        records = []
        for bundle in bundles:
            if corpus_key == CORPORA[0]:
                levels = [("all", bundle["activity_by_unit"])]
            else:
                item_count = bundle["item_count"]
                levels = [(str(level), bundle["activity_by_unit"][item_count == float(level)])
                          for level in sorted({int(v) for v in item_count.tolist()})]
            for level, activity in levels:
                check = residual_decomposition_and_identity_check(activity)
                finite = check["finite"]
                cosine = check["cosine"][finite]
                deviation = check["deviation_delivered"][finite]
                residual_norm_sq = np.sum(check["residual"][finite] ** 2, axis=1)
                dev_diff = float(np.max(np.abs(deviation - (1.0 - cosine)))) if cosine.size else 0.0
                norm_diff = float(np.max(np.abs(residual_norm_sq - (1.0 - cosine ** 2)))) if cosine.size else 0.0
                worst_diff = max(worst_diff, dev_diff, norm_diff)
                all_masks_match = all_masks_match and bool(check["identity_same_finite_mask"])
                records.append({
                    "session": bundle["session"], "level": level,
                    "n_trials": int(activity.shape[0]), "n_trials_with_defined_direction": int(finite.sum()),
                    "deviation_minus_one_minus_cosine_max_abs_diff": dev_diff,
                    "residual_norm_squared_minus_one_minus_cosine_squared_max_abs_diff": norm_diff,
                    "finite_mask_matches_between_delivered_and_recomputed": bool(check["identity_same_finite_mask"]),
                })
        detail[corpus_key] = records
    return {
        "tolerance": REPRODUCTION_TOLERANCE,
        "max_abs_diff_over_all_sessions_and_levels": worst_diff,
        "all_finite_masks_match": bool(all_masks_match),
        "passed": bool(worst_diff < REPRODUCTION_TOLERANCE and all_masks_match),
        "per_session_or_level": detail,
    }


def _delivered_reference_counts() -> dict:
    refs: dict[str, dict] = {}
    path = ROOT / "results" / "rate_free_state_geometry_behavior_link.json"
    if path.exists():
        j = json.loads(path.read_text())
        refs[CORPORA[0]] = {
            "artifact": "results/rate_free_state_geometry_behavior_link.json",
            "n_sessions_reachable": j.get("n_sessions_reachable"),
            "n_sessions_computed": j.get("n_sessions_computed"),
        }
    path = ROOT / "results" / "watters_state_geometry.json"
    if path.exists():
        j = json.loads(path.read_text())
        accounting = j.get("zero_drop_accounting", {})
        refs[CORPORA[1]] = {
            "artifact": "results/watters_state_geometry.json",
            "n_sessions_seen": j.get("n_sessions_seen"),
            "n_sessions_refused": accounting.get("n_sessions_refused"),
            "n_sessions_analysed": accounting.get("n_sessions_analysed"),
        }
    return refs


def main() -> None:
    t0 = time.time()
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Run only on the two corpora whose rate-free deviation observable passes its own orthogonality "
            "gate against total spike count: the single-item macaque lateral prefrontal cortex corpus "
            "(Panichello et al. 2024) and the multi-object macaque corpus (Watters, Gabel, Tenenbaum and "
            "Jazayeri; DANDI 000620). The mouse anterior lateral motor cortex corpus and both human corpora "
            "are excluded here on that already-measured precondition, not left pending. The multi-object "
            "corpus is analysed WITHIN item-count level throughout and combined across levels by "
            "trial-count weighting; a pooled-across-item-count number is never reported as its effect size."
        ),
        "sign_map": SIGN_TO_WORSE_BEHAVIOUR,
        "sign_map_note": (
            "Applied once, at the point a result is packaged, never inside an estimator. Every reported "
            "behavioural coefficient is against WORSE behaviour."
        ),
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass, shared by the reproduction gate and every block)")
    watters_seen, watters_loaded, watters_refused = 0, [], []
    for session in iter_watters(root, bin_ms=100.0):
        watters_seen += 1
        if session["status"] != "loaded":
            watters_refused.append({"session": session["session"], "status": session["status"]})
            continue
        watters_loaded.append(session)
    _log(f"multi-object macaque corpus: {watters_seen} seen, {len(watters_loaded)} loaded, "
         f"{len(watters_refused)} refused, elapsed={time.time() - t0:.0f}s")

    _log("running the reproduction gate on both corpora")
    gate_result, watters_arrays_by_session = full_reproduction_gate(root, watters_loaded)

    macaque_paths = _reachable_sessions(root)
    macaque_bundles, _ = _macaque_bundles(root)
    n_macaque_on_disk = len(list(_panichello_directory(root).glob("*.mat"))) if _panichello_directory(root) else 0

    watters_bundles = _watters_bundles(watters_arrays_by_session)
    n_watters_arrays_none = len(watters_loaded) - len(watters_arrays_by_session)
    _log(f"bundles: macaque {len(macaque_bundles)}/{len(macaque_paths)} reachable, "
         f"multi-object {len(watters_bundles)}/{len(watters_loaded)} loaded, elapsed={time.time() - t0:.0f}s")

    identity = _residual_identity_clause({CORPORA[0]: macaque_bundles, CORPORA[1]: watters_bundles})
    output["reproduction_gate"] = {**gate_result, "residual_identity_clause": identity}
    output["reachability"] = {
        CORPORA[0]: {
            "n_sessions_on_disk": n_macaque_on_disk,
            "n_sessions_reaching_the_reachability_floor": len(macaque_paths),
            "n_sessions_with_a_computed_bundle": len(macaque_bundles),
            "n_trials_total": sum(b["n_trials_total"] for b in macaque_bundles),
            "median_units": (float(np.median([b["activity_by_unit"].shape[1] for b in macaque_bundles]))
                             if macaque_bundles else None),
            "reference_effect_size_r_units": abs(float(_MACAQUE_DELIVERED_RAW_R)),
        },
        CORPORA[1]: {
            "n_sessions_seen": watters_seen, "n_sessions_loaded": len(watters_loaded),
            "n_sessions_refused_by_the_shared_loader": len(watters_refused),
            "n_sessions_arrays_not_computable": n_watters_arrays_none,
            "n_sessions_with_a_computed_bundle": len(watters_bundles),
            "n_trials_total": sum(int(np.asarray(s["counts"]).shape[0]) for s in watters_loaded),
            "n_trials_analysed_within_bundles": sum(int(np.asarray(b["activity_by_unit"]).shape[0])
                                                     for b in watters_bundles),
            "median_units": (float(np.median([b["activity_by_unit"].shape[1] for b in watters_bundles]))
                             if watters_bundles else None),
            "reference_effect_size_r_units": abs(float(_WATTERS_DELIVERED_RAW_R)),
        },
    }
    output["zero_drop_accounting"] = {
        CORPORA[0]: {
            "n_seen": n_macaque_on_disk,
            "n_excluded_below_reachability_floor": n_macaque_on_disk - len(macaque_paths),
            "n_reaching_floor": len(macaque_paths),
            "n_excluded_too_few_trials_with_defined_direction": len(macaque_paths) - len(macaque_bundles),
            "n_analysed": len(macaque_bundles),
            "reconciles": bool(n_macaque_on_disk == (n_macaque_on_disk - len(macaque_paths))
                               + (len(macaque_paths) - len(macaque_bundles)) + len(macaque_bundles)),
            "matches_delivered_artifact_session_count": bool(
                len(macaque_bundles) == (_delivered_reference_counts().get(CORPORA[0], {}) or {}).get(
                    "n_sessions_computed", -1)),
        },
        CORPORA[1]: {
            "n_seen": watters_seen, "n_refused_by_shared_loader": len(watters_refused),
            "n_loaded": len(watters_loaded), "n_arrays_not_computable": n_watters_arrays_none,
            "n_analysed": len(watters_bundles),
            "reconciles": bool(watters_seen == len(watters_refused) + n_watters_arrays_none + len(watters_bundles)),
            "refusal_reasons_by_session": watters_refused,
        },
    }
    output["delivered_artifact_reference_counts"] = _delivered_reference_counts()
    _flush(output)
    _log(f"reproduction gate: estimator clause={gate_result['status']}, "
         f"residual-identity clause passed={identity['passed']} "
         f"(max abs diff {identity['max_abs_diff_over_all_sessions_and_levels']:.3e})")

    if gate_result["status"] != "reproduced_exactly" or not identity["passed"]:
        output["status"] = "void_reproduction_gate_did_not_reproduce"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce; no new number was read")
        print(json.dumps({"reproduction_gate": gate_result["status"],
                          "residual_identity_passed": identity["passed"]}, indent=2))
        return

    corpora = {CORPORA[0]: macaque_bundles, CORPORA[1]: watters_bundles}
    delivered_reference_effect = {
        CORPORA[0]: abs(float(_MACAQUE_DELIVERED_RAW_R)),
        CORPORA[1]: abs(float(_WATTERS_DELIVERED_RAW_R)),
    }

    output["anisotropy_block"] = {}
    concentration: dict[str, bool] = {}
    for corpus_key, bundles in corpora.items():
        _log(f"residual-direction anisotropy: {corpus_key} ({len(bundles)} sessions)")
        result = run_anisotropy_test(bundles, corpus_key)
        output["anisotropy_block"][corpus_key] = result
        concentration[corpus_key] = result["branch"] in (
            "residual_directions_concentrate_on_an_axis",
            "residual_directions_are_anisotropic_but_not_on_a_single_axis",
        )
        output["zero_drop_accounting"][corpus_key]["n_trials_excluded_by_residual_norm_floor_across_sessions"] = int(
            sum(s.get("n_trials_excluded_by_residual_floor", 0) for s in result["per_session"]))
        _flush(output)
        _log(f"  branch: {result['branch']} elapsed={time.time() - t0:.0f}s")

    output["axis_alignment_block"] = {}
    stability_significant: dict[str, bool] = {}
    for corpus_key, bundles in corpora.items():
        if not concentration[corpus_key]:
            output["axis_alignment_block"][corpus_key] = {
                "status": "not_run_no_concentration_branch_for_this_corpus"}
            stability_significant[corpus_key] = False
            continue
        _log(f"axis alignment against reference directions: {corpus_key} ({len(bundles)} sessions)")
        result = run_axis_alignment(bundles, corpus_key)
        output["axis_alignment_block"][corpus_key] = result
        stability_significant[corpus_key] = bool(result["axis_stability_summary"].get("stable_by_majority"))
        _flush(output)
        _log(f"  axis stability stable_by_majority={stability_significant[corpus_key]} "
             f"elapsed={time.time() - t0:.0f}s")

    output["occupied_state_space_block"] = {}
    for corpus_key, bundles in corpora.items():
        if not concentration[corpus_key]:
            output["occupied_state_space_block"][corpus_key] = {
                "status": "not_run_no_concentration_branch_for_this_corpus"}
            continue
        _log(f"occupied-state-space decomposition: {corpus_key} ({len(bundles)} sessions)")
        output["occupied_state_space_block"][corpus_key] = run_occupied_space_block(bundles, corpus_key)
        _flush(output)
        _log(f"  branch: {output['occupied_state_space_block'][corpus_key]['branch']} "
             f"elapsed={time.time() - t0:.0f}s")

    output["signed_displacement_block"] = {}
    for corpus_key, bundles in corpora.items():
        if not concentration[corpus_key]:
            output["signed_displacement_block"][corpus_key] = {
                "status": "not_run_no_concentration_branch_for_this_corpus"}
            continue
        if not stability_significant[corpus_key]:
            output["signed_displacement_block"][corpus_key] = {
                "status": "not_run_within_session_axis_stability_not_above_its_null"}
            continue
        _log(f"signed versus unsigned projection: {corpus_key} ({len(bundles)} sessions)")
        output["signed_displacement_block"][corpus_key] = run_signed_displacement_test(
            bundles, corpus_key, SIGN_TO_WORSE_BEHAVIOUR[corpus_key], delivered_reference_effect[corpus_key])
        _flush(output)
        _log(f"  branch: {output['signed_displacement_block'][corpus_key]['branch']} "
             f"elapsed={time.time() - t0:.0f}s")

    output["stimulation_direction_alignment_block"] = stimulation_direction_alignment_block()
    output["human_footprint_block"] = human_footprint_block()

    output["how_this_artifact_was_assembled"] = {
        "per_unit_checkpoint_directory": str(CHECKPOINT_DIR.relative_to(ROOT)),
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": output["reproduction_gate"]["status"],
        "anisotropy_branch": {k: v["branch"] for k, v in output["anisotropy_block"].items()},
        "axis_stability_stable_by_majority": stability_significant,
        "occupied_state_space_branch": {
            k: (v.get("branch") or v.get("status")) for k, v in output["occupied_state_space_block"].items()},
        "signed_displacement_branch": {
            k: (v.get("branch") or v.get("status")) for k, v in output["signed_displacement_block"].items()},
        "wall_clock_s": output["wall_clock_s"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
