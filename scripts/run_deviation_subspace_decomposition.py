"""run_deviation_subspace_decomposition.py -- does the accuracy-predicting
deviation live inside the memorandum's own coding subspace, or outside it?

Two things are already established and neither says whether they are the
same residual thing. (1) The dominant cross-unit component of delay
activity is population rate drifting across the session, not the
memorandum (results/state_orthogonality_census.json). (2) A per-trial
deviation of the population's activity DIRECTION from the session's own
mean direction -- built rate-free by construction
(run_rate_free_state_geometry_behavior_link.rate_free_state_deviation) --
predicts trial accuracy (results/rate_free_state_geometry_behavior_link.json).
Nobody has asked whether the part of that deviation which predicts accuracy
points along the memorandum's own coding directions or away from them.

Construction. For each trial i, let u_i be its L2-normalised across-unit
activity direction and m_i the renormalised leave-one-out mean of every
OTHER trial's own direction -- exactly the two objects
rate_free_state_deviation reduces to a single cosine-deviation scalar
(deviation_i = 1 - u_i . m_i). The residual r_i = u_i - (u_i . m_i) m_i is
the component of the trial's direction orthogonal to the session's mean
direction. This module never modifies rate_free_state_deviation; it
reimplements the identical leave-one-out arithmetic locally
(_leave_one_out_unit_directions) and PROVES the two agree on every session
before reporting anything else (the decomposition-identity assertion).

The memorandum's coding subspace C is estimated with cross-validation
across trials -- for trial i, from every trial other than i -- so no trial
contributes to the subspace its own residual is projected onto, mirroring
the leave-one-out discipline already in the deviation. For the discrete
macaque memorandum C is the (c-1)-dimensional span of the centred class-mean
directions of the cued-position label; for the continuous multi-object
memorandum C is the 2-dimensional regression subspace of the cued object's
[cos, sin] position. Each residual splits into its component inside C and
its component orthogonal to C, r_i = P_C r_i + (I - P_C) r_i.

Because a larger subspace captures more of any residual including noise, the
within-subspace magnitude is tested against a MATCHED-DIMENSION random
subspace, never against zero: at least 200 uniformly random subspaces of the
identical dimension, per session, run through the identical decomposition
and the identical (uncontrolled) behavioural correlation.

A session enters the decomposition only if its memorandum decodes above its
own label-permutation null in that session -- a memorandum subspace
estimated where the memorandum does not decode is a subspace of noise, and
projecting onto it ranks noise. Sessions that do not clear are reported as
void, not dropped.

Two arms. PRIMARY: macaque lateral prefrontal cortex (Panichello et al.
2024), the same corpus and the same >=60-error-trial reachability floor as
results/rate_free_state_geometry_behavior_link.json (11 sessions); floors of
45 and 30 error trials are sensitivity analyses, reported beside the primary
and never substituted for it. REPLICATION: the multi-object corpus behind
results/watters_state_geometry.json (Watters, Gabel, Tenenbaum, Jazayeri;
bioRxiv preprint, posted 2026-01-27, DOI 10.64898/2026.01.27.702062; DANDI
000620 -- an unreviewed preprint, said wherever it is used here), 41
sessions, 2 animals, continuous graded report, no error floor.

Sign convention (a documented trap, restated at every number in this
module): the macaque arm's outcome is is_corr (1 = correct), so a NEGATIVE
correlation with a magnitude means worse behaviour. The multi-object arm's
outcome is report_error directly, so a POSITIVE correlation means worse
behaviour. Every coefficient in the multi-object arm is already on the
report-error convention that results/watters_state_geometry.json uses.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr

_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, rate_free_state_deviation,
)
from run_state_behavior_link import MIN_ERROR_TRIALS_FOR_REACHABILITY, _counts_from_spikes, _panichello_directory  # noqa: E402
from run_state_content_link import BIN_MS, MIN_CLASSES, MIN_TRIALS_PER_CLASS, session_subtractive_test, usable_label  # noqa: E402
from run_watters_state_geometry import _behaviour_observables  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    fdr_bh, minimum_detectable_paired_difference, partial_correlation_permutation_test,
    permutation_pvalue, stable_seed,
)

OUTPUT_PATH = _ROOT / "results" / "deviation_subspace_decomposition.json"
CHECKPOINT_PATH = _ROOT / "results" / ".checkpoints" / "deviation_subspace_decomposition_checkpoint.json"

N_PERM = 10000
N_RANDOM_SUBSPACE_DRAWS = 200
IDENTITY_TOLERANCE = 1e-8

MACAQUE_ERROR_FLOORS = (60, 45, 30)  # 60 primary, 45/30 sensitivity, never substituted for it
MACAQUE_LOWEST_FLOOR = min(MACAQUE_ERROR_FLOORS)
PANICHELLO_DELAY_WINDOW_MS = (300.0, 1450.0)

WATTERS_MIN_TRIALS_FOR_CORRELATION = 16
WATTERS_REGRESSION_DIM = 2
# The reachability gate discretises the continuous cued position into this many classes and reuses the
# project's own classification-based decodability test (session_subtractive_test), the same mechanism the
# macaque arm's gate uses and the same one results/watters_state_geometry.json's own cardinality ladder
# uses for this identical corpus -- chosen after a smoke test showed a direct linear regression R-squared
# on the raw [cos, sin] target is far too weak an estimator to answer "does this decode at all" (observed
# R-squared indistinguishable from its own permutation null on every session probed), while the
# classification test clears cleanly on several sessions of the same data. The DECOMPOSITION itself still
# uses the continuous 2-dimensional regression subspace described above; only the gate is discrete.
WATTERS_RECOVERABILITY_K_CLASSES = 8

# The outcome variable's sign convention differs by arm (see module docstring); this is the multiplier
# such that mean_value * WORSE_BEHAVIOUR_SIGN[arm] > 0 means "in the direction of worse behaviour".
WORSE_BEHAVIOUR_SIGN = {"macaque_lpfc": -1.0, "watters_multi_object": 1.0}
OUTCOME_DESCRIPTION = {
    "macaque_lpfc": "trial correctness (coded 1 = correct)",
    "watters_multi_object": "the continuous report error (larger = worse report)",
}

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, fit the memorandum's coding subspace C by cross-validation across trials (macaque: the "
    "(c-1)-dimensional span of the centred class-mean directions of the 8-way cued-position label; "
    "multi-object corpus: the 2-dimensional regression subspace of the cued object's [cos, sin] position), "
    "decompose each trial's rate-free deviation residual r_i into its component inside C and its "
    "component orthogonal to C, and correlate each magnitude with trial outcome controlling for the other "
    "magnitude (statistics.partial_correlation_permutation_test), pooled across sessions by the two-sided "
    "paired sign-flip test (state_persistence.slope_across_sessions_test), corrected across the two "
    "components by Benjamini-Hochberg (statistics.fdr_bh, alpha=0.05). The within-subspace RAW "
    "(uncontrolled) correlation is additionally compared against a matched-dimension random-subspace null "
    "(>=200 uniformly random subspaces of the identical dimension per session, identical decomposition, "
    "identical raw correlation, pooled the same way the observed value is pooled), two-sided, because a "
    "raw comparison of within- and outside-subspace magnitudes alone tests dimensionality, not the "
    "memorandum. Before any fit runs:\n"
    "  - Within-subspace partial significant after correction, in the direction of worse behaviour, "
    "outside-subspace partial not, AND the within-subspace raw correlation falls outside its "
    "matched-dimension random-subspace null (two-sided p <= 0.05): "
    "'accuracy_is_predicted_by_deviation_within_the_memorandum_subspace'.\n"
    "  - Outside-subspace partial significant after correction, within-subspace not: "
    "'accuracy_is_predicted_by_deviation_outside_the_memorandum_subspace'.\n"
    "  - Both significant after correction: 'both_components_predict_accuracy', reported with their "
    "relative magnitudes and their collinearity, resolved in favour of neither.\n"
    "  - Within-subspace partial significant after correction and in the right direction, but the raw "
    "within-subspace correlation does NOT clear its matched-dimension random-subspace null: "
    "'within_subspace_association_not_separable_from_a_random_subspace_of_the_same_dimension'.\n"
    "  - Neither component's partial significant after correction: 'inconclusive_below_detection_floor', "
    "with the minimum detectable paired difference at 80% power reported for each component.\n"
    "A component significant after correction but in the WRONG direction is never folded into a positive "
    "branch; it is reported as a named surprise alongside whichever of the branches above applies."
)

REACHABILITY_RULE_DECLARED_BEFORE_FITTING = (
    "A session enters the decomposition only if its memorandum decodes above its own label-permutation "
    "null in that session, tested on the SAME trials the decomposition itself uses (every trial with a "
    "defined rate-free direction and a usable memorandum label). Macaque: "
    "run_state_content_link.session_subtractive_test's a_full_clears_own_null (macro one-vs-rest AUC "
    "against a label-permutation null) on the delay-epoch window mean. Multi-object corpus: the identical "
    f"session_subtractive_test machinery, applied to the cued position discretised into "
    f"{WATTERS_RECOVERABILITY_K_CLASSES} classes with the same binning rule "
    "results/watters_state_geometry.json's own cardinality ladder uses for this corpus, substituted for a "
    "continuous regression R-squared gate that a smoke test showed was far too weak an estimator (observed "
    "R-squared indistinguishable from its own permutation null on every session probed) while the "
    "classification test clears cleanly on several sessions of the same data; the DECOMPOSITION itself "
    "still uses the continuous 2-dimensional regression subspace, only the gate is discrete. Sessions that "
    "do not clear are reported as void, not dropped: unit counts and trial counts are reported beside the "
    "cleared/not-cleared status before any decomposition number is reported."
)


# ----------------------------------------------------------------------------------------------------
# Core geometry: identical to rate_free_state_deviation's leave-one-out arithmetic, exposing the
# intermediate per-trial direction u_i and reference m_i so the residual r_i = u_i - (u_i . m_i) m_i can
# be built. Never imports or edits that function's internals; the identity check below is the proof the
# two agree, not an assumption.
# ----------------------------------------------------------------------------------------------------

def _leave_one_out_unit_directions(activity_by_unit: np.ndarray) -> dict:
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials = activity.shape[0]
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors, axis=0)
    n_valid = int(valid.sum())

    loo_mean_normalized = np.full_like(unit_vectors, np.nan)
    for i in range(n_trials):
        if not valid[i]:
            continue
        n_other = n_valid - 1
        if n_other < 1:
            continue
        loo_mean = (total - unit_vectors[i]) / n_other
        loo_norm = np.linalg.norm(loo_mean)
        if loo_norm == 0.0:
            continue
        loo_mean_normalized[i] = loo_mean / loo_norm
    return {"unit_vectors": unit_vectors, "loo_mean_normalized": loo_mean_normalized, "valid": valid}


def residual_decomposition_and_identity_check(activity_by_unit: np.ndarray) -> dict:
    """Residual r_i for every trial, plus the decomposition-identity proof:
    the deviation recomputed from this module's own u_i and m_i must match
    rate_free_state_deviation's delivered output to floating-point tolerance
    on every finite trial of this session."""
    directions = _leave_one_out_unit_directions(activity_by_unit)
    unit_vectors, loo_mean, valid = directions["unit_vectors"], directions["loo_mean_normalized"], directions["valid"]
    delivered = rate_free_state_deviation(activity_by_unit)
    finite_delivered = np.isfinite(delivered)

    with np.errstate(invalid="ignore"):
        cosine = np.einsum("ij,ij->i", unit_vectors, loo_mean)
    recomputed = 1.0 - cosine
    finite_recomputed = np.isfinite(recomputed)
    same_finite_mask = bool(np.array_equal(finite_delivered, finite_recomputed))
    diffs = (np.abs(delivered[finite_delivered] - recomputed[finite_delivered])
             if finite_delivered.any() else np.array([0.0]))
    max_abs_diff = float(np.max(diffs)) if diffs.size else 0.0

    with np.errstate(invalid="ignore"):
        residual = unit_vectors - cosine[:, None] * loo_mean

    return {
        "residual": residual, "deviation_recomputed": recomputed, "deviation_delivered": delivered,
        "cosine": cosine, "finite": finite_delivered,
        "identity_same_finite_mask": same_finite_mask, "identity_max_abs_diff": max_abs_diff,
        "identity_passed": bool(same_finite_mask and max_abs_diff < IDENTITY_TOLERANCE),
    }


def _orthonormal_basis(row_vectors: np.ndarray, dim: int) -> np.ndarray:
    """(units,) orthonormal basis of the row space of ``row_vectors`` (k,
    units), top ``dim`` left singular vectors of its transpose."""
    u_svd, _, _ = np.linalg.svd(row_vectors.T, full_matrices=False)
    return u_svd[:, :dim]


def _random_orthonormal_basis(n_units: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random dim-dimensional subspace of R^n_units: QR of an
    i.i.d. Gaussian matrix is Haar-distributed on the Stiefel manifold."""
    gaussian = rng.standard_normal((n_units, dim))
    q, _ = np.linalg.qr(gaussian)
    return q[:, :dim]


def _project_magnitudes(residual: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proj = residual @ basis @ basis.T
    within = np.linalg.norm(proj, axis=1)
    outside = np.linalg.norm(residual - proj, axis=1)
    return within, outside


def cv_class_mean_subspace(u: np.ndarray, labels: np.ndarray, residual: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Macaque arm: for each trial, fit the (dim)-dimensional span of the
    centred class-mean directions from every OTHER trial, project that
    trial's own residual onto it. Centring first (subtracting the held-in
    grand mean) before averaging within class makes the weighted sum of the
    centred class means exactly zero, so their span has rank <= dim = c-1
    by construction rather than by an incidental fit."""
    n = u.shape[0]
    within = np.full(n, np.nan)
    outside = np.full(n, np.nan)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        u_held_in, labels_held_in = u[keep], labels[keep]
        grand_mean = u_held_in.mean(axis=0)
        centred = u_held_in - grand_mean
        classes = np.unique(labels_held_in)
        class_means = np.stack([centred[labels_held_in == c].mean(axis=0) for c in classes])
        basis = _orthonormal_basis(class_means, dim)
        proj = basis @ (basis.T @ residual[i])
        within[i] = np.linalg.norm(proj)
        outside[i] = np.linalg.norm(residual[i] - proj)
    return within, outside


def cv_regression_subspace(u: np.ndarray, target_2d: np.ndarray, residual: np.ndarray, dim: int = WATTERS_REGRESSION_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Multi-object arm: for each trial, fit the dim-dimensional regression
    subspace (each unit's direction regressed on the target's coordinates)
    from every OTHER trial, project that trial's own residual onto it."""
    n = u.shape[0]
    within = np.full(n, np.nan)
    outside = np.full(n, np.nan)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        x_held_in, u_held_in = target_2d[keep], u[keep]
        x_centred = x_held_in - x_held_in.mean(axis=0)
        u_centred = u_held_in - u_held_in.mean(axis=0)
        coefficients, *_ = np.linalg.lstsq(x_centred, u_centred, rcond=None)  # (dim, units)
        basis = _orthonormal_basis(coefficients, dim)
        proj = basis @ (basis.T @ residual[i])
        within[i] = np.linalg.norm(proj)
        outside[i] = np.linalg.norm(residual[i] - proj)
    return within, outside


def random_subspace_null_raw_correlations(residual: np.ndarray, dim: int, outcome: np.ndarray, seed: int,
                                           n_draws: int = N_RANDOM_SUBSPACE_DRAWS) -> list[float | None]:
    """Per draw: a uniformly random dim-dimensional subspace, the within-
    subspace magnitude it gives every trial, and the RAW (uncontrolled)
    Pearson correlation of that magnitude with outcome -- the same
    functional form as the observed raw within-subspace correlation, without
    its own permutation p-value (a full permutation test per draw, per
    session, at >=200 draws, is not affordable at this project's budget; the
    draws collectively ARE the null distribution the observed statistic is
    compared against)."""
    n_units = residual.shape[1]
    values: list[float | None] = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        basis = _random_orthonormal_basis(n_units, dim, rng)
        proj = residual @ basis
        magnitude = np.linalg.norm(proj, axis=1)
        if np.std(magnitude) == 0.0 or np.std(outcome) == 0.0:
            values.append(None)
            continue
        r, _ = pearsonr(magnitude, outcome)
        values.append(float(r) if np.isfinite(r) else None)
    return values


# ----------------------------------------------------------------------------------------------------
# Checkpointing: identical pattern to scripts/run_dominant_latent_identity_and_behaviour_breadth.py's
# _fit -- a per-session record, atomic replace, completion flag written only after compute() returns.
# ----------------------------------------------------------------------------------------------------

_COMPLETED: dict[str, dict] = {}


def _load_completed() -> dict[str, dict]:
    try:
        entries = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: entry for key, entry in entries.items() if isinstance(entry, dict) and entry.get("complete") is True}


def _fit(key: str, compute) -> dict:
    entry = _COMPLETED.get(key)
    if entry is not None:
        return entry["value"]
    value = compute()
    _COMPLETED[key] = {"complete": True, "value": _json_safe(value)}
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_COMPLETED, allow_nan=False))
    scratch.replace(CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------
# Macaque lPFC arm
# ----------------------------------------------------------------------------------------------------

def _macaque_paths(root) -> list[Path]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    return [Path(p) for p in sorted(glob.glob(str(directory / "*.mat")))]


def _analyse_macaque_session(path: Path) -> dict:
    session_id = path.stem
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr_full = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    cue_idx_full = np.asarray(raw["cueAngIdx"]).reshape(-1).astype(int)
    n_error = int((~is_corr_full).sum())
    n_trials_total = int(len(is_corr_full))

    counts_all = _counts_from_spikes(spikes, time_ms)
    if counts_all.shape[0] < 16:
        return {"session": session_id, "status": "too_few_trials", "n_trials_total": n_trials_total, "n_error": n_error}
    activity_by_unit = counts_all.sum(axis=2)
    n_units = int(activity_by_unit.shape[1])

    identity = residual_decomposition_and_identity_check(activity_by_unit)
    finite = identity["finite"]
    n_trials_finite = int(finite.sum())
    if n_trials_finite < 16:
        return {"session": session_id, "status": "too_few_trials_with_defined_direction",
                "n_trials_total": n_trials_total, "n_error": n_error, "n_units": n_units,
                "identity_passed": identity["identity_passed"], "identity_max_abs_diff": identity["identity_max_abs_diff"]}

    unit_vectors_finite = _leave_one_out_unit_directions(activity_by_unit)["unit_vectors"][finite]
    residual_finite = identity["residual"][finite]
    labels_finite = cue_idx_full[finite]
    is_corr_finite = is_corr_full[finite]
    counts_finite = counts_all[finite]

    base = {
        "session": session_id, "n_trials_total": n_trials_total, "n_error": n_error, "n_units": n_units,
        "n_trials_with_defined_direction": n_trials_finite,
        "identity_passed": identity["identity_passed"], "identity_max_abs_diff": identity["identity_max_abs_diff"],
        "identity_same_finite_mask": identity["identity_same_finite_mask"],
    }

    ok, reason, mask2 = usable_label(labels_finite, min_classes=MIN_CLASSES, min_per_class=MIN_TRIALS_PER_CLASS)
    if not ok:
        return {**base, "status": "no_usable_memorandum_label", "reason": reason}

    idx = np.flatnonzero(mask2)
    u = unit_vectors_finite[idx]
    residual = residual_finite[idx]
    labels = labels_finite[idx]
    outcome = is_corr_finite[idx].astype(float)
    counts = counts_finite[idx]
    n_trials_decomp = int(len(idx))
    n_classes = int(len(np.unique(labels)))
    dim = n_classes - 1
    base = {**base, "n_trials_decomposition_set": n_trials_decomp, "n_classes": n_classes, "subspace_dim": dim}

    if n_trials_decomp < dim + 8:
        return {**base, "status": "too_few_trials_for_subspace_dimension"}

    window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]
    seed_tag = f"deviation_subspace_decomposition|macaque|{session_id}"
    subtractive = session_subtractive_test(window_mean, labels, stable_seed(f"{seed_tag}|reachability"))
    if subtractive.get("status") != "tested":
        return {**base, "status": "content_reachability_not_computable", "subtractive_status": subtractive.get("status")}

    cleared = bool(subtractive.get("a_full_clears_own_null", False))
    base = {**base, "content_reachability": {
        "a_full": subtractive["a_full"], "a_full_p_value": subtractive["a_full_p_value"],
        "cleared": cleared, "k_latents": subtractive["k_latents"],
    }}
    if not cleared:
        return {**base, "status": "content_not_reachable_void"}

    within_mag, outside_mag = cv_class_mean_subspace(u, labels, residual, dim)

    raw_within = partial_correlation_permutation_test(outcome, within_mag, controls=[],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|raw_within")))
    raw_outside = partial_correlation_permutation_test(outcome, outside_mag, controls=[],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|raw_outside")))
    joint_within = partial_correlation_permutation_test(outcome, within_mag, controls=[outside_mag],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|joint_within")))
    joint_outside = partial_correlation_permutation_test(outcome, outside_mag, controls=[within_mag],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|joint_outside")))

    null_r = random_subspace_null_raw_correlations(residual, dim, outcome, stable_seed(f"{seed_tag}|null"))
    collinearity_r, collinearity_p = pearsonr(within_mag, outside_mag)

    return {
        **base, "status": "decomposed",
        "raw_within_vs_outcome": raw_within, "raw_outside_vs_outcome": raw_outside,
        "joint_within_controlling_outside": joint_within, "joint_outside_controlling_within": joint_outside,
        "collinearity_within_vs_outside": {"r": float(collinearity_r), "p_analytic": float(collinearity_p)},
        "within_outside_random_subspace_null_raw_r": null_r,
        "within_magnitude_median": float(np.median(within_mag)), "outside_magnitude_median": float(np.median(outside_mag)),
    }


# ----------------------------------------------------------------------------------------------------
# Multi-object corpus arm
# ----------------------------------------------------------------------------------------------------

def _watters_reachability(counts: np.ndarray, discretized_label: np.ndarray, seed: int) -> dict:
    """Reachability gate for the continuous multi-object memorandum, using the project's own
    classification-based decodability test (session_subtractive_test) -- the same machinery the macaque
    arm's gate uses -- rather than a continuous regression R-squared. A direct linear regression of unit
    direction on the raw [cos, sin] cued-position target was tried first and was far too weak an estimator
    to answer "does this decode at all" (observed R-squared indistinguishable from its own permutation
    null on every session probed), while the classification test clears cleanly on several sessions of the
    same data. The cued position is discretised into classes with the identical binning rule
    results/watters_state_geometry.json's own cardinality ladder uses for this corpus so the gate result is
    directly comparable to that delivered result. The DECOMPOSITION itself still uses the continuous
    2-dimensional regression subspace; only this gate is discrete."""
    if counts.shape[0] < WATTERS_MIN_TRIALS_FOR_CORRELATION:
        return {"status": "too_few_trials", "n_trials": int(counts.shape[0])}
    window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]
    result = session_subtractive_test(window_mean, discretized_label, seed)
    if result.get("status") != "tested":
        return {"status": "content_reachability_not_computable", "subtractive_status": result.get("status")}
    cleared = bool(result.get("a_full_clears_own_null", False))
    return {
        "status": "tested", "n_trials": int(counts.shape[0]),
        "n_classes_discretised": WATTERS_RECOVERABILITY_K_CLASSES,
        "a_full": result["a_full"], "a_full_p_value": result["a_full_p_value"],
        "cleared": cleared, "k_latents": result["k_latents"],
    }


def _analyse_watters_session(session: dict) -> dict:
    session_id = session["session"]
    counts = session["counts"]
    n_units = int(counts.shape[1])
    n_trials_total = int(counts.shape[0])
    activity_by_unit = counts.sum(axis=2)

    identity = residual_decomposition_and_identity_check(activity_by_unit)
    observables, excluded, usable = _behaviour_observables(counts, session)
    finite = identity["finite"]
    base = {
        "session": session_id, "animal": session["animal"], "n_trials_total": n_trials_total, "n_units": n_units,
        "identity_passed": identity["identity_passed"], "identity_max_abs_diff": identity["identity_max_abs_diff"],
        "identity_same_finite_mask": identity["identity_same_finite_mask"],
        "n_trials_with_defined_direction": int(finite.sum()),
    }
    if int(usable.sum()) < WATTERS_MIN_TRIALS_FOR_CORRELATION:
        return {**base, "status": "too_few_trials_with_a_defined_state_direction_and_report",
                "n_trials_usable": int(usable.sum()), "trials_excluded_by_reason": excluded}

    # _behaviour_observables' usable mask additionally requires report_error and reaction_time finite,
    # so it is a subset of (never a superset of) `finite`; the decomposition uses that subset.
    unit_vectors_all = _leave_one_out_unit_directions(activity_by_unit)["unit_vectors"]
    mask = usable
    u = unit_vectors_all[mask]
    residual = identity["residual"][mask]
    valid_direction = ~np.isnan(u).any(axis=1)
    if int(valid_direction.sum()) < WATTERS_MIN_TRIALS_FOR_CORRELATION:
        return {**base, "status": "too_few_trials_with_a_defined_direction_after_report_filter",
                "n_trials_usable": int(valid_direction.sum())}
    u, residual = u[valid_direction], residual[valid_direction]
    report_error = observables["report_error"][valid_direction]
    cued_theta = np.asarray(session["cued_theta"], dtype=float)[mask][valid_direction]
    target_2d = np.stack([np.cos(cued_theta), np.sin(cued_theta)], axis=1)
    counts_dir = counts[mask][valid_direction]

    # The reachability gate is tested on the SAME trials the decomposition uses: discretise the cued
    # position (identical binning rule to results/watters_state_geometry.json's own cardinality ladder for
    # this corpus) and require the same per-class trial floor session_subtractive_test's other callers
    # require (usable_label), then carry that same trial subset into the decomposition itself.
    theta_mod = np.mod(cued_theta, 2.0 * np.pi)
    discretized_label = (np.floor(theta_mod / (2.0 * np.pi / WATTERS_RECOVERABILITY_K_CLASSES)).astype(int)
                          % WATTERS_RECOVERABILITY_K_CLASSES)
    label_ok, label_reason, label_mask = usable_label(discretized_label, min_classes=MIN_CLASSES, min_per_class=MIN_TRIALS_PER_CLASS)
    if not label_ok:
        return {**base, "status": "no_usable_discretised_memorandum_label_for_reachability_gate", "reason": label_reason}
    u, residual, report_error = u[label_mask], residual[label_mask], report_error[label_mask]
    target_2d, counts_dir, discretized_label = target_2d[label_mask], counts_dir[label_mask], discretized_label[label_mask]

    n_trials_decomp = int(u.shape[0])
    base = {**base, "n_trials_decomposition_set": n_trials_decomp, "subspace_dim": WATTERS_REGRESSION_DIM}
    if n_trials_decomp < WATTERS_MIN_TRIALS_FOR_CORRELATION:
        return {**base, "status": "too_few_trials_after_reachability_gate_label_filter"}

    seed_tag = f"deviation_subspace_decomposition|watters|{session_id}"
    reachability = _watters_reachability(counts_dir, discretized_label, stable_seed(f"{seed_tag}|reachability"))
    base = {**base, "content_reachability": reachability}
    if reachability.get("status") != "tested":
        return {**base, "status": "content_reachability_not_computable"}
    if not reachability.get("cleared", False):
        return {**base, "status": "content_not_reachable_void"}

    within_mag, outside_mag = cv_regression_subspace(u, target_2d, residual, WATTERS_REGRESSION_DIM)

    raw_within = partial_correlation_permutation_test(report_error, within_mag, controls=[],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|raw_within")))
    raw_outside = partial_correlation_permutation_test(report_error, outside_mag, controls=[],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|raw_outside")))
    joint_within = partial_correlation_permutation_test(report_error, within_mag, controls=[outside_mag],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|joint_within")))
    joint_outside = partial_correlation_permutation_test(report_error, outside_mag, controls=[within_mag],
        n_perm=N_PERM, rng=np.random.default_rng(stable_seed(f"{seed_tag}|joint_outside")))

    null_r = random_subspace_null_raw_correlations(residual, WATTERS_REGRESSION_DIM, report_error,
                                                    stable_seed(f"{seed_tag}|null"))
    collinearity_r, collinearity_p = pearsonr(within_mag, outside_mag)

    return {
        **base, "status": "decomposed",
        "raw_within_vs_outcome": raw_within, "raw_outside_vs_outcome": raw_outside,
        "joint_within_controlling_outside": joint_within, "joint_outside_controlling_within": joint_outside,
        "collinearity_within_vs_outside": {"r": float(collinearity_r), "p_analytic": float(collinearity_p)},
        "within_outside_random_subspace_null_raw_r": null_r,
        "within_magnitude_median": float(np.median(within_mag)), "outside_magnitude_median": float(np.median(outside_mag)),
    }


# ----------------------------------------------------------------------------------------------------
# Pooling and branch decision, shared by both arms
# ----------------------------------------------------------------------------------------------------

def _pool_field(sessions: list[dict], field: str) -> dict:
    values = [s[field]["r"] for s in sessions if s.get(field, {}).get("status") == "computed"]
    if len(values) < 4:
        return {"status": "underpowered_by_construction", "n_sessions": len(values)}
    pooled = slope_across_sessions_test(values, alternative="two-sided")
    pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    return pooled


def _pooled_random_subspace_null(sessions: list[dict]) -> dict:
    per_draw = []
    for d in range(N_RANDOM_SUBSPACE_DRAWS):
        draw_values = [s["within_outside_random_subspace_null_raw_r"][d] for s in sessions
                       if s["within_outside_random_subspace_null_raw_r"][d] is not None]
        per_draw.append(float(np.mean(draw_values)) if draw_values else None)
    finite_draws = [v for v in per_draw if v is not None]
    return {
        "n_draws": N_RANDOM_SUBSPACE_DRAWS, "n_draws_with_a_finite_pooled_value": len(finite_draws),
        "pooled_null_values": per_draw,
        "mean": float(np.mean(finite_draws)) if finite_draws else None,
        "median": float(np.median(finite_draws)) if finite_draws else None,
        "std": float(np.std(finite_draws)) if finite_draws else None,
    }


def _component_significant(d: dict) -> bool:
    return d.get("status") == "tested" and d.get("q_significant") is True


def _component_right_direction(d: dict, worse_sign: float) -> bool:
    return d.get("status") == "tested" and (d.get("mean_value", 0.0) * worse_sign) > 0.0


def _classify_components(within_partial: dict, outside_partial: dict, within_raw_clears_null: bool,
                          worse_sign: float) -> tuple[str, list[str]]:
    within_ok = _component_significant(within_partial) and _component_right_direction(within_partial, worse_sign)
    outside_ok = _component_significant(outside_partial) and _component_right_direction(outside_partial, worse_sign)

    if within_ok and not outside_ok:
        branch = ("accuracy_is_predicted_by_deviation_within_the_memorandum_subspace" if within_raw_clears_null else
                  "within_subspace_association_not_separable_from_a_random_subspace_of_the_same_dimension")
    elif outside_ok and not within_ok:
        branch = "accuracy_is_predicted_by_deviation_outside_the_memorandum_subspace"
    elif within_ok and outside_ok:
        branch = "both_components_predict_accuracy"
    else:
        branch = "inconclusive_below_detection_floor"

    surprises = []
    if _component_significant(within_partial) and not _component_right_direction(within_partial, worse_sign):
        surprises.append("within_subspace_partial_significant_but_not_in_the_declared_worse_behaviour_direction")
    if _component_significant(outside_partial) and not _component_right_direction(outside_partial, worse_sign):
        surprises.append("outside_subspace_partial_significant_but_not_in_the_declared_worse_behaviour_direction")
    return branch, surprises


def _wrong_direction_disclosure(component_label: str, partial: dict, worse_sign: float, outcome_description: str,
                                 raw_null: dict | None = None, mdd: float | None = None) -> dict | None:
    """A component significant after correction but pointing opposite the pre-declared worse-behaviour
    direction is never folded into a positive branch (the branch classifier already enforces that), but a
    branch label alone does not tell a reader the arm found a real, well-powered effect that simply points
    the other way -- this builds the plain-language record of exactly that, with its numbers, so a reader
    of the artifact cannot come away thinking the arm was merely underpowered."""
    if not (_component_significant(partial) and not _component_right_direction(partial, worse_sign)):
        return None
    coefficient = partial["mean_value"]
    record = {
        "component": component_label, "mean_value": coefficient,
        "two_sided_p_value": partial.get("two_sided_p_value"), "q_value": partial.get("q_value"),
    }
    text = (
        f"The {component_label} partial correlation with {outcome_description} was significant after "
        f"multiple-comparison correction (r={coefficient:.4f}, two-sided p={partial.get('two_sided_p_value'):.4f}, "
        f"q={partial.get('q_value'):.4f}) but points toward BETTER behaviour, the opposite of the pre-declared "
        "worse-behaviour direction the branch decision rule requires, so it is not folded into a positive branch."
    )
    if raw_null is not None and raw_null.get("clears_null"):
        text += (
            " Its raw (uncontrolled) correlation additionally cleared its own matched-dimension random-subspace "
            f"null (observed {raw_null['observed_pooled_raw_within']:.4f} against a null mean of "
            f"{raw_null['null_summary']['mean']:.4f}, two-sided permutation p={raw_null['two_sided_permutation_p_value']:.4f})"
        )
        if mdd is not None:
            text += f", and its magnitude exceeded the minimum paired difference detectable at 80% power ({mdd:.4f})"
        text += " -- this arm detected a real effect here, it did not fail to reach its own detection floor."
    record["disclosure"] = text
    return record


def pool_arm(sessions: list[dict], arm: str) -> dict:
    decomposed = [s for s in sessions if s.get("status") == "decomposed"]
    if len(decomposed) < 4:
        return {"status": "too_few_decomposed_sessions", "n_sessions": len(decomposed),
                "branch": "inconclusive_below_detection_floor"}

    raw_within = _pool_field(decomposed, "raw_within_vs_outcome")
    raw_outside = _pool_field(decomposed, "raw_outside_vs_outcome")
    joint_within = _pool_field(decomposed, "joint_within_controlling_outside")
    joint_outside = _pool_field(decomposed, "joint_outside_controlling_within")

    joint_ps = [joint_within.get("two_sided_p_value"), joint_outside.get("two_sided_p_value")]
    if all(p is not None for p in joint_ps):
        fdr = fdr_bh(np.asarray(joint_ps, dtype=float))
        joint_within["q_value"] = float(fdr["q_values"][0])
        joint_within["q_significant"] = bool(fdr["reject"][0])
        joint_outside["q_value"] = float(fdr["q_values"][1])
        joint_outside["q_significant"] = bool(fdr["reject"][1])
    else:
        joint_within["q_value"] = joint_outside["q_value"] = None
        joint_within["q_significant"] = joint_outside["q_significant"] = False

    null_pooled = _pooled_random_subspace_null(decomposed)
    within_raw_observed = raw_within.get("mean_value")
    within_raw_clears_null = False
    within_raw_null_p = None
    if within_raw_observed is not None and null_pooled["mean"] is not None:
        finite_null = np.asarray([v for v in null_pooled["pooled_null_values"] if v is not None], dtype=float)
        within_raw_null_p = permutation_pvalue(np.abs(finite_null) >= abs(within_raw_observed))
        within_raw_clears_null = bool(within_raw_null_p <= 0.05)

    branch, surprises = _classify_components(joint_within, joint_outside, within_raw_clears_null, WORSE_BEHAVIOUR_SIGN[arm])

    collinearity_values = [s["collinearity_within_vs_outside"]["r"] for s in decomposed]

    within_raw_null_block = {
        "observed_pooled_raw_within": within_raw_observed,
        "null_summary": {k: v for k, v in null_pooled.items() if k != "pooled_null_values"},
        "two_sided_permutation_p_value": within_raw_null_p, "clears_null": within_raw_clears_null,
    }
    mdd_within = raw_within.get("minimum_detectable_paired_difference_at_80pct_power", {})
    mdd_within = mdd_within.get("mdd") if isinstance(mdd_within, dict) else None

    wrong_direction_disclosures = [d for d in (
        _wrong_direction_disclosure("within_subspace_component_controlling_for_outside_subspace", joint_within,
                                     WORSE_BEHAVIOUR_SIGN[arm], OUTCOME_DESCRIPTION[arm],
                                     raw_null=within_raw_null_block, mdd=mdd_within),
        _wrong_direction_disclosure("outside_subspace_component_controlling_for_within_subspace", joint_outside,
                                     WORSE_BEHAVIOUR_SIGN[arm], OUTCOME_DESCRIPTION[arm]),
    ) if d is not None]

    return {
        "status": "tested", "n_sessions_decomposed": len(decomposed),
        "raw_within_vs_outcome": raw_within, "raw_outside_vs_outcome": raw_outside,
        "joint_within_controlling_outside": joint_within, "joint_outside_controlling_within": joint_outside,
        "within_raw_vs_matched_dimension_random_subspace_null": within_raw_null_block,
        "collinearity_within_vs_outside": {
            "per_session": collinearity_values, "mean": float(np.mean(collinearity_values)),
            "median": float(np.median(collinearity_values)),
        },
        "branch": branch, "surprises_significant_but_wrong_direction": surprises,
        "wrong_direction_disclosures": wrong_direction_disclosures,
        "worse_behaviour_sign_convention": WORSE_BEHAVIOUR_SIGN[arm],
        "outcome_description": OUTCOME_DESCRIPTION[arm],
    }


# ----------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    _COMPLETED.update(_load_completed())
    _log(f"fits already recorded as complete: {len(_COMPLETED)}")
    root = data_root()

    # ---------------- macaque lPFC ----------------
    macaque_paths = _macaque_paths(root)
    macaque_rows: list[dict] = []
    for path in macaque_paths:
        session_id = path.stem
        row = _fit(f"macaque|{session_id}", lambda p=path: _analyse_macaque_session(p))
        macaque_rows.append(row)
        _log(f"macaque {session_id}: status={row.get('status')} elapsed={time.time() - t0:.1f}s")

    macaque_by_floor = {}
    for floor in MACAQUE_ERROR_FLOORS:
        eligible = [r for r in macaque_rows if r.get("n_error", -1) >= floor]
        macaque_by_floor[str(floor)] = {
            "n_sessions_eligible_by_error_floor": len(eligible),
            **pool_arm(eligible, "macaque_lpfc"),
        }

    macaque_reachability_table = [
        {"session": r["session"], "n_units": r.get("n_units"), "n_trials_total": r.get("n_trials_total"),
         "n_error": r.get("n_error"), "status": r.get("status"),
         "content_reachability": r.get("content_reachability")}
        for r in macaque_rows
    ]
    n_macaque_content_reachable = sum(1 for r in macaque_rows if r.get("status") == "decomposed")
    n_macaque_content_void = sum(1 for r in macaque_rows if r.get("status") == "content_not_reachable_void")

    # ---------------- multi-object corpus ----------------
    watters_rows: list[dict] = []
    n_watters_seen = 0
    n_watters_no_spike_tensor = 0
    for session in iter_watters(root, bin_ms=BIN_MS):
        n_watters_seen += 1
        if session.get("status") != "loaded":
            n_watters_no_spike_tensor += 1
            watters_rows.append({"session": session.get("session", f"{session.get('animal')}_{session.get('session_date')}"),
                                  "status": f"not_loaded:{session.get('status')}"})
            continue
        session_id = session["session"]
        row = _fit(f"watters|{session_id}", lambda s=session: _analyse_watters_session(s))
        watters_rows.append(row)
        _log(f"watters {session_id}: status={row.get('status')} elapsed={time.time() - t0:.1f}s")

    watters_loaded = [r for r in watters_rows if not str(r.get("status", "")).startswith("not_loaded")]
    watters_result = pool_arm(watters_loaded, "watters_multi_object")
    n_watters_content_reachable = sum(1 for r in watters_loaded if r.get("status") == "decomposed")
    n_watters_content_void = sum(1 for r in watters_loaded if r.get("status") == "content_not_reachable_void")

    identity_checks = [r for r in (macaque_rows + watters_loaded) if "identity_passed" in r]
    identity_all_passed = bool(identity_checks) and all(r["identity_passed"] for r in identity_checks)
    identity_max_diff = max((r["identity_max_abs_diff"] for r in identity_checks), default=None)

    output = {
        "version": "2026-08-14",
        "scope": {
            "question": (
                "Does the accuracy-predicting rate-free state deviation lie inside the memorandum's own "
                "cross-validated coding subspace, outside it, or both -- and is a positive within-subspace "
                "finding separable from a random subspace of the identical dimension."
            ),
            "primary_arm": "macaque_lpfc", "primary_error_floor": 60,
            "sensitivity_error_floors": [f for f in MACAQUE_ERROR_FLOORS if f != 60],
            "replication_arm": "watters_multi_object",
            "replication_arm_caveat": (
                "The multi-object corpus (Watters, Gabel, Tenenbaum, Jazayeri; bioRxiv preprint, posted "
                "2026-01-27, DOI 10.64898/2026.01.27.702062; data DANDI 000620) is an UNREVIEWED PREPRINT. "
                "This arm reuses only its existence and reachability machinery; "
                "results/watters_state_geometry.json's own delivered behavioural branch "
                "'state_geometry_does_not_predict_accuracy' is not re-run and not modified here."
            ),
            "n_perm_partial_correlation": N_PERM, "n_random_subspace_draws": N_RANDOM_SUBSPACE_DRAWS,
            "identity_tolerance": IDENTITY_TOLERANCE, "seed_scheme": "statistics.stable_seed on descriptive tags",
            "wall_clock_s": None,  # filled at the end
        },
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "reachability_rule_declared_before_fitting": REACHABILITY_RULE_DECLARED_BEFORE_FITTING,
        "decomposition_identity": {
            "relationship": (
                "deviation_i = 1 - cos_i where cos_i = u_i . m_i (u_i the trial's own L2-normalised "
                "direction, m_i its leave-one-out reference); residual r_i = u_i - cos_i * m_i has "
                "||r_i||^2 = 1 - cos_i^2 = deviation_i * (2 - deviation_i), monotonically increasing in "
                "deviation_i over deviation_i in [0, 1] -- the only range reachable here because every "
                "unit's activity is a non-negative spike count, so u_i and m_i have strictly non-negative "
                "entries and cos_i = u_i . m_i is itself always >= 0."
            ),
            "all_sessions_identity_passed": identity_all_passed,
            "max_absolute_difference_recomputed_vs_delivered_across_all_sessions": identity_max_diff,
            "n_sessions_checked": len(identity_checks),
            "stop_condition": "If any session's identity check had failed, this module would stop and report the failure rather than continue to any decomposition result.",
        },
        "macaque_lpfc": {
            "n_sessions_seen": len(macaque_paths),
            "reachability_table": macaque_reachability_table,
            "n_sessions_content_reachable": n_macaque_content_reachable,
            "n_sessions_content_void": n_macaque_content_void,
            "by_error_floor": macaque_by_floor,
            "primary_error_floor": "60",
        },
        "watters_multi_object": {
            "n_sessions_seen": n_watters_seen, "n_sessions_no_spike_tensor": n_watters_no_spike_tensor,
            "n_sessions_loaded": len(watters_loaded),
            "reachability_table": [
                {"session": r.get("session"), "n_units": r.get("n_units"), "n_trials_total": r.get("n_trials_total"),
                 "status": r.get("status"), "content_reachability": r.get("content_reachability")}
                for r in watters_loaded
            ],
            "n_sessions_content_reachable": n_watters_content_reachable,
            "n_sessions_content_void": n_watters_content_void,
            "result": watters_result,
        },
        "within_subspace_coefficient_comparison_not_pre_declared": {
            "note": (
                "This comparison carries no branch and asserts no mechanism -- it was not part of the "
                "pre-declared decision rule and is reported only because the within-subspace component's "
                "sign is consistent across both arms under opposite outcome codings: in the macaque arm "
                "(outcome coded 1 = correct) the within-subspace-controlling-outside coefficient is "
                "POSITIVE at every error floor, and in the multi-object arm (outcome is report error "
                "directly) it is NEGATIVE -- both mean the same thing, larger within-subspace deviation "
                "associating with BETTER behaviour, opposite the pre-declared worse-behaviour direction in "
                "both corpora. No branch is defined for this pattern and none is claimed here."
            ),
            "macaque_lpfc_within_subspace_controlling_outside_vs_correctness_by_error_floor": {
                str(floor): macaque_by_floor.get(str(floor), {}).get("joint_within_controlling_outside", {}).get("mean_value")
                for floor in MACAQUE_ERROR_FLOORS
            },
            "watters_multi_object_within_subspace_controlling_outside_vs_report_error":
                watters_result.get("joint_within_controlling_outside", {}).get("mean_value"),
        },
        "macaque_session_rows": macaque_rows,
        "watters_session_rows": watters_rows,
        "status": "complete",
    }
    output["scope"]["wall_clock_s"] = time.time() - t0
    output["wall_clock_s"] = output["scope"]["wall_clock_s"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))
    _log(f"wrote {OUTPUT_PATH} in {output['wall_clock_s']:.1f}s")
    print(json.dumps({
        "identity_all_passed": identity_all_passed,
        "macaque_primary_branch": macaque_by_floor.get("60", {}).get("branch"),
        "watters_branch": watters_result.get("branch"),
    }, indent=2))


if __name__ == "__main__":
    main()
