"""Shared time-resolved stimulation-response interface.

One estimator, two corpora: this module answers "does a candidate biomarker
move under stimulation, relative to a same-session no-stimulation
counterfactual, in a causally ordered pre/post window" for any corpus that
hands it a (trial, unit) activity array and a treatment flag -- human
intracranial field-potential power and macaque intracortical spike counts
are both just (n_trials, n_units) matrices to the functions below. Nothing
here reads a file or knows a corpus identifier; window boundaries, channel
selection, and counterfactual choice are the caller's job (driven from
results/stimulation_design_census.json, not from this module).

`rate_free_state_deviation` is moved here unchanged from
scripts/run_rate_free_state_geometry_behavior_link.py (previously its only
home) so the estimator, the reference-direction/fixed-scoring pair it is
built on for a treatment contrast, and the cluster-level pooling that turns
many sessions into one participant/animal-level number all live in the same
place and are imported, never copied, by every script that needs them.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from statistics import minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed


def rate_free_state_deviation(activity_by_unit: NDArray) -> NDArray:
    """Per trial, deviation_i = 1 - cosine(unit_vector_i, renormalised
    leave-one-out mean of every OTHER trial's own unit-normalised
    direction), from a (n_trials, n_units) per-unit activity array.

    Removes total activity by construction rather than by regression: each
    trial's vector is L2-normalised to unit length before the leave-one-out
    mean is taken, so only its DIRECTION across units enters either the
    reference or the comparison. The leave-one-out mean excludes trial i's
    own unit vector from the average it is compared against (the trial must
    not contribute to its own reference) and is not conditioned on trial
    outcome in any way.

    A trial with zero total activity across all units has no defined
    direction and gets NaN, and does not contribute to any other trial's
    leave-one-out reference either.
    """
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials = activity.shape[0]
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors, axis=0)  # per-unit sum across VALID trials only (NaN rows contribute 0)
    n_valid = int(valid.sum())

    deviation = np.full(n_trials, np.nan)
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
        cosine = float(np.dot(unit_vectors[i], loo_mean / loo_norm))
        deviation[i] = 1.0 - cosine
    return deviation


def fixed_reference_direction(activity_by_unit: NDArray) -> NDArray:
    """The renormalised mean unit direction of every trial in
    `activity_by_unit`, with NO leave-one-out exclusion -- a single FIXED
    reference built from a control-trial pool, which treated trials (never
    members of that pool) are scored against without letting them define
    any part of their own comparison point."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors[valid], axis=0)
    n_valid = int(valid.sum())
    if n_valid < 1:
        return np.full(activity.shape[1], np.nan)
    mean_dir = total / n_valid
    norm = np.linalg.norm(mean_dir)
    return mean_dir / norm if norm > 0 else np.full(activity.shape[1], np.nan)


def deviation_from_fixed_reference(activity_by_unit: NDArray, reference_direction: NDArray) -> NDArray:
    """Per trial, 1 - cosine(unit_vector_i, reference_direction): the
    counterpart to rate_free_state_deviation's leave-one-out reference, for
    trials that must never contribute to the reference they are scored
    against."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    if not np.all(np.isfinite(reference_direction)):
        return np.full(activity.shape[0], np.nan)
    deviation = np.full(activity.shape[0], np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    deviation[valid] = 1.0 - unit_vectors[valid] @ reference_direction
    return deviation


def window_treatment_effect(control_activity: NDArray, treated_activity: NDArray) -> dict:
    """One window's rate-free-deviation treatment effect: leave-one-out
    deviation among control trials only, a single fixed reference direction
    built from that same control pool, and treated trials scored against
    that fixed reference. Nothing about a treated trial's own value can
    change the reference it is compared against (the same-trial-reference
    leakage rule), and the control pool never includes a treated trial (the
    treatment-unit leakage rule, enforced by the caller passing only
    control-labelled rows into `control_activity`)."""
    control_deviation = rate_free_state_deviation(control_activity)
    reference_direction = fixed_reference_direction(control_activity)
    treated_deviation = deviation_from_fixed_reference(treated_activity, reference_direction)
    finite_ctrl, finite_treated = np.isfinite(control_deviation), np.isfinite(treated_deviation)
    if finite_ctrl.sum() < 8 or finite_treated.sum() < 4:
        return {"status": "too_few_trials", "n_control_trials": int(finite_ctrl.sum()),
                "n_treated_trials": int(finite_treated.sum())}
    spontaneous_sd = float(np.nanstd(control_deviation[finite_ctrl], ddof=1)) if finite_ctrl.sum() >= 2 else float("nan")
    displacement = float(np.nanmean(treated_deviation[finite_treated]) - np.nanmean(control_deviation[finite_ctrl]))
    return {
        "status": "computed",
        "n_control_trials": int(finite_ctrl.sum()), "n_treated_trials": int(finite_treated.sum()),
        "displacement": displacement, "spontaneous_control_sd": spontaneous_sd,
        "normalised_displacement": (displacement / spontaneous_sd) if spontaneous_sd and spontaneous_sd > 0 else None,
    }


def nuisance_treatment_effect(control_totals: NDArray, treated_totals: NDArray) -> dict:
    """Plain group-mean difference (treated - control) of a scalar-per-trial
    nuisance quantity (total band power or total spike count) -- reported as
    an explicit nuisance control, never scored as a candidate biomarker."""
    control_totals, treated_totals = np.asarray(control_totals, dtype=float), np.asarray(treated_totals, dtype=float)
    finite_ctrl, finite_treated = np.isfinite(control_totals), np.isfinite(treated_totals)
    if finite_ctrl.sum() < 8 or finite_treated.sum() < 4:
        return {"status": "too_few_trials", "n_control_trials": int(finite_ctrl.sum()),
                "n_treated_trials": int(finite_treated.sum())}
    control_sd = float(np.nanstd(control_totals[finite_ctrl], ddof=1)) if finite_ctrl.sum() >= 2 else float("nan")
    change = float(np.nanmean(treated_totals[finite_treated]) - np.nanmean(control_totals[finite_ctrl]))
    return {
        "status": "computed",
        "n_control_trials": int(finite_ctrl.sum()), "n_treated_trials": int(finite_treated.sum()),
        "change": change, "control_sd": control_sd,
        "normalised_change": (change / control_sd) if control_sd and control_sd > 0 else None,
    }


def cluster_bootstrap_pooled_effect(session_values: NDArray, cluster_ids: list, seed_tag: str,
                                     n_perm: int = 10000, n_boot: int = 5000,
                                     alternative: str = "two-sided", alpha: float = 0.05,
                                     power: float = 0.80) -> dict:
    """Pool one scalar per session into a cluster-level test: each cluster's
    (participant's, or animal's) own sessions are first collapsed to their
    unweighted mean, then the collapsed cluster-level values are tested
    against zero with the paired sign-flip test -- a session-count formula
    and an ICC design effect are both refused by construction, because the
    permutation and the bootstrap both resample clusters, never sessions or
    trials. Same algorithm as
    scripts/run_human_stimulation_component_response.py's
    subject_clustered_mean_test (not imported from it, to keep this
    interface free of that script's data-loading dependencies); reimplemented
    here once as the shared primitive so both corpora's clustering --
    participant for the human arm, animal for the macaque arm -- go through
    identical code."""
    session_values = np.asarray(session_values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(session_values)
    session_values, cluster_ids = session_values[finite], cluster_ids[finite]
    unique_clusters = sorted(set(cluster_ids.tolist()))
    if len(unique_clusters) < 2:
        return {"status": "not_computable", "n_sessions": int(finite.sum()), "n_clusters": len(unique_clusters)}
    cluster_values = np.array([session_values[cluster_ids == c].mean() for c in unique_clusters])
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|{tuple(unique_clusters)}|{int(finite.sum())}"))
    test = paired_sign_flip_test(cluster_values, np.zeros_like(cluster_values), n_perm=n_perm,
                                  alternative=alternative, n_boot=n_boot, rng=rng)
    mdd = minimum_detectable_paired_difference(cluster_values, alpha=alpha, power=power)
    return {
        "status": "computed", "n_sessions": int(finite.sum()), "n_clusters": len(unique_clusters),
        "mean_value": test["mean_diff"], "p_value": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "mdd": mdd,
    }
