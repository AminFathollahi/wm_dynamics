"""run_deviation_serial_dependence_and_temporal_locus.py -- two positive-
identity tests of the rate-free direction-deviation observable
(rate_free_state_deviation), the population-state component that predicts
trial outcome in this project after every other identity has come back
negative: it is not the dominant rate mode, not the memorandum's own coding
subspace, and not an artifact of total spike count.

BLOCK A asks whether the deviation is inter-trial interference: a trial's
direction pulled toward the trial immediately before it. The population
state is already known to drift slowly across a session, and a slow drift
alone produces a decaying lag profile with zero genuine trial-to-trial
interference, so every adjacency statistic here is computed only after
removing a smooth session-time trend from the per-trial direction vectors,
at two independently declared detrending windows, against a within-session
trial-order shuffle recomputed end to end (detrending included) at least
1000 times per session.

BLOCK B asks where in the delay epoch the deviation lives, by splitting the
delay into halves (primary) and thirds (sensitivity) and recomputing the
deviation, its own orthogonality gate against total spike count, and its
behavioural association separately in each sub-window. A sub-window that
fails its own gate is reported void, never coerced into a number.

Scope is set by an existing precondition, not chosen here: only the
single-item macaque lateral prefrontal cortex corpus (Panichello et al.
2024) and the multi-object macaque corpus (Watters, Gabel, Tenenbaum and
Jazayeri; DANDI 000620) have a rate-free deviation that passes its own
orthogonality gate. The mouse and both human corpora fail that gate and are
excluded on that measured precondition, not left pending.

SIGN CONVENTION. The single-item corpus's behavioural variable is trial
correctness (higher is better); the multi-object corpus's is continuous
report error (higher is worse). Every behavioural correlation reported here
is expressed against WORSE behaviour: the multi-object corpus needs no
flip, the single-item corpus's sign is negated once, at the point each
correlation result is packaged, never inside the correlation call itself.

No estimator is forked. rate_free_state_deviation, reproduction_gate,
_observable_arrays, _session_observable_arm, _pool_cell,
partial_correlation_permutation_test, slope_across_sessions_test,
minimum_detectable_paired_difference, permutation_pvalue, stable_seed,
_reachable_sessions, _counts_from_spikes, _panichello_directory,
_behaviour_observables, iter_watters, usable_label and data_root are every
one imported unchanged from where this project already defines them. The
only new functions this module introduces are the detrending, lag-cosine,
shuffle-null, neighbour-excluding reference and sub-window machinery the two
blocks need and nothing already in the project provides.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_behavior_amplitude_rate_controls import _reachable_sessions  # noqa: E402
from run_dissociation_cross_preparation_test import (  # noqa: E402
    MIN_TRIALS_WITH_DEFINED_DIRECTION, reproduction_gate,
)
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    _observable_arrays, _pool_cell, _session_observable_arm,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import _counts_from_spikes, _panichello_directory  # noqa: E402
from run_state_content_link import usable_label  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, permutation_pvalue, stable_seed,
)
from state_persistence import slope_across_sessions_test  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "deviation_serial_dependence_and_temporal_locus.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "deviation_serial_dependence_and_temporal_locus_checkpoint.json"
ANALYSIS_VERSION = "2026-08-15"

N_PERM = 10000
REPRODUCTION_TOLERANCE = 1e-6

# Reference values already delivered elsewhere, read live from results/rate_free_state_geometry_behavior_link.json
# and results/dissociation_replication_and_counting_noise.json (block_a.results.single_and_multi_unit.pooled.
# deviation.within_item_count_level) and confirmed to full precision before this module was written.
MACAQUE_DEVIATION_GATE_R = 0.024701485905536346
MACAQUE_DEVIATION_GATE_P = 0.7384261573842615
MACAQUE_DEVIATION_RAW_R = -0.09742768955602038  # against correctness; +0.09742768955602038 against worse behaviour
MACAQUE_DEVIATION_RAW_P = 0.0034996500349965005
WATTERS_DEVIATION_GATE_R = 0.004261530908810586
WATTERS_DEVIATION_GATE_P = 0.8825117488251175
WATTERS_DEVIATION_RAW_R = 0.019673378706534905  # already against report error, i.e. against worse behaviour
WATTERS_DEVIATION_RAW_P = 0.00029997000299970003

# Sign multiplier applied once, at the point each behavioural-correlation result is packaged, so every
# reported coefficient in this artifact is against WORSE behaviour regardless of corpus.
SIGN_TO_WORSE_BEHAVIOUR = {
    "panichello_2024_macaque_lPFC_single_item": -1.0,   # native outcome is is_corr (1=correct)
    "watters_2026_macaque_multi_object": 1.0,           # native outcome is report error already
}

# Block A -- declared before any fit runs.
DETREND_WINDOWS_TRIALS = (51, 101)  # centred moving-average width in trials; both odd, both >> the 15-trial lag range
N_SHUFFLES_PER_SESSION = 1000
LAG_RANGE = tuple(range(1, 16))
BACKGROUND_LAGS = tuple(range(10, 16))
MIN_TRIALS_FOR_LAG_PROFILE = 40  # gives >=25 trial pairs at the widest lag (15) even before any exclusion
SHARP_TEST_MIN_CLASSES = 3
SHARP_TEST_MIN_PER_CLASS = 4
SHARP_TEST_MIN_QUALIFYING_TRIALS = 8
CONTENT_LABEL_K_CLASSES = 8  # matches the discretisation results/dissociation_replication_and_counting_noise.json's
                             # content_fractional_rank_third_point already uses for this corpus

# Block B -- declared before any fit runs.
SUB_WINDOW_SPLITS = {"halves": 2, "thirds": 3}
PRIMARY_SPLIT = "halves"

BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per corpus, at the declared primary detrending window (the first of DETREND_WINDOWS_TRIALS): (1) the "
    "pooled detrended adjacency statistic (lag-1 mean cosine minus the mean of lags 10-15, both after removing "
    "a centred moving-average session-time trend) is compared against a within-session trial-order shuffle null "
    "recomputed end to end (detrending included) at least 1000 times per session, pooled by averaging each "
    "shuffle draw's session-level statistic across sessions and comparing the real pooled value against that "
    "null distribution with a two-sided empirical p-value; 'adjacency present' means that p is <= 0.05. (2) "
    "'behaviour link survives' means the deviation-to-behaviour joint partial correlation, controlling the "
    "trial's own detrended lag-1 alignment together with total spike count and trial index, is significant at "
    "the two-sided 0.05 level with the same sign as the raw deviation-to-behaviour correlation. Four cells, all "
    "named:\n"
    "  - adjacency absent, behaviour link survives -> "
    "'accuracy_predicting_component_is_not_interference_from_the_preceding_trial', with the minimum adjacency "
    "effect the shuffle test could detect at 80% power reported beside it.\n"
    "  - adjacency absent, behaviour link does not survive -> "
    "'partial_control_removes_variance_the_adjacency_statistic_does_not_explain', reported as a defect in this "
    "design, not a finding.\n"
    "  - adjacency present, behaviour link does not survive -> "
    "'accuracy_predicting_component_is_interference_from_the_preceding_trial', with the raw and partial "
    "coefficients reported side by side.\n"
    "  - adjacency present, behaviour link survives -> "
    "'interference_from_the_preceding_trial_is_present_and_separable_from_the_accuracy_predicting_component'.\n"
    "The adjacency verdict is also computed at the second declared window; if it disagrees with the primary "
    "window's verdict, both are reported and the disagreement is stated, not resolved by picking one."
)

BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "The delay epoch is split into halves (primary) and thirds (sensitivity). In each sub-window, the "
    "per-unit total counts, the deviation, its own orthogonality gate against total spike count within that "
    "sub-window, and its raw association with behaviour are recomputed from scratch. A sub-window whose own "
    "gate is significant at the two-sided 0.05 level is VOID and excluded from every comparison; it is never "
    "compared against a surviving sub-window.\n"
    "  - every sub-window void -> 'temporal_locus_unreachable_at_this_count_per_window', with the counts and "
    "gates that made it so.\n"
    "  - behaviour link present (raw association significant) in every surviving sub-window, and the direct "
    "paired difference between the earliest and latest surviving sub-window is not significant -> "
    "'accuracy_predicting_component_is_present_throughout_the_delay'.\n"
    "  - behaviour link present in every surviving sub-window, and the direct paired difference is significant "
    "and larger in the earliest surviving sub-window -> 'accuracy_predicting_component_is_strongest_at_delay_onset'.\n"
    "  - behaviour link present in every surviving sub-window, and the direct paired difference is significant "
    "and larger in the latest surviving sub-window -> 'accuracy_predicting_component_grows_across_the_delay'.\n"
    "  - behaviour link absent in every surviving sub-window while present over the whole epoch (per the "
    "reproduction gate's own raw correlation) -> 'accuracy_predicting_component_requires_the_full_epoch_to_be_"
    "detected', reported as a statement about detection, not about the delay's structure.\n"
    "An ordering claim requires the earliest-vs-latest difference to be tested directly, paired within session, "
    "never inferred from two separate significance verdicts. Any other reachable combination (behaviour link "
    "present in some but not all surviving sub-windows) is not named by this rule and is reported as "
    "'block_b_outcome_not_covered_by_the_pre_declared_rule' with every sub-window's own numbers, not forced "
    "onto a listed branch. The halves split alone decides the branch; the thirds split is reported beside it as "
    "a sensitivity check and does not override the halves verdict."
)

CONTENT_SPECIFIC_SERIAL_PULL_OPERATIONALISATION = (
    "A disclosed reading of an underspecified secondary instruction. For trials t where trial t's own "
    "memorandum class differs from trial t-1's (so the comparison cannot be explained by t simply sharing its "
    "own class with its predecessor), pull_previous(t) is the cosine of trial t's direction to the leave-in "
    "mean direction of every OTHER trial carrying trial t-1's class, and pull_other_classes(t) is the mean of "
    "that same cosine to every class that is neither t's own class nor t-1's. The per-trial difference "
    "pull_previous(t) - pull_other_classes(t) is pooled per session then across sessions by the paired "
    "sign-flip test; a reliably positive pooled difference is content-specific serial pull toward what was "
    "shown on the immediately preceding trial, over and above ordinary attraction to trial t's own class "
    "(excluded from the comparison set by construction)."
)


# =======================================================================================================
# Checkpointing (fit-level; temp file + os.replace; completion flag written only after the fit returns)
# =======================================================================================================

_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    try:
        entries = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: {**entry, "value": restore_checkpoint(entry["value"])} for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("complete") is True}


def _fit(key: str, compute) -> dict:
    global _FITS_SERVED_FROM_CHECKPOINT, _FITS_COMPUTED_HERE
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        _FITS_SERVED_FROM_CHECKPOINT += 1
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    _FITS_COMPUTED_HERE += 1
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(checkpoint_safe(_COMPLETED_FITS), allow_nan=False, default=float))
    os.replace(scratch, CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


def _sign_to_worse_behaviour(result: dict, sign: float) -> dict:
    """Negates a partial_correlation_permutation_test result's 'r' field (and a pooled
    slope_across_sessions_test result's 'mean_value'/CI/direction fields) by the per-corpus sign
    multiplier, applied once here at the packaging boundary. Never touches the estimator call itself --
    p-values, n and every other field pass through unchanged, since a sign flip does not change how
    extreme a value is."""
    if sign == 1.0 or not isinstance(result, dict):
        return result
    out = dict(result)
    if "r" in out and out["r"] is not None:
        out["r"] = -out["r"]
    if "r_analytic" in out and out["r_analytic"] is not None:
        out["r_analytic"] = -out["r_analytic"]
    if "mean_value" in out and out["mean_value"] is not None:
        out["mean_value"] = -out["mean_value"]
    if out.get("ci_lower") is not None and out.get("ci_upper") is not None:
        out["ci_lower"], out["ci_upper"] = -out["ci_upper"], -out["ci_lower"]
    if "significant_negative" in out and "significant_positive" in out:
        out["significant_negative"], out["significant_positive"] = out["significant_positive"], out["significant_negative"]
    return out


# =======================================================================================================
# Reproduction gate
# =======================================================================================================

def _close(observed: float | None, expected: float, tol: float = REPRODUCTION_TOLERANCE) -> bool:
    return observed is not None and abs(observed - expected) <= tol


def _reproduce_watters_deviation_gate(watters_loaded: list[dict]) -> dict:
    """Reproduces results/dissociation_replication_and_counting_noise.json's within-item-count-level
    deviation cell (single_and_multi_unit tier, pooled group) using the identical imported functions that
    artifact used, restricted to the 'deviation' observable at that one tier -- the amplitude observable
    and the other two tiers are not needed here and are not recomputed, but the underlying per-session
    arrays this saves are the exact ones Block A and Block B reuse below, so nothing is loaded twice."""
    rows = []
    arrays_by_session: dict[str, dict] = {}
    for session in watters_loaded:
        counts = session["counts"]
        arrays, _excluded, usable = _observable_arrays(counts, session)
        if arrays is None:
            rows.append({"by_tier": {}})
            continue
        arrays_by_session[session["session"]] = {"arrays": arrays, "usable": usable, "session": session}
        deviation_arm = _session_observable_arm(
            arrays, "deviation", f"deviation_serial_dependence_and_temporal_locus|reproduction|{session['session']}")
        rows.append({"by_tier": {"single_and_multi_unit": {"status": "computed", "deviation": deviation_arm}}})
    gate = _pool_cell(rows, "single_and_multi_unit", "deviation", "within_load", "orthogonality_gate_vs_spike_count")
    raw = _pool_cell(rows, "single_and_multi_unit", "deviation", "within_load", "raw_vs_report_error")
    return {"gate": gate, "raw": raw, "n_sessions": len(arrays_by_session), "arrays_by_session": arrays_by_session}


def full_reproduction_gate(root: Path, watters_loaded: list[dict]) -> dict:
    macaque = reproduction_gate(root)
    watters = _reproduce_watters_deviation_gate(watters_loaded)
    checks = {
        "macaque_deviation_gate_r": macaque["checks"]["deviation_orthogonality_gate_r"],
        "macaque_deviation_gate_p": macaque["checks"]["deviation_orthogonality_gate_p"],
        "macaque_deviation_raw_r": macaque["checks"]["deviation_raw_r"],
        "macaque_deviation_raw_p": macaque["checks"]["deviation_raw_p"],
        "watters_deviation_gate_r": _close(watters["gate"].get("mean_value"), WATTERS_DEVIATION_GATE_R),
        "watters_deviation_gate_p": _close(watters["gate"].get("p_value"), WATTERS_DEVIATION_GATE_P),
        "watters_deviation_raw_r": _close(watters["raw"].get("mean_value"), WATTERS_DEVIATION_RAW_R),
        "watters_deviation_raw_p": _close(watters["raw"].get("p_value"), WATTERS_DEVIATION_RAW_P),
    }
    return {
        "status": "reproduced_exactly" if all(checks.values()) else "not_reproduced",
        "tolerance": REPRODUCTION_TOLERANCE,
        "checks": checks,
        "macaque": {"status": macaque["status"], "checks": macaque["checks"]},
        "watters": {"gate": watters["gate"], "raw": watters["raw"], "n_sessions": watters["n_sessions"]},
        "reference_values": {
            "macaque_deviation_gate_r": MACAQUE_DEVIATION_GATE_R, "macaque_deviation_gate_p": MACAQUE_DEVIATION_GATE_P,
            "macaque_deviation_raw_r": MACAQUE_DEVIATION_RAW_R, "macaque_deviation_raw_p": MACAQUE_DEVIATION_RAW_P,
            "watters_deviation_gate_r": WATTERS_DEVIATION_GATE_R, "watters_deviation_gate_p": WATTERS_DEVIATION_GATE_P,
            "watters_deviation_raw_r": WATTERS_DEVIATION_RAW_R, "watters_deviation_raw_p": WATTERS_DEVIATION_RAW_P,
        },
    }, watters["arrays_by_session"]


# =======================================================================================================
# Session bundles -- the (trials, units[, bins]) arrays both blocks share, in acquisition order
# =======================================================================================================

def _macaque_session_bundle(path: Path) -> dict | None:
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1).astype(float)
    counts_all = _counts_from_spikes(spikes, time_ms)
    if counts_all.shape[0] < 16:
        return None
    activity_by_unit = counts_all.sum(axis=2)
    deviation = rate_free_state_deviation(activity_by_unit)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    return {
        "session": path.stem, "corpus": "panichello_2024_macaque_lPFC_single_item",
        "activity_by_unit": activity_by_unit[finite], "deviation": deviation[finite],
        "outcome_raw": is_corr[finite].astype(float), "spike_count": activity_by_unit.sum(axis=1)[finite],
        "trial_index": np.arange(counts_all.shape[0], dtype=float)[finite],
        "memorandum_label": cue_idx[finite], "counts": counts_all[finite],
        "n_trials_total": int(counts_all.shape[0]), "n_trials_with_defined_direction": int(finite.sum()),
    }


def _watters_session_bundle(session: dict, arrays: dict, usable: np.ndarray) -> dict:
    counts = session["counts"]
    activity_by_unit = counts.sum(axis=2)[usable]
    theta = np.mod(np.asarray(session["cued_theta"], dtype=float)[usable], 2.0 * np.pi)
    label = (np.floor(theta / (2.0 * np.pi / CONTENT_LABEL_K_CLASSES)).astype(int)) % CONTENT_LABEL_K_CLASSES
    return {
        "session": session["session"], "corpus": "watters_2026_macaque_multi_object",
        "activity_by_unit": activity_by_unit, "deviation": arrays["deviation"],
        "outcome_raw": arrays["report_error"], "spike_count": arrays["spike_count"],
        "trial_index": arrays["trial_index"], "memorandum_label": label.astype(float), "counts": counts[usable],
        "n_trials_total": int(counts.shape[0]), "n_trials_with_defined_direction": int(usable.sum()),
    }


# =======================================================================================================
# BLOCK A -- serial dependence
# =======================================================================================================

def unit_direction_vectors(activity_by_unit: np.ndarray) -> np.ndarray:
    """The same per-trial L2-normalised direction vectors rate_free_state_deviation computes internally,
    exposed here because that estimator returns only the scalar deviation. Every trial handed in already
    has a defined direction (pre-filtered by the caller), so this never produces a NaN row in practice;
    the guard is kept for safety, not because it is expected to fire."""
    activity = np.asarray(activity_by_unit, dtype=float)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)


def _detrend(vectors: np.ndarray, window: int) -> np.ndarray:
    """Centred moving-average detrend per unit-vector component. Edge trials use a truncated window
    (scipy's 'nearest' boundary mode), a disclosed engineering choice: it does not invent data beyond the
    session's own ends, at the cost of a slightly under-smoothed trend estimate at the two edges."""
    trend = uniform_filter1d(vectors, size=window, axis=0, mode="nearest")
    return vectors - trend


def _cosine_at_lag(vectors: np.ndarray, lag: int) -> np.ndarray:
    if vectors.shape[0] <= lag:
        return np.zeros(0)
    a, b = vectors[:-lag], vectors[lag:]
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (norm_a > 1e-12) & (norm_b > 1e-12)
    out = np.full(a.shape[0], np.nan)
    dots = np.einsum("ij,ij->i", a[valid], b[valid])
    out[valid] = dots / (norm_a[valid] * norm_b[valid])
    return out


def _lag_profile(vectors: np.ndarray, lags: tuple[int, ...]) -> dict[str, float | None]:
    out = {}
    for lag in lags:
        vals = _cosine_at_lag(vectors, lag)
        finite = vals[np.isfinite(vals)]
        out[str(lag)] = float(np.mean(finite)) if finite.size else None
    return out


def _adjacency_statistic(vectors: np.ndarray) -> float | None:
    lag1 = _cosine_at_lag(vectors, 1)
    lag1_finite = lag1[np.isfinite(lag1)]
    if lag1_finite.size == 0:
        return None
    background = np.concatenate([_cosine_at_lag(vectors, lag) for lag in BACKGROUND_LAGS])
    background_finite = background[np.isfinite(background)]
    if background_finite.size == 0:
        return None
    return float(np.mean(lag1_finite) - np.mean(background_finite))


def rate_free_state_deviation_neighbour_excluded(activity_by_unit: np.ndarray) -> np.ndarray:
    """A variant reference set for rate_free_state_deviation's own leave-one-out construction: excludes
    trial i AND its immediate neighbours (i-1, i+1) from the reference, so the reference and the
    lag-1 adjacency term share no trial in common by construction. Reimplements only the reference set;
    the cosine-deviation formula itself is identical to the delivered estimator's."""
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials = activity.shape[0]
    vectors = unit_direction_vectors(activity)
    valid = ~np.isnan(vectors).any(axis=1)
    total = np.nansum(vectors, axis=0)
    n_valid_total = int(valid.sum())
    deviation = np.full(n_trials, np.nan)
    for i in range(n_trials):
        if not valid[i]:
            continue
        neighbours = [j for j in (i - 1, i, i + 1) if 0 <= j < n_trials and valid[j]]
        excluded_sum = np.sum(vectors[neighbours], axis=0)
        n_other = n_valid_total - len(neighbours)
        if n_other < 1:
            continue
        loo_mean = (total - excluded_sum) / n_other
        loo_norm = np.linalg.norm(loo_mean)
        if loo_norm == 0.0:
            continue
        deviation[i] = 1.0 - float(np.dot(vectors[i], loo_mean / loo_norm))
    return deviation


def _background_deviation_check(bundle: dict) -> dict:
    delivered = bundle["deviation"]
    alt = rate_free_state_deviation_neighbour_excluded(bundle["activity_by_unit"])
    both_finite = np.isfinite(delivered) & np.isfinite(alt)
    if int(both_finite.sum()) < 3:
        return {"status": "too_few_trials"}
    r = float(np.corrcoef(delivered[both_finite], alt[both_finite])[0, 1])
    return {
        "status": "computed", "n_trials": int(both_finite.sum()),
        "correlation_with_delivered_deviation": r,
        "median_absolute_difference": float(np.median(np.abs(delivered[both_finite] - alt[both_finite]))),
    }


def _adjacency_shuffle_null(vectors: np.ndarray, window: int, n_shuffles: int, seed_tag: str) -> list[float | None]:
    n = vectors.shape[0]
    draws = []
    for d in range(n_shuffles):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|shuffle{d}"))
        shuffled = vectors[rng.permutation(n)]
        draws.append(_adjacency_statistic(_detrend(shuffled, window)))
    return draws


def _session_block_a_geometry(bundle: dict, seed_prefix: str) -> dict:
    vectors = unit_direction_vectors(bundle["activity_by_unit"])
    n = vectors.shape[0]
    if n < MIN_TRIALS_FOR_LAG_PROFILE:
        return {"status": "too_few_trials_for_lag_profile", "n_trials": n}
    windows: dict[str, dict] = {}
    for window in DETREND_WINDOWS_TRIALS:
        detrended = _detrend(vectors, window)
        observed = _adjacency_statistic(detrended)
        null_draws = _adjacency_shuffle_null(vectors, window, N_SHUFFLES_PER_SESSION, f"{seed_prefix}|w{window}")
        null_finite = [d for d in null_draws if d is not None]
        windows[str(window)] = {
            "lag_profile_detrended": _lag_profile(detrended, LAG_RANGE),
            "adjacency_statistic_observed": observed,
            "n_shuffles": N_SHUFFLES_PER_SESSION, "n_shuffle_draws_usable": len(null_finite),
            "shuffle_null_draws": null_finite,
        }
    return {"status": "computed", "n_trials": n, "lag_profile_raw": _lag_profile(vectors, LAG_RANGE), "windows": windows}


def _pool_adjacency(sessions_geometry: list[dict], window: int) -> dict:
    """Pools the real per-session adjacency statistic and, separately, the per-shuffle-draw pooled null
    (same shuffle-index-averaged-across-sessions construction results/dissociation_replication_and_
    counting_noise.json's Poisson-surrogate gate test already uses), then a two-sided empirical p of the
    real pooled value against that null distribution via permutation_pvalue, reused unchanged."""
    key = str(window)
    observed_values = [g["windows"][key]["adjacency_statistic_observed"] for g in sessions_geometry
                        if g.get("status") == "computed" and g["windows"][key]["adjacency_statistic_observed"] is not None]
    real_pooled = slope_across_sessions_test(observed_values, alternative="two-sided") if observed_values else \
        {"status": "not_computed"}
    real_mdd = minimum_detectable_paired_difference(observed_values) if len(observed_values) >= 2 else \
        {"status": "not_computable", "n": len(observed_values)}

    fully_drawn = [g["windows"][key]["shuffle_null_draws"] for g in sessions_geometry
                   if g.get("status") == "computed" and len(g["windows"][key]["shuffle_null_draws"]) == N_SHUFFLES_PER_SESSION]
    pooled_null = [float(np.mean([draws[d] for draws in fully_drawn])) for d in range(N_SHUFFLES_PER_SESSION)] \
        if fully_drawn else []

    real_mean = real_pooled.get("mean_value") if real_pooled.get("status") == "tested" else None
    if real_mean is None or not pooled_null:
        return {
            "n_sessions_contributing_real": len(observed_values), "n_sessions_contributing_null": len(fully_drawn),
            "real_pooled": real_pooled, "real_minimum_detectable_difference_80pct_power": real_mdd,
            "shuffle_null_distribution": {"status": "not_computable"},
            "significant": None, "two_sided_empirical_p_value": None,
        }
    null_arr = np.asarray(pooled_null, dtype=float)
    p_value = float(permutation_pvalue(np.abs(null_arr) >= abs(real_mean)))
    return {
        "n_sessions_contributing_real": len(observed_values), "n_sessions_contributing_null": len(fully_drawn),
        "real_pooled": real_pooled, "real_minimum_detectable_difference_80pct_power": real_mdd,
        "shuffle_null_distribution": {
            "status": "computed", "n_draws": len(pooled_null), "mean": float(np.mean(null_arr)),
            "sd": float(np.std(null_arr, ddof=1)),
        },
        "real_pooled_mean_value": real_mean,
        "two_sided_empirical_p_value": p_value,
        "significant": bool(p_value <= 0.05),
    }


def _decisive_partial(bundle: dict, window: int, sign: float, seed_prefix: str) -> dict:
    vectors = unit_direction_vectors(bundle["activity_by_unit"])
    n = vectors.shape[0]
    if n < MIN_TRIALS_FOR_LAG_PROFILE:
        return {"status": "too_few_trials_for_lag_profile"}
    detrended = _detrend(vectors, window)
    lag1_align = _cosine_at_lag(detrended, 1)  # index i pairs trial i+1 against trial i
    idx = np.arange(1, n)
    valid = np.isfinite(lag1_align)
    idx, lag1_align = idx[valid], lag1_align[valid]
    if len(idx) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_lag1_alignment", "n_trials": len(idx)}
    outcome, deviation = bundle["outcome_raw"][idx], bundle["deviation"][idx]
    spike_count, trial_index = bundle["spike_count"][idx], bundle["trial_index"][idx]
    tag = f"{seed_prefix}|decisive_partial|w{window}"
    raw = partial_correlation_permutation_test(
        outcome, deviation, [], N_PERM, np.random.default_rng(stable_seed(f"{tag}|raw")))
    partial_lag1 = partial_correlation_permutation_test(
        outcome, deviation, [lag1_align], N_PERM, np.random.default_rng(stable_seed(f"{tag}|ctrl_lag1")))
    joint = partial_correlation_permutation_test(
        outcome, deviation, [lag1_align, spike_count, trial_index], N_PERM,
        np.random.default_rng(stable_seed(f"{tag}|ctrl_joint")))
    return {
        "status": "computed", "n_trials": len(idx),
        "raw": _sign_to_worse_behaviour(raw, sign),
        "partial_controlling_lag1_alignment": _sign_to_worse_behaviour(partial_lag1, sign),
        "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index": _sign_to_worse_behaviour(joint, sign),
    }


def _pool_decisive_partial(per_session: list[dict], key: str) -> dict:
    values = [s[key]["r"] for s in per_session if s.get("status") == "computed" and s[key].get("status") == "computed"]
    pooled = slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}
    if len(values) >= 2:
        pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    return pooled


def _content_specific_serial_pull(bundle: dict) -> dict:
    vectors = unit_direction_vectors(bundle["activity_by_unit"])
    label = bundle["memorandum_label"]
    n = vectors.shape[0]
    finite_label = np.isfinite(label)
    if not finite_label.any():
        return {"status": "no_finite_memorandum_label"}
    ok, reason, _mask = usable_label(
        label[finite_label], min_classes=SHARP_TEST_MIN_CLASSES, min_per_class=SHARP_TEST_MIN_PER_CLASS)
    if not ok:
        return {"status": "no_usable_memorandum_label", "reason": reason}
    classes = np.unique(label[finite_label])
    class_members = {c: np.flatnonzero((label == c) & finite_label) for c in classes}
    class_mean = {c: vectors[idx].mean(axis=0) for c, idx in class_members.items() if len(idx) >= SHARP_TEST_MIN_PER_CLASS}
    eligible_classes = set(class_mean)
    if len(eligible_classes) < SHARP_TEST_MIN_CLASSES:
        return {"status": "fewer_than_minimum_eligible_memorandum_classes", "n_eligible_classes": len(eligible_classes)}

    diffs = []
    for t in range(1, n):
        if not (finite_label[t] and finite_label[t - 1]):
            continue
        own, prev = label[t], label[t - 1]
        if own == prev or prev not in eligible_classes:
            continue
        other_classes = [c for c in eligible_classes if c != own and c != prev]
        if not other_classes:
            continue
        prev_mean = class_mean[prev]
        prev_norm = np.linalg.norm(prev_mean)
        if prev_norm == 0.0:
            continue
        pull_prev = float(np.dot(vectors[t], prev_mean / prev_norm))
        other_pulls = []
        for c in other_classes:
            m = class_mean[c]
            mn = np.linalg.norm(m)
            if mn > 0.0:
                other_pulls.append(float(np.dot(vectors[t], m / mn)))
        if not other_pulls:
            continue
        diffs.append(pull_prev - float(np.mean(other_pulls)))

    if len(diffs) < SHARP_TEST_MIN_QUALIFYING_TRIALS:
        return {"status": "too_few_qualifying_trials", "n_qualifying_trials": len(diffs)}
    return {"status": "computed", "n_qualifying_trials": len(diffs), "mean_pull_difference": float(np.mean(diffs))}


def _block_a_branch(adjacency_significant: bool | None, behaviour_survives: bool | None) -> str:
    """Implements BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING's four named cells; every combination of
    the two booleans is named, so there is no off-list outcome for this classifier by construction."""
    if adjacency_significant is None or behaviour_survives is None:
        return "not_computable"
    if not adjacency_significant and behaviour_survives:
        return "accuracy_predicting_component_is_not_interference_from_the_preceding_trial"
    if not adjacency_significant and not behaviour_survives:
        return "partial_control_removes_variance_the_adjacency_statistic_does_not_explain"
    if adjacency_significant and not behaviour_survives:
        return "accuracy_predicting_component_is_interference_from_the_preceding_trial"
    return "interference_from_the_preceding_trial_is_present_and_separable_from_the_accuracy_predicting_component"


def run_block_a(bundles: list[dict], sign: float, corpus_key: str, seed_prefix: str) -> dict:
    geometry = []
    decisive: dict[int, list[dict]] = {w: [] for w in DETREND_WINDOWS_TRIALS}
    content_pull_values = []
    background_checks = []
    for bundle in bundles:
        tag = f"{seed_prefix}|{bundle['session']}"
        g = _fit(f"geometry|{corpus_key}|{bundle['session']}",
                  lambda b=bundle, t=tag: _session_block_a_geometry(b, t))
        geometry.append(g)
        for window in DETREND_WINDOWS_TRIALS:
            dp = _fit(f"decisive_partial|{corpus_key}|{bundle['session']}|w{window}",
                       lambda b=bundle, w=window, t=tag: _decisive_partial(b, w, sign, t))
            decisive[window].append(dp)
        background_checks.append(_fit(f"background|{corpus_key}|{bundle['session']}",
                                        lambda b=bundle: _background_deviation_check(b)))
        pull = _fit(f"content_pull|{corpus_key}|{bundle['session']}",
                     lambda b=bundle: _content_specific_serial_pull(b))
        if pull.get("status") == "computed":
            content_pull_values.append(pull["mean_pull_difference"])

    adjacency_by_window = {w: _pool_adjacency(geometry, w) for w in DETREND_WINDOWS_TRIALS}
    decisive_pooled = {
        str(w): {
            "raw": _pool_decisive_partial(decisive[w], "raw"),
            "partial_controlling_lag1_alignment": _pool_decisive_partial(decisive[w], "partial_controlling_lag1_alignment"),
            "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index": _pool_decisive_partial(
                decisive[w], "joint_partial_controlling_lag1_alignment_spike_count_and_trial_index"),
        }
        for w in DETREND_WINDOWS_TRIALS
    }

    primary_window = DETREND_WINDOWS_TRIALS[0]
    branch_by_window = {}
    for window in DETREND_WINDOWS_TRIALS:
        adj = adjacency_by_window[window]
        adj_sig = adj.get("significant")
        joint = decisive_pooled[str(window)]["joint_partial_controlling_lag1_alignment_spike_count_and_trial_index"]
        raw = decisive_pooled[str(window)]["raw"]
        behaviour_survives = None
        if joint.get("status") == "tested" and raw.get("status") == "tested":
            behaviour_survives = bool(joint["significant"] and (joint["mean_value"] > 0) == (raw["mean_value"] > 0))
        branch_by_window[str(window)] = _block_a_branch(adj_sig, behaviour_survives)

    primary_branch = branch_by_window[str(primary_window)]
    windows_agree = len(set(branch_by_window.values())) == 1

    content_pull_pooled = slope_across_sessions_test(content_pull_values, alternative="two-sided") \
        if content_pull_values else {"status": "not_computed"}
    background_r_values = [c["correlation_with_delivered_deviation"] for c in background_checks if c.get("status") == "computed"]

    return {
        "sign_convention": {"corpus": corpus_key, "multiplier_applied_to_behaviour_correlations": sign},
        "decision_rule_declared_before_fitting": BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "content_specific_serial_pull_operationalisation": CONTENT_SPECIFIC_SERIAL_PULL_OPERATIONALISATION,
        "detrend_windows_trials": list(DETREND_WINDOWS_TRIALS), "primary_window": primary_window,
        "n_sessions_reaching_lag_profile_floor": sum(1 for g in geometry if g.get("status") == "computed"),
        "n_sessions_total": len(bundles),
        "adjacency_by_window": {str(w): adjacency_by_window[w] for w in DETREND_WINDOWS_TRIALS},
        "decisive_partial_by_window": decisive_pooled,
        "branch_by_window": branch_by_window,
        "branch_at_primary_window": primary_branch,
        "windows_agree": windows_agree,
        "background_deviation_neighbour_excluded_check": {
            "n_sessions": len(background_checks),
            "median_correlation_with_delivered_deviation": float(np.median(background_r_values)) if background_r_values else None,
            "per_session": background_checks,
        },
        "content_specific_serial_pull_pooled": content_pull_pooled,
        "content_specific_serial_pull_n_sessions_computed": len(content_pull_values),
    }


# =======================================================================================================
# BLOCK B -- temporal locus
# =======================================================================================================

def _sub_window_bins(n_bins: int, n_windows: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n_bins), n_windows)


def _session_subwindow(bundle: dict, bin_indices: np.ndarray, sign: float, seed_prefix: str) -> dict:
    counts = bundle["counts"][:, :, bin_indices]
    activity = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    spike_count = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_direction", "n_trials_finite": int(finite.sum())}
    dev, sc = deviation[finite], spike_count[finite]
    outcome = bundle["outcome_raw"][finite]
    tag = f"{seed_prefix}|subwindow"
    gate = partial_correlation_permutation_test(dev, sc, [], N_PERM, np.random.default_rng(stable_seed(f"{tag}|gate")))
    raw = partial_correlation_permutation_test(outcome, dev, [], N_PERM, np.random.default_rng(stable_seed(f"{tag}|raw")))
    return {
        "status": "computed", "n_trials_finite": int(finite.sum()),
        "median_total_spike_count_per_trial": float(np.median(sc)),
        "gate_vs_spike_count": gate, "raw_vs_behaviour": _sign_to_worse_behaviour(raw, sign),
    }


def _pool_subwindow(per_session: list[dict], key: str, field: str = "r") -> dict:
    values = [s[key][field] for s in per_session if s.get("status") == "computed" and s[key].get("status") == "computed"]
    pooled = slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}
    return pooled


def _split_result(bundles: list[dict], n_windows: int, sign: float, corpus_key: str, split_name: str) -> dict:
    per_session_per_window: list[dict] = []
    for bundle in bundles:
        bin_groups = _sub_window_bins(bundle["counts"].shape[2], n_windows)
        row = {"session": bundle["session"], "windows": []}
        for w_index, bin_indices in enumerate(bin_groups):
            tag = f"deviation_serial_dependence_and_temporal_locus|{corpus_key}|{bundle['session']}|{split_name}|w{w_index}"
            result = _fit(tag, lambda b=bundle, bi=bin_indices, s=sign, t=tag: _session_subwindow(b, bi, s, t))
            row["windows"].append(result)
        per_session_per_window.append(row)

    windows_summary = []
    for w_index in range(n_windows):
        per_session = [row["windows"][w_index] for row in per_session_per_window]
        gate_pooled = _pool_subwindow(per_session, "gate_vs_spike_count")
        raw_pooled = _pool_subwindow(per_session, "raw_vs_behaviour")
        void = bool(gate_pooled.get("status") == "tested" and gate_pooled.get("significant"))
        medians = [s["median_total_spike_count_per_trial"] for s in per_session if s.get("status") == "computed"]
        windows_summary.append({
            "window_index": w_index, "void": void,
            "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
            "median_total_spike_count_per_trial_across_sessions": float(np.median(medians)) if medians else None,
            "gate_pooled": gate_pooled, "raw_vs_behaviour_pooled": raw_pooled,
        })

    surviving = [w for w in windows_summary if not w["void"]]
    ordering_test = None
    if len(surviving) >= 2:
        earliest_idx, latest_idx = surviving[0]["window_index"], surviving[-1]["window_index"]
        paired = []
        for row in per_session_per_window:
            early, late = row["windows"][earliest_idx], row["windows"][latest_idx]
            if early.get("status") == "computed" and late.get("status") == "computed" and \
               early["raw_vs_behaviour"].get("status") == "computed" and late["raw_vs_behaviour"].get("status") == "computed":
                paired.append(late["raw_vs_behaviour"]["r"] - early["raw_vs_behaviour"]["r"])
        ordering_test = slope_across_sessions_test(paired, alternative="two-sided") if paired else {"status": "not_computed"}
        ordering_test["earliest_window_index"] = earliest_idx
        ordering_test["latest_window_index"] = latest_idx
        ordering_test["n_sessions_paired"] = len(paired)

    return {
        "n_windows": n_windows, "windows": windows_summary, "surviving_window_indices": [w["window_index"] for w in surviving],
        "void_window_indices": [w["window_index"] for w in windows_summary if w["void"]],
        "ordering_test_latest_minus_earliest_surviving": ordering_test,
    }


def _block_b_branch(split: dict, full_epoch_behaviour_link_present: bool) -> dict:
    surviving = [w for w in split["windows"] if not w["void"]]
    if not surviving:
        return {"branch": "temporal_locus_unreachable_at_this_count_per_window", "sub_label": None}
    present_flags = [bool(w["raw_vs_behaviour_pooled"].get("status") == "tested"
                          and w["raw_vs_behaviour_pooled"].get("significant")) for w in surviving]
    ordering = split["ordering_test_latest_minus_earliest_surviving"]
    ordering_significant = bool(ordering and ordering.get("status") == "tested" and ordering.get("significant"))

    if all(present_flags):
        if not ordering_significant:
            return {"branch": "accuracy_predicting_component_is_present_throughout_the_delay", "sub_label": None}
        if ordering["mean_value"] > 0.0:
            return {"branch": "accuracy_predicting_component_grows_across_the_delay", "sub_label": None}
        return {"branch": "accuracy_predicting_component_is_strongest_at_delay_onset", "sub_label": None}
    if not any(present_flags):
        if full_epoch_behaviour_link_present:
            return {"branch": "accuracy_predicting_component_requires_the_full_epoch_to_be_detected", "sub_label": None}
        return {"branch": "block_b_outcome_not_covered_by_the_pre_declared_rule",
                "sub_label": "absent_in_every_surviving_sub_window_but_also_absent_over_the_whole_epoch"}
    return {"branch": "block_b_outcome_not_covered_by_the_pre_declared_rule",
            "sub_label": "behaviour_link_present_in_some_but_not_all_surviving_sub_windows"}


def run_block_b(bundles: list[dict], sign: float, corpus_key: str, full_epoch_behaviour_link_present: bool) -> dict:
    splits = {name: _split_result(bundles, n, sign, corpus_key, name) for name, n in SUB_WINDOW_SPLITS.items()}
    primary = splits[PRIMARY_SPLIT]
    branch = _block_b_branch(primary, full_epoch_behaviour_link_present)
    thirds_branch = _block_b_branch(splits["thirds"], full_epoch_behaviour_link_present)
    return {
        "sign_convention": {"corpus": corpus_key, "multiplier_applied_to_behaviour_correlations": sign},
        "decision_rule_declared_before_fitting": BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "primary_split": PRIMARY_SPLIT, "full_epoch_behaviour_link_present": full_epoch_behaviour_link_present,
        "splits": splits, "branch": branch,
        "thirds_sensitivity_branch": thirds_branch,
        "primary_and_sensitivity_agree": branch["branch"] == thirds_branch["branch"],
    }


# =======================================================================================================
# Driver
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Run only on the two corpora whose rate-free deviation observable passes its own orthogonality "
            "gate against total spike count: the single-item macaque lateral prefrontal cortex corpus "
            "(Panichello et al. 2024) and the multi-object macaque corpus (Watters, Gabel, Tenenbaum and "
            "Jazayeri; DANDI 000620). The mouse anterior lateral motor cortex corpus and both human corpora "
            "are excluded here on that already-measured precondition, not left pending."
        ),
        "sign_map": SIGN_TO_WORSE_BEHAVIOUR,
        "sign_map_note": (
            "Applied once, at the point each behavioural-correlation result is packaged for this artifact, "
            "never inside the correlation estimator itself. Every reported coefficient here is against WORSE "
            "behaviour."
        ),
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass, shared by the reproduction gate and both blocks)")
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
    gate_result, watters_arrays_by_session = _fit(
        "reproduction_gate", lambda: full_reproduction_gate(root, watters_loaded))
    output["reproduction_gate"] = gate_result
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")

    if gate_result["status"] != "reproduced_exactly":
        output["status"] = "stopped_reproduction_gate_failed"
        output["block_a"] = {"status": "not_run_reproduction_gate_failed"}
        output["block_b"] = {"status": "not_run_reproduction_gate_failed"}
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce exactly; no new number was read")
        print(json.dumps({"reproduction_gate": gate_result["status"]}, indent=2))
        return

    macaque_paths = _reachable_sessions(root)
    macaque_bundles = []
    for path in macaque_paths:
        bundle = _fit(f"macaque_bundle|{path.stem}", lambda p=path: _macaque_session_bundle(p))
        if bundle is not None:
            macaque_bundles.append(bundle)
    n_macaque_on_disk = len(list(_panichello_directory(root).glob("*.mat"))) if _panichello_directory(root) else 0

    watters_bundles = []
    n_watters_arrays_none = 0
    for session in watters_loaded:
        entry = watters_arrays_by_session.get(session["session"])
        if entry is None:
            n_watters_arrays_none += 1
            continue
        watters_bundles.append(_watters_session_bundle(session, entry["arrays"], entry["usable"]))

    output["reachability"] = {
        "panichello_2024_macaque_lPFC_single_item": {
            "n_sessions_on_disk": n_macaque_on_disk, "n_sessions_reaching_the_reachability_floor": len(macaque_paths),
            "n_sessions_with_a_computed_bundle": len(macaque_bundles),
            "n_trials_total": sum(b["n_trials_total"] for b in macaque_bundles),
            "median_units": float(np.median([b["activity_by_unit"].shape[1] for b in macaque_bundles])) if macaque_bundles else None,
            "reference_effect_size_r_units": abs(MACAQUE_DEVIATION_RAW_R),
        },
        "watters_2026_macaque_multi_object": {
            "n_sessions_seen": watters_seen, "n_sessions_loaded": len(watters_loaded),
            "n_sessions_refused_by_the_shared_loader": len(watters_refused),
            "n_sessions_arrays_not_computable": n_watters_arrays_none,
            "n_sessions_with_a_computed_bundle": len(watters_bundles),
            "n_trials_total": sum(b["n_trials_total"] for b in watters_bundles),
            "median_units": float(np.median([b["activity_by_unit"].shape[1] for b in watters_bundles])) if watters_bundles else None,
            "reference_effect_size_r_units": abs(WATTERS_DEVIATION_RAW_R),
        },
        "zero_drop_accounting": {
            "panichello_2024_macaque_lPFC_single_item": {
                "n_seen": n_macaque_on_disk,
                "n_excluded_below_reachability_floor": n_macaque_on_disk - len(macaque_paths),
                "n_reaching_floor": len(macaque_paths),
                "n_excluded_too_few_trials_with_defined_direction": len(macaque_paths) - len(macaque_bundles),
                "n_analysed": len(macaque_bundles),
                "reconciles": bool(n_macaque_on_disk == (n_macaque_on_disk - len(macaque_paths)) +
                                   (len(macaque_paths) - len(macaque_bundles)) + len(macaque_bundles)),
            },
            "watters_2026_macaque_multi_object": {
                "n_seen": watters_seen, "n_refused_by_shared_loader": len(watters_refused),
                "n_loaded": len(watters_loaded), "n_arrays_not_computable": n_watters_arrays_none,
                "n_analysed": len(watters_bundles),
                "reconciles": bool(watters_seen == len(watters_refused) + n_watters_arrays_none + len(watters_bundles)),
            },
        },
    }
    _flush(output)
    _log(f"macaque bundles: {len(macaque_bundles)}/{len(macaque_paths)}; "
         f"watters bundles: {len(watters_bundles)}/{len(watters_loaded)}; elapsed={time.time() - t0:.0f}s")

    corpora = {
        "panichello_2024_macaque_lPFC_single_item": (macaque_bundles, MACAQUE_DEVIATION_RAW_P <= 0.05),
        "watters_2026_macaque_multi_object": (watters_bundles, WATTERS_DEVIATION_RAW_P <= 0.05),
    }

    output["block_a"] = {}
    for corpus_key, (bundles, _full_epoch_present) in corpora.items():
        sign = SIGN_TO_WORSE_BEHAVIOUR[corpus_key]
        _log(f"Block A: {corpus_key} ({len(bundles)} sessions)")
        output["block_a"][corpus_key] = run_block_a(
            bundles, sign, corpus_key, f"deviation_serial_dependence_and_temporal_locus|block_a|{corpus_key}")
        _flush(output)
        _log(f"  Block A {corpus_key} branch at primary window: "
             f"{output['block_a'][corpus_key]['branch_at_primary_window']} elapsed={time.time() - t0:.0f}s")

    output["block_b"] = {}
    for corpus_key, (bundles, full_epoch_present) in corpora.items():
        sign = SIGN_TO_WORSE_BEHAVIOUR[corpus_key]
        _log(f"Block B: {corpus_key} ({len(bundles)} sessions)")
        output["block_b"][corpus_key] = run_block_b(bundles, sign, corpus_key, full_epoch_present)
        _flush(output)
        _log(f"  Block B {corpus_key} branch: {output['block_b'][corpus_key]['branch']['branch']} "
             f"elapsed={time.time() - t0:.0f}s")

    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/deviation_serial_dependence_and_temporal_locus_checkpoint.json",
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": gate_result["status"],
        "block_a": {k: v["branch_at_primary_window"] for k, v in output["block_a"].items()},
        "block_b": {k: v["branch"]["branch"] for k, v in output["block_b"].items()},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
