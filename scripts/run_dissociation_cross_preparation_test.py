"""run_dissociation_cross_preparation_test.py -- does the macaque lPFC
rate/deviation dissociation reproduce in mouse ALM, where the memorandum
sits in a different place relative to the dominant population latent?

results/dominant_latent_identity_and_behaviour_breadth.json establishes a
within-corpus dissociation in macaque lateral prefrontal cortex: the dominant
latent's per-trial amplitude fails an orthogonality gate against total spike
count (it IS rate), carries the LARGER raw association with trial outcome,
and loses all of it to spike-count partialling; the rate-free direction
deviation passes the same gate, starts smaller, and survives the same
control unmoved (results/rate_free_state_geometry_behavior_link.json).

Two accounts explain that fact equally well. The general-confound account
says the dominant population mode is rate in any preparation and any rate
correlate of outcome is arousal/satiety/drift, not memory -- the
dissociation should reproduce everywhere. The content-location account says
the dissociation happens because the memorandum is not in the leading
latent in macaque lPFC specifically -- where it IS in the leading latent,
that latent's amplitude should carry outcome information and survive the
same control. results/state_content_link.json and
results/human_content_decodability.json establish that mouse ALM sits at the
opposite end of exactly this variable from macaque lPFC (content decodes
from the leading latent in ALM, mean fractional rank 0.1429, 23/23 sessions
above null; it does not in macaque, 0.6743, 24/25). This module runs the
identical two-observable pipeline on ALM, changing only the corpus, and
reports which of the two accounts the pattern is consistent with under a
rule declared before any fit runs.

No estimator is forked. The dominant-latent amplitude covariate
(trial_amplitude_covariates), the rate-free direction-deviation observable
and its decision-rule classifier (rate_free_state_deviation, _classify), the
amplitude decision-rule classifier (_classify_amplitude), the partial-
correlation permutation primitive (partial_correlation_permutation_test) and
the pooled paired sign-flip test across sessions (slope_across_sessions_test)
are all imported unchanged from the modules that produced the macaque
artifacts above. Only the corpus loader (load_alm_raw_session), the outcome
coding, the per-session array assembly and the floor/branch bookkeeping
around them are new, because ALM has no analogue of any of the reused code.

Outcome coding: Behavior.Trial_types_of_response_vector takes values 1-4 on
every analysed control trial. Cross-tabulated against the instructed side
derived from Trial_info.Trial_types, codes {1, 3} occur only on
right-instructed trials and {2, 4} only on left-instructed trials
(deterministic, asserted below per session); within each side, reading {1,
2} as correct gives session accuracy near 0.79 on the first session on disk,
while the alternative reading would put a trained animal below chance, so
{1, 2} are correct and {3, 4} are error trials. Perturbation trials are a
different experiment (a different manipulation of the same task) and never
enter -- only the control arm of load_alm_raw_session is used.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import alm_data_directory, data_root, load_alm_raw_session  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_behavior_amplitude_rate_controls import _reachable_sessions as _macaque_reachable_sessions  # noqa: E402
from run_dominant_latent_identity_and_behaviour_breadth import (  # noqa: E402
    _behaviour_session_arrays as _macaque_behaviour_session_arrays,
    _classify_amplitude,
    _correlation_family as _macaque_correlation_family,
    _load_session as _macaque_load_session,
    _pool_correlations as _macaque_pool_correlations,
    _session_paths as _macaque_all_session_paths,
    PRIMARY_ERROR_FLOOR as MACAQUE_PRIMARY_ERROR_FLOOR,
)
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, _analyze_session as _macaque_analyze_deviation_session,
    _classify, _pool as _macaque_pool_deviation, _session_arrays as _macaque_deviation_session_arrays,
    rate_free_state_deviation,
)
from run_state_behavior_link import trial_amplitude_covariates  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "dissociation_cross_preparation_test.json"
CHECKPOINT_PATH = OUTPUT_PATH.parent / ".checkpoints" / "dissociation_cross_preparation_test_checkpoint.json"
SEED_NAMESPACE = "dissociation_cross_preparation_test"

N_PERM = 10000
DELAY_WINDOW_S = 1.2  # the same delay-duration filter the pre-declared per-session error-trial table was measured at
BIN_MS = 100.0
MIN_TRIALS_WITH_DEFINED_DIRECTION = 16

ERROR_FLOORS = (30, 25, 40)
PRIMARY_ERROR_FLOOR = 30

# The macaque numbers this arm's finding is read against (results/dominant_latent_identity_and_behaviour_breadth.json,
# results/rate_free_state_geometry_behavior_link.json), quoted here as constants rather than re-typed at each use.
MACAQUE_AMPLITUDE_GATE_R = 0.723869082837011
MACAQUE_AMPLITUDE_GATE_P = 0.0033996600339966003
MACAQUE_AMPLITUDE_RAW_R = -0.1675415174221389
MACAQUE_AMPLITUDE_RAW_P = 0.0059994000599940004
MACAQUE_AMPLITUDE_PARTIAL_SPIKE_R = -0.0157
MACAQUE_AMPLITUDE_PARTIAL_SPIKE_P = 0.6755
MACAQUE_DEVIATION_GATE_R = 0.024701485905536346
MACAQUE_DEVIATION_GATE_P = 0.7384261573842615
MACAQUE_DEVIATION_RAW_R = -0.09742768955602038
MACAQUE_DEVIATION_RAW_P = 0.0034996500349965005
MACAQUE_DEVIATION_JOINT_PARTIAL_R = -0.0696
MACAQUE_DEVIATION_JOINT_PARTIAL_P = 0.021

REPRODUCTION_TOLERANCE = 1e-6  # every reused function is fully seeded and unchanged, so an exact match is expected;
                               # this tolerance only absorbs a platform-level floating-point difference

REPRODUCTION_GATE_RULE_DECLARED_BEFORE_FITTING = (
    "Before any mouse ALM number is trusted, the shared machinery is re-run on the same macaque sessions and "
    "compared to the values already recorded in results/dominant_latent_identity_and_behaviour_breadth.json "
    "(dominant-latent amplitude, floor 60) and results/rate_free_state_geometry_behavior_link.json (rate-free "
    "deviation), by calling the same functions this module imports unchanged rather than retyping the numbers. "
    "Every one of those functions is fully seeded, so an exact reproduction is expected; the declared tolerance "
    "exists only to absorb a platform-level floating-point difference, not a real discrepancy. If any of the four "
    "reproduced values (both observables' orthogonality-gate r and raw-outcome r) misses its tolerance, the run "
    "stops before touching the mouse data -- a machinery difference invalidates the entire comparison this arm "
    "exists to make."
)

CROSS_PREPARATION_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "At the primary error-trial floor, four mutually exclusive branches, checked in this order, decide between "
    "the general-confound account (the dissociation is a fact about rate in any preparation) and the "
    "content-location account (the dissociation is a fact about where the memorandum sits relative to the "
    "dominant latent), using only the amplitude observable's orthogonality gate and its deciding statistic "
    "(picked by the gate exactly as _classify_amplitude picks it) and the deviation observable's own branch "
    "(picked by _classify exactly as the macaque arm picks it):\n"
    "  1. If the dominant amplitude's orthogonality gate against total spike count is NOT significant (the "
    "amplitude passes, unlike in macaque where it fails at 0.7239, p=0.0034), the branch is "
    "'dominant_mode_is_not_rate_in_this_preparation' regardless of what either correlation does -- this single "
    "fact already bounds the macaque gate result to the corpus that produced it, independent of the "
    "general-confound/content-location question.\n"
    "  2. Else (the gate is significant, so the dominant amplitude IS rate here as in macaque): if the "
    "amplitude's deciding statistic (the partial correlation controlling total spike count, per "
    "_classify_amplitude) is significant at the two-sided 0.05 level, the branch is "
    "'dominant_amplitude_link_survives_where_content_is_dominant' -- consistent with, not a demonstration of, "
    "the content-location account, because a component can survive a rate control for reasons other than "
    "carrying the content.\n"
    "  3. Else (the amplitude's deciding statistic is not significant, so the amplitude loses its outcome "
    "association to spike-count partialling exactly as in macaque): if the deviation observable's own branch "
    "(from _classify) is 'rate_free_state_geometry_predicts_accuracy', the branch is "
    "'dissociation_reproduces_in_the_second_preparation' -- consistent with the general-confound account.\n"
    "  4. Else, if the deviation observable does not reach significance either but its minimum detectable "
    "paired difference at 80% power is below the macaque amplitude raw effect size "
    f"({abs(MACAQUE_AMPLITUDE_RAW_R):.4f} r units) and the macaque deviation raw effect size "
    f"({abs(MACAQUE_DEVIATION_RAW_R):.4f} r units), the branch is "
    "'neither_observable_predicts_outcome_in_this_preparation_off_branch_list' -- a reachable result the "
    "pre-declared rule does not name, reported as its own outcome rather than forced onto either listed branch.\n"
    "  5. Otherwise the branch is 'inconclusive_below_detection_floor', with the minimum detectable paired "
    "difference reported beside it.\n"
    "Neither the general-confound account nor the content-location account is preferred, and no outcome listed "
    "above is a failure of this arm."
)


# ============================================================================
# Reproduction gate -- run the shared machinery on macaque data first
# ============================================================================

def _reproduce_macaque_deviation(root: Path) -> dict:
    paths = _macaque_reachable_sessions(root)
    sessions = []
    for path in paths:
        arrays = _macaque_deviation_session_arrays(path)
        if arrays is None:
            continue
        sessions.append({"session": path.stem, "status": "computed",
                          "analysis": _macaque_analyze_deviation_session(path.stem, arrays)})
    gate = _macaque_pool_deviation(sessions, "orthogonality_gate")
    raw = _macaque_pool_deviation(sessions, "raw_outcome_vs_deviation")
    return {"n_sessions": len(sessions), "gate": gate, "raw": raw}


def _reproduce_macaque_amplitude(root: Path) -> dict:
    paths = _macaque_all_session_paths(root)
    per_session = []
    for path in paths:
        session = _macaque_load_session(path)
        if session["n_error"] < MACAQUE_PRIMARY_ERROR_FLOOR:
            continue
        arrays = _macaque_behaviour_session_arrays(session)
        if arrays is None:
            continue
        per_session.append({
            "n_trials_total": arrays["n_trials_total"],
            **_macaque_correlation_family(arrays, "amplitude", session["session"]),
        })
    gate = _macaque_pool_correlations(per_session, "orthogonality_gate")
    raw = _macaque_pool_correlations(per_session, "raw_outcome_vs_observable")
    return {"n_sessions": len(per_session), "gate": gate, "raw": raw}


def _close(observed: float | None, expected: float, tol: float = REPRODUCTION_TOLERANCE) -> bool:
    return observed is not None and abs(observed - expected) <= tol


def reproduction_gate(root: Path) -> dict:
    deviation = _reproduce_macaque_deviation(root)
    amplitude = _reproduce_macaque_amplitude(root)
    checks = {
        "deviation_orthogonality_gate_r": _close(deviation["gate"].get("mean_value"), MACAQUE_DEVIATION_GATE_R),
        "deviation_orthogonality_gate_p": _close(deviation["gate"].get("two_sided_p_value"), MACAQUE_DEVIATION_GATE_P),
        "deviation_raw_r": _close(deviation["raw"].get("mean_value"), MACAQUE_DEVIATION_RAW_R),
        "deviation_raw_p": _close(deviation["raw"].get("two_sided_p_value"), MACAQUE_DEVIATION_RAW_P),
        "amplitude_orthogonality_gate_r": _close(amplitude["gate"].get("mean_value"), MACAQUE_AMPLITUDE_GATE_R),
        "amplitude_orthogonality_gate_p": _close(amplitude["gate"].get("two_sided_p_value"), MACAQUE_AMPLITUDE_GATE_P),
        "amplitude_raw_r": _close(amplitude["raw"].get("mean_value"), MACAQUE_AMPLITUDE_RAW_R),
        "amplitude_raw_p": _close(amplitude["raw"].get("two_sided_p_value"), MACAQUE_AMPLITUDE_RAW_P),
    }
    return {
        "status": "reproduced_exactly" if all(checks.values()) else "not_reproduced",
        "tolerance": REPRODUCTION_TOLERANCE,
        "checks": checks,
        "deviation": deviation,
        "amplitude": amplitude,
        "rule": REPRODUCTION_GATE_RULE_DECLARED_BEFORE_FITTING,
    }


# ============================================================================
# ALM session loading and outcome coding
# ============================================================================

def _load_alm_census(root: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Every control-arm session, with the outcome cross-tabulation asserted
    per session before any behaviour statistic is computed. A session that
    fails the cross-tabulation, or that the loader cannot read, is refused
    with a named reason rather than silently skipped."""
    directory = alm_data_directory(root)
    paths = sorted(directory.glob("*.mat")) if directory.is_dir() else []
    sessions: dict[str, dict] = {}
    refused: dict[str, str] = {}
    for path in paths:
        session_id = path.stem.replace("_units", "")
        raw = load_alm_raw_session(path, bin_ms=BIN_MS, window_s=DELAY_WINDOW_S, require_both_arms=False)
        if raw is None:
            refused[session_id] = "loader_returned_none_below_minimum_trial_or_unit_count"
            continue
        code = raw["control_response_code"]
        condition = raw["control_condition"]
        right_codes = set(np.unique(code[condition == 1]).tolist())
        left_codes = set(np.unique(code[condition == 0]).tolist())
        if not right_codes <= {1, 3} or not left_codes <= {2, 4}:
            refused[session_id] = (
                f"outcome_cross_tabulation_failed: right-instructed codes {sorted(right_codes)} not subset of "
                f"{{1, 3}}, or left-instructed codes {sorted(left_codes)} not subset of {{2, 4}}"
            )
            continue
        is_corr = np.isin(code, [1, 2])
        n_trials = int(len(code))
        n_error = int((~is_corr).sum())
        sessions[session_id] = {
            "session": session_id, "mouse": raw["mouse"], "counts": raw["control_counts"],
            "is_corr": is_corr, "n_trials": n_trials, "n_correct": int(is_corr.sum()), "n_error": n_error,
            "n_units": raw["n_units_after_rate_qc"],
            "outcome_field_distinct_values": sorted(int(v) for v in np.unique(code)),
        }
    return sessions, refused


# ============================================================================
# Per-session observable arrays and correlation family (mirrors the macaque
# arm's _behaviour_session_arrays / _correlation_family exactly, extended
# with the orthogonality gate against trial index this mandate additionally
# requires)
# ============================================================================

def _behaviour_session_arrays(session: dict) -> dict | None:
    counts = session["counts"]
    if counts.shape[0] < 16:
        return None
    activity_by_unit = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity_by_unit)
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return None
    finite = np.isfinite(deviation)
    if finite.sum() < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    return {
        "is_corr": session["is_corr"][finite].astype(float),
        "deviation": deviation[finite],
        "amplitude": np.asarray(covariates["leading_component_score_gain"])[finite],
        "spike_count": activity_by_unit.sum(axis=1)[finite],
        "trial_index": np.arange(counts.shape[0], dtype=float)[finite],
        "n_trials_total": int(counts.shape[0]),
        "n_trials_with_defined_direction": int(finite.sum()),
    }


def _corr(outcome: np.ndarray, covariate: np.ndarray, controls: list[np.ndarray], seed_tag: str) -> dict:
    rng = np.random.default_rng(stable_seed(f"{SEED_NAMESPACE}|{seed_tag}"))
    return partial_correlation_permutation_test(outcome, covariate, controls=controls, n_perm=N_PERM, rng=rng)


def _correlation_family(arrays: dict, observable_key: str, session_id: str) -> dict:
    outcome, x = arrays["is_corr"], arrays[observable_key]
    spike_count, trial_index = arrays["spike_count"], arrays["trial_index"]
    tag = f"{observable_key}|{session_id}"
    return {
        "orthogonality_gate": _corr(x, spike_count, [], f"{tag}|gate_spike"),
        "orthogonality_gate_vs_trial_index": _corr(x, trial_index, [], f"{tag}|gate_trial"),
        "raw_outcome_vs_observable": _corr(outcome, x, [], f"{tag}|raw"),
        "partial_controlling_spike_count": _corr(outcome, x, [spike_count], f"{tag}|ctrl_spike"),
        "partial_controlling_trial_index": _corr(outcome, x, [trial_index], f"{tag}|ctrl_trial"),
        "joint_partial_controlling_spike_count_and_trial_index": _corr(
            outcome, x, [spike_count, trial_index], f"{tag}|ctrl_joint"),
    }


def _pool_correlations(per_session: list[dict], key: str) -> dict:
    values = [s[key]["r"] for s in per_session if s[key].get("status") == "computed"]
    return slope_across_sessions_test(values, alternative="two-sided") if values else {"status": "not_computed"}


def _mdd(per_session: list[dict], key: str) -> dict:
    values = [s[key]["r"] for s in per_session if s[key].get("status") == "computed"]
    if len(values) < 2:
        return {"status": "not_computable", "n": len(values), "reason": "fewer than 2 sessions"}
    return minimum_detectable_paired_difference(values)


def _mouse_counts(session_ids: list[str], census: dict) -> dict:
    counts: dict[str, int] = {}
    for session_id in session_ids:
        mouse = census[session_id]["mouse"]
        counts[mouse] = counts.get(mouse, 0) + 1
    return counts


# ============================================================================
# Checkpointing
# ============================================================================

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
    scratch.replace(CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    scratch.replace(OUTPUT_PATH)


# ============================================================================
# Floor blocks
# ============================================================================

def _floor_block(floor: int, rows: dict[str, dict], census: dict, session_ids: list[str],
                  excluded_other: dict[str, str], classify_as: str) -> dict:
    included = [s for s in session_ids if census[s]["n_error"] >= floor and s in rows]
    per_session = [rows[s] for s in included]
    pooled = {key: _pool_correlations(per_session, key) for key in (
        "orthogonality_gate", "orthogonality_gate_vs_trial_index", "raw_outcome_vs_observable",
        "partial_controlling_spike_count", "partial_controlling_trial_index",
        "joint_partial_controlling_spike_count_and_trial_index")}
    mdd_by_key = {key: _mdd(per_session, key)
                  for key in ("raw_outcome_vs_observable", "partial_controlling_spike_count")}
    mdd = mdd_by_key["raw_outcome_vs_observable"]
    pooled["minimum_detectable_paired_difference_at_80pct_power"] = mdd_by_key
    block = {
        "error_trial_floor": floor,
        "role": "pre-declared primary" if floor == PRIMARY_ERROR_FLOOR else "sensitivity analysis",
        "n_sessions_included": len(included),
        "sessions_included": included,
        "sessions_excluded_below_floor": {
            s: census[s]["n_error"] for s in session_ids if census[s]["n_error"] < floor},
        "sessions_excluded_for_other_reasons": {
            s: excluded_other[s] for s in excluded_other if census[s]["n_error"] >= floor},
        "sessions_by_mouse": _mouse_counts(included, census),
        "pooled": pooled,
    }
    if classify_as == "deviation":
        block["branch"] = _classify(
            pooled["orthogonality_gate"], pooled["raw_outcome_vs_observable"],
            pooled["joint_partial_controlling_spike_count_and_trial_index"],
            mdd.get("mdd") if mdd.get("status") == "computed" else None)
        raw = pooled["raw_outcome_vs_observable"]
        block["effect_size"] = {
            "pooled_r": raw.get("mean_value"), "p_value": raw.get("two_sided_p_value", raw.get("p_value")),
            "ci": [raw.get("ci_lower"), raw.get("ci_upper")], "null_reference": 0.0,
            "correlation_is_with_correct_equals_one": True,
        }
    else:
        block.update(_classify_amplitude(pooled, mdd_by_key))
    return block


# ============================================================================
# Cross-preparation branch
# ============================================================================

def _cross_preparation_branch(amplitude_primary: dict, deviation_primary: dict) -> dict:
    """Implements CROSS_PREPARATION_DECISION_RULE_DECLARED_BEFORE_FITTING."""
    gate = amplitude_primary["pooled"]["orthogonality_gate"]
    gate_significant = bool(gate.get("significant")) if gate.get("status") == "tested" else None
    deciding_significant = amplitude_primary.get("branch") == "dominant_latent_amplitude_predicts_outcome"
    deviation_branch = deviation_primary.get("branch")
    deviation_survives = deviation_branch == "rate_free_state_geometry_predicts_accuracy"

    deviation_mdd = deviation_primary["pooled"]["minimum_detectable_paired_difference_at_80pct_power"][
        "raw_outcome_vs_observable"]
    deviation_mdd_value = deviation_mdd.get("mdd") if deviation_mdd.get("status") == "computed" else None
    reachable_against_macaque = (
        deviation_mdd_value is not None
        and deviation_mdd_value < abs(MACAQUE_AMPLITUDE_RAW_R)
        and deviation_mdd_value < abs(MACAQUE_DEVIATION_RAW_R)
    )

    if gate_significant is None:
        branch = "inconclusive_below_detection_floor"
        reason = "amplitude orthogonality gate not computable at the primary floor"
    elif not gate_significant:
        branch = "dominant_mode_is_not_rate_in_this_preparation"
        reason = "amplitude orthogonality gate against total spike count is not significant"
    elif deciding_significant:
        branch = "dominant_amplitude_link_survives_where_content_is_dominant"
        reason = "amplitude's deciding statistic (partial controlling spike count) is significant"
    elif deviation_survives:
        branch = "dissociation_reproduces_in_the_second_preparation"
        reason = "amplitude loses its outcome association to spike-count partialling while the deviation's survives"
    elif reachable_against_macaque:
        branch = "neither_observable_predicts_outcome_in_this_preparation_off_branch_list"
        reason = ("neither observable predicts outcome despite being powered to detect an effect at least as "
                  "small as either macaque effect size -- a reachable result the pre-declared rule does not name")
    else:
        branch = "inconclusive_below_detection_floor"
        reason = "deviation observable does not reach significance and cannot detect an effect of the macaque arm's size"

    return {
        "branch": branch,
        "reason": reason,
        "amplitude_orthogonality_gate_significant": gate_significant,
        "amplitude_deciding_statistic_significant": deciding_significant,
        "deviation_branch": deviation_branch,
        "deviation_minimum_detectable_paired_difference": deviation_mdd_value,
        "macaque_amplitude_raw_effect_size_r_units": abs(MACAQUE_AMPLITUDE_RAW_R),
        "macaque_deviation_raw_effect_size_r_units": abs(MACAQUE_DEVIATION_RAW_R),
        "rule": CROSS_PREPARATION_DECISION_RULE_DECLARED_BEFORE_FITTING,
    }


# ============================================================================
# Driver
# ============================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": "2026-08-14",
        "scope": {
            "corpus": "mouse anterior lateral motor cortex, control arm only (Inagaki et al., random-delay "
                      "silicon-probe perturbation release, doi:10.25378/janelia.7489253); photoinhibition-"
                      "perturbation trials are a different experiment and never enter",
            "delay_window_s": DELAY_WINDOW_S,
            "bin_ms": BIN_MS,
            "window_note": "delay duration is randomised trial to trial in this task; the 1.2 s window is the "
                           "same filter the pre-declared per-session error-trial table was measured at, and is "
                           "used identically for both observables",
            "error_trial_floors": list(ERROR_FLOORS),
            "primary_error_trial_floor": PRIMARY_ERROR_FLOOR,
            "n_permutations_per_correlation": N_PERM,
            "seed_namespace": SEED_NAMESPACE,
            "outcome_coding": (
                "Behavior.Trial_types_of_response_vector in {1,2,3,4} on every analysed control trial. "
                "Cross-tabulated against instructed side (derived from Trial_info.Trial_types): codes {1,3} "
                "occur only on right-instructed trials, codes {2,4} only on left-instructed trials, asserted "
                "per session before any behaviour statistic is computed. {1,2} are correct trials and {3,4} "
                "are error trials -- the alternative reading would put a trained animal below chance."
            ),
            "caveats": [
                "The mouse content is a two-class instructed lick direction (a motor plan); the macaque "
                "content is an eight-class visual stimulus. The cross-preparation content contrast this "
                "arm is built on is therefore also a contrast between a motor plan and a stimulus at "
                "different label cardinalities -- a known, recorded property of the comparison that this "
                "arm does not resolve. The within-macaque cardinality ladder "
                "(results/content_label_cardinality_ladder.json, flat: slope -0.0251, p=0.302, every rung "
                "above null) shows label cardinality alone does not close the gap, which narrows this "
                "caveat but does not remove the motor-plan-versus-stimulus part of it.",
                "Brain region and task are not matched and cannot be: anterior lateral motor cortex in a "
                "whisker-based delayed-response task is not lateral prefrontal cortex in a visual "
                "delayed-response task.",
                "If the dominant amplitude's behaviour link survives partialling in mouse, that is "
                "consistent with the content-location account, not a demonstration of it -- a component "
                "can survive a rate control for reasons other than carrying the content.",
            ],
        },
        "cross_preparation_decision_rule_declared_before_fitting": CROSS_PREPARATION_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    # ---- reproduction gate: the shared machinery on macaque data first ------
    _log("running reproduction gate on macaque sessions")
    gate_result = _fit("reproduction_gate", lambda: reproduction_gate(root))
    output["reproduction_gate"] = gate_result
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")
    if gate_result["status"] != "reproduced_exactly":
        output["status"] = "stopped_reproduction_gate_failed"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce the macaque numbers; mouse data was not touched")
        return

    # ---- ALM census -----------------------------------------------------------
    sessions, loader_refused = _load_alm_census(root)
    session_ids = sorted(sessions)
    directory = alm_data_directory(root)
    n_on_disk = len(list(directory.glob("*.mat"))) if directory.is_dir() else 0
    census = {s: {k: v for k, v in sessions[s].items() if k not in ("counts", "is_corr")} for s in session_ids}
    output["per_session_outcome_census"] = {
        "n_session_files_on_disk": n_on_disk,
        "n_sessions_read_and_outcome_verified": len(session_ids),
        "sessions_refused": loader_refused,
        "sessions": [census[s] for s in session_ids],
    }
    _flush(output)
    for s in session_ids:
        _log(f"loaded {s} mouse={census[s]['mouse']} trials={census[s]['n_trials']} "
             f"units={census[s]['n_units']} errors={census[s]['n_error']}")

    # ---- per-session observable arrays and correlation families -------------
    behaviour_rows: dict[str, dict] = {}
    amplitude_rows: dict[str, dict] = {}
    excluded_other: dict[str, str] = {}
    lowest_floor = min(ERROR_FLOORS)
    floor_sessions = [s for s in session_ids if census[s]["n_error"] >= lowest_floor]
    for s in floor_sessions:
        # Array assembly (PCA fit + rank-1 decomposition on one session's counts) is cheap and
        # deterministic; only the permutation-heavy correlation families below are checkpointed.
        arrays = _behaviour_session_arrays(sessions[s])
        if arrays is None:
            excluded_other[s] = "no_observable_computable_for_this_session"
            continue
        behaviour_rows[s] = {
            "n_trials_total": arrays["n_trials_total"],
            "n_trials_with_defined_direction": arrays["n_trials_with_defined_direction"],
            **_fit(f"deviation_family|{s}", lambda arrays=arrays, s=s: _correlation_family(arrays, "deviation", s)),
        }
        amplitude_rows[s] = {
            "n_trials_total": arrays["n_trials_total"],
            **_fit(f"amplitude_family|{s}", lambda arrays=arrays, s=s: _correlation_family(arrays, "amplitude", s)),
        }
        _log(f"behaviour {s} raw_deviation_r={behaviour_rows[s]['raw_outcome_vs_observable']['r']:.4f} "
             f"raw_amplitude_r={amplitude_rows[s]['raw_outcome_vs_observable']['r']:.4f} "
             f"elapsed={time.time() - t0:.1f}s")
        output["_progress"] = {"status": "computing_per_session_correlation_families", "per_session_partial": behaviour_rows}
        _flush(output)

    output.pop("_progress", None)  # scratch key for incremental-flush visibility only; superseded below

    # ---- floor blocks, both observables --------------------------------------
    deviation_floors = {str(floor): _floor_block(floor, behaviour_rows, census, session_ids, excluded_other,
                                                  "deviation") for floor in ERROR_FLOORS}
    amplitude_floors = {str(floor): _floor_block(floor, amplitude_rows, census, session_ids, excluded_other,
                                                  "amplitude") for floor in ERROR_FLOORS}

    deviation_primary = deviation_floors[str(PRIMARY_ERROR_FLOOR)]
    amplitude_primary = amplitude_floors[str(PRIMARY_ERROR_FLOOR)]
    deviation_floors_agree = len({deviation_floors[f]["branch"] for f in deviation_floors}) == 1
    amplitude_floors_agree = len({amplitude_floors[f]["branch"] for f in amplitude_floors}) == 1

    # ---- reachability, reported first ----------------------------------------
    amp_mdd_raw = amplitude_primary["pooled"]["minimum_detectable_paired_difference_at_80pct_power"][
        "raw_outcome_vs_observable"]
    dev_mdd_raw = deviation_primary["pooled"]["minimum_detectable_paired_difference_at_80pct_power"][
        "raw_outcome_vs_observable"]
    amp_mdd_value = amp_mdd_raw.get("mdd") if amp_mdd_raw.get("status") == "computed" else None
    dev_mdd_value = dev_mdd_raw.get("mdd") if dev_mdd_raw.get("status") == "computed" else None
    reachability = {
        "n_sessions_at_primary_floor": amplitude_primary["n_sessions_included"],
        "macaque_amplitude_raw_effect_size_r_units": abs(MACAQUE_AMPLITUDE_RAW_R),
        "macaque_deviation_raw_effect_size_r_units": abs(MACAQUE_DEVIATION_RAW_R),
        "amplitude_minimum_detectable_paired_difference_80pct_power": amp_mdd_value,
        "deviation_minimum_detectable_paired_difference_80pct_power": dev_mdd_value,
        "amplitude_can_reach_the_macaque_amplitude_effect_size": (
            amp_mdd_value is not None and amp_mdd_value < abs(MACAQUE_AMPLITUDE_RAW_R)),
        "deviation_can_reach_the_macaque_deviation_effect_size": (
            dev_mdd_value is not None and dev_mdd_value < abs(MACAQUE_DEVIATION_RAW_R)),
        "note": (
            f"With {amplitude_primary['n_sessions_included']} sessions at the primary {PRIMARY_ERROR_FLOOR}-error "
            "floor (versus the macaque arm's 11), the minimum detectable paired difference at 80% power is "
            "reported for both observables' raw outcome association and compared against both macaque effect "
            "sizes; a branch that depends on a non-significant correlation is only informative where this "
            "comparison holds."
        ),
    }
    output["reachability"] = reachability
    _flush(output)

    # ---- orthogonality gates, both observables, both against spike count and trial index --------
    output["orthogonality_gates"] = {
        "amplitude_vs_spike_count": amplitude_primary["pooled"]["orthogonality_gate"],
        "amplitude_vs_trial_index": amplitude_primary["pooled"]["orthogonality_gate_vs_trial_index"],
        "deviation_vs_spike_count": deviation_primary["pooled"]["orthogonality_gate"],
        "deviation_vs_trial_index": deviation_primary["pooled"]["orthogonality_gate_vs_trial_index"],
        "error_trial_floor": PRIMARY_ERROR_FLOOR,
    }
    _flush(output)

    output["dominant_latent_amplitude_and_outcome"] = {
        "status": "tested",
        "observable": "per-trial score of the leading component of the centred trial-by-window matrix, the "
                      "dominant latent's own per-trial amplitude",
        "by_floor": amplitude_floors,
        "branch_at_primary_floor": amplitude_primary["branch"],
        "floors_agree_on_the_branch": amplitude_floors_agree,
        "per_session": amplitude_rows,
    }
    output["rate_free_deviation_and_outcome"] = {
        "status": "tested",
        "observable": "rate-free direction deviation (direction of the per-trial population activity vector "
                      "against the session's own leave-one-out mean direction)",
        "by_floor": deviation_floors,
        "branch_at_primary_floor": deviation_primary["branch"],
        "floors_agree_on_the_branch": deviation_floors_agree,
        "per_session": behaviour_rows,
    }
    _flush(output)

    # ---- the cross-preparation contrast ---------------------------------------
    cross = _cross_preparation_branch(amplitude_primary, deviation_primary)
    output["cross_preparation_contrast"] = {
        **cross,
        "macaque_dominant_amplitude": {
            "orthogonality_gate_r": MACAQUE_AMPLITUDE_GATE_R, "orthogonality_gate_p": MACAQUE_AMPLITUDE_GATE_P,
            "raw_r": MACAQUE_AMPLITUDE_RAW_R, "raw_p": MACAQUE_AMPLITUDE_RAW_P,
            "partial_controlling_spike_count_r": MACAQUE_AMPLITUDE_PARTIAL_SPIKE_R,
            "partial_controlling_spike_count_p": MACAQUE_AMPLITUDE_PARTIAL_SPIKE_P,
        },
        "macaque_rate_free_deviation": {
            "orthogonality_gate_r": MACAQUE_DEVIATION_GATE_R, "orthogonality_gate_p": MACAQUE_DEVIATION_GATE_P,
            "raw_r": MACAQUE_DEVIATION_RAW_R, "raw_p": MACAQUE_DEVIATION_RAW_P,
            "joint_partial_r": MACAQUE_DEVIATION_JOINT_PARTIAL_R, "joint_partial_p": MACAQUE_DEVIATION_JOINT_PARTIAL_P,
        },
        "mouse_dominant_amplitude": {
            "orthogonality_gate_r": amplitude_primary["pooled"]["orthogonality_gate"].get("mean_value"),
            "orthogonality_gate_p": amplitude_primary["pooled"]["orthogonality_gate"].get("two_sided_p_value"),
            "raw_r": amplitude_primary["pooled"]["raw_outcome_vs_observable"].get("mean_value"),
            "raw_p": amplitude_primary["pooled"]["raw_outcome_vs_observable"].get("two_sided_p_value"),
            "partial_controlling_spike_count_r": amplitude_primary["pooled"]["partial_controlling_spike_count"].get("mean_value"),
            "partial_controlling_spike_count_p": amplitude_primary["pooled"]["partial_controlling_spike_count"].get("two_sided_p_value"),
        },
        "mouse_rate_free_deviation": {
            "orthogonality_gate_r": deviation_primary["pooled"]["orthogonality_gate"].get("mean_value"),
            "orthogonality_gate_p": deviation_primary["pooled"]["orthogonality_gate"].get("two_sided_p_value"),
            "raw_r": deviation_primary["pooled"]["raw_outcome_vs_observable"].get("mean_value"),
            "raw_p": deviation_primary["pooled"]["raw_outcome_vs_observable"].get("two_sided_p_value"),
            "joint_partial_r": deviation_primary["pooled"]["joint_partial_controlling_spike_count_and_trial_index"].get("mean_value"),
            "joint_partial_p": deviation_primary["pooled"]["joint_partial_controlling_spike_count_and_trial_index"].get("two_sided_p_value"),
        },
        "content_location_context": (
            "macaque lPFC: content is not in the dominant state (mean fractional rank 0.6743, p=0.0186, n=25, "
            "24/25 above null, results/state_content_link.json). mouse ALM: content IS the dominant state "
            "(mean fractional rank 0.1429, p=9.999e-05, n=23, 23/23 above null, "
            "results/human_content_decodability.json). This arm holds the two ends of that variable."
        ),
    }
    _flush(output)

    # ---- zero-drop accounting --------------------------------------------------
    output["zero_drop_accounting"] = {
        "n_session_files_on_disk": n_on_disk,
        "n_sessions_read_and_outcome_verified": len(session_ids),
        "n_sessions_refused_by_loader_or_cross_tabulation": len(loader_refused),
        "sessions_refused": loader_refused,
        "arms": {
            "behavioural_correlation_at_the_primary_floor": {
                "seen": len(session_ids), "tested": amplitude_primary["n_sessions_included"],
                "excluded_by_reason": {
                    **{s: f"error trial count {census[s]['n_error']} below the floor of {PRIMARY_ERROR_FLOOR}"
                       for s in session_ids if census[s]["n_error"] < PRIMARY_ERROR_FLOOR},
                    **{s: reason for s, reason in excluded_other.items()},
                },
            },
        },
    }
    for arm in output["zero_drop_accounting"]["arms"].values():
        arm["counts_reconcile"] = arm["seen"] == arm["tested"] + len(arm["excluded_by_reason"])
    seen_total = n_on_disk
    tested_or_refused = len(session_ids) + len(loader_refused)
    output["zero_drop_accounting"]["top_level_counts_reconcile"] = bool(seen_total == tested_or_refused)
    _flush(output)

    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/dissociation_cross_preparation_test_checkpoint.json",
    }

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": gate_result["status"],
        "amplitude_branch_by_floor": {f: amplitude_floors[f]["branch"] for f in amplitude_floors},
        "deviation_branch_by_floor": {f: deviation_floors[f]["branch"] for f in deviation_floors},
        "cross_preparation_branch": cross["branch"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
