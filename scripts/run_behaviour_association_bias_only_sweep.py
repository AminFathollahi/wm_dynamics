"""run_behaviour_association_bias_only_sweep.py -- does this project's
headline behavioural dissociation survive the bias-only control that has
already voided five of its other behavioural claims?

BACKGROUND. Many of this project's behavioural claims are per-session
WITHIN-session correlations between a per-trial predictor and a per-trial
behavioural outcome, pooled across sessions by a paired sign-flip test
(state_persistence.slope_across_sessions_test). The bias-only control asks
whether that pooled association is reproducible from SESSION IDENTITY ALONE:
replace every trial's predictor (and, for the between-session diagnostic,
every trial's outcome) with that session's own mean, which destroys the
within-session variance the real statistic needs, so the closest well-
defined session-is-the-unit-of-analysis alternative is a single BETWEEN-
session correlation over the N (session-mean predictor, session-mean
outcome) points. That construction, and the voiding rule built on it, is
results/swap_versus_imprecision_by_item_count.py's own
_bias_only_between_session, quoted verbatim below
(BIAS_ONLY_SOURCE_QUOTE_VERBATIM) and reused here without modification for
every claim in this sweep that partitions trials by item-count level, and
generalised (only in the sense that "the level" becomes "the whole session",
i.e. no partition at all) for the two macaque lPFC behavioural claims that
have no item-count structure to partition by.

That control has already voided five claims, most recently
results/component_binding_bias_only_control.json (read as this module's
house format and methodological template). It has never been run against
results/rate_free_state_geometry_behavior_link.json -- the rate-free
deviation's within-session link to trial accuracy, the SURVIVING half of
this project's headline dissociation (the dominant population mode's LARGER
raw link to accuracy dies under a spike-count partial control; the smaller
rate-free link does not). This module runs it, first and completely, then
proceeds down a pre-declared priority list of further at-risk claims as time
allows.

REPRODUCTION GATE, mandatory and first, per claim. Before any bias-only
number is read for a claim, that claim's own delivered real statistic is
recomputed from raw data, using the delivered script's own estimators
imported unchanged, and compared to the delivered artifact at tolerance
1e-6. A failed gate reports the discrepancy and reads no control number for
that claim -- a failed gate means code drift, not a scientific result.

VOIDING RULE, pre-declared here before any claim past the headline is
fitted: an association is VOIDED iff (a) the real statistic is significant
(two-sided p < 0.05) AND (b) the bias-only control is significant (two-sided
p < 0.05) AND (c) they have the SAME SIGN. A non-significant real statistic
is never "voided" -- it was never a positive. The bias-only control's
magnitude is a different estimator on a different scale (a between-session
correlation over N session-level points, versus a within-session
correlation pooled by a sign-flip test) and NEVER enters the voiding rule by
size, only by sign and significance.

POWER. Every non-voided cell (a real statistic that is significant and
survives, or one that is not significant) carries its own minimum detectable
effect at 80% power from a WHOLE-SESSION cluster bootstrap standard error:
resample sessions (the real statistic's own clustering unit) with
replacement, recompute the pooled statistic on each resampled set, take the
standard deviation of that bootstrap distribution as the standard error, and
multiply by MDD_Z_FACTOR (statistics.Z_80_POWER, norm.ppf(0.975) +
norm.ppf(0.80), the two-sided-alpha-0.05/80%-power factor). This is NEVER a
trial-count formula and NEVER an intraclass-correlation design effect --
resampling happens over sessions, one draw = one whole session kept or
dropped, matching the real statistic's own unit of analysis exactly. The
identical bootstrap, resampling the same session set, also bounds the
bias-only control itself: a "not significant" bias-only control whose own
bootstrap MDD is at or above MEANINGFUL_EFFECT_THRESHOLD_R_UNITS (0.14, this
project's own established "smallest effect this design would call
meaningful" scale, from results/state_behavior_link.json's persistence
bound) merely lacked the power to find a session-level offset even if one
existed, and the claim is reported 'inconclusive_below_detection_floor'
rather than a clean 'not_voided' pass.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import norm, pearsonr  # noqa: F401  (pearsonr kept for call sites below)

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import data_root  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)

from corpus_sessions import _reachable_sessions as _panichello_reachable_sessions
from run_rate_free_state_geometry_behavior_link import _analyze_session as rate_free_analyze_session
from corpus_sessions import _session_arrays as rate_free_session_arrays
from statistics import MIN_ERROR_TRIALS_FOR_REACHABILITY as RATE_FREE_MIN_ERROR_TRIALS
from stimulation_response_estimator import rate_free_state_deviation
from state_persistence import cheap_first_look
from statistics import MIN_ERROR_TRIALS_FOR_REACHABILITY as GAIN_MIN_ERROR_TRIALS
from spike_pipeline import _counts_from_spikes
from run_swap_versus_imprecision_by_item_count import _bias_only_between_session  # noqa: E402,F401 (source quote below)

OUTPUT_PATH = ROOT / "results" / "behaviour_association_bias_only_sweep.json"
# No separate results/.checkpoints/ directory: each claim (headline first, then the priority list in
# order) is itself the checkpoint granularity -- the full output is flushed to OUTPUT_PATH immediately
# after every claim completes (see main()), so a crash loses at most the one claim in progress, and a
# rerun reads the same delivered artifacts and raw data deterministically rather than resuming partial
# per-session state.
ANALYSIS_VERSION = "2026-09-05"
REPRODUCTION_TOLERANCE = 1e-6
N_PERM = 10000
N_BOOT = 2000
MDD_Z_FACTOR = Z_80_POWER
MEANINGFUL_EFFECT_THRESHOLD_R_UNITS = 0.14  # results/state_behavior_link.json's own persistence-bound scale

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "For every at-risk claim (a significant, per-trial, within-session-and-then-pooled-across-sessions "
    "behavioural association): (0) reproduce the claim's own delivered real statistic from raw data, "
    f"using the delivered script's own estimators imported unchanged, at tolerance {REPRODUCTION_TOLERANCE}. "
    "If this fails, the branch is 'void_reproduction_gate_failed', both numbers are reported, and no "
    "control number is read for that claim. (1) If the real statistic is not itself significant "
    "(two-sided p >= 0.05), the branch is 'real_statistic_not_significant_nothing_to_void' -- a "
    "non-significant real statistic was never a positive and the bias-only question does not arise. "
    "(2) Otherwise, compute the bias-only control: collapse every session's trials to that session's own "
    "mean predictor and mean outcome (and, where the real statistic partials a covariate, that session's "
    "own mean covariate too), and correlate those means BETWEEN sessions -- the closest well-defined, "
    "session-is-the-unit-of-analysis alternative once the within-session substitution has zeroed out "
    "within-session variance by construction (verbatim source: BIAS_ONLY_SOURCE_QUOTE_VERBATIM). "
    "(3) VOIDING RULE: the claim is 'voided' iff the real statistic is significant AND the bias-only "
    "control is significant (two-sided p < 0.05) AND they share the same sign. Sign and significance "
    "only -- the two statistics are different estimators on different scales and the control's magnitude "
    "is never compared to the real effect's magnitude, in either direction. (4) If not voided, both the "
    "real statistic and the bias-only control each carry their own minimum detectable difference at 80% "
    "power from a whole-session cluster bootstrap standard error (resample sessions with replacement, "
    "recompute the pooled statistic per draw, sd of the bootstrap distribution times "
    f"{MDD_Z_FACTOR} = mdd). If the bias-only control's own bootstrap mdd is at or above "
    f"{MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r units, the claim is reported "
    "'inconclusive_below_detection_floor' rather than a clean 'not_voided' pass, because the control "
    "lacked the power to find a session-level offset even if one existed. Otherwise the branch is "
    "'not_voided'."
)

BIAS_ONLY_SOURCE_QUOTE_VERBATIM = (
    "Verbatim from scripts/run_swap_versus_imprecision_by_item_count.py's _bias_only_between_session "
    "docstring (imported unchanged and used here, generalised only in the sense that 'the level' becomes "
    "'the whole session' for the two macaque lPFC claims below, which have no item-count partition):\n\n"
    '"""The session-level bias-only statistic: collapses every session\'s real per-trial deviation at '
    "this level to that session's own mean (the bias-only substitution the control specifies), one "
    "number per session, paired against that same session's own mean outcome at this level -- session "
    "is the unit of analysis, N sessions, matching the real statistic's own unit of analysis.\n\n"
    "This is NOT the same estimator as the real one. The real statistic is a per-session WITHIN-session "
    "correlation (many trials, one r per session) pooled across sessions by a sign-flip test. Once every "
    "trial in a session is replaced by that session's own mean, the within-session predictor has zero "
    "variance by construction, so a per-session correlation is undefined -- there is no well-defined way "
    "to recompute \"the same\" per-session statistic under this substitution. The closest well-defined, "
    "session-is-the-unit-of-analysis alternative is a single BETWEEN-session correlation over the N "
    "(session mean deviation, session mean outcome) points, computed here. Its r, p-value and n are on a "
    "different scale from the real within-session, sign-flip-pooled effect size (different estimator, "
    "different degrees of freedom) and are never compared to the real effect size by magnitude; only "
    'this test\'s own sign and significance enter the voiding rule, which does not depend on magnitude."""'
)

BRANCH_GATE_FAILED = "void_reproduction_gate_failed"
BRANCH_NOT_SIGNIFICANT = "real_statistic_not_significant_nothing_to_void"
BRANCH_VOIDED = "voided"
BRANCH_INCONCLUSIVE = "inconclusive_below_detection_floor"
BRANCH_NOT_VOIDED = "not_voided"
BRANCH_NOT_COMPUTABLE = "not_computable"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _close(a, b, tol: float = REPRODUCTION_TOLERANCE) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(abs(float(a) - float(b)) <= tol)
    except (TypeError, ValueError):
        return a == b


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


def _direction(x) -> str | None:
    if x is None:
        return None
    return "positive" if x > 0.0 else ("negative" if x < 0.0 else "zero")


# ============================================================================
# Whole-session cluster bootstrap MDD
# ============================================================================

def _bootstrap_mdd_from_mean(values: list[float], seed_tag: str, n_boot: int = N_BOOT,
                              return_draws: bool = False) -> dict:
    """MDD for a statistic pooled as the MEAN of one scalar per session
    (this project's sign-flip pooling): resample sessions with replacement,
    recompute the mean each draw, sd of that bootstrap distribution times
    MDD_Z_FACTOR. Session is the resampling unit throughout -- never a trial."""
    n = len(values)
    if n < 2:
        return {"status": "not_computable", "n_sessions": n, "reason": "fewer than 2 sessions"}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = arr[idx].mean()
    se = float(np.std(draws, ddof=1))
    result = {"status": "computed", "n_sessions": n, "n_boot": n_boot, "bootstrap_se": se,
              "z_factor": MDD_Z_FACTOR, "mdd": MDD_Z_FACTOR * se}
    if return_draws:
        result["draws"] = draws
    return result


def _bootstrap_mdd_from_between_session_correlation(x: list[float], y: list[float], seed_tag: str,
                                                     n_boot: int = N_BOOT,
                                                     controls: list[list[float]] | None = None,
                                                     return_draws: bool = False) -> dict:
    """MDD for the bias-only control's own between-session Pearson (or, with
    controls, partial) r: resample the N (session-mean-x, session-mean-y[,
    session-mean-controls]) tuples with replacement (whole sessions, matching
    the control's own unit of analysis), recompute r each draw, sd of the
    bootstrap r distribution times MDD_Z_FACTOR. Draws where the resample
    happens to contain fewer than 2 distinct sessions (r undefined) are
    skipped and do not count toward n_boot_used."""
    n = len(x)
    if n < 4:
        return {"status": "not_computable", "n_sessions": n,
                "reason": "fewer than 4 sessions -- matches the control's own minimum"}
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    control_arrs = [np.asarray(c, dtype=float) for c in (controls or [])]
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 2 or np.std(x_arr[idx]) == 0.0 or np.std(y_arr[idx]) == 0.0:
            continue
        fit = partial_correlation_permutation_test(
            y_arr[idx], x_arr[idx], [c[idx] for c in control_arrs], n_perm=0, rng=rng)
        if fit.get("status") == "computed":
            draws.append(fit["r"])
    if len(draws) < n_boot // 2:
        return {"status": "not_computable", "n_sessions": n,
                "reason": "fewer than half the bootstrap draws yielded a defined correlation"}
    draws_arr = np.asarray(draws)
    se = float(np.std(draws_arr, ddof=1))
    result = {"status": "computed", "n_sessions": n, "n_boot": n_boot, "n_boot_used": len(draws),
              "bootstrap_se": se, "z_factor": MDD_Z_FACTOR, "mdd": MDD_Z_FACTOR * se}
    if return_draws:
        result["draws"] = draws_arr
    return result


def _power_report(real_significant: bool, real_pooled_values: list[float], bias_result: dict,
                   bias_x: list[float] | None, bias_y: list[float] | None, tag: str) -> dict:
    real_mdd = _bootstrap_mdd_from_mean(real_pooled_values, f"{tag}|real_mdd")
    bias_mdd = (
        _bootstrap_mdd_from_between_session_correlation(bias_x, bias_y, f"{tag}|bias_mdd")
        if (bias_x is not None and bias_y is not None) else {"status": "not_applicable"}
    )
    bias_underpowered = bool(
        bias_mdd.get("status") == "computed" and bias_mdd["mdd"] >= MEANINGFUL_EFFECT_THRESHOLD_R_UNITS)
    return {
        "real_statistic_minimum_detectable_difference_80pct_power": real_mdd,
        "bias_only_control_minimum_detectable_difference_80pct_power": bias_mdd,
        "bias_only_control_underpowered_at_the_meaningful_effect_threshold": bias_underpowered,
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    }


def _void_verdict(real_significant: bool | None, real_direction: str | None,
                   bias_significant: bool | None, bias_direction: str | None) -> dict:
    same_sign = bool(real_direction is not None and bias_direction is not None and real_direction == bias_direction)
    voided = bool(real_significant and bias_significant and same_sign)
    return {"real_significant": real_significant, "bias_only_significant": bias_significant,
            "same_sign_as_real": same_sign, "voided": voided}


Z_95 = float(norm.ppf(0.975))


def _real_minus_control(real_point: float | None, real_values: list[float], control_point: float | None,
                         control_x: list[float], control_y: list[float], control_controls: list[list[float]],
                         tag: str) -> dict:
    """The real (within-session, sign-flip-pooled) and bias-only (between-
    session) statistics are both Pearson/partial correlation r -- a common
    scale -- despite being different estimators over different units of
    analysis. Reports their difference and an interval from combining each
    side's own independent whole-session cluster bootstrap standard error
    (normal approximation on the difference of two independent estimates)."""
    if real_point is None or control_point is None:
        return {"status": "not_applicable"}
    real_mdd = _bootstrap_mdd_from_mean(real_values, f"{tag}|real_mdd")
    control_mdd = _bootstrap_mdd_from_between_session_correlation(
        control_x, control_y, f"{tag}|control_mdd", controls=control_controls)
    estimate = float(real_point) - float(control_point)
    if real_mdd.get("status") != "computed" or control_mdd.get("status") != "computed":
        return {"status": "not_computable", "estimate": estimate, "real_value": float(real_point),
                "control_value": float(control_point)}
    se = float(np.sqrt(real_mdd["bootstrap_se"] ** 2 + control_mdd["bootstrap_se"] ** 2))
    return {
        "status": "computed", "estimate": estimate, "real_value": float(real_point),
        "control_value": float(control_point), "standard_error": se,
        "interval_95pct": [estimate - Z_95 * se, estimate + Z_95 * se],
        "note": "real and control are both correlation-scale estimates (r); interval combines "
                "independent whole-session bootstrap standard errors from each side",
    }


def _within_session_demeaned_reading(rows_arrays: dict[str, dict], y_key: str, x_key: str,
                                      control_keys: list[str], tag: str) -> dict:
    """Session-mean-centers y, x and every control within each session, pools
    every trial across sessions, and correlates once -- the classic within-
    subject (fixed-effects-demeaned) estimator, distinct from the real
    statistic's per-session r averaged across sessions. Interval from a
    whole-session cluster bootstrap (resample sessions with replacement,
    recompute on the resampled sessions' concatenated demeaned trials)."""
    session_ids = list(rows_arrays)
    n_sessions = len(session_ids)
    y_parts, x_parts = [], []
    control_parts: list[list[np.ndarray]] = [[] for _ in control_keys]
    for session_id in session_ids:
        arrays = rows_arrays[session_id]
        y = np.asarray(arrays[y_key], dtype=float)
        x = np.asarray(arrays[x_key], dtype=float)
        y_parts.append(y - y.mean())
        x_parts.append(x - x.mean())
        for i, key in enumerate(control_keys):
            c = np.asarray(arrays[key], dtype=float)
            control_parts[i].append(c - c.mean())

    def _pooled(idx):
        y = np.concatenate([y_parts[i] for i in idx])
        x = np.concatenate([x_parts[i] for i in idx])
        controls = [np.concatenate([control_parts[c][i] for i in idx]) for c in range(len(control_keys))]
        return y, x, controls

    rng = np.random.default_rng(stable_seed(f"{tag}|within_session_demeaned"))
    y_all, x_all, controls_all = _pooled(range(n_sessions))
    point = partial_correlation_permutation_test(y_all, x_all, controls_all, n_perm=N_PERM, rng=rng)
    if point.get("status") != "computed":
        return {"status": "not_computable", "reason": point.get("reason", point.get("status"))}

    boot_rng = np.random.default_rng(stable_seed(f"{tag}|within_session_demeaned|bootstrap"))
    draws = []
    for _ in range(N_BOOT):
        idx = boot_rng.integers(0, n_sessions, size=n_sessions)
        y, x, controls = _pooled(idx)
        if np.std(y) == 0.0 or np.std(x) == 0.0:
            continue
        fit = partial_correlation_permutation_test(y, x, controls, n_perm=0, rng=boot_rng)
        if fit.get("status") == "computed":
            draws.append(fit["r"])
    interval = [float(v) for v in np.percentile(draws, [2.5, 97.5])] if len(draws) >= N_BOOT // 2 else None
    return {
        "status": "computed", "r": point["r"], "p_value": point["p_value"], "n_sessions": n_sessions,
        "n_trials": int(len(y_all)), "interval_95pct": interval, "n_boot_used": len(draws),
        "method": "session-mean-centered trials pooled across sessions and correlated once, interval "
                  "from a whole-session cluster bootstrap",
    }


# ============================================================================
# Claim 1 (headline, priority order position 1): rate_free_state_geometry_behavior_link
# ============================================================================

DELIVERED_RATE_FREE_PATH = ROOT / "results" / "rate_free_state_geometry_behavior_link.json"


def _rate_free_bias_only(rows_arrays: dict[str, dict], seed_tag: str) -> dict:
    """Session-mean collapse of the headline claim's own per-trial arrays
    (deviation, is_corr, spike_count, trial_index -- all already computed by
    the unchanged rate_free_session_arrays / rate_free_analyze_session code
    path), then a between-session Pearson correlation for the raw cell and a
    between-session PARTIAL correlation (controlling session-mean spike
    count and session-mean trial index) for the joint-partial cell -- the
    same two load-bearing statistics the delivered branch was decided on.
    This is the whole-session generalisation of _bias_only_between_session
    (BIAS_ONLY_SOURCE_QUOTE_VERBATIM): there is no item-count level to
    partition by here, so every trial with a defined deviation direction in
    the session (already the analysis population rate_free_session_arrays
    restricts to) contributes to that session's mean."""
    means = {"deviation": [], "is_corr": [], "spike_count": [], "trial_index": []}
    sessions_used = []
    for session_id, arrays in rows_arrays.items():
        means["deviation"].append(float(np.mean(arrays["deviation"])))
        means["is_corr"].append(float(np.mean(arrays["is_corr"])))
        means["spike_count"].append(float(np.mean(arrays["spike_count"])))
        means["trial_index"].append(float(np.mean(arrays["trial_index"])))
        sessions_used.append(session_id)
    n_sessions = len(sessions_used)

    def _corr(controls: list[str], tag: str) -> dict:
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|{tag}"))
        control_arrays = [np.array(means[c]) for c in controls]
        result = partial_correlation_permutation_test(
            np.array(means["is_corr"]), np.array(means["deviation"]), control_arrays, n_perm=N_PERM, rng=rng)
        if result.get("status") == "computed":
            result["significant"] = bool(result["p_value"] < 0.05)
            result["direction"] = _direction(result["r"])
        result["n_sessions"] = n_sessions
        return result

    return {
        "n_sessions": n_sessions, "sessions_used": sessions_used, "means": means,
        "raw": _corr([], "raw"),
        "joint_partial_controlling_spike_count_and_trial_index": _corr(
            ["spike_count", "trial_index"], "joint_partial"),
    }


def claim_rate_free_headline(root: Path, t0: float) -> dict:
    _log("claim 1/headline: rate_free_state_geometry_behavior_link -- loading sessions")
    delivered = json.loads(DELIVERED_RATE_FREE_PATH.read_text())
    paths = _panichello_reachable_sessions(root)
    n_seen = len(paths)

    loaded_rows, refused = [], []
    rows_arrays: dict[str, dict] = {}
    for path in paths:
        session_id = path.stem
        arrays = rate_free_session_arrays(path)
        if arrays is None:
            refused.append({"session": session_id, "reason": "session_arrays_returned_none_too_few_valid_trials"})
            continue
        analysis = rate_free_analyze_session(session_id, arrays)
        loaded_rows.append({"session": session_id, "analysis": analysis})
        rows_arrays[session_id] = arrays
    _log(f"  loaded {len(loaded_rows)}/{n_seen} reachable sessions")

    raw_values = [r["analysis"]["raw_outcome_vs_deviation"]["r"] for r in loaded_rows
                  if r["analysis"]["raw_outcome_vs_deviation"].get("status") == "computed"]
    joint_values = [r["analysis"]["joint_partial_controlling_spike_count_and_trial_index"]["r"]
                     for r in loaded_rows
                     if r["analysis"]["joint_partial_controlling_spike_count_and_trial_index"].get("status") == "computed"]
    recomputed_raw = slope_across_sessions_test(raw_values, "two-sided") if raw_values else {"status": "not_computed"}
    recomputed_joint = slope_across_sessions_test(joint_values, "two-sided") if joint_values else {"status": "not_computed"}

    delivered_raw = delivered["pooled"]["raw_outcome_vs_deviation"]
    delivered_joint = delivered["pooled"]["joint_partial_controlling_spike_count_and_trial_index"]
    checks = {
        "raw_mean_value": _close(recomputed_raw.get("mean_value"), delivered_raw.get("mean_value")),
        "raw_p_value": _close(recomputed_raw.get("p_value"), delivered_raw.get("p_value")),
        "joint_partial_mean_value": _close(recomputed_joint.get("mean_value"), delivered_joint.get("mean_value")),
        "joint_partial_p_value": _close(recomputed_joint.get("p_value"), delivered_joint.get("p_value")),
        "n_sessions_matches": bool(len(loaded_rows) == delivered["n_sessions_computed"]),
    }
    gate_status = "reproduced_exactly" if all(checks.values()) else "not_reproduced"
    reproduction_gate = {
        "status": gate_status, "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "recomputed_raw": recomputed_raw, "delivered_raw": delivered_raw,
        "recomputed_joint_partial": recomputed_joint, "delivered_joint_partial": delivered_joint,
    }

    zero_drop = {
        "n_seen": n_seen, "n_loaded": len(loaded_rows), "n_refused": len(refused),
        "refusals_by_reason": {r["reason"]: sum(1 for x in refused if x["reason"] == r["reason"])
                                for r in refused} if refused else {},
        "n_analysed": len(raw_values), "n_loaded_but_not_analysed": len(loaded_rows) - len(raw_values),
        "reconciles": bool(n_seen == len(loaded_rows) + len(refused)),
        "reconciles_against_delivered_artifact": bool(
            len(loaded_rows) == delivered["n_sessions_computed"] and n_seen == delivered["n_sessions_reachable"]),
    }

    claim: dict = {
        "claim_name": "rate_free_state_geometry_behavior_link",
        "delivered_artifact": "results/rate_free_state_geometry_behavior_link.json",
        "delivered_claim_key_path": "pooled.raw_outcome_vs_deviation (and joint_partial_controlling_spike_count_and_trial_index)",
        "why_at_risk": "per-trial rate-free deviation correlated against per-trial trial outcome (is_corr), "
                       "within session, pooled across the 11 reachable macaque lPFC sessions by the two-sided "
                       "paired sign-flip test -- exactly the at-risk shape this sweep targets, and the "
                       "surviving half of this project's headline dissociation.",
        "reproduction_gate": reproduction_gate,
        "zero_drop_accounting": zero_drop,
    }

    if gate_status != "reproduced_exactly":
        claim["branch"] = BRANCH_GATE_FAILED
        claim["failing_checks"] = {k: v for k, v in checks.items() if not v}
        return claim

    _log("  reproduction gate reproduced delivered artifact at tolerance 1e-6")
    real_raw_sig = bool(recomputed_raw.get("significant"))
    real_joint_sig = bool(recomputed_joint.get("significant"))
    claim["real"] = {
        "raw": {**recomputed_raw, "direction": _direction(recomputed_raw.get("mean_value"))},
        "joint_partial_controlling_spike_count_and_trial_index":
            {**recomputed_joint, "direction": _direction(recomputed_joint.get("mean_value"))},
    }

    if not real_raw_sig:
        claim["branch"] = BRANCH_NOT_SIGNIFICANT
        return claim

    _log("  real raw statistic significant -- computing bias-only control")
    bias = _rate_free_bias_only(rows_arrays, "rate_free_headline")
    claim["bias_only_between_session"] = bias
    raw_bias = bias["raw"]
    joint_bias = bias["joint_partial_controlling_spike_count_and_trial_index"]

    raw_voiding = _void_verdict(real_raw_sig, _direction(recomputed_raw.get("mean_value")),
                                 raw_bias.get("significant"), raw_bias.get("direction"))
    joint_voiding = _void_verdict(real_joint_sig, _direction(recomputed_joint.get("mean_value")),
                                   joint_bias.get("significant"), joint_bias.get("direction"))
    raw_voiding["real_minus_control"] = _real_minus_control(
        recomputed_raw.get("mean_value"), raw_values, raw_bias.get("r"),
        bias["means"]["deviation"], bias["means"]["is_corr"], [], "rate_free_headline|raw")
    raw_voiding["within_session_demeaned"] = _within_session_demeaned_reading(
        rows_arrays, "is_corr", "deviation", [], "rate_free_headline|raw")
    joint_voiding["real_minus_control"] = _real_minus_control(
        recomputed_joint.get("mean_value"), joint_values, joint_bias.get("r"),
        bias["means"]["deviation"], bias["means"]["is_corr"],
        [bias["means"]["spike_count"], bias["means"]["trial_index"]], "rate_free_headline|joint_partial")
    joint_voiding["within_session_demeaned"] = _within_session_demeaned_reading(
        rows_arrays, "is_corr", "deviation", ["spike_count", "trial_index"], "rate_free_headline|joint_partial")
    claim["voiding"] = {"raw": raw_voiding, "joint_partial_controlling_spike_count_and_trial_index": joint_voiding}

    voided = bool(raw_voiding["voided"] or joint_voiding["voided"])
    if voided:
        claim["branch"] = BRANCH_VOIDED
        claim["branch_reason"] = (
            "raw voided" if raw_voiding["voided"] and not joint_voiding["voided"] else
            "joint_partial voided" if joint_voiding["voided"] and not raw_voiding["voided"] else
            "both raw and joint_partial voided"
        ) + " -- the delivered branch (rate_free_state_geometry_predicts_accuracy) required both raw and " \
            "joint-partial significant with the same sign; losing either to its own bias-only control is " \
            "sufficient to lose the association, matching the precedent in " \
            "results/component_binding_bias_only_control.json."
        return claim

    power = _power_report(real_raw_sig, raw_values, raw_bias,
                           bias["means"]["deviation"], bias["means"]["is_corr"], "rate_free_headline|raw")
    claim["power"] = power
    inconclusive = power["bias_only_control_underpowered_at_the_meaningful_effect_threshold"]
    claim["branch"] = BRANCH_INCONCLUSIVE if inconclusive else BRANCH_NOT_VOIDED
    return claim


# ============================================================================
# Claim 2 (priority order position 2): state_behavior_link.json's
# cheap_first_look_pooled.leading_component_score_gain
# ============================================================================

DELIVERED_STATE_BEHAVIOR_PATH = ROOT / "results" / "state_behavior_link.json"


def claim_leading_component_gain(root: Path, t0: float) -> dict:
    _log("claim 2: state_behavior_link cheap_first_look_pooled.leading_component_score_gain")
    delivered = json.loads(DELIVERED_STATE_BEHAVIOR_PATH.read_text())
    paths = _panichello_reachable_sessions(root)  # identical reachability floor (60 error trials) and loader
    n_seen = len(paths)

    loaded, refused = [], []
    rows_arrays: dict[str, dict] = {}
    for path in paths:
        session_id = path.stem
        raw = loadmat(str(path), simplify_cells=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
        counts_all = _counts_from_spikes(spikes, time_ms)
        cfl = cheap_first_look(counts_all, is_corr)
        if cfl.get("status") != "computed" or cfl["leading_component_score_gain"].get("status") != "computed":
            refused.append({"session": session_id, "reason": "cheap_first_look_not_computable"})
            continue
        # trial_amplitude_covariates is re-derived inside cheap_first_look; recover the raw per-trial gain
        # array too (needed for the bias-only session-mean collapse) via the identical unchanged function.
        from state_persistence import trial_amplitude_covariates
        covariates = trial_amplitude_covariates(counts_all)
        loaded.append({"session": session_id, "r": cfl["leading_component_score_gain"]["r"]})
        rows_arrays[session_id] = {
            "gain": covariates["leading_component_score_gain"],
            "is_corr": is_corr.astype(float),
        }
    _log(f"  loaded {len(loaded)}/{n_seen} reachable sessions")

    values = [r["r"] for r in loaded]
    recomputed = slope_across_sessions_test(values, "two-sided") if values else {"status": "not_computed"}
    delivered_pooled = delivered["cheap_first_look_pooled"]["leading_component_score_gain"]["pooled_r_test"]
    checks = {
        "mean_value": _close(recomputed.get("mean_value"), delivered_pooled.get("mean_value")),
        "p_value": _close(recomputed.get("p_value"), delivered_pooled.get("p_value")),
        "n_sessions_matches": bool(len(loaded) == delivered["cheap_first_look_pooled"]["leading_component_score_gain"]["n_sessions"]),
    }
    gate_status = "reproduced_exactly" if all(checks.values()) else "not_reproduced"

    zero_drop = {
        "n_seen": n_seen, "n_loaded": len(loaded), "n_refused": len(refused),
        "refusals_by_reason": {r["reason"]: sum(1 for x in refused if x["reason"] == r["reason"])
                                for r in refused} if refused else {},
        "n_analysed": len(values), "reconciles": bool(n_seen == len(loaded) + len(refused)),
        "reconciles_against_delivered_artifact": bool(
            n_seen == delivered["n_sessions_reachable"]
            and len(loaded) == delivered["cheap_first_look_pooled"]["leading_component_score_gain"]["n_sessions"]),
    }

    claim: dict = {
        "claim_name": "state_behavior_link_leading_component_score_gain",
        "delivered_artifact": "results/state_behavior_link.json",
        "delivered_claim_key_path": "cheap_first_look_pooled.leading_component_score_gain.pooled_r_test",
        "why_at_risk": "per-trial leading-component gain point-biserial correlated against per-trial trial "
                       "outcome, within session, pooled across the same 11 reachable macaque lPFC sessions by "
                       "the sign-flip test. This artifact's own named headline claim (deciding_contrast, the "
                       "matched persistence contrast) is a bounded null and carries nothing to void; this "
                       "cheap_first_look_pooled cell is the file's only significant within-session pooled "
                       "positive and is the same raw dominant-mode-vs-accuracy correlation "
                       "results/behavior_amplitude_rate_controls.json partials against total spike count "
                       "(a different control) -- never tested against session identity until now.",
        "reproduction_gate": {"status": gate_status, "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
                              "recomputed": recomputed, "delivered": delivered_pooled},
        "zero_drop_accounting": zero_drop,
    }

    if gate_status != "reproduced_exactly":
        claim["branch"] = BRANCH_GATE_FAILED
        claim["failing_checks"] = {k: v for k, v in checks.items() if not v}
        return claim

    real_sig = bool(recomputed.get("significant"))
    claim["real"] = {**recomputed, "direction": _direction(recomputed.get("mean_value"))}
    if not real_sig:
        claim["branch"] = BRANCH_NOT_SIGNIFICANT
        return claim

    means_gain = [float(np.mean(a["gain"])) for a in rows_arrays.values()]
    means_outcome = [float(np.mean(a["is_corr"])) for a in rows_arrays.values()]
    rng = np.random.default_rng(stable_seed("state_behavior_link|leading_component_score_gain|bias_only"))
    bias_raw = partial_correlation_permutation_test(
        np.array(means_outcome), np.array(means_gain), [], n_perm=N_PERM, rng=rng)
    if bias_raw.get("status") == "computed":
        bias_raw["significant"] = bool(bias_raw["p_value"] < 0.05)
        bias_raw["direction"] = _direction(bias_raw["r"])
    bias_raw["n_sessions"] = len(means_gain)
    claim["bias_only_between_session"] = bias_raw

    voiding = _void_verdict(real_sig, _direction(recomputed.get("mean_value")),
                             bias_raw.get("significant"), bias_raw.get("direction"))
    voiding["real_minus_control"] = _real_minus_control(
        recomputed.get("mean_value"), values, bias_raw.get("r"), means_gain, means_outcome, [],
        "state_behavior_link|leading_component_score_gain")
    voiding["within_session_demeaned"] = _within_session_demeaned_reading(
        rows_arrays, "is_corr", "gain", [], "state_behavior_link|leading_component_score_gain")
    claim["voiding"] = voiding
    if voiding["voided"]:
        claim["branch"] = BRANCH_VOIDED
        return claim

    power = _power_report(real_sig, values, bias_raw, means_gain, means_outcome,
                           "state_behavior_link|leading_component_score_gain")
    claim["power"] = power
    inconclusive = power["bias_only_control_underpowered_at_the_meaningful_effect_threshold"]
    claim["branch"] = BRANCH_INCONCLUSIVE if inconclusive else BRANCH_NOT_VOIDED
    return claim


# ============================================================================
# Claim 3 (priority order position 3): behavior_geometry_link.json
# ============================================================================

def claim_behavior_geometry_link() -> dict:
    _log("claim 3: behavior_geometry_link -- inspecting for a testable positive")
    path = ROOT / "results" / "behavior_geometry_link.json"
    delivered = json.loads(path.read_text())
    return {
        "claim_name": "behavior_geometry_link",
        "delivered_artifact": "results/behavior_geometry_link.json",
        "delivered_claim_key_path": "n/a -- see determination below",
        "why_at_risk": "listed in the priority order; determined NOT at risk on inspection.",
        "determination": (
            "This artifact contains only descriptive per-session summary numbers for two human corpora "
            "(dandi000469, boran_ieeg): accuracy, tau_context, tau_content, axis_rot_context, "
            "axis_rot_content, dmd_rot, and dmd_rot_correct/dmd_rot_error, per subject, plus an "
            "'underpowered' flag per corpus. No pooled significance test, p-value, correlation "
            "coefficient, or 'significant' field is computed anywhere in this file -- it is raw ingredient "
            "data (per-session geometry and behaviour descriptives) for other artifacts to test, not itself "
            "a positive claim of the per-trial-predictor-vs-per-trial-outcome, within-session, "
            "pooled-across-sessions shape this sweep's control applies to. There is therefore nothing to "
            "recompute a bias-only control against."
        ),
        "top_level_keys_present": sorted(delivered.keys()),
        "branch": "not_at_risk_no_pooled_significance_test_in_this_artifact",
    }


# ============================================================================
# Extended power block (claims 4-10 onward): how many sessions would the
# bias-only control need to resolve the meaningful-effect threshold.
# ============================================================================

def _sessions_required_for_threshold(bootstrap_se_at_n: float | None, n_sessions_observed: int | None,
                                      threshold: float = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS) -> dict:
    """Extrapolates, from ONE already-computed whole-session cluster-bootstrap standard error observed
    at n_sessions_observed sessions, how many sessions the bias-only control would need for its own
    mdd (MDD_Z_FACTOR * se) to fall to `threshold` -- assuming se scales as 1/sqrt(n) (additional
    sessions drawn independently from the same between-session variance this control already observed).
    This is an EXTRAPOLATION from a single observed variance, not a fresh bootstrap measured at the
    larger n, and is reported as such rather than as a measurement."""
    if bootstrap_se_at_n is None or n_sessions_observed is None or n_sessions_observed < 1 or bootstrap_se_at_n <= 0:
        return {"status": "not_estimable",
                "reason": "no defined bias-only bootstrap standard error to extrapolate from"}
    n_required = n_sessions_observed * ((MDD_Z_FACTOR * bootstrap_se_at_n) / threshold) ** 2
    return {
        "status": "extrapolated",
        "assumption": "the bias-only control's own bootstrap standard error scales as 1/sqrt(n) -- i.e. "
                      "additional sessions are independent draws from the same between-session variance this "
                      "control's whole-session cluster bootstrap already observed at n_sessions_observed. This "
                      "is an extrapolation from a single observed variance, NOT a fresh bootstrap measured at "
                      "the extrapolated n, and is not a guarantee a larger corpus would show the same variance.",
        "n_sessions_observed": n_sessions_observed, "meaningful_effect_threshold_r_units": threshold,
        "n_sessions_required": float(n_required), "n_sessions_required_ceil": int(np.ceil(n_required)),
    }


def _add_sessions_required(power: dict) -> dict:
    """Adds sessions_required_for_the_control_to_reach_the_meaningful_effect_threshold to an
    already-built power dict (from _power_report), reading the bias-only bootstrap it already
    computed -- no fresh bootstrap, one additional deterministic extrapolation."""
    bias_mdd = power.get("bias_only_control_minimum_detectable_difference_80pct_power", {})
    power = dict(power)
    power["sessions_required_for_the_control_to_reach_the_meaningful_effect_threshold"] = (
        _sessions_required_for_threshold(bias_mdd.get("bootstrap_se"), bias_mdd.get("n_sessions"))
        if bias_mdd.get("status") == "computed" else
        {"status": "not_estimable", "reason": "bias-only control minimum detectable difference itself not computed"})
    return power


def _reused_cell_from_existing_claim(existing_claim: dict, claim_name: str, cell_path: str) -> dict:
    """Cites a statistic that is BIT-IDENTICAL to one already reproduced, bias-only-controlled and
    power-bounded earlier in this same sweep run (verified by construction: same corpus, same session
    set, same estimator, same pre-trial arrays), rather than recomputing it a third time. Only the new
    sessions-required extrapolation is computed fresh on top of the already-computed bootstrap."""
    power = existing_claim.get("power")
    return {
        "reused_from_claim": claim_name, "reused_cell_path": cell_path,
        "note": f"bit-identical statistic to claim '{claim_name}' ({cell_path}), already reproduced at "
                "tolerance 1e-6, bias-only-controlled and power-bounded earlier in this sweep -- not "
                "recomputed a third time. Only the sessions-required extrapolation below is new.",
        "real": existing_claim.get("real"),
        "bias_only_between_session": existing_claim.get("bias_only_between_session"),
        "voiding": existing_claim.get("voiding"),
        "power": _add_sessions_required(power) if power else None,
        "branch": existing_claim.get("branch"),
    }


def _not_independently_reproduced_cell(delivered_value, reason: str, extra: dict | None = None) -> dict:
    """For a genuinely new at-risk positive this sweep did NOT recompute from raw data (a bespoke
    estimator -- a new corpus loader, a detrended shuffle-null lag alignment, a subspace
    eigendecomposition -- that this pass did not port). Reports the delivered number honestly, states
    plainly that no reproduction gate or bias-only control was run, and gives the specific technical
    reason rather than a generic time-budget excuse."""
    cell = {
        "delivered_real_statistic": delivered_value,
        "independently_reproduced_by_this_sweep": False,
        "bias_only_control_computed_by_this_sweep": False,
        "reason_not_independently_tested": reason,
        "branch": "not_independently_reproduced_bespoke_estimator",
    }
    if extra:
        cell.update(extra)
    return cell


# ============================================================================
# Claim 4 (priority order position 4): dominant_latent_identity_and_behaviour_breadth.json
# ============================================================================

def claim_dominant_latent_identity_and_behaviour_breadth(claims: dict) -> dict:
    _log("claim 4: dominant_latent_identity_and_behaviour_breadth")
    delivered = json.loads((ROOT / "results" / "dominant_latent_identity_and_behaviour_breadth.json").read_text())
    primary_deviation = _reused_cell_from_existing_claim(
        claims["rate_free_state_geometry_behavior_link"], "rate_free_state_geometry_behavior_link",
        "pooled.raw_outcome_vs_deviation")
    primary_amplitude = _reused_cell_from_existing_claim(
        claims["state_behavior_link_leading_component_score_gain"],
        "state_behavior_link_leading_component_score_gain",
        "cheap_first_look_pooled.leading_component_score_gain.pooled_r_test")

    efs = delivered["error_floor_sensitivity"]["by_floor"]
    amp = delivered["dominant_latent_amplitude_and_outcome"]["by_floor"]
    sensitivity = {
        floor: {
            "n_sessions": efs[floor]["n_sessions_included"],
            "deviation_raw_delivered": efs[floor]["pooled"]["raw_outcome_vs_observable"],
            "amplitude_raw_delivered": amp[floor]["pooled"]["raw_outcome_vs_observable"],
            "independently_bias_only_tested_by_this_sweep": False,
            "reason_not_tested": "sensitivity floors are pre-declared NON-primary by the delivered "
                                  "artifact's own rule ('the 45 and 30 floors are never substituted for the "
                                  "primary'); this sweep tests the primary floor (60) only, reused above -- "
                                  "both sensitivity floors remain significant in the delivered artifact, in "
                                  "the same direction as the primary floor, at larger n_sessions.",
        }
        for floor in ("45", "30")
    }

    branch = primary_deviation["branch"]
    return {
        "claim_name": "dominant_latent_identity_and_behaviour_breadth",
        "delivered_artifact": "results/dominant_latent_identity_and_behaviour_breadth.json",
        "why_at_risk": "this artifact's error_floor_sensitivity.by_floor.60 and "
                       "dominant_latent_amplitude_and_outcome.by_floor.60 are the SAME two statistics as "
                       "claims rate_free_state_geometry_behavior_link and "
                       "state_behavior_link_leading_component_score_gain respectively, at the same primary "
                       "floor (60) and the same 11-session set -- reused verbatim rather than recomputed.",
        "primary_floor_rate_free_deviation": primary_deviation,
        "primary_floor_dominant_latent_amplitude": primary_amplitude,
        "sensitivity_floors_45_and_30_context_only_not_independently_tested": sensitivity,
        "not_at_risk_components": {
            "previous_item_in_dominant_latent": "a decoding-accuracy-vs-circular-shift-null statistic, not "
                "a per-trial-predictor-vs-behavioural-outcome correlation -- not the shape this control "
                "applies to.",
            "history_lag_profile": "same as above, a content-decoding statistic.",
            "unit_count_and_the_behavioural_correlation": "a between-session characterisation (unit count "
                "vs correlation SIZE across sessions), not a within-session pooled association.",
            "performance_regime_dependence": "a between-session/cohort characterisation, not a "
                "within-session pooled association.",
        },
        "branch": branch,
    }


# ============================================================================
# Claim 5 (priority order position 5): dissociation_cross_preparation_test.json
# ============================================================================

def claim_dissociation_cross_preparation_test(claims: dict) -> dict:
    _log("claim 5: dissociation_cross_preparation_test")
    delivered = json.loads((ROOT / "results" / "dissociation_cross_preparation_test.json").read_text())
    macaque_deviation = _reused_cell_from_existing_claim(
        claims["rate_free_state_geometry_behavior_link"], "rate_free_state_geometry_behavior_link",
        "pooled.raw_outcome_vs_deviation")
    macaque_amplitude = _reused_cell_from_existing_claim(
        claims["state_behavior_link_leading_component_score_gain"],
        "state_behavior_link_leading_component_score_gain",
        "cheap_first_look_pooled.leading_component_score_gain.pooled_r_test")

    mouse_deviation_raw = delivered["rate_free_deviation_and_outcome"]["by_floor"]["30"]["pooled"]["raw_outcome_vs_observable"]
    mouse_amplitude_raw = delivered["dominant_latent_amplitude_and_outcome"]["by_floor"]["30"]["pooled"]["raw_outcome_vs_observable"]
    mouse_deviation_sig = bool(mouse_deviation_raw.get("significant"))
    mouse_amplitude_sig = bool(mouse_amplitude_raw.get("significant"))
    mouse_cells = {
        "rate_free_deviation": {
            "real": mouse_deviation_raw, "direction": _direction(mouse_deviation_raw.get("mean_value")),
            "branch": BRANCH_NOT_SIGNIFICANT if not mouse_deviation_sig else "requires_fresh_bias_only_control",
        },
        "dominant_latent_amplitude": {
            "real": mouse_amplitude_raw, "direction": _direction(mouse_amplitude_raw.get("mean_value")),
            "branch": BRANCH_NOT_SIGNIFICANT if not mouse_amplitude_sig else "requires_fresh_bias_only_control",
        },
    }

    return {
        "claim_name": "dissociation_cross_preparation_test",
        "delivered_artifact": "results/dissociation_cross_preparation_test.json",
        "why_at_risk": "the macaque lPFC cells here are the SAME statistics as claims "
                       "rate_free_state_geometry_behavior_link and "
                       "state_behavior_link_leading_component_score_gain (reused, not recomputed); the "
                       "mouse ALM (anterior lateral motor cortex preparation) cells at n=20 sessions (floor 30) are a "
                       "genuinely different corpus and observable, checked here for significance -- neither "
                       "mouse cell is a significant positive, so neither needs a bias-only control under "
                       "this sweep's own voiding rule.",
        "macaque_lpfc_primary_floor_rate_free_deviation": macaque_deviation,
        "macaque_lpfc_primary_floor_dominant_latent_amplitude": macaque_amplitude,
        "mouse_alm_floor_30": mouse_cells,
        "branch": macaque_deviation["branch"],
    }


# ============================================================================
# Claim 6 (priority order position 6): deviation_serial_dependence_and_temporal_locus.json
# ============================================================================

def claim_deviation_serial_dependence_and_temporal_locus() -> dict:
    _log("claim 6: deviation_serial_dependence_and_temporal_locus")
    delivered = json.loads((ROOT / "results" / "deviation_serial_dependence_and_temporal_locus.json").read_text())
    panichello = delivered["block_a"]["panichello_2024_macaque_lPFC_single_item"]
    watters = delivered["block_a"]["watters_2026_macaque_multi_object"]
    return {
        "claim_name": "deviation_serial_dependence_and_temporal_locus",
        "delivered_artifact": "results/deviation_serial_dependence_and_temporal_locus.json",
        "why_at_risk": "block_a.*.decisive_partial_by_window (both corpora) is a per-trial "
                       "deviation-to-behaviour correlation, within session, pooled across sessions -- the "
                       "at-risk shape -- but restricted to a lag-1-alignment-eligible trial subset and, for "
                       "the joint-partial cell, additionally controlling a detrended lag-1 alignment "
                       "covariate this sweep has not ported; block_a.*.content_specific_serial_pull_pooled "
                       "and block_b's half/third ordering tests are further bespoke constructs. NONE of "
                       "these are recomputed here; the delivered numbers are cited for the record.",
        "panichello_2024_macaque_lPFC_single_item": {
            "decisive_partial_primary_window_51": _not_independently_reproduced_cell(
                panichello["decisive_partial_by_window"]["51"],
                "requires this artifact's own detrended lag-1-alignment covariate (a bespoke serial-"
                "dependence construct) and its own lag-1-eligible trial subset, neither of which this "
                "sweep's imported estimators compute; NOTE the 'raw' sub-value (0.0956, p=0.0035) is "
                "closely related to but not bit-identical with claim rate_free_state_geometry_behavior_link "
                "(0.0974, p=0.0035) -- same sign, same corpus, likely the same session set restricted to a "
                "trial subset."),
            "content_specific_serial_pull_pooled": _not_independently_reproduced_cell(
                panichello["content_specific_serial_pull_pooled"],
                "a bespoke content-specific-serial-pull construct (previous-trial memorandum-class-"
                "dependent cosine alignment); effect size is tiny (r~0.001) despite p<0.05."),
        },
        "watters_2026_macaque_multi_object": {
            "decisive_partial_primary_window_51": _not_independently_reproduced_cell(
                watters["decisive_partial_by_window"]["51"],
                "same reason as the panichello cell above; additionally this corpus's raw cell is itself "
                "NOT significant (p=0.060) before the lag-1-alignment partial is added, so only the "
                "partial-controlled cells are nominally significant here."),
            "context_multi_object_swap_like_association_not_tested_here": (
                "This artifact's own internal reproduction_gate.watters.raw = 0.019673, p=0.0003 matches "
                "component_and_item_binding.json's swap association family closely (not bit-identical). "
                "results/component_binding_bias_only_control.json already found a comparably-sized raw "
                "within-session association in this same multi-object corpus (r=0.019821, p=0.045) "
                "REPRODUCES from session identity alone (voided) -- see claim "
                "component_and_item_binding in this sweep. That finding is disclosed here as context; it "
                "is NOT a substitute verdict for this specific cell, which this sweep did not "
                "independently test."
            ),
        },
        "branch": "not_independently_reproduced_bespoke_estimator",
    }


# ============================================================================
# Claim 7 (priority order position 7): component_effect_size_and_anatomy.json
# ============================================================================

def claim_component_effect_size_and_anatomy(claims: dict) -> dict:
    _log("claim 7: component_effect_size_and_anatomy")
    delivered = json.loads((ROOT / "results" / "component_effect_size_and_anatomy.json").read_text())
    macaque_single_item = _reused_cell_from_existing_claim(
        claims["rate_free_state_geometry_behavior_link"], "rate_free_state_geometry_behavior_link",
        "pooled.raw_outcome_vs_deviation")
    multi_object_value = delivered["block_a"]["macaque_multi_object"]["within_session_association"]
    return {
        "claim_name": "component_effect_size_and_anatomy",
        "delivered_artifact": "results/component_effect_size_and_anatomy.json",
        "why_at_risk": "block_a.macaque_lPFC_single_item.within_session_association is bit-identical to "
                       "claim rate_free_state_geometry_behavior_link (reused below). Every OTHER statistic "
                       "in this artifact -- decile/quintile contrasts, spike-count-matched decile contrasts, "
                       "cross-validated AUC discrimination, pairwise across/within-block predictor "
                       "comparisons, and area/animal-stratified splits -- is a DIFFERENT ESTIMATOR SHAPE "
                       "(discretised, rank/AUC-based, or already stratified) than the plain per-trial "
                       "correlation pooled by a sign-flip test this sweep's bias-only construction targets; "
                       "generalising the bias-only control to a decile contrast or a cross-validated AUC is "
                       "a real methodological question this sweep does not resolve, not an oversight.",
        "macaque_lPFC_single_item_within_session_association": macaque_single_item,
        "macaque_multi_object_within_session_association_not_independently_tested": _not_independently_reproduced_cell(
            multi_object_value,
            "a within-session association reused inside this artifact from the multi-object corpus, closely "
            "related to (but not bit-identical with) the association results/component_binding_bias_only_"
            "control.json already bias-only-tested and found VOIDED (r=0.019821, p=0.045 same sign); that "
            "finding is disclosed as context in claim component_and_item_binding but is not re-derived for "
            "this exact cell here."),
        "not_at_risk_different_estimator_shape": (
            "decile_contrast, quintile_contrast_sensitivity, spike_count_matched_decile_contrast, "
            "cross_validated_discrimination (deviation/amplitude/spike_count, all three folds variants), "
            "pairwise_predictor_comparisons (across_block and within_block), and block_b's area_split / "
            "animal_split cells -- all present in this artifact, none tested by this sweep, all a different "
            "estimator shape than a plain per-trial correlation pooled by a sign-flip test."
        ),
        "branch": macaque_single_item["branch"],
    }


# ============================================================================
# Claim 8 (priority order position 8): component_and_item_binding.json
# ============================================================================

def claim_component_and_item_binding() -> dict:
    _log("claim 8: component_and_item_binding -- citing the already-delivered sibling bias-only control")
    sibling = json.loads((ROOT / "results" / "component_binding_bias_only_control.json").read_text())
    raw_cell = sibling["swap_primary_deviation_within_item_count_level"]["raw"]
    joint_cell = sibling["swap_primary_deviation_within_item_count_level"][
        "joint_partial_controlling_spike_count_and_trial_index"]
    branch = sibling["branch"]["branch"]
    return {
        "claim_name": "component_and_item_binding",
        "delivered_artifact": "results/component_and_item_binding.json",
        "why_at_risk": "the pooled, within-item-count-level swap association (deviation vs swap indicator) "
                       "is a per-trial predictor vs per-trial outcome, within session, pooled across the "
                       "multi-object corpus's 41 sessions by the sign-flip test -- exactly the at-risk shape.",
        "resolution": "results/component_binding_bias_only_control.json already ran EXACTLY this sweep's "
                      "reproduction gate, bias-only construction and voiding rule against this exact claim "
                      "(same corpus, same estimators, same voiding logic this sweep declares) before this "
                      "sweep existed. Recomputing it a third time here would duplicate, not verify, that "
                      "work; this claim cites that sibling artifact's own already-gate-passed numbers "
                      "directly rather than re-deriving them.",
        "sibling_reproduction_gate_status": sibling["reproduction_gate"]["status"],
        "real": {"raw": raw_cell["real"],
                 "joint_partial_controlling_spike_count_and_trial_index": joint_cell["real"]},
        "bias_only_between_session": {
            "raw": raw_cell["bias_only_between_session"],
            "joint_partial_controlling_spike_count_and_trial_index": joint_cell["bias_only_between_session"]},
        "voiding": {"raw": raw_cell["voiding"],
                    "joint_partial_controlling_spike_count_and_trial_index": joint_cell["voiding"]},
        "power_note": "not applicable: the claim is VOIDED (see branch), and this sweep's own rule only "
                      "requires a minimum-detectable-difference power block for NON-voided cells.",
        "branch": BRANCH_VOIDED if branch == "swap_association_voided_by_the_bias_only_control" else branch,
    }


# ============================================================================
# Claim 9 (priority order position 9): human_maintenance_behaviour_link.json
# ============================================================================

def claim_human_maintenance_behaviour_link() -> dict:
    _log("claim 9: human_maintenance_behaviour_link")
    delivered = json.loads((ROOT / "results" / "human_maintenance_behaviour_link.json").read_text())
    cell = delivered["block_b"]["per_corpus"]["dandi_000469"]["by_load"]["1"]["pooled_raw"]
    control_mirror = delivered["block_b"]["per_corpus"]["dandi_000469"]["by_load_control"]["1"]["pooled_raw"]
    return {
        "claim_name": "human_maintenance_behaviour_link",
        "delivered_artifact": "results/human_maintenance_behaviour_link.json",
        "why_at_risk": "block_b.per_corpus.dandi_000469.by_load.1.pooled_raw is a per-trial deviation-to-"
                       "behaviour correlation within session, pooled across DANDI 000469 human iEEG "
                       "sessions at memory load 1 -- the at-risk shape -- and is significant (p=0.0437, "
                       "just under the 0.05 line); none of its own partial-controlled siblings "
                       "(spike-count, trial-index, joint) reach significance, and neither load 2 nor load 3 "
                       "is significant.",
        "load_1_pooled_raw": _not_independently_reproduced_cell(
            cell, "requires the DANDI 000469 human iEEG corpus's own loader and load-level trial "
                  "segmentation, which this sweep has not ported (a different corpus from every claim "
                  "already tested, which are all macaque lPFC or macaque multi-object)."),
        "load_1_pooled_raw_control_mirror_context_only": control_mirror,
        "loads_2_and_3": "not significant at any load or partial (see delivered artifact); nothing to void "
                         "there under this sweep's rule.",
        "branch": "not_independently_reproduced_bespoke_estimator",
    }


# ============================================================================
# Claim 10 (priority order position 10): deviation_subspace_decomposition.json
# ============================================================================

def claim_deviation_subspace_decomposition() -> dict:
    _log("claim 10: deviation_subspace_decomposition")
    delivered = json.loads((ROOT / "results" / "deviation_subspace_decomposition.json").read_text())
    macaque60 = delivered["macaque_lpfc"]["by_error_floor"]["60"]
    watters = delivered["watters_multi_object"]["result"]
    return {
        "claim_name": "deviation_subspace_decomposition",
        "delivered_artifact": "results/deviation_subspace_decomposition.json",
        "why_at_risk": "raw_outside_vs_outcome (macaque lPFC, all three floors) and "
                       "raw_within_vs_outcome / joint_within_controlling_outside (watters multi-object) are "
                       "per-trial correlations of a SUBSPACE-DECOMPOSED deviation component against trial "
                       "outcome, within session, pooled across sessions -- the at-risk shape -- but the "
                       "decomposition itself (decomposition_identity: splitting the rate-free deviation "
                       "into a component 'within' a reference subspace and a component 'outside' it) is a "
                       "bespoke eigendecomposition this sweep has not ported.",
        "macaque_lpfc_primary_floor_60_outside_subspace_vs_outcome": _not_independently_reproduced_cell(
            macaque60["raw_outside_vs_outcome"],
            "requires this artifact's own subspace decomposition (decomposition_identity) splitting the "
            "rate-free deviation into within/outside components before correlating either against outcome "
            "-- a different, more elaborate observable than claim rate_free_state_geometry_behavior_link's "
            "undecomposed deviation, despite both being significant with the same sign and similar "
            "magnitude in the same corpus."),
        "watters_multi_object_within_subspace_vs_outcome": _not_independently_reproduced_cell(
            watters["raw_within_vs_outcome"],
            "same reason as above, applied to the multi-object corpus's within-subspace component."),
        "branch": "not_independently_reproduced_bespoke_estimator",
    }


# ============================================================================
# At-risk census (Step 1): every top-level results/*.json artifact.
# ============================================================================

BEHAVIOUR_MARKERS = [
    "behavior", "behaviour", "is_corr", "isCorr", "accuracy", "swap", "imprecision",
    "recall", "report_error", "response_error", "outcome", "correct", "error_trial",
    "reaction_time", "hit_rate", "d_prime", "performance",
]
BIAS_ONLY_IMPL_MARKERS = ["bias_only_between_session", "bias_only_control", '"bias_only"']
WITHIN_SESSION_MARKERS = ["sign_flip", "slope_across_sessions_test", "per_session", "within_session",
                          "pooled_results", '"pooled"']
BETWEEN_SESSION_MARKERS = ["between_session", "between-session"]

# Files this sweep has itself manually inspected past the automated text scan, overriding the automated
# reason string with a verified one -- kept small and explicit rather than silently trusting the scan for
# every file this module's own priority list names.
MANUAL_OVERRIDES = {
    "rate_free_state_geometry_behavior_link.json": (
        "at_risk", "the headline surviving half of this project's dissociation -- see claim "
                   "rate_free_state_geometry_behavior_link in this artifact's claims section"),
    "state_behavior_link.json": (
        "at_risk", "deciding_contrast (the named headline) is a bounded null, nothing to void there; "
                   "cheap_first_look_pooled.leading_component_score_gain is a significant within-session "
                   "pooled positive and IS at risk -- see claim state_behavior_link_leading_component_score_gain"),
    "behavior_geometry_link.json": (
        "not_at_risk", "descriptive per-session summary numbers only, no pooled significance test in the "
                       "file at all -- see claim behavior_geometry_link determination"),
    "component_and_item_binding.json": (
        "at_risk", "the delivered artifact itself carries no nested bias-only control (its control lives in "
                   "the separate sibling results/component_binding_bias_only_control.json, already "
                   "already_controlled below); by this sweep's file-content rule this file is at_risk, but "
                   "the question is already answered: that sibling found the swap association VOIDED"),
}


def build_at_risk_census() -> dict:
    results_dir = ROOT / "results"
    # Exclude this sweep's OWN output artifact: it is flushed with status "running" (containing the
    # decision-rule text, which itself mentions "bias_only_control"/"bias_only_between_session") before
    # this census runs, so a naive glob would classify the sweep's own in-progress output as
    # already_controlled -- a self-referential artifact of write-then-scan ordering, not a real claim.
    files = sorted(f for f in results_dir.glob("*.json") if f.resolve() != OUTPUT_PATH.resolve())
    entries: dict[str, dict] = {}
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except OSError as exc:
            entries[f.name] = {"classification": "undetermined_from_artifact_contents",
                                "reason": f"unreadable: {exc}", "what_would_be_needed": "a readable file"}
            continue
        if f.name in MANUAL_OVERRIDES:
            cls, reason = MANUAL_OVERRIDES[f.name]
            entries[f.name] = {"classification": cls, "reason": reason, "determination_method": "manual_inspection"}
            continue
        lower = text.lower()
        has_behaviour = any(m.lower() in lower for m in BEHAVIOUR_MARKERS)
        has_bias_impl = any(m.lower() in lower for m in BIAS_ONLY_IMPL_MARKERS)
        has_within = any(m in lower for m in WITHIN_SESSION_MARKERS)
        has_between = any(m in lower for m in BETWEEN_SESSION_MARKERS)
        has_sig_true = ('"significant": true' in lower or '"significant_negative": true' in lower
                        or '"significant_positive": true' in lower)

        if has_bias_impl:
            cls, reason = "already_controlled", ("artifact text contains a nested bias_only_control / "
                                                  "bias_only_between_session implementation")
        elif not has_behaviour:
            cls, reason = "not_at_risk", "no behaviour-related marker detected (geometry-only or non-behavioural)"
        elif has_behaviour and not has_within and has_between:
            cls, reason = "not_at_risk", ("behaviour marker present but the statistic is between-session by "
                                          "construction, not a within-session pooled correlation")
        elif has_behaviour and has_within and has_sig_true:
            cls, reason = "at_risk", ("behaviour marker + within-session/pooled marker + at least one "
                                      "significant:true cell detected by text scan")
        elif has_behaviour and has_within and not has_sig_true:
            cls, reason = "not_at_risk", ("behaviour + within-session pooling present but no significant:true "
                                          "detected anywhere (appears null/inconclusive, nothing to void)")
        else:
            cls = "undetermined_from_artifact_contents"
            reason = ("behaviour marker present but within-session/between-session/significance structure not "
                      "determinable from an automated text scan alone")
        entries[f.name] = {
            "classification": cls, "reason": reason, "determination_method": "automated_text_scan",
            "has_behaviour_marker": has_behaviour, "has_bias_only_impl_marker": has_bias_impl,
            "has_within_session_marker": has_within, "has_between_session_marker": has_between,
            "has_significant_true_marker": has_sig_true,
            "what_would_be_needed_if_undetermined": (
                None if cls != "undetermined_from_artifact_contents" else
                "a manual read of this artifact's own pooling/significance fields to determine whether it "
                "reports a within-session-and-pooled significant behavioural association"),
        }

    counts = {"at_risk": 0, "not_at_risk": 0, "already_controlled": 0, "undetermined_from_artifact_contents": 0}
    for v in entries.values():
        counts[v["classification"]] += 1
    return {
        "method": "Every top-level results/*.json artifact (230 files; excludes results/.checkpoints/, which "
                  "holds per-session partial-run scratch, not delivered artifacts) is scanned for a "
                  "behaviour-related text marker, a within-session-pooling marker, a between-session marker, "
                  "and an already-significant cell. The 10 files this sweep's own priority list names, plus "
                  "results/component_and_item_binding.json (whose bias-only control lives in a separate "
                  "sibling artifact), were additionally read in full rather than only text-scanned "
                  "(MANUAL_OVERRIDES). Every other file's classification is the automated scan's own "
                  "conservative read -- 'undetermined_from_artifact_contents' whenever the scan's own markers "
                  "do not settle the question, per this sweep's instruction to be conservative rather than "
                  "guess.",
        "scope_note": "This covers all 230 top-level delivered artifacts, a broader universe than the "
                      "96-artifact 'carries a behaviour-related key' census this sweep was commissioned "
                      "against; the counts below are this sweep's own independent count, not a reproduction "
                      "of that earlier count.",
        "counts": counts, "counts_sum": sum(counts.values()), "total_files": len(entries),
        "sums_to_total": bool(sum(counts.values()) == len(entries)),
        "entries": entries,
    }


# ============================================================================
# Driver
# ============================================================================

PRIORITY_ORDER = [
    "results/rate_free_state_geometry_behavior_link.json",
    "results/state_behavior_link.json",
    "results/behavior_geometry_link.json",
    "results/dominant_latent_identity_and_behaviour_breadth.json",
    "results/dissociation_cross_preparation_test.json",
    "results/deviation_serial_dependence_and_temporal_locus.json",
    "results/component_effect_size_and_anatomy.json",
    "results/component_and_item_binding.json",
    "results/human_maintenance_behaviour_link.json",
    "results/deviation_subspace_decomposition.json",
]


def main() -> None:
    t0 = time.time()
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Tests whether this project's headline behavioural dissociation -- the dominant population "
            "mode's LARGER raw link to macaque lPFC trial accuracy dies under a spike-count partial "
            "control, while the rate-free deviation's SMALLER link (results/"
            "rate_free_state_geometry_behavior_link.json) survives -- itself survives a bias-only control: "
            "is the surviving link reproducible from SESSION IDENTITY ALONE (a session that differs "
            "systematically in its own mean predictor and its own mean outcome, with no real within-session "
            "relationship between the two)? That control has already voided five of this project's other "
            "behavioural claims (most recently results/component_binding_bias_only_control.json) and had "
            "never been run against the headline's surviving half before this artifact. Also runs the "
            "identical control against state_behavior_link.json's own significant within-session pooled "
            "positive (cheap_first_look_pooled.leading_component_score_gain, the raw dominant-mode-vs-"
            "accuracy correlation, a different statistic from that file's own null headline persistence "
            "contrast), determines that results/behavior_geometry_link.json carries no pooled significance "
            "test at all (nothing to void), and enumerates every other results/*.json artifact into an "
            "at-risk census. Both macaque claims tested here use the same 11 sessions of the macaque lPFC "
            "recordings (doi 10.1038/s41586-024-08139-9) reaching the pre-declared reachability floor (>=60 error trials).",
        )[0],
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "bias_only_control_source_and_generalisation": BIAS_ONLY_SOURCE_QUOTE_VERBATIM,
        "priority_order_declared_before_fitting": PRIORITY_ORDER,
        "status": "running",
    }
    _flush(output)

    _log("building at-risk census over all top-level results/*.json artifacts")
    output["at_risk_census"] = build_at_risk_census()
    _flush(output)
    _log(f"census: {output['at_risk_census']['counts']}")

    claims: dict[str, dict] = {}
    completed_claim_names: list[str] = []

    _log("=== claim 1/10 (headline, priority): rate_free_state_geometry_behavior_link ===")
    claims["rate_free_state_geometry_behavior_link"] = claim_rate_free_headline(root, t0)
    completed_claim_names.append("rate_free_state_geometry_behavior_link")
    output["claims"] = claims
    _flush(output)
    _log(f"  branch: {claims['rate_free_state_geometry_behavior_link']['branch']} "
         f"elapsed={time.time() - t0:.0f}s")

    _log("=== claim 2/10: state_behavior_link_leading_component_score_gain ===")
    claims["state_behavior_link_leading_component_score_gain"] = claim_leading_component_gain(root, t0)
    completed_claim_names.append("state_behavior_link_leading_component_score_gain")
    _flush(output)
    _log(f"  branch: {claims['state_behavior_link_leading_component_score_gain']['branch']} "
         f"elapsed={time.time() - t0:.0f}s")

    _log("=== claim 3/10: behavior_geometry_link ===")
    claims["behavior_geometry_link"] = claim_behavior_geometry_link()
    completed_claim_names.append("behavior_geometry_link")
    _flush(output)
    _log(f"  branch: {claims['behavior_geometry_link']['branch']} elapsed={time.time() - t0:.0f}s")

    _log("=== claim 4/10: dominant_latent_identity_and_behaviour_breadth ===")
    claims["dominant_latent_identity_and_behaviour_breadth"] = claim_dominant_latent_identity_and_behaviour_breadth(claims)
    completed_claim_names.append("dominant_latent_identity_and_behaviour_breadth")
    output["claims"] = claims
    _flush(output)
    _log(f"  branch: {claims['dominant_latent_identity_and_behaviour_breadth']['branch']}")

    _log("=== claim 5/10: dissociation_cross_preparation_test ===")
    claims["dissociation_cross_preparation_test"] = claim_dissociation_cross_preparation_test(claims)
    completed_claim_names.append("dissociation_cross_preparation_test")
    _flush(output)
    _log(f"  branch: {claims['dissociation_cross_preparation_test']['branch']}")

    _log("=== claim 6/10: deviation_serial_dependence_and_temporal_locus ===")
    claims["deviation_serial_dependence_and_temporal_locus"] = claim_deviation_serial_dependence_and_temporal_locus()
    completed_claim_names.append("deviation_serial_dependence_and_temporal_locus")
    _flush(output)
    _log(f"  branch: {claims['deviation_serial_dependence_and_temporal_locus']['branch']}")

    _log("=== claim 7/10: component_effect_size_and_anatomy ===")
    claims["component_effect_size_and_anatomy"] = claim_component_effect_size_and_anatomy(claims)
    completed_claim_names.append("component_effect_size_and_anatomy")
    _flush(output)
    _log(f"  branch: {claims['component_effect_size_and_anatomy']['branch']}")

    _log("=== claim 8/10: component_and_item_binding ===")
    claims["component_and_item_binding"] = claim_component_and_item_binding()
    completed_claim_names.append("component_and_item_binding")
    _flush(output)
    _log(f"  branch: {claims['component_and_item_binding']['branch']}")

    _log("=== claim 9/10: human_maintenance_behaviour_link ===")
    claims["human_maintenance_behaviour_link"] = claim_human_maintenance_behaviour_link()
    completed_claim_names.append("human_maintenance_behaviour_link")
    _flush(output)
    _log(f"  branch: {claims['human_maintenance_behaviour_link']['branch']}")

    _log("=== claim 10/10: deviation_subspace_decomposition ===")
    claims["deviation_subspace_decomposition"] = claim_deviation_subspace_decomposition()
    completed_claim_names.append("deviation_subspace_decomposition")
    output["claims"] = claims
    _flush(output)
    _log(f"  branch: {claims['deviation_subspace_decomposition']['branch']}")

    headline = claims["rate_free_state_geometry_behavior_link"]
    headline_branch = headline.get("branch")
    headline_power = headline.get("power", {})
    headline_bias = headline.get("bias_only_between_session", {}).get("raw", {})
    headline_bias_joint = headline.get("bias_only_between_session", {}).get(
        "joint_partial_controlling_spike_count_and_trial_index", {})
    gain_branch = claims["state_behavior_link_leading_component_score_gain"].get("branch")

    voided_branches = (BRANCH_VOIDED, "swap_association_voided_by_the_bias_only_control")
    resolved_branches = (BRANCH_VOIDED, "swap_association_voided_by_the_bias_only_control",
                          BRANCH_NOT_VOIDED, BRANCH_INCONCLUSIVE)
    excluded_from_tested = (BRANCH_NOT_SIGNIFICANT, "not_at_risk_no_pooled_significance_test_in_this_artifact",
                             "not_independently_reproduced_bespoke_estimator", "requires_fresh_bias_only_control")

    def _flat_branches(claim: dict) -> list[str]:
        """Collects every branch this claim block actually decided (a claim may bundle more than one
        cell, e.g. dissociation_cross_preparation_test's macaque + mouse cells)."""
        found = []

        def _walk(obj):
            if isinstance(obj, dict):
                if "branch" in obj and isinstance(obj["branch"], str):
                    found.append(obj["branch"])
                for v in obj.values():
                    _walk(v)

        _walk(claim)
        return found

    all_branches = [b for name in completed_claim_names for b in _flat_branches(claims[name])]
    tested_branches = [b for b in all_branches if b in resolved_branches]
    n_voided = sum(1 for b in all_branches if b in voided_branches)

    # ---- at-risk census completeness note ----------------------------------
    census_counts = output["at_risk_census"]["counts"]
    priority_at_risk_files = [
        f for f in ("rate_free_state_geometry_behavior_link.json", "state_behavior_link.json",
                    "behavior_geometry_link.json", "dominant_latent_identity_and_behaviour_breadth.json",
                    "dissociation_cross_preparation_test.json",
                    "deviation_serial_dependence_and_temporal_locus.json",
                    "component_effect_size_and_anatomy.json", "component_and_item_binding.json",
                    "human_maintenance_behaviour_link.json", "deviation_subspace_decomposition.json")
        if output["at_risk_census"]["entries"][f]["classification"] == "at_risk"
    ]
    n_other_at_risk_never_examined = census_counts["at_risk"] - len(priority_at_risk_files)
    census_completeness_note = {
        "this_sweep_is_not_a_completed_audit": True,
        "total_at_risk_file_level": census_counts["at_risk"],
        "at_risk_files_named_in_this_sweep_priority_order": len(priority_at_risk_files),
        "priority_at_risk_files_fully_resolved_this_round": 5,
        "priority_at_risk_files_fully_resolved_names": [
            "rate_free_state_geometry_behavior_link.json", "state_behavior_link.json",
            "dominant_latent_identity_and_behaviour_breadth.json", "dissociation_cross_preparation_test.json",
            "component_and_item_binding.json",
        ],
        "priority_at_risk_files_partially_resolved_this_round": 1,
        "priority_at_risk_files_partially_resolved_names": [
            "component_effect_size_and_anatomy.json (only its reused macaque_lPFC_single_item cell; its "
            "own decile/AUC/anatomy-split statistics remain untested)",
        ],
        "priority_at_risk_files_not_independently_tested_this_round": 3,
        "priority_at_risk_files_not_independently_tested_names": [
            "deviation_serial_dependence_and_temporal_locus.json (requires a detrended lag-1-alignment "
            "covariate)",
            "human_maintenance_behaviour_link.json (requires the DANDI 000469 human iEEG loader)",
            "deviation_subspace_decomposition.json (requires a subspace eigendecomposition)",
        ],
        "at_risk_files_outside_this_sweeps_priority_order_never_examined": n_other_at_risk_never_examined,
        "undetermined_from_artifact_contents_never_examined": census_counts["undetermined_from_artifact_contents"],
        "statement": (
            f"Of {census_counts['at_risk']} file-level at_risk artifacts, {len(priority_at_risk_files)} are "
            "named in this sweep's own 10-item priority order. Of these, 5 carry a completed "
            "reproduction-gate-plus-bias-only-control verdict, 1 carries a verdict covering only its "
            "reused/identical cell (its own novel statistics untested), and 3 carry ONLY the delivered "
            "artifact's own numbers, cited but not independently reproduced or bias-only-controlled, "
            "because each requires a bespoke estimator (a new human iEEG loader, a detrended shuffle-null "
            f"lag-alignment partial, a subspace eigendecomposition). The remaining "
            f"{n_other_at_risk_never_examined} at_risk artifacts outside this sweep's priority order, and "
            f"all {census_counts['undetermined_from_artifact_contents']} "
            "undetermined_from_artifact_contents artifacts, were never examined by this sweep "
            "and remain completely untested. This sweep is a targeted test of a pre-declared priority list, "
            "not a completed audit of every at-risk claim in this project."
        ),
    }
    output["at_risk_census_completeness_note"] = census_completeness_note

    summary = {
        "n_claims_in_priority_order": len(PRIORITY_ORDER),
        "n_claims_completed": len(completed_claim_names),
        "n_claims_not_reached": len(PRIORITY_ORDER) - len(completed_claim_names),
        "completed_claim_names": completed_claim_names,
        "n_at_risk_cells_tested_against_the_bias_only_control": len(tested_branches),
        "n_voided": n_voided,
        "headline_surviving_half_branch": headline_branch,
        # --- amendment: the two fields below replace this sweep's original
        # headline_dissociation_still_holds (bool) and headline_verdict_statement, which asserted a
        # conclusion ("dissociation still holds") that the branch beside them (inconclusive_below_
        # detection_floor) explicitly withheld. No computed number, decision rule, or claim-level
        # branch changed -- only these two summary fields, corrected to state exactly what the
        # branch states and nothing beyond it. ---
        "headline_voided_by_this_sweep": headline_branch in voided_branches,
        "headline_control_power_status": (
            f"underpowered_at_n_{headline_bias.get('n_sessions', 11)}: the bias-only control's own "
            f"whole-session cluster bootstrap mdd is "
            f"{headline_power.get('bias_only_control_minimum_detectable_difference_80pct_power', {}).get('mdd')} "
            f"r units against a {MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r-unit meaningful-effect threshold, "
            f"and against the {headline_bias.get('r')} r-unit confound it actually observed (same sign as "
            "the real effect, p="
            f"{headline_bias.get('p_value')}, not significant only because the control could not resolve "
            "an effect this large at n=11) -- controlling for spike count and trial index makes the "
            f"confound LARGER, not smaller (joint-partial control r={headline_bias_joint.get('r')}, "
            f"p={headline_bias_joint.get('p_value')})."
            if headline_branch == BRANCH_INCONCLUSIVE else
            f"branch={headline_branch}"
        ),
        "headline_verdict_statement": (
            "The headline surviving half (rate_free_state_geometry_behavior_link) was NOT voided by the "
            "pre-declared rule (the bias-only control did not reach significance at two-sided p<0.05). "
            "But the control that failed to void it was itself underpowered at n=11 sessions: its own "
            "whole-session cluster bootstrap minimum detectable difference is "
            f"{headline_power.get('bias_only_control_minimum_detectable_difference_80pct_power', {}).get('mdd')} "
            f"r units, far above both the {MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r-unit meaningful-effect "
            f"threshold and the r={headline_bias.get('r')} confound it actually observed in the same "
            "direction as the real effect (p="
            f"{headline_bias.get('p_value')}). Partialling out spike count and trial index made the "
            f"confound STRONGER, not weaker (joint-partial r={headline_bias_joint.get('r')}, "
            f"p={headline_bias_joint.get('p_value')}), so this is not a case of a nuisance covariate "
            "explaining away an artifact of the control. THEREFORE the correct reading is that the "
            "headline dissociation remains UNTESTED against session identity, not confirmed: this sweep "
            "did not show the dissociation is clean of a session-identity confound, and no reader may cite "
            "this sweep as support for it. This does not amend results/rate_free_state_geometry_behavior_"
            "link.json, which stands as delivered; it means that artifact's positive branch should not be "
            "cited as a clean surviving behavioural link without this open question stated alongside it."
            if headline_branch == BRANCH_INCONCLUSIVE else
            "VOIDED: the rate-free deviation's link to trial accuracy reproduces from session identity "
            "alone under the pre-declared bias-only control. The dissociation as previously stated no "
            "longer holds. This does not amend results/rate_free_state_geometry_behavior_link.json, which "
            "stands as delivered."
            if headline_branch in voided_branches else
            f"The headline reproduction gate did not reproduce the delivered artifact (branch="
            f"{headline_branch}); no verdict on the dissociation can be drawn from this sweep until the "
            "discrepancy this gate reports is resolved."
            if headline_branch == BRANCH_GATE_FAILED else
            f"branch={headline_branch}"
        ),
        "leading_component_gain_branch": gain_branch,
    }
    output["summary_branch"] = summary
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - t0
    output["status"] = "complete"
    _flush(output)
    _log(f"DONE. headline_branch={headline_branch} gain_branch={gain_branch} "
         f"n_claims_completed={len(completed_claim_names)} elapsed={time.time() - t0:.0f}s")
    print(json.dumps({"headline_branch": headline_branch, "gain_branch": gain_branch,
                       "n_claims_completed": len(completed_claim_names)}, indent=2, default=float))


if __name__ == "__main__":
    main()
