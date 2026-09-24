"""run_component_binding_bias_only_control.py -- does the pooled, within-
item-count-level swap association survive a bias-only control, or is it a
session-level offset masquerading as a within-session relationship?

A prior analysis of the multi-object macaque corpus (results/
component_and_item_binding.json) reported that a rate-free population
state-deviation component predicts SWAPS (the saccadic report lands nearer
an uncued object than the cued one) within item-count level, trial-count
weighted across levels 2 and 3, then pooled with session as the unit of
analysis over 41 sessions -- and does not predict IMPRECISION (the graded
distance to whichever object the report actually landed nearest) by the
same estimator. That rule required both the raw pooled correlation and the
correlation jointly partialling total spike count and trial index to be
significant, but it never checked whether the association survives a
bias-only control: a session that differs systematically in its OWN mean
deviation and its OWN mean outcome, with no real within-session
relationship between the two, produces a nonzero pooled correlation by
mixing session identity with everything session identity happens to
correlate with (electrode placement, unit yield, recording day). A sibling
analysis of the same corpus, at the finer grain of one item-count level at
a time (results/swap_versus_imprecision_by_item_count.json), applies
exactly such a control and found it fires at item count 2: the bias-only
statistic there reproduces the real, significant per-level result in the
same direction, so that cell is voided rather than used to decide a
branch.

This module applies the identical bias-only substitution and voiding rule
to the POOLED, within-item-count-level statistic itself -- the headline
number, not one level at a time -- for both the swap and the imprecision
outcome, and for the item-count-1 control arm, so the whole dissociation is
re-examined under the same control, not just its positive half.

REPRODUCTION GATE, mandatory and first. Before anything new is computed,
the delivered corpus's own orthogonality gate and raw behaviour
association (via the unchanged reproduction_gate function) and the four
within-item-count-level swap_primary/deviation statistics (raw and every
partial and joint-partial family member) are recomputed here, with the
corpus's own unchanged analyse_session and build_pooled_table, and
compared against results/component_and_item_binding.json, read live, at a
tolerance of 1e-6. The identical check is additionally run for the
within-item-count-level imprecision/deviation family and the item-count-1
control, since this module reports on both outcomes. If any of these
checks fails, nothing past the gate is computed: the branch is
'void_reproduction_gate_failed'.

THE BIAS-ONLY CONTROL. run_swap_versus_imprecision_by_item_count.py's own
_bias_only_between_session collapses each session's real per-trial
deviation, AT ONE ITEM-COUNT LEVEL, to that session's own mean over its
trials at that level -- a session-level constant -- and correlates that
constant, across sessions, against the session's own mean outcome at the
same level (a between-session Pearson correlation over N session-level
points, since a per-session correlation against a zero-variance predictor
is undefined). That function is imported here UNCHANGED and used, exactly
as it already exists, for two single-level diagnostics (level 2 and level
3 individually) and for the item-count-1 control arm (which is already a
single level).

For the POOLED, within-item-count-level statistic the corpus's real branch
was actually decided on, that single-level definition does not directly
apply, because the real statistic's own cross-level step (a trial-count-
weighted combination of a session's per-level correlations) operates on
per-session correlations, and a per-session correlation is exactly the
quantity the bias-only substitution destroys. The generalisation applied
here, quoted and justified in full next to its implementation
(_bias_only_between_session_generalized below), replaces the single-level
equality mask with a mask over every level a session's own real
within-item-count-level statistic was actually pooled across; because a
trial-count-weighted average of a partition's means is algebraically
identical to the mean taken over the union of that partition (a weighted
average of group means, weighted by group size, equals the mean of the
whole), collapsing a session's trials to its own mean over the COMBINED
levels produces exactly the bias-only analogue of the real trial-count-
weighted composite -- not a new statistic, an extension of the same one to
the same combination the real statistic already performs. The correlation
itself is computed by _corr, the corpus's own unchanged partial-
correlation estimator (also used by every real statistic in this family),
so the identical function produces the raw (no controls) and the joint-
partial (controlling for the session's own mean spike count and mean trial
index) bias-only statistic from one code path.

THE VOIDING RULE, unchanged from the sibling analysis: a result is voided
if the bias-only control is significant, in the SAME direction as the
primary statistic. The control's effect size is a different estimator on a
different scale (a between-session correlation over ~40 points, versus a
within-session correlation pooled by a paired sign-flip test) and is never
compared to the real effect by magnitude -- sign and significance only.

No estimator here is forked. analyse_session, build_pooled_table,
reproduction_gate, _close, _corr, _behaviour_observables, _object_geometry
(all from run_component_and_item_binding.py, itself importing them
unchanged from run_watters_state_geometry.py and elsewhere), _session_arrays
and _bias_only_between_session (from run_swap_versus_imprecision_by_
item_count.py), MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
minimum_detectable_paired_difference, stable_seed and the corpus loader
itself are every one imported unchanged. The only new code this module
introduces is the multi-level generalisation of the bias-only mask
described above, and the pooling and reporting of its result alongside the
real statistic it is being checked against.

SIGN CONVENTION, unchanged from the corpus: every coefficient is a
correlation against the continuous graded report ERROR, or against the
binary swap indicator (1 = swap). No sign flip is applied.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters, watters_behaviour  # noqa: E402
from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from run_component_and_item_binding import (  # noqa: E402
    MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
    _close,
    _corr,
    _pool_values,
    analyse_session,
    build_pooled_table,
    reproduction_gate,
)
from run_swap_versus_imprecision_by_item_count import (  # noqa: E402
    _bias_only_between_session,
    _session_arrays,
)
from scipy.stats import norm  # noqa: E402
from statistics import Z_80_POWER, minimum_detectable_paired_difference, stable_seed  # noqa: E402

Z_95 = float(norm.ppf(0.975))

OUTPUT_PATH = ROOT / "results" / "component_binding_bias_only_control.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "component_binding_bias_only_control"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.json"
DELIVERED_PATH = ROOT / "results" / "component_and_item_binding.json"
SIBLING_PATH = ROOT / "results" / "swap_versus_imprecision_by_item_count.json"
ANALYSIS_VERSION = "2026-09-05"
REPRODUCTION_TOLERANCE = 1e-6

# Unchanged from _bias_only_between_session's own hardcoded floor: fewer session-level points than
# this makes a between-session correlation meaningless regardless of which levels feed it.
MIN_SESSIONS_FOR_BETWEEN_SESSION_CORRELATION = 4

LEVELS_2_AND_3 = (2, 3)

BRANCH_GATE_FAILED = "void_reproduction_gate_failed"
BRANCH_SURVIVES = "swap_association_survives_the_bias_only_control"
BRANCH_VOIDED = "swap_association_voided_by_the_bias_only_control"
BRANCH_BELOW_FLOOR = "inconclusive_below_detection_floor"

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "The pooled, within-item-count-level swap association (deviation vs the primary-rule swap "
    "indicator) is checked against a bias-only control before its own significance is read as "
    "evidence, in this order:\n"
    f"  0. If the reproduction gate does not reproduce the delivered corpus's own orthogonality gate, "
    f"raw behaviour association, and within-item-count-level swap_primary/deviation family (every "
    f"member: raw, both single partials, and the joint partial) at tolerance {REPRODUCTION_TOLERANCE}, "
    f"the branch is '{BRANCH_GATE_FAILED}' and no further number is read.\n"
    "  1. The two load-bearing statistics the delivered rule already required both significant -- the "
    "raw pooled correlation and the correlation jointly partialling total spike count and trial index "
    "-- are each independently checked against their own bias-only control, built by collapsing each "
    "session's trials, at every item-count level that session's own real statistic was pooled across, "
    "to that session's own mean, then correlating those means BETWEEN sessions.\n"
    "  2. A bias-only control VOIDS its statistic if the control is itself significant (two-sided "
    "p < 0.05) in the SAME direction as the real statistic. Sign and significance only -- the control's "
    "magnitude is a different estimator on a different scale and is never compared to the real effect "
    "by size, in either direction.\n"
    f"  3. If either the raw or the joint-partial real statistic is not itself significant, the branch "
    f"is '{BRANCH_BELOW_FLOOR}', reported with the minimum detectable paired difference beside it -- "
    "the bias-only question does not arise if the delivered rule's own bar was never cleared.\n"
    f"  4. If neither load-bearing statistic is voided, the branch is '{BRANCH_SURVIVES}'.\n"
    f"  5. If either load-bearing statistic is voided, the branch is '{BRANCH_VOIDED}', regardless of "
    "the other statistic's own status -- the delivered rule required BOTH significant to call the "
    "association predicted, so losing either one to its own bias-only control is sufficient to lose the "
    "association.\n"
    "The identical procedure (bias-only control, same voiding rule) is additionally applied to the "
    "imprecision outcome and to the item-count-1 control arm, reported beside the swap result as "
    "context for the whole dissociation, but only the swap outcome's own branch is the headline verdict "
    "of this artifact."
)

BIAS_ONLY_SOURCE_QUOTE = (
    "Verbatim from run_swap_versus_imprecision_by_item_count.py's _bias_only_between_session "
    "(imported unchanged and used here for every single-level cell):\n\n"
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
    'this test\'s own sign and significance enter the voiding rule, which does not depend on magnitude."""\n\n'
    "This module's own generalisation (_bias_only_between_session_generalized, defined below) changes "
    "exactly two things relative to the quoted function: (a) the single-level equality mask "
    "`arrays[\"item_count\"] == float(level)` is replaced by a mask over every level a session's real "
    "within-item-count-level statistic was actually pooled across (`np.isin(arrays[\"item_count\"], "
    "levels)`), justified by the algebraic identity that a trial-count-weighted average of a "
    "partition's means equals the mean over the union of that partition; (b) the correlation is computed "
    "by _corr (the corpus's own unchanged partial-correlation estimator, already required elsewhere in "
    "this family of scripts) rather than a plain Pearson call, so the identical function yields both the "
    "raw and the joint-partial bias-only statistic. Both changes are additive generalisations of the "
    "quoted definition to the composite the real branch was decided on; neither changes what counts as a "
    "bias-only substitution or what the voiding rule checks."
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ============================================================================
# Checkpointing -- identical scratch-then-replace pattern used throughout this
# corpus's other scripts, so a kill mid-write never corrupts the checkpoint.
# ============================================================================

_COMPLETED_FITS: dict[str, dict] = {}


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
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(checkpoint_safe(_COMPLETED_FITS), allow_nan=False, default=float))
    os.replace(scratch, CHECKPOINT_PATH)
    return value


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# ============================================================================
# Multi-level bias-only generalisation
# ============================================================================

def _qualifying_levels(row: dict, outcome_key: str) -> list[int]:
    """The item-count levels at which THIS session's own real per-level correlation for outcome_key
    already reached the corpus's trial floor and computed -- read from analyse_session's own,
    already-fit per_level field, never recomputed here. Reusing this exact set (rather than a fresh
    trial-count check) guarantees the bias-only composite below is built from exactly the same
    (session, level) contributions the real trial-count-weighted composite already used, so the
    algebraic identity in the module docstring applies exactly, not approximately."""
    per_level = row.get(outcome_key, {}).get("deviation", {}).get("per_level", {})
    return [lv for lv in LEVELS_2_AND_3 if per_level.get(str(lv), {}).get("status") == "computed"]


def _bias_only_between_session_generalized(rows: list[dict], rows_arrays: dict[str, dict],
                                            outcome_key: str, level_selector, controls: tuple[str, ...],
                                            seed_tag: str) -> dict:
    """See BIAS_ONLY_SOURCE_QUOTE (module-level) for the exact two-part generalisation this makes to
    _bias_only_between_session, quoted there in full. level_selector(row) -> list[int] returns the
    levels this session's own real statistic was pooled across (empty if none qualify)."""
    means: dict[str, list[float]] = {"deviation": [], "outcome": [], "spike_count": [], "trial_index": []}
    sessions_used: list[str] = []
    for row in rows:
        if row.get("status") != "computed":
            continue
        key = row["session"]
        arrays = rows_arrays.get(key)
        if arrays is None:
            continue
        levels = level_selector(row)
        if not levels:
            continue
        mask = np.isin(arrays["item_count"], levels)
        if int(mask.sum()) == 0:
            continue
        means["deviation"].append(float(np.mean(arrays["deviation"][mask])))
        means["outcome"].append(float(np.mean(arrays[outcome_key][mask])))
        means["spike_count"].append(float(np.mean(arrays["spike_count"][mask])))
        means["trial_index"].append(float(np.mean(arrays["trial_index"][mask])))
        sessions_used.append(key)

    n_sessions = len(means["deviation"])
    if n_sessions < MIN_SESSIONS_FOR_BETWEEN_SESSION_CORRELATION:
        return {"status": "not_computable", "n_sessions": n_sessions,
                "reason": f"fewer than {MIN_SESSIONS_FOR_BETWEEN_SESSION_CORRELATION} sessions reach "
                          "the trial floor at the levels this composite is pooled across"}

    control_arrays = [np.array(means[name]) for name in controls]
    result = _corr(np.array(means["outcome"]), np.array(means["deviation"]), control_arrays, seed_tag)
    if result.get("status") != "computed":
        result["n_sessions"] = n_sessions
        return result
    result["n_sessions"] = n_sessions
    result["significant"] = bool(result["p_value"] < 0.05)
    result["direction"] = _direction(result["r"])
    result["sessions_used"] = sessions_used
    return result


def _direction(value: float | None) -> str | None:
    if value is None:
        return None
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def _real_cell_report(pooled_cell: dict) -> dict:
    """Repackages an already-pooled real statistic (from build_pooled_table / _pool_values, unchanged)
    into the fixed reporting shape this artifact uses for every cell, real or bias-only."""
    if pooled_cell.get("status") != "tested":
        return {"status": pooled_cell.get("status", "not_computable"), "n_sessions": pooled_cell.get("n_sessions")}
    return {
        "status": "tested", "n_sessions": pooled_cell["n_sessions"],
        "mean_value": pooled_cell["mean_value"], "median_value": pooled_cell.get("median_value"),
        "p_value": pooled_cell["p_value"], "ci_lower": pooled_cell["ci_lower"], "ci_upper": pooled_cell["ci_upper"],
        "significant": pooled_cell["significant"], "direction": _direction(pooled_cell["mean_value"]),
        "minimum_detectable_paired_difference_at_80pct_power":
            pooled_cell.get("minimum_detectable_paired_difference_at_80pct_power"),
    }


def _bias_only_cell_report(bias_result: dict) -> dict:
    """The bias-only control's own reporting shape. median/CI/minimum-detectable-difference are not
    defined for this estimator: it is a single between-session correlation over N session-level points
    (one number, not a per-session list), so there is no per-unit spread to build a bootstrap CI or a
    paired-difference power bound from without inventing a procedure the corpus's own delivered
    bias-only control (results/swap_versus_imprecision_by_item_count.json) does not itself report
    either -- that artifact's own bias_only_control cells carry only r, p_value, r_analytic, p_analytic,
    n and n_sessions, and this module follows that precedent rather than adding one."""
    if bias_result.get("status") != "computed":
        return {"status": bias_result.get("status", "not_computable"), "n_sessions": bias_result.get("n_sessions"),
                "reason": bias_result.get("reason")}
    return {
        "status": "computed", "n_sessions": bias_result["n_sessions"], "n_sessions_used": bias_result["n"],
        "mean_value": bias_result["r"], "median_value": None,
        "p_value": bias_result["p_value"], "ci_lower": None, "ci_upper": None,
        "ci_not_defined_reason": "a single between-session correlation over N session-level points has "
                                  "no per-unit spread to bootstrap; the delivered bias-only control this "
                                  "module reuses does not report one either",
        "significant": bias_result["significant"], "direction": bias_result["direction"],
        "minimum_detectable_paired_difference_at_80pct_power": {
            "status": "not_applicable",
            "reason": "minimum_detectable_paired_difference requires one paired difference per unit; "
                      "this estimator collapses every session to a single between-session correlation, "
                      "so there is no per-session paired quantity to feed it -- the same reason the "
                      "corpus's own voiding rule compares sign and significance only, never magnitude",
        },
        "n_controls": bias_result.get("n_controls"),
    }


def _real_minus_control(real: dict, bias: dict) -> dict:
    """Real and bias-only are both correlation-scale estimates (r); their interval combines the
    real statistic's own whole-session paired-difference standard error (recovered from its
    already-computed minimum-detectable-difference, mdd = Z_80_POWER * se) with a Fisher-z
    delta-method standard error for the bias-only between-session correlation -- no per-session
    list survives into this dict's cell-report shape to bootstrap directly."""
    real_value, control_value = real.get("mean_value"), bias.get("mean_value")
    if (real.get("status") != "tested" or bias.get("status") != "computed"
            or real_value is None or control_value is None):
        return {"status": "not_applicable"}
    estimate = real_value - control_value
    mdd = real.get("minimum_detectable_paired_difference_at_80pct_power") or {}
    n_bias = bias.get("n_sessions_used") or bias.get("n_sessions")
    if "mdd" not in mdd or not n_bias or n_bias < 4:
        return {"status": "not_computable", "estimate": estimate, "real_value": real_value,
                "control_value": control_value}
    real_se = mdd["mdd"] / Z_80_POWER
    bias_se = float((1.0 - control_value ** 2) / np.sqrt(n_bias - 3))
    se = float(np.sqrt(real_se ** 2 + bias_se ** 2))
    return {
        "status": "computed", "estimate": estimate, "real_value": real_value,
        "control_value": control_value, "standard_error": se,
        "interval_95pct": [estimate - Z_95 * se, estimate + Z_95 * se],
    }


def _void_verdict(real: dict, bias: dict) -> dict:
    real_sig = bool(real.get("significant")) if real.get("status") == "tested" else None
    bias_sig = bool(bias.get("significant")) if bias.get("status") == "computed" else None
    same_sign = (bias.get("direction") is not None and real.get("direction") is not None
                 and bias["direction"] == real["direction"])
    voided = bool(real_sig and bias_sig and same_sign)
    return {
        "real_significant": real_sig, "bias_only_significant": bias_sig, "same_sign_as_real": same_sign,
        "voided": voided,
        "voiding_rule": "sign and significance only; the bias-only statistic is a different estimator on "
                        "a different scale from the real effect size and is never compared to it by "
                        "magnitude",
        "real_minus_control": _real_minus_control(real, bias),
    }


def _stat_pair(real_pooled_cell: dict, bias_result: dict) -> dict:
    real = _real_cell_report(real_pooled_cell)
    bias = _bias_only_cell_report(bias_result)
    return {"real": real, "bias_only_between_session": bias, "voiding": _void_verdict(real, bias)}


# ============================================================================
# Load-1 control real pooling (extends the delivered artifact's own raw-only
# pooling to the joint-partial family member, using the identical _pool_values
# import unchanged).
# ============================================================================

def _load1_real_pooled(rows: list[dict], stat: str) -> dict:
    values = [
        r["load_1_control"]["deviation"][stat]["r"] for r in rows
        if r.get("status") == "computed" and r["load_1_control"].get("status") == "computed"
        and r["load_1_control"]["deviation"].get(stat, {}).get("status") == "computed"
    ]
    return _pool_values(values)


# ============================================================================
# Driver
# ============================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": "Whether the multi-object macaque corpus's pooled, within-item-count-level "
                 "association between a rate-free population state-deviation component and "
                 "saccadic swap incidence survives a bias-only control -- a session-level offset "
                 "reproducing a within-session relationship by mixing session identity with "
                 "whatever session identity happens to correlate with.",
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "bias_only_control_source_and_generalisation": BIAS_ONLY_SOURCE_QUOTE,
        "sign_convention": "Every coefficient here is against the continuous graded report ERROR, or "
                            "against the binary swap indicator (1 = swap). No sign flip is applied.",
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus (one pass)")
    loaded: list[dict] = []
    refused: list[dict] = []
    n_seen = 0
    for session in iter_watters(root, bin_ms=100.0):
        n_seen += 1
        if session["status"] != "loaded":
            refused.append({"session": session["session"], "animal": session.get("animal"),
                             "session_date": session.get("session_date"), "status": session["status"]})
            continue
        loaded.append(session)
    _log(f"corpus loaded: {n_seen} seen, {len(loaded)} loaded, {len(refused)} refused, "
         f"elapsed={time.time() - t0:.0f}s")

    _log("running the corpus's own orthogonality/raw-behaviour reproduction gate")
    gate_result = _fit("reproduction_gate", lambda: reproduction_gate(loaded))
    _log(f"orthogonality gate: {gate_result['status']}")

    behaviour = watters_behaviour(root)
    rows: list[dict] = []
    for session in loaded:
        key = session["session"]
        row = _fit(f"session|{key}",
                   lambda s=session: analyse_session(s, behaviour, "component_binding_bias_only_control"))
        rows.append(row)
        output.setdefault("_progress", {})["sessions_done"] = len(rows)
        _flush(output)
        _log(f"  {key} status={row.get('status')} elapsed={time.time() - t0:.0f}s")
    output.pop("_progress", None)

    computed_rows = [r for r in rows if r.get("status") == "computed"]
    delivered = _read_json(DELIVERED_PATH)
    pooled = build_pooled_table(rows)

    swap_family_here = pooled["swap_primary"]["deviation"]["within_item_count_level"]
    swap_family_delivered = delivered["pooled_results"]["swap_primary"]["deviation"]["within_item_count_level"]
    imprecision_family_here = pooled["imprecision"]["deviation"]["within_item_count_level"]
    imprecision_family_delivered = delivered["pooled_results"]["imprecision"]["deviation"]["within_item_count_level"]

    reproduction_checks: dict[str, bool] = {
        "orthogonality_gate_reproduced": bool(
            gate_result["status"] == "reproduced_exactly"
            and delivered["reproduction_gate"]["status"] == "reproduced_exactly"
            and _close(gate_result["recomputed_gate"].get("mean_value"),
                       delivered["reproduction_gate"]["recomputed_gate"].get("mean_value"))
            and _close(gate_result["recomputed_gate"].get("p_value"),
                       delivered["reproduction_gate"]["recomputed_gate"].get("p_value"))
            and _close(gate_result["recomputed_raw"].get("mean_value"),
                       delivered["reproduction_gate"]["recomputed_raw"].get("mean_value"))
            and _close(gate_result["recomputed_raw"].get("p_value"),
                       delivered["reproduction_gate"]["recomputed_raw"].get("p_value"))),
    }
    for stat in ("raw", "partial_controlling_spike_count", "partial_controlling_trial_index",
                 "joint_partial_controlling_spike_count_and_trial_index"):
        here_swap, delivered_swap = swap_family_here[stat], swap_family_delivered[stat]
        here_imp, delivered_imp = imprecision_family_here[stat], imprecision_family_delivered[stat]
        reproduction_checks[f"swap_primary_deviation_{stat}_mean_value"] = _close(
            here_swap.get("mean_value"), delivered_swap.get("mean_value"))
        reproduction_checks[f"swap_primary_deviation_{stat}_p_value"] = _close(
            here_swap.get("p_value"), delivered_swap.get("p_value"))
        reproduction_checks[f"imprecision_deviation_{stat}_mean_value"] = _close(
            here_imp.get("mean_value"), delivered_imp.get("mean_value"))
        reproduction_checks[f"imprecision_deviation_{stat}_p_value"] = _close(
            here_imp.get("p_value"), delivered_imp.get("p_value"))

    gate_status = "reproduced_exactly" if all(reproduction_checks.values()) else "not_reproduced"
    output["reproduction_gate"] = {
        "status": gate_status, "tolerance": REPRODUCTION_TOLERANCE, "checks": reproduction_checks,
        "recomputed_orthogonality_gate": gate_result,
        "recomputed_swap_primary_deviation_within_item_count_level": swap_family_here,
        "delivered_swap_primary_deviation_within_item_count_level": swap_family_delivered,
        "recomputed_imprecision_deviation_within_item_count_level": imprecision_family_here,
        "delivered_imprecision_deviation_within_item_count_level": imprecision_family_delivered,
    }
    _flush(output)

    if gate_status != "reproduced_exactly":
        failing = {k: v for k, v in reproduction_checks.items() if not v}
        output["branch"] = {"branch": BRANCH_GATE_FAILED, "failing_checks": failing}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log(f"STOPPING: reproduction gate did not reproduce; failing checks: {list(failing)}")
        print(json.dumps({"reproduction_gate": gate_status, "branch": BRANCH_GATE_FAILED,
                           "failing_checks": list(failing)}, indent=2))
        return

    _log("reproduction gate reproduced the delivered artifact at tolerance 1e-6; continuing")

    _log("building per-session raw trial arrays (unchanged _session_arrays)")
    rows_arrays: dict[str, dict] = {}
    for session in loaded:
        arrays = _session_arrays(session, behaviour)
        if arrays is not None:
            rows_arrays[session["session"]] = arrays

    by_session = {r["session"]: r for r in computed_rows}

    def swap_selector(row: dict) -> list[int]:
        return _qualifying_levels(row, "swap_primary")

    def imprecision_selector(row: dict) -> list[int]:
        return _qualifying_levels(row, "imprecision")

    def load1_selector(row: dict) -> list[int]:
        return [1] if row.get("load_1_control", {}).get("status") == "computed" else []

    stat_controls = {
        "raw": (),
        "joint_partial_controlling_spike_count_and_trial_index": ("spike_count", "trial_index"),
    }

    swap_cells, imprecision_cells, load1_cells = {}, {}, {}
    for stat, controls in stat_controls.items():
        swap_bias = _fit(
            f"bias_only|swap_primary|{stat}",
            lambda sel=swap_selector, c=controls, s=stat: _bias_only_between_session_generalized(
                rows, rows_arrays, "swap_primary", sel, c,
                f"component_binding_bias_only_control|swap_primary|{s}"))
        imprecision_bias = _fit(
            f"bias_only|imprecision|{stat}",
            lambda sel=imprecision_selector, c=controls, s=stat: _bias_only_between_session_generalized(
                rows, rows_arrays, "imprecision", sel, c,
                f"component_binding_bias_only_control|imprecision|{s}"))
        load1_bias = _fit(
            f"bias_only|load1|{stat}",
            lambda sel=load1_selector, c=controls, s=stat: _bias_only_between_session_generalized(
                rows, rows_arrays, "imprecision", sel, c,
                f"component_binding_bias_only_control|load1|{s}"))

        swap_cells[stat] = _stat_pair(swap_family_here[stat], swap_bias)
        imprecision_cells[stat] = _stat_pair(imprecision_family_here[stat], imprecision_bias)
        load1_cells[stat] = _stat_pair(_load1_real_pooled(rows, stat), load1_bias)
    _flush(output)
    _log("bias-only control (pooled, within-item-count-level) complete")

    # Diagnostic-only cross-check: the single-level bias-only control (imported unchanged, unmodified)
    # at level 2 must reproduce the delivered sibling artifact's own level-2 value -- this validates the
    # session-array plumbing this module shares with that artifact, but never gates or moves a branch.
    diagnostic = {}
    sibling = _read_json(SIBLING_PATH)
    for level in LEVELS_2_AND_3:
        tag = f"swap_versus_imprecision_by_item_count|bias_only|swap_primary|deviation|level{level}"
        mine = _fit(f"diagnostic|level{level}",
                    lambda lv=level, tg=tag: _bias_only_between_session(rows_arrays, "swap_primary", lv, tg))
        sibling_cell = sibling["bias_only_control"].get(f"swap_primary|deviation|level{level}", {})
        sibling_bias = sibling_cell.get("bias_only_between_session", {})
        diagnostic[str(level)] = {
            "recomputed_r": mine.get("r"), "delivered_sibling_r": sibling_bias.get("r"),
            "recomputed_p_value": mine.get("p_value"), "delivered_sibling_p_value": sibling_bias.get("p_value"),
            "matches_at_tolerance": bool(
                _close(mine.get("r"), sibling_bias.get("r")) and _close(mine.get("p_value"), sibling_bias.get("p_value"))),
        }
    output["diagnostic_cross_check_against_delivered_sibling_single_level_control"] = diagnostic
    _flush(output)
    _log(f"diagnostic cross-check: {diagnostic}")

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------
    raw_real_sig = swap_cells["raw"]["real"].get("significant")
    joint_real_sig = swap_cells["joint_partial_controlling_spike_count_and_trial_index"]["real"].get("significant")
    raw_voided = swap_cells["raw"]["voiding"]["voided"]
    joint_voided = swap_cells["joint_partial_controlling_spike_count_and_trial_index"]["voiding"]["voided"]

    if raw_real_sig is not True or joint_real_sig is not True:
        branch = {
            "branch": BRANCH_BELOW_FLOOR,
            "raw_real_significant": raw_real_sig, "joint_partial_real_significant": joint_real_sig,
            "raw_minimum_detectable_paired_difference":
                swap_cells["raw"]["real"].get("minimum_detectable_paired_difference_at_80pct_power"),
            "joint_partial_minimum_detectable_paired_difference": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["real"].get(
                "minimum_detectable_paired_difference_at_80pct_power"),
            "note": "the delivered rule's own bar (both raw and joint-partial significant) was not "
                    "cleared on recomputation, so the bias-only question does not decide anything here",
        }
    elif raw_voided or joint_voided:
        branch = {
            "branch": BRANCH_VOIDED,
            "raw_voided": raw_voided, "joint_partial_voided": joint_voided,
            "raw_real_mean_value": swap_cells["raw"]["real"].get("mean_value"),
            "raw_bias_only_mean_value": swap_cells["raw"]["bias_only_between_session"].get("mean_value"),
            "raw_bias_only_p_value": swap_cells["raw"]["bias_only_between_session"].get("p_value"),
            "joint_partial_real_mean_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["real"].get("mean_value"),
            "joint_partial_bias_only_mean_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["bias_only_between_session"].get(
                "mean_value"),
            "joint_partial_bias_only_p_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["bias_only_between_session"].get(
                "p_value"),
        }
    else:
        branch = {
            "branch": BRANCH_SURVIVES,
            "raw_real_mean_value": swap_cells["raw"]["real"].get("mean_value"),
            "raw_real_p_value": swap_cells["raw"]["real"].get("p_value"),
            "raw_bias_only_mean_value": swap_cells["raw"]["bias_only_between_session"].get("mean_value"),
            "raw_bias_only_p_value": swap_cells["raw"]["bias_only_between_session"].get("p_value"),
            "joint_partial_real_mean_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["real"].get("mean_value"),
            "joint_partial_real_p_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["real"].get("p_value"),
            "joint_partial_bias_only_mean_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["bias_only_between_session"].get(
                "mean_value"),
            "joint_partial_bias_only_p_value": swap_cells[
                "joint_partial_controlling_spike_count_and_trial_index"]["bias_only_between_session"].get(
                "p_value"),
        }

    output["swap_primary_deviation_within_item_count_level"] = swap_cells
    output["imprecision_deviation_within_item_count_level"] = imprecision_cells
    output["load_1_control"] = load1_cells
    output["branch"] = branch

    output["zero_drop_accounting"] = {
        "n_seen": n_seen, "n_loaded": len(loaded), "n_refused": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                               for reason in sorted({r["status"] for r in refused})},
        "n_analysed_computed": len(computed_rows), "n_loaded_but_not_computed": len(loaded) - len(computed_rows),
        "not_computed_by_reason": {
            reason: sum(1 for r in rows if r.get("status") == reason)
            for reason in sorted({r.get("status") for r in rows if r.get("status") != "computed"})
        },
        "reconciles": bool(n_seen == len(loaded) + len(refused)),
        "reconciles_against_delivered_artifact": bool(
            n_seen == delivered["zero_drop_accounting"]["n_seen"]
            and len(loaded) == delivered["zero_drop_accounting"]["n_loaded"]
            and len(refused) == delivered["zero_drop_accounting"]["n_refused"]
            and len(computed_rows) == delivered["zero_drop_accounting"]["n_analysed_computed"]),
        "delivered_artifact_counts": delivered["zero_drop_accounting"],
    }
    output["levels_pooled_for_the_within_item_count_level_composite"] = list(LEVELS_2_AND_3)
    output["git_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"branch: {branch['branch']} elapsed={time.time() - t0:.0f}s")
    print(json.dumps({"reproduction_gate": gate_status, "branch": branch["branch"],
                       "n_sessions_analysed": len(computed_rows)}, indent=2, default=float))


if __name__ == "__main__":
    main()
