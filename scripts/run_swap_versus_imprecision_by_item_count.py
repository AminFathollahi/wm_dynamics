"""run_swap_versus_imprecision_by_item_count.py -- is the swap/imprecision
dissociation in the multi-object macaque corpus real at each item-count
level on its own, or an artefact of how the levels were combined?

A prior analysis in this corpus (results/component_and_item_binding.json)
reported two numbers as a pair: a rate-free population state-deviation
component associates with SWAPS (the saccadic report lands nearer an
uncued object than the cued one) within item-count level, trial-count
weighted across levels, and does not associate with IMPRECISION (the
graded distance from the report to whichever object it landed nearest) by
the same estimator. Two things about that pairing were never checked.
First, the imprecision cell is a trial-count-weighted combination across
item-count levels 2 and 3 only; its two levels have never been reported on
their own, and a combined null built from levels that disagree in sign is
between-level mixing, not a per-level absence of effect. Second, "swap
significant, imprecision not" is two significance verdicts, not a
demonstrated difference -- the direct paired test between the two
associations, on the same sessions, was never run.

This module recomputes both associations per item-count level (the primary
statistic this project uses everywhere a variable-load task could mix
signs across levels), runs the direct paired test the pairing needs, and
adds a commensurability control: the swap association is a point-biserial
correlation against a binary indicator, and the imprecision association is
a correlation against a continuous distance, so their attainable ranges
differ with the swap base rate. Imprecision is additionally dichotomised
per session, at that session's own realised swap rate, and the identical
paired test is recomputed binary-against-binary.

Nothing already delivered is modified, re-run in place, or re-labelled:
results/component_and_item_binding.json is read only to gate this module's
own reproduction of its numbers, at a tolerance of 1e-6, before anything
else here is computed. Every session loader, the deviation estimator, the
swap and imprecision definitions, the partial-correlation permutation
estimator, the pooling estimator and the sign convention are the identical,
unchanged functions that produced that artifact -- this module adds only
the per-level pooling, the direct paired tests, the commensurability
dichotomisation, a bias-only control and a trial-order shuffle null on top
of them.

SIGN CONVENTION, unchanged from the artifact this module reproduces: every
coefficient is a correlation against the continuous graded report ERROR, or
against the binary swap indicator (1 = swap). No sign flip is applied.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

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
    FAMILY_STAT_KEYS,
    _behaviour_observables,
    _close,
    _object_geometry,
    _predicts,
    analyse_session,
    build_disclosures,
    build_pooled_table,
    reproduction_gate,
)
from run_state_behavior_link import trial_amplitude_covariates  # noqa: E402
from statistics import (  # noqa: E402
    fdr_bh,
    minimum_detectable_paired_difference,
    partial_correlation_permutation_test,
    paired_sign_flip_test,
    pearson_permutation_test,
    permutation_pvalue,
    stable_seed,
)
from run_watters_state_geometry import MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION  # noqa: E402

SCRIPT_STEM = Path(__file__).stem
OUTPUT_PATH = ROOT / "results" / "swap_versus_imprecision_by_item_count.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / SCRIPT_STEM
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.json"
DELIVERED_PATH = ROOT / "results" / "component_and_item_binding.json"
ANALYSIS_VERSION = "2026-08-21"
REPRODUCTION_TOLERANCE = 1e-6
N_PERM = 10000
N_SHUFFLE_DRAWS = 1000
ALL_LEVELS = (1, 2, 3)

BRANCH_GATE_VOID = "void_reproduction_gate_did_not_reproduce"
BRANCH_HOLDS_EVERY_LEVEL = "the_dissociation_holds_at_every_item_count_level"
BRANCH_CARRIED_BY_ONE_LEVEL = "the_dissociation_is_carried_by_one_item_count_level"
BRANCH_NOT_ESTABLISHED = "the_dissociation_is_not_established_by_a_direct_paired_test"
BRANCH_INCONCLUSIVE = "inconclusive_below_detection_floor"
FLAG_IMPRECISION_AT_LOAD1 = "imprecision_is_predicted_at_single_item_load"
FLAG_MIXING = "the_combined_imprecision_null_is_between_level_mixing"
VOID_SESSION_OFFSET = "association_not_separable_from_a_session_level_offset"

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Six named outcomes, checked in this order, decided by a direct paired session-level test between "
    "the deviation-versus-swap association and the deviation-versus-imprecision association at each "
    "item-count level a swap is defined at (primary nearest-object rule), never by comparing two "
    "significance verdicts:\n"
    f"  0. If the reproduction gate does not match the delivered artifact's own swap association, "
    f"imprecision association, orthogonality gate and item-count-1 arm at tolerance {REPRODUCTION_TOLERANCE}, "
    f"the branch is '{BRANCH_GATE_VOID}' and no further number is read.\n"
    f"  1. '{BRANCH_HOLDS_EVERY_LEVEL}' -- at every level a swap is defined at, the swap association is "
    "significant (raw and joint-partial both significant, the same bar the delivered branch itself used), "
    "the imprecision association at that level is not significant (raw) with its minimum detectable "
    "difference below that level's own swap effect magnitude, and the direct paired test between them is "
    "significant after Benjamini-Hochberg correction across levels, under BOTH the continuous and the "
    "dichotomised commensurability version.\n"
    f"  2. '{BRANCH_CARRIED_BY_ONE_LEVEL}' -- the direct paired test is significant (BH-corrected) at some "
    "levels and not others, and a heterogeneity test between the levels is itself significant. Reported "
    "per level; never converted into a gradient or an ordering across item count.\n"
    f"  3. '{BRANCH_NOT_ESTABLISHED}' -- the direct paired test is not significant (BH-corrected) at any "
    "level and its minimum detectable difference lies below that level's own swap effect magnitude at "
    "every level. The delivered branch stands as fired; this module reports that the ordering it implies "
    "was never tested directly and does not survive being tested.\n"
    f"  4. '{BRANCH_INCONCLUSIVE}' -- none of the above; the relevant tests are not significant and their "
    "minimum detectable differences do not clear the reference effect. Never quoted without those numbers.\n"
    "Two further flags are evaluated independently of which of the above fires and reported beside it, "
    f"never in place of it: '{FLAG_IMPRECISION_AT_LOAD1}' if the item-count-1 deviation-versus-report-error "
    f"association is itself significant, and '{FLAG_MIXING}' if the per-level imprecision associations "
    "differ in sign with a significant heterogeneity test between the differing levels while the "
    "levels-2-and-3 trial-count-weighted combination this module reproduces is null.\n"
    f"A bias-only control ({VOID_SESSION_OFFSET}) is checked before any of the above is read: each "
    "session's real per-trial deviation is replaced with that session's own mean deviation over its trials "
    "at that level (a session-level constant), and the pooled correlation is recomputed on the trial-level "
    "data this collapses to (a per-session correlation against a constant predictor is undefined by "
    "construction, so this recomputation is necessarily a pooled-across-sessions statistic, not a "
    "per-session one). Any load-bearing cell the bias-only version reproduces in the same direction and "
    "significance is named void by this reason rather than used to decide a branch. If an outcome occurs "
    "that this list does not cover, it is reported in writing as a gap in this rule, with the numbers, "
    "never forced onto the nearest label."
)

COMMENSURABILITY_NOTE = (
    "The swap association correlates the deviation against a binary indicator (1 = nearest displayed "
    "object is not the cued one); the imprecision association correlates it against a continuous distance. "
    "Their attainable correlation ranges differ with the binary variable's base rate, so the two are not "
    "directly commensurable as reported. The commensurability control dichotomises imprecision per "
    "session, per level: that session's own worst-imprecision trials (largest response-to-nearest-object "
    "distance) are flagged 1, at the identical COUNT as that session's own primary-rule swap trials at that "
    "level (so the same base rate by construction), and the direct paired test is recomputed with both "
    "sides binary. Both the continuous and the dichotomised version are reported; if they disagree neither "
    "is called the corrected one."
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ============================================================================
# Checkpointing -- one directory per script, a single JSON map of completed
# fits inside it, written to a scratch file then moved into place so a kill
# mid-write never leaves a corrupt or half-written checkpoint on disk.
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
# Per-session raw trial arrays -- the same masking _behaviour_observables and
# _object_geometry (both imported unchanged) already apply inside
# analyse_session, kept here as arrays rather than only as summary
# statistics, for the new per-level tests this module adds (commensurability
# dichotomisation, the bias-only control, the shuffle null). Cheap: no
# permutation test runs here, only the identical masks analyse_session
# already computes.
# ============================================================================

def _session_arrays(session: dict, behaviour) -> dict | None:
    observables, _excluded, usable = _behaviour_observables(session["counts"], session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return None
    covariates = trial_amplitude_covariates(session["counts"])
    if covariates["status"] != "computed":
        return None
    amplitude_full = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    geometry = _object_geometry(behaviour, session)
    return {
        "item_count": observables["item_count"],
        "deviation": observables["state_deviation"],
        "amplitude": amplitude_full[usable],
        "spike_count": observables["spike_count"],
        "trial_index": observables["trial_index"],
        "swap_primary": geometry["swap_primary"][usable].astype(float),
        "imprecision": geometry["imprecision"][usable],
    }


# ============================================================================
# Small shared statistics
# ============================================================================

def _direct_paired_test(a_vals, b_vals, tag: str) -> dict:
    """The one test every ordering claim in this module rests on: a direct paired session-level
    sign-flip test of a minus b (mean difference, CI, p-value), with sd and the minimum detectable
    paired difference at 80% power computed on the identical per-session differences. Two arrays,
    same length, same session order."""
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    n = len(a_vals)
    if n < 2:
        return {"status": "not_computable", "n_sessions_paired": n,
                "reason": "fewer than 2 sessions contribute both values"}
    diffs = a_vals - b_vals
    rng = np.random.default_rng(stable_seed(tag))
    paired = paired_sign_flip_test(a_vals, b_vals, alternative="two-sided", rng=rng)
    mdd = minimum_detectable_paired_difference(diffs)
    return {
        "status": "tested", "n_sessions_paired": n,
        "mean_diff": paired["mean_diff"], "p_value": paired["p_value"],
        "ci_lower": paired["ci_lower"], "ci_upper": paired["ci_upper"],
        "sd": mdd.get("sd") if mdd.get("status") == "computed" else None,
        "minimum_detectable_paired_difference_at_80pct_power": mdd,
        "significant": bool(paired["p_value"] < 0.05),
        "mean_a": float(np.mean(a_vals)), "mean_b": float(np.mean(b_vals)),
    }


def _bh_family(tests: dict[str, dict]) -> dict:
    """Benjamini-Hochberg correction over one named family of already-computed direct paired tests
    (dict of label -> _direct_paired_test result); only "tested" entries enter the correction."""
    labels = [label for label, t in tests.items() if t.get("status") == "tested"]
    if not labels:
        return {"status": "not_computable", "reason": "no test in this family reached 2 paired sessions"}
    p_values = np.array([tests[label]["p_value"] for label in labels], dtype=float)
    corrected = fdr_bh(p_values)
    return {
        "status": "computed", "labels": labels, "alpha": corrected["alpha"],
        "q_values": {label: float(q) for label, q in zip(labels, corrected["q_values"])},
        "bh_significant": {label: bool(q <= corrected["alpha"]) for label, q in
                            zip(labels, corrected["q_values"])},
    }


def _pool_series(values: list[float]) -> dict:
    from state_persistence import slope_across_sessions_test
    if len(values) < 2:
        return {"status": "not_computable", "n_sessions": len(values)}
    pooled = slope_across_sessions_test(values, alternative="two-sided")
    pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    pooled["median_value"] = float(np.median(values))
    return pooled


def _cheap_r(outcome: np.ndarray, covariate: np.ndarray) -> float | None:
    """Plain Pearson r with no permutation p-value -- used only inside the >=1000-draws-per-session
    shuffle null below, where re-running the full permutation-based significance estimator on every
    draw is not tractable; this is the identical formula partial_correlation_permutation_test itself
    uses for its observed (no-controls) statistic."""
    if np.std(outcome) == 0.0 or np.std(covariate) == 0.0:
        return None
    return float(np.corrcoef(outcome, covariate)[0, 1])


# ============================================================================
# Block A -- per-level pooled statistics for every outcome and observable
# ============================================================================

def _level_mask_ok(arrays: dict, level: int) -> tuple[np.ndarray, int]:
    mask = arrays["item_count"] == float(level)
    return mask, int(mask.sum())


def _collect_level_family(rows: list[dict], outcome: str, observable: str, level: int) -> dict[str, list[float]]:
    """Per-session raw-family correlation values at one item-count level, read from analyse_session's
    own already-computed per_level (levels >= 2) or load_1_control (level 1, imprecision only) field
    -- no new correlation is fit here, only collected and pooled."""
    collected: dict[str, list[float]] = {stat: [] for stat in FAMILY_STAT_KEYS}
    for r in rows:
        if r.get("status") != "computed":
            continue
        if level == 1:
            if outcome != "imprecision":
                continue
            lc = r.get("load_1_control", {})
            if lc.get("status") != "computed":
                continue
            fam = lc.get(observable, {})
        else:
            entry = r.get(outcome, {}).get(observable, {}).get("per_level", {}).get(str(level), {})
            if entry.get("status") != "computed":
                continue
            fam = entry.get("family", {})
        for stat in FAMILY_STAT_KEYS:
            cell = fam.get(stat, {})
            if cell.get("status") == "computed":
                collected[stat].append(cell["r"])
    return collected


def block_a_table(rows: list[dict]) -> dict:
    computed_rows = [r for r in rows if r.get("status") == "computed"]
    table: dict = {}
    for outcome in ("swap_primary", "swap_strict", "imprecision"):
        table[outcome] = {}
        levels = ALL_LEVELS if outcome == "imprecision" else (2, 3)
        for observable in ("deviation", "amplitude"):
            table[outcome][observable] = {}
            for level in levels:
                collected = _collect_level_family(computed_rows, outcome, observable, level)
                table[outcome][observable][str(level)] = {
                    stat: _pool_series(collected[stat]) for stat in FAMILY_STAT_KEYS
                }
    return table


def block_a_level_counts(rows: list[dict]) -> dict:
    """Realised trial count, session count and swap base rate per level, recomputed fresh from the
    reproduced per-session rows -- never carried from the delivered artifact."""
    computed_rows = [r for r in rows if r.get("status") == "computed"]
    out: dict = {}
    for level in ALL_LEVELS:
        key = str(level)
        n_trials = sum(int(r["n_trials_by_item_count"].get(key, 0)) for r in computed_rows)
        n_sessions = sum(1 for r in computed_rows if int(r["n_trials_by_item_count"].get(key, 0))
                          >= MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION)
        if level == 1:
            n_swap = 0
        else:
            n_swap = sum(int(r["n_swap_primary_by_item_count"].get(key, 0)) for r in computed_rows)
        out[key] = {
            "n_trials": n_trials, "n_sessions_reaching_the_trial_floor": n_sessions,
            "n_swap_trials_primary_rule": n_swap,
            "swap_base_rate_primary_rule": (n_swap / n_trials) if n_trials else 0.0,
        }
    return out


def block_a_heterogeneity(rows: list[dict]) -> dict:
    """A direct paired session-level test between every pair of item-count levels reached, per outcome
    and observable, on sessions contributing a raw per-level value to both levels -- reported, never
    converted into a gradient, a trend or an ordering across item count."""
    computed_rows = [r for r in rows if r.get("status") == "computed"]
    out: dict = {}
    for outcome in ("swap_primary", "swap_strict", "imprecision"):
        levels = ALL_LEVELS if outcome == "imprecision" else (2, 3)
        out[outcome] = {}
        for observable in ("deviation", "amplitude"):
            pair_tests: dict[str, dict] = {}
            for i, level_i in enumerate(levels):
                for level_j in levels[i + 1:]:
                    a_vals, b_vals = [], []
                    for r in computed_rows:
                        va = _one_level_raw(r, outcome, observable, level_i)
                        vb = _one_level_raw(r, outcome, observable, level_j)
                        if va is not None and vb is not None:
                            a_vals.append(va)
                            b_vals.append(vb)
                    label = f"level{level_i}_vs_level{level_j}"
                    tag = f"swap_versus_imprecision_by_item_count|heterogeneity|{outcome}|{observable}|{label}"
                    pair_tests[label] = _direct_paired_test(a_vals, b_vals, tag)
            out[outcome][observable] = {
                "pairwise_tests": pair_tests,
                "benjamini_hochberg_across_level_pairs": _bh_family(pair_tests),
            }
    return out


def _one_level_raw(row: dict, outcome: str, observable: str, level: int) -> float | None:
    if level == 1:
        if outcome != "imprecision":
            return None
        lc = row.get("load_1_control", {})
        if lc.get("status") != "computed":
            return None
        cell = lc.get(observable, {}).get("raw", {})
    else:
        entry = row.get(outcome, {}).get(observable, {}).get("per_level", {}).get(str(level), {})
        if entry.get("status") != "computed":
            return None
        cell = entry.get("family", {}).get("raw", {})
    return cell["r"] if cell.get("status") == "computed" else None


# ============================================================================
# Block B -- the direct paired test between swap and imprecision, per level,
# continuous and commensurability-dichotomised versions.
# ============================================================================

def _dichotomize_worst(values: np.ndarray, k: int) -> np.ndarray:
    """Binary flag on the k trials with the LARGEST value (worst imprecision = largest response-to-
    nearest-object distance), the rest 0 -- matches the session's own swap trial count at that level
    exactly, so the two binary indicators share a base rate by construction."""
    flag = np.zeros(len(values), dtype=float)
    if k <= 0:
        return flag
    order = np.argsort(-values, kind="stable")
    flag[order[:k]] = 1.0
    return flag


def block_b_commensurability(rows: list[dict], rows_arrays: dict[str, dict], levels_with_swap: tuple) -> dict:
    """Per level: swap-r (reused from the already-computed per_level raw family) paired against a
    freshly computed deviation-vs-dichotomised-imprecision point-biserial correlation, on the same
    sessions, same trials, same base rate."""
    by_row = {r["session"]: r for r in rows if r.get("status") == "computed"}
    out: dict = {}
    for level in levels_with_swap:
        a_vals, b_vals, sessions_used = [], [], []
        for key, arrays in rows_arrays.items():
            row = by_row.get(key)
            if row is None:
                continue
            swap_r = _one_level_raw(row, "swap_primary", "deviation", level)
            if swap_r is None:
                continue
            mask, n_level = _level_mask_ok(arrays, level)
            if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
                continue
            swap_flag = arrays["swap_primary"][mask]
            n_swap = int(swap_flag.sum())
            imprecision_flag = _dichotomize_worst(arrays["imprecision"][mask], n_swap)
            tag = f"swap_versus_imprecision_by_item_count|commensurability|{key}|level{level}"
            rng = np.random.default_rng(stable_seed(tag))
            result = partial_correlation_permutation_test(
                imprecision_flag, arrays["deviation"][mask], [], N_PERM, rng)
            if result.get("status") != "computed":
                continue
            a_vals.append(swap_r)
            b_vals.append(result["r"])
            sessions_used.append(key)
        tag = f"swap_versus_imprecision_by_item_count|block_b_commensurability|level{level}"
        out[str(level)] = {**_direct_paired_test(a_vals, b_vals, tag),
                            "n_sessions_used": len(sessions_used)}
    return out


def block_b_primary(rows: list[dict], levels_with_swap: tuple, outcome: str = "swap_primary") -> dict:
    computed_rows = [r for r in rows if r.get("status") == "computed"]
    out: dict = {}
    for level in levels_with_swap:
        a_vals, b_vals = [], []
        for r in computed_rows:
            va = _one_level_raw(r, outcome, "deviation", level)
            vb = _one_level_raw(r, "imprecision", "deviation", level)
            if va is not None and vb is not None:
                a_vals.append(va)
                b_vals.append(vb)
        tag = f"swap_versus_imprecision_by_item_count|block_b_primary|{outcome}|level{level}"
        out[str(level)] = _direct_paired_test(a_vals, b_vals, tag)
    return out


# ============================================================================
# Block C -- the single-item arm
# ============================================================================

def block_c(rows: list[dict], load1_deviation_raw: dict, swap_within_level: dict) -> dict:
    computed_rows = [r for r in rows if r.get("status") == "computed"]
    a_vals, b_vals = [], []
    for r in computed_rows:
        lc = r.get("load_1_control", {})
        if lc.get("status") != "computed":
            continue
        lc_raw = lc.get("deviation", {}).get("raw", {})
        if lc_raw.get("status") != "computed":
            continue
        swap_val = r.get("swap_primary", {}).get("deviation", {}).get(
            "within_item_count_level_trial_count_weighted", {}).get("raw")
        if swap_val is None:
            continue
        a_vals.append(lc_raw["r"])
        b_vals.append(swap_val)
    tag = "swap_versus_imprecision_by_item_count|block_c|item_count_1_vs_swap"
    paired = _direct_paired_test(a_vals, b_vals, tag)
    return {
        "item_count_1_deviation_vs_report_error": load1_deviation_raw,
        "swap_effect_within_item_count_level_reference": swap_within_level,
        "direct_paired_test_item_count_1_vs_swap": paired,
    }


# ============================================================================
# Bias-only control
# ============================================================================

def _bias_only_between_session(rows_arrays: dict[str, dict], outcome_key: str, level: int,
                                seed_tag: str) -> dict:
    """The session-level bias-only statistic: collapses every session's real per-trial deviation at
    this level to that session's own mean (the bias-only substitution the control specifies), one
    number per session, paired against that same session's own mean outcome at this level -- session
    is the unit of analysis, N sessions, matching the real statistic's own unit of analysis.

    This is NOT the same estimator as the real one. The real statistic is a per-session WITHIN-session
    correlation (many trials, one r per session) pooled across sessions by a sign-flip test. Once every
    trial in a session is replaced by that session's own mean, the within-session predictor has zero
    variance by construction, so a per-session correlation is undefined -- there is no well-defined way
    to recompute "the same" per-session statistic under this substitution. The closest well-defined,
    session-is-the-unit-of-analysis alternative is a single BETWEEN-session correlation over the N
    (session mean deviation, session mean outcome) points, computed here. Its r, p-value and n are on a
    different scale from the real within-session, sign-flip-pooled effect size (different estimator,
    different degrees of freedom) and are never compared to the real effect size by magnitude; only
    this test's own sign and significance enter the voiding rule, which does not depend on magnitude."""
    means_x, means_y = [], []
    for arrays in rows_arrays.values():
        mask, n_level = _level_mask_ok(arrays, level)
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            continue
        outcome_vals = arrays[outcome_key][mask]
        means_x.append(float(np.mean(arrays["deviation"][mask])))
        means_y.append(float(np.mean(outcome_vals)))
    n_sessions = len(means_x)
    if n_sessions < 4:
        return {"status": "not_computable", "n_sessions": n_sessions,
                "reason": "fewer than 4 sessions reach the trial floor at this level"}
    rng = np.random.default_rng(stable_seed(seed_tag))
    result = pearson_permutation_test(np.array(means_x), np.array(means_y), n_perm=N_PERM, rng=rng)
    result["status"] = "computed"
    result["n_sessions"] = n_sessions
    return result


def bias_only_control(rows_arrays: dict[str, dict], real_table: dict, levels_with_swap: tuple) -> dict:
    """For every load-bearing (outcome, level) cell: the session-level bias-only statistic above,
    reported beside the real cell's own significance. A cell is named void
    (association_not_separable_from_a_session_level_offset) when the real cell is significant AND the
    bias-only statistic is ALSO significant, in the SAME direction -- sign and significance only, never
    a magnitude comparison, because the two statistics are different estimators on different scales."""
    cells: dict[str, dict] = {}
    for outcome in ("swap_primary", "imprecision"):
        levels = levels_with_swap if outcome == "swap_primary" else ALL_LEVELS
        outcome_key = "swap_primary" if outcome == "swap_primary" else "imprecision"
        for observable in ("deviation",):
            for level in levels:
                real = real_table[outcome][observable][str(level)]["raw"]
                real_sig = bool(real.get("significant"))
                key = f"{outcome}|{observable}|level{level}"
                tag = f"swap_versus_imprecision_by_item_count|bias_only|{key}"
                bias = _bias_only_between_session(rows_arrays, outcome_key, level, tag)
                bias_sig = bool(bias.get("status") == "computed" and bias.get("p_value", 1.0) < 0.05)
                same_sign = bool(bias.get("status") == "computed" and real.get("status") == "tested"
                                  and (bias["r"] > 0.0) == (real["mean_value"] > 0.0))
                reproduces = bool(real_sig and bias_sig and same_sign)
                cells[key] = {
                    "real_significant": real_sig, "real_mean_value": real.get("mean_value"),
                    "bias_only_between_session": bias, "bias_only_significant": bias_sig,
                    "same_sign_as_real": same_sign,
                    "reproduces_the_real_result": reproduces,
                    "voiding_rule": "sign and significance only; the bias-only statistic is a different "
                                    "estimator on a different scale from the real effect size and is never "
                                    "compared to it by magnitude",
                    "branch": VOID_SESSION_OFFSET if reproduces else None,
                }
    return cells


# ============================================================================
# Within-session trial-order shuffle null for Block B's primary paired test
# ============================================================================

def shuffle_null_block_b(rows_arrays: dict[str, dict], level: int, real_mean_diff: float,
                          n_draws: int = N_SHUFFLE_DRAWS) -> dict:
    """>=n_draws independent within-session shuffles of each contributing session's own trial order
    (deviation permuted, swap and imprecision outcomes left at their real trial positions), both
    correlations recomputed end to end on every draw, the per-session differences pooled the same
    unweighted across-session way the real sign-flip test pools them -- an independent, non-parametric
    check on Block B's primary paired-difference p-value at this level."""
    sessions = []
    for arrays in rows_arrays.values():
        mask, n_level = _level_mask_ok(arrays, level)
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            continue
        swap = arrays["swap_primary"][mask]
        if np.std(swap) == 0.0:
            continue
        sessions.append((arrays["deviation"][mask], swap, arrays["imprecision"][mask]))
    n = len(sessions)
    if n < 2:
        return {"status": "not_computable", "n_sessions": n}

    tag = f"swap_versus_imprecision_by_item_count|shuffle_null|level{level}"
    rng = np.random.default_rng(stable_seed(tag))
    null_draws = np.empty(n_draws)
    for draw in range(n_draws):
        draw_diffs = np.empty(n)
        for i, (deviation, swap, imprecision) in enumerate(sessions):
            perm = rng.permutation(len(deviation))
            deviation_shuffled = deviation[perm]
            r_swap = _cheap_r(swap, deviation_shuffled)
            r_imprecision = _cheap_r(imprecision, deviation_shuffled)
            draw_diffs[i] = (r_swap or 0.0) - (r_imprecision or 0.0)
        null_draws[draw] = float(np.mean(draw_diffs))

    p_value = permutation_pvalue(np.abs(null_draws) >= abs(real_mean_diff))
    return {
        "status": "computed", "n_sessions": n, "n_draws": n_draws,
        "real_mean_diff_being_calibrated_against": real_mean_diff,
        "p_value": p_value, "null_mean": float(np.mean(null_draws)),
        "null_sd": float(np.std(null_draws, ddof=1)),
    }


# ============================================================================
# Named outcome decision
# ============================================================================

def _branch_from_flags(per_level_flags: dict, levels_with_swap: tuple, block_a: dict,
                        block_b_primary_result: dict, heterogeneity: dict) -> str:
    """The four-way branch decision from a completed per_level_flags table -- factored out so the same
    logic runs twice: once on the raw significance flags (for the record) and once on the bias-only-
    voided flags (the one actually reported), so a caller can see exactly what the voiding changed."""
    holds_every_level = len(levels_with_swap) > 0 and all(
        per_level_flags[level]["swap_predicts"] is True
        and per_level_flags[level]["imprecision_raw_significant"] is False
        and per_level_flags[level]["imprecision_mdd_below_swap_effect"] is True
        and per_level_flags[level]["block_b_primary_bh_significant"] is True
        and per_level_flags[level]["block_b_commensurability_bh_significant"] is True
        for level in levels_with_swap
    )
    primary_sig_flags = [per_level_flags[level]["block_b_primary_bh_significant"] for level in levels_with_swap]
    some_significant = any(f is True for f in primary_sig_flags)
    some_not_significant = any(f is False for f in primary_sig_flags)
    heterogeneity_significant_swap = any(
        t.get("status") == "tested" and t.get("p_value", 1.0) < 0.05
        for t in heterogeneity["swap_primary"]["deviation"]["pairwise_tests"].values())
    carried_by_one_level = (not holds_every_level) and some_significant and some_not_significant \
        and heterogeneity_significant_swap

    none_significant = all(f is False for f in primary_sig_flags) if primary_sig_flags else False
    all_mdd_below = all(
        block_b_primary_result[str(level)].get("status") == "tested"
        and block_b_primary_result[str(level)]["minimum_detectable_paired_difference_at_80pct_power"]
            .get("status") == "computed"
        and block_b_primary_result[str(level)]["minimum_detectable_paired_difference_at_80pct_power"]["mdd"]
            < abs(block_a["swap_primary"]["deviation"][str(level)]["raw"]["mean_value"])
        for level in levels_with_swap
        if block_a["swap_primary"]["deviation"][str(level)]["raw"].get("status") == "tested"
    ) if levels_with_swap else False
    not_established = (not holds_every_level) and (not carried_by_one_level) and none_significant and all_mdd_below

    if holds_every_level:
        return BRANCH_HOLDS_EVERY_LEVEL
    if carried_by_one_level:
        return BRANCH_CARRIED_BY_ONE_LEVEL
    if not_established:
        return BRANCH_NOT_ESTABLISHED
    return BRANCH_INCONCLUSIVE


def decide_named_outcomes(block_a: dict, block_a_counts: dict, block_b_primary_result: dict,
                           block_b_primary_bh: dict, block_b_commensurability_result: dict,
                           block_b_commensurability_bh: dict, heterogeneity: dict,
                           levels_with_swap: tuple, load1_deviation_raw: dict,
                           reproduced_imprecision_combined: dict, bias_only: dict) -> dict:
    def swap_voided(level: int) -> bool:
        cell = bias_only.get(f"swap_primary|deviation|level{level}", {})
        return bool(cell.get("reproduces_the_real_result"))

    def swap_predicts(level: int, voided: bool) -> bool | None:
        if voided:
            return False
        cell = block_a["swap_primary"]["deviation"][str(level)]
        return _predicts({"within_item_count_level": cell})

    def imprecision_raw_sig(level: int) -> bool | None:
        cell = block_a["imprecision"]["deviation"][str(level)]["raw"]
        return bool(cell["significant"]) if cell.get("status") == "tested" else None

    def imprecision_mdd_below_swap(level: int) -> bool | None:
        imp_cell = block_a["imprecision"]["deviation"][str(level)]["raw"]
        swap_cell = block_a["swap_primary"]["deviation"][str(level)]["raw"]
        imp_mdd = imp_cell.get("minimum_detectable_paired_difference_at_80pct_power", {})
        if imp_mdd.get("status") != "computed" or swap_cell.get("status") != "tested":
            return None
        return bool(imp_mdd["mdd"] < abs(swap_cell["mean_value"]))

    per_level_flags_raw, per_level_flags = {}, {}
    for level in levels_with_swap:
        voided = swap_voided(level)
        primary_bh_sig = block_b_primary_bh.get("bh_significant", {}).get(str(level))
        commensurability_bh_sig = block_b_commensurability_bh.get("bh_significant", {}).get(str(level))
        raw = {
            "swap_predicts": swap_predicts(level, voided=False),
            "imprecision_raw_significant": imprecision_raw_sig(level),
            "imprecision_mdd_below_swap_effect": imprecision_mdd_below_swap(level),
            "block_b_primary_bh_significant": primary_bh_sig,
            "block_b_commensurability_bh_significant": commensurability_bh_sig,
            "primary_and_commensurability_agree": (
                None if primary_bh_sig is None or commensurability_bh_sig is None
                else bool(primary_bh_sig == commensurability_bh_sig)),
        }
        per_level_flags_raw[level] = raw
        # A swap cell the bias-only control reproduces cannot carry a branch: neither Block A's own
        # "predicts" verdict nor Block B's paired-test significance at that level may be read as
        # established evidence for it, per the bias-only rule declared before this leg was fitted.
        per_level_flags[level] = {
            **raw,
            "swap_association_bias_only_void": voided,
            "swap_predicts": False if voided else raw["swap_predicts"],
            "block_b_primary_bh_significant": False if voided else primary_bh_sig,
            "block_b_commensurability_bh_significant": False if voided else commensurability_bh_sig,
        }

    branch_before_bias_only_voiding = _branch_from_flags(
        per_level_flags_raw, levels_with_swap, block_a, block_b_primary_result, heterogeneity)
    primary_branch = _branch_from_flags(
        per_level_flags, levels_with_swap, block_a, block_b_primary_result, heterogeneity)
    voided_levels = [level for level in levels_with_swap if per_level_flags[level]["swap_association_bias_only_void"]]
    bias_only_disclosure = {
        "voided_levels": voided_levels,
        "branch_before_bias_only_voiding": branch_before_bias_only_voiding,
        "branch_after_bias_only_voiding": primary_branch,
        "superseded": bool(voided_levels and branch_before_bias_only_voiding != primary_branch),
        "reason": (
            "the swap association's own bias-only control (a between-session statistic, sign and "
            "significance only, never compared to the real effect by magnitude) reproduces the real, "
            f"significant result at level(s) {voided_levels}; a reproduced cell cannot carry a branch, "
            "per the bias-only rule declared before this leg was fitted"
        ) if voided_levels else "no load-bearing swap cell was reproduced by its bias-only control",
    }

    load1_sig = bool(load1_deviation_raw.get("significant")) if load1_deviation_raw.get("status") == "tested" \
        else None
    flag_imprecision_at_load1 = {
        "fires": bool(load1_sig), "item_count_1_association_significant": load1_sig,
        "item_count_1_mean_value": load1_deviation_raw.get("mean_value"),
        "item_count_1_p_value": load1_deviation_raw.get("p_value"),
    }

    level_means = {}
    for level in ALL_LEVELS:
        cell = block_a["imprecision"]["deviation"][str(level)]["raw"]
        if cell.get("status") == "tested":
            level_means[level] = cell["mean_value"]
    signs = {lv: (v > 0.0) for lv, v in level_means.items()}
    differ_in_sign = len(set(signs.values())) > 1
    differing_pairs_significant = False
    for label, test in heterogeneity["imprecision"]["deviation"]["pairwise_tests"].items():
        lv_i, lv_j = (int(x.replace("level", "")) for x in label.split("_vs_"))
        if lv_i in signs and lv_j in signs and signs[lv_i] != signs[lv_j] \
                and test.get("status") == "tested" and test.get("p_value", 1.0) < 0.05:
            differing_pairs_significant = True
    combined_is_null = not bool(reproduced_imprecision_combined.get("significant", False))
    flag_mixing = {
        "fires": bool(differ_in_sign and differing_pairs_significant and combined_is_null),
        "per_level_imprecision_means": level_means,
        "levels_differ_in_sign": differ_in_sign,
        "a_significant_heterogeneity_test_separates_the_differing_levels": differing_pairs_significant,
        "levels_2_and_3_trial_count_weighted_combination_is_null": combined_is_null,
        "levels_2_and_3_combination_reproduced_here": reproduced_imprecision_combined,
    }

    return {
        "primary_branch": primary_branch,
        "bias_only_voiding": bias_only_disclosure,
        "per_level_flags": {str(k): v for k, v in per_level_flags.items()},
        "per_level_flags_before_bias_only_voiding": {str(k): v for k, v in per_level_flags_raw.items()},
        FLAG_IMPRECISION_AT_LOAD1: flag_imprecision_at_load1,
        FLAG_MIXING: flag_mixing,
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
        "version": ANALYSIS_VERSION,
        "corpus": "Multi-object spatial working memory in macaque frontal cortex, DANDI 000620 (Watters, "
                  "Gabel, Tenenbaum and Jazayeri; bioRxiv preprint posted 2026-01-27, DOI "
                  "10.64898/2026.01.27.702062, unreviewed).",
        "sign_convention": "Every coefficient here is against the continuous graded report ERROR, or "
                            "against the binary swap indicator (1 = swap). No sign flip is applied.",
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "commensurability_note": COMMENSURABILITY_NOTE,
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

    _log("running reproduction gate against the delivered artifact's own gate")
    gate_result = _fit("reproduction_gate", lambda: reproduction_gate(loaded))
    output["reachability"] = {
        "n_sessions_seen": n_seen, "n_sessions_loaded": len(loaded), "n_sessions_refused": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                               for reason in sorted({r["status"] for r in refused})},
        "counts_reconcile": bool(n_seen == len(loaded) + len(refused)),
    }
    _flush(output)
    _log(f"reproduction gate (against watters_state_geometry.json): {gate_result['status']}")

    behaviour = watters_behaviour(root)
    rows: list[dict] = []
    for session in loaded:
        key = session["session"]
        row = _fit(f"session|{key}", lambda s=session: analyse_session(s, behaviour, "component_and_item_binding"))
        rows.append(row)
        output.setdefault("_progress", {})["sessions_done"] = len(rows)
        _flush(output)
        _log(f"  {key} status={row.get('status')} elapsed={time.time() - t0:.0f}s")
    output.pop("_progress", None)

    computed_rows = [r for r in rows if r.get("status") == "computed"]
    delivered = _read_json(DELIVERED_PATH)
    pooled = build_pooled_table(rows)
    load1_deviation_raw = _pool_series(
        [r["load_1_control"]["deviation"]["raw"]["r"] for r in computed_rows
         if r["load_1_control"].get("status") == "computed"
         and r["load_1_control"]["deviation"]["raw"].get("status") == "computed"])

    swap_delivered = delivered["pooled_results"]["swap_primary"]["deviation"]["within_item_count_level"]["raw"]
    imprecision_delivered = delivered["pooled_results"]["imprecision"]["deviation"]["within_item_count_level"]["raw"]
    swap_here = pooled["swap_primary"]["deviation"]["within_item_count_level"]["raw"]
    imprecision_here = pooled["imprecision"]["deviation"]["within_item_count_level"]["raw"]
    load1_delivered = delivered["load_1_control"]["deviation_vs_report_error_raw"]

    reproduction_checks = {
        "swap_association_mean_value": _close(swap_here.get("mean_value"), swap_delivered.get("mean_value")),
        "swap_association_p_value": _close(swap_here.get("p_value"), swap_delivered.get("p_value")),
        "imprecision_association_mean_value": _close(imprecision_here.get("mean_value"),
                                                       imprecision_delivered.get("mean_value")),
        "imprecision_association_p_value": _close(imprecision_here.get("p_value"), imprecision_delivered.get("p_value")),
        "orthogonality_gate_reproduced": bool(
            gate_result["status"] == "reproduced_exactly"
            and delivered["reproduction_gate"]["status"] == "reproduced_exactly"
            and _close(gate_result["recomputed_gate"].get("mean_value"),
                       delivered["reproduction_gate"]["recomputed_gate"].get("mean_value"))
            and _close(gate_result["recomputed_gate"].get("p_value"),
                       delivered["reproduction_gate"]["recomputed_gate"].get("p_value"))),
        "item_count_1_arm_mean_value": _close(load1_deviation_raw.get("mean_value"), load1_delivered.get("mean_value")),
        "item_count_1_arm_p_value": _close(load1_deviation_raw.get("p_value"), load1_delivered.get("p_value")),
    }
    reproduction_gate_status = "reproduced_exactly" if all(reproduction_checks.values()) else "not_reproduced"
    output["reproduction_gate_against_delivered_artifact"] = {
        "status": reproduction_gate_status, "tolerance": REPRODUCTION_TOLERANCE, "checks": reproduction_checks,
        "delivered_swap_association": swap_delivered, "recomputed_swap_association": swap_here,
        "delivered_imprecision_association": imprecision_delivered, "recomputed_imprecision_association": imprecision_here,
        "delivered_item_count_1_arm": load1_delivered, "recomputed_item_count_1_arm": load1_deviation_raw,
        "delivered_orthogonality_gate": delivered["reproduction_gate"]["status"],
        "recomputed_orthogonality_gate": gate_result["status"],
    }
    _flush(output)

    if reproduction_gate_status != "reproduced_exactly":
        output["branch"] = {"primary_branch": BRANCH_GATE_VOID}
        output["status"] = "complete"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce the delivered artifact; nothing further computed")
        print(json.dumps({"reproduction_gate": reproduction_gate_status, "branch": BRANCH_GATE_VOID}, indent=2))
        return

    _log("reproduction gate reproduced the delivered artifact at tolerance 1e-6; continuing")

    _log("building per-session raw trial arrays for the new per-level tests")
    rows_arrays: dict[str, dict] = {}
    for session in loaded:
        arrays = _session_arrays(session, behaviour)
        if arrays is not None:
            rows_arrays[session["session"]] = arrays

    block_a = block_a_table(rows)
    block_a_counts = block_a_level_counts(rows)
    heterogeneity = block_a_heterogeneity(rows)

    n_trials_by_level = {lv: block_a_counts[str(lv)]["n_trials"] for lv in ALL_LEVELS}
    n_swap_by_level = {lv: block_a_counts[str(lv)]["n_swap_trials_primary_rule"] for lv in (2, 3)}
    levels_with_swap = tuple(
        lv for lv in (2, 3)
        if n_swap_by_level[lv] > 0 and block_a["swap_primary"]["deviation"][str(lv)]["raw"].get("status") == "tested"
    )
    output["block_a_level_counts"] = block_a_counts
    output["block_a_per_level_table"] = block_a
    output["block_a_heterogeneity"] = heterogeneity
    output["levels_with_a_defined_swap"] = list(levels_with_swap)
    _flush(output)
    _log(f"Block A complete: levels_with_a_defined_swap={levels_with_swap}")

    block_b_primary_result = _fit(
        "block_b_primary_swap_primary",
        lambda: block_b_primary(rows, levels_with_swap, outcome="swap_primary"))
    block_b_primary_bh = _bh_family(block_b_primary_result)
    block_b_strict_result = _fit(
        "block_b_primary_swap_strict",
        lambda: block_b_primary(rows, levels_with_swap, outcome="swap_strict"))
    block_b_strict_bh = _bh_family(block_b_strict_result)
    block_b_commensurability_result = _fit(
        "block_b_commensurability",
        lambda: block_b_commensurability(rows, rows_arrays, levels_with_swap))
    block_b_commensurability_bh = _bh_family(block_b_commensurability_result)

    agreement = {}
    for level in levels_with_swap:
        p = block_b_primary_bh.get("bh_significant", {}).get(str(level))
        c = block_b_commensurability_bh.get("bh_significant", {}).get(str(level))
        agreement[str(level)] = {
            "primary_bh_significant": p, "commensurability_bh_significant": c,
            "agree": None if p is None or c is None else bool(p == c),
        }
    output["block_b"] = {
        "primary_continuous_imprecision": {"per_level": block_b_primary_result,
                                            "benjamini_hochberg_across_levels": block_b_primary_bh},
        "commensurability_dichotomised_imprecision": {"per_level": block_b_commensurability_result,
                                                        "benjamini_hochberg_across_levels": block_b_commensurability_bh},
        "strict_swap_rule_sensitivity": {"per_level": block_b_strict_result,
                                          "benjamini_hochberg_across_levels": block_b_strict_bh},
        "agreement_between_continuous_and_commensurability_versions": agreement,
    }
    _flush(output)
    _log("Block B complete")

    disclosures = build_disclosures(pooled, load1_deviation_raw, rows)
    swap_within_level_reference = pooled["swap_primary"]["deviation"]["within_item_count_level"]["raw"]
    output["block_c"] = block_c(rows, load1_deviation_raw, swap_within_level_reference)
    output["block_c"]["load1_cannot_exclude_the_swap_effect"] = disclosures["load1_cannot_exclude_the_swap_effect"]
    _flush(output)
    _log("Block C complete")

    # Checkpoint key names the between-session estimator explicitly: an earlier, superseded fit under
    # the plain "bias_only_control" key used a trial-pooled statistic not commensurable with the real,
    # session-is-the-unit-of-analysis effect size, and is left in place rather than deleted.
    bias_only = _fit("bias_only_control_between_session",
                      lambda: bias_only_control(rows_arrays, block_a, levels_with_swap))
    output["bias_only_control"] = bias_only
    _flush(output)
    _log("bias-only control complete")

    shuffle_results = {}
    for level in levels_with_swap:
        real_diff = block_b_primary_result[str(level)].get("mean_diff")
        if real_diff is None:
            shuffle_results[str(level)] = {"status": "not_computable", "reason": "primary paired test not tested"}
            continue
        shuffle_results[str(level)] = _fit(
            f"shuffle_null|level{level}",
            lambda lv=level, rd=real_diff: shuffle_null_block_b(rows_arrays, lv, rd, N_SHUFFLE_DRAWS))
    output["shuffle_null_block_b_primary"] = shuffle_results
    _flush(output)
    _log("shuffle null complete")

    reproduced_imprecision_combined = pooled["imprecision"]["deviation"]["within_item_count_level"]["raw"]
    named_outcomes = decide_named_outcomes(
        block_a, block_a_counts, block_b_primary_result, block_b_primary_bh,
        block_b_commensurability_result, block_b_commensurability_bh, heterogeneity,
        levels_with_swap, load1_deviation_raw, reproduced_imprecision_combined, bias_only)
    output["named_outcomes"] = named_outcomes
    output["branch"] = {"primary_branch": named_outcomes["primary_branch"],
                         "bias_only_voiding": named_outcomes["bias_only_voiding"]}

    output["zero_drop_accounting"] = {
        "n_seen": n_seen, "n_loaded": len(loaded), "n_refused": len(refused),
        "n_analysed_computed": len(computed_rows), "n_loaded_but_not_computed": len(loaded) - len(computed_rows),
        "reconciles": bool(n_seen == len(loaded) + len(refused)),
        "reconciles_against_delivered_artifact": bool(
            n_seen == delivered["zero_drop_accounting"]["n_seen"]
            and len(loaded) == delivered["zero_drop_accounting"]["n_loaded"]
            and len(refused) == delivered["zero_drop_accounting"]["n_refused"]),
        "delivered_artifact_counts": delivered["zero_drop_accounting"],
        "per_level_session_counts": {str(lv): block_a_counts[str(lv)]["n_sessions_reaching_the_trial_floor"]
                                      for lv in ALL_LEVELS},
    }

    output["scope"] = {
        "corpus": "watters_2026 (multi-object macaque, DANDI 000620)",
        "unit_quality_tier": "single_and_multi_unit",
        "n_sessions_seen": n_seen, "n_sessions_analysed": len(computed_rows),
        "min_trials_per_correlation": MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
        "n_shuffle_draws_per_session": N_SHUFFLE_DRAWS,
        "git_commit": git_commit(ROOT),
    }
    output["sessions"] = rows
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"branch: {named_outcomes['primary_branch']} elapsed={time.time() - t0:.0f}s")
    print(json.dumps({
        "reproduction_gate": reproduction_gate_status, "branch": named_outcomes["primary_branch"],
        "n_sessions_analysed": len(computed_rows), "levels_with_a_defined_swap": list(levels_with_swap),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
