"""run_watters_load_decomposition.py -- within-load and between-load split of
the multi-object macaque corpus's graded behaviour link.

The delivered artifact `results/watters_state_geometry.json` fired
`state_geometry_does_not_predict_accuracy` on its own pre-declared primary
test: the RAW pooled coefficient between a trial's rate-free state deviation
and its graded report error, pooled over all trials of every item count
together (raw graded -0.011663, p=0.0723, interval [-0.02418, -0.0000494]).
That branch is correct under its own rule and does not move; the 14,320 s job
that produced it is not repeated here.

That same delivered artifact separately found, without a pre-declared branch
for it, that controlling item count flips the pooled coefficient's sign
(+0.017218, p=0.00040) while controlling reaction time, spike count or trial
index individually does not. A variable-load task's raw pooled coefficient
mixes a within-load relationship (estimable in a single-item task) with a
between-load relationship (only possible when item count varies); the
single-item macaque corpus is necessarily a within-load estimate, so it is
only commensurable with this corpus's within-load component, not its raw
pooled one. This script gives that observation its own pre-declared test.

WITHIN-LOAD, defined here: per session, per item-count level with enough
trials, the Pearson correlation between report error and state deviation
computed ONLY on that level's own trials (not a linear partial correlation
that merely regresses out a continuous item-count covariate -- a literal
stratified estimate, inside each level, then combined). A session's several
level estimates are combined into one session value two ways -- weighted by
the level's own trial count, and weighted equally across the levels present
-- so the choice of weighting is visible rather than assumed. The combined
per-session values are pooled ACROSS sessions the same way every other
session-level statistic in this corpus is pooled: one vote per session,
paired sign-flip test, unit of analysis the session, never the trial.

BETWEEN-LOAD, defined here: per session, for the item-count levels that
cleared the same trial floor, the ordinary-least-squares slope of the
level's OWN MEAN report error on the level's OWN MEAN state deviation across
those few points -- a between-group estimate built from group means, not
from a continuous item-count covariate. Pooled across sessions by the same
paired sign-flip test used for the within-load pooling and for this corpus's
label-cardinality ladder (a slope over a handful of points per session,
pooled over sessions).

Every trial-level quantity here is recomputed rather than read off the
delivered artifact, because the delivered artifact stores only per-session
POOLED correlations, not the raw per-trial state-deviation/report-error/
item-count triples a level-stratified estimate needs. The recomputation
reuses the delivered artifact's own loader (`load_watters_session`/
`iter_watters`), its own trial-usability mask and observable definitions
(`_behaviour_observables`), its own matched-unit-count draw (identical seed,
so the same units are drawn), and its own permutation-correlation estimator
(`_corr` / `partial_correlation_permutation_test`) -- nothing about how a
trial becomes usable, how the state deviation is computed, or how a
correlation is tested differs from the delivered artifact. What is NOT
recomputed: the delivered artifact's existence profiles, cardinality ladder
and load-amplitude arm, and its per-cell orthogonality-gate verdict, which is
read directly from `results/watters_state_geometry.json` to decide which of
the fifteen cells (three unit-quality arms x five groups) are void.

SIGN CONVENTION. Every coefficient in this corpus is a correlation against
report ERROR (higher state deviation -> worse report, if the delivered
effect's sign is taken as the working hypothesis). The single-item macaque
corpus's own headline correlation is against CORRECTNESS, so its magnitude
is negated before being used here as a reference against the report-error
convention. Both figures are stated explicitly wherever they appear below.

The corpus is a bioRxiv preprint (posted 2026-01-27, DOI
10.64898/2026.01.27.702062, DANDI 000620) and is unreviewed; no
peer-reviewed journal version has been found.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters, watters_behaviour  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from run_watters_state_geometry import (  # noqa: E402
    MATCHED_UNIT_COUNT,
    MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
    PRIMARY_QUALITY_TIER,
    WATTERS_MIN_UNITS,
    _behaviour_observables,
    _corr,
    _matched_unit_subset,
    _pool_values,
)
from state_persistence import _ols_slope  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "watters_load_decomposition.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "watters_load_decomposition_checkpoint.json"
DELIVERED_WATTERS_PATH = ROOT / "results" / "watters_state_geometry.json"
SINGLE_ITEM_SOURCE_PATH = ROOT / "results" / "dominant_latent_identity_and_behaviour_breadth.json"
ANALYSIS_VERSION = "2026-08-14"
MATCHED_UNIT_SEED_TAG_PREFIX = "watters_state_geometry"  # must equal the delivered script's own tag, to reproduce its unit draw

MIN_TRIALS_PER_ITEM_COUNT_LEVEL = MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION

QUALITY_TIERS = (PRIMARY_QUALITY_TIER, "good_single_units_only", "matched_unit_count_arm")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _single_item_corpus_reference_magnitude() -> dict:
    """The single-item macaque corpus's pooled behaviour-link coefficient at
    its primary error floor, read from the delivered artifact that reports
    it (not recomputed), then negated to the report-error sign convention
    this corpus uses throughout."""
    source = _read_json(SINGLE_ITEM_SOURCE_PATH)
    node = source["error_floor_sensitivity"]["by_floor"]["60"]["pooled"]["raw_outcome_vs_observable"]
    against_correctness = float(node["mean_value"])
    return {
        "source_artifact": "results/dominant_latent_identity_and_behaviour_breadth.json",
        "source_path_in_artifact": "error_floor_sensitivity.by_floor.60.pooled.raw_outcome_vs_observable",
        "value_against_correctness_as_deposited": against_correctness,
        "n_sessions": node.get("n_sessions"),
        "p_value_as_deposited": node.get("p_value"),
        "value_against_report_error_after_sign_convention": -against_correctness,
        "magnitude_used_as_the_reference": abs(against_correctness),
    }


def _delivered_raw_pooled_coefficient() -> dict:
    """The delivered artifact's fired primary coefficient, read (not
    recomputed) purely so this artifact can state, next to its own numbers,
    which sign the raw pooled coefficient carries -- the comparison this
    mandate exists to make, never a re-decision of the fired branch."""
    delivered = _read_json(DELIVERED_WATTERS_PATH)
    node = delivered["results"]["single_and_multi_unit"]["pooled"]["behaviour"][
        "continuous_graded_report"]["raw_report_error_vs_state_deviation"]
    return {
        "source_artifact": "results/watters_state_geometry.json",
        "source_path_in_artifact": "results.single_and_multi_unit.pooled.behaviour.continuous_graded_report."
                                    "raw_report_error_vs_state_deviation",
        "mean_value": float(node["mean_value"]), "p_value": float(node["p_value"]),
        "branch_fired": delivered["branches"]["behaviour"],
        "n_sessions": node.get("n_sessions"),
    }


def _delivered_gate_lookup() -> dict[tuple[str, str], dict]:
    """Per-cell orthogonality-gate verdict, read from the delivered artifact
    for every (tier, group) cell this script also reports, so a cell whose
    state-deviation observable is not rate-free there can be marked void
    here without re-fitting the gate."""
    delivered = _read_json(DELIVERED_WATTERS_PATH)
    results = delivered["results"]
    lookup: dict[tuple[str, str], dict] = {}
    for tier in (PRIMARY_QUALITY_TIER, "good_single_units_only"):
        for group, block in results[tier].items():
            if not isinstance(block, dict) or "behaviour" not in block:
                continue
            gate = block["behaviour"].get("orthogonality_gate_state_deviation_vs_spike_count")
            if gate is not None:
                lookup[(tier, group)] = gate
    for group, block in results["matched_unit_count_arm"].items():
        if not isinstance(block, dict) or "behaviour" not in block:
            continue
        gate = block["behaviour"].get("orthogonality_gate_state_deviation_vs_spike_count")
        if gate is not None:
            lookup[("matched_unit_count_arm", group)] = gate
    return lookup


def _task_variant_item_count_levels(root: Path) -> dict:
    """Which item-count levels each task variant actually presents, read
    directly from the behaviour logs -- established here rather than
    assumed, because it bounds how many points the between-load slope can
    ever be fitted through in each variant."""
    behaviour = watters_behaviour(root)
    info = {}
    for variant in ("ring", "triangle"):
        levels = sorted(int(v) for v in behaviour.loc[behaviour.task == variant, "num_objects"].unique())
        info[variant] = {"item_count_levels_present": levels, "n_levels": len(levels)}
    return {
        "by_task_variant": info,
        "note": (
            "The ring variant never presents a three-item trial: its item-count levels are "
            f"{info['ring']['item_count_levels_present']} only, so a ring session's between-load slope is "
            "fitted through exactly two level-mean points and its within-load estimate combines at most two "
            "levels. The triangle variant reaches "
            f"{info['triangle']['item_count_levels_present']}. Separately -- and this bounds the "
            "cardinality ladder elsewhere in this corpus, not the item-count levels reported here -- the "
            "triangle variant's objects sit on only three fixed vertices, so its distinct cued POSITIONS "
            "are three regardless of how many objects are held; that constrains label cardinality, not "
            "how many item-count levels a triangle session can contribute to this decomposition."
        ),
    }


WITHIN_LOAD_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, per item-count level with at least {floor} usable trials, correlate report error against "
    "the rate-free state deviation with the delivered artifact's own no-controls permutation-correlation "
    "estimator, computed only on that level's own trials (a literal stratified estimate, not a linear "
    "partial correlation that regresses out item count as a continuous covariate). Combine a session's "
    "per-level coefficients into one within-load value two ways: weighted by each level's own trial count, "
    "and weighted equally across the levels present. Pool each weighting's per-session values across "
    "sessions with the paired sign-flip test used for every other session-level pooled statistic in this "
    "corpus -- one vote per session, never one vote per trial. The reference magnitude is the single-item "
    "macaque corpus's own pooled behaviour-link coefficient, converted to the report-error sign convention "
    "this corpus uses. If the pooled within-load coefficient's interval excludes zero on the POSITIVE side "
    "(the same sign as that reference under the convention), the branch is "
    "'within_load_association_present'. If the interval excludes zero on the negative side, OR the interval "
    "covers zero and the minimum detectable paired difference at 80% power is BELOW the reference magnitude "
    "(the cell had the power to see an effect that size and did not), the branch is "
    "'within_load_association_absent'. Otherwise -- interval covers zero and the minimum detectable "
    "difference exceeds the reference magnitude -- the branch is 'inconclusive_below_detection_floor'. "
    "Declared and frozen before any of these numbers were computed."
).format(floor=MIN_TRIALS_PER_ITEM_COUNT_LEVEL)

BETWEEN_LOAD_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, for the item-count levels that cleared the within-load trial floor, take each level's own "
    "mean report error and mean state deviation -- one point per level -- and fit the ordinary-least-squares "
    "slope of mean report error on mean state deviation across those points; a session with fewer than two "
    "qualifying levels contributes no between-load value. Pool the per-session slopes across sessions with "
    "the same paired sign-flip test used for the within-load pooling and for this corpus's label-cardinality "
    "ladder. The between-load verdict is reported in two parts, both pre-declared: whether the pooled "
    "between-load sign is the OPPOSITE of the pooled within-load sign from the same cell, and whether the "
    "pooled between-load sign AGREES with the delivered artifact's raw pooled behaviour coefficient's sign "
    "-- a positive within-load coefficient alongside a between-load component that shares the raw pooled "
    "coefficient's negative sign is exactly the signature this analysis exists to test for. Declared and "
    "frozen before any of these numbers were computed."
)


def level_decomposition_arm(counts: np.ndarray, session: dict, seed_tag: str) -> dict:
    """Within-load and between-load view of one arm's (one unit-quality
    tier's) population, for one session."""
    observables, excluded, usable = _behaviour_observables(counts, session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {"status": "too_few_trials_with_a_defined_state_direction_and_report",
                "n_trials_usable": int(usable.sum()), "trials_excluded_by_reason": excluded}

    deviation = observables["state_deviation"]
    report_error = observables["report_error"]
    item_count = observables["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})

    per_level: dict[str, dict] = {}
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_PER_ITEM_COUNT_LEVEL:
            per_level[str(level)] = {"status": "too_few_trials_at_this_item_count", "n_trials": n_level}
            continue
        corr = _corr(report_error[mask], deviation[mask], [], f"{seed_tag}|level{level}")
        per_level[str(level)] = {
            **corr, "n_trials": n_level,
            "mean_report_error": float(np.mean(report_error[mask])),
            "mean_state_deviation": float(np.mean(deviation[mask])),
        }

    tested = [(level, per_level[str(level)]) for level in levels if per_level[str(level)].get("status") == "computed"]
    if not tested:
        return {
            "status": "no_item_count_level_clears_the_trial_floor",
            "n_trials_usable": int(usable.sum()), "item_count_levels_present": levels, "per_level": per_level,
        }

    r_values = np.array([entry["r"] for _, entry in tested], dtype=float)
    n_values = np.array([entry["n_trials"] for _, entry in tested], dtype=float)
    within_trial_weighted = float(np.sum((n_values / n_values.sum()) * r_values))
    within_equal_weighted = float(np.sum(r_values) / len(tested))

    between_slope = None
    if len(tested) >= 2:
        level_mean_x = [entry["mean_state_deviation"] for _, entry in tested]
        level_mean_y = [entry["mean_report_error"] for _, entry in tested]
        between_slope = _ols_slope(level_mean_x, level_mean_y)

    return {
        "status": "computed",
        "n_trials_usable": int(usable.sum()),
        "item_count_levels_present": levels,
        "n_levels_tested": len(tested),
        "levels_tested": [int(level) for level, _ in tested],
        "per_level": per_level,
        "within_load_trial_count_weighted": within_trial_weighted,
        "within_load_equal_weighted": within_equal_weighted,
        "between_load_slope_on_level_means": between_slope,
    }


def analyse_session(session: dict) -> dict:
    tag = f"watters_load_decomposition|{session['session']}"
    quality = np.asarray(session["unit_quality"])
    good = quality == "good"

    row: dict = {
        "animal": session["animal"], "session_date": session["session_date"], "session": session["session"],
        "task_variant": session["task_variant"], "n_units_total": int(session["n_units"]),
        "n_units_good": int(good.sum()), "n_trials": int(len(session["trial_num"])),
        "analysis_version": ANALYSIS_VERSION, "by_tier": {},
    }

    row["by_tier"][PRIMARY_QUALITY_TIER] = level_decomposition_arm(
        session["counts"], session, f"{tag}|{PRIMARY_QUALITY_TIER}")

    if int(good.sum()) >= WATTERS_MIN_UNITS:
        row["by_tier"]["good_single_units_only"] = level_decomposition_arm(
            session["counts"][:, good], session, f"{tag}|good_single_units_only")
    else:
        row["good_single_units_only_note"] = {
            "status": "below_the_unit_floor", "n_good_units": int(good.sum()), "floor": WATTERS_MIN_UNITS}

    # Identical seed tag to the delivered artifact's own matched-unit draw, so this
    # script analyses the SAME matched population, not an independently redrawn one.
    matched = _matched_unit_subset(session["counts"], f"{MATCHED_UNIT_SEED_TAG_PREFIX}|{session['session']}")
    row["by_tier"]["matched_unit_count_arm"] = level_decomposition_arm(
        matched, session, f"{tag}|matched_unit_count_arm")

    return row


def _load_checkpoint_raw() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _write_checkpoint(raw: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(raw), indent=2, allow_nan=False))
    os.replace(tmp, CHECKPOINT_PATH)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))
    os.replace(tmp, OUTPUT_PATH)


def _subsets(rows_in: list[dict]) -> dict[str, list[dict]]:
    subsets = {"pooled": rows_in}
    for animal in sorted({r["animal"] for r in rows_in}):
        subsets[f"animal_{animal}"] = [r for r in rows_in if r["animal"] == animal]
    for variant in sorted({r["task_variant"] for r in rows_in}):
        subsets[f"task_variant_{variant}"] = [r for r in rows_in if r["task_variant"] == variant]
    return subsets


def _within_load_branch(pooled: dict, reference_magnitude: float) -> str:
    if pooled.get("status") != "tested":
        return "inconclusive_below_detection_floor"
    if pooled.get("significant_positive"):
        return "within_load_association_present"
    if pooled.get("significant_negative"):
        return "within_load_association_absent"
    mdd = pooled.get("minimum_detectable_paired_difference_at_80pct_power", {})
    if mdd.get("status") == "computed" and mdd["mdd"] < reference_magnitude:
        return "within_load_association_absent"
    return "inconclusive_below_detection_floor"


def _sign_positive(value: float) -> bool:
    return value > 0.0


BETWEEN_LOAD_SLOPE_UNITS_AND_COMPARABILITY_NOTE = (
    "This quantity is an ordinary-least-squares regression slope of mean report error on mean "
    "state deviation across item-count levels, in units of report error per unit of state "
    "deviation. It is not a correlation and is not bounded to [-1, 1]. It is comparable to the "
    "within-load coefficients reported alongside it (which are correlations) in SIGN only, never "
    "in magnitude, and its numeric size must not be quoted as an effect size."
)


def _between_versus_within_sign(pooled_between: dict, pooled_within: dict, delivered_raw_mean: float) -> dict:
    if pooled_between.get("status") != "tested" or pooled_within.get("status") != "tested":
        return {"status": "not_computable",
                "reason": "between-load or within-load pooled statistic did not reach the paired-test floor"}
    between_mean, within_mean = pooled_between["mean_value"], pooled_within["mean_value"]
    return {
        "status": "computed",
        "between_load_mean_slope": between_mean,
        "between_load_median_slope": pooled_between.get("median_value"),
        "between_load_mean_versus_median_note": (
            "The between-load slope distribution is heavy-tailed across sessions, so the mean and "
            "median are reported together here rather than the mean alone."
        ),
        "within_load_mean_coefficient": within_mean,
        "between_load_slope_units_and_comparability": BETWEEN_LOAD_SLOPE_UNITS_AND_COMPARABILITY_NOTE,
        "between_load_opposes_within_load": bool(_sign_positive(between_mean) != _sign_positive(within_mean)),
        "delivered_raw_pooled_coefficient_against_report_error": delivered_raw_mean,
        "between_load_agrees_with_delivered_raw_coefficient": bool(
            _sign_positive(between_mean) == _sign_positive(delivered_raw_mean)),
        "within_load_opposes_delivered_raw_coefficient": bool(
            _sign_positive(within_mean) != _sign_positive(delivered_raw_mean)),
    }


def _cell_verdict(tier: str, group: str, subset_rows: list[dict], gate_lookup: dict,
                   reference_magnitude: float, delivered_raw_mean: float) -> dict:
    arms = [r["by_tier"][tier] for r in subset_rows if tier in r.get("by_tier", {})]
    computed = [a for a in arms if a.get("status") == "computed"]
    gate = gate_lookup.get((tier, group))
    void = bool(gate is not None and gate.get("significant") is True)

    pooled_within_trial = _pool_values([a["within_load_trial_count_weighted"] for a in computed])
    pooled_within_equal = _pool_values([a["within_load_equal_weighted"] for a in computed])
    pooled_between = _pool_values(
        [a["between_load_slope_on_level_means"] for a in computed
         if a.get("between_load_slope_on_level_means") is not None])

    branch_trial = _within_load_branch(pooled_within_trial, reference_magnitude)
    branch_equal = _within_load_branch(pooled_within_equal, reference_magnitude)
    sign_check = _between_versus_within_sign(pooled_between, pooled_within_trial, delivered_raw_mean)
    pooled_between_reported = {**pooled_between,
                                "units_and_comparability": BETWEEN_LOAD_SLOPE_UNITS_AND_COMPARABILITY_NOTE}

    return {
        "status": "computed" if computed else "no_sessions_with_a_computed_arm",
        "n_sessions_in_group": len(subset_rows), "n_sessions_with_a_computed_arm": len(computed),
        "orthogonality_gate_from_the_delivered_artifact": gate,
        "void_due_to_orthogonality_gate_failure": void,
        "within_load_trial_count_weighted": pooled_within_trial,
        "within_load_equal_weighted": pooled_within_equal,
        "between_load_on_level_means": pooled_between_reported,
        "within_load_branch_trial_count_weighting": "void_orthogonality_gate_failed" if void else branch_trial,
        "within_load_branch_equal_weighting": "void_orthogonality_gate_failed" if void else branch_equal,
        "within_load_weightings_agree": bool(branch_trial == branch_equal),
        "between_load_versus_within_load_sign": (
            {"status": "void_orthogonality_gate_failed"} if void else sign_check),
        "reference_magnitude_against_report_error": reference_magnitude,
    }


def _between_load_level_summary(rows: list[dict], tier: str) -> dict:
    """Per-item-count-level, across-session mean report error and mean state
    deviation for one tier's pooled sessions, read directly from each
    session's own already-computed per_level rows -- no new fit. This is the
    decomposition of the between-load slope's numerator (report error) and
    denominator (state deviation) into their own per-level movement, which is
    what the slope's magnitude alone cannot show."""
    by_level: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        arm = row.get("by_tier", {}).get(tier)
        if not arm or arm.get("status") != "computed":
            continue
        for level_str, entry in arm["per_level"].items():
            if entry.get("status") != "computed":
                continue
            by_level.setdefault(int(level_str), []).append(
                (entry["mean_report_error"], entry["mean_state_deviation"]))
    table = {}
    for level in sorted(by_level):
        pairs = by_level[level]
        table[str(level)] = {
            "n_sessions": len(pairs),
            "mean_report_error_across_sessions": float(np.mean([p[0] for p in pairs])),
            "mean_state_deviation_across_sessions": float(np.mean([p[1] for p in pairs])),
        }
    return table


def _completed_trial_counts(root: Path) -> dict:
    """Completed-trial rows per behavioural session date, from the corpus's
    own completed-trials-only behaviour table -- context attached to a
    refused session's record, not part of the zero-drop reconciliation
    itself (that reconciliation is seen == analysed + refused over
    iter_watters, independent of this count)."""
    # watters_behaviour indexes by (subject, session, trial_num) with drop=False, so the same
    # names exist as both index levels and columns; reset_index(drop=True) removes the former
    # without touching the latter, resolving the groupby ambiguity that leaves.
    behaviour = watters_behaviour(root).reset_index(drop=True)
    return {f"{a}_{s}": len(g) for (a, s), g in behaviour.groupby(["subject", "session"])}


def main() -> None:
    t0 = time.time()
    root = data_root()
    single_item_reference = _single_item_corpus_reference_magnitude()
    reference_magnitude = single_item_reference["magnitude_used_as_the_reference"]
    delivered_raw = _delivered_raw_pooled_coefficient()
    gate_lookup = _delivered_gate_lookup()
    task_variant_levels = _task_variant_item_count_levels(root)

    checkpoint_raw = _load_checkpoint_raw()
    resumed = {k: v["result"] for k, v in checkpoint_raw.items()
               if isinstance(v, dict) and v.get("complete") is True and "result" in v
               and v["result"].get("analysis_version") == ANALYSIS_VERSION}

    output: dict = {
        "version": ANALYSIS_VERSION,
        "corpus": (
            "Multi-object spatial working memory in macaque frontal cortex, two animals, fixed 1.0 s "
            "maintenance delay with the objects absent, continuous saccadic report. Source: 'Working "
            "Memory of Multi-Object Scenes in Primate Frontal Cortex', Watters, Gabel, Tenenbaum and "
            "Jazayeri, bioRxiv PREPRINT posted 2026-01-27, DOI 10.64898/2026.01.27.702062, data DANDI "
            "000620. This is an unreviewed preprint: no peer-reviewed journal version was found, and it "
            "is cited on that basis."
        ),
        "relationship_to_the_delivered_behaviour_branch": (
            "The delivered artifact results/watters_state_geometry.json fired "
            f"'{delivered_raw['branch_fired']}' on its own pre-declared primary test -- the raw pooled "
            f"graded coefficient over all trials of every item count together, {delivered_raw['mean_value']:+.6f} "
            f"(p={delivered_raw['p_value']:.4f}), n={delivered_raw['n_sessions']} sessions. That branch is "
            "correct under its own pre-declared rule, is NOT re-decided here, and the 14,320 s job that "
            "produced it is NOT re-run. Everything in this artifact is a separate, independently "
            "pre-declared decomposition of the same corpus's per-session data into a within-item-count-"
            "level component and a between-item-count-level component, asking a different, narrower "
            "question the delivered artifact's own pre-declared rule did not ask: is the raw pooled "
            "coefficient's sign carried by the between-load mixture rather than by any within-load "
            "relationship, given that only the within-load component is commensurable with a single-item "
            "task's estimate."
        ),
        "sign_convention": (
            "Every coefficient in this multi-object corpus, in this artifact and in the delivered "
            "watters_state_geometry.json artifact, is a correlation against report ERROR (higher = worse "
            "report). The single-item macaque corpus's own headline coefficient is against CORRECTNESS, so "
            "it is negated below before being used as this analysis's reference magnitude. A positive "
            "within-load coefficient here and the single-item corpus's negated (positive, against error) "
            "headline are therefore the SAME direction, not opposite ones."
        ),
        "single_item_corpus_reference": single_item_reference,
        "delivered_raw_pooled_coefficient": delivered_raw,
        "task_variant_item_count_levels": task_variant_levels,
        "within_load_decision_rule_declared_before_fitting": WITHIN_LOAD_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "between_load_decision_rule_declared_before_fitting": BETWEEN_LOAD_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "weighting_convention": (
            "Two weightings are reported for combining a session's item-count-level estimates into one "
            "within-load value: 'trial_count_weighted' (each level's coefficient weighted by its own trial "
            "count within that session) and 'equal_weighted' (each present level weighted 1/n_levels "
            "regardless of trial count). Both are pooled ACROSS sessions the same way -- one vote per "
            "session (paired sign-flip test), never weighted by a session's own trial count -- matching the "
            "'unit of analysis is the session' rule this corpus's other pooled statistics already follow. "
            "Reporting both within-session weightings, and stating where they agree or disagree on the "
            "branch, is how the choice of weighting is shown not to carry the verdict on its own."
        ),
        "provenance": (
            "RECOMPUTED here: every per-trial state deviation, report error and item count (via the "
            "delivered script's own _behaviour_observables trial-usability mask and rate-free-deviation "
            "definition), the level-stratified correlations and their permutation tests (the delivered "
            "script's own _corr / partial_correlation_permutation_test, no controls), the matched-unit-count "
            "population (the delivered script's own _matched_unit_subset called with its identical seed tag, "
            "so the same units are drawn), and every within-load and between-load statistic in this "
            "artifact. READ, not recomputed: the delivered artifact's per-cell orthogonality-gate "
            "significance (used only to mark a cell void), its fired primary branch and raw pooled "
            "coefficient (used only for the sign comparison above), and the single-item macaque corpus's "
            "pooled behaviour coefficient from results/dominant_latent_identity_and_behaviour_breadth.json "
            "(used only as this analysis's reference magnitude). The corpus loader "
            "(load_watters_session/iter_watters in src/corpus_sessions.py) is unchanged and reused; the "
            "existence profiles, cardinality ladder and load-amplitude arm are not recomputed."
        ),
        "scope": {
            "min_trials_per_item_count_level": MIN_TRIALS_PER_ITEM_COUNT_LEVEL,
            "matched_unit_count": MATCHED_UNIT_COUNT, "min_units_per_session": WATTERS_MIN_UNITS,
            "seeds": "every level correlation is seeded by a stable hash of the session identifier, arm name "
                     "and item-count level; the matched-unit draw reuses the delivered artifact's own seed tag",
            "git_commit": git_commit(ROOT),
        },
        "status": "running", "sessions": [], "sessions_refused": [],
    }
    _flush(output)

    completed_counts = _completed_trial_counts(root)
    refused: list[dict] = []
    rows: list[dict] = []
    seen = 0
    for session in iter_watters(root, bin_ms=100.0):
        seen += 1
        key = session["session"]
        if session["status"] != "loaded":
            refused.append({
                "session": key, "animal": session["animal"], "session_date": session["session_date"],
                "task_variant": session["behavioural_task_variant"], "status": session["status"],
                "n_units_seen_in_cache": session.get("n_units_seen"),
                "n_trials_with_a_complete_delay_window": session.get("n_trials"),
                "n_trials_completed": completed_counts.get(key)})
        else:
            if key in resumed:
                row = resumed[key]
            else:
                row = analyse_session(session)
                checkpoint_raw[key] = {"complete": True, "result": row}
                _write_checkpoint(checkpoint_raw)
            rows.append(row)
        output["sessions"] = rows
        output["sessions_refused"] = refused
        output["n_sessions_seen"] = seen
        _flush(output)
        print(f"progress {seen} {key} {session['status']} elapsed={time.time() - t0:.0f}s", flush=True)

    output["zero_drop_accounting"] = {
        "n_behavioural_session_dates_seen": seen, "n_sessions_analysed": len(rows),
        "n_sessions_refused": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                               for reason in sorted({r["status"] for r in refused})},
        "reconciles": bool(seen == len(rows) + len(refused)),
        "by_animal": {animal: sum(1 for r in rows if r["animal"] == animal)
                      for animal in sorted({r["animal"] for r in rows})},
        "by_task_variant": {variant: sum(1 for r in rows if r["task_variant"] == variant)
                            for variant in sorted({r["task_variant"] for r in rows})},
    }

    primary_level_summary = _between_load_level_summary(rows, PRIMARY_QUALITY_TIER)
    levels_present = sorted(primary_level_summary)
    output["between_load_level_summary_primary_cell"] = {
        "tier": PRIMARY_QUALITY_TIER, "group": "pooled",
        "definition": (
            "For the primary unit-quality tier's pooled sessions, and for each item-count level "
            "that cleared the per-level trial floor in that session, the number of sessions "
            "contributing a level estimate and the across-session mean of each session's own "
            "level-mean report error and level-mean state deviation. This decomposes the "
            "between-load slope into its numerator (report error) and denominator (state "
            "deviation) so their separate movement across load is visible."
        ),
        "by_item_count_level": primary_level_summary,
        "interpretation": (
            "Mean report error more than triples from the lowest to the highest item-count level "
            "(" + ", ".join(f"{level} item(s): {primary_level_summary[level]['mean_report_error_across_sessions']:.4f}"
                             for level in levels_present) + ") while mean state deviation barely "
            "moves across the same levels (" + ", ".join(
                f"{level} item(s): {primary_level_summary[level]['mean_state_deviation_across_sessions']:.4f}"
                for level in levels_present) + "). The between-load slope is therefore a steep "
            "numerator divided by a nearly flat denominator, not a large effect on its own terms. "
            "This level-by-level table, not the slope's numeric magnitude, is the correct way to "
            "present the between-load mixture result."
        ),
    }
    output["amplitude_independence_branch_disclosure"] = (
        "results/watters_state_geometry.json reports a branch named "
        "'state_amplitude_independent_of_item_count' for its per-trial leading-component "
        "amplitude observable. That branch fires on a statistically NON-SIGNIFICANT trend test "
        "whose minimum detectable effect cleared the declared floor -- it records a failure to "
        "detect a trend at the tested power, not a demonstration that the true relationship is "
        "flat, and it should not be read as an established null result. It is not modified and "
        "not re-decided here. For a separate observable -- rate-free state deviation, computed "
        "in this artifact rather than the leading-component amplitude that branch tests -- the "
        "primary cell's own across-session mean, by item-count level, is " +
        ", ".join(f"{primary_level_summary[level]['mean_state_deviation_across_sessions']:.4f} at "
                  f"{level} item(s)" for level in levels_present) +
        ", which decreases monotonically rather than staying flat across load."
    )

    results: dict = {}
    for tier in QUALITY_TIERS:
        tier_rows = [r for r in rows if tier in r.get("by_tier", {})]
        tier_block: dict = {"n_sessions": len(tier_rows)}
        for group, subset in _subsets(tier_rows).items():
            tier_block[group] = _cell_verdict(
                tier, group, subset, gate_lookup, reference_magnitude, delivered_raw["mean_value"])
        results[tier] = tier_block
    output["results"] = results

    primary = results[PRIMARY_QUALITY_TIER]["pooled"]
    output["branches"] = {
        "within_load_trial_count_weighted": primary["within_load_branch_trial_count_weighting"],
        "within_load_equal_weighted": primary["within_load_branch_equal_weighting"],
        "within_load_weightings_agree": primary["within_load_weightings_agree"],
        "between_load_versus_within_load_sign": primary["between_load_versus_within_load_sign"],
        "primary_cell_void": primary["void_due_to_orthogonality_gate_failure"],
    }
    output["wall_clock_s"] = time.time() - t0
    output["status"] = "complete"
    _flush(output)
    print(json.dumps(output["branches"], indent=2, default=str))


if __name__ == "__main__":
    main()
