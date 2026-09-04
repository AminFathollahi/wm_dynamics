"""run_count_subsampling_ladder.py -- is the rate-free deviation observable's
orthogonality gate against total spike count an instrument property of how
many simultaneously recorded units a preparation has, or a real difference
between preparations?

A companion census (results/dissociation_replication_and_counting_noise.json,
block_b.count_separation_disclosure) found that the gate is non-significant
(passes) in two macaque preparations and significant (fails) in three others
-- a mouse motor-cortex recording and two human single-unit recordings --
and that this split tracks each preparation's median TOTAL spike count per
trial almost perfectly, while it does NOT track median spikes per UNIT per
trial: the mouse preparation fires more per unit than the macaque
preparation that passes, and still fails. That is consistent with a purely
mechanical account -- with few simultaneously recorded units, a trial's
unit-normalised activity direction is dominated by Poisson counting noise
whose relative size shrinks as more units are summed, so the deviation
observable would acquire a spurious dependence on total count that
disappears once enough units are pooled. Between the two groups sits an
unsampled range, [353.0, 989.5] total spikes/trial, that no delivered
dataset's own native firing rate lands inside.

Only the two preparations whose gate currently passes -- a macaque lateral
prefrontal cortex single-item recording and a macaque multi-object-holding
recording -- have MORE spikes/trial than that gap, so only they can be cut
down through it: for each session, units are drawn without replacement down
to a target count chosen to approximate a declared median-total-spike-count
rung, the deviation observable and its orthogonality gate against total
spike count are recomputed FROM SCRATCH on the subsampled unit set (never
approximated from the full-population fit), and the same is done for the
population's dominant-latent amplitude, so any degradation can be shown
specific to the rate-free construction or common to every population
observable. Two controls separate what a unit subsample actually changes
(fewer units, hence a noisier direction estimate) from confounds subsampling
also drags along: a trial-count-only subsample at the full unit set (does
cutting SAMPLE SIZE alone reproduce the same degradation), and a rate-
preserving unit-count-matched split of sessions by their own firing rate
(does firing rate, at fixed unit count, move the gate at all, as the
census's per-unit column already suggests it should not).

No threshold crossing is ever asserted. Every rung's value is reported on
its own; where a transition happens inside the sampled range, only the two
adjacent rungs that bracket it are named.

Reuses, unchanged: rate_free_state_deviation and the orthogonality-gate
convention (a zero-controls partial_correlation_permutation_test of the
deviation against total spike count) that produced the disclosure above;
trial_amplitude_covariates for the dominant-latent amplitude; the delay-
epoch spike-count loader for the single-item macaque corpus and the
multi-object corpus's own per-trial, per-item-count-level observable
assembly (_observable_arrays); the paired sign-flip pooling estimator
across sessions (slope_across_sessions_test) and its companion minimum-
detectable-difference estimator; and the existing reproduction-gate
machinery that re-derives the single-item corpus's headline numbers from
its own delivered .mat files before anything new is read.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters, watters_session_dates  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_dissociation_cross_preparation_test import reproduction_gate  # noqa: E402
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    MIN_TRIALS_WITH_DEFINED_DIRECTION, _load_panichello_for_block_b, _observable_arrays,
)
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import _counts_from_spikes, _panichello_directory, trial_amplitude_covariates  # noqa: E402
from run_watters_state_geometry import MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed  # noqa: E402

OUTPUT_PATH = _ROOT / "results" / "count_subsampling_ladder.json"
CHECKPOINT_DIR = _ROOT / "results" / ".checkpoints" / "run_count_subsampling_ladder"
ANALYSIS_VERSION = "2026-08-21"
REPRODUCTION_TOLERANCE = 1e-6

# ---------------------------------------------------------------------------------------------------
# Ladder rungs, declared before any subsampled number is looked at. Chosen by the census's own median
# total spikes/trial values, not by a target unit count, so every rung is directly commensurable with
# results/dissociation_replication_and_counting_noise.json's own disclosure table. The single-item
# macaque corpus's native (full, no-subsampling) point is the first rung; the multi-object corpus's own
# native point is the second -- both are ALSO the exact numbers the failing corpora's gate is compared
# against below, since they are simply this census's own passing-corpus values.
# ---------------------------------------------------------------------------------------------------
RUNG_TARGETS_MEDIAN_TOTAL_SPIKES_PER_TRIAL = [1847.5, 989.5, 700.0, 500.0, 353.0, 256.25, 132.5]
PANICHELLO_NATIVE_RUNG_TARGET = 1847.5
WATTERS_NATIVE_RUNG_TARGET = 989.5

N_DRAWS_PER_RUNG = 25  # the pre-declared floor; never reduced below this to control cost
# A >=25-draws-per-rung ladder over two corpora, up to seven rungs and (for the multi-object corpus)
# several item-count levels per draw is not tractable at the project's standard n_perm=10000 permutation
# count for every draw -- the same tension results/state_behavior_link.json's own >=200-draw matched-
# unit-count loop hit, resolved there by holding the draw count at its pre-declared floor and reducing an
# internal parameter instead. The parallel choice here: each individual draw's own r is the only thing
# that ever leaves this module (it is averaged across draws within a session before any significance
# decision is made -- see pool_draws_within_session), so a draw's own permutation p-value is never read,
# and a reduced permutation count is used for it. The deterministic, once-per-session NATIVE rung (no
# subsampling at all) keeps the full n_perm=10000 convention, since that value is also compared directly
# against results/dissociation_replication_and_counting_noise.json's own delivered numbers.
N_PERM_PER_DRAW = 2000
N_PERM_NATIVE_RUNG = 10000

# ---------------------------------------------------------------------------------------------------
# The failing-corpora reference numbers this ladder's own named outcomes are compared against, quoted
# from results/dissociation_replication_and_counting_noise.json's block_b (both the per-corpus gate
# effect sizes and the count_separation_disclosure table), following the same convention
# run_dissociation_cross_preparation_test.py's own MACAQUE_* constants use rather than re-reading that
# artifact at run time.
# ---------------------------------------------------------------------------------------------------
HIGHEST_FAILING_CORPUS_TARGET = 353.0  # inagaki_alm5_mouse_ALM's own median total spikes/trial -- the
                                       # highest among the three corpora whose gate is significant
FAILING_CORPORA_GATE_R = {
    "inagaki_alm5_mouse_ALM": -0.2056989552659982,
    "dandi_000469_human": -0.17039588572106096,
    "dandi_001187_human": -0.19121890365509575,
}
FAILING_CORPORA_GATE_P = {
    "inagaki_alm5_mouse_ALM": 0.0004999500049995,
    "dandi_000469_human": 0.028497150284971504,
    "dandi_001187_human": 0.0014998500149985001,
}
FAILING_CORPORA_MEDIAN_TOTAL_SPIKES_PER_TRIAL = {
    "inagaki_alm5_mouse_ALM": 353.0, "dandi_000469_human": 256.25, "dandi_001187_human": 132.5,
}
FAILING_CORPORA_MEDIAN_SPIKES_PER_UNIT_PER_TRIAL = {
    "inagaki_alm5_mouse_ALM": 4.0, "dandi_000469_human": 3.0, "dandi_001187_human": 3.0,
}
# The conservative (smallest-magnitude) failing-corpus gate effect: a null powered against this one is
# powered against every failing corpus, not only the nearest one.
FAILING_REFERENCE_EFFECT_ABS = min(abs(v) for v in FAILING_CORPORA_GATE_R.values())
FAILING_REFERENCE_EFFECT_SOURCE = "dandi_000469_human"

# The passing corpora's own median spikes/unit/trial, quoted the same way, needed for the recording-
# specification block's unit-count translation and reported for all five census preparations together.
CENSUS_MEDIAN_SPIKES_PER_UNIT_PER_TRIAL = {
    "panichello_2024_macaque_lPFC_single_item": 2.0,
    "watters_2026_macaque_multi_object": 6.0,
    **FAILING_CORPORA_MEDIAN_SPIKES_PER_UNIT_PER_TRIAL,
}

# Delivered block_b numbers this module's reproduction check must match before any new number is read
# (results/dissociation_replication_and_counting_noise.json, block_b.corpora).
DELIVERED_PANICHELLO_BLOCK_B_GATE_R = 0.03772845194385749
DELIVERED_PANICHELLO_BLOCK_B_GATE_P = 0.5888411158884111
DELIVERED_PANICHELLO_BLOCK_B_MEDIAN_TOTAL = 1847.5
DELIVERED_PANICHELLO_BLOCK_B_MEDIAN_PER_UNIT = 2.0
DELIVERED_WATTERS_BLOCK_B_GATE_R = -0.001607771613414055
DELIVERED_WATTERS_BLOCK_B_GATE_P = 0.9554044595540446
DELIVERED_WATTERS_BLOCK_B_MEDIAN_TOTAL = 989.5
DELIVERED_WATTERS_BLOCK_B_MEDIAN_PER_UNIT = 6.0

# ---------------------------------------------------------------------------------------------------
# Named outcomes -- declared before any subsampled number is read. Implemented by classify_block_a_branch.
# ---------------------------------------------------------------------------------------------------
BRANCH_FAILS_ONCE_CUT = "the_orthogonality_gate_fails_once_the_passing_corpora_are_cut_to_the_failing_corpora_counts"
BRANCH_SURVIVES = "the_orthogonality_gate_survives_at_the_failing_corpora_counts"
BRANCH_DEGRADES_NO_TRANSITION = "the_gate_degrades_with_no_identifiable_transition_inside_the_sampled_range"
BRANCH_INCONCLUSIVE = "inconclusive_the_subsampled_gate_is_below_its_own_detection_floor"
BRANCH_RULE_GAP = "outcome_pattern_not_covered_by_the_pre_declared_rule"

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per corpus (the single-item macaque corpus and the multi-object macaque corpus separately), the "
    "gate is the deviation-vs-total-spike-count orthogonality test pooled across sessions by the paired "
    f"sign-flip estimator. If the gate is non-significant at the native (full, no-subsampling) rung and "
    f"significant at the rung whose target equals {HIGHEST_FAILING_CORPUS_TARGET} (the highest-count "
    f"failing corpus in the five-corpus census), the branch is '{BRANCH_FAILS_ONCE_CUT}'. Else, if the "
    f"gate is non-significant at every sampled rung down to and including the lowest, AND the lowest "
    f"rung's minimum detectable paired difference at 80% power is below "
    f"{FAILING_REFERENCE_EFFECT_ABS:.4f} r units (the smallest-magnitude gate effect among the three "
    f"failing corpora, {FAILING_REFERENCE_EFFECT_SOURCE}), the branch is '{BRANCH_SURVIVES}' -- a powered "
    "null, stated with that comparison numerically. Else, if all rungs are non-significant but the lowest "
    f"rung's minimum detectable difference is at or above {FAILING_REFERENCE_EFFECT_ABS:.4f}, the branch "
    f"is '{BRANCH_INCONCLUSIVE}'. Else (some rung other than the one matching "
    f"{HIGHEST_FAILING_CORPUS_TARGET} is significant, or the native rung is already significant), the "
    f"branch is '{BRANCH_DEGRADES_NO_TRANSITION}' if the pattern is a monotone drift toward significance "
    f"that never crosses inside the sampled range or crosses only above the gap; a pattern this list does "
    f"not cover is reported as '{BRANCH_RULE_GAP}', with the per-rung numbers, rather than forced onto "
    "the nearest label."
)


# =======================================================================================================
# Pure arithmetic -- unit-target resolution, within-session and within-load pooling, branch
# classification, and the floor-to-unit-count translation. No I/O, fully covered by
# tests/test_count_subsampling_ladder.py before any real corpus is touched.
# =======================================================================================================

def resolve_unit_target(n_units_full: int, native_median_total: float, target_median_total: float) -> dict:
    """How many units, drawn from this session's own full unit set, are needed to approximate a rung's
    target median total spike count per trial, using the session's own average per-unit contribution
    (its full-population median total divided by its full unit count) as the scaling estimate.

    Returns {"status": "unreachable"} if the target exceeds what this session's full unit set can ever
    produce (subsampling only removes spikes, never adds them); {"status": "native", "n_units": N} if the
    target is at or above the session's own full median (use every unit, no subsampling); otherwise
    {"status": "subsample", "n_units": k}, 1 <= k < N. The ACHIEVED median total spike count of the drawn
    subsample -- not this estimate -- is what every caller actually reports; this is only how many units
    to draw, not a claim about what count they will produce."""
    if n_units_full <= 0 or native_median_total <= 0:
        return {"status": "unreachable", "reason": "session_cannot_reach_this_rung_target_count"}
    if target_median_total > native_median_total:
        return {"status": "unreachable", "reason": "session_cannot_reach_this_rung_target_count"}
    if target_median_total >= native_median_total - 1e-9:
        return {"status": "native", "n_units": int(n_units_full)}
    per_unit_rate = native_median_total / n_units_full
    if per_unit_rate <= 0:
        return {"status": "unreachable", "reason": "session_cannot_reach_this_rung_target_count"}
    k = int(round(target_median_total / per_unit_rate))
    k = max(1, min(n_units_full, k))
    if k >= n_units_full:
        return {"status": "native", "n_units": int(n_units_full)}
    return {"status": "subsample", "n_units": k}


def pool_draws_within_session(draw_values: list) -> float | None:
    """Mean of a session's own repeated-draw correlation coefficients -- the within-session pooling step
    that runs BEFORE any cross-session significance test, so a session is represented by one value per
    rung regardless of how many independent unit subsamples were drawn for it. None/NaN draws (a
    not-computable fit) are dropped rather than propagated; a session with no usable draw at all pools to
    None, which the caller must treat as an exclusion, not a zero."""
    finite = [float(v) for v in draw_values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def combine_within_load_trial_weighted(level_results: dict) -> float | None:
    """Combines per-item-count-level correlation coefficients into one within-load value by trial-count
    weighting -- the convention this project's multi-object corpus analyses use throughout, so a level
    with more trials contributes proportionally more. ``level_results`` maps an item-count level to
    {"n_trials": int, "r": float or None}; levels with r=None (too few trials, or a not-computable fit)
    are excluded from both the numerator and the weight total."""
    tested = [(lv["n_trials"], lv["r"]) for lv in level_results.values()
              if lv.get("r") is not None and np.isfinite(lv["r"])]
    if not tested:
        return None
    n = np.array([t[0] for t in tested], dtype=float)
    r = np.array([t[1] for t in tested], dtype=float)
    return float(np.sum((n / n.sum()) * r))


def classify_block_a_branch(rungs: list, highest_failing_target: float, failing_reference_effect_abs: float) -> dict:
    """Implements DECISION_RULE_DECLARED_BEFORE_FITTING. ``rungs``: one dict per sampled rung, each with
    'target' (float), 'significant' (bool, or None if the pooled gate was not computable/underpowered by
    construction at that rung) and 'mdd' (float or None). A rung with 'significant' None is excluded from
    the pattern read (its pooled gate could not be tested at all -- see 'zero_drop' reporting elsewhere
    for why), so the branch is decided on whichever rungs a pooled test actually ran at."""
    tested = [r for r in rungs if r.get("significant") is not None]
    if not tested:
        return {"branch": "not_computable", "reason": "no rung produced a testable pooled gate"}
    tested_sorted = sorted(tested, key=lambda r: -r["target"])
    full = tested_sorted[0]
    lowest = tested_sorted[-1]
    matching = next((r for r in tested_sorted if r["target"] == highest_failing_target), None)
    all_non_significant = all(not r["significant"] for r in tested_sorted)

    if (not full["significant"]) and matching is not None and matching["significant"]:
        lowest_still_passing = max((r["target"] for r in tested_sorted if not r["significant"]), default=None)
        return {"branch": BRANCH_FAILS_ONCE_CUT, "lowest_still_non_significant_target": lowest_still_passing}

    if all_non_significant:
        lowest_mdd = lowest.get("mdd")
        powered = lowest_mdd is not None and lowest_mdd < failing_reference_effect_abs
        return {
            "branch": BRANCH_SURVIVES if powered else BRANCH_INCONCLUSIVE,
            "lowest_rung_mdd": lowest_mdd, "failing_reference_effect_abs": failing_reference_effect_abs,
        }

    if full["significant"] and matching is not None and matching["significant"]:
        return {"branch": BRANCH_RULE_GAP,
                "reason": "gate is already significant at the native rung as well as at the rung matching "
                          "the highest failing corpus's count -- the first named outcome requires the "
                          "native rung to be non-significant, and this pattern does not fit any other "
                          "named outcome either"}

    return {"branch": BRANCH_DEGRADES_NO_TRANSITION}


def translate_floor_to_unit_count(floor_total_spikes_per_trial: float, per_unit_rate: float) -> float | None:
    """The minimum simultaneously recorded unit count needed to reach a given total-spike-count floor, at
    a preparation's own observed median spikes/unit/trial -- pure division, kept as its own named function
    since it is the arithmetic the recording-specification block is built on and is worth testing on its
    own. None if the rate is zero or unknown (no recording-spec claim can be made for that preparation)."""
    if per_unit_rate is None or per_unit_rate <= 0:
        return None
    return float(floor_total_spikes_per_trial / per_unit_rate)


def _pool_across_sessions(values: list) -> dict:
    """Pools per-session correlation coefficients across sessions by the paired sign-flip estimator
    against zero, with its companion minimum-detectable-paired-difference at 80% power attached --
    exactly the combination results/dissociation_replication_and_counting_noise.json's and
    results/watters_state_geometry.json's own per-session pooling helpers use (slope_across_sessions_test
    + minimum_detectable_paired_difference), reused here rather than re-derived."""
    if len(values) < 2:
        return {"status": "not_computable", "n_sessions": len(values)}
    pooled = dict(slope_across_sessions_test(values, alternative="two-sided"))
    pooled["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(values)
    pooled["median_value"] = float(np.median(values))
    return pooled


def _r(entry: dict | None) -> float | None:
    """Extracts the observed correlation coefficient from a partial_correlation_permutation_test-style
    result, or None if that fit was not computable."""
    if entry is None or entry.get("status") != "computed":
        return None
    return entry.get("r")


# =======================================================================================================
# Generic ladder orchestration -- corpus-agnostic. The corpus-specific per-draw statistic computation
# (which observable, which behaviour variable, whether item-count levels need splitting) is supplied by
# the caller as ``compute_draw_stats``; this function only handles which units to draw, how many times,
# checkpointed persistence, within-session pooling and across-session pooling.
# =======================================================================================================

def run_corpus_ladder(sessions: list, rung_targets: list, compute_draw_stats, n_draws: int, n_perm_draw: int,
                       n_perm_native: int, seed_namespace: str, checkpoint=None,
                       native_rung_target: float | None = None) -> dict:
    """Runs the unit-subsampling ladder over one corpus's sessions.

    sessions: list of {"session": id, "n_units_full": int, "native_median_total": float, ...corpus-
    specific fields compute_draw_stats needs...}.

    compute_draw_stats(session, unit_indices, n_perm, seed_tag) -> dict or None. ``unit_indices`` is None
    for a native (no-subsampling) draw, else a 1-D array of indices into the session's full unit set. The
    returned dict carries 'gate_dev' and, where computable, 'raw_dev', 'gate_amp', 'raw_amp' -- each a
    partial_correlation_permutation_test-style dict (status/r) or an equivalent combined-across-levels
    dict with the same two keys -- plus 'n_trials_with_defined_direction', 'achieved_median_total' and
    'achieved_median_per_unit'. A return of None means this draw could not be evaluated at all (recorded
    as an exclusion, never silently dropped).

    ``native_rung_target``, if given, forces every session to draw its FULL unit set (one deterministic
    evaluation, no subsampling) at that one rung target, regardless of that session's own native median --
    the corpus's own native/full condition, which by construction every session can always reach.

    ``checkpoint``, if given, is a (key, compute) -> value callable persisting one fit per session per
    rung per draw; if omitted, every draw is computed fresh (used only by the unit tests)."""
    if checkpoint is None:
        checkpoint = lambda key, compute: compute()

    by_rung = {}
    for target in rung_targets:
        session_rows = []
        for session in sessions:
            if native_rung_target is not None and target == native_rung_target:
                resolved = {"status": "native", "n_units": session["n_units_full"]}
            else:
                resolved = resolve_unit_target(session["n_units_full"], session["native_median_total"], target)
            if resolved["status"] == "unreachable":
                session_rows.append({"session": session["session"], "status": "unreachable",
                                      "reason": resolved.get("reason", "session_cannot_reach_this_rung_target_count")})
                continue
            is_native = resolved["status"] == "native"
            k = resolved["n_units"]
            n_draws_here = 1 if is_native else n_draws
            n_perm = n_perm_native if is_native else n_perm_draw

            gate_dev, raw_dev, gate_amp, raw_amp = [], [], [], []
            achieved_totals, achieved_per_unit, achieved_n_trials = [], [], []
            for d in range(n_draws_here):
                seed_tag = f"{seed_namespace}|{session['session']}|rung={target}|draw={d}"
                if is_native:
                    unit_indices = None
                else:
                    rng = np.random.default_rng(stable_seed(f"{seed_tag}|units"))
                    unit_indices = np.sort(rng.choice(session["n_units_full"], size=k, replace=False))

                def _compute(session=session, unit_indices=unit_indices, n_perm=n_perm, seed_tag=seed_tag):
                    return compute_draw_stats(session, unit_indices, n_perm, seed_tag)

                checkpoint_key = f"{seed_namespace}|{session['session']}|rung={target}|draw={d}"
                stats = checkpoint(checkpoint_key, _compute)
                if stats is None:
                    continue
                gate_dev.append(_r(stats.get("gate_dev")))
                raw_dev.append(_r(stats.get("raw_dev")))
                gate_amp.append(_r(stats.get("gate_amp")))
                raw_amp.append(_r(stats.get("raw_amp")))
                if stats.get("achieved_median_total") is not None:
                    achieved_totals.append(stats["achieved_median_total"])
                if stats.get("achieved_median_per_unit") is not None:
                    achieved_per_unit.append(stats["achieved_median_per_unit"])
                if stats.get("n_trials_with_defined_direction") is not None:
                    achieved_n_trials.append(stats["n_trials_with_defined_direction"])

            pooled_gate_dev = pool_draws_within_session(gate_dev)
            if pooled_gate_dev is None and not achieved_totals:
                session_rows.append({"session": session["session"], "status": "no_draw_computable",
                                      "n_units_target": k, "is_native": is_native})
                continue
            session_rows.append({
                "session": session["session"], "status": "computed", "n_units_target": k, "is_native": is_native,
                "n_draws_usable": len(achieved_totals),
                "gate_dev_r": pooled_gate_dev, "raw_dev_r": pool_draws_within_session(raw_dev),
                "gate_amp_r": pool_draws_within_session(gate_amp), "raw_amp_r": pool_draws_within_session(raw_amp),
                "achieved_median_total": float(np.median(achieved_totals)) if achieved_totals else None,
                "achieved_median_per_unit": float(np.median(achieved_per_unit)) if achieved_per_unit else None,
                "achieved_median_n_trials": float(np.median(achieved_n_trials)) if achieved_n_trials else None,
            })

        computed = [r for r in session_rows if r["status"] == "computed"]
        pooled_gate_dev = _pool_across_sessions([r["gate_dev_r"] for r in computed if r["gate_dev_r"] is not None])
        pooled_raw_dev = _pool_across_sessions([r["raw_dev_r"] for r in computed if r["raw_dev_r"] is not None])
        pooled_gate_amp = _pool_across_sessions([r["gate_amp_r"] for r in computed if r["gate_amp_r"] is not None])
        pooled_raw_amp = _pool_across_sessions([r["raw_amp_r"] for r in computed if r["raw_amp_r"] is not None])

        totals = [r["achieved_median_total"] for r in computed if r["achieved_median_total"] is not None]
        per_unit = [r["achieved_median_per_unit"] for r in computed if r["achieved_median_per_unit"] is not None]

        by_rung[target] = {
            "target_median_total_spikes_per_trial": target,
            "is_the_corpus_native_rung": native_rung_target is not None and target == native_rung_target,
            "sessions": session_rows,
            "n_sessions_seen": len(sessions),
            "n_sessions_unreachable": sum(1 for r in session_rows if r["status"] == "unreachable"),
            "n_sessions_no_draw_computable": sum(1 for r in session_rows if r["status"] == "no_draw_computable"),
            "n_sessions_computed": len(computed),
            "achieved_median_total_spikes_per_trial": float(np.median(totals)) if totals else None,
            "achieved_median_spikes_per_unit_per_trial": float(np.median(per_unit)) if per_unit else None,
            "achieved_median_unit_count": float(np.median([r["n_units_target"] for r in computed])) if computed else None,
            "pooled_gate_dev": pooled_gate_dev, "pooled_raw_dev": pooled_raw_dev,
            "pooled_gate_amp": pooled_gate_amp, "pooled_raw_amp": pooled_raw_amp,
        }

    return {"by_rung": by_rung}


# =======================================================================================================
# Corpus-specific loading and per-draw statistic computation.
# =======================================================================================================

def _load_panichello_sessions(root: Path) -> list:
    """Every single-item macaque lPFC session on disk with its full delay-epoch per-bin spike tensor and
    trial correctness label -- the same session set results/dissociation_replication_and_counting_noise
    .json's own block_b gate table is built from (_load_panichello_for_block_b), extended here with
    per-trial correctness (needed for the raw deviation-to-behaviour statistic, which that table never
    computes) and kept as a per-bin tensor rather than pre-summed (needed for the dominant-latent
    amplitude covariate, which is fit on the trial x window matrix, not the trial x unit total)."""
    directory = _panichello_directory(root)
    sessions, refused = [], []
    if directory is None:
        return sessions, refused
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        try:
            raw = loadmat(path, simplify_cells=True)
            spikes = np.asarray(raw["spks"], dtype=float)
            time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
            is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1).astype(float)
            counts_all = _counts_from_spikes(spikes, time_ms)
        except (KeyError, ValueError) as exc:
            refused.append({"session": Path(path).stem, "reason": f"load_failed: {exc}"})
            continue
        activity = counts_all.sum(axis=2)
        sessions.append({
            "session": Path(path).stem, "counts": counts_all, "is_corr": is_corr,
            "n_units_full": int(counts_all.shape[1]),
            "native_median_total": float(np.median(activity.sum(axis=1))),
            "native_median_per_unit": float(np.median(activity)),
        })
    return sessions, refused


def _panichello_draw_stats(session: dict, unit_indices, n_perm: int, seed_tag: str) -> dict | None:
    counts_full = session["counts"]
    is_corr = session["is_corr"]
    counts = counts_full if unit_indices is None else counts_full[:, unit_indices]
    activity = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    total = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    n_finite = int(finite.sum())
    achieved = {"achieved_median_total": float(np.median(total)), "achieved_median_per_unit": float(np.median(activity))}
    if n_finite < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    gate_dev = partial_correlation_permutation_test(
        deviation[finite], total[finite], controls=[], n_perm=n_perm,
        rng=np.random.default_rng(stable_seed(f"{seed_tag}|gate_dev")))
    raw_dev = partial_correlation_permutation_test(
        is_corr[finite], deviation[finite], controls=[], n_perm=n_perm,
        rng=np.random.default_rng(stable_seed(f"{seed_tag}|raw_dev")))
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] == "computed":
        gain = np.asarray(covariates["leading_component_score_gain"], dtype=float)
        gate_amp = partial_correlation_permutation_test(
            gain, total, controls=[], n_perm=n_perm, rng=np.random.default_rng(stable_seed(f"{seed_tag}|gate_amp")))
        raw_amp = partial_correlation_permutation_test(
            is_corr, gain, controls=[], n_perm=n_perm, rng=np.random.default_rng(stable_seed(f"{seed_tag}|raw_amp")))
    else:
        gate_amp = {"status": "not_computable", "reason": covariates.get("reason")}
        raw_amp = {"status": "not_computable", "reason": covariates.get("reason")}
    return {"gate_dev": gate_dev, "raw_dev": raw_dev, "gate_amp": gate_amp, "raw_amp": raw_amp,
            "n_trials_with_defined_direction": n_finite, **achieved}


def _load_watters_sessions(root: Path) -> tuple:
    sessions, refused = [], []
    for entry in iter_watters(root):
        session_id = f"{entry.get('animal', '?')}_{entry.get('session_date', '?')}"
        if entry.get("status") != "loaded":
            refused.append({"session": session_id, "reason": entry.get("status")})
            continue
        counts = entry["counts"]
        activity = counts.sum(axis=2)
        sessions.append({
            "session": entry["session"], "counts": counts,
            "num_objects": entry["num_objects"], "report_deviation": entry["report_deviation"],
            "reaction_time_ms": entry["reaction_time_ms"], "correct": entry["correct"],
            "n_units_full": int(counts.shape[1]),
            "native_median_total": float(np.median(activity.sum(axis=1))),
            "native_median_per_unit": float(np.median(activity)),
        })
    return sessions, refused


def _within_load_stat(arrays: dict, observable_key: str, stat_kind: str, n_perm: int, seed_tag: str) -> dict:
    """One statistic ('gate': observable vs spike_count, or 'raw': report_error vs observable), for one
    observable ('deviation' or 'amplitude'), combined across item-count levels by trial-count weighting
    -- the convention this corpus's analyses use throughout (see combine_within_load_trial_weighted)."""
    item_count = arrays["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    level_results = {}
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            level_results[level] = {"n_trials": n_level, "r": None, "status": "too_few_trials_at_this_item_count"}
            continue
        x = arrays[observable_key][mask]
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|level{level}"))
        if stat_kind == "gate":
            entry = partial_correlation_permutation_test(x, arrays["spike_count"][mask], controls=[], n_perm=n_perm, rng=rng)
        else:
            entry = partial_correlation_permutation_test(arrays["report_error"][mask], x, controls=[], n_perm=n_perm, rng=rng)
        level_results[level] = {"n_trials": n_level, "r": _r(entry), "status": entry.get("status")}
    combined = combine_within_load_trial_weighted(level_results)
    if combined is None:
        return {"status": "not_computable", "reason": "no_item_count_level_reached_the_trial_minimum",
                "per_level": level_results}
    return {"status": "computed", "r": combined, "per_level": level_results,
            "n_levels_tested": sum(1 for lv in level_results.values() if lv["r"] is not None)}


def _watters_draw_stats(session: dict, unit_indices, n_perm: int, seed_tag: str) -> dict | None:
    counts_full = session["counts"]
    counts = counts_full if unit_indices is None else counts_full[:, unit_indices]
    activity = counts.sum(axis=2)
    total_full = activity.sum(axis=1).astype(float)
    achieved = {"achieved_median_total": float(np.median(total_full)), "achieved_median_per_unit": float(np.median(activity))}

    arrays, _excluded, usable = _observable_arrays(counts, session)
    if arrays is None:
        return {"gate_dev": {"status": "not_computable"}, "raw_dev": {"status": "not_computable"},
                "gate_amp": {"status": "not_computable"}, "raw_amp": {"status": "not_computable"},
                "n_trials_with_defined_direction": int(usable.sum()), **achieved}

    gate_dev = _within_load_stat(arrays, "deviation", "gate", n_perm, f"{seed_tag}|dev_gate")
    raw_dev = _within_load_stat(arrays, "deviation", "raw", n_perm, f"{seed_tag}|dev_raw")
    gate_amp = _within_load_stat(arrays, "amplitude", "gate", n_perm, f"{seed_tag}|amp_gate")
    raw_amp = _within_load_stat(arrays, "amplitude", "raw", n_perm, f"{seed_tag}|amp_raw")
    return {"gate_dev": gate_dev, "raw_dev": raw_dev, "gate_amp": gate_amp, "raw_amp": raw_amp,
            "n_trials_with_defined_direction": int(usable.sum()), **achieved}


# =======================================================================================================
# Block B -- separate what a unit subsample changes (fewer units, hence a noisier direction estimate)
# from confounds subsampling drags along.
# =======================================================================================================

def _panichello_trial_subsample_stats(session: dict, n_trials_target: int, n_perm: int, seed_tag: str) -> dict | None:
    counts_full, is_corr = session["counts"], session["is_corr"]
    n_total = counts_full.shape[0]
    if n_trials_target >= n_total:
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|trials"))
        idx = np.sort(rng.choice(n_total, size=n_trials_target, replace=False))
    activity = counts_full[idx].sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    total = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    gate = partial_correlation_permutation_test(
        deviation[finite], total[finite], controls=[], n_perm=n_perm,
        rng=np.random.default_rng(stable_seed(f"{seed_tag}|gate")))
    return {"gate_dev": gate, "n_trials_with_defined_direction": int(finite.sum())}


def _watters_trial_subsample_stats(session: dict, n_trials_target: int, n_perm: int, seed_tag: str) -> dict | None:
    counts_full = session["counts"]
    n_total = counts_full.shape[0]
    if n_trials_target >= n_total:
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|trials"))
        idx = np.sort(rng.choice(n_total, size=n_trials_target, replace=False))
    activity = counts_full[idx].sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    total = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    gate = partial_correlation_permutation_test(
        deviation[finite], total[finite], controls=[], n_perm=n_perm,
        rng=np.random.default_rng(stable_seed(f"{seed_tag}|gate")))
    return {"gate_dev": gate, "n_trials_with_defined_direction": int(finite.sum())}


def run_trial_subsampling_control(sessions: list, ladder_by_rung: dict, trial_stats_fn, n_draws: int,
                                   n_perm: int, seed_namespace: str, checkpoint=None) -> dict:
    """At the FULL unit set, subsamples TRIALS (never units) down to the median effective trial count the
    unit-subsampling ladder ended up with at each rung (ladder_by_rung[target]['achieved_median_n_trials']
    per session), and recomputes the deviation-vs-spike-count gate. If this control ALSO makes the gate
    fail at the same rungs the unit ladder does, the ladder's own degradation cannot be distinguished from
    a plain reduction in the correlation test's sample size, and its branch is void by name."""
    if checkpoint is None:
        checkpoint = lambda key, compute: compute()
    session_by_id = {s["session"]: s for s in sessions}
    by_rung = {}
    for target, rung in ladder_by_rung.items():
        targets_by_session = {r["session"]: r.get("achieved_median_n_trials")
                               for r in rung["sessions"] if r.get("status") == "computed"}
        session_rows = []
        for session_id, n_target in targets_by_session.items():
            session = session_by_id.get(session_id)
            if session is None or n_target is None:
                continue
            draws = []
            n_trials_target = int(round(n_target))
            for d in range(n_draws):
                seed_tag = f"{seed_namespace}|{session_id}|rung={target}|draw={d}"

                def _compute(session=session, n_trials_target=n_trials_target, seed_tag=seed_tag):
                    return trial_stats_fn(session, n_trials_target, n_perm, seed_tag)

                stats = checkpoint(f"trial_control|{seed_namespace}|{session_id}|rung={target}|draw={d}", _compute)
                if stats is not None:
                    draws.append(_r(stats.get("gate_dev")))
            pooled_r = pool_draws_within_session(draws)
            session_rows.append({"session": session_id, "n_trials_target": n_trials_target,
                                  "status": "computed" if pooled_r is not None else "no_draw_computable",
                                  "gate_dev_r": pooled_r})
        computed = [r for r in session_rows if r["status"] == "computed"]
        pooled_gate = _pool_across_sessions([r["gate_dev_r"] for r in computed if r["gate_dev_r"] is not None])
        by_rung[target] = {"sessions": session_rows, "n_sessions_computed": len(computed), "pooled_gate_dev": pooled_gate}
    return {"by_rung": by_rung}


def run_rate_preserving_control(sessions: list, fixed_unit_count: int, draw_stats_fn, n_draws: int, n_perm: int,
                                 seed_namespace: str, checkpoint=None) -> dict:
    """At one FIXED unit count reachable by every session in the corpus, splits sessions into upper and
    lower halves by their own NATIVE (full-population, never subsampled) median spikes/unit/trial, and
    pools the deviation-vs-spike-count gate separately within each half -- a direct test of whether firing
    rate at matched unit count moves the gate at all, independent of the ladder's own ordered rungs."""
    if checkpoint is None:
        checkpoint = lambda key, compute: compute()
    session_rows = []
    for session in sessions:
        if session["n_units_full"] < fixed_unit_count:
            session_rows.append({"session": session["session"], "status": "too_few_units_for_fixed_count"})
            continue
        draws = []
        for d in range(n_draws):
            seed_tag = f"{seed_namespace}|{session['session']}|fixed{fixed_unit_count}|draw={d}"
            rng = np.random.default_rng(stable_seed(f"{seed_tag}|units"))
            unit_indices = np.sort(rng.choice(session["n_units_full"], size=fixed_unit_count, replace=False))

            def _compute(session=session, unit_indices=unit_indices, seed_tag=seed_tag):
                return draw_stats_fn(session, unit_indices, n_perm, seed_tag)

            stats = checkpoint(f"rate_control|{seed_namespace}|{session['session']}|draw={d}", _compute)
            if stats is not None:
                draws.append(_r(stats.get("gate_dev")))
        pooled_r = pool_draws_within_session(draws)
        session_rows.append({
            "session": session["session"], "status": "computed" if pooled_r is not None else "no_draw_computable",
            "gate_dev_r": pooled_r, "native_median_spikes_per_unit_per_trial": session["native_median_per_unit"],
        })
    computed = [r for r in session_rows if r["status"] == "computed"]
    if len(computed) < 4:
        return {"status": "not_computable", "sessions": session_rows, "n_sessions_computed": len(computed)}
    median_rate = float(np.median([r["native_median_spikes_per_unit_per_trial"] for r in computed]))
    upper = [r for r in computed if r["native_median_spikes_per_unit_per_trial"] >= median_rate]
    lower = [r for r in computed if r["native_median_spikes_per_unit_per_trial"] < median_rate]
    return {
        "status": "computed", "fixed_unit_count": fixed_unit_count, "median_split_rate": median_rate,
        "sessions": session_rows, "n_sessions_computed": len(computed),
        "upper_half": {"n_sessions": len(upper),
                       "pooled_gate_dev": _pool_across_sessions([r["gate_dev_r"] for r in upper])},
        "lower_half": {"n_sessions": len(lower),
                       "pooled_gate_dev": _pool_across_sessions([r["gate_dev_r"] for r in lower])},
    }


# =======================================================================================================
# Recording specification -- translates whatever floor the ladder fit into a stated unit-count
# requirement. Fires no new branch and never upgrades a delivered gate verdict.
# =======================================================================================================

def build_recording_specification(ladder_branches: dict, census_per_unit_rates: dict) -> dict:
    """ladder_branches: {corpus_name: branch_info_dict} (classify_block_a_branch's own return value, one
    per corpus this leg ran a ladder on). census_per_unit_rates: {corpus_name: median spikes/unit/trial}
    for every preparation in the five-corpus census (not only the two this leg subsampled), so the floor
    -- wherever it is established -- is translated into a unit-count requirement for every preparation on
    its own observed firing regime."""
    floors = {corpus: info["lowest_still_non_significant_target"]
              for corpus, info in ladder_branches.items()
              if info.get("branch") == BRANCH_FAILS_ONCE_CUT and info.get("lowest_still_non_significant_target") is not None}
    if not floors:
        return {
            "status": "withdrawn",
            "reason": "No corpus's ladder found the orthogonality gate to fail once cut to the failing "
                      "corpora's own counts, so this leg does not establish the gate's cross-preparation "
                      "failure as a unit-count property, and no recording-specification claim is made.",
            "ladder_branches": {corpus: info.get("branch") for corpus, info in ladder_branches.items()},
        }
    required_unit_count_by_preparation = {
        corpus: {source: {"floor_total_spikes_per_trial": floor_value,
                          "required_simultaneous_units": translate_floor_to_unit_count(floor_value, rate)}
                 for source, floor_value in floors.items()}
        for corpus, rate in census_per_unit_rates.items()
    }
    human_preparations_relative_to_floor = {}
    for corpus in ("dandi_000469_human", "dandi_001187_human"):
        observed = FAILING_CORPORA_MEDIAN_TOTAL_SPIKES_PER_TRIAL.get(corpus)
        if observed is None:
            continue
        human_preparations_relative_to_floor[corpus] = {
            source: {
                "human_observed_median_total_spikes_per_trial": observed,
                "floor_total_spikes_per_trial": floor_value,
                "sits_below_floor": bool(observed < floor_value),
                "shortfall_total_spikes_per_trial": floor_value - observed,
                "shortfall_ratio": (floor_value / observed) if observed > 0 else None,
            }
            for source, floor_value in floors.items()
        }
    return {
        "status": "reported",
        "floors_by_source_corpus_total_spikes_per_trial": floors,
        "required_unit_count_by_preparation": required_unit_count_by_preparation,
        "human_preparations_relative_to_floor": human_preparations_relative_to_floor,
        "ladder_branches": {corpus: info.get("branch") for corpus, info in ladder_branches.items()},
    }


# =======================================================================================================
# Reproduction check -- re-runs the shared estimators on the delivered sessions and compares to the
# numbers already recorded in results/dissociation_replication_and_counting_noise.json and
# results/rate_free_state_geometry_behavior_link.json, WITHOUT touching either artifact's own checkpoint
# file (this recomputation calls rate_free_state_deviation and partial_correlation_permutation_test
# directly, the same two shared estimators the delivered pipeline used, rather than going through that
# pipeline's own _fit/checkpoint wrapper, which is scoped to a different results/ file this module must
# not write to).
# =======================================================================================================

def _close(observed, expected, tol=REPRODUCTION_TOLERANCE) -> bool:
    return observed is not None and abs(observed - expected) <= tol


def _reproduce_block_b_real_gate(activity_sessions: list, seed_prefix: str, n_perm: int = 10000) -> dict:
    per_session = []
    for entry in activity_sessions:
        activity = entry["activity_by_unit"]
        deviation = rate_free_state_deviation(activity)
        total = activity.sum(axis=1).astype(float)
        finite = np.isfinite(deviation)
        row = {"session": entry["session"], "median_total_spike_count_per_trial": float(np.median(total)),
               "median_count_per_unit_per_trial": float(np.median(activity))}
        if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            row["status"] = "too_few_trials_with_a_defined_direction"
            per_session.append(row)
            continue
        rng = np.random.default_rng(stable_seed(f"{seed_prefix}|{entry['session']}|real_gate"))
        gate = partial_correlation_permutation_test(deviation[finite], total[finite], controls=[], n_perm=n_perm, rng=rng)
        row["status"] = "computed"
        row["real_deviation_gate"] = gate
        per_session.append(row)
    computed = [r for r in per_session if r["status"] == "computed"]
    real_values = [r["real_deviation_gate"]["r"] for r in computed if r["real_deviation_gate"].get("status") == "computed"]
    pooled = _pool_across_sessions(real_values)
    totals = [r["median_total_spike_count_per_trial"] for r in computed]
    per_unit = [r["median_count_per_unit_per_trial"] for r in computed]
    return {
        "n_sessions_seen": len(activity_sessions), "n_sessions_computed": len(computed),
        "real_gate_mean_value": pooled.get("mean_value"), "real_gate": pooled,
        "median_total_spike_count_per_trial_across_sessions": float(np.median(totals)) if totals else None,
        "median_count_per_unit_per_trial_across_sessions": float(np.median(per_unit)) if per_unit else None,
    }


def run_reproduction_check(root: Path, watters_sessions: list) -> dict:
    eleven_session_check = reproduction_gate(root)

    panichello_block_b_sessions = _load_panichello_for_block_b(root)
    panichello_repro = _reproduce_block_b_real_gate(
        panichello_block_b_sessions, "dissociation_replication_and_counting_noise|panichello")

    watters_activity = [{"session": s["session"], "activity_by_unit": s["counts"].sum(axis=2)} for s in watters_sessions]
    watters_repro = _reproduce_block_b_real_gate(
        watters_activity, "dissociation_replication_and_counting_noise|watters")

    checks = {
        "eleven_session_reproduction_gate": eleven_session_check.get("status") == "reproduced_exactly",
        "panichello_block_b_gate_r": _close(panichello_repro["real_gate_mean_value"], DELIVERED_PANICHELLO_BLOCK_B_GATE_R),
        "panichello_block_b_gate_p": _close(
            panichello_repro["real_gate"].get("two_sided_p_value"), DELIVERED_PANICHELLO_BLOCK_B_GATE_P),
        "panichello_block_b_median_total": _close(
            panichello_repro["median_total_spike_count_per_trial_across_sessions"], DELIVERED_PANICHELLO_BLOCK_B_MEDIAN_TOTAL),
        "panichello_block_b_median_per_unit": _close(
            panichello_repro["median_count_per_unit_per_trial_across_sessions"], DELIVERED_PANICHELLO_BLOCK_B_MEDIAN_PER_UNIT),
        "watters_block_b_gate_r": _close(watters_repro["real_gate_mean_value"], DELIVERED_WATTERS_BLOCK_B_GATE_R),
        "watters_block_b_gate_p": _close(watters_repro["real_gate"].get("two_sided_p_value"), DELIVERED_WATTERS_BLOCK_B_GATE_P),
        "watters_block_b_median_total": _close(
            watters_repro["median_total_spike_count_per_trial_across_sessions"], DELIVERED_WATTERS_BLOCK_B_MEDIAN_TOTAL),
        "watters_block_b_median_per_unit": _close(
            watters_repro["median_count_per_unit_per_trial_across_sessions"], DELIVERED_WATTERS_BLOCK_B_MEDIAN_PER_UNIT),
    }
    status = "reproduced_exactly" if all(checks.values()) else "not_reproduced"
    return {
        "status": status, "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "eleven_session_reproduction_gate": eleven_session_check,
        "panichello_block_b_reproduction": panichello_repro, "watters_block_b_reproduction": watters_repro,
    }


# =======================================================================================================
# Checkpointing -- one file per session per subsampling rung per draw, under its own directory. Never
# deleted; a corrupted or unparseable record counts as absent and is recomputed; the completion flag is
# written only after the fit itself returns, via a temp file and an atomic replace.
# =======================================================================================================

def _checkpoint_path(key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return CHECKPOINT_DIR / f"{digest}.json"


def make_checkpoint():
    def _checkpoint(key: str, compute):
        path = _checkpoint_path(key)
        try:
            record = json.loads(path.read_text())
            if isinstance(record, dict) and record.get("complete") is True and record.get("key") == key:
                return record["value"]
        except (OSError, ValueError, KeyError):
            pass  # absent or unparseable -- recomputed below, never treated as a completed fit
        value = compute()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "complete": True, "value": _json_safe(value)}
        scratch = path.with_suffix(".partial")
        scratch.write_text(json.dumps(record, allow_nan=False))
        os.replace(scratch, path)
        return value
    return _checkpoint


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Driver
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    root = data_root()
    checkpoint = make_checkpoint()

    output = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "The two macaque preparations whose rate-free deviation observable passes its own "
            "orthogonality gate against total spike count -- a single-item lateral prefrontal cortex "
            "recording and a multi-object-holding frontal recording -- cut down through the unsampled "
            "median-total-spike-count range between them and the three preparations whose gate fails (a "
            "mouse motor-cortex recording and two human single-unit recordings), to test whether the "
            "gate's cross-preparation split tracks simultaneously recorded unit count rather than a real "
            "difference between preparations. The multi-object corpus is analysed within item-count "
            "level throughout and combined by trial-count weighting across the levels reaching the trial "
            "minimum, matching the convention its own delivered analyses use."
        ),
        "seed_policy": (
            "Every random draw (which units, which trials) is seeded by a stable hash (stable_seed) of a "
            "descriptive string tag unique to that corpus, session, rung target and draw index, so the "
            "entire ladder is exactly reproducible from this script alone. At least "
            f"{N_DRAWS_PER_RUNG} independent draws are taken per session per subsampled rung (the pre-"
            "declared floor, never reduced); the native (no-subsampling) rung is a single deterministic "
            "evaluation, since there is only one way to use every unit."
        ),
        "n_perm_policy": (
            f"n_perm={N_PERM_NATIVE_RUNG} (the project's standard convention) for the once-per-session "
            f"native rung, whose value is also checked against the reproduction gate below. n_perm="
            f"{N_PERM_PER_DRAW} for every individual subsampled draw: a draw's own r is averaged with its "
            "sibling draws within a session before any significance decision is made (see "
            "pool_draws_within_session), so a single draw's own permutation p-value is never read on its "
            "own, and the reduced count keeps a >=25-draws-per-rung, two-corpus, multi-rung ladder within "
            "a tractable wall clock -- the same tradeoff results/state_behavior_link.json's own >=200-draw "
            "matched-unit-count loop already made, documented there in identical terms."
        ),
        "rung_targets_median_total_spikes_per_trial": RUNG_TARGETS_MEDIAN_TOTAL_SPIKES_PER_TRIAL,
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    print("loading watters sessions...", flush=True)
    watters_sessions, watters_refused = _load_watters_sessions(root)
    watters_seen = len(watters_session_dates(root))
    print(f"watters: {len(watters_sessions)} loaded, {len(watters_refused)} refused, {watters_seen} seen", flush=True)

    print("running reproduction check...", flush=True)
    reproduction = run_reproduction_check(root, watters_sessions)
    output["reproduction_check"] = reproduction
    _flush(output)
    print(f"reproduction check: {reproduction['status']}", flush=True)
    if reproduction["status"] != "reproduced_exactly":
        output["status"] = "void_reproduction_gate_did_not_reproduce"
        output["branch"] = "void_reproduction_gate_did_not_reproduce"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        print("REPRODUCTION DID NOT MATCH -- stopping before any new number is read.", flush=True)
        return

    print("loading panichello sessions...", flush=True)
    panichello_sessions, panichello_refused = _load_panichello_sessions(root)
    print(f"panichello: {len(panichello_sessions)} loaded, {len(panichello_refused)} refused", flush=True)

    corpora = [
        ("panichello_2024_macaque_lPFC_single_item", panichello_sessions, _panichello_draw_stats,
         _panichello_trial_subsample_stats, PANICHELLO_NATIVE_RUNG_TARGET),
        ("watters_2026_macaque_multi_object", watters_sessions, _watters_draw_stats,
         _watters_trial_subsample_stats, WATTERS_NATIVE_RUNG_TARGET),
    ]

    output["unit_subsampling_ladder"] = {}
    output["trial_subsampling_control"] = {}
    output["rate_preserving_unit_control"] = {}
    ladder_branches = {}
    fixed_unit_counts = {}

    for corpus_name, sessions, draw_fn, trial_fn, native_target in corpora:
        print(f"[{corpus_name}] running unit-subsampling ladder ({len(sessions)} sessions)...", flush=True)
        ladder = run_corpus_ladder(
            sessions=sessions, rung_targets=RUNG_TARGETS_MEDIAN_TOTAL_SPIKES_PER_TRIAL,
            compute_draw_stats=draw_fn, n_draws=N_DRAWS_PER_RUNG, n_perm_draw=N_PERM_PER_DRAW,
            n_perm_native=N_PERM_NATIVE_RUNG, seed_namespace=f"count_subsampling_ladder|{corpus_name}",
            checkpoint=checkpoint, native_rung_target=native_target,
        )
        rungs_for_branch = [
            {"target": target, "significant": rung["pooled_gate_dev"].get("significant"),
             "mdd": rung["pooled_gate_dev"].get("minimum_detectable_paired_difference_at_80pct_power", {}).get("mdd")
                    if isinstance(rung["pooled_gate_dev"].get("minimum_detectable_paired_difference_at_80pct_power"), dict) else None,
             "n_sessions": rung["n_sessions_computed"]}
            for target, rung in ladder["by_rung"].items()
        ]
        branch_info = classify_block_a_branch(rungs_for_branch, HIGHEST_FAILING_CORPUS_TARGET, FAILING_REFERENCE_EFFECT_ABS)
        ladder["branch"] = branch_info
        ladder_branches[corpus_name] = branch_info
        output["unit_subsampling_ladder"][corpus_name] = ladder
        output["zero_drop_accounting"] = output.get("zero_drop_accounting", {})
        _flush(output)
        print(f"[{corpus_name}] ladder branch: {branch_info['branch']} elapsed={time.time() - t0:.0f}s", flush=True)

        print(f"[{corpus_name}] running trial-subsampling control...", flush=True)
        trial_control = run_trial_subsampling_control(
            sessions=sessions, ladder_by_rung=ladder["by_rung"], trial_stats_fn=trial_fn,
            n_draws=N_DRAWS_PER_RUNG, n_perm=N_PERM_PER_DRAW,
            seed_namespace=f"count_subsampling_ladder_trial_control|{corpus_name}", checkpoint=checkpoint,
        )
        confounded = any(
            trial_control["by_rung"].get(target, {}).get("pooled_gate_dev", {}).get("significant")
            and not ladder["by_rung"][target]["pooled_gate_dev"].get("significant")
            for target in RUNG_TARGETS_MEDIAN_TOTAL_SPIKES_PER_TRIAL
            if target in trial_control["by_rung"] and target in ladder["by_rung"]
        )
        # A rung where the unit ladder's own gate IS significant but the trial-count-only control at the
        # matching sample size is NOT is the direct, positive evidence that unit count specifically (not
        # sample size in general) drives the ladder's own degradation.
        unit_specific_evidence = any(
            ladder["by_rung"][target]["pooled_gate_dev"].get("significant")
            and trial_control["by_rung"].get(target, {}).get("pooled_gate_dev", {}).get("significant") is False
            for target in RUNG_TARGETS_MEDIAN_TOTAL_SPIKES_PER_TRIAL
            if target in trial_control["by_rung"] and target in ladder["by_rung"]
        )
        trial_control["branch"] = (
            "count_ladder_not_separable_from_a_trial_count_effect" if confounded else
            "count_ladder_separable_from_a_trial_count_effect"
        )
        trial_control["unit_specific_evidence_found"] = unit_specific_evidence
        output["trial_subsampling_control"][corpus_name] = trial_control
        _flush(output)
        print(f"[{corpus_name}] trial control branch: {trial_control['branch']} elapsed={time.time() - t0:.0f}s", flush=True)

        fixed_unit_count = int(min(s["n_units_full"] for s in sessions)) if sessions else None
        fixed_unit_counts[corpus_name] = fixed_unit_count
        print(f"[{corpus_name}] running rate-preserving unit control at fixed unit count {fixed_unit_count}...", flush=True)
        rate_control = run_rate_preserving_control(
            sessions=sessions, fixed_unit_count=fixed_unit_count, draw_stats_fn=draw_fn,
            n_draws=N_DRAWS_PER_RUNG, n_perm=N_PERM_PER_DRAW,
            seed_namespace=f"count_subsampling_ladder_rate_control|{corpus_name}", checkpoint=checkpoint,
        )
        output["rate_preserving_unit_control"][corpus_name] = rate_control
        _flush(output)
        print(f"[{corpus_name}] rate control status: {rate_control['status']} elapsed={time.time() - t0:.0f}s", flush=True)

    print("building recording specification...", flush=True)
    output["recording_specification"] = build_recording_specification(ladder_branches, CENSUS_MEDIAN_SPIKES_PER_UNIT_PER_TRIAL)
    _flush(output)

    output["zero_drop_accounting"] = {
        "watters": {"n_seen": watters_seen, "n_loaded": len(watters_sessions), "n_refused": len(watters_refused),
                    "refused": watters_refused,
                    "reconciles": watters_seen == len(watters_sessions) + len(watters_refused)},
        "panichello": {"n_loaded": len(panichello_sessions), "n_refused": len(panichello_refused),
                       "refused": panichello_refused},
        "fixed_unit_counts_for_rate_preserving_control": fixed_unit_counts,
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({"status": "complete", "ladder_branches": {k: v["branch"] for k, v in ladder_branches.items()},
                       "wall_clock_s": output["wall_clock_s"]}, indent=2))


if __name__ == "__main__":
    main()
