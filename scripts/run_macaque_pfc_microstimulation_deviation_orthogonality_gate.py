#!/usr/bin/env python3
"""Runs the rate-free deviation observable's own orthogonality gate against
total spike count on the macaque dorsolateral prefrontal microstimulation
corpus's control (non-stimulation) trials.

Nothing here is a new statistical test. rate_free_state_deviation
(run_rate_free_state_geometry_behavior_link.py) and partial_correlation_
permutation_test with zero controls (src/statistics.py) are the exact
functions results/dissociation_replication_and_counting_noise.json's own
per-session real-gate loop calls to produce every corpus's gate value in
that census; they are imported and called here unchanged, on this corpus's
own control trials, which no prior artifact has run them on. Session
pooling reuses _pool_values (run_watters_state_geometry.py), which is
itself an unchanged call to slope_across_sessions_test and
minimum_detectable_paired_difference (src/statistics.py /
src/state_persistence.py). The reference effect size a non-significant
pooled result is checked against, so a "passes" outcome is only reported as
a real null rather than an underpowered one, is FAILING_REFERENCE_EFFECT_ABS
(run_count_subsampling_ladder.py) -- the smallest-magnitude gate value among
the three corpora whose gate is already known to fail, imported unchanged
rather than re-picked here.

Per-trial control-trial spike counts are built the identical way
results/stimulation_axis_same_session_census.json's own fresh measurement
of this corpus built them (load_macaque_pfc_microstimulation_session, crop_trial, BIN_S, all
from run_macaque_pfc_microstimulation_pipeline.py, called read-only): each control trial's
30-bin, -0.8 s to +0.7 s stimulation-onset-aligned spikerate array (Hz) is
summed over bins and multiplied by the 0.05 s bin width to recover a
per-channel spike count, and stimulated trials never enter this array at
all. Before any new gate number is reported, every session's freshly
computed median control-trial total spike count is checked against that
already-delivered census's own per-session number for the same quantity;
a mismatch halts the run rather than reporting a gate built on an unverified
trial pool -- this corpus's first prior artifact ships no gate of its own
to reproduce, only that one shared upstream number.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
    scripts/run_macaque_pfc_microstimulation_deviation_orthogonality_gate.py
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import _json_safe, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)
from run_macaque_pfc_microstimulation_pipeline import BIN_S, DATA, crop_trial, load_macaque_pfc_microstimulation_session  # noqa: E402
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_watters_state_geometry import _pool_values  # noqa: E402
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_count_subsampling_ladder import (  # noqa: E402
    FAILING_REFERENCE_EFFECT_ABS, FAILING_REFERENCE_EFFECT_SOURCE,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "macaque_pfc_microstimulation_deviation_orthogonality_gate.json"
CENSUS_PATH = RESULTS / "stimulation_axis_same_session_census.json"

N_PERM = 10000
GATE_ALPHA = 0.05

SPIKE_COUNT_FAILING_AT_OR_BELOW = 353.0
SPIKE_COUNT_PASSING_AT_OR_ABOVE = 989.5
SPIKE_COUNT_PRECONDITION_SOURCE = (
    "results/dissociation_replication_and_counting_noise.json, its count_separation_disclosure field"
)


def _control_activity_by_unit(prefix: str) -> dict:
    """(n_control_trials, n_channels) total-spike-count matrix for CONTROL
    trials only (tr["stim_cond"] == ctrl_idx); stimulated trials are never
    read into this array. Identical construction to the already-delivered
    census's own fresh macaque_pfc_microstimulation measurement."""
    corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr.get("control_idx") is None:
        return {"status": "excluded", "reason": "not_loadable_or_no_control_condition"}
    ctrl_idx = corr["control_idx"]
    rows = []
    for tr in corr["trials"]:
        if tr["stim_cond"] != ctrl_idx:
            continue
        cropped = crop_trial(tr["spikerate"])
        if cropped is None:
            continue
        rows.append(cropped.sum(axis=0) * BIN_S)
    if not rows:
        return {"status": "excluded", "reason": "no_usable_control_trials"}
    activity = np.asarray(rows, dtype=float)
    return {"status": "loaded", "activity_by_unit": activity, "n_channels": int(activity.shape[1])}


def _session_gate(session: str, activity: np.ndarray) -> dict:
    """The gate itself: rate_free_state_deviation followed by a zero-controls
    partial_correlation_permutation_test of that deviation against each
    trial's own total spike count, exactly as results/dissociation_
    replication_and_counting_noise.json's per-session real-gate loop runs
    it for every other corpus in this project's census."""
    deviation = rate_free_state_deviation(activity)
    total = activity.sum(axis=1).astype(float)
    finite = np.isfinite(deviation)
    n_finite = int(finite.sum())
    median_total = float(np.median(total))
    n_control_trials = int(activity.shape[0])
    if n_finite < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {
            "status": "excluded",
            "reason": f"fewer than {MIN_TRIALS_WITH_DEFINED_DIRECTION} control trials with a defined "
                      "deviation direction",
            "n_control_trials": n_control_trials, "n_finite_direction_trials": n_finite,
            "median_total_spike_count_per_trial": median_total,
        }
    rng = np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_deviation_orthogonality_gate|{session}"))
    gate = partial_correlation_permutation_test(deviation[finite], total[finite], controls=[],
                                                 n_perm=N_PERM, rng=rng)
    if gate["status"] != "computed":
        return {"status": "excluded", "reason": f"gate not computable: {gate.get('reason')}",
                "n_control_trials": n_control_trials, "n_finite_direction_trials": n_finite,
                "median_total_spike_count_per_trial": median_total}
    gate_passes = gate["p_value"] > GATE_ALPHA
    return {
        "status": "computed",
        "n_control_trials": n_control_trials, "n_finite_direction_trials": n_finite,
        "median_total_spike_count_per_trial": median_total,
        "gate_r": gate["r"], "gate_p_value": gate["p_value"],
        "gate_outcome": ("passes_the_deviation_observable_separates_from_spike_count" if gate_passes
                          else "fails_the_deviation_observable_does_not_separate_from_spike_count"),
    }


def main() -> None:
    t0 = time.time()
    census = json.loads(CENSUS_PATH.read_text())
    delivered_session_medians = {
        r["session"]: r["median_control_total_spike_count_per_trial"]
        for r in census["corpora"]["macaque_pfc_microstimulation"]["per_session"]
    }

    correct_dir = DATA / "correct"
    prefixes = sorted(p.stem for p in correct_dir.glob("*.mat")) if correct_dir.is_dir() else []
    n_seen = len(prefixes)

    per_session, excluded, reproduction_checks = [], [], {}
    for prefix in prefixes:
        loaded = _control_activity_by_unit(prefix)
        if loaded["status"] == "excluded":
            per_session.append({"session": prefix, "status": "excluded", "reason": loaded["reason"]})
            excluded.append({"session": prefix, "reason": loaded["reason"]})
            continue
        row = _session_gate(prefix, loaded["activity_by_unit"])
        row = {"session": prefix, "n_channels": loaded["n_channels"], **row}
        if row["status"] == "excluded":
            excluded.append({"session": prefix, "reason": row["reason"]})
        delivered = delivered_session_medians.get(prefix)
        fresh = row.get("median_total_spike_count_per_trial")
        matches = delivered is not None and fresh is not None and np.isclose(fresh, delivered, rtol=1e-9, atol=1e-9)
        reproduction_checks[prefix] = {"delivered_median_total_spike_count_per_trial": delivered,
                                        "freshly_computed_median_total_spike_count_per_trial": fresh,
                                        "matches": bool(matches)}
        per_session.append(row)

    reproduction_gate = {
        "status": "reproduced_exactly" if all(c["matches"] for c in reproduction_checks.values())
                  else "not_reproduced",
        "rule": "Every session's median control-trial total spike count computed here must exactly "
                "reproduce results/stimulation_axis_same_session_census.json's own already-delivered fresh "
                "measurement of the same quantity for this corpus, before any new deviation-vs-spike-count "
                "gate number from this module is trusted. This is the only reproduction check available: "
                "no prior artifact has ever run the gate itself on this corpus.",
        "checks": reproduction_checks,
    }
    if reproduction_gate["status"] != "reproduced_exactly":
        raise AssertionError(
            "freshly loaded control-trial spike totals do not reproduce the already-delivered census -- "
            "refusing to report a gate built on an unverified trial pool")

    computed = [r for r in per_session if r["status"] == "computed"]
    n_analysed, n_excluded = len(computed), len(excluded)

    pooled = _pool_values([r["gate_r"] for r in computed])
    pooled_p = pooled.get("two_sided_p_value")
    pooled_mean = pooled.get("mean_value")
    mdd_block = pooled.get("minimum_detectable_paired_difference_at_80pct_power")
    pooled_mdd = mdd_block.get("mdd") if isinstance(mdd_block, dict) and mdd_block.get("status") == "computed" \
        else None
    pooled_median_total = float(np.median([r["median_total_spike_count_per_trial"] for r in computed])) \
        if computed else None

    if pooled.get("status") != "tested":
        pooled_outcome = "not_testable"
        pooled_outcome_statement = (
            f"pooled gate not testable ({pooled.get('status')}) on {n_analysed} analysed sessions")
    elif pooled_p > GATE_ALPHA:
        powered = pooled_mdd is not None and pooled_mdd < FAILING_REFERENCE_EFFECT_ABS
        pooled_outcome = ("passes_the_deviation_observable_separates_from_spike_count_powered_null" if powered
                           else "passes_but_the_null_is_below_its_own_detection_floor")
        pooled_outcome_statement = (
            f"pooled gate r={pooled_mean:.4f}, p={pooled_p:.4f} (non-significant); minimum detectable "
            f"paired difference at 80% power = {pooled_mdd}, checked against the standing reference "
            f"{FAILING_REFERENCE_EFFECT_ABS:.4f} r units ({FAILING_REFERENCE_EFFECT_SOURCE}, the smallest-"
            f"magnitude gate effect among the three corpora already known to fail this same gate) -> "
            f"{'a powered null' if powered else 'below its own detection floor, not a powered null'}")
    else:
        pooled_outcome = "fails_the_deviation_observable_does_not_separate_from_spike_count"
        pooled_outcome_statement = f"pooled gate r={pooled_mean:.4f}, p={pooled_p:.4f} (significant)"

    verdict = (
        "The rate-free deviation observable's own orthogonality gate against total spike count FAILS "
        f"pooled across this corpus's control trials (r={pooled_mean:.4f}, p={pooled_p:.4f}): the "
        "observable does not separate from spike count here, so it cannot be trusted to carry a rate-"
        "free signal in this corpus, and this closes the project's only same-session route to asking "
        "whether stimulation pushes along the behavioural axis without joining across corpora."
        if pooled_outcome == "fails_the_deviation_observable_does_not_separate_from_spike_count" else
        "The rate-free deviation observable's own orthogonality gate against total spike count PASSES "
        f"pooled across this corpus's control trials (r={pooled_mean:.4f}, p={pooled_p:.4f}), a powered "
        f"null (minimum detectable difference {pooled_mdd:.4f} r units below the {FAILING_REFERENCE_EFFECT_ABS:.4f} "
        "reference), so the observable is usable here and the project's same-session stimulation-direction "
        "question is askable on this corpus."
        if pooled_outcome == "passes_the_deviation_observable_separates_from_spike_count_powered_null" else
        f"The pooled gate is non-significant (r={pooled_mean:.4f}, p={pooled_p:.4f}) but its own minimum "
        f"detectable difference ({pooled_mdd}) does not clear the {FAILING_REFERENCE_EFFECT_ABS:.4f} "
        "reference, so this is an inconclusive result, not a demonstrated pass -- the observable's "
        "usability on this corpus is undetermined."
    )

    output = {
        "analysis_id": "macaque_pfc_microstimulation_deviation_orthogonality_gate",
        "schema_version": "1.0.0",
        "trigger": (
            "results/stimulation_axis_same_session_census.json found the macaque dorsolateral prefrontal "
            "microstimulation corpus's control trials sit at a pooled median of 930.0 spikes/trial, inside "
            "the range between an already-established failing boundary (353.0 spikes/trial) and an "
            "already-established passing boundary (989.5 spikes/trial) that no corpus in this project had "
            "ever sampled before that census. This module runs the deviation observable's own gate on that "
            "corpus's control trials to find out where it actually lands."
        ),
        "code_commit": git_commit(ROOT),
        "trials_admitted": "control (non-stimulation) trials only; stimulated trials never enter the "
                            "activity arrays this gate is computed on",
        "spike_count_precondition_reference": {
            "failing_at_or_below": SPIKE_COUNT_FAILING_AT_OR_BELOW,
            "passing_at_or_above": SPIKE_COUNT_PASSING_AT_OR_ABOVE,
            "source": SPIKE_COUNT_PRECONDITION_SOURCE,
        },
        "gate_implementation_reused": {
            "deviation_observable": "rate_free_state_deviation, run_rate_free_state_geometry_behavior_link.py",
            "gate_test": "partial_correlation_permutation_test(deviation, total_spike_count, controls=[], "
                         "n_perm=10000), src/statistics.py",
            "cross_session_pooling": "_pool_values, run_watters_state_geometry.py (slope_across_sessions_test "
                                      "+ minimum_detectable_paired_difference)",
            "min_trials_with_defined_direction": MIN_TRIALS_WITH_DEFINED_DIRECTION,
            "reference_effect_for_null_claims": {
                "value_r_units": FAILING_REFERENCE_EFFECT_ABS,
                "source": FAILING_REFERENCE_EFFECT_SOURCE,
                "definition": "smallest-magnitude gate r among the three corpora whose gate is already "
                              "known to fail this same test, run_count_subsampling_ladder.py",
            },
            "note": "no new statistical test is introduced; every function named above is imported and "
                    "called unchanged",
        },
        "reproduction_gate": reproduction_gate,
        "per_session": per_session,
        "pooled": {
            "n_sessions_pooled": n_analysed,
            "gate_r_mean": pooled_mean, "gate_two_sided_p_value": pooled_p,
            "minimum_detectable_paired_difference_at_80pct_power": pooled_mdd,
            "median_total_spike_count_per_trial_across_sessions": pooled_median_total,
            "outcome": pooled_outcome,
            "statement": pooled_outcome_statement,
            "raw_pool": pooled,
        },
        "zero_drop_accounting": {
            "sessions_seen": n_seen, "sessions_analysed": n_analysed, "sessions_excluded": n_excluded,
            "exclusions": excluded,
            "reconciles": bool(n_seen == n_analysed + n_excluded),
        },
        "verdict": verdict,
        "wall_clock_s": round(time.time() - t0, 3),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH} in {output['wall_clock_s']:.1f}s -- pooled outcome: {pooled_outcome}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
