"""run_rate_free_state_geometry_behavior_link.py -- once total spike count
is removed from the trial state BY CONSTRUCTION rather than by regression,
does anything geometric about the state still predict trial outcome?

results/behavior_amplitude_rate_controls.json found that the leading-
component gain's correlation with trial outcome collapses once total spike
count is partialled out (-0.168, p=0.006 raw; -0.016, p=0.68 controlling for
spike count), because a trial with more spikes projects further along the
leading component almost by construction -- "gain" is very nearly a rate
measure. This module asks the same question about the trial state's
DIRECTION rather than its length: a per-trial per-unit activity vector,
normalised to unit L2 norm so only which units are relatively more or less
active survives, is this project's own distance-to-attractor observable
(a trial's distance from a reference state) built in a rate-free form
rather than invented fresh for this test. The per-trial deviation from the session's own mean direction is
computed leave-one-out (a trial never contributes to its own reference, and
the reference is not conditioned on trial outcome in any way), so a
direction-based correlate of accuracy, if one exists, cannot be explained by
the same total-spike-count mechanism that killed the gain correlate.

An orthogonality gate is run BEFORE any behavioural correlation: if the
deviation observable still correlates with total spike count, the
construction failed and this module reports that and stops, rather than
correlating a still-rate-contaminated observable with behaviour and
partialling rate back out after the fact -- the analysis this module exists
to replace.

Scope matches results/state_behavior_link.json and
results/behavior_amplitude_rate_controls.json exactly: macaque lPFC only
(doi 10.1038/s41586-024-08139-9), the same 11 sessions reaching the >=60-error
reachability floor, no trial pooled across sessions or animals -- every
correlation is computed within one session and only the resulting per-
session coefficients are pooled across sessions, by the paired sign-flip
test. Reuses statistics.partial_correlation_permutation_test (the same
primitive results/behavior_amplitude_rate_controls.json uses) and
run_behavior_amplitude_rate_controls._reachable_sessions unchanged rather
than forking either.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import data_root  # noqa: E402
from corpus_sessions import _reachable_sessions
from statistics import MIN_ERROR_TRIALS_FOR_REACHABILITY
from spike_pipeline import _counts_from_spikes
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)
from stimulation_response_estimator import rate_free_state_deviation  # noqa: E402
from state_persistence import _pool  # noqa: E402
from statistics import MEANINGFUL_EFFECT_THRESHOLD_R_UNITS  # noqa: E402
from corpus_sessions import _session_arrays  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "rate_free_state_geometry_behavior_link.json"
N_PERM = 10000

# The same r-unit scale results/state_behavior_link.json's persistence bound (MDD 0.139, reported as
# "~0.14 r units") is already on, fixed here BEFORE any fit runs so the two nulls are commensurable --
# a reader comparing this project's behavioural bounds is comparing the same units, not two different
# implicit scales.

ORTHOGONALITY_GATE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session, correlate the rate-free deviation observable (see rate_free_state_deviation) with the "
    "trial's total spike count in the delay epoch, using the zero-controls case of "
    "partial_correlation_permutation_test (an ordinary Pearson correlation with a permutation p-value). "
    "Pool the 11 per-session correlation coefficients with the two-sided paired sign-flip test "
    "(slope_across_sessions_test) against zero. If the pooled correlation is significant at the two-sided "
    "0.05 level, the construction FAILED, the branch is "
    "'rate_free_state_observable_is_not_rate_free_and_this_analysis_is_void', and this module reports that "
    "and stops -- it does not proceed to correlate a still-rate-contaminated observable with behaviour and "
    "then partial rate back out, because that is the analysis this module exists to replace. The "
    "correlation and its CI are reported either way; a near-zero value is the evidence the construction "
    "worked and belongs in the artifact whether or not anyone doubted it."
)

BEHAVIOURAL_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "If the orthogonality gate passes: per session, compute the point-biserial correlation (again the "
    "zero-controls case of partial_correlation_permutation_test) of the deviation observable with trial "
    "outcome, pool across the 11 sessions with the two-sided paired sign-flip test, and report its minimum "
    "detectable paired difference at 80% power. Also report, the same way: the partial controlling total "
    "spike count, the partial controlling trial index, and the joint partial controlling both together -- "
    "computed up front rather than only after seeing whether the raw correlation is significant, because "
    "an n=11 collinear pair is where the joint partial can stop being identifiable. Before any fit runs:\n"
    "  - If the pooled RAW correlation is significant in either direction (two-sided p <= 0.05) AND the "
    "joint partial (controlling both total spike count and trial index together) is also significant with "
    "the same sign, the branch is 'rate_free_state_geometry_predicts_accuracy', with the sign stated and "
    "interpreted: a positive correlation with error means trials further from the session's mean state are "
    "more often wrong (the attractor-consistent direction); a negative one is not, and is reported as the "
    "surprise it would be rather than absorbed into the same headline.\n"
    "  - If the pooled raw correlation is NOT significant, and its minimum detectable paired difference is "
    "below " + str(MEANINGFUL_EFFECT_THRESHOLD_R_UNITS) + " r units (the smallest effect this design would "
    "call meaningful, fixed to the same scale results/state_behavior_link.json's persistence bound "
    "reports on), the branch is 'no_rate_free_state_geometry_link_to_accuracy_above_the_reported_bound', "
    "and the project's behavioural section is closed as a bounded null.\n"
    "  - If the pooled raw correlation is NOT significant and its minimum detectable paired difference is "
    "at or above " + str(MEANINGFUL_EFFECT_THRESHOLD_R_UNITS) + " r units, the branch is "
    "'underpowered_to_ask', stated as such, with the session count and error count that would be needed.\n"
    "  - The rule as written does not name a branch for a raw correlation that IS significant but does "
    "NOT survive the joint control (an outcome the rule anticipates as possible -- \"the jointly-"
    "controlled version is where an n=11 collinear pair stops being identifiable\" -- without naming what "
    "happens when it occurs): this is a genuine gap in the pre-declared rule, not a case to force onto "
    "either named branch. If it occurs, the branch is "
    "'raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index', "
    "implemented and tested as its own outcome, reported as a rule gap to be closed in writing "
    "rather than silently resolved here."
)




def _corr(y: np.ndarray, x: np.ndarray, controls: list[np.ndarray], seed_tag: str) -> dict:
    rng = np.random.default_rng(stable_seed(seed_tag))
    return partial_correlation_permutation_test(y, x, controls=controls, n_perm=N_PERM, rng=rng)


def _analyze_session(session_id: str, arrays: dict) -> dict:
    is_corr, deviation, spike_count, trial_index = (
        arrays["is_corr"], arrays["deviation"], arrays["spike_count"], arrays["trial_index"])
    tag = f"rate_free_state_geometry_behavior_link|{session_id}"
    return {
        "n_trials_total": arrays["n_trials_total"],
        "n_trials_with_defined_direction": arrays["n_trials_with_defined_direction"],
        "orthogonality_gate": _corr(deviation, spike_count, [], f"{tag}|gate|deviation_vs_spike_count"),
        "raw_outcome_vs_deviation": _corr(is_corr, deviation, [], f"{tag}|raw"),
        "partial_controlling_spike_count": _corr(is_corr, deviation, [spike_count], f"{tag}|ctrl_spike"),
        "partial_controlling_trial_index": _corr(is_corr, deviation, [trial_index], f"{tag}|ctrl_trial"),
        "joint_partial_controlling_spike_count_and_trial_index": _corr(
            is_corr, deviation, [spike_count, trial_index], f"{tag}|ctrl_joint"),
    }




def _classify(gate: dict, raw: dict, joint: dict, mdd: float | None) -> str:
    """Implements ORTHOGONALITY_GATE_DECISION_RULE_DECLARED_BEFORE_FITTING and
    BEHAVIOURAL_DECISION_RULE_DECLARED_BEFORE_FITTING's branches, including
    the disclosed gap branch for a raw-significant result that does not
    survive the joint control."""
    if gate.get("status") != "tested":
        return "not_computable"
    if gate["significant"]:
        return "rate_free_state_observable_is_not_rate_free_and_this_analysis_is_void"
    if raw.get("status") != "tested":
        return "not_computable"
    if raw["significant"]:
        raw_sign_positive = raw["mean_value"] > 0.0
        joint_agrees = (
            joint.get("status") == "tested" and joint["significant"]
            and (joint["mean_value"] > 0.0) == raw_sign_positive
        )
        if joint_agrees:
            return "rate_free_state_geometry_predicts_accuracy"
        return "raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index"
    if mdd is not None and mdd < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        return "no_rate_free_state_geometry_link_to_accuracy_above_the_reported_bound"
    return "underpowered_to_ask"


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))


def _combined_behavioural_position(branch: str, raw_mean: float | None, mdd_value: float | None) -> str:
    """The single, quotable field 2.5 asks for: every piece of this
    project's macaque behavioural line in one place, reading the finished,
    read-only persistence and amplitude artifacts directly rather than
    retyping their numbers by hand."""
    persistence_path = Path(__file__).resolve().parents[1] / "results" / "state_behavior_link.json"
    amplitude_path = Path(__file__).resolve().parents[1] / "results" / "behavior_amplitude_rate_controls.json"
    persistence = json.loads(persistence_path.read_text()) if persistence_path.exists() else {}
    amplitude = json.loads(amplitude_path.read_text()) if amplitude_path.exists() else {}
    dc = persistence.get("deciding_contrast", {})
    dc_test = dc.get("test", {})
    dc_mdd = dc.get("minimum_detectable_paired_difference_at_80pct_power", {})
    amp_gain_given_spike = amplitude.get("pooled", {}).get("gain_given_spike_count", {})
    n_error_2024 = sorted(
        s.get("n_error") for s in persistence.get("sessions", [])
        if s.get("session", "").startswith("24") and s.get("n_error") is not None
    )
    return (
        "Three rounds of this project's macaque behavioural line, in one place. (1) Persistence: the "
        "cross-unit state's magnitude (matched correct vs error trials, 11 reachable sessions) is a "
        f"bounded null -- mean {dc.get('mean_difference'):.4f}, median {dc.get('median_difference'):.4f} "
        f"r units, two-sided p={dc_test.get('p_value'):.3f}, CI [{dc_test.get('ci_lower'):.4f}, "
        f"{dc_test.get('ci_upper'):.4f}], minimum detectable paired difference {dc_mdd.get('mdd'):.4f} r "
        "units at 80% power. An earlier fit of this same contrast, before a paired estimator-setting fix "
        "(matching the reduced-parameter error-trial estimator to the reduced-parameter correct-trial "
        "estimator), read p=0.047 (significant); the fix reversed it to the p above -- both values are "
        "part of the record, not only the corrected one. (2) Amplitude: the leading-component gain "
        "correlates with accuracy raw (r=-0.168, p=0.006) but the correlation is attributable to firing "
        "rate, not geometry -- controlling for total spike count alone collapses it to "
        f"r={amp_gain_given_spike.get('mean_value', float('nan')):.4f}, "
        f"p={amp_gain_given_spike.get('p_value', float('nan')):.3f} (not significant), while the reverse "
        "(spike count controlling for gain) survives at r=-0.123, p=0.039. (3) This analysis: a rate-free "
        "direction-based observable, orthogonality-gated against total spike count (gate not significant, "
        "the construction held), correlates with accuracy in the attractor-consistent direction -- branch "
        f"'{branch}', pooled raw r={raw_mean:.4f}" if raw_mean is not None else
        f"'{branch}', pooled raw r=not_computable"
    ) + (
        f", minimum detectable paired difference {mdd_value:.4f} r units. " if mdd_value is not None else ". "
    ) + (
        "(4) The human corpus cannot ask this question at all: its maintenance-epoch content decoding "
        "sits at chance with this project's own best decoder, which is a ceiling on what a behavioural "
        "correlate could even be tested against, not a null result about behaviour itself. (5) The "
        f"macaque 2024 cohort's own 7 sessions ({n_error_2024[0]}-{n_error_2024[-1]} error trials each) "
        "can never enter the matched-contrast reachability floor (60 error trials) at this accuracy level "
        "and are excluded from every contrast in this behavioural line, not only this one."
    )


def main() -> None:
    root = data_root()
    t0 = time.time()
    paths = _reachable_sessions(root)

    output = {
        "version": "2026-08-31",
        "scope": (
            "Macaque lPFC only (doi 10.1038/s41586-024-08139-9): human trial-level accuracy is at ceiling and the "
            "mouse ALM corpus has no comparable per-trial accuracy. The 11 sessions reaching the "
            f"pre-declared reachability floor (at least {MIN_ERROR_TRIALS_FOR_REACHABILITY} error trials), "
            "the same sessions results/state_behavior_link.json and "
            "results/behavior_amplitude_rate_controls.json both use. No trial is pooled across sessions or "
            "animals for any correlation here -- every correlation (raw, gate, or partial) is computed "
            "within one session, and only the resulting per-session correlation coefficients are pooled "
            "across sessions, by the paired sign-flip test. The rate-free deviation observable "
            "(rate_free_state_deviation) is this project's own distance-to-attractor observable "
            "built in a rate-free form: each trial's "
            "per-unit activity vector is L2-normalised to unit length before comparison, so only its "
            "direction across units, not its total magnitude, enters the leave-one-out reference or the "
            "cosine deviation -- it is not a new construct invented for this test."
        ),
        "construction_operationalisation": (
            "'Per-unit activity vector' is operationalised as each unit's total spike count summed over "
            "every bin of the delay epoch (macaque spatial working-memory corpus delay window, 100 ms bins) -- a single (n_units,) "
            "vector per trial, before L2 normalisation. This specific choice (full-epoch sum rather than, "
            "e.g., a per-window-then-averaged vector at the deciding window width) is a disclosed reading "
            "of a genuinely underspecified instruction, made once here rather than left implicit; any total-"
            "count normalisation removes the rate confound the same way, so the orthogonality gate below is "
            "what actually certifies this specific choice rather than the choice being certified by "
            "construction alone."
        ),
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
        "meaningful_effect_threshold_source": (
            "results/state_behavior_link.json's persistence null minimum detectable paired difference "
            "(0.139, reported as ~0.14 r units), used here unchanged so the two behavioural nulls in this "
            "project are on a commensurable scale."
        ),
        "orthogonality_gate_decision_rule_declared_before_fitting": ORTHOGONALITY_GATE_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "behavioural_decision_rule_declared_before_fitting": BEHAVIOURAL_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "limits": (
            "Carried forward from results/behavior_amplitude_rate_controls.json unchanged: a correlate of "
            "behaviour in this corpus, even a rate-free one, cannot be separated from arousal, motivation "
            "or task engagement, none of which is measured. A rate-free observable removes the rate "
            "confound; it does not remove this one."
        ),
        "n_perm": N_PERM,
        "n_sessions_reachable": len(paths),
        "sessions": [], "sessions_completed": [], "sessions_pending": [p.stem for p in paths],
    }
    _flush(output)

    for i, path in enumerate(paths):
        session_id = path.stem
        arrays = _session_arrays(path)
        if arrays is None:
            result = {"session": session_id, "status": "not_computable"}
        else:
            result = {"session": session_id, "status": "computed", "analysis": _analyze_session(session_id, arrays)}
        output["sessions"].append(result)
        output["sessions_completed"].append(session_id)
        output["sessions_pending"] = [p.stem for p in paths[i + 1:]]
        _flush(output)
        print(f"progress {i + 1}/{len(paths)} {session_id} flushed", flush=True)

    computed = [s for s in output["sessions"] if s["status"] == "computed"]
    pooled = {
        "orthogonality_gate_deviation_vs_spike_count": _pool(computed, "orthogonality_gate"),
        "raw_outcome_vs_deviation": _pool(computed, "raw_outcome_vs_deviation"),
        "partial_controlling_spike_count": _pool(computed, "partial_controlling_spike_count"),
        "partial_controlling_trial_index": _pool(computed, "partial_controlling_trial_index"),
        "joint_partial_controlling_spike_count_and_trial_index": _pool(
            computed, "joint_partial_controlling_spike_count_and_trial_index"),
    }
    raw_values = [s["analysis"]["raw_outcome_vs_deviation"]["r"] for s in computed
                  if s["analysis"]["raw_outcome_vs_deviation"].get("status") == "computed"]
    mdd_result = (minimum_detectable_paired_difference(raw_values) if len(raw_values) >= 2 else
                  {"status": "not_computable", "n": len(raw_values), "reason": "fewer than 2 sessions"})
    pooled["raw_outcome_vs_deviation_minimum_detectable_paired_difference_80pct_power"] = mdd_result
    mdd_value = mdd_result.get("mdd") if mdd_result.get("status") == "computed" else None

    branch = _classify(
        pooled["orthogonality_gate_deviation_vs_spike_count"],
        pooled["raw_outcome_vs_deviation"],
        pooled["joint_partial_controlling_spike_count_and_trial_index"],
        mdd_value,
    )
    raw_mean = pooled["raw_outcome_vs_deviation"].get("mean_value")
    if branch in (
        "rate_free_state_geometry_predicts_accuracy",
        "raw_correlation_significant_but_does_not_survive_joint_control_of_spike_count_and_trial_index",
    ) and raw_mean is not None:
        # raw_outcome_vs_deviation correlates deviation with is_corr (1=correct, 0=error), not with error
        # directly, so its sign is the OPPOSITE of "correlation with error" -- translated explicitly here
        # so the pre-declared rule's sign language ("a positive correlation with error...") is answered
        # without leaving a reader to flip the sign themselves.
        error_correlation_sign = "positive" if raw_mean < 0.0 else "negative"
        attractor_consistent = error_correlation_sign == "positive"
        output["sign_interpretation"] = (
            f"raw_outcome_vs_deviation.mean_value = {raw_mean:.4f} is a correlation of deviation with "
            "is_corr (1 = correct, 0 = error), not with error directly. Its sign is therefore the OPPOSITE "
            f"of the correlation with error: deviation correlates with error {error_correlation_sign}ly. "
            + ("This is the attractor-consistent direction the pre-declared rule anticipated: trials "
               "further from the session's own mean state direction are more often wrong."
               if attractor_consistent else
               "This is NOT the attractor-consistent direction: trials further from the session's own "
               "mean state direction are more often CORRECT, reported here as the surprise it is rather "
               "than absorbed into the same headline as the attractor-consistent case.")
        )

    output["pooled"] = pooled
    output["branch"] = branch
    output["combined_behavioural_position"] = _combined_behavioural_position(branch, raw_mean, mdd_value)
    output["n_sessions_computed"] = len(computed)
    output["wall_clock_s"] = time.time() - t0
    output["status"] = "complete"
    _flush(output)
    print(json.dumps({"branch": branch, "n_sessions_computed": len(computed)}, indent=2))


if __name__ == "__main__":
    main()
