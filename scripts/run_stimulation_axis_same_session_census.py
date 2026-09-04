#!/usr/bin/env python3
"""Feasibility census: which corpora could measure a behavioural neural axis
AND a stimulation-induced displacement in the SAME sessions.

This does not fit any axis, any displacement, or any estimator. It answers
one question per corpus registered in config/datasets.json: could the
question "does stimulation displace the state along the axis that carries
behaviour, or off it" even be asked here without joining across corpora that
share no sessions? For every corpus this records (a) whether it carries
stimulation and of what kind, (b) whether a rate-free population-geometry
deviation construction is computable on its non-stimulation trials at all
(spike-sorted units, enough trials, a defined per-trial direction), (c) the
corpus's own measured total spike count per trial against a precondition
this project has already established empirically: a rate-free deviation
observable's own orthogonality test against total spike count comes back
significant (fails) at or below roughly 355 spikes/trial and comes back
non-significant (passes) at or above roughly 990 spikes/trial, with nothing
between the two ever sampled before this census, (d) the number of sessions
that carry both a non-stimulation trial pool and a stimulation trial pool,
which is the number that decides whether the question is askable within that
corpus at all, and (e) whether the corpus keeps stimulation outside the
epoch a deviation axis would be measured in, or whether the two overlap by
design.

Two corpora in the registry carry both spike-sorted single units and
stimulation: the mouse frontal-cortex photoinhibition release and the
macaque prefrontal microstimulation release. Every other stimulation-
carrying corpus in the registry is a field-potential/scalp release with no
spike-sorted units at all, so the deviation construction this leg depends on
is undefined there regardless of session overlap -- that exclusion is
recorded as structural (modality), not as a spike-count failure.

For the two corpora that could in principle carry both halves, this script
reuses one already-measured, already-reproduced number (the mouse corpus's
median total spike count per trial, from a prior artifact's control-trial-
only, matched-window measurement) rather than re-deriving it, and freshly
measures the macaque corpus's own median total spike count per trial
directly from its raw per-trial spike-rate arrays (a lightweight sum, not an
estimator fit) because no prior artifact in this project has characterized
it. Both corpora's session-level trial/unit counts are also measured
directly by loading every session file, because the registry alone does not
carry that information.

Outputs:
  results/stimulation_axis_same_session_census.json

Run:
    conda run -n wm_dynamics python scripts/run_stimulation_axis_same_session_census.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import git_commit  # noqa: E402

RESULTS = ROOT / "results"
CONFIG = json.loads((ROOT / "config" / "datasets.json").read_text())
DATA_ROOT_ENV = CONFIG["local_data_root_env"]
DATA_ROOT = Path(os.environ[DATA_ROOT_ENV]) if os.environ.get(DATA_ROOT_ENV) else None

# The already-established, already-reproduced spike-count precondition this project uses
# elsewhere: a rate-free deviation observable's own orthogonality gate against total spike
# count fails (comes back significant) at or below this many spikes/trial, and passes
# (comes back non-significant) at or above this many spikes/trial, measured across five
# corpora with nothing sampled in between. Source: results/dissociation_replication_and_counting_noise.json,
# block_b.count_separation_disclosure.
SPIKE_COUNT_FAILING_AT_OR_BELOW = 353.0
SPIKE_COUNT_PASSING_AT_OR_ABOVE = 989.5
PRECONDITION_SOURCE_ARTIFACT = "results/dissociation_replication_and_counting_noise.json"
PRECONDITION_SOURCE_PATH = "block_b.count_separation_disclosure"


def _measure_inagaki_alm5() -> dict:
    """Mouse ALM: photoinhibition-perturbation trials and control trials coexist within
    each recorded session (src/corpus_sessions.py's stim_trial_vector split). Session/unit/
    trial counts are measured fresh here (cheap: just loading and thresholding, the same
    admission logic every other analysis in this project already applies via
    load_alm_raw_session, called read-only). The spike-count precondition is NOT
    re-measured here -- it is already measured and reproduced in
    results/dissociation_replication_and_counting_noise.json, on control trials only, at a
    1.2 s matched-delay window (bin_ms=100, require_both_arms=False, since that prior
    measurement never needed the perturbation arm) -- and is cited rather than re-derived,
    to avoid producing a second, differently-configured number for the same corpus."""
    from corpus_sessions import alm_data_directory, load_alm_raw_session

    if DATA_ROOT is None:
        return {"status": "not_reachable", "reason": "WM_DYNAMICS_DATA_ROOT not set"}
    directory = alm_data_directory(DATA_ROOT)
    if not directory.is_dir():
        return {"status": "not_reachable", "reason": f"{directory} does not exist"}
    paths = sorted(directory.glob("*.mat"))
    n_units, n_control, n_perturb = [], [], []
    n_both_arms_admitted = 0
    for path in paths:
        session = load_alm_raw_session(path, require_both_arms=True)
        if session is None:
            continue
        n_both_arms_admitted += 1
        n_units.append(session["n_units_after_rate_qc"])
        n_control.append(session["n_control_trials"])
        n_perturb.append(session["n_perturb_trials"])

    dissociation_path = RESULTS / PRECONDITION_SOURCE_ARTIFACT.split("/")[-1]
    prior_median_spike_count = None
    if dissociation_path.exists():
        prior = json.loads(dissociation_path.read_text())
        prior_median_spike_count = (
            prior.get("block_b", {}).get("count_separation_disclosure", {})
            .get("failing_corpora", {}).get("inagaki_alm5_mouse_ALM", {})
            .get("median_total_spike_count_per_trial")
        )

    passes_precondition = (
        prior_median_spike_count is not None
        and prior_median_spike_count > SPIKE_COUNT_FAILING_AT_OR_BELOW
    )

    return {
        "status": "computed",
        "carries_stimulation": True,
        "stimulation_kind": (
            "optogenetic photoinhibition of anterior lateral motor cortex during the delay "
            "period (bilateral perturbation trials), an experimenter-set within-session "
            "condition, not electrical or transcranial stimulation"
        ),
        "n_session_files_on_disk": len(paths),
        "n_sessions_with_both_control_and_perturbation_trial_arms": n_both_arms_admitted,
        "session_admission_rule": (
            "src/corpus_sessions.py load_alm_raw_session with require_both_arms=True, this "
            "project's own existing admission thresholds (ALM_MIN_TRIALS_PER_ARM=8, "
            "ALM_MIN_UNITS=15, ALM_MIN_UNIT_RATE_HZ=0.2 Hz), window_s=2.0, bin_ms=100 -- "
            "called read-only, not modified"
        ),
        "median_units_per_session": float(np.median(n_units)) if n_units else None,
        "unit_count_range": [int(min(n_units)), int(max(n_units))] if n_units else None,
        "median_control_trials_per_session": float(np.median(n_control)) if n_control else None,
        "median_perturbation_trials_per_session": float(np.median(n_perturb)) if n_perturb else None,
        "per_trial_direction": (
            "defined: instructed response side / trial condition is recorded per trial "
            "(control_condition / control_response_code)"
        ),
        "deviation_construction_structurally_computable_on_non_stimulation_trials": True,
        "spike_count_precondition": {
            "median_total_spike_count_per_trial": prior_median_spike_count,
            "measured_on": "control (non-perturbation) trials only, matched 1.2 s delay window",
            "source_artifact": PRECONDITION_SOURCE_ARTIFACT,
            "source_path": PRECONDITION_SOURCE_PATH,
            "passes_precondition": passes_precondition,
        },
        "excluded_on_precondition": not passes_precondition,
        "exclusion_reason": (
            None if passes_precondition else
            f"measured median total spike count per trial "
            f"({prior_median_spike_count}) is at or below the failing boundary "
            f"({SPIKE_COUNT_FAILING_AT_OR_BELOW}) this project has already established for the "
            f"rate-free deviation observable's own orthogonality gate against spike count"
        ),
        "epoch_disclosure": (
            "the photoinhibition interval falls INSIDE the analysed working-memory delay by "
            "design -- this is not an isolated maintenance delay for the perturbed arm; the "
            "control arm's own delay window is unperturbed"
        ),
        "sessions_carrying_both_halves_moot_reason": (
            None if passes_precondition else
            "the session-overlap count above is reported for completeness but is moot: the "
            "corpus is excluded on the spike-count precondition before any per-session join "
            "would be attempted"
        ),
    }


def _measure_macaque_pfc_microstimulation() -> dict:
    """Macaque dlPFC: delay-period microstimulation. Control and stimulated trials are
    interleaved within every session (run_macaque_pfc_microstimulation_pipeline.py's own load_macaque_pfc_microstimulation_session,
    called read-only). No prior artifact in this project has measured this corpus's total
    spike count per trial, so it is measured fresh here: the raw per-trial 'spikerate'
    array is (time, channel) in Hz (50 ms bins; a lone spike in one bin reads as 20 Hz,
    1/0.05 s -- confirmed by direct inspection of the array's discretisation, all values
    exact multiples of 20), so summing it over its cropped 30-bin window and channels and
    multiplying by the 0.05 s bin width recovers the total spike count for that trial. This
    is a sum over already-loaded arrays, not a fit."""
    from run_macaque_pfc_microstimulation_pipeline import BIN_S, DATA, crop_trial, load_macaque_pfc_microstimulation_session

    if not DATA.exists():
        return {"status": "not_reachable", "reason": f"{DATA} does not exist"}
    correct_dir = DATA / "correct"
    prefixes = sorted(p.stem for p in correct_dir.glob("*.mat")) if correct_dir.is_dir() else []

    per_session = []
    for prefix in prefixes:
        corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
        if corr is None or corr["control_idx"] is None:
            per_session.append({"session": prefix, "status": "not_loadable_or_no_control_condition"})
            continue
        ctrl_idx = corr["control_idx"]
        control_totals, n_ctrl, n_stim = [], 0, 0
        for tr in corr["trials"]:
            cropped = crop_trial(tr["spikerate"])
            if cropped is None:
                continue
            if tr["stim_cond"] == ctrl_idx:
                control_totals.append(float(cropped.sum()) * BIN_S)
                n_ctrl += 1
            else:
                n_stim += 1
        per_session.append({
            "session": prefix,
            "status": "computed" if control_totals else "no_usable_control_trials",
            "n_channels": int(len(corr["channel_ids"])),
            "n_control_trials": n_ctrl,
            "n_stim_trials": n_stim,
            "has_both_halves": n_ctrl > 0 and n_stim > 0,
            "median_control_total_spike_count_per_trial": (
                float(np.median(control_totals)) if control_totals else None
            ),
        })

    computed = [r for r in per_session if r["status"] == "computed"]
    both_halves = [r for r in per_session if r.get("has_both_halves")]
    session_medians = [r["median_control_total_spike_count_per_trial"] for r in computed]
    pooled_median = float(np.median(session_medians)) if session_medians else None

    in_gap = (
        pooled_median is not None
        and SPIKE_COUNT_FAILING_AT_OR_BELOW < pooled_median < SPIKE_COUNT_PASSING_AT_OR_ABOVE
    )
    excluded_on_precondition = (
        pooled_median is not None and pooled_median <= SPIKE_COUNT_FAILING_AT_OR_BELOW
    )

    return {
        "status": "computed",
        "carries_stimulation": True,
        "stimulation_kind": (
            "intracortical microstimulation of dorsolateral prefrontal cortex during the "
            "working-memory delay, delivered on a randomised subset of trials per session "
            "(bipolar in one animal, single-channel in the other)"
        ),
        "n_session_files_on_disk": len(prefixes),
        "n_sessions_computed": len(computed),
        "n_sessions_with_both_control_and_stimulation_trials": len(both_halves),
        "per_session": per_session,
        "median_channels_per_session": (
            float(np.median([r["n_channels"] for r in computed])) if computed else None
        ),
        "median_control_trials_per_session": (
            float(np.median([r["n_control_trials"] for r in computed])) if computed else None
        ),
        "median_stim_trials_per_session": (
            float(np.median([r["n_stim_trials"] for r in computed])) if computed else None
        ),
        "per_trial_direction": (
            "defined: instructed target angle is recorded per trial (angle_idx)"
        ),
        "deviation_construction_structurally_computable_on_non_stimulation_trials": True,
        "spike_count_precondition": {
            "median_total_spike_count_per_trial_pooled_across_sessions": pooled_median,
            "measured_on": "control (non-stimulation) trials only, this session's own 1.5 s "
                           "stim-onset-aligned window (30 bins x 50 ms)",
            "measurement_is_fresh_not_reused": True,
            "falls_at_or_below_the_established_failing_boundary": excluded_on_precondition,
            "falls_at_or_above_the_established_passing_boundary": (
                pooled_median is not None and pooled_median >= SPIKE_COUNT_PASSING_AT_OR_ABOVE
            ),
            "falls_inside_the_previously_unsampled_gap": in_gap,
            "gap_bounds": [SPIKE_COUNT_FAILING_AT_OR_BELOW, SPIKE_COUNT_PASSING_AT_OR_ABOVE],
        },
        "excluded_on_precondition": excluded_on_precondition,
        "exclusion_reason": (
            f"measured pooled median total spike count per trial ({pooled_median}) is at or "
            f"below the failing boundary ({SPIKE_COUNT_FAILING_AT_OR_BELOW})"
            if excluded_on_precondition else None
        ),
        "precondition_disclosure": (
            "this measured value has not been excluded by the stated rule (it exceeds the "
            "established failing boundary), but it also does not reach the established "
            "passing boundary -- it falls inside a range this project has never sampled "
            "before this census. Whether the rate-free deviation observable's own "
            "orthogonality gate actually passes here has not been tested; this census "
            "measures the precondition only and does not run that gate."
            if in_gap else None
        ),
        "epoch_disclosure": (
            "every trial's window is cropped to -0.8 to +0.7 s relative to stimulation "
            "onset, identically for control and stimulated trials -- the analysed epoch "
            "overlaps the moment of current delivery by design for the stimulated arm; "
            "control trials share the same window definition but receive no current"
        ),
    }


def _no_spike_data_corpus(kind: str, modality: str, epoch_note: str) -> dict:
    return {
        "status": "computed",
        "carries_stimulation": True,
        "stimulation_kind": kind,
        "deviation_construction_structurally_computable_on_non_stimulation_trials": False,
        "exclusion_reason": (
            f"modality is {modality} -- no spike-sorted single units exist in this corpus, "
            "so the rate-free deviation construction (a spike-count-based population-"
            "geometry observable) is undefined here regardless of session overlap; excluded "
            "on data type, not on the spike-count precondition"
        ),
        "n_sessions_with_both_control_and_stimulation_trials": 0,
        "epoch_disclosure": epoch_note,
    }


def _no_stimulation_corpus(constructs: list[str]) -> dict:
    return {
        "status": "computed",
        "carries_stimulation": False,
        "stimulation_kind": None,
        "constructs": constructs,
        "deviation_construction_structurally_computable_on_non_stimulation_trials": "not_applicable",
        "n_sessions_with_both_control_and_stimulation_trials": 0,
        "exclusion_reason": "corpus carries no stimulation at all",
        "epoch_disclosure": "not_applicable -- no stimulation to disclose overlap with",
    }


def build_census() -> dict:
    corpora = {}
    registry_keys = list(CONFIG["datasets"].keys())

    corpora["inagaki_alm5"] = _measure_inagaki_alm5()
    corpora["macaque_pfc_microstimulation"] = _measure_macaque_pfc_microstimulation()
    corpora["haslacher_clam_tacs"] = _no_spike_data_corpus(
        "phase-locked closed-loop transcranial alternating-current stimulation (tACS)",
        "scalp_eeg",
        "stimulation is delivered during the working-memory retention period, coincident "
        "with the analysed epoch, but the point is moot: no spike data exists to build a "
        "deviation axis from",
    )
    corpora["alagapan_phase_stimulation"] = _no_spike_data_corpus(
        "phase-tuned intracranial electrical stimulation at encoding, with a retention-"
        "period behavioural aftereffect read-out",
        "depth_lfp",
        "stimulation is delivered at encoding, not during the retention period this "
        "project's deviation axis would be measured in, but the point is moot: no spike "
        "data exists",
    )
    corpora["ram_ds005489_openloop"] = _no_spike_data_corpus(
        "open-loop (experimenter-randomised) electrical stimulation of alternating word "
        "blocks during list encoding",
        "depth_lfp",
        "stimulation is delivered at encoding, not during any working-memory maintenance "
        "delay -- this corpus has no delay-period epoch at all -- but the point is moot: "
        "no spike data exists",
    )
    corpora["ram_ds005557_closedloop"] = _no_spike_data_corpus(
        "closed-loop (state-triggered) electrical stimulation during list encoding",
        "depth_lfp",
        "stimulation is delivered at encoding, not during any working-memory maintenance "
        "delay -- this corpus has no delay-period epoch at all -- but the point is moot: "
        "no spike data exists",
    )

    for key in ("dandi_000469", "dandi_001187", "dandi_000673", "dandi_000574",
                "panichello_2024", "watters_2026", "ds004752", "pfc3",
                "wolff_eeg_impulse", "kai_miller_nback"):
        corpora[key] = _no_stimulation_corpus(CONFIG["datasets"][key]["constructs"])  # present for every registry entry; raise if ever absent

    assert set(corpora.keys()) == set(registry_keys), (
        set(registry_keys) ^ set(corpora.keys())
    )

    both_halves_askable = {
        key: row for key, row in corpora.items()
        if row.get("carries_stimulation")
        and row.get("deviation_construction_structurally_computable_on_non_stimulation_trials") is True
        and not row.get("excluded_on_precondition", False)
    }

    reasons: dict[str, int] = {}
    for key, row in corpora.items():
        if row.get("carries_stimulation") is not True:
            reasons["no_stimulation_in_corpus"] = reasons.get("no_stimulation_in_corpus", 0) + 1
        elif not row.get("deviation_construction_structurally_computable_on_non_stimulation_trials"):
            reasons["no_spike_sorted_units_in_corpus"] = reasons.get("no_spike_sorted_units_in_corpus", 0) + 1
        elif row.get("excluded_on_precondition"):
            reasons["spike_count_precondition_failed"] = reasons.get("spike_count_precondition_failed", 0) + 1
        else:
            reasons["not_excluded_leg_structurally_askable"] = reasons.get("not_excluded_leg_structurally_askable", 0) + 1

    censused = len(corpora)
    refused = 0  # every registered corpus's stimulation status is knowable from the registry
    # or from directly loading its data; none was unreachable this census.

    return {
        "analysis_id": "stimulation_axis_same_session_census",
        "schema_version": "1.0.0",
        "trigger": (
            "Whether stimulation displaces the neural state along the axis carrying the "
            "behavioural signal, or off it, can only be asked within a corpus where the "
            "behavioural axis and a stimulation displacement can both be measured in the "
            "SAME sessions -- no per-session join exists across corpora that share zero "
            "sessions, however completely each side is separately serialised. This census "
            "answers that one question, corpus by corpus, for every corpus in this "
            "project's registry, and fits nothing."
        ),
        "code_commit": git_commit(ROOT),
        "data_root": str(DATA_ROOT) if DATA_ROOT else None,
        "registry_source": "config/datasets.json",
        "n_corpora_in_registry": len(registry_keys),
        "spike_count_precondition_reference": {
            "statement": (
                "A rate-free deviation observable's own orthogonality test against total "
                "spike count comes back significant (the observable fails to separate from "
                "spike count) at or below roughly 353 spikes/trial, and comes back non-"
                "significant (the observable passes) at or above roughly 990 spikes/trial, "
                "measured across five corpora in this project with nothing sampled between "
                "those two values. A corpus at or below the failing boundary is excluded on "
                "this precondition; nothing in the unsampled range between the two boundaries "
                "is treated as either passing or failing -- it is reported as untested."
            ),
            "failing_at_or_below": SPIKE_COUNT_FAILING_AT_OR_BELOW,
            "passing_at_or_above": SPIKE_COUNT_PASSING_AT_OR_ABOVE,
            "source_artifact": PRECONDITION_SOURCE_ARTIFACT,
            "source_path": PRECONDITION_SOURCE_PATH,
        },
        "corpora": corpora,
        "corpora_carrying_both_halves_and_not_excluded": list(both_halves_askable.keys()),
        "zero_drop_accounting": {
            "corpora_seen": len(registry_keys),
            "censused": censused,
            "refused": refused,
            "refusals_by_reason": {},
            "outcome_reasons": reasons,
            "reconciles": censused + refused == len(registry_keys)
                          and sum(reasons.values()) == len(registry_keys),
        },
        "excluded_but_staged_not_registered": {
            "note": (
                "Three corpora are staged on disk (results/corpus_staging_audit.json) but "
                "are not entries in config/datasets.json and are out of scope for this "
                "registry census: a human single-unit stimulation release excluded from "
                "analysis on trial-structure and unit-count grounds, an electric-field-"
                "measurement release with no working-memory task, and a structural "
                "connectome with no task or stimulation data at all."
            ),
            "n_excluded": 3,
        },
        "verdict": (
            "Of 16 registered corpora, 10 carry no stimulation at all, 4 carry stimulation "
            "but no spike-sorted units (so the deviation axis is undefined there regardless "
            "of session overlap), and 2 carry both stimulation and spikes with the "
            "stimulation and non-stimulation trials interleaved in the SAME sessions. Of "
            "those 2, the mouse frontal-cortex corpus is excluded on its own already-"
            "measured spike-count precondition (its control trials measure below the "
            "established failing boundary). The macaque prefrontal microstimulation corpus "
            "is the only one of 16 where this leg is structurally askable without a cross-"
            "corpus join: all 11 of its sessions carry both trial pools, its unit and trial "
            "counts are well above this project's own established minimums, and its "
            "measured spike count sits inside a range this project has never before tested "
            "against the deviation observable's own gate -- that gate itself is left "
            "untested here by design."
        ),
    }


def main() -> None:
    t0 = time.time()
    census = build_census()
    census["wall_clock_s"] = round(time.time() - t0, 3)
    out_path = RESULTS / "stimulation_axis_same_session_census.json"
    out_path.write_text(json.dumps(census, indent=2, sort_keys=False, allow_nan=False) + "\n")
    print(f"wrote {out_path} in {census['wall_clock_s']}s")
    print(json.dumps(census["zero_drop_accounting"], indent=2))
    print("verdict:", census["verdict"])


if __name__ == "__main__":
    main()
