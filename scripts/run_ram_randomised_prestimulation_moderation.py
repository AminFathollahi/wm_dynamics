#!/usr/bin/env python3
"""Does the pre-stimulation population state say whether encoding-period electrical stimulation
will help or hurt the current word, in a corpus where stimulation assignment is an experimenter
design property rather than read from the subject's own state?

A separate arm of this project already asked whether the population state right before
stimulation moderates the stimulation effect, in a human intracranial corpus where stimulation is
triggered online by a classifier reading the subject's own state -- every number that arm reports
is therefore descriptive/associational, never causal, by its own admission. This module asks the
same question in the open-loop human free-recall stimulation corpus (OpenNeuro ds005489), where
word-level stimulation is assigned by one of two counterbalanced alternation phases fixed before
the session starts, independent of any measured neural state, and does not restate, re-fire or
revise that other arm's own delivered result.

MODERATOR. Per word event, "the component's own value" is the rate-free direction-deviation score
this project uses throughout its stimulation-response work: every recording channel's mean log
high-gamma power over the word's own -0.3 to +1.6 s peri-onset window (build_session_features's
window) is treated as one entry of a per-trial activity vector; each trial's vector is L2-normalised
to a unit direction, and its deviation is one minus the cosine between that unit direction and the
renormalised leave-one-out mean direction of every other word in the session (the identical
construction reused unchanged, function-for-function, from scripts/run_rate_free_state_geometry_
behavior_link.py and scripts/run_recording_tier_component_transfer.py -- this module copies rather
than imports it, matching the convention those two modules already established for this exact
function rather than pulling in either module's own unrelated corpus-loading code). Channels
directly involved in the driven bipolar pair (the stimulating anode or cathode contact) are excluded
from the feature vector for every trial in a session, so a stimulation-artifact deflection at the
driven site cannot mechanically inflate a trial's own deviation score.

The moderator for word i is that same score computed on word i-1, the immediately preceding word in
the same list -- strictly prior in time to word i's own stimulation, so no epoch is ever scored on
data its own stimulation could have altered.

TWO OUTCOMES, NEVER MERGED: the component's own displacement score on the CURRENT word, and whether
the current word was subsequently recalled.

MODEL. Per subject, the interaction between the current word's stimulation status and the moderator
is tested two ways, both reused unchanged from src/statistics.py and reported side by side rather
than one silently standing in for the other: (1) a single pooled fit with a subject random intercept
(linear_mixed_effects_test) over every admitted word event across all subjects at once, giving an
analytic Wald estimate, interval and p; (2) a subject-clustered statistic -- per subject, the partial
correlation (partial_correlation_permutation_test) between the interaction term and the outcome,
controlling for the two main effects and every covariate below, averaged across subjects -- with its
own interval from a subject-level bootstrap and its own p from a WITHIN-SUBJECT permutation null (the
moderator values are shuffled among a subject's own word events, the subject-clustered mean
recomputed, repeated many times). The decision rule below is fired off this second, within-subject-
permutation statistic, because it is the one this module's own null is built for; the pooled
random-intercept fit is reported alongside as a second, independent check, never silently substituted
for it.

CONTROLS, ALL MANDATORY, applied identically to both outcomes:
  1. Bias-only: every word's moderator is replaced by its own session's mean moderator (the same
     value repeated for every word in that session), voiding the trial-by-trial claim if it
     reproduces the native result's sign and significance.
  2. Preceding-word stimulation status: included as a covariate in the native and bias-only arms, and
     a separate restricted arm is reported using only current words whose preceding word was NOT
     itself stimulated (this covariate is dropped from the restricted arm's own regressors, since it
     is constant zero there by construction).
  3. Serial position and alternation phase: both included as covariates (phase as two indicator
     columns against a reference phase), and the realised stimulated fraction per serial position is
     reported from the sessions this module actually analysed, not carried forward as an assumption.
  4. List number: included as a covariate (time on task).

DECISION RULE, fixed before any result was read: for EACH outcome independently,
  - pre_stimulation_state_moderates_the_stimulation_effect: native significant (p<0.05), the
    restricted arm agrees in sign, and the bias-only arm does NOT reach p<0.05.
  - moderation_is_between_session_only: the bias-only arm reproduces the native estimate's sign and
    itself reaches p<0.05.
  - no_moderation_above_the_reported_bound: native not significant AND its minimum detectable effect
    at 80% power is below the 0.14 correlation-unit reference this project uses throughout.
  - underpowered_to_ask: native not significant and its minimum detectable effect is at or above 0.14.
  - native_significant_but_restricted_arm_disagrees_in_sign: a combination none of the four rules
    above covers (native significant, but the restricted arm's sign disagrees, and bias-only is not
    itself significant) -- disclosed explicitly rather than forced onto the nearest label.
  - outcomes_disagree is reported as a top-level comparison if the two outcomes' branches differ;
    both branches are always reported in full regardless.

ZERO DROP. Every one of the 78 candidate sessions, every list and every word event this module's
loader reaches is either analysed or refused with a named reason; seen == analysed + refused is
asserted at the session, subject and word level.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \\
    scripts/run_ram_randomised_prestimulation_moderation.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import csv
import json
import math
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, linear_mixed_effects_test, minimum_detectable_paired_difference,
    partial_correlation_permutation_test, permutation_pvalue, stable_seed,
)
from run_ram_openloop_pipeline import DATA, build_session_features  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "randomised_prestimulation_moderation_open_loop.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_ram_randomised_prestimulation_moderation"
SCHEMA_TAG = "v1_openloop_component_moderation_2026_08_28"

BEHAVIOURAL_REFERENCE_R_UNITS = 0.14  # this project's standing minimum-detectable-difference reference
MIN_STIM_LISTS = 3
MIN_NONSTIM_LISTS = 3
MIN_WORDS_PER_SUBJECT_FOR_EFFECT = 30
N_WITHIN_SUBJECT_NULL_DRAWS = 500
N_BOOT = 2000
N_PARTIAL_CORR_R_ONLY_PERM = 1  # The arm supplies its own pooled permutation null.
COMPOSITION_MATERIALITY_THRESHOLD = 0.05  # a 5-percentage-point difference, named explicitly and reused

PHASE_A = {1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 0, 12: 0}
PHASE_B = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0, 10: 0, 11: 1, 12: 1}

# The fixed set of serial positions each list-level label stimulates -- derived once from PHASE_A/
# PHASE_B rather than duplicated, since a list's own within-list stimulated-position pattern is
# entirely determined by its label (no_stimulation_list stimulates nothing) and never varies
# beyond these three fixed possibilities anywhere in this corpus (verified against every analysed
# session below, not assumed).
STIMULATED_SERIALPOS_BY_LABEL = {
    "phase_a": frozenset(sp for sp, v in PHASE_A.items() if v == 1),
    "phase_b": frozenset(sp for sp, v in PHASE_B.items() if v == 1),
    "no_stimulation_list": frozenset(),
}

DESIGN_BASED_NULL_RATIONALE = (
    "This experiment assigns stimulation to a whole list at a time, not to an individual word in "
    "isolation: every word in a list inherits one of three labels fixed for that whole list before "
    "the list began and independent of any measured neural state -- no stimulation at all, or one "
    "of two counterbalanced within-list serial-position patterns that each stimulate exactly half "
    "the list's own words. A permutation null that freely reshuffles the moderator value among a "
    "subject's own individual word events, as the within-subject and between-subject nulls above "
    "both do, does not respect that block structure: it treats every word as its own independently "
    "randomised unit, which this experiment's own design never did, and it also destroys the "
    "moderator's own autocorrelation within and across adjacent words, since a freely shuffled "
    "draw is less temporally structured than the real recording. Both departures push the same "
    "direction -- a null distribution built this way is narrower than the true design-based "
    "sampling distribution, so a p-value read from it is anti-conservative (optimistic). The "
    "design-based null reported alongside instead reassigns each list's own observed label among "
    "that same subject's own list slots, holding every word's own moderator value attached to its "
    "own word and holding each list's own internal stimulated-serial-position pattern intact, so "
    "the resulting null distribution is the one this experiment's own list-level randomisation "
    "actually generates. It does not replace the within-subject or between-subject null reported "
    "for every arm above; every arm's significance flag keeps the same basis it already had, "
    "recorded explicitly in that arm's own p_value_source field, and the design-based p-value is "
    "reported alongside for comparison rather than substituted in."
)

BETWEEN_SUBJECT_NULL_REASON = (
    "The bias-only moderator value is the same single number repeated for every word in one "
    "recording session, so it never varies from word to word within a session. A within-subject "
    "shuffle test works by randomly reassigning each word's own moderator value among that same "
    "subject's own words; if every word in a session already carries the identical value, "
    "reshuffling identical values among themselves changes nothing, so for a subject with only one "
    "recording session that shuffle is the mathematical identity operation and produces a null "
    "distribution with zero spread. Only a subject with more than one session, whose separate "
    "sessions can carry different mean values, contributes any real variation to that null at all, "
    "so a within-subject shuffle is near-untestable for a predictor built this way. A between-"
    "subject shuffle instead randomly reassigns each subject's own single summary moderator value "
    "to a different subject, while keeping that subject's own words grouped together, which "
    "produces genuine variation under the null regardless of how many sessions any one subject has."
)

WINNERS_CURSE_CAVEAT = (
    "This arm's own observed effect magnitude is SMALLER than the minimum effect this design could "
    "detect 80% of the time. Reaching significance despite sitting below that detection threshold "
    "is, conditional on significance, the standard winner's-curse situation: among the estimates a "
    "design this size can call significant at all, the ones that do reach significance while "
    "sitting below the design's own 80%-power detection threshold are expected to overstate the "
    "true effect. This does not weaken the significance finding itself -- the permutation test "
    "already accounts for the sample actually observed -- but the reported magnitude should be read "
    "as an upper bound on the true effect rather than a precise, robustly sized point estimate of it."
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# =======================================================================================================
# The component's per-trial value -- copied unchanged from scripts/run_rate_free_state_geometry_
# behavior_link.py / scripts/run_recording_tier_component_transfer.py (both already copy rather than
# import this function; this module follows that same established convention rather than forking it).
# =======================================================================================================

def rate_free_state_deviation(activity_by_unit: np.ndarray) -> np.ndarray:
    """Per trial, deviation_i = 1 - cosine(unit_vector_i, renormalised leave-one-out mean of every
    OTHER trial's own unit-normalised direction), from a (n_trials, n_features) array."""
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials = activity.shape[0]
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    total = np.nansum(unit_vectors, axis=0)
    n_valid = int(valid.sum())

    deviation = np.full(n_trials, np.nan)
    for i in range(n_trials):
        if not valid[i]:
            continue
        n_other = n_valid - 1
        if n_other < 1:
            continue
        loo_mean = (total - unit_vectors[i]) / n_other
        loo_norm = np.linalg.norm(loo_mean)
        if loo_norm == 0.0:
            continue
        cosine = float(np.dot(unit_vectors[i], loo_mean / loo_norm))
        deviation[i] = 1.0 - cosine
    return deviation


# =======================================================================================================
# Checkpointing -- one small JSON per session (per-word scalars only, never the raw epoch array, per
# this project's disk-floor discipline).
# =======================================================================================================

def _checkpoint_path(unit: str) -> Path:
    import re
    return CHECKPOINT_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', unit)}.json"


def _load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True or data.get("schema_tag") != SCHEMA_TAG:
        return None
    return restore_checkpoint(data["record"])


def _save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "schema_tag": SCHEMA_TAG, "record": checkpoint_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _run_checkpointed(unit: str, fit_fn):
    cached = _load_checkpoint(unit)
    if cached is not None:
        return _normalize_record(cached)
    record = fit_fn()
    _save_checkpoint(unit, record)
    return _normalize_record(record)


def _normalize_record(record: dict) -> dict:
    """A resumed (checkpoint-reloaded) record's numeric-leaf-list fields come back as float
    ndarrays (restore_checkpoint's own documented heuristic), while a freshly computed record's
    are plain Python lists of the field's real type -- this project has already been bitten by
    that divergence silently changing downstream behaviour, so every consumer of a session record
    gets the same native-Python-list, correctly-typed shape regardless of which path produced it."""
    if record.get("status") != "computed":
        return record
    record["component"] = [float(x) for x in record["component"]]
    record["recalled"] = [int(x) for x in record["recalled"]]
    record["stim_flag"] = [int(x) for x in record["stim_flag"]]
    record["serialpos"] = [int(x) for x in record["serialpos"]]
    record["list_number"] = [int(x) for x in record["list_number"]]
    record["phase"] = [str(x) for x in record["phase"]]
    return record


# =======================================================================================================
# Per-session extraction: component value per word, list/phase classification. Caches only per-word
# scalars, never the (n_words, n_bins, n_channels) epoch array build_session_features returns.
# =======================================================================================================

def _classify_lists(list_number: np.ndarray, serialpos: np.ndarray, stim_flag: np.ndarray) -> dict[int, dict]:
    by_list: dict[int, list[int]] = defaultdict(list)
    for i, ln in enumerate(list_number):
        if int(ln) == -1:
            continue
        by_list[int(ln)].append(i)
    meta = {}
    for ln, idxs in by_list.items():
        flags = {int(serialpos[i]): int(stim_flag[i]) for i in idxs}
        is_stim = any(v == 1 for v in flags.values())
        if not is_stim:
            phase = "no_stimulation_list"
        elif all(flags.get(sp) == PHASE_A.get(sp) for sp in flags):
            phase = "phase_a"
        elif all(flags.get(sp) == PHASE_B.get(sp) for sp in flags):
            phase = "phase_b"
        else:
            phase = "phase_unclassified"
        meta[ln] = {"is_stim": is_stim, "phase": phase, "n_words": len(idxs)}
    return meta


def _build_subject_list_inventory(analysed_sessions: dict[str, dict]) -> dict[str, list[dict]]:
    """Per subject, the full observed list-level design across every one of that subject's own
    analysed sessions: for every list the experiment actually ran, which of the three labels
    (no_stimulation_list / phase_a / phase_b) it was assigned. Computed straight from each
    session's own already-checkpointed per-word arrays via _classify_lists (the identical
    function _extract_session itself uses), so this never disagrees with the per-word phase
    values the rest of the module already relies on. A list this module could not classify
    (phase_unclassified) is never a candidate for permutation and is excluded here exactly as its
    own words are already excluded at the word level in _build_word_rows."""
    inventory: dict[str, list[dict]] = defaultdict(list)
    for session_key, rec in analysed_sessions.items():
        list_meta = _classify_lists(np.array(rec["list_number"]), np.array(rec["serialpos"]),
                                     np.array(rec["stim_flag"]))
        subject = rec["subject"]
        for list_number, meta in list_meta.items():
            if meta["phase"] == "phase_unclassified":
                continue
            inventory[subject].append({"session": session_key, "list_number": list_number, "phase": meta["phase"]})
    return dict(inventory)


def _list_arrangement_diagnostics(subject_list_inventory: dict[str, list[dict]]) -> dict[str, dict]:
    """Per subject, how many distinct ways that subject's own observed list-level labels can be
    reassigned among that subject's own list slots (the size of the design-based null's own
    permutation space), and the smallest two-sided p-value that subject's own within-subject
    statistic could ever reach if that space were exhaustively enumerated rather than Monte-Carlo
    sampled. Computed in log10 space (via the log-gamma function) rather than by building the
    exact factorial, since a subject with over a hundred lists has an arrangement count with
    hundreds of digits -- far too large to be a meaningful JSON integer, and unnecessary since only
    its order of magnitude is ever used."""
    diagnostics = {}
    for subject, entries in subject_list_inventory.items():
        n = len(entries)
        counts = Counter(e["phase"] for e in entries)
        log10_arrangements = (math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts.values())) / math.log(10)
        smallest_p = 0.0 if log10_arrangements > 300 else (10.0 ** -log10_arrangements if n else None)
        diagnostics[subject] = {
            "n_lists": n,
            "list_label_counts": dict(counts),
            "n_distinct_list_label_arrangements_log10": log10_arrangements if n else None,
            "smallest_attainable_two_sided_p_if_exhaustively_enumerated_for_this_subject_alone": smallest_p,
            "insufficient_arrangements_for_p_below_0_05": (
                (smallest_p is not None) and smallest_p >= 0.05
            ),
        }
    return diagnostics


def _characterise_list_level_stimulation(analysed_sessions: dict[str, dict]) -> dict:
    """Empirical characterisation of how the stimulation indicator actually varies within a list,
    measured directly from the sessions this module analysed rather than assumed: the count of
    stimulated words per stimulated list, and the distinct sets of serial positions stimulation
    was delivered at."""
    stim_word_count_histogram: dict[int, int] = defaultdict(int)
    stim_position_set_counts: dict[tuple, int] = defaultdict(int)
    n_lists_total = 0
    n_stim_lists = 0
    for rec in analysed_sessions.values():
        list_meta = _classify_lists(np.array(rec["list_number"]), np.array(rec["serialpos"]),
                                     np.array(rec["stim_flag"]))
        n_words_seen = rec["n_words_seen"]
        for list_number, meta in list_meta.items():
            n_lists_total += 1
            if not meta["is_stim"]:
                continue
            n_stim_lists += 1
            stim_sp = tuple(sorted(
                int(rec["serialpos"][i]) for i in range(n_words_seen)
                if int(rec["list_number"][i]) == list_number and int(rec["stim_flag"][i]) == 1
            ))
            stim_word_count_histogram[len(stim_sp)] += 1
            stim_position_set_counts[stim_sp] += 1
    return {
        "n_lists_characterised": n_lists_total,
        "n_stimulated_lists": n_stim_lists,
        "n_unstimulated_lists": n_lists_total - n_stim_lists,
        "stimulated_word_count_per_stimulated_list_histogram": dict(sorted(stim_word_count_histogram.items())),
        "distinct_stimulated_serial_position_sets": {
            str(list(k)): v for k, v in sorted(stim_position_set_counts.items())
        },
        "reading": (
            "Every stimulated list in this corpus stimulates exactly the same number of its own "
            "words, at exactly one of two fixed, mutually exclusive sets of serial positions. The "
            "stimulation indicator is therefore NOT constant within a stimulated list at the "
            "individual-word level -- roughly half of a stimulated list's own words are "
            "stimulated and half are not -- but the PATTERN of which half is fixed by that list's "
            "own label before the list began, independent of any measured neural state. This is "
            "the block-randomised, list-level design the design-based null above is built for."
        ) if n_stim_lists else "no stimulated lists present in the sessions this module analysed",
    }


def _extract_session(ieeg_json: Path) -> dict:
    try:
        feat = build_session_features(ieeg_json, data_root=DATA, derive_stim_from_stim_on=False, return_epochs=True)
    except Exception as e:
        return {"status": "refused", "reason": f"build_session_features_raised_{type(e).__name__}"}
    if feat is None:
        return {"status": "refused",
                "reason": "build_session_features_returned_none_insufficient_or_unusable_data_per_its_own_"
                          "internal_admission_checks"}

    ch_names = feat["ch_names"]
    anode, cathode = feat["anode"], feat["cathode"]
    driven = {anode, cathode}
    keep = np.array([not (set(name.split("-")) & driven) for name in ch_names])
    if int(keep.sum()) < 2:
        return {"status": "refused", "reason": "fewer_than_2_channels_after_driven_contact_exclusion"}

    activity = feat["epochs_log"].mean(axis=1)[:, keep]  # (n_words, n_kept_channels), time-averaged
    component = rate_free_state_deviation(activity)

    list_number = feat["list_number"]
    serialpos = feat["serialpos"]
    stim_flag = feat["stim_flag"]
    recalled = feat["recalled"]

    list_meta = _classify_lists(list_number, serialpos, stim_flag)
    n_stim_lists = sum(1 for m in list_meta.values() if m["is_stim"])
    n_nonstim_lists = sum(1 for m in list_meta.values() if not m["is_stim"])
    if n_stim_lists < MIN_STIM_LISTS or n_nonstim_lists < MIN_NONSTIM_LISTS:
        return {"status": "refused", "reason": "fewer_than_3_stimulated_or_3_nonstimulated_lists",
                "n_stim_lists": n_stim_lists, "n_nonstim_lists": n_nonstim_lists}

    phase_of_word = np.array([list_meta.get(int(ln), {}).get("phase", "invalid_list") for ln in list_number],
                              dtype=object)

    return {
        "status": "computed",
        "session": feat["session"],
        "n_words_seen": int(feat["n_words"]),
        "n_channels_kept": int(keep.sum()),
        "n_stim_lists": n_stim_lists, "n_nonstim_lists": n_nonstim_lists,
        "component": component.tolist(),
        "recalled": recalled.astype(int).tolist(),
        "stim_flag": stim_flag.astype(int).tolist(),
        "serialpos": serialpos.astype(int).tolist(),
        "list_number": list_number.astype(int).tolist(),
        "phase": list(phase_of_word),
    }


# =======================================================================================================
# Word-event table: preceding-word moderator, zero-drop word-level refusal.
# =======================================================================================================

def _build_word_rows(session_key: str, subject: str, rec: dict) -> tuple[list[dict], list[dict]]:
    """Returns (analysed_rows, refused_rows) for one computed session."""
    n = rec["n_words_seen"]
    component = rec["component"]
    stim_flag = rec["stim_flag"]
    recalled = rec["recalled"]
    serialpos = rec["serialpos"]
    list_number = rec["list_number"]
    phase = rec["phase"]

    analysed, refused = [], []
    for i in range(n):
        if list_number[i] == -1:
            refused.append({"session": session_key, "word_index": i, "reason": "invalid_list_number"})
            continue
        if i == 0 or list_number[i - 1] != list_number[i]:
            refused.append({"session": session_key, "word_index": i,
                             "reason": "first_word_of_list_no_preceding_moderator"})
            continue
        if not np.isfinite(component[i]) or not np.isfinite(component[i - 1]):
            refused.append({"session": session_key, "word_index": i, "reason": "component_undefined_for_this_or_"
                                                                                "preceding_trial"})
            continue
        if phase[i] == "phase_unclassified" or phase[i] == "invalid_list":
            refused.append({"session": session_key, "word_index": i, "reason": "list_phase_unclassified"})
            continue
        analysed.append({
            "session": session_key, "subject": subject, "word_index": i,
            "displacement": float(component[i]), "recalled": float(recalled[i]),
            "stim": float(stim_flag[i]), "preceding_stim": float(stim_flag[i - 1]),
            "moderator_native": float(component[i - 1]),
            "serialpos": float(serialpos[i]), "list_number": float(list_number[i]), "phase": phase[i],
        })
    return analysed, refused


def _raw_event_inventory(events_tsv: Path) -> dict:
    """Count word events by list before signal-level admission."""
    with events_tsv.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        list_counts: dict[int, int] = defaultdict(int)
        for row in rows:
            if row.get("trial_type") == "WORD":
                list_counts[int(row["list"])] += 1
    return {"n_words": sum(list_counts.values()), "list_counts": dict(list_counts)}


# =======================================================================================================
# Model fitting.
# =======================================================================================================

def _covariates(rows: list[dict], moderator_key: str, include_preceding_stim: bool,
                 extra_keys: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    stim = np.array([r["stim"] for r in rows], dtype=float)
    moderator = np.array([r[moderator_key] for r in rows], dtype=float)
    serialpos = np.array([r["serialpos"] for r in rows], dtype=float)
    list_number = np.array([r["list_number"] for r in rows], dtype=float)
    is_phase_b = np.array([1.0 if r["phase"] == "phase_b" else 0.0 for r in rows], dtype=float)
    is_no_stim_list = np.array([1.0 if r["phase"] == "no_stimulation_list" else 0.0 for r in rows], dtype=float)
    out = {"stim": stim, "moderator": moderator, "serialpos": serialpos, "list_number": list_number,
           "is_phase_b": is_phase_b, "is_no_stim_list": is_no_stim_list}
    if include_preceding_stim:
        out["preceding_stim"] = np.array([r["preceding_stim"] for r in rows], dtype=float)
    for key in extra_keys:
        if key == "time_on_task":
            out["time_on_task"] = np.array([r["word_index"] for r in rows], dtype=float)
        elif key == "current_word_component_amplitude":
            out["current_word_component_amplitude"] = np.array([r["displacement"] for r in rows], dtype=float)
        else:
            raise ValueError(f"unknown extra covariate key: {key}")
    return out


def _controls_list(cov: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Every main effect and nuisance covariate (stim, moderator, serial position, list number, phase
    dummies, and preceding-word stimulation where present) -- the interaction term itself is built
    separately by the caller and is never one of these, but BOTH main effects it is built from (stim,
    moderator) belong in this list, since a partial correlation of the interaction against an outcome
    that does not also control for its own main effects would attribute ordinary main-effect variance
    to the interaction."""
    return list(cov.values())


def _subject_partial_effects(rows: list[dict], outcome_key: str, moderator_key: str,
                              include_preceding_stim: bool, seed_tag: str,
                              extra_keys: tuple[str, ...] = ()) -> tuple[dict[str, float], dict]:
    subjects = sorted({r["subject"] for r in rows})
    effects, excluded = {}, {}
    for s in subjects:
        srows = [r for r in rows if r["subject"] == s]
        if len(srows) < MIN_WORDS_PER_SUBJECT_FOR_EFFECT:
            excluded[s] = {"reason": "fewer_than_min_trials_for_subject_level_effect", "n": len(srows)}
            continue
        outcome = np.array([r[outcome_key] for r in srows], dtype=float)
        cov = _covariates(srows, moderator_key, include_preceding_stim, extra_keys)
        interaction = cov["stim"] * cov["moderator"]
        controls = _controls_list(cov)
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|{s}"))
        res = partial_correlation_permutation_test(outcome, interaction, controls=controls,
                                                     n_perm=N_PARTIAL_CORR_R_ONLY_PERM, rng=rng)
        if res["status"] != "computed" or not np.isfinite(res["r"]):
            excluded[s] = {"reason": "partial_correlation_not_computable_zero_variance_residual", "n": len(srows)}
            continue
        effects[s] = res["r"]
    return effects, excluded


def _within_subject_null(rows: list[dict], outcome_key: str, moderator_key: str, include_preceding_stim: bool,
                          admissible_subjects: list[str], n_draws: int, seed_tag: str,
                          extra_keys: tuple[str, ...] = ()) -> np.ndarray:
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    draws = np.full(n_draws, np.nan)
    for k in range(n_draws):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|draw{k}"))
        per_subject_r = []
        for s in admissible_subjects:
            srows = by_subject[s]
            outcome = np.array([r[outcome_key] for r in srows], dtype=float)
            cov = _covariates(srows, moderator_key, include_preceding_stim, extra_keys)
            shuffled_moderator = rng.permutation(cov["moderator"])
            interaction = cov["stim"] * shuffled_moderator
            controls = [shuffled_moderator if k_ == "moderator" else v for k_, v in cov.items()]
            res = partial_correlation_permutation_test(outcome, interaction, controls=controls,
                                                         n_perm=N_PARTIAL_CORR_R_ONLY_PERM, rng=rng)
            if res["status"] == "computed" and np.isfinite(res["r"]):
                per_subject_r.append(res["r"])
        if per_subject_r:
            draws[k] = float(np.mean(per_subject_r))
    return draws


def _between_subject_null(rows: list[dict], outcome_key: str, moderator_key: str, include_preceding_stim: bool,
                           admissible_subjects: list[str], extra_keys: tuple[str, ...], n_draws: int,
                           seed_tag: str) -> np.ndarray:
    """Null for a subject- or session-constant moderator (e.g. a bias-only control): each
    admissible subject's own single summary moderator value (the mean of their own rows' moderator,
    which for a session-constant predictor already equals that constant) is reassigned to a
    DIFFERENT subject, holding every subject's own rows -- and every other covariate on them --
    fixed. This produces genuine null variation even when a subject contributes only one session,
    unlike a within-subject shuffle of a constant value (see BETWEEN_SUBJECT_NULL_REASON)."""
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    subject_level_moderator = {
        s: float(np.mean([r[moderator_key] for r in by_subject[s]])) for s in admissible_subjects
    }
    values_arr = np.array([subject_level_moderator[s] for s in admissible_subjects])
    draws = np.full(n_draws, np.nan)
    for k in range(n_draws):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|between{k}"))
        assigned = dict(zip(admissible_subjects, rng.permutation(values_arr)))
        per_subject_r = []
        for s in admissible_subjects:
            srows = by_subject[s]
            outcome = np.array([r[outcome_key] for r in srows], dtype=float)
            cov = _covariates(srows, moderator_key, include_preceding_stim, extra_keys)
            shuffled_moderator = np.full(len(srows), assigned[s])
            interaction = cov["stim"] * shuffled_moderator
            controls = [shuffled_moderator if k_ == "moderator" else v for k_, v in cov.items()]
            res = partial_correlation_permutation_test(outcome, interaction, controls=controls,
                                                         n_perm=N_PARTIAL_CORR_R_ONLY_PERM, rng=rng)
            if res["status"] == "computed" and np.isfinite(res["r"]):
                per_subject_r.append(res["r"])
        if per_subject_r:
            draws[k] = float(np.mean(per_subject_r))
    return draws


def _apply_list_labels_to_rows(srows: list[dict], label_of: dict[tuple, str]) -> dict[str, np.ndarray]:
    """Given one subject's own rows and a per-list label assignment (a (session, list_number) key
    to no_stimulation_list/phase_a/phase_b label), recompute the stimulation-derived covariates --
    each row's own current-word and preceding-word stimulation status, and the two phase-indicator
    columns built from it -- via the same fixed per-label serial-position pattern
    (STIMULATED_SERIALPOS_BY_LABEL) every real list in this corpus already follows. Passing the
    OBSERVED label assignment reproduces each row's own already-recorded stim status exactly;
    passing a permuted assignment is exactly what the design-based null below draws its null
    statistic from. A row whose own list is absent from label_of (never happens for a real,
    fully-populated subject_list_inventory) is treated as unstimulated."""
    n = len(srows)
    new_stim = np.zeros(n)
    new_preceding_stim = np.zeros(n)
    new_is_phase_b = np.zeros(n)
    new_is_no_stim_list = np.zeros(n)
    for i, r in enumerate(srows):
        label = label_of.get((r["session"], int(r["list_number"])))
        stim_positions = STIMULATED_SERIALPOS_BY_LABEL.get(label, frozenset())
        sp = int(r["serialpos"])
        new_stim[i] = 1.0 if sp in stim_positions else 0.0
        new_preceding_stim[i] = 1.0 if (sp - 1) in stim_positions else 0.0
        new_is_phase_b[i] = 1.0 if label == "phase_b" else 0.0
        new_is_no_stim_list[i] = 1.0 if label == "no_stimulation_list" else 0.0
    return {"stim": new_stim, "preceding_stim": new_preceding_stim,
            "is_phase_b": new_is_phase_b, "is_no_stim_list": new_is_no_stim_list}


def _design_based_null(rows: list[dict], outcome_key: str, moderator_key: str, include_preceding_stim: bool,
                        admissible_subjects: list[str], subject_list_inventory: dict[str, list[dict]],
                        n_draws: int, seed_tag: str, extra_keys: tuple[str, ...] = ()) -> np.ndarray:
    """The null appropriate to how this experiment actually randomises: at the level of a whole
    list, not an individual word. Within each subject, that subject's own observed list-level
    labels (no_stimulation_list / phase_a / phase_b) are randomly reassigned among that subject's
    own list slots -- a permutation, so the count of each label this subject actually has is
    exactly preserved on every draw, which in particular means the number of stimulated lists
    never changes. Every word's own moderator value stays attached to its own word throughout
    (this null never touches the moderator); only that word's own current- and preceding-word
    stimulation status, and the phase-indicator covariates built from it, are recomputed from the
    label its own list was reassigned on this draw, via the same fixed per-label serial-position
    pattern (STIMULATED_SERIALPOS_BY_LABEL) every real list in this corpus already follows. The
    row set analysed is held fixed exactly as it is for the within-subject and between-subject
    nulls above -- only the list-level label assignment is exchanged, never which rows are in the
    sample."""
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    draws = np.full(n_draws, np.nan)
    for k in range(n_draws):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|design{k}"))
        per_subject_r = []
        for s in admissible_subjects:
            srows = by_subject[s]
            if not srows:
                continue
            entries = subject_list_inventory.get(s, [])
            list_keys = [(e["session"], e["list_number"]) for e in entries]
            labels = np.array([e["phase"] for e in entries], dtype=object)
            shuffled_labels = rng.permutation(labels) if len(labels) else labels
            label_of = dict(zip(list_keys, shuffled_labels))

            outcome = np.array([r[outcome_key] for r in srows], dtype=float)
            moderator = np.array([r[moderator_key] for r in srows], dtype=float)
            serialpos = np.array([r["serialpos"] for r in srows], dtype=float)
            list_number = np.array([r["list_number"] for r in srows], dtype=float)
            recomputed = _apply_list_labels_to_rows(srows, label_of)
            new_stim = recomputed["stim"]

            interaction = new_stim * moderator
            cov = {"stim": new_stim, "moderator": moderator, "serialpos": serialpos, "list_number": list_number,
                   "is_phase_b": recomputed["is_phase_b"], "is_no_stim_list": recomputed["is_no_stim_list"]}
            if include_preceding_stim:
                cov["preceding_stim"] = recomputed["preceding_stim"]
            for key in extra_keys:
                if key == "time_on_task":
                    cov["time_on_task"] = np.array([r["word_index"] for r in srows], dtype=float)
                elif key == "current_word_component_amplitude":
                    cov["current_word_component_amplitude"] = np.array([r["displacement"] for r in srows],
                                                                         dtype=float)
            controls = list(cov.values())
            res = partial_correlation_permutation_test(outcome, interaction, controls=controls,
                                                         n_perm=N_PARTIAL_CORR_R_ONLY_PERM, rng=rng)
            if res["status"] == "computed" and np.isfinite(res["r"]):
                per_subject_r.append(res["r"])
        if per_subject_r:
            draws[k] = float(np.mean(per_subject_r))
    return draws


def _pooled_mixed_effects_check(rows: list[dict], outcome_key: str, moderator_key: str,
                                 include_preceding_stim: bool, extra_keys: tuple[str, ...] = ()) -> dict:
    outcome = np.array([r[outcome_key] for r in rows], dtype=float)
    subject = np.array([r["subject"] for r in rows])
    cov = _covariates(rows, moderator_key, include_preceding_stim, extra_keys)
    interaction = cov["stim"] * cov["moderator"]
    covariate_matrix = np.column_stack(_controls_list(cov))
    fit = linear_mixed_effects_test(outcome, interaction, subject, covariates=covariate_matrix)
    return {
        "beta": fit.get("beta"), "se": fit.get("se"), "p_value": fit.get("p_value"),
        "ci95_lo": (fit["beta"] - 1.959963984540054 * fit["se"]) if fit.get("converged") and np.isfinite(fit.get("se", np.nan)) else None,
        "ci95_hi": (fit["beta"] + 1.959963984540054 * fit["se"]) if fit.get("converged") and np.isfinite(fit.get("se", np.nan)) else None,
        "converged": fit.get("converged"), "reason": fit.get("reason"), "n": len(rows),
        "note": "pooled single fit with a subject random intercept (statsmodels MixedLM, Wald p and normal-"
                "approximation 95% CI); a second, independent check alongside the primary subject-clustered/"
                "within-subject-permutation statistic below, never substituted for it",
    }


def _fit_arm(rows: list[dict], outcome_key: str, moderator_key: str, include_preceding_stim: bool,
             seed_tag: str, extra_keys: tuple[str, ...] = (), significance_null: str = "within_subject",
             subject_list_inventory: dict[str, list[dict]] | None = None) -> dict:
    """significance_null selects which permutation null the 'significant' flag and reported
    p_value_source are drawn from: 'within_subject' (default, appropriate whenever the moderator
    varies from word to word within a subject, e.g. the native and restricted arms) or
    'between_subject' (appropriate for a bias-only arm, whose moderator is constant within a
    session -- see BETWEEN_SUBJECT_NULL_REASON for why a within-subject shuffle cannot test that
    case). Both nulls are always computed and reported; only the significance determination
    differs. A third null, appropriate to how this experiment actually randomises stimulation --
    at the level of a whole list, holding every word's own moderator value fixed to its own word
    (see DESIGN_BASED_NULL_RATIONALE) -- is ALSO always computed and reported whenever
    subject_list_inventory is supplied, again never substituted for whichever null
    significance_null already names."""
    if len(rows) < MIN_WORDS_PER_SUBJECT_FOR_EFFECT:
        return {"status": "not_computable", "reason": "fewer_than_min_trials", "n_words": len(rows)}
    effects, excluded = _subject_partial_effects(rows, outcome_key, moderator_key, include_preceding_stim,
                                                  seed_tag, extra_keys)
    admissible = sorted(effects.keys())
    n_subjects = len(admissible)
    if n_subjects < 2:
        return {"status": "not_computable", "reason": "fewer_than_2_subjects_with_a_computable_effect",
                "n_words": len(rows), "n_subjects_excluded": excluded}
    values = np.array([effects[s] for s in admissible], dtype=float)
    observed_mean = float(np.mean(values))
    rng_boot = np.random.default_rng(stable_seed(f"{seed_tag}|bootstrap"))
    _, ci_lo, ci_hi = bootstrap_ci(values, np.mean, n_boot=N_BOOT, rng=rng_boot)
    mdd = minimum_detectable_paired_difference(values)

    within_draws = _within_subject_null(rows, outcome_key, moderator_key, include_preceding_stim, admissible,
                                         N_WITHIN_SUBJECT_NULL_DRAWS, seed_tag, extra_keys)
    finite_within = within_draws[np.isfinite(within_draws)]
    within_p = permutation_pvalue(np.abs(finite_within) >= abs(observed_mean)) if finite_within.size else float("nan")
    within_is_degenerate = bool(finite_within.size and float(np.std(finite_within)) < 1e-12)

    pooled_check = _pooled_mixed_effects_check(rows, outcome_key, moderator_key, include_preceding_stim, extra_keys)

    inventory = subject_list_inventory or {}
    design_draws = _design_based_null(rows, outcome_key, moderator_key, include_preceding_stim, admissible,
                                       inventory, N_WITHIN_SUBJECT_NULL_DRAWS, seed_tag, extra_keys)
    finite_design = design_draws[np.isfinite(design_draws)]
    design_p = (permutation_pvalue(np.abs(finite_design) >= abs(observed_mean))
                if finite_design.size else float("nan"))
    design_is_degenerate = bool(finite_design.size and float(np.std(finite_design)) < 1e-12)
    arrangement_diagnostics = _list_arrangement_diagnostics(inventory)
    admissible_arrangement_diagnostics = {s: arrangement_diagnostics[s] for s in admissible
                                           if s in arrangement_diagnostics}
    n_insufficient_arrangements = sum(
        1 for d in admissible_arrangement_diagnostics.values() if d["insufficient_arrangements_for_p_below_0_05"]
    )

    result = {
        "status": "computed",
        "n_words": len(rows), "n_subjects_admissible": n_subjects,
        "n_subjects_excluded_insufficient_data": len(excluded), "subjects_excluded": excluded,
        "subject_clustered_mean": observed_mean,
        "bootstrap_ci95_lo": float(ci_lo), "bootstrap_ci95_hi": float(ci_hi),
        "within_subject_permutation_p": within_p, "n_null_draws_finite": int(finite_within.size),
        "within_subject_null_is_degenerate_zero_spread": within_is_degenerate,
        "design_based_permutation_p": design_p, "n_null_draws_finite_design_based": int(finite_design.size),
        "design_based_null_is_degenerate_zero_spread": design_is_degenerate,
        "design_based_null_list_arrangement_diagnostics_by_admissible_subject": admissible_arrangement_diagnostics,
        "n_admissible_subjects_with_insufficient_list_arrangements_for_p_below_0_05": n_insufficient_arrangements,
        "minimum_detectable_difference_80pct_power": mdd,
        "clears_behavioural_reference_0_14": (mdd["mdd"] < BEHAVIOURAL_REFERENCE_R_UNITS
                                               if mdd.get("status") == "computed" else None),
        "per_subject_effects": effects,
        "pooled_subject_random_intercept_check": pooled_check,
    }

    if significance_null == "between_subject":
        session_counts = defaultdict(set)
        for r in rows:
            if r["subject"] in effects:
                session_counts[r["subject"]].add(r["session"])
        n_multi_session = sum(1 for s in admissible if len(session_counts[s]) > 1)
        between_draws = _between_subject_null(rows, outcome_key, moderator_key, include_preceding_stim,
                                               admissible, extra_keys, N_WITHIN_SUBJECT_NULL_DRAWS, seed_tag)
        finite_between = between_draws[np.isfinite(between_draws)]
        between_p = (permutation_pvalue(np.abs(finite_between) >= abs(observed_mean))
                     if finite_between.size else float("nan"))
        result.update({
            "between_subject_permutation_p": between_p,
            "n_null_draws_finite_between_subject": int(finite_between.size),
            "n_admissible_subjects_with_more_than_one_session": n_multi_session,
            "reason": BETWEEN_SUBJECT_NULL_REASON,
            "p_value_source": "between_subject_permutation",
            "significant": bool(between_p < 0.05) if np.isfinite(between_p) else False,
        })
    else:
        result.update({
            "p_value_source": "within_subject_permutation",
            "significant": bool(within_p < 0.05) if np.isfinite(within_p) else False,
        })

    # Winner's-curse disclosure: whenever a significant estimate's own magnitude sits below the
    # design's own 80%-power detection threshold, that reported magnitude is expected to be
    # upward-biased -- checked for every arm, not only the one it was first noticed in, since the
    # relationship between an observed effect and its own MDD is a property of any arm's fit.
    effect_below_own_mdd = bool(
        mdd.get("status") == "computed" and np.isfinite(observed_mean) and abs(observed_mean) < mdd["mdd"]
    )
    result["effect_magnitude_below_its_own_minimum_detectable_difference"] = effect_below_own_mdd
    result["winners_curse_caveat"] = WINNERS_CURSE_CAVEAT if (effect_below_own_mdd and result["significant"]) else None
    result["significance_null_basis"] = result["p_value_source"]
    result["design_based_null_rationale"] = DESIGN_BASED_NULL_RATIONALE
    return result


def _add_bias_only_moderator(rows: list[dict]) -> None:
    """Mutates `rows` in place, adding 'moderator_bias_only' = that row's own session's mean of
    'moderator_native' across every row in the same session -- the bias-only construction this
    control always uses: replace every word's moderator with its own session's mean moderator, so
    the predictor carries only between-session information. If this reproduces the native result's
    sign and significance, the native result is not a trial-by-trial one."""
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r["moderator_native"])
    session_mean = {s: float(np.mean(v)) for s, v in by_session.items()}
    for r in rows:
        r["moderator_bias_only"] = session_mean[r["session"]]


def _restrict_to_unstimulated_preceding_words(rows: list[dict]) -> list[dict]:
    """Keep current words whose immediately preceding word was unstimulated."""
    return [row for row in rows if row["preceding_stim"] == 0.0]


def _add_bias_only_moderator_scoped(rows: list[dict], out_key: str) -> None:
    """Mutates `rows` in place, adding `out_key` = that row's own session's mean of
    'moderator_native' computed ONLY across the rows passed in here -- unlike
    _add_bias_only_moderator, which always averages over every row in a session regardless of
    which arm calls it, this lets a restricted-scope arm (e.g. current words whose own preceding
    word was not stimulated) get a bias-only control fitted on that same restricted population,
    rather than reusing a control fitted on the full, unrestricted word pool."""
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r["moderator_native"])
    session_mean = {s: float(np.mean(v)) for s, v in by_session.items()}
    for r in rows:
        r[out_key] = session_mean[r["session"]]


def _nuisance_partialling_ladder(restricted: dict, restricted_nuisance_partialled: dict,
                                  nuisance_keys: tuple[str, ...]) -> dict:
    """Packages the already-computed restricted and restricted_nuisance_partialled arms as an
    explicit two-covariate-vs-four-covariate comparison, with no new fit: serial position within
    list and list index (list number) within session are already part of the mandatory base model
    shared by every arm in this analysis (the two-covariate baseline, identical to the plain
    restricted arm above); the nuisance-partialled arm adds time-on-task and -- outcomes other than
    the current word's own displacement only -- the current word's own directional-deviation score
    on top of that same baseline (the four-covariate version)."""
    def _summary(arm: dict) -> dict:
        if arm.get("status") != "computed":
            return {"status": arm.get("status"), "reason": arm.get("reason")}
        return {
            "subject_clustered_mean": arm["subject_clustered_mean"],
            "within_subject_permutation_p": arm["within_subject_permutation_p"],
            "significant": arm["significant"], "n_words": arm["n_words"],
            "bootstrap_ci95_lo": arm["bootstrap_ci95_lo"], "bootstrap_ci95_hi": arm["bootstrap_ci95_hi"],
        }

    added_covariates = ["each word's own running position within its own recording session (time on task)"]
    if "current_word_component_amplitude" in nuisance_keys:
        added_covariates.append("the current word's own directional-deviation score")

    delta = None
    if restricted.get("status") == "computed" and restricted_nuisance_partialled.get("status") == "computed":
        delta = float(restricted_nuisance_partialled["subject_clustered_mean"] - restricted["subject_clustered_mean"])

    return {
        "two_covariate_baseline": {
            "covariates": ["serial position within list", "list index (list number) within session"],
            "note": "identical to the restricted arm reported above; these two covariates are already "
                    "part of the mandatory base model applied to every arm in this analysis, not unique "
                    "to this comparison",
            **_summary(restricted),
        },
        "four_covariate_partialled": {
            "covariates": ["serial position within list", "list index (list number) within session",
                           *added_covariates],
            "note": "identical to the restricted_nuisance_partialled arm reported above; layered on top "
                    "of the identical two-covariate baseline, so the two rows here isolate exactly what "
                    "the two added covariates change",
            **_summary(restricted_nuisance_partialled),
        },
        "change_in_subject_clustered_mean_attributable_to_the_added_covariates": delta,
    }


def _restricted_vs_its_bias_only_sign_relationship(restricted: dict, restricted_bias_only: dict) -> str:
    """Plain-words statement of whether the restricted arm's own within-subject estimate and its
    own restricted-population bias-only control (between-subject null) agree or disagree in sign,
    stated explicitly rather than left implicit in a table of numbers."""
    if restricted.get("status") != "computed" or not restricted["significant"]:
        return ("not applicable: the restricted arm itself does not reach significance, so there is no "
                "significant within-subject estimate to compare against its own bias-only control")
    if restricted_bias_only.get("status") != "computed" or not restricted_bias_only["significant"]:
        return ("not applicable: the restricted arm's own bias-only control does not itself reach "
                "significance, so there is no significant between-subject estimate to compare against "
                "the restricted arm's own within-subject estimate")
    if np.sign(restricted["subject_clustered_mean"]) == np.sign(restricted_bias_only["subject_clustered_mean"]):
        return ("the restricted arm's own within-subject estimate and its own restricted-population "
                "bias-only control are both significant and point in the SAME direction, so the "
                "between-subject association could in principle be part of the explanation for the "
                "within-subject one")
    return (
        "SIGN REVERSAL: the restricted arm's own within-subject estimate is significant in one "
        "direction while its own restricted-population bias-only control (tested with the between-"
        "subject null) is significant in the OPPOSITE direction. The association between the pre-"
        "stimulation state and this outcome runs one way ACROSS subjects and the other way WITHIN a "
        "subject. Because the two point in opposite directions, the between-subject association "
        "cannot be the explanation for the within-subject one -- but the two numbers describe two "
        "different comparisons at two different levels and must never be pooled, averaged, or quoted "
        "together as a single association."
    )


def _fmt_p(value) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.6f}"


def _stimulated_lists_only_control_reading(restricted: dict, restricted_stimulated_lists_only: dict) -> str:
    """Plain-words statement of whether the restricted arm's own displacement interaction estimate
    survives once no-stimulation lists are dropped entirely, so every remaining row -- both the
    current-word-stimulated and current-word-unstimulated rows alike -- is drawn from a list that
    itself received stimulation. Without this restriction, part of the restricted arm's own
    contrast could be a plain difference between words from stimulated lists and words from
    no-stimulation lists, rather than a moderation effect that plays out within a single list."""
    if restricted.get("status") != "computed":
        return "not applicable: the restricted arm itself is not computable"
    if restricted_stimulated_lists_only.get("status") != "computed":
        return (f"not computable: {restricted_stimulated_lists_only.get('reason', restricted_stimulated_lists_only.get('status'))}")
    same_sign = (np.sign(restricted["subject_clustered_mean"])
                 == np.sign(restricted_stimulated_lists_only["subject_clustered_mean"]))
    survives = bool(restricted_stimulated_lists_only["significant"] and same_sign)
    verdict = "SURVIVES" if survives else "DOES NOT SURVIVE"
    explanation = (
        "the effect is not explained by a plain difference between stimulated-list words and "
        "no-stimulation-list words" if survives else
        "part of the restricted arm's own contrast could be a difference between stimulated-list "
        "words and no-stimulation-list words rather than a within-list moderation effect"
    )
    return (
        f"{verdict}: restricted to words drawn only from lists that themselves received "
        f"stimulation, the interaction estimate is "
        f"{restricted_stimulated_lists_only['subject_clustered_mean']:.6f} "
        f"(within-subject permutation p={_fmt_p(restricted_stimulated_lists_only['within_subject_permutation_p'])}, "
        f"design-based permutation p={_fmt_p(restricted_stimulated_lists_only['design_based_permutation_p'])}), "
        f"versus the unrestricted-by-list-type restricted arm's {restricted['subject_clustered_mean']:.6f} "
        f"(within-subject permutation p={_fmt_p(restricted['within_subject_permutation_p'])}, "
        f"design-based permutation p={_fmt_p(restricted['design_based_permutation_p'])}). {explanation}."
    )


def _restriction_composition_and_contamination_check(all_rows: list[dict], restricted_rows: list[dict],
                                                       threshold: float) -> dict:
    """Measures, rather than assumes, whether excluding current words whose preceding word was
    stimulated shifts the retained population's composition -- and separately measures what
    fraction of the FULL (native-arm) population has a moderator value that was itself displaced by
    stimulation on the preceding word, since the moderator is meant to index an unperturbed pre-
    stimulation state."""
    retained_ids = {id(r) for r in restricted_rows}
    excluded_rows = [r for r in all_rows if id(r) not in retained_ids]
    n_all, n_retained, n_excluded = len(all_rows), len(restricted_rows), len(excluded_rows)

    def _serialpos_fractions(rows: list[dict]) -> dict[int, float]:
        counts: dict[int, int] = defaultdict(int)
        for r in rows:
            counts[int(r["serialpos"])] += 1
        n = len(rows)
        return {sp: counts[sp] / n for sp in range(1, 13) if n}

    retained_sp = _serialpos_fractions(restricted_rows)
    excluded_sp = _serialpos_fractions(excluded_rows)
    serialpos_abs_diffs = {sp: abs(retained_sp.get(sp, 0.0) - excluded_sp.get(sp, 0.0)) for sp in range(1, 13)}
    max_serialpos_abs_diff = max(serialpos_abs_diffs.values()) if serialpos_abs_diffs else 0.0

    retained_stim_fraction = float(np.mean([r["stim"] for r in restricted_rows])) if n_retained else None
    excluded_stim_fraction = float(np.mean([r["stim"] for r in excluded_rows])) if n_excluded else None
    stim_fraction_abs_diff = (abs(retained_stim_fraction - excluded_stim_fraction)
                               if retained_stim_fraction is not None and excluded_stim_fraction is not None
                               else None)

    retained_stim_list_fraction = (float(np.mean([r["phase"] != "no_stimulation_list" for r in restricted_rows]))
                                    if n_retained else None)
    excluded_stim_list_fraction = (float(np.mean([r["phase"] != "no_stimulation_list" for r in excluded_rows]))
                                    if n_excluded else None)
    stim_list_fraction_abs_diff = (
        abs(retained_stim_list_fraction - excluded_stim_list_fraction)
        if retained_stim_list_fraction is not None and excluded_stim_list_fraction is not None else None
    )

    material_checks = {
        "serial_position_distribution": max_serialpos_abs_diff > threshold,
        "current_word_own_stimulation_fraction": (stim_fraction_abs_diff or 0.0) > threshold,
        "stimulated_list_fraction": (stim_list_fraction_abs_diff or 0.0) > threshold,
    }
    any_material = any(material_checks.values())
    if any_material:
        materiality_reading = (
            "At least one of the three composition comparisons exceeds the "
            f"{threshold:g}-fraction materiality threshold: "
            + "; ".join(name for name, flagged in material_checks.items() if flagged)
            + ". This means the restricted population is not a like-for-like subset of the full "
              "population on that dimension, and any difference between the restricted and native "
              "interaction estimates could partly reflect that compositional shift rather than the "
              "pre-stimulation state alone; the restricted arm's own controls (its restricted-"
              "population bias-only arm and its nuisance-partialled re-fit, both of which already "
              "condition on serial position) are the appropriate way to guard against exactly this."
        )
    else:
        materiality_reading = (
            "None of the three composition comparisons exceeds the "
            f"{threshold:g}-fraction materiality threshold. Restricting to current words whose "
            "preceding word was not stimulated does not materially change the serial-position "
            "distribution, the current word's own stimulation rate, or the stimulated-list "
            "fraction, consistent with the preceding word's stimulation status being itself "
            "randomised rather than a source of selection on the current word."
        )

    contamination_fraction = n_excluded / n_all if n_all else None
    contamination_reading = (
        f"{n_excluded} of {n_all} native-arm word events ({contamination_fraction:.4f} of the native "
        "population) carry a moderator value taken from a preceding word that was itself stimulated "
        "-- that moderator value was displaced by the very intervention under study rather than "
        "indexing an unperturbed pre-stimulation state. This does not change which arm is designated "
        "primary, but it means a substantial minority of the native arm's own moderator values are "
        "not measuring what the moderator is defined to measure, which is exactly the population the "
        "restricted arm removes."
    )

    return {
        "n_native_arm_words": n_all, "n_retained_restricted_arm_words": n_retained,
        "n_excluded_preceding_word_stimulated_words": n_excluded,
        "materiality_threshold_fraction": threshold,
        "serial_position_fraction_by_group": {"retained": retained_sp, "excluded": excluded_sp},
        "max_absolute_serial_position_fraction_difference": max_serialpos_abs_diff,
        "current_word_own_stimulation_fraction": {
            "retained": retained_stim_fraction, "excluded": excluded_stim_fraction,
            "absolute_difference": stim_fraction_abs_diff,
        },
        "stimulated_list_fraction": {
            "retained": retained_stim_list_fraction, "excluded": excluded_stim_list_fraction,
            "absolute_difference": stim_list_fraction_abs_diff,
        },
        "material_checks": material_checks,
        "any_comparison_material": any_material,
        "materiality_reading": materiality_reading,
        "native_arm_contaminated_moderator_fraction": contamination_fraction,
        "native_arm_contamination_reading": contamination_reading,
    }


def _classify_branch(native: dict, bias_only: dict, restricted: dict, restricted_bias_only: dict,
                      restricted_nuisance_partialled: dict) -> str:
    if native.get("status") != "computed":
        return "not_computable"
    native_sig = native["significant"]
    native_mdd = native["minimum_detectable_difference_80pct_power"]
    native_sign = np.sign(native["subject_clustered_mean"])
    bias_sig = bias_only.get("status") == "computed" and bias_only["significant"]
    bias_sign_matches = (bias_only.get("status") == "computed"
                          and np.sign(bias_only["subject_clustered_mean"]) == native_sign)
    restricted_sign_matches = (restricted.get("status") == "computed"
                                and np.sign(restricted["subject_clustered_mean"]) == native_sign)

    if native_sig and restricted_sign_matches and not bias_sig:
        return "pre_stimulation_state_moderates_the_stimulation_effect"
    if bias_sig and bias_sign_matches:
        return "moderation_is_between_session_only"

    # New branch, added alongside the rules above without amending any of their text: covers a
    # native contrast that does not itself reach significance while the restricted arm -- current
    # words whose own preceding word was not stimulated, i.e. an unperturbed pre-stimulation state
    # -- is significant on its own terms, corroborated by its own restricted-population bias-only
    # control (which either is not itself significant, or is significant in a DIFFERENT direction
    # and so cannot be the explanation for the restricted arm's own sign -- the same
    # significant-AND-same-sign standard the "moderation_is_between_session_only" rule above
    # applies to the native arm) and its own nuisance-partialled re-fit (still significant, same
    # sign as the unpartialled restricted estimate).
    restricted_sig = restricted.get("status") == "computed" and restricted["significant"]
    restricted_bias_sig = restricted_bias_only.get("status") == "computed" and restricted_bias_only["significant"]
    restricted_bias_explains_restricted = (
        restricted_bias_sig and restricted.get("status") == "computed"
        and np.sign(restricted_bias_only["subject_clustered_mean"]) == np.sign(restricted["subject_clustered_mean"])
    )
    partialled_sig = (restricted_nuisance_partialled.get("status") == "computed"
                       and restricted_nuisance_partialled["significant"])
    partialled_sign_matches = (
        restricted.get("status") == "computed"
        and restricted_nuisance_partialled.get("status") == "computed"
        and np.sign(restricted_nuisance_partialled["subject_clustered_mean"])
        == np.sign(restricted["subject_clustered_mean"])
    )
    if (not native_sig and restricted_sig and not restricted_bias_explains_restricted
            and partialled_sig and partialled_sign_matches):
        return "restricted_arm_significant_with_its_own_controls_despite_native_null"

    if not native_sig and native_mdd.get("status") == "computed" and native_mdd["mdd"] < BEHAVIOURAL_REFERENCE_R_UNITS:
        return "no_moderation_above_the_reported_bound"
    if not native_sig and native_mdd.get("status") == "computed" and native_mdd["mdd"] >= BEHAVIOURAL_REFERENCE_R_UNITS:
        return "underpowered_to_ask"
    if native_sig and not restricted_sign_matches:
        return "native_significant_but_restricted_arm_disagrees_in_sign"
    return "branch_criteria_not_matched_see_full_object"


# =======================================================================================================
# Main
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    # "Seen" is every session this corpus's own BIDS layout declares (an events.tsv under
    # sub-*/ses-*/ieeg/, per the corpus's own layout), not merely the subset that also happens to
    # carry a bipolar recording -- a session with an events.tsv but no acq-bipolar ieeg.json/edf
    # pair is REFUSED with a named reason below, never silently absent from "seen".
    events_files = sorted(DATA.glob("sub-*/ses-*/ieeg/sub-*_ses-*_task-FR2_events.tsv"))
    n_sessions_seen = len(events_files)

    per_session, session_excluded, raw_inventory = {}, [], {}
    for events_tsv in events_files:
        subj = events_tsv.parts[-4]
        stem = events_tsv.name.removesuffix("_events.tsv")
        session_key = str((events_tsv.parent / stem).relative_to(DATA))
        raw_inventory[session_key] = _raw_event_inventory(events_tsv)
        ieeg_json = events_tsv.parent / f"{stem}_acq-bipolar_ieeg.json"
        if not ieeg_json.exists():
            record = {"status": "refused", "reason": "no_acq_bipolar_ieeg_recording_present_for_this_session"}
        else:
            record = _run_checkpointed(f"session|{session_key}", lambda p=ieeg_json: _extract_session(p))
        per_session[session_key] = {"subject": subj, **record}
        if record.get("status") != "computed":
            session_excluded.append({"session": session_key, "subject": subj,
                                      "reason": record.get("reason", record.get("status"))})
        _log(f"  {session_key}: {record.get('status')}"
             f"{' (' + record.get('reason', '') + ')' if record.get('status') != 'computed' else ''}")

    analysed_sessions = {k: v for k, v in per_session.items() if v.get("status") == "computed"}
    n_sessions_analysed = len(analysed_sessions)
    session_zero_drop = {
        "sessions_seen": n_sessions_seen, "sessions_analysed": n_sessions_analysed,
        "sessions_refused": len(session_excluded), "refusals": session_excluded,
        "reconciles": bool(n_sessions_seen == n_sessions_analysed + len(session_excluded)),
        "n_subjects_seen": len({p.parts[-4] for p in events_files}),
        "n_subjects_analysed": len({v["subject"] for v in analysed_sessions.values()}),
        "per_session_status": [
            {
                "session": key,
                "subject": rec["subject"],
                "status": "analysed" if rec.get("status") == "computed" else "refused",
                "reason": None if rec.get("status") == "computed" else rec.get("reason", rec.get("status")),
            }
            for key, rec in per_session.items()
        ],
    }
    assert session_zero_drop["reconciles"], "session-level zero-drop accounting does not reconcile"

    all_analysed_rows, all_refused_words = [], []
    word_rows_by_session, refused_words_by_session = {}, {}
    for session_key, rec in analysed_sessions.items():
        rows, refused = _build_word_rows(session_key, rec["subject"], rec)
        word_rows_by_session[session_key] = rows
        refused_words_by_session[session_key] = refused
        all_analysed_rows.extend(rows)
        all_refused_words.extend(refused)

    list_status, per_session_word_status = [], []
    for session_key, rec in per_session.items():
        inventory = raw_inventory[session_key]
        if rec.get("status") != "computed":
            reason = f"session_refused:{rec.get('reason', rec.get('status'))}"
            per_session_word_status.append({
                "session": session_key,
                "words_seen": inventory["n_words"],
                "words_analysed": 0,
                "words_refused": inventory["n_words"],
                "refusal_reason_counts": {reason: inventory["n_words"]},
            })
            for list_number, n_words in sorted(inventory["list_counts"].items()):
                list_status.append({
                    "session": session_key, "list_number": list_number,
                    "n_words_seen": n_words, "n_words_analysed": 0,
                    "n_words_refused": n_words, "status": "refused", "reason": reason,
                })
            continue

        rows = word_rows_by_session[session_key]
        refused = refused_words_by_session[session_key]
        analysed_by_list: dict[int, int] = defaultdict(int)
        refused_by_list: dict[int, int] = defaultdict(int)
        refusal_reasons: dict[str, int] = defaultdict(int)
        for row in rows:
            analysed_by_list[int(row["list_number"])] += 1
        for row in refused:
            list_number = int(rec["list_number"][row["word_index"]])
            refused_by_list[list_number] += 1
            refusal_reasons[row["reason"]] += 1
        per_session_word_status.append({
            "session": session_key,
            "words_seen": inventory["n_words"],
            "words_analysed": len(rows),
            "words_refused": len(refused),
            "refusal_reason_counts": dict(refusal_reasons),
        })
        for list_number, n_words in sorted(inventory["list_counts"].items()):
            n_analysed = analysed_by_list[list_number]
            n_refused = refused_by_list[list_number]
            assert n_words == n_analysed + n_refused
            list_status.append({
                "session": session_key, "list_number": list_number,
                "n_words_seen": n_words, "n_words_analysed": n_analysed,
                "n_words_refused": n_refused,
                "status": "analysed" if n_analysed else "refused",
                "reason": None if n_analysed else "no_word_in_list_has_a_computable_preceding_moderator",
            })

    n_words_seen = sum(v["n_words"] for v in raw_inventory.values())
    n_words_analysed = len(all_analysed_rows)
    n_words_refused = n_words_seen - n_words_analysed
    refusal_counts: dict[str, int] = defaultdict(int)
    for status in per_session_word_status:
        for reason, count in status["refusal_reason_counts"].items():
            refusal_counts[reason] += count
    word_zero_drop = {
        "words_seen": n_words_seen,
        "words_seen_in_analysed_sessions": sum(v["n_words_seen"] for v in analysed_sessions.values()),
        "words_analysed": n_words_analysed, "words_refused": n_words_refused,
        "refusal_reason_counts": dict(refusal_counts),
        "reconciles": bool(n_words_seen == n_words_analysed + n_words_refused),
        "per_session_status": per_session_word_status,
    }
    assert word_zero_drop["reconciles"], "word-level zero-drop accounting does not reconcile"

    n_lists_seen = len(list_status)
    n_lists_analysed = sum(row["status"] == "analysed" for row in list_status)
    list_zero_drop = {
        "lists_seen": n_lists_seen,
        "lists_analysed": n_lists_analysed,
        "lists_refused": n_lists_seen - n_lists_analysed,
        "reconciles": n_lists_seen == n_lists_analysed + (n_lists_seen - n_lists_analysed),
        "per_list_status": list_status,
    }
    assert list_zero_drop["reconciles"], "list-level zero-drop accounting does not reconcile"

    n_subjects_analysed_regression = len({r["subject"] for r in all_analysed_rows})
    subject_session_counts = defaultdict(set)
    for r in all_analysed_rows:
        subject_session_counts[r["subject"]].add(r["session"])
    n_multi_session_subjects = sum(1 for sessions in subject_session_counts.values() if len(sessions) > 1)
    subject_zero_drop = {
        "subjects_seen": session_zero_drop["n_subjects_seen"],
        "subjects_contributing_at_least_one_analysed_word": n_subjects_analysed_regression,
        "subjects_with_zero_analysed_words": session_zero_drop["n_subjects_seen"] - n_subjects_analysed_regression,
        "subjects_with_more_than_one_analysed_session": n_multi_session_subjects,
    }
    subject_zero_drop["reconciles"] = bool(
        subject_zero_drop["subjects_seen"]
        == subject_zero_drop["subjects_contributing_at_least_one_analysed_word"]
        + subject_zero_drop["subjects_with_zero_analysed_words"]
    )
    assert subject_zero_drop["reconciles"], "subject-level zero-drop accounting does not reconcile"

    # Realised stimulated fraction per serial position, over the sessions actually analysed. Reported
    # two ways: over every word (diluted by the ~20% of lists that carry no stimulation at all), and
    # restricted to words in a stimulated list only (the alternation-phase design's own balance claim,
    # which is a within-stimulated-list property, not a property of the full word pool).
    fraction_by_serialpos: dict[int, dict] = {}
    for sp in range(1, 13):
        flags_all = [rec["stim_flag"][i] for rec in analysed_sessions.values()
                     for i in range(rec["n_words_seen"]) if rec["serialpos"][i] == sp]
        flags_stim_lists = [rec["stim_flag"][i] for rec in analysed_sessions.values()
                             for i in range(rec["n_words_seen"])
                             if rec["serialpos"][i] == sp and rec["phase"][i] != "no_stimulation_list"]
        if flags_all:
            fraction_by_serialpos[sp] = {
                "n_words_all_lists": len(flags_all),
                "stimulated_fraction_all_lists": float(np.mean(flags_all)),
                "n_words_stimulated_lists_only": len(flags_stim_lists),
                "stimulated_fraction_within_stimulated_lists": (
                    float(np.mean(flags_stim_lists)) if flags_stim_lists else None),
            }

    _add_bias_only_moderator(all_analysed_rows)

    restricted_rows = _restrict_to_unstimulated_preceding_words(all_analysed_rows)
    _add_bias_only_moderator_scoped(restricted_rows, "moderator_bias_only_restricted_scope")

    # The design-based null's own permutation population: every analysed list's own observed
    # stimulation label, per subject, straight from the checkpointed session records -- verified
    # against the data (not assumed) via list_level_stimulation_pattern_characterisation below.
    subject_list_inventory = _build_subject_list_inventory(analysed_sessions)
    list_level_stimulation_pattern_characterisation = _characterise_list_level_stimulation(analysed_sessions)
    _log(f"  list-level stimulation pattern: {list_level_stimulation_pattern_characterisation['n_stimulated_lists']}"
         f" of {list_level_stimulation_pattern_characterisation['n_lists_characterised']} analysed lists "
         "stimulated, word-count-per-stimulated-list histogram="
         f"{list_level_stimulation_pattern_characterisation['stimulated_word_count_per_stimulated_list_histogram']}")

    # Words drawn only from lists that themselves received stimulation, dropping no-stimulation
    # lists entirely -- both the current-word-stimulated and current-word-unstimulated rows then
    # come from the same lists, so a plain difference between stimulated-list and no-stimulation-
    # list words cannot be part of the restricted arm's own displacement contrast (Task B control).
    restricted_stimulated_lists_only_rows = [row for row in restricted_rows
                                              if row["phase"] != "no_stimulation_list"]

    restriction_composition_and_contamination_check = _restriction_composition_and_contamination_check(
        all_analysed_rows, restricted_rows, COMPOSITION_MATERIALITY_THRESHOLD)
    _log(f"  restriction composition check: any_material="
         f"{restriction_composition_and_contamination_check['any_comparison_material']}, native "
         f"contamination fraction="
         f"{restriction_composition_and_contamination_check['native_arm_contaminated_moderator_fraction']:.4f}")

    restricted_nuisance_partialled_note = (
        "serial position and list number are already controlled in every arm's base model; this "
        "nuisance-partialled arm adds two further regressors on top of that base model: each "
        "word's running position within its own recording session as a time-on-task control, and "
        "-- for the recall outcome only -- the current word's own directional-deviation score. "
        "That second regressor is omitted for the displacement outcome because the current word's "
        "own directional-deviation score IS the displacement outcome itself for that outcome; "
        "adding a variable as a covariate for predicting that exact same variable would trivially "
        "and circularly remove all of the outcome's own variance rather than control for a "
        "genuine nuisance."
    )

    outcomes = {}
    for outcome_key, outcome_label in (("displacement", "displacement"), ("recalled", "recalled")):
        seed_base = f"randomised_prestimulation_moderation|{outcome_label}"
        native = _fit_arm(all_analysed_rows, outcome_key, "moderator_native", True, f"{seed_base}|native",
                           subject_list_inventory=subject_list_inventory)
        bias_only = _fit_arm(all_analysed_rows, outcome_key, "moderator_bias_only", True, f"{seed_base}|bias_only",
                              significance_null="between_subject", subject_list_inventory=subject_list_inventory)
        restricted = _fit_arm(restricted_rows, outcome_key, "moderator_native", False, f"{seed_base}|restricted",
                               subject_list_inventory=subject_list_inventory)
        restricted_bias_only = _fit_arm(restricted_rows, outcome_key, "moderator_bias_only_restricted_scope", False,
                                         f"{seed_base}|restricted_bias_only", significance_null="between_subject",
                                         subject_list_inventory=subject_list_inventory)
        nuisance_keys = (("time_on_task",) if outcome_key == "displacement"
                          else ("time_on_task", "current_word_component_amplitude"))
        restricted_nuisance_partialled = _fit_arm(restricted_rows, outcome_key, "moderator_native", False,
                                                   f"{seed_base}|restricted_nuisance_partialled",
                                                   extra_keys=nuisance_keys,
                                                   subject_list_inventory=subject_list_inventory)
        branch = _classify_branch(native, bias_only, restricted, restricted_bias_only,
                                   restricted_nuisance_partialled)
        outcomes[outcome_label] = {
            "native": native, "bias_only": bias_only, "restricted": restricted,
            "restricted_bias_only": restricted_bias_only,
            "restricted_nuisance_partialled": restricted_nuisance_partialled,
            "restricted_nuisance_partialled_covariates_note": restricted_nuisance_partialled_note,
            "nuisance_partialling_ladder": _nuisance_partialling_ladder(
                restricted, restricted_nuisance_partialled, nuisance_keys),
            "restricted_vs_its_bias_only_sign_relationship": _restricted_vs_its_bias_only_sign_relationship(
                restricted, restricted_bias_only),
            "restricted_arm_n_words": len(restricted_rows), "branch": branch,
        }
        if outcome_key == "displacement":
            restricted_stimulated_lists_only = _fit_arm(
                restricted_stimulated_lists_only_rows, outcome_key, "moderator_native", False,
                f"{seed_base}|restricted_stimulated_lists_only", subject_list_inventory=subject_list_inventory)
            outcomes[outcome_label]["restricted_stimulated_lists_only"] = restricted_stimulated_lists_only
            outcomes[outcome_label]["restricted_stimulated_lists_only_n_words"] = len(
                restricted_stimulated_lists_only_rows)
            outcomes[outcome_label]["restricted_stimulated_lists_only_control_reading"] = (
                _stimulated_lists_only_control_reading(restricted, restricted_stimulated_lists_only))
        _log(f"  outcome={outcome_label}: branch={branch}")

    displacement_branch = outcomes["displacement"]["branch"]
    recall_branch = outcomes["recalled"]["branch"]
    branch_comparison = "outcomes_disagree" if displacement_branch != recall_branch else "outcomes_agree"

    negative_interaction_clinical_interpretation = (
        "This analysis tests one number per outcome: how much the effect of delivering "
        "stimulation on a word changes depending on how unusual the brain's own recorded state "
        "already looked on the word immediately before that stimulation. A NEGATIVE value for "
        "that number means the following: stimulating after a pre-stimulation state that already "
        "looks far from the person's own typical, session-average state adds LESS extra "
        "displacement to the following word's own state than stimulating after a pre-stimulation "
        "state that looks close to typical. Put another way, when the brain's state right before "
        "stimulation already looks atypical, stimulation itself appears to push the following "
        "word's state comparatively less far from typical than it does when the pre-stimulation "
        "state looked calm and ordinary. For a clinician choosing when to deliver stimulation, a "
        "negative interaction of this kind would argue for timing stimulation to moments when the "
        "immediately preceding brain state already looks atypical, if the clinical goal is to "
        "minimise how far stimulation additionally displaces the brain's state, and for expecting "
        "a larger state displacement from stimulation delivered when the immediately preceding "
        "state looks calm and close to the person's own typical baseline. This describes an "
        "association between the pre-stimulation state and the SIZE of the following displacement "
        "only; it does not by itself say whether a larger or smaller displacement is good or bad "
        "for that person's memory, which is answered separately, and only for the recall outcome."
    )

    other_arm_disclosure = (
        "A separate arm of this project already tested pre-stimulation moderation in a classifier-"
        "triggered (closed-loop) human intracranial stimulation corpus, where treatment assignment reads "
        "the subject's own online-decoded state and every number that arm reports is descriptive/"
        "associational, never causal, by its own admission. This artifact answers the same question in an "
        "experimenter-scheduled (open-loop) corpus with word-level stimulation assignment fixed by a "
        "counterbalanced alternation phase independent of any measured neural state, and does not restate, "
        "re-fire or revise that other arm's own delivered result."
    )

    wall_clock_s = round(time.time() - t0, 3)
    scope = {
        "corpus": "OpenNeuro ds005489 open-loop human free-recall stimulation",
        "sessions_seen": n_sessions_seen,
        "sessions_analysed": n_sessions_analysed,
        "sessions_refused": len(session_excluded),
        "subjects_seen": subject_zero_drop["subjects_seen"],
        "subjects_analysed": n_subjects_analysed_regression,
        "lists_seen": list_zero_drop["lists_seen"],
        "words_seen": word_zero_drop["words_seen"],
        "parameters": {
            "minimum_stimulated_lists": MIN_STIM_LISTS,
            "minimum_nonstimulated_lists": MIN_NONSTIM_LISTS,
            "minimum_words_per_subject": MIN_WORDS_PER_SUBJECT_FOR_EFFECT,
            "within_subject_null_draws": N_WITHIN_SUBJECT_NULL_DRAWS,
            "bootstrap_draws": N_BOOT,
            "behavioural_reference_r_units": BEHAVIOURAL_REFERENCE_R_UNITS,
        },
        "seed": "deterministic stable_seed labels recorded by outcome and arm",
        "wall_clock_s": wall_clock_s,
    }
    output = {
        "analysis_id": "randomised_prestimulation_moderation_open_loop",
        "schema_version": "1.0.0",
        "code_commit": git_commit(ROOT),
        "corpus": "open_loop_human_free_recall_stimulation",
        "other_arm_disclosure": other_arm_disclosure,
        "negative_interaction_clinical_interpretation": negative_interaction_clinical_interpretation,
        "component_definition": (
            "per word, one minus the cosine between that word's own L2-normalised per-channel mean log "
            "high-gamma power direction (-0.3 to +1.6 s peri-word-onset window, channels at the driven "
            "stimulating contact excluded) and the renormalised leave-one-out mean direction of every "
            "other word in the session"
        ),
        "moderator_definition": "the component's own value on the immediately preceding word event within the "
                                 "same list, strictly prior in time to the current word's stimulation",
        "behavioural_reference_r_units": BEHAVIOURAL_REFERENCE_R_UNITS,
        "scope": scope,
        "session_zero_drop_accounting": session_zero_drop,
        "list_zero_drop_accounting": list_zero_drop,
        "word_zero_drop_accounting": word_zero_drop,
        "subject_zero_drop_accounting": subject_zero_drop,
        "realised_stimulated_fraction_per_serial_position": fraction_by_serialpos,
        "restriction_composition_and_contamination_check": restriction_composition_and_contamination_check,
        "list_level_stimulation_pattern_characterisation": list_level_stimulation_pattern_characterisation,
        "design_based_null_rationale": DESIGN_BASED_NULL_RATIONALE,
        "design_based_null_list_arrangement_diagnostics_by_subject": _list_arrangement_diagnostics(
            subject_list_inventory),
        "outcomes": outcomes,
        "branch_comparison": branch_comparison,
        "wall_clock_s": wall_clock_s,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)
    _log(f"wrote {OUTPUT_PATH} in {output['wall_clock_s']:.1f}s -- displacement branch: {displacement_branch}, "
         f"recall branch: {recall_branch}, comparison: {branch_comparison}")


if __name__ == "__main__":
    main()
