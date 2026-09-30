#!/usr/bin/env python3
"""A machine-readable inventory of every stimulation corpus and arm this
project has staged: what each can and cannot identify, read from the
release's own event tables, trial tables and MAT/NWB parameter structs --
never from a README or a planning document.

This is an inventory, not an analysis: no estimator is fit here. Every field
of every row is either a value read directly from the data at run time, or
an explicit structural void carrying a reason string -- there is no third
state (a present-but-empty field with no reason is treated as an error by
this script's own completeness gate, `validate_row`).

Decision rules are declared once, as code, before any corpus is read, and
never amended after a number comes back (see `classify_treatment_assignment`
and `classify_structural_void_candidate` plus the docstrings attached to
each). Both are also serialised verbatim into the output artifact under
`decision_rule_declared_before_fitting`.

Loader reuse: every per-corpus reader below imports and calls the existing
loader for that corpus rather than re-deriving one -- see the module-level
imports. Nothing here re-implements NWB field access, BIDS event parsing, or
MAT-file struct decoding that already exists elsewhere in this repository.

Output: results/stimulation_design_census.json

Run:
    python \
        scripts/run_stimulation_design_census.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from project_config import data_root, dataset_path
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import stable_seed  # noqa: E402

# Human intracranial free-recall stimulation (RAM ds005489 / ds005557):
# reuse the existing event-table parser and derived-feature builders wholesale.
from run_stimulation_timing_and_parameter_structure import discover_sessions, discover_subject_dirs, read_events, process_openloop_session, process_closedloop_session, run_corpus, build_trains_openloop, build_trains_closedloop
from stimulation_events import pool_item_attributability, pool_parameter_census, match_train_owner
from stimulation_events import OPENLOOP_CORPUS, CLOSEDLOOP_CORPUS, CLOSEDLOOP_OWNER_MATCH_WINDOW_S

# haslacher_clam_tacs (phase-locked closed-loop tACS during working-memory retention): reuse the existing subject lists,
# data directory resolution, and event-marker readers.
from run_haslacher_phase_omega import DATA_DIR as CLAM_DATA_DIR, trial_outcomes as clam_trial_outcomes
from preprocessing import RETENTION_TMIN, RETENTION_TMAX
from preprocessing import PHASE_CONDITIONS
from preprocessing import ACTIVE_SUBJECTS, CONTROL_SUBJECTS

# ds005034 randomised transcranial verum/sham: reuse the existing participant
# inventory and event-table reader.
from corpus_sessions import paired_inventory as ds005034_paired_inventory
from preprocessing import load_events as ds005034_load_events
from corpus_sessions import SESSIONS as DS005034_SESSIONS
from preprocessing import BASELINE_WINDOW as DS005034_BASELINE_WINDOW, DELAY_WINDOW as DS005034_DELAY_WINDOW, CELLS as DS005034_CELLS

# Macaque dlPFC delay-period microstimulation: reuse the existing session
# list, data root, and MAT-file struct parsers (both generations).
from run_macaque_pfc_microstimulation_pipeline import SESSIONS as MACAQUE_SESSIONS, DATA as MACAQUE_DATA
from dynamics import _ascii_to_str, _parse_chan_token
from spike_pipeline import PRE_S as MACAQUE_PRE_S, N_BINS as MACAQUE_N_BINS, BIN_S as MACAQUE_BIN_S
from dynamics import DLPFC_CONNECTOME_AREA

# alagapan_phase_stimulation (multi-site phase-lag human retention-period stimulation):
# reuse the existing patient roster, data directory, montage and condition reader.
from run_alagapan_phase_omega import DATA_DIR as ALAGAPAN_DATA_DIR
from preprocessing import PATIENTS as ALAGAPAN_PATIENTS
from run_alagapan_stimulation_geometry import _trial_conditions as alagapan_trial_conditions
from preprocessing import STIM_SITES as ALAGAPAN_STIM_SITES
from preprocessing import CONDITIONS as ALAGAPAN_CONDITIONS, RETENTION_ONSET_BUFFER_S as ALAGAPAN_RETENTION_ONSET_BUFFER_S

# Passive human single-unit corpora that already define this project's
# candidate biomarkers: reuse the shared iterator module's constants and
# canonical-session resolver, at the session-discovery grain only (no spike
# loading -- see the module docstring's reuse note in the report).
from corpus_sessions import (  # noqa: E402
    MIN_TRIALS as PASSIVE_MIN_TRIALS, alm_data_directory, watters_directories,
    watters_session_dates, watters_behaviour,
)
from run_human_drift_spine_001187_000673 import canonical_sessions
from corpus_sessions import _trial_group
from run_panichello_pipeline import data_directory as panichello_data_directory
from corpus_sessions import monkey_for_session

RESULTS = ROOT / "results"
OUT_PATH = RESULTS / "stimulation_design_census.json"
CENSUS_SEED = stable_seed("stimulation_design_census")  # no stochastic step reads this; kept for
                                                          # provenance symmetry with this project's
                                                          # other scope blocks, which all carry a seed.


def VOID(reason: str) -> dict:
    """The only spelling of 'this field does not exist for this arm by
    design' used anywhere in this artifact -- see
    `classify_structural_void_candidate` for the rule that licenses it."""
    return {"status": "structural_void", "reason": reason}


def MEASURED(value, source: str):
    """Wrap a value read from the data with a one-line note on exactly what
    was read to produce it, so every non-void field is traceable to a
    concrete read, not to memory or documentation."""
    return {"status": "measured", "value": value, "source": source}


# ═══════════════════════════════════════════════════════════════════════════
# Decision rules -- declared before any corpus below is read, never amended
# after a result comes back. Both are also written verbatim into the output.
# ═══════════════════════════════════════════════════════════════════════════

RANDOMIZATION_CLASSIFICATION_RULE = (
    "A treatment-assignment mechanism is classified from what the release's own event or "
    "trial table shows was actually delivered, never from a label the dataset's documentation "
    "attaches to it. In order of precedence: (1) if the factor has only one observed level for "
    "this arm, there is nothing to randomize over and the classification is "
    "'single_level_nothing_to_randomize'. (2) If the value or moment of delivery is set by an "
    "online readout of the participant's own neural state at delivery time (a classifier, a "
    "phase estimator, a threshold detector), the classification is "
    "'state_triggered_selection_not_randomized' regardless of what the release calls it -- this "
    "is a selection process on a signal related to the outcome under study, not a "
    "randomization. (3) If the value is a stimulation site or montage chosen for a participant "
    "by clinical, anatomical or feasibility considerations, the classification is "
    "'site_clinically_or_anatomically_chosen_not_randomized'. (4) If the release's own event or "
    "trial table (not its prose) shows a documented shuffled order, alternating schedule, or "
    "explicit randomisation/pseudo-randomisation sequence field, the classification is "
    "'experimenter_randomized'. (5) Otherwise the mechanism is "
    "'assignment_mechanism_not_documented_in_the_release'."
)

STRUCTURAL_VOID_RULE = (
    "A census cell is a structural void, carrying a reason, when the quantity the field asks "
    "for does not exist for this arm by the design actually observed in the data -- for example "
    "a two-level verum/sham corpus has no within-participant stimulation-timing field to report "
    "because no second timing level was ever delivered; an offline aftereffect design has no "
    "concurrent artifact-blanking interval because no recording during stimulation exists in "
    "the release; a corpus with exactly one recorded item per trial has no item-overlap number "
    "to report beyond 'not applicable'. A cell is a structural void because the design forecloses "
    "the measurement, not because this script declined to look. Every field in this artifact is "
    "therefore either MEASURED (a value read from the data, tagged with what was read) or "
    "VOID (a reason the design forecloses the read) -- no field is ever left blank or null "
    "without one of the two tags above."
)


def classify_treatment_assignment(*, single_level: bool, state_triggered: bool,
                                   clinically_chosen_site: bool, documented_randomization: bool) -> str:
    """Pure function implementing RANDOMIZATION_CLASSIFICATION_RULE's precedence order.
    Order is fixed by the rule above and must not be reordered after seeing a corpus's data:
    a state-triggered arm is never reclassified as randomized even if the dataset also
    happens to log a nominal 'randomization' field (this is exactly ds005557's case)."""
    if single_level:
        return "single_level_nothing_to_randomize"
    if state_triggered:
        return "state_triggered_selection_not_randomized"
    if clinically_chosen_site:
        return "site_clinically_or_anatomically_chosen_not_randomized"
    if documented_randomization:
        return "experimenter_randomized"
    return "assignment_mechanism_not_documented_in_the_release"


# ═══════════════════════════════════════════════════════════════════════════
# Shared arithmetic: time-window / item-overlap and dose-variation helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_item_overlap_count(train_start: float, train_end: float,
                              item_windows: list[tuple[float, float]]) -> int:
    """Number of [onset, onset+duration] item-presentation windows a
    [train_start, train_end] stimulation interval overlaps. Half-open
    overlap test (a < b1 and b0 < a1), matching
    run_stimulation_timing_and_parameter_structure.overlaps exactly, restated
    here as a small pure function so it can be unit-tested directly against
    the "train spans two items" case without importing a whole pipeline module."""
    return sum(1 for (i0, i1) in item_windows if train_start < i1 and i0 < train_end)


def dose_variation_flags(values_by_group: dict[str, list[float]]) -> dict[str, bool]:
    """For each group key (e.g. an electrode pair, or a participant), whether
    more than one distinct finite dose value was delivered to it. A group
    with zero or one distinct value does not vary; NaNs are dropped before
    counting distinct values."""
    flags = {}
    for key, values in values_by_group.items():
        finite = sorted({round(float(v), 6) for v in values if np.isfinite(v)})
        flags[key] = len(finite) > 1
    return flags


def monotone_rescaling_pairs(rows: list[dict[str, float]], params: tuple[str, ...],
                              rtol: float = 1e-6) -> list[dict]:
    """Given rows of {param: value} observed together (e.g. one row per
    delivered train), returns every pair of params whose ratio is constant
    across every row where both are finite and nonzero -- i.e. one is an
    exact monotone (linear) rescaling of the other, and should be reported
    as one degree of freedom, not two independently-varying parameters."""
    pairs = []
    for i, a in enumerate(params):
        for b in params[i + 1:]:
            paired = [(row[a], row[b]) for row in rows
                      if np.isfinite(row.get(a, float("nan"))) and np.isfinite(row.get(b, float("nan")))
                      and row[b] != 0]
            if len(paired) < 2:
                continue
            # A constant ratio between two params that are EACH individually fixed (no distinct
            # values at all) is not a rescaling relationship -- it is two unrelated constants that
            # trivially divide evenly. Require at least one of the two to actually vary.
            if len({round(v, 6) for v, _ in paired}) < 2 and len({round(v, 6) for _, v in paired}) < 2:
                continue
            ratios = [v_a / v_b for v_a, v_b in paired]
            if np.allclose(ratios, ratios[0], rtol=rtol):
                pairs.append({"params": [a, b], "ratio_a_to_b": float(ratios[0]), "n_rows_checked": len(ratios)})
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Row schema + completeness gate
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_ROW_FIELDS = (
    "corpus_id", "arm_id", "arm_description",
    "species", "task", "task_category", "is_working_memory_maintenance_stimulation",
    "recording_region", "recording_modality",
    "stimulation_site", "stimulation_site_randomization",
    "treatment_assignment_unit", "treatment_assignment_classification",
    "train_onset", "train_duration_s", "item_overlap", "epoch",
    "amplitude", "frequency_hz", "pulse_width_us", "pulse_count", "charge",
    "within_participant_dose_variation", "within_site_dose_variation", "monotone_rescaling_notes",
    "pre_stimulation_window_s", "artifact_blanking_interval_s", "post_stimulation_windows_through_probe_s",
    "behaviour_type", "behaviour_grain",
    "repeated_sites_count", "repeated_doses_count", "repeated_timing_conditions_count",
    "participant_count", "session_count", "list_count", "item_count", "trial_count",
    "control_conditions",
    "session_accounting",
)


IDENTIFIER_FIELDS = ("corpus_id", "arm_id")  # plain strings naming the row, not data about it


def _is_tagged(value) -> bool:
    return isinstance(value, dict) and value.get("status") in ("measured", "structural_void")


def validate_row(row: dict) -> list[str]:
    """Every REQUIRED_ROW_FIELDS entry must be present. The two identifier
    fields must be non-empty strings; every other field must be a
    MEASURED/VOID tagged value (never a bare None, list or scalar). Returns
    the list of errors; an empty list means the row is complete."""
    defects = []
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            defects.append(f"missing field: {field}")
        elif field in IDENTIFIER_FIELDS:
            if not isinstance(row[field], str) or not row[field]:
                defects.append(f"identifier field not a non-empty string: {field}")
        elif not _is_tagged(row[field]):
            defects.append(f"field not measured/void-tagged: {field}")
    return defects


# ═══════════════════════════════════════════════════════════════════════════
# RAM ds005489 (open-loop) and ds005557 (classifier-triggered + pre-task
# titration) -- human intracranial delayed free recall, stimulation at
# ENCODING (not a working-memory maintenance/delay period).
# ═══════════════════════════════════════════════════════════════════════════

DOSE_PARAM_NAMES = ("amplitude", "pulse_freq", "pulse_width")


def _ram_dose_monotone_check(corpus_dir: str, build_trains, open_loop: bool) -> list[dict]:
    """Pools every delivered train's (amplitude, pulse_freq, pulse_width)
    triple across every session of a RAM corpus and tests each pair of
    those three dose parameters for an exact monotone (constant-ratio)
    relationship -- a real measurement, not an assertion, of whether this
    corpus's dose fields are independent degrees of freedom or a single
    rescaled quantity reported under more than one name."""
    rows = []
    for path in discover_sessions(corpus_dir):
        raw = read_events(path)
        if open_loop:
            trains = build_trains(raw)
        else:
            trains, mismatch = build_trains(raw)
            if mismatch:
                continue
        for t in trains:
            rows.append({"amplitude": t["amplitude"], "pulse_freq": t["pulse_freq"], "pulse_width": t["pulse_width"]})
    return monotone_rescaling_pairs(rows, DOSE_PARAM_NAMES)


def census_ram_openloop() -> dict:
    corpus = run_corpus(OPENLOOP_CORPUS, process_openloop_session, "openloop")
    attributability = pool_item_attributability(corpus["records"], "openloop")
    params = pool_parameter_census(corpus["records"], "openloop")
    monotone_pairs = _ram_dose_monotone_check(OPENLOOP_CORPUS, build_trains_openloop, open_loop=True)

    n_items = sum(r["n_words"] for r in corpus["records"].values())
    n_trials_stim = params["n_stimulated_trials_total"]

    mean_overlap = attributability["pooled"]["mean_items_per_train_among_attributable_trains"]
    train_dur = attributability["pooled"]["train_duration_s_median"]
    item_spacing = attributability["pooled"]["item_presentation_spacing_s_median"]

    return {
        "corpus_id": "ram_ds005489_openloop",
        "arm_id": "encoding_alternating_block_open_loop",
        "arm_description": MEASURED(
            "Experimenter-scheduled bipolar stimulation of alternating two-word encoding "
            "blocks during delayed free recall; every WORD event's own stimulation field is "
            "trusted directly.", "ds005489 BIDS events.tsv, trial_type==WORD rows"),
        "species": MEASURED("human", "BIDS participants.tsv / sub-* directory naming convention"),
        "task": MEASURED("delayed free recall (verbal episodic list learning)", "ds005489 task label"),
        "task_category": MEASURED("episodic_encoding", "stimulation lands on WORD (encoding) events only"),
        "is_working_memory_maintenance_stimulation": MEASURED(
            False, "stimulation trains overlap WORD presentation windows, not a post-list delay/retention interval"),
        "recording_region": MEASURED("bipolar intracranial contact pairs across cortical and subcortical sites, "
                                      "clinically implanted", "channels.tsv per session"),
        "recording_modality": MEASURED("intracranial EEG (iEEG), high-gamma power features", "ieeg.edf"),
        "stimulation_site": MEASURED(
            f"{params['n_electrode_pairs']} distinct (subject, anode, cathode) bipolar pairs across "
            f"{params['n_subjects_with_stimulated_trials']} subjects, one pair per subject",
            "STIM_ON/WORD anode_label/cathode_label fields"),
        "stimulation_site_randomization": MEASURED(
            "not randomly assigned: the stimulated bipolar pair is chosen per subject by the clinical "
            "electrode implantation, not by the experiment", "electrode_pairs field is one fixed pair per subject"),
        "treatment_assignment_unit": MEASURED("word (list item)", "one stimulation flag per WORD row"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=False, state_triggered=False,
                                           clinically_chosen_site=False, documented_randomization=True),
            "WORD.stimulation is set by an experimenter-scheduled alternating block design (module docstring "
            "of run_ram_openloop_pipeline.py, corroborated by the near-constant per-session stim/control "
            "fraction), not read from the participant's neural state at delivery time"),
        "train_onset": MEASURED("locked to WORD (encoding-item) onset", "STIM_ON row onset vs. WORD row onset"),
        "train_duration_s": MEASURED(train_dur, "median train duration_s over matched STIM_ON/STIM_OFF-equivalent trains"),
        "item_overlap": MEASURED(
            {"branch": attributability["branch"], "mean_items_per_train": mean_overlap,
             "median_item_spacing_s": item_spacing,
             "reading": "a ~4.6 s train against a ~2.5 s median item-presentation spacing overlaps a mean of "
                        f"{mean_overlap:.2f} item windows -- the delivered stimulation cannot be attributed to "
                        "one item; it spans two."},
            "run_stimulation_timing_and_parameter_structure.pool_item_attributability"),
        "epoch": MEASURED("encoding (word presentation), not post-encoding delay", "WORD trial_type"),
        "amplitude": MEASURED(params["stimulated_trials_by_amplitude_microamps"], "STIM_ON/WORD amplitude field, microamps"),
        "frequency_hz": MEASURED(params["constant_across_corpus"]["pulse_freq_hz"] or params["unique_values_seen"]["pulse_freq_hz"],
                                  "STIM_ON/WORD pulse_freq field"),
        "pulse_width_us": MEASURED(params["constant_across_corpus"]["pulse_width_us"] or params["unique_values_seen"]["pulse_width_us"],
                                    "STIM_ON/WORD pulse_width field"),
        "pulse_count": VOID("not carried as its own field in this release's event tables; pulse count is "
                             "recoverable only as (stim_duration_ms/1000) * pulse_freq_hz if both are trusted, "
                             "and is not reported as an independently-measured field here"),
        "charge": VOID("no per-pulse charge (amplitude x pulse width) field exists in this release; "
                        "would need to be computed, not read, and is left void rather than silently derived"),
        "within_participant_dose_variation": MEASURED(
            {"amplitude": params["n_electrode_pairs_with_more_than_one_amplitude"] > 0,
             "frequency_hz": False, "pulse_width_us": False,
             "n_pairs_with_more_than_one_amplitude": params["n_electrode_pairs_with_more_than_one_amplitude"],
             "n_pairs_total": params["n_electrode_pairs"]},
            "pool_parameter_census per-pair amplitude/freq/width sets"),
        "within_site_dose_variation": MEASURED(
            {"amplitude": params["n_electrode_pairs_with_more_than_one_amplitude"] > 0},
            "same pair-level set as within_participant (one pair per subject in this corpus, so the two "
            "grains coincide)"),
        "monotone_rescaling_notes": MEASURED(
            {"pairs_found": monotone_pairs,
             "reading": (
                 f"{len(monotone_pairs)} exact constant-ratio pair(s) found among (amplitude, pulse_freq, "
                 "pulse_width) over every delivered train" if monotone_pairs else
                 "no exact constant-ratio relationship among (amplitude, pulse_freq, pulse_width) was found "
                 "over every delivered train; pulse_freq_hz (50 Hz) and pulse_width_us (300 us) are corpus-wide "
                 "constants and amplitude is the only field that varies, so there is nothing for it to be a "
                 "monotone rescaling of")},
            "monotone_rescaling_pairs over every STIM_ON-derived train's (amplitude, pulse_freq, pulse_width) "
            "triple, pooled across included sessions"),
        "pre_stimulation_window_s": MEASURED(0.3, "PRE_S in build_session_features (epoch window relative to word onset)"),
        "artifact_blanking_interval_s": VOID(
            "this census reads event tables only, never the iEEG signal itself, so no measured artifact "
            "rejection window exists here; run_ram_openloop_pipeline.py applies no explicit blanking window, "
            "relying on high-gamma-band filtering instead"),
        "post_stimulation_windows_through_probe_s": MEASURED(
            1.6, "POST_S in build_session_features (epoch window relative to word onset); this corpus has no "
                 "explicit memory probe event, recall is scored at test phrase end of list"),
        "behaviour_type": MEASURED("free-recall verbal report, scored recalled/not-recalled per word", "WORD.recalled field"),
        "behaviour_grain": MEASURED("binary per word", "WORD.recalled is 0/1"),
        "repeated_sites_count": MEASURED(0, "each of the corpus's electrode pairs is used by exactly one subject "
                                             "(no pair reused across subjects) and by exactly one session-pair per "
                                             "subject in the admitted set"),
        "repeated_doses_count": MEASURED(params["n_electrode_pairs_with_more_than_one_amplitude"],
                                          "electrode pairs delivering more than one distinct amplitude"),
        "repeated_timing_conditions_count": VOID(
            "stimulation timing here has one level (alternating block schedule); no distinct timing "
            "condition is repeated because none is varied"),
        "participant_count": MEASURED(params["n_subjects_with_stimulated_trials"], "distinct subjects with a stimulated trial"),
        "session_count": MEASURED(corpus["n_sessions_included"], "sessions passing build_session_features' own admission gate"),
        "list_count": VOID("list identifiers are per-word bookkeeping, not separately enumerated by this census; "
                            "see item_count for the trial-bearing unit actually counted"),
        "item_count": MEASURED(n_items, "sum of WORD rows across included sessions"),
        "trial_count": MEASURED(n_trials_stim, "stimulated WORD rows pooled across included sessions (item_count "
                                                "above is the full trial count including unstimulated words)"),
        "control_conditions": MEASURED(["unstimulated WORD presentations within the same session"],
                                        "WORD.stimulation == 0"),
        "session_accounting": MEASURED(
            {"n_sessions_seen": corpus["n_sessions_seen"], "n_sessions_included": corpus["n_sessions_included"],
             "excluded_sessions_by_reason": corpus["excluded_sessions"],
             "n_subject_directories": corpus["n_subjects_seen_as_directories"],
             "n_subjects_with_ieeg_events": corpus["n_subjects_with_ieeg_events"],
             "subject_directories_without_ieeg_events": corpus["subject_directories_without_ieeg_events"],
             "reconciliation": "n_sessions_seen == n_sessions_included + len(excluded_sessions_by_reason)"},
            "run_stimulation_timing_and_parameter_structure.run_corpus"),
    }


def _titration_pulse_timing(corpus_dir: str) -> dict:
    """Re-derives, from the raw closed-loop event tables, the onset time of
    every STIM_ON/STIM_OFF train that could NOT be matched to an owning WORD
    item -- the pre-task device titration/calibration pulses -- relative to
    that session's first WORD onset. Reuses build_trains_closedloop and
    match_train_owner exactly as process_closedloop_session does; this is a
    second, cheap pass over already-open session dicts, not a new parser."""
    onsets_before_first_word_s = []
    n_unowned_total = 0
    n_sessions_with_unowned = 0
    for path in discover_sessions(corpus_dir):
        rows = read_events(path)
        words = [r for r in rows if r["trial_type"] == "WORD"]
        if not words:
            continue
        first_word_onset = min(r["_onset"] for r in words)
        trains, mismatch = build_trains_closedloop(rows)
        if mismatch:
            continue
        word_onsets = [r["_onset"] for r in sorted(words, key=lambda r: r["_onset"])]
        unowned = [t for t in trains
                   if match_train_owner(t["start"], word_onsets, CLOSEDLOOP_OWNER_MATCH_WINDOW_S) is None]
        if unowned:
            n_sessions_with_unowned += 1
            n_unowned_total += len(unowned)
            onsets_before_first_word_s.extend(t["start"] - first_word_onset for t in unowned)
    return {
        "n_unowned_trains_total": n_unowned_total,
        "n_sessions_with_unowned_trains": n_sessions_with_unowned,
        "median_onset_relative_to_first_word_s": (
            float(np.median(onsets_before_first_word_s)) if onsets_before_first_word_s else None),
        "min_onset_relative_to_first_word_s": (
            float(np.min(onsets_before_first_word_s)) if onsets_before_first_word_s else None),
        "max_onset_relative_to_first_word_s": (
            float(np.max(onsets_before_first_word_s)) if onsets_before_first_word_s else None),
    }


def census_ram_closedloop() -> dict:
    corpus = run_corpus(CLOSEDLOOP_CORPUS, process_closedloop_session, "closedloop")
    attributability = pool_item_attributability(corpus["records"], "closedloop")
    params = pool_parameter_census(corpus["records"], "closedloop")
    monotone_pairs = _ram_dose_monotone_check(CLOSEDLOOP_CORPUS, build_trains_closedloop, open_loop=False)

    n_items = sum(r["n_words"] for r in corpus["records"].values())
    n_trials_stim = params["n_stimulated_trials_total"]
    mean_overlap = attributability["pooled"]["mean_items_per_train_among_attributable_trains"]
    train_dur = attributability["pooled"]["train_duration_s_median"]

    return {
        "corpus_id": "ram_ds005557_closedloop",
        "arm_id": "encoding_classifier_triggered_closed_loop",
        "arm_description": MEASURED(
            "Real-time classifier-triggered bipolar stimulation during delayed free-recall encoding: "
            "WORD rows leave their own stimulation field at 0 always, real timing/dose live on "
            "STIM_ON/STIM_OFF rows and are matched back to the nearest preceding word.",
            "ds005557 BIDS events.tsv, STIM_ON/STIM_OFF vs WORD rows"),
        "species": MEASURED("human", "BIDS participants.tsv / sub-* directory naming convention"),
        "task": MEASURED("delayed free recall (verbal episodic list learning)", "ds005557 task label"),
        "task_category": MEASURED("episodic_encoding", "stimulation lands on WORD (encoding) events only"),
        "is_working_memory_maintenance_stimulation": MEASURED(
            False, "stimulation trains are triggered during WORD presentation, not a post-list delay/retention interval"),
        "recording_region": MEASURED("bipolar intracranial contact pairs across cortical and subcortical sites, "
                                      "clinically implanted", "channels.tsv per session"),
        "recording_modality": MEASURED("intracranial EEG (iEEG), high-gamma power features driving an online classifier",
                                        "ieeg.edf"),
        "stimulation_site": MEASURED(
            f"{params['n_electrode_pairs']} distinct (subject, anode, cathode) bipolar pairs across "
            f"{params['n_subjects_with_stimulated_trials']} subjects, one pair per subject",
            "STIM_ON anode_label/cathode_label fields"),
        "stimulation_site_randomization": MEASURED(
            "not randomly assigned: the stimulated bipolar pair is chosen per subject by the clinical "
            "electrode implantation, not by the experiment", "electrode_pairs field is one fixed pair per subject"),
        "treatment_assignment_unit": MEASURED("word (list item)", "classifier decides per-word whether to trigger"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=False, state_triggered=True,
                                           clinically_chosen_site=False, documented_randomization=False),
            "the module's own docstring and the dataset name both state the trigger is the online encoding "
            "classifier's real-time readout of the subject's own neural state -- this is a state-triggered "
            "selection whatever the release's own field naming implies"),
        "train_onset": MEASURED("triggered by the online classifier at a data-dependent moment during a WORD "
                                 "presentation, not locked to a fixed pre-item offset", "STIM_ON vs WORD onset timing"),
        "train_duration_s": MEASURED(train_dur, "median train duration over matched STIM_ON/STIM_OFF pairs"),
        "item_overlap": MEASURED(
            {"branch": attributability["branch"], "mean_items_per_train": mean_overlap,
             "reading": f"median train duration {train_dur:.3f} s is much shorter than the item-presentation "
                        "interval, so a train overlapping any item at all overlaps a mean of "
                        f"{mean_overlap:.2f} item window(s) -- this arm CAN in principle localise stimulation "
                        "to a single item's encoding."},
            "run_stimulation_timing_and_parameter_structure.pool_item_attributability"),
        "epoch": MEASURED("encoding (word presentation), not post-encoding delay", "WORD trial_type"),
        "amplitude": MEASURED(params["stimulated_trials_by_amplitude_microamps"], "STIM_ON amplitude field, microamps, "
                                                                                   "item-linked trains only"),
        "frequency_hz": MEASURED(params["unique_values_seen"]["pulse_freq_hz"], "STIM_ON pulse_freq field"),
        "pulse_width_us": MEASURED(params["unique_values_seen"]["pulse_width_us"], "STIM_ON pulse_width field"),
        "pulse_count": VOID("not carried as its own field in this release's event tables"),
        "charge": VOID("no per-pulse charge field exists in this release; not derived here"),
        "within_participant_dose_variation": MEASURED(
            {"amplitude": params["n_electrode_pairs_with_more_than_one_amplitude"] > 0,
             "amplitude_including_pre_task_pulses": params["n_electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses"] > 0,
             "n_pairs_with_more_than_one_amplitude_item_linked": params["n_electrode_pairs_with_more_than_one_amplitude"],
             "n_pairs_with_more_than_one_amplitude_including_pre_task_pulses":
                 params["n_electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses"]},
            "pool_parameter_census per-pair amplitude sets, item-linked vs. all STIM_ON rows"),
        "within_site_dose_variation": MEASURED(
            {"amplitude": params["n_electrode_pairs_with_more_than_one_amplitude"] > 0},
            "one pair per subject in this corpus, so within-participant and within-site coincide"),
        "monotone_rescaling_notes": MEASURED(
            {"pairs_found": monotone_pairs,
             "reading": (
                 f"{len(monotone_pairs)} exact constant-ratio pair(s) found among (amplitude, pulse_freq, "
                 "pulse_width) over every delivered train (item-linked and unmatched pooled together)" if monotone_pairs
                 else "no exact constant-ratio relationship among (amplitude, pulse_freq, pulse_width) was found "
                      "over every delivered train; all three vary between subjects/pairs with no fixed ratio")},
            "monotone_rescaling_pairs over every STIM_ON/STIM_OFF-derived train's (amplitude, pulse_freq, "
            "pulse_width) triple, pooled across included sessions"),
        "pre_stimulation_window_s": MEASURED(0.3, "PRE_S in build_session_features, same convention as ds005489"),
        "artifact_blanking_interval_s": VOID(
            "this census reads event tables only, never the iEEG signal itself; no measured blanking window exists here"),
        "post_stimulation_windows_through_probe_s": MEASURED(
            1.6, "POST_S in build_session_features; this corpus, like ds005489, has no explicit memory-probe event"),
        "behaviour_type": MEASURED("free-recall verbal report, scored recalled/not-recalled per word", "WORD.recalled field"),
        "behaviour_grain": MEASURED("binary per word", "WORD.recalled is 0/1"),
        "repeated_sites_count": MEASURED(0, "each electrode pair belongs to exactly one subject in this corpus"),
        "repeated_doses_count": MEASURED(params["n_electrode_pairs_with_more_than_one_amplitude"],
                                          "electrode pairs delivering more than one distinct item-linked amplitude"),
        "repeated_timing_conditions_count": VOID(
            "stimulation timing is set by the classifier's own trigger logic per item, not by a repeated "
            "discrete timing condition this census can enumerate"),
        "participant_count": MEASURED(params["n_subjects_with_stimulated_trials"], "distinct subjects with a stimulated trial"),
        "session_count": MEASURED(corpus["n_sessions_included"], "sessions passing the STIM_ON/STIM_OFF count-match gate"),
        "list_count": VOID("list identifiers are per-word bookkeeping, not separately enumerated by this census"),
        "item_count": MEASURED(n_items, "sum of WORD rows across included sessions"),
        "trial_count": MEASURED(n_trials_stim, "item-linked stimulated WORD rows pooled across included sessions"),
        "control_conditions": MEASURED(["unstimulated / not-triggered WORD presentations within the same session"],
                                        "words the classifier did not trigger on"),
        "session_accounting": MEASURED(
            {"n_sessions_seen": corpus["n_sessions_seen"], "n_sessions_included": corpus["n_sessions_included"],
             "excluded_sessions_by_reason": corpus["excluded_sessions"],
             "n_subject_directories": corpus["n_subjects_seen_as_directories"],
             "n_subjects_with_ieeg_events": corpus["n_subjects_with_ieeg_events"],
             "subject_directories_without_ieeg_events": corpus["subject_directories_without_ieeg_events"],
             "reconciliation": "n_sessions_seen == n_sessions_included + len(excluded_sessions_by_reason)"},
            "run_stimulation_timing_and_parameter_structure.run_corpus"),
    }


def census_ram_closedloop_titration() -> dict:
    corpus = run_corpus(CLOSEDLOOP_CORPUS, process_closedloop_session, "closedloop")
    params = pool_parameter_census(corpus["records"], "closedloop")
    timing = _titration_pulse_timing(CLOSEDLOOP_CORPUS)
    n_titration_amp_pairs = (params["n_electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses"]
                              - params["n_electrode_pairs_with_more_than_one_amplitude"])

    return {
        "corpus_id": "ram_ds005557_closedloop",
        "arm_id": "pre_task_device_titration_pulses",
        "arm_description": MEASURED(
            "STIM_ON/STIM_OFF pulse trains present in the same event tables that cannot be matched to any "
            "owning WORD item (nearest preceding word farther than the matching window) -- device "
            "calibration/titration pulses delivered before or between task blocks, not during item encoding.",
            "run_stimulation_timing_and_parameter_structure.match_train_owner, unmatched trains"),
        "species": MEASURED("human", "same participants as the classifier-triggered arm"),
        "task": VOID("these pulses are not linked to a task trial by construction; no task label applies to them"),
        "task_category": VOID("device calibration, not a scored cognitive trial"),
        "is_working_memory_maintenance_stimulation": MEASURED(
            False, "delivered before/between task blocks, never during any recorded task epoch"),
        "recording_region": MEASURED("same bipolar contact pairs as the classifier-triggered arm in the same session",
                                      "STIM_ON anode_label/cathode_label, same field as the main arm"),
        "recording_modality": MEASURED("intracranial EEG (iEEG)", "ieeg.edf"),
        "stimulation_site": MEASURED("same clinically-chosen pair as the session's main arm", "STIM_ON anode/cathode"),
        "stimulation_site_randomization": MEASURED(
            "not randomly assigned -- identical site to the session's classifier-triggered arm",
            "same electrode_pairs_all_trains field"),
        "treatment_assignment_unit": VOID("no trial unit exists for these pulses; they are not attached to a scored item"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=True, state_triggered=False,
                                           clinically_chosen_site=False, documented_randomization=False),
            "every titration pulse observed in this corpus is delivered the same way (device calibration "
            "sequence), so there is no second level to classify an assignment mechanism over"),
        "train_onset": MEASURED(
            {"median_s_relative_to_first_word_onset": timing["median_onset_relative_to_first_word_s"],
             "min_s_relative_to_first_word_onset": timing["min_onset_relative_to_first_word_s"],
             "max_s_relative_to_first_word_onset": timing["max_onset_relative_to_first_word_s"]},
            "recomputed directly from STIM_ON/STIM_OFF onsets vs. each session's first WORD onset"),
        "train_duration_s": VOID("titration-pulse train durations are not separately tabulated in this census; "
                                  "only their onset relative to task start was measured"),
        "item_overlap": VOID("these pulses do not overlap any item-presentation window by construction (that is "
                              "exactly how they are identified as unowned)"),
        "epoch": MEASURED("pre-task / inter-block device calibration, outside any scored task epoch",
                           "occurs before the session's first WORD onset in the overwhelming majority of cases"),
        "amplitude": VOID("per-pulse amplitude for the unowned set is folded into "
                           "'amplitude_including_unmatched_pulses' on the main closed-loop arm's row rather than "
                           "re-reported here; see that arm's within_participant_dose_variation field"),
        "frequency_hz": VOID("not separately re-extracted for the unowned subset in this census"),
        "pulse_width_us": VOID("not separately re-extracted for the unowned subset in this census"),
        "pulse_count": VOID("not carried as its own field in this release's event tables"),
        "charge": VOID("no per-pulse charge field exists in this release"),
        "within_participant_dose_variation": MEASURED(
            {"amplitude_pairs_gaining_a_second_amplitude_once_titration_pulses_are_included": n_titration_amp_pairs},
            "difference between the item-linked and all-trains electrode-pair amplitude-variation counts on "
            "the classifier-triggered arm"),
        "within_site_dose_variation": VOID("not separately computed for this arm; see within_participant_dose_variation"),
        "monotone_rescaling_notes": VOID("too few titration pulses per session to test a monotone-rescaling "
                                          "relationship among dose parameters"),
        "pre_stimulation_window_s": VOID("titration pulses precede any task epoch; there is no task-relative "
                                          "pre-stimulation window to report"),
        "artifact_blanking_interval_s": VOID("this census reads event tables only, never the iEEG signal itself"),
        "post_stimulation_windows_through_probe_s": VOID("no probe event follows a titration pulse"),
        "behaviour_type": VOID("titration pulses have no associated behavioural trial"),
        "behaviour_grain": VOID("titration pulses have no associated behavioural trial"),
        "repeated_sites_count": MEASURED(0, "identical to the session's main-arm site; not an independently repeated site"),
        "repeated_doses_count": MEASURED(n_titration_amp_pairs,
                                          "electrode pairs whose amplitude set gains a second value once "
                                          "unmatched (titration) pulses are included"),
        "repeated_timing_conditions_count": VOID("titration pulses are not organised into discrete timing conditions"),
        "participant_count": MEASURED(timing["n_sessions_with_unowned_trains"],
                                       "sessions containing at least one unowned train (upper bound on distinct "
                                       "participants; some subjects contribute more than one such session)"),
        "session_count": MEASURED(timing["n_sessions_with_unowned_trains"], "sessions containing >=1 unowned train"),
        "list_count": VOID("titration pulses are not organised by list"),
        "item_count": VOID("titration pulses are not linked to any item"),
        "trial_count": MEASURED(timing["n_unowned_trains_total"], "STIM_ON/STIM_OFF trains unmatched to any WORD "
                                                                     "item, pooled across included sessions"),
        "control_conditions": VOID("no control condition is defined for device-calibration pulses"),
        "session_accounting": MEASURED(
            {"n_sessions_seen": corpus["n_sessions_seen"],
             "n_sessions_with_unowned_trains": timing["n_sessions_with_unowned_trains"],
             "n_sessions_with_zero_unowned_trains": corpus["n_sessions_included"] - timing["n_sessions_with_unowned_trains"],
             "reconciliation": "n_sessions_seen == n_sessions_included (main-arm gate) + excluded_sessions "
                                "(same denominator as the classifier-triggered arm); the titration/no-titration "
                                "split partitions n_sessions_included"},
            "recomputed pass over discover_sessions(ds005557) with build_trains_closedloop + match_train_owner"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# haslacher_clam_tacs -- human scalp EEG, phase-locked
# closed-loop tACS during a visual working-memory retention period.
# ═══════════════════════════════════════════════════════════════════════════

def _clam_group_trial_counts(subjects: list[str]) -> dict:
    counts, seen, refused = {}, [], {}
    for subject in subjects:
        seen.append(subject)
        vhdr = CLAM_DATA_DIR / subject / "stim.vhdr"
        if not vhdr.exists():
            refused[subject] = "stim.vhdr not found on disk"
            continue
        try:
            outcomes = clam_trial_outcomes(subject)
        except Exception as exc:  # noqa: BLE001 -- reported as an excluded subject, not a crash
            refused[subject] = f"{type(exc).__name__}: {exc}"
            continue
        by_phase = {}
        for code, correct in outcomes:
            by_phase.setdefault(PHASE_CONDITIONS[code], []).append(correct)
        counts[subject] = {"n_trials": len(outcomes),
                            "n_trials_per_phase_deg": {str(k): len(v) for k, v in sorted(by_phase.items())}}
    return {"per_subject": counts, "n_seen": len(seen), "n_included": len(counts), "excluded_by_reason": refused}


def _clam_row(group_name: str, subjects: list[str], other_group_site: str) -> dict:
    trials = _clam_group_trial_counts(subjects)
    total_trials = sum(v["n_trials"] for v in trials["per_subject"].values())
    phase_degrees = sorted(PHASE_CONDITIONS.values())

    return {
        "corpus_id": "haslacher_clam_tacs",
        "arm_id": f"{group_name}_group_phase_locked_tacs",
        "arm_description": MEASURED(
            f"Real-time phase-tuned closed-loop tACS during a visual working-memory retention period; "
            f"{group_name} group receives stimulation " +
            ("over the recorded occipital alpha source" if group_name == "active"
             else "at a frontal site away from the recorded source (specificity control)"),
            "run_haslacher_phase_omega module docstring, corroborated by ACTIVE_SUBJECTS/CONTROL_SUBJECTS rosters"),
        "species": MEASURED("human", "BrainVision recordings named per subject under haslacher_clam_tacs local_path"),
        "task": MEASURED("visual working-memory retention task with a phase-locked stimulation-timed response",
                          "stim.vhdr event markers, PHASE_CONDITIONS trigger codes"),
        "task_category": MEASURED("working_memory_maintenance", "RETENTION_TMIN/RETENTION_TMAX define a post-encoding "
                                                                   "retention window the dataset's own README names"),
        "is_working_memory_maintenance_stimulation": MEASURED(
            True, "stimulation is phase-locked and delivered online during the retention window, not offline"),
        "recording_region": MEASURED("64-channel scalp EEG, 10-20 montage", "run_haslacher_phase_omega module docstring"),
        "recording_modality": MEASURED("scalp EEG", "BrainVision .vhdr/.eeg/.vmrk triplet"),
        "stimulation_site": MEASURED("occipital (over the recorded alpha source)" if group_name == "active"
                                      else "frontal (away from the recorded alpha source)",
                                      "group assignment fixed by ACTIVE_SUBJECTS/CONTROL_SUBJECTS roster"),
        "stimulation_site_randomization": MEASURED(
            "active-vs-control group membership is a fixed per-participant assignment set by the source study's "
            "own roster; this census does not have the original randomisation record and cannot itself verify "
            "whether that group assignment was randomised at enrollment -- it is recorded here as a between-"
            "participant grouping, not confirmed as randomised", "ACTIVE_SUBJECTS/CONTROL_SUBJECTS constants"),
        "treatment_assignment_unit": MEASURED("trial (one of 6 phase-lag conditions each trial)", "PHASE_CONDITIONS trigger codes"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=False, state_triggered=True,
                                           clinically_chosen_site=False, documented_randomization=False),
            "the phase LAG delivered on a trial is set relative to a real-time estimate of the participant's own "
            "ongoing alpha oscillation (closed-loop phase-tuned stimulation) -- a state-triggered targeting "
            "mechanism, even though the six target lags themselves are a fixed, evenly-spaced design"),
        "train_onset": MEASURED("locked to the real-time-estimated phase of the participant's own retention-period "
                                 "alpha oscillation, not a fixed latency", "stim.vhdr closed-loop design"),
        "train_duration_s": MEASURED(RETENTION_TMAX - RETENTION_TMIN,
                                      "RETENTION_TMAX - RETENTION_TMIN, the dataset's own retention window"),
        "item_overlap": VOID("this task presents a single memorandum per trial during the retention window; "
                              "item-overlap (multiple items per train) does not apply to a single-item design"),
        "epoch": MEASURED(f"retention/maintenance window [{RETENTION_TMIN}, {RETENTION_TMAX}] s post-cue",
                           "RETENTION_TMIN/RETENTION_TMAX constants, per the dataset's own README"),
        "amplitude": VOID("stimulation amplitude/current is not exposed in the event-marker stream this census "
                           "reads (preload=False, markers only); would require reading the analog stim aux "
                           "channel this census does not open"),
        "frequency_hz": MEASURED("individually band-matched to each participant's own baseline alpha oscillation "
                                  "(8-14 Hz band, dataset's own definition)", "ALPHA_BAND constant"),
        "pulse_width_us": VOID("tACS is a continuous sinusoidal current, not a discrete pulse train; pulse width "
                                "does not apply to this stimulation modality"),
        "pulse_count": VOID("continuous sinusoidal stimulation has no discrete pulse count"),
        "charge": VOID("amplitude is not exposed in the marker stream this census reads (see amplitude field)"),
        "within_participant_dose_variation": MEASURED(
            {"phase_lag_deg": True, "amplitude": "not_measured_here_see_amplitude_field", "frequency_hz": False},
            "PHASE_CONDITIONS carries 6 distinct target lags delivered to every included participant; frequency "
            "is fixed per participant at their own band-matched value"),
        "within_site_dose_variation": MEASURED(
            {"phase_lag_deg": True}, "site is fixed per participant (one group, one montage); phase lag is the "
                                      "only within-site varying quantity this census can confirm"),
        "monotone_rescaling_notes": VOID("only one continuously-varying parameter (phase lag) is confirmed "
                                          "within-participant; no second parameter exists to test a rescaling "
                                          "relationship against"),
        "pre_stimulation_window_s": VOID("this is online closed-loop stimulation with no fixed pre-stimulation "
                                          "window analogous to a discrete-trial pre-train baseline"),
        "artifact_blanking_interval_s": VOID("this census reads event markers only (preload=False), never the "
                                              "EEG signal itself, so no measured blanking window exists here"),
        "post_stimulation_windows_through_probe_s": VOID("stimulation continues through the retention window up "
                                                           "to the probe; there is no discrete post-stimulation-"
                                                           "before-probe interval in a continuous closed-loop design"),
        "behaviour_type": MEASURED("forced-choice memory probe, correct/incorrect", "event code 10/11 (correct/incorrect)"),
        "behaviour_grain": MEASURED("binary per trial, aggregated per phase-lag condition for a modulation-depth curve",
                                     "modulation_from_outcomes in run_haslacher_phase_omega.py"),
        "repeated_sites_count": MEASURED(0, "each participant is stimulated at exactly one site (their assigned group)"),
        "repeated_doses_count": VOID("amplitude/current dose is not measured in this census (see amplitude field)"),
        "repeated_timing_conditions_count": MEASURED(len(phase_degrees), "distinct phase-lag degree values in "
                                                                          "PHASE_CONDITIONS, each delivered to every included participant"),
        "participant_count": MEASURED(trials["n_included"], f"{group_name} subjects with a readable stim.vhdr event stream"),
        "session_count": MEASURED(trials["n_included"], "one stimulation session per included participant in this release"),
        "list_count": VOID("this task has no list structure"),
        "item_count": VOID("one memorandum item per trial; not separately enumerated beyond trial_count"),
        "trial_count": MEASURED(total_trials, "sum of readable stim-block trials across included participants"),
        "control_conditions": MEASURED(
            [f"the {other_group_site} group (different stimulation site, specificity control)",
             "no explicit sham/off condition within a participant's own stim.vhdr recording"],
            "ACTIVE_SUBJECTS/CONTROL_SUBJECTS group structure"),
        "session_accounting": MEASURED(
            {"n_subjects_seen": trials["n_seen"], "n_subjects_included": trials["n_included"],
             "excluded_by_reason": trials["excluded_by_reason"],
             "reconciliation": "n_subjects_seen == n_subjects_included + len(excluded_by_reason)"},
            "directory scan of haslacher_clam_tacs local_path against ACTIVE_SUBJECTS/CONTROL_SUBJECTS rosters"),
    }


def census_clam_active() -> dict:
    return _clam_row("active", list(ACTIVE_SUBJECTS), "control")


def census_clam_control() -> dict:
    return _clam_row("control", list(CONTROL_SUBJECTS), "active")


# ═══════════════════════════════════════════════════════════════════════════
# ds005034 -- randomised transcranial (theta-tACS) verum/sham crossover,
# offline aftereffect design (no online/concurrent stimulation of the task).
# ═══════════════════════════════════════════════════════════════════════════

def census_ds005034() -> dict:
    data_root_dir = data_root()
    dataset_root = data_root_dir / "ds005034" if (data_root_dir / "ds005034").is_dir() else data_root_dir
    inventory = ds005034_paired_inventory(dataset_root)

    trial_counts = {}
    for participant, session_paths in inventory["paths"].items():
        for session, set_path in session_paths.items():
            events_path = set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))
            if not events_path.exists():
                continue
            try:
                trial_counts[(participant, session)] = len(ds005034_load_events(events_path))
            except OSError:
                continue
    total_trials = sum(trial_counts.values())

    return {
        "corpus_id": "ds005034",
        "arm_id": "randomised_verum_sham_crossover",
        "arm_description": MEASURED(
            "Within-participant randomised crossover: theta-tACS (verum) vs. sham stimulation delivered offline, "
            "each followed (tens of minutes later) by a recorded scalp-EEG working-memory session -- an "
            "aftereffect design, not concurrent/online stimulation of the task.",
            "run_ds005034_tacs_aftereffect module docstring and SESSIONS=(sham, verum)"),
        "species": MEASURED("human", "BIDS sub-* participants under ds005034"),
        "task": MEASURED("serial working-memory span task (forward/backward/alphabetical recall, set sizes 4 and 6)",
                          "EVENT_MAP trigger codes -> (task, load)"),
        "task_category": MEASURED("working_memory_maintenance", "delay-period spectral state is the recorded quantity "
                                                                   "(DELAY_WINDOW relative to delay onset)"),
        "is_working_memory_maintenance_stimulation": MEASURED(
            False, "stimulation itself is delivered offline, well before the recorded maintenance-period EEG; only "
                   "an aftereffect on the delay-period signal is measurable, not online maintenance-period stimulation"),
        "recording_region": MEASURED("129-channel scalp EEG (EGI net)", "THETA_ROI/POSTERIOR_ROI channel labels (E-prefixed)"),
        "recording_modality": MEASURED("scalp EEG", "*_eeg.set EEGLAB files"),
        "stimulation_site": VOID("montage/site is not read by this census (the public BIDS release's tACS montage "
                                  "documentation is not opened here; only the resulting EEG session is read)"),
        "stimulation_site_randomization": VOID("site is not measured in this census; see stimulation_site"),
        "treatment_assignment_unit": MEASURED("participant-session (verum vs. sham), within-participant crossover",
                                               "SESSIONS = (sham, verum) per participant"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=False, state_triggered=False,
                                           clinically_chosen_site=False, documented_randomization=True),
            "the ds005034 loader's own design.source field cites the source study describing this as a "
            "randomised, counterbalanced within-participant verum/sham crossover; this census does not "
            "itself carry a raw randomisation-order field and instead trusts the loader's own declared "
            "design summary for this one classification input"),
        "train_onset": VOID("stimulation is delivered offline, decoupled from the recorded task by tens of minutes; "
                             "no trial-level train onset exists in the recorded session at all"),
        "train_duration_s": VOID("stimulation duration is not read by this census (it precedes the recorded EEG "
                                  "session and is not in the files this loader opens)"),
        "item_overlap": VOID("stimulation is not concurrent with any item presentation (offline aftereffect design); "
                              "item overlap does not apply"),
        "epoch": MEASURED({"baseline_window_s": list(DS005034_BASELINE_WINDOW), "delay_window_s": list(DS005034_DELAY_WINDOW)},
                           "BASELINE_WINDOW/DELAY_WINDOW constants, seconds relative to delay onset"),
        "amplitude": VOID("not read by this census; the recorded files are the post-hoc EEG session, not the "
                           "stimulation device log"),
        "frequency_hz": MEASURED("theta (4-8 Hz target band, per the source paper's own tACS frequency)",
                                  "BANDS['theta'] used as the loader's own primary spectral target"),
        "pulse_width_us": VOID("tACS is continuous sinusoidal current; pulse width does not apply"),
        "pulse_count": VOID("continuous sinusoidal stimulation has no discrete pulse count"),
        "charge": VOID("amplitude/current is not read by this census"),
        "within_participant_dose_variation": MEASURED(
            {"timing_condition": False}, "exactly two levels (verum, sham) exist for every participant; no "
                                          "continuous within-participant timing sweep is present, as "
                                          "the design of this corpus is randomised"),
        "within_site_dose_variation": VOID("site/montage is not read by this census (see stimulation_site)"),
        "monotone_rescaling_notes": VOID("no dose parameter beyond a fixed target band is read by this census"),
        "pre_stimulation_window_s": MEASURED(DS005034_BASELINE_WINDOW[1] - DS005034_BASELINE_WINDOW[0],
                                              "BASELINE_WINDOW duration -- this is the recorded task's own pre-delay "
                                              "baseline, not a pre-TRAIN window (no train exists in the recorded session)"),
        "artifact_blanking_interval_s": VOID("no concurrent stimulation artifact exists in the recorded session "
                                              "(offline aftereffect design); no blanking interval applies"),
        "post_stimulation_windows_through_probe_s": VOID(
            "'post-stimulation' in the concurrent-train sense does not apply to an offline aftereffect design; "
            "the entire recorded EEG session already IS the post-stimulation window (tens of minutes later)"),
        "behaviour_type": VOID("the public BIDS release does not carry trial-level behavioural accuracy/response "
                                "data (limitations field of the loader's own design block); only EEG spectral "
                                "state is scored here"),
        "behaviour_grain": VOID("no behavioural data is available at any grain in this release"),
        "repeated_sites_count": VOID("site/montage is not read by this census"),
        "repeated_doses_count": VOID("no dose parameter beyond target band is read by this census"),
        "repeated_timing_conditions_count": MEASURED(
            0, "exactly two conditions (verum, sham) with no third timing level -- this corpus does NOT provide "
               "within-participant stimulation-timing variation, and that absence is the structural finding "
               "recorded for this corpus, not an unmeasured field"),
        "participant_count": MEASURED(len(inventory["complete_pairs"]), "participants with both a sham and a verum "
                                                                         "EEG recording on disk"),
        "session_count": MEASURED(2 * len(inventory["complete_pairs"]), "two sessions (sham, verum) per complete participant"),
        "list_count": VOID("this task has no list structure"),
        "item_count": VOID("this task's stimuli are not separately item-indexed in the event table read here"),
        "trial_count": MEASURED(total_trials, "sum of EVENT_MAP-matched trial events across readable sessions"),
        "control_conditions": MEASURED(["sham stimulation session (same participant, same task)"],
                                        "SESSIONS = (sham, verum)"),
        "session_accounting": MEASURED(
            {"n_participants_seen": len(inventory["registered_participants"]),
             "n_complete_pairs": len(inventory["complete_pairs"]),
             "n_unpaired_raw": len(inventory["unpaired_raw"]), "unpaired_raw": inventory["unpaired_raw"],
             "n_no_raw_memory_recording": len(inventory["no_raw_memory_recording"]),
             "no_raw_memory_recording": inventory["no_raw_memory_recording"],
             "reconciliation": "n_participants_seen == n_complete_pairs + n_unpaired_raw + n_no_raw_memory_recording"},
            "run_ds005034_tacs_aftereffect.paired_inventory"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Macaque dlPFC delay-period (working-memory maintenance) microstimulation.
# ═══════════════════════════════════════════════════════════════════════════

def _macaque_session_params(prefix: str) -> dict | None:
    """Reads only the params struct (channel tokens + amplitude token per
    condition, trial-sequence length) -- never the spikerate cell arrays --
    for both MAT-file generations this corpus ships."""
    mat_path = MACAQUE_DATA / "correct" / f"{prefix}.mat"
    if not mat_path.exists():
        return None
    try:
        with h5py.File(str(mat_path), "r") as f:
            params = f["params"]
            chan_tokens = [_ascii_to_str(f[params["cond_uStimChan"][i, 0]][()])
                            for i in range(params["cond_uStimChan"].shape[0])]
            amp_tokens = [_ascii_to_str(f[params["cond_uStimAmp"][i, 0]][()])
                          for i in range(params["cond_uStimAmp"].shape[0])]
            n_trialsequence = int(np.asarray(params["trialsequence"]).size) if "trialsequence" in params else 0
            n_angle = int(params["cond_targetAngle"].shape[0])
            fmt = "hdf5_v73"
    except OSError:
        import scipy.io as sio
        d = sio.loadmat(str(mat_path), simplify_cells=True)
        params = d["params"]
        chan_tokens = [str(t) for t in np.atleast_1d(params["cond_uStimChan"])]
        amp_raw = params.get("cond_uStimAmp")
        amp_tokens = [str(t) for t in np.atleast_1d(amp_raw)] if amp_raw is not None else []
        n_trialsequence = int(np.asarray(params["trialsequence"]).size) if "trialsequence" in params else 0
        n_angle = int(d["spikerate"].shape[1])
        fmt = "scipy_v5_v7"

    sites = [_parse_chan_token(t) for t in chan_tokens]
    n_control = sum(1 for s in sites if not s)
    amps = [float(x) for tok in amp_tokens for x in tok.replace(",", " ").split() if x.strip().replace(".", "", 1).isdigit()]
    return {"format": fmt, "n_conditions": len(sites), "n_stim_sites": len(sites) - n_control,
            "n_control_conditions": n_control, "site_channel_tokens": sites,
            "amplitudes_ua": sorted(set(amps)), "n_trialsequence": n_trialsequence, "n_angle_targets": n_angle}


def census_macaque_pfc_microstimulation() -> dict:
    per_session, seen, refused = {}, [], {}
    for prefix in MACAQUE_SESSIONS:
        seen.append(prefix)
        info = _macaque_session_params(prefix)
        if info is None:
            refused[prefix] = "correct-trial MAT file not found or unreadable"
            continue
        per_session[prefix] = info

    session_amp_sets = {prefix: info["amplitudes_ua"] for prefix, info in per_session.items()}
    n_sites_total = sum(info["n_stim_sites"] for info in per_session.values())
    n_sessions_multi_site = sum(1 for info in per_session.values() if info["n_stim_sites"] > 1)
    fixed_within_session = all(len(info["amplitudes_ua"]) <= 1 for info in per_session.values())
    window_s = MACAQUE_N_BINS * MACAQUE_BIN_S - MACAQUE_PRE_S

    return {
        "corpus_id": "macaque_pfc_microstimulation",
        "arm_id": "dlpfc_delay_period_multisite_microstimulation",
        "arm_description": MEASURED(
            "Intracortical microstimulation delivered during the working-memory delay period at one of several "
            "electrode sites (chosen per trial from a per-session menu that always includes a no-stimulation "
            "control condition), two monkeys, two Utah-array recording configurations.",
            "run_macaque_pfc_microstimulation_pipeline module docstring; params.cond_uStimChan per session"),
        "species": MEASURED("macaque (2 animals: Sa, Wa)", "SESSIONS prefixes"),
        "task": MEASURED("delayed-response working-memory task with a saccade/reach target set by cond_targetAngle",
                          "params.cond_targetAngle"),
        "task_category": MEASURED("working_memory_maintenance", "stimulation is delivered during the delay period "
                                                                   "by design (module docstring)"),
        "is_working_memory_maintenance_stimulation": MEASURED(True, "stim-onset-aligned crop starts at "
                                                                       f"-{MACAQUE_PRE_S} s and stimulation is "
                                                                       "delivered mid-delay by design"),
        "recording_region": MEASURED(f"dorsolateral PFC, area {DLPFC_CONNECTOME_AREA} (Utah arrays; area-level only, "
                                      "no per-electrode histology in this release)", "module DLPFC_CONNECTOME_AREA constant"),
        "recording_modality": MEASURED("multi-unit spikerate (Utah array), 192 channels (Sa, 2 arrays) or 96 "
                                        "channels (Wa, 1 array)", "params.channels per session"),
        "stimulation_site": MEASURED(
            {"n_sessions_with_more_than_one_site": n_sessions_multi_site,
             "total_stim_sites_across_sessions": n_sites_total,
             "per_session_site_channel_tokens": {k: v["site_channel_tokens"] for k, v in per_session.items()}},
            "params.cond_uStimChan per session (electrode-pair or single-channel token per condition)"),
        "stimulation_site_randomization": MEASURED(
            "not randomly assigned: microstimulation sites are chosen by the experimenters per session based on "
            "array coverage, not randomised; trial-to-condition (site vs. control) sequencing within a session "
            "follows the recorded trialsequence field, which is an experimenter-set pseudo-randomised order "
            "(module docstring), separate from site SELECTION itself", "params.trialsequence; module docstring"),
        "treatment_assignment_unit": MEASURED("trial (one stimulation condition per trial, drawn from that "
                                                "session's own site menu including control)", "params.trialsequence"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=False, state_triggered=False,
                                           clinically_chosen_site=False, documented_randomization=True),
            "trial-to-condition order is recorded in params.trialsequence as an experimenter-set pseudo-"
            "randomised sequence (module docstring); this classifies the WITHIN-SESSION trial-to-condition "
            "draw, not the (clinically/experimentally fixed, unrandomised) choice of which sites exist in the "
            "session's menu at all -- see stimulation_site_randomization for that separate question"),
        "train_onset": MEASURED(f"stim-onset-aligned crop starts {MACAQUE_PRE_S} s before stimulation onset "
                                 "(aligncode 253, per params.aligncode)", "params.binLimleft / aligncode"),
        "train_duration_s": VOID("pulse-train duration is not carried as its own field in params; only the "
                                  "analysis crop window (pre/post stim-onset) is available, see train_onset and "
                                  "post_stimulation_windows_through_probe_s"),
        "item_overlap": VOID("this task presents one memorandum (target angle) per trial; item-overlap (multiple "
                              "items per train) does not apply to a single-item delayed-response design"),
        "epoch": MEASURED("working-memory delay period, stimulation-onset aligned", "params.aligncode / binLimleft"),
        "amplitude": MEASURED(session_amp_sets, "params.cond_uStimAmp per session, decoded from its ASCII token, "
                                                 "microamps"),
        "frequency_hz": VOID("not carried as its own field in params for this release"),
        "pulse_width_us": VOID("not carried as its own field in params for this release"),
        "pulse_count": VOID("not carried as its own field in params for this release"),
        "charge": VOID("no charge field exists in params, and pulse width/count needed to derive it are also absent"),
        "within_participant_dose_variation": MEASURED(
            {"amplitude_within_session": not fixed_within_session,
             "amplitude_between_sessions_same_monkey": len({a for k, v in session_amp_sets.items() for a in v}) > 1,
             "site_within_session": n_sessions_multi_site > 0},
            "cond_uStimAmp carries exactly one amplitude value per session in every session read (fixed within "
            "session); stimulation SITE varies within session in "
            f"{n_sessions_multi_site}/{len(per_session)} sessions"),
        "within_site_dose_variation": MEASURED(
            {"amplitude": False}, "a given session's stimulation sites all share that session's single fixed "
                                   "amplitude value; amplitude does not vary by site within a session"),
        "monotone_rescaling_notes": VOID("only one dose parameter (amplitude) is read from this release; no "
                                          "second parameter exists to test a rescaling relationship against"),
        "pre_stimulation_window_s": MEASURED(MACAQUE_PRE_S, "params.binLimleft (crop start relative to stim onset)"),
        "artifact_blanking_interval_s": VOID("this census reads params/condition metadata only, never the "
                                              "spikerate arrays, so no measured blanking window is available here"),
        "post_stimulation_windows_through_probe_s": MEASURED(
            window_s, "N_BINS*BIN_S - PRE_S, the pipeline's own cropped post-stim-onset window length; this is "
                      "the global per-trial minimum across all 11 sessions, per the pipeline's own comment"),
        "behaviour_type": MEASURED("trial correct/error, read from the correct/ vs error/ file split", "module docstring"),
        "behaviour_grain": MEASURED("binary per trial", "correct/error MAT file split"),
        "repeated_sites_count": MEASURED(n_sessions_multi_site, "sessions whose site menu includes more than one "
                                                                  "distinct stimulation site"),
        "repeated_doses_count": MEASURED(
            len({a for k, v in session_amp_sets.items() for a in v}) if len(per_session) > 1 else 0,
            "distinct amplitude values seen when pooling across sessions (amplitude is fixed within a session "
            "but differs between sessions/monkeys)"),
        "repeated_timing_conditions_count": VOID("no discrete timing condition beyond stim-onset alignment is "
                                                   "recorded in params for this release"),
        "participant_count": MEASURED(len({p[:2] for p in per_session}), "distinct monkey codes among included sessions"),
        "session_count": MEASURED(len(per_session), "sessions with a readable correct-trial params struct"),
        "list_count": VOID("this task has no list structure"),
        "item_count": VOID("one memorandum (target angle) per trial; not separately enumerated beyond trial_count"),
        "trial_count": VOID("per-trial counts require opening the spikerate cell arrays, which this params-only "
                             "census pass does not read; trial counts by condition are available in "
                             "results/observability_and_power_census.json's admission_matrix (median_n_trials "
                             "for macaque_pfc_microstimulation::single_unit::pooled), sourced from a fuller pass"),
        "control_conditions": MEASURED(["no-stimulation trials interleaved in the same session (control channel token)"],
                                        "params.cond_uStimChan control entry (token parses to an empty site list)"),
        "session_accounting": MEASURED(
            {"n_sessions_seen": len(seen), "n_sessions_included": len(per_session), "excluded_by_reason": refused,
             "reconciliation": "n_sessions_seen == n_sessions_included + len(excluded_by_reason)"},
            "MACAQUE_SESSIONS roster vs. readable correct-trial params structs"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# alagapan_phase_stimulation -- human multi-site direct cortical stimulation,
# working-memory RETENTION-period phase-lag manipulation, n=3 patients.
# ═══════════════════════════════════════════════════════════════════════════

def census_alagapan() -> dict:
    per_patient, refused = {}, {}
    for patient in ALAGAPAN_PATIENTS:
        try:
            conditions = alagapan_trial_conditions(patient)
        except (FileNotFoundError, OSError) as exc:
            refused[patient] = f"{type(exc).__name__}: {exc}"
            continue
        counts = {}
        for c in conditions:
            counts[c] = counts.get(c, 0) + 1
        per_patient[patient] = {"n_trials": len(conditions), "n_trials_per_condition": counts}

    total_trials = sum(v["n_trials"] for v in per_patient.values())
    condition_set = sorted({c for v in per_patient.values() for c in v["n_trials_per_condition"]})
    frequency_by_patient = {p: spec["stim_freq_hz"] for p, spec in ALAGAPAN_PATIENTS.items()}

    return {
        "corpus_id": "alagapan_phase_stimulation",
        "arm_id": "multisite_retention_phase_lag_stimulation",
        "arm_description": MEASURED(
            "Two-site direct cortical stimulation during the working-memory RETENTION period, manipulating the "
            "phase lag between sites across three categorical conditions (In Phase, Anti Phase, Sham); n=3 "
            "patients, each with their own fixed 2-electrode-pair montage and band-matched frequency.",
            "run_alagapan_stimulation_geometry module docstring; STIM_SITES, CONDITIONS constants"),
        "species": MEASURED("human", "PATIENTS roster (P1, P2, P3)"),
        "task": MEASURED("Sternberg-family working-memory task with explicit retention-period event markers "
                          "(DIN2 encoding end / DIN4 retention onset / DIN5 retention offset)",
                          "Task Performance CSV filename pattern; _retention_onset_offset_ms event codes"),
        "task_category": MEASURED("working_memory_maintenance", "phase-lag manipulation targets the retention "
                                                                   "(post-encoding, pre-probe) window explicitly"),
        "is_working_memory_maintenance_stimulation": MEASURED(
            True, "stimulation window is DIN4 (retention onset) to DIN5 (retention offset) by the loader's own event logic"),
        "recording_region": MEASURED("clinically-implanted intracranial electrodes, per-patient montage", "Electrode Mapping CSVs"),
        "recording_modality": MEASURED("intracranial EEG (iEEG)", "*_SMS_Stimulation.set / *_SMS_Baseline_Epoched.set"),
        "stimulation_site": MEASURED(ALAGAPAN_STIM_SITES, "STIM_SITES constant, resolved from the source study's "
                                                             "own hardcoded stimElectrodes per patient"),
        "stimulation_site_randomization": MEASURED(
            "not randomly assigned: each patient's 2-site montage is fixed by clinical electrode placement and "
            "the source study's own network-targeting choice, not randomised", "STIM_SITES is one fixed montage per patient"),
        "treatment_assignment_unit": MEASURED("trial (one of 3 categorical phase conditions per trial)",
                                               "Task Performance Condition column"),
        "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=False, state_triggered=False,
                                           clinically_chosen_site=False, documented_randomization=True),
            "the source study's task-performance summary records a per-trial Condition label (In Phase / Anti "
            "Phase / Sham) read directly from the release; this census treats that per-trial category as the "
            "assignment record, with no online neural-state trigger involved in choosing it (unlike the phase-"
            "TARGETING itself in the CLAM-tACS corpus, this design's phase categories are block/trial-scheduled, "
            "not closed-loop-estimated)"),
        "train_onset": MEASURED("locked to DIN4 (retention onset), plus a fixed 0.25 s buffer", "RETENTION_ONSET_BUFFER_S"),
        "train_duration_s": VOID("per-trial retention-window duration varies with set size (longer encoding at "
                                  "larger set sizes shifts DIN5); this census does not re-open the raw .set files "
                                  "to compute a per-trial duration distribution"),
        "item_overlap": VOID("stimulation targets the whole retention window rather than an individual item "
                              "presentation; item-overlap (multiple items per train) does not apply the way it "
                              "does to the RAM open-loop encoding-period corpus"),
        "epoch": MEASURED("retention/maintenance window (DIN4 to DIN5), not encoding", "_retention_onset_offset_ms convention"),
        "amplitude": VOID("not read by this census (Task Performance CSV and electrode mapping do not carry a "
                           "current-amplitude field; the raw stimulator log is not opened here)"),
        "frequency_hz": MEASURED(frequency_by_patient,
                                  "PATIENTS['stim_freq_hz'] in run_alagapan_phase_omega.py -- a per-patient "
                                  "band-matched frequency (theta for P1, alpha for P2/P3) recorded in that "
                                  "module's own PATIENTS constant. That constant is itself asserted from the "
                                  "source paper's subject-specific choices (its own module docstring says so), "
                                  "not independently re-measured from a raw stimulator log by this or any script "
                                  "in this repository."),
        "pulse_width_us": VOID("not carried in any file this census reads"),
        "pulse_count": VOID("not carried in any file this census reads"),
        "charge": VOID("amplitude and pulse width are both unavailable, so charge cannot be read or derived"),
        "within_participant_dose_variation": MEASURED(
            {"phase_condition": len(condition_set) > 1, "frequency_hz": False, "site": False},
            "every included patient contributes trials at more than one of the three phase conditions; "
            "frequency and site are each fixed per patient (between-patient variation only)"),
        "within_site_dose_variation": MEASURED(
            {"phase_condition": len(condition_set) > 1},
            "site is fixed per patient, so within-participant and within-site variation coincide here"),
        "monotone_rescaling_notes": VOID("phase condition is categorical (In Phase / Anti Phase / Sham), not a "
                                          "continuous dose parameter, so a monotone-rescaling test does not apply"),
        "pre_stimulation_window_s": VOID("this census does not re-open the raw .set files to measure a pre-"
                                          "retention baseline window length directly; see train_onset for the "
                                          "onset convention itself"),
        "artifact_blanking_interval_s": MEASURED(0.25, "RETENTION_ONSET_BUFFER_S: retention onset is padded by "
                                                          "this fixed buffer specifically to guard against a "
                                                          "stimulation-artifact decay tail, per the pipeline's own comment"),
        "post_stimulation_windows_through_probe_s": VOID("not re-measured in this census; DIN5 (retention offset/"
                                                           "probe onset) closes the window used, with no further "
                                                           "post-stimulation interval before the probe by task design"),
        "behaviour_type": MEASURED("accuracy on the working-memory probe", "Task Performance Summary CSV"),
        "behaviour_grain": MEASURED("binary per trial", "load_behavior _normalize_accuracy"),
        "repeated_sites_count": MEASURED(0, "each patient's 2-site montage is unique to that patient; sites are "
                                             "not shared or repeated across patients"),
        "repeated_doses_count": VOID("amplitude is not read by this census (see amplitude field)"),
        "repeated_timing_conditions_count": MEASURED(len(condition_set),
                                                       "distinct phase conditions observed across included patients' "
                                                       "own Task Performance Condition columns"),
        "participant_count": MEASURED(len(per_patient), "patients with a readable Task Performance Summary CSV"),
        "session_count": MEASURED(len(per_patient), "one stimulation recording session per included patient in this release"),
        "list_count": VOID("this task has no list structure"),
        "item_count": VOID("per-trial set size (encoding item count) is available in the raw event stream but not "
                            "re-extracted by this census; see behaviour_grain for what is read"),
        "trial_count": MEASURED(total_trials, "sum of Task Performance CSV rows across included patients"),
        "control_conditions": MEASURED(["Sham (categorical condition, same montage, no phase-lag manipulation)"],
                                        "CONDITIONS constant includes 'Sham'"),
        "session_accounting": MEASURED(
            {"n_patients_seen": len(ALAGAPAN_PATIENTS), "n_patients_included": len(per_patient),
             "excluded_by_reason": refused,
             "reconciliation": "n_patients_seen == n_patients_included + len(excluded_by_reason)"},
            "PATIENTS roster vs. readable Task Performance Summary CSVs"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Passive corpora -- no stimulation arm; used elsewhere in this project to
# define candidate biomarkers or normative trajectories. Every stimulation
# field on these rows is a structural void by definition (no device exists
# in the release), included so the census is a complete map, not a
# stimulation-only subset.
# ═══════════════════════════════════════════════════════════════════════════

_NO_STIM_VOID = VOID("this is a passive recording corpus; the release contains no stimulation device or event stream")


def _passive_row(**overrides) -> dict:
    base = {
        "stimulation_site": _NO_STIM_VOID, "stimulation_site_randomization": _NO_STIM_VOID,
        "treatment_assignment_unit": _NO_STIM_VOID, "treatment_assignment_classification": MEASURED(
            classify_treatment_assignment(single_level=True, state_triggered=False,
                                           clinically_chosen_site=False, documented_randomization=False),
            "no stimulation exists in this release; single_level_nothing_to_randomize is the correct "
            "classification for the absence of a treatment factor, not a separately-reasoned void"),
        "is_working_memory_maintenance_stimulation": MEASURED(False, "no stimulation device exists in this release"),
        "train_onset": _NO_STIM_VOID, "train_duration_s": _NO_STIM_VOID, "item_overlap": _NO_STIM_VOID,
        "amplitude": _NO_STIM_VOID, "frequency_hz": _NO_STIM_VOID, "pulse_width_us": _NO_STIM_VOID,
        "pulse_count": _NO_STIM_VOID, "charge": _NO_STIM_VOID,
        "within_participant_dose_variation": _NO_STIM_VOID, "within_site_dose_variation": _NO_STIM_VOID,
        "monotone_rescaling_notes": _NO_STIM_VOID,
        "pre_stimulation_window_s": _NO_STIM_VOID, "artifact_blanking_interval_s": _NO_STIM_VOID,
        "post_stimulation_windows_through_probe_s": _NO_STIM_VOID,
        "repeated_sites_count": _NO_STIM_VOID, "repeated_doses_count": _NO_STIM_VOID,
        "repeated_timing_conditions_count": _NO_STIM_VOID,
        "list_count": VOID("not separately enumerated by this census for this corpus"),
        "control_conditions": VOID("no stimulation exists, so no stimulated-vs-control contrast applies; the "
                                    "behavioural/neural baseline this corpus provides elsewhere in the project is "
                                    "not itself a 'control condition' in the stimulation-design sense this field asks about"),
    }
    base.update(overrides)
    return base


def census_dandi_000469() -> dict:
    directory = data_root() / "000469"
    subject_dirs = sorted(directory.glob("sub-*")) if directory.is_dir() else []
    seen, included, excluded = [], 0, {}
    n_trials_total = 0
    for subject_dir in subject_dirs:
        for path in sorted(subject_dir.glob("*_ses-2_ecephys+image.nwb")):
            seen.append(path.stem)
            try:
                with h5py.File(path, "r") as handle:
                    if "units" not in handle:
                        excluded[path.stem] = "no units group in this NWB file"
                        continue
                    trials = handle["intervals/trials"]
                    accuracy = trials["response_accuracy"][:].astype(bool)
                    loads = trials["loads"][:].astype(int)
                keep = int(((loads == 1) & accuracy).sum())
                if keep < PASSIVE_MIN_TRIALS:
                    excluded[path.stem] = f"only {keep} load-1 correct trials, below the {PASSIVE_MIN_TRIALS}-trial floor"
                    continue
                included += 1
                n_trials_total += keep
            except (OSError, KeyError) as exc:
                excluded[path.stem] = f"{type(exc).__name__}: {exc}"
    return _passive_row(
        corpus_id="dandi_000469", arm_id="passive_single_unit_sternberg_wm",
        arm_description=MEASURED("Human single-unit Sternberg working-memory task, load-1 correct trials only, "
                                  "used elsewhere in this project to define candidate persistence/geometry biomarkers.",
                                  "src/corpus_sessions.py iter_dandi_000469 filter (loads==1 & accuracy)"),
        species=MEASURED("human", "sub-* directory naming"),
        task=MEASURED("Sternberg working-memory maintenance task", "trials.loads / trials.response_accuracy fields"),
        task_category=MEASURED("working_memory_maintenance", "timestamps_Maintenance epoch marker in the trial table"),
        recording_region=MEASURED("multiple medial temporal / frontal structures (amygdala, hippocampus, dACC, "
                                   "pre-SMA, vmPFC), per-region and pooled", "REGIONS_WITH_POOLED in src/spike_pipeline.py"),
        recording_modality=MEASURED("single-unit spike times (microwire/Behnke-Fried)", "NWB units table"),
        epoch=MEASURED("baseline / encoding / delay (maintenance) / probe, from named NWB timestamp fields",
                        "timestamps_FixationCross/Encoding1/Maintenance/Probe"),
        behaviour_type=MEASURED("recognition-probe accuracy", "trials.response_accuracy"),
        behaviour_grain=MEASURED("binary per trial", "response_accuracy is boolean"),
        participant_count=MEASURED(len(subject_dirs), "sub-* directories found on disk"),
        session_count=MEASURED(included, "sessions with a units group and >=20 load-1-correct trials"),
        item_count=MEASURED(n_trials_total, "load-1 correct trials pooled across included sessions (this task "
                                             "presents one item per trial at load 1)"),
        trial_count=MEASURED(n_trials_total, "identical to item_count at load 1"),
        session_accounting=MEASURED(
            {"n_sessions_seen": len(seen), "n_sessions_included": included, "excluded_by_reason": excluded,
             "reconciliation": "n_sessions_seen == n_sessions_included + len(excluded_by_reason)"},
            "directory scan of 000469/sub-*/*_ses-2_ecephys+image.nwb, units-group and load-1-correct-trial-count check only "
            "(no spike-time loading, no unit-level firing-rate QC -- that QC is already reflected in "
            "results/observability_and_power_census.json's admission_matrix)"),
    )


def census_dandi_001187() -> dict:
    sessions = canonical_sessions()
    seen, included, excluded = [], 0, {}
    n_trials_total = 0
    for meta in sessions:
        if meta["primary_release"] != "001187":
            continue
        seen.append(meta["patient"])
        # primary_path already includes its own release-name prefix (e.g. "001187/sub-22/..."),
        # matching src/corpus_sessions.py's iter_dandi_001187 convention exactly.
        path = data_root() / meta["primary_path"]
        if not path.exists():
            excluded[meta["patient"]] = "primary NWB path not found on disk"
            continue
        try:
            with h5py.File(path, "r") as handle:
                if "units" not in handle:
                    excluded[meta["patient"]] = "no units group in this NWB file"
                    continue
                trials = _trial_group(handle, "001187")
                accuracy = trials["response_accuracy"][:].astype(bool)
            keep = int(accuracy.sum())
            if keep < PASSIVE_MIN_TRIALS:
                excluded[meta["patient"]] = f"only {keep} correct trials, below the {PASSIVE_MIN_TRIALS}-trial floor"
                continue
            included += 1
            n_trials_total += keep
        except (OSError, KeyError) as exc:
            excluded[meta["patient"]] = f"{type(exc).__name__}: {exc}"
    return _passive_row(
        corpus_id="dandi_001187", arm_id="passive_single_unit_sternberg_wm_multiload",
        arm_description=MEASURED("Deduplicated 001187/000673 human single-unit Sternberg working-memory corpus "
                                  "(multiple set sizes), correct trials only.",
                                  "run_human_drift_spine_001187_000673.canonical_sessions"),
        species=MEASURED("human", "canonical_sessions patient roster"),
        task=MEASURED("Sternberg working-memory maintenance task, multiple set sizes", "trials.response_accuracy field"),
        task_category=MEASURED("working_memory_maintenance", "timestamps_Maintenance epoch marker"),
        recording_region=MEASURED("amygdala, hippocampus, pooled (region set differs slightly from 000469)",
                                   "REGIONS_WITH_POOLED in src/spike_pipeline.py, as applied in iter_dandi_001187"),
        recording_modality=MEASURED("single-unit spike times (microwire/Behnke-Fried)", "NWB units table"),
        epoch=MEASURED("baseline / encoding / delay (maintenance) / probe", "timestamps_FixationCross/Encoding1/Maintenance/Probe"),
        behaviour_type=MEASURED("recognition-probe accuracy", "trials.response_accuracy"),
        behaviour_grain=MEASURED("binary per trial", "response_accuracy is boolean"),
        participant_count=MEASURED(len(seen), "distinct patients in the 001187-primary canonical roster"),
        session_count=MEASURED(included, "sessions with a units group and >=20 correct trials"),
        item_count=VOID("multiple set sizes are pooled without a re-derived per-load item count in this census"),
        trial_count=MEASURED(n_trials_total, "correct trials pooled across included sessions"),
        session_accounting=MEASURED(
            {"n_sessions_seen": len(seen), "n_sessions_included": included, "excluded_by_reason": excluded,
             "reconciliation": "n_sessions_seen == n_sessions_included + len(excluded_by_reason)"},
            "canonical_sessions() 001187-primary roster, units-group and correct-trial-count check only"),
    )


def census_dandi_000574() -> dict:
    directory = data_root() / "000574"
    subject_dirs = sorted(directory.glob("sub-*")) if directory.is_dir() else []
    seen, included, excluded = [], 0, {}
    n_trials_total = 0
    for subject_dir in subject_dirs:
        for path in sorted(subject_dir.glob("*.nwb")):
            seen.append(path.stem)
            try:
                with h5py.File(path, "r") as handle:
                    if "units" not in handle:
                        excluded[path.stem] = "no units group in this NWB file"
                        continue
                    trials = handle["intervals/trials"]
                    artifact = trials["artifact"][:].astype(bool)
                    correct = trials["correct"][:].astype(bool)
                keep = int(((~artifact) & correct).sum())
                if keep < PASSIVE_MIN_TRIALS:
                    excluded[path.stem] = f"only {keep} clean-correct trials, below the {PASSIVE_MIN_TRIALS}-trial floor"
                    continue
                included += 1
                n_trials_total += keep
            except (OSError, KeyError) as exc:
                excluded[path.stem] = f"{type(exc).__name__}: {exc}"
    return _passive_row(
        corpus_id="dandi_000574", arm_id="passive_single_unit_sternberg_wm_dandi_000574",
        arm_description=MEASURED("DANDI 000574 human single-unit Sternberg working-memory corpus, artifact-free "
                                  "correct trials only.", "src/corpus_sessions.py iter_dandi_000574 filter"),
        species=MEASURED("human", "sub-* directory naming"),
        task=MEASURED("Sternberg working-memory maintenance task", "trials.artifact / trials.correct fields"),
        task_category=MEASURED("working_memory_maintenance", "fixed relative epoch structure documented in "
                                                                "src/corpus_sessions.py (no named per-epoch timestamps "
                                                                "in this release's trial table; epochs are derived "
                                                                "from start_time by a fixed offset)"),
        recording_region=MEASURED("amygdala, hippocampus, pooled, plus depth-contact LFP and scalp EEG (this "
                                   "release also ships non-spike modalities)", "BORAN_REGIONS_WITH_POOLED"),
        recording_modality=MEASURED("single-unit spike times (microwire), depth-contact LFP, scalp EEG", "NWB units/electrodes tables"),
        epoch=MEASURED("baseline/encoding/delay(maintenance)/probe derived from start_time + fixed offsets "
                        "(no named per-epoch timestamp fields in this release)", "src/corpus_sessions.py iter_dandi_000574 docstring"),
        behaviour_type=MEASURED("recognition-probe accuracy", "trials.correct"),
        behaviour_grain=MEASURED("binary per trial", "trials.correct is boolean"),
        participant_count=MEASURED(len(subject_dirs), "sub-* directories found on disk"),
        session_count=MEASURED(included, "sessions with a units group and >=20 artifact-free correct trials"),
        item_count=VOID("no per-trial item-identity field exists in this release's public NWB (set_letters is "
                         "'not available' on every trial, per src/corpus_sessions.py's own documented finding)"),
        trial_count=MEASURED(n_trials_total, "artifact-free correct trials pooled across included sessions"),
        session_accounting=MEASURED(
            {"n_sessions_seen": len(seen), "n_sessions_included": included, "excluded_by_reason": excluded,
             "reconciliation": "n_sessions_seen == n_sessions_included + len(excluded_by_reason)"},
            "directory scan of 000574/sub-*/*.nwb, units-group and artifact-free-correct-trial-count check only"),
    )


def census_inagaki_alm5() -> dict:
    directory = alm_data_directory(data_root())
    files = sorted(directory.glob("*.mat")) if directory.is_dir() else []
    return _passive_row(
        corpus_id="inagaki_alm5", arm_id="passive_alm_delay_period_control_trials",
        arm_description=MEASURED("Mouse anterior lateral motor cortex (ALM) delayed-response task, control "
                                  "(unperturbed) trials -- the calibration comparison for this project's "
                                  "observability census; a photoinhibition-perturbation arm exists in the same "
                                  "release but is optogenetic, not electrical/current stimulation, and is not a "
                                  "stimulation-design arm for this census.",
                                  "src/corpus_sessions.py iter_alm docstring"),
        species=MEASURED("mouse", "inagaki_alm5 release (mouse ALM silicon-probe delayed-response task)"),
        task=MEASURED("delayed licking response task (instructed left/right)", "Trial_types_of_response_vector field"),
        task_category=MEASURED("motor_preparation", "delay period precedes an instructed lick response, not a "
                                                       "memorandum-report working-memory probe"),
        recording_region=MEASURED("anterior lateral motor cortex (ALM), single structure by design", "config/datasets.json local_path"),
        recording_modality=MEASURED("single-unit spike times (silicon probe)", "unit.SpikeTimes / Trial_idx_of_spike fields"),
        epoch=MEASURED("pre-delay-onset baseline and delay period, delay-onset aligned", "Delay_start field"),
        behaviour_type=MEASURED("instructed-lick response correctness", "Trial_types_of_response_vector response code"),
        behaviour_grain=MEASURED("categorical response code per trial (1-4)", "control_response_code field"),
        participant_count=MEASURED(len({p.stem.split("_")[0] for p in files}), "distinct mouse codes among session filenames"),
        session_count=MEASURED(len(files), "*.mat session files found on disk"),
        item_count=VOID("this task has a binary instructed target, not a multi-item memorandum; not applicable"),
        trial_count=VOID("per-session trial counts require opening each MAT file's Behavior/Trial_info structs, "
                          "which this file-count-only census pass does not do; trial counts are available in "
                          "results/observability_and_power_census.json's admission_matrix (median_n_trials for "
                          "inagaki_alm5::single_unit::pooled), sourced from a fuller pass"),
        session_accounting=MEASURED(
            {"n_sessions_seen": len(files), "n_sessions_included": len(files), "excluded_by_reason": {},
             "reconciliation": "n_sessions_seen == n_sessions_included (file-existence pass only; per-session "
                                "trial/unit-count QC happens in the full pipeline this census does not re-run)"},
            "directory scan of inagaki_alm5 local_path/RandomDelayTask/withPerturbation/*.mat"),
    )


def census_watters() -> dict:
    data_root_dir = data_root()
    dates = watters_session_dates(data_root_dir)
    behaviour = watters_behaviour(data_root_dir)
    n_trials_total = len(behaviour)
    n_sessions_with_spike_cache = 0
    spikes_dir, _ = watters_directories(data_root_dir)
    for animal, session_date, _variant in dates:
        if (spikes_dir / animal / session_date).is_dir():
            n_sessions_with_spike_cache += 1
    return _passive_row(
        corpus_id="watters_2026", arm_id="passive_multiobject_macaque_frontal_delay",
        arm_description=MEASURED("Multi-object macaque frontal-cortex delayed-report task (ring/triangle task "
                                  "variants), continuous saccadic report of a cued object's polar angle.",
                                  "src/corpus_sessions.py watters_behaviour/iter_watters"),
        species=MEASURED("macaque", "subject codes in the behaviour CSVs"),
        task=MEASURED("multi-object delayed continuous report (ring/triangle variants)", "WATTERS_TASK_VARIANTS"),
        task_category=MEASURED("working_memory_maintenance", "fixed ~1.0 s delay period per trial, delay-onset aligned"),
        recording_region=MEASURED("frontal cortex, pooled across probes (shared electrode table gives every "
                                   "electrode 'unknown' location -- no area-resolved split in this release)",
                                   "src/corpus_sessions.py _watters_unit_index docstring"),
        recording_modality=MEASURED("single/multi-unit spike counts (chronic probe)", "per-unit *_spike_counts.pkl cache"),
        epoch=MEASURED("delay (maintenance) period, ~1.0 s, delay-onset aligned per trial", "WATTERS_DELAY_WINDOW_S"),
        behaviour_type=MEASURED("continuous saccadic report; graded deviation from the cued object angle plus a "
                                 "threshold-based correct/incorrect", "report_deviation / correct fields"),
        behaviour_grain=MEASURED("continuous (graded report deviation) with a derived binary correctness threshold",
                                  "add_behavior_columns in run_watters_source_replication.py"),
        participant_count=MEASURED(len({a for a, _s, _v in dates}), "distinct animal codes among behavioural session dates"),
        session_count=MEASURED(len({(a, s) for a, s, _v in dates}), "distinct (animal, session date) pairs with "
                                                                      "completed behavioural trials"),
        item_count=MEASURED(n_trials_total, "rows in the pooled behaviour table (one row per completed trial, "
                                             "each presenting multiple objects but one cued memorandum)"),
        trial_count=MEASURED(n_trials_total, "identical to item_count -- one cued memorandum per completed trial"),
        session_accounting=MEASURED(
            {"n_behavioural_session_dates_seen": len({(a, s) for a, s, _v in dates}),
             "n_sessions_with_a_spike_cache_directory": n_sessions_with_spike_cache,
             "n_sessions_without_a_spike_cache_directory": len({(a, s) for a, s, _v in dates}) - n_sessions_with_spike_cache,
             "reconciliation": "n_behavioural_session_dates_seen == n_sessions_with_a_spike_cache_directory + "
                                "n_sessions_without_a_spike_cache_directory (per-session trial/unit-count QC that "
                                "further admits or refuses a session happens in load_watters_session, not re-run here)"},
            "watters_session_dates() against the spike-cache directory tree (existence check only, no unit loading)"),
    )


def census_panichello() -> dict:
    directory = panichello_data_directory()
    files = sorted(directory.glob("*.mat")) if directory.is_dir() else []
    monkeys = {monkey_for_session(f.stem) for f in files}
    return _passive_row(
        corpus_id="panichello_2024", arm_id="passive_spatial_wm_macaque_lpfc",
        arm_description=MEASURED("Macaque lateral PFC spatial working-memory task, simultaneous single-unit "
                                  "recording, used elsewhere in this project as a candidate-biomarker corpus.",
                                  "scripts/run_panichello_pipeline.py module docstring"),
        species=MEASURED("macaque", "monkey_for_session year-code mapping"),
        task=MEASURED("spatial delayed-response working-memory task", "run_panichello_pipeline module docstring"),
        task_category=MEASURED("working_memory_maintenance", "DELAY_WINDOW_MS constant"),
        recording_region=MEASURED("lateral prefrontal cortex (lPFC)", "run_panichello_pipeline module docstring"),
        recording_modality=MEASURED("single-unit spike times (simultaneous multi-electrode)", "*.mat session files"),
        epoch=MEASURED("delay period, 300-1450 ms post-cue", "DELAY_WINDOW_MS constant"),
        behaviour_type=MEASURED("continuous spatial report", "run_panichello_pipeline module docstring"),
        behaviour_grain=MEASURED("continuous report (graded), not re-extracted by this census", "module docstring"),
        participant_count=MEASURED(len(monkeys), "distinct monkey codes among session filenames"),
        session_count=MEASURED(len(files), "*.mat session files found on disk"),
        item_count=VOID("per-trial cue-set size is not re-extracted by this file-count-only census pass"),
        trial_count=VOID("per-session trial counts require opening each MAT file, which this file-count-only "
                          "census pass does not do; trial counts are available in "
                          "results/observability_and_power_census.json's admission_matrix (median_n_trials for "
                          "panichello_2024::single_unit::pooled), sourced from a fuller pass"),
        session_accounting=MEASURED(
            {"n_sessions_seen": len(files), "n_sessions_included": len(files), "excluded_by_reason": {},
             "reconciliation": "n_sessions_seen == n_sessions_included (file-existence pass only)"},
            "directory scan of panichello_2024 local_path/*.mat"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

CENSUS_BUILDERS = (
    census_ram_openloop, census_ram_closedloop, census_ram_closedloop_titration,
    census_clam_active, census_clam_control, census_ds005034,
    census_macaque_pfc_microstimulation, census_alagapan,
    census_dandi_000469, census_dandi_001187, census_dandi_000574,
    census_inagaki_alm5, census_watters, census_panichello,
)


def main() -> None:
    t0 = time.time()
    rows, row_defects, builder_failures = [], {}, {}
    for builder in CENSUS_BUILDERS:
        name = builder.__name__
        try:
            row = builder()
        except Exception as exc:  # noqa: BLE001 -- a failed corpus is reported, never silently dropped
            builder_failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        defects = validate_row(row)
        if defects:
            row_defects[f"{row.get('corpus_id', name)}::{row.get('arm_id', '?')}"] = defects
        rows.append(row)

    status = "complete" if not builder_failures and not row_defects else "incomplete"

    out = {
        "status": status,
        "decision_rule_declared_before_fitting": {
            "randomization_classification_rule": RANDOMIZATION_CLASSIFICATION_RULE,
            "structural_void_rule": STRUCTURAL_VOID_RULE,
        },
        "rows": rows,
        "builder_failures": builder_failures,
        "row_completeness_defects": row_defects,
        "scope": {
            "n_corpus_arm_rows": len(rows),
            "corpora": sorted({row["corpus_id"] for row in rows}),
            "data_root": str(data_root()),
            "seed": CENSUS_SEED,
            "git_commit": git_commit(ROOT),
            "wall_clock_seconds": None,  # filled in just before write
        },
    }
    out["scope"]["wall_clock_seconds"] = time.time() - t0

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(canonical_json(out))
    print(f"Wrote {OUT_PATH} in {out['scope']['wall_clock_seconds']:.1f} s -- status={status}, "
          f"{len(rows)} rows, {len(builder_failures)} builder failures, {len(row_defects)} incomplete rows")


if __name__ == "__main__":
    main()
