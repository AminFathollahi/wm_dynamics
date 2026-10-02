#!/usr/bin/env python3
"""Write preregistration/analysis_freeze.json and preregistration/analysis_freeze.md.

Fixes, before any chain analysis runs, which test, outcome, unit set and state space each
hypothesis x corpus x link cell uses. The rules and the per-corpus facts are literal data in this
file; nothing is computed from neural data. Where the capability matrix and the loader code or raw
files disagree, the cell carries the fact read from the code or the raw file and names its source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    if str(ROOT / _subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / _subdir))

from effect_size import minimum_detectable_correlation  # noqa: E402
from provenance import canonical_json, sha256_file, write_code_identity_record  # noqa: E402

FREEZE_PATH = ROOT / "preregistration" / "analysis_freeze.json"
TABLE_PATH = ROOT / "preregistration" / "analysis_freeze.md"
README_PATH = ROOT / "preregistration" / "README.md"
README_HEADING = "# Analysis freeze"
DATA_LOCK_PATH = ROOT / "provenance" / "data_lock.json"
MATRIX_PATH = ROOT / "docs" / "mandates" / "CHAIN_CAPABILITY_MATRIX_2026_09_24.md"
SPEC_PATH = ROOT / "docs" / "mandates" / "CHAIN_ANALYSIS_SPEC_2026_09_24.md"
CO_PRIMARY_PATH = ROOT / "results" / "representation_benchmark_co_primary.json"

DPCA = "demixed_principal_components"
GPFA = "gaussian_process_factor_analysis"
FIELD_BENCHMARK_SPACES = [DPCA, "principal_components", "factor_analysis", GPFA,
                          "recurrent_switching_linear_dynamics"]
UNIT_SPACES = [DPCA, GPFA]
DESIGN_SPACES = {
    "human_single_units_simultaneous": UNIT_SPACES,
    "human_single_units_pseudo_population": [DPCA],
    "human_depth_field_potentials": FIELD_BENCHMARK_SPACES,
}
NO_BENCHMARK_NOTE = "this representation set was not benchmarked on this recording type"
BENCHMARK_CELLS = {
    "human_single_units_simultaneous": ("human_single_units", "regions_pooled_by_patient"),
    "human_depth_field_potentials": ("dandi_000574", "pooled"),
}

LINK_NAMES = {
    1: "stimulation to firing", 2: "stimulation to population state", 3: "stimulation to behaviour",
    4: "firing to population state", 5: "firing to behaviour", 6: "population state to behaviour",
    7: "population state to behaviour given firing", 8: "stimulation by pre-stimulation state to behaviour",
    9: "stimulation to population state to behaviour (mediation)", 10: "region or site to effect",
    11: "dose to effect", 12: "recording type to recording type",
}
LINKS_USING_POPULATION_STATE = {2, 4, 6, 7, 8, 9, 10, 11, 12}
LINKS_USING_FIRING = {1, 4, 5, 7}
STIMULATION_LINKS = {1, 2, 3, 8, 9, 11}
POOLABLE_LINKS = {4, 5, 6, 7, 10}

HYPOTHESIS_LINKS = [("H1", 4), ("H2", 6), ("H3", 7), ("H3", 5), ("H4", 6), ("H4", 10), ("H5", 1), ("H5", 2),
                    ("H6", 3), ("H6", 8), ("H7", 10), ("H7", 11), ("H8", 9), ("H10", 12)]
OUTCOME_CELLS = {("H2", 6), ("H3", 7), ("H3", 5), ("H4", 6), ("H4", 10), ("H6", 3), ("H6", 8), ("H7", 10),
                 ("H7", 11), ("H8", 9), ("H10", 12)}

REASON_TEXT = {
    "a": "no stimulation in this corpus",
    "b": "no single-unit or multi-unit recording, so firing rate is absent",
    "c": "no behavioural accuracy or reaction time in the data available locally",
    "d": "stimulation is not trial-randomised (closed-loop, state-triggered or clinically assigned): descriptive reading only",
    "e": "no dose variation: amplitude, frequency and duration are fixed",
    "f": "no region or site variation within the corpus",
    "g": "small number of patients or animals",
    "h": "population state is not simultaneously constructible (pseudo-population)",
    "i": "behaviour absent for this preparation (no comparable per-trial accuracy family)",
}

CAPABILITY_ROWS = """
macaque_pfc_microstimulation Y Y Y P P P P P P Y N(e)
ram_ds005489_openloop N(b) Y Y N(b) N(b) P N(b) Y P P(f) Y
ram_ds005557_closedloop N(b) Y(d) P(d) N(b) N(b) P N(b) Y(d) P(d) P(f) P
dandi_000469 N(a) N(a) N(a) P P Y Y N(a) N(a) N(f) N(a)
dandi_001187 N(a) N(a) N(a) P P Y Y N(a) N(a) N(f) N(a)
dandi_000673 N(a) N(a) N(a) P P P P N(a) N(a) N(f) N(a)
dandi_000574 N(a) N(a) N(a) P P Y Y N(a) N(a) N(f) N(a)
dandi_000004 N(a) N(a) N(a) P P P P N(a) N(a) P N(a)
ds004752 N(a) N(a) N(a) N(b) N(b) Y N(b) N(a) N(a) Y N(a)
wolff_eeg_impulse N(a) N(a) N(a) N(b) N(b) P N(b) N(a) N(a) N(f) N(a)
kai_miller_nback N(a) N(a) N(a) N(b) N(b) N(c) N(b) N(a) N(a) N(f) N(a)
ds006848 N(a) N(a) N(a) N(b) N(b) P N(b) N(a) N(a) N(f) N(a)
haslacher_clam_tacs N(b) Y(d) P(d) N(b) N(b) P N(b) Y(d) P(d) P(f) N(e)
alagapan_phase_stimulation N(b) P(g) P(g) N(b) N(b) P N(b) P(g) P(g) P(g,f) P(g)
ds005034 N(b) P N(c) N(b) N(b) N(c) N(b) N(c) N(c) N(f) N(e)
panichello_2024 N(a) N(a) N(a) P P Y Y N(a) N(a) N(f) N(a)
watters_2026 N(a) N(a) N(a) Y P Y Y N(a) N(a) N(f) N(a)
inagaki_alm5 N(a)* N(a)* N(i) P N(i) N(i) N(i) N(a)* N(a)* N(f) N(a)
pfc3 N(a) N(a) N(a) P(h) N N N N(a) N(a) N(f) N(a)
pfc4 N(a) N(a) N(a) P(g) P(g) P(g) P(g) N(a) N(a) P(g) N(a)
"""
CAPABILITY = {line.split()[0]: line.split()[1:] for line in CAPABILITY_ROWS.strip().splitlines()}

MIN_UNITS_PER_REGION_SESSION = 8
REGION_LABELLED_UNIT_CORPORA = ("dandi_000469", "dandi_001187", "dandi_000673", "dandi_000574")
LEVEL_ONE_CORPORA = ("dandi_000574", "ds004752")
LEVEL_ONE_COUNT = 9

# Per-corpus facts. "entry" is the corpus entry number in the capability matrix the facts came from.
# label_kind: content = a per-trial memorandum label; task_condition = only a task factor such as set size;
# none = no per-trial label recorded; content_unspecified = a continuous memorandum with no binning declared.
# rt: loaded (shared session iterator) | derived_by_delivered_code (a delivered module computes it) |
# different_construct (a recall-period latency, not probe-locked) | unresolved (per-animal only) | absent.
# Reaction time means a latency from a probe to the response.
CORPORA = {
    "macaque_pfc_microstimulation": dict(
        entry=1, species="macaque", recording_type="units", simultaneous=True, task_class="spatial_delayed_saccade",
        cluster_unit="session", cluster_n=11, animals=2, cluster_source="11 sessions, 2 animals",
        design_class="randomised_by_trial", label_kind="content", class_label="target angle, 4 classes",
        load_levels=[], rt="derived_by_delivered_code", accuracy="loaded", continuous_error=None,
        state_space_bucket="no_benchmark", trial_note="non-stimulated control trials for links without stimulation",
        identity="not_askable:the trial count per identity has not been read at the loader; it is read from labels only before this set is used"),
    "ram_ds005489_openloop": dict(
        entry=2, species="human", recording_type="depth_field_potentials", simultaneous=False,
        task_class="delayed_free_recall", cluster_unit="patient", cluster_n=37, animals=None,
        cluster_source="37 patients with usable stimulation and recording (38 patient directories on disk)",
        design_class="randomised_by_list", label_kind="none", class_label=None, load_levels=[],
        rt="different_construct", accuracy="loaded", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="encoding-period words; non-stimulated lists for links without stimulation", identity=None),
    "ram_ds005557_closedloop": dict(
        entry=3, species="human", recording_type="depth_field_potentials", simultaneous=False,
        task_class="delayed_free_recall", cluster_unit="patient", cluster_n=16, animals=None,
        cluster_source="16 patients used by the latency census (18 patient directories on disk)",
        design_class="state_triggered", label_kind="none", class_label=None, load_levels=[],
        rt="different_construct", accuracy="loaded", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="list-level assignment randomised; item-level stimulation triggered by the pre-stimulation state",
        identity=None),
    "dandi_000469": dict(
        entry=4, species="human", recording_type="units", simultaneous=True, task_class="sternberg_item_recognition",
        cluster_unit="patient", cluster_n=20, animals=None, cluster_source="20 patients, 21 admitted sessions",
        design_class="none", label_kind="content", class_label="picture category, 5 classes", load_levels=[1],
        rt="derived_by_delivered_code", accuracy="in_raw_not_loaded", continuous_error=None,
        state_space_bucket="human_single_units_simultaneous", trial_note="load 1 correct trials",
        identity="not_askable:the label field holds only the 5 category values; no finer identity is recorded"),
    "dandi_001187": dict(
        entry=5, species="human", recording_type="units", simultaneous=True, task_class="sternberg_item_recognition",
        cluster_unit="patient", cluster_n=39, animals=None, cluster_source="39 patients, 46 admitted sessions",
        design_class="none", label_kind="content", class_label="picture category (picture identifier divided by 100), 5 classes",
        load_levels=[1, 3], rt="derived_by_delivered_code", accuracy="in_raw_not_loaded", continuous_error=None,
        state_space_bucket="human_single_units_simultaneous", trial_note="correct trials, loads 1 and 3",
        identity="not_askable:46 of 46 sessions have 99 to 140 distinct picture identities with a median of 1 trial per identity; none reaches 6"),
    "dandi_000673": dict(
        entry=6, species="human", recording_type="units", simultaneous=True, task_class="sternberg_item_recognition",
        cluster_unit="patient", cluster_n=5,
        animals=None,
        cluster_source="36 patients in the release; 5 patients remain after dropping the sessions that duplicate dandi_001187 recordings",
        design_class="none", label_kind="content", class_label="picture category (picture identifier divided by 100), 5 classes",
        load_levels=[1, 3], rt="derived_by_delivered_code", accuracy="in_raw_not_loaded", continuous_error=None,
        state_space_bucket="human_single_units_simultaneous", trial_note="correct trials, loads 1 and 3; duplicate sessions dropped",
        identity="not_askable:5 of 5 retained sessions have 122 to 137 distinct picture identities with a median of 1 trial per identity; none reaches 6"),
    "dandi_000574": dict(
        entry=7, species="human", recording_type="units", simultaneous=True, task_class="sternberg_item_recognition",
        cluster_unit="patient", cluster_n=9, animals=None, cluster_source="9 patients, 37 admitted sessions",
        design_class="none", label_kind="task_condition", class_label="set size, 4/6/8; no per-trial item identity is released",
        load_levels=[4, 6, 8], rt="derived_by_delivered_code", accuracy="in_raw_not_loaded", continuous_error=None,
        state_space_bucket="dandi_000574", trial_note="correct, non-artifact trials",
        identity="not_askable:no per-trial item identity is released"),
    "dandi_000004": dict(
        entry=8, species="human", recording_type="units", simultaneous=True, task_class="new_old_recognition",
        cluster_unit="patient", cluster_n=59, animals=None, cluster_source="59 patients, 87 admitted sessions",
        design_class="none", label_kind="content", class_label="stimulus category", load_levels=[],
        rt="loaded", accuracy="loaded", continuous_error=None, state_space_bucket="human_single_units_simultaneous",
        trial_note="recognition trials; correct and error trials", identity="not_askable:image identity is not loaded and the trial count per identity has not been read at the loader"),
    "ds004752": dict(
        entry=9, species="human", recording_type="depth_field_potentials", simultaneous=False,
        task_class="sternberg_item_recognition", cluster_unit="patient", cluster_n=15, animals=None,
        cluster_source="15 patients, 68 sessions", design_class="none", label_kind="task_condition",
        class_label="set size, 4/6/8", load_levels=[4, 6, 8], rt="derived_by_delivered_code", accuracy="loaded", continuous_error=None,
        state_space_bucket="ds004752", trial_note="correct and error trials", identity=None),
    "wolff_eeg_impulse": dict(
        entry=10, species="human", recording_type="scalp_eeg", simultaneous=False,
        task_class="retro_cue_orientation_report", cluster_unit="patient", cluster_n=None, animals=None,
        cluster_source=None, design_class="none", label_kind="content",
        class_label="memorised grating orientation, 6 bins of 30 degrees", load_levels=[],
        rt="absent", accuracy="loaded", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="correct and error trials", identity=None,
        experiments={
            "experiment_1": dict(cluster_n=30, cluster_source="experiment 1: 30 subject files; never combined with experiment 2",
                                 n_trials=39900, trial_source="30 subject files times the per-file minimum of 1330 trials (a lower bound; files hold 1330 to 1344)"),
            "experiment_2": dict(cluster_n=18, cluster_source="experiment 2: 18 subject files; never combined with experiment 1",
                                 n_trials=None, trial_source="trials per file were not read for experiment 2"),
        }),
    "kai_miller_nback": dict(
        entry=11, species="human", recording_type="surface_electrocorticography", simultaneous=False,
        task_class="n_back", cluster_unit="patient", cluster_n=4, animals=None, cluster_source="4 patients, 1 continuous recording each",
        design_class="none", label_kind="task_condition", class_label="n-back load, 0/1/2", load_levels=[0, 1, 2],
        rt="absent", accuracy="absent", continuous_error=None, state_space_bucket="no_benchmark", trial_note="stimulus onsets", identity=None),
    "ds006848": dict(
        entry=12, species="human", recording_type="scalp_eeg", simultaneous=False, task_class="serial_digit_recall",
        cluster_unit="patient", cluster_n=30, animals=None, cluster_source="30 patients, 1 session each",
        design_class="none", label_kind="task_condition", class_label="presentation mode, 4 classes", load_levels=[],
        rt="absent", accuracy="loaded", continuous_error="graded recall score (digits correct in order, 0 to 7)",
        state_space_bucket="no_benchmark", trial_note="all trials", identity=None),
    "haslacher_clam_tacs": dict(
        entry=13, species="human", recording_type="scalp_eeg", simultaneous=False,
        task_class="visual_retention_with_phase_locked_stimulation", cluster_unit="patient", cluster_n=46, animals=None,
        cluster_source="46 patients: 21 active-site, 25 control-site", design_class="state_triggered", label_kind="none",
        class_label=None, load_levels=[], rt="absent", accuracy="loaded", continuous_error=None,
        state_space_bucket="no_benchmark", trial_note="phase-lag conditions interleaved within session; site assigned between patients",
        identity=None),
    "alagapan_phase_stimulation": dict(
        entry=14, species="human", recording_type="intracranial_field_potentials", simultaneous=False,
        task_class="sternberg_item_recognition", cluster_unit="patient", cluster_n=3, animals=None,
        cluster_source="3 patients", design_class="randomised_by_trial", label_kind="task_condition",
        class_label="list length (present in the raw behaviour file; not read by the loader)", load_levels=[],
        rt="derived_by_delivered_code", accuracy="loaded", continuous_error=None, state_space_bucket="human_depth_field_potentials",
        trial_note="in-phase, anti-phase and sham trials interleaved within one session per patient", identity=None),
    "ds005034": dict(
        entry=15, species="human", recording_type="scalp_eeg", simultaneous=False,
        task_class="verbal_retention_and_reordering", cluster_unit="patient", cluster_n=25, animals=None,
        cluster_source="25 patients, 2 sessions each (sham, verum)", design_class="varies_between_sessions",
        label_kind="task_condition", class_label="operation, 3 classes", load_levels=[4, 6], rt="absent",
        accuracy="absent", continuous_error=None, state_space_bucket="no_benchmark", trial_note="all trials", identity=None),
    "panichello_2024": dict(
        entry=16, species="macaque", recording_type="units", simultaneous=True, task_class="spatial_delayed_saccade",
        cluster_unit="session", cluster_n=25, animals=3, cluster_source="25 sessions, 3 animals",
        design_class="none", label_kind="content", class_label="cue angle, 8 classes", load_levels=[],
        rt="absent", accuracy="loaded", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="correct and error trials",
        identity="askable:25 of 25 sessions have all 8 cue classes with at least 6 correct trials"),
    "watters_2026": dict(
        entry=17, species="macaque", recording_type="units", simultaneous=True, task_class="spatial_delayed_saccade",
        cluster_unit="session", cluster_n=41, animals=2, cluster_source="41 sessions that pass the quality filter, of 47 behavioural session dates, 2 animals",
        design_class="none", label_kind="content", class_label="cued angle, continuous, binned into 8 equal angular sectors",
        load_levels=[1, 2, 3], rt="loaded", accuracy="loaded",
        continuous_error="continuous report deviation of the saccade from the cued position",
        state_space_bucket="no_benchmark", trial_note="correct and error trials",
        identity="not_askable:the memorandum is a continuous angle with no discrete identity"),
    "inagaki_alm5": dict(
        entry=18, species="mouse", recording_type="units", simultaneous=True, task_class="instructed_lick_delayed_response",
        cluster_unit="session", cluster_n=23, animals=5, cluster_source="23 sessions, 5 animals",
        design_class="none", label_kind="content", class_label="instructed lick direction, 2 classes", load_levels=[],
        rt="absent", accuracy="loaded", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="control trials; the optogenetic perturbation arm is outside the stimulation definition",
        identity="not_askable:the trial count per lick direction has not been read at the loader"),
    "pfc3": dict(
        entry=19, species="macaque", recording_type="units", simultaneous=False,
        task_class="spatial_delayed_match_to_sample", cluster_unit="animal", cluster_n=4, animals=4,
        cluster_source="4 animals; one isolated neuron per file, so sessions do not apply",
        design_class="none", label_kind="content", class_label="cue location, 9 classes", load_levels=[],
        rt="absent", accuracy="absent", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="resampled pseudo-population; at least 8 trials per class for a neuron to be admitted",
        identity="askable:the loader admits a neuron only with at least 8 trials per class"),
    "pfc4": dict(
        entry=20, species="macaque", recording_type="units", simultaneous=True,
        task_class="vibrotactile_delayed_discrimination", cluster_unit="session", cluster_n=196, animals=2,
        cluster_source="267 raw session files; 196 sessions survive the latency census admission floor; 2 animals",
        design_class="none", label_kind="content", class_label="first comparison frequency (2 to 14 distinct values per session)",
        load_levels=[], rt="unresolved", accuracy="loaded", continuous_error=None, state_space_bucket="no_benchmark",
        trial_note="correct and error trials", max_units_per_session=7,
        identity="askable_per_session:233 of 267 session files have every first-frequency value at 6 or more correct trials"),
}

DOSE_LEVEL_FACTS = {
    "ram_ds005489_openloop": "7 of 37 stimulated patients have 2 or more amplitude levels (the dose slope needs at least 6); pulse frequency is constant within every patient",
    "ram_ds005557_closedloop": "9 of 16 stimulated patients have 2 or more amplitude levels (the dose slope needs at least 6)",
}

TRIAL_COUNTS = {
    "macaque_pfc_microstimulation": (None, "per-session totals were spot-checked in the raw files (about 320 to 368), not totalled over the 11 sessions"),
    "ram_ds005489_openloop": (None, "event tables were not totalled (about 300 words per session over 78 sessions)"),
    "ram_ds005557_closedloop": (None, "event tables were not totalled (29 sessions)"),
    "dandi_000469": (812, "correct load-1 trials summed over the 21 admitted sessions, read from the trial tables; error trials add to it"),
    "dandi_001187": (6010, "correct trials summed over the 46 admitted sessions, read from the trial tables; error trials add to it"),
    "dandi_000673": (653, "correct trials summed over the 5 retained sessions, read from the trial tables; error trials add to it"),
    "dandi_000574": (1827, "trials with a resolved latency over 37 sessions, from the latency census"),
    "dandi_000004": (8700, "100 admitted trials in each of 87 sessions"),
    "ds004752": (3336, "trials with a latency over 68 sessions, from the latency census"),
    "kai_miller_nback": (700, "100 trials for one patient and 200 for each of three, from the corpus's own admission filter"),
    "ds006848": (6000, "200 trials for each of 30 patients"),
    "haslacher_clam_tacs": (17749, "8194 active-site and 9555 control-site trials, from the stimulation census"),
    "alagapan_phase_stimulation": (310, "90, 100 and 120 trials for the three patients"),
    "ds005034": (None, "no trial-level behaviour is available locally"),
    "panichello_2024": (None, "only the median (581) and range (376 to 942) of trials per session were read, not the total"),
    "watters_2026": (None, "the 60094-trial census covers 47 session dates and the cluster count is the 41 that pass the quality filter"),
    "inagaki_alm5": (9448, "trials over 23 sessions, from the latency census"),
    "pfc3": (None, "resampled pseudo-population with no fixed trial total"),
    "pfc4": (11401, "trials over 196 sessions, from the latency census"),
}
LEVEL_ONE_TRIALS = (1300, "about 1,300 trials in the shared patients, as stated in the analysis specification")

LINK_ENTRY_NOTES = {
    "dandi_000673": "sessions that duplicate dandi_001187 recordings are dropped; 5 patients remain",
    "pfc3": "the population state exists only as a pseudo-population; the Gaussian-process factor analysis needs simultaneously recorded trials and is expected to be not_estimable at fit",
}

HYPOTHESES = {
    "H1": dict(links=[4], primary_measure="content: out-of-fold attractor separation = mean distance of a held-out delay state to the other classes' means minus its distance to its own class mean, in units of the pooled within-class standard deviation; correct trials; lowest load, plus the first item at load 3 where the task has it. Link 4: tuning contrast (F) against own-attractor distance (P)",
               primary_model="separation against the within-session label-shuffle null (1000 draws), per session, averaged within patient, patient-resampled interval. Link 4: within-session partial correlation, cluster bootstrap"),
    "H2": dict(links=[6], primary_measure="P = out-of-fold distance to own attractor during the delay",
               primary_model="linear mixed model, outcome on P, session intercept nested in patient (mixed logistic for accuracy)"),
    "H3": dict(links=[7, 5], primary_measure="link 7: the H2 model with F added (population rate, selective-unit rate, trial spike count); the quantity is the P coefficient after F. Link 5: tuning contrast on outcome",
               primary_model="same model family as H2"),
    "H4": dict(links=[6, 10], primary_measure="the H2 measure on each recording tier's state; link 10: P x region (or tier) interaction",
               primary_model="H2 model per region or tier; difference between regions by cluster bootstrap"),
    "H5": dict(links=[1, 2], primary_measure="link 1: firing after stimulation; link 2: displacement = distance of the state after stimulation onset from the event's own pre-onset level, and time to return = first bin from which the stimulated-minus-control displacement interval contains 0 for 3 consecutive bins",
               primary_model="analysis-of-covariance mixed model: post value on pre value, stimulation, and their nesting intercepts (animal or subject, session, unit or channel); list intercept where stimulation is randomised by list; matched non-stimulated flagged events where stimulation is state-triggered"),
    "H6": dict(links=[3, 8], primary_measure="link 3: recall or correctness; link 8: stimulation x pre-stimulation P",
               primary_model="mixed logistic with the design's randomisation unit as intercept; link 8 p from the design-based re-randomisation of that unit where stimulation was randomised, matched comparison where it was state-triggered"),
    "H7": dict(links=[10, 11], primary_measure="link 10: the link 2 and link 3 effects by stimulation site (individual-subject cortical parcellation region); link 11: within-subject slope of the link 2 and link 3 effects on amplitude",
               primary_model="mixed model with subject intercept; dose slope only with at least 6 subjects having 2 or more amplitude levels"),
    "H8": dict(links=[9], primary_measure="product of the stimulation-to-P coefficient and the P-to-outcome coefficient given stimulation",
               primary_model="within-subject, cluster bootstrap of the product"),
    "H9": dict(links=[], primary_measure="delivered: results/representation_benchmark_co_primary.json",
               primary_model="co-primary rule applied to the representation benchmark; a reader analysis with no per-corpus cell"),
    "H10": dict(links=[12], primary_measure="level 1: shrinkage fraction 1 - b(V given U) / b(V) of the field measure V given the unit measure U on log reaction time, both directions",
                primary_model="mixed model with session intercept, adjusted for unit rate, field rate proxy and set size; patient-cluster bootstrap"),
}

COMMON_RULES = ("Predictors and outcomes are demeaned within session. Human inference clusters on patient; animal inference "
                "clusters on session and reports animals. Reaction time is log-transformed, correct trials only. Accuracy uses "
                "correct and error trials. Each cell reports the native-unit estimate from its model and the common-scale "
                "within-cluster partial correlation. Where a corpus has reaction time, reaction time is the primary outcome and "
                "accuracy is the declared secondary; where it has no reaction time, accuracy is primary; where it has a "
                "continuous report error, that error replaces accuracy. Covariates in every behaviour model: load or set size "
                "(where the task varies it) and trial index within session.")

SCOPE_STATEMENT = (
    "Several links already have exploratory results that have been read. The freeze therefore does not claim the tests were "
    "fixed before any look at the data. It fixes them before the chain analyses run under the corrected unit inclusion, the "
    "corrected de-duplication and the benchmarked state spaces. Every exploratory result stays labelled exploratory and is "
    "placed beside the frozen primary, never in its place."
)

STATE_SPACE_COMBINATION = (
    "Each corpus x link cell stores one estimate per space (estimate, 95% interval, p, counts). Nothing is selected by "
    "outcome. The sentence that reports a cell gives the demixed-principal-components number first, the Gaussian-process "
    "factor analysis number second, and the range over any further spaces. A statement about a link is made for the "
    "population state only where the estimates in all spaces of that cell have the same sign and every interval lies on the "
    "same side of zero; otherwise the statement names the space. Spaces count as secondary tests for multiplicity. All fits "
    "are inside training folds. A space whose fit fails on an entry is not_estimable for that entry, with the count "
    "reported; it is not replaced by another space."
)

UNIT_INCLUSION = {
    "published_reproduction": "firing rate at least 0.05 Hz averaged over the whole task; no minimum number of units per session or region; report how many units this admits beside the count under the previous rule (0.1 Hz in the maintenance window, 8-unit region-session minimum)",
    "simultaneous_populations": "the 0.05 Hz whole-task floor plus at least 8 units in the region-session",
    "linked_duplicate_sessions": "dandi_000673 sessions that duplicate dandi_001187 patients are dropped by provenance.linked_duplicate_000673_sessions in every analysis",
    "selection": "category-selective units by one-way analysis of variance plus the right-tailed permutation test in 200-1200 ms after stimulus onset, chosen inside training folds and evaluated on held-out trials; identity-selective units only where each identity has at least 6 trials in a session, otherwise not_askable",
}

UNIT_SET_GRID = [
    {"family": "maintenance firing, preferred vs non-preferred (published reproduction)", "primary": "category_selective",
     "also_reported": ["all_units", "count_matched_random_unit_null_1000_draws"]},
    {"family": "pseudo-population demixed principal components and distance to attractor (published reproduction)",
     "primary": "all_recorded_units_of_the_region_set", "also_reported": ["category_selective_subset", "count_matched_random_unit_null"]},
    {"family": "F: population rate", "primary": "all_units", "also_reported": []},
    {"family": "F: selective-unit rate and tuning contrast", "primary": "category_selective",
     "also_reported": ["identity_selective_where_askable"]},
    {"family": "P in simultaneous populations (links 2, 4, 6, 7, 8, 9)", "primary": "all_units",
     "also_reported": ["category_selective_subset_where_at_least_8_selected_units_remain_in_the_training_fold_else_not_estimable"]},
    {"family": "link 1, stimulation to firing", "primary": "category_selective", "also_reported": ["all_units"]},
    {"family": "representation benchmark", "primary": "all_units", "also_reported": []},
    {"family": "field potentials", "primary": "all_contacts_of_the_region_all_bands", "also_reported": []},
]

MULTIPLICITY = {
    "primary": "one primary per hypothesis per corpus, uncorrected, reported with its p as a number",
    "secondary": "Benjamini-Hochberg q within hypothesis x corpus across measures, spaces, unit sets and outcomes; both p and q are stored",
    "labels": "no result is labelled by a threshold",
    "declared_secondaries": ["deviation along the correct-versus-error axis", "state speed", "one-step prediction error",
                             "time constants", "accuracy where reaction time is primary", "every additional state space",
                             "the non-primary unit set", "causal and chronological read-out forms"],
}

COMMON_SCALE = {
    "estimate": "within-cluster partial correlation r between predictor and outcome after demeaning within session and regressing out the declared covariates; accuracy coded 0/1; Fisher z; 95% interval by cluster bootstrap (patient for humans, session for animals; 5000 draws)",
    "minimum_detectable_correlation": "r = tanh(Z / sqrt(n - 3)) with Z = statistics.Z_80_POWER (two-sided 0.05, power 0.80), n = cluster count; n below 4 is not_estimable; computed and stored before the test",
}

POOLING_RULE = {
    "links_poolable": sorted(POOLABLE_LINKS),
    "condition": "across corpora sharing species, recording type and task class; a patient present in two corpora counts once",
    "method": "random-effects meta-analysis on Fisher z with the Hartung-Knapp adjustment; report between-corpus variance, I-squared and the 95% prediction interval",
    "fewer_than_three_corpora": "side by side, no pooled row",
    "stimulation": "stimulation results are never pooled across corpora or recording types; the stimulation-site effect (H7 link 10) is a stimulation result and is never pooled",
    "tiers": "human, macaque and mouse are never pooled with each other",
    "shared_patients": [
        {"corpora": ["dandi_001187", "dandi_000673"], "n_patients": 31, "handling": "000673 sessions that duplicate 001187 are dropped"},
        {"corpora": ["dandi_000574", "ds004752"], "n_patients": 9, "handling": "different recording types; never pooled with each other; counted once within a recording type"},
    ],
}


def split_capability(raw: str) -> tuple[str, str, bool]:
    star = raw.endswith("*")
    raw = raw.rstrip("*")
    letter = raw[0]
    reasons = raw[raw.find("(") + 1:raw.find(")")] if "(" in raw else ""
    return letter, reasons, star


def capability_for(corpus: str, link: int) -> dict:
    if link == 12:
        if corpus in LEVEL_ONE_CORPORA:
            return {"code": "Y", "reasons": [], "text": "simultaneous unit and depth field recordings in 9 shared patients; level 1 test",
                    "source": "analysis specification, cross-recording-type comparison, level 1"}
        return {"code": "N", "reasons": [], "text": "level 1 needs the same patients and trials recorded in both recording types; level 2 is reported side by side under link 6 and is not part of this test",
                "source": "analysis specification, cross-recording-type comparison, levels 1 and 2"}
    raw = CAPABILITY[corpus][link - 1]
    letter, reasons, star = split_capability(raw)
    codes = [c for c in reasons.split(",") if c]
    text = "; ".join(REASON_TEXT[c] for c in codes if c in REASON_TEXT)
    if letter == "N" and not codes:
        text = "no trial-level correctness field was found in the released trial structure"
    if star:
        text += "; the delay-period optogenetic arm lies outside the stimulation definition"
    return {"code": letter, "reasons": codes, "text": text, "source": f"capability table row {corpus}, link {link} ({raw})"}


def state_spaces_for(corpus: str) -> dict:
    bucket = CORPORA[corpus]["state_space_bucket"]
    if bucket == "dandi_000574":
        return {"units": DESIGN_SPACES["human_single_units_simultaneous"], "depth_field_potentials": DESIGN_SPACES["human_depth_field_potentials"],
                "scalp_eeg": UNIT_SPACES}
    if bucket == "ds004752":
        return {"depth_field_potentials": DESIGN_SPACES["human_depth_field_potentials"], "scalp_eeg": UNIT_SPACES}
    if bucket == "human_single_units_simultaneous":
        return {"units": DESIGN_SPACES[bucket]}
    if bucket == "human_depth_field_potentials":
        return {CORPORA[corpus]["recording_type"]: DESIGN_SPACES[bucket]}
    return {CORPORA[corpus]["recording_type"]: UNIT_SPACES}


def outcome_for(corpus: str, link: int) -> dict:
    facts = CORPORA[corpus]
    extension = []
    if facts["rt"] == "derived_by_delivered_code":
        extension.append("response_time: computed by a separate latency module, not by the shared session iterator")
    if facts["accuracy"] == "in_raw_not_loaded":
        extension.append("error_trials: dropped by the shared session iterator; read by a separate maintenance-link script")
    out = {"primary": None, "secondary": [], "requires_loader_extension": extension}
    if facts["rt"] == "different_construct":
        out.update(primary="recall of the word (0/1)",
                   secondary=["recall-period word-production latency (never pooled with probe-locked latencies)"])
    elif facts["rt"] == "unresolved":
        out.update(primary="accuracy (correct and error trials)",
                   secondary=["log reaction time (correct trials only), reported per animal and never pooled across the two animals"])
    elif facts["rt"] in ("loaded", "derived_by_delivered_code"):
        graded = facts["continuous_error"]
        accuracy = graded or ("accuracy (correct and error trials)" if facts["accuracy"] != "absent" else None)
        out.update(primary="log reaction time (correct trials only)", secondary=[accuracy] if accuracy else [])
    elif facts["continuous_error"]:
        out.update(primary=facts["continuous_error"] + " (replaces accuracy)")
    elif facts["accuracy"] != "absent":
        out.update(primary="accuracy (correct and error trials)")
    if facts["rt"] == "absent" and corpus == "haslacher_clam_tacs":
        out["note"] = "reaction time is not available: it is not in the staged files"
    return out


def unit_sets_for(corpus: str, hypothesis: str, link: int) -> dict:
    facts = CORPORA[corpus]
    if facts["recording_type"] != "units":
        return {"primary": "all_contacts_of_the_region_all_bands", "also_reported": []}
    identity_status = facts["identity"].split(":")[0] if facts["identity"] else None
    identity_also = ["identity_selective"] if identity_status in ("askable", "askable_per_session") else []
    pseudo = not facts["simultaneous"]
    if link == 1:
        return {"primary": "category_selective", "also_reported": ["all_units"]}
    if link in (4, 5):
        primary = "category_selective (F: selective-unit rate and tuning contrast); all_units (P)" if link == 4 else "category_selective (tuning contrast); all_units (population rate)"
        if pseudo:
            primary = "all_recorded_units_of_the_region_set (as published)"
        return {"primary": primary, "also_reported": ["all_units", "category_selective_subset"] + identity_also}
    if pseudo:
        return {"primary": "all_recorded_units_of_the_region_set (as published)",
                "also_reported": ["category_selective_subset", "count_matched_random_unit_null"]}
    return {"primary": "all_units", "also_reported": ["category_selective_subset (at least 8 selected units remain in the training fold, else not_estimable)"]}


def trial_set_for(corpus: str, hypothesis: str, link: int) -> str:
    facts = CORPORA[corpus]
    if hypothesis == "H1" and link == 4:
        return "correct trials; lowest load" + ("; plus the first item at load 3" if 3 in facts["load_levels"] else "")
    return facts["trial_note"]


def test_for(hypothesis: str, link: int, corpus: str) -> str:
    facts = CORPORA[corpus]
    design = facts["design_class"]
    base = f"{HYPOTHESES[hypothesis]['primary_model']}"
    if link in STIMULATION_LINKS or (hypothesis == "H7" and link == 10):
        suffix = {
            "randomised_by_trial": "randomisation unit: trial",
            "randomised_by_list": "randomisation unit: list; list intercept; effective n is lists; link 8 p from design-based re-randomisation of lists",
            "state_triggered": "state-triggered stimulation: comparison against non-stimulated trials matched on the pre-stimulation state, which is also a covariate; the selection bias is reported; any association with the outcome is not a stimulation effect",
            "varies_between_sessions": "stimulation varies between sessions: within-subject model with a subject intercept; effective n is subjects",
        }.get(design, "no stimulation")
        base += "; " + suffix
    if (hypothesis, link) == ("H4", 10) and facts["recording_type"] == "units":
        base = "recording-region contrast (hippocampus against amygdala) of the link 6 estimate, difference by cluster bootstrap"
    if link in (4, 5, 6, 7, 10) and not (hypothesis == "H7"):
        base += "; within-session trial-level model, predictors demeaned within session"
    return base


STATE_MEASURE = {
    "class": "out-of-fold distance of the trial's delay state to the attractor (class mean) of its own class",
    "label_free": "out-of-fold distance of the trial's state from the training-fold mean state (one class); the measure is label-free",
}


def corpus_cell(hypothesis: str, link: int, corpus: str, experiment: str | None = None) -> dict:
    facts = dict(CORPORA[corpus])
    trials, trial_source = TRIAL_COUNTS[corpus] if corpus != "wolff_eeg_impulse" else (None, None)
    if experiment:
        spec = facts["experiments"][experiment]
        facts.update(cluster_n=spec["cluster_n"], cluster_source=spec["cluster_source"])
        trials, trial_source = spec["n_trials"], spec["trial_source"]
    cap = capability_for(corpus, link)
    has_stimulation = facts["design_class"] not in ("none",)
    n_clusters = facts["cluster_n"]
    if link == 12 and corpus in LEVEL_ONE_CORPORA:
        n_clusters, (trials, trial_source) = LEVEL_ONE_COUNT, LEVEL_ONE_TRIALS
    mdc = minimum_detectable_correlation(n_clusters, trials)
    mdc["trial_count_source"] = trial_source if trials is not None else None
    mdc["trial_bound_reason"] = None if trials is not None else trial_source
    cell = {
        "id": f"{hypothesis}|{corpus}|link{link}" + (f"|{experiment}" if experiment else ""), "hypothesis": hypothesis,
        "corpus": corpus, "experiment": experiment, "link": link,
        "link_name": LINK_NAMES[link], "capability": cap["code"], "capability_reasons": cap["reasons"],
        "capability_text": cap["text"], "capability_source": cap["source"], "matrix_entry": facts["entry"],
        "species": facts["species"], "recording_type": facts["recording_type"], "task_class": facts["task_class"],
        "evidence_tier": {"human": "human_primary", "macaque": "macaque_supplementary", "mouse": "mouse_supplementary"}[facts["species"]],
    }
    refusal = None
    if cap["code"] == "N":
        refusal = cap["text"]
    if (hypothesis, link) == ("H4", 10):
        if corpus in REGION_LABELLED_UNIT_CORPORA:
            cell.update(capability="P", capability_source=f"{cap['source']}, overridden by the loader: units carry hippocampus and amygdala labels (resolve_unit_regions)")
            refusal = None
        if corpus == "macaque_pfc_microstimulation":
            refusal = "one recording area (dorsolateral prefrontal cortex by design); site effects belong to the stimulation-site hypothesis"
    if hypothesis == "H7" and link == 10 and not has_stimulation:
        cell["capability_reasons"] = ["a"]
        refusal = REASON_TEXT["a"] + "; the stimulation-site effect needs stimulation"
    if hypothesis == "H7" and link == 11 and corpus == "alagapan_phase_stimulation":
        refusal = ("fewer than 6 patients (3 patients, one stimulation frequency each); the dose slope needs at least 6 "
                   "subjects with 2 or more amplitude levels")
    if (hypothesis, link) == ("H1", 4) and facts["label_kind"] != "content":
        refusal = f"the only label is {facts['class_label']}, which is a task condition and not a per-trial content label"
    if (facts["recording_type"] == "units" and link in LINKS_USING_POPULATION_STATE
            and facts.get("max_units_per_session", MIN_UNITS_PER_REGION_SESSION) < MIN_UNITS_PER_REGION_SESSION and refusal is None):
        refusal = (f"the population state needs at least {MIN_UNITS_PER_REGION_SESSION} units in the region-session and this corpus records "
                   f"at most {facts['max_units_per_session']} units per session")
    if refusal:
        cell.update(status="not_askable", reason=refusal)
        return finish_cell(cell, corpus, hypothesis, link, mdc, facts)
    cell.update(status="frozen", reason=None)
    return finish_cell(cell, corpus, hypothesis, link, mdc, facts)


def finish_cell(cell: dict, corpus: str, hypothesis: str, link: int, mdc: dict, facts: dict) -> dict:
    active = cell["status"] != "not_askable"
    experiment = cell["experiment"]
    pooled_link = link in POOLABLE_LINKS and not (hypothesis == "H7")
    n = facts["cluster_n"]
    if link == 12 and corpus in LEVEL_ONE_CORPORA:
        n = LEVEL_ONE_COUNT
    uses_state = link in LINKS_USING_POPULATION_STATE
    design = facts["design_class"] if (link in STIMULATION_LINKS or (hypothesis == "H7" and link in (10, 11))) else "within_session_association"
    if link == 12:
        design = "within_session_association"
    outcome = outcome_for(corpus, link) if (hypothesis, link) in OUTCOME_CELLS else {"primary": None, "secondary": [], "requires_loader_extension": [],
                                                                                     "note": "this link has no behavioural outcome"}
    unit_set = unit_sets_for(corpus, hypothesis, link) if active else None
    notes = [LINK_ENTRY_NOTES.get(corpus)]
    if active and (hypothesis, link) == ("H3", 5) and facts["label_kind"] != "content":
        unit_set = {"primary": "all_units (population rate; no tuning contrast is defined without a per-trial content label)", "also_reported": []}
    if active and (hypothesis, link) == ("H7", 11):
        notes.append(DOSE_LEVEL_FACTS.get(corpus))
    if active and corpus == "alagapan_phase_stimulation" and uses_state:
        notes.append("the state-space set was benchmarked on depth contacts only; 3 patients, descriptive")
    if active and corpus == "pfc4" and link == 5:
        notes.append("reaction time is reported per animal and never pooled across the two animals")
    if active and (hypothesis, link) == ("H5", 2):
        state_measure = "displacement of the state after stimulation onset from the event's own pre-onset level"
    elif active and uses_state and facts["label_kind"] == "none":
        state_measure = STATE_MEASURE["label_free"]
    else:
        state_measure = STATE_MEASURE["class"] if (active and uses_state) else None
    cell.update(
        design_class=design if active else None,
        test=test_for(hypothesis, link, corpus) if active else None,
        outcome=outcome if active else None,
        class_label=facts["class_label"] if active else None,
        class_label_kind=facts["label_kind"] if active else None,
        state_measure=state_measure,
        load_levels=facts["load_levels"] if active else None,
        covariates=(["load_or_set_size"] if facts["load_levels"] else []) + ["trial_index_within_session"] if active else None,
        unit_set=unit_set,
        identity_selective=dict(zip(("status", "evidence"), facts["identity"].split(":", 1))) if (active and facts["identity"] and link in LINKS_USING_FIRING) else None,
        trial_set=trial_set_for(corpus, hypothesis, link) if active else None,
        state_spaces=(state_spaces_for(corpus) if uses_state else {}) if active else None,
        state_space_note=(NO_BENCHMARK_NOTE if (uses_state and facts["state_space_bucket"] == "no_benchmark") else None) if active else None,
        cluster_unit=facts["cluster_unit"] if active else None,
        cluster_n=n if active else None,
        cluster_source=("9 patients shared by dandi_000574 and ds004752" if (link == 12 and corpus in LEVEL_ONE_CORPORA) else facts["cluster_source"]) if active else None,
        animals=facts["animals"] if active else None,
        minimum_detectable_correlation=mdc if active else None,
        note="; ".join(filter(None, notes)) or None if active else None,
        pooling_group=(f"{hypothesis}|{facts['species']}|{facts['recording_type']}|{facts['task_class']}|link{link}" + (f"|{experiment}" if experiment else "") if (active and pooled_link) else "never_pooled"),
    )
    return cell


def build_cells() -> list[dict]:
    lock = json.loads(DATA_LOCK_PATH.read_text())["corpora"]
    cells = []
    for hypothesis, link in HYPOTHESIS_LINKS:
        for corpus in sorted(lock):
            for experiment in CORPORA[corpus].get("experiments", {None: None}):
                cells.append(corpus_cell(hypothesis, link, corpus, experiment))
    return cells


def benchmark_state_spaces(co_primary: dict) -> dict:
    out = {}
    for key, (corpus, region_cell) in BENCHMARK_CELLS.items():
        chosen = {DPCA}
        for row in co_primary["co_primary"]:
            if row["corpus"] == corpus and row["region_cell"] == region_cell and row["co_primary"] is True:
                chosen.add(row["candidate"])
        out[key] = sorted(chosen)
    return out


def check_state_spaces_against_benchmark(co_primary: dict) -> None:
    derived = benchmark_state_spaces(co_primary)
    for key, spaces in derived.items():
        if spaces != sorted(DESIGN_SPACES[key]):
            raise SystemExit(f"state-space list for {key} differs from the benchmark: {spaces} vs {sorted(DESIGN_SPACES[key])}")


def pooling_groups(cells: list[dict]) -> dict:
    groups: dict[str, list[str]] = {}
    for cell in cells:
        if cell["status"] != "not_askable" and cell["pooling_group"] != "never_pooled":
            groups.setdefault(cell["pooling_group"], []).append(cell["corpus"])
    return {name: {"corpora": members, "pooled_row": len(members) >= 3,
                   "rule": "pooled row only with at least 3 corpora after exclusions and patient de-duplication; otherwise side by side"}
            for name, members in sorted(groups.items())}


def render_table(cells: list[dict]) -> str:
    lines = ["# Analysis freeze: cell tables", "",
             "Generated from `analysis_freeze.json`. Status is frozen or not_askable.", ""]
    for link in range(1, 13):
        rows = [c for c in cells if c["link"] == link]
        if not rows:
            continue
        lines += [f"## Link {link}: {LINK_NAMES[link]}", "",
                  "| hypothesis | corpus | status | capability | primary outcome | unit set | state spaces | cluster | n | min. detectable r (cluster/trial bound) | pooling |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in rows:
            if c["status"] == "not_askable":
                lines.append(f"| {c['hypothesis']} | {c['corpus']}{' ' + c['experiment'] if c['experiment'] else ''} | not_askable | {c['capability']} | | | | | | | {c['reason']} |")
                continue
            spaces = "; ".join(f"{k}: {', '.join(v)}" for k, v in c["state_spaces"].items()) or "none used"
            bound = c["minimum_detectable_correlation"]
            mdc = "/".join("" if bound[k] is None else f"{bound[k]:.3f}" for k in ("cluster_bound", "trial_bound"))
            outcome = c["outcome"]["primary"] or c["outcome"].get("note") or ""
            lines.append(f"| {c['hypothesis']} | {c['corpus']}{' ' + c['experiment'] if c['experiment'] else ''} | {c['status']} | {c['capability']} | {outcome} | "
                         f"{c['unit_set']['primary']} | {spaces} | {c['cluster_unit']} | {c['cluster_n']} | "
                         f"{mdc} | {c['pooling_group']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    co_primary = json.loads(CO_PRIMARY_PATH.read_text())
    check_state_spaces_against_benchmark(co_primary)
    lock = json.loads(DATA_LOCK_PATH.read_text())
    missing = sorted(set(lock["corpora"]) - set(CORPORA))
    if missing:
        raise SystemExit(f"corpora without per-corpus facts: {missing}")
    cells = build_cells()
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    freeze = {
        "version": "analysis_freeze_v1",
        "scope_statement": SCOPE_STATEMENT,
        "state_spaces": {
            "reference": DPCA,
            "by_recording_type": {
                "human_single_units_simultaneous_populations": {"benchmark_cell": "human_single_units / regions_pooled_by_patient",
                                                               "spaces": DESIGN_SPACES["human_single_units_simultaneous"]},
                "human_single_units_pseudo_populations": {"benchmark_cell": None, "spaces": DESIGN_SPACES["human_single_units_pseudo_population"],
                                                         "reason": "trial-level latent models need simultaneously recorded trials"},
                "human_depth_field_potentials": {"benchmark_cell": "dandi_000574 / pooled", "spaces": FIELD_BENCHMARK_SPACES},
                "no_benchmark": {"benchmark_cell": None, "spaces": UNIT_SPACES, "note": NO_BENCHMARK_NOTE,
                                 "applies_to": "macaque and mouse units, surface electrocorticography, scalp EEG, RAM contacts"},
            },
            "derived_from_benchmark": benchmark_state_spaces(co_primary),
            "combination": STATE_SPACE_COMBINATION,
        },
        "unit_inclusion": UNIT_INCLUSION,
        "unit_set_grid": UNIT_SET_GRID,
        "hypotheses": HYPOTHESES,
        "common_rules": COMMON_RULES,
        "cells": cells,
        "cell_counts_by_status": counts,
        "multiplicity": MULTIPLICITY,
        "common_scale": COMMON_SCALE,
        "pooling": {**POOLING_RULE, "groups": pooling_groups(cells)},
        "inputs": {
            "data_lock_sha256": sha256_file(DATA_LOCK_PATH), "capability_matrix_sha256": sha256_file(MATRIX_PATH),
            "analysis_specification_sha256": sha256_file(SPEC_PATH), "co_primary_artifact_sha256": sha256_file(CO_PRIMARY_PATH),
        },
    }
    FREEZE_PATH.write_text(canonical_json(freeze))
    TABLE_PATH.write_text(render_table(cells))
    digest = sha256_file(FREEZE_PATH)
    section = (f"\n{README_HEADING}\n\n`analysis_freeze.json` fixes the test, outcome, unit set, state spaces and cluster count of each "
               f"hypothesis x corpus x link cell before the chain analyses run. Its SHA-256 is:\n\n`{digest}`\n\n"
               "Chain artifacts record this content hash.\n")
    readme = README_PATH.read_text()
    if README_HEADING in readme:
        readme = readme[:readme.index("\n" + README_HEADING)]
    README_PATH.write_text(readme.rstrip("\n") + "\n" + section)
    write_code_identity_record(ROOT, Path(__file__), [str(FREEZE_PATH.relative_to(ROOT)), str(TABLE_PATH.relative_to(ROOT))])
    print(json.dumps({"sha256": digest, "cell_counts_by_status": counts}))


if __name__ == "__main__":
    main()
