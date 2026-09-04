"""Corpus admission audit: every trial-admitting loader in this repository,
what it admits, and which analyses its admission rule forecloses.

For each loader the audit records the file, function, line span, the
admission predicate verbatim as extracted from source at run time, a
structured verdict on whether that predicate conditions admission on an
outcome variable (correctness, accuracy, recall, confidence, reaction
time, report deviation), and what outcome-conditioned analyses the rule
therefore removes variance from. A repository-wide sweep then verifies
the registry is exhaustive: every file that touches a trial or behaviour
table is scanned for admission expressions, and any site outside the
registry is audited by the same classifier rather than dropped.

On top of the loader audit it builds an askability table: one row per
registered corpus in config/datasets.json, marking whether the corpus can
support error-trial behavioural contrasts, graded-report analyses,
stimulation-response mapping, within-maintenance intervention, and
cross-instrument pairing -- each cell yes / no / not-as-loaded, with the
path:line or measured count that justifies the mark.

Decision rules for the summary branches are declared below, before any
number is evaluated. Trial-discard counts are measured from header-level
reads only (trial-table columns, behaviour CSV flags); no spike bodies or
signal arrays are loaded.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _entry in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

ARTIFACT_PATH = _REPO_ROOT / "results" / "corpus_admission_audit.json"
DATASETS_CONFIG = _REPO_ROOT / "config" / "datasets.json"

# ---------------------------------------------------------------------------
# Pre-declared summary branches. Each rule states its threshold before any
# count is evaluated; every emitted branch carries its effect size beside
# the label.
# ---------------------------------------------------------------------------
PRE_DECLARED_BRANCHES = [
    {
        "name": "outcome_filtering_is_a_defect_class_not_a_one_off",
        "rule": "emitted when the number of loaders whose per-trial admission "
                "predicate conditions on an outcome variable is >= 3; otherwise "
                "emitted as outcome_filtering_is_confined_to_<n>_loaders",
        "threshold": "n_trial_level_outcome_filter_loaders >= 3",
    },
    {
        "name": "outcome_filtering_extends_beyond_the_human_spine_iterators",
        "rule": "emitted when at least one trial-level outcome-conditioned "
                "admission site lies outside src/corpus_sessions.py's three "
                "human iterators; otherwise emitted as "
                "outcome_filtering_confined_to_the_human_spine_iterators",
        "threshold": "n_outcome_filter_sites_outside_corpus_sessions >= 1",
    },
    {
        "name": "all_three_human_spine_iterators_discard_error_trials",
        "rule": "emitted when iter_dandi_000469, iter_dandi_001187 and "
                "iter_dandi_000574 all carry a trial-level outcome filter; "
                "otherwise emitted as {k}_of_3_human_spine_iterators_discard_error_trials",
        "threshold": "human_spine_iterators_flagged == 3 of 3",
    },
    {
        "name": "per-corpus alternate-loader dependence",
        "rule": "for each corpus whose shared or dominant tensor loader applies a "
                "trial-level outcome filter while at least one other existing "
                "loader admits trials without conditioning on outcome, emit "
                "corpus_<id>_error_contrasts_require_an_alternate_loader",
        "threshold": "flagged shared loader AND >= 1 outcome-preserving reader",
    },
    {
        "name": "per-corpus question coverage",
        "rule": "for each registered corpus emit corpus_<id>_askable_for_<k>_of_5_questions, "
                "counting yes cells among error-trial contrasts, graded report, "
                "stimulation-response mapping, within-maintenance intervention and "
                "cross-instrument pairing",
        "threshold": "descriptive; no decision threshold",
    },
    {
        "name": "graded_report_coverage",
        "rule": "always emitted, as graded_report_supported_by_<n>_of_<N>_corpora, "
                "counting corpora whose graded-report cell is yes",
        "threshold": "descriptive; no decision threshold",
    },
    {
        "name": "within_maintenance_intervention_coverage",
        "rule": "always emitted, as within_maintenance_intervention_supported_by_<n>_of_<N>_corpora",
        "threshold": "descriptive; no decision threshold",
    },
    {
        "name": "cross_instrument_pairing_coverage",
        "rule": "always emitted, as cross_instrument_pairing_supported_by_<n>_of_<N>_corpora",
        "threshold": "descriptive; no decision threshold",
    },
    {
        "name": "zero_drop_reconciliation_holds",
        "rule": "the artifact asserts loaders_found == audited + skipped_with_reason; "
                "a violation raises instead of emitting a branch",
        "threshold": "exact equality",
    },
]

# Askability cell rule, declared before the table is built:
#   "yes"           -- at least one existing repo loader admits this corpus's
#                      trials without conditioning on an outcome variable AND
#                      carries the variable the question needs;
#   "no"            -- structurally absent, stated as "this corpus does not
#                      record X" (never a power statement);
#   "not-as-loaded" -- the corpus plausibly records or could support the
#                      question but every existing admitting loader blocks it
#                      or no loader wires the needed variable yet.
ASKABILITY_CELL_RULE = (
    "yes = some existing loader admits trials unconditioned on outcome and carries the needed "
    "variable; no = the corpus does not record the needed quantity (structural reason stated); "
    "not-as-loaded = recorded or supportable but blocked by every existing admitting loader, or "
    "not yet wired into any loader"
)

# ---------------------------------------------------------------------------
# Classifier vocabulary
# ---------------------------------------------------------------------------
OUTCOME_TOKENS = (
    "response_accuracy", "response_acc", "accuracy", "isCorr", "is_correct", "correct",
    "recalled", "recall", "reaction_time", "report_deviation", "confidence",
    "n_error", "n_correct",
)
_OUTCOME_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in OUTCOME_TOKENS) + r")\b"
)
_ADMISSION_LHS = r"(?:keep|mask|good|valid\w*|eligible|admitted|usable|selected)"
ADMISSION_ASSIGN_RE = re.compile(rf"^\s*{_ADMISSION_LHS}\s*=\s*(.+)$")
OUTCOME_SUBSCRIPT_RE = re.compile(
    rf"^\s*[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*=\s*\w+\[\s*~?\s*({_ADMISSION_LHS}|correct|is_corr)\b[^\]]*\](?:\s*,.*)?$"
)
SESSION_GATE_RE = re.compile(
    r"(?:\b(?:np\.mean\(|\.mean\(\)).*" + _OUTCOME_RE.pattern + r")"
    r"|MIN_SESSION_ACCURACY"
    r"|\b(?:n_error|n_correct)\s*(?:>=|<=|>|<)"
)
# A line qualifies as an admission-relevant extraction candidate when it either
# matches one of the two assignment shapes above or names an outcome variable
# beside a comparison, a skip, or a subscript (session gates and label-carrying
# subset lines).
EXTRACT_HINT_RE = re.compile(
    r"<|>|>=|<=|\bskip\b|\bcontinue\b|\breturn\b|\[\s*~?\s*\w+\s*\]|MIN_SESSION_ACCURACY"
)

INVISIBLE_TO_OUTCOME_FILTERS = [
    "correct-versus-error neural contrast inside the admitted epoch",
    "predicting trial outcome from the maintenance/delay state",
    "outcome-conditioned drift or geometry comparison",
    "regressing a graded outcome measure against delay-period activity",
]


def _contains_outcome_token(text: str) -> bool:
    return bool(_OUTCOME_RE.search(text))


# A dict-key shape lookup such as s["is_correct"].shape mentions an outcome
# column without conditioning anything on its value.
_SHAPE_LOOKUP_RE = re.compile(r"\w+\[\"[^\"]+\"\]\.shape|\w+\['[^']+'\]\.shape")
_ISFINITE_WRAP_RE = re.compile(r"np\.isfinite\([^()]*\)|isclose\(")


def _rhs_conditions_on_outcome(rhs: str) -> bool | str:
    """True / "completeness" / False: does this right-hand side condition on
    outcome values, or only on their being defined?"""
    cleaned = _SHAPE_LOOKUP_RE.sub("", rhs)
    if not _OUTCOME_RE.search(cleaned):
        return False
    without_finite = _ISFINITE_WRAP_RE.sub("", cleaned)
    if not _OUTCOME_RE.search(without_finite):
        return "completeness"
    return True


def classify_predicate_lines(lines: list[tuple[int, str]], signature: str, body_text: str) -> dict:
    """Classify one loader's extracted admission lines.

    Returns classification, filters_on_outcome_variable, offending_expression
    and the verbatim lines kept. Classification precedence: an outcome-
    parameterised loader (macaque_pfc_microstimulation-style correct/error file pools) first, then
    direct per-trial outcome subscripting, then admission assignments whose
    right-hand side names an outcome variable, then session-level gates on a
    summary statistic of an outcome variable, then non-outcome admission.
    """
    offending = None
    session_gate_expr = None
    completeness_only = False

    if re.search(r"\bdef\s+\w+\s*\([^)]*\bcorrect\s*:\s*bool", signature) and re.search(
        r"[\"']correct[\"']\s+if\s+correct\s+else|[\"']error[\"']", body_text
    ):
        return {
            "classification": "outcome_stratified_loader",
            "filters_on_outcome_variable": True,
            "offending_expression": signature.strip(),
            "predicate_lines": lines,
        }

    for lineno, text in lines:
        assign = ADMISSION_ASSIGN_RE.match(text)
        sub = OUTCOME_SUBSCRIPT_RE.match(text)
        if sub:
            # Only the subscript index matters here: `x = x[good]` merely
            # carries a mask, while `spikes = spikes[correct]` conditions
            # admission on the outcome itself.
            if _contains_outcome_token(sub.group(1)):
                offending = text.strip()
                break
            continue
        if assign:
            condition = _rhs_conditions_on_outcome(assign.group(1))
            if condition is True:
                offending = text.strip()
                break
            if condition == "completeness":
                completeness_only = True
    if offending is None:
        # Session-level gates read as `if` comparisons rather than admission
        # assignments, so scan every extracted line for them once no per-trial
        # filter matched.
        for _, text in lines:
            if SESSION_GATE_RE.search(text) and _OUTCOME_RE.search(_SHAPE_LOOKUP_RE.sub("", text)):
                session_gate_expr = text.strip()
                break
    if offending is not None:
        return {
            "classification": "trial_level_outcome_filter",
            "filters_on_outcome_variable": True,
            "offending_expression": offending,
            "predicate_lines": lines,
        }
    if session_gate_expr is not None:
        return {
            "classification": "session_level_outcome_gate",
            "filters_on_outcome_variable": True,
            "offending_expression": session_gate_expr,
            "predicate_lines": lines,
        }
    has_admission_line = any(
        ADMISSION_ASSIGN_RE.match(text) or OUTCOME_SUBSCRIPT_RE.match(text) for _, text in lines
    )
    classification = (
        "non_outcome_admission" if has_admission_line else "outcome_preserved"
    ) if not completeness_only else "outcome_completeness_mask"
    return {
        "classification": classification,
        "filters_on_outcome_variable": False,
        "offending_expression": None,
        "predicate_lines": lines,
    }


# ---------------------------------------------------------------------------
# Loader registry. Each entry fixes where the loader lives and what it feeds;
# the predicate text itself is always extracted from source at run time so
# quoted code cannot drift from the code.
# ---------------------------------------------------------------------------
def _r(path: str, fn: str, corpora: list[str], grain: str, admits: str,
       invisible: list[str] | None = None) -> dict:
    return {
        "path": path, "function": fn, "corpora": corpora, "grain": grain,
        "admits": admits, "invisible_analyses": invisible or [],
    }


REGISTRY: list[dict] = [
    # -- src/corpus_sessions.py: the shared iteration layer -------------------
    _r("src/corpus_sessions.py", "iter_dandi_000469", ["dandi_000469"], "single_unit spikes + epoch onsets",
       "trials with load==1 AND correct response, per region subject to unit-count floors",
       ["load-1 correct-vs-error contrast", "outcome prediction from delay state"]),
    _r("src/corpus_sessions.py", "iter_dandi_001187", ["dandi_001187"], "single_unit spikes + epoch onsets",
       "trials with correct response (all loads), per region subject to unit-count floors",
       ["correct-vs-error contrast at any load", "accuracy-linked regression"]),
    _r("src/corpus_sessions.py", "iter_dandi_000574", ["dandi_000574"], "single_unit spikes + epoch onsets",
       "trials with no artifact flag AND correct response",
       ["correct-vs-error contrast", "artifact-free error-trial dynamics"]),
    _r("src/corpus_sessions.py", "iter_alm", ["inagaki_alm5"], "delay-epoch binned counts",
       "eligible control (unperturbed) trials of sessions passing arm/unit floors; response codes carried"),
    _r("src/corpus_sessions.py", "load_alm_raw_session", ["inagaki_alm5"], "delay-epoch binned counts, both arms",
       "trials in the analysed range with response code < 5 and delay >= window, split control/perturbation"),
    _r("src/corpus_sessions.py", "iter_watters", ["watters_2026"], "delay-epoch binned counts + behaviour columns",
       "every behavioural session date, loaded or refused with a reason"),
    _r("src/corpus_sessions.py", "load_watters_session", ["watters_2026"], "delay-epoch binned counts + behaviour columns",
       "completed trials whose delay window fits the cached trial vector; correctness and graded report carried"),
    # -- human single-unit pipeline drivers ----------------------------------
    _r("scripts/run_000469_pipeline.py", None, ["dandi_000469"], "single_unit PCA/CTG geometry",
       "all trials of sessions whose mean response_accuracy clears the prospective floor",
       ["between-session outcome contrasts including low-accuracy sessions"]),
    _r("scripts/run_001187_pipeline.py", "_process_session", ["dandi_001187"], "single_unit PCA/CTG geometry",
       "all trials of sessions whose mean response_accuracy clears the prospective floor",
       ["between-session outcome contrasts including low-accuracy sessions"]),
    _r("scripts/run_000673_pipeline.py", "_process_session", ["dandi_000673"], "single_unit PCA/CTG geometry",
       "all trials of sessions whose mean response_accuracy clears the prospective floor",
       ["between-session outcome contrasts including low-accuracy sessions"]),
    _r("scripts/run_000574_units_pipeline.py", "_process_session", ["dandi_000574"], "single_unit PCA/CTG geometry",
       "artifact-free trials (correct and error both retained) of sessions passing unit floors"),
    _r("scripts/run_boran_pipeline.py", "load_subject_sessions", ["dandi_000574"], "iEEG high-gamma epochs",
       "non-artifact trials whose epochs fit inside the recording"),
    _r("scripts/run_human_drift_spine_000469.py", "analyze_session", ["dandi_000469"], "single_unit drift spine",
       "sessions above the accuracy floor, restricted to repeated-item load-1 trials",
       ["between-session outcome contrasts", "error-trial drift"]),
    _r("scripts/run_human_drift_spine_000574.py", "analyze_session", ["dandi_000574"], "single_unit drift spine",
       "artifact-free trials restricted to set sizes supporting five folds"),
    _r("scripts/run_human_drift_spine_001187_000673.py", "analyze_unit_session", ["dandi_001187", "dandi_000673"],
       "single_unit drift spine", "sessions above the accuracy floor, all loads",
       ["between-session outcome contrasts"]),
    # -- macaque single-unit pipelines ---------------------------------------
    _r("scripts/run_panichello_pipeline.py", "analyze_session", ["panichello_2024"], "delay-epoch counts",
       "correct trials only (isCorr), after firing-rate unit QC",
       ["error-trial cue decoding", "accuracy-linked state geometry"]),
    _r("scripts/run_state_persistence.py", "panichello_rows", ["panichello_2024"], "delay-epoch lag-profile counts",
       "correct trials only",
       ["persistence profile of error trials", "correctness contrast"]),
    _r("scripts/run_state_persistence.py", "panichello_lag_rows", ["panichello_2024"], "delay-epoch lag counts",
       "correct trials only",
       ["lag profile of error trials"]),
    _r("scripts/run_state_persistence.py", "human_primary_rows", ["dandi_000469", "dandi_001187", "dandi_000574", "inagaki_alm5"],
       "multi-epoch persistence grid",
       "inherits iter_all_corpora admission unchanged (including its outcome filters)"),
    _r("scripts/run_state_persistence.py", "alm_matched_geometry_rows", ["inagaki_alm5"], "rate-matched delay counts",
       "control-arm trials of eligible ALM sessions, thinned to match human rates; response codes carried via the session loader"),
    _r("scripts/run_instrument_matched_content_decodability.py", "load_macaque_sessions", ["panichello_2024"],
       "donor-arm delay counts", "correct trials only (donor population matched to a correct-only human target)",
       ["content decodability of error trials"]),
    _r("scripts/run_persistence_estimator_split_count_sensitivity.py", "_panichello_sessions", ["panichello_2024"],
       "lag-bin counts", "correct trials only",
       ["split-count sensitivity of error-trial persistence"]),
    _r("scripts/run_content_label_cardinality_ladder.py", "load_macaque_session", ["panichello_2024"],
       "delay counts + graded labels", "correct trials only",
       ["cardinality ladder on error trials"]),
    _r("scripts/run_state_content_link.py", "iter_sessions_with_labels", ["panichello_2024"],
       "subtractive content test counts", "correct trials only",
       ["content linkage tested on error trials"]),
    _r("scripts/run_state_latent_identity.py", "macaque_sessions", ["panichello_2024"],
       "rank-1/residual identity counts", "correct trials only",
       ["latent identity tests on error trials"]),
    _r("scripts/run_count_subsampling_ladder.py", "_load_panichello_sessions", ["panichello_2024"],
       "per-bin count tensor", "all released trials; is_corr attached per trial, not filtered"),
    _r("scripts/run_state_orthogonality_census.py", "session_observables", ["panichello_2024"],
       "orthogonality observables", "all released trials; is_correct attached, not filtered"),
    _r("scripts/run_rate_free_state_geometry_behavior_link.py", "_session_arrays", ["panichello_2024"],
       "rate-free deviation arrays", "finite-outcome trials; is_corr carried as the behavioural variable"),
    _r("scripts/run_dominant_latent_identity_and_behaviour_breadth.py", "_load_session", ["panichello_2024"],
       "dominant-latent amplitude arrays", "finite-outcome trials; is_corr carried as the outcome"),
    _r("scripts/run_deviation_serial_dependence_and_temporal_locus.py", "_macaque_session_bundle", ["panichello_2024"],
       "serial-dependence bundle", "released trials; native binary outcome used as covariate, not filter"),
    _r("scripts/run_behavior_amplitude_rate_controls.py", "_reachable_sessions", ["panichello_2024"],
       "amplitude-covariate sessions", "sessions retained only if they hold enough ERROR trials for reachability",
       ["sessions with too few error trials are excluded from the accuracy-predicting component"]),
    _r("scripts/run_state_behavior_link.py", None, ["panichello_2024"], "cross-unit state vs outcome link",
       "error trials explicitly selected as the analysis arm (outcome used, not removed)"),
    _r("scripts/run_pfc3_content_ctg.py", "load_neuron_spatial", ["pfc3"], "pseudo-population delay counts",
       "9-class spatial-task files whose per-class trial counts clear the floor; no outcome field read"),
    _r("scripts/run_macaque_pfc_microstimulation_pipeline.py", "load_macaque_pfc_microstimulation_session", ["macaque_pfc_microstimulation"], "pre-binned trial spikerates",
       "one outcome arm per call: separate correct-only / error-only MAT files chosen by the caller"),
    _r("scripts/run_macaque_pfc_microstimulation_pipeline.py", "crop_trial", ["macaque_pfc_microstimulation"], "fixed-window crops",
       "trials at least N_BINS long; shorter trials dropped"),
    # -- stimulation / scalp / ECoG / iEEG field pipelines --------------------
    _r("scripts/run_haslacher_stimulation_geometry.py", "_retention_trials", ["haslacher_clam_tacs"],
       "retention-window scalp EEG epochs", "epochs of the requested phase-condition codes; behaviour not read"),
    _r("scripts/run_alagapan_stimulation_geometry.py", "_baseline_retention_trials", ["alagapan_phase_stimulation"],
       "retention-window depth iEEG epochs", "baseline-session epochs whose event type/latency defines a retention window"),
    _r("scripts/run_alagapan_stimulation_geometry.py", "_stimulation_retention_trials", ["alagapan_phase_stimulation"],
       "retention-window depth iEEG epochs", "stimulation-session epochs matched to condition labels; behaviour not read"),
    _r("scripts/run_wolff_corrected_analysis.py", "valid_mask", ["wolff_eeg_impulse"], "cue/impulse epoch validity",
       "trials not flagged bad by the authors' own bad-trial index; outcome not read"),
    _r("scripts/run_miller_drift_spine.py", "analyze_patient", ["kai_miller_nback"], "high-gamma epochs",
       "stimulus-locked epochs via src/preprocessing.load_subject; n-back level labels restricted to 0/1/2"),
    _r("src/preprocessing.py", "load_subject", ["kai_miller_nback"], "epochs + per-trial labels",
       "all stimulus-locked epochs; correctness returned as a label column, not applied as admission"),
    _r("scripts/run_ram_openloop_pipeline.py", "build_session_features", ["ram_ds005489_openloop"],
       "word-locked iEEG features", "words whose stimulation window fits inside the recording; recalled/stim carried per word"),
    _r("scripts/run_observability_and_power_census.py", "boran_lfp_rows", ["dandi_000574"], "depth-contact LFP tensor",
       "trials passing artifact flag AND window validity (errors retained)"),
    _r("scripts/run_observability_and_power_census.py", "boran_scalp_low_band_rows", ["dandi_000574"], "scalp EEG tensor",
       "trials passing artifact flag AND both grains' window validity (errors retained)"),
    _r("scripts/run_observability_and_power_census.py", "dandi_000673_lfp_rows", ["dandi_000673"], "hippocampal LFP tensor",
       "trials whose LFP epoch is finite and fits the recorded span"),
    _r("scripts/run_observability_and_power_census.py", "panichello_unit_rows", ["panichello_2024"], "delay counts",
       "correct trials only",
       ["observability census on error trials"]),
    _r("scripts/run_observability_and_power_census.py", "macaque_pfc_microstimulation_unit_rows", ["macaque_pfc_microstimulation"], "control-arm delay epochs",
       "no-stim control arm, CORRECT trials only",
       ["stimulation-response mapping inside error trials"]),
    _r("scripts/run_observability_and_power_census.py", "alagapan_ieeg_rows", ["alagapan_phase_stimulation"],
       "retention iEEG tensors", "delegated to the stimulation-geometry loaders' retention-window admission"),
    _r("scripts/run_observability_and_power_census.py", "haslacher_scalp_rows", ["haslacher_clam_tacs"],
       "retention scalp tensors", "delegated to the tACS-geometry loaders' phase-code admission"),
    _r("scripts/run_observability_and_power_census.py", "miller_ecog_rows", ["kai_miller_nback"], "ECoG epoch tensors",
       "epochs with n-back label in {0,1,2}"),
    _r("scripts/run_observability_and_power_census.py", "wolff_scalp_rows", ["wolff_eeg_impulse"], "impulse scalp tensors",
       "author-valid impulse epochs"),
    _r("scripts/run_observability_and_power_census.py", "ram_ieeg_rows", ["ram_ds005489_openloop"], "word-window tensors",
       "words fitting the recorded span"),
    _r("scripts/run_observability_and_power_census.py", "pfc3_unit_rows", ["pfc3"], "pseudo-population counts",
       "delegated to load_neuron_spatial class-floor admission"),
    _r("scripts/run_boran_modality_consistency.py", "registry_sessions", ["dandi_000574"], "spike + iEEG paired sessions",
       "artifact-free, window-valid trials restricted to set sizes supporting five folds; accuracy carried"),
    _r("scripts/run_band_versus_sensor_decomposition.py", "_process_session", ["dandi_000574"],
       "iEEG/scalp band tensors", "artifact-free window-valid trials (errors retained)"),
    _r("scripts/run_latent_model_observation_noise_comparison.py", "_build_dandi_000574_lfp_sessions",
       ["dandi_000574"], "LFP observation-noise tensors", "artifact-free window-valid trials (errors retained)"),
    _r("scripts/run_state_space_estimation_admissibility.py", "_boran_field_potential_session", ["dandi_000574"],
       "iEEG/scalp maintenance band power", "trials with no artifact AND correct response",
       ["field-potential outcome prediction", "error-trial band dynamics"]),
    # -- secondary readers with their own admission decisions -----------------
    _r("scripts/run_crossnobis_content_000469.py", "analyze_patient", ["dandi_000469"], "crossnobis content matrices",
       "sessions above the accuracy floor, restricted to load-1 trials",
       ["load-1 error-trial content structure"]),
    _r("scripts/run_drift_positive_controls.py", "load_000469", ["dandi_000469"], "drift positive-control arrays",
       "sessions above the accuracy floor, restricted to load-1 trials",
       ["load-1 error-trial drift"]),
    _r("scripts/run_drift_positive_controls.py", "load_000574", ["dandi_000574"], "drift positive-control arrays",
       "artifact-free trials (errors retained)"),
    _r("scripts/run_dim_robustness.py", "load_sternberg_sessions", ["dandi_000469", "dandi_001187", "dandi_000673"],
       "dimensionality sweep PSTHs", "sessions whose mean response_accuracy clears the prospective floor",
       ["dimensionality of low-accuracy sessions"]),
    _r("scripts/run_behavior_ctg.py", "_spike_session_outcome_ctg", ["dandi_000469", "dandi_001187", "dandi_000673"],
       "outcome CTG from raw spikes", "raw trials of sessions clearing the accuracy floor; correctness is the DECODED variable, not admission"),
    _r("scripts/run_behavior_ctg.py", "run_boran_units", ["dandi_000574"], "outcome CTG on Boran units",
       "artifact-free trials (errors retained); correctness decoded"),
    _r("scripts/run_behavior_ctg.py", "run_boran_ieeg", ["dandi_000574"], "outcome CTG on Boran iEEG",
       "artifact-free window-valid trials; correctness decoded"),
    _r("scripts/run_full_trial_content_decoding_000469.py", "process_subject", ["dandi_000469"],
       "full-trial content decoding arrays", "unit-QC'd sessions, restricted to load-1 trials",
       ["full-trial content decoding of error trials"]),
    _r("scripts/run_multiband_analysis.py", None, ["dandi_000574"], "multiband Boran iEEG epochs",
       "non-artifact trials"),
    _r("scripts/run_recording_tier_component_transfer.py", "_load_000574_trial_table", ["dandi_000574"],
       "tier-transfer trial table", "returns artifact/correct/set_size/start_time; callers apply ~artifact"),
    _r("scripts/run_watters_source_replication.py", "add_behavior_columns", ["watters_2026"],
       "behaviour table", "COMPLETED trials only; graded report deviation computed before any correctness threshold"),
    _r("scripts/run_human_maintenance_behaviour_link.py", "_iter_000469_admitted", ["dandi_000469"],
       "behaviour-link iterator", "trials with finite maintenance timestamps (errors and all loads retained); is_correct carried"),
    _r("scripts/run_human_maintenance_behaviour_link.py", "_iter_001187_admitted", ["dandi_001187"],
       "behaviour-link iterator", "trials with finite maintenance timestamps; is_correct carried"),
    _r("scripts/run_human_maintenance_behaviour_link.py", "_iter_000574_admitted", ["dandi_000574"],
       "behaviour-link iterator", "artifact-free trials; correct carried"),
    _r("scripts/run_intrinsic_timescale_vs_confinement.py", "_patient_region_tau", ["dandi_000469"],
       "baseline-timescale sessions", "trials whose fixation-to-encoding gap fits the baseline window"),
    _r("scripts/run_tau_estimator_calibration.py", None, ["dandi_000469"], "gap-feasibility counts",
       "counts trials whose baseline window fits, for reachability accounting only"),
]

# Registry entries whose enclosing function name differs from the symbol the
# audit looks up (module-level loops or renamed helpers).
REGISTRY_ALIASES = {
    ("scripts/run_observability_and_power_census.py", "boran_scalp_low_band_rows"): "boran_scalp_low_band_rows",
}

TRIAL_ACCESS_RE = re.compile(
    r"intervals/trials|intervals/WM_trials|['\"]isCorr['\"]|response_accuracy|"
    r"Trial_types_of_response_vector|load_macaque_pfc_microstimulation_session|add_behavior_columns|"
    r"intervals/trials/correct|\.completed\b"
)


def locate_functions(path: Path) -> dict[str, tuple[int, int]]:
    """Map function name -> (first line, last line) via AST, including
    functions nested inside other defs (several loaders are per-session
    workers defined inside main())."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return {}
    spans: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.setdefault(node.name, (node.lineno, node.end_lineno or node.lineno))
    return spans


def extract_predicate_lines(source_lines: list[str], start: int, end: int) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for idx in range(start - 1, min(end, len(source_lines))):
        text = source_lines[idx]
        if ADMISSION_ASSIGN_RE.match(text) or OUTCOME_SUBSCRIPT_RE.match(text):
            hits.append((idx + 1, text.rstrip()))
        elif _OUTCOME_RE.search(text) and EXTRACT_HINT_RE.search(text):
            hits.append((idx + 1, text.rstrip()))
    return hits[:14]


def audit_registry_entry(entry: dict, repo_root: Path) -> dict:
    path = repo_root / entry["path"]
    record = {
        **{k: v for k, v in entry.items()},
        "status": "audited",
        "line_start": None, "line_end": None,
        "predicate_verbatim": [], "predicate_line_numbers": [],
        "classification": "function_not_found",
        "filters_on_outcome_variable": False,
        "offending_expression": None,
    }
    if not path.exists():
        record["status"] = "skipped_with_reason"
        record["skip_reason"] = f"file_not_found: {entry['path']}"
        return record
    source_lines = path.read_text().splitlines()
    spans = locate_functions(path)
    name = REGISTRY_ALIASES.get((entry["path"], entry["function"]), entry["function"])
    if name is not None and name in spans:
        start, end = spans[name]
        record["line_start"], record["line_end"] = start, end
        lines = extract_predicate_lines(source_lines, start, end)
        body_text = "\n".join(source_lines[start - 1 : end])
        signature = next((ln for ln in source_lines[start - 1 : end] if ln.lstrip().startswith("def ")), "")
        verdict = classify_predicate_lines(lines, signature, body_text)
        record.update(verdict)
        record["predicate_verbatim"] = [t for _, t in verdict["predicate_lines"]]
        record["predicate_line_numbers"] = [n for n, _ in verdict["predicate_lines"]]
        if verdict["filters_on_outcome_variable"] and entry["invisible_analyses"]:
            record["invisible_analyses"] = entry["invisible_analyses"]
        elif verdict["filters_on_outcome_variable"]:
            record["invisible_analyses"] = list(INVISIBLE_TO_OUTCOME_FILTERS)
    else:
        if name is None:
            # Module-level loop loader: scan the whole file for its lines.
            lines = extract_predicate_lines(source_lines, 1, len(source_lines))
            verdict = classify_predicate_lines(lines, "", "\n".join(source_lines))
            record.update({
                "classification": verdict["classification"],
                "filters_on_outcome_variable": verdict["filters_on_outcome_variable"],
                "offending_expression": verdict["offending_expression"],
                "predicate_verbatim": [t for _, t in verdict["predicate_lines"]],
                "predicate_line_numbers": [n for n, _ in verdict["predicate_lines"]],
            })
        else:
            record["status"] = "skipped_with_reason"
            record["skip_reason"] = f"function_{name}_not_located_by_ast"
    return record


def sweep_repository(repo_root: Path, registry_spans: list[dict]) -> dict:
    """Scan every module under src/ and scripts/ for trial-table access and
    admission expressions; anything outside the registry is audited here."""
    files_scanned = 0
    files_with_trial_access = []
    extra_sites = []
    for directory in ("src", "scripts"):
        for path in sorted((repo_root / directory).glob("*.py")):
            files_scanned += 1
            text = path.read_text()
            if not TRIAL_ACCESS_RE.search(text):
                continue
            rel = str(path.relative_to(repo_root))
            lines = text.splitlines()
            sites = []
            for i, ln in enumerate(lines):
                if ADMISSION_ASSIGN_RE.match(ln) or OUTCOME_SUBSCRIPT_RE.match(ln):
                    sites.append((i + 1, ln.rstrip()))
            covered = [
                (s["line_start"], s["line_end"]) for s in registry_spans
                if s["path"] == rel and s["line_start"] is not None
            ]
            uncovered = [
                (n, q) for n, q in sites
                if not any(lo <= n <= hi for lo, hi in covered)
            ]
            if uncovered:
                for lineno, quote in uncovered:
                    verdict = classify_predicate_lines([(lineno, quote)], "", text)
                    extra_sites.append({
                        "path": rel, "line": lineno, "verbatim": quote,
                        "classification": verdict["classification"],
                        "filters_on_outcome_variable": verdict["filters_on_outcome_variable"],
                        "offending_expression": quote if verdict["filters_on_outcome_variable"] else None,
                    })
    return {
        "files_scanned": files_scanned,
        "extra_sites_outside_registry": extra_sites,
    }


SPAN_CACHE: dict[str, dict[str, tuple[int, int]]] = {}


def build_span_cache(repo_root: Path, entries: list[dict]) -> None:
    for entry in entries:
        rel = entry["path"]
        if rel in SPAN_CACHE:
            continue
        path = repo_root / rel
        SPAN_CACHE[rel] = locate_functions(path) if path.exists() else {}


# ---------------------------------------------------------------------------
# Light data checks: header-level reads only (trial-table columns, behaviour
# CSV flags). No signal arrays are loaded.
# ---------------------------------------------------------------------------
def data_root() -> Path | None:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    return Path(root) if root else None


def _h5_trials_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    import h5py
    out = {}
    with h5py.File(path, "r") as handle:
        group = "intervals/trials" if "intervals/trials" in handle else "intervals/WM_trials"
        if group not in handle:
            raise KeyError("no trials table")
        for col in columns:
            out[col] = handle[f"{group}/{col}"][:]
    return out


def measure_rule_discards(root: Path | None) -> dict:
    """Per-corpus discard counts for the outcome-conditioned rules."""
    checks: dict[str, dict] = {}

    def finalize(key, seen, measured, total, admitted, breakdown=None, note=None):
        row = {
            "n_files_seen": seen, "n_files_measured": measured,
            "n_trials_total": int(total), "n_trials_admitted_by_rule": int(admitted),
            "n_trials_discarded_by_rule": int(total - admitted),
            "discarded_fraction": float((total - admitted) / total) if total else None,
        }
        if breakdown:
            row["discard_breakdown"] = breakdown
        if note:
            row["note"] = note
        checks[key] = {"status": "measured", **row}

    # dandi_000469: keep = (loads == 1) & accuracy
    try:
        base = root / "000469"
        files = sorted(base.glob("sub-*/*_ses-2_ecephys+image.nwb"))
        tot = adm = err_only = load_only = 0
        measured = 0
        for fp in files:
            cols = _h5_trials_columns(fp, ["loads", "response_accuracy"])
            loads = cols["loads"].astype(int)
            acc = cols["response_accuracy"].astype(bool)
            tot += len(acc)
            adm += int(((loads == 1) & acc).sum())
            err_only += int(((loads == 1) & ~acc).sum())
            load_only += int((loads != 1).sum())
            measured += 1
        finalize("dandi_000469_iter_shared", len(files), measured, tot, adm,
                 {"load_1_error_trials_removed_by_accuracy_term": err_only,
                  "trials_removed_by_load_restriction": load_only})
    except Exception as exc:
        checks["dandi_000469_iter_shared"] = {"status": "failed", "reason": repr(exc)}

    # dandi_001187: keep = accuracy
    try:
        prov = _REPO_ROOT / "provenance"
        primary = json.loads((prov / "canonical_primary_records.json").read_text())
        rows = [r for r in primary if r.get("release") == "001187"]
        base = root / "001187"
        tot = adm = err = 0
        measured = missing = 0
        for row in rows:
            fp = root / row["path"]
            if not fp.exists():
                fp = base / row["path"]
            if not fp.exists():
                missing += 1
                continue
            cols = _h5_trials_columns(fp, ["response_accuracy"])
            acc = cols["response_accuracy"].astype(bool)
            tot += len(acc)
            adm += int(acc.sum())
            err += int((~acc).sum())
            measured += 1
        finalize("dandi_001187_iter_shared", len(rows), measured, tot, adm,
                 {"error_trials_removed": err},
                 note=f"{missing} primary records had no staged file")
    except Exception as exc:
        checks["dandi_001187_iter_shared"] = {"status": "failed", "reason": repr(exc)}

    # dandi_000574: keep = (~artifact) & correct  (same rule reused by the
    # field-potential admissibility reader)
    try:
        files = sorted((root / "000574").glob("sub-*/*.nwb"))
        tot = adm = art = corr = 0
        measured = 0
        for fp in files:
            cols = _h5_trials_columns(fp, ["artifact", "correct"])
            artifact = cols["artifact"].astype(bool)
            correct = cols["correct"].astype(bool)
            tot += len(correct)
            adm += int(((~artifact) & correct).sum())
            art += int(artifact.sum())
            corr += int((~correct).sum())
            measured += 1
        finalize("dandi_000574_iter_shared", len(files), measured, tot, adm,
                 {"artifact_trials_removed": art, "error_trials_removed": corr})
    except Exception as exc:
        checks["dandi_000574_iter_shared"] = {"status": "failed", "reason": repr(exc)}

    # panichello_2024: keep = isCorr (eight-plus reader sites share the rule)
    try:
        cfg = json.loads(DATASETS_CONFIG.read_text())["datasets"]["panichello_2024"]["local_path"]
        mats = sorted((root / cfg).glob("*.mat"))
        tot = adm = err = 0
        measured = 0
        for fp in mats:
            is_corr = _read_iscorr(fp)
            if is_corr is None:
                continue
            arr = np.asarray(is_corr).reshape(-1).astype(bool)
            tot += len(arr)
            adm += int(arr.sum())
            err += int((~arr).sum())
            measured += 1
        finalize("panichello_2024_correct_only_readers", len(mats), measured, tot, adm,
                 {"error_trials_removed": err})
    except Exception as exc:
        checks["panichello_2024_correct_only_readers"] = {"status": "failed", "reason": repr(exc)}

    # watters_2026: add_behavior_columns keeps completed trials only
    try:
        from run_watters_source_replication import watters_task_csvs  # type: ignore
        csv_paths = watters_task_csvs(root)
    except Exception:
        csv_paths = None
    try:
        import pandas as pd
        if csv_paths is None:
            cfg = json.loads(DATASETS_CONFIG.read_text())["datasets"]["watters_2026"]["local_path"]
            corpus = (root / cfg).parent if Path(cfg).name == "data_for_modeling" else (root / cfg)
            behav_dir = corpus / "data_for_figures" / "data_for_figures" / "behavior_processing"
            csv_paths = [behav_dir / f"{variant}.csv" for variant in ("ring", "triangle")]
        tot = adm = inc = 0
        measured = 0
        for fp in csv_paths:
            frame = pd.read_csv(fp, usecols=["completed"])
            comp = frame["completed"].astype(bool)
            tot += len(comp)
            adm += int(comp.sum())
            inc += int((~comp).values.sum())
            measured += 1
        finalize("watters_2026_completed_only", len(csv_paths), measured, tot, adm,
                 {"incomplete_trials_removed": inc})
    except Exception as exc:
        checks["watters_2026_completed_only"] = {"status": "failed", "reason": repr(exc)}

    return checks


def _read_iscorr(fp: Path):
    from scipy.io import loadmat
    try:
        raw = loadmat(str(fp), squeeze_me=True, variable_names=["isCorr"])
        value = raw.get("isCorr")
        return None if value is None else value
    except Exception:
        pass
    try:
        import h5py
        with h5py.File(str(fp), "r") as handle:
            if "isCorr" in handle:
                return handle["isCorr"][()]
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Askability table
# ---------------------------------------------------------------------------
def build_askability(discard_counts: dict) -> dict:
    p469 = discard_counts.get("dandi_000469_iter_shared", {})
    p1187 = discard_counts.get("dandi_001187_iter_shared", {})
    p574 = discard_counts.get("dandi_000574_iter_shared", {})
    panic = discard_counts.get("panichello_2024_correct_only_readers", {})
    watt = discard_counts.get("watters_2026_completed_only", {})

    def num(row, key="n_trials_discarded_by_rule"):
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
        return "count-unmeasured"

    table = {
        "dandi_000469": {
            "error_trial_contrasts": {
                "value": "yes",
                "evidence": "shared iterator discards them (src/corpus_sessions.py:123, "
                            f"{num(p469)} trials discarded across the corpus per the measured check) but "
                            "run_human_maintenance_behaviour_link.py::_iter_000469_admitted admits on finite "
                            "timestamps and carries is_correct",
                "blockers": ["src/corpus_sessions.py:123 keep = (loads == 1) & accuracy"],
            },
            "graded_report_analyses": {"value": "no",
                "evidence": "this corpus does not record a continuous report; outcome is binary response_accuracy "
                            "(RT derivable from timestamps_Probe/timestamps_Response, scripts/run_geometry_graded_behavior.py)"},
            "stimulation_response_mapping": {"value": "no",
                "evidence": "no stimulation was delivered in this corpus"},
            "within_maintenance_intervention": {"value": "no",
                "evidence": "passive Sternberg maintenance; no intervention epoch exists in the trial tables"},
            "cross_instrument_pairing": {"value": "no",
                "evidence": "single-unit release only; zero cross-release overlap with the LFP releases "
                            "(config/datasets.json view_relationship, results/patient_identity_audit.json)"},
        },
        "dandi_001187": {
            "error_trial_contrasts": {
                "value": "yes",
                "evidence": f"shared iterator keeps correct only (src/corpus_sessions.py:157; {num(p1187)} error trials discarded) "
                            "but _iter_001187_admitted carries is_correct and the delivered geometry npz stores "
                            "response_accuracy for every fitted trial (scripts/run_001187_pipeline.py)",
                "blockers": ["src/corpus_sessions.py:157 keep = accuracy"],
            },
            "graded_report_analyses": {"value": "no",
                "evidence": "binary response_accuracy only; response_time exists as a field (scripts/run_geometry_graded_behavior.py:177) but report is not continuous"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "yes",
                "evidence": "verified single-unit/LFP twin sessions in sibling release dandi_000673 "
                            "(canonical_sessions lfp_path; scripts/run_observability_and_power_census.py::dandi_000673_lfp_rows)"},
        },
        "dandi_000673": {
            "error_trial_contrasts": {"value": "yes",
                "evidence": "pipeline gates only session mean accuracy and saves response_accuracy per trial "
                            "(scripts/run_000673_pipeline.py::_process_session); LFP rows admit on finiteness only "
                            "(scripts/run_observability_and_power_census.py::dandi_000673_lfp_rows)"},
            "graded_report_analyses": {"value": "no", "evidence": "binary response_accuracy only"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "yes",
                "evidence": "LFP release paired with single-unit twins of dandi_001187 (canonical_sessions lfp_path)"},
        },
        "dandi_000574": {
            "error_trial_contrasts": {
                "value": "yes",
                "evidence": f"shared iterator discards {num(p574)} non-(artifact-free-and-correct) trials "
                            "(src/corpus_sessions.py:185) but run_behavior_ctg.run_boran_units, "
                            "run_boran_modality_consistency.registry_sessions and "
                            "_iter_000574_admitted all retain errors; run_closed_loop_behavior_flip.py requires "
                            ">=10 correct and >=10 error trials per patient",
                "blockers": ["src/corpus_sessions.py:185 keep = (~artifact) & correct"],
            },
            "graded_report_analyses": {"value": "no",
                "evidence": "verbal binary correctness + response_time; item identity unavailable "
                            "(set_letters 'not available', src/corpus_sessions.py item_id_unavailable_reason)"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "yes",
                "evidence": "same-session single-unit + depth iEEG + scalp EEG "
                            "(scripts/run_boran_modality_consistency.py; census boran_lfp_rows/scalp rows)"},
        },
        "panichello_2024": {
            "error_trial_contrasts": {
                "value": "yes",
                "evidence": f"{num(panic)} error trials discarded by every correct-only reader, yet outcome-preserving readers exist: "
                            "run_count_subsampling_ladder.py::_load_panichello_sessions, "
                            "run_state_orthogonality_census.py::session_observables, "
                            "run_rate_free_state_geometry_behavior_link.py::_session_arrays, "
                            "run_dominant_latent_identity_and_behaviour_breadth.py::_load_session, and "
                            "run_state_behavior_link.py selects ERROR trials as its analysis arm",
                "blockers": [
                    "scripts/run_panichello_pipeline.py analyze_session keep = correct",
                    "scripts/run_state_persistence.py panichello_rows/panichello_lag_rows spikes = spikes[correct]",
                    "scripts/run_state_content_link.py:286 spikes[correct]",
                    "scripts/run_state_latent_identity.py:138 spikes[correct]",
                    "scripts/run_instrument_matched_content_decodability.py:172 spikes[correct]",
                    "scripts/run_persistence_estimator_split_count_sensitivity.py:206 spikes[correct]",
                    "scripts/run_content_label_cardinality_ladder.py:194 spikes[correct]",
                    "scripts/run_observability_and_power_census.py:424 panichello_unit_rows spikes = spikes[correct]",
                ],
            },
            "graded_report_analyses": {"value": "no",
                "evidence": "native outcome is binary isCorr; the .mat files carry no response-time field "
                            "(scripts/run_state_latent_identity.py:527 note)"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "no", "evidence": "single instrument; area labels inferred at monkey-cluster granularity only"},
        },
        "watters_2026": {
            "error_trial_contrasts": {"value": "yes",
                "evidence": f"loader admits completed trials regardless of outcome and carries correct plus the graded deviation "
                            f"(src/corpus_sessions.py:496-497; {num(watt)} incomplete trials removed, errors retained)"},
            "graded_report_analyses": {"value": "yes",
                "evidence": "continuous saccadic report deviation preserved pre-threshold "
                            "(scripts/run_watters_source_replication.py::add_behavior_columns comment and frame['report_deviation'])"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "no", "evidence": "processed per-trial cache only; single instrument"},
        },
        "inagaki_alm5": {
            "error_trial_contrasts": {"value": "yes",
                "evidence": "load_alm_raw_session carries control_response_code/perturb_response_code alongside the counts "
                            "(src/corpus_sessions.py:289-290); eligibility is response-code<5 range QC, not outcome conditioning"},
            "graded_report_analyses": {"value": "no", "evidence": "instructed left/right lick choice; binary by design"},
            "stimulation_response_mapping": {"value": "yes",
                "evidence": "photoinhibition arm split by stim_trial_vector (src/corpus_sessions.py:263-264 perturb_trials)"},
            "within_maintenance_intervention": {"value": "yes",
                "evidence": "perturbation delivered inside the delay epoch (RandomDelayTask withPerturbation sessions)"},
            "cross_instrument_pairing": {"value": "no", "evidence": "single silicon-probe recording, single structure by design"},
        },
        "macaque_pfc_microstimulation": {
            "error_trial_contrasts": {"value": "yes",
                "evidence": "release ships separate correct/error trial pools; load_macaque_pfc_microstimulation_session(prefix, correct=False) reads the error pool "
                            "(scripts/run_macaque_pfc_microstimulation_pipeline.py:288) and run_macaque_pfc_microstimulation_design_corrected.py consumes both arms"},
            "graded_report_analyses": {"value": "no", "evidence": "binary correctness pools; no continuous report field wired"},
            "stimulation_response_mapping": {"value": "yes",
                "evidence": "delay microstimulation vs no-stim control arms are the corpus's design "
                            "(scripts/run_macaque_pfc_microstimulation_pipeline.py docstring; uStim during the WM delay)"},
            "within_maintenance_intervention": {"value": "yes",
                "evidence": "microstimulation delivered inside the working-memory delay"},
            "cross_instrument_pairing": {"value": "no", "evidence": "single-instrument recordings"},
        },
        "haslacher_clam_tacs": {
            "error_trial_contrasts": {"value": "not-as-loaded",
                "evidence": "no loader in this repository reads the task's trial-level outcome; "
                            "_retention_trials admits on phase-condition codes only "
                            "(scripts/run_haslacher_stimulation_geometry.py:283)"},
            "graded_report_analyses": {"value": "not-as-loaded",
                "evidence": "no repo loader reads any behavioural report variable for this corpus; presence in the release unverified"},
            "stimulation_response_mapping": {"value": "yes",
                "evidence": "phase-locked tACS vs no-stim retention epochs are the loader's two arms (_retention_trials on stim/no_stig raws)"},
            "within_maintenance_intervention": {"value": "yes",
                "evidence": "stimulation is delivered inside the retention interval (RETENTION_TMIN/TMAX windows)"},
            "cross_instrument_pairing": {"value": "no", "evidence": "scalp EEG only"},
        },
        "alagapan_phase_stimulation": {
            "error_trial_contrasts": {"value": "not-as-loaded",
                "evidence": "no loader reads the task outcome; admission is event-type/latency window logic "
                            "(scripts/run_alagapan_stimulation_geometry.py::_baseline_retention_trials)"},
            "graded_report_analyses": {"value": "not-as-loaded",
                "evidence": "no repo loader reads a behavioural report variable for this corpus"},
            "stimulation_response_mapping": {"value": "yes",
                "evidence": "baseline vs stimulation retention sessions loaded side by side (_stimulation_retention_trials)"},
            "within_maintenance_intervention": {"value": "yes",
                "evidence": "phase-locked stimulation delivered during the retention window (DIN4->DIN5)"},
            "cross_instrument_pairing": {"value": "no", "evidence": "depth iEEG only"},
        },
        "ram_ds005489_openloop": {
            "error_trial_contrasts": {"value": "yes",
                "evidence": "recalled (0/1) carried per word and never used for admission "
                            "(scripts/run_ram_openloop_pipeline.py::_derive_word_stimulation/build_session_features)"},
            "graded_report_analyses": {"value": "no", "evidence": "free-recall outcome is binary remembered/forgotten per word"},
            "stimulation_response_mapping": {"value": "yes",
                "evidence": "per-word stim flag gives T against neural features; Y=recalled (script header lines 20-30)"},
            "within_maintenance_intervention": {"value": "no",
                "evidence": "this task does not have a maintenance delay; stimulation is at encoding (loader map notes no explicit WM delay)"},
            "cross_instrument_pairing": {"value": "no", "evidence": "depth iEEG only"},
        },
        "ram_ds005557_closedloop": {
            "error_trial_contrasts": {"value": "not-as-loaded",
                "evidence": "no loader written for this corpus yet; the sibling open-loop BIDS loader is reusable "
                            "(results/loader_reuse_map.json ram_ds005557_closedloop entry)"},
            "graded_report_analyses": {"value": "not-as-loaded", "evidence": "no loader; free-recall outcome is binary in this task family"},
            "stimulation_response_mapping": {"value": "not-as-loaded",
                "evidence": "closed-loop stimulation events exist in the release but nothing here parses them yet"},
            "within_maintenance_intervention": {"value": "no",
                "evidence": "episodic-encoding stimulation; the task does not contain a maintenance delay"},
            "cross_instrument_pairing": {"value": "no", "evidence": "depth iEEG only"},
        },
        "ds004752": {
            "error_trial_contrasts": {"value": "not-as-loaded",
                "evidence": "staged but no confirmed trial-level tensor loader; loader status disputed between audits "
                            "(results/loader_reuse_map.json ds004752 entry)"},
            "graded_report_analyses": {"value": "not-as-loaded",
                "evidence": "verbal Sternberg responses exist in the BIDS release but no repo parser reads them"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered in this corpus"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "not-as-loaded",
                "evidence": "release carries depth contacts, beamformed cortical virtual sensors and scalp EEG, but no full "
                            "loader wires more than one grain; shares 9 patients with dandi_000574 "
                            "(config/datasets.json ds004752 view_relationship)"},
        },
        "pfc3": {
            "error_trial_contrasts": {"value": "not-as-loaded",
                "evidence": "the staged task tables this project parses expose class/trial structure only; "
                            "no loader reads a correct/error field (scripts/run_behavior_ctg.py:23 notes none is wired)"},
            "graded_report_analyses": {"value": "no", "evidence": "8-way instructed saccade task; no continuous report recorded in parsed tables"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "no", "evidence": "pseudo-population: neurons recorded in separate sessions, never simultaneous"},
        },
        "wolff_eeg_impulse": {
            "error_trial_contrasts": {"value": "not-as-loaded",
                "evidence": "admission uses only the authors' bad-trial index (valid_mask); no loader reads an outcome variable"},
            "graded_report_analyses": {"value": "not-as-loaded", "evidence": "no repo loader reads a behavioural report variable"},
            "stimulation_response_mapping": {"value": "not-as-loaded",
                "evidence": "impulse-event epochs exist (EEG_impulse) but whether the impulse is peripheral sensory input or "
                            "direct stimulation is not established anywhere in this repository"},
            "within_maintenance_intervention": {"value": "not-as-loaded",
                "evidence": "impulse timing relative to any maintenance interval is not established by any loader here"},
            "cross_instrument_pairing": {"value": "no", "evidence": "scalp EEG only"},
        },
        "kai_miller_nback": {
            "error_trial_contrasts": {"value": "yes",
                "evidence": "load_subject returns per-trial correct as a label column and applies no admission on it "
                            "(src/preprocessing.py:692,740); miller_ecog_rows restricts n-back level, not outcome"},
            "graded_report_analyses": {"value": "no", "evidence": "n-back correctness is binary; no continuous report"},
            "stimulation_response_mapping": {"value": "no", "evidence": "no stimulation delivered"},
            "within_maintenance_intervention": {"value": "no", "evidence": "no intervention epoch"},
            "cross_instrument_pairing": {"value": "no", "evidence": "ECoG only"},
        },
    }
    return table




# ---------------------------------------------------------------------------
# Summary branches (evaluated strictly after their pre-declaration above)
# ---------------------------------------------------------------------------
def evaluate_branches(loaders: list[dict], askability: dict, corpora: list[str]) -> list[dict]:
    trial_filters = [l for l in loaders if l.get("classification") == "trial_level_outcome_filter"]
    spine = [l for l in trial_filters if l["path"] == "src/corpus_sessions.py"
             and l["function"] in ("iter_dandi_000469", "iter_dandi_001187", "iter_dandi_000574")]
    outside = [l for l in trial_filters if not (l["path"] == "src/corpus_sessions.py")]

    preserving_readers: dict[str, int] = {}
    for l in loaders:
        if l.get("classification") in ("non_outcome_admission", "outcome_preserved", "outcome_stratified_loader"):
            for c in l.get("corpora", []):
                preserving_readers[c] = preserving_readers.get(c, 0) + 1
    flagged_corpora = sorted({c for l in spine + [x for x in trial_filters if x not in spine]
                              for c in l.get("corpora", [])
                              if preserving_readers.get(c, 0) > 0})

    graded_yes = [c for c in corpora if askability[c]["graded_report_analyses"]["value"] == "yes"]
    maint_yes = [c for c in corpora if askability[c]["within_maintenance_intervention"]["value"] == "yes"]
    pairing_yes = [c for c in corpora if askability[c]["cross_instrument_pairing"]["value"] == "yes"]

    branches = []

    def emit(name, effect, detail):
        branches.append({"branch": name, "effect_size": effect, "detail": detail})

    n = len(trial_filters)
    if n >= 3:
        emit("outcome_filtering_is_a_defect_class_not_a_one_off",
             {"n_trial_level_outcome_filter_loaders": n,
              "fraction_of_audited_loaders": round(n / max(len(loaders), 1), 3)},
             f"{n} loaders admit trials conditional on an outcome variable")
    else:
        emit(f"outcome_filtering_is_confined_to_{n}_loaders",
             {"n_trial_level_outcome_filter_loaders": n}, "below the defect-class threshold of 3")

    emit("outcome_filtering_extends_beyond_the_human_spine_iterators"
         if outside else "outcome_filtering_confined_to_the_human_spine_iterators",
         {"n_sites_outside_corpus_sessions": len(outside),
          "outside_paths": sorted({l['path'] for l in outside})},
         "sites outside the shared human iterators")

    k = len(spine)
    emit(f"{k}_of_3_human_spine_iterators_discard_error_trials" if k < 3
         else "all_three_human_spine_iterators_discard_error_trials",
         {"human_spine_iterators_flagged": k}, "src/corpus_sessions.py iterators flagged")

    for corpus_id in flagged_corpora:
        blockers = sorted({f"{l['path']}:{min(l['predicate_line_numbers'])}" if l.get("predicate_line_numbers")
                           else l["path"] for l in trial_filters if corpus_id in l.get("corpora", [])})
        emit(f"corpus_{corpus_id}_error_contrasts_require_an_alternate_loader",
             {"n_blocker_sites": len(blockers), "n_outcome_preserving_readers": preserving_readers[corpus_id],
              "blocker_sites": blockers},
             "dominant loaders remove error trials while outcome-preserving readers also exist")

    for corpus_id in corpora:
        cells = askability[corpus_id]
        k = sum(1 for cell in cells.values() if cell["value"] == "yes")
        emit(f"corpus_{corpus_id}_askable_for_{k}_of_5_questions",
             {"n_yes_cells": k,
              "cells": {name: cell["value"] for name, cell in cells.items()}},
             ASKABILITY_CELL_RULE)

    emit(f"graded_report_supported_by_{len(graded_yes)}_of_{len(corpora)}_corpora",
         {"corpora": graded_yes}, ASKABILITY_CELL_RULE)
    emit(f"within_maintenance_intervention_supported_by_{len(maint_yes)}_of_{len(corpora)}_corpora",
         {"corpora": maint_yes}, ASKABILITY_CELL_RULE)
    emit(f"cross_instrument_pairing_supported_by_{len(pairing_yes)}_of_{len(corpora)}_corpora",
         {"corpora": pairing_yes}, ASKABILITY_CELL_RULE)
    return branches


def main() -> None:
    started = time.time()

    datasets = json.loads(DATASETS_CONFIG.read_text())["datasets"]
    corpora = sorted(datasets.keys())

    build_span_cache(_REPO_ROOT, REGISTRY)
    registry_spans = []
    for entry in REGISTRY:
        name = REGISTRY_ALIASES.get((entry["path"], entry["function"]), entry["function"])
        span = SPAN_CACHE.get(entry["path"], {}).get(name) if name else None
        registry_spans.append({
            "path": entry["path"], "function": entry["function"],
            "line_start": span[0] if span else None, "line_end": span[1] if span else None,
        })
    audited = [audit_registry_entry(e, _REPO_ROOT) for e in REGISTRY]

    sweep = sweep_repository(_REPO_ROOT, registry_spans)

    files_with_trial_access = sorted({
        str(p.relative_to(_REPO_ROOT))
        for directory in ("src", "scripts")
        for p in (_REPO_ROOT / directory).glob("*.py")
        if TRIAL_ACCESS_RE.search(p.read_text())
    })

    located = [l for l in audited if l["status"] == "audited"]
    skipped = [l for l in audited if l["status"] == "skipped_with_reason"]
    extra_sites = sweep["extra_sites_outside_registry"]
    loaders_found = len(audited) + len(extra_sites)
    audited_total = len(located) + len(extra_sites)
    skipped_total = len(skipped)
    if loaders_found != audited_total + skipped_total:
        raise SystemExit("zero-drop reconciliation failed: loaders_found != audited + skipped_with_reason")

    root = data_root()
    discard_counts = measure_rule_discards(root) if root is not None else {
        k: {"status": "skipped_with_reason", "reason": "WM_DYNAMICS_DATA_ROOT not set"} for k in
        ["dandi_000469_iter_shared", "dandi_001187_iter_shared", "dandi_000574_iter_shared",
         "panichello_2024_correct_only_readers", "watters_2026_completed_only"]}

    askability = build_askability(discard_counts)
    branches = evaluate_branches(located + extra_sites, askability, corpora)

    outcome_filter_sites = [
        {"path": l["path"], "function": l.get("function"),
         "line": (l.get("predicate_line_numbers") or [l.get("line")])[0],
         "expression": l.get("offending_expression")}
        for l in located + extra_sites if l.get("filters_on_outcome_variable")
    ]

    wall_clock_s = round(time.time() - started, 2)
    artifact = {
        "analysis_id": "corpus_admission_audit",
        "schema_version": "1.0.0",
        "scope": {
            "wall_clock_seconds": wall_clock_s,
            "seed": None,
            "mode": "static analysis plus header-level trial-table reads; no signal arrays loaded",
            "data_root_used": str(root) if root else None,
            "n_registered_corpora": len(corpora),
            "modules_scanned": sweep["files_scanned"],
            "files_with_trial_table_access": len(files_with_trial_access),
            "loaders_found": loaders_found,
            "audited": audited_total,
            "skipped_with_reason": skipped_total,
            "reconciliation": "loaders_found = audited + skipped_with_reason",
            "askability_cell_rule": ASKABILITY_CELL_RULE,
        },
        "pre_declared_branches": PRE_DECLARED_BRANCHES,
        "summary_branches": branches,
        "outcome_filter_site_count": sum(1 for s in outcome_filter_sites if s.get("expression")),
        "outcome_filter_sites": [s for s in outcome_filter_sites if s.get("expression")],
        "loaders": audited,
        "sweep_extra_sites": extra_sites,
        "trial_discard_counts": discard_counts,
        "askability": askability,
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=1))

    flagged = artifact["outcome_filter_sites"]
    print(f"wrote {ARTIFACT_PATH}")
    print(f"loaders_found={loaders_found} audited={audited_total} skipped={skipped_total}")
    print(f"outcome-filter sites: {len(flagged)}")
    for s in flagged:
        print(f"  {s['path']}:{s['line']} [{s['function']}] {(s['expression'] or '')[:100]}")
    for b in branches:
        print(f"branch: {b['branch']}")


if __name__ == "__main__":
    main()
