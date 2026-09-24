#!/usr/bin/env python3
"""Build one conservative evidence-ledger row per existing JSON artifact."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, sha256_file  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT = ROOT / "provenance" / "evidence_ledger.json"

SUPERSEDED_PREFIXES = (
    "07_", "08_", "09_", "all_statistics", "amplification", "axis_rotation",
    "causal_", "closed_loop", "digital_twin", "dim_", "dmd_", "dpca_",
    "forest_", "manifold_", "pr_", "rl_policy", "target_", "targeting_",
)

INTERMEDIATE_ARTIFACTS = {
    "haslacher_phase_diffusion_author_smoke1",
    "haslacher_phase_diffusion_smoke2",
    "macaque_pfc_microstimulation_design_corrected_smoke1",
    "watters_2026_item_count_drift_smoke1",
}


# Per-estimand reconciliation for the current human-first spine.  Keeping these
# rows explicit prevents the conservative legacy fallback from mislabeling a
# newly completed real-data result as "pending_real_data_fit".
CURRENT_OVERRIDES = {
    "human_drift_spine_000469": {
        "claim_id": "current::dandi000469_predictive_history_adjudication",
        "construct": "population-wide within-trial predictive history versus confined dynamics",
        "eligibility_rule": "DANDI 000469 load-1 trials with repeated item identity; five grouped outer folds",
        "independent_unit": "patient (18 patients; folds and bins are repeated measures)",
        "estimand": "patient-mean matched-flexibility M2 contrasts and paired own-minus-neighbour R2 advantage",
        "preprocessing_version": "unsmoothed 100-ms bins; all transforms, axes, and centroids fit in outer training folds",
        "inferential_method": "frozen held-out scoring with patient-cluster percentile bootstrap intervals and patient sign count",
        "correction_family": "targeted matched-control family with patient-bootstrap intervals against zero",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "M2 fails matched-flexibility baselines; the retained own-history effect is equally strong off the content axis; leak-rate identifiability is partial",
        "model_prediction": "Confined temporal dynamics require M2 intervals above zero against both matched-flexibility baselines and an own-minus-neighbour interval above zero.",
        "prediction_match_status": "not matched for confined dynamics; matched only for population-wide predictive history",
    },
    "human_drift_behavior_000469": {
        "claim_id": "current::dandi000469_drift_error_association",
        "construct": "association between held-out probe displacement and memory error",
        "eligibility_rule": "trials with out-of-fold probe residual and response accuracy",
        "independent_unit": "patient random intercept; 18 patients and 765 trials",
        "estimand": "log-odds change in error per within-patient SD of probe displacement",
        "preprocessing_version": "out-of-fold residuals from human_drift_spine_000469",
        "inferential_method": "hierarchical logistic variational-Bayes fit plus onset/change crack chase",
        "correction_family": "single frozen downstream behavior gate",
        "status": "replication_or_informative_null",
        "gate": "G1",
        "caveat": "interval crosses zero; near-ceiling accuracy and approximate variational posterior limit threshold inference",
        "model_prediction": "Larger probe displacement should increase error odds and permit an empirical tolerance threshold.",
        "prediction_match_status": "inconclusive_interval_includes_zero",
    },
    "drift_control_payload_000469": {
        "claim_id": "current::dandi000469_control_payload_gate",
        "construct": "behavior-calibrated tolerance, passive prediction, and control-cost payload",
        "eligibility_rule": "requires identified behavior slope, tolerance, anisotropy, and session-level D/lambda",
        "independent_unit": "patient",
        "estimand": "tolerance ellipsoid, passive error probability, and minimum intervention cost",
        "preprocessing_version": "derived only from current human spine and behavior artifacts",
        "inferential_method": "deterministic gate evaluation; no numerical fallback for failed prerequisites",
        "correction_family": "G4 payload gate",
        "status": "nonidentified",
        "gate": "G4",
        "caveat": "behavior and anisotropy prerequisites did not pass; no control quantity is licensed",
        "model_prediction": "An identified positive displacement-error slope would define tolerance and permit passive and intervention calculations.",
        "prediction_match_status": "nonidentified_prerequisites_failed",
    },
    "human_drift_spine_001187_000673": {
        "claim_id": "current::dandi001187_000673_load_confinement_sensitivity",
        "construct": "load-manipulation (1 vs 3) sensitivity analysis of confinement/diffusion using the linked-view canonical registry",
        "eligibility_rule": "DANDI 001187 canonical unit view (53 canonical sessions) with DANDI 000673 linked hippocampal-LFP sensitivity view where available; each canonical session enters the unit-based primary fit exactly once",
        "independent_unit": "patient, with canonical sessions nested within patient (33 unit fits, 22 linked-LFP fits complete)",
        "estimand": "patient-cluster-bootstrap mean load-3-minus-load-1 state-space/moment lambda and diffusion, unit-based and LFP-linked views separately, and their correlation with the measured PR-vs-load slope",
        "preprocessing_version": "unsmoothed 100-ms unit PSTH bins and non-overlapping 100-ms block-averaged unsmoothed high-gamma LFP bins; leading within-fold PCA component as the load-contrast axis (no repeated item identity, so no LDA content axis, unlike 000469)",
        "inferential_method": "frozen decision-rule state-space and moment estimators; patient-cluster percentile bootstrap",
        "correction_family": "linked-view sensitivity family, not the frozen primary decision-rule family",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "Most jointly identifiable load contrasts remain unresolved (CI crosses zero; 3-7 identified patients depending on quantity); the one contrast excluding zero (unit-based moment diffusion, -0.65/s [-1.35,-0.12], n=7 patients) is exploratory, not part of the frozen confirmatory family. PR-slope correlations are not_estimable for the state-space quantities (n=3). This is a linked-view sensitivity analysis, never an independent-replication claim.",
        "model_prediction": "If diffusion increases with memory load, load-3-minus-load-1 diffusion should be positive and should correlate with the measured PR-vs-load slope.",
        "prediction_match_status": "mixed: the one identifiable contrast (unit-based moment diffusion) is negative rather than positive, opposing the naive prediction; most other contrasts are nonidentified or cross zero, and PR-slope correlations do not exclude zero (n=3-7).",
    },
    "human_drift_spine_000574": {
        "claim_id": "current::dandi000574_matched_control_replication",
        "construct": "independent replication of the matched-flexibility adjudication in a Sternberg verbal-WM cohort",
        "eligibility_rule": "DANDI 000574 (Boran) set-size 4/6/8 trials; five grouped outer folds; same frozen decision rule and estimators as human_drift_spine_000469",
        "independent_unit": "patient (8/9 patients with an identified M2-minus-M0 estimate; 37 sessions nested within patient)",
        "estimand": "patient-mean matched-flexibility M2 contrasts, own-minus-neighbour R2 sensitivity, and identified-fold lambda",
        "preprocessing_version": "unsmoothed 100-ms bins; all transforms/axes/centroids fit in outer training folds; no repeated item identity in the public release (set_letters is 'not available'), so the projection axis is the leading within-fold PCA component, not an LDA content direction",
        "inferential_method": "frozen held-out scoring with patient-cluster percentile bootstrap intervals and patient sign count, identical to the 000469 primary spine",
        "correction_family": "same matched-control decision family as DANDI 000469",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "DANDI 000574 staged release is version 0.250815.1108; legacy Boran/000574 artifacts carry no recorded version or manifest, so this is release-specific evidence, not a numerically reconciled replication of any legacy result (provenance/dandi_000574_version_reconciliation.json, status=unreconciled). No item identity is available in this release, so content-axis anisotropy cannot be tested (only permuted-axis and matched-complement contrasts, both crossing zero, n=3-5 patients).",
        "model_prediction": "Confined temporal dynamics require intervals above zero against both matched-flexibility baselines and for own-minus-neighbour prediction.",
        "prediction_match_status": "not matched: neither matched-flexibility comparison has a patient-bootstrap lower bound above zero; the raw own-minus-neighbour interval also crosses zero.",
    },
    "miller_drift_spine": {
        "claim_id": "current::miller_nback_task_generality_descriptive",
        "construct": "descriptive unmatched M2-M0 task-generality check outside the Sternberg paradigm (N-back, lateral ECoG)",
        "eligibility_rule": "4 patients (al, ca, cc, ug), house-picture 0/1/2-back ECoG; five grouped outer folds, same frozen estimators",
        "independent_unit": "patient (n=4); no cross-patient pooled interval is computed",
        "estimand": "per-patient M2-minus-M0 nats/observation and, where identifiable, per-patient state-space/moment lambda",
        "preprocessing_version": "literature-standard ECoG substitute (MAD bad-channel rejection, CAR, 60 Hz notch+harmonics, Hilbert high-gamma) -- no author QC guidance exists in the release; unsmoothed 100-ms bins",
        "inferential_method": "frozen held-out scoring with within-patient trial-cluster percentile bootstrap; no group-level inference (n=4 is below the frozen patient-cluster winner rule's applicable scale)",
        "correction_family": "descriptive; frozen decision-rule family not applied at the group level for this dataset",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "n=4 is a descriptive task-generality comparison, not a population test. M2 clears its per-patient practical threshold in 4/4 patients (0.070-0.099 nats/observation) but confinement lambda is identifiable in only 1/4 patients (ca: moment 2.00/s, state-space 1.99/s), order-of-magnitude consistent with 000469's 1.36-1.75/s. No trial-level behavioral-accuracy label is buildable from the release. No lateral-vs-MTL within-dataset comparison is possible (ECoG grids cannot reach MTL); only a between-dataset qualitative comparison is reported.",
        "model_prediction": "An unmatched M2-M0 gain is descriptive only because count variance stabilization and matched M0 controls do not apply to continuous ECoG voltage without a separate design.",
        "prediction_match_status": "descriptive M2-M0 gain only; no confined-dynamics verdict is licensed.",
    },
    "wolff_corrected_impulse": {
        "claim_id": "current::wolff_ping_evoked_impulse_response",
        "construct": "exogenous ping-evoked decodable impulse response vs pre-ping endogenous decodability, corrected doubled-orientation circular decoding",
        "eligibility_rule": "Wolff et al. 2017 experiment 1 (30 participants) plus experiment 2 (two-ping trials, 18-19 participants); five-fold participant-local cross-validation",
        "independent_unit": "participant (30 in experiment 1; 18/19 with complete two-ping trial sessions in experiment 2)",
        "estimand": "participant-level circular-Mahalanobis decoding strength pre-ping and post-ping, voltage and 8-12 Hz alpha separately, against a matched permuted-label floor; paired endogenous vs ping-evoked decay rate (lambda)",
        "preprocessing_version": "released preprocessed voltage; alpha via Hilbert 8-12 Hz band; doubled-orientation 12-bin circular Mahalanobis decoding; no temporal smoothing on the decay fit",
        "inferential_method": "participant-cluster percentile bootstrap against matched permuted-label floors; ou_moments/state-space decay fit on the ping-evoked and endogenous windows separately",
        "correction_family": "targeted frozen family (fit_impulse_decay), same estimator machinery as the primary spine",
        "status": "current_exploratory",
        "gate": "G3",
        "caveat": "Pre-ping (endogenous) decoding is at the permuted-label floor for both signals in the large majority of participants (only 1/30 above floor for voltage and for alpha). Post-ping decoding exceeds floor for voltage (participant-bootstrap CI [0.00054, 0.00498], excludes zero) but not for alpha (CI [-0.00116, 0.00452]). Ping-vs-endogenous lambda agreement is estimable only for alpha (30/30 participants identifiable; ping lambda 7.95/s vs endogenous 11.06/s, difference -3.10/s); voltage lambda agreement is not_estimable (only 6/30 paired-identifiable participants). Two-ping superposition/linearity is explicitly not_identifiable in all 18 scored sessions -- the release has sequential first/second-ping epochs but no isolated-input or summed-input condition, so equality to a linear sum cannot be tested.",
        "model_prediction": "A genuine hidden/activity-silent state should show little-to-no pre-ping decodability but a decodable, decaying ping-evoked response; an actively-maintained state should show strong pre-ping decodability too.",
        "prediction_match_status": "matched the activity-silent pattern in this corrected reanalysis: pre-ping decodability is near-floor for nearly all participants in both signals, while the ping evokes a decodable, decaying response in voltage. This is one corrected reanalysis, not a final adjudication of the Wolff/Barbosa dispute.",
    },
    "watters_2026_item_count_drift": {
        "claim_id": "current::watters2026_item_count_diffusion_heterogeneity",
        "construct": "relationship between remembered item count and process diffusion / diffusive dimensionality in macaque frontal cortex (non-human method-generality arm, no human equivalent for this manipulation)",
        "eligibility_rule": "44 sessions across 2 animals (Elgar, Perle) and 2 task configurations (triangle, ring); units present on every source trial; unsmoothed 100-ms primary and 50-ms sensitivity bins; opposite-fold PCA/normalization",
        "independent_unit": "session, nested in animal and configuration; no cross-animal or cross-configuration pooling",
        "estimand": "session-bootstrap mean slope of total diffusion, diffusion-per-item, and effective diffusive dimension per added remembered item, per animal x configuration cell",
        "preprocessing_version": "10-ms source spike cache re-binned to 100-ms primary / 50-ms sensitivity non-overlapping counts; per-PC state-space and moment diffusion fit within each session",
        "inferential_method": "session-cluster percentile bootstrap per animal x configuration cell (no cross-cell pooling)",
        "correction_family": "exploratory; project-specific extension, not a statistic reported by the source paper",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "Non-human macaque data used only because no human equivalent tests item-count scaling (N1). Every effect is animal- and configuration-specific. Diffusion-per-item DECREASES with item count and excludes zero in 3/4 animal x configuration cells (ring Perle, triangle Elgar, triangle Perle; both bin widths), opposite the naive prediction. Total diffusion increases with item count and excludes zero only in ring Perle (both bin widths); it crosses zero in triangle for both animals and is a single-session point estimate in ring Elgar (n=1 session, no bootstrap spread). Effective diffusive dimension is unresolved in all four cells (CI crosses zero). The 100 vs 50 ms bin-width sensitivity check preserves the sign pattern.",
        "model_prediction": "If more remembered items are packed into the same neural manifold, total diffusion and/or diffusive dimensionality should increase with item count.",
        "prediction_match_status": "mixed and animal/configuration-specific: total diffusion rises with item count in only one of four cells (ring Perle); diffusion-per-item instead falls with item count in three of four cells; diffusive dimension is unresolved everywhere. Reported as heterogeneity, not averaged away (N4).",
    },
    "haslacher_phase_diffusion": {
        "claim_id": "current::haslacher_clam_tacs_phase_modulation_null",
        "construct": "whether closed-loop tACS stimulation phase modulates state-space diffusion (D) at the population level, active vs control montage, author-native SASS preprocessing",
        "eligibility_rule": "author-README-ordered pyprep noisy-channel/saturation rejection, 8-14 Hz filter, SASS, post-SASS average reference; participant-level SASS gate (post-SASS alpha power must move closer to baseline and stay within a fourfold factor); auxiliary envelope/stim channels excluded from PCA/dynamics",
        "independent_unit": "participant (17 active, 25 control; 42/46 total pass the SASS QC gate, 4 fail)",
        "estimand": "participant-level circular-harmonic (cosine/sine) regression of state-space and legacy-increment diffusion, and behavior log-odds, on stimulation phase; population circular-rotation test and active-minus-control permutation test",
        "preprocessing_version": "unsmoothed 100-ms retention bins (0.6-3.6 s post-cue), baseline-frozen channel z-score/PCA, 3 latent components, author-native SASS pipeline",
        "inferential_method": "participant-bootstrap circular-harmonic amplitude/phase with a rotation-null p-value; participant-label permutation for the active-vs-control difference (5000 permutations)",
        "correction_family": "G3 perturbation candidate family",
        "status": "replication_or_informative_null",
        "gate": "G2",
        "caveat": "The artifact's own claim_gate field records G3 as 'candidate_only_pending_artifact_sensitivity', not passed, so G3 vocabulary is not licensed here. No population-level phase modulation of state-space diffusion in the active group (circular-rotation p=0.80) or control group (p=0.52); the active-minus-control difference does not exclude the null (permutation p=0.49). Legacy-increment diffusion and behavior log-odds are likewise null in both groups (all p>0.34). This is an informative null on a real, author-native-preprocessed causal design, not a nonidentified result: SASS-gate pass rate is 42/46 (91.3%), with 4 participants excluded for failing the post-SASS artifact-rejection QC.",
        "model_prediction": "If closed-loop tACS is phase-specific, the active group's diffusion should show a circular-harmonic dependence on stimulation phase that the control (sham/passive) group does not, i.e. an active-minus-control difference excluding zero.",
        "prediction_match_status": "null: no group shows a phase effect and the active-minus-control contrast does not exclude zero (p=0.49); G3 remains unestablished for this dataset.",
    },
    "boran_modality_consistency": {
        "claim_id": "current::boran_spike_lfp_confinement_rate_agreement",
        "construct": "whether co-located single-unit and LFP high-gamma confinement rates (lambda) agree within patient, on identical trials/folds (CRACK-4)",
        "eligibility_rule": "provenance/canonical_recording_registry.json filtered to release=='000574' (37 sessions, 8 patients with at least one processed session); spike and LFP arms share identical trial sets and StratifiedKFold seeds per session",
        "independent_unit": "patient (8 with at least one processed session; only 1 with a jointly-identifiable moment-estimator fold)",
        "estimand": "log(spike_lambda / lfp_lambda), moment (autocovariance) estimator primary, state-space secondary, computed only on folds where both modalities are independently identifiable",
        "preprocessing_version": "spike arm reuses scripts/run_human_drift_spine_000574.py machinery; LFP arm uses src/preprocessing.py load_boran_nwb(reject_channels=True, mains_hz=50.0) + compute_boran_hgp (70-150 Hz Hilbert envelope), 50 Hz mains (Sarnthein lab, Zurich, not 60 Hz) including the now-fixed line-noise-aware reject_bad_channels",
        "inferential_method": "patient-level moment/state-space log-ratio with fold-level CI-overlap fraction; no group-level pooled inference below this project's own small-N population-inference floor",
        "correction_family": "descriptive; frozen decision-rule family not applied at the group level for this comparison",
        "status": "current_exploratory",
        "gate": "G1",
        "caveat": "26/37 sessions completed processing; jointly-identifiable moment-estimator folds exist for only 1/8 patients (sub-02: log ratio -0.553, spike lambda ~0.58x the LFP lambda in that patient only) and 0/8 for the state-space estimator. Reported as single-patient descriptive case evidence (below this project's own small-N population-inference floor), not a population-level spike-LFP agreement or disagreement claim. CRACK-4's modality-consistency half is therefore not resolvable at the currently available identifiable sample.",
        "model_prediction": "If the same underlying WM state confinement is visible in both co-located single-unit and LFP high-gamma signals, spike- and LFP-derived lambda should be of the same order of magnitude and their log-ratio CI should not be extreme, on the same trials/folds.",
        "prediction_match_status": "not_estimable_at_population_level: only one patient has a jointly-identifiable comparison, insufficient to accept or reject modality agreement.",
    },
    "ram_stimulation_drift": {
        "claim_id": "current::ram_stimulation_displacement_encoding",
        "construct": "whether human intracranial stimulation at episodic encoding displaces a memory-relevant state, normalized by each session's own endogenous confinement scale",
        "eligibility_rule": "ds005489 open-loop: 75/76 candidate sessions complete, item/pair-level stim genuinely randomized (stim_list & within-list block order); ds005557 closed-loop: 29/29 candidate sessions complete, list-level stim randomized, item-level stim classifier-triggered (non-random)",
        "independent_unit": "word/trial for the pooled legacy test; session for the primary normalized estimand and for closed-loop list-level; subject for random-effects grouping",
        "estimand": "open-loop: normalized_displacement (RMS PC1 deviation / endogenous confinement scale) regressed on stim, mixed model with subject random effect; closed-loop: list-level stim-vs-no-stim normalized displacement (causal) and item-level classifier-triggered descriptive pattern (explicitly non-causal)",
        "preprocessing_version": "author-native ds005557 classifier recipe verbatim (bipolar montage, 0-1366 ms post-word window, 58-62 Hz 4th-order Butterworth band-stop, Morlet wavenumber 5 at 8 log-spaced frequencies 3.0-180.3 Hz, 1365 ms mirrored buffers, log power, within-session z-transform computed from unperturbed/plant trials only)",
        "inferential_method": "mixed-effects regression (subject random intercept) for the pooled/legacy and primary normalized open-loop tests; permutation test for the closed-loop list-level and item-level descriptive contrasts",
        "correction_family": "G3 perturbation candidate family; item-level closed-loop explicitly excluded from the causal family",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "Stimulation is delivered during episodic ENCODING, not WM maintenance -- these results cannot support a WM-maintenance claim, only whether stimulation displaces a memory-relevant state at encoding. Open-loop: the raw/legacy (non-normalized, not cross-session-comparable) displacement-vs-stim test is significant (beta=0.704, p=1.3e-7, n=16176, 75 subjects), but the primary, cross-session-comparable normalized estimand does NOT clear conventional significance (beta=-0.040, p=0.0502, n=1260 rows with identified confinement scale, 7 subjects) -- reported as a null result, not a positive causal result, since only the normalized estimand is licensed as the primary claim. Closed-loop list-level causal test is not_estimable (only 1/29 sessions had identifiable confinement in both list groups); closed-loop item-level pattern (mean diff 0.255, CI [-0.309, 0.824], p=0.41) is carried with causal=False, descriptive_only=True throughout, per its own propensity-selection design flaw -- never reported as a causal effect.",
        "model_prediction": "If stimulation displaces the memory-relevant state relative to its own endogenous confinement scale, the normalized-displacement-vs-stim coefficient should exclude zero in the randomized open-loop item/pair contrast; the closed-loop item-level classifier-triggered contrast is not a causal test by design and is reported descriptively only.",
        "prediction_match_status": "null for the primary normalized open-loop estimand (p=0.0502, does not clear alpha=0.05); not_estimable for closed-loop list-level; item-level closed-loop is non-causal by design, consistent with the predeclared limitation.",
    },
}

CURRENT_OVERRIDES.update({
    "macaque_pfc_microstimulation_design_corrected": {
        "claim_id": "current::macaque_pfc_microstimulation_design_gate_and_recovery",
        "construct": "whether the public macaque PFC microstimulation release identifies the randomized targeting estimand, plus descriptive neural recovery",
        "eligibility_rule": "shared original-trial/block indexing must be recoverable for causal targeting; stimulation contacts must all survive release-provided electrical-short QC for targeting comparisons",
        "independent_unit": "session for descriptive recovery; no licensed causal unit because randomized blocks are unrecoverable",
        "estimand": "design-identification gate and recovery-versus-endogenous confinement agreement",
        "inferential_method": "release-structure audit and session-bootstrap descriptive recovery",
        "correction_family": "design audit precedes all outcome modeling",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "correct/error files have no shared original-trial key, timestamp, block ID, or target-angle allocation; Sa has no eligible stimulation pattern after outcome-independent short-channel QC",
        "model_prediction": "a causal test requires recoverable randomized blocks; a recovery claim requires enough sessions with precision-identified endogenous lambda",
        "prediction_match_status": "causal design no-go; 13 descriptive recovery patterns retained, but recovery/endogenous agreement is not established (3 patterns from one session have precision-identified endogenous lambda)",
    },
    "dynamax_dependency_audit": {
        "claim_id": "current::dynamax_model_api_audit",
        "construct": "availability of trainable Poisson and switching state-space comparators",
        "eligibility_rule": "Dynamax 1.0.2 installed in the analysis environment and local class methods inspected",
        "independent_unit": "software API",
        "estimand": "whether the required model has an implemented training path",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "class exposure is not equivalent to a trainable estimator",
        "model_prediction": "a usable dependency must implement parameter fitting for PLDS or SLDS/rSLDS",
        "prediction_match_status": "not matched: exposed generalized and switching model classes have unimplemented EM steps",
    },
    "gpslds_comparator": {
        "claim_id": "current::gpslds_comparator_audit",
        "construct": "availability of a trainable and commensurately scored modern switching comparator",
        "eligibility_rule": "isolated install, Poisson training smoke test, and held-out predictive-API inspection",
        "independent_unit": "software API",
        "estimand": "whether gpSLDS can enter the frozen held-out likelihood comparison",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "training ELBO is not a held-out predictive likelihood and cannot substitute for one",
        "model_prediction": "a deciding comparator must train and score held-out trials with the same apparatus as M2",
        "prediction_match_status": "not identified: training works, but no held-out filtering or predictive score is exposed",
    },
    "drift_positive_control_000469": {
        "claim_id": "current::dandi000469_temporal_dependence_controls",
        "construct": "confined dynamics versus noise flexibility, session nonstationarity, and population-wide predictive history",
        "eligibility_rule": "same grouped held-out folds and targets in DANDI 000469 and 000574, raw and Anscombe count arms",
        "independent_unit": "patient",
        "estimand": "M2 advantages over matched-flexibility baselines and own-minus-neighbour trial prediction",
        "inferential_method": "patient bootstrap with matched false-positive and recovery simulations",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "the M2 mechanism fails and the retained own-history effect does not separate from off-content axes",
        "model_prediction": "temporal dependence requires intervals above zero against both flexible baselines and for own-minus-neighbour prediction",
        "prediction_match_status": "not matched for confined dynamics in either cohort; matched for population-wide own-minus-neighbour predictive history in DANDI 000469",
    },
    "crossnobis_content_000469": {
        "claim_id": "current::dandi000469_crossnobis_content_decay",
        "construct": "unbiased ratio-scale content-distance decay for H4",
        "eligibility_rule": "native-unit unsmoothed counts, fold-frozen transforms, and decay fits away from optimization bounds",
        "independent_unit": "patient",
        "estimand": "crossnobis content-distance matrix and bounded exponential decay timescale",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "only four patients yield bounded timescale estimates",
        "model_prediction": "1/lambda should predict crossnobis content-distance decay without a fitted scale",
        "prediction_match_status": "non-identified at the available patient count",
    },
    "switching_adjudication": {
        "claim_id": "current::switching_noise_scale_adjudication",
        "construct": "dynamics switching versus heteroscedastic confined drift",
        "eligibility_rule": "all five datasets with an existing M4 score; identical held-out folds and targets",
        "independent_unit": "patient or animal with folds nested within unit",
        "estimand": "free and tied-variance M4-minus-M2, M4-minus-heteroscedastic-drift, and fitted-model recovery",
        "inferential_method": "held-out likelihood, patient summaries, and at least 200 fitted-model simulations per dataset and direction",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "the Gaussian AR-HMM is not a Poisson recurrent switching LDS; the dependency audit is reported separately",
        "model_prediction": "dynamics switching requires a positive tied-variance advantage and exclusion of both heteroscedastic drift and the fitted-M2 null",
        "prediction_match_status": "determined by the completed adjudication artifact",
    },
    "rotation_estimator_floor": {
        "claim_id": "current::deterministic_rotation_adjudication",
        "construct": "trial-shared deterministic coding-axis rotation",
        "eligibility_rule": "DANDI 000469, DANDI 000574, and Miller folds with a fitted content or condition axis plus matched-SNR planted recovery",
        "independent_unit": "patient",
        "estimand": "M1-minus-M0, M3-minus-M2, observed-minus-stationary-floor rotation, and counter-rotation accuracy recovery",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "only 000469 has an identified recovery bound (8 rad/s); 000574 is underpowered through 8 rad/s and Miller's stationary floor is invalid",
        "model_prediction": "real deterministic rotation must exceed the matched stationary estimator floor and improve counter-rotated held-out decoding",
        "prediction_match_status": "determined by the completed rotation artifact",
    },
    "hierarchical_confinement_000469": {
        "claim_id": "current::dandi000469_hierarchical_confinement",
        "construct": "group confinement rate with partial pooling across weak individual folds",
        "eligibility_rule": "positive finite fold likelihood approximations below declared lambda/diffusion divergence bounds",
        "independent_unit": "patient",
        "estimand": "log-scale group geometric-mean lambda, patient shrinkage estimates, and log-ratio selection-controlled anisotropy contrasts",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "the former raw moment fit failed and the state-space versus moment group magnitude remains inconsistent",
        "model_prediction": "partial pooling should identify a positive group confinement rate and may resolve anisotropy without dropping weak folds",
        "prediction_match_status": "determined by the completed hierarchical artifact",
    },
    "geometry_from_drift_parameters_000469": {
        "claim_id": "current::dandi000469_parameter_free_geometry_prediction",
        "construct": "measured geometry as a consequence of fitted confined drift",
        "eligibility_rule": "patients with guarded log-scale lambda and bounded crossnobis or probe-dispersion measurements",
        "independent_unit": "patient",
        "estimand": "correlation and calibration slope for 1/lambda crossnobis timescale and D/lambda probe dispersion",
        "status": "current_exploratory",
        "gate": "G2",
        "caveat": "PR is explicitly non-estimable because the available measurement is superseded and scalar drift does not determine full-covariance PR",
        "model_prediction": "parameter-free predicted geometry should correlate with observations with calibration slope one",
        "prediction_match_status": "determined by the completed H4 artifact",
    },
})

CURRENT_OVERRIDES.update({
    "macaque_pfc_microstimulation_site_reproducibility": {
        "claim_id": "current::stimulation_site_direction_reliability",
        "dataset_view": "macaque PFC microstimulation",
        "construct": "within-session site specificity and across-session reproducibility of stimulation-evoked population displacement direction",
        "eligibility_rule": "sessions with control and stimulated trials at matched target angles; repeated-session comparisons use only channels shared by both sessions",
        "independent_unit": "session for within-session reliability (4 contributing sessions) and repeated session pair for across-session reproducibility (1 contributing pair)",
        "estimand": "same-site minus different-site mean displacement-direction cosine",
        "preprocessing_version": "control-fit session latent space within session; raw firing-rate space on shared channels across sessions",
        "inferential_method": "site-label permutation within session or repeated-session pair, with paired-difference minimum detectable effect",
        "correction_family": "two pre-declared reliability tiers reported separately",
        "status": "current_confirmatory",
        "gate": "G3",
        "caveat": "within-session site identity is detectable (contrast 0.5546, p=0.0217, 4 sessions); across-session reproducibility is underpowered (contrast 0.4413, p=0.499, 1 contributing pair, minimum detectable difference not computable). Amplitude is fixed within each session, so this result cannot identify intensity-response scaling",
        "model_prediction": "same-site displacement directions should be more aligned than different-site directions within a session and when a channel set is repeated across sessions",
        "prediction_match_status": "matched within session; inconclusive across sessions because only one repeated-session pair contributed at least two comparable sites",
    },
    "randomised_prestimulation_moderation_open_loop": {
        "claim_id": "artifact::randomised_prestimulation_moderation_open_loop_source",
        "dataset_view": "human open-loop intracranial stimulation during free-recall encoding",
        "construct": "causal moderation of the stimulation effect by the immediately preceding neural state",
        "eligibility_rule": "experimenter-scheduled word stimulation with a preceding within-list word, at least 3 stimulated and 3 non-stimulated lists per session, and a computable subject-level partial effect",
        "independent_unit": "subject (33 analysed subjects; 71 sessions and 18,007 word events are repeated measures)",
        "estimand": "subject-mean partial correlation for prestimulation-state by stimulation interaction, separately for current-word displacement and later recall",
        "preprocessing_version": "stimulated bipolar contacts excluded; within-session directional deviation; serial position, alternation phase, list number, and preceding-word stimulation controlled",
        "inferential_method": "subject-cluster bootstrap interval and participant-wise list-label permutation preserving list assignment; word-level shuffles and pooled random-intercept fits are diagnostic only",
        "correction_family": "two pre-declared outcomes with native, bias-only, and preceding-word-unstimulated sensitivity arms",
        "status": "superseded",
        "gate": "G3",
        "caveat": "source estimates and list-label permutation statistics are retained, but its stored branch labels were created from generic non-design-based significance flags; use moderation_design_based_re_adjudication for every treatment conclusion",
        "model_prediction": "if the immediately preceding neural state determines whether randomised stimulation helps or hurts, the native interaction should differ from zero while the bias-only control does not reproduce it",
        "prediction_match_status": "superseded by design-based re-adjudication",
    },
    "moderation_design_based_re_adjudication": {
        "claim_id": "current::randomised_prestimulation_state_moderation",
        "dataset_view": "human open-loop intracranial stimulation during free-recall encoding",
        "construct": "moderation of the randomized stimulation contrast by the immediately preceding neural state",
        "eligibility_rule": "same admitted words and subjects as the source analysis; every decision arm must contain a finite stored list-label permutation result",
        "independent_unit": "subject for aggregation; list for treatment randomization (33 subjects, 71 sessions, 18,007 word events)",
        "estimand": "subject-mean partial correlation for prestimulation-state by stimulation interaction, separately for current-word displacement and later recall",
        "preprocessing_version": "deterministic re-adjudication of stored estimates and list-label permutation statistics, linked to the source artifact by SHA-256",
        "inferential_method": "participant-wise list-label permutation preserving the observed within-list stimulation pattern; word-level shuffles retained as diagnostic only",
        "correction_family": "two pre-declared native outcomes; restricted, bias-only, and nuisance-adjusted arms are sensitivities",
        "status": "current_confirmatory",
        "gate": "G3",
        "caveat": "native displacement is -0.00534 (95% CI -0.0332 to 0.0231, list-label p=0.7824, MDD=0.0407) and native recall is +0.00216 (95% CI -0.0158 to 0.0227, list-label p=0.8024, MDD=0.0293). Neither native result reaches the treatment-test threshold; each MDD is below the 0.14 project reference under the analysis assumptions. This is not an equivalence test or evidence of exactly zero moderation. The restricted displacement sensitivity reaches p=0.0439 before added nuisance adjustment and p=0.0539 after it, so it is not a policy result.",
        "model_prediction": "if the immediately preceding neural state determines whether randomized stimulation helps or hurts, the native interaction should differ from zero under the list-assignment randomization test while the bias-only control does not reproduce it",
        "prediction_match_status": "not matched in either native outcome; the restricted displacement result is a fragile sensitivity",
    },
})


# Explicit overrides for the six controlled findings (F1-F6) that carry this project's current
# result, so the heuristic fallback in row_for() cannot mislabel their construct, gate, or
# inferential method the way it would for an unreconciled legacy artifact.
CURRENT_OVERRIDES.update({
    "state_persistence_lag": {
        "claim_id": "current::human_cross_unit_population_state_existence_by_lag",
        "construct": "trial-specific cross-unit population state, existence and lag range, per-unit permutation null",
        "eligibility_rule": "DANDI 000469 (session, epoch) pairs with a fitted population state at each of three window widths (2, 3, 5 bins) and one lag grid per width; deciding width is the one reaching the most lags above the floor per recording window",
        "independent_unit": "(session, epoch) pair (72 pairs at the deciding width; not further clustered by patient in this artifact)",
        "estimand": "per-lag mean permutation contrast (observed cross-unit statistic minus its per-unit-shuffled null) and its FDR q-value, at each of three window widths",
        "inferential_method": "per-unit permutation null (units reshuffled against one another, each unit's own temporal statistics preserved) with Benjamini-Hochberg FDR across the lag grid, at three window widths",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "at the deciding width (3 bins) 20 of 25 tested lags clear FDR contiguously from 0.3 to 2.2 s; the five longest lags tested, 2.3-2.7 s, do not clear. Existence agrees across all three tested widths (width 2: 24/27 lags, 0.2-2.6 s; width 3: 20/25, 0.3-2.2 s; width 5: 19/21, 0.5-2.3 s); the artifact's own coarser whole-range-slope branch does not agree across widths, which is a shape question this artifact does not decide (see results/state_persistence_shape.json). The lag range is human; the cross-species range that may be quoted is 0.3-0.8 s because the macaque window ends 1.45 s after cue onset. The same artifact's encoding-epoch companion arm (26 sessions, 15 lags) shows comparable effect sizes (existence positive at 15/15 lags, raw p as low as 0.0147) but does not clear FDR at that smaller sample and lag count -- this is a power difference between epochs at this sample size, not evidence that the state is delay-specific, and must not be read as one.",
        "model_prediction": "a trial-specific population state above each unit's own spike statistics should clear a per-unit permutation null at a contiguous, above-chance-floor range of lags.",
        "prediction_match_status": "matched at every tested width; existence is width-robust and the surviving lag range is reported as human-specific.",
    },
    "state_persistence": {
        "claim_id": "current::human_cross_unit_population_state_existence_primary",
        "construct": "trial-specific cross-unit population state, primary existence contrast underlying the lag sweep",
        "eligibility_rule": "same DANDI 000469 (session, epoch) pairs and per-unit permutation null as results/state_persistence_lag.json; this artifact carries the primary contrast and cross-corpus (ALM, macaque) companion rows the lag artifact's sweep is built on",
        "independent_unit": "(session, epoch) pair, human; session, mouse ALM and macaque lateral PFC",
        "estimand": "cross-unit permutation contrast and its significance at the deciding window width, per corpus",
        "inferential_method": "per-unit permutation null, same construction as results/state_persistence_lag.json",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "carries the human primary rows, the ALM comparison rows, and the calibration-ladder companions that results/state_persistence_lag.json's lag sweep and results/state_persistence_shape.json's breakpoint fit both build on; read together with those two artifacts, not in isolation.",
        "model_prediction": "same as results/state_persistence_lag.json.",
        "prediction_match_status": "matched; see results/state_persistence_lag.json for the lag-resolved detail.",
    },
    "state_persistence_shape": {
        "claim_id": "current::human_cross_unit_state_autocorrelation_shape",
        "construct": "two-component (fast-plus-floor) shape of the cross-unit population state's autocorrelation, and the identifiability of any timescale fitted to it",
        "eligibility_rule": "same 72 human (session, epoch) pairs as results/state_persistence_lag.json at the deciding window width (3 bins), with a fitted breakpoint (0.8 s) segmenting the lag range into early and late slopes; mouse ALM (23 sessions) and macaque lateral PFC (25 sessions, off-branch) companion arms at the same deciding width",
        "independent_unit": "(session, epoch) pair, human; session, mouse ALM and macaque",
        "estimand": "early-segment and late-segment permutation-contrast slopes either side of the fitted breakpoint, a double-difference test between them, and a five-rung planted-timescale recovery ladder (rank correlation of recovered versus planted time constant)",
        "inferential_method": "segmented-slope fit at a fitted breakpoint, per-unit permutation null, double-difference test between segments, and a pre-declared planted-recovery identifiability criterion (rho > 0.5 at p <= 0.05)",
        "status": "current_confirmatory",
        "gate": "G2",
        "caveat": "human early segment (0.3-0.8 s) mean -0.11890, one-sided p=0.0001, n=72; late segment (0.8-2.7 s) mean +0.01727, p=0.9411; the double difference between them is not significant (two-sided p=0.4192), so this is reported as a floor bound at this sensitivity, not a plateau -- 'a persistent component', 'a stable state' and 'activity-silent's opposite' are not licensed by this artifact. Mouse ALM does not show the fast component at the deciding width (p=0.2201, n=23; it appears only at a narrower width, the reverse robustness pattern from the human arm) and must never be described as matching the human rate. The macaque arm's contrast rises rather than declines at the deciding width and is off this artifact's own branch list. The five-rung planted-timescale ladder converges on every rung without hitting a fit bound, yet recovered-versus-planted rank correlation is rho=-0.30 (p=0.68, n=5), so the pre-declared identifiability criterion returns false and no tau in seconds is reported.",
        "model_prediction": "a genuine two-component autocorrelation should show a significantly negative early slope, a late slope and a breakpoint-to-endpoint double difference distinguishable from a flat floor, and a planted-timescale ladder whose recovered values track the planted ones.",
        "prediction_match_status": "partially matched: the early decline is confirmed at the deciding width and one companion width; the late-segment plateau is a floor bound, not a confirmed plateau (double difference not significant); the identifiability criterion for any timescale in seconds is not met.",
    },
    "state_content_link": {
        "claim_id": "current::content_in_dominant_state_cross_species_dissociation",
        "construct": "whether the memorandum is carried in the dominant shared population state, via a leave-one-latent-out deletion-cost observable (fractional rank, chance 0.5), matched at k=8 latents across species",
        "eligibility_rule": "sessions with a content decoder clearing its own permutation null: macaque lateral PFC (Panichello, 25 of 25 sessions with k=8), mouse ALM (Inagaki, 23 of 23 with k=8), DANDI 000469 (8 of 61 clear their own null; DANDI 001187 excluded by a label-granularity criterion, not by a property of its recordings)",
        "independent_unit": "session, within each corpus",
        "estimand": "mean fractional rank of the dominant latent's deletion cost, per corpus, against chance 0.5",
        "inferential_method": "leave-one-latent-out deletion-cost ranking with a two-sided test against chance, per corpus; the pooled cross-corpus mean is computed but not used as a corpus-level result because the two corpora with an askable question have opposite-signed effects",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "macaque lateral PFC: mean fractional rank 0.6743, two-sided p=0.0186, n=25 -- content is NOT in the dominant state, reproducing Murray et al. 2017 in an independent dataset rather than being a novel result. Mouse ALM: mean 0.1429, p=9.999e-05, n=23 -- content IS the dominant state, the expected direction for ALM's own published discrete-attractor account and read as the positive control for the macaque negative. DANDI 000469: mean 0.4249, p=0.1321, not significant either direction, from the 8/61 sessions that clear their own content-decoder null -- reported as a scope limit on what was asked, never as an established null on human recordings. The pooled fractional rank (0.4226, p=0.0414, n=109) averages opposite-signed effects and is superseded by the per-corpus resolution; it is not quoted as a result about either corpus.",
        "model_prediction": "if the memorandum is carried in the dominant shared latent, its deletion should be the most expensive of the k latents to remove (fractional rank near 0.0); if not, deletion cost should not distinguish it from the others or should rank it cheap to delete.",
        "prediction_match_status": "opposite, individually significant answers in the two corpora that can be asked: not matched in macaque (content is not dominant), matched in mouse ALM (content is dominant); not resolvable in the human corpora tested.",
    },
    "rate_free_state_geometry_behavior_link": {
        "claim_id": "current::rate_free_geometry_predicts_trial_accuracy",
        "construct": "rate-free deviation of single-trial state geometry from its session mean, correlated with trial accuracy, with firing-rate confounding excluded by construction and by a direct control",
        "eligibility_rule": "macaque lateral PFC (Panichello) sessions reaching the same 60-error-trial reachability floor as results/state_behavior_link.json; 11 of 25 sessions reachable",
        "independent_unit": "session (n=11); no patient/animal clustering in this primary estimand (see the animal-block-clustered companion field added in this artifact)",
        "estimand": "pooled paired sign-flip correlation of trial outcome with rate-free geometric deviation, plus the same correlation controlling for spike count and for trial index",
        "inferential_method": "two-sided paired sign-flip test pooling per-session partial correlations, pre-declared decision rule and pre-declared meaningful-effect threshold",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "raw_outcome_vs_deviation r=-0.0974, two-sided p=0.0035, upper CI -0.1405, n=11 -- significant. Rate-free-ness is established directly, not assumed: deviation regressed against spike count alone (orthogonality_gate_deviation_vs_spike_count) gives p=0.7384, i.e. the deviation measure carries no detectable rate signal. The measured |r|=0.0974 is below this project's own pre-declared meaningful_effect_threshold_r_units (0.14, sourced from the persistence null's own minimum detectable paired difference); the effect is significant and its magnitude does not clear that pre-declared bar, and both halves are reported together, never one without the other.",
        "model_prediction": "if state geometry independent of firing rate tracks behaviour, trial-level deviation from the session mean should correlate with accuracy after the deviation measure is shown independent of spike count.",
        "prediction_match_status": "matched for existence and direction; the pre-declared meaningfulness bar is not cleared, and this project has no defensible substitute bar (see caveat).",
    },
    "state_behavior_link": {
        "claim_id": "current::persistence_amplitude_behavior_dissociation_persistence_arm",
        "construct": "correct-versus-error contrast on the persistence-contrast (not the rate-free geometry) observable, matched trial count, as the comparison arm for F4's amplitude-not-persistence dissociation",
        "eligibility_rule": "macaque lateral PFC sessions with at least 60 error trials (pre-declared reachability floor); 11 of 25 sessions reachable",
        "independent_unit": "session (n=11)",
        "estimand": "mean paired difference, matched correct minus error trials, on the persistence-contrast level",
        "inferential_method": "two-sided paired sign-flip test with a pre-declared minimum-detectable-paired-difference companion",
        "status": "replication_or_informative_null",
        "gate": "G1",
        "caveat": "mean difference 0.0749, two-sided p=0.168, n=11, minimum detectable paired difference 0.139 -- an informative null at this sample size, not evidence of no effect. This is the persistence half of F4's amplitude-versus-persistence dissociation; the amplitude (rate-free geometry) half is results/rate_free_state_geometry_behavior_link.json, which is significant.",
        "model_prediction": "if persistence itself (rather than amplitude) tracked behaviour, correct trials should show a higher persistence-contrast level than matched error trials.",
        "prediction_match_status": "not matched: the paired difference does not exclude zero at this sample size, and its own minimum detectable difference (0.139) exceeds the measured rate-free effect size (0.097), so the two are on a common, honestly bounded scale.",
    },
    "behavior_amplitude_rate_controls": {
        "claim_id": "current::naive_gain_behavior_link_is_a_rate_proxy",
        "construct": "whether a naive per-trial leading-component gain's correlation with accuracy survives controlling for total spike count and trial index",
        "eligibility_rule": "same 11 reachable macaque lateral PFC sessions as results/rate_free_state_geometry_behavior_link.json",
        "independent_unit": "session (n=11)",
        "estimand": "partial point-biserial correlation of trial outcome with gain, controlling for total spike count (and, secondarily, trial index and both jointly), pre-declared decision rule on the spike-count-controlled correlation alone",
        "inferential_method": "two-sided paired sign-flip test on per-session partial correlations, pre-declared decision rule stated before fitting",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "raw gain vs accuracy: mean -0.1675, p=0.0060; raw spike count vs accuracy: mean -0.2424, p=0.0013 -- both significant, and gain conditioned on spike count is not (mean -0.0157, p=0.6755, branch amplitude_correlate_of_accuracy_is_rate_not_geometry): the naive gain-accuracy link is a rate proxy, not evidence of an independent geometric account, and is why the rate-free deviation measure in results/rate_free_state_geometry_behavior_link.json (which is shown independent of spike count directly, p=0.7384) is the geometry estimand F4 reports, not raw gain.",
        "model_prediction": "if a naive per-trial gain measure carried an accuracy link independent of firing rate, it should survive conditioning on total spike count.",
        "prediction_match_status": "not matched: the gain-accuracy link is fully accounted for by spike count.",
    },
    "rank1_gain_temporal_profile_closure": {
        "claim_id": "current::per_trial_gain_account_of_dominant_latent_rejected",
        "construct": "whether the macaque dominant latent's rank-1 (per-trial-gain) share exceeds its own matched-noise reference by a margin that would license describing the state itself as a per-trial gain artifact",
        "eligibility_rule": "same 25 macaque lateral PFC sessions as results/state_latent_identity.json's rank1_gain_test, stratified by the 2021 (monkey A) and 2022-plus (monkey H and J) session-date blocks",
        "independent_unit": "session (n=25), reported stratified by recording-date block as well as pooled",
        "estimand": "median observed rank-1 share versus median matched-noise-reference share, pooled and by stratum",
        "inferential_method": "matched-noise-reference comparison, reused from results/state_latent_identity.json without refitting, stratified post hoc by recording-date block",
        "status": "replication_or_informative_null",
        "gate": "G1",
        "caveat": "pooled observed share 0.8950 vs. reference 0.8926 -- statistically but only marginally different (one-sided p=0.0116, a difference of 0.0025, 21/25 sessions individually significant); both strata (2021: 0.6557 vs. 0.6390 reference; 2022-plus: 0.9768 vs. 0.9747 reference) show the same small-margin pattern, so stratifying by animal does not change the reading. A branch label alone ('state_is_a_per_trial_gain') would overstate this: the difference is real but too small to characterise the dominant state as a per-trial-gain artifact, and it does not undermine the rate-free geometry link, a different observable computed independently of gain.",
        "model_prediction": "if the dominant latent were largely a per-trial gain artifact, its rank-1 share should be substantially above a matched-noise reference that shares the same trial-count and session structure.",
        "prediction_match_status": "not matched at a meaningful margin: the difference from the matched-noise reference is statistically detectable but under a third of one percentage point, in both animal strata.",
    },
    "state_latent_identity": {
        "claim_id": "current::dominant_latent_gain_and_position_confound_controls",
        "construct": "whether the macaque dominant latent's lag-related signal survives controlling for cue position, and the size of its per-trial-gain (rank-1) share against a matched-noise reference -- methodological controls feeding F3's content-in-the-state framing and F4's per-trial-gain closure, not a content-decoder result itself",
        "eligibility_rule": "25 macaque lateral PFC sessions with a joint lag/position regression and a position-matched subset",
        "independent_unit": "session (n=25)",
        "estimand": "lag-coefficient and position-coefficient tests from a joint regression, and a position-matched-subset slope test; pooled rank-1-share versus matched-noise reference (see results/rank1_gain_temporal_profile_closure.json for the animal-stratified version)",
        "inferential_method": "joint regression with a position-matched-subset robustness check; matched-noise-reference comparison for rank-1 share",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "lag coefficient significant (mean 0.0665, p=0.0289) while the position coefficient is not (p=0.7956); the position-matched-subset slope is significant and larger (mean 0.0854, p=0.0007), so the branch lag_effect_survives_position holds -- the dominant-latent signal is not an artifact of cue position. Pooled rank-1 share 0.8950 vs. matched-noise reference 0.8926 is statistically but only marginally different (see results/rank1_gain_temporal_profile_closure.json's caveat for the exact numbers and the animal-stratified version); this artifact does not itself carry a content-decoder result (see results/state_content_link.json for that).",
        "model_prediction": "if the dominant latent's signal were a cue-position artifact rather than a lag effect, the position coefficient should be significant and the lag coefficient should not survive controlling for position.",
        "prediction_match_status": "not matched: the reverse pattern holds (lag survives, position does not).",
    },
    "persistence_estimator_split_count_sensitivity": {
        "claim_id": "current::estimator_split_count_does_not_explain_cross_species_gap",
        "construct": "whether the persistence-contrast estimator's own split/replicate-count settings, rather than a real cross-species difference, produce the human-versus-ALM autocorrelation gap",
        "eligibility_rule": "human delay and mouse ALM sessions already fitted for results/state_persistence_shape.json, re-scored under the estimator's native split-count settings for each corpus in addition to the shared default settings",
        "independent_unit": "session, human delay and ALM arms separately",
        "estimand": "early-segment permutation-contrast slope under each arm's native estimator settings versus the shared default, and whether the branch resolution changes",
        "inferential_method": "re-scoring already-fitted sessions under alternative pre-declared estimator settings, no refitting",
        "status": "replication_or_informative_null",
        "gate": "G1",
        "caveat": "both the human-delay and ALM arms resolve to estimator_settings_do_not_account_for_the_cross_species_difference: moving each arm to its own native split-count setting does not close the gap between them, so the estimator's own split-count choice is not a viable alternative explanation for F2's human-versus-ALM shape difference.",
        "model_prediction": "if the estimator's split-count settings drove the apparent species gap, re-scoring each arm under its own native settings should narrow or remove the gap.",
        "prediction_match_status": "not matched: the gap is unchanged under either arm's native settings.",
    },
    "band_versus_sensor_decomposition": {
        "claim_id": "current::depth_low_band_persistence_positive_and_noise_fraction_dissociation",
        "construct": "two observables in the same sessions: the persistence contrast (does the state exist) and a factor-analysis observation-noise fraction (which sensor looks cleaner), for depth versus scalp and low-band versus high-gamma",
        "eligibility_rule": "9 patients, 35-37 paired sessions depending on bin width and comparison; same patients, sessions, band and pipeline across both observables",
        "independent_unit": "session, with a patient-clustered companion field for every paired/one-sample test (9 patients)",
        "estimand": "persistence-contrast one-sample and paired tests (depth-vs-zero, depth-minus-scalp, band-vs-band) and factor-analysis noise-fraction paired tests (band effect at fixed sensor, sensor effect at fixed band), at 100 and 200 ms bins",
        "inferential_method": "t-test and Wilcoxon (one-sample), two-sided paired sign-flip test (paired), pre-declared branch rule keyed to the noise-fraction observable's sign and significance",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "persistence contrast: depth low-band mean +0.1185 (100 ms, p=3.7e-10, 33/36 positive), scalp indistinguishable from zero (p=0.589); paired depth-minus-scalp +0.108 (p=1.3e-05). Band-vs-band on the persistence contrast is unresolved (mean +0.003, p=0.9415, minimum detectable difference 0.108) -- report unresolved, not a small cost. On the SEPARATE pre-declared noise-fraction observable, scalp ranks at or below depth (band effect p=0.0063; sensor effect mean -0.069/median -0.008, p=0.0215): the sensor that looks cleaner on the noise-named observable is exactly the one on which the persistence contrast is not measurable, so low observation noise is not sufficient for state observability. Both observables must be named explicitly wherever this artifact is cited; neither statement stands alone. Matched-modality and sensor comparisons carry two matching failures (trial counts differ in 25/26 sessions; k mismatched in 16/36 sessions) that must accompany any number drawn from them.",
        "model_prediction": "if high gamma were required for this state, depth restricted to 8-45 Hz should not carry the persistence contrast; if observation noise explained the scalp null, scalp should rank worse than depth on a noise-named observable.",
        "prediction_match_status": "the first prediction is falsified (depth carries the state at low band); the second is falsified in the wrong direction (scalp ranks at or below depth on the noise-fraction observable while still failing to carry the persistence contrast) -- this is F5's central dissociation.",
    },
    "observability_census": {
        "claim_id": "current::human_versus_alm_nugget_fraction_floor",
        "construct": "cross-validated nugget-fraction estimator, human versus mouse ALM, the raw census this project's construct-validity work subsequently bounds",
        "eligibility_rule": "57 human sessions across the corpora carrying this estimator, 23 mouse ALM sessions",
        "independent_unit": "session",
        "estimand": "median nugget fraction, human versus ALM, trial-count-matched and unmatched",
        "inferential_method": "Mann-Whitney U, matched and unmatched",
        "status": "current_exploratory",
        "gate": "G1",
        "caveat": "human median nugget fraction 0.925 versus ALM median 0.0 (branch human_observability_floor_confirmed, p<2e-9 matched and unmatched). This is the raw finding that results/observation_noise_estimator_construct_validity.json subsequently bounds: the nugget fraction is the one of three observation-noise estimators that separates the species, ALM sits exactly at that estimator's own zero floor, and that floor is reachable at substantial true noise in synthetic data -- so this artifact's own headline is not read as an established species difference without that companion artifact.",
        "model_prediction": "if mouse ALM were genuinely less observation-noise-limited than the human corpora, its nugget fraction should sit below the human range.",
        "prediction_match_status": "matched at face value; the construct-validity companion artifact bounds how far this can be trusted as a species statement (see results/observation_noise_estimator_construct_validity.json).",
    },
    "observability_and_power_census": {
        "claim_id": "current::grain_admission_set_and_margin_aware_comparison",
        "construct": "whether the LFP and single-unit pipelines admit the same observable set and how their estimator-fit margins differ",
        "eligibility_rule": "Boran DANDI 000574 sessions with both LFP and single-unit views processed",
        "independent_unit": "session for the descriptive admission census; participant for the companion fittability test",
        "estimand": "admitted observable set per grain at 100 and 200 ms bins; summed rank difference (LFP minus unit) over shared observables",
        "inferential_method": "descriptive set-membership and rank-margin comparison, bounded by a participant-level sign-flip companion test",
        "status": "current_exploratory",
        "gate": "G1",
        "caveat": "At 100 ms both grains admit an identical seven-observable set, which is a limit of the set-valued rule rather than evidence of equivalence. The margin ranking favours the LFP pipeline, but 25 of 26 paired session keys have different trial counts and unit/contact counts are not equalized. The participant-level companion test does not separate fittability, so the margin is not evidence of anatomical or modality value.",
        "model_prediction": "if single units were the more informative grain, unit-grain sessions should be preferentially admitted and rank ahead of LFP on the shared observable set.",
        "prediction_match_status": "unresolved: the descriptive margin favours LFP, but participant-level fittability does not separate the grains and trial/population counts remain confounded.",
    },
    "observability_matched_modality_test": {
        "claim_id": "current::lfp_versus_unit_grain_participant_fittability",
        "construct": "participant-level difference in estimator fittability between LFP and single-unit pipelines over matched session keys",
        "eligibility_rule": "Boran DANDI 000574 patient/session keys with both grains attempted; grain-specific admitted trials retained and disclosed",
        "independent_unit": "participant (8); repeated session indicators averaged within participant",
        "estimand": "participant mean of the session-level LFP-fitted indicator minus the unit-fitted indicator",
        "inferential_method": "two-sided participant sign-flip test with participant bootstrap interval; session-level exact count is diagnostic only",
        "status": "current_exploratory",
        "gate": "G1",
        "caveat": "At 200 ms the participant-mean LFP-minus-unit fittability difference is 0.083 (95% CI [-0.188, 0.354], p=0.742); at 100 ms it is -0.007 (95% CI [-0.075, 0.054], p=1.0). Twenty-five of 26 session keys have different trial counts and channel counts are unmatched, so these are bounded pipeline-fittability results, not equivalence tests or modality-value estimates.",
        "model_prediction": "if one grain were more fittable in general, discordant sessions should favour it asymmetrically.",
        "prediction_match_status": "not detected at the participant level; modality interpretation remains nonidentified.",
    },
    "observation_noise_estimator_construct_validity": {
        "claim_id": "current::observation_noise_estimator_answer_key",
        "construct": "synthetic ground-truth construct validity of two estimators this project has called observation noise (factor-analysis noise-variance fraction; cross-validated nugget fraction), under diagonal and spatially correlated noise, and at each species' own smallest sessions",
        "eligibility_rule": "synthetic populations at human and mouse-ALM sample sizes, two noise-covariance models (diagonal, rank-4 spatially correlated white), a true-noise-fraction grid (0.05 to 0.95) and a true-dimensionality grid (1 to 12)",
        "independent_unit": "simulated seed (30 per grid point); this is method-validation evidence, not a biological result",
        "estimand": "resolved branch (tracks_noise_with_a_confound / tracks_dimensionality / recovers_nothing / recovers_the_noise_fraction) per estimator and noise model, plus zero-floor reachability and each species' real-data value against the synthetic-population range",
        "inferential_method": "pre-declared branch-resolution rules over the simulation grid, applied identically to both estimators and both noise models",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "synthetic method-validation evidence only; it is not a biological result. Factor-analysis fraction resolves tracks_noise_with_a_confound under diagonal noise (span 0.789 across the true-noise grid at fixed dimensionality; a real but smaller dimensionality confound, span 0.160) and recovers nothing under spatially correlated noise (median absolute error 0.335), the volume-conducted regime scalp EEG actually sits in. The nugget fraction returns near-exactly zero at true noise up to 0.80 (diagonal) / 0.40 (correlated), and at the wide bin width does not fit for mouse ALM at all. Applied to real data: factor-analysis fraction (0.708) and participation ratio (4.45) place mouse ALM inside the human range; only the nugget fraction separates the species, sitting exactly at its own zero floor. Neither instrument supports attributing an absent state in this project to observation noise.",
        "model_prediction": "an estimator that genuinely measures observation noise should be monotone in true noise fraction under both noise models and should not return near-zero at high true noise.",
        "prediction_match_status": "partially matched: the factor-analysis fraction behaves as designed under its assumed diagonal covariance and fails outside it (an expected consequence of a stated assumption, not a hidden defect); the nugget fraction floors at moderate-to-high true noise under both models.",
    },
    "latent_model_observation_noise_comparison": {
        "claim_id": "current::two_named_noise_quantities_are_uncorrelated_in_real_data",
        "construct": "real-data agreement between the two quantities this project has called observation noise, across every cell where both are computed",
        "eligibility_rule": "32 real (dataset, structure, bin-width) cells with both the factor-analysis noise-variance fraction and the cross-validated nugget fraction computed",
        "independent_unit": "cell (dataset x structure x bin width)",
        "estimand": "Pearson and Spearman correlation between the two quantities across the 32 cells",
        "inferential_method": "Pearson and Spearman correlation with their own p-values, no simulation",
        "status": "current_confirmatory",
        "gate": "G1",
        "caveat": "Pearson r=-0.005 (p=0.98), Spearman rho=-0.161 (p=0.38) across 32 real cells: the two quantities this project has called observation noise are uncorrelated, so at most one of them measures it. This is real-data evidence, not synthetic, and is the empirical motivation for the synthetic construct-validity work in results/observation_noise_estimator_construct_validity.json.",
        "model_prediction": "if both quantities measured the same underlying observation-noise construct, they should be positively correlated across cells where both are computed.",
        "prediction_match_status": "not matched: no detectable correlation.",
    },
    "minimum_recording_specification": [
        {
            "claim_id": "historical::common_grid_recording_recovery_guidance",
            "result_id": "result::minimum_recording_specification::historical_threshold_guidance",
            "dataset_view": "five open spiking corpora analysed separately",
            "construct": "held-out recovery of the project-specific rate-free deviation from common channel and calibration-trial ladders",
            "eligibility_rule": "shared corpus loaders followed by fixed reference/score split, numeric rungs available to every eligible session, and a complete joint-grid cohort",
            "independent_unit": "session; whole-session bootstrap for threshold uncertainty",
            "estimand": "sustained crossing of 50%, 70%, and 90% of each corpus's own median native recovery after subtraction of a matched random-axis correlation",
            "preprocessing_version": "fixed score set, reference-only calibration ladder, and 50 native null draws",
            "inferential_method": "session-median common-grid curves and 2000-draw whole-session percentile bootstrap with explicit right censoring",
            "correction_family": "three descriptive fidelity targets per corpus; no cross-corpus pooled claim",
            "status": "superseded",
            "disposition": "withdrawn_claim_retained_as_historical_record",
            "gate": "G1",
            "caveat": "The matched random-axis null does not remove the upward shared-reference geometric contribution, so crossings are neither bias-free biological recovery thresholds nor universal device specifications. No corpus sustains the 90% target on its common numeric channel grid; two do not reach 50% there. The proposed minimum recording specification is withdrawn because the corresponding within-corpus channel/trial ladder did not explain the cross-corpus observability split.",
            "model_prediction": "if a small common channel count preserved the observable robustly, the corpus curve would reach and remain above 90% of its own native recovery before one quarter of the median native channel count",
            "prediction_match_status": "not matched on the observed common grids; all five 90% crossings are right-censored",
        },
        {
            "claim_id": "current::independent_reference_recording_recovery_diagnostic",
            "result_id": "result::minimum_recording_specification::independent_reference_diagnostic",
            "dataset_view": "five open spiking corpora analysed separately",
            "construct": "channel-count recovery after separating target-reference, candidate-reference, and score trials",
            "eligibility_rule": "sessions admitted by the shared corpus loaders with sufficient trials for three disjoint partitions and each reported channel rung",
            "independent_unit": "session is the biological unit; 50 channel draws per session quantify algorithmic variability and are not independent biological replicates",
            "estimand": "median matched-null-subtracted candidate recovery by channel count using disjoint target-reference, candidate-reference, and score thirds",
            "preprocessing_version": "three-way trial split with the existing channel-rung sampler and matched random-axis subtraction",
            "inferential_method": "descriptive per-corpus medians and interquartile ranges; no cross-corpus pooled test or universal threshold inference",
            "correction_family": "supplementary descriptive diagnostic; no multiplicity-adjusted claim",
            "status": "current_exploratory",
            "disposition": "supporting_diagnostic",
            "gate": "G1",
            "caveat": "Recovery rises with channel count in all five corpora, but the diagnostic pools repeated channel draws and does not identify a universal minimum channel or trial count. The main common-grid fidelity conclusions remain right-censored in at least one target per corpus.",
            "model_prediction": "if shared reference trials explained the recovery curves, separating all three trial roles would remove their increase with channel count",
            "prediction_match_status": "not matched descriptively: recovery still increases with channel count, while threshold interpretation remains censored",
        },
    ],
    "ds006848_feasibility": {
        "claim_id": "current::ds006848_autonomic_control_feasibility",
        "dataset_view": "OpenNeuro ds006848 AlphaDirection1 digit-span EEG, ECG, and PPG release",
        "construct": "metadata and payload feasibility for observational autonomic nuisance control during working-memory retention",
        "eligibility_rule": "BIDS participant, behavior, event, channel, and EEG sidecar files present; signal payload availability and exact participant-condition event-to-behavior counts audited separately",
        "independent_unit": "participant for a future analysis; this artifact is a metadata census with 30 participant directories",
        "estimand": "counts of staged payloads, exact event-to-behavior linkage, channel modalities, outcomes, and retention timing",
        "preprocessing_version": "metadata-only BIDS audit; no EEG, ECG, or PPG sample payload was opened",
        "inferential_method": "deterministic metadata and tabular census without a hypothesis test",
        "correction_family": "not applicable to a feasibility census",
        "status": "current_exploratory",
        "gate": "G0",
        "caveat": "All 30 participants have working-memory behavior and event tables, 15 have complete working-memory signal payloads, and 10 also have exact participant-condition count linkage. The release can support a bounded observational nuisance-control analysis after explicit admission; it supplies no stimulation or causal contrast.",
        "model_prediction": "a feasible autonomic-control analysis requires trial outcomes, retention timing, EEG, ECG, PPG, complete signal payloads, and linkable trials in the same participants",
        "prediction_match_status": "partially matched: all required modalities and labels are represented, but payload and linkage completeness limit the currently admissible subset to 10 participants",
    },
})


# The confinement-rate, switching-model, coding-axis-rotation, and intrinsic-timescale spine
# (this project's own earlier primary result) is retired: the current result is the six
# controlled findings above. Every ledger row that supports one of those four retired claims is
# reclassified to "superseded" here, after CURRENT_OVERRIDES is otherwise fully populated, so a
# row that already carried a full descriptive override keeps its original construct/estimand/
# caveat text (an accurate historical record) and only its status changes -- nothing is deleted,
# per this project's own extend-and-reclassify rule for its evidence ledger.
RETIRED_SPINE_STEMS = (
    "human_drift_spine_000469", "human_drift_behavior_000469", "drift_control_payload_000469",
    "human_drift_spine_001187_000673", "human_drift_spine_000574", "miller_drift_spine",
    "drift_positive_control_000469", "crossnobis_content_000469", "switching_adjudication",
    "rotation_estimator_floor", "hierarchical_confinement_000469", "geometry_from_drift_parameters_000469",
    "dynamax_dependency_audit", "gpslds_comparator", "watters_2026_item_count_drift",
    "watters_2026_source_replication", "haslacher_phase_diffusion", "boran_modality_consistency",
    "ram_stimulation_drift", "region_stratified_drift_000469", "region_stratified_drift_001187_000673",
    "rotation_phase_cate", "rotation_phase_content", "rotation_power_bound", "rotation_speed_axis",
    "structure_pooled_dynamics", "structure_registry", "panichello_2024_drift_switching",
    "alagapan_retention_diffusion", "cross_modality_calibration", "noninvasive_sensor_dynamics",
    "fidelity_controllability_map", "intrinsic_timescale_vs_confinement", "lambda_estimator_limits",
    "structure_control_observables", "structure_identifiability_matched_draws",
    "structure_identifiability_model", "structure_paired_contrasts", "tau_estimator_calibration",
)
RETIRED_SPINE_CAVEAT = (
    "supports the confinement-rate, switching-model, coding-axis-rotation, or intrinsic-timescale "
    "spine, which this project's evidence ledger no longer carries as a current result -- retained "
    "for provenance in the manuscript's exploratory archive, not cited as current evidence"
)
for _stem in RETIRED_SPINE_STEMS:
    _entry = CURRENT_OVERRIDES.setdefault(_stem, {})
    _entry["status"] = "superseded"
    _entry.setdefault("caveat", RETIRED_SPINE_CAVEAT)

# These five audits are current supporting/diagnostic or evidence artifacts in their own right --
# none of them is replaced by a newer estimator, so none is "superseded". Each disposition below is
# an explicit artifact-specific classification and must not be re-derived or loosened to the generic
# superseded/legacy default.
CURRENT_OVERRIDES.setdefault("behaviour_association_bias_only_sweep", {}).update({
    "claim_id": "current::behaviour_association_control_coverage_audit",
    "disposition": "supporting_diagnostic",
    "construct": "census of which delivered behavioural-association artifacts carry a session-identity bias-only control, and whether that control voids the project's surviving headline behavioural link",
    "status": "current_exploratory",
    "caveat": "the audit that established which behavioural claims carry a bias-only control at all: of 231 top-level results/*.json files enumerated, 32 are classified at_risk (a behaviour-outcome correlation with no existing control), 130 not_at_risk, 13 already_controlled, 56 undetermined_from_artifact_contents. Of 11 at-risk cells actually tested against the control, 1 was voided. The headline surviving half (rate_free_state_geometry_behavior_link) was not voided by the pre-declared rule, but the control itself was underpowered at n=11 sessions (whole-session cluster bootstrap minimum detectable difference 0.8177 r units against this project's 0.14 r-unit reference, versus an observed same-sign confound r=-0.4409, p=0.174; joint-partial control r=-0.5700, p=0.104) -- the headline is untested by this control, not confirmed clean of it.",
})
CURRENT_OVERRIDES.setdefault("component_binding_bias_only_control", {}).update({
    "claim_id": "current::swap_association_between_session_control_category_error",
    "disposition": "unresolved",
    "construct": "between-session bias-only control applied to the multi-object macaque swap-versus-imprecision within-session association",
    "status": "current_exploratory",
    "caveat": "the delivered bias-only substitution replaces every trial in a session with that session's own mean, which collapses the within-session estimator's degrees of freedom to zero and forces a different, between-session correlation in its place -- a category error the delivered voiding did not disclose (this control cannot confound a within-session estimator; see results/bias_only_control_validity_simulation.json). Raw swap association mean 0.0198 vs. the between-session control mean 0.3109 (control p=0.0450, same sign, voided under the pre-declared rule); joint-partial real mean 0.0178 vs. control mean 0.2930 (control p=0.0692, not significant, not voided). This voiding is under repair and is not currently a valid kill; the swap association is neither restored as a positive identity nor treated as void pending that repair.",
})
CURRENT_OVERRIDES.setdefault("occupied_subspace_membership_effect_size", {}).update({
    "disposition": "evidence",
    "status": "current_exploratory",
    "caveat": "the delivered in-sample verdict ('the axis lies inside the occupied space') used a degenerate within-subspace null (off-fraction mean ~2e-17) and is not citable on its own. The held-out split-half circularity check in this same artifact is: both corpora's split-half reversal survives against a no-structure control -- single-item macaque lPFC (n=11 sessions) paired real-minus-control difference +0.2873 (whole-session bootstrap se 0.0760, 95% CI [0.1502, 0.4435], minimum detectable difference at 80% power 0.2130); multi-object macaque (n=41 sessions) paired difference +0.1385 (se 0.0170, 95% CI [0.1073, 0.1740], mdd 0.0478). Cite only these paired split-half-vs-no-structure-control effect sizes, never the phrase 'the axis lies outside the coding subspace' (that reading came from the dead degenerate null) and never the in-sample contrast alone.",
})
CURRENT_OVERRIDES.setdefault("occupied_subspace_rank_selection_repair", {}).update({
    "disposition": "supporting_diagnostic",
    "construct": "diagnostic of whether the cross-validated occupied-subspace rank selector actually selects a rank, or saturates at the ambient dimension",
    "status": "current_exploratory",
    "caveat": "the delivered rank selector chose the full ambient rank (best_k == max_k) in 100 of 117 delivered cells (9/11 single-item macaque lPFC sessions, 91/106 multi-object macaque cells), so cells reported 'at full ambient rank' were an artifact of the estimator, not a measurement of occupied dimensionality. The corrected selector's own pooled comparison is not computable for either corpus (it saturates its own smaller ceiling the same way), so the repair is inconclusive, not a reversal: neither the delivered occupied-space branch nor its opposite is supported by this artifact for either corpus.",
})
CURRENT_OVERRIDES.setdefault("within_session_permutation_control", {}).update({
    "claim_id": "current::within_session_permutation_control_survivors",
    "disposition": "evidence",
    "construct": "within-session permutation control (10000 draws, whole-session pooling) applied to four cells previously voided or left untested by the between-session bias-only control",
    "status": "current_exploratory",
    "caveat": "four independent cells survive a within-session permutation control that cannot commit the between-session control's category error; these are new findings under this control, not restorations of the claims the between-session control voided. Every permutation p-value sits at the attainable floor for 10000 draws (9.999e-05), a resolution limit, not an effect size, and must always be quoted beside its effect size: swap_vs_imprecision_identity_pooled_within_item_count_level (n=41) r=0.0198, 95% CI [0.0105, 0.0289]; swap_vs_imprecision_identity_item_count_level_2_only r=0.0311; rate_free_state_geometry_behavior_link_headline (n=11) r=-0.0974, 95% CI [-0.1405, -0.0552]; pre_cue_state_leans_away_from_reported_item (n=22) departure -0.0514, minimum detectable difference at 80% power 0.0464.",
})
CURRENT_OVERRIDES.setdefault("rank_free_component_identity", {}).update({
    "claim_id": "current::rank_free_component_identity",
    "disposition": "evidence",
    "construct": "identity of the behaviour-linked population component under a cross-fitted subspace-alignment test that refits every candidate label subspace under its own permutation null, without assuming the component's occupied rank",
    "eligibility_rule": "candidate-by-corpus cells reaching at least 4 independent units (patients, participants, or animals) after equal-unit pooling; 12 of 26 declared candidate-by-corpus cells currently reach this floor",
    "independent_unit": "patient, participant, or animal, weighted equally after within-unit session averaging; both macaque corpora (lateral PFC n=3, multi-object n=2 independent units) fall below the 4-unit floor for every candidate",
    "estimand": "equal-independent-unit mean held-out squared projection of an evaluation-fold residual axis onto a training-fold label subspace, with full candidate-subspace refit under a 1000-permutation null, 2 folds",
    "preprocessing_version": "cross-fitted per corpus and candidate; Benjamini-Hochberg correction applied over all computed candidate-by-corpus cells",
    "inferential_method": "cross-fitted squared subspace alignment against a label-permutation null that refits every training-fold label subspace per permutation, Benjamini-Hochberg corrected over all computed candidate-by-corpus cells",
    "correction_family": "Benjamini-Hochberg across all computed candidate-by-corpus cells; incomplete (12 of 26 cells; all_supported_cells_tested is false)",
    "status": "current_exploratory",
    "gate": "G1",
    "caveat": "both macaque corpora that carry this project's original dissociation (lateral PFC, 3 independent units; multi-object, 2 independent units) are entirely untested by this estimand, below the declared 4-unit floor for every candidate -- nothing here shows the identity generalising to either macaque corpus. The gain/total-spike-count candidate clears its refit null in all four corpora where it is computable: mouse ALM +0.1943 (95% CI [0.1151, 0.2850], p=0.000999, q=0.003996, n=5); DANDI 000469 +0.0527 (CI [0.0113, 0.1022], p=0.004995, q=0.014985, n=18); DANDI 001187 +0.0752 (CI [0.0266, 0.1256], p=0.000999, q=0.003996, n=26); DANDI 000574 +0.0752 (CI [0.0376, 0.1209], p=0.000999, q=0.003996, n=8). The memorandum-content candidate is computable only in DANDI 000469, where it does not clear its own refit null (+0.0114, CI [-0.0306, 0.0560], p=0.3277, q=0.4304); that interval covers the gain candidate's effect in the same corpus (+0.0527), so this design cannot exclude a gain-sized effect on content -- a power statement about this one design, never a comparison of which candidate the axis favours, and never read as showing the axis is not content. The previous-trial-content candidate does not clear its own refit null in either corpus where computable: mouse ALM +0.0102 (CI [-0.0337, 0.0719], p=0.3586, q=0.4304); DANDI 000469 +0.0100 (CI [-0.0269, 0.0476], p=0.3576, q=0.4304). Mouse ALM's task-specific upcoming-response candidate, computed only in that corpus in place of memorandum content, clears its refit null (+0.0967, CI [-0.0271, 0.2337], p=0.006993, q=0.016783); the bootstrap interval crosses zero even though the permutation test clears FDR. The within-trial-time candidate is uncomputable in every corpus because it varies along the within-trial bin axis rather than across trials, so this estimand, which collapses that axis, defers it to a design carrying a within-trial blocked time-label null; the zero session and unit counts recorded against it follow from that deferral and are not a count of available recordings. It is a placeholder pending a separate delivery. In DANDI 000574, low-frequency field-potential power clears its refit null at intracranial contacts (+0.0455, CI [0.0199, 0.0670], p=0.01399, q=0.02797) but not at scalp contacts (+0.0189, CI [-0.0069, 0.0456], p=0.1548, q=0.2654); aperiodic-slope power does not clear its refit null at intracranial contacts (+0.0028, CI [-0.0519, 0.0444], p=0.4316, q=0.4708), and at scalp contacts the alignment sits reliably below its own refit null (-0.0427, CI [-0.0866, -0.0045], both bounds negative, p=0.9940, q=0.9940) -- delivered as a fold-structure and rotation diagnostic in results/alignment_below_null_diagnostic.json, leaving the anomaly undetermined rather than an unexamined open item, and never a finding in either direction.",
    "model_prediction": "if the behaviour-linked component's identity is genuine, its evaluation-fold residual should align above a refit-under-permutation null in each corpus and candidate where the test is computable.",
    "prediction_match_status": "matched for the gain/total-spike-count candidate in all four corpora where computable; not matched for memorandum-content or previous-trial-content in the corpora where each is computable; not yet tested for within-trial time or in either macaque corpus (both below the four-independent-unit floor); the scalp aperiodic-slope diagnostic anomaly is delivered and undetermined, not unresolved, per results/alignment_below_null_diagnostic.json.",
})
CURRENT_OVERRIDES.setdefault("within_trial_time_component_identity", {}).update({
    "claim_id": "current::within_trial_time_component_identity",
    "disposition": "evidence",
    "construct": "identity of the behaviour-linked population component against the within-trial elapsed-time candidate, using the same cross-fitted subspace-alignment estimand as the candidate-identity test but with a (trial, bin) unit of observation and a null that circular-shifts each trial's own time labels within that trial",
    "eligibility_rule": "candidate reaching at least 4 independent units (patients, participants, or animals) after equal-unit pooling; 4 of 6 declared corpora currently reach this floor",
    "independent_unit": "patient, participant, or animal, weighted equally after within-unit session averaging; both macaque corpora (lateral PFC n=3, multi-object n=2 independent units) fall below the 4-unit floor",
    "estimand": "equal-independent-unit mean held-out squared projection of an evaluation-fold residual axis onto a training-fold label subspace, fit on unit-normalised (trial, bin) activity vectors, with full candidate-subspace refit under a within-trial time-label-shift null, 2 folds",
    "preprocessing_version": "cross-fitted per corpus; Benjamini-Hochberg correction applied over all four computed corpus cells",
    "inferential_method": "cross-fitted squared subspace alignment against a within-trial circular-shift time-label null that refits every training-fold label subspace per permutation draw, Benjamini-Hochberg corrected over the four computed corpus cells",
    "correction_family": "Benjamini-Hochberg across the four computed corpus cells for this one candidate",
    "status": "current_exploratory",
    "gate": "G1",
    "caveat": "both macaque corpora that carry this project's original dissociation (lateral PFC, 3 independent units; multi-object, 2 independent units) are entirely untested by this estimand, below the declared 4-unit floor -- this caveat travels with this result exactly as it does with the gain/total-spike-count result in results/rank_free_component_identity.json. The candidate clears its own refit null in zero of the four computable corpora: mouse ALM +0.0318 (95% CI [-0.0177, 0.0982], p=0.0729, q=0.1459, n=5 independent units); DANDI 000469 -0.0222 (CI [-0.0471, 0.0059], p=0.8551, q=0.8551, n=18); DANDI 001187 +0.0167 (CI [-0.0285, 0.0679], p=0.1978, q=0.2637, n=26); DANDI 000574 +0.0419 (CI [-0.0048, 0.0867], p=0.0230, q=0.0919, n=8 -- the raw p clears 0.05 but the FDR q does not, and both figures are always reported together, never the raw p alone as a positive). Null calibration over 200 synthetic replicates, under a generative model with real within-trial autocorrelation unrelated to elapsed time, gives a false-positive rate of 0.055 at nominal 0.05; a plain permutation of the time label across samples would have destroyed that autocorrelation and manufactured significance instead. In each of the four computable corpora, the design's own minimum detectable difference at 80% power sits below the gain/total-spike-count effect results/rank_free_component_identity.json measured in that same corpus: mouse ALM (mdd 0.0848 vs gain +0.1943), DANDI 000469 (mdd 0.0385 vs gain +0.0527), DANDI 001187 (mdd 0.0683 vs gain +0.0752), DANDI 000574 (mdd 0.0658 vs gain +0.0752). Had the residual axis lain in the within-trial time subspace as strongly as it lies in the gain subspace, this design would have detected that at 80% power in every one of the four, and it did not -- a power statement about this one design, never a comparison of which candidate the axis favours. This licenses that the residual axis is not elapsed time in disguise at the magnitude at which it is gain; it does not exclude a smaller time alignment. This is a different estimand in different corpora from the macaque-only rotation-null absolute-cosine drift control in results/deviation_axis_identity_controls.json; neither result replicates or confirms the other.",
    "model_prediction": "if the behaviour-linked component's identity is within-trial elapsed time, its evaluation-fold residual should align above a within-trial-shift-refit null in each corpus where the test is computable.",
    "prediction_match_status": "not matched in any of the four computable corpora; not yet tested in either macaque corpus (both below the four-independent-unit floor).",
})
CURRENT_OVERRIDES.setdefault("alignment_below_null_diagnostic", {}).update({
    "claim_id": "current::alignment_below_null_diagnostic",
    "disposition": "supporting_diagnostic",
    "construct": "whether DANDI 000574's below-null scalp aperiodic-slope alignment in results/rank_free_component_identity.json reflects within-session nonstationarity or a property of the cross-fitted alignment estimator, via contiguous block folds versus interleaved folds and an early-to-late candidate-subspace overlap measurement",
    "eligibility_rule": "every candidate-by-corpus cell in results/rank_free_component_identity.json that reached the four-independent-unit floor there; 12 of 26 declared cells",
    "independent_unit": "patient, participant, or animal, weighted equally after within-unit session averaging, identical to results/rank_free_component_identity.json's own pooling; both macaque corpora (lateral PFC n=3, multi-object n=2 independent units) fall below the 4-unit floor for every candidate and are absent here",
    "estimand": "paired block-minus-interleaved difference in cross-fitted mean alignment-above-null, and paired early-versus-late candidate-subspace overlap, both with cluster bootstrap intervals over independent units",
    "preprocessing_version": "block-fold cells reproduced by calling results/rank_free_component_identity.json's own per-cell and pooling functions, unmodified, on identical per-session data and seed; interleaved folds assign samples by trial-index alternation instead of contiguous block, holding every other estimator element fixed",
    "inferential_method": "cluster bootstrap over independent units for the paired block-minus-interleaved difference and the early-late overlap contrast; minimum detectable difference at 80% power reported alongside every interval-crossing-zero result",
    "correction_family": "no multiple-comparison correction; a diagnostic contrast reported per cell against its own detection floor, not a hypothesis-test family",
    "status": "current_exploratory",
    "gate": "G1",
    "caveat": "the block-fold reproduction gate passed for every one of the 12 tested cells (pooled mean-alignment-above-null and p-value match the delivered artifact within numerical tolerance). For the anomalous cell (DANDI 000574, scalp aperiodic-slope power at scalp contacts), block-fold alignment-above-null is -0.0427 and interleaved-fold alignment-above-null is -0.0001, a block-minus-interleaved difference of -0.0427 (cluster bootstrap CI [-0.1093, 0.0076]) against this cell's own minimum detectable difference of 0.0847: the difference moves in the direction a nonstationarity account predicts but does not clear its own detection floor. The same direction of fold-structure effect, with its own interval excluding zero, is present in 6 of the other 11 tested cells (dandi_000469_human|gain_total_spike_count, dandi_000574_human|field_potential_low_frequency_band_power_ieeg, dandi_000574_human|gain_total_spike_count, dandi_001187_human|gain_total_spike_count, inagaki_alm5_mouse_ALM|gain_total_spike_count, inagaki_alm5_mouse_ALM|upcoming_response), spanning corpora and candidates unconnected to the anomalous one -- this run does not separate a general sensitivity of the alignment estimator to fold structure from general within-session nonstationarity affecting every candidate; both would produce the same pattern. The anomalous cell is delivered as undetermined and remains a diagnostic, never a finding, in either direction. Interleaved folds give higher alignment-above-null than contiguous block folds in every corpus where the gain/total-spike-count candidate is computable, each interval excluding zero: mouse ALM block +0.1943 vs interleaved +0.3016 (difference -0.1072, CI [-0.1757, -0.0338]); DANDI 000469 +0.0527 vs +0.1004 (-0.0477, CI [-0.0909, -0.0039]); DANDI 001187 +0.0752 vs +0.1723 (-0.0971, CI [-0.1401, -0.0550]); DANDI 000574 +0.0752 vs +0.1527 (-0.0775, CI [-0.1118, -0.0335]). Because results/rank_free_component_identity.json's own delivered result used the conservative, temporally disjoint block-fold structure, it survives this contrast unweakened; the interleaved values here are never cited as effect sizes for the axis. Early-to-late candidate-subspace overlap is low across all twelve tested cells (about 0.07 to 0.60, several cells clustering near 0.42-0.46), consistent with either genuine within-session rotation or estimation noise from halving each session's data; this run carries no within-half reference at matched sample size, so it does not separate the two and does not show that the candidate subspaces rotate. A separate control comparing subspace overlap against a matched-noise reference is in progress and is not anticipated here.",
    "model_prediction": "if the anomalous cell's below-null alignment reflects within-session nonstationarity, interleaving folds should move its alignment-above-null toward or above zero relative to the block-fold cell, with a positive block-minus-interleaved difference whose interval excludes zero; if it reflects a property of the estimator, the block-minus-interleaved difference should be near zero.",
    "prediction_match_status": "undetermined for the anomalous cell -- the difference moves in the nonstationarity direction but does not clear its own detection floor, and the same direction of effect recurs broadly across unrelated cells.",
})
CURRENT_OVERRIDES.setdefault("human_behavioral_value", {}).update({
    "claim_id": "current::human_behavioral_value_incremental_prediction",
    "disposition": "evidence",
    "construct": "incremental participant-level held-out predictive value of rate-free neural state deviation for trial correctness, beyond a combined task/load, spike-rate, and two-trial-history baseline",
    "eligibility_rule": "7390 admitted trials from 52 participants across three human maintenance corpora (DANDI 000469, 001187, 000574); leave-one-participant-out logistic-regression folds",
    "independent_unit": "participant (52); trials are repeated measures within participant and folds are strictly participant-held-out",
    "estimand": "paired participant-mean difference in held-out log loss between the combined task/nuisance baseline and the same baseline plus rate-free state deviation, with a 95% participant bootstrap interval",
    "preprocessing_version": "standardized logistic-regression features; participant-held-out folds; 2000-resample bootstrap",
    "inferential_method": "leave-one-participant-out cross-validated logistic regression; paired participant-level log-loss contrast against a 95% bootstrap interval",
    "correction_family": "single pre-specified comparison; no multiple-comparison family",
    "status": "replication_or_informative_null",
    "gate": "G1",
    "caveat": "combined task/load, spike-rate, and two-trial-history baseline held-out log loss 0.26602 (Brier 0.07190); adding rate-free state deviation gives log loss 0.26635 (Brier 0.07195). Participant-level baseline-minus-combined-state log loss is -0.000343 (95% bootstrap CI [-0.000991, 0.000138], n=52 participants, 7390 trials) -- the interval includes zero, so this is a bounded negative, not evidence of exactly zero incremental value. This is a predictive comparison under participant holdout, not a causal or equivalence test.",
    "model_prediction": "if the rate-free state deviation carries trial-correctness information beyond the combined task and nuisance baseline, adding it should detectably lower held-out log loss at the participant level (bootstrap interval excluding zero).",
    "prediction_match_status": "not matched: the interval crosses zero, so no incremental predictive value is detected at this sample size.",
})
CURRENT_OVERRIDES.setdefault("dandi_000004_recognition_generalization", {}).update({
    "claim_id": "current::dandi_000004_recognition_generalization",
    "disposition": "evidence",
    "construct": "participant-level joint partial correlation between population state and new/old recognition accuracy, controlling total spike count, trial index, and old/new ground truth, in a recognition task rather than the working-memory maintenance task used elsewhere in this project",
    "eligibility_rule": "DANDI 000004 sessions with at least 40 valid recognition trials and, per region, at least 8 units surviving rate quality control; hippocampus and amygdala scored as two separate regional families, never pooled",
    "independent_unit": "participant, each contributing one trial-count-weighted mean across their own sessions; hippocampus 22 participants, amygdala 36 participants",
    "estimand": "region-specific participant-mean joint partial correlation in the first 450 ms after stimulus offset, ending before question onset, with BH correction across the two regions, plus the unmatched paired hippocampus-minus-amygdala difference in the 14 participants contributing both",
    "preprocessing_version": "response mapping 31-33 new / 34-36 old per the source release; frozen pre-question window",
    "inferential_method": "participant-clustered joint partial correlation with bootstrap confidence intervals and a minimum-detectable-difference-at-80%-power calculation against this project's 0.14 r-unit reference, per region",
    "correction_family": "Benjamini-Hochberg across the two region-specific primary tests; the paired regional difference is a separate, uncorrected contrast",
    "status": "replication_or_informative_null",
    "gate": "G1",
    "caveat": "hippocampus: r=+0.0008 (95% CI [-0.0336,0.0357], q=0.9634, n=22 participants), own minimum detectable difference at 80% power 0.0508, well below the 0.14 reference -- a powered null. Amygdala: r=-0.0148 (95% CI [-0.0415,0.0099], q=0.5577, n=36 participants), own minimum detectable difference 0.0375, also well below 0.14 -- a powered null. The two regions are reported separately and never combined into one regional estimate. Paired hippocampus-minus-amygdala difference (n=14 participants contributing both): +0.0092 (95% CI [-0.0446,0.0610], p=0.7356) -- non-significant, and the artifact reports no minimum detectable difference for this paired contrast, so no power statement accompanies it. Of 86 NWB files seen, 84 sessions computed and 2 refused at the session level for fewer than 40 valid recognition trials; at the region level, 31 amygdala and 52 hippocampus sessions were separately refused for fewer than 8 units after rate quality control, each region also carrying the same 2 session-level refusals (53 amygdala and 32 hippocampus sessions computed before collapsing to participants). The computed subset is selected on unit count, which bounds what these estimates describe. As at every other human recording tier in this project, the underlying component is present; what is absent here is its link to new/old recognition accuracy above the reference bound, not the component itself. This is a task-generalization test within this project's own evidence (recognition, not working-memory maintenance) and not independent-laboratory replication; its participants, sessions, and estimates are never pooled with any maintenance corpus.",
    "model_prediction": "if the population-state association with accuracy measured in working-memory maintenance tasks generalizes to a recognition task, the region-specific joint partial correlation should clear the project's 0.14 r-unit reference in at least one region.",
    "prediction_match_status": "not matched in either region; both are powered nulls against the 0.14 reference, and the paired regional difference is non-significant without its own power statement.",
})
CURRENT_OVERRIDES.setdefault("subspace_rotation_versus_noise", {}).update({
    "claim_id": "artifact::subspace_rotation_versus_noise",
    "disposition": "supporting_diagnostic",
    "construct": "whether the low early-to-late candidate-subspace overlap in results/alignment_below_null_diagnostic.json reflects the candidate subspace rotating within a session, or estimation noise at the sample size each half's fit uses",
    "eligibility_rule": "every (corpus, candidate) cell results/alignment_below_null_diagnostic.json computed an early-to-late subspace overlap for -- the same twelve cells",
    "independent_unit": "patient, participant, or animal, weighted equally after within-unit session averaging, identical to results/alignment_below_null_diagnostic.json's own pooling; both macaque corpora fall below the four-independent-unit floor and are absent here",
    "estimand": "per-session across-half subspace overlap minus a within-half bootstrap noise reference, pooled by cluster bootstrap over independent units",
    "preprocessing_version": "across-half overlap reproduced by calling results/alignment_below_null_diagnostic.json's own rotation function, unmodified, on identical per-session activity and target arrays",
    "inferential_method": "cluster bootstrap interval over independent units for the across-minus-within contrast; minimum detectable difference at 80% power reported alongside every interval-crossing-zero result",
    "correction_family": "no multiple-comparison correction; a diagnostic contrast reported per cell against its own detection floor, not a hypothesis-test family",
    "status": "invalidated_do_not_interpret",
    "gate": "G1",
    "caveat": "the within-half bootstrap reference draws two resamples with replacement from a single, already-contiguous half, each matched in size to that half; because both resamples share a large fraction of the same trials by construction, this reference is more alike than two fits on genuinely disjoint data, which biases it upward and the resulting across-minus-within contrast downward -- toward a rotation reading rather than away from one. Sample size was matched between the across-half fit and this reference; independence was not, and independence is what governs how much two fits of the same subspace agree. This contrast, and any per-cell rotation branch computed from it, is not interpretable and must not be quoted as a result anywhere in this project. The reproduction gate underneath it is sound: recomputing the across-half overlap by calling results/alignment_below_null_diagnostic.json's own function, unmodified, on identical per-session data passed exactly for all twelve tested cells; only the within-half noise reference and the contrast built on it are affected. A reference matched on both sample size and disjointness is reported separately in results/subspace_rotation_time_separation.json.",
    "model_prediction": "if the low early-to-late subspace overlap reflects estimation noise rather than genuine within-session rotation, the across-half overlap should sit at parity with a reference computed at the same matched sample size.",
    "prediction_match_status": "not adjudicable: the reference built to test this prediction is itself biased toward the rotation conclusion by construction, so no verdict is drawn from this artifact in either direction.",
})
CURRENT_OVERRIDES.setdefault("subspace_rotation_time_separation", {}).update({
    "claim_id": "current::subspace_rotation_time_separation",
    "disposition": "supporting_diagnostic",
    "construct": "whether the low early-to-late candidate-subspace overlap in results/alignment_below_null_diagnostic.json reflects the candidate subspace rotating within a session, or is explained by the temporal separation between an early and a late fit at matched sample size and disjointness",
    "eligibility_rule": "every (corpus, candidate) cell results/alignment_below_null_diagnostic.json computed an early-to-late subspace overlap for -- the same twelve cells",
    "independent_unit": "patient, participant, or animal, weighted equally after within-unit session averaging, identical to results/alignment_below_null_diagnostic.json's own pooling; both macaque corpora fall below the four-independent-unit floor and are absent here",
    "estimand": "per-session across-half subspace overlap minus an interleaved-chunk reference (primary chunk length 4 trials, swept over 2/4/8/16), and separately minus a deliberately conservative quarter-split reference, both pooled by cluster bootstrap over independent units",
    "preprocessing_version": "across-half overlap reproduced by calling results/alignment_below_null_diagnostic.json's own rotation function, unmodified, on identical per-session activity and target arrays; interleaved-chunk sets are disjoint, equal-sized, and each spans the session's full time range, matching the across-half comparison on both sample size and disjointness",
    "inferential_method": "cluster bootstrap interval over independent units for each contrast; minimum detectable difference at 80% power reported alongside every interval-crossing-zero result; chunk-length sweep stability reported per cell",
    "correction_family": "no multiple-comparison correction; a diagnostic contrast reported per cell against its own detection floor, not a hypothesis-test family",
    "status": "current_exploratory",
    "gate": "G1",
    "caveat": "this reference replaces the withdrawn within-half bootstrap reference in results/subspace_rotation_versus_noise.json (invalidated_do_not_interpret) with one matched on both sample size and disjointness, leaving temporal separation as the only remaining difference; the reproduction gate passed exactly for all twelve tested cells. Of twelve cells: zero show across-half overlap reliably below both the interleaved-chunk and quarter-split references (rotation established under both is achieved nowhere); four sit at parity with the interleaved-chunk reference and the rotation reading is withdrawn there -- dandi_000469_human|memorandum_content (0.2086 vs 0.2209, contrast -0.0039, CI [-0.0373, 0.0288], mdd 0.0488), dandi_000469_human|previous_trial_content (0.2034 vs 0.2016, contrast +0.0020, CI [-0.0133, 0.0221], mdd 0.0258), dandi_000574_human|field_potential_low_frequency_band_power_ieeg (0.1914 vs 0.1662, contrast +0.0252, CI [-0.0073, 0.0605], mdd 0.0491), dandi_000574_human|field_potential_aperiodic_slope_eeg (0.0802 vs 0.0851, contrast -0.0048, CI [-0.0429, 0.0308], mdd 0.0521); the remaining eight are undetermined, not null -- five underpowered against the 0.0897 meaningful-magnitude bar (inagaki_alm5_mouse_ALM|upcoming_response mdd 0.3328, inagaki_alm5_mouse_ALM|previous_trial_content mdd 0.3564, dandi_000469_human|gain_total_spike_count mdd 0.0933, dandi_000574_human|gain_total_spike_count mdd 0.1027, dandi_000574_human|field_potential_low_frequency_band_power_eeg mdd 0.1182), and three sit between the two references, clearing the matched interleaved-chunk reference but not the conservative quarter-split one, so some rotation in these cells is neither established nor excluded -- inagaki_alm5_mouse_ALM|gain_total_spike_count (0.4567 across-half against interleaved-chunk 0.8009, contrast -0.3442, CI [-0.3701, -0.3134]; quarter-split 0.4870, contrast -0.0303, CI [-0.1251, 0.0644]), dandi_001187_human|gain_total_spike_count (0.4178 against 0.5019, contrast -0.0841, CI [-0.1565, -0.0186]; quarter-split 0.2906, contrast +0.1272, CI [0.0489, 0.2037]), dandi_000574_human|field_potential_aperiodic_slope_ieeg (0.0711 against 0.1136, contrast -0.0425, CI [-0.0796, -0.0015]; quarter-split 0.0938, contrast -0.0228, CI [-0.0434, 0.0023]). The chunk-length sweep (2/4/8/16 trials) is stable in ten of twelve cells; two change verdict with chunk length and are flagged unstable rather than settled at a favourable length: dandi_000574_human|field_potential_low_frequency_band_power_eeg and dandi_000574_human|field_potential_aperiodic_slope_ieeg -- both already among the cells above whose primary-chunk-length reading is underpowered or between-references. This design does not support a claim that the candidate subspaces rotate within a session, and it does not support a claim that they do not.",
    "model_prediction": "if the candidate subspace genuinely rotates within a session, early-to-late overlap should sit reliably below both a matched-sample-size, disjoint interleaved-chunk reference and a deliberately conservative quarter-split reference.",
    "prediction_match_status": "not matched in any of the twelve tested cells (zero clear both references); withdrawn in four cells where overlap sits at parity with the matched reference; undetermined in the remaining eight, five for power and three because the two references disagree.",
})
CURRENT_OVERRIDES.setdefault("within_animal_component_identity", {}).update({
    "claim_id": "current::within_animal_component_identity",
    "disposition": "evidence",
    "construct": "within-animal identity of the behaviour-linked population component under the same cross-fitted subspace-alignment estimand as results/rank_free_component_identity.json, recomputed separately inside each macaque animal with session, not animal, as the resampling and clustering unit",
    "eligibility_rule": "animal (independent_unit) with at least 4 sessions carrying a computed per-session cell for a candidate; all 5 animals across the two macaque corpora reach this floor for gain_total_spike_count and memorandum_content",
    "independent_unit": "session within each animal; results are reported per animal, separately, and never pooled across animals into one number -- this never counts as clearing, relaxing, meeting, or substituting for the four-independent-unit floor in results/rank_free_component_identity.json, because both macaque corpora carry too few animals (lateral PFC 3, multi-object 2) to ever clear that floor with any amount of future recording",
    "estimand": "same cross-fitted, squared, held-out subspace-alignment estimand as results/rank_free_component_identity.json (2 folds, 1000-permutation label-subspace refit null), pooled per animal by whole-session cluster bootstrap instead of pooled per corpus by whole-animal cluster bootstrap",
    "preprocessing_version": "recomputed per animal, per candidate, on the same per-session fits; Benjamini-Hochberg correction applied across every computed primary cell in this artifact",
    "inferential_method": "cross-fitted squared subspace alignment against a label-permutation refit null, Benjamini-Hochberg corrected across every computed primary cell in this artifact; a non-clearing cell is converted to a bound by comparing its own minimum detectable difference at 80% power against that same animal's own gain_total_spike_count effect as an internal, same-scale reference",
    "correction_family": "Benjamini-Hochberg across every computed primary cell in this artifact (10 cells: 5 animals x 2 candidates)",
    "status": "current_exploratory",
    "gate": "G1",
    "caveat": "this is a within-animal estimate of the same cross-fitted squared subspace-alignment estimand as results/rank_free_component_identity.json, recomputed separately inside each of five macaque animals with session, not animal, as the resampling and clustering unit, because both macaque corpora carry too few animals (lateral PFC 3, multi-object 2) to ever clear the four-independent-unit floor that estimand requires -- no future recording from either laboratory can reach four animals, so that floor is not a sample-size problem this analysis or any new data can fix. Results are reported per animal, separately, and are never pooled across animals into one number; no cross-animal generalisation is claimed. Clearing this artifact's own decision rule never counts as clearing, relaxing, meeting, or substituting for the four-independent-unit floor in results/rank_free_component_identity.json; it is a different, weaker question asked in the same data, and neither result replicates, confirms, or voids the other in either direction. The spike-count gain axis clears its refit null in all five animals individually: panichello_2024_macaque_lPFC monkey_A (10 sessions) +0.0941 (95% CI [0.0396, 0.1496], p=0.000999, q=0.0025, minimum detectable difference at 80% power 0.0789); monkey_H (8 sessions) +0.1483 (CI [0.0769, 0.2189], p=0.000999, q=0.0025, mdd 0.1029); monkey_J (7 sessions) +0.0586 (CI [0.0103, 0.1041], p=0.001998, q=0.003996, mdd 0.0670); watters_2026_macaque_multi_object Elgar (22 sessions) +0.0527 (CI [0.0189, 0.0867], p=0.000999, q=0.0025, mdd 0.0480); Perle (19 sessions) +0.0975 (CI [0.0541, 0.1457], p=0.000999, q=0.0025, mdd 0.0670). The memorandum-content axis clears in none of the five animals and is a bounded negative in all five: an effect the size of that same animal's own gain effect would have been detected here and was not, while smaller effects are not excluded, in every case -- monkey_A (effect -0.0285, mdd 0.0591, against its own gain effect 0.0941); monkey_H (effect -0.0732, CI [-0.1257, -0.0213], mdd 0.0747, against 0.1483); monkey_J (effect -0.0605, CI [-0.0891, -0.0313], mdd 0.0411, against 0.0586); Elgar (effect +0.0016, mdd 0.0277, against 0.0527); Perle (effect +0.0077, mdd 0.0586, against 0.0975). monkey_H and monkey_J sit reliably below their own null (both bootstrap intervals entirely negative); this is a diagnostic only and is never reported as a finding. The one-dimensional content construction is why the bound is reachable in the multi-object corpus: a descriptive many-class diagnostic on native theta values, outside this rule's FDR family and carrying no clear/bound verdict, gives Elgar minimum detectable difference 0.0638 many-class versus 0.0277 one-dimensional, and Perle 0.1154 versus 0.0586 -- roughly half the detectable difference at identical trial counts. The Panichello corpus retained its class-based construction because its memorandum is genuinely categorical (eight discrete cued locations), not artificially discretised.",
    "model_prediction": "if the behaviour-linked component's identity is genuine at the resampling and clustering unit available within each animal, gain_total_spike_count should clear a session-level refit null separately in every animal, and memorandum_content should not exceed that same animal's own detectable bound if it is not the component's identity.",
    "prediction_match_status": "matched for gain_total_spike_count in all five animals (all clear); matched for memorandum_content as a bounded negative in all five animals (none clear, and the minimum detectable difference sits below that animal's own gain effect in every case).",
})
CURRENT_OVERRIDES.setdefault("readout_transfer_decay", {}).update({
    "claim_id": "current::readout_transfer_decay",
    "disposition": "evidence",
    "construct": "whether a read-out direction fit on one stretch of a session's trials still predicts the same quantity in a later stretch of the same session, and how held-out performance changes with the elapsed trial gap between the calibration stretch and the evaluation stretch",
    "eligibility_rule": "corpus/candidate cell reaching at least 4 independent units with at least 4 contiguous trial chunks of the primary chunk size (4 trials); both macaque corpora (lateral PFC 3, multi-object 2 independent units) do not reach this floor for gain_total_spike_count and are not_computable",
    "independent_unit": "patient, participant, or animal, weighted equally after within-unit session averaging, pooled by whole-independent-unit cluster bootstrap, identical to the pooling in results/alignment_below_null_diagnostic.json",
    "estimand": "per-session ordinary-least-squares slope of held-out squared-correlation performance against elapsed trial gap between a training chunk and a disjoint evaluation chunk, pooled equal-weight per session then per independent unit; the gap-zero intercept of the same regression is the matched no-separation reference",
    "preprocessing_version": "primary chunk size 4 trials, swept over 2/4/8/16; chunk size 2 excluded from bound consideration as a floating-point-zero degeneracy (any two points define a perfect correlation); read-out direction fit with regression_basis unchanged from results/rank_free_component_identity.json's own estimator",
    "inferential_method": "circular chunk-index shift null preserving chunk composition and within-session autocorrelation, pooled the same equal-session-then-equal-unit way as the observed slope; Benjamini-Hochberg corrected across all primary-chunk-size cells; a non-clearing cell is converted to a bound by comparing its own minimum detectable slope at 80% power against an internal reference decay rate (that corpus/candidate's own results/rank_free_component_identity.json gain-identity alignment-above-null, divided by the mean trial count of that cell's own contributing sessions)",
    "correction_family": "Benjamini-Hochberg across the ten primary-chunk-size cells this module computes",
    "status": "current_exploratory",
    "gate": "G1",
    "caveat": "of 10 attempted cells (gain_total_spike_count in inagaki_alm5_mouse_ALM, dandi_000469_human, dandi_001187_human, dandi_000574_human, panichello_2024_macaque_lPFC, and watters_2026_macaque_multi_object, plus 4 field-potential candidates in dandi_000574_human), 0 clear the pre-declared slope decision rule (BH-FDR-corrected permutation p and a cluster-bootstrap interval excluding zero, both reported together) and 0 are bounded negatives -- this is an answered, empty leg, not a pending one. Per-cell verdicts, quoted from the artifact: inagaki_alm5_mouse_ALM|gain_total_spike_count inconclusive_sweep_unstable_bound (the bounded-negative determination flips across chunk sizes 4, 8, and 16 trials, so no bound is reported); dandi_000469_human|gain_total_spike_count inconclusive_underpowered_relative_to_reference; dandi_001187_human|gain_total_spike_count inconclusive_interval_excludes_zero_not_fdr_significant (cluster-bootstrap interval [3.13e-06, 0.000331] excludes zero on the positive side while its BH-FDR q=0.1838 does not clear alpha=0.05 -- the pre-declared rule requires both, so this cell clears neither and no bound is drawn); dandi_000574_human|field_potential_aperiodic_slope_eeg and dandi_000574_human|field_potential_aperiodic_slope_ieeg both inconclusive_for_want_of_a_reference (that corpus/candidate's own gain-identity alignment-above-null is not reliably above its own null, so no internal same-scale reference exists to bound against); dandi_000574_human|field_potential_low_frequency_band_power_eeg also inconclusive_for_want_of_a_reference; dandi_000574_human|field_potential_low_frequency_band_power_ieeg and dandi_000574_human|gain_total_spike_count both inconclusive_underpowered_relative_to_reference; panichello_2024_macaque_lPFC|gain_total_spike_count and watters_2026_macaque_multi_object|gain_total_spike_count both not_computable (fewer than 4 independent units carry a computed cell at chunk size 4) -- infeasibility, not a null. The synthetic null calibration check over 2000 replicates measured a false-positive rate of 0.07 (exact binomial 95% CI [0.0592, 0.0821]), which excludes the nominal 0.05, so the test is NOT calibrated and is anti-conservative (liberal). This makes the zero-clearing outcome reported for every cell in this module more secure -- a liberal test that still rejected nothing is stronger evidence of no detectable slope -- but any cell that had cleared under this null would need its permutation p-value discounted before being trusted. This module never claims to replicate, confirm, or be confirmed by results/subspace_rotation_time_separation.json; they are different estimands (predictive transfer of a fitted read-out versus subspace overlap between two independent fits) computed on overlapping but not identical data.",
    "model_prediction": "if a read-out direction fit on one stretch of a session's trials carries information that decays with elapsed trial gap to a later stretch, held-out performance should decline with gap at a rate whose cluster-bootstrap interval excludes zero and whose BH-FDR q survives alpha=0.05.",
    "prediction_match_status": "not matched in any of the 10 tested cells: 0 clear, 0 are bounded negatives, 8 are inconclusive against their own internal reference or evidence-disagreement rule, and 2 did not reach the four-independent-unit floor.",
})

# results/stimulation_response_gate_and_panel.json carries two independently gated claims (a
# human arm and a macaque arm, each with its own admission set, clustering unit, and specificity
# check) in one JSON file, so this stem registers a list of override dicts rather than one --
# see the list-handling branch in row_for(). The ledger's "status" field is a closed enum
# (VALID_STATUSES in src/provenance.py); it does not hold a descriptive branch-verdict string, so
# both rows carry that verdict in "prediction_match_status" (free text, matching how every other
# multi-word branch verdict already in this ledger is recorded) and use "current_exploratory" for
# "status". Likewise "gate" is a closed G0-G4 enum, not a sentence -- both rows use "G3" per this
# module's own gate_for() convention for a stimulation-arm stem, and the reproduction-gate
# narrative itself is carried in full in the manuscript paragraph rather than in this field.
CURRENT_OVERRIDES["stimulation_response_gate_and_panel"] = [
    {
        "claim_id": "artifact::stimulation_response_gate_and_panel::human_post_stimulation_window_deviation",
        "result_id": "result::stimulation_response_gate_and_panel::human_post_stimulation_window_deviation",
        "construct": "stimulation-induced displacement of the rate-free angular-deviation biomarker",
        "eligibility_rule": "ram_ds005489_openloop sessions/trials admitted by results/stimulation_design_census.json's pre- and post-stimulation-window certification; excluding_stimulated_shank channel condition; open-loop arm only, ds005557 classifier-triggered arm excluded by the census",
        "estimand": "post-stimulation-window (1.6s) rate-free deviation, treated minus randomized contemporaneous control, normalised by control-trial SD",
        "dataset_view": "ram_ds005489_openloop (human intracranial, episodic encoding, open-loop arm only -- ds005557 classifier-triggered arm excluded by census)",
        "independent_unit": "participant (n=33); 73 sessions clustered by participant via whole-participant sign-flip permutation + percentile bootstrap",
        "preprocessing_version": "src/stimulation_response_estimator.py rate-free angular-deviation biomarker over census-measured pre/post windows; the upstream human displacement number this estimator is built on was reproduced exactly from its producer script before this window-split computation ran",
        "inferential_method": "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py): paired sign-flip test + bootstrap CI on participant-collapsed means",
        "correction_family": "single frozen per-arm pre/post specificity check against randomized contemporaneous controls; no correction applied across the two windows or across arms",
        "gate": "G3",
        "status": "current_exploratory",
        "manuscript_location": "PAPER_REPORT.tex, stimulation section (Section~\\ref{sec:corpus-capability}, paragraph following the design-census paragraph)",
        "caveat": (
            "mean 0.0941 (p=0.0380, CI [0.0169,0.1845]) in the post window, but the PRE-stimulation window is "
            "also significant (mean 0.0673, p=0.0453) at comparable magnitude, and that window closes before "
            "this trial's own stimulation begins -- the null-before/effect-after specificity check a per-trial "
            "causal effect predicts is failed. A neighbouring-train account of the pre-window effect is refuted "
            "by direct measurement of the release's own event tables: across 19,624 words with a preceding "
            "train the distance to that train's offset is strictly positive (median 7.975s, minimum 0.369s "
            "against a 0.3s pre-stimulation window), and every session contributes zero contaminated words "
            "(results/train_overlap_decontamination.json); the re-estimate reported there is not independent "
            "corroboration, since with no word excluded the subset is the full sample. A second reading is "
            "also refuted, and it removes the contrast rather than explaining it: direct measurement of the "
            "release's own event tables (results/within_block_stimulation_position.json) shows the delivered "
            "pulse train begins before the word it accompanies, by a fixed trigger lead of median 0.211s, so "
            "every stimulated word without exception has its train already running at word onset -- the "
            "interval scored as pre-stimulation is 70% covered by active stimulation for the first word of "
            "each stimulated pair and wholly covered for the second. No word in this corpus has a "
            "stimulation-free pre-window, so the pre/post comparison was never a specificity test: post minus "
            "pre is 0.0268 control-trial SD overall (CI [-0.035,0.095], p=0.451) and 0.0041 in first-in-pair "
            "words alone (CI [-0.063,0.080], p=0.915), both comfortably covering zero. Do not attribute this "
            "displacement to the timing of stimulation delivery, and do not read the post minus pre gap as a "
            "specificity signal. Both "
            "cells also sit below their own 80%-power minimum detectable difference (post 0.0941 vs 0.1210; "
            "pre 0.0673 vs 0.0906), so both point estimates are upward-biased conditional on reaching "
            "significance. Both are roughly one eleventh of the 1.0 control-trial SD reference this project "
            "uses on this scale, which is an internal scale marker and not a clinical or biological "
            "threshold. Never present this number as a per-trial causal displacement, and never quote it "
            "without both halves: statistically detectable, and far below the project's own reference "
            "magnitude."
        ),
        "model_prediction": (
            "A per-trial causal stimulation effect predicts a null pre-stimulation-window displacement "
            "(before that trial's own stimulation can act) paired with a significant post-stimulation-window "
            "displacement, with both windows resolvable above their own 80%-power minimum detectable difference."
        ),
        "prediction_match_status": "positive_but_not_specific_to_stimulation_timing_and_below_its_own_detection_floor",
    },
    {
        "claim_id": "artifact::stimulation_response_gate_and_panel::macaque_post_stimulation_window_deviation",
        "result_id": "result::stimulation_response_gate_and_panel::macaque_post_stimulation_window_deviation",
        "construct": "stimulation-induced displacement of the rate-free angular-deviation biomarker",
        "eligibility_rule": "macaque_pfc_microstimulation correct-trial file only, full_channel_set (no per-trial stimulated-channel exclusion), sessions/trials admitted by the same design census's pre/post-window certification",
        "estimand": "post-stimulation-window (0.7s) rate-free deviation, treated minus randomized contemporaneous control, normalised by control-trial SD",
        "dataset_view": "macaque_pfc_microstimulation (dlPFC, delay-period microstimulation, correct-trial file only, 11 sessions / 2 animals)",
        "independent_unit": "animal (n=2); 11 sessions clustered by animal",
        "preprocessing_version": "src/stimulation_response_estimator.py rate-free angular-deviation biomarker over census-measured pre/post windows; first run of this window-split macaque computation",
        "inferential_method": "cluster_bootstrap_pooled_effect, same primitive as the human-arm row above",
        "correction_family": "single frozen per-arm pre/post specificity check against randomized contemporaneous controls; the two-animal sign-flip lattice floors the attainable two-sided p-value at 0.493, so no multiplicity correction is meaningful at this n",
        "gate": "G3",
        "status": "current_exploratory",
        "manuscript_location": "PAPER_REPORT.tex, stimulation section (Section~\\ref{sec:corpus-capability}, paragraph following the design-census paragraph)",
        "caveat": (
            "post-window mean 2.13 control-trial SD (CI [0.95,3.30]), positive in 11/11 sessions and in both "
            "animals, with the smallest session value 0.949 more than four times the largest pre-window value "
            "0.217. The pre-stimulation window is a powered null on the same scale: mean 0.0207, CI "
            "[-0.0072,0.0485], mdd 0.0779 against the named 1.0 control-SD reference -- cite the mdd, never the "
            "p-value, as the evidence for that null. At n_animals=2 the sign-flip null admits four patterns and "
            "the attainable two-sided p-value floors at 0.493, so no significance statement about either window "
            "exists; p=0.493 is not evidence against the effect and p=1.0 is not evidence for the null. The "
            "post-window estimate sits below its own 80%-power minimum detectable difference (2.13 vs 3.29) and "
            "is upward-biased for that reason. No per-trial stimulated-channel exclusion is applied "
            "(full_channel_set only, unlike the human arm's excluding_stimulated_shank), so an electrical-artefact "
            "contribution from channels near the stimulation site is not ruled out; trials are read from the "
            "correct-trial file only. Do not compare this magnitude against the human arm's as a ranking: the "
            "two arms differ in species, epoch, recording modality and delivered current simultaneously."
        ),
        "model_prediction": (
            "A per-trial causal stimulation effect predicts a null pre-stimulation-window displacement paired "
            "with a significant post-stimulation-window displacement; with only 2 animals as the clustering "
            "unit, significance is unreachable by construction regardless of effect size, so the specificity "
            "pattern must be read from effect size and session-level consistency rather than from a p-value."
        ),
        "prediction_match_status": "specificity_pattern_holds_on_effect_size_and_session_consistency_significance_unreachable",
    },
]

# results/macaque_maintenance_behaviour_link.json admits error trials (for the first time in this
# corpus) and asks whether the rate-free angular-deviation biomarker separates correct from error
# trials at maintenance, cut by pre/post-stimulation window and control/stimulated arm. This is a
# positive-signed lean, not a confirmatory result and not a powered null (significance is
# structurally unreachable at 2 animal clusters), so status is "current_exploratory" and the free-text
# verdict the ticket for this fold-in specifies -- "suggestive, confounded, not closed" -- is carried
# in prediction_match_status/caveat rather than as a VALID_STATUSES member. The caveat on every row
# below carries all three reasons this is not closed: below-own-mdd, unreachable significance at 2
# clusters, and the uncontrolled/untestable time-on-task confound. Field values originated by this
# fold-in (not present verbatim in the artifact): claim_id, result_id, construct, eligibility_rule,
# estimand, inferential_method, correction_family, manuscript_location, model_prediction and
# prediction_match_status. All numeric values (means, CIs, mdd, nuisance-control values, Wa-only
# sensitivity values, validity-probe counts) are copied from the artifact's own computed fields.
CURRENT_OVERRIDES["macaque_maintenance_behaviour_link"] = [
    {
        "claim_id": "artifact::macaque_maintenance_behaviour_link::pre_stimulation_window_control_condition",
        "result_id": "result::macaque_maintenance_behaviour_link::pre_stimulation_window_control_condition",
        "construct": "rate-free angular-deviation biomarker's association with trial outcome (correct vs error) at maintenance",
        "eligibility_rule": "macaque_pfc_microstimulation release, all 11 sessions / 2 animals, both correct- and error-trial files admitted; pre-stimulation window (0.8s) per results/stimulation_design_census.json; control-arm trials only",
        "estimand": "error-minus-correct rate-free deviation, pre-stimulation window, control arm, normalised by control-trial SD",
        "dataset_view": "Macaque PFC microstimulation (dlPFC, delay-period microstimulation, 11 sessions / 2 animals)",
        "independent_unit": "animal (n=2); 11 sessions clustered by animal via whole-session sign-flip permutation + percentile bootstrap",
        "preprocessing_version": "src/stimulation_response_estimator.py rate-free angular-deviation biomarker; first computation admitting error trials for this corpus",
        "inferential_method": "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py), same primitive as the stimulation_response_gate_and_panel rows above",
        "correction_family": "single frozen per-cell decision rule declared before fitting; the two-animal sign-flip lattice floors the attainable two-sided p-value near 0.49-0.50 for any effect size, so no multiplicity correction is meaningful at this n and no cell is read by its p-value",
        "gate": "G3",
        "status": "current_exploratory",
        "manuscript_location": "PAPER_REPORT.tex, macaque PFC microstimulation maintenance-delay behaviour link section (Section~\\ref{sec:macaque-maintenance-behaviour-link})",
        "caveat": (
            "mean +0.2137 (95% CI [+0.0770,+0.3504]), positive with and without Sa210311_s224 (the "
            "one different-montage session; Wa-only descriptive value +0.3504, not a formal test at 1 "
            "cluster). The nuisance total-activity control is -0.2196 here (flips sign across the four "
            "cells of this artifact while the deviation stays positive in all four). Sits below its own "
            "80%-power mdd (0.2137 vs 0.3828), so upward-biased conditional on any apparent "
            "significance. At 2 animal clusters no significance statement is reachable in either "
            "direction (sign-flip p floor ~0.49-0.50); the mdd is reported and read, never the p-value, "
            "and clearing the 1.0 control-SD reference does not mean this design could reach "
            "significance -- that is a structural property of the 2-cluster lattice. A time-on-task "
            "confound is uncontrolled and untestable in this release: badTrialHandling is 'reshuffle' "
            "in every session with target-repetition field rpts (20/25/100), so errors accumulate "
            "wherever the animal struggles rather than being randomly interleaved in session time, and "
            "there is no shared clock between the correct- and error-trial files to test this directly. "
            "The artifact's own validity probe (KS test of correct-trial chronological rank within its "
            "own file's ordering against Uniform(0,1)) does not address this confound -- that rank is "
            "uniform by construction, 0/22 cells reject at Bonferroni or uncorrected alpha, p-values "
            "0.99999999-1.0 throughout -- and is not quoted as evidence of random interleaving. The "
            "correct-trial set is balanced by construction (pooled CV 0.033) while the error set is the "
            "unbalanced remainder (pooled CV 1.047), so n_error/(n_correct+n_error) is not an error "
            "rate and no accuracy quantity is computed from these counts. Standing verdict: suggestive, "
            "confounded, not closed."
        ),
        "model_prediction": (
            "A behaviourally meaningful maintenance-period state difference between correct and error "
            "trials predicts a positive-signed, non-total-activity deviation association at maintenance; "
            "distinguishing it from a time-on-task confound requires a shared clock between correct and "
            "error trials that this release does not provide, and distinguishing it from noise requires "
            "more than 2 animal clusters."
        ),
        "prediction_match_status": "suggestive_confounded_not_closed",
    },
    {
        "claim_id": "artifact::macaque_maintenance_behaviour_link::pre_stimulation_window_stimulated_condition",
        "result_id": "result::macaque_maintenance_behaviour_link::pre_stimulation_window_stimulated_condition",
        "construct": "rate-free angular-deviation biomarker's association with trial outcome (correct vs error) at maintenance",
        "eligibility_rule": "macaque_pfc_microstimulation release, all 11 sessions / 2 animals, both correct- and error-trial files admitted; pre-stimulation window (0.8s) per results/stimulation_design_census.json; stimulated-arm trials only",
        "estimand": "error-minus-correct rate-free deviation, pre-stimulation window, stimulated arm, normalised by control-trial SD",
        "dataset_view": "Macaque PFC microstimulation (dlPFC, delay-period microstimulation, 11 sessions / 2 animals)",
        "independent_unit": "animal (n=2); 11 sessions clustered by animal via whole-session sign-flip permutation + percentile bootstrap",
        "preprocessing_version": "src/stimulation_response_estimator.py rate-free angular-deviation biomarker; first computation admitting error trials for this corpus",
        "inferential_method": "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py), same primitive as the stimulation_response_gate_and_panel rows above",
        "correction_family": "single frozen per-cell decision rule declared before fitting; the two-animal sign-flip lattice floors the attainable two-sided p-value near 0.49-0.50 for any effect size, so no multiplicity correction is meaningful at this n and no cell is read by its p-value",
        "gate": "G3",
        "status": "current_exploratory",
        "manuscript_location": "PAPER_REPORT.tex, macaque PFC microstimulation maintenance-delay behaviour link section (Section~\\ref{sec:macaque-maintenance-behaviour-link})",
        "caveat": (
            "mean +0.2459 (95% CI [+0.1443,+0.3474]), positive with and without Sa210311_s224 (Wa-only "
            "descriptive value +0.3474, not a formal test at 1 cluster). The nuisance total-activity "
            "control is -0.2944 here (flips sign across the four cells of this artifact while the "
            "deviation stays positive in all four). Sits below its own 80%-power mdd (0.2459 vs 0.2845), "
            "so upward-biased conditional on any apparent significance. At 2 animal clusters no "
            "significance statement is reachable in either direction (sign-flip p floor ~0.49-0.50); the "
            "mdd is reported and read, never the p-value. There is no specificity to stimulation: this "
            "stimulated-arm, pre-stimulation-window cell (before this window's own stimulation acts) is "
            "the same sign and comparable magnitude to the control-arm cells. A time-on-task confound is "
            "uncontrolled and untestable in this release: badTrialHandling is 'reshuffle' in every "
            "session with target-repetition field rpts (20/25/100), so errors accumulate wherever the "
            "animal struggles rather than being randomly interleaved in session time, and there is no "
            "shared clock between the correct- and error-trial files to test this directly. The "
            "artifact's own validity probe (KS test of correct-trial chronological rank within its own "
            "file's ordering against Uniform(0,1)) does not address this confound -- that rank is "
            "uniform by construction, 0/22 cells reject at Bonferroni or uncorrected alpha, p-values "
            "0.99999999-1.0 throughout -- and is not quoted as evidence of random interleaving. The "
            "correct-trial set is balanced by construction (pooled CV 0.033) while the error set is the "
            "unbalanced remainder (pooled CV 1.047), so n_error/(n_correct+n_error) is not an error "
            "rate and no accuracy quantity is computed from these counts. Standing verdict: suggestive, "
            "confounded, not closed."
        ),
        "model_prediction": (
            "A behaviourally meaningful maintenance-period state difference between correct and error "
            "trials predicts a positive-signed, non-total-activity deviation association at maintenance; "
            "distinguishing it from a time-on-task confound requires a shared clock between correct and "
            "error trials that this release does not provide, and distinguishing it from noise requires "
            "more than 2 animal clusters."
        ),
        "prediction_match_status": "suggestive_confounded_not_closed",
    },
    {
        "claim_id": "artifact::macaque_maintenance_behaviour_link::post_stimulation_window_control_condition",
        "result_id": "result::macaque_maintenance_behaviour_link::post_stimulation_window_control_condition",
        "construct": "rate-free angular-deviation biomarker's association with trial outcome (correct vs error) at maintenance",
        "eligibility_rule": "macaque_pfc_microstimulation release, all 11 sessions / 2 animals, both correct- and error-trial files admitted; post-stimulation window (0.7s) per results/stimulation_design_census.json; control-arm trials only",
        "estimand": "error-minus-correct rate-free deviation, post-stimulation window, control arm, normalised by control-trial SD",
        "dataset_view": "Macaque PFC microstimulation (dlPFC, delay-period microstimulation, 11 sessions / 2 animals)",
        "independent_unit": "animal (n=2); 11 sessions clustered by animal via whole-session sign-flip permutation + percentile bootstrap",
        "preprocessing_version": "src/stimulation_response_estimator.py rate-free angular-deviation biomarker; first computation admitting error trials for this corpus",
        "inferential_method": "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py), same primitive as the stimulation_response_gate_and_panel rows above",
        "correction_family": "single frozen per-cell decision rule declared before fitting; the two-animal sign-flip lattice floors the attainable two-sided p-value near 0.49-0.50 for any effect size, so no multiplicity correction is meaningful at this n and no cell is read by its p-value",
        "gate": "G3",
        "status": "current_exploratory",
        "manuscript_location": "PAPER_REPORT.tex, macaque PFC microstimulation maintenance-delay behaviour link section (Section~\\ref{sec:macaque-maintenance-behaviour-link})",
        "caveat": (
            "mean +0.1090 (95% CI [-0.0185,+0.2365]), positive with and without Sa210311_s224 (Wa-only "
            "descriptive value +0.2365, not a formal test at 1 cluster). The nuisance total-activity "
            "control is +0.0615 here (flips sign across the four cells of this artifact while the "
            "deviation stays positive in all four). Sits below its own 80%-power mdd (0.1090 vs 0.3573), "
            "so upward-biased conditional on any apparent significance. At 2 animal clusters no "
            "significance statement is reachable in either direction (sign-flip p floor ~0.49-0.50); the "
            "mdd is reported and read, never the p-value. There is no specificity to the "
            "post-stimulation window: this is the smallest of the four cells, not the largest, and its "
            "sign and order of magnitude match the pre-stimulation-window cells. A time-on-task confound "
            "is uncontrolled and untestable in this release: badTrialHandling is 'reshuffle' in every "
            "session with target-repetition field rpts (20/25/100), so errors accumulate wherever the "
            "animal struggles rather than being randomly interleaved in session time, and there is no "
            "shared clock between the correct- and error-trial files to test this directly. The "
            "artifact's own validity probe (KS test of correct-trial chronological rank within its own "
            "file's ordering against Uniform(0,1)) does not address this confound -- that rank is "
            "uniform by construction, 0/22 cells reject at Bonferroni or uncorrected alpha, p-values "
            "0.99999999-1.0 throughout -- and is not quoted as evidence of random interleaving. The "
            "correct-trial set is balanced by construction (pooled CV 0.033) while the error set is the "
            "unbalanced remainder (pooled CV 1.047), so n_error/(n_correct+n_error) is not an error "
            "rate and no accuracy quantity is computed from these counts. Standing verdict: suggestive, "
            "confounded, not closed."
        ),
        "model_prediction": (
            "A behaviourally meaningful maintenance-period state difference between correct and error "
            "trials predicts a positive-signed, non-total-activity deviation association at maintenance; "
            "distinguishing it from a time-on-task confound requires a shared clock between correct and "
            "error trials that this release does not provide, and distinguishing it from noise requires "
            "more than 2 animal clusters."
        ),
        "prediction_match_status": "suggestive_confounded_not_closed",
    },
    {
        "claim_id": "artifact::macaque_maintenance_behaviour_link::post_stimulation_window_stimulated_condition",
        "result_id": "result::macaque_maintenance_behaviour_link::post_stimulation_window_stimulated_condition",
        "construct": "rate-free angular-deviation biomarker's association with trial outcome (correct vs error) at maintenance",
        "eligibility_rule": "macaque_pfc_microstimulation release, all 11 sessions / 2 animals, both correct- and error-trial files admitted; post-stimulation window (0.7s) per results/stimulation_design_census.json; stimulated-arm trials only",
        "estimand": "error-minus-correct rate-free deviation, post-stimulation window, stimulated arm, normalised by control-trial SD",
        "dataset_view": "Macaque PFC microstimulation (dlPFC, delay-period microstimulation, 11 sessions / 2 animals)",
        "independent_unit": "animal (n=2); 11 sessions clustered by animal via whole-session sign-flip permutation + percentile bootstrap",
        "preprocessing_version": "src/stimulation_response_estimator.py rate-free angular-deviation biomarker; first computation admitting error trials for this corpus",
        "inferential_method": "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py), same primitive as the stimulation_response_gate_and_panel rows above",
        "correction_family": "single frozen per-cell decision rule declared before fitting; the two-animal sign-flip lattice floors the attainable two-sided p-value near 0.49-0.50 for any effect size, so no multiplicity correction is meaningful at this n and no cell is read by its p-value",
        "gate": "G3",
        "status": "current_exploratory",
        "manuscript_location": "PAPER_REPORT.tex, macaque PFC microstimulation maintenance-delay behaviour link section (Section~\\ref{sec:macaque-maintenance-behaviour-link})",
        "caveat": (
            "mean +0.2123 (95% CI [+0.0772,+0.3474]), positive with and without Sa210311_s224 (Wa-only "
            "descriptive value +0.0772, not a formal test at 1 cluster). The nuisance total-activity "
            "control is -0.0726 here (flips sign across the four cells of this artifact while the "
            "deviation stays positive in all four). Sits below its own 80%-power mdd (0.2123 vs 0.3785), "
            "so upward-biased conditional on any apparent significance. At 2 animal clusters no "
            "significance statement is reachable in either direction (sign-flip p floor ~0.49-0.50); the "
            "mdd is reported and read, never the p-value. There is no specificity to stimulation or to "
            "the post-stimulation window: this cell's sign and order of magnitude match the "
            "pre-stimulation, control-arm cell (before any stimulation has been delivered at all). A "
            "time-on-task confound is uncontrolled and untestable in this release: badTrialHandling is "
            "'reshuffle' in every session with target-repetition field rpts (20/25/100), so errors "
            "accumulate wherever the animal struggles rather than being randomly interleaved in session "
            "time, and there is no shared clock between the correct- and error-trial files to test this "
            "directly. The artifact's own validity probe (KS test of correct-trial chronological rank "
            "within its own file's ordering against Uniform(0,1)) does not address this confound -- that "
            "rank is uniform by construction, 0/22 cells reject at Bonferroni or uncorrected alpha, "
            "p-values 0.99999999-1.0 throughout -- and is not quoted as evidence of random interleaving. "
            "The correct-trial set is balanced by construction (pooled CV 0.033) while the error set is "
            "the unbalanced remainder (pooled CV 1.047), so n_error/(n_correct+n_error) is not an error "
            "rate and no accuracy quantity is computed from these counts. Standing verdict: suggestive, "
            "confounded, not closed."
        ),
        "model_prediction": (
            "A behaviourally meaningful maintenance-period state difference between correct and error "
            "trials predicts a positive-signed, non-total-activity deviation association at maintenance; "
            "distinguishing it from a time-on-task confound requires a shared clock between correct and "
            "error trials that this release does not provide, and distinguishing it from noise requires "
            "more than 2 animal clusters."
        ),
        "prediction_match_status": "suggestive_confounded_not_closed",
    },
]

# results/train_overlap_decontamination.json directly tests the neighbouring-train-contamination
# account that the stimulation_response_gate_and_panel.json human row originally offered for its
# own failed pre/post specificity check. The decision rule's REFUTED branch fired (the
# "uncontaminated subset" partition returned zero contaminated trials, i.e. it equals the full
# corpus), so the accompanying statistical re-estimate over that subset is not a second,
# independent measurement -- it reproduces the pooled estimate exactly because nothing was
# removed. This row's status is "replication_or_informative_null" (not the artifact's own
# free-text verdict, which is not a member of VALID_STATUSES in src/provenance.py): what the
# measurement establishes is that a proposed explanation does not hold, which is an informative
# null on that hypothesis rather than confirmation of a claim the manuscript advances.
CURRENT_OVERRIDES["train_overlap_decontamination"] = {
    "claim_id": "artifact::train_overlap_decontamination::neighbouring_train_contamination_refutation",
    "result_id": "result::train_overlap_decontamination::neighbouring_train_contamination_refutation",
    "construct": (
        "test of whether cross-block/neighbouring-word stimulation-train overlap accounts for the "
        "pre-stimulation-window displacement reported in the human post-stimulation-window arm"
    ),
    "dataset_view": (
        "ram_ds005489_openloop (human intracranial, episodic encoding, open-loop arm only -- same "
        "corpus and admission gate as stimulation_response_gate_and_panel.json's human row)"
    ),
    "eligibility_rule": (
        "same 76-seen/73-used ram_ds005489_openloop admission gate as the delivered panel "
        "(channel-shank exclusion, EDF/passband, word-count exclusions); every scored WORD event "
        "independently matched to its raw BIDS event-table onset by (list, serialpos), 20,004/20,004 "
        "matched, 0 unmatched"
    ),
    "independent_unit": "participant (n=33); 73 sessions clustered by participant via whole-participant sign-flip permutation + percentile bootstrap",
    "estimand": (
        "pre-stimulation-window (0.3s) rate-free deviation displacement, restricted to the subset of "
        "scored words whose pre-stimulation window overlaps no stimulation train other than their own "
        "(a word's own assigned train and its own onset-timing trigger-latency jitter are excluded "
        "from the overlap check), tested against the reference pre-window displacement from the "
        "delivered panel"
    ),
    "preprocessing_version": (
        "src/stimulation_response_estimator.py's rate-free angular-deviation biomarker, identical to "
        "the delivered panel; each scored WORD event re-matched to its raw event-table onset (no EDF "
        "re-opened) and partitioned by neighbouring-train overlap before pooling"
    ),
    "inferential_method": (
        "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py): paired sign-flip "
        "test + bootstrap CI on participant-collapsed means, applied to the uncontaminated-subset "
        "partition. The partition returned 0 of 20,004 trials as contaminated, so this subset is the "
        "full sample and its statistical re-estimate is not independent corroboration of the "
        "delivered panel's own number -- it is arithmetically the same computation over the same "
        "trials, reproducing the pooled pre-window mean (0.0673078823256453) to every digit. The "
        "refutation of the neighbouring-train account rests entirely on the event-table timing "
        "geometry (distance from each word's onset to the nearest other train's offset), not on this "
        "re-estimate."
    ),
    "correction_family": (
        "single frozen decision rule declared before fitting (partition variable and "
        "refuted/supported/inconclusive thresholds fixed before any session was loaded); no "
        "correction applied across pre/post windows"
    ),
    "manuscript_location": (
        "PAPER_REPORT.tex, stimulation section (Section~\\ref{sec:corpus-capability}, paragraph "
        "refuting the neighbouring-train contamination account, immediately following the "
        "design-census paragraph)"
    ),
    "gate": "G3",
    "status": "replication_or_informative_null",
    "caveat": (
        "The partition variable (pre-stimulation window overlaps a stimulation train other than the "
        "word's own) returned zero contaminated trials out of 20,004 scored words in 73 sessions / 33 "
        "participants: the distance from every word's onset to the nearest OTHER train's offset is "
        "strictly positive across all 19,624 words carrying a preceding train (median 7.975s, minimum "
        "0.369s, against the 0.3s pre-stimulation window; fraction still-active 0.0). Because the "
        "'uncontaminated subset' is therefore the entire corpus, its accompanying statistical "
        "re-estimate (mean 0.0673078823256453, p=0.0459) is NOT a second, independent measurement -- "
        "it is the same pooled computation over the same full sample and reproduces the delivered "
        "panel's own pre-window number to every digit. The refutation of the neighbouring-train "
        "contamination account rests on the event-table timing geometry alone (the distance summary "
        "above), never on this re-estimate. A second account, that the second word of each "
        "stimulated pair has its own block's train running during the interval scored as its "
        "pre-stimulation window, is also refuted by direct measurement rather than left open: the "
        "delivered pulse train begins before the word it accompanies by a fixed trigger lead (median "
        "0.211s), so every stimulated word -- first- and second-in-pair alike -- already has its own "
        "train running at word onset, and no word in this corpus has a stimulation-free pre-window "
        "(results/within_block_stimulation_position.json). The pre/post comparison this artifact and "
        "the delivered panel both report on was consequently never a specificity test."
    ),
    "model_prediction": (
        "If the pre-stimulation-window displacement were explained by a neighbouring word's "
        "still-running stimulation train, restricting to words whose pre-window overlaps no train "
        "other than their own should null the effect (pooled p > 0.05) with a minimum detectable "
        "difference below the reference pre-window effect (0.0673), while the post-stimulation-window "
        "effect remains positive and significant."
    ),
    "prediction_match_status": (
        "contamination_account_refuted: the uncontaminated subset equals 100% of trials "
        "(20,004/20,004, 0 contaminated) rather than a reduced comparison group, and its pre-window "
        "displacement remains significant (mean 0.0673, p=0.0459) -- the pre-declared REFUTED branch "
        "fired directly, without reaching the power/inconclusive check"
    ),
}


# results/within_block_stimulation_position.json's own decision rule resolved to
# "account_b_confirmed" on its pre-declared trigger condition (first-in-pair pre-window p<=0.05),
# and that trigger condition did hold. But the branch's own text defines first-in-pair words as
# words "whose own block has not started stimulating yet" -- and the same artifact's
# scope.own_train_pre_window_coverage_by_position measures that premise to be false for every
# first-in-pair word (mean coverage 0.702, fraction at zero coverage 0.0 -- never zero). A
# trigger condition firing is not the same as its premise holding, so this row's status is
# "replication_or_informative_null" (matching train_overlap_decontamination above) rather than
# treating the verdict key as a confirmed positive, and prediction_match_status records the
# mismatch in free text rather than quoting the verdict key.
CURRENT_OVERRIDES["within_block_stimulation_position"] = {
    "claim_id": "artifact::within_block_stimulation_position::pre_window_coverage_refutes_specificity",
    "result_id": "result::within_block_stimulation_position::pre_window_coverage_refutes_specificity",
    "construct": (
        "whether within-pair word position (first vs. second stimulated word in each two-word "
        "block) can adjudicate a per-trial account of the pre-stimulation-window displacement, and "
        "direct measurement of each word's own-train pre-window coverage that the adjudication "
        "attempt's own decision rule presupposes to be zero for first-in-pair words"
    ),
    "eligibility_rule": (
        "same 76-seen/73-used ram_ds005489_openloop admission gate as the delivered panel "
        "(channel-shank exclusion, EDF/passband, word-count exclusions); every stimulated word "
        "matched to the pulse train whose interval encloses its own presentation window, ranked "
        "by onset within that train's own group to derive first_in_pair/second_in_pair position"
    ),
    "dataset_view": (
        "ram_ds005489_openloop (human intracranial, episodic encoding, open-loop arm only -- same "
        "corpus and admission gate as stimulation_response_gate_and_panel.json's human row)"
    ),
    "independent_unit": "participant (n=33); 73 sessions clustered by participant via whole-participant sign-flip permutation + percentile bootstrap",
    "estimand": (
        "own-train pre-stimulation-window coverage fraction by word position (3,990 first-in-pair "
        "and 3,990 second-in-pair words); pooled train-group-size histogram over all stimulated "
        "words; pre-window pooled displacement by position; post-minus-pre difference by position "
        "and pooled, with bootstrap interval"
    ),
    "preprocessing_version": (
        "src/stimulation_response_estimator.py's rate-free angular-deviation biomarker, identical "
        "to the delivered panel; each scored WORD event matched to its own enclosing pulse train "
        "from the release's raw event tables, independent of the delivered panel's pre-computed cache"
    ),
    "inferential_method": (
        "cluster_bootstrap_pooled_effect (src/stimulation_response_estimator.py): paired sign-flip "
        "test + bootstrap CI on participant-collapsed means, applied separately to first-in-pair "
        "and second-in-pair pre- and post-windows; own-train pre-window coverage is a direct "
        "arithmetic measurement (interval overlap of each word's pre-window against its own train), "
        "not an inferential statistic"
    ),
    "correction_family": (
        "single frozen decision rule declared before fitting (account_b_confirmed / "
        "account_a_confirmed / inconclusive_underpowered / neither_position_significant_despite_"
        "adequate_power, evaluated in that order); no correction applied across positions or windows"
    ),
    "gate": "G3",
    "status": "replication_or_informative_null",
    "manuscript_location": (
        "PAPER_REPORT.tex, stimulation section (Section~\\ref{sec:corpus-capability}, paragraph "
        "following the neighbouring-train refutation, replacing the earlier within-pair-position "
        "framing)"
    ),
    "caveat": (
        "The decision rule's account_b_confirmed branch fired on its own pre-declared trigger "
        "(first-in-pair pre-window p=0.0308 <= 0.05), but that branch's own text defines "
        "first-in-pair words as words 'whose own block has not started stimulating yet,' and this "
        "artifact's own measurement finds that false for every first-in-pair word: own-train "
        "pre-window coverage has mean 0.7022 and median 0.7033 (n=3,990), with 0.0 of them at zero "
        "coverage -- none. Second-in-pair words are 100% covered (mean 1.0000, n=3,990). Do not "
        "quote the account_b_confirmed verdict key as a confirmation; the trigger condition held, "
        "the premise it names did not. What this artifact usefully establishes instead is three "
        "things. (1) The coverage measurement itself: the release's pulse train begins before the "
        "word it accompanies, by a fixed trigger lead of median 0.211s (5th-95th pct 0.202-0.218s "
        "across the 4,224 first-in-pair words with a train onset before word onset), so no "
        "stimulated word in this corpus has a stimulation-free pre-window. (2) The one-train-per-"
        "two-word-block structure confirmed corpus-wide rather than inferred from one session: "
        "pooled_train_group_size_histogram is {'2': 3990} -- 3,990 trains, every one covering "
        "exactly 2 stimulated words, 0 of any other size, 0 stimulated words unmatched to an own "
        "train. (3) A null post-minus-pre difference with an interval computed for the first time: "
        "0.0268 control-trial SD overall (CI [-0.0352,0.0947], p=0.451), 0.0041 in first-in-pair "
        "words (CI [-0.0635,0.0796], p=0.915), 0.0494 in second-in-pair words (CI [-0.0404,0.1680], "
        "p=0.484) -- every arm's interval covers zero comfortably, and every cell in every arm sits "
        "below its own 80%-power minimum detectable difference, so all three point estimates are "
        "upward-biased conditional on the underlying window effects having reached significance. "
        "The within-pair-position test therefore cannot adjudicate a per-trial causal account of "
        "the pre-stimulation-window displacement, because both the account it was built to confirm "
        "and the account it was built to rule out both presuppose a stimulation-free pre-window "
        "that does not exist in this corpus."
    ),
    "model_prediction": (
        "The pre-declared account_b_confirmed branch predicts, and requires for its own premise to "
        "hold, that first-in-pair words have no stimulation running in their pre-window ('whose own "
        "block has not started stimulating yet') so that a significant first-in-pair pre-window "
        "displacement could be read as a stimulated-block-vs-control-block difference rather than "
        "an artefact of active stimulation. That premise predicts own-train pre-window coverage of "
        "0.0 for every first-in-pair word."
    ),
    "prediction_match_status": (
        "trigger_condition_fired_but_its_own_premise_measured_false: first-in-pair pre-window "
        "p=0.0308 satisfies the account_b_confirmed threshold, but own-train pre-window coverage is "
        "0.702 mean and 0.0 words at zero coverage, refuting the 'no stimulation yet' premise the "
        "branch label asserts; the verdict key must not be read as a confirmed account"
    ),
}


# Some artifacts record how the pipeline was run (environment, corpus staging, fetch planning,
# loader reuse, feasibility screening, or a plain recording-site inventory) rather than what was
# found. Holding those to the same manuscript-citation requirement as a scientific result asks for
# a citation that would misrepresent them as findings. Each of these carries no effect size, no
# p-value, no n of independent units tested, no branch verdict, no detection floor, and no minimum
# detectable difference -- if a later revision of one of these files adds any of those, it no
# longer belongs in this dict and must move back to the default "current_exploratory" status.
PIPELINE_PROVENANCE_STATUS = "pipeline_provenance"
PIPELINE_PROVENANCE_JUSTIFICATIONS = {
    "anatomical_census": (
        "an inventory of which anatomical structures and recording sites are present in each "
        "staged corpus (electrode labels, coordinates, unit counts per site); a recording-site "
        "census, not a hypothesis test, and it carries no effect size, p-value, tested-n, branch "
        "verdict, detection floor, or minimum detectable difference"
    ),
    "corpus_fetch_plan": (
        "a data-fetch plan written before any pending download ran: which candidate corpora to "
        "acquire, disk-space checks, and fetch-or-defer decisions; it describes intended data "
        "acquisition, not an analysis outcome"
    ),
    "corpus_staging_audit": (
        "an inventory of which corpora are staged on disk, their file sizes, session counts, and "
        "task/anatomy metadata as read directly from the raw files; a staging record, not an "
        "analysis"
    ),
    "environment_manifest": (
        "a record of which software packages were installed, at which versions and why, for the "
        "analysis environment; a software-provenance record, not an analysis result"
    ),
    "loader_reuse_map": (
        "an index of which staged corpora already have a working population-tensor loader "
        "somewhere in this codebase, which script owns it, and which corpora still lack one; a "
        "code-reuse map, not an analysis result"
    ),
    "watters_feasibility_assessment": (
        "a read-only feasibility check of one corpus's file formats, trial fields, and task timing "
        "to determine whether it can support a future analysis; it screens data readiness and "
        "reports descriptive corpus statistics such as trial counts and delay-epoch duration, not "
        "an inferential test with a statistical verdict"
    ),
    "ds006065_feasibility": "a metadata-only staging check that reports the corpus is absent from the configured data root",
    "open_cell_closability_audit": "a read-only inventory of unresolved cells and on-disk data availability",
    "ram_within_subject_site_feasibility": "an admission and design-feasibility gate with no outcome estimate",
    "repeated_participant_inflation_census": "a read-only exposure census that does not refit or correct any inferential result",
}
for _stem, _justification in PIPELINE_PROVENANCE_JUSTIFICATIONS.items():
    _entry = CURRENT_OVERRIDES.setdefault(_stem, {})
    _entry["status"] = PIPELINE_PROVENANCE_STATUS
    _entry["caveat"] = _justification
    _entry["model_prediction"] = "not applicable -- this artifact records pipeline operation, not a scientific prediction"
    _entry["prediction_match_status"] = "not_a_finding"


def dataset_view(stem: str) -> str:
    mappings = (
        ("000469", "DANDI 000469"), ("001187", "DANDI 001187"),
        ("000673", "DANDI 000673"), ("000574", "DANDI 000574"),
        ("boran", "Boran linked units/iEEG views"), ("miller", "Miller ECoG"),
        ("macaque_pfc_microstimulation", "Macaque PFC microstimulation"),
        ("wolff", "Wolff human EEG impulse"), ("ram", "RAM human encoding stimulation"),
        ("haslacher", "Haslacher human CLAM-tACS"),
        ("alagapan", "Alagapan human iEEG stimulation"),
        ("pfc3", "CRCNS pfc-3 pseudo-population"),
    )
    lower = stem.lower()
    for token, label in mappings:
        if token in lower:
            return label
    return "cross-dataset or method-only artifact"


def gate_for(stem: str) -> str:
    lower = stem.lower()
    if any(token in lower for token in ("closed_loop", "control", "digital_twin", "rl_policy")):
        return "G4"
    if any(token in lower for token in ("causal", "stim", "impulse", "macaque_pfc_microstimulation", "ram")):
        return "G3"
    if any(token in lower for token in ("dmd", "dynamic", "rotation", "amplification", "koopman")):
        return "G2"
    return "G1"


def construct_for(stem: str) -> str:
    lower = stem.lower()
    if "drift_simulation" in lower:
        return "estimator ground-truth validity"
    if any(token in lower for token in ("behavior", "error", "confidence", "recall")):
        return "neural-state association with behavior"
    if any(token in lower for token in ("stim", "causal", "ram", "macaque_pfc_microstimulation", "impulse")):
        return "perturbation response"
    if any(token in lower for token in ("dmd", "dynamic", "rotation", "koopman", "vstar")):
        return "latent dynamics"
    if any(token in lower for token in ("dim", "pr_", "geometry", "rsa", "ctg", "decoder")):
        return "representation or geometry"
    return "legacy analytical result"


def inferential_method(stem: str) -> str:
    tokens = re.sub(r"[_-]+", " ", stem)
    return f"artifact-specific legacy method ({tokens}); inspect artifact and source pipeline"


def row_for(path: Path, manuscript: str) -> list[dict]:
    stem = path.stem
    is_gate = stem == "drift_simulation_gate"
    superseded = stem.startswith(SUPERSEDED_PREFIXES) or stem in INTERMEDIATE_ARTIFACTS
    if is_gate:
        status = "current_confirmatory"
        caveat = "synthetic method-validation evidence only; it is not a biological result"
        prediction = "Known planted confinement, diffusion, rotation, switching, and equilibrium offsets must be recovered without smoothing."
        match = "matched"
    elif superseded:
        status = "superseded"
        caveat = (
            "smoke-test intermediate superseded by the corresponding full-cohort artifact"
            if stem in INTERMEDIATE_ARTIFACTS else
            "retained for provenance; at least one estimator, identity, null, or claim-label defect affects current interpretation"
        )
        prediction = "A future fitted (D, lambda, mu) model will determine whether this historical result matches the confined-drift account."
        match = "pending_real_data_fit"
    else:
        status = "current_exploratory"
        caveat = "legacy artifact pending per-estimand reconciliation against corrected spine outputs"
        prediction = "The fitted (D, lambda, mu) model has not yet been applied, so agreement cannot be adjudicated."
        match = "pending_real_data_fit"
    artifact_reference = path.name in manuscript
    manuscript_location = (
        "PAPER_REPORT.tex (explicit artifact reference)"
        if artifact_reference else "archival appendix (insertion pending manuscript reconciliation)"
    )
    row = {
        "result_id": f"result::{stem}",
        "claim_id": f"artifact::{stem}",
        "analysis_id": stem,
        "dataset_view": dataset_view(stem),
        "construct": construct_for(stem),
        "eligibility_rule": "as recorded in the source pipeline; requires per-estimand audit before current use",
        "independent_unit": "artifact-specific; legacy value must not be presumed independent",
        "estimand": stem.replace("_", " "),
        "preprocessing_version": "legacy or partially repaired; see source hash and implementation report",
        "inferential_method": inferential_method(stem),
        "correction_family": "exploratory unless explicitly re-declared in the frozen adjudication rule",
        "artifact_path": str(path.relative_to(ROOT)),
        "artifact_hash": sha256_file(path),
        "manuscript_location": manuscript_location,
        "status": status,
        "gate": gate_for(stem),
        "caveat": caveat,
        "model_prediction": prediction,
        "prediction_match_status": match,
    }
    overrides = CURRENT_OVERRIDES.get(stem, {})
    # An artifact normally supports exactly one ledger claim, so one override dict updates the
    # one base row. A handful of artifacts carry two or more independently gated claims (e.g. a
    # human arm and a macaque arm scored by the same panel script into one JSON file); those
    # register a list of override dicts instead, and each produces its own full row -- same
    # base fields (artifact_path/hash, default gate/status/etc.), distinct claim_id per entry.
    if isinstance(overrides, list):
        rows = []
        for override in overrides:
            merged = dict(row)
            merged.update(override)
            rows.append(merged)
        return rows
    row.update(overrides)
    return [row]


# A reduced-settings smoke run of an analysis whose full run has not landed yet is marked by this
# double-suffix naming convention (e.g. "foo.shakedown.json") rather than by listing individual
# filenames, so a new shakedown never needs a code change to stay out of the ledger. Its numbers are
# not comparable to the production run under a nearly identical filename, so it is excluded entirely
# -- never given an evidence status and never reclassified as pipeline provenance -- rather than
# risk a reduced-settings number being quoted beside a real one.
SHAKEDOWN_SUFFIX = ".shakedown.json"


def main() -> None:
    manuscript = (ROOT / "PAPER_REPORT.tex").read_text()
    all_paths = sorted(
        path for path in RESULTS.glob("*.json")
        if not path.name.endswith(".lock")
    )
    paths = []
    for path in all_paths:
        if path.name.endswith(SHAKEDOWN_SUFFIX):
            print(
                f"excluded from evidence ledger: {path.relative_to(ROOT)} "
                "(shakedown run -- reduced settings, not a scientific result)"
            )
            continue
        paths.append(path)
    rows = [row for path in paths for row in row_for(path, manuscript)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(canonical_json(rows))
    print(f"wrote {len(rows)} evidence rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
