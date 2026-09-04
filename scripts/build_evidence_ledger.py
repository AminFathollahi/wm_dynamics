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
        "claim_id": "current::randomised_prestimulation_state_moderation",
        "dataset_view": "human open-loop intracranial stimulation during free-recall encoding",
        "construct": "causal moderation of the stimulation effect by the immediately preceding neural state",
        "eligibility_rule": "experimenter-scheduled word stimulation with a preceding within-list word, at least 3 stimulated and 3 non-stimulated lists per session, and a computable subject-level partial effect",
        "independent_unit": "subject (33 analysed subjects; 71 sessions and 18,007 word events are repeated measures)",
        "estimand": "subject-mean partial correlation for prestimulation-state by stimulation interaction, separately for current-word displacement and later recall",
        "preprocessing_version": "stimulated bipolar contacts excluded; within-session directional deviation; serial position, alternation phase, list number, and preceding-word stimulation controlled",
        "inferential_method": "subject-cluster bootstrap interval and within-subject permutation test, with pooled random-intercept checks reported separately",
        "correction_family": "two pre-declared outcomes with native, bias-only, and preceding-word-unstimulated sensitivity arms",
        "status": "current_confirmatory",
        "gate": "G3",
        "caveat": "native displacement moderation is -0.00534 (95% CI -0.0332 to 0.0231, p=0.571, MDD=0.0407) and native recall moderation is 0.00216 (95% CI -0.0158 to 0.0227, p=0.786, MDD=0.0293), both in 33 subjects against the 0.14 reference; both are powered nulls at the reported bound",
        "model_prediction": "if the immediately preceding neural state determines whether randomised stimulation helps or hurts, the native interaction should differ from zero while the bias-only control does not reproduce it",
        "prediction_match_status": "not matched for either displacement or recall; both outcomes resolve to no moderation above their reported detection bounds",
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
        "construct": "whether the LFP and single-unit grains admit the same observable set, and which grain is ahead on a margin-aware ranking, in the same patients",
        "eligibility_rule": "Boran DANDI 000574 sessions with both LFP and single-unit views processed",
        "independent_unit": "session, with an admission-margin band and a rank-sum margin comparison, not only a binary admit/reject call",
        "estimand": "admitted observable set per grain at 100 and 200 ms bins; summed rank difference (LFP minus unit) over shared observables",
        "inferential_method": "set-membership comparison plus a margin-aware rank-sum comparison, both pre-declared",
        "status": "replication_or_informative_null",
        "gate": "G1",
        "caveat": "at 100 ms both grains admit an identical seven-observable set (branch grains_admit_the_same_set) -- the artifact's own margin_statement calls this a real limit of a set-valued rule, not evidence of equivalence, and that limit is quoted rather than laundered into a null result. The margin-aware comparison resolves at both bin widths to unit_grain_worse_instrumented_at_margin (summed rank difference LFP minus unit = 2 at 100 ms): at margin, the LFP grain is ahead, a useful positive for a project whose corpora are mostly clinical LFP, not a null to bury.",
        "model_prediction": "if single units were the more informative grain, unit-grain sessions should be preferentially admitted and rank ahead of LFP on the shared observable set.",
        "prediction_match_status": "not matched: the set-valued rule cannot distinguish the grains by construction, and where a margin-aware rule can distinguish them, LFP is ahead, not behind.",
    },
    "observability_matched_modality_test": {
        "claim_id": "current::lfp_versus_unit_grain_paired_discordance",
        "construct": "paired discordance test of estimator fittability between the LFP and single-unit grains, same patients and sessions",
        "eligibility_rule": "Boran DANDI 000574 sessions with both grains attempted; discordant sessions only (concordant sessions carry no directional information by construction)",
        "independent_unit": "session, with a patient-clustered companion field (patient-clustered discordant-session median code)",
        "estimand": "exact two-sided binomial test on discordant sessions favouring LFP versus favouring the unit grain",
        "inferential_method": "exact binomial test against a 50/50 null, session-level and patient-clustered",
        "status": "replication_or_informative_null",
        "gate": "G1",
        "caveat": "200 ms: 6 LFP-only versus 4 unit-only discordant sessions, p=0.754; 100 ms: 1 versus 1, p=1.0 -- null at both bin widths, session-level and patient-clustered. This paired test and the margin-aware comparison in results/observability_and_power_census.json ask different questions (symmetric discordance versus rank margin) and are not in tension: the grains are indistinguishable on this test and the LFP grain is ahead on the margin-aware one.",
        "model_prediction": "if one grain were more fittable in general, discordant sessions should favour it asymmetrically.",
        "prediction_match_status": "not matched: discordant sessions split at or near 50/50 at both bin widths.",
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


def row_for(path: Path, manuscript: str) -> dict:
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
    row.update(CURRENT_OVERRIDES.get(stem, {}))
    return row


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
    rows = [row_for(path, manuscript) for path in paths]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(canonical_json(rows))
    print(f"wrote {len(rows)} evidence rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
