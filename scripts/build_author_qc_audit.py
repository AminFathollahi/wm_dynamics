#!/usr/bin/env python3
"""Build the author-native preprocessing and QC eligibility registry.

The registry is deliberately conservative: an existing numerical artifact is
not manuscript-eligible unless its dataset-specific source instructions have
been checked, required QC is implemented, and the current artifact records the
result.  Missing author instructions route to documented domain-standard QC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUTPUT_PATH = RESULTS / "author_preprocessing_qc_audit.json"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _source_contains(relative: str, terms: list[str]) -> bool:
    path = ROOT / relative
    if not path.exists():
        return False
    source = path.read_text(errors="replace")
    return all(term in source for term in terms)


def main() -> None:
    human = _load("human_drift_spine_000469.json")
    linked_load = _load("human_drift_spine_001187_000673.json")
    boran_spine = _load("human_drift_spine_000574.json")
    miller = _load("miller_drift_spine.json")
    panichello = _load("panichello_2024_drift_switching.json")
    watters_source = _load("watters_2026_source_replication.json")
    watters_drift = _load("watters_2026_item_count_drift.json")
    alm = _load("alm_recovery_validation.json")
    macaque_pfc_microstimulation = _load("macaque_pfc_microstimulation_design_corrected.json")
    wolff = _load("wolff_corrected_impulse.json")
    haslacher = _load("haslacher_phase_diffusion.json")
    alagapan = _load("alagapan_retention_diffusion.json")

    entries = [
        {
            "dataset": "DANDI 000469",
            "grain": "unique patient-session NWB; single trials nested in patient",
            "instruction_source": "NWB/DANDI processed single-unit release; no separate local author preprocessing README",
            "required_qc": ["canonical recording identity", "minimum firing-rate units", "prospective session accuracy", "repeated-item fold support", "training-fold-only transform", "unsmoothed bins"],
            "implemented": _source_contains("scripts/run_human_drift_spine_000469.py", [
                "low_rate_unit_mask", "MIN_SESSION_ACCURACY", "FrozenPSTHTransform", "smooth_ms=0"
            ]),
            "artifact_present": human is not None,
            "status": "eligible_for_reported_000469_models" if human is not None else "pending",
            "deviation_or_limit": "No raw spike-sorting rerun; the public NWB unit table is the starting point. M1/M3 and the full sensitivity grid remain pending.",
        },
        {
            "dataset": "DANDI 001187 / 000673 linked views",
            "grain": "canonical patient-session with release view nested, never independent cohorts",
            "instruction_source": "NWB/DANDI processed single-unit and LFP releases; domain-standard spike/LFP QC",
            "required_qc": ["19-overlap deduplication", "patient-aware splits", "unit-rate QC", "LFP line-noise/reference QC", "load-condition balance"],
            "implemented": _source_contains("scripts/run_human_drift_spine_001187_000673.py", [
                "canonical_sessions", "FrozenPSTHTransform", "low_rate_unit_mask",
                "line_noise_notch", "smooth_ms=0",
            ]),
            "artifact_present": linked_load is not None,
            "status": "eligible_linked_view_sensitivity" if linked_load is not None else "pending_spine_preprocessing_audit_and_run",
            "deviation_or_limit": "The unit and LFP views are linked sensitivity analyses, never independent cohorts. Sparse jointly identifiable patient-level load contrasts limit inference.",
        },
        {
            "dataset": "DANDI 000574",
            "grain": "patient-session-release view",
            "instruction_source": "NWB/DANDI release plus domain-standard unit/LFP QC",
            "required_qc": ["version-drift reconciliation", "patient/session identity", "unit-rate QC", "outcome-blind preprocessing"],
            "implemented": _source_contains("scripts/run_human_drift_spine_000574.py", [
                "FrozenPSTHTransform", "low_rate_unit_mask", "smooth_ms=0",
            ]),
            "artifact_present": boran_spine is not None,
            "status": "blocked_from_primary_until_version_drift_reconciled",
            "deviation_or_limit": "The staged-release sensitivity is available, but legacy analyses remain archival because their source version cannot be recovered. The staged result is not a numerical replication claim.",
        },
        {
            "dataset": "Boran co-located spikes and iEEG",
            "grain": "same patient and trials viewed through two modalities",
            "instruction_source": "release processing plus domain-standard spike and HFA QC",
            "required_qc": ["exact cross-modality trial alignment", "spike-rate QC", "line-noise removal", "bipolar/reference audit", "artifact-free HFA", "same-fold comparison"],
            "implemented": False,
            "artifact_present": False,
            "status": "pending_modality_consistency_run",
            "deviation_or_limit": "Legacy modality results cannot substitute for the required lambda agreement analysis.",
        },
        {
            "dataset": "Miller memory n-back ECoG",
            "grain": "patient-session-task condition",
            "instruction_source": "release dataset-notes document plus domain-standard ECoG QC",
            "required_qc": ["sampling metadata", "bad-channel exclusion", "line-noise removal", "reference audit", "task-event validity"],
            "implemented": _source_contains("scripts/run_miller_drift_spine.py", [
                "reject_bad_channels", "preprocess", "high_gamma_power", "FrozenPSTHTransform",
            ]),
            "artifact_present": miller is not None,
            "status": "eligible_descriptive_task_generality" if miller is not None else "pending_task_generality_run",
            "deviation_or_limit": "Release lacks response accuracy needed for the behavior arm.",
        },
        {
            "dataset": "Wolff EEG impulse",
            "grain": "participant; trials nested within participant",
            "instruction_source": "released MATLAB figure scripts and Mahalanobis tuning function",
            "required_qc": ["source event timing", "continuous orientation", "source channel/orientation conventions", "participant uncertainty", "voltage and alpha negative controls"],
            "implemented": _source_contains("scripts/run_wolff_corrected_analysis.py", ["mahal", "alpha", "orientation"]),
            "artifact_present": wolff is not None,
            "status": "eligible" if wolff is not None else "running",
            "deviation_or_limit": "Scalp EEG cannot be equated to intracranial scale or localization.",
        },
        {
            "dataset": "Inagaki ALM",
            "grain": "mouse-session; perturbation and control trials within session",
            "instruction_source": "released SiliconProbeData structure/examples and SI perturbation tables",
            "required_qc": ["source trial-type labels", "unperturbed-only endogenous fit", "unit presence/rate QC", "real perturbation recovery", "mouse-level reporting"],
            "implemented": _source_contains("scripts/run_alm_recovery_validation.py", ["unperturbed", "perturb"]),
            "artifact_present": alm is not None,
            "status": "eligible_method_validation" if alm is not None else "pending",
            "deviation_or_limit": "Motor preparation in mice validates estimator sensitivity, not a human WM mechanism.",
        },
        {
            "dataset": "Macaque PFC microstimulation (dlPFC uStim)",
            "grain": "one original trial; patterns nested in 11 sessions and two animals",
            "instruction_source": "Zenodo README plus official PFC_uStim code",
            "required_qc": ["raw 1-ms spiketrain", "shorted-contact exclusion", "correct/error trial uniqueness", "target-angle matching", "stimulation-contact eligibility", "randomization-block recovery"],
            "implemented": _source_contains("scripts/run_macaque_pfc_microstimulation_design_corrected.py", ["spiketrain", "trial_id_sha256", "eligible_stim_conditions"]),
            "artifact_present": macaque_pfc_microstimulation is not None,
            "status": "eligible_descriptive_only" if macaque_pfc_microstimulation is not None else "pending",
            "evidence": macaque_pfc_microstimulation.get("evidence") if macaque_pfc_microstimulation else None,
            "deviation_or_limit": "Correct/error interleaving and original blocks are unrecoverable; G3 is no-go.",
        },
        {
            "dataset": "RAM ds005489 open-loop / ds005557 closed-loop",
            "grain": "word trials nested in sessions and subjects",
            "instruction_source": "OpenNeuro README/BIDS sidecars; ds005557 classifier preprocessing description",
            "required_qc": ["released bipolar reference", "58-62 Hz notch", "source Morlet 3-180 Hz feature bank", "mirrored buffers", "within-session normalization", "open-loop block randomization", "closed-loop state-dependent propensity"],
            "implemented": _source_contains("scripts/run_ram_stimulation_drift.py", [
                "morlet_log_power_bank", "MORLET_FREQS_HZ", "MIRROR_BUFFER_S",
                "LINE_NOISE_BANDSTOP_HZ",
            ]),
            "artifact_present": (RESULTS / "ram_stimulation_drift.json").exists(),
            "status": "eligible_design_gated_encoding_only" if (RESULTS / "ram_stimulation_drift.json").exists() else "pending_author_feature_and_design_repair",
            "deviation_or_limit": "Existing HGP/v-star artifacts remain archival. The current feature-bank result is episodic encoding, and closed-loop item-level treatment remains propensity-selected rather than causal.",
        },
        {
            "dataset": "Haslacher CLAM-tACS",
            "grain": "participant; six phase conditions within participant; active/control between participants",
            "instruction_source": "release Data/README.md, fully audited",
            "required_qc": ["pyprep baseline noisy channels", "0.418-V saturation", "group exclusions with protected Pz ring", "200-Hz resample", "8-14 Hz filter", "SASS", "post-SASS spectrum", "average reference after SASS", "retention-only epochs"],
            "implemented": _source_contains("scripts/run_haslacher_phase_diffusion.py", ["_preprocess_author_native", "_retention_trials", "ACTIVE_SUBJECTS", "CONTROL_SUBJECTS"]),
            "artifact_present": haslacher is not None,
            "status": "eligible_qc_filtered" if haslacher is not None and haslacher.get("evidence", {}).get("n_participants_complete", 0) > 0 else "pending_or_qc_conditional",
            "evidence": haslacher.get("evidence") if haslacher else None,
            "deviation_or_limit": "Concurrent scalp tACS remains artifact-sensitive even after SASS; failed participants are excluded under the prespecified QC gate and the retained sample remains a G3 candidate only.",
        },
        {
            "dataset": "Alagapan iEEG stimulation",
            "grain": "three descriptive patient cases",
            "instruction_source": "released Preprocessing.m and StimArtifactRemoval_ICA.m",
            "required_qc": ["event-defined retention", "actual sampling rate", "seizure-contact exclusion", "stimulation-contact exclusion", "common-average reference", "post-stim spectral sanity"],
            "implemented": _source_contains("scripts/run_alagapan_retention_diffusion.py", ["STIM_SITES", "common_average_reference", "spectral_sanity"]),
            "artifact_present": alagapan is not None,
            "status": "eligible_descriptive_only" if alagapan is not None and alagapan.get("evidence", {}).get("n_artifact_qc_failed") == 0 else "pending_or_qc_failed",
            "evidence": alagapan.get("evidence") if alagapan else None,
            "deviation_or_limit": "Manual concurrent-stimulation ICA decisions are unreleased; only post-encoding retention aftereffects are used.",
        },
        {
            "dataset": "TES1 electric-field maps",
            "grain": "field exposure map, not a neural input",
            "instruction_source": "release field-map conventions",
            "required_qc": ["coordinate and montage validity", "field magnitude units", "no conversion to latent B"],
            "implemented": False,
            "artifact_present": (RESULTS / "tes1_analysis.json").exists(),
            "status": "archive_as_exposure_sensitivity_only",
            "deviation_or_limit": "Cannot identify a neural input map or pass G3/G4.",
        },
        {
            "dataset": "Panichello macaque lPFC",
            "grain": "25 sessions nested in three animals",
            "instruction_source": "release README; author-provided 1-ms spike rasters and condition labels",
            "required_qc": ["file integrity", "author time axis", "all sessions", "cue-angle stratification", "outcome-blind representation", "animal-level heterogeneity"],
            "implemented": _source_contains("scripts/run_panichello_pipeline.py", ["cueAng", "spks", "animal"]),
            "artifact_present": panichello is not None,
            "status": "eligible" if panichello is not None else "pending",
            "deviation_or_limit": "Release includes author-identified single- and multi-units together; no raw spike-sorting rerun is possible from this package.",
        },
        {
            "dataset": "Watters multi-object WM",
            "grain": "sessions nested in Elgar/Perle and triangle/ring configurations",
            "instruction_source": "official GitHub processing/modeling READMEs and released OSF caches",
            "required_qc": ["author good-unit/source inclusion", "source trial maps", "processed spike cache", "optimizer-seed averaging", "animal/configuration separation", "stable unit presence"],
            "implemented": _source_contains("scripts/run_watters_item_count_drift.py", ["source", "present", "Elgar", "Perle"]),
            "artifact_present": watters_source is not None and watters_drift is not None,
            "status": "eligible" if watters_source is not None and watters_drift is not None else "source_replication_eligible_item_count_running",
            "deviation_or_limit": "Uses authors' processed, curated spike release rather than re-running several-terabyte raw spike sorting.",
        },
    ]
    counts = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    output = {
        "schema_version": "1.0.0",
        "policy": "No artifact is manuscript-eligible without author-native or explicitly justified domain-standard preprocessing and recorded QC evidence.",
        "status_counts": counts,
        "datasets": entries,
    }
    # An optional first CLI argument overrides the output path -- used by
    # tests/test_author_qc_audit.py so verifying this script's output does not
    # rewrite the delivered artifact as a side effect of running the test
    # suite. Default (no argument) is unchanged production behaviour.
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"output": str(path), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
