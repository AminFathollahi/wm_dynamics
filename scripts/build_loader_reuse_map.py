#!/usr/bin/env python3
"""build_loader_reuse_map.py -- for every corpus results/corpus_staging_audit.json
lists, whether some script in this repository already builds a trial-by-
unit-by-bin (or trial-by-channel-by-bin) population tensor from it, which
script and which function build it, and whether that loader is attached to
an analysis unrelated to the one that might next need it.

This is a survey, not a computation: every entry below was confirmed by
reading the named function's own docstring and import list, not inferred.
A round that needs a new corpus's tensor lost real time re-discovering
loaders that already existed for an unrelated analysis; this artifact is
the map that avoids re-discovering it again.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "loader_reuse_map.json"

# has_tensor_loader: "yes" (a function in this repo returns or directly
# supports building a (trials, units-or-channels, bins) array from this
# corpus's raw staged files), "cache_only" (a loader exists but consumes an
# already-fitted/processed cache, not the raw staged data), "no" (no loader
# anywhere in this repo touches this corpus's raw format), or
# "not_applicable" (the corpus is not a trial-structured recording at all).
CORPUS_LOADER_MAP = {
    "dandi_000469": {
        "has_tensor_loader": "yes",
        "script": "src/corpus_sessions.py", "function": "iter_dandi_000469",
        "attached_to": (
            "not tied to one analysis -- the general-purpose human single-unit loader, called directly by "
            "scripts/run_latent_model_comparison.py, scripts/run_latent_model_observation_noise_comparison.py, "
            "scripts/run_observability_and_power_census.py (via carry-forward, not refit), and several "
            "drift-spine scripts"
        ),
    },
    "dandi_001187": {
        "has_tensor_loader": "yes",
        "script": "src/corpus_sessions.py", "function": "iter_dandi_001187",
        "attached_to": "same shared, multi-caller status as iter_dandi_000469 above",
    },
    "dandi_000574": {
        "has_tensor_loader": "yes",
        "script": "src/corpus_sessions.py", "function": "iter_dandi_000574 (single-unit grain)",
        "attached_to": "same shared, multi-caller status as iter_dandi_000469 above",
        "additional_loader": {
            "grain": "depth-contact LFP (a different tensor from the same corpus, not the single-unit one above)",
            "script": "scripts/run_boran_modality_consistency.py",
            "function": "registry_sessions, lfp_maintenance_tensor",
            "attached_to": "originally the same-patient spike-vs-LFP modality test; reused by scripts/run_observability_and_power_census.py's boran_lfp_rows without modification",
        },
    },
    "dandi_000673": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_observability_and_power_census.py", "function": "dandi_000673_lfp_rows (hippocampus-labelled LFP channels only)",
        "attached_to": "written for this census; reuses scripts/run_human_drift_spine_001187_000673.py's canonical_sessions dedup for which sessions have an LFP twin",
    },
    "inagaki_alm5": {
        "has_tensor_loader": "yes",
        "script": "src/corpus_sessions.py", "function": "iter_alm (control/unperturbed trials) or load_alm_raw_session directly (both arms)",
        "attached_to": "same shared, multi-caller status as iter_dandi_000469 above",
    },
    "panichello_2024": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_state_persistence.py", "function": "panichello_rows (file-parsing convention)",
        "attached_to": "originally the persistence-contrast pipeline; reused unmodified by scripts/run_observability_and_power_census.py's panichello_unit_rows",
    },
    "pfc3": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_pfc3_content_ctg.py", "function": "load_neuron_spatial (pseudo-population, neurons not simultaneously recorded)",
        "attached_to": "originally the content-CTG test; reused unmodified by scripts/run_observability_and_power_census.py's pfc3_unit_rows",
    },
    "macaque_pfc_microstimulation": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_macaque_pfc_microstimulation_pipeline.py", "function": "load_macaque_pfc_microstimulation_session, crop_trial",
        "attached_to": "originally the causal-stimulation pipeline; reused unmodified by scripts/run_observability_and_power_census.py's macaque_pfc_microstimulation_unit_rows",
    },
    "alagapan_phase_stimulation": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_alagapan_stimulation_geometry.py", "function": "_baseline_retention_trials",
        "attached_to": "originally the stimulation-geometry pipeline; reused unmodified by scripts/run_observability_and_power_census.py's alagapan_ieeg_rows",
    },
    "haslacher_clam_tacs": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_haslacher_stimulation_geometry.py", "function": "_preprocess_author_native, _retention_trials",
        "attached_to": "originally the stimulation-geometry pipeline; reused unmodified by scripts/run_observability_and_power_census.py's haslacher_scalp_rows",
    },
    "wolff_eeg_impulse": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_wolff_corrected_analysis.py", "function": "data_directory, prepare_epoch, valid_mask",
        "attached_to": "originally the impulse-perturbation reanalysis; reused unmodified by scripts/run_observability_and_power_census.py's wolff_scalp_rows",
    },
    "kai_miller_nback": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_miller_drift_spine.py", "function": "bin_time_axis, plus src/preprocessing.py's load_subject/high_gamma_power/epoch_data",
        "attached_to": "originally the drift-spine pipeline; reused unmodified by scripts/run_observability_and_power_census.py's miller_ecog_rows",
    },
    "ram_ds005489_openloop": {
        "has_tensor_loader": "yes",
        "script": "scripts/run_observability_and_power_census.py", "function": "ram_ieeg_rows (word-presentation window, no explicit WM delay in this task)",
        "attached_to": "written for this census; matches scripts/run_ram_openloop_pipeline.py's PRE_S/POST_S convention",
    },
    "ram_ds005557_closedloop": {
        "has_tensor_loader": "no",
        "reason": (
            "not fitted by any script yet, but ram_ieeg_rows's BIDS-layout loader for the sibling "
            "ram_ds005489_openloop corpus is directly reusable -- same directory structure, same "
            "acq-bipolar iEEG convention -- and was not run only for time-budget reasons "
            "(results/observability_and_power_census.json's own not_computable_this_pass entry)"
        ),
    },
    "watters_2026": {
        "has_tensor_loader": "cache_only",
        "reason": (
            "raw per-unit spike-count pickles exist on disk in a trial-metadata schema no loader in this "
            "repo parses; every existing Watters script (scripts/run_watters_source_replication.py, "
            "scripts/run_watters_item_count_drift.py) instead consumes the OSF data_for_figures release's "
            "already-fitted per-unit likelihoods and per-trial latent gains -- a processed model-comparison "
            "cache, not a binned population tensor (results/observability_and_power_census.json's own "
            "not_computable_this_pass entry)"
        ),
    },
    "campbell": {
        "has_tensor_loader": "no",
        "reason": "staged on disk (4.9 GB) but its source publication was not yet identified as of the last staging audit, so it is not usable regardless of staging; no loader has been written",
    },
    "ds004752": {
        "has_tensor_loader": "unknown_not_confirmed",
        "reason": (
            "results/corpus_staging_audit.json's own status field reads staged, but "
            "results/observability_and_power_census.json's not_staged_corpora entry for this dataset says "
            "staging it is reserved for a separate, later effort -- the two artifacts disagree about "
            "whether this corpus is usable yet, and this map does not adjudicate that; scripts/"
            "run_lfp_structure_dynamics.py and scripts/run_variance_partition.py reference it, but whether "
            "either builds a full trial-by-channel-by-bin tensor from the raw multimodal release was not "
            "confirmed -- a corpus is listed as reachable only once its on-disk format is confirmed "
            "parseable by something in this repo, so it is listed as unconfirmed instead"
        ),
    },
    "ds005034": {
        "has_tensor_loader": "no",
        "reason": "on disk, never opened by any script in this repo (results/observability_and_power_census.json's own not_staged_corpora entry)",
    },
    "dandi_000004": {
        "has_tensor_loader": "no",
        "reason": "not staged as part of any census pass to date; a recognition-memory task, not a maintenance arm, so out of construct scope even once staged",
    },
    "tes1": {
        "has_tensor_loader": "not_applicable",
        "reason": "electric-field measurement during transcranial electrical stimulation, no working-memory task -- not a trial-structured recording this project's estimators apply to",
    },
    "connectomes_markov2014": {
        "has_tensor_loader": "not_applicable",
        "reason": "an anatomical structural connectome (macaque FLNe matrix), not a task recording -- has no trial or bin axis to build a tensor from",
    },
}


def build_loader_reuse_map() -> dict:
    n_yes = sum(1 for v in CORPUS_LOADER_MAP.values() if v["has_tensor_loader"] == "yes")
    n_cache_only = sum(1 for v in CORPUS_LOADER_MAP.values() if v["has_tensor_loader"] == "cache_only")
    n_no = sum(1 for v in CORPUS_LOADER_MAP.values() if v["has_tensor_loader"] == "no")
    n_not_applicable = sum(1 for v in CORPUS_LOADER_MAP.values() if v["has_tensor_loader"] == "not_applicable")
    n_unconfirmed = sum(1 for v in CORPUS_LOADER_MAP.values() if v["has_tensor_loader"] == "unknown_not_confirmed")
    return {
        "schema_version": "1.0.0",
        "code_commit": git_commit(ROOT),
        "corpora": CORPUS_LOADER_MAP,
        "n_corpora": len(CORPUS_LOADER_MAP),
        "n_with_a_working_tensor_loader": n_yes,
        "n_with_only_a_processed_cache_loader": n_cache_only,
        "n_with_no_loader": n_no,
        "n_not_applicable_no_trial_structure": n_not_applicable,
        "n_loader_status_unconfirmed": n_unconfirmed,
        "reading": (
            f"{n_yes} of {len(CORPUS_LOADER_MAP)} staged corpora already have a working trial-by-unit-or-"
            "channel-by-bin tensor loader somewhere in this repository, almost always attached to an "
            "analysis unrelated to whatever next needs the tensor. A round needing a corpus's population "
            "tensor should check this map before writing a new loader."
        ),
    }


def main() -> None:
    payload = build_loader_reuse_map()
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"wrote {OUTPUT_PATH}: {payload['reading']}")


if __name__ == "__main__":
    main()
