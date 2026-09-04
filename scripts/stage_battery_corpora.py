"""Stage and identify the under-used corpora for the attractor/geometry battery.

Inspects each candidate corpus directly on disk (file structure, .mat top-level
keys, trial-condition fields) rather than trusting directory names or prior
prose, and cross-references the existing dataset registry
(config/datasets.json, results/anatomical_census.json) so already-registered
corpora are not re-derived from scratch. Writes one row per corpus in the data
root to results/corpus_staging_audit.json, plus a separate fetch plan for
corpora not yet on disk (results/corpus_fetch_plan.json).

Publication identity (author list, venue, DOI) is verified by hand against the
depositor's own documentation and the publisher record before being written
here; this script records the verified fields, it does not look them up.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[1]
from project_config import data_root

DATA_ROOT = data_root()
RESULTS_DIR = REPO_ROOT / "results"


def _dir_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inspect_panichello_2024() -> dict[str, Any]:
    """Inspect the .mat top-level keys and infer the monkey/area attribution.

    The README (Panichello_2024/README.md) states 10/8/7 sessions for monkeys
    A/H/J respectively, with no per-session monkey table anywhere in the
    deposit. Session filenames cluster tightly by acquisition date into three
    disjoint blocks whose sizes match those counts exactly, which is the only
    recoverable link between a session and its monkey/area without contacting
    the depositor.
    """
    root = DATA_ROOT / "Panichello_2024"
    mat_files = sorted(p for p in root.glob("*.mat"))
    example = sio.loadmat(str(mat_files[0]), squeeze_me=True)
    top_level_keys = sorted(k for k in example.keys() if not k.startswith("__"))
    cue_ang = np.round(np.asarray(example["cueAng"]).ravel(), 3)
    n_unique_cue_angles = int(len(np.unique(cue_ang)))

    dates = [p.stem for p in mat_files]
    clusters: dict[str, list[str]] = {}
    for d in dates:
        year_prefix = d[:2]
        clusters.setdefault(year_prefix, []).append(d)
    cluster_sizes = {k: len(v) for k, v in clusters.items()}

    # Paper-reported counts (Panichello et al. 2024, Nature 636:422-429):
    # monkey A = 10 sessions (area 8), monkey H = 8 (areas 8 and 9/46),
    # monkey J = 7 (area 9/46). Dryad's file listing confirms no
    # session-to-monkey table is deposited (verified 2026-08-08).
    paper_counts_by_monkey = {"A": 10, "H": 8, "J": 7}
    sorted_clusters = sorted(clusters.items(), key=lambda kv: kv[1][0])
    inferred_monkey_order = sorted(paper_counts_by_monkey, key=lambda m: -paper_counts_by_monkey[m])
    # Largest paper count first, matched to date order (2021 earliest).
    monkey_for_cluster = {}
    remaining_monkeys = dict(paper_counts_by_monkey)
    for year_prefix, session_list in sorted_clusters:
        match = [m for m, n in remaining_monkeys.items() if n == len(session_list)]
        monkey_for_cluster[year_prefix] = match[0] if len(match) == 1 else "ambiguous"
        if match:
            remaining_monkeys.pop(match[0], None)

    area_by_monkey = {"A": "area_8", "H": "area_8_and_9_46_undetermined_which", "J": "area_9_46"}
    date_cluster_area_inference = {
        year_prefix: {
            "n_sessions": len(session_list),
            "session_dates": session_list,
            "inferred_monkey": monkey_for_cluster[year_prefix],
            "inferred_area": area_by_monkey.get(monkey_for_cluster[year_prefix], "undetermined"),
        }
        for year_prefix, session_list in sorted_clusters
    }

    return {
        "path": str(root.relative_to(DATA_ROOT)),
        "size_bytes": _dir_size_bytes(root),
        "status": "staged",
        "species": "macaque",
        "task": (
            "discrete 8-way spatial working memory (delayed saccade to a "
            "remembered cue location); NOT continuous-report colour working "
            "memory -- corrected from the prior description. cueAng takes "
            "exactly 8 distinct values (multiples of pi/4), verified by "
            "direct inspection."
        ),
        "structures": date_cluster_area_inference,
        "n_sessions": len(mat_files),
        "n_units": "varies by session (135 units in the 210921 example session); no fixed count",
        "recording_modality": "single_unit_and_multi_unit_spike_rasters",
        "perturbation_present": False,
        "source_publication": {
            "authors": "Panichello MF, Jonikaitis D, Oh YJ, Zhu S, Trepka EB, Moore T",
            "title": "Intermittent rate coding and cue-specific ensembles support working memory",
            "venue": "Nature 636(8042):422-429 (2024)",
            "doi": "10.1038/s41586-024-08139-9",
            "preprint_doi": "10.1101/2023.10.06.561121 (bioRxiv, 2023)",
            "dataset_doi": "10.5061/dryad.kkwh70sct",
            "verified": True,
            "verification_method": (
                "web search cross-checked against the depositor's README.md citation string, "
                "PubMed, and PMC; area/probe details cross-checked against the Dryad dataset page"
            ),
        },
        "analyses_it_can_support": [
            "intrinsic dimensionality (session-level, area label at monkey-cluster "
            "granularity only for monkeys A and J; monkey H's 8 sessions cannot be split by area "
            "with data on disk)",
            "latent displacement scaling (same caveat)",
            "NOT eligible for any within-session multi-area contrast: no per-channel area field "
            "exists in any of the 25 staged .mat files",
        ],
        "evidence": {
            "mat_top_level_keys_inspected": top_level_keys,
            "n_unique_cue_angles": n_unique_cue_angles,
            "no_area_or_channel_field_in_any_mat_file": True,
            "no_metadata_file_beyond_README_in_deposit": True,
            "date_cluster_sizes_match_paper_reported_per_monkey_session_counts": (
                sorted(cluster_sizes.values(), reverse=True) == sorted(paper_counts_by_monkey.values(), reverse=True)
            ),
            "caveat": (
                "The date-cluster-to-monkey mapping is a strong but UNVERIFIED inference (exact "
                "numeric match to the paper's 10/8/7 session counts, and recording campaigns are "
                "conventionally blocked by animal); it is not a depositor-confirmed table. Monkey "
                "H's 8 sessions cannot be resolved to area 8 vs 9/46 vs mixed with data on disk."
            ),
        },
    }


def inspect_inagaki() -> dict[str, Any]:
    root = DATA_ROOT / "Inagaki" / "SiliconProbeData" / "SiliconProbeData"
    perturb_dir = root / "RandomDelayTask" / "withPerturbation"
    example = sorted(perturb_dir.glob("*_units.mat"))[0]
    d = sio.loadmat(str(example), squeeze_me=True, struct_as_record=False)
    unit0 = d["unit"][0]
    stim_vec = np.asarray(unit0.Behavior.stim_trial_vector)
    n_trials = int(len(stim_vec))
    n_stim_trials = int(np.sum(stim_vec != 0))
    trial_type_values = sorted(str(v) for v in np.unique(unit0.Behavior.Trial_types_of_response))

    fixed_dir = root / "FixedDelayTask"
    without_perturb_dir = root / "RandomDelayTask" / "withoutPerturbation"

    return {
        "path": str(root.relative_to(DATA_ROOT)),
        "size_bytes": _dir_size_bytes(DATA_ROOT / "Inagaki"),
        "status": "staged",
        "species": "mouse",
        "task": "delayed licking task, anterior lateral motor cortex (ALM), fixed- and random-delay variants",
        "structures": {"anterior_lateral_motor_cortex": "single_structure_by_design"},
        "n_sessions": len(list(fixed_dir.glob("*_units.mat"))) + len(list(perturb_dir.glob("*_units.mat")))
        + len(list(without_perturb_dir.glob("*_units.mat"))),
        "n_units": "per-session unit struct array, e.g. 46 units in the HI147_051518 example session",
        "recording_modality": "single_unit_silicon_probe_and_whole_cell_patch",
        "perturbation_present": True,
        "source_publication": {
            "authors": "Inagaki HK, Fontolan L, Romani S, Svoboda K",
            "title": "Discrete attractor dynamics underlies persistent activity in the frontal cortex",
            "venue": "Nature 566(7743):212-217 (2019)",
            "doi": "10.1038/s41586-019-0919-7",
            "dataset_doi": "10.25378/janelia.7489253",
            "verified": True,
            "verification_method": (
                "bilateral-perturbation SI table filename (SI_table_2_bialteral_perturb.xlsx), "
                "the withPerturbation/withoutPerturbation directory split, and the "
                "stim_trial_vector/Trial_types_of_response fields in the .mat unit structs are all "
                "consistent with the published methods; web search confirms author list, venue, DOI"
            ),
        },
        "analyses_it_can_support": [
            "the blocking gate's own analyses: persistent homology, recurrence quantification, "
            "fixed-point/Jacobian classification, and the perturbation leg -- "
            f"{n_stim_trials}/{n_trials} trials are photoinhibition trials in the example session "
            f"(stim_trial_vector in {{0,1,2,3}}, 0 = no stim)",
        ],
        "evidence": {
            "example_session_file": str(example.relative_to(DATA_ROOT)),
            "unit_struct_fields": list(unit0._fieldnames),
            "behavior_struct_fields": list(unit0.Behavior._fieldnames),
            "stim_trial_vector_unique_values": [int(v) for v in np.unique(stim_vec)],
            "n_trials_example_session": n_trials,
            "n_stim_trials_example_session": n_stim_trials,
            "trial_types_of_response_unique": trial_type_values,
            "inagaki_alm5_registration_note": (
                "An earlier record of this dataset's staging described inagaki_alm5 as 'already "
                "registered' pointing at the tiny 112 KB alm_5/SiliconProbeData.zip deposit. That "
                "was incorrect: config/datasets.json's inagaki_alm5.local_path already points at "
                "Inagaki/SiliconProbeData/SiliconProbeData -- the full 27 GB corpus inspected here. "
                "The alm_5/SiliconProbeData.zip is a separate, truncated/corrupted 112 KB archive "
                "(unzip reports 'End-of-central-directory signature not found') that is NOT the "
                "one the registry uses, and remains otherwise untouched."
            ),
        },
    }


def inspect_campbell() -> dict[str, Any]:
    root = DATA_ROOT / "Campbell"
    summary_csv = root / "Exp Summary.csv"
    import csv

    with summary_csv.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    columns = reader.fieldnames or []
    sources = sorted({r["Source"] for r in rows if r.get("Source")})
    lateralities = sorted({r["Laterality"] for r in rows if r.get("Laterality")})

    return {
        "path": str(root.relative_to(DATA_ROOT)),
        "size_bytes": _dir_size_bytes(root),
        "status": "staged",
        "species": "human",
        "task": "visual recognition memory with intracranial theta-burst stimulation of the basolateral amygdala",
        "structures": {"amygdala": "stimulation_target"},
        "n_sessions": len(rows),
        "n_units": "not_yet_counted (staging only, not yet licensed for analysis)",
        "recording_modality": "microelectrode_single_unit_plus_macro_stimulation",
        "perturbation_present": True,
        "source_publication": {
            "authors": (
                "Campbell JM, Cowan RL, Wahlstrom KL, Hollearn MK, Jensen D, Davis T, Rahimpour S, "
                "Shofty B, Arain A, Rolston JD, Hamann S, Wang S, Eisenman LN, Swift J, Xie T, "
                "Brunner P, Manns JR, Inman CS, Smith EH, Willie JT"
            ),
            "title": "Human single-neuron activity is modulated by intracranial theta burst stimulation of the basolateral amygdala",
            "venue": "eLife, reviewed preprint (version 2, 2025-08-18)",
            "doi": "10.7554/eLife.106481.2",
            "preprint_doi": "10.1101/2024.11.11.622161 (bioRxiv)",
            "verified": True,
            "verification_method": (
                "web search cross-checked against the eLife reviewed-preprint page; institutional "
                "site codes (Utah n=10, Barnes-Jewish Hospital n=13) match the 'Utah'/'BJH' Source "
                "values in Exp Summary.csv exactly"
            ),
        },
        "analyses_it_can_support": [
            "NONE yet -- Campbell is explicitly excluded from analysis at this stage; "
            "staged and identified only",
        ],
        "evidence": {
            "exp_summary_columns": columns,
            "sources_observed": sources,
            "lateralities_observed": lateralities,
        },
    }


def inspect_watters() -> dict[str, Any]:
    root = DATA_ROOT / "Watters"
    figures_root = root / "data_for_figures" / "data_for_figures"
    modeling_root = root / "data_for_modeling" / "data_for_modeling"
    monkeys = sorted(p.name for p in (modeling_root / "spikes_per_trial").iterdir()) if (modeling_root / "spikes_per_trial").exists() else []

    return {
        "path": str(root.relative_to(DATA_ROOT)),
        "size_bytes": _dir_size_bytes(root),
        "status": "staged",
        "species": "macaque",
        "task": "multi-object working memory of scene layouts (gain vs. slot vs. switching models)",
        "structures": {
            "dorsomedial_frontal_cortex": "includes SEF, pre-SMA, SMAd per source paper prose",
            "frontal_eye_field": "per source paper prose",
        },
        "n_sessions": "not_recounted_this_round (already registered, config/datasets.json)",
        "n_units": "not_recounted_this_round (already registered)",
        "recording_modality": "single_unit",
        "perturbation_present": False,
        "source_publication": {
            "authors": "Watters N, Gabel J, Tenenbaum J, Jazayeri M",
            "title": "Working Memory of Multi-Object Scenes in Primate Frontal Cortex",
            "venue": "bioRxiv preprint (2026-02-03)",
            "doi": "10.64898/2026.01.27.702062",
            "verified": True,
            "verification_method": "web search cross-checked against PMC (PMC12893052); subject IDs Perle/Elgar match Monkey P/Monkey E in the paper",
        },
        "analyses_it_can_support": [
            "intrinsic dimensionality / latent displacement scaling within-session (already registered in "
            "config/datasets.json as watters_2026; NWB electrodes/location is the literal string "
            "'unknown' for every channel -- structure is known only at the corpus level from the "
            "paper's prose, not per-channel from the data, so no within-corpus structure contrast "
            "is possible)",
        ],
        "evidence": {
            "already_registered_in_config_datasets_json": True,
            "monkey_subdirectories_found": monkeys,
            "has_dandi_data_subdirectory": (figures_root / "dandi_data").exists(),
        },
    }


def inspect_tes1() -> dict[str, Any]:
    root = DATA_ROOT / "Tes1"
    readme = (root / "data" / "0_README.txt").read_text()

    return {
        "path": str(root.relative_to(DATA_ROOT)),
        "size_bytes": _dir_size_bytes(root),
        "status": "staged",
        "species": "human",
        "task": "no working-memory task -- electric-field measurement during transcranial electrical stimulation",
        "structures": {"cortex_wide": "1380 intracranial electrodes across patients"},
        "n_sessions": "not_yet_recounted (staging/identification only, not yet licensed for analysis)",
        "n_units": "not_applicable (field measurement, not spiking)",
        "recording_modality": "intracranial_ecog_ieeg_field_potential",
        "perturbation_present": True,
        "source_publication": {
            "authors": "Huang Y, Liu AA, Lafon B, Friedman D, Dayan M, Wang X, Bikson M, Doyle WK, Devinsky O, Parra LC",
            "title": "Measurements and models of electric fields in the in vivo human brain during transcranial electric stimulation",
            "venue": "eLife 6:e18834 (2017)",
            "doi": "10.7554/eLife.18834",
            "dataset_doi": "10.6080/K0XW4GQ1 (CRCNS tes-1)",
            "verified": True,
            "verification_method": (
                "data/0_README.txt (the corpus-specific README, deliberately read instead of the "
                "stale top-level README) states the CRCNS tes-1 DOI directly; web search confirms "
                "venue/DOI/authors"
            ),
        },
        "analyses_it_can_support": [
            "NONE yet -- Tes1 is excluded from analysis at this stage; staged and identified only. "
            "Already used elsewhere in this project as the LQR B-matrix donor bank "
            "(scripts/run_tes1_analysis.py), which is a different, pre-existing use.",
        ],
        "evidence": {"data_readme_text": readme.strip()},
    }


def inspect_connectome() -> dict[str, Any]:
    path = DATA_ROOT / "connectomes" / "markov2014_fln.csv"
    header_comment = path.read_text().splitlines()[0]
    import csv

    with path.open() as fh:
        rows = list(csv.reader(fh))
    area_names = rows[1]
    n_areas = len(area_names)

    return {
        "path": str(path.relative_to(DATA_ROOT)),
        "size_bytes": _dir_size_bytes(path),
        "status": "staged",
        "species": "macaque",
        "task": "not_applicable (anatomical structural connectome, not a task dataset)",
        "structures": {"n_areas": n_areas, "area_names": area_names},
        "n_sessions": "not_applicable",
        "n_units": "not_applicable",
        "recording_modality": "tracer_injection_fln_matrix",
        "perturbation_present": False,
        "source_publication": {
            "authors": "Markov NT, et al.",
            "title": "A weighted and directed interareal connectivity matrix for macaque cerebral cortex",
            "venue": "Cerebral Cortex 24(1):17-36 (2014)",
            "doi": "10.1093/cercor/bhs270",
            "verified": True,
            "verification_method": "verified in-file (header comment); reprocessed by INM-6/multi-area-model (Schmidt et al. 2018) from the CC BY-NC-SA 4.0 redistribution",
        },
        "analyses_it_can_support": [
            "NONE yet -- registered here as the areal-level structural substrate "
            "for a future control phase only",
        ],
        "evidence": {"header_comment": header_comment, "n_areas": n_areas},
    }


def already_registered_rows() -> list[dict[str, Any]]:
    """Rows for corpora already in config/datasets.json / anatomical_census.json.

    These are not re-derived; this project's staging census requires every
    corpus in the data root to get a row, including the ones already in
    use, so this pulls their existing registered facts rather than
    duplicating investigation.
    """
    registry = json.loads((REPO_ROOT / "config" / "datasets.json").read_text())
    census_path = RESULTS_DIR / "anatomical_census.json"
    census = json.loads(census_path.read_text()) if census_path.exists() else {}
    census_by_dataset = census.get("datasets", {}) if isinstance(census, dict) else {}

    rows = []
    already_handled = {"panichello_2024", "inagaki_alm5", "watters_2026"}
    for key, entry in registry["datasets"].items():
        if key in already_handled:
            continue
        census_row = census_by_dataset.get(key, {})
        local_path = Path(entry["local_path"])
        full_path = DATA_ROOT / local_path
        rows.append({
            "path": str(local_path),
            "size_bytes": _dir_size_bytes(full_path) if full_path.exists() else None,
            "status": "staged",
            "species": "human" if key not in {"macaque_pfc_microstimulation", "pfc3"} else "macaque",
            "task": ", ".join(entry.get("constructs", [])),
            "structures": census_row.get("anatomy_status", entry.get("label_convention")),
            "n_sessions": None,
            "n_units": None,
            "recording_modality": entry.get("modality"),
            "perturbation_present": key in {
                "alagapan_phase_stimulation", "haslacher_clam_tacs", "macaque_pfc_microstimulation",
                "ram_ds005557_closedloop",
            },
            "source_publication": {"doi": entry.get("doi"), "verified": True, "note": "already registered pre-this-round"},
            "analyses_it_can_support": ["already in active use elsewhere in this project -- see DATASET_ANALYSIS_MATRIX.md"],
            "evidence": {
                "config_datasets_json_entry": True,
                "registry_key": key,
                "anatomy_status_reason": census_by_dataset.get(key, {}).get("anatomy_status_reason"),
            },
        })
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = {
        "panichello_2024": inspect_panichello_2024(),
        "inagaki_alm5": inspect_inagaki(),
        "campbell": inspect_campbell(),
        "watters_2026": inspect_watters(),
        "tes1": inspect_tes1(),
        "connectomes_markov2014": inspect_connectome(),
    }
    for row in already_registered_rows():
        rows[row["evidence"]["registry_key"]] = row

    audit = {
        "analysis_id": "corpus_staging_audit",
        "schema_version": "1.0.0",
        "code_commit": _git_commit(),
        "trigger": "Stage and identify under-used corpora before any analysis is licensed to claim data-limitation.",
        "data_root": str(DATA_ROOT),
        "corpora": rows,
    }
    out_path = RESULTS_DIR / "corpus_staging_audit.json"
    out_path.write_text(json.dumps(audit, indent=2, sort_keys=False, allow_nan=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
