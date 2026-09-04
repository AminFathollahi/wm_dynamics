#!/usr/bin/env python3
"""Anatomical census: where every dataset records, in every patient, in every
session, before any per-structure claim is drawn from any of them.

For each dataset declared in ``config/datasets.json`` this walks every staged
patient/session and emits one row per recording site (electrode, channel, or
probe contact) carrying its raw label verbatim, a controlled-vocabulary
normalization (via ``src/spike_pipeline.normalize_region_label``), hemisphere,
coordinates with their space named, modality, unit count, and stimulation-site
identity where determined. Datasets with no staged anatomy, a single
structure by design, or anatomy dropped at staging get an explicit
``anatomy_status`` and reason instead of rows.

Output: results/anatomical_census.json, plus the structure x dataset roll-up
rendered into DATASET_ANALYSIS_MATRIX.md by
scripts/render_anatomical_census_matrix.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from spike_pipeline import (  # noqa: E402
    COARSE_REGION_GROUP,
    SCALP_1020_LABELS,
    normalize_region_label,
)

SCHEMA_VERSION = "1.0.0"

_HERE = Path(__file__).resolve().parent.parent
CONFIG_PATH = _HERE / "config" / "datasets.json"


def _data_root() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise RuntimeError(
            "WM_DYNAMICS_DATA_ROOT is not set. Export it before running -- "
            "no dataset path may be hardcoded in analysis logic."
        )
    return Path(root)


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _decode(x):
    return x.decode() if isinstance(x, bytes) else x


def _coarse_group(structure: str) -> str:
    """Coarse grouping for a fine controlled-vocabulary structure.

    Single source of truth is :data:`spike_pipeline.COARSE_REGION_GROUP`;
    ``declared_structure`` values for single-structure-by-design corpora
    (e.g. "prefrontal_cortex", "anterior_lateral_motor_cortex") are not in
    that table since they never go through :func:`normalize_region_label` --
    handled here so the roll-up still groups them sensibly.
    """
    extra = {
        "prefrontal_cortex": "lateral_frontal",
        "anterior_lateral_motor_cortex": "lateral_frontal",
    }
    return COARSE_REGION_GROUP.get(structure, extra.get(structure, "other"))


def _site(raw, structure, hemisphere, coords, modality, n_units, is_stim, source_field):
    return {
        "raw_label": raw,
        "normalized_structure": structure,
        "coarse_group": _coarse_group(structure),
        "hemisphere": hemisphere or "unknown",
        "coordinates": coords,
        "modality": modality,
        "n_units": int(n_units),
        "is_stimulation_site": is_stim,
        "source_field": source_field,
    }


# --------------------------------------------------------------------------
# NWB corpora: DANDI 000469 / 001187 / 000673 (pure microwire single-unit,
# ``{structure}_{left|right}`` labels) and 000574 / Boran (mixed scalp EEG +
# depth macro/micro electrodes, Brainnetome-hybrid labels).
# --------------------------------------------------------------------------

def _census_nwb_electrodes(fpath: Path, dataset_key: str, boran: bool) -> list[dict]:
    import h5py

    sites = []
    with h5py.File(fpath, "r") as f:
        e = f["general/extracellular_ephys/electrodes"]
        n_elec = e["id"].shape[0]
        locations = [_decode(x) for x in e["location"][:]]
        labels = [_decode(x) for x in e["label"][:]] if "label" in e else [None] * n_elec
        group_names = (
            [_decode(x) for x in e["group_name"][:]] if "group_name" in e else [None] * n_elec
        )
        xs = e["x"][:] if "x" in e else np.full(n_elec, np.nan)
        ys = e["y"][:] if "y" in e else np.full(n_elec, np.nan)
        zs = e["z"][:] if "z" in e else np.full(n_elec, np.nan)

        # Units-per-electrode count, handling both the flat-index layout
        # (000469/001187/000673) and the ragged VectorIndex layout (000574).
        units_per_electrode = np.zeros(n_elec, dtype=int)
        if "units" in f and "electrodes" in f["units"]:
            if "units/electrodes_index" in f:
                elec_flat = f["units/electrodes"][:]
                for row in elec_flat:
                    units_per_electrode[row] += 1
            else:
                elec_rows = f["units/electrodes"][:]
                for row in elec_rows:
                    units_per_electrode[row] += 1

        for i in range(n_elec):
            raw = locations[i]
            label = labels[i]
            gname = group_names[i]
            coords = {
                "x": float(xs[i]) if np.isfinite(xs[i]) else None,
                "y": float(ys[i]) if np.isfinite(ys[i]) else None,
                "z": float(zs[i]) if np.isfinite(zs[i]) else None,
                "space": "native_implant" if boran else "native_microwire_bundle",
            }
            n_units = int(units_per_electrode[i])
            if boran:
                is_scalp = (gname == "eeg") or (
                    label is not None and label.lower() in SCALP_1020_LABELS
                )
                if is_scalp:
                    structure, hemisphere = "scalp", None
                    modality = "scalp_eeg"
                else:
                    structure, hemisphere = normalize_region_label(
                        raw, "nwb_boran_brainnetome_hybrid"
                    )
                    modality = "single_unit" if n_units > 0 else "depth_lfp"
            else:
                structure, hemisphere = normalize_region_label(
                    raw, "nwb_structure_hemisphere_suffix"
                )
                modality = "single_unit"
            sites.append(
                _site(
                    raw, structure, hemisphere, coords, modality, n_units,
                    is_stim=False,
                    source_field="general/extracellular_ephys/electrodes/location",
                )
            )
    return sites


def _census_rutishauser_family(dataset_key: str, local_path: Path) -> dict:
    boran = dataset_key == "dandi_000574"
    patients: dict[str, dict] = {}
    nwb_files = sorted(local_path.glob("sub-*/*.nwb"))
    for fpath in nwb_files:
        stem = fpath.stem  # e.g. sub-1_ses-2_ecephys+image or sub-01_ses-01
        parts = stem.split("_")
        sub = next((p for p in parts if p.startswith("sub-")), fpath.parent.name)
        ses = next((p for p in parts if p.startswith("ses-")), "ses-1")
        try:
            sites = _census_nwb_electrodes(fpath, dataset_key, boran)
        except Exception as exc:
            sites = [{"error": f"resolve_unit_regions-equivalent failed: {exc!r}"}]
        patients.setdefault(sub, {"sessions": {}})["sessions"][ses] = {
            "recording_sites": sites,
            "source_file": str(fpath),
        }
    return {
        "anatomy_status": "labelled",
        "anatomy_status_reason": (
            "NWB general/extracellular_ephys/electrodes/location present for every session; "
            + ("mixed scalp EEG + depth macro/micro, Brainnetome-hybrid labels" if boran
               else "pure microwire single-unit, {structure}_{hemisphere} labels")
        ),
        "patients": patients,
    }


# --------------------------------------------------------------------------
# BIDS iEEG corpora with electrodes.tsv: RAM ds005489/ds005557 (Desikan-
# Killiany `ind.region`) and ds004752 (Brainnetome-hybrid `AnatomicalLocation`).
# --------------------------------------------------------------------------

def _read_tsv(path: Path) -> list[dict]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _census_ram(dataset_key: str, local_path: Path) -> dict:
    patients: dict[str, dict] = {}
    for sub_dir in sorted(local_path.glob("sub-*")):
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            ieeg_dir = ses_dir / "ieeg"
            if not ieeg_dir.is_dir():
                continue
            elec_files = sorted(ieeg_dir.glob("*_electrodes.tsv"))
            if not elec_files:
                continue
            elec_path = elec_files[0]
            rows = _read_tsv(elec_path)
            stim_channels: set[str] = set()
            for ev_path in sorted(ieeg_dir.glob("*_events.tsv")):
                for row in _read_tsv(ev_path):
                    for key in ("anode_label", "cathode_label"):
                        v = row.get(key, "")
                        if v and v != "n/a":
                            stim_channels.add(v)
            sites = []
            for row in rows:
                # Prefer the hand/subcortical annotation columns over the
                # cortical-surface pipeline: das.region/stein.region carry
                # hippocampal-subfield and amygdala labels that ind.region
                # (a surface parcellation) cannot represent at all, and
                # wb.region (ds005557 only) is whole-brain including
                # subcortex. Falls back to ind.region, then "n/a".
                raw, source_col = None, "ind.region"
                for col in ("stein.region", "das.region", "wb.region", "ind.region"):
                    v = (row.get(col) or "").strip()
                    if v and v.lower() != "n/a":
                        raw, source_col = v, col
                        break
                if raw is None:
                    raw = "n/a"
                structure, hemi_from_label = normalize_region_label(
                    raw, "bids_desikan_killiany_ind_region"
                )
                hemi = hemi_from_label or {
                    "l": "left", "r": "right",
                }.get((row.get("hemisphere") or "").strip().lower())
                coords = {
                    "x": _to_float(row.get("x")), "y": _to_float(row.get("y")),
                    "z": _to_float(row.get("z")), "space": "MNI152NLin6ASym",
                }
                sites.append(
                    _site(
                        raw, structure, hemi, coords, "depth_lfp", 0,
                        is_stim=(row.get("name") in stim_channels),
                        source_field=f"{elec_path.name}#{source_col}",
                    )
                )
            patients.setdefault(sub, {"sessions": {}})["sessions"][ses] = {
                "recording_sites": sites, "source_file": str(elec_path),
            }
    return {
        "anatomy_status": "labelled",
        "anatomy_status_reason": (
            "BIDS electrodes.tsv, preferring stein.region > das.region > "
            "wb.region (hand/subcortical MTL annotations, recovering "
            "hippocampus/amygdala that the surface pipeline cannot represent) "
            "over ind.region (Desikan-Killiany surface pipeline, used only "
            "where none of the finer columns are populated), plus Talairach/"
            "MNI coordinates and events.tsv anode_label/cathode_label for "
            "stimulation-site identity, every session."
        ),
        "patients": patients,
    }


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _census_ds004752(local_path: Path) -> dict:
    patients: dict[str, dict] = {}
    for sub_dir in sorted(local_path.glob("sub-*")):
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            ieeg_dir = ses_dir / "ieeg"
            if not ieeg_dir.is_dir():
                continue
            elec_files = sorted(ieeg_dir.glob("*_electrodes.tsv"))
            if not elec_files:
                continue
            elec_path = elec_files[0]
            rows = _read_tsv(elec_path)
            sites = []
            for row in rows:
                raw = row.get("AnatomicalLocation") or "n/a"
                structure, hemi = normalize_region_label(
                    raw, "bids_brainnetome_anatomical_location"
                )
                coords = {
                    "x": _to_float(row.get("x")), "y": _to_float(row.get("y")),
                    "z": _to_float(row.get("z")), "space": "MNI",
                }
                sites.append(
                    _site(
                        raw, structure, hemi, coords, "depth_lfp", 0,
                        is_stim=False,
                        source_field=f"{elec_path.name}#AnatomicalLocation",
                    )
                )
            patients.setdefault(sub, {"sessions": {}})["sessions"][ses] = {
                "recording_sites": sites, "source_file": str(elec_path),
            }
    return {
        "anatomy_status": "labelled",
        "anatomy_status_reason": (
            "BIDS electrodes.tsv `AnatomicalLocation` (MNI coords mapped to the "
            "Brainnetome atlas), every session. This dataset also carries LCMV-"
            "beamformed cortical virtual-sensor and scalp-EEG derivatives with no "
            "per-channel anatomical label of their own -- those rungs inherit the "
            "iEEG electrode census only where results/cross_modality_calibration.json "
            "already links them; they are not separately censused here."
        ),
        "patients": patients,
    }


# --------------------------------------------------------------------------
# Coordinate-only corpora: record coordinates_only and defer -- an explicitly
# acceptable outcome, not a partial result -- with no atlas dependency added.
# --------------------------------------------------------------------------

def _census_coordinates_only(dataset_key: str, local_path: Path) -> dict:
    patients: dict[str, dict] = {}
    if dataset_key == "alagapan_phase_stimulation":
        mni_dir = local_path / "Electrode Information" / "Electrode Locations (MNI Space)"
        for fp in sorted(mni_dir.glob("*_ElectrodeLocations_MNI.csv")):
            sub = fp.stem.split("_")[0]
            rows = []
            import csv
            with open(fp, newline="") as f:
                rows = list(csv.DictReader(f))
            sites = [
                _site(
                    row.get("label", "n/a"), "other", None,
                    {"x": _to_float(row.get("x")), "y": _to_float(row.get("y")),
                     "z": _to_float(row.get("z")), "space": "MNI"},
                    "depth_lfp", 0, is_stim=None,
                    source_field=f"{fp.name}",
                )
                for row in rows
            ]
            patients.setdefault(sub, {"sessions": {}})["sessions"]["ses-1"] = {
                "recording_sites": sites, "source_file": str(fp),
            }
        reason = (
            "Per-patient MNI electrode CSVs (id,x,y,z,label) with NO anatomical "
            "label -- label is a channel name (e.g. 'RPS4'), not a structure. This "
            "project's standing rule for this situation is to record coordinates_only "
            "and defer rather than hand-assign or run a nearest-neighbour atlas "
            "lookup; no atlas lookup has been run, so every site here is 'other' with "
            "its raw channel name preserved, not a structure guess."
        )
    elif dataset_key == "kai_miller_nback":
        import scipy.io as sio
        locs_dir = local_path / "locs"
        for fp in sorted(locs_dir.glob("*_electrodes.mat")):
            sub = fp.stem.replace("_electrodes", "")
            d = sio.loadmat(fp, simplify_cells=True)
            coords_arr = np.atleast_2d(np.asarray(d["electrodes"]))
            sites = [
                _site(
                    f"ch{i+1}", "other", None,
                    {"x": float(row[0]), "y": float(row[1]), "z": float(row[2]), "space": "MNI"},
                    "ecog", 0, is_stim=None, source_field=f"{fp.name}#electrodes",
                )
                for i, row in enumerate(coords_arr)
            ]
            patients.setdefault(sub, {"sessions": {}})["sessions"]["ses-1"] = {
                "recording_sites": sites, "source_file": str(fp),
            }
        reason = (
            "40 ECoG electrode MNI coordinates per subject (locs/*_electrodes.mat), "
            "NO structure labels. This project's standing rule for this situation is "
            "to record coordinates_only and defer rather than hand-assign; no atlas "
            "lookup has been run."
        )
    else:
        raise ValueError(dataset_key)
    return {"anatomy_status": "coordinates_only", "anatomy_status_reason": reason, "patients": patients}


def _census_panichello_monkey_cluster(local_path: Path) -> dict:
    """Panichello_2024 has no per-channel area label in any staged .mat file
    (verified by direct inspection); the finest anatomy this deposit
    supports is a per-session monkey/area cluster inferred from session-date
    grouping, sourced from results/corpus_staging_audit.json's own
    panichello_2024 evidence rather than re-derived here. Each session's
    file stem is its recording date (e.g. 210921.mat), which is exactly the
    date string that evidence groups sessions by."""
    evidence_path = _HERE / "results" / "corpus_staging_audit.json"
    clusters = json.loads(evidence_path.read_text())["corpora"]["panichello_2024"]["structures"]
    date_to_cluster = {
        date: (cluster_key, cluster["inferred_area"])
        for cluster_key, cluster in clusters.items()
        for date in cluster["session_dates"]
    }
    patients: dict[str, dict] = {}
    for fp in sorted(local_path.glob("*.mat")):
        date = fp.stem
        cluster_key, area = date_to_cluster.get(date, (None, None))
        if area is None:
            sites: list[dict] = []
        else:
            sites = [_site(
                f"monkey_cluster_{cluster_key}", area, None, {"space": "none"},
                "single_unit", 0, is_stim=False, source_field=f"{fp.name}#inferred_from_corpus_staging_audit",
            )]
        patients[date] = {"sessions": {date: {"recording_sites": sites, "source_file": str(fp)}}}
    reason = (
        "No area/channel field exists in any staged Panichello_2024 .mat file (verified by direct "
        "inspection: only spike rasters, timing, and behaviour are present). The finest anatomy this "
        "deposit supports is a per-session monkey/area cluster inferred from session-date grouping "
        "(results/corpus_staging_audit.json's panichello_2024 evidence), not per-channel anatomy -- a "
        "structure label here means 'this session belongs to a monkey/date cluster inferred to record "
        "from this area', not 'this unit was localized to this area'. One cluster's area is itself "
        "reported as undetermined between two areas (2022, monkey H) and is carried through as such."
    )
    return {"anatomy_status": "single_structure_by_design_inferred_cluster", "anatomy_status_reason": reason, "patients": patients}


# --------------------------------------------------------------------------
# Non-computable statuses: not_staged, single_structure_by_design,
# no_anatomy_available. No per-site rows -- the status and reason ARE the
# census answer.
# --------------------------------------------------------------------------

def _census_not_staged_panichello(local_path: Path) -> dict:
    import scipy.io as sio
    n_sessions = len(list(local_path.glob("*.mat")))
    patients = {}
    for fp in sorted(local_path.glob("*.mat")):
        try:
            m = sio.loadmat(fp, simplify_cells=True)
            fields = sorted(k for k in m.keys() if not k.startswith("__"))
        except Exception as exc:
            fields = [f"<load failed: {exc!r}>"]
        patients[fp.stem] = {
            "sessions": {
                fp.stem: {
                    "recording_sites": [],
                    "source_file": str(fp),
                    "staged_fields": fields,
                }
            }
        }
    return {
        "anatomy_status": "not_staged",
        "anatomy_status_reason": (
            f"Staged extract ({n_sessions} session .mat files) carries only spike "
            "rasters ('spks'), timing ('tc'), and behaviour ('cueAng', 'cueAngIdx', "
            "'isCorr') -- verified by direct inspection, no area/channel field is "
            "present in any staged file, although the source study (Panichello, "
            "Jonikaitis, Oh, Zhu, Trepka & Moore 2023/2024, lateral prefrontal "
            "cortex, macaque) recorded from multiple sub-areas. ACQUISITION ITEM: "
            "recovering area identity requires re-downloading "
            "from the source Dryad release (doi 10.5061/dryad.kkwh70sct) or a "
            "session-metadata file not included in the currently staged extract; "
            "not staged or acted on, reported only."
        ),
        "patients": patients,
    }


def _census_single_structure(dataset_key: str, declared_structure: str, note: str) -> dict:
    return {
        "anatomy_status": "single_structure_by_design",
        "declared_structure": declared_structure,
        "anatomy_status_reason": note,
        "patients": {},
    }


def _census_no_anatomy(dataset_key: str, note: str) -> dict:
    return {
        "anatomy_status": "no_anatomy_available",
        "anatomy_status_reason": note,
        "patients": {},
    }


# --------------------------------------------------------------------------
# Roll-up: structure x dataset matrix.
# --------------------------------------------------------------------------

def _roll_up(datasets: dict) -> dict:
    matrix: dict[str, dict[str, dict]] = {}
    for dkey, drec in datasets.items():
        per_structure: dict[str, dict] = {}
        for sub, srec in drec.get("patients", {}).items():
            structures_this_patient: set[str] = set()
            for ses_rec in srec.get("sessions", {}).values():
                for site in ses_rec.get("recording_sites", []):
                    struct = site.get("normalized_structure")
                    if struct is None:
                        continue
                    structures_this_patient.add(struct)
                    d = per_structure.setdefault(struct, {"n_patients": 0, "n_sites": 0, "n_units": 0})
                    d["n_sites"] += 1
                    d["n_units"] += site.get("n_units", 0) or 0
            for struct in structures_this_patient:
                per_structure[struct]["n_patients"] += 1
        drec["structure_roll_up"] = per_structure
        for struct, counts in per_structure.items():
            matrix.setdefault(struct, {})[dkey] = counts
    return matrix


def build_census(config: dict) -> dict:
    root = _data_root()
    out_datasets: dict[str, dict] = {}
    for dkey, drec in config["datasets"].items():
        conv = drec["label_convention"]
        local_path = root / drec["local_path"] if "local_path" in drec else None
        if conv == "nwb_structure_hemisphere_suffix" or (
            conv == "nwb_boran_brainnetome_hybrid"
        ):
            out_datasets[dkey] = _census_rutishauser_family(dkey, local_path)
        elif conv == "bids_desikan_killiany_ind_region":
            out_datasets[dkey] = _census_ram(dkey, local_path)
        elif conv == "bids_brainnetome_anatomical_location":
            out_datasets[dkey] = _census_ds004752(local_path)
        elif conv == "mni_coordinates_only":
            out_datasets[dkey] = _census_coordinates_only(dkey, local_path)
        elif conv == "not_staged":
            out_datasets[dkey] = _census_not_staged_panichello(local_path)
        elif conv == "monkey_cluster_area_inferred":
            out_datasets[dkey] = _census_panichello_monkey_cluster(local_path)
        elif conv == "single_structure_by_design":
            note = {
                "inagaki_alm5": (
                    "Mouse anterior lateral motor cortex (ALM), silicon-probe "
                    "recordings, by experimental design (Janelia ALM5 release; "
                    "Inagaki lab perturbation/recovery studies). Single structure "
                    "is a recorded fact verified against the source documentation, "
                    "not an unverified exemption."
                ),
                "macaque_pfc_microstimulation": (
                    "Macaque prefrontal cortex microstimulation, by experimental "
                    "design (the 2025 JNeurosci release paper, README.txt "
                    "verified directly: 'Robustness of working memory to "
                    "prefrontal cortex microstimulation'). config/datasets.json "
                    "additionally labels this dlPFC; the README itself says only "
                    "'prefrontal cortex' -- reported as declared by the source "
                    "documentation; the more specific claim is not independently "
                    "verified in the staged files."
                ),
                "pfc3": (
                    "Macaque prefrontal cortex, four monkeys (ADR, BEN, ELV, SCR), "
                    "by experimental design (CRCNS pfc-3; Meyer, Qi, Stanford & "
                    "Constantinidis 2011; Qi et al. 2011 -- data description PDF "
                    "verified directly). The source papers distinguish dorsal vs "
                    "ventral PFC per-neuron in SummaryDatabase.xlsx; that finer "
                    "split has not yet been parsed and is flagged for follow-up, "
                    "not assumed."
                ),
            }[dkey]
            out_datasets[dkey] = _census_single_structure(dkey, drec.get("declared_structure"), note)
        elif conv == "no_anatomy_available":
            note = {
                "watters_2026": (
                    "NWB general/extracellular_ephys/electrodes/location is the "
                    "literal string 'unknown' for every channel in every staged "
                    "spikesorting NWB file (verified by direct inspection, "
                    "sub-Perle). config/datasets.json's construct describes "
                    "macaque frontal cortex, but this project's standing rule "
                    "requires verifying single-structure identity against the "
                    "dataset's own documentation, not assuming it -- no anatomical label "
                    "is present in the staged metadata to verify against, so no "
                    "structure is asserted here. Reported as no_anatomy_available "
                    "pending source-paper confirmation, not single_structure_by_design."
                ),
                "haslacher_clam_tacs": (
                    "Scalp EEG (BrainVision .vhdr/.eeg/.vmrk), 10-20-family cap. "
                    "No anatomy exists for scalp EEG, by construction -- correct "
                    "and final."
                ),
                "wolff_eeg_impulse": (
                    "Scalp EEG. No anatomy exists for scalp EEG, by construction "
                    "-- correct and final."
                ),
            }[dkey]
            out_datasets[dkey] = _census_no_anatomy(dkey, note)
        else:
            raise ValueError(f"unhandled label_convention {conv!r} for {dkey}")
    structure_by_dataset_matrix = _roll_up(out_datasets)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope_note": (
            "Covers all 16 datasets registered in config/datasets.json as of "
            "2026-08-04 (widened from 13 -- see config/datasets.json "
            "'registry_note' -- to add wolff_eeg_impulse, kai_miller_nback and "
            "pfc3, all three required census subjects for this project's data "
            "inventory). No dataset is silently excluded; every one has an "
            "anatomy_status below, including the ones with zero recording sites."
        ),
        "datasets": out_datasets,
        "structure_by_dataset_matrix": structure_by_dataset_matrix,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(_HERE / "results" / "anatomical_census.json"))
    args = ap.parse_args()

    config = _load_config()
    census = build_census(config)

    def _clean(obj):
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    census = _clean(census)
    with open(args.out, "w") as f:
        json.dump(census, f, indent=2)

    n_datasets = len(census["datasets"])
    n_labelled = sum(1 for d in census["datasets"].values() if d["anatomy_status"] == "labelled")
    print(f"wrote {args.out}: {n_datasets} datasets, {n_labelled} labelled with per-site rows")
    for dkey, drec in census["datasets"].items():
        n_patients = len(drec.get("patients", {}))
        print(f"  {dkey}: anatomy_status={drec['anatomy_status']!r} n_patients={n_patients}")


if __name__ == "__main__":
    main()
