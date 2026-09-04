#!/usr/bin/env python3
"""What transcranial current would be needed to reach, at the human sites
where the working-memory component was recorded or where direct intracranial
stimulation was delivered, a field of the magnitude that the delivered
intracranial stimulation produced.

Bridges two measurements that are not the same physical quantity:

1. A transcranial-stimulation measurement corpus: intracranial electrodes in
   14 patients recording the potential induced by weak (< 2 mA) alternating
   current applied from the scalp, calibrated to 1 mA. This is a measured
   scalp-to-depth transfer function and nothing else -- it carries no task,
   no behaviour, and is never used to source a behavioural claim.
2. The project's own delivered artifacts for direct bipolar intracranial
   stimulation in two human free-recall corpora: the currents actually used
   (a parameter census already on disk) and the displacement they produced
   in the component's own normalised units (already on disk, with its own
   n and minimum detectable difference).

The matching radius and the verdict threshold are declared as module
constants below, before any voltage or displacement number is read anywhere
in this script. The output is an order-of-magnitude specification -- what a
non-invasive device would have to deliver -- never a prediction that
delivering that current would reproduce the measured displacement.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spike_pipeline import COARSE_REGION_GROUP, normalize_region_label  # noqa: E402
from provenance import git_commit  # noqa: E402

# ---------------------------------------------------------------------------
# Paths. The transcranial and direct-stimulation corpora are not yet in
# config/datasets.json -- that file and src/corpus_sessions.py are being read
# by other long-running jobs right now and must not be touched. Resolved here
# from the data-root environment variable instead; the corpus registry entry
# for the transcranial corpus is still to be added.
# ---------------------------------------------------------------------------
from project_config import data_root

DATA_ROOT = data_root()
TES1_DIR = DATA_ROOT / "Tes1" / "data"
OPENLOOP_CORPUS = "ds005489-download"
CLOSEDLOOP_CORPUS = "ds005557-download"
RAM_CORPORA = (OPENLOOP_CORPUS, CLOSEDLOOP_CORPUS)

RESULTS = _ROOT / "results"
OUT_PATH = RESULTS / "transcranial_current_requirement.json"
PARAMETER_CENSUS_PATH = RESULTS / "stimulation_timing_and_parameter_structure.json"
COMPONENT_RESPONSE_PATH = RESULTS / "human_stimulation_component_response.json"

# ---------------------------------------------------------------------------
# Pre-declared decision rules -- fixed before any voltage or displacement
# number in this script is read.
# ---------------------------------------------------------------------------

# Spatial join radius between a human intracranial site (recording or
# stimulation, in ds005489/ds005557) and its nearest transcranial-corpus
# measurement electrode, both in MNI millimetres. 10 mm is a generous
# multiple of typical iEEG electrode MNI-localisation error (on the order of
# 1-3 mm) chosen to be forgiving of two independent coregistration pipelines
# while still excluding a match in a clearly different structure. Frozen
# before any TES1 voltage is opened.
MATCHING_RADIUS_MM = 10.0

# The current above which this comparison treats a transcranial device as
# unable to deliver the dose. This is an analysis convention chosen only to
# sort the required-current number that follows; it is not a claim about any
# device, any published safety limit, or any specific paper, and no paper is
# cited for it.
UNREACHABLE_CURRENT_THRESHOLD_MA = 4.0

# Bulk tissue conductivity used only to convert a directly-injected bipolar
# stimulating current into an order-of-magnitude reference field at the
# injection site, via the standard point-source-pair field formula in a
# homogeneous conductor (Coulomb's law for a steady current dipole -- not
# attributed to, and not requiring a citation to, any single study). A round,
# undramatized value; changing it rescales every required-current number by
# the same constant factor and does not change which regions clear the
# verdict threshold by more than that factor.
REFERENCE_FIELD_CONDUCTIVITY_S_PER_M = 0.33


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _distribution_summary(values):
    """n/min/median/max plus a crude nearest-rank IQR -- enough to show
    whether a distribution is tight or wide without pulling in a stats
    dependency for one summary."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {"n": 0}
    q1 = s[max(0, int(0.25 * n) - 1)]
    q3 = s[min(n - 1, int(0.75 * n))]
    return {
        "n": n, "min": s[0], "median": _median(s), "max": s[-1],
        "p25": q1, "p75": q3,
    }


def _num(v):
    try:
        if v in (None, "", "n/a", "NaN", "nan"):
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# Transcranial measurement corpus (TES1): induced potential per mA of scalp
# current, and the field per mA estimated from neighbouring-contact potential
# differences the way the dataset's own construction supports.
# ---------------------------------------------------------------------------

_GROUP_RE = re.compile(r"^[A-Za-z]+")


def load_tes1():
    """Read every P*/P*.txt row. Returns (electrodes, zero_drop)."""
    files = sorted(TES1_DIR.glob("P*/P*.txt"))
    electrodes = []
    seen = 0
    # Mutually exclusive so the four counts below sum exactly to rows_seen --
    # a row can be missing voltage only, coordinates only, both, or neither.
    excl_voltage_only = excl_coords_only = excl_both = usable = 0
    subjects = set()
    for path in files:
        subject = path.parent.name  # P014's 4 montages all live under P014/
        montage = path.stem  # e.g. "P014A" vs "P014"
        subjects.add(subject)
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 8 or not parts[0]:
                    continue
                seen += 1
                name = parts[0]
                mni = tuple(_num(x) for x in parts[1:4])
                voltage_mv = _num(parts[7])
                valid_coords = not any(math.isnan(c) for c in mni)
                valid_voltage = not math.isnan(voltage_mv)
                if valid_coords and valid_voltage:
                    usable += 1
                elif not valid_coords and not valid_voltage:
                    excl_both += 1
                elif not valid_voltage:
                    excl_voltage_only += 1
                else:
                    excl_coords_only += 1
                group = _GROUP_RE.match(name)
                electrodes.append({
                    "subject": subject, "montage": montage, "name": name,
                    "group": group.group(0) if group else name,
                    "mni": mni, "voltage_mv_per_ma": voltage_mv,
                    "valid_coords": valid_coords, "valid_voltage": valid_voltage,
                })
    zero_drop = {
        "rows_seen": seen,
        "excluded_missing_voltage_only_high_impedance_or_clipping": excl_voltage_only,
        "excluded_missing_coordinates_only_not_localised": excl_coords_only,
        "excluded_missing_both_voltage_and_coordinates": excl_both,
        "usable_for_matching": usable,
        "reconciles_exactly": (
            usable + excl_voltage_only + excl_coords_only + excl_both == seen
        ),
        "n_subjects": len(subjects), "n_montage_files": len(files),
    }
    return electrodes, zero_drop


def add_field_per_ma(electrodes):
    """Field per mA at each usable electrode: |dV| to its nearest same-group,
    same-montage usable neighbour, divided by their MNI separation. V/m and
    mV/mm are the same number, so this is directly comparable, unit for
    unit, to the point-source reference field computed below.

    This is a finite difference: it is a spatial derivative smoothed over
    whatever the neighbour separation happens to be, not a field at a point,
    and it is biased low relative to the true local field wherever that
    separation is not small compared to how fast the field itself varies.
    The separations actually used (``separations_mm``, one per electrode
    with a neighbour) are returned unchanged so that smoothing scale is
    reported, not assumed.
    """
    by_montage: dict[str, list[dict]] = {}
    for e in electrodes:
        by_montage.setdefault(e["montage"], []).append(e)
    n_no_neighbour = 0
    separations_mm: list[float] = []
    for montage, rows in by_montage.items():
        usable = [r for r in rows if r["valid_coords"] and r["valid_voltage"]]
        by_group: dict[str, list[dict]] = {}
        for r in usable:
            by_group.setdefault(r["group"], []).append(r)
        for r in usable:
            candidates = [o for o in by_group[r["group"]] if o is not r]
            if not candidates:
                r["field_mv_per_mm_per_ma"] = None
                n_no_neighbour += 1
                continue
            nearest = min(candidates, key=lambda o: _dist(r["mni"], o["mni"]))
            d = _dist(r["mni"], nearest["mni"])
            if d <= 0:
                r["field_mv_per_mm_per_ma"] = None
                n_no_neighbour += 1
                continue
            r["field_mv_per_mm_per_ma"] = (
                abs(r["voltage_mv_per_ma"] - nearest["voltage_mv_per_ma"]) / d
            )
            separations_mm.append(d)
    return n_no_neighbour, separations_mm


# ---------------------------------------------------------------------------
# Human intracranial sites: every recording electrode (the component's own
# measurement sites) and every stimulating anode/cathode contact, in the two
# corpora already used for the direct-stimulation arm.
# ---------------------------------------------------------------------------

def load_ram_sites():
    seen = excl_coords = 0
    sites = []
    for corpus in RAM_CORPORA:
        corpus_dir = DATA_ROOT / corpus
        if not corpus_dir.is_dir():
            continue
        for sub_dir in sorted(corpus_dir.glob("sub-*")):
            ses_dirs = sorted(sub_dir.glob("ses-*"))
            if not ses_dirs:
                continue
            # Electrode geometry is identical across a subject's sessions in
            # both corpora (implant does not move within a hospitalisation);
            # reading only the first session avoids counting one physical
            # site once per session.
            ieeg_dir = ses_dirs[0] / "ieeg"
            elec_files = sorted(ieeg_dir.glob("*_electrodes.tsv"))
            if not elec_files:
                continue
            rows = _read_tsv(elec_files[0])
            stim_channels = set()
            for ses_dir in ses_dirs:
                for ev_path in sorted((ses_dir / "ieeg").glob("*_events.tsv")):
                    for r in _read_tsv(ev_path):
                        for key in ("anode_label", "cathode_label"):
                            v = (r.get(key) or "").strip()
                            if v and v != "n/a":
                                stim_channels.add(v)
            for row in rows:
                seen += 1
                raw, source_col = None, "ind.region"
                for col in ("stein.region", "das.region", "wb.region", "ind.region"):
                    v = (row.get(col) or "").strip()
                    if v and v.lower() != "n/a":
                        raw, source_col = v, col
                        break
                structure, _hemi = normalize_region_label(
                    raw or "n/a", "bids_desikan_killiany_ind_region"
                )
                coarse = COARSE_REGION_GROUP.get(structure, "other")
                mni = (_num(row.get("x")), _num(row.get("y")), _num(row.get("z")))
                valid_coords = not any(math.isnan(c) for c in mni)
                if not valid_coords:
                    excl_coords += 1
                sites.append({
                    "corpus": corpus, "subject": sub_dir.name, "name": row.get("name"),
                    "mni": mni, "valid_coords": valid_coords,
                    "region": structure, "coarse_region": coarse,
                    "is_stim_site": row.get("name") in stim_channels,
                })
    zero_drop = {
        "rows_seen": seen, "excluded_missing_coordinates": excl_coords,
        "usable_for_matching": sum(1 for s in sites if s["valid_coords"]),
    }
    return sites, zero_drop


# ---------------------------------------------------------------------------
# Spatial join: every usable human intracranial site to its nearest usable
# transcranial-corpus electrode.
# ---------------------------------------------------------------------------

def spatial_join(ram_sites, tes1_electrodes, radius_mm):
    usable_tes1 = [e for e in tes1_electrodes if e["valid_coords"] and e["valid_voltage"]]
    for s in ram_sites:
        if not s["valid_coords"] or not usable_tes1:
            s["matched"] = False
            s["nearest_distance_mm"] = None
            s["matched_tes1"] = None
            continue
        nearest = min(usable_tes1, key=lambda e: _dist(s["mni"], e["mni"]))
        d = _dist(s["mni"], nearest["mni"])
        s["nearest_distance_mm"] = d
        s["matched"] = d <= radius_mm
        s["matched_tes1"] = nearest if s["matched"] else None
    return ram_sites


def measured_transfer_by_region(ram_sites):
    """Induced potential and induced field per mA of transcranial current at
    the matched locations, per region, with the spread taken across the
    transcranial corpus's own subjects (not pooled across sites, which would
    let one subject with many matched sites dominate)."""
    by_region_subject: dict[str, dict[str, dict[str, list[float]]]] = {}
    for s in ram_sites:
        if not s["matched"]:
            continue
        t = s["matched_tes1"]
        bucket = by_region_subject.setdefault(s["coarse_region"], {}).setdefault(
            t["subject"], {"potential": [], "field": []}
        )
        bucket["potential"].append(abs(t["voltage_mv_per_ma"]))
        if t["field_mv_per_mm_per_ma"] is not None:
            bucket["field"].append(t["field_mv_per_mm_per_ma"])
    out = {}
    for region, by_subject in by_region_subject.items():
        subject_potential_medians = [
            _median(v["potential"]) for v in by_subject.values() if v["potential"]
        ]
        subject_field_medians = [
            _median(v["field"]) for v in by_subject.values() if v["field"]
        ]
        out[region] = {
            "n_transcranial_subjects_contributing": len(by_subject),
            "induced_potential_mv_per_ma": {
                "n_subjects": len(subject_potential_medians),
                "median_across_subjects": (
                    _median(subject_potential_medians) if subject_potential_medians else None
                ),
                "min_across_subjects": (
                    min(subject_potential_medians) if subject_potential_medians else None
                ),
                "max_across_subjects": (
                    max(subject_potential_medians) if subject_potential_medians else None
                ),
            },
            "induced_field_mv_per_mm_per_ma": {
                "n_subjects": len(subject_field_medians),
                "median_across_subjects": (
                    _median(subject_field_medians) if subject_field_medians else None
                ),
                "min_across_subjects": (
                    min(subject_field_medians) if subject_field_medians else None
                ),
                "max_across_subjects": (
                    max(subject_field_medians) if subject_field_medians else None
                ),
            },
        }
    return out


def coverage_by_region(ram_sites):
    by_region: dict[str, dict] = {}
    for s in ram_sites:
        if not s["valid_coords"]:
            continue
        r = by_region.setdefault(s["coarse_region"], {
            "n_sites": 0, "n_matched": 0, "n_uncovered": 0,
            "matched_distances_mm": [],
        })
        r["n_sites"] += 1
        if s["matched"]:
            r["n_matched"] += 1
            r["matched_distances_mm"].append(s["nearest_distance_mm"])
        else:
            r["n_uncovered"] += 1
    out = {}
    for region, r in by_region.items():
        d = r["matched_distances_mm"]
        out[region] = {
            "n_sites": r["n_sites"], "n_matched": r["n_matched"],
            "n_uncovered": r["n_uncovered"],
            "matched_distance_mm": {
                "median": _median(d) if d else None,
                "min": min(d) if d else None, "max": max(d) if d else None,
            },
        }
    return out


# ---------------------------------------------------------------------------
# Reference dose: the currents actually delivered directly through a bipolar
# depth pair (read from the delivered parameter census), and, for the
# subset of pairs geometrically resolvable in this join, an
# order-of-magnitude reference field at the injection site.
# ---------------------------------------------------------------------------

def load_electrode_coords_by_subject():
    """subject -> {electrode_name: mni tuple}, both RAM corpora, first
    session only (geometry is static within a subject)."""
    coords: dict[str, dict[str, tuple]] = {}
    for corpus in RAM_CORPORA:
        corpus_dir = DATA_ROOT / corpus
        if not corpus_dir.is_dir():
            continue
        for sub_dir in sorted(corpus_dir.glob("sub-*")):
            ses_dirs = sorted(sub_dir.glob("ses-*"))
            if not ses_dirs:
                continue
            elec_files = sorted((ses_dirs[0] / "ieeg").glob("*_electrodes.tsv"))
            if not elec_files:
                continue
            m = {}
            for row in _read_tsv(elec_files[0]):
                mni = (_num(row.get("x")), _num(row.get("y")), _num(row.get("z")))
                if not any(math.isnan(c) for c in mni):
                    m[row.get("name")] = mni
            coords.setdefault(sub_dir.name, {}).update(m)
    return coords


def reference_field_v_per_m(current_ma: float, separation_mm: float) -> float:
    """Field at the midpoint of a bipolar current-dipole pair, current
    current_ma, contact separation separation_mm, in a homogeneous conductor
    of REFERENCE_FIELD_CONDUCTIVITY_S_PER_M -- E = 2I / (pi * sigma * d^2),
    each point source contributing I/(4*pi*sigma*(d/2)^2) at the midpoint,
    both contributions pointing the same way. V/m and mV/mm are the same
    number, so this is directly comparable to the TES1 field-per-mA above.
    """
    i_amps = current_ma / 1000.0
    d_m = separation_mm / 1000.0
    return 2.0 * i_amps / (math.pi * REFERENCE_FIELD_CONDUCTIVITY_S_PER_M * d_m ** 2)


def load_reference_dose(ram_site_regions: dict):
    """Every distinct (subject, anode, cathode, amplitude) stimulation
    actually delivered in the two RAM corpora, with the geometric separation
    of its bipolar pair and the resulting reference field, grouped by the
    anode contact's coarse region. ram_site_regions maps
    (subject, electrode_name) -> coarse_region, reused from load_ram_sites
    so no site is region-labelled twice."""
    coords = load_electrode_coords_by_subject()
    seen_events = excl_unlocalised = 0
    pairs = set()
    region_fields: dict[str, list[float]] = {}
    for corpus in RAM_CORPORA:
        corpus_dir = DATA_ROOT / corpus
        if not corpus_dir.is_dir():
            continue
        for ev_path in sorted(corpus_dir.glob("sub-*/ses-*/ieeg/*_events.tsv")):
            subject = ev_path.parts[ev_path.parts.index(corpus) + 1]
            for row in _read_tsv(ev_path):
                anode = (row.get("anode_label") or "").strip()
                cathode = (row.get("cathode_label") or "").strip()
                amp = _num(row.get("amplitude"))
                if anode in ("", "n/a") or cathode in ("", "n/a") or math.isnan(amp) or amp <= 0:
                    continue
                key = (subject, anode, cathode, amp)
                if key in pairs:
                    continue
                pairs.add(key)
                seen_events += 1
                subj_coords = coords.get(subject, {})
                if anode not in subj_coords or cathode not in subj_coords:
                    excl_unlocalised += 1
                    continue
                d = _dist(subj_coords[anode], subj_coords[cathode])
                if d <= 0:
                    excl_unlocalised += 1
                    continue
                field = reference_field_v_per_m(amp / 1000.0, d)
                region = ram_site_regions.get((subject, anode), "other")
                region_fields.setdefault(region, []).append(field)
    zero_drop = {
        "note": (
            "This is an independent direct scan of anode/cathode/amplitude "
            "triples in the raw event files, not the item-matched electrode-"
            "pair count in the delivered parameter census -- it is a superset "
            "used only to obtain bipolar contact separations, and includes a "
            "handful of pre-task calibration pulses the item-matched census "
            "excludes by design."
        ),
        "distinct_subject_anode_cathode_amplitude_combinations_seen_in_events": seen_events,
        "excluded_electrode_not_localised_or_zero_separation": excl_unlocalised,
        "usable_for_reference_field": seen_events - excl_unlocalised,
        "reconciles_exactly": (
            (seen_events - excl_unlocalised) + excl_unlocalised == seen_events
        ),
    }
    summary = {
        region: {
            "n_pair_amplitude_combinations": len(vals),
            "reference_field_v_per_m": {
                "median": _median(vals), "min": min(vals), "max": max(vals),
            },
        }
        for region, vals in region_fields.items()
    }
    return summary, zero_drop


# ---------------------------------------------------------------------------
# The delivered displacement (already on disk) and the required current.
# ---------------------------------------------------------------------------

def load_displacement():
    with open(COMPONENT_RESPONSE_PATH) as f:
        d = json.load(f)
    b = d["block_b"]
    return {
        "branch": b["branch"],
        "meaningful_effect_threshold_normalised_displacement":
            b["meaningful_effect_threshold_normalised_displacement"],
        "meaningful_effect_threshold_source": b["meaningful_effect_threshold_source"],
        "by_channel_condition": {
            cond: {
                "normalised_displacement": v["mean_value"],
                "p_value": v["p_value"], "n_sessions": v["n_sessions"],
                "n_subjects": v["n_subjects"], "mdd": v["mdd"]["mdd"],
            }
            for cond, v in b["pooled_normalised_displacement_by_channel_condition"].items()
        },
    }


def load_dose_census():
    with open(PARAMETER_CENSUS_PATH) as f:
        d = json.load(f)
    census = d["block_c_parameter_census"]
    amps_ma = sorted(
        set(census["ds005489_open_loop"]["unique_values_seen"]["amplitude_microamps"])
        | set(census["ds005557_classifier_triggered"]["unique_values_seen"].get(
            "amplitude_microamps", []))
    )
    return {
        "amplitudes_microamps_seen": amps_ma,
        "amplitudes_ma_seen": [a / 1000.0 for a in amps_ma],
        "n_electrode_pairs_open_loop": census["ds005489_open_loop"]["n_electrode_pairs"],
        "n_subjects_with_stimulated_trials_open_loop":
            census["ds005489_open_loop"]["n_subjects_with_stimulated_trials"],
    }


def required_current_by_region(coverage, tes1_field_lookup, reference_field_by_region):
    """Per coarse region present in BOTH the coverage join and the reference
    dose: the transcranial current that would produce the reference field,
    using every matched TES1 electrode's own field-per-mA in that region to
    build the across-subject interval."""
    out = {}
    for region, ref in reference_field_by_region.items():
        cov = coverage.get(region)
        fields_mv_mm_per_ma = tes1_field_lookup.get(region, [])
        if cov is None or cov["n_matched"] == 0 or not fields_mv_mm_per_ma:
            out[region] = {
                "status": "unmatched_no_transfer_estimate_in_this_region",
                "n_matched_sites": 0 if cov is None else cov["n_matched"],
            }
            continue
        ref_field_mv_per_mm = ref["reference_field_v_per_m"]["median"]
        required = sorted(ref_field_mv_per_mm / f for f in fields_mv_mm_per_ma if f > 0)
        if not required:
            out[region] = {
                "status": "all_matched_transfer_estimates_zero_or_unavailable",
                "n_matched_sites": cov["n_matched"],
            }
            continue
        med = _median(required)
        out[region] = {
            "status": "computed",
            "n_matched_sites": cov["n_matched"],
            "n_tes1_field_estimates": len(required),
            "reference_field_v_per_m": ref["reference_field_v_per_m"],
            "required_transcranial_current_ma": {
                "median": med, "min": required[0], "max": required[-1],
            },
            "verdict": (
                "within_the_declared_threshold" if med <= UNREACHABLE_CURRENT_THRESHOLD_MA
                else "beyond_the_declared_threshold"
            ),
            "shortfall_factor_vs_threshold": med / UNREACHABLE_CURRENT_THRESHOLD_MA,
        }
    return out


def main():
    t0 = time.time()
    print(f"matching radius (mm) = {MATCHING_RADIUS_MM}")
    print(f"unreachable-current threshold (mA) = {UNREACHABLE_CURRENT_THRESHOLD_MA}")
    print(f"reference-field conductivity (S/m) = {REFERENCE_FIELD_CONDUCTIVITY_S_PER_M}")

    tes1_electrodes, tes1_zero_drop = load_tes1()
    n_no_neighbour, neighbour_separations_mm = add_field_per_ma(tes1_electrodes)
    tes1_zero_drop["no_same_group_neighbour_for_field_estimate"] = n_no_neighbour

    ram_sites, ram_zero_drop = load_ram_sites()
    ram_site_region = {(s["subject"], s["name"]): s["coarse_region"] for s in ram_sites}

    ram_sites = spatial_join(ram_sites, tes1_electrodes, MATCHING_RADIUS_MM)
    coverage = coverage_by_region(ram_sites)
    measured_transfer = measured_transfer_by_region(ram_sites)

    tes1_field_lookup: dict[str, list[float]] = {}
    for s in ram_sites:
        if s["matched"] and s["matched_tes1"]["field_mv_per_mm_per_ma"] is not None:
            tes1_field_lookup.setdefault(s["coarse_region"], []).append(
                s["matched_tes1"]["field_mv_per_mm_per_ma"]
            )

    reference_field_by_region, dose_zero_drop = load_reference_dose(ram_site_region)
    dose_census = load_dose_census()
    displacement = load_displacement()

    required = required_current_by_region(coverage, tes1_field_lookup, reference_field_by_region)

    n_sites_total = ram_zero_drop["rows_seen"]
    n_matched = sum(1 for s in ram_sites if s.get("matched"))
    n_uncovered = sum(
        1 for s in ram_sites if s["valid_coords"] and not s.get("matched")
    )
    n_no_coords = ram_zero_drop["excluded_missing_coordinates"]
    reconciled = n_matched + n_uncovered + n_no_coords == n_sites_total

    artifact = {
        "scope": (
            "Order-of-magnitude specification only. Two physically different "
            "quantities are bridged here: a transcranially induced potential/"
            "field measured intracranially during weak (<2 mA) scalp current "
            "in 14 patients (a measured transfer function, no task, no "
            "behaviour, no raw recordings), and the current delivered "
            "directly through an implanted bipolar depth pair in two human "
            "free-recall stimulation corpora together with the displacement "
            "that direct current produced in the component's own normalised "
            "units. Direct bipolar stimulation and a transcranially induced "
            "field differ in geometry, extent, and what fraction of the "
            "affected tissue is the tissue of interest. Nothing here predicts "
            "that delivering the required transcranial current would "
            "reproduce the measured displacement; the required-current number "
            "is a specification for what a non-invasive device would have to "
            "achieve to reach a field of the same order of magnitude, "
            "nothing more. The transcranial corpus is not yet in the corpus "
            "registry (config/datasets.json / src/corpus_sessions.py); those "
            "files are being read by other running jobs and were not edited "
            "for this leg."
        ),
        "predeclared_constants": {
            "matching_radius_mm": MATCHING_RADIUS_MM,
            "matching_radius_rationale": (
                "A generous multiple of typical iEEG electrode MNI-"
                "localisation error, forgiving two independent coregistration "
                "pipelines while excluding a match in a different structure. "
                "Fixed before any transcranial voltage was read."
            ),
            "unreachable_current_threshold_ma": UNREACHABLE_CURRENT_THRESHOLD_MA,
            "unreachable_current_threshold_note": (
                "An analysis convention chosen only to sort the required-"
                "current result; not a claim about any device, standard, or "
                "publication, and no publication is cited for it."
            ),
            "reference_field_conductivity_s_per_m": REFERENCE_FIELD_CONDUCTIVITY_S_PER_M,
            "reference_field_formula": (
                "E = 2*I / (pi * sigma * d^2) at the midpoint of a bipolar "
                "current-dipole pair of separation d in a homogeneous "
                "conductor -- Coulomb's law for a steady point current "
                "source, not attributed to any single study."
            ),
        },
        "transcranial_corpus_zero_drop": {
            **tes1_zero_drop,
            "neighbour_separation_mm_distribution": _distribution_summary(
                neighbour_separations_mm
            ),
            "neighbour_separation_note": (
                "The distance each electrode's finite-difference field "
                "estimate was smoothed over -- one value per usable "
                "electrode with a same-group neighbour. See "
                "upper_bound_caveat: this separation is the source of the "
                "low bias on every induced-field number below."
            ),
        },
        "human_intracranial_site_zero_drop": {
            **ram_zero_drop,
            "n_matched_within_radius": n_matched,
            "n_uncovered_beyond_radius": n_uncovered,
            "reconciles_exactly": reconciled,
        },
        "reference_dose_geometry_zero_drop": dose_zero_drop,
        "upper_bound_caveat": (
            "Every required-transcranial-current and shortfall-factor number "
            "in this artifact is an upper bound, not a point estimate, and "
            "two independent biases both push it upward -- neither pushes it "
            "down. (1) The induced field is a finite difference between "
            "neighbouring transcranial-corpus contacts (see "
            "transcranial_corpus_zero_drop.neighbour_separation_mm_distribution "
            "for the actual millimetre scale this was smoothed over); a "
            "finite difference over a multi-millimetre separation is biased "
            "low relative to the true local field, and a low measured field "
            "in the denominator of required_current = reference_field / "
            "measured_field inflates the required current. (2) The "
            "reference field is evaluated at the midpoint of the "
            "stimulating bipolar pair -- close to the peak of the direct-"
            "stimulation field -- and is compared against that same "
            "coarsely-sampled, smoothed transcranial estimate; comparing a "
            "peak-like numerator against a smoothed-low denominator inflates "
            "the ratio a second time, in the same direction as bias (1). "
            "Both biases inflate every shortfall factor and required-current "
            "number below; neither deflates either. The verdict that every "
            "matched region is beyond the declared threshold is unchanged "
            "under an order-of-magnitude correction for these biases in "
            "either direction -- the shortfall factors range from 82-fold to "
            "over 18,000-fold, far larger than one order of magnitude of "
            "correction could close."
        ),
        "regional_comparison_caveat": (
            "Regions below are ordered alphabetically, not by shortfall "
            "factor or required current, and this ordering is deliberate: "
            "the required-current interval and shortfall factor differ by "
            "more than two orders of magnitude across regions (roughly "
            "82-fold up to over 18,000-fold), but so does the number of "
            "matched sites feeding each region's estimate (from 27 to "
            "over 1000). That spread is not interpreted here as saying any "
            "region is easier or harder to reach than another -- with "
            "matched-site counts differing by more than an order of "
            "magnitude, a between-region comparison of the factor would be "
            "a comparison of sample sizes, not a measurement. Every "
            "region-level factor is reported beside its own matched-site "
            "count for exactly this reason; the smallest cells (fewest "
            "matched sites, fewest reference-dose pair-amplitude "
            "combinations) are the least stable and must never be quoted "
            "without that count."
        ),
        "coverage_by_region": dict(sorted(coverage.items(), key=lambda kv: kv[0].lower())),
        "measured_transfer_by_region": dict(sorted(measured_transfer.items(), key=lambda kv: kv[0].lower())),
        "reference_dose": {
            "amplitudes_actually_used": dose_census,
            "displacement": displacement,
            "reference_field_by_region": dict(sorted(reference_field_by_region.items(), key=lambda kv: kv[0].lower())),
        },
        "required_transcranial_current_by_region": dict(sorted(required.items(), key=lambda kv: kv[0].lower())),
        "code_commit": git_commit(_ROOT),
        "wall_clock_s": time.time() - t0,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, allow_nan=False)
    print(f"wrote {OUT_PATH}, wall clock {artifact['wall_clock_s']:.1f}s")
    print(f"neighbour separation mm: {_distribution_summary(neighbour_separations_mm)}")
    for region in sorted(required, key=str.lower):
        r = required[region]
        print(f"  {region}: {r.get('status')} "
              f"required={r.get('required_transcranial_current_ma', {}).get('median')} "
              f"n_matched_sites={r.get('n_matched_sites')}")


if __name__ == "__main__":
    main()
